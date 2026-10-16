"""Versioned subscription streaming transport; terminal truncation is never accepted.

The supervisor stops its CLI process group as soon as an invalid stream event is
observed. It cannot guarantee that the CLI has not already started a second
server-side request. No automatic retry, API-key fallback, trimming or cap raise.
"""

from __future__ import annotations
import hashlib
import math
from dataclasses import dataclass
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import tempfile
import threading
import time

from . import claude_cli as v1
from .storage import digest

PROTOCOL = "claude-cli-subscription-stream-policy-v3"
MODEL = v1.MODEL
ClaudeTransportError = v1.ClaudeTransportError
MAX_STDOUT_BYTES = 4 * 1024**2
MAX_STDERR_BYTES = 1024**2
LIMITATION = "CLI stream supervision cannot guarantee that a second server-side request never started."


@dataclass(frozen=True)
class BillingGate:
    """Explicit human setting remains valid for this run; UI observations expire."""

    usage_credits_disabled: bool
    evidence_source: str
    observed_at_unix: float

    def check(self):
        observed = self.observed_at_unix
        if (
            self.usage_credits_disabled is not True
            or type(observed) not in (int, float)
            or not math.isfinite(observed)
            or observed < 0
            or observed > time.time()
            or self.evidence_source not in {"user_confirmed_usage_settings", "claude_usage_live"}
        ):
            raise v1.ClaudeTransportError("valid_usage_credits_disabled_evidence_required")
        if self.evidence_source == "claude_usage_live" and time.time() - observed > 3600:
            raise v1.ClaudeTransportError("fresh_live_usage_observation_required")


def billing_gate_from(path):
    """Read only private, non-symlink setting evidence; never reads credentials."""
    path = Path(path).absolute()
    if ".." in path.parts or any(p.is_symlink() for p in (path, *path.parents)):
        raise v1.ClaudeTransportError("unsafe_billing_evidence_path")
    if path.stat().st_mode & 0o077:
        raise v1.ClaudeTransportError("billing_evidence_not_private")
    try:
        value = json.loads(path.read_bytes())
        gate = BillingGate(
            **{k: value[k] for k in ("usage_credits_disabled", "evidence_source", "observed_at_unix")}
        )
    except (ValueError, KeyError, TypeError):
        raise v1.ClaudeTransportError("invalid_billing_evidence") from None
    gate.check()
    return gate


class ClaudeStreamInvalid(v1.ClaudeTransportError):
    """Terminal invalid generation; retained and never automatically retried."""


def command(binary):
    original = v1.command(binary)
    return [*original[:-1], "--include-partial-messages", original[-1]]


def validate_cap(max_output_tokens):
    if type(max_output_tokens) is not int or max_output_tokens not in {512, 2048}:
        raise v1.ClaudeTransportError("output_cap_policy")


def parse_response(stdout: str, returncode: int, *, max_output_tokens: int) -> tuple[str, dict]:
    validate_cap(max_output_tokens)
    if returncode:
        raise ClaudeTransportError("cli_nonzero_exit")
    try:
        events = [json.loads(line) for line in stdout.splitlines() if line.strip()]
    except (ValueError, TypeError) as exc:
        raise ClaudeTransportError("invalid_stream_json") from exc
    if not events or not all(isinstance(e, dict) for e in events):
        raise ClaudeTransportError("invalid_event_shape")
    init = [e for e in events if e.get("type") == "system" and e.get("subtype") == "init"]
    results = [e for e in events if e.get("type") == "result"]
    messages = [e.get("message") for e in events if e.get("type") == "assistant"]
    if len(init) != 1 or init[0].get("model") != MODEL:
        raise ClaudeTransportError("unexpected_init_model")
    if init[0].get("tools") or init[0].get("mcp_servers"):
        raise ClaudeTransportError("unexpected_tools")
    if (
        len(results) != 1
        or events[-1] != results[0]
        or results[0].get("is_error") is not False
        or results[0].get("subtype") != "success"
    ):
        raise ClaudeTransportError("unsuccessful_result")
    result = results[0]
    if result.get("num_turns") != 1:
        raise ClaudeTransportError("unexpected_turn_count")
    if not messages or any(not isinstance(m, dict) or m.get("model") != MODEL for m in messages):
        raise ClaudeTransportError("unexpected_response_model")
    message_ids = {m.get("id") for m in messages}
    if None in message_ids or len(message_ids) != 1:
        raise ClaudeTransportError("multiple_model_messages")
    stops = {m.get("stop_reason") for m in messages if m.get("stop_reason") is not None}
    # Current CLI assistant events precede the final message_delta. The closed
    # result carries that final stop reason; reject any conflicting earlier one.
    if result.get("stop_reason") is not None:
        stops.add(result["stop_reason"])
    if stops != {"end_turn"}:
        raise ClaudeTransportError("non_normal_or_unknown_stop")
    text_blocks = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list) or not all(isinstance(b, dict) for b in content):
            raise ClaudeTransportError("invalid_content_blocks")
        for block in content:
            if block.get("type") in {"tool_use", "server_tool_use"}:
                raise ClaudeTransportError("unexpected_tool_use")
            if block.get("type") == "text":
                if not isinstance(block.get("text"), str):
                    raise ClaudeTransportError("invalid_text_block")
                text_blocks.append(block.get("text", ""))
    answer = result.get("result")
    if not isinstance(answer, str) or not answer.strip() or answer != "".join(text_blocks):
        raise ClaudeTransportError("answer_stream_mismatch")
    model_usage = result.get("modelUsage")
    if not isinstance(model_usage, dict) or set(model_usage) != {MODEL}:
        raise ClaudeTransportError("unexpected_usage_model")
    if not isinstance(model_usage[MODEL], dict):
        raise ClaudeTransportError("invalid_model_usage")
    for key in ("inputTokens", "outputTokens", "cacheReadInputTokens", "cacheCreationInputTokens"):
        value = model_usage[MODEL].get(key)
        if type(value) is not int or value < 0:
            raise ClaudeTransportError("invalid_model_usage")
    usage = result.get("usage")
    if not isinstance(usage, dict):
        raise ClaudeTransportError("missing_usage")
    safe_usage = {}
    for key in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
        value = usage.get(key)
        if type(value) is not int or value < 0:
            raise ClaudeTransportError("invalid_usage")
        safe_usage[key] = value
    if not 0 < safe_usage["output_tokens"] <= max_output_tokens:
        raise ClaudeTransportError("output_token_cap")
    if not 0 < model_usage[MODEL]["outputTokens"] <= max_output_tokens:
        raise ClaudeTransportError("output_token_cap")
    cost = result.get("total_cost_usd")
    if type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0:
        raise ClaudeTransportError("invalid_cost_estimate")
    return answer, {
        **safe_usage,
        "model": MODEL,
        "finish_reason": "end_turn",
        "cached": False,
        "cli_estimated_cost_usd": result.get("total_cost_usd"),
        "cost_is_invoice": False,
        "thinking": "adaptive_low_not_disabled",
        "max_output_tokens_requested": max_output_tokens,
    }


class StreamGuard:
    """Pure event observer, independent of transport and response text."""

    def __init__(self, *, max_output_tokens):
        validate_cap(max_output_tokens)
        self.max_output_tokens = max_output_tokens
        self.message_ids = set()
        self.events = 0
        self.last_event_type = None
        self.stop_reason = None

    def message(self, message):
        if message.get("model") is not None and message["model"] != v1.MODEL:
            raise v1.ClaudeTransportError("unexpected_stream_model")
        identifier = message.get("id")
        if identifier:
            self.message_ids.add(identifier)
            if len(self.message_ids) > 1:
                raise ClaudeStreamInvalid("second_assistant_message")
        reason = message.get("stop_reason")
        if reason is not None:
            self.stop(reason)

    def stop(self, reason):
        self.stop_reason = (
            reason
            if reason in {"end_turn", "max_tokens", "stop_sequence", "refusal", "tool_use"}
            else "other"
        )
        if reason != "end_turn":
            raise ClaudeStreamInvalid("terminal_stop_" + self.stop_reason)

    def observe(self, event):
        if not isinstance(event, dict):
            raise v1.ClaudeTransportError("invalid_stream_event")
        self.events += 1
        self.last_event_type = event.get("type")
        if event.get("type") == "rate_limit_event":
            info = event.get("rate_limit_info")
            if not isinstance(info, dict):
                raise v1.ClaudeTransportError("invalid_rate_limit_info")
            if info.get("isUsingOverage") is True:
                raise v1.ClaudeTransportError("billing_overage_observed")
            if info.get("overageStatus") == "allowed":
                raise v1.ClaudeTransportError("usage_credits_policy_changed")
        if event.get("type") == "user" and event.get("isSynthetic") is True:
            raise ClaudeStreamInvalid("synthetic_user_recovery")
        if event.get("type") == "assistant":
            self.message(event.get("message", {}))
        if event.get("type") == "stream_event":
            frame = event.get("event", {})
            if not isinstance(frame, dict):
                raise v1.ClaudeTransportError("invalid_stream_frame")
            self.last_event_type = "stream_event:" + str(frame.get("type"))
            if frame.get("type") == "message_start":
                self.message(frame.get("message", {}))
            if frame.get("type") == "message_delta":
                reason = frame.get("delta", {}).get("stop_reason")
                if reason is not None:
                    self.stop(reason)
                output = frame.get("usage", {}).get("output_tokens")
                if output is not None:
                    if type(output) is not int or output < 0:
                        raise v1.ClaudeTransportError("invalid_stream_usage")
                    if output > self.max_output_tokens:
                        raise ClaudeStreamInvalid("terminal_output_token_cap")


def audit_stream(stdout, returncode, *, max_output_tokens):
    """Pure replay of event-stop policy plus the unchanged strict V1 final parser."""
    guard = StreamGuard(max_output_tokens=max_output_tokens)
    for line in stdout.splitlines():
        if line.strip():
            try:
                event = json.loads(line)
            except (ValueError, UnicodeError):
                raise v1.ClaudeTransportError("invalid_stream_json") from None
            guard.observe(event)
    return parse_response(stdout, returncode, max_output_tokens=max_output_tokens)


def cleanup(child):
    """Only the new-session process group owned by this child; no broad signals."""
    with v1.owned_cleanup_signals():
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait(timeout=5)


def stream(child, prompt, guard, *, timeout):
    """Drain both pipes and write stdin nonblocking in the owning main thread."""
    stdout = bytearray()
    stderr = bytearray()
    line = bytearray()
    started = time.monotonic()
    failure = None
    with selectors.DefaultSelector() as selector:
        pipes = [child.stdout, child.stderr, child.stdin]
        for pipe in pipes:
            os.set_blocking(pipe.fileno(), False)
        selector.register(child.stdout, selectors.EVENT_READ, "stdout")
        selector.register(child.stderr, selectors.EVENT_READ, "stderr")
        selector.register(child.stdin, selectors.EVENT_WRITE, "stdin")
        pending = memoryview(prompt.encode())
        sent = 0
        try:
            while selector.get_map():
                remaining = timeout - (time.monotonic() - started)
                if remaining <= 0:
                    raise v1.ClaudeTransportError("cli_stream_timeout")
                for key, _ in selector.select(min(0.1, remaining)):
                    pipe, kind = key.fileobj, key.data
                    if kind == "stdin":
                        try:
                            count = os.write(pipe.fileno(), pending[sent : sent + 8192])
                        except BrokenPipeError:
                            raise v1.ClaudeTransportError("cli_stdin_closed") from None
                        sent += count
                        if sent == len(pending):
                            selector.unregister(pipe)
                            pipe.close()
                        continue
                    raw = os.read(pipe.fileno(), 65536)
                    if not raw:
                        selector.unregister(pipe)
                        continue
                    target = stdout if kind == "stdout" else stderr
                    target.extend(raw)
                    if len(target) > (MAX_STDOUT_BYTES if kind == "stdout" else MAX_STDERR_BYTES):
                        raise v1.ClaudeTransportError(kind + "_size_bound")
                    if kind == "stdout":
                        line.extend(raw)
                        while b"\n" in line:
                            value, _, rest = line.partition(b"\n")
                            line[:] = rest
                            if value.strip():
                                try:
                                    event = json.loads(value)
                                except (ValueError, UnicodeError):
                                    raise v1.ClaudeTransportError("invalid_stream_json") from None
                                guard.observe(event)
            if line.strip():
                try:
                    guard.observe(json.loads(line))
                except (ValueError, UnicodeError):
                    raise v1.ClaudeTransportError("invalid_stream_json") from None
            child.wait(timeout=max(0.01, timeout - (time.monotonic() - started)))
        except BaseException as error:
            failure = error
        finally:
            if failure is not None:
                cleanup(child)
                # Preserve any already buffered bytes, without observing/accepting further output.
                for pipe, target, cap in [
                    (child.stdout, stdout, MAX_STDOUT_BYTES),
                    (child.stderr, stderr, MAX_STDERR_BYTES),
                ]:
                    while len(target) < cap:
                        try:
                            raw = os.read(pipe.fileno(), min(65536, cap - len(target)))
                        except (BlockingIOError, OSError):
                            break
                        if not raw:
                            break
                        target.extend(raw)
            for pipe in pipes:
                if not pipe.closed:
                    pipe.close()
    return bytes(stdout), bytes(stderr), failure, time.monotonic() - started


class ClaudeSubscriptionPolicyGenerator(v1.ClaudeSubscriptionGenerator):
    def __init__(self, *, billing_gate, max_output_tokens, binary="claude", timeout=180):
        validate_cap(max_output_tokens)
        self.max_output_tokens = max_output_tokens
        super().__init__(billing_gate=billing_gate, binary=binary, timeout=timeout)
        self.env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(max_output_tokens)
        self.billing_gate = BillingGate(
            billing_gate.usage_credits_disabled, billing_gate.evidence_source, billing_gate.observed_at_unix
        )
        self.identity = digest(
            {
                "adapter": PROTOCOL,
                "model": v1.MODEL,
                "cli_version": v1.CLI_VERSION,
                "system": v1.GENERATOR_SYSTEM,
                "max_output_tokens_requested": max_output_tokens,
                "thinking": "adaptive_low_not_disabled",
                "temperature": "cli_not_exposed",
                "seed": "cli_not_exposed",
                "tools": [],
                "fresh_session_per_request": True,
                "billing_gate_policy": "user-confirmed-setting-persistent-for-run;live-observation-max-3600s;future-invalid",
                "rate_limit_billing_policy": "fatal-if-isUsingOverage-true-or-overageStatus-allowed",
                "partial_messages": True,
                "abort_on_non_end_turn": True,
                "abort_on_synthetic_user": True,
                "abort_on_second_message": True,
                "server_second_request_prevention_guaranteed": False,
            }
        )

    def generate(self, question, contexts, receipt_dir):
        if threading.current_thread() is not threading.main_thread():
            raise v1.ClaudeTransportError("stream_owner_main_thread_required")
        self.preflight()
        prompt = v1.prompt_for(question, contexts)
        request_id = digest({"provider": self.identity, "prompt": prompt})
        receipt_dir = Path(receipt_dir).absolute()
        if ".." in receipt_dir.parts or any(p.is_symlink() for p in (receipt_dir, *receipt_dir.parents)):
            raise v1.ClaudeTransportError("unsafe_receipt_path")
        receipt_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        if receipt_dir.stat().st_mode & 0o077:
            raise v1.ClaudeTransportError("receipt_directory_not_private")
        request = receipt_dir / request_id
        try:
            request.mkdir(mode=0o700)
        except FileExistsError:
            raise v1.ClaudeTransportError("request_already_claimed") from None

        def save(name, value):
            raw = (
                value
                if isinstance(value, bytes)
                else json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
            )
            fd = os.open(request / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            fd = os.open(request, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)

        save(
            "claim.json",
            {
                "protocol": PROTOCOL,
                "max_output_tokens_requested": self.max_output_tokens,
                "request_id": request_id,
                "provider_identity": self.identity,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "started_at_unix": time.time(),
                "pid": os.getpid(),
                "server_second_request_prevention_guaranteed": False,
                "limitation": LIMITATION,
            },
        )
        guard = StreamGuard(max_output_tokens=self.max_output_tokens)
        child = None
        stdout = stderr = b""
        failure = None
        elapsed = 0.0
        previous = signal.getsignal(signal.SIGTERM)

        def interrupted(_sig, _frame):
            raise KeyboardInterrupt()

        signal.signal(signal.SIGTERM, interrupted)
        try:
            with tempfile.TemporaryDirectory(prefix="claude-rag-stream-empty-") as cwd:
                try:
                    child = subprocess.Popen(
                        command(self.binary),
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        env=self.env,
                        cwd=cwd,
                        start_new_session=True,
                    )
                    stdout, stderr, failure, elapsed = stream(child, prompt, guard, timeout=self.timeout)
                except BaseException as error:
                    failure = error
                    if child is not None:
                        cleanup(child)
        finally:
            signal.signal(signal.SIGTERM, previous)
        save("stdout.private.jsonl", stdout)
        save("stderr.private.log", stderr)
        if failure is not None:
            terminal = isinstance(failure, ClaudeStreamInvalid)
            code = (
                str(failure)
                if isinstance(failure, v1.ClaudeTransportError)
                else "cli_interrupted_or_transport_failure"
            )
            save(
                "stream-stop.json",
                {
                    "code": code,
                    "outcome": "terminal_invalid" if terminal else "fatal",
                    "event_type": guard.last_event_type,
                    "stop_reason": guard.stop_reason,
                    "events_observed": guard.events,
                    "elapsed_seconds": elapsed,
                    "request_id": request_id,
                    "provider_identity": self.identity,
                    "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
                    "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
                    "server_second_request_prevention_guaranteed": False,
                },
            )
            save(
                "failure.json",
                {
                    "code": code,
                    "outcome": "terminal_invalid" if terminal else "fatal",
                    "automatic_retry": False,
                },
            )
            if terminal:
                raise failure
            raise v1.ClaudeTransportError(code) from None
        try:
            answer, usage = audit_stream(
                stdout.decode("utf8"), child.returncode, max_output_tokens=self.max_output_tokens
            )
        except (v1.ClaudeTransportError, UnicodeError) as error:
            code = str(error) if isinstance(error, v1.ClaudeTransportError) else "invalid_stream_encoding"
            save("failure.json", {"code": code, "outcome": "fatal", "automatic_retry": False})
            raise v1.ClaudeTransportError(code) from None
        self.last_usage = {
            **usage,
            "seconds": elapsed,
            "request_id": request_id,
            "provider_identity": self.identity,
            "transport": PROTOCOL,
            "max_output_tokens_requested": self.max_output_tokens,
            "server_second_request_prevention_guaranteed": False,
        }
        save(
            "completed.json",
            {"answer": answer, "usage": self.last_usage, "stdout_sha256": hashlib.sha256(stdout).hexdigest()},
        )
        return answer

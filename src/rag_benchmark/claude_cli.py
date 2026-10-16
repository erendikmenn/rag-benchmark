"""Claude Code subscription transport; explicit billing gate, no API-key fallback.

This is a separate protocol from HTTP/API and historical local generation. The
caller must establish that account usage credits are off before enabling calls.
Auth status proves subscription authentication, not billing settings or quota.
"""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import threading
import time

from .models import GENERATOR_SYSTEM
from .storage import digest

MODEL = "claude-haiku-5-5"
CLI_VERSION = "2.1.291 (Claude Code)"


class ClaudeTransportError(RuntimeError):
    """Only a safe code, never provider output or private source text."""


@contextmanager
def owned_cleanup_signals():
    """Finish reaping owned children even if cancellation is requested twice."""
    handlers = {}
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGTERM, signal.SIGINT):
            handlers[sig] = signal.signal(sig, signal.SIG_IGN)
    try:
        yield
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def subscription_environment(source: dict[str, str]) -> dict[str, str]:
    # Do not mutate login files, shell configuration, or the parent's credentials.
    env = {k: v for k, v in source.items()
           if not k.startswith(("ANTHROPIC_", "CLAUDE_"))
           and k not in {"MAX_THINKING_TOKENS", "MAX_STRUCTURED_OUTPUT_RETRIES",
                         "DISABLE_EXTRA_USAGE_COMMAND"}}
    env.update({
        "CLAUDE_CODE_SAFE_MODE": "1",
        "CLAUDE_CODE_DISABLE_TERMINAL_TITLE": "1",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "512",
        "CLAUDE_CODE_MAX_RETRIES": "0",
        "CLAUDE_CODE_SKIP_PROMPT_HISTORY": "1",
        "CLAUDE_CODE_EFFORT_LEVEL": "low",
        # This hides the purchase command. It is NOT a billing guarantee.
        "DISABLE_EXTRA_USAGE_COMMAND": "1",
    })
    return env


def require_subscription(status: dict) -> dict:
    if (status.get("loggedIn") is not True or status.get("authMethod") != "claude.ai"
            or status.get("apiProvider") != "firstParty"
            or status.get("subscriptionType") not in {"max", "pro"}):
        raise ClaudeTransportError("subscription_auth_required")
    return {key: status[key] for key in ("loggedIn", "authMethod", "apiProvider", "subscriptionType")}


def prompt_for(question: str, contexts: list[dict]) -> str:
    if not isinstance(question, str) or not question.strip() or not 0 <= len(contexts) <= 5:
        raise ClaudeTransportError("question_or_context_count")
    evidence = []
    seen = set()
    for row in contexts:
        if not isinstance(row.get("id"), str) or not row["id"] or row["id"] in seen:
            raise ClaudeTransportError("context_identity")
        if not isinstance(row.get("text"), str) or not row["text"].strip():
            raise ClaudeTransportError("context_text")
        if not isinstance(row.get("title", ""), str):
            raise ClaudeTransportError("context_title")
        seen.add(row["id"])
        # Intentionally select fields: never serialize answers, labels, or scores.
        evidence.append({"source_id": row["id"], "title": row.get("title", ""), "text": row["text"]})
    return "Soru: " + question + "\n\nKaynak pasajlar (JSON):\n" + json.dumps(evidence, ensure_ascii=False)


def command(binary: str) -> list[str]:
    return [binary, "--safe-mode", "--restricted", "--setting-sources", "",
            "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
            "--no-chrome", "--disable-slash-commands", "--tools", "",
            "--permission-mode", "dontAsk", "--model", MODEL, "--effort", "low",
            "--system-prompt", GENERATOR_SYSTEM, "--no-session-persistence",
            "--max-turns", "1", "--output-format", "stream-json", "--verbose", "-p"]


def parse_response(stdout: str, returncode: int) -> tuple[str, dict]:
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
    if (len(results) != 1 or events[-1] != results[0] or results[0].get("is_error") is not False
            or results[0].get("subtype") != "success"):
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
    if not 0 < safe_usage["output_tokens"] <= 512:
        raise ClaudeTransportError("output_token_cap")
    cost = result.get("total_cost_usd")
    if type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0:
        raise ClaudeTransportError("invalid_cost_estimate")
    return answer, {**safe_usage, "model": MODEL, "finish_reason": "end_turn",
                    "cached": False, "cli_estimated_cost_usd": result.get("total_cost_usd"),
                    "cost_is_invoice": False, "thinking": "adaptive_low_not_disabled"}


@dataclass(frozen=True)
class BillingGate:
    usage_credits_disabled: bool
    evidence_source: str
    observed_at_unix: float

    def check(self) -> None:
        age = time.time() - self.observed_at_unix
        if (self.usage_credits_disabled is not True
                or self.evidence_source not in {"claude_usage_live", "user_confirmed_usage_settings"}
                or not 0 <= age <= 3600):
            raise ClaudeTransportError("fresh_usage_credits_disabled_evidence_required")


class ClaudeSubscriptionGenerator:
    def __init__(self, *, billing_gate: BillingGate, binary: str = "claude", timeout: float = 180):
        self.billing_gate, self.binary, self.timeout = billing_gate, binary, timeout
        self.env = subscription_environment(dict(os.environ))
        self.last_usage: dict = {}
        self.identity = digest({"adapter": "claude-cli-subscription-v1", "model": MODEL,
            "cli_version": CLI_VERSION, "system": GENERATOR_SYSTEM, "max_output_tokens_requested": 512,
            "thinking": "adaptive_low_not_disabled", "temperature": "cli_not_exposed",
            "seed": "cli_not_exposed", "tools": [], "fresh_session_per_request": True})

    def preflight(self) -> dict:
        self.billing_gate.check()
        try:
            version = subprocess.run([self.binary, "--version"], env=self.env, capture_output=True,
                                     text=True, check=True, timeout=15).stdout.strip()
            status = subprocess.run([self.binary, "--safe-mode", "--restricted", "--setting-sources", "",
                                     "auth", "status"], env=self.env, capture_output=True,
                                    text=True, check=True, timeout=15)
            auth = require_subscription(json.loads(status.stdout))
        except (subprocess.SubprocessError, OSError, ValueError) as exc:
            raise ClaudeTransportError("auth_preflight_failed") from exc
        if version != CLI_VERSION:
            raise ClaudeTransportError("cli_version_changed")
        return {"version": version, "auth": auth, "model_requested": MODEL,
                "model_execution_verified": False, "billing_evidence": self.billing_gate.evidence_source}

    def generate(self, question: str, contexts: list[dict], receipt_dir: Path) -> str:
        self.preflight()
        prompt = prompt_for(question, contexts)
        request_id = digest({"provider": self.identity, "prompt": prompt})
        receipt_dir = Path(receipt_dir).absolute()
        if ".." in receipt_dir.parts or any(p.is_symlink() for p in (receipt_dir, *receipt_dir.parents)):
            raise ClaudeTransportError("unsafe_receipt_path")
        receipt_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        if receipt_dir.stat().st_mode & 0o077:
            raise ClaudeTransportError("receipt_directory_not_private")
        request_dir = receipt_dir / request_id
        try:
            request_dir.mkdir(mode=0o700)
        except FileExistsError as exc:
            # Never turn an ambiguous prior submission into an automatic retry.
            raise ClaudeTransportError("request_already_claimed") from exc
        def save(name: str, value: str) -> None:
            fd = os.open(request_dir / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(value)
                handle.flush()
                os.fsync(handle.fileno())
            fd = os.open(request_dir, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        save("claim.json", json.dumps({"request_id": request_id, "provider_identity": self.identity,
                                      "started_at_unix": time.time(), "pid": os.getpid()}))
        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="claude-rag-empty-") as cwd:
            previous_handler = None
            install_handler = threading.current_thread() is threading.main_thread()
            def interrupted(_signum, _frame):
                raise KeyboardInterrupt
            if install_handler:
                previous_handler = signal.signal(signal.SIGTERM, interrupted)
            try:
                child = subprocess.Popen(command(self.binary), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE, text=True, env=self.env, cwd=cwd,
                                         start_new_session=True)
            except OSError:
                if install_handler:
                    signal.signal(signal.SIGTERM, previous_handler)
                save("failure.json", json.dumps({"code": "cli_spawn_failed", "automatic_retry": False}))
                raise ClaudeTransportError("cli_spawn_failed") from None
            try:
                stdout, stderr = child.communicate(prompt, timeout=self.timeout)
            except BaseException:
                with owned_cleanup_signals():
                    try:
                        os.killpg(child.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        stdout, stderr = child.communicate(timeout=5)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(child.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        stdout, stderr = child.communicate(timeout=5)
                save("stdout.private.jsonl", stdout)
                save("stderr.private.log", stderr)
                raise ClaudeTransportError("cli_interrupted_or_timeout") from None
            finally:
                if install_handler:
                    signal.signal(signal.SIGTERM, previous_handler)
        save("stdout.private.jsonl", stdout)
        save("stderr.private.log", stderr)
        try:
            answer, usage = parse_response(stdout, child.returncode)
        except ClaudeTransportError as exc:
            save("failure.json", json.dumps({"code": str(exc), "returncode": child.returncode,
                                             "automatic_retry": False}))
            raise
        self.last_usage = {**usage, "seconds": time.monotonic() - started,
                           "request_id": request_id, "provider_identity": self.identity}
        save("completed.json", json.dumps({"answer": answer, "usage": self.last_usage,
            "stdout_sha256": hashlib.sha256(stdout.encode()).hexdigest()}, ensure_ascii=False))
        return answer

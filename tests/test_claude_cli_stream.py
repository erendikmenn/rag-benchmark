"""Real pipes and fake children only; no subprocess/CLI/API/model calls."""

import copy
import importlib.util
import json
import os
from pathlib import Path
import signal
import threading
import subprocess
import time
import pytest
from rag_benchmark import claude_cli_stream as m

s = importlib.util.spec_from_file_location(
    "stream_base_fixture", Path(__file__).with_name("test_claude_cli.py")
)
base = importlib.util.module_from_spec(s)
s.loader.exec_module(base)


def lines(events):
    return ("\n".join(json.dumps(e) for e in events) + "\n").encode()


class PipeChild:
    pid = 987654

    def __init__(self, raw, stderr=b"", hang=False):
        stdin_read, stdin_write = os.pipe()
        stdout_read, stdout_write = os.pipe()
        stderr_read, stderr_write = os.pipe()
        self.stdin = os.fdopen(stdin_write, "wb", buffering=0)
        self.stdout = os.fdopen(stdout_read, "rb", buffering=0)
        self.stderr = os.fdopen(stderr_read, "rb", buffering=0)
        self.returncode = None
        self.stop = threading.Event()
        self.prompt = b""

        def writer():
            try:
                with os.fdopen(stdin_read, "rb", buffering=0) as source:
                    while chunk := source.read(8192):
                        self.prompt += chunk
                if hang:
                    self.stop.wait(2)
                else:
                    for fd, data in [(stderr_write, stderr), (stdout_write, raw)]:
                        for start in range(0, len(data), 8192):
                            os.write(fd, data[start : start + 8192])
            except (BrokenPipeError, OSError):
                pass
            finally:
                for fd in [stdout_write, stderr_write]:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
                self.returncode = -15 if self.stop.is_set() else 0

        self.thread = threading.Thread(target=writer)
        self.thread.start()

    def wait(self, timeout):
        self.thread.join(timeout)
        if self.thread.is_alive():
            raise subprocess.TimeoutExpired("fake", timeout)
        return self.returncode

    def poll(self):
        return self.returncode


def generator(monkeypatch, raw, *, stderr=b"", timeout=1, hang=False):
    gate = m.v1.BillingGate(True, "user_confirmed_usage_settings", time.time())
    g = m.ClaudeSubscriptionStreamGenerator(billing_gate=gate, timeout=timeout)
    monkeypatch.setattr(g, "preflight", lambda: {})
    children = []
    kills = []

    def launch(argv, **kwargs):
        assert "--include-partial-messages" in argv and kwargs["start_new_session"] is True
        assert not any(k.startswith("ANTHROPIC_") for k in kwargs["env"])
        child = PipeChild(raw, stderr=stderr, hang=hang)
        children.append(child)
        return child

    def killpg(pid, sig):
        assert pid == PipeChild.pid
        kills.append((pid, sig))
        children[-1].stop.set()

    monkeypatch.setattr(m.subprocess, "Popen", launch)
    monkeypatch.setattr(m.os, "killpg", killpg)
    return g, children, kills


def test_observed_v1_normal_events_still_valid_and_same_prompt(monkeypatch, tmp_path):
    raw = lines(base.events())
    g, children, kills = generator(monkeypatch, raw, stderr=b"diagnostic" * 20000)
    result = g.generate(
        "Synthetic question", [{"id": "s", "title": "T", "text": "Synthetic source"}], tmp_path
    )
    assert result == "Synthetic answer" and not kills
    assert children[0].prompt.decode() == m.v1.prompt_for(
        "Synthetic question", [{"id": "s", "title": "T", "text": "Synthetic source"}]
    )
    completed = json.loads(next(tmp_path.rglob("completed.json")).read_text())
    assert (
        completed["usage"]["transport"] == m.PROTOCOL
        and completed["usage"]["server_second_request_prevention_guaranteed"] is False
    )
    assert next(tmp_path.rglob("stderr.private.log")).stat().st_size == len(b"diagnostic" * 20000)
    assert m.audit_stream(raw.decode(), 0)[0] == result


@pytest.mark.parametrize("reason", ["max_tokens", "stop_sequence", "refusal", "tool_use"])
def test_explicit_first_non_normal_delta_stops_owned_group(monkeypatch, tmp_path, reason):
    events = base.events()[:1] + [
        {
            "type": "stream_event",
            "event": {
                "type": "message_delta",
                "delta": {"stop_reason": reason},
                "usage": {"output_tokens": 512},
            },
        }
    ]
    g, children, kills = generator(monkeypatch, lines(events))
    with pytest.raises(m.ClaudeStreamInvalid, match="terminal_stop_"):
        g.generate("Synthetic question", [], tmp_path)
    assert kills == [(PipeChild.pid, signal.SIGTERM)] and children[0].poll() is not None
    assert not list(tmp_path.rglob("completed.json"))
    failure = json.loads(next(tmp_path.rglob("failure.json")).read_text())
    assert failure["outcome"] == "terminal_invalid" and failure["automatic_retry"] is False
    assert next(tmp_path.rglob("stdout.private.jsonl")).read_bytes() == lines(events)


@pytest.mark.parametrize("kind", ["synthetic", "second_message", "over_cap"])
def test_auto_continuation_and_total_cap_rejected(monkeypatch, tmp_path, kind):
    events = base.events()[:-1]
    if kind == "synthetic":
        events.append({"type": "user", "isSynthetic": True, "message": {"role": "user", "content": []}})
    elif kind == "second_message":
        new = copy.deepcopy(events[-1])
        new["message"]["id"] = "different-model-message"
        events.append(new)
    else:
        events.append(
            {
                "type": "stream_event",
                "event": {"type": "message_delta", "delta": {}, "usage": {"output_tokens": 513}},
            }
        )
    g, _, kills = generator(monkeypatch, lines(events))
    with pytest.raises(m.ClaudeStreamInvalid):
        g.generate("Synthetic question", [], tmp_path)
    assert len(kills) == 1 and not list(tmp_path.rglob("completed.json"))


def test_timeout_is_bounded_fatal_and_never_retry(monkeypatch, tmp_path):
    g, children, kills = generator(monkeypatch, b"", timeout=0.02, hang=True)
    started = time.monotonic()
    with pytest.raises(m.v1.ClaudeTransportError, match="cli_stream_timeout"):
        g.generate("Synthetic question", [], tmp_path)
    assert time.monotonic() - started < 1 and len(children) == 1 and len(kills) == 1
    assert json.loads(next(tmp_path.rglob("failure.json")).read_text())["outcome"] == "fatal"


def test_repeated_real_signals_during_owned_cleanup_restore_handlers(monkeypatch, tmp_path):
    events = [{"type": "user", "isSynthetic": True}]
    g, children, kills = generator(monkeypatch, lines(events))
    previous = {s: signal.getsignal(s) for s in [signal.SIGTERM, signal.SIGINT]}

    def repeat(pid, sig):
        kills.append((pid, sig))
        children[-1].stop.set()
        for repeated in previous:
            assert signal.getsignal(repeated) == signal.SIG_IGN
            os.kill(os.getpid(), repeated)

    monkeypatch.setattr(m.os, "killpg", repeat)
    with pytest.raises(m.ClaudeStreamInvalid):
        g.generate("Synthetic question", [], tmp_path)
    assert {s: signal.getsignal(s) for s in previous} == previous and len(kills) == 1


def test_v1_final_guards_and_new_identity(monkeypatch, tmp_path):
    events = base.events()
    events[-1]["num_turns"] = 2
    g, _, _ = generator(monkeypatch, lines(events))
    with pytest.raises(m.v1.ClaudeTransportError, match="unexpected_turn_count"):
        g.generate("Synthetic question", [], tmp_path)
    old = m.v1.ClaudeSubscriptionGenerator(billing_gate=g.billing_gate)
    assert old.identity != g.identity and not list(tmp_path.rglob("completed.json"))


def test_duplicate_claim_never_spawns_again(monkeypatch, tmp_path):
    g, children, _ = generator(monkeypatch, lines(base.events()))
    g.generate("Synthetic question", [], tmp_path)
    with pytest.raises(m.v1.ClaudeTransportError, match="request_already_claimed"):
        g.generate("Synthetic question", [], tmp_path)
    assert len(children) == 1


def test_main_thread_required_before_spawn(monkeypatch, tmp_path):
    g, children, _ = generator(monkeypatch, lines(base.events()))
    errors = []

    def thread():
        try:
            g.generate("Synthetic question", [], tmp_path)
        except m.v1.ClaudeTransportError as error:
            errors.append(str(error))

    t = threading.Thread(target=thread)
    t.start()
    t.join()
    assert errors == ["stream_owner_main_thread_required"] and not children


@pytest.mark.parametrize(
    "info,code",
    [
        ({"isUsingOverage": True, "overageStatus": "allowed"}, "billing_overage_observed"),
        ({"isUsingOverage": True, "overageStatus": "not_allowed"}, "billing_overage_observed"),
        ({"isUsingOverage": False, "overageStatus": "allowed"}, "usage_credits_policy_changed"),
    ],
)
def test_rate_limit_billing_change_is_fatal_and_stops_owned_child(monkeypatch, tmp_path, info, code):
    events = base.events()[:1] + [{"type": "rate_limit_event", "rate_limit_info": info}]
    g, children, kills = generator(monkeypatch, lines(events))
    with pytest.raises(m.v1.ClaudeTransportError, match=code):
        g.generate("Synthetic question", [], tmp_path)
    failure = json.loads(next(tmp_path.rglob("failure.json")).read_text())
    assert failure["outcome"] == "fatal" and failure["code"] == code and failure["automatic_retry"] is False
    assert len(children) == len(kills) == 1 and not list(tmp_path.rglob("completed.json"))


def test_rate_limit_no_overage_accepts_normal_response(monkeypatch, tmp_path):
    events = base.events()
    events.insert(
        1,
        {
            "type": "rate_limit_event",
            "rate_limit_info": {"isUsingOverage": False, "overageStatus": "not_allowed"},
        },
    )
    g, _, kills = generator(monkeypatch, lines(events))
    assert g.generate("Synthetic question", [], tmp_path) == "Synthetic answer" and not kills


def test_explicit_user_confirmation_remains_valid_and_preserves_timestamp(monkeypatch):
    now = time.time()
    observed = now - 7200
    gate = m.BillingGate(True, "user_confirmed_usage_settings", observed)
    gate.check()
    inherited = m.v1.BillingGate(True, "user_confirmed_usage_settings", observed)
    g = m.ClaudeSubscriptionStreamGenerator(billing_gate=inherited)
    assert isinstance(g.billing_gate, m.BillingGate) and g.billing_gate.observed_at_unix == observed
    g.billing_gate.check()
    with pytest.raises(m.v1.ClaudeTransportError):
        inherited.check()


@pytest.mark.parametrize("observed", [float("nan"), float("inf"), -1, "yesterday", True, None])
def test_invalid_confirmation_timestamp_rejected(observed):
    with pytest.raises(m.v1.ClaudeTransportError):
        m.BillingGate(True, "user_confirmed_usage_settings", observed).check()


def test_future_unknown_source_and_stale_live_observation_rejected():
    for gate in [
        m.BillingGate(True, "user_confirmed_usage_settings", time.time() + 60),
        m.BillingGate(True, "unknown", time.time()),
        m.BillingGate(False, "user_confirmed_usage_settings", time.time()),
        m.BillingGate(True, "claude_usage_live", time.time() - 3601),
    ]:
        with pytest.raises(m.v1.ClaudeTransportError):
            gate.check()
    m.BillingGate(True, "claude_usage_live", time.time() - 10).check()


def test_persistent_gate_still_performs_fresh_auth_each_preflight(monkeypatch):
    calls = []
    g = m.ClaudeSubscriptionStreamGenerator(
        billing_gate=m.BillingGate(True, "user_confirmed_usage_settings", time.time() - 7200)
    )

    def run(argv, **kwargs):
        calls.append(argv)
        from types import SimpleNamespace

        if "--version" in argv:
            return SimpleNamespace(stdout=m.v1.CLI_VERSION)
        return SimpleNamespace(
            stdout=json.dumps(
                {
                    "loggedIn": True,
                    "authMethod": "claude.ai",
                    "apiProvider": "firstParty",
                    "subscriptionType": "max",
                }
            )
        )

    monkeypatch.setattr(m.v1.subprocess, "run", run)
    g.preflight()
    g.preflight()
    assert len(calls) == 4 and sum("auth" in command for command in calls) == 2


def test_private_billing_loader_uses_v2_gate_without_expiring_user_confirmation(tmp_path):
    p = tmp_path / "billing.json"
    observed = time.time() - 7200
    p.write_text(
        json.dumps(
            {
                "usage_credits_disabled": True,
                "evidence_source": "user_confirmed_usage_settings",
                "observed_at_unix": observed,
            }
        )
    )
    p.chmod(0o600)
    assert m.billing_gate_from(p).observed_at_unix == observed
    p.chmod(0o644)
    with pytest.raises(m.v1.ClaudeTransportError, match="not_private"):
        m.billing_gate_from(p)
    p.chmod(0o600)
    link = tmp_path / "symlink.json"
    link.symlink_to(p)
    with pytest.raises(m.v1.ClaudeTransportError, match="unsafe_billing"):
        m.billing_gate_from(link)

"""Subscription transport contract tests; all process/model calls are fake."""

import copy
import json
import signal
import subprocess
import time
from types import SimpleNamespace

import pytest

from rag_benchmark import claude_cli as m


def events():
    usage = {
        "input_tokens": 10,
        "output_tokens": 4,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    }
    return [
        {"type": "system", "subtype": "init", "model": m.MODEL, "tools": [], "mcp_servers": []},
        {
            "type": "assistant",
            "message": {
                "id": "synthetic-message",
                "model": m.MODEL,
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "Synthetic answer"}],
            },
        },
        {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "num_turns": 1,
            "result": "Synthetic answer",
            "usage": usage,
            "modelUsage": {
                m.MODEL: {
                    "inputTokens": 10,
                    "outputTokens": 4,
                    "cacheReadInputTokens": 0,
                    "cacheCreationInputTokens": 0,
                }
            },
            "total_cost_usd": 0.000003,
        },
    ]


def parse(rows):
    return m.parse_response("\n".join(json.dumps(e) for e in rows), 0)


def test_valid_single_model_normal_stop():
    answer, usage = parse(events())
    assert answer == "Synthetic answer"
    assert usage["model"] == m.MODEL
    assert usage["cost_is_invoice"] is False
    assert usage["thinking"] == "adaptive_low_not_disabled"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda e: e[0].update(model="other-model"),
        lambda e: e[0].update(tools=["Bash"]),
        lambda e: e[1]["message"].update(model="other-model"),
        lambda e: e[1]["message"].update(stop_reason="max_tokens"),
        lambda e: e[1]["message"]["content"].append({"type": "tool_use", "id": "fake"}),
        lambda e: e[2]["modelUsage"].update({"other-model": {}}),
        lambda e: e[2]["usage"].update(output_tokens=True),
        lambda e: e[2]["usage"].update(input_tokens=-1),
        lambda e: e[2].update(result="Different answer"),
    ],
)
def test_rejects_model_tools_stop_and_usage_mutations(mutation):
    rows = events()
    mutation(rows)
    with pytest.raises(m.ClaudeTransportError):
        parse(rows)


@pytest.mark.parametrize("content", [None, {}, [None], ["bad"]])
def test_bad_content_has_safe_error(content):
    rows = events()
    rows[1]["message"]["content"] = content
    with pytest.raises(m.ClaudeTransportError):
        parse(rows)


def test_enforces_output_cap_even_on_normal_stop():
    rows = events()
    rows[2]["usage"]["output_tokens"] = 513
    rows[2]["modelUsage"][m.MODEL]["outputTokens"] = 513
    with pytest.raises(m.ClaudeTransportError):
        parse(rows)


@pytest.mark.parametrize("value", [None, [], "bad", {"outputTokens": True}])
def test_rejects_invalid_per_model_usage(value):
    rows = events()
    rows[2]["modelUsage"][m.MODEL] = value
    with pytest.raises(m.ClaudeTransportError):
        parse(rows)


def test_second_model_message_rejected():
    rows = events()
    extra = copy.deepcopy(rows[1])
    extra["message"]["id"] = "other-message"
    rows.insert(2, extra)
    with pytest.raises(m.ClaudeTransportError):
        parse(rows)


def test_environment_never_mutates_parent_or_keeps_api_overrides():
    source = {
        "PATH": "/synthetic",
        "ANTHROPIC_API_KEY": "fake-secret",
        "ANTHROPIC_BASE_URL": "https://invalid.example",
        "CLAUDE_CODE_OAUTH_TOKEN": "fake",
        "CLAUDE_CODE_USE_BEDROCK": "1",
        "MAX_THINKING_TOKENS": "100",
    }
    previous = source.copy()
    result = m.subscription_environment(source)
    assert source == previous
    assert not any(
        k in result
        for k in [
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_BASE_URL",
            "CLAUDE_CODE_OAUTH_TOKEN",
            "CLAUDE_CODE_USE_BEDROCK",
        ]
    )
    assert result["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "512"


@pytest.mark.parametrize(
    "field,value",
    [
        ("authMethod", "api_key"),
        ("apiProvider", "thirdParty"),
        ("subscriptionType", "free"),
        ("loggedIn", False),
    ],
)
def test_no_api_auth_fallback(field, value):
    status = {
        "loggedIn": True,
        "authMethod": "claude.ai",
        "apiProvider": "firstParty",
        "subscriptionType": "max",
    }
    status[field] = value
    with pytest.raises(m.ClaudeTransportError):
        m.require_subscription(status)


def test_gold_extra_fields_never_enter_prompt():
    text = m.prompt_for(
        "Synthetic question",
        [
            {
                "id": "s",
                "text": "Synthetic evidence",
                "title": "Synthetic title",
                "answer": "GOLD_SENTINEL",
                "labels": ["GOLD_SENTINEL"],
                "score": 123,
            }
        ],
    )
    assert "GOLD_SENTINEL" not in text
    assert '"score"' not in text


def test_nested_title_is_not_a_hidden_gold_serialization_channel():
    with pytest.raises(m.ClaudeTransportError):
        m.prompt_for(
            "Synthetic question",
            [{"id": "s", "text": "Synthetic evidence", "title": {"answer": "GOLD_SENTINEL"}}],
        )


def test_command_disables_tools_and_customizations_without_bare():
    args = m.command("synthetic-cli")
    for flag in ["--safe-mode", "--restricted", "--strict-mcp-config", "--no-session-persistence"]:
        assert flag in args
    assert args[args.index("--tools") + 1] == ""
    assert args[args.index("--model") + 1] == m.MODEL
    assert args[args.index("--permission-mode") + 1] == "dontAsk"
    assert "--bare" not in args


def generator(monkeypatch):
    gate = m.BillingGate(True, "user_confirmed_usage_settings", time.time())
    g = m.ClaudeSubscriptionGenerator(billing_gate=gate)
    monkeypatch.setattr(g, "preflight", lambda: {})
    return g


def test_duplicate_claim_never_relaunches(monkeypatch, tmp_path):
    g = generator(monkeypatch)
    calls = []

    class Child:
        returncode = 0

        def communicate(self, prompt, timeout):
            calls.append(prompt)
            return "\n".join(json.dumps(e) for e in events()), ""

    monkeypatch.setattr(m.subprocess, "Popen", lambda *a, **kw: Child())
    context = [{"id": "s", "text": "Synthetic evidence"}]
    assert g.generate("Synthetic question", context, tmp_path) == "Synthetic answer"
    with pytest.raises(m.ClaudeTransportError, match="request_already_claimed"):
        g.generate("Synthetic question", context, tmp_path)
    assert len(calls) == 1
    for p in tmp_path.rglob("*"):
        assert p.stat().st_mode & 0o077 == 0


def test_timeout_terminates_only_owned_group_and_retains_raw(monkeypatch, tmp_path):
    g = generator(monkeypatch)
    kills = []

    class Child:
        pid = 987654
        returncode = -15
        count = 0

        def communicate(self, *args, **kwargs):
            self.count += 1
            if self.count == 1:
                raise subprocess.TimeoutExpired("synthetic-cli", 1)
            return "synthetic partial", "synthetic diagnostic"

    monkeypatch.setattr(m.subprocess, "Popen", lambda *a, **kw: Child())
    monkeypatch.setattr(m.os, "killpg", lambda pid, sig: kills.append((pid, sig)))
    with pytest.raises(m.ClaudeTransportError, match="cli_interrupted_or_timeout"):
        g.generate("Synthetic question", [{"id": "s", "text": "Synthetic evidence"}], tmp_path)
    assert kills == [(987654, signal.SIGTERM)]
    assert len(list(tmp_path.rglob("stdout.private.jsonl"))) == 1
    assert not list(tmp_path.rglob("completed.json"))


def test_preflight_rejects_api_before_any_model_process(monkeypatch):
    g = m.ClaudeSubscriptionGenerator(
        billing_gate=m.BillingGate(True, "user_confirmed_usage_settings", time.time())
    )
    responses = iter(
        [
            SimpleNamespace(stdout=m.CLI_VERSION),
            SimpleNamespace(
                stdout=json.dumps(
                    {
                        "loggedIn": True,
                        "authMethod": "api_key",
                        "apiProvider": "firstParty",
                        "subscriptionType": "max",
                    }
                )
            ),
        ]
    )
    monkeypatch.setattr(m.subprocess, "run", lambda *a, **kw: next(responses))
    with pytest.raises(m.ClaudeTransportError):
        g.preflight()


def test_symlink_receipt_parent_rejected(monkeypatch, tmp_path):
    g = generator(monkeypatch)
    target = tmp_path / "outside"
    target.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(target, target_is_directory=True)
    monkeypatch.setattr(m.subprocess, "Popen", lambda *a, **kw: pytest.fail("must reject before child"))
    with pytest.raises(m.ClaudeTransportError):
        g.generate("Synthetic question", [{"id": "s", "text": "Synthetic evidence"}], alias)


def test_sigterm_during_child_cleans_up_and_restores_handler(monkeypatch, tmp_path):
    g = generator(monkeypatch)
    previous = signal.getsignal(signal.SIGTERM)
    kills = []
    handlers = []

    class Child:
        pid = 987654
        returncode = -15
        count = 0

        def communicate(self, *args, **kwargs):
            self.count += 1
            if self.count == 1:
                handler = signal.getsignal(signal.SIGTERM)
                handlers.append(callable(handler))
                if callable(handler):
                    handler(signal.SIGTERM, None)
                raise KeyboardInterrupt("fake signal interrupted request")
            return "synthetic partial", ""

    monkeypatch.setattr(m.subprocess, "Popen", lambda *a, **kw: Child())
    monkeypatch.setattr(m.os, "killpg", lambda pid, sig: kills.append((pid, sig)))
    with pytest.raises(m.ClaudeTransportError, match="cli_interrupted_or_timeout"):
        g.generate("Synthetic question", [{"id": "s", "text": "Synthetic evidence"}], tmp_path)
    assert handlers == [True], "Owned child must have scoped SIGTERM cleanup"
    assert signal.getsignal(signal.SIGTERM) == previous
    assert kills == [(987654, signal.SIGTERM)]
    assert not list(tmp_path.rglob("completed.json"))


def test_spawn_failure_restores_sigterm_handler(monkeypatch, tmp_path):
    g = generator(monkeypatch)
    previous = signal.getsignal(signal.SIGTERM)

    def fail_spawn(*args, **kwargs):
        raise OSError("synthetic spawn failure")

    monkeypatch.setattr(m.subprocess, "Popen", fail_spawn)
    try:
        with pytest.raises((m.ClaudeTransportError, OSError)):
            g.generate("Synthetic question", [{"id": "s", "text": "Synthetic evidence"}], tmp_path)
        assert signal.getsignal(signal.SIGTERM) == previous
        assert not list(tmp_path.rglob("completed.json"))
    finally:
        signal.signal(signal.SIGTERM, previous)


def test_actual_cli_shape_final_result_supplies_normal_stop():
    rows = events()
    rows[1]["message"]["stop_reason"] = None
    rows[2]["stop_reason"] = "end_turn"
    # This is a model capability field, not evidence of the request's wire cap.
    rows[2]["modelUsage"][m.MODEL]["maxOutputTokens"] = 32000
    answer, usage = parse(rows)
    assert answer == "Synthetic answer"
    assert usage["finish_reason"] == "end_turn"
    assert usage["cost_is_invoice"] is False


@pytest.mark.parametrize(
    "assistant_stop,result_stop",
    [
        (None, None),
        (None, "max_tokens"),
        ("end_turn", "max_tokens"),
        ("max_tokens", "end_turn"),
        ("refusal", "end_turn"),
        (None, "tool_use"),
    ],
)
def test_final_result_stop_rejects_unknown_or_conflicting_stop(assistant_stop, result_stop):
    rows = events()
    rows[1]["message"]["stop_reason"] = assistant_stop
    rows[2]["stop_reason"] = result_stop
    with pytest.raises(m.ClaudeTransportError, match="non_normal_or_unknown_stop"):
        parse(rows)


def test_repeated_real_self_signals_do_not_interrupt_owned_cli_cleanup(monkeypatch, tmp_path):
    """Real signals target only this pytest process; the child/group are entirely fake."""
    import os

    g = generator(monkeypatch)
    before = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    kills, communications = [], []

    def cancel(_sig, _frame):
        raise KeyboardInterrupt("synthetic owner cancellation")

    class Child:
        pid = 987654
        returncode = -15

        def communicate(self, *args, **kwargs):
            communications.append(kwargs.get("timeout"))
            if len(communications) == 1:
                os.kill(os.getpid(), signal.SIGTERM)
                raise AssertionError("initial cancellation was not delivered")
            for sig in (signal.SIGTERM, signal.SIGINT):
                assert signal.getsignal(sig) == signal.SIG_IGN
                os.kill(os.getpid(), sig)
            return "retained synthetic partial", "retained synthetic error"

    def kill_group(pid, sig):
        kills.append((pid, sig))
        assert pid == Child.pid
        for repeated in (signal.SIGTERM, signal.SIGINT):
            assert signal.getsignal(repeated) == signal.SIG_IGN
            os.kill(os.getpid(), repeated)

    try:
        for sig in before:
            signal.signal(sig, cancel)
        scoped = {sig: signal.getsignal(sig) for sig in before}
        monkeypatch.setattr(m.subprocess, "Popen", lambda *a, **kw: Child())
        monkeypatch.setattr(m.os, "killpg", kill_group)
        with pytest.raises(m.ClaudeTransportError, match="cli_interrupted_or_timeout"):
            g.generate("Synthetic question", [{"id": "s", "text": "Synthetic evidence"}], tmp_path)
        assert {sig: signal.getsignal(sig) for sig in before} == scoped
    finally:
        for sig, handler in before.items():
            signal.signal(sig, handler)
    assert kills == [(Child.pid, signal.SIGTERM)] and communications == [180, 5]
    assert len(list(tmp_path.rglob("stdout.private.jsonl"))) == 1
    assert not list(tmp_path.rglob("completed.json"))

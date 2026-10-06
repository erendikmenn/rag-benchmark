"""The ordinary CLI must never trigger hosted judging implicitly."""

from pathlib import Path

import pytest

from rag_benchmark.cli import main
from rag_benchmark import semantic


@pytest.mark.parametrize("budget", ["-1", "1"])
def test_cli_blocks_invalid_or_unacknowledged_hosted_execution(monkeypatch, capsys, budget):
    monkeypatch.setattr(semantic, "evaluate_bundle", lambda *a, **kw: pytest.fail("must not execute"))
    assert main(["semantic-judge", "--evaluation-dir", "missing", "--max-requests", budget]) == 1
    assert "Error:" in capsys.readouterr().err


def test_cli_default_is_zero_request_dry_run(monkeypatch):
    calls = []
    monkeypatch.setattr(semantic, "evaluate_bundle", lambda *a, **kw: calls.append((a, kw)))
    assert main(["semantic-judge", "--evaluation-dir", "evaluation"]) == 0
    assert calls == [((Path("evaluation"),), {"max_requests": 0})]


def test_cli_passes_explicit_hosted_budget_without_a_secret_argument(monkeypatch):
    calls = []
    monkeypatch.setattr(semantic, "evaluate_bundle", lambda *a, **kw: calls.append((a, kw)))
    assert main(["semantic-judge", "--evaluation-dir", "evaluation", "--max-requests", "40",
                 "--allow-hosted-judge"]) == 0
    assert calls == [((Path("evaluation"),), {"max_requests": 40})]

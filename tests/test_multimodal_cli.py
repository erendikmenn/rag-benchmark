"""Public exports reflect only measured cells; local servers cannot shift context."""
import json

import pytest

from rag_benchmark.multimodal_cli import export_report, generation_server_command


@pytest.mark.parametrize("command", ["asr", "prepare", "preflight", "analyze"])
def test_delegated_command_help_preserves_argument_contract(command, capsys):
    from rag_benchmark.multimodal_cli import main
    with pytest.raises(SystemExit) as result:
        main([command, "--help"])
    assert result.value.code == 0
    assert "usage:" in capsys.readouterr().out


def test_export_retains_coverage_and_never_copies_local_errors(tmp_path):
    run, out = tmp_path / "run", tmp_path / "public"
    run.mkdir()
    report = {"dataset": {"id": "frozen", "revision": "a" * 40, "query_count": 9, "corpus_count": 100},
        "scope": "partial", "evaluated_query_count": 2,
        "cells": [{"variant_id": "measured", "status": "completed", "primary": True,
            "budget_mode": "per_channel", "candidate_k": None, "metrics": {"hit@5": 0.5}},
            {"variant_id": "not_measured", "status": "planned", "primary": True, "metrics": {"hit@5": 1}}]}
    (run / "report.json").write_text(json.dumps(report))
    (run / "matrix.csv").write_text("status\ncompleted\nplanned\n")
    (run / "failures.jsonl").write_text("SECRET LOCAL ERROR")
    export_report(run, out)
    text = (out / "report.md").read_text()
    assert "2 evaluated queries out of 9" in text
    assert "| measured |" in text and "0.5000" in text
    assert "| not_measured |" not in text
    assert not (out / "failures.jsonl").exists()
    assert json.loads((out / "report.json").read_text())["cells"][1]["status"] == "planned"


def test_server_verifies_both_files_and_disables_context_shift(tmp_path, monkeypatch):
    paths = {}
    for name in ("binary", "model", "projector"):
        paths[name] = tmp_path / name
        paths[name].write_bytes(name.encode())
    config = {"model_path": str(paths["model"]), "projector_path": str(paths["projector"]),
        "model_sha256": "model-hash", "projector_sha256": "projector-hash", "model": "local"}
    monkeypatch.setattr("rag_benchmark.cli.file_sha256", lambda path: str(path).split("/")[-1] + "-hash")
    command = generation_server_command(paths["binary"], config, context=8192, parallel=2)
    assert "--no-context-shift" in command
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert command[command.index("--ctx-size") + 1] == "16384"
    assert command[command.index("--mmproj") + 1] == str(paths["projector"])
    config["projector_sha256"] = "wrong"
    with pytest.raises(ValueError, match="projector checksum"):
        generation_server_command(paths["binary"], config)

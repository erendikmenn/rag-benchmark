import json
import shutil
import sqlite3

import pytest

from rag_benchmark.models import stable_hash
from rag_benchmark.multimodal_qa import sha256
from rag_benchmark.multimodal_qa_plan import build_plan, expected_dataset_identity


def dataset(tmp_path, identifier="vidore-v3-fixture-en"):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    from rag_benchmark.multimodal_data import write_dataset

    root = tmp_path / "data/multimodal" / identifier
    (root / "media").mkdir(parents=True)
    (root / "media/p.jpg").write_bytes(b"fixed media")
    raw = root.parent / "raw" / identifier.rsplit("-", 1)[0] / "queries/source.parquet"
    raw.parent.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "query_id": "q",
                    "query": "Private question",
                    "language": "english",
                    "answer": "Private gold",
                    "raw_answers": [],
                }
            ]
        ),
        raw,
    )
    write_dataset(
        root,
        dataset_id=identifier,
        track="document",
        revision="fixed",
        corpus=[{"id": "p", "text": "Private source", "media": {"image": "media/p.jpg"}}],
        queries=[
            {
                "id": "q",
                "text": "Private question",
                "metadata": {
                    "language": "english",
                    "answer": "Private gold",
                    "answer_use": "evaluation_only",
                },
            }
        ],
        qrels=[{"query_id": "q", "corpus_id": "p", "relevance": 1}],
        sources=[{"url": "https://fixture/queries/source.parquet", "sha256": sha256(raw)}],
        license="fixture",
        text_source="published_markdown",
        metadata={"language": "en"},
    )
    return root


def report(
    tmp_path,
    data,
    *,
    name="run",
    status="completed",
    scope="frozen_collection",
    partial=False,
    identity=None,
    variant="document__b__none",
    store=True,
):
    run = tmp_path / "runs/multimodal" / name
    run.mkdir(parents=True)
    cell = {
        "variant_id": variant,
        "budget_mode": "per_channel",
        "candidate_k": None,
        "status": status,
        "query_count": 1,
        "completed_queries": 1 if status == "completed" else 0,
    }
    if identity or status == "completed":
        cell.update(identity=identity or stable_hash(name), metrics={"fixture": 1.0})
    value = {
        "scope": scope,
        "evaluated_query_count": 0 if partial else 1,
        "dataset": {"id": data.name, "identity": expected_dataset_identity(data), "query_count": 1},
        "cells": [cell],
    }
    (run / "report.json").write_text(json.dumps(value))
    if store:
        with sqlite3.connect(run / "progress.sqlite3") as connection:
            connection.execute("CREATE TABLE results (cell TEXT,query_id TEXT,payload TEXT)")
            connection.execute(
                "INSERT INTO results VALUES (?,?,?)", (cell.get("identity", "pending"), "q", "{}")
            )
    return run


def plan(tmp_path, datasets):
    return build_plan(
        datasets,
        workspace=tmp_path,
        reports_root=tmp_path / "reports/multimodal",
        runs_root=tmp_path / "runs/multimodal",
        output_root=tmp_path / "runs/multimodal-qa",
    )


def test_controls_once_per_dataset_not_multiplied_across_cells(tmp_path):
    source = dataset(tmp_path)
    report(tmp_path, source, name="one", identity=stable_hash("one"))
    report(tmp_path, source, name="two", identity=stable_hash("two"))
    result = plan(tmp_path, [source])
    assert result["controls_job_count"] == 1 and result["retrieved_job_count"] == 2
    assert result["controls_planned_qa_tasks"] == 4 and result["retrieved_planned_qa_tasks"] == 4
    assert result["schedulable_planned_qa_tasks"] == 8
    assert result["qa_generation_completed_by_plan"] == 0 and result["model_requests_started"] == 0
    assert [row["conditions"] for row in result["jobs"]].count(["closed_book", "oracle"]) == 1
    assert all(
        row["conditions"] == ["retrieved"] for row in result["jobs"] if row["kind"] == "retrieved_only"
    )


def test_identical_report_exports_group_by_actual_identity(tmp_path):
    source = dataset(tmp_path)
    run = report(tmp_path, source)
    public = tmp_path / "reports/multimodal" / run.name
    public.mkdir(parents=True)
    shutil.copyfile(run / "report.json", public / "report.json")
    result = plan(tmp_path, [source])
    assert result["retrieved_job_count"] == 1
    item = next(row for row in result["jobs"] if row["kind"] == "retrieved_only")
    assert len(item["observed_source_reports"]) == 2
    assert item["retrieval_report_sha256"] == sha256(run / "report.json")
    assert "--conditions retrieved --representations text image" in item["command"]


def test_failed_pending_partial_smoke_exclusion_and_exact_denominator(tmp_path):
    source = dataset(tmp_path)
    report(tmp_path, source, name="failed", status="failed")
    report(tmp_path, source, name="partial", partial=True)
    report(tmp_path, source, name="smoke", scope="smoke", variant="document__e__none")
    result = plan(tmp_path, [source])
    assert result["retrieved_job_count"] == 0 and result["schedulable_planned_qa_tasks"] == 4
    assert result["pending_observed_retrieval_cells"] == 1
    assert result["pending_future_qa_tasks_if_sources_complete"] == 2
    assert len(result["excluded_reports"]) == 2
    assert result["pending_cells"][0]["retrieval_cell_identity"] is None
    assert result["pending_cells"][0]["observed_status_counts"] == {"failed": 1}


def test_missing_actual_storage_is_pending_not_executable(tmp_path):
    source = dataset(tmp_path)
    report(tmp_path, source, store=False)
    result = plan(tmp_path, [source])
    assert result["retrieved_job_count"] == 0 and result["pending_observed_retrieval_cells"] == 1
    assert result["pending_cells"][0]["qa_status"] == "pending_actual_ranking_storage"
    assert result["pending_cells"][0]["retrieval_cell_identity"] == stable_hash("run")


def test_stale_dataset_identity_excluded_and_completed_duplicate_conflict_rejected(tmp_path):
    source = dataset(tmp_path)
    run = report(tmp_path, source)
    file = run / "report.json"
    value = json.loads(file.read_text())
    value["dataset"]["identity"] = "stale"
    file.write_text(json.dumps(value))
    result = plan(tmp_path, [source])
    assert result["retrieved_job_count"] == 0
    assert "identity/count differs" in result["excluded_reports"][0]["reason"]
    value["dataset"]["identity"] = expected_dataset_identity(source)
    file.write_text(json.dumps(value))
    public = tmp_path / "reports/multimodal" / run.name
    public.mkdir(parents=True)
    value["cells"][0]["metrics"] = {"fixture": 0.0}
    (public / "report.json").write_text(json.dumps(value))
    with pytest.raises(ValueError, match="conflicting metrics"):
        plan(tmp_path, [source])


def test_manifest_no_private_payload_and_frozen_sources_unchanged(tmp_path):
    source = dataset(tmp_path)
    report(tmp_path, source)
    before = {str(path): sha256(path) for path in (tmp_path / "data").rglob("*") if path.is_file()}
    result = plan(tmp_path, [source])
    serialized = json.dumps(result)
    for private in ["Private question", "Private gold", "Private source", "query_id", "raw_answers"]:
        assert private not in serialized
    assert str(tmp_path) not in serialized
    assert not result["actual_generation_reuse_claimed"]
    assert before == {str(path): sha256(path) for path in (tmp_path / "data").rglob("*") if path.is_file()}
    assert not (tmp_path / "runs/multimodal-qa").exists()


def test_two_datasets_each_have_one_control_job(tmp_path):
    first = dataset(tmp_path, "vidore-v3-one-en")
    second = dataset(tmp_path, "vidore-v3-two-en")
    report(tmp_path, first, name="first")
    report(tmp_path, second, name="second")
    result = plan(tmp_path, [first, second])
    assert result["dataset_count"] == result["controls_job_count"] == 2
    assert result["retrieved_job_count"] == 2
    assert result["schedulable_planned_qa_tasks"] == 12


def test_runner_cli_selects_only_requested_conditions_representations(monkeypatch, tmp_path):
    from rag_benchmark import multimodal_qa_runner

    captured = {}

    def fake_run(*args, **kwargs):
        captured.update(kwargs["config"]["protocol"])
        return {"status": "partial", "completed_tasks": 0, "planned_query_condition_representation_tasks": 1}

    monkeypatch.setattr(multimodal_qa_runner, "run_qa", fake_run)
    multimodal_qa_runner.main(
        [
            "--dataset",
            str(tmp_path),
            "--raw-queries",
            str(tmp_path / "raw"),
            "--run-dir",
            str(tmp_path / "run"),
            "--conditions",
            "retrieved",
            "--representations",
            "image",
        ]
    )
    assert captured["conditions"] == ["retrieved"]
    assert captured["representations"] == ["image"]
    with pytest.raises(SystemExit):
        multimodal_qa_runner.main(
            [
                "--dataset",
                str(tmp_path),
                "--raw-queries",
                str(tmp_path / "raw"),
                "--run-dir",
                str(tmp_path / "run"),
                "--conditions",
                "invented",
            ]
        )


def test_plan_commands_enforce_dataset_and_retrieval_hashes(tmp_path):
    source = dataset(tmp_path)
    report(tmp_path, source)
    result = plan(tmp_path, [source])
    for job in result["jobs"]:
        args = job["command_argv"]
        assert args[args.index("--expected-dataset-identity") + 1] == job["dataset_identity"]
        if job["kind"] == "retrieved_only":
            assert (
                args[args.index("--expected-retrieval-report-sha256") + 1] == job["retrieval_report_sha256"]
            )
        else:
            assert "--expected-retrieval-report-sha256" not in args

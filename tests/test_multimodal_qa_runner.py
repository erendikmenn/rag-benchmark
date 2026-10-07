import copy
import json
import sqlite3
from types import SimpleNamespace

import pytest

from rag_benchmark.multimodal_qa import sha256
from rag_benchmark.multimodal_qa_runner import frozen_config, run_qa, verified_dataset, verified_retrieval


class FakeLocalTransport:
    identity = "verified-fixture-model-and-runtime"

    def __init__(self, *, fail=False):
        self.requests = []
        self.fail = fail

    def preflight(self):
        return {"model_sha256": "fixture-verified"}

    def generate(self, request):
        self.requests.append(copy.deepcopy(request))
        if self.fail:
            raise RuntimeError("fixture retained generation failure")
        return {
            "text": json.dumps(
                {
                    "answer": "Correct fact",
                    "citations": [request["sources"][0]["id"]] if request["sources"] else [],
                }
            ),
            "usage": {"new_tokens": 4},
        }


def fixture(tmp_path, monkeypatch, *, empty_text=False, reference="Correct fact"):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    from rag_benchmark.multimodal_data import write_dataset

    root = tmp_path / "dataset"
    (root / "media").mkdir(parents=True)
    (root / "media/p.jpg").write_bytes(b"fixture image bytes")
    raw = tmp_path / "raw.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "query_id": "q",
                    "query": "What fact?",
                    "language": "english",
                    "answer": reference,
                    "raw_answers": ["Not certified alias"],
                }
            ]
        ),
        raw,
    )
    write_dataset(
        root,
        dataset_id="qa-fixture",
        track="document",
        revision="frozen",
        corpus=[{"id": "p", "text": "" if empty_text else "Source fact", "media": {"image": "media/p.jpg"}}],
        queries=[
            {
                "id": "q",
                "text": "What fact?",
                "metadata": {
                    "language": "english",
                    "answer": reference,
                    "answer_use": "evaluation_only",
                },
            }
        ],
        qrels=[{"query_id": "q", "corpus_id": "p", "relevance": 2}],
        sources=[{"url": "https://fixture/resolve/frozen/queries/raw.parquet", "sha256": sha256(raw)}],
        license="fixture",
        text_source="published_markdown",
        metadata={"language": "en"},
    )
    # Temporary fixture paths are not real private outputs. Only mock git-ignore check.
    monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: SimpleNamespace(returncode=0))
    return root, raw, tmp_path / "runs"


def make_retrieval(tmp_path, dataset):
    from rag_benchmark.multimodal import graded_metrics

    run = tmp_path / "retrieval"
    run.mkdir()
    metric = graded_metrics(["p"], dataset.qrels["q"])
    report = {
        "scope": "frozen_collection",
        "evaluated_query_count": 1,
        "dataset": {"identity": dataset.identity},
        "cells": [
            {
                "identity": "actual-cell",
                "status": "completed",
                "query_count": 1,
                "completed_queries": 1,
                "metrics": metric,
            }
        ],
    }
    (run / "report.json").write_text(json.dumps(report))
    with sqlite3.connect(run / "progress.sqlite3") as connection:
        connection.execute("CREATE TABLE results (cell TEXT,query_id TEXT,payload TEXT)")
        connection.execute(
            "INSERT INTO results VALUES (?,?,?)",
            (
                "actual-cell",
                "q",
                json.dumps(
                    {"query_id": "q", "ranking": [{"id": "p", "score": 1, "rank": 1}], "metrics": metric}
                ),
            ),
        )
    return run


def test_resume_gold_exclusion_and_honest_denominator(tmp_path, monkeypatch):
    root, raw, out = fixture(tmp_path, monkeypatch)
    transport = FakeLocalTransport()
    first = run_qa(root, [raw], out, transport=transport, max_new=1)
    assert first["completed_tasks"] == 1 and first["planned_query_condition_representation_tasks"] == 6
    assert first["status_counts"]["pending"] == 5 and first["status"] == "partial"
    second = run_qa(root, [raw], out, transport=transport)
    assert second["completed_tasks"] == 4 and second["status_counts"]["unsupported"] == 2
    assert len(transport.requests) == 4
    third = run_qa(root, [raw], out, transport=transport)
    assert third["completed_tasks"] == second["completed_tasks"] and len(transport.requests) == 4
    assert third["invocation"]["new_inference_requests"] == 0
    for request in transport.requests:
        text = json.dumps(request)
        assert "Correct fact" not in text and "Not certified alias" not in text
        assert "references" not in text and "metadata" not in text and "answer_use" not in text
    assert all(group["oracle"] == (group["condition"] == "oracle") for group in second["evaluated_groups"])
    assert second["human_semantic_accuracy"] is None


def test_real_completed_retrieval_cell_integration(tmp_path, monkeypatch):
    root, raw, out = fixture(tmp_path, monkeypatch)
    retrieval = make_retrieval(tmp_path, verified_dataset(root))
    transport = FakeLocalTransport()
    result = run_qa(
        root, [raw], out, transport=transport, retrieval_run=retrieval, retrieval_cell="actual-cell"
    )
    assert result["completed_tasks"] == 6 and result["status"] == "completed"
    assert len(transport.requests) == 6
    assert "p" in transport.requests[-1]["sources"][0]["id"]


def test_altered_source_config_runtime_reject_cached_results(tmp_path, monkeypatch):
    root, raw, out = fixture(tmp_path, monkeypatch)
    run_qa(root, [raw], out, transport=FakeLocalTransport(), max_new=1)
    config = frozen_config()
    config["generator"]["seed"] = 43
    with pytest.raises(ValueError, match="Source/config/retrieval changed"):
        run_qa(root, [raw], out, config=config, transport=FakeLocalTransport())
    other = FakeLocalTransport()
    other.identity = "changed-runtime"
    with pytest.raises(ValueError, match="model/runtime changed"):
        run_qa(root, [raw], out, transport=other)
    (root / "media/p.jpg").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        run_qa(root, [raw], out, transport=FakeLocalTransport())


def test_evidence_failures_retained_and_no_silent_query_skips(tmp_path, monkeypatch):
    root, raw, out = fixture(tmp_path, monkeypatch, empty_text=True)
    transport = FakeLocalTransport()
    config = frozen_config()
    config["protocol"]["conditions"] = ["oracle"]
    result = run_qa(root, [raw], out, config=config, transport=transport)
    assert result["planned_query_condition_representation_tasks"] == 2
    assert result["completed_tasks"] == 1 and result["status_counts"]["failed"] == 1
    assert len(transport.requests) == 1
    with sqlite3.connect(out / "qa.sqlite3") as connection:
        payload = json.loads(
            connection.execute("SELECT payload FROM predictions WHERE status='failed'").fetchone()[0]
        )
    assert "no usable text" in payload["error"]
    assert (
        run_qa(root, [raw], out, config=config, transport=transport)["status_counts"]
        == result["status_counts"]
    )


def test_generation_failure_retry_explicit_and_private_payload(tmp_path, monkeypatch):
    root, raw, out = fixture(tmp_path, monkeypatch)
    config = frozen_config()
    config["protocol"]["conditions"] = ["closed_book"]
    config["protocol"]["representations"] = ["text"]
    result = run_qa(root, [raw], out, config=config, transport=FakeLocalTransport(fail=True))
    assert result["completed_tasks"] == 0 and result["status_counts"]["failed"] == 1
    assert result["evaluated_groups"][0]["completed_only_mean_metrics"] is None
    passing = FakeLocalTransport()
    run_qa(root, [raw], out, config=config, transport=passing)
    assert not passing.requests
    retried = run_qa(root, [raw], out, config=config, transport=passing, retry_failed=True)
    assert retried["completed_tasks"] == 1
    public = json.dumps(retried)
    assert "What fact?" not in public and "Correct fact" not in public
    assert "fixture retained" not in public


def test_retrieval_identity_coverage_and_metrics_tampering_rejected(tmp_path, monkeypatch):
    root, _, _ = fixture(tmp_path, monkeypatch)
    dataset = verified_dataset(root)
    run = make_retrieval(tmp_path, dataset)
    verified_retrieval(run, "actual-cell", dataset)
    path = run / "report.json"
    report = json.loads(path.read_text())
    report["dataset"]["identity"] = "wrong"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="identity differs"):
        verified_retrieval(run, "actual-cell", dataset)
    report["dataset"]["identity"] = dataset.identity
    path.write_text(json.dumps(report))
    with sqlite3.connect(run / "progress.sqlite3") as connection:
        connection.execute("DELETE FROM results")
    with pytest.raises(ValueError, match="exact frozen query set"):
        verified_retrieval(run, "actual-cell", dataset)


def test_completed_prediction_parser_failure_not_counted(tmp_path, monkeypatch):
    root, raw, out = fixture(tmp_path, monkeypatch)
    transport = FakeLocalTransport()
    transport.generate = lambda request: {
        "text": json.dumps({"answer": "Correct fact", "citations": ["not-supplied"]})
    }
    config = frozen_config()
    config["protocol"]["conditions"] = ["closed_book"]
    config["protocol"]["representations"] = ["text"]
    result = run_qa(root, [raw], out, config=config, transport=transport)
    assert result["completed_tasks"] == 0 and result["status_counts"]["failed"] == 1


def test_valid_changed_source_and_retrieval_metrics_invalidate(tmp_path, monkeypatch):
    root, raw, out = fixture(tmp_path, monkeypatch)
    transport = FakeLocalTransport()
    run_qa(root, [raw], out, transport=transport, max_new=1)
    corpus_path = root / "corpus.jsonl"
    rows = [json.loads(line) for line in corpus_path.read_text().splitlines()]
    rows[0]["text"] = "Updated source fact"
    corpus_path.write_text(json.dumps(rows[0]) + "\n")
    manifest_path = root / "dataset.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["corpus.jsonl"]["sha256"] = sha256(corpus_path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Source/config/retrieval changed"):
        run_qa(root, [raw], out, transport=transport)
    assert len(transport.requests) == 1
    dataset = verified_dataset(root)
    run = make_retrieval(tmp_path, dataset)
    with sqlite3.connect(run / "progress.sqlite3") as connection:
        value = json.loads(connection.execute("SELECT payload FROM results").fetchone()[0])
        value["metrics"]["hit@1"] = 0
        connection.execute("UPDATE results SET payload=?", (json.dumps(value),))
    with pytest.raises(ValueError, match="reproduce stored labeled metrics"):
        verified_retrieval(run, "actual-cell", dataset)


def test_run_owner_and_private_directory_guard(tmp_path, monkeypatch):
    import fcntl

    root, raw, out = fixture(tmp_path, monkeypatch)
    out.mkdir()
    with (out / "qa.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="active owner"):
            run_qa(root, [raw], out, transport=FakeLocalTransport())
    monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: SimpleNamespace(returncode=1))
    with pytest.raises(ValueError, match="git-ignored"):
        run_qa(root, [raw], tmp_path / "public", transport=FakeLocalTransport())


def test_explicit_context_budget_unsupported_without_inference(tmp_path, monkeypatch):
    root, raw, out = fixture(tmp_path, monkeypatch)
    config = frozen_config()
    config["protocol"]["maximum_text_characters"] = 1
    transport = FakeLocalTransport()
    result = run_qa(root, [raw], out, config=config, transport=transport)
    assert result["completed_tasks"] == 0
    assert result["status_counts"]["unsupported"] == 6
    assert not transport.requests


@pytest.mark.parametrize("damage", ["score_nan", "score_bool", "rank", "order", "tie_order"])
def test_retrieval_score_rank_order_are_verified(tmp_path, monkeypatch, damage):
    from rag_benchmark.multimodal import graded_metrics

    root, _, _ = fixture(tmp_path, monkeypatch)
    dataset = verified_dataset(root)
    # Add a second valid local gallery ID without touching production data.
    dataset.corpus.append({"id": "z", "text": "other source"})
    run = make_retrieval(tmp_path, dataset)
    with sqlite3.connect(run / "progress.sqlite3") as connection:
        value = json.loads(connection.execute("SELECT payload FROM results").fetchone()[0])
        if damage == "score_nan":
            value["ranking"][0]["score"] = float("nan")
        if damage == "score_bool":
            value["ranking"][0]["score"] = True
        if damage == "rank":
            value["ranking"][0]["rank"] = 999
        if damage == "order":
            value["ranking"] = [{"id": "p", "score": 1, "rank": 1}, {"id": "z", "score": 2, "rank": 2}]
        if damage == "tie_order":
            value["ranking"] = [{"id": "z", "score": 1, "rank": 1}, {"id": "p", "score": 1, "rank": 2}]
        value["metrics"] = graded_metrics([x["id"] for x in value["ranking"]], dataset.qrels["q"])
        connection.execute("UPDATE results SET payload=?", (json.dumps(value),))
    with pytest.raises(ValueError, match="finite and sequential|score order"):
        verified_retrieval(run, "actual-cell", dataset)


def test_full_ranking_hash_binds_score_changes_and_empty_rank_retained(tmp_path, monkeypatch):
    from rag_benchmark.multimodal import graded_metrics

    root, raw, out = fixture(tmp_path, monkeypatch)
    dataset = verified_dataset(root)
    run = make_retrieval(tmp_path, dataset)
    _, before = verified_retrieval(run, "actual-cell", dataset)
    with sqlite3.connect(run / "progress.sqlite3") as connection:
        value = json.loads(connection.execute("SELECT payload FROM results").fetchone()[0])
        value["ranking"][0]["score"] = 2.0
        connection.execute("UPDATE results SET payload=?", (json.dumps(value),))
    _, after = verified_retrieval(run, "actual-cell", dataset)
    assert before["rankings_sha256"] != after["rankings_sha256"]
    value["ranking"] = []
    value["metrics"] = graded_metrics([], dataset.qrels["q"])
    with sqlite3.connect(run / "progress.sqlite3") as connection:
        connection.execute("UPDATE results SET payload=?", (json.dumps(value),))
    path = run / "report.json"
    report = json.loads(path.read_text())
    report["cells"][0]["metrics"] = value["metrics"]
    path.write_text(json.dumps(report))
    rankings, _ = verified_retrieval(run, "actual-cell", dataset)
    assert rankings == {"q": []}
    transport = FakeLocalTransport()
    result = run_qa(root, [raw], out, transport=transport, retrieval_run=run, retrieval_cell="actual-cell")
    assert result["completed_tasks"] == 4 and result["status_counts"]["unsupported"] == 2
    assert len(transport.requests) == 4
    with sqlite3.connect(out / "qa.sqlite3") as connection:
        rows = connection.execute("SELECT payload FROM predictions WHERE condition='retrieved'").fetchall()
    assert all("closed-book substitution is forbidden" in json.loads(row[0])["error"] for row in rows)


@pytest.mark.parametrize("damage", ["metric", "request", "raw_citations", "parsed_prediction"])
def test_completed_cache_integrity_failure_retained_never_reused(tmp_path, monkeypatch, damage):
    root, raw, out = fixture(tmp_path, monkeypatch)
    config = frozen_config()
    config["protocol"]["conditions"] = ["closed_book"]
    config["protocol"]["representations"] = ["text"]
    transport = FakeLocalTransport()
    run_qa(root, [raw], out, config=config, transport=transport)
    with sqlite3.connect(out / "qa.sqlite3") as connection:
        payload = json.loads(connection.execute("SELECT payload FROM predictions").fetchone()[0])
        if damage == "metric":
            payload["metrics"]["answer_em"] = 2.0
        if damage == "request":
            payload["request"]["question"] = "altered query"
        if damage == "raw_citations":
            payload["raw_prediction"] = json.dumps({"answer": "Correct fact", "citations": ["not-supplied"]})
        if damage == "parsed_prediction":
            payload["prediction"]["answer"] = "different parsed answer"
        connection.execute("UPDATE predictions SET payload=?", (json.dumps(payload),))
    result = run_qa(root, [raw], out, config=config, transport=transport, retry_failed=True)
    assert result["completed_tasks"] == 0 and result["status_counts"]["failed"] == 1
    assert len(transport.requests) == 1
    assert result["invocation"]["new_inference_requests"] == 0
    assert result["evaluated_groups"][0]["completed_only_mean_metrics"] is None
    again = run_qa(root, [raw], out, config=config, transport=transport, retry_failed=True)
    assert again["completed_tasks"] == 0 and len(transport.requests) == 1
    with sqlite3.connect(out / "qa.sqlite3") as connection:
        payload = json.loads(connection.execute("SELECT payload FROM predictions").fetchone()[0])
    assert payload["cache_validation_error"]


def test_question_language_frozen_request_projection(tmp_path, monkeypatch):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    root, raw, out = fixture(tmp_path, monkeypatch)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "query_id": "q",
                    "query": "What fact?",
                    "language": "french",
                    "answer": "Correct fact",
                    "raw_answers": [],
                }
            ]
        ),
        raw,
    )
    queries = root / "queries.jsonl"
    query = json.loads(queries.read_text())
    query["metadata"]["language"] = "french"
    queries.write_text(json.dumps(query) + "\n")
    manifest_path = root / "dataset.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["metadata"]["language"] = "fr"
    manifest["files"]["queries.jsonl"]["sha256"] = sha256(queries)
    manifest["sources"][0]["sha256"] = sha256(raw)
    manifest_path.write_text(json.dumps(manifest))
    transport = FakeLocalTransport()
    run_qa(root, [raw], out, transport=transport, max_new=1)
    assert transport.requests[0]["language"] == "french"
    assert "explicitly requested answer language" in transport.requests[0]["system"]
    assert "Correct fact" not in json.dumps(transport.requests[0])


@pytest.mark.parametrize("damage", ["dataset", "retrieval_report"])
def test_stale_execution_plan_fails_before_transport_preflight(tmp_path, monkeypatch, damage):
    root, raw, out = fixture(tmp_path, monkeypatch)
    dataset = verified_dataset(root)
    retrieval = make_retrieval(tmp_path, dataset)
    expected_dataset = dataset.identity
    expected_report = sha256(retrieval / "report.json")
    if damage == "dataset":
        path = root / "corpus.jsonl"
        row = json.loads(path.read_text())
        row["text"] = "Changed valid source"
        path.write_text(json.dumps(row) + "\n")
        manifest_path = root / "dataset.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["files"]["corpus.jsonl"]["sha256"] = sha256(path)
        manifest_path.write_text(json.dumps(manifest))
    else:
        path = retrieval / "report.json"
        report = json.loads(path.read_text())
        report["extra_snapshot_metadata"] = True
        path.write_text(json.dumps(report))
    transport = FakeLocalTransport()

    def forbidden_preflight():
        raise AssertionError("Stale plan must fail before model/transport preflight")

    transport.preflight = forbidden_preflight
    with pytest.raises(
        ValueError, match="Planned dataset identity changed|Planned retrieval report hash changed"
    ):
        run_qa(
            root,
            [raw],
            out,
            transport=transport,
            retrieval_run=retrieval,
            retrieval_cell="actual-cell",
            expected_dataset_identity=expected_dataset,
            expected_retrieval_report_sha256=expected_report,
        )
    assert not transport.requests and not out.exists()


def test_matching_expected_plan_hashes_execute_same_frozen_cell(tmp_path, monkeypatch):
    root, raw, out = fixture(tmp_path, monkeypatch)
    dataset = verified_dataset(root)
    retrieval = make_retrieval(tmp_path, dataset)
    transport = FakeLocalTransport()
    result = run_qa(
        root,
        [raw],
        out,
        transport=transport,
        retrieval_run=retrieval,
        retrieval_cell="actual-cell",
        expected_dataset_identity=dataset.identity,
        expected_retrieval_report_sha256=sha256(retrieval / "report.json"),
        max_new=1,
    )
    assert result["completed_tasks"] == 1 and len(transport.requests) == 1

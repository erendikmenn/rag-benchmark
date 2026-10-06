"""Publication tests deliberately mix public metadata and private local fields."""

import hashlib
import json

import pytest

from rag_benchmark.export import export_report


def make_run(tmp_path):
    run = tmp_path / "runs" / "pilot"
    run.mkdir(parents=True)
    private = "PRIVATE_QUESTION_ANSWER_AND_CONTEXT"
    model = {"model_id": "org/public-model", "revision": "b" * 40, "model_sha256": "c" * 64,
             "model_path": "/Users/private-person/secret-model.gguf", "base_url": "http://private-host:8080",
             "api_key": "SECRET_TOKEN", "system_prompt": private, "max_tokens": 256}
    manifest = {
        "fingerprint": "f" * 64, "split": "dev", "variants": ["bm25"], "question_ids": ["web:article#q1"],
        "retrieval_only": False, "hostname": "private-host", "corpus_sha256": "d" * 64,
        "dataset": {"dataset": "metunlp/ragturk", "revision": "a" * 40, "fingerprint": "e" * 64,
                    "dataset_license": "CC-BY-NC-SA-4.0", "counts": {"corpus_chunks": 100, "dev_questions": 1},
                    "file_sha256": {"corpus.jsonl": "d" * 64}, "question": private},
        "config": {"dataset": {"path": str(tmp_path / "data" / "ragturk"), "seed": 42},
                   "generator": model, "embeddings": {"bge": model}, "laya": model,
                   "experiment": {"candidates": 50, "context_k": 5, "api_key": "SECRET_TOKEN"}},
        "runtime": {"python": "3.12.12", "platform": "macOS-27.0-arm64", "source_hash": "1" * 64,
                    "git_revision": "2" * 40, "packages": {"numpy": "2.5.3", "hostname": "private-host"}},
        "generator_server": {"model_sha256": "c" * 64, "runtime_fingerprint": "3" * 64,
                             "props": {"model_path": "/Users/private-person/model.gguf", "build_info": "b11451-2207c8e57",
                                       "chat_template": private, "hostname": "private-host",
                                       "default_generation_settings": {"n_ctx": 8192, "params": {"temperature": 0.0, "generation_prompt": private, "token": "SECRET_TOKEN"}}}},
    }
    stats = {"n": 1, "complete": True, "metrics": {"recall@5": 1.0, "answer_token_f1": 0.5, "answer": private},
             "retrieval_s": {"n": 1, "p50": .1, "p95": .1, "hostname": "private-host"},
             "by_source": {"web": {"n": 1, "metrics": {"recall@5": 1.0}}}, "question": private}
    summary = {"split": "dev", "retrieval_only": False, "expected_per_variant": 1, "variants": {"bm25": stats}, "prompt": private}
    row = {"variant": "bm25", "query_id": "web:article#q1", "article_id": "web:article", "category": "FACTUAL", "source": "web",
           "question": private, "answer": private, "reference_answers": [private], "context": private,
           "gold_ids": ["web:article#c1"], "candidate_ids": ["web:article#c1"], "ranked_ids": ["web:article#c1"], "context_ids": ["web:article#c1"],
           "metrics": stats["metrics"], "generation_cache_hit": False, "generation_s": 1.25,
           "generator_usage": {"completion_tokens": 42, "finish_reason": "stop", "prompt": private,
                               "prompt_tokens_details": {"cached_tokens": 10, "token": "SECRET_TOKEN"}},
           "retrieval_usage": {"query_cache_hits": {"bge": True}, "method": "bm25", "search": "exact", "corpus_size": 100},
           "reranker_usage": {"candidates": 50, "retained": 50, "backend": "sdk", "dtype": "float32", "prompt": private}}
    for name, value in (("manifest.json", manifest), ("summary.json", summary)):
        (run / name).write_text(json.dumps(value))
    (run / "predictions.jsonl").write_text(json.dumps(row) + "\n")
    (run / "report.md").write_text(private + "\n/Users/private-person\nSECRET_TOKEN\nprivate-host")
    return run, private


def test_export_excludes_source_text_secrets_paths_and_preserves_identity(tmp_path):
    run, private = make_run(tmp_path)
    output = tmp_path / "reports" / "pilot"
    result = export_report(run, output)
    combined = "\n".join(path.read_text() for path in output.iterdir())
    for forbidden in (private, "SECRET_TOKEN", "private-host", "/Users/", str(tmp_path), "generation_prompt", "system_prompt"):
        assert forbidden not in combined
    assert set(result["files"]) == {"summary.json", "report.md", "provenance.json", "per-query-metrics.jsonl"}
    assert result["exported_rows"] == 1
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["dataset"]["revision"] == "a" * 40
    assert provenance["dataset"]["file_sha256"]["corpus.jsonl"] == "d" * 64
    assert provenance["runtime"]["source_hash"] == "1" * 64
    assert provenance["settings"]["generator"]["model_id"] == "org/public-model"
    assert provenance["settings"]["generator"]["model_path"] == "<local>"
    assert provenance["generator_server"]["chat_template_sha256"] == hashlib.sha256(private.encode()).hexdigest()
    row = json.loads((output / "per-query-metrics.jsonl").read_text())
    assert row["query_id"] == "web:article#q1"
    assert row["gold_ids"] == ["web:article#c1"]
    assert row["generator_usage"]["completion_tokens"] == 42
    assert row["reranker_usage"]["retained"] == 50
    assert row["retrieval_usage"]["query_cache_hits"] == {"bge": True}
    assert "final benchmark claims" in (output / "report.md").read_text()
    for name, expected in result["files"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == expected


def test_export_is_deterministic_and_does_not_modify_local_run(tmp_path):
    run, _ = make_run(tmp_path)
    before = {path.name: path.read_bytes() for path in run.iterdir()}
    first = export_report(run, tmp_path / "reports" / "one")
    second = export_report(run, tmp_path / "reports" / "two")
    assert first == second
    assert before == {path.name: path.read_bytes() for path in run.iterdir()}


@pytest.mark.parametrize("destination", ["run", "run-child", "run-parent", "dataset", "dataset-child", "data-parent"])
def test_refuses_run_or_dataset_overlap(tmp_path, destination):
    run, _ = make_run(tmp_path)
    targets = {"run": run, "run-child": run / "export", "run-parent": run.parent,
               "dataset": tmp_path / "data" / "ragturk", "dataset-child": tmp_path / "data" / "ragturk" / "export",
               "data-parent": tmp_path / "data"}
    with pytest.raises(ValueError, match="overlaps"):
        export_report(run, targets[destination])


def test_existing_output_is_never_overwritten(tmp_path):
    run, _ = make_run(tmp_path)
    output = tmp_path / "existing"
    output.mkdir()
    keep = output / "important.txt"
    keep.write_text("keep")
    with pytest.raises(FileExistsError):
        export_report(run, output)
    assert keep.read_text() == "keep"


def test_rejects_text_in_identifier_without_partial_export(tmp_path):
    run, _ = make_run(tmp_path)
    row = json.loads((run / "predictions.jsonl").read_text())
    row["query_id"] = "A private question masquerading as an identifier?"
    (run / "predictions.jsonl").write_text(json.dumps(row) + "\n")
    output = tmp_path / "bad-export"
    with pytest.raises(ValueError, match="identifier"):
        export_report(run, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".rag-export-*"))

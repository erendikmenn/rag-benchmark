import hashlib
import json
import sqlite3

import numpy as np
import pytest

from rag_benchmark.multimodal import run_matrix
from rag_benchmark.multimodal_dimensions import main, project_prefix, run_dimension_sweep
from rag_benchmark.multimodal_models import EmbeddingGemma2Adapter
from rag_benchmark.multimodal_specialists import SpecialistAdapter


def _unit(rows):
    values = np.zeros((len(rows), 768), dtype=np.float32)
    for i, fields in enumerate(rows):
        for column, value in fields.items():
            values[i, column] = value
    return values / np.linalg.norm(values, axis=1, keepdims=True)


def _fixture(tmp_path, monkeypatch, *, specialist=True, legacy=False):
    dataset, run, cache = (tmp_path / name for name in ("data", "base", "cache"))
    dataset.mkdir()
    manifest = {"id": "dimension-fixture", "revision": "fixed-revision", "track": "photo", "split": "test"}
    corpus = []
    for identifier in "abcdef":
        (dataset / (identifier + ".png")).write_bytes(b"synthetic-media-fixture")
        corpus.append({"id": identifier, "text": "PRIVATE_SOURCE", "media": {"image": identifier + ".png"}})
    queries = [{"id": "q1", "text": "PRIVATE_QUERY_1"}, {"id": "q2", "text": "PRIVATE_QUERY_2"}]
    qrels = [{"query_id": q, "corpus_id": d, "relevance": rel}
             for q, d, rel in (("q1", "a", 1), ("q1", "c", 3), ("q2", "d", 1), ("q2", "f", 3))]
    (dataset / "dataset.json").write_text(json.dumps(manifest))
    for name, rows in (("corpus", corpus), ("queries", queries), ("qrels", qrels)):
        (dataset / (name + ".jsonl")).write_text("\n".join(json.dumps(row) for row in rows))
    documents = _unit([{5: .1, 600: 1}, {1: 1, 600: 1}, {0: 1}, {6: .1, 700: 1}, {3: 1, 700: 1}, {2: 1}])
    questions = _unit([{0: 1, 600: 3}, {2: 1, 700: 3}])
    native = EmbeddingGemma2Adapter({"device": "cpu", "dtype": "float32"})
    adapters = {"N": native}
    if specialist:
        adapters["S"] = SpecialistAdapter("siglip2", {"device": "cpu", "dtype": "float32"})
    for channel, adapter in adapters.items():
        def encode(items, role, adapter=adapter, channel=channel):
            adapter.last_usage = {"items": [{"id": row["id"], "modalities": ["text" if role == "query" else "image"]} for row in items]}
            values = documents if role == "document" else questions
            if channel == "S":
                values = values.copy()
                values[:, 128:] = 0
                values /= np.linalg.norm(values, axis=1, keepdims=True)
            return values
        monkeypatch.setattr(adapter, "encode", encode)
        monkeypatch.setattr(adapter, "unload", lambda: None)
    report = run_matrix(dataset, run, adapters=adapters, requested_channels=list(adapters),
        candidate_grid=(50,), budget_modes=("per_channel", "total"), channel_budget=2, cache_dir=cache)
    if legacy:
        for path in cache.glob("vectors/*/*.usage.json"):
            path.unlink()
    # Any attempt by the sweep to run the native model fails, including cache misses.
    monkeypatch.setattr(EmbeddingGemma2Adapter, "encode", lambda *a, **k: pytest.fail("cache-only sweep called encode"))
    monkeypatch.setattr(EmbeddingGemma2Adapter, "_load", lambda *a, **k: pytest.fail("cache-only sweep loaded a model"))
    return dataset, run, cache, report, documents, questions


def _hashes(root):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*") if path.is_file()}


def test_full_corpus_prefix_search_reproduces_baseline_and_preserves_cache(tmp_path, monkeypatch):
    dataset, run, cache, base, documents, _ = _fixture(tmp_path, monkeypatch)
    before = _hashes(cache)
    result = run_dimension_sweep(dataset, run, tmp_path / "result", cache_dir=cache)
    assert result["status"] == "completed"
    assert result["provenance"]["specialist"]["status"] == "validated"
    assert len(result["cells"]) == 16  # four dimensions, N/N+S, both original budgets
    cells = { (cell["dimension"], tuple(cell["channels"]), cell["budget_mode"]): cell for cell in result["cells"] }
    native = cells[(768, ("N",), "per_channel")]
    assert native["metrics"] == next(row["metrics"] for row in base["cells"] if row["variant_id"] == "photo__n__none" and row["budget_mode"] == "per_channel")
    # Relevant c/f are absent from the original native top-2, but prefix search
    # finds them from the full corpus rather than reranking the saved top-2.
    small = cells[(128, ("N",), "per_channel")]
    assert small["metrics"]["ndcg@1"] == 1 > native["metrics"]["ndcg@1"]
    assert small["n_vector_storage_ratio"] == 1 / 6
    assert small["n_document_vector_bytes"] == len(documents) * 128 * 4
    assert result["model_inference_calls"] == 0 and result["inference_seconds"] is None
    algorithm = result["provenance"]["algorithm"]
    assert len(algorithm["sweep_source_sha256"]) == len(algorithm["shared_engine_source_sha256"]) == 64
    assert algorithm["numpy_version"] == np.__version__
    from rag_benchmark.models import stable_hash
    assert result["identity"] == stable_hash({"experiment": result["experiment"], "dataset": result["dataset"]["identity"],
        "configuration": result["configuration"], "provenance": result["provenance"]})
    assert _hashes(cache) == before
    serialized = (tmp_path / "result/dimension-sweep.json").read_text()
    assert "PRIVATE" not in serialized and str(tmp_path) not in serialized
    assert "query_ids" not in serialized and '"ranking"' not in serialized


def test_legacy_blocks_are_explicit_and_only_new_hashes_are_claimed(tmp_path, monkeypatch):
    dataset, run, cache, *_ = _fixture(tmp_path, monkeypatch, specialist=False, legacy=True)
    result = run_dimension_sweep(dataset, run, tmp_path / "result", cache_dir=cache, dimensions=(128,))
    assert result["status"] == "completed"
    role = result["provenance"]["document_vectors"]
    assert role["blocks_without_usage_metadata"] == role["blocks_without_explicit_row_ids"] == 1
    assert role["original_vector_checksums_available"] is False
    assert len(role["newly_observed_blocks"][0]["newly_observed_array_sha256"]) == 64
    assert result["provenance"]["specialist"]["status"] == "unavailable"
    assert all(row["channels"] == ["N"] for row in result["cells"])
    assert all(row["status"] == "reproduced" for row in result["provenance"]["baseline_768"])


@pytest.mark.parametrize("corruption,reason", [
    ("missing_block", "missing_document_vector_blocks"),
    ("swapped_vector_rows", "native_768_ranking_cache_mismatch"),
    ("metadata_order", "vector_metadata_id_order_mismatch"),
    ("aggregate_metadata_order", "aggregate_vector_metadata_id_order_mismatch"),
    ("query_cache_order", "ranking_cache_identity_or_query_order_mismatch"),
    ("native_mode", "base_requires_native_768_eg2_spec"),
    ("dataset", "base_frozen_dataset_mismatch"),
    ("database", "missing_base_result_database"),
    ("metric", "baseline_per_query_metric_mismatch"),
])
def test_invalid_required_evidence_is_unavailable_without_fallback(tmp_path, monkeypatch, corruption, reason):
    dataset, run, cache, report, *_ = _fixture(tmp_path, monkeypatch, specialist=False)
    block = next(path for path in cache.glob("vectors/*/*.npy") if np.load(path).shape[0] == 6)
    if corruption == "missing_block":
        block.unlink()
    elif corruption == "swapped_vector_rows":
        values = np.load(block)
        values[[0, 2]] = values[[2, 0]]
        np.save(block, values)
    elif corruption == "metadata_order":
        metadata_path = block.with_suffix(".usage.json")
        metadata = json.loads(metadata_path.read_text())
        metadata["items"] = list(reversed(metadata["items"]))
        metadata_path.write_text(json.dumps(metadata))
    elif corruption == "aggregate_metadata_order":
        metadata_path = block.parent / "diagnostics.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["items"].reverse()
        metadata_path.write_text(json.dumps(metadata))
    elif corruption == "query_cache_order":
        path = next(cache.glob("rankings/*/rankings.json"))
        data = json.loads(path.read_text())
        data["query_ids"].reverse()
        path.write_text(json.dumps(data))
    elif corruption == "native_mode":
        report["adapter_specs"]["channels"]["N"]["config"]["mode"] = "joint"
        (run / "report.json").write_text(json.dumps(report))
    elif corruption == "dataset":
        data = json.loads((dataset / "dataset.json").read_text())
        data["revision"] = "different-revision"
        (dataset / "dataset.json").write_text(json.dumps(data))
    elif corruption == "database":
        (run / "progress.sqlite3").unlink()
    elif corruption == "metric":
        with sqlite3.connect(run / "progress.sqlite3") as db:
            cell, qid, payload = db.execute("SELECT cell,query_id,payload FROM results LIMIT 1").fetchone()
            data = json.loads(payload)
            data["metrics"]["hit@5"] = .125
            db.execute("UPDATE results SET payload=? WHERE cell=? AND query_id=?", (json.dumps(data), cell, qid))
    result = run_dimension_sweep(dataset, run, tmp_path / "result", cache_dir=cache)
    assert result["status"] == "unavailable" and result["reason_code"] == reason
    assert result["cells"] == [] and result["model_inference_calls"] == 0


def test_missing_specialist_ranking_keeps_verified_standalone_sweep(tmp_path, monkeypatch):
    dataset, run, cache, report, *_ = _fixture(tmp_path, monkeypatch)
    for path in cache.glob("rankings/*/rankings.json"):
        if json.loads(path.read_text())["identity"] == report["channels"]["S"]["identity"]:
            path.unlink()
    result = run_dimension_sweep(dataset, run, tmp_path / "result", cache_dir=cache, dimensions=(512, 768))
    assert result["status"] == "completed"
    assert result["provenance"]["specialist"]["reason_code"] == "missing_s_ranking_cache"
    assert all(row["channels"] == ["N"] for row in result["cells"])


def test_projection_never_mutates_baseline_and_rejects_undefined_prefix():
    values = _unit([{0: 1, 600: 3}])
    unchanged = values.copy()
    assert project_prefix(values, 768) is values
    small = project_prefix(values, 128)
    assert np.linalg.norm(small[0]) == pytest.approx(1)
    np.testing.assert_array_equal(values, unchanged)
    assert not np.shares_memory(small, values)
    with pytest.raises(ValueError, match="zero_prefix_norm"):
        project_prefix(_unit([{600: 1}]), 128)
    with pytest.raises(ValueError, match="supported dimension"):
        project_prefix(values, 64)


def test_cli_uses_sweep_specific_filenames_and_reports_unavailability(tmp_path, monkeypatch):
    dataset, run, cache, *_ = _fixture(tmp_path, monkeypatch, specialist=False)
    output = tmp_path / "cli"
    command = ["--dataset", str(dataset), "--base-run", str(run), "--output", str(output),
               "--cache-dir", str(cache), "--dimensions", "256,768", "--no-specialist"]
    assert main(command) == 0
    assert (output / "dimension-sweep.json").is_file() and (output / "dimension-sweep.md").is_file()
    assert not (output / "report.json").exists()
    (run / "progress.sqlite3").unlink()
    assert main(command) == 2
    assert json.loads((output / "dimension-sweep.json").read_text())["cells"] == []

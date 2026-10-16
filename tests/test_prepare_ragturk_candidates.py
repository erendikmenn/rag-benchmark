"""Small CPU-only frozen matrices and injected lexical scores; no model/DB/cloud calls."""

import fcntl
import importlib.util
import json
import os
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT = Path(__file__).parents[1]


def module(path, name):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    sys.modules[name] = m
    s.loader.exec_module(m)
    return m


m = module(ROOT / "scripts/prepare_ragturk_candidates.py", "candidate_producer_test")
f = module(ROOT / "tests/test_ragturk_pgvector_store.py", "candidate_prepared_fixture")
bundle = f.bundle


class Lexical:
    def get_scores(self, tokens):
        return np.array([2.0, 1.0], dtype=np.float32)


@pytest.fixture
def prepared(bundle):
    return f.m.validate_bundle(bundle, counts=f.COUNTS)


def test_full_two_splits_source_only_and_exact_values(bundle, prepared, tmp_path):
    out = tmp_path / "candidate-output"
    result = m.run(bundle, out, prepared=prepared, bm25=Lexical(), block_rows=1, query_batch=1)
    assert (
        result["status"] == "completed"
        and result["queries"] == 2
        and result["dev_queries"] == result["test_queries"] == 1
    )
    rows = [m.read(p)["rows"][0] for p in sorted((out / "blocks").glob("*.json"))]
    assert [r["query_id"] for r in rows] == ["q1", "q2"] and [r["split"] for r in rows] == ["dev", "test"]
    assert rows[0]["methods"]["bge"]["candidate_ids"] == ["a", "b"]
    assert rows[1]["methods"]["bge"]["candidate_ids"] == ["b", "a"]
    assert rows[1]["methods"]["bge"]["candidate_scores"] == [1.0, 0.0]
    payload = json.dumps(rows)
    assert all(
        word not in payload
        for word in ("gold_ids", "answers", "reference", "article_id", "category", "fixture one")
    )
    manifest = m.read(out / "manifest.json")
    assert manifest["embedding_provenance"]["bge"]["device"] == "mps"
    assert manifest["new_model_calls"] == 0 and manifest["pilot"] is False
    assert all(set(r["methods"]) == set(m.METHODS) for r in rows)


def test_stable_corpus_ties_positive_bm25_and_exact_rrf():
    ids = [str(i) for i in range(120)]
    ranks = m.rank(np.ones(120, dtype=np.float32), ids)
    assert [r["id"] for r in ranks] == ids[:100]
    assert m.rank(np.zeros(120, dtype=np.float32), ids, positive=True) == []
    _, rrf, _ = m.pure_retrieval()
    a = m.rank(np.array([1.0, 0.0], dtype=np.float32), ["a", "b"])
    b = m.rank(np.array([0.0, 1.0], dtype=np.float32), ["a", "b"])
    fused = rrf([a, b], k=60, top_k=50)
    assert [r["id"] for r in fused] == ["a", "b"] and fused[0]["score"] == pytest.approx(1 / 61 + 1 / 62)


def test_dense_batch_matches_fp32_reference(prepared):
    rows = list(m.produce(prepared, [0, 1], Lexical(), query_batch=2))
    for i, row in enumerate(rows):
        for family in ["bge", "embeddinggemma"]:
            scores = prepared.vectors[family]["documents"] @ prepared.vectors[family]["queries"][i]
            assert row["methods"][family]["candidate_scores"] == sorted(map(float, scores), reverse=True)


def test_explicit_pilot_and_test_selection(bundle, prepared, tmp_path):
    result = m.run(bundle, tmp_path / "pilot", prepared=prepared, bm25=Lexical(), split="test", limit=1)
    assert result["dev_queries"] == 0 and result["test_queries"] == 1
    manifest = m.read(tmp_path / "pilot/manifest.json")
    assert manifest["pilot"] is True and manifest["limit"] == 1
    with pytest.raises(ValueError):
        m.select(prepared, "all", 0)


def test_interruption_durable_resume_and_changed_identity_rejected(bundle, prepared, tmp_path):
    class Interrupted:
        def __init__(self):
            self.n = 0

        def get_scores(self, tokens):
            self.n += 1
            if self.n == 2:
                raise KeyboardInterrupt()
            return Lexical().get_scores(tokens)

    out = tmp_path / "interrupted"
    with pytest.raises(KeyboardInterrupt):
        m.run(bundle, out, prepared=prepared, bm25=Interrupted(), block_rows=1)
    assert m.read(out / "state.json")["status"] == "interrupted"
    first = out / "blocks/000000000-000000001.json"
    before = first.read_bytes()
    m.run(bundle, out, prepared=prepared, bm25=Lexical(), block_rows=1, resume=True)
    assert first.read_bytes() == before
    with pytest.raises(ValueError, match="resume_identity_changed"):
        m.run(bundle, out, prepared=prepared, bm25=Lexical(), block_rows=2, resume=True)


@pytest.mark.parametrize("defect", ["block", "missing", "input", "summary"])
def test_corruption_is_not_recomputed_or_silently_accepted(bundle, prepared, tmp_path, defect):
    out = tmp_path / "corrupt"
    m.run(bundle, out, prepared=prepared, bm25=Lexical(), block_rows=1)
    path = out / "blocks/000000000-000000001.json"
    if defect == "block":
        value = m.read(path)
        value["rows"][0]["methods"]["bge"]["candidate_ids"].reverse()
        path.write_text(json.dumps(value))
    elif defect == "missing":
        path.unlink()
    elif defect == "input":
        prepared.files["corpus.jsonl"].write_text("mutated")
    else:
        value = m.read(out / "summary.json")
        value["queries"] = 999
        m.write(out / "summary.json", value, replace=True)
    with pytest.raises(ValueError):
        m.run(bundle, out, prepared=prepared, bm25=Lexical(), block_rows=1, resume=True)


def test_one_writer_and_no_implicit_resume(bundle, prepared, tmp_path):
    out = tmp_path / "locked"
    m.run(bundle, out, prepared=prepared, bm25=Lexical())
    with pytest.raises(ValueError, match="explicit_resume_required"):
        m.run(bundle, out, prepared=prepared, bm25=Lexical())
    fd = os.open(out / ".lock", os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            m.run(bundle, out, prepared=prepared, bm25=Lexical(), resume=True)
    finally:
        os.close(fd)


def test_cli_validator_never_uses_fixture_counts(bundle):
    with pytest.raises(ValueError, match="dataset_counts"):
        m.load_prepared(bundle)


def test_no_adapter_constructed(prepared, monkeypatch):
    import rag_benchmark.models

    def forbidden(*a, **kw):
        raise AssertionError("no model construction")

    monkeypatch.setattr(rag_benchmark.models.DenseEmbedder, "__init__", forbidden)
    monkeypatch.setattr(rag_benchmark.models.LayaReranker, "__init__", forbidden)
    assert len(list(m.produce(prepared, [0, 1], Lexical(), query_batch=2))) == 2


def test_native_bm25_loads_exact_fixture_files(prepared):
    lexical = m.load_bm25(prepared)
    assert np.asarray(lexical.get_scores(["fixture"])).tolist() == [1.0, 0.0]
    assert np.asarray(lexical.get_scores(["second"])).tolist() == [0.0, 2.0]


def test_nonfinite_scores_rejected():
    with pytest.raises(ValueError, match="invalid_scores"):
        m.rank(np.array([np.nan], dtype=np.float32), ["a"])

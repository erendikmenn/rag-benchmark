import json

import pytest

from rag_benchmark import runner
from rag_benchmark.storage import Store


def fixture_run(tmp_path, monkeypatch):
    corpus = [{"id": "a#c1", "article_id": "a", "title": "Türkiye", "text": "Ankara başkenttir."}]
    questions = [{"id": f"q{i}", "article_id": f"a{i}", "question": f"Türkiye başkent sorusu {i}?", "gold_ids": ["a#c1"], "answers": ["Ankara"], "source": "web", "category": "FACTUAL"} for i in range(2)]
    data = tmp_path / "data"
    data.mkdir()
    (data / "corpus.jsonl").write_text(json.dumps(corpus))
    (data / "questions.dev.jsonl").write_text(json.dumps(questions))
    monkeypatch.setattr("rag_benchmark.data.load_dataset", lambda *a, **kw: (corpus, questions, {"test": "fixture"}))
    config = {"dataset": {"path": str(data), "seed": 42}, "experiment": {"cache_dir": str(tmp_path / "cache"), "candidates": 50, "context_k": 5, "variants": ["bm25"]}, "generator": {"model": "fixture"}}
    calls = []

    class Generator:
        last_usage = {"completion_tokens": 4, "finish_reason": "length"}
        def __init__(self, config):
            pass
        def preflight(self):
            return {"runtime": "fixture"}
        def generate(self, question, contexts):
            # The inference boundary must not contain gold IDs/reference answers.
            assert set(contexts[0]) <= {"id", "article_id", "title", "text", "score", "rank"}
            calls.append(question)
            return "Ankara. [a#c1]"
    monkeypatch.setattr("rag_benchmark.models.LocalGenerator", Generator)
    return config, calls


def test_resume_and_cache_preserve_measurement_boundaries(tmp_path, monkeypatch):
    config, calls = fixture_run(tmp_path, monkeypatch)
    directory = tmp_path / "run"
    result = runner.run(config, directory)
    assert len(calls) == 2
    assert result["variants"]["bm25"]["metrics"]["answer_em"] == 1
    runner.run(config, directory)
    assert len(calls) == 2
    runtime = runner.runtime_identity()
    monkeypatch.setattr(runner, "runtime_identity", lambda: {**runtime, "git_revision": "documentation-only-change"})
    runner.run(config, directory)
    assert len(calls) == 2
    result = runner.run(config, tmp_path / "second-run")
    assert len(calls) == 2
    assert result["variants"]["bm25"]["generation_cache_hits"] == 2
    assert result["variants"]["bm25"]["generation_s"] is None
    assert result["variants"]["bm25"]["length_limited_answers"] == 2
    config["experiment"]["context_k"] = 3
    with pytest.raises(ValueError, match="identity changed"):
        runner.run(config, directory)


def test_failed_generation_stays_incomplete_and_can_resume(tmp_path, monkeypatch):
    config, _ = fixture_run(tmp_path, monkeypatch)
    good_generate = __import__("rag_benchmark.models", fromlist=["LocalGenerator"]).LocalGenerator.generate
    attempts = []
    def flaky(self, question, contexts):
        attempts.append(question)
        if len(attempts) == 2:
            raise RuntimeError("server unavailable")
        return good_generate(self, question, contexts)
    monkeypatch.setattr("rag_benchmark.models.LocalGenerator.generate", flaky)
    directory = tmp_path / "run"
    with pytest.raises(RuntimeError, match="unavailable"):
        runner.run(config, directory)
    partial = runner.report(directory)
    assert partial["variants"]["bm25"]["n"] == 1
    assert partial["variants"]["bm25"]["complete"] is False
    result = runner.run(config, directory)
    assert len(attempts) == 3
    assert result["variants"]["bm25"]["complete"] is True
    assert not (directory / "error.json").exists()
    store = Store(directory / "results.sqlite")
    rows = store.rows()
    store.close()
    assert rows[0]["answer"] == "Ankara. [a#c1]"
    assert rows[0]["answer_for_scoring"] == "Ankara. "


def test_retrieval_only_never_initializes_generator(tmp_path, monkeypatch):
    config, calls = fixture_run(tmp_path, monkeypatch)
    def forbidden(*args, **kwargs):
        raise AssertionError("Generator must not load")
    monkeypatch.setattr("rag_benchmark.models.LocalGenerator", forbidden)
    summary = runner.run(config, tmp_path / "retrieval", retrieval_only=True)
    assert calls == []
    assert "answer_em" not in summary["variants"]["bm25"]["metrics"]


def test_records_retrieval_and_laya_scores_in_their_own_orders(tmp_path, monkeypatch):
    config, _ = fixture_run(tmp_path, monkeypatch)
    config["laya"] = {}
    config["experiment"]["variants"] = ["bm25", "bm25_laya"]

    class Index:
        last_usage = {}
        def __init__(self, *args, **kwargs):
            pass
        def build(self, method):
            pass
        def search(self, *args, **kwargs):
            return [{"id": "a#c1", "text": "Ankara", "score": 12.5}, {"id": "a#c2", "text": "Other", "score": 3.25}]

    class Reranker:
        last_usage = {}
        def __init__(self, config):
            pass
        def rerank(self, question, candidates):
            # Real adapter returns new dictionaries; this deliberately mutates
            # originals too, proving retrieval scores were captured beforehand.
            candidates[0]["score"], candidates[1]["score"] = .1, .9
            return list(reversed(candidates))

    monkeypatch.setattr("rag_benchmark.retrieval.RetrievalIndex", Index)
    monkeypatch.setattr("rag_benchmark.models.LayaReranker", Reranker)
    directory = tmp_path / "scored"
    runner.run(config, directory, retrieval_only=True)
    rows = [json.loads(line) for line in (directory / "predictions.jsonl").read_text().splitlines()]
    for row in rows:
        assert row["candidate_ids"] == ["a#c1", "a#c2"]
        assert row["candidate_scores"] == [12.5, 3.25]
        assert row["candidate_score_type"] == "bm25"
        if row["variant"] == "bm25_laya":
            assert row["ranked_ids"] == ["a#c2", "a#c1"]
            assert row["ranked_scores"] == [.9, .1]
            assert row["ranked_score_type"] == "laya_p_true"
        else:
            assert row["ranked_scores"] == row["candidate_scores"]
            assert row["ranked_score_type"] == "bm25"


@pytest.mark.parametrize("value", ["private text", float("nan"), float("inf"), True])
def test_record_scores_rejects_non_numeric_or_nonfinite_values(value):
    with pytest.raises(ValueError, match="finite numbers"):
        runner._record_scores([{"score": value}])

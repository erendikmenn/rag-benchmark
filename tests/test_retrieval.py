
import numpy as np
import pytest

from rag_benchmark.retrieval import RetrievalIndex, reciprocal_rank_fusion, turkish_tokens


CORPUS = [
    {"id": "ankara", "title": "Türkiye", "text": "Türkiye'nin başkenti Ankara'dır."},
    {"id": "kedi", "text": "Kediler süt içebilir."},
    {"id": "izmir", "text": "İzmir Ege kıyısında bir şehirdir."},
]


def test_turkish_dotted_and_dotless_i():
    assert turkish_tokens("IĞDIR İZMİR ışık i") == ["ığdır", "izmir", "ışık", "i"]


def test_bm25_real_index_roundtrip(tmp_path):
    index = RetrievalIndex(CORPUS, tmp_path)
    first = index.search("Türkiye başkenti Ankara", "bm25", 20)
    assert first[0]["id"] == "ankara"
    assert first[0]["score"] > 0
    loaded = RetrievalIndex(CORPUS, tmp_path).search("Türkiye başkenti Ankara", "bm25", 20)
    assert loaded == first
    assert index.search("varolmayanxyz", "bm25") == []


def test_rrf_rewards_agreement_and_ignores_duplicate_candidate():
    result = reciprocal_rank_fusion([
        [{"id": "a", "text": "a"}, {"id": "b", "text": "b"}, {"id": "a", "text": "a"}],
        [{"id": "b", "text": "b"}, {"id": "c", "text": "c"}],
    ])
    assert [x["id"] for x in result] == ["b", "a", "c"]
    assert result[1]["score"] == pytest.approx(1 / 61)
    assert result[0]["score"] == pytest.approx(1 / 61 + 1 / 62)


def test_corpus_identity_tracks_text_title_and_order(tmp_path):
    original = RetrievalIndex(CORPUS, tmp_path).identity_for("bm25")
    changed = [{**CORPUS[0], "title": "Changed"}, *CORPUS[1:]]
    assert original != RetrievalIndex(changed, tmp_path).identity_for("bm25")
    assert original != RetrievalIndex(list(reversed(CORPUS)), tmp_path).identity_for("bm25")


def test_duplicate_ids_fail(tmp_path):
    with pytest.raises(ValueError, match="Duplicate"):
        RetrievalIndex([CORPUS[0], CORPUS[0]], tmp_path)


class StubEmbedder:
    """Known numerical fixture; not a model-quality benchmark."""
    dimension = 2
    identity = "fixture-v1"

    def __init__(self):
        self.doc_calls = 0
        self.query_calls = 0

    def embed_documents(self, corpus):
        self.doc_calls += 1
        return np.array([[1, 0], [0, 1], [-1, 0]], dtype=np.float32)

    def embed_query(self, query):
        self.query_calls += 1
        return np.array([1, 0], dtype=np.float32)


def test_dense_cache_and_latency_bypass(tmp_path):
    index = RetrievalIndex(CORPUS, tmp_path)
    fixture = StubEmbedder()
    index._embedders["bge"] = fixture
    assert index.search("q", "bge")[0]["id"] == "ankara"
    index.search("q", "bge")
    assert fixture.doc_calls == 1 and fixture.query_calls == 1
    assert index.last_usage["query_cache_hits"]["bge"] is True
    index.search("q", "bge", cache_query=False)
    assert fixture.query_calls == 2
    assert index.last_usage["query_cache_hits"]["bge"] is False
    second = RetrievalIndex(CORPUS, tmp_path)
    other = StubEmbedder()
    second._embedders["bge"] = other
    second.search("q", "bge")
    assert other.doc_calls == other.query_calls == 0


def test_invalid_dense_cache_is_rejected(tmp_path):
    index = RetrievalIndex(CORPUS, tmp_path)
    index._embedders["bge"] = StubEmbedder()
    index.search("q", "bge")
    query_path = next((tmp_path / "queries").rglob("*.npy"))
    np.save(query_path, np.array([float("nan"), 0]))
    with pytest.raises(RuntimeError, match="query embedding"):
        index.search("q", "bge")


def test_dense_build_resumes_completed_blocks(tmp_path):
    corpus = [{"id": str(i), "text": f"passage {i}"} for i in range(257)]

    class InterruptedEmbedder(StubEmbedder):
        def embed_documents(self, docs):
            self.doc_calls += 1
            if self.doc_calls == 2:
                raise RuntimeError("interrupted")
            return np.tile(np.array([[1, 0]], dtype=np.float32), (len(docs), 1))

    first = RetrievalIndex(corpus, tmp_path)
    first._embedders["bge"] = InterruptedEmbedder()
    with pytest.raises(RuntimeError, match="interrupted"):
        first.build("bge")
    assert not list(tmp_path.rglob("documents.npy"))
    resumed = RetrievalIndex(corpus, tmp_path)
    embedder = InterruptedEmbedder()
    resumed._embedders["bge"] = embedder
    resumed.build("bge")
    assert embedder.doc_calls == 1
    assert resumed._vectors["bge"].shape == (257, 2)

import copy
import json

import pytest

from rag_benchmark.multimodal_code import SharedCodeSegmenter
from rag_benchmark.multimodal_code_rerank import SharedCodeReranker


class ByteTokenizer:
    def __call__(self, text, **kwargs):
        assert kwargs["truncation"] is False
        return {"input_ids": [1] * (len(text.encode("utf-8")) + 2)}


def segmenter(cap=8):
    value = SharedCodeSegmenter(max_tokens=64, max_chunk_chars=cap)
    value.tokenizers = {"bge": ByteTokenizer(), "embeddinggemma": ByteTokenizer()}
    return value


class FakeBase:
    pointwise = True

    def __init__(self, identity="fake-test-base-v1"):
        self.identity, self.calls, self.last_usage = identity, [], {}
        self.unloaded = False

    def score(self, query, candidates):
        self.calls.append((query, copy.deepcopy(candidates)))
        self.last_usage = {"pairs": len(candidates)}
        return [-1.0 if "A" in row["text"] else -7.0 for row in candidates]

    def unload(self):
        self.unloaded = True


def test_max_scores_preserve_negative_values_order_complete_query_and_every_source_byte():
    common, base = segmenter(), FakeBase()
    wrapper = SharedCodeReranker(base, common)
    query = {"id": "query", "text": "complete query " * 40, "media": {}}
    long_source = "B\r\n" * 30 + "Aç🐍\r\n" + "B\r\n" * 20
    candidates = [{"id": "b", "text": "B", "media": {}},
                  {"id": "a", "text": long_source, "media": {}},
                  {"id": "c", "text": "A", "media": {}}]
    untouched = copy.deepcopy(candidates)
    expected_chunks, _ = common.partition(candidates[1])
    assert wrapper.score(query, candidates) == [-7.0, -1.0, -1.0]
    assert candidates == untouched
    seen_query, seen = base.calls[0]
    assert seen_query is query
    assert seen[0] == candidates[0] and seen[-1] == candidates[-1]
    assert [part["text"] for part in seen[1:-1]] == [part["text"] for part in expected_chunks]
    assert b"".join(part["text"].encode() for part in seen[1:-1]) == long_source.encode()
    assert len({part["id"] for part in seen}) == len(seen)
    usage = wrapper.last_usage
    assert usage["original_function_count"] == 3
    assert usage["chunk_count"] == len(expected_chunks) + 2
    assert usage["segmented_function_count"] == 1
    assert usage["native_short_function_count"] == 2
    assert usage["source_utf8_bytes"] == usage["covered_source_utf8_bytes"]
    assert usage["source_characters"] == usage["covered_source_characters"]
    assert usage["base_usage"] == {"pairs": len(seen)}
    assert wrapper.score(query, [candidates[1]]) == [-1.0]  # independent of candidate pool
    wrapper.unload()
    assert base.unloaded


@pytest.mark.parametrize("invalid", [[], [float("nan")], [float("inf")], [True], ["0.1"], [0.2, 0.3]])
def test_no_missing_nonfinite_or_nonreal_chunk_scores_are_accepted(invalid):
    base = FakeBase()
    base.score = lambda query, rows: invalid
    wrapper = SharedCodeReranker(base, segmenter())
    with pytest.raises(RuntimeError, match="one finite real score"):
        wrapper.score({"text": "query"}, [{"id": "a", "text": "source"}])
    assert wrapper.last_usage == {}


def test_native_context_errors_propagate_without_clipping_or_retry():
    base, received = FakeBase(), []

    def fail(query, candidates):
        received.append((query, candidates))
        raise ValueError("pair context exceeds native model capacity")

    base.score = fail
    wrapper = SharedCodeReranker(base, segmenter())
    query, source = {"text": "long query " * 200}, "A" * 150
    with pytest.raises(ValueError, match="pair context exceeds"):
        wrapper.score(query, [{"id": "function", "text": source}])
    assert len(received) == 1
    assert received[0][0] is query
    assert "".join(part["text"] for part in received[0][1]) == source
    assert wrapper.last_usage == {}


def test_reranker_only_character_chunks_avoid_serialized_pair_overflow_and_preserve_max():
    class SerializedPairBase(FakeBase):
        def score(self, query, candidates):
            if any(len(json.dumps({"query": query["text"], "passage": row["text"]})) > 120 for row in candidates):
                raise ValueError("native serialized query-passage context overflow")
            return super().score(query, candidates)

    base = SerializedPairBase()
    original = SharedCodeSegmenter(max_tokens=512, max_chunk_chars=32)
    bounded = SharedCodeSegmenter(max_tokens=512, max_chunk_chars=32, enforce_character_limit=True)
    for value in (original, bounded):
        value.tokenizers = {"bge": ByteTokenizer(), "embeddinggemma": ByteTokenizer()}
    candidate = {"id": "original-function", "text": "    \tB\r\n" * 20 + "A", "media": {}}
    query = {"id": "q", "text": "complete query"}
    assert len(original.partition(candidate)[0]) == 1
    with pytest.raises(ValueError, match="serialized"):
        SharedCodeReranker(base, original).score(query, [candidate])
    reranker = SharedCodeReranker(base, bounded)
    assert reranker.score(query, [candidate]) == [-1.0]
    passed_query, passed_chunks = base.calls[0]
    assert passed_query is query
    assert "".join(row["text"] for row in passed_chunks) == candidate["text"]
    assert all(len(row["text"]) <= 32 for row in passed_chunks)
    assert reranker.last_usage["original_function_count"] == 1
    assert reranker.last_usage["source_utf8_bytes"] == reranker.last_usage["covered_source_utf8_bytes"]
    assert "character_bounded" in reranker.protocol["condition"]
    assert reranker.identity != SharedCodeReranker(base, original).identity


def test_identity_seals_base_and_shared_segmentation_and_rejects_listwise():
    common, base = segmenter(), FakeBase()
    wrapper = SharedCodeReranker(base, common)
    assert wrapper.identity != SharedCodeReranker(FakeBase("changed-model"), common).identity
    assert wrapper.identity != SharedCodeReranker(base, segmenter(cap=4)).identity
    base.identity = "preflight-changed-server"
    with pytest.raises(ValueError, match="finalize base preflight first"):
        wrapper.score({"text": "q"}, [{"id": "a", "text": "source"}])
    base.identity = "fake-test-base-v1"
    wrapper.protocol["candidate_policy"] = "drop"
    with pytest.raises(ValueError, match="configuration changed"):
        wrapper.score({"text": "q"}, [])
    base.pointwise = False
    with pytest.raises(ValueError, match="pointwise"):
        SharedCodeReranker(base, common)


def test_identity_change_during_inference_is_not_accepted():
    base = FakeBase()
    wrapper = SharedCodeReranker(base, segmenter())

    def change_identity(query, candidates):
        base.identity = "different-serving-model"
        return [0.5] * len(candidates)

    base.score = change_identity
    with pytest.raises(ValueError, match="configuration changed"):
        wrapper.score({"text": "q"}, [{"id": "a", "text": "source"}])
    assert wrapper.last_usage == {}


def test_empty_pool_skips_model_and_invalid_function_inputs_fail():
    base = FakeBase()
    wrapper = SharedCodeReranker(base, segmenter())
    assert wrapper.score({"text": "query"}, []) == []
    assert not base.calls
    assert wrapper.last_usage["chunk_count"] == 0
    with pytest.raises(ValueError, match="unique nonempty"):
        wrapper.score({"text": "q"}, [{"id": "a", "text": "x"}, {"id": "a", "text": "y"}])
    with pytest.raises(ValueError, match="source-text"):
        wrapper.score({"text": "q"}, [{"id": "a"}])
    with pytest.raises(ValueError, match="without media"):
        wrapper.score({"text": "q", "media": {"image": "unused"}}, [])

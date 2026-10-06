from types import SimpleNamespace

import numpy as np
import pytest

from rag_benchmark.multimodal_code import SegmentedCodeAdapter, SharedCodeSegmenter


class ByteTokenizer:
    def __init__(self, multiplier=1):
        self.multiplier = multiplier

    def __call__(self, text, **kwargs):
        assert kwargs["truncation"] is False
        return {"input_ids": [1] * (len(text.encode("utf-8")) * self.multiplier + 2)}


def segmenter(max_tokens=64, cap=20):
    value = SharedCodeSegmenter(max_tokens=max_tokens, max_chunk_chars=cap)
    value.tokenizers = {"bge": ByteTokenizer(), "embeddinggemma": ByteTokenizer()}
    return value


def test_native_function_is_unchanged_when_it_fits_both_models():
    model = segmenter(max_tokens=512, cap=8)
    source = "function native() { return 'a rather long literal'; }\r\n"
    chunks, summary = model.partition({"id": "native", "text": source})
    assert [chunk["text"] for chunk in chunks] == [source]
    assert summary["segmented"] is False
    assert summary["source_utf8_bytes"] == len(source.encode())


def test_shared_chunks_preserve_unicode_crlf_and_all_source_bytes():
    model = segmenter()
    source = "function şey() {\r\n" + "  // 🐍 ç İ byte coverage\n" * 12 + "}\n"
    chunks, summary = model.partition({"id": "same-function", "text": source})
    assert len(chunks) > 1
    assert b"".join(chunk["text"].encode() for chunk in chunks) == source.encode()
    assert [chunk["byte_start"] for chunk in chunks[1:]] == [chunk["byte_end"] for chunk in chunks[:-1]]
    assert [chunk["char_start"] for chunk in chunks[1:]] == [chunk["char_end"] for chunk in chunks[:-1]]
    for chunk in chunks:
        assert max(chunk["token_counts"].values()) <= 64
        assert source[chunk["char_start"]:chunk["char_end"]] == chunk["text"]
    assert chunks[-1]["byte_end"] == summary["source_utf8_bytes"]
    second, second_summary = model.partition({"id": "same-function", "text": source,
        "metadata": {"gold_docstring": "NEVER USE THIS"}})
    assert second == chunks
    assert second_summary == summary


def test_long_lines_fall_back_to_character_splits_without_loss():
    model = segmenter(cap=4096)
    source = "🐍" * 100
    chunks, _ = model.partition({"id": "long-line", "text": source})
    assert "".join(chunk["text"] for chunk in chunks) == source
    assert len(chunks) > 1
    assert all(max(chunk["token_counts"].values()) <= 64 for chunk in chunks)


def test_line_boundaries_preferred_and_both_native_prefixes_counted():
    model = segmenter(max_tokens=64, cap=16)
    source = "line-one\nline-two\nline-three\n" * 3
    chunks, _ = model.partition({"id": "function", "text": source})
    assert all(chunk["text"].endswith("\n") for chunk in chunks)
    assert all(chunk["token_counts"]["embeddinggemma"] > chunk["token_counts"]["bge"] for chunk in chunks)
    # A function that fits BGE but not EG2 is segmented identically for both.
    borderline = "a" * 50
    pieces, summary = model.partition({"id": "borderline", "text": borderline})
    assert summary["whole_function_token_counts"]["bge"] < 64
    assert summary["whole_function_token_counts"]["embeddinggemma"] > 64
    assert len(pieces) > 1


def test_unrepresentable_title_and_mutated_protocol_rejected():
    model = segmenter()
    with pytest.raises(ValueError, match="one source character"):
        model.partition({"id": "bad-title", "text": "source", "title": "title" * 100})
    model.max_chunk_chars = 1
    with pytest.raises(ValueError, match="protocol changed"):
        model.partition({"id": "x", "text": "x"})
    with pytest.raises(ValueError, match="budgets must match"):
        SharedCodeSegmenter(bge_config={"max_length": 4096})


def test_function_max_cosine_deduplicates_chunks_and_preserves_original_ids(monkeypatch, tmp_path):
    common = segmenter(max_tokens=32, cap=8)
    adapter = SegmentedCodeAdapter("bge", {"device": "cpu", "max_length": 32}, segmenter=common)
    seen = []

    def embed_documents(documents):
        seen.extend(documents)
        # Function a repeats the best token many times. It must still have one
        # result and max score 1, not a sum that rewards its chunk count.
        return np.asarray([[1, 0] if "A" in item["text"] else [-1, 0] for item in documents], dtype=np.float32)

    fake = SimpleNamespace(_model=None, config=adapter.encoder.config, identity="unit-test",
        format_query=lambda text: text, format_document=lambda row: row["text"],
        embed_documents=embed_documents, _encode=lambda texts: np.tile(np.array([[1, 0]], np.float32), (len(texts), 1)))
    adapter.encoder = fake
    adapter.dimension = 2
    documents = [{"id": "b", "text": "B"}, {"id": "a", "text": "A" * 50}, {"id": "c", "text": "A"}]
    queries = [{"id": "q", "text": "query"}]
    ranking, usage = adapter.rank(documents, queries, 3, tmp_path)
    assert ranking == [[{"id": "a", "score": 1.0}, {"id": "c", "score": 1.0}, {"id": "b", "score": -1.0}]]
    assert usage["original_function_count"] == 3
    assert usage["segmented_function_count"] == 1
    assert usage["chunk_count"] > 3
    assert usage["source_characters"] == usage["covered_source_characters"] == 52
    first_count = len(seen)
    second, resumed = adapter.rank(documents, queries, 1, tmp_path, exclusions=[{"a"}])
    assert second == [[{"id": "c", "score": 1.0}]]
    assert len(seen) == first_count
    assert resumed["document_encoding"]["inference_seconds"] is None
    # Cache metadata contains offsets/hashes, never source code or gold labels.
    manifest = (tmp_path / "shared-segmentation.json").read_text()
    assert "A" * 50 not in manifest


def test_query_cache_role_and_revision_are_not_conflated(tmp_path):
    common = segmenter()
    bge = SegmentedCodeAdapter("bge", {"device": "cpu", "max_length": 64}, segmenter=common)
    eg2 = SegmentedCodeAdapter("embeddinggemma", {"device": "cpu", "max_length": 64}, segmenter=common)
    assert bge.segmenter.identity == eg2.segmenter.identity
    assert bge.identity != eg2.identity
    with pytest.raises(ValueError, match="revision differs"):
        SegmentedCodeAdapter("bge", {"revision": "a" * 40, "max_length": 64}, segmenter=common)

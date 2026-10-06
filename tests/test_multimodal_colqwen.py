import json

import numpy as np
import pytest

from rag_benchmark.multimodal_colqwen import COLQWEN_DEFAULTS, ColQwenAdapter, local_overlay, validate_tokens


def unit(coordinates):
    result = np.zeros((len(coordinates), 128), dtype=np.float32)
    for row, (index, value) in enumerate(coordinates):
        result[row, index] = value
    return result


def test_exact_maxsim_includes_negative_maxima_and_all_query_tokens():
    pytest.importorskip("torch")
    pytest.importorskip("sentence_transformers")
    from rag_benchmark.multimodal_colqwen import exact_maxsim
    query = unit([(0, 1), (1, 1)])
    positive = unit([(0, 1), (1, 1)])
    negative = -np.ones((1, 128), dtype=np.float32) / np.sqrt(128)
    scores = exact_maxsim([query], [positive, negative], device="cpu", chunk_elements=1)
    expected = [np.sum(np.max(query @ doc.T, axis=1)) for doc in [positive, negative]]
    assert np.allclose(scores[0], expected)
    assert scores[0, 0] == 2
    assert scores[0, 1] < 0


def test_pinned_base_overlay_preserves_source_and_weight_links(tmp_path):
    adapter, base = tmp_path / "adapter", tmp_path / "base"
    adapter.mkdir()
    base.mkdir()
    original = {"base_model_name_or_path": COLQWEN_DEFAULTS["base_model_id"], "revision": None}
    (adapter / "adapter_config.json").write_text(json.dumps(original))
    (adapter / "adapter_model.safetensors").write_bytes(b"unit-test-weights")
    (base / "config.json").write_text('{}')
    config = {**COLQWEN_DEFAULTS, "cache_dir": str(tmp_path / "cache")}
    path = local_overlay(adapter, base, config)
    assert json.loads((adapter / "adapter_config.json").read_text()) == original
    modified = json.loads((path / "adapter_config.json").read_text())
    assert modified["base_model_name_or_path"] == str(base.resolve())
    assert modified["revision"] == config["base_revision"]
    assert (path / "adapter_model.safetensors").resolve() == (adapter / "adapter_model.safetensors").resolve()
    assert local_overlay(adapter, base, config) == path
    modified["revision"] = "bad"
    (path / "adapter_config.json").write_text(json.dumps(modified))
    with pytest.raises(ValueError, match="different base"):
        local_overlay(adapter, base, config)


def test_mutable_base_revision_and_alternate_scoring_rejected():
    with pytest.raises(ValueError, match="immutable"):
        ColQwenAdapter({"base_revision": "main"})
    with pytest.raises(ValueError, match="scoring is fixed"):
        ColQwenAdapter({"scoring": "mean_pool"})
    with pytest.raises(ValueError, match="offline"):
        ColQwenAdapter({"local_files_only": False})


@pytest.mark.parametrize("array", [np.zeros((1, 128)), np.ones((0, 128)), np.ones((2, 768)), np.full((1, 128), np.nan)])
def test_invalid_token_vectors_are_rejected(array):
    with pytest.raises(RuntimeError):
        validate_tokens(array)


def test_rank_excludes_before_top_k_with_stable_ties_and_resumes(monkeypatch, tmp_path):
    model = ColQwenAdapter({"device": "cpu", "score_device": "cpu"})
    calls = []

    def encode(items, role):
        calls.append((items[0]["id"], role))
        model.last_usage = {"inference_seconds": .01}
        return [unit([(0, 1)])]

    monkeypatch.setattr(model, "encode_tokens", encode)
    monkeypatch.setattr("rag_benchmark.multimodal_colqwen.exact_maxsim", lambda qs, ds, **kw: np.ones((len(qs), len(ds)), dtype=np.float32))
    documents = [{"id": "c"}, {"id": "b"}, {"id": "a"}]
    queries = [{"id": "q", "text": "question"}]
    ranking, usage = model.rank(documents, queries, 1, tmp_path, [{"a"}])
    assert ranking == [[{"id": "b", "score": 1.0}]]
    assert len(calls) == 4
    assert usage["document_encoding"]["new_items"] == 3
    _, second = model.rank(documents, queries, 1, tmp_path, [{"a"}])
    assert len(calls) == 4
    assert second["document_encoding"]["inference_seconds"] is None
    # A changed query must not reuse the preceding query's token cache.
    model.rank(documents, [{"id": "q", "text": "changed"}], 1, tmp_path, [{"a"}])
    assert len(calls) == 5


def test_media_bytes_participate_in_token_cache_key(tmp_path):
    path = tmp_path / "page.png"
    path.write_bytes(b"first-image")
    model = ColQwenAdapter({"device": "cpu", "score_device": "cpu"})
    item = {"id": "d", "media": {"image": str(path)}}
    first = model._cache_key(item, "document")
    path.write_bytes(b"second-image")
    assert model._cache_key(item, "document") != first

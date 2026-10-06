"""Contract/failure tests; model accuracy is measured only by labelled real datasets."""
from types import SimpleNamespace

import numpy as np
import pytest

from rag_benchmark.multimodal_models import (
    EmbeddingGemma2Adapter, SpecialistAdapter, _local_media, normalized,
)


@pytest.mark.parametrize("config,reason", [
    ({"revision": "main"}, "immutable"),
    ({"dtype": "float16"}, "float16"),
    ({"local_files_only": False}, "offline"),
    ({"vision_budget": 69}, "budgets"),
    ({"video_fps": 0}, "positive"),
    ({"dimension": 1024}, "dimensions"),
    ({"max_length": 8193}, "8192"),
    ({"model_id": "another/checkpoint"}, "substituting"),
])
def test_invalid_conditions_rejected_before_load(config, reason):
    with pytest.raises(ValueError, match=reason):
        EmbeddingGemma2Adapter(config)


def test_identity_covers_prompts_precision_and_vision_budget():
    baseline = EmbeddingGemma2Adapter().identity
    for override in [{"vision_budget": 70}, {"dtype": "float32"}, {"query_prompt": "other: "}, {"video_max_frames": 8}]:
        assert EmbeddingGemma2Adapter(override).identity != baseline
    assert EmbeddingGemma2Adapter(mode="joint").identity != baseline
    model = EmbeddingGemma2Adapter()
    model.config["vision_budget"] = 70
    with pytest.raises(ValueError, match="changed"):
        model._load()


def test_native_joint_and_composed_query_do_not_conflate_text(tmp_path):
    image_module = pytest.importorskip("PIL.Image")
    path = tmp_path / "image.png"
    image_module.new("RGB", (4, 4), "red").save(path)
    item = {"id": "one", "text": "A generated caption.", "media": {"image": str(path)}}
    native = EmbeddingGemma2Adapter(mode="native")
    joint = EmbeddingGemma2Adapter(mode="joint")
    n_doc, _ = native._prepare(item, "document")
    j_doc, _ = joint._prepare(item, "document")
    n_query, _ = native._prepare(item, "query")
    assert list(n_doc) == ["image"]
    assert list(j_doc) == ["image", "text"]
    assert j_doc["text"] == "title: none | text: A generated caption."
    assert n_query["text"] == "task: search result | query: A generated caption."
    empty_payload, empty_metadata = joint._prepare({**item, "text": ""}, "document")
    assert empty_payload["text"] == ""
    assert empty_metadata["derived_text_empty"] is True
    with pytest.raises(ValueError, match="explicit derived text"):
        joint._prepare({"id": "one", "media": item["media"]}, "document")
    with pytest.raises(ValueError, match="actual media"):
        native._prepare({"id": "one", "text": "A caption"}, "document")


def test_media_must_be_local_existing_files(tmp_path):
    with pytest.raises(ValueError, match="local"):
        _local_media({"media": {"image": "https://example.org/private.png"}})
    with pytest.raises(FileNotFoundError, match="Missing image"):
        _local_media({"id": "a", "media": {"image": str(tmp_path / "missing.png")}})
    with pytest.raises(ValueError, match="map image/audio/video"):
        _local_media({"media": {"caption": "secret"}})


def test_media_order_is_canonical_for_hashed_inputs(tmp_path):
    path = tmp_path / "fixture"
    path.write_bytes(b"data")
    assert list(_local_media({"media": {"video": path, "audio": path, "image": path}})) == ["audio", "image", "video"]


@pytest.mark.parametrize("values", [np.zeros((1, 2)), np.array([[np.nan, 1]]), np.ones((2, 2))])
def test_nonfinite_zero_and_wrong_shape_embeddings_rejected(values):
    with pytest.raises(RuntimeError):
        normalized(values, 1, 2)


def test_normalization_after_dimension_truncation():
    assert np.allclose(normalized([[3, 4]], 1, 2), [[.6, .8]])


def test_multimodal_context_guard_runs_before_forward(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("sentence_transformers")
    seen = []

    class FakeModel:
        def preprocess(self, inputs, **kwargs):
            assert kwargs["processing_kwargs"]["text"]["truncation"] is False
            assert kwargs["processing_kwargs"]["audio"]["truncation"] is False
            return {"attention_mask": torch.ones((1, 101), dtype=torch.long)}

        def __call__(self, features):
            seen.append(features)
            raise AssertionError("Forward must not run for over-budget multimodal input")

    model = EmbeddingGemma2Adapter({"max_length": 100, "device": "cpu"})
    monkeypatch.setattr(model, "_load", lambda: FakeModel())
    monkeypatch.setattr(model, "_prepare", lambda item, role: ({"text": "x"}, {}))
    with pytest.raises(ValueError, match="Expanded multimodal context 101"):
        model.encode([{"id": "x"}])
    assert not seen


def test_colqwen_cannot_be_silently_mean_pooled():
    from rag_benchmark.multimodal import UnsupportedConfiguration
    with pytest.raises(UnsupportedConfiguration, match="MaxSim"):
        SpecialistAdapter("colqwen")


def test_audio_resampling_keeps_duration_and_rejects_empty(tmp_path):
    sf = pytest.importorskip("soundfile")
    pytest.importorskip("scipy")
    from rag_benchmark.multimodal_models import load_audio
    path = tmp_path / "tone.wav"
    sf.write(path, np.ones((8000, 2), dtype=np.float32) * .1, 8000)
    waveform, details = load_audio(path, 16000)
    assert waveform.shape == (16000,)
    assert details["source_channels"] == 2
    assert details["duration_seconds"] == 1
    sf.write(path, np.zeros((0, 1), dtype=np.float32), 8000)
    with pytest.raises(ValueError, match="nonempty"):
        load_audio(path, 16000)


def test_text_context_is_not_silently_truncated():
    model = SpecialistAdapter("siglip2")
    model._processor = SimpleNamespace(tokenizer=lambda inputs, **kw: {"input_ids": [[1] * 65]})
    with pytest.raises(ValueError, match="64-token"):
        model._text_guard(["long query"], 64)


def test_factory_cannot_mislabel_joint_as_native():
    from rag_benchmark.multimodal_models import make_multimodal_adapter
    with pytest.raises(ValueError, match="conflicts"):
        make_multimodal_adapter("N", {"mode": "joint"})

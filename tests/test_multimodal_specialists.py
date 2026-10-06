from types import SimpleNamespace

import numpy as np
import pytest

from rag_benchmark.multimodal_specialists import SpecialistAdapter


def test_overflow_policy_is_explicit_and_hashed():
    strict = SpecialistAdapter("siglip2")
    truncated = SpecialistAdapter("siglip2", {"text_overflow_policy": "truncate_to_model_limit"})
    assert strict.identity != truncated.identity
    assert strict.config["text_overflow_policy"] == "error"
    with pytest.raises(ValueError, match="text_overflow_policy"):
        SpecialistAdapter("clap", {"text_overflow_policy": "automatic"})


def test_declared_truncation_preserves_original_lengths_and_counts(monkeypatch):
    torch = pytest.importorskip("torch")
    model = SpecialistAdapter("siglip2", {"device": "cpu", "dtype": "float32", "batch_size": 2,
        "text_overflow_policy": "truncate_to_model_limit"})

    class Processor:
        tokenizer = staticmethod(lambda texts, **kw: {"input_ids": [[1] * len(text) for text in texts]})

        def __call__(self, text, **kwargs):
            assert kwargs["truncation"] is True
            assert kwargs["max_length"] == 64
            # Native SigLIP deliberately emits no attention mask.
            return {"input_ids": torch.ones((len(text), 64), dtype=torch.long)}

    model._processor = Processor()
    backend = SimpleNamespace(get_text_features=lambda **kw: torch.ones((len(kw["input_ids"]), 768)))
    monkeypatch.setattr(model, "_load", lambda: backend)
    monkeypatch.setattr(model, "_verify_device", lambda: ["cpu"])
    result = model.encode([{"id": "long", "text": "a" * 80}, {"id": "short", "text": "x" * 10}])
    assert np.allclose(np.linalg.norm(result, axis=1), 1)
    assert model.last_usage["truncated_text_items"] == 1
    assert [x["original_tokens"] for x in model.last_usage["items"]] == [80, 10]
    assert [x["retained_tokens"] for x in model.last_usage["items"]] == [64, 10]


def test_default_fails_before_forward(monkeypatch):
    model = SpecialistAdapter("clip_video", {"device": "cpu"})
    model._processor = SimpleNamespace(tokenizer=lambda *a, **kw: {"input_ids": [[1] * 78]})
    monkeypatch.setattr(model, "_load", lambda: None)
    pytest.importorskip("torch")
    with pytest.raises(ValueError, match="explicit overflow policy"):
        model.encode([{"id": "long", "text": "long"}])

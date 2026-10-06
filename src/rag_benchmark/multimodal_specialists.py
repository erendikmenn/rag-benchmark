"""Explicit native-token-limit policies for SigLIP2, CLAP, and CLIP text queries.

Separate from the native-media module so specialist policy changes do not alter
EmbeddingGemma identities or invalidate its expensive image/audio/video cache.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .models import stable_hash, synchronize
from .multimodal_models import SpecialistAdapter as _NativeSpecialistAdapter
from .multimodal_models import _local_media, _unsupported, normalized

_SOURCE_AUDIT = stable_hash(Path(__file__).read_text())


class SpecialistAdapter(_NativeSpecialistAdapter):
    def __init__(self, name: str, config: dict | None = None):
        config = {"text_overflow_policy": "error", **(config or {})}
        if config["text_overflow_policy"] not in {"error", "truncate_to_model_limit"}:
            raise ValueError("text_overflow_policy must be error or truncate_to_model_limit.")
        super().__init__(name, config)
        self.identity = stable_hash({"native_identity": self.identity, "text_policy_adapter": "explicit-native-limit-v1",
            "source_sha256": _SOURCE_AUDIT, "text_overflow_policy": self.config["text_overflow_policy"]})
        self.native_text_limit = {"siglip2": 64, "clip_video": 77, "clap": 512}[name]

    def encode(self, items: list[dict], role: str = "query") -> np.ndarray:
        if role != "query" or not items:
            output = super().encode(items, role=role)
            self.last_usage.update({"text_overflow_policy": self.config["text_overflow_policy"], "truncated_text_items": 0})
            return output
        import torch
        model = self._load()
        tokenizer = self._processor.tokenizer
        texts = []
        for item in items:
            text = item.get("text")
            if _local_media(item) or not isinstance(text, str) or not text.strip():
                raise _unsupported(f"{self.name} specialist accepts nonempty text-only queries in this condition.")
            texts.append(text)
        original = tokenizer(texts, padding=False, truncation=False)["input_ids"]
        original_lengths = [len(tokens) for tokens in original]
        overflow = [length > self.native_text_limit for length in original_lengths]
        policy = self.config["text_overflow_policy"]
        if any(overflow) and policy == "error":
            raise ValueError(f"{self.name} query exceeds {self.native_text_limit} tokens; choose an explicit overflow policy.")
        values, records = [], []
        start = time.perf_counter()
        for offset in range(0, len(items), self.config["batch_size"]):
            chunk = texts[offset:offset + self.config["batch_size"]]
            kwargs = {"padding": "max_length" if self.name == "siglip2" else True,
                "truncation": policy == "truncate_to_model_limit", "return_tensors": "pt"}
            if self.name == "siglip2" or kwargs["truncation"]:
                kwargs["max_length"] = self.native_text_limit
            features = self._processor(text=chunk, **kwargs)
            processed_length = features["input_ids"].shape[1]
            if "attention_mask" in features:
                retained = features["attention_mask"].sum(dim=1).tolist()
            else:
                # SigLIP2 deliberately omits an attention mask and always consumes
                # fixed-length padded input. Record retained content separately.
                retained = [min(original_lengths[i], processed_length)
                    for i in range(offset, offset + len(chunk))]
            if max(retained) > self.native_text_limit:
                raise RuntimeError("Processor did not honor the declared native text limit.")
            with torch.inference_mode():
                vector = model.get_text_features(**self._features(features))
                if hasattr(vector, "pooler_output"):
                    vector = vector.pooler_output
                values.append(vector.float().cpu().numpy())
            for position, length in enumerate(retained, offset):
                records.append({"id": items[position].get("id"), "modalities": ["text"],
                    "original_tokens": original_lengths[position], "retained_tokens": int(length),
                    "processed_tokens_including_padding": processed_length,
                    "native_token_limit": self.native_text_limit, "truncated": overflow[position]})
        synchronize(self.config["device"])
        result = normalized(np.concatenate(values), len(items), self.dimension)
        self.last_usage = {"inference_seconds": time.perf_counter() - start, "items": records,
            "devices": self._verify_device(), "dtype": self.config["dtype"],
            "text_overflow_policy": policy, "truncated_text_items": sum(overflow),
            "native_text_limit": self.native_text_limit, "implementation_source_sha256": _SOURCE_AUDIT}
        return result

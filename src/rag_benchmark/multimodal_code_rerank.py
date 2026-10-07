"""Lossless shared code chunks with function-level maximum relevance score.

This is an explicit reranking condition. It preserves the original candidate
pool and each complete query. Native pair-context checks remain the base
reranker's responsibility: any overflowing pair fails without truncation.
"""
from __future__ import annotations

import copy
import hashlib
import math
import numbers
import time
from pathlib import Path

from .models import stable_hash
from .multimodal_code import SharedCodeSegmenter

_VERSION = "shared-code-segments-max-relevance-v1"
_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class SharedCodeReranker:
    """Score every shared source chunk, then return one maximum per function.

    Construct this only after the base's preflight has finalized its identity.
    The base must expose pointwise scores in the input candidate order. No
    thresholds, missing scores, fabricated floors, or listwise scores are valid.
    """
    pointwise = True

    def __init__(self, base, segmenter: SharedCodeSegmenter):
        if getattr(base, "pointwise", False) is not True or not callable(getattr(base, "score", None)):
            raise ValueError("Shared code reranking requires a pointwise base.score(query, candidates).")
        if not isinstance(getattr(base, "identity", None), str) or not base.identity:
            raise ValueError("The base reranker must have a finalized model and prompt identity.")
        if not isinstance(segmenter, SharedCodeSegmenter):
            raise TypeError("A SharedCodeSegmenter is required for shared retrieval/reranking boundaries.")
        self.base, self.segmenter = base, segmenter
        self._base_identity, self._segmentation_identity = base.identity, segmenter.identity
        self.protocol = {
            "version": _VERSION,
            "condition": "source_complete_character_bounded_function_max_relevance"
                if segmenter.enforce_character_limit else "shared_segmentation_function_max_relevance",
            "base_identity": base.identity,
            "segmentation_identity": segmenter.identity,
            "function_score": "maximum_actual_base_relevance_score_over_all_shared_source_chunks",
            "native_single_chunk_policy": "keep_complete_original_candidate_when_both_tokenizers_and_character_cap_fit"
                if segmenter.enforce_character_limit else "keep_complete_original_candidate_when_both_tokenizers_fit",
            "query_policy": "complete_query_as_provided_without_segmentation_or_truncation",
            "context_policy": "base_pair_context_validation_remains_strict;no_truncation_or_retry_with_clipping",
            "source_coverage": "exact_concatenation_no_overlap_no_normalization_no_dropped_characters",
            "candidate_policy": "one_score_per_original_function_in_original_order;no_drops_or_score_floor",
        }
        self.identity = stable_hash({"protocol": self.protocol, "source_sha256": _SOURCE_SHA256})
        self._protocol_seal = stable_hash(self.protocol)
        self.last_usage = {}

    def _check_identity(self) -> None:
        if (self.base.identity != self._base_identity
                or self.segmenter.identity != self._segmentation_identity
                or stable_hash(self.protocol) != self._protocol_seal
                or getattr(self.base, "pointwise", False) is not True):
            raise ValueError("Code reranking configuration changed after its identity was fixed; finalize base preflight first.")

    def unload(self) -> None:
        if callable(getattr(self.base, "unload", None)):
            self.base.unload()

    def score(self, query: dict, candidates: list[dict]) -> list[float]:
        self._check_identity()
        self.last_usage = {}
        ids = [candidate.get("id") for candidate in candidates]
        if any(not isinstance(identifier, str) or not identifier for identifier in ids) or len(ids) != len(set(ids)):
            raise ValueError("Original function IDs must be unique nonempty strings.")
        if not isinstance(query.get("text"), str) or query.get("media"):
            raise ValueError("Code reranking requires a complete text query without media.")
        started = time.perf_counter()
        chunks, spans, summaries = [], [], []
        for candidate in candidates:
            parts, summary = self.segmenter.partition(candidate)
            start = len(chunks)
            if len(parts) == 1:
                # Preserve the native short-function input, including its ID.
                chunks.append(copy.deepcopy(candidate))
            else:
                chunks.extend({"id": part["id"], "text": part["text"], "media": {},
                               **({"title": part["title"]} if "title" in candidate else {})}
                              for part in parts)
            spans.append((start, len(chunks)))
            summaries.append(summary)
        if len({chunk["id"] for chunk in chunks}) != len(chunks):
            raise ValueError("Shared code chunk IDs collide with another candidate ID.")
        segmentation_seconds = time.perf_counter() - started
        inference_start = time.perf_counter()
        # Any native context failure propagates. A partial function is never scored.
        values = list(self.base.score(query, chunks)) if chunks else []
        self._check_identity()
        if (len(values) != len(chunks) or any(isinstance(value, bool) or not isinstance(value, numbers.Real)
                or not math.isfinite(float(value)) for value in values)):
            raise RuntimeError("Base reranker must return one finite real score per source chunk in input order.")
        scores = [max(float(value) for value in values[start:stop]) for start, stop in spans]
        self.last_usage = {
            "condition": self.protocol["condition"], "protocol": copy.deepcopy(self.protocol),
            "segmentation_identity": self._segmentation_identity,
            "original_function_count": len(candidates), "chunk_count": len(chunks),
            "segmented_function_count": sum(summary["segmented"] for summary in summaries),
            "native_short_function_count": sum(not summary["segmented"] for summary in summaries),
            "source_characters": sum(len(candidate["text"]) for candidate in candidates),
            "covered_source_characters": sum(len(chunk["text"]) for chunk in chunks),
            "source_utf8_bytes": sum(len(candidate["text"].encode("utf-8")) for candidate in candidates),
            "covered_source_utf8_bytes": sum(len(chunk["text"].encode("utf-8")) for chunk in chunks),
            "aggregation": "maximum_actual_base_relevance_score_per_original_function",
            "segmentation_seconds": segmentation_seconds,
            "inference_seconds": time.perf_counter() - inference_start if chunks else None,
            "base_usage": copy.deepcopy(getattr(self.base, "last_usage", {})) if chunks else {},
        }
        return scores

"""Explicit complete-audio windows with maximum pointwise relevance per source.

The score is the strongest individual audio-window evidence. It does not combine
audio evidence across windows. Complete source text, when provided, is repeated
unchanged in each window. This condition is separate from whole-source scoring.
"""
from __future__ import annotations

import copy
import hashlib
import math
import numbers
import time
import uuid
from pathlib import Path

import numpy as np

from .models import package_versions, stable_hash
from .multimodal_models import load_audio

_VERSION = "source-audio-windows-max-relevance-v1"
_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_item(item: dict) -> dict:
    """Keep complete model content, excluding arbitrary metadata and gold fields."""
    result = {key: copy.deepcopy(item[key]) for key in ("id", "text", "media") if key in item}
    if "text" in result and not isinstance(result["text"], str):
        raise ValueError("Provided source text must be an explicit string.")
    return result


def _window_file(path: Path, waveform: np.ndarray, sampling_rate: int) -> None:
    """Persist exact float32 samples; validate cached data rather than trust its name."""
    import soundfile as sf
    if path.is_file():
        saved, rate = sf.read(path, dtype="float32", always_2d=False)
        if rate != sampling_rate or saved.shape != waveform.shape or not np.array_equal(saved, waveform):
            raise ValueError("Cached source-audio window does not match the complete original samples.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix("." + uuid.uuid4().hex + ".tmp")
    try:
        sf.write(temporary, waveform, sampling_rate, format="WAV", subtype="FLOAT")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class SourceAudioWindowReranker:
    """Retain all audio samples, score contiguous windows, take one actual maximum.

    The base must finalize preflight before construction and expose pointwise
    scores in candidate input order. Short sources use their native original
    audio file. Long sources are resampled once before contiguous slicing, so
    window boundaries cannot drop samples or reset a resampling filter.
    """
    pointwise = True

    def __init__(self, base, max_seconds: float = 30.0, *,
                 cache_dir: Path = Path(".cache/multimodal-audio-windows")):
        if getattr(base, "pointwise", False) is not True or not callable(getattr(base, "score", None)):
            raise ValueError("Source audio windows require a pointwise base.score(query, candidates).")
        if not isinstance(getattr(base, "identity", None), str) or not base.identity:
            raise ValueError("The base reranker must have a finalized model and prompt identity.")
        if (isinstance(max_seconds, bool) or not isinstance(max_seconds, numbers.Real)
                or not math.isfinite(max_seconds) or max_seconds <= 0):
            raise ValueError("max_seconds must be finite and positive.")
        sampling_rate = 16000
        window_samples = math.floor(float(max_seconds) * sampling_rate)
        if window_samples < 1:
            raise ValueError("An audio window must retain at least one sample.")
        base_maximum = getattr(base, "config", {}).get("audio_max_duration_seconds")
        if base_maximum is not None and float(max_seconds) > float(base_maximum):
            raise ValueError("Audio window duration exceeds the base reranker's declared native limit.")
        self.base, self.cache_dir = base, Path(cache_dir)
        self.config = {"max_seconds": float(max_seconds), "sampling_rate": sampling_rate,
                       "max_window_samples": window_samples, "window_format": "WAV_FLOAT32"}
        self.protocol = {
            "version": _VERSION, "condition": "source_audio_windows_max_relevance",
            "window_policy": "contiguous_nonoverlapping_in_source_order_last_window_keeps_all_remaining_samples",
            "audio_preprocessing": "whole_source_mono_arithmetic_mean_then_scipy_polyphase_resample_16000Hz",
            "coverage": "every_preprocessed_source_sample_once;no_overlap_no_truncation_no_dropped_windows",
            "native_short_policy": "complete_original_audio_file_and_provided_source_text",
            "source_text_policy": "complete_provided_source_text_repeated_unchanged_in_every_window",
            "query_policy": "complete_provided_query_content_without_truncation;metadata_excluded",
            "aggregation": "maximum_actual_base_score_per_original_candidate",
            "interpretation": "maximum_window_audio_evidence_with_complete_provided_source_text;no_cross_window_audio_evidence_synthesis",
            "base_context_policy": "strict_pair_context_validation;no_clipping_or_retry_with_less_source",
            "base_identity": base.identity,
        }
        self.identity = stable_hash({"config": self.config, "protocol": self.protocol,
            "source_sha256": _SOURCE_SHA256, "audio_preprocessing_source_sha256":
                hashlib.sha256(Path(__file__).with_name("multimodal_models.py").read_bytes()).hexdigest(),
            "packages": package_versions(("numpy", "soundfile", "scipy"))})
        self._seal = stable_hash({"config": self.config, "protocol": self.protocol})
        self._base_identity = base.identity
        self.last_usage = {}

    def _check_identity(self) -> None:
        if (self.base.identity != self._base_identity or getattr(self.base, "pointwise", False) is not True
                or stable_hash({"config": self.config, "protocol": self.protocol}) != self._seal):
            raise ValueError("Audio reranking configuration changed after identity was fixed; finalize base preflight first.")

    def unload(self) -> None:
        if callable(getattr(self.base, "unload", None)):
            self.base.unload()

    def _windows(self, candidate: dict) -> tuple[list[dict], dict, tuple[Path, str]]:
        item = _source_item(candidate)
        media = item.get("media")
        if not isinstance(media, dict) or set(media) != {"audio"}:
            raise ValueError("Source audio window candidates require exactly one local audio source.")
        location = media["audio"]
        if not isinstance(location, (str, Path)) or "://" in str(location):
            raise ValueError("Source audio windows accept existing local files only.")
        path = Path(location).resolve()
        if not path.is_file():
            raise FileNotFoundError("Source audio file is missing.")
        item["media"] = {"audio": str(path)}
        digest = _file_hash(path)
        waveform, _ = load_audio(path, self.config["sampling_rate"])
        if _file_hash(path) != digest:
            raise ValueError("Source audio changed during window preparation.")
        total_samples = len(waveform)
        maximum = self.config["max_window_samples"]
        windows, retained, ranges = [], 0, []
        if total_samples <= maximum:
            windows.append(item)
            retained = total_samples
            ranges.append({"sample_start": 0, "sample_end": total_samples})
        else:
            # The storage key is independent of query/model, but includes the exact
            # preprocessing implementation and all numerical window settings.
            key = stable_hash({"source_bytes": digest, "config": self.config, "wrapper": self.identity})
            for index, start in enumerate(range(0, total_samples, maximum)):
                stop = min(start + maximum, total_samples)
                piece = waveform[start:stop]
                window_path = (self.cache_dir / key / f"{start}-{stop}.wav").resolve()
                _window_file(window_path, piece, self.config["sampling_rate"])
                window = {**item, "id": stable_hash({"source_id": item["id"], "source_bytes": digest,
                          "window_index": index, "sample_start": start, "sample_end": stop}),
                          "media": {"audio": str(window_path)}}
                windows.append(window)
                ranges.append({"sample_start": start, "sample_end": stop})
                retained += len(piece)
        if retained != total_samples or not windows:
            raise RuntimeError("Audio windows did not retain every source sample.")
        summary = {"id": item["id"], "audio_window_count": len(windows),
            "segmented_candidate_count": int(len(windows) > 1),
            "native_short_candidate_count": int(len(windows) == 1),
            "source_audio_samples": total_samples, "retained_audio_samples": retained,
            "source_audio_duration_ms": total_samples * 1000 // self.config["sampling_rate"],
            "retained_audio_duration_ms": retained * 1000 // self.config["sampling_rate"],
            "repeated_source_text_window_count": len(windows) if len(windows) > 1 and item.get("text") else 0,
            "windows": ranges}
        return windows, summary, (path, digest)

    def score(self, query: dict, candidates: list[dict]) -> list[float]:
        self._check_identity()
        self.last_usage = {}
        identifiers = [candidate.get("id") for candidate in candidates]
        if (any(not isinstance(identifier, str) or not identifier for identifier in identifiers)
                or len(identifiers) != len(set(identifiers))):
            raise ValueError("Original audio candidate IDs must be unique nonempty strings.")
        query_content = _source_item(query)
        start = time.perf_counter()
        windows, spans, summaries, sources = [], [], [], []
        for candidate in candidates:
            parts, summary, source = self._windows(candidate)
            spans.append((len(windows), len(windows) + len(parts)))
            windows.extend(parts)
            summaries.append(summary)
            sources.append(source)
        if len({window["id"] for window in windows}) != len(windows):
            raise ValueError("Audio window IDs collide with another original candidate ID.")
        preparation_seconds = time.perf_counter() - start
        inference_start = time.perf_counter()
        values = list(self.base.score(query_content, windows)) if windows else []
        self._check_identity()
        if (len(values) != len(windows) or any(isinstance(value, bool) or not isinstance(value, numbers.Real)
                or not math.isfinite(float(value)) for value in values)):
            raise RuntimeError("Base reranker must return one finite real score per audio window in input order.")
        if any(_file_hash(path) != digest for path, digest in sources):
            raise ValueError("Source audio changed during relevance inference.")
        scores = [max(float(value) for value in values[begin:end]) for begin, end in spans]
        counts = {key: sum(summary[key] for summary in summaries) for key in (
            "audio_window_count", "segmented_candidate_count", "native_short_candidate_count",
            "source_audio_samples", "retained_audio_samples", "source_audio_duration_ms",
            "retained_audio_duration_ms", "repeated_source_text_window_count")}
        self.last_usage = {"condition": self.protocol["condition"], "protocol": copy.deepcopy(self.protocol),
            "aggregation": self.protocol["aggregation"], "original_candidate_count": len(candidates), **counts,
            "sampling_rate": self.config["sampling_rate"], "max_window_samples": self.config["max_window_samples"],
            "preparation_seconds": preparation_seconds,
            "inference_seconds": time.perf_counter() - inference_start if windows else None,
            "items": summaries}
        return scores

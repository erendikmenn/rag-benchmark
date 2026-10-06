"""Frozen-data multimodal matrix, exact retrieval, and resumable evaluation.

No model is downloaded by this module. Adapters encode a list of {id,text,media}
items with an explicit role and expose a stable ``identity`` and ``dimension``.
Rerankers expose ``score(query, candidates)`` and ``identity``; ``pointwise=True``
permits reuse of independent pair scores. Unsupported channels never fall back.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
import gc
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import sqlite3
import time
from typing import Any
import uuid

import numpy as np

from .models import DenseEmbedder, package_versions, stable_hash
from .retrieval import turkish_tokens


TRACKS = {
    "photo": ("XM3600", "BGENSJ", "SigLIP2"),
    "document": ("ViDoRe V3 public", "BGENSJ", "ColQwen2.5"),
    "environment_audio": ("Clotho v2.1 eval", "BGENSJ", "CLAP"),
    "speech": ("FLEURS tr_tr test", "BGENJ", "none"),
    "video": ("MSR-VTT 1K-A", "BGENSJ", "CLIP frame pooling"),
    "code": ("Cleaned CodeSearchNet", "BGE", "none"),
    "composed_image": ("CIRR public validation", "BGENJ", "none"),
}
RERANKERS = ("none", "laya_text", "bge_reranker_text", "gemma4_relevance")
ENGINE_VERSION = "multimodal-v1"
BM25_CANDIDATE_SELECTION = "positive_scores_only_v1"
_PROCESSING_COUNTS = ("original_function_count", "chunk_count", "segmented_function_count",
    "native_short_function_count", "unchanged_function_count", "source_characters", "covered_source_characters",
    "source_utf8_bytes", "covered_source_utf8_bytes", "original_query_count")


class UnsupportedConfiguration(RuntimeError):
    """An explicit capability gap, not a lower-quality substitute implementation."""


def _safe_model_config(config: Any) -> dict:
    """Publish only declared numerical settings, model identifiers and checksums."""
    if not isinstance(config, dict):
        return {}
    result = {}
    for key in ("batch_size", "dimension", "max_length", "max_len", "context_limit", "native_text_limit",
                "vision_budget", "video_vision_budget", "video_fps", "video_max_frames", "max_image_patches",
                "score_query_batch", "score_chunk_elements", "max_tokens", "max_new_tokens", "temperature", "seed",
                "image_max_side", "video_max_duration_seconds", "audio_max_duration_seconds", "threshold",
                "max_tokens_after_native_document_formatting", "maximum_characters_per_chunk_for_overflowing_functions"):
        value = config.get(key)
        if value is None and key in config:
            result[key] = None
        elif isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool) and math.isfinite(value):
            result[key] = value.item() if isinstance(value, np.generic) else value
    for key in ("model_id", "base_model_id"):
        value = config.get(key)
        if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", value):
            result[key] = value
    for key in ("revision", "base_revision", "model_sha256", "projector_sha256"):
        value = config.get(key)
        if isinstance(value, str) and re.fullmatch(r"[a-f0-9]{40}|[a-f0-9]{64}", value):
            result[key] = value
    enums = {"dtype": {"float32", "bfloat16", "float16"}, "score_dtype": {"float32", "bfloat16"},
             "text_overflow_policy": {"error", "truncate_to_model_limit"}, "backend": {"sdk", "http"},
             "mode": {"native", "joint", "text"}}
    for key, allowed in enums.items():
        if isinstance(config.get(key), str) and config[key] in allowed:
            result[key] = config[key]
    for key in ("device", "score_device"):
        if isinstance(config.get(key), str) and re.fullmatch(r"(?:cpu|mps|cuda)(?::[0-9]+)?", config[key]):
            result[key] = config[key]
    for key in ("model", "quantization", "pooling", "scoring", "token_mask", "token_normalization"):
        if isinstance(config.get(key), str) and re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", config[key]):
            result[key] = config[key]
    for key in ("local_files_only", "enable_thinking"):
        if isinstance(config.get(key), bool):
            result[key] = config[key]
    return result


def _safe_protocol(protocol: Any) -> dict:
    if not isinstance(protocol, dict):
        return {}
    result = _safe_model_config(protocol)
    for key in ("version", "condition", "boundary_policy", "native_single_chunk_policy", "source_coverage",
                "function_score", "query_policy", "context_policy", "candidate_policy", "base_identity", "segmentation_identity"):
        value = protocol.get(key)
        if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.;:-]{1,256}", value):
            result[key] = value
    tokenizers = protocol.get("tokenizers", {})
    if isinstance(tokenizers, dict):
        result["tokenizers"] = {name: _safe_model_config(tokenizers[name])
                                for name in ("bge", "embeddinggemma") if name in tokenizers}
    return result


def public_adapter_spec(adapter: Any, *, _seen: set[int] | None = None) -> dict:
    """Unwrap known adapters without serializing paths, prompts or private state."""
    seen = set() if _seen is None else _seen
    result = {"class": type(adapter).__module__ + "." + type(adapter).__qualname__}
    if id(adapter) in seen:
        result["recursive_reference_omitted"] = True
        return result
    seen.add(id(adapter))
    identity = getattr(adapter, "identity", None)
    if isinstance(identity, str) and re.fullmatch(r"[a-f0-9]{64}", identity):
        result["identity"] = identity
    result["config"] = _safe_model_config(getattr(adapter, "config", {}))
    for name in ("dimension", "native_text_limit"):
        value = getattr(adapter, name, None)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            result[name] = value
    if hasattr(adapter, "protocol"):
        result["protocol"] = _safe_protocol(adapter.protocol)
    for name in ("embedder", "base", "adapter", "encoder", "segmenter", "generator", "verifier"):
        child = getattr(adapter, name, None)
        if child is not None:
            result[name] = public_adapter_spec(child, _seen=seen)
    seen.remove(id(adapter))
    return result


def matrix_registry(track: str | None = None) -> list[dict]:
    """321 nonempty retrieval subsets and four conditions = 1,284 families."""
    if track is not None and track not in TRACKS:
        raise ValueError(f"Unknown track {track!r}")
    rows = []
    for name, (dataset, channel_string, specialist) in TRACKS.items():
        if track is not None and name != track:
            continue
        for size in range(1, len(channel_string) + 1):
            for channels in itertools.combinations(channel_string, size):
                for reranker in RERANKERS:
                    rows.append({
                        "variant_id": name + "__" + "_".join(channels).lower() + "__" + reranker,
                        "track": name, "dataset": dataset, "retrieval_channels": list(channels),
                        "specialist": specialist if "S" in channels else "none",
                        "fusion": "identity" if size == 1 else "equal_weight_rrf_k60",
                        "reranker": reranker, "primary_rerank_k": None if reranker == "none" else 50,
                        "rerank_candidate_grid": [] if reranker == "none" else [20, 50, 100],
                        "status": "planned",
                    })
    return rows


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_array(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"Expected object at {path.name}:{line_number}")
                rows.append(row)
    return rows


def _validate_provenance(provenance: dict) -> None:
    if provenance.get("gold_fields_used") is True or provenance.get("query_independent") is False:
        raise ValueError("Candidate text provenance indicates gold/query leakage.")
    source = str(provenance.get("source", "")).lower()
    if any(token in source for token in ("gold_caption", "gold_transcript", "gold_answer", "gold_docstring")):
        raise ValueError("Gold annotations cannot be candidate text.")


@dataclass
class Dataset:
    root: Path
    manifest: dict
    corpus: list[dict]
    queries: list[dict]
    qrels: dict[str, dict[str, float]]
    identity: str

    @property
    def track(self) -> str:
        return self.manifest["track"]

    def model_item(self, item: dict, *, text_only: bool = False) -> dict:
        # Labels, gold captions, transcripts, answers and arbitrary metadata are
        # deliberately unavailable to encoders and relevance-scoring adapters.
        result = {"id": item["id"]}
        if "text" in item:
            result["text"] = item["text"]
        if text_only and item.get("media"):
            surrogate = item.get("metadata", {}).get("text_surrogate")
            if isinstance(surrogate, dict):
                result["text"] = surrogate["text"]
            elif isinstance(surrogate, str):
                result["text"] = surrogate
        result["media"] = {} if text_only else {kind: str((self.root / relative).resolve())
                                               for kind, relative in item.get("media", {}).items()}
        return result

    def excluded_ids(self, query: dict) -> set[str]:
        metadata = query.get("metadata", {})
        values = metadata.get("exclude_ids", [])
        if not isinstance(values, list):
            raise ValueError("query metadata exclude_ids must be a list")
        excluded = {str(value) for value in values}
        if self.track == "composed_image":
            reference = metadata.get("reference_id", metadata.get("reference_image_id"))
            if reference is None:
                raise ValueError("CIRR query requires reference_id to exclude its reference image.")
            excluded.add(str(reference))
        return excluded

    def require_candidate_text(self) -> None:
        provenance = self.manifest.get("text_provenance", {})
        if provenance.get("query_independent") is not True or provenance.get("gold_fields_used") is not False:
            raise UnsupportedConfiguration("Text channels require explicit query-independent, gold-free provenance.")
        for item in self.corpus:
            override = item.get("metadata", {}).get("text_provenance", provenance)
            _validate_provenance(override)
            # An actually empty source extraction is a valid representation,
            # retained as empty and counted. Missing, not-yet-generated text is
            # a capability gap; it must never be filled with gold annotations.
            empty_is_valid = override.get("empty_text_is_valid", False) or override.get("source") in {
                "published_markdown", "published_extracted_text", "ocr", "asr", "whisper_asr"}
            if not isinstance(item.get("text"), str) or (not item["text"].strip() and not empty_is_valid):
                raise UnsupportedConfiguration("Candidate text is absent; source-only preprocessing is required.")

    def require_text(self) -> None:
        self.require_candidate_text()
        for query in self.queries:
            if query.get("media"):
                surrogate = query.get("metadata", {}).get("text_surrogate")
                if isinstance(surrogate, dict):
                    text = surrogate.get("text")
                    provenance = surrogate.get("provenance", {})
                    _validate_provenance(provenance)
                    if provenance.get("gold_fields_used") is not False:
                        raise UnsupportedConfiguration("Media-query text surrogate needs explicit gold-free provenance.")
                else:
                    text = surrogate
                if not isinstance(text, str) or not text.strip():
                    raise UnsupportedConfiguration("Text channels need an explicit complete text_surrogate for each media query.")
        query_texts = [self.model_item(query, text_only=True).get("text") for query in self.queries]
        if any(not isinstance(text, str) or not text.strip() for text in query_texts):
            raise UnsupportedConfiguration("Text retrieval requires a declared text representation for every query.")

    def public_summary(self) -> dict:
        return {"id": self.manifest.get("id", self.manifest.get("name", self.root.name)),
                "revision": self.manifest["revision"], "track": self.track, "identity": self.identity,
                "corpus_count": len(self.corpus), "query_count": len(self.queries),
                "qrel_count": sum(len(values) for values in self.qrels.values()),
                "positive_qrel_count": sum(sum(r > 0 for r in values.values()) for values in self.qrels.values()),
                "empty_candidate_text_count": sum(isinstance(item.get("text"), str) and not item["text"].strip()
                                                  for item in self.corpus),
                "missing_candidate_text_count": sum(not isinstance(item.get("text"), str) for item in self.corpus),
                "split": self.manifest.get("split", "unspecified")}


def load_dataset(path: Path | str) -> Dataset:
    root = Path(path).resolve()
    manifest = json.loads((root / "dataset.json").read_text())
    if manifest.get("track") not in TRACKS or not manifest.get("revision"):
        raise ValueError("dataset.json requires a canonical track and a frozen revision.")
    _validate_provenance(manifest.get("text_provenance", {}))
    corpus, queries = _jsonl(root / "corpus.jsonl"), _jsonl(root / "queries.jsonl")
    if not corpus or not queries:
        raise ValueError("A dataset needs nonempty corpus and queries.")
    media_hashes = {}
    for rows, label in ((corpus, "corpus"), (queries, "query")):
        seen = set()
        for row in rows:
            identifier = row.get("id")
            if identifier is None or not str(identifier) or str(identifier) in seen:
                raise ValueError(f"Missing or duplicate {label} id")
            row["id"] = str(identifier)
            seen.add(row["id"])
            if "text" in row and row["text"] is not None and not isinstance(row["text"], str):
                raise ValueError("Item text must be a string or absent")
            if label == "corpus":
                _validate_provenance(row.get("metadata", {}).get("text_provenance", {}))
            for kind, relative in row.get("media", {}).items():
                if kind not in {"image", "audio", "video"} or not isinstance(relative, str):
                    raise ValueError("Media must map image/audio/video to relative file paths")
                media_path = (root / relative).resolve()
                if Path(relative).is_absolute() or not media_path.is_relative_to(root):
                    raise ValueError("Media paths must remain inside the dataset folder")
                if not media_path.is_file():
                    raise ValueError(f"Missing media file for {label} id {row['id']}")
                if relative not in media_hashes:
                    media_hashes[relative] = _file_digest(media_path)
    doc_ids = {d["id"] for d in corpus}
    query_ids = {q["id"] for q in queries}
    qrels: dict[str, dict[str, float]] = {identifier: {} for identifier in query_ids}
    for row in _jsonl(root / "qrels.jsonl"):
        qid, did, relevance = str(row["query_id"]), str(row["corpus_id"]), float(row["relevance"])
        if qid not in query_ids or did not in doc_ids:
            raise ValueError("Qrel references an unknown query or corpus id")
        if did in qrels[qid] or not math.isfinite(relevance) or relevance < 0 or relevance > 30:
            raise ValueError("Duplicate or invalid qrel; grades must be finite in [0,30]")
        qrels[qid][did] = relevance
    if any(not any(grade > 0 for grade in grades.values()) for grades in qrels.values()):
        raise ValueError("Every evaluated query needs at least one positive qrel")
    identity = stable_hash({"files": {name: _file_digest(root / name) for name in
                           ("dataset.json", "corpus.jsonl", "queries.jsonl", "qrels.jsonl")},
                            "media": media_hashes})
    dataset = Dataset(root, manifest, corpus, queries, qrels, identity)
    for query in queries:
        excluded = dataset.excluded_ids(query)
        if any(qrels[query["id"]].get(identifier, 0) > 0 for identifier in excluded):
            raise ValueError("A query excludes a positive qrel")
    return dataset


def graded_metrics(ranked_ids: list[str], qrels: dict[str, float], ks=(1, 5, 10, 20),
                   ap_k: int = 16) -> dict[str, float]:
    """Hit is binary; Recall counts all positives; nDCG uses exponential gains."""
    positives = {identifier for identifier, grade in qrels.items() if grade > 0}
    if not positives:
        raise ValueError("Positive qrels are required")
    if any(k < 1 for k in ks) or ap_k < 1:
        raise ValueError("Metric cutoffs must be positive")
    ranked = list(dict.fromkeys(str(identifier) for identifier in ranked_ids))
    ideal_grades = sorted((float(grade) for grade in qrels.values() if grade > 0), reverse=True)
    values = {}
    for k in ks:
        top = ranked[:k]
        count = sum(identifier in positives for identifier in top)
        values[f"hit@{k}"] = float(count > 0)
        values[f"recall@{k}"] = count / len(positives)
        values[f"mrr@{k}"] = next((1 / rank for rank, identifier in enumerate(top, 1)
                                  if identifier in positives), 0.0)
        dcg = sum((2 ** qrels.get(identifier, 0) - 1) / math.log2(rank + 1)
                  for rank, identifier in enumerate(top, 1))
        ideal = sum((2 ** grade - 1) / math.log2(rank + 1)
                    for rank, grade in enumerate(ideal_grades[:k], 1))
        values[f"ndcg@{k}"] = dcg / ideal
    hits = 0
    precision_sum = 0.0
    for rank, identifier in enumerate(ranked[:ap_k], 1):
        if identifier in positives:
            hits += 1
            precision_sum += hits / rank
    values[f"map@{ap_k}"] = precision_sum / min(ap_k, len(positives))
    return values


def aggregate_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = set(rows[0])
    if any(set(row) != keys for row in rows):
        raise ValueError("Cannot aggregate mismatched metric definitions")
    return {key: float(np.mean([row[key] for row in rows])) for key in sorted(keys)}


def paired_group_bootstrap(left: dict[str, float], right: dict[str, float], groups: dict[str, str], *,
                           samples: int = 1000, seed: int = 0) -> dict:
    """Paired right-minus-left differences, resampling shared-source query groups.

    This is an optional comparison of a specified metric, never an average of
    unlike metrics or a claim that correlated captions are independent queries.
    """
    if not left or set(left) != set(right) or set(left) != set(groups) or samples < 1:
        raise ValueError("Paired bootstrap requires identical nonempty query ids and explicit groups")
    identifiers = sorted(left)
    differences = np.asarray([right[identifier] - left[identifier] for identifier in identifiers], dtype=float)
    if not np.isfinite(differences).all():
        raise ValueError("Bootstrap scores must be finite")
    group_names = sorted(set(groups.values()))
    indices = {group: [i for i, identifier in enumerate(identifiers) if groups[identifier] == group]
               for group in group_names}
    totals = np.asarray([differences[indices[group]].sum() for group in group_names])
    counts = np.asarray([len(indices[group]) for group in group_names])
    rng = np.random.default_rng(seed)
    bootstrapped = []
    for _ in range(samples):
        selected = rng.integers(0, len(group_names), size=len(group_names))
        bootstrapped.append(float(totals[selected].sum() / counts[selected].sum()))
    return {"metric_direction": "right_minus_left", "mean_difference": float(differences.mean()),
            "ci95_low": float(np.percentile(bootstrapped, 2.5)),
            "ci95_high": float(np.percentile(bootstrapped, 97.5)),
            "wins": int((differences > 0).sum()), "losses": int((differences < 0).sum()),
            "ties": int((differences == 0).sum()), "query_count": len(identifiers),
            "group_count": len(group_names), "bootstrap_samples": samples, "seed": seed}


def channel_allocations(channels: tuple[str, ...] | list[str], mode: str, budget: int = 100) -> dict[str, int]:
    if not channels or len(set(channels)) != len(channels) or budget < 1:
        raise ValueError("Channels must be unique and budget positive")
    canonical = sorted(channels, key="BGENSJ".index)
    if mode == "per_channel":
        return {channel: budget for channel in canonical}
    if mode != "total":
        raise ValueError("Budget mode must be per_channel or total")
    quotient, remainder = divmod(budget, len(canonical))
    return {channel: quotient + (index < remainder) for index, channel in enumerate(canonical)}


def fuse_rankings(rankings: dict[str, list[dict]], channels: tuple[str, ...] | list[str], *,
                  budget_mode: str = "per_channel", budget: int = 100, rrf_k: int = 60) -> tuple[list[dict], dict]:
    if rrf_k < 0:
        raise ValueError("RRF constant must be nonnegative")
    allocations = channel_allocations(channels, budget_mode, budget)
    scores, pulled = {}, 0
    for channel, count in allocations.items():
        seen = set()
        for rank, row in enumerate(rankings[channel][:count], 1):
            identifier = str(row["id"])
            if identifier in seen:
                raise ValueError("A base ranking contains duplicate ids")
            seen.add(identifier)
            pulled += 1
            score = float(row["score"]) if len(channels) == 1 else 1 / (rrf_k + rank)
            scores[identifier] = scores.get(identifier, 0.0) + score
    ordered = sorted(scores, key=lambda identifier: (-scores[identifier], identifier))
    rows = [{"id": identifier, "score": scores[identifier], "rank": rank}
            for rank, identifier in enumerate(ordered, 1)]
    return rows, {"allocations": allocations, "raw_candidates": pulled, "unique_candidates": len(rows),
                  "refill": False}


def exact_dot_rankings(document_vectors: np.ndarray, query_vectors: np.ndarray, doc_ids: list[str], *,
                       top_k: int = 100, query_batch_size: int = 64,
                       exclusions: list[set[str]] | None = None) -> list[list[dict]]:
    documents = np.asarray(document_vectors, dtype=np.float32)
    queries = np.asarray(query_vectors, dtype=np.float32)
    if documents.ndim != 2 or queries.ndim != 2 or documents.shape[1] != queries.shape[1]:
        raise ValueError("Dense arrays must be 2-D with matching dimensions")
    if len(doc_ids) != len(documents) or len(set(doc_ids)) != len(doc_ids):
        raise ValueError("Document ids must uniquely match dense rows")
    if not np.isfinite(documents).all() or not np.isfinite(queries).all():
        raise ValueError("Dense arrays contain nonfinite values")
    if top_k < 1 or query_batch_size < 1:
        raise ValueError("Search cutoffs and batch size must be positive")
    exclusions = exclusions or [set() for _ in queries]
    if len(exclusions) != len(queries):
        raise ValueError("Exclusions must match query rows")
    identifiers = np.asarray(doc_ids)
    output = []
    for offset in range(0, len(queries), query_batch_size):
        scores = queries[offset:offset + query_batch_size] @ documents.T
        for index, row in enumerate(scores):
            eligible = np.asarray([i for i, identifier in enumerate(doc_ids)
                                   if identifier not in exclusions[offset + index]], dtype=np.int64)
            if len(eligible) > top_k:
                boundary = np.partition(row[eligible], -top_k)[-top_k]
                # Include every boundary tie before the final ID-stable ordering.
                eligible = eligible[row[eligible] >= boundary]
            order = np.lexsort((identifiers[eligible], -row[eligible]))
            keep = eligible[order[:top_k]]
            output.append([{"id": doc_ids[i], "score": float(row[i]), "rank": rank}
                           for rank, i in enumerate(keep, 1)])
    return output


class TextEmbeddingAdapter:
    """Bridge existing pinned BGE/EmbeddingGemma text models to the common API."""
    def __init__(self, family: str, config: dict | None = None):
        self.embedder = DenseEmbedder(family, config)
        self.dimension = self.embedder.dimension
        self.identity = self.embedder.identity
        self.prompt_identity = {"query": "native-search", "document": "native-document"}
        self.text_only = True

    def unload(self) -> None:
        self.embedder._model = None
        gc.collect()
        if str(self.embedder.config["device"]).startswith("mps"):
            import torch
            torch.mps.empty_cache()

    def encode(self, items: list[dict], role: str) -> np.ndarray:
        if role == "document":
            return self.embedder.embed_documents(items)
        if role == "query":
            return self.embedder._encode([self.embedder.format_query(item["text"]) for item in items])
        raise ValueError("Unknown encoder role")


class BGETextReranker:
    """Pinned BGE scoring that retains legitimate explicitly empty extractions.

    Every candidate, including ``text=''``, receives an actual model score. No
    candidate is dropped, fabricated, filled, or assigned a synthetic score floor.
    """
    pointwise = True

    def __init__(self, config: dict | None = None):
        from .multimodal_models import BGEReranker
        self.base = BGEReranker(config)
        self.identity = stable_hash({"base": self.base.identity, "wrapper": "explicit-empty-text-v1"})
        self.last_usage = {}

    def unload(self) -> None:
        self.base.unload()

    def score(self, query: dict, candidates: list[dict]) -> list[float]:
        if not candidates:
            return []
        question = query.get("text")
        if not isinstance(question, str) or not question.strip() or any(not isinstance(item.get("text"), str) for item in candidates):
            raise UnsupportedConfiguration("BGE requires a complete query and an explicit candidate text field")
        import torch
        from .models import synchronize
        model = self.base._load()
        config, scores = self.base.config, []
        started = time.perf_counter()
        for offset in range(0, len(candidates), config["batch_size"]):
            pairs = [(question, item["text"]) for item in candidates[offset:offset + config["batch_size"]]]
            inputs = self.base._processor(pairs, padding=True, truncation=False, return_tensors="pt")
            if int(inputs["attention_mask"].sum(dim=1).max()) > config["max_length"]:
                raise ValueError("BGE reranker pair exceeds configured context; explicit segmentation is required")
            with torch.inference_mode():
                logits = model(**{key: value.to(config["device"]) for key, value in inputs.items()}).logits
                scores.extend(logits.float().cpu().reshape(-1).tolist())
        synchronize(config["device"])
        if len(scores) != len(candidates) or not np.isfinite(scores).all():
            raise RuntimeError("BGE reranker produced invalid scores")
        self.last_usage = {"inference_seconds": time.perf_counter() - started, "pairs": len(scores),
                           "empty_candidate_text_items": sum(not item["text"].strip() for item in candidates),
                           "empty_candidate_policy": "native_model_score", "score_type": "raw_relevance_logit"}
        return scores


class PrecomputedAdapter:
    """Explicit vectors with exact dataset fingerprint/ID coverage; no speed claim."""
    def __init__(self, documents: np.ndarray, queries: np.ndarray, *, document_ids: list[str],
                 query_ids: list[str], identity: str, dataset_identity: str):
        self.values = {"document": np.asarray(documents), "query": np.asarray(queries)}
        self.ids = {"document": list(document_ids), "query": list(query_ids)}
        self.dimension = int(self.values["document"].shape[1])
        self.dataset_identity = dataset_identity
        for role in self.values:
            _validate_vectors(self.values[role], len(self.ids[role]), self.dimension)
            if len(set(self.ids[role])) != len(self.ids[role]):
                raise ValueError("Precomputed ids must be unique")
        self.lookup = {role: {identifier: i for i, identifier in enumerate(ids)} for role, ids in self.ids.items()}
        self.identity = stable_hash({"adapter": "precomputed-v1", "source_identity": identity,
                                     "dataset": dataset_identity, "ids": self.ids,
                                     "vectors": {role: hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()
                                                 for role, values in self.values.items()}})
        self.precomputed = True

    def encode(self, items: list[dict], role: str) -> np.ndarray:
        return self.values[role][[self.lookup[role][item["id"]] for item in items]]


def _validate_vectors(values: np.ndarray, count: int, dimension: int) -> None:
    if values.shape != (count, dimension) or not np.isfinite(values).all():
        raise ValueError("Encoder returned the wrong shape or nonfinite vectors")
    if count and not np.allclose(np.linalg.norm(values, axis=1), 1.0, atol=2e-3, rtol=2e-3):
        raise ValueError("Encoder vectors must be normalized; zero vectors are invalid")


def encoding_cache_identity(dataset: Dataset, adapter: Any, role: str) -> str:
    if role not in {"query", "document"} or not getattr(adapter, "identity", None):
        raise ValueError("Adapter identity and explicit query/document role are required")
    dimension = int(adapter.dimension)
    if dimension < 1:
        raise ValueError("Adapter dimension must be positive")
    if getattr(adapter, "dataset_identity", dataset.identity) != dataset.identity:
        raise ValueError("Precomputed vectors belong to a different frozen dataset")
    return stable_hash({"engine": ENGINE_VERSION, "dataset": dataset.identity, "revision": dataset.manifest["revision"],
                        "adapter": adapter.identity, "role": role, "dimension": dimension,
                        "prompts": getattr(adapter, "prompt_identity", None)})


def _capture_diagnostics(usage: Any, item_ids: list[str]) -> dict:
    """Copy only numerical processing facts and known settings, never input text."""
    usage = usage if isinstance(usage, dict) else {}
    result = {"rows": len(item_ids), "reported": bool(usage), "items": []}
    policy = usage.get("text_overflow_policy")
    if policy in {"error", "truncate_to_model_limit"}:
        result["text_overflow_policy"] = policy
    for key in ("native_text_limit", "context_limit", "truncated_text_items", "empty_derived_text_items", "empty_candidate_text_items"):
        value = usage.get(key)
        if isinstance(value, (int, np.integer)) and not isinstance(value, bool) and value >= 0:
            result[key] = int(value)
    if usage.get("dtype") in {"float32", "bfloat16", "float16"}:
        result["dtype"] = usage["dtype"]
    if usage.get("empty_candidate_policy") == "native_model_score":
        result["empty_candidate_policy"] = "native_model_score"
    result.update(_processing_diagnostics(usage))
    records = usage.get("items", [])
    if isinstance(records, list) and records:
        if len(records) != len(item_ids):
            raise ValueError("Adapter diagnostics do not match the encoded row count")
        for identifier, record in zip(item_ids, records):
            if not isinstance(record, dict):
                raise ValueError("Adapter processing diagnostics must contain records")
            if record.get("id") is not None and str(record["id"]) != identifier:
                raise ValueError("Adapter diagnostic IDs do not match the encoded row order")
            safe = {"id": identifier}
            for key in ("original_tokens", "retained_tokens", "processed_tokens_including_padding",
                        "native_token_limit", "expanded_tokens"):
                value = record.get(key)
                if isinstance(value, (int, np.integer)) and not isinstance(value, bool) and value >= 0:
                    safe[key] = int(value)
            for key in ("truncated", "derived_text_empty"):
                if isinstance(record.get(key), bool):
                    safe[key] = record[key]
            if isinstance(record.get("modalities"), list):
                safe["modalities"] = [value for value in record["modalities"] if value in {"text", "image", "audio", "video"}]
            result["items"].append(safe)
        flags = [record["truncated"] for record in result["items"] if "truncated" in record]
        if len(flags) == len(item_ids):
            if "truncated_text_items" in result and result["truncated_text_items"] != sum(flags):
                raise ValueError("Adapter truncation count disagrees with its per-item diagnostics")
            result["truncated_text_items"] = sum(flags)
    return result


def _processing_diagnostics(usage: dict) -> dict:
    """Aggregate source/chunk counts cannot be attributed to cached score pairs."""
    result = {}
    for key in _PROCESSING_COUNTS:
        value = usage.get(key)
        if isinstance(value, (int, np.integer)) and not isinstance(value, bool) and value >= 0:
            result[key] = int(value)
    for key in ("condition", "aggregation"):
        value = usage.get(key)
        if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", value):
            result[key] = value
    identity = usage.get("segmentation_identity")
    if isinstance(identity, str) and re.fullmatch(r"[a-f0-9]{64}", identity):
        result["segmentation_identity"] = identity
    if isinstance(usage.get("protocol"), dict):
        result["protocol"] = _safe_protocol(usage["protocol"])
    return result


def _merge_diagnostics(blocks: list[dict], total_rows: int) -> dict:
    reported = sum(block["rows"] for block in blocks if block.get("reported"))
    records = [record for block in blocks for record in block.get("items", [])]
    complete_truncation = sum(block["rows"] for block in blocks if "truncated_text_items" in block) == total_rows
    return {"status": "complete" if reported == total_rows else "partial" if reported else "unavailable",
            "total_rows": total_rows, "reported_rows": reported, "missing_rows": total_rows - reported,
            "text_overflow_policies": sorted({block["text_overflow_policy"] for block in blocks if "text_overflow_policy" in block}),
            "native_text_limits": sorted({block["native_text_limit"] for block in blocks if "native_text_limit" in block}),
            "truncated_text_items": sum(block.get("truncated_text_items", 0) for block in blocks) if complete_truncation else None,
            "observed_truncated_text_items": sum(block.get("truncated_text_items", 0) for block in blocks),
            "empty_derived_text_items": sum(block.get("empty_derived_text_items", 0) for block in blocks)
                if blocks and reported == total_rows and all("empty_derived_text_items" in block for block in blocks) else None,
            "empty_candidate_text_items_observed": sum(block.get("empty_candidate_text_items", 0) for block in blocks),
            "expanded_tokens_max": max((record["expanded_tokens"] for record in records if "expanded_tokens" in record), default=None),
            "original_tokens_total": sum(record.get("original_tokens", 0) for record in records)
                if any("original_tokens" in record for record in records) else None,
            "retained_tokens_total": sum(record.get("retained_tokens", 0) for record in records)
                if any("retained_tokens" in record for record in records) else None,
            "processing_counts_observed": {key: sum(block.get(key, 0) for block in blocks)
                for key in _PROCESSING_COUNTS if any(key in block for block in blocks)},
            "processing_counts_scope": "sum_over_reported_calls; repeated_function_candidates_count_again",
            "processing_policies": [json.loads(value) for value in sorted({json.dumps(
                {key: block[key] for key in ("condition", "aggregation", "segmentation_identity", "protocol") if key in block},
                sort_keys=True) for block in blocks if any(key in block for key in ("condition", "aggregation", "protocol"))})],
            "items": records, "cache_reads_are_inference": False}


def _encoding_diagnostics_path(dataset: Dataset, adapter: Any, role: str, cache_dir: Path, text_only: bool) -> Path:
    identity = stable_hash({"encoding": encoding_cache_identity(dataset, adapter, role), "text_only": text_only})
    return Path(cache_dir) / "vectors" / identity / "diagnostics.json"


def _saved_encoding_diagnostics(dataset: Dataset, adapter: Any, role: str, cache_dir: Path, text_only: bool) -> dict:
    path = _encoding_diagnostics_path(dataset, adapter, role, cache_dir, text_only)
    count = len(dataset.corpus if role == "document" else dataset.queries)
    if path.is_file():
        return json.loads(path.read_text())
    result = _merge_diagnostics([], count)
    result["reason"] = "Legacy cache contains no processing diagnostics; counts are unknown, not zero"
    return result


def cached_encode(dataset: Dataset, adapter: Any, role: str, cache_dir: Path, *,
                  block_size: int = 128, text_only: bool = False) -> tuple[np.ndarray, dict]:
    if block_size < 1:
        raise ValueError("Encoding block size must be positive")
    items = dataset.corpus if role == "document" else dataset.queries
    identity = stable_hash({"encoding": encoding_cache_identity(dataset, adapter, role), "text_only": text_only})
    path = Path(cache_dir) / "vectors" / identity
    blocks, diagnostic_blocks, computed, cached, seconds = [], [], 0, 0, 0.0
    for start in range(0, len(items), block_size):
        stop = min(start + block_size, len(items))
        block_path = path / f"{start:09d}-{stop:09d}.npy"
        diagnostics_path = block_path.with_suffix(".usage.json")
        item_ids = [item["id"] for item in items[start:stop]]
        if block_path.is_file():
            values = np.load(block_path, allow_pickle=False)
            cached += stop - start
            diagnostics = json.loads(diagnostics_path.read_text()) if diagnostics_path.is_file() else {
                "rows": stop - start, "reported": False, "items": []}
            if diagnostics.get("rows") != stop - start:
                raise ValueError("Cached processing diagnostics have the wrong row count")
        else:
            tick = time.perf_counter()
            values = np.asarray(adapter.encode([dataset.model_item(item, text_only=text_only) for item in items[start:stop]], role=role),
                                dtype=np.float32)
            elapsed = time.perf_counter() - tick
            _validate_vectors(values, stop - start, adapter.dimension)
            diagnostics = _capture_diagnostics(getattr(adapter, "last_usage", getattr(adapter, "usage", {})), item_ids)
            _atomic_json(diagnostics_path, diagnostics)
            _atomic_array(block_path, values)
            computed += stop - start
            if not getattr(adapter, "precomputed", False):
                seconds += elapsed
        _validate_vectors(values, stop - start, adapter.dimension)
        blocks.append(values)
        diagnostic_blocks.append(diagnostics)
    values = np.concatenate(blocks, axis=0)
    diagnostics = _merge_diagnostics(diagnostic_blocks, len(items))
    _atomic_json(path / "diagnostics.json", diagnostics)
    return values, {"identity": identity, "role": role, "cache_rows": cached,
                    "new_rows": computed, "inference_seconds": seconds if computed and not getattr(adapter, "precomputed", False) else None,
                    "precomputed": bool(getattr(adapter, "precomputed", False)),
                    "adapter_diagnostics": diagnostics,
                    "cache_reads_are_inference": False}


def _code_tokens(text: str) -> list[str]:
    original = re.findall(r"\w+", text, re.UNICODE)
    parts = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text).replace("_", " ")
    return [token.casefold() for token in original] + [token.casefold() for token in re.findall(r"\w+", parts)]


def _bm25_rankings(dataset: Dataset, top_k: int) -> list[list[dict]]:
    import bm25s
    dataset.require_text()
    tokenize = _code_tokens if dataset.track == "code" else turkish_tokens
    tokens = [tokenize(row["text"]) for row in dataset.corpus]
    if not any(tokens):
        raise UnsupportedConfiguration("The corpus has no lexical terms")
    model = bm25s.BM25(k1=1.5, b=0.75, method="lucene", csc_backend="numpy")
    model.index(tokens, show_progress=False)
    document_ids = [row["id"] for row in dataset.corpus]
    identifiers = np.asarray(document_ids)
    results = []
    for query in dataset.queries:
        query_tokens = tokenize(dataset.model_item(query, text_only=True)["text"])
        if not query_tokens:
            results.append([])
            continue
        scores = np.asarray(model.get_scores(query_tokens))
        eligible = np.flatnonzero(scores > 0)
        excluded = dataset.excluded_ids(query)
        if excluded:
            eligible = np.asarray([i for i in eligible if document_ids[i] not in excluded], dtype=np.int64)
        order = np.lexsort((identifiers[eligible], -scores[eligible]))
        keep = eligible[order[:top_k]]
        results.append([{"id": document_ids[i], "score": float(scores[i]), "rank": rank}
                        for rank, i in enumerate(keep, 1)])
    return results


def _channel_identity(dataset: Dataset, channel: str, adapters: dict) -> str:
    if channel == "B":
        model = {"tokenizer": "code-identifiers-v1" if dataset.track == "code" else "turkish-unicode-v1",
                 "bm25": {"k1": 1.5, "b": 0.75, "method": "lucene"}, "packages": package_versions(("bm25s",)),
                 "candidate_selection": BM25_CANDIDATE_SELECTION}
    else:
        if channel not in adapters:
            raise UnsupportedConfiguration(f"No {channel} adapter configured")
        if hasattr(adapters[channel], "rank"):
            if not getattr(adapters[channel], "identity", None):
                raise ValueError("Direct ranking adapters require a full model/prompt identity")
            model = {"direct_ranking_adapter": adapters[channel].identity}
        else:
            model = {role: encoding_cache_identity(dataset, adapters[channel], role) for role in ("document", "query")}
    return stable_hash({"engine": ENGINE_VERSION, "dataset": dataset.identity, "channel": channel, "model": model})


def _direct_usage_public(usage: Any) -> dict:
    """Accept known direct-ranking measurements, not arbitrary adapter payloads."""
    if not isinstance(usage, dict):
        return {}
    result = {}
    for key in ("inference_seconds", "scoring_seconds", "exact_search_seconds", "segmentation_seconds", "exact", "total_items", "new_items",
                "cache_hits", "token_count_min", "token_count_max", "token_count_total", "truncated_text_items"):
        value = usage.get(key)
        if value is None or isinstance(value, (int, float, bool)) and math.isfinite(value):
            if key in usage:
                result[key] = value
    for key in ("adapter_identity", "model_revision", "base_revision", "search", "scoring", "score_device", "score_dtype", "token_mask"):
        value = usage.get(key)
        if isinstance(value, str) and re.fullmatch(r"[a-z0-9_.:-]{1,128}", value):
            result[key] = value
    for key in ("document_encoding", "query_encoding"):
        if key in usage:
            result[key] = _direct_usage_public(usage[key])
    result.update(_processing_diagnostics(usage))
    verification = usage.get("projection_verification", {})
    if isinstance(verification, dict) and isinstance(verification.get("projection_verified"), bool):
        result["projection_verification"] = {"projection_verified": verification["projection_verified"]}
        error = verification.get("max_absolute_merge_error")
        if isinstance(error, (int, float)) and math.isfinite(error):
            result["projection_verification"]["max_absolute_merge_error"] = error
    return result


def _cache_reused_usage(usage: dict) -> dict:
    result = {}
    for key, value in usage.items():
        if key.endswith("seconds"):
            result[key] = None
        elif key in {"new_rows", "new_items"}:
            result[key] = 0
        elif isinstance(value, dict):
            result[key] = _cache_reused_usage(value)
        else:
            result[key] = value
    return result


def base_rankings(dataset: Dataset, channel: str, adapters: dict, cache_dir: Path, *, top_k: int = 100) -> tuple[list, dict]:
    identity = _channel_identity(dataset, channel, adapters)
    path = Path(cache_dir) / "rankings" / stable_hash({"identity": identity, "top_k": top_k}) / "rankings.json"
    if path.is_file():
        payload = json.loads(path.read_text())
        rankings = payload["rankings"]
        if len(rankings) != len(dataset.queries):
            raise ValueError("Invalid ranking cache query count")
        usage = _cache_reused_usage(payload.get("usage", {}))
        if channel != "B" and not hasattr(adapters[channel], "rank"):
            for role in ("document", "query"):
                role_usage = usage.setdefault(role + "_encoding", {})
                role_usage.update(inference_seconds=None, new_rows=0,
                    cache_rows=len(dataset.corpus if role == "document" else dataset.queries))
                if "adapter_diagnostics" not in role_usage:
                    role_usage["adapter_diagnostics"] = _saved_encoding_diagnostics(
                        dataset, adapters[channel], role, cache_dir, channel in {"G", "E"})
        for role in ("document", "query"):
            role_usage = usage.get(role + "_encoding", {})
            if "total_items" in role_usage:
                role_usage["cache_hits"] = role_usage["total_items"]
        return rankings, {"identity": identity, "ranking_cache_hit": True, "inference_seconds": None,
                          "cache_reads_are_inference": False, **usage}
    tick = time.perf_counter()
    if channel == "B":
        rankings = _bm25_rankings(dataset, top_k)
        usage = {"bm25_build_and_search_seconds": time.perf_counter() - tick}
    elif hasattr(adapters[channel], "rank"):
        rankings, usage = adapters[channel].rank(
            [dataset.model_item(item) for item in dataset.corpus],
            [dataset.model_item(item) for item in dataset.queries], top_k=top_k,
            cache_dir=Path(cache_dir) / "direct" / identity,
            exclusions=[dataset.excluded_ids(query) for query in dataset.queries])
        usage = _direct_usage_public(usage)
        doc_ids = {item["id"] for item in dataset.corpus}
        if len(rankings) != len(dataset.queries):
            raise ValueError("Direct ranking adapter changed query count")
        for query, ranking in zip(dataset.queries, rankings):
            ids = [row["id"] for row in ranking]
            if len(ids) != len(set(ids)) or len(ids) > top_k or not set(ids) <= doc_ids:
                raise ValueError("Direct ranking adapter returned invalid candidate ids")
            if set(ids) & dataset.excluded_ids(query):
                raise ValueError("Direct ranking adapter returned an excluded candidate")
            if any(not math.isfinite(float(row["score"])) for row in ranking):
                raise ValueError("Direct ranking adapter returned a nonfinite score")
            ranking.sort(key=lambda row: (-float(row["score"]), row["id"]))
            for rank, row in enumerate(ranking, 1):
                row["rank"] = rank
    else:
        if channel in {"G", "E"}:
            dataset.require_text()
        elif channel == "J":
            dataset.require_candidate_text()
        documents, doc_usage = cached_encode(dataset, adapters[channel], "document", cache_dir, text_only=channel in {"G", "E"})
        queries, query_usage = cached_encode(dataset, adapters[channel], "query", cache_dir, text_only=channel in {"G", "E"})
        search_start = time.perf_counter()
        rankings = exact_dot_rankings(documents, queries, [row["id"] for row in dataset.corpus], top_k=top_k,
                                      exclusions=[dataset.excluded_ids(query) for query in dataset.queries])
        usage = {"document_encoding": doc_usage, "query_encoding": query_usage,
                 "exact_search_seconds": time.perf_counter() - search_start}
    _atomic_json(path, {"identity": identity, "query_ids": [q["id"] for q in dataset.queries], "rankings": rankings, "usage": usage})
    return rankings, {"identity": identity, "ranking_cache_hit": False, "cache_reads_are_inference": False, **usage}


class _RunStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS results (cell TEXT, query_id TEXT, payload TEXT, PRIMARY KEY(cell,query_id))")
        self.db.execute("CREATE TABLE IF NOT EXISTS scores (key TEXT PRIMARY KEY, payload TEXT)")

    def result(self, cell: str, query_id: str) -> dict | None:
        row = self.db.execute("SELECT payload FROM results WHERE cell=? AND query_id=?", (cell, query_id)).fetchone()
        return json.loads(row[0]) if row else None

    def put_result(self, cell: str, query_id: str, value: dict) -> None:
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO results VALUES (?,?,?)", (cell, query_id, json.dumps(value)))

    def score(self, key: str) -> Any:
        row = self.db.execute("SELECT payload FROM scores WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put_score(self, key: str, value: Any) -> None:
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO scores VALUES (?,?)", (key, json.dumps(value)))


def _rerank(dataset: Dataset, query: dict, candidate_ids: list[str], adapter: Any,
            store: _RunStore, corpus_by_id: dict, *, text_only: bool = False) -> tuple[list[dict], dict]:
    if not getattr(adapter, "identity", None):
        raise ValueError("A reranker must expose its full model/prompt identity")
    query_item = dataset.model_item(query, text_only=text_only)
    candidates = [dataset.model_item(corpus_by_id[identifier]) for identifier in candidate_ids]
    base = {"engine": ENGINE_VERSION, "dataset": dataset.identity, "adapter": adapter.identity,
            "query": stable_hash(query_item), "prompts": getattr(adapter, "prompt_identity", None)}
    pointwise = getattr(adapter, "pointwise", False)
    cached_count, seconds = 0, 0.0
    diagnostics = {"rows": 0, "reported": False, "items": []}
    if pointwise:
        keys = [stable_hash({**base, "candidate": stable_hash(item)}) for item in candidates]
        values = [store.score(key) for key in keys]
        missing = [i for i, value in enumerate(values) if value is None]
        cached_count = len(values) - len(missing)
        if missing:
            tick = time.perf_counter()
            fresh = list(adapter.score(query_item, [candidates[i] for i in missing]))
            seconds = time.perf_counter() - tick
            if len(fresh) != len(missing) or not all(math.isfinite(float(v)) for v in fresh):
                raise ValueError("Reranker returned invalid scores or changed candidate count")
            diagnostics = _capture_diagnostics(getattr(adapter, "last_usage", getattr(adapter, "usage", {})),
                                               [candidate_ids[i] for i in missing])
            for i, value in zip(missing, fresh):
                values[i] = float(value)
                store.put_score(keys[i], values[i])
    else:
        key = stable_hash({**base, "candidate_list": candidates})
        values = store.score(key)
        if values is None:
            tick = time.perf_counter()
            values = list(adapter.score(query_item, candidates))
            seconds = time.perf_counter() - tick
            if len(values) != len(candidates) or not all(math.isfinite(float(v)) for v in values):
                raise ValueError("Reranker returned invalid scores or changed candidate count")
            diagnostics = _capture_diagnostics(getattr(adapter, "last_usage", getattr(adapter, "usage", {})), candidate_ids)
            values = [float(v) for v in values]
            store.put_score(key, values)
        else:
            cached_count = len(candidates)
    scored = sorted(zip(candidate_ids, values), key=lambda row: (-row[1], row[0]))
    return [{"id": identifier, "score": float(score), "rank": rank}
            for rank, (identifier, score) in enumerate(scored, 1)], {
                "cache_pairs": cached_count, "new_pairs": len(candidates) - cached_count,
                "inference_seconds": seconds if len(candidates) > cached_count else None,
                "adapter_diagnostics": diagnostics,
                "adapter_diagnostics_scope": "fresh_score_pairs_only; cached_pair_processing_unknown",
                "cache_reads_are_inference": False}


def _write_report(output_dir: Path, report: dict) -> None:
    _atomic_json(output_dir / "report.json", report)
    rows = report["families"]
    path = output_dir / "matrix.csv"
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("variant_id", "track", "status", "reason", "completed_cells", "planned_cells"))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in writer.fieldnames})
    temporary.replace(path)


def run_matrix(dataset_dir: Path | str, output_dir: Path | str, *, adapters: dict | None = None,
               rerankers: dict | None = None, requested_channels: list[str] | tuple[str, ...] | None = None,
               candidate_grid=(20, 50, 100), budget_modes=("per_channel", "total"), channel_budget: int = 100,
               rrf_k: int = 60, query_limit: int | None = None, cache_dir: Path | str | None = None,
               progress_callback=None) -> dict:
    """Run all families for one frozen collection; preserve every unsupported cell.

    Raw rankings and resumable SQLite are under ``output_dir`` (normally ignored
    ``runs/``). ``report.json`` and ``matrix.csv`` contain metrics and IDs only and
    can be copied into public reports. No cached latency is replayed as inference.
    """
    dataset = load_dataset(dataset_dir)
    adapters, rerankers = adapters or {}, rerankers or {}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(cache_dir) if cache_dir else output_dir / "cache"
    canonical = tuple(TRACKS[dataset.track][1])
    requested = tuple(requested_channels) if requested_channels is not None else canonical
    if not requested or len(set(requested)) != len(requested) or not set(requested) <= set(canonical):
        raise ValueError("Requested channels must be a nonempty unique subset of the track")
    if not candidate_grid or any(k not in (20, 50, 100) for k in candidate_grid) or len(set(candidate_grid)) != len(candidate_grid):
        raise ValueError("Candidate grid must be a unique subset of 20, 50, 100")
    if not budget_modes or len(set(budget_modes)) != len(budget_modes):
        raise ValueError("Budget modes must be nonempty and unique")
    for mode in budget_modes:
        channel_allocations(requested, mode, channel_budget)
    if rrf_k < 0 or (query_limit is not None and query_limit < 1):
        raise ValueError("Invalid RRF constant or query limit")
    queries = dataset.queries[:query_limit] if query_limit else dataset.queries
    configuration = {"engine": ENGINE_VERSION, "dataset": dataset.identity, "candidate_grid": list(candidate_grid),
                     "budget_modes": list(budget_modes), "channel_budget": channel_budget, "rrf_k": rrf_k,
                     "requested_channels": list(requested), "query_ids": [query["id"] for query in queries]}
    report = {"schema_version": 1, "engine": ENGINE_VERSION, "dataset": dataset.public_summary(),
              "scope": "smoke" if len(queries) < len(dataset.queries) else "frozen_collection",
              "evaluated_query_count": len(queries), "configuration": configuration,
              "planned_global_family_count": 1284, "planned_global_retrieval_subset_count": 321,
              "cache_reads_are_inference": False, "channels": {}, "families": matrix_registry(dataset.track), "cells": []}
    # Descriptive provenance is deliberately outside cache-key configuration.
    report["adapter_specs"] = {"channels": {channel: public_adapter_spec(adapters[channel])
        for channel in requested if channel in adapters and channel != "B"},
        "rerankers": {name: public_adapter_spec(rerankers[name]) for name in RERANKERS if name in rerankers}}
    if "B" in requested:
        report["adapter_specs"]["channels"]["B"] = {"class": "bm25s.BM25", "config": {
            "k1": 1.5, "b": 0.75, "method": "lucene",
            "candidate_selection": BM25_CANDIDATE_SELECTION,
            "tokenizer": "code-identifiers-v1" if dataset.track == "code" else "turkish-unicode-v1"}}
    _write_report(output_dir, report)
    rankings, channel_states = {}, {}
    for channel in requested:
        try:
            rankings[channel], usage = base_rankings(dataset, channel, adapters, cache_dir, top_k=channel_budget)
            channel_states[channel] = {"status": "completed", **usage}
        except UnsupportedConfiguration as exc:
            channel_states[channel] = {"status": "unsupported", "reason": str(exc)}
        except Exception as exc:
            channel_states[channel] = {"status": "failed", "reason": type(exc).__name__ + "; see local failures.jsonl"}
            with (output_dir / "failures.jsonl").open("a") as handle:
                handle.write(json.dumps({"channel": channel, "error": repr(exc)}) + "\n")
        finally:
            if channel in adapters and hasattr(adapters[channel], "unload"):
                adapters[channel].unload()
        report["channels"] = channel_states
        if channel in adapters and channel != "B":
            report["adapter_specs"]["channels"][channel] = public_adapter_spec(adapters[channel])
        _write_report(output_dir, report)
        if progress_callback:
            progress_callback({"stage": "channel", "channel": channel, **channel_states[channel]})
    store = _RunStore(output_dir / "progress.sqlite3")
    corpus_by_id = {row["id"]: row for row in dataset.corpus}
    try:
        previous_reranker = None
        # Finish each model's work before loading the next GPU model. The report
        # retains canonical registry order, independent of execution scheduling.
        for family in sorted(report["families"], key=lambda row: RERANKERS.index(row["reranker"])):
            channels = family["retrieval_channels"]
            reranker_name = family["reranker"]
            if reranker_name != previous_reranker:
                previous_adapter = rerankers.get(previous_reranker)
                if previous_adapter is not None and hasattr(previous_adapter, "unload"):
                    previous_adapter.unload()
                previous_reranker = reranker_name
            grid = (None,) if reranker_name == "none" else candidate_grid
            family["planned_cells"] = len(grid) * len(budget_modes)
            family["completed_cells"] = 0
            if not set(channels) <= set(requested):
                family["reason"] = "Channels not requested in this invocation"
                continue
            unavailable = [channel for channel in channels if channel_states[channel]["status"] != "completed"]
            reason, unavailable_status = None, None
            if unavailable:
                unavailable_status = "failed" if any(channel_states[c]["status"] == "failed" for c in unavailable) else "unsupported"
                reason = "Unavailable retrieval channels: " + ",".join(unavailable)
            elif reranker_name != "none" and reranker_name not in rerankers:
                unavailable_status, reason = "unsupported", "No " + reranker_name + " adapter configured"
            elif reranker_name in {"laya_text", "bge_reranker_text"}:
                try:
                    dataset.require_text()
                except UnsupportedConfiguration as exc:
                    unavailable_status, reason = "unsupported", str(exc)
            for mode in budget_modes:
                for candidate_k in grid:
                    cell = {"variant_id": family["variant_id"], "budget_mode": mode, "candidate_k": candidate_k,
                            "status": "planned", "query_count": len(queries), "completed_queries": 0,
                            "primary": mode == "per_channel" and candidate_k in (None, 50)}
                    report["cells"].append(cell)
                    if unavailable_status:
                        cell.update(status=unavailable_status, reason=reason)
                        continue
                    reranker = rerankers.get(reranker_name)
                    cell_identity = stable_hash({**configuration, "variant": family["variant_id"], "mode": mode,
                                                 "candidate_k": candidate_k,
                                                 "channel_identities": {channel: channel_states[channel]["identity"] for channel in channels},
                                                 "reranker": getattr(reranker, "identity", None),
                                                 "reranker_prompts": getattr(reranker, "prompt_identity", None)})
                    cell["identity"] = cell_identity
                    metric_rows, pools, fresh_seconds, rerank_diagnostics, fresh_diagnostics = [], [], [], [], []
                    result_cache_hits, pair_cache_hits, new_pairs = 0, 0, 0
                    try:
                        for query_index, query in enumerate(queries):
                            result = store.result(cell_identity, query["id"])
                            if result is not None:
                                result_cache_hits += 1
                            else:
                                fused, pool = fuse_rankings({channel: rankings[channel][query_index] for channel in channels},
                                    channels, budget_mode=mode, budget=channel_budget, rrf_k=rrf_k)
                                candidates = fused if candidate_k is None else fused[:candidate_k]
                                usage = {}
                                if reranker is not None:
                                    candidates, usage = _rerank(dataset, query, [row["id"] for row in candidates], reranker,
                                        store, corpus_by_id, text_only=reranker_name in {"laya_text", "bge_reranker_text"})
                                result = {"query_id": query["id"], "ranking": candidates,
                                          "metrics": graded_metrics([row["id"] for row in candidates], dataset.qrels[query["id"]]),
                                          "pool": {**pool, "rerank_candidates": len(candidates) if reranker else 0}, "usage": usage}
                                store.put_result(cell_identity, query["id"], result)
                                pair_cache_hits += usage.get("cache_pairs", 0)
                                new_pairs += usage.get("new_pairs", 0)
                                if usage.get("new_pairs", 0) and usage.get("adapter_diagnostics"):
                                    fresh_diagnostics.append(usage["adapter_diagnostics"])
                                if usage.get("inference_seconds") is not None:
                                    fresh_seconds.append(usage["inference_seconds"])
                            metric_rows.append(result["metrics"])
                            pools.append(result["pool"])
                            if result.get("usage", {}).get("adapter_diagnostics"):
                                rerank_diagnostics.append(result["usage"]["adapter_diagnostics"])
                        cell.update(status="completed", completed_queries=len(metric_rows), metrics=aggregate_metrics(metric_rows),
                                    mean_raw_candidates=float(np.mean([p["raw_candidates"] for p in pools])),
                                    mean_unique_candidates=float(np.mean([p["unique_candidates"] for p in pools])),
                                    mean_rerank_candidates=float(np.mean([p["rerank_candidates"] for p in pools])),
                                    result_cache_hits=result_cache_hits, score_cache_pairs=pair_cache_hits, new_score_pairs=new_pairs,
                                    rerank_fresh_inference_seconds=sum(fresh_seconds) if fresh_seconds else None,
                                    rerank_fresh_p50_seconds=float(np.percentile(fresh_seconds, 50)) if fresh_seconds else None,
                                    rerank_fresh_p95_seconds=float(np.percentile(fresh_seconds, 95)) if fresh_seconds else None,
                                    latency_scope="fresh_missing_pairs_only; reused scores excluded")
                        if reranker is not None:
                            cell["adapter_diagnostics"] = _merge_diagnostics(rerank_diagnostics,
                                sum(pool["rerank_candidates"] for pool in pools))
                            cell["adapter_diagnostics"]["scope"] = "recorded_result_provenance; may_include_previous_invocations"
                            cell["fresh_adapter_diagnostics"] = _merge_diagnostics(fresh_diagnostics, new_pairs)
                            cell["fresh_adapter_diagnostics"]["scope"] = "current_invocation_fresh_score_pairs_only"
                        family["completed_cells"] += 1
                    except Exception as exc:
                        status = "unsupported" if isinstance(exc, UnsupportedConfiguration) else "failed"
                        cell.update(status=status, completed_queries=len(metric_rows), reason=str(exc) if status == "unsupported" else type(exc).__name__ + "; see local failures.jsonl")
                        with (output_dir / "failures.jsonl").open("a") as handle:
                            handle.write(json.dumps({"cell": cell_identity, "query_id": query["id"], "error": repr(exc)}) + "\n")
                    if progress_callback:
                        progress_callback({"stage": "cell", **cell})
            states = [cell["status"] for cell in report["cells"] if cell["variant_id"] == family["variant_id"]]
            family["status"] = "failed" if "failed" in states else "unsupported" if "unsupported" in states else "completed"
            if reason:
                family["reason"] = reason
            if reranker_name in rerankers:
                report["adapter_specs"]["rerankers"][reranker_name] = public_adapter_spec(rerankers[reranker_name])
            report["family_status_counts"] = {status: sum(row["status"] == status for row in report["families"])
                                               for status in ("completed", "planned", "failed", "unsupported")}
            _write_report(output_dir, report)
    finally:
        store.db.close()
        for adapter in rerankers.values():
            if hasattr(adapter, "unload"):
                adapter.unload()
    report["family_status_counts"] = {status: sum(row["status"] == status for row in report["families"])
                                      for status in ("completed", "planned", "failed", "unsupported")}
    report["cell_status_counts"] = {status: sum(row["status"] == status for row in report["cells"])
                                    for status in ("completed", "planned", "failed", "unsupported")}
    _write_report(output_dir, report)
    return report

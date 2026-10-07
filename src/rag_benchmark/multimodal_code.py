"""Shared lossless code segmentation with exact function-level maximum cosine.

This is an explicitly separate representation condition, never a replacement for
strict whole-function G/E results. Source functions and qrels remain unchanged.
Both pinned tokenizers validate the same boundaries after their native document
formatting. By default, a function fitting both encoders remains unsegmented.
An explicit character-bound condition also splits native-fit functions for
reranking; the consuming scorer must still validate its complete pair context.
"""
from __future__ import annotations

import argparse
import bisect
import copy
import gc
import hashlib
import json
import time
import uuid
from pathlib import Path

import numpy as np

from .models import DenseEmbedder, _snapshot, package_versions, stable_hash

_CODE_VERSION = "shared-code-segments-max-cosine-v1"
_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _digest_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix("." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class SharedCodeSegmenter:
    """One deterministic lossless boundary policy checked against both tokenizers."""
    def __init__(self, *, max_tokens: int = 8192, max_chunk_chars: int = 4096,
                 bge_config: dict | None = None, embeddinggemma_config: dict | None = None,
                 enforce_character_limit: bool = False):
        if type(max_tokens) is not int or not 1 <= max_tokens <= 8192:
            raise ValueError("max_tokens must be an integer in 1..8192.")
        if type(max_chunk_chars) is not int or max_chunk_chars < 1:
            raise ValueError("max_chunk_chars must be a positive integer.")
        if type(enforce_character_limit) is not bool:
            raise ValueError("enforce_character_limit must be a boolean.")
        self.max_tokens, self.max_chunk_chars = max_tokens, max_chunk_chars
        self.enforce_character_limit = enforce_character_limit
        self.formatters = {name: DenseEmbedder(name, {"max_length": max_tokens, **copy.deepcopy(config or {})})
            for name, config in [("bge", bge_config), ("embeddinggemma", embeddinggemma_config)]}
        if any(model.config["max_length"] != max_tokens for model in self.formatters.values()):
            raise ValueError("Both tokenizer validation budgets must match the shared max_tokens.")
        self._settings_seal = stable_hash({"max_tokens": max_tokens, "max_chunk_chars": max_chunk_chars,
            "enforce_character_limit": enforce_character_limit,
            "formatters": {name: model.config for name, model in self.formatters.items()}})
        self.tokenizers = {}
        self._memo = {}
        self.protocol = {
            "version": _CODE_VERSION, "condition": "source_complete_character_bounded_segments"
                if enforce_character_limit else "shared_segmentation_function_max_cosine",
            "max_tokens_after_native_document_formatting": max_tokens,
            "maximum_characters_per_chunk_for_overflowing_functions": max_chunk_chars,
            "boundary_policy": "last_complete_source_line_within_cap_else_character_boundary;halve_until_both_fit",
            "native_single_chunk_policy": "keep_whole_function_if_it_fits_both_tokenizers_and_character_cap"
                if enforce_character_limit else "keep_whole_function_if_it_fits_both_tokenizers",
            "source_coverage": "exact_concatenation_no_overlap_no_normalization_no_dropped_characters",
            "function_score": "assigned_by_consuming_adapter" if enforce_character_limit else "maximum_cosine_over_all_its_chunks",
            "tokenizers": {name: {"model_id": model.config["model_id"], "revision": model.config["revision"]}
                for name, model in self.formatters.items()},
            "native_document_formats": {name: model.format_document({"text": "<SOURCE>", "title": "<TITLE>"})
                for name, model in self.formatters.items()},
        }
        self.identity = stable_hash({"protocol": self.protocol, "source_sha256": _SOURCE_SHA256,
            "packages": package_versions(("transformers", "tokenizers"))})
        self._protocol_seal = stable_hash(self.protocol)

    def _load_tokenizers(self):
        if not self.tokenizers:
            from transformers import AutoTokenizer
            self.tokenizers = {name: AutoTokenizer.from_pretrained(str(_snapshot(model.config, download=False)),
                local_files_only=True, trust_remote_code=False) for name, model in self.formatters.items()}
        return self.tokenizers

    def token_counts(self, text: str, title: str = "") -> dict[str, int]:
        tokenizers = self._load_tokenizers()
        return {name: len(tokenizer(self.formatters[name].format_document({"text": text, "title": title}),
                                    truncation=False, add_special_tokens=True)["input_ids"])
                for name, tokenizer in tokenizers.items()}

    def partition(self, item: dict) -> tuple[list[dict], dict]:
        settings = {"max_tokens": self.max_tokens, "max_chunk_chars": self.max_chunk_chars,
            "enforce_character_limit": self.enforce_character_limit,
            "formatters": {name: model.config for name, model in self.formatters.items()}}
        if stable_hash(self.protocol) != self._protocol_seal or stable_hash(settings) != self._settings_seal:
            raise ValueError("Shared segmentation protocol changed after its identity was fixed.")
        text = item.get("text")
        if not isinstance(text, str) or item.get("media"):
            raise ValueError("Code segmentation requires an explicit source-text string and no media.")
        title = item.get("title") or ""
        if not isinstance(title, str):
            raise ValueError("Code source title must be a string.")
        source_hash = _digest_text(text)
        key = stable_hash({"text": source_hash, "title": title})
        if key in self._memo:
            boundaries, original_counts = self._memo[key]
        else:
            original_counts = self.token_counts(text, title)
            boundaries = []
            if max(original_counts.values()) <= self.max_tokens and (
                    not self.enforce_character_limit or len(text) <= self.max_chunk_chars):
                boundaries.append((0, len(text), original_counts))
            else:
                # Existing line endings, including CRLF, are preserved exactly.
                line_ends, total = [], 0
                for line in text.splitlines(keepends=True):
                    total += len(line)
                    line_ends.append(total)
                start = 0
                while start < len(text):
                    end = min(start + self.max_chunk_chars, len(text))
                    while True:
                        position = bisect.bisect_right(line_ends, end) - 1
                        if position >= 0 and line_ends[position] > start:
                            end = line_ends[position]
                        counts = self.token_counts(text[start:end], title)
                        if max(counts.values()) <= self.max_tokens:
                            break
                        if end - start <= 1:
                            raise ValueError("Even one source character plus the native document prefix/title exceeds a tokenizer budget.")
                        end = start + max(1, (end - start) // 2)
                    boundaries.append((start, end, counts))
                    start = end
            self._memo[key] = (boundaries, original_counts)
        chunks, byte_start = [], 0
        for index, (start, end, counts) in enumerate(boundaries):
            piece = text[start:end]
            byte_end = byte_start + len(piece.encode("utf-8"))
            chunks.append({"id": stable_hash({"function_id": item.get("id"), "source": source_hash, "chunk": index}),
                "function_id": item.get("id"), "chunk_index": index, "text": piece, "title": title,
                "char_start": start, "char_end": end, "byte_start": byte_start, "byte_end": byte_end,
                "token_counts": dict(counts), "text_sha256": _digest_text(piece)})
            byte_start = byte_end
        if not chunks or "".join(chunk["text"] for chunk in chunks) != text:
            raise RuntimeError("Code segmentation did not preserve the entire original function.")
        summary = {"function_id": item.get("id"), "source_sha256": source_hash,
            "source_characters": len(text), "source_utf8_bytes": len(text.encode("utf-8")),
            "whole_function_token_counts": original_counts, "chunk_count": len(chunks),
            "segmented": len(chunks) > 1, "chunks": [{k: v for k, v in chunk.items() if k not in {"text", "title"}}
                                                       for chunk in chunks]}
        return chunks, summary


def _check_vectors(values, rows: int, dimension: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.shape != (rows, dimension) or not np.isfinite(values).all():
        raise RuntimeError("Code encoder returned invalid vectors or changed item count.")
    if not np.allclose(np.linalg.norm(values, axis=1), 1, atol=2e-5):
        raise RuntimeError("Code embeddings must be nonzero unit vectors for maximum cosine scoring.")
    return values


class SegmentedCodeAdapter:
    """Native dense encoder, shared source chunks, original-function exact ranking."""
    def __init__(self, name: str, config: dict | None = None, *, segmenter: SharedCodeSegmenter | None = None):
        if name not in {"bge", "embeddinggemma"}:
            raise ValueError("Segmented code supports the explicit bge or embeddinggemma encoder.")
        self.name = name
        self.segmenter = segmenter or SharedCodeSegmenter()
        self.encoder = DenseEmbedder(name, {"device": "mps", "dtype": "bfloat16", "batch_size": 4,
            "max_length": self.segmenter.max_tokens, **copy.deepcopy(config or {})})
        expected = self.segmenter.protocol["tokenizers"][name]
        if any(self.encoder.config[key] != expected[key] for key in ("model_id", "revision")):
            raise ValueError("Encoder revision differs from the shared segmentation tokenizer revision.")
        if int(self.encoder.config.get("max_length", 8192)) != self.segmenter.max_tokens:
            raise ValueError("Encoder and shared segmentation context budgets must match.")
        self.dimension = self.encoder.dimension
        self.identity = stable_hash({"adapter": _CODE_VERSION, "encoder": self.encoder.identity,
            "segmentation": self.segmenter.identity, "source_sha256": _SOURCE_SHA256,
            "query_format": self.encoder.format_query("<QUERY>"), "score_dtype": "float32"})
        self._encoder_config_seal = stable_hash(self.encoder.config)
        self.last_usage = {}

    def unload(self):
        had_model = self.encoder._model is not None
        self.encoder._model = None
        gc.collect()
        if had_model and self.encoder.config["device"].startswith("mps"):
            import torch
            torch.mps.empty_cache()

    def _cached_vectors(self, items: list[dict], role: str, cache_dir: Path) -> tuple[np.ndarray, dict]:
        directory = cache_dir / "vectors" / role
        directory.mkdir(parents=True, exist_ok=True)
        arrays, fresh, inference_seconds = [], 0, 0.0
        for offset in range(0, len(items), 32):
            batch = items[offset:offset + 32]
            content = [self.encoder.format_query(x["text"]) if role == "query"
                       else self.encoder.format_document(x) for x in batch]
            key = stable_hash({"adapter": self.identity, "role": role, "formatted_inputs": content})
            path = directory / (key + ".npy")
            if path.is_file():
                vectors = _check_vectors(np.load(path, allow_pickle=False), len(batch), self.dimension)
            else:
                tick = time.perf_counter()
                vectors = self.encoder._encode(content) if role == "query" else self.encoder.embed_documents(batch)
                vectors = _check_vectors(vectors, len(batch), self.dimension)
                inference_seconds += time.perf_counter() - tick
                fresh += len(batch)
                temporary = path.with_suffix("." + uuid.uuid4().hex + ".tmp")
                try:
                    with temporary.open("wb") as handle:
                        np.save(handle, vectors, allow_pickle=False)
                    temporary.replace(path)
                finally:
                    temporary.unlink(missing_ok=True)
            arrays.append(vectors)
        result = np.concatenate(arrays) if arrays else np.empty((0, self.dimension), dtype=np.float32)
        return result, {"total_items": len(items), "new_items": fresh, "cache_hits": len(items) - fresh,
            "inference_seconds": inference_seconds if fresh else None, "cache_reads_are_inference": False}

    def rank(self, documents: list[dict], queries: list[dict], top_k: int, cache_dir: Path,
             exclusions: list[set[str]] | None = None) -> tuple[list[list[dict]], dict]:
        if stable_hash(self.encoder.config) != self._encoder_config_seal:
            raise ValueError("Code encoder configuration changed after its identity was fixed.")
        if type(top_k) is not int or top_k < 1:
            raise ValueError("top_k must be a positive integer.")
        ids = [item["id"] for item in documents]
        if any(not isinstance(identifier, str) for identifier in ids) or len(ids) != len(set(ids)):
            raise ValueError("Original function IDs must be unique strings.")
        if any(not isinstance(item.get("text"), str) or item.get("media") for item in queries):
            raise ValueError("Code queries require text without media.")
        exclusions = exclusions if exclusions is not None else [set() for _ in queries]
        if len(exclusions) != len(queries):
            raise ValueError("There must be one exclusion set per query.")
        chunks, offsets, summaries = [], [], []
        segmentation_start = time.perf_counter()
        for item in documents:
            offsets.append(len(chunks))
            parts, summary = self.segmenter.partition(item)
            chunks.extend(parts)
            summaries.append(summary)
        segmentation_seconds = time.perf_counter() - segmentation_start
        cache_dir = Path(cache_dir)
        _atomic_json(cache_dir / "shared-segmentation.json", {"identity": self.segmenter.identity,
            "protocol": self.segmenter.protocol, "functions": summaries})
        document_vectors, document_usage = self._cached_vectors(chunks, "document", cache_dir)
        query_vectors, query_usage = self._cached_vectors(queries, "query", cache_dir)
        self.unload()
        rankings = []
        tick = time.perf_counter()
        for start in range(0, len(queries), 32):
            chunk_scores = query_vectors[start:start + 32] @ document_vectors.T
            function_scores = np.maximum.reduceat(chunk_scores, offsets, axis=1) if offsets else np.empty((len(chunk_scores), 0))
            for query_index, row in enumerate(function_scores, start):
                available = [index for index, identifier in enumerate(ids) if identifier not in exclusions[query_index]]
                positions = sorted(available, key=lambda index: (-float(row[index]), ids[index]))[:top_k]
                rankings.append([{"id": ids[index], "score": float(row[index])} for index in positions])
        usage = {"condition": self.segmenter.protocol["condition"], "segmentation_identity": self.segmenter.identity,
            "protocol": self.segmenter.protocol, "original_function_count": len(documents), "chunk_count": len(chunks),
            "segmented_function_count": sum(row["segmented"] for row in summaries),
            "unchanged_function_count": sum(not row["segmented"] for row in summaries),
            "covered_source_characters": sum(len(chunk["text"]) for chunk in chunks),
            "source_characters": sum(len(item["text"]) for item in documents),
            "source_utf8_bytes": sum(row["source_utf8_bytes"] for row in summaries),
            "covered_source_utf8_bytes": sum(len(chunk["text"].encode("utf-8")) for chunk in chunks),
            "original_query_count": len(queries),
            "segmentation_seconds": segmentation_seconds, "document_encoding": document_usage,
            "query_encoding": query_usage, "scoring_seconds": time.perf_counter() - tick,
            "score_dtype": "float32", "scoring": "max_chunk_cosine_per_original_function", "exact": True}
        self.last_usage = usage
        return rankings, usage


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CPU tokenizer-only shared segmentation preflight; no model inference.")
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports/code-segmentation-tokenizer-preflight.json"))
    args = parser.parse_args(argv)
    segmenter = SharedCodeSegmenter()
    records, count, chunks, characters, source_bytes = [], 0, 0, 0, 0
    maximum_tokens = {"bge": 0, "embeddinggemma": 0}
    started = time.perf_counter()
    with args.corpus.open() as handle:
        for line in handle:
            item = json.loads(line)
            parts, summary = segmenter.partition(item)
            count += 1
            chunks += len(parts)
            characters += len(item["text"])
            source_bytes += summary["source_utf8_bytes"]
            for part in parts:
                for name, length in part["token_counts"].items():
                    maximum_tokens[name] = max(maximum_tokens[name], length)
            if summary["segmented"]:
                records.append(summary)
    result = {"purpose": "actual_pinned_tokenizer_context_validation_only_no_retrieval_accuracy",
        "status": "ok", "protocol": segmenter.protocol, "segmentation_identity": segmenter.identity,
        "original_function_count": count, "chunk_count": chunks, "segmented_function_count": len(records),
        "unchanged_function_count": count - len(records), "preserved_source_characters": characters,
        "preserved_source_utf8_bytes": source_bytes, "source_coverage_verified": True,
        "maximum_document_tokens_by_model": maximum_tokens,
        "tokenizer_validation_seconds": time.perf_counter() - started, "segmented_functions": records}
    _atomic_json(args.output, result)
    print(json.dumps({key: result[key] for key in ["status", "original_function_count", "chunk_count", "segmented_function_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

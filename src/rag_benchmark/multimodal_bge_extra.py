"""Staged BGE sparse/ColBERT math and extraction. No real runtime backend or CLI.

The main G adapter is untouched. An injected CPU backend supplies exactly one
hidden-state pass; production model execution remains an explicit prerequisite.
Head projection, normalization, scoring and token storage use float32. A real
backend must separately verify that its hidden-state precision reproduces the
existing dense baseline under this strict CLS check. No input is truncated.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np

from .models import MODEL_DEFAULTS, stable_hash
from .multimodal import _atomic_array, _atomic_json

MODEL_REVISION = MODEL_DEFAULTS["bge"]["revision"]
DIMENSION = 1024
VOCABULARY = 250002
FORMULA_COMMIT = "fd1a2bdf69488ffebe0327999d4400d8c8058a0b"
FORMULA_HASHES = {
    "inference": "e795219b34d96340c7adaf4902f771aaca9246448f9d1dd334b1757cb0c1f978",
    "modeling": "ef9f37f511cf9b7ceaf89a2bd569276ca0058cc8dc401545b9b4b688c9801f4f",
    "runner": "a2e683bb671610d8809748554e4b35115df601eb418728d5da1bdda761a601f1",
}
HEAD_HASHES = {
    "sparse_linear.pt": "45c93804d2142b8f6d7ec6914ae23a1eee9c6a1d27d83d908a20d2afb3595ad9",
    "colbert_linear.pt": "19bfbae397c2b7524158c919d0e9b19393c5639d098f0a66932c91ed8f5f9abb",
}
_IMPLEMENTATION = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _array(value, *, ndim=None):
    value = np.asarray(value, dtype=np.float32)
    if ndim is not None and value.ndim != ndim or not np.isfinite(value).all():
        raise ValueError("Invalid finite array shape")
    return value


def normalize_tokens(values):
    """Official normalize epsilon, preserving legitimately all-zero vectors."""
    values = _array(values, ndim=2)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(norms).all():
        raise ValueError("Normalization arithmetic became nonfinite")
    return _array(values / np.maximum(norms, np.float32(1e-12)), ndim=2)


def lexical_weights(hidden, token_ids, weight, bias, *, special_ids=(0, 1, 2, 3)):
    hidden, weight, bias = _array(hidden, ndim=2), _array(weight, ndim=2), _array(bias, ndim=1)
    ids = np.asarray(token_ids)
    if ids.shape != (len(hidden),) or ids.dtype.kind not in "iu" or np.any(ids < 0) or np.any(ids >= VOCABULARY):
        raise ValueError("Lexical input IDs differ from full token rows")
    if weight.shape != (1, hidden.shape[1]) or bias.shape != (1,):
        raise ValueError("Sparse trained head shape differs")
    scores = np.maximum((hidden @ weight.T).reshape(-1) + bias[0], np.float32(0))
    if not np.isfinite(scores).all():
        raise ValueError("Sparse projection arithmetic became nonfinite")
    output = {}
    for token, score in zip(ids, scores, strict=True):
        if int(token) not in special_ids and score > 0:
            output[int(token)] = max(output.get(int(token), 0.0), float(score))
    return output


def projected_tokens(hidden, weight, bias):
    """Input is already unpadded; drop CLS only, retain the actual EOS vector."""
    hidden, weight, bias = _array(hidden, ndim=2), _array(weight, ndim=2), _array(bias, ndim=1)
    if len(hidden) < 2 or weight.shape[1] != hidden.shape[1] or bias.shape != (weight.shape[0],):
        raise ValueError("ColBERT trained head/input shape differs")
    return normalize_tokens(hidden[1:] @ weight.T + bias)


def sparse_score(query, document):
    for weights in (query, document):
        if any(type(token) is not int or token < 0 or token >= VOCABULARY
               or not np.isfinite(value) or value < 0 for token, value in weights.items()):
            raise ValueError("Sparse weights must be finite, nonnegative native token IDs")
    score = float(sum(value * document.get(token, 0.0) for token, value in query.items()))
    if not np.isfinite(score):
        raise ValueError("Sparse score arithmetic became nonfinite")
    return score


def maxsim_score(query, document, *, max_score_cells=65536):
    """Exact mean-MaxSim with a hard temporary dot-matrix cell bound; no zero clamp."""
    query, document = _array(query, ndim=2), _array(document, ndim=2)
    if not len(query) or not len(document) or query.shape[1] != document.shape[1]:
        raise ValueError("MaxSim requires nonempty compatible token matrices")
    if type(max_score_cells) is not int or max_score_cells < 1:
        raise ValueError("MaxSim memory bound must be positive")
    best = np.full(len(query), -np.inf, dtype=np.float32)
    query_block = min(len(query), max(1, int(np.sqrt(max_score_cells))))
    for start in range(0, len(query), query_block):
        current = query[start:start + query_block]
        document_block = max(1, max_score_cells // len(current))
        for offset in range(0, len(document), document_block):
            scores = _array(current @ document[offset:offset + document_block].T, ndim=2)
            best[start:start + len(current)] = np.maximum(best[start:start + len(current)], scores.max(axis=1))
    score = float(np.mean(best, dtype=np.float32))
    if not np.isfinite(score):
        raise ValueError("MaxSim reduction arithmetic became nonfinite")
    return score


def rank_sources(query, chunks, *, mode, top_k=100, exclusions=(), max_score_cells=65536):
    """Full gallery; same source's chunk scores take max, including negative scores."""
    if mode not in {"sparse", "colbert"} or type(top_k) is not int or top_k < 1:
        raise ValueError("Explicit head mode and positive top_k required")
    scores = {}
    for chunk in chunks:
        source = chunk["source_id"]
        if not isinstance(source, str) or not source:
            raise ValueError("Original source identity required")
        value = sparse_score(query["sparse"], chunk["sparse"]) if mode == "sparse" else maxsim_score(
            query["colbert"], chunk["colbert"], max_score_cells=max_score_cells)
        scores[source] = max(scores.get(source, -np.inf), value)
    return [{"id": source, "score": float(score), "rank": rank}
            for rank, (source, score) in enumerate(sorted(
                ((source, score) for source, score in scores.items() if source not in exclusions),
                key=lambda pair: (-pair[1], pair[0]))[:top_k], 1)]


def load_trained_heads(directory):
    """Lazy future runtime path; never called by CPU fake tests or module import."""
    paths = {name: Path(directory) / name for name in HEAD_HASHES}
    for name, path in paths.items():
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != HEAD_HASHES[name]:
            raise ValueError("Missing or incompatible pinned trained head; random initialization forbidden")
    import torch
    heads = {name: torch.load(path, map_location="cpu", weights_only=True) for name, path in paths.items()}
    for name, rows in (("sparse_linear.pt", 1), ("colbert_linear.pt", DIMENSION)):
        if set(heads[name]) != {"weight", "bias"} or tuple(heads[name]["weight"].shape) != (rows, DIMENSION) or tuple(heads[name]["bias"].shape) != (rows,):
            raise ValueError("Pinned trained head tensor shape differs")
        if not all(torch.isfinite(value).all() for value in heads[name].values()):
            raise ValueError("Trained head has nonfinite tensors")
    return heads


class ExtraHeadExtractor:
    """Injected staged backend only. Real model/runtime bridge is not implemented."""
    def __init__(self, *, backend, config, head_weights, source_identity):
        if backend is None:
            raise NotImplementedError("Reviewed production hidden-state backend is still pending")
        if any(config.get(key) != MODEL_DEFAULTS["bge"][key] for key in ("model_id", "revision", "dimension")):
            raise ValueError("Pinned native BGE model contract required")
        if config.get("max_length") != 8192 or config.get("token_vector_storage") != "float32":
            raise ValueError("Strict native 8192 cap and explicit float32 token storage required")
        if not isinstance(source_identity, str) or not source_identity or not isinstance(backend.identity, str) or not backend.identity:
            raise ValueError("Frozen source and backend identity required")
        self.backend, self.config, self.heads = backend, dict(config), head_weights
        sw, sb = head_weights["sparse"]
        cw, cb = head_weights["colbert"]
        if _array(sw).shape != (1, DIMENSION) or _array(sb).shape != (1,) or _array(cw).shape != (DIMENSION, DIMENSION) or _array(cb).shape != (DIMENSION,):
            raise ValueError("Both native head shapes required")
        self.identity = stable_hash({"version": "bge-extra-staged-v1", "config": self.config,
            "backend": backend.identity, "source": source_identity, "formula_commit": FORMULA_COMMIT,
            "formula_files": FORMULA_HASHES, "trained_head_artifacts": HEAD_HASHES,
            "injected_head_arrays": {key: [hashlib.sha256(np.ascontiguousarray(_array(part)).tobytes()).hexdigest()
                for part in value] for key, value in head_weights.items()}, "implementation": _IMPLEMENTATION,
            "function_aggregation": "maximum_chunk_score", "head_compute_dtype": "float32",
            "dense_comparison": "float32_normalized_actual_cls", "dense_comparison_atol": 2e-6})
        self._seal = self._state()

    def _state(self):
        return stable_hash({"config": self.config, "backend": self.backend.identity,
            "heads": {key: [hashlib.sha256(np.ascontiguousarray(_array(part)).tobytes()).hexdigest()
                for part in value] for key, value in self.heads.items()}})

    def extract(self, records, baseline_dense, *, cache_dir):
        if self._state() != self._seal:
            raise ValueError("Extractor config/runtime/heads changed")
        input_seal = stable_hash(records)
        baseline_seal = hashlib.sha256(_array(baseline_dense, ndim=2).tobytes()).hexdigest()

        def verify_state():
            if self._state() != self._seal or stable_hash(records) != input_seal:
                raise ValueError("Extractor config/runtime/heads/input changed")
            if hashlib.sha256(_array(baseline_dense, ndim=2).tobytes()).hexdigest() != baseline_seal:
                raise ValueError("Reused baseline changed during extraction")

        ids = [record["id"] for record in records]
        if len(ids) != len(set(ids)) or any(not isinstance(identifier, str) or not identifier for identifier in ids):
            raise ValueError("Ordered unique extraction row IDs required")
        raw_tokens = [self.backend.tokenize(record["formatted_text"]) for record in records]
        tokens = []
        for ids_row in raw_tokens:
            array = np.asarray(ids_row)
            if not 2 <= len(ids_row) <= 8192 or array.ndim != 1 or array.dtype.kind not in "iu" or np.any(array < 0) or np.any(array >= VOCABULARY) or ids_row[0] != 0 or ids_row[-1] != 2:
                raise ValueError("Full native token IDs exceed limit or special-token contract")
            tokens.append([int(token) for token in array])
        verify_state()
        baseline = _array(baseline_dense, ndim=2).copy()
        if baseline.shape != (len(records), DIMENSION):
            raise ValueError("Reused dense baseline source coverage/shape differs")
        key = stable_hash({"extractor": self.identity, "records": records, "token_ids": tokens,
                           "dense_baseline_sha256": hashlib.sha256(baseline.tobytes()).hexdigest()})
        directory = Path(cache_dir) / key
        manifest_path = directory / "manifest.json"
        if directory.exists():
            if not manifest_path.is_file():
                raise ValueError("Incomplete extra-head cache; implicit model retry forbidden")
            manifest_bytes = manifest_path.read_bytes()
            checksum_path = directory / "manifest.sha256"
            if not checksum_path.is_file() or checksum_path.read_text().strip() != hashlib.sha256(manifest_bytes).hexdigest():
                raise ValueError("Extra-head manifest checksum differs; implicit model retry forbidden")
            manifest = json.loads(manifest_bytes)
            if not isinstance(manifest.get("rows"), list) or len(manifest["rows"]) != len(records):
                raise ValueError("Extra-head cache row coverage differs")
            if manifest.get("key") != key or manifest.get("records_sha256") != stable_hash(records):
                raise ValueError("Extra-head cache input identity differs")
            output = []
            for index, row in enumerate(manifest["rows"]):
                if row["file"] != f"{index:09d}.npy":
                    raise ValueError("Extra-head cache file identity differs")
                path = directory / row["file"]
                data = path.read_bytes()
                if hashlib.sha256(data).hexdigest() != row["sha256"]:
                    raise ValueError("Extra-head cache checksum differs; implicit model retry forbidden")
                vectors = np.load(io.BytesIO(data), allow_pickle=False)
                if vectors.shape != (len(tokens[index])-1, DIMENSION) or vectors.dtype != np.float32 or not np.isfinite(vectors).all():
                    raise ValueError("Extra-head ragged cache token coverage differs")
                sparse = {int(token): value for token, value in row["sparse"].items()}
                sparse_score(sparse, {})
                output.append({"id": records[index]["id"], "source_id": records[index]["source_id"],
                    "sparse": sparse, "colbert": vectors})
            if len(output) != len(records):
                raise ValueError("Extra-head cache row coverage differs")
            verify_state()
            return output, {"hidden_forward_calls": 0, "cached_rows": len(records), "new_rows": 0}
        record_seal = stable_hash(records)
        token_seal = stable_hash(tokens)
        directory.mkdir(parents=True)
        _atomic_json(directory / "pending.json", {"key": key, "status": "pending"})
        inputs = self.backend.forward(tokens)
        verify_state()
        if stable_hash(records) != record_seal or stable_hash(tokens) != token_seal:
            raise ValueError("Extraction input changed during hidden forward")
        hidden, actual_ids, masks = _array(inputs["hidden"], ndim=3), np.asarray(inputs["token_ids"]), np.asarray(inputs["attention_mask"])
        if actual_ids.shape != masks.shape or hidden.shape[:2] != masks.shape or hidden.shape[2] != DIMENSION or len(hidden) != len(records):
            raise ValueError("Hidden pass changed full input tensor dimensions")
        if any(not np.array_equal(actual_ids[index][masks[index].astype(bool)], row) for index, row in enumerate(tokens)) or not np.isin(masks, [0, 1]).all():
            raise ValueError("Actual model tokens/masks silently dropped or changed input")
        for index, row in enumerate(tokens):
            expected_mask = np.arange(masks.shape[1]) < len(row)
            if not np.array_equal(masks[index].astype(bool), expected_mask) or np.any(actual_ids[index][~expected_mask] != 1):
                raise ValueError("Native right-padding and position-zero CLS contract violated")
        native_dense = normalize_tokens(hidden[:, 0])
        if "native_dense" in inputs:
            supplied_dense = _array(inputs["native_dense"], ndim=2)
            if supplied_dense.shape != native_dense.shape or not np.allclose(supplied_dense, native_dense, atol=2e-6, rtol=0):
                raise ValueError("Backend dense override differs from actual normalized CLS")
        if native_dense.shape != baseline.shape or not np.allclose(native_dense, baseline, atol=2e-6, rtol=0):
            raise ValueError("Incidental CLS output differs from reused dense baseline")
        if self._state() != self._seal:
            raise ValueError("Extractor config/runtime/heads changed during hidden forward")
        output, rows = [], []
        sw, sb = self.heads["sparse"]
        cw, cb = self.heads["colbert"]
        for index, (record, ids_row) in enumerate(zip(records, tokens, strict=True)):
            states = hidden[index][masks[index].astype(bool)]
            sparse = lexical_weights(states, ids_row, sw, sb)
            vectors = projected_tokens(states, cw, cb)
            path = directory / f"{index:09d}.npy"
            _atomic_array(path, vectors)
            rows.append({"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "sparse": sparse})
            output.append({"id": record["id"], "source_id": record["source_id"], "sparse": sparse, "colbert": vectors})
        verify_state()
        _atomic_json(manifest_path, {"key": key, "records_sha256": stable_hash(records), "rows": rows,
            "token_storage": "float32", "dense_baseline_reused": True})
        (directory / "manifest.sha256").write_text(hashlib.sha256(manifest_path.read_bytes()).hexdigest())
        verify_state()
        return output, {"hidden_forward_calls": 1, "cached_rows": 0, "new_rows": len(records)}

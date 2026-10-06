"""Offline ColQwen2.5 token embeddings and exact MaxSim document retrieval.

Both LoRA adapter and base checkpoint are immutable. The local loading overlay
pins the adapter's otherwise unpinned base_model_name_or_path without changing
its downloaded checkpoint or weights. No token pooling or ANN is used.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .models import _torch_device, package_versions, stable_hash, synchronize, validate_revision
from .multimodal_models import _local_media

COLQWEN_DEFAULTS = {
    "model_id": "vidore/colqwen2.5-v0.2",
    "revision": "dcbe8d9cede518bce830488364ba0e40c873645b",
    "base_model_id": "vidore/colqwen2.5-base",
    "base_revision": "92908120384b7a2110c5beda3ab29cbdb2c08e49",
    "device": "mps", "score_device": "mps", "dtype": "bfloat16", "batch_size": 1,
    "max_length": 8192, "max_image_patches": 768, "score_query_batch": 8,
    "score_chunk_elements": 16_000_000, "cache_dir": ".cache/models", "local_files_only": True,
    "query_format": "checkpoint_sentence_transformers_Query_prefix_plus_10_endoftext",
    "document_format": "checkpoint_sentence_transformers_Describe_the_image",
    "token_mask": "checkpoint_default_all_nonpadding", "scoring": "sum_query_token_max_document_token_dot",
    "token_normalization": "float32_l2", "score_dtype": "float32",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pinned_snapshots(config: dict, *, download: bool = False) -> tuple[Path, Path]:
    from huggingface_hub import snapshot_download
    patterns = ["*.json", "*.txt", "*.model", "*.jinja", "*.safetensors"]
    paths = []
    for model_id, revision in [(config["model_id"], config["revision"]), (config["base_model_id"], config["base_revision"])]:
        validate_revision({"model_id": model_id, "revision": revision})
        paths.append(Path(snapshot_download(model_id, revision=revision,
            cache_dir=str(Path(config["cache_dir"]).resolve()), allow_patterns=patterns,
            local_files_only=not download)))
    return paths[0], paths[1]


def local_overlay(adapter: Path, base: Path, config: dict) -> Path:
    """Symlink immutable assets, rewrite only base location, record exact provenance."""
    original = json.loads((adapter / "adapter_config.json").read_text())
    if original.get("base_model_name_or_path") != config["base_model_id"]:
        raise ValueError("LoRA base model identity differs from the explicitly pinned base model.")
    name = stable_hash({"adapter_revision": config["revision"], "base_revision": config["base_revision"],
        "adapter_config_sha256": _sha256(adapter / "adapter_config.json"), "overlay_version": 1})
    destination = Path(config["cache_dir"]).resolve() / "colqwen-local-overlays" / name
    destination.mkdir(parents=True, exist_ok=True)
    for source in adapter.rglob("*"):
        if not source.is_file() or source.name == "adapter_config.json":
            continue
        target = destination / source.relative_to(adapter)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if not target.is_symlink() or target.resolve() != source.resolve():
                raise ValueError("Existing ColQwen overlay asset does not match immutable source.")
        else:
            target.symlink_to(source.resolve())
    rewritten = {**original, "base_model_name_or_path": str(base.resolve()), "revision": config["base_revision"]}
    path = destination / "adapter_config.json"
    if path.exists() and json.loads(path.read_text()) != rewritten:
        raise ValueError("Existing ColQwen overlay points to a different base artifact.")
    path.write_text(json.dumps(rewritten, sort_keys=True, indent=2) + "\n")
    provenance = {"model_id": config["model_id"], "revision": config["revision"],
        "base_model_id": config["base_model_id"], "base_revision": config["base_revision"],
        "source_adapter_config_sha256": _sha256(adapter / "adapter_config.json"),
        "base_config_sha256": _sha256(base / "config.json"), "weights_modified": False,
        "change": "Resolve base_model_name_or_path to the immutable local snapshot and set base revision."}
    (destination / "benchmark-model.json").write_text(json.dumps(provenance, sort_keys=True, indent=2) + "\n")
    return destination


def verify_projection(adapter: Path, base: Path) -> dict:
    """Check the ST Dense head equals the original base projection plus LoRA.

    AutoModel intentionally reports custom_text_proj as unexpected: ST moves that
    trained head into 1_Dense. Verify equivalence before accepting that report.
    """
    from safetensors import safe_open
    index = json.loads((base / "model.safetensors.index.json").read_text())["weight_map"]
    with safe_open(base / index["custom_text_proj.weight"], framework="pt", device="cpu") as handle:
        weight = handle.get_tensor("custom_text_proj.weight").float()
        bias = handle.get_tensor("custom_text_proj.bias").float()
    with safe_open(adapter / "adapter_model.safetensors", framework="pt", device="cpu") as handle:
        left = handle.get_tensor("base_model.model.custom_text_proj.lora_A.weight").float()
        right = handle.get_tensor("base_model.model.custom_text_proj.lora_B.weight").float()
    with safe_open(adapter / "1_Dense/model.safetensors", framework="pt", device="cpu") as handle:
        dense = handle.get_tensor("linear.weight").float()
        dense_bias = handle.get_tensor("linear.bias").float()
    config = json.loads((adapter / "adapter_config.json").read_text())
    merged = weight + (right @ left) * (config["lora_alpha"] / config["r"])
    error = max(float((dense - merged).abs().max()), float((dense_bias - bias).abs().max()))
    if error > 1e-6:
        raise RuntimeError("ColQwen ST projection differs from the pinned original projection plus LoRA.")
    return {"projection_verified": True, "max_absolute_merge_error": error,
        "relocated_projection": "custom_text_proj -> 1_Dense", "projection_shape": list(dense.shape)}


def validate_tokens(values, dimension: int = 128) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != dimension or values.shape[0] < 1 or not np.isfinite(values).all():
        raise RuntimeError("ColQwen must produce a nonempty finite [tokens,128] matrix for every input.")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if (norms <= 0).any():
        raise RuntimeError("ColQwen produced a zero token vector.")
    return values / norms


def exact_maxsim(queries: list[np.ndarray], documents: list[np.ndarray], *, device: str,
                 chunk_elements: int) -> np.ndarray:
    """Sum_i max_j dot(q_i,d_j), including negative maxima and native query expansion."""
    import torch
    from sentence_transformers.util import maxsim
    _torch_device(device)
    with torch.inference_mode():
        result = maxsim([torch.from_numpy(np.asarray(x, dtype=np.float32)) for x in queries], [torch.from_numpy(np.asarray(x, dtype=np.float32)) for x in documents],
            device=device, chunk_elements=chunk_elements, length_normalize=False).float().cpu().numpy()
    synchronize(device)
    if result.shape != (len(queries), len(documents)) or not np.isfinite(result).all():
        raise RuntimeError("Exact MaxSim produced invalid scores.")
    return result


class ColQwenAdapter:
    """A rank-only adapter: variable token vectors cannot implement a dense .encode."""
    def __init__(self, config: dict | None = None):
        self.config = {**COLQWEN_DEFAULTS, **copy.deepcopy(config or {})}
        for key in ("model_id", "base_model_id"):
            if self.config[key] != COLQWEN_DEFAULTS[key]:
                raise ValueError("ColQwen adapter and base model IDs cannot be substituted.")
        validate_revision(self.config)
        validate_revision({"model_id": self.config["base_model_id"], "revision": self.config["base_revision"]})
        if self.config["local_files_only"] is not True:
            raise ValueError("ColQwen inference is offline; prepare pinned snapshots separately.")
        if self.config["dtype"] not in {"float32", "bfloat16"}:
            raise ValueError("ColQwen dtype must be float32 or bfloat16.")
        for field in ("batch_size", "max_length", "max_image_patches", "score_query_batch", "score_chunk_elements"):
            if not isinstance(self.config[field], int) or self.config[field] < 1:
                raise ValueError(f"{field} must be a positive integer.")
        for field in ("query_format", "document_format", "token_mask", "scoring", "token_normalization", "score_dtype"):
            if self.config[field] != COLQWEN_DEFAULTS[field]:
                raise ValueError(f"{field} is fixed for this checkpoint adapter.")
        self.identity = stable_hash({"adapter": "colqwen-native-maxsim-v1", "config": self.config,
            "source_sha256": _sha256(Path(__file__)), "packages": package_versions(("torch", "transformers", "sentence-transformers", "peft", "Pillow", "numpy"))})
        self._config_seal = stable_hash(self.config)
        self._model = None
        self.last_usage = {}
        self.projection_verification = {}

    def _check_config(self):
        if stable_hash(self.config) != self._config_seal:
            raise ValueError("ColQwen configuration changed after identity was fixed.")

    def _load(self):
        self._check_config()
        if self._model is None:
            import torch
            from sentence_transformers import MultiVectorEncoder
            path, base = pinned_snapshots(self.config)
            self.projection_verification = verify_projection(path, base)
            bundle = local_overlay(path, base, self.config)
            self._model = MultiVectorEncoder(str(bundle), device=str(_torch_device(self.config["device"])),
                local_files_only=True, trust_remote_code=False,
                model_kwargs={"dtype": getattr(torch, self.config["dtype"]), "attn_implementation": "sdpa"})
            self._model.eval()
            devices = {str(p.device).split(":")[0] for p in self._model.parameters()}
            if devices != {self.config["device"].split(":")[0]}:
                raise RuntimeError("ColQwen loaded on an unexpected device; CPU fallback is not allowed.")
        return self._model

    def unload(self):
        self._model = None
        gc.collect()
        if self.config["device"].startswith("mps"):
            import torch
            torch.mps.empty_cache()

    def encode_tokens(self, items: list[dict], role: str) -> list[np.ndarray]:
        if role not in {"query", "document"}:
            raise ValueError("role must be query or document.")
        if not items:
            return []
        import torch
        from PIL import Image
        from sentence_transformers.util import batch_to_device
        model = self._load()
        outputs = []
        start = time.perf_counter()
        token_lengths = []
        for offset in range(0, len(items), self.config["batch_size"]):
            inputs = []
            for item in items[offset:offset + self.config["batch_size"]]:
                media = _local_media(item)
                if role == "document":
                    if set(media) != {"image"}:
                        raise ValueError("ColQwen documents require exactly one actual page image.")
                    with Image.open(media["image"]) as image:
                        inputs.append({"image": image.convert("RGB")})
                else:
                    text = item.get("text") or ""
                    if media or not isinstance(text, str) or not text.strip():
                        raise ValueError("ColQwen queries require nonempty text without media.")
                    inputs.append(text)
            features = model.preprocess(inputs, prompt="", task=role, processing_kwargs={
                "text": {"truncation": False}, "image": {"size": {"longest_edge": self.config["max_image_patches"] * 28 * 28, "shortest_edge": 3136}}})
            lengths = features["attention_mask"].sum(dim=1)
            if int(lengths.max()) > self.config["max_length"]:
                raise ValueError("Expanded ColQwen context exceeds configured limit; explicit segmentation is required.")
            with torch.inference_mode():
                result = model(batch_to_device(features, self.config["device"]), task=role)
                masks = result["attention_mask"].bool()
                for tokens, mask in zip(result["token_embeddings"], masks, strict=True):
                    array = validate_tokens(tokens[mask].float().cpu().numpy())
                    outputs.append(array)
                    token_lengths.append(len(array))
        synchronize(self.config["device"])
        self.last_usage = {"inference_seconds": time.perf_counter() - start, "items": len(items),
            "token_counts": token_lengths, "token_dimension": 128, "device": self.config["device"],
            "dtype": self.config["dtype"], "max_image_patches": self.config["max_image_patches"],
            "context_limit": self.config["max_length"], "pooling": "none"}
        return outputs

    def _cache_key(self, item: dict, role: str) -> str:
        media = _local_media(item)
        content = {"id": item.get("id"), "text": item.get("text") if role == "query" else None,
            "media": {kind: _sha256(path) for kind, path in media.items()}}
        return stable_hash({"adapter": self.identity, "role": role, "content": content})

    def _cached_tokens(self, items: list[dict], role: str, cache_dir: Path) -> tuple[list[np.ndarray], dict]:
        destination = cache_dir / "tokens" / role
        destination.mkdir(parents=True, exist_ok=True)
        result, fresh, elapsed = [], 0, 0.0
        # Each input is a durable checkpoint. Failed inputs never disappear from a run.
        for item in items:
            path = destination / (self._cache_key(item, role) + ".npy")
            if path.is_file():
                value = np.load(path, allow_pickle=False)
                checked = validate_tokens(value)
                if not np.allclose(value, checked, atol=2e-6):
                    raise ValueError("Cached ColQwen token vectors are not normalized.")
            else:
                value = self.encode_tokens([item], role)[0]
                elapsed += self.last_usage["inference_seconds"]
                fresh += 1
                temporary = path.with_suffix(".tmp")
                with temporary.open("wb") as handle:
                    np.save(handle, value, allow_pickle=False)
                temporary.replace(path)
            result.append(value)
        counts = [len(x) for x in result]
        return result, {"total_items": len(items), "new_items": fresh, "cache_hits": len(items) - fresh,
            "inference_seconds": elapsed if fresh else None, "token_count_min": min(counts, default=0),
            "token_count_max": max(counts, default=0), "token_count_total": sum(counts)}

    def rank(self, documents: list[dict], queries: list[dict], top_k: int, cache_dir: Path,
             exclusions: list[set[str]] | None = None) -> tuple[list[list[dict]], dict]:
        self._check_config()
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer.")
        ids = [item["id"] for item in documents]
        if len(ids) != len(set(ids)):
            raise ValueError("Document IDs must be unique.")
        exclusions = exclusions if exclusions is not None else [set() for _ in queries]
        if len(exclusions) != len(queries):
            raise ValueError("There must be one exclusion set for every query.")
        document_tokens, document_usage = self._cached_tokens(documents, "document", Path(cache_dir))
        query_tokens, query_usage = self._cached_tokens(queries, "query", Path(cache_dir))
        # Drop the backbone before scoring; token caches remain on CPU and chunks
        # move to the explicitly configured scoring device.
        self.unload()
        rankings = []
        started = time.perf_counter()
        for offset in range(0, len(queries), self.config["score_query_batch"]):
            scores = exact_maxsim(query_tokens[offset:offset + self.config["score_query_batch"]], document_tokens,
                device=self.config["score_device"], chunk_elements=self.config["score_chunk_elements"])
            for index, row in enumerate(scores):
                available = [j for j, identifier in enumerate(ids) if identifier not in exclusions[offset + index]]
                selected = sorted(available, key=lambda j: (-float(row[j]), ids[j]))[:top_k]
                rankings.append([{"id": ids[j], "score": float(row[j])} for j in selected])
        usage = {"adapter_identity": self.identity, "document_encoding": document_usage, "query_encoding": query_usage,
            "scoring_seconds": time.perf_counter() - started, "scoring": self.config["scoring"],
            "score_device": self.config["score_device"], "score_dtype": "float32", "exact": True,
            "query_format": self.config["query_format"], "token_mask": self.config["token_mask"],
            "model_revision": self.config["revision"], "base_revision": self.config["base_revision"], "projection_verification": self.projection_verification}
        self.last_usage = usage
        return rankings, usage


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    parser.add_argument("--output", default="reports/multimodal-colqwen-preflight.json")
    args = parser.parse_args(argv)
    config = {**COLQWEN_DEFAULTS, "device": args.device, "score_device": args.device, "dtype": args.dtype}
    if args.download:
        pinned_snapshots(config, download=True)
    from PIL import Image
    directory = Path("work/model-colqwen")
    directory.mkdir(parents=True, exist_ok=True)
    documents = []
    for name, color in [("red", "red"), ("blue", "blue")]:
        path = (directory / f"synthetic-{name}.png").resolve()
        Image.new("RGB", (224, 224), color).save(path)
        documents.append({"id": name, "media": {"image": str(path)}})
    model = ColQwenAdapter(config)
    try:
        rankings, usage = model.rank(documents, [{"id": "query", "text": "A red page."}], 2, directory / "cache")
        result = {"purpose": "synthetic_backend_smoke_only_not_benchmark_results", "status": "ok",
            "finite": all(np.isfinite(row["score"]) for ranking in rankings for row in ranking),
            "rankings_count": len(rankings), "usage": usage}
    except Exception as exc:
        result = {"purpose": "synthetic_backend_smoke_only_not_benchmark_results", "status": "failed",
            "error_type": type(exc).__name__, "reason": str(exc).replace(str(Path.home()), "~")}
        import traceback
        traceback.print_exc()
    finally:
        model.unload()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    return int(result["status"] != "ok")


if __name__ == "__main__":
    raise SystemExit(main())

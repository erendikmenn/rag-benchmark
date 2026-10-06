"""Immutable offline multimodal adapters; native media is never replaced with captions.

Preparation is explicit: ``python -m rag_benchmark.multimodal_models --download``.
The preflight uses synthetic fixtures solely to check backend health, never accuracy.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from .models import MODEL_DEFAULTS, _snapshot, _torch_device, package_versions, stable_hash, synchronize, validate_revision

MULTIMODAL_DEFAULTS = {
    "embeddinggemma_native": {**MODEL_DEFAULTS["embeddinggemma"], "mode": "native"},
    "embeddinggemma_joint": {**MODEL_DEFAULTS["embeddinggemma"], "mode": "joint"},
    "embeddinggemma_text": {**MODEL_DEFAULTS["embeddinggemma"], "mode": "text"},
    "siglip2": {"model_id": "google/siglip2-base-patch16-224", "revision": "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2", "dimension": 768},
    "clap": {"model_id": "laion/clap-htsat-fused", "revision": "365dea6ef167def6676140ed93bbc43f84dabb28", "dimension": 512},
    "clip_video": {"model_id": "openai/clip-vit-base-patch32", "revision": "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268", "dimension": 512},
    "colqwen": {"model_id": "vidore/colqwen2.5-v0.2", "revision": "dcbe8d9cede518bce830488364ba0e40c873645b"},
    "bge_reranker": {"model_id": "BAAI/bge-reranker-v2-m3", "revision": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"},
}
_PACKAGES = ("torch", "transformers", "sentence-transformers", "numpy", "Pillow", "soundfile", "scipy", "av")


def _unsupported(reason: str):
    from .multimodal import UnsupportedConfiguration
    return UnsupportedConfiguration(reason)


def _model_snapshot(config: dict, *, download: bool = False) -> Path:
    """Prepare only one weight format, avoiding redundant binary+safetensors downloads."""
    if config.get("path") or config.get("local_path"):
        return _snapshot(config, download=download)
    from huggingface_hub import snapshot_download
    validate_revision(config)
    patterns = ["*.json", "*.txt", "*.model", "*.jinja", "tokenizer/*", "1_Pooling/*", "2_Normalize/*"]
    patterns += ["pytorch_model.bin"] if config["model_id"] == "openai/clip-vit-base-patch32" else ["model*.safetensors"]
    return Path(snapshot_download(config["model_id"], revision=config["revision"],
        cache_dir=str(Path(config.get("cache_dir", ".cache/models")).resolve()),
        allow_patterns=patterns, local_files_only=not download))


def normalized(values: Any, rows: int, dimension: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.shape != (rows, dimension) or not np.isfinite(values).all():
        raise RuntimeError(f"Expected finite embeddings [{rows}, {dimension}]; got {values.shape}.")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if (norms <= 0).any():
        raise RuntimeError("Model produced a zero embedding.")
    return values / norms


def _local_media(item: dict) -> dict[str, Path]:
    media = item.get("media") or {}
    if not isinstance(media, dict) or set(media) - {"image", "audio", "video"}:
        raise ValueError("media must map image/audio/video to local file paths.")
    result = {}
    for modality, value in sorted(media.items()):
        if not isinstance(value, (str, Path)) or "://" in str(value):
            raise ValueError("Inference accepts existing local media only; URLs and inline payloads are disabled.")
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Missing {modality} media for item {item.get('id', '<no-id>')}.")
        result[modality] = path
    return result


def load_audio(path: Path, sampling_rate: int) -> tuple[np.ndarray, dict]:
    """Explicit mono mean and polyphase resampling; no duration truncation."""
    import soundfile as sf
    from scipy.signal import resample_poly
    audio, original_rate = sf.read(path, dtype="float32", always_2d=True)
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError("Audio must contain nonempty finite samples.")
    channels = audio.shape[1]
    waveform = audio.mean(axis=1)
    if original_rate != sampling_rate:
        divisor = math.gcd(original_rate, sampling_rate)
        waveform = resample_poly(waveform, sampling_rate // divisor, original_rate // divisor).astype(np.float32)
    return waveform, {"source_sampling_rate": original_rate, "sampling_rate": sampling_rate,
        "source_channels": channels, "mono_rule": "arithmetic_mean", "resampling": "scipy_resample_poly",
        "duration_seconds": len(waveform) / sampling_rate, "sample_count": len(waveform)}


def sample_video(path: Path, fps: float, max_frames: int) -> tuple[np.ndarray, dict]:
    """Sample nearest decoded timestamps at FPS, then cap by uniform index selection.

    Two decoding passes avoid keeping all full-resolution frames in memory. Audio is
    intentionally excluded from the visual benchmark; AV passes a separate audio path.
    """
    import av
    timestamps = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0] if container.streams.video else None
        if stream is None:
            raise ValueError("Video file has no visual stream.")
        rate = float(stream.average_rate) if stream.average_rate else None
        for index, frame in enumerate(container.decode(stream)):
            if frame.time is None and rate is None:
                raise ValueError("Video has neither frame timestamps nor a declared frame rate.")
            timestamps.append(float(frame.time) if frame.time is not None else index / rate)
    if not timestamps:
        raise ValueError("Video decoded no frames.")
    times = np.asarray(timestamps) - timestamps[0]
    if np.any(np.diff(times) < 0):
        raise ValueError("Video timestamps are not monotonic.")
    targets = np.arange(0, times[-1] + 1e-8, 1 / fps)
    insertion = np.searchsorted(times, targets).clip(0, len(times) - 1)
    previous = np.maximum(0, insertion - 1)
    nearest = np.where(abs(times[previous] - targets) <= abs(times[insertion] - targets), previous, insertion)
    indices = np.unique(nearest)
    if len(indices) > max_frames:
        indices = indices[np.linspace(0, len(indices) - 1, max_frames).round().astype(int)]
    selected = set(indices.tolist())
    frames = []
    with av.open(str(path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index in selected:
                frames.append(frame.to_ndarray(format="rgb24"))
    if len(frames) != len(indices):
        raise RuntimeError("The video changed between sampling and decoding passes.")
    if len({x.shape for x in frames}) != 1:
        raise ValueError("Variable-resolution video requires an explicit resize preprocessing condition.")
    return np.stack(frames), {"sampling_fps": fps, "max_frames": max_frames,
        "sampling_rule": "nearest_timestamp_then_uniform_cap", "frame_indices": indices.tolist(),
        "timestamps_seconds": times[indices].tolist(), "decoded_frames": len(times), "used_frames": len(frames),
        "audio_included": False}


class _BaseAdapter:
    def _initialize(self, name: str, config: dict | None):
        self.name = name
        self.config = {**MULTIMODAL_DEFAULTS[name], "device": "mps", "dtype": "bfloat16",
            "batch_size": 1, "local_files_only": True, "max_length": 8192,
            "vision_budget": 280, "video_vision_budget": 140, "video_fps": 1.0, "video_max_frames": 16,
            "query_prompt": "task: search result | query: ", "document_prompt": "title: none | text: ",
            "seed": 0, **copy.deepcopy(config or {})}
        validate_revision(self.config)
        if self.config["model_id"] != MULTIMODAL_DEFAULTS[name]["model_id"]:
            raise ValueError("This adapter does not permit substituting a different model_id.")
        if self.config["local_files_only"] is not True:
            raise ValueError("Inference is strictly offline. Download the immutable snapshot explicitly first.")
        if self.config["dtype"] not in {"float32", "bfloat16"}:
            raise ValueError("Supported numeric conditions are float32 and bfloat16; float16 is forbidden.")
        if not isinstance(self.config["batch_size"], int) or self.config["batch_size"] < 1:
            raise ValueError("batch_size must be a positive integer.")
        if not 1 <= int(self.config["max_length"]) <= 8192:
            raise ValueError("max_length must be in 1..8192; no silent truncation is performed.")
        if self.config["vision_budget"] not in {70, 140, 280, 560, 1120} or self.config["video_vision_budget"] not in {70, 140, 280, 560, 1120}:
            raise ValueError("Vision token budgets must be 70, 140, 280, 560, or 1120.")
        if not math.isfinite(float(self.config["video_fps"])) or self.config["video_fps"] <= 0 or self.config["video_max_frames"] < 1:
            raise ValueError("Video sampling FPS and maximum frame count must be positive.")
        self.dimension = int(self.config.get("dimension", 0))
        self.identity = stable_hash({"adapter": "multimodal-local-v1", "family": name,
            "config": self.config, "source_sha256": stable_hash(Path(__file__).read_text()), "packages": package_versions(_PACKAGES)})
        self._model = None
        self._processor = None
        self.last_usage: dict = {}
        self._config_seal = stable_hash(self.config)

    def _check_immutable(self):
        if stable_hash(self.config) != self._config_seal:
            raise ValueError("Adapter configuration changed after identity was fixed; create a new adapter.")

    def unload(self):
        self._model = self._processor = None
        gc.collect()
        if str(self.config["device"]).startswith("mps"):
            import torch
            torch.mps.empty_cache()

    def _verify_device(self):
        devices = {str(p.device) for p in self._model.parameters()}
        expected = str(_torch_device(self.config["device"]))
        if any(d.split(":")[0] != expected.split(":")[0] for d in devices):
            raise RuntimeError(f"Requested {expected}, but model parameters are on {sorted(devices)}.")
        return sorted(devices)

    def _text_guard(self, texts, maximum: int):
        tokenizer = getattr(self._processor, "tokenizer", self._processor)
        tokens = tokenizer(texts, padding=False, truncation=False)
        if any(len(x) > maximum for x in tokens["input_ids"]):
            raise ValueError(f"Text exceeds the native {maximum}-token context. Segment explicitly; truncation is disabled.")


class EmbeddingGemma2Adapter(_BaseAdapter):
    """N and J use the full native towers. J concatenates media and text before forward."""
    def __init__(self, config: dict | None = None, *, mode: str = "native"):
        mode = (config or {}).get("mode", mode)
        if mode not in {"native", "joint", "text"}:
            raise ValueError("EG2 mode must be native, joint, or text.")
        self._initialize(f"embeddinggemma_{mode}", config)
        if self.dimension not in {128, 256, 512, 768}:
            raise ValueError("EG2 supports only 128/256/512/768 dimensions with renormalization.")
        self.mode = mode

    def _load(self):
        self._check_immutable()
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer
            device = _torch_device(self.config["device"])
            self._model = SentenceTransformer(str(_model_snapshot(self.config)), device=str(device),
                local_files_only=True, trust_remote_code=False,
                model_kwargs={"dtype": getattr(torch, self.config["dtype"]), "attn_implementation": "sdpa"})
            self._model.eval()
            base = self._model[0].auto_model
            if base.vision_tower is None or base.audio_tower is None:
                raise RuntimeError("Native EG2 requires both complete vision and audio towers.")
            self._processor = self._model[0].processor
            self.towers = {"vision": type(base.vision_tower).__name__, "audio": type(base.audio_tower).__name__}
            self._verify_device()
        return self._model

    def _prepare(self, item: dict, role: str) -> tuple[Any, dict]:
        from PIL import Image
        media = _local_media(item)
        text = item.get("text") or ""
        if not isinstance(text, str):
            raise ValueError("text must be a string.")
        if self.mode == "text":
            if not text.strip():
                raise ValueError("Text channel cannot encode an empty representation.")
            return self.config[f"{role}_prompt"] + text, {"modalities": ["text"]}
        if role == "document" and not media:
            raise ValueError("Native/joint document encoding requires actual media.")
        if role == "document" and self.mode == "joint" and not isinstance(item.get("text"), str):
            raise ValueError("J requires both actual media and an explicit derived text field; recorded empty extraction is allowed.")
        if not media and not text.strip():
            raise ValueError("Input must contain text or media.")
        # Native document text is deliberately not consumed. Query instructions are part
        # of composed queries; joint text follows media inside the same forward pass.
        include_text = (bool(text.strip()) and role == "query") or (role == "document" and self.mode == "joint")
        payload, metadata = {}, {"modalities": list(media)}
        for modality, path in media.items():
            if modality == "image":
                with Image.open(path) as im:
                    payload["image"] = im.convert("RGB")
            elif modality == "audio":
                waveform, details = load_audio(path, 16000)
                payload["audio"] = {"array": waveform, "sampling_rate": 16000}
                metadata["audio"] = details
            else:
                payload["video"], metadata["video"] = sample_video(path, self.config["video_fps"], self.config["video_max_frames"])
        if include_text:
            payload["text"] = self.config[f"{role}_prompt"] + text if text.strip() else ""
            metadata["modalities"].append("text")
            metadata["derived_text_empty"] = role == "document" and not text.strip()
        return payload, metadata

    def encode(self, items: list[dict], role: str = "query") -> np.ndarray:
        if role not in {"query", "document"}:
            raise ValueError("role must be query or document.")
        if not items:
            return np.empty((0, self.dimension), dtype=np.float32)
        import torch
        from sentence_transformers.util import batch_to_device
        model = self._load()
        records, vectors = [], []
        start = time.perf_counter()
        batch_size = self.config["batch_size"]
        for offset in range(0, len(items), batch_size):
            prepared = [self._prepare(item, role) for item in items[offset:offset + batch_size]]
            features = model.preprocess([x[0] for x in prepared], prompt="", processing_kwargs={
                "text": {"truncation": False, "padding": True},
                "audio": {"sampling_rate": 16000, "truncation": False},
                "image": {"max_soft_tokens": self.config["vision_budget"]},
                "video": {"max_soft_tokens": self.config["video_vision_budget"], "do_sample_frames": False, "add_timestamps": False}})
            lengths = features["attention_mask"].sum(dim=1).tolist()
            if max(lengths) > self.config["max_length"]:
                raise ValueError(f"Expanded multimodal context {max(lengths)} exceeds {self.config['max_length']} tokens; split the input explicitly.")
            for (_, record), length in zip(prepared, lengths, strict=True):
                records.append({**record, "expanded_tokens": int(length)})
            with torch.inference_mode():
                embedding = model(batch_to_device(features, self.config["device"]))["sentence_embedding"]
                vectors.append(embedding[:, :self.dimension].float().cpu().numpy())
        synchronize(self.config["device"])
        values = normalized(np.concatenate(vectors), len(items), self.dimension)
        self.last_usage = {"inference_seconds": time.perf_counter() - start, "items": records,
            "empty_derived_text_items": sum(bool(x.get("derived_text_empty")) for x in records),
            "devices": self._verify_device(), "dtype": self.config["dtype"], "context_limit": self.config["max_length"], "towers": self.towers}
        return values


class SpecialistAdapter(_BaseAdapter):
    """Native SigLIP2/CLAP and explicitly defined normalized mean CLIP frame pooling."""
    def __init__(self, name: str, config: dict | None = None):
        if name == "colqwen":
            raise _unsupported("ColQwen2.5 requires variable-length token embeddings and MaxSim late interaction; the dense encode contract cannot represent this faithfully. No pooled-vector substitute is used.")
        if name not in {"siglip2", "clap", "clip_video"}:
            raise ValueError(f"Unknown specialist: {name}")
        self._initialize(name, config)
        if self.dimension != MULTIMODAL_DEFAULTS[name]["dimension"]:
            raise ValueError("Specialist dimensions are fixed to the checkpoint's native projection size.")
        self.config["pooling"] = "normalize_each_frame_then_mean_then_normalize" if name == "clip_video" else "native_projection"
        self.identity = stable_hash({"base": self.identity, "pooling": self.config["pooling"]})
        self._config_seal = stable_hash(self.config)

    def _load(self):
        self._check_immutable()
        if self._model is None:
            import torch
            from transformers import AutoModel, AutoProcessor
            path = str(_model_snapshot(self.config))
            self._processor = AutoProcessor.from_pretrained(path, local_files_only=True, trust_remote_code=False)
            self._model = AutoModel.from_pretrained(path, local_files_only=True, trust_remote_code=False,
                dtype=getattr(torch, self.config["dtype"])).to(_torch_device(self.config["device"])).eval()
            self._verify_device()
        return self._model

    def _features(self, inputs):
        import torch
        dtype = getattr(torch, self.config["dtype"])
        return {k: v.to(self.config["device"], dtype=dtype if v.is_floating_point() else v.dtype) for k, v in inputs.items()}

    def encode(self, items: list[dict], role: str = "query") -> np.ndarray:
        if role not in {"query", "document"}:
            raise ValueError("role must be query or document.")
        if not items:
            return np.empty((0, self.dimension), dtype=np.float32)
        import torch
        from PIL import Image
        model = self._load()
        start, values, records = time.perf_counter(), [], []
        for item in items:
            media, text = _local_media(item), item.get("text") or ""
            if role == "query":
                if media or not text.strip():
                    raise _unsupported(f"{self.name} specialist accepts text-only queries in this benchmark condition.")
                maximum = 64 if self.name == "siglip2" else 77 if self.name == "clip_video" else 512
                self._text_guard([text], maximum)
                inputs = self._processor(text=[text], padding="max_length" if self.name == "siglip2" else True,
                    max_length=maximum, truncation=False, return_tensors="pt")
                with torch.inference_mode():
                    vector = model.get_text_features(**self._features(inputs))
                records.append({"modalities": ["text"]})
            elif self.name == "siglip2":
                if set(media) != {"image"}:
                    raise _unsupported("SigLIP2 documents require exactly one actual image.")
                with Image.open(media["image"]) as im:
                    inputs = self._processor(images=[im.convert("RGB")], return_tensors="pt")
                with torch.inference_mode():
                    vector = model.get_image_features(**self._features(inputs))
                records.append({"modalities": ["image"]})
            elif self.name == "clap":
                if set(media) != {"audio"}:
                    raise _unsupported("CLAP documents require exactly one actual audio waveform.")
                waveform, details = load_audio(media["audio"], 48000)
                # Native fused CLAP compresses long audio via fusion rather than replacing
                # it with text. Its random crop draws are pinned independently for each input.
                state = np.random.get_state()
                try:
                    np.random.seed(self.config["seed"])
                    inputs = self._processor(audio=[waveform], sampling_rate=48000, return_tensors="pt")
                finally:
                    np.random.set_state(state)
                with torch.inference_mode():
                    vector = model.get_audio_features(**self._features(inputs))
                records.append({"modalities": ["audio"], "audio": details, "long_audio_policy": "checkpoint_native_fusion", "seed": self.config["seed"]})
            else:
                if set(media) != {"video"}:
                    raise _unsupported("CLIP video documents require exactly one actual video.")
                frames, details = sample_video(media["video"], self.config["video_fps"], self.config["video_max_frames"])
                inputs = self._processor(images=list(frames), return_tensors="pt")
                with torch.inference_mode():
                    frame_vectors = model.get_image_features(**self._features(inputs))
                    if hasattr(frame_vectors, "pooler_output"):
                        frame_vectors = frame_vectors.pooler_output
                    frame_vectors = frame_vectors.float()
                    frame_vectors /= frame_vectors.norm(dim=-1, keepdim=True)
                    vector = frame_vectors.mean(dim=0, keepdim=True)
                records.append({"modalities": ["video"], "video": details, "pooling": self.config["pooling"]})
            if hasattr(vector, "pooler_output"):
                vector = vector.pooler_output
            values.append(vector.float().cpu().numpy())
        synchronize(self.config["device"])
        result = normalized(np.concatenate(values), len(items), self.dimension)
        self.last_usage = {"inference_seconds": time.perf_counter() - start, "items": records,
            "devices": self._verify_device(), "dtype": self.config["dtype"]}
        return result


class BGEReranker(_BaseAdapter):
    """Pointwise BGE v2 m3 logits on fixed derived text; no media substitution."""
    pointwise = True

    def __init__(self, config: dict | None = None):
        self._initialize("bge_reranker", config)

    def _load(self):
        self._check_immutable()
        if self._model is None:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            path = str(_model_snapshot(self.config))
            self._processor = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
            self._model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True,
                trust_remote_code=False, dtype=getattr(torch, self.config["dtype"])).to(_torch_device(self.config["device"])).eval()
            self._verify_device()
        return self._model

    def score(self, query: dict, candidates: list[dict]) -> list[float]:
        if not candidates:
            return []
        import torch
        model = self._load()
        question = query.get("text") or ""
        if not question.strip() or any(not (x.get("text") or "").strip() for x in candidates):
            raise _unsupported("BGE reranking requires fixed nonempty query and candidate text representations.")
        start, scores = time.perf_counter(), []
        for offset in range(0, len(candidates), self.config["batch_size"]):
            pairs = [(question, x["text"]) for x in candidates[offset:offset + self.config["batch_size"]]]
            inputs = self._processor(pairs, padding=True, truncation=False, return_tensors="pt")
            if int(inputs["attention_mask"].sum(dim=1).max()) > self.config["max_length"]:
                raise ValueError("BGE reranker pair exceeds configured context; explicit segmentation is required.")
            with torch.inference_mode():
                logits = model(**{k: v.to(self.config["device"]) for k, v in inputs.items()}).logits
                scores.extend(logits.float().cpu().reshape(-1).tolist())
        synchronize(self.config["device"])
        if len(scores) != len(candidates) or not np.isfinite(scores).all():
            raise RuntimeError("BGE reranker produced invalid scores.")
        self.last_usage = {"inference_seconds": time.perf_counter() - start, "pairs": len(scores),
            "score_type": "raw_relevance_logit", "devices": self._verify_device(), "dtype": self.config["dtype"]}
        return scores


def make_multimodal_adapter(name: str, config: dict | None = None):
    if name in {"N", "embeddinggemma_native", "J", "embeddinggemma_joint"}:
        mode = "native" if name in {"N", "embeddinggemma_native"} else "joint"
        if (config or {}).get("mode", mode) != mode:
            raise ValueError("Configured mode conflicts with the requested native/joint channel.")
        return EmbeddingGemma2Adapter(config, mode=mode)
    if name == "embeddinggemma_text":
        return EmbeddingGemma2Adapter(config, mode="text")
    if name == "bge_reranker":
        return BGEReranker(config)
    return SpecialistAdapter(name, config)


def _fixtures(directory: Path) -> dict:
    import av
    import soundfile as sf
    from PIL import Image
    directory.mkdir(parents=True, exist_ok=True)
    image = directory / "synthetic-red.png"
    audio = directory / "synthetic-tone.wav"
    video = directory / "synthetic-colors.mp4"
    Image.new("RGB", (224, 224), "red").save(image)
    waveform = .1 * np.sin(2 * np.pi * 440 * np.arange(32000) / 16000).astype(np.float32)
    sf.write(audio, waveform, 16000)
    with av.open(str(video), "w") as container:
        stream = container.add_stream("mpeg4", rate=2)
        stream.width, stream.height, stream.pix_fmt = 224, 224, "yuv420p"
        for color in ["red", "blue", "green", "yellow"]:
            frame = av.VideoFrame.from_ndarray(np.asarray(Image.new("RGB", (224, 224), color)), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return {"image": image.resolve(), "audio": audio.resolve(), "video": video.resolve()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="embeddinggemma_native,embeddinggemma_joint,siglip2,clap,clip_video,bge_reranker,colqwen")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--download", action="store_true", help="Explicitly prepare immutable local snapshots.")
    parser.add_argument("--output", default="reports/multimodal-model-preflight.json")
    parser.add_argument("--work", default="work/model-preflight")
    args = parser.parse_args(argv)
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    media = _fixtures(work)
    report = {"schema_version": 1, "purpose": "synthetic_backend_smoke_only_not_benchmark_results",
        "packages": package_versions(_PACKAGES), "conditions": []}
    for name in args.models.split(","):
        base = {"adapter": name, "model_id": MULTIMODAL_DEFAULTS[name]["model_id"],
            "revision": MULTIMODAL_DEFAULTS[name]["revision"], "device": args.device, "dtype": args.dtype}
        adapter = None
        try:
            if args.download and name != "colqwen":
                _model_snapshot(MULTIMODAL_DEFAULTS[name], download=True)
            adapter = make_multimodal_adapter(name, {"device": args.device, "dtype": args.dtype})
            if name == "bge_reranker":
                scores = adapter.score({"text": "What color is the square?"}, [{"text": "The square is red."}, {"text": "A dog is sleeping."}])
                report["conditions"].append({**base, "condition": "text_pair", "status": "ok", "finite": bool(np.isfinite(scores).all()), "count": len(scores), "usage": adapter.last_usage})
            else:
                conditions = [("text", {"id": "synthetic-query", "text": "A red square."}, "query")]
                modalities = ["image", "audio", "video"] if name.startswith("embeddinggemma") else [{"siglip2": "image", "clap": "audio", "clip_video": "video"}[name]]
                conditions.extend((modality, {"id": f"synthetic-{modality}", "text": "A synthetic colored shape or tone.", "media": {modality: str(media[modality])}}, "document") for modality in modalities)
                for condition, item, role in conditions:
                    try:
                        result = adapter.encode([item], role=role)
                        report["conditions"].append({**base, "condition": condition, "status": "ok", "identity": adapter.identity,
                            "shape": list(result.shape), "finite": bool(np.isfinite(result).all()),
                            "norms": np.linalg.norm(result, axis=1).tolist(), "usage": adapter.last_usage})
                    except Exception as exc:
                        report["conditions"].append({**base, "condition": condition, "status": "failed", "error_type": type(exc).__name__, "reason": str(exc).replace(str(Path.home()), "~")})
        except Exception as exc:
            report["conditions"].append({**base, "status": "unsupported" if type(exc).__name__ == "UnsupportedConfiguration" else "failed", "error_type": type(exc).__name__, "reason": str(exc).replace(str(Path.home()), "~")})
        finally:
            if adapter is not None:
                adapter.unload()
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": args.output, "conditions": len(report["conditions"]), "failures": sum(c["status"] == "failed" for c in report["conditions"])}))
    return int(any(c["status"] == "failed" for c in report["conditions"]))


if __name__ == "__main__":
    raise SystemExit(main())

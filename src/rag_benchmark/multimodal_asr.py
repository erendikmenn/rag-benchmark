"""Source-only, pinned Whisper ASR views; references never enter transcription.

Download explicitly with ``python -m rag_benchmark.multimodal_asr --download``.
Inference thereafter uses the offline snapshot. Transcription accepts only local
media paths; queries, captions, gold transcripts, and vocabulary hints are not an
input. Empty recognizer outputs are retained. WER is a separate post-hoc step.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
from pathlib import Path
import shutil
import time
import unicodedata
import uuid

import numpy as np

from .models import _snapshot, _torch_device, package_versions, stable_hash, synchronize, validate_revision
from .multimodal_data import read_jsonl, sha256, validate_dataset, write_dataset
from .multimodal_models import load_audio

ASR_DEFAULTS = {
    "model_id": "openai/whisper-large-v3-turbo",
    "revision": "41f01f3fe87f28c78e2fbf8b568835947dd65ed9",
    "cache_dir": ".cache/models", "transcript_cache_dir": ".cache/multimodal-asr",
    "local_files_only": True, "device": "mps", "dtype": "float32",
    "language": "turkish", "task": "transcribe", "seed": 0,
    "max_new_tokens": 440, "no_speech_threshold": 0.6, "logprob_threshold": -1.0,
    "condition_on_prev_tokens": False, "audio_stream": 0,
}
_ASR_PACKAGES = ("transformers", "torch", "numpy", "soundfile", "scipy", "av")


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _config(config: dict | None) -> dict:
    config = copy.deepcopy(config or {})
    allowed = set(ASR_DEFAULTS) | {"path", "local_path"}
    unknown = set(config) - allowed
    if unknown:
        raise ValueError("Unknown ASR configuration fields: " + ", ".join(sorted(unknown)))
    config = {**ASR_DEFAULTS, **config}
    validate_revision(config)
    if config["model_id"] != ASR_DEFAULTS["model_id"]:
        raise ValueError("This adapter requires the declared official Whisper checkpoint")
    if config["task"] != "transcribe":
        raise ValueError("ASR views transcribe the source language; translation is a separate experiment")
    if config["local_files_only"] is not True:
        raise ValueError("Inference is offline; call download_asr_model explicitly during preparation")
    if config["dtype"] not in {"float32", "bfloat16"}:
        raise ValueError("Whisper comparison supports explicit float32 or bfloat16")
    if not isinstance(config["language"], str) or not config["language"].strip():
        raise ValueError("An explicit source language is required")
    if not isinstance(config["max_new_tokens"], int) or not 1 <= config["max_new_tokens"] <= 440:
        raise ValueError("max_new_tokens must be in 1..440 and token-limit outputs fail closed")
    if not isinstance(config["seed"], int) or not isinstance(config["audio_stream"], int) or config["audio_stream"] < 0:
        raise ValueError("ASR seed and nonnegative audio_stream must be integers")
    if not math.isfinite(float(config["no_speech_threshold"])) or not 0 <= config["no_speech_threshold"] <= 1:
        raise ValueError("Invalid no-speech probability threshold")
    if not math.isfinite(float(config["logprob_threshold"])):
        raise ValueError("Invalid log-probability threshold")
    return config


def download_asr_model(config: dict | None = None) -> Path:
    """The sole network-enabled model operation; pin resolves before downloading."""
    config = _config(config)
    if config.get("path") or config.get("local_path"):
        return _snapshot(config, download=False)
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(config["model_id"], revision=config["revision"],
        cache_dir=str(Path(config["cache_dir"]).resolve()), local_files_only=False,
        allow_patterns=["*.json", "*.txt", "*.model", "model.safetensors"]))


def _video_audio(path: Path, stream_index: int) -> tuple[np.ndarray, dict]:
    import av
    with av.open(str(path)) as container:
        streams = list(container.streams.audio)
        if stream_index >= len(streams):
            from .multimodal import UnsupportedConfiguration
            raise UnsupportedConfiguration("Video has no requested audio stream; no caption substitute is used")
        stream = streams[stream_index]
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)
        chunks = []
        for frame in container.decode(stream):
            chunks.extend(np.asarray(converted.to_ndarray(), dtype=np.float32).reshape(-1)
                          for converted in resampler.resample(frame))
        chunks.extend(np.asarray(converted.to_ndarray(), dtype=np.float32).reshape(-1)
                      for converted in resampler.resample(None))
    waveform = np.concatenate(chunks) if chunks else np.asarray([], dtype=np.float32)
    if waveform.size == 0 or not np.isfinite(waveform).all():
        raise ValueError("Video audio must contain nonempty finite samples")
    return waveform, {"sampling_rate": 16000, "duration_seconds": len(waveform) / 16000,
                      "sample_count": len(waveform), "audio_stream": stream_index,
                      "resampling": "PyAV AudioResampler mono 16000Hz float32", "truncated": False}


class WhisperTranscriber:
    """A transcriber whose only per-record input is a local audio/video file."""
    def __init__(self, config: dict | None = None):
        self.config = _config(config)
        # Cache paths affect storage, never recognition; portable caches are keyed
        # by model revision, source bytes, preprocessing, generation and packages.
        identity_config = {key: value for key, value in self.config.items()
                           if key not in {"cache_dir", "transcript_cache_dir", "path", "local_path"}}
        self.identity = stable_hash({"adapter": "whisper-source-only-v1", "config": identity_config,
            "prompt": "none; language and transcribe task only", "packages": package_versions(_ASR_PACKAGES),
            "audio_preprocessing": "native-full-waveform; 16000Hz; no truncation",
            "long_form": "native_timestamp_sequential"})
        self.prompt_identity = stable_hash({"language": self.config["language"], "task": "transcribe", "prompt": None})
        self._config_seal = stable_hash(self.config)
        self._model = self._processor = None
        self.last_usage: dict = {}

    def _load(self) -> None:
        if stable_hash(self.config) != self._config_seal:
            raise ValueError("ASR configuration changed after its identity was fixed")
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        path = _snapshot(self.config, download=False)
        device = _torch_device(self.config["device"])
        self._processor = AutoProcessor.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
        self._model = AutoModelForSpeechSeq2Seq.from_pretrained(str(path), local_files_only=True,
            trust_remote_code=False, dtype=getattr(torch, self.config["dtype"]), attn_implementation="eager")
        self._model.to(device).eval()
        if any(parameter.device.type != device.type for parameter in self._model.parameters()):
            raise RuntimeError("Whisper model is not entirely on the requested device")
        if self._processor.feature_extractor.sampling_rate != 16000:
            raise RuntimeError("Pinned Whisper preprocessing unexpectedly changed sampling rate")

    def unload(self) -> None:
        self._model = self._processor = None
        gc.collect()
        if str(self.config["device"]).startswith("mps"):
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

    def transcribe(self, path: Path | str, *, media_kind: str = "audio") -> str:
        """No text/hint argument exists; native long-form processing keeps all audio."""
        import torch
        path = Path(path).resolve()
        if not path.is_file():
            raise FileNotFoundError("ASR requires an existing local media file")
        if media_kind == "audio":
            waveform, audio_usage = load_audio(path, 16000)
        elif media_kind == "video":
            waveform, audio_usage = _video_audio(path, self.config["audio_stream"])
        else:
            raise ValueError("Whisper accepts an audio file or the audio stream of a video")
        self._load()
        torch.manual_seed(self.config["seed"])
        features = self._processor(waveform, sampling_rate=16000, return_tensors="pt",
            truncation=False, padding="max_length", return_attention_mask=True)
        # Whisper's centered STFT drops its final frame, so native frame count is
        # floor(samples / hop), including for non-integral final audio frames.
        expected_frames = len(waveform) // self._processor.feature_extractor.hop_length
        if features["input_features"].shape[-1] < expected_frames:
            raise RuntimeError("Whisper feature extraction truncated the source waveform")
        device = self.config["device"]
        inputs = {"input_features": features["input_features"].to(device=device, dtype=getattr(torch, self.config["dtype"])),
                  "attention_mask": features["attention_mask"].to(device=device)}
        long_form = len(waveform) > self._processor.feature_extractor.n_samples
        generation = {"language": self.config["language"], "task": "transcribe", "do_sample": False,
            "num_beams": 1, "temperature": 0.0, "max_new_tokens": self.config["max_new_tokens"],
            "condition_on_prev_tokens": self.config["condition_on_prev_tokens"],
            "no_speech_threshold": self.config["no_speech_threshold"],
            "logprob_threshold": self.config["logprob_threshold"], "compression_ratio_threshold": None,
            "return_timestamps": long_form, "return_segments": long_form, "return_dict_in_generate": True}
        synchronize(device)
        tick = time.perf_counter()
        with torch.inference_mode():
            output = self._model.generate(**inputs, **generation)
        synchronize(device)
        seconds = time.perf_counter() - tick
        sequences = output["sequences"]
        eos = self._model.generation_config.eos_token_id
        eos = {eos} if isinstance(eos, int) else set(eos)
        if not long_form:
            result_sequences = [sequences[0]]
        else:
            result_sequences = [segment["result"]["sequences"]
                                for segment in output.get("segments", [[]])[0] if "result" in segment]
        for sequence in result_sequences:
            tokens = sequence.detach().cpu().reshape(-1).tolist()
            if tokens and not (set(tokens) & eos):
                raise RuntimeError("Whisper decoder reached its token limit without EOS; transcript is not accepted")
        transcripts = self._processor.batch_decode(sequences, skip_special_tokens=True)
        if len(transcripts) != 1 or not isinstance(transcripts[0], str):
            raise RuntimeError("Whisper returned an invalid number of transcripts")
        transcript = transcripts[0].strip()
        self.last_usage = {**audio_usage, "inference_seconds": seconds, "device": device,
            "dtype": self.config["dtype"], "language": self.config["language"], "task": "transcribe",
            "long_form": long_form, "truncated": False, "input_feature_frames": int(features["input_features"].shape[-1]),
            "empty_transcript": transcript == "", "token_count": int(sequences.shape[-1]),
            "segment_count": len(output.get("segments", [[]])[0]) if long_form else 1,
            "cache_hit": False, "cache_reads_are_inference": False}
        return transcript


def _resolve_media(root: Path, item: dict) -> tuple[str, Path]:
    media = item.get("media", {})
    kind = "audio" if "audio" in media else "video" if "video" in media else None
    if kind is None:
        from .multimodal import UnsupportedConfiguration
        raise UnsupportedConfiguration("Every ASR candidate must have a real audio or video source")
    relative = Path(media[kind])
    path = (root / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root) or not path.is_file():
        raise ValueError("ASR media must be an existing file within the source dataset")
    return kind, path


def _link_assets(source: Path, destination: Path, rows: list[dict]) -> None:
    for item in rows:
        for relative in item.get("media", {}).values():
            path = (source / relative).resolve()
            target = (destination / relative).resolve()
            if Path(relative).is_absolute() or not path.is_relative_to(source) or not target.is_relative_to(destination):
                raise ValueError("Dataset view assets must remain inside their dataset folders")
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if sha256(path) != sha256(target):
                    raise ValueError("Existing view asset differs from the source")
            else:
                try:
                    os.link(path, target)
                except OSError:
                    shutil.copyfile(path, target)


def prepare_asr_view(source: Path | str, destination: Path | str, config: dict | None = None, *,
                     progress_callback=None) -> dict:
    """Write an ASR-only corpus view preserving all IDs, queries, qrels and media.

    Queries and evaluation-only transcripts are not read until every transcription
    is frozen in the cache. They are never passed to ``WhisperTranscriber``.
    """
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination:
        raise ValueError("ASR preparation requires a separate destination dataset view")
    source_manifest = json.loads((source / "dataset.json").read_text())
    if source_manifest.get("status") != "ready":
        raise ValueError("ASR source dataset must have ready status")
    config = _config(config)
    language = source_manifest.get("metadata", {}).get("language")
    if source_manifest.get("id", "").startswith("fleurs-tr_tr") or language == "tr":
        if config["language"].casefold() not in {"turkish", "tr", "<|tr|>"}:
            raise ValueError("FLEURS Turkish requires explicit Turkish transcription, not language autodetection")
    transcriber = WhisperTranscriber(config)
    view_identity = stable_hash({"adapter": transcriber.identity, "source_revision": source_manifest["revision"],
                                 "source_corpus_sha256": sha256(source / "corpus.jsonl"),
                                 "source_assets_sha256": sha256(source / "assets.jsonl")})
    destination.mkdir(parents=True, exist_ok=True)
    ready_path = destination / "dataset.json"
    if ready_path.exists():
        prior = json.loads(ready_path.read_text())
        prior_identity = prior.get("metadata", {}).get("asr", {}).get("view_identity")
        if prior.get("status") == "ready" and prior_identity != view_identity:
            raise ValueError("Destination already contains a different frozen dataset view")
    corpus = read_jsonl(source / "corpus.jsonl")
    if not corpus:
        raise ValueError("ASR source corpus must not be empty")
    # Validate all required sources before any expensive generation.
    inputs = [_resolve_media(source, item) for item in corpus]
    cache_root = Path(config["transcript_cache_dir"])
    completed, cache_hits, new_count, empty_count, inference_seconds = [], 0, 0, 0, 0.0
    provenance = {"source": "whisper_asr", "query_independent": True, "gold_fields_used": False,
                  "empty_text_is_valid": True, "model_id": config["model_id"], "revision": config["revision"],
                  "adapter_identity": transcriber.identity, "language": config["language"], "task": "transcribe"}
    progress = {"status": "preparing", "view_identity": view_identity, "source_id": source_manifest["id"],
                "model_id": config["model_id"], "model_revision": config["revision"], "corpus_count": len(corpus),
                "completed": 0, "cached": 0, "new": 0, "empty_transcripts": 0, "cache_reads_are_inference": False}
    _atomic_json(destination / "asr-preparation.json", progress)
    try:
        for item, (media_kind, path) in zip(corpus, inputs):
            source_hash = sha256(path)
            key = stable_hash({"model": transcriber.identity, "prompt": transcriber.prompt_identity,
                               "source_sha256": source_hash, "media_kind": media_kind})
            cache_path = cache_root / transcriber.identity / (key + ".json")
            if cache_path.exists():
                record = json.loads(cache_path.read_text())
                if record.get("source_sha256") != source_hash or record.get("adapter_identity") != transcriber.identity:
                    raise ValueError("Stale or corrupt ASR cache identity")
                transcript = record.get("transcript")
                if not isinstance(transcript, str):
                    raise ValueError("ASR cache transcript must be a string, including legitimately empty output")
                cache_hits += 1
            else:
                transcript = transcriber.transcribe(path, media_kind=media_kind)
                if not isinstance(transcript, str):
                    raise ValueError("ASR transcriber must return a string")
                if sha256(path) != source_hash:
                    raise ValueError("Source media changed during ASR inference")
                record = {"schema_version": 1, "source_sha256": source_hash, "adapter_identity": transcriber.identity,
                          "prompt_identity": transcriber.prompt_identity, "transcript": transcript,
                          "usage": copy.deepcopy(transcriber.last_usage)}
                _atomic_json(cache_path, record)
                inference_seconds += float(transcriber.last_usage.get("inference_seconds", 0.0))
                new_count += 1
            row = copy.deepcopy(item)
            row["text"] = transcript
            row["metadata"] = {**row.get("metadata", {}), "text_provenance": provenance,
                               "text_status": "asr_completed", "asr_source_sha256": source_hash}
            completed.append(row)
            empty_count += transcript == ""
            progress.update(completed=len(completed), cached=cache_hits, new=new_count, empty_transcripts=empty_count,
                            fresh_inference_seconds=inference_seconds if new_count else None)
            _atomic_json(destination / "asr-preparation.json", progress)
            if progress_callback:
                progress_callback({**progress, "last_corpus_id": item["id"]})
    except Exception as error:
        progress.update(status="failed", error_type=type(error).__name__, error=str(error))
        _atomic_json(destination / "asr-preparation.json", progress)
        raise
    finally:
        transcriber.unload()
    # Evaluation files become available only after inference is complete.
    validate_dataset(source)
    queries = read_jsonl(source / "queries.jsonl")
    qrels = read_jsonl(source / "qrels.jsonl")
    _link_assets(source, destination, completed + queries)
    summary = {"view_identity": view_identity, "adapter_identity": transcriber.identity,
               "model_id": config["model_id"], "revision": config["revision"], "language": config["language"],
               "task": "transcribe", "prompt": "none", "source_dataset_id": source_manifest["id"],
               "source_revision": source_manifest["revision"], "source_only": True,
               "corpus_count": len(completed), "empty_transcripts": empty_count,
               "cache_hits": cache_hits, "new_transcripts": new_count,
               "fresh_inference_seconds": inference_seconds if new_count else None,
               "cache_reads_are_inference": False, "packages": package_versions(_ASR_PACKAGES),
               "device": config["device"], "dtype": config["dtype"]}
    manifest = write_dataset(destination, dataset_id=destination.name, track=source_manifest["track"],
        revision=stable_hash({"source": source_manifest["revision"], "asr": view_identity}),
        corpus=completed, queries=queries, qrels=qrels, sources=source_manifest.get("sources", []),
        license=source_manifest.get("license", "unspecified"), text_source="whisper_asr",
        expected_counts={key: source_manifest["counts"][key] for key in ("corpus", "queries", "qrels")},
        metadata={**source_manifest.get("metadata", {}), "asr": summary, "representation": "source_only_asr"})
    progress.update(status="completed", destination_id=manifest["id"])
    _atomic_json(destination / "asr-preparation.json", progress)
    return manifest


def _wer_tokens(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).translate(str.maketrans({"I": "ı", "İ": "i"})).casefold()
    text = "".join(" " if unicodedata.category(character).startswith("P") else character for character in text)
    return text.split()


def word_errors(reference: str, hypothesis: str) -> dict:
    """Levenshtein alignment, with explicit insertion/deletion/substitution totals."""
    gold, predicted = _wer_tokens(reference), _wer_tokens(hypothesis)
    if not gold:
        raise ValueError("WER requires a nonempty reference")
    # tuples: total errors, substitutions, deletions, insertions; tie policy fixed.
    previous = [(j, 0, 0, j) for j in range(len(predicted) + 1)]
    for i, word in enumerate(gold, 1):
        current = [(i, 0, i, 0)]
        for j, output in enumerate(predicted, 1):
            if word == output:
                current.append(previous[j - 1])
            else:
                options = []
                for base, column in ((previous[j - 1], 1), (previous[j], 2), (current[j - 1], 3)):
                    option = list(base)
                    option[0] += 1
                    option[column] += 1
                    options.append(tuple(option))
                current.append(min(options, key=lambda value: (value[0], -value[1], -value[2])))
        previous = current
    errors, substitutions, deletions, insertions = previous[-1]
    return {"errors": errors, "substitutions": substitutions, "deletions": deletions,
            "insertions": insertions, "reference_words": len(gold), "wer": errors / len(gold)}


def evaluate_asr_wer(source: Path | str, view: Path | str) -> dict:
    """Post-hoc evaluation only: references are never an ASR preparation input."""
    source, view = Path(source), Path(view)
    validate_dataset(view)
    manifest = json.loads((view / "dataset.json").read_text())
    provenance = manifest.get("metadata", {}).get("asr", {})
    if not provenance.get("source_only"):
        raise ValueError("WER view must declare source-only ASR provenance")
    original = json.loads((source / "dataset.json").read_text())
    if original["id"] != provenance["source_dataset_id"] or original["revision"] != provenance["source_revision"]:
        raise ValueError("WER references must belong to the ASR view's declared source dataset")
    references = read_jsonl(source / "evaluation_transcripts.jsonl")
    reference_by_id = {row["corpus_id"]: row["normalized_transcript"] if "normalized_transcript" in row else row["transcript"]
                       for row in references}
    corpus = read_jsonl(view / "corpus.jsonl")
    if len(reference_by_id) != len(references) or set(reference_by_id) != {row["id"] for row in corpus}:
        raise ValueError("WER references must exactly cover each candidate once")
    values = [word_errors(reference_by_id[row["id"]], row["text"]) for row in corpus]
    totals = {key: sum(value[key] for value in values)
              for key in ("errors", "substitutions", "deletions", "insertions", "reference_words")}
    result = {**totals, "wer": totals["errors"] / totals["reference_words"], "record_count": len(corpus),
              "empty_transcripts": sum(row["text"] == "" for row in corpus),
              "normalization": "NFKC; Turkish I/İ case mapping; casefold; Unicode punctuation to spaces; whitespace",
              "scope": "post_hoc_reference_evaluation_only", "asr_identity": manifest["metadata"]["asr"]["adapter_identity"]}
    _atomic_json(view / "asr-wer.json", result)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, nargs="?")
    parser.add_argument("destination", type=Path, nargs="?")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"))
    parser.add_argument("--language")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"))
    parser.add_argument("--score-wer", action="store_true")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text()) if args.config else {}
    config.update({key: getattr(args, key) for key in ("device", "language", "dtype") if getattr(args, key) is not None})
    if args.download:
        print(json.dumps({"snapshot": str(download_asr_model(config))}))
    if args.source is not None or args.destination is not None:
        if args.source is None or args.destination is None:
            parser.error("Source and destination are both required for preparation")
        result = prepare_asr_view(args.source, args.destination, config,
            progress_callback=lambda progress: print(json.dumps(progress), flush=True))
        print(json.dumps({"dataset_id": result["id"], "counts": result["counts"], "asr": result["metadata"]["asr"]}))
        if args.score_wer:
            print(json.dumps(evaluate_asr_wer(args.source, args.destination)))
    elif not args.download:
        parser.error("Provide source and destination or --download")


if __name__ == "__main__":
    main()

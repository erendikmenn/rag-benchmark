import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rag_benchmark import multimodal_asr as asr
from rag_benchmark.models import stable_hash
from rag_benchmark.multimodal import load_dataset
from rag_benchmark.multimodal_data import read_jsonl, write_dataset


def source_dataset(tmp_path):
    source = tmp_path / "source"
    (source / "media").mkdir(parents=True)
    for identifier in ("a", "b"):
        (source / "media" / (identifier + ".wav")).write_bytes((identifier * 8).encode())
    write_dataset(source, dataset_id="fleurs-tr_tr-test", track="speech", revision="source-revision",
        corpus=[{"id": identifier, "media": {"audio": f"media/{identifier}.wav"},
                 "metadata": {"text_status": "requires_asr"}} for identifier in ("a", "b")],
        queries=[{"id": "q", "text": "GOLD QUERY NEVER INTO ASR"}],
        qrels=[{"query_id": "q", "corpus_id": identifier, "relevance": 1} for identifier in ("a", "b")],
        sources=[], license="CC-BY-4.0", text_source="absent_requires_asr", metadata={"language": "tr"})
    (source / "evaluation_transcripts.jsonl").write_text("\n".join(json.dumps({
        "corpus_id": identifier, "transcript": "GOLD REFERENCE NEVER INTO ASR", "normalized_transcript": "bir iki"})
        for identifier in ("a", "b")))
    return source


class FakeTranscriber:
    calls = []
    fail_on = None
    unload_count = 0

    def __init__(self, config):
        self.config = config
        self.identity = stable_hash({"model": config["revision"], "language": config["language"]})
        self.prompt_identity = stable_hash(config["language"])
        self.last_usage = {}

    def transcribe(self, path, *, media_kind):
        assert isinstance(path, Path) and path.is_file()
        assert media_kind in {"audio", "video"}
        self.calls.append((path.name, media_kind))
        if path.name == self.fail_on:
            raise RuntimeError("interrupted")
        self.last_usage = {"inference_seconds": 2.0, "duration_seconds": 1.0}
        return "bir iki" if path.stem == "a" else ""

    def unload(self):
        type(self).unload_count += 1


@pytest.fixture
def fake_transcriber(monkeypatch):
    FakeTranscriber.calls = []
    FakeTranscriber.fail_on = None
    FakeTranscriber.unload_count = 0
    monkeypatch.setattr(asr, "WhisperTranscriber", FakeTranscriber)
    return FakeTranscriber


def test_asr_view_preserves_ids_media_labels_empty_output_and_source(tmp_path, fake_transcriber):
    source = source_dataset(tmp_path)
    original = (source / "corpus.jsonl").read_bytes()
    destination = tmp_path / "asr-view"
    result = asr.prepare_asr_view(source, destination, {"device": "cpu", "transcript_cache_dir": str(tmp_path / "cache")})
    assert fake_transcriber.calls == [("a.wav", "audio"), ("b.wav", "audio")]
    assert fake_transcriber.unload_count == 1
    corpus = read_jsonl(destination / "corpus.jsonl")
    assert [(row["id"], row["text"]) for row in corpus] == [("a", "bir iki"), ("b", "")]
    assert (source / "corpus.jsonl").read_bytes() == original
    assert read_jsonl(source / "queries.jsonl") == read_jsonl(destination / "queries.jsonl")
    assert read_jsonl(source / "qrels.jsonl") == read_jsonl(destination / "qrels.jsonl")
    assert (source / "media/a.wav").stat().st_ino == (destination / "media/a.wav").stat().st_ino
    assert result["text_ready"] is True and result["text_empty_count"] == 1
    assert result["metadata"]["asr"]["fresh_inference_seconds"] == 4.0
    assert "GOLD" not in json.dumps(corpus)
    load_dataset(destination).require_text()


def test_preparation_does_not_read_queries_or_references_during_transcription(tmp_path, fake_transcriber, monkeypatch):
    source = source_dataset(tmp_path)
    original_read = asr.read_jsonl
    reads = []
    def tracked_read(path):
        reads.append(path.name)
        if path.name in {"queries.jsonl", "qrels.jsonl", "evaluation_transcripts.jsonl"}:
            assert len(fake_transcriber.calls) == 2
        return original_read(path)
    monkeypatch.setattr(asr, "read_jsonl", tracked_read)
    asr.prepare_asr_view(source, tmp_path / "view", {"device": "cpu", "transcript_cache_dir": str(tmp_path / "cache")})
    assert "evaluation_transcripts.jsonl" not in reads


def test_cached_transcripts_resume_without_replaying_inference_latency(tmp_path, fake_transcriber):
    source = source_dataset(tmp_path)
    config = {"device": "cpu", "transcript_cache_dir": str(tmp_path / "cache")}
    asr.prepare_asr_view(source, tmp_path / "view", config)
    result = asr.prepare_asr_view(source, tmp_path / "view", config)
    assert len(fake_transcriber.calls) == 2
    summary = result["metadata"]["asr"]
    assert summary["cache_hits"] == 2 and summary["new_transcripts"] == 0
    assert summary["fresh_inference_seconds"] is None
    assert summary["cache_reads_are_inference"] is False


def test_failed_record_keeps_prior_checkpoint_and_no_ready_view(tmp_path, fake_transcriber):
    source = source_dataset(tmp_path)
    config = {"device": "cpu", "transcript_cache_dir": str(tmp_path / "cache")}
    fake_transcriber.fail_on = "b.wav"
    with pytest.raises(RuntimeError, match="interrupted"):
        asr.prepare_asr_view(source, tmp_path / "view", config)
    assert not (tmp_path / "view/dataset.json").exists()
    assert json.loads((tmp_path / "view/asr-preparation.json").read_text())["status"] == "failed"
    fake_transcriber.fail_on = None
    result = asr.prepare_asr_view(source, tmp_path / "view", config)
    assert fake_transcriber.calls == [("a.wav", "audio"), ("b.wav", "audio"), ("b.wav", "audio")]
    assert result["metadata"]["asr"]["cache_hits"] == 1
    assert result["metadata"]["asr"]["new_transcripts"] == 1


def test_source_bytes_and_model_revision_both_change_cache_key(tmp_path, fake_transcriber):
    source = source_dataset(tmp_path)
    config = {"device": "cpu", "transcript_cache_dir": str(tmp_path / "cache")}
    asr.prepare_asr_view(source, tmp_path / "view1", config)
    asr.prepare_asr_view(source, tmp_path / "view2", {**config, "revision": "a" * 40})
    assert len(fake_transcriber.calls) == 4
    # The cached model identity is never enough when source bytes change.
    item = read_jsonl(source / "corpus.jsonl")[0]
    _, audio = asr._resolve_media(source, item)
    before = asr.sha256(audio)
    audio.write_bytes(b"new source content")
    assert before != asr.sha256(audio)


def test_fleurs_language_translation_prompt_and_mutable_revision_are_rejected(tmp_path, fake_transcriber):
    source = source_dataset(tmp_path)
    for config in ({"language": "english"}, {"task": "translate"}, {"revision": "main"}, {"prompt": "GOLD"}):
        with pytest.raises(ValueError):
            asr.prepare_asr_view(source, tmp_path / "view", config)
    assert fake_transcriber.calls == []


def test_unsafe_media_path_fails_before_inference(tmp_path, fake_transcriber):
    source = source_dataset(tmp_path)
    rows = read_jsonl(source / "corpus.jsonl")
    rows[0]["media"]["audio"] = "../outside.wav"
    (tmp_path / "outside.wav").write_bytes(b"outside")
    (source / "corpus.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError, match="within"):
        asr.prepare_asr_view(source, tmp_path / "view", {"device": "cpu"})
    assert fake_transcriber.calls == []


def test_existing_other_view_is_not_overwritten(tmp_path, fake_transcriber):
    source = source_dataset(tmp_path)
    config = {"device": "cpu", "transcript_cache_dir": str(tmp_path / "cache")}
    asr.prepare_asr_view(source, tmp_path / "view", config)
    with pytest.raises(ValueError, match="different"):
        asr.prepare_asr_view(source, tmp_path / "view", {**config, "revision": "a" * 40})


def test_word_errors_and_posthoc_wer_include_empty_transcripts(tmp_path, fake_transcriber):
    assert asr.word_errors("IŞIK, İKİ!", "ışık iki")["wer"] == 0
    assert asr.word_errors("bir iki", "")["deletions"] == 2
    assert asr.word_errors("a b", "a c d")["errors"] == 2
    source = source_dataset(tmp_path)
    asr.prepare_asr_view(source, tmp_path / "view", {"device": "cpu", "transcript_cache_dir": str(tmp_path / "cache")})
    summary = asr.evaluate_asr_wer(source, tmp_path / "view")
    assert summary["record_count"] == 2 and summary["wer"] == 0.5
    assert summary["deletions"] == 2 and summary["empty_transcripts"] == 1
    assert "GOLD" not in json.dumps(summary)


def test_native_asr_call_uses_transcribe_language_no_prompt_and_checks_eos(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    path = tmp_path / "audio.wav"
    path.write_bytes(b"source")
    monkeypatch.setattr(asr, "load_audio", lambda path, rate: (np.zeros(16000, dtype=np.float32), {"duration_seconds": 1.0}))
    calls = []
    class Processor:
        feature_extractor = SimpleNamespace(hop_length=160, n_samples=480000)

        def __call__(self, waveform, **kwargs):
            assert kwargs["truncation"] is False and kwargs["return_attention_mask"] is True
            return {"input_features": torch.zeros((1, 128, 3000)), "attention_mask": torch.ones((1, 3000), dtype=torch.int64)}

        def batch_decode(self, sequences, **kwargs):
            return [""]
    class Model:
        generation_config = SimpleNamespace(eos_token_id=0)
        missing_eos = False

        def generate(self, **kwargs):
            calls.append(kwargs)
            return {"sequences": torch.tensor([[1, 2, 3 if self.missing_eos else 0]])}
    adapter = asr.WhisperTranscriber({"device": "cpu"})
    adapter._processor, adapter._model = Processor(), Model()
    monkeypatch.setattr(adapter, "_load", lambda: None)
    assert adapter.transcribe(path) == ""
    assert calls[0]["language"] == "turkish" and calls[0]["task"] == "transcribe"
    assert "prompt_ids" not in calls[0]
    assert adapter.last_usage["empty_transcript"] is True
    adapter._model.missing_eos = True
    with pytest.raises(RuntimeError, match="token limit"):
        adapter.transcribe(path)


def test_identity_changes_with_language_and_settings_but_not_cache_location():
    base = asr.WhisperTranscriber({"device": "cpu"})
    other_cache = asr.WhisperTranscriber({"device": "cpu", "cache_dir": "elsewhere", "transcript_cache_dir": "different"})
    assert base.identity == other_cache.identity
    assert base.identity != asr.WhisperTranscriber({"device": "cpu", "language": "en"}).identity
    assert base.identity != asr.WhisperTranscriber({"device": "cpu", "max_new_tokens": 128}).identity


def test_long_form_nonintegral_frame_is_kept_and_eos_checked(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    path = tmp_path / "audio.wav"
    path.write_bytes(b"source")
    monkeypatch.setattr(asr, "load_audio", lambda path, rate: (np.zeros(496001, dtype=np.float32), {"duration_seconds": 31.0000625}))
    class Processor:
        feature_extractor = SimpleNamespace(hop_length=160, n_samples=480000)

        def __call__(self, waveform, **kwargs):
            assert len(waveform) == 496001 and kwargs["truncation"] is False
            return {"input_features": torch.zeros((1, 128, 3100)), "attention_mask": torch.ones((1, 3100), dtype=torch.int64)}

        def batch_decode(self, sequences, **kwargs):
            return ["uzun kayıt"]
    class Model:
        generation_config = SimpleNamespace(eos_token_id=0)

        def generate(self, **kwargs):
            assert kwargs["return_segments"] is True and kwargs["return_timestamps"] is True
            return {"sequences": torch.tensor([[4, 5]]), "segments": [[{
                "result": {"sequences": torch.tensor([1, 4, 5, 0])}}]]}
    adapter = asr.WhisperTranscriber({"device": "cpu"})
    adapter._processor, adapter._model = Processor(), Model()
    monkeypatch.setattr(adapter, "_load", lambda: None)
    assert adapter.transcribe(path) == "uzun kayıt"
    assert adapter.last_usage["long_form"] is True
    assert adapter.last_usage["input_feature_frames"] == 3100


def test_mutating_asr_configuration_is_rejected_before_load():
    adapter = asr.WhisperTranscriber({"device": "cpu"})
    adapter.config["language"] = "english"
    with pytest.raises(ValueError, match="changed"):
        adapter._load()

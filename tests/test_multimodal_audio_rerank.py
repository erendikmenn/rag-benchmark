import copy
import json

import numpy as np
import pytest


from rag_benchmark.multimodal_audio_rerank import SourceAudioWindowReranker
from rag_benchmark.multimodal_models import load_audio

sf = pytest.importorskip("soundfile")
pytest.importorskip("scipy")


def source(tmp_path, name, samples, *, rate=16000, text=None):
    path = tmp_path / (name + ".wav")
    sf.write(path, np.asarray(samples, dtype=np.float32), rate, subtype="FLOAT")
    return {"id": name, "media": {"audio": str(path)}, **({"text": text} if text is not None else {})}


class FakeBase:
    pointwise = True

    def __init__(self, identity="test-base-v1"):
        self.identity = identity
        self.config = {"audio_max_duration_seconds": 30}
        self.calls = []

    def score(self, query, candidates):
        self.calls.append((copy.deepcopy(query), copy.deepcopy(candidates)))
        result = []
        for row in candidates:
            waveform, _ = load_audio(row["media"]["audio"], 16000)
            assert len(waveform) <= 30 * 16000
            result.append(-1.0 if waveform.max() > 0.5 else -7.0)
        return result


def test_actual_36_84_seconds_split_covers_all_samples_and_keeps_negative_max_and_source_text(tmp_path):
    waveform = np.zeros(int(36.84 * 16000), dtype=np.float32)
    waveform[-200:] = 0.75
    long = source(tmp_path, "long", waveform, text="Complete source-only ASR text")
    short = source(tmp_path, "short", [0.1] * 1600, text="Native short text")
    candidates = [short, long]
    original = copy.deepcopy(candidates)
    candidates[1]["metadata"] = {"gold_answer": "MUST NEVER REACH MODEL"}
    query = {"id": "q", "text": "complete query " * 60, "metadata": {"gold": "MUST NEVER REACH MODEL"}}
    base = FakeBase()
    wrapper = SourceAudioWindowReranker(base, cache_dir=tmp_path / "windows")
    assert wrapper.score(query, candidates) == [-7.0, -1.0]
    seen_query, seen = base.calls[0]
    assert seen_query == {"id": "q", "text": query["text"]}
    assert seen[0] == short
    assert [row["text"] for row in seen[1:]] == [long["text"], long["text"]]
    assert all("metadata" not in row for row in seen)
    pieces = [sf.read(row["media"]["audio"], dtype="float32")[0] for row in seen[1:]]
    assert [len(piece) for piece in pieces] == [480000, 109440]
    assert np.array_equal(np.concatenate(pieces), waveform)
    assert [row["text"] for row in candidates] == [row["text"] for row in original]
    usage = wrapper.last_usage
    assert usage["original_candidate_count"] == 2
    assert usage["audio_window_count"] == 3
    assert usage["segmented_candidate_count"] == usage["native_short_candidate_count"] == 1
    assert usage["source_audio_samples"] == usage["retained_audio_samples"] == len(waveform) + 1600
    assert usage["source_audio_duration_ms"] == usage["retained_audio_duration_ms"] == 36940
    assert usage["repeated_source_text_window_count"] == 2
    assert usage["items"][1]["windows"] == [
        {"sample_start": 0, "sample_end": 480000}, {"sample_start": 480000, "sample_end": 589440}]
    serialized = json.dumps(usage)
    assert str(tmp_path) not in serialized and long["text"] not in serialized and "gold_answer" not in serialized
    assert "no_cross_window_audio_evidence_synthesis" in serialized
    assert wrapper.score(query, [long]) == [-1.0]  # Pointwise score independent of other candidates.


def test_resample_complete_stereo_source_before_lossless_window_boundaries(tmp_path):
    stereo = np.column_stack([np.linspace(-0.3, 0.7, 9999), np.linspace(0.1, -0.2, 9999)])
    row = source(tmp_path, "stereo", stereo, rate=8000)
    expected, _ = load_audio(row["media"]["audio"], 16000)
    base = FakeBase()
    wrapper = SourceAudioWindowReranker(base, max_seconds=0.25, cache_dir=tmp_path / "windows")
    wrapper.score({"text": "q"}, [row])
    seen = base.calls[0][1]
    pieces = [sf.read(window["media"]["audio"], dtype="float32")[0] for window in seen]
    assert len(pieces) == 5
    assert all(0 < len(piece) <= 4000 for piece in pieces)
    assert np.array_equal(np.concatenate(pieces), expected)
    assert wrapper.last_usage["source_audio_samples"] == wrapper.last_usage["retained_audio_samples"] == 19998


def test_exact_limit_stays_native_and_one_extra_sample_is_not_dropped(tmp_path):
    exact = source(tmp_path, "exact", np.zeros(16000))
    extra = source(tmp_path, "extra", np.zeros(16001))
    base = FakeBase()
    wrapper = SourceAudioWindowReranker(base, max_seconds=1, cache_dir=tmp_path / "windows")
    wrapper.score({"text": "q"}, [extra, exact])
    seen = base.calls[0][1]
    assert [sf.info(row["media"]["audio"]).frames for row in seen] == [16000, 1, 16000]
    assert seen[-1] == exact
    assert wrapper.last_usage["source_audio_samples"] == wrapper.last_usage["retained_audio_samples"] == 32001


def test_cached_windows_are_validated_and_storage_location_does_not_change_identity(tmp_path):
    base = FakeBase()
    first = SourceAudioWindowReranker(base, 0.5, cache_dir=tmp_path / "first")
    second = SourceAudioWindowReranker(base, 0.5, cache_dir=tmp_path / "second")
    assert first.identity == second.identity
    row = source(tmp_path, "source", np.zeros(20000))
    first.score({"text": "q"}, [row])
    path = base.calls[0][1][0]["media"]["audio"]
    sf.write(path, np.ones(8000), 16000, subtype="FLOAT")
    with pytest.raises(ValueError, match="Cached source-audio window"):
        first.score({"text": "q"}, [row])
    assert len(base.calls) == 1
    assert first.last_usage == {}


@pytest.mark.parametrize("values", [[], [float("nan")], [float("inf")], [True], ["0.5"], [0.1, 0.2]])
def test_every_window_requires_one_finite_real_model_score(tmp_path, values):
    base = FakeBase()
    base.score = lambda query, candidates: values
    wrapper = SourceAudioWindowReranker(base, cache_dir=tmp_path / "windows")
    row = source(tmp_path, "short", np.zeros(16))
    with pytest.raises(RuntimeError, match="one finite real score"):
        wrapper.score({"text": "q"}, [row])
    assert wrapper.last_usage == {}


def test_base_context_failure_preserves_complete_query_and_audio_and_never_retries(tmp_path):
    base, received = FakeBase(), []

    def fail(query, candidates):
        received.append((query, candidates))
        raise ValueError("native pair context overflow")

    base.score = fail
    wrapper = SourceAudioWindowReranker(base, 1.0, cache_dir=tmp_path / "windows")
    row = source(tmp_path, "long", np.zeros(20000), text="full source text")
    query = {"text": "query " * 1000}
    with pytest.raises(ValueError, match="native pair context overflow"):
        wrapper.score(query, [row])
    assert len(received) == 1 and received[0][0] == query
    assert sum(sf.info(item["media"]["audio"]).frames for item in received[0][1]) == 20000
    assert all(item["text"] == row["text"] for item in received[0][1])
    assert wrapper.last_usage == {}


def test_identity_and_pointwise_contract_are_sealed(tmp_path):
    base = FakeBase()
    wrapper = SourceAudioWindowReranker(base, 1.0, cache_dir=tmp_path)
    assert wrapper.identity != SourceAudioWindowReranker(FakeBase("other-base"), 1.0).identity
    assert wrapper.identity != SourceAudioWindowReranker(base, 2.0).identity
    wrapper.config["max_window_samples"] = 1
    with pytest.raises(ValueError, match="configuration changed"):
        wrapper.score({"text": "q"}, [])
    wrapper = SourceAudioWindowReranker(base)
    base.identity = "preflight-changed-server"
    with pytest.raises(ValueError, match="finalize base preflight first"):
        wrapper.score({"text": "q"}, [])
    base.pointwise = False
    with pytest.raises(ValueError, match="pointwise"):
        SourceAudioWindowReranker(base)
    with pytest.raises(ValueError, match="native limit"):
        SourceAudioWindowReranker(FakeBase(), 31.0)


def test_identity_change_during_inference_and_source_mutation_fail_closed(tmp_path):
    base = FakeBase()
    wrapper = SourceAudioWindowReranker(base, cache_dir=tmp_path / "windows")
    row = source(tmp_path, "short", np.zeros(16))

    def change(query, candidates):
        base.identity = "changed"
        return [1.0]

    base.score = change
    with pytest.raises(ValueError, match="configuration changed"):
        wrapper.score({"text": "q"}, [row])
    base = FakeBase()
    wrapper = SourceAudioWindowReranker(base, cache_dir=tmp_path / "windows")

    def mutate(query, candidates):
        sf.write(row["media"]["audio"], np.ones(16), 16000, subtype="FLOAT")
        return [1.0]

    base.score = mutate
    with pytest.raises(ValueError, match="Source audio changed during relevance"):
        wrapper.score({"text": "q"}, [row])


def test_empty_and_invalid_candidates_never_drop_or_invent_scores(tmp_path):
    base = FakeBase()
    wrapper = SourceAudioWindowReranker(base, cache_dir=tmp_path / "windows")
    assert wrapper.score({"text": "q"}, []) == []
    assert not base.calls and wrapper.last_usage["audio_window_count"] == 0
    with pytest.raises(ValueError, match="unique nonempty"):
        wrapper.score({"text": "q"}, [{"id": "a"}, {"id": "a"}])
    with pytest.raises(ValueError, match="exactly one local audio"):
        wrapper.score({"text": "q"}, [{"id": "a", "media": {"image": "image.png"}}])
    with pytest.raises(ValueError, match="local files only"):
        wrapper.score({"text": "q"}, [{"id": "a", "media": {"audio": "https://example.invalid/audio"}}])

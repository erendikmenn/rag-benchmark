"""Keep gold annotations out of source descriptions and reject invalid rank scores."""
import json

import pytest

from rag_benchmark.multimodal_generation import (
    LocalMultimodalGenerator, MultimodalLayaReranker, media_content,
)


def test_description_does_not_serialize_gold_or_arbitrary_metadata(monkeypatch):
    generator = LocalMultimodalGenerator()
    captured = []
    monkeypatch.setattr(generator, "complete", lambda system, content: captured.append(content) or "A source.")
    assert generator.describe({"id": "not-model-input", "text": "Visible source", "media": {},
        "gold_answer": "DO NOT LEAK", "metadata": {"question": "SECRET QUESTION", "answer": "SECRET ANSWER"}}) == "A source."
    assert captured == [[{"type": "text", "text": "Visible source"}]]


def test_local_only_media_and_empty_source_rejected():
    with pytest.raises(ValueError, match="local"):
        media_content({"media": {"image": "https://example.org/source.png"}}, {})
    with pytest.raises(ValueError, match="no usable"):
        media_content({"metadata": {"answer": "not source content"}}, {})


@pytest.mark.parametrize("value", [True, -1, 4, 1.5, "3", None])
def test_gemma_relevance_rejects_invalid_labels(monkeypatch, value):
    generator = LocalMultimodalGenerator()
    monkeypatch.setattr(generator, "complete", lambda *args, **kwargs: json.dumps({"score": value}))
    with pytest.raises(ValueError, match="integer"):
        generator.score({"text": "request"}, [{"text": "source"}])


def test_gemma_relevance_keeps_candidate_order(monkeypatch):
    generator = LocalMultimodalGenerator()
    results = iter(['{"score":3}', '{"score":0}', '{"score":2}'])
    monkeypatch.setattr(generator, "complete", lambda *args, **kwargs: next(results))
    assert generator.score({"text": "request"}, [{"text": str(n)} for n in range(3)]) == [3.0, 0.0, 2.0]


def test_laya_pointwise_restores_original_order(monkeypatch):
    reranker = MultimodalLayaReranker({"device": "cpu"})
    monkeypatch.setattr(reranker.adapter, "rerank", lambda q, docs: [
        {"id": "b", "laya_score": 0.9}, {"id": "a", "laya_score": 0.1}])
    assert reranker.score({"text": "query"}, [{"id": "a", "text": "a"}, {"id": "b", "text": "b"}]) == [0.1, 0.9]


def test_laya_requires_full_composed_query():
    from rag_benchmark.multimodal import UnsupportedConfiguration
    reranker = MultimodalLayaReranker({"device": "cpu"})
    with pytest.raises(UnsupportedConfiguration, match="complete query"):
        reranker.score({"text": "make it red", "media": {"image": "reference.png"}}, [{"id": "a", "text": "source"}])


def test_laya_scores_explicit_empty_passages_with_actual_adapter(monkeypatch):
    reranker = MultimodalLayaReranker({"device": "cpu"})
    seen = []

    def score(query, documents):
        seen.extend(documents)
        return [{"id": "full", "laya_score": .8}, {"id": "empty", "laya_score": .37}]

    monkeypatch.setattr(reranker.adapter, "rerank", score)
    candidates = [{"id": "empty", "text": ""}, {"id": "full", "text": "source"}]
    assert reranker.score({"text": "query"}, candidates) == [.37, .8]
    assert seen == candidates
    assert reranker.last_usage["empty_candidate_text_count"] == 1
    from rag_benchmark.multimodal import UnsupportedConfiguration
    with pytest.raises(UnsupportedConfiguration, match="fixed candidate"):
        reranker.score({"text": "query"}, [{"id": "missing"}])
    with pytest.raises(ValueError, match="thresholds"):
        MultimodalLayaReranker({"threshold": .5})


def test_video_guard_uses_full_duration_before_capped_frame_sampling(monkeypatch, tmp_path):
    from rag_benchmark import multimodal_generation as generation
    path = tmp_path / "long.mp4"
    path.write_bytes(b"test video")
    monkeypatch.setattr(generation, "_video_duration_seconds", lambda path: 61.0)
    # The guard must fire before sampling: sample_video has no duration_seconds field.
    monkeypatch.setattr("rag_benchmark.multimodal_models.sample_video", lambda *args: pytest.fail("must reject before sampling"))
    with pytest.raises(ValueError, match="exceeds 60"):
        media_content({"media": {"video": str(path)}}, generation.E4B_CONFIG)


def test_actual_video_duration_and_frame_timestamps(tmp_path):
    av = pytest.importorskip("av")
    np = pytest.importorskip("numpy")
    from rag_benchmark.multimodal_generation import E4B_CONFIG, _video_duration_seconds
    path = tmp_path / "two-seconds.mp4"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=2)
        stream.width, stream.height, stream.pix_fmt = 16, 16, "yuv420p"
        for _ in range(4):
            frame = av.VideoFrame.from_ndarray(np.zeros((16, 16, 3), dtype=np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    assert _video_duration_seconds(path) == pytest.approx(2.0)
    content = media_content({"media": {"video": str(path)}}, E4B_CONFIG)
    assert content[0]["text"].endswith("source duration 2.000 seconds:")
    assert [row["text"] for row in content if row.get("type") == "text"][1:] == [
        "Frame at 0.000 seconds:", "Frame at 1.000 seconds:"]


def _mock_completion(monkeypatch, response, config=None):
    import httpx
    generator = LocalMultimodalGenerator({"max_tokens": 8, **(config or {})})
    generator._ready = True
    generator._context_limit = 64
    requests = []

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, json):
            requests.append(json)
            return httpx.Response(200, json=response, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "Client", lambda **kwargs: Client())
    return generator, requests


def test_completion_honors_hashed_settings_and_context_reservation(monkeypatch):
    result = {"choices": [{"message": {"content": "Good description."}, "finish_reason": "stop"}],
              "usage": {"prompt_tokens": 10, "completion_tokens": 3}}
    generator, requests = _mock_completion(monkeypatch, result, {"temperature": .2, "seed": 17})
    assert generator.complete("system", [{"type": "text", "text": "source"}]) == "Good description."
    assert requests[0]["seed"] == 17
    assert requests[0]["temperature"] == .2
    assert requests[0]["n_keep"] == -1
    assert generator.last_usage["context_limit"] == 64
    generator.config["seed"] = 18
    with pytest.raises(ValueError, match="configuration changed"):
        generator.complete("system", [{"type": "text", "text": "source"}])


@pytest.mark.parametrize("change,match", [
    ({"truncated": True}, "truncated input"),
    ({"usage": {"prompt_tokens": 60}}, "reserved output"),
    ({"usage": {}}, "prompt token usage"),
    ({"choices": [{"message": {"content": "incomplete"}, "finish_reason": "length"}]}, "finish normally"),
])
def test_completion_rejects_incomplete_context_and_output(monkeypatch, change, match):
    result = {"choices": [{"message": {"content": "description"}, "finish_reason": "stop"}],
              "usage": {"prompt_tokens": 10}, **change}
    generator, _ = _mock_completion(monkeypatch, result)
    with pytest.raises((ValueError, RuntimeError), match=match):
        generator.complete("system", [{"type": "text", "text": "source"}])


def test_preflight_requires_actual_server_context_and_projector(monkeypatch, tmp_path):
    import hashlib
    projector = tmp_path / "mmproj.gguf"
    projector.write_bytes(b"fixture")
    generator = LocalMultimodalGenerator({"projector_path": str(projector),
        "projector_sha256": hashlib.sha256(b"fixture").hexdigest()})
    monkeypatch.setattr(generator.verifier, "preflight", lambda: {"props": {"default_generation_settings": {"n_ctx": 8192}}})
    generator.preflight()
    assert generator._context_limit == 8192
    projector.write_bytes(b"modified")
    with pytest.raises(ValueError, match="projector checksum"):
        generator.preflight()


def _description_source(tmp_path):
    from rag_benchmark.multimodal_data import write_dataset
    source = tmp_path / "source"
    (source / "media").mkdir(parents=True)
    for name in ["a", "b"]:
        (source / "media" / f"{name}.png").write_bytes(("raw-" + name).encode())
    write_dataset(source, dataset_id="source", track="photo", revision="immutable-fixture", corpus=[
        {"id": name, "media": {"image": f"media/{name}.png"}, "metadata": {"gold_answer": "SECRET ANSWER"}}
        for name in ["a", "b"]], queries=[{"id": "q", "text": "SECRET GOLD QUERY"}],
        qrels=[{"query_id": "q", "corpus_id": "a", "relevance": 1}], sources=[], license="fixture",
        text_source="unavailable", metadata={"split": "validation"})
    return source


class _DescriptionGenerator:
    identity = "fixed-fixture-generator"
    last_usage = {"seconds": .01}

    def __init__(self):
        self.seen = []

    def preflight(self):
        return {}

    def describe(self, item):
        from pathlib import Path
        self.seen.append(item)
        return "Description " + Path(item["media"]["image"]).read_text()


@pytest.mark.parametrize("legacy_split", [False, True])
def test_caption_resumption_uses_only_sources_and_preserves_split(tmp_path, legacy_split):
    from rag_benchmark.multimodal_generation import prepare_described_view
    source = _description_source(tmp_path)
    if legacy_split:
        manifest = json.loads((source / "dataset.json").read_text())
        manifest.pop("split")
        (source / "dataset.json").write_text(json.dumps(manifest))
    destination = tmp_path / "described"
    generator = _DescriptionGenerator()
    first = prepare_described_view(source, destination, generator, max_new=1)
    assert first["status"] == "preparing"
    assert first["new_descriptions"] == 1
    assert not (destination / "dataset.json").exists()
    second = prepare_described_view(source, destination, generator)
    assert second["status"] == "ready"
    assert second["split"] == "validation"
    assert len(generator.seen) == 2  # First source was reused, not regenerated.
    assert "SECRET" not in json.dumps(generator.seen)
    assert all(set(item) == {"id", "media"} for item in generator.seen)
    assert (destination / "media/a.png").read_bytes() == (source / "media/a.png").read_bytes()


def test_caption_resume_rejects_stale_media_and_corrupt_text(tmp_path):
    from rag_benchmark.multimodal_generation import prepare_described_view
    source = _description_source(tmp_path)
    destination = tmp_path / "described"
    (destination / "media").mkdir(parents=True)
    (destination / "media/a.png").write_bytes(b"stale other source")
    generator = _DescriptionGenerator()
    with pytest.raises(ValueError, match="differs from the source"):
        prepare_described_view(source, destination, generator)
    cache = next((tmp_path / ".description-cache" / generator.identity).glob("*.json"))
    record = json.loads(cache.read_text())
    record["text"] = "   "
    cache.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="cache identity mismatch"):
        prepare_described_view(source, destination, generator)
    with pytest.raises(ValueError, match="must not overwrite"):
        prepare_described_view(source, source, generator)

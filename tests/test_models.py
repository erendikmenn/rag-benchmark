import json
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

from rag_benchmark.models import DenseEmbedder, LayaReranker, LocalGenerator, local_base_url


@pytest.mark.parametrize("url", ["https://api.openai.com/v1", "http://192.168.1.2/v1", "http://127.0.0.1.evil.test", "file:///tmp/model", "http://user@127.0.0.1", "http://127.0.0.1/v1?api_key=secret"])
def test_external_model_addresses_are_rejected(url):
    with pytest.raises(ValueError):
        local_base_url(url)


def test_loopback_normalization():
    assert local_base_url("http://localhost:8080/v1/") == "http://127.0.0.1:8080/v1"
    assert local_base_url("http://[::1]:8080/v1") == "http://[::1]:8080/v1"


def test_native_prefixes_and_dimensions():
    bge = DenseEmbedder("bge")
    gemma = DenseEmbedder("embeddinggemma")
    doc = {"text": "Ankara başkenttir.", "title": "Türkiye"}
    assert bge.format_query("Başkent?") == "Başkent?"
    assert gemma.format_query("Başkent?") == "task: search result | query: Başkent?"
    assert bge.format_document(doc) == "Türkiye\nAnkara başkenttir."
    assert gemma.format_document(doc) == "title: Türkiye | text: Ankara başkenttir."
    assert gemma.format_document({"text": "metin"}) == "title: none | text: metin"
    assert (bge.dimension, gemma.dimension) == (1024, 768)


def test_unsafe_precision_and_mutable_revisions_rejected():
    with pytest.raises(ValueError, match="float16"):
        DenseEmbedder("embeddinggemma", {"dtype": "float16"})
    with pytest.raises(ValueError, match="immutable"):
        DenseEmbedder("bge", {"revision": "main"})
    with pytest.raises(ValueError, match="native dimensions"):
        DenseEmbedder("bge", {"dimension": 768})
    with pytest.raises(ValueError, match="offline"):
        DenseEmbedder("bge", {"local_files_only": False})


def test_generator_sends_question_evidence_and_disabled_thinking(tmp_path):
    received = []
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"protocol-test-fixture-not-a-real-model")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            received.append((self.path, json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"choices": [{"message": {"content": "Ankara. [doc1]"}, "finish_reason": "stop"}], "usage": {"completion_tokens": 7}}).encode())

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            data = {"model_path": str(gguf)} if self.path == "/props" else {"data": [{"id": "gemma4"}]}
            self.wfile.write(json.dumps(data).encode())

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        generator = LocalGenerator({"base_url": f"http://127.0.0.1:{server.server_port}/v1", "model_path": str(gguf), "model_sha256": hashlib.sha256(gguf.read_bytes()).hexdigest()})
        runtime = generator.preflight()
        assert runtime["props"]["model_path"] == str(gguf)
        assert runtime["server_reports_build"] is False
        assert len(runtime["runtime_fingerprint"]) == 64
        assert generator.generate("Başkent neresi?", [{"id": "doc1", "text": "Başkent Ankara."}]) == "Ankara. [doc1]"
        assert received[0][0] == "/v1/chat/completions"
        payload = received[0][1]
        assert "Başkent neresi?" in payload["messages"][1]["content"]
        assert '"source_id": "doc1"' in payload["messages"][1]["content"]
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}
        assert payload["cache_prompt"] is False
        assert generator.last_usage["completion_tokens"] == 7
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


def test_generator_identity_requires_pinned_model():
    with pytest.raises(ValueError, match="immutable"):
        LocalGenerator({"model": "latest"})


@pytest.mark.parametrize("failure", ["path", "alias", "checksum"])
def test_generator_rejects_different_served_artifact(tmp_path, failure):
    gguf = tmp_path / "exact.gguf"
    gguf.write_bytes(b"fixture")
    config = {"model_path": str(gguf), "model_sha256": hashlib.sha256(b"fixture").hexdigest()}
    if failure == "checksum":
        config["model_sha256"] = "b" * 64
    generator = LocalGenerator(config)

    class Response:
        def __init__(self, value):
            self.value = value
        def raise_for_status(self):
            pass
        def json(self):
            return self.value

    class Client:
        def get(self, url):
            if url.endswith("/props"):
                return Response({"model_path": str(tmp_path / "wrong.gguf") if failure == "path" else str(gguf)})
            return Response({"data": [{"id": "wrong" if failure == "alias" else "gemma4"}]})

    with pytest.raises(RuntimeError):
        generator._verify_server(Client())
    assert generator._verified is False


def test_laya_reranks_probabilities_and_passes_actual_question(monkeypatch):
    states_seen = []

    class Agent:
        device = "cpu"
        cpu_fallback_count = 0

        def predict_batch(self, states, questions, **kwargs):
            states_seen.extend(states)
            return [{"answers": {"relevant": {"noul": x}}, "usage": {"truncated": False}} for x in (0.2, 0.9)]

    model = LayaReranker({"threshold": 0.5})
    monkeypatch.setattr(model, "_load", lambda: Agent())
    monkeypatch.setattr("rag_benchmark.models.synchronize", lambda device: None)
    result = model.rerank("Türkiye başkenti?", [{"id": "a", "text": "Kedi", "score": 8}, {"id": "b", "text": "Ankara", "score": 1}])
    assert [x["id"] for x in result] == ["b"]
    assert result[0]["retrieval_score"] == 1
    assert states_seen[1] == {"query": "Türkiye başkenti?", "passage": "Ankara"}


def test_laya_truncation_fails_instead_of_fabricating_ranks(monkeypatch):
    class Agent:
        device = "cpu"
        def predict_batch(self, *args, **kwargs):
            return [{"answers": {"relevant": {"noul": 0.9}}, "usage": {"truncated": True}}]
    model = LayaReranker({})
    monkeypatch.setattr(model, "_load", lambda: Agent())
    monkeypatch.setattr("rag_benchmark.models.synchronize", lambda device: None)
    with pytest.raises(ValueError, match="truncated"):
        model.rerank("q", [{"id": "d", "text": "x"}])

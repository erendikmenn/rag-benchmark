"""Injected fake CPU tensors/heads only; no real model or trained-weight load."""

import copy
import json
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pytest
from rag_benchmark import multimodal_bge_extra_backend as m
import test_multimodal_bge_extra_baseline as baseline_tests


class FakeBridge:
    identity = "verified-fixture-g"

    def __init__(self):
        self.config = {
            **m.native.MODEL_DEFAULTS["bge"],
            "dtype": "bfloat16",
            "device": "cpu",
            "batch_size": 1,
        }
        self.rows = {
            "document": [
                {
                    "id": "chunk0",
                    "source_id": "source",
                    "formatted_text": "A",
                    "char_start": 0,
                    "char_end": 1,
                },
                {
                    "id": "chunk1",
                    "source_id": "source",
                    "formatted_text": "B",
                    "char_start": 1,
                    "char_end": 2,
                },
            ],
            "query": [{"id": "q", "source_id": "q", "formatted_text": "Question"}],
        }
        self.vectors = {
            role: np.zeros((len(rows), 1024), dtype=np.float32) for role, rows in self.rows.items()
        }
        for values in self.vectors.values():
            values[:, 0] = 1
        self.seal = self.state()

    def state(self):
        return m.stable_hash(
            {
                "config": self.config,
                "rows": self.rows,
                "vectors": {k: m._sha(v) for k, v in self.vectors.items()},
            }
        )

    def validate_for_reuse(self):
        if self.state() != self.seal:
            raise ValueError("Fake frozen baseline changed")

    def verified_inputs(self, role):
        self.validate_for_reuse()
        return copy.deepcopy(self.rows[role]), self.vectors[role].view()


class Tokenizer:
    identity = "exact-fixture-tokenizer"

    def tokenize(self, text):
        return np.array([0, 10, 2], dtype=np.int64)


class Hidden:
    identity = "native-fixture-hidden"

    def __init__(self):
        self.calls = 0
        self.received = []

    def forward(self, tokens):
        torch = pytest.importorskip("torch")
        self.calls += 1
        self.received.append(copy.deepcopy(tokens))
        assert all(type(t) is int for row in tokens for t in row)
        width = max(map(len, tokens))
        hidden = torch.zeros((len(tokens), width, 1024), dtype=torch.bfloat16, device="cpu")
        hidden[:, :, 0] = 1
        ids = np.ones((len(tokens), width), dtype=np.int64)
        mask = np.zeros_like(ids)
        for i, row in enumerate(tokens):
            ids[i, : len(row)] = row
            mask[i, : len(row)] = 1
        return {"hidden": hidden, "token_ids": ids, "attention_mask": mask}


@pytest.fixture
def setup(monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setattr(
        m.native, "normalization_evidence", lambda: {"synthetic_audited_native_norm_count": 2}
    )
    monkeypatch.setattr(m.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0))
    bridge, tokenizer, hidden = FakeBridge(), Tokenizer(), Hidden()
    heads = {
        "sparse": (np.ones((1, 1024), dtype=np.float32), np.zeros(1, dtype=np.float32)),
        "colbert": (np.eye(1024, dtype=np.float32), np.zeros(1024, dtype=np.float32)),
    }
    backend = m.InjectedDualHeadBackend(bridge, tokenizer=tokenizer, hidden_backend=hidden, head_arrays=heads)
    return backend, bridge, tokenizer, hidden


def read_manifest(tmp_path, role="document"):
    path = next(tmp_path.glob("*/" + role + "/*/completed.json"))
    return path, m._read_json(path)


def test_both_heads_one_hidden_per_batch_roles_js_and_resume(setup, tmp_path):
    backend, _, _, hidden = setup
    result = backend.extract("document", tmp_path, max_batches=1)
    assert result["status"] == "preparing" and hidden.calls == 1
    result = backend.extract("document", tmp_path)
    assert (
        result["status"] == "completed"
        and result["new_rows"] == result["cached_rows"] == 1
        and hidden.calls == 2
    )
    _, manifest = read_manifest(tmp_path)
    assert manifest["request"]["role"] == "document"
    assert manifest["rows"][0]["record"]["source_id"] == "source"
    assert manifest["rows"][0]["record"]["char_start"] == 0
    assert manifest["rows"][0]["sparse"] == {"10": 1.0}
    assert manifest["native_dtype"] == "bfloat16" and manifest["token_storage"] == "float32"
    assert backend.extract("document", tmp_path)["new_rows"] == 0 and hidden.calls == 2
    result = backend.extract("query", tmp_path)
    assert result["role"] == "query" and hidden.calls == 3
    assert read_manifest(tmp_path, "query")[1]["rows"][0]["record"]["id"] == "q"
    assert result["new_dense_encoder_calls"] == 0 and not result["actual_model_compatibility_measured"]


@pytest.mark.parametrize(
    "damage", ["wrong_cls", "override", "tokens", "config", "runtime", "baseline", "token_mutation"]
)
def test_bad_native_callback_fail_closed_no_heads_or_retry(setup, tmp_path, damage):
    backend, bridge, _, hidden = setup
    original = hidden.forward

    def bad(tokens):
        value = original(tokens)
        if damage == "wrong_cls":
            value["hidden"][:, 0] *= -1
        elif damage == "override":
            value["native_dense"] = bridge.vectors["document"][:1]
        elif damage == "tokens":
            value["token_ids"][0, 1] = 11
        elif damage == "config":
            bridge.config["batch_size"] = 2
        elif damage == "runtime":
            hidden.identity = "changed"
        elif damage == "baseline":
            bridge.vectors["document"][0, 0] = -1
        else:
            tokens[0][1] = 11
        return value

    hidden.forward = bad
    with pytest.raises(ValueError):
        backend.extract("document", tmp_path)
    assert hidden.calls == 1 and not list(tmp_path.glob("**/completed.json"))
    if damage in ("config", "runtime", "baseline"):
        return
    with pytest.raises(ValueError):
        backend.extract("document", tmp_path)
    assert hidden.calls == 1


@pytest.mark.parametrize("damage", ["manifest", "token_file", "missing_file", "pending"])
def test_cache_corruption_never_hidden_fallback(setup, tmp_path, damage):
    backend, _, _, hidden = setup
    backend.extract("document", tmp_path, max_batches=1)
    path, _ = read_manifest(tmp_path)
    if damage == "manifest":
        data = json.loads(path.read_text())
        data["payload"]["rows"][0]["sparse"]["10"] = 999
        path.write_text(json.dumps(data))
    elif damage == "pending":
        path.unlink()
    elif damage == "missing_file":
        next(path.parent.glob("*.npy")).unlink()
    else:
        next(path.parent.glob("*.npy")).write_bytes(b"bad")
    with pytest.raises((ValueError, FileNotFoundError)):
        backend.extract("document", tmp_path)
    assert hidden.calls == 1


def test_long_numpy_tokens_bound_and_full_identity(setup, tmp_path):
    backend, bridge, tokenizer, hidden = setup
    tokenizer.tokenize = lambda text: np.array([0] + [10] * 1200 + [2], dtype=np.int64)
    backend.extract("query", tmp_path)
    assert len(hidden.received[0][0]) == 1202
    tokenizer.tokenize = lambda text: np.array([0] + [10] * 600 + [11] + [10] * 599 + [2], dtype=np.int64)
    with pytest.raises(ValueError, match="incompatible"):
        backend.extract("query", tmp_path)
    assert hidden.calls == 1
    tokenizer.tokenize = lambda text: np.array([0] + [10] * 8191 + [2], dtype=np.int64)
    with pytest.raises(ValueError, match="no truncation"):
        backend.extract("document", tmp_path)
    assert hidden.calls == 1


def test_process_lock_serializes_identical_requests(setup, tmp_path):
    backend, _, _, hidden = setup
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: backend.extract("query", tmp_path), range(2)))
    assert hidden.calls == 1 and sum(r["new_rows"] for r in results) == 1
    assert sum(r["cached_rows"] for r in results) == 1
    path, _ = read_manifest(tmp_path, "query")
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700


def test_tokenizer_mutation_on_cached_path_rejected(setup, tmp_path):
    backend, _, tokenizer, hidden = setup
    backend.extract("query", tmp_path)

    def bad(text):
        hidden.identity = "mutated tokenizer callback"
        return [0, 10, 2]

    tokenizer.tokenize = bad
    with pytest.raises(ValueError, match="changed"):
        backend.extract("query", tmp_path)
    assert hidden.calls == 1


def test_state_checksum_and_failed_status_cannot_resume(setup, tmp_path):
    backend, _, _, hidden = setup
    backend.extract("query", tmp_path)
    path, manifest = read_manifest(tmp_path, "query")
    m._json(
        path.parent / "state.json", {"status": "failed", "request_sha256": m.stable_hash(manifest["request"])}
    )
    with pytest.raises(ValueError, match="Failed/incompatible"):
        backend.extract("query", tmp_path)
    assert hidden.calls == 1


def test_legitimate_zero_and_epsilon_tokens_roundtrip(setup, tmp_path):
    backend, bridge, tokenizer, hidden = setup
    heads = {
        "sparse": (np.zeros((1, 1024), dtype=np.float32), np.zeros(1, dtype=np.float32)),
        "colbert": (np.zeros((1024, 1024), dtype=np.float32), np.zeros(1024, dtype=np.float32)),
    }
    # Tiny projected bias exercises official epsilon-normalization norms below1.
    heads["colbert"][1][0] = 5e-13
    alternate = m.InjectedDualHeadBackend(
        bridge, tokenizer=tokenizer, hidden_backend=hidden, head_arrays=heads
    )
    alternate.extract("query", tmp_path)
    assert alternate.extract("query", tmp_path)["cached_rows"] == 1
    _, manifest = read_manifest(tmp_path, "query")
    assert manifest["rows"][0]["sparse"] == {}
    assert hidden.calls == 1


def test_unverified_baseline_rejects_before_any_runtime():
    with pytest.raises(ValueError, match="Verified original G bridge"):
        m.InjectedDualHeadBackend(None, tokenizer=None, hidden_backend=None, head_arrays={})


def test_public_raw_cache_rejected_without_optional_models(tmp_path, monkeypatch):
    monkeypatch.setattr(m.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=1))
    with pytest.raises(ValueError, match="git-ignored"):
        m._private(tmp_path)


@pytest.fixture
def original_g_fixture(tmp_path):
    return baseline_tests.baseline_fixture.__wrapped__(tmp_path)


def test_real_verified_g_bridge_interface_synthetic_cache_only(original_g_fixture, tmp_path, monkeypatch):
    pytest.importorskip("torch")
    dataset, run, cache, _ = original_g_fixture
    baseline = m.bridge_module.VerifiedGBaseline(dataset, run, baseline_cache=cache)
    monkeypatch.setattr(
        m.native, "normalization_evidence", lambda: {"synthetic_audited_native_norm_count": 2}
    )
    monkeypatch.setattr(m.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0))
    hidden = Hidden()
    original = hidden.forward

    def native_float32(tokens):
        result = original(tokens)
        result["hidden"] = result["hidden"].float()
        return result

    hidden.forward = native_float32
    heads = {
        "sparse": (np.ones((1, 1024), dtype=np.float32), np.zeros(1, dtype=np.float32)),
        "colbert": (np.eye(1024, dtype=np.float32), np.zeros(1024, dtype=np.float32)),
    }
    backend = m.InjectedDualHeadBackend(
        baseline, tokenizer=Tokenizer(), hidden_backend=hidden, head_arrays=heads
    )
    result = backend.extract("query", tmp_path / "extra")
    assert result["status"] == "completed" and result["native_hidden_dtype"] == "float32"
    assert result["new_dense_encoder_calls"] == 0 and hidden.calls == 1
    assert backend.extract("query", tmp_path / "extra")["cached_rows"] == 1


@pytest.mark.parametrize("artifact", ["batch.lock", "0000.npy", None])
def test_abandoned_batch_never_restarts(setup, tmp_path, artifact):
    backend, _, _, hidden = setup
    directory = tmp_path / backend.identity / "query" / "000000000"
    directory.mkdir(parents=True)
    if artifact:
        (directory / artifact).write_bytes(b"orphan")
    with pytest.raises(ValueError, match="Interrupted preexisting"):
        backend.extract("query", tmp_path)
    assert hidden.calls == 0


def _process_reservation(root, directory, queue):
    with m._reserved_batch(root, directory, {"key": "same"}) as fresh:
        queue.put((fresh, m._read_json(directory / "state.json")["status"]))


def test_process_waiter_sees_durable_producer_state(tmp_path):
    import multiprocessing

    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    directory = tmp_path / "batch"
    with m._reserved_batch(tmp_path, directory, {"key": "same"}) as fresh:
        assert fresh
        child = ctx.Process(target=_process_reservation, args=(tmp_path, directory, queue))
        child.start()
        m._json(directory / "state.json", {"status": "completed"})
    child.join(timeout=10)
    assert child.exitcode == 0
    assert queue.get(timeout=2) == (False, "completed")
    queue.close()

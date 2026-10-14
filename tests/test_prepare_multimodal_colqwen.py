import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "colprep", Path(__file__).parents[1] / "scripts/prepare_multimodal_colqwen.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class FakeAdapter:
    identity = "adapter"
    config = {"local_files_only": True, "batch_size": 1, "dtype": "bfloat16"}

    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail
        self.unloaded = False

    def _cache_key(self, item, role):
        assert not {"gold", "answers", "metadata"} & set(item)
        return role + "-" + item["id"]

    def rank(self, *args):
        raise AssertionError("search forbidden")

    def _cached_tokens(self, items, role, cache):
        self.calls.append((role, items))
        if self.fail:
            raise RuntimeError("private error not printed")
        p = cache / "tokens" / role
        p.mkdir(parents=True, exist_ok=True)
        a = np.zeros((3, 128), dtype=np.float32)
        a[:, 0] = 1
        np.save(p / (self._cache_key(items[0], role) + ".npy"), a)
        return [a], {}

    def unload(self):
        self.unloaded = True


def fixture(tmp_path, fail=False):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    for name in ["dataset.json", "corpus.jsonl", "queries.jsonl", "qrels.jsonl"]:
        (dataset / name).write_text("{}\n")
    code = tmp_path / "code"
    code.write_bytes(b"code")
    model = tmp_path / "model"
    model.write_bytes(b"model")
    d = SimpleNamespace(
        track="document",
        identity="frozen",
        corpus=[{"id": "d", "gold": "secret"}],
        queries=[{"id": "q", "answers": ["secret"]}],
    )
    d.model_item = lambda item: {"id": item["id"], "text": "input", "media": {}}
    a = FakeAdapter(fail)

    def record(p):
        return {"path": str(p), "sha256": m.sha(p), "size": p.stat().st_size}

    value = {
        "version": 1,
        "artifact_root": str(tmp_path),
        "dataset_root": str(dataset),
        "frozen_identity": "frozen",
        "cache_dir": "cache",
        "adapter_config": a.config,
        "expected_adapter_identity": "adapter",
        "code_files": [record(code)],
        "model_files": [record(model)],
        "existing_tokens": {},
    }
    return value, d, a, lambda _: (d, a)


def test_dry_run_never_encodes(tmp_path):
    s, _, a, factory = fixture(tmp_path)
    result = m.prepare(s, tmp_path / "state", factory=factory)
    assert result["missing_items"] == 2 and not a.calls
    assert not (tmp_path / "state").exists()


def test_per_item_complete_manifest_no_labels_search(tmp_path):
    s, _, a, factory = fixture(tmp_path)
    result = m.prepare(s, tmp_path / "state", execute=True, factory=factory)
    assert result["new_items"] == 2 and a.unloaded
    manifest = json.loads((tmp_path / "state/import-manifest.json").read_text())
    sets = manifest["collections"][0]["sets"]
    assert [(x["role"], x["kind"], x["dimension"]) for x in sets] == [
        ("document", "ragged", 128),
        ("query", "ragged", 128),
    ]
    assert all(x["blocks"][0]["shape"] == [3, 128] for x in sets)
    a.calls.clear()
    assert m.prepare(s, tmp_path / "state", execute=True, resume=True, factory=factory)["new_items"] == 0
    assert not a.calls


def test_failed_state_not_auto_reset(tmp_path):
    s, _, a, factory = fixture(tmp_path, fail=True)
    with pytest.raises(RuntimeError):
        m.prepare(s, tmp_path / "state", execute=True, factory=factory)
    state = json.loads((tmp_path / "state/state.json").read_text())
    assert state["status"] == "failed" and state["error_type"] == "RuntimeError" and a.unloaded
    with pytest.raises(ValueError, match="explicit_resume"):
        m.prepare(s, tmp_path / "state", execute=True, resume=True, factory=factory)


def test_unsealed_existing_tokens_rejected(tmp_path):
    s, _, a, factory = fixture(tmp_path)
    a._cached_tokens([{"id": "d"}], "document", tmp_path / "cache")
    with pytest.raises(ValueError, match="unsealed"):
        m.prepare(s, tmp_path / "state", factory=factory)


@pytest.mark.parametrize("condition", ["source", "adapter", "model"])
def test_pin_fail_closed(tmp_path, condition):
    s, d, _, factory = fixture(tmp_path)
    if condition == "source":
        d.identity = "changed"
    if condition == "adapter":
        s["expected_adapter_identity"] = "changed"
    if condition == "model":
        (tmp_path / "model").write_bytes(b"changed")
    with pytest.raises(ValueError):
        m.prepare(s, tmp_path / "state", factory=factory)


def test_source_changed_after_encode_fails(tmp_path):
    s, d, _, factory = fixture(tmp_path)
    calls = 0

    def changed(_):
        nonlocal calls
        calls += 1
        if calls == 2:
            d.identity = "changed"
        return factory(_)

    with pytest.raises(ValueError, match="dataset_changed"):
        m.prepare(s, tmp_path / "state", execute=True, factory=changed)
    assert json.loads((tmp_path / "state/state.json").read_text())["status"] == "failed"


def test_interrupt_durable_and_explicit_resume(tmp_path):
    s, _, a, factory = fixture(tmp_path)
    original = a._cached_tokens

    def interrupted(*args):
        raise KeyboardInterrupt()

    a._cached_tokens = interrupted
    with pytest.raises(KeyboardInterrupt):
        m.prepare(s, tmp_path / "state", execute=True, factory=factory)
    assert json.loads((tmp_path / "state/state.json").read_text())["status"] == "interrupted"
    with pytest.raises(ValueError, match="explicit_resume"):
        m.prepare(s, tmp_path / "state", execute=True, factory=factory)
    a._cached_tokens = original
    assert (
        m.prepare(s, tmp_path / "state", execute=True, resume=True, factory=factory)["status"] == "completed"
    )


def test_model_symlink_is_confined_and_hashed(tmp_path):
    s, _, _, factory = fixture(tmp_path)
    p = tmp_path / "model-link"
    p.symlink_to(tmp_path / "model")
    s["model_files"][0]["path"] = str(p)
    assert m.prepare(s, tmp_path / "state", factory=factory)["model_loaded"] is False
    p.unlink()
    p.symlink_to("/etc/hosts")
    with pytest.raises(ValueError, match="unsafe_path"):
        m.prepare(s, tmp_path / "state", factory=factory)


def test_deleted_completed_token_never_reencodes(tmp_path):
    s, _, a, factory = fixture(tmp_path)
    m.prepare(s, tmp_path / "state", execute=True, factory=factory)
    a.calls.clear()
    (tmp_path / "cache/tokens/document/document-d.npy").unlink()
    with pytest.raises(ValueError, match="sealed_token_missing_or_changed"):
        m.prepare(s, tmp_path / "state", execute=True, resume=True, factory=factory)
    assert not a.calls
    assert json.loads((tmp_path / "state/state.json").read_text())["status"] == "completed"


def test_initial_sealed_missing_never_encodes(tmp_path):
    s, _, a, factory = fixture(tmp_path)
    s["existing_tokens"] = {"cache/tokens/document/document-d.npy": "0" * 64}
    with pytest.raises(ValueError, match="sealed_token_missing_or_changed"):
        m.prepare(s, tmp_path / "state", execute=True, factory=factory)
    assert not a.calls


def test_interrupted_sealed_corruption_never_encodes(tmp_path):
    s, _, a, factory = fixture(tmp_path)
    m.prepare(s, tmp_path / "state", execute=True, factory=factory)
    state_path = tmp_path / "state/state.json"
    state = json.loads(state_path.read_text())
    state["status"] = "interrupted"
    state_path.write_text(json.dumps(state))
    (tmp_path / "cache/tokens/document/document-d.npy").write_bytes(b"corrupt")
    a.calls.clear()
    with pytest.raises(ValueError, match="sealed_token_missing_or_changed"):
        m.prepare(s, tmp_path / "state", execute=True, resume=True, factory=factory)
    assert not a.calls


def test_atomic_fsync_file_and_directory(tmp_path, monkeypatch):
    calls = []
    original = m.os.fsync

    def record(fd):
        calls.append(fd)
        original(fd)

    monkeypatch.setattr(m.os, "fsync", record)
    m.atomic(tmp_path / "state.json", {"state": "safe"})
    assert len(calls) == 2
    assert json.loads((tmp_path / "state.json").read_text()) == {"state": "safe"}

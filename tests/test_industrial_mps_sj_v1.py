"""Synthetic only: no actual corpus, snapshot, model constructor, MPS or network."""
import io
import json
import os
from pathlib import Path
import struct
import types

import numpy as np
import pytest

from rag_benchmark import industrial_mps_sj_v1 as run


class FakeAdapter:
    identity = "synthetic-native"
    config = {"synthetic": True}
    projection_verification = {"projection_verified": True, "max_absolute_merge_error": 0.0}

    def __init__(self, channel="J", fail=False):
        self.channel, self.fail, self.calls = channel, fail, 0

    def encode(self, items, role):
        self.calls += 1
        if self.fail:
            raise RuntimeError("synthetic interrupted forward")
        item = items[0]
        modalities = ["image", "text"] if role == "document" else ["text"]
        row = {"modalities": modalities, "expanded_tokens": 7}
        if role == "document":
            row["derived_text_empty"] = not item["text"].strip()
        self.last_usage = {"inference_seconds": .25, "items": [row], "devices": ["mps:0"],
                           "dtype": "bfloat16", "context_limit": 8192, "towers": {"vision": "fake", "audio": "fake"}}
        values = np.zeros((1, 768), dtype=np.float32)
        values[0, 0] = 1
        return values

    def encode_tokens(self, items, role):
        self.calls += 1
        if self.fail:
            raise RuntimeError("synthetic interrupted forward")
        self.last_usage = {"inference_seconds": .25, "items": 1, "device": "mps", "dtype": "bfloat16",
                           "context_limit": 8192, "max_image_patches": 768, "token_counts": [2]}
        value = np.zeros((2, 128), dtype=np.float32)
        value[:, 0] = 1
        return [value]

    def unload(self):
        pass


def jobs(tmp_path, channel="J", role="query", rows=None, adapter=None, resume=False):
    directory = tmp_path / "cache"
    directory.mkdir(exist_ok=True)
    rows = rows or [{"id": "q1", "text": "synthetic", "media": {}}]
    adapter = adapter or FakeAdapter(channel)
    value = run.encode_jobs(tmp_path, {"media": {}}, channel, role, rows, directory, adapter, resume, "manifest", "channel")
    return value, adapter


@pytest.mark.parametrize("channel", ["J", "S"])
def test_own_terminal_resume_has_zero_fresh_time(tmp_path, channel):
    (_, first), adapter = jobs(tmp_path, channel)
    (_, cached), again = jobs(tmp_path, channel, resume=True)
    assert adapter.calls == 1 and again.calls == 0
    assert first == {"total_items": 1, "fresh_inference_items": 1, "validated_cache_items": 0,
                     "fresh_inference_seconds": .25}
    assert cached == {"total_items": 1, "fresh_inference_items": 0, "validated_cache_items": 1,
                      "fresh_inference_seconds": None}


@pytest.mark.parametrize("channel", ["J", "S"])
def test_failed_forward_cannot_be_repeated(tmp_path, channel):
    adapter = FakeAdapter(channel, fail=True)
    with pytest.raises(RuntimeError, match="interrupted"):
        jobs(tmp_path, channel, adapter=adapter)
    retry = FakeAdapter(channel)
    with pytest.raises(ValueError, match="terminal"):
        jobs(tmp_path, channel, adapter=retry, resume=True)
    assert adapter.calls == 1 and retry.calls == 0
    assert json.loads((tmp_path / "cache/000000.terminal.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("suffix", [".claim.json", ".npy"])
def test_ambiguous_job_never_calls_adapter(tmp_path, suffix):
    (tmp_path / "cache").mkdir()
    (tmp_path / ("cache/000000" + suffix)).write_text("interrupted")
    adapter = FakeAdapter()
    with pytest.raises(ValueError, match="Ambiguous"):
        jobs(tmp_path, adapter=adapter, resume=True)
    assert adapter.calls == 0


def test_existing_terminal_requires_explicit_resume(tmp_path):
    jobs(tmp_path)
    with pytest.raises(ValueError, match="Resume"):
        jobs(tmp_path)


def test_changed_query_or_order_rejects_before_forward(tmp_path):
    jobs(tmp_path)
    adapter = FakeAdapter()
    with pytest.raises(ValueError, match="claim"):
        jobs(tmp_path, rows=[{"id": "q1", "text": "changed", "media": {}}], adapter=adapter, resume=True)
    assert adapter.calls == 0


def test_extra_cache_file_rejects_before_forward(tmp_path):
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache/extra").write_text("x")
    adapter = FakeAdapter()
    with pytest.raises(ValueError, match="Unexpected"):
        jobs(tmp_path, adapter=adapter)
    assert adapter.calls == 0


def test_broken_terminal_symlink_rejected_before_forward(tmp_path):
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache/000000.terminal.json").symlink_to(tmp_path / "missing")
    adapter = FakeAdapter()
    with pytest.raises(ValueError, match="Unexpected"):
        jobs(tmp_path, adapter=adapter, resume=True)
    assert adapter.calls == 0


def test_corrupted_vector_and_escape_terminal_rejected(tmp_path):
    jobs(tmp_path)
    data = tmp_path / "cache/000000.npy"
    data.write_bytes(data.read_bytes() + b"x")
    with pytest.raises(ValueError, match="Sealed"):
        jobs(tmp_path, resume=True)
    terminal_path = tmp_path / "cache/000000.terminal.json"
    terminal = json.loads(terminal_path.read_text())
    terminal["output"]["path"] = "elsewhere.npy"
    terminal_path.write_text(json.dumps(terminal))
    with pytest.raises(ValueError, match="outside"):
        jobs(tmp_path, resume=True)


@pytest.mark.parametrize("channel,shape", [("J", (1, 768)), ("S", (7, 128))])
def test_bounded_normalized_vector_roundtrip(channel, shape):
    value = np.zeros(shape, dtype=np.float32)
    value[:, 0] = 1
    raw = run.vector_bytes(value, channel)
    assert np.array_equal(run.read_vector(raw, channel), value)
    with pytest.raises(ValueError, match="payload"):
        run.read_vector(raw + b"trailing", channel)
    with pytest.raises(ValueError, match="payload"):
        run.read_vector(raw[:-1], channel)


@pytest.mark.parametrize("kind", ["pickle", "nonfinite", "zero", "wrong_dtype", "too_many", "header"])
def test_bad_vector_rejected(kind):
    value = np.zeros((1, 768), dtype=np.float32)
    value[:, 0] = 1
    if kind == "pickle":
        stream = io.BytesIO()
        np.save(stream, np.array([[object()]], dtype=object), allow_pickle=True)
        raw = stream.getvalue()
    elif kind == "header":
        raw = b"\x93NUMPY\x02\x00" + struct.pack("<I", 100000) + b"x"
    else:
        if kind == "nonfinite":
            value[0, 1] = np.nan
        elif kind == "zero":
            value[:] = 0
        elif kind == "wrong_dtype":
            value = value.astype(np.float64)
        elif kind == "too_many":
            value = np.repeat(value, 2, axis=0)
        stream = io.BytesIO()
        np.save(stream, value, allow_pickle=False)
        raw = stream.getvalue()
    with pytest.raises(ValueError):
        run.read_vector(raw, "J")


def test_joint_diagnostics_exact_full_modalities():
    fake = FakeAdapter()
    item = {"id": "d", "text": "", "media": {}}
    fake.encode([item], "document")
    run.checked_usage(fake.last_usage, item, "document", "J")
    fake.last_usage["items"][0]["modalities"] = ["image"]
    with pytest.raises(ValueError, match="joint"):
        run.checked_usage(fake.last_usage, item, "document", "J")


def test_source_loaded_globals_and_defaults_guards(monkeypatch):
    from rag_benchmark import multimodal
    path = Path(multimodal.__file__).resolve()
    run.verify_module(multimodal, path)
    original = multimodal.fuse_rankings
    clone = types.FunctionType(original.__code__, dict(original.__globals__), original.__name__,
                               original.__defaults__, original.__closure__)
    clone.__kwdefaults__ = original.__kwdefaults__.copy()
    monkeypatch.setattr(multimodal, "fuse_rankings", clone)
    with pytest.raises(ValueError, match="globals"):
        run.verify_module(multimodal, path)


def test_source_mutated_adapter_constant_rejected(monkeypatch):
    from rag_benchmark import multimodal_colqwen
    monkeypatch.setitem(multimodal_colqwen.COLQWEN_DEFAULTS, "device", "cpu")
    with pytest.raises(ValueError, match="constant"):
        run.verify_module(multimodal_colqwen, Path(multimodal_colqwen.__file__).resolve())


def test_source_method_default_rejected(monkeypatch):
    from rag_benchmark import multimodal_models
    method = multimodal_models.EmbeddingGemma2Adapter.encode
    monkeypatch.setattr(method, "__defaults__", ("document",))
    with pytest.raises(ValueError, match="defaults"):
        run.verify_module(multimodal_models, Path(multimodal_models.__file__).resolve())


def test_source_import_binding_rejected(monkeypatch):
    from rag_benchmark import multimodal_models
    monkeypatch.setattr(multimodal_models, "normalized", lambda *a: None)
    with pytest.raises(ValueError, match="code"):
        run.verify_module(multimodal_models, Path(multimodal_models.__file__).resolve())


def test_source_closure_imports_no_models():
    import subprocess
    command = [sys_executable(), "-B", "-c", "from pathlib import Path; import sys; "
               "from rag_benchmark import industrial_mps_sj_v1 as r; p=Path.cwd(); "
               "r.sources(p,{x:r.frozen.record(p,x)['sha256'] for x in r.SOURCES}); "
               "assert not any(x in sys.modules for x in ('torch','transformers','sentence_transformers'))"]
    assert subprocess.run(command, capture_output=True, text=True).returncode == 0


def sys_executable():
    import sys
    return sys.executable


def test_lock_preserved_busy_and_replaced(tmp_path):
    directory = tmp_path / "work/overnight"
    directory.mkdir(parents=True)
    path = directory / "gpu.lock"
    path.touch()
    expected = run.lock_record(tmp_path)
    with run.ExistingLock(tmp_path, expected):
        with pytest.raises(BlockingIOError):
            with run.ExistingLock(tmp_path, expected):
                pytest.fail("second owner")
    assert run.lock_record(tmp_path) == expected and path.read_bytes() == b""
    old = directory / "old.lock"
    path.rename(old)
    path.touch()
    with pytest.raises(ValueError, match="inode"):
        with run.ExistingLock(tmp_path, expected):
            pytest.fail("replaced lock")


def test_lock_missing_never_created(tmp_path):
    with pytest.raises(FileNotFoundError):
        run.lock_record(tmp_path)
    assert not (tmp_path / "work").exists()


def synthetic_assets(tmp_path):
    snapshots = {}
    for key, (_, revision) in run.MODEL_REVISIONS.items():
        origin = tmp_path / "original" / revision
        origin.mkdir(parents=True)
        files = {"config.json": b"{}", "weights.safetensors": b"synthetic weights"}
        if key == "S":
            files["adapter_config.json"] = json.dumps({"base_model_name_or_path": run.MODEL_REVISIONS["S_base"][0]}).encode()
            files["benchmark-model.json"] = b'{"original": true}'
            files["1_Dense/config.json"] = b"{}"
            files["1_Dense/model.safetensors"] = b"projection"
        assets = {}
        for name, value in files.items():
            target = origin / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(value)
            assets[name] = {"file": run.frozen.record(tmp_path, target.relative_to(tmp_path).as_posix())}
        snapshots[key] = {"revision": revision, "assets": assets}
    return {"assets": snapshots}


def test_overlay_all_metadata_private_nested_projection_preserved(tmp_path):
    manifest = synthetic_assets(tmp_path)
    originals = {p: p.read_bytes() for p in (tmp_path / "original").rglob("*") if p.is_file()}
    paths = run.local_models(tmp_path, manifest, "S", tmp_path / "view")
    for key in ("S", "S_base"):
        for path in paths[key].rglob("*"):
            if path.is_file():
                assert path.is_symlink() == path.name.endswith(".safetensors")
    config = json.loads((paths["S"] / "adapter_config.json").read_text())
    assert config["base_model_name_or_path"] == str(paths["S_base"])
    (paths["S"] / "benchmark-model.json").write_text("private rewrite")
    assert all(p.read_bytes() == value for p, value in originals.items())
    assert (paths["S"] / "1_Dense/model.safetensors").is_symlink()


def test_missing_snapshot_does_not_download(tmp_path):
    with pytest.raises(FileNotFoundError, match="downloads"):
        run.asset_inventory(tmp_path, "J")
    assert not (tmp_path / ".cache").exists()


def test_manifest_drift_rejected_before_validation(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(run, "validate", lambda *a: calls.append("validate"))
    with pytest.raises(ValueError, match="manifest"):
        run.execute(tmp_path, {"x": "changed"}, run.frozen.digest({"x": "original"}), "J")
    assert calls == []


def test_condition_preserves_inherited_hardware_and_distinct_channels():
    manifest = {"dataset_identity": run.DATASET_IDENTITY}
    assert run.HARDWARE["B"]["hardware"] == "CPU"
    assert all(run.HARDWARE[c]["hardware"] == "NVIDIA L4" for c in "GEN")
    assert run.stage_identity(manifest, "manifest", "S") != run.stage_identity(manifest, "manifest", "J")
    assert run.stage_identity(manifest, "manifest", "J") != run.stage_identity(manifest, "changed", "J")


def test_output_permissions_and_exclusive_creation(tmp_path):
    path = tmp_path / "sealed.json"
    run.frozen.exclusive_write(path, b"{}")
    assert os.stat(path).st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        run.frozen.exclusive_write(path, b"changed")
    assert path.read_bytes() == b"{}"


def test_positive_stage_closes_only_sealed_rankings(tmp_path, monkeypatch):
    from rag_benchmark import multimodal
    (tmp_path / "work/overnight").mkdir(parents=True)
    (tmp_path / "work/overnight/gpu.lock").touch()
    (tmp_path / run.BASE).mkdir()
    manifest = {"dataset_identity": run.DATASET_IDENTITY, "lock": run.lock_record(tmp_path), "media": {}}
    original = run.frozen.canonical(manifest)
    corpus = [{"id": "d2", "text": "two", "media": {}}, {"id": "d1", "text": "one", "media": {}}]
    queries = [{"id": "q", "text": "query", "media": {}}]
    monkeypatch.setattr(run, "validate", lambda *a: ({"multimodal": multimodal}, corpus, queries, {}, [set()]))
    adapter = FakeAdapter()
    monkeypatch.setattr(run, "local_models", lambda *a: {})
    monkeypatch.setattr(run, "make_adapter", lambda *a: adapter)
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    result = run.execute(tmp_path, manifest, run.frozen.digest(manifest), "J")
    closed = json.loads(run.frozen.checked(tmp_path, result))
    assert closed["status"] == "completed" and len(closed["terminal_records"]) == 3
    payload = json.loads(run.frozen.checked(tmp_path, closed["files"]["rankings.json"]))
    assert [r["id"] for r in payload["rankings"][0]] == ["d1", "d2"]
    assert payload["query_ids"] == ["q"]
    report = json.loads(run.frozen.checked(tmp_path, closed["files"]["report.json"]))
    assert report["none_cells_completed"] == 0 and report["laya_cells_completed"] == 0
    assert report["usage"]["document"]["fresh_inference_items"] == 2
    assert report["usage"]["query"]["fresh_inference_items"] == 1
    assert adapter.calls == 3 and run.frozen.canonical(manifest) == original
    with pytest.raises(ValueError, match="immutable"):
        run.execute(tmp_path, manifest, run.frozen.digest(manifest), "J", resume=True)
    assert adapter.calls == 3


def test_bounded_cache_file_rejects_size_before_read(tmp_path):
    path = tmp_path / "oversized.npy"
    with path.open("wb") as handle:
        handle.truncate(100000)
    with pytest.raises(ValueError, match="Oversized"):
        run.bounded_bytes(path, 20)


def test_cached_token_count_and_projection_are_revalidated(tmp_path):
    jobs(tmp_path, "S")
    path = tmp_path / "cache/000000.terminal.json"
    terminal = json.loads(path.read_text())
    terminal["usage"]["token_counts"] = [3]
    path.write_text(json.dumps(terminal))
    with pytest.raises(ValueError, match="counts"):
        jobs(tmp_path, "S", resume=True)
    terminal["usage"]["token_counts"] = [2]
    terminal["projection_verification"] = {}
    path.write_text(json.dumps(terminal))
    with pytest.raises(ValueError, match="projection"):
        jobs(tmp_path, "S", resume=True)


@pytest.mark.parametrize("missing", ["1_Dense/config.json", "1_Dense/model.safetensors"])
def test_missing_nested_dense_asset_blocks_preparation(tmp_path, monkeypatch, missing):
    model, revision = run.MODEL_REVISIONS["S"]
    snapshot = tmp_path / (".cache/models/models--" + model.replace("/", "--") + "/snapshots/" + revision)
    names = ["modules.json", "config_sentence_transformers.json", "sentence_bert_config.json", "tokenizer.json",
             "tokenizer_config.json", "preprocessor_config.json", "merges.txt", "vocab.json", "special_tokens_map.json",
             "added_tokens.json", "adapter_config.json", "adapter_model.safetensors", "1_Dense/config.json",
             "1_Dense/model.safetensors", "chat_template.jinja", "video_preprocessor_config.json"]
    for name in names:
        if name != missing:
            path = snapshot / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}")
    monkeypatch.setitem(run.CONFIG_HASHES, "S", {})
    with pytest.raises(FileNotFoundError, match="required local model assets"):
        run.asset_inventory(tmp_path, "S")


def test_symlink_metadata_copied_not_followed_for_writes(tmp_path):
    manifest = synthetic_assets(tmp_path)
    paths = run.local_models(tmp_path, manifest, "J", tmp_path / "view")
    meta = paths["J"] / "config.json"
    assert not meta.is_symlink()
    meta.write_text('{"private":true}')
    original = manifest["assets"]["J"]["assets"]["config.json"]["file"]
    assert run.frozen.checked(tmp_path, original) == b"{}"

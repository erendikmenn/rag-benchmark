"""V2 shared-blob delta only; the immutable V1 suite supplies 38 baseline checks."""
import ast
import hashlib
from pathlib import Path
import subprocess
import sys

import pytest

from rag_benchmark import industrial_mps_sj_v2 as run


def example(root):
    cache = root / ".cache/models"
    snapshot = cache / "models--synthetic--model/snapshots/revision"
    repository = cache / "models--synthetic--model/blobs"
    shared = cache / "blobs/ab/value"
    snapshot.mkdir(parents=True)
    repository.mkdir()
    shared.parent.mkdir(parents=True)
    shared.write_bytes(b"synthetic weights")
    (repository / "hub-key").symlink_to("../../blobs/ab/value")
    logical = snapshot / "model.safetensors"
    logical.symlink_to("../../blobs/hub-key")
    return logical.relative_to(root).as_posix(), repository, shared


def test_two_hop_shared_blob_chain_and_final_sha(tmp_path):
    logical, repository, shared = example(tmp_path)
    before = {p: p.lstat() for p in (tmp_path / logical, repository / "hub-key", shared)}
    chain = run.asset_chain(tmp_path, logical)
    assert [x["path"] for x in chain["symlink_chain"]] == [logical, (repository / "hub-key").relative_to(tmp_path).as_posix()]
    assert chain["resolved_path"] == shared.relative_to(tmp_path).as_posix()
    final = run.stream_record(tmp_path, chain["resolved_path"])
    assert final["sha256"] == hashlib.sha256(b"synthetic weights").hexdigest()
    assert run.asset_chain(tmp_path, logical) == chain
    for path, status in before.items():
        assert path.lstat().st_ino == status.st_ino
        assert path.lstat().st_mtime_ns == status.st_mtime_ns


def test_direct_file_and_single_hop(tmp_path):
    cache = tmp_path / ".cache/models"
    cache.mkdir(parents=True)
    direct = cache / "config.json"
    direct.write_text("{}")
    chain = run.asset_chain(tmp_path, direct.relative_to(tmp_path).as_posix())
    assert chain["symlink_chain"] == []
    alias = cache / "alias.json"
    alias.symlink_to(direct)
    assert len(run.asset_chain(tmp_path, alias.relative_to(tmp_path).as_posix())["symlink_chain"]) == 1


def test_retargeted_intermediate_link_changes_binding(tmp_path):
    logical, repository, shared = example(tmp_path)
    before = run.asset_chain(tmp_path, logical)
    alternate = shared.parent / "identical-bytes-other-inode"
    alternate.write_bytes(shared.read_bytes())
    (repository / "hub-key").unlink()
    (repository / "hub-key").symlink_to(alternate)
    after = run.asset_chain(tmp_path, logical)
    assert after != before and after["symlink_chain"][0] == before["symlink_chain"][0]


def test_escape_from_cache_rejected(tmp_path):
    logical, repository, _ = example(tmp_path)
    outside = tmp_path / "outside-model-cache"
    outside.write_bytes(b"x")
    (repository / "hub-key").unlink()
    (repository / "hub-key").symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        run.asset_chain(tmp_path, logical)


def test_cycle_rejected(tmp_path):
    logical, repository, _ = example(tmp_path)
    (repository / "hub-key").unlink()
    (repository / "hub-key").symlink_to(tmp_path / logical)
    with pytest.raises(ValueError, match="Cyclic"):
        run.asset_chain(tmp_path, logical)


def test_missing_final_asset_rejected(tmp_path):
    logical, _, shared = example(tmp_path)
    shared.unlink()
    with pytest.raises(FileNotFoundError):
        run.asset_chain(tmp_path, logical)


def test_noncanonical_logical_path_rejected(tmp_path):
    example(tmp_path)
    with pytest.raises(ValueError, match="Canonical"):
        run.asset_chain(tmp_path, ".cache/models/../models/anything")


def test_v2_only_changes_asset_chain_and_versioned_references():
    base = Path(run.__file__).parent
    v1 = ast.parse((base / "industrial_mps_sj_v1.py").read_text())
    normalized = (base / "industrial_mps_sj_v2.py").read_text().replace("industrial_mps_sj_v2", "industrial_mps_sj_v1")
    normalized = normalized.replace("industrial-l4-bgen-local-mps-sj-v2", "industrial-l4-bgen-local-mps-sj-v1")
    normalized = normalized.replace("prepared-manifest-v2.json", "prepared-manifest-v1.json").replace('"/actual-v2"', '"/actual-v1"')
    v2 = ast.parse(normalized)
    def filtered(tree):
        return [ast.dump(n, include_attributes=False) for n in tree.body
                if not isinstance(n, ast.FunctionDef) or n.name not in {"asset_chain", "asset_inventory"}]
    assert filtered(v1) == filtered(v2)


def test_v2_pinned_source_guard_without_model_imports():
    code = ("from pathlib import Path; import sys; from rag_benchmark import industrial_mps_sj_v2 as r; "
            "p=Path.cwd(); r.sources(p,{x:r.frozen.record(p,x)['sha256'] for x in r.SOURCES}); "
            "assert not any(x in sys.modules for x in ('torch','transformers','sentence_transformers'))")
    result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

"""New path-authority delta only; old loader/model/cache suites are not replayed."""
import ast
import hashlib
import importlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from rag_benchmark.multimodal import Dataset

MODULE = os.environ.get("FLEURS_PATH_REGRESSION_MODULE", "rag_benchmark.fleurs_remaining_data_v2")
m = importlib.import_module(MODULE)


def reference(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return reference(path)


@pytest.fixture
def seam(tmp_path, monkeypatch):
    canonical, asr = tmp_path / "canonical", tmp_path / "asr"
    canonical.mkdir()
    asr.mkdir()
    raw = b'{"id":"source1","text":"synthetic source","media":{}}\n'
    (canonical / "queries.jsonl").write_bytes(raw)
    (asr / "queries.jsonl").write_bytes(raw)
    source_ref = reference(canonical / "queries.jsonl")
    corpus = [{"id": "d1", "text": "synthetic ASR", "media": {}}]
    source = Dataset(asr, {"track": "speech", "text_provenance": {
        "source": "whisper_asr", "query_independent": True, "gold_fields_used": False}},
        corpus, [json.loads(raw)], {"source1": {"d1": 1}}, "old-source")
    pilot = SimpleNamespace(source_dataset=source, dataset=SimpleNamespace(identity="old25"),
        provenance={"query_lineage": {"source_ref": source_ref}, "documents": {}})
    selected = {"source": source_ref}
    config = {"protocol": m.PROTOCOL, "module_sha256": reference(Path(m.__file__))["sha256"],
        "pilot_manifest": write_json(tmp_path / "pilot.json", {"synthetic": True}),
        "selection": write_json(tmp_path / "selection.json", selected),
        "drafts": write_json(tmp_path / "drafts.json", {"synthetic": True}),
        "meaning_review": write_json(tmp_path / "review.json", {"synthetic": True})}
    config_ref = write_json(tmp_path / "config.json", config)
    observed = []
    monkeypatch.setattr(m, "load_verified_pilot", lambda *a, **kw: pilot)
    # This test isolates the existing load_verified_remaining path gate. The
    # unchanged derive_queries function's 304/qrel contract has its prior tests.
    def derive(*args):
        observed.append(args[-1])
        return [{"id": "ep001", "text": "synthetic paraphrase", "media": {}}], {"ep001": {"d1": 1}}, []
    monkeypatch.setattr(m, "derive_queries", derive)
    monkeypatch.setattr(m, "REPOSITORY_ROOT", tmp_path, raising=False)
    return SimpleNamespace(root=tmp_path, canonical=canonical, asr=asr, raw=raw, pilot=pilot,
                           selected=selected, config=config, config_ref=config_ref, observed=observed)


def test_real_loader_seam_accepts_authoritative_original_path_not_asr_copy(seam):
    f = seam
    assert not (f.canonical / "queries.jsonl").samefile(f.asr / "queries.jsonl")
    assert (f.canonical / "queries.jsonl").read_bytes() == (f.asr / "queries.jsonl").read_bytes()
    result = m.load_verified_remaining(f.config_ref["path"], expected_sha256=f.config_ref["sha256"])
    assert f.observed == [f.raw] and result.source_dataset.root == f.asr
    assert result.dataset.model_item(result.dataset.queries[0]) == {
        "id": "ep001", "text": "synthetic paraphrase", "media": {}}


@pytest.mark.parametrize("kind", ["equal_bytes_wrong_path", "asr_copy", "changed_sha", "different_evidence",
    "leaf_symlink", "parent_symlink", "outside_root", "dotdot", "relative", "fifo", "directory"])
def test_narrow_authority_binding_rejects_other_paths_and_types(seam, kind):
    f = seam
    selected = json.loads(json.dumps(f.selected))
    expected = f.pilot.provenance["query_lineage"]["source_ref"]
    if kind == "equal_bytes_wrong_path":
        other = f.root / "samebytes.jsonl"
        other.write_bytes(f.raw)
        selected["source"] = reference(other)
    elif kind == "asr_copy":
        selected["source"] = reference(f.asr / "queries.jsonl")
    elif kind == "changed_sha":
        expected["sha256"] = selected["source"]["sha256"] = "f" * 64
    elif kind == "different_evidence":
        selected["source"]["unbound_extra"] = True
    elif kind == "leaf_symlink":
        link = f.root / "linked.jsonl"
        link.symlink_to(f.canonical / "queries.jsonl")
        expected["path"] = selected["source"]["path"] = str(link)
    elif kind == "parent_symlink":
        link = f.root / "linked-parent"
        link.symlink_to(f.canonical, target_is_directory=True)
        expected["path"] = selected["source"]["path"] = str(link / "queries.jsonl")
    elif kind == "outside_root":
        monkey_root = f.root / "unrelated-root"
        monkey_root.mkdir()
        m.REPOSITORY_ROOT = monkey_root
    elif kind == "dotdot":
        path = str(f.asr / ".." / "canonical" / "queries.jsonl")
        expected["path"] = selected["source"]["path"] = path
    elif kind == "relative":
        expected["path"] = selected["source"]["path"] = "canonical/queries.jsonl"
    elif kind == "fifo":
        path = f.root / "pipe.jsonl"
        os.mkfifo(path)
        expected["path"] = selected["source"]["path"] = str(path)
    else:
        expected["path"] = selected["source"]["path"] = str(f.canonical)
    with pytest.raises((ValueError, OSError)):
        m.authoritative_source_bytes(selected, f.pilot)


def test_authority_reads_only_the_bound_file_once(seam, monkeypatch):
    calls = []
    original = m.os.open
    def opened(path, flags, *args, **kwargs):
        calls.append((str(path), flags))
        return original(path, flags, *args, **kwargs)
    monkeypatch.setattr(m.os, "open", opened)
    assert m.authoritative_source_bytes(seam.selected, seam.pilot) == seam.raw
    assert len(calls) == 1 and calls[0][0] == seam.selected["source"]["path"]
    assert calls[0][1] & os.O_NOFOLLOW and calls[0][1] & os.O_NONBLOCK


def test_other_bridge_functions_and_types_are_exactly_preserved():
    root = Path(__file__).resolve().parents[1]
    old = ast.parse((root / "src/rag_benchmark/fleurs_remaining_data.py").read_text())
    new = ast.parse(Path(m.__file__).read_text())
    def functions(tree):
        return {node.name: ast.dump(node) for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    old, new = functions(old), functions(new)
    assert set(new) == set(old) | {"authoritative_source_bytes"}
    assert all(old[name] == new[name] for name in old if name != "load_verified_remaining")


def test_runner_revision_only_rebinds_new_bridge():
    root = Path(__file__).resolve().parents[1]
    old = (root / "scripts/run_fleurs_remaining_quality.py").read_bytes()
    new = (root / "scripts/run_fleurs_remaining_quality_v2.py").read_bytes()
    assert hashlib.sha256(old).hexdigest() == "7851a3e8228562abb8146c232d8e026446a0fba6685861e7a6f4929c11b792ca"
    assert new == old.replace(b"from rag_benchmark import fleurs_remaining_data\n",
        b"from rag_benchmark import fleurs_remaining_data_v2 as fleurs_remaining_data\n")

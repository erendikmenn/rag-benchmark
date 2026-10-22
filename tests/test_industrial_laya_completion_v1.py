"""Source-only Industrial admission tests; no runner/model/runtime imports or payload reads."""
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
NEW_SOURCE = ROOT / "src/rag_benchmark/industrial_laya_completion_v1.py"
OLD_SOURCE = ROOT / "src/rag_benchmark/vidore_laya_completion_v4.py"
NEW_CLI = ROOT / "scripts/run_industrial_laya_completion_v1.py"
OLD_CLI = ROOT / "scripts/run_vidore_laya_completion_v4.py"
FROZEN_V4_SOURCE_SHA = "48bc81edb11baabc383695f0f24392496fa6bc0a40abeed397c27c9ff4482d9e"
FROZEN_V4_CLI_SHA = "db8b7b12b5cae354ab1c403d8b45eed6865d80f5d732e8192c81e4cf30ec8ac8"
INDUSTRIAL_CONSTANTS = {
    "INDUSTRIAL_IDENTITY", "INDUSTRIAL_NONE_BASE", "INDUSTRIAL_MPS_BASE", "INDUSTRIAL_HARDWARE",
}


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def reference(path):
    return {"path": path, "bytes": 123, "sha256": digest(path.encode())}


def assigned_name(node):
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        return node.targets[0].id
    return None


@pytest.fixture
def gate():
    """Execute only the actual gate/helper definitions, never runner imports or startup."""
    tree = ast.parse(NEW_SOURCE.read_text())
    constants = INDUSTRIAL_CONSTANTS | {"CHANNELS", "TARGETS"}
    function_names = {"industrial_inputs", "require", "sha", "canonical"}
    nodes = [
        node for node in tree.body
        if assigned_name(node) in constants or
        isinstance(node, ast.FunctionDef) and node.name in function_names
    ]
    assert {assigned_name(node) for node in nodes if isinstance(node, ast.Assign)} == constants
    assert {node.name for node in nodes if isinstance(node, ast.FunctionDef)} == function_names
    def unread(*_args):
        raise AssertionError("The in-memory read_json fixture was not installed")
    namespace = {"hashlib": hashlib, "frozen": SimpleNamespace(canonical=canonical, read_json=unread)}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(NEW_SOURCE), "exec"), namespace)
    return SimpleNamespace(**namespace)


@pytest.fixture
def bundle(gate, monkeypatch):
    base, mps_base = gate.INDUSTRIAL_NONE_BASE, gate.INDUSTRIAL_MPS_BASE
    refs = {
        "handoff": reference(base + "/six-ranking-inputs.json"),
        "none_closed": reference(base + "/closed.json"),
        "actual_review": reference("work/synthetic/independent-industrial-review.json"),
        "baseline": reference(base + "/report.json"),
        "mps": reference(mps_base + "/prepared-manifest-v2.json"),
        "S": reference(mps_base + "/actual-v2/S/closed.json"),
        "J": reference(mps_base + "/actual-v2/J/closed.json"),
        "inherited_S_review": reference("work/synthetic/independent-S-actual-review.json"),
        "inherited_J_review": reference("work/synthetic/independent-J-actual-review.json"),
    }
    rankings = {channel: reference(base + "/" + channel + "-ranking.json") for channel in gate.CHANNELS}
    origins = {
        channel: {
            **gate.INDUSTRIAL_HARDWARE[channel],
            "report": reference(base + "/" + channel + "-origin-report.json"),
            "channel_identity": digest(("synthetic-identity-" + channel).encode()),
        }
        for channel in gate.CHANNELS
    }
    accepted_names = [
        "handoff", "none_closed", "actual_review", "mps", "S", "J",
        "inherited_S_review", "inherited_J_review",
    ]
    manifest = {
        "dataset_identity": gate.INDUSTRIAL_IDENTITY,
        "industrial_evidence": {
            **{name: copy.deepcopy(refs[name]) for name in ("handoff", "none_closed", "actual_review")},
            "actual_review_status": "PASS_SYNTHETIC_INDEPENDENT_INDUSTRIAL_ACTUAL_REVIEW",
        },
        "rankings": rankings, "retrieval_provenance": origins,
        "baseline_report": copy.deepcopy(refs["baseline"]),
        "accepted_evidence": [copy.deepcopy(refs[name]) for name in accepted_names],
    }
    mixed, cpu_manifest = digest(b"mixed-condition"), digest(b"cpu-manifest")
    mps = {
        "version": "industrial-l4-bgen-local-mps-sj-v2",
        "dataset_id": "vidore-v3-industrial-en", "dataset_identity": gate.INDUSTRIAL_IDENTITY,
    }
    documents = {
        "handoff": {
            "protocol": "industrial-none-derivation-v1", "dataset_id": "vidore-v3-industrial-en",
            "dataset_identity": gate.INDUSTRIAL_IDENTITY,
            "rankings": copy.deepcopy(rankings), "retrieval_provenance": copy.deepcopy(origins),
            "baseline_report": copy.deepcopy(refs["baseline"]), "mixed_condition_identity": mixed,
            "input_manifest": copy.deepcopy(refs["mps"]),
            "stage_closures": {channel: copy.deepcopy(refs[channel]) for channel in "SJ"},
            "accepted_evidence": [copy.deepcopy(refs[name]) for name in
                                  ("inherited_S_review", "inherited_J_review")],
        },
        "none_closed": {
            "status": "closed", "completed_cells": 126, "prior_cell_agreements": 8,
            "mixed_condition_identity": mixed, "manifest_sha256": cpu_manifest,
            "files": {"six-ranking-inputs.json": copy.deepcopy(refs["handoff"]),
                      "report.json": copy.deepcopy(refs["baseline"]),
                      **{channel + "-ranking.json": copy.deepcopy(rankings[channel])
                         for channel in gate.CHANNELS},
                      **{channel + "-origin-report.json": copy.deepcopy(origins[channel]["report"])
                         for channel in gate.CHANNELS}},
        },
        "actual_review": {
            "status": manifest["industrial_evidence"]["actual_review_status"], "blocking_findings": [],
        },
        "baseline": {
            "derivation_protocol": "industrial-none-derivation-v1",
            "retrieval_protocol": "industrial-l4-bgen-local-mps-sj-v2",
            "mixed_condition_identity": mixed, "source_manifest_sha256": cpu_manifest,
            "prior_cell_agreements": 8,
            "original_standalone_evidence": [{"synthetic_cell": i} for i in range(8)],
            "new_model_calls": 0, "new_embedding_rows": 0, "gpu_calls": 0, "laya_cells_completed": 0,
        },
        "mps": mps,
    }
    for channel in "SJ":
        documents[channel] = {
            "status": "completed", "manifest_sha256": digest(canonical(mps)),
            "identity": origins[channel]["channel_identity"],
            "files": {
                "rankings.json": {**copy.deepcopy(rankings[channel]),
                                  "path": mps_base + "/actual-v2/" + channel + "/rankings.json"},
                "report.json": {**copy.deepcopy(origins[channel]["report"]),
                                "path": mps_base + "/actual-v2/" + channel + "/report.json"},
            },
        }
    calls = []
    by_path = {refs[name]["path"]: name for name in documents}
    expected_refs = copy.deepcopy(refs)
    root = Path("/synthetic-industrial-root")
    def read_json(actual_root, ref):
        assert actual_root == root
        name = by_path[ref["path"]]
        if ref != expected_refs[name]:
            raise ValueError("synthetic_seal_mismatch")
        calls.append(copy.deepcopy(ref))
        return documents[name]
    monkeypatch.setattr(gate.frozen, "read_json", read_json)
    return SimpleNamespace(manifest=manifest, documents=documents, refs=refs, calls=calls, root=root)


def test_normalized_source_and_cli_only_extend_the_industrial_gate():
    old_raw, new_raw = OLD_SOURCE.read_bytes(), NEW_SOURCE.read_bytes()
    assert digest(old_raw) == FROZEN_V4_SOURCE_SHA
    assert digest(OLD_CLI.read_bytes()) == FROZEN_V4_CLI_SHA
    normalized = new_raw.decode().replace("industrial_laya_completion_v1", "vidore_laya_completion_v4")
    normalized = normalized.replace("industrial-laya-sealed-union-v1", "vidore-laya-sealed-union-v4")
    normalized = normalized.replace("industrial-laya-completion-v1", "vidore-laya-completion-v4")
    old, new = ast.parse(old_raw), ast.parse(normalized)
    old_target = next(node for node in old.body if assigned_name(node) == "TARGETS")
    new_target = next(node for node in new.body if assigned_name(node) == "TARGETS")
    assert ast.literal_eval(new_target.value) == {"vidore-v3-industrial-en": (5244, 283)}
    assert {assigned_name(node) for node in new.body
            if (assigned_name(node) or "").startswith("INDUSTRIAL_")} == INDUSTRIAL_CONSTANTS
    assert len([node for node in new.body
                if isinstance(node, ast.FunctionDef) and node.name == "industrial_inputs"]) == 1
    retained = []
    for node in new.body:
        name = assigned_name(node)
        if name in INDUSTRIAL_CONSTANTS or isinstance(node, ast.FunctionDef) and node.name == "industrial_inputs":
            continue
        if name == "TARGETS":
            node = copy.deepcopy(old_target)
        if isinstance(node, ast.FunctionDef) and node.name == "prepare_manifest":
            required = next(item for item in node.body if assigned_name(item) == "required")
            assert isinstance(required.value, ast.Set)
            assert sum(isinstance(item, ast.Constant) and item.value == "industrial_evidence"
                       for item in required.value.elts) == 1
            required.value.elts = [item for item in required.value.elts
                                   if not isinstance(item, ast.Constant) or item.value != "industrial_evidence"]
        if isinstance(node, ast.FunctionDef) and node.name == "validate":
            added = [item for item in node.body if isinstance(item, ast.Expr)
                     and isinstance(item.value, ast.Call) and isinstance(item.value.func, ast.Name)
                     and item.value.func.id == "industrial_inputs"]
            assert len(added) == 1 and node.body.index(added[0]) == 4
            assert ast.dump(added[0]) == ast.dump(ast.parse("industrial_inputs(root, manifest)").body[0])
            node.body.remove(added[0])
        retained.append(node)
    new.body = retained
    assert ast.dump(new) == ast.dump(old)
    assert NEW_CLI.read_text().replace("industrial_laya_completion_v1", "vidore_laya_completion_v4") == OLD_CLI.read_text()


def test_industrial_inputs_accepts_only_sealed_metadata_and_preserves_freshness(gate, bundle):
    before = copy.deepcopy((bundle.manifest, bundle.documents))
    assert gate.TARGETS == {"vidore-v3-industrial-en": (5244, 283)}
    assert gate.industrial_inputs(bundle.root, bundle.manifest) is None
    assert bundle.calls == [bundle.refs[name] for name in
                            ("handoff", "none_closed", "actual_review", "baseline", "mps", "S", "J")]
    required_refs = [bundle.refs[name] for name in
                     ("handoff", "none_closed", "actual_review", "mps", "S", "J",
                      "inherited_S_review", "inherited_J_review")]
    assert all(ref in bundle.manifest["accepted_evidence"] for ref in required_refs)
    assert all(ref in bundle.manifest["accepted_evidence"] + [bundle.manifest["baseline_report"]]
               for ref in bundle.calls)
    assert (bundle.manifest, bundle.documents) == before


@pytest.mark.parametrize("target,key,value,reason", [
    ("actual_review", "status", "SOURCE_ONLY", "industrial_actual_review"),
    ("actual_review", "blocking_findings", ["synthetic blocker"], "industrial_actual_review"),
    ("none_closed", "status", "pending", "industrial_none_closed"),
    ("none_closed", "completed_cells", 125, "industrial_none_closed"),
    ("none_closed", "prior_cell_agreements", 7, "industrial_none_closed"),
    ("none_closed", "mixed_condition_identity", "wrong", "industrial_none_closed"),
    ("baseline", "source_manifest_sha256", "wrong", "industrial_none_provenance"),
    ("baseline", "original_standalone_evidence", [], "industrial_none_provenance"),
    ("baseline", "new_model_calls", 1, "industrial_none_provenance"),
    ("baseline", "new_embedding_rows", 1, "industrial_none_provenance"),
    ("baseline", "gpu_calls", 1, "industrial_none_provenance"),
    ("baseline", "laya_cells_completed", 378, "industrial_none_provenance"),
    ("mps", "dataset_id", "wrong", "industrial_mps_manifest"),
    ("S", "status", "pending", "industrial_stage_closed"),
    ("J", "status", "pending", "industrial_stage_closed"),
    ("S", "manifest_sha256", "wrong", "industrial_stage_closed"),
    ("J", "identity", "wrong", "industrial_stage_closed"),
])
def test_incomplete_or_mismatched_review_and_stage_metadata_reject(gate, bundle, target, key, value, reason):
    bundle.documents[target][key] = value
    with pytest.raises(ValueError, match="^" + reason + "$"):
        gate.industrial_inputs(bundle.root, bundle.manifest)


@pytest.mark.parametrize("name,reason", [
    ("handoff", "industrial_freshness_evidence"),
    ("none_closed", "industrial_freshness_evidence"),
    ("actual_review", "industrial_freshness_evidence"),
    ("mps", "industrial_stage_freshness"),
    ("S", "industrial_stage_freshness"),
    ("J", "industrial_stage_freshness"),
    ("inherited_S_review", "industrial_inherited_review_freshness"),
    ("inherited_J_review", "industrial_inherited_review_freshness"),
])
def test_each_gate_or_inherited_review_ref_must_remain_fresh(gate, bundle, name, reason):
    bundle.manifest["accepted_evidence"].remove(bundle.refs[name])
    with pytest.raises(ValueError, match="^" + reason + "$"):
        gate.industrial_inputs(bundle.root, bundle.manifest)


@pytest.mark.parametrize("channel,field,value", [
    *((channel, "hardware", "wrong") for channel in "BGENSJ"),
    ("B", "dtype", None),
    ("S", "stage", "wrong"),
])
def test_hardware_cannot_be_relabelled_even_when_handoff_agrees(gate, bundle, channel, field, value):
    bundle.manifest["retrieval_provenance"][channel][field] = value
    bundle.documents["handoff"]["retrieval_provenance"][channel][field] = value
    with pytest.raises(ValueError, match="^industrial_original_channel$"):
        gate.industrial_inputs(bundle.root, bundle.manifest)


@pytest.mark.parametrize("channel,name", [(channel, name) for channel in "SJ"
                                         for name in ("rankings.json", "report.json")])
def test_stage_copy_bytes_must_match_the_original(gate, bundle, channel, name):
    bundle.documents[channel]["files"][name]["bytes"] += 1
    with pytest.raises(ValueError, match="^industrial_stage_copy$"):
        gate.industrial_inputs(bundle.root, bundle.manifest)


@pytest.mark.parametrize("mutation,reason", [
    ("missing_S", "industrial_stage_scope"),
    ("missing_J", "industrial_stage_scope"),
    ("empty_inherited_reviews", "industrial_inherited_review_freshness"),
    ("nonpass_status", "industrial_actual_review"),
    ("wrong_handoff_path", "industrial_dataset_or_paths"),
    ("wrong_closed_report", "industrial_none_closed"),
    ("wrong_B_ranking_ref", "industrial_original_channel"),
])
def test_missing_closures_review_status_and_changed_refs_reject(gate, bundle, mutation, reason):
    if mutation.startswith("missing_"):
        bundle.documents["handoff"]["stage_closures"].pop(mutation[-1])
    elif mutation == "empty_inherited_reviews":
        bundle.documents["handoff"]["accepted_evidence"] = []
    elif mutation == "nonpass_status":
        bundle.manifest["industrial_evidence"]["actual_review_status"] = "SOURCE_ONLY"
        bundle.documents["actual_review"]["status"] = "SOURCE_ONLY"
    elif mutation == "wrong_handoff_path":
        bundle.manifest["industrial_evidence"]["handoff"]["path"] = "work/synthetic/wrong.json"
    elif mutation == "wrong_closed_report":
        bundle.documents["none_closed"]["files"]["report.json"]["sha256"] = digest(b"wrong")
    elif mutation == "wrong_B_ranking_ref":
        bundle.documents["none_closed"]["files"]["B-ranking.json"]["sha256"] = digest(b"wrong")
    else:
        raise AssertionError("Unknown synthetic mutation")
    with pytest.raises(ValueError, match="^" + reason + "$"):
        gate.industrial_inputs(bundle.root, bundle.manifest)

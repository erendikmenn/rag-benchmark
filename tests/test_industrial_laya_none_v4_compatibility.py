"""Only new one-descriptor / V4 lineage deltas; no runner or model imports."""
import ast
import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "src/rag_benchmark/industrial_laya_completion_v4.py"
BOUNDED = ROOT / "src/rag_benchmark/industrial_laya_bounded_v4.py"


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def isolated(path, names, scope):
    nodes = [n for n in ast.parse(path.read_text()).body
             if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names
             or isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<new V4 metadata-only seam>", "exec"), scope)
    return SimpleNamespace(**scope)


def ref(path, size=1):
    return {"path": path, "bytes": size, "sha256": hashlib.sha256(path.encode()).hexdigest()}


@pytest.fixture
def adapter_case():
    gate = isolated(PARENT, {"carried_original_adapter_metadata"}, {"require": require, "Path": Path})
    names = ("original_manifest", "operational_envelope", "root_acceptance", "independent_review",
             "s_actual_review", "s_actual_evidence", "s_full_nested_closure", "accepted_audit_source",
             "s_execution_claim", "s_actual_child_wait", "s_actual_parent_tool_wait")
    registry = {name: ref("work/synthetic/" + name + ".json") for name in names}
    controls = list(registry.values())
    descriptor = ref(".cache/models/S/blobs/exact-original-adapter", 813)
    registry.update({"asset_channel": "S", "asset_name": "adapter_config.json",
                     "asset_revision": "dcbe8d9cede518bce830488364ba0e40c873645b",
                     "descriptor": descriptor, "count": 1, "payload_reads": 0,
                     "live_payload_freshness_attested": False})
    upstream = {name: registry[name] for name in ("original_manifest", "operational_envelope", "s_execution_claim")}
    upstream.update(sj_root_output_acceptance=registry["root_acceptance"],
                    sj_actual_output_review=registry["independent_review"], s_actual_wait=registry["s_actual_child_wait"])
    handoff = {"original_adapter_metadata_registry": registry, "original_adapter_metadata_payloads_read": 0,
               "model_view_metadata_registry": {"channels": {"S": {"full_nested_closure": registry["s_full_nested_closure"]}}},
               "accepted_evidence": controls.copy()}
    operational = {"assets": {"S": {"revision": registry["asset_revision"],
                                    "assets": {"adapter_config.json": {"file": copy.deepcopy(descriptor)}}}}}
    return gate, handoff, upstream, controls, operational


def test_exact_single_adapter_descriptor_returns_only_controls_without_io(adapter_case):
    gate, handoff, upstream, controls, operational = adapter_case
    with patch("builtins.open", side_effect=AssertionError("payload I/O forbidden")), \
            patch.object(Path, "open", side_effect=AssertionError("payload I/O forbidden")):
        assert gate.carried_original_adapter_metadata(handoff, upstream, controls, operational) == controls
    assert handoff["original_adapter_metadata_registry"]["descriptor"] not in controls


@pytest.mark.parametrize("mutation", ["count_bool", "count_two", "reads_bool", "reads_one", "handoff_reads_bool",
                                     "live_claim", "changed_control", "changed_namespace_control", "missing_control",
                                     "path", "bytes", "sha", "channel", "asset", "revision", "raw_payload"])
def test_unaccepted_original_adapter_metadata_rejected_without_payload_io(adapter_case, mutation):
    gate, handoff, upstream, controls, operational = adapter_case
    r = handoff["original_adapter_metadata_registry"]
    if mutation == "count_bool":
        r["count"] = True
    elif mutation == "count_two":
        r["count"] = 2
    elif mutation == "reads_bool":
        r["payload_reads"] = False
    elif mutation == "reads_one":
        r["payload_reads"] = 1
    elif mutation == "handoff_reads_bool":
        handoff["original_adapter_metadata_payloads_read"] = False
    elif mutation == "live_claim":
        r["live_payload_freshness_attested"] = True
    elif mutation == "changed_control":
        r["original_manifest"] = ref("work/other-original.json")
    elif mutation == "changed_namespace_control":
        r["s_full_nested_closure"] = ref("work/other-namespace.json")
    elif mutation == "missing_control":
        controls.remove(r["accepted_audit_source"])
    elif mutation == "path":
        r["descriptor"]["path"] = ".cache/models/arbitrary/blobs/other"
    elif mutation == "bytes":
        r["descriptor"]["bytes"] += 1
    elif mutation == "sha":
        r["descriptor"]["sha256"] = "0" * 64
    elif mutation == "channel":
        r["asset_channel"] = "J"
    elif mutation == "asset":
        r["asset_name"] = "adapter_model.safetensors"
    elif mutation == "revision":
        r["asset_revision"] = "0" * 40
    elif mutation == "raw_payload":
        controls.append(r["descriptor"])
    with patch("builtins.open", side_effect=AssertionError("payload I/O forbidden")), \
            patch.object(Path, "open", side_effect=AssertionError("payload I/O forbidden")), pytest.raises(ValueError):
        gate.carried_original_adapter_metadata(handoff, upstream, controls, operational)


def defs(path):
    return {n.name: n for n in ast.parse(path.read_text()).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}


REPLACEMENTS = [("industrial_laya_completion_v4", "industrial_laya_completion_v3"),
                ("industrial_laya_bounded_v4", "industrial_laya_bounded_v3"),
                ("industrial-laya-completion-v4", "industrial-laya-completion-v3"),
                ("industrial-laya-sealed-union-v4", "industrial-laya-sealed-union-v3"),
                ("industrial-laya-bounded-v4", "industrial-laya-bounded-v3"),
                ("industrial-none-derivation-v4/actual-v4", "industrial-none-derivation-v3/actual-v3"),
                ("industrial-none-derivation-v4", "industrial-none-derivation-v3"),
                ("INDUSTRIAL_NONE_V4", "INDUSTRIAL_NONE_V3"),
                ("INDUSTRIAL_BOUNDED_V4", "INDUSTRIAL_BOUNDED_V3"), ("None V4", "None V3")]


class Normalize(ast.NodeTransformer):
    def visit_Constant(self, node):
        if isinstance(node.value, str):
            for new, old in REPLACEMENTS:
                node.value = node.value.replace(new, old)
        return node


def normalized(node):
    return ast.dump(Normalize().visit(copy.deepcopy(node)), include_attributes=False)


def test_parent_existing_43_registry_and_all_science_helpers_unchanged():
    old = defs(ROOT / "src/rag_benchmark/industrial_laya_completion_v3.py")
    new = defs(PARENT)
    assert set(new) - set(old) == {"carried_original_adapter_metadata"}
    for name in old.keys() - {"industrial_inputs"}:
        assert normalized(new[name]) == ast.dump(old[name], include_attributes=False), name
    n = copy.deepcopy(new["industrial_inputs"])
    n.body = [x for x in n.body if not (
        isinstance(x, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "adapter_controls" for t in x.targets)
        or isinstance(x, ast.For) and isinstance(x.iter, ast.Name) and x.iter.id == "adapter_controls")]
    assert normalized(n) == ast.dump(old["industrial_inputs"], include_attributes=False)


def test_bounded_all_science_memory_timing_lifecycle_helpers_unchanged():
    old = defs(ROOT / "src/rag_benchmark/industrial_laya_bounded_v3.py")
    new = defs(BOUNDED)
    assert set(new) == set(old)
    for name in old.keys() - {"history_check"}:
        assert normalized(new[name]) == ast.dump(old[name], include_attributes=False), name
    n = copy.deepcopy(new["history_check"])
    paths = next(x for x in n.body if isinstance(x, ast.Assign)).value.elts
    paths[:] = [x for x in paths if not (isinstance(x, ast.BinOp) and isinstance(x.left, ast.Constant)
               and x.left.value in {"work/industrial-laya-completion-v3/runs/", "work/industrial-laya-bounded-v3/runs/"})]
    assert normalized(n) == ast.dump(old["history_check"], include_attributes=False)


@pytest.mark.parametrize("kind", ["completion", "bounded"])
def test_old_v3_namespace_prevents_new_v4_run(kind):
    path = f"work/industrial-laya-{kind}-v3/runs/vidore-v3-industrial-en"
    original = SimpleNamespace(frozen=SimpleNamespace(path_at=lambda root, p: SimpleNamespace(exists=lambda: p == path)))
    gate = isolated(BOUNDED, {"history_check", "TARGET"}, {"require": require, "original": original})
    with pytest.raises(ValueError, match="unexpected_original_run"):
        gate.history_check(None, {"inheritance": None}, {"output": "work/new/run"})

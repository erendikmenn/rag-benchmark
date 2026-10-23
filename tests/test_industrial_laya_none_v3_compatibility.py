"""Only NoneV3 lineage/descriptor compatibility; no runner, model or data imports."""
import ast
import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "src/rag_benchmark/industrial_laya_completion_v3.py"
BOUNDED = ROOT / "src/rag_benchmark/industrial_laya_bounded_v3.py"


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def definitions(path):
    return {n.name: n for n in ast.parse(path.read_text()).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}


def isolated(path, names, scope):
    tree = ast.parse(path.read_text())
    selected = [n for n in tree.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                and n.targets[0].id in names or isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    exec(compile(ast.Module(body=selected, type_ignores=[]), "<isolated compatibility AST>", "exec"), scope)
    return SimpleNamespace(**scope)


def ref(path):
    return {"path": path, "bytes": 1, "sha256": hashlib.sha256(path.encode()).hexdigest()}


@pytest.fixture
def registry_case():
    gate = isolated(PARENT, {"carried_model_metadata", "INDUSTRIAL_MPS_BASE"}, {"Path": Path, "require": require})
    root, review, j, s = [ref("work/synthetic/" + n + ".json") for n in ("root", "review", "J", "S")]
    controls = [root, review, j, s]
    channels = {"J": {"pid": 1001, "count": 12, "full_nested_closure": j},
                "S": {"pid": 1002, "count": 31, "full_nested_closure": s}}
    descriptors = [ref(gate.INDUSTRIAL_MPS_BASE + f"/actual-v3/{c}/model-view-{v['pid']}/snapshot/file-{i}.json")
                   for c, v in channels.items() for i in range(v["count"])]
    registry = {"root_acceptance": root, "independent_review": review, "channels": channels,
                "descriptors": descriptors, "payload_reads": 0}
    handoff = {"model_view_metadata_registry": registry, "model_view_metadata_payloads_read": 0,
               "accepted_evidence": controls.copy()}
    upstream = {"sj_root_output_acceptance": root, "sj_actual_output_review": review}
    return gate, handoff, upstream, controls


def test_none_v3_registry_only_returns_raw_authorizing_controls(registry_case):
    gate, handoff, upstream, controls = registry_case
    assert gate.carried_model_metadata(handoff, upstream, controls) == controls
    assert len(handoff["model_view_metadata_registry"]["descriptors"]) == 43


def test_new_registry_cannot_substitute_boolean_for_unread_payload_count(registry_case):
    gate, handoff, upstream, controls = registry_case
    handoff["model_view_metadata_registry"]["payload_reads"] = False
    with pytest.raises(ValueError, match="industrial_model_metadata_registry"):
        gate.carried_model_metadata(handoff, upstream, controls)


@pytest.mark.parametrize("mutation", ["missing", "changed_root", "read_flag", "duplicate", "new_path", "pid",
                                     "count", "raw_payload", "missing_proof", "bad_hash"])
def test_new_registry_lineage_mismatch_fails_closed(registry_case, mutation):
    gate, handoff, upstream, controls = copy.deepcopy(registry_case)
    r = handoff["model_view_metadata_registry"]
    if mutation == "missing":
        del r["payload_reads"]
    elif mutation == "changed_root":
        r["root_acceptance"] = ref("work/unreviewed.json")
    elif mutation == "read_flag":
        r["payload_reads"] = 1
    elif mutation == "duplicate":
        r["descriptors"][1] = r["descriptors"][0]
    elif mutation == "new_path":
        r["descriptors"][0] = ref("work/unreviewed/model-view-1001/config.json")
    elif mutation == "pid":
        r["channels"]["S"]["pid"] += 1
    elif mutation == "count":
        r["channels"]["S"]["count"] -= 1
    elif mutation == "raw_payload":
        controls.append(r["descriptors"][0])
    elif mutation == "missing_proof":
        controls.pop()
    elif mutation == "bad_hash":
        r["descriptors"][0]["sha256"] = "g" * 64
    with pytest.raises((ValueError, KeyError)):
        gate.carried_model_metadata(handoff, upstream, controls)


@pytest.fixture
def bounded():
    original = SimpleNamespace(TARGETS={"vidore-v3-industrial-en": (5244, 283)})
    scope = {"original": original, "require": require}
    names = {"VERSION", "WORK", "TARGET", "POPULATION", "SEGMENT_QUERIES", "RECIPE_FIELDS",
             "recipe_scope", "history_check"}
    return isolated(BOUNDED, names, scope)


def bounded_recipe(bounded):
    return {"protocol": bounded.VERSION, "dataset_id": bounded.TARGET, "parent_manifest": {},
            "context_evidence": {}, "output": bounded.WORK + "/runs/" + bounded.TARGET,
            "source_sha256": {}, "segment_query_limit": 16, "inheritance": None, "accepted_evidence": []}


def test_new_bounded_protocol_accepted(bounded):
    bounded.recipe_scope(bounded_recipe(bounded))


@pytest.mark.parametrize("key,value", [("protocol", "industrial-laya-bounded-v2"),
                                       ("output", "work/industrial-laya-bounded-v2/runs/vidore-v3-industrial-en")])
def test_old_bounded_identity_rejected_by_new_source(bounded, key, value):
    r = bounded_recipe(bounded)
    r[key] = value
    with pytest.raises(ValueError):
        bounded.recipe_scope(r)


@pytest.mark.parametrize("kind", ["completion", "bounded"])
def test_new_run_rejects_prior_v2_namespace_without_reading_payloads(bounded, kind):
    forbidden = f"work/industrial-laya-{kind}-v2/runs/{bounded.TARGET}"
    bounded.original.frozen = SimpleNamespace(path_at=lambda root, path: SimpleNamespace(exists=lambda: path == forbidden))
    with pytest.raises(ValueError, match="unexpected_original_run"):
        bounded.history_check(None, {"inheritance": None}, {"output": "work/newparent/run"})


REPLACEMENTS = [("industrial_laya_completion_v3", "industrial_laya_completion_v2"),
                ("industrial_laya_bounded_v3", "industrial_laya_bounded_v2"),
                ("industrial-laya-completion-v3", "industrial-laya-completion-v2"),
                ("industrial-laya-sealed-union-v3", "industrial-laya-sealed-union-v2"),
                ("industrial-laya-bounded-v3", "industrial-laya-bounded-v2"),
                ("industrial-none-derivation-v3/actual-v3", "industrial-none-derivation-v2/actual-v2"),
                ("industrial-none-derivation-v3", "industrial-none-derivation-v2"),
                ("INDUSTRIAL_NONE_V3", "INDUSTRIAL_NONE_V2"),
                ("INDUSTRIAL_BOUNDED_V3", "INDUSTRIAL_BOUNDED_V2"), ("None V3", "None V2")]


class Normalize(ast.NodeTransformer):
    def visit_Constant(self, node):
        if isinstance(node.value, str):
            for new, old in REPLACEMENTS:
                node.value = node.value.replace(new, old)
        return node


def normalized(node):
    return ast.dump(Normalize().visit(copy.deepcopy(node)), include_attributes=False)


def test_parent_all_prior_functions_unchanged_except_explicit_metadata_handoff_gate():
    old = definitions(ROOT / "src/rag_benchmark/industrial_laya_completion_v2.py")
    new = definitions(PARENT)
    assert set(new) - set(old) == {"carried_model_metadata"}
    for name in old.keys() - {"industrial_inputs"}:
        assert normalized(new[name]) == ast.dump(old[name], include_attributes=False), name
    body = copy.deepcopy(new["industrial_inputs"])
    class RemoveDelta(ast.NodeTransformer):
        def visit_Assign(self, node):
            if any(isinstance(t, ast.Name) and t.id == "registry_controls" for t in node.targets):
                return None
            if any(isinstance(t, ast.Name) and t.id == "extra" for t in node.targets):
                node.value.elts = [e for e in node.value.elts if not (isinstance(e, ast.Starred)
                                  and isinstance(e.value, ast.Name) and e.value.id == "registry_controls")]
            return self.generic_visit(node)
        def visit_Expr(self, node):
            if isinstance(node.value, ast.Call) and node.value.args and isinstance(node.value.args[-1], ast.Constant):
                if node.value.args[-1].value == "industrial_model_metadata_claim_pid":
                    return None
            return self.generic_visit(node)
    assert normalized(RemoveDelta().visit(body)) == ast.dump(old["industrial_inputs"], include_attributes=False)


def test_bounded_all_science_memory_timing_and_lifecycle_functions_unchanged():
    old = definitions(ROOT / "src/rag_benchmark/industrial_laya_bounded_v2.py")
    new = definitions(BOUNDED)
    assert set(new) == set(old)
    for name in old.keys() - {"history_check"}:
        assert normalized(new[name]) == ast.dump(old[name], include_attributes=False), name
    tree = ast.parse(BOUNDED.read_text())
    parent_import = [n for n in tree.body if isinstance(n, ast.ImportFrom) and n.level]
    assert len(parent_import) == 1 and parent_import[0].names[0].name == "industrial_laya_completion_v3"


def test_cli_binds_new_parent_prepare_and_bounded_only():
    for kind in ("completion", "bounded"):
        tree = ast.parse((ROOT / f"scripts/run_industrial_laya_{kind}_v3.py").read_text())
        imports = [n.module for n in tree.body if isinstance(n, ast.ImportFrom)]
        assert imports == [f"rag_benchmark.industrial_laya_{kind}_v3"]
    constants = ast.parse(PARENT.read_text())
    assert "execute" not in [n.value for n in ast.walk(constants) if isinstance(n, ast.Constant) and n.value == "execute"]

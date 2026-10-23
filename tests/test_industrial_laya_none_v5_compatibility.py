"""New V5 identity/loaded-verifier regression only; no actual data or model calls."""
import ast
import copy
import hashlib
import importlib
import importlib.abc
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def normalized(text):
    pairs = (("industrial_laya_completion_v5", "industrial_laya_completion_v4"),
             ("industrial_laya_bounded_v5", "industrial_laya_bounded_v4"),
             ("industrial-laya-completion-v5", "industrial-laya-completion-v4"),
             ("industrial-laya-bounded-v5", "industrial-laya-bounded-v4"),
             ("industrial-laya-sealed-union-v5", "industrial-laya-sealed-union-v4"),
             ("industrial-none-derivation-v5", "industrial-none-derivation-v4"),
             ("actual-v5", "actual-v4"), ("INDUSTRIAL_NONE_V5", "INDUSTRIAL_NONE_V4"),
             ("INDUSTRIAL_BOUNDED_V5", "INDUSTRIAL_BOUNDED_V4"),
             ("INDUSTRIAL_LAYA_V5", "INDUSTRIAL_LAYA_V4"),
             ("None V5", "None V4"), ("parent V5", "parent V4"))
    for old, new in pairs:
        text = text.replace(old, new)
    return text


def functions(path):
    return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(path.read_text()).body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef))}


def test_parent_science_registry_timing_and_verifier_AST_unchanged():
    previous = functions(ROOT / "src/rag_benchmark/industrial_laya_completion_v4.py")
    current = ast.parse(normalized((ROOT / "src/rag_benchmark/industrial_laya_completion_v5.py").read_text()))
    assert previous == {n.name: ast.dump(n, include_attributes=False) for n in current.body
                        if isinstance(n, (ast.FunctionDef, ast.ClassDef))}


def test_bounded_AST_unchanged_except_explicit_prior_v4_absence():
    previous = functions(ROOT / "src/rag_benchmark/industrial_laya_bounded_v4.py")
    text = (ROOT / "src/rag_benchmark/industrial_laya_bounded_v5.py").read_text()
    addition = ',\n             "work/industrial-laya-completion-v4/runs/" + TARGET,\n             "work/industrial-laya-bounded-v4/runs/" + TARGET'
    assert text.count(addition) == 1
    tree = ast.parse(normalized(text.replace(addition, "")))
    assert previous == {n.name: ast.dump(n, include_attributes=False) for n in tree.body
                        if isinstance(n, (ast.FunctionDef, ast.ClassDef))}


def load_bounded():
    return importlib.import_module("rag_benchmark.industrial_laya_bounded_v5")


def test_old_v4_recipe_identity_rejected_before_data_io():
    module = load_bounded()
    recipe = {k: None for k in module.RECIPE_FIELDS}
    recipe.update(protocol="industrial-laya-bounded-v4", dataset_id=module.TARGET,
                  output=module.WORK + "/runs/" + module.TARGET,
                  segment_query_limit=16, source_sha256={}, accepted_evidence=[])
    with pytest.raises(ValueError, match="industrial_bounded_recipe_fields"):
        module.recipe_scope(recipe)
    accepted = copy.deepcopy(recipe)
    accepted["protocol"] = module.VERSION
    module.recipe_scope(accepted)


@pytest.mark.parametrize("namespace", ["industrial-laya-completion-v4", "industrial-laya-bounded-v4"])
def test_prior_v4_namespace_rejected_without_reading_contents(tmp_path, namespace):
    module = load_bounded()
    target = tmp_path / "work" / namespace / "runs" / module.TARGET
    target.mkdir(parents=True)
    with pytest.raises(ValueError, match="unexpected_original_run"):
        module.history_check(tmp_path, {"inheritance": None}, {"output": "work/new-unused/runs/" + module.TARGET})


def test_real_loaded_module_source_verifiers_not_only_isolated_AST():
    forbidden = {"torch", "transformers", "sentence_transformers", "huggingface_hub"}
    assert not forbidden.intersection(sys.modules)
    class NoModelImports(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in forbidden:
                raise AssertionError("Model imports forbidden in source-only verification")
    guard = NoModelImports()
    sys.meta_path.insert(0, guard)
    try:
        parent = importlib.import_module("rag_benchmark.industrial_laya_completion_v5")
        bounded = load_bounded()
        origin = importlib.import_module("rag_benchmark.industrial_mps_sj_v1")
        parent_pins = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in parent.SOURCE_FILES}
        bounded_pins = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in bounded.SOURCES}
        parent.sources(ROOT, parent_pins)
        bounded.check_source(ROOT, {"source_sha256": bounded_pins})
        origin.verify_module(bounded, (ROOT / "src/rag_benchmark/industrial_laya_bounded_v5.py").resolve())
        assert parent.INDUSTRIAL_NONE_BASE == "work/industrial-none-derivation-v5/actual-v5"
        assert bounded.original is parent
        assert not forbidden.intersection(sys.modules)
    finally:
        sys.meta_path.remove(guard)

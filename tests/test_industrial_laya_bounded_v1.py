"""New Industrial target/delegation deltas only; no model or actual data imports."""
import ast
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/rag_benchmark/industrial_laya_bounded_v1.py"
RECOVERY = ROOT / "src/rag_benchmark/vidore_laya_recovery_v2.py"
INDUSTRIAL = ROOT / "src/rag_benchmark/industrial_laya_completion_v1.py"


@pytest.fixture
def runner():
    original = SimpleNamespace(TARGETS={"vidore-v3-industrial-en": (5244, 283)})
    def require(condition, reason):
        if not condition:
            raise ValueError(reason)
    original.require = require
    original.frozen = SimpleNamespace(path_at=lambda root, relative: root / relative)
    module = ModuleType("synthetic_industrial_bounded")
    module.original = original
    tree = ast.parse(SOURCE.read_text())
    imports = [n for n in tree.body if isinstance(n, ast.ImportFrom) and n.level]
    assert len(imports) == 1 and imports[0].names[0].name == "industrial_laya_completion_v1"
    tree.body = [n for n in tree.body if n not in imports]
    exec(compile(tree, str(SOURCE), "exec"), module.__dict__)
    return module


def recipe(runner):
    return {"protocol": runner.VERSION, "dataset_id": runner.TARGET,
            "parent_manifest": {"name": "manifest", "sha256": "a" * 64},
            "context_evidence": {"name": "context", "sha256": "b" * 64},
            "output": runner.WORK + "/runs/" + runner.TARGET, "source_sha256": {},
            "segment_query_limit": 16, "inheritance": None,
            "accepted_evidence": [{"name": "evidence"}]}


def test_exact_fresh_population_and_process_partition(runner):
    runner.recipe_scope(recipe(runner))
    assert runner.POPULATION == (5244, 283)
    groups = runner.partition_indices(283, [])
    assert [len(group) for group in groups] == [16] * 17 + [11]
    assert [i for group in groups for i in group] == list(range(283))
    assert 283 * 378 == 106974


@pytest.mark.parametrize("name", ["vidore-v3-energy-fr", "vidore-v3-finance_en-en",
    "vidore-v3-finance_fr-fr", "vidore-v3-hr-en", "vidore-v3-pharmaceuticals-en", "vidore-v3-physics-fr"])
def test_six_recovery_targets_cannot_enter_industrial(runner, name):
    value = recipe(runner)
    value["dataset_id"] = name
    with pytest.raises(ValueError, match="^industrial_bounded_target_or_limit$"):
        runner.recipe_scope(value)


@pytest.mark.parametrize("inheritance", [{}, [], False, {"closed_indices": [0]},
                                        {"closed_indices": list(range(229)), "unknown_indices": [229]}])
def test_inherited_and_unknown_attempt_exceptions_are_forbidden(runner, inheritance, tmp_path):
    value = recipe(runner)
    value["inheritance"] = inheritance
    for call in (lambda: runner.recipe_scope(value),
                 lambda: runner.inherited_values(tmp_path, value, None, None),
                 lambda: runner.unknown_claim(tmp_path, value, 0)):
        with pytest.raises(ValueError, match="^industrial_all_queries_must_be_fresh$"):
            call()


def test_existing_legacy_scoring_output_blocks_fresh_run(runner, tmp_path):
    value = recipe(runner)
    manifest = {"output": "legacy-scoring"}
    assert runner.history_observations(tmp_path, value, manifest) == {}
    assert runner.inherited_values(tmp_path, value, None, None) == {}
    assert runner.unknown_claim(tmp_path, value, 282) is None
    (tmp_path / manifest["output"]).mkdir()
    with pytest.raises(ValueError, match="^unexpected_original_run$"):
        runner.history_observations(tmp_path, value, manifest)


@pytest.mark.parametrize("input_failure", [False, True])
def test_load_delegates_to_industrial_input_and_existing_context_guards(runner, tmp_path, input_failure):
    value = recipe(runner)
    manifest = {"dataset_id": runner.TARGET, "output": "legacy-scoring"}
    admission = {"protocol": runner.VERSION, "status": "PASS_LOCAL_MPS_INDUSTRIAL_BOUNDED_AUTHORIZED",
        "recipe_sha256": "c" * 64, "source_sha256": {}, "parent_manifest_sha256": "a" * 64,
        "context_sha256": "b" * 64, "baseline_metric_replay": False, "tokenizer_repeated": False,
        "model_scoring_authorized": True, "maximum_queries_per_process": 16}
    objects = {"recipe": value, "manifest": manifest, "admission": admission, "evidence": {}}
    runner.read_pinned = lambda _, ref: objects[ref["name"]]
    runner.check_source = lambda *_: None
    calls = []
    dataset, rankings = object(), object()
    def validate(root, actual_manifest, *, verify_baseline):
        assert root == tmp_path and actual_manifest is manifest and verify_baseline is False
        calls.append("industrial_validate")
        if input_failure:
            raise ValueError("industrial_actual_review")
        return dataset, rankings
    def context(root, actual_manifest, sha, ref, actual_dataset, actual_rankings):
        assert (root, actual_manifest, sha, ref, actual_dataset, actual_rankings) == (
            tmp_path, manifest, "a" * 64, value["context_evidence"], dataset, rankings)
        calls.append("existing_context_proof")
    runner.original.validate = validate
    runner.original.context_evidence = context
    args = (tmp_path, {"name": "recipe", "sha256": "c" * 64}, {"name": "admission"})
    if input_failure:
        with pytest.raises(ValueError, match="^industrial_actual_review$"):
            runner.load(*args)
        assert calls == ["industrial_validate"]
    else:
        assert runner.load(*args) == (value, manifest, dataset, rankings)
        assert calls == ["industrial_validate", "existing_context_proof"]


def test_only_declared_target_and_inheritance_deltas_change_reviewed_functions():
    old_text, new_text = RECOVERY.read_text(), SOURCE.read_text()
    old = {n.name: n for n in ast.parse(old_text).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    new = {n.name: n for n in ast.parse(new_text).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    assert set(old) == set(new)
    changed = {"recipe_scope", "history_check", "history_observations", "inherited_values", "unknown_claim",
               "load", "execute_segment", "finalize"}
    for name in set(old) - changed:
        assert ast.dump(old[name]) == ast.dump(new[name]), name
    replacements = {
        '"PASS_LOCAL_MPS_RECOVERY_AUTHORIZED"': '"PASS_LOCAL_MPS_INDUSTRIAL_BOUNDED_AUTHORIZED"',
        'inherited = recipe["inheritance"]["closed_indices"] if recipe["inheritance"] else []': 'inherited = []',
        '    if recipe["inheritance"] is not None:\n        refs.append(recipe["inheritance"]["independent_reuse_admission"])\n': '',
        'unknown = recipe["inheritance"]["unknown_indices"] if recipe["inheritance"] else []': 'unknown = []',
        '        if recipe["inheritance"] is not None:\n            final_refs.append(recipe["inheritance"]["independent_reuse_admission"])\n': '',
    }
    for name in ("load", "execute_segment", "finalize"):
        expected = ast.get_source_segment(old_text, old[name])
        for before, after in replacements.items():
            expected = expected.replace(before, after)
        assert ast.dump(ast.parse(expected).body[0]) == ast.dump(new[name]), name


def test_frozen_industrial_parent_retains_exact_condition_and_input_gate():
    industrial_tree = ast.parse(INDUSTRIAL.read_text())
    recovery_parent = ast.parse((ROOT / "src/rag_benchmark/vidore_laya_completion_v4.py").read_text())
    def constants(tree):
        return {n.targets[0].id: ast.dump(n.value) for n in tree.body
                if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                and n.targets[0].id in {"MEMORY", "LAYA_CONFIG", "CONTEXT"}}
    assert constants(industrial_tree) == constants(recovery_parent)
    validate = next(n for n in industrial_tree.body if isinstance(n, ast.FunctionDef) and n.name == "validate")
    calls = [n.func.id for n in ast.walk(validate) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    assert "industrial_inputs" in calls

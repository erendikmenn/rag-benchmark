"""New loaded-source verifier compatibility and separate V5 identity only."""
import ast
from pathlib import Path

import pytest

from rag_benchmark import industrial_none_derivation_v4 as old
from rag_benchmark import industrial_none_derivation_v5 as run


def test_real_frozen_loaded_module_seam_red_green():
    # Same frozen verifier; no prepare/input/scoring function is invoked.
    with pytest.raises(NameError, match="frozenset"):
        old.origin.verify_module(old, Path(old.__file__).resolve())
    run.origin.verify_module(run, Path(run.__file__).resolve())


def test_same_membership_literal_tuple_and_only_membership_consumer():
    assert type(run.MODEL_VIEW_NAMES) is tuple
    assert len(run.MODEL_VIEW_NAMES) == len(set(run.MODEL_VIEW_NAMES)) == 23
    assert set(run.MODEL_VIEW_NAMES) == old.MODEL_VIEW_NAMES
    for value in (*run.MODEL_VIEW_NAMES, "weights.safetensors", "../config.json", "other.json", ""):
        assert (value in run.MODEL_VIEW_NAMES) == (value in old.MODEL_VIEW_NAMES)
    tree = ast.parse(Path(run.__file__).read_bytes())
    assignment = next(n for n in tree.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "MODEL_VIEW_NAMES" for t in n.targets))
    assert isinstance(assignment.value, ast.Tuple)
    assert all(isinstance(v, ast.Constant) and type(v.value) is str for v in assignment.value.elts)
    uses = [n for n in ast.walk(tree) if isinstance(n, ast.Name)
            and n.id == "MODEL_VIEW_NAMES" and isinstance(n.ctx, ast.Load)]
    assert len(uses) == 1
    compare = next(n for n in ast.walk(tree) if isinstance(n, ast.Compare) and uses[0] in n.comparators)
    assert any(isinstance(op, ast.In) for op in compare.ops)


def test_real_frozen_verifier_rejects_changed_loaded_tuple(monkeypatch):
    monkeypatch.setattr(run, "MODEL_VIEW_NAMES", (*run.MODEL_VIEW_NAMES, "other.json"))
    with pytest.raises(ValueError, match="Loaded constant differs"):
        run.origin.verify_module(run, Path(run.__file__).resolve())


def test_real_frozen_verifier_rejects_same_members_as_wrong_type(monkeypatch):
    monkeypatch.setattr(run, "MODEL_VIEW_NAMES", frozenset(run.MODEL_VIEW_NAMES))
    with pytest.raises(ValueError, match="Loaded constant differs"):
        run.origin.verify_module(run, Path(run.__file__).resolve())


def test_new_identity_preserves_upstream_and_prior_source_byte_closure(tmp_path):
    assert run.VERSION == "industrial-none-derivation-v5"
    assert run.BASE == "work/industrial-none-derivation-v5"
    assert run.OUTPUT == run.BASE + "/actual-v5"
    assert run.MPS_MANIFEST == old.MPS_MANIFEST
    assert run.STAGE_CLOSED == old.STAGE_CLOSED
    assert run.UPSTREAM_KEYS == old.UPSTREAM_KEYS and len(run.UPSTREAM_KEYS) == 18
    assert run.SOURCES == (*old.SOURCES,
                           "src/rag_benchmark/industrial_none_derivation_v5.py",
                           "scripts/run_industrial_none_derivation_v5.py")
    request = run.recipe(tmp_path, {name: "a" * 64 for name in run.SOURCES})
    assert request["version"] == run.VERSION and request["output"] == run.OUTPUT
    assert request["expected_none_cells"] == 126 and request["expected_original_standalone_cells"] == 8
    cli = Path(__file__).resolve().parents[1] / "scripts/run_industrial_none_derivation_v5.py"
    assert "from rag_benchmark.industrial_none_derivation_v5 import main" in cli.read_text()


@pytest.mark.parametrize("path", [
    "work/industrial-none-derivation-v3/actual-v3",
    "work/industrial-none-derivation-v3/prepared-manifest-v3.json",
    "work/industrial-none-derivation-v4/actual-v4",
    "work/industrial-none-derivation-v4/prepared-manifest-v4.json",
])
def test_prior_real_artifact_is_not_silently_aliased(tmp_path, path):
    target = tmp_path / path
    target.parent.mkdir(parents=True)
    target.write_text("historical artifact")
    with pytest.raises(ValueError):
        run.prior_cpu_namespace_absent(tmp_path)


def test_failed_v4_stage_preserved_without_being_treated_as_success(tmp_path):
    stage = tmp_path / "work/industrial-none-derivation-v4/root-prepare-001"
    stage.mkdir(parents=True)
    failure = stage / "wait.json"
    failure.write_text('{"actual_wait":true,"exit_code":1,"signal":null}')
    before = failure.read_bytes()
    run.prior_cpu_namespace_absent(tmp_path)
    assert failure.read_bytes() == before
    assert not (stage.parent / "prepared-manifest-v4.json").exists()
    assert not (stage.parent / "actual-v4").exists()


def test_broken_prior_artifact_symlink_is_rejected(tmp_path):
    target = tmp_path / "work/industrial-none-derivation-v4/actual-v4"
    target.parent.mkdir(parents=True)
    target.symlink_to(tmp_path / "absent")
    with pytest.raises(ValueError):
        run.prior_cpu_namespace_absent(tmp_path)


def test_all_non_identity_functions_classes_and_science_aliases_unchanged_ast():
    old_tree = ast.parse(Path(old.__file__).read_bytes())
    new_tree = ast.parse(Path(run.__file__).read_bytes())
    old_defs = {n.name: n for n in old_tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    new_defs = {n.name: n for n in new_tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    assert old_defs.keys() == new_defs.keys()
    identity_only = {"sources", "source_admissions", "main", "prior_cpu_namespace_absent"}
    for name in old_defs.keys() - identity_only:
        assert ast.dump(old_defs[name], include_attributes=False) == ast.dump(new_defs[name], include_attributes=False), name
    for name in ("configuration", "derive_rows", "strict_ranking", "metrics_equal", "prior_agreement", "archive_bundle"):
        assert getattr(run, name) is getattr(old, name) is getattr(run.previous, name)

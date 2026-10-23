"""Only Industrial parentV2/proof-loader namespace deltas; no actual runs."""

import ast
import hashlib
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/rag_benchmark/industrial_laya_bounded_v2.py"
OLD = ROOT / "src/rag_benchmark/industrial_laya_bounded_v1.py"


@pytest.fixture
def runner():
    original = SimpleNamespace(TARGETS={"vidore-v3-industrial-en": (5244, 283)})

    def require(condition, reason):
        if not condition:
            raise ValueError(reason)

    original.require = require
    original.frozen = SimpleNamespace(path_at=lambda root, path: root / path)
    module = ModuleType("synthetic_bounded_v2")
    module.original = original
    tree = ast.parse(SOURCE.read_text())
    imports = [n for n in tree.body if isinstance(n, ast.ImportFrom) and n.level]
    assert len(imports) == 1 and imports[0].names[0].name == "industrial_laya_completion_v2"
    tree.body = [n for n in tree.body if n not in imports]
    exec(compile(tree, str(SOURCE), "exec"), module.__dict__)
    return module


def bundle(runner, root):
    recipe = {
        "protocol": runner.VERSION,
        "dataset_id": runner.TARGET,
        "parent_manifest": {"key": "manifest", "sha256": "a" * 64},
        "context_evidence": {"key": "context", "sha256": "b" * 64},
        "output": runner.WORK + "/runs/" + runner.TARGET,
        "source_sha256": {},
        "segment_query_limit": 16,
        "inheritance": None,
        "accepted_evidence": [{"key": "raw"}, {"key": "empty"}],
    }
    manifest = {
        "protocol": "industrial-laya-sealed-union-v2",
        "dataset_id": runner.TARGET,
        "output": "work/industrial-laya-completion-v2/runs/" + runner.TARGET,
    }
    admission = {
        "protocol": runner.VERSION,
        "status": "PASS_LOCAL_MPS_INDUSTRIAL_BOUNDED_V2_AUTHORIZED",
        "recipe_sha256": "c" * 64,
        "source_sha256": {},
        "parent_manifest_sha256": "a" * 64,
        "context_sha256": "b" * 64,
        "baseline_metric_replay": False,
        "tokenizer_repeated": False,
        "model_scoring_authorized": True,
        "maximum_queries_per_process": 16,
    }
    semantic = {"recipe": recipe, "manifest": manifest, "admission": admission}
    raw = {"raw": b"not-json-source-bytes", "empty": b""}
    calls = []

    def read_json(_root, ref):
        calls.append("json:" + ref["key"])
        return semantic[ref["key"]]

    def checked(_root, ref):
        calls.append("raw:" + ref["key"])
        return raw[ref["key"]]

    runner.original.frozen.read_json = read_json
    runner.original.frozen.checked = checked
    runner.check_source = lambda *_: calls.append("new_source_check")
    dataset, rankings = object(), object()

    def validate(actual_root, value, *, verify_baseline):
        assert actual_root == root and value is manifest and verify_baseline is False
        runner.require(value["protocol"] == "industrial-laya-sealed-union-v2", "parent_scope")
        calls.append("new_parent_validation")
        return dataset, rankings

    def context(actual_root, value, digest, ref, data, ranks):
        assert (actual_root, value, digest, ref, data, ranks) == (
            root,
            manifest,
            "a" * 64,
            recipe["context_evidence"],
            dataset,
            rankings,
        )
        calls.append("new_parent_context")

    runner.original.validate = validate
    runner.original.context_evidence = context
    return SimpleNamespace(
        recipe=recipe,
        manifest=manifest,
        admission=admission,
        calls=calls,
        args=(root, {"key": "recipe", "sha256": "c" * 64}, {"key": "admission"}),
        dataset=dataset,
        rankings=rankings,
    )


def test_v2_load_accepts_raw_and_empty_proofs_without_json_and_uses_new_parent(runner, tmp_path):
    value = bundle(runner, tmp_path)
    assert runner.load(*value.args) == (value.recipe, value.manifest, value.dataset, value.rankings)
    assert "raw:raw" in value.calls and "raw:empty" in value.calls
    assert "json:raw" not in value.calls and "json:empty" not in value.calls
    assert value.calls[-2:] == ["new_parent_validation", "new_parent_context"]


def test_raw_proof_failure_stops_before_parent_context(runner, tmp_path):
    value = bundle(runner, tmp_path)

    def changed(*_):
        raise ValueError("changed_raw_proof")

    runner.original.frozen.checked = changed
    with pytest.raises(ValueError, match="changed_raw_proof"):
        runner.load(*value.args)
    assert "new_parent_validation" not in value.calls and "new_parent_context" not in value.calls


@pytest.mark.parametrize("change", ["old_admission", "old_parent", "old_recipe", "old_output"])
def test_old_generation_identity_cannot_enter_new_run(runner, tmp_path, change):
    value = bundle(runner, tmp_path)
    if change == "old_admission":
        value.admission["status"] = "PASS_LOCAL_MPS_INDUSTRIAL_BOUNDED_AUTHORIZED"
    elif change == "old_parent":
        value.manifest["protocol"] = "industrial-laya-sealed-union-v1"
    elif change == "old_recipe":
        value.recipe["protocol"] = "industrial-laya-bounded-v1"
    elif change == "old_output":
        value.recipe["output"] = "work/industrial-laya-bounded-v1/runs/" + runner.TARGET
    with pytest.raises(ValueError):
        runner.load(*value.args)


@pytest.mark.parametrize("old_run", ["industrial-laya-completion-v1", "industrial-laya-bounded-v1"])
def test_any_prior_industrial_scoring_namespace_blocks_fresh_bounded_start(runner, tmp_path, old_run):
    value = bundle(runner, tmp_path)
    prior = tmp_path / "work" / old_run / "runs" / runner.TARGET
    prior.mkdir(parents=True)
    with pytest.raises(ValueError, match="unexpected_original_run"):
        runner.load(*value.args)


def test_only_namespace_source_parent_raw_proof_and_history_deltas_change_code():
    texts = [p.read_text() for p in (OLD, SOURCE)]
    functions = [
        {n.name: n for n in ast.parse(text).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        for text in texts
    ]
    assert set(functions[0]) == set(functions[1])
    for name in functions[0].keys() - {"load", "history_check"}:
        assert ast.dump(functions[0][name]) == ast.dump(functions[1][name]), name
    expected = ast.get_source_segment(texts[0], functions[0]["load"])
    expected = expected.replace(
        "PASS_LOCAL_MPS_INDUSTRIAL_BOUNDED_AUTHORIZED", "PASS_LOCAL_MPS_INDUSTRIAL_BOUNDED_V2_AUTHORIZED"
    )
    expected = expected.replace("read_pinned(root, evidence)", "original.frozen.checked(root, evidence)")
    assert ast.dump(ast.parse(expected).body[0]) == ast.dump(functions[1]["load"])
    assert (
        hashlib.sha256(OLD.read_bytes()).hexdigest()
        == "53633c5440426769237cfe20b70750dd138b0c738e347417285a7c79024830be"
    )


def test_new_scope_still_has_exact_283_fresh_queries_18_score_one_finalizer(runner, tmp_path):
    value = bundle(runner, tmp_path)
    runner.recipe_scope(value.recipe)
    parts = runner.partition_indices(283, [])
    assert len(parts) == 18 and [len(p) for p in parts] == [16] * 17 + [11]
    assert runner.inherited_values(tmp_path, value.recipe, None, None) == {}
    assert runner.unknown_claim(tmp_path, value.recipe, 0) is None
    assert runner.SOURCES == (
        "src/rag_benchmark/industrial_laya_bounded_v2.py",
        "scripts/run_industrial_laya_bounded_v2.py",
    )

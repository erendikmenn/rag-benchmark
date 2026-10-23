"""Only the three post-freeze recovery gate regressions; no models or old suites."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("recovery_fixture_v1", ROOT / "tests/test_vidore_laya_recovery_v1.py")
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)
fixture.SOURCE = ROOT / "src/rag_benchmark/vidore_laya_recovery_v2.py"
recovery = fixture.recovery


@pytest.mark.parametrize("damage", [None, "pid", "recipe_sha256", "segment_index", "protocol"])
def test_wait_binds_exact_preceding_process_and_condition(recovery, tmp_path, damage):
    root = tmp_path.resolve()
    o = recovery.original
    recipe, recipe_sha = {"output": "result"}, "a" * 64
    output = root / "result"
    for path in (output, output / "segments", output / "segments/segment-0000",
                 output / "segments/segment-0000/jobs"):
        o.mkdir_private(path)
    stage = output / "segments/segment-0000"
    o.fresh(output / "recipe.json", recipe)
    o.fresh(stage / "claim.json", {"protocol": recovery.VERSION, "recipe_sha256": recipe_sha,
        "segment_index": 0, "indices": [], "score_segment": recovery.segment_token(recipe_sha, 0), "pid": 123})
    o.frozen.exclusive_write(stage / "private-library.log", b"")
    o.fresh(stage / "closed.json", {"status": "closed", "files": o.private_tree(stage)})
    wait = {"protocol": recovery.VERSION, "recipe_sha256": recipe_sha, "segment_index": 0,
            "actual_wait": True, "pid": 123, "exit_code": 0, "signal": None}
    if damage:
        wait[damage] = {"pid": 124, "recipe_sha256": "b" * 64, "segment_index": 1,
                        "protocol": "wrong"}[damage]
    o.fresh(root / "wait.json", wait)
    o.fresh(root / "prior.json", {"protocol": recovery.VERSION, "recipe_sha256": recipe_sha,
        "next_segment_index": 1, "tree": o.private_tree(output),
        "actual_child_wait": o.frozen.record(root, "wait.json")})
    ref = o.frozen.record(root, "prior.json")
    if damage:
        with pytest.raises(ValueError, match="^predecessor_actual_wait$"):
            recovery.stage_root(root, recipe, {"sha256": recipe_sha}, [[]], 1, predecessor_ref=ref)
    else:
        assert recovery.stage_root(root, recipe, {"sha256": recipe_sha}, [[]], 1, predecessor_ref=ref) == output


def minimal_environment(recovery, root):
    o = recovery.original
    def ref(name, value):
        o.fresh(root / (name + ".json"), value)
        return o.frozen.record(root, name + ".json")
    recipe_ref, admission_ref = ref("recipe", {}), ref("admission", {})
    wait_ref = ref("wait", {})
    prior_ref = ref("prior", {"actual_child_wait": wait_ref})
    proof_ref = ref("reuse-proof", {})
    recipe = {"output": "output", "inheritance": {"closed_indices": [], "unknown_indices": [],
              "independent_reuse_admission": proof_ref}, "accepted_evidence": [],
              "parent_manifest": ref("manifest", {}), "context_evidence": ref("context", {}),
              "dataset_id": "synthetic"}
    o.frozen.exclusive_write(root / "gpu.lock", b"")
    lock = (root / "gpu.lock").stat()
    manifest = {"gpu_lock": {"path": "gpu.lock", "device": lock.st_dev, "inode": lock.st_ino},
                "source_sha256": {}, "sdk_source_sha256": {}, "retrieval_provenance": {},
                "baseline_report": {}}
    dataset = SimpleNamespace(corpus=[{"id": "d"}], queries=[{"id": "q"}], qrels={"q": {"d": 1}},
                              model_item=lambda row, **_: row)
    o.mkdir_private(root / "output")
    o.mkdir_private(root / "output/segments")
    o.fresh(root / "output/recipe.json", recipe)
    recovery.load = lambda *_: (recipe, manifest, dataset, {})
    recovery.stage_root = lambda *_args, **_kwargs: root / "output"
    recovery.check_source = recovery.history_check = lambda *_: None
    recovery.history_observations = lambda *_: {}
    o.sources = o.sdk_sources = lambda *_: None
    o.observations = lambda *_: {}
    o.graph = lambda *_: []
    o.union_ids = lambda *_: ["d"]
    o.query_binding = lambda *_: {"binding": True}
    o.scatter = lambda *_: []
    o.validate = lambda *_args, **_kwargs: (dataset, {})
    return o, recipe, manifest, recipe_ref, admission_ref, prior_ref, wait_ref, proof_ref


@pytest.mark.parametrize("changed", ["reuse", "wait"])
def test_transitive_refs_are_checked_after_scoring(recovery, tmp_path, monkeypatch, changed):
    root = tmp_path.resolve()
    o, _, _, recipe_ref, admission_ref, prior_ref, wait_ref, proof_ref = minimal_environment(recovery, root)
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    target = root / (proof_ref if changed == "reuse" else wait_ref)["path"]
    class FakeBackend:
        def __init__(self, _):
            pass
        def score(self, query, documents, publish):
            publish("before", {})
            publish("after", {})
            target.write_bytes(b'{"changed":true}\n')
            return [0.5], 0.1
        def close(self):
            pass
    recovery.Backend = FakeBackend
    with pytest.raises(ValueError, match="Sealed input bytes changed"):
        recovery.execute_segment(root, recipe_ref, admission_ref, 0, predecessor_ref=prior_ref)
    job = root / "output/segments/segment-0000/jobs/q-000000"
    assert (job / "claim.json").exists() and not (job / "result.json").exists()


def test_finalizer_raw_validation_precedes_publication(recovery, tmp_path):
    root = tmp_path.resolve()
    o, _, _, recipe_ref, admission_ref, prior_ref, _, _ = minimal_environment(recovery, root)
    o.frozen.exclusive_write(root / "output/owner.lock", b"")
    recovery.inherited_values = lambda *_: {}
    recovery.closed_job = lambda *_: {"fresh_unique_pairs": 1, "union_scoring_seconds": 0.1}
    o.closed_value = lambda *_: [{"key": "x-k20", "candidate_k": 20, "candidate_count": 1,
                                 "metrics": {"hit@5": 1.0}}]
    o.pairs = lambda: [{"channels": ["B"], "budget_mode": "per_channel"}]
    o.matrix = SimpleNamespace(aggregate_metrics=lambda rows: rows[0])
    calls = []
    def validate(*_, verify_baseline):
        calls.append(verify_baseline)
        raise ValueError("synthetic_raw_input_changed")
    o.validate = validate
    with pytest.raises(ValueError, match="^synthetic_raw_input_changed$"):
        recovery.finalize(root, recipe_ref, admission_ref, predecessor_ref=prior_ref)
    assert calls == [False]
    assert not (root / "output/summary.json").exists()
    assert not (root / "output/closed.json").exists()


def test_unchanged_scientific_and_memory_functions(recovery):
    import ast
    first = ast.parse((ROOT / "src/rag_benchmark/vidore_laya_recovery_v1.py").read_text())
    second = ast.parse((ROOT / "src/rag_benchmark/vidore_laya_recovery_v2.py").read_text())
    names = {"Backend", "memory_observation", "memory_within_caps", "partition_indices", "closed_job",
             "unknown_claim", "inherited_values"}
    def selected(tree):
        return {n.name: ast.dump(n) for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                and n.name in names}
    assert selected(first) == selected(second)
    assert recovery.VERSION == "vidore-laya-bounded-recovery-v2"

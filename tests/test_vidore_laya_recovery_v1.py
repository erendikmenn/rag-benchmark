"""Synthetic recovery seams; no project imports, models, or actual output reads."""
import ast
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys
from types import ModuleType, SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/rag_benchmark/vidore_laya_recovery_v1.py"
GIB = 1024**3


def definitions(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == names
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)


@pytest.fixture
def recovery():
    """Retain actual control flow and private-file helpers, excluding model imports."""
    frozen = ModuleType("synthetic_frozen")
    frozen.__dict__.update(json=json, hashlib=hashlib, os=os, stat=stat, Path=Path)
    definitions(ROOT / "src/rag_benchmark/vidore_cache_recovery_v2.py", {
        "canonical", "path_at", "file_bytes", "record", "checked", "read_json",
        "_fsync_directory", "exclusive_write",
    }, frozen.__dict__)
    original = ModuleType("synthetic_original")
    original.__dict__.update(frozen=frozen, hashlib=hashlib, os=os, stat=stat, Path=Path,
                             fcntl=fcntl, contextmanager=contextmanager, math=math)
    definitions(ROOT / "src/rag_benchmark/vidore_laya_completion_v4.py", {
        "require", "sha", "canonical", "fresh", "mkdir_private", "lock", "private_tree", "closed_value",
    }, original.__dict__)
    original.MEMORY = {"maximum_process_rss_bytes": 64 * GIB, "maximum_mps_driver_bytes": 64 * GIB}
    original.utc = lambda: "2000-01-01T00:00:00+00:00"
    module = ModuleType("synthetic_recovery")
    module.original = original
    tree = ast.parse(SOURCE.read_text())
    relative = [n for n in tree.body if isinstance(n, ast.ImportFrom) and n.level]
    assert len(relative) == 1 and relative[0].names[0].name == "vidore_laya_completion_v4"
    nodes = [n for n in tree.body if n not in relative and not isinstance(n, ast.If)]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), module.__dict__)
    return module


@pytest.mark.parametrize("failure", [None, "driver", "peak"])
def test_actual_score_cleanup_caps_and_durable_observations(recovery, monkeypatch, tmp_path, failure):
    events = []
    state = {"peak": 10 * GIB, "driver": 2 * GIB, "scored": False}
    fake_torch = ModuleType("torch")
    def empty_cache():
        events.append("empty_cache")
        if failure != "driver":
            state["driver"] = 2 * GIB
    fake_torch.mps = SimpleNamespace(
        empty_cache=empty_cache, driver_allocated_memory=lambda: state["driver"],
        current_allocated_memory=lambda: GIB,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    recovery.resource = SimpleNamespace(RUSAGE_SELF=0,
        getrusage=lambda _: SimpleNamespace(ru_maxrss=state["peak"]))
    recovery.sys = SimpleNamespace(platform="darwin")
    recovery.current_rss_bytes = lambda: 3 * GIB
    recovery.gc = SimpleNamespace(collect=lambda: events.append("collect"))
    recovery.original.models = SimpleNamespace(synchronize=lambda device: events.append("sync:" + device))
    class Score:
        def __init__(self, value):
            self.value = value
        def __float__(self):
            events.append("float:" + str(self.value))
            return self.value
    class Adapter:
        def score(self, query, documents):
            assert query == {"id": "q"} and documents == [{"id": "a"}, {"id": "b"}, {"id": "c"}]
            events.append("score")
            state.update(scored=True, driver=70 * GIB)
            if failure == "peak":
                state["peak"] = 65 * GIB
            return [Score(0.0), Score(0.25), Score(1.0)]
        def unload(self):
            events.append("unload")
    backend = object.__new__(recovery.Backend)
    backend.adapter = Adapter()
    def publish(phase, observation):
        events.append("publish:" + phase)
        recovery.original.fresh(tmp_path / ("memory-" + phase + ".json"), observation)
    args = ({"id": "q"}, [{"id": "a"}, {"id": "b"}, {"id": "c"}], publish)
    if failure:
        with pytest.raises(ValueError, match="^memory_guard$"):
            backend.score(*args)
    else:
        scores, elapsed = backend.score(*args)
        assert scores == [0.0, 0.25, 1.0] and all(type(score) is float for score in scores)
        assert elapsed >= 0
    assert state["scored"]
    before, after = [json.loads((tmp_path / ("memory-" + phase + ".json")).read_bytes())
                     for phase in ("before", "after")]
    assert before["within_caps"] is True
    assert after["within_caps"] is (failure is None)
    assert after["caps"] == {"maximum_process_rss_bytes": 64 * GIB, "maximum_mps_driver_bytes": 64 * GIB}
    assert after["before_cache_release"]["mps_driver_allocated_bytes"] == 70 * GIB
    assert after["after_cache_release"] == {
        "peak_process_rss_bytes": (65 if failure == "peak" else 10) * GIB,
        "current_process_rss_bytes": 3 * GIB,
        "mps_driver_allocated_bytes": (70 if failure == "driver" else 2) * GIB,
        "mps_current_allocated_bytes": GIB,
    }
    assert events == ["sync:mps", "collect", "empty_cache", "sync:mps", "publish:before",
                      "sync:mps", "score", "float:0.0", "float:0.25", "float:1.0", "sync:mps",
                      "sync:mps", "collect", "empty_cache", "sync:mps", "publish:after"]
    backend.close()
    assert events[-1] == "unload"


def test_partition_excludes_exactly_229_inherited_queries(recovery):
    groups = recovery.partition_indices(308, list(range(229)))
    flattened = [index for group in groups for index in group]
    assert [len(group) for group in groups] == [16, 16, 16, 16, 15]
    assert flattened == list(range(229, 308))
    assert len(flattened) == 79 and flattened[0] == 229
    assert recovery.partition_indices(2, [0, 1]) == []


@pytest.mark.parametrize("count,inherited", [
    (0, []), (True, []), (3, [1, 0]), (3, [1, 1]), (3, [-1]), (3, [3]),
    (3, [True]), (3, [1.0]), (3, (0, 1)),
])
def test_partition_rejects_malformed_inheritance(recovery, count, inherited):
    with pytest.raises(ValueError, match="^inherited_indices$"):
        recovery.partition_indices(count, inherited)


def make_closed_job(recovery, path):
    original = recovery.original
    original.mkdir_private(path)
    binding = {"manifest_sha256": "a" * 64, "query_index": 229}
    recipe_sha = "b" * 64
    claim_sha = original.fresh(path / "claim.json", {
        "protocol": recovery.VERSION, "recipe_sha256": recipe_sha, "binding": binding,
    })
    for phase in ("before", "after"):
        original.fresh(path / ("memory-" + phase + ".json"), {"phase": phase, "within_caps": True})
    value = {"scores": [0.125, 0.875], "union_ids": ["a", "b"]}
    original.fresh(path / "result.json", {"claim_sha256": claim_sha, "value": value})
    original.fresh(path / "closed.json", {"status": "closed", "files": original.private_tree(path)})
    return binding, recipe_sha, value


@pytest.mark.parametrize("damage,reason", [
    (None, None), ("missing", "recovery_job_incomplete"), ("extra", "recovery_job_incomplete"),
    ("binding", "recovery_job_binding"), ("recipe", "recovery_job_binding"),
    ("claim_link", "recovery_job_binding"), ("memory_hash", "recovery_job_hash"),
])
def test_closed_job_exact_namespace_hash_and_binding(recovery, tmp_path, damage, reason):
    job = tmp_path / "job"
    binding, recipe_sha, value = make_closed_job(recovery, job)
    if damage == "missing":
        (job / "memory-before.json").unlink()
    elif damage == "extra":
        recovery.original.fresh(job / "extra.json", {})
    elif damage == "binding":
        binding = {**binding, "query_index": 230}
    elif damage == "recipe":
        recipe_sha = "c" * 64
    elif damage == "claim_link":
        raw = json.loads((job / "result.json").read_bytes())
        raw["claim_sha256"] = "d" * 64
        (job / "result.json").write_bytes(recovery.original.canonical(raw))
    elif damage == "memory_hash":
        (job / "memory-after.json").write_bytes(b'{"within_caps":false}\n')
    if damage:
        with pytest.raises(ValueError, match="^" + reason + "$"):
            recovery.closed_job(job, binding, recipe_sha)
    else:
        assert recovery.closed_job(job, binding, recipe_sha) == value


def test_unknown_claim_is_new_attempt_with_pinned_historical_link(recovery, tmp_path):
    root = tmp_path.resolve()
    path = root / "old/jobs/q-000229/claim.json"
    path.parent.mkdir(parents=True)
    recovery.original.fresh(path, {"historical": "synthetic unknown completion"})
    ref = recovery.original.frozen.record(root, "old/jobs/q-000229/claim.json")
    recipe = {"inheritance": {"output": "old", "unknown_indices": [229],
                             "tree": {"jobs/q-000229/claim.json": ref["sha256"]}}}
    assert recovery.unknown_claim(root, recipe, 229) == {
        "historical_claim": ref, "historical_inference_completion": "unknown",
        "new_attempt": True, "cached_score_reuse": False,
    }
    assert recovery.unknown_claim(root, recipe, 230) is None
    assert recovery.unknown_claim(root, {"inheritance": None}, 229) is None
    path.write_bytes(b'{"changed":true}\n')
    with pytest.raises(ValueError, match="^unknown_claim_changed$"):
        recovery.unknown_claim(root, recipe, 229)


@pytest.mark.parametrize("fail_second", [False, True])
def test_execute_segment_closure_reuse_and_stop_without_retry(recovery, monkeypatch, tmp_path, fail_second):
    """Actual segment flow and file/lock guards; only scoring/input preparation are fake."""
    root = tmp_path.resolve()
    original = recovery.original
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        monkeypatch.setenv(name, "synthetic-before")
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    (root / "inputs").mkdir(mode=0o700)
    def ref(name, value):
        relative = "inputs/" + name + ".json"
        original.fresh(root / relative, value)
        return original.frozen.record(root, relative)
    parent_ref, context_ref = ref("parent", {}), ref("context", {})
    evidence_ref, admission_ref = ref("evidence", {}), ref("admission", {})
    (root / "runs").mkdir(mode=0o700)
    lock_path = root / "synthetic-gpu.lock"
    original.frozen.exclusive_write(lock_path, b"")
    lock_stat = lock_path.stat()
    recipe = {"protocol": recovery.VERSION, "output": "runs/synthetic", "inheritance": {"closed_indices": [0, 1], "unknown_indices": []},
              "parent_manifest": parent_ref, "context_evidence": context_ref,
              "accepted_evidence": [evidence_ref]}
    recipe_ref = ref("recipe", recipe)
    manifest = {"gpu_lock": {"path": lock_path.name, "device": lock_stat.st_dev, "inode": lock_stat.st_ino},
                "source_sha256": {}, "sdk_source_sha256": {}}
    dataset = SimpleNamespace(
        corpus=[{"id": "a"}, {"id": "b"}], queries=[{"id": f"q{i}"} for i in range(19)],
        qrels={f"q{i}": {"a": 1} for i in range(19)}, model_item=lambda row, **_: row,
    )
    rankings = object()
    recovery.load = lambda *_: (recipe, manifest, dataset, rankings)
    recovery.inherited_values = lambda *_: {0: {"inherited": True}, 1: {"inherited": True}}
    recovery.check_source = lambda *_: None
    recovery.history_check = lambda *_: None
    recovery.history_observations = lambda *_: {}
    original.sources = original.sdk_sources = lambda *_: None
    original.observations = lambda *_: {"synthetic_input": "unchanged"}
    validations = []
    def validate(_root, _manifest, *, verify_baseline):
        assert _root == root and _manifest is manifest and verify_baseline is False
        validations.append("validated")
        return dataset, rankings
    original.validate = validate
    original.graph = lambda _rankings, index: {"index": index}
    original.union_ids = lambda _graph: ["a", "b"]
    original.query_binding = lambda sha, index, *_: {"manifest_sha256": sha, "query_index": index}
    scatter_calls, calls, closed, instances = [], [], [], []
    def scatter(graph, ids, scores, qrels):
        assert ids == ["a", "b"] and scores == [0.25, 0.75] and qrels == {"a": 1}
        scatter_calls.append(graph["index"])
    original.scatter = scatter
    class FakeBackend:
        def __init__(self, config):
            assert config is manifest
            instances.append(self)
        def score(self, query, documents, publish):
            assert documents == [{"id": "a"}, {"id": "b"}]
            calls.append(query["id"])
            for phase in ("before", "after"):
                publish(phase, {"synthetic": True, "phase": phase, "within_caps": True})
            if fail_second and len(calls) == 2:
                raise ValueError("synthetic_score_failure")
            return [0.25, 0.75], 0.01
        def close(self):
            closed.append(self)
    recovery.Backend = FakeBackend
    output = root / recipe["output"]
    stage = output / "segments/segment-0000"
    if fail_second:
        with pytest.raises(ValueError, match="^synthetic_score_failure$"):
            recovery.execute_segment(root, recipe_ref, admission_ref, 0)
        assert calls == ["q2", "q3"] and scatter_calls == [2]
        assert not (stage / "closed.json").exists()
        assert set(p.name for p in (stage / "jobs").iterdir()) == {"q-000002", "q-000003"}
        assert set(original.private_tree(stage / "jobs/q-000003")) == {
            "claim.json", "memory-before.json", "memory-after.json",
        }
        assert validations == []
    else:
        first = recovery.execute_segment(root, recipe_ref, admission_ref, 0)
        assert first["status"] == "segment_closed_pending_independent_review" and first["new_queries"] == 16
        assert first["segment_index"] == 0 and first["segment_wall_seconds"] >= 0
        recovery.sealed_segment(stage, recipe_ref["sha256"], 0, list(range(2, 18)))
        preserved = original.private_tree(stage)
        wait_ref = ref("wait", {"actual_wait": True, "exit_code": 0, "signal": None, "pid": 123})
        predecessor = ref("predecessor", {"protocol": recovery.VERSION, "recipe_sha256": recipe_ref["sha256"],
            "next_segment_index": 1, "tree": original.private_tree(output), "actual_child_wait": wait_ref})
        with pytest.raises(ValueError, match="^predecessor_seal_required$"):
            recovery.execute_segment(root, recipe_ref, admission_ref, 1)
        second = recovery.execute_segment(root, recipe_ref, admission_ref, 1, predecessor_ref=predecessor)
        assert second["new_queries"] == 1 and second["segment_index"] == 1
        assert original.private_tree(stage) == preserved
        recovery.sealed_segment(output / "segments/segment-0001", recipe_ref["sha256"], 1, [18])
        assert calls == [f"q{i}" for i in range(2, 19)] and scatter_calls == list(range(2, 19))
        assert validations == ["validated", "validated"]
    binding = {"manifest_sha256": parent_ref["sha256"], "query_index": 2}
    value = recovery.closed_job(stage / "jobs/q-000002", binding, recipe_ref["sha256"])
    assert value["scores"] == [0.25, 0.75]
    token = recovery.segment_token(recipe_ref["sha256"], 0)
    assert value["segment"] == token and len(token) == 32
    assert json.loads((stage / "claim.json").read_bytes())["score_segment"] == token
    original.closed_value(value, {"index": 2}, {"a": 1})
    with pytest.raises(ValueError, match="^closed_segment$"):
        original.closed_value({**value, "segment": stage.name}, {"index": 2}, {"a": 1})
    assert closed == instances and len(instances) == (1 if fail_second else 2)
    assert set(p.name for p in output.iterdir()) == {"recipe.json", "segments", "owner.lock"}
    unchanged = original.private_tree(output)
    prior_calls = list(calls)
    with pytest.raises(ValueError, match="^initial_segment_no_retry$"):
        recovery.execute_segment(root, recipe_ref, admission_ref, 0)
    assert original.private_tree(output) == unchanged and calls == prior_calls


@pytest.mark.parametrize("damage,reason", [
    (None, None), ("tree", "predecessor_tree"), ("recipe", "predecessor_tree"),
    ("index", "predecessor_tree"), ("failure", "predecessor_actual_wait"),
    ("signal", "predecessor_actual_wait"), ("not_waited", "predecessor_actual_wait"),
])
def test_predecessor_binds_tree_recipe_and_actual_successful_wait(recovery, tmp_path, damage, reason):
    root = tmp_path.resolve()
    original = recovery.original
    recipe = {"output": "result"}
    recipe_sha = "a" * 64
    output = root / "result"
    original.mkdir_private(output)
    original.fresh(output / "recipe.json", recipe)
    original.mkdir_private(output / "segments")
    stage = output / "segments/segment-0000"
    original.mkdir_private(stage)
    original.mkdir_private(stage / "jobs")
    original.fresh(stage / "claim.json", {"protocol": recovery.VERSION, "recipe_sha256": recipe_sha,
        "segment_index": 0, "indices": [], "score_segment": recovery.segment_token(recipe_sha, 0)})
    original.frozen.exclusive_write(stage / "private-library.log", b"")
    original.fresh(stage / "closed.json", {"status": "closed", "files": original.private_tree(stage)})
    wait = {"actual_wait": damage != "not_waited", "exit_code": 1 if damage == "failure" else 0,
            "signal": 9 if damage == "signal" else None, "pid": 123}
    original.fresh(root / "wait.json", wait)
    prior = {"protocol": recovery.VERSION, "recipe_sha256": "b" * 64 if damage == "recipe" else recipe_sha,
             "next_segment_index": 2 if damage == "index" else 1,
             "tree": {} if damage == "tree" else original.private_tree(output),
             "actual_child_wait": original.frozen.record(root, "wait.json")}
    original.fresh(root / "prior.json", prior)
    ref = original.frozen.record(root, "prior.json")
    before = original.private_tree(output)
    if reason:
        with pytest.raises(ValueError, match="^" + reason + "$"):
            recovery.stage_root(root, recipe, {"sha256": recipe_sha}, [[]], 1, final=True, predecessor_ref=ref)
    else:
        assert recovery.stage_root(root, recipe, {"sha256": recipe_sha}, [[]], 1,
                                   final=True, predecessor_ref=ref) == output
    assert original.private_tree(output) == before

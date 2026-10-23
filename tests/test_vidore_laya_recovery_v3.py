"""Reboot-only synthetic delta; no models, tokenizer, or real receipt replay."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("reboot_fixture", ROOT / "tests/test_vidore_laya_recovery_v1.py")
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)
fixture.SOURCE = ROOT / "src/rag_benchmark/vidore_laya_recovery_v3.py"


@pytest.fixture
def recovery():
    module = fixture.recovery.__wrapped__()
    module.original.TARGETS = {"vidore-v3-energy-fr": (2225, 308),
        "vidore-v3-finance_en-en": (2942, 309), "vidore-v3-finance_fr-fr": (2384, 320),
        "vidore-v3-hr-en": (1110, 318), "vidore-v3-pharmaceuticals-en": (2313, 364),
        "vidore-v3-physics-fr": (1674, 302)}
    return module


def make_recipe(m, name):
    recipe = dict.fromkeys(m.RECIPE_FIELDS)
    recipe.update(protocol=m.VERSION, dataset_id=name, output=m.WORK + "/runs/" + name,
                  segment_query_limit=16)
    if name == m.FINANCE:
        recipe["inheritance"] = {"output": m.PREVIOUS_WORK + "/runs/" + name,
            "closed_indices": list(range(231)), "unknown_indices": [231]}
    return recipe


def test_partition_only_remaining_queries_and_energy_excluded(recovery):
    groups = []
    for name, (_, count) in recovery.original.TARGETS.items():
        recipe = make_recipe(recovery, name)
        if name == recovery.ENERGY:
            with pytest.raises(ValueError, match="^reboot_target_or_segment_limit$"):
                recovery.recipe_scope(recipe)
            continue
        recovery.recipe_scope(recipe)
        parts = recovery.partition_indices(count, list(range(231)) if name == recovery.FINANCE else [])
        groups += parts
        if name == recovery.FINANCE:
            assert [i for part in parts for i in part] == list(range(231, 309))
    assert len(groups) == 87 and sum(map(len, groups)) == 1382
    assert max(map(len, groups)) == 16


@pytest.mark.parametrize("change", ["missing", "closed", "unknown", "output", "other_inherit"])
def test_reboot_scope_is_fixed_and_cannot_replay_closed_queries(recovery, change):
    recipe = make_recipe(recovery, recovery.FINANCE)
    if change == "missing":
        recipe["inheritance"] = None
    elif change == "closed":
        recipe["inheritance"]["closed_indices"] = list(range(230))
    elif change == "unknown":
        recipe["inheritance"]["unknown_indices"] = []
    elif change == "output":
        recipe["output"] = recovery.PREVIOUS_WORK + "/runs/" + recovery.FINANCE
    else:
        recipe["dataset_id"] = "vidore-v3-hr-en"
        recipe["output"] = recovery.WORK + "/runs/" + recipe["dataset_id"]
    with pytest.raises(ValueError):
        recovery.recipe_scope(recipe)


def history_fixture(m, root):
    o = m.original
    def ref(path, value):
        p = root / path
        p.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        o.fresh(p, value)
        return o.frozen.record(root, path)
    b = m.PREVIOUS_WORK
    source = {}
    for path in ("src/rag_benchmark/vidore_laya_recovery_v2.py", "scripts/run_vidore_laya_recovery_v2.py"):
        p = root / path
        p.parent.mkdir(parents=True, exist_ok=True)
        o.frozen.exclusive_write(p, b"# synthetic source\n")
        source[path] = o.frozen.record(root, path)["sha256"]
    refs = {}
    refs["parent_claim"] = ref(b + "/root-stages/scoring-001/claim.json", {
        "protocol": m.PREVIOUS_VERSION, "pid": 65332, "started_at_utc": "2026-10-23T03:04:59Z"})
    refs["active_child_started"] = ref(b + "/root-stages/scoring-001/" + m.FINANCE + "-segment-0014.started.json",
        {"pid": 78605, "started_at_utc": "2026-10-23T06:22:00Z"})
    refs["unknown_query_claim"] = ref(b + "/runs/" + m.FINANCE + "/segments/segment-0014/jobs/q-000231/claim.json", {})
    refs["reboot_evidence"] = ref("proof/observation.json", {
        "schema": "vidore-laya-reboot-process-observation-v1", "boot_epoch_seconds": 1792737027,
        "boot_time_utc": "2026-10-23T06:30:27Z", "observed_at_utc": "2026-10-23T06:45:04Z",
        "parent_claim": refs["parent_claim"], "active_child_started": refs["active_child_started"],
        "historical_process_cessation": "machine_reboot", "reboot_cause": "UNKNOWN",
        "global_detached_absence_claimed": False,
        "known_numeric_pid_absence_at_observation": {"65322": True, "65332": True, "78605": True},
        "parent_exit_code": None, "parent_signal": None, "active_child_exit_code": None,
        "active_child_signal": None, "tool_session_poll": {"actual_exit_code": None, "actual_signal": None}})
    refs["independent_reuse_admission"] = ref("proof/reuse.json", {
        "status": "PASS_231_CLOSED_RECORDS_REUSE_ONLY", "blocking_findings": [],
        "eligible_closed_queries": 231, "closed_queries_with_true_segment_wait": 224,
        "closed_queries_without_segment_terminal_wait": 7, "unresolved_historical_claims": 1,
        "never_claimed_remaining_queries": 77, "final_dataset_science_pass": False,
        "model_launch_authorized": False})
    refs["previous_recipe"] = ref(b + "/recipes/" + m.FINANCE + ".json", {
        "protocol": m.PREVIOUS_VERSION, "dataset_id": m.FINANCE, "output": b + "/runs/" + m.FINANCE,
        "inheritance": None, "source_sha256": source})
    refs["energy_summary"] = ref(b + "/runs/" + m.ENERGY + "/summary.json", {
        "protocol": m.PREVIOUS_VERSION, "status": "completed", "dataset_id": m.ENERGY,
        "query_count": 308, "inherited_closed_queries": 229, "new_completed_queries": 79,
        "reattempted_unknown_outcome_queries": 1, "recipe_sha256": "e" * 64})
    refs["energy_closed"] = ref(b + "/runs/" + m.ENERGY + "/closed.json", {})
    refs["energy_finalizer_wait"] = ref(b + "/root-stages/scoring-001/" + m.ENERGY + "-finalize.wait.json", {
        "protocol": m.PREVIOUS_VERSION, "action": "finalize", "dataset_id": m.ENERGY,
        "recipe_sha256": "e" * 64, "actual_wait": True, "exit_code": 0, "signal": None})
    refs["original_v4_wait"] = ref("proof/original-wait.json", {"exit_code": 1, "tool_status": "completed"})
    refs["recovery_tree_seal"] = ref("proof/recovery-tree.json", {})
    refs["original_tree_seal"] = ref("proof/original-tree.json", {})
    refs["reboot_diagnosis"] = ref("proof/diagnosis.json", {
        "status": "CONFIRMED_BOOT_BOUNDARY_WITH_UNKNOWN_TERMINAL_STATUS",
        "boot_observation": refs["reboot_evidence"], "recovery_tree_seal": refs["recovery_tree_seal"],
        "original_v4_tree_seal": refs["original_tree_seal"], "reboot_cause": "UNKNOWN",
        "same_run_retry_authorized": False, "historical_v4_unknown_attempt_remains_preserved": True,
        "parent_exit_code": None, "parent_signal": None, "active_child_exit_code": None,
        "active_child_signal": None})
    history = {"protocol": "vidore-laya-reboot-history-v1",
        "status": "PASS_ROOT_REBOOT_HISTORY_REUSE_AUTHORIZED", "refs": refs,
        "historical_process_cessation": "machine_reboot", "reboot_cause": "UNKNOWN",
        "parent_exit_code": None, "parent_signal": None, "active_child_exit_code": None,
        "active_child_signal": None, "finance_closed_indices": list(range(231)),
        "finance_unknown_indices": [231], "energy_reused_queries": 308, "finance_reused_queries": 231,
        "new_queries": 1382, "new_score_children": 87, "new_finalizers": 5}
    return history, ref


@pytest.mark.parametrize("damage", [None, "parent_exit", "signal", "boot", "observed_preboot",
    "active_postboot", "parent_postboot", "global_absence", "reuse", "old_wait", "energy_wait", "source"])
def test_reboot_boundary_unknown_exits_and_reuse_authority(recovery, tmp_path, damage):
    root = tmp_path.resolve()
    history, ref = history_fixture(recovery, root)
    refs = history["refs"]
    if damage == "parent_exit":
        history["parent_exit_code"] = 0
    elif damage == "signal":
        history["active_child_signal"] = 9
    else:
        changes = {
            "boot": ("reboot_evidence", "boot_epoch_seconds", 1792737028),
            "observed_preboot": ("reboot_evidence", "observed_at_utc", "2026-10-23T06:00:00Z"),
            "active_postboot": ("active_child_started", "started_at_utc", "2026-10-23T07:00:00Z"),
            "parent_postboot": ("parent_claim", "started_at_utc", "2026-10-23T07:00:00Z"),
            "global_absence": ("reboot_evidence", "global_detached_absence_claimed", True),
            "reuse": ("independent_reuse_admission", "closed_queries_without_segment_terminal_wait", 0),
            "old_wait": ("original_v4_wait", "exit_code", 0),
            "energy_wait": ("energy_finalizer_wait", "exit_code", None),
        }
        if damage in changes:
            key, field, value = changes[damage]
            p = root / refs[key]["path"]
            obj = json.loads(p.read_bytes())
            obj[field] = value
            p.write_bytes(recovery.original.canonical(obj) + b"\n")
            refs[key] = recovery.original.frozen.record(root, refs[key]["path"])
        elif damage == "source":
            (root / "src/rag_benchmark/vidore_laya_recovery_v2.py").write_bytes(b"changed")
    recipe = {"reboot_history": ref("proof/history.json", history)}
    if damage:
        with pytest.raises(ValueError):
            recovery.reboot_history(root, recipe)
    else:
        assert recovery.reboot_history(root, recipe) == history
        linked = recovery.history_refs(root, recipe)
        assert len(linked) == 1 + len(refs) + 2


def historical_job(m, path, *, damage=None):
    o = m.original
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    o.mkdir_private(path)
    binding = {"query_index": 230}
    claim = {"protocol": m.PREVIOUS_VERSION, "recipe_sha256": "b" * 64, "binding": binding, "pid": 78605}
    if damage in ("protocol", "pid", "binding"):
        claim[damage] = {"protocol": m.VERSION, "pid": 1, "binding": {"query_index": 229}}[damage]
    digest = o.fresh(path / "claim.json", claim)
    value = {"fresh_unique_pairs": 2, "union_ids": ["d0", "d1"], "scores": [0.2, 0.8],
             "union_scoring_seconds": 0.4}
    for phase in ("before", "after"):
        o.fresh(path / ("memory-" + phase + ".json"), {"within_caps": True,
            "caps": o.MEMORY, "after_cache_release": {"peak_process_rss_bytes": 1024,
                "mps_driver_allocated_bytes": 65 * 1024**3 if damage == "memory" else 2048}})
    if damage != "claim_only":
        o.fresh(path / "result.json", {"claim_sha256": digest, "value": value})
        o.fresh(path / "closed.json", {"status": "closed", "files": o.private_tree(path)})
    if damage == "hash":
        (path / "memory-before.json").write_bytes(b"{}")
    return binding, value


@pytest.mark.parametrize("damage", [None, "protocol", "pid", "binding", "claim_only", "hash", "memory"])
def test_partial_segment_reuses_only_fully_closed_bound_jobs(recovery, tmp_path, damage):
    path = tmp_path.resolve() / "partial/jobs/q-000230"
    binding, value = historical_job(recovery, path, damage=damage)
    args = path, binding, "b" * 64, {"pid": 78605}
    if damage:
        with pytest.raises(ValueError):
            recovery.historical_closed_job(*args)
    else:
        assert recovery.historical_closed_job(*args) == value
        assert not (path.parent.parent / "closed.json").exists()
        assert not (path.parent.parent / "wait.json").exists()


def test_unknown_attempt_stays_linked_and_never_becomes_a_cache_hit(recovery, tmp_path):
    root = tmp_path.resolve()
    recipe = make_recipe(recovery, recovery.FINANCE)
    rel = recipe["inheritance"]["output"] + "/segments/segment-0014/jobs/q-000231/claim.json"
    path = root / rel
    path.parent.mkdir(parents=True)
    recovery.original.fresh(path, {"synthetic": True})
    pinned = recovery.original.frozen.record(root, rel)
    recovery.reboot_history = lambda *_: {"refs": {"unknown_query_claim": pinned}}
    value = recovery.unknown_claim(root, recipe, 231)
    assert value["historical_child_exit_code"] is None
    assert value["historical_inference_completion"] == "unknown"
    assert value["new_attempt"] is True and value["cached_score_reuse"] is False
    assert recovery.unknown_claim(root, recipe, 232) is None
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="^unknown_claim_changed$"):
        recovery.unknown_claim(root, recipe, 231)


def test_new_history_ref_is_rechecked_after_scoring(recovery, tmp_path, monkeypatch):
    root = tmp_path.resolve()
    o = recovery.original
    def ref(name, value):
        o.fresh(root / name, value)
        return o.frozen.record(root, name)
    proof = ref("proof.json", {"synthetic": True})
    recipe_ref, admission_ref = ref("recipe.json", {}), ref("admission.json", {})
    recipe = {"output": "output", "inheritance": None, "accepted_evidence": [],
              "parent_manifest": ref("manifest.json", {}), "context_evidence": ref("context.json", {})}
    o.frozen.exclusive_write(root / "gpu.lock", b"")
    lock = (root / "gpu.lock").stat()
    manifest = {"gpu_lock": {"path": "gpu.lock", "device": lock.st_dev, "inode": lock.st_ino},
                "source_sha256": {}, "sdk_source_sha256": {}}
    dataset = SimpleNamespace(corpus=[{"id": "d"}], queries=[{"id": "q"}], qrels={"q": {"d": 1}},
                              model_item=lambda row, **_: row)
    o.mkdir_private(root / "output")
    o.mkdir_private(root / "output/segments")
    o.fresh(root / "output/recipe.json", recipe)
    recovery.load = lambda *_: (recipe, manifest, dataset, {})
    recovery.stage_root = lambda *_args, **_kwargs: root / "output"
    recovery.check_source = lambda *_: None
    recovery.history_observations = lambda *_: {}
    recovery.history_refs = lambda *_: [proof]
    recovery.lock_rebinding_refs = lambda *_: []
    recovery.lock_rebinding = lambda *_: {}
    recovery.operational_manifest = lambda _root, _recipe, value: value
    o.sources = o.sdk_sources = lambda *_: None
    o.observations = lambda *_: {}
    o.graph = lambda *_: []
    o.union_ids = lambda *_: ["d"]
    o.query_binding = lambda *_: {"binding": True}
    o.scatter = lambda *_: []
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    class Backend:
        def __init__(self, _):
            pass
        def score(self, query, documents, publish):
            publish("before", {})
            publish("after", {})
            (root / "proof.json").write_bytes(b"changed")
            return [0.5], 0.1
        def close(self):
            pass
    recovery.Backend = Backend
    with pytest.raises(ValueError, match="Sealed input bytes changed"):
        recovery.execute_segment(root, recipe_ref, admission_ref, 0)
    job = root / "output/segments/segment-0000/jobs/q-000000"
    assert (job / "claim.json").exists() and not (job / "result.json").exists()


def test_old_incomplete_namespace_cannot_resume_and_science_memory_unchanged(recovery, tmp_path):
    root = tmp_path.resolve()
    recovery.original.mkdir_private(root / "output")
    with pytest.raises(ValueError, match="^initial_segment_no_retry$"):
        recovery.stage_root(root, {"output": "output"}, {"sha256": "a" * 64}, [[0]], 0)
    with pytest.raises(ValueError, match="^predecessor_seal_required$"):
        recovery.stage_root(root, {"output": "output"}, {"sha256": "a" * 64}, [[0]], 1)
    unchanged = {"Backend", "memory_observation", "memory_within_caps", "partition_indices",
        "closed_job", "sealed_segment", "stage_root", "segment_token", "main"}
    def definitions(path):
        tree = ast.parse(path.read_text())
        return {n.name: ast.dump(n) for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                and n.name in unchanged}
    assert definitions(ROOT / "src/rag_benchmark/vidore_laya_recovery_v2.py") == definitions(fixture.SOURCE)
    tree = ast.parse(fixture.SOURCE.read_text())
    final = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "finalize")
    calls = [n for n in ast.walk(final) if isinstance(n, ast.Call)]
    assert any(ast.unparse(n.func) == "original.validate" and
               any(k.arg == "verify_baseline" and k.value.value is False for k in n.keywords) for n in calls)
    assert not any(ast.unparse(n.func) == "Backend" for n in calls)


def lock_fixture(m, root):
    o = m.original
    def ref(name, value):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        o.fresh(path, value)
        return o.frozen.record(root, name)
    lock_path = root / "work/overnight/gpu.lock"
    lock_path.parent.mkdir(parents=True)
    o.frozen.exclusive_write(lock_path, b"")
    lock_path.chmod(0o644)
    locked = o.frozen.record(root, "work/overnight/gpu.lock")
    state = lock_path.stat()
    old = {"path": locked["path"], "device": 16777232, "inode": 135900020}
    new = {**old, "device": 16777234}
    boot = ref("boot.json", {"synthetic": True})
    recipe = make_recipe(m, m.FINANCE)
    recipe["reboot_history"] = ref("history.json", {"refs": {"reboot_evidence": boot}})
    recipe["parent_manifest"] = ref("manifest.json", {"synthetic": True})
    observation = {"schema": "vidore-laya-reboot-current-lock-observation-v1",
        "status": "OBSERVED_EXISTING_LOCK_WITH_EXCLUSIVE_NONMUTATING_FLOCK",
        "boot_epoch_seconds": 1792737027, "boot_observation": boot,
        "observed_at_utc": "2026-10-23T06:53:31Z", "original_identity": old,
        "current_identity": {**new, "size": 0, "sha256": locked["sha256"], "mode": "0644",
            "uid": state.st_uid, "gid": state.st_gid, "nlink": state.st_nlink},
        "original_manifest_refs": {m.FINANCE: recipe["parent_manifest"]},
        "allowed_operational_projection_fields": ["gpu_lock.device"],
        "historical_volume_uuid_attestation_available": False, "device_number_change_cause": "UNKNOWN",
        "descriptor_matches_path_before_and_after": True, "exclusive_flock_acquired": True,
        "exclusive_flock_released": True, "lock_file_created": False, "lock_bytes_written": False,
        "lock_file_unlinked_or_replaced": False, "immutable_manifest_changes_performed": False}
    observation_ref = ref("observation.json", observation)
    review_ref = ref("review.json", {"status": "PASS_INDEPENDENT_BOOT_BOUND_LOCK_REBINDING",
        "blocking_findings": [], "observation_sha256": observation_ref["sha256"]})
    binding = {"protocol": "vidore-laya-boot-lock-rebinding-v1",
        "status": "PASS_ROOT_BOOT_BOUND_GPU_LOCK_REBINDING", "changed_fields": ["gpu_lock.device"],
        "cause": "UNKNOWN", "historical_volume_identity_attested": False,
        "boot_epoch_seconds": 1792737027, "old_binding": old, "new_binding": new,
        "current_lock": locked, "current_mode": 0o644,
        "refs": {"observation": observation_ref, "independent_review": review_ref, "boot_observation": boot}}
    manifest = {"gpu_lock": old, "other": {"nested": ["untouched"]}}
    actual_path_at = o.frozen.path_at
    class LockPath:
        def __fspath__(self):
            return str(lock_path)
        def is_symlink(self):
            return lock_path.is_symlink()
        def stat(self, **kwargs):
            current = lock_path.stat(**kwargs)
            fields = {key: getattr(current, key) for key in dir(current) if key.startswith("st_")}
            fields.update(st_dev=16777234, st_ino=135900020)
            return SimpleNamespace(**fields)
    o.frozen.path_at = lambda r, p: LockPath() if p == locked["path"] else actual_path_at(r, p)
    m.current_boot_epoch = lambda: 1792737027
    return recipe, manifest, binding, ref, lock_path


@pytest.mark.parametrize("damage", [None, "unadmitted", "inode", "path", "content", "boot", "mode", "old_device",
                                      "extra_projection", "volume_attestation"])
def test_boot_bound_lock_projection_is_exact_and_fail_closed(recovery, tmp_path, damage):
    root = tmp_path.resolve()
    recipe, manifest, binding, ref, lock_path = lock_fixture(recovery, root)
    frozen_before = copy.deepcopy(manifest)
    if damage == "unadmitted":
        binding["status"] = "SOURCE_ONLY"
    elif damage == "inode":
        binding["new_binding"]["inode"] += 1
    elif damage == "path":
        binding["new_binding"]["path"] = "work/overnight/new.lock"
    elif damage == "content":
        lock_path.write_bytes(b"changed")
    elif damage == "boot":
        recovery.current_boot_epoch = lambda: 1792737028
    elif damage == "mode":
        lock_path.chmod(0o600)
    elif damage == "old_device":
        binding["old_binding"]["device"] = 1
    elif damage == "extra_projection":
        binding["changed_fields"].append("gpu_lock.inode")
    elif damage == "volume_attestation":
        binding["historical_volume_identity_attested"] = True
    recipe["lock_rebinding"] = ref("binding.json", binding)
    if damage:
        with pytest.raises(ValueError):
            recovery.operational_manifest(root, recipe, manifest)
    else:
        # Execute the exact old validator lock seam, with all filesystem reads
        # real except the two boot-specific device/inode observation values.
        source = ast.parse((ROOT / "src/rag_benchmark/vidore_laya_completion_v4.py").read_text())
        validate = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == "validate")
        start = next(i for i, n in enumerate(validate.body) if ast.unparse(n).startswith("gpu ="))
        validate.body = validate.body[start:start + 4] + [ast.Return(value=ast.Constant(True))]
        module = ast.fix_missing_locations(ast.Module(body=[validate], type_ignores=[]))
        exec(compile(module, "old_validate_lock_seam", "exec"), recovery.original.__dict__)
        with pytest.raises(ValueError, match="^gpu_lock_changed$"):
            recovery.original.validate(root, manifest, verify_baseline=False)
        projected = recovery.operational_manifest(root, recipe, manifest)
        assert projected == {**manifest, "gpu_lock": {**manifest["gpu_lock"], "device": 16777234}}
        assert recovery.original.validate(root, projected, verify_baseline=False) is True
        assert manifest == frozen_before
        projected["other"]["nested"].append("only-copy")
        assert manifest == frozen_before

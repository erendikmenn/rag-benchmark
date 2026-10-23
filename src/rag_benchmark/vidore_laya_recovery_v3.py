"""Separate bounded post-reboot Laya recovery; V4 and V2 evidence remain immutable.

No tokenizer, retrieval inference, baseline metric replay, or automatic retry.
The historical failed score attempt has unknown inference completion. A new
attempt is explicitly linked to it; no discarded score is presented as reused.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
import copy
import gc
import json
import os
from pathlib import Path
import resource
import re
import stat
import subprocess
import sys
import time

from . import vidore_laya_completion_v4 as original

VERSION = "vidore-laya-bounded-recovery-v3"
WORK = "work/vidore-laya-recovery-v3"
SEGMENT_QUERIES = 16
PREVIOUS_VERSION = "vidore-laya-bounded-recovery-v2"
PREVIOUS_WORK = "work/vidore-laya-recovery-v2"
FINANCE = "vidore-v3-finance_en-en"
ENERGY = "vidore-v3-energy-fr"
ORIGINAL_ENERGY = "work/vidore-laya-completion-v4/runs/" + ENERGY
HISTORY_REF_KEYS = {"reboot_evidence", "reboot_diagnosis", "independent_reuse_admission",
    "previous_recipe", "energy_summary", "energy_closed", "energy_finalizer_wait",
    "recovery_tree_seal", "original_tree_seal", "parent_claim", "active_child_started",
    "unknown_query_claim", "original_v4_wait"}
SOURCES = ("src/rag_benchmark/vidore_laya_recovery_v3.py",
           "scripts/run_vidore_laya_recovery_v3.py")
RECIPE_FIELDS = {"protocol", "dataset_id", "parent_manifest", "context_evidence", "output",
                 "source_sha256", "segment_query_limit", "inheritance", "accepted_evidence", "reboot_history",
                 "lock_rebinding"}


def require(condition, reason):
    original.require(condition, reason)


def current_rss_bytes():
    """Current OS resident pages, diagnostic only; the original peak cap remains."""
    result = subprocess.run(["/bin/ps", "-o", "rss=", "-p", str(os.getpid())],
                            check=True, capture_output=True, text=True)
    value = int(result.stdout.strip()) * 1024
    require(value > 0, "resident_measurement")
    return value


def memory_observation():
    import torch
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != "darwin":
        peak *= 1024
    return {"peak_process_rss_bytes": int(peak), "current_process_rss_bytes": current_rss_bytes(),
            "mps_driver_allocated_bytes": int(torch.mps.driver_allocated_memory()),
            "mps_current_allocated_bytes": int(torch.mps.current_allocated_memory())}


def memory_within_caps(value):
    return (value["peak_process_rss_bytes"] <= original.MEMORY["maximum_process_rss_bytes"] and
            value["mps_driver_allocated_bytes"] <= original.MEMORY["maximum_mps_driver_bytes"])


class Backend:
    """Same adapter/configuration; cleanup is confined to query boundaries."""
    def __init__(self, manifest):
        original.Backend.__init__(self, manifest)

    def guard(self, phase, publish):
        import torch
        original.models.synchronize("mps")
        before = memory_observation()
        gc.collect()
        torch.mps.empty_cache()
        original.models.synchronize("mps")
        after = memory_observation()
        # Publish observations before rejecting, including the post-score case.
        publish(phase, {"before_cache_release": before, "after_cache_release": after,
                        "caps": original.MEMORY, "within_caps": memory_within_caps(after),
                        "observed_at_utc": original.utc()})
        require(memory_within_caps(after), "memory_guard")

    def score(self, query, documents, publish):
        self.guard("before", publish)
        original.models.synchronize("mps")
        tick = time.perf_counter_ns()
        scores = list(map(float, self.adapter.score(query, documents)))
        original.models.synchronize("mps")
        elapsed = (time.perf_counter_ns() - tick) / 1e9
        self.guard("after", publish)
        return scores, elapsed

    def close(self):
        self.adapter.unload()


def partition_indices(count, inherited):
    require(type(count) is int and count > 0 and
            inherited == sorted(set(inherited)) and all(type(i) is int and 0 <= i < count
                                                       for i in inherited), "inherited_indices")
    remaining = [i for i in range(count) if i not in set(inherited)]
    return [remaining[i:i + SEGMENT_QUERIES] for i in range(0, len(remaining), SEGMENT_QUERIES)]


def check_source(root, recipe):
    require(set(recipe["source_sha256"]) == set(SOURCES), "recovery_source_population")
    for path, digest in recipe["source_sha256"].items():
        require(original.frozen.record(root, path)["sha256"] == digest, "recovery_source_changed")


def read_pinned(root, ref):
    return original.frozen.read_json(root, ref)


def current_boot_epoch():
    result = subprocess.run(["/usr/sbin/sysctl", "-n", "kern.boottime"],
                            check=True, capture_output=True, text=True)
    match = re.search(r"\bsec\s*=\s*(\d+)\s*[,}]", result.stdout)
    require(match is not None, "current_boot_measurement")
    return int(match.group(1))


def lock_rebinding(root, recipe, manifest):
    """Admit one boot-specific device-number change; no old evidence is rewritten."""
    binding = read_pinned(root, recipe["lock_rebinding"])
    require(binding["protocol"] == "vidore-laya-boot-lock-rebinding-v1" and
            binding["status"] == "PASS_ROOT_BOOT_BOUND_GPU_LOCK_REBINDING" and
            binding["changed_fields"] == ["gpu_lock.device"] and
            binding["cause"] == "UNKNOWN" and binding["historical_volume_identity_attested"] is False,
            "lock_rebinding_admission")
    require(set(binding["refs"]) >= {"observation", "independent_review", "boot_observation"},
            "lock_rebinding_refs")
    for ref in binding["refs"].values():
        original.frozen.checked(root, ref)
    review = read_pinned(root, binding["refs"]["independent_review"])
    require(review["status"] == "PASS_INDEPENDENT_BOOT_BOUND_LOCK_REBINDING" and
            review["blocking_findings"] == [] and
            review["observation_sha256"] == binding["refs"]["observation"]["sha256"],
            "lock_rebinding_independent_review")
    history = read_pinned(root, recipe["reboot_history"])
    require(binding["refs"]["boot_observation"] == history["refs"]["reboot_evidence"] and
            binding["boot_epoch_seconds"] == 1792737027 == current_boot_epoch(), "lock_rebinding_boot")
    old, new = binding["old_binding"], binding["new_binding"]
    require(old == manifest["gpu_lock"] and set(old) == set(new) == {"path", "device", "inode"} and
            old["path"] == new["path"] == "work/overnight/gpu.lock" and
            old["inode"] == new["inode"] == 135900020 and old["device"] == 16777232 and
            new["device"] == 16777234, "lock_rebinding_identity")
    ref = binding["current_lock"]
    require(ref["path"] == new["path"], "lock_rebinding_content_path")
    original.frozen.checked(root, ref)
    observation = read_pinned(root, binding["refs"]["observation"])
    current = observation["current_identity"]
    require(observation["schema"] == "vidore-laya-reboot-current-lock-observation-v1" and
            observation["status"] == "OBSERVED_EXISTING_LOCK_WITH_EXCLUSIVE_NONMUTATING_FLOCK" and
            observation["boot_epoch_seconds"] == binding["boot_epoch_seconds"] and
            observation["boot_observation"] == binding["refs"]["boot_observation"] and
            timestamp(observation["observed_at_utc"]) > binding["boot_epoch_seconds"] and
            observation["original_identity"] == old and
            {k: current[k] for k in ("path", "device", "inode")} == new and
            current["size"] == ref["bytes"] and current["sha256"] == ref["sha256"] and
            int(current["mode"], 8) == binding["current_mode"] and
            observation["original_manifest_refs"][recipe["dataset_id"]] == recipe["parent_manifest"] and
            observation["allowed_operational_projection_fields"] == ["gpu_lock.device"] and
            observation["historical_volume_uuid_attestation_available"] is False and
            observation["device_number_change_cause"] == "UNKNOWN" and
            all(observation[k] is True for k in ("descriptor_matches_path_before_and_after",
                "exclusive_flock_acquired", "exclusive_flock_released")) and
            all(observation[k] is False for k in ("lock_file_created", "lock_bytes_written",
                "lock_file_unlinked_or_replaced", "immutable_manifest_changes_performed")),
            "lock_rebinding_observation")
    path = original.frozen.path_at(root, new["path"])
    state = path.stat(follow_symlinks=False)
    require(not path.is_symlink() and stat.S_ISREG(state.st_mode) and
            (state.st_dev, state.st_ino) == (new["device"], new["inode"]) and
            stat.S_IMODE(state.st_mode) == binding["current_mode"] and
            (state.st_uid, state.st_gid, state.st_nlink) == (current["uid"], current["gid"], current["nlink"]),
            "lock_rebinding_current_identity")
    return binding


def operational_manifest(root, recipe, manifest):
    binding = lock_rebinding(root, recipe, manifest)
    projected = copy.deepcopy(manifest)
    projected["gpu_lock"]["device"] = binding["new_binding"]["device"]
    restored = copy.deepcopy(projected)
    restored["gpu_lock"]["device"] = binding["old_binding"]["device"]
    require(restored == manifest and projected != manifest, "lock_rebinding_exact_diff")
    return projected


def lock_rebinding_refs(root, recipe, manifest):
    binding = lock_rebinding(root, recipe, manifest)
    return [recipe["lock_rebinding"], binding["current_lock"], *binding["refs"].values()]


def recipe_scope(recipe):
    require(set(recipe) == RECIPE_FIELDS and recipe["protocol"] == VERSION, "recovery_recipe_fields")
    require(recipe["dataset_id"] in set(original.TARGETS) - {ENERGY} and
            recipe["segment_query_limit"] == SEGMENT_QUERIES, "reboot_target_or_segment_limit")
    require(recipe["output"] == WORK + "/runs/" + recipe["dataset_id"], "recovery_output")
    inherited = recipe["inheritance"]
    if recipe["dataset_id"] == FINANCE:
        require(inherited == {"output": PREVIOUS_WORK + "/runs/" + FINANCE,
                "closed_indices": list(range(231)), "unknown_indices": [231]}, "reboot_inheritance_scope")
    else:
        require(inherited is None, "unexpected_reboot_inheritance")


def timestamp(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(parsed.tzinfo is not None, "historical_timestamp_timezone")
    return parsed.timestamp()


def reboot_history(root, recipe):
    history = read_pinned(root, recipe["reboot_history"])
    require(history["protocol"] == "vidore-laya-reboot-history-v1" and
            history["status"] == "PASS_ROOT_REBOOT_HISTORY_REUSE_AUTHORIZED", "reboot_history_admission")
    require(HISTORY_REF_KEYS <= set(history["refs"]), "reboot_history_refs")
    for ref in history["refs"].values():
        original.frozen.checked(root, ref)
    require(history["historical_process_cessation"] == "machine_reboot" and
            history["reboot_cause"] == "UNKNOWN" and
            all(history[k] is None for k in ("parent_exit_code", "parent_signal",
                "active_child_exit_code", "active_child_signal")), "historical_exit_unknown")
    require(history["finance_closed_indices"] == list(range(231)) and
            history["finance_unknown_indices"] == [231] and history["energy_reused_queries"] == 308 and
            history["finance_reused_queries"] == 231 and history["new_queries"] == 1382 and
            history["new_score_children"] == 87 and history["new_finalizers"] == 5,
            "reboot_history_population")
    refs = history["refs"]
    expected_paths = {
        "previous_recipe": PREVIOUS_WORK + "/recipes/" + FINANCE + ".json",
        "parent_claim": PREVIOUS_WORK + "/root-stages/scoring-001/claim.json",
        "active_child_started": PREVIOUS_WORK + "/root-stages/scoring-001/" + FINANCE + "-segment-0014.started.json",
        "energy_summary": PREVIOUS_WORK + "/runs/" + ENERGY + "/summary.json",
        "energy_closed": PREVIOUS_WORK + "/runs/" + ENERGY + "/closed.json",
        "energy_finalizer_wait": PREVIOUS_WORK + "/root-stages/scoring-001/" + ENERGY + "-finalize.wait.json",
        "unknown_query_claim": PREVIOUS_WORK + "/runs/" + FINANCE + "/segments/segment-0014/jobs/q-000231/claim.json",
    }
    require(all(refs[name]["path"] == path for name, path in expected_paths.items()), "historical_reference_path")
    observation = read_pinned(root, refs["reboot_evidence"])
    parent = read_pinned(root, refs["parent_claim"])
    active = read_pinned(root, refs["active_child_started"])
    boot = observation["boot_epoch_seconds"]
    require(observation["schema"] == "vidore-laya-reboot-process-observation-v1" and
            boot == 1792737027 and timestamp(observation["boot_time_utc"]) == boot and
            timestamp(parent["started_at_utc"]) < boot and timestamp(active["started_at_utc"]) < boot and
            timestamp(observation["observed_at_utc"]) > boot and
            observation["parent_claim"] == refs["parent_claim"] and
            observation["active_child_started"] == refs["active_child_started"] and
            parent["protocol"] == PREVIOUS_VERSION and parent["pid"] == 65332 and active["pid"] == 78605,
            "reboot_boundary_binding")
    require(observation["historical_process_cessation"] == "machine_reboot" and
            observation["reboot_cause"] == "UNKNOWN" and
            observation["global_detached_absence_claimed"] is False and
            observation["known_numeric_pid_absence_at_observation"] ==
                {"65322": True, "65332": True, "78605": True} and
            all(observation[k] is None for k in ("parent_exit_code", "parent_signal",
                "active_child_exit_code", "active_child_signal")) and
            observation["tool_session_poll"]["actual_exit_code"] is None and
            observation["tool_session_poll"]["actual_signal"] is None, "reboot_observation_scope")
    review = read_pinned(root, refs["independent_reuse_admission"])
    require(review["status"] == "PASS_231_CLOSED_RECORDS_REUSE_ONLY" and
            review["blocking_findings"] == [] and review["eligible_closed_queries"] == 231 and
            review["closed_queries_with_true_segment_wait"] == 224 and
            review["closed_queries_without_segment_terminal_wait"] == 7 and
            review["unresolved_historical_claims"] == 1 and
            review["never_claimed_remaining_queries"] == 77 and
            review["final_dataset_science_pass"] is False and review["model_launch_authorized"] is False,
            "reboot_reuse_review")
    diagnosis = read_pinned(root, refs["reboot_diagnosis"])
    require(diagnosis["status"] == "CONFIRMED_BOOT_BOUNDARY_WITH_UNKNOWN_TERMINAL_STATUS" and
            diagnosis["boot_observation"] == refs["reboot_evidence"] and
            diagnosis["recovery_tree_seal"] == refs["recovery_tree_seal"] and
            diagnosis["original_v4_tree_seal"] == refs["original_tree_seal"] and
            diagnosis["reboot_cause"] == "UNKNOWN" and diagnosis["same_run_retry_authorized"] is False and
            diagnosis["historical_v4_unknown_attempt_remains_preserved"] is True and
            all(diagnosis[k] is None for k in ("parent_exit_code", "parent_signal",
                "active_child_exit_code", "active_child_signal")), "reboot_diagnosis_binding")
    prior = read_pinned(root, refs["previous_recipe"])
    require(prior["protocol"] == PREVIOUS_VERSION and prior["dataset_id"] == FINANCE and
            prior["output"] == PREVIOUS_WORK + "/runs/" + FINANCE and prior["inheritance"] is None,
            "previous_recipe_scope")
    require(set(prior["source_sha256"]) == {"src/rag_benchmark/vidore_laya_recovery_v2.py",
            "scripts/run_vidore_laya_recovery_v2.py"}, "previous_source_population")
    for path, digest in prior["source_sha256"].items():
        require(original.frozen.record(root, path)["sha256"] == digest, "previous_source_changed")
    original_wait = read_pinned(root, refs["original_v4_wait"])
    require(original_wait["exit_code"] == 1 and original_wait["tool_status"] == "completed",
            "original_memory_failure_preserved")
    energy = read_pinned(root, refs["energy_summary"])
    energy_wait = read_pinned(root, refs["energy_finalizer_wait"])
    require(energy["protocol"] == PREVIOUS_VERSION and energy["status"] == "completed" and
            energy["dataset_id"] == ENERGY and energy["query_count"] == 308 and
            energy["inherited_closed_queries"] == 229 and energy["new_completed_queries"] == 79 and
            energy["reattempted_unknown_outcome_queries"] == 1 and
            energy_wait["protocol"] == PREVIOUS_VERSION and energy_wait["action"] == "finalize" and
            energy_wait["dataset_id"] == ENERGY and energy_wait["recipe_sha256"] == energy["recipe_sha256"] and
            energy_wait["actual_wait"] is True and energy_wait["exit_code"] == 0 and
            energy_wait["signal"] is None, "closed_energy_reuse_scope")
    return history


def history_refs(root, recipe):
    history = reboot_history(root, recipe)
    refs = [recipe["reboot_history"], *history["refs"].values()]
    previous = read_pinned(root, history["refs"]["previous_recipe"])
    refs.extend(original.frozen.record(root, path) for path in previous["source_sha256"])
    return refs


def history_check(root, recipe, manifest):
    history = reboot_history(root, recipe)
    trees = history["sealed_trees"]
    require(len(trees) == 2 and {t["output"] for t in trees} == {PREVIOUS_WORK, ORIGINAL_ENERGY},
            "reboot_tree_population")
    for item in trees:
        require(set(item) == {"output", "tree"}, "reboot_tree_fields")
        seal_name = "recovery_tree_seal" if item["output"] == PREVIOUS_WORK else "original_tree_seal"
        seal = read_pinned(root, history["refs"][seal_name])
        require(seal["root"] == item["output"] and seal["files"] == item["tree"] and
                original.private_tree(original.frozen.path_at(root, item["output"])) == item["tree"],
                "historical_tree_changed")
    previous = read_pinned(root, history["refs"]["previous_recipe"])
    if recipe["dataset_id"] == FINANCE:
        require(previous["parent_manifest"] == recipe["parent_manifest"] and
                previous["context_evidence"] == recipe["context_evidence"], "reused_parent_context")
        old = original.frozen.path_at(root, recipe["inheritance"]["output"])
        require({p.name for p in (old / "segments").iterdir()} ==
                {f"segment-{i:04d}" for i in range(15)}, "historical_segment_population")
        for index in range(231):
            path = old / "segments" / f"segment-{index // 16:04d}" / "jobs" / f"q-{index:06d}"
            require(set(original.private_tree(path)) == {"claim.json", "memory-before.json",
                    "memory-after.json", "result.json", "closed.json"}, "inherited_closed_namespace")
        unknown = old / "segments/segment-0014/jobs/q-000231"
        require(set(original.private_tree(unknown)) == {"claim.json", "memory-before.json"},
                "unknown_attempt_namespace")
    else:
        require(not original.frozen.path_at(root, PREVIOUS_WORK + "/runs/" + recipe["dataset_id"]).exists(),
                "unexpected_previous_run")
    require(not original.frozen.path_at(root, manifest["output"]).exists(), "unexpected_original_run")



def history_observations(root, recipe, manifest):
    """Metadata guards cover both immutable generations without scientific replay."""
    history = read_pinned(root, recipe["reboot_history"])
    result = {}
    for tree in history["sealed_trees"]:
        path = original.frozen.path_at(root, tree["output"])
        for item in [path, *path.rglob("*")]:
            require(not item.is_symlink(), "historical_symlink")
            value = item.stat(follow_symlinks=False)
            require(stat.S_ISREG(value.st_mode) or stat.S_ISDIR(value.st_mode), "historical_type")
            result[tree["output"] + "/" + item.relative_to(path).as_posix()] = (
                value.st_dev, value.st_ino, value.st_mode, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
    return result

def load(root, recipe_ref, admission_ref):
    recipe = read_pinned(root, recipe_ref)
    recipe_scope(recipe)
    check_source(root, recipe)
    manifest = read_pinned(root, recipe["parent_manifest"])
    require(manifest["dataset_id"] == recipe["dataset_id"], "parent_dataset")
    admission = read_pinned(root, admission_ref)
    require(admission["protocol"] == VERSION and
            admission["status"] == "PASS_LOCAL_MPS_REBOOT_RECOVERY_AUTHORIZED" and
            admission["recipe_sha256"] == recipe_ref["sha256"] and
            admission["reboot_history_sha256"] == recipe["reboot_history"]["sha256"] and
            admission["lock_rebinding_sha256"] == recipe["lock_rebinding"]["sha256"] and
            admission["source_sha256"] == recipe["source_sha256"] and
            admission["parent_manifest_sha256"] == recipe["parent_manifest"]["sha256"] and
            admission["context_sha256"] == recipe["context_evidence"]["sha256"] and
            admission["baseline_metric_replay"] is False and admission["tokenizer_repeated"] is False and
            admission["model_scoring_authorized"] is True and
            admission["maximum_queries_per_process"] == SEGMENT_QUERIES,
            "root_recovery_admission")
    require(bool(recipe["accepted_evidence"]), "accepted_recovery_evidence")
    for evidence in recipe["accepted_evidence"]:
        read_pinned(root, evidence)
    # All original source/model/input guards remain. The already independently
    # audited None metrics are not recomputed; context verification is pure.
    dataset, rankings = original.validate(root, operational_manifest(root, recipe, manifest), verify_baseline=False)
    original.context_evidence(root, manifest, recipe["parent_manifest"]["sha256"],
                              recipe["context_evidence"], dataset, rankings)
    history_check(root, recipe, manifest)
    return recipe, manifest, dataset, rankings


def historical_closed_job(path, binding, recipe_sha, segment_claim):
    names = {"claim.json", "memory-before.json", "memory-after.json", "result.json", "closed.json"}
    require(set(original.private_tree(path)) == names, "inherited_job_incomplete")
    raw = {name: original.frozen.file_bytes(path / name) for name in names}
    claim, result, closed = [json.loads(raw[name]) for name in ("claim.json", "result.json", "closed.json")]
    require(claim["protocol"] == PREVIOUS_VERSION and claim["recipe_sha256"] == recipe_sha and
            claim["binding"] == binding and claim["pid"] == segment_claim["pid"] and
            result["claim_sha256"] == original.sha(raw["claim.json"]), "inherited_job_binding")
    require(closed == {"status": "closed", "files": {name: original.sha(raw[name]) for name in names
                                                      if name != "closed.json"}}, "inherited_job_hash")
    for phase in ("before", "after"):
        memory = json.loads(raw["memory-" + phase + ".json"])
        require(memory["within_caps"] is True and memory["caps"] == original.MEMORY and
                memory_within_caps(memory["after_cache_release"]), "inherited_memory_receipt")
    return result["value"]


def inherited_values(root, recipe, dataset, rankings):
    inheritance = recipe["inheritance"]
    if inheritance is None:
        return {}
    history = reboot_history(root, recipe)
    previous_sha = history["refs"]["previous_recipe"]["sha256"]
    corpus = {r["id"]: dataset.model_item(r) for r in dataset.corpus}
    values = {}
    for index in inheritance["closed_indices"]:
        path = original.frozen.path_at(root, inheritance["output"]) / "segments" / f"segment-{index // 16:04d}"
        segment_claim = json.loads(original.frozen.file_bytes(path / "claim.json"))
        require(segment_claim["protocol"] == PREVIOUS_VERSION and
                segment_claim["recipe_sha256"] == previous_sha and segment_claim["segment_index"] == index // 16 and
                segment_claim["indices"] == list(range((index // 16) * 16, (index // 16 + 1) * 16)) and
                segment_claim["score_segment"] == segment_token(previous_sha, index // 16),
                "inherited_segment_binding")
        graph = original.graph(rankings, index)
        binding = original.query_binding(recipe["parent_manifest"]["sha256"], index, dataset, graph, corpus)
        values[index] = historical_closed_job(path / "jobs" / f"q-{index:06d}", binding, previous_sha, segment_claim)
    return values


def unknown_claim(root, recipe, index):
    inheritance = recipe["inheritance"]
    if inheritance is None or index not in inheritance["unknown_indices"]:
        return None
    history = reboot_history(root, recipe)
    path = inheritance["output"] + "/segments/segment-0014/jobs/q-000231/claim.json"
    ref = original.frozen.record(root, path)
    require(ref == history["refs"]["unknown_query_claim"], "unknown_claim_changed")
    return {"historical_claim": ref, "historical_inference_completion": "unknown",
            "historical_process_cessation": "machine_reboot", "historical_child_exit_code": None,
            "new_attempt": True, "cached_score_reuse": False}


def closed_job(path, binding, recipe_sha):
    names = {"claim.json", "memory-before.json", "memory-after.json", "result.json", "closed.json"}
    require(set(original.private_tree(path)) == names, "recovery_job_incomplete")
    raw = {n: original.frozen.file_bytes(path / n) for n in names}
    claim, result, closed = [json.loads(raw[n]) for n in ("claim.json", "result.json", "closed.json")]
    require(claim["protocol"] == VERSION and claim["recipe_sha256"] == recipe_sha and
            claim["binding"] == binding and result["claim_sha256"] == original.sha(raw["claim.json"]),
            "recovery_job_binding")
    require(closed == {"status": "closed", "files": {n: original.sha(raw[n]) for n in names
                                                      if n != "closed.json"}}, "recovery_job_hash")
    return result["value"]


def segment_token(recipe_sha, ordinal):
    return original.sha(original.canonical({"recipe_sha256": recipe_sha, "segment_index": ordinal}))[:32]


def sealed_segment(path, recipe_sha, index, indices):
    tree = original.private_tree(path)
    require(set(p.name for p in path.iterdir()) ==
            {"claim.json", "jobs", "private-library.log", "closed.json"}, "segment_namespace")
    closed = json.loads(original.frozen.file_bytes(path / "closed.json"))
    claim = json.loads(original.frozen.file_bytes(path / "claim.json"))
    require(closed == {"status": "closed", "files": {k: v for k, v in tree.items() if k != "closed.json"}},
            "segment_tree_changed")
    require(claim["protocol"] == VERSION and claim["recipe_sha256"] == recipe_sha and
            claim["segment_index"] == index and claim["indices"] == indices and
            claim["score_segment"] == segment_token(recipe_sha, index), "segment_binding")
    require({p.name for p in (path / "jobs").iterdir()} == {f"q-{i:06d}" for i in indices}, "segment_jobs")


def stage_root(root, recipe, recipe_ref, segments, ordinal, *, final=False, predecessor_ref=None):
    output = original.frozen.path_at(root, recipe["output"])
    if ordinal == 0:
        require(predecessor_ref is None and not output.exists(), "initial_segment_no_retry")
    else:
        require(predecessor_ref is not None, "predecessor_seal_required")
        prior = read_pinned(root, predecessor_ref)
        require(prior["protocol"] == VERSION and prior["recipe_sha256"] == recipe_ref["sha256"] and
                prior["next_segment_index"] == ordinal and prior["tree"] == original.private_tree(output),
                "predecessor_tree")
        wait = read_pinned(root, prior["actual_child_wait"])
        preceding = json.loads(original.frozen.file_bytes(
            output / "segments" / f"segment-{ordinal - 1:04d}" / "claim.json"))
        require(wait["actual_wait"] is True and wait["exit_code"] == 0 and wait["signal"] is None and
                type(wait["pid"]) is int and wait["pid"] > 0 and wait["pid"] == preceding["pid"] and
                wait["protocol"] == VERSION and wait["recipe_sha256"] == recipe_ref["sha256"] and
                wait["segment_index"] == ordinal - 1, "predecessor_actual_wait")
    if not output.exists():
        require(ordinal == 0 and not final and output.parent.is_dir(), "new_recovery_parent")
        original.mkdir_private(output)
        original.fresh(output / "recipe.json", recipe)
        original.mkdir_private(output / "segments")
    require(original.frozen.file_bytes(output / "recipe.json") == original.canonical(recipe) + b"\n",
            "stored_recovery_recipe")
    expected = {f"segment-{i:04d}" for i in range(ordinal)}
    require({p.name for p in (output / "segments").iterdir()} == expected, "segment_sequence_no_retry")
    for i in range(ordinal):
        sealed_segment(output / "segments" / f"segment-{i:04d}", recipe_ref["sha256"], i, segments[i])
    return output


def execute_segment(root, recipe_ref, admission_ref, ordinal, *, predecessor_ref=None):
    recipe, manifest, dataset, rankings = load(root, recipe_ref, admission_ref)
    inherited = recipe["inheritance"]["closed_indices"] if recipe["inheritance"] else []
    segments = partition_indices(len(dataset.queries), inherited)
    require(type(ordinal) is int and 0 <= ordinal < len(segments), "segment_index")
    require(os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "1", "cpu_fallback_environment")
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[name] = "1"
    before = original.observations(root, manifest)
    historical_before = history_observations(root, recipe, manifest)
    refs = [recipe_ref, admission_ref, recipe["parent_manifest"], recipe["context_evidence"]]
    refs += recipe["accepted_evidence"]
    refs += history_refs(root, recipe)
    refs += lock_rebinding_refs(root, recipe, manifest)
    if predecessor_ref is not None:
        refs += [predecessor_ref, read_pinned(root, predecessor_ref)["actual_child_wait"]]
    with original.lock(original.frozen.path_at(root, manifest["gpu_lock"]["path"]),
                       {k: operational_manifest(root, recipe, manifest)["gpu_lock"][k]
                        for k in ("device", "inode")}) as gpu_check:
        output = stage_root(root, recipe, recipe_ref, segments, ordinal, predecessor_ref=predecessor_ref)
        with original.lock(output / "owner.lock", create=True) as owner_check:
            require({p.name for p in output.iterdir()} == {"recipe.json", "segments", "owner.lock"},
                    "recovery_root_namespace")
            stage = output / "segments" / f"segment-{ordinal:04d}"
            original.mkdir_private(stage)
            original.fresh(stage / "claim.json", {"protocol": VERSION, "recipe_sha256": recipe_ref["sha256"],
                "admission_sha256": admission_ref["sha256"], "segment_index": ordinal,
                "indices": segments[ordinal], "score_segment": segment_token(recipe_ref["sha256"], ordinal),
                "pid": os.getpid(), "started_at_utc": original.utc()})
            original.mkdir_private(stage / "jobs")
            expected_tree = original.private_tree(output)
            corpus = {r["id"]: dataset.model_item(r) for r in dataset.corpus}
            active = None
            def guard():
                gpu_check()
                lock_rebinding(root, recipe, manifest)
                owner_check()
                check_source(root, recipe)
                original.sources(root, manifest["source_sha256"])
                original.sdk_sources(root, manifest["sdk_source_sha256"])
                for ref in refs:
                    original.frozen.checked(root, ref)
                require(original.observations(root, manifest) == before, "input_freshness")
                require(history_observations(root, recipe, manifest) == historical_before, "historical_freshness")
                now = original.private_tree(output)
                require(all(now.get(k) == v for k, v in expected_tree.items()), "recovery_receipt_changed")
                prefix = f"segments/segment-{ordinal:04d}/"
                allowed = {prefix + "private-library.log"}
                if active is not None:
                    base = prefix + "jobs/" + active + "/"
                    allowed |= {base} | {base + name for name in
                        ("claim.json", "memory-before.json", "memory-after.json", "result.json", "closed.json")}
                require(set(now) <= set(expected_tree) | allowed, "recovery_namespace_changed")
            backend = None
            tick = time.perf_counter_ns()
            try:
                fd = os.open(stage / "private-library.log", os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                             os.O_NOFOLLOW, 0o600)
                with os.fdopen(fd, "w", encoding="utf-8") as log:
                    directory_fd = os.open(stage, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
                    try:
                        os.fsync(directory_fd)
                    finally:
                        os.close(directory_fd)
                    try:
                        with redirect_stdout(log), redirect_stderr(log):
                            for index in segments[ordinal]:
                                active = f"q-{index:06d}"
                                guard()
                                graph = original.graph(rankings, index)
                                ids = original.union_ids(graph)
                                binding = original.query_binding(recipe["parent_manifest"]["sha256"], index,
                                                                  dataset, graph, corpus)
                                job = stage / "jobs" / active
                                original.mkdir_private(job)
                                claim_sha = original.fresh(job / "claim.json", {"protocol": VERSION,
                                    "recipe_sha256": recipe_ref["sha256"], "binding": binding,
                                    "historical_unknown_attempt": unknown_claim(root, recipe, index),
                                    "pid": os.getpid(), "started_at_utc": original.utc()})
                                if backend is None:
                                    backend = Backend(manifest)
                                def publish(phase, observation):
                                    require(phase in ("before", "after"), "memory_phase")
                                    original.fresh(job / ("memory-" + phase + ".json"), observation)
                                scores, seconds = backend.score(dataset.model_item(dataset.queries[index], text_only=True),
                                                                 [corpus[j] for j in ids], publish)
                                original.scatter(graph, ids, scores, dataset.qrels[dataset.queries[index]["id"]])
                                value = {"union_ids": ids, "scores": scores, "fresh_unique_pairs": len(ids),
                                    "union_scoring_seconds": seconds, "segment": segment_token(recipe_ref["sha256"], ordinal),
                                    "per_cell_latency_measured": False, "measured_at_utc": original.utc()}
                                guard()
                                original.fresh(job / "result.json", {"claim_sha256": claim_sha, "value": value})
                                original.fresh(job / "closed.json", {"status": "closed",
                                                                     "files": original.private_tree(job)})
                                expected_tree.update({k: v for k, v in original.private_tree(output).items()
                                                      if not k.endswith("private-library.log")})
                    finally:
                        log.flush()
                        os.fsync(log.fileno())
            finally:
                if backend is not None:
                    backend.close()
            guard()
            original.validate(root, operational_manifest(root, recipe, manifest), verify_baseline=False)
            guard()
            require(original.observations(root, manifest) == before, "post_input_freshness")
            original.fresh(stage / "closed.json", {"status": "closed", "files": original.private_tree(stage)})
            return {"status": "segment_closed_pending_independent_review", "segment_index": ordinal,
                    "new_queries": len(segments[ordinal]), "segment_wall_seconds": (time.perf_counter_ns()-tick)/1e9}


def finalize(root, recipe_ref, admission_ref, *, predecessor_ref=None):
    """Pure CPU final aggregation; no backend or GPU lock/model is instantiated."""
    recipe, manifest, dataset, rankings = load(root, recipe_ref, admission_ref)
    inherited = inherited_values(root, recipe, dataset, rankings)
    segments = partition_indices(len(dataset.queries), sorted(inherited))
    output = stage_root(root, recipe, recipe_ref, segments, len(segments), final=True, predecessor_ref=predecessor_ref)
    before = original.private_tree(output)
    inputs = original.observations(root, manifest)
    with original.lock(output / "owner.lock") as owner_check:
        require({p.name for p in output.iterdir()} == {"recipe.json", "segments", "owner.lock"},
                "recovery_root_namespace")
        corpus = {r["id"]: dataset.model_item(r) for r in dataset.corpus}
        records = {}
        for i, value in inherited.items():
            records[i] = value
        for ordinal, indices in enumerate(segments):
            for i in indices:
                graph = original.graph(rankings, i)
                binding = original.query_binding(recipe["parent_manifest"]["sha256"], i, dataset, graph, corpus)
                records[i] = closed_job(output / "segments" / f"segment-{ordinal:04d}" / "jobs" / f"q-{i:06d}",
                                        binding, recipe_ref["sha256"])
        require(set(records) == set(range(len(dataset.queries))), "complete_query_population")
        per_query = [original.closed_value(records[i], original.graph(rankings, i),
                    dataset.qrels[dataset.queries[i]["id"]]) for i in range(len(dataset.queries))]
        cells = []
        for j, row in enumerate(per_query[0]):
            counts = [r[j]["candidate_count"] for r in per_query]
            pair = original.pairs()[j // 3]
            cells.append({"key": row["key"], "candidate_k": row["candidate_k"], "channels": pair["channels"],
                "budget_mode": pair["budget_mode"], "reranker": "laya_text", "status": "completed",
                "primary": pair["budget_mode"] == "per_channel" and row["candidate_k"] == 50,
                "query_count": len(records), "metrics": original.matrix.aggregate_metrics([r[j]["metrics"] for r in per_query]),
                "candidate_count": {"min": min(counts), "max": max(counts), "sum": sum(counts)},
                "per_cell_latency_measured": False})
        unknown = recipe["inheritance"]["unknown_indices"] if recipe["inheritance"] else []
        summary = {"protocol": VERSION, "status": "completed", "dataset_id": recipe["dataset_id"],
            "recipe_sha256": recipe_ref["sha256"], "manifest_sha256": recipe["parent_manifest"]["sha256"],
            "context_preflight": recipe["context_evidence"], "query_count": len(records), "gallery_count": len(corpus),
            "laya_cells": 378, "completed_query_cells": len(records)*378, "cells": cells,
            "inherited_closed_queries": len(inherited), "new_completed_queries": len(records)-len(inherited),
            "reattempted_unknown_outcome_queries": len(unknown),
            "previously_unclaimed_queries": len(records)-len(inherited)-len(unknown),
            "historical_unknown_attempt_inference_count": None,
            "reboot_history": recipe["reboot_history"],
            "boot_bound_lock_rebinding": recipe["lock_rebinding"],
            "historical_recovery_parent_exit_code": None, "historical_recovery_active_child_exit_code": None,
            "historical_recovery_reboot_cause": "UNKNOWN",
            "inherited_record_generation": "vidore-laya-bounded-recovery-v2" if inherited else None,
            "fresh_unique_pairs_recorded": sum(r["fresh_unique_pairs"] for r in records.values()),
            "inherited_unique_pairs_recorded": sum(r["fresh_unique_pairs"] for i, r in records.items() if i in inherited),
            "new_unique_pairs_recorded": sum(r["fresh_unique_pairs"] for i, r in records.items() if i not in inherited),
            "new_completed_union_scoring_seconds": sum(r["union_scoring_seconds"] for i, r in records.items() if i not in inherited),
            "timing_scope": "sum of successful new union score calls only; excludes failed historical attempt, loading, cleanup and validation; not online latency",
            "per_cell_latency_measured": False, "cache_reads_are_inference": False,
            "retrieval_inference_count": 0, "retrieval_inference_seconds": None,
            "retrieval_provenance": manifest["retrieval_provenance"], "rerank_device": "mps", "rerank_dtype": "float32",
            "sdk_language_argument": "tr", "language_scope": "inherited SDK lang=tr on EN/FR; not language-matched settings",
            "baseline_report": manifest["baseline_report"], "memory_caps": original.MEMORY,
            "maximum_queries_per_fresh_process": SEGMENT_QUERIES,
            "completed_at_utc": original.utc()}
        original.validate(root, operational_manifest(root, recipe, manifest), verify_baseline=False)
        owner_check()
        check_source(root, recipe)
        original.sources(root, manifest["source_sha256"])
        original.sdk_sources(root, manifest["sdk_source_sha256"])
        history_check(root, recipe, manifest)
        final_refs = [recipe_ref, admission_ref, predecessor_ref, recipe["parent_manifest"],
                      recipe["context_evidence"], read_pinned(root, predecessor_ref)["actual_child_wait"]]
        final_refs += recipe["accepted_evidence"]
        final_refs += history_refs(root, recipe)
        final_refs += lock_rebinding_refs(root, recipe, manifest)
        for ref in final_refs:
            original.frozen.checked(root, ref)
        require(original.private_tree(output) == before and original.observations(root, manifest) == inputs,
                "finalization_freshness")
        original.fresh(output / "summary.json", summary)
        digest = original.fresh(output / "closed.json", {"status": "closed", "recipe_sha256": recipe_ref["sha256"],
                                                        "files": original.private_tree(output)})
        return {"status": "completed_pending_independent_review", "laya_cells": 378,
                "query_count": len(records), "closed_sha256": digest}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--recipe-sha256", required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--admission-sha256", required=True)
    parser.add_argument("--predecessor-seal", type=Path)
    parser.add_argument("--predecessor-seal-sha256")
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--execute-segment", type=int)
    actions.add_argument("--finalize", action="store_true")
    args = parser.parse_args(argv)
    os.umask(0o077)
    refs = []
    for path, digest in ((args.recipe, args.recipe_sha256), (args.admission, args.admission_sha256)):
        require(path.is_relative_to(args.root), "root_relative_input")
        ref = original.frozen.record(args.root, path.relative_to(args.root).as_posix())
        require(ref["sha256"] == digest, "input_sha")
        refs.append(ref)
    require(bool(args.predecessor_seal) == bool(args.predecessor_seal_sha256), "predecessor_arguments")
    prior = None
    if args.predecessor_seal:
        require(args.predecessor_seal.is_relative_to(args.root), "root_relative_predecessor")
        prior = original.frozen.record(args.root, args.predecessor_seal.relative_to(args.root).as_posix())
        require(prior["sha256"] == args.predecessor_seal_sha256, "predecessor_sha")
    result = (finalize(args.root, *refs, predecessor_ref=prior) if args.finalize else
              execute_segment(args.root, *refs, args.execute_segment, predecessor_ref=prior))
    print(json.dumps(result))
    return 0

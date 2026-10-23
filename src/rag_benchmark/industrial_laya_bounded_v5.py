"""Fresh bounded-process Industrial Laya over the reviewed None V5 / S-J V3 contract.

All 283 queries are new scoring work; inherited scores/unknown-attempt exceptions
are forbidden. Existing preparation/context proof and S/J+None closure guards
remain in the separately versioned Industrial parent V5 module. No automatic retry or fallback.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import gc
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

from . import industrial_laya_completion_v5 as original

VERSION = "industrial-laya-bounded-v5"
WORK = "work/industrial-laya-bounded-v5"
SEGMENT_QUERIES = 16
TARGET = "vidore-v3-industrial-en"
POPULATION = (5244, 283)
SOURCES = ("src/rag_benchmark/industrial_laya_bounded_v5.py",
           "scripts/run_industrial_laya_bounded_v5.py")
RECIPE_FIELDS = {"protocol", "dataset_id", "parent_manifest", "context_evidence", "output",
                 "source_sha256", "segment_query_limit", "inheritance", "accepted_evidence"}


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


def recipe_scope(recipe):
    require(set(recipe) == RECIPE_FIELDS and recipe["protocol"] == VERSION, "industrial_bounded_recipe_fields")
    require(recipe["dataset_id"] == TARGET and original.TARGETS == {TARGET: POPULATION} and
            recipe["segment_query_limit"] == SEGMENT_QUERIES, "industrial_bounded_target_or_limit")
    require(recipe["output"] == WORK + "/runs/" + TARGET, "industrial_bounded_output")
    require(recipe["inheritance"] is None, "industrial_all_queries_must_be_fresh")


def history_check(root, recipe, manifest):
    require(recipe["inheritance"] is None, "industrial_all_queries_must_be_fresh")
    paths = (manifest["output"], "work/industrial-laya-completion-v1/runs/" + TARGET,
             "work/industrial-laya-bounded-v1/runs/" + TARGET,
             "work/industrial-laya-completion-v2/runs/" + TARGET,
             "work/industrial-laya-bounded-v2/runs/" + TARGET,
             "work/industrial-laya-completion-v3/runs/" + TARGET,
             "work/industrial-laya-bounded-v3/runs/" + TARGET,
             "work/industrial-laya-completion-v4/runs/" + TARGET,
             "work/industrial-laya-bounded-v4/runs/" + TARGET)
    require(all(not original.frozen.path_at(root, path).exists() for path in paths), "unexpected_original_run")


def history_observations(root, recipe, manifest):
    """A legacy Industrial scoring run must never appear during bounded scoring."""
    history_check(root, recipe, manifest)
    return {}


def load(root, recipe_ref, admission_ref):
    recipe = read_pinned(root, recipe_ref)
    recipe_scope(recipe)
    check_source(root, recipe)
    manifest = read_pinned(root, recipe["parent_manifest"])
    require(manifest["dataset_id"] == recipe["dataset_id"], "parent_dataset")
    admission = read_pinned(root, admission_ref)
    require(admission["protocol"] == VERSION and
            admission["status"] == "PASS_LOCAL_MPS_INDUSTRIAL_BOUNDED_V5_AUTHORIZED" and
            admission["recipe_sha256"] == recipe_ref["sha256"] and
            admission["source_sha256"] == recipe["source_sha256"] and
            admission["parent_manifest_sha256"] == recipe["parent_manifest"]["sha256"] and
            admission["context_sha256"] == recipe["context_evidence"]["sha256"] and
            admission["baseline_metric_replay"] is False and admission["tokenizer_repeated"] is False and
            admission["model_scoring_authorized"] is True and
            admission["maximum_queries_per_process"] == SEGMENT_QUERIES,
            "root_recovery_admission")
    require(bool(recipe["accepted_evidence"]), "accepted_recovery_evidence")
    for evidence in recipe["accepted_evidence"]:
        original.frozen.checked(root, evidence)
    # All original source/model/input guards remain. The already independently
    # audited None metrics are not recomputed; context verification is pure.
    dataset, rankings = original.validate(root, manifest, verify_baseline=False)
    original.context_evidence(root, manifest, recipe["parent_manifest"]["sha256"],
                              recipe["context_evidence"], dataset, rankings)
    history_check(root, recipe, manifest)
    return recipe, manifest, dataset, rankings


def inherited_values(root, recipe, dataset, rankings):
    require(recipe["inheritance"] is None, "industrial_all_queries_must_be_fresh")
    return {}


def unknown_claim(root, recipe, index):
    require(recipe["inheritance"] is None, "industrial_all_queries_must_be_fresh")
    return None


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
    inherited = []
    segments = partition_indices(len(dataset.queries), inherited)
    require(type(ordinal) is int and 0 <= ordinal < len(segments), "segment_index")
    require(os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "1", "cpu_fallback_environment")
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[name] = "1"
    before = original.observations(root, manifest)
    historical_before = history_observations(root, recipe, manifest)
    refs = [recipe_ref, admission_ref, recipe["parent_manifest"], recipe["context_evidence"]]
    refs += recipe["accepted_evidence"]
    if predecessor_ref is not None:
        refs += [predecessor_ref, read_pinned(root, predecessor_ref)["actual_child_wait"]]
    with original.lock(original.frozen.path_at(root, manifest["gpu_lock"]["path"]),
                       {k: manifest["gpu_lock"][k] for k in ("device", "inode")}) as gpu_check:
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
            original.validate(root, manifest, verify_baseline=False)
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
        unknown = []
        summary = {"protocol": VERSION, "status": "completed", "dataset_id": recipe["dataset_id"],
            "recipe_sha256": recipe_ref["sha256"], "manifest_sha256": recipe["parent_manifest"]["sha256"],
            "context_preflight": recipe["context_evidence"], "query_count": len(records), "gallery_count": len(corpus),
            "laya_cells": 378, "completed_query_cells": len(records)*378, "cells": cells,
            "inherited_closed_queries": len(inherited), "new_completed_queries": len(records)-len(inherited),
            "reattempted_unknown_outcome_queries": len(unknown),
            "previously_unclaimed_queries": len(records)-len(inherited)-len(unknown),
            "historical_unknown_attempt_inference_count": None,
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
        original.validate(root, manifest, verify_baseline=False)
        owner_check()
        check_source(root, recipe)
        original.sources(root, manifest["source_sha256"])
        original.sdk_sources(root, manifest["sdk_source_sha256"])
        history_check(root, recipe, manifest)
        final_refs = [recipe_ref, admission_ref, predecessor_ref, recipe["parent_manifest"],
                      recipe["context_evidence"], read_pinned(root, predecessor_ref)["actual_child_wait"]]
        final_refs += recipe["accepted_evidence"]
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

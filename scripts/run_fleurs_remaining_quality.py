"""Offline remaining-304 quality run; shared pointwise scores are not latency evidence."""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
import uuid

import numpy as np

from rag_benchmark import fleurs_remaining_data
from rag_benchmark.models import package_versions, stable_hash, synchronize
from rag_benchmark.multimodal import aggregate_metrics, exact_dot_rankings, fuse_rankings, graded_metrics

ROOT = Path(__file__).resolve().parents[1]
PILOT_SOURCE = ROOT / "scripts/run_fleurs_paraphrase_pilot.py"
PILOT_SHA = "4b79b95cf85703837e0a0562eba0aea33d5f67a40c7da0b56acb0bf532d2a658"
PROTOCOL = "fleurs-remaining-shared-quality-v1"
SEED = "fleurs-remaining-quality-order-v1|2026-10-23|"


def load_pilot_utilities():
    # No global patch: a separately named, exact frozen module supplies pure
    # utilities, local adapters and locks, never its pilot execution function.
    raw = fleurs_remaining_data.pinned_bytes({"path": str(PILOT_SOURCE), "sha256": PILOT_SHA})
    spec = importlib.util.spec_from_file_location("fleurs_remaining_frozen_pilot_utilities", PILOT_SOURCE)
    module = importlib.util.module_from_spec(spec)
    exec(compile(raw, str(PILOT_SOURCE), "exec"), module.__dict__)
    return module


p = load_pilot_utilities()
require = p.require


class QualityStore(p.ReceiptStore):
    def existing(self, key, binding):
        value = super().existing(key, binding)
        if value is not None:
            require(p.read_json(self.location(key) / "claim.json")["protocol"] == PROTOCOL,
                    "quality_claim_protocol")
        return value

    def job(self, key, binding, operation):
        self.validate()
        previous = self.existing(key, binding)
        if previous is not None:
            self.reused_jobs += 1
            return previous, True
        path = self.location(key)
        path.mkdir(mode=0o700)
        claim = p.fresh_json(path / "claim.json", {"protocol": PROTOCOL, "job_key": key,
            "run_identity": self.identity, "binding": binding, "pid": os.getpid(), "started_at_utc": p.utc()})
        value = operation()
        self.validate()
        result = {"claim_sha256": claim, "measured_at_utc": p.utc(), "value": value}
        result_sha = p.fresh_json(path / "result.json", result)
        p.fresh_json(path / "completed.json", {"claim_sha256": claim,
            "result_sha256": result_sha, "status": "completed"})
        self.new_jobs += 1
        return result, False


def pairs():
    return [{"key": f"{''.join(c['channels'])}-{c['budget_mode']}", "channels": c["channels"],
             "budget_mode": c["budget_mode"]} for c in p.cells() if c["reranker"] == "none"]


def candidate_graph(base, doc_ids):
    outputs = []
    for pair in pairs():
        fused, pool = fuse_rankings(base, pair["channels"], budget_mode=pair["budget_mode"], budget=100, rrf_k=60)
        candidates = p.clean_ranking(fused[:50], set(doc_ids), 50)
        outputs.append({**pair, "candidates": candidates, "pool": pool})
    return outputs


def scatter_scores(graph, union_ids, scores, qrels):
    require(len(union_ids) == len(set(union_ids)) and len(scores) == len(union_ids) and
            all(math.isfinite(float(v)) and 0 <= v <= 1 for v in scores), "quality_union_scores")
    require(set(union_ids) == {row["id"] for cell in graph for row in cell["candidates"]},
            "quality_union_membership")
    lookup = dict(zip(union_ids, map(float, scores), strict=True))
    output = []
    for cell in graph:
        ids = [row["id"] for row in cell["candidates"]]
        ordered = sorted(range(len(ids)), key=lambda i: (-lookup[ids[i]], i))
        final = [{"id": ids[i], "score": lookup[ids[i]], "rank": rank}
                 for rank, i in enumerate(ordered, 1)]
        output.append({**cell, "laya_ranking": final,
            "metrics": {"none": graded_metrics(ids, qrels),
                        "laya": graded_metrics([row["id"] for row in final], qrels)}})
    return output


def validate_closed(value, doc_ids, qrels):
    """Reconstruct cheap candidate/scoring algebra, never rerun a closed model call."""
    require(set(value["base_rankings"]) == set("BGENJ"), "quality_base_channels")
    for rows in value["base_rankings"].values():
        require(p.clean_ranking(rows, set(doc_ids)) == rows, "quality_base_ranking")
    graph = candidate_graph(value["base_rankings"], doc_ids)
    union = sorted({row["id"] for cell in graph for row in cell["candidates"]})
    require(value["union_ids"] == union and value["cells"] ==
            scatter_scores(graph, union, value["union_scores"], qrels), "quality_closed_algebra")
    require(value["fresh_query_inference_count"] == 4 and value["query_cache_hits"] == 0 and
            value["document_inference_count"] == 0 and value["fresh_laya_unique_pairs"] == len(union) and
            value["candidate_pair_occurrences"] == sum(len(c["candidates"]) for c in graph),
            "quality_closed_counts")
    require(value["pointwise_score_policy"] == "one_fresh_score_per_query_document_union_then_exact_lookup" and
            value["per_cell_latency_measured"] is False, "quality_closed_scope")


def run_quality(data, store, session, *, backend_factory=p.LocalBackend, sync=None, guard=lambda: None,
                index_factory=p.BM25):
    sync = sync or (lambda: synchronize("mps"))
    dataset = data.dataset
    doc_ids = [row["id"] for row in dataset.corpus]
    require(len(doc_ids) == len(set(doc_ids)), "quality_document_ids")
    corpus = {row["id"]: dataset.model_item(row, text_only=True) for row in dataset.corpus}
    items = [p.query_item(dataset, row) for row in dataset.queries]
    require(len({row["id"] for row in items}) == len(items), "quality_query_ids")
    vectors = data.document_vectors
    require(set(vectors) == set("GENJ"), "quality_vector_channels")
    for array in vectors.values():
        require(array.dtype == np.dtype("float32") and array.ndim == 2 and len(array) == len(doc_ids) and
                np.isfinite(array).all() and np.allclose(np.linalg.norm(array, axis=1), 1, atol=1e-4),
                "quality_document_vectors")
    jobs = sorted(items, key=lambda row: stable_hash(SEED + row["id"]))
    order_sha = stable_hash([q["id"] for q in jobs])
    core = {"protocol": PROTOCOL, "data_config_sha256": data.manifest_sha256,
        "dataset_identity": dataset.identity, "query_digest": stable_hash(items),
        "corpus_digest": stable_hash([corpus[i] for i in doc_ids]), "qrels_digest": stable_hash(dataset.qrels),
        "document_vectors_sha256": {c: hashlib.sha256(a.tobytes()).hexdigest() for c, a in vectors.items()},
        "execution_order_sha256": order_sha, "conditions": pairs(), "candidate_k": 50, "budget": 100, "rrf_k": 60,
        "frozen_adapter_source_sha256": PILOT_SHA,
        "score_policy": "fresh_pointwise_union_once_per_query_no_cross_query_or_old_run_cache"}
    bindings = {q["id"]: {**core, "query_sha256": stable_hash(q)} for q in jobs}
    keys = {"query-" + q["id"] for q in jobs}
    require({path.name for path in store.root.iterdir()} <= keys, "quality_unknown_job")
    pending = [store.existing("query-" + q["id"], bindings[q["id"]]) is None for q in jobs]
    setup, backends, records = [], {}, []
    try:
        if any(pending):
            index, seconds = p.timed(lambda: index_factory([corpus[i] for i in doc_ids]))
            _, warm = p.timed(lambda: index.search(p.WARM_QUERY["text"]))
            setup.append({"channel": "B", "index_build_seconds": seconds, "synthetic_warmup_seconds": warm})
            for channel in ("G", "E", "N", "J", "laya"):
                guard()
                backend, constructor = p.timed(lambda: backend_factory(channel, data))
                backends[channel] = backend
                if channel == "laya":
                    require(getattr(backend.adapter, "pointwise", False) is True and
                            backend.config.get("backend") == "sdk" and backend.config.get("threshold") is None and
                            backend.config.get("batch_size") == 8 and backend.config.get("device") == "mps" and
                            backend.config.get("local_files_only") is True, "quality_laya_pointwise_contract")
                _, cold = p.timed(backend.load, sync)
                guard()
                _, warm = p.timed(backend.warmup, sync)
                guard()
                row = {"channel": channel, "constructor_seconds": constructor, "cold_load_seconds": cold,
                       "synthetic_warmup_seconds": warm, "warmup_invocations": 1,
                       "warmup_pairs": 50 if channel == "laya" else None, "adapter": backend.public_spec(),
                       "adapter_identity": backend.identity}
                setup.append(row)
                p.fresh_json(store.root.parent / "sessions" / session / f"setup-{channel}.json", row)
        for position, item in enumerate(jobs):
            def operation():
                guard()
                sync()
                started = time.perf_counter_ns()
                base, stages = {}, {}
                raw, seconds = p.timed(lambda: index.search(item["text"]))
                base["B"] = p.clean_ranking(raw, set(doc_ids))
                stages["B"] = {"search_seconds": seconds, "query_encode_seconds": 0.0}
                for channel in "GENJ":
                    encoded, encode_seconds = p.timed(lambda: backends[channel].encode(item), sync)
                    require(encoded.shape == (1, vectors[channel].shape[1]) and np.isfinite(encoded).all() and
                            np.allclose(np.linalg.norm(encoded, axis=1), 1, atol=1e-4), "quality_query_vector")
                    raw, search_seconds = p.timed(lambda: exact_dot_rankings(vectors[channel], encoded, doc_ids,
                                                           top_k=100, query_batch_size=1)[0])
                    base[channel] = p.clean_ranking(raw, set(doc_ids))
                    stages[channel] = {"query_encode_seconds": encode_seconds, "search_seconds": search_seconds}
                graph, fusion_seconds = p.timed(lambda: candidate_graph(base, doc_ids))
                union = sorted({row["id"] for cell in graph for row in cell["candidates"]})
                documents = [corpus[i] for i in union]
                scores, score_seconds = p.timed(lambda: backends["laya"].score(item, documents), sync)
                require(len(scores) == len(union) and all(math.isfinite(float(s)) and 0 <= s <= 1 for s in scores),
                        "quality_union_scores")
                sync()
                finished = time.perf_counter_ns()
                guard()
                require(finished >= started, "quality_clock")
                # Gold is accessed only after the fresh scoring graph completed.
                cells = scatter_scores(graph, union, scores, dataset.qrels[item["id"]])
                return {"base_rankings": base, "union_ids": union, "union_scores": list(map(float, scores)),
                    "cells": cells, "candidate_pair_occurrences": sum(len(c["candidates"]) for c in graph),
                    "fresh_laya_unique_pairs": len(union), "fresh_query_inference_count": 4,
                    "query_cache_hits": 0, "document_inference_count": 0, "per_cell_latency_measured": False,
                    "pointwise_score_policy": "one_fresh_score_per_query_document_union_then_exact_lookup",
                    "union_input_sha256": stable_hash({"query": item, "documents": documents}),
                    "laya_adapter_identity": backends["laya"].identity,
                    "laya_config_sha256": stable_hash(backends["laya"].config),
                    "stage_seconds": stages, "all_fusions_seconds": fusion_seconds,
                    "union_laya_seconds": score_seconds,
                    "shared_graph_wall_seconds": (finished - started) / 1e9,
                    "shared_graph_monotonic_ns": {"start": started, "end": finished},
                    "session": session, "execution_position": position}
            result, old = store.job("query-" + item["id"], bindings[item["id"]], operation)
            value = result["value"]
            validate_closed(value, doc_ids, dataset.qrels[item["id"]])
            require(value["union_input_sha256"] == stable_hash({"query": item,
                    "documents": [corpus[i] for i in value["union_ids"]]}), "quality_closed_input")
            records.append((result, old))
    finally:
        for backend in reversed(list(backends.values())):
            backend.unload()
    store.validate()
    require({path.name for path in store.root.iterdir()} == keys, "quality_job_population")
    output = []
    for offset, pair in enumerate(pairs()):
        cell_rows = [r["value"]["cells"][offset] for r, _ in records]
        counts = [len(row["candidates"]) for row in cell_rows]
        for reranker in ("none", "laya"):
            output.append({**pair, "key": pair["key"] + "-" + reranker, "reranker": reranker,
                "query_count": len(records), "primary": pair["budget_mode"] == "total", "candidate_k": 50,
                "metrics": aggregate_metrics([row["metrics"][reranker] for row in cell_rows]),
                "candidate_count": {"min": min(counts), "max": max(counts), "sum": sum(counts)},
                "per_cell_latency_measured": False, "scores_shared_from_query_union": reranker == "laya"})
    unique_pairs = sum(r["value"]["fresh_laya_unique_pairs"] for r, _ in records)
    occurrences = sum(r["value"]["candidate_pair_occurrences"] for r, _ in records)
    times = [r["measured_at_utc"] for r, _ in records]
    return {"protocol": PROTOCOL, "status": "completed", "query_count": len(records), "gallery_count": len(doc_ids),
        "subset_count": 31, "cell_count": 124, "completed_query_cells": len(records) * 124,
        "human_validated": False, "qrels_full_validation": False, "semantic_ground_truth_verified": False,
        "qrels_policy": "source_group_transferred_exploratory_agent_reviewed", "answer_generation_performed": False,
        "document_inference_count": 0, "ASR_inference_count": 0, "query_cache_hits": 0,
        "pilot_queries_reissued": 0, "combined_329_report": False,
        "recorded_query_inference_count": 4 * len(records),
        "new_query_inference_count_this_launch": 4 * sum(not old for _, old in records),
        "new_closed_queries_this_launch": sum(not old for _, old in records),
        "reused_closed_queries_this_launch": sum(old for _, old in records),
        "recorded_fresh_unique_laya_pairs": unique_pairs,
        "new_fresh_unique_laya_pairs_this_launch": sum(r["value"]["fresh_laya_unique_pairs"] for r, old in records if not old),
        "candidate_pair_occurrences": occurrences, "within_query_score_reuses": occurrences - unique_pairs,
        "scoring_policy": "pointwise fresh union scores once per query, reused across all 62 candidate lists; stable input-order ties",
        "pipeline_end_to_end_measured": False, "per_cell_latency_measured": False,
        "recorded_shared_graph_wall": p.distribution([r["value"]["shared_graph_wall_seconds"] for r, _ in records]),
        "timing_scope": "quality workflow shared graph only; no per-cell latency or reconstructed online latency; 25-query pilot timing remains separate",
        "recorded_measurements_time_range": {"first": min(times), "last": max(times)},
        "resume_policy": "closed queries retain original measurements; incomplete claim fails closed; new segment load/warmup separate",
        "execution_order": {"seed": SEED, "sha256": order_sha},
        "topology": "all_models_resident_serial_query_stages; concurrency1; querybatch1; no E/N/J query sharing",
        "setup_this_launch": setup, "cells": output,
        "limitations": ["Transferred source-group labels are exploratory, not complete semantic relevance judgments",
            "BM25 positive scores only; no padding. Laya scores Whisper text, not raw audio",
            "Laya pointwise batching across the union differs from pilot per-list batching; no bitwise equality claim",
            "No combined329 aggregate is produced here; protocol and cohort equivalence require separate review"]}


def source_files():
    return p.source_files() | {str(Path(__file__).resolve()), str(Path(fleurs_remaining_data.__file__).resolve())}


def validate_config(config):
    require(set(config) == {"protocol", "data_config", "source_sha256", "output_directory", "shared_gpu_lock",
                           "memory_guard", "mode"} and config["protocol"] == PROTOCOL and
            config["mode"] == "remaining304_quality_shared_pointwise_union", "quality_config_scope")
    require(set(config["source_sha256"]) == source_files(), "quality_source_population")
    for path, digest in config["source_sha256"].items():
        require(p.sha(path) == digest, "quality_source_changed")
    require(config["source_sha256"][str(PILOT_SOURCE)] == PILOT_SHA, "quality_pilot_pin")
    require(config["memory_guard"] == {"minimum_physical_bytes": 64 * 1024 ** 3,
            "maximum_process_rss_bytes": 64 * 1024 ** 3, "maximum_mps_driver_bytes": 64 * 1024 ** 3},
            "quality_memory_policy")
    require(Path(config["shared_gpu_lock"]).resolve() == ROOT / "work/overnight/gpu.lock", "quality_lock_path")
    require(p.sha(config["data_config"]["path"]) == config["data_config"]["sha256"], "quality_data_config")
    metadata = p.read_json(config["data_config"]["path"])
    for key in ("pilot_manifest", "selection", "drafts", "meaning_review"):
        require(p.sha(metadata[key]["path"]) == metadata[key]["sha256"], "quality_metadata_changed")
    output = Path(config["output_directory"]).absolute()
    require(output.parent == ROOT / "work/fleurs-paraphrase-full-v1" and
            output.name.startswith("quality-run-"), "quality_output_scope")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    session_path = None
    try:
        os.umask(0o077)
        for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
            os.environ[name] = "1"
        require(os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "1", "quality_fallback_forbidden")
        raw = p.safe_bytes(args.config)
        require(hashlib.sha256(raw).hexdigest() == args.config_sha256, "quality_config_changed")
        config = json.loads(raw)
        validate_config(config)
        root = Path(config["output_directory"]).absolute()
        require(not root.exists() or args.resume, "quality_existing_run")
        started = time.perf_counter_ns()
        physical = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip())
        require(physical >= config["memory_guard"]["minimum_physical_bytes"], "quality_physical_memory")
        with p.shared_gpu_lock(Path(config["shared_gpu_lock"])), p.run_lock(root):
            require({x.name for x in root.iterdir()} <= {"owner.lock", "sessions", "claim.json", "jobs"},
                    "quality_run_namespace")
            (root / "sessions").mkdir(mode=0o700, exist_ok=True)
            require(not (root / "sessions").is_symlink(), "quality_session_symlink")
            session = uuid.uuid4().hex
            session_path = root / "sessions" / session
            session_path.mkdir(mode=0o700)
            environment = {"python": platform.python_version(), "machine": platform.machine(),
                "system": platform.system(), "packages": package_versions(
                    ("numpy", "bm25s", "torch", "transformers", "sentence-transformers", "laya"))}
            identity = stable_hash({"config_sha256": args.config_sha256, "environment": environment})
            if (root / "claim.json").exists():
                require(p.read_json(root / "claim.json")["identity"] == identity, "quality_resume_identity")
            else:
                p.fresh_json(root / "claim.json", {"protocol": PROTOCOL, "identity": identity,
                    "config_sha256": args.config_sha256, "pid": os.getpid(), "started_at_utc": p.utc()})
            p.fresh_json(session_path / "claim.json", {"pid": os.getpid(), "identity": identity,
                        "started_at_utc": p.utc(), "resume": args.resume})
            with (session_path / "private-library.log").open("x", encoding="utf-8") as log:
                with redirect_stdout(log), redirect_stderr(log):
                    data, verification = p.timed(lambda: fleurs_remaining_data.load_verified_remaining(
                        config["data_config"]["path"], expected_sha256=config["data_config"]["sha256"]))
                    require(len(data.dataset.corpus) == 743 and
                            [q["id"] for q in data.dataset.queries] == fleurs_remaining_data.IDS,
                            "quality_fixed_population")
                    def verify():
                        require(p.sha(args.config) == args.config_sha256, "quality_config_changed")
                        validate_config(config)
                    store = QualityStore(root / "jobs", identity, verify)
                    memory = p.MemoryObservations()
                    summary = run_quality(data, store, session, guard=memory)
                    memory()
                    post, post_seconds = p.timed(lambda: fleurs_remaining_data.load_verified_remaining(
                        config["data_config"]["path"], expected_sha256=config["data_config"]["sha256"]))
                    require(post.dataset.identity == data.dataset.identity, "quality_post_identity")
                    verify()
            summary.update(environment=environment, data_config_sha256=data.manifest_sha256,
                dataset_identity=data.dataset.identity, config_sha256=args.config_sha256,
                runner_sha256=p.sha(__file__), input_verification_seconds=verification,
                post_input_verification_seconds=post_seconds,
                actual_workflow_this_launch_wall_seconds=(time.perf_counter_ns() - started) / 1e9,
                memory_preflight={"physical_bytes": physical, **config["memory_guard"]},
                memory_observations=memory.summary(), completed_at_utc=p.utc())
            digest = p.fresh_json(root / "public-summary.json", summary)
            p.fresh_json(root / "completed.json", {"identity": identity, "public_summary_sha256": digest,
                "session": session, "completed_at_utc": p.utc(), "intended_exit_code": 0})
        print(json.dumps({"status": "completed", "queries": 304, "cells": 124, "public_summary_sha256": digest}))
        return 0
    except Exception as exc:
        safe = {"status": "failed_requires_review", "exception_class": type(exc).__name__
                if type(exc).__name__ in {"ValueError", "RuntimeError", "OSError", "FileExistsError",
                                         "FileNotFoundError", "BlockingIOError"} else "OtherError"}
        if session_path is not None:
            try:
                digest = p.fresh_bytes(session_path / "private-error-traceback.txt", traceback.format_exc().encode())
                p.fresh_json(session_path / "failure.json", {**safe, "observed_at_utc": p.utc(),
                    "private_traceback_sha256": digest,
                    "memory_observations": memory.summary() if "memory" in locals() else None})
            except Exception:
                pass
        print(json.dumps(safe), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

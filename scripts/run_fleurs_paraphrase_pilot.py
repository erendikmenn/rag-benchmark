"""Offline, fresh-query FLEURS pilot. This script never encodes documents.

Private receipts retain exact query/candidate bindings; public-summary.json is an
allowlisted aggregate. Run only after separately reviewing the pinned data bridge.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import fcntl
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import re
import resource
import subprocess
import stat
import sys
import time
import traceback
import uuid

import numpy as np

from rag_benchmark.models import package_versions, stable_hash, synchronize
from rag_benchmark.multimodal import (
    TextEmbeddingAdapter, aggregate_metrics, exact_dot_rankings, fuse_rankings, graded_metrics,
)
from rag_benchmark.retrieval import turkish_tokens

CHANNELS = "BGENJ"
PROTOCOL = "fleurs-paraphrase-fresh-query-pilot-v1"
WARM_QUERY = {"id": "synthetic-warmup", "text": "Bu yalnız sentetik bir ısınma sorgusudur.", "media": {}}
WARM_DOCS = [{"id": f"warm-{i:02d}", "text": "Bu yalnız sentetik bir ısınma pasajıdır.", "media": {}}
             for i in range(50)]


def require(condition, code):
    if not condition:
        raise ValueError(code)


def utc():
    return datetime.now(timezone.utc).isoformat()


def safe_bytes(path):
    path = Path(path)
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    fd = os.open(path, flags)
    with os.fdopen(fd, "rb") as stream:
        require(stat.S_ISREG(os.fstat(stream.fileno()).st_mode), "nonregular_input")
        return stream.read()


def sha(path):
    return hashlib.sha256(safe_bytes(path)).hexdigest()


def read_json(path):
    return json.loads(safe_bytes(path))


def fresh_bytes(path, data):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    directory_fd = os.open(Path(path).parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return hashlib.sha256(data).hexdigest()


def fresh_json(path, value):
    data = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    return fresh_bytes(path, data)


def exact_members(path, expected):
    require(not path.is_symlink() and path.is_dir(), "receipt_directory_type")
    require({p.name for p in path.iterdir()} == set(expected), "receipt_namespace")
    require(all(p.is_file() and not p.is_symlink() for p in path.iterdir()), "receipt_file_type")


class ReceiptStore:
    """No reissue of claimed work. A missing completion always requires review."""

    def __init__(self, root, identity, validate=lambda: None):
        self.root, self.identity, self.validate = Path(root), identity, validate
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        require(not self.root.is_symlink(), "receipt_root_symlink")
        self.new_jobs = self.reused_jobs = 0

    def location(self, key):
        require(bool(re.fullmatch(r"[A-Za-z0-9_-]+", key)), "job_key")
        return self.root / key

    def existing(self, key, binding):
        path = self.location(key)
        if not path.exists():
            return None
        exact_members(path, ("claim.json", "result.json", "completed.json"))
        claim, result, completed = (read_json(path / name) for name in
                                    ("claim.json", "result.json", "completed.json"))
        require(claim["run_identity"] == self.identity and claim["binding"] == binding
                and claim["job_key"] == key, "closed_job_binding")
        require(completed == {"claim_sha256": sha(path / "claim.json"),
                              "result_sha256": sha(path / "result.json"), "status": "completed"},
                "closed_job_seal")
        require(result["claim_sha256"] == completed["claim_sha256"], "result_claim")
        return result

    def job(self, key, binding, operation):
        self.validate()
        previous = self.existing(key, binding)
        if previous is not None:
            self.reused_jobs += 1
            return previous, True
        path = self.location(key)
        path.mkdir(mode=0o700)
        claim_sha = fresh_json(path / "claim.json", {"protocol": PROTOCOL, "job_key": key,
            "run_identity": self.identity, "binding": binding, "pid": os.getpid(), "started_at_utc": utc()})
        value = operation()
        self.validate()
        result = {"claim_sha256": claim_sha, "measured_at_utc": utc(), "value": value}
        result_sha = fresh_json(path / "result.json", result)
        fresh_json(path / "completed.json", {"claim_sha256": claim_sha,
                   "result_sha256": result_sha, "status": "completed"})
        self.new_jobs += 1
        return result, False


def timed(operation, sync=lambda: None):
    sync()
    start = time.perf_counter_ns()
    result = operation()
    sync()
    elapsed = (time.perf_counter_ns() - start) / 1e9
    require(math.isfinite(elapsed) and elapsed >= 0, "invalid_timer")
    return result, elapsed


def cells():
    return [{"channels": list(subset), "budget_mode": mode, "reranker": reranker,
             "key": f"{''.join(subset)}-{mode}-{reranker}"}
            for width in range(1, 6) for subset in itertools.combinations(CHANNELS, width)
            for mode in ("total", "per_channel") for reranker in ("none", "laya")]


def query_item(dataset, query):
    item = dataset.model_item(query, text_only=True)
    require(item.get("media") == {} and isinstance(item.get("text"), str) and item["text"].strip(),
            "query_must_be_text")
    return {key: item[key] for key in ("id", "text", "media")}


def clean_ranking(rows, ids, maximum=100):
    require(len(rows) <= maximum and len({r["id"] for r in rows}) == len(rows), "ranking_count")
    for row in rows:
        require(row["id"] in ids and math.isfinite(float(row["score"])), "ranking_invalid")
    return [{"id": r["id"], "score": float(r["score"]), "rank": i} for i, r in enumerate(rows, 1)]


class BM25:
    def __init__(self, corpus):
        import bm25s
        self.ids = [row["id"] for row in corpus]
        self.model = bm25s.BM25(k1=1.5, b=0.75, method="lucene", csc_backend="numpy")
        self.model.index([turkish_tokens(row["text"]) for row in corpus], show_progress=False)

    def search(self, text):
        tokens = turkish_tokens(text)
        if not tokens:
            return []
        scores = np.asarray(self.model.get_scores(tokens))
        eligible = np.flatnonzero(scores > 0)
        identifiers = np.asarray(self.ids)
        order = np.lexsort((identifiers[eligible], -scores[eligible]))
        return [{"id": self.ids[i], "score": float(scores[i])} for i in eligible[order[:100]]]


class LocalBackend:
    """Pinned local adapter; no HTTP or document encode route."""

    def __init__(self, channel, pilot):
        self.channel = channel
        if channel == "laya":
            from rag_benchmark.multimodal_generation import MultimodalLayaReranker
            self.adapter = MultimodalLayaReranker({"device": "mps", "dtype": "float32", "backend": "sdk",
                "batch_size": 8, "threshold": None, "local_files_only": True})
            self.config = self.adapter.adapter.config
        else:
            original = pilot.query_adapter_configs[channel]
            config = {**original, "batch_size": 1}
            require(config.get("device") == "mps" and config.get("local_files_only", True) is True,
                    "offline_mps_required")
            if channel in "GE":
                self.adapter = TextEmbeddingAdapter("bge" if channel == "G" else "embeddinggemma", config)
                self.config = self.adapter.embedder.config
            else:
                from rag_benchmark.multimodal_models import make_multimodal_adapter
                self.adapter = make_multimodal_adapter(channel, config)
                self.config = self.adapter.config
        if channel != "laya":
            pilot.validate_query_adapter(channel, self.adapter, batch_size=1)
        self.identity = self.adapter.identity

    def load(self):
        if self.channel in "GE" and self.channel != "laya":
            self.adapter.embedder._load()
        elif self.channel == "laya":
            self.adapter.adapter._load()
        else:
            self.adapter._load()

    def warmup(self):
        if self.channel == "laya":
            self.score(WARM_QUERY, WARM_DOCS)
        else:
            self.encode(WARM_QUERY)

    def encode(self, item):
        return self.adapter.encode([item], role="query")

    def score(self, query, documents):
        return self.adapter.score(query, documents)

    def unload(self):
        self.adapter.unload()
        import torch
        torch.mps.empty_cache()

    def public_spec(self):
        return {key: self.config[key] for key in ("model_id", "revision", "device", "dtype", "dimension",
                "batch_size", "backend", "mode", "max_length", "max_len", "threshold") if key in self.config}


def distribution(values):
    require(bool(values) and all(math.isfinite(v) and v >= 0 for v in values), "latency_values")
    return {"n": len(values), "mean_seconds": float(np.mean(values)),
            "p50_seconds": float(np.percentile(values, 50, method="linear")),
            "p95_seconds": float(np.percentile(values, 95, method="linear")),
            "sum_seconds": float(sum(values)), "percentile_method": "numpy_linear"}


def execution_plan(items):
    """Fixed hash order for pairs and queries, chosen without ranking outcomes."""
    pairs = [c for c in cells() if c["reranker"] == "none"]
    seed = "fleurs-paraphrase-pilot-order-v1|2026-10-22|"
    pairs.sort(key=lambda c: stable_hash(seed + c["key"]))
    return [({"channels": c["channels"], "budget_mode": c["budget_mode"],
              "key": f"{''.join(c['channels'])}-{c['budget_mode']}"}, q)
            for c in pairs for q in sorted(items, key=lambda q: stable_hash(seed + c["key"] + q["id"]))]


def memory_guard(limit_bytes=64 * 1024 ** 3):
    """Separate process/MPS budgets; neither number is added to the other."""
    import torch
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != "darwin":
        rss *= 1024
    driver = torch.mps.driver_allocated_memory()
    require(rss <= limit_bytes and driver <= limit_bytes, "resident_memory_limit")
    return {"process_peak_resident_bytes": rss, "mps_driver_allocated_bytes": driver,
            "mps_current_allocated_bytes": torch.mps.current_allocated_memory(),
            "separate_limit_bytes": limit_bytes}


class MemoryObservations:
    """Observed maxima at guard points, not an unobserved continuous device peak."""
    def __init__(self, probe=memory_guard):
        self.probe, self.count, self.latest, self.maxima = probe, 0, None, {}

    def __call__(self):
        value = self.probe()
        self.count += 1
        self.latest = value
        for key in ("process_peak_resident_bytes", "mps_driver_allocated_bytes", "mps_current_allocated_bytes"):
            self.maxima[key] = max(self.maxima.get(key, 0), value[key])
        return value

    def summary(self):
        return {"observation_count": self.count, "latest": self.latest,
                "observed_maxima": self.maxima,
                "scope": "process lifetime high-water RSS; MPS sampled at load/warmup and query-pair boundaries; not continuous MPS peak"}


def run_pilot(pilot, store, session, *, backend_factory=LocalBackend, sync=None,
              guard=lambda: None, jobs=None):
    """Paired real endpoints: None=t1-t0, Laya=t2-t0. No estimated totals."""
    sync = sync or (lambda: synchronize("mps"))
    dataset = pilot.dataset
    doc_ids = [row["id"] for row in dataset.corpus]
    require(len(doc_ids) == len(set(doc_ids)), "document_ids")
    items = [query_item(dataset, q) for q in dataset.queries]
    corpus = {row["id"]: dataset.model_item(row, text_only=True) for row in dataset.corpus}
    require(all(isinstance(row.get("text"), str) for row in corpus.values()), "candidate_text_required")
    vectors = pilot.document_vectors
    require(set(vectors) == set("GENJ"), "document_vector_channels")
    for array in vectors.values():
        require(array.ndim == 2 and len(array) == len(doc_ids) and np.isfinite(array).all(), "document_vectors_shape")
        require(np.allclose(np.linalg.norm(array, axis=1), 1, atol=1e-4), "document_vector_norm")
    core = {"manifest_sha256": pilot.manifest_sha256, "dataset_identity": dataset.identity,
            "query_digest": stable_hash(items), "corpus_digest": stable_hash([corpus[i] for i in doc_ids]),
            "qrels_digest": stable_hash(dataset.qrels),
            "document_vectors_sha256": {c: hashlib.sha256(a.tobytes()).hexdigest() for c, a in vectors.items()},
            "timing_mode": "paired_actual_end_to_end", "topology": "all_models_resident_serial_query_stages"}
    jobs = execution_plan(items) if jobs is None else jobs
    execution_digest = stable_hash([(c, q["id"]) for c, q in jobs])
    def binding(pair, item):
        return {**core, "pair": pair, "query_sha256": stable_hash(item),
                "candidate_k": 50, "channel_budget": 100, "rrf_k": 60,
                "execution_order_sha256": execution_digest}
    def job_key(pair, item):
        return f"pair-{pair['key']}-{item['id']}"
    expected_keys = {job_key(c, q) for c, q in jobs}
    require(len(expected_keys) == len(jobs), "duplicate_planned_job")
    require({p.name for p in store.root.iterdir()} <= expected_keys, "unplanned_job_namespace")
    pending = [store.existing(job_key(c, q), binding(c, q)) is None for c, q in jobs]
    setup, backends, index = [], {}, None
    values, usage, measured_times = {}, {}, []
    try:
        if any(pending):
            index, index_seconds = timed(lambda: BM25([corpus[i] for i in doc_ids]))
            _, warm_seconds = timed(lambda: index.search(WARM_QUERY["text"]))
            setup.append({"channel": "B", "scope": "fresh_BM25_index_build", "seconds": index_seconds,
                          "synthetic_warmup_seconds": warm_seconds})
            for channel in ("G", "E", "N", "J", "laya"):
                guard()
                backend, constructor = timed(lambda: backend_factory(channel, pilot))
                backends[channel] = backend
                _, cold = timed(backend.load, sync)
                guard()
                _, warm = timed(backend.warmup, sync)
                guard()
                record = {"channel": channel, "model_constructor_seconds": constructor,
                          "cold_model_load_seconds": cold, "synthetic_warmup_seconds": warm,
                          "warmup_invocations": 1, "warmup_pairs": 50 if channel == "laya" else None,
                          "warmup_in_quality_sample": False, "adapter": backend.public_spec()}
                setup.append(record)
                fresh_json(store.root.parent / "sessions" / session / f"setup-{channel}.json", record)
        for position, (pair, item) in enumerate(jobs):
            channels = pair["channels"]
            def operation():
                guard()
                sync()
                t0 = time.perf_counter_ns()
                base, stages = {}, {}
                for channel in channels:
                    if channel == "B":
                        raw, search_seconds = timed(lambda: index.search(item["text"]))
                        encode_seconds = 0.0
                    else:
                        encoded, encode_seconds = timed(lambda: backends[channel].encode(item), sync)
                        require(encoded.shape == (1, vectors[channel].shape[1]) and np.isfinite(encoded).all(),
                                "query_vector_shape")
                        require(np.allclose(np.linalg.norm(encoded, axis=1), 1, atol=1e-4), "query_vector_norm")
                        raw, search_seconds = timed(lambda: exact_dot_rankings(vectors[channel], encoded, doc_ids,
                                                                    top_k=100, query_batch_size=1)[0])
                    base[channel] = clean_ranking(raw, set(doc_ids))
                    stages[channel] = {"query_encode_seconds": encode_seconds, "search_seconds": search_seconds}
                (fused, pool), fusion_seconds = timed(lambda: fuse_rankings(base, channels,
                    budget_mode=pair["budget_mode"], budget=100, rrf_k=60))
                candidates = clean_ranking(fused[:50], set(doc_ids), 50)
                candidate_ids = [row["id"] for row in candidates]
                sync()
                t1 = time.perf_counter_ns()
                # Continue immediately: no disk I/O, qrels, logging or content hashing between endpoints.
                documents = [corpus[identifier] for identifier in candidate_ids]
                scores, score_seconds = timed(lambda: backends["laya"].score(item, documents), sync)
                require(len(scores) == len(candidates) and all(math.isfinite(float(s)) and 0 <= s <= 1
                                                             for s in scores), "laya_score_count_or_range")
                order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))
                final = clean_ranking([{"id": candidate_ids[i], "score": float(scores[i])} for i in order],
                                      set(candidate_ids), 50)
                sync()
                t2 = time.perf_counter_ns()
                guard()
                require(t0 <= t1 <= t2, "endpoint_order")
                return {"none_ranking": candidates, "laya_ranking": final,
                        "candidate_input_sha256": stable_hash({"query": item, "documents": documents}),
                        "candidate_ids": candidate_ids, "candidate_count": len(candidates), "pool": pool,
                        "metrics": {"none": graded_metrics(candidate_ids, dataset.qrels[item["id"]]),
                                    "laya": graded_metrics([r["id"] for r in final], dataset.qrels[item["id"]])},
                        "channel_stages": stages, "fusion_seconds": fusion_seconds,
                        "fresh_full_laya_seconds": score_seconds, "fresh_laya_pairs": len(candidates),
                        "new_query_inference_count": sum(c != "B" for c in channels),
                        "score_cache_pairs": 0, "query_cache_hits": 0, "document_inference_count": 0,
                        "monotonic_endpoints_ns": {"t0": t0, "t1": t1, "t2": t2},
                        "measured_pipeline_end_to_end_seconds": {"none": (t1 - t0) / 1e9, "laya": (t2 - t0) / 1e9},
                        "incremental_laya_wall_seconds": (t2 - t1) / 1e9,
                        "shared_prefix_invocations": 1, "execution_position": position, "session": session}
            result, reused = store.job(job_key(pair, item), binding(pair, item), operation)
            values.setdefault(pair["key"], []).append(result["value"])
            usage.setdefault(pair["key"], []).append(reused)
            measured_times.append(result["measured_at_utc"])
    finally:
        for backend in reversed(list(backends.values())):
            backend.unload()
    store.validate()
    require({p.name for p in store.root.iterdir()} == expected_keys, "closed_job_population")
    output = []
    by_key = {c["key"]: c for c, _ in jobs}
    for key in sorted(values):
        pair, rows, reused = by_key[key], values[key], usage[key]
        require(len(rows) == len(items), "incomplete_pair_condition")
        for reranker in ("none", "laya"):
            output.append({**pair, "key": key + "-" + reranker, "reranker": reranker,
                "status": "completed", "query_count": len(rows),
                "primary": pair["budget_mode"] == "total", "candidate_k": 50,
                "metrics": aggregate_metrics([v["metrics"][reranker] for v in rows]),
                "reused_closed_queries_this_launch": sum(reused), "new_closed_queries_this_launch": len(rows) - sum(reused),
                "candidate_count": {"min": min(v["candidate_count"] for v in rows),
                                    "max": max(v["candidate_count"] for v in rows),
                                    "sum": sum(v["candidate_count"] for v in rows)},
                "recorded_fresh_full_laya_pairs": sum(v["fresh_laya_pairs"] for v in rows) if reranker == "laya" else 0,
                "new_fresh_full_laya_pairs_this_launch": sum(v["fresh_laya_pairs"] for v, old in zip(rows, reused) if not old)
                    if reranker == "laya" else 0,
                "score_cache_pairs": 0, "query_cache_hits": 0,
                "recorded_fusion_shared_prefix": distribution([v["fusion_seconds"] for v in rows]),
                "recorded_fresh_full_laya": distribution([v["fresh_full_laya_seconds"] for v in rows])
                    if reranker == "laya" else None,
                "measured_pipeline_end_to_end": distribution([v["measured_pipeline_end_to_end_seconds"][reranker]
                                                              for v in rows]),
                "channel_stages_shared_prefix": {c: {stage: distribution([v["channel_stages"][c][stage] for v in rows])
                    for stage in ("query_encode_seconds", "search_seconds")} for c in pair["channels"]}})
    return {"protocol": PROTOCOL, "status": "completed", "query_count": len(items), "gallery_count": len(doc_ids),
        "human_validated": False, "qrels_policy": "source_group_transferred_exploratory_agent_reviewed",
        "semantic_ground_truth_verified": False, "answer_generation_performed": False,
        "subset_count": 31, "cell_count": len(output), "completed_query_cells": len(jobs) * 2,
        "paired_query_invocations": len(jobs), "document_inference_count": 0, "ASR_inference_count": 0,
        "new_shared_query_prefixes_this_launch": sum(not old for rows in usage.values() for old in rows),
        "new_query_inference_count_this_launch": sum(v["new_query_inference_count"] for k, rows in values.items()
                                                     for v, old in zip(rows, usage[k]) if not old),
        "latency_scope": "paired real endpoints None=t1-t0,Laya=t2-t0; one shared fresh retrieval prefix; metrics and I/O excluded",
        "topology": "all_models_resident_serial_query_stages; concurrency1; querybatch1; no E/N/J sharing",
        "cache_reads_are_inference": False, "pipeline_end_to_end_measured": True,
        "recorded_measurements_time_range": {"first": min(measured_times), "last": max(measured_times)},
        "resume_policy": "closed measurements retain original timestamps; current invocation new/reused counts separate",
        "execution_order": {"seed": "fleurs-paraphrase-pilot-order-v1|2026-10-22|",
                            "sha256": execution_digest, "policy": "hash_ordered_subset_budget_pairs_and_queries"},
        "warmup_policy": "one_synthetic_query_per_encoder; one_synthetic_50pair_Laya_call; excluded_from_quality",
        "setup_this_launch": setup, "cells": output,
        "limitations": ["25-query exploratory pilot, not population SLA or final independent semantic benchmark",
            "None and Laya share one fresh at-most-50 prefix; these are paired endpoints, not independent requests",
            "Channel and prefix timings displayed in both paired cells must not be summed as separate work",
            "BM25 positive scores only; no padding. Laya reads Whisper text, not raw audio",
            "Document arrays are verified reuse; every non-resumed pair encodes and scores fresh",
            "Original timestamps and full precision per-query measurements remain in private receipts"]}


def source_files():
    from rag_benchmark import fleurs_pilot_data, models, multimodal, multimodal_generation, multimodal_models, retrieval
    return {str(Path(path).resolve()) for path in (str(Path(__file__).resolve()), fleurs_pilot_data.__file__,
            models.__file__, multimodal.__file__, multimodal_models.__file__,
            multimodal_generation.__file__, retrieval.__file__)}


def validate_config(config):
    require(config.get("protocol") == PROTOCOL, "config_protocol")
    require(set(config) == {"protocol", "manifest", "source_sha256", "output_directory", "timing_mode", "topology", "memory_guard", "shared_gpu_lock"}, "config_fields")
    require(config["timing_mode"] == "paired_actual_end_to_end"
            and config["topology"] == "all_models_resident_serial_query_stages", "timing_topology")
    require(config["memory_guard"] == {"minimum_physical_bytes": 64 * 1024 ** 3,
            "maximum_process_rss_bytes": 64 * 1024 ** 3, "maximum_mps_driver_bytes": 64 * 1024 ** 3}, "memory_policy")
    require(Path(config["shared_gpu_lock"]).resolve() == Path(__file__).resolve().parents[1] / "work/overnight/gpu.lock",
            "shared_gpu_lock_path")
    require(set(config["source_sha256"]) == source_files(), "source_population")
    for path, digest in config["source_sha256"].items():
        require(sha(path) == digest, "source_changed")
    require(sha(config["manifest"]["path"]) == config["manifest"]["sha256"], "manifest_changed")


@contextmanager
def run_lock(root):
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    require(not root.is_symlink(), "run_root_symlink")
    fd = os.open(root / "owner.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "r+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


@contextmanager
def shared_gpu_lock(path):
    # Existing shared lock is never truncated, replaced or removed.
    fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "r+") as handle:
        require(stat.S_ISREG(os.fstat(handle.fileno()).st_mode), "shared_gpu_lock_type")
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


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
        require(os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "1", "mps_fallback_forbidden")
        config_bytes = safe_bytes(args.config)
        require(hashlib.sha256(config_bytes).hexdigest() == args.config_sha256, "config_changed")
        config = json.loads(config_bytes)
        validate_config(config)
        root = Path(config["output_directory"]).absolute()
        require(not root.exists() or args.resume, "existing_run_requires_explicit_resume")
        from rag_benchmark.fleurs_pilot_data import load_verified_pilot
        started = time.perf_counter_ns()
        physical = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip())
        require(physical >= config["memory_guard"]["minimum_physical_bytes"], "insufficient_physical_memory")
        with shared_gpu_lock(Path(config["shared_gpu_lock"])), run_lock(root):
            (root / "sessions").mkdir(mode=0o700, exist_ok=True)
            session = uuid.uuid4().hex
            session_path = root / "sessions" / session
            session_path.mkdir(mode=0o700)
            environment = {"python": platform.python_version(), "machine": platform.machine(),
                           "system": platform.system(), "packages": package_versions(
                               ("numpy", "bm25s", "torch", "transformers", "sentence-transformers", "laya"))}
            identity = stable_hash({"config_sha256": args.config_sha256, "environment": environment})
            if (root / "claim.json").exists():
                require(read_json(root / "claim.json")["identity"] == identity, "resume_identity_changed")
                require(not (root / "completed.json").exists(), "run_already_completed")
            else:
                fresh_json(root / "claim.json", {"identity": identity, "config_sha256": args.config_sha256,
                    "pid": os.getpid(), "started_at_utc": utc(), "protocol": PROTOCOL})
            fresh_json(session_path / "claim.json", {"pid": os.getpid(), "identity": identity,
                       "started_at_utc": utc(), "resume": args.resume})
            # Third-party libraries can emit input-bearing errors. Keep them private; the CLI emits enums only.
            with (session_path / "private-library.log").open("x", encoding="utf-8") as log:
                with redirect_stdout(log), redirect_stderr(log):
                    pilot, verification = timed(lambda: load_verified_pilot(Path(config["manifest"]["path"]), expected_sha256=config["manifest"]["sha256"]))
                    require(pilot.manifest_sha256 == config["manifest"]["sha256"], "pilot_manifest_binding")
                    require(len(pilot.dataset.queries) == 25 and len(pilot.dataset.corpus) == 743,
                            "fixed_pilot_population")
                    require([q["id"] for q in pilot.dataset.queries] == [f"pp{i:02d}" for i in range(1, 26)],
                            "fixed_query_ids")
                    def verify():
                        require(sha(args.config) == args.config_sha256, "config_changed")
                        validate_config(config)
                    store = ReceiptStore(root / "jobs", identity, verify)
                    memory = MemoryObservations()
                    summary = run_pilot(pilot, store, session, guard=memory)
                    memory()
                    post, post_seconds = timed(lambda: load_verified_pilot(Path(config["manifest"]["path"]), expected_sha256=config["manifest"]["sha256"]))
                    require(post.dataset.identity == pilot.dataset.identity and post.manifest_sha256 == pilot.manifest_sha256,
                            "post_input_identity")
                    verify()
            summary.update(environment=environment, manifest_sha256=pilot.manifest_sha256,
                config_sha256=args.config_sha256, runner_sha256=sha(__file__),
                input_verification_load_seconds=verification, post_input_verification_seconds=post_seconds,
                memory_preflight={"physical_bytes": physical, **config["memory_guard"]},
                memory_observations=memory.summary(),
                actual_workflow_this_launch_wall_seconds=(time.perf_counter_ns() - started) / 1e9,
                completed_at_utc=utc())
            summary_sha = fresh_json(root / "public-summary.json", summary)
            fresh_json(root / "completed.json", {"identity": identity, "public_summary_sha256": summary_sha,
                "session": session, "completed_at_utc": utc(), "intended_exit_code": 0})
        print(json.dumps({"status": "completed", "cells": 124, "query_cells": 3100,
                          "public_summary_sha256": summary_sha}))
        return 0
    except Exception as exc:
        safe = {"status": "failed_requires_review", "exception_class": type(exc).__name__
                if type(exc).__name__ in {"ValueError", "RuntimeError", "OSError", "FileExistsError",
                                         "FileNotFoundError", "BlockingIOError"} else "OtherError"}
        if session_path is not None:
            try:
                diagnostic_sha = fresh_bytes(session_path / "private-error-traceback.txt",
                                             traceback.format_exc().encode("utf-8"))
                fresh_json(session_path / "failure.json", {**safe, "observed_at_utc": utc(),
                    "private_traceback_sha256": diagnostic_sha,
                    "memory_observations": memory.summary() if "memory" in locals() else None})
            except Exception:
                pass
        print(json.dumps(safe), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

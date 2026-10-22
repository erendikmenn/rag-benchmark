"""Offline Laya-only deltas over six sealed ViDoRe ranking channels.

No retrieval adapter, encoding cache writer, archive extractor or fallback is used.
Every query/document union is scored once; shared scores are quality evidence,
never a measurement of an individual online retrieval/rerank pipeline.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import fcntl
import hashlib
import inspect
import itertools
import json
import math
import os
from pathlib import Path
import platform
import resource
import stat
import sys
import time
import types
import uuid

from . import models, multimodal as matrix, multimodal_generation as generation
from . import vidore_cache_recovery_v2 as frozen

VERSION = "vidore-laya-sealed-union-v1"
CHANNELS = "BGENSJ"
TARGETS = {
    "vidore-v3-energy-fr": (2225, 308),
    "vidore-v3-finance_en-en": (2942, 309),
    "vidore-v3-finance_fr-fr": (2384, 320),
    "vidore-v3-hr-en": (1110, 318),
    "vidore-v3-pharmaceuticals-en": (2313, 364),
    "vidore-v3-physics-fr": (1674, 302),
}
SOURCE_FILES = (
    "src/rag_benchmark/__init__.py", "src/rag_benchmark/models.py",
    "src/rag_benchmark/retrieval.py", "src/rag_benchmark/multimodal.py",
    "src/rag_benchmark/multimodal_generation.py", "src/rag_benchmark/vidore_cache_recovery_v2.py",
    "src/rag_benchmark/vidore_laya_completion_v1.py", "scripts/run_vidore_laya_completion_v1.py",
)
LAYA_CONFIG = {
    "model_id": "convaiinnovations/laya-multilingual",
    "revision": "1720e3e3357cfe1e281542e223f8273b0890ca34",
    "backend": "sdk", "device": "mps", "dtype": "float32", "batch_size": 8,
    "max_len": 8192, "threshold": None, "local_files_only": True,
}
MEMORY = {"maximum_process_rss_bytes": 64 * 1024**3, "maximum_mps_driver_bytes": 64 * 1024**3}
MODEL_REPOSITORY = ".cache/models/models--convaiinnovations--laya-multilingual"
MODEL_SNAPSHOT = MODEL_REPOSITORY + "/snapshots/" + LAYA_CONFIG["revision"]


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return frozen.canonical(value)


def fresh(path, value):
    raw = canonical(value) + b"\n"
    frozen.exclusive_write(Path(path), raw)
    return sha(raw)


def runtime():
    return {"python": platform.python_version(), "system": platform.system(), "machine": platform.machine(),
            "packages": models.package_versions(("numpy", "laya", "torch", "transformers"))}


def adapter_identity():
    """Pure equivalent of the pinned adapter constructors; no instance or load."""
    config = {**LAYA_CONFIG, "instruction": generation.LAYA_RELEVANCE}
    inner = models.stable_hash({"adapter": "laya-v1", "config": config,
        "questions": {"relevant": {"type": "noul", "instructions": generation.LAYA_RELEVANCE}},
        "packages": models.package_versions(("laya", "torch", "transformers"))})
    return models.stable_hash({"adapter": "multimodal-laya-v2", "inner": inner,
                              "empty_text_policy": "score_explicit_empty_passage_with_actual_model"})


def sources(root, pins):
    require(set(pins) == set(SOURCE_FILES), "source_population")
    for relative in SOURCE_FILES:
        require(frozen.record(root, relative)["sha256"] == pins[relative], "source_changed")
    modules = (models, matrix, generation, frozen, sys.modules[__name__])
    for module in modules:
        expected = frozen.path_at(root, "src/" + module.__name__.replace(".", "/") + ".py")
        require(Path(module.__file__).absolute() == expected, "import_origin")
        raw = frozen.file_bytes(expected)
        compiled = compile(raw, str(expected), "exec", dont_inherit=True)
        codes = {}
        definitions = {}
        def definitions_in(nodes, prefix=""):
            for node in nodes:
                if isinstance(node, ast.FunctionDef):
                    definitions[prefix + node.name] = node.args
                elif isinstance(node, ast.ClassDef):
                    definitions_in(node.body, prefix + node.name + ".")
        definitions_in(ast.parse(raw).body)
        def collect(code):
            for child in code.co_consts:
                if isinstance(child, types.CodeType):
                    codes[child.co_qualname] = child
                    collect(child)
        collect(compiled)
        functions = [v for v in vars(module).values() if isinstance(v, types.FunctionType)
                     and v.__module__ == module.__name__]
        for cls in [v for v in vars(module).values() if isinstance(v, type) and v.__module__ == module.__name__]:
            functions.extend(v for v in vars(cls).values() if isinstance(v, types.FunctionType))
        for function in functions:
            if function.__qualname__ in {"Dataset.__init__", "Dataset.__repr__", "Dataset.__eq__"}:
                # dataclass-generated methods are not source definitions used by this protocol.
                continue
            function = inspect.unwrap(function)
            require(function.__code__ == codes.get(function.__qualname__)
                    and function.__globals__ is vars(module), "loaded_code_changed")
            arguments = definitions[function.__qualname__]
            defaults = tuple(ast.literal_eval(x) for x in arguments.defaults) or None
            keywords = {k.arg: ast.literal_eval(v) for k, v in zip(arguments.kwonlyargs, arguments.kw_defaults)
                        if v is not None} or None
            require(function.__defaults__ == defaults and function.__kwdefaults__ == keywords,
                    "loaded_defaults_changed")
    require(matrix.np is models.np and matrix.math is math, "pure_dependency_binding")


def pairs():
    return [{"channels": list(c), "budget_mode": mode,
             "key": "".join(c) + "-" + mode}
            for width in range(1, 7) for c in itertools.combinations(CHANNELS, width)
            for mode in ("per_channel", "total")]


def graph(rankings, query_index):
    result = []
    for pair in pairs():
        fused, pool = matrix.fuse_rankings({c: rankings[c][query_index] for c in pair["channels"]},
                                          pair["channels"], budget_mode=pair["budget_mode"],
                                          budget=100, rrf_k=60)
        result.append({**pair, "fused": fused, "pool": pool})
    return result


def union_ids(candidate_graph):
    return sorted({row["id"] for pair in candidate_graph for row in pair["fused"][:100]})


def scatter(candidate_graph, ids, scores, qrels):
    require(ids == union_ids(candidate_graph) and len(scores) == len(ids), "union_membership")
    require(all(type(x) in (int, float) and math.isfinite(x) and 0 <= x <= 1 for x in scores), "scores")
    lookup = dict(zip(ids, scores, strict=True))
    result = []
    for pair in candidate_graph:
        for k in (20, 50, 100):
            candidates = [row["id"] for row in pair["fused"][:k]]
            ranked = sorted(candidates, key=lambda identifier: (-lookup[identifier], identifier))
            result.append({"key": pair["key"] + "-k" + str(k), "candidate_k": k,
                           "candidate_count": len(candidates),
                           "metrics": matrix.graded_metrics(ranked, qrels)})
    return result


def closed_value(value, candidate_graph, qrels):
    require(set(value) == {"union_ids", "scores", "fresh_unique_pairs", "union_scoring_seconds",
            "segment", "per_cell_latency_measured", "measured_at_utc"}, "closed_value_fields")
    require(type(value["fresh_unique_pairs"]) is int and
            value["fresh_unique_pairs"] == len(union_ids(candidate_graph)) and
            value["per_cell_latency_measured"] is False, "closed_query_scope")
    seconds = value["union_scoring_seconds"]
    require(type(seconds) in (int, float) and math.isfinite(seconds) and seconds >= 0, "closed_timer")
    require(isinstance(value["segment"], str) and len(value["segment"]) == 32 and
            all(c in "0123456789abcdef" for c in value["segment"]), "closed_segment")
    return scatter(candidate_graph, value["union_ids"], value["scores"], qrels)


def baseline_cells(report, query_count):
    expected = {"document__" + "_".join(p["channels"]).lower() + "__none|" + p["budget_mode"]
                for p in pairs()}
    rows = [c for c in report["cells"] if c["variant_id"].endswith("__none")]
    require(len(rows) == 126, "baseline_count")
    result = {}
    for row in rows:
        key = row["variant_id"] + "|" + row["budget_mode"]
        require(key in expected and key not in result and row["status"] == "completed"
                and row["candidate_k"] is None and row["query_count"] == query_count
                and row["completed_queries"] == query_count, "baseline_scope")
        result[key] = row["metrics"]
    return result


def snapshot_files(root):
    directory = frozen.path_at(root, MODEL_SNAPSHOT)
    result = set()
    for parent, directories, files in os.walk(directory, followlinks=False):
        require(all(not (Path(parent) / name).is_symlink() for name in directories), "model_directory_symlink")
        result.update((Path(parent) / name).relative_to(directory).as_posix() for name in files)
    return result


def model_record(root, relative):
    """Bind a normal HF snapshot symlink AND the immutable local blob it targets."""
    rel = Path(relative)
    require(not rel.is_absolute() and rel.parts and rel.as_posix() == relative
            and not any(p in (".", "..") for p in rel.parts), "model_relative_path")
    parent = MODEL_SNAPSHOT if rel.parent == Path(".") else MODEL_SNAPSHOT + "/" + rel.parent.as_posix()
    logical = frozen.path_at(root, parent) / rel.name
    target = os.readlink(logical) if logical.is_symlink() else None
    resolved = logical.resolve(strict=True)
    require(resolved.is_relative_to(root), "model_blob_boundary")
    path = resolved.relative_to(root).as_posix()
    require(path.startswith(MODEL_REPOSITORY + "/blobs/") or path.startswith(MODEL_SNAPSHOT + "/"),
            "model_blob_repository")
    return {"snapshot_path": MODEL_SNAPSHOT + "/" + relative, "symlink_target": target,
            "resolved_file": frozen.record(root, path)}


def prepare_manifest(root, recipe):
    """Seal a fully specified recipe; no rankings, model calls or old writes."""
    required = {"protocol", "dataset_id", "source_sha256", "dataset_identity", "tables", "media",
                "ordered_id_sha256", "rankings", "retrieval_provenance", "baseline_report",
                "accepted_evidence", "model_snapshot", "model_files", "laya_adapter_identity", "gpu_lock"}
    require(set(recipe) == required and recipe["protocol"] == VERSION, "recipe_fields")
    manifest = {**recipe, "root": str(root), "runtime": runtime(), "memory_guard": MEMORY,
                "laya_config": LAYA_CONFIG, "sdk_language_argument": "tr",
                "matrix": {"channels": list(CHANNELS), "budget_modes": ["per_channel", "total"],
                           "candidate_k": [20, 50, 100], "budget": 100, "rrf_k": 60},
                "output": "work/vidore-laya-completion-v1/runs/" + recipe["dataset_id"],
                "recipe_sha256": sha(canonical(recipe) + b"\n")}
    validate(root, manifest)
    return manifest


def validate(root, manifest, *, verify_baseline=True):
    require(manifest.get("protocol") == VERSION and manifest.get("root") == str(root), "manifest_scope")
    name = manifest["dataset_id"]
    require(name in TARGETS and manifest["memory_guard"] == MEMORY, "target_or_memory")
    require(manifest["output"] == "work/vidore-laya-completion-v1/runs/" + name, "output_scope")
    require(manifest["runtime"] == runtime() and manifest["laya_config"] == LAYA_CONFIG, "runtime_or_model")
    require(manifest["laya_adapter_identity"] == adapter_identity() and
            manifest["sdk_language_argument"] == "tr", "adapter_identity_or_language")
    gpu = manifest["gpu_lock"]
    require(set(gpu) == {"path", "device", "inode"} and gpu["path"] == "work/overnight/gpu.lock"
            and type(gpu["device"]) is int and type(gpu["inode"]) is int, "gpu_lock_config")
    lock_stat = frozen.path_at(root, gpu["path"]).stat(follow_symlinks=False)
    require(stat.S_ISREG(lock_stat.st_mode) and (lock_stat.st_dev, lock_stat.st_ino) ==
            (gpu["device"], gpu["inode"]), "gpu_lock_changed")
    require(manifest["matrix"] == {"channels": list(CHANNELS), "budget_modes": ["per_channel", "total"],
            "candidate_k": [20, 50, 100], "budget": 100, "rrf_k": 60}, "matrix_scope")
    sources(root, manifest["source_sha256"])
    metadata, corpus, queries, qrels, exclusions = frozen._dataset(root, manifest)
    require((len(corpus), len(queries)) == TARGETS[name], "population")
    dataset = matrix.Dataset(frozen.path_at(root, "data/multimodal/" + name), metadata,
                             corpus, queries, qrels, manifest["dataset_identity"])
    dataset.require_text()
    require(all(not q.get("media") and isinstance(q.get("text"), str) and q["text"].strip()
                for q in queries), "written_queries")
    require(set(manifest["rankings"]) == set(CHANNELS)
            and set(manifest["retrieval_provenance"]) == set(CHANNELS), "six_sealed_channels")
    rankings = {}
    for channel in CHANNELS:
        ref = manifest["rankings"][channel]
        origin = manifest["retrieval_provenance"][channel]
        require(set(origin) == {"report", "channel_identity", "hardware", "dtype", "stage"}
                and all(isinstance(origin[k], str) and origin[k] for k in ("hardware", "dtype", "stage")),
                "retrieval_origin")
        report = frozen.read_json(root, origin["report"])
        require(report["dataset"]["identity"] == dataset.identity, "origin_dataset")
        identity = report.get("channel_identities", {}).get(channel)
        if identity is None:
            row = report["channels"][channel]
            require(row["status"] == "completed", "origin_channel_incomplete")
            identity = row["identity"]
        require(identity == origin["channel_identity"], "origin_identity")
        payload = frozen.read_json(root, ref)
        require(isinstance(payload.get("rankings"), list) and all(isinstance(rows, list) and all(
            isinstance(row, dict) and set(row) == {"id", "score", "rank"} and
            isinstance(row["id"], str) and type(row["rank"]) is int and
            type(row["score"]) in (int, float) for row in rows) for rows in payload["rankings"]),
            "ranking_scalar_types")
        rankings[channel] = frozen._rankings(payload, identity,
            [r["id"] for r in corpus], [r["id"] for r in queries], exclusions, bm25=channel == "B")
    require(bool(manifest["accepted_evidence"]), "accepted_evidence_missing")
    for ref in manifest["accepted_evidence"]:
        frozen.checked(root, ref)
    require(set(manifest["model_files"]) >= {"model.safetensors", "rl_agent_config.json",
            "encoder/config.json", "tokenizer/tokenizer.json"}, "model_snapshot_files")
    require(manifest["model_snapshot"] == MODEL_SNAPSHOT and
            set(manifest["model_files"]) == snapshot_files(root), "model_snapshot_population")
    for relative, ref in manifest["model_files"].items():
        require(ref == model_record(root, relative), "model_snapshot_binding")
    baseline = frozen.read_json(root, manifest["baseline_report"])
    require(baseline["dataset"]["identity"] == dataset.identity, "baseline_dataset")
    config = baseline["configuration"]
    require(config["query_ids"] == [q["id"] for q in queries] and
            config["requested_channels"] == list(CHANNELS) and config["rrf_k"] == 60 and
            config["channel_budget"] == 100 and config["budget_modes"] == ["per_channel", "total"] and
            config["candidate_grid"] == [20, 50, 100], "baseline_configuration")
    expected = baseline_cells(baseline, len(queries))
    if verify_baseline:
        metrics = {p["key"]: [] for p in pairs()}
        for i, query in enumerate(queries):
            for pair in graph(rankings, i):
                metrics[pair["key"]].append(matrix.graded_metrics(
                    [r["id"] for r in pair["fused"]], qrels[query["id"]]))
        for p in pairs():
            actual = matrix.aggregate_metrics(metrics[p["key"]])
            key = "document__" + "_".join(p["channels"]).lower() + "__none|" + p["budget_mode"]
            require(set(actual) == set(expected[key]) and all(
                abs(actual[k] - expected[key][k]) <= 1e-12 for k in actual), "baseline_disagreement")
    return dataset, rankings


def observations(root, manifest):
    """Content is sealed at admission/completion; every query guards file freshness."""
    refs = list(manifest["tables"].values()) + list(manifest["media"].values())
    refs += list(manifest["rankings"].values())
    refs += [r["resolved_file"] for r in manifest["model_files"].values()]
    refs += [v["report"] for v in manifest["retrieval_provenance"].values()]
    refs += manifest["accepted_evidence"] + [manifest["baseline_report"]]
    result = {}
    for ref in refs:
        path = frozen.path_at(root, ref["path"])
        s = path.stat(follow_symlinks=False)
        require(stat.S_ISREG(s.st_mode), "input_type")
        result[ref["path"]] = (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    require(set(manifest["model_files"]) == snapshot_files(root), "model_snapshot_population")
    for relative, ref in manifest["model_files"].items():
        rel = Path(relative)
        parent = MODEL_SNAPSHOT if rel.parent == Path(".") else MODEL_SNAPSHOT + "/" + rel.parent.as_posix()
        logical = frozen.path_at(root, parent) / rel.name
        s = logical.lstat()
        target = os.readlink(logical) if logical.is_symlink() else None
        require(target == ref["symlink_target"] and
                logical.resolve(strict=True) == frozen.path_at(root, ref["resolved_file"]["path"]),
                "model_link_changed")
        result[ref["snapshot_path"]] = (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns, target)
    return result


@contextmanager
def lock(path, expected=None, *, create=False):
    flags = os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK
    fd = os.open(path, flags | (os.O_CREAT if create else 0), 0o600)
    try:
        current = os.fstat(fd)
        require(stat.S_ISREG(current.st_mode), "lock_type")
        if expected is not None:
            require({"device": current.st_dev, "inode": current.st_ino} == expected, "lock_identity")
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def check():
            s = Path(path).stat(follow_symlinks=False)
            require(stat.S_ISREG(s.st_mode) and (s.st_dev, s.st_ino) ==
                    (current.st_dev, current.st_ino), "lock_replaced")
        check()
        yield check
        check()
    finally:
        os.close(fd)


def private_tree(path):
    result = {}
    require(path.is_dir() and not path.is_symlink() and stat.S_IMODE(path.stat().st_mode) == 0o700,
            "private_directory")
    for entry in path.iterdir():
        require(not entry.is_symlink(), "private_symlink")
        if entry.is_dir():
            for key, value in private_tree(entry).items():
                result[entry.name + "/" + key] = value
        else:
            require(entry.is_file() and stat.S_IMODE(entry.stat().st_mode) == 0o600, "private_file")
            result[entry.name] = sha(frozen.file_bytes(entry))
    return result


class Receipts:
    def __init__(self, directory, manifest_sha):
        self.directory, self.manifest_sha = Path(directory), manifest_sha

    def read(self, key, binding):
        path = self.directory / key
        if not path.exists():
            return None
        require(set(private_tree(path)) == {"claim.json", "result.json", "closed.json"}, "incomplete_claim")
        claim_raw, result_raw, closed_raw = [frozen.file_bytes(path / name)
                                           for name in ("claim.json", "result.json", "closed.json")]
        claim, result, closed = map(json.loads, (claim_raw, result_raw, closed_raw))
        require(claim["protocol"] == VERSION and claim["manifest_sha256"] == self.manifest_sha
                and claim["key"] == key and claim["binding"] == binding, "receipt_binding")
        require(closed == {"status": "closed", "claim_sha256": sha(claim_raw),
                           "result_sha256": sha(result_raw)} and result["claim_sha256"] == sha(claim_raw),
                "receipt_hash")
        return result["value"]

    def run(self, key, binding, operation, guard):
        guard()
        old = self.read(key, binding)
        if old is not None:
            return old, True
        path = self.directory / key
        path.mkdir(mode=0o700)
        claim_sha = fresh(path / "claim.json", {"protocol": VERSION, "manifest_sha256": self.manifest_sha,
                          "key": key, "binding": binding, "started_at_utc": utc(), "pid": os.getpid()})
        value = operation()
        guard()
        result_sha = fresh(path / "result.json", {"claim_sha256": claim_sha, "value": value})
        fresh(path / "closed.json", {"status": "closed", "claim_sha256": claim_sha,
                                    "result_sha256": result_sha})
        return value, False


class Backend:
    def __init__(self, manifest):
        self.adapter = generation.MultimodalLayaReranker(LAYA_CONFIG)
        require(self.adapter.identity == manifest["laya_adapter_identity"]
                and self.adapter.pointwise is True, "laya_identity")
        require(self.adapter.adapter.config == {**LAYA_CONFIG, "instruction": generation.LAYA_RELEVANCE},
                "laya_configuration")
        require(Path(models._snapshot(self.adapter.adapter.config, download=False)).resolve() ==
                frozen.path_at(Path(manifest["root"]), manifest["model_snapshot"]), "model_snapshot")

    def guard(self):
        import torch
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform != "darwin":
            rss *= 1024
        require(rss <= MEMORY["maximum_process_rss_bytes"] and
                torch.mps.driver_allocated_memory() <= MEMORY["maximum_mps_driver_bytes"], "memory_guard")

    def score(self, query, documents):
        self.guard()
        models.synchronize("mps")
        tick = time.perf_counter_ns()
        scores = self.adapter.score(query, documents)
        models.synchronize("mps")
        elapsed = (time.perf_counter_ns() - tick) / 1e9
        self.guard()
        return list(map(float, scores)), elapsed

    def close(self):
        self.adapter.unload()


def execute(root, manifest, manifest_sha, *, resume_seal=None):
    require(sha(canonical(manifest) + b"\n") == manifest_sha, "manifest_bytes_convention")
    dataset, rankings = validate(root, manifest)
    output = frozen.path_at(root, manifest["output"])
    gpu = frozen.path_at(root, "work/overnight/gpu.lock")
    require(manifest["gpu_lock"]["path"] == "work/overnight/gpu.lock", "gpu_lock_path")
    require(os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "1", "cpu_fallback_environment")
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[name] = "1"
    before = observations(root, manifest)
    with lock(gpu, {k: manifest["gpu_lock"][k] for k in ("device", "inode")}) as gpu_check:
        if output.exists():
            require(resume_seal is not None, "resume_seal_required")
        else:
            require(resume_seal is None and output.parent.is_dir(), "new_output_parent")
            output.mkdir(mode=0o700)
        with lock(output / "owner.lock", create=True) as owner_check:
            if resume_seal is not None:
                require(resume_seal["protocol"] == VERSION and resume_seal["manifest_sha256"] == manifest_sha
                        and resume_seal["files"] == private_tree(output), "resume_hashes")
                require(not (output / "closed.json").exists(), "already_closed_do_not_rerun")
            else:
                fresh(output / "manifest.json", manifest)
                fresh(output / "claim.json", {"protocol": VERSION, "manifest_sha256": manifest_sha,
                                              "started_at_utc": utc(), "pid": os.getpid()})
                (output / "jobs").mkdir(mode=0o700)
                (output / "sessions").mkdir(mode=0o700)
            require(frozen.file_bytes(output / "manifest.json") == canonical(manifest) + b"\n", "run_manifest")
            run_claim = json.loads(frozen.file_bytes(output / "claim.json"))
            require(run_claim["protocol"] == VERSION and run_claim["manifest_sha256"] == manifest_sha,
                    "run_claim")
            require({p.name for p in output.iterdir()} ==
                    {"owner.lock", "manifest.json", "claim.json", "jobs", "sessions"}, "run_namespace")
            for old_session in (output / "sessions").iterdir():
                require("claim.json" in private_tree(old_session) and
                        set(private_tree(old_session)) <= {"claim.json", "private-library.log", "closed.json"},
                        "prior_session_namespace")
                require(json.loads(frozen.file_bytes(old_session / "claim.json"))["manifest_sha256"] ==
                        manifest_sha, "prior_session_manifest")
            store = Receipts(output / "jobs", manifest_sha)
            keys = {f"q-{i:06d}" for i in range(len(dataset.queries))}
            require({p.name for p in store.directory.iterdir()} <= keys, "job_namespace")
            bindings = []
            corpus = {r["id"]: dataset.model_item(r) for r in dataset.corpus}
            for i, query in enumerate(dataset.queries):
                candidate_graph = graph(rankings, i)
                ids = union_ids(candidate_graph)
                item = dataset.model_item(query, text_only=True)
                binding = {"manifest_sha256": manifest_sha, "query_index": i,
                           "input_sha256": models.stable_hash({"query": item, "documents": [corpus[j] for j in ids]}),
                           "candidate_graph_sha256": models.stable_hash(candidate_graph)}
                # Reject every incomplete/changed receipt before any model load.
                prior = store.read(f"q-{i:06d}", binding)
                if prior is not None:
                    closed_value(prior, candidate_graph, dataset.qrels[query["id"]])
                bindings.append(binding)
            segment = output / "sessions" / uuid.uuid4().hex
            segment.mkdir(mode=0o700)
            fresh(segment / "claim.json", {"manifest_sha256": manifest_sha, "started_at_utc": utc(),
                                          "pid": os.getpid(), "resume": resume_seal is not None})
            expected_tree = private_tree(output)
            known_jobs = {p.name for p in store.directory.iterdir()}
            known_sessions = {p.name for p in (output / "sessions").iterdir()}
            active_key = None
            def guard():
                gpu_check()
                owner_check()
                sources(root, manifest["source_sha256"])
                require(observations(root, manifest) == before, "input_freshness")
                require({p.name for p in output.iterdir()} ==
                        {"owner.lock", "manifest.json", "claim.json", "jobs", "sessions"}, "run_namespace")
                require({p.name for p in (output / "sessions").iterdir()} == known_sessions,
                        "session_namespace")
                expected_jobs = known_jobs | ({active_key} if active_key and
                                             (store.directory / active_key).exists() else set())
                require({p.name for p in store.directory.iterdir()} == expected_jobs, "job_namespace")
                for key in expected_jobs:
                    entries = set(private_tree(store.directory / key))
                    require(entries == {"claim.json", "result.json", "closed.json"} or
                            (key == active_key and entries == {"claim.json"}), "job_file_namespace")
                require({p.name for p in segment.iterdir()} <= {"claim.json", "private-library.log"},
                        "active_session_namespace")
                current = private_tree(output)
                require(all(current.get(k) == v for k, v in expected_tree.items()), "receipt_mutation")
            backend = None
            started = time.perf_counter_ns()
            records = []
            try:
                log_fd = os.open(segment / "private-library.log", os.O_WRONLY | os.O_CREAT |
                                 os.O_EXCL | os.O_NOFOLLOW, 0o600)
                with os.fdopen(log_fd, "w", encoding="utf-8") as log:
                    with redirect_stdout(log), redirect_stderr(log):
                        for i, query in enumerate(dataset.queries):
                            active_key = f"q-{i:06d}"
                            candidate_graph = graph(rankings, i)
                            ids = union_ids(candidate_graph)
                            item = dataset.model_item(query, text_only=True)
                            def operation():
                                nonlocal backend
                                if backend is None:
                                    backend = Backend(manifest)
                                scores, seconds = backend.score(item, [corpus[j] for j in ids])
                                scatter(candidate_graph, ids, scores, dataset.qrels[query["id"]])
                                return {"union_ids": ids, "scores": scores, "fresh_unique_pairs": len(ids),
                                        "union_scoring_seconds": seconds, "segment": segment.name,
                                        "per_cell_latency_measured": False, "measured_at_utc": utc()}
                            value, reused = store.run(active_key, bindings[i], operation, guard)
                            cells = closed_value(value, candidate_graph, dataset.qrels[query["id"]])
                            records.append((value, reused, cells))
                            known_jobs.add(active_key)
                            expected_tree.update({k: v for k, v in private_tree(output).items()
                                                  if not k.endswith("private-library.log")})
                    log.flush()
                    os.fsync(log.fileno())
            finally:
                if backend is not None:
                    backend.close()
            guard()
            validate(root, manifest)
            require(observations(root, manifest) == before, "post_input_freshness")
            cells = []
            for j, row in enumerate(records[0][2]):
                counts = [r[2][j]["candidate_count"] for r in records]
                pair = pairs()[j // 3]
                cells.append({"key": row["key"], "candidate_k": row["candidate_k"],
                    "channels": pair["channels"], "budget_mode": pair["budget_mode"], "reranker": "laya_text",
                    "primary": pair["budget_mode"] == "per_channel" and row["candidate_k"] == 50,
                    "query_count": len(records), "status": "completed",
                    "metrics": matrix.aggregate_metrics([r[2][j]["metrics"] for r in records]),
                    "candidate_count": {"min": min(counts), "max": max(counts), "sum": sum(counts)},
                    "per_cell_latency_measured": False})
            summary = {"protocol": VERSION, "status": "completed", "dataset_id": manifest["dataset_id"],
                "manifest_sha256": manifest_sha, "query_count": len(records), "gallery_count": len(corpus),
                "laya_cells": 378, "completed_query_cells": len(records) * 378, "cells": cells,
                "fresh_unique_pairs_recorded": sum(r[0]["fresh_unique_pairs"] for r in records),
                "new_unique_pairs_this_segment": sum(r[0]["fresh_unique_pairs"] for r in records if not r[1]),
                "closed_queries_reused_this_segment": sum(r[1] for r in records),
                "retrieval_inference_count": 0, "retrieval_inference_seconds": None,
                "per_cell_latency_measured": False, "cache_reads_are_inference": False,
                "actual_quality_segment_wall_seconds": (time.perf_counter_ns() - started) / 1e9,
                "retrieval_provenance": manifest["retrieval_provenance"], "rerank_device": "mps",
                "rerank_dtype": "float32", "baseline_report": manifest["baseline_report"],
                "timing_scope": "shared union scoring; no online latency or per-cell p95; loading may be included",
                "batching_scope": "fresh query/document union; no bitwise equivalence to historical per-list batches",
                "sdk_language_argument": "tr",
                "language_scope": "inherited source-compatible SDK lang=tr on EN/FR data; not language-matched settings",
                "completed_at_utc": utc()}
            fresh(segment / "closed.json", {"status": "closed", "queries": len(records)})
            fresh(output / "summary.json", summary)
            closed_sha = fresh(output / "closed.json", {"status": "closed", "manifest_sha256": manifest_sha,
                                                        "files": private_tree(output)})
            return {"status": "completed_pending_independent_review", "laya_cells": 378,
                    "query_count": len(records), "closed_sha256": closed_sha}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", help="SHA256 of complete canonical file bytes")
    parser.add_argument("--prepare-from-recipe", type=Path)
    parser.add_argument("--recipe-sha256")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume-seal", type=Path)
    parser.add_argument("--resume-seal-sha256")
    args = parser.parse_args(argv)
    os.umask(0o077)
    if args.prepare_from_recipe:
        require(args.recipe_sha256 and not args.manifest_sha256 and not args.execute
                and not args.resume_seal and not args.resume_seal_sha256, "prepare_arguments")
        raw = frozen.file_bytes(args.prepare_from_recipe)
        require(sha(raw) == args.recipe_sha256, "recipe_sha")
        recipe = json.loads(raw)
        require(raw == canonical(recipe) + b"\n", "canonical_recipe")
        manifest = prepare_manifest(args.root, recipe)
        expected = frozen.path_at(args.root, "work/vidore-laya-completion-v1/manifests/" +
                                  recipe["dataset_id"] + ".json")
        require(args.manifest == expected and expected.parent.is_dir(), "manifest_destination")
        digest = fresh(expected, manifest)
        print(json.dumps({"status": "prepared_not_executed", "manifest_sha256": digest}))
        return 0
    require(args.manifest_sha256 and not args.recipe_sha256, "manifest_arguments")
    raw = frozen.file_bytes(args.manifest)
    require(sha(raw) == args.manifest_sha256, "manifest_sha")
    manifest = json.loads(raw)
    require(raw == canonical(manifest) + b"\n", "canonical_manifest")
    require(bool(args.resume_seal) == bool(args.resume_seal_sha256), "resume_arguments")
    resume = None
    if args.resume_seal:
        raw = frozen.file_bytes(args.resume_seal)
        require(sha(raw) == args.resume_seal_sha256, "resume_seal_sha")
        resume = json.loads(raw)
    if args.execute:
        result = execute(args.root, manifest, args.manifest_sha256, resume_seal=resume)
    else:
        require(resume is None, "resume_requires_execute")
        dataset, _ = validate(args.root, manifest)
        result = {"status": "validated_not_executed", "queries": len(dataset.queries), "laya_cells": 378}
    print(json.dumps(result))
    return 0

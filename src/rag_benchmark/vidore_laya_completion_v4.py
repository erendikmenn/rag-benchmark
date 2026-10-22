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
import dataclasses
import fcntl
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
from pathlib import Path
import platform
import resource
import re
import stat
import sys
import time
import types
import uuid

from . import models, retrieval, multimodal as matrix, multimodal_generation as generation
from . import vidore_cache_recovery_v2 as frozen

VERSION = "vidore-laya-sealed-union-v4"
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
    "src/rag_benchmark/vidore_laya_completion_v4.py", "scripts/run_vidore_laya_completion_v4.py",
)
LAYA_CONFIG = {
    "model_id": "convaiinnovations/laya-multilingual",
    "revision": "1720e3e3357cfe1e281542e223f8273b0890ca34",
    "backend": "sdk", "device": "mps", "dtype": "float32", "batch_size": 8,
    "max_len": 8192, "threshold": None, "local_files_only": True,
}
MEMORY = {"maximum_process_rss_bytes": 64 * 1024**3, "maximum_mps_driver_bytes": 64 * 1024**3}
CONTEXT = {"max_len": 8192, "maximum_pairs_per_query": 600,
           "maximum_state_utf8_bytes": 2 * 1024**2, "maximum_wall_seconds": 1800,
           "maximum_process_rss_bytes": 4 * 1024**3}
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


def mkdir_private(path):
    path.mkdir(mode=0o700)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


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
    modules = (models, retrieval, matrix, generation, frozen, sys.modules[__name__])
    for module in modules:
        expected = frozen.path_at(root, "src/" + module.__name__.replace(".", "/") + ".py")
        require(Path(module.__file__).absolute() == expected, "import_origin")
        raw = frozen.file_bytes(expected)
        compiled = compile(raw, str(expected), "exec", dont_inherit=True)
        codes = {}
        tree = ast.parse(raw)
        def collect(code):
            for child in code.co_consts:
                if isinstance(child, types.CodeType):
                    codes[child.co_qualname] = child
                    collect(child)
        collect(compiled)
        def definitions_in(nodes, owner, prefix=""):
            for node in nodes:
                if isinstance(node, ast.ClassDef):
                    cls = vars(owner).get(node.name)
                    require(type(cls) is type and cls.__module__ == module.__name__ and
                            cls.__qualname__ == prefix + node.name, "loaded_class_binding")
                    expected_bases = (RuntimeError,) if node.name == "UnsupportedConfiguration" else (object,)
                    require(cls.__bases__ == expected_bases, "loaded_class_bases")
                    if module is matrix and node.name == "Dataset":
                        expected_init = dataclasses.make_dataclass("Dataset", [
                            "root", "manifest", "corpus", "queries", "qrels", "identity"]).__init__
                        actual_init = vars(cls).get("__init__")
                        require(isinstance(actual_init, types.FunctionType) and
                                actual_init.__code__ == expected_init.__code__ and
                                actual_init.__defaults__ is None and actual_init.__kwdefaults__ is None and
                                actual_init.__closure__ is None, "dataset_constructor_changed")
                    definitions_in(node.body, cls, prefix + node.name + ".")
                elif isinstance(node, ast.FunctionDef):
                    function = vars(owner).get(node.name)
                    if isinstance(function, property):
                        function = function.fget
                    elif isinstance(function, (staticmethod, classmethod)):
                        function = function.__func__
                    require(isinstance(function, types.FunctionType), "loaded_function_binding")
                    if node.decorator_list and all(isinstance(d, ast.Name) and d.id == "contextmanager"
                                                   for d in node.decorator_list):
                        helper = next(c for c in contextmanager.__code__.co_consts
                                      if isinstance(c, types.CodeType) and c.co_name == "helper")
                        require(function.__code__ == helper and
                                function.__globals__ is vars(sys.modules["contextlib"]) and
                                function.__closure__ is not None and len(function.__closure__) == 1 and
                                function.__closure__[0].cell_contents is function.__wrapped__ and
                                function.__defaults__ is None and function.__kwdefaults__ is None,
                                "contextmanager_wrapper_changed")
                        function = function.__wrapped__
                    require(function.__module__ == module.__name__ and
                            function.__qualname__ == prefix + node.name and
                            function.__code__ == codes.get(prefix + node.name) and
                            function.__globals__ is vars(module), "loaded_code_changed")
                    arguments = node.args
                    defaults = tuple(ast.literal_eval(x) for x in arguments.defaults) or None
                    keywords = {k.arg: ast.literal_eval(v) for k, v in zip(arguments.kwonlyargs, arguments.kw_defaults)
                                if v is not None} or None
                    require(function.__defaults__ == defaults and function.__kwdefaults__ == keywords,
                            "loaded_defaults_changed")
        definitions_in(tree.body, module)
        # Validate imported call targets as well as definitions: discovery never
        # trusts a replacement object's __module__ to decide what to inspect.
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.asname or alias.name.split(".")[0]
                    expected_name = alias.name if alias.asname else alias.name.split(".")[0]
                    require(vars(module).get(name) is sys.modules[expected_name], "import_binding")
            elif isinstance(node, ast.ImportFrom) and node.module != "__future__":
                prefix = module.__package__ + "." if node.level else ""
                origin = prefix + (node.module or "")
                for alias in node.names:
                    expected_value = (sys.modules[origin + alias.name] if node.module is None else
                                      getattr(sys.modules[origin], alias.name))
                    require(vars(module).get(alias.asname or alias.name) is expected_value,
                            "imported_call_binding")
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


def sdk_paths(root):
    directory = Path(importlib.metadata.distribution("laya").locate_file("laya")).absolute()
    require(directory.is_relative_to(root), "sdk_boundary")
    frozen.path_at(root, directory.relative_to(root).as_posix())
    return sorted(p.relative_to(root).as_posix() for p in directory.rglob("*.py"))


def sdk_sources(root, pins):
    require(sorted(pins) == sdk_paths(root), "sdk_source_population")
    for path, digest in pins.items():
        require(frozen.record(root, path)["sha256"] == digest, "sdk_source_changed")


def sdk_bindings(root, pins):
    """Import SDK definitions only; never create Agent or its model."""
    sdk_sources(root, pins)
    import laya
    from laya import agent, common
    require(laya.Agent is agent.Agent, "sdk_agent_binding")
    for name in ("build_sequence", "serialize_state", "encode_text", "render_options", "uses_parallel_layout"):
        require(getattr(agent, name) is getattr(common, name), "sdk_import_binding")
    targets = {agent: ["Agent._to_internal", "Agent._encode_state"],
               common: ["build_sequence", "build_head", "render_options", "_resolve_noul_labels",
                        "serialize_state", "encode_text", "_encode_question_text", "parallel_layout",
                        "uses_parallel_layout"]}
    for module, names in targets.items():
        path = Path(module.__file__).absolute()
        require(path.is_relative_to(root) and path.relative_to(root).as_posix() in pins, "sdk_import_origin")
        code = compile(frozen.file_bytes(path), str(path), "exec", dont_inherit=True)
        codes = {}
        def collect(item):
            for child in item.co_consts:
                if isinstance(child, types.CodeType):
                    codes[child.co_qualname] = child
                    collect(child)
        collect(code)
        for name in names:
            function = module
            for part in name.split("."):
                function = getattr(function, part)
            require(isinstance(function, types.FunctionType) and function.__code__ == codes[name]
                    and function.__globals__ is vars(module), "sdk_loaded_code_changed")
    return agent, common


def model_record(root, relative):
    """Bind every HF/shared-store link and the immutable content-addressed blob."""
    rel = Path(relative)
    require(not rel.is_absolute() and rel.parts and rel.as_posix() == relative
            and not any(p in (".", "..") for p in rel.parts), "model_relative_path")
    parent = MODEL_SNAPSHOT if rel.parent == Path(".") else MODEL_SNAPSHOT + "/" + rel.parent.as_posix()
    logical = frozen.path_at(root, parent) / rel.name
    target = os.readlink(logical) if logical.is_symlink() else None
    current, chain, seen = logical, [], set()
    while current.is_symlink():
        require(current not in seen and len(chain) < 4, "model_link_cycle")
        seen.add(current)
        raw_target = os.readlink(current)
        require(not Path(raw_target).is_absolute(), "model_absolute_link")
        following = Path(os.path.normpath(current.parent / raw_target))
        require(following.is_relative_to(root), "model_blob_boundary")
        path = following.relative_to(root).as_posix()
        require(path.startswith(MODEL_REPOSITORY + "/blobs/") or
                re.fullmatch(r"\.cache/models/blobs/[0-9a-f]{2}/[0-9a-f]{64}", path),
                "model_blob_repository")
        frozen.path_at(root, following.parent.relative_to(root).as_posix())
        chain.append({"path": current.relative_to(root).as_posix(), "target": raw_target,
                      "next_path": path})
        current = following
    path = current.relative_to(root).as_posix()
    ref = frozen.record(root, path)
    if path.startswith(".cache/models/blobs/"):
        # Shared-cache filenames are storage keys, not payload digests. The
        # immutable manifest separately pins the actual content SHA256 and size.
        require(current.parent.name == current.name[:2], "shared_blob_layout")
    return {"snapshot_path": MODEL_SNAPSHOT + "/" + relative, "symlink_target": target,
            "link_chain": chain, "resolved_file": ref}


def prepare_manifest(root, recipe):
    """Seal a fully specified recipe; no rankings, model calls or old writes."""
    required = {"protocol", "dataset_id", "source_sha256", "dataset_identity", "tables", "media",
                "ordered_id_sha256", "rankings", "retrieval_provenance", "baseline_report",
                "accepted_evidence", "model_snapshot", "model_files", "laya_adapter_identity", "gpu_lock",
                "sdk_source_sha256"}
    require(set(recipe) == required and recipe["protocol"] == VERSION, "recipe_fields")
    manifest = {**recipe, "root": str(root), "runtime": runtime(), "memory_guard": MEMORY,
                "laya_config": LAYA_CONFIG, "sdk_language_argument": "tr", "context_protocol": CONTEXT,
                "matrix": {"channels": list(CHANNELS), "budget_modes": ["per_channel", "total"],
                           "candidate_k": [20, 50, 100], "budget": 100, "rrf_k": 60},
                "output": "work/vidore-laya-completion-v4/runs/" + recipe["dataset_id"],
                "recipe_sha256": sha(canonical(recipe) + b"\n")}
    validate(root, manifest)
    return manifest


def validate(root, manifest, *, verify_baseline=True):
    require(manifest.get("protocol") == VERSION and manifest.get("root") == str(root), "manifest_scope")
    name = manifest["dataset_id"]
    require(name in TARGETS and manifest["memory_guard"] == MEMORY, "target_or_memory")
    require(manifest["output"] == "work/vidore-laya-completion-v4/runs/" + name, "output_scope")
    require(manifest["runtime"] == runtime() and manifest["laya_config"] == LAYA_CONFIG, "runtime_or_model")
    require(manifest["context_protocol"] == CONTEXT, "context_protocol")
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
    sdk_sources(root, manifest["sdk_source_sha256"])
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
    refs += [{"path": path} for path in manifest["sdk_source_sha256"]]
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
        for link in ref["link_chain"]:
            path = root / link["path"]
            state = path.lstat()
            require(path.is_symlink() and os.readlink(path) == link["target"] and
                    Path(os.path.normpath(path.parent / link["target"])) == root / link["next_path"],
                    "model_link_chain_changed")
            result[link["path"]] = (state.st_dev, state.st_ino, state.st_size,
                                    state.st_mtime_ns, state.st_ctime_ns, link["target"])
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
            result[entry.name + "/"] = "directory:0700"
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
        mkdir_private(path)
        claim_sha = fresh(path / "claim.json", {"protocol": VERSION, "manifest_sha256": self.manifest_sha,
                          "key": key, "binding": binding, "started_at_utc": utc(), "pid": os.getpid()})
        value = operation()
        guard()
        result_sha = fresh(path / "result.json", {"claim_sha256": claim_sha, "value": value})
        fresh(path / "closed.json", {"status": "closed", "claim_sha256": claim_sha,
                                    "result_sha256": result_sha})
        return value, False


def query_binding(manifest_sha, index, dataset, candidate_graph, corpus):
    ids = union_ids(candidate_graph)
    item = dataset.model_item(dataset.queries[index], text_only=True)
    return {"manifest_sha256": manifest_sha, "query_index": index,
            "input_sha256": models.stable_hash({"query": item, "documents": [corpus[j] for j in ids]}),
            "candidate_graph_sha256": models.stable_hash(candidate_graph)}


class TokenizerProbe:
    """Exact SDK encoding without constructing Agent, weights, tensors or devices."""
    def __init__(self, root, manifest):
        self.agent, self.common = sdk_bindings(root, manifest["sdk_source_sha256"])
        self.cfg = frozen.read_json(root, manifest["model_files"]["rl_agent_config.json"]["resolved_file"])
        token_config = frozen.read_json(root, manifest["model_files"]["tokenizer/tokenizer_config.json"]["resolved_file"])
        require(token_config.get("tokenizer_class") not in (None, "TokenizersBackend") and
                not isinstance(token_config.get("extra_special_tokens"), list), "tokenizer_would_require_write")
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(str(frozen.path_at(root, MODEL_SNAPSHOT + "/tokenizer")),
                                                local_files_only=True, trust_remote_code=False)
        self.parallel_options = self.common.uses_parallel_layout(self.cfg)
        self.internal = {"relevant": self.agent.Agent._to_internal(
            {"type": "noul", "instructions": generation.LAYA_RELEVANCE})}
        q = self.internal["relevant"]
        options = self.common.render_options(q)
        full_options = [self.common.encode_text(self.tok, " " + text.replace(self.tok.mask_token, " "),
                        add_special_tokens=False)["input_ids"] for text in options]
        require(all(len(ids) <= 48 for ids in full_options), "question_option_truncation")
        full_head = self.common.encode_text(self.tok, "noul question: " + q["ins"].replace(self.tok.mask_token, " "),
                                           add_special_tokens=False)["input_ids"]
        require(len(full_head) <= self.cfg.get("head_max_len", 192) - sum(1 + len(ids) for ids in full_options),
                "question_instruction_truncation")

    def check(self, query, document):
        state = {"query": query["text"],
                 "passage": (str(document["title"]) + "\n" if document.get("title") else "") + document["text"]}
        require(len(self.common.serialize_state(state).encode("utf-8")) <= CONTEXT["maximum_state_utf8_bytes"],
                "context_state_byte_bound")
        # The exact unbound method needs only tok/cfg/parallel_options. No Agent
        # instance is created and no _load/_infer/predict path can be reached.
        items = self.agent.Agent._encode_state(self, state, ["relevant"], self.internal,
                                               max_len=8192, head_max_len=self.cfg.get("head_max_len", 192))
        require(len(items) == 1, "context_question_count")
        item = items[0]
        require(not item["state_stats"]["truncated"] and item["state_stats"]["state_tokens_dropped"] == 0
                and item["options"]["options_distinct"] == item["options"]["options"], "context_overflow")
        return len(item["ids"])


def preflight(root, manifest, manifest_sha):
    require(sha(canonical(manifest) + b"\n") == manifest_sha, "manifest_bytes_convention")
    dataset, rankings = validate(root, manifest, verify_baseline=False)
    before = observations(root, manifest)
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[name] = "1"
    output = frozen.path_at(root, "work/vidore-laya-completion-v4/preflight/" + manifest["dataset_id"] + ".json")
    require(output.parent.is_dir() and not output.exists(), "preflight_destination")
    started = time.perf_counter()
    probe = TokenizerProbe(root, manifest)
    corpus = {r["id"]: dataset.model_item(r) for r in dataset.corpus}
    records = []
    for index, query in enumerate(dataset.queries):
        candidate_graph = graph(rankings, index)
        ids = union_ids(candidate_graph)
        require(len(ids) <= CONTEXT["maximum_pairs_per_query"], "context_pair_bound")
        tokens = []
        for identifier in ids:
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
            require(time.perf_counter() - started <= CONTEXT["maximum_wall_seconds"] and
                    rss <= CONTEXT["maximum_process_rss_bytes"], "context_resource_bound")
            tokens.append(probe.check(dataset.model_item(query, text_only=True), corpus[identifier]))
        records.append({"binding": query_binding(manifest_sha, index, dataset, candidate_graph, corpus),
                        "pairs": len(ids), "max_tokens": max(tokens, default=0)})
        sources(root, manifest["source_sha256"])
        sdk_sources(root, manifest["sdk_source_sha256"])
        require(observations(root, manifest) == before, "preflight_input_freshness")
    validate(root, manifest, verify_baseline=False)
    require(observations(root, manifest) == before, "preflight_post_input_freshness")
    evidence = {"protocol": VERSION, "status": "PASS_TOKENIZER_ONLY_CONTEXT_PREFLIGHT",
                "manifest_sha256": manifest_sha, "queries": records, "max_len": 8192,
                "head_max_len": probe.cfg.get("head_max_len", 192), "truncated_pairs": 0,
                "model_instances": 0, "model_forwards": 0, "gpu_calls": 0,
                "pairs": sum(r["pairs"] for r in records), "maximum_tokens": max(r["max_tokens"] for r in records),
                "completed_at_utc": utc()}
    digest = fresh(output, evidence)
    return {"status": evidence["status"], "queries": len(records), "pairs": evidence["pairs"],
            "maximum_tokens": evidence["maximum_tokens"], "preflight_sha256": digest}


def context_evidence(root, manifest, manifest_sha, ref, dataset, rankings):
    require(ref is not None and ref["path"] == "work/vidore-laya-completion-v4/preflight/" +
            manifest["dataset_id"] + ".json", "context_preflight_required")
    evidence = frozen.read_json(root, ref)
    require(evidence["protocol"] == VERSION and evidence["status"] == "PASS_TOKENIZER_ONLY_CONTEXT_PREFLIGHT"
            and evidence["manifest_sha256"] == manifest_sha and evidence["max_len"] == 8192
            and evidence["truncated_pairs"] == 0 and evidence["model_instances"] == 0
            and evidence["model_forwards"] == 0 and evidence["gpu_calls"] == 0
            and len(evidence["queries"]) == len(dataset.queries), "context_preflight_scope")
    corpus = {r["id"]: dataset.model_item(r) for r in dataset.corpus}
    for index, row in enumerate(evidence["queries"]):
        candidate_graph = graph(rankings, index)
        require(row["binding"] == query_binding(manifest_sha, index, dataset, candidate_graph, corpus) and
                row["pairs"] == len(union_ids(candidate_graph)) and type(row["max_tokens"]) is int and
                0 < row["max_tokens"] <= 8192, "context_preflight_binding")
    require(evidence["pairs"] == sum(r["pairs"] for r in evidence["queries"]) and
            evidence["maximum_tokens"] == max(r["max_tokens"] for r in evidence["queries"]),
            "context_preflight_counts")


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


def execute(root, manifest, manifest_sha, *, resume_seal=None, preflight_ref=None):
    require(sha(canonical(manifest) + b"\n") == manifest_sha, "manifest_bytes_convention")
    dataset, rankings = validate(root, manifest)
    context_evidence(root, manifest, manifest_sha, preflight_ref, dataset, rankings)
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
            mkdir_private(output)
        with lock(output / "owner.lock", create=True) as owner_check:
            if resume_seal is not None:
                require(resume_seal["protocol"] == VERSION and resume_seal["manifest_sha256"] == manifest_sha
                        and resume_seal["files"] == private_tree(output), "resume_hashes")
                require(not (output / "closed.json").exists(), "already_closed_do_not_rerun")
            else:
                fresh(output / "manifest.json", manifest)
                fresh(output / "claim.json", {"protocol": VERSION, "manifest_sha256": manifest_sha,
                                              "started_at_utc": utc(), "pid": os.getpid()})
                mkdir_private(output / "jobs")
                mkdir_private(output / "sessions")
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
                binding = query_binding(manifest_sha, i, dataset, candidate_graph, corpus)
                # Reject every incomplete/changed receipt before any model load.
                prior = store.read(f"q-{i:06d}", binding)
                if prior is not None:
                    closed_value(prior, candidate_graph, dataset.qrels[query["id"]])
                bindings.append(binding)
            segment = output / "sessions" / uuid.uuid4().hex
            mkdir_private(segment)
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
                sdk_sources(root, manifest["sdk_source_sha256"])
                frozen.checked(root, preflight_ref)
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
                "context_preflight": preflight_ref,
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
    parser.add_argument("--context-preflight", action="store_true")
    parser.add_argument("--context-evidence", type=Path)
    parser.add_argument("--context-evidence-sha256")
    parser.add_argument("--resume-seal", type=Path)
    parser.add_argument("--resume-seal-sha256")
    args = parser.parse_args(argv)
    os.umask(0o077)
    if args.prepare_from_recipe:
        require(args.recipe_sha256 and not args.manifest_sha256 and not args.execute
                and not args.resume_seal and not args.resume_seal_sha256 and not args.context_preflight
                and not args.context_evidence and not args.context_evidence_sha256, "prepare_arguments")
        raw = frozen.file_bytes(args.prepare_from_recipe)
        require(sha(raw) == args.recipe_sha256, "recipe_sha")
        recipe = json.loads(raw)
        require(raw == canonical(recipe) + b"\n", "canonical_recipe")
        manifest = prepare_manifest(args.root, recipe)
        expected = frozen.path_at(args.root, "work/vidore-laya-completion-v4/manifests/" +
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
    require(not (args.execute and args.context_preflight) and
            bool(args.context_evidence) == bool(args.context_evidence_sha256), "context_arguments")
    if args.execute:
        require(args.context_evidence is not None and args.context_evidence.is_relative_to(args.root),
                "context_preflight_required")
        ref = frozen.record(args.root, args.context_evidence.relative_to(args.root).as_posix())
        require(ref["sha256"] == args.context_evidence_sha256, "context_evidence_sha")
        result = execute(args.root, manifest, args.manifest_sha256, resume_seal=resume, preflight_ref=ref)
    elif args.context_preflight:
        require(resume is None and args.context_evidence is None, "context_preflight_arguments")
        result = preflight(args.root, manifest, args.manifest_sha256)
    else:
        require(resume is None and args.context_evidence is None, "resume_requires_execute")
        dataset, _ = validate(args.root, manifest)
        result = {"status": "validated_not_executed", "queries": len(dataset.queries), "laya_cells": 378}
    print(json.dumps(result))
    return 0

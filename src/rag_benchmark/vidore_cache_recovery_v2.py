"""Strict CPU recovery of two missing ViDoRe rankings into a new private delta.

This module never constructs an adapter or calls the ordinary matrix/cache runner.
Preparation seals current retained bytes; it does not certify their historical
generation. Execution requires the separately reviewed manifest's SHA256.
"""
from __future__ import annotations

import argparse
import ast
import copy
import csv
import hashlib
import importlib
import io
import itertools
import json
import math
import os
from pathlib import Path
import platform
import sqlite3
import stat
import struct
import sys
import time
import types

import numpy as np

VERSION = "vidore-cache-recovery-v2"
CHANNELS = "BGENSJ"
TARGETS = {"vidore-v3-finance_fr-fr": "E", "vidore-v3-hr-en": "J"}
SOURCE_FILES = (
    "src/rag_benchmark/__init__.py", "src/rag_benchmark/models.py",
    "src/rag_benchmark/retrieval.py", "src/rag_benchmark/multimodal.py",
    "src/rag_benchmark/vidore_cache_recovery_v2.py",
    "scripts/run_vidore_cache_recovery_v2.py",
)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def path_at(root, relative):
    """Require the exact relative path, rejecting aliases and symlink components."""
    root = Path(root)
    rel = Path(relative)
    if not root.is_absolute() or rel.is_absolute() or rel.as_posix() != relative:
        raise ValueError("Canonical root-relative path required")
    if not rel.parts or any(part in {".", ".."} for part in rel.parts):
        raise ValueError("Canonical root-relative path required")
    current = root
    for part in rel.parts:
        current /= part
        if current.is_symlink():
            raise ValueError("Symlink inputs or destinations are forbidden")
    if root.resolve() != root:
        raise ValueError("Canonical root required")
    return current


def file_bytes(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("Regular file required")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            value = stream.read()
        after = os.fstat(fd)
        current = Path(path).stat(follow_symlinks=False)
        def fields(item):
            return item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_ctime_ns
        if fields(before) != fields(after) or fields(after) != fields(current):
            raise ValueError("Input changed during read")
        return value
    finally:
        os.close(fd)


def record(root, relative):
    value = file_bytes(path_at(root, relative))
    return {"path": relative, "bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}


def checked(root, entry):
    value = file_bytes(path_at(root, entry["path"]))
    if len(value) != entry["bytes"] or hashlib.sha256(value).hexdigest() != entry["sha256"]:
        raise ValueError("Sealed input bytes changed")
    return value


def read_json(root, entry):
    return json.loads(checked(root, entry))


def _sources(root, pins):
    if set(pins) != set(SOURCE_FILES):
        raise ValueError("Exact source closure required")
    for relative in SOURCE_FILES:
        if record(root, relative)["sha256"] != pins[relative]:
            raise ValueError("Reviewed source changed")


def _pure(root, pins):
    _sources(root, pins)
    # No model module imports torch at module scope. No adapter is constructed.
    sys.dont_write_bytecode = True
    module = importlib.import_module("rag_benchmark.multimodal")
    own_path = path_at(root, "src/rag_benchmark/vidore_cache_recovery_v2.py")
    if Path(__file__).resolve() != own_path:
        raise ValueError("Recovery implementation path differs")
    for name in ("multimodal", "models", "retrieval"):
        imported = sys.modules["rag_benchmark." + name]
        if Path(imported.__file__).resolve() != path_at(root, "src/rag_benchmark/" + name + ".py"):
            raise ValueError("Imported source is outside the pinned source closure")
    def verify_code(source_path, functions, intended_globals):
        source = file_bytes(source_path)
        compiled = compile(source, str(source_path), "exec", dont_inherit=True)
        codes = {code.co_name: code for code in compiled.co_consts if isinstance(code, types.CodeType)}
        definitions = {node.name: node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}
        for name, function in functions.items():
            if (not isinstance(function, types.FunctionType) or name not in codes
                    or function.__globals__ is not intended_globals
                    or function.__code__ != codes[name]
                    or function.__code__.co_filename != str(source_path)):
                raise ValueError("Loaded code differs from the pinned source")
            args = definitions[name].args
            defaults = tuple(ast.literal_eval(value) for value in args.defaults) or None
            keywords = {key.arg: ast.literal_eval(value) for key, value in zip(args.kwonlyargs, args.kw_defaults)
                        if value is not None} or None
            if function.__defaults__ != defaults or function.__kwdefaults__ != keywords:
                raise ValueError("Loaded function defaults differ")
    names = ("exact_dot_rankings", "fuse_rankings", "graded_metrics", "aggregate_metrics", "channel_allocations")
    verify_code(Path(module.__file__), {name: module.__dict__[name] for name in names}, module.__dict__)
    own_names = [node.name for node in ast.parse(file_bytes(own_path)).body if isinstance(node, ast.FunctionDef)]
    verify_code(own_path, {name: globals().get(name) for name in own_names}, globals())
    if module.__dict__["np"] is not np or module.__dict__["math"] is not math:
        raise ValueError("Pure function dependency bindings differ")
    return {name: getattr(module, name) for name in
            ("exact_dot_rankings", "fuse_rankings", "graded_metrics", "aggregate_metrics")}


def _table_rows(value):
    rows = [json.loads(line) for line in value.decode().splitlines() if line.strip()]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError("Nonempty object table required")
    return rows


def _dataset(root, manifest):
    tables = manifest["tables"]
    directory = "data/multimodal/" + manifest["dataset_id"]
    if set(tables) != {"dataset.json", "corpus.jsonl", "queries.jsonl", "qrels.jsonl"}:
        raise ValueError("Exact four frozen tables required")
    for name, entry in tables.items():
        if entry["path"] != directory + "/" + name:
            raise ValueError("Table path identity differs")
    metadata = read_json(root, tables["dataset.json"])
    corpus = _table_rows(checked(root, tables["corpus.jsonl"]))
    queries = _table_rows(checked(root, tables["queries.jsonl"]))
    labels = _table_rows(checked(root, tables["qrels.jsonl"]))
    ids = [[str(row["id"]) for row in rows] for rows in (corpus, queries)]
    if any(len(values) != len(set(values)) for values in ids):
        raise ValueError("Duplicate row identity")
    if [digest(values) for values in ids] != manifest["ordered_id_sha256"]:
        raise ValueError("Frozen row order differs")
    if metadata.get("id") != manifest["dataset_id"] or metadata.get("track") != "document":
        raise ValueError("Wrong dataset identity")
    if metadata.get("full_gallery") is not True:
        raise ValueError("Full gallery required")
    if metadata.get("text_provenance") != {
        "source": "published_markdown", "query_independent": True, "gold_fields_used": False
    }:
        raise ValueError("Original source Markdown provenance required")
    if (metadata.get("counts", {}).get("corpus"), metadata.get("counts", {}).get("queries")) != (
        len(corpus), len(queries)
    ):
        raise ValueError("Dataset counts differ")
    media = {}
    for row in corpus + queries:
        for relative in row.get("media", {}).values():
            expected = directory + "/" + relative
            entry = manifest["media"].get(relative)
            if entry is None or entry["path"] != expected:
                raise ValueError("Missing media provenance")
            checked(root, entry)
            media[relative] = entry["sha256"]
    if set(media) != set(manifest["media"]):
        raise ValueError("Unexpected media provenance")
    identity = digest({"files": {name: entry["sha256"] for name, entry in tables.items()}, "media": media})
    if identity != manifest["dataset_identity"]:
        raise ValueError("Frozen dataset identity differs")
    qrels = {identifier: {} for identifier in ids[1]}
    gallery = set(ids[0])
    for row in labels:
        qid, did, grade = str(row["query_id"]), str(row["corpus_id"]), float(row["relevance"])
        if qid not in qrels or did not in gallery or did in qrels[qid]:
            raise ValueError("Invalid label identity")
        if not math.isfinite(grade) or not 0 <= grade <= 30:
            raise ValueError("Invalid relevance grade")
        qrels[qid][did] = grade
    exclusions = []
    for query in queries:
        values = query.get("metadata", {}).get("exclude_ids", [])
        if not isinstance(values, list):
            raise ValueError("Invalid query exclusions")
        excluded = set(map(str, values))
        positives = {key for key, value in qrels[str(query["id"])].items() if value > 0}
        if not positives or excluded & positives or not excluded <= gallery:
            raise ValueError("Invalid positives or exclusions")
        exclusions.append(excluded)
    return metadata, corpus, queries, qrels, exclusions


def _encoding(report, channel, role):
    spec = report["adapter_specs"]["channels"][channel]
    prompts = {"query": "native-search", "document": "native-document"} if channel == "E" else None
    if spec.get("dimension") != 768:
        raise ValueError("Original 768-dimensional representation required")
    return digest({"engine": report["engine"], "dataset": report["dataset"]["identity"],
                   "revision": report["dataset"]["revision"], "adapter": spec["identity"],
                   "role": role, "dimension": 768, "prompts": prompts})


def _missing_identity(report, channel):
    return digest({"engine": report["engine"], "dataset": report["dataset"]["identity"],
                   "channel": channel, "model": {role: _encoding(report, channel, role)
                                                 for role in ("document", "query")}})


def _vector_paths(report, channel, role, count):
    key = digest({"encoding": _encoding(report, channel, role), "text_only": channel == "E"})
    directory = ".cache/multimodal/vectors/" + key
    names = [f"{start:09d}-{min(start + 128, count):09d}" for start in range(0, count, 128)]
    return directory, [directory + "/" + name + suffix for name in names
                       for suffix in (".npy", ".usage.json")] + [directory + "/diagnostics.json"]


def _vectors(root, manifest, report, role, count):
    directory, paths = _vector_paths(report, manifest["missing_channel"], role, count)
    entries = manifest["vectors"][role]
    if set(entries) != set(paths) or any(entry["path"] != key for key, entry in entries.items()):
        raise ValueError("Vector block path identity differs")
    if {x.name for x in path_at(root, directory).iterdir()} != {Path(x).name for x in paths}:
        raise ValueError("Missing or extra cache block")
    blocks, diagnostics = [], []
    table = manifest["tables"]["corpus.jsonl" if role == "document" else "queries.jsonl"]
    item_ids = [str(row["id"]) for row in _table_rows(checked(root, table))]
    for start in range(0, count, 128):
        stop = min(start + 128, count)
        name = directory + f"/{start:09d}-{stop:09d}"
        raw = checked(root, entries[name + ".npy"])
        payload_size = (stop - start) * 768 * 4
        if not payload_size + 10 <= len(raw) <= payload_size + 4108 or raw[:6] != b"\x93NUMPY":
            raise ValueError("Invalid bounded NPY payload")
        version = tuple(raw[6:8])
        if version not in {(1, 0), (2, 0)}:
            raise ValueError("Unsupported NPY version")
        width = 2 if version == (1, 0) else 4
        length = struct.unpack("<H" if width == 2 else "<I", raw[8:8 + width])[0]
        if length > 4096:
            raise ValueError("Oversized NPY header")
        header = ast.literal_eval(raw[8 + width:8 + width + length].decode("ascii").strip())
        if header != {"descr": "<f4", "fortran_order": False, "shape": (stop - start, 768)}:
            raise ValueError("Wrong vector shape, dtype or order")
        payload = raw[8 + width + length:]
        if len(payload) != payload_size:
            raise ValueError("Trailing or truncated NPY payload")
        value = np.frombuffer(payload, dtype="<f4").reshape(stop - start, 768)
        if not np.isfinite(value).all() or not np.allclose(np.linalg.norm(value, axis=1), 1, atol=.002, rtol=.002):
            raise ValueError("Invalid normalized vectors")
        usage = read_json(root, entries[name + ".usage.json"])
        if usage.get("rows") != stop - start or not isinstance(usage.get("reported"), bool):
            raise ValueError("Diagnostic block row count differs")
        items = usage.get("items")
        if not isinstance(items, list) or (items and [item.get("id") for item in items] != item_ids[start:stop]):
            raise ValueError("Diagnostic item order differs")
        for item in items:
            if (item.get("truncated", False) is not False or item.get("native_token_limit", 8192) != 8192
                    or item.get("retained_tokens", 0) < item.get("original_tokens", 0)):
                raise ValueError("Original full-content generation limits differ")
        if usage.get("truncated_text_items", 0) != 0 or usage.get("native_text_limit", 8192) != 8192:
            raise ValueError("Original truncation policy differs")
        diagnostics.append(usage)
        blocks.append(value)
    aggregate = read_json(root, entries[directory + "/diagnostics.json"])
    reported = sum(item["rows"] for item in diagnostics if item["reported"])
    if (aggregate.get("total_rows") != count or aggregate.get("reported_rows") != reported
            or aggregate.get("missing_rows") != count - reported
            or aggregate.get("items") != [item for block in diagnostics for item in block["items"]]):
        raise ValueError("Aggregate diagnostic provenance differs")
    return np.concatenate(blocks)


def _ranking_path(identity):
    return ".cache/multimodal/rankings/" + digest({"identity": identity, "top_k": 100}) + "/rankings.json"


def _rankings(payload, identity, corpus_ids, query_ids, exclusions, *, bm25=False):
    if payload.get("identity") != identity or payload.get("query_ids") != query_ids:
        raise ValueError("Ranking identity or query order differs")
    rankings = payload.get("rankings")
    if not isinstance(rankings, list) or len(rankings) != len(query_ids):
        raise ValueError("Ranking row count differs")
    gallery = set(corpus_ids)
    for index, rows in enumerate(rankings):
        ids = [row["id"] for row in rows]
        if len(ids) != len(set(ids)) or len(ids) > 100 or not set(ids) <= gallery:
            raise ValueError("Invalid gallery ranking")
        if not bm25 and len(ids) != min(100, len(gallery - exclusions[index])):
            raise ValueError("Truncated dense ranking")
        if bm25 and any(float(row["score"]) <= 0 for row in rows):
            raise ValueError("BM25 positive-score policy differs")
        if set(ids) & exclusions[index] or any(not math.isfinite(float(row["score"])) for row in rows):
            raise ValueError("Invalid ranked scores or exclusions")
        if [row["rank"] for row in rows] != list(range(1, len(rows) + 1)):
            raise ValueError("Invalid base ranking positions")
        if rows != sorted(rows, key=lambda row: (-float(row["score"]), row["id"])):
            raise ValueError("Base ranking order differs")
    return rankings


def prepare_manifest(root, recipe):
    """Read and seal explicitly named inputs; never recover rankings or write files."""
    root = Path(root)
    if recipe.get("version") != VERSION or recipe.get("dataset_id") not in TARGETS:
        raise ValueError("Unknown recovery recipe")
    name = recipe["dataset_id"]
    channel = TARGETS[name]
    if recipe.get("missing_channel") != channel:
        raise ValueError("Wrong missing channel")
    _sources(root, recipe["source_sha256"])
    for entry in recipe["expected_records"]:
        checked(root, entry)
    directory = "data/multimodal/" + name
    tables = {key: record(root, directory + "/" + key)
              for key in ("dataset.json", "corpus.jsonl", "queries.jsonl", "qrels.jsonl")}
    corpus, queries = [_table_rows(checked(root, tables[key])) for key in ("corpus.jsonl", "queries.jsonl")]
    main_path = "runs/multimodal/" + name + "-all-retrieval/report.json"
    report_record = record(root, main_path)
    report = read_json(root, report_record)
    if digest(report["adapter_specs"]) != recipe["original_adapter_specs_sha256"]:
        raise ValueError("Original adapter metadata changed")
    media = {relative: record(root, directory + "/" + relative)
             for row in corpus + queries for relative in row.get("media", {}).values()}
    identities = {key: value["identity"] for key, value in report["channels"].items()
                  if value.get("status") == "completed"}
    if set(identities) != set(CHANNELS) - {channel}:
        raise ValueError("Unexpected retained channel coverage")
    identities[channel] = _missing_identity(report, channel)
    if identities != recipe["channel_identities"]:
        raise ValueError("Reviewed original channel identities differ")
    vectors = {}
    for role, count in (("document", len(corpus)), ("query", len(queries))):
        _, paths = _vector_paths(report, channel, role, count)
        vectors[role] = {path: record(root, path) for path in paths}
    rankings = {key: record(root, _ranking_path(identity)) for key, identity in identities.items() if key != channel}
    if path_at(root, _ranking_path(identities[channel])).exists():
        raise ValueError("Target ranking already exists; original inventory is stale")
    prior = []
    for suffix in ("all-retrieval", "bm25"):
        base = "runs/multimodal/" + name + "-" + suffix
        db = base + "/progress.sqlite3"
        wal = path_at(root, db + "-wal")
        if wal.exists() and wal.stat().st_size:
            raise ValueError("Prior SQLite is not closed")
        prior.append({"report": record(root, base + "/report.json"), "sqlite": record(root, db),
                      "wal": record(root, db + "-wal") if wal.exists() else None})
    result = {"version": VERSION, "root": str(root), "dataset_id": name, "missing_channel": channel,
              "recipe_sha256": digest(recipe), "source_sha256": recipe["source_sha256"],
              "expected_records": recipe["expected_records"], "cpu_runtime": runtime_identity(),
              "dataset_identity": report["dataset"]["identity"], "tables": tables, "media": media,
              "ordered_id_sha256": [digest([str(row["id"]) for row in rows]) for rows in (corpus, queries)],
              "main_report": report_record, "original_adapter_specs": report["adapter_specs"],
              "channel_identities": identities, "vectors": vectors, "rankings": rankings, "prior": prior,
              "additional_metadata": [record(root, path) for path in recipe["additional_metadata"]],
              "output": "work/vidore-cache-recovery-v2/delta/" + name,
              "provenance_scope": "current_retained_bytes_only_not_historical_generation_seal",
              "execution_authority": "existing_user_authorization_after_speech_publication_and_root_review"}
    validate_inputs(root, result)
    return result


def validate_inputs(root, manifest):
    if manifest.get("version") != VERSION or manifest.get("root") != str(root):
        raise ValueError("Manifest root or version differs")
    name = manifest["dataset_id"]
    if TARGETS.get(name) != manifest.get("missing_channel"):
        raise ValueError("Manifest target differs")
    if manifest["output"] != "work/vidore-cache-recovery-v2/delta/" + name:
        raise ValueError("Exclusive delta namespace required")
    _sources(root, manifest["source_sha256"])
    if manifest["cpu_runtime"] != runtime_identity():
        raise ValueError("Reviewed CPU runtime differs")
    for entry in manifest["expected_records"]:
        checked(root, entry)
    _, corpus, queries, qrels, exclusions = _dataset(root, manifest)
    report = read_json(root, manifest["main_report"])
    expected = "runs/multimodal/" + name + "-all-retrieval/report.json"
    if manifest["main_report"]["path"] != expected:
        raise ValueError("Main report path identity differs")
    query_ids = [str(row["id"]) for row in queries]
    corpus_ids = [str(row["id"]) for row in corpus]
    config = report["configuration"]
    if (report.get("engine") != "multimodal-v1" or report["dataset"]["identity"] != manifest["dataset_identity"]
            or report["adapter_specs"] != manifest["original_adapter_specs"]
            or config != {"engine": "multimodal-v1", "dataset": manifest["dataset_identity"],
                          "candidate_grid": [20, 50, 100], "budget_modes": ["per_channel", "total"],
                          "channel_budget": 100, "rrf_k": 60, "requested_channels": list(CHANNELS),
                          "query_ids": query_ids}):
        raise ValueError("Original protocol or adapter provenance differs")
    channel = manifest["missing_channel"]
    identities = manifest["channel_identities"]
    if set(identities) != set(CHANNELS) or identities[channel] != _missing_identity(report, channel):
        raise ValueError("Derived channel identity differs")
    if set(manifest["rankings"]) != set(CHANNELS) - {channel}:
        raise ValueError("Exact retained ranking channels required")
    rankings = {}
    for key, entry in manifest["rankings"].items():
        if entry["path"] != _ranking_path(identities[key]):
            raise ValueError("Ranking path identity differs")
        if (report["channels"][key].get("status") != "completed"
                or report["channels"][key]["identity"] != identities[key]):
            raise ValueError("Ranking source provenance differs")
        rankings[key] = _rankings(read_json(root, entry), identities[key], corpus_ids, query_ids, exclusions,
                                   bm25=key == "B")
    if path_at(root, _ranking_path(identities[channel])).exists():
        raise ValueError("Original target ranking unexpectedly appeared")
    arrays = {role: _vectors(root, manifest, report, role, count)
              for role, count in (("document", len(corpus)), ("query", len(queries)))}
    for entry in manifest["additional_metadata"]:
        checked(root, entry)
    expected_prior = {"runs/multimodal/" + name + "-" + suffix for suffix in ("all-retrieval", "bm25")}
    if (len(manifest["prior"]) != 2
            or {str(Path(item["report"]["path"]).parent) for item in manifest["prior"]} != expected_prior):
        raise ValueError("Exact prior result stores required")
    for prior in manifest["prior"]:
        base = str(Path(prior["report"]["path"]).parent)
        if prior["sqlite"]["path"] != base + "/progress.sqlite3":
            raise ValueError("Prior SQLite path identity differs")
        checked(root, prior["report"])
        checked(root, prior["sqlite"])
        wal = path_at(root, prior["sqlite"]["path"] + "-wal")
        if prior["wal"] is not None:
            if prior["wal"]["path"] != prior["sqlite"]["path"] + "-wal" or checked(root, prior["wal"]):
                raise ValueError("Prior WAL is not sealed empty")
        elif wal.exists():
            raise ValueError("Prior WAL appeared")
    return report, corpus_ids, query_ids, qrels, exclusions, rankings, arrays


def _fsync_directory(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def exclusive_write(path, value):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(value)
            stream.flush()
            os.fsync(fd)
    finally:
        os.close(fd)
    _fsync_directory(Path(path).parent)


def _prior_results(root, manifest):
    result, handles = {}, []
    for prior in manifest["prior"]:
        report = read_json(root, prior["report"])
        if report["dataset"]["identity"] != manifest["dataset_identity"]:
            raise ValueError("Prior dataset differs")
        if report["configuration"]["query_ids"] != read_json(root, manifest["main_report"])["configuration"]["query_ids"]:
            raise ValueError("Prior query order differs")
        uri = path_at(root, prior["sqlite"]["path"]).as_uri() + "?mode=ro&immutable=1"
        connection = sqlite3.connect(uri, uri=True)
        handles.append(connection)
        coverage = []
        for cell in report["cells"]:
            if cell.get("status") == "completed":
                if not cell["variant_id"].endswith("__none") or cell.get("candidate_k") is not None:
                    raise ValueError("Unexpected prior reranking result")
                key = (cell["variant_id"], cell["budget_mode"])
                coverage.append(key)
                subset = cell["variant_id"].split("__")[1].upper().split("_")
                identity = digest({**report["configuration"], "variant": cell["variant_id"],
                                   "mode": cell["budget_mode"], "candidate_k": None,
                                   "channel_identities": {key: manifest["channel_identities"][key]
                                                          for key in subset},
                                   "reranker": None, "reranker_prompts": None})
                if identity != cell["identity"]:
                    raise ValueError("Prior cell identity differs")
                result.setdefault(key, []).append((cell, connection))
        if prior["report"]["path"].endswith("-bm25/report.json"):
            subsets = [("B",)]
        elif manifest["missing_channel"] == "E":
            available = [key for key in CHANNELS if key != "E"]
            subsets = [subset for width in range(1, 6) for subset in itertools.combinations(available, width)]
        else:
            subsets = []
        expected = {("document__" + "_".join(key.lower() for key in subset) + "__none", mode)
                    for subset in subsets for mode in ("per_channel", "total")}
        if len(coverage) != len(set(coverage)) or set(coverage) != expected:
            raise ValueError("Prior completed coverage differs")
    return result, handles


def _same_metrics(actual, expected):
    if (set(actual) != set(expected)
            or any(not math.isfinite(actual[key]) or not math.isfinite(expected[key])
                   or abs(actual[key] - expected[key]) > 1e-12 for key in actual)):
        raise ValueError("Prior metric agreement failed")


def runtime_identity():
    return {"python": platform.python_version(), "numpy": np.__version__, "machine": platform.machine(),
            "numpy_build_sha256": digest(np.__config__.CONFIG)}


def execute(root, manifest, manifest_sha256, *, resume_closed=False, closed_sha256=None):
    """Recover only the missing ranking and derive126 cells; originals are read-only."""
    snapshot = canonical(manifest)
    if hashlib.sha256(snapshot).hexdigest() != manifest_sha256:
        raise ValueError("Reviewed manifest seal differs")
    manifest = json.loads(snapshot)
    root = Path(root)
    report, corpus_ids, query_ids, qrels, exclusions, rankings, arrays = validate_inputs(root, manifest)
    pure = _pure(root, manifest["source_sha256"])
    output = path_at(root, manifest["output"])
    if output.exists():
        if not resume_closed:
            raise FileExistsError("Existing delta preserved; explicit closed verification required")
        if not (output / "closed.json").is_file():
            raise ValueError("Incomplete delta preserved; no automatic retry")
        closed = json.loads(file_bytes(output / "closed.json"))
        required = {"manifest.json", "recovered-ranking.json", "results.sqlite3", "report.json", "matrix.csv"}
        if (not closed_sha256 or digest(closed) != closed_sha256 or set(closed.get("files", {})) != required
                or closed.get("completed_cells") != 126):
            raise ValueError("Reviewed closed seal or mandatory outputs differ")
        if closed.get("manifest_sha256") != manifest_sha256 or closed.get("status") != "closed":
            raise ValueError("Existing delta is not a matching closed result")
        if {x.name for x in output.iterdir()} != set(closed["files"]) | {"closed.json"}:
            raise ValueError("Closed delta files differ")
        for name, entry in closed["files"].items():
            if entry["path"] != manifest["output"] + "/" + name:
                raise ValueError("Closed result path differs")
            checked(root, entry)
            if stat.S_IMODE((output / name).stat().st_mode) != 0o600:
                raise ValueError("Closed output permissions changed")
        if file_bytes(output / "manifest.json") != snapshot + b"\n":
            raise ValueError("Closed manifest differs")
        return {"status": "verified_closed_unchanged", "completed_cells": 126}
    if resume_closed:
        raise ValueError("No closed result exists")
    if not output.parent.is_dir():
        raise ValueError("Reviewed private delta parent must already exist")
    start = time.perf_counter()
    missing = manifest["missing_channel"]
    rankings[missing] = pure["exact_dot_rankings"](arrays["document"], arrays["query"], corpus_ids,
                                                  top_k=100, exclusions=exclusions)
    search_seconds = time.perf_counter() - start
    payload = {"identity": manifest["channel_identities"][missing], "query_ids": query_ids,
               "rankings": rankings[missing], "usage": {"new_embedding_rows": 0,
               "embedding_inference_seconds": None, "cpu_exact_search_seconds": search_seconds,
               "cache_reads_are_inference": False, "source_manifest_sha256": manifest_sha256}}
    _rankings(payload, payload["identity"], corpus_ids, query_ids, exclusions)
    prior, handles = _prior_results(root, manifest)
    output.mkdir(mode=0o700)
    _fsync_directory(output.parent)
    connection = None
    try:
        exclusive_write(output / "manifest.json", canonical(manifest) + b"\n")
        exclusive_write(output / "recovered-ranking.json", canonical(payload) + b"\n")
        exclusive_write(output / "results.sqlite3", b"")
        connection = sqlite3.connect(output / "results.sqlite3")
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("CREATE TABLE results (cell TEXT, query_id TEXT, payload TEXT, PRIMARY KEY(cell,query_id))")
        cells, agreements = [], 0
        for width in range(1, 7):
            for subset in itertools.combinations(CHANNELS, width):
                variant = "document__" + "_".join(key.lower() for key in subset) + "__none"
                for mode in ("per_channel", "total"):
                    identity = digest({**report["configuration"], "variant": variant, "mode": mode,
                                       "candidate_k": None, "channel_identities": {
                                           key: manifest["channel_identities"][key] for key in subset},
                                       "reranker": None, "reranker_prompts": None})
                    metrics, pools = [], []
                    old = prior.get((variant, mode), [])
                    for index, qid in enumerate(query_ids):
                        ranking, pool = pure["fuse_rankings"]({key: rankings[key][index] for key in subset},
                                                              subset, budget_mode=mode, budget=100, rrf_k=60)
                        values = pure["graded_metrics"]([row["id"] for row in ranking], qrels[qid])
                        result = {"query_id": qid, "ranking": ranking, "metrics": values,
                                  "pool": {**pool, "rerank_candidates": 0}, "usage": {}}
                        for old_cell, old_db in old:
                            stored = old_db.execute("SELECT payload FROM results WHERE cell=? AND query_id=?",
                                                    (old_cell["identity"], qid)).fetchone()
                            if stored is None:
                                raise ValueError("Prior per-query result missing")
                            previous = json.loads(stored[0])
                            if previous["ranking"] != ranking or previous["pool"] != result["pool"]:
                                raise ValueError("Prior per-query ranking or pool differs")
                            _same_metrics(values, previous["metrics"])
                        connection.execute("INSERT INTO results VALUES (?,?,?)", (identity, qid, canonical(result).decode()))
                        metrics.append(values)
                        pools.append(pool)
                    aggregate = pure["aggregate_metrics"](metrics)
                    for old_cell, _ in old:
                        _same_metrics(aggregate, old_cell["metrics"])
                        means = {"mean_raw_candidates": float(np.mean([x["raw_candidates"] for x in pools])),
                                 "mean_unique_candidates": float(np.mean([x["unique_candidates"] for x in pools])),
                                 "mean_rerank_candidates": 0.0}
                        if (old_cell["completed_queries"] != len(query_ids)
                                or any(not math.isfinite(old_cell[key]) or abs(old_cell[key] - value) > 1e-12
                                       for key, value in means.items())):
                            raise ValueError("Prior completed-query count or pool means differ")
                        agreements += 1
                    cells.append({"variant_id": variant, "identity": identity, "budget_mode": mode,
                                  "candidate_k": None, "status": "completed", "query_count": len(query_ids),
                                  "completed_queries": len(query_ids), "metrics": aggregate,
                                  "mean_raw_candidates": float(np.mean([x["raw_candidates"] for x in pools])),
                                  "mean_unique_candidates": float(np.mean([x["unique_candidates"] for x in pools])),
                                  "new_embedding_rows": 0, "rerank_fresh_inference_seconds": None,
                                  "latency_scope": "CPU reconstruction; cached embeddings are not fresh inference"})
                    connection.commit()
        if len(cells) != 126 or set(prior) - {(x["variant_id"], x["budget_mode"]) for x in cells}:
            raise ValueError("Incomplete matrix or prior agreement")
        connection.close()
        connection = None
        with (output / "results.sqlite3").open("rb") as stream:
            os.fsync(stream.fileno())
        result_report = {"schema_version": 1, "engine": report["engine"], "dataset": report["dataset"],
                         "configuration": report["configuration"], "adapter_specs": copy.deepcopy(report["adapter_specs"]),
                         "cells": cells, "recovery_protocol": VERSION, "source_manifest_sha256": manifest_sha256,
                         "channel_identities": manifest["channel_identities"], "prior_cell_agreements": agreements,
                         "new_model_calls": 0, "new_embedding_rows": 0, "gpu_calls": 0,
                         "embedding_inference_seconds": None, "cpu_exact_search_seconds": search_seconds,
                         "cache_reads_are_inference": False, "hardware": "CPU reconstruction of retained MPS vectors"}
        result_report["cpu_runtime"] = runtime_identity()
        exclusive_write(output / "report.json", canonical(result_report) + b"\n")
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=["variant_id", "budget_mode", "hit@5", "hit@20"])
        writer.writeheader()
        for cell in cells:
            writer.writerow({"variant_id": cell["variant_id"], "budget_mode": cell["budget_mode"],
                             "hit@5": cell["metrics"]["hit@5"], "hit@20": cell["metrics"]["hit@20"]})
        exclusive_write(output / "matrix.csv", buffer.getvalue().encode())
        # Re-read every sealed original before closure; no historical file is written.
        validate_inputs(root, manifest)
        if canonical(manifest) != snapshot:
            raise ValueError("Detached manifest unexpectedly changed")
        files = {path.name: record(root, manifest["output"] + "/" + path.name) for path in output.iterdir()}
        closed = {"status": "closed", "manifest_sha256": manifest_sha256, "completed_cells": 126,
                  "prior_cell_agreements": agreements, "files": files}
        exclusive_write(output / "closed.json", canonical(closed) + b"\n")
        return {"status": "completed_pending_independent_audit", "completed_cells": 126,
                "prior_cell_agreements": agreements, "new_model_calls": 0, "closed_sha256": digest(closed)}
    finally:
        if connection is not None:
            connection.close()
        for handle in handles:
            handle.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", help="SHA256 of canonical JSON, without final newline")
    parser.add_argument("--prepare-from-recipe", type=Path)
    parser.add_argument("--recipe-sha256", help="SHA256 of canonical recipe JSON, without final newline")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume-closed", action="store_true")
    parser.add_argument("--closed-sha256", help="Required independently retained closed seal for closed verification")
    args = parser.parse_args(argv)
    if args.prepare_from_recipe:
        if args.execute or args.resume_closed or args.manifest_sha256 or not args.recipe_sha256:
            parser.error("Preparation requires only a reviewed recipe and a new manifest destination")
        recipe = json.loads(file_bytes(args.prepare_from_recipe))
        if digest(recipe) != args.recipe_sha256:
            raise ValueError("Recipe seal differs")
        expected = path_at(args.root, "work/vidore-cache-recovery-v2/manifests/" + recipe["dataset_id"] + ".json")
        if args.manifest != expected or not expected.parent.is_dir():
            raise ValueError("Exact existing private manifest parent required")
        manifest = prepare_manifest(args.root, recipe)
        exclusive_write(expected, canonical(manifest) + b"\n")
        print(json.dumps({"status": "prepared_not_executed", "manifest_sha256": digest(manifest)}))
        return 0
    if not args.manifest_sha256 or args.recipe_sha256:
        parser.error("A reviewed manifest seal is required")
    manifest = json.loads(file_bytes(args.manifest))
    if digest(manifest) != args.manifest_sha256:
        raise ValueError("Manifest seal differs")
    if args.execute:
        result = execute(args.root, manifest, args.manifest_sha256, resume_closed=args.resume_closed,
                         closed_sha256=args.closed_sha256)
    else:
        if args.resume_closed:
            parser.error("--resume-closed requires --execute")
        validate_inputs(args.root, manifest)
        result = {"status": "validated_without_execution", "model_calls": 0}
    print(json.dumps(result))
    return 0

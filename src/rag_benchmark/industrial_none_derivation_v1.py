"""Pure CPU None126 derivation from six already sealed industrial rankings.

No embedding cache, model assets, model constructor, GPU lock or inference seam
is used. Retained archive databases are checked in memory, never promoted into
old storage. Failed partial outputs are preserved; only sealed closure can replay.
"""
from __future__ import annotations

import argparse
import csv
import io
import itertools
import json
import math
import os
from pathlib import Path
import sqlite3
import stat
import sys
import tarfile

from . import industrial_mps_sj_v2 as origin
from . import vidore_cache_recovery_v2 as frozen

VERSION = "industrial-none-derivation-v1"
BASE = "work/" + VERSION
OUTPUT = BASE + "/actual-v1"
CHANNELS = "BGENSJ"
MODES = ("per_channel", "total")
METRICS = {f"{name}@{k}" for name in ("hit", "recall", "mrr", "ndcg") for k in (1, 5, 10, 20)} | {"map@16"}
SOURCES = frozen.SOURCE_FILES + origin.SOURCES + (
    "src/rag_benchmark/industrial_none_derivation_v1.py", "scripts/run_industrial_none_derivation_v1.py")
MPS_MANIFEST = origin.BASE + "/prepared-manifest-v2.json"
STAGE_CLOSED = {c: origin.DELTA + "/" + c + "/closed.json" for c in "SJ"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sources(root, pins):
    require(set(pins) == set(SOURCES), "Exact CPU source closure required")
    for path in SOURCES:
        require(frozen.record(root, path)["sha256"] == pins[path], "CPU source changed")
    pure = frozen._pure(root, {p: pins[p] for p in frozen.SOURCE_FILES})
    origin.verify_module(origin, frozen.path_at(root, "src/rag_benchmark/industrial_mps_sj_v2.py"))
    origin.verify_module(sys.modules[__name__], frozen.path_at(root, "src/rag_benchmark/industrial_none_derivation_v1.py"))
    return pure


def recipe(root, pins):
    return {"version": VERSION, "root": str(root), "source_pins": pins, "output": OUTPUT,
            "dataset_id": origin.DATASET, "dataset_identity": origin.DATASET_IDENTITY,
            "mps_manifest_path": MPS_MANIFEST, "stage_closed_paths": STAGE_CLOSED,
            "channels": CHANNELS, "budget": 100, "rrf_k": 60, "budget_modes": list(MODES),
            "candidate_grid_for_laya": [20, 50, 100], "expected_none_cells": 126,
            "expected_original_standalone_cells": 8}


def admitted_file(root, path, expected_sha):
    require(isinstance(expected_sha, str) and len(expected_sha) == 64, "Out-of-band SHA required")
    entry = frozen.record(root, path)
    require(entry["sha256"] == expected_sha, "Admitted file SHA differs")
    return entry, frozen.read_json(root, entry)


def archive_bundle(root, mps):
    """Read only exact allowlisted tar members and retain original raw-byte seals."""
    archive_record = origin.stream_record(root, origin.ARCHIVE)
    require(archive_record == mps["inherited"]["archive"] and
            archive_record["sha256"] == origin.ARCHIVE_SHA and archive_record["bytes"] == 90193920,
            "Inherited archive seal differs")
    records, payloads = {}, {}
    with tarfile.open(frozen.path_at(root, origin.ARCHIVE), "r:") as archive:
        members = archive.getmembers()
        require(len({m.name for m in members}) == len(members), "Duplicate archive member")
        for m in members:
            require(not Path(m.name).is_absolute() and "\\" not in m.name and
                    not any(p in ("", ".", "..") for p in m.name.split("/")) and
                    not m.issym() and not m.islnk(), "Unsafe archive member")
            require(not (m.name.endswith(("-wal", "-journal")) and m.size), "Uncheckpointed prior database")
        for channel in "BGEN":
            saved = mps["inherited"]["channels"][channel]
            require(saved["identity"] == origin.INHERITED_IDS[channel], "Inherited channel identity differs")
            records[channel], payloads[channel] = {}, {}
            for kind, name in (("report", "runs/" + channel + "/report.json"),
                               ("ranking", saved["ranking"]["member"]),
                               ("sqlite", "runs/" + channel + "/progress.sqlite3")):
                member = archive.getmember(name)
                require(member.isfile() and 0 < member.size <= 32 * 1024 * 1024, "Bounded regular member required")
                raw = archive.extractfile(member).read()
                ref = {"member": name, "bytes": len(raw), "sha256": origin.sha(raw)}
                if kind in ("report", "ranking"):
                    require(ref == saved[kind], "Inherited member seal differs")
                if kind == "report":
                    require(ref["sha256"] == origin.REPORT_HASHES[channel], "Historical report pin differs")
                records[channel][kind], payloads[channel][kind] = ref, raw
    return archive_record, records, payloads


def stage_bundle(root, mps, mps_sha, channel, closed_ref):
    closed = frozen.read_json(root, closed_ref)
    identity = origin.stage_identity(mps, mps_sha, channel)
    require(closed_ref["path"] == STAGE_CLOSED[channel] and closed.get("status") == "completed" and
            closed.get("identity") == identity and closed.get("manifest_sha256") == mps_sha,
            "Wrong completed stage identity")
    require(set(closed.get("files", {})) == {"rankings.json", "report.json", "scoring.claim.json"},
            "Mandatory stage outputs missing")
    terminal_records = closed.get("terminal_records", [])
    expected_paths = {origin.DELTA + "/" + channel + "/" + role + f"/{index:06d}.terminal.json"
                      for role, count in (("document", 5244), ("query", 283)) for index in range(count)}
    require(len(terminal_records) == len(expected_paths) and {r["path"] for r in terminal_records} == expected_paths
            and closed.get("terminal_sha256") == frozen.digest(terminal_records), "Stage terminal coverage differs")
    # Verify terminal metadata seals; do not open any embedding/token file.
    for ref in terminal_records:
        terminal = frozen.read_json(root, ref)
        require(terminal.get("status") == "completed", "Failed stage terminal")
    payloads = {}
    for name, ref in closed["files"].items():
        require(ref["path"] == origin.DELTA + "/" + channel + "/" + name, "Stage output path differs")
        payloads[name] = frozen.read_json(root, ref)
    report, ranking = payloads["report.json"], payloads["rankings.json"]
    require(report["condition"] == origin.VERSION and report["manifest_sha256"] == mps_sha and
            report["dataset"]["identity"] == origin.DATASET_IDENTITY and report["hardware"] == origin.HARDWARE and
            report["channels"][channel] == {"status": "completed", "identity": identity}, "Stage report differs")
    require(ranking.get("identity") == identity and ranking.get("condition") == origin.VERSION and
            ranking.get("manifest_sha256") == mps_sha and ranking.get("dataset_identity") == origin.DATASET_IDENTITY and
            ranking.get("hardware") == origin.HARDWARE[channel], "Stage ranking provenance differs")
    return {"identity": identity, "closed": closed_ref, "report": closed["files"]["report.json"],
            "ranking": closed["files"]["rankings.json"]}, report, ranking


def strict_ranking(payload, identity, corpus_ids, query_ids, exclusions, channel):
    require(isinstance(payload.get("rankings"), list) and all(isinstance(rows, list) and all(
        isinstance(row, dict) and set(row) == {"id", "score", "rank"} and isinstance(row["id"], str)
        and type(row["rank"]) is int and type(row["score"]) in (int, float) for row in rows)
        for rows in payload["rankings"]), "Ranking scalar schema differs")
    return frozen._rankings(payload, identity, corpus_ids, query_ids, exclusions, bm25=channel == "B")


def metrics_equal(actual, expected):
    require(set(actual) == METRICS and set(expected) == METRICS, "Exactly 17 metrics required")
    require(all(type(v) in (int, float) and math.isfinite(v) for v in actual.values()) and
            all(type(v) in (int, float) and math.isfinite(v) for v in expected.values()), "Nonfinite metric")
    frozen._same_metrics(actual, expected)


def memory_database(raw):
    require(0 < len(raw) <= 32 * 1024 * 1024 and raw.startswith(b"SQLite format 3\x00") and
            raw[18:20] in (b"\x01\x01", b"\x02\x02"), "Unsupported prior SQLite image")
    # sqlite3.deserialize cannot attach a WAL-mode image. An isolated in-memory
    # copy changes only its journal-mode header; original archive bytes stay sealed.
    # archive_bundle forbids any nonempty WAL/journal, and exact rows/metrics below
    # reject an incomplete checkpoint. No disk-backed original is ever opened.
    memory = raw[:18] + b"\x01\x01" + raw[20:]
    db = sqlite3.connect(":memory:")
    try:
        db.deserialize(memory)
        db.execute("PRAGMA query_only=ON")
        db.execute("PRAGMA trusted_schema=OFF")
        require(db.execute("PRAGMA integrity_check").fetchall() == [("ok",)], "Prior SQLite integrity differs")
        require(db.execute("SELECT type FROM sqlite_master WHERE name='results'").fetchall() == [("table",)],
                "Prior results must be a table")
        require([r[1] for r in db.execute("PRAGMA table_info(results)")] == ["cell", "query_id", "payload"],
                "Prior SQLite schema differs")
        return db
    except BaseException:
        db.close()
        raise


def prior_agreement(channel, report, sqlite_bytes, rankings, query_ids, qrels, pure, dataset_identity):
    """Exact two original cells, exact query coverage, rankings/pools and 17 metrics."""
    config = report["configuration"]
    identity = origin.INHERITED_IDS[channel]
    require(report["dataset"]["identity"] == dataset_identity and config["dataset"] == dataset_identity and
            config["query_ids"] == query_ids and config["requested_channels"] == [channel] and
            config["budget_modes"] == list(MODES) and config["channel_budget"] == 100 and
            config["rrf_k"] == 60 and config["candidate_grid"] == [20, 50, 100] and
            report["scope"] == "frozen_collection" and report["evaluated_query_count"] == len(query_ids) and
            report["channels"][channel]["status"] == "completed" and
            report["channels"][channel]["identity"] == identity, "Prior scope/identity differs")
    variant = "document__" + channel.lower() + "__none"
    completed = [c for c in report["cells"] if c.get("status") == "completed"]
    require(len(completed) == 2 and {(c["variant_id"], c["budget_mode"]) for c in completed} ==
            {(variant, mode) for mode in MODES}, "Exact unique original two-cell coverage required")
    cells = {}
    for cell in completed:
        mode = cell["budget_mode"]
        old_id = frozen.digest({**config, "variant": variant, "mode": mode, "candidate_k": None,
            "channel_identities": {channel: identity}, "reranker": None, "reranker_prompts": None})
        require(cell["identity"] == old_id and cell["candidate_k"] is None and
                cell["query_count"] == len(query_ids) and cell["completed_queries"] == len(query_ids),
                "Original cell identity/count differs")
        cells[old_id] = cell
    db = memory_database(sqlite_bytes)
    try:
        rows = db.execute("SELECT cell, query_id, payload FROM results").fetchall()
    finally:
        db.close()
    require(len(rows) == 2 * len(query_ids), "Prior row count differs")
    expected_keys = {(cell, qid) for cell in cells for qid in query_ids}
    require({(c, q) for c, q, _ in rows} == expected_keys, "Prior query coverage differs")
    stored = {(c, q): json.loads(value) for c, q, value in rows}
    agreements = []
    for old_id, cell in cells.items():
        metrics, pools = [], []
        for index, qid in enumerate(query_ids):
            ranking, pool = pure["fuse_rankings"]({channel: rankings[index]}, [channel],
                budget_mode=cell["budget_mode"], budget=100, rrf_k=60)
            values = pure["graded_metrics"]([row["id"] for row in ranking], qrels[qid])
            previous = stored[(old_id, qid)]
            require(previous.get("query_id") == qid and previous["ranking"] == ranking and
                    previous["pool"] == {**pool, "rerank_candidates": 0}, "Prior per-query ranking/pool differs")
            metrics_equal(values, previous["metrics"])
            metrics.append(values)
            pools.append(pool)
        metrics_equal(pure["aggregate_metrics"](metrics), cell["metrics"])
        means = {"mean_raw_candidates": sum(p["raw_candidates"] for p in pools) / len(pools),
                 "mean_unique_candidates": sum(p["unique_candidates"] for p in pools) / len(pools),
                 "mean_rerank_candidates": 0.0}
        require(all(type(cell[k]) in (int, float) and math.isfinite(cell[k]) and abs(cell[k] - v) <= 1e-12
                    for k, v in means.items()), "Prior pool means differ")
        agreements.append({"channel": channel, "variant_id": variant, "budget_mode": cell["budget_mode"],
                           "original_identity": old_id, "verified_queries": len(query_ids), "metrics": cell["metrics"]})
    return agreements


def inspect_inputs(root, request, admission):
    pure = sources(root, request["source_pins"])
    require(request == recipe(root, request["source_pins"]), "Fixed CPU recipe differs")
    require(set(admission) == {"mps_manifest_sha256", "closed_sha256", "accepted_evidence"} and
            set(admission["closed_sha256"]) == {"S", "J"} and admission["accepted_evidence"],
            "Exact admitted MPS manifest, S/J closures and review evidence required")
    for ref in admission["accepted_evidence"]:
        frozen.checked(root, ref)
    mps_ref, mps = admitted_file(root, MPS_MANIFEST, admission["mps_manifest_sha256"])
    require(mps.get("version") == origin.VERSION and mps["root"] == str(root) and
            mps["dataset_id"] == origin.DATASET and mps["dataset_identity"] == origin.DATASET_IDENTITY and
            mps["dataset_revision"] == origin.DATASET_REVISION and mps["hardware"] == origin.HARDWARE and
            mps["output"] == origin.DELTA, "Wrong admitted mixed condition")
    require(set(mps["source_pins"]) == set(origin.SOURCES) and
            all(mps["source_pins"][p] == request["source_pins"][p] for p in origin.SOURCES) and
            mps["recipe_sha256"] == frozen.digest(origin.recipe(root, mps["source_pins"])),
            "Admitted MPS source/recipe binding differs")
    mps_sha = frozen.digest(mps)
    metadata, corpus, queries, qrels, exclusions = frozen._dataset(root, mps)
    require((len(corpus), len(queries)) == (5244, 283) and metadata["revision"] == origin.DATASET_REVISION,
            "Industrial population/revision differs")
    corpus_ids, query_ids = [str(r["id"]) for r in corpus], [str(q["id"]) for q in queries]
    archive, members, raw = archive_bundle(root, mps)
    reports, rankings, refs, ids, agreements = {}, {}, {}, {}, []
    for channel in "BGEN":
        reports[channel] = json.loads(raw[channel]["report"])
        require(reports[channel]["dataset"]["revision"] == origin.DATASET_REVISION and
                reports[channel]["dataset"]["corpus_count"] == len(corpus) and
                reports[channel]["dataset"]["query_count"] == len(queries), "Prior dataset revision/population differs")
        ids[channel] = origin.INHERITED_IDS[channel]
        rankings[channel] = strict_ranking(json.loads(raw[channel]["ranking"]), ids[channel], corpus_ids,
                                            query_ids, exclusions, channel)
        agreements.extend(prior_agreement(channel, reports[channel], raw[channel]["sqlite"], rankings[channel],
                                          query_ids, qrels, pure, origin.DATASET_IDENTITY))
        refs[channel] = {"identity": ids[channel], **members[channel]}
    for channel in "SJ":
        closed_ref, _ = admitted_file(root, STAGE_CLOSED[channel], admission["closed_sha256"][channel])
        refs[channel], reports[channel], payload = stage_bundle(root, mps, mps_sha, channel, closed_ref)
        ids[channel] = refs[channel]["identity"]
        rankings[channel] = strict_ranking(payload, ids[channel], corpus_ids, query_ids, exclusions, channel)
    require(len(agreements) == 8, "Eight inherited cell agreements required")
    evidence = {"mps_manifest": mps_ref, "mps_manifest_canonical_sha256": mps_sha, "archive": archive,
                "channels": refs, "dataset_id": mps["dataset_id"], "dataset_identity": mps["dataset_identity"],
                "tables": mps["tables"], "media": mps["media"], "ordered_id_sha256": mps["ordered_id_sha256"],
                "channel_identities": ids, "hardware": origin.HARDWARE, "prior_agreements": agreements}
    return evidence, rankings, reports, raw, query_ids, qrels, pure


def prepare(root, request, request_sha, admission, admission_sha):
    request, admission = origin.detached(request), origin.detached(admission)
    require(frozen.digest(request) == request_sha and frozen.digest(admission) == admission_sha,
            "Recipe/admission SHA differs")
    evidence, *_ = inspect_inputs(root, request, admission)
    return {"recipe": request, "recipe_sha256": request_sha, "admission": admission,
            "admission_sha256": admission_sha, "runtime": frozen.runtime_identity(), "evidence": evidence}


def validate(root, manifest):
    require(manifest["recipe_sha256"] == frozen.digest(manifest["recipe"]) and
            manifest["admission_sha256"] == frozen.digest(manifest["admission"]) and
            manifest["runtime"] == frozen.runtime_identity(), "CPU manifest runtime/seals differ")
    found = inspect_inputs(root, manifest["recipe"], manifest["admission"])
    require(found[0] == manifest["evidence"], "Sealed CPU input evidence differs")
    return found


def configuration(dataset_identity, query_ids, condition):
    return {"engine": "multimodal-v1", "dataset": dataset_identity, "candidate_grid": [20, 50, 100],
            "budget_modes": list(MODES), "channel_budget": 100, "rrf_k": 60,
            "requested_channels": list(CHANNELS), "query_ids": query_ids, "retrieval_condition": condition}


def derive_rows(rankings, query_ids, qrels, identities, config, pure):
    """Yield one completed cell and its exact per-query results; no inference paths."""
    for width in range(1, 7):
        for subset in itertools.combinations(CHANNELS, width):
            variant = "document__" + "_".join(c.lower() for c in subset) + "__none"
            for mode in MODES:
                identity = frozen.digest({**config, "variant": variant, "mode": mode, "candidate_k": None,
                    "channel_identities": {c: identities[c] for c in subset}, "reranker": None, "reranker_prompts": None})
                rows = []
                for index, qid in enumerate(query_ids):
                    ranking, pool = pure["fuse_rankings"]({c: rankings[c][index] for c in subset}, subset,
                                                        budget_mode=mode, budget=100, rrf_k=60)
                    metrics = pure["graded_metrics"]([r["id"] for r in ranking], qrels[qid])
                    metrics_equal(metrics, metrics)
                    require(all(math.isfinite(float(r["score"])) for r in ranking), "Nonfinite fused score")
                    rows.append({"query_id": qid, "ranking": ranking, "metrics": metrics,
                                 "pool": {**pool, "rerank_candidates": 0}, "usage": {}})
                aggregate = pure["aggregate_metrics"]([r["metrics"] for r in rows])
                metrics_equal(aggregate, aggregate)
                cell = {"variant_id": variant, "identity": identity, "budget_mode": mode, "candidate_k": None,
                    "status": "completed", "query_count": len(query_ids), "completed_queries": len(rows),
                    "metrics": aggregate, "mean_raw_candidates": sum(r["pool"]["raw_candidates"] for r in rows) / len(rows),
                    "mean_unique_candidates": sum(r["pool"]["unique_candidates"] for r in rows) / len(rows),
                    "mean_rerank_candidates": 0.0, "new_embedding_rows": 0, "rerank_fresh_inference_seconds": None,
                    "latency_scope": "Pure CPU derivation from sealed rankings; no inference timing"}
                yield cell, rows


def required_output_names():
    return {"manifest.json", "report.json", "results.sqlite3", "matrix.csv", "six-ranking-inputs.json"} | {
        c + suffix for c in CHANNELS for suffix in ("-ranking.json", "-origin-report.json")}


def verify_closed(root, output, snapshot, manifest_sha, closed_sha):
    require(closed_sha is not None, "Out-of-band closed SHA required")
    raw = frozen.file_bytes(output / "closed.json")
    require(origin.sha(raw) == closed_sha, "Closed raw SHA differs")
    closed = json.loads(raw)
    require(closed.get("status") == "closed" and closed.get("manifest_sha256") == manifest_sha and
            closed.get("completed_cells") == 126 and closed.get("prior_cell_agreements") == 8 and
            set(closed.get("files", {})) == required_output_names(), "Closed scope/files differ")
    require({p.name for p in output.iterdir()} == required_output_names() | {"closed.json"}, "Closed output set differs")
    require(frozen.file_bytes(output / "manifest.json") == snapshot, "Closed manifest bytes differ")
    for name, ref in closed["files"].items():
        require(ref["path"] == OUTPUT + "/" + name, "Closed file path differs")
        frozen.checked(root, ref)
        require(stat.S_IMODE((output / name).stat().st_mode) == 0o600, "Private output mode differs")
    return {"status": "verified_closed_unchanged", "completed_cells": 126, "new_model_calls": 0}


def execute(root, manifest, manifest_sha, *, closed_sha=None):
    snapshot = frozen.canonical(manifest)
    require(origin.sha(snapshot) == manifest_sha, "Reviewed CPU manifest SHA differs")
    manifest = json.loads(snapshot)
    evidence, rankings, reports, raw, query_ids, qrels, pure = validate(root, manifest)
    output = frozen.path_at(root, OUTPUT)
    if output.exists():
        return verify_closed(root, output, snapshot, manifest_sha, closed_sha)
    require(closed_sha is None and output.parent.is_dir(), "Fresh output parent required")
    condition = frozen.digest({"protocol": VERSION, "retrieval_protocol": origin.VERSION,
        "dataset": evidence["dataset_identity"], "ordered_ids": evidence["ordered_id_sha256"],
        "channels": evidence["channels"], "hardware": evidence["hardware"], "source_manifest_sha256": manifest_sha,
        "budget": 100, "rrf_k": 60, "modes": list(MODES)})
    config = configuration(evidence["dataset_identity"], query_ids, condition)
    output.mkdir(mode=0o700)
    frozen._fsync_directory(output.parent)
    frozen.exclusive_write(output / "manifest.json", snapshot)
    origins, ranking_refs = {}, {}
    for channel in CHANNELS:
        if channel in "BGEN":
            report_bytes, ranking_bytes = raw[channel]["report"], raw[channel]["ranking"]
        else:
            report_bytes = frozen.checked(root, evidence["channels"][channel]["report"])
            ranking_bytes = frozen.checked(root, evidence["channels"][channel]["ranking"])
        for suffix, content in (("-origin-report.json", report_bytes), ("-ranking.json", ranking_bytes)):
            frozen.exclusive_write(output / (channel + suffix), content)
        ranking_refs[channel] = frozen.record(root, OUTPUT + "/" + channel + "-ranking.json")
        origins[channel] = {"report": frozen.record(root, OUTPUT + "/" + channel + "-origin-report.json"),
            "channel_identity": evidence["channel_identities"][channel], **origin.HARDWARE[channel]}
        if channel == "B":
            origins[channel]["dtype"] = "not_applicable_bm25"
    frozen.exclusive_write(output / "results.sqlite3", b"")
    db = sqlite3.connect(output / "results.sqlite3")
    try:
        db.execute("PRAGMA journal_mode=DELETE")
        db.execute("PRAGMA synchronous=FULL")
        db.execute("CREATE TABLE results (cell TEXT, query_id TEXT, payload TEXT, PRIMARY KEY(cell,query_id))")
        cells = []
        for cell, rows in derive_rows(rankings, query_ids, qrels, evidence["channel_identities"], config, pure):
            db.executemany("INSERT INTO results VALUES (?,?,?)", [(cell["identity"], r["query_id"],
                frozen.canonical(r).decode()) for r in rows])
            cells.append(cell)
        require(len(cells) == 126 and len({c["identity"] for c in cells}) == 126, "Exact unique None126 required")
        require(db.execute("SELECT COUNT(*) FROM results").fetchone()[0] == 126 * len(query_ids), "Derived row count differs")
        db.commit()
    finally:
        db.close()
    with (output / "results.sqlite3").open("rb") as handle:
        os.fsync(handle.fileno())
    derived = {(c["variant_id"], c["budget_mode"]): c for c in cells}
    linked_agreements = []
    for prior in evidence["prior_agreements"]:
        cell = derived[(prior["variant_id"], prior["budget_mode"])]
        metrics_equal(cell["metrics"], prior["metrics"])
        linked_agreements.append({**prior, "derived_identity": cell["identity"]})
    report = {"schema_version": 1, "engine": "multimodal-v1", "dataset": reports["B"]["dataset"],
        "scope": "frozen_collection", "evaluated_query_count": len(query_ids), "configuration": config,
        "cells": cells, "channels": {c: {"status": "completed", "identity": evidence["channel_identities"][c]}
                                   for c in CHANNELS}, "channel_identities": evidence["channel_identities"],
        "derivation_protocol": VERSION, "retrieval_protocol": origin.VERSION, "mixed_condition_identity": condition,
        "hardware": evidence["hardware"], "source_manifest_sha256": manifest_sha, "prior_cell_agreements": 8,
        "original_standalone_evidence": linked_agreements, "new_model_calls": 0, "new_embedding_rows": 0,
        "gpu_calls": 0, "retrieval_inference_seconds": None, "rerank_fresh_inference_seconds": None,
        "cache_reads_are_inference": False, "laya_cells_completed": 0, "cpu_runtime": manifest["runtime"]}
    frozen.exclusive_write(output / "report.json", frozen.canonical(report))
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=["variant_id", "budget_mode", *sorted(METRICS)])
    writer.writeheader()
    for cell in cells:
        writer.writerow({"variant_id": cell["variant_id"], "budget_mode": cell["budget_mode"], **cell["metrics"]})
    frozen.exclusive_write(output / "matrix.csv", buffer.getvalue().encode())
    handoff = {"protocol": VERSION, "dataset_id": origin.DATASET, "dataset_identity": evidence["dataset_identity"],
        "mixed_condition_identity": condition, "rankings": ranking_refs, "retrieval_provenance": origins,
        "baseline_report": frozen.record(root, OUTPUT + "/report.json"),
        "accepted_evidence": manifest["admission"]["accepted_evidence"],
        "input_manifest": evidence["mps_manifest"], "original_archive": evidence["archive"],
        "original_members": {c: evidence["channels"][c] for c in "BGEN"},
        "stage_closures": {c: evidence["channels"][c]["closed"] for c in "SJ"}}
    frozen.exclusive_write(output / "six-ranking-inputs.json", frozen.canonical(handoff))
    validate(root, manifest)
    require({p.name for p in output.iterdir()} == required_output_names(), "Unexpected derived output")
    closed = {"status": "closed", "manifest_sha256": manifest_sha, "completed_cells": 126,
              "prior_cell_agreements": 8, "mixed_condition_identity": condition,
              "files": {name: frozen.record(root, OUTPUT + "/" + name) for name in sorted(required_output_names())}}
    frozen.exclusive_write(output / "closed.json", frozen.canonical(closed))
    return {"status": "completed_pending_independent_audit", "completed_cells": 126, "prior_cell_agreements": 8,
            "new_model_calls": 0, "closed_sha256": origin.sha(frozen.canonical(closed))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "execute"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--admission")
    parser.add_argument("--admission-sha256")
    parser.add_argument("--closed-sha256")
    args = parser.parse_args(argv)
    sys.dont_write_bytecode = True
    value = json.loads(frozen.file_bytes(frozen.path_at(args.root, args.input)))
    if args.mode == "prepare":
        require(args.admission and args.admission_sha256 and not args.closed_sha256, "Admission evidence required")
        admission = json.loads(frozen.file_bytes(frozen.path_at(args.root, args.admission)))
        manifest = prepare(args.root, value, args.sha256, admission, args.admission_sha256)
        path = frozen.path_at(args.root, BASE + "/prepared-manifest-v1.json")
        frozen.exclusive_write(path, frozen.canonical(manifest))
        print(json.dumps({"manifest": frozen.record(args.root, path.relative_to(args.root).as_posix()),
                          "canonical_sha256": frozen.digest(manifest), "model_calls": 0}))
    else:
        require(not args.admission and not args.admission_sha256, "Execute accepts admitted manifest only")
        print(json.dumps(execute(args.root, value, args.sha256, closed_sha=args.closed_sha256)))
    return 0

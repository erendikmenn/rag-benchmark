"""Cache-only CPU prefix-dimension ablations of completed native EG2 retrieval.

No encoder or model-loading function is called. Original 768-dimensional vectors
are used unchanged. Smaller prefixes are copied and L2 normalized, then searched
against the entire frozen corpus. Reports contain aggregate results only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
from types import SimpleNamespace
import uuid

import numpy as np

from .models import stable_hash
from .multimodal import (ENGINE_VERSION, _channel_identity, _safe_model_config, _validate_vectors,
    aggregate_metrics, encoding_cache_identity, exact_dot_rankings, fuse_rankings, graded_metrics,
    load_dataset, public_adapter_spec)

SWEEP_VERSION = "native-eg2-cached-prefix-dimensions-v1"
DIMENSIONS = (128, 256, 512, 768)


class EvidenceUnavailable(ValueError):
    """A fixed, public-safe reason code; never include paths or source text."""


def _require(condition, code: str) -> None:
    if not condition:
        raise EvidenceUnavailable(code)


def _read_json(path: Path, missing: str) -> tuple[dict, str]:
    _require(path.is_file(), missing)
    content = path.read_bytes()
    return json.loads(content), hashlib.sha256(content).hexdigest()


def _native_adapter(report: dict):
    # Constructor fixes configuration/identity only. encode/_load are never used.
    from .multimodal_models import EmbeddingGemma2Adapter
    spec = report.get("adapter_specs", {}).get("channels", {}).get("N", {})
    config = spec.get("config", {})
    _require(spec.get("class") == "rag_benchmark.multimodal_models.EmbeddingGemma2Adapter"
             and config.get("model_id") == "google/embeddinggemma-2"
             and config.get("mode") == "native" and config.get("dimension") == 768
             and spec.get("dimension") == 768, "base_requires_native_768_eg2_spec")
    adapter = EmbeddingGemma2Adapter(config, mode="native")
    _require(adapter.identity == spec.get("identity"), "native_adapter_identity_mismatch")
    return adapter


def _vectors(dataset, adapter, role: str, cache_dir: Path) -> tuple[np.ndarray, dict]:
    identifiers = [item["id"] for item in (dataset.corpus if role == "document" else dataset.queries)]
    encoding = encoding_cache_identity(dataset, adapter, role)
    identity = stable_hash({"encoding": encoding, "text_only": False})
    root = cache_dir / "vectors" / identity
    _require(root.is_dir(), "missing_" + role + "_vector_cache")
    paths = sorted(root.glob("*.npy"))
    _require(bool(paths), "missing_" + role + "_vector_blocks")
    arrays, observed, cursor, missing_metadata, missing_ids = [], [], 0, 0, 0
    for path in paths:
        match = re.fullmatch(r"([0-9]{9})-([0-9]{9})\.npy", path.name)
        _require(match is not None, "invalid_vector_block_name")
        begin, end = map(int, match.groups())
        _require(begin == cursor and begin < end <= len(identifiers), "vector_block_gap_overlap_or_bounds")
        payload = path.read_bytes()
        # Load the same bytes that are hashed; concurrent replacement cannot mix evidence.
        import io
        values = np.load(io.BytesIO(payload), allow_pickle=False)
        _require(values.dtype == np.float32, "vector_dtype_must_be_float32")
        _validate_vectors(values, end - begin, 768)
        entry = {"begin": begin, "end": end, "newly_observed_array_sha256": hashlib.sha256(payload).hexdigest(),
                 "newly_observed_ordered_id_sha256": stable_hash(identifiers[begin:end])}
        metadata_path = path.with_suffix(".usage.json")
        if metadata_path.is_file():
            metadata, digest = _read_json(metadata_path, "missing_vector_metadata")
            _require(metadata.get("rows") == end - begin, "vector_metadata_row_count_mismatch")
            records = metadata.get("items", [])
            if records:
                _require([item.get("id") for item in records] == identifiers[begin:end], "vector_metadata_id_order_mismatch")
            else:
                missing_ids += 1
            entry["usage_metadata_sha256"] = digest
        else:
            missing_metadata += 1
            missing_ids += 1
        observed.append(entry)
        arrays.append(values)
        cursor = end
    _require(cursor == len(identifiers), "incomplete_vector_cache")
    aggregate_evidence = {"status": "absent"}
    aggregate_path = root / "diagnostics.json"
    if aggregate_path.is_file():
        metadata, digest = _read_json(aggregate_path, "missing_aggregate_vector_metadata")
        _require(metadata.get("total_rows") == cursor, "aggregate_vector_metadata_row_count_mismatch")
        recorded_ids = [item.get("id") for item in metadata.get("items", [])]
        positions = {identifier: i for i, identifier in enumerate(identifiers)}
        _require(len(recorded_ids) == len(set(recorded_ids)) and set(recorded_ids) <= set(positions)
                 and recorded_ids == sorted(recorded_ids, key=positions.get), "aggregate_vector_metadata_id_order_mismatch")
        aggregate_evidence = {"status": "verified", "sha256": digest, "explicit_row_id_count": len(recorded_ids)}
    return np.concatenate(arrays), {"role": role, "cache_identity": identity,
        "encoding_identity": encoding, "rows": cursor, "dimension": 768, "dtype": "float32",
        "newly_observed_ordered_id_sha256": stable_hash(identifiers),
        "newly_observed_blocks": observed, "blocks_without_usage_metadata": missing_metadata,
        "blocks_without_explicit_row_ids": missing_ids,
        "aggregate_usage_metadata": aggregate_evidence,
        "original_vector_checksums_available": False,
        "binding": "frozen_dataset_order_and_adapter_role_cache_key; exact_768_rankings_and_results_verified",
        "checksum_note": "Array and ordered-ID hashes are newly observed, not original cache checksums."}


def _ranking_cache(dataset, report: dict, channel: str, cache_dir: Path, budget: int) -> tuple[list, dict]:
    state = report.get("channels", {}).get(channel, {})
    _require(state.get("status") == "completed" and re.fullmatch(r"[a-f0-9]{64}", state.get("identity", "")),
             "base_" + channel.lower() + "_not_completed")
    path = cache_dir / "rankings" / stable_hash({"identity": state["identity"], "top_k": budget}) / "rankings.json"
    payload, digest = _read_json(path, "missing_" + channel.lower() + "_ranking_cache")
    identifiers = [query["id"] for query in dataset.queries]
    _require(payload.get("identity") == state["identity"] and payload.get("query_ids") == identifiers,
             "ranking_cache_identity_or_query_order_mismatch")
    rows = payload.get("rankings", [])
    _require(len(rows) == len(identifiers), "ranking_cache_query_count_mismatch")
    corpus_ids = {item["id"] for item in dataset.corpus}
    for query, ranking in zip(dataset.queries, rows):
        excluded = dataset.excluded_ids(query)
        ids = [row.get("id") for row in ranking]
        _require(len(ids) == min(budget, len(corpus_ids - excluded)) and len(ids) == len(set(ids))
                 and set(ids) <= corpus_ids and not set(ids) & excluded, "ranking_cache_candidate_coverage_mismatch")
        _require(all(isinstance(row.get("score"), (int, float)) and not isinstance(row["score"], bool)
                     and math.isfinite(row["score"]) and row.get("rank") == i
                     for i, row in enumerate(ranking, 1)), "ranking_cache_invalid_score_or_rank")
        _require(ranking == sorted(ranking, key=lambda row: (-row["score"], row["id"])), "ranking_cache_order_mismatch")
    return rows, {"channel_identity": state["identity"], "ranking_cache_sha256": digest,
                  "newly_observed_ordered_query_id_sha256": stable_hash(identifiers)}


def _same_rankings(actual: list, expected: list) -> bool:
    return len(actual) == len(expected) and all(a["id"] == b.get("id")
        and a["rank"] == b.get("rank") and isinstance(b.get("score"), (float, int))
        and math.isclose(a["score"], b["score"], abs_tol=1e-6, rel_tol=1e-5)
        for a, b in zip(actual, expected))


def _evaluate(dataset, rankings: dict, channels: tuple, mode: str, budget: int, rrf_k: int) -> tuple[list, dict]:
    rows, metric_rows, pools = [], [], []
    for i, query in enumerate(dataset.queries):
        fused, pool = fuse_rankings({channel: rankings[channel][i] for channel in channels}, channels,
                                   budget_mode=mode, budget=budget, rrf_k=rrf_k)
        rows.append(fused)
        metric_rows.append(graded_metrics([row["id"] for row in fused], dataset.qrels[query["id"]]))
        pools.append(pool)
    return rows, {"metrics": aggregate_metrics(metric_rows),
        "mean_raw_candidates": float(np.mean([p["raw_candidates"] for p in pools])),
        "mean_unique_candidates": float(np.mean([p["unique_candidates"] for p in pools]))}


def _verify_cell(dataset, report: dict, db: sqlite3.Connection, rankings: dict,
                 channels: tuple, mode: str, budget: int, rrf_k: int) -> dict:
    variant = dataset.track + "__" + "_".join(channels).lower() + "__none"
    cells = [cell for cell in report.get("cells", []) if cell.get("variant_id") == variant
             and cell.get("budget_mode") == mode and cell.get("candidate_k") is None]
    _require(len(cells) == 1 and cells[0].get("status") == "completed", "missing_completed_baseline_cell")
    cell = cells[0]
    identity = stable_hash({**report["configuration"], "variant": variant, "mode": mode, "candidate_k": None,
        "channel_identities": {channel: report["channels"][channel]["identity"] for channel in channels},
        "reranker": None, "reranker_prompts": None})
    _require(cell.get("identity") == identity, "baseline_cell_identity_mismatch")
    fused, summary = _evaluate(dataset, rankings, channels, mode, budget, rrf_k)
    positions = {query["id"]: i for i, query in enumerate(dataset.queries)}
    seen = set()
    for query_id, content in db.execute("SELECT query_id,payload FROM results WHERE cell=?", (identity,)):
        _require(query_id in positions and query_id not in seen, "baseline_result_query_coverage_mismatch")
        stored = json.loads(content)
        _require(stored.get("query_id") == query_id and _same_rankings(fused[positions[query_id]], stored.get("ranking", [])),
                 "baseline_result_ranking_mismatch")
        expected = graded_metrics([row["id"] for row in fused[positions[query_id]]], dataset.qrels[query_id])
        _require(_same_metrics(expected, stored.get("metrics", {})), "baseline_per_query_metric_mismatch")
        seen.add(query_id)
    _require(seen == set(positions) and cell.get("query_count") == len(seen)
             and cell.get("completed_queries") == len(seen), "incomplete_baseline_results")
    _require(_same_metrics(summary["metrics"], cell.get("metrics", {})), "baseline_aggregate_metric_mismatch")
    return {"cell_identity": identity, "channels": list(channels), "budget_mode": mode,
            "query_count": len(seen), "status": "reproduced", **summary}


def _same_metrics(expected: dict, actual: dict) -> bool:
    return set(expected) == set(actual) and all(isinstance(actual[key], (int, float))
        and math.isclose(value, actual[key], abs_tol=1e-10, rel_tol=1e-10) for key, value in expected.items())


def project_prefix(values: np.ndarray, dimension: int) -> np.ndarray:
    """768 is the original baseline, never normalized a second time."""
    if dimension not in DIMENSIONS or values.ndim != 2 or values.shape[1] != 768:
        raise ValueError("Prefix projection requires 768 columns and a supported dimension")
    if dimension == 768:
        return values
    projected = values[:, :dimension].copy()
    norms = np.linalg.norm(projected, axis=1, keepdims=True)
    _require(np.isfinite(projected).all() and np.isfinite(norms).all() and (norms > 0).all(), "invalid_or_zero_prefix_norm")
    return projected / norms


def _execute(dataset_dir, base_run: Path, cache_dir: Path, dimensions: tuple, include_specialist: bool) -> dict:
    dataset = load_dataset(dataset_dir)
    source, source_hash = _read_json(base_run / "report.json", "missing_base_report")
    config = source.get("configuration", {})
    _require(source.get("engine") == ENGINE_VERSION and config.get("engine") == ENGINE_VERSION, "base_engine_mismatch")
    _require(source.get("scope") == "frozen_collection" and source.get("dataset", {}).get("identity") == dataset.identity
             and config.get("dataset") == dataset.identity, "base_frozen_dataset_mismatch")
    _require(config.get("query_ids") == [q["id"] for q in dataset.queries]
             and source.get("evaluated_query_count") == len(dataset.queries), "base_full_query_order_mismatch")
    budget, rrf_k, modes = config.get("channel_budget"), config.get("rrf_k"), config.get("budget_modes")
    _require(type(budget) is int and budget > 0 and type(rrf_k) is int and rrf_k >= 0 and isinstance(modes, list)
             and bool(modes) and len(modes) == len(set(modes)) and set(modes) <= {"per_channel", "total"}, "base_budget_invalid")
    adapter = _native_adapter(source)
    _require(_channel_identity(dataset, "N", {"N": adapter}) == source.get("channels", {}).get("N", {}).get("identity"),
             "native_channel_identity_mismatch")
    documents, document_provenance = _vectors(dataset, adapter, "document", cache_dir)
    queries, query_provenance = _vectors(dataset, adapter, "query", cache_dir)
    native, native_provenance = _ranking_cache(dataset, source, "N", cache_dir, budget)
    exclusions = [dataset.excluded_ids(query) for query in dataset.queries]
    ids = [item["id"] for item in dataset.corpus]
    reproduced = exact_dot_rankings(documents, queries, ids, top_k=budget, exclusions=exclusions)
    _require(all(_same_rankings(a, b) for a, b in zip(reproduced, native)), "native_768_ranking_cache_mismatch")
    database_path = base_run / "progress.sqlite3"
    _require(database_path.is_file(), "missing_base_result_database")
    baseline, specialist, specialist_rows = [], {"status": "unavailable", "reason_code": "specialist_not_requested"}, None
    with sqlite3.connect("file:" + str(database_path.resolve()) + "?mode=ro", uri=True) as db:
        db.execute("BEGIN")
        for mode in modes:
            baseline.append(_verify_cell(dataset, source, db, {"N": reproduced}, ("N",), mode, budget, rrf_k))
        if include_specialist:
            try:
                specialist_rows, evidence = _ranking_cache(dataset, source, "S", cache_dir, budget)
                spec = source.get("adapter_specs", {}).get("channels", {}).get("S", {})
                _require(re.fullmatch(r"[a-f0-9]{64}", spec.get("identity", "")), "missing_specialist_adapter_identity")
                proxy = SimpleNamespace(identity=spec["identity"], dimension=spec.get("dimension"))
                if spec.get("class") == "rag_benchmark.multimodal_colqwen.ColQwenAdapter":
                    proxy.rank = True
                _require(_channel_identity(dataset, "S", {"S": proxy}) == evidence["channel_identity"], "specialist_channel_identity_mismatch")
                validated = []
                for mode in modes:
                    validated.append(_verify_cell(dataset, source, db, {"S": specialist_rows}, ("S",), mode, budget, rrf_k))
                    validated.append(_verify_cell(dataset, source, db, {"N": reproduced, "S": specialist_rows}, ("N", "S"), mode, budget, rrf_k))
                baseline.extend(validated)
                specialist = {"status": "validated", **evidence, "adapter_identity": spec["identity"],
                              "config": _safe_model_config(spec.get("config", {}))}
            except (EvidenceUnavailable, ValueError, KeyError, TypeError) as exc:
                specialist_rows = None
                specialist = {"status": "unavailable", "reason_code": str(exc) if isinstance(exc, EvidenceUnavailable) else "invalid_specialist_evidence"}
    provenance = {"base_report_sha256": source_hash, "native_adapter": public_adapter_spec(adapter),
        "algorithm": {"sweep_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "shared_engine_source_sha256": hashlib.sha256(Path(__file__).with_name("multimodal.py").read_bytes()).hexdigest(),
            "numpy_version": np.__version__, "numeric_dtype": "float32"},
        "document_vectors": document_provenance, "query_vectors": query_provenance, "native_rankings": native_provenance,
        "baseline_768": baseline, "specialist": specialist}
    output = {"schema_version": 1, "experiment": SWEEP_VERSION, "status": "completed", "dataset": dataset.public_summary(),
        "cache_only": True, "execution_device": "cpu", "model_inference_calls": 0, "inference_seconds": None,
        "configuration": {"dimensions": list(dimensions), "baseline_dimension": 768, "budget_modes": modes,
            "channel_budget": budget, "rrf_k": rrf_k, "projection": "first_coordinates_then_float32_l2_normalize",
            "baseline_projection": "unmodified_original_768_vectors", "search": "exact_full_corpus_dot_product_id_stable_ties"},
        "provenance": provenance, "cells": [],
        "storage_scope": "Raw FP32 N-vector payload only; excludes model, process RSS, index overhead and specialist storage."}
    output["identity"] = stable_hash({"experiment": SWEEP_VERSION, "dataset": dataset.identity,
        "configuration": output["configuration"], "provenance": provenance})
    for dimension in dimensions:
        rows = reproduced if dimension == 768 else exact_dot_rankings(project_prefix(documents, dimension),
            project_prefix(queries, dimension), ids, top_k=budget, exclusions=exclusions)
        sources = {"N": rows, **({"S": specialist_rows} if specialist_rows is not None else {})}
        for channels in (("N",), ("N", "S")) if specialist_rows is not None else (("N",),):
            for mode in modes:
                _, summary = _evaluate(dataset, sources, channels, mode, budget, rrf_k)
                output["cells"].append({"status": "completed", "dimension": dimension, "channels": list(channels),
                    "budget_mode": mode, "query_count": len(dataset.queries), **summary,
                    "n_vector_storage_ratio": dimension / 768, "n_document_vector_bytes": len(documents) * dimension * 4,
                    "n_query_vector_bytes": len(queries) * dimension * 4})
    return output


def _write(output_dir: Path, result: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = ["# Cache-only EG2 dimension sweep", "", "Status: **" + result["status"] + "**.", "",
             "Fresh model inference calls: 0. No inference speed is inferred from cached embeddings.", ""]
    if result["status"] == "completed":
        lines += ["The unchanged 768-dimensional baseline was reproduced before evaluation. Prefixes search the full frozen corpus.",
            "", result["storage_scope"], "", "| Dimension | Method | Budget | Hit@5 | Recall@5 | nDCG@10 | N vector ratio |",
            "|---:|---|---|---:|---:|---:|---:|"]
        for cell in result["cells"]:
            m = cell["metrics"]
            lines.append(f"| {cell['dimension']} | {'+'.join(cell['channels'])} | {cell['budget_mode']} | {m['hit@5']:.6f} | {m['recall@5']:.6f} | {m['ndcg@10']:.6f} | {cell['n_vector_storage_ratio']:.3f} |")
        specialist = result["provenance"]["specialist"]
        lines += ["", "Specialist fusion: " + specialist["status"] + (" (" + specialist["reason_code"] + ")" if "reason_code" in specialist else "") + ".",
                  "", "Current array/ID hashes are newly observed integrity records. Legacy caches lack original independent vector checksums."]
    else:
        lines += ["Reason: `" + result["reason_code"] + "`. No dimension metrics were published."]
    for name, value in (("dimension-sweep.json", json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"),
                        ("dimension-sweep.md", "\n".join(lines) + "\n")):
        path = output_dir / name
        temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
        temporary.write_text(value)
        temporary.replace(path)


def run_dimension_sweep(dataset_dir: Path | str, base_run: Path | str, output_dir: Path | str, *,
                        cache_dir: Path | str = Path(".cache/multimodal"), dimensions=DIMENSIONS,
                        include_specialist: bool = True) -> dict:
    dimensions = tuple(dimensions)
    if not dimensions or len(dimensions) != len(set(dimensions)) or any(type(d) is not int or d not in DIMENSIONS for d in dimensions):
        raise ValueError("Choose a unique nonempty subset of 128,256,512,768")
    try:
        result = _execute(dataset_dir, Path(base_run), Path(cache_dir), dimensions, include_specialist)
    except EvidenceUnavailable as exc:
        result = {"schema_version": 1, "experiment": SWEEP_VERSION, "status": "unavailable", "reason_code": str(exc),
                  "cache_only": True, "model_inference_calls": 0, "inference_seconds": None, "cells": []}
    except (OSError, ValueError, KeyError, TypeError, sqlite3.DatabaseError):
        result = {"schema_version": 1, "experiment": SWEEP_VERSION, "status": "unavailable",
                  "reason_code": "invalid_or_incomplete_cache_evidence", "cache_only": True,
                  "model_inference_calls": 0, "inference_seconds": None, "cells": []}
    _write(Path(output_dir), result)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/multimodal"))
    parser.add_argument("--dimensions", default="128,256,512,768")
    parser.add_argument("--no-specialist", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = run_dimension_sweep(args.dataset, args.base_run, args.output, cache_dir=args.cache_dir,
            dimensions=tuple(int(d) for d in args.dimensions.split(",")), include_specialist=not args.no_specialist)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps({"status": result["status"], "completed_cells": len(result["cells"]),
                      **({"reason_code": result["reason_code"]} if "reason_code" in result else {})}))
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())

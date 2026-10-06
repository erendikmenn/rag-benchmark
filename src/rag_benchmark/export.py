"""Export only explicitly allowed aggregate data and record identifiers."""

import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import tempfile

METRICS = {"context_recall", "answer_em", "answer_token_f1"} | {
    f"{name}@{k}" for name in ("recall", "hit", "all_evidence", "mrr", "ndcg") for k in (1, 5, 10, 50)
}
TIMINGS = ("retrieval_s", "rerank_s", "generation_s", "index_preparation_s")
MODEL_NUMBERS = ("batch_size", "max_len", "max_tokens", "temperature", "seed", "timeout", "enable_thinking", "dimensions")
PATH_KEYS = ("path", "local_path", "cache_dir", "model_path", "base_url")
COUNTS = ("articles_format_json", "articles_format_markdown_char_ranges", "articles_web", "articles_wikipedia",
          "articles_with_valid_questions", "articles_without_question_file", "corpus_chunks", "declared_question_count_mismatches",
          "dev_questions", "input_articles", "parsed_articles", "raw_chunks", "raw_questions", "rejected_articles",
          "rejected_questions", "rejected_chunks", "test_questions", "valid_questions")


def _hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _numbers(value, keys=None):
    return {k: v for k, v in (value or {}).items() if (keys is None or k in keys)
            and isinstance(v, (int, float)) and math.isfinite(v)}


def _label(value):
    # IDs and repository names are metadata, never free-form text or paths.
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.:#/@+\-]{0,255}", value) or ".." in value:
        raise ValueError("Unsupported public identifier; refusing to export free-form text or a path")
    if "/" in value and not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", value):
        raise ValueError("Unsupported public repository identifier")
    return value


def _fields(data, keys):
    return {key: None if data[key] is None else _label(data[key]) for key in keys if key in data}


def _hashes(data):
    return {_label(k): v for k, v in (data or {}).items()
            if isinstance(v, str) and re.fullmatch(r"[0-9a-f]{40,64}", v)}


def _model(data):
    result = _fields(data, ("model_id", "revision", "model", "filename", "device", "dtype", "backend"))
    result.update(_hashes({k: data[k] for k in ("model_sha256",) if k in data}))
    result.update(_numbers(data, MODEL_NUMBERS))
    result.update({key: "<local>" for key in PATH_KEYS if key in data})
    return result


def _stats(data):
    result = _numbers(data, ("n", "complete", "generation_cache_hits", "length_limited_answers"))
    result["metrics"] = _numbers(data.get("metrics"), METRICS)
    for stage in TIMINGS:
        if stage in data:
            result[stage] = _numbers(data[stage], ("n", "p50", "p95")) if data[stage] else None
    for field, allowed in (("by_source", {"web", "wikipedia", "unknown"}),
                           ("by_category", {"FACTUAL", "INTERPRETATION", "unknown"})):
        if field in data:
            result[field] = {key: _stats(value) for key, value in data[field].items() if key in allowed}
    return result


def _summary(data):
    result = {"split": _label(data["split"]), **_numbers(data, ("retrieval_only", "expected_per_variant")),
              "variants": {_label(k): _stats(v) for k, v in data["variants"].items()}, "paired_laya_deltas": {}}
    for method, value in data.get("paired_laya_deltas", {}).items():
        comparison = {**_numbers(value, ("paired_questions", "complete_pair")), "metrics": {}}
        for metric, stats in value.get("metrics", {}).items():
            if metric in METRICS:
                clean = _numbers(stats, ("delta",))
                interval = stats.get("article_bootstrap_ci95", [])
                if len(interval) == 2 and all(isinstance(x, (int, float)) and math.isfinite(x) for x in interval):
                    clean["article_bootstrap_ci95"] = interval
                comparison["metrics"][metric] = clean
        result["paired_laya_deltas"][_label(method)] = comparison
    return result


def _row(row):
    result = _fields(row, ("variant", "query_id", "article_id", "category", "source"))
    for key in ("gold_ids", "candidate_ids", "ranked_ids", "context_ids"):
        result[key] = [_label(value) for value in row.get(key, [])]
    for stage in ("candidate", "ranked"):
        key, type_key = f"{stage}_scores", f"{stage}_score_type"
        if key in row:
            values = row[key]
            if (not isinstance(values, list) or len(values) != len(result[f"{stage}_ids"])
                    or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values)):
                raise ValueError("Public score arrays must contain finite numbers aligned with their IDs")
            if row.get(type_key) not in {"bm25", "cosine_similarity", "rrf", "laya_p_true"}:
                raise ValueError("Unsupported public score type")
            result[key], result[type_key] = values, row[type_key]
    result["metrics"] = _numbers(row.get("metrics"), METRICS)
    result.update(_numbers(row, (*TIMINGS, "generation_cache_hit", "first_question_after_preparation", "first_fresh_generation")))
    for key in ("retrieval_usage", "reranker_usage", "generator_usage"):
        usage = row.get(key) or {}
        clean = _numbers(usage, ("candidate_pool", "corpus_size", "completion_tokens", "prompt_tokens", "total_tokens",
                                 "seconds", "cached", "batch_size", "pairs", "truncated_pairs", "max_len", "candidates", "retained"))
        for name, allowed in (("method", {"bm25", "bge", "embeddinggemma", "bm25_bge", "bm25_embeddinggemma"}),
                              ("search", {"exact"}), ("finish_reason", {"stop", "length", "eos", "tool_calls"}),
                              ("backend", {"sdk", "transformers"}), ("dtype", {"float32", "float16", "bfloat16"})):
            if usage.get(name) in allowed:
                clean[name] = usage[name]
        for name, allowed in (("query_cache_hits", ("bge", "embeddinggemma")), ("prompt_tokens_details", ("cached_tokens",)),
                              ("timings", ("cache_n", "prompt_n", "prompt_ms", "prompt_per_token_ms", "prompt_per_second",
                                           "predicted_n", "predicted_ms", "predicted_per_token_ms", "predicted_per_second"))):
            if name in usage:
                clean[name] = _numbers(usage[name], allowed)
        result[key] = clean
    return result


def _provenance(manifest):
    dataset, config, runtime = (manifest.get(key, {}) for key in ("dataset", "config", "runtime"))
    data = _fields(dataset, ("dataset", "revision", "fingerprint", "dataset_license"))
    data.update({key: _numbers(dataset.get(key), allowed) for key, allowed in (
        ("counts", COUNTS), ("category_counts", ("FACTUAL", "INTERPRETATION")), ("source_question_counts", ("web", "wikipedia")))})
    data["file_sha256"] = _hashes(dataset.get("file_sha256"))
    raw = dataset.get("raw_files", {})
    data["raw_files"] = {**_numbers(raw, ("count", "bytes")), **_hashes({"inventory_sha256": raw.get("inventory_sha256")})}
    data["split"] = _numbers(dataset.get("split"), ("seed", "requested_dev_questions", "dev_questions", "test_questions", "groups", "largest_group_questions", "near_duplicate_guarantee"))
    settings = {"dataset": {**_numbers(config.get("dataset"), ("seed", "dev_size")), "path": "<local>"},
                "experiment": _numbers(config.get("experiment"), ("candidates", "context_k")),
                "embeddings": {_label(k): _model(v) for k, v in config.get("embeddings", {}).items()},
                "laya": _model(config.get("laya", {})), "generator": _model(config.get("generator", {}))}
    code = _fields(runtime, ("python", "platform", "source_hash", "git_revision"))
    code["packages"] = _fields(runtime.get("packages", {}), ("rag-benchmark", "numpy", "bm25s", "torch", "sentence-transformers", "transformers", "laya"))
    server = manifest.get("generator_server") or {}
    props = server.get("props", {})
    generator = {**_hashes({k: server[k] for k in ("model_sha256", "runtime_fingerprint") if k in server}),
                 **_fields(props, ("build_info",)), "model_path": "<local>",
                 **_numbers(props, ("total_slots",))}
    defaults = props.get("default_generation_settings", {})
    generator["defaults"] = {**_numbers(defaults, ("n_ctx",)), "params": _numbers(defaults.get("params"),
        ("seed", "temperature", "top_k", "top_p", "min_p", "repeat_penalty", "max_tokens", "n_predict", "ignore_eos"))}
    if isinstance(props.get("chat_template"), str):
        generator["chat_template_sha256"] = hashlib.sha256(props["chat_template"].encode()).hexdigest()
    return {"schema_version": 1, "run_fingerprint": _label(manifest["fingerprint"]), "dataset": data,
            "runtime": code, "settings": settings, "generator_server": generator,
            "split": _label(manifest["split"]), "variants": [_label(x) for x in manifest["variants"]],
            "retrieval_only": bool(manifest["retrieval_only"]), "question_count": len(manifest["question_ids"]),
            **_hashes({key: manifest.get(key) for key in ("corpus_sha256", "questions_sha256")})}


def export_report(run_dir: Path, output_dir: Path) -> dict:
    """Create a new, text-free public export; never overwrite an existing report."""
    run_dir, output_dir = Path(run_dir).resolve(), Path(output_dir).resolve()
    inputs = ("manifest.json", "summary.json", "predictions.jsonl")
    input_hashes = {name: _hash(run_dir / name) for name in inputs}
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    protected = [run_dir]
    data_path = manifest.get("config", {}).get("dataset", {}).get("path")
    if data_path:
        protected.append(Path(data_path).resolve().parent)
    if any(output_dir == path or output_dir in path.parents or path in output_dir.parents for path in protected):
        raise ValueError("Export directory overlaps a run or dataset directory")
    if output_dir.exists():
        raise FileExistsError("Export requires a new output directory; existing files are never overwritten")
    summary = _summary(json.loads((run_dir / "summary.json").read_text(encoding="utf-8")))
    provenance = _provenance(manifest)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".rag-export-", dir=output_dir.parent))
    try:
        rows = 0
        with (run_dir / "predictions.jsonl").open(encoding="utf-8") as source, (temporary / "per-query-metrics.jsonl").open("w", encoding="utf-8") as target:
            for line in source:
                if line.strip():
                    target.write(json.dumps(_row(json.loads(line)), ensure_ascii=False, sort_keys=True) + "\n")
                    rows += 1
        provenance["exported_rows"] = rows
        if input_hashes != {name: _hash(run_dir / name) for name in inputs}:
            raise RuntimeError("Run files changed during export; retry after reporting finishes")
        provenance["local_input_sha256"] = input_hashes
        lines = ["# RAG benchmark export", "", f"Split: **{summary['split']}**. Exported rows: **{rows}**.", "",
                 "Pilot and partial runs are not final benchmark claims. Completeness is relative to the selected question set.", "",
                 "| Variant | Questions | Complete | Recall@5 | nDCG@10 | Answer EM | Token F1 |", "|---|---:|---|---:|---:|---:|---:|"]
        for name, stats in summary["variants"].items():
            scores = [f"{stats['metrics'][key]:.4f}" if key in stats["metrics"] else "—" for key in ("recall@5", "ndcg@10", "answer_em", "answer_token_f1")]
            lines.append(f"| {name} | {stats.get('n', 0)} | {stats.get('complete', False)} | " + " | ".join(scores) + " |")
        lines += ["", "Answer scores measure lexical agreement, not factual correctness. Synthetic labels and source-quality limitations apply.", "",
                  "The JSON files contain provenance, aggregate breakdowns, paired intervals, and per-query IDs/metrics. Source text, questions, reference answers, generated answers, prompts, and local paths are excluded.", "",
                  "Optional candidate_scores and ranked_scores align with their respective ID arrays. Score types: bm25 = lexical BM25 score; cosine_similarity = normalized dense-vector dot product; rrf = reciprocal-rank fusion score; laya_p_true = Laya's relevance P(true) model output, not a calibrated probability. Higher scores rank first. Different score types are not directly comparable; older runs may omit scores."]
        (temporary / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        for name, value in (("summary.json", summary), ("provenance.json", provenance)):
            (temporary / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.rename(output_dir)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    return {"exported_rows": rows, "run_fingerprint": provenance["run_fingerprint"],
            "files": {p.name: _hash(p) for p in sorted(output_dir.iterdir())}}

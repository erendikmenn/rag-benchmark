"""Fixed-budget ablations with transactional resumption and explicit provenance."""

from collections import defaultdict
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import math
from numbers import Real
import os
from pathlib import Path
import platform
import subprocess
import time
import tomllib

import numpy as np

from .metrics import answer_metrics, retrieval_metrics, strip_source_citations
from .storage import Store, atomic_json, digest

METHODS = ("bm25", "bge", "embeddinggemma", "bm25_bge", "bm25_embeddinggemma")
VARIANTS = tuple(v for method in METHODS for v in (method, method + "_laya"))


def _record_scores(documents):
    """Capture aligned model scores without accepting strings or nonfinite values."""
    values = [document["score"] for document in documents]
    if any(isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) for value in values):
        raise ValueError("Candidate/ranked scores must be finite numbers")
    return [float(value) for value in values]


def read_config(path: Path) -> dict:
    path = path.resolve()
    config = tomllib.loads(path.read_text(encoding="utf-8"))
    root = path.parent.parent
    def resolve(value):
        p = Path(value).expanduser()
        return str(p if p.is_absolute() else root / p)
    config["dataset"]["path"] = resolve(config["dataset"]["path"])
    config["experiment"]["cache_dir"] = resolve(config["experiment"]["cache_dir"])
    for model in [*config.get("embeddings", {}).values(), config.get("laya", {}), config.get("generator", {})]:
        for name in ("cache_dir", "local_path", "path", "model_path"):
            if name in model:
                model[name] = resolve(model[name])
    candidates = config["experiment"]["candidates"]
    context_k = config["experiment"]["context_k"]
    if not 1 <= context_k <= candidates:
        raise ValueError("Require 1 <= context_k <= candidates")
    if candidates < 50:
        raise ValueError("Main protocol requires >=50 candidates for Recall@50")
    return config


def runtime_identity():
    versions = {}
    for package in ("rag-benchmark", "numpy", "bm25s", "torch", "sentence-transformers", "transformers", "laya"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    source = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        revision = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": versions, "source_hash": digest(source), "git_revision": revision}


def choose_questions(questions, limit, seed):
    if limit is None:
        return sorted(questions, key=lambda q: digest([seed, q["id"]]))
    if limit < 1:
        raise ValueError("limit must be positive")
    # Interleave source/category groups for an informative small pilot.
    groups = defaultdict(list)
    for q in questions:
        groups[(q.get("source", "unknown"), q.get("category", "unknown"))].append(q)
    for group in groups.values():
        group.sort(key=lambda q: digest([seed, q["id"]]))
    ordered, offset = [], 0
    while len(ordered) < min(limit, len(questions)):
        for key in sorted(groups):
            if offset < len(groups[key]) and len(ordered) < limit:
                ordered.append(groups[key][offset])
        offset += 1
    return ordered


def run(config, run_dir: Path, split="dev", limit=None, variants=None, retrieval_only=False):
    # Downloads must happen in a separate, explicit preparation command.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from .data import load_dataset
    from .models import LayaReranker, LocalGenerator
    from .retrieval import RetrievalIndex

    variants = list(variants or config["experiment"].get("variants", VARIANTS))
    if len(variants) != len(set(variants)) or set(variants) - set(VARIANTS):
        raise ValueError(f"Choose distinct variants from {VARIANTS}")
    if split not in ("dev", "test"):
        raise ValueError("split must be dev or test")
    run_dir.mkdir(parents=True, exist_ok=True)
    lock = (run_dir / ".lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock.close()
        raise RuntimeError("Another process is already writing this run directory") from exc
    store = cache = None
    try:
        data_dir = Path(config["dataset"]["path"])
        corpus, questions, data_manifest = load_dataset(data_dir, split=split)
        questions = choose_questions(questions, limit, config["dataset"]["seed"])
        if not questions or not corpus:
            raise ValueError("Dataset has no corpus or evaluation questions")
        generator = None if retrieval_only else LocalGenerator(config["generator"])
        server_identity = generator.preflight() if generator is not None else None
        identity = {
            "config": config, "dataset": data_manifest, "runtime": runtime_identity(),
            "corpus_sha256": hashlib.sha256((data_dir / "corpus.jsonl").read_bytes()).hexdigest(),
            "questions_sha256": hashlib.sha256((data_dir / f"questions.{split}.jsonl").read_bytes()).hexdigest(),
            "split": split, "question_ids": [q["id"] for q in questions],
            "variants": variants, "retrieval_only": retrieval_only,
            "generator_server": server_identity,
        }
        # Documentation-only commits do not change the executable experiment.
        # Keep the Git revision as provenance, while code-content hashes govern resume.
        fingerprint = digest({**identity, "runtime": {k: v for k, v in identity["runtime"].items() if k != "git_revision"}})
        manifest_path = run_dir / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest["fingerprint"] != fingerprint:
                raise ValueError("Run identity changed; use a new --run-dir (config/data/code/environment must match)")
        else:
            manifest = {"fingerprint": fingerprint, "created_at": datetime.now(timezone.utc).isoformat(), **identity}
            atomic_json(manifest_path, manifest)
        cache_dir = Path(config["experiment"]["cache_dir"])
        store = Store(run_dir / "results.sqlite")
        cache = Store(cache_dir / "answers.sqlite")
        index = RetrievalIndex(corpus, cache_dir=cache_dir / "indexes", embeddings=config.get("embeddings", {}))
        reranker = None
        fresh_generation_seen = False
        started = time.perf_counter()
        for variant in variants:
            with_laya = variant.endswith("_laya")
            method = variant.removesuffix("_laya")
            pending = [q for q in questions if store.get(variant, q["id"]) is None]
            if not pending:
                print(f"{variant}: already complete", flush=True)
                continue
            if with_laya and reranker is None:
                reranker = LayaReranker(config["laya"])
            build_started = time.perf_counter()
            index.build(method)
            build_seconds = time.perf_counter() - build_started
            print(f"{variant}: {len(pending)} pending questions; index preparation {build_seconds:.2f}s", flush=True)
            for i, q in enumerate(pending, 1):
                tick = time.perf_counter()
                candidates = index.search(q["question"], method=method, top_k=config["experiment"]["candidates"], cache_query=False)
                retrieval_s = time.perf_counter() - tick
                original_ids = [c["id"] for c in candidates]
                original_scores = _record_scores(candidates)
                candidate_score_type = "bm25" if method == "bm25" else "rrf" if method.startswith("bm25_") else "cosine_similarity"
                tick = time.perf_counter()
                ranked = reranker.rerank(q["question"], candidates) if with_laya else candidates
                rerank_s = time.perf_counter() - tick if with_laya else 0.0
                if set(c["id"] for c in ranked) != set(original_ids) or len(ranked) != len(candidates):
                    raise ValueError("Reranker changed the candidate set")
                contexts = ranked[:config["experiment"]["context_k"]]
                metrics = retrieval_metrics([c["id"] for c in ranked], q["gold_ids"])
                metrics["context_recall"] = len({c["id"] for c in contexts} & set(q["gold_ids"])) / len(set(q["gold_ids"]))
                row = {
                    "variant": variant, "query_id": q["id"], "category": q.get("category", "unknown"),
                    "source": q.get("source", "unknown"), "article_id": q["article_id"],
                    "question": q["question"], "reference_answers": q["answers"], "gold_ids": q["gold_ids"],
                    "candidate_ids": original_ids, "ranked_ids": [c["id"] for c in ranked],
                    "candidate_scores": original_scores, "ranked_scores": _record_scores(ranked),
                    "candidate_score_type": candidate_score_type,
                    "ranked_score_type": "laya_p_true" if with_laya else candidate_score_type,
                    "context_ids": [c["id"] for c in contexts], "metrics": metrics,
                    "retrieval_s": retrieval_s, "rerank_s": rerank_s, "generation_s": None,
                    "generation_cache_hit": False, "index_preparation_s": build_seconds,
                    "first_question_after_preparation": i == 1, "retrieval_usage": index.last_usage,
                    "reranker_usage": reranker.last_usage if with_laya else None,
                }
                if generator is not None:
                    key = digest({"generator": config["generator"], "server": server_identity, "source_hash": identity["runtime"]["source_hash"],
                                  "question": q["question"], "contexts": [{k: c.get(k) for k in ("id", "title", "text")} for c in contexts]})
                    cached = cache.cached_answer(key)
                    row["generation_cache_hit"] = cached is not None
                    if cached is None:
                        tick = time.perf_counter()
                        answer = generator.generate(q["question"], contexts)
                        row["generation_s"] = time.perf_counter() - tick
                        row["first_fresh_generation"] = not fresh_generation_seen
                        fresh_generation_seen = True
                        cache.save_answer(key, answer, generator.last_usage)
                        row["generator_usage"] = generator.last_usage
                    else:
                        answer = cached["answer"]
                        row["generator_usage"] = {**cached["usage"], "cached": True}
                    row["answer"] = answer
                    row["answer_for_scoring"] = strip_source_citations(answer, row["context_ids"])
                    metrics.update(answer_metrics(row["answer_for_scoring"], q["answers"]))
                store.put(row)
                if i == 1 or i % 25 == 0 or i == len(pending):
                    print(f"{variant}: {i}/{len(pending)} saved", flush=True)
            report(run_dir)
        atomic_json(run_dir / "completion.json", {"completed_at": datetime.now(timezone.utc).isoformat(), "session_wall_s": time.perf_counter() - started, "rows": len(store.rows())})
        error = run_dir / "error.json"
        if error.exists():
            error.unlink()
        return report(run_dir)
    except Exception as exc:
        atomic_json(run_dir / "error.json", {"type": type(exc).__name__, "message": str(exc), "at": datetime.now(timezone.utc).isoformat()})
        raise
    finally:
        if store:
            store.close()
        if cache:
            cache.close()
        lock.close()


def summarize(rows):
    result = {"n": len(rows), "metrics": {}}
    keys = sorted({k for row in rows for k in row["metrics"]})
    for key in keys:
        values = [r["metrics"][key] for r in rows if key in r["metrics"]]
        result["metrics"][key] = float(np.mean(values))
    result["generation_cache_hits"] = sum(r["generation_cache_hit"] for r in rows)
    result["length_limited_answers"] = sum(r.get("generator_usage", {}).get("finish_reason") == "length" for r in rows)
    for stage in ("retrieval_s", "rerank_s", "generation_s"):
        values = [r[stage] for r in rows if r.get(stage) is not None and not r.get("first_question_after_preparation", False)
                  and not (stage == "generation_s" and r.get("first_fresh_generation", False))]
        result[stage] = {"n": len(values), "p50": float(np.percentile(values, 50)), "p95": float(np.percentile(values, 95))} if values else None
    return result


def report(run_dir: Path):
    if not (run_dir / "results.sqlite").exists():
        raise ValueError("No results.sqlite in run directory")
    store = Store(run_dir / "results.sqlite")
    rows = store.rows()
    store.close()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["variant"]].append(row)
    result = {"split": manifest["split"], "retrieval_only": manifest["retrieval_only"], "expected_per_variant": len(manifest["question_ids"]), "variants": {}, "paired_laya_deltas": {}}
    for variant in manifest["variants"]:
        group = grouped[variant]
        summary = summarize(group)
        summary["complete"] = len(group) == len(manifest["question_ids"])
        for field in ("category", "source"):
            summary[f"by_{field}"] = {name: summarize([r for r in group if r[field] == name]) for name in sorted({r[field] for r in group})}
        result["variants"][variant] = summary
    # Resample article groups, not correlated questions from the same article.
    for method in METHODS:
        if method not in grouped or method + "_laya" not in grouped:
            continue
        left = {r["query_id"]: r for r in grouped[method]}
        right = {r["query_id"]: r for r in grouped[method + "_laya"]}
        ids = sorted(left.keys() & right.keys())
        if not ids:
            continue
        comparison = {"paired_questions": len(ids), "complete_pair": len(ids) == len(manifest["question_ids"]), "metrics": {}}
        for metric in ("recall@5", "ndcg@10", "answer_em", "answer_token_f1"):
            if metric not in left[ids[0]]["metrics"]:
                continue
            clusters = defaultdict(list)
            for qid in ids:
                clusters[left[qid]["article_id"]].append(right[qid]["metrics"][metric] - left[qid]["metrics"][metric])
            sums = np.array([sum(v) for v in clusters.values()])
            counts = np.array([len(v) for v in clusters.values()])
            rng = np.random.default_rng(42)
            means = []
            for _ in range(1000):
                sample = rng.integers(0, len(sums), len(sums))
                means.append(float(sums[sample].sum() / counts[sample].sum()))
            comparison["metrics"][metric] = {"delta": float(sums.sum() / counts.sum()), "article_bootstrap_ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]}
        result["paired_laya_deltas"][method] = comparison
    atomic_json(run_dir / "summary.json", result)
    with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    lines = ["# RAG benchmark results", "", f"Split: **{result['split']}**. Mode: **{'retrieval only' if result['retrieval_only'] else 'end to end'}**.",
             "", "Scores below are means in [0, 1]. Partial and pilot runs are not final benchmark claims.",
             "", "| Variant | Questions | Complete | Recall@5 | nDCG@10 | Answer EM | Token F1 |", "|---|---:|---|---:|---:|---:|---:|"]
    def fmt(metrics, key):
        return f"{metrics[key]:.4f}" if key in metrics else "—"
    for variant, v in result["variants"].items():
        m = v["metrics"]
        lines.append(f"| {variant} | {v['n']} | {v['complete']} | {fmt(m, 'recall@5')} | {fmt(m, 'ndcg@10')} | {fmt(m, 'answer_em')} | {fmt(m, 'answer_token_f1')} |")
    lines += ["", "Answer EM/token F1 are lexical agreement, not factuality or faithfulness. Interpretive answers need separate review.",
              "", "summary.json includes source/category breakdowns, paired article-bootstrap intervals, measured stage latency and generation-cache hits. Cached generations and the first question after each preparation/resumption are excluded from latency summaries. Raw timings remain in predictions.jsonl. Index preparation/model loading are separate. Known source citation markers are removed for lexical scoring; raw answers are preserved.",
              "", "When present, candidate_scores align with candidate_ids and ranked_scores align with ranked_ids. Score types: bm25 = lexical BM25 score; cosine_similarity = normalized dense-vector dot product; rrf = reciprocal-rank fusion score; laya_p_true = Laya's relevance P(true) model output, not a calibrated probability. Higher scores rank first; scores from different types are not directly comparable. Older runs may omit score arrays."]
    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result

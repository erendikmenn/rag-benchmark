"""CPU-only paired comparisons from completed, frozen multimodal result cells.

Confidence intervals resample source-connected query groups with replacement;
methods retain paired query observations. No source text or model inference is
used. Missing group provenance yields descriptive differences without an interval.
As a conservative reporting safeguard, intervals are also withheld when one source
group contains more than half the queries. This is not a formal statistical threshold.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import uuid

import numpy as np

from .models import stable_hash
from .multimodal import Dataset, load_dataset, paired_group_bootstrap

ANALYSIS_VERSION = "paired-source-components-v2"
DEFAULT_METRICS = ("hit@5", "ndcg@10")
MAX_CI_GROUP_FRACTION = 0.5
METHOD_REFERENCES = [
    {"title": "SciPy paired bootstrap documentation", "url": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html"},
    {"title": "Stata cluster bootstrap documentation", "url": "https://www.stata.com/support/faqs/statistics/bootstrap-with-panel-data/"},
]
_STRATEGIES = {
    "photo": "positive_media_components", "environment_audio": "positive_media_components",
    "video": "positive_media_components", "speech": "verified_transcript_components",
    "code": "positive_code_components", "document": "positive_document_components",
    "composed_image": "reference_and_target_components",
}
_DESCRIPTIONS = {
    "positive_media_components": "Queries sharing any positively labelled media item are connected; all queries in each connected component are resampled together.",
    "verified_transcript_components": "Declared transcript groups are verified against every positive recording's group_id; shared recordings and transcript groups connect queries.",
    "positive_code_components": "Queries sharing a positively labelled source function are connected; corpus IDs identify functions, not entire repositories.",
    "positive_document_components": "Positive pages map to corpus.metadata.doc_id. Queries touching the same source document are connected transitively across multi-document positives.",
    "reference_and_target_components": "Queries are connected by reused reference images and positively labelled target images, including reference-to-target reuse.",
}


class GroupingUnavailable(ValueError):
    """The frozen collection lacks sufficient source provenance for resampling."""


def source_groups(dataset: Dataset, query_ids: list[str] | None = None, *, strategy: str = "auto") -> tuple[dict[str, str], dict]:
    """Return source components without substituting independent query IDs."""
    strategy = _STRATEGIES.get(dataset.track) if strategy == "auto" else strategy
    if strategy not in _DESCRIPTIONS:
        raise ValueError("Unknown source-group strategy")
    if strategy != _STRATEGIES.get(dataset.track):
        raise ValueError("Source-group strategy must match the dataset track and its declared source units")
    identifiers = sorted(query_ids if query_ids is not None else [query["id"] for query in dataset.queries])
    if not identifiers or len(set(identifiers)) != len(identifiers):
        raise ValueError("Source grouping requires nonempty unique query IDs")
    queries = {query["id"]: query for query in dataset.queries}
    corpus = {item["id"]: item for item in dataset.corpus}
    if not set(identifiers) <= set(queries):
        raise ValueError("Unknown query ID requested for source grouping")
    parent = {identifier: identifier for identifier in identifiers}
    def find(identifier):
        while parent[identifier] != identifier:
            parent[identifier] = parent[parent[identifier]]
            identifier = parent[identifier]
        return identifier
    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parent[max(left, right)] = min(left, right)
    first_query, associations = {}, 0
    for identifier in identifiers:
        query = queries[identifier]
        positives = [did for did, relevance in dataset.qrels[identifier].items() if relevance > 0]
        if not positives:
            raise GroupingUnavailable("A selected query has no positive source association")
        sources = set()
        if strategy == "verified_transcript_components":
            declared_group = query.get("metadata", {}).get("group_id")
            if not isinstance(declared_group, (str, int)) or not str(declared_group):
                raise GroupingUnavailable("A speech query lacks a declared transcript group")
            sources.add("transcript:" + str(declared_group))
        for did in positives:
            item = corpus[did]
            if strategy == "positive_document_components":
                document_id = item.get("metadata", {}).get("doc_id")
                if not isinstance(document_id, (str, int)) or not str(document_id):
                    raise GroupingUnavailable("Positive document pages lack source doc_id; query IDs cannot substitute for document groups")
                sources.add("document:" + str(document_id))
            elif strategy == "verified_transcript_components":
                if str(item.get("metadata", {}).get("group_id", "")) != str(declared_group):
                    raise GroupingUnavailable("Positive recording and query transcript-group provenance disagree")
                sources.add("media:" + did)
            elif strategy == "positive_code_components":
                sources.add("function:" + did)
            else:
                if not item.get("media"):
                    raise GroupingUnavailable("Positive media items lack a declared media source")
                sources.add("media:" + did)
        if strategy == "reference_and_target_components":
            reference = query.get("metadata", {}).get("reference_id", query.get("metadata", {}).get("reference_image_id"))
            if reference is None or str(reference) not in corpus:
                raise GroupingUnavailable("Composed queries need a declared reference image in the frozen corpus")
            sources.add("media:" + str(reference))
        associations += len(sources)
        for source in sorted(sources):
            if source in first_query:
                union(identifier, first_query[source])
            else:
                first_query[source] = identifier
    components = defaultdict(list)
    for identifier in identifiers:
        components[find(identifier)].append(identifier)
    groups = {}
    for members in components.values():
        group = stable_hash({"strategy": strategy, "members": sorted(members)})
        groups.update({identifier: group for identifier in members})
    sizes = [len(members) for members in components.values()]
    summary = {"status": "available", "strategy": strategy, "description": _DESCRIPTIONS[strategy],
               "query_count": len(identifiers), "group_count": len(sizes), "source_unit_count": len(first_query),
               "source_association_count": associations, "smallest_group_queries": min(sizes),
               "largest_group_queries": max(sizes), "largest_group_fraction": max(sizes) / len(identifiers),
               "mapping_sha256": stable_hash(groups), "fallback_to_independent_queries": False,
               "resampling_assumption": "Identified source components are treated as exchangeable resampling units; unrecorded cross-component dependencies may remain."}
    return groups, summary


def _select_cell(report: dict, selector: str) -> dict:
    matches = [cell for cell in report.get("cells", [])
               if cell.get("identity") == selector or cell.get("variant_id") == selector]
    matches = [cell for cell in matches if cell.get("primary") and cell.get("status") == "completed"]
    if len(matches) != 1:
        raise ValueError("Selector must identify exactly one completed primary cell")
    return matches[0]


def default_comparisons(report: dict) -> list[dict]:
    """Bounded intended contrasts; never an uncontrolled all-pairs matrix."""
    variants = {cell["variant_id"] for cell in report.get("cells", [])
                if cell.get("primary") and cell.get("status") == "completed"}
    track = report["dataset"]["track"]
    pairs = (("n", "s"), ("n", "n_s"), ("s", "n_s"), ("n", "j"),
             ("b", "g"), ("b", "e"), ("g", "e"), ("b", "b_g_e"))
    result = []
    for left, right in pairs:
        left, right = f"{track}__{left}__none", f"{track}__{right}__none"
        if left in variants and right in variants:
            result.append({"left": left, "right": right})
    return result


def _metric_rows(database: sqlite3.Connection, cell: dict, query_ids: set[str], metrics: tuple[str, ...]) -> dict[str, dict[str, float]]:
    rows = {}
    for query_id, payload in database.execute("SELECT query_id,payload FROM results WHERE cell=? ORDER BY query_id", (cell["identity"],)):
        record = json.loads(payload)
        if record.get("query_id") != query_id or query_id in rows:
            raise ValueError("Stored query IDs do not match result payloads")
        values = record.get("metrics", {})
        if any(key not in values or not isinstance(values[key], (int, float)) or not math.isfinite(values[key])
               or not 0 <= values[key] <= 1 for key in metrics):
            raise ValueError("Stored retrieval metrics are missing, nonfinite, or outside [0,1]")
        rows[query_id] = {key: float(values[key]) for key in metrics}
    if set(rows) != query_ids or len(rows) != cell.get("completed_queries") or len(rows) != cell.get("query_count"):
        raise ValueError("Completed cell does not contain the exact frozen query set; no intersection-only comparison is allowed")
    for metric in metrics:
        aggregate = float(np.mean([values[metric] for values in rows.values()]))
        if metric not in cell.get("metrics", {}) or not math.isclose(aggregate, cell["metrics"][metric], abs_tol=1e-10):
            raise ValueError("Per-query metrics do not reproduce the completed report aggregate")
    return rows


def _cell_summary(cell: dict) -> dict:
    return {key: cell[key] for key in ("identity", "variant_id", "budget_mode", "candidate_k", "query_count")}


def analyze_run(dataset_dir: Path | str, run_dir: Path | str, output_dir: Path | str | None = None, *,
                comparisons: list[dict] | None = None, metrics=DEFAULT_METRICS,
                samples: int = 5000, seed: int = 42, group_strategy: str = "auto") -> dict:
    """Measure right-minus-left differences from existing completed primary cells.

    ``comparisons=[{'left': variant_or_cell_id, 'right': variant_or_cell_id}]``
    selects arbitrary methods from the same frozen report. Metrics are evaluated
    separately. Intervals are marginal percentile intervals, without multiplicity
    adjustment; no significance badges or generated-answer accuracy are produced.
    """
    dataset, run_dir = load_dataset(dataset_dir), Path(run_dir)
    report_bytes = (run_dir / "report.json").read_bytes()
    report = json.loads(report_bytes)
    if report.get("dataset", {}).get("identity") != dataset.identity:
        raise ValueError("Analysis dataset fingerprint differs from the frozen run")
    if report.get("scope") != "frozen_collection":
        raise ValueError("Benchmark confidence intervals require a full frozen collection, not a smoke subset")
    query_ids = {query["id"] for query in dataset.queries}
    if set(report.get("configuration", {}).get("query_ids", [])) != query_ids or report.get("evaluated_query_count") != len(query_ids):
        raise ValueError("Run query coverage differs from the full frozen dataset")
    metrics = tuple(metrics)
    if not metrics or len(set(metrics)) != len(metrics) or any(not re.fullmatch(r"(?:hit|recall|ndcg|mrr|map)@[1-9][0-9]*", key) for key in metrics):
        raise ValueError("Choose unique labelled retrieval metrics with explicit cutoffs")
    if not isinstance(samples, int) or samples < 1 or not isinstance(seed, int) or seed < 0:
        raise ValueError("Bootstrap samples must be positive and seed nonnegative")
    requested = default_comparisons(report) if comparisons is None else comparisons
    try:
        groups, grouping = source_groups(dataset, sorted(query_ids), strategy=group_strategy)
    except GroupingUnavailable as error:
        groups = None
        grouping = {"status": "unavailable", "reason": str(error), "fallback_to_independent_queries": False}
    interval_reason = None
    if groups is None:
        interval_reason = "Source grouping is unavailable"
    elif grouping["group_count"] < 2:
        interval_reason = "Only one connected source group; a resampling confidence interval is not reported"
    elif grouping["largest_group_fraction"] > MAX_CI_GROUP_FRACTION:
        interval_reason = (
            f"Largest connected source group contains {grouping['largest_group_fraction']:.1%} of queries "
            f"(>{MAX_CI_GROUP_FRACTION:.0%}); interval withheld under a conservative reporting safeguard, "
            "not a formal statistical threshold"
        )
    result = {"schema_version": 1, "analysis_version": ANALYSIS_VERSION,
              "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "numpy_version": np.__version__,
              "dataset": dataset.public_summary(), "source_report_sha256": hashlib.sha256(report_bytes).hexdigest(),
              "scope": "paired_primary_cells_same_frozen_collection", "metrics": list(metrics),
              "bootstrap": {"method": "paired_source_group_percentile", "confidence_level": 0.95,
                            "samples": samples, "seed": seed, "direction": "right_minus_left",
                            "estimator": "query-weighted mean difference; groups resampled together",
                            "interval_reporting": "eligible" if interval_reason is None else "withheld",
                            "withholding_reason": interval_reason,
                            "dominant_group_guard": {
                                "maximum_group_fraction": MAX_CI_GROUP_FRACTION,
                                "comparison": "strictly_greater_than",
                                "interpretation": "conservative reporting safeguard, not a formal statistical threshold"},
                            "multiple_comparison_adjustment": "none; intervals are marginal exploratory comparisons"},
              "grouping": grouping, "comparisons": [], "method_references": METHOD_REFERENCES,
              "model_inference_performed": False, "generated_answer_accuracy_measured": False}
    database_path = (run_dir / "progress.sqlite3").resolve()
    if not database_path.is_file():
        raise FileNotFoundError("Completed per-query progress.sqlite3 is required")
    database = sqlite3.connect(database_path.as_uri() + "?mode=ro", uri=True)
    database.execute("BEGIN")
    cache = {}
    try:
        for requested_pair in requested:
            if set(requested_pair) != {"left", "right"} or any(not isinstance(value, str) for value in requested_pair.values()):
                raise ValueError("Each comparison must contain left and right cell/variant selectors")
            if any(not re.fullmatch(r"(?:[a-z0-9_]+__[a-z0-9_]+__[a-z0-9_]+|[a-f0-9]{64})", value)
                   for value in requested_pair.values()):
                raise ValueError("Selectors must be canonical variant IDs or cell identity hashes, not arbitrary text")
            comparison = {"status": "unavailable", "requested_left": requested_pair["left"], "requested_right": requested_pair["right"]}
            result["comparisons"].append(comparison)
            try:
                left, right = (_select_cell(report, requested_pair[key]) for key in ("left", "right"))
                if left["identity"] == right["identity"]:
                    raise ValueError("A comparison requires two distinct completed cells")
                for cell in (left, right):
                    if cell["identity"] not in cache:
                        cache[cell["identity"]] = _metric_rows(database, cell, query_ids, metrics)
                comparison.update(left=_cell_summary(left), right=_cell_summary(right), metrics={})
                for metric in metrics:
                    left_values = {qid: values[metric] for qid, values in cache[left["identity"]].items()}
                    right_values = {qid: values[metric] for qid, values in cache[right["identity"]].items()}
                    delta = np.asarray([right_values[qid] - left_values[qid] for qid in sorted(query_ids)])
                    measured = {"left_mean": float(np.mean(list(left_values.values()))),
                                "right_mean": float(np.mean(list(right_values.values()))),
                                "mean_difference": float(delta.mean()), "query_count": len(delta),
                                "wins": int((delta > 0).sum()), "losses": int((delta < 0).sum()), "ties": int((delta == 0).sum())}
                    if interval_reason is None:
                        measured.update(paired_group_bootstrap(left_values, right_values, groups, samples=samples, seed=seed))
                    comparison["metrics"][metric] = measured
                comparison["status"] = "completed" if interval_reason is None else "descriptive_only"
                if interval_reason is not None:
                    comparison["reason"] = interval_reason
                    if groups is not None:
                        comparison["largest_group_fraction"] = grouping["largest_group_fraction"]
            except ValueError as error:
                comparison.update(status="unavailable", reason=str(error))
    finally:
        database.close()
    counts = Counter(comparison["status"] for comparison in result["comparisons"])
    result["comparison_status_counts"] = dict(counts)
    result["status"] = "no_comparable_cells" if not requested else "completed" if counts.get("completed") == len(requested) else "partial"
    if output_dir is not None:
        export_analysis(result, output_dir)
    return result


def export_analysis(report: dict, output_dir: Path | str) -> None:
    """Write aggregate JSON/Markdown only; no query, answer or source text."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "paired-comparisons.json"
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    def escape(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines = ["# Paired retrieval comparisons", "", f"Dataset: {escape(report['dataset']['id'])}.", "",
             "Differences are right minus left, in metric units. Positive values favor the right method. "
             "Where reported, intervals are 95% paired source-group percentile intervals; methods use the same frozen queries and gallery. "
             "These are marginal exploratory comparisons without a multiple-comparison correction.", "",
             f"Grouping: {escape(report['grouping'].get('description', report['grouping'].get('reason')))}", ""]
    if report["grouping"]["status"] == "available":
        lines.append(f"Source groups: {report['grouping']['group_count']}; queries: {report['grouping']['query_count']}; "
                     f"largest group: {report['grouping']['largest_group_queries']} queries "
                     f"({report['grouping']['largest_group_fraction']:.1%}).")
        lines.append("")
    guard = report["bootstrap"].get("dominant_group_guard")
    if guard:
        lines += [f"Conservative reporting safeguard: withhold intervals when one source group contains more than "
                  f"{guard['maximum_group_fraction']:.0%} of queries. This is not a formal statistical threshold.", ""]
    lines += ["| Left | Right | Metric | Left mean | Right mean | Delta | 95% interval | Wins / losses / ties |",
              "|---|---|---|---:|---:|---:|---|---|"]
    for comparison in report["comparisons"]:
        if "metrics" not in comparison:
            continue
        for metric, values in comparison["metrics"].items():
            interval = f"[{values['ci95_low']:+.4f}, {values['ci95_high']:+.4f}]" if "ci95_low" in values else "Not reported; see reason below"
            lines.append(f"| {escape(comparison['left']['variant_id'])} | {escape(comparison['right']['variant_id'])} | {metric} "
                         f"| {values['left_mean']:.4f} | {values['right_mean']:.4f} | {values['mean_difference']:+.4f} "
                         f"| {interval} | {values['wins']} / {values['losses']} / {values['ties']} |")
    for comparison in report["comparisons"]:
        if comparison.get("reason"):
            lines += ["", f"{escape(comparison['requested_left'])} → {escape(comparison['requested_right'])}: {escape(comparison['reason'])}."]
    lines += ["", f"Bootstrap settings for eligible comparisons: {report['bootstrap']['samples']} samples; seed: {report['bootstrap']['seed']}. "
              "No model inference or generated-answer judging was performed.", "",
              "Pairing follows [SciPy's paired resampling definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html); "
              "source groups follow [cluster resampling](https://www.stata.com/support/faqs/statistics/bootstrap-with-panel-data/).", ""]
    (output_dir / "paired-comparisons.md").write_text("\n".join(lines))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--comparisons", type=Path, help="JSON list of {left, right} variant or cell selectors")
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--group-strategy", default="auto", choices=["auto", *_DESCRIPTIONS])
    args = parser.parse_args(argv)
    result = analyze_run(args.dataset, args.run_dir, args.output,
        comparisons=json.loads(args.comparisons.read_text()) if args.comparisons else None,
        metrics=args.metrics.split(","), samples=args.samples, seed=args.seed, group_strategy=args.group_strategy)
    print(json.dumps({"status": result["status"], "comparisons": result["comparison_status_counts"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

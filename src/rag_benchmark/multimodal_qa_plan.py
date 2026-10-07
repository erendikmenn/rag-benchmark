"""CPU-only QA job manifest: controls once per dataset, retrieved-only per cell.

Does not launch inference or claim generation reuse. Only observed reports/cells
are enumerated; incomplete retrieval cells are pending, never invented future
results. Expected identity uses frozen prepared files + asset hash inventory;
the QA runner must verify actual media bytes and stored ranking metrics before
inference. Public manifest excludes raw questions, answers, query IDs and ranks.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import shlex
import sqlite3

from .models import stable_hash
from .multimodal_qa import prepare_qa, read_jsonl, sha256
from .multimodal_qa_runner import frozen_config


def expected_dataset_identity(dataset):
    dataset = Path(dataset)
    inventory = read_jsonl(dataset / "assets.jsonl")
    media = {row["path"]: row["sha256"] for row in inventory}
    if len(media) != len(inventory):
        raise ValueError("Duplicate frozen asset inventory")
    referenced = {
        path
        for name in ["corpus.jsonl", "queries.jsonl"]
        for row in read_jsonl(dataset / name)
        for path in row.get("media", {}).values()
    }
    if referenced != set(media):
        raise ValueError("Frozen inventory differs from referenced media")
    return stable_hash(
        {
            "files": {
                name: sha256(dataset / name)
                for name in ["dataset.json", "corpus.jsonl", "queries.jsonl", "qrels.jsonl"]
            },
            "media": media,
        }
    )


def build_plan(dataset_dirs, *, workspace, reports_root, runs_root, output_root,
               shared_generation_cache=None):
    workspace = Path(workspace).resolve()
    reports_root, runs_root, output_root = (
        Path(path).resolve() for path in [reports_root, runs_root, output_root]
    )

    def relative(path):
        return Path(path).resolve().relative_to(workspace).as_posix()

    shared_cache = None
    if shared_generation_cache is not None:
        path = Path(shared_generation_cache)
        shared_cache = relative(path if path.is_absolute() else workspace / path)

    datasets = {}
    jobs, exclusions = [], []
    for dataset in sorted(map(Path, dataset_dirs)):
        dataset = dataset.resolve()
        manifest = json.loads((dataset / "dataset.json").read_text())
        raw = sorted(
            (dataset.parent / "raw" / manifest["id"].rsplit("-", 1)[0] / "queries").glob("*.parquet")
        )
        _, readiness = prepare_qa(dataset, raw)
        if readiness["status"] != "ready_for_generation":
            exclusions.append({"dataset": manifest["id"], "reason": "QA source/gold coverage unavailable"})
            continue
        identifier = manifest["id"]
        if identifier in datasets:
            raise ValueError("Ambiguous duplicate frozen dataset ID")
        datasets[identifier] = {
            "dataset": dataset,
            "raw": raw,
            "readiness": readiness,
            "identity": expected_dataset_identity(dataset),
        }
    observations = defaultdict(list)
    report_sources = {}
    report_cache = {}
    coverage_cache = {}

    def read_report(path):
        if path not in report_cache:
            report_cache[path] = json.loads(path.read_text())
        return report_cache[path]

    for path in sorted({*reports_root.glob("*/report.json"), *runs_root.glob("*/report.json")}):
        report = read_report(path)
        data = report.get("dataset", {})
        if data.get("id") not in datasets:
            continue
        frozen = datasets[data["id"]]
        report_id = relative(path)
        report_sources[report_id] = sha256(path)
        if (
            report.get("scope") != "frozen_collection"
            or report.get("evaluated_query_count") != frozen["readiness"]["query_count"]
        ):
            exclusions.append({"report": report_id, "reason": "Smoke/partial-query report excluded"})
            continue
        if (
            data.get("identity") != frozen["identity"]
            or data.get("query_count") != frozen["readiness"]["query_count"]
        ):
            exclusions.append({"report": report_id, "reason": "Report frozen dataset identity/count differs"})
            continue
        local = path.parent if path.is_relative_to(runs_root) else runs_root / path.parent.name
        # Public milestones may have no matching private store; another exact-ID
        # observation can provide the real storage later in this grouping.
        available = (local / "report.json").is_file() and (local / "progress.sqlite3").is_file()
        local_report = read_report(local / "report.json") if available else None
        if available and (
            local_report.get("dataset", {}).get("identity") != frozen["identity"]
            or local_report.get("scope") != "frozen_collection"
            or local_report.get("evaluated_query_count") != frozen["readiness"]["query_count"]
            or local_report.get("dataset", {}).get("query_count") != frozen["readiness"]["query_count"]
        ):
            available = False
        for cell in report.get("cells", []):
            semantic = (data["id"], cell["variant_id"], cell["budget_mode"], cell["candidate_k"])
            observation = {
                "report": report_id,
                "report_sha256": report_sources[report_id],
                "cell": cell,
                "local": local,
                "storage_available": available,
            }
            observations[semantic].append(observation)
    pending = []
    completed_identities = {}
    for semantic, rows in sorted(observations.items(), key=lambda item: str(item[0])):
        frozen = datasets[semantic[0]]
        n = frozen["readiness"]["query_count"]
        eligible = [
            row
            for row in rows
            if row["cell"].get("status") == "completed"
            and row["cell"].get("query_count") == n
            and row["cell"].get("completed_queries") == n
            and isinstance(row["cell"].get("identity"), str)
            and row["cell"]["identity"]
        ]
        if not eligible:
            pending.append(
                {
                    "dataset": semantic[0],
                    "dataset_identity": frozen["identity"],
                    "variant_id": semantic[1],
                    "budget_mode": semantic[2],
                    "candidate_k": semantic[3],
                    "observed_status_counts": dict(Counter(row["cell"].get("status") for row in rows)),
                    "source_reports": sorted({row["report"] for row in rows}),
                    "retrieval_cell_identity": None,
                    "qa_status": "pending_retrieval_completion",
                    "future_retrieved_qa_tasks_if_completed": 2 * n,
                }
            )
            continue
        by_identity = defaultdict(list)
        for row in eligible:
            by_identity[row["cell"]["identity"]].append(row)
        for identity, matches in by_identity.items():
            signature = stable_hash({"semantic": semantic, "metrics": matches[0]["cell"].get("metrics")})
            if any(
                stable_hash({"semantic": semantic, "metrics": row["cell"].get("metrics")}) != signature
                for row in matches
            ):
                raise ValueError("Duplicate completed retrieval identity has conflicting metrics")
            if identity in completed_identities and completed_identities[identity]["signature"] != signature:
                raise ValueError("Retrieval identity is reused across different semantic cells")
            candidates = []
            for row in matches:
                if not row["storage_available"]:
                    continue
                local_report = read_report(row["local"] / "report.json")
                exact = [cell for cell in local_report.get("cells", []) if cell.get("identity") == identity]
                if (
                    len(exact) != 1
                    or exact[0].get("status") != "completed"
                    or exact[0].get("completed_queries") != n
                ):
                    continue
                database = row["local"] / "progress.sqlite3"
                if database not in coverage_cache:
                    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
                        coverage_cache[database] = dict(
                            connection.execute(
                                "SELECT cell,COUNT(DISTINCT query_id) FROM results GROUP BY cell"
                            ).fetchall()
                        )
                count = coverage_cache[database].get(identity, 0)
                if count == n:
                    candidates.append(row["local"])
            completed_identities[identity] = {
                "signature": signature,
                "semantic": semantic,
                "frozen": frozen,
                "sources": sorted({row["report"] for row in matches}),
                "local": sorted(set(candidates))[0] if candidates else None,
            }

    def job(frozen, conditions, identity=None, local=None, sources=None):
        identifier = frozen["readiness"]["dataset"]
        config = frozen_config()
        config["protocol"]["conditions"] = conditions
        label = "controls" if identity is None else "retrieved-" + identity
        directory = output_root / identifier / label
        command = [
            ".venv/bin/python",
            "-m",
            "rag_benchmark.multimodal_qa_runner",
            "--dataset",
            relative(frozen["dataset"]),
            "--expected-dataset-identity",
            frozen["identity"],
            "--raw-queries",
            *map(relative, frozen["raw"]),
            "--run-dir",
            relative(directory),
            "--conditions",
            *conditions,
            "--representations",
            "text",
            "image",
        ]
        if identity:
            command += [
                "--retrieval-run",
                relative(local),
                "--retrieval-cell",
                identity,
                "--expected-retrieval-report-sha256",
                sha256(local / "report.json"),
            ]
        if shared_cache is not None:
            command += ["--shared-generation-cache", shared_cache]
        return {
            "job_id": stable_hash(
                {
                    "dataset_identity": frozen["identity"],
                    "conditions": conditions,
                    "retrieval_cell_identity": identity,
                    "config_sha256": stable_hash(config),
                    "conditions_representations_config": {
                        "conditions": conditions,
                        "representations": ["text", "image"],
                        "source_budget": 5,
                    },
                }
            ),
            "kind": "controls_only" if identity is None else "retrieved_only",
            "dataset": identifier,
            "dataset_identity": frozen["identity"],
            "query_count": frozen["readiness"]["query_count"],
            "conditions": conditions,
            "representations": ["text", "image"],
            "planned_qa_tasks": frozen["readiness"]["query_count"] * len(conditions) * 2,
            "retrieval_cell_identity": identity,
            "retrieval_report": relative(local / "report.json") if local else None,
            "retrieval_report_sha256": sha256(local / "report.json") if local else None,
            "observed_source_reports": sources or [],
            "config_sha256": stable_hash(config),
            "conditions_representations_config": {
                "conditions": conditions,
                "representations": ["text", "image"],
                "source_budget": 5,
            },
            "source_manifest_sha256": frozen["readiness"]["manifest_sha256"],
            "raw_query_source_sha256": frozen["readiness"]["raw_query_source_sha256"],
            "run_directory": relative(directory),
            "status": "planned_not_executed",
            "command_argv": command,
            "command": shlex.join(command),
        }

    for frozen in datasets.values():
        jobs.append(job(frozen, ["closed_book", "oracle"]))
    for identity, cell in sorted(completed_identities.items()):
        if cell["local"] is None:
            pending.append(
                {
                    "dataset": cell["semantic"][0],
                    "retrieval_cell_identity": identity,
                    "qa_status": "pending_actual_ranking_storage",
                    "source_reports": cell["sources"],
                    "future_retrieved_qa_tasks_if_completed": 2 * cell["frozen"]["readiness"]["query_count"],
                }
            )
        else:
            jobs.append(job(cell["frozen"], ["retrieved"], identity, cell["local"], cell["sources"]))
    for report_path, digest in report_sources.items():
        if sha256(workspace / report_path) != digest:
            raise ValueError("Source report changed during planning; regenerate the snapshot")
    control_tasks = sum(row["planned_qa_tasks"] for row in jobs if row["kind"] == "controls_only")
    retrieved_tasks = sum(row["planned_qa_tasks"] for row in jobs if row["kind"] == "retrieved_only")
    return {
        "schema_version": 1,
        "status": "execution_plan_only",
        "model_requests_started": 0,
        "qa_generation_completed_by_plan": 0,
        "actual_generation_reuse_claimed": False,
        "shared_generation_cache": shared_cache,
        "generation_reuse_policy": "Exact request, source bytes and verified generation/runtime identity only; per-task metrics remain independent. Actual calls/reuse are measured by the runner, not predicted by this plan." if shared_cache else "Disabled; controls scheduled once per dataset.",
        "scope": "Observed source-verified document datasets and completed retrieval identities only; not all future variants or the primary18460matrix",
        "dataset_count": len(datasets),
        "controls_job_count": sum(row["kind"] == "controls_only" for row in jobs),
        "retrieved_job_count": sum(row["kind"] == "retrieved_only" for row in jobs),
        "schedulable_planned_qa_tasks": control_tasks + retrieved_tasks,
        "controls_planned_qa_tasks": control_tasks,
        "retrieved_planned_qa_tasks": retrieved_tasks,
        "pending_observed_retrieval_cells": len(pending),
        "pending_future_qa_tasks_if_sources_complete": sum(
            row["future_retrieved_qa_tasks_if_completed"] for row in pending
        ),
        "completion_denominator": "sum query_count × selected conditions ×2representations over scheduled jobs; completed tasks currently zero; pending retrieval is separately conditional",
        "execution_prerequisites": [
            "Runner actual media-byte checksum verification",
            "Runner exact frozen query/ranking coverage and score/metric validation",
            "Root-owned verified local Gemma server; no concurrent active model run",
        ],
        "planner_sha256": sha256(Path(__file__)),
        "runner_sha256": sha256(Path(__file__).with_name("multimodal_qa_runner.py")),
        "source_report_sha256": report_sources,
        "jobs": jobs,
        "pending_cells": pending,
        "excluded_reports": exclusions,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/multimodal"))
    parser.add_argument("--reports-root", type=Path, default=Path("reports/multimodal"))
    parser.add_argument("--runs-root", type=Path, default=Path("runs/multimodal"))
    parser.add_argument("--qa-runs-root", type=Path, default=Path("runs/multimodal-qa"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shared-generation-cache", type=Path,
                        help="Optional common private cache passed to every job; the runner requires a git-ignored path.")
    args = parser.parse_args(argv)
    result = build_plan(
        sorted(path.parent for path in args.dataset_root.glob("vidore-v3-*/dataset.json")),
        workspace=Path.cwd(),
        reports_root=args.reports_root,
        runs_root=args.runs_root,
        output_root=args.qa_runs_root,
        shared_generation_cache=args.shared_generation_cache,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in [
                    "status",
                    "dataset_count",
                    "controls_job_count",
                    "retrieved_job_count",
                    "schedulable_planned_qa_tasks",
                    "pending_observed_retrieval_cells",
                ]
            }
        )
    )


if __name__ == "__main__":
    main()

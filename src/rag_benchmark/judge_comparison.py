"""Validate blinded agent judgments and compare them with frozen Jev labels.

This module makes no model calls. Agent verdicts are observations, not human
ground truth. Public exports omit answer/evidence text and judge explanations.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from .semantic import RUBRICS, _load_bundle, _responses
from .storage import atomic_json, digest

SHARDS = ("correctness", "grounding_a", "grounding_b")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_agent_verdicts(agent_dir, benchmark_tasks):
    """Require complete coverage and reconstruct every permitted input state."""
    agent_dir = Path(agent_dir)
    manifest = read_json(agent_dir / "manifest.json")
    for name, expected in manifest["files"].items():
        if Path(name).name != name or sha256(agent_dir / name) != expected:
            raise ValueError("Frozen agent input changed")
    if manifest["rubric_fingerprint"] != digest(RUBRICS):
        raise ValueError("Judge rubrics differ")
    mapping_rows = read_json(agent_dir / "private-mapping.json")
    mapping = {row["task_id"]: row for row in mapping_rows}
    if len(mapping) != len(mapping_rows):
        raise ValueError("Duplicate task mapping")
    if {task_id for task_id, row in mapping.items() if row["scope"] == "benchmark"} != set(benchmark_tasks):
        raise ValueError("Agent benchmark tasks differ from the Jev bundle")
    inputs, verdicts = {}, {}
    output_hashes = {}
    for shard in SHARDS:
        pack = read_json(agent_dir / (shard + ".json"))
        kind = pack["kind"]
        if pack["rubric"] != {"type": "choice", **RUBRICS[kind]}:
            raise ValueError("Agent rubric was changed")
        shard_inputs = {}
        for group in pack["groups"]:
            for case in group["cases"]:
                task_id = case["task_id"]
                if task_id in inputs or task_id not in mapping:
                    raise ValueError("Duplicate or unmapped agent input")
                passages = [group["passages"][key] for key in case["passage_keys"]]
                state = {"question": group["question"], "answer": case["answer"]}
                field = "gold_passages" if kind == "correctness" else "context_passages"
                state[field] = passages
                if kind == "correctness":
                    state["reference_answers"] = case["reference_answers"]
                meta = mapping[task_id]
                if (digest(state) != case["input_sha256"] or meta["input_sha256"] != digest(state)
                        or meta["kind"] != kind or meta["shard"] != shard):
                    raise ValueError("Agent input identity does not match its mapping")
                if meta["scope"] == "benchmark":
                    original = benchmark_tasks[task_id]
                    if original["kind"] != kind or original["payload"]["state"] != state:
                        raise ValueError("Agent and Jev did not receive identical evidence")
                elif meta["scope"] != "control":
                    raise ValueError("Unknown evaluation scope")
                inputs[task_id] = {"kind": kind, "state": state, "scope": meta["scope"]}
                shard_inputs[task_id] = {"hash": digest(state), "sources": {p["source_id"] for p in passages}}
        verdict_path = agent_dir / (shard + ".verdicts.jsonl")
        judgments = read_jsonl(verdict_path)
        output_hashes[verdict_path.name] = sha256(verdict_path)
        seen = set()
        for row in judgments:
            task_id = row.get("task_id")
            if task_id not in shard_inputs or task_id in seen:
                raise ValueError("Unexpected or duplicate agent verdict")
            seen.add(task_id)
            expected = shard_inputs[task_id]
            if (row.get("input_sha256") != expected["hash"] or row.get("label") not in RUBRICS[kind]["criteria"]
                    or not isinstance(row.get("reason"), str) or not row["reason"].strip()
                    or not isinstance(row.get("needs_review"), bool)
                    or not isinstance(row.get("evidence_refs"), list)
                    or any(not isinstance(ref, str) or ref not in expected["sources"] for ref in row["evidence_refs"])):
                raise ValueError("Invalid agent verdict or evidence outside the case")
            verdicts[task_id] = row
        if seen != set(shard_inputs):
            raise ValueError("Agent evaluation incomplete; missing judgments cannot be dropped")
    if set(inputs) != set(mapping):
        raise ValueError("Agent pack coverage differs from the mapping")
    return manifest, inputs, mapping, verdicts, output_hashes


def agreement(pairs):
    """Agreement measures consistency between judges, never judge accuracy."""
    cells = Counter(pairs)
    n = len(pairs)
    return {
        "n": n,
        "agree": sum(left == right for left, right in pairs),
        "agreement_rate": sum(left == right for left, right in pairs) / n if n else None,
        "matrix": {left: dict(Counter(right for a, right in pairs if a == left)) for left in sorted({a for a, _ in pairs})},
        "cells": [{"jev": left, "sol": right, "count": count} for (left, right), count in sorted(cells.items())],
    }


def summarize_answer_rows(rows):
    n = len(rows)
    summary = {"answers": n, "unique_questions": len({r["query_id"] for r in rows})}
    for judge in ("jev", "sol"):
        summary[judge] = {
            kind + "_labels": dict(Counter(row[judge][kind] for row in rows))
            for kind in RUBRICS
        }
        correct = sum(row[judge]["correctness"] == "correct" for row in rows)
        joint = sum(row[judge] == {"correctness": "correct", "grounding": "supported"} for row in rows)
        summary[judge].update(correct_count=correct, correct_rate=correct / n if n else None,
                              correct_and_supported_count=joint,
                              correct_and_supported_rate=joint / n if n else None)
    summary["agreement"] = {
        kind: agreement([(row["jev"][kind], row["sol"][kind]) for row in rows]) for kind in RUBRICS
    }
    return summary


def compare_judges(jev_dir, agent_dir, output_dir, *, controls_path=None):
    jev_dir, agent_dir, output_dir = map(Path, (jev_dir, agent_dir, output_dir))
    if any(output_dir.resolve() == root.resolve() or root.resolve() in output_dir.resolve().parents
           or output_dir.resolve() in root.resolve().parents for root in (jev_dir, agent_dir)):
        raise ValueError("Comparison output overlaps private inputs")
    if controls_path is not None and (output_dir.resolve() == Path(controls_path).resolve()
                                     or output_dir.resolve() in Path(controls_path).resolve().parents):
        raise ValueError("Comparison output overlaps control input")
    bundle, tasks, answer_mapping = _load_bundle(jev_dir)
    tasks = {task["id"]: task for task in tasks}
    outcomes = _responses(jev_dir, bundle, list(tasks.values()))
    if len(outcomes) != len(tasks) or any(row["status"] != "success" for row in outcomes.values()):
        raise ValueError("The source Jev evaluation is incomplete")
    manifest, inputs, mapping, verdicts, verdict_hashes = validate_agent_verdicts(agent_dir, tasks)
    if manifest["source_bundle_fingerprint"] != bundle["fingerprint"]:
        raise ValueError("Comparison refers to another Jev bundle")
    public_rows = []
    for original in answer_mapping:
        row = {key: original[key] for key in ("variant", "query_id", "article_id", "source", "category")}
        row["jev"], row["sol"], row["sol_needs_review"] = {}, {}, {}
        row["task_ids"] = original["tasks"]
        for kind, task_id in original["tasks"].items():
            row["jev"][kind] = outcomes[task_id]["response"]["answers"]["decision"]["choice"]
            row["sol"][kind] = verdicts[task_id]["label"]
            row["sol_needs_review"][kind] = verdicts[task_id]["needs_review"]
        public_rows.append(row)
    summary = {
        "schema_version": 1, "complete": True,
        "interpretation": "Same saved answers, different automated judges. Agreement is not accuracy; no human adjudication has been performed.",
        "jev_model": bundle["model"], "jev_served_model": bundle["served_model"],
        "sol_model": manifest["requested_model"], "sol_reasoning_effort": manifest["requested_reasoning_effort"],
        "sol_execution_method": manifest["execution_method"],
        "agent_configuration": [{k: agent[k] for k in ("shard", "model", "reasoning_effort", "fork_turns")}
                                for agent in manifest.get("agents", [])],
        "sol_model_identity_scope": "Model and effort configured in Codex spawn_agent; a served snapshot, token usage and USD cost are not exposed by this workflow.",
        "source_run_fingerprint": bundle["run_fingerprint"], "source_jev_bundle_fingerprint": bundle["fingerprint"],
        "rubric_fingerprint": bundle["rubric_fingerprint"], "agent_input_files_sha256": manifest["files"],
        "agent_verdict_files_sha256": verdict_hashes, "comparison_code_sha256": sha256(Path(__file__)),
        "unique_benchmark_tasks": len(tasks), "unique_control_tasks": sum(r["scope"] == "control" for r in mapping.values()),
        "overall": summarize_answer_rows(public_rows),
        "by_variant": {v: summarize_answer_rows([r for r in public_rows if r["variant"] == v])
                       for v in sorted({r["variant"] for r in public_rows})},
        "unique_task_agreement": {
            kind: agreement([(outcomes[t]["response"]["answers"]["decision"]["choice"], verdicts[t]["label"])
                             for t in tasks if tasks[t]["kind"] == kind]) for kind in RUBRICS
        },
    }
    if controls_path is not None:
        controls = read_json(controls_path)
        if controls["rubric_fingerprint"] != bundle["rubric_fingerprint"]:
            raise ValueError("Control rubric differs from the benchmark")
        expected_controls = [meta for meta in mapping.values() if meta["scope"] == "control"]
        results = controls["results"]
        if (controls.get("validated_judgments") != len(expected_controls)
                or controls.get("logical_judgments") != len(expected_controls)
                or len(results) != len(expected_controls)
                or {meta["control_index"] for meta in expected_controls} != set(range(len(results)))):
            raise ValueError("Control results are incomplete or duplicated")
        identities = set()
        for result in results:
            kind = result.get("kind")
            if (kind not in RUBRICS or not isinstance(result.get("case"), str)
                    or not result["case"] or len(result["case"]) > 80
                    or not all(c.isascii() and (c.isalnum() or c in "_-") for c in result["case"])
                    or result.get("served_model") != bundle["served_model"]
                    or not isinstance(result.get("expected_label"), str)
                    or not isinstance(result.get("actual_label"), str)
                    or result["expected_label"] not in RUBRICS[kind]["criteria"]
                    or result["actual_label"] not in RUBRICS[kind]["criteria"]):
                raise ValueError("Invalid control labels, identity or served model")
            identities.add((result["case"], kind))
        if len(identities) != len(results):
            raise ValueError("Duplicate control case")
        control_rows = []
        for task_id, meta in mapping.items():
            if meta["scope"] != "control":
                continue
            original = controls["results"][meta["control_index"]]
            if original["state"] != inputs[task_id]["state"] or original["kind"] != meta["kind"]:
                raise ValueError("Agent and Jev controls differ")
            control_rows.append({"case": original["case"], "kind": meta["kind"],
                                 "expected": original["expected_label"], "jev": original["actual_label"],
                                 "sol": verdicts[task_id]["label"]})
        summary["synthetic_controls"] = {
            "interpretation": "Assistant-authored behavioral expectations, not human labels or a judge accuracy estimate.",
            "source_sha256": sha256(controls_path),
            "judgments": len(control_rows), "rows": control_rows,
            "jev_matches": sum(r["jev"] == r["expected"] for r in control_rows),
            "sol_matches": sum(r["sol"] == r["expected"] for r in control_rows),
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "summary.json", summary)
    (output_dir / "per-answer-comparison.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in public_rows), encoding="utf-8")
    lines = ["# Jev and GPT 6.1 Sol Ultra judge comparison", "", summary["interpretation"], "",
             f"Same {summary['overall']['unique_questions']} questions, {len(public_rows)} answers and frozen rubrics. "
             f"{len(tasks)} unique benchmark judgments per evaluator. Sol used separate blinded Codex agents for correctness and grounding.", "",
             "| Variant | Answers | Jev correct | Sol correct | Jev correct + supported | Sol correct + supported |",
             "|---|---:|---:|---:|---:|---:|"]
    for variant, row in summary["by_variant"].items():
        lines.append(f"| {variant} | {row['answers']} | {row['jev']['correct_count']} | {row['sol']['correct_count']} | "
                     f"{row['jev']['correct_and_supported_count']} | {row['sol']['correct_and_supported_count']} |")
    lines += ["", "All rates retain every planned answer, including abstained and unjudgeable outcomes. "
              "Repeated questions across variants are not independent samples. Greater agreement or higher scores alone do not establish the better judge.", "",
              summary["sol_model_identity_scope"], "",
              "[Full summary](summary.json) · [Per-answer labels](per-answer-comparison.jsonl)", ""]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jev-dir", type=Path, required=True)
    parser.add_argument("--agent-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--controls", type=Path)
    args = parser.parse_args()
    compare_judges(args.jev_dir, args.agent_dir, args.output_dir, controls_path=args.controls)
    print(args.output_dir / "report.md")


if __name__ == "__main__":
    main()

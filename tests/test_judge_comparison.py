"""No model calls: verify identical evidence, denominators and blinded imports."""

import json

import pytest

from rag_benchmark.judge_comparison import agreement, compare_judges, sha256, summarize_answer_rows
from rag_benchmark.semantic import prepare_evaluation, RUBRICS, SERVED_MODEL
from rag_benchmark.storage import atomic_json, digest
from test_semantic import fixture_run, valid_response


def write_lines(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def setup_comparison(tmp_path):
    source = fixture_run(tmp_path)
    jev = tmp_path / "jev"
    prepare_evaluation(source, jev)
    bundle = json.loads((jev / "bundle.json").read_text())
    tasks = [json.loads(line) for line in (jev / "tasks.jsonl").read_text().splitlines()]
    agent = tmp_path / "agent"
    agent.mkdir()
    shards = {"correctness": [], "grounding_a": [], "grounding_b": []}
    ground_index = 0
    for task in tasks:
        if task["kind"] == "correctness":
            shard = "correctness"
        else:
            shard = "grounding_a" if ground_index % 2 else "grounding_b"
            ground_index += 1
        shards[shard].append(task)
        atomic_json(jev / "responses" / (task["id"] + ".json"), {
            "task_id": task["id"], "bundle_fingerprint": bundle["fingerprint"], "status": "success",
            "response": valid_response(task["kind"]),
        })
    mapping = []
    for name, selected in shards.items():
        kind = "correctness" if name == "correctness" else "grounding"
        groups, judgments = [], []
        for index, task in enumerate(selected):
            state = task["payload"]["state"]
            field = "gold_passages" if kind == "correctness" else "context_passages"
            passages = {str(i): p for i, p in enumerate(state[field])}
            case = {"task_id": task["id"], "input_sha256": digest(state),
                    "answer": state["answer"], "passage_keys": list(passages)}
            if kind == "correctness":
                case["reference_answers"] = state["reference_answers"]
            groups.append({"question": state["question"], "passages": passages, "cases": [case]})
            mapping.append({"task_id": task["id"], "kind": kind, "input_sha256": digest(state),
                            "scope": "benchmark", "shard": name})
            label = ("partial" if index == 0 else "correct") if kind == "correctness" else "supported"
            judgments.append({"task_id": task["id"], "input_sha256": digest(state), "label": label,
                              "reason": "PRIVATE_JUDGE_REASON", "needs_review": False,
                              "evidence_refs": [p["source_id"] for p in state[field]]})
        atomic_json(agent / (name + ".json"), {"kind": kind, "rubric": {"type": "choice", **RUBRICS[kind]},
                                               "groups": groups})
        write_lines(agent / (name + ".verdicts.jsonl"), judgments)
    atomic_json(agent / "private-mapping.json", mapping)
    atomic_json(agent / "manifest.json", {
        "files": {p.name: sha256(p) for p in agent.glob("*.json")},
        "rubric_fingerprint": digest(RUBRICS), "source_bundle_fingerprint": bundle["fingerprint"],
        "requested_model": "gpt-6.1-sol", "requested_reasoning_effort": "ultra", "execution_method": "fixture",
    })
    return jev, agent, tmp_path / "public"


def test_comparison_preserves_answer_denominators_and_omits_private_text(tmp_path):
    jev, agent, public = setup_comparison(tmp_path)
    summary = compare_judges(jev, agent, public)
    assert summary["complete"]
    assert summary["unique_benchmark_tasks"] == 5
    assert summary["overall"]["answers"] == 4
    assert summary["overall"]["unique_questions"] == 2
    assert summary["overall"]["jev"]["correct_count"] == 4
    assert summary["overall"]["sol"]["correct_count"] == 2
    assert summary["overall"]["agreement"]["correctness"]["agreement_rate"] == .5
    assert summary["unique_task_agreement"]["correctness"]["n"] == 2
    for path in public.iterdir():
        text = path.read_text()
        assert "PRIVATE_" not in text
        assert str(tmp_path) not in text


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "input_hash", "label", "evidence", "review_type"])
def test_bad_or_incomplete_verdicts_are_rejected(tmp_path, mutation):
    jev, agent, public = setup_comparison(tmp_path)
    path = agent / "correctness.verdicts.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows.append(rows[0])
    elif mutation == "input_hash":
        rows[0]["input_sha256"] = "changed"
    elif mutation == "label":
        rows[0]["label"] = "supported"
    elif mutation == "evidence":
        rows[0]["evidence_refs"] = ["unprovided-source"]
    else:
        rows[0]["needs_review"] = "false"
    write_lines(path, rows)
    with pytest.raises(ValueError):
        compare_judges(jev, agent, public)
    assert not public.exists()


def test_changed_evidence_is_rejected_even_with_recomputed_agent_hashes(tmp_path):
    jev, agent, public = setup_comparison(tmp_path)
    path = agent / "correctness.json"
    pack = json.loads(path.read_text())
    group = pack["groups"][0]
    group["question"] = "Different PRIVATE question"
    case = group["cases"][0]
    state = {"question": group["question"], "answer": case["answer"],
             "reference_answers": case["reference_answers"],
             "gold_passages": [group["passages"][key] for key in case["passage_keys"]]}
    case["input_sha256"] = digest(state)
    atomic_json(path, pack)
    mapping_path = agent / "private-mapping.json"
    rows = json.loads(mapping_path.read_text())
    for row in rows:
        if row["task_id"] == case["task_id"]:
            row["input_sha256"] = case["input_sha256"]
    atomic_json(mapping_path, rows)
    m = json.loads((agent / "manifest.json").read_text())
    for name in m["files"]:
        m["files"][name] = sha256(agent / name)
    atomic_json(agent / "manifest.json", m)
    with pytest.raises(ValueError, match="identical evidence"):
        compare_judges(jev, agent, public)


def test_joint_rate_keeps_uncertain_and_abstaining_answers_in_denominator():
    rows = []
    for index, (correctness, grounding) in enumerate([
        ("correct", "supported"), ("correct", "contradicted"), ("unjudgeable", "supported"),
        ("abstained", "abstained"),
    ]):
        labels = {"correctness": correctness, "grounding": grounding}
        rows.append({"query_id": str(index), "jev": labels, "sol": labels})
    result = summarize_answer_rows(rows)
    assert result["sol"]["correct_rate"] == .5
    assert result["sol"]["correct_and_supported_rate"] == .25
    assert result["agreement"]["correctness"]["agreement_rate"] == 1
    assert agreement([])["agreement_rate"] is None


def test_private_input_directory_cannot_be_overwritten(tmp_path):
    jev, agent, _ = setup_comparison(tmp_path)
    with pytest.raises(ValueError, match="overlaps"):
        compare_judges(jev, agent, agent)


def add_control(agent, path):
    pack_path = agent / "correctness.json"
    pack = json.loads(pack_path.read_text())
    group = pack["groups"][0]
    case = {**group["cases"][0], "task_id": "synthetic-control-id"}
    group["cases"].append(case)
    atomic_json(pack_path, pack)
    mapping_path = agent / "private-mapping.json"
    mapping = json.loads(mapping_path.read_text())
    mapping.append({"task_id": case["task_id"], "input_sha256": case["input_sha256"], "kind": "correctness",
                    "shard": "correctness", "scope": "control", "control_index": 0})
    atomic_json(mapping_path, mapping)
    verdict_path = agent / "correctness.verdicts.jsonl"
    rows = [json.loads(line) for line in verdict_path.read_text().splitlines()]
    rows.append({**rows[0], "task_id": case["task_id"], "label": "correct"})
    write_lines(verdict_path, rows)
    manifest = json.loads((agent / "manifest.json").read_text())
    manifest["files"] = {name: sha256(agent / name) for name in manifest["files"]}
    atomic_json(agent / "manifest.json", manifest)
    state = {"question": group["question"], "answer": case["answer"],
             "reference_answers": case["reference_answers"],
             "gold_passages": [group["passages"][key] for key in case["passage_keys"]]}
    controls = {"rubric_fingerprint": digest(RUBRICS), "logical_judgments": 1, "validated_judgments": 1,
                "results": [{"case": "fixture", "kind": "correctness", "state": state,
                             "served_model": SERVED_MODEL, "expected_label": "correct", "actual_label": "correct"}]}
    atomic_json(path, controls)
    return controls


def test_controls_stay_separate_from_benchmark_denominator(tmp_path):
    jev, agent, public = setup_comparison(tmp_path)
    controls_path = tmp_path / "controls.json"
    add_control(agent, controls_path)
    result = compare_judges(jev, agent, public, controls_path=controls_path)
    assert result["overall"]["answers"] == 4
    assert result["unique_benchmark_tasks"] == 5
    assert result["unique_control_tasks"] == 1
    assert result["synthetic_controls"]["sol_matches"] == 1


@pytest.mark.parametrize("mutation", ["label", "model", "coverage", "state"])
def test_invalid_control_results_cannot_be_published(tmp_path, mutation):
    jev, agent, public = setup_comparison(tmp_path)
    controls_path = tmp_path / "controls.json"
    controls = add_control(agent, controls_path)
    if mutation == "label":
        controls["results"][0]["actual_label"] = "PRIVATE leaked response"
    elif mutation == "model":
        controls["results"][0]["served_model"] = "other-judge"
    elif mutation == "coverage":
        controls["validated_judgments"] = 0
    else:
        controls["results"][0]["state"]["answer"] = "Different answer"
    atomic_json(controls_path, controls)
    with pytest.raises(ValueError):
        compare_judges(jev, agent, public, controls_path=controls_path)
    assert not public.exists()


def test_control_input_cannot_be_overwritten_by_report(tmp_path):
    jev, agent, public = setup_comparison(tmp_path)
    controls_path = public / "summary.json"
    add_control(agent, controls_path)
    original = controls_path.read_bytes()
    with pytest.raises(ValueError, match="overlaps control"):
        compare_judges(jev, agent, public, controls_path=controls_path)
    assert controls_path.read_bytes() == original

"""Offline fixtures and mocked HTTP only; no model calls or real credentials."""

import copy
import hashlib
import json

import httpx
import pytest

from rag_benchmark.semantic import (
    ENDPOINT,
    MODEL,
    PROVIDER,
    SERVED_MODEL,
    RUBRICS,
    EvaluationIncomplete,
    evaluate_bundle,
    prepare_evaluation,
    summarize_evaluation,
    validate_response,
)
from rag_benchmark.storage import digest


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def fixture_run(tmp_path):
    data = tmp_path / "data" / "fixture"
    run = tmp_path / "runs" / "fixture"
    data.mkdir(parents=True)
    corpus = [
        {"id": "a#c1", "title": "PRIVATE_TITLE", "text": "PRIVATE_GOLD seven facts."},
        {"id": "a#c2", "title": "PRIVATE_CONTEXT_TITLE", "text": "PRIVATE_CONTEXT other facts."},
    ]
    questions = [
        {
            "id": f"q{i}",
            "question": f"PRIVATE_QUESTION {i}?",
            "answers": ["PRIVATE_REFERENCE seven"],
            "gold_ids": ["a#c1"],
            "article_id": f"a{i}",
            "source": "web",
            "category": "FACTUAL",
        }
        for i in range(2)
    ]
    for name, rows in (("corpus.jsonl", corpus), ("questions.dev.jsonl", questions)):
        (data / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in data.iterdir()}
    dataset = {"revision": "a" * 40, "fingerprint": digest(hashes), "file_sha256": hashes}
    write_json(data / "manifest.json", dataset)
    rows = []
    for variant in ("bm25", "bm25_embeddinggemma"):
        for i, q in enumerate(questions):
            context_ids = ["a#c2"] if variant == "bm25_embeddinggemma" and i == 0 else ["a#c1"]
            rows.append(
                {
                    **{k: q[k] for k in ("question", "gold_ids", "article_id", "source", "category")},
                    "variant": variant,
                    "query_id": q["id"],
                    "reference_answers": q["answers"],
                    "answer": "  PRIVATE_ANSWER seven [a#c1].  ",
                    "context_ids": context_ids,
                    "metrics": {"answer_token_f1": 0.533},
                    "generator_model": "SECRET_GENERATOR_NAME",
                    "ranked_ids": context_ids,
                }
            )
    identity = {
        "config": {"dataset": {"path": str(data)}, "experiment": {"context_k": 1}},
        "dataset": dataset,
        "runtime": {"source_hash": "historical-generation-code", "git_revision": "old-generation-commit"},
        "corpus_sha256": hashes["corpus.jsonl"],
        "questions_sha256": hashes["questions.dev.jsonl"],
        "split": "dev",
        "question_ids": [q["id"] for q in questions],
        "variants": ["bm25", "bm25_embeddinggemma"],
        "retrieval_only": False,
        "generator_server": {},
    }
    fingerprint = digest({**identity, "runtime": {"source_hash": "historical-generation-code"}})
    write_json(run / "manifest.json", {**identity, "fingerprint": fingerprint, "created_at": "fixture"})
    (run / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return run


def prepared(tmp_path, variants=None):
    run = fixture_run(tmp_path)
    bundle = tmp_path / "evaluation"
    prepare_evaluation(run, bundle, variants)
    return run, bundle


def valid_response(kind, choice=None, confidence=0.9):
    options = list(RUBRICS[kind]["criteria"])
    choice = choice or options[0]
    return {
        "model": SERVED_MODEL,
        "provider": PROVIDER,
        "id": "gen-fixture",
        "answers": {
            "decision": {
                "type": "choice",
                "choice": choice,
                "probabilities": {
                    label: 0.8 if label == choice else 0.2 / (len(options) - 1) for label in options
                },
                "confidence": confidence,
            }
        },
        "usage": {"input_tokens": 100, "output_tokens": 10, "cost": 0.001},
        "ignored_secret": "SECRET_TOKEN",
    }


def mock_client(monkeypatch, handler):
    original = httpx.Client

    def create(**kwargs):
        assert kwargs == {"timeout": 60, "trust_env": False, "follow_redirects": False}
        return original(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(httpx, "Client", create)


def test_preparation_blinds_and_separates_correctness_from_grounding(tmp_path, monkeypatch):
    monkeypatch.setattr(httpx, "Client", lambda **kw: pytest.fail("Preparation must remain offline"))
    _, bundle = prepared(tmp_path)
    tasks = [json.loads(line) for line in (bundle / "tasks.jsonl").read_text().splitlines()]
    assert len(tasks) == 5  # 4 answers, 8 logical judgments, 5 distinct payloads.
    for task in tasks:
        payload, state = task["payload"], task["payload"]["state"]
        assert payload["model"] == MODEL
        assert state["answer"] == "  PRIVATE_ANSWER seven [a#c1].  "
        assert not set(state) & {
            "variant",
            "query_id",
            "metrics",
            "ranked_ids",
            "model",
            "generator_model",
            "article_id",
        }
        assert "SECRET_GENERATOR_NAME" not in json.dumps(payload)
        if task["kind"] == "correctness":
            assert set(state) == {"question", "answer", "reference_answers", "gold_passages"}
            assert state["reference_answers"] == ["PRIVATE_REFERENCE seven"]
            assert state["gold_passages"][0]["text"] == "PRIVATE_GOLD seven facts."
        else:
            assert set(state) == {"question", "answer", "context_passages"}
            assert "PRIVATE_REFERENCE" not in json.dumps(payload)
        instructions = payload["questions"]["decision"]["instructions"]
        assert all(word in instructions for word in ("Short answers", "negation", "untrusted", "unjudgeable"))
    summary = evaluate_bundle(bundle)  # Default never creates an HTTP client.
    assert summary["workload"] == {
        "unique_questions": 2,
        "planned_answers": 4,
        "tasks_before_deduplication": 8,
        "unique_tasks": 5,
        "successful_tasks": 0,
        "failed_tasks": 0,
        "inflight_tasks": 0,
        "pending_tasks": 5,
    }
    assert summary["overall"]["semantic_correct_rate_of_planned"] is None
    assert summary["overall"]["correct_and_supported_rate_of_planned"] is None
    public = (
        (bundle / "summary.json").read_text()
        + (bundle / "report.md").read_text()
        + (bundle / "per-answer-judgments.jsonl").read_text()
    )
    assert all(s not in public for s in ("PRIVATE_", str(tmp_path), "SECRET_"))


def test_selected_variant_prepares_only_its_actual_answers(tmp_path):
    _, bundle = prepared(tmp_path, ["bm25_embeddinggemma"])
    summary = summarize_evaluation(bundle)
    assert summary["workload"]["planned_answers"] == 2
    assert summary["workload"]["unique_tasks"] == 4
    assert set(summary["by_variant"]) == {"bm25_embeddinggemma"}


def test_successful_calls_are_deduplicated_and_resumed_without_replay(tmp_path, monkeypatch):
    _, bundle = prepared(tmp_path)
    calls = []

    def handler(request):
        assert str(request.url) == ENDPOINT
        assert request.headers["authorization"] == "Bearer SECRET_TOKEN"
        payload = json.loads(request.content)
        calls.append(digest(payload))
        kind = "correctness" if "reference_answers" in payload["state"] else "grounding"
        return httpx.Response(200, json=valid_response(kind))

    mock_client(monkeypatch, handler)
    with pytest.raises(EvaluationIncomplete) as partial:
        evaluate_bundle(bundle, "SECRET_TOKEN", max_requests=1)
    assert partial.value.summary["workload"]["successful_tasks"] == 1
    assert partial.value.summary["overall"]["planned_answers"] == 4
    assert partial.value.summary["overall"]["semantic_correct_rate_of_planned"] is None
    assert partial.value.summary["overall"]["correct_and_supported_rate_of_planned"] is None
    result = evaluate_bundle(bundle, "SECRET_TOKEN", max_requests=10)
    assert len(calls) == len(set(calls)) == 5
    assert result["complete"]
    assert result["overall"]["semantic_correct_count"] == 4
    assert result["overall"]["correct_and_supported_count"] == 4
    assert result["overall"]["semantic_correct_rate_of_planned"] == 1
    assert result["usage"] == {"input_tokens": 500, "output_tokens": 50, "cost": 0.005}
    assert result["served_models"] == [SERVED_MODEL]
    assert result["provider"] == PROVIDER
    public_rows = [
        json.loads(line) for line in (bundle / "per-answer-judgments.jsonl").read_text().splitlines()
    ]
    assert all(
        row["correctness"]["label"] == "correct" and row["grounding"]["label"] == "supported"
        for row in public_rows
    )
    assert evaluate_bundle(bundle, max_requests=10) == result  # Completed cache needs no key.
    assert len(calls) == 5
    assert "SECRET_TOKEN" not in "".join(p.read_text() for p in bundle.rglob("*.json"))


@pytest.mark.parametrize("failure", ["http", "transport", "malformed"])
def test_failures_are_sanitized_counted_and_never_automatically_retried(tmp_path, monkeypatch, failure):
    _, bundle = prepared(tmp_path)
    calls = []

    def handler(request):
        calls.append(request)
        if failure == "transport":
            raise httpx.ConnectError("SECRET_TOKEN PRIVATE_SERVER_RESPONSE", request=request)
        if failure == "malformed":
            return httpx.Response(200, text="SECRET_TOKEN PRIVATE_SERVER_RESPONSE")
        return httpx.Response(500, text="SECRET_TOKEN PRIVATE_SERVER_RESPONSE")

    mock_client(monkeypatch, handler)
    with pytest.raises(EvaluationIncomplete) as error:
        evaluate_bundle(bundle, "SECRET_TOKEN", max_requests=10)
    summary = error.value.summary
    assert summary["workload"]["failed_tasks"] == 1
    assert summary["workload"]["pending_tasks"] == 4
    assert summary["overall"]["planned_answers"] == 4
    assert summary["overall"]["semantic_correct_rate_of_planned"] is None
    with pytest.raises(EvaluationIncomplete):
        evaluate_bundle(bundle, "SECRET_TOKEN", max_requests=10)
    assert len(calls) == 1
    stored = "".join(p.read_text() for p in (bundle / "responses").iterdir())
    assert "SECRET_TOKEN" not in stored + str(error.value)
    assert "PRIVATE_SERVER_RESPONSE" not in stored + str(error.value)


@pytest.mark.parametrize(
    "mutation",
    [
        "model",
        "provider",
        "id",
        "cost",
        "type",
        "options",
        "nan",
        "negative",
        "sum",
        "confidence",
        "choice",
        "usage",
        "shape",
    ],
)
def test_rejects_malformed_choice_responses(mutation):
    response = valid_response("correctness")
    decision = response["answers"]["decision"]
    if mutation == "model":
        response["model"] = "jev-latest"
    if mutation == "provider":
        response["provider"] = "unknown"
    if mutation == "id":
        response["id"] = "PRIVATE response body with spaces"
    if mutation == "cost":
        response["usage"]["cost"] = float("nan")
    if mutation == "type":
        decision["type"] = "score"
    if mutation == "options":
        decision["probabilities"]["unknown"] = 0
    if mutation == "nan":
        decision["probabilities"]["correct"] = float("nan")
    if mutation == "negative":
        decision["probabilities"]["partial"] = -0.1
    if mutation == "sum":
        decision["probabilities"]["correct"] = 0.6
    if mutation == "confidence":
        decision["confidence"] = 1.1
    if mutation == "choice":
        decision["choice"] = "incorrect"
    if mutation == "usage":
        response["usage"]["input_tokens"] = "SECRET_TOKEN"
    if mutation == "shape":
        response["answers"]["decision"] = []
    with pytest.raises(ValueError, match="Invalid Jev"):
        validate_response(response, "correctness")


@pytest.mark.parametrize("target", ["source", "payload", "mapping", "bundle"])
def test_detects_hash_drift_before_network(tmp_path, monkeypatch, target):
    run, bundle = prepared(tmp_path)
    path = {
        "source": run / "predictions.jsonl",
        "payload": bundle / "tasks.jsonl",
        "mapping": bundle / "rows.jsonl",
        "bundle": bundle / "bundle.json",
    }[target]
    if target == "bundle":
        value = json.loads(path.read_text())
        value["model"] = "unapproved"
        write_json(path, value)
    else:
        path.write_text(path.read_text() + "\n")
    monkeypatch.setattr(httpx, "Client", lambda **kw: pytest.fail("Changed bundle must never execute"))
    with pytest.raises(ValueError):
        evaluate_bundle(bundle, "SECRET_TOKEN", max_requests=1)


def test_incomplete_source_run_is_rejected_instead_of_dropping_rows(tmp_path):
    run = fixture_run(tmp_path)
    rows = (run / "predictions.jsonl").read_text().splitlines()
    (run / "predictions.jsonl").write_text("\n".join(rows[:-1]) + "\n")
    with pytest.raises(ValueError, match="incomplete"):
        prepare_evaluation(run, tmp_path / "evaluation")


def test_no_overwrite_and_invalid_historical_fingerprint(tmp_path):
    run, bundle = prepared(tmp_path)
    with pytest.raises(FileExistsError):
        prepare_evaluation(run, bundle)
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["fingerprint"] = "bad"
    write_json(run / "manifest.json", manifest)
    with pytest.raises(ValueError, match="fingerprint"):
        prepare_evaluation(run, tmp_path / "other")


def test_label_counts_intersection_unjudgeable_and_confidence_not_mean_probability(tmp_path, monkeypatch):
    _, bundle = prepared(tmp_path, ["bm25"])

    def handler(request):
        state = json.loads(request.content)["state"]
        first = state["question"].endswith("0?")
        kind = "correctness" if "reference_answers" in state else "grounding"
        label = (
            ("correct" if first else "unjudgeable")
            if kind == "correctness"
            else ("unsupported" if first else "supported")
        )
        return httpx.Response(200, json=valid_response(kind, label, confidence=0.5 if first else 0.9))

    mock_client(monkeypatch, handler)
    result = evaluate_bundle(bundle, "SECRET_TOKEN", max_requests=4)
    overall = result["overall"]
    assert overall["semantic_correct_count"] == 1
    assert (
        overall["semantic_correct_rate_of_planned"] == 0.5
    )  # 1/2, not confidence .5 or chosen probability .8.
    assert overall["correct_and_supported_count"] == 0
    assert overall["correct_and_supported_rate_of_planned"] == 0
    assert overall["unjudgeable_answers"] == 1
    assert overall["correctness_low_confidence"] == overall["grounding_low_confidence"] == 1
    assert result["by_source"]["web"] == overall


@pytest.mark.parametrize("field", ["source", "category", "article_id", "context_ids"])
def test_rejects_incorrect_row_metadata_or_context_selection(tmp_path, field):
    run = fixture_run(tmp_path)
    path = run / "predictions.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0][field] = ["a#c2"] if field == "context_ids" else "wrong"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="labels|context"):
        prepare_evaluation(run, tmp_path / "evaluation")


@pytest.mark.parametrize("status", ["failed", "inflight"])
def test_non_success_response_leftovers_never_receive_credit(tmp_path, status):
    _, bundle = prepared(tmp_path, ["bm25"])
    info = json.loads((bundle / "bundle.json").read_text())
    tasks = [json.loads(line) for line in (bundle / "tasks.jsonl").read_text().splitlines()]
    for task in tasks:
        write_json(
            bundle / "responses" / (task["id"] + ".json"),
            {
                "task_id": task["id"],
                "bundle_fingerprint": info["fingerprint"],
                "status": status,
                "response": valid_response(task["kind"]),
            },
        )
    summary = summarize_evaluation(bundle)
    assert summary["overall"]["semantic_correct_count"] == 0
    assert summary["overall"]["correct_and_supported_count"] == 0
    assert summary["overall"]["judged_both"] == 0
    assert summary["overall"]["semantic_correct_rate_of_planned"] is None
    assert summary["usage"] == {"input_tokens": 0, "output_tokens": 0, "cost": 0}
    public = [json.loads(line) for line in (bundle / "per-answer-judgments.jsonl").read_text().splitlines()]
    assert all(row["correctness"]["label"] is None and row["grounding"]["label"] is None for row in public)


@pytest.mark.parametrize("drift", ["code", "rubric"])
def test_execution_blocks_judge_drift_but_read_only_summary_is_available(tmp_path, monkeypatch, drift):
    from rag_benchmark import semantic

    _, bundle = prepared(tmp_path)
    if drift == "code":
        real_sha = semantic._sha
        monkeypatch.setattr(
            semantic, "_sha", lambda path: "changed" if str(path) == semantic.__file__ else real_sha(path)
        )
    else:
        changed = copy.deepcopy(RUBRICS)
        changed["correctness"]["instructions"] += " Different rubric."
        monkeypatch.setattr(semantic, "RUBRICS", changed)
    assert evaluate_bundle(bundle, max_requests=0)["workload"]["successful_tasks"] == 0
    monkeypatch.setattr(httpx, "Client", lambda **kw: pytest.fail("Drift must block hosted execution"))
    with pytest.raises(ValueError, match="implementation or rubric changed"):
        evaluate_bundle(bundle, "SECRET_TOKEN", max_requests=1)


def test_two_decimal_probability_rounding_is_bounded_and_preserved():
    response = valid_response("correctness")
    response["answers"]["decision"]["probabilities"]["partial"] = 0.04
    result = validate_response(response, "correctness")
    assert result["answers"]["decision"]["probabilities"] == response["answers"]["decision"]["probabilities"]
    assert sum(result["answers"]["decision"]["probabilities"].values()) == pytest.approx(0.99)
    response["answers"]["decision"]["probabilities"]["correct"] = 0.75
    with pytest.raises(ValueError, match="probability sum"):
        validate_response(response, "correctness")
    response["answers"]["decision"]["probabilities"]["correct"] = 0.799
    with pytest.raises(ValueError, match="probability sum"):
        validate_response(response, "correctness")


def test_validation_failure_retains_static_diagnostic_without_response_text(tmp_path, monkeypatch):
    _, bundle = prepared(tmp_path)

    def handler(request):
        state = json.loads(request.content)["state"]
        kind = "correctness" if "reference_answers" in state else "grounding"
        response = valid_response(kind)
        response["answers"]["decision"]["confidence"] = -1
        response["private_error"] = "SECRET_TOKEN PRIVATE_SERVER_RESPONSE"
        return httpx.Response(200, json=response)

    mock_client(monkeypatch, handler)
    with pytest.raises(EvaluationIncomplete):
        evaluate_bundle(bundle, "SECRET_TOKEN", max_requests=1)
    record = json.loads(next((bundle / "responses").glob("*.json")).read_text())
    assert record["error"]["reason"] == "Invalid Jev choice confidence"
    assert "PRIVATE_SERVER_RESPONSE" not in json.dumps(record)
    assert "SECRET_TOKEN" not in json.dumps(record)

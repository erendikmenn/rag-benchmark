"""Injected HTTP and tiny synthetic images only; no API/model/DB calls."""

import asyncio
import importlib.util
import json
from pathlib import Path

import pytest

Image = pytest.importorskip("PIL.Image")

spec = importlib.util.spec_from_file_location(
    "api_caption_tests", Path(__file__).parents[1] / "scripts/prepare_api_captions_v2.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class FastGate:
    def __init__(self):
        self.active, self.limit, self.rps = 0, 100, 25

    async def enter(self, stopped):
        if stopped():
            return False
        self.active += 1
        return True

    def leave(self):
        self.active -= 1

    def good(self):
        pass

    def penalize(self):
        self.limit = max(1, self.limit // 2)


def response(text="A blue object.", *, finish="stop", cost=0.000015, provider=m.PROVIDER):
    return {
        "status_code": 200,
        "done": True,
        "events": [
            {
                "model": m.MODEL,
                "provider": provider,
                "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": finish}],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 10,
                    "cost": cost,
                    "completion_tokens_details": {"reasoning_tokens": 0},
                },
            }
        ],
    }


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    from rag_benchmark.multimodal_data import write_dataset
    from rag_benchmark.multimodal import load_dataset

    source = tmp_path / "native"
    (source / "media").mkdir(parents=True)
    for i in range(2):
        Image.new("RGB", (4, 3), (i * 20, 30, 200)).save(source / f"media/{i}.jpg")
    write_dataset(
        source,
        dataset_id="lasco-test",
        track="composed_image",
        revision="r",
        corpus=[{"id": str(i), "media": {"image": f"media/{i}.jpg"}} for i in range(2)],
        queries=[
            {
                "id": "q",
                "text": "make it blue",
                "media": {"image": "media/0.jpg"},
                "metadata": {"reference_id": "0", "exclude_ids": ["0"], "transition_text_empty": False},
            }
        ],
        qrels=[{"query_id": "q", "corpus_id": "1", "relevance": 1.0}],
        sources=[],
        license="synthetic",
        text_source="absent_requires_generation",
    )
    dataset = load_dataset(source)
    assets = {
        x["path"]: x["sha256"] for x in map(json.loads, (source / "assets.jsonl").read_text().splitlines())
    }
    value = {
        "prior_run": {
            "files": {},
            "prior_debit_usd": 0.0530309,
            "prior_result_sha256": "a" * 64,
            "prior_spec_sha256": "b" * 64,
        },
        "fresh_full_generation": True,
        "local_or_v1_caption_reuse": False,
        "protocol": m.PROTOCOL,
        "source": str(source),
        "source_identity": dataset.identity,
        "counts": {"corpus": 2, "queries": 1},
        "output": str(tmp_path / "run"),
        "destination": str(tmp_path / "derived"),
        "files": {
            str(source / n): m.sha(source / n)
            for n in ["dataset.json", "corpus.jsonl", "queries.jsonl", "qrels.jsonl", "assets.jsonl"]
        },
        "assets": assets,
        "policy": m.policy(),
        "budget_usd": 15.0,
        "request_bound_usd": m.BOUND_USD,
        "concurrency": 100,
        "initial_max_rps": 25,
        "max_attempts": 3,
        "benchmark_runs": 0,
        "database_writes": False,
    }
    original = m.safe
    monkeypatch.setattr(m, "safe", lambda p, root=None: original(p, root or tmp_path))
    return value, dataset


def run(value, ds, transport, **kw):
    return asyncio.run(m.run(value, ds, transport, spec_sha256="f" * 64, gate=FastGate(), **kw))


def test_full_success_cost_and_elapsed(fixture):
    value, ds = fixture

    async def transport(payload):
        return response()

    result = run(value, ds, transport)
    assert result["status"] == "rows_complete"
    assert result["charged_usd"] == pytest.approx(0.0530309 + 0.000030)
    assert result["new_job_charged_usd"] == pytest.approx(0.000030)
    assert result["elapsed_seconds"] >= 0 and result["start_utc"]
    assert len(m.audit_rows(value, ds, value["output"], spec_sha256="f" * 64)) == 2


def test_partial_read_retry_same_payload_and_conservative_cost(fixture, monkeypatch):
    value, ds = fixture
    seen = {}

    async def no_sleep(_):
        pass

    monkeypatch.setattr(m.asyncio, "sleep", no_sleep)

    async def transport(payload):
        key = m.seal(payload)
        seen[key] = seen.get(key, 0) + 1
        if seen[key] == 1:
            return {
                "status_code": 200,
                "events": [{"choices": [{"delta": {"content": "broken"}}]}],
                "done": False,
                "error_type": "ReadError",
            }
        return response()

    result = run(value, ds, transport)
    assert result["status"] == "rows_complete" and result["attempts"] == 4
    assert set(seen.values()) == {2}
    assert result["charged_usd"] == pytest.approx(0.0530309 + 2 * m.BOUND_USD + 0.000030)
    assert result["reserved_usd"] == 0 and not result["budget_unknown"]
    m.audit_rows(value, ds, value["output"], spec_sha256="f" * 64)


def test_exhausted_transport_isolated_and_three_attempts(fixture, monkeypatch):
    value, ds = fixture

    async def no_sleep(_):
        pass

    monkeypatch.setattr(m.asyncio, "sleep", no_sleep)

    async def transport(payload):
        return {"status_code": None, "events": [], "done": False, "error_type": "ReadTimeout"}

    result = run(value, ds, transport)
    assert result["status"] == "rows_partial" and result["attempts"] == 6 and result["failed"] == 2
    assert result["charged_usd"] == pytest.approx(0.0530309 + 6 * m.BOUND_USD)
    assert not list(Path(value["output"]).glob("rows/*/accepted.json"))


def test_content_length_no_retry_other_rows_continue(fixture):
    value, ds = fixture
    count = 0

    async def transport(payload):
        nonlocal count
        count += 1
        return response(finish="length") if count == 1 else response()

    result = run(value, ds, transport)
    assert count == 2 and result["status"] == "rows_partial" and result["caption_rows"] == 1
    assert result["failed"] == 1 and len(result["failed_indices"]) == 1


def test_cost_above_bound_full_debit_global_stop(fixture):
    value, ds = fixture

    async def transport(payload):
        return response(cost=0.2)

    result = run(value, ds, transport)
    assert result["status"] == "billing_bound_violation"
    assert result["charged_usd"] >= 0.0530309 + 0.2
    assert result["caption_rows"] == 0
    assert result["known_cost_usd"] >= 0.2


def test_prior_chain_mutation_before_calls(fixture, tmp_path):
    value, ds = fixture
    p = tmp_path / "old.json"
    p.write_text("old")
    value["prior_run"]["files"] = {str(p): m.sha(p)}
    p.write_text("changed")

    async def transport(payload):
        raise AssertionError("must not call")

    with pytest.raises(ValueError, match="prior_debit_binding"):
        run(value, ds, transport)


def test_transient_sse502_and_unknown_cost():
    raw = {"status_code": 200, "events": [{"error": {"code": 502}}], "done": True}
    assert m.retryable(raw)
    assert m.charge(raw) == {
        "charged_usd": m.BOUND_USD,
        "cost_known": False,
        "bounded_uncertain_attempt": True,
    }
    raw = response(cost=float("nan"))
    assert m.charge(raw)["budget_unknown"]


def test_prior_full_proof(tmp_path, monkeypatch):
    original = m.safe
    monkeypatch.setattr(m, "safe", lambda p: original(p, tmp_path))
    out = tmp_path / "v1"
    session = out / "sessions/session-000000"
    session.mkdir(parents=True)
    old = {"protocol": "lasco-haiku55-source-caption-en-v1", "policy": m.policy(), "output": str(out)}
    spec = tmp_path / "spec.json"
    m.write(spec, old)
    h = m.sha(spec)
    result = {
        "producer_closed": True,
        "requests_closed": True,
        "spec_sha256": h,
        "source_identity": "source",
        "caption_rows": 342,
        "failed": 1,
        "attempts": 343,
        "reserved_usd": 0,
        "status": "budget_unknown",
        "charged_usd": 0.0530309,
        "known_cost_usd": 0.0519557,
        "session": str(session),
    }
    for p, v in [
        (out / "result.json", result),
        (out / "status.json", result),
        (session / "result.json", result),
        (session / "claim.json", {}),
        (out / "pins.json", old),
        (out / "claim.json", {"spec_sha256": h}),
    ]:
        m.write(p, v)
    proof = m.prior_proof(spec, h, out / "result.json", m.sha(out / "result.json"), "source")
    assert len(proof["files"]) == 7 and proof["prior_debit_usd"] == 0.0530309
    (out / "status.json").write_text("{}")
    with pytest.raises(ValueError, match="prior_chain"):
        m.prior_proof(spec, h, out / "result.json", m.sha(out / "result.json"), "source")


def test_unfinished_eof_retry_but_not_explicit_length():
    assert m.retryable(
        {"status_code": 200, "done": False, "events": [{"choices": [{"delta": {"content": "partial"}}]}]}
    )
    raw = response(finish="length")
    raw["done"] = False
    assert not m.retryable(raw)


@pytest.mark.parametrize("finish", ["length", "refusal", "content_filter"])
def test_terminal_finish_overrides_transport_error(finish):
    raw = response(finish=finish)
    raw.update(error_type="ReadError", done=False)
    assert not m.retryable(raw)

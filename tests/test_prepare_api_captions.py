"""Injected HTTP and tiny synthetic images only; no API/model/DB calls."""

import asyncio
import copy
import importlib.util
import json
from pathlib import Path

import pytest

Image = pytest.importorskip("PIL.Image")

spec = importlib.util.spec_from_file_location(
    "api_caption_tests", Path(__file__).parents[1] / "scripts/prepare_api_captions.py"
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


def test_source_only_payload_and_successful_full_export(fixture):
    value, ds = fixture
    sent = []

    async def transport(payload):
        sent.append(copy.deepcopy(payload))
        return response()

    result = run(value, ds, transport)
    assert (
        result["status"] == "rows_complete"
        and result["caption_rows"] == 2
        and result["requests_closed"] is True
    )
    assert len(sent) == 2 and all(len(x["messages"][1]["content"]) == 1 for x in sent)
    assert "make it blue" not in json.dumps(sent) and "qrels" not in json.dumps(sent)
    rows = m.audit_rows(value, ds, value["output"], spec_sha256="f" * 64)
    identity = m.publish(value, ds, rows, spec_sha256="f" * 64)
    from rag_benchmark.multimodal import load_dataset

    derived = load_dataset(value["destination"])
    assert derived.identity == identity and derived.qrels == ds.qrels
    assert derived.queries[0]["text"] == ds.queries[0]["text"]
    assert derived.queries[0]["metadata"]["reference_id"] == "0"
    assert (Path(value["destination"]) / "qrels.jsonl").read_bytes() == (ds.root / "qrels.jsonl").read_bytes()
    assert derived.manifest["metadata"]["caption_protocol"] == m.PROTOCOL


@pytest.mark.parametrize("defect", ["length", "provider", "reasoning", "cost", "unfinished"])
def test_strict_failures_never_retry(fixture, defect):
    value, ds = fixture
    calls = []

    async def transport(payload):
        calls.append(1)
        raw = response()
        if defect == "length":
            raw["events"][0]["choices"][0]["finish_reason"] = "length"
        if defect == "provider":
            raw["events"][0]["provider"] = "Other"
        if defect == "reasoning":
            raw["events"][0]["usage"]["completion_tokens_details"]["reasoning_tokens"] = 1
        if defect == "cost":
            del raw["events"][0]["usage"]["cost"]
        if defect == "unfinished":
            raw["done"] = False
        return raw

    result = run(value, ds, transport)
    assert result["status"] != "rows_complete"
    assert all(len(list(p.glob("attempt-*"))) == 1 for p in (Path(value["output"]) / "rows").iterdir())
    assert not list((Path(value["output"]) / "rows").glob("*/accepted.json"))
    assert len(calls) <= 2


def test_preoutput_sse502_retries_same_payload_and_preserves_receipts(fixture):
    value, ds = fixture
    calls = {}

    async def transport(payload):
        key = m.seal(payload)
        calls[key] = calls.get(key, 0) + 1
        if calls[key] == 1:
            return {
                "status_code": 200,
                "done": True,
                "events": [{"error": {"code": 502, "message": "synthetic rejection"}}],
            }
        return response()

    result = run(value, ds, transport)
    assert result["status"] == "rows_complete" and all(n == 2 for n in calls.values())
    assert result["charged_usd"] >= 2 * m.BOUND_USD
    assert len(m.audit_rows(value, ds, value["output"], spec_sha256="f" * 64)) == 2


def test_sse502_after_output_is_not_retryable():
    raw = response()
    raw["events"].append({"error": {"code": 502}})
    assert m.transient_code(raw) is None


def test_adaptive429_reduces_shared_limits():
    gate = m.Gate()
    gate.penalize()
    assert gate.limit == 50 and gate.rps == 12.5 and gate.cooldown > 0


def test_explicit_resume_settled_rejection_continues_attempt_one(fixture):
    value, ds = fixture
    out = Path(value["output"])
    out.mkdir()
    (out / "rows").mkdir()
    m.write(out / "pins.json", value)
    m.write(out / "claim.json", {"spec_sha256": "f" * 64})
    m.write(out / "status.json", {"status": "interrupted"})
    payload, asset = m.make_payload(value, ds, ds.corpus[0])
    row = out / "rows/000000000"
    row.mkdir()
    m.write(row / "claim.json", m.row_claim(value, ds.corpus[0], 0, asset, payload, "f" * 64))
    m.write(row / "request.json", payload)
    attempt = row / "attempt-0"
    attempt.mkdir()
    raw = {"status_code": 429, "events": [], "done": False}
    m.write(attempt / "reservation.json", {"request_sha256": m.seal(payload), "bound_usd": m.BOUND_USD})
    m.write(attempt / "raw-response.json", raw)
    m.write(attempt / "settled.json", {**m.charge(raw), "raw_response_sha256": m.seal(raw)})

    async def transport(p):
        return response()

    result = run(value, ds, transport, resume=True)
    assert result["status"] == "rows_complete"
    assert m.read(row / "accepted.json")["attempt"] == 1
    assert len(list(row.glob("attempt-*"))) == 2


def test_uncertain_inflight_reservation_never_replayed(fixture):
    value, ds = fixture

    async def transport(p):
        return response()

    assert run(value, ds, transport)["status"] == "rows_complete"
    out = Path(value["output"])
    row = out / "rows/000000000"
    (row / "accepted.json").unlink()
    (row / "attempt-0/settled.json").unlink()
    m.write(out / "status.json", {"status": "interrupted"}, fresh=False)

    async def forbidden(p):
        pytest.fail("uncertain request replayed")

    with pytest.raises(FileNotFoundError):
        run(value, ds, forbidden, resume=True)


def test_source_mutation_during_transport_rejected(fixture):
    value, ds = fixture

    async def transport(p):
        (ds.root / "media/0.jpg").write_bytes(b"mutated")
        return response()

    assert run(value, ds, transport)["status"] != "rows_complete"
    assert not (Path(value["output"]) / "rows/000000000/accepted.json").exists()


def test_concurrency_bounded100_continuous(fixture):
    value, ds = fixture
    original = copy.deepcopy(ds.corpus[0])
    ds.corpus[:] = [dict(copy.deepcopy(original), id=str(i)) for i in range(220)]
    value["counts"]["corpus"] = 220
    active, maximum = 0, 0

    async def transport(p):
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        await asyncio.sleep(0.01)
        active -= 1
        return response()

    result = run(value, ds, transport)
    assert result["status"] == "rows_complete" and 2 <= maximum <= 100


def test_known_cost_and_reservation_prevent_cap_overdispatch(fixture):
    value, ds = fixture
    original = copy.deepcopy(ds.corpus[0])
    ds.corpus[:] = [dict(copy.deepcopy(original), id=str(i)) for i in range(2)]

    async def transport(p):
        return response(cost=m.BOUND_USD)

    # Resume ledger is reconstructed from receipts rather than trusted status totals.
    result = run(value, ds, transport)
    assert result["charged_usd"] + result["reserved_usd"] <= 15
    assert result["attempts"] == 2


def test_reservation_near_cap_stops_new_posts_even_with_many_pending(fixture, monkeypatch):
    value, ds = fixture
    original = copy.deepcopy(ds.corpus[0])
    ds.corpus[:] = [dict(copy.deepcopy(original), id=str(i)) for i in range(220)]
    value['counts']['corpus'] = 220
    output = Path(value['output'])
    output.mkdir()
    (output / 'rows').mkdir()
    (output / 'rows/000000000').mkdir()
    m.write(output / 'pins.json', value)
    m.write(output / 'claim.json', {'spec_sha256': 'f' * 64})
    m.write(output / 'status.json', {'status': 'interrupted'})
    # Inject a previously independently validated ledger total, not status.json's total.
    monkeypatch.setattr(m, 'audit_row', lambda *a, **kw: ({'source_id': '0'}, 14.95))
    calls = []
    async def transport(payload):
        calls.append(1)
        await asyncio.sleep(.03)
        return response(cost=m.BOUND_USD)
    result = run(value, ds, transport, resume=True)
    assert result['status'] == 'budget_stop'
    assert 0 < len(calls) <= int((15 - 14.95) / m.BOUND_USD)
    assert result['charged_usd'] + result['reserved_usd'] <= 15
    assert result['reserved_usd'] == 0


def test_graceful_interrupt_between_attempts_and_explicit_resume(fixture, monkeypatch):
    value, ds = fixture
    seen = {}
    callbacks = {}
    async def first():
        loop = asyncio.get_running_loop()
        monkeypatch.setattr(loop, 'add_signal_handler', lambda sig, callback, *args: callbacks.update({sig: (callback, args)}))
        async def transport(payload):
            key = m.seal(payload)
            seen[key] = seen.get(key, 0) + 1
            callback, args = callbacks[m.signal.SIGTERM]
            loop.call_soon(callback, *args)  # Invoke the registered callback, never signal a real process.
            return {'status_code': 429, 'events': [], 'done': False}
        return await m.run(value, ds, transport, spec_sha256='f' * 64, gate=FastGate())
    result = asyncio.run(first())
    assert result['status'] == 'interrupted' and not result['failed']
    async def healthy(payload): return response()
    resumed = run(value, ds, healthy, resume=True)
    assert resumed['status'] == 'rows_complete'
    assert len(list((Path(value['output']) / 'sessions').iterdir())) == 2
    assert len(m.audit_rows(value, ds, value['output'], spec_sha256='f' * 64)) == 2

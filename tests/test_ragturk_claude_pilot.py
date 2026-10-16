"""Pure fixtures and fake children only; no CLI, network, DB, or model calls."""

import importlib.util
from pathlib import Path
import subprocess

import pytest

spec = importlib.util.spec_from_file_location(
    "claude_pilot_tests", Path(__file__).parents[1] / "scripts/run_ragturk_claude_pilot.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def fixtures():
    chunks = [{"id": "s", "title": "Synthetic title", "text": "Synthetic source", "gold": "GOLD_SENTINEL"}]
    rows = [
        {
            "query_id": "q",
            "question": "Synthetic question",
            "split": "dev",
            "gold": "GOLD_SENTINEL",
            "methods": {
                method: {"method": method, "query_id": "q", "candidate_ids": ["s"], "candidate_scores": [1]}
                for method in m.METHODS
            },
        }
    ]
    return chunks, rows


def test_exact_prompt_dedup_preserves_all_five_aliases_without_gold():
    chunks, rows = fixtures()
    jobs = m.select_jobs(rows, chunks, "provider", 3)
    assert len(jobs) == 1
    assert [a["method"] for a in jobs[0]["aliases"]] == list(m.METHODS)
    assert all(a["reranker"] == "none" for a in jobs[0]["aliases"])
    assert "GOLD_SENTINEL" not in str(jobs)


def test_deterministic_order_limit_and_distinct_contexts():
    chunks, rows = fixtures()
    chunks.append({"id": "s2", "title": "", "text": "Other source"})
    rows[0]["methods"]["bge"]["candidate_ids"] = ["s2"]
    first = m.select_jobs(rows, chunks, "provider", 1)
    all_jobs = m.select_jobs(rows, chunks, "provider", 100)
    assert first == all_jobs[:1]
    assert len(all_jobs) == 2
    assert first[0]["aliases"][0]["method"] == "bm25"


@pytest.mark.parametrize("limit", [0, 101, True])
def test_bounded_requests(limit):
    chunks, rows = fixtures()
    with pytest.raises(ValueError, match="request_limit"):
        m.select_jobs(rows, chunks, "provider", limit)


def test_unknown_source_fails_before_dispatch():
    chunks, rows = fixtures()
    rows[0]["methods"]["bm25"]["candidate_ids"] = ["absent"]
    with pytest.raises(ValueError, match="candidate_identity"):
        m.select_jobs(rows, chunks, "provider", 3)


class Child:
    def __init__(self, rc=0):
        self.rc = None
        self.final = rc
        self.terminated = False
        self.killed = False

    def poll(self):
        return self.rc

    def terminate(self):
        self.terminated = True
        self.rc = -15

    def wait(self, timeout):
        return self.rc

    def kill(self):
        self.killed = True
        self.rc = -9


def test_scheduler_bounded_three_children_and_exact_dispatch():
    children, peaks = [], []

    def launch(index, job):
        child = Child()
        children.append(child)
        peaks.append(sum(c.poll() is None for c in children))
        return child

    def pause(_):
        next(c for c in children if c.rc is None).rc = 0

    result = m.schedule(
        [{}] * 7, launch, concurrency=3, read_result=lambda i, rc: {"status": "completed"}, pause=pause
    )
    assert result["requests_dispatched"] == 7
    assert len(result["results"]) == 7
    assert max(peaks) == 3


def test_first_failure_stops_admission_and_terminates_only_owned_siblings():
    children = []

    def launch(index, job):
        child = Child()
        children.append(child)
        return child

    def pause(_):
        children[0].rc = 1

    result = m.schedule(
        [{}] * 20, launch, concurrency=3, read_result=lambda i, rc: {"status": "failed"}, pause=pause
    )
    assert result["status"] == "failed"
    assert result["requests_dispatched"] == 3
    assert all(c.terminated for c in children[1:])


def test_exit_zero_without_complete_receipt_is_failure():
    child = Child()
    child.rc = 0
    result = m.schedule(
        [{}] * 3,
        lambda i, j: child,
        concurrency=1,
        read_result=lambda i, rc: {"status": "partial"},
        pause=lambda _: None,
    )
    assert result["status"] == "failed"
    assert result["requests_dispatched"] == 1


def test_interrupt_cleanup_escalates_bounded_owned_child():
    class Slow(Child):
        def wait(self, timeout):
            if not self.killed:
                raise subprocess.TimeoutExpired("synthetic", timeout)
            return -9

    child = Slow()

    def pause(_):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        m.schedule([{}], lambda i, j: child, concurrency=1, read_result=lambda i, rc: {}, pause=pause)
    assert child.terminated and child.killed


def test_seal_mutation_and_duplicate_private_output(tmp_path):
    p = tmp_path / "pin.json"
    m.private(p, {"synthetic": 1})
    seals = {str(p): m.sha(p)}
    m.verify_seals(seals)
    with pytest.raises(FileExistsError):
        m.private(p, {"synthetic": 2})
    p.write_text("{}")
    with pytest.raises(ValueError, match="pinned_input_changed"):
        m.verify_seals(seals)


@pytest.mark.parametrize("mutation", [None, "not_closed", "block_sha", "question", "source_identity"])
def test_closed_producer_rejects_mutations_without_retrieval(monkeypatch, tmp_path, mutation):
    from types import SimpleNamespace
    import json

    root = tmp_path
    (root / "scripts").mkdir()
    (root / "scripts/prepare_ragturk_candidates.py").write_text("# synthetic helper")
    (root / "src/rag_benchmark").mkdir(parents=True)
    (root / "src/rag_benchmark/claude_cli.py").write_text("# synthetic transport")
    producer = root / "producer"
    producer.mkdir()
    (producer / "blocks").mkdir()
    bundle = root / "bundle.json"
    bundle.write_text("{}")
    chunks, template = fixtures()
    rows, queries = [], []
    for i in range(20):
        qid = f"q{i}"
        query = {"id": qid, "question": "Synthetic question"}
        queries.append(query)
        rows.append(
            {
                "query_id": qid,
                "question": query["question"],
                "split": "dev",
                "methods": {key: {**value, "query_id": qid} for key, value in template[0]["methods"].items()},
            }
        )
    prepared = SimpleNamespace(
        identity="bundle-id",
        refs={},
        chunks=chunks,
        queries=queries,
        splits=["dev"] * 20,
        stable_inputs=lambda: None,
    )
    helper_sha = m.sha(root / "scripts/prepare_ragturk_candidates.py")
    helper = SimpleNamespace(
        load_prepared=lambda _: prepared,
        PROTOCOL="synthetic",
        seal=m.digest,
        select=lambda *a: list(range(20)),
    )
    monkeypatch.setattr(m, "ROOT", root)
    monkeypatch.setattr(m, "verified_module", lambda _: (helper, helper_sha))
    manifest = {
        "protocol": "synthetic",
        "pilot": True,
        "queries": 20,
        "limit": 20,
        "bundle_identity": prepared.identity,
        "bundle_sha256": m.sha(bundle),
        "source_file_sha256": {},
        "code_sha256": helper_sha,
        "labels_used": False,
        "methods": list(m.METHODS),
        "corpus_order_sha256": m.digest(["s"]),
        "split": "all",
    }
    if mutation == "source_identity":
        manifest["bundle_identity"] = "wrong"
    identity = m.digest(manifest)
    manifest["identity"] = identity
    if mutation == "question":
        rows[0]["question"] = "Altered source question"
    block = {"protocol": "synthetic", "identity": identity, "start": 0, "stop": 20, "rows": rows}
    blockpath = producer / "blocks/000.json"
    blockpath.write_text(json.dumps(block))
    entries = [{"name": "000.json", "sha256": m.sha(blockpath), "rows": 20}]
    if mutation == "block_sha":
        blockpath.write_text("{}")
    values = {
        "manifest.json": manifest,
        "summary.json": {
            "identity": identity,
            "status": "completed",
            "labels_used": False,
            "new_model_calls": 0,
            "methods": list(m.METHODS),
            "queries": 20,
            "corpus_rows": 1,
            "blocks": entries,
        },
        "state.json": {"identity": identity, "status": "completed", "completed_rows": 20, "blocks": entries},
        "claim.json": {"identity": identity},
    }
    if mutation == "not_closed":
        values["state.json"]["status"] = "running"
    for name, value in values.items():
        (producer / name).write_text(json.dumps(value))
    if mutation is None:
        actual, actual_rows, seals = m.closed_inputs(producer, bundle)
        assert actual is prepared and actual_rows == rows
        m.verify_seals(seals)
    else:
        with pytest.raises(ValueError):
            m.closed_inputs(producer, bundle)


def test_repeated_real_self_signals_reap_all_owned_workers_and_restore_handlers():
    """No subprocess is spawned; real signals exercise this test's scoped handlers."""
    import os
    import signal

    before = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    children, waited = [], []

    def cancel(_sig, _frame):
        raise KeyboardInterrupt("synthetic scheduler cancellation")

    class RepeatedSignalChild(Child):
        def terminate(self):
            for sig in (signal.SIGTERM, signal.SIGINT):
                assert signal.getsignal(sig) == signal.SIG_IGN
                os.kill(os.getpid(), sig)
            super().terminate()

        def wait(self, timeout):
            for sig in (signal.SIGTERM, signal.SIGINT):
                assert signal.getsignal(sig) == signal.SIG_IGN
                os.kill(os.getpid(), sig)
            waited.append(self)
            return super().wait(timeout)

    def launch(index, job):
        child = RepeatedSignalChild()
        children.append(child)
        return child

    def pause(_):
        os.kill(os.getpid(), signal.SIGTERM)
        raise AssertionError("initial cancellation was not delivered")

    try:
        for sig in before:
            signal.signal(sig, cancel)
        scoped = {sig: signal.getsignal(sig) for sig in before}
        with pytest.raises(KeyboardInterrupt):
            m.schedule([{}] * 20, launch, concurrency=3, read_result=lambda *a: {}, pause=pause)
        assert {sig: signal.getsignal(sig) for sig in before} == scoped
    finally:
        for sig, handler in before.items():
            signal.signal(sig, handler)
    assert len(children) == len(waited) == 3
    assert all(child.terminated and not child.killed for child in children)

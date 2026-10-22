"""CPU fixtures only: no model load, real dataset reads, or GPU work."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

PATH = Path(__file__).parents[1] / "scripts/run_fleurs_paraphrase_pilot.py"
spec = importlib.util.spec_from_file_location("fleurs_pilot_runner", PATH)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class FakeClock:
    def __init__(self):
        self.now = 100

    def tick(self):
        self.now += 1_000_000
        return self.now


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    events = []
    clock = FakeClock()
    monkeypatch.setattr(m.time, "perf_counter_ns", clock.tick)
    corpus = [{"id": f"d{i:03d}", "text": "ortak belge içeriği", "media": {}} for i in range(60)]
    queries = [{"id": f"pp{i:02d}", "text": "ortak özel sorgu", "media": {}} for i in range(1, 3)]
    dataset = SimpleNamespace(corpus=corpus, queries=queries, identity="fixture-dataset",
        qrels={q["id"]: {"d000": 1, "d001": 1} for q in queries},
        model_item=lambda row, text_only: {k: row[k] for k in ("id", "text", "media")})
    vectors = np.zeros((60, 2), dtype=np.float32)
    vectors[:, 0] = 1
    pilot = SimpleNamespace(dataset=dataset, document_vectors={c: vectors.copy() for c in "GENJ"},
                            manifest_sha256="fixture-manifest")
    class Backend:
        def __init__(self, channel, _pilot):
            self.channel = channel
            events.append(("construct", channel))

        def load(self):
            events.append(("load", self.channel))

        def warmup(self):
            events.append(("warmup", self.channel))

        def public_spec(self):
            return {"device": "fake_cpu", "batch_size": 1}

        def encode(self, item):
            assert set(item) == {"id", "text", "media"}
            assert item["media"] == {}
            events.append(("encode", self.channel, item["id"]))
            return np.asarray([[1, 0]], dtype=np.float32)

        def score(self, query, documents):
            assert all(set(row) == {"id", "text", "media"} for row in documents)
            events.append(("score", query["id"], len(documents)))
            return [1 / (i + 1) for i in range(len(documents))]

        def unload(self):
            events.append(("unload", self.channel))
    session = "fixture-session"
    (tmp_path / "sessions" / session).mkdir(parents=True)
    store = m.ReceiptStore(tmp_path / "jobs", "identity")
    def sync():
        events.append(("sync",))
    return SimpleNamespace(pilot=pilot, backend=Backend, events=events, store=store,
                           session=session, sync=sync, root=tmp_path)


def subset_jobs(f, channels="GE", mode="total"):
    pair = {"channels": list(channels), "budget_mode": mode, "key": f"{channels}-{mode}"}
    return [(pair, q) for q in f.pilot.dataset.queries]


def run(f, **kwargs):
    return m.run_pilot(f.pilot, f.store, f.session, backend_factory=f.backend,
                       sync=f.sync, jobs=kwargs.pop("jobs", subset_jobs(f)), **kwargs)


def test_paired_real_endpoints_fresh_calls_same_candidates_and_timer_scope(fixture, monkeypatch):
    f = fixture
    original_metrics = m.graded_metrics
    def metrics(*args, **kwargs):
        f.events.append(("metric",))
        return original_metrics(*args, **kwargs)
    monkeypatch.setattr(m, "graded_metrics", metrics)
    summary = run(f)
    assert summary["cell_count"] == 2 and summary["paired_query_invocations"] == 2
    assert summary["completed_query_cells"] == 4 and summary["pipeline_end_to_end_measured"]
    assert summary["new_query_inference_count_this_launch"] == 4
    assert [e for e in f.events if e[0] == "encode"] == [("encode", c, q["id"])
        for q in f.pilot.dataset.queries for c in "GE"]
    assert [e for e in f.events if e[0] == "score"] == [("score", "pp01", 50), ("score", "pp02", 50)]
    assert max(i for i, e in enumerate(f.events) if e[0] == "warmup") < min(
        i for i, e in enumerate(f.events) if e[0] == "encode")
    first_score = next(i for i, e in enumerate(f.events) if e[0] == "score")
    first_metric = next(i for i, e in enumerate(f.events) if e[0] == "metric")
    assert first_score < first_metric and f.events[first_metric - 1] == ("sync",)
    for path in (f.root / "jobs").glob("*/result.json"):
        row = m.read_json(path)["value"]
        ends = row["monotonic_endpoints_ns"]
        assert ends["t0"] < ends["t1"] < ends["t2"]
        assert row["measured_pipeline_end_to_end_seconds"]["none"] == (ends["t1"] - ends["t0"]) / 1e9
        assert row["measured_pipeline_end_to_end_seconds"]["laya"] == (ends["t2"] - ends["t0"]) / 1e9
        assert set(r["id"] for r in row["none_ranking"]) == set(r["id"] for r in row["laya_ranking"])
        assert row["score_cache_pairs"] == row["query_cache_hits"] == row["document_inference_count"] == 0
        assert row["shared_prefix_invocations"] == 1
    assert "ortak özel sorgu" not in json.dumps(summary, ensure_ascii=False)
    assert "d000" not in json.dumps(summary)
    assert summary["human_validated"] is False and summary["semantic_ground_truth_verified"] is False


def test_closed_resume_skips_all_models_and_keeps_original_measurements(fixture):
    first = run(fixture)
    original = {p: p.read_bytes() for p in (fixture.root / "jobs").rglob("*.json")}
    fixture.events.clear()
    second = run(fixture)
    assert fixture.events == []
    assert second["setup_this_launch"] == []
    assert second["new_query_inference_count_this_launch"] == 0
    assert second["new_shared_query_prefixes_this_launch"] == 0
    assert second["recorded_measurements_time_range"] == first["recorded_measurements_time_range"]
    assert all(c["reused_closed_queries_this_launch"] == 2 for c in second["cells"])
    assert {p: p.read_bytes() for p in original} == original


def test_incomplete_claim_is_never_reissued(fixture):
    target = fixture.store.location("pair-GE-total-pp01")
    target.mkdir()
    m.fresh_json(target / "claim.json", {"incomplete": True})
    with pytest.raises(ValueError, match="receipt_namespace"):
        run(fixture)
    assert fixture.events == []


def test_changed_query_or_doc_vector_rejects_existing_result(fixture):
    run(fixture)
    fixture.pilot.dataset.queries[0]["text"] = "changed"
    fixture.events.clear()
    with pytest.raises(ValueError, match="closed_job_binding"):
        run(fixture)
    assert not fixture.events


@pytest.mark.parametrize("mutation", ["ids", "dimensions", "norm", "nan", "missing_channel"])
def test_incompatible_vectors_and_ids_fail_closed(fixture, mutation):
    if mutation == "ids":
        fixture.pilot.dataset.corpus[1]["id"] = "d000"
    elif mutation == "dimensions":
        fixture.pilot.document_vectors["G"] = np.ones((60, 1), dtype=np.float32)
    elif mutation == "norm":
        fixture.pilot.document_vectors["G"][0] = 0
    elif mutation == "nan":
        fixture.pilot.document_vectors["G"][0, 0] = np.nan
    else:
        del fixture.pilot.document_vectors["G"]
    with pytest.raises(ValueError):
        run(fixture)
    assert not list((fixture.root / "jobs").glob("*/completed.json"))


def test_wrong_laya_count_fails_pair_without_half_closed_none(fixture):
    fixture.backend.score = lambda *_: [0.1]
    with pytest.raises(ValueError, match="laya_score_count_or_range"):
        run(fixture)
    path = next((fixture.root / "jobs").iterdir())
    assert [p.name for p in path.iterdir()] == ["claim.json"]


def test_freshness_failure_after_operation_prevents_result(tmp_path):
    count = 0
    def check():
        nonlocal count
        count += 1
        if count > 1:
            raise ValueError("changed")
    store = m.ReceiptStore(tmp_path, "run", check)
    with pytest.raises(ValueError, match="changed"):
        store.job("a", {}, lambda: {"value": 1})
    assert not (tmp_path / "a/result.json").exists()


def test_result_mutation_or_unknown_file_is_not_reused(tmp_path):
    store = m.ReceiptStore(tmp_path, "run")
    store.job("a", {}, lambda: {"value": 1})
    (tmp_path / "a/result.json").write_text('{}')
    with pytest.raises(ValueError, match="closed_job_seal"):
        store.existing("a", {})
    (tmp_path / "a/extra").write_text('')
    with pytest.raises(ValueError, match="receipt_namespace"):
        store.existing("a", {})


def test_bm25_does_not_pad_zero_scores():
    bm25 = m.BM25([{"id": "a", "text": "kedi"}, {"id": "b", "text": "köpek"}])
    assert bm25.search("kuş") == []
    assert [r["id"] for r in bm25.search("kedi")] == ["a"]


def test_small_actual_candidate_count_not_assumed_fifty(fixture):
    for row in fixture.pilot.dataset.corpus[1:]:
        row["text"] = "ilgisiz"
    summary = run(fixture, jobs=subset_jobs(fixture, "B"))
    assert [e for e in fixture.events if e[0] == "score"] == [("score", "pp01", 1), ("score", "pp02", 1)]
    assert summary["cells"][1]["recorded_fresh_full_laya_pairs"] == 2


def test_fixed_full_plan_coverage_and_seeded_order():
    items = [{"id": f"pp{i:02d}"} for i in range(1, 26)]
    jobs = m.execution_plan(items)
    assert len(m.cells()) == 124 and len(jobs) == 1550
    assert jobs == m.execution_plan(list(reversed(items)))
    assert len({pair["key"] for pair, _ in jobs}) == 62
    assert len({("".join(pair["channels"])) for pair, _ in jobs}) == 31
    assert sum(len([c for c in pair["channels"] if c != "B"]) for pair, _ in jobs) == 3200
    assert all(sum(m.fuse_rankings({c: [] for c in pair["channels"]}, pair["channels"],
        budget_mode="total")[1]["allocations"].values()) == 100 for pair, _ in jobs)


def test_shared_lock_exclusive_no_truncate(tmp_path):
    lock = tmp_path / "gpu.lock"
    lock.write_text("retained-lock-bytes")
    with m.shared_gpu_lock(lock):
        with pytest.raises(BlockingIOError):
            with m.shared_gpu_lock(lock):
                pass
    assert lock.read_text() == "retained-lock-bytes"


@pytest.mark.parametrize("kind", ["symlink", "fifo", "directory"])
def test_safe_reader_rejects_nonregular_without_blocking(tmp_path, kind):
    import os
    path = tmp_path / "target"
    if kind == "symlink":
        (tmp_path / "original").write_text("private")
        path.symlink_to(tmp_path / "original")
    elif kind == "fifo":
        os.mkfifo(path)
    else:
        path.mkdir()
    with pytest.raises((ValueError, OSError)):
        m.safe_bytes(path)


def test_safe_failure_cli_does_not_emit_private_message(tmp_path, monkeypatch, capsys):
    path = tmp_path / "config"
    path.write_text('{"private":"not-to-print"}')
    def fail(_):
        raise RuntimeError("SECRET QUERY MUST NOT PRINT")
    monkeypatch.setattr(m, "validate_config", fail)
    assert m.main(["--config", str(path), "--config-sha256", m.sha(path)]) == 1
    output = capsys.readouterr()
    assert "SECRET" not in output.err and "not-to-print" not in output.err
    assert json.loads(output.err)["status"] == "failed_requires_review"


def test_real_backend_uses_unchanged_document_query_contract_without_loading(monkeypatch):
    from rag_benchmark.models import MODEL_DEFAULTS
    called = []
    pilot = SimpleNamespace(query_adapter_configs={"G": {**MODEL_DEFAULTS["bge"],
        "device": "mps", "dtype": "bfloat16", "batch_size": 8}},
        validate_query_adapter=lambda channel, adapter, batch_size: called.append((channel, batch_size)))
    backend = m.LocalBackend("G", pilot)
    assert backend.config["batch_size"] == 1
    assert pilot.query_adapter_configs["G"]["batch_size"] == 8
    assert called == [("G", 1)] and backend.adapter.embedder._model is None


def test_p95_computed_from_real_per_query_endpoints_not_sum_of_percentiles(fixture):
    summary = run(fixture)
    assert all(c["measured_pipeline_end_to_end"]["percentile_method"] == "numpy_linear" for c in summary["cells"])
    assert summary["cells"][1]["measured_pipeline_end_to_end"]["mean_seconds"] > summary["cells"][0]["measured_pipeline_end_to_end"]["mean_seconds"]
    assert "reconstructed" not in json.dumps(summary)


def test_memory_observations_are_sampled_maxima_not_limits():
    data = iter([{"process_peak_resident_bytes": 20, "mps_driver_allocated_bytes": 10,
                  "mps_current_allocated_bytes": 9, "separate_limit_bytes": 100},
                 {"process_peak_resident_bytes": 25, "mps_driver_allocated_bytes": 5,
                  "mps_current_allocated_bytes": 3, "separate_limit_bytes": 100}])
    record = m.MemoryObservations(lambda: next(data))
    record()
    record()
    summary = record.summary()
    assert summary["observation_count"] == 2
    assert summary["observed_maxima"] == {"process_peak_resident_bytes": 25,
        "mps_driver_allocated_bytes": 10, "mps_current_allocated_bytes": 9}
    assert summary["latest"]["mps_driver_allocated_bytes"] == 5
    assert "not continuous" in summary["scope"]


def test_unknown_job_namespace_and_duplicate_plan_block_before_model_load(fixture):
    (fixture.root / "jobs/unplanned").mkdir()
    with pytest.raises(ValueError, match="unplanned_job_namespace"):
        run(fixture)
    assert fixture.events == []
    (fixture.root / "jobs/unplanned").rmdir()
    jobs = subset_jobs(fixture)
    with pytest.raises(ValueError, match="duplicate_planned_job"):
        run(fixture, jobs=jobs + jobs[:1])


def test_private_error_trace_is_retained_but_cli_safe(tmp_path, monkeypatch, capsys):
    from rag_benchmark import fleurs_pilot_data
    lock = tmp_path / "gpu.lock"
    lock.write_text("unchanged")
    config = {"output_directory": str(tmp_path / "output"), "shared_gpu_lock": str(lock),
              "memory_guard": {"minimum_physical_bytes": 1},
              "manifest": {"path": "fake-only", "sha256": "0" * 64}}
    path = tmp_path / "config.json"
    m.fresh_json(path, config)
    monkeypatch.setattr(m, "validate_config", lambda _: None)
    monkeypatch.setattr(m.subprocess, "check_output", lambda *_a, **_kw: "1024")
    def stop(*_args, **_kwargs):
        raise RuntimeError("PRIVATE_DIAGNOSTIC_SENTINEL")
    monkeypatch.setattr(fleurs_pilot_data, "load_verified_pilot", stop)
    assert m.main(["--config", str(path), "--config-sha256", m.sha(path)]) == 1
    output = capsys.readouterr()
    assert "PRIVATE_DIAGNOSTIC_SENTINEL" not in output.out + output.err
    traces = list((tmp_path / "output/sessions").glob("*/private-error-traceback.txt"))
    assert len(traces) == 1 and "PRIVATE_DIAGNOSTIC_SENTINEL" in traces[0].read_text()
    assert traces[0].stat().st_mode & 0o777 == 0o600
    failure = m.read_json(traces[0].parent / "failure.json")
    assert failure["private_traceback_sha256"] == m.sha(traces[0])

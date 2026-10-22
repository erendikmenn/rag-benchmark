"""New fake-backend quality-graph tests; frozen pilot tests are not repeated."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

PATH = Path(__file__).parents[1] / "scripts/run_fleurs_remaining_quality.py"
spec = importlib.util.spec_from_file_location("remaining_quality_runner", PATH)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


@pytest.fixture
def fixture(tmp_path):
    events = []
    corpus = [{"id": f"d{i:03d}", "text": "private document sentinel", "media": {}} for i in range(60)]
    queries = [{"id": f"ep{i:03d}", "text": "private query sentinel", "media": {}} for i in range(1, 3)]
    dataset = SimpleNamespace(identity="new-remaining", corpus=corpus, queries=queries,
        qrels={q["id"]: {"d001": 1, "d059": 1} for q in queries},
        model_item=lambda row, text_only: {k: row[k] for k in ("id", "text", "media")})
    vectors = np.zeros((60, 2), dtype=np.float32)
    vectors[:, 0] = 1
    data = SimpleNamespace(dataset=dataset, document_vectors={c: vectors.copy() for c in "GENJ"},
                           manifest_sha256="new-data-config")
    class Backend:
        def __init__(self, channel, data):
            self.channel = channel
            self.adapter = SimpleNamespace(pointwise=True)
            self.identity = "identity-" + channel
            self.config = {"backend": "sdk", "threshold": None, "batch_size": 8, "device": "mps",
                           "local_files_only": True}
            events.append(("construct", channel))

        def load(self):
            events.append(("load", self.channel))

        def warmup(self):
            events.append(("warmup", self.channel))

        def encode(self, item):
            assert set(item) == {"id", "text", "media"}
            events.append(("encode", self.channel, item["id"]))
            return np.asarray([[1, 0]], dtype=np.float32)

        def score(self, query, documents):
            assert all(set(d) == {"id", "text", "media"} for d in documents)
            events.append(("score", query["id"], [d["id"] for d in documents]))
            return [int(d["id"][1:]) / 100 for d in documents]

        def public_spec(self):
            return {"device": "fake_cpu", "batch_size": 1}

        def unload(self):
            events.append(("unload", self.channel))
    class Index:
        def __init__(self, corpus):
            events.append(("index",))

        def search(self, text):
            return [{"id": f"d{i:03d}", "score": float(i + 1)} for i in range(59, 39, -1)]
    session = "fixture-session"
    (tmp_path / "sessions" / session).mkdir(parents=True)
    store = m.QualityStore(tmp_path / "jobs", "identity")
    return SimpleNamespace(data=data, backend=Backend, index=Index, events=events, session=session,
                           store=store, root=tmp_path, sync=lambda: events.append(("sync",)))


def run(f):
    return m.run_quality(f.data, f.store, f.session, backend_factory=f.backend,
                         index_factory=f.index, sync=f.sync)


def test_single_fresh_graph_all124_quality_conditions(fixture):
    f = fixture
    summary = run(f)
    assert len(summary["cells"]) == 124 and summary["completed_query_cells"] == 248
    assert summary["new_query_inference_count_this_launch"] == 8
    assert len([e for e in f.events if e[0] == "encode"]) == 8
    scores = [e for e in f.events if e[0] == "score"]
    assert len(scores) == 2 and all(len(e[2]) == len(set(e[2])) == 60 for e in scores)
    assert summary["recorded_fresh_unique_laya_pairs"] == 120
    assert summary["within_query_score_reuses"] == summary["candidate_pair_occurrences"] - 120
    assert not summary["pipeline_end_to_end_measured"] and not summary["per_cell_latency_measured"]
    assert all(not cell["per_cell_latency_measured"] for cell in summary["cells"])
    assert "private" not in json.dumps(summary) and "d001" not in json.dumps(summary)
    assert summary["pilot_queries_reissued"] == 0 and summary["combined_329_report"] is False
    assert max(i for i, e in enumerate(f.events) if e[0] == "warmup") < min(
        i for i, e in enumerate(f.events) if e[0] == "encode")
    for path in (f.root / "jobs").glob("*/result.json"):
        value = m.p.read_json(path)["value"]
        assert len(value["cells"]) == 62
        for cell in value["cells"]:
            ids = [r["id"] for r in cell["candidates"]]
            final = [r["id"] for r in cell["laya_ranking"]]
            assert set(ids) == set(final) and final == sorted(ids, reverse=True)
            assert len(ids) <= 50


def test_resume_reuses_exact_closed_queries_with_zero_models(fixture):
    first = run(fixture)
    saved = {p: p.read_bytes() for p in (fixture.root / "jobs").rglob("*.json")}
    fixture.events.clear()
    second = run(fixture)
    assert fixture.events == [] and second["setup_this_launch"] == []
    assert second["new_query_inference_count_this_launch"] == 0
    assert second["new_fresh_unique_laya_pairs_this_launch"] == 0
    assert second["cells"] == first["cells"]
    assert second["recorded_measurements_time_range"] == first["recorded_measurements_time_range"]
    assert all(p.read_bytes() == value for p, value in saved.items())


def test_incomplete_query_never_loads_model_or_reissues(fixture):
    path = fixture.store.location("query-ep001")
    path.mkdir()
    m.p.fresh_json(path / "claim.json", {"incomplete": True})
    with pytest.raises(ValueError, match="receipt_namespace"):
        run(fixture)
    assert not fixture.events


@pytest.mark.parametrize("kind", ["query", "document", "qrels", "data_identity", "vectors", "unknown_job"])
def test_closed_inputs_and_namespace_cannot_drift(fixture, kind):
    f = fixture
    run(f)
    f.events.clear()
    if kind == "query":
        f.data.dataset.queries[0]["text"] = "different private text"
    elif kind == "document":
        f.data.dataset.corpus[0]["text"] = "changed"
    elif kind == "qrels":
        f.data.dataset.qrels["ep001"]["d001"] = 0
    elif kind == "data_identity":
        f.data.manifest_sha256 = "different"
    elif kind == "vectors":
        f.data.document_vectors["E"][0] = [0, 1]
    else:
        (f.store.root / "query-pp01").mkdir()
    with pytest.raises(ValueError):
        run(f)
    assert not f.events


@pytest.mark.parametrize("kind", ["wrong_count", "nan", "negative", "too_large", "not_pointwise", "http"])
def test_laya_contract_fail_closed(fixture, kind):
    f = fixture
    original = f.backend
    class Bad(original):
        def __init__(self, channel, data):
            super().__init__(channel, data)
            if kind == "not_pointwise":
                self.adapter.pointwise = False
            if kind == "http":
                self.config["backend"] = "http"

        def score(self, query, documents):
            values = super().score(query, documents)
            if kind == "wrong_count":
                return values[:-1]
            if kind in ("nan", "negative", "too_large"):
                values[0] = {"nan": float("nan"), "negative": -0.1, "too_large": 1.1}[kind]
            return values
    f.backend = Bad
    with pytest.raises(ValueError):
        run(f)
    assert not list(f.store.root.glob("*/completed.json"))


def test_scatter_ties_keep_each_candidate_order_and_reject_nonunion():
    graph = [{"candidates": [{"id": "b"}, {"id": "a"}]}]
    rows = m.scatter_scores(graph, ["a", "b"], [0.5, 0.5], {"a": 1})
    assert [r["id"] for r in rows[0]["laya_ranking"]] == ["b", "a"]
    with pytest.raises(ValueError, match="quality_union_membership"):
        m.scatter_scores(graph, ["a", "other"], [0.5, 0.5], {"a": 1})


def test_short_bm25_not_padded_or_refilled():
    base = {c: [{"id": f"d{i}", "score": 100 - i} for i in range(100)] for c in "GENJ"}
    base["B"] = [{"id": "d0", "score": 2}]
    rows = {r["key"]: r for r in m.candidate_graph(base, [f"d{i}" for i in range(100)])}
    assert len(rows["B-total"]["candidates"]) == 1
    assert rows["BGENJ-total"]["pool"]["allocations"] == {"B": 20, "G": 20, "E": 20, "N": 20, "J": 20}
    assert rows["BGENJ-total"]["pool"]["raw_candidates"] == 81
    assert rows["BGENJ-total"]["pool"]["refill"] is False


def test_pre_and_post_job_source_validation(fixture):
    f = fixture
    count = 0
    def validate():
        nonlocal count
        count += 1
        if count == 2:
            raise ValueError("source_changed")
    f.store.validate = validate
    with pytest.raises(ValueError, match="source_changed"):
        run(f)
    assert len(list(f.store.root.glob("*/claim.json"))) == 1
    assert not list(f.store.root.glob("*/result.json"))

"""Offline schedule contracts; no model or head loader invocation."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from rag_benchmark import multimodal_bge_extra_runtime as runtime


class Baseline:
    identity = "verified-baseline"
    config = {"batch_size": 8}

    def __init__(self, rows=145, chunks=False):
        self.rows = [
            {
                "id": str(i),
                "source_id": "parent" if chunks else str(i),
                "formatted_text": "x" * (i % 11 + 1),
                "chunk_index": i,
            }
            for i in range(rows)
        ]

    def validate_for_reuse(self):
        pass

    def verified_inputs(self, role):
        return copy.deepcopy(self.rows), np.empty((len(self.rows), 1024), dtype=np.float32)


@pytest.fixture
def evidence(monkeypatch):
    monkeypatch.setattr(
        runtime, "scheduling_evidence", lambda: {"audited_fake": True, "numpy": np.__version__}
    )


@pytest.mark.parametrize("chunks,block", [(False, 128), (True, 32)])
def test_invocation_boundaries_and_numpy_ties(evidence, chunks, block):
    baseline = Baseline(chunks=chunks)
    plan = runtime.PinnedBatchPlan(baseline)
    records, _ = baseline.verified_inputs("document")
    actual = plan.order_for("document", records)
    expected = [
        start + int(i)
        for start in range(0, len(records), block)
        for i in np.argsort([-len(r["formatted_text"]) for r in records[start : start + block]])
    ]
    assert actual == expected
    assert sorted(actual) == list(range(len(records)))
    assert plan.order_for("query", records) == expected
    assert plan.summary()["roles"]["query"]["invocation_rows"] == block
    assert not plan.summary()["production_runtime_available"]


def test_changed_row_config_or_evidence_rejected(evidence, monkeypatch):
    baseline = Baseline()
    plan = runtime.PinnedBatchPlan(baseline)
    records, _ = baseline.verified_inputs("query")
    records[0]["formatted_text"] = "changed"
    with pytest.raises(ValueError, match="row identity"):
        plan.order_for("query", records)
    baseline.config = {"batch_size": 4}
    with pytest.raises(ValueError, match="changed"):
        plan.order_for("query", baseline.rows)
    baseline.config = {"batch_size": 8}
    monkeypatch.setattr(runtime, "scheduling_evidence", lambda: {"changed": True})
    with pytest.raises(ValueError, match="changed"):
        plan.order_for("query", baseline.rows)


def test_unavailable_factory_never_calls_loader():
    def forbidden():
        pytest.fail("model/head loading forbidden")

    with pytest.raises(ValueError, match="unavailable"):
        runtime.production_factory(loader=forbidden)


def test_missing_scheduling_contract_rejected(monkeypatch):
    monkeypatch.setattr(runtime.native, "normalization_evidence", lambda: {})
    monkeypatch.setattr(
        runtime,
        "distribution",
        lambda _: SimpleNamespace(version="6.1.0", locate_file=lambda _: "/not-present"),
    )
    with pytest.raises(ValueError, match="unavailable"):
        runtime.scheduling_evidence()

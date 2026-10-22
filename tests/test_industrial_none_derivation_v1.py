"""New CPU derivation tests use memory/tiny files only, never actual benchmark data."""
import copy
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import types

import pytest

from rag_benchmark import industrial_none_derivation_v1 as run
from rag_benchmark import multimodal as matrix

PURE = {name: getattr(matrix, name) for name in ("fuse_rankings", "graded_metrics", "aggregate_metrics")}


def examples():
    ids = ["q1", "q2"]
    qrels = {"q1": {"d2": 2.0}, "q2": {"d3": 1.0}}
    dense = [[{"id": "d1", "score": .9, "rank": 1}, {"id": "d2", "score": .7, "rank": 2},
              {"id": "d3", "score": .2, "rank": 3}],
             [{"id": "d3", "score": .9, "rank": 1}, {"id": "d2", "score": .6, "rank": 2},
              {"id": "d1", "score": .1, "rank": 3}]]
    rankings = {c: copy.deepcopy(dense) for c in "GENSJ"}
    rankings["B"] = [[{"id": "d1", "score": 2.0, "rank": 1}], []]
    identities = {**run.origin.INHERITED_IDS, "S": "synthetic-S", "J": "synthetic-J"}
    return ids, qrels, rankings, identities


def prior(channel):
    ids, qrels, rankings, identities = examples()
    config = {"engine": "multimodal-v1", "dataset": "synthetic-dataset", "candidate_grid": [20, 50, 100],
              "budget_modes": list(run.MODES), "channel_budget": 100, "rrf_k": 60,
              "requested_channels": [channel], "query_ids": ids}
    report = {"configuration": config, "dataset": {"identity": "synthetic-dataset"},
              "scope": "frozen_collection", "evaluated_query_count": 2,
              "channels": {channel: {"status": "completed", "identity": identities[channel]}}, "cells": []}
    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE results (cell TEXT, query_id TEXT, payload TEXT, PRIMARY KEY(cell,query_id))")
    for mode in run.MODES:
        variant = "document__" + channel.lower() + "__none"
        identity = run.frozen.digest({**config, "variant": variant, "mode": mode, "candidate_k": None,
            "channel_identities": {channel: identities[channel]}, "reranker": None, "reranker_prompts": None})
        rows, pools = [], []
        for index, qid in enumerate(ids):
            ranking, pool = matrix.fuse_rankings({channel: rankings[channel][index]}, [channel], budget_mode=mode)
            metrics = matrix.graded_metrics([r["id"] for r in ranking], qrels[qid])
            value = {"query_id": qid, "ranking": ranking, "pool": {**pool, "rerank_candidates": 0}, "metrics": metrics}
            db.execute("INSERT INTO results VALUES (?,?,?)", (identity, qid, json.dumps(value)))
            rows.append(metrics)
            pools.append(pool)
        report["cells"].append({"variant_id": variant, "budget_mode": mode, "candidate_k": None,
            "identity": identity, "status": "completed", "query_count": 2, "completed_queries": 2,
            "metrics": matrix.aggregate_metrics(rows),
            "mean_raw_candidates": sum(p["raw_candidates"] for p in pools) / 2,
            "mean_unique_candidates": sum(p["unique_candidates"] for p in pools) / 2,
            "mean_rerank_candidates": 0.0})
    db.commit()
    raw = db.serialize()
    db.close()
    return report, raw


def agreement(channel, report, raw):
    ids, qrels, rankings, _ = examples()
    return run.prior_agreement(channel, report, raw, rankings[channel], ids, qrels, PURE, "synthetic-dataset")


def test_exact_126_cells_17_metrics_two_modes_distinct_mixed_ids():
    ids, qrels, rankings, identities = examples()
    config = run.configuration("synthetic-dataset", ids, "mixed-A")
    cells = list(run.derive_rows(rankings, ids, qrels, identities, config, PURE))
    assert len(cells) == 126 and len({c["identity"] for c, _ in cells}) == 126
    assert {c["budget_mode"] for c, _ in cells} == set(run.MODES)
    assert all(c["candidate_k"] is None and c["completed_queries"] == 2 and set(c["metrics"]) == run.METRICS
               and len(rows) == 2 for c, rows in cells)
    changed = list(run.derive_rows(rankings, ids, qrels, identities,
                                  run.configuration("synthetic-dataset", ids, "mixed-B"), PURE))
    assert all(a[0]["identity"] != b[0]["identity"] and a[0]["metrics"] == b[0]["metrics"]
               for a, b in zip(cells, changed, strict=True))


def test_full_union_is_not_truncated_and_total_budget_has_no_refill():
    rankings = {}
    for index, channel in enumerate(run.CHANNELS):
        rankings[channel] = [[{"id": f"d{index * 100 + i:03d}", "score": float(100 - i), "rank": i + 1}
                              for i in range(100)]]
    config = run.configuration("dataset", ["q"], "mixed")
    result = {(c["variant_id"], c["budget_mode"]): rows for c, rows in run.derive_rows(
        rankings, ["q"], {"q": {"d599": 1}}, {c: c for c in run.CHANNELS}, config, PURE)}
    name = "document__b_g_e_n_s_j__none"
    assert len(result[(name, "per_channel")][0]["ranking"]) == 600
    total = result[(name, "total")][0]
    assert len(total["ranking"]) == 100
    assert total["pool"]["allocations"] == dict(zip(run.CHANNELS, (17, 17, 17, 17, 16, 16), strict=True))
    assert total["pool"]["refill"] is False
    assert result[("document__b__none", "per_channel")][0]["ranking"][0]["score"] == 100.0


@pytest.mark.parametrize("channel", list("BGEN"))
def test_exact_eight_original_cells_agree(channel):
    report, raw = prior(channel)
    saved = raw
    found = agreement(channel, report, raw)
    assert len(found) == 2 and {v["budget_mode"] for v in found} == set(run.MODES)
    assert {v["original_identity"] for v in found} == {c["identity"] for c in report["cells"]}
    assert raw == saved


@pytest.mark.parametrize("mutation", ["duplicate", "wrong_identity", "wrong_query_count", "nonfinite_metric", "pool", "query_order"])
def test_original_report_mismatches_fail(mutation):
    report, raw = prior("G")
    if mutation == "duplicate":
        report["cells"][1] = copy.deepcopy(report["cells"][0])
    elif mutation == "wrong_identity":
        report["cells"][0]["identity"] = "changed"
    elif mutation == "wrong_query_count":
        report["cells"][0]["completed_queries"] = 1
    elif mutation == "nonfinite_metric":
        report["cells"][0]["metrics"]["hit@1"] = float("nan")
    elif mutation == "pool":
        report["cells"][0]["mean_raw_candidates"] += 1
    else:
        report["configuration"]["query_ids"].reverse()
    with pytest.raises(ValueError):
        agreement("G", report, raw)


@pytest.mark.parametrize("mutation", ["missing", "unknown_query", "ranking", "pool", "metric"])
def test_original_per_query_mismatches_fail(mutation):
    report, raw = prior("E")
    db = sqlite3.connect(":memory:")
    db.deserialize(raw)
    cell, qid, value = db.execute("SELECT cell,query_id,payload FROM results LIMIT 1").fetchone()
    if mutation == "missing":
        db.execute("DELETE FROM results WHERE cell=? AND query_id=?", (cell, qid))
    elif mutation == "unknown_query":
        db.execute("UPDATE results SET query_id='unknown' WHERE cell=? AND query_id=?", (cell, qid))
    else:
        payload = json.loads(value)
        if mutation == "ranking":
            payload["ranking"].reverse()
        elif mutation == "pool":
            payload["pool"]["rerank_candidates"] = 100
        else:
            payload["metrics"]["map@16"] = -1
        db.execute("UPDATE results SET payload=? WHERE cell=? AND query_id=?", (json.dumps(payload), cell, qid))
    db.commit()
    changed = db.serialize()
    db.close()
    with pytest.raises(ValueError):
        agreement("E", report, changed)


def test_wal_header_copy_is_memory_only_and_query_only():
    report, raw = prior("N")
    wal = raw[:18] + b"\x02\x02" + raw[20:]
    before = bytes(wal)
    found = agreement("N", report, wal)
    assert len(found) == 2 and wal == before and wal[18:20] == b"\x02\x02"
    db = run.memory_database(wal)
    with pytest.raises(sqlite3.OperationalError):
        db.execute("DELETE FROM results")
    db.close()


@pytest.mark.parametrize("mutation", ["bool_score", "extra_field", "truncated", "nonfinite", "reordered_queries"])
def test_strict_ranking_rejects_incompatible_laya_inputs(mutation):
    ids, _, rankings, _ = examples()
    payload = {"identity": "G", "query_ids": ids, "rankings": rankings["G"]}
    if mutation == "bool_score":
        payload["rankings"][0][0]["score"] = True
    elif mutation == "extra_field":
        payload["rankings"][0][0]["label"] = 1
    elif mutation == "truncated":
        payload["rankings"][0].pop()
    elif mutation == "nonfinite":
        payload["rankings"][0][0]["score"] = float("inf")
    else:
        payload["query_ids"] = list(reversed(ids))
    with pytest.raises(ValueError):
        run.strict_ranking(payload, "G", ["d1", "d2", "d3"], ids, [set(), set()], "G")


def synthetic_validated_inputs(root):
    ids, qrels, rankings, identities = examples()
    refs, reports, raw, agreements = {}, {}, {}, []
    for channel in "BGEN":
        report, sqlite_bytes = prior(channel)
        payload = {"identity": identities[channel], "query_ids": ids, "rankings": rankings[channel]}
        reports[channel] = report
        raw[channel] = {"report": run.frozen.canonical(report), "ranking": run.frozen.canonical(payload), "sqlite": sqlite_bytes}
        refs[channel] = {"identity": identities[channel], "report": {"member": channel + "/report"},
                         "ranking": {"member": channel + "/ranking"}, "sqlite": {"member": channel + "/sqlite"}}
        agreements.extend(agreement(channel, report, sqlite_bytes))
    for channel in "SJ":
        reports[channel] = {"dataset": {"identity": "synthetic-dataset"},
                            "channels": {channel: {"status": "completed", "identity": identities[channel]}}}
        payload = {"identity": identities[channel], "query_ids": ids, "rankings": rankings[channel]}
        refs[channel] = {"identity": identities[channel], "closed": {"synthetic": True}}
        for kind, value in (("report", reports[channel]), ("ranking", payload)):
            path = root / (channel + "-input-" + kind + ".json")
            path.write_bytes(run.frozen.canonical(value))
            refs[channel][kind] = run.frozen.record(root, path.name)
    evidence = {"dataset_identity": "synthetic-dataset", "ordered_id_sha256": ["corpus", "queries"],
                "channels": refs, "channel_identities": identities, "hardware": run.origin.HARDWARE,
                "prior_agreements": agreements, "mps_manifest": {"synthetic": True}, "archive": {"synthetic": True}}
    return evidence, rankings, reports, raw, ids, qrels, PURE


def test_synthetic_output_workflow_and_laya_baseline_schema(tmp_path, monkeypatch):
    found = synthetic_validated_inputs(tmp_path)
    monkeypatch.setattr(run, "validate", lambda *a: found)
    (tmp_path / run.BASE).mkdir(parents=True)
    manifest = {"runtime": {"synthetic": True}, "admission": {"accepted_evidence": [{"synthetic": True}]}}
    before = run.frozen.canonical(manifest)
    old_files = {p: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    result = run.execute(tmp_path, manifest, run.frozen.digest(manifest))
    assert result["completed_cells"] == 126 and result["prior_cell_agreements"] == 8 and result["new_model_calls"] == 0
    output = tmp_path / run.OUTPUT
    report = json.loads((output / "report.json").read_text())
    assert len(report["cells"]) == 126 and report["laya_cells_completed"] == 0
    assert report["configuration"]["requested_channels"] == list("BGENSJ")
    assert report["configuration"]["candidate_grid"] == [20, 50, 100]
    assert report["retrieval_inference_seconds"] is None and report["gpu_calls"] == 0
    assert all(a["original_identity"] != a["derived_identity"] for a in report["original_standalone_evidence"])
    handoff = json.loads((output / "six-ranking-inputs.json").read_text())
    assert set(handoff["rankings"]) == set("BGENSJ")
    assert handoff["retrieval_provenance"]["B"]["hardware"] == "CPU"
    assert handoff["retrieval_provenance"]["B"]["dtype"] == "not_applicable_bm25"
    assert all(set(v) == {"report", "channel_identity", "hardware", "dtype", "stage"}
               for v in handoff["retrieval_provenance"].values())
    with sqlite3.connect(output / "results.sqlite3") as db:
        assert db.execute("SELECT COUNT(*) FROM results").fetchone()[0] == 252
    assert all(p.read_bytes() == data for p, data in old_files.items())
    assert run.frozen.canonical(manifest) == before
    snapshot = {p.name: p.read_bytes() for p in output.iterdir()}
    with pytest.raises(ValueError, match="Out-of-band"):
        run.execute(tmp_path, manifest, run.frozen.digest(manifest))
    again = run.execute(tmp_path, manifest, run.frozen.digest(manifest), closed_sha=result["closed_sha256"])
    assert again["status"] == "verified_closed_unchanged"
    assert {p.name: p.read_bytes() for p in output.iterdir()} == snapshot
    (output / "G-ranking.json").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Sealed"):
        run.execute(tmp_path, manifest, run.frozen.digest(manifest), closed_sha=result["closed_sha256"])


def test_empty_closed_file_manifest_cannot_replay(tmp_path):
    output = tmp_path / run.OUTPUT
    output.mkdir(parents=True)
    snapshot = b"{}"
    (output / "manifest.json").write_bytes(snapshot)
    closed = {"status": "closed", "manifest_sha256": "manifest", "completed_cells": 126,
              "prior_cell_agreements": 8, "files": {}}
    raw = run.frozen.canonical(closed)
    (output / "closed.json").write_bytes(raw)
    with pytest.raises(ValueError, match="scope/files"):
        run.verify_closed(tmp_path, output, snapshot, "manifest", run.origin.sha(raw))


def test_own_alternate_globals_fail_closed(monkeypatch):
    function = run.derive_rows
    alternate = types.FunctionType(function.__code__, dict(function.__globals__), function.__name__,
                                   function.__defaults__, function.__closure__)
    alternate.__kwdefaults__ = function.__kwdefaults__
    monkeypatch.setattr(run, "derive_rows", alternate)
    with pytest.raises(ValueError, match="globals"):
        run.origin.verify_module(run, Path(run.__file__).resolve())


def test_cpu_source_guard_imports_no_model_framework():
    code = ("from pathlib import Path; import sys; from rag_benchmark import industrial_none_derivation_v1 as r; "
            "p=Path.cwd(); r.sources(p,{x:r.frozen.record(p,x)['sha256'] for x in r.SOURCES}); "
            "assert not any(x in sys.modules for x in ('torch','transformers','sentence_transformers'))")
    result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

import json
from pathlib import Path
import sqlite3

import numpy as np
import pytest

from rag_benchmark.models import stable_hash
from rag_benchmark.multimodal import Dataset, load_dataset
from rag_benchmark.multimodal_analysis import GroupingUnavailable, analyze_run, default_comparisons, source_groups


def grouping_dataset(track, corpus, queries, qrels):
    return Dataset(Path("."), {"track": track, "revision": "test"}, corpus, queries, qrels, "fixture")


def test_photo_queries_cluster_by_shared_positive_media_and_multiple_positives_connect():
    data = grouping_dataset("photo", [{"id": did, "media": {"image": f"{did}.png"}} for did in "abc"],
        [{"id": qid} for qid in ("q1", "q2", "q3", "q4")],
        {"q1": {"a": 1}, "q2": {"a": 2, "b": 1}, "q3": {"b": 1}, "q4": {"c": 1}})
    groups, info = source_groups(data)
    assert groups["q1"] == groups["q2"] == groups["q3"] != groups["q4"]
    assert info["group_count"] == 2 and info["largest_group_queries"] == 3
    assert info["fallback_to_independent_queries"] is False


def test_document_groups_use_source_documents_not_unique_query_group_ids():
    corpus = [{"id": "p1", "metadata": {"doc_id": "book-a"}},
              {"id": "p2", "metadata": {"doc_id": "book-a"}},
              {"id": "p3", "metadata": {"doc_id": "book-b"}},
              {"id": "p4", "metadata": {"doc_id": "book-c"}}]
    queries = [{"id": qid, "metadata": {"group_id": qid}} for qid in ("q1", "q2", "q3", "q4")]
    data = grouping_dataset("document", corpus, queries,
        {"q1": {"p1": 1}, "q2": {"p2": 1, "p3": 1}, "q3": {"p3": 1}, "q4": {"p4": 1}})
    groups, info = source_groups(data)
    assert groups["q1"] == groups["q2"] == groups["q3"] != groups["q4"]
    assert info["strategy"] == "positive_document_components"
    del corpus[0]["metadata"]["doc_id"]
    with pytest.raises(GroupingUnavailable, match="doc_id"):
        source_groups(data)


def test_fleurs_unique_transcript_query_with_multiple_recordings_forms_one_source_group():
    corpus = [{"id": "r1", "metadata": {"group_id": "g1"}}, {"id": "r2", "metadata": {"group_id": "g1"}},
              {"id": "r3", "metadata": {"group_id": "g2"}}]
    queries = [{"id": "q1", "metadata": {"group_id": "g1"}}, {"id": "q2", "metadata": {"group_id": "g2"}}]
    data = grouping_dataset("speech", corpus, queries, {"q1": {"r1": 1, "r2": 1}, "q2": {"r3": 1}})
    groups, info = source_groups(data)
    assert len(set(groups.values())) == 2 and info["source_unit_count"] == 5
    corpus[0]["metadata"]["group_id"] = "wrong"
    with pytest.raises(GroupingUnavailable, match="disagree"):
        source_groups(data)


def test_cirr_reused_reference_connects_distinct_targets():
    corpus = [{"id": did, "media": {"image": f"{did}.png"}} for did in ("ref", "a", "b", "other", "c")]
    queries = [{"id": "q1", "metadata": {"reference_id": "ref"}},
               {"id": "q2", "metadata": {"reference_id": "ref"}},
               {"id": "q3", "metadata": {"reference_id": "other"}}]
    data = grouping_dataset("composed_image", corpus, queries, {"q1": {"a": 1}, "q2": {"b": 1}, "q3": {"c": 1}})
    groups, _ = source_groups(data)
    assert groups["q1"] == groups["q2"] != groups["q3"]


def frozen_run(tmp_path, *, track="code", one_group=False, missing_documents=False):
    source, run = tmp_path / "data", tmp_path / "run"
    source.mkdir()
    run.mkdir()
    corpus = [{"id": did, "text": "PRIVATE SOURCE", "metadata": {"doc_id": "shared" if one_group else did}}
              for did in ("a", "b", "c")]
    if missing_documents:
        for row in corpus:
            row["metadata"] = {}
    queries = [{"id": qid, "text": "PRIVATE QUERY", "metadata": {"answer": "PRIVATE GOLD ANSWER"}}
               for qid in ("q1", "q2", "q3", "q4")]
    qrels = [{"query_id": qid, "corpus_id": did, "relevance": 1}
             for qid, did in zip(("q1", "q2", "q3", "q4"), ("a", "a", "a" if one_group else "b", "a" if one_group else "c"))]
    (source / "dataset.json").write_text(json.dumps({"id": "fixture", "track": track, "revision": "frozen-r1"}))
    for name, rows in (("corpus", corpus), ("queries", queries), ("qrels", qrels)):
        (source / f"{name}.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    dataset = load_dataset(source)
    left = [{"hit@5": value, "ndcg@10": value * 0.8} for value in (0, 0, 1, 0)]
    right = [{"hit@5": value, "ndcg@10": value * 0.8} for value in (1, 1, 0, 1)]
    cells = []
    database = sqlite3.connect(run / "progress.sqlite3")
    database.execute("CREATE TABLE results (cell TEXT, query_id TEXT, payload TEXT, PRIMARY KEY(cell,query_id))")
    for channel, metrics in (("b", left), ("g", right)):
        variant = f"{track}__{channel}__none"
        identity = stable_hash(variant)
        cells.append({"variant_id": variant, "identity": identity, "status": "completed", "primary": True,
                      "budget_mode": "per_channel", "candidate_k": None, "query_count": 4, "completed_queries": 4,
                      "metrics": {key: float(np.mean([row[key] for row in metrics])) for key in metrics[0]}})
        for query, values in zip(queries, metrics):
            database.execute("INSERT INTO results VALUES (?,?,?)", (identity, query["id"], json.dumps({
                "query_id": query["id"], "metrics": values, "ignored_source_text": "PRIVATE RAW DB INPUT"})))
    database.commit()
    database.close()
    report = {"dataset": dataset.public_summary(), "configuration": {"query_ids": [q["id"] for q in queries]},
              "evaluated_query_count": 4, "scope": "frozen_collection", "cells": cells}
    (run / "report.json").write_text(json.dumps(report))
    return source, run, report


def test_measured_paired_comparisons_are_reproducible_and_public_exports_text_free(tmp_path):
    source, run, report = frozen_run(tmp_path)
    output = tmp_path / "public"
    first = analyze_run(source, run, output, samples=5000, seed=42)
    again = analyze_run(source, run, samples=5000, seed=42)
    assert first == again
    assert first["status"] == "completed"
    comparison = first["comparisons"][0]
    assert comparison["left"]["variant_id"] == "code__b__none"
    measured = comparison["metrics"]["hit@5"]
    assert measured["left_mean"] == 0.25 and measured["right_mean"] == 0.75
    assert measured["mean_difference"] == 0.5
    assert (measured["wins"], measured["losses"], measured["ties"]) == (3, 1, 0)
    assert measured["group_count"] == 3 and measured["query_count"] == 4
    assert -1 <= measured["ci95_low"] <= 0.5 <= measured["ci95_high"] <= 1
    assert "PRIVATE" not in (output / "paired-comparisons.json").read_text()
    assert "PRIVATE" not in (output / "paired-comparisons.md").read_text()
    assert json.loads((run / "report.json").read_text()) == report
    assert first["model_inference_performed"] is False


def test_generic_reversed_selectors_reverse_delta_and_interval(tmp_path):
    source, run, report = frozen_run(tmp_path)
    forward = analyze_run(source, run, samples=1000)["comparisons"][0]["metrics"]["ndcg@10"]
    reverse = analyze_run(source, run, comparisons=[{"left": report["cells"][1]["identity"],
                                                  "right": report["cells"][0]["identity"]}], samples=1000)
    measured = reverse["comparisons"][0]["metrics"]["ndcg@10"]
    assert measured["mean_difference"] == -forward["mean_difference"]
    assert measured["ci95_low"] == pytest.approx(-forward["ci95_high"])
    assert measured["ci95_high"] == pytest.approx(-forward["ci95_low"])


def test_partial_or_mismatched_queries_never_compare_intersection(tmp_path):
    source, run, report = frozen_run(tmp_path)
    with sqlite3.connect(run / "progress.sqlite3") as database:
        database.execute("DELETE FROM results WHERE cell=? AND query_id='q4'", (report["cells"][1]["identity"],))
    result = analyze_run(source, run, samples=50)
    comparison = result["comparisons"][0]
    assert comparison["status"] == "unavailable"
    assert "exact frozen query set" in comparison["reason"]
    assert "metrics" not in comparison


def test_report_and_per_query_aggregate_must_agree(tmp_path):
    source, run, report = frozen_run(tmp_path)
    report["cells"][0]["metrics"]["hit@5"] = 0.99
    (run / "report.json").write_text(json.dumps(report))
    comparison = analyze_run(source, run, samples=50)["comparisons"][0]
    assert comparison["status"] == "unavailable" and "aggregate" in comparison["reason"]


def test_missing_groups_reports_only_descriptive_differences(tmp_path):
    source, run, _ = frozen_run(tmp_path, track="document", missing_documents=True)
    result = analyze_run(source, run, samples=50)
    assert result["grouping"]["status"] == "unavailable"
    assert result["grouping"]["fallback_to_independent_queries"] is False
    comparison = result["comparisons"][0]
    assert comparison["status"] == "descriptive_only"
    assert comparison["metrics"]["hit@5"]["mean_difference"] == 0.5
    assert "ci95_low" not in comparison["metrics"]["hit@5"]


def test_single_connected_source_group_does_not_create_false_zero_width_interval(tmp_path):
    source, run, _ = frozen_run(tmp_path, track="document", one_group=True)
    result = analyze_run(source, run, samples=50)
    assert result["grouping"]["group_count"] == 1
    comparison = result["comparisons"][0]
    assert comparison["status"] == "descriptive_only"
    assert "ci95_low" not in comparison["metrics"]["hit@5"]


def test_cross_dataset_fingerprint_and_smoke_scope_are_rejected(tmp_path):
    source, run, report = frozen_run(tmp_path)
    report["dataset"]["identity"] = "different"
    (run / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="fingerprint"):
        analyze_run(source, run)
    report["dataset"]["identity"] = load_dataset(source).identity
    report["scope"] = "smoke"
    (run / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="smoke"):
        analyze_run(source, run)


def test_uncompleted_or_nonprimary_requested_cell_has_no_comparison_score(tmp_path):
    source, run, report = frozen_run(tmp_path)
    report["cells"][1]["primary"] = False
    (run / "report.json").write_text(json.dumps(report))
    assert default_comparisons(report) == []
    assert analyze_run(source, run)["status"] == "no_comparable_cells"
    explicit = analyze_run(source, run, comparisons=[{"left": "code__b__none", "right": "code__g__none"}])
    assert explicit["comparisons"][0]["status"] == "unavailable"
    assert "metrics" not in explicit["comparisons"][0]

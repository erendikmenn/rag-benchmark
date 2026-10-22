import copy
import io
import itertools
import json
from pathlib import Path
import socket
import sqlite3
import stat
import subprocess

import numpy as np
import pytest

from rag_benchmark import multimodal as old
from rag_benchmark import vidore_cache_recovery_v1 as recovery


REPO = Path(recovery.__file__).resolve().parents[2]
PURE = {name: getattr(old, name) for name in
        ("exact_dot_rankings", "fuse_rankings", "graded_metrics", "aggregate_metrics")}


def save(root, relative, value):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value if isinstance(value, bytes) else recovery.canonical(value) + b"\n")
    return path


def prior_cell(config, identities, subset, mode, rankings, qrels, connection):
    variant = "document__" + "_".join(x.lower() for x in subset) + "__none"
    identity = recovery.digest({**config, "variant": variant, "mode": mode, "candidate_k": None,
                                "channel_identities": {key: identities[key] for key in subset},
                                "reranker": None, "reranker_prompts": None})
    metrics, pools = [], []
    for index, qid in enumerate(config["query_ids"]):
        ranking, pool = old.fuse_rankings({key: rankings[key][index] for key in subset}, subset,
                                          budget_mode=mode, budget=100, rrf_k=60)
        values = old.graded_metrics([row["id"] for row in ranking], qrels[qid])
        result = {"query_id": qid, "ranking": ranking, "metrics": values,
                  "pool": {**pool, "rerank_candidates": 0}, "usage": {}}
        connection.execute("INSERT INTO results VALUES (?,?,?)", (identity, qid, json.dumps(result)))
        metrics.append(values)
        pools.append(pool)
    return {"variant_id": variant, "identity": identity, "budget_mode": mode, "candidate_k": None,
            "status": "completed", "completed_queries": len(config["query_ids"]),
            "metrics": old.aggregate_metrics(metrics),
            "mean_raw_candidates": float(np.mean([x["raw_candidates"] for x in pools])),
            "mean_unique_candidates": float(np.mean([x["unique_candidates"] for x in pools])),
            "mean_rerank_candidates": 0.0}


def fixture(tmp_path, monkeypatch, channel="E"):
    root = tmp_path / "root"
    root.mkdir()
    for relative in recovery.SOURCE_FILES:
        save(root, relative, (REPO / relative).read_bytes())
    pins = {relative: recovery.record(root, relative)["sha256"] for relative in recovery.SOURCE_FILES}
    # Fixture data lives in a separate temporary root. The production path check
    # is covered separately; integration uses the exact four imported pure seams.
    monkeypatch.setattr(recovery, "_pure", lambda root, pins: PURE)
    name = "vidore-v3-finance_fr-fr" if channel == "E" else "vidore-v3-hr-en"
    base = "data/multimodal/" + name
    corpus = [{"id": f"d{index:03}", "text": "synthetic document"} for index in range(130)]
    queries = [{"id": "q0", "text": "synthetic request"}, {"id": "q1", "text": "synthetic request"}]
    labels = [{"query_id": "q0", "corpus_id": "d000", "relevance": 3},
              {"query_id": "q1", "corpus_id": "d129", "relevance": 1}]
    metadata = {"id": name, "revision": "synthetic-fixed", "track": "document", "full_gallery": True,
                "counts": {"corpus": 130, "queries": 2},
                "text_provenance": {"source": "published_markdown", "query_independent": True,
                                    "gold_fields_used": False}}
    save(root, base + "/dataset.json", metadata)
    for key, rows in [("corpus", corpus), ("queries", queries), ("qrels", labels)]:
        save(root, base + "/" + key + ".jsonl", b"".join(recovery.canonical(row) + b"\n" for row in rows))
    dataset_identity = recovery.digest({"files": {key: recovery.record(root, base + "/" + key)["sha256"]
                                                 for key in ["dataset.json", "corpus.jsonl", "queries.jsonl", "qrels.jsonl"]},
                                         "media": {}})
    config = {"engine": "multimodal-v1", "dataset": dataset_identity, "candidate_grid": [20, 50, 100],
              "budget_modes": ["per_channel", "total"], "channel_budget": 100, "rrf_k": 60,
              "requested_channels": list(recovery.CHANNELS), "query_ids": ["q0", "q1"]}
    report = {"engine": "multimodal-v1", "dataset": {"id": name, "identity": dataset_identity,
              "revision": metadata["revision"]}, "configuration": config, "cells": [],
              "adapter_specs": {"channels": {key: {"identity": recovery.digest(key), "dimension": 768}
                                              for key in recovery.CHANNELS}, "rerankers": {}}}
    identities = {key: recovery.digest({"channel": key}) for key in recovery.CHANNELS}
    identities[channel] = recovery._missing_identity(report, channel)
    report["channels"] = {key: {"status": "completed", "identity": identities[key]}
                           for key in recovery.CHANNELS if key != channel}
    if channel == "E":
        report["channels"][channel] = {"status": "failed"}
    document_vectors = np.zeros((130, 768), dtype=np.float32)
    for index in range(130):
        document_vectors[index, index] = 1
    query_vectors = document_vectors[[0, 129]]
    for role, values in [("document", document_vectors), ("query", query_vectors)]:
        directory, _ = recovery._vector_paths(report, channel, role, len(values))
        diagnostics = []
        for start in range(0, len(values), 128):
            stop = min(start + 128, len(values))
            buffer = io.BytesIO()
            np.save(buffer, values[start:stop], allow_pickle=False)
            save(root, directory + f"/{start:09d}-{stop:09d}.npy", buffer.getvalue())
            usage = {"rows": stop - start, "reported": False, "items": []}
            diagnostics.append(usage)
            save(root, directory + f"/{start:09d}-{stop:09d}.usage.json", usage)
        save(root, directory + "/diagnostics.json", old._merge_diagnostics(diagnostics, len(values)))
    rankings = {}
    for offset, key in enumerate(recovery.CHANNELS):
        if key == channel:
            continue
        order = corpus[offset * 7:] + corpus[:offset * 7]
        rankings[key] = [[{"id": row["id"], "score": float(100 - index), "rank": index + 1}
                          for index, row in enumerate(order[:100])] for _ in queries]
        save(root, recovery._ranking_path(identities[key]),
             {"identity": identities[key], "query_ids": ["q0", "q1"], "rankings": rankings[key], "usage": {}})
    qrels = {"q0": {"d000": 3}, "q1": {"d129": 1}}
    for suffix in ("all-retrieval", "bm25"):
        run = "runs/multimodal/" + name + "-" + suffix
        path = root / run
        path.mkdir(parents=True)
        connection = sqlite3.connect(path / "progress.sqlite3")
        connection.execute("CREATE TABLE results (cell TEXT, query_id TEXT, payload TEXT, PRIMARY KEY(cell,query_id))")
        current = copy.deepcopy(report)
        subsets = []
        if suffix == "bm25":
            current["configuration"]["requested_channels"] = ["B"]
            current["channels"] = {"B": report["channels"]["B"]}
            subsets = [("B",)]
        elif channel == "E":
            keys = [key for key in recovery.CHANNELS if key != channel]
            subsets = [subset for width in range(1, 6) for subset in itertools.combinations(keys, width)]
        current["cells"] = [prior_cell(current["configuration"], identities, subset, mode, rankings, qrels, connection)
                             for subset in subsets for mode in ("per_channel", "total")]
        connection.commit()
        connection.close()
        save(root, run + "/report.json", current)
    recipe = {"version": recovery.VERSION, "dataset_id": name, "missing_channel": channel,
              "source_sha256": pins, "channel_identities": identities, "additional_metadata": [],
              "original_adapter_specs_sha256": recovery.digest(report["adapter_specs"]), "expected_records": []}
    manifest = recovery.prepare_manifest(root, recipe)
    (root / "work/vidore-cache-recovery-v1/delta").mkdir(parents=True, mode=0o700)
    return root, manifest


@pytest.mark.parametrize("channel,agreements", [("E", 64), ("J", 2)])
def test_complete_recovery_preserves_inputs_and_closed_resume(tmp_path, monkeypatch, channel, agreements):
    root, manifest = fixture(tmp_path, monkeypatch, channel)
    original = {str(path.relative_to(root)): recovery.record(root, str(path.relative_to(root)))["sha256"]
                for base in [root / "data", root / ".cache", root / "runs"] for path in base.rglob("*") if path.is_file()}
    def forbidden(*args, **kwargs):
        raise AssertionError("Forbidden model/network/process/orchestration path called")
    for name in ("run_matrix", "cached_encode", "base_rankings", "TextEmbeddingAdapter", "BGETextReranker"):
        monkeypatch.setattr(old, name, forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    result = recovery.execute(root, manifest, recovery.digest(manifest))
    assert result["completed_cells"] == 126
    assert result["prior_cell_agreements"] == agreements
    output = root / manifest["output"]
    report = json.loads((output / "report.json").read_text())
    assert len({cell["variant_id"] for cell in report["cells"]}) == 63
    assert {cell["budget_mode"] for cell in report["cells"]} == {"per_channel", "total"}
    assert report["new_model_calls"] == report["new_embedding_rows"] == report["gpu_calls"] == 0
    assert report["embedding_inference_seconds"] is None
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in output.iterdir())
    assert all(recovery.record(root, path)["sha256"] == value for path, value in original.items())
    closed_bytes = {path.name: path.read_bytes() for path in output.iterdir()}
    with pytest.raises(FileExistsError):
        recovery.execute(root, manifest, recovery.digest(manifest))
    assert recovery.execute(root, manifest, recovery.digest(manifest), resume_closed=True,
                            closed_sha256=result["closed_sha256"])["status"] == "verified_closed_unchanged"
    assert {path.name: path.read_bytes() for path in output.iterdir()} == closed_bytes


@pytest.mark.parametrize("mutation", ["missing", "extra", "truncate", "reorder", "source", "same_bytes_wrong_path",
                                      "identity", "query_order", "diagnostic", "nan", "wrong_shape",
                                      "trailing", "object", "runtime", "dense_truncation", "aggregate"])
def test_changed_inputs_fail_before_output(tmp_path, monkeypatch, mutation):
    root, manifest = fixture(tmp_path, monkeypatch)
    entry = next(value for key, value in manifest["vectors"]["document"].items() if key.endswith(".npy"))
    path = root / entry["path"]
    if mutation == "missing":
        path.unlink()
    elif mutation == "extra":
        path.with_name("extra.npy").write_bytes(path.read_bytes())
    elif mutation == "truncate":
        path.write_bytes(path.read_bytes()[:-4])
    elif mutation == "reorder":
        rows = np.load(path, allow_pickle=False)
        np.save(path, rows[::-1], allow_pickle=False)
    elif mutation == "source":
        with (root / recovery.SOURCE_FILES[0]).open("a") as stream:
            stream.write("\n# changed\n")
    elif mutation == "same_bytes_wrong_path":
        alias = root / "alias.npy"
        alias.write_bytes(path.read_bytes())
        entry["path"] = "alias.npy"
    elif mutation == "identity":
        manifest["channel_identities"]["B"] = "a" * 64
    elif mutation == "query_order":
        manifest["ordered_id_sha256"][1] = recovery.digest(["q1", "q0"])
    elif mutation == "diagnostic":
        diagnostic = path.with_suffix(".usage.json")
        diagnostic.write_text('{"rows": 1}')
        manifest["vectors"]["document"][str(diagnostic.relative_to(root))] = recovery.record(root, str(diagnostic.relative_to(root)))
    elif mutation in {"nan", "wrong_shape"}:
        rows = np.load(path, allow_pickle=False)
        if mutation == "nan":
            rows[0, 0] = np.nan
        else:
            rows = rows[:, :767]
        np.save(path, rows, allow_pickle=False)
        manifest["vectors"]["document"][entry["path"]] = recovery.record(root, entry["path"])
    elif mutation in {"trailing", "object"}:
        if mutation == "trailing":
            path.write_bytes(path.read_bytes() + b"extra")
        else:
            np.save(path, np.asarray([{"unsafe": "object"}], dtype=object), allow_pickle=True)
        manifest["vectors"]["document"][entry["path"]] = recovery.record(root, entry["path"])
    elif mutation == "runtime":
        manifest["cpu_runtime"]["numpy"] = "changed"
    elif mutation == "dense_truncation":
        ranking_entry = manifest["rankings"]["G"]
        payload = recovery.read_json(root, ranking_entry)
        payload["rankings"][0].pop()
        save(root, ranking_entry["path"], payload)
        manifest["rankings"]["G"] = recovery.record(root, ranking_entry["path"])
    elif mutation == "aggregate":
        key = next(key for key in manifest["vectors"]["document"] if key.endswith("/diagnostics.json"))
        save(root, key, {"total_rows": 1})
        manifest["vectors"]["document"][key] = recovery.record(root, key)
    with pytest.raises((ValueError, FileNotFoundError)):
        recovery.execute(root, manifest, recovery.digest(manifest))
    assert not (root / manifest["output"]).exists()


def test_prior_metric_disagreement_cannot_close_delta(tmp_path, monkeypatch):
    root, manifest = fixture(tmp_path, monkeypatch)
    entry = manifest["prior"][1]["report"]
    data = recovery.read_json(root, entry)
    data["cells"][0]["metrics"]["hit@5"] += .1
    save(root, entry["path"], data)
    manifest["prior"][1]["report"] = recovery.record(root, entry["path"])
    with pytest.raises(ValueError, match="metric agreement"):
        recovery.execute(root, manifest, recovery.digest(manifest))
    assert not (root / manifest["output"] / "closed.json").exists()
    with pytest.raises(ValueError):
        recovery.execute(root, manifest, recovery.digest(manifest), resume_closed=True)


def test_source_imports_are_exact_and_cpu_only():
    pins = {relative: recovery.record(REPO, relative)["sha256"] for relative in recovery.SOURCE_FILES}
    assert recovery._pure(REPO, pins) == PURE


def test_changed_loaded_function_is_rejected(monkeypatch):
    pins = {relative: recovery.record(REPO, relative)["sha256"] for relative in recovery.SOURCE_FILES}
    monkeypatch.setattr(old, "graded_metrics", lambda *args, **kwargs: {})
    with pytest.raises(ValueError, match="Loaded code"):
        recovery._pure(REPO, pins)


def test_duplicate_baseline_cannot_mask_missing_condition(tmp_path, monkeypatch):
    root, manifest = fixture(tmp_path, monkeypatch)
    entry = manifest["main_report"]
    report = recovery.read_json(root, entry)
    report["cells"][-1] = copy.deepcopy(report["cells"][0])
    save(root, entry["path"], report)
    updated = recovery.record(root, entry["path"])
    manifest["main_report"] = manifest["prior"][0]["report"] = updated
    with pytest.raises(ValueError, match="coverage"):
        recovery.execute(root, manifest, recovery.digest(manifest))
    assert not (root / manifest["output"]).exists()


@pytest.mark.parametrize("field", ["metric", "pool"])
def test_nan_baseline_cannot_pass_agreement(tmp_path, monkeypatch, field):
    root, manifest = fixture(tmp_path, monkeypatch)
    entry = manifest["prior"][1]["report"]
    report = recovery.read_json(root, entry)
    if field == "metric":
        report["cells"][0]["metrics"]["hit@5"] = float("nan")
    else:
        report["cells"][0]["mean_raw_candidates"] = float("nan")
    (root / entry["path"]).write_text(json.dumps(report))
    manifest["prior"][1]["report"] = recovery.record(root, entry["path"])
    with pytest.raises(ValueError):
        recovery.execute(root, manifest, recovery.digest(manifest))
    assert not (root / manifest["output"] / "closed.json").exists()


def test_wrong_diagnostic_row_order_is_rejected(tmp_path, monkeypatch):
    root, manifest = fixture(tmp_path, monkeypatch, "J")
    key = next(key for key in manifest["vectors"]["document"] if key.endswith(".usage.json"))
    usage = {"rows": 128, "reported": True, "items": [{"id": f"d{index:03}"} for index in reversed(range(128))]}
    save(root, key, usage)
    manifest["vectors"]["document"][key] = recovery.record(root, key)
    with pytest.raises(ValueError, match="item order"):
        recovery.execute(root, manifest, recovery.digest(manifest))


def test_empty_closed_marker_cannot_claim_completion(tmp_path, monkeypatch):
    root, manifest = fixture(tmp_path, monkeypatch)
    output = root / manifest["output"]
    output.mkdir()
    closed = {"status": "closed", "manifest_sha256": recovery.digest(manifest), "completed_cells": 126, "files": {}}
    (output / "closed.json").write_bytes(recovery.canonical(closed))
    with pytest.raises(ValueError, match="mandatory outputs"):
        recovery.execute(root, manifest, recovery.digest(manifest), resume_closed=True,
                         closed_sha256=recovery.digest(closed))


def test_caller_mutation_cannot_change_admitted_manifest(tmp_path, monkeypatch):
    root, manifest = fixture(tmp_path, monkeypatch)
    seal = recovery.digest(manifest)
    output = root / manifest["output"]
    def mutate_caller(*args, **kwargs):
        manifest["dataset_id"] = "changed-after-admission"
        return old.exact_dot_rankings(*args, **kwargs)
    monkeypatch.setattr(recovery, "_pure", lambda root, pins: {**PURE, "exact_dot_rankings": mutate_caller})
    result = recovery.execute(root, manifest, seal)
    assert result["completed_cells"] == 126
    assert recovery.digest(json.loads((output / "manifest.json").read_text())) == seal


def test_cli_manifest_preparation_does_not_execute_recovery(tmp_path, monkeypatch, capsys):
    root, manifest = fixture(tmp_path, monkeypatch)
    recipe = {"version": recovery.VERSION, "dataset_id": manifest["dataset_id"],
              "missing_channel": manifest["missing_channel"], "source_sha256": manifest["source_sha256"],
              "channel_identities": manifest["channel_identities"], "additional_metadata": [],
              "expected_records": [], "original_adapter_specs_sha256": recovery.digest(manifest["original_adapter_specs"])}
    recipe_path = save(root, "work/vidore-cache-recovery-v1/recipe.json", recipe)
    output = root / "work/vidore-cache-recovery-v1/manifests" / (manifest["dataset_id"] + ".json")
    output.parent.mkdir(mode=0o700)
    def forbidden(*args, **kwargs):
        raise AssertionError("Preparation attempted recovery")
    monkeypatch.setattr(recovery, "execute", forbidden)
    argv = ["--root", str(root), "--manifest", str(output), "--prepare-from-recipe", str(recipe_path),
            "--recipe-sha256", recovery.digest(recipe)]
    assert recovery.main(argv) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "prepared_not_executed"
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert not (root / manifest["output"]).exists()
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        recovery.main(argv)
    assert output.read_bytes() == before


def test_exclusive_write_fsync_and_no_overwrite(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(recovery.os, "fsync", lambda fd: calls.append(fd))
    target = tmp_path / "private.json"
    recovery.exclusive_write(target, b"original")
    assert len(calls) == 2
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    with pytest.raises(FileExistsError):
        recovery.exclusive_write(target, b"replacement")
    assert target.read_bytes() == b"original"


def test_old_cache_hit_rewrites_diagnostics_and_miss_encodes(tmp_path):
    class Stub:
        identity = "synthetic"
        dimension = 768
        calls = 0

        def encode(self, items, role):
            self.calls += 1
            raise AssertionError("Old seam attempted inference")

    dataset = old.Dataset(tmp_path, {"track": "document", "revision": "fixed"},
                          [{"id": "a", "text": "synthetic"}], [{"id": "q", "text": "synthetic"}], {}, "fixed")
    adapter = Stub()
    key = recovery.digest({"encoding": old.encoding_cache_identity(dataset, adapter, "document"), "text_only": False})
    directory = tmp_path / "vectors" / key
    directory.mkdir(parents=True)
    values = np.zeros((1, 768), dtype=np.float32)
    values[0, 0] = 1
    block = directory / "000000000-000000001.npy"
    np.save(block, values, allow_pickle=False)
    (directory / "diagnostics.json").write_bytes(b"sentinel")
    old.cached_encode(dataset, adapter, "document", tmp_path)
    assert (directory / "diagnostics.json").read_bytes() != b"sentinel"
    assert adapter.calls == 0
    block.unlink()
    with pytest.raises(AssertionError, match="attempted inference"):
        old.cached_encode(dataset, adapter, "document", tmp_path)
    assert adapter.calls == 1


def test_symlink_alias_is_rejected(tmp_path):
    (tmp_path / "a").write_bytes(b"same")
    (tmp_path / "b").symlink_to(tmp_path / "a")
    with pytest.raises(ValueError, match="Symlink"):
        recovery.record(tmp_path, "b")


def test_recovery_import_closure_does_not_contain_model_execution():
    source = (REPO / "src/rag_benchmark/vidore_cache_recovery_v1.py").read_text()
    assert "import torch" not in source
    assert ".encode(" not in source.replace(".encode()", "").replace('.encode("', "")
    assert "run_matrix(" not in source
    assert "cached_encode(" not in source
    assert "subprocess" not in source

import hashlib
import json
import os
from pathlib import Path
import stat
import types

import pytest

from rag_benchmark import vidore_laya_completion_v2 as runner


def rankings():
    result = {}
    for offset, channel in enumerate(runner.CHANNELS):
        ids = [f"d-{i:03d}" for i in range(offset * 5, offset * 5 + 100)]
        result[channel] = [[{"id": identifier, "score": 1 - i / 200, "rank": i + 1}
                            for i, identifier in enumerate(ids)]]
    return result


def test_union_scatter_preserves_every_prefix_budget_and_id_tie():
    graph = runner.graph(rankings(), 0)
    ids = runner.union_ids(graph)
    assert len(graph) == 126
    assert len(ids) == 125
    scores = [0.5 if i % 3 else 0.8 for i in range(len(ids))]
    qrels = {"d-002": 1.0, "d-085": 1.0}
    result = runner.scatter(graph, ids, scores, qrels)
    assert len(result) == 378
    lookup = dict(zip(ids, scores))
    for index, pair in enumerate(graph):
        for k_index, k in enumerate((20, 50, 100)):
            candidate_ids = [x["id"] for x in pair["fused"][:k]]
            expected = sorted(candidate_ids, key=lambda key: (-lookup[key], key))
            row = result[index * 3 + k_index]
            assert row["candidate_count"] == len(expected)
            assert row["metrics"] == runner.matrix.graded_metrics(expected, qrels)
    assert graph[0]["pool"]["allocations"] == {"B": 100}
    multi = next(g for g in graph if len(g["channels"]) == 6 and g["budget_mode"] == "total")
    assert sum(multi["pool"]["allocations"].values()) == 100


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, 1.1, True])
def test_scatter_rejects_invalid_scores(bad):
    graph = runner.graph(rankings(), 0)
    ids = runner.union_ids(graph)
    with pytest.raises(ValueError, match="scores"):
        runner.scatter(graph, ids, [bad] * len(ids), {ids[0]: 1})


def test_scatter_rejects_changed_union_and_does_not_retrieve(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("retrieval or model path was called")
    monkeypatch.setattr(runner.matrix, "base_rankings", forbidden)
    monkeypatch.setattr(runner.matrix, "cached_encode", forbidden)
    monkeypatch.setattr(runner.generation, "MultimodalLayaReranker", forbidden)
    graph = runner.graph(rankings(), 0)
    ids = runner.union_ids(graph)
    runner.scatter(graph, ids, [0.5] * len(ids), {ids[0]: 1})
    with pytest.raises(ValueError, match="union_membership"):
        runner.scatter(graph, ids[::-1], [0.5] * len(ids), {ids[0]: 1})


def test_receipts_resume_without_reissuing_and_reject_mutation(tmp_path):
    jobs = tmp_path / "jobs"
    jobs.mkdir(mode=0o700)
    store = runner.Receipts(jobs, "manifest-seal")
    count = []
    def operation():
        count.append(1)
        return {"scores": [0.2]}
    first, reused = store.run("q-000000", {"input": "a"}, operation, lambda: None)
    assert not reused
    assert store.run("q-000000", {"input": "a"}, operation, lambda: None) == (first, True)
    assert len(count) == 1
    with pytest.raises(ValueError, match="receipt_binding"):
        store.read("q-000000", {"input": "changed"})
    result = jobs / "q-000000/result.json"
    result.write_text('{"claim_sha256":"changed","value":{}}')
    with pytest.raises(ValueError, match="receipt_hash"):
        store.read("q-000000", {"input": "a"})


def test_incomplete_claim_fails_closed_without_retry(tmp_path):
    jobs = tmp_path / "jobs"
    jobs.mkdir(mode=0o700)
    store = runner.Receipts(jobs, "manifest")
    count = []
    def crash():
        count.append(1)
        raise RuntimeError("interruption")
    with pytest.raises(RuntimeError):
        store.run("q-000000", {}, crash, lambda: None)
    with pytest.raises(ValueError, match="incomplete_claim"):
        store.run("q-000000", {}, crash, lambda: None)
    assert len(count) == 1


def test_prior_receipt_guard_failure_never_closes_result(tmp_path):
    jobs = tmp_path / "jobs"
    jobs.mkdir(mode=0o700)
    store = runner.Receipts(jobs, "manifest")
    calls = []
    def guard():
        calls.append(1)
        if len(calls) == 2:
            raise ValueError("input_freshness")
    with pytest.raises(ValueError, match="input_freshness"):
        store.run("q-000000", {}, lambda: {"scores": [0.2]}, guard)
    assert set(runner.private_tree(jobs / "q-000000")) == {"claim.json"}


def test_private_tree_rejects_extra_link(tmp_path):
    tmp_path.chmod(0o700)
    target = tmp_path / "file.json"
    runner.fresh(target, {})
    (tmp_path / "alias").symlink_to(target)
    with pytest.raises(ValueError, match="private_symlink"):
        runner.private_tree(tmp_path)


def test_model_snapshot_link_and_resolved_blob_are_both_bound(tmp_path):
    snapshot = tmp_path / runner.MODEL_SNAPSHOT
    snapshot.mkdir(parents=True)
    blobs = tmp_path / runner.MODEL_REPOSITORY / "blobs"
    blobs.mkdir()
    blob = blobs / "original"
    blob.write_bytes(b"model-fixture")
    (snapshot / "model.safetensors").symlink_to("../../blobs/original")
    record = runner.model_record(tmp_path, "model.safetensors")
    assert record["symlink_target"] == "../../blobs/original"
    assert record["resolved_file"]["sha256"] == hashlib.sha256(b"model-fixture").hexdigest()
    assert runner.snapshot_files(tmp_path) == {"model.safetensors"}
    with pytest.raises(ValueError, match="model_relative_path"):
        runner.model_record(tmp_path, "../model.safetensors")


def test_sources_match_current_loaded_code_without_model_load():
    root = Path(__file__).resolve().parents[1]
    pins = {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in runner.SOURCE_FILES}
    runner.sources(root, pins)


def test_baseline_requires_all_none_conditions():
    cells = [{"variant_id": "document__" + "_".join(p["channels"]).lower() + "__none",
              "budget_mode": p["budget_mode"], "candidate_k": None, "status": "completed",
              "query_count": 3, "completed_queries": 3, "metrics": {"hit@5": 1.0}}
             for p in runner.pairs()]
    assert len(runner.baseline_cells({"cells": cells}, 3)) == 126
    with pytest.raises(ValueError, match="baseline_count"):
        runner.baseline_cells({"cells": cells[:-1]}, 3)
    bad = json.loads(json.dumps(cells))
    bad[0]["completed_queries"] = 2
    with pytest.raises(ValueError, match="baseline_scope"):
        runner.baseline_cells({"cells": bad}, 3)


def test_synthetic_execution_resumes_only_sealed_closed_queries(tmp_path, monkeypatch):
    """Full receipt/namespace closure with a fake scorer; no model import/load."""
    root = tmp_path
    parent = root / "work/vidore-laya-completion-v2/runs"
    parent.mkdir(parents=True, mode=0o700)
    gpu = root / "work/overnight/gpu.lock"
    gpu.parent.mkdir(parents=True)
    gpu.write_bytes(b"")
    gpu.chmod(0o600)
    corpus = [{"id": f"d-{i:03d}", "text": "fixture", "media": {}} for i in range(125)]
    queries = [{"id": f"q{i}", "text": "fixture query", "media": {}} for i in range(2)]
    data = runner.matrix.Dataset(root, {"track": "document"}, corpus, queries,
                                 {q["id"]: {"d-002": 1} for q in queries}, "fixture")
    ranks = rankings()
    ranks = {k: [v[0], v[0]] for k, v in ranks.items()}
    validate_calls = []
    def validate(*args, **kwargs):
        validate_calls.append(1)
        if len(validate_calls) == 2:
            raise ValueError("simulated_post_completion_gate")
        return data, ranks
    monkeypatch.setattr(runner, "validate", validate)
    monkeypatch.setattr(runner, "sources", lambda *args: None)
    monkeypatch.setattr(runner, "sdk_sources", lambda *args: None)
    monkeypatch.setattr(runner, "observations", lambda *args: {})
    scored = []
    class FakeBackend:
        def __init__(self, manifest):
            pass
        def score(self, query, documents):
            scored.append((query["id"], len(documents)))
            return [0.5] * len(documents), 0.1
        def close(self):
            pass
    monkeypatch.setattr(runner, "Backend", FakeBackend)
    manifest = {"dataset_id": "vidore-v3-energy-fr", "source_sha256": {}, "sdk_source_sha256": {},
                "output": "work/vidore-laya-completion-v2/runs/vidore-v3-energy-fr",
                "gpu_lock": {"path": "work/overnight/gpu.lock", "device": gpu.stat().st_dev,
                             "inode": gpu.stat().st_ino},
                "retrieval_provenance": {}, "baseline_report": {}}
    seal = runner.sha(runner.canonical(manifest) + b"\n")
    preflight_dir = root / "work/vidore-laya-completion-v2/preflight"
    preflight_dir.mkdir(mode=0o700)
    evidence = preflight_dir / "vidore-v3-energy-fr.json"
    model_corpus = {r["id"]: data.model_item(r) for r in data.corpus}
    records = [{"binding": runner.query_binding(seal, i, data, runner.graph(ranks, i), model_corpus),
                "pairs": 125, "max_tokens": 64} for i in range(2)]
    runner.fresh(evidence, {"protocol": runner.VERSION, "status": "PASS_TOKENIZER_ONLY_CONTEXT_PREFLIGHT",
        "manifest_sha256": seal, "max_len": 8192, "queries": records, "pairs": 250,
        "maximum_tokens": 64, "truncated_pairs": 0, "model_instances": 0, "model_forwards": 0, "gpu_calls": 0})
    ref = runner.frozen.record(root, evidence.relative_to(root).as_posix())
    with pytest.raises(ValueError, match="simulated_post_completion_gate"):
        runner.execute(root, manifest, seal, preflight_ref=ref)
    output = root / manifest["output"]
    assert len(scored) == 2
    resume = {"protocol": runner.VERSION, "manifest_sha256": seal, "files": runner.private_tree(output)}
    result = runner.execute(root, manifest, seal, resume_seal=resume, preflight_ref=ref)
    assert result["laya_cells"] == 378
    assert len(scored) == 2
    summary = json.loads((output / "summary.json").read_text())
    assert summary["closed_queries_reused_this_segment"] == 2
    assert summary["new_unique_pairs_this_segment"] == 0
    assert summary["retrieval_inference_seconds"] is None
    assert summary["sdk_language_argument"] == "tr"
    assert len(summary["cells"]) == 378
    assert all(c["per_cell_latency_measured"] is False for c in summary["cells"])
    with pytest.raises(ValueError, match="resume_hashes"):
        runner.execute(root, manifest, seal, resume_seal=resume, preflight_ref=ref)


@pytest.fixture
def sealed_fixture(tmp_path, monkeypatch):
    """Small real sealed tables/rankings/model files, never a model invocation."""
    name = "vidore-v3-energy-fr"
    monkeypatch.setitem(runner.TARGETS, name, (125, 2))
    monkeypatch.setattr(runner, "sources", lambda *args: None)
    sdk = tmp_path / "fixture_sdk.py"
    sdk.write_text("# synthetic SDK source seal only\n")
    monkeypatch.setattr(runner, "sdk_paths", lambda root: ["fixture_sdk.py"])
    def forbidden(*args, **kwargs):
        raise AssertionError("model construction is forbidden")
    monkeypatch.setattr(runner.generation, "MultimodalLayaReranker", forbidden)
    directory = tmp_path / "data/multimodal" / name
    directory.mkdir(parents=True)
    corpus = [{"id": f"d-{i:03d}", "text": "source", "media": {}} for i in range(125)]
    queries = [{"id": f"q{i}", "text": "query", "media": {}} for i in range(2)]
    metadata = {"id": name, "track": "document", "full_gallery": True,
                "counts": {"corpus": 125, "queries": 2},
                "text_provenance": {"source": "published_markdown", "query_independent": True,
                                    "gold_fields_used": False}}
    labels = [{"query_id": q["id"], "corpus_id": "d-002", "relevance": 1} for q in queries]
    for filename, data in (("dataset.json", [metadata]), ("corpus.jsonl", corpus),
                           ("queries.jsonl", queries), ("qrels.jsonl", labels)):
        (directory / filename).write_bytes(b"\n".join(runner.canonical(r) for r in data) + b"\n")
    tables = {f.name: runner.frozen.record(tmp_path, f.relative_to(tmp_path).as_posix())
              for f in directory.iterdir()}
    identity = runner.frozen.digest({"files": {k: v["sha256"] for k, v in tables.items()}, "media": {}})
    ranks = {k: [v[0], v[0]] for k, v in rankings().items()}
    identities = {c: runner.sha(c.encode()) for c in runner.CHANNELS}
    refs = {}
    for c in runner.CHANNELS:
        path = tmp_path / (c + ".json")
        runner.fresh(path, {"identity": identities[c], "query_ids": [q["id"] for q in queries],
                            "rankings": ranks[c]})
        refs[c] = runner.frozen.record(tmp_path, path.name)
    cells = []
    for p in runner.graph(ranks, 0):
        cells.append({"variant_id": "document__" + "_".join(p["channels"]).lower() + "__none",
                      "budget_mode": p["budget_mode"], "candidate_k": None, "status": "completed",
                      "query_count": 2, "completed_queries": 2,
                      "metrics": runner.matrix.graded_metrics([r["id"] for r in p["fused"]], {"d-002": 1})})
    report = {"dataset": {"identity": identity}, "channel_identities": identities, "cells": cells,
              "configuration": {"query_ids": [q["id"] for q in queries], "requested_channels": list(runner.CHANNELS),
                                "rrf_k": 60, "channel_budget": 100, "budget_modes": ["per_channel", "total"],
                                "candidate_grid": [20, 50, 100]}}
    runner.fresh(tmp_path / "baseline.json", report)
    baseline = runner.frozen.record(tmp_path, "baseline.json")
    model_files = {}
    for relative in ("model.safetensors", "rl_agent_config.json", "encoder/config.json", "tokenizer/tokenizer.json",
                     "tokenizer/tokenizer_config.json"):
        path = tmp_path / runner.MODEL_SNAPSHOT / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        value = ({"head_max_len": 256} if relative == "rl_agent_config.json" else
                 {"tokenizer_class": "PreTrainedTokenizerFast", "extra_special_tokens": {}})
        path.write_bytes(runner.canonical(value) if relative in ("rl_agent_config.json", "tokenizer/tokenizer_config.json")
                         else b"model fixture only")
        model_files[relative] = runner.model_record(tmp_path, relative)
    gpu = tmp_path / "work/overnight/gpu.lock"
    gpu.parent.mkdir(parents=True)
    gpu.write_bytes(b"")
    recipe = {"protocol": runner.VERSION, "dataset_id": name, "source_sha256": {},
              "sdk_source_sha256": {"fixture_sdk.py": runner.sha(sdk.read_bytes())},
              "dataset_identity": identity, "tables": tables, "media": {},
              "ordered_id_sha256": [runner.frozen.digest([r["id"] for r in rows]) for rows in (corpus, queries)],
              "rankings": refs, "retrieval_provenance": {c: {"report": baseline,
                    "channel_identity": identities[c], "hardware": "fixture", "dtype": "fixture",
                    "stage": "sealed fixture"} for c in runner.CHANNELS},
              "baseline_report": baseline, "accepted_evidence": [baseline],
              "model_snapshot": runner.MODEL_SNAPSHOT, "model_files": model_files,
              "laya_adapter_identity": runner.adapter_identity(),
              "gpu_lock": {"path": "work/overnight/gpu.lock", "device": gpu.stat().st_dev,
                           "inode": gpu.stat().st_ino}}
    manifest = runner.prepare_manifest(tmp_path, recipe)
    return tmp_path, manifest


def test_full_manifest_preparation_is_pure_and_rejects_missing_channel(sealed_fixture):
    root, manifest = sealed_fixture
    assert len(runner.validate(root, manifest)[0].queries) == 2
    del manifest["rankings"]["J"]
    with pytest.raises(ValueError, match="six_sealed_channels"):
        runner.validate(root, manifest)


def test_changed_model_blob_rejected_before_scoring(sealed_fixture):
    root, manifest = sealed_fixture
    path = root / manifest["model_files"]["model.safetensors"]["resolved_file"]["path"]
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="model_snapshot_binding"):
        runner.validate(root, manifest)


def test_changed_ranking_pin_rejected_before_scoring(sealed_fixture):
    root, manifest = sealed_fixture
    (root / manifest["rankings"]["E"]["path"]).write_bytes(b"{}")
    with pytest.raises(ValueError, match="Sealed input bytes changed"):
        runner.validate(root, manifest)


@pytest.mark.parametrize("field", ["rank", "score"])
def test_boolean_ranking_scalar_rejected_even_when_resealed(sealed_fixture, field):
    root, manifest = sealed_fixture
    path = root / manifest["rankings"]["E"]["path"]
    payload = json.loads(path.read_text())
    payload["rankings"][0][0][field] = True
    path.write_bytes(runner.canonical(payload) + b"\n")
    manifest["rankings"]["E"] = runner.frozen.record(root, path.name)
    with pytest.raises(ValueError, match="ranking_scalar_types"):
        runner.validate(root, manifest)


def test_complete_but_wrong_none_baseline_rejected(sealed_fixture):
    root, manifest = sealed_fixture
    path = root / manifest["baseline_report"]["path"]
    baseline = json.loads(path.read_text())
    baseline["cells"][0]["metrics"]["hit@5"] = 0.0
    path.write_bytes(runner.canonical(baseline) + b"\n")
    ref = runner.frozen.record(root, path.name)
    manifest["baseline_report"] = ref
    manifest["accepted_evidence"] = [ref]
    for origin in manifest["retrieval_provenance"].values():
        origin["report"] = ref
    with pytest.raises(ValueError, match="baseline_disagreement"):
        runner.validate(root, manifest)


@pytest.mark.parametrize("forged_wrapped", [False, True])
def test_source_attestation_rejects_foreign_outer_callable(monkeypatch, forged_wrapped):
    root = Path(__file__).resolve().parents[1]
    pins = {p: runner.sha((root / p).read_bytes()) for p in runner.SOURCE_FILES}
    original = runner.matrix.fuse_rankings
    def foreign(*args, **kwargs):
        raise AssertionError("foreign callable must never execute")
    if forged_wrapped:
        foreign.__wrapped__ = original
    monkeypatch.setattr(runner.matrix, "fuse_rankings", foreign)
    with pytest.raises(ValueError, match="loaded_code_changed"):
        runner.sources(root, pins)


def test_source_attestation_rejects_changed_imported_target(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    pins = {p: runner.sha((root / p).read_bytes()) for p in runner.SOURCE_FILES}
    monkeypatch.setattr(runner.generation, "LayaReranker", object)
    with pytest.raises(ValueError, match="imported_call_binding"):
        runner.sources(root, pins)


def test_generated_dataset_constructor_is_attested(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    pins = {p: runner.sha((root / p).read_bytes()) for p in runner.SOURCE_FILES}
    monkeypatch.setattr(runner.matrix.Dataset, "__init__", lambda *args: None)
    with pytest.raises(ValueError, match="dataset_constructor_changed"):
        runner.sources(root, pins)


def test_empty_directory_changes_seal_and_rejects_closed_receipt(tmp_path):
    jobs = tmp_path / "jobs"
    jobs.mkdir(mode=0o700)
    store = runner.Receipts(jobs, "manifest")
    store.run("q-000000", {}, lambda: {}, lambda: None)
    before = runner.private_tree(jobs)
    (jobs / "q-000000/unexpected-empty-directory").mkdir(mode=0o700)
    assert runner.private_tree(jobs) != before
    with pytest.raises(ValueError, match="incomplete_claim"):
        store.read("q-000000", {})


def test_query_parent_directory_is_fsynced_before_operation(tmp_path, monkeypatch):
    jobs = tmp_path / "jobs"
    jobs.mkdir(mode=0o700)
    synced = []
    original = os.fsync
    def fsync(fd):
        s = os.fstat(fd)
        if stat.S_ISDIR(s.st_mode):
            synced.append(s.st_ino)
        original(fd)
    monkeypatch.setattr(runner.os, "fsync", fsync)
    def operation():
        assert jobs.stat().st_ino in synced
        return {}
    runner.Receipts(jobs, "manifest").run("q-000000", {}, operation, lambda: None)


def test_nested_shared_blob_links_are_exact_and_content_addressed(tmp_path):
    raw = b"synthetic shared blob"
    digest = runner.sha(raw)
    shared = tmp_path / ".cache/models/blobs" / digest[:2] / digest
    shared.parent.mkdir(parents=True)
    shared.write_bytes(raw)
    repo = tmp_path / runner.MODEL_REPOSITORY / "blobs"
    repo.mkdir(parents=True)
    (repo / "repo-key").symlink_to("../../blobs/" + digest[:2] + "/" + digest)
    snapshot = tmp_path / runner.MODEL_SNAPSHOT
    snapshot.mkdir(parents=True)
    (snapshot / "model.safetensors").symlink_to("../../blobs/repo-key")
    record = runner.model_record(tmp_path, "model.safetensors")
    assert len(record["link_chain"]) == 2
    assert record["resolved_file"]["sha256"] == digest
    shared.write_bytes(b"changed")
    with pytest.raises(ValueError, match="shared_blob_digest"):
        runner.model_record(tmp_path, "model.safetensors")


def test_tokenizer_only_preflight_binds_union_before_any_claim(sealed_fixture, monkeypatch):
    root, manifest = sealed_fixture
    output = root / "work/vidore-laya-completion-v2/preflight"
    output.mkdir(parents=True, mode=0o700)
    checked = []
    class FakeProbe:
        cfg = {"head_max_len": 256}
        def __init__(self, root, manifest):
            pass
        def check(self, query, document):
            checked.append((query["id"], document["id"]))
            return 64
    monkeypatch.setattr(runner, "TokenizerProbe", FakeProbe)
    seal = runner.sha(runner.canonical(manifest) + b"\n")
    result = runner.preflight(root, manifest, seal)
    assert result["pairs"] == 250 and len(set(checked)) == 250
    assert not (root / manifest["output"]).exists()
    ref = runner.frozen.record(root, (output / "vidore-v3-energy-fr.json").relative_to(root).as_posix())
    dataset, ranks = runner.validate(root, manifest)
    runner.context_evidence(root, manifest, seal, ref, dataset, ranks)
    evidence = json.loads((root / ref["path"]).read_text())
    evidence["queries"][0]["binding"]["input_sha256"] = "changed"
    (root / ref["path"]).write_bytes(runner.canonical(evidence) + b"\n")
    ref = runner.frozen.record(root, ref["path"])
    with pytest.raises(ValueError, match="context_preflight_binding"):
        runner.context_evidence(root, manifest, seal, ref, dataset, ranks)


def test_preflight_overflow_leaves_no_receipt_or_run_claim(sealed_fixture, monkeypatch):
    root, manifest = sealed_fixture
    output = root / "work/vidore-laya-completion-v2/preflight"
    output.mkdir(parents=True, mode=0o700)
    class Overflow:
        def __init__(self, root, manifest):
            pass
        def check(self, query, document):
            raise ValueError("context_overflow")
    monkeypatch.setattr(runner, "TokenizerProbe", Overflow)
    seal = runner.sha(runner.canonical(manifest) + b"\n")
    with pytest.raises(ValueError, match="context_overflow"):
        runner.preflight(root, manifest, seal)
    assert list(output.iterdir()) == []
    assert not (root / manifest["output"]).exists()


def test_execution_requires_context_receipt_before_output_claim(sealed_fixture):
    root, manifest = sealed_fixture
    seal = runner.sha(runner.canonical(manifest) + b"\n")
    with pytest.raises(ValueError, match="context_preflight_required"):
        runner.execute(root, manifest, seal)
    assert not (root / manifest["output"]).exists()


def test_probe_calls_exact_unbound_encoder_with_8192_and_checkpoint_head_budget():
    seen = []
    class FakeAgent:
        @staticmethod
        def _encode_state(holder, state, ids, internal, *, max_len, head_max_len):
            seen.append((state, ids, internal, max_len, head_max_len))
            return [{"ids": [1] * 60, "state_stats": {"truncated": False, "state_tokens_dropped": 0},
                     "options": {"options_distinct": 2, "options": 2}}]
    probe = object.__new__(runner.TokenizerProbe)
    probe.agent = types.SimpleNamespace(Agent=FakeAgent)
    probe.common = types.SimpleNamespace(serialize_state=lambda state: json.dumps(state, ensure_ascii=False))
    probe.cfg = {"head_max_len": 256}
    probe.internal = {"relevant": {"t": "noul"}}
    assert probe.check({"text": "synthetic query"}, {"title": "title", "text": "synthetic passage"}) == 60
    assert seen == [({"query": "synthetic query", "passage": "title\nsynthetic passage"},
                     ["relevant"], probe.internal, 8192, 256)]


def test_incompatible_tokenizer_configuration_rejected_without_constructor(sealed_fixture, monkeypatch):
    root, manifest = sealed_fixture
    monkeypatch.setattr(runner, "sdk_bindings", lambda *args: (None, None))
    ref = manifest["model_files"]["tokenizer/tokenizer_config.json"]["resolved_file"]
    path = root / ref["path"]
    path.write_bytes(b'{"tokenizer_class":"TokenizersBackend"}')
    manifest["model_files"]["tokenizer/tokenizer_config.json"]["resolved_file"] = runner.frozen.record(root, ref["path"])
    with pytest.raises(ValueError, match="tokenizer_would_require_write"):
        runner.TokenizerProbe(root, manifest)

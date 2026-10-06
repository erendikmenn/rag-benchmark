import json
import math
from pathlib import Path

import numpy as np
import pytest

from rag_benchmark.multimodal import (
    PrecomputedAdapter, UnsupportedConfiguration, aggregate_metrics, base_rankings,
    cached_encode, channel_allocations, encoding_cache_identity, exact_dot_rankings,
    fuse_rankings, graded_metrics, load_dataset, matrix_registry, public_adapter_spec, run_matrix,
)


def fixture_dataset(tmp_path, *, track="code", corpus=None, queries=None, qrels=None):
    root = tmp_path / "data"
    root.mkdir(parents=True, exist_ok=True)
    manifest = {"id": "fixture", "revision": "fixed-r1", "track": track, "split": "test",
                "text_provenance": {"source": "source_code_docstrings_removed", "query_independent": True,
                                    "gold_fields_used": False}}
    corpus = corpus or [{"id": "c", "text": "slow read"}, {"id": "a", "text": "readFile write data"},
                        {"id": "b", "text": "read_file data"}]
    queries = queries or [{"id": "q1", "text": "read file"}, {"id": "q2", "text": "write"}]
    qrels = qrels or [{"query_id": "q1", "corpus_id": "a", "relevance": 3},
                      {"query_id": "q1", "corpus_id": "b", "relevance": 1},
                      {"query_id": "q2", "corpus_id": "a", "relevance": 2}]
    (root / "dataset.json").write_text(json.dumps(manifest))
    for name, rows in (("corpus", corpus), ("queries", queries), ("qrels", qrels)):
        (root / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return root


class CountingEncoder:
    identity = "model-revision-and-prompts-r1"
    dimension = 3
    prompt_identity = {"query": "search", "document": "passage"}

    def __init__(self):
        self.calls = []

    def encode(self, items, role):
        self.calls.append((role, [item["id"] for item in items]))
        assert all(set(item) <= {"id", "text", "media"} for item in items)
        mapping = {"a": [1, 0, 0], "b": [0, 1, 0], "c": [0, 0, 1],
                   "q1": [1, 0, 0], "q2": [0, 1, 0]}
        return np.asarray([mapping[item["id"]] for item in items], dtype=np.float32)


class RecordingReranker:
    pointwise = True

    def __init__(self, identity):
        self.identity, self.calls = identity, []

    def score(self, query, candidates):
        self.calls.append((query["id"], [candidate["id"] for candidate in candidates]))
        return [float({"a": 3, "b": 2, "c": 1}[candidate["id"]]) for candidate in candidates]


def test_full_registry_counts_unique_families_and_no_artificial_specialist():
    rows = matrix_registry()
    assert len(rows) == len({row["variant_id"] for row in rows}) == 1284
    assert len({(row["track"], tuple(row["retrieval_channels"])) for row in rows}) == 321
    assert len(matrix_registry("code")) == 28
    assert len(matrix_registry("speech")) == 124
    assert all("S" not in row["retrieval_channels"] for row in matrix_registry("speech"))


def test_registry_matches_approved_csv_if_available():
    import csv
    csv_path = Path(__file__).resolve().parents[2] / "multimodal-cekirdek-varyantlar.csv"
    if not csv_path.exists():
        pytest.skip("Planning artifact is outside installed package")
    expected = {row["variant_id"] for row in csv.DictReader(csv_path.open())}
    assert {row["variant_id"] for row in matrix_registry()} == expected


def test_multiple_positive_hit_recall_and_graded_ndcg_are_distinct():
    values = graded_metrics(["b", "b", "x", "a"], {"a": 3, "b": 1}, ks=(1, 3))
    assert values["hit@1"] == 1
    assert values["recall@1"] == 0.5
    assert values["recall@3"] == 1
    assert values["mrr@3"] == 1
    assert values["ndcg@1"] == pytest.approx(1 / 7)
    expected = (1 + 7 / math.log2(4)) / (7 + 1 / math.log2(3))
    assert values["ndcg@3"] == pytest.approx(expected)
    assert values["map@16"] == pytest.approx((1 + 2 / 3) / 2)
    assert aggregate_metrics([values, values]) == values


def test_exact_ranking_breaks_ties_by_id_and_excludes_before_cutoff():
    docs = np.asarray([[1, 0], [1, 0], [0, 1]], dtype=np.float32)
    queries = np.asarray([[1, 0], [0, 1]], dtype=np.float32)
    rows = exact_dot_rankings(docs, queries, ["z", "a", "c"], top_k=2, query_batch_size=1,
                             exclusions=[{"a"}, set()])
    assert [row["id"] for row in rows[0]] == ["z", "c"]
    assert [row["id"] for row in rows[1]] == ["c", "a"]


def test_rrf_is_channel_order_independent_and_budget_never_refills():
    def ranking(ids):
        return [{"id": identifier, "score": 10 - i} for i, identifier in enumerate(ids)]
    rankings = {"B": ranking(["a", "b", "c"]), "G": ranking(["a", "b", "d"]), "E": ranking(["b", "a", "e"])}
    full, full_usage = fuse_rankings(rankings, ("B", "G", "E"), budget=3)
    reverse, _ = fuse_rankings(rankings, ("E", "G", "B"), budget=3)
    assert full == reverse
    assert full_usage["raw_candidates"] == 9
    controlled, usage = fuse_rankings(rankings, ("B", "G", "E"), budget_mode="total", budget=3)
    assert usage["raw_candidates"] == 3
    assert usage["unique_candidates"] == 2
    assert usage["refill"] is False
    assert len(controlled) == 2
    assert channel_allocations(("G", "B", "E"), "total", 100) == {"B": 34, "G": 33, "E": 33}
    tied, _ = fuse_rankings({"B": ranking(["z", "a"]), "G": ranking(["a", "z"])}, ["B", "G"])
    assert [row["id"] for row in tied] == ["a", "z"]


def test_encoder_cache_separates_roles_revision_prompts_and_dimension(tmp_path):
    root = fixture_dataset(tmp_path)
    dataset = load_dataset(root)
    adapter = CountingEncoder()
    values, first = cached_encode(dataset, adapter, "document", tmp_path / "cache", block_size=2)
    again, second = cached_encode(dataset, adapter, "document", tmp_path / "cache", block_size=2)
    assert len(adapter.calls) == 2
    np.testing.assert_equal(values, again)
    assert first["new_rows"] == 3 and second["cache_rows"] == 3
    assert second["inference_seconds"] is None
    identity = encoding_cache_identity(dataset, adapter, "document")
    assert identity != encoding_cache_identity(dataset, adapter, "query")
    adapter.prompt_identity = {"query": "changed", "document": "changed"}
    assert identity != encoding_cache_identity(dataset, adapter, "document")
    adapter.prompt_identity = CountingEncoder.prompt_identity
    adapter.dimension = 2
    assert identity != encoding_cache_identity(dataset, adapter, "document")
    adapter.dimension = 3
    manifest = json.loads((root / "dataset.json").read_text())
    manifest["revision"] = "fixed-r2"
    (root / "dataset.json").write_text(json.dumps(manifest))
    assert identity != encoding_cache_identity(load_dataset(root), adapter, "document")


def test_partial_embedding_blocks_resume_without_repeating_completed_blocks(tmp_path):
    dataset = load_dataset(fixture_dataset(tmp_path))
    adapter = CountingEncoder()
    original_encode = adapter.encode
    def interrupted(items, role):
        if items[0]["id"] == "b":
            raise RuntimeError("interrupted")
        return original_encode(items, role)
    adapter.encode = interrupted
    with pytest.raises(RuntimeError):
        cached_encode(dataset, adapter, "document", tmp_path / "cache", block_size=2)
    adapter.encode = original_encode
    _, usage = cached_encode(dataset, adapter, "document", tmp_path / "cache", block_size=2)
    assert usage["cache_rows"] == 2 and usage["new_rows"] == 1


def test_leakage_is_rejected_and_gold_metadata_not_forwarded(tmp_path):
    root = fixture_dataset(tmp_path)
    dataset = load_dataset(root)
    candidate = {"id": "a", "text": "independent", "metadata": {"gold_answer": "SECRET"}}
    assert "SECRET" not in json.dumps(dataset.model_item(candidate))
    manifest = json.loads((root / "dataset.json").read_text())
    manifest["text_provenance"]["gold_fields_used"] = True
    (root / "dataset.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="leakage"):
        load_dataset(root)


def test_qrels_missing_positive_and_unknown_id_are_rejected(tmp_path):
    root = fixture_dataset(tmp_path, qrels=[{"query_id": "q1", "corpus_id": "a", "relevance": 1}])
    with pytest.raises(ValueError, match="positive"):
        load_dataset(root)
    (root / "qrels.jsonl").write_text(json.dumps({"query_id": "q1", "corpus_id": "unknown", "relevance": 1}))
    with pytest.raises(ValueError, match="unknown"):
        load_dataset(root)


def test_media_bytes_invalidate_dataset_fingerprint(tmp_path):
    root = fixture_dataset(tmp_path, track="photo")
    media = root / "image.png"
    media.write_bytes(b"image-v1")
    rows = [json.loads(line) for line in (root / "corpus.jsonl").read_text().splitlines()]
    rows[0]["media"] = {"image": "image.png"}
    (root / "corpus.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    before = load_dataset(root)
    media.write_bytes(b"image-v2")
    after = load_dataset(root)
    assert before.identity != after.identity


def test_precomputed_vectors_validate_dataset_and_do_not_claim_inference_speed(tmp_path):
    dataset = load_dataset(fixture_dataset(tmp_path))
    adapter = PrecomputedAdapter(np.eye(3), np.eye(3)[:2], document_ids=["c", "a", "b"],
                                 query_ids=["q1", "q2"], identity="external-model-r1", dataset_identity=dataset.identity)
    _, usage = cached_encode(dataset, adapter, "document", tmp_path / "cache")
    assert usage["precomputed"] is True and usage["inference_seconds"] is None
    adapter.dataset_identity = "stale"
    with pytest.raises(ValueError, match="different"):
        encoding_cache_identity(dataset, adapter, "query")


def test_matrix_same_pools_resume_and_public_report_has_no_text(tmp_path):
    root = fixture_dataset(tmp_path)
    encoders = {"G": CountingEncoder(), "E": CountingEncoder()}
    rerankers = {name: RecordingReranker(name) for name in ("laya_text", "bge_reranker_text", "gemma4_relevance")}
    kwargs = dict(adapters=encoders, rerankers=rerankers, requested_channels=["B", "G", "E"],
                  candidate_grid=(20, 50, 100), budget_modes=("per_channel", "total"))
    report = run_matrix(root, tmp_path / "run", **kwargs)
    assert report["family_status_counts"] == {"completed": 28, "planned": 0, "failed": 0, "unsupported": 0}
    assert report["cell_status_counts"]["completed"] == 140
    for reranker in rerankers.values():
        # Sparse lexical pools can require later dense-only pairs, but each pair
        # is actually scored once across all subsets, cutoffs and budget modes.
        pairs = [(query, identifier) for query, identifiers in reranker.calls for identifier in identifiers]
        assert len(pairs) == len(set(pairs)) == 6
        assert set(pairs) == {(query, identifier) for query in ("q1", "q2") for identifier in ("a", "b", "c")}
    assert all(reranker.calls == rerankers["laya_text"].calls for reranker in rerankers.values())
    before_calls = sum(len(encoder.calls) for encoder in encoders.values())
    again = run_matrix(root, tmp_path / "run", **kwargs)
    assert sum(len(encoder.calls) for encoder in encoders.values()) == before_calls
    assert all(cell["result_cache_hits"] == 2 for cell in again["cells"])
    assert all(cell["rerank_fresh_inference_seconds"] is None for cell in again["cells"])
    public = (tmp_path / "run" / "report.json").read_text()
    assert "readFile" not in public and "slow read" not in public and '"text"' not in public
    assert all(row["ranking_cache_hit"] for row in again["channels"].values())


def test_missing_channels_and_rerankers_are_unsupported_not_fallback(tmp_path):
    root = fixture_dataset(tmp_path)
    report = run_matrix(root, tmp_path / "run", candidate_grid=(50,), budget_modes=("total",))
    completed = [row for row in report["families"] if row["status"] == "completed"]
    assert [row["variant_id"] for row in completed] == ["code__b__none"]
    assert report["family_status_counts"]["unsupported"] == 27
    assert all("metrics" not in cell for cell in report["cells"] if cell["status"] == "unsupported")


def test_missing_candidate_text_does_not_block_native_channel(tmp_path):
    root = fixture_dataset(tmp_path, track="photo", corpus=[{"id": "c"}, {"id": "a"}, {"id": "b"}])
    report = run_matrix(root, tmp_path / "run", adapters={"N": CountingEncoder()}, requested_channels=["B", "N"],
                        candidate_grid=(50,), budget_modes=("per_channel",))
    assert report["channels"]["B"]["status"] == "unsupported"
    assert report["channels"]["N"]["status"] == "completed"
    assert next(row for row in report["families"] if row["variant_id"] == "photo__n__none")["status"] == "completed"


def test_invalid_encoder_output_marks_failed_without_score(tmp_path):
    root = fixture_dataset(tmp_path)
    adapter = CountingEncoder()
    adapter.encode = lambda items, role: np.zeros((len(items), 3))
    report = run_matrix(root, tmp_path / "run", adapters={"G": adapter}, requested_channels=["G"],
                        candidate_grid=(50,), budget_modes=("per_channel",))
    assert report["channels"]["G"]["status"] == "failed"
    assert all("metrics" not in cell for cell in report["cells"])


def test_cirr_reference_is_excluded_and_positive_reference_is_invalid(tmp_path):
    query = [{"id": "q1", "text": "change", "metadata": {"reference_id": "c"}}]
    qrels = [{"query_id": "q1", "corpus_id": "a", "relevance": 1}]
    root = fixture_dataset(tmp_path, track="composed_image", queries=query, qrels=qrels)
    dataset = load_dataset(root)
    ranks, _ = base_rankings(dataset, "N", {"N": CountingEncoder()}, tmp_path / "cache")
    assert "c" not in [row["id"] for row in ranks[0]]
    qrels[0]["corpus_id"] = "c"
    (root / "qrels.jsonl").write_text(json.dumps(qrels[0]))
    with pytest.raises(ValueError, match="excludes"):
        load_dataset(root)


def test_bm25_code_identifiers_are_split(tmp_path):
    dataset = load_dataset(fixture_dataset(tmp_path))
    rankings, _ = base_rankings(dataset, "B", {}, tmp_path / "cache")
    assert {row["id"] for row in rankings[0][:2]} == {"a", "b"}


def test_bm25_positive_only_no_padding_exclusions_and_exact_boundary_ties(tmp_path):
    corpus = [{"id": "z/source", "text": "needle evidence"}, {"id": "a:source", "text": "needle evidence"},
              {"id": "zero", "text": "unrelated document"}]
    queries = [{"id": "matches", "text": "needle"}, {"id": "unknown", "text": "absentvocabulary"},
               {"id": "excluded", "text": "needle", "metadata": {"exclude_ids": ["a:source"]}},
               {"id": "no_tokens", "text": "!!!"}]
    qrels = [{"query_id": query["id"], "corpus_id": "z/source", "relevance": 1} for query in queries]
    dataset = load_dataset(fixture_dataset(tmp_path, corpus=corpus, queries=queries, qrels=qrels))
    ranks, _ = base_rankings(dataset, "B", {}, tmp_path / "cache", top_k=10)
    assert [row["id"] for row in ranks[0]] == ["a:source", "z/source"]
    assert ranks[0][0]["score"] == ranks[0][1]["score"] > 0
    assert ranks[1] == [] and ranks[3] == []
    assert [row["id"] for row in ranks[2]] == ["z/source"]
    short, _ = base_rankings(dataset, "B", {}, tmp_path / "cache", top_k=1)
    assert short[0] == ranks[0][:1]
    assert short[2] == ranks[2]


def test_bm25_policy_invalidates_legacy_b_cache_only_and_is_public(tmp_path):
    from rag_benchmark.models import package_versions, stable_hash
    from rag_benchmark.multimodal import BM25_CANDIDATE_SELECTION, ENGINE_VERSION, _channel_identity
    dataset = load_dataset(fixture_dataset(tmp_path))
    legacy_model = {"tokenizer": "code-identifiers-v1", "bm25": {"k1": 1.5, "b": 0.75, "method": "lucene"},
                    "packages": package_versions(("bm25s",))}
    legacy_identity = stable_hash({"engine": ENGINE_VERSION, "dataset": dataset.identity, "channel": "B", "model": legacy_model})
    old_path = tmp_path / "cache/rankings" / stable_hash({"identity": legacy_identity, "top_k": 100}) / "rankings.json"
    old_path.parent.mkdir(parents=True)
    old_path.write_text(json.dumps({"rankings": [[{"id": "c", "score": 0, "rank": 1}]] * 2, "usage": {}}))
    rows, usage = base_rankings(dataset, "B", {}, tmp_path / "cache")
    assert usage["identity"] != legacy_identity and usage["ranking_cache_hit"] is False
    assert [row["id"] for row in rows[1]] == ["a"]
    assert old_path.exists()  # Older observations remain available.
    adapter = CountingEncoder()
    legacy_dense = {role: encoding_cache_identity(dataset, adapter, role) for role in ("document", "query")}
    assert _channel_identity(dataset, "G", {"G": adapter}) == stable_hash({"engine": ENGINE_VERSION,
        "dataset": dataset.identity, "channel": "G", "model": legacy_dense})
    report = run_matrix(dataset.root, tmp_path / "run", requested_channels=["B"],
                        candidate_grid=(50,), budget_modes=("per_channel",))
    assert report["adapter_specs"]["channels"]["B"]["config"]["candidate_selection"] == BM25_CANDIDATE_SELECTION
    assert BM25_CANDIDATE_SELECTION == "positive_scores_only_v1"


def test_text_provenance_must_be_explicit(tmp_path):
    root = fixture_dataset(tmp_path)
    manifest = json.loads((root / "dataset.json").read_text())
    del manifest["text_provenance"]
    (root / "dataset.json").write_text(json.dumps(manifest))
    dataset = load_dataset(root)
    with pytest.raises(UnsupportedConfiguration, match="provenance"):
        dataset.require_text()


def test_media_query_text_surrogate_preserves_complete_context(tmp_path):
    query = [{"id": "q1", "text": "change", "media": {"image": "ref.png"},
              "metadata": {"reference_id": "c"}}]
    qrels = [{"query_id": "q1", "corpus_id": "a", "relevance": 1}]
    root = fixture_dataset(tmp_path, track="composed_image", queries=query, qrels=qrels)
    (root / "ref.png").write_bytes(b"local-fixture")
    dataset = load_dataset(root)
    with pytest.raises(UnsupportedConfiguration, match="text_surrogate"):
        dataset.require_text()
    query[0]["metadata"]["text_surrogate"] = {"text": "reference photograph: cat; change: running",
                                               "provenance": {"gold_fields_used": False}}
    (root / "queries.jsonl").write_text(json.dumps(query[0]))
    dataset = load_dataset(root)
    dataset.require_text()
    assert dataset.model_item(dataset.queries[0], text_only=True)["text"] == "reference photograph: cat; change: running"
    assert dataset.model_item(dataset.queries[0])["text"] == "change"
    assert "image" in dataset.model_item(dataset.queries[0])["media"]


def test_reranker_cannot_drop_or_add_candidates(tmp_path):
    root = fixture_dataset(tmp_path)
    bad = RecordingReranker("broken-count")
    bad.score = lambda query, candidates: [1.0]
    report = run_matrix(root, tmp_path / "run", rerankers={"laya_text": bad}, requested_channels=["B"],
                        candidate_grid=(50,), budget_modes=("per_channel",))
    cell = next(cell for cell in report["cells"] if cell["variant_id"] == "code__b__laya_text")
    assert cell["status"] == "failed"
    assert "metrics" not in cell


def test_bootstrap_pairs_and_groups_shared_sources():
    from rag_benchmark.multimodal import paired_group_bootstrap
    left = {"caption1": 0.0, "caption2": 0.0, "caption3": 1.0}
    right = {"caption1": 1.0, "caption2": 1.0, "caption3": 0.0}
    groups = {"caption1": "image-a", "caption2": "image-a", "caption3": "image-b"}
    result = paired_group_bootstrap(left, right, groups, samples=500, seed=7)
    assert result == paired_group_bootstrap(left, right, groups, samples=500, seed=7)
    assert result["group_count"] == 2 and result["query_count"] == 3
    assert result["wins"] == 2 and result["losses"] == 1
    assert result["mean_difference"] == pytest.approx(1 / 3)
    assert result["ci95_low"] == -1 and result["ci95_high"] == 1
    with pytest.raises(ValueError, match="identical"):
        paired_group_bootstrap(left, {"caption1": 1}, groups)


def test_joint_requires_gold_free_candidate_text(tmp_path):
    root = fixture_dataset(tmp_path, track="photo")
    manifest = json.loads((root / "dataset.json").read_text())
    del manifest["text_provenance"]
    (root / "dataset.json").write_text(json.dumps(manifest))
    with pytest.raises(UnsupportedConfiguration, match="provenance"):
        base_rankings(load_dataset(root), "J", {"J": CountingEncoder()}, tmp_path / "cache")


def test_direct_late_interaction_adapter_receives_exclusions_and_frozen_cache(tmp_path):
    query = [{"id": "q1", "text": "change", "metadata": {"reference_id": "c"}}]
    qrels = [{"query_id": "q1", "corpus_id": "a", "relevance": 1}]
    dataset = load_dataset(fixture_dataset(tmp_path, track="composed_image", queries=query, qrels=qrels))
    class DirectRanker:
        identity = "token-maxsim-model-and-prompt-r1"

        def rank(self, documents, queries, *, top_k, cache_dir, exclusions):
            assert exclusions == [{"c"}]
            assert cache_dir.name != self.identity  # engine includes dataset fingerprint
            assert [item["id"] for item in documents] == ["c", "a", "b"]
            return [[{"id": "b", "score": 1.0}, {"id": "a", "score": 1.0}]], {"search": "exact_maxsim"}
    rankings, usage = base_rankings(dataset, "N", {"N": DirectRanker()}, tmp_path / "cache", top_k=2)
    assert [row["id"] for row in rankings[0]] == ["a", "b"]
    assert usage["search"] == "exact_maxsim"


def test_declared_empty_source_extraction_is_preserved_without_filler(tmp_path):
    root = fixture_dataset(tmp_path, corpus=[{"id": "c", "text": ""}, {"id": "a", "text": "read"},
                                             {"id": "b", "text": "read file"}])
    manifest = json.loads((root / "dataset.json").read_text())
    manifest["text_provenance"]["source"] = "published_markdown"
    (root / "dataset.json").write_text(json.dumps(manifest))
    dataset = load_dataset(root)
    dataset.require_text()
    assert dataset.public_summary()["empty_candidate_text_count"] == 1
    assert dataset.public_summary()["missing_candidate_text_count"] == 0
    assert dataset.model_item(dataset.corpus[0])["text"] == ""
    rankings, _ = base_rankings(dataset, "B", {}, tmp_path / "cache")
    assert {row["id"] for row in rankings[0]} == {"a", "b"}
    assert rankings[1] == []  # No positive lexical match for "write".
    assert {row["id"] for row in dataset.corpus} == {"a", "b", "c"}


class DiagnosticEncoder(CountingEncoder):
    def encode(self, items, role):
        values = super().encode(items, role)
        records = []
        for item in items:
            original = 70 if item["id"] == "q1" else 40
            records.append({"id": item["id"], "original_tokens": original, "retained_tokens": min(64, original),
                            "native_token_limit": 64, "truncated": original > 64,
                            "text": "PRIVATE ADAPTER INPUT", "media_path": "/PRIVATE/path"})
        self.last_usage = {"items": records, "native_text_limit": 64,
                           "truncated_text_items": sum(row["truncated"] for row in records),
                           "text_overflow_policy": "truncate_to_model_limit", "inference_seconds": 999.0,
                           "prompt": "PRIVATE PROMPT"}
        return values


def test_processing_diagnostics_aggregate_across_blocks_and_survive_vector_cache(tmp_path):
    dataset = load_dataset(fixture_dataset(tmp_path))
    adapter = DiagnosticEncoder()
    _, first = cached_encode(dataset, adapter, "query", tmp_path / "cache", block_size=1)
    _, again = cached_encode(dataset, adapter, "query", tmp_path / "cache", block_size=1)
    assert len(adapter.calls) == 2
    assert first["adapter_diagnostics"] == again["adapter_diagnostics"]
    diagnostics = again["adapter_diagnostics"]
    assert diagnostics["status"] == "complete"
    assert diagnostics["truncated_text_items"] == 1
    assert diagnostics["original_tokens_total"] == 110
    assert diagnostics["retained_tokens_total"] == 104
    assert diagnostics["text_overflow_policies"] == ["truncate_to_model_limit"]
    assert [item["id"] for item in diagnostics["items"]] == ["q1", "q2"]
    assert "PRIVATE" not in json.dumps(diagnostics)
    assert again["inference_seconds"] is None


def test_full_report_preserves_diagnostics_on_ranking_cache_hit_without_old_timing(tmp_path):
    root = fixture_dataset(tmp_path, track="photo")
    kwargs = dict(adapters={"S": DiagnosticEncoder()}, requested_channels=["S"],
                  candidate_grid=(50,), budget_modes=("per_channel",))
    first = run_matrix(root, tmp_path / "run", **kwargs)
    again = run_matrix(root, tmp_path / "run", **kwargs)
    first_diag = first["channels"]["S"]["query_encoding"]["adapter_diagnostics"]
    again_diag = again["channels"]["S"]["query_encoding"]["adapter_diagnostics"]
    assert first_diag == again_diag and again_diag["truncated_text_items"] == 1
    assert again["channels"]["S"]["ranking_cache_hit"] is True
    assert again["channels"]["S"]["query_encoding"]["inference_seconds"] is None
    assert again["channels"]["S"]["exact_search_seconds"] is None
    assert "PRIVATE" not in (tmp_path / "run/report.json").read_text()


def test_legacy_cache_reports_unknown_truncation_without_reencoding(tmp_path):
    dataset = load_dataset(fixture_dataset(tmp_path))
    adapter = DiagnosticEncoder()
    base_rankings(dataset, "G", {"G": adapter}, tmp_path / "cache")
    for path in (tmp_path / "cache/vectors").rglob("*.json"):
        path.unlink()
    ranking_path = next((tmp_path / "cache/rankings").rglob("rankings.json"))
    payload = json.loads(ranking_path.read_text())
    del payload["usage"]
    ranking_path.write_text(json.dumps(payload))
    calls = len(adapter.calls)
    _, usage = base_rankings(dataset, "G", {"G": adapter}, tmp_path / "cache")
    assert len(adapter.calls) == calls
    diagnostics = usage["query_encoding"]["adapter_diagnostics"]
    assert diagnostics["status"] == "unavailable"
    assert diagnostics["missing_rows"] == 2
    assert diagnostics["truncated_text_items"] is None


def test_text_only_item_removes_media_after_resolving_complete_surrogate(tmp_path):
    dataset = load_dataset(fixture_dataset(tmp_path))
    item = {"id": "q1", "text": "instruction", "media": {"image": "reference.png"},
            "metadata": {"text_surrogate": "source description plus instruction"}}
    result = dataset.model_item(item, text_only=True)
    assert result == {"id": "q1", "text": "source description plus instruction", "media": {}}
    assert dataset.model_item(item)["text"] == "instruction"


def test_bge_empty_candidate_uses_real_model_input_no_score_floor(monkeypatch):
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace
    from rag_benchmark.multimodal import BGETextReranker
    from rag_benchmark import multimodal_models
    seen = []
    class Base:
        def __init__(self, config):
            self.identity = "fixed-native-model"
            self.config = {"batch_size": 8, "max_length": 20, "device": "cpu"}

        def _processor(self, pairs, **kwargs):
            seen.extend(pairs)
            assert kwargs["truncation"] is False
            return {"attention_mask": torch.ones((len(pairs), 3), dtype=torch.int64)}

        def _load(self):
            return lambda **kwargs: SimpleNamespace(logits=torch.tensor([[0.7], [-0.3]]))
    monkeypatch.setattr(multimodal_models, "BGEReranker", Base)
    reranker = BGETextReranker()
    scores = reranker.score({"text": "question"}, [{"id": "a", "text": "evidence"}, {"id": "b", "text": ""}])
    assert scores == pytest.approx([0.7, -0.3])
    assert seen == [("question", "evidence"), ("question", "")]
    assert reranker.last_usage["empty_candidate_text_items"] == 1
    with pytest.raises(UnsupportedConfiguration, match="explicit"):
        reranker.score({"text": "question"}, [{"id": "b"}])


def test_direct_adapter_usage_is_sanitized_and_cache_latency_not_replayed(tmp_path):
    dataset = load_dataset(fixture_dataset(tmp_path))
    class Direct:
        identity = "token-model-v1"

        def rank(self, documents, queries, **kwargs):
            ranking = [{"id": row["id"], "score": 1.0} for row in documents]
            return [ranking.copy() for _ in queries], {"scoring_seconds": 99.0,
                "query_text": "PRIVATE QUERY", "search": "exact_maxsim",
                "query_encoding": {"total_items": 2, "new_items": 2, "inference_seconds": 50.0, "prompt": "PRIVATE PROMPT"}}
    _, fresh = base_rankings(dataset, "S", {"S": Direct()}, tmp_path / "cache")
    _, cached = base_rankings(dataset, "S", {"S": Direct()}, tmp_path / "cache")
    assert fresh["scoring_seconds"] == 99.0 and cached["scoring_seconds"] is None
    assert cached["query_encoding"]["inference_seconds"] is None
    assert cached["query_encoding"]["cache_hits"] == 2
    assert "PRIVATE" not in json.dumps(fresh) and "PRIVATE" not in json.dumps(cached)


def test_public_adapter_specs_unwrap_known_adapters_and_exclude_private_configuration():
    from types import SimpleNamespace
    config = {"model_id": "organization/model", "revision": "a" * 40, "base_model_id": "organization/base",
        "base_revision": "b" * 40, "model_sha256": "c" * 64, "projector_sha256": "d" * 64,
        "device": "mps", "dtype": "bfloat16", "batch_size": 8, "dimension": 768,
        "max_length": 8192, "text_overflow_policy": "truncate_to_model_limit", "vision_budget": 280,
        "video_fps": 1, "video_max_frames": 16, "temperature": 0, "seed": 42,
        "model_path": "/PRIVATE/model.gguf", "cache_dir": "/PRIVATE/cache", "base_url": "http://PRIVATE",
        "api_key": "PRIVATE_KEY", "query_prompt": "PRIVATE_PROMPT", "query": "PRIVATE_QUERY"}
    model = SimpleNamespace(config=config, dimension=768, identity="e" * 64)
    segmenter = SimpleNamespace(identity="f" * 64, protocol={
        "source_coverage": "exact_concatenation_no_overlap_no_normalization_no_dropped_characters",
        "max_tokens_after_native_document_formatting": 8192,
        "tokenizers": {"bge": config, "embeddinggemma": config, "PRIVATE": config},
        "native_document_formats": {"bge": "PRIVATE_PROMPT"}, "cache_dir": "/PRIVATE/cache"})
    wrapper = SimpleNamespace(base=SimpleNamespace(adapter=model), encoder=model, embedder=model,
        generator=SimpleNamespace(verifier=model), segmenter=segmenter)
    spec = public_adapter_spec(wrapper)
    assert spec["base"]["adapter"]["config"]["revision"] == "a" * 40
    assert spec["encoder"]["config"]["model_id"] == "organization/model"
    assert spec["generator"]["verifier"]["config"]["projector_sha256"] == "d" * 64
    assert spec["segmenter"]["protocol"]["tokenizers"]["bge"]["max_length"] == 8192
    serialized = json.dumps(spec)
    assert "PRIVATE" not in serialized
    assert not any(key in serialized for key in ("model_path", "cache_dir", "api_key", "query_prompt", "native_document_formats"))
    assert spec["embedder"]["config"]["vision_budget"] == 280
    assert spec["embedder"]["config"]["text_overflow_policy"] == "truncate_to_model_limit"
    model.config = {"model_id": "/private/local/model", "dtype": ["PRIVATE"], "seed": float("nan")}
    assert public_adapter_spec(model)["config"] == {}
    wrapper.base = wrapper
    assert public_adapter_spec(wrapper)["base"]["recursive_reference_omitted"] is True


def test_report_exposes_adapter_configs_without_altering_existing_cache_identity(tmp_path):
    root = fixture_dataset(tmp_path)
    dataset = load_dataset(root)
    encoder = CountingEncoder()
    encoder.config = {"model_id": "organization/dense", "revision": "1" * 40, "batch_size": 2,
                      "dtype": "float32", "device": "cpu", "prompt": "PRIVATE_PROMPT"}
    reranker = RecordingReranker("fixed-rerank")
    reranker.config = {"model_id": "organization/rerank", "revision": "2" * 40, "batch_size": 3}
    kwargs = dict(adapters={"G": encoder}, rerankers={"bge_reranker_text": reranker}, requested_channels=["G"],
                  candidate_grid=(50,), budget_modes=("per_channel",))
    before = encoding_cache_identity(dataset, encoder, "document")
    first = run_matrix(root, tmp_path / "run", **kwargs)
    assert first["adapter_specs"]["channels"]["G"]["config"]["batch_size"] == 2
    assert first["adapter_specs"]["rerankers"]["bge_reranker_text"]["config"]["model_id"] == "organization/rerank"
    # Reporting reads configuration but does not introduce a second identity scheme.
    encoder.config["batch_size"] = 4
    again = run_matrix(root, tmp_path / "run", **kwargs)
    assert again["adapter_specs"]["channels"]["G"]["config"]["batch_size"] == 4
    assert before == encoding_cache_identity(dataset, encoder, "document")
    assert first["configuration"] == again["configuration"]
    assert [row.get("identity") for row in first["cells"]] == [row.get("identity") for row in again["cells"]]
    assert again["channels"]["G"]["ranking_cache_hit"] is True
    assert "PRIVATE" not in (tmp_path / "run/report.json").read_text()


def test_code_rerank_diagnostics_preserve_processing_but_count_only_fresh_work(tmp_path):
    import sqlite3
    root = fixture_dataset(tmp_path)
    class CodeReranker(RecordingReranker):
        def score(self, query, candidates):
            scores = super().score(query, candidates)
            self.last_usage = {"original_function_count": len(candidates), "chunk_count": len(candidates) * 2,
                "segmented_function_count": len(candidates), "native_short_function_count": 0,
                "source_utf8_bytes": 100, "covered_source_utf8_bytes": 100,
                "aggregation": "maximum_actual_base_relevance_score_per_original_function",
                "segmentation_identity": "a" * 64,
                "protocol": {"candidate_policy": "one_score_per_original_function_in_original_order;no_drops_or_score_floor",
                             "native_document_formats": "PRIVATE_PROMPT"},
                "base_usage": {"query": "PRIVATE_QUERY", "cache_dir": "/PRIVATE/cache"}}
            return scores
    reranker = CodeReranker("code-rerank-r1")
    kwargs = dict(adapters={"G": CountingEncoder(), "E": CountingEncoder()},
        rerankers={"bge_reranker_text": reranker}, requested_channels=["G", "E"],
        candidate_grid=(20, 50), budget_modes=("per_channel", "total"))
    first = run_matrix(root, tmp_path / "run", **kwargs)
    reranked = [row for row in first["cells"] if "fresh_adapter_diagnostics" in row]
    assert sum(row["new_score_pairs"] for row in reranked) == 6
    assert sum(row["fresh_adapter_diagnostics"]["processing_counts_observed"].get("chunk_count", 0)
               for row in reranked) == 12
    assert sum(row["fresh_adapter_diagnostics"]["processing_counts_observed"].get("original_function_count", 0)
               for row in reranked) == 6
    initial = next(row for row in reranked if row["new_score_pairs"])
    assert initial["adapter_diagnostics"]["processing_counts_observed"]["covered_source_utf8_bytes"] == 200
    assert initial["adapter_diagnostics"]["processing_policies"][0]["aggregation"].startswith("maximum_actual_base")
    again = run_matrix(root, tmp_path / "run", **kwargs)
    repeated = next(row for row in again["cells"] if row.get("identity") == initial["identity"])
    assert repeated["adapter_diagnostics"] == initial["adapter_diagnostics"]
    assert repeated["fresh_adapter_diagnostics"]["total_rows"] == 0
    assert repeated["fresh_adapter_diagnostics"]["processing_counts_observed"] == {}
    assert "previous_invocations" in repeated["adapter_diagnostics"]["scope"]
    assert len(reranker.calls) == 2
    with sqlite3.connect(tmp_path / "run/progress.sqlite3") as db:
        payloads = [json.loads(row[0]) for row in db.execute("SELECT payload FROM results")]
    usage = next(row["usage"] for row in payloads if row.get("usage", {}).get("new_pairs"))
    assert usage["adapter_diagnostics"]["chunk_count"] == 6
    assert usage["adapter_diagnostics_scope"].startswith("fresh_score_pairs_only")
    assert "PRIVATE" not in json.dumps(usage)


def test_direct_code_processing_survives_cache_without_replaying_work_time(tmp_path):
    dataset = load_dataset(fixture_dataset(tmp_path))
    class Direct:
        identity = "segmented-code-r1"

        def rank(self, documents, queries, **kwargs):
            ranking = [{"id": row["id"], "score": 1.0} for row in documents]
            return [ranking.copy() for _ in queries], {"segmentation_seconds": 10,
                "original_function_count": 3, "chunk_count": 7, "segmented_function_count": 1,
                "source_utf8_bytes": 123, "covered_source_utf8_bytes": 123,
                "condition": "shared_segmentation_function_max_cosine", "segmentation_identity": "f" * 64,
                "protocol": {"function_score": "maximum_cosine_over_all_its_chunks",
                             "native_document_formats": "PRIVATE_PROMPT"}}
    _, first = base_rankings(dataset, "G", {"G": Direct()}, tmp_path / "cache")
    _, again = base_rankings(dataset, "G", {"G": Direct()}, tmp_path / "cache")
    assert first["segmentation_seconds"] == 10 and again["segmentation_seconds"] is None
    assert again["ranking_cache_hit"] is True
    assert again["chunk_count"] == 7 and again["source_utf8_bytes"] == again["covered_source_utf8_bytes"] == 123
    assert again["protocol"]["function_score"] == "maximum_cosine_over_all_its_chunks"
    assert "PRIVATE" not in json.dumps(first) and "PRIVATE" not in json.dumps(again)

import json

import pytest

from rag_benchmark.multimodal_qa import (
    QARecord,
    citation_source_set_metrics,
    evaluate_prediction,
    generation_query,
    lexical_metrics,
    normalize_lexical,
    prepare_qa,
    select_evidence,
    sha256,
)


def fixture(tmp_path, *, answer="Correct fact", raw_answer=None):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    root = tmp_path / "dataset"
    root.mkdir()
    rows = {
        "queries.jsonl": [
            {
                "id": "q",
                "text": "What fact?",
                "metadata": {"language": "english", "answer": answer, "answer_use": "evaluation_only"},
            }
        ],
        "corpus.jsonl": [
            {"id": "p", "text": "Source fact", "media": {"image": "media/p.jpg"}},
            {"id": "n", "text": "Other source", "media": {"image": "media/n.jpg"}},
        ],
        "qrels.jsonl": [{"query_id": "q", "corpus_id": "p", "relevance": 2}],
        "assets.jsonl": [{"path": "media/p.jpg", "sha256": "fixture", "size_bytes": 5}],
    }
    for name, values in rows.items():
        (root / name).write_text("\n".join(json.dumps(x) for x in values) + "\n")
    (root / "dataset.json").write_text(
        json.dumps(
            {
                "id": "qa-fixture",
                "track": "document",
                "metadata": {"language": "en"},
                "files": {name: {"sha256": sha256(root / name)} for name in rows},
            }
        )
    )
    raw = tmp_path / "raw.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "query_id": "q",
                    "query": "What fact?",
                    "language": "english",
                    "answer": answer if raw_answer is None else raw_answer,
                    "raw_answers": ["Not a certified alternate"],
                }
            ]
        ),
        raw,
    )
    manifest_path = root / "dataset.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["sources"] = [
        {"url": "https://fixture/resolve/frozen/queries/raw.parquet", "sha256": sha256(raw)}
    ]
    manifest_path.write_text(json.dumps(manifest))
    return root, raw, rows


def test_source_mapping_and_public_safe_readiness(tmp_path):
    root, raw, _ = fixture(tmp_path)
    records, audit = prepare_qa(root, [raw])
    assert audit["status"] == "ready_for_generation" and audit["eligible_query_count"] == 1
    assert not audit["generation_executed"] and not audit["quality_measured"]
    assert records[0].references == ("Correct fact",)
    assert "references" not in generation_query(records[0])
    serialized = json.dumps(audit)
    for private in [
        "Correct fact",
        "What fact?",
        "Source fact",
        "Not a certified alternate",
        "query_id",
        "positive_source_ids",
    ]:
        assert private not in serialized


def test_source_answer_mismatch_and_tampered_prepared_file_rejected(tmp_path):
    root, raw, _ = fixture(tmp_path, raw_answer="Wrong source")
    with pytest.raises(ValueError, match="differs from published"):
        prepare_qa(root, [raw])
    (root / "queries.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        prepare_qa(root, [raw])


def test_missing_gold_is_unavailable_not_query_as_answer(tmp_path):
    root, raw, _ = fixture(tmp_path, answer="")
    records, audit = prepare_qa(root, [raw])
    assert records == [] and audit["status"] == "unavailable"
    assert audit["excluded_reasons"] == {"missing_usable_published_answer": 1}
    assert prepare_qa(root, [])[1]["status"] == "unavailable"


def test_unicode_language_and_multigold_policy():
    assert normalize_lexical("İSTANBUL, IĞDIR!", "turkish") == "istanbul ığdır"
    assert lexical_metrics("ISTANBUL", "istanbul".split(), "english")["answer_em"] == 1
    assert lexical_metrics("Cafe\u0301", ["CAFÉ"], "french")["answer_em"] == 1
    scores = lexical_metrics("red red bird", ["red bird", "red red bird"], "english")
    assert scores["answer_em"] == scores["answer_token_f1"] == 1
    assert lexical_metrics("red red bird", ["red bird"], "english")["answer_token_f1"] == pytest.approx(0.8)
    assert lexical_metrics("", ["answer"], "english") == {
        "answer_em": 0.0,
        "answer_token_f1": 0.0,
        "nonanswer": True,
    }
    for references in [[], [""], ["!!!"]]:
        with pytest.raises(ValueError):
            lexical_metrics("answer", references, "english")
    with pytest.raises(ValueError):
        lexical_metrics(None, ["answer"], "english")


def test_citations_are_deduplicated_source_sets_not_entailment():
    score = citation_source_set_metrics(["p", "p", "wrong"], ["p", "other"])
    assert score["citation_source_set_precision"] == 0.5
    assert score["citation_source_set_recall"] == 0.5
    assert score["citation_entailment_accuracy"] is None
    assert citation_source_set_metrics([], ["p"])["citation_source_set_precision"] == 0
    with pytest.raises(ValueError):
        citation_source_set_metrics(["p"], [])


def test_evidence_conditions_exclude_gold_and_keep_actual_retrieval_order():
    record = QARecord("q", "Question", "english", ("Secret gold",), ("p",))
    corpus = [
        {"id": "p", "text": "Source", "media": {"image": "p.jpg"}, "metadata": {"answer": "Secret gold"}},
        {"id": "n", "text": "Negative", "media": {"image": "n.jpg"}},
    ]
    assert select_evidence(record, corpus, condition="closed_book", representation="image") == []
    assert select_evidence(record, corpus, condition="oracle", representation="text") == [
        {"id": "p", "text": "Source"}
    ]
    retrieved = select_evidence(
        record, corpus, condition="retrieved", representation="image", retrieved_ids=["n", "p"]
    )
    assert retrieved == [{"id": "n", "media": {"image": "n.jpg"}}, {"id": "p", "media": {"image": "p.jpg"}}]
    with pytest.raises(ValueError):
        select_evidence(
            record, corpus, condition="retrieved", representation="text", retrieved_ids=["p", "p"]
        )
    with pytest.raises(ValueError):
        select_evidence(
            record, corpus, condition="retrieved", representation="text", retrieved_ids=["absent"]
        )
    assert (
        evaluate_prediction(record, answer="Secret gold", cited_source_ids=["p"])["human_semantic_accuracy"]
        is None
    )


def refresh_prepared_hash(root, filename):
    path = root / "dataset.json"
    manifest = json.loads(path.read_text())
    manifest["files"][filename]["sha256"] = sha256(root / filename)
    path.write_text(json.dumps(manifest))


def test_arbitrary_raw_source_not_frozen_manifest_bound(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    root, raw, _ = fixture(tmp_path)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "query_id": "q",
                    "query": "What fact?",
                    "language": "english",
                    "answer": "Correct fact",
                    "raw_answers": ["changed"],
                }
            ]
        ),
        raw,
    )
    with pytest.raises(ValueError, match="not bound"):
        prepare_qa(root, [raw])


def test_qrel_hash_and_unknown_mapping_rejected(tmp_path):
    root, raw, _ = fixture(tmp_path)
    (root / "qrels.jsonl").write_text(
        json.dumps({"query_id": "q", "corpus_id": "unknown", "relevance": 1}) + "\n"
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        prepare_qa(root, [raw])
    refresh_prepared_hash(root, "qrels.jsonl")
    with pytest.raises(ValueError, match="absent frozen records"):
        prepare_qa(root, [raw])


def test_gold_not_evaluation_only_rejected(tmp_path):
    root, raw, rows = fixture(tmp_path)
    rows["queries.jsonl"][0]["metadata"]["answer_use"] = "generation"
    (root / "queries.jsonl").write_text(json.dumps(rows["queries.jsonl"][0]) + "\n")
    refresh_prepared_hash(root, "queries.jsonl")
    with pytest.raises(ValueError, match="restricted to evaluation"):
        prepare_qa(root, [raw])


def test_selected_empty_ocr_missing_image_and_unknown_image_fail(tmp_path):
    record = QARecord("q", "Question", "english", ("Gold",), ("p",))
    corpus = [
        {
            "id": "p",
            "text": "   ",
            "media": {},
            "metadata": {"answer": "Gold", "ocr": "Should not silently substitute"},
        }
    ]
    with pytest.raises(ValueError, match="no usable text"):
        select_evidence(record, corpus, condition="oracle", representation="text")
    with pytest.raises(ValueError, match="no image"):
        select_evidence(record, corpus, condition="oracle", representation="image")
    corpus[0]["media"] = {"image": "missing.jpg"}
    with pytest.raises(ValueError, match="absent or outside"):
        select_evidence(record, corpus, condition="oracle", representation="image", dataset_root=tmp_path)
    corpus[0]["media"] = {"image": "../outside.jpg"}
    with pytest.raises(ValueError, match="absent or outside"):
        select_evidence(record, corpus, condition="oracle", representation="image", dataset_root=tmp_path)


def test_metadata_never_enters_generation_evidence_in_any_condition():
    record = QARecord("q", "Question", "english", ("Secret gold",), ("p",))
    corpus = [
        {
            "id": "p",
            "text": "Source",
            "media": {"image": "p.jpg"},
            "metadata": {"answer": "Secret gold", "prompt": "Leaked prompt"},
        }
    ]
    for condition in ["closed_book", "oracle", "retrieved"]:
        for representation in ["text", "image"]:
            result = select_evidence(
                record, corpus, condition=condition, representation=representation, retrieved_ids=["p"]
            )
            assert "Secret gold" not in json.dumps(result)
            assert "Leaked prompt" not in json.dumps(result)
            assert "metadata" not in json.dumps(result)

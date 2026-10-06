import pytest
from rag_benchmark.metrics import answer_metrics, normalize_answer, retrieval_metrics, strip_source_citations


def test_only_known_source_citations_are_removed():
    assert strip_source_citations("1453. [a:c1, b:c2]", ["a:c1", "b:c2"]) == "1453. "
    assert strip_source_citations("[unknown]", ["a:c1"]) == "[unknown]"


def test_multi_evidence_and_duplicate_results():
    m = retrieval_metrics(["a", "a", "x", "b"], ["a", "b"], ks=(1, 3))
    assert m["recall@1"] == 0.5
    assert m["hit@1"] == 1
    assert m["all_evidence@1"] == 0
    assert m["recall@3"] == 1
    assert m["ndcg@3"] < 1


def test_turkish_normalization_and_multiple_references():
    assert normalize_answer("İSTANBUL, IĞDIR!") == "istanbul ığdır"
    assert answer_metrics("İstanbul", ["Ankara", "istanbul"]) == {"answer_em": 1, "answer_token_f1": 1}
    assert answer_metrics("Osmanlı 1453 yılında", ["1453"])["answer_token_f1"] == 0.5
    assert answer_metrics("", ["1453"])["answer_em"] == 0


def test_missing_labels_are_errors():
    with pytest.raises(ValueError):
        retrieval_metrics([], [])
    with pytest.raises(ValueError):
        answer_metrics("", [])

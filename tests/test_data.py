"""Small synthetic fixtures only: no copyrighted benchmark text in tests."""
import hashlib
import json

import pytest

from rag_benchmark.data import (
    _download_release,
    _parse_markdown_article,
    _safe_path,
    load_dataset,
    normalize_release,
    split_questions,
)


def article(name, text, question=None, answer="Synthetic answer", gold=None):
    return {
        "article": {"id": name, "title": name, "url": f"https://tr.wikipedia.org/wiki/{name}", "lang": "tr"},
        "chunks": [{"id": "c0000", "content": text}],
        "questions": {"items": [{"question": question or f"Question about {name}?", "answer": answer,
                                  "related_chunk_ids": gold or ["c0000"], "category": "FACTUAL"}],
                      "total_questions": 1},
    }


def fixture_release(tmp_path, rows):
    release = tmp_path / "release"
    entries = []
    for index, row in enumerate(rows):
        name = f"formal_5k/dataset/json/{index}.json"
        target = release / name
        target.parent.mkdir(parents=True, exist_ok=True)
        body = json.dumps(row).encode()
        target.write_bytes(body)
        entries.append({"path": name, "size": len(body), "sha256": hashlib.sha256(body).hexdigest()})
    return release, {"revision": "a" * 40, "files": entries}


def markdown_fixture(tmp_path, body="Birinci bilgi. İkinci bilgi.", question=True):
    directory = tmp_path / "web"
    directory.mkdir()
    (directory / "article.md").write_text(
        "---\nid: example\ntitle: Örnek\nurl: file://example.md\nlang: en\n"
        f"stats:\n  char_count: {len(body)}\n  num_chunks: 2\noptions:\n  total_questions: 1\n---\n{body}",
        encoding="utf-8",
    )
    (directory / "chunks_index.md").write_text(
        "| ID | Section | Heading Path | Char Range | Preview |\n"
        f"| c0000 | one | nested | heading | 0-14 | preview | with pipe |\n"
        f"| c0001 | two | two | 14-{len(body)} | preview |\n",
        encoding="utf-8",
    )
    if question:
        (directory / "dataset.md").write_text(
            "| # | Question | Answer | Category | Related_Chunk_IDs |\n"
            "| 1 | Hangi bilgi? | Birinci bilgi. | FACTUAL | c0000 |\n", encoding="utf-8"
        )
    return directory / "article.md"


def test_markdown_preserves_published_character_ranges_with_pipes(tmp_path):
    path = markdown_fixture(tmp_path)
    data = _parse_markdown_article(path)
    assert [row["content"] for row in data["chunks"]] == ["Birinci bilgi.", " İkinci bilgi."]
    assert data["questions"]["items"][0]["related_chunk_ids"] == ["c0000"]
    assert data["article"]["lang"] == "en"  # preserve dubious source metadata for audit


def test_markdown_missing_questions_keeps_chunks(tmp_path):
    data = _parse_markdown_article(markdown_fixture(tmp_path, question=False))
    assert len(data["chunks"]) == 2
    assert data["questions"]["items"] == []
    assert data["import_issues"][0]["reason"] == "missing_question_file"


def test_markdown_bad_range_is_rejected_not_rechunked(tmp_path):
    path = markdown_fixture(tmp_path)
    index = path.with_name("chunks_index.md")
    index.write_text(index.read_text().replace("0-14", "0-999"))
    with pytest.raises(ValueError, match="out_of_bounds"):
        _parse_markdown_article(path)


def test_normalization_qualifies_ids_and_reports_invalid_gold(tmp_path):
    rows = [article("one", "One evidence"), article("two", "Two evidence"),
            article("bad", "Bad evidence", gold=["missing"])]
    release, index = fixture_release(tmp_path, rows)
    output = tmp_path / "prepared"
    manifest = normalize_release(release, index, output, dev_size=1)
    corpus, questions, loaded_manifest = load_dataset(output, "test")
    assert len({row["id"] for row in corpus}) == 3
    assert manifest["counts"]["valid_questions"] == 2
    assert manifest["rejection_counts"]["missing_gold_chunk"] == 1
    assert all(question["gold_ids"][0].startswith(question["article_id"] + "#") for question in questions)
    assert manifest["fingerprint"] == loaded_manifest["fingerprint"]


def test_group_split_connects_duplicate_questions_and_gold_evidence(tmp_path):
    rows = [article("a", "Evidence A", question="Same question?"),
            article("b", "Evidence B", question="Same question?"),
            article("c", "Evidence B", question="Different question?"),
            article("d", "Evidence D"), article("e", "Evidence E")]
    release, index = fixture_release(tmp_path, rows)
    output = tmp_path / "prepared"
    normalize_release(release, index, output, dev_size=2, seed=42)
    corpus, dev, _ = load_dataset(output, "dev")
    _, test, _ = load_dataset(output, "test")
    side = {question["article_id"]: split for split, values in [("dev", dev), ("test", test)] for question in values}
    assert side["wikipedia:a"] == side["wikipedia:b"] == side["wikipedia:c"]
    again = split_questions(corpus, dev + test, 2, 42)
    assert [row["id"] for row in dev] == [row["id"] for row in again[0]]


def test_group_split_keeps_all_questions_from_same_article(tmp_path):
    first = article("one", "One evidence")
    first["questions"]["items"].append({"question": "Second question", "answer": "answer",
                                        "related_chunk_ids": ["c0000"], "category": "INTERPRETATION"})
    release, index = fixture_release(tmp_path, [first, article("two", "Two evidence"), article("three", "Three")])
    output = tmp_path / "prepared"
    normalize_release(release, index, output, dev_size=1)
    _, dev, _ = load_dataset(output, "dev")
    _, test, _ = load_dataset(output, "test")
    assert {q["article_id"] for q in dev}.isdisjoint(q["article_id"] for q in test)


def test_repeated_import_is_byte_identical(tmp_path):
    release, index = fixture_release(tmp_path, [article(str(i), f"Evidence {i}") for i in range(8)])
    output = tmp_path / "prepared"
    first = normalize_release(release, index, output, dev_size=2, seed=42)
    second = normalize_release(release, index, output, dev_size=2, seed=42)
    assert first == second


def test_loader_detects_changed_prepared_data(tmp_path):
    release, index = fixture_release(tmp_path, [article("one", "One"), article("two", "Two")])
    output = tmp_path / "prepared"
    normalize_release(release, index, output, dev_size=1)
    with (output / "corpus.jsonl").open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_dataset(output)


def test_raw_cache_is_verified_and_offline(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    release, index = fixture_release(raw, [article("one", "One")])
    (raw / "release-files.json").write_text(json.dumps(index))
    def no_network(*args, **kwargs):
        raise AssertionError("cache hit must stay offline")
    monkeypatch.setattr("urllib.request.urlopen", no_network)
    assert _download_release(raw, index["revision"])[0] == release
    next(release.rglob("*.json")).write_text("corrupted")
    with pytest.raises(ValueError, match="missing or changed"):
        _download_release(raw, index["revision"])


@pytest.mark.parametrize("path", ["../escape", "/absolute", "ok/../../escape", "dir\\escape"])
def test_untrusted_snapshot_path_cannot_escape(path):
    with pytest.raises(ValueError, match="Unsafe"):
        _safe_path(path)

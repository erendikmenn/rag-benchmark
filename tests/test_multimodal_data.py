import json
from pathlib import Path

import pytest

from rag_benchmark.multimodal_data import (
    _extract_tar,
    _relative,
    normalize_transcript,
    select_codesearchnet_ids,
    strip_python_docstrings,
    strip_source_comments,
    validate_dataset,
    write_dataset,
)


def make_dataset(tmp_path: Path, **overrides):
    asset = tmp_path / "media" / "image.png"
    asset.parent.mkdir()
    asset.write_bytes(b"example media bytes")
    arguments = dict(dataset_id="example", track="photo", revision="frozen-sha", corpus=[
        {"id": "image", "media": {"image": "media/image.png"}}
    ], queries=[{"id": "q", "text": "reference caption", "metadata": {"group_id": "image"}}],
        qrels=[{"query_id": "q", "corpus_id": "image", "relevance": 1.0}], sources=[],
        license="test", text_source="absent_requires_generation", expected_counts={"corpus": 1, "queries": 1})
    arguments.update(overrides)
    return write_dataset(tmp_path, **arguments)


def test_roundtrip_manifest_and_media_integrity(tmp_path):
    manifest = make_dataset(tmp_path)
    assert manifest["counts"] == {"corpus": 1, "queries": 1, "qrels": 1, "media_files": 1}
    assert manifest["text_ready"] is False
    assert validate_dataset(tmp_path) == manifest
    (tmp_path / "media/image.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="modified media"):
        validate_dataset(tmp_path)


def test_reject_missing_gold_positive(tmp_path):
    with pytest.raises(ValueError, match="Invalid or duplicate qrel"):
        make_dataset(tmp_path, qrels=[{"query_id": "q", "corpus_id": "absent", "relevance": 1.0}])
    assert json.loads((tmp_path / "dataset.json").read_text())["status"] == "failed"


def test_reject_duplicate_ids(tmp_path):
    with pytest.raises(ValueError, match="duplicate corpus IDs"):
        make_dataset(tmp_path, corpus=[{"id": "same"}, {"id": "same"}], expected_counts={})


def test_reject_unanswered_query(tmp_path):
    with pytest.raises(ValueError, match="at least one positive"):
        make_dataset(tmp_path, qrels=[])


def test_no_gold_candidate_provenance(tmp_path):
    with pytest.raises(ValueError, match="leakage-safe provenance"):
        make_dataset(tmp_path, corpus=[{"id": "image", "text": "caption", "metadata": {
            "text_provenance": {"gold_fields_used": True, "query_independent": False}}}])


def test_gold_captions_are_query_only(tmp_path):
    make_dataset(tmp_path)
    assert "reference caption" not in (tmp_path / "corpus.jsonl").read_text()
    assert "reference caption" in (tmp_path / "queries.jsonl").read_text()


def test_no_silent_reduced_gallery(tmp_path):
    with pytest.raises(ValueError, match="Full dataset count mismatch"):
        make_dataset(tmp_path, expected_counts={"corpus": 2})


def test_manifest_catches_changed_queries(tmp_path):
    make_dataset(tmp_path)
    (tmp_path / "queries.jsonl").write_text('{"id":"q","text":"changed"}\n')
    with pytest.raises(ValueError, match="checksum mismatch: queries"):
        validate_dataset(tmp_path)


def test_docstrings_comments_removed_but_literal_code_kept():
    source = '''def search_name(value):
    """Gold description of a function."""
    # another description
    marker = "must stay"
    return value + marker
'''
    cleaned = strip_python_docstrings(source)
    assert "Gold description" not in cleaned
    assert "another description" not in cleaned
    assert '"must stay"' in cleaned
    assert "search_name" in cleaned


def test_python_two_docstrings_removed():
    source = '''def old(value):
    """secret gold caption"""
    print value
    return "important literal"
'''
    cleaned = strip_python_docstrings(source)
    assert "secret gold caption" not in cleaned
    assert "important literal" in cleaned


def test_unicode_docstring_offsets():
    source = 'def f():\n    """Türkçe açıklama"""; return "işlev"\n'
    cleaned = strip_python_docstrings(source)
    assert "açıklama" not in cleaned
    assert "işlev" in cleaned


def test_path_traversal_rejected():
    for path in ["../escape", "/absolute", "a\\..\\b"]:
        with pytest.raises(ValueError, match="Unsafe"):
            _relative(path)


def test_tar_symlink_rejected(tmp_path):
    import tarfile
    archive = tmp_path / "bad.tar"
    with tarfile.open(archive, "w") as handle:
        member = tarfile.TarInfo("link")
        member.type = tarfile.SYMTYPE
        member.linkname = "../escape"
        handle.addfile(member)
    with pytest.raises(ValueError, match="Unsupported archive member"):
        _extract_tar(archive, tmp_path / "out")


def test_transcript_grouping_is_normalized():
    assert normalize_transcript("  SAME\t sentence  ") == normalize_transcript("same sentence")


def test_official_codebase_intersection_preserves_published_gallery():
    selected, audit = select_codesearchnet_ids(["a", "absent", "b"], ["a"], {"a", "b"}, 2)
    assert selected == ["a", "b"]
    assert audit["upstream_urls_absent_from_raw_validation_test"] == 1
    with pytest.raises(ValueError, match="cleaned gallery mismatch"):
        select_codesearchnet_ids(["a", "absent", "b"], ["a"], {"a"}, 2)
    with pytest.raises(ValueError, match="test queries"):
        select_codesearchnet_ids(["a", "b"], ["missing-query"], {"a", "b"}, 2)


def test_nonleading_python_two_string_comment_removed_after_dedent():
    source = "def old(x):\n    if x:\n        print x\n    \"\"\"gold description\"\"\"\n    return 'preserved literal'\n"
    cleaned = strip_python_docstrings(source)
    assert "gold description" not in cleaned
    assert "preserved literal" in cleaned


@pytest.mark.parametrize("language,source,literal", [
    ("python", 'def f():\n    """gold comment"""\n    return "retained // # string"\n', "retained // # string"),
    ("ruby", 'def f()\n # gold comment\n "retained # string"\nend', "retained # string"),
    ("go", 'func f() string { /* gold comment */ return "retained // string" }', "retained // string"),
    ("java", 'String f() { /* gold comment */ return "retained // string"; }', "retained // string"),
    ("javascript", 'function f() { // gold comment\n return "retained // string"; }', "retained // string"),
    ("php", 'function f() { /* gold comment */ return "retained // string"; }', "retained // string"),
])
def test_syntax_aware_comment_cleanup(language, source, literal):
    pytest.importorskip("tree_sitter_language_pack")
    cleaned = strip_source_comments(source, language)
    assert "gold comment" not in cleaned
    assert literal in cleaned
    assert "f()" in cleaned


def test_runtime_string_can_legitimately_repeat_docstring_without_label_leakage():
    pytest.importorskip("tree_sitter_language_pack")
    source = 'def f():\n    """Build a model"""\n    print("Build a model for these samples")\n'
    result = strip_source_comments(source, "python")
    assert '"""Build a model"""' not in result
    assert 'print("Build a model for these samples")' in result

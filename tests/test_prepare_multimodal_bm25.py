"""Index-only synthetic CPU checks; query/search functions fail if called."""
import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import bm25s
import numpy as np
import pytest

PATH = Path(__file__).parents[1] / "scripts/prepare_multimodal_bm25.py"
spec = importlib.util.spec_from_file_location("bm25_preparation", PATH)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    root = tmp_path / "data"
    root.mkdir()
    corpus = [{"id": "a", "text": "fixture bilgi bir", "title": "TITLE_MUST_NOT_BE_TOKENIZED"},
              {"id": "b", "text": "fixture ikinci bilgi", "title": ""}]
    (root / "dataset.json").write_text('{}')
    (root / "corpus.jsonl").write_text('\n'.join(json.dumps(v) for v in corpus))
    ds = SimpleNamespace(root=root, corpus=corpus, track="document", identity="a" * 64,
                         manifest={"revision": "b" * 40, "files": {"corpus.jsonl": {}}}, require_text=lambda: None)
    monkeypatch.setattr(m, "load_dataset", lambda _: ds)
    for name in ("retrieve", "get_scores", "get_scores_from_ids"):
        monkeypatch.setattr(bm25s.BM25, name, lambda *_a, **_k: pytest.fail("query/search forbidden"))
    return ds


def test_index_only_native_roundtrip_and_readonly_reuse(dataset, tmp_path):
    output = tmp_path / "index"
    result = m.prepare(dataset.root, output, dataset.identity)
    assert result["new_index_builds"] == 1 and result["retrieval_calls"] == 0
    manifest = json.loads((output / "prepared.json").read_text())
    assert len(manifest["files"]) == 7
    assert manifest["validation"]["loadback_verified"]
    assert json.loads((output / "documents.json").read_text())["document_lengths"] == [3, 3]
    assert "title_must_not_be_tokenized" not in json.loads((output / "vocab.index.json").read_text())
    before = {p.name: (m.sha(p), p.stat().st_mtime_ns) for p in output.iterdir()}
    reused = m.prepare(dataset.root, output, dataset.identity)
    assert reused["status"] == "verified_reuse" and reused["new_index_builds"] == 0
    assert before == {p.name: (m.sha(p), p.stat().st_mtime_ns) for p in output.iterdir()}
    assert all(p.stat().st_mode & 0o777 == 0o600 for p in output.iterdir())


def test_wrong_dataset_pin_before_any_output(dataset, tmp_path):
    with pytest.raises(ValueError, match="frozen_dataset_identity"):
        m.prepare(dataset.root, tmp_path / "index", "c" * 64)
    assert not (tmp_path / "index").exists()


def test_raw_media_missing_source_text_unavailable(dataset, tmp_path):
    def missing():
        raise ValueError("source_only_text_absent")
    dataset.require_text = missing
    with pytest.raises(ValueError, match="source_only_text_absent"):
        m.prepare(dataset.root, tmp_path / "index", dataset.identity)
    assert not (tmp_path / "index").exists()


def test_failed_save_retains_staging_and_blocks_retry(dataset, tmp_path, monkeypatch):
    monkeypatch.setattr(bm25s.BM25, "save", lambda *_a, **_k: (_ for _ in ()).throw(OSError("fixture interruption")))
    output = tmp_path / "index"
    with pytest.raises(OSError):
        m.prepare(dataset.root, output, dataset.identity)
    assert not output.exists()
    assert len(list(tmp_path.glob('.index.staging-*'))) == 1
    with pytest.raises(ValueError, match="interrupted_staging"):
        m.prepare(dataset.root, output, dataset.identity)


def test_completed_corrupt_file_not_overwritten(dataset, tmp_path):
    output = tmp_path / "index"
    m.prepare(dataset.root, output, dataset.identity)
    path = output / "vocab.index.json"
    path.write_text('{}')
    with pytest.raises(ValueError, match="existing_complete_payload_changed"):
        m.prepare(dataset.root, output, dataset.identity)
    assert path.read_text() == '{}'


def test_source_change_during_index_rejected(dataset, tmp_path, monkeypatch):
    original = bm25s.BM25.save
    def changed(self, path):
        original(self, path)
        (dataset.root / 'corpus.jsonl').write_text('changed source')
    monkeypatch.setattr(bm25s.BM25, 'save', changed)
    with pytest.raises(ValueError, match="source_changed"):
        m.prepare(dataset.root, tmp_path / 'index', dataset.identity)
    assert not (tmp_path / 'index').exists()


def test_code_exact_identifier_tokenizer(dataset, tmp_path):
    dataset.track = 'code'
    dataset.corpus[0]['text'] = 'fooBar snake_case test123'
    output = tmp_path / 'index'
    m.prepare(dataset.root, output, dataset.identity)
    lengths = json.loads((output / 'documents.json').read_text())['document_lengths']
    assert lengths[0] == len(m._code_tokens(dataset.corpus[0]['text']))
    assert json.loads((output / 'prepared.json').read_text())['pins']['tokenizer'] == 'code-identifiers-v1'


def test_out_of_range_nonempty_sentinel_rejected(dataset, tmp_path):
    output = tmp_path / 'index'
    m.prepare(dataset.root, output, dataset.identity)
    vocab = json.loads((output / 'vocab.index.json').read_text())
    if '' in vocab:
        vocab['not_empty'] = vocab.pop('')
    else:
        vocab['not_empty'] = len(vocab)
    (output / 'vocab.index.json').write_text(json.dumps(vocab))
    with pytest.raises(ValueError, match='sentinel'):
        m.verify_native(output, expected_ids=['a', 'b'], expected_lengths=[3, 3],
                        expected_corpus_hash=json.loads((output / 'ready.json').read_text())['corpus_hash'])


def test_int64_native_pointers_and_empty_sentinel(dataset, tmp_path):
    output = tmp_path / 'index'
    m.prepare(dataset.root, output, dataset.identity)
    ptr = np.load(output / 'indptr.csc.index.npy')
    assert ptr.dtype == np.int64
    vocab = json.loads((output / 'vocab.index.json').read_text())
    assert vocab[''] == len(ptr) - 1


def test_no_query_dispatch_in_producer_ast():
    tree = ast.parse(PATH.read_text())
    names = {node.func.id if isinstance(node.func, ast.Name) else node.func.attr for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, (ast.Name, ast.Attribute))}
    assert not names & {'_bm25_rankings', 'get_scores', 'get_scores_from_ids', 'retrieve', 'rank',
                        'run_matrix', 'base_rankings', 'encode', 'generate', 'predict', 'rerank'}

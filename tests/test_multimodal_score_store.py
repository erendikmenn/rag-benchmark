import sqlite3
from types import SimpleNamespace

import pytest

from rag_benchmark.multimodal import _RunStore, _rerank


class Scorer:
    identity = 'fake-pointwise-v1'
    pointwise = True

    def __init__(self):
        self.calls = []

    def score(self, query, candidates):
        self.calls.append([item['id'] for item in candidates])
        return [float(item['text']) for item in candidates]


def test_bulk_scores_use_one_transaction_and_resume_missing_pairs(tmp_path):
    path = tmp_path / 'progress.sqlite3'
    store = _RunStore(path)
    transactions = []
    store.db.set_trace_callback(transactions.append)
    corpus = {key: {'id': key, 'text': score} for key, score in [('a', '-1'), ('b', '-1'), ('c', '-2')]}
    dataset = SimpleNamespace(identity='frozen-dataset', model_item=lambda row, **kwargs: dict(row))
    scorer = Scorer()
    query = {'id': 'q', 'text': 'query'}
    ranked, usage = _rerank(dataset, query, ['b', 'a'], scorer, store, corpus)
    assert [row['id'] for row in ranked] == ['a', 'b']
    assert usage['new_pairs'] == 2 and usage['cache_pairs'] == 0
    assert sum(statement == 'COMMIT' for statement in transactions) == 1
    store.db.close()
    store = _RunStore(path)
    ranked, usage = _rerank(dataset, query, ['c', 'b', 'a'], scorer, store, corpus)
    assert scorer.calls == [['b', 'a'], ['c']]
    assert [row['id'] for row in ranked] == ['a', 'b', 'c']
    assert usage['new_pairs'] == 1 and usage['cache_pairs'] == 2
    ranked_again, usage = _rerank(dataset, query, ['a', 'b', 'c'], scorer, store, corpus)
    assert ranked_again == ranked and usage['new_pairs'] == 0
    assert usage['inference_seconds'] is None
    store.db.close()


def test_bulk_write_failure_rolls_back_all_new_scores(tmp_path):
    store = _RunStore(tmp_path / 'progress.sqlite3')
    store.put_score('existing', -7)
    store.db.execute("CREATE TRIGGER reject_score BEFORE INSERT ON scores WHEN NEW.key='bad' BEGIN SELECT RAISE(ABORT, 'rejected'); END")
    with pytest.raises(sqlite3.IntegrityError):
        store.put_scores([('good', -1), ('bad', -2)])
    assert store.score('good') is None and store.score('bad') is None
    assert store.score('existing') == -7
    with pytest.raises(TypeError):
        store.put_scores([('good', 1), ('bad', object())])
    assert store.score('good') is None
    store.db.close()

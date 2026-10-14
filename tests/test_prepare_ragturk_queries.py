import importlib.util
from pathlib import Path
import numpy as np
import pytest
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('query_prep_review', ROOT / 'scripts/prepare_ragturk_queries.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class Fake:
    name = 'embeddinggemma'
    dimension = 2

    def __init__(self):
        self.calls = []

    def format_query(self, question):
        return 'task: search result | query: ' + question

    def _encode(self, texts):
        self.calls.append(texts)
        return np.tile(np.array([[1, 0]], dtype=np.float32), (len(texts), 1))

def test_only_native_question_format_and_resume(tmp_path):
    rows = [{'id': str(i), 'question': 'q' + str(i), 'answers': ['SECRET'], 'gold_ids': ['GOLD']} for i in range(3)]
    f = Fake()
    result = m.encode_blocks(f, rows, tmp_path, 'frozen', block_size=2)
    assert f.calls == [['task: search result | query: q0', 'task: search result | query: q1'], ['task: search result | query: q2']]
    assert result['fresh_rows'] == 3
    f2 = Fake()
    result = m.encode_blocks(f2, rows, tmp_path, 'frozen', block_size=2)
    assert not f2.calls and result['reused_rows'] == 3
    with pytest.raises(ValueError, match='identity/checksum'):
        m.encode_blocks(Fake(), rows, tmp_path, 'changed', block_size=2)

def test_corrupt_and_partial_checkpoint_rejected(tmp_path):
    rows = [{'id': 'x', 'question': 'q'}]
    m.encode_blocks(Fake(), rows, tmp_path, 'frozen')
    (tmp_path / 'queries-000000.npy').write_bytes(b'changed')
    with pytest.raises(ValueError, match='identity/checksum'):
        m.encode_blocks(Fake(), rows, tmp_path, 'frozen')
    (tmp_path / 'queries-000000.json').unlink()
    with pytest.raises(ValueError, match='Incomplete'):
        m.encode_blocks(Fake(), rows, tmp_path, 'frozen')

def test_order_change_rejected(tmp_path):
    rows = [{'id': 'a', 'question': 'qa'}, {'id': 'b', 'question': 'qb'}]
    m.encode_blocks(Fake(), rows, tmp_path, 'frozen')
    with pytest.raises(ValueError, match='identity/checksum'):
        m.encode_blocks(Fake(), rows[::-1], tmp_path, 'frozen')

def test_owner_lock_is_exclusive(tmp_path):
    with m.exclusive_lock(tmp_path / 'lock'):
        with pytest.raises(BlockingIOError):
            with m.exclusive_lock(tmp_path / 'lock'):
                pass

def test_final_seal_failure_does_not_publish_complete(tmp_path):
    import json

    def fail():
        raise ValueError('changed source')
    with pytest.raises(ValueError, match='changed source'):
        m.encode_blocks(Fake(), [{'id': 'a', 'question': 'q'}], tmp_path, 'frozen', final_verify=fail)
    assert not (tmp_path / 'result.json').exists()
    assert json.loads((tmp_path / 'status.json').read_text())['status'] != 'complete'
    assert (tmp_path / 'queries.npy').exists()

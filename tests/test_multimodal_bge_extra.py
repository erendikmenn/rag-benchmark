"""Staged CPU-only formula/extraction fixtures; no tensor/model loading."""
import json

import numpy as np
import pytest

from rag_benchmark import multimodal_bge_extra as m


def test_sparse_official_relu_repeated_max_specials():
    h = np.array([[9], [2], [-3], [4], [10]], dtype=np.float32)
    assert m.lexical_weights(h, [0, 9, 8, 9, 2], [[1]], [0]) == {9: 4.0}
    assert m.sparse_score({9: 4, 8: 2}, {9: 3, 7: 9}) == 12


@pytest.mark.parametrize('bound', [1, 2, 5, 100])
def test_maxsim_exact_negative_zero_and_blocking(bound):
    q = np.array([[1., 0], [0, 1]])
    d = np.array([[-1., -2], [-3, -1]])
    assert m.maxsim_score(q, d, max_score_cells=bound) == -1
    assert m.maxsim_score(np.zeros((1, 2)), d, max_score_cells=bound) == 0
    assert m.maxsim_score(q, d, max_score_cells=bound) == float((q @ d.T).max(axis=1).mean())


def test_projection_retains_eos_and_zero():
    assert np.array_equal(m.projected_tokens([[99, 99], [0, 0], [0, 3]], np.eye(2), [0, 0]), [[0, 0], [0, 1]])


def test_full_source_max_ties_and_multiple_positives():
    chunks = [{'source_id': s, 'sparse': {7: v}} for s, v in [('b', 2), ('a', 1), ('a', 2), ('c', 0)]]
    ranked = m.rank_sources({'sparse': {7: 1}}, chunks, mode='sparse')
    assert [r['id'] for r in ranked] == ['a', 'b', 'c']
    positives = {'a', 'b'}
    assert len(set(r['id'] for r in ranked[:2]) & positives) / len(positives) == 1


class Fake:
    identity = 'cpu-fake'
    calls = 0
    def tokenize(self, text):
        return [0, 10, 2]
    def forward(self, tokens):
        self.calls += 1
        hidden = np.zeros((len(tokens), 3, m.DIMENSION), dtype=np.float32)
        hidden[:, :, 0] = 1
        return {'hidden': hidden, 'token_ids': tokens, 'attention_mask': np.ones((len(tokens), 3), dtype=int)}


def fixture():
    backend = Fake()
    heads = {'sparse': (np.ones((1, m.DIMENSION)), np.zeros(1)), 'colbert': (np.eye(m.DIMENSION), np.zeros(m.DIMENSION))}
    config = {**m.MODEL_DEFAULTS['bge'], 'max_length': 8192, 'token_vector_storage': 'float32'}
    extractor = m.ExtraHeadExtractor(backend=backend, config=config, head_weights=heads, source_identity='frozen-source')
    rows = [{'id': 'row', 'source_id': 'source', 'formatted_text': 'fixture'}]
    dense = np.zeros((1, m.DIMENSION), dtype=np.float32)
    dense[:, 0] = 1
    return extractor, backend, rows, dense


def test_one_forward_both_heads_resume_and_corruption(tmp_path):
    e, b, rows, dense = fixture()
    output, usage = e.extract(rows, dense, cache_dir=tmp_path)
    assert b.calls == usage['hidden_forward_calls'] == 1
    assert output[0]['sparse'] == {10: 1.0}
    assert output[0]['colbert'].shape == (2, m.DIMENSION)
    _, usage = e.extract(rows, dense, cache_dir=tmp_path)
    assert b.calls == 1 and usage['cached_rows'] == 1
    path = next(tmp_path.glob('*/manifest.json'))
    manifest = json.loads(path.read_text())
    manifest['rows'][0]['sparse']['10'] = 999
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='checksum'):
        e.extract(rows, dense, cache_dir=tmp_path)
    assert b.calls == 1


def test_baseline_mismatch_is_durable_no_implicit_retry(tmp_path):
    e, b, rows, dense = fixture()
    dense[:, 0] = -1
    with pytest.raises(ValueError, match='CLS'):
        e.extract(rows, dense, cache_dir=tmp_path)
    with pytest.raises(ValueError, match='Incomplete'):
        e.extract(rows, dense, cache_dir=tmp_path)
    assert b.calls == 1


def test_full_token_limit_before_forward(tmp_path):
    e, b, rows, dense = fixture()
    b.tokenize = lambda text: [0] + [10] * 8191 + [2]
    with pytest.raises(ValueError, match='limit'):
        e.extract(rows, dense, cache_dir=tmp_path)
    assert b.calls == 0


def test_actual_token_drop_rejected(tmp_path):
    e, b, rows, dense = fixture()
    original = b.forward
    def bad(tokens):
        result = original(tokens)
        result['attention_mask'][0, 1] = 0
        return result
    b.forward = bad
    with pytest.raises(ValueError, match='tokens/masks'):
        e.extract(rows, dense, cache_dir=tmp_path)


def test_trained_heads_missing_fail_before_tensor_load(tmp_path):
    with pytest.raises(ValueError, match='random initialization forbidden'):
        m.load_trained_heads(tmp_path)


def test_backend_dense_override_cannot_hide_wrong_cls(tmp_path):
    e, b, rows, dense = fixture()
    original = b.forward
    def bad(tokens):
        result = original(tokens)
        result['hidden'][:, 0, 0] = -1
        result['native_dense'] = dense.copy()
        return result
    b.forward = bad
    with pytest.raises(ValueError, match='override'):
        e.extract(rows, dense, cache_dir=tmp_path)
    assert not list(tmp_path.glob('*/manifest.json'))


def test_left_padding_cannot_compare_masked_pad_cls(tmp_path):
    e, b, rows, dense = fixture()
    original = b.forward
    def bad(tokens):
        result = original(tokens)
        result['hidden'] = np.pad(result['hidden'], ((0, 0), (1, 0), (0, 0)))
        result['hidden'][0, 0, 0] = 1
        result['token_ids'] = [[1, 0, 10, 2]]
        result['attention_mask'] = [[0, 1, 1, 1]]
        return result
    b.forward = bad
    with pytest.raises(ValueError, match='right-padding'):
        e.extract(rows, dense, cache_dir=tmp_path)


def test_baseline_mutation_during_forward_rejected(tmp_path):
    e, b, rows, dense = fixture()
    original = b.forward
    def bad(tokens):
        result = original(tokens)
        dense[0, 0] = -1
        return result
    b.forward = bad
    with pytest.raises(ValueError, match='baseline changed'):
        e.extract(rows, dense, cache_dir=tmp_path)
    assert not list(tmp_path.glob('*/manifest.json'))


@pytest.mark.parametrize('mutation', ['config', 'backend', 'heads', 'records', 'baseline'])
def test_tokenizer_mutation_rejected_on_cached_path(tmp_path, mutation):
    e, b, rows, dense = fixture()
    e.extract(rows, dense, cache_dir=tmp_path)
    def bad(text):
        if mutation == 'config':
            e.config['max_length'] = 1
        elif mutation == 'backend':
            b.identity = 'changed'
        elif mutation == 'heads':
            e.heads['sparse'][0][0, 0] = 2
        elif mutation == 'records':
            rows[0]['source_id'] = 'changed'
        else:
            dense[0, 0] = -1
        return [0, 10, 2]
    b.tokenize = bad
    with pytest.raises(ValueError, match='changed'):
        e.extract(rows, dense, cache_dir=tmp_path)
    assert b.calls == 1


@pytest.mark.parametrize('operation', ['sparse', 'projection', 'maxsim', 'normalization', 'sparse_score'])
def test_finite_extremes_never_return_nonfinite(operation):
    large = np.finfo(np.float32).max
    with np.errstate(over='ignore', invalid='ignore'), pytest.raises(ValueError):
        if operation == 'sparse':
            m.lexical_weights([[large]], [10], [[large]], [0])
        elif operation == 'projection':
            m.projected_tokens([[1], [large]], [[large]], [0])
        elif operation == 'maxsim':
            m.maxsim_score([[large]], [[large]])
        elif operation == 'normalization':
            m.normalize_tokens([[large, large]])
        else:
            m.sparse_score({10: 1e308}, {10: 1e308})


def test_long_numpy_middle_token_changes_exact_cache_key(tmp_path):
    e, b, rows, dense = fixture()
    ids = np.array([0] + [10] * 1200 + [2], dtype=np.int64)
    b.tokenize = lambda text: ids.copy()
    received = []
    def forward(tokens):
        b.calls += 1
        assert isinstance(tokens[0], list) and all(type(t) is int for t in tokens[0])
        received.append(tokens[0].copy())
        hidden = np.zeros((1, len(tokens[0]), m.DIMENSION), dtype=np.float32)
        hidden[:, :, 0] = 1
        return {'hidden': hidden, 'token_ids': tokens, 'attention_mask': np.ones((1, len(tokens[0])), dtype=int)}
    b.forward = forward
    e.extract(rows, dense, cache_dir=tmp_path)
    ids[600] = 11
    e.extract(rows, dense, cache_dir=tmp_path)
    assert b.calls == 2 and len(list(tmp_path.glob('*/manifest.json'))) == 2
    assert received[0][600] == 10 and received[1][600] == 11


def test_long_numpy_forward_middle_mutation_rejected(tmp_path):
    e, b, rows, dense = fixture()
    b.tokenize = lambda text: np.array([0] + [10] * 1200 + [2], dtype=np.int64)
    def forward(tokens):
        b.calls += 1
        tokens[0][600] = 11
        hidden = np.zeros((1, len(tokens[0]), m.DIMENSION), dtype=np.float32)
        hidden[:, :, 0] = 1
        return {'hidden': hidden, 'token_ids': tokens, 'attention_mask': np.ones((1, len(tokens[0])), dtype=int)}
    b.forward = forward
    with pytest.raises(ValueError, match='input changed'):
        e.extract(rows, dense, cache_dir=tmp_path)
    assert b.calls == 1 and not list(tmp_path.glob('*/manifest.json'))

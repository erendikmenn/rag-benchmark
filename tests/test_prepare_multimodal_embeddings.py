import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


PATH = Path(__file__).parents[1] / 'scripts/prepare_multimodal_embeddings.py'
spec = importlib.util.spec_from_file_location('prepare_multimodal_embeddings', PATH)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def fixture(tmp_path):
    from rag_benchmark import models, multimodal
    source = tmp_path / 'source'
    source.write_text('frozen')
    dataset = SimpleNamespace(identity='dataset', track='document', manifest={'revision': 'rev'},
                              corpus=[{'id': 'd1'}], queries=[{'id': 'q1'}],
                              require_text=lambda: None, require_candidate_text=lambda: None)
    adapter = SimpleNamespace(identity='adapter', dimension=2, unload=lambda: None)
    config = {'version': 1, 'protocol': 'source-embeddings-only-v1', 'channel': 'E',
              'dataset_identity': 'dataset', 'adapter_identity': 'adapter',
              'roles': ['document', 'query'], 'block_size': 128, 'config': {},
              'dataset_root': str(tmp_path), 'cache_dir': str(tmp_path / 'cache'),
              'source_sha256': {str(p): m.sha256(p) for p in
                               (source, PATH, Path(models.__file__), Path(multimodal.__file__))}}
    return config, dataset, adapter


@pytest.mark.parametrize('field,value', [('dataset_identity', 'other'), ('adapter_identity', 'other'),
                                        ('roles', ['query']), ('block_size', 32), ('version', 2)])
def test_identity_rejections(tmp_path, field, value):
    config, dataset, adapter = fixture(tmp_path)
    config[field] = value
    with pytest.raises(ValueError):
        m.validate_spec(config, dataset, adapter)


def test_code_never_silently_unsegments(tmp_path):
    config, dataset, adapter = fixture(tmp_path)
    dataset.track = 'code'
    with pytest.raises(ValueError, match='segmentation'):
        m.validate_spec(config, dataset, adapter)


def test_source_and_cached_payload_mutation_rejected(tmp_path):
    config, _, _ = fixture(tmp_path)
    m.verify_seals(config)
    (tmp_path / 'source').write_text('changed')
    with pytest.raises(ValueError, match='seal'):
        m.verify_seals(config)
    config['source_sha256'] = {str(tmp_path / 'source'): m.sha256(tmp_path / 'source')}
    config['existing_cache_sha256'] = {str(tmp_path / 'source'): '0' * 64}
    with pytest.raises(ValueError, match='payload'):
        m.verify_seals(config)


def test_interrupted_output_is_not_retried(tmp_path, monkeypatch):
    config, dataset, adapter = fixture(tmp_path)
    monkeypatch.setattr(m, 'load_dataset', lambda _: dataset)
    monkeypatch.setattr(m, 'make_adapter', lambda *_: adapter)
    monkeypatch.setattr(m, 'cached_encode', lambda *_a, **_k: pytest.fail('must not encode'))
    output = tmp_path / 'out'
    m.atomic_json(output / 'status.json', {'status': 'encoding'})
    with pytest.raises(ValueError, match='Interrupted'):
        m.execute(config, output)


def test_empty_implementation_pin_set_rejected(tmp_path):
    config, dataset, adapter = fixture(tmp_path)
    config['source_sha256'] = {}
    with pytest.raises(ValueError, match='implementations'):
        m.validate_spec(config, dataset, adapter)


def test_unreviewed_cache_block_rejected(tmp_path):
    config, dataset, adapter = fixture(tmp_path)
    identity = m.stable_hash({'encoding': m.encoding_cache_identity(dataset, adapter, 'document'),
                              'text_only': True})
    directory = Path(config['cache_dir']) / 'vectors' / identity
    directory.mkdir(parents=True)
    np.save(directory / '000000000-000000001.npy', np.array([[1, 0]], dtype=np.float32))
    with pytest.raises(ValueError, match='inventory'):
        m.verify_initial_inventory(config, dataset, adapter)


def test_complete_roles_and_readonly_idempotence(tmp_path, monkeypatch):
    config, dataset, adapter = fixture(tmp_path)
    monkeypatch.setattr(m, 'load_dataset', lambda _: dataset)
    monkeypatch.setattr(m, 'make_adapter', lambda *_: adapter)
    calls = []
    def encode(ds, ad, role, cache, **kwargs):
        assert ds is dataset and ad is adapter and kwargs == {'block_size': 128, 'text_only': True}
        calls.append(role)
        identity = m.stable_hash({'encoding': m.encoding_cache_identity(ds, ad, role), 'text_only': True})
        directory = cache / 'vectors' / identity
        directory.mkdir(parents=True)
        np.save(directory / '000000000-000000001.npy', np.array([[1, 0]], dtype=np.float32))
        return None, {'new_rows': 1, 'cache_rows': 0, 'role': role}
    monkeypatch.setattr(m, 'cached_encode', encode)
    output = tmp_path / 'out'
    first = m.execute(config, output)
    assert calls == ['document', 'query']
    assert [x['blocks'][0]['item_ids'] for x in first['sets']] == [['d1'], ['q1']]
    assert first['benchmark_runs'] == first['reranker_calls'] == first['answer_generation_calls'] == 0
    assert m.execute(config, output) == first
    assert calls == ['document', 'query']
    Path(first['sets'][0]['blocks'][0]['path']).unlink()
    with pytest.raises(ValueError, match='incomplete'):
        m.execute(config, output)
    assert calls == ['document', 'query']


def test_producer_has_no_benchmark_dispatch():
    import ast
    tree = ast.parse(PATH.read_text())
    forbidden = {'run_matrix', 'base_rankings', '_bm25_rankings', 'rank', 'retrieve', 'get_scores'}
    calls = {node.func.id if isinstance(node.func, ast.Name) else node.func.attr
             for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, (ast.Name, ast.Attribute))}
    assert not forbidden & calls

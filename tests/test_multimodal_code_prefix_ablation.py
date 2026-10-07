import json

import pytest

from rag_benchmark import multimodal_code_prefix_ablation as module
from rag_benchmark.models import DenseEmbedder


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    path = tmp_path / 'config_sentence_transformers.json'
    path.write_text(json.dumps({'prompts': {'CodeRetrieval': module.QUERY_PREFIX,
                                         'Document': module.DOCUMENT_PREFIX}}))
    monkeypatch.setattr(module, '_snapshot', lambda *args, **kwargs: tmp_path)
    return path


def test_exact_prefix_and_baseline_identity(evidence):
    baseline = DenseEmbedder('embeddinggemma', {'device': 'cpu', 'dtype': 'float32'})
    before = baseline.identity
    ablation = module.CodePrefixEmbedder({'device': 'cpu', 'dtype': 'float32'})
    assert ablation.format_query('q') == 'task: code retrieval | query: q'
    assert ablation.format_document({'text': 'code'}) == 'title: none | text: code'
    assert ablation.identity != before
    assert DenseEmbedder('embeddinggemma', {'device': 'cpu', 'dtype': 'float32'}).identity == before
    assert ablation._model is None
    with pytest.raises(ValueError, match='untitled'):
        ablation.format_document({'text': 'code', 'title': 'title'})


def test_unverified_contract_rejected(evidence):
    evidence.write_text(json.dumps({'prompts': {'CodeRetrieval': 'invented'}}))
    with pytest.raises(ValueError, match='verify'):
        module.CodePrefixEmbedder()


def test_evidence_hash_bound(evidence):
    first = module.CodePrefixEmbedder()
    payload = json.loads(evidence.read_text())
    payload['evidence_version'] = 2
    evidence.write_text(json.dumps(payload))
    assert module.CodePrefixEmbedder().identity != first.identity


def test_mutation_rejected_before_model_load(evidence):
    model = module.CodePrefixEmbedder()
    model.config['max_length'] = 4
    with pytest.raises(ValueError, match='configuration changed'):
        model._encode(['q'])
    assert model._model is None


def test_javascript_shared_protocol_and_independent_cache(evidence):
    model = module.CodePrefixSegmentedAdapter({'device': 'cpu', 'dtype': 'float32'})
    ordinary = module.SegmentedCodeAdapter('embeddinggemma', {'device': 'cpu', 'dtype': 'float32'})
    assert model.segmenter.identity == ordinary.segmenter.identity
    assert model.segmenter.protocol == ordinary.segmenter.protocol
    assert model.identity != ordinary.identity
    assert model.encoder.format_query('q') == module.QUERY_PREFIX + 'q'
    assert model.encoder._model is None


def test_cli_javascript_uses_manifest_not_directory(evidence, tmp_path, monkeypatch):
    from types import SimpleNamespace
    dataset = SimpleNamespace(track='code', identity='frozen', queries=[{'text': 'q'}], corpus=[{'text': 'code'}],
                              manifest={'metadata': {'language': 'javascript'}})
    monkeypatch.setattr(module, 'load_dataset', lambda path: dataset)
    calls = []
    monkeypatch.setattr(module.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(returncode=0))
    def fake_matrix(*args, **kwargs):
        calls.append(kwargs)
        return {'cells': [{'variant_id': 'code__e__none', 'status': 'completed',
                           'query_count': 1, 'completed_queries': 1}]}
    monkeypatch.setattr(module, 'run_matrix', fake_matrix, raising=False)
    assert module.main(['--dataset', str(tmp_path / 'renamed'), '--run-dir',
                        str(tmp_path / 'ablations' / 'code'), '--device', 'cpu',
                        '--dtype', 'float32']) == 1
    assert calls == []
    readiness = json.loads((tmp_path / 'ablations' / 'code' / 'ablation-readiness.json').read_text())
    assert readiness['reason'] == 'verified_baseline_run_and_cache_required'


@pytest.mark.parametrize('status,completed', [('failed', 0), ('unsupported', 0), ('completed', 0)])
def test_cli_incomplete_nonzero_and_protocol(evidence, tmp_path, monkeypatch, status, completed):
    from types import SimpleNamespace
    dataset = SimpleNamespace(track='code', identity='frozen', queries=[{}], corpus=[{'text': 'code'}],
                              manifest={'metadata': {'language': 'python'}})
    monkeypatch.setattr(module, 'load_dataset', lambda path: dataset)
    monkeypatch.setattr(module.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(returncode=0))
    monkeypatch.setattr(module, 'run_matrix', lambda *args, **kwargs: {'cells': [
        {'variant_id': 'code__e__none', 'status': status, 'query_count': 1, 'completed_queries': completed}]}, raising=False)
    encoder = module.CodePrefixEmbedder()
    monkeypatch.setattr(module, 'VerifiedDocumentBridge', lambda *args, **kwargs: SimpleNamespace(
        encoder=encoder, identity=encoder.identity, provenance={'precomputed_documents': True}))
    run = tmp_path / 'ablations' / 'code'
    assert module.main(['--dataset', str(tmp_path), '--run-dir', str(run),
                        '--baseline-run', str(tmp_path / 'base'), '--baseline-cache', str(tmp_path / 'cache')]) == 1
    protocol = json.loads((run / 'ablation-protocol.json').read_text())
    assert protocol['ordinary_baseline'] is False
    assert protocol['main_denominator_contribution'] == 0
    assert protocol['prompt_contract']['query_prefix'] == module.QUERY_PREFIX
    assert protocol['encoder_config']['revision'] == module.MODEL_DEFAULTS['embeddinggemma']['revision']


def test_cli_rejects_public_raw_directory(evidence, tmp_path, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(module, 'load_dataset', lambda path: SimpleNamespace(track='code'))
    monkeypatch.setattr(module.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(returncode=1))
    monkeypatch.setattr(module, 'run_matrix', lambda *args, **kwargs: pytest.fail('must not execute'), raising=False)
    with pytest.raises(SystemExit) as error:
        module.main(['--dataset', str(tmp_path), '--run-dir', str(tmp_path / 'reports' / 'ablations')])
    assert error.value.code == 2


@pytest.mark.parametrize('config', [{'revision': 'a' * 40}, {'dimension': 256}, {'model_id': 'other/model'}])
def test_unapproved_model_contract_rejected(evidence, config):
    with pytest.raises(ValueError):
        module.CodePrefixEmbedder(config)


@pytest.fixture
def baseline_fixture(evidence, tmp_path):
    import numpy as np
    from types import SimpleNamespace
    dataset = SimpleNamespace(identity='frozen', corpus=[{'id': 'a', 'text': 'A'}, {'id': 'b', 'text': 'B'}],
        queries=[{'id': 'q', 'text': 'find'}], qrels={'q': {'a': 1}},
        manifest={'revision': 'frozen', 'metadata': {'language': 'python'}},
        model_item=lambda item: dict(item), excluded_ids=lambda query: set())
    ordinary = module.TextEmbeddingAdapter('embeddinggemma', {'device': 'cpu', 'dtype': 'float32'})
    vectors = np.zeros((2, 768), dtype=np.float32)
    vectors[0, 0] = 1
    vectors[1, 1] = 1
    query = vectors[:1]
    cache = tmp_path / 'cache'
    identity = module._channel_identity(dataset, 'E', {'E': ordinary})
    ranks = module._cosine_rankings(vectors, query, ['a', 'b'], None, [set()])
    rankpath = cache / 'rankings' / module.stable_hash({'identity': identity, 'top_k': 100}) / 'rankings.json'
    rankpath.parent.mkdir(parents=True)
    rankpath.write_text(json.dumps({'identity': identity, 'query_ids': ['q'], 'rankings': ranks}))
    encodings = {}
    for role, rows in [('document', vectors), ('query', query)]:
        key = module.stable_hash({'encoding': module.encoding_cache_identity(dataset, ordinary, role), 'text_only': True})
        directory = cache / 'vectors' / key
        directory.mkdir(parents=True)
        np.save(directory / f'000000000-{len(rows):09d}.npy', rows)
        encodings[role + '_encoding'] = {'identity': key}
    metrics = module.aggregate_metrics([module.graded_metrics(['a', 'b'], {'a': 1})])
    configuration = {'query_ids': ['q']}
    cell_identity = module.stable_hash({**configuration, 'variant': 'code__e__none', 'mode': 'total',
        'candidate_k': None, 'channel_identities': {'E': identity}, 'reranker': None, 'reranker_prompts': None})
    report = {'configuration': configuration, 'dataset': {'identity': 'frozen'}, 'scope': 'frozen_collection', 'evaluated_query_count': 1,
        'channels': {'E': {'identity': identity, 'status': 'completed', **encodings}},
        'adapter_specs': {'channels': {'E': {'identity': ordinary.identity, 'embedder': {'config': ordinary.embedder.config}}}},
        'cells': [{'variant_id': 'code__e__none', 'identity': cell_identity, 'budget_mode': 'total',
                   'candidate_k': None, 'status': 'completed', 'completed_queries': 1, 'metrics': metrics}]}
    run = tmp_path / 'baseline'
    run.mkdir()
    (run / 'report.json').write_text(json.dumps(report))
    return dataset, run, cache, vectors


def test_bridge_queries_only_and_resume(baseline_fixture, tmp_path, monkeypatch):
    import numpy as np
    dataset, run, cache, vectors = baseline_fixture
    bridge = module.VerifiedDocumentBridge(dataset, run, {'device': 'cpu', 'dtype': 'float32'}, baseline_cache=cache)
    calls = []
    monkeypatch.setattr(bridge.encoder, '_encode', lambda content: calls.append(content) or vectors[:1])
    monkeypatch.setattr(bridge.encoder, 'embed_documents', lambda items: pytest.fail('document inference forbidden'))
    ranks, usage = bridge.rank(dataset.corpus, dataset.queries, 100, tmp_path / 'new', [set()])
    assert ranks[0][0]['id'] == 'a'
    assert calls == [[module.QUERY_PREFIX + 'find']]
    assert usage['document_encoding']['new_items'] == 0
    assert bridge.provenance['precomputed_documents'] is True
    again, usage = bridge.rank(dataset.corpus, dataset.queries, 100, tmp_path / 'new', [set()])
    assert again == ranks and len(calls) == 1
    assert usage['query_encoding']['new_items'] == 0
    assert np.array_equal(bridge.documents, vectors)
    with pytest.raises(ValueError, match='forbidden'):
        bridge._cached_vectors(dataset.corpus, 'document', tmp_path)


def test_missing_baseline_never_infers(baseline_fixture, monkeypatch):
    dataset, run, cache, _ = baseline_fixture
    next((cache / 'vectors').rglob('*.npy')).unlink()
    monkeypatch.setattr(module.DenseEmbedder, '_load', lambda *args: pytest.fail('model load forbidden'))
    with pytest.raises(ValueError, match='unavailable'):
        module.VerifiedDocumentBridge(dataset, run, {'device': 'cpu', 'dtype': 'float32'}, baseline_cache=cache)


def test_corrupt_baseline_ranking_rejected(baseline_fixture):
    dataset, run, cache, _ = baseline_fixture
    path = next((cache / 'rankings').rglob('rankings.json'))
    data = json.loads(path.read_text())
    data['rankings'][0][0]['score'] = .5
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='rankings'):
        module.VerifiedDocumentBridge(dataset, run, {'device': 'cpu', 'dtype': 'float32'}, baseline_cache=cache)


def test_query_cache_corruption_rejects_without_inference(baseline_fixture, tmp_path, monkeypatch):
    import numpy as np
    dataset, run, cache, vectors = baseline_fixture
    bridge = module.VerifiedDocumentBridge(dataset, run, {'device': 'cpu', 'dtype': 'float32'}, baseline_cache=cache)
    monkeypatch.setattr(bridge.encoder, '_encode', lambda content: vectors[:1])
    bridge.rank(dataset.corpus, dataset.queries, 100, tmp_path / 'new', [set()])
    path = next((tmp_path / 'new').rglob('*.npy'))
    np.save(path, vectors[1:])
    monkeypatch.setattr(bridge.encoder, '_encode', lambda content: pytest.fail('no fallback'))
    with pytest.raises(ValueError, match='checksum'):
        bridge.rank(dataset.corpus, dataset.queries, 100, tmp_path / 'new', [set()])


def test_javascript_bridge_preserves_shared_chunks(baseline_fixture, monkeypatch, tmp_path):
    dataset, run, cache, vectors = baseline_fixture
    dataset.manifest['metadata']['language'] = 'javascript'
    monkeypatch.setattr(module.SharedCodeSegmenter, 'token_counts', lambda self, text, title='': {'bge': len(text), 'embeddinggemma': len(text)})
    ordinary = module.SegmentedCodeAdapter('embeddinggemma', {'device': 'cpu', 'dtype': 'float32'})
    identity = module._channel_identity(dataset, 'E', {'E': ordinary})
    direct = cache / 'direct' / identity
    chunks, summaries = [], []
    for item in dataset.corpus:
        parts, summary = ordinary.segmenter.partition(item)
        chunks.extend(parts)
        summaries.append(summary)
    module._atomic_json(direct / 'shared-segmentation.json', {'identity': ordinary.segmenter.identity,
        'protocol': ordinary.segmenter.protocol, 'functions': summaries})
    for role, items, values in [('document', chunks, vectors), ('query', dataset.queries, vectors[:1])]:
        content = [ordinary.encoder.format_document(item) if role == 'document' else ordinary.encoder.format_query(item['text']) for item in items]
        key = module.stable_hash({'adapter': ordinary.identity, 'role': role, 'formatted_inputs': content})
        module._atomic_array(direct / 'vectors' / role / (key + '.npy'), values)
    ranks = module._cosine_rankings(vectors, vectors[:1], ['a', 'b'], [0, 1], [set()])
    module._atomic_json(cache / 'rankings' / module.stable_hash({'identity': identity, 'top_k': 100}) / 'rankings.json', {'identity': identity, 'query_ids': ['q'], 'rankings': ranks})
    report = json.loads((run / 'report.json').read_text())
    report['adapter_specs']['channels']['E'] = {'identity': ordinary.identity, 'encoder': {'config': ordinary.encoder.config}}
    report['cells'][0]['identity'] = module.stable_hash({**report['configuration'], 'variant': 'code__e__none',
        'mode': 'total', 'candidate_k': None, 'channel_identities': {'E': identity}, 'reranker': None, 'reranker_prompts': None})
    report['channels']['E'] = {'status': 'completed', 'identity': identity, 'chunk_count': 2, 'original_function_count': 2,
        'source_characters': 2, 'covered_source_characters': 2, 'source_utf8_bytes': 2, 'covered_source_utf8_bytes': 2}
    (run / 'report.json').write_text(json.dumps(report))
    bridge = module.VerifiedDocumentBridge(dataset, run, {'device': 'cpu', 'dtype': 'float32'}, baseline_cache=cache)
    assert bridge.provenance['segmentation_identity'] == ordinary.segmenter.identity
    assert bridge.offsets == [0, 1]
    monkeypatch.setattr(bridge.encoder, '_encode', lambda inputs: vectors[:1])
    actual, usage = bridge.rank(dataset.corpus, dataset.queries, 100, tmp_path / 'ablation', [set()])
    assert actual == ranks
    assert usage['document_encoding']['fresh_document_encoder_calls'] == 0
    payload = json.loads((direct / 'shared-segmentation.json').read_text())
    payload['functions'][0]['source_sha256'] = 'changed'
    (direct / 'shared-segmentation.json').write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='segmentation'):
        module.VerifiedDocumentBridge(dataset, run, {'device': 'cpu', 'dtype': 'float32'}, baseline_cache=cache)


def test_cli_verified_completion_and_verify_only(baseline_fixture, tmp_path, monkeypatch):
    dataset, run, cache, _ = baseline_fixture
    dataset.track = 'code'
    monkeypatch.setattr(module, 'load_dataset', lambda path: dataset)
    from types import SimpleNamespace
    monkeypatch.setattr(module.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(returncode=0))
    calls = []
    def fake_matrix(*args, **kwargs):
        calls.append(kwargs)
        assert isinstance(kwargs['adapters']['E'], module.VerifiedDocumentBridge)
        return {'cells': [{'variant_id': 'code__e__none', 'status': 'completed', 'query_count': 1, 'completed_queries': 1}]}
    monkeypatch.setattr(module, 'run_matrix', fake_matrix)
    args = ['--dataset', str(tmp_path), '--baseline-run', str(run), '--baseline-cache', str(cache),
            '--run-dir', str(tmp_path / 'ablations' / 'run'), '--device', 'cpu', '--dtype', 'float32']
    assert module.main(args + ['--verify-baseline-only']) == 0
    assert calls == []
    assert module.main(args) == 0
    assert len(calls) == 1


def test_verified_bridge_changed_runtime_inputs_reject(baseline_fixture, tmp_path, monkeypatch):
    dataset, run, cache, _ = baseline_fixture
    bridge = module.VerifiedDocumentBridge(dataset, run, {'device': 'cpu', 'dtype': 'float32'}, baseline_cache=cache)
    monkeypatch.setattr(bridge.encoder, '_encode', lambda inputs: pytest.fail('no inference'))
    changed = [dict(item) for item in dataset.corpus]
    changed[0]['text'] = 'changed'
    with pytest.raises(ValueError, match='Frozen'):
        bridge.rank(changed, dataset.queries, 100, tmp_path, [set()])


def test_baseline_metric_corruption_rejected(baseline_fixture):
    dataset, run, cache, _ = baseline_fixture
    path = run / 'report.json'
    report = json.loads(path.read_text())
    report['cells'][0]['metrics']['hit@5'] = .5
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='metrics'):
        module.VerifiedDocumentBridge(dataset, run, {'device': 'cpu', 'dtype': 'float32'}, baseline_cache=cache)


@pytest.mark.parametrize('target', ['ranking_identity', 'ranking_query_order', 'cell_identity'])
def test_baseline_provenance_headers_rejected(baseline_fixture, target):
    dataset, run, cache, _ = baseline_fixture
    if target == 'cell_identity':
        path = run / 'report.json'
        payload = json.loads(path.read_text())
        payload['cells'][0]['identity'] = 'wrong'
    else:
        path = next((cache / 'rankings').rglob('rankings.json'))
        payload = json.loads(path.read_text())
        payload['identity' if target == 'ranking_identity' else 'query_ids'] = 'wrong'
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='identity'):
        module.VerifiedDocumentBridge(dataset, run, {'device': 'cpu', 'dtype': 'float32'}, baseline_cache=cache)


def test_bridge_inherits_exact_baseline_config_and_rejects_unused_difference(baseline_fixture):
    dataset, run, cache, _ = baseline_fixture
    bridge = module.VerifiedDocumentBridge(dataset, run, {'device': 'cpu', 'dtype': 'float32'}, baseline_cache=cache)
    report = json.loads((run / 'report.json').read_text())
    original = report['adapter_specs']['channels']['E']['embedder']['config']
    assert bridge.encoder.config == original
    with pytest.raises(ValueError, match='budget'):
        module.VerifiedDocumentBridge(dataset, run, {'device': 'cpu', 'dtype': 'float32', 'max_length': 128}, baseline_cache=cache)
    with pytest.raises(ValueError, match='settings'):
        module.VerifiedDocumentBridge(dataset, run, {'device': 'cpu', 'dtype': 'float32', 'unused_setting': 1}, baseline_cache=cache)
    explicit = module.VerifiedDocumentBridge(dataset, run, {'device': 'cpu', 'dtype': 'float32', 'batch_size': 8}, baseline_cache=cache)
    assert explicit.encoder.config == {**original, 'batch_size': 8}

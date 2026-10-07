"""Independent synthetic original G cache proof; no models loaded."""
import json
import numpy as np
import pytest
from rag_benchmark import multimodal_bge_extra_baseline as module
from rag_benchmark.models import DenseEmbedder

@pytest.fixture
def baseline_fixture(tmp_path):
    import numpy as np
    root = tmp_path / 'data'
    root.mkdir()
    (root / 'dataset.json').write_text(json.dumps({'id': 'fixture', 'track': 'code', 'revision': 'frozen', 'metadata': {'language': 'python'}}))
    for name, rows in [('corpus', [{'id': 'a', 'text': 'A'}, {'id': 'b', 'text': 'B'}]),
                       ('queries', [{'id': 'q', 'text': 'find'}]),
                       ('qrels', [{'query_id': 'q', 'corpus_id': 'a', 'relevance': 1}])]:
        (root / (name + '.jsonl')).write_text(''.join(json.dumps(row) + '\n' for row in rows))
    dataset = module.load_dataset(root)
    ordinary = module.TextEmbeddingAdapter('bge', {'device': 'cpu', 'dtype': 'float32'})
    vectors = np.zeros((2, 1024), dtype=np.float32)
    vectors[0, 0] = 1
    vectors[1, 1] = 1
    query = vectors[:1]
    cache = tmp_path / 'cache'
    identity = module._channel_identity(dataset, 'G', {'G': ordinary})
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
    cell_identity = module.stable_hash({**configuration, 'variant': 'code__g__none', 'mode': 'total',
        'candidate_k': None, 'channel_identities': {'G': identity}, 'reranker': None, 'reranker_prompts': None})
    report = {'configuration': configuration, 'dataset': {'identity': dataset.identity}, 'scope': 'frozen_collection', 'evaluated_query_count': 1,
        'channels': {'G': {'identity': identity, 'status': 'completed', **encodings}},
        'adapter_specs': {'channels': {'G': {'identity': ordinary.identity, 'embedder': {'config': ordinary.embedder.config}}}},
        'cells': [{'variant_id': 'code__g__none', 'identity': cell_identity, 'budget_mode': 'total',
                   'candidate_k': None, 'status': 'completed', 'completed_queries': 1, 'metrics': metrics}]}
    run = tmp_path / 'baseline'
    run.mkdir()
    (run / 'report.json').write_text(json.dumps(report))
    return dataset, run, cache, vectors


def test_verified_original_g_and_no_model_load(baseline_fixture, monkeypatch):
    dataset, run, cache, vectors = baseline_fixture
    monkeypatch.setattr(DenseEmbedder, "_load", lambda *args: pytest.fail("Model loading forbidden"))
    proof = module.VerifiedGBaseline(dataset, run, baseline_cache=cache)
    assert np.array_equal(proof.documents, vectors)
    assert not proof.documents.flags.writeable and not proof.queries.flags.writeable
    assert proof.provenance["verified_metric_count"] == 17
    assert proof.records[0] == {"id": "a", "source_id": "a", "formatted_text": "A"}
    assert proof.provenance["fresh_encoder_calls"] == 0
    assert "vector_files" not in proof.public_summary()


@pytest.mark.parametrize("corruption", ["source", "ranking", "metric", "query_ids", "model", "dtype", "missing", "vector"])
def test_corruption_never_falls_back(baseline_fixture, monkeypatch, corruption):
    dataset, run, cache, vectors = baseline_fixture
    monkeypatch.setattr(DenseEmbedder, "_load", lambda *args: pytest.fail("Model loading forbidden"))
    path = run / "report.json"
    report = json.loads(path.read_text())
    if corruption == "source":
        dataset.corpus[0]["text"] = "changed"
    elif corruption == "ranking":
        rankpath = next((cache / "rankings").rglob("rankings.json"))
        data = json.loads(rankpath.read_text())
        data["rankings"][0][0]["score"] = .5
        rankpath.write_text(json.dumps(data))
    elif corruption == "metric":
        report["cells"][0]["metrics"]["hit@5"] = .5
    elif corruption == "query_ids":
        report["configuration"]["query_ids"] = ["different"]
    elif corruption == "model":
        report["adapter_specs"]["channels"]["G"]["embedder"]["config"]["revision"] = "0" * 40
    elif corruption == "dtype":
        report["adapter_specs"]["channels"]["G"]["embedder"]["config"]["dtype"] = "bfloat16"
    elif corruption == "missing":
        next((cache / "vectors").rglob("*.npy")).unlink()
    else:
        np.save(next((cache / "vectors").rglob("*.npy")), np.zeros((2, 1024), dtype=np.float32))
    path.write_text(json.dumps(report))
    with pytest.raises((ValueError, FileNotFoundError)):
        module.VerifiedGBaseline(dataset, run, baseline_cache=cache)


@pytest.fixture
def javascript_fixture(baseline_fixture, monkeypatch):
    from types import SimpleNamespace
    dataset, run, cache, _ = baseline_fixture
    manifest = dataset.manifest
    manifest['metadata']['language'] = 'javascript'
    (dataset.root / 'dataset.json').write_text(json.dumps(manifest))
    (dataset.root / 'corpus.jsonl').write_text(json.dumps({'id': 'a', 'text': 'AA'}) + '\n' + json.dumps({'id': 'b', 'text': 'B'}) + '\n')
    dataset = module.load_dataset(dataset.root)
    def partition(item):
        parts = [{'id': item['id'] + str(i), 'function_id': item['id'], 'text': char, 'chunk_index': i,
                  'char_start': i, 'char_end': i + 1, 'byte_start': i, 'byte_end': i + 1,
                  'text_sha256': module.stable_hash(char)} for i, char in enumerate(item['text'])]
        return parts, {'function_id': item['id'], 'chunks': [part['id'] for part in parts]}
    segmenter = SimpleNamespace(identity='segments', protocol={'lossless': True}, partition=partition)
    def adapter(name, config):
        return SimpleNamespace(identity=module.stable_hash(config), encoder=DenseEmbedder(name, config),
                               segmenter=segmenter, rank=lambda: None)
    monkeypatch.setattr(module, 'SegmentedCodeAdapter', adapter)
    config = DenseEmbedder('bge').config
    ordinary = adapter('bge', config)
    channel = module._channel_identity(dataset, 'G', {'G': ordinary})
    direct = cache / 'direct' / channel
    direct.mkdir(parents=True)
    chunks, summaries = [], []
    for item in dataset.corpus:
        parts, summary = partition(item)
        chunks.extend(parts)
        summaries.append(summary)
    (direct / 'shared-segmentation.json').write_text(json.dumps({'identity': segmenter.identity, 'protocol': segmenter.protocol, 'functions': summaries}))
    documents = np.zeros((3, 1024), dtype=np.float32)
    documents[0, 1] = documents[2, 1] = 1
    documents[1, 0] = 1
    queries = documents[1:2]
    for role, items, vectors in [('document', chunks, documents), ('query', dataset.queries, queries)]:
        content = [ordinary.encoder.format_document(item) if role == 'document' else ordinary.encoder.format_query(item['text']) for item in items]
        key = module.stable_hash({'adapter': ordinary.identity, 'role': role, 'formatted_inputs': content})
        directory = direct / 'vectors' / role
        directory.mkdir(parents=True)
        np.save(directory / (key + '.npy'), vectors)
    rankings = module._cosine_rankings(documents, queries, ['a', 'b'], [0, 2], [set()])
    rankpath = cache / 'rankings' / module.stable_hash({'identity': channel, 'top_k': 100}) / 'rankings.json'
    rankpath.parent.mkdir(parents=True)
    rankpath.write_text(json.dumps({'identity': channel, 'query_ids': ['q'], 'rankings': rankings}))
    report = json.loads((run / 'report.json').read_text())
    report['dataset']['identity'] = dataset.identity
    report['adapter_specs']['channels']['G'] = {'identity': ordinary.identity, 'encoder': {'config': config}}
    report['channels']['G'] = {'status': 'completed', 'identity': channel, 'chunk_count': 3, 'original_function_count': 2,
        'source_characters': 3, 'covered_source_characters': 3, 'source_utf8_bytes': 3, 'covered_source_utf8_bytes': 3}
    cell = report['cells'][0]
    cell['identity'] = module.stable_hash({**report['configuration'], 'variant': cell['variant_id'], 'mode': cell['budget_mode'],
        'candidate_k': cell['candidate_k'], 'channel_identities': {'G': channel}, 'reranker': None, 'reranker_prompts': None})
    (run / 'report.json').write_text(json.dumps(report))
    return dataset, run, cache, direct


def test_javascript_exact_segment_order_offsets_source_max(javascript_fixture):
    dataset, run, cache, _ = javascript_fixture
    proof = module.VerifiedGBaseline(dataset, run, baseline_cache=cache)
    assert proof.offsets == [0, 2]
    assert [row['id'] for row in proof.records] == ['a0', 'a1', 'b0']
    assert [row['char_start'] for row in proof.records] == [0, 1, 0]
    assert proof.provenance['source_aggregation'] == 'maximum_chunk_score'
    assert proof.provenance['document_rows'] == 3


def test_javascript_segmentation_corruption_no_fallback(javascript_fixture):
    dataset, run, cache, direct = javascript_fixture
    path = direct / 'shared-segmentation.json'
    payload = json.loads(path.read_text())
    payload['functions'][0]['chunks'].reverse()
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='segmentation differs'):
        module.VerifiedGBaseline(dataset, run, baseline_cache=cache)


def test_model_item_callback_mutation_rejected(baseline_fixture):
    dataset, run, cache, _ = baseline_fixture
    original = dataset.model_item
    def mutating(item, **kwargs):
        dataset.corpus[0]['text'] = 'mutated during callback'
        return original(item, **kwargs)
    dataset.model_item = mutating
    with pytest.raises(ValueError, match='rows changed during verification'):
        module.VerifiedGBaseline(dataset, run, baseline_cache=cache)


def test_public_state_is_defensive_and_reuse_guarded(baseline_fixture):
    dataset, run, cache, _ = baseline_fixture
    proof = module.VerifiedGBaseline(dataset, run, baseline_cache=cache)
    identity = proof.identity
    summary = proof.public_summary()
    summary['config']['dtype'] = 'mutated'
    proof.config['dtype'] = 'changed'
    proof.provenance['config']['dtype'] = 'changed'
    proof.records[0]['formatted_text'] = 'changed'
    proof.query_records[0]['formatted_text'] = 'changed'
    proof.ids.append('changed')
    assert proof.identity == identity and proof.config['dtype'] == 'float32'
    assert proof.records[0]['formatted_text'] == 'A'
    proof.validate_for_reuse()
    with pytest.raises(ValueError):
        proof.documents.flags.writeable = True
    dataset.corpus[0]['text'] = 'changed'
    with pytest.raises(ValueError, match='supplied source/query rows changed'):
        proof.verified_inputs('document')


def test_metric_tolerance_is_explicit_and_strict(baseline_fixture):
    dataset, run, cache, _ = baseline_fixture
    path = run / 'report.json'
    report = json.loads(path.read_text())
    report['cells'][0]['metrics']['hit@5'] += 1e-7
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='metrics'):
        module.VerifiedGBaseline(dataset, run, baseline_cache=cache)

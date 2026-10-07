import json
import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest

from rag_benchmark.models import MODEL_DEFAULTS
from rag_benchmark.multimodal_reverse_data import prepare_reverse
from rag_benchmark.multimodal_reverse_runner import (
    GALLERY_PREFIX, QUERY_PREFIX, ReverseEG2Bridge, run_reverse as actual_run_reverse, verified_reverse,
)
from test_multimodal_reverse_data import parent


class FakeEncoder:
    dimension = 768

    def __init__(self, role):
        self.role = role
        self.config = {**MODEL_DEFAULTS['embeddinggemma'], 'device': 'cpu', 'dtype': 'float32',
            'max_length': 8192, 'batch_size': 1, 'local_files_only': True}
        if role == 'query':
            self.config.update(mode='native', query_prompt=QUERY_PREFIX, document_prompt=GALLERY_PREFIX)
        self.prompt_identity = {'document': GALLERY_PREFIX} if role == 'document' else None
        self.identity = 'independent-fixture-' + role
        self.calls = []
        self.fail_after = None

    def encode(self, items, role):
        assert role == self.role
        if self.fail_after is not None and len(self.calls) >= self.fail_after:
            raise ValueError('fixture expanded context overflow; no truncation')
        self.calls.append(items)
        result = np.zeros((len(items), 768), dtype=np.float32)
        for index, item in enumerate(items):
            if role == 'query':
                assert item['media'] and 'text' not in item
                coordinate = 0 if item['id'] == 'a' else 1
            else:
                assert not item['media'] and item['text']
                coordinate = 1 if item['id'] == '3' else 0
            result[index, coordinate] = 1
        self.last_usage = {'context_limit': 8192, 'truncated_text_items': 0,
            'items': [{'id': item['id'], 'original_tokens': 4, 'retained_tokens': 4} for item in items]}
        return result

    def unload(self):
        pass



def baseline_fixture(source, bridge):
    from rag_benchmark.multimodal import load_dataset, encoding_cache_identity, _channel_identity, exact_dot_rankings, graded_metrics, aggregate_metrics
    from rag_benchmark.models import stable_hash
    dataset = load_dataset(source)
    baseline = source.parent / 'baseline'
    cache = source.parent / 'baseline-cache'
    baseline.mkdir(exist_ok=True)
    channel = _channel_identity(dataset, 'N', {'N': bridge.query})
    config = {'dataset': dataset.identity, 'query_ids': [row['id'] for row in dataset.queries], 'channel_budget': 100}
    vectors = {}
    usages = {}
    for role, items in (('document', dataset.corpus), ('query', dataset.queries)):
        key = stable_hash({'encoding': encoding_cache_identity(dataset, bridge.query, role), 'text_only': False})
        directory = cache / 'vectors' / key
        directory.mkdir(parents=True, exist_ok=True)
        values = np.zeros((len(items), 768), dtype=np.float32)
        for index, item in enumerate(items):
            values[index, 0 if role == 'query' or item['id'] == 'a' else 1] = 1
        vectors[role] = values
        block = directory / f'000000000-{len(items):09d}.npy'
        np.save(block, values)
        block.with_suffix('.usage.json').write_text(json.dumps({'rows':len(items), 'reported':True,
            'items':[{'id':row['id'], 'modalities':list(row.get('media',{})), 'expanded_tokens':4} for row in items]}))
        usages[role + '_encoding'] = {'identity':key}
    rankings = exact_dot_rankings(vectors['document'], vectors['query'], [row['id'] for row in dataset.corpus], top_k=100)
    metric_rows = [graded_metrics([row['id'] for row in ranking], dataset.qrels[query['id']]) for query, ranking in zip(dataset.queries,rankings)]
    cell = {'variant_id':dataset.track+'__n__none', 'budget_mode':'per_channel', 'candidate_k':None,
        'status':'completed', 'completed_queries':len(dataset.queries), 'metrics':aggregate_metrics(metric_rows)}
    cell['identity'] = stable_hash({**config,'variant':cell['variant_id'],'mode':'per_channel','candidate_k':None,
        'channel_identities':{'N':channel},'reranker':None,'reranker_prompts':None})
    report = {'dataset':{'identity':dataset.identity},'scope':'frozen_collection','evaluated_query_count':len(dataset.queries),
        'configuration':config,'adapter_specs':{'channels':{'N':{'class':'rag_benchmark.multimodal_models.EmbeddingGemma2Adapter',
            'config':bridge.query.config,'identity':bridge.query.identity}}},'channels':{'N':{'status':'completed','identity':channel,**usages}},'cells':[cell]}
    (baseline/'report.json').write_text(json.dumps(report))
    with sqlite3.connect(baseline/'progress.sqlite3') as connection:
        connection.execute('CREATE TABLE results (cell TEXT, query_id TEXT,payload TEXT)')
        for query, ranking, metrics in zip(dataset.queries,rankings,metric_rows):
            connection.execute('INSERT INTO results VALUES (?,?,?)',(cell['identity'],query['id'],json.dumps({'query_id':query['id'],'ranking':ranking,'metrics':metrics})))
    bridge.baseline_run, bridge.baseline_cache = baseline, cache
    return baseline, cache


def run_reverse(dataset, parent, out, *, bridge, **kwargs):
    return actual_run_reverse(dataset,parent,out,bridge=bridge,
        media_baseline_run=bridge.baseline_run,media_baseline_cache=bridge.baseline_cache,**kwargs)

def fixture(tmp_path, monkeypatch, *, speech=False):
    source = parent(tmp_path, speech=speech)
    derived = tmp_path / 'reverse'
    prepare_reverse(source, derived)
    query, gallery = FakeEncoder('query'), FakeEncoder('document')
    bridge = ReverseEG2Bridge(device='cpu', dtype='float32', query_adapter=query, gallery_adapter=gallery)
    monkeypatch.setattr('rag_benchmark.multimodal_reverse_runner.subprocess.run',
                        lambda *args, **kwargs: SimpleNamespace(returncode=0))
    baseline_fixture(source, bridge)
    monkeypatch.setattr('rag_benchmark.multimodal_reverse_runner._baseline_native_adapter',lambda config: bridge.query)
    return source, derived, tmp_path / 'run', bridge


def test_actual_constructor_bridge_pinned_compatible_without_loading_models():
    bridge = ReverseEG2Bridge(device='cpu', dtype='float32')
    assert bridge.query._model is None and bridge.gallery.base.embedder._model is None
    assert bridge.query.config['model_id'] == bridge.gallery.config['model_id']
    assert bridge.query.config['revision'] == bridge.gallery.config['revision']
    assert bridge.query.dimension == bridge.gallery.dimension == 768
    assert bridge.query.identity != bridge.gallery.identity
    assert bridge.gallery.base.embedder.format_document({'text': 'caption'}) == GALLERY_PREFIX + 'caption'
    assert 'no textual query prefix' in bridge.contract['query_condition']


@pytest.mark.parametrize('speech', [False, True])
def test_full_gallery_mult_positive_recall_grouping_and_resume(tmp_path, monkeypatch, speech):
    source, derived, out, bridge = fixture(tmp_path, monkeypatch, speech=speech)
    original_manifest = (derived / 'dataset.json').read_bytes()
    first = run_reverse(derived, source, out, bridge=bridge, block_size=1)
    assert first['status'] == 'completed' and first['metrics']['recall@5'] == 1
    assert first['completed_queries'] == 2
    assert first['grouping']['source_groups'] == (1 if speech else 2)
    assert first['main_matrix_denominator_contribution'] == 0
    before = (len(bridge.query.calls), len(bridge.gallery.calls))
    second = run_reverse(derived, source, out, bridge=bridge, block_size=1)
    assert (len(bridge.query.calls), len(bridge.gallery.calls)) == before
    assert second['result_cache_hits'] == 2 and second['metrics'] == first['metrics']
    assert second['gallery_encoding']['new_rows'] == second['query_encoding']['new_rows'] == 0
    assert second['gallery_encoding']['inference_seconds'] is None
    assert (derived / 'dataset.json').read_bytes() == original_manifest
    assert 'same duplicated caption' not in json.dumps(second)
    assert str(tmp_path) not in json.dumps(second)


@pytest.mark.parametrize('field,value', [('revision', 'wrong'), ('model_id', 'wrong/model'),
    ('dimension', 128), ('device', 'mps'), ('dtype', 'bfloat16'), ('document_prompt', 'wrong')])
def test_model_device_dimension_prefix_compatibility_is_strict(field, value):
    query, gallery = FakeEncoder('query'), FakeEncoder('document')
    query.config[field] = value
    with pytest.raises(ValueError):
        ReverseEG2Bridge(device='cpu', dtype='float32', query_adapter=query, gallery_adapter=gallery)


def test_parent_hash_and_intended_reference_provenance_are_strict(tmp_path, monkeypatch):
    source, derived, out, bridge = fixture(tmp_path, monkeypatch)
    manifest_path = source / 'dataset.json'
    data = json.loads(manifest_path.read_text())
    data['license'] = 'changed'
    manifest_path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='fingerprint'):
        verified_reverse(derived, source)
    assert not bridge.query.calls and not bridge.gallery.calls


def test_completed_cache_metric_and_vector_corruption_rejected(tmp_path, monkeypatch):
    source, derived, out, bridge = fixture(tmp_path, monkeypatch)
    run_reverse(derived, source, out, bridge=bridge)
    with sqlite3.connect(out / 'progress.sqlite3') as connection:
        payload = json.loads(connection.execute('SELECT payload FROM results LIMIT 1').fetchone()[0])
        payload['metrics']['hit@5'] = 2
        connection.execute('UPDATE results SET payload=? WHERE query_id=?', (json.dumps(payload), payload['query_id']))
    assert run_reverse(derived, source, out, bridge=bridge)['status'] == 'failed'
    vector = next((out / 'cache').rglob('*.npy'))
    values = np.load(vector, allow_pickle=False)
    values[:] = 0
    np.save(vector, values)
    calls = (len(bridge.query.calls), len(bridge.gallery.calls))
    assert run_reverse(derived, source, out, bridge=bridge)['status'] == 'failed'
    assert (len(bridge.query.calls), len(bridge.gallery.calls)) == calls


def test_partial_failed_encoding_resumes_only_integrity_sealed_blocks(tmp_path, monkeypatch):
    source, derived, out, bridge = fixture(tmp_path, monkeypatch)
    bridge.gallery.fail_after = 1
    first = run_reverse(derived, source, out, bridge=bridge, block_size=1)
    assert first['status'] == 'failed' and not bridge.query.calls
    assert len(list((out / 'cache').rglob('*.checksums.json'))) == 1
    bridge.gallery.fail_after = None
    second = run_reverse(derived, source, out, bridge=bridge, block_size=1)
    assert second['status'] == 'completed' and second['gallery_encoding']['cache_rows'] == 1
    assert len(bridge.gallery.calls) == 3


def test_changed_query_or_gallery_identity_cannot_reuse_old_run(tmp_path, monkeypatch):
    source, derived, out, bridge = fixture(tmp_path, monkeypatch)
    run_reverse(derived, source, out, bridge=bridge)
    gallery = FakeEncoder('document')
    gallery.identity = 'changed-gallery'
    other = ReverseEG2Bridge(device='cpu', dtype='float32', query_adapter=FakeEncoder('query'), gallery_adapter=gallery)
    other.baseline_run, other.baseline_cache = bridge.baseline_run, bridge.baseline_cache
    with pytest.raises(ValueError, match='separate run'):
        run_reverse(derived, source, out, bridge=other)
    assert not gallery.calls


def test_forward_public_directory_guard_remains_separate(tmp_path, monkeypatch):
    source, derived, out, bridge = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr('rag_benchmark.multimodal_reverse_runner.subprocess.run',
                        lambda *args, **kwargs: SimpleNamespace(returncode=1))
    with pytest.raises(ValueError, match='git-ignored'):
        run_reverse(derived, source, out, bridge=bridge)


def test_honest_reference_gallery_policy_not_forward_gold_free_override(tmp_path, monkeypatch):
    from rag_benchmark.models import stable_hash
    source, derived, out, bridge = fixture(tmp_path, monkeypatch)
    path = derived / 'dataset.json'
    manifest = json.loads(path.read_text())
    manifest['gallery_provenance']['gold_fields_used'] = False
    manifest.pop('identity')
    manifest['identity'] = stable_hash(manifest)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        verified_reverse(derived, source)
    assert not bridge.query.calls and not bridge.gallery.calls


def test_public_diagnostics_explicit_availability_and_complete_sample_counts_only():
    from rag_benchmark.multimodal_reverse_runner import _public_usage
    usage = {'role': 'query', 'adapter_diagnostics': {'status': 'complete', 'total_rows': 2,
        'missing_rows': 0, 'reported_rows': 2, 'items': [
            {'id': 'private-a', 'source_audio_samples': 16, 'retained_audio_samples': 16},
            {'id': 'private-b', 'source_audio_samples': 24, 'retained_audio_samples': 24}]}}
    report = _public_usage(usage, audio_required=True)
    assert report['diagnostics']['status'] == 'complete'
    assert report['diagnostics']['total_rows'] == 2 and report['diagnostics']['missing_rows'] == 0
    assert report['audio_sample_coverage']['status'] == 'complete'
    assert report['audio_sample_coverage']['source_audio_samples_total'] == 40
    assert report['audio_sample_coverage']['retained_audio_samples_total'] == 40
    assert 'private-' not in json.dumps(report)
    usage['adapter_diagnostics']['items'][1].pop('retained_audio_samples')
    missing = _public_usage(usage, audio_required=True)['audio_sample_coverage']
    assert missing['status'] == 'unavailable'
    assert missing['source_audio_samples_total'] is missing['retained_audio_samples_total'] is None
    assert _public_usage(usage)['audio_sample_coverage']['status'] == 'not_applicable'


@pytest.mark.parametrize('damage', ['missing', 'norm', 'order', 'score', 'config'])
def test_media_baseline_invalid_is_unavailable_without_encoding(tmp_path, monkeypatch, damage):
    source, derived, out, bridge = fixture(tmp_path, monkeypatch)
    if damage in ('missing','norm'):
        path = next(bridge.baseline_cache.rglob('*.npy'))
        if damage == 'missing':
            path.unlink()
        else:
            np.save(path, np.zeros_like(np.load(path)))
    elif damage == 'order':
        path = next(bridge.baseline_cache.rglob('*.usage.json'))
        payload=json.loads(path.read_text())
        payload['items'].reverse()
        path.write_text(json.dumps(payload))
    elif damage == 'score':
        with sqlite3.connect(bridge.baseline_run/'progress.sqlite3') as connection:
            row=connection.execute('SELECT query_id,payload FROM results LIMIT 1').fetchone()
            payload=json.loads(row[1])
            payload['ranking'][0]['score']=0.5
            connection.execute('UPDATE results SET payload=? WHERE query_id=?',(json.dumps(payload),row[0]))
    else:
        path=bridge.baseline_run/'report.json'
        payload=json.loads(path.read_text())
        payload['adapter_specs']['channels']['N']['identity']='wrong'
        path.write_text(json.dumps(payload))
    report=run_reverse(derived,source,out,bridge=bridge)
    assert report['status']=='unavailable' and report['completed_queries']==0
    assert not bridge.query.calls and not bridge.gallery.calls
    assert not out.exists()


def test_missing_baseline_never_falls_back_to_media_inference(tmp_path, monkeypatch):
    source,derived,out,bridge=fixture(tmp_path,monkeypatch)
    report=actual_run_reverse(derived,source,out,bridge=bridge)
    assert report['status']=='unavailable' and not bridge.query.calls and not bridge.gallery.calls


def test_reused_media_queries_never_encode_and_bind_baseline_proof(tmp_path, monkeypatch):
    source,derived,out,bridge=fixture(tmp_path,monkeypatch)
    report=run_reverse(derived,source,out,bridge=bridge)
    assert report['status']=='completed' and not bridge.query.calls
    assert bridge.gallery.calls and report['query_encoding']['new_rows']==0
    assert report['media_baseline']['stored_forward_results_verified']==3
    assert report['media_baseline']['source_report_sha256']


def test_real_native_image_prepare_query_and_document_are_identical_without_prefix(tmp_path):
    Image = pytest.importorskip('PIL.Image')
    path = tmp_path / 'image.png'
    Image.new('RGB', (3, 2), color=(10, 20, 30)).save(path)
    bridge = ReverseEG2Bridge(device='cpu', dtype='float32')
    query, query_meta = bridge.query._prepare({'media': {'image': str(path)}}, 'query')
    document, doc_meta = bridge.query._prepare({'media': {'image': str(path)}, 'text': 'ignored reference'}, 'document')
    assert set(query) == set(document) == {'image'}
    assert query['image'].tobytes() == document['image'].tobytes()
    assert query_meta == doc_meta and bridge.query._model is None


@pytest.mark.parametrize('damage', ['finite', 'shape', 'processing'])
def test_media_baseline_rejects_numerical_or_processing_mismatch(tmp_path, monkeypatch, damage):
    source, derived, out, bridge = fixture(tmp_path, monkeypatch)
    if damage == 'processing':
        path = bridge.baseline_run / 'report.json'
        report = json.loads(path.read_text())
        report['adapter_specs']['channels']['N']['config']['vision_budget'] = 560
        path.write_text(json.dumps(report))
    else:
        path = next(bridge.baseline_cache.rglob('*.npy'))
        values = np.load(path)
        if damage == 'finite':
            values[0, 0] = np.nan
        else:
            values = values[:, :128]
        np.save(path, values)
    report = run_reverse(derived, source, out, bridge=bridge)
    assert report['status'] == 'unavailable'
    assert not bridge.query.calls and not bridge.gallery.calls


def test_parent_forward_payload_metadata_is_not_reinterpreted_as_model_input(tmp_path, monkeypatch):
    source, derived, out, bridge = fixture(tmp_path, monkeypatch)
    with sqlite3.connect(bridge.baseline_run / 'progress.sqlite3') as connection:
        for qid, payload in connection.execute('SELECT query_id,payload FROM results').fetchall():
            value = json.loads(payload)
            value.update(pool=100, usage={'cached': True})
            value['ranking'][0]['score'] -= 1e-7
            connection.execute('UPDATE results SET payload=? WHERE query_id=?', (json.dumps(value), qid))
    report = run_reverse(derived, source, out, bridge=bridge)
    assert report['status'] == 'completed'
    assert report['media_baseline']['maximum_observed_score_delta'] <= 2e-6
    assert not bridge.query.calls


def test_explicit_other_language_source_reuses_only_identical_media_bytes(tmp_path, monkeypatch):
    import shutil
    from rag_benchmark.multimodal_reverse_runner import verified_media_baseline
    source, derived, out, bridge = fixture(tmp_path, monkeypatch)
    other = tmp_path / 'other-language'
    shutil.copytree(source, other)
    reverse = verified_reverse(derived, source)
    values, usage = verified_media_baseline(reverse, source, bridge, bridge.baseline_run,
        bridge.baseline_cache, other)
    assert len(values) == 2 and usage['new_rows'] == 0 and not bridge.query.calls
    (other / 'a.bin').write_bytes(b'different source')
    with pytest.raises(ValueError):
        verified_media_baseline(reverse, source, bridge, bridge.baseline_run, bridge.baseline_cache, other)
    assert not bridge.query.calls and not bridge.gallery.calls

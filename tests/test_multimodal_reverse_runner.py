import json
import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest

from rag_benchmark.models import MODEL_DEFAULTS
from rag_benchmark.multimodal_reverse_data import prepare_reverse
from rag_benchmark.multimodal_reverse_runner import (
    GALLERY_PREFIX, QUERY_PREFIX, ReverseEG2Bridge, run_reverse, verified_reverse,
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


def fixture(tmp_path, monkeypatch, *, speech=False):
    source = parent(tmp_path, speech=speech)
    derived = tmp_path / 'reverse'
    prepare_reverse(source, derived)
    query, gallery = FakeEncoder('query'), FakeEncoder('document')
    bridge = ReverseEG2Bridge(device='cpu', dtype='float32', query_adapter=query, gallery_adapter=gallery)
    monkeypatch.setattr('rag_benchmark.multimodal_reverse_runner.subprocess.run',
                        lambda *args, **kwargs: SimpleNamespace(returncode=0))
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

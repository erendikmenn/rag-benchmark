"""Fake CPU-only preparation; no model, HTTP, subprocess, server or downloads."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

S = importlib.util.spec_from_file_location('composed_captions', Path(__file__).parents[1] / 'scripts/prepare_composed_captions.py')
m = importlib.util.module_from_spec(S)
S.loader.exec_module(m)


def dataset(tmp_path, documents=2, queries=2):
    (tmp_path / 'image.jpg').write_bytes(b'FAKE IMAGE BYTES; never decoded')
    return SimpleNamespace(root=tmp_path, track='composed_image',
        corpus=[{'id': str(i), 'media': {'image': 'image.jpg'}, 'metadata': {'gold_caption': 'MUST NOT ENTER REQUEST'}} for i in range(documents)],
        queries=[{'id': 'q' + str(i), 'text': 'make it red', 'media': {'image': 'image.jpg'},
                  'metadata': {'reference_id': str(i % documents), 'target_caption': 'SECRET TARGET', 'target_id': 'gold', 'exclude_ids': [str(i % documents)]}} for i in range(queries)])


def spec():
    return {'config': {}, 'runtime_proof': {'context_limit': 8192}, 'protocol': m.PROTOCOL}


def response(text='A visible source object.', finish='stop'):
    body = json.dumps({'choices': [{'finish_reason': finish, 'message': {'content': text}}], 'usage': {'prompt_tokens': 100, 'completion_tokens': 10}})
    return {'status_code': 200, 'body': body, 'body_sha256': hashlib.sha256(body.encode()).hexdigest()}


def fake_payload(item, config):
    assert set(item) == {'text', 'media'} and item['text'] == ''
    return {'source_image': item['media']['image'], 'max_tokens': 512, 'repeat_penalty': 1.1}


def test_claim_raw_before_strict_and_completed_only_resume(tmp_path):
    ds = dataset(tmp_path)
    run = tmp_path / 'run'
    run.mkdir()
    def transport(request):
        directory = run / 'rows/000000000'
        assert (directory / 'claim.json').is_file()
        assert not (directory / 'accepted.json').exists()
        return response()
    accepted, complete = m.prepare_rows(ds, spec(), run, transport, payload_builder=fake_payload, max_new=1)
    assert len(accepted) == 1 and not complete
    assert (run / 'rows/000000000/raw-response.json').is_file()
    calls = []
    accepted, complete = m.prepare_rows(ds, spec(), run, lambda r: calls.append(r) or response(), payload_builder=fake_payload)
    assert complete and len(accepted) == 2 and len(calls) == 1
    accepted_again, complete = m.prepare_rows(ds, spec(), run, None, payload_builder=fake_payload, read_only=True)
    assert complete and accepted_again == accepted


def test_failed_strict_retains_raw_and_refuses_retry(tmp_path):
    ds = dataset(tmp_path)
    run = tmp_path / 'run'
    run.mkdir()
    with pytest.raises(ValueError, match='normal_stop'):
        m.prepare_rows(ds, spec(), run, lambda _: response(finish='length'), payload_builder=fake_payload)
    directory = run / 'rows/000000000'
    assert (directory / 'raw-response.json').is_file() and (directory / 'failed.json').is_file()
    assert not (directory / 'accepted.json').exists()
    calls = []
    with pytest.raises(ValueError, match='claimed_incomplete_no_retry'):
        m.prepare_rows(ds, spec(), run, lambda r: calls.append(r), payload_builder=fake_payload)
    assert not calls


def test_metadata_gold_cannot_change_source_payload(tmp_path):
    ds = dataset(tmp_path)
    before = m.source_item(ds, ds.corpus[0])
    ds.corpus[0]['metadata']['gold_caption'] = 'DIFFERENT SECRET'
    ds.queries[0]['text'] = 'DIFFERENT QUERY'
    assert m.source_item(ds, ds.corpus[0]) == before
    ds.corpus[0]['text'] = 'gold description'
    with pytest.raises(ValueError, match='gallery_pixels_only'):
        m.source_item(ds, ds.corpus[0])


def test_actual_payload_explicit_policy_and_image_only(monkeypatch):
    import rag_benchmark.multimodal_generation as generation
    captured = []
    monkeypatch.setattr(generation, 'LocalMultimodalGenerator', lambda config: SimpleNamespace(config={'model': 'pinned'}, caption_system='source only'))
    monkeypatch.setattr(generation, 'media_content', lambda item, config: captured.append(copy.deepcopy(item)) or [{'type': 'image_url', 'image_url': {'url': 'fake'}}])
    result = m.payload({'text': '', 'media': {'image': '/fake/source.jpg'}}, {'cache_prompt': False})
    assert result['max_tokens'] == 512 and result['repeat_penalty'] == 1.1
    assert result['temperature'] == 0 and result['seed'] == 42 and result['chat_template_kwargs'] == {'enable_thinking': False}
    assert captured == [{'text': '', 'media': {'image': '/fake/source.jpg'}}]


def test_full_39826_gallery_30037_query_surrogates_cpu(tmp_path):
    ds = dataset(tmp_path, documents=39826, queries=30037)
    from rag_benchmark.lasco_data import ANNOTATION_SHA256, GALLERY_SHA256, REVISION
    ds.manifest = {'id': 'lasco-validation', 'revision': REVISION,
        'metadata': {'annotation_revision': REVISION, 'split': 'validation', 'empty_transition_count': 6},
        'sources': [{'file': 'lasco_val.json', 'sha256': ANNOTATION_SHA256, 'size_bytes': 1},
                    {'file': 'lasco_val_corpus.json', 'sha256': GALLERY_SHA256, 'size_bytes': 1},
                    {'file': 'val2014.zip', 'sha256': 'a' * 64, 'size_bytes': 1}]}
    for query in ds.queries[:6]:
        query['text'] = ''
        query['metadata']['transition_text_empty'] = True
    m.validate_source(ds)
    asset = m.sha(tmp_path / 'image.jpg')
    accepted = [{'source_id': str(i), 'text': 'Generated source ' + str(i), 'asset_sha256': asset, 'spec_sha256': 'pinned'} for i in range(39826)]
    corpus, queries = m.publication_rows(ds, accepted)
    assert len(corpus) == 39826 and len(queries) == 30037
    assert all(q['text'] == ds.queries[i]['text'] and q['media'] == {'image': 'image.jpg'} for i, q in enumerate(queries))
    assert sum(q['metadata']['transition_text_empty'] for q in queries) == 6
    for i, query in enumerate(queries):
        assert query['metadata']['text_surrogate']['text'] == f'Reference: Generated source {i}\nRequested change: ' + ds.queries[i]['text']
        assert query['metadata']['text_surrogate']['provenance']['gold_fields_used'] is False
    assert 'SECRET TARGET' not in json.dumps(queries) and 'MUST NOT ENTER REQUEST' not in json.dumps(corpus)
    assert all('target_id' not in q['metadata'] for q in queries)
    ds.queries[0]['metadata']['reference_id'] = 'target-outside-gallery'
    with pytest.raises(ValueError, match='gallery_reference_required'):
        m.validate_source(ds)


def test_source_mutation_during_request_and_receipt_mutation(tmp_path):
    ds = dataset(tmp_path)
    run = tmp_path / 'run'
    run.mkdir()
    def mutate(_):
        (tmp_path / 'image.jpg').write_bytes(b'CHANGED')
        return response()
    with pytest.raises(ValueError, match='source_changed_during_request'):
        m.prepare_rows(ds, spec(), run, mutate, payload_builder=fake_payload)
    assert (run / 'rows/000000000/raw-response.json').exists()


def test_strict_rejects_repetition_and_context(tmp_path):
    with pytest.raises(ValueError, match='pathological_repetition'):
        m.strict(response('word ' * 21), 8192)
    with pytest.raises(ValueError, match='runtime_context'):
        m.strict(response(), 600)


def test_exclusive_and_canonical_boundary(tmp_path):
    output = tmp_path / 'receipt.json'
    m.write_once(output, {'original': True})
    with pytest.raises(FileExistsError):
        m.write_once(output, {'original': False})
    assert json.loads(output.read_text()) == {'original': True}
    with pytest.raises(ValueError, match='path_escape'):
        m.safe(tmp_path / '..' / 'escape', tmp_path)


@pytest.mark.parametrize('mutation', ['spec_config', 'row', 'request', 'callback_config'])
def test_callback_mutation_fails_before_acceptance(tmp_path, mutation):
    ds = dataset(tmp_path)
    pins = spec()
    run = tmp_path / 'run'
    run.mkdir()
    def transport(request):
        if mutation == 'spec_config':
            pins['config']['max_tokens'] = 1
        elif mutation == 'row':
            ds.corpus[0]['media']['image'] = 'other.jpg'
        elif mutation == 'request':
            request['max_tokens'] = 1
        return response()
    def builder(item, config):
        if mutation == 'callback_config':
            config['max_tokens'] = 1
        return fake_payload(item, config)
    with pytest.raises(ValueError, match='mutation'):
        m.prepare_rows(ds, pins, run, transport, payload_builder=builder)
    assert not (run / 'rows/000000000/accepted.json').exists()


def test_accepted_resume_rejects_changed_request_or_source(tmp_path):
    ds = dataset(tmp_path, documents=1, queries=1)
    run = tmp_path / 'run'
    run.mkdir()
    m.prepare_rows(ds, spec(), run, lambda _: response(), payload_builder=fake_payload)
    accepted = run / 'rows/000000000/accepted.json'
    value = json.loads(accepted.read_text())
    value['request_sha256'] = 'wrong'
    accepted.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='accepted_receipt_binding'):
        m.prepare_rows(ds, spec(), run, None, payload_builder=fake_payload, read_only=True)


def test_official_identity_and_source_hashes_are_required(tmp_path):
    from rag_benchmark.lasco_data import ANNOTATION_SHA256, GALLERY_SHA256, REVISION
    ds = dataset(tmp_path, documents=2, queries=6)
    ds.manifest = {'id': 'lasco-validation', 'revision': REVISION,
        'metadata': {'annotation_revision': REVISION, 'split': 'validation', 'empty_transition_count': 6},
        'sources': [{'file': 'lasco_val.json', 'sha256': ANNOTATION_SHA256, 'size_bytes': 1},
                    {'file': 'lasco_val_corpus.json', 'sha256': GALLERY_SHA256, 'size_bytes': 1},
                    {'file': 'val2014.zip', 'sha256': 'a' * 64, 'size_bytes': 1}]}
    for q in ds.queries:
        q['text'] = ''
        q['metadata']['transition_text_empty'] = True
    m.validate_official_source(ds)
    ds.manifest['sources'][0]['sha256'] = 'b' * 64
    with pytest.raises(ValueError, match='official_annotation_hashes'):
        m.validate_official_source(ds)
    ds.manifest['id'] = 'unrelated-dataset'
    with pytest.raises(ValueError, match='official_lasco_identity'):
        m.validate_official_source(ds)


@pytest.mark.parametrize('context', [8192.0, True, '8192', 4096])
def test_runtime_context_is_exact_integer8192(context):
    with pytest.raises(ValueError, match='runtime_context'):
        m.strict(response(), context)

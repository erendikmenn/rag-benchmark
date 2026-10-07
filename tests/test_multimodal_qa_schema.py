"""Offline QA schema contract tests; mocked transport, no local server."""
import copy
import json
import sqlite3
from types import SimpleNamespace

import pytest

from rag_benchmark import multimodal_qa_runner as qa
from test_multimodal_qa_runner import FakeLocalTransport, fixture, make_retrieval


@pytest.mark.parametrize('citation', [0, 1, True, 'unknown', 'source title'])
def test_schema_does_not_weaken_exact_parser(citation):
    with pytest.raises(ValueError, match='actually supplied evidence'):
        qa.parse_prediction(json.dumps({'answer': 'fact', 'citations': [citation]}), {'p'})


def test_all_six_conditions_only_supplied_ids_enter_schema(tmp_path, monkeypatch):
    root, raw, out = fixture(tmp_path, monkeypatch)
    retrieval = make_retrieval(tmp_path, qa.verified_dataset(root))
    transport = FakeLocalTransport()
    result = qa.run_qa(root, [raw], out, transport=transport,
                       retrieval_run=retrieval, retrieval_cell='actual-cell')
    assert result['completed_tasks'] == 6
    assert len(transport.requests) == 6
    for request in transport.requests:
        response = request['response_format']
        assert response['type'] == 'json_schema' and response['json_schema']['strict'] is True
        schema = response['json_schema']['schema']
        assert schema['required'] == ['answer', 'citations'] and schema['additionalProperties'] is False
        assert schema['properties']['answer'] == {'type': 'string'}
        ids = [s['id'] for s in request['sources']]
        if ids:
            assert schema['properties']['citations'] == {'type': 'array', 'items': {'type': 'string', 'enum': ids}}
        else:
            assert schema['properties']['citations'] == {'type': 'array', 'const': []}
        assert 'Correct fact' not in json.dumps(request) and 'Not certified alias' not in json.dumps(request)


def test_schema_changes_request_identity_and_preserves_blocked_v2(tmp_path, monkeypatch):
    monkeypatch.setattr('subprocess.run', lambda *a, **k: SimpleNamespace(returncode=0))
    cache = qa.SharedGenerationCache(tmp_path / 'shared')
    transport = FakeLocalTransport()
    request = {'system': 'Synthetic v2 allowed IDs', 'question': 'Synthetic?', 'language': 'English',
               'sources': [{'id': 'p', 'text': 'Synthetic evidence'}]}
    config = qa.frozen_config()
    with pytest.raises(ValueError):
        cache.generate(request, config, transport,
                       lambda: {'text': json.dumps({'answer': 'fact', 'citations': [123]})})
    schema_request = copy.deepcopy(request)
    schema_request['response_format'] = qa.citation_response_format(request['sources'])
    _, reused, _ = cache.generate(schema_request, config, transport, lambda: transport.generate(schema_request))
    assert not reused
    with pytest.raises(ValueError, match='no implicit regeneration'):
        cache.generate(request, config, transport, lambda: pytest.fail('old blocked response cannot retry'), retry_failed=True)
    with sqlite3.connect(cache.directory / 'generation.sqlite3') as database:
        assert dict(database.execute('select status,count(*) from responses group by status')) == {
            'blocked_integrity': 1, 'completed': 1}
    assert qa.QA_PROMPT_CONTRACT_VERSION != 'qa-exact-source-id-citations-v2'


def test_completed_cached_schema_mutation_blocks_inference(tmp_path, monkeypatch):
    root, raw, out = fixture(tmp_path, monkeypatch)
    transport = FakeLocalTransport()
    config = qa.frozen_config()
    config['protocol']['conditions'] = ['closed_book']
    config['protocol']['representations'] = ['text']
    qa.run_qa(root, [raw], out, config=config, transport=transport)
    with sqlite3.connect(out / 'qa.sqlite3') as database:
        payload = json.loads(database.execute('select payload from predictions').fetchone()[0])
        payload['request']['response_format']['json_schema']['schema']['additionalProperties'] = True
        database.execute('update predictions set payload=?', (json.dumps(payload),))
    result = qa.run_qa(root, [raw], out, config=config, transport=transport, retry_failed=True)
    assert result['completed_tasks'] == 0 and result['status_counts']['failed'] == 1
    assert len(transport.requests) == 1 and result['invocation']['new_inference_requests'] == 0


def test_local_transport_exact_schema_payload_and_full_output_budget(monkeypatch):
    import httpx
    captured = []
    config = qa.frozen_config()['generator']
    class Client:
        def __init__(self, **kwargs):
            assert kwargs == {'timeout': config['timeout'], 'trust_env': False, 'follow_redirects': False}
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def post(self, url, json):
            captured.append(copy.deepcopy(json))
            result = {'choices': [{'message': {'content': '{"answer":"fact","citations":["p"]}'},
                                  'finish_reason': 'stop'}], 'usage': {'prompt_tokens': 12}}
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: result)
    monkeypatch.setattr(httpx, 'Client', Client)
    from rag_benchmark.multimodal_generation import LocalMultimodalGenerator
    generator = object.__new__(LocalMultimodalGenerator)
    generator.config = config
    generator._ready = True
    generator._context_limit = 8192
    generator._check_config = lambda: None
    generator.verifier = SimpleNamespace(base_url='http://fixture')
    transport = object.__new__(qa.LocalQATransport)
    transport.generator = generator
    request = {'system': 'Synthetic contract', 'question': 'Synthetic?', 'language': 'English',
               'sources': [{'id': 'p', 'text': 'Synthetic evidence'}]}
    request['response_format'] = qa.citation_response_format(request['sources'])
    assert json.loads(transport.generate(request)['text'])['citations'] == ['p']
    assert captured[0]['response_format'] == request['response_format']
    assert captured[0]['max_tokens'] == config['max_tokens'] == 384
    request['response_format']['json_schema']['schema']['properties']['citations']['items']['enum'] = ['unknown']
    with pytest.raises(ValueError, match='schema differs'):
        transport.generate(request)
    assert len(captured) == 1

import copy
import json
import sqlite3
from types import SimpleNamespace

import pytest

from rag_benchmark.multimodal_qa_runner import SharedGenerationCache, frozen_config, run_qa
from test_multimodal_qa_runner import FakeLocalTransport, fixture, make_retrieval
from rag_benchmark.multimodal_qa_runner import verified_dataset


def shared(tmp_path, monkeypatch):
    monkeypatch.setattr('subprocess.run', lambda *args, **kwargs: SimpleNamespace(returncode=0))
    return SharedGenerationCache(tmp_path / 'shared')


def request(text='Evidence'):
    return {'system': 'system', 'question': 'question', 'language': 'English',
            'sources': [{'id': 'p', 'text': text}]}


def test_exact_request_cache_excludes_protocol_and_references(tmp_path, monkeypatch):
    cache = shared(tmp_path, monkeypatch)
    transport = FakeLocalTransport()
    config = frozen_config()
    first, reused, key = cache.generate(request(), config, transport, lambda: transport.generate(request()))
    assert reused is False
    config['protocol']['conditions'] = ['retrieved']
    second, reused, again = cache.generate(request(), config, transport, lambda: pytest.fail('duplicate call'))
    assert reused is True and first == second and key == again
    assert len(transport.requests) == 1
    with sqlite3.connect(cache.directory / 'generation.sqlite3') as db:
        payload = json.loads(db.execute('SELECT payload FROM responses').fetchone()[0])
    assert set(payload) == {'identity', 'response'}
    assert 'qa_readiness' not in payload['identity'] and 'references' not in payload['identity']


@pytest.mark.parametrize('changed', ['source', 'config', 'runtime', 'citation_id', 'media'])
def test_changed_generation_input_does_not_share(tmp_path, monkeypatch, changed):
    cache = shared(tmp_path, monkeypatch)
    transport = FakeLocalTransport()
    config, req = frozen_config(), request()
    if changed == 'media':
        path = tmp_path / 'media.jpg'
        path.write_bytes(b'first')
        req['sources'][0]['media'] = {'image': str(path)}
    _, _, firstkey = cache.generate(req, config, transport, lambda: transport.generate(req))
    req = copy.deepcopy(req)
    if changed == 'source':
        req['sources'][0]['text'] += 'changed'
    if changed == 'config':
        config['generator']['seed'] += 1
    if changed == 'runtime':
        transport.identity += '-changed'
    if changed == 'citation_id':
        req['sources'][0]['id'] = 'another'
    if changed == 'media':
        path.write_bytes(b'changed bytes')
    _, reused, key = cache.generate(req, config, transport, lambda: transport.generate(req))
    assert reused is False and key != firstkey and len(transport.requests) == 2


@pytest.mark.parametrize('damage', ['payload', 'status', 'incomplete'])
def test_corruption_and_incomplete_never_fall_back(tmp_path, monkeypatch, damage):
    from rag_benchmark.models import stable_hash
    cache = shared(tmp_path, monkeypatch)
    transport = FakeLocalTransport()
    config = frozen_config()
    cache.generate(request(), config, transport, lambda: transport.generate(request()))
    with sqlite3.connect(cache.directory / 'generation.sqlite3') as db:
        if damage == 'payload':
            db.execute("UPDATE responses SET payload='{}'")
        if damage == 'status':
            db.execute("UPDATE responses SET status='failed'")
        if damage == 'incomplete':
            payload = db.execute('SELECT payload FROM responses').fetchone()[0]
            db.execute('UPDATE responses SET status=?,checksum=?', ('running', stable_hash({'status': 'running', 'payload': payload})))
    with pytest.raises(ValueError):
        cache.generate(request(), config, transport, lambda: pytest.fail('regeneration forbidden'), retry_failed=True)
    assert len(transport.requests) == 1


def test_two_cells_identical_full_request_one_transport_call(tmp_path, monkeypatch):
    root, raw, out = fixture(tmp_path, monkeypatch)
    retrieval = make_retrieval(tmp_path, verified_dataset(root))
    config = frozen_config()
    config['protocol']['conditions'] = ['retrieved']
    config['protocol']['representations'] = ['text']
    transport = FakeLocalTransport()
    cache = tmp_path / 'shared'
    first = run_qa(root, [raw], out, config=config, transport=transport,
                   retrieval_run=retrieval, retrieval_cell='actual-cell', shared_generation_cache=cache)
    # A different completed cell with byte-identical source request.
    report_path = retrieval / 'report.json'
    report = json.loads(report_path.read_text())
    report['cells'][0]['identity'] = 'another-cell'
    report_path.write_text(json.dumps(report))
    with sqlite3.connect(retrieval / 'progress.sqlite3') as db:
        db.execute("UPDATE results SET cell='another-cell'")
    second = run_qa(root, [raw], tmp_path / 'run-two', config=config, transport=transport,
                   retrieval_run=retrieval, retrieval_cell='another-cell', shared_generation_cache=cache)
    assert first['completed_tasks'] == second['completed_tasks'] == 1
    assert len(transport.requests) == 1
    assert first['invocation']['new_inference_requests'] == 1
    assert second['invocation']['new_inference_requests'] == 0
    assert second['invocation']['cross_run_shared_responses_reused'] == 1
    assert second['invocation']['cached_completed_tasks_validated'] == 0
    again = run_qa(root, [raw], tmp_path / 'run-two', config=config, transport=transport,
                   retrieval_run=retrieval, retrieval_cell='another-cell', shared_generation_cache=cache)
    assert again['invocation']['cached_completed_tasks_validated'] == 1
    assert again['invocation']['cross_run_shared_responses_reused'] == 0


def test_shared_answer_has_fresh_independent_gold_metrics(tmp_path, monkeypatch):
    first_root, first_raw, first_out = fixture(tmp_path / 'first', monkeypatch)
    second_root, second_raw, second_out = fixture(tmp_path / 'second', monkeypatch, reference='Different reference')
    config = frozen_config()
    config['protocol']['conditions'] = ['oracle']
    config['protocol']['representations'] = ['text']
    transport = FakeLocalTransport()
    cache = tmp_path / 'shared'
    first = run_qa(first_root, [first_raw], first_out, config=config, transport=transport, shared_generation_cache=cache)
    second = run_qa(second_root, [second_raw], second_out, config=config, transport=transport, shared_generation_cache=cache)
    assert first['completed_tasks'] == second['completed_tasks'] == 1
    assert len(transport.requests) == 1
    assert first['evaluated_groups'][0]['completed_only_mean_metrics']['answer_em'] == 1
    assert second['evaluated_groups'][0]['completed_only_mean_metrics']['answer_em'] == 0
    assert second['invocation']['cross_run_shared_responses_reused'] == 1
    with sqlite3.connect(cache / 'generation.sqlite3') as db:
        encoded = db.execute('SELECT payload FROM responses').fetchone()[0]
    assert 'Different reference' not in encoded and 'Not certified alias' not in encoded


def test_corrupt_shared_response_marks_run_failed_without_new_call(tmp_path, monkeypatch):
    root, raw, out = fixture(tmp_path, monkeypatch)
    config = frozen_config()
    config['protocol']['conditions'] = ['oracle']
    config['protocol']['representations'] = ['text']
    cache = tmp_path / 'shared'
    transport = FakeLocalTransport()
    first = run_qa(root, [raw], out, config=config, transport=transport, shared_generation_cache=cache)
    assert first['completed_tasks'] == 1
    with sqlite3.connect(cache / 'generation.sqlite3') as db:
        db.execute("UPDATE responses SET payload='{}'")
    result = run_qa(root, [raw], tmp_path / 'second', config=config, transport=transport,
                    shared_generation_cache=cache, retry_failed=True)
    assert result['completed_tasks'] == 0 and result['status_counts']['failed'] == 1
    assert result['invocation']['new_inference_requests'] == 0 and len(transport.requests) == 1
    again = run_qa(root, [raw], tmp_path / 'second', config=config, transport=transport,
                   shared_generation_cache=cache, retry_failed=True)
    assert again['completed_tasks'] == 0 and again['invocation']['new_inference_requests'] == 0


def test_media_changed_during_generation_not_shared(tmp_path, monkeypatch):
    cache = shared(tmp_path, monkeypatch)
    path = tmp_path / 'image.jpg'
    path.write_bytes(b'original')
    req = request()
    req['sources'][0]['media'] = {'image': str(path)}
    transport = FakeLocalTransport()
    def mutate():
        response = transport.generate(req)
        path.write_bytes(b'changed during generation')
        return response
    with pytest.raises(ValueError, match='media changed'):
        cache.generate(req, frozen_config(), transport, mutate)
    with sqlite3.connect(cache.directory / 'generation.sqlite3') as db:
        assert db.execute('SELECT status FROM responses').fetchone()[0] == 'blocked_integrity'
    path.write_bytes(b'original')
    with pytest.raises(ValueError):
        cache.generate(req, frozen_config(), transport, lambda: pytest.fail('media integrity retry forbidden'), retry_failed=True)


def test_shared_cache_requires_private_directory_before_preflight(tmp_path, monkeypatch):
    root, raw, out = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr('subprocess.run', lambda args, **kwargs: SimpleNamespace(
        returncode=1 if str(tmp_path / 'public') in args else 0))
    transport = FakeLocalTransport()
    transport.preflight = lambda: pytest.fail('public cache must reject before preflight')
    with pytest.raises(ValueError, match='git-ignored'):
        run_qa(root, [raw], out, transport=transport, shared_generation_cache=tmp_path / 'public')


def test_concurrent_identical_requests_have_one_fresh_call(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    cache = shared(tmp_path, monkeypatch)
    transport = FakeLocalTransport()
    start = threading.Barrier(2)
    def invoke():
        start.wait()
        return cache.generate(request(), frozen_config(), transport, lambda: transport.generate(request()))
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: invoke(), range(2)))
    assert sorted(row[1] for row in results) == [False, True]
    assert len(transport.requests) == 1


def test_explicit_retry_transport_failure_but_not_cache_corruption(tmp_path, monkeypatch):
    cache = shared(tmp_path, monkeypatch)
    transport = FakeLocalTransport(fail=True)
    with pytest.raises(RuntimeError):
        cache.generate(request(), frozen_config(), transport, lambda: transport.generate(request()))
    transport.fail = False
    with pytest.raises(ValueError, match='implicit'):
        cache.generate(request(), frozen_config(), transport, lambda: pytest.fail('implicit retry'))
    response, reused, _ = cache.generate(request(), frozen_config(), transport,
        lambda: transport.generate(request()), retry_failed=True)
    assert reused is False and len(transport.requests) == 2
    assert json.loads(response['text'])['answer'] == 'Correct fact'


def test_failed_raw_response_retained_locally_with_shared_cache(tmp_path, monkeypatch):
    root, raw, out = fixture(tmp_path, monkeypatch)
    config = frozen_config()
    config['protocol']['conditions'] = ['closed_book']
    config['protocol']['representations'] = ['text']
    transport = FakeLocalTransport()
    transport.generate = lambda request: {'text': 'invalid json response', 'usage': {'new_tokens': 3}}
    result = run_qa(root, [raw], out, config=config, transport=transport, shared_generation_cache=tmp_path / 'shared')
    assert result['status_counts']['failed'] == 1 and result['invocation']['new_inference_requests'] == 1
    with sqlite3.connect(out / 'qa.sqlite3') as db:
        payload = json.loads(db.execute('SELECT payload FROM predictions').fetchone()[0])
    assert payload['raw_prediction'] == 'invalid json response'
    assert payload['usage']['new_tokens'] == 3


@pytest.mark.parametrize('mutation', ['request', 'config', 'runtime'])
def test_detected_mutation_blocks_even_explicit_retry(tmp_path, monkeypatch, mutation):
    cache = shared(tmp_path, monkeypatch)
    req, config, transport = request(), frozen_config(), FakeLocalTransport()
    original_req, original_config, original_runtime = copy.deepcopy(req), copy.deepcopy(config), transport.identity
    def mutate():
        response = transport.generate(req)
        if mutation == 'request':
            req['question'] = 'changed'
        if mutation == 'config':
            config['generator']['seed'] += 1
        if mutation == 'runtime':
            transport.identity = 'changed runtime'
        return response
    with pytest.raises(ValueError):
        cache.generate(req, config, transport, mutate)
    with sqlite3.connect(cache.directory / 'generation.sqlite3') as db:
        assert db.execute('SELECT status FROM responses').fetchone()[0] == 'blocked_integrity'
    req.clear()
    req.update(original_req)
    config.clear()
    config.update(original_config)
    transport.identity = original_runtime
    with pytest.raises(ValueError):
        cache.generate(req, config, transport, lambda: pytest.fail('integrity retry forbidden'), retry_failed=True)
    assert len(transport.requests) == 1


def test_existing_shared_directory_and_files_private(tmp_path, monkeypatch):
    path = tmp_path / 'shared'
    path.mkdir(mode=0o755)
    cache = shared(tmp_path, monkeypatch)
    transport = FakeLocalTransport()
    cache.generate(request(), frozen_config(), transport, lambda: transport.generate(request()))
    assert cache.directory.stat().st_mode & 0o777 == 0o700
    assert (cache.directory / 'generation.sqlite3').stat().st_mode & 0o777 == 0o600
    assert (cache.directory / 'generation.lock').stat().st_mode & 0o777 == 0o600

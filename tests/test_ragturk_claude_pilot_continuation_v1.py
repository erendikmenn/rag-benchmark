"""Synthetic closed receipts only; never launch CLI, GPU, API or DB work."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from rag_benchmark.claude_cli import MODEL

spec = importlib.util.spec_from_file_location('continuation_tests', Path(__file__).parents[1] / 'scripts/run_ragturk_claude_pilot_continuation_v1.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def data(base):
    chunks = [{'id': 'a', 'title': 'A', 'text': 'Synthetic A', 'answer': 'GOLD_SENTINEL'},
              {'id': 'b', 'title': 'B', 'text': 'Synthetic B'}]
    rows = [{'query_id': f'q{i}', 'question': f'Synthetic question {i}', 'methods': {
        method: {'method': method, 'query_id': f'q{i}', 'candidate_ids': ['a', 'b'],
                 'candidate_scores': [2., 1.]} for method in base.METHODS}} for i in range(20)]
    rankings = {(r['query_id'], method): ['a', 'b'] for r in rows for method in base.METHODS}
    return chunks, rows, rankings


def test_full200_labels_exact_prompt_dedup_and_no_gold():
    base, _ = m.load_base()
    chunks, rows, rankings = data(base)
    jobs = m.jobs_for(base, rows, chunks, rankings, 'provider')
    assert len(jobs) == 20
    assert sum(len(j['aliases']) for j in jobs) == 200
    assert all(len(j['aliases']) == 10 for j in jobs)
    assert 'GOLD_SENTINEL' not in str(jobs)
    assert m.jobs_for(base, rows, chunks, rankings, 'provider') == jobs


def test_explicit_none_only100_does_not_require_laya():
    base, _ = m.load_base()
    chunks, rows, _ = data(base)
    jobs = m.jobs_for(base, rows, chunks, {}, 'provider', ['none'])
    assert len(jobs) == 20
    assert sum(len(j['aliases']) for j in jobs) == 100
    assert {a['reranker'] for j in jobs for a in j['aliases']} == {'none'}


def test_laya_context_order_creates_separate_prompt():
    base, _ = m.load_base()
    chunks, rows, rankings = data(base)
    rankings[('q0', 'bm25')] = ['b', 'a']
    jobs = m.jobs_for(base, rows, chunks, rankings, 'provider')
    assert len(jobs) == 21
    assert jobs[1]['contexts'][0]['id'] == 'b'


def completed(base, directory, job):
    directory.mkdir()
    usage = {'input_tokens': 1, 'output_tokens': 3, 'cache_read_input_tokens': 0,
             'cache_creation_input_tokens': 0}
    events = [{'type': 'system', 'subtype': 'init', 'model': MODEL, 'tools': [], 'mcp_servers': []},
              {'type': 'assistant', 'message': {'id': 'synthetic', 'model': MODEL, 'stop_reason': None,
                                               'content': [{'type': 'text', 'text': 'Synthetic answer'}]}},
              {'type': 'result', 'subtype': 'success', 'is_error': False, 'num_turns': 1,
               'stop_reason': 'end_turn', 'result': 'Synthetic answer', 'usage': usage,
               'total_cost_usd': .001, 'modelUsage': {MODEL: {'inputTokens': 1, 'outputTokens': 3,
                   'cacheReadInputTokens': 0, 'cacheCreationInputTokens': 0}}}]
    raw = '\n'.join(json.dumps(e) for e in events)
    _, parsed_usage = m.parse_response(raw, 0)
    parsed_usage.update(request_id=job['request_id'], provider_identity='provider', seconds=1.)
    base.private(directory / 'input.json', {'provider_identity': 'provider', 'job': job, 'seals': {}})
    base.private(directory / 'result.json', {'status': 'completed', 'request_id': job['request_id'], 'usage': parsed_usage})
    receipt = directory / 'receipts' / job['request_id']
    receipt.mkdir(parents=True)
    base.private(receipt / 'claim.json', {'request_id': job['request_id'], 'provider_identity': 'provider'})
    (receipt / 'stdout.private.jsonl').write_text(raw)
    (receipt / 'stderr.private.log').write_text('')
    base.private(receipt / 'completed.json', {'answer': 'Synthetic answer', 'usage': parsed_usage,
                 'stdout_sha256': hashlib.sha256(raw.encode()).hexdigest()})
    return receipt


@pytest.mark.parametrize('mutation', [None, 'failed', 'ambiguous', 'raw', 'prompt', 'provider'])
def test_prior_success_reuse_requires_exact_closed_receipt(tmp_path, mutation):
    base, _ = m.load_base()
    chunks, rows, rankings = data(base)
    job = m.jobs_for(base, rows, chunks, rankings, 'provider')[0]
    directory = tmp_path / 'worker-000'
    receipt = completed(base, directory, job)
    if mutation == 'failed':
        base.private(receipt / 'failure.json', {'code': 'synthetic'})
    elif mutation == 'ambiguous':
        (receipt / 'completed.json').unlink()
    elif mutation == 'raw':
        (receipt / 'stdout.private.jsonl').write_text('{}')
    elif mutation in ['prompt', 'provider']:
        value = base.read(directory / 'input.json')
        if mutation == 'prompt':
            value['job']['question'] = 'Mutated'
        else:
            value['provider_identity'] = 'other'
        (directory / 'input.json').write_text(json.dumps(value))
    if mutation is None:
        seals = m.audited_success(base, directory, job, 'provider')
        base.verify_seals(seals)
    else:
        with pytest.raises((ValueError, FileNotFoundError)):
            m.audited_success(base, directory, job, 'provider')


def laya_fixture(base, tmp_path):
    chunks, rows, rankings = data(base)
    output = tmp_path / 'laya'
    output.mkdir()
    runtime = {'adapter_identity': 'synthetic-adapter', 'runtime': {'gpu': 'Synthetic GPU'},
               'condition': 'new_cuda_float32_batch8_not_MPS_equivalent'}
    base.private(output / 'runtime.json', runtime)
    refs = {}
    payloads = {'corpus': [{k: c.get(k, '') for k in ['id', 'title', 'text']} for c in chunks],
                'questions': [{'id': r['query_id'], 'question': r['question']} for r in rows],
                'candidates': [r['methods'][method] for r in rows for method in base.METHODS],
                'producer_summary': [], 'producer_manifest': []}
    for key, value in payloads.items():
        path = tmp_path / (key + '.jsonl')
        path.write_text('\n'.join(json.dumps(r) for r in value))
        refs[key] = {'path': str(path), 'sha256': base.sha(path)}
    config = {'model_id': 'convaiinnovations/laya-multilingual',
              'revision': '1720e3e3357cfe1e281542e223f8273b0890ca34', 'device': 'cuda',
              'dtype': 'float32', 'backend': 'sdk', 'batch_size': 8, 'max_len': 8192, 'threshold': None}
    specification = {'protocol': 'ragturk-prepared-laya-cuda-draft-v1', 'expected_gpu': 'Synthetic GPU', 'laya': config, **refs,
                     'producer_blocks': [], 'implementation': [], 'model_files': []}
    specpath = tmp_path / 'spec.json'
    base.private(specpath, specification)
    hashes = []
    docs = {c['id']: {k: c.get(k, '') for k in ['id', 'title', 'text']} for c in chunks}
    for index, candidate in enumerate(payloads['candidates']):
        request = {'question': rows[index // 5]['question'],
                   'candidates': [{**docs[sid], 'score': score} for sid, score in
                                  zip(candidate['candidate_ids'], candidate['candidate_scores'], strict=True)],
                   'adapter_identity': runtime['adapter_identity'], 'runtime': runtime['runtime']}
        payload = {'method': candidate['method'], 'query_id': candidate['query_id'],
                   'input_candidate_ids': candidate['candidate_ids'], 'input_scores': candidate['candidate_scores'],
                   'ranked_ids': candidate['candidate_ids'], 'scores': [2., 1.],
                   'request_sha256': hashlib.sha256(json.dumps(request, ensure_ascii=False, sort_keys=True).encode()).hexdigest()}
        path = output / f'row-{index:08d}.json'
        base.private(path, payload)
        hashes.append(base.sha(path))
    result = {'status': 'completed_preparation_only', 'protocol': specification['protocol'],
              'spec_sha256': base.sha(specpath), 'rows': 100, 'scored_candidates': 200, 'row_receipt_sha256': hashes,
              'retrieval_calls': 0, 'answer_generation_calls': 0, 'benchmark_evaluations': 0}
    base.private(output / 'result.json', result)
    return specpath, output, rows, chunks, rankings


@pytest.mark.parametrize('mutation', [None, 'not_closed', 'row_sha', 'input_order', 'request', 'tie_order'])
def test_full100_laya_source_and_receipt_contract(tmp_path, mutation):
    base, _ = m.load_base()
    specpath, output, rows, chunks, rankings = laya_fixture(base, tmp_path)
    result = base.read(output / 'result.json')
    path = output / 'row-00000000.json'
    row = base.read(path)
    if mutation == 'not_closed':
        result['status'] = 'failed'
    elif mutation == 'row_sha':
        result['row_receipt_sha256'][0] = '0' * 64
    elif mutation == 'input_order':
        row['input_candidate_ids'].reverse()
    elif mutation == 'request':
        row['request_sha256'] = '0' * 64
    elif mutation == 'tie_order':
        row['ranked_ids'] = ['b', 'a']
        row['scores'] = [1., 1.]
    if mutation in ['input_order', 'request', 'tie_order']:
        path.write_text(json.dumps(row))
        result['row_receipt_sha256'][0] = base.sha(path)
    if mutation:
        (output / 'result.json').write_text(json.dumps(result))
    if mutation is None:
        actual, seals = m.laya_rows(base, specpath, output, rows, chunks)
        assert actual == rankings
        base.verify_seals(seals)
    else:
        with pytest.raises(ValueError):
            m.laya_rows(base, specpath, output, rows, chunks)


def test_external_plan_sha_rejects_before_claim(tmp_path):
    path = tmp_path / 'plan.json'
    path.write_text('{}')
    with pytest.raises(ValueError, match='external_plan_sha'):
        m.execute(path, '0' * 64)
    assert list(tmp_path.iterdir()) == [path]


def test_failed_initial_pilot_blocks_reusing_even_successful_rows(tmp_path):
    base, _ = m.load_base()
    base.private(tmp_path / 'summary.json', {'status': 'failed', 'provider_identity': 'provider'})
    base.private(tmp_path / 'claim.json', {'request_ids': []})
    with pytest.raises(ValueError, match='prior_not_completed'):
        m.prior_reuse(base, tmp_path, [], 'provider')


@pytest.mark.parametrize('mutation', [None, 'mapping_sha', 'model_inventory'])
def test_explicit_transport_mapping_preserves_remote_spec_without_weight_copy(tmp_path, mutation):
    base, _ = m.load_base()
    specpath, output, rows, chunks, expected = laya_fixture(base, tmp_path)
    specification = base.read(specpath)
    specification['model_files'] = [{'path': '/remote-model-only/weight.safetensors', 'sha256': 'a' * 64}]
    files = []
    for key in ['corpus', 'questions', 'candidates', 'producer_summary', 'producer_manifest']:
        ref = specification[key]
        local = Path(ref['path'])
        remote = '/remote-source/' + local.name
        files.append({'remote_path': remote, 'local_path': str(local), 'sha256': ref['sha256'],
                      'bytes': local.stat().st_size})
        ref['path'] = remote
    specpath.write_text(json.dumps(specification))
    original_spec_sha = base.sha(specpath)
    result = base.read(output / 'result.json')
    result['spec_sha256'] = original_spec_sha
    (output / 'result.json').write_text(json.dumps(result))
    proof = {'protocol': 'ragturk-laya-closed-local-transport-v1', 'status': 'passed_closed_source_transport',
             'spec_sha256': original_spec_sha, 'result_sha256': base.sha(output / 'result.json'),
             'runtime_sha256': base.sha(output / 'runtime.json'),
             'model_inventory_sha256': m.digest(specification['model_files']),
             'source_inventory_verified': True, 'model_end_seals_verified': True, 'files': files}
    if mutation == 'mapping_sha':
        proof['files'][0]['sha256'] = 'b' * 64
    elif mutation == 'model_inventory':
        proof['model_inventory_sha256'] = 'b' * 64
    proofpath = tmp_path / 'transport.json'
    base.private(proofpath, proof)
    if mutation:
        with pytest.raises(ValueError):
            m.laya_rows(base, specpath, output, rows, chunks, proofpath)
    else:
        rankings, seals = m.laya_rows(base, specpath, output, rows, chunks, proofpath)
        assert rankings == expected
        assert '/remote-model-only/weight.safetensors' not in seals
        assert base.sha(specpath) == original_spec_sha

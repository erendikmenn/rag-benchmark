"""Explicitly planned 20-query none/Laya QA continuation, with receipt-verified reuse.

Prepare is CPU-only. Execute requires the externally reviewed plan SHA. No
implicit resume, retry, quota bypass, gold scoring or full-dataset promotion.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from rag_benchmark.claude_cli import ClaudeSubscriptionGenerator, MODEL, parse_response, prompt_for
from rag_benchmark.storage import digest

ROOT = Path(__file__).absolute().parents[1]
BASE = ROOT / 'scripts/run_ragturk_claude_pilot.py'
PROTOCOL = 'ragturk-claude-none-laya-continuation-v1'


def require(ok, code):
    if not ok:
        raise ValueError(code)


def load_base():
    raw = BASE.read_bytes()
    spec = importlib.util.spec_from_file_location('_verified_claude_pilot_base', BASE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(raw, str(BASE), 'exec'), module.__dict__)
    return module, hashlib.sha256(raw).hexdigest()


def laya_rows(base, spec_path, output, rows, chunks, transport_proof=None):
    spec_path, output = base.safe(spec_path), base.safe(output)
    spec, result, runtime = base.read(spec_path), base.read(output / 'result.json'), base.read(output / 'runtime.json')
    require(result.get('status') == 'completed_preparation_only'
            and result.get('protocol') == spec.get('protocol') == 'ragturk-prepared-laya-cuda-draft-v1'
            and result.get('spec_sha256') == base.sha(spec_path)
            and result.get('rows') == 100
            and all(result.get(k) == 0 for k in ['retrieval_calls', 'answer_generation_calls', 'benchmark_evaluations'])
            and not (output / 'failed.json').exists(), 'laya_not_closed')
    require(runtime.get('condition') == 'new_cuda_float32_batch8_not_MPS_equivalent'
            and isinstance(runtime.get('adapter_identity'), str)
            and isinstance(runtime.get('runtime'), dict)
            and runtime['runtime'].get('gpu') == spec.get('expected_gpu'), 'laya_runtime')
    config = spec['laya']
    require(config.get('model_id') == 'convaiinnovations/laya-multilingual'
            and config.get('revision') == '1720e3e3357cfe1e281542e223f8273b0890ca34'
            and config.get('backend') == 'sdk' and config.get('max_len') == 8192
            and config.get('device') == 'cuda' and config.get('dtype') == 'float32'
            and config.get('batch_size') == 8 and config.get('threshold') is None, 'laya_config')
    seals = {str(spec_path): base.sha(spec_path), str(output / 'result.json'): base.sha(output / 'result.json'),
             str(output / 'runtime.json'): base.sha(output / 'runtime.json')}
    mapping = {}
    if transport_proof is not None:
        proof_path = base.safe(transport_proof)
        proof = base.read(proof_path)
        require(proof.get('protocol') == 'ragturk-laya-closed-local-transport-v1'
                and proof.get('status') == 'passed_closed_source_transport'
                and proof.get('spec_sha256') == base.sha(spec_path)
                and proof.get('result_sha256') == base.sha(output / 'result.json')
                and proof.get('runtime_sha256') == base.sha(output / 'runtime.json')
                and proof.get('model_inventory_sha256') == digest(spec['model_files'])
                and proof.get('source_inventory_verified') is True
                and proof.get('model_end_seals_verified') is True, 'laya_transport_proof')
        seals[str(proof_path)] = base.sha(proof_path)
        for ref in proof['files']:
            local = base.safe(ref['local_path'])
            require(ref['remote_path'] not in mapping and local.stat().st_size == ref['bytes']
                    and base.sha(local) == ref['sha256'], 'laya_transport_file')
            mapping[ref['remote_path']] = ref
            seals[str(local)] = ref['sha256']

    def local_ref(ref):
        if transport_proof is None:
            local = base.safe(ref['path'])
        else:
            require(ref['path'] in mapping and mapping[ref['path']]['sha256'] == ref['sha256'],
                    'laya_transport_ref')
            local = base.safe(mapping[ref['path']]['local_path'])
        require(base.sha(local) == ref['sha256'], 'laya_input_seal')
        seals[str(local)] = ref['sha256']
        return local

    local_sources = {key: local_ref(spec[key]) for key in
                     ['corpus', 'questions', 'candidates', 'producer_summary', 'producer_manifest']}
    for ref in spec['producer_blocks'] + spec['implementation']:
        local_ref(ref)
    if transport_proof is None:
        for ref in spec['model_files']:
            local_ref(ref)
    # With a pinned closed transport proof, preserve the original remote weight
    # inventory in the untouched spec; do not download or duplicate its weights.

    docs = {r['id']: {k: r.get(k, '') for k in ['id', 'title', 'text']} for r in chunks}
    corpus = [json.loads(line) for line in local_sources['corpus'].read_text().splitlines() if line]
    questions = [json.loads(line) for line in local_sources['questions'].read_text().splitlines() if line]
    candidates = [json.loads(line) for line in local_sources['candidates'].read_text().splitlines() if line]
    require(corpus == list(docs.values()), 'laya_source_corpus')
    require(questions == [{'id': r['query_id'], 'question': r['question']} for r in rows], 'laya_questions')
    expected = [r['methods'][method] for r in rows for method in base.METHODS]
    require(candidates == expected, 'laya_original_candidates')
    require(result.get('scored_candidates') == sum(len(r['candidate_ids']) for r in expected), 'laya_scored_count')
    hashes = result['row_receipt_sha256']
    require(isinstance(hashes, list) and len(hashes) == len(expected) == 100, 'laya_receipt_count')
    require({p.name for p in output.glob('row-*.json')} == {f'row-{i:08d}.json' for i in range(100)}, 'laya_receipt_inventory')
    rankings = {}
    for index, (candidate, expected_sha) in enumerate(zip(expected, hashes, strict=True)):
        path = output / f'row-{index:08d}.json'
        require(base.sha(path) == expected_sha, 'laya_row_sha')
        seals[str(path)] = expected_sha
        row = base.read(path)
        ids, scores = row['ranked_ids'], row['scores']
        original = candidate['candidate_ids']
        require(row['method'] == candidate['method'] and row['query_id'] == candidate['query_id']
                and row['input_candidate_ids'] == original and row['input_scores'] == candidate['candidate_scores']
                and len(ids) == len(set(ids)) == len(original) and set(ids) == set(original)
                and len(scores) == len(ids) and all(type(v) in (int, float) and math.isfinite(v) for v in scores),
                'laya_permutation')
        ordinal = {sid: n for n, sid in enumerate(original)}
        require(list(zip(ids, scores)) == sorted(zip(ids, scores), key=lambda x: (-x[1], ordinal[x[0]])), 'laya_stable_order')
        source_row = rows[index // 5]
        request = {'question': source_row['question'],
                   'candidates': [{**docs[sid], 'score': score} for sid, score in
                                  zip(original, candidate['candidate_scores'], strict=True)],
                   'adapter_identity': runtime['adapter_identity'], 'runtime': runtime['runtime']}
        expected_request = hashlib.sha256(json.dumps(request, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        require(row['request_sha256'] == expected_request, 'laya_source_request')
        rankings[(row['query_id'], row['method'])] = ids
    return rankings, seals


def jobs_for(base, rows, chunks, rankings, provider_identity, rerankers=('none', 'laya')):
    docs = {r['id']: r for r in chunks}
    jobs = {}
    for row in rows:
        for method in base.METHODS:
            for reranker in rerankers:
                ids = row['methods'][method]['candidate_ids'] if reranker == 'none' else rankings[(row['query_id'], method)]
                contexts = [{k: docs[sid].get(k, '') for k in ['id', 'title', 'text']} for sid in ids[:5]]
                prompt = prompt_for(row['question'], contexts)
                key = digest({'provider': provider_identity, 'prompt': prompt})
                if key not in jobs:
                    jobs[key] = {'request_id': key, 'question': row['question'], 'contexts': contexts,
                                 'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(), 'aliases': []}
                jobs[key]['aliases'].append({'query_id': row['query_id'], 'method': method, 'reranker': reranker})
    require(sum(len(j['aliases']) for j in jobs.values()) == 100 * len(rerankers), 'logical_coverage')
    return list(jobs.values())


def audited_success(base, directory, job, provider_identity):
    directory = base.safe(directory)
    require(not (directory / 'failure.json').exists(), 'prior_failure_blocks_reuse')
    input_path = directory / 'input.json'
    inp, result = base.read(input_path), base.read(directory / 'result.json')
    require(inp['provider_identity'] == provider_identity and result['status'] == 'completed'
            and result['request_id'] == job['request_id'] and inp['job']['request_id'] == job['request_id']
            and inp['job']['question'] == job['question'] and inp['job']['contexts'] == job['contexts'], 'prior_request')
    base.verify_seals(inp['seals'])
    prompt = prompt_for(job['question'], job['contexts'])
    require(digest({'provider': provider_identity, 'prompt': prompt}) == job['request_id'], 'prior_prompt')
    require({p.name for p in (directory / 'receipts').iterdir()} == {job['request_id']},
            'unknown_or_ambiguous_prior_claim')
    receipt = directory / 'receipts' / job['request_id']
    require(not (receipt / 'failure.json').exists(), 'prior_failure_blocks_reuse')
    claim, complete = base.read(receipt / 'claim.json'), base.read(receipt / 'completed.json')
    raw_path = receipt / 'stdout.private.jsonl'
    raw = raw_path.read_bytes()
    require(claim['request_id'] == job['request_id'] and claim['provider_identity'] == provider_identity
            and complete['stdout_sha256'] == hashlib.sha256(raw).hexdigest(), 'prior_raw_seal')
    answer, usage = parse_response(raw.decode(), 0)
    require(complete['answer'] == answer and all(complete['usage'].get(k) == v for k, v in usage.items())
            and complete['usage']['request_id'] == job['request_id']
            and complete['usage']['provider_identity'] == provider_identity
            and result['usage'] == complete['usage'], 'prior_completed_binding')
    return {str(p): base.sha(p) for p in [input_path, directory / 'result.json', receipt / 'claim.json',
                                         receipt / 'completed.json', raw_path, receipt / 'stderr.private.log']}


def prior_reuse(base, prior, jobs, provider_identity):
    prior = base.safe(prior)
    require(not list(prior.rglob('*.partial')), 'ambiguous_prior_partial')
    summary = base.read(prior / 'summary.json')
    claim = base.read(prior / 'claim.json')
    require(summary['status'] == 'completed' and summary['provider_identity'] == provider_identity
            and not (prior / 'failure.json').exists(), 'prior_not_completed')
    by_id = {job['request_id']: job for job in jobs}
    reuse, seals = {}, {str(prior / n): base.sha(prior / n) for n in ['summary.json', 'claim.json']}
    workers = sorted(prior.glob('worker-*'))
    require(len(workers) == summary['requests_dispatched'] == len(claim['request_ids']), 'prior_worker_inventory')
    for directory in workers:
        inp = base.read(directory / 'input.json')
        key = inp['job']['request_id']
        require(key in by_id and key not in reuse, 'prior_unknown_or_duplicate_request')
        seals.update(audited_success(base, directory, by_id[key], provider_identity))
        reuse[key] = str(directory)
    require(set(reuse) == set(claim['request_ids']), 'prior_claim_coverage')
    return reuse, seals


def prepare(args):
    base, base_sha = load_base()
    gate = base.gate_from(args.billing_evidence)
    prepared, rows, seals = base.closed_inputs(args.producer, args.bundle)
    rerankers = ['none'] if args.none_only else ['none', 'laya']
    rankings, laya_seals = ({}, {}) if args.none_only else laya_rows(
        base, args.laya_spec, args.laya_output, rows, prepared.chunks, args.laya_transport_proof)
    seals.update(laya_seals)
    generator = ClaudeSubscriptionGenerator(billing_gate=gate)
    jobs = jobs_for(base, rows, prepared.chunks, rankings, generator.identity, rerankers)
    reuse, prior_seals = prior_reuse(base, args.prior, jobs, generator.identity)
    seals.update(prior_seals)
    seals[str(BASE)] = base_sha
    seals[str(base.safe(__file__))] = base.sha(__file__)
    seals[str(base.safe(args.billing_evidence))] = base.sha(args.billing_evidence)
    output = base.safe(args.output)
    require(not output.exists(), 'fresh_output_required')
    model_paths = set() if args.none_only else {str(base.safe(ref['path'])) for ref in base.read(args.laya_spec)['model_files']}
    boundary_seals = {path: value for path, value in seals.items() if path not in model_paths}
    plan = {'protocol': PROTOCOL, 'producer': str(base.safe(args.producer)), 'bundle': str(base.safe(args.bundle)),
            'laya_spec': None if args.none_only else str(base.safe(args.laya_spec)),
            'laya_output': None if args.none_only else str(base.safe(args.laya_output)),
            'laya_transport_proof': str(base.safe(args.laya_transport_proof)) if args.laya_transport_proof else None,
            'prior': str(base.safe(args.prior)), 'billing_evidence': str(base.safe(args.billing_evidence)),
            'output': str(output), 'provider_identity': generator.identity, 'model': MODEL,
            'concurrency': args.concurrency, 'rerankers': rerankers,
            'logical_conditions': 100 * len(rerankers), 'jobs': jobs, 'reuse': reuse,
            'seals': seals, 'boundary_seals': boundary_seals, 'automatic_retry': False, 'scope': '20-query-pilot-only-not-full-dataset'}
    base.verify_seals(seals)
    base.private(args.plan, plan)
    return {'status': 'prepared', 'plan_sha256': base.sha(args.plan), 'logical_conditions': plan['logical_conditions'],
            'unique_requests': len(jobs), 'reused_requests': len(reuse), 'new_requests': len(jobs) - len(reuse)}


def audit(base, plan, output):
    base.verify_seals(plan['seals'])
    require(plan['rerankers'] in [['none'], ['none', 'laya']]
            and plan['logical_conditions'] == 100 * len(plan['rerankers']), 'logical_coverage')
    prepared, rows, _ = base.closed_inputs(plan['producer'], plan['bundle'])
    rankings = {}
    if 'laya' in plan['rerankers']:
        rankings, _ = laya_rows(base, plan['laya_spec'], plan['laya_output'], rows, prepared.chunks, plan.get('laya_transport_proof'))
    require(jobs_for(base, rows, prepared.chunks, rankings, plan['provider_identity'], plan['rerankers']) == plan['jobs'],
            'reconstructed_jobs_changed')
    completed = 0
    for index, job in enumerate(plan['jobs']):
        directory = Path(plan['reuse'][job['request_id']]) if job['request_id'] in plan['reuse'] else output / f'worker-{index:03d}'
        audited_success(base, directory, job, plan['provider_identity'])
        completed += 1
    return {'status': 'passed_full_pilot_receipt_audit', 'logical_conditions': plan['logical_conditions'],
            'unique_completed_requests': completed, 'reused_requests': len(plan['reuse']),
            'new_completed_requests': completed - len(plan['reuse']), 'unresolved_requests': 0,
            'answer_quality_evaluated': False, 'full_dataset_promoted': False}


def execute(plan_path, expected_sha):
    base, _ = load_base()
    raw = base.safe(plan_path).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected_sha, 'external_plan_sha')
    plan = json.loads(raw)
    require(plan['protocol'] == PROTOCOL and plan['model'] == MODEL and 1 <= plan['concurrency'] <= 5, 'plan_policy')
    base.verify_seals(plan['seals'])
    gate = base.gate_from(plan['billing_evidence'])
    output = base.safe(plan['output'])
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    base.private(output / 'claim.json', {'protocol': PROTOCOL, 'plan_sha256': expected_sha, 'pid': os.getpid()})
    pending = [(i, job) for i, job in enumerate(plan['jobs']) if job['request_id'] not in plan['reuse']]
    logs, closed_results = [], []
    started = time.monotonic()

    def launch(index, item):
        ordinal, job = item
        gate.check()
        base.verify_seals(plan['boundary_seals'])
        directory = output / f'worker-{ordinal:03d}'
        directory.mkdir(mode=0o700)
        input_path = directory / 'input.json'
        base.private(input_path, {'job': job, 'seals': plan['boundary_seals'], 'billing_evidence': plan['billing_evidence'],
                                 'provider_identity': plan['provider_identity'], 'worker_output': str(directory)})
        fd = os.open(directory / 'terminal.private.log', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        log = os.fdopen(fd, 'w')
        logs.append(log)
        return subprocess.Popen([str(base.PYTHON), str(BASE), '--single-worker', str(input_path),
                                 '--single-worker-sha256', base.sha(input_path)], cwd=ROOT,
                                env=base.subscription_environment(dict(os.environ)), stdout=log, stderr=log)

    def result(index, rc):
        ordinal, job = pending[index]
        path = output / f'worker-{ordinal:03d}' / 'result.json'
        value = base.read(path) if rc == 0 and path.exists() else {'status': 'failed', 'returncode': rc}
        if value.get('status') == 'completed':
            require(value['request_id'] == job['request_id'], 'worker_request')
        closed_results.append(value.get('status') == 'completed')
        base.private(output / f'progress-{len(closed_results):03d}.json', {
            'status': 'running' if all(closed_results) else 'failed_stopping',
            'logical_conditions': plan['logical_conditions'], 'new_requests_closed': len(closed_results),
            'new_requests_completed': sum(closed_results), 'reused_requests': len(plan['reuse']),
            'elapsed_seconds': time.monotonic() - started, 'ttft_seconds': None})
        return value

    old = signal.getsignal(signal.SIGTERM)

    def interrupted(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    try:
        result_summary = base.schedule(pending, launch, concurrency=plan['concurrency'], read_result=result)
        base.private(output / 'execution-results.json', result_summary)
        require(result_summary['status'] == 'completed', 'partial_pilot_requires_review')
        proof = audit(base, plan, output)
        base.private(output / 'full-receipt-audit.json', proof)
        summary = {**proof, 'status': 'completed_receipt_verified', 'protocol': PROTOCOL,
                   'plan_sha256': expected_sha, 'model': MODEL, 'provider_identity': plan['provider_identity'],
                   'elapsed_seconds': time.monotonic() - started, 'ttft_seconds': None,
                   'new_requests_dispatched': result_summary['requests_dispatched'],
                   'new_completed_requests_per_second': len(pending) / max(time.monotonic() - started, 1e-9),
                   'timing_scope': 'coordinator execution interval, including private worker and receipt verification overhead; not provider-only speed',
                   'logical_labels': {r: 100 for r in plan['rerankers']}, 'database_writes': 0}
        base.private(output / 'summary.json', summary)
        return summary
    except BaseException as exc:
        base.private(output / 'failure.json', {'status': 'failed_no_retry', 'error_type': type(exc).__name__})
        raise
    finally:
        signal.signal(signal.SIGTERM, old)
        for log in logs:
            log.close()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--plan-sha256')
    for key in ['producer', 'bundle', 'laya-spec', 'laya-output', 'prior', 'billing-evidence', 'output']:
        p.add_argument('--' + key, type=Path)
    p.add_argument('--concurrency', type=int, default=3)
    p.add_argument('--none-only', action='store_true')
    p.add_argument('--laya-transport-proof', type=Path)
    args = p.parse_args(argv)
    if args.execute:
        require(args.plan_sha256 is not None, 'external_plan_sha_required')
        value = execute(args.plan, args.plan_sha256)
    else:
        require(all(getattr(args, key.replace('-', '_')) is not None for key in
                    ['producer', 'bundle', 'prior', 'billing-evidence', 'output']), 'prepare_paths')
        require(args.none_only or (args.laya_spec is not None and args.laya_output is not None), 'closed_laya_paths')
        require(1 <= args.concurrency <= 5, 'bounded_concurrency')
        value = prepare(args)
    print(json.dumps(value))


if __name__ == '__main__':
    try:
        main()
    except (Exception, KeyboardInterrupt):
        print(json.dumps({'status': 'failed', 'private_details_omitted': True}))
        sys.exit(1)

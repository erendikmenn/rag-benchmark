"""Explicit bounded stream-controlled QA experiment; valid and invalid outcomes count.

CPU prepare, externally SHA-approved execute. No v1 claim reuse, implicit retry,
quality filtering, hidden denominator changes, or full-dataset promotion.
"""
from __future__ import annotations

import argparse
import ast
import math
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from rag_benchmark.storage import digest

ROOT = Path(__file__).absolute().parents[1]
CONTINUATION = ROOT / 'scripts/run_ragturk_claude_pilot_continuation_v1.py'
PROTOCOL = 'ragturk-claude-stream-outcomes-pilot-v2'


def require(ok, code):
    if not ok:
        raise ValueError(code)


def helpers():
    raw = CONTINUATION.read_bytes()
    spec = importlib.util.spec_from_file_location('_retained_pilot_continuation_v1', CONTINUATION)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(raw, str(CONTINUATION), 'exec'), module.__dict__)
    base, _ = module.load_base()
    return module, base


def adapter():
    from rag_benchmark.claude_cli_stream import ClaudeStreamInvalid, ClaudeSubscriptionStreamGenerator
    return ClaudeSubscriptionStreamGenerator, ClaudeStreamInvalid


def billing_gate_from(path):
    from rag_benchmark.claude_cli_stream import billing_gate_from as loader
    return loader(path)


def outcome_valid(value):
    return value.get('outcome') in {'valid', 'terminal_invalid'} and value.get('status') == 'completed_outcome'


def scheduler_outcome(value, returncode):
    return {**value, 'status': 'completed' if returncode == 0 and outcome_valid(value) else 'failed'}


def counts(outcomes):
    return {'valid_requests': sum(x.get('outcome') == 'valid' for x in outcomes),
            'terminal_invalid_requests': sum(x.get('outcome') == 'terminal_invalid' for x in outcomes),
            'fatal_requests': sum(x.get('outcome') == 'fatal' for x in outcomes)}


def verify_outcome(base, directory, job, provider_identity):
    """CPU independent binding/replay; no transport preflight or inference."""
    from rag_benchmark.claude_cli_stream import audit_stream, ClaudeStreamInvalid, PROTOCOL as TRANSPORT
    from rag_benchmark.claude_cli import prompt_for
    from rag_benchmark.storage import digest
    directory = base.safe(directory)
    inp, result = base.read(directory / 'input.json'), base.read(directory / 'result.json')
    require(inp['protocol'] == PROTOCOL and inp['provider_identity'] == provider_identity
            and inp['job']['request_id'] == job['request_id']
            and inp['job']['question'] == job['question'] and inp['job']['contexts'] == job['contexts']
            and outcome_valid(result) and result['request_id'] == job['request_id']
            and result['provider_identity'] == provider_identity, 'closed_outcome_binding')
    base.verify_seals(inp['boundary_seals'])
    receipt_root = directory / 'receipts'
    require({p.name for p in receipt_root.iterdir()} == {job['request_id']}, 'ambiguous_receipt_inventory')
    receipt = receipt_root / job['request_id']
    claim = base.read(receipt / 'claim.json')
    prompt = prompt_for(job['question'], job['contexts'])
    require(claim['request_id'] == job['request_id'] and claim['provider_identity'] == provider_identity
            and claim['protocol'] == TRANSPORT and digest({'provider': provider_identity, 'prompt': prompt}) == job['request_id']
            and claim['prompt_sha256'] == hashlib.sha256(prompt.encode()).hexdigest(), 'outcome_claim_binding')
    stdout_path, stderr_path = receipt / 'stdout.private.jsonl', receipt / 'stderr.private.log'
    raw, stderr = stdout_path.read_bytes(), stderr_path.read_bytes()
    try:
        answer, usage = audit_stream(raw.decode(), 0)
    except ClaudeStreamInvalid as exc:
        failure, stop = base.read(receipt / 'failure.json'), base.read(receipt / 'stream-stop.json')
        require(not (receipt / 'completed.json').exists() and failure.get('outcome') == 'terminal_invalid'
                and failure.get('code') == stop.get('code') == str(exc)
                and failure.get('automatic_retry') is False and stop.get('outcome') == 'terminal_invalid'
                and stop.get('stdout_sha256') == hashlib.sha256(raw).hexdigest()
                and stop.get('stderr_sha256') == hashlib.sha256(stderr).hexdigest()
                and stop.get('request_id') == job['request_id'] and stop.get('provider_identity') == provider_identity
                and stop.get('server_second_request_prevention_guaranteed') is False, 'invalid_stream_stop_binding')
        proof = {'outcome': 'terminal_invalid', 'code': str(exc)}
    else:
        complete = base.read(receipt / 'completed.json')
        require(not (receipt / 'failure.json').exists() and not (receipt / 'stream-stop.json').exists()
                and complete['answer'] == answer and complete['stdout_sha256'] == hashlib.sha256(raw).hexdigest()
                and all(complete['usage'].get(k) == v for k, v in usage.items())
                and complete['usage'].get('transport') == TRANSPORT
                and complete['usage'].get('request_id') == job['request_id']
                and complete['usage'].get('provider_identity') == provider_identity, 'valid_completion_binding')
        proof = {'outcome': 'valid', 'usage': complete['usage']}

    require(proof['outcome'] == result['outcome'], 'outcome_replay')
    if result['outcome'] == 'terminal_invalid':
        require(result['invalid_contribution'] == 0 and result['code'] == proof['code'], 'invalid_outcome_binding')
    else:
        require(result['usage'] == proof['usage'], 'valid_usage_binding')
    return result, {str(p): base.sha(p) for p in [directory / 'input.json', directory / 'result.json',
                   *receipt.iterdir()] if p.is_file()}


def reuse_v2(base, prior_paths, jobs, provider_identity):
    expected = {j['request_id']: j for j in jobs}
    reuse, seals = {}, {}
    for path in prior_paths:
        path = base.safe(path)
        summary, claim = base.read(path / 'summary.json'), base.read(path / 'claim.json')
        require(summary.get('protocol') == PROTOCOL and summary.get('status') == 'completed_experiment'
                and summary.get('provider_identity') == provider_identity
                and not (path / 'failure.json').exists(), 'prior_v2_not_closed')
        require(not list(path.rglob('*.partial')), 'prior_v2_ambiguous')
        workers = sorted(path.glob('worker-*'))
        require(len(workers) == summary['new_requests_dispatched'] == len(claim['dispatched_request_ids']),
                'prior_v2_worker_inventory')
        found = set()
        for directory in workers:
            key = base.read(directory / 'input.json')['job']['request_id']
            require(key in expected and key not in reuse, 'prior_v2_unknown_or_duplicate')
            _, refs = verify_outcome(base, directory, expected[key], provider_identity)
            seals.update(refs)
            reuse[key] = str(directory)
            found.add(key)
        require(found == set(claim['dispatched_request_ids']), 'prior_v2_claim_coverage')
        for key, directory in claim.get('reused_outcomes', {}).items():
            require(key in expected and (key not in reuse or reuse[key] == directory), 'prior_v2_reuse_conflict')
            _, refs = verify_outcome(base, Path(directory), expected[key], provider_identity)
            seals.update(refs)
            reuse[key] = directory

        seals.update({str(path / n): base.sha(path / n) for n in ['summary.json', 'claim.json']})
    return reuse, seals



def derived_laya(base, acceptance_path, rows, chunks):
    """Verify explicit root-accepted forensic closure; never relabel the failed run."""
    private_root = ROOT / 'work/ragturk-colab-claude-v1'
    acceptance_path = base.safe(acceptance_path)
    require(acceptance_path.is_relative_to(private_root), 'acceptance_private_root')
    acceptance = base.read(acceptance_path)
    require(acceptance.get('protocol') == 'ragturk-laya-sdk-normalization-root-acceptance-v1'
            and acceptance.get('status') == 'accepted_for_bounded_qa_pilot', 'root_acceptance_required')
    seals = {str(acceptance_path): base.sha(acceptance_path)}

    def relative(name):
        path = Path(name)
        require(not path.is_absolute() and '..' not in path.parts, 'relative_evidence_path')
        return base.safe(private_root / path)

    refs = {}
    required = {'candidate', 'derived_review', 'archive', 'spec', 'runtime', 'failed', 'original_tokenizer',
                'observed_tokenizer', 'sdk_agent', 'remote_prepared', 'models_source', 'small_seals',
                'small_supplement', 'weight_seals', 'closure'}
    require(required <= set(acceptance['files']), 'root_evidence_inventory')
    for key, ref in acceptance['files'].items():
        path = relative(ref['relativepath'])
        require(path.stat().st_size == ref['bytes'] and base.sha(path) == ref['sha256'], 'root_evidence_seal')
        refs[key] = path
        seals[str(path)] = ref['sha256']
    candidate, review, spec, runtime = (base.read(refs[k]) for k in ['candidate', 'derived_review', 'spec', 'runtime'])
    require(base.sha(refs['candidate']) == acceptance['candidate_sha256']
            and base.sha(refs['derived_review']) == acceptance['derived_review_sha256'] == candidate['review_sha256']
            and base.sha(refs['spec']) == acceptance['original_spec_sha256'] == candidate['spec_sha256'] == review['spec_sha256']
            and base.sha(refs['archive']) == candidate['archive_sha256'], 'root_proof_crossbinding')
    require(review['status'] == 'passed_scoring_artifact_integrity_pending_root_acceptance'
            and review['original_worker_status'] == candidate['original_worker_status'] == 'failed_no_automatic_retry'
            and base.read(refs['failed'])['status'] == 'failed_no_automatic_retry'
            and review['questions'] == candidate['questions'] == 20
            and review['lists'] == candidate['lists'] == 100 and review['pairs'] == candidate['pairs'] == 5000
            and review['final_inventory_expected'] == review['final_inventory_verified'] == 50
            and review['final_inventory_missing'] == 0 and review['model_reruns'] == 0
            and review['answer_generation_calls'] == review['benchmark_evaluations'] == 0,
            'derived_closure_scope')
    capture, pack = relative(acceptance['capture_relativepath']), relative(acceptance['pack_relativepath'])
    require(not (capture / 'result.json').exists(), 'failed_worker_relabel_forbidden')
    claim_path = capture / 'claim.json'
    require(base.read(claim_path)['spec_sha256'] == base.sha(refs['spec']), 'failed_claim_spec')
    seals[str(claim_path)] = base.sha(claim_path)
    docs = {r['id']: {k: r.get(k, '') for k in ['id', 'title', 'text']} for r in chunks}
    expected = [r['methods'][method] for r in rows for method in base.METHODS]
    require(len(expected) == 100 and sum(len(r['candidate_ids']) for r in expected) == 5000, 'forensic_actual_pairs')
    for key, value in [('corpus', list(docs.values())),
                       ('questions', [{'id': r['query_id'], 'question': r['question']} for r in rows]),
                       ('candidates', expected)]:
        path = pack / (key + '.jsonl')
        require(base.sha(path) == spec[key]['sha256']
                and [json.loads(line) for line in path.read_text().splitlines() if line] == value,
                'forensic_source_payload')
        seals[str(path)] = base.sha(path)
    config = spec['laya']
    require(config['model_id'] == 'convaiinnovations/laya-multilingual'
            and config['revision'] == '1720e3e3357cfe1e281542e223f8273b0890ca34'
            and config['device'] == 'cuda' and config['dtype'] == 'float32' and config['backend'] == 'sdk'
            and config['batch_size'] == 8 and config['max_len'] == 8192 and config['threshold'] is None
            and runtime['runtime']['gpu'] == review['gpu'] == 'NVIDIA L4', 'forensic_laya_condition')
    source = refs['models_source'].read_bytes()
    source_ref = next(r for r in spec['implementation'] if r['path'].endswith('/src/rag_benchmark/models.py'))
    require(hashlib.sha256(source).hexdigest() == source_ref['sha256'], 'forensic_adapter_source')
    instruction = next(ast.literal_eval(n.value) for n in ast.parse(source).body if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == 'LAYA_INSTRUCTION' for t in n.targets))
    packages = base.read(refs['remote_prepared'])['packages']
    adapter_identity = digest({'adapter': 'laya-v1', 'config': config,
        'questions': {'relevant': {'type': 'noul', 'instructions': instruction}},
        'packages': {n: packages[n] for n in ['laya', 'torch', 'transformers']}})
    require(adapter_identity == runtime['adapter_identity'] == review['adapter_identity'] == candidate['adapter_identity'],
            'forensic_adapter_identity')
    sdk = refs['sdk_agent'].read_text()
    function = next(n for n in ast.parse(sdk).body if isinstance(n, ast.FunctionDef) and n.name == '_fix_tokenizer_config')
    transition = review['sdk_compatibility_transition']
    require(all(acceptance['sdk_transition'].get(k) == transition[k] for k in
                ['original_sha256', 'after_sha256', 'function_source_sha256', 'sdk_agent_source_sha256'])
            and base.sha(refs['sdk_agent']) == transition['sdk_agent_source_sha256']
            and hashlib.sha256(ast.get_source_segment(sdk, function).encode()).hexdigest() == transition['function_source_sha256']
            and base.sha(refs['original_tokenizer']) == transition['original_sha256']
            and base.sha(refs['observed_tokenizer']) == transition['after_sha256'], 'exact_sdk_transition_seals')
    changed = base.read(refs['original_tokenizer'])
    if changed.get('tokenizer_class') in (None, 'TokenizersBackend'):
        changed['tokenizer_class'] = 'PreTrainedTokenizerFast'
        changed.pop('backend', None)
        changed.pop('is_local', None)
    if isinstance(changed.get('extra_special_tokens'), list):
        changed['extra_special_tokens'] = {f'extra_{i}': t for i, t in enumerate(changed['extra_special_tokens'])}
    require(json.dumps(changed, indent=2).encode() == refs['observed_tokenizer'].read_bytes(), 'exact_sdk_bytes')
    observations = base.read(refs['small_seals']) + base.read(refs['small_supplement']) + base.read(refs['weight_seals'])
    all_refs = spec['implementation'] + spec['model_files']
    require(len(all_refs) == 50, 'forensic_inventory_count')
    for ref in all_refs:
        actual = [v for v in observations if v['path'] == ref['path']]
        require(actual, 'forensic_missing_seal')
        target = transition['after_sha256'] if ref['path'].endswith('/tokenizer/tokenizer_config.json') else ref['sha256']
        expected_size = ref.get('size')
        if expected_size is None:
            require(Path(ref['path']).name == 'laya_worker_draft.py', 'missing_inventory_size')
            local_worker = base.safe(private_root / 'laya_worker_draft.py')
            require(base.sha(local_worker) == ref['sha256'], 'worker_source_actual_sha')
            expected_size = local_worker.stat().st_size
            seals[str(local_worker)] = ref['sha256']
        require(any(v.get('actual_sha256') == target and v.get('actual_size', v.get('size')) ==
                    (refs['observed_tokenizer'].stat().st_size if target == transition['after_sha256'] else expected_size)
                    for v in actual), 'forensic_final_seal')
        if target == transition['after_sha256']:
            require(ref['sha256'] == transition['original_sha256'], 'forensic_original_tokenizer_seal')
    closure = base.read(refs['closure'])
    require(base.sha(refs['closure']) == review['closure_observation_sha256']
            and closure.get('kernel_idle') is True and closure.get('previous_owned_child_absent') is True,
            'forensic_owned_closure')
    require(len(candidate['rows']) == 100, 'forensic_export_count')
    hashes, rankings = [], {}
    for index, original in enumerate(expected):
        path = capture / f'row-{index:08d}.json'
        record, export = base.read(path), candidate['rows'][index]
        row_sha = base.sha(path)
        hashes.append(row_sha)
        seals[str(path)] = row_sha
        ids, scores = record['ranked_ids'], record['scores']
        require(record['method'] == original['method'] and record['query_id'] == original['query_id']
                and record['input_candidate_ids'] == original['candidate_ids'] and record['input_scores'] == original['candidate_scores']
                and len(ids) == len(scores) == len(set(ids)) == len(original['candidate_ids'])
                and set(ids) == set(original['candidate_ids'])
                and all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in scores), 'forensic_row_source')
        ordinal = {sid: n for n, sid in enumerate(original['candidate_ids'])}
        require(list(zip(ids, scores)) == sorted(zip(ids, scores), key=lambda x: (-x[1], ordinal[x[0]])), 'forensic_row_order')
        request = {'question': rows[index // 5]['question'], 'candidates': [{**docs[sid], 'score': score} for sid, score in
                   zip(original['candidate_ids'], original['candidate_scores'], strict=True)],
                   'adapter_identity': adapter_identity, 'runtime': runtime['runtime']}
        require(record['request_sha256'] == hashlib.sha256(json.dumps(request, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
                and record['usage']['candidates'] == record['usage']['retained'] == len(ids)
                and record['usage']['backend'] == 'sdk' and record['usage']['dtype'] == 'float32', 'forensic_row_request')
        require(export['variant'] == original['method'] + '_laya' and export['query_id'] == original['query_id']
                and export['candidate_ids'] == original['candidate_ids'] and export['candidate_scores'] == original['candidate_scores']
                and export['ranked_ids'] == ids and export['ranked_scores'] == scores
                and export['context_ids'] == ids[:5] and export['captured_receipt_sha256'] == row_sha, 'forensic_export_binding')
        rankings[(original['query_id'], original['method'])] = ids
    inventory_sha = hashlib.sha256(json.dumps(hashes, separators=(',', ':')).encode()).hexdigest()
    require(inventory_sha == acceptance['row_receipt_inventory_sha256'] == review['row_receipt_inventory_sha256'], 'forensic_row_inventory')
    return rankings, seals

def prepare(args):
    core, base = helpers()
    cls, _ = adapter()
    gate = billing_gate_from(args.billing_evidence)
    prepared, rows, seals = base.closed_inputs(args.producer, args.bundle)
    rerankers = ['none'] if args.none_only else ['none', 'laya']
    rankings, laya_seals = ({}, {})
    if not args.none_only:
        require(args.laya_root_acceptance is not None, 'accepted_laya_closure_required')
        rankings, laya_seals = derived_laya(base, args.laya_root_acceptance, rows, prepared.chunks)
    seals.update(laya_seals)
    generator = cls(billing_gate=gate)
    all_jobs = core.jobs_for(base, rows, prepared.chunks, rankings, generator.identity, rerankers)
    reuse, refs = reuse_v2(base, args.prior_v2, all_jobs, generator.identity)
    seals.update(refs)
    selected = [j for j in all_jobs if j['request_id'] not in reuse][:args.limit_requests]
    selected_keys = {j['request_id'] for j in selected} | set(reuse)
    jobs = [j for j in all_jobs if j['request_id'] in selected_keys]
    require(jobs, 'no_selected_outcomes')
    for path in [Path(__file__).absolute(), CONTINUATION, ROOT / 'scripts/run_ragturk_claude_pilot.py',
                 ROOT / 'src/rag_benchmark/claude_cli.py', ROOT / 'src/rag_benchmark/claude_cli_stream.py',
                 base.safe(args.billing_evidence)]:
        seals[str(path)] = base.sha(path)
    model_paths = set()  # Derived closure pins observed inventory without downloading weights.
    boundary = {p: h for p, h in seals.items() if p not in model_paths}
    out = base.safe(args.output)
    require(not out.exists(), 'fresh_output_required')
    plan = {'protocol': PROTOCOL, 'model': generator.last_usage.get('model', 'claude-haiku-5-5'),
            'provider_identity': generator.identity, 'output': str(out), 'producer': str(base.safe(args.producer)),
            'bundle': str(base.safe(args.bundle)), 'rerankers': rerankers,
            'laya_spec': str(base.safe(args.laya_spec)) if args.laya_spec else None,
            'laya_output': str(base.safe(args.laya_output)) if args.laya_output else None,
            'laya_transport_proof': str(base.safe(args.laya_transport_proof)) if args.laya_transport_proof else None,
            'laya_root_acceptance': str(base.safe(args.laya_root_acceptance)) if args.laya_root_acceptance else None,
            'billing_evidence': str(base.safe(args.billing_evidence)), 'concurrency': args.concurrency,
            'scope': 'technical_bounded_subset' if len(jobs) < len(all_jobs) else 'full20_query_pilot',
            'full_logical_conditions': 100 * len(rerankers),
            'selected_logical_conditions': sum(len(j['aliases']) for j in jobs),
            'jobs': jobs, 'reuse': reuse, 'seals': seals, 'boundary_seals': boundary,
            'invalid_outcome_policy': 'terminal_invalid_counts_zero_for_every_alias_no_retry',
            'fatal_policy': 'stop_admission_and_close_owned_workers', 'v1_claim_reuse': False,
            'server_second_request_prevention_guaranteed': False}
    base.verify_seals(seals)
    base.private(args.plan, plan)
    return {'status': 'prepared', 'plan_sha256': base.sha(args.plan), 'scope': plan['scope'],
            'selected_logical_conditions': plan['selected_logical_conditions'],
            'full_logical_conditions': plan['full_logical_conditions'],
            'new_requests': len(selected), 'reused_completed_outcomes': len(reuse)}


def worker(path, expected_sha):
    _, base = helpers()
    cls, invalid = adapter()
    raw = base.safe(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected_sha, 'worker_input_sha')
    inp = json.loads(raw)
    require(inp['protocol'] == PROTOCOL, 'worker_protocol')
    base.verify_seals(inp['boundary_seals'])
    generator = cls(billing_gate=billing_gate_from(inp['billing_evidence']))
    require(generator.identity == inp['provider_identity'], 'adapter_identity_changed')
    out, job = base.safe(inp['worker_output']), inp['job']
    started = time.monotonic()
    value = {'status': 'completed_outcome', 'request_id': job['request_id'],
             'provider_identity': generator.identity, 'ttft_seconds': None, 'quality_evaluated': False}
    try:
        generator.generate(job['question'], job['contexts'], out / 'receipts')
        value.update(outcome='valid', usage=generator.last_usage)
    except invalid as exc:
        value.update(outcome='terminal_invalid', code=str(exc), invalid_contribution=0)
    except BaseException as exc:
        base.private(out / 'failure.json', {'status': 'fatal', 'error_type': type(exc).__name__, 'automatic_retry': False})
        return 1
    base.verify_seals(inp['boundary_seals'])
    value['elapsed_seconds'] = time.monotonic() - started
    base.private(out / 'result.json', value)
    verify_outcome(base, out, job, generator.identity)
    return 0


def execute(path, expected_sha):
    core, base = helpers()
    raw = base.safe(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected_sha, 'external_plan_sha')
    plan = json.loads(raw)
    require(plan['protocol'] == PROTOCOL and type(plan['concurrency']) is int
            and 1 <= plan['concurrency'] <= 5 and plan['v1_claim_reuse'] is False, 'plan_contract')
    base.verify_seals(plan['seals'])
    gate = billing_gate_from(plan['billing_evidence'])
    out = base.safe(plan['output'])
    out.mkdir(mode=0o700, parents=True, exist_ok=False)
    pending = [(i, j) for i, j in enumerate(plan['jobs']) if j['request_id'] not in plan['reuse']]
    base.private(out / 'claim.json', {'protocol': PROTOCOL, 'plan_sha256': expected_sha, 'pid': os.getpid(),
                                    'dispatched_request_ids': [j['request_id'] for _, j in pending],
                                    'reused_outcomes': plan['reuse']})
    logs, closed = [], []
    started = time.monotonic()

    def launch(index, item):
        ordinal, job = item
        gate.check()
        base.verify_seals(plan['boundary_seals'])
        directory = out / f'worker-{ordinal:03d}'
        directory.mkdir(mode=0o700)
        input_path = directory / 'input.json'
        base.private(input_path, {'protocol': PROTOCOL, 'job': job, 'boundary_seals': plan['boundary_seals'],
                                 'billing_evidence': plan['billing_evidence'], 'provider_identity': plan['provider_identity'],
                                 'worker_output': str(directory)})
        fd = os.open(directory / 'terminal.private.log', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        log = os.fdopen(fd, 'w')
        logs.append(log)
        return subprocess.Popen([str(base.PYTHON), str(Path(__file__).absolute()), '--single-worker', str(input_path),
                                 '--single-worker-sha256', base.sha(input_path)], cwd=ROOT,
                                env=base.subscription_environment(dict(os.environ)), stdout=log, stderr=log)

    def result(index, rc):
        ordinal, job = pending[index]
        value = base.read(out / f'worker-{ordinal:03d}' / 'result.json') if rc == 0 else {'outcome': 'fatal'}
        if rc == 0:
            require(value['request_id'] == job['request_id'] and outcome_valid(value), 'worker_result')
        closed.append(value)
        base.private(out / f'progress-{len(closed):03d}.json', {**counts(closed), 'closed_requests': len(closed),
                     'elapsed_seconds': time.monotonic() - started, 'ttft_seconds': None})
        # Reviewed v1 scheduler uses completed to mean that the child produced its
        # required receipt. A terminal invalid is complete experiment data here.
        return scheduler_outcome(value, rc)

    old = signal.getsignal(signal.SIGTERM)

    def interrupted(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    try:
        dispatch = base.schedule(pending, launch, concurrency=plan['concurrency'], read_result=result)
        base.private(out / 'execution-results.json', dispatch)
        require(dispatch['status'] == 'completed', 'fatal_stopped_experiment')
        outcomes = []
        for i, job in enumerate(plan['jobs']):
            directory = Path(plan['reuse'][job['request_id']]) if job['request_id'] in plan['reuse'] else out / f'worker-{i:03d}'
            value, _ = verify_outcome(base, directory, job, plan['provider_identity'])
            outcomes.append(value)
        base.verify_seals(plan['seals'])
        prepared, rows, _ = base.closed_inputs(plan['producer'], plan['bundle'])
        rankings = {}
        if 'laya' in plan['rerankers']:
            rankings, _ = derived_laya(base, plan['laya_root_acceptance'], rows, prepared.chunks)
        reconstructed = core.jobs_for(base, rows, prepared.chunks, rankings, plan['provider_identity'], plan['rerankers'])
        require([j for j in reconstructed if j['request_id'] in {x['request_id'] for x in plan['jobs']}] == plan['jobs'],
                'reconstructed_selected_jobs')
        logical = {'valid': 0, 'terminal_invalid': 0}
        for job, value in zip(plan['jobs'], outcomes, strict=True):
            logical[value['outcome']] += len(job['aliases'])
        summary = {'status': 'completed_experiment', 'protocol': PROTOCOL, 'scope': plan['scope'],
                   'plan_sha256': expected_sha, 'provider_identity': plan['provider_identity'], **counts(outcomes),
                   'logical_outcomes': logical, 'selected_logical_conditions': plan['selected_logical_conditions'],
                   'full_logical_conditions': plan['full_logical_conditions'], 'new_requests_dispatched': len(pending),
                   'reused_completed_outcomes': len(plan['reuse']), 'elapsed_seconds': time.monotonic() - started,
                   'ttft_seconds': None, 'invalid_quality_contribution': 0, 'quality_evaluated': False,
                   'server_second_request_prevention_guaranteed': False, 'automatic_retry': False,
                   'full_dataset_promoted': False}
        base.private(out / 'full-outcome-audit.json', summary)
        base.private(out / 'summary.json', summary)
        return summary
    except BaseException as exc:
        base.private(out / 'failure.json', {'status': 'fatal_stopped', 'error_type': type(exc).__name__,
                                           'automatic_retry': False, **counts(closed)})
        raise
    finally:
        signal.signal(signal.SIGTERM, old)
        for log in logs:
            log.close()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--plan-sha256')
    p.add_argument('--single-worker', type=Path)
    p.add_argument('--single-worker-sha256')
    for key in ['producer', 'bundle', 'laya-spec', 'laya-output', 'laya-transport-proof', 'laya-root-acceptance', 'billing-evidence', 'output']:
        p.add_argument('--' + key, type=Path)
    p.add_argument('--prior-v2', type=Path, action='append', default=[])
    p.add_argument('--none-only', action='store_true')
    p.add_argument('--limit-requests', type=int, default=3)
    p.add_argument('--concurrency', type=int, default=3)
    args = p.parse_args(argv)
    if args.single_worker:
        require(args.single_worker_sha256 is not None, 'worker_sha_required')
        return worker(args.single_worker, args.single_worker_sha256)
    require(args.plan is not None, 'plan_path')
    if args.execute:
        require(args.plan_sha256 is not None, 'external_plan_sha_required')
        value = execute(args.plan, args.plan_sha256)
    else:
        require(all(getattr(args, k) is not None for k in ['producer', 'bundle', 'billing_evidence', 'output']), 'prepare_paths')
        require(args.none_only or args.laya_root_acceptance is not None, 'closed_laya_required')
        require(1 <= args.limit_requests <= 200 and 1 <= args.concurrency <= 5, 'bounded_experiment')
        value = prepare(args)
    print(json.dumps(value))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (Exception, KeyboardInterrupt):
        print(json.dumps({'status': 'failed', 'private_details_omitted': True}))
        sys.exit(1)

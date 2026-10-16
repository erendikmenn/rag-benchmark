"""Bounded subscription-only QA pilot from closed, source-verified candidates.

No reranking, retrieval, gold scoring, automatic retry, or full-run promotion.
Only the owning coordinator starts worker subprocesses; each worker calls the
subscription CLI in its main thread. Private payloads never appear in argv.
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

from rag_benchmark.claude_cli import (
    BillingGate, ClaudeSubscriptionGenerator, MODEL, owned_cleanup_signals, prompt_for, subscription_environment,
)
from rag_benchmark.storage import digest

ROOT = Path(__file__).absolute().parents[1]
PYTHON = ROOT / '.venv/bin/python'
PROTOCOL = 'ragturk-claude-subscription-bounded-pilot-v1'
METHODS = ('bm25', 'bge', 'embeddinggemma', 'bm25_bge', 'bm25_embeddinggemma')


def require(ok, code):
    if not ok:
        raise ValueError(code)


def safe(path):
    path = Path(path).absolute()
    require('..' not in path.parts and not any(p.is_symlink() for p in (path, *path.parents)), 'unsafe_path')
    return path


def sha(path):
    return hashlib.sha256(safe(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(safe(path).read_bytes())


def private(path, value):
    path = safe(path)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as h:
        json.dump(value, h, ensure_ascii=False, allow_nan=False)
        h.flush()
        os.fsync(h.fileno())
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def verified_module(path):
    path = safe(path)
    raw = path.read_bytes()
    spec = importlib.util.spec_from_file_location('_pilot_candidate_reader', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    return module, hashlib.sha256(raw).hexdigest()


def gate_from(path):
    path = safe(path)
    require(path.stat().st_mode & 0o077 == 0, 'billing_evidence_not_private')
    value = read(path)
    gate = BillingGate(**{k: value[k] for k in ['usage_credits_disabled', 'evidence_source', 'observed_at_unix']})
    gate.check()
    return gate


def select_jobs(rows, chunks, provider_identity, limit):
    require(type(limit) is int and 1 <= limit <= 100, 'request_limit')
    corpus = {r['id']: r for r in chunks}
    groups = {}
    for row in rows:
        require(tuple(row['methods']) == METHODS, 'method_order')
        for method in METHODS:
            candidate = row['methods'][method]
            ids = candidate['candidate_ids']
            require(candidate['method'] == method and candidate['query_id'] == row['query_id']
                    and isinstance(ids, list) and len(ids) == len(set(ids)) and len(ids) <= 50
                    and all(source_id in corpus for source_id in ids)
                    and len(candidate['candidate_scores']) == len(ids)
                    and all(type(v) in (int, float) and math.isfinite(v) for v in candidate['candidate_scores']),
                    'candidate_identity')
            contexts = []
            for source_id in ids[:5]:
                require(source_id in corpus, 'unknown_source')
                source = corpus[source_id]
                contexts.append({k: source.get(k, '') for k in ('id', 'title', 'text')})
            prompt = prompt_for(row['question'], contexts)
            key = digest({'provider': provider_identity, 'prompt': prompt})
            if key not in groups:
                groups[key] = {'request_id': key, 'question': row['question'], 'contexts': contexts,
                               'aliases': [], 'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest()}
            groups[key]['aliases'].append({'query_id': row['query_id'], 'method': method, 'reranker': 'none'})
    return list(groups.values())[:limit]


def closed_inputs(producer, bundle):
    producer, bundle = safe(producer), safe(bundle)
    helper, helper_sha = verified_module(ROOT / 'scripts/prepare_ragturk_candidates.py')
    prepared = helper.load_prepared(bundle)
    manifest, summary, state, claim = (read(producer / n) for n in
                                     ('manifest.json', 'summary.json', 'state.json', 'claim.json'))
    require(summary['status'] == state['status'] == 'completed', 'producer_not_closed')
    identity = manifest['identity']
    require(identity == summary['identity'] == state['identity'] == claim['identity'], 'producer_identity')
    require(identity == helper.seal({k: v for k, v in manifest.items() if k != 'identity'}), 'manifest_seal')
    require(manifest['protocol'] == helper.PROTOCOL and manifest['pilot'] is True
            and manifest['queries'] == 20 and manifest['limit'] == 20
            and manifest['bundle_identity'] == prepared.identity and manifest['bundle_sha256'] == sha(bundle)
            and manifest['source_file_sha256'] == {k: v['sha256'] for k, v in prepared.refs.items()}
            and manifest['code_sha256'] == helper_sha and manifest['labels_used'] is False
            and summary['labels_used'] is False and summary['new_model_calls'] == 0
            and tuple(manifest['methods']) == tuple(summary['methods']) == METHODS
            and summary['queries'] == 20 and summary['corpus_rows'] == len(prepared.chunks)
            and manifest['corpus_order_sha256'] == helper.seal([r['id'] for r in prepared.chunks]),
            'producer_source_contract')
    require(summary['blocks'] == state['blocks'] and state['completed_rows'] == 20, 'closed_blocks')
    seals = {str(producer / n): sha(producer / n) for n in ['manifest.json', 'summary.json', 'state.json', 'claim.json']}
    seals[str(bundle)] = sha(bundle)
    seals[str(ROOT / 'scripts/prepare_ragturk_candidates.py')] = helper_sha
    seals[str(ROOT / 'src/rag_benchmark/claude_cli.py')] = sha(ROOT / 'src/rag_benchmark/claude_cli.py')
    seals[str(Path(__file__).absolute())] = sha(__file__)
    rows, offset = [], 0
    names = set()
    indices = helper.select(prepared, manifest['split'], manifest['limit'])
    for entry in summary['blocks']:
        name = entry['name']
        require(Path(name).name == name and name not in names, 'block_name')
        names.add(name)
        path = producer / 'blocks' / name
        require(sha(path) == entry['sha256'], 'block_sha')
        seals[str(path)] = entry['sha256']
        block = read(path)
        require(block['identity'] == identity and block['protocol'] == manifest['protocol']
                and block['start'] == offset and block['stop'] - offset == entry['rows']
                and len(block['rows']) == entry['rows'], 'block_shape')
        rows.extend(block['rows'])
        offset = block['stop']
    require(offset == 20 and names == {p.name for p in (producer / 'blocks').iterdir()}, 'block_inventory')
    require([r['query_id'] for r in rows] == [prepared.queries[i]['id'] for i in indices], 'query_order')
    for row, i in zip(rows, indices, strict=True):
        require(row['question'] == prepared.queries[i]['question'] and row['split'] == prepared.splits[i],
                'question_source')
    prepared.stable_inputs()
    return prepared, rows, seals


def verify_seals(seals):
    require(all(sha(path) == expected for path, expected in seals.items()), 'pinned_input_changed')


def worker(path, expected_sha):
    path = safe(path)
    raw = path.read_bytes()
    require(path.stat().st_mode & 0o077 == 0 and hashlib.sha256(raw).hexdigest() == expected_sha,
            'worker_input_seal')
    payload = json.loads(raw)
    verify_seals(payload['seals'])
    gate = gate_from(payload['billing_evidence'])
    job = payload['job']
    generator = ClaudeSubscriptionGenerator(billing_gate=gate)
    require(generator.identity == payload['provider_identity'], 'provider_changed')
    prompt = prompt_for(job['question'], job['contexts'])
    require(job['request_id'] == digest({'provider': generator.identity, 'prompt': prompt})
            and job['prompt_sha256'] == hashlib.sha256(prompt.encode()).hexdigest(), 'worker_request')
    out = safe(payload['worker_output'])
    try:
        generator.generate(job['question'], job['contexts'], out / 'receipts')
        verify_seals(payload['seals'])
        private(out / 'result.json', {'status': 'completed', 'request_id': job['request_id'],
                'usage': generator.last_usage, 'ttft_seconds': None,
                'ttft_scope': 'not measured by this CLI transport', 'answer_quality_evaluated': False})
        return 0
    except BaseException as exc:
        private(out / 'failure.json', {'status': 'failed', 'error_type': type(exc).__name__,
                                      'automatic_retry': False})
        return 1


def schedule(jobs, launch, *, concurrency, read_result, pause=time.sleep):
    require(type(concurrency) is int and 1 <= concurrency <= 5, 'concurrency_limit')
    running, results, next_index, failed = {}, [], 0, False
    try:
        while running or (next_index < len(jobs) and not failed):
            # Observe all closed workers before admitting replacements.
            for index, child in list(running.items()):
                rc = child.poll()
                if rc is not None:
                    running.pop(index)
                    result = read_result(index, rc)
                    results.append({'ordinal': index, **result})
                    failed |= rc != 0 or result.get('status') != 'completed'
            if failed:
                break
            while len(running) < concurrency and next_index < len(jobs):
                running[next_index] = launch(next_index, jobs[next_index])
                next_index += 1
            if running:
                pause(.05)
    finally:
        # Only the exact children created above; never source services or other CLIs.
        with owned_cleanup_signals():
            for child in running.values():
                if child.poll() is None:
                    child.terminate()
            deadline = time.monotonic() + 20
            for child in running.values():
                try:
                    child.wait(timeout=max(.01, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=5)
    for index in running:
        results.append({'ordinal': index, 'status': 'stopped_after_peer_failure', 'automatic_retry': False})
    return {'status': 'failed' if failed else 'completed', 'requests_dispatched': next_index,
            'results': sorted(results, key=lambda r: r['ordinal']), 'selected_requests': len(jobs)}


def run(args):
    gate = gate_from(args.billing_evidence)
    prepared, rows, seals = closed_inputs(args.producer, args.bundle)
    generator = ClaudeSubscriptionGenerator(billing_gate=gate)
    jobs = select_jobs(rows, prepared.chunks, generator.identity, args.limit_requests)
    require(jobs, 'no_jobs')
    seals[str(safe(args.billing_evidence))] = sha(args.billing_evidence)
    out = safe(args.output)
    out.mkdir(mode=0o700, parents=True, exist_ok=False)
    private(out / 'claim.json', {'protocol': PROTOCOL, 'pid': os.getpid(), 'seals': seals,
                               'provider_identity': generator.identity, 'request_ids': [j['request_id'] for j in jobs]})
    logs = []

    def launch(index, job):
        gate.check()
        verify_seals(seals)
        directory = out / f'worker-{index:03d}'
        directory.mkdir(mode=0o700)
        input_path = directory / 'input.json'
        private(input_path, {'job': job, 'seals': seals, 'billing_evidence': str(safe(args.billing_evidence)),
                            'provider_identity': generator.identity, 'worker_output': str(directory)})
        fd = os.open(directory / 'terminal.private.log', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        log = os.fdopen(fd, 'w')
        logs.append(log)
        return subprocess.Popen([str(PYTHON), str(Path(__file__).absolute()), '--single-worker', str(input_path), '--single-worker-sha256', sha(input_path)],
                                cwd=ROOT, env=subscription_environment(dict(os.environ)),
                                stdout=log, stderr=log)

    def result(index, rc):
        p = out / f'worker-{index:03d}' / 'result.json'
        if rc != 0 or not p.exists():
            return {'status': 'failed', 'returncode': rc}
        value = read(p)
        require(value.get('request_id') == jobs[index]['request_id'], 'worker_result_identity')
        return value

    old_handler = signal.getsignal(signal.SIGTERM)

    def interrupted(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    try:
        summary = schedule(jobs, launch, concurrency=args.concurrency, read_result=result)
        verify_seals(seals)
        prepared.stable_inputs()
        summary.update(protocol=PROTOCOL, model=MODEL, provider_identity=generator.identity,
                       aliases_selected=sum(len(j['aliases']) for j in jobs),
                       method_alias_counts={method: sum(a['method'] == method for j in jobs for a in j['aliases'])
                                            for method in METHODS},
                       deduplication='exact adapter identity and prompt before dispatch',
                       rerankers=['none'], answer_quality_evaluated=False, full_run_promoted=False,
                       database_writes=0)
        private(out / 'summary.json', summary)
        return 0 if summary['status'] == 'completed' else 1
    except BaseException as exc:
        private(out / 'failure.json', {'status': 'failed', 'error_type': type(exc).__name__, 'automatic_retry': False})
        return 1
    finally:
        signal.signal(signal.SIGTERM, old_handler)
        for log in logs:
            log.close()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--single-worker', type=Path)
    p.add_argument('--single-worker-sha256')
    p.add_argument('--producer', type=Path)
    p.add_argument('--bundle', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--billing-evidence', type=Path)
    p.add_argument('--limit-requests', type=int, default=3)
    p.add_argument('--concurrency', type=int, default=3)
    args = p.parse_args(argv)
    if args.single_worker:
        require(args.single_worker_sha256 is not None, 'worker_sha_required')
        return worker(args.single_worker, args.single_worker_sha256)
    require(all(getattr(args, k) is not None for k in ['producer', 'bundle', 'output', 'billing_evidence']), 'required_paths')
    require(1 <= args.limit_requests <= 100 and 1 <= args.concurrency <= 5, 'bounded_pilot')
    return run(args)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception:
        print(json.dumps({'status': 'failed_preflight', 'private_details_omitted': True}))
        sys.exit(1)

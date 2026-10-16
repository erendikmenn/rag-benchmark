"""Full 14,530-question QA ledger; explicit CPU admission and bounded dispatch.

No inferred promotion, retries, live Laya consumption, or V1/V2 claim reuse.
SQLite is private and single-writer; an interrupted claim remains blocked.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import tarfile
import time

from rag_benchmark.claude_cli import prompt_for, subscription_environment
from rag_benchmark.claude_cli_stream_v3 import billing_gate_from, ClaudeSubscriptionPolicyGenerator
from rag_benchmark.storage import digest

ROOT = Path(__file__).absolute().parents[1]
PILOT = ROOT / 'scripts/run_ragturk_claude_pilot_v3.py'
PROTOCOL = 'ragturk-claude-full-14530-cap2048-ledger-v1'
CAP = 2048
GLOBAL_LEDGER = ROOT / 'work/ragturk-colab-claude-v1/claude-full-v3-ledger-v1'
METHODS = ('bm25', 'bge', 'embeddinggemma', 'bm25_bge', 'bm25_embeddinggemma')


def require(ok, code):
    if not ok:
        raise ValueError(code)


def safe(path):
    p = Path(path).absolute()
    require('..' not in p.parts and not any(x.is_symlink() for x in [p, *p.parents]), 'unsafe_path')
    require(p.is_relative_to(ROOT), 'outside_repository')
    return p


def sha(path):
    with safe(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path):
    return json.loads(safe(path).read_bytes())


def private(path, value):
    p = safe(path)
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True) + '\n').encode()
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb') as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    fd = os.open(p.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return hashlib.sha256(raw).hexdigest()


def module(path, expected=None):
    p = safe(path)
    raw = p.read_bytes()
    require(expected is None or hashlib.sha256(raw).hexdigest() == expected, 'dependency_sha')
    spec = importlib.util.spec_from_file_location('_full_' + hashlib.sha256(raw).hexdigest(), p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    exec(compile(raw, str(p), 'exec'), mod.__dict__)
    return mod


def file_stamps(seals):
    result = {}
    for path in seals:
        p = safe(path)
        v = p.stat()
        result[str(p)] = [v.st_dev, v.st_ino, v.st_size, v.st_mtime_ns, v.st_ctime_ns]
    return result


def verify_seals(seals):
    require(all(sha(p) == h for p, h in seals.items()), 'source_changed')


def full_inputs(producer, bundle):
    """Validate closed full candidate source without retrieval/model calls."""
    helper_path = ROOT / 'scripts/prepare_ragturk_candidates.py'
    helper = module(helper_path)
    prepared = helper.load_prepared(safe(bundle))
    producer = safe(producer)
    manifest, summary, state, claim = [read(producer / n) for n in
                                     ('manifest.json', 'summary.json', 'state.json', 'claim.json')]
    ident = manifest['identity']
    require(summary['status'] == state['status'] == 'completed'
            and ident == summary['identity'] == state['identity'] == claim['identity']
            and ident == helper.seal({k: v for k, v in manifest.items() if k != 'identity'}), 'closed_source')
    require(manifest['protocol'] == helper.PROTOCOL and manifest['pilot'] is False
            and manifest['split'] == 'all' and manifest['limit'] is None
            and manifest['queries'] == summary['queries'] == 14530
            and manifest['dev_queries'] == summary['dev_queries'] == 2000
            and manifest['test_queries'] == summary['test_queries'] == 12530
            and manifest['bundle_identity'] == prepared.identity and manifest['bundle_sha256'] == sha(bundle)
            and manifest['source_file_sha256'] == {k: v['sha256'] for k, v in prepared.refs.items()}
            and manifest['code_sha256'] == sha(helper_path)
            and manifest['labels_used'] is summary['labels_used'] is False
            and summary['new_model_calls'] == 0 and tuple(manifest['methods']) == tuple(summary['methods']) == METHODS
            and manifest['corpus_order_sha256'] == helper.seal([r['id'] for r in prepared.chunks])
            and summary['corpus_rows'] == len(prepared.chunks), 'full_source_contract')
    require(summary['blocks'] == state['blocks'] and state['completed_rows'] == 14530, 'source_blocks')
    seals = {str(producer / n): sha(producer / n) for n in ('manifest.json', 'summary.json', 'state.json', 'claim.json')}
    seals.update({str(safe(bundle)): sha(bundle), str(helper_path): sha(helper_path)})
    seals.update({str(safe(path)): prepared.refs[name]['sha256'] for name, path in prepared.files.items()})
    validator = ROOT / 'scripts/ragturk_pgvector_store.py'
    seals[str(validator)] = sha(validator)
    rows, offset, names = [], 0, set()
    for entry in summary['blocks']:
        name = entry['name']
        require(Path(name).name == name and name not in names, 'block_name')
        names.add(name)
        path = producer / 'blocks' / name
        require(sha(path) == entry['sha256'], 'block_sha')
        block = read(path)
        require(block['identity'] == ident and block['protocol'] == manifest['protocol']
                and block['start'] == offset and block['stop'] == offset + entry['rows']
                and len(block['rows']) == entry['rows'], 'block_shape')
        rows.extend(block['rows'])
        offset = block['stop']
        seals[str(path)] = entry['sha256']
    require(offset == 14530 and names == {p.name for p in (producer / 'blocks').iterdir()}, 'full_inventory')
    require(manifest['selected_query_order_sha256'] == helper.seal([r['id'] for r in prepared.queries])
            and manifest['selected_split_order_sha256'] == helper.seal(list(prepared.splits)), 'source_order_seals')
    require(list(prepared.splits) == ['dev'] * 2000 + ['test'] * 12530, 'full_split_order')
    for row, q, split in zip(rows, prepared.queries, prepared.splits, strict=True):
        require(row['query_id'] == q['id'] and row['question'] == q['question'] and row['split'] == split
                and tuple(row['methods']) == METHODS, 'source_query')
    # Validates source membership/scores and constructs only allowlisted prompt fields.
    jobs_for(rows, prepared.chunks, 'validation', 'none')
    prepared.stable_inputs()
    return prepared, rows, seals


def jobs_for(rows, chunks, identity, reranker, rankings=None):
    docs = {r['id']: r for r in chunks}
    jobs = {}
    for row in rows:
        for method in METHODS:
            candidate = row['methods'][method]
            ids = candidate['candidate_ids']
            scores = candidate['candidate_scores']
            import math
            require(candidate['method'] == method and candidate['query_id'] == row['query_id']
                    and 0 < len(ids) <= 50 and len(ids) == len(set(ids)) == len(scores)
                    and all(i in docs for i in ids)
                    and all(type(x) in (int, float) and math.isfinite(x) for x in scores), 'candidate_contract')
            chosen = ids if reranker == 'none' else rankings[(row['query_id'], method)]
            require(len(chosen) == len(ids) and set(chosen) == set(ids), 'rank_membership')
            contexts = [{k: docs[i].get(k, '') for k in ('id', 'title', 'text')} for i in chosen[:5]]
            prompt = prompt_for(row['question'], contexts)
            key = digest({'provider': identity, 'prompt': prompt})
            job = jobs.setdefault(key, {'request_id': key, 'question': row['question'], 'contexts': contexts,
                                       'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(), 'aliases': []})
            job['aliases'].append({'query_id': row['query_id'], 'method': method, 'reranker': reranker,
                                   'split': row['split']})
    return list(jobs.values())


@contextmanager
def ledger(output, *, create=False):
    out = safe(output)
    if create:
        out.mkdir(mode=0o700, parents=True, exist_ok=False)
    require(out.stat().st_mode & 0o077 == 0, 'ledger_directory_private')
    lock = os.open(out / 'owner.lock', os.O_RDWR | os.O_CREAT, 0o600)
    db = None
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = out / 'ledger.sqlite3'
        require(not path.is_symlink(), 'ledger_symlink')
        if create:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            os.close(fd)
        require(path.exists() and path.stat().st_mode & 0o077 == 0, 'private_existing_ledger')
        db = sqlite3.connect(path)
        db.execute('PRAGMA journal_mode=DELETE')
        db.execute('PRAGMA synchronous=FULL')
        if create:
            db.executescript('''CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
            CREATE TABLE requests(id TEXT PRIMARY KEY,job TEXT NOT NULL,state TEXT NOT NULL,directory TEXT,proof TEXT);
            CREATE TABLE aliases(query_id TEXT,method TEXT,reranker TEXT,split TEXT,request_id TEXT,
              PRIMARY KEY(query_id,method,reranker));
            CREATE TABLE blocks(start INTEGER PRIMARY KEY,stop INTEGER NOT NULL,sha TEXT NOT NULL,proof TEXT NOT NULL);
            CREATE TABLE batches(id TEXT PRIMARY KEY,plan_sha TEXT NOT NULL,status TEXT NOT NULL);''')
        yield db
    finally:
        if db:
            db.close()
        os.close(lock)


def add_jobs(db, jobs):
    for job in jobs:
        payload = {k: v for k, v in job.items() if k != 'aliases'}
        text = json.dumps(payload, sort_keys=True)
        old = db.execute('SELECT job FROM requests WHERE id=?', (job['request_id'],)).fetchone()
        require(old is None or old[0] == text, 'request_identity_collision')
        db.execute('INSERT OR IGNORE INTO requests VALUES(?,?,?,NULL,NULL)', (job['request_id'], text, 'ready'))
        for alias in job['aliases']:
            key = (alias['query_id'], alias['method'], alias['reranker'])
            old = db.execute('SELECT request_id FROM aliases WHERE query_id=? AND method=? AND reranker=?', key).fetchone()
            require(old is not None and old[0] in (None, job['request_id']), 'alias_identity_changed')
            db.execute('UPDATE aliases SET request_id=? WHERE query_id=? AND method=? AND reranker=?', (job['request_id'], *key))


def reserve(db, request_id, directory):
    # Commit BEFORE spawn: crash/ambiguous/fatal claims can never be retried automatically.
    with db:
        require(db.execute("UPDATE requests SET state='claimed',directory=? WHERE id=? AND state='ready'",
                           (str(directory), request_id)).rowcount == 1, 'request_already_claimed')


def settle(db, request_id, outcome, proof):
    require(outcome in ('valid', 'terminal_invalid', 'fatal'), 'outcome')
    with db:
        require(db.execute("UPDATE requests SET state=?,proof=? WHERE id=? AND state='claimed'",
                           (outcome, json.dumps(proof, sort_keys=True), request_id)).rowcount == 1, 'settle_claim')


def progress(db):
    closure = db.execute("SELECT value FROM meta WHERE key='laya_final_closure'").fetchone()
    counts = dict(db.execute('SELECT state,COUNT(*) FROM requests GROUP BY state'))
    logical = {split: dict(db.execute('''SELECT COALESCE(r.state,'pending_laya'),COUNT(*) FROM aliases a
       LEFT JOIN requests r ON r.id=a.request_id WHERE a.split=? GROUP BY COALESCE(r.state,'pending_laya')''', (split,)))
               for split in ('dev', 'test')}
    return {'protocol': PROTOCOL, 'declared_logical_conditions': 145300, 'dev_logical_conditions': 20000,
            'test_logical_conditions': 125300, 'requests': counts, 'logical': logical,
            'answered_logical_conditions': sum(v.get('valid', 0) for v in logical.values()),
            'terminal_invalid_zero_quality': sum(v.get('terminal_invalid', 0) for v in logical.values()),
            'automatic_retry': False, 'quality_evaluated': False,
            'laya_final_closure_verified': closure is not None,
            'status': ('completed_full_experiment' if closure else 'all_qa_outcomes_closed_pending_final_laya_closure') if sum(sum(v.values()) for v in logical.values()) == 145300
            and all(not any(k not in ('valid', 'terminal_invalid') and n for k, n in v.items()) for v in logical.values())
            else 'partial'}


def audit_outcome(pilot, base, directory, job, identity):
    result, seals = pilot.verify_outcome(base, directory, job, identity)
    claim = read(Path(directory) / 'receipts' / job['request_id'] / 'claim.json')
    require(claim.get('max_output_tokens_requested') == CAP, 'actual_claim_cap')
    return result, seals


def pilot_gate(pilot, base, plan_path, plan_sha, audit_path, audit_sha, identity):
    require(sha(plan_path) == plan_sha and sha(audit_path) == audit_sha, 'pilot_external_pins')
    plan, audit = read(plan_path), read(audit_path)
    require(plan['protocol'] == pilot.PROTOCOL and plan['provider_identity'] == identity
            and plan['requested_output_token_cap'] == CAP and plan['full_logical_conditions'] == 200
            and plan['selected_logical_conditions'] == 200
            and audit['status'] == 'passed_independent_full_qa_pilot_metric_audit'
            and audit['protocol'] == 'ragturk-independent-qa-pilot-metrics-v3'
            and audit['requested_output_token_cap'] == CAP
            and audit['metric_source_sha256']['claude_cli_stream_v3.py'] == sha(ROOT / 'src/rag_benchmark/claude_cli_stream_v3.py')
            and audit['plan_sha256'] == plan_sha and audit['provider_identity'] == identity
            and audit['queries'] == 20 and audit['logical_conditions'] == 200
            and audit['pilot_source_sha256'] == sha(PILOT), 'full_pilot_gate')
    aliases = [alias for job in plan['jobs'] for alias in job['aliases']]
    keys = {(x['query_id'], x['method'], x['reranker']) for x in aliases}
    queries = {x['query_id'] for x in aliases}
    require(len(aliases) == len(keys) == 200 and len(queries) == 20
            and keys == {(q, method, reranker) for q in queries for method in METHODS for reranker in ('none', 'laya')},
            'pilot_actual_200_aliases')
    summary = read(Path(plan['output']) / 'summary.json')
    require(summary['status'] == 'completed_experiment' and summary['plan_sha256'] == plan_sha
            and summary['full_logical_conditions'] == summary['selected_logical_conditions'] == 200
            and summary['fatal_requests'] == 0 and summary['requested_output_token_cap'] == CAP, 'pilot_closed')
    seals = {str(safe(plan_path)): plan_sha, str(safe(audit_path)): audit_sha,
             str(safe(plan['output']) / 'summary.json'): sha(Path(plan['output']) / 'summary.json')}
    verified = {}
    for i, job in enumerate(plan['jobs']):
        directory = Path(plan['reuse'][job['request_id']]) if job['request_id'] in plan['reuse'] else Path(plan['output']) / f'worker-{i:03d}'
        value, refs = audit_outcome(pilot, base, directory, job, identity)
        verified[job['request_id']] = (job, str(directory), value, refs)
        seals.update(refs)
    verify_seals(plan['seals'])
    return verified, seals


def prepare(a):
    require(a.concurrency == 3, 'shared_concurrency_three')
    require(safe(a.output) == GLOBAL_LEDGER and not GLOBAL_LEDGER.exists(), 'fresh_fixed_global_ledger')
    pilot = module(PILOT)
    _, base = pilot.helpers()
    gate = billing_gate_from(safe(a.billing_evidence))
    generator = ClaudeSubscriptionPolicyGenerator(billing_gate=gate, max_output_tokens=CAP)
    prepared, rows, seals = full_inputs(a.producer, a.bundle)
    prior, refs = pilot_gate(pilot, base, a.pilot_plan, a.pilot_plan_sha256,
                             a.pilot_audit, a.pilot_audit_sha256, generator.identity)
    seals.update(refs)
    laya = read(a.laya_plan)
    require(sha(a.laya_plan) == a.laya_plan_sha256 and laya['questions'] == 14530
            and laya['producer_identity'] == read(Path(a.producer) / 'manifest.json')['identity']
            and laya['producer_manifest']['sha256'] == sha(Path(a.producer) / 'manifest.json')
            and any(Path(r['path']).name == Path(a.laya_worker).name and r['sha256'] == sha(a.laya_worker)
                    for r in laya['implementation'])
            and laya['seed_pilot_rows'] is False
            and laya['expected_adapter_identity'] and laya['protocol'] == 'ragturk-closed-candidates-laya-full-v2', 'future_laya_plan')
    for path in [Path(__file__).absolute(), PILOT, ROOT / 'scripts/run_ragturk_claude_pilot.py',
                 ROOT / 'scripts/run_ragturk_claude_pilot_continuation_v1.py',
                 ROOT / 'src/rag_benchmark/claude_cli.py', ROOT / 'src/rag_benchmark/claude_cli_stream.py',
                 ROOT / 'src/rag_benchmark/claude_cli_stream_v3.py', safe(a.billing_evidence), safe(a.laya_plan), safe(a.laya_worker)]:
        seals[str(path)] = sha(path)
    plan = {'protocol': PROTOCOL, 'provider_identity': generator.identity, 'cap': CAP,
            'producer': str(safe(a.producer)), 'bundle': str(safe(a.bundle)), 'output': str(safe(a.output)),
            'billing_evidence': str(safe(a.billing_evidence)), 'concurrency': 3, 'seals': seals,
            'laya_plan': str(safe(a.laya_plan)), 'laya_plan_sha256': a.laya_plan_sha256,
            'laya_worker': str(safe(a.laya_worker)), 'prior': {k: v[1] for k, v in prior.items()},
            'source_identity': prepared.identity, 'declared_logical_conditions': 145300,
            'dev_logical_conditions': 20000, 'test_logical_conditions': 125300,
            'policy': 'terminal_invalid_zero_no_retry_fatal_stop_pending_laya_distinct',
            'source_stamps': file_stamps(seals),
            'worker_seals': {p: h for p, h in seals.items() if '/src/' in p or '/scripts/' in p or p == str(safe(a.billing_evidence))}}
    verify_seals(seals)
    private(a.plan, plan)
    return {'status': 'prepared', 'plan_sha256': sha(a.plan), 'declared_logical_conditions': 145300}


def initialize(plan, raw_sha):
    pilot = module(PILOT)
    _, base = pilot.helpers()
    prepared, rows, _ = full_inputs(plan['producer'], plan['bundle'])
    with ledger(plan['output'], create=True) as db:
        with db:
            db.execute('INSERT INTO meta VALUES(?,?)', ('plan_sha256', raw_sha))
            for row in rows:
                for method in METHODS:
                    for reranker in ('none', 'laya'):
                        db.execute('INSERT INTO aliases VALUES(?,?,?,?,NULL)', (row['query_id'], method, reranker, row['split']))
            add_jobs(db, jobs_for(rows, prepared.chunks, plan['provider_identity'], 'none'))
            import_prior(db, plan, pilot, base)
    return rows


def import_prior(db, plan, pilot, base):
    for key, directory in plan['prior'].items():
        record = db.execute('SELECT job,state FROM requests WHERE id=?', (key,)).fetchone()
        if record and record[1] == 'ready':
            value, refs = audit_outcome(pilot, base, directory, json.loads(record[0]), plan['provider_identity'])
            db.execute('UPDATE requests SET state=?,directory=?,proof=? WHERE id=?',
                       (value['outcome'], directory, json.dumps(refs, sort_keys=True), key))


def load_plan(path, expected):
    raw = safe(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected, 'external_plan_sha')
    plan = json.loads(raw)
    require(plan['protocol'] == PROTOCOL and plan['cap'] == CAP and plan['concurrency'] == 3
            and safe(plan['output']) == GLOBAL_LEDGER, 'plan_contract')
    verify_seals(plan['seals'])
    return plan


def check_ledger(db, expected):
    require(db.execute('SELECT value FROM meta WHERE key=?', ('plan_sha256',)).fetchone() == (expected,), 'ledger_plan')
    closure = db.execute("SELECT value FROM meta WHERE key='laya_final_closure'").fetchone()
    if closure:
        verify_seals(json.loads(closure[0]))
    require(not db.execute("SELECT 1 FROM requests WHERE state IN ('claimed','fatal') LIMIT 1").fetchone(), 'ambiguous_or_fatal_claim_requires_review')
    require(not db.execute("SELECT 1 FROM batches WHERE status!='closed' LIMIT 1").fetchone(), 'unclosed_batch_requires_review')


def checkpoint_rankings(plan, archive, archive_sha, metadata, rows, chunks):
    """No extraction; exact externally pinned immutable 16-query checkpoint only."""
    archive_path = safe(archive)
    require(archive_path.stat().st_size <= 32_000_000, 'checkpoint_size_bound')
    archive_bytes = archive_path.read_bytes()
    require(hashlib.sha256(archive_bytes).hexdigest() == archive_sha, 'checkpoint_sha')
    info = read(metadata)
    require(info['status'] == 'closed_immutable_scoring_chunk' and info['plan_sha256'] == plan['laya_plan_sha256']
            and info['sha256'] == archive_sha and info['size'] == safe(archive).stat().st_size
            , 'checkpoint_closed')
    require(Path(info['name']).name == info['name'] and info['name'].endswith('.tar'), 'checkpoint_proof_name')
    stem = Path(info['name']).stem
    require(Path(archive).name in (info['name'], f'{stem}-{archive_sha}.tar'), 'checkpoint_local_name')
    start, stop = map(int, stem.split('-'))
    require(stem == f'{start:06d}-{stop:06d}' and start % 16 == 0 and stop == min(start + 16, 14530), 'checkpoint_range')
    expected = {f'chunks/{stem}.json', 'condition.json'} | {f'{kind}/{i:08d}.json'
                     for i in range(start * 5, stop * 5) for kind in ('rows', 'claims')}
    with tarfile.open(fileobj=io.BytesIO(archive_bytes)) as tar:
        members = tar.getmembers()
        require(len(members) == len(expected) and {m.name for m in members} == expected
                and all(m.isfile() and m.size <= 2_000_000 for m in members), 'checkpoint_members')
        data = {m.name: tar.extractfile(m).read() for m in members}
    chunk = json.loads(data[f'chunks/{stem}.json'])
    condition = json.loads(data['condition.json'])
    upstream = read(plan['laya_plan'])
    require(condition['plan_sha256'] == chunk['plan_sha256'] == plan['laya_plan_sha256']
            and condition['adapter_identity'] == upstream['expected_adapter_identity']
            and chunk['start_query'] == start and chunk['stop_query'] == stop
            and len(chunk['receipts']) == (stop-start)*5
            and len(chunk['initial_file_stamp_sha256']) == 64, 'checkpoint_condition')
    require(hashlib.sha256(data[f'chunks/{stem}.json']).hexdigest() == info['chunk_sha256'], 'chunk_sha')
    worker = module(plan['laya_worker'], plan['seals'][plan['laya_worker']])
    require(condition['runtime'] == upstream['expected_runtime'], 'checkpoint_runtime')
    docs = {r['id']: r for r in chunks}
    rankings = {}
    for n, entry in enumerate(chunk['receipts'], start * 5):
        name = f'{n:08d}.json'
        require(entry['name'] == name and hashlib.sha256(data[f'rows/{name}']).hexdigest() == entry['sha256'], 'checkpoint_row_sha_order')
        row, method = rows[n // 5], METHODS[n % 5]
        value = json.loads(data[f'rows/{name}'])
        request = worker.request_for(row, method, docs, condition['adapter_identity'], condition['runtime'])
        worker.check_receipt(value, row, method, request)
        require(json.loads(data[f'claims/{name}']) == {'plan_sha256': plan['laya_plan_sha256'],
                'request_sha256': worker.seal(request)}, 'checkpoint_claim')
        rankings[(row['query_id'], method)] = value['ranked_ids']
    return start, stop, rankings


def admit(a):
    plan = load_plan(a.plan, a.plan_sha256)
    prepared, rows, _ = full_inputs(plan['producer'], plan['bundle'])
    start, stop, rankings = checkpoint_rankings(plan, a.checkpoint, a.checkpoint_sha256, a.checkpoint_metadata,
                                                rows, prepared.chunks)
    with ledger(plan['output']) as db:
        check_ledger(db, a.plan_sha256)
        with db:
            require(not db.execute('SELECT 1 FROM blocks WHERE start=?', (start,)).fetchone(), 'block_already_admitted')
            add_jobs(db, jobs_for(rows[start:stop], prepared.chunks, plan['provider_identity'], 'laya', rankings))
            db.execute('INSERT INTO blocks VALUES(?,?,?,?)', (start, stop, a.checkpoint_sha256,
                       json.dumps({'archive': str(safe(a.checkpoint)), 'metadata': str(safe(a.checkpoint_metadata)),
                                   'metadata_sha256': sha(a.checkpoint_metadata)}, sort_keys=True)))
            pilot = module(PILOT)
            _, base = pilot.helpers()
            import_prior(db, plan, pilot, base)
        result = progress(db)
    return {**result, 'admitted_queries': stop-start}


def audit_ledger(db, plan, rows, chunks):
    """Reconstruct every admitted alias/request from sealed source and closed blocks."""
    expected = {}
    expected_jobs = {}
    def collect(jobs):
        for job in jobs:
            expected_jobs[job['request_id']] = json.dumps({k: v for k, v in job.items() if k != 'aliases'}, sort_keys=True)
            for alias in job['aliases']:
                expected[(alias['query_id'], alias['method'], alias['reranker'])] = (alias['split'], job['request_id'])
    for row in rows:
        for method in METHODS:
            expected[(row['query_id'], method, 'laya')] = (row['split'], None)
    collect(jobs_for(rows, chunks, plan['provider_identity'], 'none'))
    for start, stop, checksum, raw in db.execute('SELECT * FROM blocks ORDER BY start'):
        refs = json.loads(raw)
        require(sha(refs['metadata']) == refs['metadata_sha256'], 'block_metadata_mutated')
        actual_start, actual_stop, rankings = checkpoint_rankings(plan, refs['archive'], checksum, refs['metadata'], rows, chunks)
        require((start, stop) == (actual_start, actual_stop), 'block_range_mutated')
        collect(jobs_for(rows[start:stop], chunks, plan['provider_identity'], 'laya', rankings))
    actual = {(q, method, reranker): (split, key) for q, method, reranker, split, key in db.execute('SELECT * FROM aliases')}
    require(actual == expected, 'ledger_alias_mapping_changed')
    actual_jobs = dict(db.execute('SELECT id,job FROM requests'))
    require(actual_jobs == expected_jobs, 'ledger_source_prompt_changed')


def spawn_owned(factory, children):
    """Record ownership before delivering a signal deferred during Popen return."""
    pending = []
    previous = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    def deferred(signum, _frame):
        pending.append(signum)
    for s in previous:
        signal.signal(s, deferred)
    try:
        child = factory()
        children.append(child)
    finally:
        for s, handler in previous.items():
            signal.signal(s, handler)
    if pending:
        raise KeyboardInterrupt
    return child


def cleanup_children(base, children):
    # Registry also covers a launch interrupted before schedule() registers it.
    with base.owned_cleanup_signals():
        for child in children:
            if child.poll() is None:
                child.terminate()
        deadline = time.monotonic() + 20
        for child in children:
            try:
                child.wait(timeout=max(.01, deadline-time.monotonic()))
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)


def execute(a):
    plan = load_plan(a.plan, a.plan_sha256)
    require(type(a.limit_requests) is int and 1 <= a.limit_requests <= 145300, 'bounded_explicit_limit')
    require(a.batch_id and all(c.isalnum() or c in '-_' for c in a.batch_id), 'batch_id')
    if not a.resume:
        initialize(plan, a.plan_sha256)
    pilot = module(PILOT)
    _, base = pilot.helpers()
    out = safe(plan['output'])
    gate = billing_gate_from(plan['billing_evidence'])
    with ledger(out) as db:
        check_ledger(db, a.plan_sha256)
        prepared, source_rows, _ = full_inputs(plan['producer'], plan['bundle'])
        audit_ledger(db, plan, source_rows, prepared.chunks)
        # Validate all already completed outcome bytes before reuse/resume.
        for key, payload, state, directory, proof in db.execute("SELECT * FROM requests WHERE state IN ('valid','terminal_invalid')"):
            value, refs = audit_outcome(pilot, base, directory, json.loads(payload), plan['provider_identity'])
            require(value['outcome'] == state and refs == json.loads(proof), 'prior_outcome_mutation')
        with db:
            db.execute('INSERT INTO batches VALUES(?,?,?)', (a.batch_id, a.plan_sha256, 'running'))
        batch = out / a.batch_id
        batch.mkdir(mode=0o700)
        pending = db.execute("SELECT id,job FROM requests WHERE state='ready' ORDER BY rowid LIMIT ?", (a.limit_requests,)).fetchall()
        private(batch / 'claim.json', {'protocol': PROTOCOL, 'plan_sha256': a.plan_sha256, 'pid': os.getpid(),
                                    'request_ids': [x[0] for x in pending], 'automatic_retry': False})
        logs, children = [], []
        started = time.monotonic()

        def launch(index, item):
            key, payload = item
            gate.check()
            require(file_stamps(plan['seals']) == plan['source_stamps'], 'source_stamp_changed')
            directory = batch / f'worker-{index:06d}'
            reserve(db, key, directory)
            directory.mkdir(mode=0o700)
            inp = directory / 'input.json'
            private(inp, {'protocol': pilot.PROTOCOL, 'requested_output_token_cap': CAP,
                         'provider_identity': plan['provider_identity'], 'billing_evidence': plan['billing_evidence'],
                         'boundary_seals': plan['worker_seals'], 'job': json.loads(payload), 'worker_output': str(directory)})
            fd = os.open(directory / 'terminal.private.log', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            log = os.fdopen(fd, 'w')
            logs.append(log)
            return spawn_owned(lambda: subprocess.Popen(
                [str(base.PYTHON), str(Path(__file__).absolute()), '--single-worker', str(inp),
                 '--single-worker-sha256', sha(inp)], cwd=ROOT,
                env=subscription_environment(dict(os.environ)), stdout=log, stderr=log), children)

        def result(index, rc):
            key, payload = pending[index]
            directory = batch / f'worker-{index:06d}'
            if rc:
                settle(db, key, 'fatal', {'returncode': rc})
                return {'status': 'failed'}
            value, refs = audit_outcome(pilot, base, directory, json.loads(payload), plan['provider_identity'])
            settle(db, key, value['outcome'], refs)
            private(batch / f'progress-{index:06d}.json', {**progress(db), 'elapsed_seconds': time.monotonic()-started})
            return {'status': 'completed'}

        old = signal.getsignal(signal.SIGTERM)
        def interrupted(_s, _f):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, interrupted)
        try:
            dispatch = base.schedule(pending, launch, concurrency=3, read_result=result)
            require(dispatch['status'] == 'completed', 'fatal_stop')
            verify_seals(plan['seals'])
            with db:
                db.execute("UPDATE batches SET status='closed' WHERE id=?", (a.batch_id,))
            report = {**progress(db), 'batch_id': a.batch_id, 'plan_sha256': a.plan_sha256,
                      'dispatched_requests': len(pending), 'elapsed_seconds': time.monotonic()-started,
                      'requests_per_second': len(pending) / max(time.monotonic()-started, 1e-9),
                      'ttft_seconds': None, 'server_second_request_prevention_guaranteed': False}
            private(batch / 'summary.json', report)
            return report
        except BaseException as exc:
            private(batch / 'failure.json', {'status': 'requires_review', 'error_type': type(exc).__name__,
                                            'automatic_retry': False, 'plan_sha256': a.plan_sha256})
            raise
        finally:
            try:
                cleanup_children(base, children)
            finally:
                signal.signal(signal.SIGTERM, old)
            for log in logs:
                log.close()


def close_laya(a):
    """CPU final admission; externally pinned independent all-source scoring audit."""
    plan = load_plan(a.plan, a.plan_sha256)
    refs = {}
    for key in ('laya_final_result', 'laya_final_state', 'laya_final_audit'):
        path, expected = safe(getattr(a, key)), getattr(a, key + '_sha256')
        require(sha(path) == expected, 'laya_closure_external_sha')
        refs[str(path)] = expected
    result, state, proof = [read(getattr(a, k)) for k in ('laya_final_result', 'laya_final_state', 'laya_final_audit')]
    upstream = read(plan['laya_plan'])
    require(result['status'] == 'completed_scoring_only' and state['status'] == 'completed'
            and {k: v for k, v in state.items() if k != 'status'} == {k: v for k, v in result.items() if k != 'status'}
            and result['plan_sha256'] == plan['laya_plan_sha256'] and result['questions'] == 14530
            and result['lists'] == 72650 and result['pairs'] == upstream['pairs'], 'full_laya_result')
    require(proof['status'] == 'passed_independent_full_laya_scoring_audit'
            and proof['plan_sha256'] == plan['laya_plan_sha256']
            and proof['result_sha256'] == a.laya_final_result_sha256
            and proof['state_sha256'] == a.laya_final_state_sha256
            and proof['worker_source_sha256'] == plan['seals'][plan['laya_worker']]
            and proof['producer_identity'] == upstream['producer_identity']
            and proof['questions'] == 14530 and proof['lists'] == 72650
            and proof['source_and_model_end_seals_verified'] is True
            and proof['closed'] is True and proof['encoding_calls'] == proof['answer_generation_calls'] == 0,
            'independent_full_laya_closure')
    with ledger(plan['output']) as db:
        check_ledger(db, a.plan_sha256)
        prepared, rows, _ = full_inputs(plan['producer'], plan['bundle'])
        audit_ledger(db, plan, rows, prepared.chunks)
        blocks = list(db.execute('SELECT start,stop,proof FROM blocks ORDER BY start'))
        require([(s, e) for s, e, _ in blocks] == [(s, min(s+16, 14530)) for s in range(0, 14530, 16)], 'full_laya_block_coverage')
        expected = []
        for s, e, raw in blocks:
            metadata = read(json.loads(raw)['metadata'])
            expected.append({'name': f'{s:06d}-{e:06d}.json', 'sha256': metadata['chunk_sha256']})
        require(result['chunks'] == expected, 'final_chunk_inventory')
        with db:
            db.execute('INSERT INTO meta VALUES(?,?)', ('laya_final_closure', json.dumps(refs, sort_keys=True)))
        private(Path(plan['output']) / 'laya-final-closure.json', {'protocol': PROTOCOL, 'refs': refs,
                                                               'plan_sha256': a.plan_sha256})
        return progress(db)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--admit-checkpoint', action='store_true')
    p.add_argument('--close-laya', action='store_true')
    p.add_argument('--single-worker')
    p.add_argument('--single-worker-sha256')
    for name in ('plan', 'plan-sha256', 'producer', 'bundle', 'output', 'billing-evidence', 'pilot-plan',
                 'pilot-plan-sha256', 'pilot-audit', 'pilot-audit-sha256', 'laya-plan', 'laya-plan-sha256',
                 'laya-worker', 'checkpoint', 'checkpoint-sha256', 'checkpoint-metadata', 'batch-id',
                 'laya-final-result', 'laya-final-result-sha256', 'laya-final-state', 'laya-final-state-sha256',
                 'laya-final-audit', 'laya-final-audit-sha256'):
        p.add_argument('--'+name)
    p.add_argument('--concurrency', type=int, default=3)
    p.add_argument('--limit-requests', type=int, default=3)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    require(sum(map(bool, (a.prepare, a.execute, a.admit_checkpoint, a.close_laya, a.single_worker))) == 1, 'one_action')
    if a.single_worker:
        # Frozen pilot worker is a dependency, not rewritten or globally patched.
        return module(PILOT).worker(a.single_worker, a.single_worker_sha256)
    value = prepare(a) if a.prepare else admit(a) if a.admit_checkpoint else close_laya(a) if a.close_laya else execute(a)
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (Exception, KeyboardInterrupt) as error:
        print(json.dumps({'status': 'failed_requires_review', 'error_type': type(error).__name__}), file=sys.stderr)
        raise SystemExit(1)

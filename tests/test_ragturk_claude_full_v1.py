"""Synthetic CPU ledger, prompt and archive tests; never spawn a real child."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile

import pytest

P = Path(__file__).parents[1] / 'scripts/run_ragturk_claude_full_v1.py'
spec = importlib.util.spec_from_file_location('full_qa_tests', P)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


@pytest.fixture
def local(tmp_path, monkeypatch):
    monkeypatch.setattr(m, 'ROOT', tmp_path)
    monkeypatch.setattr(m, 'GLOBAL_LEDGER', tmp_path / 'global')
    return tmp_path


def source(n=1):
    chunks = [{'id': 's', 'title': 'Title', 'text': 'Source', 'gold': 'FORBIDDEN_GOLD'}]
    rows = [{'query_id': f'q{i}', 'question': 'Same question', 'split': 'dev',
             'answer': 'FORBIDDEN_ANSWER', 'methods': {method: {'method': method, 'query_id': f'q{i}',
              'candidate_ids': ['s'], 'candidate_scores': [1.]} for method in m.METHODS}} for i in range(n)]
    return rows, chunks


def seed(db, rows):
    db.execute('INSERT INTO meta VALUES(?,?)', ('plan_sha256', 'plan'))
    for row in rows:
        for method in m.METHODS:
            for reranker in ('none', 'laya'):
                db.execute('INSERT INTO aliases VALUES(?,?,?,?,NULL)', (row['query_id'], method, reranker, row['split']))


def test_global_prompt_dedup_aliases_and_gold_elimination():
    rows, chunks = source(2)
    jobs = m.jobs_for(rows, chunks, 'provider', 'none')
    assert len(jobs) == 1 and len(jobs[0]['aliases']) == 10
    assert 'FORBIDDEN' not in json.dumps(jobs)
    assert len(m.jobs_for(rows, chunks, 'other_provider', 'none')) == 1
    assert jobs[0]['request_id'] != m.jobs_for(rows, chunks, 'other_provider', 'none')[0]['request_id']


@pytest.mark.parametrize('mutation', ['scores', 'unknown', 'duplicate', 'method', 'empty'])
def test_candidate_mutations_rejected(mutation):
    rows, chunks = source()
    v = rows[0]['methods'][m.METHODS[0]]
    if mutation == 'scores':
        v['candidate_scores'] = [float('nan')]
    elif mutation == 'unknown':
        v['candidate_ids'] = ['missing']
    elif mutation == 'duplicate':
        v['candidate_ids'] = ['s', 's']
        v['candidate_scores'] = [1., 1.]
    elif mutation == 'method':
        v['method'] = 'wrong'
    else:
        v['candidate_ids'] = v['candidate_scores'] = []
    with pytest.raises(ValueError, match='candidate_contract'):
        m.jobs_for(rows, chunks, 'p', 'none')


def test_ledger_global_dedup_laya_pending_and_no_retry(local):
    rows, chunks = source(2)
    job = m.jobs_for(rows, chunks, 'p', 'none')[0]
    with m.ledger(local / 'out', create=True) as db:
        with db:
            seed(db, rows)
            m.add_jobs(db, [job])
        assert m.progress(db)['logical']['dev'] == {'pending_laya': 10, 'ready': 10}
        m.reserve(db, job['request_id'], local / 'receipt')
        with pytest.raises(ValueError, match='already_claimed'):
            m.reserve(db, job['request_id'], local / 'retry')
        with pytest.raises(ValueError, match='ambiguous_or_fatal'):
            m.check_ledger(db, 'plan')
        m.settle(db, job['request_id'], 'terminal_invalid', {'bound': True})
        with pytest.raises(ValueError, match='already_claimed'):
            m.reserve(db, job['request_id'], local / 'retry')
        rankings = {(r['query_id'], method): ['s'] for r in rows for method in m.METHODS}
        with db:
            m.add_jobs(db, m.jobs_for(rows, chunks, 'p', 'laya', rankings))
        assert db.execute('SELECT COUNT(*) FROM requests').fetchone()[0] == 1
        p = m.progress(db)
        assert p['terminal_invalid_zero_quality'] == 20
        assert p['answered_logical_conditions'] == 0 and p['status'] == 'partial'


def test_fatal_claim_persists_across_explicit_resume(local):
    rows, chunks = source()
    job = m.jobs_for(rows, chunks, 'p', 'none')[0]
    with m.ledger(local / 'out', create=True) as db:
        with db:
            seed(db, rows)
            m.add_jobs(db, [job])
        m.reserve(db, job['request_id'], local / 'receipt')
        m.settle(db, job['request_id'], 'fatal', {'returncode': 1})
    with m.ledger(local / 'out') as db:
        with pytest.raises(ValueError, match='ambiguous_or_fatal'):
            m.check_ledger(db, 'plan')
    with pytest.raises(FileExistsError):
        with m.ledger(local / 'out', create=True):
            pass


def test_single_writer_lock(local):
    with m.ledger(local / 'out', create=True):
        with pytest.raises(BlockingIOError):
            with m.ledger(local / 'out'):
                pass


def test_existing_batch_and_wrong_plan_block(local):
    with m.ledger(local / 'out', create=True) as db:
        with db:
            db.execute('INSERT INTO meta VALUES(?,?)', ('plan_sha256', 'plan'))
            db.execute('INSERT INTO batches VALUES(?,?,?)', ('batch', 'plan', 'running'))
        with pytest.raises(ValueError, match='ledger_plan'):
            m.check_ledger(db, 'different')
        with pytest.raises(ValueError, match='unclosed_batch'):
            m.check_ledger(db, 'plan')


def test_private_exclusive_and_no_traversal(local):
    path = local / 'proof.json'
    m.private(path, {'safe': True})
    assert path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        m.private(path, {'safe': False})
    with pytest.raises(ValueError, match='unsafe_path'):
        m.safe(local / '..' / 'outside')
    (local / 'link').symlink_to(path)
    with pytest.raises(ValueError, match='unsafe_path'):
        m.read(local / 'link')


def test_stamp_change_and_sha_change_detected(local):
    path = local / 'source'
    path.write_text('before')
    seals = {str(path): m.sha(path)}
    stamps = m.file_stamps(seals)
    path.write_text('after')
    assert stamps != m.file_stamps(seals)
    with pytest.raises(ValueError, match='source_changed'):
        m.verify_seals(seals)


def test_claim_cap_extra_independent_guard(local, monkeypatch):
    class Pilot:
        @staticmethod
        def verify_outcome(*_):
            return {'outcome': 'valid'}, {}
    directory = local / 'worker'
    receipt = directory / 'receipts' / 'request'
    receipt.mkdir(parents=True)
    path = receipt / 'claim.json'
    path.write_text(json.dumps({'max_output_tokens_requested': 512}))
    with pytest.raises(ValueError, match='actual_claim_cap'):
        m.audit_outcome(Pilot, None, directory, {'request_id': 'request'}, 'p')
    path.write_text(json.dumps({'max_output_tokens_requested': 2048}))
    assert m.audit_outcome(Pilot, None, directory, {'request_id': 'request'}, 'p')[0]['outcome'] == 'valid'


def checkpoint(local, monkeypatch, mutate=None):
    rows, docs = source(16)
    runtime = {'gpu': 'fakeL4'}
    upstream = local / 'upstream.json'
    upstream.write_text(json.dumps({'expected_runtime': runtime, 'expected_adapter_identity': 'laya'}))
    plan = {'laya_plan': str(upstream), 'laya_plan_sha256': 'sealed-plan', 'laya_worker': str(local / 'worker.py'),
            'seals': {str(local / 'worker.py'): 'pinned'}}
    class Worker:
        @staticmethod
        def request_for(row, method, docs, identity, runtime):
            return {'question': row['question'], 'ids': row['methods'][method]['candidate_ids'], 'identity': identity}
        @staticmethod
        def seal(v):
            return m.digest(v)
        @staticmethod
        def check_receipt(v, row, method, request):
            m.require(v['query_id'] == row['query_id'] and v['method'] == method
                      and v['ranked_ids'] == row['methods'][method]['candidate_ids'], 'fake_receipt_binding')
    monkeypatch.setattr(m, 'module', lambda *_: Worker)
    data, receipts = {}, []
    for i in range(80):
        row, method = rows[i // 5], m.METHODS[i % 5]
        request = Worker.request_for(row, method, {}, 'laya', runtime)
        raw = json.dumps({'query_id': row['query_id'], 'method': method, 'ranked_ids': ['s']}).encode()
        name = f'{i:08d}.json'
        data[f'rows/{name}'] = raw
        data[f'claims/{name}'] = json.dumps({'plan_sha256': 'sealed-plan', 'request_sha256': m.digest(request)}).encode()
        receipts.append({'name': name, 'sha256': hashlib.sha256(raw).hexdigest()})
    chunk = {'plan_sha256': 'sealed-plan', 'start_query': 0, 'stop_query': 16, 'receipts': receipts,
             'initial_file_stamp_sha256': 'a'*64}
    condition = {'plan_sha256': 'sealed-plan', 'adapter_identity': 'laya', 'runtime': runtime}
    if mutate == 'row_order':
        receipts.reverse()
    if mutate == 'runtime':
        condition['runtime'] = {'gpu': 'wrong'}
    if mutate == 'claim':
        data['claims/00000000.json'] = b'{}'
    data['chunks/000000-000016.json'] = json.dumps(chunk).encode()
    data['condition.json'] = json.dumps(condition).encode()
    if mutate == 'traversal':
        data['../outside'] = b'{}'
    archive = local / '000000-000016.tar'
    with tarfile.open(archive, 'w') as tar:
        for name, raw in data.items():
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            tar.addfile(info, io.BytesIO(raw))
    checksum = m.sha(archive)
    metadata = local / 'checkpoint.json'
    metadata.write_text(json.dumps({'status': 'closed_immutable_scoring_chunk', 'plan_sha256': 'sealed-plan',
            'size': archive.stat().st_size, 'name': archive.name, 'sha256': checksum,
            'chunk_sha256': hashlib.sha256(data['chunks/000000-000016.json']).hexdigest()}))
    return plan, archive, checksum, metadata, rows, docs


def test_closed_checkpoint_exact_order_mapping(local, monkeypatch):
    result = m.checkpoint_rankings(*checkpoint(local, monkeypatch))
    assert result[:2] == (0, 16) and len(result[2]) == 80


@pytest.mark.parametrize('mutation', ['row_order', 'runtime', 'claim', 'traversal'])
def test_checkpoint_mutations_fail_closed(local, monkeypatch, mutation):
    with pytest.raises((ValueError, KeyError)):
        m.checkpoint_rankings(*checkpoint(local, monkeypatch, mutation))


def test_source_module_retained_bytes(local):
    path = local / 'helper.py'
    path.write_text('VALUE=1\n')
    expected = m.sha(path)
    assert m.module(path, expected).VALUE == 1
    path.write_text('VALUE=2\n')
    with pytest.raises(ValueError, match='dependency_sha'):
        m.module(path, expected)


def test_ledger_reconstruction_detects_alias_and_prompt_mutation(local):
    rows, chunks = source()
    plan = {'provider_identity': 'p'}
    with m.ledger(local / 'out', create=True) as db:
        with db:
            seed(db, rows)
            m.add_jobs(db, m.jobs_for(rows, chunks, 'p', 'none'))
        m.audit_ledger(db, plan, rows, chunks)
        with db:
            db.execute("UPDATE aliases SET request_id='invented' WHERE reranker='none'")
        with pytest.raises(ValueError, match='alias_mapping'):
            m.audit_ledger(db, plan, rows, chunks)
        with db:
            db.execute("UPDATE aliases SET request_id=(SELECT id FROM requests) WHERE reranker='none'")
            db.execute("UPDATE requests SET job='{}'")
        with pytest.raises(ValueError, match='source_prompt'):
            m.audit_ledger(db, plan, rows, chunks)


@pytest.mark.parametrize('fail', [False, True])
def test_execute_fake_children_bounded_stop_and_durable_resume(local, monkeypatch, fail):
    from types import SimpleNamespace
    # Load only the reviewed pure scheduler, never invoke its transport.
    base_path = P.parent / 'run_ragturk_claude_pilot.py'
    spec = importlib.util.spec_from_file_location('reviewed_schedule_full_test', base_path)
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    base.PYTHON = Path('/literal/.venv/bin/python')
    actual_schedule = base.schedule
    base.schedule = lambda *args, **kw: actual_schedule(*args, **kw, pause=lambda _: None)
    rows, chunks = source(7)
    for i, row in enumerate(rows):
        row['question'] = f'Question {i}'
    plan = {'output': str(local / 'out'), 'producer': '', 'bundle': '', 'provider_identity': 'p',
            'seals': {}, 'source_stamps': {}, 'worker_seals': {}, 'billing_evidence': 'private-evidence'}
    with m.ledger(local / 'out', create=True) as db:
        with db:
            seed(db, rows)
            m.add_jobs(db, m.jobs_for(rows, chunks, 'p', 'none'))
    monkeypatch.setattr(m, 'load_plan', lambda *_: plan)
    monkeypatch.setattr(m, 'full_inputs', lambda *_: (SimpleNamespace(chunks=chunks), rows, {}))
    monkeypatch.setattr(m, 'billing_gate_from', lambda _: SimpleNamespace(check=lambda: None))
    monkeypatch.setattr(m, 'module', lambda *_: SimpleNamespace(PROTOCOL='pilot-protocol', helpers=lambda: (None, base)))
    monkeypatch.setattr(m, 'audit_outcome', lambda *_: ({'outcome': 'valid'}, {}))
    calls = []
    class Child:
        def __init__(self, rc):
            self.rc = rc
        def poll(self):
            return self.rc
        def wait(self, **_):
            return self.rc
        def terminate(self):
            self.rc = -15
        def kill(self):
            self.rc = -9
    def fake_popen(argv, **kwargs):
        assert argv[0] == '/literal/.venv/bin/python'
        assert not any('Question ' in x for x in argv)
        inp = json.loads(Path(argv[3]).read_bytes())
        assert inp['requested_output_token_cap'] == 2048
        calls.append(argv)
        return Child(1 if fail and len(calls) == 1 else 0)
    monkeypatch.setattr(m.subprocess, 'Popen', fake_popen)
    args = SimpleNamespace(plan='unused', plan_sha256='plan', limit_requests=7, batch_id='batch-1', resume=True)
    if fail:
        with pytest.raises(ValueError, match='fatal_stop'):
            m.execute(args)
        assert len(calls) == 3  # first wave only, no fourth admission
        assert (local / 'out/batch-1/failure.json').exists()
        with pytest.raises(ValueError, match='ambiguous_or_fatal'):
            m.execute(SimpleNamespace(**{**vars(args), 'batch_id': 'batch-2'}))
        assert len(calls) == 3
    else:
        result = m.execute(args)
        assert len(calls) == 7 and result['dispatched_requests'] == 7
        assert result['logical']['dev'] == {'pending_laya': 35, 'valid': 35}
        result = m.execute(SimpleNamespace(**{**vars(args), 'batch_id': 'batch-2'}))
        assert result['dispatched_requests'] == 0 and len(calls) == 7


def test_output_fixed_global_namespace_rejects_new_claim_bypass(local):
    from types import SimpleNamespace
    with pytest.raises(ValueError, match='fresh_fixed_global_ledger'):
        m.prepare(SimpleNamespace(concurrency=3, output=str(local / 'another-run')))


@pytest.mark.parametrize('bad', ['v2', 'cap', 'adapter', 'aliases'])
def test_full_promotion_rejects_unbound_or_v2_pilot(local, monkeypatch, bad):
    from types import SimpleNamespace
    plan_path, audit_path = local / 'plan.json', local / 'audit.json'
    adapter = local / 'src/rag_benchmark/claude_cli_stream_v3.py'
    adapter.parent.mkdir(parents=True)
    adapter.write_text('adapter')
    monkeypatch.setattr(m, 'PILOT', local / 'pilot.py')
    m.PILOT.write_text('pilot')
    plan = {'protocol': 'v3', 'provider_identity': 'provider', 'requested_output_token_cap': 2048,
            'full_logical_conditions': 200, 'selected_logical_conditions': 200, 'jobs': []}
    audit = {'status': 'passed_independent_full_qa_pilot_metric_audit', 'protocol': 'ragturk-independent-qa-pilot-metrics-v3',
             'requested_output_token_cap': 2048, 'metric_source_sha256': {'claude_cli_stream_v3.py': m.sha(adapter)}, 'provider_identity': 'provider', 'queries': 20,
             'logical_conditions': 200, 'pilot_source_sha256': m.sha(m.PILOT)}
    if bad == 'v2':
        audit['protocol'] = 'ragturk-independent-qa-pilot-metrics-v2'
    elif bad == 'cap':
        audit['requested_output_token_cap'] = 512
    elif bad == 'adapter':
        audit['metric_source_sha256']['claude_cli_stream_v3.py'] = 'wrong'
    plan_path.write_text(json.dumps(plan))
    audit['plan_sha256'] = m.sha(plan_path)
    audit_path.write_text(json.dumps(audit))
    with pytest.raises(ValueError, match='pilot_actual_200_aliases' if bad == 'aliases' else 'full_pilot_gate'):
        m.pilot_gate(SimpleNamespace(PROTOCOL='v3'), None, plan_path, m.sha(plan_path), audit_path, m.sha(audit_path), 'provider')


def test_all_outcomes_cannot_claim_complete_without_final_laya_closure(local):
    with m.ledger(local / 'out', create=True) as db:
        with db:
            db.execute('INSERT INTO requests VALUES(?,?,?,?,?)', ('one', '{}', 'valid', 'receipt', '{}'))
            # Synthetic full denominator; no prompts/source IDs are published.
            db.executemany('INSERT INTO aliases VALUES(?,?,?,?,?)',
                           [(f'q{i}', str(j), r, 'dev' if i < 2000 else 'test', 'one')
                            for i in range(14530) for j in range(5) for r in ('none', 'laya')])
        result = m.progress(db)
        assert result['answered_logical_conditions'] == 145300
        assert result['status'] == 'all_qa_outcomes_closed_pending_final_laya_closure'
        assert result['laya_final_closure_verified'] is False


def test_receiver_hashed_archive_with_preserved_raw_proof(local, monkeypatch):
    args = list(checkpoint(local, monkeypatch))
    archive, checksum, proof = args[1:4]
    renamed = archive.with_name(f'{archive.stem}-{checksum}.tar')
    archive.rename(renamed)
    saved_proof = proof.with_name(f'000000-000016-{checksum}.proof.json')
    proof.rename(saved_proof)
    args[1], args[3] = renamed, saved_proof
    assert len(m.checkpoint_rankings(*args)[2]) == 80
    # Transport status proves copying, not scoring closure.
    saved_proof.write_text(json.dumps({'status': 'transport_verified_closed_checkpoint'}))
    with pytest.raises((KeyError, ValueError)):
        m.checkpoint_rankings(*args)


def test_receiver_wrong_hashed_filename_rejected(local, monkeypatch):
    args = list(checkpoint(local, monkeypatch))
    archive = args[1]
    renamed = archive.with_name(f'{archive.stem}-wrong.tar')
    archive.rename(renamed)
    args[1] = renamed
    with pytest.raises(ValueError, match='checkpoint_local_name'):
        m.checkpoint_rankings(*args)


def test_spawn_signal_window_child_registered_before_interrupt(monkeypatch):
    handlers = {m.signal.SIGINT: object(), m.signal.SIGTERM: object()}
    originals = dict(handlers)
    monkeypatch.setattr(m.signal, 'getsignal', lambda s: handlers[s])
    monkeypatch.setattr(m.signal, 'signal', lambda s, h: handlers.__setitem__(s, h))
    children = []
    child = object()
    def factory():
        handlers[m.signal.SIGTERM](m.signal.SIGTERM, None)
        return child
    with pytest.raises(KeyboardInterrupt):
        m.spawn_owned(factory, children)
    assert children == [child] and handlers == originals


def test_cleanup_registry_covers_unregistered_live_child(monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace
    events = []
    class Child:
        def poll(self):
            return None
        def terminate(self):
            events.append('terminate')
        def wait(self, timeout):
            events.append('wait')
            if events.count('wait') == 1:
                raise m.subprocess.TimeoutExpired('fake', timeout)
        def kill(self):
            events.append('kill')
    m.cleanup_children(SimpleNamespace(owned_cleanup_signals=nullcontext), [Child()])
    assert events == ['terminate', 'wait', 'kill', 'wait']

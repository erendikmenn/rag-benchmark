"""CPU-only v2 outcome/reuse policy; synthetic streams, no real children."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from rag_benchmark.claude_cli import MODEL, prompt_for
from rag_benchmark.claude_cli_stream import PROTOCOL as TRANSPORT, ClaudeStreamInvalid, audit_stream
from rag_benchmark.storage import digest

spec = importlib.util.spec_from_file_location('pilot_v2_tests', Path(__file__).parents[1] / 'scripts/run_ragturk_claude_pilot_v2.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def job():
    question = 'Synthetic question'
    contexts = [{'id': 's', 'title': '', 'text': 'Synthetic source'}]
    prompt = prompt_for(question, contexts)
    return {'question': question, 'contexts': contexts, 'request_id': digest({'provider': 'v2', 'prompt': prompt}),
            'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(), 'aliases': [{'method': 'bm25', 'reranker': 'none'}]}


def streams(terminal=False):
    usage = {'input_tokens': 2, 'output_tokens': 4, 'cache_read_input_tokens': 0, 'cache_creation_input_tokens': 0}
    result = {'type': 'result', 'subtype': 'success', 'is_error': False, 'num_turns': 1,
              'stop_reason': 'end_turn', 'result': 'Synthetic answer', 'usage': usage, 'total_cost_usd': .001,
              'modelUsage': {MODEL: {'inputTokens': 2, 'outputTokens': 4, 'cacheReadInputTokens': 0,
                                     'cacheCreationInputTokens': 0}}}
    rows = [{'type': 'system', 'subtype': 'init', 'model': MODEL, 'tools': [], 'mcp_servers': []},
            {'type': 'stream_event', 'event': {'type': 'message_start', 'message': {'id': 'm', 'model': MODEL}}}]
    if terminal:
        rows += [{'type': 'stream_event', 'event': {'type': 'message_delta',
                  'delta': {'stop_reason': 'max_tokens'}, 'usage': {'output_tokens': 512}}}]
    else:
        rows += [{'type': 'assistant', 'message': {'id': 'm', 'model': MODEL, 'stop_reason': None,
                   'content': [{'type': 'text', 'text': 'Synthetic answer'}]}}, result]
    return '\n'.join(json.dumps(e) for e in rows)


def receipt(base, directory, terminal=False):
    j = job()
    directory.mkdir()
    root = directory / 'receipts' / j['request_id']
    root.mkdir(parents=True)
    raw = streams(terminal)
    base.private(directory / 'input.json', {'protocol': m.PROTOCOL, 'provider_identity': 'v2',
                                          'job': j, 'boundary_seals': {}})
    base.private(root / 'claim.json', {'request_id': j['request_id'], 'provider_identity': 'v2',
                 'protocol': TRANSPORT, 'prompt_sha256': j['prompt_sha256']})
    (root / 'stdout.private.jsonl').write_text(raw)
    (root / 'stderr.private.log').write_text('')
    value = {'status': 'completed_outcome', 'provider_identity': 'v2', 'request_id': j['request_id']}
    if terminal:
        value.update(outcome='terminal_invalid', code='terminal_stop_max_tokens', invalid_contribution=0)
        base.private(root / 'failure.json', {'code': value['code'], 'outcome': 'terminal_invalid', 'automatic_retry': False})
        base.private(root / 'stream-stop.json', {'code': value['code'], 'outcome': 'terminal_invalid',
                     'request_id': j['request_id'], 'provider_identity': 'v2',
                     'stdout_sha256': hashlib.sha256(raw.encode()).hexdigest(),
                     'stderr_sha256': hashlib.sha256(b'').hexdigest(),
                     'server_second_request_prevention_guaranteed': False})
    else:
        _, usage = audit_stream(raw, 0)
        usage.update(transport=TRANSPORT, request_id=j['request_id'], provider_identity='v2', seconds=1.)
        value.update(outcome='valid', usage=usage)
        base.private(root / 'completed.json', {'answer': 'Synthetic answer', 'usage': usage,
                     'stdout_sha256': hashlib.sha256(raw.encode()).hexdigest()})
    base.private(directory / 'result.json', value)
    return j, value, root


@pytest.mark.parametrize('terminal', [False, True])
def test_completed_valid_and_terminal_invalid_replay_without_inference(tmp_path, terminal):
    _, base = m.helpers()
    j, value, _ = receipt(base, tmp_path / 'worker-000', terminal)
    actual, seals = m.verify_outcome(base, tmp_path / 'worker-000', j, 'v2')
    assert actual == value
    assert m.outcome_valid(actual)
    base.verify_seals(seals)
    assert m.counts([actual]) == {'valid_requests': int(not terminal),
                               'terminal_invalid_requests': int(terminal), 'fatal_requests': 0}


@pytest.mark.parametrize('mutation', ['rawsha', 'pretend_valid', 'nonzero_contribution', 'unknown_claim'])
def test_terminal_invalid_cannot_be_laundered_as_valid_or_retried(tmp_path, mutation):
    _, base = m.helpers()
    j, value, root = receipt(base, tmp_path / 'worker-000', True)
    if mutation == 'rawsha':
        stop = base.read(root / 'stream-stop.json')
        stop['stdout_sha256'] = '0' * 64
        (root / 'stream-stop.json').write_text(json.dumps(stop))
    elif mutation == 'pretend_valid':
        value['outcome'] = 'valid'
    elif mutation == 'nonzero_contribution':
        value['invalid_contribution'] = 1
    else:
        (root.parent / 'unexpected').mkdir()
    (tmp_path / 'worker-000/result.json').write_text(json.dumps(value))
    with pytest.raises((ValueError, KeyError)):
        m.verify_outcome(base, tmp_path / 'worker-000', j, 'v2')


def test_first_max_tokens_vetoes_later_synthetic_success():
    with pytest.raises(ClaudeStreamInvalid, match='terminal_stop_max_tokens'):
        audit_stream(streams(True) + '\n' + streams(False), 0)


@pytest.mark.parametrize('status,protocol', [('failed', m.PROTOCOL), ('completed', 'v1'),
                                           ('completed_experiment', 'v1')])
def test_failed_or_v1_history_never_reused(tmp_path, status, protocol):
    _, base = m.helpers()
    base.private(tmp_path / 'summary.json', {'status': status, 'protocol': protocol, 'provider_identity': 'v2'})
    base.private(tmp_path / 'claim.json', {})
    with pytest.raises(ValueError, match='prior_v2_not_closed'):
        m.reuse_v2(base, [tmp_path], [job()], 'v2')


def test_terminal_invalid_completed_v2_is_reused_without_new_call(tmp_path):
    _, base = m.helpers()
    prior = tmp_path / 'prior'
    prior.mkdir()
    j, _, _ = receipt(base, prior / 'worker-000', True)
    base.private(prior / 'summary.json', {'status': 'completed_experiment', 'protocol': m.PROTOCOL,
                                        'provider_identity': 'v2', 'new_requests_dispatched': 1})
    base.private(prior / 'claim.json', {'dispatched_request_ids': [j['request_id']]})
    reused, seals = m.reuse_v2(base, [prior], [j], 'v2')
    assert reused == {j['request_id']: str(prior / 'worker-000')}
    base.verify_seals(seals)


def test_diagnostic_and_full_scope_do_not_drop_invalid_aliases():
    outcomes = [{'status': 'completed_outcome', 'outcome': 'valid'},
                {'status': 'completed_outcome', 'outcome': 'terminal_invalid'}, {'outcome': 'fatal'}]
    assert [m.outcome_valid(x) for x in outcomes] == [True, True, False]
    assert m.counts(outcomes) == {'valid_requests': 1, 'terminal_invalid_requests': 1, 'fatal_requests': 1}


def test_wrong_external_plan_sha_fails_before_claim(tmp_path):
    path = tmp_path / 'plan.json'
    path.write_text('{}')
    with pytest.raises(ValueError, match='external_plan_sha'):
        m.execute(path, '0' * 64)
    assert list(tmp_path.iterdir()) == [path]


def test_scheduler_continues_after_terminal_invalid_but_stops_on_fatal():
    _, base = m.helpers()

    class Child:
        def poll(self):
            return 0

    terminal = {'status': 'completed_outcome', 'outcome': 'terminal_invalid'}
    valid = {'status': 'completed_outcome', 'outcome': 'valid'}
    result = base.schedule([{}] * 5, lambda i, j: Child(), concurrency=3,
        read_result=lambda i, rc: m.scheduler_outcome(terminal if i == 0 else valid, rc), pause=lambda _: None)
    assert result['status'] == 'completed'
    assert result['requests_dispatched'] == 5
    fatal = {'status': 'failed', 'outcome': 'fatal'}
    result = base.schedule([{}] * 5, lambda i, j: Child(), concurrency=1,
        read_result=lambda i, rc: m.scheduler_outcome(fatal, rc), pause=lambda _: None)
    assert result['status'] == 'failed'
    assert result['requests_dispatched'] == 1


def test_ambiguous_claim_cannot_be_reused_even_with_matching_success(tmp_path):
    _, base = m.helpers()
    prior = tmp_path / 'prior'
    prior.mkdir()
    j, _, _ = receipt(base, prior / 'worker-000', False)
    base.private(prior / 'summary.json', {'status': 'completed_experiment', 'protocol': m.PROTOCOL,
                                        'provider_identity': 'v2', 'new_requests_dispatched': 2})
    base.private(prior / 'claim.json', {'dispatched_request_ids': [j['request_id'], 'ambiguous']})
    with pytest.raises(ValueError, match='prior_v2_worker_inventory'):
        m.reuse_v2(base, [prior], [j], 'v2')


def test_durable_user_setting_gate_does_not_invent_authorization_expiry(tmp_path):
    import time
    path = tmp_path / 'billing.json'
    _, base = m.helpers()
    base.private(path, {'usage_credits_disabled': True, 'evidence_source': 'user_confirmed_usage_settings',
                        'observed_at_unix': time.time() - 86400})
    m.billing_gate_from(path).check()


def test_stale_live_usage_observation_fails_closed(tmp_path):
    import time
    from rag_benchmark.claude_cli import ClaudeTransportError
    path = tmp_path / 'billing.json'
    _, base = m.helpers()
    base.private(path, {'usage_credits_disabled': True, 'evidence_source': 'claude_usage_live',
                        'observed_at_unix': time.time() - 7200})
    with pytest.raises(ClaudeTransportError, match='fresh_live_usage_observation_required'):
        m.billing_gate_from(path)


@pytest.mark.parametrize('info,code', [({'isUsingOverage': True}, 'billing_overage_observed'),
                                     ({'isUsingOverage': False, 'overageStatus': 'allowed'},
                                      'usage_credits_policy_changed')])
def test_observed_billing_transition_is_fatal_never_terminal_invalid(info, code):
    from rag_benchmark.claude_cli import ClaudeTransportError
    from rag_benchmark.claude_cli_stream import StreamGuard
    with pytest.raises(ClaudeTransportError, match=code) as error:
        StreamGuard().observe({'type': 'rate_limit_event', 'rate_limit_info': info})
    assert not isinstance(error.value, ClaudeStreamInvalid)


@pytest.mark.parametrize('mutation', ['generic_boolean', 'absolute_ref', 'dotdot_ref', 'wrong_size'])
def test_derived_laya_requires_narrow_root_proof_and_confined_bytes(monkeypatch, tmp_path, mutation):
    _, base = m.helpers()
    monkeypatch.setattr(m, 'ROOT', tmp_path)
    private_root = tmp_path / 'work/ragturk-colab-claude-v1'
    private_root.mkdir(parents=True)
    evidence = private_root / 'synthetic.json'
    base.private(evidence, {})
    ref = {'relativepath': 'synthetic.json', 'sha256': base.sha(evidence), 'bytes': evidence.stat().st_size}
    roles = ['candidate', 'derived_review', 'archive', 'spec', 'runtime', 'failed', 'original_tokenizer',
             'observed_tokenizer', 'sdk_agent', 'remote_prepared', 'models_source', 'small_seals',
             'small_supplement', 'weight_seals', 'closure']
    acceptance = {'protocol': 'ragturk-laya-sdk-normalization-root-acceptance-v1',
                  'status': 'accepted_for_bounded_qa_pilot', 'files': {k: dict(ref) for k in roles}}
    if mutation == 'generic_boolean':
        acceptance = {'passed': True}
    elif mutation == 'absolute_ref':
        acceptance['files']['candidate']['relativepath'] = str(evidence)
    elif mutation == 'dotdot_ref':
        acceptance['files']['candidate']['relativepath'] = '../synthetic.json'
    else:
        acceptance['files']['candidate']['bytes'] = 999
    path = private_root / 'acceptance.json'
    base.private(path, acceptance)
    with pytest.raises(ValueError):
        m.derived_laya(base, path, [], [])

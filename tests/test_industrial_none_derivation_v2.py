"""Only new reboot/CPU handoff seams; tiny metadata, no actual data or inference."""
import ast
import copy
import json
from pathlib import Path
import subprocess

import pytest

from rag_benchmark import industrial_none_derivation_v2 as run

ROOT = Path(__file__).resolve().parents[1]


def pin(root, path, value):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(value if isinstance(value, bytes) else run.frozen.canonical(value))
    return run.frozen.record(root, path)


def load(root, ref):
    return json.loads((root / ref['path']).read_bytes())


def repin(root, ref, value):
    ref.update(pin(root, ref['path'], value))


def descriptor(path, size=256):
    return {'path': path, 'bytes': size, 'sha256': 'a' * 64}


def stage_fixture(root, channel='J'):
    original = descriptor(run.upstream.PARENT_PATH)
    binding = descriptor('proof/binding.json')
    envelope = {'original_manifest': original, 'rebinding': binding, 'source_pins': {}}
    digest = run.frozen.digest(envelope)
    identity = run.upstream.stage_identity(envelope, digest, channel)
    corpus, queries = [{'id': 'd0', 'text': ''}], [{'id': 'q0', 'text': 'synthetic'}]
    stage = run.upstream.DELTA + '/' + channel
    execution = {'protocol': run.upstream.VERSION, 'status': 'PASS_ROOT_INDUSTRIAL_REBOOT_EXECUTION_AUTHORIZED',
        'actual_model_execution_authorized': True, 'channel': channel, 'manifest_sha256': digest,
        'source_pins': {}, 'rebinding': binding, 'boot_epoch_seconds': run.upstream.BOOT_EPOCH}
    execution_ref = pin(root, 'proof/' + channel + '-execution.json', execution)
    claim = {'protocol': run.upstream.VERSION, 'channel': channel, 'manifest_sha256': digest,
        'execution_admission': execution_ref, 'boot_epoch_seconds': run.upstream.BOOT_EPOCH, 'pid': 12345}
    files = {'execution.claim.json': pin(root, stage + '/execution.claim.json', claim),
        'scoring.claim.json': pin(root, stage + '/scoring.claim.json',
            {'identity': identity, 'manifest': digest, 'query_count': 1, 'top_k': 100}),
        'report.json': pin(root, stage + '/report.json', {'condition': run.upstream.VERSION,
            'manifest_sha256': digest, 'dataset': {'identity': run.origin.DATASET_IDENTITY},
            'hardware': run.origin.HARDWARE, 'channels': {channel: {'status': 'completed', 'identity': identity}},
            'original_manifest': original, 'rebinding': binding, 'execution_admission': execution_ref,
            'none_cells_completed': 0, 'laya_cells_completed': 0}),
        'rankings.json': pin(root, stage + '/rankings.json', {'identity': identity, 'condition': run.upstream.VERSION,
            'manifest_sha256': digest, 'dataset_identity': run.origin.DATASET_IDENTITY,
            'hardware': run.origin.HARDWARE[channel]})}
    terminals = []
    for role, row in [('document', corpus[0]), ('query', queries[0])]:
        name = stage + '/' + role + '/000000'
        job = {'manifest': digest, 'channel': channel, 'role': role, 'index': 0,
               'row_sha256': run.frozen.digest(row), 'channel_identity': identity}
        job_ref = pin(root, name + '.claim.json', job)
        usage = {'inference_seconds': 1.2, 'dtype': 'bfloat16', 'context_limit': 8192}
        if channel == 'J':
            usage.update(devices=['mps'], towers={'vision': True, 'audio': True}, items=[{
                'modalities': ['image', 'text'] if role == 'document' else ['text'],
                'expanded_tokens': 1, 'derived_text_empty': not row['text'].strip()}])
        else:
            usage.update(items=1, device='mps', max_image_patches=768, token_counts=[1])
        terminal = {'status': 'completed', 'claim_sha256': job_ref['sha256'], 'usage': usage,
            'output': descriptor(name + '.npy')}
        if channel == 'S':
            terminal['projection_verification'] = {'projection_verified': True, 'max_absolute_merge_error': 0.0}
        terminals.append(pin(root, name + '.terminal.json', terminal))
    closed = {'protocol': run.upstream.VERSION, 'status': 'completed', 'manifest_sha256': digest,
        'identity': identity, 'original_manifest': original, 'rebinding': binding, 'execution_admission': execution_ref,
        'files': files, 'terminal_records': terminals, 'terminal_sha256': run.frozen.digest(terminals)}
    closed_ref = pin(root, run.STAGE_CLOSED[channel], closed)
    wait = {'actual_wait': True, 'exit_code': 0, 'signal': None, 'channel': channel,
            'manifest_sha256': digest, 'pid': claim['pid'], 'execution_claim': files['execution.claim.json'],
            'closed': closed_ref}
    refs = {'original_manifest': original, 'boot_rebinding': binding,
            channel.lower() + '_execution_admission': execution_ref,
            channel.lower() + '_execution_claim': files['execution.claim.json'],
            channel.lower() + '_actual_wait': pin(root, 'proof/' + channel + '-wait.json', wait)}
    return envelope, refs, closed_ref, corpus, queries


def check_stage(root, fixture, channel='J'):
    envelope, refs, closed, corpus, queries = fixture
    proofs = run.Proofs(root)
    result = run.stage_bundle(proofs, envelope, channel, refs, closed, corpus, queries)
    return proofs, result


def test_old_namespace_red_and_new_cpu_stage_green_without_vector_files(tmp_path):
    fixture = stage_fixture(tmp_path)
    # The old accepted seam cannot silently alias a reboot V3 stage.
    with pytest.raises(ValueError):
        run.previous.stage_bundle(tmp_path, {'dataset_identity': run.origin.DATASET_IDENTITY},
                                  run.frozen.digest(fixture[0]), 'J', fixture[2])
    proofs, result = check_stage(tmp_path, fixture)
    assert result[0]['closed'] == fixture[2]
    assert len(proofs.descriptors) == 2
    assert not list(tmp_path.rglob('*.npy'))
    assert all(not p.endswith('.npy') for p in proofs.refs)
    proofs.fresh()


@pytest.mark.parametrize('channel', ['J', 'S'])
def test_terminal_population_is_row_based_and_records_usage_without_payload_read(tmp_path, channel):
    proofs, result = check_stage(tmp_path, stage_fixture(tmp_path, channel), channel)
    assert len(proofs.descriptors) == 2
    assert result[1]['channels'][channel]['status'] == 'completed'
    assert all(not (tmp_path / p).exists() for p in proofs.descriptors)
    # Frozen encoder writes one vector and terminal per row for either model.
    node = next(n for n in ast.parse((ROOT / 'src/rag_benchmark/industrial_mps_sj_v2.py').read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == 'encode_jobs')
    assert any(isinstance(n, ast.For) and ast.unparse(n.iter) == 'enumerate(rows)' for n in ast.walk(node))


@pytest.mark.parametrize('damage', ['missing', 'duplicate', 'outside', 'failed', 'claim_sha', 'row_sha',
    'vector_path', 'vector_size', 'wrong_modality', 'wrong_empty', 'wrong_dtype', 'bad_projection'])
def test_nested_terminal_damage_is_rejected_before_science(tmp_path, damage):
    channel = 'S' if damage == 'bad_projection' else 'J'
    fixture = stage_fixture(tmp_path, channel)
    envelope, refs, closed_ref, corpus, queries = fixture
    closed = load(tmp_path, closed_ref)
    terminal_ref = closed['terminal_records'][0]
    terminal = load(tmp_path, terminal_ref)
    if damage == 'missing':
        closed['terminal_records'].pop()
    elif damage == 'duplicate':
        closed['terminal_records'][1] = copy.deepcopy(terminal_ref)
    elif damage == 'outside':
        terminal_ref['path'] = 'different/terminal.json'
    elif damage == 'failed':
        terminal['status'] = 'failed'
    elif damage == 'claim_sha':
        terminal['claim_sha256'] = '0' * 64
    elif damage == 'row_sha':
        corpus[0]['text'] = 'changed'
    elif damage == 'vector_path':
        terminal['output']['path'] = run.previous.OUTPUT + '/vector.npy'
    elif damage == 'vector_size':
        terminal['output']['bytes'] = 10**12
    elif damage == 'wrong_modality':
        terminal['usage']['items'][0]['modalities'] = ['text']
    elif damage == 'wrong_empty':
        terminal['usage']['items'][0]['derived_text_empty'] = False
    elif damage == 'wrong_dtype':
        terminal['usage']['dtype'] = 'float32'
    else:
        terminal['projection_verification']['projection_verified'] = False
    if damage not in ('missing', 'duplicate', 'outside'):
        repin(tmp_path, terminal_ref, terminal)
    closed['terminal_sha256'] = run.frozen.digest(closed['terminal_records'])
    repin(tmp_path, closed_ref, closed)
    wait_ref = refs[channel.lower() + '_actual_wait']
    wait = load(tmp_path, wait_ref)
    wait['closed'] = closed_ref
    repin(tmp_path, wait_ref, wait)
    with pytest.raises(ValueError):
        check_stage(tmp_path, fixture, channel)


@pytest.mark.parametrize('damage', ['pid', 'exit', 'unknown_exit', 'signal', 'not_actual', 'manifest', 'closed',
    'claim_namespace', 'claim_boot', 'claim_pid_bool', 'ranking_protocol', 'report_device', 'scoring_count'])
def test_true_wait_claim_and_stage_provenance_are_required(tmp_path, damage):
    fixture = stage_fixture(tmp_path)
    _, refs, closed_ref, _, _ = fixture
    closed = load(tmp_path, closed_ref)
    wait_ref = refs['j_actual_wait']
    wait = load(tmp_path, wait_ref)
    if damage in ('pid', 'exit', 'unknown_exit', 'signal', 'not_actual', 'manifest', 'closed'):
        key, value = {'pid': ('pid', 12346), 'exit': ('exit_code', 1), 'unknown_exit': ('exit_code', None),
            'signal': ('signal', 9), 'not_actual': ('actual_wait', False),
            'manifest': ('manifest_sha256', '0' * 64), 'closed': ('closed', descriptor('old/closed.json'))}[damage]
        wait[key] = value
    else:
        name = 'execution.claim.json' if damage.startswith('claim_') else (
            'rankings.json' if damage == 'ranking_protocol' else 'report.json' if damage == 'report_device' else 'scoring.claim.json')
        value = load(tmp_path, closed['files'][name])
        if damage == 'claim_namespace':
            closed['files'][name]['path'] = 'old/execution.claim.json'
        else:
            key, changed = {'claim_boot': ('boot_epoch_seconds', 0), 'claim_pid_bool': ('pid', True),
                'ranking_protocol': ('condition', run.origin.VERSION), 'report_device': ('hardware', {'J': 'cpu'}),
                'scoring_count': ('query_count', 2)}[damage]
            value[key] = changed
            repin(tmp_path, closed['files'][name], value)
        repin(tmp_path, closed_ref, closed)
        refs['j_execution_claim'] = closed['files']['execution.claim.json']
        wait['execution_claim'] = refs['j_execution_claim']
        wait['closed'] = closed_ref
    repin(tmp_path, wait_ref, wait)
    with pytest.raises(ValueError):
        check_stage(tmp_path, fixture)


def test_raw_empty_proofs_and_descriptors_have_distinct_freshness_paths(tmp_path):
    proofs = run.Proofs(tmp_path)
    raw = pin(tmp_path, 'proof/source.py', b'# raw source\n')
    empty = pin(tmp_path, 'proof/empty', b'')
    vec = descriptor('stage/000000.npy')
    lock = {'path': 'work/overnight/gpu.lock', 'bytes': 0, 'sha256': run.origin.sha(b'')}
    proofs.descriptor(vec)
    proofs.descriptor(lock)
    proofs.embedded({'raw': raw, 'empty': empty, 'vector': vec, 'lock': lock})
    assert set(proofs.refs) == {raw['path'], empty['path']}
    assert not (tmp_path / lock['path']).exists()
    proofs.fresh()
    (tmp_path / raw['path']).write_bytes(b'# changed\n')
    with pytest.raises(ValueError):
        proofs.fresh()


@pytest.mark.parametrize('path', ['.cache/models/config.json', 'any/model.safetensors', 'any/model.bin',
    'any/vector.npy', 'work/overnight/gpu.lock', 'work/model-view-J/config.json', '../escape', '/absolute'])
def test_cpu_reads_refuse_payloads_locks_and_escaping_paths(tmp_path, path):
    with pytest.raises(ValueError):
        run.Proofs(tmp_path).raw(descriptor(path))


def test_embedded_unregistered_or_changed_payload_is_rejected(tmp_path):
    proofs = run.Proofs(tmp_path)
    vector = descriptor('stage/000001.npy')
    with pytest.raises(ValueError, match='cpu_metadata_boundary'):
        proofs.embedded({'vector': vector})
    proofs.descriptor(vector)
    with pytest.raises(ValueError, match='payload_descriptor_changed'):
        proofs.embedded({'vector': {**vector, 'sha256': 'b' * 64}})


def acceptance_fixture(root):
    original = descriptor(run.upstream.PARENT_PATH)
    envelope = {'original_manifest': original, 'rebinding': descriptor('proof/binding.json')}
    digest = run.frozen.digest(envelope)
    requirements = pin(root, 'proof/requirements.json', {'scope': 'synthetic'})
    refs = {'sj_actual_config_requirements': requirements}
    admission = {'closed_sha256': {c: 'c' * 64 for c in 'SJ'}}
    raw = pin(root, 'proof/transitive.py', b'# source proof\n')
    manifest_review = pin(root, 'proof/manifest-review.json', {
        'status': 'PASS_INDEPENDENT_INDUSTRIAL_REBOOT_MANIFEST', 'blocking_findings': [],
        'manifest_sha256': digest, 'actual_configuration_requirements': requirements})
    heavy_ref = pin(root, 'proof/heavy.json', {'status': 'PASS_ROOT_ALL_SIX_LAYA_HEAVY_CLOSED',
        'dataset_count': 6, 'active_heavy_workers': 0, 'actual_waits_verified': True, 'refs': {'raw': raw}})
    executions, configs = {}, {}
    for channel in 'JS':
        key = channel.lower()
        refs[key + '_actual_wait'] = pin(root, 'proof/' + channel + '-wait.json', {'status': 'synthetic'})
        refs[key + '_execution_claim'] = pin(root, 'proof/' + channel + '-claim.json', {'pid': 1})
        executions[channel] = {'refs': {'independent_manifest_review': manifest_review, 'heavy_completion': heavy_ref},
            'prior_j_stage': None if channel == 'J' else {
                'closed': {'path': run.STAGE_CLOSED['J'], 'bytes': 1, 'sha256': admission['closed_sha256']['J']},
                'wait': refs['j_actual_wait'], 'execution_claim': refs['j_execution_claim']}}
        # For embedded byte checks the synthetic closed descriptor is a real tiny file.
        if channel == 'S':
            prior_ref = pin(root, run.STAGE_CLOSED['J'], b'{}')
            admission['closed_sha256']['J'] = prior_ref['sha256']
            executions[channel]['prior_j_stage']['closed'] = prior_ref
        refs[key + '_execution_admission'] = pin(root, 'proof/' + channel + '-admission.json', executions[channel])
        configs[channel] = pin(root, 'proof/' + channel + '-config.json', {
            'status': 'PASS_INDEPENDENT_INDUSTRIAL_REBOOT_EXECUTION_CONFIG', 'blocking_findings': [],
            'channel': channel, 'manifest_sha256': digest, 'execution_admission': refs[key + '_execution_admission'],
            'actual_config_requirements': requirements, 'refs': {'raw': raw}, 'checks': {
                'all_six_heavy_closures_verified': True, 'all_required_actual_parent_and_child_waits_verified': True,
                'complete_transitive_proof_closure_verified': True, 'full_prior_j_nested_closure_verified': True}})
    review = {'status': 'PASS_INDEPENDENT_INDUSTRIAL_SJ_ACTUAL_OUTPUTS', 'blocking_findings': [],
        'manifest_sha256': digest, 'original_manifest_sha256': run.upstream.PARENT_SHA256,
        'closed_sha256': admission['closed_sha256'], 'wait_sha256': {c: refs[c.lower() + '_actual_wait']['sha256'] for c in 'SJ'},
        'vector_payload_validation_completed': True, 'metadata_only_cpu_reuse_allowed': True,
        'execution_config_reviews': configs}
    refs['sj_actual_output_review'] = pin(root, 'proof/actual-review.json', review)
    accepted = {'status': 'PASS_ROOT_INDUSTRIAL_SJ_REUSE_FOR_NONE',
        'independent_review_sha256': refs['sj_actual_output_review']['sha256'], 'manifest_sha256': digest,
        'closed_sha256': admission['closed_sha256'], 'wait_sha256': review['wait_sha256'],
        'complete_upstream_actual_config_reviewed': True,
        'execution_config_review_sha256': {c: ref['sha256'] for c, ref in configs.items()}}
    refs['sj_root_output_acceptance'] = pin(root, 'proof/actual-acceptance.json', accepted)
    return refs, admission, envelope, executions


def test_reviewed_actual_metadata_closure_and_raw_source_are_fresh(tmp_path):
    fixture = acceptance_fixture(tmp_path)
    proofs = run.Proofs(tmp_path)
    run.actual_output_acceptance(proofs, *fixture)
    assert 'proof/transitive.py' in proofs.refs
    proofs.fresh()
    (tmp_path / 'proof/transitive.py').write_bytes(b'# mutated\n')
    with pytest.raises(ValueError):
        proofs.fresh()


@pytest.mark.parametrize('damage', ['review_status', 'vector_audit', 'metadata_reuse', 'root_status',
    'root_config', 'empty_heavy_refs', 'five_heavy', 'active_heavy', 'missing_config',
    'config_waits', 'config_full_closure', 'config_prior_j', 'manifest_review', 'wrong_prior_wait'])
def test_unreviewed_or_partial_actual_proof_closure_is_rejected(tmp_path, damage):
    refs, admission, envelope, executions = acceptance_fixture(tmp_path)
    review_ref, root_ref = refs['sj_actual_output_review'], refs['sj_root_output_acceptance']
    review, accepted = load(tmp_path, review_ref), load(tmp_path, root_ref)
    if damage in ('review_status', 'vector_audit', 'metadata_reuse'):
        key, val = {'review_status': ('status', 'SOURCE_ONLY'), 'vector_audit': ('vector_payload_validation_completed', False),
                    'metadata_reuse': ('metadata_only_cpu_reuse_allowed', False)}[damage]
        review[key] = val
    elif damage in ('root_status', 'root_config'):
        accepted['status' if damage == 'root_status' else 'complete_upstream_actual_config_reviewed'] = False
    elif damage in ('empty_heavy_refs', 'five_heavy', 'active_heavy'):
        ref = executions['J']['refs']['heavy_completion']
        heavy = load(tmp_path, ref)
        key, val = {'empty_heavy_refs': ('refs', {}), 'five_heavy': ('dataset_count', 5),
                    'active_heavy': ('active_heavy_workers', 1)}[damage]
        heavy[key] = val
        repin(tmp_path, ref, heavy)
    elif damage == 'missing_config':
        review['execution_config_reviews'].pop('S')
    elif damage.startswith('config_'):
        ref = review['execution_config_reviews']['S']
        config = load(tmp_path, ref)
        key = {'config_waits': 'all_required_actual_parent_and_child_waits_verified',
               'config_full_closure': 'complete_transitive_proof_closure_verified',
               'config_prior_j': 'full_prior_j_nested_closure_verified'}[damage]
        config['checks'][key] = False
        repin(tmp_path, ref, config)
        accepted['execution_config_review_sha256']['S'] = ref['sha256']
    elif damage == 'manifest_review':
        ref = executions['J']['refs']['independent_manifest_review']
        manifest = load(tmp_path, ref)
        manifest['status'] = 'SOURCE_ONLY'
        repin(tmp_path, ref, manifest)
    else:
        executions['S']['prior_j_stage']['wait'] = refs['s_actual_wait']
    repin(tmp_path, review_ref, review)
    accepted['independent_review_sha256'] = review_ref['sha256']
    repin(tmp_path, root_ref, accepted)
    with pytest.raises(ValueError):
        run.actual_output_acceptance(run.Proofs(tmp_path), refs, admission, envelope, executions)


def test_original_scientific_helpers_are_direct_calls_and_new_namespace_is_explicit():
    for name in ('configuration', 'derive_rows', 'strict_ranking', 'metrics_equal', 'prior_agreement', 'archive_bundle'):
        assert getattr(run, name) is getattr(run.previous, name)
    assert run.OUTPUT != run.previous.OUTPUT
    assert run.MPS_MANIFEST != run.previous.MPS_MANIFEST
    assert len(run.UPSTREAM_KEYS) == 18
    source = (ROOT / 'src/rag_benchmark/industrial_none_derivation_v2.py').read_text()
    tree = ast.parse(source)
    forbidden = {'load', 'validate', 'check_live_metadata', 'check_execution_admission', 'encode_jobs', 'job_item'}
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and isinstance(n.func.value, ast.Name) and n.func.value.id in ('upstream', 'origin')
                and n.func.attr in forbidden]
    assert 'prior_agreement(' in source and 'len(agreements) == 8' in source
    assert 'validate(root, manifest)\n    require({p.name' in source
    assert '"vector_payloads_read": 0' in source and '"new_inference_claimed": False' in source


def test_literal_import_does_not_load_model_libraries():
    result = subprocess.run([str(ROOT / '.venv/bin/python'), '-B', '-c',
        'import sys; from rag_benchmark import industrial_none_derivation_v2; '
        'assert not any(x in sys.modules for x in ("torch", "transformers", "laya", "sentence_transformers"))'],
        cwd=ROOT, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_new_source_closure_is_checked_without_loading_models():
    pins = {p: run.frozen.record(ROOT, p)["sha256"] for p in run.SOURCES}
    pure = run.sources(ROOT, pins)
    assert {"fuse_rankings", "graded_metrics", "aggregate_metrics"} <= set(pure)
    del pins["scripts/run_industrial_none_derivation_v2.py"]
    with pytest.raises(ValueError, match="exact_cpu_source_closure"):
        run.sources(ROOT, pins)


def source_fixture(root):
    source = pin(root, "proof/source.py", b"# synthetic source")
    request = {"source_pins": {source["path"]: source["sha256"]}}
    contract = {"protocol": run.VERSION, "scope": "SOURCE_ONLY_NO_ACTUAL_DERIVATION",
                "source_pins": request["source_pins"], "source_record": source}
    refs = {"none_source_contract": pin(root, "proof/contract.json", contract)}
    review = {"status": "PASS_INDEPENDENT_INDUSTRIAL_NONE_V2_SOURCE", "blocking_findings": [],
              "source_contract_sha256": refs["none_source_contract"]["sha256"]}
    refs["none_source_review"] = pin(root, "proof/review.json", review)
    accepted = {"status": "PASS_ROOT_INDUSTRIAL_NONE_V2_SOURCE",
        "source_contract_sha256": refs["none_source_contract"]["sha256"],
        "independent_review_sha256": refs["none_source_review"]["sha256"]}
    refs["none_root_source_acceptance"] = pin(root, "proof/accepted.json", accepted)
    return refs, request


def test_source_admission_closure_checks_raw_bytes_and_exact_pins(tmp_path):
    refs, request = source_fixture(tmp_path)
    proofs = run.Proofs(tmp_path)
    run.source_admissions(proofs, refs, request)
    assert "proof/source.py" in proofs.refs
    request["source_pins"]["proof/source.py"] = "b" * 64
    with pytest.raises(ValueError, match="none_source_contract"):
        run.source_admissions(proofs, refs, request)


@pytest.mark.parametrize("key", ["none_source_review", "none_root_source_acceptance"])
def test_source_admission_requires_independent_and_root_actual_source_acceptance(tmp_path, key):
    refs, request = source_fixture(tmp_path)
    proof = load(tmp_path, refs[key])
    proof["status"] = "PENDING"
    repin(tmp_path, refs[key], proof)
    with pytest.raises(ValueError):
        run.source_admissions(run.Proofs(tmp_path), refs, request)

"""CPU-only worker protocol tests; no real models or GPU dependencies."""
import hashlib
import itertools
import json
import tarfile

import pytest

from rag_benchmark import colab_worker as worker


def report(channels=('B',)):
    job = {'id': 'fake', 'dataset': {'query_count': 2},
           'query_ids_sha256': hashlib.sha256(b'["a","b"]').hexdigest()}
    cells = []
    for size in range(1, len(channels) + 1):
        for subset in itertools.combinations(channels, size):
            for mode in ('per_channel', 'total'):
                cells.append({'variant_id': 'document__' + '_'.join(subset).lower() + '__none',
                              'budget_mode': mode, 'candidate_k': None, 'status': 'completed',
                              'completed_queries': 2, 'metrics': {name: 0.5 for name in worker.METRICS}})
    value = {'dataset': job['dataset'], 'scope': 'frozen_collection', 'evaluated_query_count': 2,
             'configuration': {'requested_channels': list(channels), 'budget_modes': ['per_channel', 'total'],
                               'query_ids': ['a', 'b']},
             'channels': {c: {'status': 'completed'} for c in channels}, 'cells': cells}
    return job, value


def test_full_grid_and_failed_channel_rejected(tmp_path):
    job, value = report(worker.CHANNELS)
    path = tmp_path / 'report.json'
    path.write_text(json.dumps(value))
    assert len(worker.validate_report(path, job, worker.CHANNELS)['cells']) == 126
    value['channels']['S']['status'] = 'failed'
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='Channel'):
        worker.validate_report(path, job, worker.CHANNELS)


@pytest.mark.parametrize('change', ['nan', 'partial', 'duplicate', 'missing_metric'])
def test_invalid_cells(tmp_path, change):
    job, value = report()
    if change == 'nan':
        value['cells'][0]['metrics']['hit@1'] = float('nan')
    elif change == 'partial':
        value['cells'][0]['completed_queries'] = 1
    elif change == 'duplicate':
        value['cells'].append(value['cells'][0])
    else:
        value['cells'][0]['metrics'].pop('hit@1')
    path = tmp_path / 'report.json'
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        worker.validate_report(path, job, ('B',))


def test_command_uses_shared_cuda_cache_without_limit(tmp_path):
    args = worker.command(tmp_path / 'data', tmp_path / 'work', worker.CHANNELS, 'all')
    assert args[args.index('--device') + 1] == 'cuda'
    assert args[args.index('--dtype') + 1] == 'bfloat16'
    assert '--limit' not in args
    assert args[args.index('--cache-dir') + 1] == str(tmp_path / 'work/cache')


def test_checkpoint_is_verified_pointer_and_omits_symlinks(tmp_path):
    workspace = tmp_path / 'work'
    workspace.mkdir()
    (workspace / 'state.json').write_text('{}')
    (workspace / 'secret.tmp').write_text('partial')
    (workspace / 'link').symlink_to(workspace / 'state.json')
    target = tmp_path / 'drive'
    worker.checkpoint(workspace, target, 'fake')
    pointer = json.loads((target / 'fake-latest.json').read_text())
    archive = target / pointer['archive']
    assert worker.digest(archive) == pointer['sha256']
    with tarfile.open(archive) as handle:
        assert handle.getnames() == ['state.json']


def test_existing_state_requires_resume_and_matching_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, 'runtime_identity', lambda: {'gpu': 'fake'})
    path = tmp_path / 'state.json'
    path.write_text(json.dumps({'identity': {}, 'status': 'failed', 'completed': []}))
    with pytest.raises(ValueError, match='explicit'):
        worker.execute({}, tmp_path, tmp_path, {})
    with pytest.raises(ValueError, match='mismatch'):
        worker.execute({}, tmp_path, tmp_path, {}, resume=True)
    assert json.loads(path.read_text())['status'] == 'failed'


def test_subprocess_failure_stops_and_preserves_failed_state(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, 'runtime_identity', lambda: {'gpu': 'fake'})
    calls = []
    def fail(args, log):
        calls.append(args)
        raise ValueError('fake failure')
    monkeypatch.setattr(worker, 'run_child', fail)
    with pytest.raises(ValueError):
        worker.execute({'id': 'fake'}, tmp_path, tmp_path, {})
    assert len(calls) == 1
    state = json.loads((tmp_path / 'state.json').read_text())
    assert state['status'] == 'failed' and state['completed'] == []


@pytest.mark.parametrize('device', ['cuda', 'cuda:1', 'mps', 'cpu'])
def test_colqwen_cli_score_device_preserves_defaults(device):
    from rag_benchmark.multimodal_cli import colqwen_cli_config
    common = {'device': device, 'dtype': 'bfloat16', 'batch_size': 1}
    config = colqwen_cli_config(common)
    assert common == {'device': device, 'dtype': 'bfloat16', 'batch_size': 1}
    if device.startswith('cuda'):
        assert config['score_device'] == device
    else:
        assert 'score_device' not in config

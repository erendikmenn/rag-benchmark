"""Explicit CUDA retrieval worker. Dry-run never imports torch or loads models."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import tarfile
import uuid

CHANNELS = ('B', 'G', 'E', 'N', 'S', 'J')
PROTOCOL = 'colab-cuda-bf16-retrieval-v1'
METRICS = {f'{name}@{k}' for name in ('hit', 'recall', 'mrr', 'ndcg') for k in (1, 5, 10, 20)} | {'map@16'}


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def atomic_json(path, value):
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('w') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def validate_dataset(path, job):
    from .multimodal import load_dataset
    if digest(path / 'dataset.json') != job['dataset_json_sha256']:
        raise ValueError('Frozen manifest mismatch')
    dataset = load_dataset(path)
    if dataset.public_summary() != job['dataset']:
        raise ValueError('Frozen dataset identity/counts mismatch')
    return dataset


def runtime_identity():
    import torch
    if not torch.cuda.is_available():
        raise ValueError('CUDA is required')
    index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    if properties.major < 8 or not torch.cuda.is_bf16_supported(including_emulation=False):
        raise ValueError('Native CUDA BF16 is required; no fallback')
    packages = {}
    for name in ('rag-benchmark', 'torch', 'transformers', 'sentence-transformers', 'accelerate',
                 'numpy', 'bm25s', 'huggingface-hub', 'peft', 'colpali-engine'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    driver = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                            capture_output=True, text=True, check=True).stdout.strip()
    lock = Path(__file__).resolve().parents[2] / 'uv.lock'
    return {'driver': driver, 'uv_lock_sha256': digest(lock) if lock.is_file() else None,
            'python': platform.python_version(), 'packages': packages, 'cuda': torch.version.cuda,
            'gpu': {'name': properties.name,
                    'capability': [properties.major, properties.minor], 'memory': properties.total_memory},
            'source_sha256': {str(p.relative_to(Path(__file__).parent)): digest(p)
                              for p in sorted(Path(__file__).parent.glob('*.py'))}}


@contextmanager
def exclusive_workspace(workspace):
    workspace = workspace.resolve()
    # A host-wide lock prevents independent workers sharing one VM.
    workspace.parent.mkdir(parents=True, exist_ok=True)
    with (Path('/tmp/rag-benchmark-colab-gpu.lock')).open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('Another worker owns this VM workspace') from None
        workspace.mkdir(mode=0o700, exist_ok=True)
        os.chmod(workspace, 0o700)
        # Reject any path tracked or unignored inside a Git checkout.
        git = subprocess.run(['git', '-C', str(workspace), 'rev-parse', '--show-toplevel'],
                             capture_output=True, text=True)
        if git.returncode == 0:
            root = Path(git.stdout.strip())
            relative = str(workspace.relative_to(root)) + '/'
            ignored = subprocess.run(['git', '-C', str(root), 'check-ignore', '-q', relative])
            if ignored.returncode:
                raise ValueError('Workspace must be gitignored or outside Git')
        yield workspace


def command(dataset, workspace, channels, stage):
    return [sys.executable, '-m', 'rag_benchmark.multimodal_cli', 'run', '--dataset', str(dataset),
            '--run-dir', str(workspace / 'runs' / stage), '--cache-dir', str(workspace / 'cache'),
            '--channels', ','.join(channels), '--rerankers', 'none', '--device', 'cuda',
            '--dtype', 'bfloat16', '--dimension', '768', '--vision-budget', '280',
            '--batch-size', '1', '--text-batch-size', '8', '--budget-modes', 'per_channel,total']


def validate_report(path, job, channels):
    report = json.loads(path.read_text())
    if report['dataset'] != job['dataset'] or report['scope'] != 'frozen_collection':
        raise ValueError('Report dataset or scope mismatch')
    count = job['dataset']['query_count']
    if report['evaluated_query_count'] != count:
        raise ValueError('Incomplete queries')
    config = report['configuration']
    if config['requested_channels'] != list(channels) or config['budget_modes'] != ['per_channel', 'total']:
        raise ValueError('Report configuration mismatch')
    query_hash = hashlib.sha256(json.dumps(config['query_ids'], separators=(',', ':')).encode()).hexdigest()
    if (len(config['query_ids']) != count or len(set(config['query_ids'])) != count
            or query_hash != job['query_ids_sha256']):
        raise ValueError('Query IDs are incomplete')
    for channel in channels:
        if report['channels'].get(channel, {}).get('status') != 'completed':
            raise ValueError('Channel did not complete')
    expected = {(('document__' + '_'.join(c).lower() + '__none'), mode)
                for size in range(1, len(channels) + 1)
                for c in itertools.combinations(channels, size) for mode in ('per_channel', 'total')}
    cells = [c for c in report['cells'] if (c['variant_id'], c['budget_mode']) in expected]
    if len(cells) != len(expected) or {(c['variant_id'], c['budget_mode']) for c in cells} != expected:
        raise ValueError('Target none grid is incomplete or duplicated')
    for cell in cells:
        metrics = cell.get('metrics', {})
        if (cell['status'] != 'completed' or cell.get('completed_queries') != count
                or cell.get('candidate_k') is not None or set(metrics) != METRICS
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1
                       for v in metrics.values())):
            raise ValueError('Target cell lacks full queries and finite 17 metrics')
    return report


def run_child(args, log):
    with log.open('ab') as handle:
        child = subprocess.Popen(args, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = child.wait()
        except BaseException:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            raise
    if code:
        raise ValueError('Subprocess failed; inspect private logs')


def checkpoint(workspace, destination, job_id):
    if destination is None:
        return
    destination = destination.resolve()
    if destination.is_relative_to(workspace):
        raise ValueError('Checkpoint must be outside workspace')
    destination.mkdir(parents=True, mode=0o700, exist_ok=True)
    git = subprocess.run(['git', '-C', str(destination), 'rev-parse', '--show-toplevel'],
                         capture_output=True, text=True)
    if git.returncode == 0:
        root = Path(git.stdout.strip())
        if subprocess.run(['git', '-C', str(root), 'check-ignore', '-q',
                           str(destination.relative_to(root)) + '/']).returncode:
            raise ValueError('Checkpoint must be gitignored or outside Git')
    os.chmod(destination, 0o700)
    # Content-addressed archives keep the previous complete pointer valid until
    # the new archive and its digest have both reached durable storage.
    temporary = destination / (job_id + '.' + uuid.uuid4().hex + '.tar.tmp')
    try:
        with tarfile.open(temporary, 'w') as archive:
            for path in sorted(workspace.rglob('*')):
                if path.is_symlink() or not path.is_file() or path.name.endswith(('.tmp', '.lock')):
                    continue
                archive.add(path, arcname=str(path.relative_to(workspace)), recursive=False)
        with temporary.open('rb') as handle:
            os.fsync(handle.fileno())
        checksum = digest(temporary)
        final = destination / (job_id + '-' + checksum + '.tar')
        temporary.replace(final)
        atomic_json(destination / (job_id + '-latest.json'),
                    {'archive': final.name, 'sha256': checksum, 'job': job_id, 'protocol': PROTOCOL})
    finally:
        temporary.unlink(missing_ok=True)


def execute(job, dataset, workspace, plan, resume=False, checkpoint_dir=None):
    from .multimodal_cli import export_report
    identity = {'protocol': PROTOCOL, 'plan': plan, 'job': job, 'runtime': runtime_identity(),
                'condition': {'device': 'cuda', 'dtype': 'bfloat16', 'dimension': 768,
                              'vision_budget': 280, 'batch_size': 1, 'text_batch_size': 8}}
    state_path = workspace / 'state.json'
    if state_path.exists():
        if not resume:
            raise ValueError('Existing state requires explicit --resume')
        state = json.loads(state_path.read_text())
        if state['identity'] != identity:
            raise ValueError('Resume config/dataset/runtime/GPU identity mismatch')
    else:
        if resume:
            raise ValueError('No state to resume')
        if any(workspace.iterdir()):
            raise ValueError('Nonempty workspace without durable state')
        state = {'identity': identity, 'status': 'prepared', 'completed': []}
    old_handler = signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        for stage, channels in [(c, (c,)) for c in CHANNELS] + [('all', CHANNELS)]:
            report_path = workspace / 'runs' / stage / 'report.json'
            if stage in state['completed']:
                validate_report(report_path, job, channels)
                continue
            state.update(status='running', stage=stage)
            atomic_json(state_path, state)
            print(json.dumps({'stage': stage, 'status': 'running'}), flush=True)
            run_child(command(dataset, workspace, channels, stage), workspace / (stage + '.log'))
            validate_report(report_path, job, channels)
            state['completed'].append(stage)
            state['status'] = 'stage_completed'
            atomic_json(state_path, state)
            checkpoint(workspace, checkpoint_dir, job['id'])
        export_report(workspace / 'runs' / 'all', workspace / 'review' / job['id'])
        atomic_json(workspace / 'review' / job['id'] / 'cuda-runtime.json', identity)
        state['status'] = 'completed'
        atomic_json(state_path, state)
        checkpoint(workspace, checkpoint_dir, job['id'])
        print(json.dumps({'status': 'completed', 'target_none_cells': 126}), flush=True)
    except BaseException as exc:
        state['status'] = 'interrupted' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 'failed'
        atomic_json(state_path, state)
        checkpoint(workspace, checkpoint_dir, job['id'])
        raise
    finally:
        signal.signal(signal.SIGTERM, old_handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=Path('configs/colab-retrieval-jobs.json'))
    parser.add_argument('--job', required=True)
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--workspace', required=True, type=Path)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--checkpoint-dir', type=Path)
    args = parser.parse_args(argv)
    try:
        plan = json.loads(args.plan.read_text())
        if plan['protocol'] != PROTOCOL or plan['channels'] != list(CHANNELS):
            raise ValueError('Unsupported worker plan')
        job = next(j for j in plan['jobs'] if j['id'] == args.job)
        validate_dataset(args.dataset.resolve(), job)
        if not args.execute:
            print(json.dumps({'status': 'dry_run', 'job': job['id'], 'target_none_cells': 126,
                              'query_count': job['dataset']['query_count'], 'model_inference': False}))
            return 0
        with exclusive_workspace(args.workspace) as workspace:
            execute(job, args.dataset.resolve(), workspace, plan, args.resume, args.checkpoint_dir)
        return 0
    except KeyboardInterrupt:
        print('{"status":"interrupted"}')
        return 130
    except Exception:
        # Raw exceptions may contain source text or paths; keep public output fixed.
        print('{"status":"rejected_or_failed","detail":"inspect private workspace state and logs"}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

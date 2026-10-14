"""Prepare pinned source representations only; never retrieve, rerank or answer."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from rag_benchmark.models import stable_hash
from rag_benchmark.multimodal import (
    TextEmbeddingAdapter, cached_encode, encoding_cache_identity, load_dataset,
)


def sha256(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)


@contextmanager
def exclusive_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def make_adapter(channel: str, config: dict):
    if channel in {'G', 'E'}:
        return TextEmbeddingAdapter('bge' if channel == 'G' else 'embeddinggemma', config)
    if channel in {'N', 'J'}:
        from rag_benchmark.multimodal_models import make_multimodal_adapter
        return make_multimodal_adapter(channel, config)
    raise ValueError('This worker supports ordinary dense G/E/N/J only')


def validate_spec(spec: dict, dataset, adapter) -> None:
    from rag_benchmark import models, multimodal
    required = {Path(__file__).absolute(), Path(models.__file__).absolute(), Path(multimodal.__file__).absolute()}
    if spec.get('channel') in {'N', 'J'}:
        from rag_benchmark import multimodal_models
        required.add(Path(multimodal_models.__file__).absolute())
    if not required <= {Path(path).absolute() for path in spec.get('source_sha256', {})}:
        raise ValueError('Producer and actual model/cache implementations must be sealed')
    if spec.get('version') != 1 or spec.get('protocol') != 'source-embeddings-only-v1':
        raise ValueError('Unknown preparation protocol')
    if dataset.identity != spec.get('dataset_identity'):
        raise ValueError('Frozen dataset identity changed')
    if adapter.identity != spec.get('adapter_identity'):
        raise ValueError('Actual adapter differs from reviewed identity')
    if spec.get('roles') != ['document', 'query']:
        raise ValueError('Both explicitly ordered roles are required')
    if spec.get('block_size') != 128:
        raise ValueError('Preserve original 128-row cache partition')
    if spec['channel'] in {'G', 'E'}:
        dataset.require_text()
    elif spec['channel'] == 'J':
        dataset.require_candidate_text()
    # A segmented code condition cannot silently become a whole-function pass.
    if dataset.track == 'code':
        raise ValueError('Code requires its separately verified segmentation preparation')


def verify_seals(spec: dict) -> None:
    if not spec.get('source_sha256'):
        raise ValueError('Explicit implementation/source seals are required')
    for path, digest in spec['source_sha256'].items():
        if sha256(Path(path)) != digest:
            raise ValueError('Preparation implementation/source seal changed')
    for path, digest in spec.get('existing_cache_sha256', {}).items():
        if sha256(Path(path)) != digest:
            raise ValueError('Previously verified cache payload changed')


def describe_role(dataset, adapter, role: str, cache_dir: Path, channel: str) -> dict:
    text_only = channel in {'G', 'E'}
    identity = stable_hash({'encoding': encoding_cache_identity(dataset, adapter, role),
                            'text_only': text_only})
    directory = cache_dir / 'vectors' / identity
    items = dataset.corpus if role == 'document' else dataset.queries
    blocks = []
    for start in range(0, len(items), 128):
        stop = min(start + 128, len(items))
        path = directory / f'{start:09d}-{stop:09d}.npy'
        if not path.is_file():
            raise ValueError('Prepared role is incomplete')
        values = np.load(path, mmap_mode='r', allow_pickle=False)
        if values.dtype != np.float32 or values.shape != (stop - start, adapter.dimension):
            raise ValueError('Prepared vector shape/dtype differs')
        if not np.isfinite(values).all() or not np.allclose(
                np.linalg.norm(values, axis=1), 1, atol=2e-3, rtol=2e-3):
            raise ValueError('Prepared vectors are nonfinite or not normalized')
        blocks.append({'path': str(path.absolute()), 'sha256': sha256(path),
                       'size': path.stat().st_size, 'shape': list(values.shape),
                       'item_ids': [item['id'] for item in items[start:stop]]})
    return {'set_id': identity, 'channel': channel, 'role': role, 'kind': 'dense',
            'dimension': adapter.dimension, 'blocks': blocks}


def verify_initial_inventory(spec: dict, dataset, adapter) -> None:
    current = {}
    for role in spec['roles']:
        identity = stable_hash({'encoding': encoding_cache_identity(dataset, adapter, role),
                                'text_only': spec['channel'] in {'G', 'E'}})
        directory = Path(spec['cache_dir']) / 'vectors' / identity
        for path in directory.glob('*.npy'):
            current[str(path.absolute())] = sha256(path)
    expected = {str(Path(path).absolute()): digest
                for path, digest in spec.get('existing_cache_sha256', {}).items()}
    if current != expected:
        raise ValueError('Actual initial cache inventory differs from reviewed pins')


def execute(spec: dict, output: Path) -> dict:
    verify_seals(spec)
    dataset = load_dataset(spec['dataset_root'])
    adapter = make_adapter(spec['channel'], spec['config'])
    validate_spec(spec, dataset, adapter)
    identity = stable_hash(spec)
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    pins_path = output / 'pins.json'
    if pins_path.exists() and json.loads(pins_path.read_text()) != spec:
        raise ValueError('Existing preparation output has different pins')
    status_path = output / 'status.json'
    if status_path.exists():
        status = json.loads(status_path.read_text())
        if status.get('status') != 'completed':
            raise ValueError('Interrupted/failed preparation needs explicit review; no automatic retry')
        result_path = output / 'result.json'
        if status.get('result_sha256') != sha256(result_path):
            raise ValueError('Completed preparation result checksum differs')
        prior = json.loads(result_path.read_text())
        if prior.get('identity') != identity or prior.get('status') != 'completed':
            raise ValueError('Completed preparation identity differs')
        if [x.get('role') for x in prior.get('sets', [])] != spec['roles']:
            raise ValueError('Completed preparation does not contain both roles')
        for representation in prior['sets']:
            if describe_role(dataset, adapter, representation['role'], Path(spec['cache_dir']),
                             spec['channel']) != {key: value for key, value in representation.items()
                                                  if key != 'provenance'}:
                raise ValueError('Completed preparation cache changed')
        return prior
    verify_initial_inventory(spec, dataset, adapter)
    atomic_json(pins_path, spec)
    started = time.monotonic()
    usage = []
    try:
        for role in spec['roles']:
            verify_seals(spec)
            atomic_json(status_path, {'status': 'encoding', 'identity': identity,
                        'channel': spec['channel'], 'role': role, 'owner_pid': os.getpid(),
                        'started_unix': time.time(), 'benchmark_runs': 0})
            _, facts = cached_encode(dataset, adapter, role, Path(spec['cache_dir']),
                                     block_size=128, text_only=spec['channel'] in {'G', 'E'})
            usage.append(facts)
        verify_seals(spec)
        if load_dataset(spec['dataset_root']).identity != dataset.identity:
            raise ValueError('Dataset changed during preparation')
        sets = [describe_role(dataset, adapter, role, Path(spec['cache_dir']), spec['channel'])
                for role in spec['roles']]
        for representation in sets:
            representation['provenance'] = {'adapter_identity': adapter.identity,
                'config': spec['config'], 'preparation_spec_sha256': identity,
                'preparation_protocol': spec['protocol'], 'historical_payload_attestation': False}
        result = {'status': 'completed', 'identity': identity, 'sets': sets, 'usage': usage,
                  'elapsed_seconds': time.monotonic() - started, 'benchmark_runs': 0,
                  'reranker_calls': 0, 'answer_generation_calls': 0}
        atomic_json(output / 'result.json', result)
        atomic_json(status_path, {**{key: value for key, value in result.items() if key != 'sets'},
                                 'result_sha256': sha256(output / 'result.json')})
        return result
    except BaseException as exc:
        atomic_json(status_path, {'status': 'failed', 'identity': identity,
                                 'error_type': type(exc).__name__, 'benchmark_runs': 0})
        raise
    finally:
        adapter.unload()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu-lock', type=Path, default=Path('work/overnight/gpu.lock'))
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    os.umask(0o077)
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    spec = json.loads(args.spec.read_bytes())
    if not args.execute:
        verify_seals(spec)
        dataset = load_dataset(spec['dataset_root'])
        adapter = make_adapter(spec['channel'], spec['config'])
        validate_spec(spec, dataset, adapter)
        print(json.dumps({'status': 'ready_not_executed', 'dataset': dataset.manifest['id'],
                          'channel': spec['channel'], 'rows': len(dataset.corpus) + len(dataset.queries)}))
        return
    with exclusive_lock(args.gpu_lock), exclusive_lock(args.output / 'owner.lock'):
        result = execute(spec, args.output)
        print(json.dumps({k: result[k] for k in ['status', 'elapsed_seconds', 'benchmark_runs']}))


if __name__ == '__main__':
    main()

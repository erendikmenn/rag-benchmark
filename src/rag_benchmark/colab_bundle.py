"""Private, allowlisted frozen dataset transport; requires only the standard library."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import tempfile

REQUIRED = ('dataset.json', 'corpus.jsonl', 'queries.jsonl', 'qrels.jsonl', 'assets.jsonl')
OPTIONAL = ('documents_metadata.json',)
MANIFEST = 'colab_bundle_manifest.json'


def _digest(path: Path) -> str:
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def _name(value: str) -> str:
    if not isinstance(value, str) or not value or '\\' in value:
        raise ValueError('Invalid bundle path')
    path = PurePosixPath(value)
    if path.is_absolute() or any(p in {'', '.', '..'} for p in value.split('/')):
        raise ValueError('Unsafe bundle path')
    return value


def _json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def _rows(path: Path):
    with path.open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _inventory(root: Path) -> tuple[dict, dict, str]:
    metadata = _json(root / 'dataset.json')
    if metadata.get('status') != 'ready' or not metadata.get('revision'):
        raise ValueError('A ready dataset with frozen revision is required')
    referenced = set()
    for filename in ('corpus.jsonl', 'queries.jsonl'):
        for row in _rows(root / filename):
            referenced.update(_name(value) for value in row.get('media', {}).values())
    forbidden = {'raw', 'cache', 'caches', '.cache', 'models', 'runs', 'logs', 'errors', '__pycache__'}
    if any(set(PurePosixPath(n).parts) & forbidden or n.endswith('.log') for n in referenced):
        raise ValueError('Referenced media includes a forbidden private working path')
    assets = _rows(root / 'assets.jsonl')
    if len({row['path'] for row in assets}) != len(assets):
        raise ValueError('Duplicate asset inventory')
    if referenced != {row['path'] for row in assets}:
        raise ValueError('Asset inventory differs from referenced media')
    if referenced & set((*REQUIRED, *OPTIONAL, MANIFEST)):
        raise ValueError('Media collides with bundle metadata')
    names = set(REQUIRED) | referenced | {n for n in OPTIONAL if (root / n).exists()}
    inventory = {}
    for name in sorted(names):
        target = (root / name).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            raise ValueError('Bundle files must remain within dataset root')
        inventory[name] = {'sha256': _digest(target), 'size_bytes': target.stat().st_size}
    for row in assets:
        if row['sha256'] != inventory[row['path']]['sha256'] or row['size_bytes'] != inventory[row['path']]['size_bytes']:
            raise ValueError('Asset checksum or size mismatch')
    for name, expected in metadata.get('files', {}).items():
        if name not in set(REQUIRED) | set(OPTIONAL) or name not in inventory:
            raise ValueError('Dataset declares a file outside the transport allowlist')
        if (expected['sha256'] != inventory[name]['sha256'] or
                                  expected['size_bytes'] != inventory[name]['size_bytes']):
            raise ValueError('Dataset checksum or size mismatch')
    identity_value = {'files': {n: inventory[n]['sha256'] for n in REQUIRED[:4]},
                      'media': {n: inventory[n]['sha256'] for n in referenced}}
    identity = hashlib.sha256(json.dumps(identity_value, ensure_ascii=False, sort_keys=True,
                                        separators=(',', ':'), default=str).encode()).hexdigest()
    return inventory, {k: metadata[k] for k in ('id', 'track', 'revision') if k in metadata}, identity


def _summary(path: Path, inventory: dict) -> dict:
    return {'sha256': _digest(path), 'archive_bytes': path.stat().st_size,
            'file_count': len(inventory), 'payload_bytes': sum(v['size_bytes'] for v in inventory.values())}


def pack(dataset: Path, output: Path) -> dict:
    root, output = Path(dataset).resolve(), Path(output).absolute()
    if output.resolve() != output:
        raise ValueError('Archive output cannot traverse symlinks')
    # Never produce an accidentally commit-able archive, even outside this checkout.
    output.parent.mkdir(parents=True, exist_ok=True)
    checked = subprocess.run(['git', 'check-ignore', '--quiet', '--', str(output)],
                             cwd=output.parent, capture_output=True)
    if checked.returncode != 0:
        raise ValueError('Archive output must be inside a Git-ignored directory')
    if output.exists() or output.is_symlink():
        raise ValueError('Archive output already exists')
    inventory, metadata, identity = _inventory(root)
    manifest = {'schema_version': 1, 'dataset': metadata, 'dataset_identity': identity, 'entries': inventory}
    descriptor, temporary = tempfile.mkstemp(prefix='.colab-bundle-', dir=output.parent)
    os.close(descriptor)
    try:
        with tarfile.open(temporary, 'w', format=tarfile.PAX_FORMAT) as archive:
            payload = json.dumps(manifest, sort_keys=True).encode()
            info = tarfile.TarInfo(MANIFEST)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
            for name in inventory:
                target = (root / name).resolve()
                info = tarfile.TarInfo(name)
                info.size = inventory[name]['size_bytes']
                with target.open('rb') as handle:
                    archive.addfile(info, handle)
        # Catch changed files while packing and verify all archive bytes before publication.
        with tempfile.TemporaryDirectory(dir=output.parent) as check:
            unpack(Path(temporary), _digest(Path(temporary)), Path(check) / 'dataset')
        os.link(temporary, output)
        return _summary(output, inventory)
    finally:
        Path(temporary).unlink(missing_ok=True)


def unpack(archive: Path, sha256: str, output: Path) -> dict:
    archive, output = Path(archive), Path(output).absolute()
    if len(sha256) != 64 or _digest(archive) != sha256.lower():
        raise ValueError('Archive SHA-256 mismatch')
    if output.is_symlink() or (output.exists() and (not output.is_dir() or any(output.iterdir()))):
        raise ValueError('Output must be absent or an empty directory')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.colab-unpack-', dir=output.parent) as temporary:
        staging = Path(temporary) / 'dataset'
        staging.mkdir(mode=0o700)
        with tarfile.open(archive, 'r:*') as source:
            members = source.getmembers()
            names = [_name(m.name) for m in members]
            if len(set(names)) != len(names) or any(not m.isfile() for m in members):
                raise ValueError('Duplicate or non-regular archive entry')
            if MANIFEST not in names:
                raise ValueError('Missing bundle manifest')
            manifest = json.load(source.extractfile(source.getmember(MANIFEST)))
            inventory = manifest['entries']
            if manifest.get('schema_version') != 1 or set(names) != set(inventory) | {MANIFEST}:
                raise ValueError('Bundle inventory mismatch')
            for name, expected in inventory.items():
                _name(name)
                member = source.getmember(name)
                if member.size != expected['size_bytes']:
                    raise ValueError('Bundle entry size mismatch')
                target = staging / name
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                for directory in target.parents:
                    if directory == staging.parent:
                        break
                    directory.chmod(0o700)
                descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with source.extractfile(member) as handle, os.fdopen(descriptor, 'wb') as destination:
                    shutil.copyfileobj(handle, destination)
                if _digest(target) != expected['sha256']:
                    raise ValueError('Bundle entry checksum mismatch')
        actual, metadata, identity = _inventory(staging)
        if actual != inventory or metadata != manifest['dataset'] or identity != manifest['dataset_identity']:
            raise ValueError('Dataset identity or allowlist mismatch')
        if output.exists():
            if any(output.iterdir()):
                raise ValueError('Output became nonempty')
            output.rmdir()
        os.rename(staging, output)
    return _summary(archive, inventory)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    pack_parser = commands.add_parser('pack')
    pack_parser.add_argument('--dataset', required=True, type=Path)
    pack_parser.add_argument('--output', required=True, type=Path)
    unpack_parser = commands.add_parser('unpack')
    unpack_parser.add_argument('--archive', required=True, type=Path)
    unpack_parser.add_argument('--sha256', required=True)
    unpack_parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    try:
        result = pack(args.dataset, args.output) if args.command == 'pack' else unpack(args.archive, args.sha256, args.output)
    except (ValueError, OSError, KeyError, TypeError, tarfile.TarError, json.JSONDecodeError):
        parser.exit(1, 'Bundle operation failed; no dataset content is printed.\n')
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()

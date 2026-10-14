import hashlib
import io
import json
import subprocess
import tarfile

import pytest

from rag_benchmark.colab_bundle import pack, unpack


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def fixture(tmp_path):
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    (tmp_path / '.gitignore').write_text('private/\n')
    root = tmp_path / 'source'
    root.mkdir()
    (root / 'media').mkdir()
    (root / 'media/image.bin').write_bytes(b'\x00private image\xff')
    (root / 'dataset.json').write_text(json.dumps({'id': 'fixture', 'track': 'photo', 'revision': 'frozen', 'status': 'ready'}))
    (root / 'corpus.jsonl').write_text(json.dumps({'id': 'd', 'media': {'image': 'media/image.bin'}}) + '\n')
    (root / 'queries.jsonl').write_text(json.dumps({'id': 'q', 'text': 'query'}) + '\n')
    (root / 'qrels.jsonl').write_text(json.dumps({'query_id': 'q', 'corpus_id': 'd', 'relevance': 1}) + '\n')
    (root / 'assets.jsonl').write_text(json.dumps({'path': 'media/image.bin', 'sha256': digest(root / 'media/image.bin'), 'size_bytes': 15}) + '\n')
    (root / 'documents_metadata.json').write_text('{}\n')
    (root / 'secret.log').write_text('must never leave')
    return root, tmp_path / 'private/bundle.tar'


def test_roundtrip_allowlist_and_internal_symlink(fixture, tmp_path):
    root, archive = fixture
    (root / 'media/image.bin').rename(root / 'media/original.bin')
    (root / 'media/image.bin').symlink_to('original.bin')
    result = pack(root, archive)
    destination = tmp_path / 'restored'
    unpack(archive, result['sha256'], destination)
    assert not (destination / 'secret.log').exists()
    assert not (destination / 'media/original.bin').exists()
    for name in ('dataset.json', 'corpus.jsonl', 'queries.jsonl', 'qrels.jsonl', 'assets.jsonl', 'documents_metadata.json', 'media/image.bin'):
        assert (root / name).read_bytes() == (destination / name).read_bytes()
    assert not (destination / 'media/image.bin').is_symlink()
    assert destination.stat().st_mode & 0o777 == 0o700
    assert (destination / 'media').stat().st_mode & 0o777 == 0o700
    assert (destination / 'media/image.bin').stat().st_mode & 0o777 == 0o600
    assert set(result) == {'sha256', 'archive_bytes', 'file_count', 'payload_bytes'}


def test_output_must_be_ignored(fixture):
    root, _ = fixture
    with pytest.raises(ValueError, match='Git-ignored'):
        pack(root, root.parent / 'public.tar')


def test_wrong_hash_and_existing_output(fixture, tmp_path):
    root, archive = fixture
    result = pack(root, archive)
    output = tmp_path / 'restored'
    with pytest.raises(ValueError, match='SHA-256'):
        unpack(archive, '0' * 64, output)
    assert not output.exists()
    output.mkdir()
    (output / 'keep').write_text('keep')
    with pytest.raises(ValueError, match='empty'):
        unpack(archive, result['sha256'], output)
    assert (output / 'keep').read_text() == 'keep'


@pytest.mark.parametrize('kind', ['traversal', 'absolute', 'duplicate', 'symlink', 'hardlink', 'device', 'modified', 'extra'])
def test_rejects_unsafe_or_modified_archive(fixture, tmp_path, kind):
    root, archive = fixture
    pack(root, archive)
    bad = tmp_path / 'bad.tar'
    with tarfile.open(archive) as source, tarfile.open(bad, 'w') as target:
        for member in source.getmembers():
            content = source.extractfile(member).read()
            if kind == 'modified' and member.name == 'corpus.jsonl':
                content = b'X' + content[1:]
            target.addfile(member, io.BytesIO(content))
        if kind != 'modified':
            name = {'traversal': '../escape', 'absolute': '/escape', 'duplicate': 'dataset.json'}.get(kind, 'extra')
            member = tarfile.TarInfo(name)
            if kind in {'symlink', 'hardlink', 'device'}:
                member.type = {'symlink': tarfile.SYMTYPE, 'hardlink': tarfile.LNKTYPE, 'device': tarfile.CHRTYPE}[kind]
                member.linkname = 'dataset.json'
            target.addfile(member, io.BytesIO(b''))
    output = tmp_path / 'restored'
    with pytest.raises(ValueError):
        unpack(bad, digest(bad), output)
    assert not output.exists()


def test_rejects_external_symlink(fixture, tmp_path):
    root, archive = fixture
    external = tmp_path / 'external'
    external.write_bytes((root / 'media/image.bin').read_bytes())
    (root / 'media/image.bin').unlink()
    (root / 'media/image.bin').symlink_to(external)
    with pytest.raises(ValueError, match='within'):
        pack(root, archive)
    assert not archive.exists()


def test_preserves_loader_identity(fixture, tmp_path):
    root, archive = fixture
    # Match the identity formula used by multimodal.load_dataset, without importing numpy/models.
    identity_value = {'files': {name: digest(root / name) for name in
                               ('dataset.json', 'corpus.jsonl', 'queries.jsonl', 'qrels.jsonl')},
                      'media': {'media/image.bin': digest(root / 'media/image.bin')}}
    expected = hashlib.sha256(json.dumps(identity_value, ensure_ascii=False, sort_keys=True,
                                        separators=(',', ':'), default=str).encode()).hexdigest()
    pack(root, archive)
    with tarfile.open(archive) as source:
        manifest = json.load(source.extractfile('colab_bundle_manifest.json'))
    assert manifest['dataset_identity'] == expected
    assert manifest['dataset'] == {'id': 'fixture', 'track': 'photo', 'revision': 'frozen'}


def test_bad_asset_inventory_never_publishes(fixture):
    root, archive = fixture
    (root / 'assets.jsonl').write_text('')
    with pytest.raises(ValueError, match='inventory'):
        pack(root, archive)
    assert not archive.exists()


def test_actual_loader_identity_roundtrip(fixture, tmp_path):
    from rag_benchmark.multimodal import load_dataset

    root, archive = fixture
    original = load_dataset(root)
    result = pack(root, archive)
    destination = tmp_path / 'restored'
    unpack(archive, result['sha256'], destination)
    restored = load_dataset(destination)
    assert original.identity == restored.identity
    assert original.manifest == restored.manifest


@pytest.mark.parametrize('failure', ['size', 'unsupported'])
def test_manifest_declarations_enforced(fixture, failure):
    root, archive = fixture
    metadata = json.loads((root / 'dataset.json').read_text())
    name = 'corpus.jsonl' if failure == 'size' else 'evaluation_transcripts.jsonl'
    metadata['files'] = {name: {'sha256': digest(root / 'corpus.jsonl'), 'size_bytes': 999}}
    (root / 'dataset.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        pack(root, archive)
    assert not archive.exists()

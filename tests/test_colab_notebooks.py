"""Notebook preparation tests do not execute network or GPU operations."""
import ast
import hashlib
import io
import json
from pathlib import Path
import runpy
import sys
import tarfile
import types

import pytest

ROOT = Path(__file__).parents[1]
builder = runpy.run_path(str(ROOT / 'scripts/build_colab_notebooks.py'))


@pytest.fixture
def notebook():
    job = json.loads((ROOT / 'configs/colab-retrieval-jobs.json').read_text())['jobs'][0]
    return builder['notebook'](job, 'a' * 40)


def sources(notebook):
    return [''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code']


def test_three_distinct_full_query_jobs_and_output_free_cells():
    jobs = json.loads((ROOT / 'configs/colab-retrieval-jobs.json').read_text())['jobs']
    assert len(jobs) == len({j['id'] for j in jobs}) == 3
    assert sum(j['dataset']['query_count'] for j in jobs) == 949
    for job in jobs:
        notebook = builder['notebook'](job, 'a' * 40)
        assert notebook['metadata']['rag_benchmark']['source_ref'] == 'a' * 40
        assert notebook['metadata']['colab']['private_outputs'] is True
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                assert cell['outputs'] == [] and cell['execution_count'] is None
                compile(''.join(cell['source']), '<cell>', 'exec')
        assert 'RUN_FULL = False' in '\n'.join(sources(notebook))


def test_new_clone_checks_out_files_before_dirty_check(notebook):
    source = sources(notebook)[0]
    assert "'clone', '--no-checkout'" not in source
    assert "'clone', 'https://github.com/erendikmenn/rag-benchmark.git'" in source
    assert "'checkout', '--detach', SOURCE_REF" in source
    assert "'sync', '--frozen', '--all-extras', '--python', '3.12'" in source


def test_nested_scripts_compile_and_prepare_pinned_models(notebook, monkeypatch):
    scripts = {}
    for source in sources(notebook):
        for node in ast.parse(source).body:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if name in {'probe', 'model_setup'} and isinstance(node.value, ast.Constant):
                    scripts[name] = '\n'.join(line[8:] if line.startswith('        ') else line
                                             for line in node.value.value.splitlines())
    assert set(scripts) == {'probe', 'model_setup'}
    for source in scripts.values():
        compile(source, '<remote-script>', 'exec')
    calls = []
    models = types.ModuleType('rag_benchmark.models')
    models.MODEL_DEFAULTS = {'bge': {'revision': 'bge-pin'}, 'embeddinggemma': {'revision': 'eg-pin'}}
    models.fetch_model = lambda config: calls.append(config)
    colqwen = types.ModuleType('rag_benchmark.multimodal_colqwen')
    colqwen.COLQWEN_DEFAULTS = {'revision': 'adapter-pin', 'base_revision': 'base-pin'}
    colqwen.pinned_snapshots = lambda config, **kw: calls.append((config, kw))
    monkeypatch.setitem(sys.modules, 'rag_benchmark.models', models)
    monkeypatch.setitem(sys.modules, 'rag_benchmark.multimodal_colqwen', colqwen)
    exec(scripts['model_setup'], {})
    assert calls == [{'revision': 'bge-pin'}, {'revision': 'eg-pin'},
                     ({'revision': 'adapter-pin', 'base_revision': 'base-pin'}, {'download': True})]


@pytest.mark.parametrize('member_name,kind', [('../escape', 'file'), ('/escape', 'file'),
                                             ('state.json', 'symlink')])
def test_restore_rejects_unsafe_archive(notebook, tmp_path, member_name, kind):
    job = notebook['metadata']['rag_benchmark']['job']
    root = tmp_path / 'drive'
    folder = root / 'checkpoints' / job
    folder.mkdir(parents=True)
    archive = folder / (job + '-test.tar')
    with tarfile.open(archive, 'w') as tar:
        member = tarfile.TarInfo(member_name)
        if kind == 'symlink':
            member.type = tarfile.SYMTYPE
            member.linkname = '/outside'
        tar.addfile(member, io.BytesIO())
    checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
    (folder / (job + '-latest.json')).write_text(json.dumps({'job': job, 'protocol': 'colab-cuda-bf16-retrieval-v1',
                                                          'archive': archive.name, 'sha256': checksum}))
    source = next(s for s in sources(notebook) if 'RESTORE_CHECKPOINT = False' in s)
    source = source.replace('RESTORE_CHECKPOINT = False', 'RESTORE_CHECKPOINT = True')
    workspace = tmp_path / 'local' / job
    with pytest.raises(RuntimeError, match='arşiv girdisi'):
        exec(source, {'JOB': job, 'WORKSPACE': workspace, 'DRIVE_ROOT': root, 'Path': Path, 'json': json})
    assert not workspace.exists()

"""Explicit BGE-only batching condition without model inference."""
from types import SimpleNamespace

import pytest

from rag_benchmark.multimodal import BGETextReranker
from rag_benchmark.multimodal_cli import main


def run_cli(tmp_path, monkeypatch, extra=(), *, fake=False):
    captured = {}
    monkeypatch.setattr('rag_benchmark.multimodal.load_dataset', lambda _: SimpleNamespace(track='video'))
    def run(*args, **kwargs):
        captured.update(kwargs)
        return {'scope': 'partial', 'evaluated_query_count': 0}
    monkeypatch.setattr('rag_benchmark.multimodal.run_matrix', run)
    if fake:
        def adapter(name, config):
            return SimpleNamespace(name=name, config=dict(config), identity=(name, tuple(sorted(config.items()))))
        monkeypatch.setattr('rag_benchmark.multimodal.TextEmbeddingAdapter', adapter)
        monkeypatch.setattr('rag_benchmark.multimodal_models.make_multimodal_adapter', adapter)
        monkeypatch.setattr('rag_benchmark.multimodal_specialists.SpecialistAdapter', adapter)
        monkeypatch.setattr('rag_benchmark.multimodal.BGETextReranker', lambda config: adapter('reranker-bge', config))
        monkeypatch.setattr('rag_benchmark.multimodal_generation.MultimodalLayaReranker', lambda config: adapter('laya', config))
        class Generator:
            identity = 'fixed-gemma'
            def preflight(self): pass
        monkeypatch.setattr('rag_benchmark.multimodal_generation.LocalMultimodalGenerator', Generator)
    channels = 'G,E,N,S,J' if fake else 'B'
    rerankers = 'laya_text,bge_reranker_text,gemma4_relevance' if fake else 'bge_reranker_text'
    assert main(['run', '--dataset', str(tmp_path), '--run-dir', str(tmp_path / 'run'),
                 '--channels', channels, '--rerankers', rerankers, *extra]) == 0
    return captured


@pytest.mark.parametrize('common_batch', [1, 3])
def test_omitted_flag_preserves_actual_bge_config_and_identity(tmp_path, monkeypatch, common_batch):
    current = run_cli(tmp_path, monkeypatch, ['--batch-size', str(common_batch)])['rerankers']['bge_reranker_text']
    old = BGETextReranker({'device': 'mps', 'dtype': 'bfloat16', 'batch_size': common_batch})
    assert current.base.config == old.base.config
    assert current.identity == old.identity
    assert current.base._model is None


def test_explicit_bge_batch_has_distinct_actual_identity(tmp_path, monkeypatch):
    current = run_cli(tmp_path, monkeypatch, ['--bge-reranker-batch-size', '8'])['rerankers']['bge_reranker_text']
    baseline = BGETextReranker({'device': 'mps', 'dtype': 'bfloat16', 'batch_size': 1})
    assert current.base.config['batch_size'] == 8
    assert current.identity != baseline.identity
    assert current.base._model is None


def test_only_bge_reranker_changes_with_explicit_flag(tmp_path, monkeypatch):
    old = run_cli(tmp_path, monkeypatch, fake=True)
    current = run_cli(tmp_path, monkeypatch, ['--bge-reranker-batch-size', '8'], fake=True)
    for channel in ('G', 'E', 'N', 'S', 'J'):
        assert current['adapters'][channel].identity == old['adapters'][channel].identity
    assert current['adapters']['G'].config['batch_size'] == 8
    assert current['adapters']['E'].config['batch_size'] == 8
    assert current['adapters']['N'].config['batch_size'] == 1
    for name in ('laya_text', 'gemma4_relevance'):
        assert current['rerankers'][name].identity == old['rerankers'][name].identity
    assert current['rerankers']['bge_reranker_text'].config['batch_size'] == 8
    assert current['rerankers']['bge_reranker_text'].identity != old['rerankers']['bge_reranker_text'].identity


@pytest.mark.parametrize('value', ['0', '-1', '1.5', 'nan'])
def test_invalid_reranker_batch_is_rejected(tmp_path, value):
    with pytest.raises(SystemExit) as exc:
        main(['run', '--dataset', str(tmp_path), '--run-dir', str(tmp_path / 'run'),
              '--bge-reranker-batch-size', value])
    assert exc.value.code == 2

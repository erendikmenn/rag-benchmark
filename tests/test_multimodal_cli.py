"""Public exports reflect only measured cells; local servers cannot shift context."""
import json

import pytest

from rag_benchmark.multimodal_cli import export_report, generation_server_command


@pytest.mark.parametrize("command", ["asr", "prepare", "preflight", "analyze", "dimensions"])
def test_delegated_command_help_preserves_argument_contract(command, capsys):
    from rag_benchmark.multimodal_cli import main
    with pytest.raises(SystemExit) as result:
        main([command, "--help"])
    assert result.value.code == 0
    assert "usage:" in capsys.readouterr().out


def test_export_retains_coverage_and_never_copies_local_errors(tmp_path):
    run, out = tmp_path / "run", tmp_path / "public"
    run.mkdir()
    report = {"dataset": {"id": "frozen", "revision": "a" * 40, "query_count": 9, "corpus_count": 100},
        "scope": "partial", "evaluated_query_count": 2,
        "cells": [{"variant_id": "measured", "status": "completed", "primary": True,
            "budget_mode": "per_channel", "candidate_k": None, "metrics": {"hit@5": 0.5}},
            {"variant_id": "not_measured", "status": "planned", "primary": True, "metrics": {"hit@5": 1}}]}
    (run / "report.json").write_text(json.dumps(report))
    (run / "matrix.csv").write_text("status\ncompleted\nplanned\n")
    (run / "failures.jsonl").write_text("SECRET LOCAL ERROR")
    export_report(run, out)
    text = (out / "report.md").read_text()
    assert "2 evaluated queries out of 9" in text
    assert "| measured |" in text and "0.5000" in text
    assert "| not_measured |" not in text
    assert not (out / "failures.jsonl").exists()
    assert json.loads((out / "report.json").read_text())["cells"][1]["status"] == "planned"


def test_server_verifies_both_files_and_disables_context_shift(tmp_path, monkeypatch):
    paths = {}
    for name in ("binary", "model", "projector"):
        paths[name] = tmp_path / name
        paths[name].write_bytes(name.encode())
    config = {"model_path": str(paths["model"]), "projector_path": str(paths["projector"]),
        "model_sha256": "model-hash", "projector_sha256": "projector-hash", "model": "local"}
    monkeypatch.setattr("rag_benchmark.cli.file_sha256", lambda path: str(path).split("/")[-1] + "-hash")
    command = generation_server_command(paths["binary"], config, context=8192, parallel=2)
    assert "--no-context-shift" in command
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert command[command.index("--ctx-size") + 1] == "16384"
    assert command[command.index("--mmproj") + 1] == str(paths["projector"])
    config["projector_sha256"] = "wrong"
    with pytest.raises(ValueError, match="projector checksum"):
        generation_server_command(paths["binary"], config)


@pytest.mark.parametrize('track,names', [('photo', 'gemma4_relevance'), ('speech', 'none')])
def test_audio_window_flag_rejects_conditions_that_cannot_use_it(tmp_path, monkeypatch, track, names):
    from types import SimpleNamespace
    from rag_benchmark.multimodal_cli import main
    monkeypatch.setattr('rag_benchmark.multimodal.load_dataset', lambda _: SimpleNamespace(track=track))
    with pytest.raises(SystemExit) as exc:
        main(['run', '--dataset', str(tmp_path), '--run-dir', str(tmp_path / 'run'),
              '--channels', 'B', '--rerankers', names, '--audio-overlength', 'source_windows_max'])
    assert exc.value.code == 2


def test_audio_wrapper_is_fixed_after_real_generator_preflight_contract(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from rag_benchmark.multimodal_audio_rerank import SourceAudioWindowReranker
    from rag_benchmark.multimodal_cli import main
    class Generator:
        pointwise = True
        config = {'audio_max_duration_seconds': 30.0}
        identity = 'unverified'
        def preflight(self):
            self.identity = 'verified-model-and-prompt'
        def score(self, query, candidates):
            raise AssertionError('This CLI wiring test must not start inference')
    def run_matrix(*args, **kwargs):
        wrapper = kwargs['rerankers']['gemma4_relevance']
        assert isinstance(wrapper, SourceAudioWindowReranker)
        assert wrapper.protocol['base_identity'] == 'verified-model-and-prompt'
        assert wrapper.config['max_window_samples'] == 480000
        return {'scope': 'partial', 'evaluated_query_count': 0}
    monkeypatch.setattr('rag_benchmark.multimodal.load_dataset', lambda _: SimpleNamespace(track='speech'))
    monkeypatch.setattr('rag_benchmark.multimodal.run_matrix', run_matrix)
    monkeypatch.setattr('rag_benchmark.multimodal_generation.LocalMultimodalGenerator', Generator)
    assert main(['run', '--dataset', str(tmp_path), '--run-dir', str(tmp_path / 'run'),
                 '--channels', 'B', '--rerankers', 'gemma4_relevance',
                 '--audio-overlength', 'source_windows_max']) == 0


def test_code_reranker_chunks_do_not_change_text_retrieval_adapters(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from rag_benchmark.multimodal_cli import main
    from rag_benchmark.multimodal_code_rerank import SharedCodeReranker
    class Generator:
        pointwise = True
        identity = 'unverified'
        def preflight(self):
            self.identity = 'verified-code-generator'
        def score(self, query, candidates):
            raise AssertionError('CLI wiring does not need model inference')
    adapters = {}
    def embedding(family, config):
        adapters[family] = object()
        return adapters[family]
    def run_matrix(*args, **kwargs):
        assert kwargs['adapters'] == {'G': adapters['bge'], 'E': adapters['embeddinggemma']}
        wrapper = kwargs['rerankers']['gemma4_relevance']
        assert isinstance(wrapper, SharedCodeReranker)
        assert wrapper.segmenter.enforce_character_limit is True
        assert wrapper.segmenter.max_chunk_chars == 4096
        assert wrapper.base.identity == 'verified-code-generator'
        return {'scope': 'partial', 'evaluated_query_count': 0}
    monkeypatch.setattr('rag_benchmark.multimodal.load_dataset', lambda _: SimpleNamespace(track='code'))
    monkeypatch.setattr('rag_benchmark.multimodal.TextEmbeddingAdapter', embedding)
    monkeypatch.setattr('rag_benchmark.multimodal.run_matrix', run_matrix)
    monkeypatch.setattr('rag_benchmark.multimodal_generation.LocalMultimodalGenerator', Generator)
    assert main(['run', '--dataset', str(tmp_path), '--run-dir', str(tmp_path / 'run'),
                 '--channels', 'G,E', '--rerankers', 'gemma4_relevance',
                 '--code-rerank-overlength', 'source_chunks_max']) == 0


def test_description_output_budget_is_explicit_and_preserves_default(tmp_path, monkeypatch):
    from rag_benchmark.multimodal_cli import main
    from rag_benchmark.multimodal_generation import LocalMultimodalGenerator
    observed = []
    def prepare(source, destination, generator, **kwargs):
        observed.append(generator)
        return {"status": "preparing"}
    monkeypatch.setattr('rag_benchmark.multimodal_generation.prepare_described_view', prepare)
    argv = ['describe', '--dataset', str(tmp_path), '--output', str(tmp_path / 'out'),
            '--caption-language', 'tr']
    assert main(argv) == 0
    assert main([*argv, '--max-output-tokens', '512']) == 0
    assert observed[0].identity == LocalMultimodalGenerator({'caption_language': 'tr'}).identity
    assert observed[0].config['max_tokens'] == 192
    assert observed[1].config['max_tokens'] == 512
    assert observed[0].identity != observed[1].identity
    with pytest.raises(SystemExit) as exc:
        main([*argv, '--max-output-tokens', '0'])
    assert exc.value.code == 2
    assert len(observed) == 2

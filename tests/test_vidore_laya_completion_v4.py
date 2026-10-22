"""Focused V4 typo regression; prior suites remain immutable and are not replayed."""
import hashlib
import json
import os
from pathlib import Path
import socket

import pytest

from rag_benchmark import vidore_laya_completion_v4 as runner


def test_v4_is_only_attribute_fix_and_version_namespace():
    root = Path(__file__).resolve().parents[1]
    for kind in ("src/rag_benchmark/vidore_laya_completion", "scripts/run_vidore_laya_completion"):
        old = (root / (kind + "_v3.py")).read_text()
        expected = old.replace("vidore_laya_completion_v3", "vidore_laya_completion_v4")
        expected = expected.replace("vidore-laya-sealed-union-v3", "vidore-laya-sealed-union-v4")
        expected = expected.replace("vidore-laya-completion-v3", "vidore-laya-completion-v4")
        expected = expected.replace("child.co_qualname__", "child.co_qualname")
        assert (root / (kind + "_v4.py")).read_text() == expected
    assert runner.VERSION == "vidore-laya-sealed-union-v4"
    assert "child.co_qualname__" not in (root / "src/rag_benchmark/vidore_laya_completion_v4.py").read_text()


@pytest.mark.skipif(os.environ.get("VIDORE_LAYA_V4_INSTALLED_SDK_REGRESSION") != "1",
                    reason="Local pinned SDK definitions-only compatibility check is explicitly opt-in")
def test_installed_sdk_definitions_bind_without_agent_tokenizer_or_model(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    recipe = root / "work/vidore-laya-completion-v3/tokenizer-preflight-recipe-v3.json"
    raw = recipe.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == "5a6b49f730c2ce5f2bdd5ef18b3aa5d6f4059476480b175c0d2c18f5521aada6"
    pins = json.loads(raw)["sdk_source_sha256"]
    assert len(pins) == 42
    def forbidden(*args, **kwargs):
        raise AssertionError("Definitions-only regression reached a forbidden execution path")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    import laya
    from laya import agent, common
    monkeypatch.setattr(agent.Agent, "__init__", forbidden)
    monkeypatch.setattr(agent, "build_model", forbidden)
    monkeypatch.setattr(common, "build_model", forbidden)
    monkeypatch.setattr(agent, "_fix_tokenizer_config", forbidden)
    monkeypatch.setattr(runner, "preflight", forbidden)
    monkeypatch.setattr(runner, "prepare_manifest", forbidden)
    monkeypatch.setattr(runner.generation, "MultimodalLayaReranker", forbidden)
    assert laya.Agent is agent.Agent
    assert runner.sdk_bindings(root, pins) == (agent, common)

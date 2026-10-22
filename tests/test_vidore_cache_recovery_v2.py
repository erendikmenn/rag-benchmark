"""V2-only regressions for loaded function global-dictionary identity."""
from pathlib import Path
import types

import pytest

from rag_benchmark import multimodal as old
from rag_benchmark import vidore_cache_recovery_v2 as recovery


REPO = Path(recovery.__file__).resolve().parents[2]


def source_pins():
    return {path: recovery.record(REPO, path)["sha256"] for path in recovery.SOURCE_FILES}


def clone_with_other_globals(function, replacements):
    clone = types.FunctionType(function.__code__, {**function.__globals__, **replacements},
                               function.__name__, function.__defaults__, function.__closure__)
    clone.__kwdefaults__ = function.__kwdefaults__
    assert clone.__code__ is function.__code__
    assert clone.__code__.co_filename == function.__code__.co_filename
    assert clone.__defaults__ == function.__defaults__
    assert clone.__kwdefaults__ == function.__kwdefaults__
    assert clone.__globals__ is not function.__globals__
    return clone


def test_real_function_globals_are_accepted():
    functions = recovery._pure(REPO, source_pins())
    assert set(functions) == {"exact_dot_rankings", "fuse_rankings", "graded_metrics", "aggregate_metrics"}
    for name, function in functions.items():
        assert function is old.__dict__[name]
        assert function.__globals__ is old.__dict__
    assert recovery._same_metrics.__globals__ is recovery.__dict__


@pytest.mark.parametrize("name", ["fuse_rankings", "channel_allocations"])
def test_pure_function_alternate_globals_fail_closed(monkeypatch, name):
    calls = []

    def forbidden_dependency(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("An alternate global dependency must never be invoked")

    clone = clone_with_other_globals(old.__dict__[name],
                                    {"channel_allocations": forbidden_dependency})
    monkeypatch.setattr(old, name, clone)
    with pytest.raises(ValueError, match="Loaded code differs from the pinned source"):
        recovery._pure(REPO, source_pins())
    assert calls == []


@pytest.mark.parametrize("name", ["_same_metrics", "validate_inputs"])
def test_own_function_alternate_globals_fail_closed(monkeypatch, name):
    calls = []

    def forbidden_dependency(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("An alternate global dependency must never be invoked")

    clone = clone_with_other_globals(recovery.__dict__[name], {"checked": forbidden_dependency})
    monkeypatch.setattr(recovery, name, clone)
    with pytest.raises(ValueError, match="Loaded code differs from the pinned source"):
        recovery._pure(REPO, source_pins())
    assert calls == []

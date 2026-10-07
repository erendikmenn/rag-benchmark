"""CPU fake-loader tests; no pretrained tensors or model artifacts are loaded."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from rag_benchmark import multimodal_bge_extra_loader as loader
import test_multimodal_bge_extra_backend as backend_tests


@pytest.fixture
def setup(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(loader.native, "normalization_evidence", lambda: {"audited_fake": True})
    monkeypatch.setattr(loader.runtime, "scheduling_evidence", lambda: {"audited_fake_schedule": True})
    monkeypatch.setattr(
        loader.backend_module.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0)
    )
    bridge = backend_tests.FakeBridge()
    bridge.config["batch_size"] = 8
    bridge.seal = bridge.state()
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "tokenizer.json").write_text(json.dumps({"fake": "json"}))
    monkeypatch.setattr(loader, "verify_artifacts", lambda _: {"fake": "verified-local-by-fixture"})

    class Tokenizer:
        padding_side = "right"
        cls_token_id = 0
        pad_token_id = 1
        eos_token_id = 2
        unk_token_id = 3
        backend_tokenizer = SimpleNamespace(to_str=lambda: json.dumps({"fake": "json"}))

        def __len__(self):
            return 250002

        def __call__(self, texts, **kwargs):
            assert kwargs == {"truncation": False, "padding": False}
            return {"input_ids": [[0, 10, 2] for _ in texts]}

    class Transformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16))
            self.config = SimpleNamespace(
                model_type="xlm-roberta", hidden_size=1024, _attn_implementation="sdpa"
            )

    class First(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Transformer()
            self.tokenizer = Tokenizer()
            self.can_flatten_inputs = False
            self.do_lower_case = False
            self.backend = "torch"
            self.processing_kwargs = {}
            self.query_length = None
            self.document_length = None
            self.transformer_task = "feature-extraction"
            self.module_output_name = "token_embeddings"
            self.modality_config = {"text": {"method": "forward", "method_output_name": "last_hidden_state"}}
            self.calls = 0

        def forward(self, features):
            self.calls += 1
            shape = features["input_ids"].shape
            hidden = torch.zeros((*shape, 1024), dtype=torch.bfloat16)
            hidden[:, :, 0] = self.model.weight
            return {**features, "token_embeddings": hidden}

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.first = First()
            self.max_seq_length = 8192
            self.eval()

        def __getitem__(self, index):
            assert index == 0
            return self.first

        def _can_flatten_inputs(self):
            return self.first.can_flatten_inputs

    model = Model()
    calls = {"model": 0, "head": 0}

    def model_load(path, config):
        calls["model"] += 1
        assert path == snapshot and config == bridge.config
        return model

    def head_load(path):
        calls["head"] += 1
        return {
            "sparse": (np.ones((1, 1024), dtype=np.float32), np.zeros(1, dtype=np.float32)),
            "colbert": (np.eye(1024, dtype=np.float32), np.zeros(1024, dtype=np.float32)),
        }

    return bridge, snapshot, model, calls, model_load, head_load


def create(setup):
    bridge, snapshot, _, _, model_load, head_load = setup
    return loader.create_extraction(
        bridge,
        snapshot,
        snapshot,
        model_loader=model_load,
        head_loader=head_load,
        instrumentation_fields={"first": ["calls"]},
    )


def test_fake_factory_load_once_same_forward_both_heads_and_cache(setup, tmp_path):
    extraction = create(setup)
    model = setup[2]
    result = extraction.extract("document", tmp_path / "cache")
    assert result["new_rows"] == 2 and result["hidden_forward_calls"] == 1
    assert model.first.calls == 1 and setup[3] == {"model": 1, "head": 1}
    assert result["factory_provenance"]["load_origin"] == "injected_not_claimed_trained"
    assert result["new_dense_encoder_calls"] == 0
    assert extraction.extract("document", tmp_path / "cache")["cached_rows"] == 2
    assert model.first.calls == 1


@pytest.mark.parametrize(
    "bad", ["flatten", "attention", "dtype", "padding", "lower", "max_length", "tokenizer"]
)
def test_actual_loaded_contract_rejected_before_forward(setup, bad):
    model = setup[2]
    if bad == "flatten":
        model.first.can_flatten_inputs = True
    elif bad == "attention":
        model.first.model.config._attn_implementation = "eager"
    elif bad == "dtype":
        model.float()
    elif bad == "padding":
        model.first.tokenizer.padding_side = "left"
    elif bad == "lower":
        model.first.do_lower_case = True
    elif bad == "max_length":
        model.max_seq_length = 100
    else:
        model.first.tokenizer.backend_tokenizer = SimpleNamespace(to_str=lambda: '{"changed":true}')
    with pytest.raises(ValueError):
        create(setup)
    assert model.first.calls == 0 and setup[3]["head"] == 0


def test_parameter_data_mutation_rejected_on_cached_request(setup, tmp_path):
    extraction = create(setup)
    extraction.extract("query", tmp_path / "cache")
    setup[2].first.model.weight.data.fill_(2)
    with pytest.raises(ValueError, match="state changed"):
        extraction.extract("query", tmp_path / "cache")
    assert setup[2].first.calls == 1


def test_wrong_native_cls_never_publishes_heads(setup, tmp_path):
    setup[2].first.model.weight.data.fill_(-1)
    extraction = create(setup)
    with pytest.raises(ValueError, match="CLS"):
        extraction.extract("query", tmp_path / "cache")
    assert not list(tmp_path.rglob("completed.json"))


def test_callback_model_mutation_rejected(setup, tmp_path):
    extraction = create(setup)
    model = setup[2]
    forward = model.first.forward

    def bad(features):
        result = forward(features)
        model.first.model.weight.data.fill_(2)
        return result

    model.first.forward = bad
    with pytest.raises(ValueError, match="state changed"):
        extraction.extract("query", tmp_path / "cache")
    assert not list(tmp_path.rglob("completed.json"))


def test_artifacts_changed_after_load_rejected(setup, monkeypatch):
    counter = iter([{"initial": True}, {"changed": True}])
    monkeypatch.setattr(loader, "verify_artifacts", lambda _: next(counter))
    with pytest.raises(ValueError, match="changed during loading"):
        create(setup)
    assert setup[3]["head"] == 0


def test_verified_artifact_bytes_fail_closed(tmp_path, monkeypatch):
    file = tmp_path / "weight.bin"
    file.write_bytes(b"fake-not-a-tensor")
    monkeypatch.setattr(loader, "ARTIFACT_HASHES", {"weight.bin": loader._digest(file)})
    assert loader.verify_artifacts(tmp_path) == loader.ARTIFACT_HASHES
    file.write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifacts differ"):
        loader.verify_artifacts(tmp_path)


def test_load_arguments_offline_exact_candidate(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    import sys

    captured = {}

    def fake(path, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(eval=lambda: None)

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=fake))
    config = {"device": "cpu", "dtype": "bfloat16"}
    model = loader._load_model(tmp_path, config)
    assert model.max_seq_length == 8192
    assert captured == {
        "device": "cpu",
        "local_files_only": True,
        "trust_remote_code": False,
        "model_kwargs": {
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "sdpa",
            "use_safetensors": False,
        },
    }


def test_tokenizer_noninteger_ids_rejected(setup, tmp_path):
    token = setup[2].first.tokenizer
    token.__class__.__call__ = lambda self, *a, **k: {"input_ids": [[0, 10.5, 2]]}
    extraction = create(setup)
    with pytest.raises(ValueError, match="integer IDs"):
        extraction.extract("query", tmp_path / "cache")
    assert setup[2].first.calls == 0


def test_factory_wrong_revision_rejected_before_load(setup):
    bridge = setup[0]
    bridge.config["revision"] = "unapproved"
    bridge.seal = bridge.state()
    with pytest.raises(ValueError, match="Pinned original G"):
        create(setup)
    assert setup[3] == {"model": 0, "head": 0}


def test_factory_provenance_mutation_rejected(setup, tmp_path):
    extraction = create(setup)
    extraction._provenance["baseline"] = "changed"
    with pytest.raises(ValueError, match="provenance"):
        extraction.extract("query", tmp_path / "cache")
    assert setup[2].first.calls == 0


def test_full_length_and_reject_overlength_without_forward(setup, tmp_path):
    token = setup[2].first.tokenizer
    token.__class__.__call__ = lambda self, *a, **k: {"input_ids": [[0] + [10] * 8190 + [2]]}
    extraction = create(setup)
    assert len(extraction.runtime.tokenize("synthetic")) == 8192
    token.__class__.__call__ = lambda self, *a, **k: {"input_ids": [[0] + [10] * 8191 + [2]]}
    extraction = create(setup)
    with pytest.raises(ValueError, match="no truncation"):
        extraction.extract("query", tmp_path / "cache")
    assert setup[2].first.calls == 0


def test_forward_replacement_preserving_cls_rejected(setup, tmp_path):
    extraction = create(setup)
    model = setup[2]
    original = model.first.forward

    def changed(features):
        value = original(features)
        value["token_embeddings"][:, 1:, 0] = 0
        value["token_embeddings"][:, 1:, 1] = 1
        return value

    model.first.forward = changed
    with pytest.raises(ValueError, match="state changed"):
        extraction.extract("query", tmp_path / "cache")
    assert model.first.calls == 0
    assert not list(tmp_path.rglob("completed.json"))


@pytest.mark.parametrize("change", ["class_forward", "hook", "pre_hook", "submodule", "tokenizer_callable"])
def test_callable_graph_and_hooks_rejected_on_cache_reuse(setup, tmp_path, change):
    extraction = create(setup)
    model = setup[2]
    extraction.extract("query", tmp_path / "cache")
    if change == "class_forward":
        original = type(model.first).forward

        def changed(self, features):
            return original(self, features)

        type(model.first).forward = changed
    elif change == "hook":
        model.first.register_forward_hook(lambda *args: None)
    elif change == "pre_hook":
        model.first.register_forward_pre_hook(lambda *args: None)
    elif change == "submodule":
        import torch

        model.first.add_module("extra", torch.nn.Identity())
    else:
        type(model.first.tokenizer).__call__ = lambda self, *a, **k: {"input_ids": [[0, 11, 2]]}
    with pytest.raises(ValueError, match="state changed"):
        extraction.extract("query", tmp_path / "cache")
    assert model.first.calls == 1


def test_callback_adds_hook_rejected_before_publication(setup, tmp_path):
    extraction = create(setup)
    model = setup[2]
    # Instrumentation is injected before sealing a new identity; its later mutation
    # must still be rejected, never accepted as an ordinary runtime.
    original = type(model.first).forward

    def changed(self, features):
        result = original(self, features)
        self.register_forward_hook(lambda *args: None)
        return result

    type(model.first).forward = changed
    extraction = create(setup)
    with pytest.raises(ValueError, match="state changed"):
        extraction.extract("query", tmp_path / "cache")
    assert not list(tmp_path.rglob("completed.json"))


def test_global_hook_rejected(setup, tmp_path):
    import torch

    extraction = create(setup)
    hook = torch.nn.modules.module.register_module_forward_hook(lambda *args: None)
    try:
        with pytest.raises(ValueError, match="state changed"):
            extraction.extract("query", tmp_path / "cache")
    finally:
        hook.remove()
    assert setup[2].first.calls == 0


@pytest.mark.parametrize("change", ["activation", "layernorm_eps", "scalar", "container", "config"])
def test_mutable_execution_fields_rejected(setup, tmp_path, change):
    import torch

    model = setup[2]
    model.first.act = lambda value: value
    model.first.add_module("activation", torch.nn.GELU())
    model.first.add_module("norm", torch.nn.LayerNorm(4).to(torch.bfloat16))
    model.first.execution_scale = 1.0
    model.first.execution_values = {"scales": [1.0]}
    extraction = create(setup)
    if change == "activation":
        model.first.activation.act = lambda value: value
    elif change == "layernorm_eps":
        model.first.norm.eps = 0.123
    elif change == "scalar":
        model.first.execution_scale = 2.0
    elif change == "container":
        model.first.execution_values["scales"][0] = 2.0
    else:
        model.first.model.config.hidden_dropout_prob = 0.1
    with pytest.raises(ValueError, match="state changed"):
        extraction.extract("query", tmp_path / "cache")
    assert model.first.calls == 0


def test_callable_instance_act_preserving_cls_rejected(setup, tmp_path):
    model = setup[2]
    model.first.act = lambda hidden: hidden
    extraction = create(setup)
    model.first.act = lambda hidden: -hidden
    with pytest.raises(ValueError, match="state changed"):
        extraction.extract("query", tmp_path / "cache")
    assert model.first.calls == 0


def test_opaque_execution_state_rejected_before_heads(setup):
    setup[2].first.opaque_execution = object()
    with pytest.raises(ValueError, match="opaque execution field"):
        create(setup)
    assert setup[3]["head"] == 0


def test_undeclared_diagnostic_counter_changes_rejected(setup, tmp_path):
    bridge, snapshot, model, _, model_load, head_load = setup
    extraction = loader.create_extraction(
        bridge, snapshot, snapshot, model_loader=model_load, head_loader=head_load
    )
    with pytest.raises(ValueError, match="state changed"):
        extraction.extract("query", tmp_path / "cache")
    assert not list(tmp_path.rglob("completed.json"))


def test_external_bound_activation_receiver_rejected(setup):
    import torch

    setup[2].first.act = torch.nn.GELU().forward
    with pytest.raises(ValueError, match="external bound executable receiver"):
        create(setup)
    assert setup[3]["head"] == 0


def test_inspected_real_gelu_activation_act_mutation_rejected(setup, tmp_path):
    pytest.importorskip("transformers")
    from transformers.activations import GELUActivation

    setup[2].first.add_module("real_gelu", GELUActivation())
    extraction = create(setup)
    setup[2].first.real_gelu.act = lambda value: value
    with pytest.raises(ValueError, match="state changed"):
        extraction.extract("query", tmp_path / "cache")
    assert setup[2].first.calls == 0


@pytest.mark.parametrize("changed", ["content", "single_word", "lstrip", "rstrip", "normalized", "special"])
def test_exact_added_token_fields_are_sealed(changed):
    tokenizers = pytest.importorskip("tokenizers")
    values = {
        "content": "<mask>",
        "single_word": False,
        "lstrip": False,
        "rstrip": False,
        "normalized": False,
        "special": True,
    }
    original = tokenizers.AddedToken(**values)
    sealed = loader._execution_value(original)
    assert sealed["fields"] == values
    assert sealed["trusted_type"] == "tokenizers.AddedToken"
    values[changed] = "<changed>" if changed == "content" else not values[changed]
    altered = tokenizers.AddedToken(**values)
    assert loader._execution_value(altered) != sealed


def test_added_token_in_tokenizer_kwargs_mutation_rejected(setup, tmp_path):
    tokenizers = pytest.importorskip("tokenizers")
    token = setup[2].first.tokenizer
    token.init_kwargs = {
        "mask_token": tokenizers.AddedToken("<mask>", special=True),
        "added_tokens_decoder": {250001: tokenizers.AddedToken("<mask>", special=True)},
    }
    extraction = create(setup)
    token.init_kwargs["added_tokens_decoder"][250001] = tokenizers.AddedToken(
        "<mask>", special=True, lstrip=True
    )
    with pytest.raises(ValueError, match="state changed"):
        extraction.extract("query", tmp_path / "cache")
    assert setup[2].first.calls == 0


@pytest.fixture(scope='module')
def real_offline_tokenizer():
    from pathlib import Path
    transformers = pytest.importorskip('transformers')
    snapshot = Path('.cache/models/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181')
    if not (snapshot / 'tokenizer.json').is_file():
        pytest.skip('Pinned tokenizer artifacts unavailable; no download')
    token = transformers.AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False, model_max_length=8192)
    return snapshot, token


def test_real_native_tokenizer_construction_contract(real_offline_tokenizer):
    snapshot, token = real_offline_tokenizer
    contract = loader.native_tokenizer_contract(snapshot, loader.ARTIFACT_HASHES)
    assert contract['condition'] == 'bge-native-tokenizer-construction-v2'
    assert contract['expected_state_sha256'] == loader.stable_hash(loader._tokenizer_state(token))
    assert contract['historical_baseline_tokenizer_identity'] == 'unrecorded_not_claimed'


def test_real_tokenizer_ordinary_padding_state_and_full_input_preserved(real_offline_tokenizer):
    _, token = real_offline_tokenizer
    token(['hello', 'hello world'], padding=False, truncation=False)
    initial = loader._tokenizer_state(token)
    plain = token([' hello  world ', 'a\nb'], padding=False, truncation=False)['input_ids']
    token(['hello', 'hello world'], padding=True, truncation=False)
    assert json.loads(token.backend_tokenizer.to_str())['padding'] is not None
    assert loader._tokenizer_state(token) == initial
    again = token([' hello  world ', 'a\nb'], padding=False, truncation=False)['input_ids']
    assert plain == again
    assert loader._tokenizer_state(token) == initial


@pytest.mark.parametrize('mutation', ['fixed_padding', 'left_padding', 'wrong_pad_id', 'truncation', 'normalizer'])
def test_real_tokenizer_meaningful_backend_mutation_rejected(real_offline_tokenizer, mutation):
    snapshot, _ = real_offline_tokenizer
    from transformers import AutoTokenizer
    token = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False, model_max_length=8192)
    initial = loader.stable_hash(loader._tokenizer_state(token))
    if mutation == 'fixed_padding':
        token.backend_tokenizer.enable_padding(length=10, pad_id=1, pad_token='<pad>')
    elif mutation == 'left_padding':
        token.backend_tokenizer.enable_padding(direction='left', pad_id=1, pad_token='<pad>')
    elif mutation == 'wrong_pad_id':
        token.backend_tokenizer.enable_padding(pad_id=2, pad_token='<pad>')
    elif mutation == 'truncation':
        token.backend_tokenizer.enable_truncation(max_length=10)
    else:
        from tokenizers.normalizers import Lowercase
        token.backend_tokenizer.normalizer = Lowercase()
    if mutation == 'normalizer':
        assert loader.stable_hash(loader._tokenizer_state(token)) != initial
    else:
        with pytest.raises(ValueError, match='state unsupported'):
            loader._tokenizer_state(token)


def test_native_factory_source_version_or_artifact_change_fails_closed(real_offline_tokenizer, monkeypatch):
    snapshot, _ = real_offline_tokenizer
    with pytest.raises(ValueError, match='ten verified'):
        loader.native_tokenizer_contract(snapshot, {'tokenizer.json': loader.ARTIFACT_HASHES['tokenizer.json']})
    monkeypatch.setitem(loader._NATIVE_LIBRARIES, 'transformers', 'unsupported')
    with pytest.raises(ValueError, match='Unsupported native'):
        loader.native_tokenizer_contract(snapshot, loader.ARTIFACT_HASHES)


def test_actual_tokenizer_runtime_gate_accepts_native_contract_and_padding(setup, real_offline_tokenizer):
    snapshot, _ = real_offline_tokenizer
    from transformers import AutoTokenizer
    token = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False, model_max_length=8192)
    token(['hello', 'hello world'], padding=False, truncation=False)
    bridge, _, model, *_ = setup
    model.first.tokenizer = token
    stored = json.loads((snapshot / 'tokenizer.json').read_bytes())
    with pytest.raises(ValueError, match='Actual loaded tokenizer bytes/semantics differ'):
        loader._LoadedRuntime(model, bridge.config, loader.ARTIFACT_HASHES, stored, {'first': ['calls']})
    contract = loader.native_tokenizer_contract(snapshot, loader.ARTIFACT_HASHES)
    runtime = loader._LoadedRuntime(model, bridge.config, loader.ARTIFACT_HASHES,
                                   json.loads((snapshot / 'tokenizer.json').read_bytes()),
                                   {'first': ['calls']}, contract)
    text = ' hello  world '
    expected = token([text], padding=False, truncation=False)['input_ids'][0]
    token(['hello', 'hello world'], padding=True, truncation=False)
    assert runtime.tokenize(text) == expected
    assert runtime.calls == model.first.calls == 0
    from tokenizers.normalizers import Lowercase
    token.backend_tokenizer.normalizer = Lowercase()
    with pytest.raises(ValueError, match='semantics differ'):
        runtime.tokenize(text)

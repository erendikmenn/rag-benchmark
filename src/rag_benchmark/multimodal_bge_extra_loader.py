"""Offline pinned BGE factory; tests inject loaders, never pretrained artifacts.

Candidate attention is explicit, never attributed to the historical baseline.
Completion requires every-row strict native CLS proof in the published backend.
Parameter hashing is deliberately conservative and a production performance gate.
"""

from __future__ import annotations
import copy
import hashlib
import inspect
import functools
import marshal
import json
from pathlib import Path

import numpy as np

from . import multimodal_bge_extra_backend as backend_module
from . import multimodal_bge_extra_native as native
from . import multimodal_bge_extra_runtime as runtime
from .models import MODEL_DEFAULTS, stable_hash

ARTIFACT_HASHES = {
    "pytorch_model.bin": "b5e0ce3470abf5ef3831aa1bd5553b486803e83251590ab7ff35a117cf6aad38",
    "config.json": "26159e7ad065073448460117eb24b7a4572f6f4e78eadff65dc0a11c052449fa",
    "sentence_bert_config.json": "eb9b44b13c0f52a3b3685c3b1cbdea1ba8b04bea123b98f61610048940776eb1",
    "config_sentence_transformers.json": "1eef72430e7194a1e59680e635aed81ffa083f05668dbc5bb1c56c04c0999c38",
    "modules.json": native.MODULES_SHA256,
    "1_Pooling/config.json": native.POOLING_SHA256,
    "tokenizer.json": "21106b6d7dab2952c1d496fb21d5dc9db75c28ed361a05f5020bbba27810dd08",
    "tokenizer_config.json": "a62b2b6784f990259fddef5f16388693a8043be4f69179e6a5257eeb3f9abac4",
    "special_tokens_map.json": "8c785abebea9ae3257b61681b4e6fd8365ceafde980c21970d001e834cf10835",
    "sentencepiece.bpe.model": "cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865",
}
_IMPLEMENTATION = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


def verify_artifacts(snapshot):
    snapshot = Path(snapshot)
    actual = {name: _digest(snapshot / name) for name in ARTIFACT_HASHES}
    if actual != ARTIFACT_HASHES or (snapshot / "2_Normalize/config.json").exists():
        raise ValueError("Pinned model/tokenizer artifacts differ")
    return actual


def _load_model(snapshot, config):
    import torch
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        str(snapshot),
        device=config["device"],
        local_files_only=True,
        trust_remote_code=False,
        model_kwargs={
            "torch_dtype": getattr(torch, config["dtype"]),
            "attn_implementation": "sdpa",
            "use_safetensors": False,
        },
    )
    model.max_seq_length = 8192
    model.eval()
    return model


def _parameters(model):
    """Hash exact native bytes in bounded CPU chunks, including mutable tensors."""
    import torch

    rows = []
    for name, value in list(model.named_parameters()) + list(model.named_buffers()):
        if not value.is_contiguous():
            raise ValueError("Noncontiguous runtime tensor unsupported")
        if value.is_floating_point():
            flat = value.detach().reshape(-1)
            for start in range(0, flat.numel(), 262144):
                if not torch.isfinite(flat[start : start + 262144]).all().item():
                    raise ValueError("Nonfinite runtime parameter")
        data = value.detach().reshape(-1).view(torch.uint8)
        digest = hashlib.sha256()
        for start in range(0, data.numel(), 1024 * 1024):
            digest.update(data[start : start + 1024 * 1024].cpu().numpy().tobytes())
        rows.append(
            {
                "name": name,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "device": str(value.device),
                "sha256": digest.hexdigest(),
            }
        )
    if not rows:
        raise ValueError("Loaded native parameters unavailable")
    return rows


def _callable_seal(value, seen=None):
    """Bind Python implementation/defaults/closures, not process memory addresses."""
    seen = set() if seen is None else seen
    if isinstance(value, functools.partial):
        return {
            "partial": _callable_seal(value.func, seen),
            "args": _execution_value(value.args),
            "kwargs": _execution_value(value.keywords),
        }
    value = getattr(value, "__func__", value)
    name = getattr(value, "__module__", "") + "." + getattr(value, "__qualname__", type(value).__qualname__)
    if id(value) in seen:
        return {"recursive": name}
    if inspect.isfunction(value):
        seen.add(id(value))

        def constant(item):
            if item is None or isinstance(item, (str, int, float, bool)):
                return item
            if isinstance(item, (tuple, list)):
                return [constant(x) for x in item]
            if isinstance(item, dict):
                return {str(k): constant(v) for k, v in item.items()}
            if inspect.isfunction(item) or inspect.ismethod(item):
                return _callable_seal(item, seen)
            if inspect.ismodule(item):
                return {"module": item.__name__, "version": str(getattr(item, "__version__", ""))}
            if inspect.isclass(item):
                return {"class": item.__module__ + "." + item.__qualname__}
            raise ValueError("Loaded runtime state changed: opaque callable closure unsupported")

        return {
            "name": name,
            "code_sha256": hashlib.sha256(marshal.dumps(value.__code__)).hexdigest(),
            "defaults": constant(value.__defaults__),
            "kwdefaults": constant(value.__kwdefaults__),
            "closure": [constant(cell.cell_contents) for cell in (value.__closure__ or ())],
        }
    if inspect.isbuiltin(value) or inspect.ismethoddescriptor(value):
        return {"name": name, "native_type": type(value).__qualname__}
    raise ValueError("Loaded runtime state changed: unverified callable type")


def _execution_value(value):
    """Structured execution data only; opaque objects are never repr-hashed."""
    import torch
    from types import SimpleNamespace

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (torch.dtype, torch.device)):
        return {"torch_type": str(value)}
    if type(value).__module__ == "tokenizers" and type(value).__name__ == "AddedToken":
        from tokenizers import AddedToken
        from importlib.metadata import version

        if type(value) is not AddedToken:
            raise ValueError("Loaded runtime state changed: untrusted AddedToken type")
        fields = {
            name: getattr(value, name)
            for name in ("content", "single_word", "lstrip", "rstrip", "normalized", "special")
        }
        if type(fields["content"]) is not str or any(
            type(fields[name]) is not bool for name in fields if name != "content"
        ):
            raise ValueError("Loaded runtime state changed: malformed AddedToken fields")
        return {
            "trusted_type": "tokenizers.AddedToken",
            "package_version": version("tokenizers"),
            "fields": fields,
        }
    if isinstance(value, bytes):
        return {"bytes_sha256": hashlib.sha256(value).hexdigest()}
    if isinstance(value, (list, tuple)):
        return [_execution_value(x) for x in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_execution_value(x) for x in value), key=stable_hash)
    if isinstance(value, dict):
        return sorted(([_execution_value(k), _execution_value(v)] for k, v in value.items()), key=stable_hash)
    if hasattr(value, "backend_tokenizer"):
        return {
            "tokenizer": stable_hash(json.loads(value.backend_tokenizer.to_str())),
            "callable": _callable_seal(type(value).__call__),
            "settings": {
                name: _execution_value(getattr(value, name, None))
                for name in ("padding_side", "truncation_side", "model_max_length", "init_kwargs")
            },
        }
    if callable(value):
        return {"callable": _callable_seal(value)}
    cls = type(value)
    if cls.__module__ == "sentence_transformers.base.modality" and cls.__name__ == "InputFormatter":
        source_hash = hashlib.sha256(Path(inspect.getfile(cls)).read_bytes()).hexdigest()
        if source_hash != "e7defe33c33caf56e4279423a76585c6468622115a3aa58c019c578ed74fb74e":
            raise ValueError("Loaded runtime state changed: pinned input formatter source differs")
        return {
            "class": cls.__module__ + "." + cls.__qualname__,
            "source_sha256": source_hash,
            "fields": _execution_value(vars(value)),
        }
    if (
        isinstance(value, SimpleNamespace)
        or cls.__module__.startswith("transformers.configuration_utils")
        or (cls.__module__.startswith("transformers.models.") and cls.__name__.endswith("Config"))
    ):
        return {"class": cls.__module__ + "." + cls.__qualname__, "fields": _execution_value(vars(value))}
    raise ValueError(
        "Loaded runtime state changed: unsupported opaque execution field "
        + cls.__module__
        + "."
        + cls.__qualname__
    )


def _execution_fields(module, ignored):
    # Registered tensors and children are independently checked by byte hashes
    # and the submodule graph. Model-card metadata is explicitly administrative.
    structural = {"_parameters", "_buffers", "_modules"}
    fields = {}
    for name, value in vars(module).items():
        if name in structural:
            continue
        if name in ignored:
            if type(value) is not int or value < 0:
                raise ValueError("Loaded runtime state changed: invalid declared diagnostic counter")
            fields[name] = {"explicit_diagnostic_counter": True}
        elif name == "model_card_data":
            if not type(value).__module__.startswith("sentence_transformers.model_card"):
                raise ValueError("Loaded runtime state changed: unverified model card metadata")
            fields[name] = {"administrative_class": type(value).__module__ + "." + type(value).__qualname__}
        else:
            target = value.func if isinstance(value, functools.partial) else value
            if (
                inspect.ismethod(target)
                and target.__self__ is not module
                and not inspect.isclass(target.__self__)
            ):
                raise ValueError(
                    "Loaded runtime state changed: external bound executable receiver unsupported"
                )
            fields[name] = _execution_value(value)
    return fields


def _graph_seal(model, instrumentation):
    """No hooks or instance forward overrides are implicitly approved."""
    import torch

    global_hooks = torch.nn.modules.module
    for name in (
        "_global_forward_hooks",
        "_global_forward_pre_hooks",
        "_global_backward_hooks",
        "_global_backward_pre_hooks",
    ):
        if getattr(global_hooks, name, {}):
            raise ValueError("Loaded runtime state changed: global hooks unsupported")
    rows = []
    for name, module in model.named_modules(remove_duplicate=False):
        methods = {}
        for method in (
            "forward",
            "__call__",
            "_call_impl",
            "_wrapped_call_impl",
            "named_parameters",
            "named_buffers",
            "named_modules",
            "__getitem__",
            "_can_flatten_inputs",
        ):
            if not hasattr(type(module), method):
                if method in vars(module):
                    raise ValueError("Loaded runtime state changed: instance callable override")
                continue
            if method in vars(module):
                raise ValueError("Loaded runtime state changed: instance callable override")
            methods[method] = _callable_seal(getattr(type(module), method))
        if any(
            vars(module).get(key)
            for key in (
                "_forward_hooks",
                "_forward_pre_hooks",
                "_backward_hooks",
                "_backward_pre_hooks",
                "_state_dict_hooks",
                "_load_state_dict_pre_hooks",
            )
        ):
            raise ValueError("Loaded runtime state changed: module hooks unsupported")
        if (
            getattr(module, "_hf_hook", None) is not None
            or getattr(module, "_compiled_call_impl", None) is not None
        ):
            raise ValueError("Loaded runtime state changed: unverified instrumentation")
        rows.append(
            {
                "path": name,
                "class": type(module).__module__ + "." + type(module).__qualname__,
                "methods": methods,
                "training": module.training,
                "execution_fields": _execution_fields(module, instrumentation.get(name, [])),
            }
        )
    return rows


class _LoadedRuntime:
    def __init__(self, model, config, artifacts, tokenizer_json, instrumentation=None):
        self.model = model
        self._instrumentation = copy.deepcopy(instrumentation or {})
        if any(
            not isinstance(path, str) or not isinstance(names, list) or any(name != "calls" for name in names)
            for path, names in self._instrumentation.items()
        ):
            raise ValueError("Only explicitly declared diagnostic calls counters are supported")
        self.config = copy.deepcopy(config)
        self.artifacts = copy.deepcopy(artifacts)
        self._tokenizer_json = stable_hash(tokenizer_json)
        self._source = _IMPLEMENTATION
        self._initial = self._state()
        self._initial_seal = stable_hash(self._initial)
        self.identity = stable_hash(
            {
                "condition": "pinned-bge-candidate-sdpa-runtime-v1",
                "state": self._initial,
                "implementation": _IMPLEMENTATION,
                "historical_attention": "unknown_not_claimed",
                "completion_gate": "every_row_native_CLS_atol_2e-6_rtol_0",
            }
        )
        self._identity = self.identity
        self.can_flatten_inputs = False
        self.calls = 0

    def _state(self):
        import torch

        first = self.model[0]
        token = first.tokenizer
        cfg = first.model.config
        if (
            self.model.training
            or first.model.training
            or self.model.max_seq_length != 8192
            or first.can_flatten_inputs is not False
            or self.model._can_flatten_inputs() is not False
            or first.do_lower_case is not False
            or first.processing_kwargs != {}
            or first.query_length is not None
            or first.document_length is not None
            or first.backend != "torch"
            or first.transformer_task != "feature-extraction"
            or first.module_output_name != "token_embeddings"
            or first.modality_config
            != {"text": {"method": "forward", "method_output_name": "last_hidden_state"}}
            or cfg.model_type != "xlm-roberta"
            or cfg.hidden_size != 1024
            or cfg._attn_implementation != "sdpa"
            or token.padding_side != "right"
            or [token.cls_token_id, token.pad_token_id, token.eos_token_id, token.unk_token_id]
            != [0, 1, 2, 3]
            or len(token) != 250002
        ):
            raise ValueError("Actual loaded attention/tokenizer/native contract unsupported")
        if stable_hash(json.loads(token.backend_tokenizer.to_str())) != self._tokenizer_json:
            raise ValueError("Actual loaded tokenizer bytes/semantics differ")
        graph = _graph_seal(self.model, self._instrumentation)
        token_callable = _callable_seal(type(token).__call__)
        if "__call__" in vars(token):
            raise ValueError("Loaded runtime state changed: tokenizer callable override")
        parameters = _parameters(self.model)
        expected_dtype = str(getattr(torch, self.config["dtype"]))
        expected_device = self.config["device"]
        for row in parameters:
            if row["dtype"].startswith("torch.float") or row["dtype"] == "torch.bfloat16":
                if row["dtype"] != expected_dtype or row["device"].split(":")[0] != expected_device:
                    raise ValueError("Loaded dtype/device differs from original G")
        return {
            "config": self.config,
            "artifacts": self.artifacts,
            "parameters": parameters,
            "graph": graph,
            "instrumentation": self._instrumentation,
            "token_callable": token_callable,
            "tokenizer": self._tokenizer_json,
            "attention": "sdpa",
            "flattened": False,
            "class": type(self.model).__module__ + "." + type(self.model).__qualname__,
        }

    def verify(self):
        if (
            self.identity != self._identity
            or self.can_flatten_inputs is not False
            or hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != self._source
            or stable_hash(self._initial) != self._initial_seal
            or self._state() != self._initial
        ):
            raise ValueError("Loaded runtime state changed")

    def tokenize(self, text):
        self.verify()
        encoded = self.model[0].tokenizer([text], truncation=False, padding=False)
        ids = encoded["input_ids"]
        if len(ids) != 1 or not 2 <= len(ids[0]) <= 8192:
            raise ValueError("Full tokenizer input exceeds native limit; no truncation")
        values = np.asarray(ids[0])
        if values.ndim != 1 or values.dtype.kind not in "iu":
            raise ValueError("Actual tokenizer integer IDs required")
        result = [int(i) for i in values]
        self.verify()
        return result

    def forward(self, tokens):
        import torch

        self.verify()
        frozen = stable_hash(tokens)
        width = max(map(len, tokens))
        ids = np.full((len(tokens), width), 1, dtype=np.int64)
        mask = np.zeros_like(ids)
        for index, row in enumerate(tokens):
            ids[index, : len(row)] = row
            mask[index, : len(row)] = 1
        features = {
            "input_ids": torch.tensor(ids, device=self.config["device"]),
            "attention_mask": torch.tensor(mask, device=self.config["device"]),
        }
        with torch.inference_mode():
            self.calls += 1
            output = self.model[0](features)
        self.verify()
        if stable_hash(tokens) != frozen:
            raise ValueError("Runtime mutated full token request")
        actual_ids = output["input_ids"].detach().cpu().numpy().copy()
        actual_mask = output["attention_mask"].detach().cpu().numpy().copy()
        return {"hidden": output["token_embeddings"], "token_ids": actual_ids, "attention_mask": actual_mask}


class LoadedExtraction:
    """Separate factory provenance; published injected backend identity stays intact."""

    def __init__(self, backend, native_runtime, provenance):
        self.backend = backend
        self.runtime = native_runtime
        self._provenance = copy.deepcopy(provenance)
        self._seal = stable_hash(self._provenance)
        self._backend_identity = backend.identity
        self._runtime_identity = native_runtime.identity

    def extract(self, role, cache_dir, **kwargs):
        if (
            stable_hash(self._provenance) != self._seal
            or self.backend.identity != self._backend_identity
            or self.runtime.identity != self._runtime_identity
        ):
            raise ValueError("Factory provenance changed")
        self.runtime.verify()
        result = self.backend.extract(role, cache_dir, **kwargs)
        self.runtime.verify()
        return {
            **result,
            "factory_provenance": copy.deepcopy(self._provenance),
            "compatibility_scope": "every_processed_row_only",
            "full_runtime_health_completed": False,
        }


def create_extraction(
    baseline, snapshot, heads_directory, *, model_loader=None, head_loader=None, instrumentation_fields=None
):
    """Explicit future load operation, not executed by import or this-stage tests."""
    baseline.validate_for_reuse()
    config = baseline.config
    if any(config.get(k) != MODEL_DEFAULTS["bge"][k] for k in ("model_id", "revision", "dimension")):
        raise ValueError("Pinned original G model required")
    native.NativeNormalizationPolicy(config, baseline_identity=baseline.identity)
    evidence = runtime.scheduling_evidence()
    snapshot = Path(snapshot)
    artifacts = verify_artifacts(snapshot)
    tokenizer_json = json.loads((snapshot / "tokenizer.json").read_bytes())
    model = (model_loader or _load_model)(snapshot, copy.deepcopy(config))
    loaded = _LoadedRuntime(model, config, artifacts, tokenizer_json, instrumentation_fields)
    if verify_artifacts(snapshot) != artifacts:
        raise ValueError("Artifacts changed during loading")
    heads = (head_loader or backend_module.verified_trained_head_arrays)(heads_directory)
    loaded.verify()
    baseline.validate_for_reuse()
    plan = runtime.PinnedBatchPlan(baseline)
    backend = backend_module.InjectedDualHeadBackend(
        baseline, tokenizer=loaded, hidden_backend=loaded, head_arrays=heads, batch_plan=plan
    )
    injected = model_loader is not None or head_loader is not None
    return LoadedExtraction(
        backend,
        loaded,
        {
            "condition": "pinned-bge-candidate-sdpa-factory-v1",
            "baseline": baseline.identity,
            "batch_plan": plan.identity,
            "artifact_hashes": artifacts,
            "evidence": evidence,
            "native_dtype": config["dtype"],
            "device": config["device"],
            "head_and_storage_dtype": "float32",
            "head_hashes": backend_module.core.HEAD_HASHES,
            "load_origin": "injected_not_claimed_trained" if injected else "verified_official_artifacts",
            "historical_attention": "unknown_not_claimed",
            "instrumentation_fields": copy.deepcopy(instrumentation_fields or {}),
            "new_dense_encoder_calls": 0,
            "implementation": _IMPLEMENTATION,
        },
    )

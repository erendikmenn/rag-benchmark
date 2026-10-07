"""Original G native normalization policy, using injected hidden tensors only.

No model, trained-head loader, forward backend, cache writer or CLI exists here.
Synthetic CPU tensors exercise arithmetic; actual MPS compatibility is unmeasured.
Sparse/ColBERT heads remain a future separate FP32 head/storage contract.
"""
from __future__ import annotations

import copy
import hashlib
import json
from importlib.metadata import distribution
from pathlib import Path

import numpy as np

from . import models
from .models import MODEL_DEFAULTS, package_versions, stable_hash
from .multimodal_bge_extra import DIMENSION, VOCABULARY, FORMULA_COMMIT, FORMULA_HASHES, HEAD_HASHES

POLICY = "module_normalize_then_encode_normalize_then_fp32_renormalize"
POLICY_VERSION = 1
NORMALIZE_SOURCE_SHA256 = "70df2751f2b7af5312ff5ea52c1650eb566e186bc08312a89457e587c059024f"
MODULE_LOADER_SHA256 = "082aa947667a3db02a0ed57be76af7bf79b72ff85d6b97d2c257a19d03be4fe0"
FILE_LOADER_SHA256 = "cf32f552607a4bf40f77bef9de5ba7193cb772a0749e382836d1524126279b2a"
MODULES_SHA256 = "84e40c8e006c9b1d6c122e02cba9b02458120b5fb0c87b746c41e0207cf642cf"
POOLING_SHA256 = "e54c164a07274f2eb45bb724f54a79d1efcc90c41573887cd9a29aeee0597352"
ST_SOURCE_SHA256 = "3d677757bf3e684c45d3634ae67918cd9a67cee6aac6760e5fa1c72dfe820650"
DENSE_SOURCE_SHA256 = "5fcf6f8bb9093dc8a2351bf8eb1522883f0522793d2caf3e953865be9c2284ef"
_IMPLEMENTATION = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def normalization_evidence():
    """Require exactly the inspected installed arithmetic contract; no download."""
    try:
        package = distribution("sentence-transformers")
        path = Path(package.locate_file("sentence_transformers/sentence_transformer/model.py"))
        st_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        dense_hash = hashlib.sha256(Path(models.__file__).read_bytes()).hexdigest()
        sources = {name: hashlib.sha256(Path(package.locate_file(relative)).read_bytes()).hexdigest()
            for name, relative in {
                "normalize": "sentence_transformers/base/modules/normalize.py",
                "module_loader": "sentence_transformers/base/modules/module.py",
                "file_loader": "sentence_transformers/util/file_io.py"}.items()}
        if (package.version != "6.1.0" or st_hash != ST_SOURCE_SHA256 or dense_hash != DENSE_SOURCE_SHA256
                or sources != {"normalize": NORMALIZE_SOURCE_SHA256, "module_loader": MODULE_LOADER_SHA256, "file_loader": FILE_LOADER_SHA256}):
            raise ValueError("Native normalization contract source differs or is ambiguous")
        snapshot = Path('.cache/models/models--BAAI--bge-m3/snapshots') / MODEL_DEFAULTS['bge']['revision']
        module_bytes = (snapshot / 'modules.json').read_bytes()
        pooling_bytes = (snapshot / '1_Pooling/config.json').read_bytes()
        if (snapshot / '2_Normalize/config.json').exists():
            raise ValueError("Pinned Normalize defaults differ; unexpected config")
        if hashlib.sha256(module_bytes).hexdigest() != MODULES_SHA256 or hashlib.sha256(pooling_bytes).hexdigest() != POOLING_SHA256:
            raise ValueError("Pinned BGE module/CLS pooling contract differs")
        if [row['type'] for row in json.loads(module_bytes)] != [
            'sentence_transformers.models.Transformer', 'sentence_transformers.models.Pooling', 'sentence_transformers.models.Normalize']:
            raise ValueError("Pinned Normalize module chain differs")
        pooling = json.loads(pooling_bytes)
        if pooling.get('pooling_mode_cls_token') is not True or pooling.get('word_embedding_dimension') != DIMENSION:
            raise ValueError("Pinned native CLS pooling differs")
    except (OSError, LookupError, ModuleNotFoundError) as error:
        raise ValueError("Native normalization contract source unavailable") from error
    return {"sentence_transformers_version": package.version, "sentence_transformer_source_sha256": st_hash,
            "dense_models_source_sha256": dense_hash, "normalization_module_sources": sources,
            "modules_sha256": MODULES_SHA256, "cls_pooling_sha256": POOLING_SHA256,
            "module_normalize_config": "absent_default_sentence_embedding", "native_torch_normalization_count": 2}


class NativeNormalizationPolicy:
    """Exact original three-stage arithmetic and strict reused-vector comparison.

The caller must obtain baseline arrays via VerifiedGBaseline.verified_inputs.
Baseline identity is explicit, independently bound from FP32 extra-head metrics.
"""
    def __init__(self, config, *, baseline_identity):
        if any(config.get(key) != MODEL_DEFAULTS['bge'][key] for key in ('model_id', 'revision', 'dimension')):
            raise ValueError("Pinned native BGE contract required")
        if config.get('dtype') not in {'bfloat16', 'float32'} or config.get('device') not in {'cpu', 'mps'}:
            raise ValueError("Explicit native dtype/device contract required")
        if type(config.get('batch_size')) is not int or config['batch_size'] < 1 or type(config.get('max_length', 8192)) is not int or config.get('max_length', 8192) != 8192:
            raise ValueError("Original batch size and full native token limit required")
        if not isinstance(baseline_identity, str) or not baseline_identity:
            raise ValueError("Verified original G baseline identity required")
        self._config = copy.deepcopy(config)
        self._baseline_identity = baseline_identity
        self._evidence = normalization_evidence()
        self._packages = package_versions(('sentence-transformers', 'transformers', 'torch', 'numpy'))
        self._identity = stable_hash({'policy': POLICY, 'policy_version': POLICY_VERSION, 'config': self._config, 'baseline': baseline_identity,
            'evidence': self._evidence, 'packages': self._packages, 'implementation': _IMPLEMENTATION,
            'cls_comparison_atol': 2e-6, 'cls_comparison_rtol': 0,
            'formula_commit': FORMULA_COMMIT, 'formula_hashes': FORMULA_HASHES, 'head_hashes': HEAD_HASHES,
            'extra_head_arithmetic': 'separate_fp32_projection_normalization_scoring_v1',
            'token_vector_storage': 'float32', 'normalization_fallback': 'forbidden'})
        self._seal = self._state()

    def _state(self):
        return stable_hash({'config': self._config, 'baseline': self._baseline_identity,
                            'evidence': self._evidence, 'packages': self._packages, 'identity': self._identity})

    def _verify_contract(self):
        if self._state() != self._seal or normalization_evidence() != self._evidence:
            raise ValueError("Native normalization frozen contract changed")
        if package_versions(('sentence-transformers', 'transformers', 'torch', 'numpy')) != self._packages:
            raise ValueError("Native normalization packages changed")

    @property
    def identity(self):
        return self._identity

    @property
    def config(self):
        return copy.deepcopy(self._config)

    def verify_hidden(self, hidden, token_ids, attention_mask, expected_tokens, baseline_dense):
        """Derive CLS from same real-token hidden states; no supplied-dense override."""
        self._verify_contract()
        import torch
        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3 or hidden.shape[2] != DIMENSION or not len(hidden) or not 2 <= hidden.shape[1] <= 8192:
            raise ValueError("Native hidden tensor shape/type required")
        if hidden.dtype != getattr(torch, self._config['dtype']) or hidden.device.type != self._config['device']:
            raise ValueError("Native hidden tensor dtype/device differs from original baseline")
        if not torch.isfinite(hidden).all().item():
            raise ValueError("Native hidden tensor is nonfinite")
        tokens = []
        for row in expected_tokens:
            array = np.asarray(row)
            if array.ndim != 1 or array.dtype.kind not in 'iu' or not 2 <= len(array) <= 8192 or np.any(array < 0) or np.any(array >= VOCABULARY) or array[0] != 0 or array[-1] != 2:
                raise ValueError("Full native token contract invalid")
            tokens.append([int(value) for value in array])
        ids, masks = np.asarray(token_ids), np.asarray(attention_mask)
        if ids.dtype.kind not in 'iu' or ids.shape != masks.shape or ids.shape != tuple(hidden.shape[:2]) or len(tokens) != len(hidden) or not np.isin(masks, [0, 1]).all():
            raise ValueError("Actual native IDs/mask shape or type differs")
        for index, row in enumerate(tokens):
            expected_mask = np.arange(ids.shape[1]) < len(row)
            if not np.array_equal(masks[index].astype(bool), expected_mask) or not np.array_equal(ids[index][expected_mask], row) or np.any(ids[index][~expected_mask] != 1):
                raise ValueError("Actual native full tokens/right-padding/position-zero CLS differs")
        if np.asarray(baseline_dense).dtype != np.float32:
            raise ValueError("Verified baseline score representation must be float32")
        baseline = np.array(baseline_dense, dtype=np.float32, copy=True)
        if baseline.shape != (len(hidden), DIMENSION) or not np.isfinite(baseline).all():
            raise ValueError("Verified baseline array coverage/shape invalid")
        if not np.allclose(np.linalg.norm(baseline, axis=1), 1, atol=2e-6, rtol=0):
            raise ValueError("Verified baseline vectors must be finite nonzero unit vectors")
        with torch.no_grad():
            # Keep original native dtype/device through first normalization.
            module_normalized = torch.nn.functional.normalize(hidden[:, 0].detach(), p=2, dim=-1)
            normalized = torch.nn.functional.normalize(module_normalized, p=2, dim=1)
            values = normalized.cpu().float().numpy().copy()
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        if not np.isfinite(values).all() or not np.isfinite(norms).all() or np.any(norms <= 0):
            raise ValueError("Native normalization produced zero/nonfinite dense values")
        dense = values / norms
        if not np.isfinite(dense).all() or not np.allclose(dense, baseline, atol=2e-6, rtol=0):
            raise ValueError("Actual native normalized CLS differs from verified reused baseline")
        self._verify_contract()
        dense.flags.writeable = False
        return dense, {'policy': POLICY, 'policy_version': POLICY_VERSION, 'policy_identity': self.identity,
            'baseline_identity': self._baseline_identity, 'cls_verified_atol': 2e-6,
            'full_token_ids_sha256': stable_hash(tokens), 'hidden_state_dtype': self._config['dtype'],
            'baseline_score_dtype': 'float32', 'extra_token_storage_dtype': 'float32',
            'new_dense_encoder_calls': 0, 'actual_model_compatibility_measured': False}

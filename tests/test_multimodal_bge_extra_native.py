"""Synthetic CPU tensors only: no pretrained tensors, model or head loads."""
import json
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import pytest
from rag_benchmark import multimodal_bge_extra_native as m


@pytest.fixture(autouse=True)
def synthetic_audited_contract(request, monkeypatch):
    """Arithmetic fixtures never require a model cache or download preparation."""
    if request.node.name in {'test_contract_evidence_missing_or_changed', 'test_actual_cached_normalization_contract'}:
        return
    evidence = {'sentence_transformer_source_sha256': m.ST_SOURCE_SHA256,
        'dense_models_source_sha256': m.DENSE_SOURCE_SHA256,
        'normalization_module_source_sha256': m.NORMALIZE_SOURCE_SHA256,
        'modules_sha256': m.MODULES_SHA256, 'cls_pooling_sha256': m.POOLING_SHA256,
        'module_normalize_config': 'absent_default_sentence_embedding', 'native_torch_normalization_count': 2}
    monkeypatch.setattr(m, 'normalization_evidence', lambda: dict(evidence))


def config(dtype='bfloat16'):
    return {**m.MODEL_DEFAULTS['bge'], 'dtype': dtype, 'device': 'cpu', 'batch_size': 8}


def policy(dtype='bfloat16'):
    return m.NativeNormalizationPolicy(config(dtype), baseline_identity='verified-synthetic-baseline')


def sample(dtype='bfloat16', values=None):
    torch = pytest.importorskip('torch')
    hidden = torch.zeros((1, 3, 1024), dtype=getattr(torch, dtype), device='cpu')
    hidden[0, 0, :3] = torch.tensor(values or [1.2, 2.3, 3.4], dtype=getattr(torch, dtype))
    # Independently reproduce inspected original baseline normalization sequence.
    module_output = torch.nn.functional.normalize(hidden[:, 0], p=2, dim=-1)
    first = torch.nn.functional.normalize(module_output, p=2, dim=1)
    baseline = np.asarray([row.float().numpy() if dtype == 'bfloat16' else row.numpy() for row in first], dtype=np.float32)
    baseline = baseline / np.linalg.norm(baseline, axis=1, keepdims=True)
    return hidden, [[0, 10, 2]], [[1, 1, 1]], baseline


@pytest.mark.parametrize('dtype', ['float32', 'bfloat16'])
def test_actual_original_two_stage_sequence(dtype):
    hidden, ids, mask, baseline = sample(dtype)
    result, usage = policy(dtype).verify_hidden(hidden, ids, mask, ids, baseline)
    np.testing.assert_array_equal(result, baseline)
    assert not result.flags.writeable
    assert usage['new_dense_encoder_calls'] == 0 and not usage['actual_model_compatibility_measured']
    assert usage['baseline_score_dtype'] == usage['extra_token_storage_dtype'] == 'float32'


def test_bf16_policy_is_not_fp32_only_normalization():
    hidden, ids, mask, baseline = sample()
    cls = hidden[:, 0].float().numpy()
    fp32_only = cls / np.linalg.norm(cls, axis=1, keepdims=True)
    assert np.max(np.abs(fp32_only - baseline)) > 2e-6
    with pytest.raises(ValueError, match='differs from verified'):
        policy().verify_hidden(hidden, ids, mask, ids, fp32_only)


@pytest.mark.parametrize('damage', ['wrong_cls', 'zero', 'nonfinite', 'dtype', 'baseline_dtype', 'left_padding', 'mask_hole', 'token_drop'])
def test_no_override_fallback_or_invalid_state(damage):
    torch = pytest.importorskip('torch')
    hidden, ids, mask, baseline = sample()
    if damage == 'wrong_cls':
        hidden[:, 0] *= -1
    elif damage == 'zero':
        hidden[:, 0] = 0
    elif damage == 'nonfinite':
        hidden[0, 1, 0] = float('nan')
    elif damage == 'dtype':
        hidden = hidden.float()
    elif damage == 'baseline_dtype':
        baseline = baseline.astype(np.float64)
    elif damage == 'left_padding':
        hidden = torch.cat((hidden[:, :1], hidden), dim=1)
        ids, mask = [[1, 0, 10, 2]], [[0, 1, 1, 1]]
    elif damage == 'mask_hole':
        mask = [[1, 0, 1]]
    else:
        ids = [[0, 11, 2]]
    with pytest.raises(ValueError):
        policy().verify_hidden(hidden, ids, mask, [[0, 10, 2]], baseline)


def test_dense_override_parameter_not_supported():
    hidden, ids, mask, baseline = sample()
    with pytest.raises(TypeError):
        policy().verify_hidden(hidden, ids, mask, ids, baseline, native_dense=baseline)


@pytest.mark.parametrize('values', [[0., 0., 0.], [3e38, 3e38, 3e38], [1e-40, 1e-40, 1e-40]])
def test_corner_zero_overflow_tiny_values_fail_explicitly(values):
    torch = pytest.importorskip('torch')
    hidden = torch.zeros((1, 3, 1024), dtype=torch.bfloat16)
    hidden[0, 0, :3] = torch.tensor(values, dtype=torch.bfloat16)
    baseline = np.zeros((1, 1024), dtype=np.float32)
    baseline[0, 0] = 1
    with pytest.raises(ValueError):
        policy().verify_hidden(hidden, [[0, 10, 2]], [[1, 1, 1]], [[0, 10, 2]], baseline)


def test_contract_evidence_missing_or_changed(tmp_path, monkeypatch):
    monkeypatch.setattr(m, 'distribution', lambda name: SimpleNamespace(version='6.1.0', locate_file=lambda path: tmp_path / 'missing.py'))
    with pytest.raises(ValueError, match='unavailable'):
        policy()
    (tmp_path / 'missing.py').write_text('ambiguous normalization')
    with pytest.raises(ValueError, match='ambiguous'):
        policy()


def test_frozen_config_identity_and_source_guard(monkeypatch):
    monkeypatch.setattr(m, "normalization_evidence", lambda: {"synthetic_identity_only": True})
    p = policy()
    p.config['dtype'] = 'float32'
    assert p.config['dtype'] == 'bfloat16'
    assert policy('float32').identity != p.identity
    hidden, ids, mask, baseline = sample()
    monkeypatch.setattr(m, 'normalization_evidence', lambda: {'changed': True})
    with pytest.raises(ValueError, match='contract changed'):
        p.verify_hidden(hidden, ids, mask, ids, baseline)


def test_long_full_token_identity_and_right_padding():
    pytest.importorskip('torch')
    hidden, _, _, baseline = sample()
    hidden = hidden[:, :1].repeat(1, 1203, 1)
    ids = np.array([[0] + [10] * 1200 + [2, 1]], dtype=np.int64)
    mask = [[1] * 1202 + [0]]
    p = policy()
    _, first = p.verify_hidden(hidden, ids, mask, ids[0:1, :1202], baseline)
    ids[0, 600] = 11
    _, second = p.verify_hidden(hidden, ids, mask, ids[0:1, :1202], baseline)
    assert first['full_token_ids_sha256'] != second['full_token_ids_sha256']


def test_official_formula_and_policy_identity_metadata(monkeypatch):
    monkeypatch.setattr(m, "normalization_evidence", lambda: {"synthetic_identity_only": True})
    p = policy()
    assert m.FORMULA_COMMIT == 'fd1a2bdf69488ffebe0327999d4400d8c8058a0b'
    assert len(m.FORMULA_HASHES) == 3 and len(m.HEAD_HASHES) == 2
    assert len(p.identity) == 64
    assert m.POLICY == 'module_normalize_then_encode_normalize_then_fp32_renormalize'
    assert json.dumps(p.config)


@pytest.mark.parametrize('fixture_kind', ['seeded_full', 'independent_three_values'])
def test_module_and_encode_bf16_normalizations_are_not_idempotent(fixture_kind):
    torch = pytest.importorskip('torch')
    generator = torch.Generator(device='cpu').manual_seed(0)
    cls = torch.randn((1, 1024), generator=generator, dtype=torch.float32).to(torch.bfloat16)
    if fixture_kind == 'independent_three_values':
        cls.zero_()
        cls[0, :3] = torch.tensor([0.1650390625, -0.1015625, -1.484375], dtype=torch.bfloat16)
    hidden = cls[:, None, :].repeat(1, 3, 1)
    module = torch.nn.functional.normalize(cls, p=2, dim=-1)
    encode = torch.nn.functional.normalize(module, p=2, dim=1)
    single = module.float().numpy()
    single = single / np.linalg.norm(single, axis=1, keepdims=True)
    baseline = encode.float().numpy()
    baseline = baseline / np.linalg.norm(baseline, axis=1, keepdims=True)
    assert np.max(np.abs(single - baseline)) > 2e-6
    result, _ = policy().verify_hidden(hidden, [[0, 10, 2]], [[1, 1, 1]], [[0, 10, 2]], baseline)
    np.testing.assert_array_equal(result, baseline)
    with pytest.raises(ValueError, match='differs from verified'):
        policy().verify_hidden(hidden, [[0, 10, 2]], [[1, 1, 1]], [[0, 10, 2]], single)


def test_noninteger_context_limit_rejected():
    cfg = config()
    cfg['max_length'] = 8192.5
    with pytest.raises(ValueError, match='token limit'):
        m.NativeNormalizationPolicy(cfg, baseline_identity='synthetic')


def test_actual_cached_normalization_contract():
    from importlib.metadata import distribution, PackageNotFoundError
    try:
        package = distribution('sentence-transformers')
    except PackageNotFoundError:
        pytest.skip('Optional installed SentenceTransformer source unavailable')
    if not Path(package.locate_file('sentence_transformers/sentence_transformer/model.py')).is_file():
        pytest.skip('Optional installed SentenceTransformer source unavailable')
    # Read-only filesystem lookup; no Hub/snapshot-download call in this guard.
    snapshot = Path('.cache/models/models--BAAI--bge-m3/snapshots') / m.MODEL_DEFAULTS['bge']['revision']
    if not (snapshot / 'modules.json').is_file():
        pytest.skip('Optional pinned BGE config artifacts unavailable')
    evidence = m.normalization_evidence()
    assert evidence['native_torch_normalization_count'] == 2
    assert evidence['modules_sha256'] == m.MODULES_SHA256
    assert evidence['cls_pooling_sha256'] == m.POOLING_SHA256

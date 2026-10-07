"""Injected-only BGE dual-head extraction; no production model/CLI scheduling.

Native hidden precision and original G CLS arithmetic are independently bound
from FP32 extra-head projection/storage. Token memory is bounded by native batch
size; dense baseline arrays are supplied by the already verified G bridge.
Production performance gate: full baseline validation currently repeats per batch;
large runs require profiling without weakening its integrity checks.
"""

from __future__ import annotations
from contextlib import contextmanager
import copy
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import uuid

import numpy as np

from . import multimodal_bge_extra as core
from . import multimodal_bge_extra_native as native
from . import multimodal_bge_extra_baseline as bridge_module
from .models import stable_hash

_IMPLEMENTATION = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _private(path):
    if subprocess.run(
        ["git", "check-ignore", "--quiet", str(path.resolve())], capture_output=True
    ).returncode:
        raise ValueError("Dual-head raw cache must be git-ignored")
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path, 0o700)


def _write(path, data):
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _json(path, value):
    encoded = json.dumps(value, sort_keys=True, allow_nan=False).encode()
    _write(
        path,
        json.dumps(
            {"payload": value, "sha256": hashlib.sha256(encoded).hexdigest()}, sort_keys=True, allow_nan=False
        ).encode(),
    )


@contextmanager
def _reserved_batch(root, directory, request):
    """Serialize initial publication before a same-key waiter can acquire its lock."""
    role_path = root / "creation.lock"
    with role_path.open("a+b") as role_lock:
        os.chmod(role_path, 0o600)
        fcntl.flock(role_lock, fcntl.LOCK_EX)
        fresh = not directory.exists()
        if fresh:
            directory.mkdir(mode=0o700)
        os.chmod(directory, 0o700)
        lock_path = directory / "batch.lock"
        lock = lock_path.open("a+b")
        try:
            os.chmod(lock_path, 0o600)
            fcntl.flock(lock, fcntl.LOCK_EX)
            if fresh:
                _json(directory / "state.json", {"status": "pending", "request_sha256": stable_hash(request)})
            elif not (directory / "state.json").exists():
                raise ValueError("Interrupted preexisting batch; no inference fallback")
        except BaseException:
            lock.close()
            raise
    try:
        yield fresh
    finally:
        lock.close()


def _read_json(path):
    value = json.loads(path.read_bytes())
    if (
        set(value) != {"payload", "sha256"}
        or hashlib.sha256(json.dumps(value["payload"], sort_keys=True, allow_nan=False).encode()).hexdigest()
        != value["sha256"]
    ):
        raise ValueError("Dual-head cache manifest checksum differs; no inference fallback")
    return value["payload"]


def verified_trained_head_arrays(directory):
    """Future explicit preparation only; uses published SHA + weights_only loader.

    Never called by injected backend construction or CPU tests. No heads are
    loaded until a separately authorized caller explicitly invokes this helper.
    """
    heads = core.load_trained_heads(directory)
    return {
        "sparse": tuple(heads["sparse_linear.pt"][key].cpu().float().numpy() for key in ("weight", "bias")),
        "colbert": tuple(heads["colbert_linear.pt"][key].cpu().float().numpy() for key in ("weight", "bias")),
    }


class InjectedDualHeadBackend:
    def __init__(self, baseline, *, tokenizer, hidden_backend, head_arrays, batch_plan=None):
        if not callable(getattr(baseline, "verified_inputs", None)) or not callable(
            getattr(baseline, "validate_for_reuse", None)
        ):
            raise ValueError("Verified original G bridge required; no dense encoding fallback")
        baseline.validate_for_reuse()
        self._baseline = baseline
        self._config = copy.deepcopy(baseline.config)
        self._baseline_identity = baseline.identity
        self._tokenizer, self._backend = tokenizer, hidden_backend
        if (
            not isinstance(tokenizer.identity, str)
            or not tokenizer.identity
            or not isinstance(hidden_backend.identity, str)
            or not hidden_backend.identity
        ):
            raise ValueError("Explicit injected tokenizer and native hidden identities required")
        self._heads = {
            name: tuple(np.array(value, dtype=np.float32, copy=True) for value in head_arrays[name])
            for name in ("sparse", "colbert")
        }
        expected = {
            "sparse": ((1, core.DIMENSION), (1,)),
            "colbert": ((core.DIMENSION, core.DIMENSION), (core.DIMENSION,)),
        }
        for name, shapes in expected.items():
            if len(self._heads[name]) != 2 or any(
                value.shape != shape or not np.isfinite(value).all()
                for value, shape in zip(self._heads[name], shapes, strict=True)
            ):
                raise ValueError("Explicit finite native head array shapes required")
            for value in self._heads[name]:
                value.flags.writeable = False
        self._policy = native.NativeNormalizationPolicy(
            self._config, baseline_identity=self._baseline_identity
        )
        self._sources = {
            name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
            for name, module in {"core": core, "native": native, "bridge": bridge_module}.items()
        }
        self._identity = stable_hash(
            {
                "condition": "injected-bge-dual-head-v1",
                "baseline": self._baseline_identity,
                "config": self._config,
                "policy": self._policy.identity,
                "tokenizer": tokenizer.identity,
                "hidden_backend": hidden_backend.identity,
                "heads": {name: [_sha(value) for value in values] for name, values in self._heads.items()},
                "head_origin": "injected_arrays_not_claimed_trained",
                "official_head_pins": core.HEAD_HASHES,
                "formula_commit": core.FORMULA_COMMIT,
                "formula_hashes": core.FORMULA_HASHES,
                "native_hidden_dtype": self._config["dtype"],
                "head_and_token_dtype": "float32",
                "source_aggregation": "maximum_chunk_score",
                "sources": self._sources,
                "implementation": _IMPLEMENTATION,
            }
        )
        self._batch_plan = batch_plan
        self._batch_plan_identity = None
        if batch_plan is not None:
            from .multimodal_bge_extra_runtime import PinnedBatchPlan

            if (
                not isinstance(batch_plan, PinnedBatchPlan)
                or batch_plan._baseline_identity != self._baseline_identity
            ):
                raise ValueError("Pinned original baseline batch plan required")
            if getattr(hidden_backend, "can_flatten_inputs", None) is not False:
                raise ValueError("Pinned non-flattened loaded runtime declaration required")
            self._batch_plan_identity = batch_plan.identity
            self._identity = stable_hash({"backend": self._identity, "batch_plan": self._batch_plan_identity})
        self._seal = self._state()

    @property
    def identity(self):
        return self._identity

    def _state(self):
        return stable_hash(
            {
                "batch_plan": self._batch_plan.identity if self._batch_plan is not None else None,
                "baseline": self._baseline.identity,
                "config": self._baseline.config,
                "backend_config": self._config,
                "identity": self._identity,
                "policy_identity": self._policy.identity,
                "tokenizer": self._tokenizer.identity,
                "backend": self._backend.identity,
                "heads": {name: [_sha(value) for value in values] for name, values in self._heads.items()},
            }
        )

    def _verify(self):
        if (
            self._state() != self._seal
            or hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != _IMPLEMENTATION
        ):
            raise ValueError("Frozen extraction config/runtime/heads changed")
        for name, module in {"core": core, "native": native, "bridge": bridge_module}.items():
            if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != self._sources[name]:
                raise ValueError("Frozen extraction implementation changed")
        self._baseline.validate_for_reuse()
        self._policy._verify_contract()
        if self._batch_plan is not None:
            self._batch_plan.validate()
            if getattr(self._backend, "can_flatten_inputs", None) is not False:
                raise ValueError("Pinned non-flattened runtime changed")

    def _tokens(self, records):
        output = []
        for record in records:
            array = np.asarray(self._tokenizer.tokenize(record["formatted_text"]))
            if (
                array.ndim != 1
                or array.dtype.kind not in "iu"
                or not 2 <= len(array) <= 8192
                or np.any(array < 0)
                or np.any(array >= core.VOCABULARY)
                or array[0] != 0
                or array[-1] != 2
            ):
                raise ValueError("Full exact native token IDs required; no truncation")
            output.append([int(value) for value in array])
        return output

    def _read_batch(self, directory, request):
        path = directory / "completed.json"
        if not path.is_file():
            raise ValueError("Incomplete/failed dual-head cache; automatic inference retry forbidden")
        state = _read_json(directory / "state.json")
        if state.get("status") not in {"pending", "completed"} or state.get("request_sha256") != stable_hash(
            request
        ):
            raise ValueError("Failed/incompatible dual-head state; no inference fallback")
        manifest = _read_json(path)
        if (
            manifest.get("native_dtype") != self._config["dtype"]
            or manifest.get("head_dtype") != "float32"
            or manifest.get("token_storage") != "float32"
            or manifest.get("head_origin") != "injected_arrays_not_claimed_trained"
            or manifest.get("new_dense_encoder_calls") != 0
        ):
            raise ValueError("Dual-head cache precision/origin contract differs")
        if manifest["request"] != request or len(manifest["rows"]) != len(request["records"]):
            raise ValueError("Dual-head cache source/query/role coverage differs")
        output = []
        for index, row in enumerate(manifest["rows"]):
            if row["file"] != f"{index:04d}.npy" or row["record"] != request["records"][index]:
                raise ValueError("Dual-head cache ordered source/chunk identity differs")
            data = (directory / row["file"]).read_bytes()
            if hashlib.sha256(data).hexdigest() != row["sha256"]:
                raise ValueError("Dual-head token cache checksum differs; no inference fallback")
            vectors = np.load(io.BytesIO(data), allow_pickle=False)
            if (
                vectors.dtype != np.float32
                or vectors.shape != (len(request["tokens"][index]) - 1, core.DIMENSION)
                or not np.isfinite(vectors).all()
            ):
                raise ValueError("Dual-head token dtype/full coverage differs")
            norms = np.linalg.norm(vectors, axis=1)
            if not np.all(np.isfinite(norms) & (norms <= 1 + 2e-6)):
                raise ValueError("Dual-head token normalization differs")
            sparse = {int(key): value for key, value in row["sparse"].items()}
            core.sparse_score(sparse, {})
            if any(token not in request["tokens"][index] or token in {0, 1, 2, 3} for token in sparse):
                raise ValueError("Dual-head sparse token/source coverage differs")
            output.append({"record": copy.deepcopy(row["record"]), "sparse": sparse, "colbert": vectors})
        return output

    def extract(self, role, cache_dir, *, max_batches=None):
        """Stream bounded native batches to private cache; return aggregates only."""
        if (
            role not in {"document", "query"}
            or max_batches is not None
            and (type(max_batches) is not int or max_batches < 1)
        ):
            raise ValueError("Explicit role and positive optional batch limit required")
        self._verify()
        records, vectors = self._baseline.verified_inputs(role)
        if (
            not records
            or vectors.dtype != np.float32
            or vectors.shape != (len(records), core.DIMENSION)
            or len({row["id"] for row in records}) != len(records)
        ):
            raise ValueError("Original G ordered role/source coverage differs")
        records = copy.deepcopy(records)
        record_seal = stable_hash(records)
        batch_size = self._config["batch_size"]
        cache_base = Path(cache_dir)
        _private(cache_base)
        _private(cache_base / self.identity)
        root = cache_base / self.identity / role
        _private(root)
        original_indices = list(range(len(records)))
        if self._batch_plan is not None:
            original_indices = self._batch_plan.order_for(role, records)
            records = [records[index] for index in original_indices]
            record_seal = stable_hash(records)
        row_kind = (
            "queries"
            if role == "query"
            else (
                "document_chunks"
                if any(row["id"] != row["source_id"] for row in records)
                else "document_whole_sources"
            )
        )
        fresh = cached = batches = 0
        for start in range(0, len(records), batch_size):
            if max_batches is not None and batches >= max_batches:
                break
            batch = records[start : start + batch_size]
            baseline = np.array(
                vectors[original_indices[start : start + batch_size]], dtype=np.float32, copy=True
            )
            tokens = self._tokens(batch)
            self._verify()
            request = {
                "backend": self.identity,
                "role": role,
                "row_kind": row_kind,
                "start": start,
                "records": batch,
                "tokens": tokens,
                "baseline_dense_sha256": _sha(baseline),
                "native_policy": self._policy.identity,
            }
            if self._batch_plan is not None:
                request["batch_plan"] = self._batch_plan_identity
                request["original_row_indices"] = original_indices[start : start + batch_size]
            directory = root / f"{start:09d}"
            with _reserved_batch(root, directory, request) as fresh_batch:
                self._verify()
                if not fresh_batch:
                    self._read_batch(directory, request)
                    self._verify()
                    cached += len(batch)
                else:
                    _json(
                        directory / "state.json",
                        {"status": "pending", "request_sha256": stable_hash(request)},
                    )
                    try:
                        request_seal = stable_hash(request)
                        result = self._backend.forward(tokens)
                        self._verify()
                        if (
                            stable_hash(request) != request_seal
                            or stable_hash(records) != record_seal
                            or _sha(baseline) != request["baseline_dense_sha256"]
                        ):
                            raise ValueError("Native callback mutated frozen full inputs")
                        if set(result) != {"hidden", "token_ids", "attention_mask"}:
                            raise ValueError(
                                "Native callback must return same hidden/IDs/mask only; dense override forbidden"
                            )
                        self._policy.verify_hidden(
                            result["hidden"], result["token_ids"], result["attention_mask"], tokens, baseline
                        )
                        rows = []
                        sw, sb = self._heads["sparse"]
                        cw, cb = self._heads["colbert"]
                        for index, record in enumerate(batch):
                            # CLS proof succeeded before either head is published.
                            states = (
                                result["hidden"][index, : len(tokens[index])].detach().cpu().float().numpy()
                            )
                            sparse = core.lexical_weights(states, tokens[index], sw, sb)
                            token_vectors = core.projected_tokens(states, cw, cb)
                            buffer = io.BytesIO()
                            np.save(buffer, token_vectors, allow_pickle=False)
                            data = buffer.getvalue()
                            path = directory / f"{index:04d}.npy"
                            _write(path, data)
                            rows.append(
                                {
                                    "record": record,
                                    "file": path.name,
                                    "sha256": hashlib.sha256(data).hexdigest(),
                                    "sparse": sparse,
                                }
                            )
                        self._verify()
                        if stable_hash(request) != request_seal:
                            raise ValueError("Extraction inputs changed before cache publication")
                        _json(
                            directory / "completed.json",
                            {
                                "request": request,
                                "rows": rows,
                                "native_dtype": self._config["dtype"],
                                "head_dtype": "float32",
                                "token_storage": "float32",
                                "head_origin": "injected_arrays_not_claimed_trained",
                                "new_dense_encoder_calls": 0,
                            },
                        )
                        _json(
                            directory / "state.json", {"status": "completed", "request_sha256": request_seal}
                        )
                        del result, states, token_vectors, buffer, data
                        fresh += len(batch)
                    except Exception:
                        _json(
                            directory / "state.json",
                            {"status": "failed", "request_sha256": stable_hash(request)},
                        )
                        raise
                batches += 1
        self._verify()
        return {
            "status": "completed" if fresh + cached == len(records) else "preparing",
            "role": role,
            "row_kind": row_kind,
            "source_or_query_count": len({row["source_id"] for row in records}),
            "row_count": len(records),
            "processed_rows": fresh + cached,
            "new_rows": fresh,
            "cached_rows": cached,
            "hidden_forward_calls": (fresh + batch_size - 1) // batch_size,
            "new_dense_encoder_calls": 0,
            "native_hidden_dtype": self._config["dtype"],
            "head_and_storage_dtype": "float32",
            "actual_model_compatibility_measured": False,
            "production_backend_available": False,
            "batch_plan_identity": self._batch_plan_identity,
        }

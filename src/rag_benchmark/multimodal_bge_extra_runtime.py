"""Pinned ST batch planning only; production loading/forward remains unavailable.

Original encode calls sorted per cache invocation, not per entire collection.
No model/tensor/head loading is implemented by this module.
"""

from __future__ import annotations

import copy
import hashlib
from importlib.metadata import distribution
from pathlib import Path

import numpy as np

from . import multimodal_bge_extra_native as native
from .models import stable_hash

PROTOCOL = "st61_original_cache_block_length_sort_batch_v1"
SOURCE_HASHES = {
    "sentence_transformers/base/model.py": "60b8aaeac7e02bb67071b026461dedbf2ceef9820de21d0a5c60db621107fbef",
    "sentence_transformers/base/modules/transformer.py": "895603634b3b63325cb7e5eaa2e8e2cdbc708b3b0e31a5c8b277ecd711a24be4",
}
_IMPLEMENTATION = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def scheduling_evidence():
    """Inspect installed pinned sources only; never instantiate a model."""
    evidence = native.normalization_evidence()
    try:
        package = distribution("sentence-transformers")
        hashes = {
            name: hashlib.sha256(Path(package.locate_file(name)).read_bytes()).hexdigest()
            for name in SOURCE_HASHES
        }
    except (OSError, LookupError) as error:
        raise ValueError("Pinned scheduling source unavailable") from error
    if hashes != SOURCE_HASHES or package.version != "6.1.0":
        raise ValueError("Pinned scheduling source changed")
    return {
        "normalization": evidence,
        "sources": hashes,
        "numpy_version": np.__version__,
        "sorting": "np.argsort([-len(formatted_string)]) default kind; no stable tie substitution",
        "flattened_inputs": "not_observed_requires_loaded_runtime_false_guard",
    }


class PinnedBatchPlan:
    """Original row permutation, conditioned on non-flattened pinned runtime.

    The production factory must verify can_flatten_inputs=False before using this
    plan. Until then this is a CPU-prepared condition, not measured compatibility.
    """

    def __init__(self, baseline):
        baseline.validate_for_reuse()
        self._baseline = baseline
        self._baseline_identity = baseline.identity
        self._config = baseline.config
        if self._config.get("batch_size") != 8:
            raise ValueError("Original batch8 contract required")
        self._evidence = scheduling_evidence()
        self._roles = {}
        for role in ("document", "query"):
            records, _ = baseline.verified_inputs(role)
            chunks = role == "document" and any(r["id"] != r["source_id"] for r in records)
            self._roles[role] = {
                "records_hash": stable_hash(records),
                "count": len(records),
                "chunked": chunks,
            }
        # JS queries use the same 32-row invocation boundary as its documents.
        block_size = 32 if self._roles["document"]["chunked"] else 128
        for role in ("document", "query"):
            records, _ = baseline.verified_inputs(role)
            order = []
            for start in range(0, len(records), block_size):
                texts = [r["formatted_text"] for r in records[start : start + block_size]]
                if any(not isinstance(t, str) for t in texts):
                    raise ValueError("Pinned string-only input length contract required")
                order.extend(start + int(i) for i in np.argsort([-len(t) for t in texts]))
            self._roles[role].update({"original_indices": order, "invocation_rows": block_size})
        self._identity = stable_hash(
            {
                "protocol": PROTOCOL,
                "baseline": self._baseline_identity,
                "config": self._config,
                "evidence": self._evidence,
                "roles": self._roles,
                "implementation": _IMPLEMENTATION,
                "production_runtime_available": False,
            }
        )
        self._seal = self._state()

    def _state(self):
        return stable_hash(
            {
                "baseline": self._baseline.identity,
                "config": self._baseline.config,
                "owned_config": self._config,
                "evidence": self._evidence,
                "roles": self._roles,
                "identity": self._identity,
            }
        )

    @property
    def identity(self):
        return self._identity

    def validate(self):
        if (
            self._state() != self._seal
            or scheduling_evidence() != self._evidence
            or hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != _IMPLEMENTATION
        ):
            raise ValueError("Pinned batch schedule changed")
        self._baseline.validate_for_reuse()

    def order_for(self, role, records):
        self.validate()
        if role not in self._roles or stable_hash(records) != self._roles[role]["records_hash"]:
            raise ValueError("Pinned batch schedule row identity differs")
        return list(self._roles[role]["original_indices"])

    def summary(self):
        self.validate()
        return copy.deepcopy(
            {
                "identity": self.identity,
                "protocol": PROTOCOL,
                "batch_size": 8,
                "roles": {
                    role: {k: v for k, v in data.items() if k != "original_indices"}
                    for role, data in self._roles.items()
                },
                "evidence": self._evidence,
                "production_runtime_available": False,
                "required_loaded_runtime_guard": "can_flatten_inputs is False",
                "actual_native_compatibility_measured": False,
            }
        )


def production_factory(*args, **kwargs):
    """No unverified loading callback may bypass the unavailable runtime gate."""
    raise ValueError("Production BGE extraction unavailable: loaded runtime/tokenizer/weight proof pending")

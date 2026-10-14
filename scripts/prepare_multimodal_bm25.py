"""Frozen multimodal BM25 INDEX preparation only: never query, rank or evaluate.

Six native bm25s files, ordered document IDs/lengths and immutable provenance
are committed as one private directory. Raw-media collections without an
approved source-only gallery text representation remain unavailable.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import uuid

import numpy as np

from rag_benchmark.models import stable_hash
from rag_benchmark.multimodal import _code_tokens, load_dataset
from rag_benchmark.retrieval import turkish_tokens

NATIVE_FILES = {"ready.json", "params.index.json", "vocab.index.json", "data.csc.index.npy",
                "indices.csc.index.npy", "indptr.csc.index.npy"}
PROTOCOL = "multimodal-native-bm25-index-only-v1"
SETTINGS = {"k1": 1.5, "b": 0.75, "method": "lucene", "csc_backend": "numpy"}


def sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def safe(path: Path | str) -> Path:
    path = Path(path).absolute()
    if ".." in path.parts or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("unsafe_private_path")
    return path


def json_write(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    path.chmod(0o600)


def file_record(path: Path) -> dict:
    if not safe(path).is_file():
        raise ValueError("regular_file_required")
    return {"filename": path.name, "size": path.stat().st_size, "sha256": sha(path)}


def implementation() -> dict:
    import bm25s
    import bm25s.scoring
    from rag_benchmark import multimodal, retrieval
    paths = [Path(__file__), Path(multimodal.__file__), Path(retrieval.__file__),
             Path(bm25s.__file__), Path(bm25s.scoring.__file__)]
    return {"files": {str(p.absolute()): sha(p) for p in paths},
            "packages": {name: importlib.metadata.version(name) for name in ("bm25s", "numpy")}}


def source_seals(dataset) -> dict:
    paths = {"dataset.json": dataset.root / "dataset.json"}
    paths.update({name: dataset.root / name for name in dataset.manifest["files"]})
    return {name: {"sha256": sha(path), "size": path.stat().st_size} for name, path in paths.items()}


def verify_native(directory: Path, *, expected_ids: list[str], expected_lengths: list[int],
                  expected_corpus_hash: str) -> dict:
    directory = safe(directory)
    for name in NATIVE_FILES:
        if not safe(directory / name).is_file():
            raise ValueError("native_index_incomplete")
    ready = json.loads((directory / "ready.json").read_text())
    if ready != {"corpus_hash": expected_corpus_hash, "count": len(expected_ids)}:
        raise ValueError("native_ready_identity")
    params = json.loads((directory / "params.index.json").read_text())
    if (params.get("num_docs") != len(expected_ids) or params.get("k1") != 1.5 or
            params.get("b") != .75 or params.get("method") != "lucene"):
        raise ValueError("native_index_settings")
    vocab = json.loads((directory / "vocab.index.json").read_text())
    data, indices, indptr = [np.load(directory / (name + ".csc.index.npy"), mmap_mode="r", allow_pickle=False)
                             for name in ("data", "indices", "indptr")]
    ints = {np.dtype("int32"), np.dtype("int64")}
    if (data.ndim != 1 or indices.ndim != 1 or indptr.ndim != 1 or data.dtype != np.dtype("float32") or
            indices.dtype not in ints or indptr.dtype not in ints or len(indptr) < 2 or
            len(data) != len(indices) or indptr[0] != 0 or indptr[-1] != len(data) or
            not np.all((indptr >= 0) & (indptr <= len(data))) or not np.all(np.diff(indptr) >= 0) or
            not np.isfinite(data).all() or not np.all((indices >= 0) & (indices < len(expected_ids)))):
        raise ValueError("native_csc_invalid")
    columns = len(indptr) - 1
    if not isinstance(vocab, dict) or not all(isinstance(k, str) and type(v) is int for k, v in vocab.items()):
        raise ValueError("native_vocab_invalid")
    ordinary = len(vocab) == columns and set(vocab.values()) == set(range(columns))
    physical = {k: v for k, v in vocab.items() if v < columns}
    sentinel = (len(vocab) == columns + 1 and vocab.get("") == columns and len(physical) == columns
                and set(physical.values()) == set(range(columns)))
    if not (ordinary or sentinel):
        raise ValueError("native_empty_token_sentinel_invalid")
    order = json.loads((directory / "documents.json").read_text())
    if order != {"item_ids": expected_ids, "document_lengths": expected_lengths,
                 "order_sha256": stable_hash(expected_ids)}:
        raise ValueError("native_document_order_or_lengths")
    # Save/load exercised without any query, search, ranking or score function.
    import bm25s
    loaded = bm25s.BM25.load(str(directory), load_corpus=False, mmap=True)
    if loaded.scores["num_docs"] != len(expected_ids) or loaded.vocab_dict != vocab:
        raise ValueError("native_loadback_identity")
    for key, array in (("data", data), ("indices", indices), ("indptr", indptr)):
        if not np.array_equal(loaded.scores[key], array):
            raise ValueError("native_loadback_bytes")
    return {"physical_vocabulary_columns": columns, "empty_token_sentinel": sentinel,
            "nonzero_scores": len(data), "pointer_dtype": str(indptr.dtype),
            "indices_dtype": str(indices.dtype), "loadback_verified": True}


@contextmanager
def destination_lock(output: Path):
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock = safe(output.parent / ("." + output.name + ".lock"))
    descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def prepare(dataset_root: Path, output: Path, expected_identity: str) -> dict:
    output = safe(output)
    dataset = load_dataset(dataset_root)
    if dataset.identity != expected_identity:
        raise ValueError("frozen_dataset_identity")
    # Exact source-only prerequisites used by the production BM25 channel.
    dataset.require_text()
    source = source_seals(dataset)
    impl = implementation()
    tokenizer = _code_tokens if dataset.track == "code" else turkish_tokens
    tokens = [tokenizer(row["text"]) for row in dataset.corpus]
    if not any(tokens):
        raise ValueError("no_lexical_terms")
    ids = [row["id"] for row in dataset.corpus]
    lengths = [len(row) for row in tokens]
    corpus_hash = stable_hash([{key: row[key] for key in ("id", "text")} for row in dataset.corpus])
    pins = {"protocol": PROTOCOL, "dataset_identity": dataset.identity, "dataset_revision": dataset.manifest["revision"],
            "settings": SETTINGS, "tokenizer": "code-identifiers-v1" if dataset.track == "code" else "turkish-unicode-v1",
            "tokenizer_input": "corpus.text only; no title prefix; exact multimodal BM25 channel",
            "item_order_sha256": stable_hash(ids), "document_lengths_sha256": stable_hash(lengths),
            "source_files": source, "implementation": impl}
    identity = stable_hash(pins)
    with destination_lock(output):
        if output.exists():
            manifest = json.loads((output / "prepared.json").read_text())
            if manifest.get("status") != "complete" or manifest.get("identity") != identity or manifest.get("pins") != pins:
                raise ValueError("existing_destination_collision_or_interrupted")
            allowed = NATIVE_FILES | {"documents.json", "prepared.json"}
            if {p.name for p in output.iterdir()} != allowed:
                raise ValueError("unexpected_existing_files")
            for record in manifest["files"]:
                if file_record(output / record["filename"]) != record:
                    raise ValueError("existing_complete_payload_changed")
            if len(manifest["files"]) != 7 or {r['filename'] for r in manifest["files"]} != NATIVE_FILES | {"documents.json"}:
                raise ValueError("existing_complete_inventory")
            verify_native(output, expected_ids=ids, expected_lengths=lengths, expected_corpus_hash=corpus_hash)
            return {"status": "verified_reuse", "identity": identity, "documents": len(ids), "new_index_builds": 0,
                    "benchmark_runs": 0, "retrieval_calls": 0}
        remnants = list(output.parent.glob("." + output.name + ".staging-*"))
        if remnants:
            raise ValueError("interrupted_staging_retained_requires_review")
        staging = output.parent / ("." + output.name + ".staging-" + uuid.uuid4().hex)
        staging.mkdir(mode=0o700)
        try:
            import bm25s
            model = bm25s.BM25(**SETTINGS)
            model.index(tokens, show_progress=False)
            model.save(str(staging))
            json_write(staging / "ready.json", {"corpus_hash": corpus_hash, "count": len(ids)})
            json_write(staging / "documents.json", {"item_ids": ids, "document_lengths": lengths,
                                                   "order_sha256": stable_hash(ids)})
            for path in staging.iterdir():
                if not path.is_file() or path.name not in NATIVE_FILES | {"documents.json"}:
                    raise ValueError("unexpected_native_save_payload")
                path.chmod(0o600)
                with path.open("rb") as handle:
                    os.fsync(handle.fileno())
            validation = verify_native(staging, expected_ids=ids, expected_lengths=lengths,
                                       expected_corpus_hash=corpus_hash)
            if source_seals(dataset) != source or implementation() != impl or load_dataset(dataset_root).identity != expected_identity:
                raise ValueError("source_changed_during_preparation")
            manifest = {"schema_version": 1, "status": "complete", "identity": identity, "pins": pins,
                        "documents": len(ids), "files": [file_record(staging / n) for n in sorted(NATIVE_FILES | {"documents.json"})],
                        "validation": validation, "benchmark_runs": 0, "retrieval_calls": 0,
                        "reranker_calls": 0, "answer_generation_calls": 0,
                        "native_bm25_stored_not_sql_search": True}
            json_write(staging / "prepared.json", manifest)
            if output.exists():
                raise ValueError("destination_appeared_no_overwrite")
            descriptor = os.open(staging, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            os.rename(staging, output)
            descriptor = os.open(output.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except BaseException:
            # Keep failed staging for explicit review, never automatically reset it.
            raise
    return {"status": "index_prepared", "identity": identity, "documents": len(ids), "new_index_builds": 1,
            "benchmark_runs": 0, "retrieval_calls": 0}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-identity", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        if not re.fullmatch(r"[0-9a-f]{64}", args.dataset_identity):
            raise ValueError("expected_identity_required")
        repo = Path(__file__).absolute().parents[1]
        relative = safe(args.output).relative_to(repo)
        if relative.parts[0] not in {"work", ".cache"}:
            raise ValueError("private_output_required")
        os.umask(0o077)
        if args.execute:
            result = prepare(args.dataset, args.output, args.dataset_identity)
        else:
            dataset = load_dataset(args.dataset)
            if dataset.identity != args.dataset_identity:
                raise ValueError("frozen_dataset_identity")
            dataset.require_text()
            result = {"status": "ready_not_executed", "documents": len(dataset.corpus),
                      "benchmark_runs": 0, "retrieval_calls": 0}
        print(json.dumps(result, sort_keys=True))
        return 0
    except Exception as error:
        print(json.dumps({"status": "index_preparation_failed", "error_type": type(error).__name__,
                          "benchmark_runs": 0, "retrieval_calls": 0}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

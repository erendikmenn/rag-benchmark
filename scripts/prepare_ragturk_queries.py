"""Prepare all RAGTurk query vectors only; never search, rerank or generate answers."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from rag_benchmark.models import DenseEmbedder, package_versions, stable_hash, synchronize
from rag_benchmark.runner import read_config


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)


@contextmanager
def exclusive_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def read_queries(data_dir: Path) -> tuple[list[dict], dict]:
    manifest = json.loads((data_dir / "manifest.json").read_text())
    rows = []
    hashes = {"manifest.json": sha256(data_dir / "manifest.json")}
    for split, count in (("dev", 2000), ("test", 12530)):
        name = f"questions.{split}.jsonl"
        path = data_dir / name
        hashes[name] = sha256(path)
        expected = manifest["file_sha256"][name]
        if isinstance(expected, dict):
            expected = expected["sha256"]
        if hashes[name] != expected:
            raise ValueError(f"Frozen {split} source hash mismatch")
        part = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if len(part) != count:
            raise ValueError(f"Unexpected {split} question count")
        for row in part:
            if not isinstance(row.get("id"), str) or not isinstance(row.get("question"), str) or not row["question"].strip():
                raise ValueError("Missing query ID or question text")
        rows.extend(part)
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate query ID across frozen splits")
    return rows, hashes


def validate_vectors(values: np.ndarray, rows: int, dimension: int) -> None:
    if values.dtype != np.dtype("float32") or values.shape != (rows, dimension):
        raise ValueError("Query vector shape/dtype mismatch")
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite query vectors")
    if not np.allclose(np.linalg.norm(values, axis=1), 1, atol=2e-6, rtol=0):
        raise ValueError("Query vectors are not normalized")


def encode_blocks(adapter, queries: list[dict], output: Path, identity: str, *, block_size: int = 128,
                  sync=lambda: None, final_verify=lambda: None) -> dict:
    """Durable, identity-bound blocks; labels are never passed to the encoder."""
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    started = time.perf_counter()
    fresh_rows = reused_rows = 0
    blocks = []
    for start in range(0, len(queries), block_size):
        part = queries[start:start + block_size]
        path = output / f"queries-{start:06d}.npy"
        metadata_path = path.with_suffix(".json")
        order_hash = stable_hash([row["id"] for row in part])
        expected = {"identity": identity, "start": start, "rows": len(part), "query_order_sha256": order_hash}
        if path.exists() != metadata_path.exists():
            raise ValueError("Incomplete query checkpoint; inspect before resuming")
        if path.exists():
            metadata = json.loads(metadata_path.read_text())
            if any(metadata.get(key) != value for key, value in expected.items()) or metadata.get("sha256") != sha256(path):
                raise ValueError("Query checkpoint identity/checksum mismatch")
            values = np.load(path, allow_pickle=False)
            validate_vectors(values, len(part), adapter.dimension)
            reused_rows += len(part)
        else:
            # Only the question is encoded. Reference answers and gold IDs are excluded.
            texts = [adapter.format_query(row["question"]) for row in part]
            tick = time.perf_counter()
            values = adapter._encode(texts)
            sync()
            validate_vectors(values, len(part), adapter.dimension)
            temp = path.with_suffix(".npy.tmp")
            with temp.open("wb") as stream:
                np.save(stream, values, allow_pickle=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(temp, 0o600)
            os.replace(temp, path)
            metadata = {**expected, "sha256": sha256(path), "encoding_seconds": time.perf_counter() - tick}
            atomic_json(metadata_path, metadata)
            fresh_rows += len(part)
        blocks.append({**metadata, "file": path.name})
        status = {"status": "encoding_queries", "identity": identity, "model": adapter.name,
                  "completed_rows": start + len(part), "total_rows": len(queries),
                  "fresh_rows": fresh_rows, "reused_rows": reused_rows,
                  "elapsed_seconds": time.perf_counter() - started}
        atomic_json(output / "status.json", status)
        print(json.dumps(status), flush=True)
    full = np.concatenate([np.load(output / b["file"], allow_pickle=False) for b in blocks])
    validate_vectors(full, len(queries), adapter.dimension)
    destination = output / "queries.npy"
    if destination.exists():
        old = np.load(destination, allow_pickle=False)
        if not np.array_equal(old, full):
            raise ValueError("Existing full query matrix differs from verified blocks")
    else:
        with destination.with_suffix(".npy.tmp").open("wb") as stream:
            np.save(stream, full, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(destination.with_suffix(".npy.tmp"), 0o600)
        os.replace(destination.with_suffix(".npy.tmp"), destination)
    final_verify()
    result = {**status, "status": "complete", "sha256": sha256(destination), "bytes": destination.stat().st_size,
              "shape": list(full.shape), "dtype": "float32", "blocks": blocks,
              "benchmark_runs": 0, "reranker_calls": 0, "answer_generation_calls": 0}
    atomic_json(output / "result.json", result)
    atomic_json(output / "status.json", {k: v for k, v in result.items() if k != "blocks"})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/ragturk.toml"))
    parser.add_argument("--model", choices=("bge", "embeddinggemma"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-lock", type=Path, default=Path("work/overnight/gpu.lock"))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    os.umask(0o077)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    config = read_config(args.config)
    data_dir = Path(config["dataset"]["path"])
    queries, source_hashes = read_queries(data_dir)
    adapter = DenseEmbedder(args.model, config["embeddings"][args.model])
    model_source = Path(__import__("rag_benchmark.models", fromlist=["__file__"]).__file__)
    pins = {"protocol": "ragturk-all-query-embedding-storage-v1", "model": args.model,
            "config": adapter.config, "adapter_identity": adapter.identity,
            "source_files": source_hashes, "query_order_sha256": stable_hash([row["id"] for row in queries]),
            "block_size": 128, "query_order": "dev-file-order-then-test-file-order",
            "packages": package_versions(("numpy", "sentence-transformers", "transformers", "torch")),
            "producer_sha256": sha256(Path(__file__)), "adapter_source_sha256": sha256(model_source),
            "config_sha256": sha256(args.config), "historical_single_query_bit_identity_claimed": False}
    identity = stable_hash(pins)
    if not args.execute:
        print(json.dumps({"status": "ready_not_executed", "rows": len(queries), "dimension": adapter.dimension,
                          "identity": identity, "model": args.model}))
        return
    with exclusive_lock(args.gpu_lock), exclusive_lock(args.output / "owner.lock"):
        pins_path = args.output / "pins.json"
        if pins_path.exists() and json.loads(pins_path.read_text()) != pins:
            raise ValueError("Existing query preparation pins differ")
        atomic_json(pins_path, pins)
        def final_verify():
            _, final_source_hashes = read_queries(data_dir)
            if (final_source_hashes != source_hashes or sha256(model_source) != pins["adapter_source_sha256"]
                    or sha256(args.config) != pins["config_sha256"] or sha256(Path(__file__)) != pins["producer_sha256"]):
                raise ValueError("Sources changed during query preparation")
        try:
            encode_blocks(adapter, queries, args.output, identity,
                          sync=lambda: synchronize(adapter.config["device"]), final_verify=final_verify)
        except BaseException as exc:
            atomic_json(args.output / "status.json", {"status": "failed", "identity": identity,
                                                     "error_type": type(exc).__name__})
            raise


if __name__ == "__main__":
    main()

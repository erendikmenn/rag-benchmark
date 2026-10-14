"""Pinned document ColQwen token preparation only; no search or benchmark."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import time

import numpy as np


def require(ok, code):
    if not ok:
        raise ValueError(code)


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def seal(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic(path, value):
    temp = path.with_suffix(".tmp")
    with temp.open("x") as handle:
        json.dump(value, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def inside(root, value, *, allow_links=False):
    path = Path(value)
    path = path if path.is_absolute() else root / path
    require(
        path.resolve().is_relative_to(root)
        and (allow_links or not any(x.is_symlink() for x in (path, *path.parents))),
        "unsafe_path",
    )
    return path.resolve()


def pinned_files(root, records, *, allow_links=False):
    require(bool(records), "required_file_inventory")
    result = {}
    for entry in records:
        path = inside(root, entry["path"], allow_links=allow_links)
        require(
            path not in result
            and path.is_file()
            and path.stat().st_size == entry["size"]
            and sha(path) == entry["sha256"],
            "pinned_file_mismatch",
        )
        result[path] = entry["sha256"]
    return result


def production(spec):
    from rag_benchmark.multimodal import load_dataset
    from rag_benchmark.multimodal_colqwen import ColQwenAdapter, pinned_snapshots

    dataset = load_dataset(spec["dataset_root"])
    adapter = ColQwenAdapter(spec["adapter_config"])
    snapshots = pinned_snapshots(adapter.config, download=False)
    pinned_paths = {
        inside(Path(spec["artifact_root"]).resolve(), x["path"], allow_links=True)
        for x in spec["model_files"]
    }
    require(
        all(
            p.resolve() in pinned_paths for snapshot in snapshots for p in snapshot.rglob("*") if p.is_file()
        ),
        "model_inventory_incomplete",
    )
    required = [
        Path(__file__).resolve(),
        *[
            Path(__file__).resolve().parents[1] / "src/rag_benchmark" / name
            for name in ("multimodal.py", "multimodal_colqwen.py", "multimodal_models.py", "models.py")
        ],
    ]
    code_paths = {inside(Path(spec["artifact_root"]).resolve(), x["path"]) for x in spec["code_files"]}
    require(all(p in code_paths for p in required), "code_inventory_incomplete")
    return dataset, adapter


def tokens(path):
    a = np.load(path, mmap_mode="r", allow_pickle=False)
    require(a.dtype == np.float32 and a.ndim == 2 and a.shape[1] == 128 and len(a) > 0, "native_token_shape")
    require(
        np.isfinite(a).all() and np.allclose(np.linalg.norm(a, axis=1), 1, atol=2e-6), "native_token_norm"
    )
    return list(a.shape)


def prepare(spec, state_dir, *, execute=False, resume=False, factory=production, clock=time.monotonic):
    require(spec.get("version") == 1, "spec_version")
    root = Path(spec["artifact_root"])
    require(root.is_absolute() and root.is_dir(), "artifact_root")
    root = inside(root.resolve(), str(root))
    dataset_path = inside(root, spec["dataset_root"])
    cache = inside(root, spec["cache_dir"])
    state_dir = inside(root, str(state_dir))
    code = pinned_files(root, spec["code_files"])
    models = pinned_files(root, spec["model_files"], allow_links=True)
    dataset, adapter = factory({**spec, "dataset_root": str(dataset_path)})
    require(
        dataset.track == "document" and dataset.identity == spec["frozen_identity"],
        "frozen_document_identity",
    )
    require(adapter.identity == spec["expected_adapter_identity"], "adapter_identity_mismatch")
    initial_identity = dataset.identity
    config = adapter.config
    require(
        config.get("local_files_only") is True
        and config.get("batch_size") == 1
        and config.get("dtype") == "bfloat16",
        "pinned_native_condition",
    )
    binding = seal(spec)
    state_path = state_dir / "state.json"
    prior = json.loads(state_path.read_text()) if state_path.exists() else None
    if prior:
        require(
            resume and prior["binding"] == binding and prior["status"] in {"interrupted", "completed"},
            "explicit_resume_required",
        )
    else:
        require(
            not resume and (not state_dir.exists() or not any(state_dir.iterdir())), "fresh_state_required"
        )
    existing = dict(spec.get("existing_tokens", {}))
    if prior:
        for row in prior["items"]:
            require(
                row["path"] not in existing or existing[row["path"]] == row["sha256"],
                "sealed_token_collision",
            )
            existing[row["path"]] = row["sha256"]
    for relative, expected in existing.items():
        path = inside(root, relative)
        require(path.is_file() and sha(path) == expected, "sealed_token_missing_or_changed")
        tokens(path)
    pending = []
    inventory = []
    for role, items in [("document", dataset.corpus), ("query", dataset.queries)]:
        for item in items:
            model_item = dataset.model_item(item)
            path = cache / "tokens" / role / (adapter._cache_key(model_item, role) + ".npy")
            relative = str(path.relative_to(root))
            if path.exists():
                require(relative in existing and sha(path) == existing[relative], "unsealed_existing_token")
                shape = tokens(path)
                inventory.append(
                    {
                        "role": role,
                        "id": item["id"],
                        "path": relative,
                        "sha256": existing[relative],
                        "shape": shape,
                        "size": path.stat().st_size,
                    }
                )
            else:
                pending.append((role, item, model_item, path))
    require(not prior or prior["status"] != "completed" or not pending, "completed_token_missing")
    if not execute:
        return {
            "status": "dry_run",
            "sealed_existing_items": len(inventory),
            "missing_items": len(pending),
            "model_loaded": False,
        }
    state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    state_dir.chmod(0o700)
    state = {
        "binding": binding,
        "status": "running",
        "items": inventory,
        "missing_initial": len(pending),
        "adapter_identity": adapter.identity,
    }
    atomic(state_path, state)
    started = clock()
    try:
        for role, item, model_item, path in pending:
            require(not path.exists(), "unexpected_token_appeared")
            adapter._cached_tokens([model_item], role, cache)
            shape = tokens(path)
            require(not path.is_symlink(), "unsafe_token_path")
            row = {
                "role": role,
                "id": item["id"],
                "path": str(path.relative_to(root)),
                "sha256": sha(path),
                "shape": shape,
                "size": path.stat().st_size,
            }
            state["items"].append(row)
            state["elapsed_seconds"] = clock() - started
            atomic(state_path, state)
        for path, expected in {**code, **models}.items():
            require(sha(path) == expected, "pin_changed_during_preparation")
        # Full dataset/media verification only once after the work, never per item.
        after, _ = factory({**spec, "dataset_root": str(dataset_path)})
        require(after.identity == initial_identity, "dataset_changed_during_preparation")
        by_id = {(r["role"], r["id"]): r for r in state["items"]}
        require(len(by_id) == len(dataset.corpus) + len(dataset.queries), "complete_item_count")
        sets = []
        for role, items in [("document", dataset.corpus), ("query", dataset.queries)]:
            blocks = []
            for item in items:
                row = by_id[(role, item["id"])]
                path = root / row["path"]
                require(sha(path) == row["sha256"] and tokens(path) == row["shape"], "final_token_seal")
                blocks.append(
                    {k: row[k] for k in ["path", "sha256", "size", "shape"]} | {"item_ids": [item["id"]]}
                )
            sets.append(
                {
                    "set_id": seal({"adapter": adapter.identity, "role": role, "dataset": dataset.identity}),
                    "channel": "S",
                    "role": role,
                    "kind": "ragged",
                    "dimension": 128,
                    "blocks": blocks,
                    "provenance": {
                        "adapter_identity": adapter.identity,
                        "adapter_config": config,
                        "preparation_spec_sha256": binding,
                        "code_files": spec["code_files"],
                        "model_files": spec["model_files"],
                        "search_executed": False,
                    },
                }
            )
        source_files = []
        for path in sorted(dataset_path.iterdir()):
            if path.is_file() and path.suffix in {".json", ".jsonl"}:
                source_files.append({"path": str(path), "sha256": sha(path), "size": path.stat().st_size})
        media = {}
        for item in dataset.corpus + dataset.queries:
            for relative in item.get("media", {}).values():
                path = dataset_path / relative
                media[str(path)] = {"path": str(path), "sha256": sha(path), "size": path.stat().st_size}
        output = {
            "version": 1,
            "artifact_root": str(root),
            "collections": [
                {
                    "dataset_root": str(dataset_path),
                    "frozen_identity": dataset.identity,
                    "source_files": source_files,
                    "media": list(media.values()),
                    "sets": sets,
                }
            ],
        }
        atomic(state_dir / "import-manifest.json", output)
        state["status"] = "completed"
        atomic(state_path, state)
        return {
            "status": "completed",
            "items": len(by_id),
            "new_items": len(pending),
            "manifest_sha256": sha(state_dir / "import-manifest.json"),
            "search_executed": False,
        }
    except BaseException as exc:
        state["status"] = "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed"
        state["error_type"] = type(exc).__name__
        atomic(state_path, state)
        raise
    finally:
        adapter.unload()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--spec-sha256", required=True)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    os.umask(0o077)
    try:
        path = Path(args.spec)
        require(sha(path) == args.spec_sha256, "spec_pin_mismatch")
        spec = json.loads(path.read_text())
        lock_path = inside(Path(spec["artifact_root"]).resolve(), spec["gpu_lock_path"])
        require(
            lock_path == Path(spec["artifact_root"]).resolve() / "work/overnight/gpu.lock",
            "shared_lock_identity",
        )
        require(lock_path.parent.is_dir(), "shared_lock_missing")
        with lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            previous = signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
            try:
                result = prepare(spec, Path(args.state_dir), execute=args.execute, resume=args.resume)
            finally:
                signal.signal(signal.SIGTERM, previous)
        print(json.dumps(result))
        return 0
    except BaseException as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

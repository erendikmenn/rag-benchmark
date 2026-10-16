"""Prepared-only exact RAGTurk candidates; no model construction, encoding or evaluation."""

from __future__ import annotations
import argparse
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).absolute().parents[1]
PROTOCOL = "ragturk-prepared-exact-candidates-v1"
CHANNELS = ("bm25", "bge", "embeddinggemma")
METHODS = CHANNELS + ("bm25_bge", "bm25_embeddinggemma")


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def seal(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def safe(path):
    path = Path(path).absolute()
    require(not any(p.is_symlink() for p in (path, *path.parents)), "symlink_path")
    return path


def write(path, value, *, replace=False):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    require(replace or not path.exists(), "immutable_output_exists")
    with temporary.open("x") as handle:
        os.chmod(temporary, 0o600)
        json.dump(value, handle, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def pure_retrieval():
    """Production pure functions; importing these constructs or loads no models."""
    from rag_benchmark.retrieval import turkish_tokens, reciprocal_rank_fusion

    return turkish_tokens, reciprocal_rank_fusion, sha(ROOT / "src/rag_benchmark/retrieval.py")


def bundle_validator():
    path = ROOT / "scripts/ragturk_pgvector_store.py"
    raw = path.read_bytes()
    name = "_ragturk_prepared_candidate_validator"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module, hashlib.sha256(raw).hexdigest()


def load_prepared(bundle):
    validator, _ = bundle_validator()
    # The CLI never exposes the tiny-fixture counts override.
    return validator.validate_bundle(bundle)


def runtime():
    return {
        "numpy": np.__version__,
        "bm25s": importlib.metadata.version("bm25s"),
        "python": sys.version,
        "search": "numpy-fp32-exact-dot-stable-corpus-ordinal",
        "blas": str(np.__config__.CONFIG),
        "encoding_calls": 0,
    }


def select(prepared, split, limit):
    require(split in {"all", "dev", "test"}, "split")
    require(limit is None or type(limit) is int and limit > 0, "explicit_positive_limit")
    indices = [i for i, s in enumerate(prepared.splits) if split == "all" or s == split]
    if limit is not None:
        indices = indices[:limit]
    require(indices, "empty_selection")
    return indices


def identity(prepared, bundle, indices, *, split, limit, query_batch, block_rows):
    _, _, retrieval_sha = pure_retrieval()
    _, validator_sha = bundle_validator()
    return {
        "protocol": PROTOCOL,
        "bundle_identity": prepared.identity,
        "bundle_sha256": sha(bundle),
        "source_file_sha256": {k: v["sha256"] for k, v in prepared.refs.items()},
        "corpus_order_sha256": seal([r["id"] for r in prepared.chunks]),
        "selected_query_order_sha256": seal([prepared.queries[i]["id"] for i in indices]),
        "selected_split_order_sha256": seal([prepared.splits[i] for i in indices]),
        "split": split,
        "limit": limit,
        "pilot": limit is not None,
        "queries": len(indices),
        "dev_queries": sum(prepared.splits[i] == "dev" for i in indices),
        "test_queries": sum(prepared.splits[i] == "test" for i in indices),
        "corpus_rows": len(prepared.chunks),
        "channels": list(CHANNELS),
        "methods": list(METHODS),
        "channel_pool": 100,
        "candidates": 50,
        "rrf_k": 60,
        "query_batch": query_batch,
        "block_rows": block_rows,
        "embedding_provenance": {
            k: v["model_provenance"] for k, v in prepared.manifest["embeddings"].items()
        },
        "runtime": runtime(),
        "code_sha256": sha(__file__),
        "retrieval_source_sha256": retrieval_sha,
        "validator_source_sha256": validator_sha,
        "new_model_calls": 0,
        "labels_used": False,
    }


def load_bm25(prepared):
    import bm25s

    paths = [p for k, p in prepared.files.items() if k.startswith("bm25/")]
    require(len(paths) == 6 and len({p.parent for p in paths}) == 1, "native_bm25_directory")
    return bm25s.BM25.load(str(paths[0].parent), load_corpus=False, mmap=True)


def rank(scores, corpus_ids, *, positive=False):
    scores = np.asarray(scores)
    require(
        scores.ndim == 1 and len(scores) == len(corpus_ids) and bool(np.isfinite(scores).all()),
        "invalid_scores",
    )
    ordered = np.argsort(-scores, kind="stable")[:100]
    if positive:
        ordered = ordered[scores[ordered] > 0]
    return [
        {"id": corpus_ids[int(i)], "score": float(scores[i]), "rank": n} for n, i in enumerate(ordered, 1)
    ]


def produce(prepared, indices, bm25, *, query_batch):
    """Only approved question text/query matrices and corpus ordinal IDs reach ranking."""
    tokens, rrf, _ = pure_retrieval()
    ids = [r["id"] for r in prepared.chunks]
    for start in range(0, len(indices), query_batch):
        group = indices[start : start + query_batch]
        dense = {}
        for family in CHANNELS[1:]:
            docs = prepared.vectors[family]["documents"]
            queries = prepared.vectors[family]["queries"][group]
            require(docs.dtype == queries.dtype == np.dtype("float32"), "fp32_only")
            scores = queries @ docs.T
            require(
                scores.dtype == np.dtype("float32") and scores.shape == (len(group), len(ids)),
                "exact_score_shape",
            )
            dense[family] = scores
        for offset, index in enumerate(group):
            query = prepared.queries[index]
            channels = {"bm25": rank(bm25.get_scores(tokens(query["question"])), ids, positive=True)}
            channels.update({family: rank(dense[family][offset], ids) for family in CHANNELS[1:]})
            methods = {family: channels[family][:50] for family in CHANNELS}
            methods.update(
                {
                    "bm25_" + family: rrf([channels["bm25"], channels[family]], k=60, top_k=50)
                    for family in CHANNELS[1:]
                }
            )
            methods = {
                method: {
                    "method": method,
                    "query_id": query["id"],
                    "candidate_ids": [r["id"] for r in ranking],
                    "candidate_scores": [r["score"] for r in ranking],
                }
                for method, ranking in methods.items()
            }
            yield {
                "query_id": query["id"],
                "split": prepared.splits[index],
                "question": query["question"],
                "channels": channels,
                "methods": methods,
            }


def check_block(block, expected_rows, prepared, indices, start, stop, run_identity):
    require(
        block["protocol"] == PROTOCOL
        and block["identity"] == run_identity
        and block["start"] == start
        and block["stop"] == stop
        and len(block["rows"]) == stop - start,
        "block_identity",
    )
    require(block["rows"] == expected_rows, "block_payload_changed")
    require(
        [r["query_id"] for r in block["rows"]]
        == [prepared.queries[indices[i]]["id"] for i in range(start, stop)],
        "block_query_order",
    )


def run(
    bundle,
    output,
    *,
    split="all",
    limit=None,
    query_batch=16,
    block_rows=128,
    resume=False,
    prepared=None,
    bm25=None,
):
    """One writer, immutable sequential blocks; resumption revalidates every completed block."""
    require(
        type(query_batch) is int
        and 1 <= query_batch <= 64
        and type(block_rows) is int
        and 1 <= block_rows <= 1024,
        "bounded_batches",
    )
    bundle = safe(bundle)
    output = safe(output)
    prepared = prepared if prepared is not None else load_prepared(bundle)
    indices = select(prepared, split, limit)
    details = identity(
        prepared, bundle, indices, split=split, limit=limit, query_batch=query_batch, block_rows=block_rows
    )
    run_identity = seal(details)
    prepared.stable_inputs()
    if output.exists():
        require(resume, "explicit_resume_required")
    else:
        require(not resume, "resume_missing_output")
        output.mkdir(mode=0o700, parents=True)
    lock = os.open(output / ".lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if resume:
            require(
                read(output / "manifest.json") == {"identity": run_identity, **details},
                "resume_identity_changed",
            )
            require(read(output / "claim.json")["identity"] == run_identity, "resume_claim")
            state = read(output / "state.json")
            require(
                state["status"] in {"running", "interrupted", "completed"}, "failed_state_requires_review"
            )
        else:
            require(not any(p.name != ".lock" for p in output.iterdir()), "unknown_output_contents")
            write(output / "claim.json", {"identity": run_identity, "pid": os.getpid()})
            write(output / "manifest.json", {"identity": run_identity, **details})
            (output / "blocks").mkdir(mode=0o700)
            state = {"status": "running", "identity": run_identity, "completed_rows": 0, "blocks": []}
            write(output / "state.json", state)
        require(not list(output.rglob("*.partial")), "incomplete_atomic_write_requires_review")
        bm25 = bm25 if bm25 is not None else load_bm25(prepared)
        expected_names = {
            f"{start:09d}-{min(start + block_rows, len(indices)):09d}.json"
            for start in range(0, len(indices), block_rows)
        }
        require({p.name for p in (output / "blocks").iterdir()} <= expected_names, "unknown_checkpoint_block")
        recorded = {r["name"]: r for r in state["blocks"]}
        completed = []
        begun = time.monotonic()
        try:
            for start in range(0, len(indices), block_rows):
                stop = min(start + block_rows, len(indices))
                name = f"{start:09d}-{stop:09d}.json"
                path = output / "blocks" / name
                expected = list(produce(prepared, indices[start:stop], bm25, query_batch=query_batch))
                if path.exists():
                    # Recompute searches only, never embeddings; verify payload even after publication/state interruption.
                    check_block(read(path), expected, prepared, indices, start, stop, run_identity)
                    if name in recorded:
                        require(recorded[name]["sha256"] == sha(path), "checkpoint_sha_changed")
                else:
                    require(name not in recorded, "completed_block_missing")
                    write(
                        path,
                        {
                            "protocol": PROTOCOL,
                            "identity": run_identity,
                            "start": start,
                            "stop": stop,
                            "rows": expected,
                        },
                    )
                completed.append({"name": name, "sha256": sha(path), "rows": stop - start})
                state = {
                    "status": "running",
                    "identity": run_identity,
                    "completed_rows": stop,
                    "blocks": completed,
                    "elapsed_seconds": time.monotonic() - begun,
                }
                write(output / "state.json", state, replace=True)
            prepared.stable_inputs()
            require(
                identity(
                    prepared,
                    bundle,
                    indices,
                    split=split,
                    limit=limit,
                    query_batch=query_batch,
                    block_rows=block_rows,
                )
                == details,
                "inputs_or_runtime_changed",
            )
            summary = {
                "status": "completed",
                "identity": run_identity,
                "queries": len(indices),
                "dev_queries": details["dev_queries"],
                "test_queries": details["test_queries"],
                "corpus_rows": len(prepared.chunks),
                "methods": list(METHODS),
                "candidate_records": len(indices) * 5,
                "blocks": completed,
                "new_model_calls": 0,
                "labels_used": False,
                "benchmark_runs": 0,
                "retrieval_preparation_only": True,
                "elapsed_seconds": time.monotonic() - begun,
            }
            if (output / "summary.json").exists():
                require(
                    {k: v for k, v in read(output / "summary.json").items() if k != "elapsed_seconds"}
                    == {k: v for k, v in summary.items() if k != "elapsed_seconds"},
                    "summary_identity",
                )
            else:
                write(output / "summary.json", summary)
            state.update(status="completed")
            write(output / "state.json", state, replace=True)
            return summary
        except BaseException as error:
            state.update(
                status="interrupted"
                if isinstance(error, (KeyboardInterrupt, SystemExit))
                else "failed_requires_review",
                error_type=type(error).__name__,
            )
            write(output / "state.json", state, replace=True)
            raise
    finally:
        os.close(lock)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split", choices=["all", "dev", "test"], default="all")
    p.add_argument("--limit", type=int)
    p.add_argument("--query-batch", type=int, default=16)
    p.add_argument("--block-rows", type=int, default=128)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    os.umask(0o077)
    result = run(
        a.bundle,
        a.output,
        split=a.split,
        limit=a.limit,
        query_batch=a.query_batch,
        block_rows=a.block_rows,
        resume=a.resume,
    )
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "status",
                    "identity",
                    "queries",
                    "dev_queries",
                    "test_queries",
                    "candidate_records",
                    "new_model_calls",
                )
            }
        )
    )


if __name__ == "__main__":
    main()

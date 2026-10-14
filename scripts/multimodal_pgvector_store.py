"""Private, encoding-free multimodal preparation import; never touches RAGTurk."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

CHUNK = 4 * 1024 * 1024
SCHEMA = "multimodal_prepared"


def require(ok, code):
    if not ok:
        raise ValueError(code)


def seal(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def resolve(root, value):
    p = Path(value)
    p = p if p.is_absolute() else root / p
    require(p.resolve().is_relative_to(root), "path_escape")
    require(not any(x.is_symlink() for x in (p, *p.parents)), "symlink_path")
    return p.resolve()


def checked_file(root, spec):
    p = resolve(root, spec["path"])
    require(type(spec["size"]) is int and spec["size"] >= 0, "file_size_schema")
    require(p.is_file() and p.stat().st_size == spec["size"] and digest(p) == spec["sha256"], "file_seal")
    return p


def array(path, block, dimension, kind):
    a = np.load(path, mmap_mode="r", allow_pickle=False)
    require(a.dtype == np.dtype("float32") and a.ndim == 2, "fp32_matrix_required")
    require(list(a.shape) == block["shape"] and a.shape[1] == dimension, "shape_binding")
    require(a.shape[0] > 0, "empty_matrix")
    require(len(block["item_ids"]) == (a.shape[0] if kind == "dense" else 1), "item_shape_binding")
    for start in range(0, len(a), 1024):
        rows = a[start : start + 1024]
        require(np.isfinite(rows).all(), "nonfinite")
        require(np.allclose(np.linalg.norm(rows, axis=1), 1, atol=2e-3, rtol=2e-3), "normalization")
    return a


def inspect_manifest(manifest):
    require(
        manifest.get("version") == 1 and isinstance(manifest.get("collections"), list), "manifest_version"
    )
    root = Path(manifest["artifact_root"])
    require(root.is_absolute() and root.is_dir(), "artifact_root")
    root = resolve(root.resolve(), str(root))
    seen = set()
    for c in manifest["collections"]:
        identity = c["frozen_identity"]
        require(identity not in seen and len(identity) == 64, "collection_identity")
        seen.add(identity)
        dataset = resolve(root, c["dataset_root"])
        files = {resolve(root, f["path"]): f for f in c["source_files"]}
        require(len(files) == len(c["source_files"]), "duplicate_source_path")
        hashes = {}
        ids = {}
        source_rows = {}
        for name in ("dataset.json", "corpus.jsonl", "queries.jsonl", "qrels.jsonl"):
            p = dataset / name
            require(p in files, "canonical_source_required")
            checked_file(root, files[p])
            hashes[name] = files[p]["sha256"]
        media_hashes = {}
        for role, name in [("document", "corpus.jsonl"), ("query", "queries.jsonl")]:
            rows = [json.loads(line) for line in (dataset / name).read_text().splitlines() if line.strip()]
            ids[role] = [str(row["id"]) for row in rows]
            source_rows[role] = {str(row["id"]): row for row in rows}
            require(ids[role] and len(ids[role]) == len(set(ids[role])), "source_ids")
            for row in rows:
                for relative in row.get("media", {}).values():
                    p = resolve(root, str(dataset / relative))
                    require(p.is_relative_to(dataset), "media_dataset_escape")
                    media_hashes[relative] = digest(p)
        require(seal({"files": hashes, "media": media_hashes}) == identity, "frozen_identity_mismatch")
        media_specs = {checked_file(root, f) for f in c.get("media", [])}
        require(
            all(resolve(root, str(dataset / p)) in media_specs for p in media_hashes),
            "media_inventory_missing",
        )
        for f in c["source_files"] + c.get("lexical_artifacts", []):
            checked_file(root, f)
        sets = set()
        for s in c["sets"]:
            require(s["set_id"] not in sets, "duplicate_set")
            sets.add(s["set_id"])
            require(
                s["channel"] in "BGENSJ" and s["role"] in ids and s["kind"] in {"dense", "ragged"},
                "set_schema",
            )
            require(
                type(s["dimension"]) is int and s["dimension"] > 0 and bool(s["provenance"]),
                "representation_schema",
            )
            require(s["channel"] != "B", "BM25_is_lexical")
            order = []
            source_order = []
            for block in s["blocks"]:
                a = array(checked_file(root, block), block, s["dimension"], s["kind"])
                del a
                order.extend(block["item_ids"])
                source_order.extend(block.get("source_item_ids", block["item_ids"]))
            if "source_segments" in s:
                require(s["role"] == "document" and s["kind"] == "dense", "segmented_set_kind")
                require(len(order) == len(set(order)), "duplicate_chunk_id")
                segments = s["source_segments"]
                require([x["source_item_id"] for x in segments] == ids["document"], "segmented_source_order")
                expected_chunks = []
                expected_sources = []
                for entry in segments:
                    source_id = entry["source_item_id"]
                    text = source_rows["document"][source_id].get("text")
                    require(isinstance(text, str) and len(text) > 0, "segmented_source_text")
                    require(
                        entry["source_length"] == len(text)
                        and entry["source_sha256"] == hashlib.sha256(text.encode()).hexdigest(),
                        "segmented_source_seal",
                    )
                    stop = 0
                    for chunk in entry["chunks"]:
                        require(
                            type(chunk["start"]) is int
                            and type(chunk["end"]) is int
                            and chunk["start"] == stop
                            and stop < chunk["end"] <= len(text),
                            "segment_gap_or_overlap",
                        )
                        stop = chunk["end"]
                        expected_chunks.append(chunk["item_id"])
                        expected_sources.append(source_id)
                    require(stop == len(text), "segment_incomplete")
                require(order == expected_chunks and source_order == expected_sources, "chunk_order_binding")
            else:
                require(order == ids[s["role"]] and source_order == order, "source_order_binding")
    return root


def setup(cur):
    cur.execute("SELECT 1 FROM pg_extension WHERE extname='vector'")
    require(cur.fetchone() is not None, "pgvector_extension_required")
    for sql in [
        "CREATE SCHEMA IF NOT EXISTS multimodal_prepared",
        "CREATE TABLE IF NOT EXISTS multimodal_prepared.collections(identity text PRIMARY KEY, manifest_sha text NOT NULL,manifest jsonb NOT NULL)",
        "CREATE TABLE IF NOT EXISTS multimodal_prepared.blobs(sha text PRIMARY KEY,size bigint NOT NULL)",
        "CREATE TABLE IF NOT EXISTS multimodal_prepared.blob_chunks(sha text REFERENCES multimodal_prepared.blobs,ordinal integer,data bytea NOT NULL,PRIMARY KEY(sha,ordinal))",
        "CREATE TABLE IF NOT EXISTS multimodal_prepared.source_files(collection text REFERENCES multimodal_prepared.collections,path text,sha text REFERENCES multimodal_prepared.blobs,kind text,PRIMARY KEY(collection,path))",
        "CREATE TABLE IF NOT EXISTS multimodal_prepared.labels(collection text REFERENCES multimodal_prepared.collections,path text,sha text REFERENCES multimodal_prepared.blobs,PRIMARY KEY(collection,path))",
        "CREATE TABLE IF NOT EXISTS multimodal_prepared.sets(collection text REFERENCES multimodal_prepared.collections,id text,manifest_sha text,status text,manifest jsonb NOT NULL,PRIMARY KEY(collection,id))",
        "CREATE TABLE IF NOT EXISTS multimodal_prepared.embeddings(collection text,set_id text,ordinal integer,item_id text,dimension integer NOT NULL CHECK(dimension>0),dense vector,ragged bytea,rows integer NOT NULL CHECK(rows>0),sha text NOT NULL CHECK(sha ~ '^[0-9a-f]{64}$'),PRIMARY KEY(collection,set_id,ordinal),UNIQUE(collection,set_id,item_id),CHECK(ordinal>=0),CHECK((dense IS NOT NULL AND ragged IS NULL AND rows=1 AND vector_dims(dense)=dimension) OR (dense IS NULL AND ragged IS NOT NULL AND octet_length(ragged)=rows::bigint*dimension*4)),FOREIGN KEY(collection,set_id) REFERENCES multimodal_prepared.sets)",
    ]:
        cur.execute(sql)


def blob(cur, path, spec, importing):
    sha = spec["sha256"]
    cur.execute("SELECT size FROM multimodal_prepared.blobs WHERE sha=%s", (sha,))
    found = cur.fetchone()
    if found is None:
        require(importing, "missing_blob")
        cur.execute("INSERT INTO multimodal_prepared.blobs VALUES (%s,%s)", (sha, spec["size"]))
        with path.open("rb") as h:
            ordinal = 0
            while data := h.read(CHUNK):
                cur.execute(
                    "INSERT INTO multimodal_prepared.blob_chunks VALUES (%s,%s,%s)", (sha, ordinal, data)
                )
                ordinal += 1
    else:
        require(found[0] == spec["size"], "blob_collision")
    h = hashlib.sha256()
    count = 0
    ordinal = 0
    while True:
        cur.execute(
            "SELECT data FROM multimodal_prepared.blob_chunks WHERE sha=%s AND ordinal=%s", (sha, ordinal)
        )
        row = cur.fetchone()
        if row is None:
            break
        data = bytes(row[0])
        require(0 < len(data) <= CHUNK, "blob_chunk_size")
        h.update(data)
        count += len(data)
        ordinal += 1
    cur.execute("SELECT count(*) FROM multimodal_prepared.blob_chunks WHERE sha=%s", (sha,))
    require(cur.fetchone()[0] == ordinal and count == spec["size"] and h.hexdigest() == sha, "blob_readback")
    require(digest(path) == sha, "source_changed_during_import")


def row_bytes(value):
    return np.asarray(value, dtype="<f4").tobytes(order="C")


def check_row(stored, expected, dimension, rows):
    item_id, dim, dense, ragged, count, sha = stored
    require(dim == dimension and count == rows, "stored_shape")
    if dense is not None:
        require(ragged is None and rows == 1, "stored_kind")
        data = row_bytes(json.loads(dense, parse_int=float))
    else:
        require(ragged is not None, "stored_kind")
        data = bytes(ragged)
    require(data == expected and hashlib.sha256(data).hexdigest() == sha, "vector_readback")
    return item_id


def validate_set_state(old, binding, manifest=None):
    require(old[:2] == (binding, "complete"), "set_collision_or_incomplete")
    if manifest is not None:
        require(len(old) == 3 and old[2] == manifest and seal(old[2]) == binding, "stored_manifest_mutation")


def collection_source(c):
    return {k: c[k] for k in ("dataset_root", "frozen_identity", "source_files", "media")}


def vector_batch(cur, key, s, ordinal, items, values, inserting):
    expected = []
    params = []
    for index, item in enumerate(items):
        value = values[index] if s["kind"] == "dense" else values
        data = row_bytes(value)
        count = 1 if s["kind"] == "dense" else len(values)
        expected.append((item, data, count))
        params.append(
            (
                *key,
                ordinal + index,
                item,
                s["dimension"],
                json.dumps(value.tolist()) if s["kind"] == "dense" else None,
                data if s["kind"] == "ragged" else None,
                count,
                hashlib.sha256(data).hexdigest(),
            )
        )
    if inserting:
        cur.executemany(
            "INSERT INTO multimodal_prepared.embeddings VALUES (%s,%s,%s,%s,%s,%s::vector,%s,%s,%s)", params
        )
    cur.execute(
        "SELECT ordinal,item_id,dimension,dense::text,ragged,rows,sha FROM multimodal_prepared.embeddings WHERE collection=%s AND set_id=%s AND ordinal>=%s AND ordinal<%s ORDER BY ordinal",
        (*key, ordinal, ordinal + len(items)),
    )
    rows = cur.fetchall()
    require(len(rows) == len(expected), "missing_embedding")
    for offset, (row, (item, data, count)) in enumerate(zip(rows, expected, strict=True)):
        require(row[0] == ordinal + offset, "stored_ordinal")
        require(check_row(row[1:], data, s["dimension"], count) == item, "stored_item_id")
    return len(items)


def transfer(conn, manifest, importing):
    root = inspect_manifest(manifest)
    with conn.transaction():
        cur = conn.cursor()
        if not importing:
            cur.execute("SET TRANSACTION READ ONLY")
        if importing:
            setup(cur)
        for c in manifest["collections"]:
            identity = c["frozen_identity"]
            source = collection_source(c)
            binding = seal(source)
            cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (identity,))
            cur.execute(
                "SELECT manifest_sha,manifest FROM multimodal_prepared.collections WHERE identity=%s",
                (identity,),
            )
            old = cur.fetchone()
            if old is None:
                require(importing, "missing_collection")
                cur.execute(
                    "INSERT INTO multimodal_prepared.collections VALUES (%s,%s,%s::jsonb)",
                    (identity, binding, json.dumps(source)),
                )
            else:
                require(
                    old[0] == binding and old[1] == source and seal(old[1]) == binding, "collection_mutation"
                )
            for kind in ("source_files", "media", "lexical_artifacts"):
                for f in c.get(kind, []):
                    path = checked_file(root, f)
                    blob(cur, path, f, importing)
                    label = path.name == "qrels.jsonl" or f.get("private_labels") is True
                    table = "labels" if label else "source_files"
                    relative = str(path.relative_to(root))
                    cur.execute(
                        f"SELECT sha FROM multimodal_prepared.{table} WHERE collection=%s AND path=%s",
                        (identity, relative),
                    )
                    row = cur.fetchone()
                    if row is None:
                        require(importing, "missing_file_reference")
                        if label:
                            cur.execute(
                                "INSERT INTO multimodal_prepared.labels VALUES (%s,%s,%s)",
                                (identity, relative, f["sha256"]),
                            )
                        else:
                            cur.execute(
                                "INSERT INTO multimodal_prepared.source_files VALUES (%s,%s,%s,%s)",
                                (identity, relative, f["sha256"], kind),
                            )
                    else:
                        require(row[0] == f["sha256"], "file_reference_collision")
    # Separate transactions allow completed sets to resume after interruption.
    for c in manifest["collections"]:
        for s in c["sets"]:
            with conn.transaction():
                cur = conn.cursor()
                if not importing:
                    cur.execute("SET TRANSACTION READ ONLY")
                key = (c["frozen_identity"], s["set_id"])
                binding = seal(s)
                cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (seal(key),))
                cur.execute(
                    "SELECT manifest_sha,status,manifest FROM multimodal_prepared.sets WHERE collection=%s AND id=%s",
                    key,
                )
                old = cur.fetchone()
                if old is None:
                    require(importing, "missing_set")
                    cur.execute(
                        "INSERT INTO multimodal_prepared.sets VALUES (%s,%s,%s,%s,%s::jsonb)",
                        (*key, binding, "importing", json.dumps(s)),
                    )
                else:
                    validate_set_state(old, binding, s)
                ordinal = 0
                for block in s["blocks"]:
                    path = checked_file(root, block)
                    a = array(path, block, s["dimension"], s["kind"])
                    batch_size = 256 if s["kind"] == "dense" else 1
                    for start in range(0, len(block["item_ids"]), batch_size):
                        items = block["item_ids"][start : start + batch_size]
                        values = a[start : start + len(items)] if s["kind"] == "dense" else a
                        ordinal += vector_batch(cur, key, s, ordinal, items, values, old is None)
                    require(digest(path) == block["sha256"], "block_changed_during_import")
                cur.execute(
                    "SELECT count(*) FROM multimodal_prepared.embeddings WHERE collection=%s AND set_id=%s",
                    key,
                )
                require(cur.fetchone()[0] == ordinal, "stored_count")
                if old is None:
                    cur.execute(
                        "UPDATE multimodal_prepared.sets SET status='complete' WHERE collection=%s AND id=%s",
                        key,
                    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["import", "verify"])
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dsn-file")
    args = parser.parse_args(argv)
    try:
        if args.dsn_file:
            p = Path(args.dsn_file)
            require(not p.is_symlink() and p.stat().st_mode & 0o077 == 0, "private_dsn_file_required")
            dsn = p.read_text().strip()
        else:
            dsn = os.environ.get("MULTIMODAL_PG_DSN", "")
        require(bool(dsn), "dsn_required")
        manifest = json.loads(Path(args.manifest).read_text())
        import psycopg

        with psycopg.connect(dsn) as conn:
            transfer(conn, manifest, args.mode == "import")
        print(json.dumps({"status": "verified", "collections": len(manifest["collections"])}))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

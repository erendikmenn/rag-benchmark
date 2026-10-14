"""Validate/import frozen RAGTurk preparation only; no inference or benchmark calls.

The native bm25s files are stored verbatim. This is NOT a PostgreSQL BM25 search
implementation. psycopg and an already installed pgvector extension are needed
only for explicitly requested import/verify commands.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

import numpy as np

SCHEMA = "ragturk_prepared"
COUNTS = {"corpus": 37511, "dev": 2000, "test": 12530}
DIMENSIONS = {"bge": 1024, "embeddinggemma": 768}
MODEL_IDS = {"bge": "BAAI/bge-m3", "embeddinggemma": "google/embeddinggemma-2"}
BM25_FILES = {"ready.json", "params.index.json", "vocab.index.json", "data.csc.index.npy",
              "indices.csc.index.npy", "indptr.csc.index.npy"}
TABLES = {"bundle", "chunks", "questions", "question_labels", "files"}
MAX_METADATA_BYTES = 8 * 1024 * 1024


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def require(condition: bool, code: str) -> None:
    if not condition:
        raise ValueError(code)


def safe(path: Path | str) -> Path:
    path = Path(path).absolute()
    require(".." not in path.parts and not any(p.is_symlink() for p in (path, *path.parents)), "unsafe_path")
    require(path.is_file(), "regular_file_required")
    return path


def json_file(path: Path, *, bounded: bool = False) -> dict:
    path = safe(path)
    require(not bounded or path.stat().st_size <= MAX_METADATA_BYTES, "metadata_bound")
    def pairs(values):
        result = {}
        for key, value in values:
            require(key not in result, "duplicate_json_key")
            result[key] = value
        return result
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite_json")))


def checked_ref(ref: dict, base: Path) -> Path:
    require(isinstance(ref, dict) and set(ref) == {"path", "sha256", "bytes"}, "file_ref_schema")
    require(isinstance(ref["path"], str) and bool(ref["path"]), "file_path")
    path = Path(ref["path"])
    path = safe(path if path.is_absolute() else base / path)
    require(type(ref["bytes"]) is int and ref["bytes"] >= 0 and path.stat().st_size == ref["bytes"], "file_size")
    require(isinstance(ref["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", ref["sha256"]), "file_sha_shape")
    require(file_sha(path) == ref["sha256"], "file_sha")
    return path


def records(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        values = [json.loads(line) for line in handle if line.strip()]
    require(all(isinstance(row, dict) for row in values), "record_shape")
    return values


def strings(value: Any, code: str, *, nonempty: bool = True) -> None:
    require(isinstance(value, str) and (not nonempty or bool(value.strip())) and "\x00" not in value, code)


def vector_array(path: Path, rows: int, dimension: int) -> np.ndarray:
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    require(values.dtype == np.dtype("float32") and values.shape == (rows, dimension), "vector_shape_dtype")
    for start in range(0, rows, 256):
        block = values[start:start + 256]
        require(bool(np.isfinite(block).all()), "vector_nonfinite")
        norms = np.linalg.norm(block, axis=1)
        require(bool(np.all(np.abs(norms - 1.0) <= 1e-5)), "vector_normalization")
    return values


@dataclass
class Prepared:
    manifest: dict
    identity: str
    chunks: list[dict]
    queries: list[dict]
    splits: list[str]
    vectors: dict[str, dict[str, np.ndarray]]
    files: dict[str, Path]
    refs: dict[str, dict]

    def stable_inputs(self) -> None:
        for name, path in self.files.items():
            require(safe(path).stat().st_size == self.refs[name]["bytes"] and
                    file_sha(path) == self.refs[name]["sha256"], "input_changed")

    def summary(self) -> dict:
        return {"status": "prepared_bundle_validated", "identity": self.identity,
                "chunks": len(self.chunks), "questions": len(self.queries),
                "dev_questions": self.splits.count("dev"), "test_questions": self.splits.count("test"),
                "dimensions": DIMENSIONS, "bm25_native_files": 6, "new_model_calls": 0,
                "native_bm25_stored_not_sql_search": True}


def validate_bundle(bundle: Path | str, *, counts: dict | None = None) -> Prepared:
    """counts injection exists only for small CPU fixtures, not CLI production overrides."""
    bundle = safe(bundle)
    manifest = json_file(bundle, bounded=True)
    require(manifest.get("schema_version") == 1, "bundle_schema")
    identity = manifest.get("identity")
    require(identity == canonical_sha({k: v for k, v in manifest.items() if k != "identity"}), "bundle_identity")
    expected = COUNTS if counts is None else counts
    dataset = manifest["dataset"]
    require(dataset.get("id") == "metunlp/ragturk" and isinstance(dataset.get("revision"), str)
            and re.fullmatch(r"[0-9a-f]{40}", dataset["revision"]), "dataset_revision")
    files, refs = {}, {}
    for key, filename in (("manifest", "manifest.json"), ("corpus", "corpus.jsonl"),
                          ("questions_dev", "questions.dev.jsonl"), ("questions_test", "questions.test.jsonl")):
        path = checked_ref(dataset[key], bundle.parent)
        files[filename], refs[filename] = path, dataset[key]
    source = json_file(files["manifest.json"], bounded=True)
    require(source.get("dataset") == dataset["id"] and source.get("revision") == dataset["revision"], "source_manifest_identity")
    for name in ("corpus.jsonl", "questions.dev.jsonl", "questions.test.jsonl"):
        require(source["file_sha256"][name] == refs[name]["sha256"], "frozen_source_hash")
    chunks = records(files["corpus.jsonl"])
    dev, test = records(files["questions.dev.jsonl"]), records(files["questions.test.jsonl"])
    require((len(chunks), len(dev), len(test)) == (expected["corpus"], expected["dev"], expected["test"]), "dataset_counts")
    require(source["counts"]["corpus_chunks"] == len(chunks) and source["counts"]["dev_questions"] == len(dev)
            and source["counts"]["test_questions"] == len(test), "source_counts")
    queries, splits = dev + test, ["dev"] * len(dev) + ["test"] * len(test)
    chunk_ids, query_ids = [r["id"] for r in chunks], [r["id"] for r in queries]
    require(len(chunk_ids) == len(set(chunk_ids)) and len(query_ids) == len(set(query_ids)), "duplicate_ids")
    require(canonical_sha(chunk_ids) == dataset["corpus_order_sha256"] and
            canonical_sha(query_ids) == dataset["query_order_sha256"], "row_order")
    gallery = set(chunk_ids)
    for row in chunks:
        require(set(row) == {"id", "text", "title", "article_id", "source"}, "chunk_fields")
        for key in ("id", "text", "article_id", "source"):
            strings(row[key], "chunk_text")
        strings(row["title"], "chunk_title", nonempty=False)
    for row in queries:
        require(set(row) == {"id", "question", "article_id", "source", "category", "answers", "gold_ids"}, "query_fields")
        for key in ("id", "question", "article_id", "source", "category"):
            strings(row[key], "query_text")
        require(isinstance(row["answers"], list) and bool(row["answers"]), "answers")
        for answer in row["answers"]:
            strings(answer, "answer_text")
        require(isinstance(row["gold_ids"], list) and bool(row["gold_ids"]) and
                len(row["gold_ids"]) == len(set(row["gold_ids"])) and set(row["gold_ids"]) <= gallery, "gold_ids")
    vectors = {}
    require(set(manifest["embeddings"]) == set(DIMENSIONS), "embedding_families")
    for name, dimension in DIMENSIONS.items():
        spec = manifest["embeddings"][name]
        provenance = spec["model_provenance"]
        require(spec["dimension"] == dimension and provenance["model_id"] == MODEL_IDS[name]
                and re.fullmatch(r"[0-9a-f]{40}", provenance["revision"]), "model_pin_dimension")
        require(provenance["dtype"] == "float32" and provenance["device"] in {"cpu", "mps", "cuda"}
                and isinstance(provenance["package_versions"], dict) and bool(provenance["package_versions"]), "model_provenance")
        require(spec["corpus_order_sha256"] == dataset["corpus_order_sha256"] and
                spec["query_order_sha256"] == dataset["query_order_sha256"], "vector_order_binding")
        vectors[name] = {}
        for role, size in (("documents", len(chunks)), ("queries", len(queries))):
            ref = spec[role]
            path = checked_ref(ref, bundle.parent)
            key = name + "/" + role + ".npy"
            files[key], refs[key] = path, ref
            vectors[name][role] = vector_array(path, size, dimension)
    bm25 = manifest["bm25"]
    require(bm25["corpus_order_sha256"] == dataset["corpus_order_sha256"] and
            isinstance(bm25["identity"], str) and re.fullmatch(r"[0-9a-f]{64}", bm25["identity"]), "bm25_identity")
    require(isinstance(bm25["files"], list) and len(bm25["files"]) == 6, "bm25_inventory")
    names = set()
    for ref in bm25["files"]:
        path = checked_ref(ref, bundle.parent)
        name = path.name
        require(name in BM25_FILES and name not in names, "bm25_names")
        names.add(name)
        files["bm25/" + name], refs["bm25/" + name] = path, ref
    require(names == BM25_FILES, "bm25_inventory")
    ready = json_file(files["bm25/ready.json"])
    require(ready["count"] == len(chunks) and ready["corpus_hash"] ==
            canonical_sha([{k: r[k] for k in ("id", "title", "text")} for r in chunks]), "bm25_corpus")
    params = json_file(files["bm25/params.index.json"])
    require(params["num_docs"] == len(chunks) and params["method"] == "lucene" and
            params["k1"] == 1.5 and params["b"] == 0.75, "bm25_parameters")
    data, indices, indptr = [np.load(files["bm25/" + name + ".csc.index.npy"], mmap_mode="r", allow_pickle=False)
                             for name in ("data", "indices", "indptr")]
    vocab = json_file(files["bm25/vocab.index.json"])
    integer_types = {np.dtype("int32"), np.dtype("int64")}
    require(data.ndim == indices.ndim == indptr.ndim == 1 and data.dtype == np.dtype("float32") and
            indices.dtype in integer_types and indptr.dtype in integer_types and len(indptr) >= 2 and
            len(data) == len(indices) and indptr[0] == 0 and indptr[-1] == len(data)
            and bool(np.all((indptr >= 0) & (indptr <= len(data))))
            and bool(np.all(np.diff(indptr.astype(np.int64)) >= 0)) and bool(np.isfinite(data).all())
            and bool(np.all((indices >= 0) & (indices < len(chunks)))) and isinstance(vocab, dict)
            and all(isinstance(token, str) and type(index) is int for token, index in vocab.items()), "bm25_csc")
    columns = len(indptr) - 1
    # bm25s 0.3.12 builds physical CSC columns first, then appends the empty
    # query token at max(vocab)+1. Its NumPy CSC backend emits int64 indptr.
    # Permit exactly that sentinel, never a nonempty out-of-range token.
    physical = {token: index for token, index in vocab.items() if index < columns}
    ordinary = len(vocab) == columns and set(vocab.values()) == set(range(columns))
    sentinel = (len(vocab) == columns + 1 and vocab.get("") == columns and
                len(physical) == columns and set(physical.values()) == set(range(columns)))
    require(ordinary or sentinel, "bm25_vocabulary_columns")
    prepared = Prepared(manifest, identity, chunks, queries, splits, vectors, files, refs)
    prepared.stable_inputs()
    return prepared


def vector_text(values: np.ndarray) -> str:
    return "[" + ",".join(str(float(x)) for x in values) + "]"


def ddl(cursor) -> None:
    cursor.execute("SELECT 1 FROM pg_extension WHERE extname='vector'")
    require(cursor.fetchone() is not None, "pgvector_extension_required")
    cursor.execute("CREATE SCHEMA ragturk_prepared")
    cursor.execute("CREATE TABLE ragturk_prepared.bundle (singleton integer PRIMARY KEY CHECK(singleton=1), identity text NOT NULL UNIQUE, state text NOT NULL CHECK(state IN ('prepared','complete')), manifest jsonb NOT NULL)")
    cursor.execute("CREATE TABLE ragturk_prepared.chunks (id text PRIMARY KEY, ordinal integer NOT NULL UNIQUE CHECK(ordinal>=0), text text NOT NULL, title text NOT NULL, article_id text NOT NULL, source text NOT NULL, bge vector(1024) NOT NULL, embeddinggemma vector(768) NOT NULL)")
    cursor.execute("CREATE TABLE ragturk_prepared.questions (id text PRIMARY KEY, ordinal integer NOT NULL UNIQUE CHECK(ordinal>=0), split text NOT NULL CHECK(split IN ('dev','test')), question text NOT NULL, article_id text NOT NULL, source text NOT NULL, category text NOT NULL, bge vector(1024) NOT NULL, embeddinggemma vector(768) NOT NULL)")
    cursor.execute("CREATE TABLE ragturk_prepared.question_labels (question_id text PRIMARY KEY REFERENCES ragturk_prepared.questions(id), reference_answers jsonb NOT NULL, gold_ids jsonb NOT NULL)")
    cursor.execute("CREATE TABLE ragturk_prepared.files (name text PRIMARY KEY, sha256 text NOT NULL, bytes bigint NOT NULL CHECK(bytes>=0), payload bytea NOT NULL CHECK(octet_length(payload)=bytes))")


def chunk_rows(prepared: Prepared):
    for ordinal, row in enumerate(prepared.chunks):
        yield (row["id"], ordinal, row["text"], row["title"], row["article_id"], row["source"],
               vector_text(prepared.vectors["bge"]["documents"][ordinal]),
               vector_text(prepared.vectors["embeddinggemma"]["documents"][ordinal]))


def question_rows(prepared: Prepared):
    for ordinal, (row, split) in enumerate(zip(prepared.queries, prepared.splits, strict=True)):
        yield (row["id"], ordinal, split, row["question"], row["article_id"], row["source"], row["category"],
               vector_text(prepared.vectors["bge"]["queries"][ordinal]),
               vector_text(prepared.vectors["embeddinggemma"]["queries"][ordinal]))


def copy_rows(cursor, table: str, values) -> None:
    require(table in {"chunks", "questions", "question_labels"}, "copy_table")
    # Text COPY accepts vector literals; no pgvector Python adapter or tensor load.
    with cursor.copy("COPY ragturk_prepared." + table + " FROM STDIN") as copy:
        for value in values:
            copy.write_row(value)


def verify_database(cursor, prepared: Prepared) -> dict:
    cursor.execute("SELECT identity,state,manifest FROM ragturk_prepared.bundle WHERE singleton=1")
    found = cursor.fetchone()
    require(found is not None and found[0] == prepared.identity and found[1] == "complete" and
            found[2] == prepared.manifest, "existing_bundle_mismatch")
    for table, count in (("bundle", 1), ("chunks", len(prepared.chunks)), ("questions", len(prepared.queries)),
                         ("question_labels", len(prepared.queries)), ("files", 10)):
        cursor.execute("SELECT count(*) FROM ragturk_prepared." + table)
        require(cursor.fetchone()[0] == count, "database_counts")
    for table, expected, fields in (("chunks", chunk_rows(prepared), "id,ordinal,text,title,article_id,source,bge::text,embeddinggemma::text"),
                                    ("questions", question_rows(prepared), "id,ordinal,split,question,article_id,source,category,bge::text,embeddinggemma::text")):
        expected = iter(expected)
        for start in range(0, len(prepared.chunks) if table == "chunks" else len(prepared.queries), 128):
            cursor.execute("SELECT " + fields + " FROM ragturk_prepared." + table + " WHERE ordinal >= %s ORDER BY ordinal LIMIT 128", (start,))
            actual = cursor.fetchall()
            for row in actual:
                value = next(expected)
                require(tuple(row[:-2]) == value[:-2], "database_record_order")
                for stored, wanted in zip(row[-2:], value[-2:], strict=True):
                    require(isinstance(stored, str) and stored.startswith("[") and stored.endswith("]"), "database_vector_format")
                    a, b = [np.fromstring(v[1:-1], sep=",", dtype=np.float32) for v in (stored, wanted)]
                    require(np.array_equal(a, b), "database_vector_bytes")
        require(next(expected, None) is None, "database_row_coverage")
    cursor.execute("SELECT question_id,reference_answers,gold_ids FROM ragturk_prepared.question_labels")
    labels = {row[0]: row[1:] for row in cursor.fetchall()}
    require(len(labels) == len(prepared.queries) and all(labels.get(r["id"]) == (r["answers"], r["gold_ids"])
            for r in prepared.queries), "database_labels")
    for name, path in prepared.files.items():
        if name.endswith(".npy") and not name.startswith("bm25/"):
            continue
        cursor.execute("SELECT sha256,bytes,payload FROM ragturk_prepared.files WHERE name=%s", (name,))
        row = cursor.fetchone()
        require(row is not None and row[0] == prepared.refs[name]["sha256"] and row[1] == path.stat().st_size and
                hashlib.sha256(bytes(row[2])).hexdigest() == row[0] and len(row[2]) == row[1], "database_file_bytes")
    prepared.stable_inputs()
    return {**prepared.summary(), "status": "database_complete_verified", "database_mutations": 0}


def import_database(connection, prepared: Prepared) -> dict:
    """One transaction, no drop/update of prior bundles, and complete only at commit."""
    prepared.stable_inputs()
    with connection.transaction():
        with connection.cursor() as cursor:
            cursor.execute("SELECT pg_advisory_xact_lock(74153037511)")
            cursor.execute("SELECT 1 FROM pg_namespace WHERE nspname=%s", (SCHEMA,))
            if cursor.fetchone() is not None:
                return verify_database(cursor, prepared)
            ddl(cursor)
            cursor.execute("INSERT INTO ragturk_prepared.bundle VALUES (1,%s,'prepared',%s::jsonb)",
                           (prepared.identity, json.dumps(prepared.manifest, allow_nan=False)))
            copy_rows(cursor, "chunks", chunk_rows(prepared))
            copy_rows(cursor, "questions", question_rows(prepared))
            copy_rows(cursor, "question_labels",
                          ((r["id"], json.dumps(r["answers"], ensure_ascii=False), json.dumps(r["gold_ids"])) for r in prepared.queries))
            for name, path in prepared.files.items():
                if name.endswith(".npy") and not name.startswith("bm25/"):
                    continue
                payload = path.read_bytes()
                require(hashlib.sha256(payload).hexdigest() == prepared.refs[name]["sha256"], "input_changed")
                cursor.execute("INSERT INTO ragturk_prepared.files VALUES (%s,%s,%s,%s)",
                               (name, prepared.refs[name]["sha256"], len(payload), payload))
            prepared.stable_inputs()
            cursor.execute("UPDATE ragturk_prepared.bundle SET state='complete' WHERE singleton=1 AND identity=%s AND state='prepared'", (prepared.identity,))
            result = verify_database(cursor, prepared)
    return {**result, "status": "database_import_committed", "database_mutations": 1}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "import", "verify"))
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--dsn-env", default="RAGTURK_PG_DSN")
    args = parser.parse_args(argv)
    try:
        prepared = validate_bundle(args.bundle)
        if args.command == "validate":
            result = prepared.summary()
        else:
            import psycopg  # Optional; never imported for CPU validation.
            dsn = os.environ.get(args.dsn_env)
            require(bool(dsn), "dsn_environment_required")
            with psycopg.connect(dsn) as connection:
                if args.command == "import":
                    result = import_database(connection, prepared)
                else:
                    with connection.transaction():
                        with connection.cursor() as cursor:
                            cursor.execute("SET TRANSACTION READ ONLY")
                            result = verify_database(cursor, prepared)
        print(json.dumps(result, sort_keys=True))
        return 0
    except Exception as error:
        # Driver messages can contain DSNs, queries or source data; never emit them.
        print(json.dumps({"status": "preparation_failed", "error_type": type(error).__name__, "new_model_calls": 0}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

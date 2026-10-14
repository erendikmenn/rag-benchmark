import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "pgprep", Path(__file__).parents[1] / "scripts/multimodal_pgvector_store.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def file_spec(p):
    return {"path": str(p), "size": p.stat().st_size, "sha256": m.digest(p)}


def fixture(root, kind="dense"):
    for name, value in [
        ("dataset.json", "{}"),
        ("corpus.jsonl", '{"id":"d"}\n'),
        ("queries.jsonl", '{"id":"q"}\n'),
        ("qrels.jsonl", '{"query_id":"q","corpus_id":"d","relevance":1}\n'),
    ]:
        (root / name).write_text(value)
    hashes = {n: m.digest(root / n) for n in ["dataset.json", "corpus.jsonl", "queries.jsonl", "qrels.jsonl"]}
    a = np.asarray([[1, 0], [0, 1]] if kind == "ragged" else [[1, 0]], dtype=np.float32)
    np.save(root / "v.npy", a)
    block = {**file_spec(root / "v.npy"), "shape": list(a.shape), "item_ids": ["d"]}
    c = {
        "dataset_root": str(root),
        "frozen_identity": m.seal({"files": hashes, "media": {}}),
        "source_files": [file_spec(root / n) for n in hashes],
        "media": [],
        "sets": [
            {
                "set_id": "native",
                "channel": "N",
                "role": "document",
                "kind": kind,
                "dimension": 2,
                "provenance": {"historical_config": "unattested"},
                "blocks": [block],
            }
        ],
    }
    return {"version": 1, "artifact_root": str(root), "collections": [c]}


@pytest.mark.parametrize("kind", ["dense", "ragged"])
def test_valid_no_search_or_model(tmp_path, kind):
    assert m.inspect_manifest(fixture(tmp_path, kind)) == tmp_path


@pytest.mark.parametrize(
    "mutation,code",
    [
        ("item", "source_order_binding"),
        ("shape", "shape_binding"),
        ("sha", "file_seal"),
        ("identity", "frozen_identity_mismatch"),
    ],
)
def test_binding_rejects_mutation(tmp_path, mutation, code):
    v = fixture(tmp_path)
    c = v["collections"][0]
    b = c["sets"][0]["blocks"][0]
    if mutation == "item":
        b["item_ids"] = ["q"]
    if mutation == "shape":
        b["shape"] = [3, 2]
    if mutation == "sha":
        b["sha256"] = "0" * 64
    if mutation == "identity":
        c["frozen_identity"] = "0" * 64
    with pytest.raises(ValueError, match=code):
        m.inspect_manifest(v)


@pytest.mark.parametrize("values,code", [([[float("nan"), 0]], "nonfinite"), ([[0, 0]], "normalization")])
def test_invalid_vectors(tmp_path, values, code):
    v = fixture(tmp_path)
    p = tmp_path / "v.npy"
    np.save(p, np.asarray(values, dtype=np.float32))
    v["collections"][0]["sets"][0]["blocks"][0].update(file_spec(p))
    with pytest.raises(ValueError, match=code):
        m.inspect_manifest(v)


def test_symlink_and_escape(tmp_path):
    fixture(tmp_path)
    (tmp_path / "link").symlink_to(tmp_path / "v.npy")
    with pytest.raises(ValueError, match="symlink"):
        m.resolve(tmp_path, "link")
    with pytest.raises(ValueError, match="escape"):
        m.resolve(tmp_path, "../outside")


@pytest.mark.parametrize("ragged", [False, True])
def test_readback_exact_fp32(tmp_path, ragged):
    a = np.asarray([[1, 0], [0, 1]] if ragged else [1, 0], dtype=np.float32)
    data = m.row_bytes(a)
    row = (
        "d",
        2,
        None if ragged else json.dumps(a.tolist()),
        data if ragged else None,
        2 if ragged else 1,
        __import__("hashlib").sha256(data).hexdigest(),
    )
    assert m.check_row(row, data, 2, 2 if ragged else 1) == "d"
    with pytest.raises(ValueError, match="vector_readback"):
        m.check_row(row, b"bad", 2, 2 if ragged else 1)


class BlobCursor:
    def __init__(self):
        self.size = None
        self.chunks = {}
        self.result = None

    def execute(self, sql, args):
        if sql.startswith("SELECT size"):
            self.result = None if self.size is None else (self.size,)
        elif sql.startswith("INSERT INTO multimodal_prepared.blobs"):
            self.size = args[1]
        elif sql.startswith("INSERT INTO multimodal_prepared.blob_chunks"):
            self.chunks[args[1]] = args[2]
        elif sql.startswith("SELECT data"):
            self.result = (self.chunks[args[1]],) if args[1] in self.chunks else None
        elif sql.startswith("SELECT count"):
            self.result = (len(self.chunks),)
        else:
            raise AssertionError(sql)

    def fetchone(self):
        return self.result


def test_chunked_blob_resume_and_corruption(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "CHUNK", 3)
    p = tmp_path / "blob"
    p.write_bytes(b"abcdefghij")
    f = file_spec(p)
    cur = BlobCursor()
    m.blob(cur, p, f, True)
    assert len(cur.chunks) == 4
    m.blob(cur, p, f, False)
    cur.chunks[1] = b"xxx"
    with pytest.raises(ValueError, match="blob_readback"):
        m.blob(cur, p, f, False)


def test_dsn_failure_sanitized(tmp_path, capsys):
    p = tmp_path / "dsn"
    p.write_text("secret-password")
    p.chmod(0o644)
    assert m.main(["verify", "--manifest", "missing", "--dsn-file", str(p)]) == 1
    out = capsys.readouterr().out
    assert "secret-password" not in out and "ValueError" in out


@pytest.mark.parametrize("bad", [None, "gap", "source", "order"])
def test_segment_coverage(tmp_path, bad):
    v = fixture(tmp_path)
    c = v["collections"][0]
    (tmp_path / "corpus.jsonl").write_text('{"id":"d","text":"abcd"}\n')
    c["source_files"] = [
        file_spec(tmp_path / n) for n in ["dataset.json", "corpus.jsonl", "queries.jsonl", "qrels.jsonl"]
    ]
    c["frozen_identity"] = m.seal(
        {"files": {Path(x["path"]).name: x["sha256"] for x in c["source_files"]}, "media": {}}
    )
    a = np.asarray([[1, 0], [0, 1]], dtype=np.float32)
    np.save(tmp_path / "v.npy", a)
    s = c["sets"][0]
    s["blocks"] = [
        {
            **file_spec(tmp_path / "v.npy"),
            "shape": [2, 2],
            "item_ids": ["d:0", "d:1"],
            "source_item_ids": ["d", "d"],
        }
    ]
    s["source_segments"] = [
        {
            "source_item_id": "d",
            "source_length": 4,
            "source_sha256": __import__("hashlib").sha256(b"abcd").hexdigest(),
            "chunks": [{"item_id": "d:0", "start": 0, "end": 2}, {"item_id": "d:1", "start": 2, "end": 4}],
        }
    ]
    if bad == "gap":
        s["source_segments"][0]["chunks"][1]["start"] = 3
    if bad == "source":
        s["source_segments"][0]["source_sha256"] = "0" * 64
    if bad == "order":
        s["blocks"][0]["item_ids"].reverse()
    if bad:
        with pytest.raises(ValueError):
            m.inspect_manifest(v)
    else:
        m.inspect_manifest(v)


@pytest.mark.parametrize("state", [("wrong", "complete"), ("expected", "importing"), ("expected", "failed")])
def test_resume_rejects_mutation_or_incomplete(state):
    with pytest.raises(ValueError, match="set_collision_or_incomplete"):
        m.validate_set_state(state, "expected")


def test_resume_requires_exact_complete():
    m.validate_set_state(("expected", "complete"), "expected")


def test_collection_allows_new_sets_but_rejects_source_mutation(tmp_path):
    c = fixture(tmp_path)["collections"][0]
    binding = m.seal(m.collection_source(c))
    c["sets"].append({"set_id": "future_model"})
    c["lexical_artifacts"] = [{"path": "new-lexical"}]
    assert m.seal(m.collection_source(c)) == binding
    c["source_files"][0]["sha256"] = "0" * 64
    assert m.seal(m.collection_source(c)) != binding


class VectorCursor:
    def __init__(self):
        self.rows = []
        self.inserts = 0
        self.selects = 0

    def executemany(self, sql, params):
        self.inserts += 1
        self.rows = [(p[2], p[3], p[4], p[5], p[6], p[7], p[8]) for p in params]

    def execute(self, sql, args):
        assert sql.startswith("SELECT ordinal")
        self.selects += 1

    def fetchall(self):
        return self.rows


@pytest.mark.parametrize("kind", ["dense", "ragged"])
def test_batch_insert_and_complete_readback(kind):
    cur = VectorCursor()
    s = {"kind": kind, "dimension": 2}
    values = np.asarray([[1, 0], [0, 1]], dtype=np.float32)
    items = ["a", "b"] if kind == "dense" else ["a"]
    assert m.vector_batch(cur, ("collection", "set"), s, 7, items, values, True) == len(items)
    assert cur.inserts == 1 and cur.selects == 1
    assert m.vector_batch(cur, ("collection", "set"), s, 7, items, values, False) == len(items)
    assert cur.inserts == 1 and cur.selects == 2
    cur.rows[0] = (6, *cur.rows[0][1:])
    with pytest.raises(ValueError, match="stored_ordinal"):
        m.vector_batch(cur, ("collection", "set"), s, 7, items, values, False)


def test_stored_provenance_json_must_match():
    manifest = {"model": "pinned"}
    binding = m.seal(manifest)
    m.validate_set_state((binding, "complete", manifest), binding, manifest)
    with pytest.raises(ValueError, match="stored_manifest_mutation"):
        m.validate_set_state((binding, "complete", {"model": "tampered"}), binding, manifest)

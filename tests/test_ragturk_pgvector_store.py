"""Synthetic CPU fixtures only: no PostgreSQL, models or benchmark runs."""
from contextlib import contextmanager
import copy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("ragturk_pgvector_store", Path(__file__).parents[1] / "scripts/ragturk_pgvector_store.py")
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)
COUNTS = {"corpus": 2, "dev": 1, "test": 1}


@pytest.fixture
def bundle(tmp_path):
    corpus = [{"id": "a", "text": "fixture one", "title": "", "article_id": "article1", "source": "web"},
              {"id": "b", "text": "fixture two", "title": "title", "article_id": "article2", "source": "wikipedia"}]
    queries = [{"id": "q1", "question": "fixture question", "article_id": "article1", "source": "web", "category": "FACTUAL", "answers": ["fixture reference"], "gold_ids": ["a"]},
               {"id": "q2", "question": "fixture second", "article_id": "article2", "source": "wikipedia", "category": "INTERPRETATION", "answers": ["fixture second reference"], "gold_ids": ["b"]}]
    def dump(name, value, jsonl=False):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(json.dumps(v) for v in value) + "\n" if jsonl else json.dumps(value))
        return path
    def ref(path):
        return {"path": str(path), "sha256": m.file_sha(path), "bytes": path.stat().st_size}
    paths = {"corpus": dump("corpus.jsonl", corpus, True), "questions_dev": dump("questions.dev.jsonl", queries[:1], True), "questions_test": dump("questions.test.jsonl", queries[1:], True)}
    source = {"dataset": "metunlp/ragturk", "revision": "a" * 40, "counts": {"corpus_chunks": 2, "dev_questions": 1, "test_questions": 1}, "file_sha256": {p.name: m.file_sha(p) for p in paths.values()}}
    paths["manifest"] = dump("manifest.json", source)
    order = m.canonical_sha(["a", "b"])
    qorder = m.canonical_sha(["q1", "q2"])
    dataset = {"id": "metunlp/ragturk", "revision": "a" * 40, "corpus_order_sha256": order, "query_order_sha256": qorder, **{k: ref(p) for k, p in paths.items()}}
    embeddings = {}
    for name, dimension in m.DIMENSIONS.items():
        arrays = {}
        for role in ("documents", "queries"):
            path = tmp_path / (name + "-" + role + ".npy")
            values = np.zeros((2, dimension), dtype=np.float32)
            values[0, 0] = 1
            values[1, 1] = 1
            np.save(path, values)
            arrays[role] = ref(path)
        embeddings[name] = {"dimension": dimension, "corpus_order_sha256": order, "query_order_sha256": qorder,
                            "model_provenance": {"model_id": m.MODEL_IDS[name], "revision": "b" * 40,
                                                 "device": "mps", "dtype": "float32", "package_versions": {"numpy": "fixture"},
                                                 "docs_reused": True, "historical": {"producer_identity": "fixture"}, "fresh_queries": {"device": "cpu"}}, **arrays}
    bm = tmp_path / "bm25"
    bm.mkdir()
    dump("bm25/ready.json", {"count": 2, "corpus_hash": m.canonical_sha([{k: row[k] for k in ("id", "title", "text")} for row in corpus])})
    dump("bm25/params.index.json", {"num_docs": 2, "method": "lucene", "k1": 1.5, "b": 0.75})
    dump("bm25/vocab.index.json", {"fixture": 0, "second": 1})
    for name, values, dtype in [("data", [1, 2], "float32"), ("indices", [0, 1], "int32"), ("indptr", [0, 1, 2], "int32")]:
        np.save(bm / (name + ".csc.index.npy"), np.array(values, dtype=dtype))
    manifest = {"schema_version": 1, "dataset": dataset, "embeddings": embeddings,
                "bm25": {"identity": "c" * 64, "corpus_order_sha256": order, "producer_provenance": {"version": "fixture"}, "files": [ref(p) for p in sorted(bm.iterdir())]}}
    manifest["identity"] = m.canonical_sha(manifest)
    path = dump("bundle.json", manifest)
    return path


def mutate(bundle, fn):
    value = json.loads(bundle.read_text())
    fn(value)
    value["identity"] = m.canonical_sha({k: v for k, v in value.items() if k != "identity"})
    bundle.write_text(json.dumps(value))


def test_valid_all_splits_dimensions_no_inference(bundle):
    prepared = m.validate_bundle(bundle, counts=COUNTS)
    assert prepared.splits == ["dev", "test"]
    assert prepared.summary()["questions"] == 2
    assert len(prepared.files) == 14
    assert all(x.shape == (2, m.DIMENSIONS[name]) for name, roles in prepared.vectors.items() for x in roles.values())
    assert "reference_answers" not in str(list(m.question_rows(prepared))[0])


@pytest.mark.parametrize("change", ["shape", "float64", "nan", "unnormalized", "order", "filehash", "model", "duplicatebm"])
def test_integrity_rejections(bundle, change):
    def edit(value):
        ref = value["embeddings"]["bge"]["queries"]
        if change in {"shape", "float64", "nan", "unnormalized"}:
            path = Path(ref["path"])
            a = np.load(path)
            if change == "shape":
                a = a[:, :-1]
            elif change == "float64":
                a = a.astype(np.float64)
            elif change == "nan":
                a[0, 0] = np.nan
            else:
                a[0, 0] = .5
            np.save(path, a)
            ref.update(sha256=m.file_sha(path), bytes=path.stat().st_size)
        elif change == "order":
            value["dataset"]["query_order_sha256"] = m.canonical_sha(["q2", "q1"])
        elif change == "filehash":
            ref["sha256"] = "0" * 64
        elif change == "model":
            value["embeddings"]["bge"]["model_provenance"]["model_id"] = "other"
        else:
            value["bm25"]["files"][0] = dict(value["bm25"]["files"][1])
    mutate(bundle, edit)
    with pytest.raises(ValueError):
        m.validate_bundle(bundle, counts=COUNTS)


def test_source_changed_after_validation(bundle):
    prepared = m.validate_bundle(bundle, counts=COUNTS)
    prepared.files["corpus.jsonl"].write_text("changed")
    with pytest.raises(ValueError, match="input_changed"):
        prepared.stable_inputs()


def test_symlink_ref_rejected(bundle, tmp_path):
    target = Path(json.loads(bundle.read_text())["dataset"]["corpus"]["path"])
    link = tmp_path / "linked"
    link.symlink_to(target)
    mutate(bundle, lambda value: value["dataset"]["corpus"].update(path=str(link)))
    with pytest.raises(ValueError, match="unsafe_path"):
        m.validate_bundle(bundle, counts=COUNTS)


class FakeDB:
    def __init__(self, fail_copy=False):
        self.schema = False
        self.tables = {}
        self.sql = []
        self.fail_copy = fail_copy
        self.commits = 0
        self.rollbacks = 0

    @contextmanager
    def transaction(self):
        snapshot = self.schema, copy.deepcopy(self.tables)
        try:
            yield
        except Exception:
            self.schema, self.tables = snapshot
            self.rollbacks += 1
            raise
        else:
            self.commits += 1

    @contextmanager
    def cursor(self):
        yield FakeCursor(self)


class FakeCursor:
    def __init__(self, db):
        self.db = db
        self.result = []

    def execute(self, sql, params=None):
        self.db.sql.append(sql)
        self.result = []
        if sql.startswith("SELECT 1 FROM pg_namespace"):
            self.result = [(1,)] if self.db.schema else []
        elif sql.startswith("SELECT 1 FROM pg_extension"):
            self.result = [(1,)]
        elif sql.startswith("CREATE SCHEMA"):
            self.db.schema = True
        elif sql.startswith("CREATE TABLE"):
            table = sql.split()[2].split(".")[1]
            self.db.tables[table] = []
        elif sql.startswith("INSERT INTO ragturk_prepared.bundle"):
            self.db.tables["bundle"] = [(params[0], "prepared", json.loads(params[1]))]
        elif sql.startswith("UPDATE ragturk_prepared.bundle"):
            identity, _, manifest = self.db.tables["bundle"][0]
            self.db.tables["bundle"][0] = (identity, "complete", manifest)
        elif sql.startswith("INSERT INTO ragturk_prepared.files"):
            self.db.tables["files"].append(params)
        elif sql.startswith("SELECT identity,state,manifest"):
            self.result = list(self.db.tables.get("bundle", []))
        elif sql.startswith("SELECT count(*)"):
            self.result = [(len(self.db.tables[sql.rsplit(".", 1)[1]]),)]
        elif "FROM ragturk_prepared.chunks WHERE" in sql or "FROM ragturk_prepared.questions WHERE" in sql:
            table = "chunks" if ".chunks " in sql else "questions"
            self.result = sorted([r for r in self.db.tables[table] if r[1] >= params[0]], key=lambda r: r[1])[:128]
        elif sql.startswith("SELECT question_id,"):
            self.result = [(r[0], json.loads(r[1]), json.loads(r[2])) for r in self.db.tables["question_labels"]]
        elif sql.startswith("SELECT sha256,bytes,payload"):
            self.result = [r[1:] for r in self.db.tables["files"] if r[0] == params[0]]
        elif not sql.startswith("SELECT pg_advisory_xact_lock"):
            raise AssertionError("Unexpected SQL statement")

    def fetchone(self):
        return self.result[0] if self.result else None

    def fetchall(self):
        return self.result

    @contextmanager
    def copy(self, sql):
        table = sql.split()[1].split(".")[1]
        self.db.sql.append(sql)
        cursor = self
        class Writer:
            def write_row(self, row):
                if cursor.db.fail_copy:
                    raise OSError("injected copy failure")
                cursor.db.tables[table].append(row)
        yield Writer()


def test_atomic_import_native_bm25_and_readonly_reuse(bundle):
    prepared = m.validate_bundle(bundle, counts=COUNTS)
    db = FakeDB()
    result = m.import_database(db, prepared)
    assert result["status"] == "database_import_committed" and db.commits == 1
    assert db.tables["bundle"][0][1] == "complete"
    assert len(db.tables["files"]) == 10
    assert sum(name.startswith("bm25/") for name, *_ in db.tables["files"]) == 6
    before = copy.deepcopy(db.tables)
    db.sql.clear()
    assert m.import_database(db, prepared)["database_mutations"] == 0
    assert db.tables == before and all(sql.startswith("SELECT") for sql in db.sql)


def test_copy_failure_rolls_back_schema_and_all_rows(bundle):
    prepared = m.validate_bundle(bundle, counts=COUNTS)
    db = FakeDB(fail_copy=True)
    with pytest.raises(OSError):
        m.import_database(db, prepared)
    assert not db.schema and db.tables == {} and db.commits == 0 and db.rollbacks == 1


@pytest.mark.parametrize("tamper", ["identity", "state", "vector", "bytea", "labels", "missingrow"])
def test_existing_corrupt_complete_rejected_no_overwrite(bundle, tamper):
    prepared = m.validate_bundle(bundle, counts=COUNTS)
    db = FakeDB()
    m.import_database(db, prepared)
    if tamper in {"identity", "state"}:
        identity, state, manifest = db.tables["bundle"][0]
        db.tables["bundle"][0] = ("wrong" if tamper == "identity" else identity,
                                  "prepared" if tamper == "state" else state, manifest)
    elif tamper == "vector":
        row = db.tables["chunks"][0]
        db.tables["chunks"][0] = (*row[:-2], "[0]", row[-1])
    elif tamper == "bytea":
        row = db.tables["files"][0]
        db.tables["files"][0] = (*row[:-1], b"bad")
    elif tamper == "labels":
        row = db.tables["question_labels"][0]
        db.tables["question_labels"][0] = (row[0], '["wrong"]', row[2])
    else:
        db.tables["questions"].pop()
    before = copy.deepcopy(db.tables)
    db.sql.clear()
    with pytest.raises(ValueError):
        m.import_database(db, prepared)
    assert db.tables == before and all(sql.startswith("SELECT") for sql in db.sql)

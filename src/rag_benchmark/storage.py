"""Transactional progress and content-addressed answer reuse."""

import hashlib
import json
from pathlib import Path
import sqlite3


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS results (variant TEXT, query_id TEXT, payload TEXT, PRIMARY KEY(variant,query_id))")
        self.db.execute("CREATE TABLE IF NOT EXISTS answers_v2 (key TEXT PRIMARY KEY, payload TEXT)")

    def get(self, variant, query_id):
        row = self.db.execute("SELECT payload FROM results WHERE variant=? AND query_id=?", (variant, query_id)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, row):
        with self.db:
            self.db.execute("INSERT INTO results VALUES (?,?,?)", (row["variant"], row["query_id"], json.dumps(row, ensure_ascii=False)))

    def cached_answer(self, key):
        row = self.db.execute("SELECT payload FROM answers_v2 WHERE key=?", (key,)).fetchone()
        return None if row is None else json.loads(row[0])

    def save_answer(self, key, answer, usage):
        # Keep finish reasons/token counts but never replay an old latency as new.
        usage = {k: v for k, v in usage.items() if k not in ("seconds", "timings")}
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO answers_v2 VALUES (?,?)", (key, json.dumps({"answer": answer, "usage": usage})))

    def rows(self):
        return [json.loads(r[0]) for r in self.db.execute("SELECT payload FROM results ORDER BY variant,query_id")]

    def close(self):
        self.db.close()

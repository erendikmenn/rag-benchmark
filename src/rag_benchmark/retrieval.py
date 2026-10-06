"""BM25, exact normalized dense search, and rank-based fusion."""
from __future__ import annotations

import json
import re
import unicodedata
import uuid
from pathlib import Path

import numpy as np

from .models import DenseEmbedder, stable_hash, package_versions


METHODS = {
    "bm25": ("bm25",), "bge": ("bge",), "embeddinggemma": ("embeddinggemma",),
    "bm25_bge": ("bm25", "bge"), "bm25_embeddinggemma": ("bm25", "embeddinggemma"),
    "bge_embeddinggemma": ("bge", "embeddinggemma"),
    "bm25_bge_embeddinggemma": ("bm25", "bge", "embeddinggemma"),
}


def turkish_tokens(text: str) -> list[str]:
    """Preserve accents and single-character terms; respect Turkish dotted I."""
    text = unicodedata.normalize("NFC", text).translate(str.maketrans({"I": "ı", "İ": "i"})).lower()
    return re.findall(r"\w+", text, flags=re.UNICODE)


def reciprocal_rank_fusion(rankings: list[list[dict]], k: int = 60, top_k: int | None = None) -> list[dict]:
    if k < 0:
        raise ValueError("RRF k must be non-negative.")
    documents, scores, first = {}, {}, {}
    for source, ranking in enumerate(rankings):
        seen = set()
        for rank, document in enumerate(ranking, 1):
            identifier = str(document["id"])
            if identifier in seen:
                continue
            seen.add(identifier)
            documents.setdefault(identifier, document)
            first.setdefault(identifier, (source, rank, identifier))
            scores[identifier] = scores.get(identifier, 0.0) + 1.0 / (k + rank)
    ordered = sorted(scores, key=lambda identifier: (-scores[identifier], first[identifier]))
    if top_k is not None:
        ordered = ordered[:top_k]
    return [{**documents[identifier], "score": scores[identifier], "rank": rank} for rank, identifier in enumerate(ordered, 1)]


def _atomic_array(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class RetrievalIndex:
    def __init__(self, corpus: list[dict], cache_dir: Path, embeddings: dict[str, dict] | None = None,
                 *, rrf_k: int = 60, fusion_pool: int = 100, bm25_config: dict | None = None):
        if not corpus:
            raise ValueError("Cannot build retrieval on an empty corpus.")
        self.corpus = []
        seen = set()
        for document in corpus:
            identifier = str(document["id"])
            if identifier in seen:
                raise ValueError(f"Duplicate corpus id: {identifier}")
            if not isinstance(document.get("text"), str) or not document["text"].strip():
                raise ValueError(f"Missing text for corpus id {identifier}")
            seen.add(identifier)
            self.corpus.append({**document, "id": identifier})
        if rrf_k < 0 or fusion_pool < 1:
            raise ValueError("Invalid RRF or fusion pool configuration.")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings = embeddings or {}
        self.rrf_k, self.fusion_pool = rrf_k, fusion_pool
        self.bm25_config = {"k1": 1.5, "b": 0.75, "method": "lucene", "csc_backend": "numpy", **(bm25_config or {})}
        self.corpus_hash = stable_hash([{"id": d["id"], "title": d.get("title", ""), "text": d["text"]} for d in self.corpus])
        self._embedders: dict[str, DenseEmbedder] = {}
        self._vectors: dict[str, np.ndarray] = {}
        self._bm25 = None
        self.last_usage: dict = {}

    def _embedder(self, name: str) -> DenseEmbedder:
        if name not in self._embedders:
            config = self.embeddings.get(name, {})
            self._embedders[name] = DenseEmbedder(name, config)
        return self._embedders[name]

    def identity_for(self, method: str) -> str:
        if method not in METHODS:
            raise ValueError(f"Unknown retrieval method {method!r}; expected one of {list(METHODS)}")
        identities = {name: self._embedder(name).identity if name != "bm25" else {
            "version": "turkish-unicode-v1", "settings": self.bm25_config, "packages": package_versions(("bm25s",))
        } for name in METHODS[method]}
        return stable_hash({"corpus": self.corpus_hash, "method": method, "models": identities,
            "rrf_k": self.rrf_k, "fusion_pool": self.fusion_pool, "search": "exact-cosine-v1"})

    @property
    def identity(self) -> str:
        return stable_hash({"corpus": self.corpus_hash, "embeddings": self.embeddings, "bm25": self.bm25_config,
            "rrf_k": self.rrf_k, "fusion_pool": self.fusion_pool})

    def build(self, method: str = "bm25") -> None:
        if method not in METHODS:
            raise ValueError(f"Unknown retrieval method: {method}")
        for name in METHODS[method]:
            if name == "bm25":
                self._build_bm25()
            else:
                self._build_dense(name)

    def _build_bm25(self) -> None:
        if self._bm25 is not None:
            return
        import bm25s
        path = self.cache_dir / "bm25" / self.identity_for("bm25")
        if (path / "ready.json").is_file():
            self._bm25 = bm25s.BM25.load(str(path), load_corpus=False, mmap=True)
            return
        tokens = [turkish_tokens((str(d.get("title") or "") + "\n" + d["text"]).strip()) for d in self.corpus]
        if not any(tokens):
            raise ValueError("The corpus contains no BM25 tokens.")
        self._bm25 = bm25s.BM25(**self.bm25_config)
        self._bm25.index(tokens, show_progress=False)
        path.mkdir(parents=True, exist_ok=True)
        self._bm25.save(str(path))
        (path / "ready.json").write_text(json.dumps({"corpus_hash": self.corpus_hash, "count": len(self.corpus)}))

    def _build_dense(self, name: str) -> None:
        if name in self._vectors:
            return
        embedder = self._embedder(name)
        path = self.cache_dir / "dense" / name / stable_hash({"corpus": self.corpus_hash, "embedder": embedder.identity}) / "documents.npy"
        if not path.is_file():
            # Completed blocks survive interruption; an incomplete final file is never published.
            path.parent.mkdir(parents=True, exist_ok=True)
            block_size = 256
            blocks = []
            for start in range(0, len(self.corpus), block_size):
                stop = min(start + block_size, len(self.corpus))
                block_path = path.parent / "blocks" / f"{start:09d}-{stop:09d}.npy"
                if not block_path.is_file():
                    _atomic_array(block_path, embedder.embed_documents(self.corpus[start:stop]))
                block = np.load(block_path, mmap_mode="r", allow_pickle=False)
                if block.shape != (stop - start, embedder.dimension) or not np.isfinite(block).all():
                    raise RuntimeError(f"Invalid embedding checkpoint block: {block_path}")
                blocks.append((start, stop, block_path))
            temporary = path.with_name("documents." + uuid.uuid4().hex + ".tmp.npy")
            try:
                merged = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=(len(self.corpus), embedder.dimension))
                for start, stop, block_path in blocks:
                    merged[start:stop] = np.load(block_path, mmap_mode="r", allow_pickle=False)
                merged.flush()
                del merged
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        if values.shape != (len(self.corpus), embedder.dimension):
            raise RuntimeError(f"Corrupt dense cache: expected {(len(self.corpus), embedder.dimension)}, got {values.shape}.")
        self._vectors[name] = values

    def _query_vector(self, name: str, question: str, cache_query: bool) -> tuple[np.ndarray, bool]:
        embedder = self._embedder(name)
        path = self.cache_dir / "queries" / name / embedder.identity / (stable_hash(question) + ".npy")
        if cache_query and path.is_file():
            value = np.load(path, allow_pickle=False)
            cached = True
        else:
            value = embedder.embed_query(question)
            cached = False
            if cache_query:
                _atomic_array(path, value)
        if value.shape != (embedder.dimension,) or not np.isfinite(value).all():
            raise RuntimeError("Invalid cached query embedding.")
        return value, cached

    def _rank_scores(self, scores: np.ndarray, top_k: int, *, positive_only: bool = False) -> list[dict]:
        scores = np.asarray(scores).reshape(-1)
        if len(scores) != len(self.corpus) or not np.isfinite(scores).all():
            raise RuntimeError("Retriever returned invalid scores.")
        indices = np.argsort(-scores, kind="stable")[:top_k]
        if positive_only:
            indices = indices[scores[indices] > 0]
        return [{**self.corpus[int(index)], "score": float(scores[index]), "rank": rank}
                for rank, index in enumerate(indices, 1)]

    def search(self, question: str, method: str, top_k: int = 50, *, cache_query: bool = True) -> list[dict]:
        if method not in METHODS:
            raise ValueError(f"Unknown retrieval method: {method}")
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer.")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("A non-empty question is required.")
        self.build(method)
        components = METHODS[method]
        pool = min(len(self.corpus), max(top_k, self.fusion_pool) if len(components) > 1 else top_k)
        rankings, cached_queries = [], {}
        for name in components:
            if name == "bm25":
                scores = self._bm25.get_scores(turkish_tokens(question))
                ranking = self._rank_scores(scores, pool, positive_only=True)
            else:
                query, cached_queries[name] = self._query_vector(name, question, cache_query)
                scores = self._vectors[name] @ query
                ranking = self._rank_scores(scores, pool)
            rankings.append(ranking)
        self.last_usage = {"query_cache_hits": cached_queries, "method": method, "candidate_pool": pool,
            "search": "exact", "corpus_size": len(self.corpus)}
        if len(rankings) == 1:
            return rankings[0][:top_k]
        return reciprocal_rank_fusion(rankings, k=self.rrf_k, top_k=top_k)


# Short alias for callers that describe the fusion explicitly.
rrf = reciprocal_rank_fusion

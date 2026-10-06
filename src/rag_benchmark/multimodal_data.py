"""Pinned full-gallery multimodal dataset preparation (no model inference).

Gold captions/transcripts/answers are exclusively queries or evaluation metadata.
Raw downloads and media stay under ignored data/. Each import fails closed on
missing media/counts, writes SHA-256 inventories, and can be validated offline.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import tarfile
import tokenize
import unicodedata
import urllib.parse
import urllib.request
import zipfile
import warnings
from functools import lru_cache
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Iterable

XM3600_REVISION = "ad44c380c86cb1240666361caabaf5684507d0ef"
XM3600_SHA256 = {'captions.zip': '7964f969d39319a96769499b199042015754f8d5607b6aad2d1c2c2e55f5e562', 'images.tgz': 'ba57c58d7f7a35d95315157b609ca280d55ba10a107026f67d2e98371e221188', 'image_attributions.csv': '288de413d25e9c70dadfd42ee59256aa1082673e615d0eda02c3875fe2b36ecd'}
FLEURS_REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
VIDORE_REVISIONS = {
    "computer_science": "d5cc75883d92e294f0c0fc2662551c9708a06ebc",
    "hr": "0cdf0979f2c5a0fd3e335e6373b9da48a9fe3bc3",
    "finance_en": "7f432c176d82e27546501ad8064a713ac3071809",
    "industrial": "e26c864724f5dd71a3d7d739272d95637764cee9",
    "pharmaceuticals": "3abd4aa8a9445fb5538a78a19ba50bd57bd22b5c",
    "energy": "caec06d3c73434d635f710f93bcd898331c59f20",
    "physics": "a0de276f515acc044b72cae8de53a44bb5a8f1f5",
    "finance_fr": "1d808daa08032ffecdf62da151a7f7a8fe2bd0c9",
}
VIDORE_COLLECTIONS = ("computer_science", "hr", "finance_en", "industrial", "pharmaceuticals", "energy", "physics", "finance_fr")
# Public release counts, not a license to truncate or synthesize missing examples.
VIDORE_COUNTS = {"computer_science": (1360, 215), "hr": (1110, 318), "finance_en": (2942, 309),
                 "industrial": (5244, 283), "pharmaceuticals": (2313, 364), "energy": (2229, 308),
                 "physics": (1674, 302), "finance_fr": (2384, 320)}


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _relative(value: str) -> Path:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError(f"Unsafe relative path: {value}")
    return Path(*path.parts)


def _download(url: str, path: Path, expected_sha256: str | None = None) -> dict:
    """Atomically cache a source and pin its first observed checksum on disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sidecar = path.with_name(path.name + ".source.json")
    prior = json.loads(sidecar.read_text()) if sidecar.exists() else None
    if prior and prior["url"] != url:
        raise ValueError(f"Source changed for cache {path}")
    if not path.exists():
        temporary = path.with_name(path.name + ".partial")
        request = urllib.request.Request(url, headers={"User-Agent": "rag-benchmark-dataset-preparation/1"})
        with urllib.request.urlopen(request, timeout=180) as response, temporary.open("wb") as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)
        temporary.replace(path)
    digest = sha256(path)
    expected = expected_sha256 or (prior or {}).get("sha256")
    if expected and digest != expected:
        raise ValueError(f"Source checksum mismatch: {path}")
    result = {"url": url, "sha256": digest, "size_bytes": path.stat().st_size, "file": path.name}
    _json(sidecar, result)
    return result


def _extract_tar(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as source:
        for member in source:
            relative = _relative(member.name)
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError(f"Unsupported archive member {member.name}")
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.stat().st_size == member.size:
                continue
            with source.extractfile(member) as handle, target.open("wb") as output:
                shutil.copyfileobj(handle, output)


def _link_asset(source: Path, target: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"Required media missing: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        try:
            os.link(source, target)
        except OSError:
            shutil.copyfile(source, target)
    elif sha256(target) != sha256(source):
        raise ValueError(f"Existing asset differs from source: {target}")


def _provenance(source: str) -> dict:
    return {"source": source, "query_independent": True, "gold_fields_used": False}


def write_dataset(destination: Path, *, dataset_id: str, track: str, revision: str,
                  corpus: list[dict], queries: list[dict], qrels: list[dict], sources: list[dict],
                  license: str, text_source: str, expected_counts: dict | None = None,
                  metadata: dict | None = None) -> dict:
    """Validate referential integrity before publishing the ready manifest."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    _json(destination / "dataset.json", {"id": dataset_id, "track": track, "revision": revision, "status": "preparing"})
    for name, rows in (("corpus", corpus), ("queries", queries), ("qrels", qrels)):
        _jsonl(destination / f"{name}.jsonl", rows)
    assets = {}
    for row in corpus + queries:
        for kind, filename in row.get("media", {}).items():
            if kind not in {"image", "audio", "video"}:
                raise ValueError(f"Unsupported media kind {kind}")
            path = destination / _relative(filename)
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f"Required media missing or empty: {path}")
            assets[filename] = {"path": filename, "sha256": sha256(path), "size_bytes": path.stat().st_size}
    _jsonl(destination / "assets.jsonl", [assets[key] for key in sorted(assets)])
    counts = {"corpus": len(corpus), "queries": len(queries), "qrels": len(qrels), "media_files": len(assets)}
    for key, expected in (expected_counts or {}).items():
        if counts[key] != expected:
            raise ValueError(f"Full dataset count mismatch for {key}: {counts[key]} != {expected}")
    result = {"schema_version": 1, "id": dataset_id, "name": dataset_id, "track": track,
              "revision": revision, "status": "ready", "full_gallery": True, "counts": counts,
              "expected_counts": expected_counts or {}, "sources": sources, "license": license,
              "text_provenance": _provenance(text_source),
              "text_ready": all(isinstance(row.get("text"), str) for row in corpus),
              "text_nonempty_count": sum(bool(row.get("text", "").strip()) for row in corpus),
              "text_empty_count": sum(row.get("text") == "" for row in corpus),
              "files": {name: {"sha256": sha256(destination / name), "size_bytes": (destination / name).stat().st_size}
                        for name in ("corpus.jsonl", "queries.jsonl", "qrels.jsonl", "assets.jsonl",
                                     "documents_metadata.json", "evaluation_transcripts.jsonl")
                        if (destination / name).is_file()},
              "split": (metadata or {}).get("split", "evaluation"), "metadata": metadata or {}}
    _json(destination / "dataset.json", result)
    try:
        validate_dataset(destination)
    except Exception as error:
        result["status"] = "failed"
        result["error"] = str(error)
        _json(destination / "dataset.json", result)
        raise
    return result


def validate_dataset(destination: Path, verify_assets: bool = True) -> dict:
    destination = Path(destination)
    manifest = json.loads((destination / "dataset.json").read_text())
    if manifest["status"] != "ready":
        raise ValueError("Dataset status is not ready")
    for name, expected in manifest["files"].items():
        if sha256(destination / _relative(name)) != expected["sha256"]:
            raise ValueError(f"Manifest checksum mismatch: {name}")
    corpus, queries, qrels = [read_jsonl(destination / f"{name}.jsonl") for name in ("corpus", "queries", "qrels")]
    for name, rows in (("corpus", corpus), ("queries", queries)):
        ids = [row["id"] for row in rows]
        if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
            raise ValueError(f"Invalid or duplicate {name} IDs")
        if manifest["counts"][name] != len(rows):
            raise ValueError(f"Manifest count mismatch: {name}")
    corpus_ids, query_ids = {row["id"] for row in corpus}, {row["id"] for row in queries}
    positives, pairs = set(), set()
    for row in qrels:
        pair = (row["query_id"], row["corpus_id"])
        if pair[0] not in query_ids or pair[1] not in corpus_ids or pair in pairs:
            raise ValueError(f"Invalid or duplicate qrel: {pair}")
        if not math.isfinite(row["relevance"]) or row["relevance"] < 0:
            raise ValueError("Invalid relevance")
        pairs.add(pair)
        if row["relevance"] > 0:
            positives.add(pair[0])
    if positives != query_ids:
        raise ValueError("Every query must have at least one positive in the complete gallery")
    forbidden = {"gold_caption", "gold_transcript", "gold_docstring", "gold_answer", "reference_caption"}
    for row in corpus:
        if forbidden & row.keys():
            raise ValueError("Gold fields cannot be in corpus records")
        provenance = row.get("metadata", {}).get("text_provenance", manifest["text_provenance"])
        if row.get("text") and (provenance.get("gold_fields_used") is not False or
                                provenance.get("query_independent") is not True):
            raise ValueError("Candidate text lacks leakage-safe provenance")
    if manifest["counts"]["qrels"] != len(qrels):
        raise ValueError("Manifest count mismatch: qrels")
    inventory = {row["path"]: row for row in read_jsonl(destination / "assets.jsonl")}
    referenced = {filename for row in corpus + queries for filename in row.get("media", {}).values()}
    if referenced != inventory.keys():
        raise ValueError("Media inventory does not match referenced assets")
    if manifest["counts"]["media_files"] != len(inventory):
        raise ValueError("Manifest count mismatch: media_files")
    for filename, entry in inventory.items():
        path = destination / _relative(filename)
        if not path.is_file() or (verify_assets and sha256(path) != entry["sha256"]):
            raise ValueError(f"Missing or modified media: {filename}")
    return manifest


def prepare_xm3600(root: Path, languages: tuple[str, ...] = ("tr", "en")) -> list[dict]:
    root = Path(root)
    raw = root / "raw" / "xm3600"
    urls = {"captions.zip": "https://google.github.io/crossmodal-3600/web-data/captions.zip",
            "images.tgz": "https://open-images-dataset.s3.amazonaws.com/crossmodal-3600/images.tgz",
            "image_attributions.csv": "https://google.github.io/crossmodal-3600/web-data/image_attributions.csv"}
    with ThreadPoolExecutor(max_workers=3) as pool:
        sources = list(pool.map(lambda item: _download(item[1], raw / item[0], XM3600_SHA256[item[0]]), urls.items()))
    with zipfile.ZipFile(raw / "captions.zip") as archive:
        rows = [json.loads(line) for line in archive.read("captions.jsonl").splitlines()]
    _extract_tar(raw / "images.tgz", raw / "extracted")
    image_files = {p.stem: p for p in (raw / "extracted").rglob("*.jpg")}
    results = []
    for language in languages:
        destination = root / f"xm3600-{language}"
        corpus, queries, qrels = [], [], []
        for row in rows:
            identifier = row["image/key"]
            filename = f"media/{identifier}.jpg"
            _link_asset(image_files[identifier], destination / filename)
            corpus.append({"id": identifier, "media": {"image": filename},
                           "metadata": {"image_locale": row["image/locale"], "text_status": "requires_generation"}})
            for index, caption in enumerate(row[language]["caption"]):
                query_id = f"{language}:{identifier}:{index}"
                queries.append({"id": query_id, "text": caption,
                                "metadata": {"language": language, "group_id": identifier,
                                             "query_source": "native_human_caption"}})
                qrels.append({"query_id": query_id, "corpus_id": identifier, "relevance": 1.0})
        results.append(write_dataset(destination, dataset_id=f"xm3600-{language}", track="photo",
                                     revision=XM3600_REVISION, corpus=corpus, queries=queries, qrels=qrels,
                                     sources=sources, license="annotations: CC-BY-4.0; images: per-image attribution CSV",
                                     text_source="absent_requires_generation",
                                     expected_counts={"corpus": 3600, "queries": {"tr": 7233, "en": 7200}[language]},
                                     metadata={"split": "evaluation", "language": language,
                                               "source": "https://google.github.io/crossmodal-3600/",
                                               "gold_captions_in_candidate_text": False}))
    return results


def _hf_metadata(repository: str, raw: Path, revision: str | None = None) -> dict:
    metadata_path = raw / "hf-metadata.json"
    if metadata_path.exists():
        result = json.loads(metadata_path.read_text())
        if revision and result["sha"] != revision:
            raise ValueError("Requested revision differs from frozen dataset snapshot")
        return result
    url = f"https://huggingface.co/api/datasets/{repository}" + (f"/revision/{revision}" if revision else "")
    with urllib.request.urlopen(url, timeout=60) as response:
        result = json.load(response)
    if not re.fullmatch(r"[0-9a-f]{40}", result["sha"]):
        raise ValueError("Missing immutable Hugging Face revision")
    _json(metadata_path, result)
    return result


def _hf_files(repository: str, raw: Path, prefixes: tuple[str, ...], revision: str | None = None) -> tuple[dict, list[dict]]:
    info = _hf_metadata(repository, raw, revision)
    names = sorted(row["rfilename"] for row in info["siblings"] if row["rfilename"].startswith(prefixes))
    def fetch(name):
        return _download(f"https://huggingface.co/datasets/{repository}/resolve/{info['sha']}/{name}", raw / _relative(name))
    with ThreadPoolExecutor(max_workers=4) as pool:
        sources = list(pool.map(fetch, names))
    return info, sources


def _parquet_rows(raw: Path, prefix: str) -> Iterable[dict]:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("ViDoRe import requires the optional pyarrow data dependency") from error
    for path in sorted((raw / prefix).glob("*.parquet")):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=64):
            yield from batch.to_pylist()


def prepare_vidore(root: Path, collection: str, language: str | None = None) -> dict:
    if collection not in VIDORE_COLLECTIONS:
        raise ValueError(f"Unknown public ViDoRe V3 collection {collection}")
    language = language or ("fr" if collection in {"finance_fr", "energy", "physics"} else "en")
    repository = f"vidore/vidore_v3_{collection}"
    raw = Path(root) / "raw" / f"vidore-v3-{collection}"
    info, sources = _hf_files(repository, raw, ("corpus/", "queries/", "qrels/", "documents_metadata/"), VIDORE_REVISIONS[collection])
    destination = Path(root) / f"vidore-v3-{collection}-{language}"
    snapshot_counts = {config["config_name"]: sum(split["num_examples"] for split in config["splits"])
                       for config in info["cardData"]["dataset_info"]}
    corpus = []
    for row in _parquet_rows(raw, "corpus"):
        identifier = str(row["corpus_id"])
        image = row["image"]
        if not image or not image.get("bytes"):
            raise ValueError(f"Missing embedded page image: {identifier}")
        payload = image["bytes"]
        extension = ".png" if payload.startswith(b"\x89PNG") else ".jpg"
        filename = f"media/{identifier}{extension}"
        target = destination / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        corpus.append({"id": identifier, "text": row.get("markdown") or "", "media": {"image": filename},
                       "metadata": {"doc_id": row["doc_id"], "page_number": row["page_number_in_doc"],
                                    "text_provenance": _provenance("published_markdown")}})
    queries = [{"id": str(row["query_id"]), "text": row["query"],
                "metadata": {"language": row["language"], "group_id": str(row["query_id"]),
                             "query_types": row["query_types"], "content_type": row["content_type"],
                             "answer": row.get("answer"), "answer_use": "evaluation_only"}}
               for row in _parquet_rows(raw, "queries") if row["language"] == {"en": "english", "fr": "french", "es": "spanish", "it": "italian", "de": "german", "pt": "portuguese"}.get(language, language)]
    query_ids = {row["id"] for row in queries}
    qrels = [{"query_id": str(row["query_id"]), "corpus_id": str(row["corpus_id"]), "relevance": float(row["score"])}
             for row in _parquet_rows(raw, "qrels") if str(row["query_id"]) in query_ids]
    documents = list(_parquet_rows(raw, "documents_metadata"))
    _json(destination / "documents_metadata.json", documents)
    return write_dataset(destination, dataset_id=destination.name, track="document", revision=info["sha"],
                         corpus=corpus, queries=queries, qrels=qrels, sources=sources,
                         license="CC-BY-4.0 annotations; document-specific licenses in documents_metadata.json",
                         text_source="published_markdown", expected_counts={"corpus": snapshot_counts["corpus"], "queries": VIDORE_COUNTS[collection][1]},
                         metadata={"repository": repository, "split": "test", "language": language,
                                   "query_bounding_boxes_excluded": True, "gold_answers_in_candidate_text": False,
                                   "paper_corpus_count": VIDORE_COUNTS[collection][0],
                                   "pinned_snapshot_corpus_count": snapshot_counts["corpus"],
                                   "paper_count_discrepancy": snapshot_counts["corpus"] != VIDORE_COUNTS[collection][0]})


def normalize_transcript(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def prepare_fleurs(root: Path) -> dict:
    raw = Path(root) / "raw" / "fleurs-tr_tr"
    info, sources = _hf_files("google/fleurs", raw, ("data/tr_tr/test.tsv", "data/tr_tr/audio/test.tar.gz"), FLEURS_REVISION)
    _extract_tar(raw / "data/tr_tr/audio/test.tar.gz", raw / "extracted")
    media = {path.name: path for path in (raw / "extracted").rglob("*.wav")}
    with (raw / "data/tr_tr/test.tsv").open(encoding="utf-8") as handle:
        rows = list(csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE))
    destination = Path(root) / "fleurs-tr_tr-test"
    corpus, queries, qrels, references = [], [], [], []
    groups = defaultdict(list)
    for row in rows:
        if len(row) != 7:
            raise ValueError(f"Expected seven FLEURS TSV fields, got {len(row)}")
        utterance, filename, transcript, normalized, _, samples, gender = row
        identifier = Path(filename).stem
        _link_asset(media[filename], destination / "media" / filename)
        normalized = normalize_transcript(normalized)
        group = hashlib.sha256(normalized.encode()).hexdigest()[:24]
        corpus.append({"id": identifier, "media": {"audio": f"media/{filename}"},
                       "metadata": {"sentence_id": utterance, "group_id": group, "samples": int(samples),
                                    "sampling_rate": 16000, "speaker_gender": gender, "text_status": "requires_asr"}})
        groups[group].append((identifier, normalized))
        references.append({"corpus_id": identifier, "transcript": transcript, "normalized_transcript": normalized,
                           "usage": "WER evaluation only; prohibited as candidate ASR text"})
    for group, members in sorted(groups.items()):
        queries.append({"id": group, "text": members[0][1],
                        "metadata": {"language": "tr", "group_id": group, "query_source": "gold_normalized_transcript",
                                     "query_type": "literal_content_search", "positive_count": len(members)}})
        qrels.extend({"query_id": group, "corpus_id": identifier, "relevance": 1.0} for identifier, _ in members)
    _jsonl(destination / "evaluation_transcripts.jsonl", references)
    return write_dataset(destination, dataset_id=destination.name, track="speech", revision=info["sha"],
                         corpus=corpus, queries=queries, qrels=qrels, sources=sources, license="CC-BY-4.0",
                         text_source="absent_requires_asr", expected_counts={"corpus": 743, "qrels": 743},
                         metadata={"language": "tr", "split": "test", "unique_transcript_groups": len(groups),
                                   "grouping": "NFKC + casefold + whitespace normalization of official normalized transcript",
                                   "paraphrase_extension": "not_created_requires_human_validation"})


def strip_python_docstrings(code: str) -> str:
    """Strip parsed docstrings and comments; retain formatting and identifiers.

    Token fallback handles Python 2 syntax in the original CodeSearchNet release.
    Source code without a successful token pass is rejected, never used unchanged.
    """
    ranges = []
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(code)
        lines = code.splitlines(keepends=True)
        for node in ast.walk(tree):
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                ranges.append((node.lineno, node.col_offset, node.end_lineno, node.end_col_offset))
        for start, col, end, end_col in sorted(ranges, reverse=True):
            # AST column offsets are UTF-8 byte offsets, not Unicode codepoints.
            before = lines[start - 1].encode()[:col].decode()
            after = lines[end - 1].encode()[end_col:].decode()
            lines[start - 1:end] = [before + "pass" + after]
        code = "".join(lines)
    except (SyntaxError, ValueError):
        pass
    tokens = []
    previous = tokenize.INDENT
    for token in tokenize.generate_tokens(io.StringIO(code).readline):
        if token.type == tokenize.COMMENT:
            continue
        if token.type == tokenize.STRING and previous in (tokenize.INDENT, tokenize.NEWLINE, tokenize.DEDENT):
            # A standalone string at block start is a docstring; other literals stay.
            tokens.append((tokenize.NAME, "pass"))
        else:
            tokens.append((token.type, token.string))
        if token.type not in (tokenize.NL, tokenize.COMMENT):
            previous = token.type
    return tokenize.untokenize(tokens).strip()


@lru_cache(maxsize=8)
def _code_parser(language: str):
    try:
        from tree_sitter_language_pack import get_parser
    except ImportError as error:
        raise RuntimeError("CodeSearchNet preparation requires tree-sitter-language-pack") from error
    return get_parser(language)


def strip_source_comments(code: str, language: str) -> str:
    """Remove syntax-tree comment nodes and Python standalone string comments.

    This operates without reading a query/docstring label. Runtime string
    literals, identifiers, original whitespace and non-comment syntax remain.
    Tree-sitter also handles Python 2 and mixed-indentation source snippets.
    """
    prefix = b"<?php\n" if language == "php" and not code.lstrip().startswith("<?") else b""
    payload = prefix + code.encode("utf-8")
    tree = _code_parser(language).parse(payload)
    ranges = []
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        is_comment = "comment" in node.type
        line_start = payload.rfind(b"\n", 0, node.start_byte) + 1
        line_end = payload.find(b"\n", node.end_byte)
        if line_end < 0:
            line_end = len(payload)
        lexical_statement = (not payload[line_start:node.start_byte].strip()
                             and not payload[node.end_byte:line_end].strip())
        standalone_python_string = language == "python" and node.type in {"string", "concatenated_string"} and (
            node.parent is not None and (node.parent.type in {"block", "module"} or (
                node.parent.type == "expression_statement" and len(node.parent.named_children) == 1)
                or (node.parent.type == "ERROR" and lexical_statement)))
        if is_comment or standalone_python_string:
            ranges.append((node.start_byte, node.end_byte))
        else:
            stack.extend(node.children)
    for start, end in sorted(ranges, reverse=True):
        # Keep newlines so adjacent source tokens never accidentally concatenate.
        removed = payload[start:end]
        payload = payload[:start] + bytes(10 if byte == 10 else 32 for byte in removed) + payload[end:]
    return payload[len(prefix):].decode("utf-8").strip()


def select_codesearchnet_ids(code_ids: list[str], query_ids: list[str], available: set[str],
                            expected_corpus_count: int) -> tuple[list[str], dict]:
    """Reproduce the official preprocess.py intersection, auditing every exclusion.

    The released codebase.txt files contain URLs absent from the associated raw
    validation/test archives. Upstream explicitly emits only available URLs;
    the resulting gallery must equal the published cleaned benchmark count.
    """
    missing_queries = set(query_ids) - available
    if missing_queries:
        raise ValueError(f"Missing {len(missing_queries)} official CodeSearchNet test queries")
    selected = [url for url in code_ids if url in available]
    if len(selected) != expected_corpus_count:
        raise ValueError(f"Official cleaned gallery mismatch: {len(selected)} != {expected_corpus_count}")
    return selected, {"upstream_codebase_url_lines": len(code_ids),
                      "upstream_urls_absent_from_raw_validation_test": len(code_ids) - len(selected),
                      "selection_rule": "Official preprocess.py: emit codebase.txt URL only if present in raw validation/test",
                      "published_cleaned_corpus_count": expected_corpus_count}


def prepare_codesearchnet(root: Path, language: str = "python") -> dict:
    import gzip
    counts = {"python": (43827, 14918), "javascript": (13981, 3291), "java": (40347, 10955),
              "go": (28120, 8122), "php": (52660, 14014), "ruby": (4360, 1261)}
    if language not in counts:
        raise ValueError(language)
    revision = "c0de43d3aaf38e89290f1efb771f8de845e7a489"
    raw = Path(root) / "raw" / "codesearchnet"
    sources = [_download(f"https://raw.githubusercontent.com/microsoft/CodeBERT/{revision}/GraphCodeBERT/codesearch/dataset.zip", raw / "dataset.zip"),
               _download(f"https://zenodo.org/records/7857872/files/{language}.zip", raw / f"{language}.zip")]
    with zipfile.ZipFile(raw / "dataset.zip") as archive:
        code_ids = archive.read(f"dataset/{language}/codebase.txt").decode().splitlines()
        query_ids = archive.read(f"dataset/{language}/test.txt").decode().splitlines()
    wanted = set(code_ids) | set(query_ids)
    examples = {}
    with zipfile.ZipFile(raw / f"{language}.zip") as archive:
        for name in sorted(archive.namelist()):
            if "/final/" not in name or not any(f"/{split}/" in name for split in ("test", "valid")):
                continue
            if not name.endswith((".jsonl", ".jsonl.gz")):
                continue
            with archive.open(name) as stream:
                handle = gzip.GzipFile(fileobj=stream) if name.endswith(".gz") else stream
                for line in handle:
                    row = json.loads(line)
                    if row["url"] in wanted:
                        examples[row["url"]] = row
    code_ids, selection_audit = select_codesearchnet_ids(code_ids, query_ids, set(examples), counts[language][0])
    corpus = []
    for url in code_ids:
        row = examples[url]
        text = strip_source_comments(row["code"], language)
        # Comments can disrupt parsing of old Python 2 snippets. Reparse after
        # removal until stable; labels never influence which text is removed.
        for _ in range(3):
            cleaned = strip_source_comments(text, language)
            if cleaned == text:
                break
            text = cleaned
        else:
            raise ValueError(f"Source cleanup did not stabilize: {url}")
        corpus.append({"id": url, "text": text, "metadata": {"language": language,
                       "repository": row.get("repo"), "path": row.get("path"),
                       "text_provenance": _provenance("source_code_docstrings_removed")}})
    queries = [{"id": str(index), "text": " ".join(examples[url]["docstring_tokens"]),
                "metadata": {"language": "en", "programming_language": language, "group_id": url}}
               for index, url in enumerate(query_ids)]
    qrels = [{"query_id": str(index), "corpus_id": url, "relevance": 1.0} for index, url in enumerate(query_ids)]
    destination = Path(root) / f"codesearchnet-{language}-test"
    return write_dataset(destination, dataset_id=destination.name, track="code", revision=revision,
                         corpus=corpus, queries=queries, qrels=qrels, sources=sources,
                         license="Zenodo mirror declares CC-BY-4.0; original source-code repository licenses apply; CodeSearchNet tooling MIT",
                         text_source="source_code_docstrings_removed", expected_counts=dict(zip(("corpus", "queries"), counts[language])),
                         metadata={"split": "test", "gallery": "official cleaned validation+test codebase.txt",
                                   "language": language, "query_language": "en", "source_docstrings_removed": True, "comment_removal": "tree-sitter-language-pack 1.21.0 syntax nodes",
                                   "selection_audit": selection_audit})


def prepare_clotho(root: Path) -> dict:
    import subprocess
    raw = Path(root) / "raw" / "clotho-v2.1"
    names = ("clotho_audio_evaluation.7z", "clotho_captions_evaluation.csv", "clotho_metadata_evaluation.csv", "LICENSE")
    sources = [_download(f"https://zenodo.org/records/4783391/files/{name}", raw / name) for name in names]
    extractor = shutil.which("bsdtar")
    if extractor is None:
        raise RuntimeError("Clotho .7z extraction needs libarchive bsdtar")
    archive = raw / "clotho_audio_evaluation.7z"
    members = subprocess.check_output([extractor, "-tf", str(archive)], text=True).splitlines()
    for member in members:
        _relative(member)
    extracted = raw / "extracted"
    extracted.mkdir(parents=True, exist_ok=True)
    subprocess.run([extractor, "-xf", str(archive.resolve()), "-C", str(extracted.resolve())], check=True)
    media = {path.name: path for path in extracted.rglob("*.wav")}
    with (raw / "clotho_captions_evaluation.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    destination = Path(root) / "clotho-v2.1-evaluation"
    corpus, queries, qrels = [], [], []
    for row in rows:
        filename = row["file_name"]
        identifier = Path(filename).stem
        _link_asset(media[filename], destination / "media" / filename)
        corpus.append({"id": identifier, "media": {"audio": f"media/{filename}"}, "metadata": {"text_status": "requires_generation"}})
        for index in range(1, 6):
            query_id = f"{identifier}:{index}"
            queries.append({"id": query_id, "text": row[f"caption_{index}"], "metadata": {"language": "en", "group_id": identifier}})
            qrels.append({"query_id": query_id, "corpus_id": identifier, "relevance": 1.0})
    return write_dataset(destination, dataset_id=destination.name, track="environment_audio", revision="zenodo:4783391:v2.1",
                         corpus=corpus, queries=queries, qrels=qrels, sources=sources,
                         license="Captions: Tampere noncommercial attribution license; audio: individual Freesound licenses",
                         text_source="absent_requires_generation", expected_counts={"corpus": 1045, "queries": 5225},
                         metadata={"split": "evaluation", "multi_positive_extension": "not_in_this_release_import"})


def prepare_clotho_additional(root: Path) -> dict:
    """Retain all rows in the official DCASE2025 multi-positive release.

    The task webpage says 1,069 queries, but its linked CSV at this immutable
    revision has 1,037. This named protocol preserves the discrepancy explicitly.
    """
    root = Path(root)
    base = root / "clotho-v2.1-evaluation"
    manifest = validate_dataset(base)
    revision = "c78422fcbed579877919620a30075baccf82bf2b"
    raw = root / "raw" / "clotho-v2.1" / "metadata_eval.csv"
    source = _download(f"https://raw.githubusercontent.com/CPJKU/dcase2025_task6_baseline/{revision}/resources/metadata_eval.csv", raw)
    with raw.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    corpus = read_jsonl(base / "corpus.jsonl")
    destination = root / "clotho-dcase2025-additional-relevance"
    for row in corpus:
        for filename in row["media"].values():
            _link_asset(base / filename, destination / filename)
    original_groups = {normalize_transcript(row["text"]): row["metadata"]["group_id"]
                       for row in read_jsonl(base / "queries.jsonl")}
    queries, qrels = [], []
    for index, row in enumerate(rows):
        query_id = str(index)
        references = ast.literal_eval(row["audio_filenames"])
        if not isinstance(references, list) or not all(isinstance(x, str) for x in references):
            raise ValueError("Invalid official additional audio references")
        group = original_groups.get(normalize_transcript(row["query"]), "query:" + query_id)
        queries.append({"id": query_id, "text": row["query"], "metadata": {"language": "en", "group_id": group}})
        qrels.extend({"query_id": query_id, "corpus_id": Path(filename).stem, "relevance": 1.0}
                     for filename in references)
    return write_dataset(destination, dataset_id=destination.name, track="environment_audio", revision=revision,
                         corpus=corpus, queries=queries, qrels=qrels, sources=manifest["sources"] + [source],
                         license=manifest["license"], text_source="absent_requires_generation",
                         expected_counts={"corpus": 1045, "queries": 1037},
                         metadata={"split": "DCASE2025 development-testing additional relevance", "primary_metric": "mAP@16",
                                   "upstream_webpage_claims_queries": 1069, "actual_pinned_csv_queries": 1037,
                                   "count_discrepancy": "Official linked CSV has 1037 rows; all are retained without subsampling."})


def _unavailable(destination: Path, *, dataset_id: str, track: str, revision: str, reason: str,
                 sources: list[dict], expected_counts: dict) -> dict:
    result = {"schema_version": 1, "id": dataset_id, "name": dataset_id, "track": track, "revision": revision,
              "status": "unavailable", "reason": reason, "sources": sources, "expected_counts": expected_counts,
              "counts": {"corpus": 0, "queries": 0, "qrels": 0, "media_files": 0}, "full_gallery": False}
    _json(destination / "dataset.json", result)
    return result


def prepare_cirr(root: Path, media_dir: Path | None = None) -> dict:
    revision = "2276fba2d2c53862c8ae79ea18e1f1ae03719f56"
    raw = Path(root) / "raw" / "cirr"
    files = ("captions/cap.rc2.val.json", "image_splits/split.rc2.val.json")
    sources = [_download(f"https://raw.githubusercontent.com/Cuberick-Orion/CIRR/{revision}/{name}", raw / name) for name in files]
    annotations = json.loads((raw / files[0]).read_text())
    images = json.loads((raw / files[1]).read_text())
    destination = Path(root) / "cirr-validation"
    missing = [name for name, path in images.items() if media_dir is None or not (Path(media_dir) / _relative(path)).is_file()]
    if missing:
        return _unavailable(destination, dataset_id=destination.name, track="composed_image", revision=revision,
                            reason=f"{len(missing)}/{len(images)} validation images unavailable. Official NLVR2 raw-image access requires its terms form. Supply --media-dir pointing to authorized img_raw/.",
                            sources=sources, expected_counts={"corpus": len(images), "queries": len(annotations)})
    corpus = []
    for identifier, path in sorted(images.items()):
        filename = f"media/{identifier}.png"
        _link_asset(Path(media_dir) / _relative(path), destination / filename)
        corpus.append({"id": identifier, "media": {"image": filename}, "metadata": {"text_status": "requires_generation"}})
    queries = [{"id": str(row["pairid"]), "text": row["caption"], "media": {"image": f"media/{row['reference']}.png"},
                "metadata": {"reference_id": row["reference"], "exclude_ids": [row["reference"]],
                             "group_id": row["reference"], "group_members": row["img_set"]["members"], "language": "en"}}
               for row in annotations]
    qrels = [{"query_id": str(row["pairid"]), "corpus_id": row["target_hard"], "relevance": 1.0} for row in annotations]
    return write_dataset(destination, dataset_id=destination.name, track="composed_image", revision=revision,
                         corpus=corpus, queries=queries, qrels=qrels, sources=sources,
                         license="CIRR annotations MIT; images original copyrights and NLVR2 terms", text_source="absent_requires_generation",
                         expected_counts={"corpus": len(images), "queries": 4181}, metadata={"split": "validation", "test_server_score": False})


def prepare_msrvtt(root: Path, media_dir: Path | None = None, download_media: bool = True) -> dict:
    raw = Path(root) / "raw" / "msrvtt"
    sources = [_download("https://github.com/ArrowLuo/CLIP4Clip/releases/download/v0.0/msrvtt_data.zip", raw / "msrvtt_data.zip")]
    with zipfile.ZipFile(raw / "msrvtt_data.zip") as archive:
        names = [name for name in archive.namelist() if name.endswith("MSRVTT_JSFUSION_test.csv")]
        if len(names) != 1:
            raise ValueError("Missing official MSRVTT 1K-A split")
        rows = list(csv.DictReader(io.StringIO(archive.read(names[0]).decode())))
    if media_dir is None and download_media:
        sources.append(_download("https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/MSRVTT.zip", raw / "MSRVTT.zip"))
        with zipfile.ZipFile(raw / "MSRVTT.zip") as archive:
            wanted = {f"{row['video_id']}.mp4" for row in rows}
            for name in archive.namelist():
                if Path(name).name in wanted:
                    target = raw / "extracted" / Path(name).name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(name) as handle, target.open("wb") as output:
                        shutil.copyfileobj(handle, output)
        media_dir = raw / "extracted"
    media = {p.stem: p for p in Path(media_dir).rglob("*.mp4")} if media_dir else {}
    missing = [row["video_id"] for row in rows if row["video_id"] not in media]
    destination = Path(root) / "msrvtt-1k-a"
    if missing:
        return _unavailable(destination, dataset_id=destination.name, track="video", revision="CLIP4Clip:v0.0:MSRVTT_JSFUSION_test",
                            reason=f"{len(missing)}/1000 prescribed videos unavailable; no caption substitution.",
                            sources=sources, expected_counts={"corpus": 1000, "queries": 1000})
    corpus, queries, qrels = [], [], []
    for row in rows:
        identifier = row["video_id"]
        filename = f"media/{identifier}.mp4"
        _link_asset(media[identifier], destination / filename)
        corpus.append({"id": identifier, "media": {"video": filename}, "metadata": {"text_status": "requires_frame_captioning"}})
        queries.append({"id": identifier, "text": row["sentence"], "metadata": {"language": "en", "group_id": identifier}})
        qrels.append({"query_id": identifier, "corpus_id": identifier, "relevance": 1.0})
    return write_dataset(destination, dataset_id=destination.name, track="video", revision="CLIP4Clip:v0.0:MSRVTT_JSFUSION_test",
                         corpus=corpus, queries=queries, qrels=qrels, sources=sources,
                         license="MSR-VTT original dataset terms; source videos retain original copyrights",
                         text_source="absent_requires_frame_captioning", expected_counts={"corpus": 1000, "queries": 1000},
                         metadata={"split": "1K-A", "protocol": "JSFusion", "audio_in_visual_main_condition": False})


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=["xm3600", "vidore", "fleurs", "codesearchnet", "clotho", "clotho-additional", "cirr", "msrvtt", "validate"])
    parser.add_argument("--root", type=Path, default=Path("data/multimodal"))
    parser.add_argument("--collection", choices=VIDORE_COLLECTIONS, default="computer_science")
    parser.add_argument("--language")
    parser.add_argument("--media-dir", type=Path)
    parser.add_argument("--no-download-media", action="store_true")
    args = parser.parse_args(argv)
    if args.dataset == "xm3600":
        result = prepare_xm3600(args.root, (args.language,) if args.language else ("tr", "en"))
    elif args.dataset == "vidore":
        result = prepare_vidore(args.root, args.collection, args.language)
    elif args.dataset == "fleurs":
        result = prepare_fleurs(args.root)
    elif args.dataset == "codesearchnet":
        result = prepare_codesearchnet(args.root, args.language or "python")
    elif args.dataset == "clotho":
        result = prepare_clotho(args.root)
    elif args.dataset == "clotho-additional":
        result = prepare_clotho_additional(args.root)
    elif args.dataset == "cirr":
        result = prepare_cirr(args.root, args.media_dir)
    elif args.dataset == "msrvtt":
        result = prepare_msrvtt(args.root, args.media_dir, not args.no_download_media)
    else:
        result = validate_dataset(args.root)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

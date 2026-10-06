"""Pinned RAGTurk import, source auditing and leakage-aware article splits.

Dataset licensing is separate from this package's MIT code license. The upstream
release is CC BY-NC-SA 4.0; downloaded and normalized text stays in ignored data/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import subprocess
import unicodedata
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

REPOSITORY = "metunlp/ragturk"
PINNED_REVISION = "86d35335d01b30869ab8eb392e380921fa1e2185"
PAPER_COUNTS = {"articles": 11196, "chunks": 58289, "questions": 20459}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _safe_path(name: str) -> Path:
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or "\\" in name:
        raise ValueError(f"Unsafe upstream file path: {name!r}")
    return Path(*relative.parts)


def _selected_file(name: str) -> bool:
    return (
        "/dataset/json/" in name and name.endswith(".json")
    ) or (
        "/dataset/raw/articles/" in name
        and name.rsplit("/", 1)[-1] in {"article.md", "chunks_index.md", "dataset.md"}
    )


def _download_release(raw_dir: Path, revision: str | None = None) -> tuple[Path, dict]:
    """Fetch an immutable Git snapshot, avoiding thousands of HTTP resolve calls.

    RAGTurk consists of thousands of small files. Git transfers these as a pack;
    individual resolve requests quickly exhaust Hugging Face anonymous limits.
    Existing snapshots are checked against their saved SHA-256 inventory offline.
    """
    requested = revision or PINNED_REVISION
    raw_dir.mkdir(parents=True, exist_ok=True)
    index_path = raw_dir / "release-files.json"
    release = raw_dir / "release"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        if index.get("revision") == requested:
            for entry in index["files"]:
                path = release / _safe_path(entry["path"])
                if not path.is_file() or _sha256(path) != entry["sha256"]:
                    raise ValueError(f"Raw cache is missing or changed: {path}; restore the pinned snapshot")
            return release, index
    metadata_path = raw_dir / "hf-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    if metadata.get("sha") != requested:
        url = f"https://huggingface.co/api/datasets/{REPOSITORY}/revision/{urllib.parse.quote(requested, safe='')}"
        with urllib.request.urlopen(url, timeout=60) as response:
            metadata = json.load(response)
        _write_json(metadata_path, metadata)
    resolved = metadata["sha"]
    if not re.fullmatch(r"[0-9a-f]{40}", resolved):
        raise ValueError("The upstream API did not return an immutable Git revision")
    paths = sorted(row["rfilename"] for row in metadata["siblings"] if _selected_file(row["rfilename"]))
    if not paths:
        raise ValueError("No supported RAGTurk article files found in the upstream inventory")
    checkout = raw_dir / "hf-checkout"
    if shutil.which("git") is None:
        raise RuntimeError("Preparing this many-file dataset requires Git; benchmark runs themselves are offline")
    if not (checkout / ".git").is_dir():
        if checkout.exists():
            raise ValueError(f"Cannot use non-Git snapshot directory: {checkout}")
        subprocess.run(
            ["git", "clone", "--depth", "1", f"https://huggingface.co/datasets/{REPOSITORY}", str(checkout)],
            check=True,
        )
    head = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    if head != resolved:
        subprocess.run(["git", "-C", str(checkout), "fetch", "--depth", "1", "origin", resolved], check=True)
        subprocess.run(["git", "-C", str(checkout), "checkout", "--detach", resolved], check=True)
    entries = []
    for name in paths:
        relative = _safe_path(name)
        origin = checkout / relative
        if not origin.is_file() or origin.is_symlink():
            raise ValueError(f"Pinned snapshot does not contain a regular file: {name}")
        target = release / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(origin, target)
        if target.read_bytes().startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise ValueError(f"Unresolved Git LFS pointer: {name}; fetch its data before import")
        entries.append({"path": name, "sha256": _sha256(target), "size": target.stat().st_size})
    index = {
        "repository": REPOSITORY,
        "revision": resolved,
        "license": "cc-by-nc-sa-4.0",
        "upstream_file_count": len(metadata["siblings"]),
        "files": entries,
    }
    _write_json(index_path, index)
    return release, index


def _scalar(frontmatter: str, key: str) -> str:
    match = re.search(rf"^\s*{re.escape(key)}:\s*(.*?)\s*$", frontmatter, flags=re.MULTILINE)
    if not match:
        return ""
    value = match.group(1)
    if value.startswith('"') and value.endswith('"'):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
    return value.strip("'\"").replace("''", "'")


def _table_cells(line: str) -> list[str]:
    """Unescape Markdown table separators, preserving all actual cell text."""
    cells = re.split(r"(?<!\\)\|", line.strip().strip("|"))
    return [cell.strip().replace(r"\|", "|").replace("<br>", "\n").replace("<br/>", "\n") for cell in cells]


def _parse_markdown_article(article_path: Path) -> dict:
    text = article_path.read_text(encoding="utf-8-sig")
    match = re.fullmatch(r"---\r?\n(.*?)\r?\n---\r?\n(.*)", text, flags=re.DOTALL)
    if not match:
        raise ValueError("missing_frontmatter")
    metadata, body = match.groups()
    declared_length = _scalar(metadata, "char_count")
    if declared_length:
        expected = int(declared_length)
        if len(body) != expected and len(body.rstrip("\r\n")) == expected:
            body = body.rstrip("\r\n")
        if len(body) != expected:
            raise ValueError(f"article_length_mismatch: {len(body)} != {expected}")
    chunks = []
    for line in article_path.with_name("chunks_index.md").read_text(encoding="utf-8-sig").splitlines():
        identifier = re.match(r"^\|\s*(c\d+)\s*\|", line)
        if not identifier:
            continue
        # Titles/previews sometimes contain unescaped pipes. The explicitly
        # delimited character range and chunk ID remain unambiguous.
        spans = re.findall(r"\|\s*(\d+)\s*[-–]\s*(\d+)\s*\|", line)
        if len(spans) != 1:
            raise ValueError("malformed_chunk_range")
        start, end = map(int, spans[0])
        if not 0 <= start < end <= len(body):
            raise ValueError(f"chunk_range_out_of_bounds: {start}-{end} for {len(body)}")
        chunks.append({"id": identifier.group(1), "content": body[start:end], "start_char": start, "end_char": end})
    declared_chunks = _scalar(metadata, "num_chunks")
    if declared_chunks and len(chunks) != int(declared_chunks):
        raise ValueError("chunk_count_mismatch")
    questions, issues = [], []
    question_file = article_path.with_name("dataset.md")
    if question_file.exists():
        question_lines = question_file.read_text(encoding="utf-8-sig").splitlines()
    else:
        question_lines = []
        issues.append({"scope": "article_questions", "reason": "missing_question_file"})
    for line_number, line in enumerate(question_lines, 1):
        if not re.match(r"^\|\s*\d+\s*\|", line):
            continue
        cells = _table_cells(line)
        if len(cells) != 5:
            issues.append({"scope": "question", "reason": "malformed_question_table", "line": line_number})
            continue
        questions.append({
            "question": cells[1], "answer": cells[2], "category": cells[3],
            "related_chunk_ids": [part.strip().strip('`') for part in cells[4].split(",") if part.strip()],
        })
    return {
        "article": {
            "id": _scalar(metadata, "id") or article_path.parent.name,
            "title": _scalar(metadata, "title"), "url": _scalar(metadata, "url"),
            "lang": _scalar(metadata, "lang"),
            "options": {"llm_model": _scalar(metadata, "llm_model")},
        },
        "chunks": chunks,
        "questions": {"items": questions, "total_questions": int(_scalar(metadata, "total_questions") or len(questions))},
        "import_issues": issues,
        "format": "markdown_char_ranges",
    }


def _text_key(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


class _UnionFind:
    def __init__(self, values: list[str]):
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: str, right: str) -> None:
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[max(left, right)] = min(left, right)


def _group_questions(corpus: list[dict], questions: list[dict]) -> dict[str, list[dict]]:
    """Join articles sharing exact question text or exact labeled evidence text.

    Unreferenced boilerplate chunks are not evidence and do not merge unrelated
    articles. Whitespace/Unicode normalization also catches formatting duplicates.
    """
    union = _UnionFind(sorted({question["article_id"] for question in questions}))
    documents = {document["id"]: document for document in corpus}
    question_owner, evidence_owner = {}, {}
    for question in questions:
        article = question["article_id"]
        key = _text_key(question["question"])
        if key in question_owner:
            union.union(article, question_owner[key])
        else:
            question_owner[key] = article
        for gold_id in question["gold_ids"]:
            evidence = _text_key(documents[gold_id]["text"])
            if evidence in evidence_owner:
                union.union(article, evidence_owner[evidence])
            else:
                evidence_owner[evidence] = article
    groups = defaultdict(list)
    for question in questions:
        groups[union.find(question["article_id"])].append(question)
    return dict(groups)


def split_questions(corpus: list[dict], questions: list[dict], dev_size: int = 2000, seed: int = 42) -> tuple[list[dict], list[dict], dict]:
    if dev_size < 0 or dev_size >= len(questions):
        raise ValueError("dev_size must be nonnegative and smaller than the number of valid questions")
    groups = _group_questions(corpus, questions)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    dev, test = [], []
    for key in keys:
        group = groups[key]
        if abs(len(dev) + len(group) - dev_size) < abs(len(dev) - dev_size):
            dev.extend(group)
        else:
            test.extend(group)
    if not test or (dev_size > 0 and not dev):
        raise ValueError("Source/duplicate grouping leaves no usable dev/test separation")
    for rows in (dev, test):
        rows.sort(key=lambda row: row["id"])
    return dev, test, {
        "seed": seed, "requested_dev_questions": dev_size, "dev_questions": len(dev),
        "test_questions": len(test), "groups": len(groups),
        "largest_group_questions": max(map(len, groups.values())),
        "grouping": "article_id + normalized exact question + normalized exact gold-evidence text",
        "near_duplicate_guarantee": False,
    }


def normalize_release(release: Path, file_index: dict, output_dir: Path, dev_size: int = 2000, seed: int = 42) -> dict:
    """Normalize an already downloaded release; deterministic and network-free."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory = {entry["path"]: entry for entry in file_index["files"]}
    inputs = sorted(name for name in inventory if "/dataset/json/" in name and name.endswith(".json"))
    inputs += sorted(name for name in inventory if name.endswith("/article.md"))
    corpus, questions, rejections = [], [], []
    counts, categories, sources, declared_models, declared_languages = Counter(), Counter(), Counter(), Counter(), Counter()
    seen_articles, seen_questions = set(), set()
    for name in inputs:
        counts["input_articles"] += 1
        path = release / _safe_path(name)
        try:
            if name.endswith(".json"):
                data = json.loads(path.read_text(encoding="utf-8-sig"))
                data["format"] = "json"
            else:
                data = _parse_markdown_article(path)
            article = data["article"]
            source = "wikipedia" if "wikipedia.org/" in article.get("url", "") else "web"
            original_id = str(article.get("id", "")).strip()
            if not original_id:
                raise ValueError("missing_article_id")
            article_id = source + ":" + original_id
            if article_id in seen_articles:
                raise ValueError("duplicate_article_id")
            seen_articles.add(article_id)
            chunks = data["chunks"]
            qa_container = data.get("questions", {})
            items = qa_container.get("items", []) if isinstance(qa_container, dict) else qa_container
            if not isinstance(chunks, list) or not isinstance(items, list):
                raise ValueError("invalid_article_schema")
        except (ValueError, TypeError, KeyError, OSError) as error:
            counts["rejected_articles"] += 1
            reason = "missing_companion_file" if isinstance(error, FileNotFoundError) else str(error)
            rejections.append({"path": name, "scope": "article", "reason": reason, "detail": str(error)})
            continue
        counts["parsed_articles"] += 1
        counts["articles_" + source] += 1
        counts["articles_format_" + data["format"]] += 1
        declared_models[str(article.get("options", {}).get("llm_model", "unknown"))] += 1
        declared_languages[str(article.get("lang", "unknown"))] += 1
        for issue in data.get("import_issues", []):
            if issue.get("scope") == "question":
                counts["rejected_questions"] += 1
                counts["raw_questions"] += 1
            else:
                counts["articles_without_question_file"] += 1
            rejections.append({"path": name, "scope": "question", **issue})
        article_documents = {}
        for index, chunk in enumerate(chunks):
            counts["raw_chunks"] += 1
            local_id = str(chunk.get("id", "")).strip()
            content = chunk.get("content", "")
            if not local_id or not isinstance(content, str) or not content.strip() or local_id in article_documents:
                counts["rejected_chunks"] += 1
                rejections.append({"path": name, "scope": "chunk", "index": index, "reason": "empty_or_duplicate_chunk"})
                continue
            document = {"id": article_id + "#" + local_id, "text": content, "title": article.get("title", ""),
                        "article_id": article_id, "source": source}
            article_documents[local_id] = document
            corpus.append(document)
        counts["raw_questions"] += len(items)
        if isinstance(qa_container, dict) and qa_container.get("total_questions") != len(items):
            counts["declared_question_count_mismatches"] += 1
        for index, item in enumerate(items):
            try:
                question = item.get("question", "")
                answer = item.get("answer", "")
                answers = answer if isinstance(answer, list) else [answer]
                gold = item.get("related_chunk_ids", [])
                if not isinstance(question, str) or not question.strip():
                    raise ValueError("empty_question")
                if not answers or not all(isinstance(value, str) and value.strip() for value in answers):
                    raise ValueError("empty_answer")
                if not isinstance(gold, list) or not gold or not all(isinstance(value, str) for value in gold):
                    raise ValueError("empty_or_invalid_gold_ids")
                if any(value not in article_documents for value in gold):
                    raise ValueError("missing_gold_chunk")
                gold_ids = sorted({article_documents[value]["id"] for value in gold})
                key = (article_id, _text_key(question), tuple(answers), tuple(gold_ids))
                if key in seen_questions:
                    raise ValueError("duplicate_question_record")
                seen_questions.add(key)
                category = str(item.get("category", "UNKNOWN"))
                questions.append({"id": article_id + f"#q{index:04d}", "question": question.strip(),
                                  "answers": [value.strip() for value in answers], "gold_ids": gold_ids,
                                  "article_id": article_id, "category": category, "source": source})
                categories[category] += 1
                sources[source] += 1
            except (ValueError, TypeError, AttributeError) as error:
                counts["rejected_questions"] += 1
                rejections.append({"path": name, "scope": "question", "index": index, "reason": str(error)})
    if not corpus or not questions:
        raise ValueError("No usable corpus/questions remained after auditing the release")
    corpus.sort(key=lambda row: row["id"])
    dev, test, split_info = split_questions(corpus, questions, dev_size, seed)
    counts["corpus_chunks"] = len(corpus)
    counts["valid_questions"] = len(questions)
    counts["dev_questions"] = len(dev)
    counts["test_questions"] = len(test)
    counts["articles_with_valid_questions"] = len({question["article_id"] for question in questions})
    for filename, rows in [("corpus.jsonl", corpus), ("questions.dev.jsonl", dev), ("questions.test.jsonl", test),
                           ("rejections.jsonl", rejections)]:
        _write_jsonl(output_dir / filename, rows)
    checksums = {name: _sha256(output_dir / name) for name in
                 ["corpus.jsonl", "questions.dev.jsonl", "questions.test.jsonl", "rejections.jsonl"]}
    fingerprint = hashlib.sha256(json.dumps(checksums, sort_keys=True).encode()).hexdigest()
    warnings = [
        "QA and relevance labels are synthetic, not independently human-verified ground truth.",
        "Only exact normalized duplicates are grouped; near-duplicate leakage is not ruled out.",
        "The public snapshot does not equal the full dataset described in the paper.",
        "Markdown chunks are reconstructed only from published text and published character ranges.",
        "Some upstream language/model metadata disagrees with the paper; no language filtering is inferred from it.",
    ]
    manifest = {
        "schema_version": 1, "dataset": REPOSITORY, "revision": file_index["revision"],
        "dataset_license": "CC-BY-NC-SA-4.0", "paper_counts": PAPER_COUNTS,
        "counts": dict(sorted(counts.items())), "split": split_info,
        "category_counts": dict(sorted(categories.items())), "source_question_counts": dict(sorted(sources.items())),
        "declared_article_models": dict(sorted(declared_models.items())),
        "declared_article_languages": dict(sorted(declared_languages.items())),
        "rejection_counts": dict(sorted(Counter(row["reason"] for row in rejections).items())),
        "raw_files": {"count": len(file_index["files"]), "bytes": sum(row["size"] for row in file_index["files"]),
                      "inventory_sha256": hashlib.sha256(json.dumps(file_index, sort_keys=True).encode()).hexdigest()},
        "file_sha256": checksums, "fingerprint": fingerprint, "warnings": warnings,
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def prepare_ragturk(output_dir: Path, dev_size: int = 2000, seed: int = 42, revision: str | None = None) -> dict:
    """Download/cache, audit, normalize and split the public RAGTurk snapshot."""
    output_dir = Path(output_dir)
    release, file_index = _download_release(output_dir.parent / "raw", revision)
    return normalize_release(release, file_index, output_dir, dev_size, seed)


def load_dataset(data_dir: Path, split: str = "dev") -> tuple[list[dict], list[dict], dict]:
    """Load locally prepared records and verify hashes/labels without networking."""
    if split not in {"dev", "test"}:
        raise ValueError("split must be 'dev' or 'test'")
    data_dir = Path(data_dir)
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    def load(name: str) -> list[dict]:
        path = data_dir / name
        expected = manifest["file_sha256"][name]
        if _sha256(path) != expected:
            raise ValueError(f"Prepared data hash mismatch: {path}")
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    corpus = load("corpus.jsonl")
    questions = load(f"questions.{split}.jsonl")
    corpus_ids = {document["id"] for document in corpus}
    if len(corpus_ids) != len(corpus):
        raise ValueError("Duplicate corpus IDs")
    for question in questions:
        if not question["gold_ids"] or not set(question["gold_ids"]).issubset(corpus_ids):
            raise ValueError(f"Question references missing evidence: {question['id']}")
    return corpus, questions, manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/ragturk"))
    parser.add_argument("--dev-size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--revision", default=PINNED_REVISION)
    args = parser.parse_args()
    result = prepare_ragturk(args.output_dir, args.dev_size, args.seed, args.revision)
    print(json.dumps(result, ensure_ascii=False, indent=2))

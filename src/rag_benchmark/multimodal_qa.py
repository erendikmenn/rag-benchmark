"""Source-verified QA readiness and lexical/set metrics; no model inference.

A future runner must call prepare_qa(), keep QARecord references evaluation-only,
pass generation_query(record) plus select_evidence() to its frozen generator,
and save predictions separately before evaluate_prediction(). Never place gold
answers in generation prompts. Readiness is not measured QA quality. Raw answer
strings stay in local memory; the readiness manifest contains counts/hashes only.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import unicodedata


@dataclass(frozen=True)
class QARecord:
    query_id: str
    question: str
    language: str
    references: tuple[str, ...]
    positive_source_ids: tuple[str, ...]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def normalize_lexical(text: str, language: str) -> str:
    if not isinstance(text, str):
        raise ValueError("Answers must be strings")
    text = unicodedata.normalize("NFKC", text)
    if language in {"tr", "turkish", "tr_tr"}:
        text = text.replace("I", "ı").replace("İ", "i")
    text = text.casefold()
    text = "".join(" " if unicodedata.category(char).startswith("P") else char for char in text)
    return " ".join(text.split())


def lexical_metrics(prediction: str, references: tuple[str, ...] | list[str], language: str) -> dict:
    if not references or any(
        not isinstance(ref, str) or not normalize_lexical(ref, language) for ref in references
    ):
        raise ValueError("Nonempty certified reference answers are required")
    pred = normalize_lexical(prediction, language)
    if not pred:
        return {"answer_em": 0.0, "answer_token_f1": 0.0, "nonanswer": True}
    tokens = pred.split()
    em = f1 = 0.0
    for reference in references:
        ref = normalize_lexical(reference, language)
        target = ref.split()
        common = sum((Counter(tokens) & Counter(target)).values())
        em = max(em, float(pred == ref))
        f1 = max(f1, 2 * common / (len(tokens) + len(target)))
    return {"answer_em": em, "answer_token_f1": f1, "nonanswer": False}


def citation_source_set_metrics(cited_ids: list[str], positive_ids: tuple[str, ...] | list[str]) -> dict:
    """Qrel source-ID set overlap; does not test support/entailment of answer text."""
    if not positive_ids:
        raise ValueError("Positive source labels are required")
    if any(not isinstance(item, str) or not item for item in [*cited_ids, *positive_ids]):
        raise ValueError("Source IDs must be nonempty strings")
    cited, positives = set(cited_ids), set(positive_ids)
    matched = len(cited & positives)
    return {
        "citation_source_set_precision": matched / len(cited) if cited else 0.0,
        "citation_source_set_recall": matched / len(positives),
        "citation_source_set_hit": float(matched > 0),
        "citation_entailment_accuracy": None,
    }


def generation_query(record: QARecord) -> dict:
    """Only this projection should be supplied to the generation runner."""
    return {"query_id": record.query_id, "question": record.question, "language": record.language}


def select_evidence(
    record: QARecord,
    corpus: list[dict],
    *,
    condition: str,
    representation: str,
    retrieved_ids: list[str] | None = None,
    dataset_root: Path | None = None,
) -> list[dict]:
    if condition not in {"closed_book", "oracle", "retrieved"} or representation not in {"text", "image"}:
        raise ValueError("Unknown frozen QA condition")
    by_id = {row["id"]: row for row in corpus}
    if len(by_id) != len(corpus):
        raise ValueError("Duplicate evidence ID")
    if condition == "closed_book":
        selected = []
    elif condition == "oracle":
        # Sorted positive IDs, capped to the same five-source budget as retrieval.
        selected = sorted(record.positive_source_ids)[:5]
    else:
        if retrieved_ids is None:
            raise ValueError("Retrieved condition requires actual ranked source IDs")
        if len(retrieved_ids) != len(set(retrieved_ids)):
            raise ValueError("Retrieved source rankings must not contain duplicates")
        selected = retrieved_ids[:5]
    result = []
    for identifier in selected:
        if identifier not in by_id:
            raise ValueError("Evidence ID absent from frozen corpus")
        row = by_id[identifier]
        if representation == "text":
            if not isinstance(row.get("text"), str) or not row["text"].strip():
                raise ValueError("Selected source has no usable text evidence")
            result.append({"id": identifier, "text": row["text"]})
        else:
            image = row.get("media", {}).get("image")
            if not isinstance(image, str) or not image:
                raise ValueError("Selected source has no image evidence")
            if dataset_root is not None:
                root = Path(dataset_root).resolve()
                target = (root / image).resolve()
                if not target.is_relative_to(root) or not target.is_file():
                    raise ValueError("Selected image is absent or outside frozen dataset root")
            result.append({"id": identifier, "media": {"image": image}})
    return result


def evaluate_prediction(record: QARecord, *, answer: str, cited_source_ids: list[str]) -> dict:
    return {
        **lexical_metrics(answer, record.references, record.language),
        **citation_source_set_metrics(cited_source_ids, record.positive_source_ids),
        "human_semantic_accuracy": None,
    }


def protocol() -> dict:
    return {
        "representations": ["text", "image"],
        "conditions": ["closed_book", "oracle", "retrieved"],
        "source_budget": 5,
        "evidence_availability": "every selected source must pass representation checks; missing text/image is an error, never silently removed",
        "oracle_selection": "positive qrel IDs sorted lexically, first five",
        "retrieved_selection": "actual frozen retrieval/rerank ranking first five; no gold filtering",
        "reference_policy": "published canonical answer only; raw_answers are not certified alternate references",
        "explicit_certified_multigold_policy": "maximum EM and maximum token F1 independently across references",
        "lexical_normalization": "Unicode NFKC; language-aware Turkish dotted/dotless I; casefold; Unicode punctuation to spaces; whitespace tokens; no article stripping",
        "citation_metric": "positive qrel source-ID set overlap; not entailment or claim-support correctness",
        "human_semantic_accuracy": "pending human-labeled calibration set",
        "frozen_generator_prompt_settings_and_evidence_budget": "required before generation; no runner executed",
        "main_18460_matrix_inclusion": False,
        "existing_text_qa_pilot_inclusion": False,
    }


def prepare_qa(dataset: Path, raw_queries: list[Path]) -> tuple[list[QARecord], dict]:
    """Fail on changed raw/prepared mapping; return local records + public-safe audit."""
    dataset = Path(dataset)
    manifest_path = dataset / "dataset.json"
    manifest = json.loads(manifest_path.read_text())
    audit = {
        "schema_version": 1,
        "dataset": manifest["id"],
        "protocol": protocol(),
        "generation_executed": False,
        "quality_measured": False,
        "manifest_sha256": sha256(manifest_path),
        "qa_implementation_sha256": sha256(Path(__file__)),
    }
    if manifest.get("track") != "document":
        return [], {
            **audit,
            "status": "unavailable",
            "reason": "Only source-verified document QA is supported",
        }
    required = ["queries.jsonl", "qrels.jsonl", "corpus.jsonl", "assets.jsonl"]
    hashes = {name: sha256(dataset / name) for name in required}
    for name, digest in hashes.items():
        if manifest.get("files", {}).get(name, {}).get("sha256") != digest:
            raise ValueError("Prepared source file hash mismatch: " + name)
    audit["prepared_source_sha256"] = hashes
    queries, qrels, corpus = (read_jsonl(dataset / name) for name in required[:3])
    if not raw_queries:
        return [], {
            **audit,
            "status": "unavailable",
            "reason": "Published raw query/answer sources are required",
            "query_count": len(queries),
        }
    bound_sources = [
        source
        for source in manifest.get("sources", [])
        if "/queries/" in source.get("url", "") and source.get("sha256")
    ]
    supplied_hashes = [sha256(path) for path in raw_queries]
    if not bound_sources or sorted(supplied_hashes) != sorted(source["sha256"] for source in bound_sources):
        raise ValueError("Supplied raw query hashes are not bound to frozen manifest sources")
    audit["raw_source_binding"] = (
        "supplied query parquet SHA256 set equals frozen manifest /queries/ source SHA256 set; exact query/answer matching below"
    )
    import pyarrow.parquet as pq

    raw = {}
    raw_hashes = []
    raw_alias_fields = Counter()
    for path in raw_queries:
        raw_hashes.append(sha256(path))
        table = pq.read_table(path)
        if not {"query_id", "query", "language", "answer"} <= set(table.column_names):
            return [], {
                **audit,
                "status": "unavailable",
                "reason": "Published raw source lacks gold answer fields",
            }
        for row in table.select(
            [
                key
                for key in ["query_id", "query", "language", "answer", "raw_answers"]
                if key in table.column_names
            ]
        ).to_pylist():
            key = (str(row["query_id"]), row["language"])
            if key in raw:
                raise ValueError("Ambiguous published query ID/language")
            raw[key] = row
    audit["raw_query_source_sha256"] = sorted(raw_hashes)
    expected_language = {
        "en": "english",
        "fr": "french",
        "tr": "turkish",
        "de": "german",
        "es": "spanish",
        "it": "italian",
        "pt": "portuguese",
    }.get(manifest.get("metadata", {}).get("language"))
    if expected_language is None:
        raise ValueError("Frozen dataset must declare an explicit supported language")
    by_id = {row["id"]: row for row in corpus}
    if len(by_id) != len(corpus) or len({row["id"] for row in queries}) != len(queries):
        raise ValueError("Duplicate frozen corpus/query IDs")
    positives = {row["id"]: set() for row in queries}
    seen_labels = set()
    for label in qrels:
        pair = (label["query_id"], label["corpus_id"])
        if pair in seen_labels or not math.isfinite(float(label["relevance"])):
            raise ValueError("Duplicate or nonfinite frozen qrel labels")
        seen_labels.add(pair)
        if label["query_id"] not in positives or label["corpus_id"] not in by_id:
            raise ValueError("Qrel mapping references absent frozen records")
        if label["relevance"] > 0:
            positives[label["query_id"]].add(label["corpus_id"])
    records, reasons = [], Counter()
    for query in queries:
        metadata = query.get("metadata", {})
        language = metadata.get("language")
        if language != expected_language:
            raise ValueError("Prepared query differs from frozen language selection")
        source = raw.get((query["id"], language))
        if source is None or source["query"] != query["text"] or source["answer"] != metadata.get("answer"):
            raise ValueError("Prepared question/answer differs from published source")
        raw_alias_fields[len(source.get("raw_answers") or [])] += 1
        answer = metadata.get("answer")
        if metadata.get("answer_use") != "evaluation_only":
            raise ValueError("Gold answer must be restricted to evaluation")
        if not isinstance(answer, str) or not normalize_lexical(answer, language):
            reasons["missing_usable_published_answer"] += 1
        elif not positives[query["id"]]:
            reasons["missing_positive_source_labels"] += 1
        else:
            records.append(
                QARecord(
                    query["id"], query["text"], language, (answer,), tuple(sorted(positives[query["id"]]))
                )
            )
    if sorted(sha256(path) for path in raw_queries) != sorted(raw_hashes):
        raise ValueError("Published raw query source changed during verification")
    selected_raw_count = sum(language == expected_language for _, language in raw)
    if selected_raw_count != len(queries):
        raise ValueError("Prepared queries do not cover the complete published language split")
    audit.update(
        status="ready_for_generation" if len(records) == len(queries) and records else "unavailable",
        query_count=len(queries),
        eligible_query_count=len(records),
        excluded_query_count=len(queries) - len(records),
        excluded_reasons=dict(reasons),
        selected_language=expected_language,
        exact_published_question_answer_matches=len(queries),
        raw_answer_list_lengths=dict(raw_alias_fields),
        text_source_count=sum(
            isinstance(row.get("text"), str) and bool(row["text"].strip()) for row in corpus
        ),
        image_source_count=sum(bool(row.get("media", {}).get("image")) for row in corpus),
        source_asset_verification="frozen prepared-file hashes and asset inventory; media bytes must be validated before generator use",
    )
    return records, audit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--raw-queries", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    _, audit = prepare_qa(args.dataset, args.raw_queries)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": audit["status"],
                "eligible_query_count": audit.get("eligible_query_count", 0),
                "generation_executed": False,
            }
        )
    )


if __name__ == "__main__":
    main()

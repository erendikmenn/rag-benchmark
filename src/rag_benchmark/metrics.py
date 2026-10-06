"""Retrieval labels and lexical answer matching measure different things."""

from collections import Counter
import math
import re
import unicodedata


def strip_source_citations(text: str, context_ids: list[str]) -> str:
    """Remove only explicit citations to supplied IDs; preserve raw answers separately."""
    allowed = set(context_ids)
    def replace(match):
        identifiers = [part.strip() for part in re.split(r"[,;]", match.group(1))]
        return "" if identifiers and all(part in allowed for part in identifiers) else match.group(0)
    return re.sub(r"\[([^\[\]\n]+)\]", replace, text)


def normalize_answer(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(str.maketrans({"I": "ı", "İ": "i"})).lower()
    text = "".join(" " if unicodedata.category(c).startswith("P") else c for c in text)
    return " ".join(text.split())


def answer_metrics(prediction: str, answers: list[str]) -> dict[str, float]:
    if not answers:
        raise ValueError("Reference answers are required for lexical scoring")
    pred = normalize_answer(prediction)
    p = pred.split()
    em, f1 = 0.0, 0.0
    for answer in answers:
        ref = normalize_answer(answer)
        r = ref.split()
        em = max(em, float(pred == ref))
        common = sum((Counter(p) & Counter(r)).values())
        score = 2 * common / (len(p) + len(r)) if p or r else 1.0
        f1 = max(f1, score)
    return {"answer_em": em, "answer_token_f1": f1}


def retrieval_metrics(ranked_ids: list[str], gold_ids: list[str], ks=(1, 5, 10, 50)) -> dict[str, float]:
    gold = set(gold_ids)
    if not gold:
        raise ValueError("At least one relevant chunk is required")
    # A repeated result must not create extra relevance credit.
    ranked = list(dict.fromkeys(ranked_ids))
    values = {}
    for k in ks:
        top = ranked[:k]
        hits = sum(x in gold for x in top)
        values[f"recall@{k}"] = hits / len(gold)
        values[f"hit@{k}"] = float(hits > 0)
        values[f"all_evidence@{k}"] = float(hits == len(gold))
        values[f"mrr@{k}"] = next((1.0 / i for i, x in enumerate(top, 1) if x in gold), 0.0)
        dcg = sum(1 / math.log2(i + 1) for i, x in enumerate(top, 1) if x in gold)
        ideal = sum(1 / math.log2(i + 1) for i in range(1, min(k, len(gold)) + 1))
        values[f"ndcg@{k}"] = dcg / ideal
    return values

"""Synthetic source metadata only; no real media/cache/model reads."""
import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

from rag_benchmark import fleurs_remaining_data as m
from rag_benchmark.multimodal import Dataset


@pytest.fixture
def fixture(tmp_path):
    corpus = [{"id": f"d{i}", "text": "ASR only", "media": {"audio": f"{i}.wav"},
               "metadata": {"secret_gold": "never model input"}} for i in range(743)]
    originals = [{"id": f"s{i}", "text": f"source {i}", "media": {}} for i in range(329)]
    lines = [(json.dumps(r) + "\n").encode() for r in originals]
    raw = b"".join(lines)
    source_ref = {"path": str(tmp_path / "queries.jsonl"), "sha256": hashlib.sha256(raw).hexdigest()}
    draft_ref = {"path": "old25", "sha256": "0" * 64}
    source = Dataset(tmp_path, {"track": "speech"}, corpus, originals,
                     {r["id"]: {"d0": 1} for r in originals}, "original")
    pilot = SimpleNamespace(source_dataset=source,
        dataset=SimpleNamespace(queries=[{"id": f"pp{i:02d}"} for i in range(1, 26)], identity="pilot"),
        provenance={"drafts": draft_ref, "query_lineage": {"source_ref": source_ref,
            "query_links": [{"parent_query_id": f"s{i}"} for i in range(25)]}})
    config = {"drafts": {"path": "new304", "sha256": "1" * 64},
              "selection": {"path": "selection", "sha256": "2" * 64}}
    choices, rows, decisions = [], [], []
    for offset, source_index in enumerate(range(25, 329), 1):
        row = {"query_id": f"ep{offset:03d}", "parent_query_id": f"s{source_index}",
               "source_line_1based": source_index + 1,
               "source_line_sha256": hashlib.sha256(lines[source_index]).hexdigest()}
        choices.append({**row, "source_text": originals[source_index]["text"]})
        text = f"different source {source_index}"
        rows.append({**row, "query_text": text, "agent_authored": True, "human_validated": False,
                     "semantic_preservation_notes": "private note", "source_context_notes": "private context"})
        decisions.append({"query_id": row["query_id"], "parent_query_id": row["parent_query_id"],
            "source_line_sha256": row["source_line_sha256"], "query_text_sha256": m.text_sha(text),
            "decision": "PASS", "unresolved_semantic_issues": 0})
    selection = {"protocol": "fleurs-paraphrase-remaining-source-selection-v1", "remaining_count": 304,
        "preserved_pilot_count": 25, "total_target_count": 329, "human_validated": False,
        "source": source_ref, "preserved_pilot": draft_ref, "sources": choices}
    drafts = {"selection_ref": config["selection"], "count": 304, "human_validation_complete": False,
              "qrels_full_validation": False, "queries": rows}
    review = {"protocol": m.REVIEW_PROTOCOL, "status": "PASS_AGENT_MEANING_REVIEW", **{
        "drafts_ref": config["drafts"], "selection_ref": config["selection"]}, "query_count": 304,
        "unresolved_semantic_issues": 0, "human_validated": False, "qrels_full_validation": False,
        "per_query": decisions}
    return SimpleNamespace(config=config, pilot=pilot, selection=selection, drafts=drafts,
                           review=review, raw=raw, source=source)


def derive(f):
    return m.derive_queries(f.config, f.pilot, f.selection, f.drafts, f.review, f.raw)


def test_exact_complement_and_gold_stripped(fixture):
    f = fixture
    queries, qrels, lineage = derive(f)
    assert [r["id"] for r in queries] == m.IDS and len(qrels) == len(lineage) == 304
    assert {r["parent_query_id"] for r in lineage}.isdisjoint({f"s{i}" for i in range(25)})
    assert all(set(r) == {"id", "text", "media"} for r in queries)
    dataset = m.RemainingDataset(f.source.root, {}, copy.deepcopy(f.source.corpus), queries, qrels, "new", f.source)
    assert set(dataset.model_item(queries[0])) == {"id", "text", "media"}
    assert set(dataset.model_item(f.source.corpus[0])) == {"id", "text", "media"}
    assert dataset.model_item(f.source.corpus[0], text_only=True)["media"] == {}
    assert "private" not in json.dumps(queries)


@pytest.mark.parametrize("kind", ["pilot_parent", "duplicate_parent", "old_id", "line_bool", "line_hash",
    "source_text", "source_bytes", "unchanged_text", "human", "notes", "review_missing", "review_text",
    "review_status", "review_ref", "review_issue", "review_bool", "review_human", "selection_ref", "qrel"])
def test_corrupt_bindings_fail(fixture, kind):
    f = fixture
    r = f.drafts["queries"][0]
    if kind == "pilot_parent":
        r["parent_query_id"] = "s0"
    elif kind == "duplicate_parent":
        r["parent_query_id"] = f.drafts["queries"][1]["parent_query_id"]
    elif kind == "old_id":
        r["query_id"] = "pp01"
    elif kind == "line_bool":
        r["source_line_1based"] = f.selection["sources"][0]["source_line_1based"] = True
    elif kind == "line_hash":
        r["source_line_sha256"] = f.selection["sources"][0]["source_line_sha256"] = "f" * 64
    elif kind == "source_text":
        f.selection["sources"][0]["source_text"] = "wrong"
    elif kind == "source_bytes":
        f.raw += b"\n"
    elif kind == "unchanged_text":
        r["query_text"] = "source 25"
    elif kind == "human":
        r["human_validated"] = True
    elif kind == "notes":
        r["source_context_notes"] = None
    elif kind == "review_missing":
        f.review["per_query"].pop()
    elif kind == "review_text":
        f.review["per_query"][0]["query_text_sha256"] = "f" * 64
    elif kind == "review_status":
        f.review["status"] = "pending"
    elif kind == "review_ref":
        f.review["drafts_ref"] = {"sha256": "f" * 64}
    elif kind == "review_issue":
        f.review["per_query"][0]["unresolved_semantic_issues"] = 1
    elif kind == "review_bool":
        f.review["per_query"][0]["unresolved_semantic_issues"] = False
    elif kind == "review_human":
        f.review["human_validated"] = True
    elif kind == "selection_ref":
        f.drafts["selection_ref"] = {"path": "wrong"}
    else:
        f.source.qrels["s25"] = {"unknown": 1}
    with pytest.raises(ValueError):
        derive(f)


@pytest.mark.parametrize("kind", ["unknown", "modified_query", "notes_query", "media_query", "modified_doc"])
def test_model_item_rejects_nonmembers_and_extra_fields(fixture, kind):
    f = fixture
    queries, qrels, _ = derive(f)
    dataset = m.RemainingDataset(f.source.root, {}, f.source.corpus, queries, qrels, "new", f.source)
    item = copy.deepcopy(queries[0])
    if kind == "unknown":
        item["id"] = "unbound"
    elif kind == "modified_query":
        item["text"] = "other"
    elif kind == "notes_query":
        item["notes"] = "gold"
    elif kind == "media_query":
        item["media"] = {"audio": "unbound.wav"}
    else:
        item = {**f.source.corpus[0], "text": "changed"}
    with pytest.raises(ValueError):
        dataset.model_item(item, text_only=True)


def test_qrels_are_copied_not_aliases(fixture):
    _, qrels, _ = derive(fixture)
    qrels["ep001"]["d0"] = 0
    assert fixture.source.qrels["s25"]["d0"] == 1

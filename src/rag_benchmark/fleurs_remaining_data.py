"""Verified remaining-304 queries over the immutable pilot's 743 document arrays.

No preparation, model load, cache write, or query-vector reuse occurs here. The
accepted review certifies agent meaning review, not human/gallery qrel truth.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from rag_benchmark.fleurs_pilot_data import load_verified_pilot, pinned_bytes, pinned_json, require
from rag_benchmark.models import stable_hash
from rag_benchmark.multimodal import Dataset

PROTOCOL = "fleurs-remaining-quality-data-v1"
REVIEW_PROTOCOL = "fleurs-remaining-independent-meaning-acceptance-v1"
IDS = [f"ep{i:03d}" for i in range(1, 305)]


def text_sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def review_binding(review, config, rows):
    require(review.get("protocol") == REVIEW_PROTOCOL and
            review.get("status") == "PASS_AGENT_MEANING_REVIEW", "remaining_review_status")
    require(review.get("drafts_ref") == config["drafts"] and
            review.get("selection_ref") == config["selection"], "remaining_review_refs")
    require(type(review.get("query_count")) is int and review["query_count"] == 304 and
            type(review.get("unresolved_semantic_issues")) is int and
            review["unresolved_semantic_issues"] == 0, "remaining_review_count")
    require(review.get("human_validated") is False and review.get("qrels_full_validation") is False,
            "remaining_review_limits")
    decisions = review.get("per_query", [])
    require(len(decisions) == 304 and [r.get("query_id") for r in decisions] == IDS,
            "remaining_review_population")
    for row, decision in zip(rows, decisions, strict=True):
        require(decision.get("parent_query_id") == row["parent_query_id"] and
                decision.get("source_line_sha256") == row["source_line_sha256"] and
                decision.get("query_text_sha256") == text_sha(row["query_text"]) and
                decision.get("decision") == "PASS" and
                type(decision.get("unresolved_semantic_issues")) is int and
                decision["unresolved_semantic_issues"] == 0, "remaining_query_review")


def derive_queries(config, pilot, selected, drafts, review, source_bytes):
    """Pure metadata derivation; exposed separately for synthetic fixtures."""
    source = pilot.source_dataset
    require(len(source.corpus) == 743 and len(source.queries) == 329 and
            len(pilot.dataset.queries) == 25, "remaining_source_population")
    require(selected.get("protocol") == "fleurs-paraphrase-remaining-source-selection-v1" and
            type(selected.get("remaining_count")) is int and selected["remaining_count"] == 304 and
            type(selected.get("preserved_pilot_count")) is int and selected["preserved_pilot_count"] == 25 and
            type(selected.get("total_target_count")) is int and selected["total_target_count"] == 329 and
            selected.get("human_validated") is False, "remaining_selection_scope")
    require(selected["preserved_pilot"] == pilot.provenance["drafts"] and
            selected["source"] == pilot.provenance["query_lineage"]["source_ref"],
            "remaining_source_authorities")
    require(drafts.get("selection_ref") == config["selection"] and
            type(drafts.get("count")) is int and drafts["count"] == 304 and
            drafts.get("human_validation_complete") is False and
            drafts.get("qrels_full_validation") is False, "remaining_draft_scope")
    rows, choices = drafts["queries"], selected["sources"]
    require(len(rows) == len(choices) == 304 and
            [r.get("query_id") for r in rows] == IDS == [r.get("query_id") for r in choices],
            "remaining_ids")
    source_rows = {row["id"]: row for row in source.queries}
    require(len(source_rows) == 329, "remaining_source_duplicates")
    prior = {row["parent_query_id"] for row in pilot.provenance["query_lineage"]["query_links"]}
    parents = [row["parent_query_id"] for row in rows]
    require(len(prior) == 25 and len(set(parents)) == 304 and
            set(parents) == set(source_rows) - prior and not set(parents) & prior,
            "remaining_exact_complement")
    require(not set(IDS) & {r["id"] for r in source.corpus + source.queries + pilot.dataset.queries},
            "remaining_new_namespace")
    require(hashlib.sha256(source_bytes).hexdigest() == selected["source"]["sha256"],
            "remaining_source_bytes")
    lines = source_bytes.splitlines(keepends=True)
    require(len(lines) == 329, "remaining_source_line_count")
    queries, qrels, lineage = [], {}, []
    for row, choice in zip(rows, choices, strict=True):
        for key in ("query_id", "parent_query_id", "source_line_1based", "source_line_sha256"):
            require(row[key] == choice[key], "remaining_selection_changed")
        number, parent, identifier = row["source_line_1based"], row["parent_query_id"], row["query_id"]
        require(type(number) is int and 1 <= number <= len(lines), "remaining_source_line_bounds")
        line = lines[number - 1]
        require(hashlib.sha256(line).hexdigest() == row["source_line_sha256"] and
                json.loads(line) == source_rows[parent] and choice["source_text"] == source_rows[parent]["text"],
                "remaining_source_line")
        text = row["query_text"]
        require(isinstance(text, str) and text.strip() and text != source_rows[parent]["text"] and
                row.get("agent_authored") is True and row.get("human_validated") is False,
                "remaining_query_scope")
        require(isinstance(row.get("semantic_preservation_notes"), str) and
                isinstance(row.get("source_context_notes"), str), "remaining_notes_type")
        require(parent in source.qrels and source.qrels[parent] and
                set(source.qrels[parent]) <= {r["id"] for r in source.corpus}, "remaining_qrel_group")
        queries.append({"id": identifier, "text": text, "media": {}})
        qrels[identifier] = copy.deepcopy(source.qrels[parent])
        lineage.append({"query_id": identifier, "parent_query_id": parent,
                        "source_line_sha256": row["source_line_sha256"], "query_text_sha256": text_sha(text)})
    review_binding(review, config, rows)
    return queries, qrels, lineage


@dataclass
class RemainingDataset(Dataset):
    source_dataset: Dataset

    def model_item(self, item, *, text_only=False):
        identifier = item.get("id")
        documents = {row["id"]: row for row in self.source_dataset.corpus}
        queries = {row["id"]: row for row in self.queries}
        if identifier in documents:
            require(item == documents[identifier], "remaining_document_changed")
            result = self.source_dataset.model_item(documents[identifier], text_only=text_only)
        else:
            require(identifier in queries and item == queries[identifier] and item.get("media") == {},
                    "remaining_query_membership")
            result = {"id": identifier, "text": item["text"], "media": {}}
        require(set(result) <= {"id", "text", "media"}, "remaining_model_field_leak")
        return result


@dataclass
class VerifiedRemaining:
    dataset: RemainingDataset
    pilot: object
    provenance: dict
    manifest_sha256: str

    @property
    def source_dataset(self):
        return self.pilot.source_dataset

    @property
    def document_vectors(self):
        return self.pilot.document_vectors

    @property
    def query_adapter_configs(self):
        return self.pilot.query_adapter_configs

    @property
    def adapter_specs(self):
        return self.pilot.adapter_specs

    def validate_query_adapter(self, channel, adapter, *, batch_size=1):
        self.pilot.validate_query_adapter(channel, adapter, batch_size=batch_size)


def load_verified_remaining(config_path, *, expected_sha256):
    path = Path(config_path).absolute()
    config_ref = {"path": str(path), "sha256": expected_sha256}
    config = pinned_json(config_ref)
    require(set(config) == {"protocol", "module_sha256", "pilot_manifest", "selection", "drafts", "meaning_review"}
            and config["protocol"] == PROTOCOL, "remaining_config_scope")
    require(hashlib.sha256(pinned_bytes({"path": str(Path(__file__).resolve()),
            "sha256": config["module_sha256"]})).hexdigest() == config["module_sha256"], "remaining_module")
    pilot = load_verified_pilot(config["pilot_manifest"]["path"], expected_sha256=config["pilot_manifest"]["sha256"])
    selected, drafts, review = (pinned_json(config[key]) for key in ("selection", "drafts", "meaning_review"))
    source_bytes = pinned_bytes(selected["source"])
    require(Path(selected["source"]["path"]).resolve() == pilot.source_dataset.root / "queries.jsonl",
            "remaining_source_path")
    queries, qrels, lineage = derive_queries(config, pilot, selected, drafts, review, source_bytes)
    provenance = {"protocol": PROTOCOL, "config": config_ref, "pilot_manifest": config["pilot_manifest"],
        "source_identity": pilot.source_dataset.identity, "preserved_pilot_identity": pilot.dataset.identity,
        "selection": config["selection"], "drafts": config["drafts"], "meaning_review": config["meaning_review"],
        "query_lineage": lineage, "query_rows_sha256": stable_hash(queries), "qrels_sha256": stable_hash(qrels),
        "corpus_sha256": stable_hash(pilot.source_dataset.corpus),
        "documents": copy.deepcopy(pilot.provenance["documents"]),
        "human_validated": False, "qrels_full_validation": False, "new_document_inference": 0,
        "old_query_vectors_reused": False, "cache_reads_are_inference": False}
    identity = stable_hash(provenance)
    require(identity not in (pilot.dataset.identity, pilot.source_dataset.identity), "remaining_identity")
    manifest = {"id": "fleurs-tr-remaining-paraphrases-v1", "track": "speech", "revision": identity,
        "split": "remaining_304_exploratory_source_groups", "counts": {"corpus": 743, "queries": 304},
        "text_provenance": copy.deepcopy(pilot.source_dataset.manifest["text_provenance"])}
    dataset = RemainingDataset(path.parent, manifest, copy.deepcopy(pilot.source_dataset.corpus), queries,
                               qrels, identity, pilot.source_dataset)
    dataset.require_text()
    # Recheck this new metadata chain after the inherited document verification.
    for key in ("pilot_manifest", "selection", "drafts", "meaning_review"):
        pinned_bytes(config[key])
    pinned_bytes(config_ref)
    return VerifiedRemaining(dataset, pilot, provenance, expected_sha256)

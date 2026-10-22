"""CPU-only, exact-source bridge for the exploratory 25-query FLEURS pilot.

Never calls encode, downloads a model, opens SQL, or alters an old cache. The
new Dataset is deliberately loaded only by this module: its corpus/media stay
in the verified source namespace, while its queries, qrels and identity are new.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass, field
import hashlib
import io
import json
import os
from pathlib import Path
import re
from typing import Any

import numpy as np

from . import models, multimodal, multimodal_models
from .models import stable_hash
from .multimodal import Dataset, TextEmbeddingAdapter, encoding_cache_identity, load_dataset
from .multimodal_models import make_multimodal_adapter

CHANNELS = ("G", "E", "N", "J")
QUERY_IDS = [f"pp{i:02d}" for i in range(1, 26)]
SCOPE = {
    "status": "exploratory_agent_reviewed_source_group_qrels",
    "human_validated": False,
    "human_validation_complete": False,
    "qrels_full_validation": False,
    "source_group_qrels_transferred": True,
    "semantic_benchmark_certified": False,
    "new_document_inference": 0,
    "old_query_vectors_reused": False,
    "cache_reads_are_inference": False,
}


class PilotDataError(ValueError):
    """Only fixed, non-content-bearing diagnostic codes leave this module."""


def require(condition: bool, code: str) -> None:
    if not condition:
        raise PilotDataError(code)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_bytes(path: Path) -> bytes:
    try:
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            data = handle.read()
            after = os.fstat(handle.fileno())
        require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
                == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), "file_changed_during_read")
        return data
    except OSError:
        raise PilotDataError("required_file_unavailable") from None


def ref(path: Path) -> dict:
    path = path.resolve()
    data = read_bytes(path)
    return {"path": str(path), "sha256": digest(data)}


def pinned_bytes(value: dict) -> bytes:
    require(isinstance(value, dict) and isinstance(value.get("path"), str)
            and re.fullmatch(r"[a-f0-9]{64}", str(value.get("sha256"))) is not None, "invalid_reference")
    data = read_bytes(Path(value["path"]))
    require(digest(data) == value["sha256"], "reference_digest_mismatch")
    return data


def pinned_json(value: dict) -> Any:
    try:
        return json.loads(pinned_bytes(value))
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise PilotDataError("invalid_json") from None


def fresh_bytes(path: Path, data: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def fresh_json(path: Path, value: Any) -> None:
    fresh_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode())


def code_refs() -> dict:
    return {name: ref(Path(module.__file__)) for name, module in
            (("models", models), ("multimodal", multimodal), ("multimodal_models", multimodal_models))}


def verify_code(seals: dict) -> None:
    require(seals == code_refs(), "source_code_changed")


def _adapter(channel: str, config: dict):
    # Constructors only. Both implementations load weights lazily in encode.
    if channel in {"G", "E"}:
        return TextEmbeddingAdapter("bge" if channel == "G" else "embeddinggemma", config)
    return make_multimodal_adapter(channel, config)


def _effective_config(channel: str, adapter) -> dict:
    return copy.deepcopy(adapter.embedder.config if channel in {"G", "E"} else adapter.config)


def _relative_ref(value: dict, root: Path) -> dict:
    path = Path(value["path"])
    return {"path": str((path if path.is_absolute() else root / path).resolve()), "sha256": value["sha256"]}


def _source(config: dict) -> tuple[Dataset, dict, dict]:
    require(config.get("protocol") == "fleurs-paraphrase-pilot-data-config-v1", "config_protocol")
    verify_code(config["source_code"])
    source = load_dataset(config["source_root"])
    report = pinned_json(config["source_report"])
    require(source.identity == config["source_identity"] == report["dataset"]["identity"], "source_identity")
    require(source.track == "speech" and len(source.corpus) == 743 and len(source.queries) == 329,
            "source_population")
    source.require_text()
    require(source.manifest["text_provenance"] ==
            {"source": "whisper_asr", "query_independent": True, "gold_fields_used": False}, "source_asr_provenance")
    require(report["dataset"]["revision"] == source.manifest["revision"], "source_revision")
    inventory = pinned_json(config["source_cache_inventory"])
    matches = [row for row in inventory["collections"] if row.get("frozen_identity") == source.identity]
    require(len(matches) == 1, "source_inventory_unique")
    collection = matches[0]
    require(Path(collection["dataset_root"]).resolve() == source.root, "source_inventory_root")
    names = [row["filename"] for row in collection["source_files"]]
    require(len(names) == len(set(names)) and set(names) ==
            {"dataset.json", "assets.jsonl", "corpus.jsonl", "queries.jsonl", "qrels.jsonl"}, "source_file_scope")
    for row in collection["source_files"]:
        require(Path(row["path"]).resolve() == source.root / row["filename"], "source_file_path")
        data = pinned_bytes(row)
        require(len(data) == row["size"], "source_file_size")
    expected_media = {(source.root / item["media"]["audio"]).resolve() for item in source.corpus}
    require(len(expected_media) == 743 and all(set(item.get("media", {})) == {"audio"}
            for item in source.corpus), "source_audio_scope")
    media = collection["media"]
    require(len(media) == 743 and {Path(row["path"]).resolve() for row in media} == expected_media,
            "source_media_inventory")
    for row in media:
        require(len(pinned_bytes(row)) == row["size"], "source_media_size")
    return source, report, collection


def _queries(config: dict, source: Dataset) -> tuple[list[dict], dict, dict]:
    drafts = pinned_json(config["drafts"])
    review = pinned_json(config["meaning_review"])
    repository = Path(config["repository_root"]).resolve()
    draft_root = Path(config["drafts"]["path"]).resolve().parent
    require(drafts.get("protocol") == "fleurs-paraphrase-pilot-agent-drafts-v2", "draft_protocol")
    require(drafts.get("human_validated") is False and drafts.get("human_validation_complete") is False,
            "draft_human_status")
    require(review.get("status") == "PASS_AGENT_DELTA_REVIEW"
            and review["summary"]["agent_reviewed_queries"] == 25
            and review["summary"]["remaining_identified_semantic_issues"] == 0
            and review["summary"]["human_validated"] is False
            and review["summary"]["human_validation_complete"] is False
            and review["summary"]["qrels_full_validation"] is False, "meaning_review_scope")
    # The V2 review explicitly inherits the unchanged twenty from V1; pin that
    # whole chain rather than silently treating a five-query delta as 25 reviews.
    for row in review["refs"].values():
        pinned_bytes(_relative_ref(row, repository))
    require(_relative_ref(review["refs"]["queries-draft-v2.json"], repository) == config["drafts"],
            "review_draft_binding")
    selected_ref = _relative_ref(drafts["selected_sources_ref"], draft_root)
    selected = pinned_json(selected_ref)
    require(_relative_ref(review["refs"]["selected-sources.json"], repository) == selected_ref,
            "selection_review_binding")
    rows = drafts["queries"]
    require(drafts["query_count"] == 25 and [row["query_id"] for row in rows] == QUERY_IDS, "pilot_query_ids")
    require([row["query_id"] for row in selected["selected_sources"]] == QUERY_IDS, "selection_query_ids")
    source_ref = _relative_ref(drafts["source_ref"], repository)
    source_bytes = pinned_bytes(source_ref)
    require(source_bytes == read_bytes(source.root / "queries.jsonl"), "source_query_file_binding")
    lines = source_bytes.splitlines(keepends=True)
    source_queries = {row["id"]: row for row in source.queries}
    require(len({row["parent_query_id"] for row in rows}) == 25, "duplicate_parent_group")
    require(not set(QUERY_IDS) & {row["id"] for row in source.corpus + source.queries}, "new_query_namespace")
    output, qrels, links = [], {}, []
    for row, choice in zip(rows, selected["selected_sources"], strict=True):
        for field_name in ("parent_query_id", "source_line_1based", "source_refhash", "candidate_positive_group"):
            require(row[field_name] == choice[field_name], "selected_source_changed")
        parent = row["parent_query_id"]
        require(parent in source_queries, "parent_query_unknown")
        line_number = row["source_line_1based"]
        require(type(line_number) is int and 1 <= line_number <= len(lines), "source_line_bounds")
        line = lines[line_number - 1]
        require(digest(line) == row["source_refhash"] and json.loads(line) == source_queries[parent],
                "source_line_binding")
        group = row["candidate_positive_group"]
        require(group["parent_group_id"] == parent and group["status"] ==
                "source_group_reference_only_not_new_qrels_validation", "qrel_transfer_scope")
        positives = sum(grade > 0 for grade in source.qrels[parent].values())
        require(positives == group["source_declared_positive_count"], "qrel_transfer_count")
        text = row["query_text"]
        require(isinstance(text, str) and bool(text.strip()) and text != source_queries[parent].get("text"),
                "paraphrase_text_required")
        identifier = row["query_id"]
        output.append({"id": identifier, "text": text, "media": {}, "metadata": {
            "parent_query_id": parent, "group_id": parent,
            "label_scope": "exploratory_transferred_source_group", "human_validated": False}})
        qrels[identifier] = copy.deepcopy(source.qrels[parent])
        links.append({"query_id": identifier, "parent_query_id": parent,
                      "source_line_sha256": row["source_refhash"], "positive_count": positives})
    return output, qrels, {"source_ref": source_ref, "selected_sources": selected_ref, "query_links": links}


def _vectors(config: dict, source: Dataset, report: dict, collection: dict) -> tuple[dict, dict, dict]:
    arrays, evidence, configurations = {}, {}, {}
    ordered_ids = [row["id"] for row in source.corpus]
    cache_root = Path(config["cache_root"]).resolve()
    for channel in CHANNELS:
        spec = report["adapter_specs"]["channels"][channel]
        settings = spec["embedder"]["config"] if channel in {"G", "E"} else spec["config"]
        adapter = _adapter(channel, settings)
        require(adapter.identity == spec["identity"], "current_adapter_identity")
        require((adapter.embedder._model if channel in {"G", "E"} else adapter._model) is None,
                "unexpected_model_load")
        dimension = 1024 if channel == "G" else 768
        require(adapter.dimension == dimension, "channel_dimension")
        key = stable_hash({"encoding": encoding_cache_identity(source, adapter, "document"),
                           "text_only": channel in {"G", "E"}})
        require(key == report["channels"][channel]["document_encoding"]["identity"], "document_cache_identity")
        entries = [row for row in collection["sets"] if row["channel"] == channel and row["role"] == "document"]
        require(len(entries) == 1, "document_set_unique")
        entry = entries[0]
        require(entry["set_id"] == key and entry["kind"] == "dense" and entry["dimension"] == dimension
                and entry["item_ids"] == ordered_ids, "document_set_binding")
        provenance = entry["provenance"]
        require(provenance["adapter_spec"] == spec and provenance["frozen_dataset_identity"] == source.identity
                and provenance["cache_identity"] == key, "document_provenance")
        directory = cache_root / "vectors" / key
        intervals = [(start, min(start + 128, 743)) for start in range(0, 743, 128)]
        require(len(entry["blocks"]) == len(intervals), "cache_block_count")
        require({p.name for p in directory.glob("*.npy")} ==
                {f"{start:09d}-{stop:09d}.npy" for start, stop in intervals}, "cache_block_inventory")
        blocks, receipts = [], []
        for row, (start, stop) in zip(entry["blocks"], intervals, strict=True):
            path = directory / f"{start:09d}-{stop:09d}.npy"
            require(Path(row["path"]).resolve() == path and row["item_ids"] == ordered_ids[start:stop]
                    and row["shape"] == [stop - start, dimension], "cache_block_order")
            raw = pinned_bytes(row)
            require(len(raw) == row["size"], "cache_block_size")
            values = np.load(io.BytesIO(raw), allow_pickle=False)
            require(values.dtype == np.dtype("float32") and values.shape == (stop - start, dimension),
                    "cache_array_shape_dtype")
            require(bool(np.isfinite(values).all()) and bool(np.allclose(np.linalg.norm(values, axis=1),
                    1, atol=2e-3, rtol=2e-3)), "cache_array_normalization")
            usage_path = path.with_suffix(".usage.json")
            usage_ref = ref(usage_path)
            usage = pinned_json(usage_ref)
            require(usage["rows"] == stop - start, "cache_usage_rows")
            if channel in {"N", "J"}:
                require(usage.get("reported") is True and usage.get("context_limit") == 8192
                        and usage.get("dtype") == settings["dtype"], "native_usage_context")
                require([x["id"] for x in usage["items"]] == ordered_ids[start:stop], "native_usage_order")
                require(all(type(x.get("expanded_tokens")) is int and 0 < x["expanded_tokens"] <= 8192
                            for x in usage["items"]), "native_context_overflow")
            require(usage.get("truncated_text_items", 0) == 0
                    and not any(x.get("truncated") is True for x in usage.get("items", [])), "recorded_truncation")
            blocks.append(values)
            receipts.append({"start": start, "stop": stop, "array": {"path": str(path), "sha256": row["sha256"]},
                             "usage": usage_ref, "ordered_ids_sha256": stable_hash(ordered_ids[start:stop])})
        array = np.concatenate(blocks, axis=0)
        array.setflags(write=False)
        arrays[channel] = array
        configurations[channel] = _effective_config(channel, adapter)
        evidence[channel] = {"adapter_identity": adapter.identity, "cache_identity": key,
            "dimension": dimension, "dtype": "float32", "rows": 743,
            "ordered_ids_sha256": stable_hash(ordered_ids), "array_sha256": digest(array.tobytes()),
            "blocks": receipts, "source_adapter_spec": copy.deepcopy(spec),
            "effective_adapter_config": configurations[channel],
            "historical_file_checksum_attested": provenance.get("historical_file_checksum_attested", False),
            "previous_snapshot_payload_hash_verified": True,
            "processing_diagnostics": "per_item_native_context_verified" if channel in {"N", "J"}
                                      else "legacy_per_item_diagnostics_unavailable_source_guard_preserved",
            "new_document_inference": 0, "cache_reads_are_inference": False}
    return arrays, evidence, configurations


@dataclass
class PilotDataset(Dataset):
    source_dataset: Dataset = field(repr=False)

    def model_item(self, item: dict, *, text_only: bool = False) -> dict:
        # Preserve the existing strict id/text/media allowlist. The new namespace
        # references verified source media explicitly, never by path traversal or
        # falsely assigning the old complete dataset identity to new queries.
        candidates = {row["id"]: row for row in self.source_dataset.corpus}
        if item.get("id") in candidates:
            require(item == candidates[item["id"]], "pilot_candidate_changed")
        else:
            require(item in self.queries and not item.get("media"), "pilot_query_changed")
        return self.source_dataset.model_item(item, text_only=text_only)


@dataclass
class VerifiedPilot:
    dataset: PilotDataset
    source_dataset: Dataset
    document_vectors: dict[str, np.ndarray]
    adapter_specs: dict
    query_adapter_configs: dict
    provenance: dict
    manifest_sha256: str

    def validate_query_adapter(self, channel: str, adapter, *, batch_size: int = 1) -> None:
        require(channel in CHANNELS and batch_size == 1, "query_adapter_scope")
        expected = copy.deepcopy(self.query_adapter_configs[channel])
        expected["batch_size"] = batch_size
        require(_effective_config(channel, adapter) == expected, "query_adapter_config_changed")
        require(adapter.identity == _adapter(channel, expected).identity, "query_adapter_identity_changed")


def _derive(config: dict, output_root: Path) -> tuple[PilotDataset, Dataset, dict, dict, dict]:
    source, report, collection = _source(config)
    queries, qrels, query_lineage = _queries(config, source)
    arrays, vector_evidence, configs = _vectors(config, source, report, collection)
    provenance = {"protocol": "fleurs-paraphrase-document-reuse-v1", "scope": SCOPE,
        "source_identity": source.identity, "source_report": config["source_report"],
        "source_cache_inventory": config["source_cache_inventory"], "drafts": config["drafts"],
        "meaning_review": config["meaning_review"], "source_code": config["source_code"],
        "corpus_rows_sha256": stable_hash(source.corpus), "source_qrels_sha256": stable_hash(source.qrels),
        "query_lineage": query_lineage, "documents": vector_evidence,
        "query_rows_sha256": stable_hash(queries), "transferred_qrels_sha256": stable_hash(qrels)}
    identity = stable_hash({"protocol": "fleurs-paraphrase-pilot-dataset-v1", "provenance": provenance})
    require(identity != source.identity, "pilot_identity_must_be_new")
    manifest = {"id": "fleurs-tr-paraphrase-pilot-v1", "track": "speech", "revision": identity,
        "split": "exploratory_pilot_source_test_groups", "text_provenance": copy.deepcopy(source.manifest["text_provenance"]),
        "counts": {"corpus": 743, "queries": 25, "qrels": sum(map(len, qrels.values()))},
        "source_dataset_identity": source.identity, "pilot_scope": copy.deepcopy(SCOPE)}
    dataset = PilotDataset(output_root, manifest, copy.deepcopy(source.corpus), queries, qrels, identity, source)
    dataset.require_text()
    return dataset, source, arrays, provenance, configs


def prepare_verified_pilot(config_path: Path | str, config_sha256: str, output_dir: Path | str) -> dict:
    """One CPU preparation; refuse an existing namespace rather than overwrite."""
    config_ref = {"path": str(Path(config_path).resolve()), "sha256": config_sha256}
    config = pinned_json(config_ref)
    output = Path(output_dir).resolve()
    require(output == Path(config["output_directory"]).resolve(), "output_directory_binding")
    require(not output.exists(), "output_namespace_exists")
    dataset, _, _, provenance, _ = _derive(config, output)
    # All input and array checks precede creation of the new private namespace.
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    fresh_json(output / "dataset.json", dataset.manifest)
    fresh_bytes(output / "queries.jsonl", b"".join((json.dumps(row, ensure_ascii=False) + "\n").encode()
                                                for row in dataset.queries))
    fresh_bytes(output / "qrels.jsonl", b"".join((json.dumps({"query_id": q, "corpus_id": d, "relevance": grade})
                + "\n").encode() for q in QUERY_IDS for d, grade in dataset.qrels[q].items()))
    fresh_json(output / "provenance.json", provenance)
    manifest = {"protocol": "fleurs-paraphrase-pilot-prepared-manifest-v1", "config": config_ref,
        "module_source": ref(Path(__file__)), "dataset_identity": dataset.identity, "scope": SCOPE,
        "files": {name: ref(output / name) for name in ("dataset.json", "queries.jsonl", "qrels.jsonl", "provenance.json")}}
    fresh_json(output / "manifest.json", manifest)
    return ref(output / "manifest.json")


def load_verified_pilot(manifest_path: Path | str, *, expected_sha256: str | None = None) -> VerifiedPilot:
    """Read-only verification, including old blocks; never regenerates a miss."""
    path = Path(manifest_path).resolve()
    manifest_ref = ref(path) if expected_sha256 is None else {"path": str(path), "sha256": expected_sha256}
    manifest = pinned_json(manifest_ref)
    require(manifest["protocol"] == "fleurs-paraphrase-pilot-prepared-manifest-v1" and manifest["scope"] == SCOPE,
            "prepared_manifest_scope")
    require(manifest["module_source"] == ref(Path(__file__)), "bridge_source_changed")
    config = pinned_json(manifest["config"])
    require(path.parent == Path(config["output_directory"]).resolve(), "prepared_namespace_binding")
    files = manifest["files"]
    require(set(files) == {"dataset.json", "queries.jsonl", "qrels.jsonl", "provenance.json"}, "prepared_file_scope")
    for name, value in files.items():
        require(Path(value["path"]).resolve() == path.parent / name, "prepared_file_path")
        pinned_bytes(value)
    dataset, source, arrays, provenance, configs = _derive(config, path.parent)
    require(dataset.identity == manifest["dataset_identity"], "prepared_identity_changed")
    require(pinned_json(files["dataset.json"]) == dataset.manifest
            and pinned_json(files["provenance.json"]) == provenance, "prepared_derivation_changed")
    require([json.loads(line) for line in pinned_bytes(files["queries.jsonl"]).splitlines()] == dataset.queries,
            "prepared_queries_changed")
    expected_qrels = [{"query_id": q, "corpus_id": d, "relevance": grade}
                      for q in QUERY_IDS for d, grade in dataset.qrels[q].items()]
    require([json.loads(line) for line in pinned_bytes(files["qrels.jsonl"]).splitlines()] == expected_qrels,
            "prepared_qrels_changed")
    return VerifiedPilot(dataset, source, arrays,
        {c: copy.deepcopy(provenance["documents"][c]["source_adapter_spec"]) for c in CHANNELS},
        configs, provenance, manifest_ref["sha256"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = prepare_verified_pilot(args.config, args.config_sha256, args.output_directory)
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "error_type": type(exc).__name__,
                          "code": str(exc) if isinstance(exc, PilotDataError) else "unexpected_preparation_failure"}))
        return 1
    print(json.dumps({"status": "PASS_CPU_PREPARATION", "manifest_sha256": result["sha256"],
                      "documents": 743, "queries": 25, "new_model_calls": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

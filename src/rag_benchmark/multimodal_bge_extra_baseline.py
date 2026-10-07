"""Read-only proof of original G vectors; no model loading or encoding fallback.

This bridge verifies whole-function galleries and JavaScript shared segments
against original source identities, ordered vectors, rankings and all metrics.
"""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from .models import MODEL_DEFAULTS, stable_hash
from .multimodal import TextEmbeddingAdapter, encoding_cache_identity, _channel_identity, graded_metrics, aggregate_metrics, load_dataset
from .multimodal_code import SegmentedCodeAdapter
from .multimodal_code_prefix_ablation import _read_vectors, _cosine_rankings

class VerifiedGBaseline:
    def __init__(self, dataset, baseline_run, *, baseline_cache):
        if dataset.track != "code" or dataset.manifest.get("metadata", {}).get("language") not in {"go", "java", "javascript", "php", "python", "ruby"}:
            raise ValueError("This bridge requires one of the six frozen code galleries")
        row_seal = self._dataset_rows_seal(dataset)
        frozen = load_dataset(dataset.root)
        if dataset.identity != frozen.identity or stable_hash({"manifest": dataset.manifest, "corpus": dataset.corpus, "queries": dataset.queries, "qrels": dataset.qrels}) != stable_hash({"manifest": frozen.manifest, "corpus": frozen.corpus, "queries": frozen.queries, "qrels": frozen.qrels}):
            raise ValueError("Frozen dataset files and supplied source/query rows differ")
        baseline_run = Path(baseline_run)
        report_bytes = (baseline_run / "report.json").read_bytes()
        report = json.loads(report_bytes)
        if report["dataset"]["identity"] != dataset.identity or report["scope"] != "frozen_collection":
            raise ValueError("Baseline report belongs to a different or partial frozen dataset")
        if report["evaluated_query_count"] != len(dataset.queries) or report["channels"]["G"]["status"] != "completed":
            raise ValueError("Baseline G is incomplete")
        query_ids = [item["id"] for item in dataset.queries]
        if report["configuration"]["query_ids"] != query_ids:
            raise ValueError("Baseline report query ID order differs")
        javascript = dataset.manifest["metadata"]["language"] == "javascript"
        spec = report["adapter_specs"]["channels"]["G"]
        base_config = spec["encoder" if javascript else "embedder"]["config"]
        for key in ("model_id", "revision", "dimension"):
            if base_config[key] != MODEL_DEFAULTS["bge"][key]:
                raise ValueError("Unapproved baseline model contract")
        if int(base_config.get("max_length", 8192)) != 8192:
            raise ValueError("Baseline context budget differs")
        ordinary = SegmentedCodeAdapter("bge", base_config) if javascript else TextEmbeddingAdapter("bge", base_config)
        if ordinary.identity != spec["identity"]:
            raise ValueError("Baseline adapter implementation/identity cannot be reconstructed")
        channel_identity = _channel_identity(dataset, "G", {"G": ordinary})
        if channel_identity != report["channels"]["G"]["identity"]:
            raise ValueError("Baseline channel identity differs")
        cache = Path(baseline_cache) if baseline_cache is not None else baseline_run / "cache"
        rank_path = cache / "rankings" / stable_hash({"identity": channel_identity, "top_k": 100}) / "rankings.json"
        rank_bytes = rank_path.read_bytes()
        ranking_payload = json.loads(rank_bytes)
        if ranking_payload["identity"] != channel_identity or ranking_payload["query_ids"] != query_ids:
            raise ValueError("Baseline ranking payload identity/query order differs")
        saved_rankings = ranking_payload["rankings"]
        self._ids = [item["id"] for item in dataset.corpus]
        self._documents_seal = stable_hash([dataset.model_item(item) for item in dataset.corpus])
        self.query_inputs_seal = stable_hash([dataset.model_item(item) for item in dataset.queries])
        self._offsets = None
        seals = []
        self._records = []
        if javascript:
            self.segmenter = ordinary.segmenter
            chunks, summaries, self._offsets = [], [], []
            for item in dataset.corpus:
                self._offsets.append(len(chunks))
                parts, summary = self.segmenter.partition(dataset.model_item(item))
                if "".join(part["text"] for part in parts) != item["text"]:
                    raise ValueError("Shared segments dropped source content")
                chunks.extend(parts)
                summaries.append(summary)
            direct = cache / "direct" / channel_identity
            segmentation_bytes = (direct / "shared-segmentation.json").read_bytes()
            saved = json.loads(segmentation_bytes)
            if saved != {"identity": self.segmenter.identity, "protocol": self.segmenter.protocol, "functions": summaries}:
                raise ValueError("Baseline shared segmentation differs from frozen sources")
            def blocks(items, role):
                arrays = []
                for start in range(0, len(items), 32):
                    batch = items[start:start + 32]
                    formatted = [ordinary.encoder.format_document(item) if role == "document"
                                 else ordinary.encoder.format_query(item["text"]) for item in batch]
                    key = stable_hash({"adapter": ordinary.identity, "role": role, "formatted_inputs": formatted})
                    arrays.append(_read_vectors(direct / "vectors" / role / (key + ".npy"), len(batch), 1024, seals))
                return np.concatenate(arrays)
            self._records = [{"id": part["id"], "source_id": part["function_id"], "formatted_text": ordinary.encoder.format_document(part),
                    "chunk_index": part["chunk_index"], "char_start": part["char_start"], "char_end": part["char_end"],
                    "byte_start": part["byte_start"], "byte_end": part["byte_end"], "text_sha256": part["text_sha256"]}
                for part in chunks]
            self._documents = blocks(chunks, "document")
            baseline_queries = blocks(dataset.queries, "query")
            usage = report["channels"]["G"]
            expected_counts = {"chunk_count": len(chunks), "original_function_count": len(dataset.corpus),
                "source_characters": sum(len(item["text"]) for item in dataset.corpus),
                "covered_source_characters": sum(len(item["text"]) for item in chunks),
                "source_utf8_bytes": sum(len(item["text"].encode()) for item in dataset.corpus),
                "covered_source_utf8_bytes": sum(len(item["text"].encode()) for item in chunks)}
            if any(usage.get(key) != value for key, value in expected_counts.items()):
                raise ValueError("Baseline source coverage/count metrics differ")
        else:
            def blocks(items, role):
                key = stable_hash({"encoding": encoding_cache_identity(dataset, ordinary, role), "text_only": True})
                if key != report["channels"]["G"][role + "_encoding"]["identity"]:
                    raise ValueError("Baseline encoding identity differs")
                arrays = [_read_vectors(cache / "vectors" / key / f"{start:09d}-{min(start+128,len(items)):09d}.npy",
                            min(128, len(items)-start), 1024, seals) for start in range(0, len(items), 128)]
                return np.concatenate(arrays)
            self._records = [{"id": item["id"], "source_id": item["id"], "formatted_text": ordinary.embedder.format_document(dataset.model_item(item))} for item in dataset.corpus]
            self._documents = blocks(dataset.corpus, "document")
            baseline_queries = blocks(dataset.queries, "query")
        exclusions = [dataset.excluded_ids(query) for query in dataset.queries]
        recomputed = _cosine_rankings(self._documents, baseline_queries, self._ids, self._offsets, exclusions)
        if len(saved_rankings) != len(recomputed):
            raise ValueError("Baseline ranking query coverage differs")
        for actual, saved in zip(recomputed, saved_rankings):
            if [row.get("rank") for row in saved] != list(range(1, len(saved) + 1)) or [row["id"] for row in actual] != [row["id"] for row in saved] or not np.allclose(
                    [row["score"] for row in actual], [row["score"] for row in saved], atol=2e-6, rtol=2e-6):
                raise ValueError("Baseline rankings do not match saved document/query vectors")
        metric_rows = [graded_metrics([row["id"] for row in ranking], dataset.qrels.get(query["id"], {}))
                       for query, ranking in zip(dataset.queries, recomputed)]
        metrics = aggregate_metrics(metric_rows)
        cells = [cell for cell in report["cells"] if cell["variant_id"] == "code__g__none"]
        for cell in cells:
            expected_cell = stable_hash({**report["configuration"], "variant": cell["variant_id"],
                "mode": cell["budget_mode"], "candidate_k": cell["candidate_k"],
                "channel_identities": {"G": channel_identity}, "reranker": None, "reranker_prompts": None})
            if cell["identity"] != expected_cell:
                raise ValueError("Baseline cell identity differs")
        if not cells or any(cell["status"] != "completed" or cell["completed_queries"] != len(dataset.queries)
                            or set(cell["metrics"]) != set(metrics) or any(not np.isclose(cell["metrics"][key], value, atol=1e-10, rtol=0)
                            for key, value in metrics.items()) for cell in cells):
            raise ValueError("Baseline G metrics do not match verified rankings")
        if len({row["id"] for row in self._records}) != len(self._records):
            raise ValueError("Baseline document/chunk row IDs are not unique")
        if (baseline_run / "report.json").read_bytes() != report_bytes or rank_path.read_bytes() != rank_bytes:
            raise ValueError("Baseline report/rankings changed during verification")
        if self._dataset_rows_seal(dataset) != row_seal:
            raise ValueError("Supplied source/query rows changed during verification")
        latest = load_dataset(dataset.root)
        if latest.identity != dataset.identity:
            raise ValueError("Frozen dataset changed during verification")
        self._queries = baseline_queries
        self._query_records = [{"id": item["id"], "source_id": item["id"], "formatted_text":
            (ordinary.encoder if javascript else ordinary.embedder).format_query(item["text"])} for item in dataset.queries]
        self._config = dict(base_config)
        self._provenance = {"baseline_report_sha256": hashlib.sha256(report_bytes).hexdigest(),
            "baseline_rankings_sha256": hashlib.sha256(rank_bytes).hexdigest(),
            "baseline_adapter_identity": ordinary.identity, "baseline_channel_identity": channel_identity,
            "config": self._config, "dataset_identity": dataset.identity,
            "source_input_hash": self._documents_seal, "query_input_hash": self.query_inputs_seal,
            "formatted_document_hash": stable_hash(self._records), "formatted_query_hash": stable_hash(self._query_records),
            "source_count": len(self._ids), "document_rows": len(self._documents), "query_rows": len(self._queries),
            "vector_files": seals, "verified_metric_count": len(metrics),
            "segmentation_identity": ordinary.segmenter.identity if javascript else None,
            "source_aggregation": "maximum_chunk_score" if javascript else "whole_function",
            "fresh_encoder_calls": 0, "ranking_and_metrics_revalidated": True}
        self._identity = stable_hash({"condition": "verified-original-g-baseline-v1", "provenance": self._provenance,
            "implementation": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
        self._documents.flags.writeable = False
        self._queries.flags.writeable = False

        self._dataset = dataset
        self._dataset_identity = dataset.identity
        self._row_seal = row_seal
        self._private_seal = self._state_seal()

    @staticmethod
    def _dataset_rows_seal(dataset):
        return stable_hash({"manifest": dataset.manifest, "corpus": dataset.corpus,
            "queries": dataset.queries, "qrels": dataset.qrels})

    def _state_seal(self):
        return stable_hash({"records": self._records, "query_records": self._query_records,
            "config": self._config, "provenance": self._provenance, "ids": self._ids,
            "offsets": self._offsets, "identity": self._identity,
            "documents_sha256": hashlib.sha256(self._documents.tobytes()).hexdigest(),
            "queries_sha256": hashlib.sha256(self._queries.tobytes()).hexdigest()})

    def validate_for_reuse(self):
        """Required handoff guard; never reloads an encoder or regenerates data."""
        if self._state_seal() != self._private_seal:
            raise ValueError("Verified baseline state changed")
        if self._dataset_rows_seal(self._dataset) != self._row_seal:
            raise ValueError("Verified supplied source/query rows changed")
        if load_dataset(self._dataset.root).identity != self._dataset_identity:
            raise ValueError("Verified frozen dataset files changed")

    @property
    def records(self):
        return copy.deepcopy(self._records)

    @property
    def query_records(self):
        return copy.deepcopy(self._query_records)

    @property
    def config(self):
        return copy.deepcopy(self._config)

    @property
    def provenance(self):
        return copy.deepcopy(self._provenance)

    @property
    def ids(self):
        return list(self._ids)

    @property
    def offsets(self):
        return copy.deepcopy(self._offsets)

    @property
    def documents(self):
        return self._documents.view()

    @property
    def queries(self):
        return self._queries.view()

    @property
    def identity(self):
        return self._identity

    def verified_inputs(self, role):
        self.validate_for_reuse()
        if role == "document":
            return self.records, self.documents
        if role == "query":
            return self.query_records, self.queries
        raise ValueError("Explicit document/query role required")

    def public_summary(self):
        self.validate_for_reuse()
        return copy.deepcopy({key: value for key, value in self._provenance.items() if key != "vector_files"})

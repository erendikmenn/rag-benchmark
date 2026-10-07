"""Isolated pinned EG2 code-query prompt ablation; no ordinary baseline mutation."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess

import numpy as np
from pathlib import Path

from . import models, multimodal
from .models import DenseEmbedder, MODEL_DEFAULTS, _snapshot, stable_hash
from .multimodal import (TextEmbeddingAdapter, load_dataset, run_matrix, encoding_cache_identity,
                         _channel_identity, _validate_vectors, graded_metrics, aggregate_metrics)
from .multimodal_code import SegmentedCodeAdapter, SharedCodeSegmenter, _atomic_json
from .multimodal import _atomic_array

QUERY_PREFIX = "task: code retrieval | query: "
DOCUMENT_PREFIX = "title: none | text: "
_SOURCE_HASH = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class CodePrefixEmbedder(DenseEmbedder):
    def __init__(self, config=None):
        super().__init__("embeddinggemma", config)
        if self.config["revision"] != MODEL_DEFAULTS["embeddinggemma"]["revision"]:
            raise ValueError("Code-prefix ablation requires the verified pinned revision")
        evidence_path = _snapshot(self.config, download=False) / "config_sentence_transformers.json"
        evidence_bytes = evidence_path.read_bytes()
        prompts = json.loads(evidence_bytes)["prompts"]
        if prompts.get("CodeRetrieval") != QUERY_PREFIX or prompts.get("Document") != DOCUMENT_PREFIX:
            raise ValueError("Pinned snapshot does not verify the exact code-query/document prompt contract")
        self.contract = {"query_prefix": QUERY_PREFIX, "document_prefix": DOCUMENT_PREFIX,
                         "title_policy": "none_only", "overlength": "error_without_truncation",
                         "evidence_sha256": hashlib.sha256(evidence_bytes).hexdigest()}
        self.identity = stable_hash({"condition": "eg2-code-prefix-ablation-v1", "base": self.identity,
                                     "contract": self.contract, "implementation": _SOURCE_HASH,
                                     "dense_implementation": hashlib.sha256(Path(models.__file__).read_bytes()).hexdigest()})
        self._config_seal = stable_hash(self.config)

    def format_query(self, question):
        return QUERY_PREFIX + question

    def format_document(self, item):
        if item.get("title"):
            raise ValueError("Code-prefix ablation requires untitled frozen code sources")
        return DOCUMENT_PREFIX + item["text"]

    def embed_documents(self, documents):
        raise ValueError("Fresh document encoding is forbidden in the code-prefix ablation")

    def _encode(self, texts):
        if stable_hash(self.config) != self._config_seal:
            raise ValueError("Code-prefix encoder configuration changed after identity creation")
        return super()._encode(texts)


class CodePrefixTextAdapter(TextEmbeddingAdapter):
    def __init__(self, config=None):
        self.embedder = CodePrefixEmbedder(config)
        self.dimension = self.embedder.dimension
        self.identity = self.embedder.identity
        self.prompt_identity = self.embedder.contract.copy()
        self.text_only = True


class CodePrefixSegmentedAdapter(SegmentedCodeAdapter):
    def __init__(self, config=None):
        segmenter = SharedCodeSegmenter()
        super().__init__("embeddinggemma", config, segmenter=segmenter)
        self.encoder = CodePrefixEmbedder(self.encoder.config)
        self.identity = stable_hash({"condition": "eg2-code-prefix-shared-segments-max-v1",
                                     "parent_adapter": self.identity, "encoder": self.encoder.identity,
                                     "implementation": _SOURCE_HASH})
        self._encoder_config_seal = stable_hash(self.encoder.config)



def _read_vectors(path, count, dimension, file_seals):
    if not path.is_file():
        raise ValueError("Verified baseline vector block unavailable")
    data = path.read_bytes()
    file_seals.append({"path": str(path.resolve()), "sha256": hashlib.sha256(data).hexdigest()})
    vectors = np.load(io.BytesIO(data), allow_pickle=False)
    _validate_vectors(vectors, count, dimension)
    if vectors.dtype != np.float32:
        raise ValueError("Baseline vector cache must use recorded float32 score representation")
    return vectors


def _cosine_rankings(documents, queries, ids, offsets, exclusions, top_k=100):
    output = []
    identifiers = np.asarray(ids)
    for start in range(0, len(queries), 32):
        scores = queries[start:start + 32] @ documents.T
        if offsets is not None:
            scores = np.maximum.reduceat(scores, offsets, axis=1)
        for index, row in enumerate(scores, start):
            eligible = np.asarray([i for i, identifier in enumerate(ids) if identifier not in exclusions[index]], dtype=np.int64) if exclusions[index] else np.arange(len(ids))
            if len(eligible) > top_k:
                boundary = np.partition(row[eligible], -top_k)[-top_k]
                eligible = eligible[row[eligible] >= boundary]
            order = np.lexsort((identifiers[eligible], -row[eligible]))
            output.append([{"id": ids[i], "score": float(row[i]), "rank": rank}
                           for rank, i in enumerate(eligible[order[:top_k]], 1)])
    return output


class VerifiedDocumentBridge(SegmentedCodeAdapter):
    """Only query vectors can be newly encoded; documents are sealed baseline arrays."""
    def __init__(self, dataset, baseline_run, config, *, baseline_cache=None):
        baseline_run = Path(baseline_run)
        report_bytes = (baseline_run / "report.json").read_bytes()
        report = json.loads(report_bytes)
        if report["dataset"]["identity"] != dataset.identity or report["scope"] != "frozen_collection":
            raise ValueError("Baseline report belongs to a different or partial frozen dataset")
        if report["evaluated_query_count"] != len(dataset.queries) or report["channels"]["E"]["status"] != "completed":
            raise ValueError("Baseline E is incomplete")
        query_ids = [item["id"] for item in dataset.queries]
        if report["configuration"]["query_ids"] != query_ids:
            raise ValueError("Baseline report query ID order differs")
        javascript = dataset.manifest["metadata"]["language"] == "javascript"
        spec = report["adapter_specs"]["channels"]["E"]
        base_config = spec["encoder" if javascript else "embedder"]["config"]
        for key in ("model_id", "revision", "dimension"):
            if base_config[key] != MODEL_DEFAULTS["embeddinggemma"][key]:
                raise ValueError("Unapproved baseline model contract")
        if any(base_config[key] != config[key] for key in ("dtype", "device")):
            raise ValueError("Baseline and query dtype/device must match")
        if int(base_config.get("max_length", 8192)) != 8192:
            raise ValueError("Baseline context budget differs")
        ordinary = SegmentedCodeAdapter("embeddinggemma", base_config) if javascript else TextEmbeddingAdapter("embeddinggemma", base_config)
        if ordinary.identity != spec["identity"]:
            raise ValueError("Baseline adapter implementation/identity cannot be reconstructed")
        channel_identity = _channel_identity(dataset, "E", {"E": ordinary})
        if channel_identity != report["channels"]["E"]["identity"]:
            raise ValueError("Baseline channel identity differs")
        cache = Path(baseline_cache) if baseline_cache is not None else baseline_run / "cache"
        rank_path = cache / "rankings" / stable_hash({"identity": channel_identity, "top_k": 100}) / "rankings.json"
        rank_bytes = rank_path.read_bytes()
        ranking_payload = json.loads(rank_bytes)
        if ranking_payload["identity"] != channel_identity or ranking_payload["query_ids"] != query_ids:
            raise ValueError("Baseline ranking payload identity/query order differs")
        saved_rankings = ranking_payload["rankings"]
        self.ids = [item["id"] for item in dataset.corpus]
        self.documents_seal = stable_hash([dataset.model_item(item) for item in dataset.corpus])
        self.query_inputs_seal = stable_hash([dataset.model_item(item) for item in dataset.queries])
        self.offsets = None
        seals = []
        if javascript:
            self.segmenter = ordinary.segmenter
            chunks, summaries, self.offsets = [], [], []
            for item in dataset.corpus:
                self.offsets.append(len(chunks))
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
                    arrays.append(_read_vectors(direct / "vectors" / role / (key + ".npy"), len(batch), 768, seals))
                return np.concatenate(arrays)
            self.documents = blocks(chunks, "document")
            baseline_queries = blocks(dataset.queries, "query")
            usage = report["channels"]["E"]
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
                if key != report["channels"]["E"][role + "_encoding"]["identity"]:
                    raise ValueError("Baseline encoding identity differs")
                arrays = [_read_vectors(cache / "vectors" / key / f"{start:09d}-{min(start+128,len(items)):09d}.npy",
                            min(128, len(items)-start), 768, seals) for start in range(0, len(items), 128)]
                return np.concatenate(arrays)
            self.documents = blocks(dataset.corpus, "document")
            baseline_queries = blocks(dataset.queries, "query")
        exclusions = [dataset.excluded_ids(query) for query in dataset.queries]
        recomputed = _cosine_rankings(self.documents, baseline_queries, self.ids, self.offsets, exclusions)
        if len(saved_rankings) != len(recomputed):
            raise ValueError("Baseline ranking query coverage differs")
        for actual, saved in zip(recomputed, saved_rankings):
            if [row.get("rank") for row in saved] != list(range(1, len(saved) + 1)) or [row["id"] for row in actual] != [row["id"] for row in saved] or not np.allclose(
                    [row["score"] for row in actual], [row["score"] for row in saved], atol=2e-6, rtol=2e-6):
                raise ValueError("Baseline rankings do not match saved document/query vectors")
        metric_rows = [graded_metrics([row["id"] for row in ranking], dataset.qrels.get(query["id"], {}))
                       for query, ranking in zip(dataset.queries, recomputed)]
        metrics = aggregate_metrics(metric_rows)
        cells = [cell for cell in report["cells"] if cell["variant_id"] == "code__e__none"]
        for cell in cells:
            expected_cell = stable_hash({**report["configuration"], "variant": cell["variant_id"],
                "mode": cell["budget_mode"], "candidate_k": cell["candidate_k"],
                "channel_identities": {"E": channel_identity}, "reranker": None, "reranker_prompts": None})
            if cell["identity"] != expected_cell:
                raise ValueError("Baseline cell identity differs")
        if not cells or any(cell["status"] != "completed" or cell["completed_queries"] != len(dataset.queries)
                            or set(cell["metrics"]) != set(metrics) or any(not np.isclose(cell["metrics"][key], value)
                            for key, value in metrics.items()) for cell in cells):
            raise ValueError("Baseline E metrics do not match verified rankings")
        for key, value in config.items():
            if key == "batch_size":
                if type(value) is not int or value < 1:
                    raise ValueError("Explicit query batch size must be positive")
            elif key == "max_length":
                if value != int(base_config.get("max_length", 8192)):
                    raise ValueError("Query context budget differs from baseline")
            elif key not in base_config or value != base_config[key]:
                raise ValueError("Query settings differ from verified baseline")
        self.encoder = CodePrefixEmbedder({**base_config, **config})
        self.dimension = 768
        self.provenance = {"baseline_report_sha256": hashlib.sha256(report_bytes).hexdigest(),
            "baseline_rankings_sha256": hashlib.sha256(rank_bytes).hexdigest(),
            "baseline_adapter_identity": ordinary.identity, "baseline_channel_identity": channel_identity,
            "baseline_encoder_config": base_config, "document_prefix": DOCUMENT_PREFIX,
            "dataset_identity": dataset.identity, "source_input_hash": self.documents_seal,
            "precomputed_documents": True, "fresh_document_encoder_calls": 0,
            "document_rows": len(self.documents), "source_count": len(self.ids),
            "vector_files": seals, "ranking_and_metrics_revalidated": True,
            "segmentation_identity": ordinary.segmenter.identity if javascript else None}
        self.identity = stable_hash({"condition": "verified-documents-code-query-v1", "provenance": self.provenance,
                                     "query_encoder": self.encoder.identity, "implementation": _SOURCE_HASH,
                                     "engine_implementation": hashlib.sha256(Path(multimodal.__file__).read_bytes()).hexdigest()})
        self._encoder_config_seal = stable_hash(self.encoder.config)
        self.last_usage = {}
        self.documents.flags.writeable = False

    def rank(self, documents, queries, top_k, cache_dir, exclusions=None):
        if stable_hash(documents) != self.documents_seal or stable_hash(queries) != self.query_inputs_seal:
            raise ValueError("Frozen source/query inputs differ from verified baseline")
        if stable_hash(self.encoder.config) != self._encoder_config_seal:
            raise ValueError("Query encoder configuration changed")
        vectors, usage = self._cached_vectors(queries, "query", Path(cache_dir))
        self.unload()
        rankings = _cosine_rankings(self.documents, vectors, self.ids, self.offsets, exclusions, top_k)
        return rankings, {"document_encoding": {"precomputed": True, "new_items": 0,
                            "total_items": len(self.documents), "cache_hits": len(self.documents),
                            "inference_seconds": None, "fresh_document_encoder_calls": 0},
                          "query_encoding": usage, "exact": True,
                          "condition": "verified_documents_code_query_prefix"}

    def _cached_vectors(self, items, role, cache_dir):
        if role != "query":
            raise ValueError("Fresh document encoding is forbidden")
        directory = Path(cache_dir) / "vectors" / "query"
        arrays, fresh = [], 0
        for start in range(0, len(items), 32):
            batch = items[start:start + 32]
            content = [self.encoder.format_query(item["text"]) for item in batch]
            key = stable_hash({"adapter": self.identity, "role": "query", "formatted_inputs": content})
            path = directory / (key + ".npy")
            checksum = path.with_suffix(".sha256.json")
            if path.exists():
                data = path.read_bytes()
                if not checksum.exists() or json.loads(checksum.read_text()).get("sha256") != hashlib.sha256(data).hexdigest():
                    raise ValueError("Code query cache checksum missing or changed")
                vectors = np.load(io.BytesIO(data), allow_pickle=False)
            else:
                vectors = np.asarray(self.encoder._encode(content), dtype=np.float32)
                _validate_vectors(vectors, len(batch), self.dimension)
                _atomic_array(path, vectors)
                _atomic_json(checksum, {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
                fresh += len(batch)
            _validate_vectors(vectors, len(batch), self.dimension)
            arrays.append(vectors)
        return np.concatenate(arrays), {"total_items": len(items), "new_items": fresh,
            "cache_hits": len(items) - fresh, "cache_reads_are_inference": False,
            "inference_seconds": None}

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--baseline-run", type=Path, help="Verified completed ordinary E run report")
    parser.add_argument("--baseline-cache", type=Path, help="Explicit existing ordinary E cache directory")
    parser.add_argument("--verify-baseline-only", action="store_true", help="CPU verification only; no query encoding")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=None, help="Optional explicit override; omitted inherits baseline batch size")
    args = parser.parse_args(argv)
    dataset = load_dataset(args.dataset)
    if dataset.track != "code" or args.batch_size is not None and args.batch_size < 1:
        parser.error("Requires frozen code dataset and positive batch size")
    if "ablations" not in args.run_dir.parts:
        parser.error("Run directory must be a separate ablations directory")
    if subprocess.run(["git", "check-ignore", "--quiet", str(args.run_dir.resolve())],
                      capture_output=True).returncode != 0:
        parser.error("Raw output requires a git-ignored run directory")
    if any(item.get("title") for item in dataset.corpus):
        parser.error("Requires untitled code sources")
    config = {"device": args.device, "dtype": args.dtype}
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    language = dataset.manifest.get("metadata", {}).get("language")
    if language not in {"go", "java", "javascript", "php", "python", "ruby"}:
        parser.error("Requires explicit frozen CodeSearchNet language metadata")
    javascript = language == "javascript"
    if args.baseline_run is None or args.baseline_cache is None:
        _atomic_json(args.run_dir / "ablation-readiness.json", {"status": "unsupported",
            "reason": "verified_baseline_run_and_cache_required", "fresh_document_encoder_calls": 0})
        return 1
    try:
        adapter = VerifiedDocumentBridge(dataset, args.baseline_run, config, baseline_cache=args.baseline_cache)
    except (ValueError, FileNotFoundError, KeyError, TypeError) as exc:
        _atomic_json(args.run_dir / "ablation-readiness.json", {"status": "unsupported",
            "reason": "baseline_verification_failed", "error_type": type(exc).__name__,
            "fresh_document_encoder_calls": 0})
        return 1
    protocol = {"condition": "eg2-code-query-prefix-ablation-v1", "main_denominator_contribution": 0,
                "ordinary_baseline": False, "dataset_identity": dataset.identity,
                "adapter_identity": adapter.identity, "encoder_config":
                    adapter.encoder.config,
                "prompt_contract": adapter.encoder.contract,
                "source_representation": "shared_segments_max" if javascript else "strict_whole_function",
                "implementation_sha256": _SOURCE_HASH, "precomputed_documents": True,
                "fresh_document_encoder_calls": 0, "baseline_provenance": adapter.provenance}
    protocol_path = args.run_dir / "ablation-protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        parser.error("Existing ablation protocol differs; use a separate run directory")
    _atomic_json(protocol_path, protocol)
    if args.verify_baseline_only:
        _atomic_json(args.run_dir / "ablation-readiness.json", {"status": "baseline_verified",
            "fresh_document_encoder_calls": 0, "query_encoder_calls": 0})
        return 0
    result = run_matrix(args.dataset, args.run_dir, adapters={"E": adapter}, requested_channels=["E"],
                        rerankers={}, candidate_grid=(100,), budget_modes=("total",))
    cells = [cell for cell in result.get("cells", []) if cell.get("variant_id") == "code__e__none"]
    success = len(cells) == 1 and all(cell.get("status") == "completed"
        and cell.get("query_count") == len(dataset.queries)
        and cell.get("completed_queries") == len(dataset.queries) for cell in cells)
    _atomic_json(args.run_dir / "ablation-readiness.json", {"status": "completed" if success else "failed",
        "fresh_document_encoder_calls": 0})
    return 0 if success else 1





if __name__ == "__main__":
    raise SystemExit(main())

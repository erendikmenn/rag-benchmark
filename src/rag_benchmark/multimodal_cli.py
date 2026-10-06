"""Explicit preparation, offline execution and public multimodal reports."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def prepare_generation(config: dict, *, cache_dir: Path = Path(".cache/generation-models")) -> None:
    """Explicit model acquisition; normal benchmark execution remains offline."""
    from huggingface_hub import hf_hub_download
    from .cli import file_sha256
    from .models import validate_revision
    validate_revision(config)
    for prefix in ("model", "projector"):
        target = Path(config[prefix + "_path"])
        if not target.exists():
            blob = Path(hf_hub_download(config["model_id"], filename=target.name,
                revision=config["revision"], cache_dir=str(cache_dir)))
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(blob.resolve(), target)
        if file_sha256(target) != config[prefix + "_sha256"]:
            raise ValueError(f"{prefix} checksum mismatch")


def generation_server_command(binary: Path, config: dict, *, context: int = 8192, parallel: int = 1,
                              device: str = "mps") -> list[str]:
    from .cli import file_sha256
    if not binary.is_file() or context < 256 or parallel < 1 or device not in {"mps", "cpu"}:
        raise ValueError("Existing server binary, positive parallelism and context >=256 are required")
    for prefix in ("model", "projector"):
        path = Path(config[prefix + "_path"])
        if not path.is_file() or file_sha256(path) != config[prefix + "_sha256"]:
            raise ValueError(f"{prefix} checksum mismatch or missing file")
    return [str(binary.resolve()), "--model", str(Path(config["model_path"]).resolve()),
        "--mmproj", str(Path(config["projector_path"]).resolve()), "--alias", config["model"],
        "--host", "127.0.0.1", "--port", "8081", "--ctx-size", str(context * parallel),
        "--n-gpu-layers", "99" if device == "mps" else "0", "--parallel", str(parallel),
        "--jinja", "--no-context-shift", "--reasoning", "off",
        "--chat-template-kwargs", '{"enable_thinking":false}'] + (
        [] if device == "mps" else ["--no-mmproj-offload", "--threads", "4", "--threads-batch", "4"])


def export_report(run_dir: Path, output_dir: Path) -> dict:
    """Publish aggregate metrics/IDs only; never copy local errors, media or prompts."""
    report = json.loads((run_dir / "report.json").read_text())
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("report.json", "matrix.csv"):
        shutil.copyfile(run_dir / filename, output_dir / filename)
    dataset = report["dataset"]
    lines = ["# Multimodal retrieval results", "",
        f"Dataset: **{dataset['id']}**, revision `{dataset['revision']}`.",
        f"Scope: **{report['scope']}**; {report['evaluated_query_count']:,} evaluated queries "
        f"out of {dataset['query_count']:,}, searching {dataset['corpus_count']:,} candidates.", "",
        "Only completed cells below have measured scores. Planned, unsupported and failed "
        "families remain visible in matrix.csv. A completed collection is not the complete seven-track benchmark.", "",
        "| Method | Budget | Rerank K | Hit@1 | Hit@5 | Recall@5 | MRR@10 | nDCG@10 |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for cell in report["cells"]:
        if cell.get("status") != "completed" or not cell.get("primary"):
            continue
        metrics = cell.get("metrics", {})
        def number(key):
            value = metrics.get(key)
            return f"{value:.4f}" if isinstance(value, (int, float)) else "—"
        lines.append(f"| {cell['variant_id']} | {cell['budget_mode']} | {cell['candidate_k'] or '—'} "
                     f"| {number('hit@1')} | {number('hit@5')} | {number('recall@5')} "
                     f"| {number('mrr@10')} | {number('ndcg@10')} |")
    lines += ["", "Hit@K measures whether at least one labelled relevant item was retrieved. "
        "Recall@K measures the fraction of all labelled relevant items retrieved. "
        "These can differ substantially for multi-positive datasets.", "",
        "This report measures retrieval against dataset labels, not generated-answer correctness. "
        "Cached results are never reported as new model inference latency. "
        "Full configuration, channel timing and coverage are in report.json.", ""]
    (output_dir / "report.md").write_text("\n".join(lines))
    return report


def write_report_index(root: Path) -> None:
    """List frozen runs without pooling languages, protocols, or partial tests."""
    lines = ["# Measured multimodal runs", "",
        "Each row is a separate run. Query and candidate counts across language/protocol views "
        "overlap and must not be added as independent data. Partial runs and synthetic model preflights "
        "are not full-split accuracy results. The 1,284-family registry remains planned scope.", "",
        "| Run | Scope | Queries | Candidates | Completed primary cells | Failed cells |",
        "|---|---|---:|---:|---:|---:|"]
    for path in sorted(root.glob("*/report.json")):
        report = json.loads(path.read_text())
        cells = report.get("cells", [])
        completed = sum(x.get("primary", False) and x.get("status") == "completed" for x in cells)
        failed = sum(x.get("status") == "failed" for x in cells)
        name = path.parent.name
        lines.append(f"| [{name}]({name}/report.md) | {report['scope']} | "
                     f"{report['evaluated_query_count']:,} / {report['dataset']['query_count']:,} | "
                     f"{report['dataset']['corpus_count']:,} | {completed} | {failed} |")
    lines += ["", "A failed comparison does not erase completed methods in the same run. "
              "Each report retains all coverage states and records the actual model configuration. "
              "Hit/Recall measure source retrieval, not generated-answer correctness.", ""]
    root.mkdir(parents=True, exist_ok=True)
    (root / "README.md").write_text("\n".join(lines))


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in {"prepare", "preflight", "asr", "analyze"}:
        if argv[0] == "prepare":
            from .multimodal_data import main as delegated
        elif argv[0] == "asr":
            from .multimodal_asr import main as delegated
        elif argv[0] == "analyze":
            from .multimodal_analysis import main as delegated
        else:
            from .multimodal_models import main as delegated
        return delegated(argv[1:]) or 0
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("registry")
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("run")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--channels", default="N,S")
    p.add_argument("--rerankers", default="none")
    p.add_argument("--device", default="mps")
    p.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    p.add_argument("--batch-size", type=int, default=1, help="Native media and specialist batch size")
    p.add_argument("--text-batch-size", type=int, default=8)
    p.add_argument("--dimension", type=int, choices=(128, 256, 512, 768), default=768)
    p.add_argument("--vision-budget", type=int, choices=(70, 140, 280, 560, 1120), default=280)
    p.add_argument("--specialist-text-overflow", choices=("error", "truncate_to_model_limit"), default="error",
                   help="Explicit native specialist query limit policy; clipping is recorded in diagnostics.")
    p.add_argument("--code-overlength", choices=("error", "shared_segments_max"), default="error",
                   help="Separate source-complete function-level aggregation protocol for long code.")
    p.add_argument("--candidate-k", default="20,50,100")
    p.add_argument("--budget-modes", default="per_channel,total")
    p.add_argument("--cache-dir", type=Path, default=Path(".cache/multimodal"))
    p.add_argument("--limit", type=int)
    p = sub.add_parser("describe")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-new", type=int)
    p.add_argument("--caption-language", choices=("auto", "en", "tr", "fr"), default="auto",
                   help="Auto uses the frozen dataset language when declared, otherwise English.")
    p = sub.add_parser("prepare-generation")
    p.add_argument("--include-runtime", action="store_true")
    p = sub.add_parser("serve")
    p.add_argument("--binary", type=Path)
    p.add_argument("--context", type=int, default=8192, help="Context tokens per slot; no context shifting")
    p.add_argument("--parallel", type=int, default=1)
    p.add_argument("--device", choices=("mps", "cpu"), default="mps")
    p = sub.add_parser("export")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("index")
    p.add_argument("--root", type=Path, default=Path("reports/multimodal"))
    args = parser.parse_args(argv)
    try:
        if args.command == "registry":
            from .multimodal import matrix_registry
            rows = matrix_registry()
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                for row in rows:
                    writer.writerow({key: "|".join(map(str, value)) if isinstance(value, list) else value
                                     for key, value in row.items()})
            print(json.dumps({"planned_method_families": len(rows), "output": str(args.output)}))
        elif args.command == "export":
            export_report(args.run_dir, args.output)
            print(args.output / "report.md")
        elif args.command == "index":
            write_report_index(args.root)
            print(args.root / "README.md")
        elif args.command == "describe":
            from .multimodal_generation import LocalMultimodalGenerator, prepare_described_view
            if args.max_new is not None and args.max_new < 1:
                raise ValueError("--max-new must be positive")
            language = args.caption_language
            if language == "auto":
                manifest = json.loads((args.dataset / "dataset.json").read_text())
                language = manifest.get("metadata", {}).get("language", "en").split("_")[0]
                if language not in {"en", "tr", "fr"}:
                    raise ValueError("Dataset language needs an explicit supported --caption-language")
            result = prepare_described_view(args.dataset, args.output,
                LocalMultimodalGenerator({"caption_language": language}), max_new=args.max_new)
            print(json.dumps({key: result[key] for key in ("status", "counts", "ready_candidates", "new_descriptions") if key in result}))
        elif args.command == "prepare-generation":
            from .multimodal_generation import E4B_CONFIG
            prepare_generation(E4B_CONFIG)
            if args.include_runtime:
                from .cli import install_runtime
                install_runtime(Path.cwd())
            print("Pinned generation model and projector verified.")
        elif args.command == "serve":
            from .cli import install_runtime
            from .multimodal_generation import E4B_CONFIG
            binary = args.binary or install_runtime(Path.cwd())
            command = generation_server_command(binary, E4B_CONFIG, context=args.context,
                                                parallel=args.parallel, device=args.device)
            print("Starting verified local multimodal Gemma server on 127.0.0.1:8081", flush=True)
            return subprocess.call(command)
        else:
            from .multimodal import BGETextReranker, RERANKERS, TextEmbeddingAdapter, load_dataset, run_matrix
            from .multimodal_models import make_multimodal_adapter
            dataset = load_dataset(args.dataset)
            channels = args.channels.split(",")
            names = args.rerankers.split(",")
            if not set(names) <= set(RERANKERS):
                raise ValueError("Unknown reranker name")
            common = {"device": args.device, "dtype": args.dtype, "batch_size": args.batch_size}
            text_config = {**common, "batch_size": args.text_batch_size}
            eg = {**common, "dimension": args.dimension, "vision_budget": args.vision_budget}
            adapters = {}
            segmenter = None
            if args.code_overlength == "shared_segments_max":
                if dataset.track != "code":
                    raise ValueError("--code-overlength shared_segments_max is a code-only protocol")
                from .multimodal_code import SharedCodeSegmenter
                segmenter = SharedCodeSegmenter()
            for channel in channels:
                if channel in {"G", "E"} and segmenter is not None:
                    from .multimodal_code import SegmentedCodeAdapter
                    adapters[channel] = SegmentedCodeAdapter("bge" if channel == "G" else "embeddinggemma",
                                                             text_config, segmenter=segmenter)
                elif channel == "G":
                    adapters[channel] = TextEmbeddingAdapter("bge", text_config)
                elif channel == "E":
                    adapters[channel] = TextEmbeddingAdapter("embeddinggemma", text_config)
                    if args.dimension != 768:
                        raise ValueError("E dimension truncation requires an explicit precomputed representation; N/J support --dimension.")
                elif channel in {"N", "J"}:
                    adapters[channel] = make_multimodal_adapter(channel, eg)
                elif channel == "S":
                    expert = {"photo": "siglip2", "environment_audio": "clap", "video": "clip_video"}.get(dataset.track)
                    if dataset.track == "document":
                        from .multimodal_colqwen import ColQwenAdapter
                        adapters[channel] = ColQwenAdapter(common)
                    elif expert:
                        from .multimodal_specialists import SpecialistAdapter
                        adapters[channel] = SpecialistAdapter(expert, {
                            **common, "text_overflow_policy": args.specialist_text_overflow})
            rerankers = {}
            if "laya_text" in names:
                from .multimodal_generation import MultimodalLayaReranker
                rerankers["laya_text"] = MultimodalLayaReranker({"device": args.device})
            if "bge_reranker_text" in names:
                rerankers["bge_reranker_text"] = BGETextReranker(common)
            if "gemma4_relevance" in names:
                from .multimodal_generation import LocalMultimodalGenerator
                generator = LocalMultimodalGenerator()
                generator.preflight()
                rerankers["gemma4_relevance"] = generator
            if segmenter is not None:
                from .multimodal_code_rerank import SharedCodeReranker
                rerankers = {name: SharedCodeReranker(adapter, segmenter)
                             for name, adapter in rerankers.items()}
            result = run_matrix(args.dataset, args.run_dir, adapters=adapters, rerankers=rerankers,
                requested_channels=channels, candidate_grid=tuple(map(int, args.candidate_k.split(","))),
                budget_modes=tuple(args.budget_modes.split(",")), query_limit=args.limit,
                cache_dir=args.cache_dir, progress_callback=lambda row: print(json.dumps(row), flush=True))
            print(json.dumps({"scope": result["scope"], "queries": result["evaluated_query_count"],
                              "run_dir": str(args.run_dir)}))
        return 0
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())

"""Preparation may use the network; benchmark inference is local only."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tarfile
import urllib.request

from .runner import read_config, report, run

RUNTIME_VERSION = "b11451"
RUNTIME_ARCHIVE = "llama-b11451-bin-macos-arm64.tar.gz"
RUNTIME_SHA256 = "b9a95d2749ca3be8b245502f9c2d5e4e833e8775e08a9d8f20c6b0de071cba30"


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def install_runtime(root):
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise ValueError("Bundled runtime setup is for Apple Silicon. Install llama.cpp for your platform and use serve --binary.")
    directory = root / ".cache" / "llama.cpp"
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / RUNTIME_ARCHIVE
    if not archive.exists():
        temporary = archive.with_suffix(".partial")
        urllib.request.urlretrieve(f"https://github.com/ggml-org/llama.cpp/releases/download/{RUNTIME_VERSION}/{RUNTIME_ARCHIVE}", temporary)
        temporary.replace(archive)
    if file_sha256(archive) != RUNTIME_SHA256:
        raise ValueError("Runtime archive checksum mismatch")
    with tarfile.open(archive) as f:
        f.extractall(directory, filter="data")
    binaries = list(directory.rglob("llama-server"))
    if len(binaries) != 1:
        raise ValueError(f"Expected one llama-server binary, found {binaries}")
    return binaries[0]


def prepare_models(config, names, include_runtime=False, root=None):
    from .models import fetch_model
    for name in names:
        if name in ("bge", "embeddinggemma", "laya"):
            spec = config["laya"] if name == "laya" else config["embeddings"][name]
            print(f"Preparing {name}: {spec['model_id']}@{spec['revision']}", flush=True)
            print(fetch_model(spec), flush=True)
        elif name == "generator":
            from huggingface_hub import hf_hub_download
            spec = config["generator"]
            target = Path(spec["model_path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                print(f"Downloading {spec['model_id']}@{spec['revision']}", flush=True)
                downloaded = Path(hf_hub_download(spec["model_id"], spec["filename"], revision=spec["revision"], cache_dir=str(target.parent / ".hub")))
                # Link the already downloaded immutable blob instead of duplicating 14 GB.
                os.link(downloaded.resolve(), target)
            print("Verifying generator SHA-256…", flush=True)
            if file_sha256(target) != spec["model_sha256"]:
                raise ValueError("Generator file checksum mismatch; it will not be served")
            print(target, flush=True)
        else:
            raise ValueError(f"Unknown model: {name}")
    if include_runtime:
        print(install_runtime(root), flush=True)


def doctor(config):
    from .models import _snapshot, local_base_url
    result = {"platform": platform.platform(), "architecture": platform.machine(), "packages": {}, "models": {}, "dataset": {}}
    for name in ("numpy", "bm25s", "torch", "sentence_transformers", "laya"):
        result["packages"][name] = importlib.util.find_spec(name) is not None
    for name, spec in {**config.get("embeddings", {}), "laya": config["laya"]}.items():
        try:
            path = _snapshot(spec, download=False)
            result["models"][name] = {"snapshot_found": True, "path": str(path), "inference_verified": False}
        except Exception as exc:
            result["models"][name] = {"snapshot_found": False, "error": str(exc)}
    if result["packages"]["torch"]:
        import torch
        result["mps_available"] = torch.backends.mps.is_available()
    path = Path(config["dataset"]["path"]) / "manifest.json"
    result["dataset"] = json.loads(path.read_text()) if path.exists() else {"prepared": False}
    result["generator"] = {"file_exists": Path(config["generator"]["model_path"]).is_file(), "base_url": local_base_url(config["generator"]["base_url"])}
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Offline Turkish RAG benchmark")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "prepare-models", "doctor", "run", "serve"):
        p = sub.add_parser(name)
        p.add_argument("--config", type=Path, default=Path("configs/ragturk.toml"))
        if name == "prepare-models":
            p.add_argument("--only", default="bge,embeddinggemma,laya,generator")
            p.add_argument("--include-runtime", action="store_true")
        if name == "run":
            p.add_argument("--split", choices=("dev", "test"), default="dev")
            p.add_argument("--limit", type=int)
            p.add_argument("--variants", help="Comma-separated variant IDs; default: all ten")
            p.add_argument("--run-dir", type=Path, required=True)
            p.add_argument("--retrieval-only", action="store_true")
        if name == "serve":
            p.add_argument("--binary", type=Path)
            p.add_argument("--port", type=int, default=8080)
    p = sub.add_parser("report")
    p.add_argument("--run-dir", type=Path, required=True)
    p = sub.add_parser("export-report")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "report":
            report(args.run_dir)
            print(args.run_dir / "report.md")
            return 0
        if args.command == "export-report":
            from .export import export_report
            export_report(args.run_dir, args.output_dir)
            print(args.output_dir / "report.md")
            return 0
        config = read_config(args.config)
        root = args.config.resolve().parent.parent
        if args.command == "prepare":
            from .data import prepare_ragturk
            spec = config["dataset"]
            manifest = prepare_ragturk(Path(spec["path"]), dev_size=spec["dev_size"], seed=spec["seed"], revision=spec["revision"])
            print(json.dumps(manifest, ensure_ascii=False, indent=2))
        elif args.command == "prepare-models":
            prepare_models(config, args.only.split(","), args.include_runtime, root)
        elif args.command == "doctor":
            doctor(config)
        elif args.command == "run":
            run(config, args.run_dir, args.split, args.limit, args.variants.split(",") if args.variants else None, args.retrieval_only)
            print(args.run_dir / "report.md")
        elif args.command == "serve":
            binary = args.binary.resolve() if args.binary else install_runtime(root)
            spec = config["generator"]
            if not Path(spec["model_path"]).is_file():
                raise ValueError("Generator is not prepared. Run prepare-models --only generator first.")
            if file_sha256(spec["model_path"]) != spec["model_sha256"]:
                raise ValueError("Generator SHA-256 mismatch")
            command = [str(binary), "--model", spec["model_path"], "--alias", spec["model"], "--host", "127.0.0.1", "--port", str(args.port), "--ctx-size", "8192", "--n-gpu-layers", "99", "--parallel", "1", "--jinja", "--no-context-shift"]
            print("Starting verified local Gemma server", flush=True)
            return subprocess.call(command)
        return 0
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

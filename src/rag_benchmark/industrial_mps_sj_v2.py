"""Isolated industrial S/J completion; preparation is read-only, execution is explicit.

Only the execute subcommand can construct a model, after sealed-input validation
and acquisition of the existing GPU lock. No ordinary benchmark CLI is invoked.
The output is sealed S/J top-100 rankings, not a claim of 126 completed None cells.
"""
from __future__ import annotations

import argparse
import ast
import fcntl
import hashlib
import importlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import platform
import stat
import struct
import sys
import tarfile
import time
import types

import numpy as np

from . import vidore_cache_recovery_v2 as frozen

VERSION = "industrial-l4-bgen-local-mps-sj-v2"
BASE = "work/" + VERSION
DELTA = BASE + "/actual-v2"
DATASET = "vidore-v3-industrial-en"
DATASET_REVISION = "e26c864724f5dd71a3d7d739272d95637764cee9"
DATASET_IDENTITY = "e1f5feea6658a945a0cf02eeafe7fdb0f4ed03c5505d1382dc543c2b67452742"
MODEL_REVISIONS = {
    "J": ("google/embeddinggemma-2", "914f7f89142e33e77833254d9c9b90c3cef7303b"),
    "S": ("vidore/colqwen2.5-v0.2", "dcbe8d9cede518bce830488364ba0e40c873645b"),
    "S_base": ("vidore/colqwen2.5-base", "92908120384b7a2110c5beda3ab29cbdb2c08e49"),
}
INHERITED_IDS = {
    "B": "7302fd49ca08d62141a44f79fc614a9ff51a371f96f4e24101fab981b39960d3",
    "G": "08d93e52db50863191c0bcc70251294b674d96b62d7856e93e98bf3f1af11816",
    "E": "53f3ebc277ca9c490f470356f7530fcd2d634e747c2af190b70ae7af359c8f43",
    "N": "12a884f9a81e1fdf8e4d0b4e477806822d2de6c6cf0314400888fe135d7cfdc9",
}
REPORT_HASHES = {
    "B": "575b1667122e939cd7aaf3e321cd0c2dce5f532d9f7995c5c830d13d8b1dfd5a",
    "G": "462af1d463d0a7bbd60df4527cc3f26c419799ae0ee93b2fdd25f918c952522b",
    "E": "55d5c02479abae4ab9f77b75464233b014f4c55cfc4e2de8493deb7a991fb194",
    "N": "8e7702d38b80a23dae90d7dd83968e5efce6f6c3ec85e0d9683d512164e429d2",
}
CONFIG_HASHES = {
    "J": {"config.json": "b8f1e9931b57fbc054acdb445c41765d55b0074c58d145fa82839941ad1b5bb3",
          "modules.json": "3d02572a0455b832de67fb8e63a54981bc7e8b46e337c95e917bd8122a533bfd"},
    "S": {"modules.json": "2b09d169b37950de5f82f835269c17883c6773d5325e6d8c6546ecf86dcc4477",
          "adapter_config.json": "b19199f316cd17bb5198239e7d4d462657b694b1b386ef54486f32fd36495107"},
    "S_base": {"config.json": "926d8dfab0b8cb7cc5382f073609365851faec995eb21013414aea893938d8fc",
               "model.safetensors.index.json": "1005cf1c7dd7e0fb6ab269c6e4f1dd33122d4fce1aae697aac44bce960626cbe"},
}
ARCHIVE_SHA = "1fa79e268572bb36a1ff8f03cbd8142846deea3609bdd24f0868fc7ec9eba382"
ARCHIVE = "work/colab/checkpoints/industrial-recovery/" + DATASET + "-" + ARCHIVE_SHA + ".tar"
HARDWARE = {
    "B": {"hardware": "CPU", "dtype": None, "stage": "inherited-L4-stage-BM25"},
    **{c: {"hardware": "NVIDIA L4", "dtype": "bfloat16", "stage": "inherited-CUDA"} for c in "GEN"},
    "S": {"hardware": "local MPS", "dtype": "bfloat16", "stage": "new-MPS-MaxSim-float32"},
    "J": {"hardware": "local MPS", "dtype": "bfloat16", "stage": "new-MPS-CPU-dot-float32"},
}
MODULES = ("models", "retrieval", "multimodal", "multimodal_models", "multimodal_colqwen",
           "vidore_cache_recovery_v2", "industrial_mps_sj_v2")
SOURCES = tuple("src/rag_benchmark/" + name + ".py" for name in MODULES) + (
    "src/rag_benchmark/__init__.py", "scripts/run_industrial_mps_sj_v2.py", "uv.lock", "pyproject.toml")
PACKAGES = ("numpy", "torch", "transformers", "sentence-transformers", "peft", "Pillow", "safetensors",
            "huggingface-hub", "tokenizers", "accelerate", "soundfile", "scipy", "av")


def sha(value):
    return hashlib.sha256(value).hexdigest()


def detached(value):
    return json.loads(frozen.canonical(value))


def runtime():
    return {"python": platform.python_version(), "machine": platform.machine(), "system": platform.system(),
            "numpy_build": frozen.digest(np.__config__.CONFIG),
            "packages": {name: importlib.metadata.version(name) for name in PACKAGES}}


def verify_module(module, path):
    """Check original code, globals ownership and literal defaults, including methods."""
    if Path(module.__file__).resolve() != path:
        raise ValueError("Imported source path differs")
    raw = frozen.file_bytes(path)
    compiled = compile(raw, str(path), "exec", dont_inherit=True)
    tree = ast.parse(raw)

    def check(function, node, code):
        if isinstance(function, (staticmethod, classmethod)):
            function = function.__func__
        if isinstance(function, property):
            function = function.fget
        if (not isinstance(function, types.FunctionType) or function.__globals__ is not module.__dict__
                or function.__code__ != code or function.__code__.co_filename != str(path)):
            raise ValueError("Loaded code or globals binding differs")
        defaults = tuple(ast.literal_eval(x) for x in node.args.defaults) or None
        keywords = {a.arg: ast.literal_eval(v) for a, v in zip(node.args.kwonlyargs, node.args.kw_defaults)
                    if v is not None} or None
        if function.__defaults__ != defaults or function.__kwdefaults__ != keywords:
            raise ValueError("Loaded defaults differ")

    codes = {c.co_name: c for c in compiled.co_consts if isinstance(c, types.CodeType)}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            check(module.__dict__.get(node.name), node, codes[node.name])
        elif isinstance(node, ast.ClassDef):
            cls = module.__dict__.get(node.name)
            if not isinstance(cls, type) or cls.__module__ != module.__name__:
                raise ValueError("Loaded class differs")
            methods = {c.co_name: c for c in codes[node.name].co_consts if isinstance(c, types.CodeType)}
            for child in node.body:
                if isinstance(child, ast.FunctionDef):
                    check(cls.__dict__.get(child.name), child, methods[child.name])
        elif isinstance(node, ast.ImportFrom) and node.module != "__future__":
            target = importlib.import_module("." * node.level + (node.module or ""), module.__package__)
            for alias in node.names:
                if module.__dict__.get(alias.asname or alias.name) is not getattr(target, alias.name):
                    raise ValueError("Imported function binding differs")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname or alias.name.split(".")[0]
                target = importlib.import_module(alias.name if alias.asname else name)
                if module.__dict__.get(name) is not target:
                    raise ValueError("Imported module binding differs")
        elif isinstance(node, ast.Assign):
            # These reviewed modules have only constant expressions and one pure
            # function alias at top level. No model or IO call is evaluated here.
            allowed = {**module.__dict__, "__builtins__": {"tuple": tuple}}
            expected = eval(compile(ast.Expression(node.value), str(path), "eval"), allowed)
            for target in node.targets:
                if not isinstance(target, ast.Name) or module.__dict__.get(target.id) != expected:
                    raise ValueError("Loaded constant differs")


def sources(root, pins):
    if set(pins) != set(SOURCES):
        raise ValueError("Exact source closure required")
    for name in SOURCES:
        if frozen.record(root, name)["sha256"] != pins[name]:
            raise ValueError("Source seal differs")
    verify_module(sys.modules[__name__], frozen.path_at(root, "src/rag_benchmark/industrial_mps_sj_v2.py"))
    modules = {}
    for name in MODULES:
        module = importlib.import_module("rag_benchmark." + name)
        verify_module(module, frozen.path_at(root, "src/rag_benchmark/" + name + ".py"))
        if hasattr(module, "np") and module.np is not np:
            raise ValueError("Numpy binding differs")
        modules[name] = module
    return modules


def stream_record(root, relative):
    """Bound memory even for multi-GB weights; require one stable regular inode."""
    path = frozen.path_at(root, relative)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("Regular asset required")
        h = hashlib.sha256()
        with os.fdopen(fd, "rb", closefd=False) as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                h.update(block)
        after = os.fstat(fd)
        def fields(s):
            return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns
        if fields(before) != fields(after) or fields(after) != fields(path.stat(follow_symlinks=False)):
            raise ValueError("Asset changed during hashing")
        return {"path": relative, "bytes": after.st_size, "sha256": h.hexdigest()}
    finally:
        os.close(fd)


def asset_chain(root, logical):
    """Bind every HF snapshot/repository/shared-blob symlink without loading assets."""
    cache = frozen.path_at(root, ".cache/models")
    relative = Path(logical)
    if relative.is_absolute() or relative.as_posix() != logical or ".." in relative.parts:
        raise ValueError("Canonical logical asset path required")
    pending = root / relative
    links, seen = [], set()
    for _ in range(17):
        if not pending.is_relative_to(cache):
            raise ValueError("Asset chain escapes local model cache")
        parts = pending.relative_to(cache).parts
        current = cache
        for index, part in enumerate(parts):
            current /= part
            status = current.lstat()
            if stat.S_ISLNK(status.st_mode):
                if current in seen or len(links) == 16:
                    raise ValueError("Cyclic or excessive asset symlink chain")
                seen.add(current)
                target = os.readlink(current)
                after = current.lstat()
                if (status.st_dev, status.st_ino, status.st_mtime_ns, status.st_ctime_ns) != (
                        after.st_dev, after.st_ino, after.st_mtime_ns, after.st_ctime_ns):
                    raise ValueError("Asset symlink changed during inspection")
                links.append({"path": current.relative_to(root).as_posix(), "target": target,
                              "device": status.st_dev, "inode": status.st_ino,
                              "mtime_ns": status.st_mtime_ns, "ctime_ns": status.st_ctime_ns})
                destination = Path(target) if Path(target).is_absolute() else current.parent / target
                pending = Path(os.path.normpath(destination.joinpath(*parts[index + 1:])))
                break
        else:
            if not stat.S_ISREG(pending.stat(follow_symlinks=False).st_mode):
                raise ValueError("Regular final model asset required")
            return {"logical_path": logical, "symlink_chain": links,
                    "resolved_path": pending.relative_to(root).as_posix()}
    raise ValueError("Excessive asset symlink chain")


def asset_inventory(root, key):
    model, revision = MODEL_REVISIONS[key]
    relative = ".cache/models/models--" + model.replace("/", "--") + "/snapshots/" + revision
    snapshot = frozen.path_at(root, relative)
    if not snapshot.is_dir():
        raise FileNotFoundError("Missing pinned local snapshot; downloads are forbidden")
    assets = {}
    for path in sorted(snapshot.rglob("*")):
        if path.is_dir():
            if path.is_symlink():
                raise ValueError("Snapshot directory symlink forbidden")
            continue
        name = path.relative_to(snapshot).as_posix()
        if name.endswith((".py", ".bin", ".pkl", ".pickle")):
            raise ValueError("Remote code and pickle model assets are forbidden")
        chain = asset_chain(root, path.relative_to(root).as_posix())
        final = stream_record(root, chain["resolved_path"])
        if asset_chain(root, path.relative_to(root).as_posix()) != chain:
            raise ValueError("Asset symlink chain changed while hashing")
        assets[name] = {**chain, "file": final}
    required = {"config.json"} if key == "S_base" else {"modules.json", "config_sentence_transformers.json",
                                                               "sentence_bert_config.json"}
    required |= {"tokenizer.json", "tokenizer_config.json", "preprocessor_config.json"}
    required |= {"merges.txt", "vocab.json", "special_tokens_map.json", "added_tokens.json"} if key != "J" else set()
    required |= {"model.safetensors.index.json"} if key == "S_base" else set()
    required |= {"adapter_config.json", "adapter_model.safetensors", "1_Dense/config.json",
                 "1_Dense/model.safetensors"} if key == "S" else set()
    required |= {"config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
                 "processor_config.json", "tokenizer.model", "chat_template.jinja"} if key == "J" else set()
    required |= {"chat_template.jinja", "video_preprocessor_config.json"} if key == "S" else set()
    if not required <= assets.keys():
        raise FileNotFoundError("Missing required local model assets: " + str(sorted(required - assets.keys())))
    if any(assets[name]["file"]["sha256"] != value for name, value in CONFIG_HASHES[key].items()):
        raise ValueError("Historical model configuration pin differs")
    for name in list(assets):
        if name.endswith(".safetensors.index.json"):
            index = json.loads(frozen.checked(root, assets[name]["file"]))
            if not set(index["weight_map"].values()) <= assets.keys():
                raise FileNotFoundError("Missing indexed model weight shard")
    if "modules.json" in assets:
        modules = json.loads(frozen.checked(root, assets["modules.json"]["file"]))
        for entry in modules:
            path = entry["path"]
            if path and (Path(path).is_absolute() or ".." in Path(path).parts
                         or (not entry["type"].endswith(".Normalize")
                             and not any(n.startswith(path + "/") for n in assets))):
                raise FileNotFoundError("Missing nested SentenceTransformer module")
    return {"model_id": model, "revision": revision, "snapshot": relative, "assets": assets}


def archive_inputs(root):
    entry = stream_record(root, ARCHIVE)
    if entry["bytes"] != 90193920 or entry["sha256"] != ARCHIVE_SHA:
        raise ValueError("Inherited archive differs")
    result = {"archive": entry, "channels": {}}
    with tarfile.open(frozen.path_at(root, ARCHIVE), "r:") as archive:
        members = archive.getmembers()
        names = [m.name for m in members]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate archive names")
        for member in members:
            path = Path(member.name)
            if path.is_absolute() or ".." in path.parts or member.issym() or member.islnk():
                raise ValueError("Unsafe archive member")
        for channel, identity in INHERITED_IDS.items():
            suffix = frozen._ranking_path(identity).removeprefix(".cache/multimodal/")
            candidates = [m for m in members if m.name == suffix or m.name.endswith("/" + suffix)]
            report_candidates = [m for m in members if m.name == "runs/" + channel + "/report.json"]
            if len(candidates) != 1 or len(report_candidates) != 1:
                raise ValueError("Exact unique inherited ranking/report members required")
            found = {}
            for kind, member in (("ranking", candidates[0]), ("report", report_candidates[0])):
                if not member.isfile() or not 0 < member.size <= 32 * 1024 * 1024:
                    raise ValueError("Bounded regular archive member required")
                raw = archive.extractfile(member).read()
                found[kind] = {"member": member.name, "bytes": len(raw), "sha256": sha(raw)}
                if kind == "report" and sha(raw) != REPORT_HASHES[channel]:
                    raise ValueError("Original report seal differs")
            result["channels"][channel] = {**found, "identity": identity, **HARDWARE[channel]}
        state_member = archive.getmember("state.json")
        if not state_member.isfile() or not 0 < state_member.size <= 8 * 1024 * 1024:
            raise ValueError("Bounded regular original state required")
        raw = archive.extractfile(state_member).read()
        state = json.loads(raw)
        state_identity = frozen.digest(state["identity"])
        if (state_identity != "8344a69d009356f3caf843c7c11599cb470580f0006b80bd84f17f06239613f0"
                or state["identity"]["runtime"]["gpu"]["name"] != "NVIDIA L4"):
            raise ValueError("Original L4 hardware/state identity differs")
        result["state"] = {"member": "state.json", "bytes": len(raw), "sha256": sha(raw),
                           "identity_sha256": state_identity, "runtime": state["identity"]["runtime"]}
    return result


def read_member(root, manifest, channel, kind):
    expected = manifest["inherited"]["channels"][channel][kind]
    with tarfile.open(frozen.path_at(root, ARCHIVE), "r:") as archive:
        member = archive.getmember(expected["member"])
        if not member.isfile() or member.size != expected["bytes"] or member.size > 32 * 1024 * 1024:
            raise ValueError("Inherited member type/size differs")
        raw = archive.extractfile(member).read()
    if sha(raw) != expected["sha256"]:
        raise ValueError("Inherited member seal differs")
    return raw


def lock_record(root):
    path = frozen.path_at(root, "work/overnight/gpu.lock")
    status = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(status.st_mode) or status.st_size != 0:
        raise ValueError("Existing empty regular GPU lock required")
    return {"path": "work/overnight/gpu.lock", "device": status.st_dev, "inode": status.st_ino, "bytes": 0}


def recipe(root, pins):
    return {"version": VERSION, "source_pins": pins, "dataset_id": DATASET, "dataset_identity": DATASET_IDENTITY,
            "dataset_revision": DATASET_REVISION, "output": DELTA, "model_revisions": detached(MODEL_REVISIONS),
            "archive_sha256": ARCHIVE_SHA, "hardware": HARDWARE,
            "lock": {"path": "work/overnight/gpu.lock", "device": 16777232, "inode": 135900020, "bytes": 0},
            "root": str(root), "mode": "read-only-preparation-no-model-no-download"}


def prepare(root, request, request_sha256):
    request = detached(request)
    if frozen.digest(request) != request_sha256 or request != recipe(root, request["source_pins"]):
        raise ValueError("Exact preparation recipe required")
    sources(root, request["source_pins"])
    if lock_record(root) != request["lock"]:
        raise ValueError("Existing lock inode differs")
    directory = "data/multimodal/" + DATASET
    tables = {name: frozen.record(root, directory + "/" + name)
              for name in ("dataset.json", "corpus.jsonl", "queries.jsonl", "qrels.jsonl")}
    corpus = frozen._table_rows(frozen.checked(root, tables["corpus.jsonl"]))
    queries = frozen._table_rows(frozen.checked(root, tables["queries.jsonl"]))
    media = {rel: frozen.record(root, directory + "/" + rel)
             for row in corpus + queries for rel in row.get("media", {}).values()}
    manifest = {**request, "recipe_sha256": request_sha256, "tables": tables, "media": media,
                "ordered_id_sha256": [frozen.digest([str(r["id"]) for r in rows]) for rows in (corpus, queries)],
                "assets": {key: asset_inventory(root, key) for key in MODEL_REVISIONS}, "runtime": runtime(),
                "inherited": archive_inputs(root)}
    validate(root, manifest)
    return manifest


def validate(root, manifest):
    expected = recipe(root, manifest["source_pins"])
    if any(manifest.get(k) != v for k, v in expected.items()):
        raise ValueError("Fixed industrial condition differs")
    if manifest["recipe_sha256"] != frozen.digest(expected) or manifest["runtime"] != runtime():
        raise ValueError("Recipe/runtime differs")
    modules = sources(root, manifest["source_pins"])
    if lock_record(root) != manifest["lock"]:
        raise ValueError("Lock inode changed")
    metadata, corpus, queries, qrels, exclusions = frozen._dataset(root, manifest)
    if (metadata["revision"] != DATASET_REVISION or len(corpus) != 5244 or len(queries) != 283
            or sum(sum(g > 0 for g in row.values()) for row in qrels.values()) != 1614
            or sum(not row.get("text", "").strip() for row in corpus) != 205):
        raise ValueError("Frozen industrial population differs")
    for row in corpus:
        if not isinstance(row.get("text"), str) or set(row.get("media", {})) != {"image"}:
            raise ValueError("Industrial documents require original Markdown and one page image")
    if any(q.get("media") or not isinstance(q.get("text"), str) or not q["text"].strip() for q in queries):
        raise ValueError("Text-only complete queries required")
    if manifest["assets"] != {key: asset_inventory(root, key) for key in MODEL_REVISIONS}:
        raise ValueError("Local asset closure changed")
    if manifest["inherited"] != archive_inputs(root):
        raise ValueError("Inherited evidence changed")
    ids, query_ids = [str(r["id"]) for r in corpus], [str(q["id"]) for q in queries]
    for channel, identity in INHERITED_IDS.items():
        payload = json.loads(read_member(root, manifest, channel, "ranking"))
        frozen._rankings(payload, identity, ids, query_ids, exclusions, bm25=channel == "B")
        report = json.loads(read_member(root, manifest, channel, "report"))
        if (report["dataset"]["identity"] != DATASET_IDENTITY
                or report["channels"][channel]["status"] != "completed"
                or report["channels"][channel]["identity"] != identity):
            raise ValueError("Inherited report/channel identity differs")
    return modules, corpus, queries, qrels, exclusions


class ExistingLock:
    def __init__(self, root, expected):
        self.root, self.expected, self.fd = root, expected, None

    def __enter__(self):
        if lock_record(self.root) != self.expected:
            raise ValueError("Lock inode changed before open")
        self.fd = os.open(frozen.path_at(self.root, self.expected["path"]), os.O_RDWR | os.O_NOFOLLOW)
        try:
            current = os.fstat(self.fd)
            if (current.st_dev, current.st_ino, current.st_size) != (
                    self.expected["device"], self.expected["inode"], 0):
                raise ValueError("Opened wrong lock inode")
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if lock_record(self.root) != self.expected:
                raise ValueError("Lock inode replaced during acquisition")
        except BaseException:
            os.close(self.fd)
            self.fd = None
            raise
        return self

    def __exit__(self, *_):
        if self.fd is not None:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            os.close(self.fd)
            self.fd = None


def mkdir(path):
    path.mkdir(mode=0o700)
    frozen._fsync_directory(path.parent)


def local_models(root, manifest, channel, destination):
    """All metadata is privately copied; only sealed safetensor weights are linked."""
    mkdir(destination)
    paths = {}
    for key in (["J"] if channel == "J" else ["S", "S_base"]):
        spec = manifest["assets"][key]
        directory = destination / spec["revision"]
        mkdir(directory)
        for name, entry in spec["assets"].items():
            target = frozen.path_at(directory, name)
            target.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
            source = frozen.path_at(root, entry["file"]["path"])
            if name.endswith(".safetensors"):
                target.symlink_to(source)
                frozen._fsync_directory(target.parent)
            else:
                frozen.exclusive_write(target, frozen.checked(root, entry["file"]))
        paths[key] = directory
    if channel == "S":
        config_path = paths["S"] / "adapter_config.json"
        original = json.loads(frozen.file_bytes(config_path))
        if original["base_model_name_or_path"] != MODEL_REVISIONS["S_base"][0]:
            raise ValueError("ColQwen base model differs")
        # This is a new private metadata file, never a symlink into original storage.
        rewritten = {**original, "base_model_name_or_path": str(paths["S_base"]),
                     "revision": MODEL_REVISIONS["S_base"][1]}
        config_path.unlink()
        frozen.exclusive_write(config_path, frozen.canonical(rewritten))
    return paths


def make_adapter(modules, channel, paths):
    """Called only under ExistingLock; no global adapter mutation or hub resolution."""
    if channel == "J":
        return modules["multimodal_models"].EmbeddingGemma2Adapter({
            "path": str(paths["J"]), "mode": "joint", "device": "mps", "dtype": "bfloat16",
            "batch_size": 1, "dimension": 768, "max_length": 8192, "vision_budget": 280,
            "local_files_only": True, "revision": MODEL_REVISIONS["J"][1]}, mode="joint")
    original = modules["multimodal_colqwen"]

    class IsolatedColQwen(original.ColQwenAdapter):
        def _load(self):
            self._check_config()
            if self._model is None:
                import torch
                from sentence_transformers import MultiVectorEncoder
                self.projection_verification = original.verify_projection(paths["S"], paths["S_base"])
                self._model = MultiVectorEncoder(str(paths["S"]),
                    device=str(modules["models"]._torch_device("mps")), local_files_only=True,
                    trust_remote_code=False, model_kwargs={"dtype": torch.bfloat16, "attn_implementation": "sdpa"})
                self._model.eval()
                if {str(p.device).split(":")[0] for p in self._model.parameters()} != {"mps"}:
                    raise RuntimeError("CPU fallback forbidden")
            return self._model

    return IsolatedColQwen({"cache_dir": str(paths["S"].parent), "device": "mps", "score_device": "mps"})


def vector_bytes(value, channel):
    value = np.asarray(value)
    expected_dim = 768 if channel == "J" else 128
    if (value.dtype != np.dtype("float32") or value.ndim != 2 or value.shape[1] != expected_dim
            or not 1 <= value.shape[0] <= (1 if channel == "J" else 8192)
            or not np.isfinite(value).all()
            or not np.allclose(np.linalg.norm(value, axis=1), 1, atol=.002, rtol=.002)):
        raise ValueError("Invalid native normalized output")
    stream = io.BytesIO()
    np.save(stream, np.ascontiguousarray(value), allow_pickle=False)
    return stream.getvalue()


def read_vector(raw, channel):
    if len(raw) < 10 or raw[:6] != b"\x93NUMPY" or raw[6:8] not in (b"\x01\x00", b"\x02\x00"):
        raise ValueError("Invalid bounded NPY")
    width = 2 if raw[6] == 1 else 4
    length = struct.unpack("<H" if width == 2 else "<I", raw[8:8 + width])[0]
    if length > 4096:
        raise ValueError("Oversized header")
    header = ast.literal_eval(raw[8 + width:8 + width + length].decode("ascii").strip())
    shape = header.get("shape")
    dimension = 768 if channel == "J" else 128
    if (not isinstance(shape, tuple) or len(shape) != 2 or type(shape[0]) is not int
            or not 1 <= shape[0] <= (1 if channel == "J" else 8192) or shape[1] != dimension
            or header != {"descr": "<f4", "fortran_order": False, "shape": shape}):
        raise ValueError("Invalid header shape/dtype")
    payload = raw[8 + width + length:]
    if len(payload) != shape[0] * dimension * 4:
        raise ValueError("Truncated/trailing payload")
    result = np.frombuffer(payload, dtype="<f4").reshape(shape)
    vector_bytes(result, channel)
    return result


def bounded_bytes(path, maximum):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or not 0 <= before.st_size <= maximum:
            raise ValueError("Oversized or nonregular own-cache file")
        with os.fdopen(fd, "rb", closefd=False) as handle:
            value = handle.read(maximum + 1)
        after = os.fstat(fd)
        current = path.stat(follow_symlinks=False)
        def fields(s):
            return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns
        if len(value) != before.st_size or fields(before) != fields(after) or fields(after) != fields(current):
            raise ValueError("Own-cache file changed during read")
        return value
    finally:
        os.close(fd)


def job_item(root, manifest, row, role, channel):
    item = {"id": str(row["id"]), "media": {}}
    for kind, relative in row.get("media", {}).items():
        frozen.checked(root, manifest["media"][relative])
        item["media"][kind] = str(frozen.path_at(root, manifest["media"][relative]["path"]))
    if channel == "J" or role == "query":
        item["text"] = row.get("text", "")
    # Identity includes the entire frozen row, including order via index in claim.
    return item


def checked_usage(usage, item, role, channel):
    usage = detached(usage)
    seconds = usage.get("inference_seconds")
    if not isinstance(seconds, (float, int)) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError("Missing finite fresh inference timing")
    if usage.get("dtype") != "bfloat16" or usage.get("context_limit") != 8192:
        raise ValueError("Generation condition differs")
    if channel == "S":
        counts = usage.get("token_counts")
        if (usage.get("items") != 1 or usage.get("device") != "mps" or usage.get("max_image_patches") != 768
                or not isinstance(counts, list) or len(counts) != 1 or type(counts[0]) is not int
                or not 1 <= counts[0] <= 8192):
            raise ValueError("Invalid native token diagnostics")
    else:
        rows = usage.get("items")
        if ({x.split(":")[0] for x in usage.get("devices", [])} != {"mps"}
                or not isinstance(rows, list) or len(rows) != 1
                or set(rows[0].get("modalities", [])) != ({"image", "text"} if role == "document" else {"text"})):
            raise ValueError("Invalid joint row diagnostics")
        count = rows[0].get("expanded_tokens")
        if type(count) is not int or not 1 <= count <= 8192:
            raise ValueError("Invalid expanded context")
        if role == "document" and rows[0].get("derived_text_empty") != (not item["text"].strip()):
            raise ValueError("Joint text preservation differs")
        if set(usage.get("towers", {})) != {"vision", "audio"}:
            raise ValueError("Complete native towers required")
    return usage


def encode_jobs(root, manifest, channel, role, rows, directory, adapter, resume, manifest_sha256, job_identity):
    values, fresh, elapsed = [], 0, 0.0
    frozen.path_at(root, directory.relative_to(root).as_posix())
    allowed_names = {f"{index:06d}" + suffix for index in range(len(rows))
                     for suffix in (".claim.json", ".terminal.json", ".npy")}
    entries = list(directory.iterdir())
    if (not {p.name for p in entries} <= allowed_names
            or any(not stat.S_ISREG(p.stat(follow_symlinks=False).st_mode) for p in entries)):
        raise ValueError("Unexpected own-cache files before inference")
    expected_names = set()
    for index, row in enumerate(rows):
        stem = f"{index:06d}"
        claim_path, terminal_path, data_path = [directory / (stem + suffix)
                                                for suffix in (".claim.json", ".terminal.json", ".npy")]
        expected_names.update(p.name for p in (claim_path, terminal_path, data_path))
        claim = {"manifest": manifest_sha256, "channel": channel, "role": role, "index": index,
                 "row_sha256": frozen.digest(row), "channel_identity": job_identity}
        if terminal_path.exists():
            if not resume or json.loads(bounded_bytes(claim_path, 65536)) != claim:
                raise ValueError("Resume claim differs")
            terminal = json.loads(bounded_bytes(terminal_path, 65536))
            if terminal.get("claim_sha256") != frozen.digest(claim) or terminal.get("status") != "completed":
                raise ValueError("Invalid terminal identity")
            expected = terminal["output"]
            if expected["path"] != data_path.relative_to(root).as_posix():
                raise ValueError("Terminal points outside own cache job")
            raw = bounded_bytes(data_path, (768 if channel == "J" else 8192 * 128) * 4 + 4108)
            if len(raw) != expected["bytes"] or sha(raw) != expected["sha256"]:
                raise ValueError("Sealed own-cache vector differs")
            value = read_vector(raw, channel)
            item = job_item(root, manifest, row, role, channel)
            usage = checked_usage(terminal["usage"], item, role, channel)
            if channel == "S" and usage["token_counts"] != [len(value)]:
                raise ValueError("Cached token output/diagnostic counts differ")
            if channel == "S":
                proof = terminal.get("projection_verification", {})
                if proof.get("projection_verified") is not True or not 0 <= proof.get("max_absolute_merge_error", 1) <= 1e-6:
                    raise ValueError("Cached projection proof missing")
        else:
            if claim_path.exists() or data_path.exists():
                raise ValueError("Ambiguous prior job; automatic retry forbidden")
            frozen.exclusive_write(claim_path, frozen.canonical(claim))
            item = job_item(root, manifest, row, role, channel)
            try:
                value = adapter.encode([item], role) if channel == "J" else adapter.encode_tokens([item], role)[0]
                raw = vector_bytes(value, channel)
                usage = checked_usage(adapter.last_usage, item, role, channel)
                if channel == "S" and usage["token_counts"] != [len(value)]:
                    raise ValueError("Token output/diagnostic counts differ")
                frozen.exclusive_write(data_path, raw)
                terminal = {"status": "completed", "claim_sha256": frozen.digest(claim), "usage": usage,
                            "output": frozen.record(root, data_path.relative_to(root).as_posix())}
                if channel == "S":
                    proof = detached(adapter.projection_verification)
                    if proof.get("projection_verified") is not True or not 0 <= proof.get("max_absolute_merge_error", 1) <= 1e-6:
                        raise ValueError("Projection proof missing")
                    terminal["projection_verification"] = proof
                frozen.exclusive_write(terminal_path, frozen.canonical(terminal))
            except BaseException as error:
                # An errored/ambiguous model call can never be silently repeated.
                if not terminal_path.exists():
                    frozen.exclusive_write(terminal_path, frozen.canonical({"status": "failed",
                        "claim_sha256": frozen.digest(claim), "error_type": type(error).__name__}))
                raise
            fresh += 1
            elapsed += usage["inference_seconds"]
        values.append(value)
    if {p.name for p in directory.iterdir()} != expected_names:
        raise ValueError("Unexpected own-cache files")
    return values, {"total_items": len(rows), "fresh_inference_items": fresh, "validated_cache_items": len(rows) - fresh,
                    "fresh_inference_seconds": elapsed if fresh else None}


def stage_identity(manifest, manifest_sha256, channel):
    return frozen.digest({"condition": VERSION, "manifest_sha256": manifest_sha256, "channel": channel,
                          "dataset": manifest["dataset_identity"], "hardware": HARDWARE[channel], "top_k": 100})


def execute(root, manifest, manifest_sha256, channel, *, resume=False):
    manifest = detached(manifest)
    if channel not in ("S", "J") or frozen.digest(manifest) != manifest_sha256:
        raise ValueError("Exact admitted manifest and S/J stage required")
    modules, corpus, queries, _, exclusions = validate(root, manifest)
    # Enforce offline package behavior in this isolated child, never in a shared adapter module.
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[key] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0":
        raise ValueError("MPS fallback environment forbidden")
    output = frozen.path_at(root, DELTA)
    with ExistingLock(root, manifest["lock"]):
        if not output.exists():
            mkdir(output)
            frozen.exclusive_write(output / "manifest.json", frozen.canonical(manifest))
        elif sha(frozen.file_bytes(output / "manifest.json")) != manifest_sha256:
            raise ValueError("Existing output manifest differs")
        stage = frozen.path_at(root, DELTA + "/" + channel)
        if stage.exists():
            if not resume or (stage / "closed.json").exists():
                raise ValueError("Existing stage requires partial resume; closed stages are immutable")
            if any(os.path.lexists(stage / name) for name in ("scoring.claim.json", "rankings.json", "report.json", "closed.json")):
                raise ValueError("Ambiguous scoring/publication job; automatic retry forbidden")
        else:
            mkdir(stage)
            for role in ("document", "query"):
                mkdir(stage / role)
        # New process-specific metadata prevents ST rewrites from touching original or previous caches.
        paths = local_models(root, manifest, channel, stage / ("model-view-" + str(os.getpid())))
        adapter = make_adapter(modules, channel, paths)
        native_adapter_identity = adapter.identity
        # Native identities contain private paths. Own jobs use a separately
        # named channel identity bound to the manifest, source and asset closure.
        identity = stage_identity(manifest, manifest_sha256, channel)
        try:
            documents, doc_usage = encode_jobs(root, manifest, channel, "document", corpus, stage / "document",
                                               adapter, resume, manifest_sha256, identity)
            query_values, query_usage = encode_jobs(root, manifest, channel, "query", queries, stage / "query",
                                                    adapter, resume, manifest_sha256, identity)
            adapter.unload()
            scoring_claim = stage / "scoring.claim.json"
            if scoring_claim.exists():
                raise ValueError("Ambiguous scoring/publication job; automatic retry forbidden")
            frozen.exclusive_write(scoring_claim, frozen.canonical({"identity": identity,
                "manifest": manifest_sha256, "query_count": len(queries), "top_k": 100}))
            started = time.perf_counter()
            ids = [str(row["id"]) for row in corpus]
            if channel == "J":
                rankings = modules["multimodal"].exact_dot_rankings(np.concatenate(documents),
                    np.concatenate(query_values), ids, top_k=100, exclusions=exclusions)
            else:
                rankings = []
                for offset in range(0, len(queries), 8):
                    scores = modules["multimodal_colqwen"].exact_maxsim(query_values[offset:offset + 8], documents,
                                                                         device="mps", chunk_elements=16_000_000)
                    for index, row in enumerate(scores):
                        eligible = [i for i, identifier in enumerate(ids) if identifier not in exclusions[offset + index]]
                        selected = sorted(eligible, key=lambda i: (-float(row[i]), ids[i]))[:100]
                        rankings.append([{"id": ids[i], "score": float(row[i]), "rank": rank}
                                         for rank, i in enumerate(selected, 1)])
            payload = {"identity": identity, "query_ids": [str(q["id"]) for q in queries],
                       "rankings": rankings, "condition": VERSION, "manifest_sha256": manifest_sha256,
                       "dataset_identity": DATASET_IDENTITY, "hardware": HARDWARE[channel]}
            frozen._rankings(payload, identity, ids, payload["query_ids"], exclusions)
            report = {"condition": VERSION, "manifest_sha256": manifest_sha256,
                      "dataset": {"id": DATASET, "identity": DATASET_IDENTITY, "revision": DATASET_REVISION},
                      "channels": {channel: {"status": "completed", "identity": identity}},
                      "hardware": HARDWARE, "usage": {"document": doc_usage, "query": query_usage,
                      "scoring_seconds": time.perf_counter() - started},
                      "projection_verification": getattr(adapter, "projection_verification", None),
                      "native_adapter_identity_this_process": native_adapter_identity,
                      "native_config_this_process": adapter.config,
                      "none_cells_completed": 0, "laya_cells_completed": 0}
            # Revalidate original bytes before closing a result. This never repairs changed inputs.
            validate(root, manifest)
            frozen.exclusive_write(stage / "rankings.json", frozen.canonical(payload))
            frozen.exclusive_write(stage / "report.json", frozen.canonical(report))
            files = {name: frozen.record(root, (stage / name).relative_to(root).as_posix())
                     for name in ("rankings.json", "report.json", "scoring.claim.json")}
            terminals = [frozen.record(root, p.relative_to(root).as_posix()) for role in ("document", "query")
                         for p in sorted((stage / role).glob("*.terminal.json"))]
            closed = {"status": "completed", "identity": identity, "manifest_sha256": manifest_sha256,
                      "files": files, "terminal_records": terminals, "terminal_sha256": frozen.digest(terminals)}
            frozen.exclusive_write(stage / "closed.json", frozen.canonical(closed))
            return frozen.record(root, (stage / "closed.json").relative_to(root).as_posix())
        finally:
            adapter.unload()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "execute"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--sha256", required=True, help="Out-of-band canonical recipe/manifest SHA256")
    parser.add_argument("--channel", choices=("S", "J"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    sys.dont_write_bytecode = True
    value = json.loads(frozen.file_bytes(frozen.path_at(args.root, args.input)))
    if args.mode == "prepare":
        if args.channel or args.resume:
            raise ValueError("Preparation cannot execute or resume")
        result = prepare(args.root, value, args.sha256)
        path = frozen.path_at(args.root, BASE + "/prepared-manifest-v2.json")
        frozen.exclusive_write(path, frozen.canonical(result))
        print(json.dumps({"manifest": frozen.record(args.root, path.relative_to(args.root).as_posix()),
                          "canonical_sha256": frozen.digest(result), "model_calls": 0}))
    else:
        print(json.dumps({"closed": execute(args.root, value, args.sha256, args.channel, resume=args.resume)}))
    return 0

"""Pinned LaSCo validation import: explicit downloads, offline preparation, no inference."""

from __future__ import annotations

import argparse
import ctypes
import fcntl
import hashlib
import json
import os
import re
import stat
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

from .multimodal_data import sha256, write_dataset

REVISION = "311f4d7d4bc069fc2d8abd3bf523cdfae6b5800f"
BASE_URL = f"https://raw.githubusercontent.com/levymsn/LaSCo/{REVISION}/downloads"
IMAGE_URL = "http://images.cocodataset.org/zips/val2014.zip"
ANNOTATION_SHA256 = "7d25c8bd296ac463da3754f2737b2a713a94657c0efb50599a9191732ff3bf86"
GALLERY_SHA256 = "448783f45a20cab61e0f099638011979b2fe81d34bed3d69cbc02ecc955557ac"
GALLERY_COUNT, QUERY_COUNT = 39826, 30037
MAX_IMAGE_BYTES = 64 * 1024**2
_SHA = re.compile(r"[a-f0-9]{64}\Z")
_IMAGE = re.compile(r"val2014/COCO_val2014_(\d{12})\.jpg\Z")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def identifier(value):
    require(type(value) is int and value >= 0, "Invalid integer identifier")
    return str(value)


def image(value):
    require(isinstance(value, (list, tuple)) and len(value) == 2, "Invalid image reference")
    key = identifier(value[0])
    match = _IMAGE.fullmatch(value[1]) if isinstance(value[1], str) else None
    require(match is not None and int(match[1]) == int(key), "Image ID/path mismatch")
    return key, value[1]


def build_rows(annotations, gallery, *, expected_gallery=GALLERY_COUNT, expected_queries=QUERY_COUNT):
    """Validate complete labels before extraction; smaller counts are only for CPU fixtures."""
    require(isinstance(annotations, list) and isinstance(gallery, list), "Expected lists")
    require(len(gallery) == expected_gallery and len(annotations) == expected_queries, "Incomplete split")
    images, paths = {}, set()
    for row in gallery:
        require(isinstance(row, dict) and set(row) == {"id", "path"}, "Invalid gallery schema")
        gallery_path = row["path"]
        require(isinstance(gallery_path, str), "Invalid gallery path")
        # Official gallery adds val/ while triplets and COCO ZIP use val2014/.
        if gallery_path.startswith("val/val2014/"):
            gallery_path = gallery_path.removeprefix("val/")
        key, path = image([row["id"], gallery_path])
        require(key not in images and path not in paths, "Duplicate gallery image")
        images[key] = path
        paths.add(path)
    corpus = [
        {
            "id": key,
            "media": {"image": f"media/{key}.jpg"},
            "metadata": {"source_path": path, "text_status": "requires_generation"},
        }
        for key, path in images.items()
    ]
    queries, qrels, seen = [], [], set()
    for row in annotations:
        require(
            isinstance(row, dict) and set(row) == {"qid", "query-image", "query-text", "target-image"},
            "Invalid triplet schema",
        )
        require(type(row["qid"]) is int, "Invalid query identifier")
        qid = str(row["qid"])  # Official reverse-direction queries use negative QIDs.
        ref, rp = image(row["query-image"])
        target, tp = image(row["target-image"])
        require(qid not in seen, "Duplicate query")
        seen.add(qid)
        require(images.get(ref) == rp and images.get(target) == tp, "Image outside gallery")
        require(ref != target, "Target equals excluded reference")
        text = row["query-text"]
        require(isinstance(text, str), "Invalid transition text")
        queries.append(
            {
                "id": qid,
                "text": text,
                "media": {"image": f"media/{ref}.jpg"},
                "metadata": {
                    "reference_id": ref,
                    "exclude_ids": [ref],
                    "language": "en",
                    "transition_text_empty": not bool(text.strip()),
                },
            }
        )
        qrels.append({"query_id": qid, "corpus_id": target, "relevance": 1.0})
    return corpus, queries, qrels, images


def safe_member(info):
    name = info.filename
    path = PurePosixPath(name)
    require(
        name
        and "\\" not in name
        and ":" not in name
        and "\x00" not in name
        and not path.is_absolute()
        and all(x not in {"", ".", ".."} for x in name.rstrip("/").split("/")),
        "Unsafe ZIP path",
    )
    kind = stat.S_IFMT(info.external_attr >> 16)
    require(kind in {0, stat.S_IFREG, stat.S_IFDIR} and not info.flag_bits & 1, "Unsafe ZIP type")
    require(
        info.is_dir() == (kind == stat.S_IFDIR) or kind in {0, stat.S_IFREG} and not info.is_dir(),
        "ZIP directory mismatch",
    )
    return name.rstrip("/")


def extract_required(archive, destination, images):
    """Read selected members to EOF (CRC verified); never extractall or overwrite assets."""
    require(isinstance(images, dict), "Invalid required image map")
    for key, path in images.items():
        require(
            isinstance(key, str) and key.isascii() and key.isdigit() and str(int(key)) == key,
            "Invalid required image ID",
        )
        require(image([int(key), path]) == (key, path), "Invalid required image path")
    members = {}
    for info in archive.infolist():
        name = safe_member(info)
        require(name not in members, "Duplicate normalized ZIP member")
        members[name] = info
    for path in images.values():
        require(path in members and not members[path].is_dir(), "Missing required media")
        require(0 < members[path].file_size <= MAX_IMAGE_BYTES, "Media size bound")
    media = Path(destination) / "media"
    media.mkdir(mode=0o700)
    for key, path in images.items():
        size = 0
        with archive.open(members[path]) as source, (media / f"{key}.jpg").open("xb") as output:
            while chunk := source.read(1024**2):
                size += len(chunk)
                require(size <= members[path].file_size and size <= MAX_IMAGE_BYTES, "ZIP size overrun")
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        require(size == members[path].file_size, "ZIP size mismatch")


def pinned_json(path, digest):
    raw = Path(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == digest, "Annotation checksum mismatch")
    return json.loads(raw), {"file": Path(path).name, "sha256": digest, "size_bytes": len(raw)}


def publish_directory(stage, destination):
    """Atomic directory publication that cannot replace even an empty destination."""
    libc = ctypes.CDLL(None, use_errno=True)
    source, target = os.fsencode(stage), os.fsencode(destination)
    if sys.platform == "darwin":
        rename = libc.renamex_np
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        code = rename(source, target, 4)  # RENAME_EXCL
    elif sys.platform.startswith("linux"):
        rename = libc.renameat2
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        code = rename(-100, source, -100, target, 1)  # AT_FDCWD, RENAME_NOREPLACE
    else:
        raise ValueError("Atomic no-replace publication requires Linux or macOS")
    if code:
        error = ctypes.get_errno()
        raise OSError(error, "Dataset publication failed without replacing destination")
    directory = os.open(Path(destination).parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def import_validation(annotations, gallery, images_zip, destination, *, images_sha256):
    """Offline full import. Failed private staging remains unready for inspection.

    images_sha256 is a caller-observed archive seal, NOT an upstream attestation.
    """
    require(isinstance(images_sha256, str) and _SHA.fullmatch(images_sha256), "Full archive SHA required")
    annotations, gallery, images_zip, destination = map(Path, (annotations, gallery, images_zip, destination))
    destination = destination.absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(
        destination.parent / f".{destination.name}.lasco.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600
    )
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not destination.exists() and not destination.is_symlink(), "Destination exists")
        ann, asource = pinned_json(annotations, ANNOTATION_SHA256)
        gal, gsource = pinned_json(gallery, GALLERY_SHA256)
        corpus, queries, qrels, images = build_rows(
            ann, gal, expected_gallery=GALLERY_COUNT, expected_queries=QUERY_COUNT
        )
        stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent))
        with images_zip.open("rb") as handle:
            require(
                hashlib.file_digest(handle, "sha256").hexdigest() == images_sha256, "Archive SHA mismatch"
            )
            size = os.fstat(handle.fileno()).st_size
            handle.seek(0)
            with zipfile.ZipFile(handle) as archive:
                extract_required(archive, stage, images)
            handle.seek(0)
            require(hashlib.file_digest(handle, "sha256").hexdigest() == images_sha256, "Archive changed")
        require(
            sha256(annotations) == ANNOTATION_SHA256 and sha256(gallery) == GALLERY_SHA256,
            "Annotation changed",
        )
        asource["url"], gsource["url"] = f"{BASE_URL}/lasco_val.json", f"{BASE_URL}/lasco_val_corpus.json"
        result = write_dataset(
            stage,
            dataset_id="lasco-validation",
            track="composed_image",
            revision=REVISION,
            corpus=corpus,
            queries=queries,
            qrels=qrels,
            sources=[
                asource,
                gsource,
                {
                    "file": "val2014.zip",
                    "url": IMAGE_URL,
                    "sha256": images_sha256,
                    "size_bytes": size,
                    "sha256_attestation": "locally_observed",
                },
            ],
            license="LaSCo annotations MIT; COCO images retain original per-image licenses",
            text_source="absent_requires_generation",
            expected_counts={
                "corpus": GALLERY_COUNT,
                "queries": QUERY_COUNT,
                "qrels": QUERY_COUNT,
                "media_files": GALLERY_COUNT,
            },
            metadata={
                "split": "validation",
                "annotation_revision": REVISION,
                "annotation_origin": "VQA2.0_pairs_GPT3_transition_text",
                "reference_exclusion": True,
                "gold_targets_only_in_qrels": True,
                "zip_selected_crc_verified": True,
                "test_server_score": False,
                "benchmark_runs": 0,
                "empty_transition_count": sum(q["metadata"]["transition_text_empty"] for q in queries),
                "query_modalities": "reference_image_and_verbatim_modifier; empty_modifier_is_image_only",
            },
        )
        require(not destination.exists() and not destination.is_symlink(), "Destination appeared")
        publish_directory(stage, destination)
        return result
    finally:
        os.close(fd)


def download_sources(raw_dir, *, include_images=False, images_sha256=None):
    """Explicit streaming download. No overwrite, retry, unknown-checksum media, or inference."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    files = [
        ("lasco_val.json", f"{BASE_URL}/lasco_val.json", ANNOTATION_SHA256, 32 * 1024**2),
        ("lasco_val_corpus.json", f"{BASE_URL}/lasco_val_corpus.json", GALLERY_SHA256, 8 * 1024**2),
    ]
    if include_images:
        require(isinstance(images_sha256, str) and _SHA.fullmatch(images_sha256), "Full archive SHA required")
        files.append(("val2014.zip", IMAGE_URL, images_sha256, 8 * 1024**3))
    results = []
    for name, url, expected, limit in files:
        target = raw_dir / name
        require(not target.is_symlink(), "Symlink source")
        if not target.exists():
            partial = raw_dir / (name + ".partial")
            with partial.open("xb") as output, urllib.request.urlopen(url, timeout=180) as response:
                size, digest = 0, hashlib.sha256()
                while chunk := response.read(1024**2):
                    size += len(chunk)
                    require(size <= limit, "Download size bound")
                    output.write(chunk)
                    digest.update(chunk)
                output.flush()
                os.fsync(output.fileno())
            require(digest.hexdigest() == expected, "Download SHA mismatch")
            os.link(partial, target)
            partial.unlink()
        require(sha256(target) == expected, "Existing source SHA mismatch")
        results.append({"file": name, "url": url, "sha256": expected, "size_bytes": target.stat().st_size})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    network = commands.add_parser("download", help="Explicit network action; no import")
    network.add_argument("--raw-dir", type=Path, required=True)
    network.add_argument("--include-images", action="store_true")
    network.add_argument("--images-sha256")
    offline = commands.add_parser("import", help="Offline full-split preparation")
    for name in ("annotations", "gallery", "images-zip", "destination"):
        offline.add_argument("--" + name, type=Path, required=True)
    offline.add_argument("--images-sha256", required=True)
    args = parser.parse_args()
    os.umask(0o077)
    if args.command == "download":
        value = {
            "sources": download_sources(
                args.raw_dir, include_images=args.include_images, images_sha256=args.images_sha256
            )
        }
    else:
        value = import_validation(
            args.annotations,
            args.gallery,
            args.images_zip,
            args.destination,
            images_sha256=args.images_sha256,
        )
    print(json.dumps({"status": "prepared_no_inference", "counts": value.get("counts"), "benchmark_runs": 0}))


if __name__ == "__main__":
    main()

"""Boot-bound Industrial S/J adapter; historical V2 artifacts stay immutable.

The only operational input differences are one lock device and the 57 recorded
asset-symlink device fields. No global substitution, download, automatic resume,
or new baseline audit is permitted. Actual preparation and execution require
separate root and independent admissions; source acceptance alone is insufficient.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time

from . import industrial_mps_sj_v2 as original

VERSION = "industrial-l4-bgen-local-mps-sj-v3"
BASE = "work/" + VERSION
DELTA = BASE + "/actual-v3"
DATASET = original.DATASET
DATASET_IDENTITY = original.DATASET_IDENTITY
DATASET_REVISION = original.DATASET_REVISION
HARDWARE = original.HARDWARE
MODEL_REVISIONS = original.MODEL_REVISIONS
INHERITED_IDS = original.INHERITED_IDS
SOURCES = ("src/rag_benchmark/industrial_mps_sj_v3.py", "scripts/run_industrial_mps_sj_v3.py")
PARENT_PATH = original.BASE + "/prepared-manifest-v2.json"
PARENT_SHA256 = "f0cd1fa2c5fae10de17615082d27963876d01b5c02c5f7ddd29e92d3614cde3e"
BOOT_EPOCH = 1792737027
OLD_DEVICE = 16777232
NEW_DEVICE = 16777234
LOCK_INODE = 135900020
LINK_OCCURRENCES = 57
RECIPE_FIELDS = {"protocol", "root", "dataset_id", "output", "original_manifest", "rebinding",
                 "source_pins", "accepted_evidence"}
ENVELOPE_FIELDS = RECIPE_FIELDS | {"recipe_sha256", "operational_manifest"}
BINDING_PROTOCOL = "industrial-sj-boot-device-rebinding-v1"
OBSERVATION_PROTOCOL = "industrial-sj-boot-device-observation-v1"
frozen = original.frozen
np = original.np


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def read_pinned(root, ref):
    return json.loads(frozen.checked(root, ref))


def current_boot_epoch():
    value = subprocess.run(["/usr/sbin/sysctl", "-n", "kern.boottime"],
                           check=True, capture_output=True, text=True)
    match = re.search(r"\bsec\s*=\s*(\d+)\s*[,}]", value.stdout)
    require(match is not None, "current_boot_measurement")
    return int(match.group(1))


def timestamp(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(parsed.tzinfo is not None, "observation_timezone")
    return parsed.timestamp()


def checked_refs(root, refs, required):
    require(isinstance(refs, dict) and set(refs) >= set(required), "proof_ref_population")
    for ref in refs.values():
        frozen.checked(root, ref)
    # Extra proof refs may be raw source or empty bytes; never assume JSON.
    return {name: read_pinned(root, refs[name]) for name in required}


def check_sources(root, recipe):
    require(set(recipe["source_pins"]) == set(SOURCES), "adapter_source_population")
    for path, digest in recipe["source_pins"].items():
        require(frozen.record(root, path)["sha256"] == digest, "adapter_source_changed")
    original.verify_module(sys.modules[__name__], frozen.path_at(root, SOURCES[0]))


def source_admissions(root, recipe):
    refs = recipe["accepted_evidence"]
    values = checked_refs(root, refs, ("source_contract", "independent_source_review", "root_source_acceptance"))
    contract = values["source_contract"]
    require(contract["protocol"] == VERSION and
            contract["scope"] == "SOURCE_ONLY_NO_ACTUAL_PREPARE_OR_EXECUTION" and
            contract["source_pins"] == recipe["source_pins"], "source_contract")
    review, acceptance = values["independent_source_review"], values["root_source_acceptance"]
    require(review["status"] == "PASS_INDEPENDENT_INDUSTRIAL_REBOOT_SOURCE" and
            review["blocking_findings"] == [] and review["source_pins"] == recipe["source_pins"] and
            review["source_contract_sha256"] == refs["source_contract"]["sha256"],
            "independent_source_review")
    require(acceptance["status"] == "PASS_ROOT_INDUSTRIAL_REBOOT_SOURCE" and
            acceptance["source_pins"] == recipe["source_pins"] and
            acceptance["source_contract_sha256"] == refs["source_contract"]["sha256"] and
            acceptance["independent_review_sha256"] == refs["independent_source_review"]["sha256"],
            "root_source_acceptance")


def device_changes(manifest):
    require(manifest["lock"] == {"path": "work/overnight/gpu.lock", "device": OLD_DEVICE,
            "inode": LOCK_INODE, "bytes": 0}, "historical_lock_identity")
    changes = [{"field": ["lock", "device"], "old": OLD_DEVICE, "new": NEW_DEVICE}]
    for model in sorted(manifest["assets"]):
        for name, asset in sorted(manifest["assets"][model]["assets"].items()):
            for index, link in enumerate(asset["symlink_chain"]):
                require(set(link) == {"path", "target", "device", "inode", "mtime_ns", "ctime_ns"}
                        and link["device"] == OLD_DEVICE, "historical_symlink_identity")
                changes.append({"field": ["assets", model, "assets", name, "symlink_chain", index, "device"],
                                "old": OLD_DEVICE, "new": NEW_DEVICE})
    require(len(changes) == LINK_OCCURRENCES + 1, "exact_device_change_population")
    return changes


def at_field(value, keys):
    for key in keys:
        value = value[key]
    return value


def project_manifest(manifest, changes):
    require(changes == device_changes(manifest), "exact_device_change_list")
    projected = original.detached(manifest)
    for change in changes:
        parent = at_field(projected, change["field"][:-1])
        require(parent[change["field"][-1]] == change["old"], "device_predecessor")
        parent[change["field"][-1]] = change["new"]
    restored = original.detached(projected)
    for change in changes:
        at_field(restored, change["field"][:-1])[change["field"][-1]] = change["old"]
    require(restored == manifest and projected != manifest, "projection_exact_diff")
    return projected


def check_live_metadata(root, operational, binding):
    require(current_boot_epoch() == BOOT_EPOCH, "boot_changed")
    lock = binding["current_lock"]
    require(lock["path"] == operational["lock"]["path"] and lock["bytes"] == 0,
            "current_lock_content")
    frozen.checked(root, lock)
    path = frozen.path_at(root, lock["path"])
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        opened, current = os.fstat(fd), path.stat(follow_symlinks=False)
        def fields(value):
            return (value.st_dev, value.st_ino, value.st_size, stat.S_IMODE(value.st_mode),
                    value.st_uid, value.st_gid, value.st_nlink)
        expected = (NEW_DEVICE, LOCK_INODE, 0, binding["current_mode"],
                    binding["lock_uid"], binding["lock_gid"], binding["lock_nlink"])
        require(stat.S_ISREG(opened.st_mode) and fields(opened) == fields(current) == expected,
                "live_lock_descriptor_identity")
        require(os.read(fd, 1) == b"" and fields(os.fstat(fd)) == fields(path.stat(follow_symlinks=False)),
                "live_lock_changed")
    finally:
        os.close(fd)
    for spec in operational["assets"].values():
        for asset in spec["assets"].values():
            observed = original.asset_chain(root, asset["logical_path"])
            require(observed == {key: asset[key] for key in ("logical_path", "symlink_chain", "resolved_path")},
                    "live_asset_chain_changed")


def checked_binding(root, recipe, manifest):
    binding = read_pinned(root, recipe["rebinding"])
    changes = device_changes(manifest)
    require(binding["protocol"] == BINDING_PROTOCOL and
            binding["status"] == "PASS_ROOT_INDUSTRIAL_BOOT_DEVICE_REBINDING" and
            binding["original_manifest"] == recipe["original_manifest"] and
            binding["boot_epoch_seconds"] == BOOT_EPOCH and
            binding["old_device"] == OLD_DEVICE and binding["new_device"] == NEW_DEVICE and
            binding["changed_fields"] == changes and binding["cause"] == "UNKNOWN" and
            binding["historical_volume_identity_attested"] is False,
            "boot_device_binding")
    refs = binding["refs"]
    evidence = checked_refs(root, refs, ("observation", "independent_review", "boot_observation"))
    review, observation, boot = (evidence[key] for key in ("independent_review", "observation", "boot_observation"))
    require(review["status"] == "PASS_INDEPENDENT_INDUSTRIAL_BOOT_DEVICE_REBINDING" and
            review["blocking_findings"] == [] and
            review["observation_sha256"] == refs["observation"]["sha256"] and
            review["original_manifest_sha256"] == recipe["original_manifest"]["sha256"] and
            review["changed_fields_sha256"] == frozen.digest(changes), "independent_binding_review")
    require(boot["schema"] == "vidore-laya-reboot-process-observation-v1" and
            boot["boot_epoch_seconds"] == BOOT_EPOCH and boot["reboot_cause"] == "UNKNOWN",
            "boot_evidence")
    projected = project_manifest(manifest, changes)
    require(observation["protocol"] == OBSERVATION_PROTOCOL and
            observation["status"] == "OBSERVED_POSTBOOT_LOCK_AND_ASSET_METADATA" and
            observation["original_manifest"] == recipe["original_manifest"] and
            observation["boot_epoch_seconds"] == BOOT_EPOCH and
            observation["boot_observation"] == refs["boot_observation"] and
            timestamp(observation["observed_at_utc"]) > BOOT_EPOCH and
            observation["changed_fields"] == changes and
            observation["original_lock"] == manifest["lock"] and
            observation["operational_lock"] == projected["lock"] and
            observation["current_lock"] == binding["current_lock"] and
            observation["current_mode"] == binding["current_mode"] == 0o644 and
            observation["lock_uid"] == binding["lock_uid"] and
            observation["lock_gid"] == binding["lock_gid"] and
            observation["lock_nlink"] == binding["lock_nlink"] == 1 and
            observation["descriptor_matches_path_before_and_after"] is True and
            observation["lock_created_written_unlinked_replaced"] is False and
            observation["immutable_files_modified"] is False and
            observation["historical_volume_identity_attested"] is False and
            observation["cause"] == "UNKNOWN", "binding_observation")
    expected_links = [{"field": change["field"][:-1],
                       "original": at_field(manifest, change["field"][:-1]),
                       "operational": at_field(projected, change["field"][:-1])} for change in changes[1:]]
    require(observation["symlink_occurrences"] == expected_links, "observed_symlink_occurrences")
    check_live_metadata(root, projected, binding)
    return binding, projected


def recipe_scope(root, recipe):
    require(set(recipe) == RECIPE_FIELDS and recipe["protocol"] == VERSION and
            recipe["root"] == str(root) and recipe["dataset_id"] == DATASET and
            recipe["output"] == DELTA, "recipe_scope")
    require(recipe["original_manifest"]["path"] == PARENT_PATH and
            recipe["original_manifest"]["sha256"] == PARENT_SHA256, "original_manifest_identity")


def load(root, manifest):
    require(set(manifest) == ENVELOPE_FIELDS, "envelope_fields")
    recipe = {key: manifest[key] for key in RECIPE_FIELDS}
    recipe_scope(root, recipe)
    require(frozen.digest(recipe) == manifest["recipe_sha256"], "recipe_digest")
    check_sources(root, recipe)
    source_admissions(root, recipe)
    parent = read_pinned(root, recipe["original_manifest"])
    require(frozen.digest(parent) == PARENT_SHA256, "original_canonical_manifest")
    expected = original.recipe(root, parent["source_pins"])
    require(all(parent.get(key) == value for key, value in expected.items()) and
            parent["recipe_sha256"] == frozen.digest(expected), "historical_recipe_identity")
    binding, projected = checked_binding(root, recipe, parent)
    require(manifest["operational_manifest"] == projected, "operational_manifest_diff")
    return recipe, parent, projected, binding


def prepare(root, request, request_sha256):
    request = original.detached(request)
    recipe_scope(root, request)
    require(frozen.digest(request) == request_sha256, "preparation_recipe_digest")
    check_sources(root, request)
    source_admissions(root, request)
    parent = read_pinned(root, request["original_manifest"])
    _, projected = checked_binding(root, request, parent)
    envelope = {**request, "recipe_sha256": request_sha256, "operational_manifest": projected}
    validate(root, envelope)
    return envelope


def validate(root, envelope):
    _, parent, manifest, _ = load(root, envelope)
    # Preserve the original fixed scientific condition; the operational copy has
    # only the admitted device-number differences checked above.
    if parent["runtime"] != original.runtime():
        raise ValueError("Recipe/runtime differs")
    modules = original.sources(root, parent["source_pins"])
    if original.lock_record(root) != manifest["lock"]:
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
    if manifest["assets"] != {key: original.asset_inventory(root, key) for key in MODEL_REVISIONS}:
        raise ValueError("Local asset closure changed")
    if manifest["inherited"] != original.archive_inputs(root):
        raise ValueError("Inherited evidence changed")
    ids, query_ids = [str(r["id"]) for r in corpus], [str(q["id"]) for q in queries]
    for channel, identity in INHERITED_IDS.items():
        payload = json.loads(original.read_member(root, manifest, channel, "ranking"))
        frozen._rankings(payload, identity, ids, query_ids, exclusions, bm25=channel == "B")
        report = json.loads(original.read_member(root, manifest, channel, "report"))
        if (report["dataset"]["identity"] != DATASET_IDENTITY
                or report["channels"][channel]["status"] != "completed"
                or report["channels"][channel]["identity"] != identity):
            raise ValueError("Inherited report/channel identity differs")
    return modules, corpus, queries, qrels, exclusions


def check_execution_admission(root, envelope, digest, channel, admission_ref):
    admission = read_pinned(root, admission_ref)
    require(channel in ("J", "S") and admission["protocol"] == VERSION and
            admission["status"] == "PASS_ROOT_INDUSTRIAL_REBOOT_EXECUTION_AUTHORIZED" and
            admission["actual_model_execution_authorized"] is True and
            admission["channel"] == channel and admission["manifest_sha256"] == digest and
            admission["source_pins"] == envelope["source_pins"] and
            admission["rebinding"] == envelope["rebinding"] and
            admission["boot_epoch_seconds"] == BOOT_EPOCH == current_boot_epoch(),
            "execution_admission")
    proof = checked_refs(root, admission["refs"], ("independent_manifest_review", "heavy_completion"))
    review = proof["independent_manifest_review"]
    require(review["status"] == "PASS_INDEPENDENT_INDUSTRIAL_REBOOT_MANIFEST" and
            review["blocking_findings"] == [] and review["manifest_sha256"] == digest,
            "independent_manifest_review")
    heavy = proof["heavy_completion"]
    require(heavy["status"] == "PASS_ROOT_ALL_SIX_LAYA_HEAVY_CLOSED" and
            heavy["dataset_count"] == 6 and heavy["active_heavy_workers"] == 0 and
            heavy["actual_waits_verified"] is True, "six_laya_heavy_completion")
    checked_refs(root, heavy["refs"], ())
    prior = admission["prior_j_stage"]
    if channel == "J":
        require(prior is None, "unexpected_prior_stage")
    else:
        require(set(prior) == {"closed", "wait", "execution_claim"} and
                prior["closed"]["path"] == DELTA + "/J/closed.json", "prior_j_stage_refs")
        closed, wait, claim = (read_pinned(root, prior[key]) for key in ("closed", "wait", "execution_claim"))
        require(closed["status"] == "completed" and closed["manifest_sha256"] == digest and
                closed["identity"] == stage_identity(envelope, digest, "J") and
                wait["actual_wait"] is True and wait["exit_code"] == 0 and wait["signal"] is None and
                wait["channel"] == "J" and wait["manifest_sha256"] == digest and
                wait["closed"] == prior["closed"] and
                prior["execution_claim"]["path"] == DELTA + "/J/execution.claim.json" and
                closed["files"]["execution.claim.json"] == prior["execution_claim"] and
                claim["protocol"] == VERSION and claim["channel"] == "J" and
                claim["manifest_sha256"] == digest and claim["boot_epoch_seconds"] == BOOT_EPOCH and
                type(claim["pid"]) is int and claim["pid"] > 0 and
                wait["pid"] == claim["pid"] and wait["execution_claim"] == prior["execution_claim"],
                "prior_j_true_wait")
    return admission


def stage_identity(manifest, manifest_sha256, channel):
    return frozen.digest({"condition": VERSION, "manifest_sha256": manifest_sha256, "channel": channel,
                          "dataset": DATASET_IDENTITY, "hardware": HARDWARE[channel], "top_k": 100,
                          "original_manifest_sha256": manifest["original_manifest"]["sha256"],
                          "rebinding_sha256": manifest["rebinding"]["sha256"]})


def execute(root, envelope, manifest_sha256, channel, admission_ref):
    envelope = original.detached(envelope)
    require(frozen.digest(envelope) == manifest_sha256, "exact_execution_envelope")
    check_execution_admission(root, envelope, manifest_sha256, channel, admission_ref)
    _, _, manifest, _ = load(root, envelope)
    resume = False
    if channel not in ("S", "J"):
        raise ValueError("Exact admitted manifest and S/J stage required")
    modules, corpus, queries, _, exclusions = validate(root, envelope)
    # Enforce offline package behavior in this isolated child, never in a shared adapter module.
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[key] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0":
        raise ValueError("MPS fallback environment forbidden")
    output = frozen.path_at(root, DELTA)
    with original.ExistingLock(root, manifest["lock"]):
        if not output.exists():
            original.mkdir(output)
            frozen.exclusive_write(output / "manifest.json", frozen.canonical(envelope))
        elif original.sha(frozen.file_bytes(output / "manifest.json")) != manifest_sha256:
            raise ValueError("Existing output manifest differs")
        stage = frozen.path_at(root, DELTA + "/" + channel)
        if stage.exists():
            raise ValueError("Existing stage is immutable; automatic retry forbidden")
        original.mkdir(stage)
        for role in ("document", "query"):
            original.mkdir(stage / role)
        frozen.exclusive_write(stage / "execution.claim.json", frozen.canonical({
            "protocol": VERSION, "channel": channel, "pid": os.getpid(), "parent_pid": os.getppid(),
            "manifest_sha256": manifest_sha256, "execution_admission": admission_ref,
            "boot_epoch_seconds": BOOT_EPOCH}))
        # New process-specific metadata prevents ST rewrites from touching original or previous caches.
        check_execution_admission(root, envelope, manifest_sha256, channel, admission_ref)
        load(root, envelope)
        paths = original.local_models(root, manifest, channel, stage / ("model-view-" + str(os.getpid())))
        adapter = original.make_adapter(modules, channel, paths)
        native_adapter_identity = adapter.identity
        # Native identities contain private paths. Own jobs use a separately
        # named channel identity bound to the manifest, source and asset closure.
        identity = stage_identity(envelope, manifest_sha256, channel)
        try:
            documents, doc_usage = original.encode_jobs(root, manifest, channel, "document", corpus, stage / "document",
                                               adapter, resume, manifest_sha256, identity)
            query_values, query_usage = original.encode_jobs(root, manifest, channel, "query", queries, stage / "query",
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
                      "none_cells_completed": 0, "laya_cells_completed": 0,
                      "original_manifest": envelope["original_manifest"], "rebinding": envelope["rebinding"],
                      "execution_admission": admission_ref,
                      "operational_changes": device_changes(read_pinned(root, envelope["original_manifest"])),
                      "future_handoff_requires_separate_reviewed_adapter": True}
            # Revalidate original bytes before closing a result. This never repairs changed inputs.
            validate(root, envelope)
            check_execution_admission(root, envelope, manifest_sha256, channel, admission_ref)
            frozen.exclusive_write(stage / "rankings.json", frozen.canonical(payload))
            frozen.exclusive_write(stage / "report.json", frozen.canonical(report))
            files = {name: frozen.record(root, (stage / name).relative_to(root).as_posix())
                     for name in ("rankings.json", "report.json", "scoring.claim.json", "execution.claim.json")}
            terminals = [frozen.record(root, p.relative_to(root).as_posix()) for role in ("document", "query")
                         for p in sorted((stage / role).glob("*.terminal.json"))]
            closed = {"status": "completed", "identity": identity, "manifest_sha256": manifest_sha256,
                      "protocol": VERSION, "execution_admission": admission_ref,
                      "original_manifest": envelope["original_manifest"], "rebinding": envelope["rebinding"],
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
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--channel", choices=("J", "S"))
    parser.add_argument("--admission")
    parser.add_argument("--admission-sha256")
    args = parser.parse_args(argv)
    sys.dont_write_bytecode = True
    ref = frozen.record(args.root, args.input)
    require(ref["sha256"] == args.sha256, "cli_input_bytes")
    value = read_pinned(args.root, ref)
    if args.mode == "prepare":
        require(not args.channel and not args.admission and not args.admission_sha256,
                "preparation_cannot_execute")
        result = prepare(args.root, value, args.sha256)
        path = frozen.path_at(args.root, BASE + "/prepared-manifest-v3.json")
        frozen.exclusive_write(path, frozen.canonical(result))
        print(json.dumps({"manifest": frozen.record(args.root, path.relative_to(args.root).as_posix()),
                          "canonical_sha256": frozen.digest(result), "model_calls": 0}))
    else:
        require(args.channel and args.admission and args.admission_sha256, "explicit_execution_admission")
        admission_ref = frozen.record(args.root, args.admission)
        require(admission_ref["sha256"] == args.admission_sha256, "cli_admission_bytes")
        print(json.dumps({"closed": execute(args.root, value, args.sha256, args.channel, admission_ref)}))
    return 0

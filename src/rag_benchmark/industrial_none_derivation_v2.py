"""CPU-only None126 adapter for reviewed Industrial S/J V3 closures.

Original CPU science helpers are called normally. No model assets, vector payload,
GPU lock or model constructors are read or invoked. Historical V2 manifest bytes
remain the original provenance; V3 envelope/stage identities stay distinct.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
from pathlib import Path
import re
import sqlite3
import stat
import sys

from . import industrial_none_derivation_v1 as previous
from . import industrial_mps_sj_v3 as upstream

origin = previous.origin
frozen = previous.frozen
VERSION = "industrial-none-derivation-v2"
BASE = "work/" + VERSION
OUTPUT = BASE + "/actual-v2"
CHANNELS = previous.CHANNELS
MODES = previous.MODES
METRICS = previous.METRICS
SOURCES = previous.SOURCES + upstream.SOURCES + (
    "src/rag_benchmark/industrial_none_derivation_v2.py", "scripts/run_industrial_none_derivation_v2.py")
MPS_MANIFEST = upstream.BASE + "/prepared-manifest-v3.json"
STAGE_CLOSED = {c: upstream.DELTA + "/" + c + "/closed.json" for c in "SJ"}
UPSTREAM_KEYS = {"original_manifest", "operational_envelope", "boot_rebinding", "sj_source_contract",
    "sj_source_review", "sj_root_acceptance", "sj_actual_config_requirements", "j_execution_admission",
    "s_execution_admission", "j_execution_claim", "s_execution_claim", "j_actual_wait", "s_actual_wait",
    "sj_actual_output_review", "sj_root_output_acceptance", "none_source_contract", "none_source_review",
    "none_root_source_acceptance"}
SJ_SOURCE_REVIEW_SHA = "299f9f654771ac46bd7bf137158f5eb290c10fa83c43f7f2f65e477312487b52"
SJ_REQUIREMENTS_SHA = "eed737d4389f3cc044eab1c909658502f925e1fe0fa5e5fe5048f3e335aed202"
SJ_CONTRACT_SHA = "270ce47b0783e7a24cc92775e178e8ee3b481985c190d2ed87805195df51f3ab"
configuration = previous.configuration
derive_rows = previous.derive_rows
strict_ranking = previous.strict_ranking
metrics_equal = previous.metrics_equal
prior_agreement = previous.prior_agreement
archive_bundle = previous.archive_bundle


def require(condition, reason):
    previous.require(condition, reason)


def metadata_ref(ref):
    require(isinstance(ref, dict) and set(ref) == {"path", "bytes", "sha256"} and
            isinstance(ref["path"], str) and type(ref["bytes"]) is int and ref["bytes"] >= 0 and
            isinstance(ref["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", ref["sha256"]), "ref_schema")
    path = Path(ref["path"])
    require(not path.is_absolute() and path.as_posix() == ref["path"] and
            not any(p in ("..", ".") for p in path.parts), "ref_path")
    return ref


def cpu_readable(ref):
    metadata_ref(ref)
    name = ref["path"]
    require(not name.startswith(".cache/models/") and not name.endswith((".npy", ".safetensors", ".bin"))
            and name != "work/overnight/gpu.lock" and "/model-view-" not in name,
            "cpu_metadata_boundary")


class Proofs:
    """Exact metadata freshness ledger; vector/lock descriptors never become reads."""
    def __init__(self, root):
        self.root, self.refs, self.descriptors = root, {}, {}

    def raw(self, ref):
        cpu_readable(ref)
        prior = self.refs.get(ref["path"])
        require(prior is None or prior == ref, "conflicting_proof_ref")
        result = frozen.checked(self.root, ref)
        self.refs[ref["path"]] = ref
        return result

    def read(self, ref):
        return json.loads(self.raw(ref))

    def descriptor(self, ref):
        metadata_ref(ref)
        old = self.descriptors.get(ref["path"])
        require(old is None or old == ref, "conflicting_payload_descriptor")
        self.descriptors[ref["path"]] = ref

    def embedded(self, value):
        if isinstance(value, dict):
            if set(value) == {"path", "bytes", "sha256"}:
                metadata_ref(value)
                if value["path"] in self.descriptors:
                    require(value == self.descriptors[value["path"]], "payload_descriptor_changed")
                else:
                    self.raw(value)
            else:
                for item in value.values():
                    self.embedded(item)
        elif isinstance(value, list):
            for item in value:
                self.embedded(item)

    def fresh(self):
        for ref in self.refs.values():
            frozen.checked(self.root, ref)


def sources(root, pins):
    require(set(pins) == set(SOURCES), "exact_cpu_source_closure")
    for path in SOURCES:
        require(frozen.record(root, path)["sha256"] == pins[path], "cpu_source_changed")
    pure = previous.sources(root, {p: pins[p] for p in previous.SOURCES})
    origin.verify_module(upstream, frozen.path_at(root, upstream.SOURCES[0]))
    origin.verify_module(sys.modules[__name__], frozen.path_at(root, "src/rag_benchmark/industrial_none_derivation_v2.py"))
    return pure


def recipe(root, pins):
    return {"version": VERSION, "root": str(root), "source_pins": pins, "output": OUTPUT,
            "dataset_id": origin.DATASET, "dataset_identity": origin.DATASET_IDENTITY,
            "mps_manifest_path": MPS_MANIFEST, "original_manifest_path": upstream.PARENT_PATH,
            "stage_closed_paths": STAGE_CLOSED, "channels": CHANNELS, "budget": 100, "rrf_k": 60,
            "budget_modes": list(MODES), "candidate_grid_for_laya": [20, 50, 100],
            "expected_none_cells": 126, "expected_original_standalone_cells": 8}


def source_admissions(proofs, refs, request):
    contract, review, accepted = [proofs.read(refs[k]) for k in
        ("none_source_contract", "none_source_review", "none_root_source_acceptance")]
    require(contract["protocol"] == VERSION and contract["source_pins"] == request["source_pins"] and
            contract["scope"] == "SOURCE_ONLY_NO_ACTUAL_DERIVATION", "none_source_contract")
    require(review["status"] == "PASS_INDEPENDENT_INDUSTRIAL_NONE_V2_SOURCE" and review["blocking_findings"] == [] and
            review["source_contract_sha256"] == refs["none_source_contract"]["sha256"], "none_source_review")
    require(accepted["status"] == "PASS_ROOT_INDUSTRIAL_NONE_V2_SOURCE" and
            accepted["source_contract_sha256"] == refs["none_source_contract"]["sha256"] and
            accepted["independent_review_sha256"] == refs["none_source_review"]["sha256"], "none_source_acceptance")
    for value in (contract, review, accepted):
        proofs.embedded(value)


def admitted_inputs(proofs, request, admission):
    require(set(admission) == {"mps_manifest_sha256", "closed_sha256", "accepted_evidence", "upstream_evidence"}
            and set(admission["closed_sha256"]) == {"S", "J"} and admission["accepted_evidence"], "cpu_admission_fields")
    refs = admission["upstream_evidence"]
    require(set(refs) == UPSTREAM_KEYS, "upstream_proof_population")
    require(all(ref in admission["accepted_evidence"] for ref in refs.values()), "upstream_proof_closure")
    for ref in admission["accepted_evidence"]:
        proofs.raw(ref)
    source_admissions(proofs, refs, request)
    envelope_ref, old_ref = refs["operational_envelope"], refs["original_manifest"]
    require(envelope_ref["path"] == MPS_MANIFEST and envelope_ref["sha256"] == admission["mps_manifest_sha256"] and
            old_ref["path"] == upstream.PARENT_PATH and old_ref["sha256"] == upstream.PARENT_SHA256, "input_manifest_refs")
    envelope, old = proofs.read(envelope_ref), proofs.read(old_ref)
    require(set(envelope) == upstream.ENVELOPE_FIELDS and envelope["protocol"] == upstream.VERSION and
            envelope["root"] == str(proofs.root) and envelope["dataset_id"] == origin.DATASET and
            envelope["output"] == upstream.DELTA and envelope["original_manifest"] == old_ref and
            envelope["rebinding"] == refs["boot_rebinding"] and
            frozen.digest(envelope) == envelope_ref["sha256"] and frozen.digest(old) == upstream.PARENT_SHA256,
            "operational_envelope_identity")
    r = {key: envelope[key] for key in upstream.RECIPE_FIELDS}
    require(envelope["recipe_sha256"] == frozen.digest(r) and
            set(envelope["source_pins"]) == set(upstream.SOURCES) and
            all(envelope["source_pins"][p] == request["source_pins"][p] for p in upstream.SOURCES), "upstream_source_recipe")
    expected = origin.recipe(proofs.root, old["source_pins"])
    require(all(old.get(k) == v for k, v in expected.items()) and old["recipe_sha256"] == frozen.digest(expected) and
            set(old["source_pins"]) == set(origin.SOURCES) and
            all(old["source_pins"][p] == request["source_pins"][p] for p in origin.SOURCES), "original_fixed_recipe")
    changes = upstream.device_changes(old)
    require(envelope["operational_manifest"] == upstream.project_manifest(old, changes), "exact_58_device_projection")
    binding = proofs.read(refs["boot_rebinding"])
    require(binding["protocol"] == upstream.BINDING_PROTOCOL and
            binding["status"] == "PASS_ROOT_INDUSTRIAL_BOOT_DEVICE_REBINDING" and
            binding["original_manifest"] == old_ref and binding["changed_fields"] == changes and
            binding["boot_epoch_seconds"] == upstream.BOOT_EPOCH and binding["old_device"] == upstream.OLD_DEVICE and
            binding["new_device"] == upstream.NEW_DEVICE and binding["cause"] == "UNKNOWN" and
            binding["historical_volume_identity_attested"] is False, "historical_binding_seal")
    # The accepted historical empty lock descriptor is carried, never reopened.
    lock = binding["current_lock"]
    require(lock == {"path": "work/overnight/gpu.lock", "bytes": 0, "sha256": origin.sha(b"")}, "empty_lock_descriptor")
    proofs.descriptor(lock)
    proof_refs = binding["refs"]
    require(set(proof_refs) >= {"observation", "independent_review", "boot_observation"}, "binding_proof_population")
    review = proofs.read(proof_refs["independent_review"])
    require(review["status"] == "PASS_INDEPENDENT_INDUSTRIAL_BOOT_DEVICE_REBINDING" and review["blocking_findings"] == [] and
            review["observation_sha256"] == proof_refs["observation"]["sha256"] and
            review["original_manifest_sha256"] == old_ref["sha256"] and
            review["changed_fields_sha256"] == frozen.digest(changes), "binding_independent_review")
    observation, boot = proofs.read(proof_refs["observation"]), proofs.read(proof_refs["boot_observation"])
    require(observation["protocol"] == upstream.OBSERVATION_PROTOCOL and
            observation["status"] == "OBSERVED_POSTBOOT_LOCK_AND_ASSET_METADATA" and
            observation["original_manifest"] == old_ref and observation["boot_epoch_seconds"] == upstream.BOOT_EPOCH and
            observation["boot_observation"] == proof_refs["boot_observation"] and
            observation["changed_fields"] == changes and boot["boot_epoch_seconds"] == upstream.BOOT_EPOCH,
            "historical_binding_observation")
    require(refs["sj_source_contract"]["sha256"] == SJ_CONTRACT_SHA and
            refs["sj_source_review"]["sha256"] == SJ_SOURCE_REVIEW_SHA and
            envelope["accepted_evidence"]["source_contract"] == refs["sj_source_contract"] and
            envelope["accepted_evidence"]["independent_source_review"] == refs["sj_source_review"] and
            envelope["accepted_evidence"]["root_source_acceptance"] == refs["sj_root_acceptance"], "accepted_sj_source_refs")
    sj_contract, sj_review, sj_acceptance = [proofs.read(refs[k]) for k in
        ("sj_source_contract", "sj_source_review", "sj_root_acceptance")]
    require(sj_review["status"] == "PASS_INDEPENDENT_INDUSTRIAL_REBOOT_SOURCE" and sj_review["blocking_findings"] == [] and
            sj_acceptance["status"] == "PASS_ROOT_INDUSTRIAL_REBOOT_SOURCE" and
            sj_acceptance["independent_review_sha256"] == refs["sj_source_review"]["sha256"] and
            all(v["source_contract_sha256"] == SJ_CONTRACT_SHA and v["source_pins"] == envelope["source_pins"]
                for v in (sj_review, sj_acceptance)) and sj_contract["source_pins"] == envelope["source_pins"], "sj_source_acceptance")
    requirements = proofs.read(refs["sj_actual_config_requirements"])
    require(refs["sj_actual_config_requirements"]["sha256"] == SJ_REQUIREMENTS_SHA and
            requirements["schema"] == "independent-industrial-reboot-actual-config-requirements-v1" and
            requirements["status"] == "SOURCE_REVIEW_REQUIREMENTS_NOT_ACTUAL_ADMISSION" and
            requirements["independent_source_review"] == refs["sj_source_review"], "sj_actual_config_addendum")
    for value in (r["accepted_evidence"], binding, review, observation, boot, sj_contract, sj_review, sj_acceptance, requirements):
        proofs.embedded(value)
    return refs, envelope, old


def stage_bundle(proofs, envelope, channel, refs, closed_ref, corpus, queries):
    digest = frozen.digest(envelope)
    identity = upstream.stage_identity(envelope, digest, channel)
    stage = upstream.DELTA + "/" + channel
    closed = proofs.read(closed_ref)
    require(closed_ref["path"] == STAGE_CLOSED[channel] and closed["status"] == "completed" and
            closed["protocol"] == upstream.VERSION and closed["manifest_sha256"] == digest and
            closed["identity"] == identity and closed["original_manifest"] == refs["original_manifest"] and
            closed["rebinding"] == refs["boot_rebinding"] and
            set(closed["files"]) == {"rankings.json", "report.json", "scoring.claim.json", "execution.claim.json"},
            "stage_closure_identity")
    payloads = {}
    for name, ref in closed["files"].items():
        require(ref["path"] == stage + "/" + name, "stage_file_namespace")
        payloads[name] = proofs.read(ref)
    key = channel.lower()
    claim_ref, wait_ref, execution_ref = [refs[key + suffix] for suffix in
        ("_execution_claim", "_actual_wait", "_execution_admission")]
    require(claim_ref == closed["files"]["execution.claim.json"] and closed["execution_admission"] == execution_ref,
            "stage_claim_admission_refs")
    claim, wait = payloads["execution.claim.json"], proofs.read(wait_ref)
    require(claim["protocol"] == upstream.VERSION and claim["channel"] == channel and
            claim["manifest_sha256"] == digest and claim["execution_admission"] == execution_ref and
            claim["boot_epoch_seconds"] == upstream.BOOT_EPOCH and type(claim["pid"]) is int and claim["pid"] > 0 and
            wait["actual_wait"] is True and type(wait["exit_code"]) is int and wait["exit_code"] == 0 and
            wait["signal"] is None and wait["channel"] == channel and wait["manifest_sha256"] == digest and
            wait["pid"] == claim["pid"] and wait["execution_claim"] == claim_ref and wait["closed"] == closed_ref,
            "stage_true_wait_pid")
    execution = proofs.read(execution_ref)
    require(execution["protocol"] == upstream.VERSION and
            execution["status"] == "PASS_ROOT_INDUSTRIAL_REBOOT_EXECUTION_AUTHORIZED" and
            execution["actual_model_execution_authorized"] is True and execution["channel"] == channel and
            execution["manifest_sha256"] == digest and execution["source_pins"] == envelope["source_pins"] and
            execution["rebinding"] == refs["boot_rebinding"] and execution["boot_epoch_seconds"] == upstream.BOOT_EPOCH,
            "stage_execution_admission")
    require(payloads["scoring.claim.json"] == {"identity": identity, "manifest": digest,
        "query_count": len(queries), "top_k": 100}, "scoring_claim_identity")
    terminal_refs = closed["terminal_records"]
    expected = {stage + "/" + role + f"/{i:06d}.terminal.json": (role, i, row)
                for role, rows in (("document", corpus), ("query", queries)) for i, row in enumerate(rows)}
    require(len(terminal_refs) == len(expected) and {ref["path"] for ref in terminal_refs} == set(expected) and
            closed["terminal_sha256"] == frozen.digest(terminal_refs), "terminal_population")
    for ref in terminal_refs:
        role, index, row = expected[ref["path"]]
        terminal = proofs.read(ref)
        require(terminal["status"] == "completed", "terminal_status")
        job_claim = {"manifest": digest, "channel": channel, "role": role, "index": index,
                     "row_sha256": frozen.digest(row), "channel_identity": identity}
        claim_path = stage + "/" + role + f"/{index:06d}.claim.json"
        job_ref = frozen.record(proofs.root, claim_path)
        require(proofs.read(job_ref) == job_claim and job_ref["sha256"] == terminal["claim_sha256"] == frozen.digest(job_claim),
                "terminal_claim_binding")
        vector = terminal["output"]
        metadata_ref(vector)
        require(vector["path"] == stage + "/" + role + f"/{index:06d}.npy" and
                10 < vector["bytes"] <= (768 if channel == "J" else 8192 * 128) * 4 + 4108,
                "vector_descriptor_scope")
        proofs.descriptor(vector)
        item = {"text": row.get("text", ""), "media": row.get("media", {})}
        origin.checked_usage(terminal["usage"], item, role, channel)
        if channel == "S":
            proof = terminal["projection_verification"]
            require(proof.get("projection_verified") is True and
                    0 <= proof.get("max_absolute_merge_error", 1) <= 1e-6, "projection_metadata")
        proofs.embedded(terminal)
    report, ranking = payloads["report.json"], payloads["rankings.json"]
    require(report["condition"] == upstream.VERSION and report["manifest_sha256"] == digest and
            report["dataset"]["identity"] == origin.DATASET_IDENTITY and report["hardware"] == origin.HARDWARE and
            report["channels"][channel] == {"status": "completed", "identity": identity} and
            report["original_manifest"] == refs["original_manifest"] and report["rebinding"] == refs["boot_rebinding"] and
            report["execution_admission"] == execution_ref and report["none_cells_completed"] == report["laya_cells_completed"] == 0,
            "stage_report_provenance")
    require(ranking["identity"] == identity and ranking["condition"] == upstream.VERSION and
            ranking["manifest_sha256"] == digest and ranking["dataset_identity"] == origin.DATASET_IDENTITY and
            ranking["hardware"] == origin.HARDWARE[channel], "stage_ranking_provenance")
    # Both channels must register their exact vector descriptors before traversing
    # S admission refs, which include the predecessor J full closure.
    return {"identity": identity, "closed": closed_ref, "report": closed["files"]["report.json"],
            "ranking": closed["files"]["rankings.json"]}, report, ranking, execution


def actual_output_acceptance(proofs, refs, admission, envelope, executions):
    review, accepted = [proofs.read(refs[k]) for k in ("sj_actual_output_review", "sj_root_output_acceptance")]
    digest = frozen.digest(envelope)
    waits = {c: refs[c.lower() + "_actual_wait"]["sha256"] for c in "SJ"}
    require(review["status"] == "PASS_INDEPENDENT_INDUSTRIAL_SJ_ACTUAL_OUTPUTS" and review["blocking_findings"] == [] and
            review["manifest_sha256"] == digest and review["original_manifest_sha256"] == upstream.PARENT_SHA256 and
            review["closed_sha256"] == admission["closed_sha256"] and review["wait_sha256"] == waits and
            review["vector_payload_validation_completed"] is True and review["metadata_only_cpu_reuse_allowed"] is True,
            "independent_actual_output_review")
    require(accepted["status"] == "PASS_ROOT_INDUSTRIAL_SJ_REUSE_FOR_NONE" and
            accepted["independent_review_sha256"] == refs["sj_actual_output_review"]["sha256"] and
            accepted["manifest_sha256"] == digest and accepted["closed_sha256"] == admission["closed_sha256"] and
            accepted["wait_sha256"] == waits and accepted["complete_upstream_actual_config_reviewed"] is True,
            "root_actual_output_acceptance")
    config_refs = review["execution_config_reviews"]
    require(set(config_refs) == {"S", "J"} and accepted["execution_config_review_sha256"] ==
            {c: ref["sha256"] for c, ref in config_refs.items()}, "execution_config_reviews")
    for channel, execution in executions.items():
        config = proofs.read(config_refs[channel])
        require(config["status"] == "PASS_INDEPENDENT_INDUSTRIAL_REBOOT_EXECUTION_CONFIG" and
                config["blocking_findings"] == [] and config["channel"] == channel and
                config["manifest_sha256"] == digest and
                config["execution_admission"] == refs[channel.lower() + "_execution_admission"] and
                config["actual_config_requirements"] == refs["sj_actual_config_requirements"] and
                all(config["checks"][key] is True for key in (
                    "all_six_heavy_closures_verified", "all_required_actual_parent_and_child_waits_verified",
                    "complete_transitive_proof_closure_verified")) and
                isinstance(config["refs"], dict) and bool(config["refs"]), "independent_execution_config")
        if channel == "S":
            require(config["checks"]["full_prior_j_nested_closure_verified"] is True,
                    "independent_s_full_j_closure")
        proofs.embedded(config)
        require(set(execution["refs"]) >= {"heavy_completion", "independent_manifest_review"}, "execution_ref_population")
        manifest_review = proofs.read(execution["refs"]["independent_manifest_review"])
        require(manifest_review["status"] == "PASS_INDEPENDENT_INDUSTRIAL_REBOOT_MANIFEST" and
                manifest_review["blocking_findings"] == [] and manifest_review["manifest_sha256"] == digest and
                manifest_review["actual_configuration_requirements"] == refs["sj_actual_config_requirements"],
                "independent_manifest_review")
        proofs.embedded(manifest_review)
        heavy = proofs.read(execution["refs"]["heavy_completion"])
        require(heavy["status"] == "PASS_ROOT_ALL_SIX_LAYA_HEAVY_CLOSED" and heavy["dataset_count"] == 6 and
                heavy["active_heavy_workers"] == 0 and heavy["actual_waits_verified"] is True and
                isinstance(heavy["refs"], dict) and bool(heavy["refs"]), "heavy_completion_proofs")
        prior = execution["prior_j_stage"]
        if channel == "J":
            require(prior is None, "j_initial_stage")
        else:
            require(set(prior) == {"closed", "wait", "execution_claim"} and
                    prior["closed"]["sha256"] == admission["closed_sha256"]["J"] and
                    prior["closed"]["path"] == STAGE_CLOSED["J"] and prior["wait"] == refs["j_actual_wait"] and
                    prior["execution_claim"] == refs["j_execution_claim"], "s_predecessor_j")
        proofs.embedded(heavy)
        proofs.embedded(execution)
    proofs.embedded(review)
    proofs.embedded(accepted)


def inspect_inputs(root, request, admission):
    pure = sources(root, request["source_pins"])
    require(request == recipe(root, request["source_pins"]), "fixed_cpu_recipe")
    proofs = Proofs(root)
    refs, envelope, mps = admitted_inputs(proofs, request, admission)
    metadata, corpus, queries, qrels, exclusions = frozen._dataset(root, mps)
    require((len(corpus), len(queries)) == (5244, 283) and metadata["revision"] == origin.DATASET_REVISION,
            "industrial_population_revision")
    corpus_ids, query_ids = [str(r["id"]) for r in corpus], [str(q["id"]) for q in queries]
    archive, members, raw = archive_bundle(root, mps)
    reports, rankings, channels, ids, agreements = {}, {}, {}, {}, []
    for channel in "BGEN":
        reports[channel] = json.loads(raw[channel]["report"])
        require(reports[channel]["dataset"]["revision"] == origin.DATASET_REVISION and
                reports[channel]["dataset"]["corpus_count"] == len(corpus) and
                reports[channel]["dataset"]["query_count"] == len(queries), "prior_population_revision")
        ids[channel] = origin.INHERITED_IDS[channel]
        rankings[channel] = strict_ranking(json.loads(raw[channel]["ranking"]), ids[channel], corpus_ids,
                                            query_ids, exclusions, channel)
        agreements.extend(prior_agreement(channel, reports[channel], raw[channel]["sqlite"], rankings[channel],
                                          query_ids, qrels, pure, origin.DATASET_IDENTITY))
        channels[channel] = {"identity": ids[channel], **members[channel]}
    executions = {}
    for channel in "SJ":
        closed_ref = frozen.record(root, STAGE_CLOSED[channel])
        require(closed_ref["sha256"] == admission["closed_sha256"][channel], "admitted_stage_closure")
        channels[channel], reports[channel], payload, executions[channel] = stage_bundle(
            proofs, envelope, channel, refs, closed_ref, corpus, queries)
        ids[channel] = channels[channel]["identity"]
        rankings[channel] = strict_ranking(payload, ids[channel], corpus_ids, query_ids, exclusions, channel)
    actual_output_acceptance(proofs, refs, admission, envelope, executions)
    require(len(agreements) == 8, "eight_original_agreements")
    proofs.fresh()
    evidence = {"mps_manifest": refs["operational_envelope"], "mps_manifest_canonical_sha256": frozen.digest(envelope),
                "original_manifest": refs["original_manifest"], "upstream_evidence": refs,
                "metadata_proofs": sorted(proofs.refs.values(), key=lambda r: r["path"]),
                "vector_payloads_read": 0, "vector_descriptors": sum(p.endswith(".npy") for p in proofs.descriptors),
                "archive": archive, "channels": channels, "dataset_id": mps["dataset_id"],
                "dataset_identity": mps["dataset_identity"], "tables": mps["tables"], "media": mps["media"],
                "ordered_id_sha256": mps["ordered_id_sha256"], "channel_identities": ids,
                "hardware": origin.HARDWARE, "prior_agreements": agreements}
    return evidence, rankings, reports, raw, query_ids, qrels, pure


def prepare(root, request, request_sha, admission, admission_sha):
    request, admission = origin.detached(request), origin.detached(admission)
    require(frozen.digest(request) == request_sha and frozen.digest(admission) == admission_sha,
            "Recipe/admission SHA differs")
    evidence, *_ = inspect_inputs(root, request, admission)
    return {"recipe": request, "recipe_sha256": request_sha, "admission": admission,
            "admission_sha256": admission_sha, "runtime": frozen.runtime_identity(), "evidence": evidence}


def validate(root, manifest):
    require(manifest["recipe_sha256"] == frozen.digest(manifest["recipe"]) and
            manifest["admission_sha256"] == frozen.digest(manifest["admission"]) and
            manifest["runtime"] == frozen.runtime_identity(), "CPU manifest runtime/seals differ")
    found = inspect_inputs(root, manifest["recipe"], manifest["admission"])
    require(found[0] == manifest["evidence"], "Sealed CPU input evidence differs")
    return found


def required_output_names():
    return {"manifest.json", "report.json", "results.sqlite3", "matrix.csv", "six-ranking-inputs.json"} | {
        c + suffix for c in CHANNELS for suffix in ("-ranking.json", "-origin-report.json")}


def verify_closed(root, output, snapshot, manifest_sha, closed_sha):
    require(closed_sha is not None, "Out-of-band closed SHA required")
    raw = frozen.file_bytes(output / "closed.json")
    require(origin.sha(raw) == closed_sha, "Closed raw SHA differs")
    closed = json.loads(raw)
    require(closed.get("status") == "closed" and closed.get("manifest_sha256") == manifest_sha and
            closed.get("completed_cells") == 126 and closed.get("prior_cell_agreements") == 8 and
            set(closed.get("files", {})) == required_output_names(), "Closed scope/files differ")
    require({p.name for p in output.iterdir()} == required_output_names() | {"closed.json"}, "Closed output set differs")
    require(frozen.file_bytes(output / "manifest.json") == snapshot, "Closed manifest bytes differ")
    for name, ref in closed["files"].items():
        require(ref["path"] == OUTPUT + "/" + name, "Closed file path differs")
        frozen.checked(root, ref)
        require(stat.S_IMODE((output / name).stat().st_mode) == 0o600, "Private output mode differs")
    return {"status": "verified_closed_unchanged", "completed_cells": 126, "new_model_calls": 0}


def execute(root, manifest, manifest_sha, *, closed_sha=None):
    snapshot = frozen.canonical(manifest)
    require(origin.sha(snapshot) == manifest_sha, "Reviewed CPU manifest SHA differs")
    manifest = json.loads(snapshot)
    evidence, rankings, reports, raw, query_ids, qrels, pure = validate(root, manifest)
    output = frozen.path_at(root, OUTPUT)
    if output.exists():
        return verify_closed(root, output, snapshot, manifest_sha, closed_sha)
    require(closed_sha is None and output.parent.is_dir(), "Fresh output parent required")
    condition = frozen.digest({"protocol": VERSION, "retrieval_protocol": upstream.VERSION,
        "dataset": evidence["dataset_identity"], "ordered_ids": evidence["ordered_id_sha256"],
        "channels": evidence["channels"], "hardware": evidence["hardware"], "source_manifest_sha256": manifest_sha,
        "budget": 100, "rrf_k": 60, "modes": list(MODES)})
    config = configuration(evidence["dataset_identity"], query_ids, condition)
    output.mkdir(mode=0o700)
    frozen._fsync_directory(output.parent)
    frozen.exclusive_write(output / "manifest.json", snapshot)
    origins, ranking_refs = {}, {}
    for channel in CHANNELS:
        if channel in "BGEN":
            report_bytes, ranking_bytes = raw[channel]["report"], raw[channel]["ranking"]
        else:
            report_bytes = frozen.checked(root, evidence["channels"][channel]["report"])
            ranking_bytes = frozen.checked(root, evidence["channels"][channel]["ranking"])
        for suffix, content in (("-origin-report.json", report_bytes), ("-ranking.json", ranking_bytes)):
            frozen.exclusive_write(output / (channel + suffix), content)
        ranking_refs[channel] = frozen.record(root, OUTPUT + "/" + channel + "-ranking.json")
        origins[channel] = {"report": frozen.record(root, OUTPUT + "/" + channel + "-origin-report.json"),
            "channel_identity": evidence["channel_identities"][channel], **origin.HARDWARE[channel]}
        if channel == "B":
            origins[channel]["dtype"] = "not_applicable_bm25"
    frozen.exclusive_write(output / "results.sqlite3", b"")
    db = sqlite3.connect(output / "results.sqlite3")
    try:
        db.execute("PRAGMA journal_mode=DELETE")
        db.execute("PRAGMA synchronous=FULL")
        db.execute("CREATE TABLE results (cell TEXT, query_id TEXT, payload TEXT, PRIMARY KEY(cell,query_id))")
        cells = []
        for cell, rows in derive_rows(rankings, query_ids, qrels, evidence["channel_identities"], config, pure):
            db.executemany("INSERT INTO results VALUES (?,?,?)", [(cell["identity"], r["query_id"],
                frozen.canonical(r).decode()) for r in rows])
            cells.append(cell)
        require(len(cells) == 126 and len({c["identity"] for c in cells}) == 126, "Exact unique None126 required")
        require(db.execute("SELECT COUNT(*) FROM results").fetchone()[0] == 126 * len(query_ids), "Derived row count differs")
        db.commit()
    finally:
        db.close()
    with (output / "results.sqlite3").open("rb") as handle:
        os.fsync(handle.fileno())
    derived = {(c["variant_id"], c["budget_mode"]): c for c in cells}
    linked_agreements = []
    for prior in evidence["prior_agreements"]:
        cell = derived[(prior["variant_id"], prior["budget_mode"])]
        metrics_equal(cell["metrics"], prior["metrics"])
        linked_agreements.append({**prior, "derived_identity": cell["identity"]})
    report = {"schema_version": 1, "engine": "multimodal-v1", "dataset": reports["B"]["dataset"],
        "scope": "frozen_collection", "evaluated_query_count": len(query_ids), "configuration": config,
        "cells": cells, "channels": {c: {"status": "completed", "identity": evidence["channel_identities"][c]}
                                   for c in CHANNELS}, "channel_identities": evidence["channel_identities"],
        "derivation_protocol": VERSION, "retrieval_protocol": upstream.VERSION, "mixed_condition_identity": condition,
        "hardware": evidence["hardware"], "source_manifest_sha256": manifest_sha, "prior_cell_agreements": 8,
        "original_standalone_evidence": linked_agreements, "new_model_calls": 0, "new_embedding_rows": 0,
        "gpu_calls": 0, "retrieval_inference_seconds": None, "rerank_fresh_inference_seconds": None,
        "cache_reads_are_inference": False, "laya_cells_completed": 0, "cpu_runtime": manifest["runtime"]}
    frozen.exclusive_write(output / "report.json", frozen.canonical(report))
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=["variant_id", "budget_mode", *sorted(METRICS)])
    writer.writeheader()
    for cell in cells:
        writer.writerow({"variant_id": cell["variant_id"], "budget_mode": cell["budget_mode"], **cell["metrics"]})
    frozen.exclusive_write(output / "matrix.csv", buffer.getvalue().encode())
    handoff = {"protocol": VERSION, "dataset_id": origin.DATASET, "dataset_identity": evidence["dataset_identity"],
        "mixed_condition_identity": condition, "rankings": ranking_refs, "retrieval_provenance": origins,
        "baseline_report": frozen.record(root, OUTPUT + "/report.json"),
        "accepted_evidence": evidence["metadata_proofs"],
        "upstream_evidence": evidence["upstream_evidence"],
        "input_manifest": evidence["mps_manifest"], "original_archive": evidence["archive"],
        "original_members": {c: evidence["channels"][c] for c in "BGEN"},
        "stage_closures": {c: evidence["channels"][c]["closed"] for c in "SJ"},
        "vector_payloads_read": 0, "new_inference_claimed": False}
    frozen.exclusive_write(output / "six-ranking-inputs.json", frozen.canonical(handoff))
    validate(root, manifest)
    require({p.name for p in output.iterdir()} == required_output_names(), "Unexpected derived output")
    closed = {"status": "closed", "manifest_sha256": manifest_sha, "completed_cells": 126,
              "prior_cell_agreements": 8, "mixed_condition_identity": condition,
              "files": {name: frozen.record(root, OUTPUT + "/" + name) for name in sorted(required_output_names())}}
    frozen.exclusive_write(output / "closed.json", frozen.canonical(closed))
    return {"status": "completed_pending_independent_audit", "completed_cells": 126, "prior_cell_agreements": 8,
            "new_model_calls": 0, "closed_sha256": origin.sha(frozen.canonical(closed))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "execute"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--admission")
    parser.add_argument("--admission-sha256")
    parser.add_argument("--closed-sha256")
    args = parser.parse_args(argv)
    sys.dont_write_bytecode = True
    value = json.loads(frozen.file_bytes(frozen.path_at(args.root, args.input)))
    if args.mode == "prepare":
        require(args.admission and args.admission_sha256 and not args.closed_sha256, "Admission evidence required")
        admission = json.loads(frozen.file_bytes(frozen.path_at(args.root, args.admission)))
        manifest = prepare(args.root, value, args.sha256, admission, args.admission_sha256)
        path = frozen.path_at(args.root, BASE + "/prepared-manifest-v2.json")
        frozen.exclusive_write(path, frozen.canonical(manifest))
        print(json.dumps({"manifest": frozen.record(args.root, path.relative_to(args.root).as_posix()),
                          "canonical_sha256": frozen.digest(manifest), "model_calls": 0}))
    else:
        require(not args.admission and not args.admission_sha256, "Execute accepts admitted manifest only")
        print(json.dumps(execute(args.root, value, args.sha256, closed_sha=args.closed_sha256)))
    return 0

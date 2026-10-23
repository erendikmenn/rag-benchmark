"""First Industrial Laya preparation/context over reviewed None V3 and S/J V3.

Scientific helpers are normal calls to the frozen V1 implementation. This
adapter exposes no unbounded scoring route; bounded V3 is the only scorer.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
import dataclasses
import json
import os
from pathlib import Path
import resource
import stat
import sys
import time
import types

from . import industrial_laya_completion_v1 as previous
from .industrial_laya_completion_v1 import (
    Backend, TokenizerProbe, CHANNELS, CONTEXT, INDUSTRIAL_HARDWARE, INDUSTRIAL_IDENTITY,
    LAYA_CONFIG, MEMORY, MODEL_SNAPSHOT, TARGETS, adapter_identity, baseline_cells,
    canonical, closed_value, fresh, frozen, generation, graph, lock, math, matrix,
    mkdir_private, model_record, models, observations, pairs, private_tree,
    query_binding, require, retrieval, runtime, sdk_sources, sha, snapshot_files,
    union_ids, scatter, utc,
)

# Explicit exports consumed by the bounded adapter; no unbounded execute export.
__all__ = ["Backend", "closed_value", "generation", "lock", "mkdir_private", "private_tree", "retrieval", "scatter"]

VERSION = "industrial-laya-sealed-union-v3"
WORK = "work/industrial-laya-completion-v3"
SOURCE_FILES = (*previous.SOURCE_FILES, "src/rag_benchmark/industrial_laya_completion_v3.py",
                "scripts/run_industrial_laya_completion_v3.py")
INDUSTRIAL_NONE_BASE = "work/industrial-none-derivation-v3/actual-v3"
INDUSTRIAL_MPS_BASE = "work/industrial-l4-bgen-local-mps-sj-v3"
ORIGINAL_MPS_PATH = "work/industrial-l4-bgen-local-mps-sj-v2/prepared-manifest-v2.json"
ORIGINAL_MPS_SHA256 = "f0cd1fa2c5fae10de17615082d27963876d01b5c02c5f7ddd29e92d3614cde3e"
UPSTREAM_REQUIRED = {
    "original_manifest", "operational_envelope", "boot_rebinding", "sj_source_contract",
    "sj_source_review", "sj_root_acceptance", "sj_actual_config_requirements", "j_execution_admission",
    "s_execution_admission", "j_execution_claim", "s_execution_claim", "j_actual_wait", "s_actual_wait",
    "sj_actual_output_review", "sj_root_output_acceptance", "none_source_contract",
    "none_source_review", "none_root_source_acceptance",
}


def sources(root, pins):
    require(set(pins) == set(SOURCE_FILES), "source_population")
    for relative in SOURCE_FILES:
        require(frozen.record(root, relative)["sha256"] == pins[relative], "source_changed")
    previous.sources(root, {path: pins[path] for path in previous.SOURCE_FILES})
    modules = (sys.modules[__name__],)
    for module in modules:
        expected = frozen.path_at(root, "src/" + module.__name__.replace(".", "/") + ".py")
        require(Path(module.__file__).absolute() == expected, "import_origin")
        raw = frozen.file_bytes(expected)
        compiled = compile(raw, str(expected), "exec", dont_inherit=True)
        codes = {}
        tree = ast.parse(raw)
        def collect(code):
            for child in code.co_consts:
                if isinstance(child, types.CodeType):
                    codes[child.co_qualname] = child
                    collect(child)
        collect(compiled)
        def definitions_in(nodes, owner, prefix=""):
            for node in nodes:
                if isinstance(node, ast.ClassDef):
                    cls = vars(owner).get(node.name)
                    require(type(cls) is type and cls.__module__ == module.__name__ and
                            cls.__qualname__ == prefix + node.name, "loaded_class_binding")
                    expected_bases = (RuntimeError,) if node.name == "UnsupportedConfiguration" else (object,)
                    require(cls.__bases__ == expected_bases, "loaded_class_bases")
                    if module is matrix and node.name == "Dataset":
                        expected_init = dataclasses.make_dataclass("Dataset", [
                            "root", "manifest", "corpus", "queries", "qrels", "identity"]).__init__
                        actual_init = vars(cls).get("__init__")
                        require(isinstance(actual_init, types.FunctionType) and
                                actual_init.__code__ == expected_init.__code__ and
                                actual_init.__defaults__ is None and actual_init.__kwdefaults__ is None and
                                actual_init.__closure__ is None, "dataset_constructor_changed")
                    definitions_in(node.body, cls, prefix + node.name + ".")
                elif isinstance(node, ast.FunctionDef):
                    function = vars(owner).get(node.name)
                    if isinstance(function, property):
                        function = function.fget
                    elif isinstance(function, (staticmethod, classmethod)):
                        function = function.__func__
                    require(isinstance(function, types.FunctionType), "loaded_function_binding")
                    if node.decorator_list and all(isinstance(d, ast.Name) and d.id == "contextmanager"
                                                   for d in node.decorator_list):
                        helper = next(c for c in contextmanager.__code__.co_consts
                                      if isinstance(c, types.CodeType) and c.co_name == "helper")
                        require(function.__code__ == helper and
                                function.__globals__ is vars(sys.modules["contextlib"]) and
                                function.__closure__ is not None and len(function.__closure__) == 1 and
                                function.__closure__[0].cell_contents is function.__wrapped__ and
                                function.__defaults__ is None and function.__kwdefaults__ is None,
                                "contextmanager_wrapper_changed")
                        function = function.__wrapped__
                    require(function.__module__ == module.__name__ and
                            function.__qualname__ == prefix + node.name and
                            function.__code__ == codes.get(prefix + node.name) and
                            function.__globals__ is vars(module), "loaded_code_changed")
                    arguments = node.args
                    defaults = tuple(ast.literal_eval(x) for x in arguments.defaults) or None
                    keywords = {k.arg: ast.literal_eval(v) for k, v in zip(arguments.kwonlyargs, arguments.kw_defaults)
                                if v is not None} or None
                    require(function.__defaults__ == defaults and function.__kwdefaults__ == keywords,
                            "loaded_defaults_changed")
        definitions_in(tree.body, module)
        # Validate imported call targets as well as definitions: discovery never
        # trusts a replacement object's __module__ to decide what to inspect.
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.asname or alias.name.split(".")[0]
                    expected_name = alias.name if alias.asname else alias.name.split(".")[0]
                    require(vars(module).get(name) is sys.modules[expected_name], "import_binding")
            elif isinstance(node, ast.ImportFrom) and node.module != "__future__":
                prefix = module.__package__ + "." if node.level else ""
                origin = prefix + (node.module or "")
                for alias in node.names:
                    expected_value = (sys.modules[origin + alias.name] if node.module is None else
                                      getattr(sys.modules[origin], alias.name))
                    require(vars(module).get(alias.asname or alias.name) is expected_value,
                            "imported_call_binding")
    require(matrix.np is models.np and matrix.math is math, "pure_dependency_binding")


def carried_model_metadata(handoff, upstream, accepted_evidence):
    """Bind reviewed descriptor-only provenance without opening its payloads."""
    registry = handoff["model_view_metadata_registry"]
    require(set(registry) == {"root_acceptance", "independent_review", "channels", "descriptors", "payload_reads"}
            and type(registry["payload_reads"]) is int and type(handoff["model_view_metadata_payloads_read"]) is int
            and registry["payload_reads"] == handoff["model_view_metadata_payloads_read"] == 0
            and registry["root_acceptance"] == upstream["sj_root_output_acceptance"]
            and registry["independent_review"] == upstream["sj_actual_output_review"]
            and set(registry["channels"]) == {"J", "S"}, "industrial_model_metadata_registry")
    descriptors = registry["descriptors"]
    require(type(descriptors) is list and len(descriptors) == 43, "industrial_model_metadata_population")
    paths = set()
    for ref in descriptors:
        require(type(ref) is dict and set(ref) == {"path", "bytes", "sha256"}
                and type(ref["path"]) is str and type(ref["bytes"]) is int and ref["bytes"] >= 0
                and type(ref["sha256"]) is str and len(ref["sha256"]) == 64
                and all(c in "0123456789abcdef" for c in ref["sha256"]), "industrial_model_metadata_ref")
        path = Path(ref["path"])
        require(not path.is_absolute() and path.as_posix() == ref["path"] and ".." not in path.parts
                and ref["path"] not in paths, "industrial_model_metadata_path")
        paths.add(ref["path"])
    controls = [registry["root_acceptance"], registry["independent_review"]]
    assigned = set()
    for channel, expected in (("J", 12), ("S", 31)):
        item = registry["channels"][channel]
        require(set(item) == {"pid", "full_nested_closure", "count"} and type(item["pid"]) is int
                and item["pid"] > 0 and type(item["count"]) is int and item["count"] == expected,
                "industrial_model_metadata_channel")
        prefix = INDUSTRIAL_MPS_BASE + "/actual-v3/" + channel + "/model-view-" + str(item["pid"]) + "/"
        matches = {p for p in paths if p.startswith(prefix)}
        require(len(matches) == expected, "industrial_model_metadata_namespace")
        assigned |= matches
        controls.append(item["full_nested_closure"])
    require(assigned == paths and all(ref in handoff["accepted_evidence"] and ref in accepted_evidence
                                      for ref in controls), "industrial_model_metadata_control_freshness")
    raw_paths = {r["path"] for r in [*handoff["accepted_evidence"], *accepted_evidence]}
    require(not paths & raw_paths, "industrial_model_metadata_must_not_be_payload_reads")
    return controls


def industrial_inputs(root, manifest):
    """Admit reviewed metadata and copied rankings; no vector or archive replay."""
    evidence = manifest["industrial_evidence"]
    require(set(evidence) == {"handoff", "none_closed", "actual_review", "root_acceptance"},
            "industrial_evidence_fields")
    require(manifest["dataset_identity"] == INDUSTRIAL_IDENTITY and
            evidence["handoff"]["path"] == INDUSTRIAL_NONE_BASE + "/six-ranking-inputs.json" and
            evidence["none_closed"]["path"] == INDUSTRIAL_NONE_BASE + "/closed.json",
            "industrial_dataset_or_paths")
    refs = [evidence[k] for k in ("handoff", "none_closed", "actual_review", "root_acceptance")]
    require(all(ref in manifest["accepted_evidence"] for ref in refs), "industrial_freshness_evidence")
    handoff, closed, review, accepted = (frozen.read_json(root, ref) for ref in refs)
    none_manifest = closed["files"]["manifest.json"]
    require(none_manifest["path"] == INDUSTRIAL_NONE_BASE + "/manifest.json" and
            none_manifest["sha256"] == closed["manifest_sha256"], "industrial_none_manifest")
    closure_refs = list(closed["files"].values())
    require(all(ref in manifest["accepted_evidence"] for ref in closure_refs),
            "industrial_none_closed_freshness")
    for proof, status in ((review, "PASS_INDEPENDENT_INDUSTRIAL_NONE_V3_ACTUAL_OUTPUT"),
                          (accepted, "PASS_ROOT_INDUSTRIAL_NONE_V3_ACTUAL_OUTPUT_ACCEPTED")):
        require(proof["status"] == status and proof["blocking_findings"] == [] and
                proof["handoff_sha256"] == evidence["handoff"]["sha256"] and
                proof["none_closed_sha256"] == evidence["none_closed"]["sha256"] and
                proof["baseline_report_sha256"] == manifest["baseline_report"]["sha256"] and
                proof["none_manifest_sha256"] == none_manifest["sha256"] and
                type(proof["completed_cells"]) is int and proof["completed_cells"] == 126 and
                type(proof["prior_cell_agreements"]) is int and proof["prior_cell_agreements"] == 8 and
                type(proof["metrics_per_cell"]) is int and proof["metrics_per_cell"] == 17,
                "industrial_none_actual_review")
        require(isinstance(proof["accepted_evidence"], list) and proof["accepted_evidence"] and
                proof["actual_wait"] in proof["accepted_evidence"] and
                all(ref in manifest["accepted_evidence"] for ref in proof["accepted_evidence"]),
                "industrial_none_review_proof_freshness")
        closure_refs += proof["accepted_evidence"]
    require(accepted["independent_review_sha256"] == evidence["actual_review"]["sha256"],
            "industrial_none_root_review_binding")
    require(accepted["actual_wait"] == review["actual_wait"], "industrial_none_review_wait_binding")
    wait = frozen.read_json(root, review["actual_wait"])
    require(wait["actual_wait"] is True and type(wait["exit_code"]) is int and wait["exit_code"] == 0 and
            wait["signal"] is None and wait["manifest_sha256"] == none_manifest["sha256"] and
            wait["closed"] == evidence["none_closed"], "industrial_none_actual_wait")
    require(handoff["protocol"] == "industrial-none-derivation-v3" and
            handoff["dataset_id"] == "vidore-v3-industrial-en" and
            handoff["dataset_identity"] == INDUSTRIAL_IDENTITY and
            handoff["rankings"] == manifest["rankings"] and
            handoff["retrieval_provenance"] == manifest["retrieval_provenance"] and
            handoff["baseline_report"] == manifest["baseline_report"], "industrial_handoff_binding")
    require(isinstance(handoff["accepted_evidence"], list) and handoff["accepted_evidence"] and
            all(ref in manifest["accepted_evidence"] for ref in handoff["accepted_evidence"]),
            "industrial_inherited_review_freshness")
    upstream = handoff["upstream_evidence"]
    require(type(upstream) is dict and UPSTREAM_REQUIRED <= set(upstream), "industrial_upstream_population")
    registry_controls = carried_model_metadata(handoff, upstream, manifest["accepted_evidence"])
    extra = [handoff["input_manifest"], *handoff["stage_closures"].values(), *upstream.values(), *registry_controls]
    require(all(ref in manifest["accepted_evidence"] for ref in extra), "industrial_upstream_freshness")
    # Metadata/source/empty proof files remain raw refs; only semantic schemas below are JSON.
    for ref in [*refs, *closure_refs, *handoff["accepted_evidence"], *extra]:
        frozen.checked(root, ref)
    require(closed["status"] == "closed" and closed["completed_cells"] == 126 and
            closed["prior_cell_agreements"] == 8 and
            closed["mixed_condition_identity"] == handoff["mixed_condition_identity"] and
            closed["files"]["six-ranking-inputs.json"] == evidence["handoff"] and
            closed["files"]["report.json"] == manifest["baseline_report"], "industrial_none_closed")
    baseline = frozen.read_json(root, manifest["baseline_report"])
    require(baseline["derivation_protocol"] == "industrial-none-derivation-v3" and
            baseline["retrieval_protocol"] == "industrial-l4-bgen-local-mps-sj-v3" and
            baseline["mixed_condition_identity"] == handoff["mixed_condition_identity"] and
            baseline["source_manifest_sha256"] == closed["manifest_sha256"] and
            baseline["prior_cell_agreements"] == 8 and
            len(baseline["original_standalone_evidence"]) == 8 and
            baseline["new_model_calls"] == 0 and baseline["new_embedding_rows"] == 0 and
            baseline["gpu_calls"] == 0 and baseline["laya_cells_completed"] == 0,
            "industrial_none_provenance")
    for channel in CHANNELS:
        origin = manifest["retrieval_provenance"][channel]
        require(all(origin[k] == v for k, v in INDUSTRIAL_HARDWARE[channel].items()) and
                closed["files"][channel + "-ranking.json"] == manifest["rankings"][channel] and
                closed["files"][channel + "-origin-report.json"] == origin["report"],
                "industrial_original_channel")
    require(set(handoff["stage_closures"]) == {"S", "J"} and
            handoff["input_manifest"]["path"] == INDUSTRIAL_MPS_BASE + "/prepared-manifest-v3.json" and
            handoff["input_manifest"] == upstream["operational_envelope"], "industrial_stage_scope")
    mps = frozen.read_json(root, handoff["input_manifest"])
    require(mps["protocol"] == "industrial-l4-bgen-local-mps-sj-v3" and
            mps["dataset_id"] == "vidore-v3-industrial-en" and
            mps["original_manifest"] == upstream["original_manifest"] and
            mps["rebinding"] == upstream["boot_rebinding"] and
            upstream["original_manifest"]["path"] == ORIGINAL_MPS_PATH and
            upstream["original_manifest"]["sha256"] == ORIGINAL_MPS_SHA256,
            "industrial_mps_envelope")
    operational = mps["operational_manifest"]
    require(operational["dataset_identity"] == INDUSTRIAL_IDENTITY and
            all(operational[key] == manifest[key] for key in ("tables", "media", "ordered_id_sha256")),
            "industrial_current_parent_dataset_binding")
    mps_sha = sha(canonical(mps))
    require(mps_sha == handoff["input_manifest"]["sha256"], "industrial_mps_canonical_identity")
    for channel in "SJ":
        prefix = channel.lower()
        ref = handoff["stage_closures"][channel]
        require(ref["path"] == INDUSTRIAL_MPS_BASE + "/actual-v3/" + channel + "/closed.json",
                "industrial_stage_path")
        stage = frozen.read_json(root, ref)
        claim = frozen.read_json(root, upstream[prefix + "_execution_claim"])
        wait = frozen.read_json(root, upstream[prefix + "_actual_wait"])
        require(claim["pid"] == handoff["model_view_metadata_registry"]["channels"][channel]["pid"],
                "industrial_model_metadata_claim_pid")
        require(stage["status"] == "completed" and stage["protocol"] == "industrial-l4-bgen-local-mps-sj-v3" and
                stage["manifest_sha256"] == mps_sha and stage["original_manifest"] == upstream["original_manifest"] and
                stage["rebinding"] == upstream["boot_rebinding"] and
                stage["identity"] == manifest["retrieval_provenance"][channel]["channel_identity"] and
                set(stage["files"]) == {"rankings.json", "report.json", "scoring.claim.json", "execution.claim.json"} and
                stage["execution_admission"] == upstream[prefix + "_execution_admission"] and
                stage["files"]["execution.claim.json"] == upstream[prefix + "_execution_claim"],
                "industrial_stage_closed")
        require(claim["protocol"] == "industrial-l4-bgen-local-mps-sj-v3" and claim["channel"] == channel and
                claim["manifest_sha256"] == mps_sha and
                claim["execution_admission"] == upstream[prefix + "_execution_admission"] and
                wait["actual_wait"] is True and type(wait["exit_code"]) is int and wait["exit_code"] == 0 and
                wait["signal"] is None and wait["channel"] == channel and wait["manifest_sha256"] == mps_sha and
                wait["closed"] == ref and wait["execution_claim"] == upstream[prefix + "_execution_claim"] and
                type(claim["pid"]) is int and claim["pid"] > 0 and
                type(wait["pid"]) is int and wait["pid"] == claim["pid"],
                "industrial_stage_true_wait")
        for original, copied in ((stage["files"]["rankings.json"], manifest["rankings"][channel]),
                                 (stage["files"]["report.json"], manifest["retrieval_provenance"][channel]["report"])):
            require(all(original[k] == copied[k] for k in ("bytes", "sha256")), "industrial_stage_copy")


def prepare_manifest(root, recipe):
    """Seal a fully specified recipe; no rankings, model calls or old writes."""
    required = {"protocol", "dataset_id", "source_sha256", "dataset_identity", "tables", "media",
                "ordered_id_sha256", "rankings", "retrieval_provenance", "baseline_report",
                "accepted_evidence", "model_snapshot", "model_files", "laya_adapter_identity", "gpu_lock",
                "sdk_source_sha256", "industrial_evidence"}
    require(set(recipe) == required and recipe["protocol"] == VERSION, "recipe_fields")
    manifest = {**recipe, "root": str(root), "runtime": runtime(), "memory_guard": MEMORY,
                "laya_config": LAYA_CONFIG, "sdk_language_argument": "tr", "context_protocol": CONTEXT,
                "matrix": {"channels": list(CHANNELS), "budget_modes": ["per_channel", "total"],
                           "candidate_k": [20, 50, 100], "budget": 100, "rrf_k": 60},
                "output": "work/industrial-laya-completion-v3/runs/" + recipe["dataset_id"],
                "recipe_sha256": sha(canonical(recipe) + b"\n")}
    validate(root, manifest)
    return manifest


def validate(root, manifest, *, verify_baseline=True):
    require(manifest.get("protocol") == VERSION and manifest.get("root") == str(root), "manifest_scope")
    name = manifest["dataset_id"]
    require(name in TARGETS and manifest["memory_guard"] == MEMORY, "target_or_memory")
    require(manifest["output"] == "work/industrial-laya-completion-v3/runs/" + name, "output_scope")
    industrial_inputs(root, manifest)
    require(manifest["runtime"] == runtime() and manifest["laya_config"] == LAYA_CONFIG, "runtime_or_model")
    require(manifest["context_protocol"] == CONTEXT, "context_protocol")
    require(manifest["laya_adapter_identity"] == adapter_identity() and
            manifest["sdk_language_argument"] == "tr", "adapter_identity_or_language")
    gpu = manifest["gpu_lock"]
    require(set(gpu) == {"path", "device", "inode"} and gpu["path"] == "work/overnight/gpu.lock"
            and type(gpu["device"]) is int and type(gpu["inode"]) is int, "gpu_lock_config")
    lock_stat = frozen.path_at(root, gpu["path"]).stat(follow_symlinks=False)
    require(stat.S_ISREG(lock_stat.st_mode) and (lock_stat.st_dev, lock_stat.st_ino) ==
            (gpu["device"], gpu["inode"]), "gpu_lock_changed")
    require(manifest["matrix"] == {"channels": list(CHANNELS), "budget_modes": ["per_channel", "total"],
            "candidate_k": [20, 50, 100], "budget": 100, "rrf_k": 60}, "matrix_scope")
    sources(root, manifest["source_sha256"])
    sdk_sources(root, manifest["sdk_source_sha256"])
    metadata, corpus, queries, qrels, exclusions = frozen._dataset(root, manifest)
    require((len(corpus), len(queries)) == TARGETS[name], "population")
    dataset = matrix.Dataset(frozen.path_at(root, "data/multimodal/" + name), metadata,
                             corpus, queries, qrels, manifest["dataset_identity"])
    dataset.require_text()
    require(all(not q.get("media") and isinstance(q.get("text"), str) and q["text"].strip()
                for q in queries), "written_queries")
    require(set(manifest["rankings"]) == set(CHANNELS)
            and set(manifest["retrieval_provenance"]) == set(CHANNELS), "six_sealed_channels")
    rankings = {}
    for channel in CHANNELS:
        ref = manifest["rankings"][channel]
        origin = manifest["retrieval_provenance"][channel]
        require(set(origin) == {"report", "channel_identity", "hardware", "dtype", "stage"}
                and all(isinstance(origin[k], str) and origin[k] for k in ("hardware", "dtype", "stage")),
                "retrieval_origin")
        report = frozen.read_json(root, origin["report"])
        require(report["dataset"]["identity"] == dataset.identity, "origin_dataset")
        identity = report.get("channel_identities", {}).get(channel)
        if identity is None:
            row = report["channels"][channel]
            require(row["status"] == "completed", "origin_channel_incomplete")
            identity = row["identity"]
        require(identity == origin["channel_identity"], "origin_identity")
        payload = frozen.read_json(root, ref)
        require(isinstance(payload.get("rankings"), list) and all(isinstance(rows, list) and all(
            isinstance(row, dict) and set(row) == {"id", "score", "rank"} and
            isinstance(row["id"], str) and type(row["rank"]) is int and
            type(row["score"]) in (int, float) for row in rows) for rows in payload["rankings"]),
            "ranking_scalar_types")
        rankings[channel] = frozen._rankings(payload, identity,
            [r["id"] for r in corpus], [r["id"] for r in queries], exclusions, bm25=channel == "B")
    require(bool(manifest["accepted_evidence"]), "accepted_evidence_missing")
    for ref in manifest["accepted_evidence"]:
        frozen.checked(root, ref)
    require(set(manifest["model_files"]) >= {"model.safetensors", "rl_agent_config.json",
            "encoder/config.json", "tokenizer/tokenizer.json"}, "model_snapshot_files")
    require(manifest["model_snapshot"] == MODEL_SNAPSHOT and
            set(manifest["model_files"]) == snapshot_files(root), "model_snapshot_population")
    for relative, ref in manifest["model_files"].items():
        require(ref == model_record(root, relative), "model_snapshot_binding")
    baseline = frozen.read_json(root, manifest["baseline_report"])
    require(baseline["dataset"]["identity"] == dataset.identity, "baseline_dataset")
    config = baseline["configuration"]
    require(config["query_ids"] == [q["id"] for q in queries] and
            config["requested_channels"] == list(CHANNELS) and config["rrf_k"] == 60 and
            config["channel_budget"] == 100 and config["budget_modes"] == ["per_channel", "total"] and
            config["candidate_grid"] == [20, 50, 100], "baseline_configuration")
    expected = baseline_cells(baseline, len(queries))
    if verify_baseline:
        metrics = {p["key"]: [] for p in pairs()}
        for i, query in enumerate(queries):
            for pair in graph(rankings, i):
                metrics[pair["key"]].append(matrix.graded_metrics(
                    [r["id"] for r in pair["fused"]], qrels[query["id"]]))
        for p in pairs():
            actual = matrix.aggregate_metrics(metrics[p["key"]])
            key = "document__" + "_".join(p["channels"]).lower() + "__none|" + p["budget_mode"]
            require(set(actual) == set(expected[key]) and all(
                abs(actual[k] - expected[key][k]) <= 1e-12 for k in actual), "baseline_disagreement")
    return dataset, rankings


def preflight(root, manifest, manifest_sha):
    require(sha(canonical(manifest) + b"\n") == manifest_sha, "manifest_bytes_convention")
    dataset, rankings = validate(root, manifest, verify_baseline=False)
    before = observations(root, manifest)
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[name] = "1"
    output = frozen.path_at(root, "work/industrial-laya-completion-v3/preflight/" + manifest["dataset_id"] + ".json")
    require(output.parent.is_dir() and not output.exists(), "preflight_destination")
    started = time.perf_counter()
    probe = TokenizerProbe(root, manifest)
    corpus = {r["id"]: dataset.model_item(r) for r in dataset.corpus}
    records = []
    for index, query in enumerate(dataset.queries):
        candidate_graph = graph(rankings, index)
        ids = union_ids(candidate_graph)
        require(len(ids) <= CONTEXT["maximum_pairs_per_query"], "context_pair_bound")
        tokens = []
        for identifier in ids:
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
            require(time.perf_counter() - started <= CONTEXT["maximum_wall_seconds"] and
                    rss <= CONTEXT["maximum_process_rss_bytes"], "context_resource_bound")
            tokens.append(probe.check(dataset.model_item(query, text_only=True), corpus[identifier]))
        records.append({"binding": query_binding(manifest_sha, index, dataset, candidate_graph, corpus),
                        "pairs": len(ids), "max_tokens": max(tokens, default=0)})
        sources(root, manifest["source_sha256"])
        sdk_sources(root, manifest["sdk_source_sha256"])
        require(observations(root, manifest) == before, "preflight_input_freshness")
    validate(root, manifest, verify_baseline=False)
    require(observations(root, manifest) == before, "preflight_post_input_freshness")
    evidence = {"protocol": VERSION, "status": "PASS_TOKENIZER_ONLY_CONTEXT_PREFLIGHT",
                "manifest_sha256": manifest_sha, "queries": records, "max_len": 8192,
                "head_max_len": probe.cfg.get("head_max_len", 192), "truncated_pairs": 0,
                "model_instances": 0, "model_forwards": 0, "gpu_calls": 0,
                "pairs": sum(r["pairs"] for r in records), "maximum_tokens": max(r["max_tokens"] for r in records),
                "completed_at_utc": utc()}
    digest = fresh(output, evidence)
    return {"status": evidence["status"], "queries": len(records), "pairs": evidence["pairs"],
            "maximum_tokens": evidence["maximum_tokens"], "preflight_sha256": digest}


def context_evidence(root, manifest, manifest_sha, ref, dataset, rankings):
    require(ref is not None and ref["path"] == "work/industrial-laya-completion-v3/preflight/" +
            manifest["dataset_id"] + ".json", "context_preflight_required")
    evidence = frozen.read_json(root, ref)
    require(evidence["protocol"] == VERSION and evidence["status"] == "PASS_TOKENIZER_ONLY_CONTEXT_PREFLIGHT"
            and evidence["manifest_sha256"] == manifest_sha and evidence["max_len"] == 8192
            and evidence["truncated_pairs"] == 0 and evidence["model_instances"] == 0
            and evidence["model_forwards"] == 0 and evidence["gpu_calls"] == 0
            and len(evidence["queries"]) == len(dataset.queries), "context_preflight_scope")
    corpus = {r["id"]: dataset.model_item(r) for r in dataset.corpus}
    for index, row in enumerate(evidence["queries"]):
        candidate_graph = graph(rankings, index)
        require(row["binding"] == query_binding(manifest_sha, index, dataset, candidate_graph, corpus) and
                row["pairs"] == len(union_ids(candidate_graph)) and type(row["max_tokens"]) is int and
                0 < row["max_tokens"] <= 8192, "context_preflight_binding")
    require(evidence["pairs"] == sum(r["pairs"] for r in evidence["queries"]) and
            evidence["maximum_tokens"] == max(r["max_tokens"] for r in evidence["queries"]),
            "context_preflight_counts")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", help="SHA256 of complete canonical file bytes")
    parser.add_argument("--prepare-from-recipe", type=Path)
    parser.add_argument("--recipe-sha256")
    parser.add_argument("--context-preflight", action="store_true")
    args = parser.parse_args(argv)
    os.umask(0o077)
    if args.prepare_from_recipe:
        require(args.recipe_sha256 and not args.manifest_sha256 and not args.context_preflight,
                "prepare_arguments")
        raw = frozen.file_bytes(args.prepare_from_recipe)
        require(sha(raw) == args.recipe_sha256, "recipe_sha")
        recipe = json.loads(raw)
        require(raw == canonical(recipe) + b"\n", "canonical_recipe")
        manifest = prepare_manifest(args.root, recipe)
        expected = frozen.path_at(args.root, WORK + "/manifests/" + recipe["dataset_id"] + ".json")
        require(args.manifest == expected and expected.parent.is_dir(), "manifest_destination")
        digest = fresh(expected, manifest)
        print(json.dumps({"status": "prepared_not_executed", "manifest_sha256": digest}))
        return 0
    require(args.manifest_sha256 and not args.recipe_sha256, "manifest_arguments")
    raw = frozen.file_bytes(args.manifest)
    require(sha(raw) == args.manifest_sha256, "manifest_sha")
    manifest = json.loads(raw)
    require(raw == canonical(manifest) + b"\n", "canonical_manifest")
    if args.context_preflight:
        result = preflight(args.root, manifest, args.manifest_sha256)
    else:
        dataset, _ = validate(args.root, manifest)
        result = {"status": "validated_not_executed", "queries": len(dataset.queries), "laya_cells": 378}
    print(json.dumps(result))
    return 0

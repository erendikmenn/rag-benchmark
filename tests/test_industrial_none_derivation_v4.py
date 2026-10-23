"""New original-adapter descriptor boundary only; no dataset or model execution."""
import ast
import copy
import hashlib
import json
from pathlib import Path

import pytest

from rag_benchmark import industrial_none_derivation_v3 as old
from rag_benchmark import industrial_none_derivation_v4 as run


def ref(path, digest=None):
    return {"path": path, "bytes": 1, "sha256": digest or hashlib.sha256(path.encode()).hexdigest()}


class MemoryControls(run.Proofs):
    """Mock only the trusted, byte-checked control layer; asset paths never open."""
    def __init__(self, values):
        super().__init__(None)
        self.values = values
        self.calls = []

    def raw(self, r):
        run.cpu_readable(r)
        self.calls.append(r)
        self.refs[r["path"]] = r
        return json.dumps(self.values[r["path"]]).encode()


def controls():
    desc = copy.deepcopy(run.ORIGINAL_ADAPTER_DESCRIPTOR)
    original = ref(run.upstream.PARENT_PATH, run.upstream.PARENT_SHA256)
    envelope = ref(run.MPS_MANIFEST, run.ADAPTER_ENVELOPE_SHA)
    combined = ref("proofs/combined.json", run.ADAPTER_COMBINED_SHA)
    accepted = ref("proofs/accepted.json", run.ADAPTER_ROOT_SHA)
    s_review = ref("proofs/S-review.json", run.ADAPTER_S_REVIEW_SHA)
    evidence = ref("proofs/S-evidence.json", run.ADAPTER_S_EVIDENCE_SHA)
    source = ref("work/industrial-S-independent-v1/audit_actual_S_v2.py", run.ADAPTER_AUDIT_SOURCE_SHA)
    claim, wait, parent, closure, closed, execution = [ref("proofs/" + n + ".json") for n in
        ("claim", "wait", "parent", "closure", "closed", "execution")]
    refs = {"original_manifest": original, "operational_envelope": envelope,
        "sj_actual_output_review": combined, "sj_root_output_acceptance": accepted,
        "s_execution_claim": claim, "s_actual_wait": wait, "s_execution_admission": execution}
    admission = {"mps_manifest_sha256": run.ADAPTER_ENVELOPE_SHA,
        "closed_sha256": {"J": "1" * 64, "S": closed["sha256"]}}
    asset = {"revision": run.ADAPTER_REVISION, "assets": {"adapter_config.json": {
        "file": desc, "logical_path": ".cache/models/models--vidore--colqwen2.5-v0.2/snapshots/" +
        run.ADAPTER_REVISION + "/adapter_config.json", "resolved_path": desc["path"]}}}
    shared = {"manifest_sha256": run.ADAPTER_ENVELOPE_SHA,
        "original_manifest_sha256": run.upstream.PARENT_SHA256,
        "vector_payload_validation_completed": True, "metadata_only_cpu_reuse_allowed": True,
        "closed_sha256": admission["closed_sha256"], "full_nested_closures": {"S": closure},
        "actual_child_waits": {"S": wait}, "actual_parent_tool_waits": {"S": parent},
        "accepted_evidence": [desc, s_review, evidence, source, closure, claim, wait, parent]}
    values = {
        original["path"]: {"assets": {"S": copy.deepcopy(asset)}},
        envelope["path"]: {"protocol": run.upstream.VERSION, "original_manifest": original,
            "operational_manifest": {"assets": {"S": copy.deepcopy(asset)}}},
        combined["path"]: {**copy.deepcopy(shared), "status": "PASS_INDEPENDENT_INDUSTRIAL_SJ_ACTUAL_OUTPUTS",
            "blocking_findings": [], "channel_reviews": {"S": s_review}},
        accepted["path"]: {**copy.deepcopy(shared), "status": "PASS_ROOT_INDUSTRIAL_SJ_REUSE_FOR_NONE",
            "independent_review": combined, "independent_review_sha256": combined["sha256"],
            "complete_upstream_actual_config_reviewed": True},
        s_review["path"]: {"status": "PASS_INDEPENDENT_INDUSTRIAL_S_ACTUAL_OUTPUT", "blocking_findings": [],
            "channel": "S", "manifest_sha256": run.ADAPTER_ENVELOPE_SHA,
            "original_manifest_sha256": run.upstream.PARENT_SHA256,
            "vector_payload_validation_completed": True, "metadata_only_cpu_reuse_allowed": True,
            "evidence": evidence, "full_nested_closure": closure, "execution_claim": claim,
            "actual_wait": wait, "actual_parent_tool_wait": parent, "closed": closed,
            "execution_admission": execution, "closed_sha256": {"S": closed["sha256"]},
            "wait_sha256": {"S": wait["sha256"]}},
        evidence["path"]: {"audit_source": source, "full_nested_closure": closure,
            "accepted_evidence": [desc], "all_new_vectors_and_claims_checked": True,
            "all_283_rankings_independently_recomputed": True, "source_input_control_and_S_output_bytes_fresh": True},
        source["path"]: "synthetic trusted control source",
        claim["path"]: {"protocol": run.upstream.VERSION, "channel": "S", "manifest_sha256": run.ADAPTER_ENVELOPE_SHA,
            "pid": 321},
        wait["path"]: {"actual_wait": True, "exit_code": 0, "signal": None, "pid": 321,
            "execution_claim": claim, "closed": closed, "channel": "S", "manifest_sha256": run.ADAPTER_ENVELOPE_SHA},
        closure["path"]: {"status": "PASS_S_EXACT_CLOSED_NAMESPACE_AND_NESTED_REFS", "pre_post_identical": True,
            "execution_claim": claim, "actual_wait": wait, "closed": closed, "actual_parent_tool_wait": parent},
        parent["path"]: {"protocol": "industrial-S-actual-parent-tool-wait-v1", "status": "ACTUAL_PARENT_TOOL_WAIT_EXIT_0",
            "actual_tool_wait_observed": True, "exit_code": 0, "tool_result": {"exit_code": 0}, "child_pid": 321}}
    return MemoryControls(values), refs, admission


def test_exact_original_descriptor_registered_without_any_payload_io_and_separate_counts():
    ledger, refs, admission = controls()
    before = {}
    for i in range(43):
        r = ref(f"work/example/model-view-321/{i}/config.json")
        before[r["path"]] = r
    for i in range(11054):
        r = ref(f"work/example/vectors/{i}.npy")
        before[r["path"]] = r
    lock = {"path": "work/overnight/gpu.lock", "bytes": 0, "sha256": hashlib.sha256(b"").hexdigest()}
    before[lock["path"]] = lock
    ledger.descriptors = copy.deepcopy(before)
    registry = run.register_original_adapter_metadata(ledger, refs, admission)
    assert registry["descriptor"] == run.ORIGINAL_ADAPTER_DESCRIPTOR
    assert registry["asset_channel"] == "S" and registry["asset_name"] == "adapter_config.json"
    assert type(registry["count"]) is int and registry["count"] == 1
    assert type(registry["payload_reads"]) is int and registry["payload_reads"] == 0
    assert registry["live_payload_freshness_attested"] is False
    assert len(ledger.descriptors) == 11099
    assert all(ledger.descriptors[p] == r for p, r in before.items())
    assert all(not r["path"].startswith(".cache/") for r in ledger.calls)
    ledger.embedded(registry)
    assert run.ORIGINAL_ADAPTER_DESCRIPTOR["path"] not in ledger.refs


@pytest.mark.parametrize("damage", ["asset_key", "asset_name", "path", "sha", "bytes", "revision", "logical",
    "envelope", "root_status", "combined_status", "independent_pin", "S_review_pin", "S_evidence_pin",
    "audit_source_pin", "root_missing", "S_missing", "conflicting", "pid", "false_exit", "parent_pid",
    "parent_false_exit", "closure", "control_missing", "S_source_fresh"])
def test_new_registration_rejects_any_unbound_original_metadata_before_registration(damage):
    ledger, refs, admission = controls()
    original = ledger.values[refs["original_manifest"]["path"]]
    envelope = ledger.values[refs["operational_envelope"]["path"]]
    accepted = ledger.values[refs["sj_root_output_acceptance"]["path"]]
    combined = ledger.values[refs["sj_actual_output_review"]["path"]]
    sr = ledger.values[combined["channel_reviews"]["S"]["path"]]
    ev = ledger.values[sr["evidence"]["path"]]
    asset = original["assets"]["S"]
    entry = asset["assets"]["adapter_config.json"]
    if damage == "asset_key":
        original["assets"]["J"] = original["assets"].pop("S")
    elif damage == "asset_name":
        asset["assets"]["config.json"] = asset["assets"].pop("adapter_config.json")
    elif damage in ("path", "sha", "bytes"):
        field = "sha256" if damage == "sha" else damage
        entry["file"][field] = {"path": ".cache/models/unaccepted/blob", "sha256": "0" * 64, "bytes": 814}[field]
    elif damage == "revision":
        asset["revision"] = "wrong"
    elif damage == "logical":
        entry["logical_path"] = "wrong/adapter_config.json"
    elif damage == "envelope":
        envelope["operational_manifest"]["assets"]["S"]["assets"]["adapter_config.json"]["file"]["bytes"] += 1
    elif damage == "root_status":
        accepted["status"] = "SOURCE_ONLY"
    elif damage == "combined_status":
        combined["blocking_findings"] = ["blocked"]
    elif damage == "independent_pin":
        refs["sj_actual_output_review"]["sha256"] = "0" * 64
    elif damage == "S_review_pin":
        combined["channel_reviews"]["S"]["sha256"] = "0" * 64
    elif damage == "S_evidence_pin":
        sr["evidence"]["sha256"] = "0" * 64
    elif damage == "audit_source_pin":
        ev["audit_source"]["sha256"] = "0" * 64
    elif damage == "root_missing":
        accepted["accepted_evidence"].pop(0)
    elif damage == "S_missing":
        ev["accepted_evidence"] = [ref("proofs/unrelated.json")]
    elif damage == "conflicting":
        accepted["accepted_evidence"].append({**run.ORIGINAL_ADAPTER_DESCRIPTOR, "bytes": 814})
    elif damage == "pid":
        ledger.values[refs["s_execution_claim"]["path"]]["pid"] = 999
    elif damage == "false_exit":
        ledger.values[refs["s_actual_wait"]["path"]]["exit_code"] = False
    elif damage == "parent_pid":
        ledger.values[sr["actual_parent_tool_wait"]["path"]]["child_pid"] = 999
    elif damage == "parent_false_exit":
        ledger.values[sr["actual_parent_tool_wait"]["path"]]["tool_result"]["exit_code"] = False
    elif damage == "closure":
        ledger.values[sr["full_nested_closure"]["path"]]["pre_post_identical"] = False
    elif damage == "control_missing":
        accepted["accepted_evidence"] = [r for r in accepted["accepted_evidence"] if r != ev["audit_source"]]
    elif damage == "S_source_fresh":
        ev["source_input_control_and_S_output_bytes_fresh"] = False
    with pytest.raises((ValueError, KeyError)):
        run.register_original_adapter_metadata(ledger, refs, admission)
    assert ledger.descriptors == {}
    assert all(not r["path"].startswith(".cache/") for r in ledger.calls)


def test_registered_blob_is_descriptor_only_and_arbitrary_cache_stays_forbidden(tmp_path):
    ledger = run.Proofs(tmp_path)
    d = run.ORIGINAL_ADAPTER_DESCRIPTOR
    with pytest.raises(ValueError, match="cpu_metadata_boundary"):
        old.Proofs(tmp_path).embedded(d)
    ledger.descriptor(d)
    ledger.embedded(d)
    for action in (ledger.raw, ledger.read):
        with pytest.raises(ValueError, match="cpu_metadata_boundary"):
            action(d)
    with pytest.raises(ValueError, match="payload_descriptor_changed"):
        ledger.embedded({**d, "bytes": 814})
    with pytest.raises(ValueError, match="cpu_metadata_boundary"):
        ledger.embedded({**d, "path": d["path"] + "another"})
    assert ledger.refs == {}


@pytest.mark.parametrize("suffix", ["actual-v3", "prepared-manifest-v3.json"])
def test_old_v3_actual_namespace_must_be_absent(tmp_path, suffix):
    base = tmp_path / "work/industrial-none-derivation-v3"
    base.mkdir(parents=True)
    run.prior_cpu_namespace_absent(tmp_path)
    (base / suffix).touch()
    with pytest.raises(ValueError, match="prior_v3_cpu_namespace_not_absent"):
        run.prior_cpu_namespace_absent(tmp_path)


def test_existing_science_model_view_and_raw_boundaries_unchanged():
    before, after = [ast.parse(Path(x.__file__).read_text()) for x in (old, run)]
    def functions(tree):
        return {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    a, b = functions(before), functions(after)
    unchanged = set(a) - {"Proofs", "sources", "source_admissions", "admitted_inputs", "inspect_inputs", "execute", "main"}
    for name in unchanged:
        assert ast.dump(a[name], include_attributes=False) == ast.dump(b[name], include_attributes=False), name
    am, bm = [{n.name: n for n in nodes["Proofs"].body if isinstance(n, ast.FunctionDef)} for nodes in (a, b)]
    for name in ("raw", "read", "embedded", "fresh", "descriptor"):
        assert ast.dump(am[name], include_attributes=False) == ast.dump(bm[name], include_attributes=False)
    for name in ("configuration", "derive_rows", "strict_ranking", "metrics_equal", "prior_agreement", "archive_bundle"):
        assert getattr(run, name) is getattr(old, name)
    class StripNewMetadata(ast.NodeTransformer):
        def visit_Dict(self, node):
            pairs = [(k, v) for k, v in zip(node.keys, node.values) if not
                (isinstance(k, ast.Constant) and str(k.value).startswith("original_adapter_metadata"))]
            node.keys, node.values = [p[0] for p in pairs], [p[1] for p in pairs]
            return self.generic_visit(node)
        def visit_Expr(self, node):
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == "prior_cpu_namespace_absent":
                return None
            return self.generic_visit(node)
    for name in ("execute", "inspect_inputs"):
        normalized = StripNewMetadata().visit(copy.deepcopy(b[name]))
        assert ast.dump(a[name], include_attributes=False) == ast.dump(normalized, include_attributes=False)
    assert run.VERSION == "industrial-none-derivation-v4"
    assert run.OUTPUT == "work/industrial-none-derivation-v4/actual-v4"
    assert run.UPSTREAM_KEYS == old.UPSTREAM_KEYS and len(run.UPSTREAM_KEYS) == 18
    assert run.MPS_MANIFEST == old.MPS_MANIFEST == run.upstream.BASE + "/prepared-manifest-v3.json"
    code = ast.unparse(b["admitted_inputs"])
    assert code.index("register_original_adapter_metadata") < code.index("source_admissions")
    assert "/prepared-manifest-v4.json" in ast.unparse(b["main"])

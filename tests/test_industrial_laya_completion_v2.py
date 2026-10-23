"""Industrial NoneV2 handoff adapter deltas only; no runner/model/data imports."""

import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/rag_benchmark/industrial_laya_completion_v2.py"
OLD = ROOT / "src/rag_benchmark/industrial_laya_completion_v1.py"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def ref(path):
    return {"path": path, "bytes": 123, "sha256": sha(path.encode())}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def definitions(path):
    return {
        n.name: n for n in ast.parse(path.read_text()).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
    }


@pytest.fixture
def gate():
    constants = {
        "VERSION",
        "WORK",
        "INDUSTRIAL_NONE_BASE",
        "INDUSTRIAL_MPS_BASE",
        "ORIGINAL_MPS_PATH",
        "ORIGINAL_MPS_SHA256",
        "UPSTREAM_REQUIRED",
    }
    inherited = {
        "CHANNELS",
        "TARGETS",
        "INDUSTRIAL_IDENTITY",
        "INDUSTRIAL_HARDWARE",
        "MEMORY",
        "LAYA_CONFIG",
        "CONTEXT",
    }
    nodes = []
    for path, names in ((SOURCE, constants), (OLD, inherited)):
        nodes += [
            n
            for n in ast.parse(path.read_text()).body
            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in names
        ]
    nodes += [
        definitions(SOURCE)[name] for name in ("industrial_inputs", "prepare_manifest", "context_evidence")
    ]
    namespace = {
        "require": require,
        "sha": sha,
        "canonical": canonical,
        "frozen": SimpleNamespace(),
        "runtime": lambda: {"synthetic": True},
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return SimpleNamespace(**namespace)


@pytest.fixture
def bundle(gate):
    base, mpsbase = gate.INDUSTRIAL_NONE_BASE, gate.INDUSTRIAL_MPS_BASE
    refs = {
        "handoff": ref(base + "/six-ranking-inputs.json"),
        "none_closed": ref(base + "/closed.json"),
        "actual_review": ref("work/synthetic/none-review.json"),
        "root_acceptance": ref("work/synthetic/none-root.json"),
        "baseline": ref(base + "/report.json"),
        "none_manifest": {**ref(base + "/manifest.json"), "sha256": "d" * 64},
        "none_wait": ref("work/synthetic/none-actual-wait.json"),
    }
    upstream = {key: ref("work/synthetic/" + key + ".proof") for key in gate.UPSTREAM_REQUIRED}
    upstream["original_manifest"] = {
        "path": gate.ORIGINAL_MPS_PATH,
        "bytes": 123,
        "sha256": gate.ORIGINAL_MPS_SHA256,
    }
    upstream["operational_envelope"] = ref(mpsbase + "/prepared-manifest-v3.json")
    rankings = {c: ref(base + "/" + c + "-ranking.json") for c in gate.CHANNELS}
    origins = {
        c: {
            **gate.INDUSTRIAL_HARDWARE[c],
            "report": ref(base + "/" + c + "-origin-report.json"),
            "channel_identity": sha((c + "-stage-identity").encode()),
        }
        for c in gate.CHANNELS
    }
    manifest = {
        "dataset_identity": gate.INDUSTRIAL_IDENTITY,
        "tables": {},
        "media": {},
        "ordered_id_sha256": ["a" * 64, "b" * 64],
        "rankings": rankings,
        "retrieval_provenance": origins,
        "baseline_report": refs["baseline"],
        "industrial_evidence": {
            key: refs[key] for key in ("handoff", "none_closed", "actual_review", "root_acceptance")
        },
    }
    envelope = {
        "protocol": "industrial-l4-bgen-local-mps-sj-v3",
        "dataset_id": "vidore-v3-industrial-en",
        "original_manifest": upstream["original_manifest"],
        "rebinding": upstream["boot_rebinding"],
        "operational_manifest": {
            key: copy.deepcopy(manifest[key])
            for key in ("dataset_identity", "tables", "media", "ordered_id_sha256")
        },
    }
    envelope_sha = sha(canonical(envelope))
    upstream["operational_envelope"]["sha256"] = envelope_sha
    stage_refs = {c: ref(mpsbase + "/actual-v3/" + c + "/closed.json") for c in "SJ"}
    mixed, none_manifest = "c" * 64, "d" * 64
    docs = {
        refs["handoff"]["path"]: {
            "protocol": "industrial-none-derivation-v2",
            "dataset_id": "vidore-v3-industrial-en",
            "dataset_identity": gate.INDUSTRIAL_IDENTITY,
            "rankings": rankings,
            "retrieval_provenance": origins,
            "baseline_report": refs["baseline"],
            "mixed_condition_identity": mixed,
            "input_manifest": upstream["operational_envelope"],
            "stage_closures": stage_refs,
            "upstream_evidence": upstream,
            "accepted_evidence": [ref("work/synthetic/raw-empty.proof")],
        },
        refs["none_closed"]["path"]: {
            "status": "closed",
            "completed_cells": 126,
            "prior_cell_agreements": 8,
            "mixed_condition_identity": mixed,
            "manifest_sha256": none_manifest,
            "files": {
                "manifest.json": refs["none_manifest"],
                "six-ranking-inputs.json": refs["handoff"],
                "report.json": refs["baseline"],
                **{c + "-ranking.json": rankings[c] for c in gate.CHANNELS},
                **{c + "-origin-report.json": origins[c]["report"] for c in gate.CHANNELS},
            },
        },
        refs["baseline"]["path"]: {
            "derivation_protocol": "industrial-none-derivation-v2",
            "retrieval_protocol": "industrial-l4-bgen-local-mps-sj-v3",
            "mixed_condition_identity": mixed,
            "source_manifest_sha256": none_manifest,
            "prior_cell_agreements": 8,
            "original_standalone_evidence": list(range(8)),
            "new_model_calls": 0,
            "new_embedding_rows": 0,
            "gpu_calls": 0,
            "laya_cells_completed": 0,
        },
        upstream["operational_envelope"]["path"]: envelope,
        refs["none_wait"]["path"]: {
            "actual_wait": True,
            "exit_code": 0,
            "signal": None,
            "manifest_sha256": none_manifest,
            "closed": refs["none_closed"],
        },
    }
    for key, status in (
        ("actual_review", "PASS_INDEPENDENT_INDUSTRIAL_NONE_V2_ACTUAL_OUTPUT"),
        ("root_acceptance", "PASS_ROOT_INDUSTRIAL_NONE_V2_ACTUAL_OUTPUT_ACCEPTED"),
    ):
        docs[refs[key]["path"]] = {
            "status": status,
            "blocking_findings": [],
            "handoff_sha256": refs["handoff"]["sha256"],
            "none_closed_sha256": refs["none_closed"]["sha256"],
            "baseline_report_sha256": refs["baseline"]["sha256"],
            "none_manifest_sha256": none_manifest,
            "actual_wait": refs["none_wait"],
            "accepted_evidence": [refs["none_wait"], ref("work/synthetic/none-review-raw.proof")],
            "completed_cells": 126,
            "prior_cell_agreements": 8,
            "metrics_per_cell": 17,
        }
    docs[refs["root_acceptance"]["path"]]["independent_review_sha256"] = refs["actual_review"]["sha256"]
    for c in "SJ":
        prefix = c.lower()
        docs[stage_refs[c]["path"]] = {
            "status": "completed",
            "protocol": envelope["protocol"],
            "manifest_sha256": envelope_sha,
            "original_manifest": upstream["original_manifest"],
            "rebinding": upstream["boot_rebinding"],
            "identity": origins[c]["channel_identity"],
            "execution_admission": upstream[prefix + "_execution_admission"],
            "files": {
                "rankings.json": {**rankings[c], "path": mpsbase + "/actual-v3/" + c + "/rankings.json"},
                "report.json": {**origins[c]["report"], "path": mpsbase + "/actual-v3/" + c + "/report.json"},
                "execution.claim.json": upstream[prefix + "_execution_claim"],
                "scoring.claim.json": ref(mpsbase + "/actual-v3/" + c + "/scoring.claim.json"),
            },
        }
        docs[upstream[prefix + "_execution_claim"]["path"]] = {
            "protocol": envelope["protocol"],
            "channel": c,
            "manifest_sha256": envelope_sha,
            "execution_admission": upstream[prefix + "_execution_admission"],
            "pid": 123,
        }
        docs[upstream[prefix + "_actual_wait"]["path"]] = {
            "actual_wait": True,
            "exit_code": 0,
            "signal": None,
            "channel": c,
            "manifest_sha256": envelope_sha,
            "closed": stage_refs[c],
            "execution_claim": upstream[prefix + "_execution_claim"],
            "pid": 123,
        }
    handoff = docs[refs["handoff"]["path"]]
    manifest["accepted_evidence"] = (
        list(refs.values())
        + list(upstream.values())
        + list(stage_refs.values())
        + handoff["accepted_evidence"]
        + list(docs[refs["none_closed"]["path"]]["files"].values())
        + docs[refs["actual_review"]["path"]]["accepted_evidence"]
    )
    checked, read, changed = [], [], set()
    expected = {r["path"]: copy.deepcopy(r) for r in manifest["accepted_evidence"]}

    def check(_root, reference):
        require(
            reference == expected[reference["path"]] and reference["path"] not in changed,
            "synthetic_raw_changed",
        )
        checked.append(reference["path"])
        return b"" if reference["path"].endswith("raw-empty.proof") else b"not-json-source-proof"

    def read_json(root, reference):
        check(root, reference)
        read.append(reference["path"])
        return docs[reference["path"]]

    gate.frozen.checked = check
    gate.frozen.read_json = read_json
    return SimpleNamespace(
        root=Path("/synthetic"),
        manifest=manifest,
        refs=refs,
        upstream=upstream,
        docs=docs,
        checked=checked,
        read=read,
        changed=changed,
        handoff=handoff,
        stage_refs=stage_refs,
    )


def test_new_handoff_accepts_raw_empty_proofs_without_json_and_propagates_all_18_refs(gate, bundle):
    gate.industrial_inputs(bundle.root, bundle.manifest)
    assert set(r["path"] for r in bundle.upstream.values()) <= set(bundle.checked)
    raw = bundle.handoff["accepted_evidence"][0]["path"]
    assert raw in bundle.checked and raw not in bundle.read
    assert bundle.upstream["none_source_contract"]["path"] in bundle.checked
    assert bundle.upstream["none_source_contract"]["path"] not in bundle.read
    raw_review = "work/synthetic/none-review-raw.proof"
    assert raw_review in bundle.checked and raw_review not in bundle.read


@pytest.mark.parametrize("path", ["manifest.json", "none-review-raw.proof"])
def test_none_closure_and_nested_review_proof_are_freshness_bound(gate, bundle, path):
    selected = next(r for r in bundle.manifest["accepted_evidence"] if r["path"].endswith("/" + path))
    bundle.changed.add(selected["path"])
    with pytest.raises(ValueError, match="synthetic_raw_changed"):
        gate.industrial_inputs(bundle.root, bundle.manifest)
    bundle.changed.clear()
    bundle.manifest["accepted_evidence"] = [r for r in bundle.manifest["accepted_evidence"] if r != selected]
    with pytest.raises(ValueError, match="industrial_none_(closed|review_proof)_freshness"):
        gate.industrial_inputs(bundle.root, bundle.manifest)


@pytest.mark.parametrize(
    "key,value",
    [
        ("actual_wait", False),
        ("exit_code", None),
        ("manifest_sha256", "f" * 64),
        ("closed", ref("wrong-closed")),
    ],
)
def test_none_actual_wait_cannot_be_assumed_from_source_or_counters(gate, bundle, key, value):
    bundle.docs[bundle.refs["none_wait"]["path"]][key] = value
    with pytest.raises(ValueError, match="industrial_none_actual_wait"):
        gate.industrial_inputs(bundle.root, bundle.manifest)


@pytest.mark.parametrize(
    "key", ["none_source_review", "sj_actual_output_review", "j_actual_wait", "s_execution_claim"]
)
def test_transitive_upstream_missing_or_changed_rejected(gate, bundle, key):
    bundle.changed.add(bundle.upstream[key]["path"])
    with pytest.raises(ValueError, match="synthetic_raw_changed"):
        gate.industrial_inputs(bundle.root, bundle.manifest)
    bundle.changed.clear()
    bundle.manifest["accepted_evidence"].remove(bundle.upstream[key])
    with pytest.raises(ValueError, match="industrial_upstream_freshness"):
        gate.industrial_inputs(bundle.root, bundle.manifest)


@pytest.mark.parametrize(
    "mutation",
    [
        "old_handoff",
        "old_baseline",
        "root_binding",
        "wait_pid",
        "wait_missing",
        "copy_sha",
        "claim_admission",
        "ordered_ids",
    ],
)
def test_new_lineage_and_true_wait_gates_fail_closed(gate, bundle, mutation):
    docs, refs, upstream = bundle.docs, bundle.refs, bundle.upstream
    if mutation == "old_handoff":
        bundle.handoff["protocol"] = "industrial-none-derivation-v1"
    elif mutation == "old_baseline":
        docs[refs["baseline"]["path"]]["retrieval_protocol"] = "industrial-l4-bgen-local-mps-sj-v2"
    elif mutation == "root_binding":
        docs[refs["root_acceptance"]["path"]]["independent_review_sha256"] = "f" * 64
    elif mutation == "wait_pid":
        docs[upstream["j_actual_wait"]["path"]]["pid"] = 124
    elif mutation == "wait_missing":
        docs[upstream["s_actual_wait"]["path"]]["actual_wait"] = False
    elif mutation == "copy_sha":
        docs[bundle.stage_refs["S"]["path"]]["files"]["rankings.json"]["sha256"] = "f" * 64
    elif mutation == "claim_admission":
        docs[upstream["j_execution_claim"]["path"]]["execution_admission"] = ref("different")
    elif mutation == "ordered_ids":
        bundle.manifest["ordered_id_sha256"] = ["f" * 64] * 2
    with pytest.raises(ValueError):
        gate.industrial_inputs(bundle.root, bundle.manifest)


def test_parent_prepare_is_new_namespace_and_first_context_has_new_binding(gate):
    captured = []
    gate.prepare_manifest.__globals__["validate"] = lambda root, value: captured.append((root, value))
    keys = {
        "protocol",
        "dataset_id",
        "source_sha256",
        "dataset_identity",
        "tables",
        "media",
        "ordered_id_sha256",
        "rankings",
        "retrieval_provenance",
        "baseline_report",
        "accepted_evidence",
        "model_snapshot",
        "model_files",
        "laya_adapter_identity",
        "gpu_lock",
        "sdk_source_sha256",
        "industrial_evidence",
    }
    recipe = dict.fromkeys(keys)
    recipe.update(
        protocol=gate.VERSION,
        dataset_id="vidore-v3-industrial-en",
        gpu_lock={"path": "work/overnight/gpu.lock", "device": 123, "inode": 456},
    )
    manifest = gate.prepare_manifest(Path("/synthetic"), recipe)
    assert manifest["output"] == "work/industrial-laya-completion-v2/runs/vidore-v3-industrial-en"
    assert manifest["gpu_lock"] == recipe["gpu_lock"] and len(captured) == 1
    with pytest.raises(ValueError, match="context_preflight_required"):
        gate.context_evidence(
            Path("/synthetic"),
            manifest,
            "a" * 64,
            {"path": "work/industrial-laya-completion-v1/preflight/vidore-v3-industrial-en.json"},
            None,
            None,
        )


def test_scientific_helpers_are_normal_imports_and_validate_tail_context_semantics_unchanged():
    old, new = definitions(OLD), definitions(SOURCE)
    tree = ast.parse(SOURCE.read_text())
    imported = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == "industrial_laya_completion_v1"
        for alias in node.names
    }
    assert {
        "pairs",
        "graph",
        "union_ids",
        "scatter",
        "closed_value",
        "baseline_cells",
        "query_binding",
        "TokenizerProbe",
    } <= imported
    assert "execute" not in new and "execute" not in imported
    for name in ("prepare_manifest", "validate", "preflight", "context_evidence"):
        expected = ast.get_source_segment(OLD.read_text(), old[name]).replace(
            "work/industrial-laya-completion-v1/", "work/industrial-laya-completion-v2/"
        )
        assert ast.dump(ast.parse(expected).body[0]) == ast.dump(new[name]), name
    source_text = ast.get_source_segment(SOURCE.read_text(), new["sources"])
    assert "previous.sources(root" in source_text and "loaded_code_changed" in source_text
    assert not any(
        isinstance(n, (ast.Assign, ast.AugAssign))
        and any(
            isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == "previous"
            for t in (n.targets if isinstance(n, ast.Assign) else [n.target])
        )
        for n in ast.walk(tree)
    )
    main = ast.get_source_segment(SOURCE.read_text(), new["main"])
    assert '"--execute"' not in main and '"--resume-seal"' not in main


def test_frozen_parent_bytes_preserved():
    assert sha(OLD.read_bytes()) == "62d4a41fac2753bfbbf9bf05353ee86e75826e6274b297a191b7256de6b67baa"

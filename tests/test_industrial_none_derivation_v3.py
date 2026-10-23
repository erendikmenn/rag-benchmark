"""Only the new admitted model-view descriptor boundary; no actual derivation."""

import ast
import copy
import hashlib
import json
from pathlib import Path

import pytest

from rag_benchmark import industrial_none_derivation_v2 as old
from rag_benchmark import industrial_none_derivation_v3 as run


def pin(root, path, value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    p = root / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return {"path": path, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def fixture(root, damage=None):
    values, refs, control, closures, configs, waits, parents, descriptors = {}, {}, [], {}, {}, {}, {}, []
    digest = "a" * 64
    requirement = pin(root, "proofs/config-requirements.json", {"synthetic": True})
    refs["sj_actual_config_requirements"] = requirement
    for channel, count in run.MODEL_VIEW_COUNTS.items():
        lower, pid = channel.lower(), (321 if channel == "J" else 322)
        stage = run.upstream.DELTA + "/" + channel
        ex = pin(root, "proofs/execution-" + channel + ".json", {"synthetic": True})
        claim = pin(
            root,
            stage + "/execution.claim.json",
            {
                "protocol": run.upstream.VERSION,
                "channel": channel,
                "manifest_sha256": digest,
                "execution_admission": ex,
                "boot_epoch_seconds": run.upstream.BOOT_EPOCH,
                "pid": pid,
            },
        )
        closed = pin(root, stage + "/closed.json", {"synthetic": True, "channel": channel})
        wait = pin(
            root,
            "proofs/wait-" + channel + ".json",
            {
                "actual_wait": True,
                "exit_code": 0,
                "signal": None,
                "channel": channel,
                "manifest_sha256": digest,
                "pid": pid,
                "execution_claim": claim,
                "closed": closed,
            },
        )
        parent = pin(root, "proofs/parent-" + channel + ".json", {"actual_wait": True, "exit_code": 0})
        config = pin(
            root,
            "proofs/config-" + channel + ".json",
            {
                "status": "PASS_INDEPENDENT_INDUSTRIAL_REBOOT_EXECUTION_CONFIG",
                "blocking_findings": [],
                "channel": channel,
                "manifest_sha256": digest,
                "execution_admission": ex,
                "actual_config_requirements": requirement,
                "checks": {
                    k: True
                    for k in (
                        "all_six_heavy_closures_verified",
                        "all_required_actual_parent_and_child_waits_verified",
                        "complete_transitive_proof_closure_verified",
                        "full_prior_j_nested_closure_verified",
                    )
                },
            },
        )
        namespace = {}
        for index in range(count):
            name = stage + "/model-view-" + str(pid) + "/" + str(index) + "/config.json"
            desc = {"path": name, "bytes": 17, "sha256": hashlib.sha256(name.encode()).hexdigest()}
            namespace[name] = {**desc, "kind": "file", "mode": "0600"}
            descriptors.append(desc)
        closure = {
            "status": "PASS_" + channel + "_EXACT_CLOSED_NAMESPACE_AND_NESTED_REFS",
            "channel": channel,
            "manifest_sha256": digest,
            "closed": closed,
            "actual_wait": wait,
            "execution_claim": claim,
            "pre_post_identical": True,
            "actual_parent_tool_wait": parent,
            "namespace": namespace,
        }
        values[channel] = closure
        configs[channel], waits[channel], parents[channel] = config, wait, parent
        refs.update(
            {
                lower + "_execution_admission": ex,
                lower + "_execution_claim": claim,
                lower + "_actual_wait": wait,
            }
        )
        control.extend([ex, claim, wait, parent, config])
    if damage in ("symlink", "weight", "unknown_name", "wrong_pid", "mode"):
        ns = values["J"]["namespace"]
        first = next(iter(ns))
        item = ns.pop(first)
        if damage == "symlink":
            item["kind"] = "symlink"
        elif damage == "mode":
            item["mode"] = "0777"
        else:
            suffix = {
                "weight": "model.safetensors",
                "unknown_name": "arbitrary.secret",
                "wrong_pid": "config.json",
            }[damage]
            first = first.rsplit("/", 1)[0] + "/" + suffix
            if damage == "wrong_pid":
                first = first.replace("model-view-321", "model-view-999")
            item["path"] = first
        ns[first] = item
    for channel in "JS":
        closures[channel] = pin(root, "proofs/closure-" + channel + ".json", values[channel])
        control.append(closures[channel])
    accepted = control + descriptors
    review = {
        "status": "PASS_INDEPENDENT_INDUSTRIAL_SJ_ACTUAL_OUTPUTS",
        "blocking_findings": [],
        "manifest_sha256": digest,
        "original_manifest_sha256": run.upstream.PARENT_SHA256,
        "closed_sha256": {c: values[c]["closed"]["sha256"] for c in "SJ"},
        "wait_sha256": {c: waits[c]["sha256"] for c in "SJ"},
        "vector_payload_validation_completed": True,
        "metadata_only_cpu_reuse_allowed": True,
        "full_nested_closures": closures,
        "execution_config_reviews": configs,
        "accepted_evidence": copy.deepcopy(accepted),
        "actual_child_waits": waits,
        "actual_parent_tool_waits": parents,
    }
    if damage == "independent_missing":
        review["accepted_evidence"].remove(descriptors[0])
    if damage == "review_status":
        review["status"] = "SOURCE_ONLY"
    if damage == "review_blocked":
        review["blocking_findings"] = ["not reviewed"]
    review_ref = pin(root, "proofs/combined-review.json", review)
    root_value = {
        "status": "PASS_ROOT_INDUSTRIAL_SJ_REUSE_FOR_NONE",
        "independent_review": review_ref,
        "independent_review_sha256": review_ref["sha256"],
        "complete_upstream_actual_config_reviewed": True,
        **{
            k: review[k]
            for k in (
                "manifest_sha256",
                "original_manifest_sha256",
                "closed_sha256",
                "wait_sha256",
                "vector_payload_validation_completed",
                "metadata_only_cpu_reuse_allowed",
                "full_nested_closures",
                "actual_child_waits",
                "actual_parent_tool_waits",
            )
        },
        "execution_config_review_sha256": {c: configs[c]["sha256"] for c in "JS"},
        "accepted_evidence": copy.deepcopy(accepted),
    }
    if damage == "root_missing":
        root_value["accepted_evidence"].remove(descriptors[0])
    if damage in ("changed_sha", "changed_bytes", "conflict"):
        i = next(
            i for i, r in enumerate(root_value["accepted_evidence"]) if r["path"] == descriptors[0]["path"]
        )
        changed = {
            **descriptors[0],
            ("bytes" if damage == "changed_bytes" else "sha256"): (
                18 if damage == "changed_bytes" else "f" * 64
            ),
        }
        if damage == "conflict":
            root_value["accepted_evidence"].append(changed)
        else:
            root_value["accepted_evidence"][i] = changed
    if damage == "extra_model_view":
        root_value["accepted_evidence"].append(
            {
                "path": run.upstream.DELTA + "/J/model-view-321/extra/config.json",
                "bytes": 9,
                "sha256": "1" * 64,
            }
        )
    if damage == "root_status":
        root_value["status"] = "SOURCE_ONLY"
    if damage == "wrong_review_sha":
        root_value["independent_review_sha256"] = "0" * 64
    if damage == "root_vector_audit":
        root_value["vector_payload_validation_completed"] = False
    if damage == "root_config_sha":
        root_value["execution_config_review_sha256"]["S"] = "0" * 64
    refs["sj_actual_output_review"] = review_ref
    refs["sj_root_output_acceptance"] = pin(root, "proofs/root-acceptance.json", root_value)
    admission = {"mps_manifest_sha256": digest, "closed_sha256": review["closed_sha256"]}
    return refs, admission, descriptors, root_value, review


def test_prior_boundary_red_new_exact_registry_green_without_model_view_files(tmp_path):
    refs, admission, descriptors, accepted, review = fixture(tmp_path)
    with pytest.raises(ValueError, match="cpu_metadata_boundary"):
        old.Proofs(tmp_path).embedded(descriptors[0])
    proofs = run.Proofs(tmp_path)
    registry = run.register_model_view_metadata(proofs, refs, admission)
    assert registry["channels"]["J"]["count"] == 12
    assert registry["channels"]["S"]["count"] == 31
    assert registry["payload_reads"] == 0
    assert len(proofs.descriptors) == 43
    assert set(proofs.descriptors) == {r["path"] for r in descriptors}
    proofs.embedded(accepted)
    proofs.embedded(review)
    proofs.fresh()
    assert not any("/model-view-" in p for p in proofs.refs)
    assert all(not (tmp_path / r["path"]).exists() for r in descriptors)
    for ref in descriptors:
        with pytest.raises(ValueError, match="cpu_metadata_boundary"):
            proofs.raw(ref)


@pytest.mark.parametrize(
    "damage",
    [
        "symlink",
        "weight",
        "unknown_name",
        "wrong_pid",
        "mode",
        "independent_missing",
        "root_missing",
        "review_status",
        "review_blocked",
        "changed_sha",
        "changed_bytes",
        "conflict",
        "extra_model_view",
        "root_status",
        "wrong_review_sha",
        "root_vector_audit",
        "root_config_sha",
    ],
)
def test_registry_rejects_unadmitted_or_changed_metadata_and_proof_bindings(tmp_path, damage):
    refs, admission, _descriptors, _accepted, _review = fixture(tmp_path, damage)
    proofs = run.Proofs(tmp_path)
    with pytest.raises(ValueError):
        run.register_model_view_metadata(proofs, refs, admission)
    assert proofs.descriptors == {}


@pytest.mark.parametrize("field,value", [("sha256", "0" * 64), ("bytes", 19)])
def test_changed_registered_descriptor_and_unregistered_prefix_still_fail(tmp_path, field, value):
    refs, admission, descriptors, _accepted, _review = fixture(tmp_path)
    proofs = run.Proofs(tmp_path)
    run.register_model_view_metadata(proofs, refs, admission)
    with pytest.raises(ValueError, match="payload_descriptor_changed"):
        proofs.embedded({**descriptors[0], field: value})
    with pytest.raises(ValueError, match="cpu_metadata_boundary"):
        proofs.embedded({**descriptors[0], "path": descriptors[0]["path"] + ".new"})


def test_cpu_boundary_and_pure_helpers_remain_unchanged_and_registry_precedes_nested_walk():
    before = ast.parse(Path(old.__file__).read_text())
    after = ast.parse(Path(run.__file__).read_text())

    def node(tree, name):
        return next(n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)

    for name in ("metadata_ref", "cpu_readable", "stage_bundle", "actual_output_acceptance"):
        assert ast.dump(node(before, name), include_attributes=False) == ast.dump(
            node(after, name), include_attributes=False
        )
    old_methods = {n.name: n for n in node(before, "Proofs").body if isinstance(n, ast.FunctionDef)}
    new_methods = {n.name: n for n in node(after, "Proofs").body if isinstance(n, ast.FunctionDef)}
    for name in ("raw", "read", "descriptor", "embedded", "fresh"):
        assert ast.dump(old_methods[name], include_attributes=False) == ast.dump(
            new_methods[name], include_attributes=False
        )
    for name in (
        "configuration",
        "derive_rows",
        "strict_ranking",
        "metrics_equal",
        "prior_agreement",
        "archive_bundle",
    ):
        assert getattr(run, name) is getattr(old, name) is getattr(run.previous, name)
    code = ast.unparse(node(after, "admitted_inputs"))
    assert code.index("register_model_view_metadata") < code.index("source_admissions")
    assert run.VERSION == "industrial-none-derivation-v3"
    assert run.OUTPUT == "work/industrial-none-derivation-v3/actual-v3"
    assert run.UPSTREAM_KEYS == old.UPSTREAM_KEYS and len(run.UPSTREAM_KEYS) == 18


def test_new_cli_prepared_identity_is_v3_without_calling_prepare():
    tree = ast.parse(Path(run.__file__).read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    strings = {n.value for n in ast.walk(main) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert "/prepared-manifest-v3.json" in strings
    assert "/prepared-manifest-v2.json" not in strings

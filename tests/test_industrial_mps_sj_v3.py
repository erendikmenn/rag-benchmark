"""Industrial reboot-only synthetic delta; no actual inputs, models or prepare/run."""
import ast
import copy
from pathlib import Path

import pytest

from rag_benchmark import industrial_mps_sj_v3 as run


ROOT = Path(__file__).resolve().parents[1]


def pin(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value if isinstance(value, bytes) else run.frozen.canonical(value))
    return run.frozen.record(root, name)


def manifest_fixture():
    return {"lock": {"path": "work/overnight/gpu.lock", "device": run.OLD_DEVICE,
                     "inode": run.LOCK_INODE, "bytes": 0},
            "assets": {"J": {"revision": "synthetic", "assets": {
                str(i): {"logical_path": f".cache/models/link{i}", "resolved_path": f".cache/models/blob{i}",
                    "file": {"path": f".cache/models/blob{i}", "bytes": 1, "sha256": "a" * 64},
                    "symlink_chain": [{"path": f".cache/models/link{i}", "target": f"blob{i}",
                        "device": run.OLD_DEVICE, "inode": i + 1, "mtime_ns": 100, "ctime_ns": 200}]}
                for i in range(run.LINK_OCCURRENCES)}}},
            "hardware": {"J": {"device": "mps", "dtype": "bfloat16"}},
            "inherited": {"B": "CPU", "GEN": "NVIDIA L4"}}


def binding_fixture(root, manifest):
    changes = run.device_changes(manifest)
    projected = run.project_manifest(manifest, changes)
    parent = {"path": run.PARENT_PATH, "bytes": 1, "sha256": run.PARENT_SHA256}
    boot = pin(root, "proof/boot.json", {"schema": "vidore-laya-reboot-process-observation-v1",
        "boot_epoch_seconds": run.BOOT_EPOCH, "reboot_cause": "UNKNOWN"})
    lock = pin(root, "work/overnight/gpu.lock", b"")
    observation = {"protocol": run.OBSERVATION_PROTOCOL,
        "status": "OBSERVED_POSTBOOT_LOCK_AND_ASSET_METADATA", "original_manifest": parent,
        "boot_epoch_seconds": run.BOOT_EPOCH, "boot_observation": boot,
        "observed_at_utc": "2026-10-23T07:00:00Z", "changed_fields": changes,
        "original_lock": manifest["lock"], "operational_lock": projected["lock"], "current_lock": lock,
        "current_mode": 0o644, "lock_uid": 501, "lock_gid": 20, "lock_nlink": 1,
        "descriptor_matches_path_before_and_after": True, "lock_created_written_unlinked_replaced": False,
        "immutable_files_modified": False, "historical_volume_identity_attested": False, "cause": "UNKNOWN",
        "symlink_occurrences": [{"field": c["field"][:-1], "original": run.at_field(manifest, c["field"][:-1]),
            "operational": run.at_field(projected, c["field"][:-1])} for c in changes[1:]]}
    observation_ref = pin(root, "proof/observation.json", observation)
    review = {"status": "PASS_INDEPENDENT_INDUSTRIAL_BOOT_DEVICE_REBINDING", "blocking_findings": [],
        "observation_sha256": observation_ref["sha256"], "original_manifest_sha256": parent["sha256"],
        "changed_fields_sha256": run.frozen.digest(changes)}
    binding = {"protocol": run.BINDING_PROTOCOL, "status": "PASS_ROOT_INDUSTRIAL_BOOT_DEVICE_REBINDING",
        "original_manifest": parent, "boot_epoch_seconds": run.BOOT_EPOCH, "old_device": run.OLD_DEVICE,
        "new_device": run.NEW_DEVICE, "changed_fields": changes, "cause": "UNKNOWN",
        "historical_volume_identity_attested": False, "current_lock": lock, "current_mode": 0o644,
        "lock_uid": 501, "lock_gid": 20, "lock_nlink": 1,
        "refs": {"observation": observation_ref, "boot_observation": boot,
                 "independent_review": pin(root, "proof/independent.json", review),
                 "raw_source": pin(root, "proof/source.py", b"# synthetic source\n"),
                 "empty": pin(root, "proof/empty", b"")}}
    recipe = {"original_manifest": parent, "rebinding": pin(root, "proof/binding.json", binding)}
    return recipe, binding, observation, review, projected


def test_old_fixed_recipe_rejects_device_projection_before_payload_or_model(tmp_path):
    # Actual old fixed-condition seam, synthetic inputs; the first guard rejects.
    old = run.original.recipe(tmp_path, {})
    changed = copy.deepcopy(old)
    changed["lock"]["device"] = run.NEW_DEVICE
    with pytest.raises(ValueError, match="Fixed industrial condition"):
        run.original.validate(tmp_path, changed)


def test_only_58_explicit_device_fields_change_and_old_object_is_unchanged():
    old = manifest_fixture()
    sealed = run.frozen.canonical(old)
    changes = run.device_changes(old)
    assert len(changes) == 58
    projected = run.project_manifest(old, changes)
    assert run.frozen.canonical(old) == sealed
    assert projected["inherited"] == old["inherited"]
    assert projected["hardware"] == old["hardware"]
    for change in changes:
        assert run.at_field(projected, change["field"]) == run.NEW_DEVICE
    for asset in projected["assets"]["J"]["assets"].values():
        assert asset["file"]["sha256"] == "a" * 64


@pytest.mark.parametrize("damage", ["missing", "duplicate", "extra", "inode", "wrong_device", "wrong_count"])
def test_projection_rejects_scope_expansion_or_partial_substitution(damage):
    old = manifest_fixture()
    changes = run.device_changes(old)
    if damage == "missing":
        changes.pop()
    elif damage == "duplicate":
        changes[-1] = copy.deepcopy(changes[-2])
    elif damage == "extra":
        changes.append({"field": ["hardware", "J", "device"], "old": "mps", "new": "cpu"})
    elif damage == "inode":
        changes[0]["field"][-1] = "inode"
    elif damage == "wrong_device":
        changes[1]["new"] = run.NEW_DEVICE + 1
    else:
        old["assets"]["J"]["assets"].pop("0")
    with pytest.raises(ValueError):
        run.project_manifest(old, changes)


def test_full_binding_accepts_only_metadata_delta_with_raw_and_empty_proof_refs(tmp_path, monkeypatch):
    old = manifest_fixture()
    recipe, binding, _, _, projected = binding_fixture(tmp_path, old)
    observed = []
    monkeypatch.setattr(run, "check_live_metadata", lambda *args: observed.append(args))
    result, operational = run.checked_binding(tmp_path, recipe, old)
    assert result == binding and operational == projected and len(observed) == 1
    (tmp_path / binding["refs"]["raw_source"]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError):
        run.checked_binding(tmp_path, recipe, old)


@pytest.mark.parametrize("damage", ["boot", "cause", "volume", "review", "review_hash", "parent", "observed_preboot",
    "lock_inode", "link_inode", "link_target", "link_time", "mode", "descriptors", "mutated", "occurrence_missing"])
def test_binding_rejects_unreviewed_identity_or_provenance_change(tmp_path, monkeypatch, damage):
    old = manifest_fixture()
    recipe, binding, observation, review, _ = binding_fixture(tmp_path, old)
    monkeypatch.setattr(run, "check_live_metadata", lambda *args: None)
    if damage == "boot":
        binding["boot_epoch_seconds"] += 1
    elif damage == "cause":
        binding["cause"] = "assumed APFS renumbering"
    elif damage == "volume":
        binding["historical_volume_identity_attested"] = True
    elif damage == "review":
        review["status"] = "PASS_SOURCE_ONLY"
    elif damage == "review_hash":
        review["changed_fields_sha256"] = "f" * 64
    elif damage == "parent":
        binding["original_manifest"] = {**recipe["original_manifest"], "sha256": "f" * 64}
    elif damage == "observed_preboot":
        observation["observed_at_utc"] = "2026-10-23T06:00:00Z"
    elif damage == "lock_inode":
        observation["operational_lock"] = {**observation["operational_lock"], "inode": 99}
    elif damage.startswith("link_"):
        key = {"link_inode": "inode", "link_target": "target", "link_time": "mtime_ns"}[damage]
        observation["symlink_occurrences"][0]["operational"] = {
            **observation["symlink_occurrences"][0]["operational"], key: "changed"}
    elif damage == "mode":
        binding["current_mode"] = observation["current_mode"] = 0o666
    elif damage == "descriptors":
        observation["descriptor_matches_path_before_and_after"] = False
    elif damage == "mutated":
        observation["immutable_files_modified"] = True
    else:
        observation["symlink_occurrences"].pop()
    # Re-seal to show these semantic gates reject even mutually consistent byte refs.
    binding["refs"]["observation"] = pin(tmp_path, "proof/observation.json", observation)
    review["observation_sha256"] = binding["refs"]["observation"]["sha256"]
    binding["refs"]["independent_review"] = pin(tmp_path, "proof/independent.json", review)
    recipe["rebinding"] = pin(tmp_path, "proof/binding.json", binding)
    with pytest.raises(ValueError):
        run.checked_binding(tmp_path, recipe, old)


def test_live_metadata_checks_current_boot_without_payload_hashing(tmp_path, monkeypatch):
    old = manifest_fixture()
    _, binding, _, _, operational = binding_fixture(tmp_path, old)
    monkeypatch.setattr(run, "current_boot_epoch", lambda: run.BOOT_EPOCH + 1)
    with pytest.raises(ValueError, match="boot_changed"):
        run.check_live_metadata(tmp_path, operational, binding)


def test_live_metadata_rejects_replaced_lock_descriptor(tmp_path, monkeypatch):
    old = manifest_fixture()
    _, binding, _, _, operational = binding_fixture(tmp_path, old)
    monkeypatch.setattr(run, "current_boot_epoch", lambda: run.BOOT_EPOCH)
    # Fixture lock is a real unrelated inode; it must not pass merely because bytes match.
    with pytest.raises(ValueError, match="live_lock_descriptor_identity"):
        run.check_live_metadata(tmp_path, operational, binding)


def admission_fixture(root):
    envelope = {"original_manifest": {"sha256": run.PARENT_SHA256},
                "rebinding": {"sha256": "b" * 64}, "source_pins": {"synthetic": "s" * 64}}
    digest = run.frozen.digest(envelope)
    heavy = pin(root, "proof/heavy.json", {"status": "PASS_ROOT_ALL_SIX_LAYA_HEAVY_CLOSED",
        "dataset_count": 6, "active_heavy_workers": 0, "actual_waits_verified": True,
        "refs": {"raw_wait_proof": pin(root, "proof/wait-bytes", b"synthetic actual wait")}})
    review = pin(root, "proof/manifest-review.json", {"status": "PASS_INDEPENDENT_INDUSTRIAL_REBOOT_MANIFEST",
        "blocking_findings": [], "manifest_sha256": digest})
    admission = {"protocol": run.VERSION, "status": "PASS_ROOT_INDUSTRIAL_REBOOT_EXECUTION_AUTHORIZED",
        "actual_model_execution_authorized": True, "channel": "J", "manifest_sha256": digest,
        "source_pins": envelope["source_pins"], "rebinding": envelope["rebinding"],
        "boot_epoch_seconds": run.BOOT_EPOCH, "refs": {"heavy_completion": heavy,
        "independent_manifest_review": review}, "prior_j_stage": None}
    return envelope, digest, admission


@pytest.mark.parametrize("damage", [None, "source_only", "boot", "channel", "manifest", "active_heavy", "incomplete", "unknown_wait", "raw_stale"])
def test_execution_admission_requires_reviewed_manifest_and_all_six_heavy_closed(tmp_path, monkeypatch, damage):
    envelope, digest, admission = admission_fixture(tmp_path)
    monkeypatch.setattr(run, "current_boot_epoch", lambda: run.BOOT_EPOCH)
    if damage == "source_only":
        admission["actual_model_execution_authorized"] = False
    elif damage == "boot":
        admission["boot_epoch_seconds"] += 1
    elif damage == "channel":
        admission["channel"] = "S"
    elif damage == "manifest":
        admission["manifest_sha256"] = "x" * 64
    elif damage == "raw_stale":
        (tmp_path / "proof/wait-bytes").write_bytes(b"changed")
    elif damage in ("active_heavy", "incomplete", "unknown_wait"):
        heavy = run.read_pinned(tmp_path, admission["refs"]["heavy_completion"])
        heavy[{"active_heavy": "active_heavy_workers", "incomplete": "dataset_count", "unknown_wait": "actual_waits_verified"}[damage]] = 1
        admission["refs"]["heavy_completion"] = pin(tmp_path, "proof/heavy.json", heavy)
    ref = pin(tmp_path, "proof/execute-admission.json", admission)
    if damage:
        with pytest.raises(ValueError):
            run.check_execution_admission(tmp_path, envelope, digest, "J", ref)
    else:
        assert run.check_execution_admission(tmp_path, envelope, digest, "J", ref) == admission


@pytest.mark.parametrize("damage", [None, "missing", "unknown_exit", "wrong_pid", "wrong_closed", "stale_claim"])
def test_s_requires_exact_j_closed_and_true_wait_pid_binding(tmp_path, monkeypatch, damage):
    envelope, digest, admission = admission_fixture(tmp_path)
    monkeypatch.setattr(run, "current_boot_epoch", lambda: run.BOOT_EPOCH)
    claim = pin(tmp_path, run.DELTA + "/J/execution.claim.json", {"protocol": run.VERSION, "channel": "J",
        "manifest_sha256": digest, "boot_epoch_seconds": run.BOOT_EPOCH, "pid": 42})
    closed = pin(tmp_path, run.DELTA + "/J/closed.json", {"status": "completed", "manifest_sha256": digest,
        "identity": run.stage_identity(envelope, digest, "J"), "files": {"execution.claim.json": claim}})
    wait = {"actual_wait": True, "exit_code": 0, "signal": None, "channel": "J", "manifest_sha256": digest,
            "closed": closed, "pid": 42, "execution_claim": claim}
    if damage == "unknown_exit":
        wait["exit_code"] = None
    elif damage == "wrong_pid":
        wait["pid"] = 43
    elif damage == "wrong_closed":
        wait["closed"] = {**closed, "sha256": "x" * 64}
    elif damage == "stale_claim":
        (tmp_path / claim["path"]).write_bytes(b"{}")
    admission["channel"] = "S"
    admission["prior_j_stage"] = {"closed": closed, "wait": pin(tmp_path, "proof/j-wait.json", wait), "execution_claim": claim}
    if damage == "missing":
        admission["prior_j_stage"] = {}
    ref = pin(tmp_path, "proof/execute-admission.json", admission)
    if damage:
        with pytest.raises((ValueError, KeyError)):
            run.check_execution_admission(tmp_path, envelope, digest, "S", ref)
    else:
        assert run.check_execution_admission(tmp_path, envelope, digest, "S", ref) == admission


def test_operational_envelope_rejects_any_non_device_input_change(tmp_path, monkeypatch):
    # Exercise actual envelope check with prior gates stubbed, no real assets read.
    old = manifest_fixture()
    projected = run.project_manifest(old, run.device_changes(old))
    recipe = dict.fromkeys(run.RECIPE_FIELDS)
    recipe.update(source_pins={}, original_manifest={"path": "synthetic"}, rebinding={})
    old["source_pins"], old["recipe_sha256"] = {}, run.frozen.digest({})
    monkeypatch.setattr(run, "recipe_scope", lambda *args: None)
    monkeypatch.setattr(run, "check_sources", lambda *args: None)
    monkeypatch.setattr(run, "source_admissions", lambda *args: None)
    monkeypatch.setattr(run, "read_pinned", lambda *args: old)
    monkeypatch.setattr(run, "PARENT_SHA256", run.frozen.digest(old))
    monkeypatch.setattr(run.original, "recipe", lambda *args: {})
    monkeypatch.setattr(run, "checked_binding", lambda *args: ({}, projected))
    envelope = {**recipe, "recipe_sha256": run.frozen.digest(recipe), "operational_manifest": projected}
    run.load(tmp_path, envelope)
    envelope["operational_manifest"] = copy.deepcopy(projected)
    envelope["operational_manifest"]["assets"]["J"]["assets"]["0"]["file"]["sha256"] = "changed"
    with pytest.raises(ValueError, match="operational_manifest_diff"):
        run.load(tmp_path, envelope)


def test_original_scientific_validation_block_is_unchanged():
    old = (ROOT / "src/rag_benchmark/industrial_mps_sj_v2.py").read_text()
    new = Path(run.__file__).read_text()
    start = "    metadata, corpus, queries, qrels, exclusions = frozen._dataset(root, manifest)"
    end = "    return modules, corpus, queries, qrels, exclusions"
    def block(text):
        return text[text.index(start):text.index(end, text.index(start)) + len(end)]
    expected = block(old)
    actual = block(new).replace("original.asset_inventory", "asset_inventory").replace("original.archive_inputs", "archive_inputs").replace("original.read_member", "read_member")
    assert actual == expected


def test_original_scoring_body_is_byte_identical_and_model_helpers_are_delegated():
    old = (ROOT / "src/rag_benchmark/industrial_mps_sj_v2.py").read_text()
    new = Path(run.__file__).read_text()
    start, end = "            started = time.perf_counter()", "            payload ="
    assert old[old.index(start):old.index(end)] == new[new.index(start):new.index(end)]
    tree = ast.parse(new)
    execute = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "execute")
    calls = [ast.unparse(n.func) for n in ast.walk(execute) if isinstance(n, ast.Call)]
    assert "original.make_adapter" in calls and "original.encode_jobs" in calls
    assert "original.ExistingLock" in calls
    assert not any(isinstance(n, ast.Assign) and any(isinstance(t, ast.Attribute) and
        isinstance(t.value, ast.Name) and t.value.id == "original" for t in n.targets) for n in ast.walk(tree))


def test_no_resume_or_old_namespace_write_path_and_execution_admission_precedes_model():
    text = Path(run.__file__).read_text()
    assert 'parser.add_argument("--resume"' not in text
    assert run.DELTA != run.original.DELTA and run.PARENT_PATH.startswith(run.original.BASE)
    execute = text[text.index("def execute("):text.index("def main(")]
    assert execute.index("check_execution_admission(") < execute.index("original.make_adapter(")
    assert "Existing stage is immutable; automatic retry forbidden" in execute


def test_source_binding_guard_on_new_module_without_model_imports():
    import sys
    # The new guard is exercised; the old source suite is not rerun.
    run.check_sources(ROOT, {"source_pins": {path: run.frozen.record(ROOT, path)["sha256"] for path in run.SOURCES}})
    assert not any(name in sys.modules for name in ("torch", "transformers", "sentence_transformers"))


@pytest.mark.parametrize("damage", [None, "inode", "target", "mtime_ns", "ctime_ns", "device", "resolved_path"])
def test_live_asset_metadata_is_checked_without_rehashing_weights(tmp_path, monkeypatch, damage):
    old = manifest_fixture()
    _, binding, _, _, operational = binding_fixture(tmp_path, old)
    path = tmp_path / binding["current_lock"]["path"]
    path.chmod(0o644)
    current = path.stat()
    monkeypatch.setattr(run, "current_boot_epoch", lambda: run.BOOT_EPOCH)
    monkeypatch.setattr(run, "LOCK_INODE", current.st_ino)
    monkeypatch.setattr(run, "NEW_DEVICE", current.st_dev)
    binding.update(lock_uid=current.st_uid, lock_gid=current.st_gid)
    calls = []
    def asset_chain(root, logical):
        calls.append(logical)
        value = next(a for a in operational["assets"]["J"]["assets"].values() if a["logical_path"] == logical)
        result = {key: copy.deepcopy(value[key]) for key in ("logical_path", "symlink_chain", "resolved_path")}
        if damage == "resolved_path":
            result[damage] = "changed"
        elif damage:
            result["symlink_chain"][0][damage] = "changed"
        return result
    monkeypatch.setattr(run.original, "asset_chain", asset_chain)
    monkeypatch.setattr(run.original, "stream_record", lambda *args: pytest.fail("Weight hash replay outside full guard"))
    if damage:
        with pytest.raises(ValueError, match="live_asset_chain_changed"):
            run.check_live_metadata(tmp_path, operational, binding)
    else:
        run.check_live_metadata(tmp_path, operational, binding)
        assert len(calls) == 57


@pytest.mark.parametrize("damage", [None, "contract", "review_contract", "root_review", "raw_stale"])
def test_source_acceptance_pins_exact_contract_and_independent_review(tmp_path, damage):
    pins = {name: "a" * 64 for name in run.SOURCES}
    contract = pin(tmp_path, "proof/source-contract.json", {"protocol": run.VERSION,
        "scope": "SOURCE_ONLY_NO_ACTUAL_PREPARE_OR_EXECUTION", "source_pins": pins})
    review = {"status": "PASS_INDEPENDENT_INDUSTRIAL_REBOOT_SOURCE", "blocking_findings": [],
        "source_pins": pins, "source_contract_sha256": contract["sha256"]}
    if damage == "review_contract":
        review["source_contract_sha256"] = "x" * 64
    review_ref = pin(tmp_path, "proof/review.json", review)
    root = {"status": "PASS_ROOT_INDUSTRIAL_REBOOT_SOURCE", "source_pins": pins,
        "source_contract_sha256": contract["sha256"], "independent_review_sha256": review_ref["sha256"]}
    if damage == "root_review":
        root["independent_review_sha256"] = "x" * 64
    refs = {"source_contract": contract, "independent_source_review": review_ref,
            "root_source_acceptance": pin(tmp_path, "proof/root.json", root),
            "raw": pin(tmp_path, "proof/raw-source", b"synthetic source")}
    if damage == "contract":
        refs["source_contract"] = pin(tmp_path, "proof/source-contract.json", {"protocol": run.VERSION,
            "scope": "SOURCE_ONLY_NO_ACTUAL_PREPARE_OR_EXECUTION", "source_pins": {}})
    elif damage == "raw_stale":
        (tmp_path / "proof/raw-source").write_bytes(b"changed")
    recipe = {"source_pins": pins, "accepted_evidence": refs}
    if damage:
        with pytest.raises(ValueError):
            run.source_admissions(tmp_path, recipe)
    else:
        run.source_admissions(tmp_path, recipe)

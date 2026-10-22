"""New synthetic-only regressions for the explicit FLEURS document bridge."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rag_benchmark import fleurs_pilot_data as pilot
from rag_benchmark.models import stable_hash
from rag_benchmark.multimodal import encoding_cache_identity, load_dataset


def put(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class FakeAdapter:
    def __init__(self, channel, config):
        self.config = copy.deepcopy(config)
        self.dimension = 1024 if channel == "G" else 768
        self.identity = stable_hash({"fake_channel": channel, "config": config})
        self._model = None
        self.embedder = SimpleNamespace(config=self.config, _model=None)

    def encode(self, *_args, **_kwargs):
        raise AssertionError("Model inference must never occur in the data bridge")


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    monkeypatch.setattr(pilot, "_adapter", FakeAdapter)
    source = root / "source"
    source.mkdir()
    (source / "media").mkdir()
    corpus = []
    for i in range(743):
        media = f"media/{i}.wav"
        (source / media).write_bytes(b"synthetic-never-decoded")
        corpus.append({"id": f"d{i:03d}", "text": "synthetic source-only ASR", "media": {"audio": media}})
    queries = [{"id": f"q{i:03d}", "text": f"synthetic literal {i}", "media": {}} for i in range(329)]
    qrels = [{"query_id": f"q{i % 329:03d}", "corpus_id": row["id"], "relevance": 1}
             for i, row in enumerate(corpus)]
    put(source / "dataset.json", {"id": "synthetic_fleurs", "track": "speech", "revision": "fake-revision",
        "text_provenance": {"source": "whisper_asr", "query_independent": True, "gold_fields_used": False}})
    jsonl(source / "corpus.jsonl", corpus)
    jsonl(source / "queries.jsonl", queries)
    jsonl(source / "qrels.jsonl", qrels)
    (source / "assets.jsonl").write_text("", encoding="utf-8")
    dataset = load_dataset(source)
    lines = (source / "queries.jsonl").read_bytes().splitlines(keepends=True)
    draft_rows, choices = [], []
    for i, qid in enumerate(pilot.QUERY_IDS):
        parent = queries[i]["id"]
        common = {"query_id": qid, "parent_query_id": parent, "source_line_1based": i + 1,
            "source_refhash": pilot.digest(lines[i]), "candidate_positive_group": {"parent_group_id": parent,
            "source_declared_positive_count": len(dataset.qrels[parent]),
            "status": "source_group_reference_only_not_new_qrels_validation"}}
        choices.append(common)
        draft_rows.append({**common, "query_text": f"synthetic paraphrase {i}"})
    selected_path = root / "selected.json"
    put(selected_path, {"selected_sources": choices})
    drafts_path = root / "drafts.json"
    put(drafts_path, {"protocol": "fleurs-paraphrase-pilot-agent-drafts-v2", "query_count": 25,
        "human_validated": False, "human_validation_complete": False, "queries": draft_rows,
        "selected_sources_ref": pilot.ref(selected_path), "source_ref": pilot.ref(source / "queries.jsonl")})
    review_path = root / "review.json"
    put(review_path, {"status": "PASS_AGENT_DELTA_REVIEW", "summary": {"agent_reviewed_queries": 25,
        "remaining_identified_semantic_issues": 0, "human_validated": False,
        "human_validation_complete": False, "qrels_full_validation": False},
        "refs": {"selected-sources.json": pilot.ref(selected_path), "queries-draft-v2.json": pilot.ref(drafts_path)}})
    cache = root / "cache"
    sets, specs, channels = [], {}, {}
    ordered_ids = [row["id"] for row in corpus]
    for channel in pilot.CHANNELS:
        config = {"batch_size": 8 if channel in {"G", "E"} else 1, "dtype": "bfloat16",
                  "device": "mps", "revision": "fake", "normalization": "unit"}
        adapter = FakeAdapter(channel, config)
        spec = {"identity": adapter.identity, "dimension": adapter.dimension, "config": config}
        if channel in {"G", "E"}:
            spec = {"identity": adapter.identity, "config": {}, "embedder": {"config": config},
                    "dimension": adapter.dimension}
        specs[channel] = spec
        key = stable_hash({"encoding": encoding_cache_identity(dataset, adapter, "document"),
                           "text_only": channel in {"G", "E"}})
        channels[channel] = {"document_encoding": {"identity": key}}
        directory = cache / "vectors" / key
        directory.mkdir(parents=True)
        blocks = []
        for begin in range(0, 743, 128):
            end = min(begin + 128, 743)
            array = np.zeros((end - begin, adapter.dimension), dtype=np.float32)
            array[:, 0] = 1
            path = directory / f"{begin:09d}-{end:09d}.npy"
            np.save(path, array, allow_pickle=False)
            usage = {"rows": end - begin, "reported": channel in {"N", "J"}, "items": []}
            if channel in {"N", "J"}:
                usage.update(context_limit=8192, dtype="bfloat16",
                    items=[{"id": identifier, "expanded_tokens": 24} for identifier in ordered_ids[begin:end]])
            put(path.with_suffix(".usage.json"), usage)
            blocks.append({**pilot.ref(path), "size": path.stat().st_size,
                "shape": list(array.shape), "item_ids": ordered_ids[begin:end]})
        sets.append({"channel": channel, "role": "document", "kind": "dense", "set_id": key,
            "dimension": adapter.dimension, "item_ids": ordered_ids, "blocks": blocks,
            "provenance": {"adapter_spec": spec, "frozen_dataset_identity": dataset.identity,
                "cache_identity": key, "historical_file_checksum_attested": False}})
    report_path = root / "report.json"
    put(report_path, {"dataset": {"identity": dataset.identity, "revision": dataset.manifest["revision"]},
                     "adapter_specs": {"channels": specs}, "channels": channels})
    inventory_path = root / "inventory.json"
    collection = {"dataset_root": str(source), "frozen_identity": dataset.identity, "sets": sets,
        "source_files": [{**pilot.ref(source / name), "filename": name, "size": (source / name).stat().st_size}
            for name in ("dataset.json", "corpus.jsonl", "queries.jsonl", "qrels.jsonl", "assets.jsonl")],
        "media": [{**pilot.ref(source / item["media"]["audio"]),
                   "size": (source / item["media"]["audio"]).stat().st_size} for item in corpus]}
    put(inventory_path, {"collections": [collection]})
    config = {"protocol": "fleurs-paraphrase-pilot-data-config-v1", "repository_root": str(root),
        "source_root": str(source), "source_identity": dataset.identity, "cache_root": str(cache),
        "source_report": pilot.ref(report_path), "source_cache_inventory": pilot.ref(inventory_path),
        "drafts": pilot.ref(drafts_path), "meaning_review": pilot.ref(review_path),
        "source_code": pilot.code_refs(), "output_directory": str(root / "prepared")}
    config_path = root / "config.json"
    put(config_path, config)
    return SimpleNamespace(root=root, source=source, cache=cache, config=config, config_path=config_path,
        inventory=inventory_path, drafts=drafts_path, review=review_path, output=root / "prepared")


def prepare(bundle):
    put(bundle.config_path, bundle.config)
    return pilot.prepare_verified_pilot(bundle.config_path, pilot.ref(bundle.config_path)["sha256"], bundle.output)


def refresh_inventory(bundle, value):
    put(bundle.inventory, value)
    bundle.config["source_cache_inventory"] = pilot.ref(bundle.inventory)


def replace_drafts(bundle, mutate):
    draft = json.loads(bundle.drafts.read_text())
    mutate(draft)
    put(bundle.drafts, draft)
    bundle.config["drafts"] = pilot.ref(bundle.drafts)
    review = json.loads(bundle.review.read_text())
    review["refs"]["queries-draft-v2.json"] = pilot.ref(bundle.drafts)
    put(bundle.review, review)
    bundle.config["meaning_review"] = pilot.ref(bundle.review)


def test_roundtrip_new_namespace_readonly_arrays_and_no_inference(bundle):
    before = {str(p): pilot.ref(p)["sha256"] for p in bundle.cache.rglob("*") if p.is_file()}
    result = prepare(bundle)
    ready = pilot.load_verified_pilot(result["path"], expected_sha256=result["sha256"])
    assert ready.dataset.identity != ready.source_dataset.identity
    assert ready.dataset.root == bundle.output
    assert [row["id"] for row in ready.dataset.queries] == pilot.QUERY_IDS
    assert ready.dataset.corpus == ready.source_dataset.corpus
    assert ready.dataset.manifest["pilot_scope"] == pilot.SCOPE
    assert all(not values.flags.writeable and values.shape[0] == 743 for values in ready.document_vectors.values())
    assert ready.dataset.qrels["pp01"] == ready.source_dataset.qrels["q000"]
    model_item = ready.dataset.model_item(ready.dataset.corpus[0])
    assert Path(model_item["media"]["audio"]).is_relative_to(bundle.source)
    assert set(ready.dataset.model_item(ready.dataset.queries[0])) == {"id", "text", "media"}
    assert all((p.stat().st_mode & 0o777) == 0o600 for p in bundle.output.iterdir())
    assert before == {str(p): pilot.ref(p)["sha256"] for p in bundle.cache.rglob("*") if p.is_file()}


def test_existing_output_is_not_overwritten(bundle):
    prepare(bundle)
    with pytest.raises(pilot.PilotDataError, match="output_namespace_exists"):
        prepare(bundle)


@pytest.mark.parametrize("target", ["corpus.jsonl", "queries.jsonl", "qrels.jsonl", "media/0.wav"])
def test_changed_source_rejected_before_new_namespace(bundle, target):
    path = bundle.source / target
    path.write_bytes(path.read_bytes() + (b"x" if target.endswith("wav") else b"\n"))
    with pytest.raises(pilot.PilotDataError):
        prepare(bundle)
    assert not bundle.output.exists()


def test_missing_block_fails_without_regeneration(bundle):
    block = next(bundle.cache.rglob("*.npy"))
    block.unlink()
    with pytest.raises(pilot.PilotDataError, match="cache_block_inventory"):
        prepare(bundle)
    assert not block.exists()


@pytest.mark.parametrize("damage", ["reordered_ids", "wrong_dimension", "duplicate_set", "wrong_prompt_identity"])
def test_source_inventory_mapping_guards(bundle, damage):
    value = json.loads(bundle.inventory.read_text())
    sets = value["collections"][0]["sets"]
    if damage == "reordered_ids":
        sets[0]["blocks"][0]["item_ids"].reverse()
    elif damage == "wrong_dimension":
        sets[0]["dimension"] = 768
    elif damage == "duplicate_set":
        sets.append(copy.deepcopy(sets[0]))
    else:
        sets[0]["provenance"]["adapter_spec"]["identity"] = "0" * 64
    refresh_inventory(bundle, value)
    with pytest.raises(pilot.PilotDataError):
        prepare(bundle)


@pytest.mark.parametrize("damage", ["nan", "zero", "float64"])
def test_pinned_but_invalid_array_rejected(bundle, damage):
    value = json.loads(bundle.inventory.read_text())
    block = value["collections"][0]["sets"][0]["blocks"][0]
    path = Path(block["path"])
    array = np.load(path, allow_pickle=False)
    if damage == "float64":
        array = array.astype(np.float64)
    else:
        array[0] = np.nan if damage == "nan" else 0
    np.save(path, array, allow_pickle=False)
    block.update(pilot.ref(path), size=path.stat().st_size)
    refresh_inventory(bundle, value)
    with pytest.raises(pilot.PilotDataError, match="cache_array_"):
        prepare(bundle)


@pytest.mark.parametrize("damage", ["duplicate_parent", "old_query_id", "fake_human", "literal_text", "wrong_qrel_count"])
def test_query_transfer_scope_guards(bundle, damage):
    def mutate(draft):
        rows = draft["queries"]
        if damage == "duplicate_parent":
            rows[1]["parent_query_id"] = rows[0]["parent_query_id"]
        elif damage == "old_query_id":
            rows[0]["query_id"] = "q000"
        elif damage == "fake_human":
            draft["human_validated"] = True
        elif damage == "literal_text":
            rows[0]["query_text"] = "synthetic literal 0"
        else:
            rows[0]["candidate_positive_group"]["source_declared_positive_count"] += 1
    replace_drafts(bundle, mutate)
    with pytest.raises(pilot.PilotDataError):
        prepare(bundle)


def test_native_context_overflow_rejected(bundle):
    inventory = json.loads(bundle.inventory.read_text())
    row = inventory["collections"][0]["sets"][2]["blocks"][0]
    path = Path(row["path"]).with_suffix(".usage.json")
    usage = json.loads(path.read_text())
    usage["items"][0]["expanded_tokens"] = 8193
    put(path, usage)
    with pytest.raises(pilot.PilotDataError, match="native_context_overflow"):
        prepare(bundle)


def test_load_detects_later_cache_changes(bundle):
    result = prepare(bundle)
    path = next(bundle.cache.rglob("*.npy"))
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(pilot.PilotDataError, match="reference_digest_mismatch"):
        pilot.load_verified_pilot(result["path"], expected_sha256=result["sha256"])


def test_query_adapter_only_batch_override_allowed(bundle):
    result = prepare(bundle)
    ready = pilot.load_verified_pilot(result["path"], expected_sha256=result["sha256"])
    config = copy.deepcopy(ready.query_adapter_configs["E"])
    config["batch_size"] = 1
    ready.validate_query_adapter("E", FakeAdapter("E", config))
    config["normalization"] = "changed"
    with pytest.raises(pilot.PilotDataError, match="query_adapter_config_changed"):
        ready.validate_query_adapter("E", FakeAdapter("E", config))


def test_candidate_mutation_cannot_leak_query_or_gold(bundle):
    result = prepare(bundle)
    ready = pilot.load_verified_pilot(result["path"], expected_sha256=result["sha256"])
    candidate = copy.deepcopy(ready.dataset.corpus[0])
    candidate["text"] = ready.dataset.queries[0]["text"]
    with pytest.raises(pilot.PilotDataError, match="pilot_candidate_changed"):
        ready.dataset.model_item(candidate)

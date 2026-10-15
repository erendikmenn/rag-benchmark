"""Uniform source-only LaSCo Haiku captions. Explicit API execution; no models or DB."""

from __future__ import annotations

import argparse
import ast
import asyncio
import base64
import copy
import fcntl
import hashlib
import io
import json
import math
import os
import re
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType

ROOT = Path(__file__).absolute().parents[1]
PROTOCOL = "lasco-haiku55-source-caption-en-v2"
MODEL, ROUTE, PROVIDER = "anthropic/claude-haiku-5.5", "google-vertex/global", "Google"
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
BOUND_USD = (8192 * 0.1 + 512 * 0.5) / 1_000_000
MAX_ATTEMPTS, MAX_RAW_BYTES = 3, 2 * 1024**2


def require(value, reason):
    if not value:
        raise ValueError(reason)


def seal(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def safe(path, root=ROOT):
    path = Path(path).absolute()
    require(not any(p.is_symlink() for p in (path, *path.parents)), "symlink_path")
    path = path.resolve()
    require(path.is_relative_to(Path(root).resolve()), "path_escape")
    return path


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path, value, *, fresh=True):
    path = Path(path)
    temporary = path if fresh else path.with_name(path.name + ".partial")
    with temporary.open("x") as handle:
        os.chmod(temporary, 0o600)
        json.dump(value, handle, ensure_ascii=False, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    if not fresh:
        temporary.replace(path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def system_prompt():
    source = (ROOT / "src/rag_benchmark/multimodal_generation.py").read_text()
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "_CAPTION_SYSTEM_TEMPLATE" for t in node.targets
        ):
            return ast.literal_eval(node.value).format(language="English")
    raise ValueError("caption_prompt_missing")


def policy():
    return {
        "model": MODEL,
        "route": ROUTE,
        "provider": PROVIDER,
        "system_prompt": system_prompt(),
        "reasoning_effort": "none",
        "max_tokens": 512,
        "jpeg_quality": 95,
        "image_max_side": 1120,
        "prompt_token_bound": 8192,
        "max_price_prompt": 0.1,
        "max_price_completion": 0.5,
    }


def prepare(
    dataset, output, destination, *, prior_spec, prior_spec_sha256, prior_result, prior_result_sha256
):
    from rag_benchmark.lasco_data import REVISION, ANNOTATION_SHA256, GALLERY_SHA256, build_rows, pinned_json

    require(
        dataset.manifest["id"] == "lasco-validation" and dataset.manifest["revision"] == REVISION,
        "official_native_source",
    )
    require(len(dataset.corpus) == 39826 and len(dataset.queries) == 30037, "full_source_counts")
    sources = {x["file"]: x for x in dataset.manifest["sources"]}
    require(
        sources["lasco_val.json"]["sha256"] == ANNOTATION_SHA256
        and sources["lasco_val_corpus.json"]["sha256"] == GALLERY_SHA256,
        "official_annotations",
    )
    raw = ROOT / "data/multimodal/raw/lasco"
    annotations, _ = pinned_json(raw / "lasco_val.json", ANNOTATION_SHA256)
    gallery, _ = pinned_json(raw / "lasco_val_corpus.json", GALLERY_SHA256)
    official_corpus, official_queries, labels, _ = build_rows(annotations, gallery)
    require(
        dataset.corpus == official_corpus
        and dataset.queries == official_queries
        and dataset.qrels == {x["query_id"]: {x["corpus_id"]: x["relevance"]} for x in labels},
        "exact_official_source_rows",
    )
    assets = {
        x["path"]: x["sha256"]
        for x in map(json.loads, (dataset.root / "assets.jsonl").read_text().splitlines())
    }
    require(len(assets) == 39826, "full_source_assets")
    by = {r["id"]: r for r in dataset.corpus}
    require(len(by) == 39826, "unique_source_ids")
    for row in dataset.corpus:
        require(
            not row.get("text") and set(row["media"]) == {"image"} and row["media"]["image"] in assets,
            "source_pixels_only",
        )
    for query in dataset.queries:
        ref = str(query["metadata"]["reference_id"])
        require(
            ref in by and query["media"] == by[ref]["media"] and ref in dataset.excluded_ids(query),
            "reference_image_exclusion",
        )
        require(
            len(dataset.qrels[query["id"]]) == 1 and ref not in dataset.qrels[query["id"]],
            "distinct_gold_only_labels",
        )
    output, destination = safe(output), safe(destination)
    require(
        output.is_relative_to(ROOT / "work") and not output.exists() and not destination.exists(),
        "fresh_paths",
    )
    files = {
        str(p): sha(p)
        for p in [
            raw / "lasco_val.json",
            raw / "lasco_val_corpus.json",
            Path(__file__).absolute(),
            ROOT / "src/rag_benchmark/multimodal_generation.py",
            ROOT / "src/rag_benchmark/multimodal_data.py",
            ROOT / "src/rag_benchmark/multimodal.py",
            ROOT / "src/rag_benchmark/lasco_data.py",
            *[
                dataset.root / n
                for n in ["dataset.json", "corpus.jsonl", "queries.jsonl", "qrels.jsonl", "assets.jsonl"]
            ],
        ]
    }
    prior = prior_proof(prior_spec, prior_spec_sha256, prior_result, prior_result_sha256, dataset.identity)
    return {
        "prior_run": prior,
        "fresh_full_generation": True,
        "local_or_v1_caption_reuse": False,
        "protocol": PROTOCOL,
        "source": str(dataset.root),
        "source_identity": dataset.identity,
        "counts": {"corpus": 39826, "queries": 30037},
        "output": str(output),
        "destination": str(destination),
        "files": files,
        "assets": assets,
        "policy": policy(),
        "budget_usd": 15.0,
        "request_bound_usd": BOUND_USD,
        "concurrency": 100,
        "initial_max_rps": 25,
        "max_attempts": MAX_ATTEMPTS,
        "benchmark_runs": 0,
        "database_writes": False,
    }


def prior_proof(spec_path, spec_sha, result_path, result_sha, source_identity):
    spec_path, result_path = safe(spec_path), safe(result_path)
    require(sha(spec_path) == spec_sha and sha(result_path) == result_sha, "prior_external_sha")
    old, result = read(spec_path), read(result_path)
    require(
        old["protocol"] == "lasco-haiku55-source-caption-en-v1" and old["policy"] == policy(), "prior_policy"
    )
    require(
        result["producer_closed"] is True
        and result["requests_closed"] is True
        and result["spec_sha256"] == spec_sha
        and result["source_identity"] == source_identity
        and result["caption_rows"] == 342
        and result["failed"] == 1
        and result["attempts"] == 343
        and result["reserved_usd"] == 0
        and result["status"] == "budget_unknown"
        and abs(result["charged_usd"] - 0.0530309) < 1e-12,
        "prior_closed_debit",
    )
    output, session = safe(old["output"]), safe(result["session"])
    require(result_path == output / "result.json" and session.parent == output / "sessions", "prior_paths")
    require(
        read(session / "result.json") == result
        and read(output / "status.json") == result
        and read(output / "pins.json") == old
        and read(output / "claim.json")["spec_sha256"] == spec_sha,
        "prior_chain",
    )
    paths = [
        spec_path,
        result_path,
        session / "result.json",
        session / "claim.json",
        output / "status.json",
        output / "pins.json",
        output / "claim.json",
    ]
    return {
        "files": {str(p): sha(p) for p in paths},
        "prior_spec_sha256": spec_sha,
        "prior_result_sha256": result_sha,
        "prior_debit_usd": 0.0530309,
        "prior_known_cost_usd": result["known_cost_usd"],
    }


def verify(spec, dataset):
    prior = spec["prior_run"]
    require(
        prior["prior_debit_usd"] == 0.0530309 and all(sha(p) == h for p, h in prior["files"].items()),
        "prior_debit_binding",
    )
    require(
        spec["fresh_full_generation"] is True and spec["local_or_v1_caption_reuse"] is False,
        "fresh_uniform_job",
    )
    require(spec["protocol"] == PROTOCOL and spec["policy"] == policy(), "fixed_api_policy")
    require(
        spec["budget_usd"] == 15
        and spec["request_bound_usd"] == BOUND_USD
        and spec["concurrency"] == 100
        and spec["initial_max_rps"] == 25
        and spec["max_attempts"] == MAX_ATTEMPTS,
        "fixed_budget_limits",
    )
    require(
        spec["source_identity"] == dataset.identity
        and len(dataset.corpus) == spec["counts"]["corpus"]
        and len(dataset.queries) == spec["counts"]["queries"],
        "frozen_source_identity",
    )
    require(all(sha(p) == digest for p, digest in spec["files"].items()), "source_or_code_changed")


def make_payload(spec, dataset, row):
    from PIL import Image

    require(not row.get("text") and set(row["media"]) == {"image"}, "source_pixels_only")
    relative = row["media"]["image"]
    path = safe(dataset.root / relative, dataset.root)
    raw = path.read_bytes()
    asset = hashlib.sha256(raw).hexdigest()
    require(asset == spec["assets"][relative], "frozen_asset_changed")
    with Image.open(io.BytesIO(raw)) as image:
        image = image.convert("RGB")
        image.thumbnail((1120, 1120))
        wire = io.BytesIO()
        image.save(wire, format="JPEG", quality=95)
    encoded = wire.getvalue()
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": spec["policy"]["system_prompt"]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(encoded).decode()},
                    }
                ],
            },
        ],
        "reasoning": {"effort": "none"},
        "max_tokens": 512,
        "stream": True,
        "stream_options": {"include_usage": True},
        "provider": {
            "only": [ROUTE],
            "allow_fallbacks": False,
            "require_parameters": True,
            "max_price": {"prompt": 0.1, "completion": 0.5},
        },
    }
    return payload, asset


def parse_response(response):
    require(response["status_code"] == 200 and not response.get("error_type"), "http_or_transport_failure")
    text, reasoning, finish, usage, provider, model = [], [], None, None, None, None
    require(response.get("done") is True, "incomplete_stream")
    for event in response["events"]:
        require(
            not event.get("error") and not event.get("truncated") and not event.get("tokens_dropped", 0),
            "stream_error",
        )
        if event.get("provider"):
            require(provider in (None, event["provider"]), "provider_changed")
            provider = event["provider"]
        if event.get("model"):
            require(model in (None, event["model"]), "model_changed")
            model = event["model"]
        if event.get("usage") is not None:
            usage = event["usage"]
        choices = event.get("choices", [])
        require(len(choices) <= 1, "single_choice")
        for choice in choices:
            require(choice.get("index", 0) == 0 and not choice.get("truncated"), "single_untruncated_choice")
            delta = choice.get("delta", {})
            if delta.get("content"):
                require(isinstance(delta["content"], str), "content_type")
                text.append(delta["content"])
            reasoning.extend(
                [delta[k] for k in ("reasoning", "reasoning_content", "reasoning_details") if delta.get(k)]
            )
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
    require(provider == PROVIDER and model == MODEL, "exact_model_provider")
    require(finish == "stop" and not reasoning, "normal_stop_no_reasoning")
    require(
        isinstance(usage, dict)
        and type(usage.get("prompt_tokens")) is int
        and 0 < usage["prompt_tokens"] <= 8192
        and type(usage.get("completion_tokens")) is int
        and 0 < usage["completion_tokens"] <= 512,
        "usage_bounds",
    )
    require(usage.get("completion_tokens_details", {}).get("reasoning_tokens") == 0, "reasoning_usage")
    cost = usage.get("cost")
    require(
        type(cost) in (int, float) and math.isfinite(cost) and 0 <= cost <= BOUND_USD, "known_bounded_cost"
    )
    caption = "".join(text)
    require(bool(caption.strip()), "empty_caption")
    last, length = None, 0
    for word in re.findall(r"\w+", caption.casefold()):
        length = length + 1 if word == last else 1
        require(length <= 20, "pathological_repetition")
        last = word
    return caption, usage


def transient_code(response):
    """Only explicit provider rejection before any output permits another attempt."""
    if response.get("error_type"):
        return None
    events = response.get("events", [])
    for event in events:
        for choice in event.get("choices", []):
            delta = choice.get("delta", {})
            if any(delta.get(k) for k in ("content", "reasoning", "reasoning_content", "reasoning_details")):
                return None
    if response.get("status_code") in {429, 502, 503, 504}:
        return response["status_code"]
    if response.get("status_code") == 200:
        for event in events:
            error = event.get("error")
            if isinstance(error, dict):
                code = error.get("code")
                if str(code) in {"429", "502", "503", "504"}:
                    return int(code)
    return None


def retryable(response):
    # A transport error cannot turn an explicit terminal content rejection into a retry.
    if any(
        c.get("finish_reason") in {"length", "refusal", "content_filter"}
        or c.get("delta", {}).get("refusal")
        or c.get("message", {}).get("refusal")
        for e in response.get("events", [])
        for c in e.get("choices", [])
    ):
        return False
    unfinished_eof = (
        response.get("status_code") == 200
        and not response.get("done")
        and not response.get("error_type")
        and not any(
            e.get("error") or any(c.get("finish_reason") for c in e.get("choices", []))
            for e in response.get("events", [])
        )
    )
    return (
        unfinished_eof
        or transient_code(response) is not None
        or response.get("error_type")
        in {
            "ReadError",
            "ReadTimeout",
            "ConnectTimeout",
            "ConnectError",
            "RemoteProtocolError",
            "ProtocolError",
            "TimeoutError",
            "PoolTimeout",
            "WriteError",
            "WriteTimeout",
            "CloseError",
        }
    )


def charge(response):
    usages = [e["usage"] for e in response.get("events", []) if isinstance(e.get("usage"), dict)]
    cost = usages[-1].get("cost") if usages else None
    if type(cost) in (int, float) and math.isfinite(cost) and cost >= 0:
        bill = {"charged_usd": float(cost), "cost_known": True}
        if cost > BOUND_USD:
            bill["bound_violation"] = True
        return bill
    if retryable(response):
        return {"charged_usd": BOUND_USD, "cost_known": False, "bounded_uncertain_attempt": True}
    return {"charged_usd": BOUND_USD, "cost_known": False, "budget_unknown": True}


class RowRejection(ValueError):
    pass


def strict_caption(raw):
    try:
        return parse_response(raw)
    except ValueError as error:
        # These are terminal row defects, never candidates for content retry.
        if str(error) in {"normal_stop_no_reasoning", "empty_caption", "pathological_repetition"}:
            # Reasoning is a policy violation, not an isolated content defect.
            require(
                not any(
                    any(
                        c.get("delta", {}).get(k)
                        for k in ("reasoning", "reasoning_content", "reasoning_details")
                    )
                    for e in raw.get("events", [])
                    for c in e.get("choices", [])
                ),
                "reasoning_policy",
            )
            raise RowRejection(str(error)) from error
        raise


class Gate:
    def __init__(self):
        self.active, self.limit, self.rps, self.next, self.cooldown, self.successes = (
            0,
            100,
            25.0,
            0.0,
            0.0,
            0,
        )
        self.lock = asyncio.Lock()

    async def enter(self, stopped):
        while not stopped():
            async with self.lock:
                now = time.monotonic()
                if self.active < self.limit and now >= max(self.next, self.cooldown):
                    self.active += 1
                    self.next = now + 1 / self.rps
                    return True
                delay = min(0.1, max(0.005, max(self.next, self.cooldown) - now))
            await asyncio.sleep(delay)
        return False

    def leave(self):
        self.active -= 1

    def penalize(self):
        self.limit, self.rps = max(1, self.limit // 2), max(1.0, self.rps / 2)
        self.cooldown = max(self.cooldown, time.monotonic() + 2)
        self.successes = 0

    def good(self):
        self.successes += 1
        if self.successes >= 100:
            self.limit, self.rps = min(100, self.limit + 5), min(25.0, self.rps + 1)
            self.successes = 0


def row_claim(spec, row, index, asset, payload, spec_sha256):
    return {
        "protocol": PROTOCOL,
        "source_index": index,
        "source_id": row["id"],
        "asset_sha256": asset,
        "request_sha256": seal(payload),
        "spec_sha256": spec_sha256,
    }


def audit_row(
    spec, dataset, row, index, output, spec_sha256, payload_builder=make_payload, *, allow_pending=False
):
    payload, asset = payload_builder(spec, dataset, row)
    claim = row_claim(spec, row, index, asset, payload, spec_sha256)
    directory = Path(output) / "rows" / f"{index:09d}"
    require(
        read(directory / "claim.json") == claim and read(directory / "request.json") == payload,
        "row_claim_request",
    )
    require(not (directory / "failed.json").exists(), "failed_row_requires_review")
    accepted = read(directory / "accepted.json") if (directory / "accepted.json").exists() else None
    attempts = sorted(directory.glob("attempt-*"))
    require(0 <= len(attempts) <= MAX_ATTEMPTS, "attempt_bound")
    spent = 0.0
    for ordinal, attempt in enumerate(attempts):
        require(attempt.name == f"attempt-{ordinal}", "attempt_order")
        reserve, raw, settled = [
            read(attempt / n) for n in ("reservation.json", "raw-response.json", "settled.json")
        ]
        require(
            reserve == {"request_sha256": claim["request_sha256"], "bound_usd": BOUND_USD},
            "attempt_reservation",
        )
        expected = {**charge(raw), "raw_response_sha256": seal(raw)}
        require(settled == expected and not settled.get("budget_unknown"), "billing_receipt")
        spent += settled["charged_usd"]
        if ordinal < len(attempts) - 1 or accepted is None:
            require(retryable(raw), "retry_policy")
    if accepted is None:
        require(allow_pending and len(attempts) < MAX_ATTEMPTS, "unaccepted_row_requires_review")
        return None, spent
    require(attempts, "accepted_without_attempt")
    text, usage = parse_response(raw)
    require(
        accepted
        == {
            **claim,
            "text": text,
            "attempt": len(attempts) - 1,
            "raw_response_sha256": seal(raw),
            "usage": usage,
            "provider": PROVIDER,
        },
        "accepted_binding",
    )
    require(sha(dataset.root / row["media"]["image"]) == asset, "asset_changed")
    return accepted, spent


def audit_rows(spec, dataset, output, *, spec_sha256, payload_builder=make_payload):
    """Pure filesystem audit; returns accepted rows in exact corpus order, never calls transport."""
    verify(spec, dataset)
    records, total = [], spec["prior_run"]["prior_debit_usd"]
    require(read(Path(output) / "pins.json") == spec, "output_spec")
    require(
        {p.name for p in (Path(output) / "rows").iterdir()}
        == {f"{i:09d}" for i in range(len(dataset.corpus))},
        "complete_row_inventory",
    )
    for index, row in enumerate(dataset.corpus):
        accepted, spent = audit_row(spec, dataset, row, index, output, spec_sha256, payload_builder)
        records.append(accepted)
        total += spent
    require(total <= 15, "total_budget")
    verify(spec, dataset)
    return records


async def run(
    spec, dataset, transport, *, spec_sha256, resume=False, payload_builder=make_payload, gate=None
):
    """Injected async transport; all attempted responses persist before strict validation."""
    verify(spec, dataset)
    frozen = seal(spec)
    controls = {k: copy.deepcopy(v) for k, v in spec.items() if k not in {"assets", "files"}}
    frozen_controls = seal(controls)
    frozen_assets = MappingProxyType(dict(spec["assets"]))

    def control_guard():
        require(
            seal({k: v for k, v in spec.items() if k not in {"assets", "files"}}) == frozen_controls,
            "policy_mutation",
        )

    output = safe(spec["output"])
    if not output.exists():
        require(not resume, "resume_missing_output")
        output.mkdir(mode=0o700)
    else:
        require(resume, "explicit_resume_required")
    lock = os.open(output / ".lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if resume:
            require(read(output / "pins.json") == spec, "resume_pins")
            require(
                read(output / "status.json")["status"] in {"interrupted", "budget_stop"},
                "failed_state_no_retry",
            )
        else:
            write(output / "pins.json", spec)
            write(output / "claim.json", {"spec_sha256": spec_sha256, "owner_pid": os.getpid()})
            (output / "rows").mkdir(mode=0o700)
        require(read(output / "claim.json")["spec_sha256"] == spec_sha256, "resume_spec_raw_sha")
        sessions = output / "sessions"
        sessions.mkdir(mode=0o700, exist_ok=True)
        session = sessions / f"session-{len(list(sessions.glob('session-*'))):06d}"
        session.mkdir(mode=0o700)
        write(
            session / "claim.json", {"spec_sha256": spec_sha256, "resume": resume, "owner_pid": os.getpid()}
        )
        started = time.monotonic()
        ledger = {
            "start_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": 0.0,
            "failed_indices": [],
            "failed_reasons": {},
            "prior_debit_usd": spec["prior_run"]["prior_debit_usd"],
            "prior_result_sha256": spec["prior_run"]["prior_result_sha256"],
            "prior_spec_sha256": spec["prior_run"]["prior_spec_sha256"],
            "new_job_charged_usd": 0.0,
            "charged_usd": spec["prior_run"]["prior_debit_usd"],
            "known_cost_usd": 0.0,
            "reserved_usd": 0.0,
            "reserved_attempts": 0,
            "accepted": 0,
            "failed": 0,
            "attempts": 0,
            "budget_unknown": False,
            "stop_reason": None,
        }
        pending = []
        for index, row in enumerate(dataset.corpus):
            directory = output / "rows" / f"{index:09d}"
            if directory.exists():
                accepted, cost = audit_row(
                    spec, dataset, row, index, output, spec_sha256, payload_builder, allow_pending=True
                )
                ledger["charged_usd"] += cost
                ledger["new_job_charged_usd"] += cost
                count = len(list(directory.glob("attempt-*")))
                ledger["attempts"] += count
                for attempt in directory.glob("attempt-*"):
                    bill = read(attempt / "settled.json")
                    if bill["cost_known"]:
                        ledger["known_cost_usd"] += bill["charged_usd"]
                if accepted is None:
                    pending.append((index, row, count))
                else:
                    ledger["accepted"] += 1
            else:
                pending.append((index, row, 0))
        require(ledger["charged_usd"] <= 15, "resume_budget")
        gate = gate or Gate()

        def stop(reason):
            ledger["stop_reason"] = ledger["stop_reason"] or reason

        def stopped():
            return ledger["stop_reason"] is not None

        loop = asyncio.get_running_loop()
        handlers = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
        for sig in handlers:
            loop.add_signal_handler(sig, stop, "interrupted")

        async def one(index, row, start_attempt):
            directory = output / "rows" / f"{index:09d}"
            claim = None
            try:
                control_guard()
                original_row = copy.deepcopy(row)
                view = {"policy": copy.deepcopy(controls["policy"]), "assets": frozen_assets}
                payload, asset = payload_builder(view, dataset, row)
                require(
                    view["policy"] == controls["policy"] and row == original_row, "payload_builder_mutation"
                )
                request_sha = seal(payload)
                claim = row_claim(spec, row, index, asset, payload, spec_sha256)
                for attempt_number in range(start_attempt, MAX_ATTEMPTS):
                    if not await gate.enter(stopped):
                        return
                    if ledger["charged_usd"] + ledger["reserved_usd"] + BOUND_USD > 15 + 1e-12:
                        gate.leave()
                        stop("budget_stop")
                        return
                    ledger["reserved_attempts"] += 1
                    ledger["reserved_usd"] = ledger["reserved_attempts"] * BOUND_USD
                    try:
                        if not directory.exists():
                            directory.mkdir(mode=0o700)
                            write(directory / "claim.json", claim)
                            write(directory / "request.json", payload)
                        attempt = directory / f"attempt-{attempt_number}"
                        attempt.mkdir(mode=0o700)
                        write(
                            attempt / "reservation.json",
                            {"request_sha256": request_sha, "bound_usd": BOUND_USD},
                        )
                        ledger["attempts"] += 1
                        try:
                            async with asyncio.timeout(95):
                                raw = await transport(payload)
                        except Exception as error:
                            raw = {
                                "status_code": None,
                                "events": [],
                                "done": False,
                                "error_type": type(error).__name__,
                            }
                        write(attempt / "raw-response.json", raw)
                        billing = {**charge(raw), "raw_response_sha256": seal(raw)}
                        write(attempt / "settled.json", billing)
                        ledger["reserved_attempts"] -= 1
                        ledger["reserved_usd"] = ledger["reserved_attempts"] * BOUND_USD
                        ledger["charged_usd"] += billing["charged_usd"]
                        ledger["new_job_charged_usd"] += billing["charged_usd"]
                        if billing["cost_known"]:
                            ledger["known_cost_usd"] += billing["charged_usd"]
                        if billing.get("bound_violation"):
                            stop("billing_bound_violation")
                            raise ValueError("billing_bound_violation")
                        if billing.get("budget_unknown"):
                            ledger["budget_unknown"] = True
                            stop("budget_unknown")
                        control_guard()
                        require(
                            seal(payload) == request_sha and row == original_row, "request_or_source_mutation"
                        )
                        require(
                            sha(dataset.root / row["media"]["image"]) == asset,
                            "source_changed_during_request",
                        )
                        code = transient_code(raw)
                        if code == 429:
                            gate.penalize()
                        if retryable(raw):
                            if attempt_number + 1 < MAX_ATTEMPTS and not stopped():
                                await asyncio.sleep(min(4, 2**attempt_number))
                                continue
                            if stopped():
                                return
                            raise RowRejection("transport_retries_exhausted")
                        text, usage = strict_caption(raw)
                        write(
                            directory / "accepted.json",
                            {
                                **claim,
                                "text": text,
                                "attempt": attempt_number,
                                "raw_response_sha256": seal(raw),
                                "usage": usage,
                                "provider": PROVIDER,
                            },
                        )
                        ledger["accepted"] += 1
                        gate.good()
                        return
                    finally:
                        gate.leave()
            except BaseException as error:
                ledger["failed"] += 1
                ledger["failed_indices"].append(index)
                ledger["failed_reasons"][str(index)] = (
                    str(error) if isinstance(error, RowRejection) else type(error).__name__
                )
                if not isinstance(error, RowRejection):
                    stop("failed_requires_review")
                if directory.exists() and not (directory / "failed.json").exists():
                    write(
                        directory / "failed.json",
                        {
                            "error_type": type(error).__name__,
                            "reason_code": str(error)
                            if isinstance(error, RowRejection)
                            else type(error).__name__,
                        },
                    )

        tasks, iterator = set(), iter(pending)
        exhausted = False
        try:
            while tasks or not exhausted:
                while len(tasks) < 100 and not stopped() and not exhausted:
                    item = next(iterator, None)
                    if item is None:
                        exhausted = True
                    else:
                        tasks.add(asyncio.create_task(one(*item)))
                if stopped():
                    exhausted = True
                if tasks:
                    _, tasks = await asyncio.wait(tasks, timeout=2, return_when=asyncio.FIRST_COMPLETED)
                ledger["elapsed_seconds"] = time.monotonic() - started
                write(
                    output / "status.json",
                    {
                        "status": "running",
                        **ledger,
                        "inflight": len(tasks),
                        "adaptive_limit": gate.limit,
                        "max_rps": gate.rps,
                        "spec_sha256": spec_sha256,
                    },
                    fresh=False,
                )
                if not tasks and exhausted:
                    break
            require(seal(spec) == frozen, "full_spec_mutation")
            verify(spec, dataset)
            status = (
                "rows_complete"
                if ledger["accepted"] == len(dataset.corpus) and not ledger["failed"]
                else (ledger["stop_reason"] or "rows_partial")
            )
            ledger["elapsed_seconds"] = time.monotonic() - started
            result = {
                "fresh_full_generation": True,
                "local_or_v1_caption_reuse": False,
                "status": status,
                **ledger,
                "requests_closed": True,
                "spec_sha256": spec_sha256,
                "source_identity": dataset.identity,
                "session": str(session),
                "caption_rows": ledger["accepted"],
                "benchmark_runs": 0,
                "retrieval_calls": 0,
                "reranker_calls": 0,
                "answer_generation_calls": 0,
                "database_writes": 0,
            }
            write(output / "status.json", result, fresh=False)
            return result
        finally:
            for sig, prior in handlers.items():
                loop.remove_signal_handler(sig)
                signal.signal(sig, prior)
    finally:
        os.close(lock)


async def http_transport(client, payload):
    raw = {"status_code": None, "events": [], "done": False, "wire_lines": []}
    size = 0
    try:
        async with asyncio.timeout(90):
            async with client.stream("POST", ENDPOINT, json=payload) as response:
                raw["status_code"] = response.status_code
                async for line in response.aiter_lines():
                    size += len(line.encode())
                    require(size <= MAX_RAW_BYTES, "response_size_bound")
                    raw["wire_lines"].append(line)
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        raw["done"] = True
                        break
                    raw["events"].append(json.loads(data))
    except Exception as error:
        raw["error_type"] = type(error).__name__
    return raw


def publish(spec, dataset, accepted, *, spec_sha256):
    """Fresh full derived snapshot; no ordinary/Gemma/pilot cache is altered."""
    from rag_benchmark.multimodal_data import write_dataset, validate_dataset
    from rag_benchmark.multimodal import load_dataset
    from rag_benchmark.lasco_data import publish_directory

    verify(spec, dataset)
    require(len(accepted) == len(dataset.corpus), "all_captions_required")
    destination = safe(spec["destination"])
    stage = destination.with_name(destination.name + ".preparing")
    require(not destination.exists() and not stage.exists(), "fresh_export_required")
    stage.mkdir(mode=0o700)
    corpus, by = [], {}
    for row, receipt in zip(dataset.corpus, accepted, strict=True):
        require(
            receipt["source_id"] == row["id"] and receipt["spec_sha256"] == spec_sha256,
            "caption_order_policy",
        )
        relative = row["media"]["image"]
        require(
            sha(dataset.root / relative) == receipt["asset_sha256"] == spec["assets"][relative],
            "export_source_asset",
        )
        target = safe(stage / relative, stage)
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.link(safe(dataset.root / relative, dataset.root), target)
        by[row["id"]] = receipt
        corpus.append(
            {
                "id": row["id"],
                "text": receipt["text"],
                "media": copy.deepcopy(row["media"]),
                "metadata": {
                    "text_provenance": {
                        "source": "generated_from_source_only",
                        "query_independent": True,
                        "gold_fields_used": False,
                        "caption_receipt_sha256": seal(receipt),
                        "spec_sha256": spec_sha256,
                        "caption_protocol": PROTOCOL,
                    }
                },
            }
        )
    queries = []
    for row in dataset.queries:
        ref = str(row["metadata"]["reference_id"])
        metadata = copy.deepcopy(row["metadata"])
        metadata["text_surrogate"] = {
            "text": "Reference: " + by[ref]["text"] + "\nRequested change: " + row["text"],
            "provenance": {
                "source": "reference_description_and_instruction",
                "gold_fields_used": False,
                "gallery_caption_query_independent": True,
                "reference_id": ref,
                "reference_asset_sha256": by[ref]["asset_sha256"],
                "caption_receipt_sha256": seal(by[ref]),
                "transition_sha256": seal(row["text"]),
                "format": "reference-requested-change-v1",
            },
        }
        queries.append(
            {"id": row["id"], "text": row["text"], "media": copy.deepcopy(row["media"]), "metadata": metadata}
        )
    qrels = [
        json.loads(line) for line in (dataset.root / "qrels.jsonl").read_text().splitlines() if line.strip()
    ]
    manifest = write_dataset(
        stage,
        dataset_id="lasco-validation-haiku55-source-caption-en-v2",
        track="composed_image",
        revision=seal({"source": dataset.identity, "spec_sha256": spec_sha256, "protocol": PROTOCOL}),
        corpus=corpus,
        queries=queries,
        qrels=qrels,
        sources=dataset.manifest["sources"],
        license=dataset.manifest["license"],
        text_source="generated_from_source_only",
        expected_counts=spec["counts"],
        metadata={
            "source_dataset_identity": dataset.identity,
            "caption_protocol": PROTOCOL,
            "caption_spec_sha256": spec_sha256,
            "split": "validation",
        },
    )
    (stage / "qrels.jsonl").write_bytes((dataset.root / "qrels.jsonl").read_bytes())
    manifest["files"]["qrels.jsonl"] = {
        "sha256": sha(stage / "qrels.jsonl"),
        "size_bytes": (stage / "qrels.jsonl").stat().st_size,
    }
    write(stage / "dataset.json", manifest, fresh=False)
    validate_dataset(stage)
    derived = load_dataset(stage)
    require(derived.qrels == dataset.qrels, "labels_changed")
    derived.require_text()
    verify(spec, dataset)
    publish_directory(stage, destination)
    return derived.identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--execute", action="store_true")
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--spec-sha256")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prior-spec", type=Path)
    parser.add_argument("--prior-spec-sha256")
    parser.add_argument("--prior-result", type=Path)
    parser.add_argument("--prior-result-sha256")
    args = parser.parse_args()
    os.umask(0o077)
    from rag_benchmark.multimodal import load_dataset

    if args.prepare:
        require(args.dataset and args.output and args.destination and not args.resume, "prepare_arguments")
        spec = prepare(
            load_dataset(args.dataset),
            args.output,
            args.destination,
            prior_spec=args.prior_spec,
            prior_spec_sha256=args.prior_spec_sha256,
            prior_result=args.prior_result,
            prior_result_sha256=args.prior_result_sha256,
        )
        write(args.spec, spec)
        print(json.dumps({"status": "prepared_not_executed", "spec_sha256": sha(args.spec)}))
        return
    raw = args.spec.read_bytes()
    require(args.spec_sha256 and hashlib.sha256(raw).hexdigest() == args.spec_sha256, "external_spec_sha")
    spec = json.loads(raw)
    require(spec["counts"] == {"corpus": 39826, "queries": 30037}, "production_full_counts")
    dataset = load_dataset(spec["source"])

    async def actual():
        import httpx

        key = os.environ["OPENROUTER_API_KEY"]
        async with httpx.AsyncClient(
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
            timeout=httpx.Timeout(65, connect=15),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=100),
            trust_env=False,
            follow_redirects=False,
        ) as client:
            return await run(
                spec,
                dataset,
                lambda payload: http_transport(client, payload),
                spec_sha256=args.spec_sha256,
                resume=args.resume,
            )

    result = asyncio.run(actual())
    if result["status"] == "rows_complete":
        accepted = audit_rows(spec, dataset, spec["output"], spec_sha256=args.spec_sha256)
        identity = publish(spec, dataset, accepted, spec_sha256=args.spec_sha256)
        result.update(status="completed", producer_closed=True, derived_identity=identity)
    else:
        result["producer_closed"] = True
    write(Path(result["session"]) / "result.json", result)
    write(Path(spec["output"]) / "result.json", result, fresh=False)
    write(Path(spec["output"]) / "status.json", result, fresh=False)
    print(json.dumps({k: result[k] for k in ("status", "caption_rows", "charged_usd", "producer_closed")}))
    if result["status"] != "completed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

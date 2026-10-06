# Contributing

Contributions should make comparisons easier to reproduce or interpret. Explain the concrete change, its effect on an experiment and how it was checked.

## Keep experiments comparable

- Read [the protocol](docs/protocol.md) before changing retrieval, prompts, metrics or splitting.
- Preserve the same query IDs, corpus and generation policy when comparing a controlled pair.
- Version changes to tokenization, prompts, truncation, fusion, model revisions and cache keys. Old cached results must not silently survive incompatible settings.
- Tune on development data. Do not select settings from the final test and present that test as untouched.
- Separate a proposed improvement from a measured one. “Faster” and “better” need an explicit baseline, scope and result artifact.

## Validation

Install the locked lightweight development environment with `uv sync --locked`, then run `uv run --locked pytest`. Tests using fixtures do not require the large model dependencies. For real model work, use `uv sync --extra models --locked` and keep `--extra models` on subsequent `uv run --locked --extra models ...` commands so dependency synchronization preserves that environment.

Use focused tests for changed behavior, especially ID/label mapping, article-group splitting, cache invalidation, ranking ties, metrics, truncation and failure accounting. Fixtures must be visibly identified. They do not demonstrate that a pretrained model loaded or that answer quality improved.

Local model integration checks should identify the model revision, device/backend and sample size. A full matrix is not required for a documentation change, but a backend change needs a real integration check before it is labeled verified. Record unavailable runtimes honestly.

## Results and data

Do not add invented, copied or fixture-derived model scores to [the results page](docs/results.md). A result contribution should include its manifest, effective configuration, dataset/split identity, completed/failed counts and compact per-query metrics where redistribution is permitted.

For public artifacts, use `uv run --locked --extra models rag-benchmark export-report --run-dir runs/name --output-dir reports/new-name`. The destination must be new. Commit the resulting allowlisted metrics/provenance export rather than raw `predictions.jsonl`; keep the original run locally for audit.

Keep raw datasets, downloaded model weights, secrets and local caches out of Git. RAGTurk data and model weights retain their own licenses; MIT licensing of the harness does not override them. Predictions can reproduce source text, so review their redistribution separately from aggregate numeric reports.

## Documentation

English is the primary README. Keep [README.tr.md](README.tr.md) consistent with user-facing commands, status and conclusions. Update the protocol when the experiment changes, and update the evidence ledger only after validation actually occurs.

For a bug report, include the command, sanitized configuration, code revision and concise error. Do not include access tokens or private documents.

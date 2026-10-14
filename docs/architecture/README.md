# Modular benchmark architecture — design review

Experiment execution is **stopped at the user's request**. This directory contains research and a migration proposal, not a newly implemented distributed engine. Existing frozen runs, public results and cache identities are preserved. No new model or provider calls were made for this design.

| Document | Purpose |
|---|---|
| [Architecture proposal (Turkish)](modular-benchmark.tr.md) | Interchangeable datasets, representations, retrievers, fusion, decision scorers, rerankers, generators and evaluators; identity, lifecycle and migration gates |
| [Scenario catalog (Turkish)](scenario-catalog.tr.md) | Controlled component swaps, hybrids, multimodal inputs, offline/online operation and heterogeneous execution options |
| [Current system audit](current-system-audit.md) | Existing abstractions and hardcoded couplings, with source locations |
| [Distributed execution options](distributed-execution-options.md) | Pull workers, Ray and AWS Batch; GPU batching, sharding and failure semantics |
| [Non-executable configuration example](../../configs/examples/README.md) | Concrete illustration of the proposed plan boundary |
| [Execution stop record](../../reports/colab-execution-stop.json) | Completed work retained, partial work limitations and paused monitoring |

Recommended direction: preserve the scientific core, wrap legacy implementations with versioned capability contracts, compile explicit experiment plans, then add a small durable pull-worker executor. Ray, AWS Batch, Haystack and provider gateways remain optional integrations behind those boundaries. GPU placement may adapt to measured capacity; scientific inputs and numerical conditions cannot change silently.

Implementation should proceed in small steps: legacy parity fixtures → registry and CPU plan validation → cancellation/checkpoint ownership → shard/merge equivalence → new plugins → separately authorized real execution. New adapters are not verified until their own integration evidence exists. This proposal does not authorize resuming the paused experiment queue.

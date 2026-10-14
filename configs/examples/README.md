# Architecture examples — not executable configurations

[`ragturk-ten-jobs.proposal.json`](ragturk-ten-jobs.proposal.json) defines ten scenario jobs: five retrieval methods × none/Laya, historical pilot evidence per job, separate full-development outputs and shared preparation. It is a workflow proposal, not a scheduler accepted by the current CLI. Execution is disabled; no workers are allocated. [Workflow and current gaps](../../docs/architecture/ragturk-ten-jobs.tr.md).

[`modular-benchmark.proposal.json`](modular-benchmark.proposal.json) illustrates the proposed dataset/component/study/execution boundary. It is **not supported by the current CLI**, is not a complete schema, and must not be passed to an experiment runner. `execution_enabled` and cloud allocation are false. Null revisions, unresolved protocol references, unspecified budgets and unapproved online providers intentionally block future execution.

The example separates retrieval-only, fixed-candidate decision-reranker and fixed-evidence generator studies. It does not silently replace the original B/G/E registry: these short labels are local aliases within the illustrative plan. Model, input and numerical identities must be resolved into a frozen plan before a future planner could execute it.

This example is a conservative reference-profile skeleton. The design's adaptive heterogeneous profile would need a separate policy/condition identifier and a declared assignment/quality rule; it is not enabled by relaxing this example's cross-condition merge gate. Dense retrieval channels explicitly bind an embedding provider to exact search and unresolved normalization/tie/exclusion policies.

No secrets, raw data, provider calls, model loads or cloud allocations are included. [Design](../../docs/architecture/modular-benchmark.tr.md) · [Scenarios](../../docs/architecture/scenario-catalog.tr.md).

# Current benchmark architecture audit

This is a source review and migration recommendation, not an implemented architecture or evidence of new experiments. Existing frozen results, numerical conditions, cache keys and publication populations must remain intact. No model, cloud or network execution was performed for this audit.

## Existing boundaries and couplings

| Area | Existing implementation | Useful boundary and current coupling |
|---|---|---|
| Dataset preparation | `src/rag_benchmark/data.py:66`; `multimodal_data.py:141`, `:188`, `:314`, `:505` | Preparation is distinct from execution, with revisions/checksums and modality-specific converters. Converters explicitly target RAGTurk, ViDoRe, CodeSearchNet and other named collections; there is no common provider registry. |
| Frozen dataset and model inputs | `multimodal.py:214`, `:301` | `Dataset` holds corpus, queries, qrels and identity. `model_item()` removes labels, answers and arbitrary metadata from inference input. It also resolves media to host absolute paths. This is a valuable security/provenance boundary, but path resolution leaks host representation into downstream score identity. |
| Embedding and retrieval | `models.py:137`; `retrieval.py:60`; `multimodal.py:498`, `:567`, `:875` | Adapters already expose identity, dimension, encoding and optional direct ranking; precomputed vectors are supported. Dispatch relies on letter channels and `hasattr(adapter, "rank")`, rather than declared capabilities. Text embedding families and IDs/dimensions are intentionally constrained in `DenseEmbedder`. |
| Fusion and candidate pools | `retrieval.py:29`; `multimodal.py:429`, `:441` | RRF and per-channel/total budget allocation are reusable pure functions. Their ordering, no-refill behavior and single-channel score handling are protocol semantics, not interchangeable implementation details. Two retrieval pipelines maintain related fusion logic. |
| Rerankers | `multimodal.py:522`, `:970`; `multimodal_generation.py:262`; `models.py:206` | `score(query, candidates)`, identity and `pointwise` provide a working informal contract. Pointwise scores reuse across cells; listwise cache keys include the entire candidate list. Candidate-preservation, finite-score, truncation and requested-device checks exist. They should become explicit contracts without changing their current behavior. |
| Generators and transports | `models.py:286`; `multimodal_generation.py:147`; `multimodal_qa_runner.py:62`, `:356` | Local text/multimodal generation and injectable QA transport exist. Generator prompts/model defaults and QA evidence serialization are closely coupled to named models and existing local endpoints. Transport implementation and experimental generation policy are not yet independently selectable abstractions. |
| Evaluation | `metrics.py:24`, `:40`; `multimodal.py:356`, `:398`; `multimodal_qa.py:54`, `:141`; `semantic.py:98`, `:267`, `:486` | Lexical answer, binary/graded retrieval, citation and semantic evaluation are separate implementations. Their populations, cutoffs, relevance definitions and grouping assumptions differ. A single generic “accuracy” interface would erase distinctions that the current documentation deliberately preserves. |
| Cache and resume | `storage.py:20`; `multimodal.py:597`, `:736`, `:939`, `:970`; `multimodal_qa_runner.py:100` | Dataset/model/prompt/role/dimension identities, per-run SQLite, shared exact-request responses and checkpointing already exist. They are several storage implementations, not a distributed cache/shard protocol. A cache read is explicitly not fresh inference latency. |
| Matrix planning | `multimodal.py:29`, `:40`, `:141`, `:1034`; `multimodal_qa_plan.py:49` | Track/channel/reranker registries, powersets and full-query plans exist. Track names embed particular datasets/specialists; rerankers and K grid20/50/100 are closed sets; global family counts are literal report fields. Configuration cannot yet express arbitrary new component combinations without code edits. |
| CLI and orchestration | `multimodal_cli.py:134`; `runner.py:92`; `colab_worker.py:100`, `:193` | CLI constructs concrete adapters; the Colab worker fixes CUDA BF16, dimension768, vision280 and batch settings while staging channels. Provider selection, execution ownership, artifact transfer and scientific protocol are not a single abstraction. Private operational helpers under `work/colab` are preparation tools, not supported public orchestration APIs. |

References above are repository-relative paths; abbreviated filenames in the table remain under `src/rag_benchmark/`.

## Preserve the current scientific contracts

`configs/ragturk.toml:1` already separates dataset, experiment and model configuration for the original text benchmark. The multimodal JSON plans instead bind concrete frozen dataset summaries and ordered query hashes (`configs/colab-retrieval-jobs.json:1`). Both styles are useful inputs to a normalized plan, but neither should be rewritten in place to change the meaning of an existing run.

`docs/protocol.md:1` distinguishes intended comparison from completed evidence and forbids leakage of reference answers into model requests. `multimodal_qa_runner.py:183` and `:199` freeze generation/evidence policy: source budget5, fixed prompt contract, explicit output limits and allowed representations. `run_matrix()` constrains the candidate grid and records smoke versus full-collection scope (`multimodal.py:1034`). These restrictions preserve the existing baseline. Future flexibility should create a new named protocol rather than weakening legacy validation.

Retain explicit unsupported, failed, partial and completed states. A model unable to accept a modality, token length, batch or numerical condition must not silently use a surrogate, truncate, switch dtype/device, or drop candidates. Some alternatives are valid experiments, but require separate plan identities and reported coverage. Hardware adaptation changes execution conditions; it does not make old CUDA/MPS results equivalent.

## Minimal target contracts

Introduce a small versioned component registry and validate a declarative plan before allocating resources. Avoid a rewrite of the runners as the first step. Wrap existing implementations behind these contracts:

- **Dataset provider:** preparation recipe and immutable revision; frozen inventory; ordered query IDs; task and relevance schema; permitted input views; exclusions; grouping provenance; license/access and private/public artifact policy. Keep gold labels in the evaluator side of the boundary.
- **Retriever:** accepted query/document modalities, direct-rank versus vector interface, query/document roles, similarity/normalization, native dimension, preprocessing/prompt identity and explicit maximum-input policy. An embedding model is not itself a complete retriever.
- **Fusion:** input channel ordering, budget allocation, duplicate treatment, refill policy, tie handling, output cutoff and parameters. Legacy RRF is one frozen implementation.
- **Reranker:** pointwise or listwise semantics; accepted input view; candidate count/order preservation; score domain and deterministic ranking policy; batch/dtype/device/overflow constraints. Reuse pointwise scores only when the full scoring request identity matches.
- **Generator:** request/response schema, accepted evidence representations, context/output budgets, prompt version, decoding settings, finish/truncation diagnostics and transport mode. Keep evidence selection independently identifiable.
- **Evaluator:** task, required labels, eligibility denominator, metric version/cutoffs, citation/reference policy and aggregation/grouping assumptions. Bootstrap eligibility must be derived from actual source structure, not copied from a different dataset.

Each component should return a validated result plus execution diagnostics and a public provenance summary. The registry must describe capabilities without loading model weights; CPU planning should identify incompatible combinations early. A plugin must not be able to claim generic support for a modality merely because it can receive a filename.

## Offline, online and adaptive execution

Make offline inference/precomputed replay and online provider transport explicit execution modes. Existing `models.py:62` and `:92` already separate fetching from offline inference, while `LocalQATransport` is injectable. Preserve that separation: preparation can materialize pinned artifacts; execution should not fetch a changing model or dataset implicitly. Online requests need provider/model version, request schema, retry/timeout policy and retained private response provenance; account secrets never belong in plan hashes or public reports.

An adaptive planner may choose among **predeclared** compatible hardware/batch/shard configurations after a bounded capacity check. It must record the selected condition before execution. A retry that changes batch, context, preprocessing or dtype becomes another explicit condition unless equivalence has been established under a documented rule. Throughput estimates should separately report measured sample scope, failures, setup and transfer/audit overhead; they are not accuracy or successful-completion guarantees.

## Cache identity and reproducibility migration

Keep existing engine/version1 keys and frozen artifacts readable. Introduce a new versioned identity schema only for new protocols, composed from immutable dataset/input content, component implementation/configuration/prompt identities and numerical conditions. Separate scientific request identity from provider allocation/session metadata, while retaining actual hardware/software provenance for comparisons.

For new media requests, consider content-addressed references plus a resolver instead of absolute paths. Current `_rerank()` hashes model items containing absolute media paths (`multimodal.py:970`), so portability cannot be repaired by relabeling old cache entries. Legacy scores must retain their original identity; any migration requires a separately verified mapping and explicit provenance.

A new artifact manifest should inventory byte hashes, completion state, protocol and query/shard coverage. Closed checkpoints and independently audited aggregate exports remain separate artifacts. Query sharding requires disjoint ownership, collision-safe caches, duplicate/conflicting result rejection and recomputation of complete-population metrics after merge. Existing SQLite/shared caches do not establish those guarantees by themselves.

## Migration priorities and acceptance gates

1. **Freeze and describe legacy protocols.** Add declarative manifests and wrappers for existing dataset/adapters/fusion/evaluators. Keep existing entrypoints and identities. Acceptance: replay synthetic fixtures through old and wrapped paths yields identical rankings, metrics, requests and cache keys.
2. **Add capability validation and a plan compiler.** Replace new-adapter CLI branching with registry resolution, keeping legacy aliases. Emit supported/unsupported cells, eligibility populations, identities and resource requirements without model loads. Acceptance: incompatible modalities, missing labels, unsupported lengths and unsafe fallback policies fail before execution.
3. **Extract execution/storage services.** Give adapters explicit preparation/load/unload/execute lifecycle and transport interfaces; give orchestration ownership, cancellation, checkpoint and timeout contracts. Acceptance: interruption preserves verified artifacts, incomplete runs cannot export full results, and no concurrency path bypasses device ownership.
4. **Add versioned portable identities and immutable shard manifests.** Preserve all existing keys. Acceptance: changed content/model/prompt/protocol invalidates the appropriate new cache; shuffled or duplicated shards cannot pass full coverage; cache reuse is reported separately from fresh inference.
5. **Expand models/datasets through contract fixtures.** Register one new dataset and one alternative component only after the above gates. Acceptance: plugin contract tests cover missing/empty input distinctions, gold exclusion, finite scores, candidate preservation, truncation/fallback and public/private export boundaries.

Existing tests provide starting gates, not evidence of the proposed architecture: `tests/test_multimodal.py:118` tests identity separation by role/revision/prompt/dimension, `tests/test_multimodal_qa_runner.py` exercises injectable transport and exact-request behavior, and model/data/CLI tests cover existing supported settings. Keep these regressions; add focused plugin/planner tests instead of replacing established scientific checks.

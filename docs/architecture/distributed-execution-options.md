# Distributed adaptive execution options

Research date: **2026-10-14**. Sources below are official documentation accessed on that date. This document proposes an execution architecture; it does not report provisioned infrastructure, new experiments or measured throughput. It complements [the current system audit](current-system-audit.md). Existing frozen protocols, cache identities and results remain unchanged.

## Recommendation

Start with a backend-neutral **pull queue, transactional lease ledger and immutable artifact store**, with workers that keep one compatible model resident when useful. Add Ray within a stable managed GPU pool if actor reuse and Python pipeline scheduling justify its operational complexity. Add AWS Batch as an AWS execution adapter when managed container scheduling is useful. Add vLLM as a separately validated LLM inference adapter; it does not replace the task ledger or scientific artifact protocol.

This is a design recommendation, not a claim that one framework is universally faster. It lets Mac, ephemeral Colab, managed AWS and API workers share task contracts without pretending that their numerical execution conditions are interchangeable.

**Twenty jobs do not require twenty simultaneously loaded models.** A job is queued work; a worker is an execution resource; a resident model consumes memory; a model replica may process multiple compatible requests. For illustration, three workers admitted to exclusive single-GPU jobs can execute three jobs while seventeen wait. That example is not an observation of current capacity. CPU preparation, GPU inference and API requests can overlap only within their separate measured resource and quota limits.

## Options and boundaries

| Option | Useful role | Advantages | Obligations and limits |
|---|---|---|---|
| Pull queue + lease ledger + durable object store | Common control plane across all backends | Small interoperable worker protocol; workers can join and disappear independently; durable progress need not depend on a notebook or process | Implement leases, fencing, fairness, idempotent commits, dependency/coverage checks and retry classification. Queue delivery alone is insufficient. |
| Ray | Execution within a stable Python compute pool | Stateful model actors, dependency scheduling, resource admission and batched data pipelines | Logical GPU resources do not reserve physical VRAM. External durable commits and idempotent retries remain necessary. A mixed WAN cluster introduces networking and environment obligations. |
| AWS Batch | AWS container execution adapter | Managed queues, GPU reservations, array jobs, dependencies, retries and fair-share policies | Does not manage Mac or Colab workers. Instance availability, container startup and model loading affect completion. Per-job GPU count is not a token/VRAM capacity controller. |
| vLLM | Compatible LLM engine inside a worker/service | Offline prompt batches and online serving with batching/token controls | Model/backend compatibility and numerical identity need validation. It is neither a durable workload queue nor a general reranker/encoder scheduler. |

Ray advertises CPU, GPU and custom **logical** resources; these control placement, not physical memory isolation. Fractional GPU scheduling therefore needs an application memory budget. Stateful `map_batches` workers can reuse model loads, while tasks in flight and genuinely concurrent actor computation are different settings. [Ray resources](https://docs.ray.io/en/latest/ray-core/scheduling/resources.html), [accelerators](https://docs.ray.io/en/latest/ray-core/scheduling/accelerators.html), [map_batches](https://docs.ray.io/en/latest/data/api/doc/ray.data.Dataset.map_batches.html), [actor pool strategy](https://docs.ray.io/en/latest/data/api/doc/ray.data.ActorPoolStrategy.html).

Ray provides Apple silicon wheels, but its installation documentation says Apple silicon multi-node clusters are untested. Native MPS GPU scheduling is not listed in the accelerator support table. Use an explicit Mac capability lane or standalone pull worker rather than claiming that a CUDA resource declaration schedules MPS correctly. A single mixed Mac/Colab/AWS Ray cluster needs separate validation; the recommended first Ray deployment is a stable managed pool. [Ray installation](https://docs.ray.io/en/latest/ray-overview/installation.html), [accelerator support](https://docs.ray.io/en/latest/ray-core/scheduling/accelerators.html).

AWS Batch reserves the requested GPUs for a job and supports NVIDIA GPU instance families, including P5. Arrays describe independent child jobs, not guaranteed simultaneous execution. FIFO or fair-share scheduling and dependencies are available; consumable resources can additionally represent constrained external capacities. An application still needs length-aware inference admission and provider-specific quota accounting. [GPU jobs](https://docs.aws.amazon.com/batch/latest/userguide/gpu-jobs.html), [array jobs](https://docs.aws.amazon.com/batch/latest/userguide/array_jobs.html), [scheduling](https://docs.aws.amazon.com/batch/latest/userguide/job_scheduling.html), [resource-aware scheduling](https://docs.aws.amazon.com/batch/latest/userguide/resource-aware-scheduling.html).

## Separate scientific identity from execution flexibility

Use three distinct records:

- **Work identity:** immutable input/content hashes, ordered source/evidence IDs, dataset population, component implementation and semantic configuration, prompts, output schema and planned shard coverage.
- **Execution identity:** actual device/backend, GPU family and memory, driver/framework/package versions, model/tokenizer/processor revisions and weight hashes, dtype/quantization, batch/context settings and any numerical scheduling condition.
- **Attempt identity:** worker, lease/fencing token, timestamps, transport, infrastructure failures and artifact publication steps.

A strict frozen condition runs only on a worker satisfying its existing compatibility contract. A flexible throughput condition may choose among predeclared configurations, but records its selection as a new condition whenever equivalence has not been established. Do not rewrite historical MPS keys into CUDA keys, substitute L4/A100/H100 identities, or label cached answers as fresh inference. Existing media-path-bearing keys must keep their original path semantics; a portable key schema requires explicit versioning and verified migration, not arbitrary path stripping.

A seed alone does not establish cross-device equality. vLLM documents that defaults do not guarantee reproducibility, and its reproducibility guidance limits configured guarantees to the same hardware and vLLM version. Batch invariance is beta with a documented support envelope; it must not be assumed for every project model. [vLLM reproducibility](https://docs.vllm.ai/en/stable/usage/reproducibility/), [batch invariance](https://docs.vllm.ai/en/stable/features/batch_invariance/).

A worker capability card should report observed hardware and available RAM/disk, supported modalities/lengths, model residency, execution identity and quota scope. Requested device names are insufficient: PyTorch can permit unsupported MPS operations to fall back to CPU. Colab resource availability, GPU types and VM lifetimes are dynamic; paid access is not a durable worker guarantee. Treat local worker storage as replaceable and upload validated shards promptly. API workers record provider/model/version and observable provenance; their underlying GPU is generally unknown. [MPS environment variables](https://docs.pytorch.org/docs/stable/mps_environment_variables.html), [Colab FAQ](https://research.google.com/colaboratory/faq.html).

## Adaptive capacity without changing the experiment silently

Admission must account for model weights, activations, workspaces, live KV cache and measured safety headroom within usable memory. Host RAM and disk also constrain loading, media decoding and transfer. A padded encoder/reranker batch depends on both batch count and maximum sequence/media size; an LLM server depends on simultaneous live tokens and requested output lengths. Ten short requests and ten long multimodal requests are not equivalent resource demands.

For a compatible vLLM adapter, `max_num_seqs`, token budgets and memory/KV controls are useful independent knobs. Offline `LLM.generate` accepts prompt lists; online serving batches requests. Current documentation also lists Apple silicon through a separate vLLM-Metal plugin, while GGUF support is experimental and moved to a plugin. Neither is a drop-in equivalence claim for the current MPS or llama.cpp/GGUF execution path. [Quickstart](https://docs.vllm.ai/en/stable/getting_started/quickstart/), [serve controls](https://docs.vllm.ai/en/stable/cli/serve/), [optimization](https://docs.vllm.ai/en/stable/configuration/optimization/), [installation](https://docs.vllm.ai/en/stable/getting_started/installation/), [GGUF](https://docs.vllm.ai/en/stable/features/quantization/gguf/).

The proposed controller should:

1. Measure representative modality/length buckets, separating cold load and warm execution. Record valid units per second, failures, peak memory, latency, queue wait and transfer/audit cost.
2. Try only bounded, predeclared batch/concurrency candidates. Increase gradually; retain failure evidence and cooldown after OOM rather than repeatedly launching the same oversized workload.
3. Pack compatible requests by measured cost and length, keep models warm, and reserve capacity for aged/long work so short requests do not starve them.
4. Preserve input content, truncation policy, output limits, schemas and gold exclusion. A pinned batch setting cannot be changed under the old identity merely to make it fit.
5. Stop treating a GPU fallback, invalid response or partial artifact as a successful completed unit.

For API workers, admission additionally uses provider/project/model quotas, bounded in-flight requests and token accounting. Gemini's documented limits include requests and input tokens per minute and requests per day, scoped to a project rather than an API key; other providers require their own adapters. Async batch is a separate deadline policy: Gemini documents a target of 24 hours and explicitly non-idempotent job creation. Persist submission IDs and reconcile unknown outcomes before resubmission. [Gemini limits](https://ai.google.dev/gemini-api/docs/rate-limits), [batch API](https://ai.google.dev/gemini-api/docs/batch-api).

## Durable ownership, fairness and retries

The ledger should distinguish queued, leased, computing, uploading, validated, committed and audited states. Heartbeats extend ownership for real work; they are not fake notebook keepalives. Each new lease gets a fencing token, and an expired worker cannot commit over the current owner. Group compatible model work to amortize loads, but combine that preference with age/fairness budgets.

Standard SQS delivers at least once and can duplicate or reorder messages. Visibility timeout temporarily hides a leased message, supports extension, and has a maximum of twelve hours. Consequently, choose bounded useful shards and design for duplicate attempts. SQS fair queues can mitigate noisy-neighbor dwell time; they do not impose a precise per-tenant token cap or ordering. [Standard queues](https://docs.aws.amazon.com/AWSSimpleQueueService/latest/SQSDeveloperGuide/standard-queues.html), [visibility timeout](https://docs.aws.amazon.com/AWSSimpleQueueService/latest/SQSDeveloperGuide/sqs-visibility-timeout.html), [fair queues](https://docs.aws.amazon.com/AWSSimpleQueueService/latest/SQSDeveloperGuide/sqs-fair-queues.html).

Write immutable content-addressed blobs, verify their full SHA-256/size and coverage, then conditionally commit a manifest in the ledger before acknowledging the queue message. S3 offers strong consistency and conditional writes for object updates; this does not create an atomic transaction across the queue, ledger and multiple objects. An ETag may be a concurrency token, but multipart ETags are not full-object content hashes. Exactly-once **visible publication** is the design objective; duplicate computation is still possible. [S3 consistency](https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html), [conditional writes](https://docs.aws.amazon.com/AmazonS3/latest/userguide/conditional-writes.html), [integrity checks](https://docs.aws.amazon.com/AmazonS3/latest/userguide/checking-object-integrity-upload.html).

Retry bounded infrastructure failures after reconciling ownership and durable outputs. Quarantine deterministic invalid inputs, auth/configuration errors and repeated OOM; do not repeatedly charge an API or rerun a model because a response was lost. Ray task/actor retries and Batch retries need explicit policy and idempotent effects. Preserve closed failed/interrupted artifacts under a distinct partial protocol; they cannot pass a completed-run audit. [Ray tasks](https://docs.ray.io/en/latest/ray-core/fault_tolerance/tasks.html), [Ray actors](https://docs.ray.io/en/latest/ray-core/fault_tolerance/actors.html), [Batch retries](https://docs.aws.amazon.com/batch/latest/userguide/job_retries.html).

## Sharding and merging correctly

The simplest retrieval shard is a disjoint query set against the same complete frozen gallery. Gallery sharding is also possible, but exact global top-K requires all gallery shards, comparable scores, identical exclusions/normalization and one deterministic global tie rule. With these conditions, merging each shard's local top-K is sufficient: an item outside its shard's top-K already has at least K items ahead of it in that shard. Approximate indexes or incomparable local scores require a different declared protocol.

BM25 must preserve global corpus statistics, tokenizer and document-length conventions; shard-local IDF changes the experiment. Elasticsearch documents the difference between local and global term-statistic scoring. RRF must operate on global per-channel ranks, rather than fuse shard-local ranks and call the result equivalent. [Elasticsearch search types](https://www.elastic.co/guide/en/elasticsearch/reference/8.19/search-search.html), [consistent scoring](https://www.elastic.co/guide/en/elasticsearch/reference/current/consistent-scoring.html).

Pointwise reranking can reuse exact query-candidate scores across K prefixes, budgets and subsets only under the same score/request identity. Listwise reranking depends on the whole candidate list. QA sharing requires the actual canonical request, including ordered evidence, media hashes, representation, prompts and generation configuration; similar questions are not exact duplicates. Embedding shard manifests must cover every frozen gallery ID with the expected order/content hashes.

Merge rejects missing, duplicate or conflicting shard outputs. Compute metrics over the combined per-query population, not an unweighted mean of shard means. Derive source/dependency groups globally; workers and shards are not independent statistical samples. A result becomes complete only after planned coverage, identities and closed artifacts pass independent audit.

## Implementation sequence and estimate rules

First implement CPU/fake-worker acceptance cases for duplicate delivery, lease expiry, stale-owner commit, interrupted upload, out-of-order shards and conflicting/missing merge inputs. Then verify one small real shard against the legacy request/ranking/metric contract. Measure representative short/long and multimodal work on each actual resource lane before expanding the pool. Add Ray or Batch only through the same contracts; no architecture choice alone supplies a measured speedup.

Estimate work from distinct compatible computation units, not matrix-cell counts. For each stage and execution lane, use remaining eligible units divided by measured valid throughput, with separate model loading, transfer, retry and audit costs. Dependencies impose a critical path, and shared GPU resources prevent summing overlapping capacity twice. Do not pool heterogeneous GPU rates without measurements, divide time by twenty jobs, or turn a lower bound into a completion promise.

# Evaluation protocol

This document defines the comparison to be measured. A requirement below is not evidence that a model run has already satisfied it. Completed checks and measured runs belong in [results.md](results.md).

## Questions

1. Does EmbeddingGemma 2 improve Turkish dense retrieval relative to BGE-M3?
2. Does BM25 add useful lexical evidence to either dense retriever?
3. Does Laya improve the final context when each retriever gives it the same candidate pool as its no-reranker control?
4. With the generator fixed, do those context changes improve reference-answer agreement, and at what latency and memory cost?

The experiment does not establish multimodal quality, universal model rankings, human factual accuracy or production performance on unrelated corpora.

## Dataset preparation and split

Use the pinned `metunlp/ragturk` release. Preserve upstream record IDs and source IDs; namespace them if source subsets reuse an ID. Record the dataset revision, source files, checksums, source inventory and retained counts.

Audit empty questions/answers/chunks, duplicate IDs, exact and normalized duplicate questions, source-link coverage and article grouping. If a record is excluded, record why. Do not silently convert malformed or missing labels into an empty relevant-document set and score the query as a miss.

The pinned public release reconstructs to 37,511 chunks and 14,530 usable QA records, below the paper's 58,289 chunks and 20,459 questions. Publish the actual inventory and exclusions rather than filling that gap with fabricated or duplicated material. The split has 2,000 development and 12,530 final-test questions, using seed 42. Group records connected by source article, normalized exact question or normalized exact gold-evidence text. A group cannot contribute question labels to both development and test. The audit does not rule out semantic near-duplicates; exact grouping alone does not prove their absence.

Use one shared corpus for retrieval. Test-source passages remain in that corpus because retrieving them is the task. Never include reference answers, gold IDs or relevance labels in the retrieval query, Laya prompt or generation prompt.

Distinguish these evaluation populations:

- QA records with usable reference answers.
- Records with verifiable relevant chunk IDs, eligible for chunk retrieval metrics.
- Records with only article-level source labels, if any.

Do not mix article-level and chunk-level recall under one metric name. Report the denominator and coverage of each population. Synthetic references and source associations may contain errors; a measured disagreement is not automatically a model error.

## Retrieval controls

Run five paths: BM25, BGE-M3 dense, EmbeddingGemma 2 dense, BM25+BGE-M3 and BM25+EmbeddingGemma 2. Use the same corpus snapshot and query IDs for all paths.

Freeze BM25 tokenization and parameters, embedding task instructions, dimensionality, normalization, truncation policy, dense similarity function, hybrid fusion weights and source depth before final testing. Use the correct query/document formatting for each embedding model and record it. Fairness means each model receives its documented interface; it does not mean imposing an incompatible shared prompt.

Each embedding model gets its own document vectors and index. Never search a BGE document matrix with an EmbeddingGemma query vector. Index cache identity must include the corpus fingerprint and all encoding settings, not just a filename.

For each hybrid, rank BM25 and dense results separately and combine them with reciprocal-rank fusion. Both hybrids use the same fusion policy and source depth. Preserve deterministic tie-breaking and corpus order; disclose the effective policy in the implementation. If approximate nearest-neighbor search is used later, disclose its settings and retrieval approximation separately; do not attribute index approximation effects to model quality.

The candidate budget is 50 per query/path. Save the candidates with IDs, scores and order. BM25 retains positive-score matches and can return fewer candidates; record the actual count rather than padding its ranking with zero-score documents. The context budget is five. Retain the full ranking needed by each metric rather than computing a metric from an unavailable prefix.

## Laya on/off control

For each retrieval path, keep its candidate pool identical across the off/on pair. The off control returns the original first five. The on variant scores those same candidates with a pinned `laya-multilingual` checkpoint and returns up to five, using a deterministic tie-break. Check the recorded candidate IDs when auditing a paired comparison.

The intended initial interface is one question/passage pair per state, evaluated in batches. This avoids copying an entire 50-passage pool into a small context and silently truncating later candidates. Record encoded lengths, truncation counts, batch sizes and device/dtype. Document any difference if the implemented adapter uses another format.

Use the same evidence-relevance question and criteria for every candidate and every retrieval path. Keep document instructions as untrusted content. Do not show the reference answer to Laya. Return relevance scores; do not ask it to write the answer.

The initial protocol has no hard relevance threshold. A low numeric value is not a calibrated estimate of answer failure, especially for this domain/language. Any label-order workaround, threshold, fine-tuning or calibration is chosen on training/development data and documented before final testing.

Laya is independent from TypeSafe Jev. Historical Jev scores from another repository, corpus or candidate pool are not a controlled Laya comparison.

## Generation control

Use the same local Gemma 4 26B-A4B checkpoint and backend across all ten paths. The initial config uses temperature 0, seed 42, a 256-token output cap and thinking disabled. Freeze model revision/quantization, context-token limit, prompt template, temperature/sampling settings, seed where supported, output limit and citation format.

The prompt contains the question and selected source passages. Reference answers and gold IDs are evaluation data only. A common context budget applies after source formatting. Record which context IDs were actually included and whether any text was truncated; five selected IDs do not prove that all five passages reached the generator intact.

Use an explicit insufficient-evidence response policy. Separate abstention, empty/error output and a generated answer that disagrees with the reference. Keep runtime failures in completion accounting; dropping failed rows can inflate a score.

Identical question/context/prompt/model combinations may reuse a generation result. The cache key must include all effective generation settings. An answer cache is an execution optimization, not independent evidence from another model call.

## Metrics and denominators

Report retrieval and answer metrics separately:

| Metric | Interpretation |
|---|---|
| Candidate Recall@50 | Fraction of labeled relevant passages available to the reranker |
| Context Recall@5 | Fraction of labeled relevant passages selected for the generator |
| MRR@k | Reciprocal rank of the first labeled relevant passage, zero if absent |
| nDCG@k | Ranking quality with the disclosed relevance grades and cutoff |
| Answer EM | Exact agreement after the documented normalization |
| Answer token F1 | Token overlap with a reference after the documented normalization |

With one gold passage, per-query recall is a hit/miss. With multiple gold passages, it is a fraction, not merely “any relevant result found.” Use the same relevance interpretation across methods.

For answer metrics, disclose Unicode/case/punctuation/whitespace handling, citation removal and multi-reference aggregation. The implementation strips known context citation IDs for scoring, while retaining the raw answer. It uses Unicode normalization, Turkish case handling, punctuation/whitespace normalization and maximum agreement over reference answers. Token F1 is sensitive to length, paraphrase and morphology; do not label `1 - F1` as hallucination rate. A citation pointing to an existing ID is not a semantic support verdict.

Report number attempted, completed, excluded and failed for every method. Use the shared set of eligible query IDs for paired quality comparisons. Report paired differences, not only independently rounded aggregate scores. Confidence intervals should respect source grouping when feasible; question-level resampling alone can overstate precision when multiple questions share an article. If no interval has been computed, omit it rather than imply significance.

## Timing, caching and resource use

Measure indexing/encoding separately from online query work. Separate query embedding, retrieval/fusion, Laya, prompt preparation and generation where instrumentation permits. Report p50/p95 and the number of measured calls, with device synchronization where needed for asynchronous GPU execution.

Keep these conditions distinct:

- Cold start: model loading and any first-call compilation.
- Warm model, fresh request: actual inference after warm-up.
- Cache hit: loading an already computed embedding, ranking or answer.

Do not compare a cached candidate path against a newly computed one as though both measured serving latency. If a second variant reuses first-stage work, report its actual execution time and mark reused stages. A counterfactual latency reconstructed from earlier measurements must be labeled as such.

Record concurrency, batch size, backend/dtype, cache status and hardware. Do not launch all models concurrently on a shared GPU merely to fill CPU threads; resource contention changes the experiment. Local API expenditure may be zero while elapsed time, memory and energy are nonzero. Do not invent electricity-cost figures without power measurements and an explicit tariff.

## Development and final test

Start with plumbing checks and a small development slice. Tune only on development data. Freeze the protocol/config and inspect the dataset audit before starting the final test.

A final test used to select prompts, thresholds, fusion weights, quantization or context budgets becomes development evidence for those choices. Reserve a new untouched test or clearly label the result exploratory. Do not relabel the same queries as an untouched test after tuning. The CLI's `--retrieval-only` mode is a separate evaluation mode; it does not produce answer quality measurements.

A partial run stays partial. Infrastructure fixtures and synthetic mock model outputs may validate code, but they cannot populate real-model result tables.

## Publishable artifacts

A published result needs a run manifest, dataset audit/split identity, per-query scores and candidate/context IDs, aggregate report, completion status and enough environment information to reproduce the comparison. Pin code and model revisions; record effective values, not only friendly model aliases.

Keep raw dataset text and model weights out of the source repository. Review text-bearing predictions before redistribution under upstream data terms. Publish compact derived metrics, configuration and provenance where permitted. MIT applies to this harness's code, not to the RAGTurk data or third-party model weights.

Sources: [RAGTurk paper](https://aclanthology.org/2026.sigturk-1.15/), [RAGTurk release](https://huggingface.co/datasets/metunlp/ragturk), [Laya source and limits](https://github.com/NandhaKishorM/laya), [Laya multilingual model card](https://huggingface.co/convaiinnovations/laya-multilingual).

# Results and validation status

**The ten-variant pilot is complete: 20 development questions × ten variants, 200/200 outputs.** All retrieval methods searched the full 37,511-passage corpus. A separate BM25 retrieval baseline covers all 2,000 development questions; the ten-variant full-development comparison and 12,530-question final test have not run. These pilot results do not establish a general model winner.

## Evidence ledger

| Evidence | Current status | What it establishes |
|---|---|---|
| Dataset inventory and normalization | Prepared and manifest inspected, 2026-10-06 | 37,511 passages; 14,530 usable QA records; 2,000 development / 12,530 test |
| Infrastructure checks | 124 passing local tests; [CI workflow](https://github.com/erendikmenn/rag-benchmark/actions/workflows/ci.yml) | Code behavior and failure handling, not pretrained model quality |
| BM25 development baseline | Completed 2,000/2,000 real questions against all 37,511 passages | Full development retrieval metrics; no answer generation in this run |
| BGE-M3 and EmbeddingGemma 2 integration | Both full 37,511-passage indexes built and validated; pilot completed in FP32/MPS | Real local dense retrieval over the complete corpus |
| Local Laya integration | All five 20-question reranking pairs completed in FP32/MPS | Same candidate pools within each pair; observed quality is reported below |
| Local Gemma 4 integration | 200/200 full-matrix pilot outputs; 11 cache hits; three length limits at 512 | Real local generation with the pinned GGUF and verified server identity |
| Ten-variant development pilot | Completed all 20 questions per variant | Complete pilot table, not a full-development or final-test result |
| OpenRouter Jev semantic pilot | Completed 200 answers / 319 unique requests; 108 local tests pass | Exploratory semantic labels, with documented judge errors; human calibration pending |
| GPT 6.1 Sol Ultra parallel-agent comparison | Completed 319 benchmark + 12 separate control judgments | Same 200 answers and rubric; agreement measured, human accuracy not established |
| Ten-variant full-development comparison | Not run | Only BM25 retrieval currently covers all 2,000 development questions |
| Frozen final test comparison | Not run/reported | No final leaderboard |

Update this ledger only from actual commands, manifests and inspected artifacts. Include the checked code/config revision and exact scope when changing a row.

The [68-test CI run](https://github.com/erendikmenn/rag-benchmark/actions/runs/37527343627) succeeded for commit `e094aa4719a381ce9786e21d9503a2f61596ee76`; local tests and Ruff also passed. Real-model checks are separate from CI. The run reports, completion markers and summaries below were inspected; the final test split remains untouched.

## Complete ten-variant development pilot

Run `matrix-dev-pilot-512` completed on 2026-10-06. Each variant evaluated the same 20 development questions against the full 37,511-passage corpus, with 50 candidates, five context passages and the same Gemma 4 26B-A4B generator at 512 output tokens. Each Laya on/off pair has identical candidate lists. The slice has ten factual and ten interpretation questions, and ten web and ten Wikipedia questions; it is not weighted like the complete development split.

Public artifacts: [report](../reports/matrix-dev-pilot-512/report.md), [summary](../reports/matrix-dev-pilot-512/summary.json), [provenance](../reports/matrix-dev-pilot-512/provenance.json), [per-query metrics](../reports/matrix-dev-pilot-512/per-query-metrics.jsonl). All scores below are means in [0, 1]; each row contains **20 completed outputs**.

| Retrieval | Laya | Recall@50 | Recall@5 | nDCG@10 | Answer token F1 | Length limits | Answer-cache hits |
|---|---|---:|---:|---:|---:|---:|---:|
| BM25 | Off | 0.9833 | 0.8667 | 0.8603 | 0.5100 | 0 | 0 |
| BM25 | On | 0.9833 | 0.6583 | 0.6283 | 0.4085 | 0 | 0 |
| BGE-M3 | Off | 1.0000 | 0.8833 | 0.8633 | 0.4661 | 1 | 0 |
| BGE-M3 | On | 1.0000 | 0.6917 | 0.6270 | 0.4409 | 0 | 0 |
| EmbeddingGemma 2 | Off | 0.9500 | 0.8167 | 0.8133 | 0.4592 | 1 | 1 |
| EmbeddingGemma 2 | On | 0.9500 | 0.6417 | 0.5736 | 0.4112 | 0 | 2 |
| BM25 + BGE-M3 | Off | 1.0000 | 0.9333 | 0.9217 | 0.5209 | 1 | 0 |
| BM25 + BGE-M3 | On | 1.0000 | 0.6250 | 0.6241 | 0.4126 | 0 | 2 |
| BM25 + EmbeddingGemma 2 | Off | 1.0000 | 0.9500 | 0.8854 | 0.5326 | 0 | 3 |
| BM25 + EmbeddingGemma 2 | On | 1.0000 | 0.5917 | 0.6083 | 0.3986 | 0 | 3 |

The two hybrids without Laya had the highest observed Recall@5 and F1. BM25+EmbeddingGemma 2 led those measures; BM25+BGE-M3 led nDCG@10. Laya reduced Recall@5, nDCG@10 and F1 in all five paired comparisons on this slice. These are descriptive findings from 20 questions with synthetic labels. The exported summary includes paired article-bootstrap intervals; many F1 intervals include zero, and this pilot does not establish a universal ranking or a final winner.

Three outputs reached the 512-token cap, all interpretation answers, and remain in all metrics. The earlier two-variant 512-token pilot's zero-truncation result applied only to that pair. The broader matrix shows that 512 does not guarantee complete answers for every context. Eleven identical generation requests were reused from cache, leaving 189 fresh calls for 200 scored outputs. Cache hits preserve answer/finish metadata and do not count as fresh generation timings.

Prompt suitability and output budgets need evaluation on broader development data before the final configuration is frozen. The current Laya prompt's declines are a reason to inspect its task fit; they are not evidence that all rerankers or all uses of Laya fail. Retrieval results and lexical answer overlap are separate outcomes; neither alone verifies factual accuracy.

The run's recorded session duration was **768.514 seconds (12 minutes 49 seconds)**, after both dense indexes had been built. This is a run-completion measurement, not an end-to-end request percentile or the cost of preparing the full corpus. The summary retains diagnostic stage timings with 19 retrieval observations per variant and 16–19 fresh-generation observations, excluding cache hits and the disclosed warm-up rows. These small samples are not production latency guarantees.

Provenance is fixed to source commit `e094aa4719a381ce9786e21d9503a2f61596ee76` and source hash `983fd297fcfc49976612e6c734e565da692d69538cc0f7661ab43321b2c47158`. Run fingerprint: `045252d34c401a7286e7c03cea4c7059b283219a02ad1a618d1aa8cb0830ba10`. The local server reported llama.cpp `b11451-2207c8e57`, with server fingerprint `862c23d56e3fbdcc8b2f01edf3885120b86baf93e165e133ca9bfe61979cf352`. Its verified Gemma Q4_0 GGUF SHA-256 is `3eca3b8f6d7baf218a7dd6bba5fb59a56ee25fe2d567b6f5f589b4f697eca51d`; context size is 8,192, temperature 0, seed 42, thinking disabled, and the output cap is 512 for every variant. Model revisions, package versions and dataset hashes are in the linked provenance artifact.

## Exploratory semantic evaluation through OpenRouter

The saved 200 pilot answers now have separate Jev correctness and grounding judgments. **All 319 distinct hosted requests succeeded**, covering 20 unique questions × ten variants × two judgment types, with identical payloads deduplicated. Requested model: `typesafe/jev-1.13`; served snapshot: `typesafe/jev-1.13-20260917`. RAG generation remained local and was not rerun.

| Retrieval | Jev correct without Laya | Jev correct with Laya |
|---|---:|---:|
| BM25 | 11/20 (55%) | 9/20 (45%) |
| BGE-M3 | 12/20 (60%) | 9/20 (45%) |
| EmbeddingGemma 2 | 14/20 (70%) | 8/20 (40%) |
| BM25 + BGE-M3 | 13/20 (65%) | 8/20 (40%) |
| BM25 + EmbeddingGemma 2 | 14/20 (70%) | 7/20 (35%) |

BM25 + EmbeddingGemma 2 retains its token F1 of **0.5326**; Jev classifies **14 answers as correct and six as partial**. F1 is lexical overlap, while this new percentage counts semantic labels. Pooled correctness across all variants is 105/200; that mixes different systems and is not any single system's success probability.

**These are uncalibrated model judgments.** In a verified Xperia/Z5 example, Jev missed an explicit source contradiction in the grounding pass. Separate synthetic controls also exposed uncertainty-handling errors. A blind assistant audit disagreed with some interpretation labels; human calibration is still pending. The 20-question slice and differences of one or two answers cannot establish a general winner or predict the probability that a new user question will be answered correctly.

Main judge usage was 761,303 input tokens, 20,523 output tokens and **US$0.031974726** reported by OpenRouter. Controls and diagnostics are separate. Public artifacts: [results and review notes](../reports/semantic-pilot-openrouter/review-notes.md), [summary](../reports/semantic-pilot-openrouter/summary.json), [per-answer judgments](../reports/semantic-pilot-openrouter/per-answer-judgments.jsonl), [synthetic controls](../reports/semantic-controls-tr/report.md). The [semantic protocol](semantic-evaluation.md) records the data boundaries, rubric and limitations. No final-test questions were evaluated.

## Blinded comparison with GPT 6.1 Sol Ultra

Three fresh Codex agents were explicitly configured with `gpt-6.1-sol` and reasoning effort `ultra`. One judged correctness and two judged grounding, with Jev labels and method identities withheld until verdicts were complete. All 319 unique benchmark judgments and 12 separate control judgments were completed; each reconstructed case input matches the original Jev input. No answer generation or Jev judging was rerun.

Across 200 answer rows, correctness-label agreement is **150/200 (75%)** and grounding agreement is **166/200 (83%)**. Unique-task agreement is **83/130 (63.85%)** for correctness and **158/189 (83.60%)** for grounding. Repeated answer rows across variants receive their normal per-variant weight; they are not independent questions.

Jev labeled 105 answers correct and Sol labeled 107; this pool spans ten systems and is not a deployment success rate. Sol also returned 19 unjudgeable labels. BM25 + EmbeddingGemma 2 without Laya received **14/20 correct from both judges, with only 11 answers jointly labeled correct**. Sol classified its other answers as four partial and two unjudgeable, and rated 13/20 both correct and supported.

The [full comparison](../reports/semantic-judge-comparison/analysis.tr.md), [summary](../reports/semantic-judge-comparison/summary.json) and [per-answer labels](../reports/semantic-judge-comparison/per-answer-comparison.jsonl) preserve disagreements. Sol matched 12/12 synthetic control expectations versus Jev's 10/12; these controls are not human calibration. Human adjudication remains pending. Sol processed grouped cases with persistent agent context, whereas Jev received isolated API requests; this is an exploratory agent-workflow comparison, not an identical-serving benchmark. The configured model/effort is recorded, but this workflow exposes no served snapshot, token usage or USD cost.

## Development retrieval baseline

Run `bm25-dev-2000` evaluated all 2,000 development questions against 37,511 passages, with no Laya or answer model. Metrics are means in [0, 1]. Public artifacts: [report](../reports/bm25-dev-2000/report.md), [summary](../reports/bm25-dev-2000/summary.json), [provenance](../reports/bm25-dev-2000/provenance.json), [per-query metrics](../reports/bm25-dev-2000/per-query-metrics.jsonl).

| Method | Questions | Recall@50 | Recall@5 | Hit@5 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 2,000 | 0.9350 | 0.8591 | 0.9180 | 0.8632 | 0.8434 |

Recall is the fraction of labeled relevant passages found; Hit@5 means at least one was found. They differ when a question has multiple gold passages. Retrieval p50 was 8.992 ms and p95 64.991 ms across 1,999 measured queries after excluding the first query. These are local stage timings, not end-to-end response latency or a comparison with dense retrieval. These metrics were reproduced on committed source `0b573f4`. The run fingerprint is `ec557103a353c35f81e4ed7b2642d77f36860178bbfdf4299075b4289cd040f1`.

## Twenty-question end-to-end pilot

Run `gemma-dev-pilot` evaluated the same 20 development questions with BM25 and BM25+Laya, followed by the same local Gemma 4 26B-A4B generator. Both variants completed 20/20 outputs with zero answer-cache hits. The selected slice contains ten factual and ten interpretation questions, and ten web and ten Wikipedia questions. This balanced pilot is not weighted like the complete development set.

Public artifacts retain the historical 256-token setting in their directory name: [report](../reports/gemma-dev-pilot-256/report.md), [summary](../reports/gemma-dev-pilot-256/summary.json), [provenance](../reports/gemma-dev-pilot-256/provenance.json), [per-query metrics](../reports/gemma-dev-pilot-256/per-query-metrics.jsonl). These exports exclude source text, questions, reference/generated answers, prompts and local paths.

| Variant | Recall@50 | Recall@5 | nDCG@10 | Answer EM | Answer token F1 | Length-limited answers |
|---|---:|---:|---:|---:|---:|---:|
| BM25 → Gemma 4 | 0.9833 | 0.8667 | 0.8603 | 0.0500 | 0.4998 | 5/20 |
| BM25 → Laya → Gemma 4 | 0.9833 | 0.6583 | 0.6283 | 0.0000 | 0.4033 | 3/20 |

The recorded candidate lists are identical across the two variants for each question. Laya reduced Recall@5 by 0.2083 and answer token F1 by 0.0965 on this slice. The article-bootstrap 95% intervals for those differences are respectively [-0.4333, 0.0250] and [-0.2075, 0.0269]. This small pilot supports inspecting the current reranking prompt and behavior on development data; it does not establish that Laya is generally worse, or that another retriever will behave the same way.

The 256-token output cap was reached by eight of the 40 answers, all interpretation answers: **8/20 interpretation outputs, or 40%**. They remain in the metrics. EM/F1 measure lexical overlap with synthetic references, not human-verified answer accuracy.

This development finding motivated a shared **512-token cap for every variant in the later full-matrix pilot**. The 256-token results above remain unchanged and labeled with their original setting. New results use a new run identity; the final test has not been consulted. Further prompt or budget changes must also be evaluated and disclosed on development data before final settings are frozen.

| Pilot stage | BM25 p50 / p95 | BM25+Laya p50 / p95 |
|---|---|---|
| Laya reranking | Not applicable | 0.997 s / 1.328 s |
| Fresh Gemma generation | 2.079 s / 3.903 s | 2.121 s / 3.810 s |

Each timing column contains 19 observations after the disclosed warm-up exclusions. Stage percentiles cannot be summed to obtain end-to-end percentiles. These are measurements from this pilot on the M4 Max host, not general serving benchmarks.

The pilot used llama.cpp `b11451-2207c8e57`, a verified Q4_0 GGUF, 8,192-token server context, temperature 0, seed 42, thinking disabled and a 256-token output limit. Run fingerprint: `de38b96ddcd74e2f25f6553cac022abb70207383433d1d0254cb5fde14139bb6`. The manifest records package versions, source hash, model revisions, GGUF checksum and server properties; the pilot's Git revision field was unset at execution, and its recorded source hash was subsequently verified to match commit `a0c3a76` exactly.

## Follow-up at 512 tokens

The [512-token pilot report](../reports/gemma-dev-pilot-512/report.md) records **40/40 completed answers, zero cache hits and zero length-limited answers**. BM25 answer EM/F1 were **0.0500 / 0.5100**; BM25+Laya were **0.0000 / 0.4085**. Retrieval results were unchanged. The paired F1 difference was −0.1015, with an article-bootstrap 95% interval of [-0.2159, 0.0274]. This remains a small development comparison.

The configured generation cap was the only experiment-setting change from the earlier pilot. Query IDs, candidate lists, final contexts and reported server identity match. The run therefore confirms that the higher cap removed observed truncation on these questions; the historical 256-token results remain visible. Its recorded Git revision is `0b573f4ea6c1a85b46bc3c297faccc8ccd1054b0`; later source changes do not rewrite that provenance. Public [summary](../reports/gemma-dev-pilot-512/summary.json), [provenance](../reports/gemma-dev-pilot-512/provenance.json) and [per-query metrics](../reports/gemma-dev-pilot-512/per-query-metrics.jsonl) are available.

**Timing caveat:** dense indexing was running concurrently with this follow-up. Its generation p50 values, 13.650 s for BM25 and 9.192 s for BM25+Laya, are therefore not a controlled speed comparison with the earlier pilot or between variants. Preserve these raw measurements as run evidence, and measure latency separately without competing indexing work before making speed claims.

Pilot stage timings are diagnostic: model checks and index preparation may overlap on the same GPU. They are not controlled comparative latency measurements. Final timing runs must use an otherwise idle inference workload.

## Audited public dataset

Preparation used `metunlp/ragturk` revision `86d35335d01b30869ab8eb392e380921fa1e2185`, split seed 42. The manifest fingerprint is `71289eaa5e6b4d6f5c8285ab0336fdc3568c1828972645d9b4456337b8326d1c`.

| Inventory item | Count |
|---|---:|
| Published article records examined | 8,105 |
| Usable articles | 8,104 |
| Wikipedia articles | 2,790 |
| Web articles reconstructed from text and character ranges | 5,314 |
| Searchable chunks | 37,511 |
| Raw question records | 14,534 |
| Usable questions with reference answers and linked evidence | 14,530 |
| Development questions | 2,000 |
| Test questions | 12,530 |
| Question groups used by the split | 8,058 |

Four malformed question rows and one article with a missing companion file were excluded. Valid passages from eight articles without a question file remain in the corpus. Upstream declared question counts disagree with parsed counts for 1,665 articles; parsed usable records determine the denominator. Of the usable questions, 8,951 come from web material and 5,579 from Wikipedia; 8,454 are labeled factual and 6,076 interpretation.

The split groups records connected by article ID, normalized exact question or normalized exact gold-evidence text. It does not guarantee the absence of semantic near-duplicates. Upstream web article metadata declares language `en` while Wikipedia declares `tr`; some language/model metadata conflicts with the paper. These values are preserved as a known source inconsistency rather than silently used to relabel or filter records. The preparation audit verifies parsing and links, not linguistic quality or human correctness of synthetic answers.

The local manifest records input inventory and output file hashes. Raw data and text-bearing predictions are not committed. This public snapshot is smaller than the 58,289 chunks and 20,459 questions described in the paper; the initial 20,000-question / 100,000-passage target is therefore not met by this source. A full ten-variant test on the available 12,530 questions produces **125,300 question–variant rows**. This row count is a workload size, not a measured result or a promise of distinct generation calls.

## Final-test matrix

The final-test table is deliberately empty of scores. A dash means **unmeasured**, not zero. Development results above are not copied into final-test cells.

| Retrieval | Laya | Questions completed | Recall@50 | Recall@5 | MRR | nDCG | Answer EM/F1 | Fresh-request latency |
|---|---|---:|---:|---:|---:|---:|---|---|
| BM25 | Off | — | — | — | — | — | — | — |
| BM25 | On | — | — | — | — | — | — | — |
| BGE-M3 | Off | — | — | — | — | — | — | — |
| BGE-M3 | On | — | — | — | — | — | — | — |
| EmbeddingGemma 2 | Off | — | — | — | — | — | — | — |
| EmbeddingGemma 2 | On | — | — | — | — | — | — | — |
| BM25 + BGE-M3 | Off | — | — | — | — | — | — | — |
| BM25 + BGE-M3 | On | — | — | — | — | — | — | — |
| BM25 + EmbeddingGemma 2 | Off | — | — | — | — | — | — | — |
| BM25 + EmbeddingGemma 2 | On | — | — | — | — | — | — | — |

When filling a table, state the exact metric cutoffs, query population and denominator; separate development and final test. Cache hits do not belong in a fresh-generation timing column.

## Interpretation

The [RAGTurk paper](https://aclanthology.org/2026.sigturk-1.15/) and earlier Jev experiments use different systems and controls. Their scores are not reproduced by copying them into this table. A local answer model, changed retrieval pool, different metrics or a different split makes a new experiment.

The [protocol](protocol.md) defines how to turn run artifacts into a defensible comparison. Report errors, truncated inputs and label coverage alongside successful results.

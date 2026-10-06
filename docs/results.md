# Results and validation status

**Development measurements are available; the full ten-variant comparison and final test are pending.** BM25 has completed all 2,000 development questions, and a two-variant pilot has produced 40 real Gemma 4 answers. These runs do not establish a general model winner.

## Evidence ledger

| Evidence | Current status | What it establishes |
|---|---|---|
| Dataset inventory and normalization | Prepared and manifest inspected, 2026-10-06 | 37,511 passages; 14,530 usable QA records; 2,000 development / 12,530 test |
| Infrastructure checks | Latest local suite: 55 passing tests; [earlier GitHub CI succeeded](https://github.com/erendikmenn/rag-benchmark/actions/runs/37523395067) | Code behavior and failure handling, not pretrained model quality |
| BM25 development baseline | Completed 2,000/2,000 real questions against all 37,511 passages | Full development retrieval metrics; no answer generation in this run |
| BGE-M3 and EmbeddingGemma 2 integration | Real FP32/MPS checks passed on a small Turkish fixture | Actual local encoding works; full-corpus retrieval comparison still pending |
| Local Laya integration | Real FP32/MPS fixture check and 20-question BM25 reranking pilot completed | Scoring/reranking works; observed quality is reported below |
| Local Gemma 4 integration | 40/40 pilot outputs completed, 20 questions × two variants | Real local generation with the pinned GGUF and verified server identity |
| Ten-variant development comparison | Dense index construction in progress | No complete ten-variant table yet |
| Frozen final test comparison | Not run/reported | No final leaderboard |

Update this ledger only from actual commands, manifests and inspected artifacts. Include the checked code/config revision and exact scope when changing a row.

The linked CI run completed successfully for commit `a0c3a760f01c70be05e971baa52830c46384eab3`, before the latest export and runtime checks; its 44-test scope is separate from the later 55-test local result. Local real-model checks are also separate from CI. The run reports, completion markers and summaries below were inspected; the full test split remains untouched.

## Development retrieval baseline

Run `bm25-dev-2000` evaluated all 2,000 development questions against 37,511 passages, with no Laya or answer model. Metrics are means in [0, 1]. Public artifacts: [report](../reports/bm25-dev-2000/report.md), [summary](../reports/bm25-dev-2000/summary.json), [provenance](../reports/bm25-dev-2000/provenance.json), [per-query metrics](../reports/bm25-dev-2000/per-query-metrics.jsonl).

| Method | Questions | Recall@50 | Recall@5 | Hit@5 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 2,000 | 0.9350 | 0.8591 | 0.9180 | 0.8632 | 0.8434 |

Recall is the fraction of labeled relevant passages found; Hit@5 means at least one was found. They differ when a question has multiple gold passages. Retrieval p50 was 1.676 ms and p95 2.211 ms across 1,999 measured queries after excluding the first query. These are local stage timings, not end-to-end response latency or a comparison with dense retrieval. The run fingerprint is `41696b1bbd15f3cb8c4be5b0195937e974db65037eafdc8e6bf785cd97387a53`.

## Twenty-question end-to-end pilot

Run `gemma-dev-pilot` evaluated the same 20 development questions with BM25 and BM25+Laya, followed by the same local Gemma 4 26B-A4B generator. Both variants completed 20/20 outputs with zero answer-cache hits. The selected slice contains ten factual and ten interpretation questions, and ten web and ten Wikipedia questions. This balanced pilot is not weighted like the complete development set.

Public artifacts retain the historical 256-token setting in their directory name: [report](../reports/gemma-dev-pilot-256/report.md), [summary](../reports/gemma-dev-pilot-256/summary.json), [provenance](../reports/gemma-dev-pilot-256/provenance.json), [per-query metrics](../reports/gemma-dev-pilot-256/per-query-metrics.jsonl). These exports exclude source text, questions, reference/generated answers, prompts and local paths.

| Variant | Recall@50 | Recall@5 | nDCG@10 | Answer EM | Answer token F1 | Length-limited answers |
|---|---:|---:|---:|---:|---:|---:|
| BM25 → Gemma 4 | 0.9833 | 0.8667 | 0.8603 | 0.0500 | 0.4998 | 5/20 |
| BM25 → Laya → Gemma 4 | 0.9833 | 0.6583 | 0.6283 | 0.0000 | 0.4033 | 3/20 |

The recorded candidate lists are identical across the two variants for each question. Laya reduced Recall@5 by 0.2083 and answer token F1 by 0.0965 on this slice. The article-bootstrap 95% intervals for those differences are respectively [-0.4333, 0.0250] and [-0.2075, 0.0269]. This small pilot supports inspecting the current reranking prompt and behavior on development data; it does not establish that Laya is generally worse, or that another retriever will behave the same way.

The 256-token output cap was reached by eight of the 40 answers, all interpretation answers: **8/20 interpretation outputs, or 40%**. They remain in the metrics. EM/F1 measure lexical overlap with synthetic references, not human-verified answer accuracy.

This development finding motivated a shared **512-token cap for every variant in the next full-matrix pilot**. The 256-token results above remain unchanged and labeled with their original setting. New results use a new run identity; the final test has not been consulted. Further prompt or budget changes must also be evaluated and disclosed on development data before final settings are frozen.

| Pilot stage | BM25 p50 / p95 | BM25+Laya p50 / p95 |
|---|---|---|
| Laya reranking | Not applicable | 0.997 s / 1.328 s |
| Fresh Gemma generation | 2.079 s / 3.903 s | 2.121 s / 3.810 s |

Each timing column contains 19 observations after the disclosed warm-up exclusions. Stage percentiles cannot be summed to obtain end-to-end percentiles. These are measurements from this pilot on the M4 Max host, not general serving benchmarks.

The pilot used llama.cpp `b11451-2207c8e57`, a verified Q4_0 GGUF, 8,192-token server context, temperature 0, seed 42, thinking disabled and a 256-token output limit. Run fingerprint: `de38b96ddcd74e2f25f6553cac022abb70207383433d1d0254cb5fde14139bb6`. The manifest records package versions, source hash, model revisions, GGUF checksum and server properties; the pilot's Git revision field is unset, so its source hash supplies the recorded code identity.

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

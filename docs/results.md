# Results and validation status

**No full real-model benchmark has been published.** The project is preparing a controlled ten-variant evaluation on RAGTurk. There is no supported winner, measured speedup or answer-quality claim yet.

## Evidence ledger

| Evidence | Current status | What it establishes |
|---|---|---|
| Dataset inventory and normalization | Prepared and manifest inspected, 2026-10-06 | 37,511 passages; 14,530 usable QA records; 2,000 development / 12,530 test |
| Infrastructure checks | Focused tests pass; validation is continuing | Code behavior and failure handling, not pretrained model quality |
| BM25 development smoke run | Completed 20/20 real questions against all 37,511 passages | Retrieval plumbing works on prepared data; no Laya or answer generation was exercised |
| Local embedding integration | Real-model validation pending | Backend availability alone is not retrieval performance |
| Local Laya integration | Real-model validation pending | A typed response alone is not a useful ranking |
| Local Gemma 4 integration | Real-model validation pending | One generated answer alone is not a benchmark |
| Ten-variant development comparison | Not reported | No development quality/latency table yet |
| Frozen final test comparison | Not run/reported | No final leaderboard |

Update this ledger only from actual commands, manifests and inspected artifacts. Include the checked code/config revision and exact scope when changing a row.

The initial BM25 check used retrieval-only mode on a 20-question development slice, with run fingerprint `85245eb8afcdedac08046ebebc241cff57643251afc2f92753a682fa72497d15`. Its completion and summary artifacts were inspected. This is a small integration check, not a final test result or a controlled model comparison.

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

## Planned result matrix

The table is deliberately empty of scores. A dash means **unmeasured**, not zero.

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

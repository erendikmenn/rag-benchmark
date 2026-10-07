# Clotho additional relevance: descriptions versus native audio

All **63 retrieval subsets × two candidate budgets = 126 conditions** completed on the separate **1,037-query / 1,045-clip / 3,116-positive-label** protocol. This reuses the same 1,045 source-only Gemma descriptions as the original Clotho experiment. It is not a second independent caption-generation experiment. No reranker or answer model is involved in these retrieval scores.

| Primary channel | Hit@5 | Recall@5 | nDCG@10 |
|---|---:|---:|---:|
| BM25 on Gemma descriptions | 6.56% | 2.17% | 0.0293 |
| BGE-M3 on Gemma descriptions | 8.29% | 2.89% | 0.0345 |
| EG2 on Gemma descriptions | 8.39% | 3.20% | 0.0398 |
| EG2 source audio | 25.84% | 11.32% | 0.1211 |
| CLAP source audio | 64.51% | 35.08% | 0.3771 |
| EG2 source audio + Gemma descriptions | 18.80% | 7.07% | 0.0775 |

Unlike the original single-positive protocol, queries here may have several relevant clips. Hit@5 means at least one relevant clip was found; Recall@5 measures the fraction of all labelled relevant clips recovered. They must not be equated.

**CLAP has the highest observed Hit@5 in both budgets across all 63 subsets.** Adding the generated description to EG2 audio changes Hit@5 from 25.84% to 18.80% (−7.04 percentage points), and Recall@5 from 11.32% to 7.07%. This fixed representation did not improve these measures; the cause and human caption correctness are unmeasured.

**No confidence intervals are reported:** an independently reconstructed connected component contains 897 of 1,037 queries (86.50%). There are 122 connected source groups, but one dominates. The eight paired comparisons retain descriptive differences only.

Independent CPU formulas reproduced **130,662 full rankings and 2,221,254 per-query metric values**, with exact source-channel/RRF IDs, scores and ranks. All 17 metrics support graded or multiple-positive labels. Six N/S/N+S baseline cells exactly reproduce the original additional-relevance run (6,222 rankings), so primary progress increases by **120 cells**. The 1,134 unsupported reranker cells are not complete.

All frozen source queries, qrels, actual media hashes, generator/cache keys and 1,045 description texts match the original source-only gallery. No caption inference was required for this second view. G/E per-input diagnostics remain unavailable, not measured zero truncation. Human description correctness and historical per-call audio sample retention remain unmeasured.

[Full grid](clotho-dcase2025-additional-relevance-gemma-described-all-retrieval/report.md) · [Descriptive paired comparisons](clotho-dcase2025-additional-relevance-gemma-described-all-retrieval/paired-comparisons.md) · [Independent audit](clotho-dcase2025-additional-relevance-gemma-described-all-retrieval/independent-audit.json) · [Shared description audit](clotho-description-audit.md) · [Aggregate source hashes](clotho-additional-description-retrieval-summary.json)

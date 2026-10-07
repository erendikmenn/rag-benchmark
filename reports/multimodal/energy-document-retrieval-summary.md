# French Energy document retrieval

All **126 retrieval conditions** are complete on **308 queries and 2,225 pages**, with 1,103 positive query–page labels. Six individual channels and all their nonempty equal-weight RRF combinations are measured under two candidate budgets. No reranker or answer generator is used.

| Standalone method | Hit@5 | Recall@5 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|
| BM25 page text | 77.27% | 54.65% | 0.6225 | 0.5568 |
| BGE-M3 page text | 75.32% | 50.70% | 0.6062 | 0.5342 |
| EG2 page text | 82.79% | 56.85% | 0.6504 | 0.5732 |
| EG2 page image | 68.83% | 44.65% | 0.5047 | 0.4461 |
| ColQwen page image | 86.69% | 62.25% | 0.7002 | 0.6465 |
| EG2 page image + text | 83.77% | 57.80% | 0.6607 | 0.5829 |

Hit@5 asks whether at least one labelled page appears among the first five results. Recall@5 measures the fraction of all labelled pages recovered there. They are different on this multi-positive task.

The primary condition permits up to 100 candidates per channel; the control splits a total budget of 100 between selected channels before deduplication. Fusion uses RRF k=60. Exact ties and all variant/budget metrics remain in the full report. The following maxima are observed on this test collection, not settings tuned on separate development data.

- Highest observed hit@5: **0.886364**, shared by: `document__b_e_s_j__none`, `document__b_g_e_s_j__none`.
- Highest observed recall@5: **0.623162**, shared by: `document__s_j__none`.
- Highest observed ndcg@10: **0.656485**, shared by: `document__b_s_j__none`.

Independent formulas and source-channel reconstruction verified **38,808 query rankings and 659,736 metric values** across all 126 cells. Both existing BM25 cells match exactly, so this publication adds **124** main-matrix cells. Reading and validating completed caches is not new model extraction; any actual new model call remains fresh inference.

The ranking audit reconstructs combinations from saved channel results and independently checks all metric formulas; it does not rerun base encoders. A separate independent verifier reproduced all 16 metric contrasts and 80,000 bootstrap draws, including interval bounds, means and win/loss/tie counts. The paired analysis resamples **26 connected source-document groups**, with the largest containing 80/308 queries (25.97%). It uses 5,000 bootstrap draws and seed 42; residual dependencies and multiple comparisons limit interpretation. G/E per-input diagnostics may be unavailable; that does not establish zero truncation or zero extraction time. Visual token budgets remain model-specific, so these are recorded-setting quality comparisons, not equal-compute measurements.

[Full 126-cell report](vidore-v3-energy-fr-all-retrieval/report.md) · [Paired comparisons](vidore-v3-energy-fr-all-retrieval/paired-comparisons.md) · [Independent ranking audit](vidore-v3-energy-fr-all-retrieval/independent-audit.json) · [Independent interval audit](vidore-v3-energy-fr-all-retrieval/paired-independent-audit.json) · [Machine-readable summary](energy-document-retrieval-summary.json)

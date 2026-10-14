# Physics FR: full CUDA retrieval results

All **302 queries** search **1,674 pages**, with **2,178 positive graded labels**. All **126 no-reranker conditions** completed on A100-SXM4-80GB / CUDA BF16. These separately labelled CUDA results add no cells to the existing MPS-derived main count in this publication.

| Channel | Method | Hit@5 | Recall@5 | nDCG@10 |
|---|---|---:|---:|---:|
| B | BM25 page text | 72.52% | 30.57% | 0.3904 |
| G | BGE-M3 page text | 70.53% | 29.87% | 0.3818 |
| E | EG2 page text | 76.16% | 33.14% | 0.4168 |
| N | EG2 page image | 73.84% | 33.01% | 0.4059 |
| S | ColQwen page image | 80.46% | 37.23% | 0.4854 |
| J | EG2 page image + text | 74.50% | 32.01% | 0.3976 |

Standalone ColQwen has the highest observed Hit@5, Recall@5 and nDCG@10 among all 63 channel subsets in the primary per-channel budget. This statement describes the frozen sample; it is not a claim of general superiority.

Independent audit reproduced all **38,052 rankings and 646,884 metric values** using source-channel reconstruction and independent graded formulas. A separate check reproduced all 16 paired metric contrasts and 80,000 bootstrap metric evaluations. The two existing BM25 conditions match all 604 query results across the recorded environments; neural device equivalence was not assumed.

The paired analysis resamples 19 positive-source connected components, with the largest holding 148/302 queries (49.01%). It uses 5,000 samples and seed42 for each metric contrast. Intervals remain exploratory, unadjusted for multiple comparisons, and sensitive to source concentration; arithmetic agreement does not establish independence. The seeded selections are reused, so 80,000 metric evaluations are not 80,000 independent experiments.

Image-plus-text EG2 minus image-only EG2 is +0.66 percentage points Hit@5, with a 95% exploratory interval [-6.81, +4.92] points. Its nDCG@10 difference is -0.0083, interval [-0.0259, +0.0246]. Both intervals include zero.

The complete immutable checkpoint was downloaded and SHA-verified before the original A100 runtime disappeared. Publication and audits use that preserved result; they do not run inference again. Model-specific visual budgets are retained, and unavailable per-input diagnostics are not measured zeros. These scores evaluate retrieval against dataset labels, not generated-answer correctness.

[Full report](vidore-v3-physics-fr-cuda-all-retrieval/report.md) · [Paired comparisons](vidore-v3-physics-fr-cuda-all-retrieval/paired-comparisons.md) · [Ranking audit](vidore-v3-physics-fr-cuda-all-retrieval/independent-audit.json) · [Paired audit](vidore-v3-physics-fr-cuda-all-retrieval/paired-independent-audit.json) · [Aggregate JSON](physics-retrieval-summary.json)

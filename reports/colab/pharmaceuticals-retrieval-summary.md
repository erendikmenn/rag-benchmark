# Pharmaceuticals EN: full recovery CUDA retrieval results

All **364 queries** search **2,313 pages**, with **1,732 positive graded page labels**. All **126 no-reranker conditions** completed on **NVIDIA L4 / CUDA BF16**. These separately labelled recovery results add **zero cells** to the existing main count.

| Channel | Method | Hit@5 | Recall@5 | nDCG@10 |
|---|---|---:|---:|---:|
| B | BM25 page text | 81.59% | 50.84% | 0.5529 |
| G | BGE-M3 page text | 82.97% | 49.38% | 0.5502 |
| E | EG2 page text | 84.62% | 52.82% | 0.5783 |
| N | EG2 page image | 81.87% | 49.80% | 0.5521 |
| S | ColQwen page image | 89.56% | 58.04% | 0.6490 |
| J | EG2 page image + text | 85.99% | 53.48% | 0.6025 |

Independent audit reproduced **45,864 rankings and 779,688 metric values** across all queries and both budgets. Source-channel/RRF reconstruction and independent graded formulas agreed. The two BM25 baseline conditions matched all **728 query results** across the recorded environments; neural device equivalence was not assumed.

An independent BFS reconstructed **23 positive-source connected components** and **50 source document units**, with a largest group of **70/364 queries (19.23%)**. All eight standard document comparisons and their 16 metric contrasts matched independent query-weighted cluster-bootstrap arithmetic: 5,000 samples and seed42 per contrast, or 80,000 metric evaluations. Intervals remain exploratory and unadjusted for multiple comparisons; source exchangeability and hidden dependencies are not established by arithmetic agreement. Seeded selections are reused, so these are not 80,000 independent experiments.

The complete recovery checkpoint (**1,417,523,200 bytes**) was received and SHA-verified before these CPU-only audits. The historical original run took **1,880.183 seconds** and its runtime disappeared before a durable full backup was confirmed; the fresh recovery run took **1,867.651 seconds**. These are distinct runs of the **same frozen Pharmaceuticals dataset**, not independent datasets. The recovered result supplies this report.

Visual budgets remain model-specific. Missing per-input diagnostics are not measured zeros. Retrieval scores evaluate source relevance against dataset labels, not generated-answer correctness. Full BGE/Laya reranking and QA are separate work.

[Full report](vidore-v3-pharmaceuticals-en-cuda-all-retrieval/report.md) · [Paired comparisons](vidore-v3-pharmaceuticals-en-cuda-all-retrieval/paired-comparisons.md) · [Ranking audit](vidore-v3-pharmaceuticals-en-cuda-all-retrieval/independent-audit.json) · [Paired audit](vidore-v3-pharmaceuticals-en-cuda-all-retrieval/paired-independent-audit.json) · [Aggregate JSON](pharmaceuticals-retrieval-summary.json)

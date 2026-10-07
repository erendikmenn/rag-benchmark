# Laya document reranking: complete grid

ViDoRe V3 computer science: **215 queries / 1,360 pages**, with the same frozen query IDs and labels.
**378 Laya cells** cover all 63 retrieval subsets × K=20/50/100 × two candidate budgets. The snapshot also retains 126 retrieval-only cells. BGE and full Gemma reranking are not completed in this snapshot.

At primary K=50 / per-channel budget, Laya lowered Hit@5, Recall@5 and nDCG@10 in **all 63 matched comparisons**.
Laya receives the query and source page text; it does not inspect page images. This is an observed result for this checkpoint, prompt and dataset, not a diagnosis of why the scores fell.

| Retrieval | Hit@5 before → Laya | Recall@5 before → Laya | nDCG@10 before → Laya |
|---|---:|---:|---:|
| BM25 | 92.09% → 51.16% | 53.53% → 19.47% | 0.6334 → 0.2638 |
| BGE-M3 text | 95.35% → 54.88% | 52.45% → 20.24% | 0.6356 → 0.2829 |
| EG2 text | 94.42% → 56.28% | 54.40% → 20.63% | 0.6623 → 0.2873 |
| EG2 image | 91.63% → 55.35% | 52.70% → 19.87% | 0.6296 → 0.2620 |
| ColQwen image | 97.21% → 54.42% | 65.34% → 20.43% | 0.7585 → 0.2688 |
| EG2 image + text | 94.42% → 55.35% | 54.54% → 20.55% | 0.6753 → 0.2776 |
| All six channels | 96.74% → 54.88% | 61.24% → 20.33% | 0.7351 → 0.2774 |

Changing rerank K also changes the candidates scored. Ranges below describe the 63 per-channel retrieval methods at each fixed K; they are not confidence intervals.

| Rerank K | Hit@5 range | Recall@5 range | nDCG@10 range |
|---:|---:|---:|---:|
| 20 | 72.56%–78.60% | 31.39%–34.34% | 0.4077–0.4489 |
| 50 | 51.16%–57.67% | 19.26%–21.14% | 0.2620–0.2883 |
| 100 | 42.33%–47.91% | 14.65%–16.58% | 0.1889–0.2040 |

All 215 queries connect into one source-document group. Confidence intervals are withheld; the 63 same-query contrasts are descriptive. This evaluates source retrieval, not generated-answer correctness.

Laya checkpoint `convaiinnovations/laya-multilingual@1720e3e3357cfe1e281542e223f8273b0890ca34`: SDK backend, MPS FP32, batch 8, maximum length 8,192, threshold disabled. Per-input token telemetry is unavailable; missing diagnostics are not a measured zero-truncation result.

An independent CPU audit verified the positive-class score direction and candidate mapping. Replaying the actual tokenizer on the 10,750 ColQwen/K=50 query–page pairs retained every input, with a maximum of 2,061/8,192 tokens. This checks that condition only; it does not identify the cause of the performance loss. [Integration audit](vidore-v3-computer_science-en-laya-complete/integration-audit.json).

[All measured cells and frozen snapshot](vidore-v3-computer_science-en-laya-complete/report.md) · [63 paired comparisons](vidore-v3-computer_science-en-laya-complete/paired-comparisons.md) · [Exact summary and source hashes](laya-document-summary.json)

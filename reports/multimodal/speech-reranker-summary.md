# Turkish speech: complete Laya and BGE reranking grids

All **329 transcript queries and 743 recordings** are retained. **434 completed cells:** 62 retrieval-only, 186 Laya and 186 BGE. Each reranker covers 31 retrieval subsets × K=20/50/100 × two budgets. Full Gemma is a separate pending job.

Queries equal reference transcript text. Both rerankers only see source-only Whisper transcript text, including for native-audio retrieval. Transcript grouping may not cover speaker/topic dependence. Primary contrasts and parameter choices are exploratory. No semantic question-answer correctness is measured.

The primary table uses K=50 and per-channel candidate budgets. Each entry is **Hit@5 / Recall@5 / nDCG@10**.

| Retrieval | None | Laya | BGE reranker |
|---|---:|---:|---:|
| BM25 ASR | 100.00% / 100.00% / 1.0000 | 100.00% / 99.24% / 0.9880 | 100.00% / 100.00% / 1.0000 |
| BGE-M3 ASR | 100.00% / 100.00% / 1.0000 | 99.09% / 98.63% / 0.9837 | 100.00% / 100.00% / 1.0000 |
| EG2 ASR | 100.00% / 100.00% / 1.0000 | 99.39% / 98.63% / 0.9811 | 100.00% / 100.00% / 1.0000 |
| EG2 audio | 100.00% / 99.54% / 0.9970 | 100.00% / 99.49% / 0.9874 | 100.00% / 100.00% / 1.0000 |
| EG2 audio + ASR | 100.00% / 100.00% / 1.0000 | 99.70% / 99.19% / 0.9844 | 100.00% / 100.00% / 1.0000 |
| All five channels | 100.00% / 100.00% / 1.0000 | 99.39% / 98.89% / 0.9842 | 100.00% / 100.00% / 1.0000 |

The audit reproduced **142,786 stored rankings and 2,427,362 metric values**, validated identity-bound pair-score caches and preserved all 62 previously published retrieval baselines. A separate audit using independently written formulas reproduced all 17 per-query metrics. Reranker conditions overlap in candidates and queries; they are not independent samples.

[93 primary paired comparisons](fleurs-tr_tr-whisper-text-rerankers/paired-comparisons.md) use 329 transcript groups, 5,000 bootstrap samples and seed 42. Intervals are marginal and exploratory; additional speaker/topic dependence is not represented.

[Full report](fleurs-tr_tr-whisper-text-rerankers/report.md) · [Exact primary contrasts and source hashes](speech-reranker-summary.json)

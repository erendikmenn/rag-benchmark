# Document reranking: complete Laya and BGE grids

ViDoRe V3 computer science: **215 queries, 1,360 pages and 1,049 positive query–page labels**. All methods use the same frozen queries, gallery and graded labels.

**882 completed cells:** 126 retrieval-only + 378 Laya + 378 BGE. Each reranker covers all 63 nonempty retrieval subsets × K=20/50/100 × two candidate budgets. There are no failed cells for these methods. The full Gemma grid is unsupported in this run; its separate five-query technical smoke is not full coverage.

At primary K=50, BGE raises nDCG@10 in 60/63 matched retrieval conditions and lowers it in 3/63. ColQwen without reranking still has higher nDCG@10 than every BGE/K=50 condition. BGE exceeds Laya in all 63 primary conditions on Hit@5, Recall@5 and nDCG@10.

Both rerankers score query text against fixed source page text, including two published empty page texts. They do not inspect page images. The following primary table uses **K=50 and the per-channel candidate budget**. Each entry lists **Hit@5 / Recall@5 / nDCG@10**. Hit requires at least one positive page; Recall measures the fraction of all positive pages retrieved, while nDCG uses graded relevance.

| Retrieval | None | Laya | BGE reranker |
|---|---:|---:|---:|
| BM25 | 92.09% / 53.53% / 0.6334 | 51.16% / 19.47% / 0.2638 | 96.28% / 62.55% / 0.7227 |
| BGE-M3 text | 95.35% / 52.45% / 0.6356 | 54.88% / 20.24% / 0.2829 | 97.21% / 64.35% / 0.7360 |
| EG2 text | 94.42% / 54.40% / 0.6623 | 56.28% / 20.63% / 0.2873 | 97.21% / 64.15% / 0.7369 |
| EG2 image | 91.63% / 52.70% / 0.6296 | 55.35% / 19.87% / 0.2620 | 97.21% / 64.04% / 0.7370 |
| ColQwen image | 97.21% / 65.34% / 0.7585 | 54.42% / 20.43% / 0.2688 | 96.74% / 64.08% / 0.7419 |
| EG2 image + text | 94.42% / 54.54% / 0.6753 | 55.35% / 20.55% / 0.2776 | 97.21% / 64.37% / 0.7402 |
| All six channels | 96.74% / 61.24% / 0.7351 | 54.88% / 20.33% / 0.2774 | 96.74% / 64.02% / 0.7416 |

At primary K=50, the direction counts below compare each of the 63 subsets with its own matched comparator. Counts are across conditions, not independent datasets or significance tests.

| Paired contrast | Metric | Higher | Lower | Equal |
|---|---|---:|---:|---:|
| bge minus none | hit@5 | 34 | 15 | 14 |
| bge minus none | recall@5 | 62 | 1 | 0 |
| bge minus none | ndcg@10 | 60 | 3 | 0 |
| bge minus laya | hit@5 | 63 | 0 | 0 |
| bge minus laya | recall@5 | 63 | 0 | 0 |
| bge minus laya | ndcg@10 | 63 | 0 | 0 |
| laya minus none | hit@5 | 0 | 63 | 0 |
| laya minus none | recall@5 | 0 | 63 | 0 |
| laya minus none | ndcg@10 | 0 | 63 | 0 |

Changing K also changes the scored candidate set. The ranges below span all 63 retrieval subsets within each fixed budget and K; they are **not confidence intervals**.

| Budget | Reranker | K | Hit@5 range | Recall@5 range | nDCG@10 range |
|---|---|---:|---:|---:|---:|
| per_channel | laya_text | 20 | 72.56%–78.60% | 31.39%–34.34% | 0.4077–0.4489 |
| per_channel | laya_text | 50 | 51.16%–57.67% | 19.26%–21.14% | 0.2620–0.2883 |
| per_channel | laya_text | 100 | 42.33%–47.91% | 14.65%–16.58% | 0.1889–0.2040 |
| per_channel | bge_reranker_text | 20 | 93.95%–97.21% | 60.86%–64.96% | 0.7024–0.7547 |
| per_channel | bge_reranker_text | 50 | 96.28%–97.21% | 62.55%–64.49% | 0.7227–0.7425 |
| per_channel | bge_reranker_text | 100 | 96.74%–97.21% | 63.13%–64.26% | 0.7277–0.7412 |
| total | laya_text | 20 | 72.09%–79.53% | 30.84%–35.93% | 0.4077–0.4551 |
| total | laya_text | 50 | 51.16%–62.79% | 19.47%–23.51% | 0.2620–0.3259 |
| total | laya_text | 100 | 42.33%–62.79% | 14.65%–23.45% | 0.1889–0.3213 |
| total | bge_reranker_text | 20 | 93.95%–97.21% | 60.86%–64.91% | 0.7024–0.7526 |
| total | bge_reranker_text | 50 | 96.28%–97.21% | 62.55%–64.66% | 0.7227–0.7446 |
| total | bge_reranker_text | 100 | 96.74%–97.21% | 63.13%–64.62% | 0.7277–0.7435 |

All 215 queries connect into **one source-document group** through two source documents. Confidence intervals are withheld. Paired differences are descriptive; leaders among the 63 subsets and K values are exploratory. This evaluates retrieval, not generated-answer correctness.

BGE uses `BAAI/bge-reranker-v2-m3@953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`, MPS bfloat16, batch 1, maximum length 8,192. Laya uses `convaiinnovations/laya-multilingual@1720e3e3357cfe1e281542e223f8273b0890ca34`, MPS float32, SDK batch 8, maximum length 8,192 and no threshold. Missing per-input token telemetry is not a measured zero-truncation result. The earlier Laya tokenizer audit covers ColQwen/K=50 only.

The CPU audit recomputed **189,630 stored query rankings and 3,223,710 metric values**, checked full query coverage, candidate IDs, score order and identity-bound score caches, and matched the public export. No new model inference was performed. Historical stored-result score counts and the latest resume's cache hits are distinguished in the JSON.

The [earlier Laya milestone](vidore-v3-computer_science-en-laya-complete/report.md) is retained: its 504 completed cells overlap this result and are not additional unique experiments. Their metrics and the retrieval-only baseline remain identical.

[Full report](vidore-v3-computer_science-en-text-rerankers/report.md) · [Exact aggregate summary, paired differences and source hashes](document-reranker-summary.json)

# English Finance document retrieval

All 126 retrieval conditions completed on 309 queries and 2,942 pages, with 1,461 positive page labels (400 grade 2 and 1,061 grade 1). All 63 nonempty channel subsets are measured under two candidate budgets. These are retrieval scores, not generated-answer correctness.

| Standalone method | Hit@5 | Recall@5 | nDCG@10 |
|---|---:|---:|---:|
| BM25 page text | 78.96% | 46.95% | 0.5313 |
| BGE-M3 page text | 70.23% | 37.58% | 0.4387 |
| EG2 page text | 86.08% | 52.94% | 0.5583 |
| EG2 page image | 77.67% | 43.32% | 0.4795 |
| ColQwen page image | 87.70% | 54.82% | 0.6170 |
| EG2 page image + text | 85.11% | 51.36% | 0.5716 |

Hit@5 checks whether at least one labelled page is retrieved in five results; Recall@5 measures the fraction of all labelled pages recovered. Graded nDCG uses relevance grades.

| Best fusion by observed Hit@5 | Channels | Hit@5 | Recall@5 | nDCG@10 |
|---|---|---:|---:|---:|
| per_channel | S+J | 91.59% | 59.82% | 0.6578 |
| total | B+E+S+J | 91.59% | 58.77% | 0.6406 |

Independent source-channel/RRF and metric formulas verified 38,934 rankings and 661,878 values across all 17 metrics. Both prior BM25 cells match exactly (618 query results), leaving 124 new main-matrix cells after overlap.

The primary candidate budget permits 100 results per channel; the total-budget control divides 100 across selected channels before deduplication. RRF uses k=60. The highest observed fusion is descriptive, not a significance claim.

All 309 queries belong to ONE connected positive source-document component spanning six documents. Confidence intervals are withheld; no query-independent bootstrap fallback is used. This grouping does not establish repository independence. Visual budgets differ by model, and missing diagnostics are unavailable rather than zero.

[Full report](vidore-v3-finance_en-en-all-retrieval/report.md) · [Independent audit](vidore-v3-finance_en-en-all-retrieval/independent-audit.json)

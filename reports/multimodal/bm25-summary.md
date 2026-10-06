# Full-split BM25 baselines

All 14 prepared code/document collections completed: **54,980 queries**. This milestone covers lexical retrieval only; dense models, fusion and rerankers have separate execution states.

Only positive lexical matches are returned (`positive_scores_only_v1`); unmatched queries keep empty results. Code uses `code-identifiers-v1`; documents use `turkish-unicode-v1`. No query-specific vocabulary, captions or gold answers are indexed.

| Collection | Queries | Candidates | Hit@5 | Recall@5 | nDCG@10 |
|---|---:|---:|---:|---:|---:|
| [codesearchnet-go-test-bm25](codesearchnet-go-test-bm25/report.md) | 8,122 | 28,120 | 0.6194 | 0.6194 | 0.5495 |
| [codesearchnet-java-test-bm25](codesearchnet-java-test-bm25/report.md) | 10,955 | 40,347 | 0.3942 | 0.3942 | 0.3323 |
| [codesearchnet-javascript-test-bm25](codesearchnet-javascript-test-bm25/report.md) | 3,291 | 13,981 | 0.3662 | 0.3662 | 0.3124 |
| [codesearchnet-php-test-bm25](codesearchnet-php-test-bm25/report.md) | 14,014 | 52,660 | 0.3327 | 0.3327 | 0.2797 |
| [codesearchnet-python-test-bm25](codesearchnet-python-test-bm25/report.md) | 14,918 | 43,827 | 0.3893 | 0.3893 | 0.3318 |
| [codesearchnet-ruby-test-bm25](codesearchnet-ruby-test-bm25/report.md) | 1,261 | 4,360 | 0.4560 | 0.4560 | 0.3852 |
| [vidore-v3-computer_science-en-bm25](vidore-v3-computer_science-en-bm25/report.md) | 215 | 1,360 | 0.9209 | 0.5353 | 0.6334 |
| [vidore-v3-energy-fr-bm25](vidore-v3-energy-fr-bm25/report.md) | 308 | 2,225 | 0.7727 | 0.5465 | 0.5568 |
| [vidore-v3-finance_en-en-bm25](vidore-v3-finance_en-en-bm25/report.md) | 309 | 2,942 | 0.7896 | 0.4695 | 0.5313 |
| [vidore-v3-finance_fr-fr-bm25](vidore-v3-finance_fr-fr-bm25/report.md) | 320 | 2,384 | 0.5531 | 0.3141 | 0.3483 |
| [vidore-v3-hr-en-bm25](vidore-v3-hr-en-bm25/report.md) | 318 | 1,110 | 0.7862 | 0.4297 | 0.4984 |
| [vidore-v3-industrial-en-bm25](vidore-v3-industrial-en-bm25/report.md) | 283 | 5,244 | 0.6855 | 0.4158 | 0.4615 |
| [vidore-v3-pharmaceuticals-en-bm25](vidore-v3-pharmaceuticals-en-bm25/report.md) | 364 | 2,313 | 0.8159 | 0.5084 | 0.5529 |
| [vidore-v3-physics-fr-bm25](vidore-v3-physics-fr-bm25/report.md) | 302 | 1,674 | 0.7252 | 0.3057 | 0.3904 |

Hit@5 means at least one labelled relevant item in the first five. Recall@5 measures the fraction of all labelled relevant items; code has one positive per query, while document queries may have several. Collections differ in task and difficulty, so no pooled success percentage is reported. These are not generated-answer scores.

The [machine-readable summary](bm25-summary.json) records each source report hash. Earlier zero-score-eligible baselines remain historical and are not counted here.

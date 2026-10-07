# Code dense retrieval comparisons

2 complete language collections, 16,179 queries and 28 method/budget cells. All seven nonempty combinations of BM25, BGE-M3 dense and EmbeddingGemma 2 text retrieval are measured. No reranker or answer generator is used.

Candidates contain cleaned source code, without the reference docstring/comments. Official English queries and the frozen gallery are fixed within each language. Results from these test collections are observations, not development-set tuning or a claim of general code correctness.

Both text encoders use MPS/BF16, batch size 8, complete-input context checks and normalized FP32 retrieval vectors. BGE-M3 uses 1,024 dimensions; EG2 uses 768 and its general search-result query prefix, not the separately planned code-specific prompt. Per-item encoding diagnostics were unavailable for these text adapters; independent corpus preflight and runtime hard guards checked context limits. Exact model revisions and prompts remain in each source report.

Fusion uses equal-weight RRF (k=60). The primary per-channel condition takes up to 100 results per channel. The control divides a total budget of 100 across channels before deduplication, without filling missing or duplicate candidates. BM25 keeps positive lexical matches only.

## Python: 14,918 queries / 43,827 functions

| Method | Hit@5 per-channel | Hit@5 total-100 | MRR@10 per-channel | nDCG@10 per-channel |
|---|---:|---:|---:|---:|
| BM25 | 38.93% | 38.93% | 0.2931 | 0.3318 |
| BGE-M3 | 62.18% | 62.18% | 0.4982 | 0.5456 |
| EmbeddingGemma 2 | 84.46% | 84.46% | 0.7224 | 0.7641 |
| BM25 + BGE-M3 | 59.41% | 60.05% | 0.4618 | 0.5135 |
| BM25 + EG2 | 69.25% | 72.43% | 0.5380 | 0.6011 |
| BGE-M3 + EG2 | 76.77% | 76.91% | 0.6378 | 0.6842 |
| BM25 + BGE-M3 + EG2 | 73.56% | 75.26% | 0.5807 | 0.6372 |

[Full report](codesearchnet-python-test-dense-fusions/report.md) · [Paired comparisons](codesearchnet-python-test-dense-fusions/paired-comparisons.md)

## Ruby: 1,261 queries / 4,360 functions

| Method | Hit@5 per-channel | Hit@5 total-100 | MRR@10 per-channel | nDCG@10 per-channel |
|---|---:|---:|---:|---:|
| BM25 | 45.60% | 45.60% | 0.3441 | 0.3852 |
| BGE-M3 | 68.20% | 68.20% | 0.5554 | 0.5995 |
| EmbeddingGemma 2 | 86.28% | 86.28% | 0.7551 | 0.7910 |
| BM25 + BGE-M3 | 64.71% | 64.31% | 0.5190 | 0.5665 |
| BM25 + EG2 | 70.74% | 72.40% | 0.5723 | 0.6254 |
| BGE-M3 + EG2 | 78.75% | 78.59% | 0.6726 | 0.7145 |
| BM25 + BGE-M3 + EG2 | 74.94% | 76.61% | 0.6112 | 0.6628 |

[Full report](codesearchnet-ruby-test-dense-fusions/report.md) · [Paired comparisons](codesearchnet-ruby-test-dense-fusions/paired-comparisons.md)

In the Python and Ruby collections, standalone EG2 has the highest observed Hit@5 among the completed methods. Equal-weight fusion with the other channels lowers its score here; this does not establish that every fusion strategy is worse. The paired intervals in the linked reports group shared source functions, not whole repositories. They are exploratory marginal intervals, without multiple-comparison correction.

These scores measure finding labelled source functions, not generating correct code or answering a question. Remaining language/model/reranker conditions are not included until completed. [Machine-readable summary](code-dense-summary.json).

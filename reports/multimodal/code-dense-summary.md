# Code dense retrieval comparisons

4 complete language collections, 35,256 queries and 56 method/budget cells. All seven nonempty combinations of BM25, BGE-M3 dense and EmbeddingGemma 2 text retrieval are measured. No reranker or answer generator is used.

Candidates contain cleaned source code, without the reference docstring/comments. Official English queries and the frozen gallery are fixed within each language. Results from these test collections are observations, not development-set tuning or a claim of general code correctness.

Both text encoders use MPS/BF16, batch size 8, complete-input context checks and normalized FP32 retrieval vectors. BGE-M3 uses 1,024 dimensions; EG2 uses 768 and its general `task: search result | query:` prefix, not the separately planned code-specific prompt. Per-item encoding diagnostics were unavailable for these text adapters; independent corpus preflight and runtime hard guards checked context limits. Missing diagnostics are not a measured zero-truncation result. Exact model revisions remain in each source report.

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

Primary EG2 − BGE-M3: Hit@5 **+22.28 percentage points** (95% paired source-function interval [21.56, 23.01]); nDCG@10 **+0.2185** [0.2126, 0.2245]. The analysis uses 14,918 labelled source-function groups.

Gallery scope: **43,827 cleaned functions** from 54,654 upstream `codebase.txt` URL lines; **10,827 URLs are absent from the raw validation/test archives**. The manifest's selection rule is: “Official preprocess.py: emit codebase.txt URL only if present in raw validation/test”. This reproduces the published cleaned-gallery count; it does not cover all original CodeSearchNet functions. All 14,918 official test queries remain included.

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

Primary EG2 − BGE-M3: Hit@5 **+18.08 percentage points** (95% paired source-function interval [15.94, 20.30]); nDCG@10 **+0.1915** [0.1734, 0.2093]. The analysis uses 1,261 labelled source-function groups.

Gallery scope: **4,360 cleaned functions** from 5,914 upstream `codebase.txt` URL lines; **1,554 URLs are absent from the raw validation/test archives**. The manifest's selection rule is: “Official preprocess.py: emit codebase.txt URL only if present in raw validation/test”. This reproduces the published cleaned-gallery count; it does not cover all original CodeSearchNet functions. All 1,261 official test queries remain included.

[Full report](codesearchnet-ruby-test-dense-fusions/report.md) · [Paired comparisons](codesearchnet-ruby-test-dense-fusions/paired-comparisons.md)

## Go: 8,122 queries / 28,120 functions

| Method | Hit@5 per-channel | Hit@5 total-100 | MRR@10 per-channel | nDCG@10 per-channel |
|---|---:|---:|---:|---:|
| BM25 | 61.94% | 61.94% | 0.5038 | 0.5495 |
| BGE-M3 | 87.70% | 87.70% | 0.7677 | 0.8029 |
| EmbeddingGemma 2 | 96.28% | 96.28% | 0.9054 | 0.9232 |
| BM25 + BGE-M3 | 85.71% | 86.09% | 0.7322 | 0.7756 |
| BM25 + EG2 | 89.51% | 90.84% | 0.7728 | 0.8153 |
| BGE-M3 + EG2 | 93.39% | 93.44% | 0.8506 | 0.8776 |
| BM25 + BGE-M3 + EG2 | 92.07% | 92.44% | 0.8113 | 0.8468 |

Primary EG2 − BGE-M3: Hit@5 **+8.58 percentage points** (95% paired source-function interval [7.93, 9.22]); nDCG@10 **+0.1203** [0.1144, 0.1264]. The analysis uses 8,122 labelled source-function groups.

Gallery scope: **28,120 cleaned functions** from 34,944 upstream `codebase.txt` URL lines; **6,824 URLs are absent from the raw validation/test archives**. The manifest's selection rule is: “Official preprocess.py: emit codebase.txt URL only if present in raw validation/test”. This reproduces the published cleaned-gallery count; it does not cover all original CodeSearchNet functions. All 8,122 official test queries remain included.

[Full report](codesearchnet-go-test-dense-fusions/report.md) · [Paired comparisons](codesearchnet-go-test-dense-fusions/paired-comparisons.md)

## Java: 10,955 queries / 40,347 functions

| Method | Hit@5 per-channel | Hit@5 total-100 | MRR@10 per-channel | nDCG@10 per-channel |
|---|---:|---:|---:|---:|
| BM25 | 39.42% | 39.42% | 0.2908 | 0.3323 |
| BGE-M3 | 63.73% | 63.73% | 0.5104 | 0.5574 |
| EmbeddingGemma 2 | 83.80% | 83.80% | 0.7227 | 0.7617 |
| BM25 + BGE-M3 | 61.36% | 61.73% | 0.4752 | 0.5283 |
| BM25 + EG2 | 71.11% | 73.52% | 0.5477 | 0.6095 |
| BGE-M3 + EG2 | 76.99% | 77.09% | 0.6433 | 0.6888 |
| BM25 + BGE-M3 + EG2 | 74.66% | 75.98% | 0.5940 | 0.6485 |

Primary EG2 − BGE-M3: Hit@5 **+20.06 percentage points** (95% paired source-function interval [19.24, 20.90]); nDCG@10 **+0.2043** [0.1974, 0.2111]. The analysis uses 10,955 labelled source-function groups.

Gallery scope: **40,347 cleaned functions** from 55,005 upstream `codebase.txt` URL lines; **14,658 URLs are absent from the raw validation/test archives**. The manifest's selection rule is: “Official preprocess.py: emit codebase.txt URL only if present in raw validation/test”. This reproduces the published cleaned-gallery count; it does not cover all original CodeSearchNet functions. All 10,955 official test queries remain included.

[Full report](codesearchnet-java-test-dense-fusions/report.md) · [Paired comparisons](codesearchnet-java-test-dense-fusions/paired-comparisons.md)

Across the completed Python, Ruby, Go, Java collections, standalone EG2 has the highest observed Hit@5 among all seven methods in both budget modes. Equal-weight fusion with the other channels lowers its score here; this does not establish that every fusion strategy is worse. The paired intervals in the linked reports group shared source functions, not whole repositories. They are exploratory marginal intervals, without multiple-comparison correction.

These scores measure finding labelled source functions, not generating correct code or answering a question. Remaining language/model/reranker conditions are not included until completed. No pooled accuracy is computed. [Machine-readable summary](code-dense-summary.json).

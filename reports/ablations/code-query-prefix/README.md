# Code query-prefix comparison

All six full-gallery EG2 comparisons completed on the same frozen **52,561 queries and 183,295 functions**. The ordinary query prefix `task: search result | query: ` is compared with the pinned code prefix `task: code retrieval | query: `. The document prefix remains `title: none | text: `. The existing **183,398 document/chunk vectors were verified and reused**; only the new-prefix queries were encoded. These six extra conditions do not add to the 18,460-cell main matrix.

| Language | Queries | Ordinary EG2 Hit@5 | Code-prefix Hit@5 | Difference (percentage points) | nDCG@10 before → after |
|---|---:|---:|---:|---:|---:|
| go | 8,122 | 96.28% | 96.90% | +0.62 | 0.9232 → 0.9294 |
| java | 10,955 | 83.80% | 84.49% | +0.69 | 0.7617 → 0.7676 |
| javascript | 3,291 | 79.91% | 80.46% | +0.55 | 0.7213 → 0.7245 |
| php | 14,014 | 75.75% | 75.65% | -0.11 | 0.6747 → 0.6733 |
| python | 14,918 | 84.46% | 85.07% | +0.62 | 0.7641 → 0.7717 |
| ruby | 1,261 | 86.28% | 86.12% | -0.16 | 0.7910 → 0.7891 |

The code-specific prefix has higher observed Hit@5 in four languages and lower Hit@5 in PHP and Ruby. The subsequent [paired source-function analysis](paired-comparisons.md) adds exploratory 95% intervals: Go, Java and Python remain above zero, while JavaScript, PHP and Ruby span zero. These marginal intervals have no multiple-comparison adjustment and do not account for dependencies between functions in the same repository. All 17 metric means, paired changes and win/loss/tie counts are retained in the JSON summary.

Both sides use the same full query/gallery, pinned EmbeddingGemma 2 revision, 768 dimensions, MPS bfloat16 and batch size 8. JavaScript preserves two long functions through shared lossless segments and uses the maximum chunk cosine at the original function ID; the other five languages use whole functions. Rerankers and generated-code evaluation are outside this condition.

Independent CPU formulas reconstructed full-gallery dot products, deterministic score/ID sorting and **52,561 rankings / 893,537 metric values**. The baseline document-cache proof also revalidates the original ordinary-prefix rankings and metrics. Missing/incompatible cached document vectors cannot trigger fresh document encoding. The new-query extraction count is 52,561; query timing was not recorded and is unavailable, not zero. No inference-speed saving is measured.

[All paired metric values](summary.json) · [Independent audit](independent-audit.json) · [Recorded encoding counts](encoding.json) · [Frozen execution commands](../../../configs/multimodal-code-prefix-jobs.json)

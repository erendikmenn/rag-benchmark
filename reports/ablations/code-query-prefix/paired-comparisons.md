# Code-query-prefix paired comparison

This adds uncertainty estimates to the six existing EG2 query-prefix comparisons. **No new encoder calls or benchmark conditions** were run. The 52,561 frozen queries, 183,295 source functions, baseline vectors and measured metric values are unchanged. The main matrix remains 1,928 / 18,460.

Differences below are **code prefix minus ordinary prefix**, expressed as metric value × 100. For Hit@5 and Recall@5 these are percentage points. Intervals are marginal exploratory 95% paired source-function bootstrap intervals (5,000 draws, seed 42); no multiple-comparison adjustment is applied.

| Language | Queries / source-function groups | Hit@5 difference [95% interval], percentage points | nDCG@10 difference [95% interval], × 100 |
|---|---:|---:|---:|
| go | 8,122 / 8,122 | +0.62 [+0.38, +0.86] | +0.61 [+0.42, +0.81] |
| java | 10,955 / 10,955 | +0.69 [+0.30, +1.08] | +0.59 [+0.32, +0.86] |
| javascript | 3,291 / 3,291 | +0.55 [-0.18, +1.28] | +0.31 [-0.17, +0.80] |
| php | 14,014 / 14,014 | -0.11 [-0.52, +0.31] | -0.14 [-0.41, +0.13] |
| python | 14,918 / 14,918 | +0.62 [+0.28, +0.96] | +0.76 [+0.52, +1.00] |
| ruby | 1,261 / 1,261 | -0.16 [-1.11, +0.79] | -0.19 [-0.88, +0.49] |

Go, Java and Python have intervals above zero for the three reported metrics. JavaScript, PHP and Ruby have intervals spanning zero. Each query has one positive source, so Recall@5 equals Hit@5 and its interval is identical; these two metrics do not supply independent evidence.

Groups were reconstructed from positive source-function relations and independently checked. Every group contains one query in these six frozen splits; this is an observed property, not a fallback grouping rule. Functions from the same repository may remain dependent across groups. These intervals therefore do not establish an improvement across repositories. Languages remain separate with no pooled estimate, and the analysis does not measure generated-code correctness.

Both compared standalone E conditions allocate 100 candidates. The baseline `per_channel` and `total` modes were verified to have exactly equal per-query rankings and all 17 metrics for all 52,561 queries. Thus the different stored mode labels introduce no candidate-budget difference. JavaScript retains the original shared-segment source-maximum protocol; the other five languages retain whole functions.

The independent reviewer reconstructed the three primary metrics from qrels, reproduced all 18 intervals with the published group-bootstrap implementation and checked all 102 descriptive metric means, full query pairing, payload hashes and source/report/protocol bindings. The 17 existing metric means match the previously published summary exactly. This CPU analysis reuses recorded rankings; the earlier full-gallery vector audit remains separate. The original summary/audit files retain their historical description-only, no-interval scope; the new paired files supply the later interval analysis.

[Full paired values, source hashes and grouping](paired-comparisons.json) · [Independent arithmetic audit](paired-independent-audit.json) · [Independent provenance audit](paired-provenance-audit.json) · [Original results and protocol](README.md)

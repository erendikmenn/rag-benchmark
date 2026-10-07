# Cached EG2 dimension results

**9 full dataset views, 136 dimension/method/budget cells.** Every unchanged 768 baseline was reproduced against original full-corpus rankings and stored per-query results before lower-dimensional evaluation. No new encoder inference was performed.

This summary shows standalone native EG2 (N), with per-channel candidate budgets. Each linked report also contains Recall@5, nDCG@10, both candidate-budget modes and N+S fusion where its specialist evidence validated. No reranker or other representation parameter is varied here.

| Dataset view | Queries | Candidates | N Hit@5:128 | N Hit@5:256 | N Hit@5:512 | N Hit@5:768 |
|---|---:|---:|---:|---:|---:|---:|
| [xm3600-tr](xm3600-tr/dimension-sweep.md) | 7,233 | 3,600 | 66.65% | 82.07% | 84.54% | 84.64% |
| [xm3600-en](xm3600-en/dimension-sweep.md) | 7,200 | 3,600 | 72.56% | 82.47% | 83.56% | 83.75% |
| [clotho-v2.1-evaluation](clotho-v2.1-evaluation/dimension-sweep.md) | 5,225 | 1,045 | 7.71% | 11.37% | 12.06% | 11.75% |
| [clotho-dcase2025-additional-relevance](clotho-dcase2025-additional-relevance/dimension-sweep.md) | 1,037 | 1,045 | 18.51% | 23.63% | 26.13% | 25.84% |
| [fleurs-tr_tr-test](fleurs-tr_tr-test/dimension-sweep.md) | 329 | 743 | 100.00% | 100.00% | 100.00% | 100.00% |
| [msrvtt-1k-a](msrvtt-1k-a/dimension-sweep.md) | 1,000 | 1,000 | 60.30% | 73.50% | 74.70% | 75.00% |
| [vidore-v3-computer_science-en](vidore-v3-computer_science-en/dimension-sweep.md) | 215 | 1,360 | 82.33% | 86.98% | 93.02% | 91.63% |
| [vidore-v3-energy-fr](vidore-v3-energy-fr/dimension-sweep.md) | 308 | 2,225 | 53.25% | 65.26% | 66.56% | 68.83% |
| [vidore-v3-finance_en-en](vidore-v3-finance_en-en/dimension-sweep.md) | 309 | 2,942 | 58.25% | 70.87% | 76.38% | 77.67% |

For Turkish photos, 512 dimensions retain Hit@5 84.54% versus 84.64% at 768, an observed difference of −0.10 percentage points. English photos give 83.56% versus 83.75%. These point estimates do not establish statistical equivalence or a universal best dimension.

Video Hit@5 is 74.70% at 512 dimensions, 73.50% at 256 and 60.30% at 128, versus 75.00% at 768. Video input remains the same sampled visual frames (1 fps, maximum 16), without soundtrack.

The speech Hit@5 stays 100%, but Recall@5 falls from 99.54% at 768 to 98.53% at 128 and nDCG@10 also changes. A saturated hit rate alone does not mean all ranking quality is preserved.

For computer-science pages, 512 dimensions give higher observed Hit@5 (93.02% versus 91.63% at 768) but slightly lower nDCG@10 (0.6266 versus 0.6296). This is not a universal ranking improvement or a test-set-based choice of a new default.

Raw FP32 N-vector payload at 128/256/512 dimensions is 16.7% / 33.3% / 66.7% of the 768-dimensional payload. These are vector-storage ratios, not reductions in total process memory, specialist storage or latency. The base model weights, original inference precision and media preprocessing are held fixed.

All candidates are searched again after prefix projection and L2 normalization. Full source/vector hashes and algorithm/runtime identities are in the JSON files; legacy checksum absence is declared. No independent new FP32 encoder inference or bitwise equality to direct low-dimensional forward output is claimed.

Dataset views overlap in source assets and differ in task/labels; scores are not pooled. The additional-relevance audio protocol is separate from the original captions. [Machine-readable summary](summary.json).

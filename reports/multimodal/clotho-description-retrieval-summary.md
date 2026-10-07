# Clotho: source descriptions versus native audio

All **63 retrieval subsets × two candidate budgets = 126 conditions** completed on the original **5,225 queries / 1,045 clips**. All 1,045 descriptions were generated from source audio only with local Gemma 4 E4B Q4; references were not supplied to the generator. No reranker or answer model is involved in the following retrieval scores.

| Primary channel | Hit@5 / Recall@5 | nDCG@10 |
|---|---:|---:|
| BM25 on Gemma descriptions | 1.44% | 0.0145 |
| BGE-M3 on Gemma descriptions | 2.22% | 0.0215 |
| EG2 on Gemma descriptions | 2.79% | 0.0249 |
| EG2 source audio | 11.75% | 0.0969 |
| CLAP source audio | 37.42% | 0.3028 |
| EG2 source audio + Gemma descriptions | 6.41% | 0.0551 |

Hit@5 and Recall@5 coincide here because every original query has one positive clip. **CLAP has the highest observed Hit@5 in both budgets across all 63 subsets.** Searching the generated text alone is weak in this setting; adding it to EG2 audio changes Hit@5 from 11.75% to 6.41% (−5.34 percentage points; paired 95% source-group interval −6.87 to −3.83). This measures the fixed representation and models; the decline’s cause and human caption correctness have not been established.

The description audit found 737 distinct text hashes across 1,045 clips. Duplicate descriptions are a quality observation, not an identity failure. G/E per-input token diagnostics remain unavailable. The six N/S/N+S baseline cells exactly reproduce the original rankings, scores and metrics, so they are counted once in primary progress: **120 new cells**, not 126 new independent conditions.

Independent CPU formulas reproduced **658,350 rankings and 11,191,950 per-query metric values**. Every source-channel/RRF result and all 17 metrics matched. All frozen queries, qrels and media mappings were preserved. The paired analysis uses 1,045 source groups, 5,000 bootstrap resamples and seed 42; reported intervals are exploratory and marginal.

[Full grid](clotho-v2.1-evaluation-gemma-described-all-retrieval/report.md) · [Paired comparisons](clotho-v2.1-evaluation-gemma-described-all-retrieval/paired-comparisons.md) · [Independent audit](clotho-v2.1-evaluation-gemma-described-all-retrieval/independent-audit.json) · [Description generation audit](clotho-description-audit.md) · [Aggregate source hashes](clotho-description-retrieval-summary.json)

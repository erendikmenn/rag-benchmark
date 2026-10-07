# Video: source visual descriptions and native retrieval

All 63 retrieval subsets × two candidate budgets completed on the original 1,000 queries and 1,000 videos. Descriptions use source visual evidence only; no audio, gold captions, queries or reference answers were supplied to description generation. These are retrieval scores, not caption or generated-answer correctness.

| Primary channel | Hit@5 | Recall@5 | nDCG@10 |
|---|---:|---:|---:|
| BM25 on source visual descriptions | 27.60% | 27.60% | 0.2409 |
| BGE-M3 on source visual descriptions | 40.90% | 40.90% | 0.3419 |
| EG2 on source visual descriptions | 56.90% | 56.90% | 0.4962 |
| EG2 source video | 75.00% | 75.00% | 0.6694 |
| CLIP source video | 53.80% | 53.80% | 0.4594 |
| EG2 source video + visual descriptions | 75.30% | 75.30% | 0.6583 |

Hit@5 and Recall@5 coincide because each query has one positive video.

| Best multi-channel fusion by observed Hit@5 | Method | Hit@5 | Recall@5 | nDCG@10 |
|---|---|---:|---:|---:|
| per_channel | N+J | 75.90% | 75.90% | 0.6684 |
| total | N+J | 75.90% | 75.90% | 0.6684 |

Independent CPU formulas verified 126,000 rankings, RRF derivations and 2,142,000 values across all 17 metrics. Six N/S/N+S cells exactly match the original native/specialist results, leaving 120 new primary cells after deduplication.

Native/specialist encoding in the derived dataset has its own identities. Recorded usage distinguishes new encoder rows from vector-cache reads for the latest report invocation only. These counts are not cumulative model calls; identical final rankings do not establish reuse of original-baseline encoder vectors.

| Channel | Document new / cached rows | Query new / cached rows | Document diagnostics | Query diagnostics |
|---|---:|---:|---|---|
| B | unavailable / unavailable | unavailable / unavailable | unavailable | unavailable |
| G | 0 / 1000 | 0 / 1000 | unavailable | unavailable |
| E | 0 / 1000 | 0 / 1000 | unavailable | unavailable |
| N | 0 / 1000 | 0 / 1000 | complete | complete |
| S | 0 / 1000 | 0 / 1000 | complete | complete |
| J | 0 / 1000 | 0 / 1000 | complete | complete |

Fixed visual sampling does not mean every video frame was consumed. Description semantic correctness and the cause of score differences have not been established. No significance claim is made for the best observed fusion.

[Full grid](msrvtt-1k-a-gemma-described-all-retrieval/report.md) · [Independent audit](msrvtt-1k-a-gemma-described-all-retrieval/independent-audit.json) · [Aggregate hashes and usage](video-description-retrieval-summary.json)

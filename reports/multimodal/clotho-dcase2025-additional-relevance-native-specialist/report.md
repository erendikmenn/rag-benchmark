# Multimodal retrieval results

Dataset: **clotho-dcase2025-additional-relevance**, revision `c78422fcbed579877919620a30075baccf82bf2b`.
Scope: **frozen_collection**; 1,037 evaluated queries out of 1,037, searching 1,045 candidates.

Only completed cells below have measured scores. Planned, unsupported and failed families remain visible in matrix.csv. A completed collection is not the complete seven-track benchmark.

| Method | Budget | Rerank K | Hit@1 | Hit@5 | Recall@5 | MRR@10 | nDCG@10 |
|---|---|---:|---:|---:|---:|---:|---:|
| environment_audio__n__none | per_channel | — | 0.0800 | 0.2584 | 0.1132 | 0.1536 | 0.1211 |
| environment_audio__s__none | per_channel | — | 0.2912 | 0.6451 | 0.3508 | 0.4385 | 0.3771 |
| environment_audio__n_s__none | per_channel | — | 0.2112 | 0.5072 | 0.2540 | 0.3357 | 0.2764 |

Hit@K measures whether at least one labelled relevant item was retrieved. Recall@K measures the fraction of all labelled relevant items retrieved. These can differ substantially for multi-positive datasets.

This report measures retrieval against dataset labels, not generated-answer correctness. Cached results are never reported as new model inference latency. Full configuration, channel timing and coverage are in report.json.

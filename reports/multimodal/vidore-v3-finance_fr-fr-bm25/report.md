# Multimodal retrieval results

Dataset: **vidore-v3-finance_fr-fr**, revision `1d808daa08032ffecdf62da151a7f7a8fe2bd0c9`.
Scope: **frozen_collection**; 320 evaluated queries out of 320, searching 2,384 candidates.

Only completed cells below have measured scores. Planned, unsupported and failed families remain visible in matrix.csv. A completed collection is not the complete seven-track benchmark.

| Method | Budget | Rerank K | Hit@1 | Hit@5 | Recall@5 | MRR@10 | nDCG@10 |
|---|---|---:|---:|---:|---:|---:|---:|
| document__b__none | per_channel | — | 0.2812 | 0.5531 | 0.3141 | 0.4079 | 0.3483 |

Hit@K measures whether at least one labelled relevant item was retrieved. Recall@K measures the fraction of all labelled relevant items retrieved. These can differ substantially for multi-positive datasets.

This report measures retrieval against dataset labels, not generated-answer correctness. Cached results are never reported as new model inference latency. Full configuration, channel timing and coverage are in report.json.

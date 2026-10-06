# Multimodal retrieval results

Dataset: **vidore-v3-hr-en**, revision `0cdf0979f2c5a0fd3e335e6373b9da48a9fe3bc3`.
Scope: **frozen_collection**; 318 evaluated queries out of 318, searching 1,110 candidates.

Only completed cells below have measured scores. Planned, unsupported and failed families remain visible in matrix.csv. A completed collection is not the complete seven-track benchmark.

| Method | Budget | Rerank K | Hit@1 | Hit@5 | Recall@5 | MRR@10 | nDCG@10 |
|---|---|---:|---:|---:|---:|---:|---:|
| document__b__none | per_channel | — | 0.4969 | 0.7862 | 0.4297 | 0.6187 | 0.4984 |

Hit@K measures whether at least one labelled relevant item was retrieved. Recall@K measures the fraction of all labelled relevant items retrieved. These can differ substantially for multi-positive datasets.

This report measures retrieval against dataset labels, not generated-answer correctness. Cached results are never reported as new model inference latency. Full configuration, channel timing and coverage are in report.json.

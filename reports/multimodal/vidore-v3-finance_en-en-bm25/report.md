# Multimodal retrieval results

Dataset: **vidore-v3-finance_en-en**, revision `7f432c176d82e27546501ad8064a713ac3071809`.
Scope: **frozen_collection**; 309 evaluated queries out of 309, searching 2,942 candidates.

Only completed cells below have measured scores. Planned, unsupported and failed families remain visible in matrix.csv. A completed collection is not the complete seven-track benchmark.

| Method | Budget | Rerank K | Hit@1 | Hit@5 | Recall@5 | MRR@10 | nDCG@10 |
|---|---|---:|---:|---:|---:|---:|---:|
| document__b__none | per_channel | — | 0.5340 | 0.7896 | 0.4695 | 0.6474 | 0.5313 |

Hit@K measures whether at least one labelled relevant item was retrieved. Recall@K measures the fraction of all labelled relevant items retrieved. These can differ substantially for multi-positive datasets.

This report measures retrieval against dataset labels, not generated-answer correctness. Cached results are never reported as new model inference latency. Full configuration, channel timing and coverage are in report.json.

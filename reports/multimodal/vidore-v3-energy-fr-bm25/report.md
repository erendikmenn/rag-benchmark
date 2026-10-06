# Multimodal retrieval results

Dataset: **vidore-v3-energy-fr**, revision `caec06d3c73434d635f710f93bcd898331c59f20`.
Scope: **frozen_collection**; 308 evaluated queries out of 308, searching 2,225 candidates.

Only completed cells below have measured scores. Planned, unsupported and failed families remain visible in matrix.csv. A completed collection is not the complete seven-track benchmark.

| Method | Budget | Rerank K | Hit@1 | Hit@5 | Recall@5 | MRR@10 | nDCG@10 |
|---|---|---:|---:|---:|---:|---:|---:|
| document__b__none | per_channel | — | 0.5065 | 0.7727 | 0.5465 | 0.6225 | 0.5568 |

Hit@K measures whether at least one labelled relevant item was retrieved. Recall@K measures the fraction of all labelled relevant items retrieved. These can differ substantially for multi-positive datasets.

This report measures retrieval against dataset labels, not generated-answer correctness. Cached results are never reported as new model inference latency. Full configuration, channel timing and coverage are in report.json.

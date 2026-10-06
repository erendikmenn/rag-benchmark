# Multimodal retrieval results

Dataset: **vidore-v3-pharmaceuticals-en**, revision `3abd4aa8a9445fb5538a78a19ba50bd57bd22b5c`.
Scope: **frozen_collection**; 364 evaluated queries out of 364, searching 2,313 candidates.

Only completed cells below have measured scores. Planned, unsupported and failed families remain visible in matrix.csv. A completed collection is not the complete seven-track benchmark.

| Method | Budget | Rerank K | Hit@1 | Hit@5 | Recall@5 | MRR@10 | nDCG@10 |
|---|---|---:|---:|---:|---:|---:|---:|
| document__b__none | per_channel | — | 0.5385 | 0.8159 | 0.5084 | 0.6510 | 0.5529 |

Hit@K measures whether at least one labelled relevant item was retrieved. Recall@K measures the fraction of all labelled relevant items retrieved. These can differ substantially for multi-positive datasets.

This report measures retrieval against dataset labels, not generated-answer correctness. Cached results are never reported as new model inference latency. Full configuration, channel timing and coverage are in report.json.

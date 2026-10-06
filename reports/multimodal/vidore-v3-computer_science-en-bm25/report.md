# Multimodal retrieval results

Dataset: **vidore-v3-computer_science-en**, revision `d5cc75883d92e294f0c0fc2662551c9708a06ebc`.
Scope: **frozen_collection**; 215 evaluated queries out of 215, searching 1,360 candidates.

Only completed cells below have measured scores. Planned, unsupported and failed families remain visible in matrix.csv. A completed collection is not the complete seven-track benchmark.

| Method | Budget | Rerank K | Hit@1 | Hit@5 | Recall@5 | MRR@10 | nDCG@10 |
|---|---|---:|---:|---:|---:|---:|---:|
| document__b__none | per_channel | — | 0.6977 | 0.9209 | 0.5353 | 0.7918 | 0.6334 |

Hit@K measures whether at least one labelled relevant item was retrieved. Recall@K measures the fraction of all labelled relevant items retrieved. These can differ substantially for multi-positive datasets.

This report measures retrieval against dataset labels, not generated-answer correctness. Cached results are never reported as new model inference latency. Full configuration, channel timing and coverage are in report.json.

# Multimodal retrieval results

Dataset: **vidore-v3-industrial-en**, revision `e26c864724f5dd71a3d7d739272d95637764cee9`.
Scope: **frozen_collection**; 283 evaluated queries out of 283, searching 5,244 candidates.

Only completed cells below have measured scores. Planned, unsupported and failed families remain visible in matrix.csv. A completed collection is not the complete seven-track benchmark.

| Method | Budget | Rerank K | Hit@1 | Hit@5 | Recall@5 | MRR@10 | nDCG@10 |
|---|---|---:|---:|---:|---:|---:|---:|
| document__b__none | per_channel | — | 0.4558 | 0.6855 | 0.4158 | 0.5538 | 0.4615 |

Hit@K measures whether at least one labelled relevant item was retrieved. Recall@K measures the fraction of all labelled relevant items retrieved. These can differ substantially for multi-positive datasets.

This report measures retrieval against dataset labels, not generated-answer correctness. Cached results are never reported as new model inference latency. Full configuration, channel timing and coverage are in report.json.

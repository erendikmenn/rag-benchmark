# Multimodal retrieval results

Dataset: **vidore-v3-physics-fr**, revision `a0de276f515acc044b72cae8de53a44bb5a8f1f5`.
Scope: **frozen_collection**; 302 evaluated queries out of 302, searching 1,674 candidates.

Only completed cells below have measured scores. Planned, unsupported and failed families remain visible in matrix.csv. A completed collection is not the complete seven-track benchmark.

| Method | Budget | Rerank K | Hit@1 | Hit@5 | Recall@5 | MRR@10 | nDCG@10 |
|---|---|---:|---:|---:|---:|---:|---:|
| document__b__none | per_channel | — | 0.3675 | 0.7252 | 0.3057 | 0.5174 | 0.3904 |

Hit@K measures whether at least one labelled relevant item was retrieved. Recall@K measures the fraction of all labelled relevant items retrieved. These can differ substantially for multi-positive datasets.

This report measures retrieval against dataset labels, not generated-answer correctness. Cached results are never reported as new model inference latency. Full configuration, channel timing and coverage are in report.json.

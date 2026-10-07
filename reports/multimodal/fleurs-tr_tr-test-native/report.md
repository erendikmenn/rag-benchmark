# Multimodal retrieval results

Dataset: **fleurs-tr_tr-test**, revision `70bb2e84b976b7e960aa89f1c648e09c59f894dd`.
Scope: **frozen_collection**; 329 evaluated queries out of 329, searching 743 candidates.

Only completed cells below have measured scores. Planned, unsupported and failed families remain visible in matrix.csv. A completed collection is not the complete seven-track benchmark.

| Method | Budget | Rerank K | Hit@1 | Hit@5 | Recall@5 | MRR@10 | nDCG@10 |
|---|---|---:|---:|---:|---:|---:|---:|
| speech__n__none | per_channel | — | 0.9970 | 1.0000 | 0.9954 | 0.9985 | 0.9970 |

Hit@K measures whether at least one labelled relevant item was retrieved. Recall@K measures the fraction of all labelled relevant items retrieved. These can differ substantially for multi-positive datasets.

This report measures retrieval against dataset labels, not generated-answer correctness. Cached results are never reported as new model inference latency. Full configuration, channel timing and coverage are in report.json.

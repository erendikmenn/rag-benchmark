# Multimodal retrieval results

Dataset: **codesearchnet-go-test**, revision `c0de43d3aaf38e89290f1efb771f8de845e7a489`.
Scope: **frozen_collection**; 8,122 evaluated queries out of 8,122, searching 28,120 candidates.

Only completed cells below have measured scores. Planned, unsupported and failed families remain visible in matrix.csv. A completed collection is not the complete seven-track benchmark.

| Method | Budget | Rerank K | Hit@1 | Hit@5 | Recall@5 | MRR@10 | nDCG@10 |
|---|---|---:|---:|---:|---:|---:|---:|
| code__b__none | per_channel | — | 0.4153 | 0.6194 | 0.6194 | 0.5038 | 0.5495 |

Hit@K measures whether at least one labelled relevant item was retrieved. Recall@K measures the fraction of all labelled relevant items retrieved. These can differ substantially for multi-positive datasets.

This report measures retrieval against dataset labels, not generated-answer correctness. Cached results are never reported as new model inference latency. Full configuration, channel timing and coverage are in report.json.

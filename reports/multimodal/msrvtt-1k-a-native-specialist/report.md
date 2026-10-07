# Multimodal retrieval results

Dataset: **msrvtt-1k-a**, revision `CLIP4Clip:v0.0:MSRVTT_JSFUSION_test`.
Scope: **frozen_collection**; 1,000 evaluated queries out of 1,000, searching 1,000 candidates.

Only completed cells below have measured scores. Planned, unsupported and failed families remain visible in matrix.csv. A completed collection is not the complete seven-track benchmark.

| Method | Budget | Rerank K | Hit@1 | Hit@5 | Recall@5 | MRR@10 | nDCG@10 |
|---|---|---:|---:|---:|---:|---:|---:|
| video__n__none | per_channel | — | 0.5190 | 0.7500 | 0.7500 | 0.6184 | 0.6694 |
| video__s__none | per_channel | — | 0.3070 | 0.5380 | 0.5380 | 0.4050 | 0.4594 |
| video__n_s__none | per_channel | — | 0.4300 | 0.6710 | 0.6710 | 0.5331 | 0.5900 |

Hit@K measures whether at least one labelled relevant item was retrieved. Recall@K measures the fraction of all labelled relevant items retrieved. These can differ substantially for multi-positive datasets.

This report measures retrieval against dataset labels, not generated-answer correctness. Cached results are never reported as new model inference latency. Full configuration, channel timing and coverage are in report.json.

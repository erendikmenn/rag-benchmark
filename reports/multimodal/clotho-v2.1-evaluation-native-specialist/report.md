# Multimodal retrieval results

Dataset: **clotho-v2.1-evaluation**, revision `zenodo:4783391:v2.1`.
Scope: **frozen_collection**; 5,225 evaluated queries out of 5,225, searching 1,045 candidates.

Only completed cells below have measured scores. Planned, unsupported and failed families remain visible in matrix.csv. A completed collection is not the complete seven-track benchmark.

| Method | Budget | Rerank K | Hit@1 | Hit@5 | Recall@5 | MRR@10 | nDCG@10 |
|---|---|---:|---:|---:|---:|---:|---:|
| environment_audio__n__none | per_channel | — | 0.0375 | 0.1175 | 0.1175 | 0.0731 | 0.0969 |
| environment_audio__s__none | per_channel | — | 0.1462 | 0.3742 | 0.3742 | 0.2428 | 0.3028 |
| environment_audio__n_s__none | per_channel | — | 0.0928 | 0.2647 | 0.2647 | 0.1663 | 0.2150 |

Hit@K measures whether at least one labelled relevant item was retrieved. Recall@K measures the fraction of all labelled relevant items retrieved. These can differ substantially for multi-positive datasets.

This report measures retrieval against dataset labels, not generated-answer correctness. Cached results are never reported as new model inference latency. Full configuration, channel timing and coverage are in report.json.

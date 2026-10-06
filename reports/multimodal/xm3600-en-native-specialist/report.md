# Multimodal retrieval results

Dataset: **xm3600-en**, revision `ad44c380c86cb1240666361caabaf5684507d0ef`.
Scope: **frozen_collection**; 7,200 evaluated queries out of 7,200, searching 3,600 candidates.

Only completed cells below have measured scores. Planned, unsupported and failed families remain visible in matrix.csv. A completed collection is not the complete seven-track benchmark.

| Method | Budget | Rerank K | Hit@1 | Hit@5 | Recall@5 | MRR@10 | nDCG@10 |
|---|---|---:|---:|---:|---:|---:|---:|
| photo__n__none | per_channel | — | 0.6175 | 0.8375 | 0.8375 | 0.7126 | 0.7582 |
| photo__s__none | per_channel | — | 0.5244 | 0.7689 | 0.7689 | 0.6281 | 0.6794 |
| photo__n_s__none | per_channel | — | 0.5956 | 0.8292 | 0.8292 | 0.6957 | 0.7445 |

Hit@K measures whether at least one labelled relevant item was retrieved. Recall@K measures the fraction of all labelled relevant items retrieved. These can differ substantially for multi-positive datasets.

This report measures retrieval against dataset labels, not generated-answer correctness. Cached results are never reported as new model inference latency. Full configuration, channel timing and coverage are in report.json.

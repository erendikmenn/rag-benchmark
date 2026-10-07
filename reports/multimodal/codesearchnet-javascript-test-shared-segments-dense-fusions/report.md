# Multimodal retrieval results

Dataset: **codesearchnet-javascript-test**, revision `c0de43d3aaf38e89290f1efb771f8de845e7a489`.
Scope: **frozen_collection**; 3,291 evaluated queries out of 3,291, searching 13,981 candidates.

Only completed cells below have measured scores. Planned, unsupported and failed families remain visible in matrix.csv. A completed collection is not the complete seven-track benchmark.

| Method | Budget | Rerank K | Hit@1 | Hit@5 | Recall@5 | MRR@10 | nDCG@10 |
|---|---|---:|---:|---:|---:|---:|---:|
| code__b__none | per_channel | — | 0.2091 | 0.3662 | 0.3662 | 0.2753 | 0.3124 |
| code__g__none | per_channel | — | 0.3895 | 0.5822 | 0.5822 | 0.4722 | 0.5150 |
| code__e__none | per_channel | — | 0.5898 | 0.7991 | 0.7991 | 0.6804 | 0.7213 |
| code__b_g__none | per_channel | — | 0.3528 | 0.5640 | 0.5640 | 0.4445 | 0.4925 |
| code__b_e__none | per_channel | — | 0.4020 | 0.6421 | 0.6421 | 0.5079 | 0.5653 |
| code__g_e__none | per_channel | — | 0.5084 | 0.7125 | 0.7125 | 0.5963 | 0.6397 |
| code__b_g_e__none | per_channel | — | 0.4464 | 0.6855 | 0.6855 | 0.5475 | 0.6001 |

Hit@K measures whether at least one labelled relevant item was retrieved. Recall@K measures the fraction of all labelled relevant items retrieved. These can differ substantially for multi-positive datasets.

This report measures retrieval against dataset labels, not generated-answer correctness. Cached results are never reported as new model inference latency. Full configuration, channel timing and coverage are in report.json.

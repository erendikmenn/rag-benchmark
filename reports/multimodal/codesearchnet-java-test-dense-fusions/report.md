# Multimodal retrieval results

Dataset: **codesearchnet-java-test**, revision `c0de43d3aaf38e89290f1efb771f8de845e7a489`.
Scope: **frozen_collection**; 10,955 evaluated queries out of 10,955, searching 40,347 candidates.

Only completed cells below have measured scores. Planned, unsupported and failed families remain visible in matrix.csv. A completed collection is not the complete seven-track benchmark.

| Method | Budget | Rerank K | Hit@1 | Hit@5 | Recall@5 | MRR@10 | nDCG@10 |
|---|---|---:|---:|---:|---:|---:|---:|
| code__b__none | per_channel | — | 0.2132 | 0.3942 | 0.3942 | 0.2908 | 0.3323 |
| code__g__none | per_channel | — | 0.4166 | 0.6373 | 0.6373 | 0.5104 | 0.5574 |
| code__e__none | per_channel | — | 0.6358 | 0.8380 | 0.8380 | 0.7227 | 0.7617 |
| code__b_g__none | per_channel | — | 0.3707 | 0.6136 | 0.6136 | 0.4752 | 0.5283 |
| code__b_e__none | per_channel | — | 0.4271 | 0.7111 | 0.7111 | 0.5477 | 0.6095 |
| code__g_e__none | per_channel | — | 0.5465 | 0.7699 | 0.7699 | 0.6433 | 0.6888 |
| code__b_g_e__none | per_channel | — | 0.4809 | 0.7466 | 0.7466 | 0.5940 | 0.6485 |

Hit@K measures whether at least one labelled relevant item was retrieved. Recall@K measures the fraction of all labelled relevant items retrieved. These can differ substantially for multi-positive datasets.

This report measures retrieval against dataset labels, not generated-answer correctness. Cached results are never reported as new model inference latency. Full configuration, channel timing and coverage are in report.json.

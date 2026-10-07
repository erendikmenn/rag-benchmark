# Multimodal retrieval results

Dataset: **codesearchnet-php-test**, revision `c0de43d3aaf38e89290f1efb771f8de845e7a489`.
Scope: **frozen_collection**; 14,014 evaluated queries out of 14,014, searching 52,660 candidates.

Only completed cells below have measured scores. Planned, unsupported and failed families remain visible in matrix.csv. A completed collection is not the complete seven-track benchmark.

| Method | Budget | Rerank K | Hit@1 | Hit@5 | Recall@5 | MRR@10 | nDCG@10 |
|---|---|---:|---:|---:|---:|---:|---:|
| code__b__none | per_channel | — | 0.1743 | 0.3327 | 0.3327 | 0.2421 | 0.2797 |
| code__g__none | per_channel | — | 0.3681 | 0.5774 | 0.5774 | 0.4576 | 0.5051 |
| code__e__none | per_channel | — | 0.5290 | 0.7575 | 0.7575 | 0.6277 | 0.6747 |
| code__b_g__none | per_channel | — | 0.3210 | 0.5410 | 0.5410 | 0.4157 | 0.4658 |
| code__b_e__none | per_channel | — | 0.3711 | 0.6258 | 0.6258 | 0.4797 | 0.5389 |
| code__g_e__none | per_channel | — | 0.4779 | 0.7081 | 0.7081 | 0.5766 | 0.6251 |
| code__b_g_e__none | per_channel | — | 0.4202 | 0.6748 | 0.6748 | 0.5289 | 0.5851 |

Hit@K measures whether at least one labelled relevant item was retrieved. Recall@K measures the fraction of all labelled relevant items retrieved. These can differ substantially for multi-positive datasets.

This report measures retrieval against dataset labels, not generated-answer correctness. Cached results are never reported as new model inference latency. Full configuration, channel timing and coverage are in report.json.

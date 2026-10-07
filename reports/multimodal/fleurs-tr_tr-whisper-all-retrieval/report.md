# Multimodal retrieval results

Dataset: **fleurs-tr_tr-whisper**, revision `0db3c05e0e8109f758a6d3c38b57f00a621d6927d0ba03f27448e61cd85beffc`.
Scope: **frozen_collection**; 329 evaluated queries out of 329, searching 743 candidates.

Only completed cells below have measured scores. Planned, unsupported and failed families remain visible in matrix.csv. A completed collection is not the complete seven-track benchmark.

| Method | Budget | Rerank K | Hit@1 | Hit@5 | Recall@5 | MRR@10 | nDCG@10 |
|---|---|---:|---:|---:|---:|---:|---:|
| speech__b__none | per_channel | — | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| speech__g__none | per_channel | — | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| speech__e__none | per_channel | — | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| speech__n__none | per_channel | — | 0.9970 | 1.0000 | 0.9954 | 0.9985 | 0.9970 |
| speech__j__none | per_channel | — | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| speech__b_g__none | per_channel | — | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| speech__b_e__none | per_channel | — | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| speech__b_n__none | per_channel | — | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| speech__b_j__none | per_channel | — | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| speech__g_e__none | per_channel | — | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| speech__g_n__none | per_channel | — | 1.0000 | 1.0000 | 0.9990 | 1.0000 | 0.9998 |
| speech__g_j__none | per_channel | — | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| speech__e_n__none | per_channel | — | 1.0000 | 1.0000 | 0.9990 | 1.0000 | 0.9995 |
| speech__e_j__none | per_channel | — | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| speech__n_j__none | per_channel | — | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9995 |
| speech__b_g_e__none | per_channel | — | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| speech__b_g_n__none | per_channel | — | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| speech__b_g_j__none | per_channel | — | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| speech__b_e_n__none | per_channel | — | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| speech__b_e_j__none | per_channel | — | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| speech__b_n_j__none | per_channel | — | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| speech__g_e_n__none | per_channel | — | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| speech__g_e_j__none | per_channel | — | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| speech__g_n_j__none | per_channel | — | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| speech__e_n_j__none | per_channel | — | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| speech__b_g_e_n__none | per_channel | — | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| speech__b_g_e_j__none | per_channel | — | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| speech__b_g_n_j__none | per_channel | — | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| speech__b_e_n_j__none | per_channel | — | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| speech__g_e_n_j__none | per_channel | — | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| speech__b_g_e_n_j__none | per_channel | — | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

Hit@K measures whether at least one labelled relevant item was retrieved. Recall@K measures the fraction of all labelled relevant items retrieved. These can differ substantially for multi-positive datasets.

This report measures retrieval against dataset labels, not generated-answer correctness. Cached results are never reported as new model inference latency. Full configuration, channel timing and coverage are in report.json.

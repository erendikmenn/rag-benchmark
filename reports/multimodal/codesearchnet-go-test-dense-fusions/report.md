# Multimodal retrieval results

Dataset: **codesearchnet-go-test**, revision `c0de43d3aaf38e89290f1efb771f8de845e7a489`.
Scope: **frozen_collection**; 8,122 evaluated queries out of 8,122, searching 28,120 candidates.

Only completed cells below have measured scores. Planned, unsupported and failed families remain visible in matrix.csv. A completed collection is not the complete seven-track benchmark.

| Method | Budget | Rerank K | Hit@1 | Hit@5 | Recall@5 | MRR@10 | nDCG@10 |
|---|---|---:|---:|---:|---:|---:|---:|
| code__b__none | per_channel | — | 0.4153 | 0.6194 | 0.6194 | 0.5038 | 0.5495 |
| code__g__none | per_channel | — | 0.6865 | 0.8770 | 0.8770 | 0.7677 | 0.8029 |
| code__e__none | per_channel | — | 0.8598 | 0.9628 | 0.9628 | 0.9054 | 0.9232 |
| code__b_g__none | per_channel | — | 0.6353 | 0.8571 | 0.8571 | 0.7322 | 0.7756 |
| code__b_e__none | per_channel | — | 0.6788 | 0.8951 | 0.8951 | 0.7728 | 0.8153 |
| code__g_e__none | per_channel | — | 0.7827 | 0.9339 | 0.9339 | 0.8506 | 0.8776 |
| code__b_g_e__none | per_channel | — | 0.7285 | 0.9207 | 0.9207 | 0.8113 | 0.8468 |

Hit@K measures whether at least one labelled relevant item was retrieved. Recall@K measures the fraction of all labelled relevant items retrieved. These can differ substantially for multi-positive datasets.

This report measures retrieval against dataset labels, not generated-answer correctness. Cached results are never reported as new model inference latency. Full configuration, channel timing and coverage are in report.json.

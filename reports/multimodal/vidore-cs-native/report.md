# Multimodal retrieval results

Dataset: **vidore-v3-computer_science-en**, revision `d5cc75883d92e294f0c0fc2662551c9708a06ebc`.
Scope: **frozen_collection**; 215 evaluated queries out of 215, searching 1,360 candidates.

Only completed cells below have measured scores. Planned, unsupported and failed families remain visible in matrix.csv. A completed collection is not the complete seven-track benchmark.

| Method | Budget | Rerank K | Hit@1 | Hit@5 | Recall@5 | MRR@10 | nDCG@10 |
|---|---|---:|---:|---:|---:|---:|---:|
| document__n__none | per_channel | — | 0.6651 | 0.9163 | 0.5270 | 0.7787 | 0.6296 |

Hit@K measures whether at least one labelled relevant item was retrieved. Recall@K measures the fraction of all labelled relevant items retrieved. These can differ substantially for multi-positive datasets.

This report measures retrieval against dataset labels, not generated-answer correctness. Cached results are never reported as new model inference latency. Full configuration, channel timing and coverage are in report.json.

Historical launch note: this worker started before the explicit-empty-OCR handling fix. Its B/G/E/J availability flags belong to that earlier process. Native-image retrieval completed on all 215 queries; the corrected all-channel rerun is queued. The separate BM25 report retained the same 1,360-page gallery.

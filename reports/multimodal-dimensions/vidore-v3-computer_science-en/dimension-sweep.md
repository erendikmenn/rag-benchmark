# Cache-only EG2 dimension sweep

Status: **completed**.

Fresh model inference calls: 0. No inference speed is inferred from cached embeddings.

The unchanged 768-dimensional baseline was reproduced before evaluation. Prefixes search the full frozen corpus.

Raw FP32 N-vector payload only; excludes model, process RSS, index overhead and specialist storage.

| Dimension | Method | Budget | Hit@5 | Recall@5 | nDCG@10 | N vector ratio |
|---:|---|---|---:|---:|---:|---:|
| 128 | N | per_channel | 0.823256 | 0.414646 | 0.490232 | 0.167 |
| 128 | N | total | 0.823256 | 0.414646 | 0.490232 | 0.167 |
| 128 | N+S | per_channel | 0.948837 | 0.551032 | 0.654774 | 0.167 |
| 128 | N+S | total | 0.948837 | 0.551032 | 0.653225 | 0.167 |
| 256 | N | per_channel | 0.869767 | 0.483718 | 0.586750 | 0.333 |
| 256 | N | total | 0.869767 | 0.483718 | 0.586750 | 0.333 |
| 256 | N+S | per_channel | 0.962791 | 0.602013 | 0.712374 | 0.333 |
| 256 | N+S | total | 0.962791 | 0.602013 | 0.710829 | 0.333 |
| 512 | N | per_channel | 0.930233 | 0.537963 | 0.626555 | 0.667 |
| 512 | N | total | 0.930233 | 0.537963 | 0.626555 | 0.667 |
| 512 | N+S | per_channel | 0.962791 | 0.609265 | 0.720375 | 0.667 |
| 512 | N+S | total | 0.962791 | 0.609265 | 0.720387 | 0.667 |
| 768 | N | per_channel | 0.916279 | 0.527001 | 0.629610 | 1.000 |
| 768 | N | total | 0.916279 | 0.527001 | 0.629610 | 1.000 |
| 768 | N+S | per_channel | 0.967442 | 0.616291 | 0.726442 | 1.000 |
| 768 | N+S | total | 0.967442 | 0.616291 | 0.725627 | 1.000 |

Specialist fusion: validated.

Current array/ID hashes are newly observed integrity records. Legacy caches lack original independent vector checksums.

# Cache-only EG2 dimension sweep

Status: **completed**.

Fresh model inference calls: 0. No inference speed is inferred from cached embeddings.

The unchanged 768-dimensional baseline was reproduced before evaluation. Prefixes search the full frozen corpus.

Raw FP32 N-vector payload only; excludes model, process RSS, index overhead and specialist storage.

| Dimension | Method | Budget | Hit@5 | Recall@5 | nDCG@10 | N vector ratio |
|---:|---|---|---:|---:|---:|---:|
| 128 | N | per_channel | 1.000000 | 0.985309 | 0.980206 | 0.167 |
| 128 | N | total | 1.000000 | 0.985309 | 0.980206 | 0.167 |
| 256 | N | per_channel | 1.000000 | 0.993414 | 0.995926 | 0.333 |
| 256 | N | total | 1.000000 | 0.993414 | 0.995926 | 0.333 |
| 512 | N | per_channel | 1.000000 | 0.995441 | 0.997851 | 0.667 |
| 512 | N | total | 1.000000 | 0.995441 | 0.997851 | 0.667 |
| 768 | N | per_channel | 1.000000 | 0.995441 | 0.997026 | 1.000 |
| 768 | N | total | 1.000000 | 0.995441 | 0.997026 | 1.000 |

Specialist fusion: unavailable (base_s_not_completed).

Current array/ID hashes are newly observed integrity records. Legacy caches lack original independent vector checksums.

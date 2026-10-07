# Cache-only EG2 dimension sweep

Status: **completed**.

Fresh model inference calls: 0. No inference speed is inferred from cached embeddings.

The unchanged 768-dimensional baseline was reproduced before evaluation. Prefixes search the full frozen corpus.

Raw FP32 N-vector payload only; excludes model, process RSS, index overhead and specialist storage.

| Dimension | Method | Budget | Hit@5 | Recall@5 | nDCG@10 | N vector ratio |
|---:|---|---|---:|---:|---:|---:|
| 128 | N | per_channel | 0.532468 | 0.302941 | 0.319568 | 0.167 |
| 128 | N | total | 0.532468 | 0.302941 | 0.319568 | 0.167 |
| 128 | N+S | per_channel | 0.724026 | 0.478949 | 0.499156 | 0.167 |
| 128 | N+S | total | 0.727273 | 0.483603 | 0.512686 | 0.167 |
| 256 | N | per_channel | 0.652597 | 0.409131 | 0.419644 | 0.333 |
| 256 | N | total | 0.652597 | 0.409131 | 0.419644 | 0.333 |
| 256 | N+S | per_channel | 0.792208 | 0.554691 | 0.554996 | 0.333 |
| 256 | N+S | total | 0.788961 | 0.551119 | 0.557747 | 0.333 |
| 512 | N | per_channel | 0.665584 | 0.427813 | 0.447466 | 0.667 |
| 512 | N | total | 0.665584 | 0.427813 | 0.447466 | 0.667 |
| 512 | N+S | per_channel | 0.821429 | 0.571449 | 0.566426 | 0.667 |
| 512 | N+S | total | 0.821429 | 0.571449 | 0.565337 | 0.667 |
| 768 | N | per_channel | 0.688312 | 0.446546 | 0.446102 | 1.000 |
| 768 | N | total | 0.688312 | 0.446546 | 0.446102 | 1.000 |
| 768 | N+S | per_channel | 0.818182 | 0.568968 | 0.569613 | 1.000 |
| 768 | N+S | total | 0.818182 | 0.568968 | 0.567876 | 1.000 |

Specialist fusion: validated.

Current array/ID hashes are newly observed integrity records. Legacy caches lack original independent vector checksums.

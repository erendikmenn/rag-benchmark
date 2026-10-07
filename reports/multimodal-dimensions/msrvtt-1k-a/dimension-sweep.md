# Cache-only EG2 dimension sweep

Status: **completed**.

Fresh model inference calls: 0. No inference speed is inferred from cached embeddings.

The unchanged 768-dimensional baseline was reproduced before evaluation. Prefixes search the full frozen corpus.

Raw FP32 N-vector payload only; excludes model, process RSS, index overhead and specialist storage.

| Dimension | Method | Budget | Hit@5 | Recall@5 | nDCG@10 | N vector ratio |
|---:|---|---|---:|---:|---:|---:|
| 128 | N | per_channel | 0.603000 | 0.603000 | 0.531064 | 0.167 |
| 128 | N | total | 0.603000 | 0.603000 | 0.531064 | 0.167 |
| 128 | N+S | per_channel | 0.660000 | 0.660000 | 0.568040 | 0.167 |
| 128 | N+S | total | 0.658000 | 0.658000 | 0.566228 | 0.167 |
| 256 | N | per_channel | 0.735000 | 0.735000 | 0.644267 | 0.333 |
| 256 | N | total | 0.735000 | 0.735000 | 0.644267 | 0.333 |
| 256 | N+S | per_channel | 0.671000 | 0.671000 | 0.584201 | 0.333 |
| 256 | N+S | total | 0.669000 | 0.669000 | 0.583994 | 0.333 |
| 512 | N | per_channel | 0.747000 | 0.747000 | 0.664246 | 0.667 |
| 512 | N | total | 0.747000 | 0.747000 | 0.664246 | 0.667 |
| 512 | N+S | per_channel | 0.667000 | 0.667000 | 0.587065 | 0.667 |
| 512 | N+S | total | 0.665000 | 0.665000 | 0.587562 | 0.667 |
| 768 | N | per_channel | 0.750000 | 0.750000 | 0.669420 | 1.000 |
| 768 | N | total | 0.750000 | 0.750000 | 0.669420 | 1.000 |
| 768 | N+S | per_channel | 0.671000 | 0.671000 | 0.589966 | 1.000 |
| 768 | N+S | total | 0.669000 | 0.669000 | 0.590366 | 1.000 |

Specialist fusion: validated.

Current array/ID hashes are newly observed integrity records. Legacy caches lack original independent vector checksums.

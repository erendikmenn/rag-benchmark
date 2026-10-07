# Cache-only EG2 dimension sweep

Status: **completed**.

Fresh model inference calls: 0. No inference speed is inferred from cached embeddings.

The unchanged 768-dimensional baseline was reproduced before evaluation. Prefixes search the full frozen corpus.

Raw FP32 N-vector payload only; excludes model, process RSS, index overhead and specialist storage.

| Dimension | Method | Budget | Hit@5 | Recall@5 | nDCG@10 | N vector ratio |
|---:|---|---|---:|---:|---:|---:|
| 128 | N | per_channel | 0.077129 | 0.077129 | 0.063203 | 0.167 |
| 128 | N | total | 0.077129 | 0.077129 | 0.063203 | 0.167 |
| 128 | N+S | per_channel | 0.215502 | 0.215502 | 0.180833 | 0.167 |
| 128 | N+S | total | 0.222584 | 0.222584 | 0.188482 | 0.167 |
| 256 | N | per_channel | 0.113684 | 0.113684 | 0.094874 | 0.333 |
| 256 | N | total | 0.113684 | 0.113684 | 0.094874 | 0.333 |
| 256 | N+S | per_channel | 0.259904 | 0.259904 | 0.216286 | 0.333 |
| 256 | N+S | total | 0.262392 | 0.262392 | 0.219072 | 0.333 |
| 512 | N | per_channel | 0.120574 | 0.120574 | 0.098935 | 0.667 |
| 512 | N | total | 0.120574 | 0.120574 | 0.098935 | 0.667 |
| 512 | N+S | per_channel | 0.261818 | 0.261818 | 0.215829 | 0.667 |
| 512 | N+S | total | 0.266603 | 0.266603 | 0.221286 | 0.667 |
| 768 | N | per_channel | 0.117512 | 0.117512 | 0.096901 | 1.000 |
| 768 | N | total | 0.117512 | 0.117512 | 0.096901 | 1.000 |
| 768 | N+S | per_channel | 0.264689 | 0.264689 | 0.214992 | 1.000 |
| 768 | N+S | total | 0.268900 | 0.268900 | 0.221736 | 1.000 |

Specialist fusion: validated.

Current array/ID hashes are newly observed integrity records. Legacy caches lack original independent vector checksums.

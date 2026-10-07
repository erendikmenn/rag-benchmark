# Cache-only EG2 dimension sweep

Status: **completed**.

Fresh model inference calls: 0. No inference speed is inferred from cached embeddings.

The unchanged 768-dimensional baseline was reproduced before evaluation. Prefixes search the full frozen corpus.

Raw FP32 N-vector payload only; excludes model, process RSS, index overhead and specialist storage.

| Dimension | Method | Budget | Hit@5 | Recall@5 | nDCG@10 | N vector ratio |
|---:|---|---|---:|---:|---:|---:|
| 128 | N | per_channel | 0.582524 | 0.293959 | 0.323068 | 0.167 |
| 128 | N | total | 0.582524 | 0.293959 | 0.323068 | 0.167 |
| 128 | N+S | per_channel | 0.834951 | 0.485418 | 0.531483 | 0.167 |
| 128 | N+S | total | 0.838188 | 0.488137 | 0.530241 | 0.167 |
| 256 | N | per_channel | 0.708738 | 0.386964 | 0.427184 | 0.333 |
| 256 | N | total | 0.708738 | 0.386964 | 0.427184 | 0.333 |
| 256 | N+S | per_channel | 0.847896 | 0.495646 | 0.568317 | 0.333 |
| 256 | N+S | total | 0.847896 | 0.496647 | 0.563050 | 0.333 |
| 512 | N | per_channel | 0.763754 | 0.427998 | 0.474194 | 0.667 |
| 512 | N | total | 0.763754 | 0.427998 | 0.474194 | 0.667 |
| 512 | N+S | per_channel | 0.864078 | 0.510683 | 0.580648 | 0.667 |
| 512 | N+S | total | 0.864078 | 0.510683 | 0.578707 | 0.667 |
| 768 | N | per_channel | 0.776699 | 0.433214 | 0.479545 | 1.000 |
| 768 | N | total | 0.776699 | 0.433214 | 0.479545 | 1.000 |
| 768 | N+S | per_channel | 0.860841 | 0.517475 | 0.583338 | 1.000 |
| 768 | N+S | total | 0.860841 | 0.517475 | 0.579622 | 1.000 |

Specialist fusion: validated.

Current array/ID hashes are newly observed integrity records. Legacy caches lack original independent vector checksums.

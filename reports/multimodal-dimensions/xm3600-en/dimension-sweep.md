# Cache-only EG2 dimension sweep

Status: **completed**.

Fresh model inference calls: 0. No inference speed is inferred from cached embeddings.

The unchanged 768-dimensional baseline was reproduced before evaluation. Prefixes search the full frozen corpus.

Raw FP32 N-vector payload only; excludes model, process RSS, index overhead and specialist storage.

| Dimension | Method | Budget | Hit@5 | Recall@5 | nDCG@10 | N vector ratio |
|---:|---|---|---:|---:|---:|---:|
| 128 | N | per_channel | 0.725556 | 0.725556 | 0.637579 | 0.167 |
| 128 | N | total | 0.725556 | 0.725556 | 0.637579 | 0.167 |
| 128 | N+S | per_channel | 0.795417 | 0.795417 | 0.706289 | 0.167 |
| 128 | N+S | total | 0.794444 | 0.794444 | 0.705245 | 0.167 |
| 256 | N | per_channel | 0.824722 | 0.824722 | 0.739978 | 0.333 |
| 256 | N | total | 0.824722 | 0.824722 | 0.739978 | 0.333 |
| 256 | N+S | per_channel | 0.825139 | 0.825139 | 0.741056 | 0.333 |
| 256 | N+S | total | 0.825139 | 0.825139 | 0.740661 | 0.333 |
| 512 | N | per_channel | 0.835556 | 0.835556 | 0.754337 | 0.667 |
| 512 | N | total | 0.835556 | 0.835556 | 0.754337 | 0.667 |
| 512 | N+S | per_channel | 0.826528 | 0.826528 | 0.743842 | 0.667 |
| 512 | N+S | total | 0.826250 | 0.826250 | 0.743679 | 0.667 |
| 768 | N | per_channel | 0.837500 | 0.837500 | 0.758208 | 1.000 |
| 768 | N | total | 0.837500 | 0.837500 | 0.758208 | 1.000 |
| 768 | N+S | per_channel | 0.829167 | 0.829167 | 0.744466 | 1.000 |
| 768 | N+S | total | 0.829444 | 0.829444 | 0.744437 | 1.000 |

Specialist fusion: validated.

Current array/ID hashes are newly observed integrity records. Legacy caches lack original independent vector checksums.

# Cache-only EG2 dimension sweep

Status: **completed**.

Fresh model inference calls: 0. No inference speed is inferred from cached embeddings.

The unchanged 768-dimensional baseline was reproduced before evaluation. Prefixes search the full frozen corpus.

Raw FP32 N-vector payload only; excludes model, process RSS, index overhead and specialist storage.

| Dimension | Method | Budget | Hit@5 | Recall@5 | nDCG@10 | N vector ratio |
|---:|---|---|---:|---:|---:|---:|
| 128 | N | per_channel | 0.666528 | 0.666528 | 0.576553 | 0.167 |
| 128 | N | total | 0.666528 | 0.666528 | 0.576553 | 0.167 |
| 128 | N+S | per_channel | 0.726531 | 0.726531 | 0.634936 | 0.167 |
| 128 | N+S | total | 0.727222 | 0.727222 | 0.636113 | 0.167 |
| 256 | N | per_channel | 0.820683 | 0.820683 | 0.735888 | 0.333 |
| 256 | N | total | 0.820683 | 0.820683 | 0.735888 | 0.333 |
| 256 | N+S | per_channel | 0.757362 | 0.757362 | 0.664766 | 0.333 |
| 256 | N+S | total | 0.761233 | 0.761233 | 0.669140 | 0.333 |
| 512 | N | per_channel | 0.845431 | 0.845431 | 0.760824 | 0.667 |
| 512 | N | total | 0.845431 | 0.845431 | 0.760824 | 0.667 |
| 512 | N+S | per_channel | 0.762201 | 0.762201 | 0.670869 | 0.667 |
| 512 | N+S | total | 0.766902 | 0.766902 | 0.674953 | 0.667 |
| 768 | N | per_channel | 0.846398 | 0.846398 | 0.763934 | 1.000 |
| 768 | N | total | 0.846398 | 0.846398 | 0.763934 | 1.000 |
| 768 | N+S | per_channel | 0.762754 | 0.762754 | 0.672394 | 1.000 |
| 768 | N+S | total | 0.767455 | 0.767455 | 0.675927 | 1.000 |

Specialist fusion: validated.

Current array/ID hashes are newly observed integrity records. Legacy caches lack original independent vector checksums.

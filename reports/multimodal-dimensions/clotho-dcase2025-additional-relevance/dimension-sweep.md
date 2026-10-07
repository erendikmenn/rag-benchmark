# Cache-only EG2 dimension sweep

Status: **completed**.

Fresh model inference calls: 0. No inference speed is inferred from cached embeddings.

The unchanged 768-dimensional baseline was reproduced before evaluation. Prefixes search the full frozen corpus.

Raw FP32 N-vector payload only; excludes model, process RSS, index overhead and specialist storage.

| Dimension | Method | Budget | Hit@5 | Recall@5 | nDCG@10 | N vector ratio |
|---:|---|---|---:|---:|---:|---:|
| 128 | N | per_channel | 0.185149 | 0.076380 | 0.086853 | 0.167 |
| 128 | N | total | 0.185149 | 0.076380 | 0.086853 | 0.167 |
| 128 | N+S | per_channel | 0.444552 | 0.210859 | 0.236282 | 0.167 |
| 128 | N+S | total | 0.449373 | 0.218236 | 0.243362 | 0.167 |
| 256 | N | per_channel | 0.236258 | 0.106032 | 0.118656 | 0.333 |
| 256 | N | total | 0.236258 | 0.106032 | 0.118656 | 0.333 |
| 256 | N+S | per_channel | 0.500482 | 0.249985 | 0.276953 | 0.333 |
| 256 | N+S | total | 0.509161 | 0.257949 | 0.281514 | 0.333 |
| 512 | N | per_channel | 0.261331 | 0.115063 | 0.124660 | 0.667 |
| 512 | N | total | 0.261331 | 0.115063 | 0.124660 | 0.667 |
| 512 | N+S | per_channel | 0.502411 | 0.249829 | 0.279825 | 0.667 |
| 512 | N+S | total | 0.513018 | 0.259438 | 0.284060 | 0.667 |
| 768 | N | per_channel | 0.258438 | 0.113189 | 0.121080 | 1.000 |
| 768 | N | total | 0.258438 | 0.113189 | 0.121080 | 1.000 |
| 768 | N+S | per_channel | 0.507232 | 0.253953 | 0.276440 | 1.000 |
| 768 | N+S | total | 0.521697 | 0.264408 | 0.285327 | 1.000 |

Specialist fusion: validated.

Current array/ID hashes are newly observed integrity records. Legacy caches lack original independent vector checksums.

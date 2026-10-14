# Measured multimodal runs

Each row is a separate run. Query and candidate counts across language/protocol views overlap and must not be added as independent data. Partial runs and synthetic model preflights are not full-split accuracy results. The 1,284-family registry remains planned scope.

| Run | Scope | Queries | Candidates | Completed primary cells | Failed cells |
|---|---|---:|---:|---:|---:|
| [vidore-v3-pharmaceuticals-en-cuda-all-retrieval](vidore-v3-pharmaceuticals-en-cuda-all-retrieval/report.md) | frozen_collection | 364 / 364 | 2,313 | 63 | 0 |
| [vidore-v3-physics-fr-cuda-all-retrieval](vidore-v3-physics-fr-cuda-all-retrieval/report.md) | frozen_collection | 302 / 302 | 1,674 | 63 | 0 |

A failed comparison does not erase completed methods in the same run. Each report retains all coverage states and records the actual model configuration. Hit/Recall measure source retrieval, not generated-answer correctness.

# Measured multimodal runs

Each row is a separate run. Query and candidate counts across language/protocol views overlap and must not be added as independent data. Partial runs and synthetic model preflights are not full-split accuracy results. The 1,284-family registry remains planned scope.

| Run | Scope | Queries | Candidates | Completed primary cells | Failed cells |
|---|---|---:|---:|---:|---:|
| [codesearchnet-go-test-bm25](codesearchnet-go-test-bm25/report.md) | frozen_collection | 8,122 / 8,122 | 28,120 | 1 | 0 |
| [codesearchnet-java-test-bm25](codesearchnet-java-test-bm25/report.md) | frozen_collection | 10,955 / 10,955 | 40,347 | 1 | 0 |
| [vidore-cs-bm25](vidore-cs-bm25/report.md) | frozen_collection | 215 / 215 | 1,360 | 1 | 0 |
| [vidore-cs-native](vidore-cs-native/report.md) | frozen_collection | 215 / 215 | 1,360 | 1 | 0 |
| [xm3600-en-native-specialist](xm3600-en-native-specialist/report.md) | frozen_collection | 7,200 / 7,200 | 3,600 | 3 | 0 |
| [xm3600-tr-native](xm3600-tr-native/report.md) | frozen_collection | 7,233 / 7,233 | 3,600 | 1 | 40 |

A failed comparison does not erase completed methods in the same run. Each report retains all coverage states and records the actual model configuration. Hit/Recall measure source retrieval, not generated-answer correctness.

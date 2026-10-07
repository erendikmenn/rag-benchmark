# Published primary-matrix progress

**1,928 / 18,460 primary experiment cells complete (10.44%).** 16,532 remain. This is a published-result snapshot, not a percentage of elapsed time or all project work.

Snapshot: 2026-10-07T15:28:57.889820+00:00.

The denominator fixes **21 main collections**: two XM3600 languages, eight ViDoRe collections, two Clotho relevance protocols, one FLEURS collection, one MSR-VTT collection, six CodeSearchNet languages and CIRR validation. They expand to **923 collection × retrieval-subset conditions**. Each has two budgets and `1 no-reranker + 3 rerankers × K=20/50/100`: **923 × 2 × 10 = 18,460 cells**.

| Track | Collections | Completed cells | Planned cells | Coverage |
|---|---:|---:|---:|---:|
| photo | 2 | 12 | 2,520 | 0.48% |
| document | 8 | 1,020 | 10,080 | 10.12% |
| environment_audio | 2 | 252 | 2,520 | 10.00% |
| speech | 1 | 434 | 620 | 70.00% |
| video | 1 | 126 | 1,260 | 10.00% |
| code | 6 | 84 | 840 | 10.00% |
| composed_image | 1 | 0 | 620 | 0.00% |

Only `completed` cells using every query in a frozen collection count. The key is logical collection + method family + budget + rerank K. Historical BM25/native exports, the Laya milestone and other overlapping snapshots are counted once; all duplicate metric values must agree. The FLEURS native result repeated under the ASR-view identity remains the same logical condition.

The input contains 2,662 completed export cells; 734 duplicate occurrences are removed. There are zero duplicate metric mismatches. Failed, unsupported and planned cells contribute zero. The five-query Gemma technical smoke contributes zero.

**418 / 1,284 global method families (32.55%)** have at least one full-collection result. That higher percentage collapses languages and collections and must not replace the primary-matrix percentage. 20 collections have some completed cells; 0 have every primary cell completed.

The **120 published dimension-sweep cells** are separate derived experiments and are excluded from both sides of this main-matrix fraction. Other representation/parameter ablations, human-validated language/paraphrase extensions, reverse tasks and QA do not have a fully enumerated common denominator here. Active or unpublished work is excluded. CIRR stays in the denominator while its authorized media access is unresolved.

[Method registry](../../configs/multimodal-variants.csv) · [Experiment plan](../../docs/multimodal-plan.tr.md) · [Exact counts, per-collection coverage and source hashes](progress-summary.json)

# Jev and GPT 6.1 Sol Ultra judge comparison

Same saved answers, different automated judges. Agreement is not accuracy; no human adjudication has been performed.

Same 20 questions, 200 answers and frozen rubrics. 319 unique benchmark judgments per evaluator. Sol used separate blinded Codex agents for correctness and grounding.

| Variant | Answers | Jev correct | Sol correct | Jev correct + supported | Sol correct + supported |
|---|---:|---:|---:|---:|---:|
| bge | 20 | 12 | 12 | 12 | 11 |
| bge_laya | 20 | 9 | 9 | 9 | 9 |
| bm25 | 20 | 11 | 14 | 11 | 13 |
| bm25_bge | 20 | 13 | 13 | 13 | 12 |
| bm25_bge_laya | 20 | 8 | 8 | 8 | 8 |
| bm25_embeddinggemma | 20 | 14 | 14 | 14 | 13 |
| bm25_embeddinggemma_laya | 20 | 7 | 8 | 7 | 8 |
| bm25_laya | 20 | 9 | 9 | 9 | 9 |
| embeddinggemma | 20 | 14 | 11 | 14 | 10 |
| embeddinggemma_laya | 20 | 8 | 9 | 8 | 9 |

All rates retain every planned answer, including abstained and unjudgeable outcomes. Repeated questions across variants are not independent samples. Greater agreement or higher scores alone do not establish the better judge.

Model and effort configured in Codex spawn_agent; a served snapshot, token usage and USD cost are not exposed by this workflow.

[Full summary](summary.json) · [Per-answer labels](per-answer-comparison.jsonl)

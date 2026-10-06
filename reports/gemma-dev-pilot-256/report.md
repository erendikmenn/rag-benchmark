# RAG benchmark export

Split: **dev**. Exported rows: **40**.

Pilot and partial runs are not final benchmark claims. Completeness is relative to the selected question set.

| Variant | Questions | Complete | Recall@5 | nDCG@10 | Answer EM | Token F1 |
|---|---:|---|---:|---:|---:|---:|
| bm25 | 20 | True | 0.8667 | 0.8603 | 0.0500 | 0.4998 |
| bm25_laya | 20 | True | 0.6583 | 0.6283 | 0.0000 | 0.4033 |

Answer scores measure lexical agreement, not factual correctness. Synthetic labels and source-quality limitations apply.

The JSON files contain provenance, aggregate breakdowns, paired intervals, and per-query IDs/metrics. Source text, questions, reference answers, generated answers, prompts, and local paths are excluded.

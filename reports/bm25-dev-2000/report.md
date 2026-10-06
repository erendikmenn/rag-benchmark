# RAG benchmark export

Split: **dev**. Exported rows: **2000**.

Pilot and partial runs are not final benchmark claims. Completeness is relative to the selected question set.

| Variant | Questions | Complete | Recall@5 | nDCG@10 | Answer EM | Token F1 |
|---|---:|---|---:|---:|---:|---:|
| bm25 | 2000 | True | 0.8591 | 0.8434 | — | — |

Answer scores measure lexical agreement, not factual correctness. Synthetic labels and source-quality limitations apply.

The JSON files contain provenance, aggregate breakdowns, paired intervals, and per-query IDs/metrics. Source text, questions, reference answers, generated answers, prompts, and local paths are excluded.

# RAG benchmark export

Split: **dev**. Exported rows: **200**.

Pilot and partial runs are not final benchmark claims. Completeness is relative to the selected question set.

| Variant | Questions | Complete | Recall@5 | nDCG@10 | Answer EM | Token F1 |
|---|---:|---|---:|---:|---:|---:|
| bm25 | 20 | True | 0.8667 | 0.8603 | 0.0500 | 0.5100 |
| bm25_laya | 20 | True | 0.6583 | 0.6283 | 0.0000 | 0.4085 |
| bge | 20 | True | 0.8833 | 0.8633 | 0.0500 | 0.4661 |
| bge_laya | 20 | True | 0.6917 | 0.6270 | 0.0000 | 0.4409 |
| embeddinggemma | 20 | True | 0.8167 | 0.8133 | 0.0000 | 0.4592 |
| embeddinggemma_laya | 20 | True | 0.6417 | 0.5736 | 0.0000 | 0.4112 |
| bm25_bge | 20 | True | 0.9333 | 0.9217 | 0.0500 | 0.5209 |
| bm25_bge_laya | 20 | True | 0.6250 | 0.6241 | 0.0000 | 0.4126 |
| bm25_embeddinggemma | 20 | True | 0.9500 | 0.8854 | 0.0500 | 0.5326 |
| bm25_embeddinggemma_laya | 20 | True | 0.5917 | 0.6083 | 0.0000 | 0.3986 |

Answer scores measure lexical agreement, not factual correctness. Synthetic labels and source-quality limitations apply.

The JSON files contain provenance, aggregate breakdowns, paired intervals, and per-query IDs/metrics. Source text, questions, reference answers, generated answers, prompts, and local paths are excluded.

Optional candidate_scores and ranked_scores align with their respective ID arrays. Score types: bm25 = lexical BM25 score; cosine_similarity = normalized dense-vector dot product; rrf = reciprocal-rank fusion score; laya_p_true = Laya's relevance P(true) model output, not a calibrated probability. Higher scores rank first. Different score types are not directly comparable; older runs may omit scores.

# Paired retrieval comparisons

Dataset: vidore-v3-pharmaceuticals-en.

Differences are right minus left, in metric units. Positive values favor the right method. Where reported, intervals are 95% paired source-group percentile intervals; methods use the same frozen queries and gallery. These are marginal exploratory comparisons without a multiple-comparison correction.

Grouping: Positive pages map to corpus.metadata.doc_id. Queries touching the same source document are connected transitively across multi-document positives.

Source groups: 23; queries: 364; largest group: 70 queries (19.2%).

Conservative reporting safeguard: withhold intervals when one source group contains more than 50% of queries. This is not a formal statistical threshold.

| Left | Right | Metric | Left mean | Right mean | Delta | 95% interval | Wins / losses / ties |
|---|---|---|---:|---:|---:|---|---|
| document__n__none | document__s__none | hit@5 | 0.8187 | 0.8956 | +0.0769 | [+0.0355, +0.1198] | 40 / 12 / 312 |
| document__n__none | document__s__none | ndcg@10 | 0.5521 | 0.6490 | +0.0970 | [+0.0654, +0.1410] | 185 / 86 / 93 |
| document__n__none | document__n_s__none | hit@5 | 0.8187 | 0.8901 | +0.0714 | [+0.0346, +0.1096] | 32 / 6 / 326 |
| document__n__none | document__n_s__none | ndcg@10 | 0.5521 | 0.6200 | +0.0679 | [+0.0476, +0.0943] | 176 / 58 / 130 |
| document__s__none | document__n_s__none | hit@5 | 0.8956 | 0.8901 | -0.0055 | [-0.0246, +0.0123] | 12 / 14 / 338 |
| document__s__none | document__n_s__none | ndcg@10 | 0.6490 | 0.6200 | -0.0291 | [-0.0518, -0.0122] | 106 / 135 / 123 |
| document__n__none | document__j__none | hit@5 | 0.8187 | 0.8599 | +0.0412 | [+0.0125, +0.0746] | 29 / 14 / 321 |
| document__n__none | document__j__none | ndcg@10 | 0.5521 | 0.6025 | +0.0504 | [+0.0271, +0.0801] | 165 / 76 / 123 |
| document__b__none | document__g__none | hit@5 | 0.8159 | 0.8297 | +0.0137 | [-0.0239, +0.0463] | 35 / 30 / 299 |
| document__b__none | document__g__none | ndcg@10 | 0.5529 | 0.5502 | -0.0027 | [-0.0313, +0.0219] | 135 / 140 / 89 |
| document__b__none | document__e__none | hit@5 | 0.8159 | 0.8462 | +0.0302 | [+0.0000, +0.0572] | 33 / 22 / 309 |
| document__b__none | document__e__none | ndcg@10 | 0.5529 | 0.5783 | +0.0254 | [+0.0018, +0.0509] | 157 / 122 / 85 |
| document__g__none | document__e__none | hit@5 | 0.8297 | 0.8462 | +0.0165 | [-0.0184, +0.0476] | 25 / 19 / 320 |
| document__g__none | document__e__none | ndcg@10 | 0.5502 | 0.5783 | +0.0281 | [+0.0078, +0.0493] | 152 / 106 / 106 |
| document__b__none | document__b_g_e__none | hit@5 | 0.8159 | 0.8764 | +0.0604 | [+0.0230, +0.0976] | 31 / 9 / 324 |
| document__b__none | document__b_g_e__none | ndcg@10 | 0.5529 | 0.6154 | +0.0625 | [+0.0413, +0.0814] | 167 / 76 / 121 |

Bootstrap settings for eligible comparisons: 5000 samples; seed: 42. No model inference or generated-answer judging was performed.

Pairing follows [SciPy's paired resampling definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html); source groups follow [cluster resampling](https://www.stata.com/support/faqs/statistics/bootstrap-with-panel-data/).

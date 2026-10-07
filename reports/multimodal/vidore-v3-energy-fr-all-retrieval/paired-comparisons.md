# Paired retrieval comparisons

Dataset: vidore-v3-energy-fr.

Differences are right minus left, in metric units. Positive values favor the right method. Where reported, intervals are 95% paired source-group percentile intervals; methods use the same frozen queries and gallery. These are marginal exploratory comparisons without a multiple-comparison correction.

Grouping: Positive pages map to corpus.metadata.doc_id. Queries touching the same source document are connected transitively across multi-document positives.

Source groups: 26; queries: 308; largest group: 80 queries (26.0%).

Conservative reporting safeguard: withhold intervals when one source group contains more than 50% of queries. This is not a formal statistical threshold.

| Left | Right | Metric | Left mean | Right mean | Delta | 95% interval | Wins / losses / ties |
|---|---|---|---:|---:|---:|---|---|
| document__n__none | document__s__none | hit@5 | 0.6883 | 0.8669 | +0.1786 | [+0.0887, +0.2383] | 70 / 15 / 223 |
| document__n__none | document__s__none | ndcg@10 | 0.4461 | 0.6465 | +0.2004 | [+0.1430, +0.2497] | 189 / 65 / 54 |
| document__n__none | document__n_s__none | hit@5 | 0.6883 | 0.8182 | +0.1299 | [+0.0598, +0.1847] | 43 / 3 / 262 |
| document__n__none | document__n_s__none | ndcg@10 | 0.4461 | 0.5696 | +0.1235 | [+0.1018, +0.1447] | 183 / 36 / 89 |
| document__s__none | document__n_s__none | hit@5 | 0.8669 | 0.8182 | -0.0487 | [-0.1008, -0.0099] | 13 / 28 / 267 |
| document__s__none | document__n_s__none | ndcg@10 | 0.6465 | 0.5696 | -0.0769 | [-0.1172, -0.0324] | 82 / 132 / 94 |
| document__n__none | document__j__none | hit@5 | 0.6883 | 0.8377 | +0.1494 | [+0.0578, +0.2137] | 55 / 9 / 244 |
| document__n__none | document__j__none | ndcg@10 | 0.4461 | 0.5829 | +0.1368 | [+0.0772, +0.1784] | 167 / 61 / 80 |
| document__b__none | document__g__none | hit@5 | 0.7727 | 0.7532 | -0.0195 | [-0.0774, +0.0216] | 32 / 38 / 238 |
| document__b__none | document__g__none | ndcg@10 | 0.5568 | 0.5342 | -0.0225 | [-0.0724, +0.0239] | 109 / 126 / 73 |
| document__b__none | document__e__none | hit@5 | 0.7727 | 0.8279 | +0.0552 | [-0.0122, +0.1014] | 42 / 25 / 241 |
| document__b__none | document__e__none | ndcg@10 | 0.5568 | 0.5732 | +0.0165 | [-0.0210, +0.0633] | 124 / 114 / 70 |
| document__g__none | document__e__none | hit@5 | 0.7532 | 0.8279 | +0.0747 | [+0.0305, +0.1154] | 41 / 18 / 249 |
| document__g__none | document__e__none | ndcg@10 | 0.5342 | 0.5732 | +0.0390 | [+0.0155, +0.0738] | 120 / 92 / 96 |
| document__b__none | document__b_g_e__none | hit@5 | 0.7727 | 0.8636 | +0.0909 | [+0.0379, +0.1275] | 38 / 10 / 260 |
| document__b__none | document__b_g_e__none | ndcg@10 | 0.5568 | 0.6206 | +0.0639 | [+0.0285, +0.0966] | 146 / 74 / 88 |

Bootstrap settings for eligible comparisons: 5000 samples; seed: 42. No model inference or generated-answer judging was performed.

Pairing follows [SciPy's paired resampling definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html); source groups follow [cluster resampling](https://www.stata.com/support/faqs/statistics/bootstrap-with-panel-data/).

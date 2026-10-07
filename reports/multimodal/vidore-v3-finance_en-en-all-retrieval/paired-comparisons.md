# Paired retrieval comparisons

Dataset: vidore-v3-finance_en-en.

Differences are right minus left, in metric units. Positive values favor the right method. Where reported, intervals are 95% paired source-group percentile intervals; methods use the same frozen queries and gallery. These are marginal exploratory comparisons without a multiple-comparison correction.

Grouping: Positive pages map to corpus.metadata.doc_id. Queries touching the same source document are connected transitively across multi-document positives.

Source groups: 1; queries: 309; largest group: 309 queries (100.0%).

Conservative reporting safeguard: withhold intervals when one source group contains more than 50% of queries. This is not a formal statistical threshold.

| Left | Right | Metric | Left mean | Right mean | Delta | 95% interval | Wins / losses / ties |
|---|---|---|---:|---:|---:|---|---|
| document__n__none | document__s__none | hit@5 | 0.7767 | 0.8770 | +0.1003 | Not reported; see reason below | 55 / 24 / 230 |
| document__n__none | document__s__none | ndcg@10 | 0.4795 | 0.6170 | +0.1374 | Not reported; see reason below | 177 / 84 / 48 |
| document__n__none | document__n_s__none | hit@5 | 0.7767 | 0.8608 | +0.0841 | Not reported; see reason below | 31 / 5 / 273 |
| document__n__none | document__n_s__none | ndcg@10 | 0.4795 | 0.5833 | +0.1038 | Not reported; see reason below | 187 / 45 / 77 |
| document__s__none | document__n_s__none | hit@5 | 0.8770 | 0.8608 | -0.0162 | Not reported; see reason below | 21 / 26 / 262 |
| document__s__none | document__n_s__none | ndcg@10 | 0.6170 | 0.5833 | -0.0336 | Not reported; see reason below | 118 / 128 / 63 |
| document__n__none | document__j__none | hit@5 | 0.7767 | 0.8511 | +0.0744 | Not reported; see reason below | 32 / 9 / 268 |
| document__n__none | document__j__none | ndcg@10 | 0.4795 | 0.5716 | +0.0921 | Not reported; see reason below | 163 / 80 / 66 |
| document__b__none | document__g__none | hit@5 | 0.7896 | 0.7023 | -0.0874 | Not reported; see reason below | 31 / 58 / 220 |
| document__b__none | document__g__none | ndcg@10 | 0.5313 | 0.4387 | -0.0926 | Not reported; see reason below | 102 / 157 / 50 |
| document__b__none | document__e__none | hit@5 | 0.7896 | 0.8608 | +0.0712 | Not reported; see reason below | 43 / 21 / 245 |
| document__b__none | document__e__none | ndcg@10 | 0.5313 | 0.5583 | +0.0270 | Not reported; see reason below | 148 / 109 / 52 |
| document__g__none | document__e__none | hit@5 | 0.7023 | 0.8608 | +0.1586 | Not reported; see reason below | 55 / 6 / 248 |
| document__g__none | document__e__none | ndcg@10 | 0.4387 | 0.5583 | +0.1195 | Not reported; see reason below | 176 / 76 / 57 |
| document__b__none | document__b_g_e__none | hit@5 | 0.7896 | 0.8576 | +0.0680 | Not reported; see reason below | 38 / 17 / 254 |
| document__b__none | document__b_g_e__none | ndcg@10 | 0.5313 | 0.5864 | +0.0550 | Not reported; see reason below | 157 / 89 / 63 |

document__n__none → document__s__none: Only one connected source group; a resampling confidence interval is not reported.

document__n__none → document__n_s__none: Only one connected source group; a resampling confidence interval is not reported.

document__s__none → document__n_s__none: Only one connected source group; a resampling confidence interval is not reported.

document__n__none → document__j__none: Only one connected source group; a resampling confidence interval is not reported.

document__b__none → document__g__none: Only one connected source group; a resampling confidence interval is not reported.

document__b__none → document__e__none: Only one connected source group; a resampling confidence interval is not reported.

document__g__none → document__e__none: Only one connected source group; a resampling confidence interval is not reported.

document__b__none → document__b_g_e__none: Only one connected source group; a resampling confidence interval is not reported.

Bootstrap settings for eligible comparisons: 5000 samples; seed: 42. No model inference or generated-answer judging was performed.

Pairing follows [SciPy's paired resampling definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html); source groups follow [cluster resampling](https://www.stata.com/support/faqs/statistics/bootstrap-with-panel-data/).

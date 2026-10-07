# Paired retrieval comparisons

Dataset: vidore-v3-computer_science-en.

Differences are right minus left, in metric units. Positive values favor the right method. Where reported, intervals are 95% paired source-group percentile intervals; methods use the same frozen queries and gallery. These are marginal exploratory comparisons without a multiple-comparison correction.

Grouping: Positive pages map to corpus.metadata.doc_id. Queries touching the same source document are connected transitively across multi-document positives.

Source groups: 1; queries: 215; largest group: 215 queries (100.0%).

Conservative reporting safeguard: withhold intervals when one source group contains more than 50% of queries. This is not a formal statistical threshold.

| Left | Right | Metric | Left mean | Right mean | Delta | 95% interval | Wins / losses / ties |
|---|---|---|---:|---:|---:|---|---|
| document__n__none | document__s__none | hit@5 | 0.9163 | 0.9721 | +0.0558 | Not reported; see reason below | 15 / 3 / 197 |
| document__n__none | document__s__none | ndcg@10 | 0.6296 | 0.7585 | +0.1288 | Not reported; see reason below | 136 / 54 / 25 |
| document__n__none | document__n_s__none | hit@5 | 0.9163 | 0.9674 | +0.0512 | Not reported; see reason below | 14 / 3 / 198 |
| document__n__none | document__n_s__none | ndcg@10 | 0.6296 | 0.7264 | +0.0968 | Not reported; see reason below | 140 / 34 / 41 |
| document__s__none | document__n_s__none | hit@5 | 0.9721 | 0.9674 | -0.0047 | Not reported; see reason below | 3 / 4 / 208 |
| document__s__none | document__n_s__none | ndcg@10 | 0.7585 | 0.7264 | -0.0320 | Not reported; see reason below | 67 / 102 / 46 |
| document__n__none | document__j__none | hit@5 | 0.9163 | 0.9442 | +0.0279 | Not reported; see reason below | 8 / 2 / 205 |
| document__n__none | document__j__none | ndcg@10 | 0.6296 | 0.6753 | +0.0457 | Not reported; see reason below | 110 / 64 / 41 |
| document__b__none | document__g__none | hit@5 | 0.9209 | 0.9535 | +0.0326 | Not reported; see reason below | 15 / 8 / 192 |
| document__b__none | document__g__none | ndcg@10 | 0.6334 | 0.6356 | +0.0022 | Not reported; see reason below | 92 / 87 / 36 |
| document__b__none | document__e__none | hit@5 | 0.9209 | 0.9442 | +0.0233 | Not reported; see reason below | 13 / 8 / 194 |
| document__b__none | document__e__none | ndcg@10 | 0.6334 | 0.6623 | +0.0288 | Not reported; see reason below | 104 / 82 / 29 |
| document__g__none | document__e__none | hit@5 | 0.9535 | 0.9442 | -0.0093 | Not reported; see reason below | 2 / 4 / 209 |
| document__g__none | document__e__none | ndcg@10 | 0.6356 | 0.6623 | +0.0267 | Not reported; see reason below | 101 / 70 / 44 |
| document__b__none | document__b_g_e__none | hit@5 | 0.9209 | 0.9674 | +0.0465 | Not reported; see reason below | 12 / 2 / 201 |
| document__b__none | document__b_g_e__none | ndcg@10 | 0.6334 | 0.7005 | +0.0671 | Not reported; see reason below | 119 / 53 / 43 |

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

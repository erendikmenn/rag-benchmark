# Paired retrieval comparisons

Dataset: clotho-dcase2025-additional-relevance-described.

Differences are right minus left, in metric units. Positive values favor the right method. Where reported, intervals are 95% paired source-group percentile intervals; methods use the same frozen queries and gallery. These are marginal exploratory comparisons without a multiple-comparison correction.

Grouping: Queries sharing any positively labelled media item are connected; all queries in each connected component are resampled together.

Source groups: 122; queries: 1037; largest group: 897 queries (86.5%).

Conservative reporting safeguard: withhold intervals when one source group contains more than 50% of queries. This is not a formal statistical threshold.

| Left | Right | Metric | Left mean | Right mean | Delta | 95% interval | Wins / losses / ties |
|---|---|---|---:|---:|---:|---|---|
| environment_audio__n__none | environment_audio__s__none | hit@5 | 0.2584 | 0.6451 | +0.3867 | Not reported; see reason below | 468 / 67 / 502 |
| environment_audio__n__none | environment_audio__s__none | ndcg@10 | 0.1211 | 0.3771 | +0.2560 | Not reported; see reason below | 706 / 135 / 196 |
| environment_audio__n__none | environment_audio__n_s__none | hit@5 | 0.2584 | 0.5072 | +0.2488 | Not reported; see reason below | 284 / 26 / 727 |
| environment_audio__n__none | environment_audio__n_s__none | ndcg@10 | 0.1211 | 0.2764 | +0.1554 | Not reported; see reason below | 600 / 67 / 370 |
| environment_audio__s__none | environment_audio__n_s__none | hit@5 | 0.6451 | 0.5072 | -0.1379 | Not reported; see reason below | 94 / 237 / 706 |
| environment_audio__s__none | environment_audio__n_s__none | ndcg@10 | 0.3771 | 0.2764 | -0.1006 | Not reported; see reason below | 316 / 521 / 200 |
| environment_audio__n__none | environment_audio__j__none | hit@5 | 0.2584 | 0.1880 | -0.0704 | Not reported; see reason below | 94 / 167 / 776 |
| environment_audio__n__none | environment_audio__j__none | ndcg@10 | 0.1211 | 0.0775 | -0.0436 | Not reported; see reason below | 161 / 298 / 578 |
| environment_audio__b__none | environment_audio__g__none | hit@5 | 0.0656 | 0.0829 | +0.0174 | Not reported; see reason below | 59 / 41 / 937 |
| environment_audio__b__none | environment_audio__g__none | ndcg@10 | 0.0293 | 0.0345 | +0.0052 | Not reported; see reason below | 105 / 88 / 844 |
| environment_audio__b__none | environment_audio__e__none | hit@5 | 0.0656 | 0.0839 | +0.0183 | Not reported; see reason below | 53 / 34 / 950 |
| environment_audio__b__none | environment_audio__e__none | ndcg@10 | 0.0293 | 0.0398 | +0.0105 | Not reported; see reason below | 127 / 81 / 829 |
| environment_audio__g__none | environment_audio__e__none | hit@5 | 0.0829 | 0.0839 | +0.0010 | Not reported; see reason below | 41 / 40 / 956 |
| environment_audio__g__none | environment_audio__e__none | ndcg@10 | 0.0345 | 0.0398 | +0.0053 | Not reported; see reason below | 101 / 87 / 849 |
| environment_audio__b__none | environment_audio__b_g_e__none | hit@5 | 0.0656 | 0.0849 | +0.0193 | Not reported; see reason below | 47 / 27 / 963 |
| environment_audio__b__none | environment_audio__b_g_e__none | ndcg@10 | 0.0293 | 0.0373 | +0.0080 | Not reported; see reason below | 107 / 63 / 867 |

environment_audio__n__none → environment_audio__s__none: Largest connected source group contains 86.5% of queries (>50%); interval withheld under a conservative reporting safeguard, not a formal statistical threshold.

environment_audio__n__none → environment_audio__n_s__none: Largest connected source group contains 86.5% of queries (>50%); interval withheld under a conservative reporting safeguard, not a formal statistical threshold.

environment_audio__s__none → environment_audio__n_s__none: Largest connected source group contains 86.5% of queries (>50%); interval withheld under a conservative reporting safeguard, not a formal statistical threshold.

environment_audio__n__none → environment_audio__j__none: Largest connected source group contains 86.5% of queries (>50%); interval withheld under a conservative reporting safeguard, not a formal statistical threshold.

environment_audio__b__none → environment_audio__g__none: Largest connected source group contains 86.5% of queries (>50%); interval withheld under a conservative reporting safeguard, not a formal statistical threshold.

environment_audio__b__none → environment_audio__e__none: Largest connected source group contains 86.5% of queries (>50%); interval withheld under a conservative reporting safeguard, not a formal statistical threshold.

environment_audio__g__none → environment_audio__e__none: Largest connected source group contains 86.5% of queries (>50%); interval withheld under a conservative reporting safeguard, not a formal statistical threshold.

environment_audio__b__none → environment_audio__b_g_e__none: Largest connected source group contains 86.5% of queries (>50%); interval withheld under a conservative reporting safeguard, not a formal statistical threshold.

Bootstrap settings for eligible comparisons: 5000 samples; seed: 42. No model inference or generated-answer judging was performed.

Pairing follows [SciPy's paired resampling definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html); source groups follow [cluster resampling](https://www.stata.com/support/faqs/statistics/bootstrap-with-panel-data/).

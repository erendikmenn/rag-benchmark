# Paired retrieval comparisons

Dataset: msrvtt-1k-a.

Differences are right minus left, in metric units. Positive values favor the right method. Where reported, intervals are 95% paired source-group percentile intervals; methods use the same frozen queries and gallery. These are marginal exploratory comparisons without a multiple-comparison correction.

Grouping: Queries sharing any positively labelled media item are connected; all queries in each connected component are resampled together.

Source groups: 1000; queries: 1000; largest group: 1 queries (0.1%).

Conservative reporting safeguard: withhold intervals when one source group contains more than 50% of queries. This is not a formal statistical threshold.

| Left | Right | Metric | Left mean | Right mean | Delta | 95% interval | Wins / losses / ties |
|---|---|---|---:|---:|---:|---|---|
| video__n__none | video__s__none | hit@5 | 0.7500 | 0.5380 | -0.2120 | [-0.2420, -0.1820] | 38 / 250 / 712 |
| video__n__none | video__s__none | ndcg@10 | 0.6694 | 0.4594 | -0.2101 | [-0.2322, -0.1873] | 103 / 465 / 432 |
| video__n__none | video__n_s__none | hit@5 | 0.7500 | 0.6710 | -0.0790 | [-0.1040, -0.0540] | 46 / 125 / 829 |
| video__n__none | video__n_s__none | ndcg@10 | 0.6694 | 0.5900 | -0.0795 | [-0.0972, -0.0618] | 146 / 276 / 578 |
| video__s__none | video__n_s__none | hit@5 | 0.5380 | 0.6710 | +0.1330 | [+0.1100, +0.1560] | 144 / 11 / 845 |
| video__s__none | video__n_s__none | ndcg@10 | 0.4594 | 0.5900 | +0.1306 | [+0.1171, +0.1432] | 404 / 44 / 552 |

Bootstrap settings for eligible comparisons: 5000 samples; seed: 42. No model inference or generated-answer judging was performed.

Pairing follows [SciPy's paired resampling definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html); source groups follow [cluster resampling](https://www.stata.com/support/faqs/statistics/bootstrap-with-panel-data/).

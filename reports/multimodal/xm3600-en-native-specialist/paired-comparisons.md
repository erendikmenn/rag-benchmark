# Paired retrieval comparisons

Dataset: xm3600-en.

Differences are right minus left, in metric units. Positive values favor the right method. Intervals are 95% paired source-group percentile intervals; methods use the same frozen queries and gallery. These are marginal exploratory comparisons without a multiple-comparison correction.

Grouping: Queries sharing any positively labelled media item are connected; all queries in each connected component are resampled together.

Source groups: 3600; queries: 7200; largest group: 2 queries.

| Left | Right | Metric | Left mean | Right mean | Delta | 95% interval | Wins / losses / ties |
|---|---|---|---:|---:|---:|---|---|
| photo__n__none | photo__s__none | hit@5 | 0.8375 | 0.7689 | -0.0686 | [-0.0782, -0.0594] | 288 / 782 / 6130 |
| photo__n__none | photo__s__none | ndcg@10 | 0.7582 | 0.6794 | -0.0788 | [-0.0863, -0.0712] | 1019 / 2093 / 4088 |
| photo__n__none | photo__n_s__none | hit@5 | 0.8375 | 0.8292 | -0.0083 | [-0.0156, -0.0014] | 262 / 322 / 6616 |
| photo__n__none | photo__n_s__none | ndcg@10 | 0.7582 | 0.7445 | -0.0137 | [-0.0188, -0.0088] | 1116 / 1117 / 4967 |
| photo__s__none | photo__n_s__none | hit@5 | 0.7689 | 0.8292 | +0.0603 | [+0.0536, +0.0669] | 521 / 87 / 6592 |
| photo__s__none | photo__n_s__none | ndcg@10 | 0.6794 | 0.7445 | +0.0651 | [+0.0605, +0.0697] | 1969 / 501 / 4730 |

Bootstrap samples: 5000; seed: 42. No model inference or generated-answer judging was performed.

Pairing follows [SciPy's paired resampling definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html); source groups follow [cluster resampling](https://www.stata.com/support/faqs/statistics/bootstrap-with-panel-data/).

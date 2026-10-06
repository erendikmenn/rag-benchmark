# Paired retrieval comparisons

Dataset: xm3600-tr.

Differences are right minus left, in metric units. Positive values favor the right method. Intervals are 95% paired source-group percentile intervals; methods use the same frozen queries and gallery. These are marginal exploratory comparisons without a multiple-comparison correction.

Grouping: Queries sharing any positively labelled media item are connected; all queries in each connected component are resampled together.

Source groups: 3600; queries: 7233; largest group: 4 queries.

| Left | Right | Metric | Left mean | Right mean | Delta | 95% interval | Wins / losses / ties |
|---|---|---|---:|---:|---:|---|---|
| photo__n__none | photo__s__none | hit@5 | 0.8464 | 0.5988 | -0.2476 | [-0.2602, -0.2352] | 199 / 1990 / 5044 |
| photo__n__none | photo__s__none | ndcg@10 | 0.7639 | 0.5171 | -0.2468 | [-0.2569, -0.2362] | 713 / 3545 / 2975 |
| photo__n__none | photo__n_s__none | hit@5 | 0.8464 | 0.7628 | -0.0836 | [-0.0932, -0.0744] | 245 / 850 / 6138 |
| photo__n__none | photo__n_s__none | ndcg@10 | 0.7639 | 0.6724 | -0.0915 | [-0.0991, -0.0837] | 1006 / 2051 / 4176 |
| photo__s__none | photo__n_s__none | hit@5 | 0.5988 | 0.7628 | +0.1640 | [+0.1545, +0.1735] | 1238 / 52 / 5943 |
| photo__s__none | photo__n_s__none | ndcg@10 | 0.5171 | 0.6724 | +0.1553 | [+0.1494, +0.1612] | 3102 / 268 / 3863 |

Bootstrap samples: 5000; seed: 42. No model inference or generated-answer judging was performed.

Pairing follows [SciPy's paired resampling definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html); source groups follow [cluster resampling](https://www.stata.com/support/faqs/statistics/bootstrap-with-panel-data/).

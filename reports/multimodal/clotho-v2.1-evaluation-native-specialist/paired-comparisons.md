# Paired retrieval comparisons

Dataset: clotho-v2.1-evaluation.

Differences are right minus left, in metric units. Positive values favor the right method. Intervals are 95% paired source-group percentile intervals; methods use the same frozen queries and gallery. These are marginal exploratory comparisons without a multiple-comparison correction.

Grouping: Queries sharing any positively labelled media item are connected; all queries in each connected component are resampled together.

Source groups: 1045; queries: 5225; largest group: 5 queries.

| Left | Right | Metric | Left mean | Right mean | Delta | 95% interval | Wins / losses / ties |
|---|---|---|---:|---:|---:|---|---|
| environment_audio__n__none | environment_audio__s__none | hit@5 | 0.1175 | 0.3742 | +0.2567 | [+0.2325, +0.2796] | 1621 / 280 / 3324 |
| environment_audio__n__none | environment_audio__s__none | ndcg@10 | 0.0969 | 0.3028 | +0.2059 | [+0.1877, +0.2235] | 2303 / 500 / 2422 |
| environment_audio__n__none | environment_audio__n_s__none | hit@5 | 0.1175 | 0.2647 | +0.1472 | [+0.1315, +0.1629] | 858 / 89 / 4278 |
| environment_audio__n__none | environment_audio__n_s__none | ndcg@10 | 0.0969 | 0.2150 | +0.1181 | [+0.1077, +0.1287] | 1656 / 195 / 3374 |
| environment_audio__s__none | environment_audio__n_s__none | hit@5 | 0.3742 | 0.2647 | -0.1095 | [-0.1301, -0.0882] | 453 / 1025 / 3747 |
| environment_audio__s__none | environment_audio__n_s__none | ndcg@10 | 0.3028 | 0.2150 | -0.0878 | [-0.1039, -0.0718] | 1033 / 1708 / 2484 |

Bootstrap samples: 5000; seed: 42. No model inference or generated-answer judging was performed.

Pairing follows [SciPy's paired resampling definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html); source groups follow [cluster resampling](https://www.stata.com/support/faqs/statistics/bootstrap-with-panel-data/).

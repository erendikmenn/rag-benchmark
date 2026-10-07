# Paired retrieval comparisons

Dataset: fleurs-tr_tr-test.

Differences are right minus left, in metric units. Positive values favor the right method. Intervals are 95% paired source-group percentile intervals; methods use the same frozen queries and gallery. These are marginal exploratory comparisons without a multiple-comparison correction.

Grouping: Declared transcript groups are verified against every positive recording's group_id; shared recordings and transcript groups connect queries.

Source groups: 329; queries: 329; largest group: 1 queries.

| Left | Right | Metric | Left mean | Right mean | Delta | 95% interval | Wins / losses / ties |
|---|---|---|---:|---:|---:|---|---|

Bootstrap samples: 5000; seed: 42. No model inference or generated-answer judging was performed.

Pairing follows [SciPy's paired resampling definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html); source groups follow [cluster resampling](https://www.stata.com/support/faqs/statistics/bootstrap-with-panel-data/).

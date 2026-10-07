# Paired retrieval comparisons

Dataset: codesearchnet-php-test.

Differences are right minus left, in metric units. Positive values favor the right method. Where reported, intervals are 95% paired source-group percentile intervals; methods use the same frozen queries and gallery. These are marginal exploratory comparisons without a multiple-comparison correction.

Grouping: Queries sharing a positively labelled source function are connected; corpus IDs identify functions, not entire repositories.

Source groups: 14014; queries: 14014; largest group: 1 queries (0.0%).

Conservative reporting safeguard: withhold intervals when one source group contains more than 50% of queries. This is not a formal statistical threshold.

| Left | Right | Metric | Left mean | Right mean | Delta | 95% interval | Wins / losses / ties |
|---|---|---|---:|---:|---:|---|---|
| code__b__none | code__g__none | hit@5 | 0.3327 | 0.5774 | +0.2447 | [+0.2363, +0.2530] | 3968 / 539 / 9507 |
| code__b__none | code__g__none | ndcg@10 | 0.2797 | 0.5051 | +0.2254 | [+0.2188, +0.2321] | 6321 / 1360 / 6333 |
| code__b__none | code__e__none | hit@5 | 0.3327 | 0.7575 | +0.4249 | [+0.4161, +0.4333] | 6163 / 209 / 7642 |
| code__b__none | code__e__none | ndcg@10 | 0.2797 | 0.6747 | +0.3950 | [+0.3880, +0.4020] | 8777 / 661 / 4576 |
| code__g__none | code__e__none | hit@5 | 0.5774 | 0.7575 | +0.1802 | [+0.1726, +0.1877] | 2983 / 458 / 10573 |
| code__g__none | code__e__none | ndcg@10 | 0.5051 | 0.6747 | +0.1696 | [+0.1637, +0.1756] | 5591 / 1504 / 6919 |
| code__b__none | code__b_g_e__none | hit@5 | 0.3327 | 0.6748 | +0.3421 | [+0.3342, +0.3501] | 4855 / 61 / 9098 |
| code__b__none | code__b_g_e__none | ndcg@10 | 0.2797 | 0.5851 | +0.3054 | [+0.2998, +0.3110] | 7988 / 266 / 5760 |

Bootstrap settings for eligible comparisons: 5000 samples; seed: 42. No model inference or generated-answer judging was performed.

Pairing follows [SciPy's paired resampling definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html); source groups follow [cluster resampling](https://www.stata.com/support/faqs/statistics/bootstrap-with-panel-data/).

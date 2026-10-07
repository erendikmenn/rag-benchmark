# Paired retrieval comparisons

Dataset: codesearchnet-javascript-test.

Differences are right minus left, in metric units. Positive values favor the right method. Where reported, intervals are 95% paired source-group percentile intervals; methods use the same frozen queries and gallery. These are marginal exploratory comparisons without a multiple-comparison correction.

Grouping: Queries sharing a positively labelled source function are connected; corpus IDs identify functions, not entire repositories.

Source groups: 3291; queries: 3291; largest group: 1 queries (0.0%).

Conservative reporting safeguard: withhold intervals when one source group contains more than 50% of queries. This is not a formal statistical threshold.

| Left | Right | Metric | Left mean | Right mean | Delta | 95% interval | Wins / losses / ties |
|---|---|---|---:|---:|---:|---|---|
| code__b__none | code__g__none | hit@5 | 0.3662 | 0.5822 | +0.2160 | [+0.1987, +0.2340] | 868 / 157 / 2266 |
| code__b__none | code__g__none | ndcg@10 | 0.3124 | 0.5150 | +0.2026 | [+0.1886, +0.2171] | 1367 / 339 / 1585 |
| code__b__none | code__e__none | hit@5 | 0.3662 | 0.7991 | +0.4330 | [+0.4157, +0.4506] | 1455 / 30 / 1806 |
| code__b__none | code__e__none | ndcg@10 | 0.3124 | 0.7213 | +0.4090 | [+0.3950, +0.4236] | 2039 / 128 / 1124 |
| code__g__none | code__e__none | hit@5 | 0.5822 | 0.7991 | +0.2170 | [+0.2015, +0.2325] | 766 / 52 / 2473 |
| code__g__none | code__e__none | ndcg@10 | 0.5150 | 0.7213 | +0.2064 | [+0.1943, +0.2187] | 1385 / 232 / 1674 |
| code__b__none | code__b_g_e__none | hit@5 | 0.3662 | 0.6855 | +0.3194 | [+0.3036, +0.3358] | 1064 / 13 / 2214 |
| code__b__none | code__b_g_e__none | ndcg@10 | 0.3124 | 0.6001 | +0.2877 | [+0.2763, +0.2999] | 1768 / 70 / 1453 |

Bootstrap settings for eligible comparisons: 5000 samples; seed: 42. No model inference or generated-answer judging was performed.

Pairing follows [SciPy's paired resampling definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html); source groups follow [cluster resampling](https://www.stata.com/support/faqs/statistics/bootstrap-with-panel-data/).

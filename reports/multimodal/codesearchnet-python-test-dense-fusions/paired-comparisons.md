# Paired retrieval comparisons

Dataset: codesearchnet-python-test.

Differences are right minus left, in metric units. Positive values favor the right method. Where reported, intervals are 95% paired source-group percentile intervals; methods use the same frozen queries and gallery. These are marginal exploratory comparisons without a multiple-comparison correction.

Grouping: Queries sharing a positively labelled source function are connected; corpus IDs identify functions, not entire repositories.

Source groups: 14918; queries: 14918; largest group: 1 queries (0.0%).

Conservative reporting safeguard: withhold intervals when one source group contains more than 50% of queries. This is not a formal statistical threshold.

| Left | Right | Metric | Left mean | Right mean | Delta | 95% interval | Wins / losses / ties |
|---|---|---|---:|---:|---:|---|---|
| code__b__none | code__g__none | hit@5 | 0.3893 | 0.6218 | +0.2325 | [+0.2246, +0.2407] | 4114 / 645 / 10159 |
| code__b__none | code__g__none | ndcg@10 | 0.3318 | 0.5456 | +0.2137 | [+0.2072, +0.2203] | 6578 / 1598 / 6742 |
| code__b__none | code__e__none | hit@5 | 0.3893 | 0.8446 | +0.4553 | [+0.4470, +0.4635] | 6962 / 170 / 7786 |
| code__b__none | code__e__none | ndcg@10 | 0.3318 | 0.7641 | +0.4323 | [+0.4253, +0.4393] | 9702 / 635 / 4581 |
| code__g__none | code__e__none | hit@5 | 0.6218 | 0.8446 | +0.2228 | [+0.2156, +0.2301] | 3586 / 263 / 11069 |
| code__g__none | code__e__none | ndcg@10 | 0.5456 | 0.7641 | +0.2185 | [+0.2126, +0.2245] | 6607 / 1130 / 7181 |
| code__b__none | code__b_g_e__none | hit@5 | 0.3893 | 0.7356 | +0.3464 | [+0.3385, +0.3543] | 5222 / 55 / 9641 |
| code__b__none | code__b_g_e__none | ndcg@10 | 0.3318 | 0.6372 | +0.3053 | [+0.3000, +0.3108] | 8508 / 289 / 6121 |

Bootstrap settings for eligible comparisons: 5000 samples; seed: 42. No model inference or generated-answer judging was performed.

Pairing follows [SciPy's paired resampling definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html); source groups follow [cluster resampling](https://www.stata.com/support/faqs/statistics/bootstrap-with-panel-data/).

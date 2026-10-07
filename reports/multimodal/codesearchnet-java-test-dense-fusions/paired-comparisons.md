# Paired retrieval comparisons

Dataset: codesearchnet-java-test.

Differences are right minus left, in metric units. Positive values favor the right method. Where reported, intervals are 95% paired source-group percentile intervals; methods use the same frozen queries and gallery. These are marginal exploratory comparisons without a multiple-comparison correction.

Grouping: Queries sharing a positively labelled source function are connected; corpus IDs identify functions, not entire repositories.

Source groups: 10955; queries: 10955; largest group: 1 queries (0.0%).

Conservative reporting safeguard: withhold intervals when one source group contains more than 50% of queries. This is not a formal statistical threshold.

| Left | Right | Metric | Left mean | Right mean | Delta | 95% interval | Wins / losses / ties |
|---|---|---|---:|---:|---:|---|---|
| code__b__none | code__g__none | hit@5 | 0.3942 | 0.6373 | +0.2431 | [+0.2330, +0.2529] | 3179 / 516 / 7260 |
| code__b__none | code__g__none | ndcg@10 | 0.3323 | 0.5574 | +0.2252 | [+0.2170, +0.2332] | 4991 / 1228 / 4736 |
| code__b__none | code__e__none | hit@5 | 0.3942 | 0.8380 | +0.4437 | [+0.4340, +0.4536] | 4966 / 105 / 5884 |
| code__b__none | code__e__none | ndcg@10 | 0.3323 | 0.7617 | +0.4294 | [+0.4214, +0.4375] | 7070 / 435 / 3450 |
| code__g__none | code__e__none | hit@5 | 0.6373 | 0.8380 | +0.2006 | [+0.1924, +0.2090] | 2410 / 212 / 8333 |
| code__g__none | code__e__none | ndcg@10 | 0.5574 | 0.7617 | +0.2043 | [+0.1974, +0.2111] | 4581 / 854 / 5520 |
| code__b__none | code__b_g_e__none | hit@5 | 0.3942 | 0.7466 | +0.3524 | [+0.3434, +0.3614] | 3906 / 46 / 7003 |
| code__b__none | code__b_g_e__none | ndcg@10 | 0.3323 | 0.6485 | +0.3162 | [+0.3096, +0.3225] | 6349 / 227 / 4379 |

Bootstrap settings for eligible comparisons: 5000 samples; seed: 42. No model inference or generated-answer judging was performed.

Pairing follows [SciPy's paired resampling definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html); source groups follow [cluster resampling](https://www.stata.com/support/faqs/statistics/bootstrap-with-panel-data/).

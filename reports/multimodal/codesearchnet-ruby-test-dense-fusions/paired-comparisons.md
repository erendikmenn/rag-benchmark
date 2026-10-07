# Paired retrieval comparisons

Dataset: codesearchnet-ruby-test.

Differences are right minus left, in metric units. Positive values favor the right method. Where reported, intervals are 95% paired source-group percentile intervals; methods use the same frozen queries and gallery. These are marginal exploratory comparisons without a multiple-comparison correction.

Grouping: Queries sharing a positively labelled source function are connected; corpus IDs identify functions, not entire repositories.

Source groups: 1261; queries: 1261; largest group: 1 queries (0.1%).

Conservative reporting safeguard: withhold intervals when one source group contains more than 50% of queries. This is not a formal statistical threshold.

| Left | Right | Metric | Left mean | Right mean | Delta | 95% interval | Wins / losses / ties |
|---|---|---|---:|---:|---:|---|---|
| code__b__none | code__g__none | hit@5 | 0.4560 | 0.6820 | +0.2260 | [+0.1990, +0.2530] | 330 / 45 / 886 |
| code__b__none | code__g__none | ndcg@10 | 0.3852 | 0.5995 | +0.2142 | [+0.1926, +0.2357] | 546 / 122 / 593 |
| code__b__none | code__e__none | hit@5 | 0.4560 | 0.8628 | +0.4068 | [+0.3791, +0.4346] | 519 / 6 / 736 |
| code__b__none | code__e__none | ndcg@10 | 0.3852 | 0.7910 | +0.4058 | [+0.3825, +0.4285] | 773 / 42 / 446 |
| code__g__none | code__e__none | hit@5 | 0.6820 | 0.8628 | +0.1808 | [+0.1594, +0.2030] | 237 / 9 / 1015 |
| code__g__none | code__e__none | ndcg@10 | 0.5995 | 0.7910 | +0.1915 | [+0.1734, +0.2093] | 498 / 76 / 687 |
| code__b__none | code__b_g_e__none | hit@5 | 0.4560 | 0.7494 | +0.2934 | [+0.2680, +0.3188] | 375 / 5 / 881 |
| code__b__none | code__b_g_e__none | ndcg@10 | 0.3852 | 0.6628 | +0.2776 | [+0.2598, +0.2951] | 675 / 22 / 564 |

Bootstrap settings for eligible comparisons: 5000 samples; seed: 42. No model inference or generated-answer judging was performed.

Pairing follows [SciPy's paired resampling definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html); source groups follow [cluster resampling](https://www.stata.com/support/faqs/statistics/bootstrap-with-panel-data/).

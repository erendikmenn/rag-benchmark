# Paired retrieval comparisons

Dataset: codesearchnet-go-test.

Differences are right minus left, in metric units. Positive values favor the right method. Where reported, intervals are 95% paired source-group percentile intervals; methods use the same frozen queries and gallery. These are marginal exploratory comparisons without a multiple-comparison correction.

Grouping: Queries sharing a positively labelled source function are connected; corpus IDs identify functions, not entire repositories.

Source groups: 8122; queries: 8122; largest group: 1 queries (0.0%).

Conservative reporting safeguard: withhold intervals when one source group contains more than 50% of queries. This is not a formal statistical threshold.

| Left | Right | Metric | Left mean | Right mean | Delta | 95% interval | Wins / losses / ties |
|---|---|---|---:|---:|---:|---|---|
| code__b__none | code__g__none | hit@5 | 0.6194 | 0.8770 | +0.2576 | [+0.2474, +0.2679] | 2260 / 168 / 5694 |
| code__b__none | code__g__none | ndcg@10 | 0.5495 | 0.8029 | +0.2534 | [+0.2446, +0.2622] | 3757 / 654 / 3711 |
| code__b__none | code__e__none | hit@5 | 0.6194 | 0.9628 | +0.3434 | [+0.3333, +0.3539] | 2816 / 27 / 5279 |
| code__b__none | code__e__none | ndcg@10 | 0.5495 | 0.9232 | +0.3737 | [+0.3646, +0.3826] | 4451 / 183 / 3488 |
| code__g__none | code__e__none | hit@5 | 0.8770 | 0.9628 | +0.0858 | [+0.0793, +0.0922] | 748 / 51 / 7323 |
| code__g__none | code__e__none | ndcg@10 | 0.8029 | 0.9232 | +0.1203 | [+0.1144, +0.1264] | 2122 / 310 / 5690 |
| code__b__none | code__b_g_e__none | hit@5 | 0.6194 | 0.9207 | +0.3013 | [+0.2914, +0.3111] | 2461 / 14 / 5647 |
| code__b__none | code__b_g_e__none | ndcg@10 | 0.5495 | 0.8468 | +0.2973 | [+0.2897, +0.3047] | 4220 / 92 / 3810 |

Bootstrap settings for eligible comparisons: 5000 samples; seed: 42. No model inference or generated-answer judging was performed.

Pairing follows [SciPy's paired resampling definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html); source groups follow [cluster resampling](https://www.stata.com/support/faqs/statistics/bootstrap-with-panel-data/).

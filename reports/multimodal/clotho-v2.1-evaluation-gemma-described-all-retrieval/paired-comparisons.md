# Paired retrieval comparisons

Dataset: clotho-v2.1-evaluation-described.

Differences are right minus left, in metric units. Positive values favor the right method. Where reported, intervals are 95% paired source-group percentile intervals; methods use the same frozen queries and gallery. These are marginal exploratory comparisons without a multiple-comparison correction.

Grouping: Queries sharing any positively labelled media item are connected; all queries in each connected component are resampled together.

Source groups: 1045; queries: 5225; largest group: 5 queries (0.1%).

Conservative reporting safeguard: withhold intervals when one source group contains more than 50% of queries. This is not a formal statistical threshold.

| Left | Right | Metric | Left mean | Right mean | Delta | 95% interval | Wins / losses / ties |
|---|---|---|---:|---:|---:|---|---|
| environment_audio__n__none | environment_audio__s__none | hit@5 | 0.1175 | 0.3742 | +0.2567 | [+0.2325, +0.2796] | 1621 / 280 / 3324 |
| environment_audio__n__none | environment_audio__s__none | ndcg@10 | 0.0969 | 0.3028 | +0.2059 | [+0.1877, +0.2235] | 2303 / 500 / 2422 |
| environment_audio__n__none | environment_audio__n_s__none | hit@5 | 0.1175 | 0.2647 | +0.1472 | [+0.1315, +0.1629] | 858 / 89 / 4278 |
| environment_audio__n__none | environment_audio__n_s__none | ndcg@10 | 0.0969 | 0.2150 | +0.1181 | [+0.1077, +0.1287] | 1656 / 195 / 3374 |
| environment_audio__s__none | environment_audio__n_s__none | hit@5 | 0.3742 | 0.2647 | -0.1095 | [-0.1301, -0.0882] | 453 / 1025 / 3747 |
| environment_audio__s__none | environment_audio__n_s__none | ndcg@10 | 0.3028 | 0.2150 | -0.0878 | [-0.1039, -0.0718] | 1033 / 1708 / 2484 |
| environment_audio__n__none | environment_audio__j__none | hit@5 | 0.1175 | 0.0641 | -0.0534 | [-0.0687, -0.0383] | 178 / 457 / 4590 |
| environment_audio__n__none | environment_audio__j__none | ndcg@10 | 0.0969 | 0.0551 | -0.0418 | [-0.0533, -0.0304] | 364 / 771 / 4090 |
| environment_audio__b__none | environment_audio__g__none | hit@5 | 0.0144 | 0.0222 | +0.0078 | [+0.0013, +0.0144] | 91 / 50 / 5084 |
| environment_audio__b__none | environment_audio__g__none | ndcg@10 | 0.0145 | 0.0215 | +0.0070 | [+0.0022, +0.0118] | 209 / 113 / 4903 |
| environment_audio__b__none | environment_audio__e__none | hit@5 | 0.0144 | 0.0279 | +0.0136 | [+0.0069, +0.0209] | 111 / 40 / 5074 |
| environment_audio__b__none | environment_audio__e__none | ndcg@10 | 0.0145 | 0.0249 | +0.0104 | [+0.0058, +0.0150] | 232 / 100 / 4893 |
| environment_audio__g__none | environment_audio__e__none | hit@5 | 0.0222 | 0.0279 | +0.0057 | [-0.0006, +0.0124] | 84 / 54 / 5087 |
| environment_audio__g__none | environment_audio__e__none | ndcg@10 | 0.0215 | 0.0249 | +0.0034 | [-0.0005, +0.0074] | 171 / 153 / 4901 |
| environment_audio__b__none | environment_audio__b_g_e__none | hit@5 | 0.0144 | 0.0274 | +0.0130 | [+0.0069, +0.0191] | 98 / 30 / 5097 |
| environment_audio__b__none | environment_audio__b_g_e__none | ndcg@10 | 0.0145 | 0.0233 | +0.0088 | [+0.0049, +0.0126] | 206 / 78 / 4941 |

Bootstrap settings for eligible comparisons: 5000 samples; seed: 42. No model inference or generated-answer judging was performed.

Pairing follows [SciPy's paired resampling definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html); source groups follow [cluster resampling](https://www.stata.com/support/faqs/statistics/bootstrap-with-panel-data/).

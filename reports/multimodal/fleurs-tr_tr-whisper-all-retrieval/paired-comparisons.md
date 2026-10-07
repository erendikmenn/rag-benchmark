# Paired retrieval comparisons

Dataset: fleurs-tr_tr-whisper.

Differences are right minus left, in metric units. Positive values favor the right method. Where reported, intervals are 95% paired source-group percentile intervals; methods use the same frozen queries and gallery. These are marginal exploratory comparisons without a multiple-comparison correction.

Grouping: Declared transcript groups are verified against every positive recording's group_id; shared recordings and transcript groups connect queries.

Source groups: 329; queries: 329; largest group: 1 queries (0.3%).

Conservative reporting safeguard: withhold intervals when one source group contains more than 50% of queries. This is not a formal statistical threshold.

| Left | Right | Metric | Left mean | Right mean | Delta | 95% interval | Wins / losses / ties |
|---|---|---|---:|---:|---:|---|---|
| speech__n__none | speech__j__none | hit@5 | 1.0000 | 1.0000 | +0.0000 | [+0.0000, +0.0000] | 0 / 0 / 329 |
| speech__n__none | speech__j__none | ndcg@10 | 0.9970 | 1.0000 | +0.0030 | [+0.0009, +0.0056] | 8 / 0 / 321 |
| speech__b__none | speech__g__none | hit@5 | 1.0000 | 1.0000 | +0.0000 | [+0.0000, +0.0000] | 0 / 0 / 329 |
| speech__b__none | speech__g__none | ndcg@10 | 1.0000 | 1.0000 | +0.0000 | [+0.0000, +0.0000] | 0 / 0 / 329 |
| speech__b__none | speech__e__none | hit@5 | 1.0000 | 1.0000 | +0.0000 | [+0.0000, +0.0000] | 0 / 0 / 329 |
| speech__b__none | speech__e__none | ndcg@10 | 1.0000 | 1.0000 | +0.0000 | [+0.0000, +0.0000] | 0 / 0 / 329 |
| speech__g__none | speech__e__none | hit@5 | 1.0000 | 1.0000 | +0.0000 | [+0.0000, +0.0000] | 0 / 0 / 329 |
| speech__g__none | speech__e__none | ndcg@10 | 1.0000 | 1.0000 | +0.0000 | [+0.0000, +0.0000] | 0 / 0 / 329 |
| speech__b__none | speech__b_g_e__none | hit@5 | 1.0000 | 1.0000 | +0.0000 | [+0.0000, +0.0000] | 0 / 0 / 329 |
| speech__b__none | speech__b_g_e__none | ndcg@10 | 1.0000 | 1.0000 | +0.0000 | [+0.0000, +0.0000] | 0 / 0 / 329 |

Bootstrap settings for eligible comparisons: 5000 samples; seed: 42. No model inference or generated-answer judging was performed.

Pairing follows [SciPy's paired resampling definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html); source groups follow [cluster resampling](https://www.stata.com/support/faqs/statistics/bootstrap-with-panel-data/).

# Paired retrieval comparisons

Dataset: vidore-v3-physics-fr.

Differences are right minus left, in metric units. Positive values favor the right method. Where reported, intervals are 95% paired source-group percentile intervals; methods use the same frozen queries and gallery. These are marginal exploratory comparisons without a multiple-comparison correction.

Grouping: Positive pages map to corpus.metadata.doc_id. Queries touching the same source document are connected transitively across multi-document positives.

Source groups: 19; queries: 302; largest group: 148 queries (49.0%).

Conservative reporting safeguard: withhold intervals when one source group contains more than 50% of queries. This is not a formal statistical threshold.

| Left | Right | Metric | Left mean | Right mean | Delta | 95% interval | Wins / losses / ties |
|---|---|---|---:|---:|---:|---|---|
| document__n__none | document__s__none | hit@5 | 0.7384 | 0.8046 | +0.0662 | [+0.0259, +0.1316] | 38 / 18 / 246 |
| document__n__none | document__s__none | ndcg@10 | 0.4059 | 0.4854 | +0.0795 | [+0.0568, +0.1051] | 154 / 90 / 58 |
| document__n__none | document__n_s__none | hit@5 | 0.7384 | 0.7781 | +0.0397 | [-0.0065, +0.0738] | 21 / 9 / 272 |
| document__n__none | document__n_s__none | ndcg@10 | 0.4059 | 0.4608 | +0.0548 | [+0.0360, +0.0664] | 153 / 64 / 85 |
| document__s__none | document__n_s__none | hit@5 | 0.8046 | 0.7781 | -0.0265 | [-0.0725, -0.0082] | 10 / 18 / 274 |
| document__s__none | document__n_s__none | ndcg@10 | 0.4854 | 0.4608 | -0.0247 | [-0.0490, -0.0118] | 96 / 124 / 82 |
| document__n__none | document__j__none | hit@5 | 0.7384 | 0.7450 | +0.0066 | [-0.0681, +0.0492] | 23 / 21 / 258 |
| document__n__none | document__j__none | ndcg@10 | 0.4059 | 0.3976 | -0.0083 | [-0.0259, +0.0246] | 105 / 119 / 78 |
| document__b__none | document__g__none | hit@5 | 0.7252 | 0.7053 | -0.0199 | [-0.0457, +0.0313] | 31 / 37 / 234 |
| document__b__none | document__g__none | ndcg@10 | 0.3904 | 0.3818 | -0.0087 | [-0.0483, +0.0399] | 119 / 118 / 65 |
| document__b__none | document__e__none | hit@5 | 0.7252 | 0.7616 | +0.0364 | [+0.0000, +0.1111] | 43 / 32 / 227 |
| document__b__none | document__e__none | ndcg@10 | 0.3904 | 0.4168 | +0.0264 | [+0.0031, +0.0796] | 145 / 106 / 51 |
| document__g__none | document__e__none | hit@5 | 0.7053 | 0.7616 | +0.0563 | [+0.0109, +0.1102] | 39 / 22 / 241 |
| document__g__none | document__e__none | ndcg@10 | 0.3818 | 0.4168 | +0.0350 | [+0.0180, +0.0729] | 136 / 93 / 73 |
| document__b__none | document__b_g_e__none | hit@5 | 0.7252 | 0.7815 | +0.0563 | [+0.0298, +0.1280] | 35 / 18 / 249 |
| document__b__none | document__b_g_e__none | ndcg@10 | 0.3904 | 0.4427 | +0.0523 | [+0.0270, +0.0966] | 144 / 90 / 68 |

Bootstrap settings for eligible comparisons: 5000 samples; seed: 42. No model inference or generated-answer judging was performed.

Pairing follows [SciPy's paired resampling definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html); source groups follow [cluster resampling](https://www.stata.com/support/faqs/statistics/bootstrap-with-panel-data/).

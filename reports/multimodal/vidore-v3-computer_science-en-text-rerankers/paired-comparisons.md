# Paired retrieval comparisons

Dataset: vidore-v3-computer_science-en.

Differences are right minus left, in metric units. Positive values favor the right method. Where reported, intervals are 95% paired source-group percentile intervals; methods use the same frozen queries and gallery. These are marginal exploratory comparisons without a multiple-comparison correction.

Grouping: Positive pages map to corpus.metadata.doc_id. Queries touching the same source document are connected transitively across multi-document positives.

Source groups: 1; queries: 215; largest group: 215 queries (100.0%).

Conservative reporting safeguard: withhold intervals when one source group contains more than 50% of queries. This is not a formal statistical threshold.

| Left | Right | Metric | Left mean | Right mean | Delta | 95% interval | Wins / losses / ties |
|---|---|---|---:|---:|---:|---|---|
| document__b__none | document__b__bge_reranker_text | hit@5 | 0.9209 | 0.9628 | +0.0419 | Not reported; see reason below | 12 / 3 / 200 |
| document__b__none | document__b__bge_reranker_text | recall@5 | 0.5353 | 0.6255 | +0.0903 | Not reported; see reason below | 78 / 15 / 122 |
| document__b__none | document__b__bge_reranker_text | ndcg@10 | 0.6334 | 0.7227 | +0.0893 | Not reported; see reason below | 135 / 40 / 40 |
| document__b__laya_text | document__b__bge_reranker_text | hit@5 | 0.5116 | 0.9628 | +0.4512 | Not reported; see reason below | 97 / 0 / 118 |
| document__b__laya_text | document__b__bge_reranker_text | recall@5 | 0.1947 | 0.6255 | +0.4308 | Not reported; see reason below | 176 / 1 / 38 |
| document__b__laya_text | document__b__bge_reranker_text | ndcg@10 | 0.2638 | 0.7227 | +0.4589 | Not reported; see reason below | 194 / 9 / 12 |
| document__g__none | document__g__bge_reranker_text | hit@5 | 0.9535 | 0.9721 | +0.0186 | Not reported; see reason below | 7 / 3 / 205 |
| document__g__none | document__g__bge_reranker_text | recall@5 | 0.5245 | 0.6435 | +0.1190 | Not reported; see reason below | 86 / 19 / 110 |
| document__g__none | document__g__bge_reranker_text | ndcg@10 | 0.6356 | 0.7360 | +0.1004 | Not reported; see reason below | 132 / 47 / 36 |
| document__g__laya_text | document__g__bge_reranker_text | hit@5 | 0.5488 | 0.9721 | +0.4233 | Not reported; see reason below | 91 / 0 / 124 |
| document__g__laya_text | document__g__bge_reranker_text | recall@5 | 0.2024 | 0.6435 | +0.4411 | Not reported; see reason below | 184 / 3 / 28 |
| document__g__laya_text | document__g__bge_reranker_text | ndcg@10 | 0.2829 | 0.7360 | +0.4531 | Not reported; see reason below | 191 / 11 / 13 |
| document__e__none | document__e__bge_reranker_text | hit@5 | 0.9442 | 0.9721 | +0.0279 | Not reported; see reason below | 6 / 0 / 209 |
| document__e__none | document__e__bge_reranker_text | recall@5 | 0.5440 | 0.6415 | +0.0975 | Not reported; see reason below | 81 / 33 / 101 |
| document__e__none | document__e__bge_reranker_text | ndcg@10 | 0.6623 | 0.7369 | +0.0746 | Not reported; see reason below | 122 / 58 / 35 |
| document__e__laya_text | document__e__bge_reranker_text | hit@5 | 0.5628 | 0.9721 | +0.4093 | Not reported; see reason below | 89 / 1 / 125 |
| document__e__laya_text | document__e__bge_reranker_text | recall@5 | 0.2063 | 0.6415 | +0.4352 | Not reported; see reason below | 177 / 3 / 35 |
| document__e__laya_text | document__e__bge_reranker_text | ndcg@10 | 0.2873 | 0.7369 | +0.4496 | Not reported; see reason below | 191 / 12 / 12 |
| document__n__none | document__n__bge_reranker_text | hit@5 | 0.9163 | 0.9721 | +0.0558 | Not reported; see reason below | 14 / 2 / 199 |
| document__n__none | document__n__bge_reranker_text | recall@5 | 0.5270 | 0.6404 | +0.1134 | Not reported; see reason below | 87 / 24 / 104 |
| document__n__none | document__n__bge_reranker_text | ndcg@10 | 0.6296 | 0.7370 | +0.1074 | Not reported; see reason below | 130 / 56 / 29 |
| document__n__laya_text | document__n__bge_reranker_text | hit@5 | 0.5535 | 0.9721 | +0.4186 | Not reported; see reason below | 91 / 1 / 123 |
| document__n__laya_text | document__n__bge_reranker_text | recall@5 | 0.1987 | 0.6404 | +0.4417 | Not reported; see reason below | 182 / 4 / 29 |
| document__n__laya_text | document__n__bge_reranker_text | ndcg@10 | 0.2620 | 0.7370 | +0.4750 | Not reported; see reason below | 195 / 8 / 12 |
| document__s__none | document__s__bge_reranker_text | hit@5 | 0.9721 | 0.9674 | -0.0047 | Not reported; see reason below | 2 / 3 / 210 |
| document__s__none | document__s__bge_reranker_text | recall@5 | 0.6534 | 0.6408 | -0.0127 | Not reported; see reason below | 37 / 47 / 131 |
| document__s__none | document__s__bge_reranker_text | ndcg@10 | 0.7585 | 0.7419 | -0.0165 | Not reported; see reason below | 76 / 98 / 41 |
| document__s__laya_text | document__s__bge_reranker_text | hit@5 | 0.5442 | 0.9674 | +0.4233 | Not reported; see reason below | 92 / 1 / 122 |
| document__s__laya_text | document__s__bge_reranker_text | recall@5 | 0.2043 | 0.6408 | +0.4365 | Not reported; see reason below | 178 / 3 / 34 |
| document__s__laya_text | document__s__bge_reranker_text | ndcg@10 | 0.2688 | 0.7419 | +0.4731 | Not reported; see reason below | 194 / 11 / 10 |
| document__j__none | document__j__bge_reranker_text | hit@5 | 0.9442 | 0.9721 | +0.0279 | Not reported; see reason below | 8 / 2 / 205 |
| document__j__none | document__j__bge_reranker_text | recall@5 | 0.5454 | 0.6437 | +0.0984 | Not reported; see reason below | 80 / 29 / 106 |
| document__j__none | document__j__bge_reranker_text | ndcg@10 | 0.6753 | 0.7402 | +0.0649 | Not reported; see reason below | 115 / 61 / 39 |
| document__j__laya_text | document__j__bge_reranker_text | hit@5 | 0.5535 | 0.9721 | +0.4186 | Not reported; see reason below | 91 / 1 / 123 |
| document__j__laya_text | document__j__bge_reranker_text | recall@5 | 0.2055 | 0.6437 | +0.4382 | Not reported; see reason below | 179 / 5 / 31 |
| document__j__laya_text | document__j__bge_reranker_text | ndcg@10 | 0.2776 | 0.7402 | +0.4626 | Not reported; see reason below | 194 / 11 / 10 |
| document__b_g__none | document__b_g__bge_reranker_text | hit@5 | 0.9488 | 0.9721 | +0.0233 | Not reported; see reason below | 6 / 1 / 208 |
| document__b_g__none | document__b_g__bge_reranker_text | recall@5 | 0.5733 | 0.6359 | +0.0626 | Not reported; see reason below | 56 / 22 / 137 |
| document__b_g__none | document__b_g__bge_reranker_text | ndcg@10 | 0.6771 | 0.7352 | +0.0581 | Not reported; see reason below | 124 / 52 / 39 |
| document__b_g__laya_text | document__b_g__bge_reranker_text | hit@5 | 0.5535 | 0.9721 | +0.4186 | Not reported; see reason below | 90 / 0 / 125 |
| document__b_g__laya_text | document__b_g__bge_reranker_text | recall@5 | 0.2079 | 0.6359 | +0.4280 | Not reported; see reason below | 176 / 3 / 36 |
| document__b_g__laya_text | document__b_g__bge_reranker_text | ndcg@10 | 0.2744 | 0.7352 | +0.4609 | Not reported; see reason below | 195 / 8 / 12 |
| document__b_e__none | document__b_e__bge_reranker_text | hit@5 | 0.9535 | 0.9674 | +0.0140 | Not reported; see reason below | 5 / 2 / 208 |
| document__b_e__none | document__b_e__bge_reranker_text | recall@5 | 0.5788 | 0.6349 | +0.0561 | Not reported; see reason below | 60 / 31 / 124 |
| document__b_e__none | document__b_e__bge_reranker_text | ndcg@10 | 0.6956 | 0.7349 | +0.0392 | Not reported; see reason below | 107 / 64 / 44 |
| document__b_e__laya_text | document__b_e__bge_reranker_text | hit@5 | 0.5628 | 0.9674 | +0.4047 | Not reported; see reason below | 88 / 1 / 126 |
| document__b_e__laya_text | document__b_e__bge_reranker_text | recall@5 | 0.2101 | 0.6349 | +0.4248 | Not reported; see reason below | 179 / 3 / 33 |
| document__b_e__laya_text | document__b_e__bge_reranker_text | ndcg@10 | 0.2824 | 0.7349 | +0.4524 | Not reported; see reason below | 194 / 10 / 11 |
| document__b_n__none | document__b_n__bge_reranker_text | hit@5 | 0.9442 | 0.9721 | +0.0279 | Not reported; see reason below | 7 / 1 / 207 |
| document__b_n__none | document__b_n__bge_reranker_text | recall@5 | 0.5781 | 0.6428 | +0.0647 | Not reported; see reason below | 68 / 26 / 121 |
| document__b_n__none | document__b_n__bge_reranker_text | ndcg@10 | 0.6843 | 0.7388 | +0.0545 | Not reported; see reason below | 114 / 54 / 47 |
| document__b_n__laya_text | document__b_n__bge_reranker_text | hit@5 | 0.5628 | 0.9721 | +0.4093 | Not reported; see reason below | 89 / 1 / 125 |
| document__b_n__laya_text | document__b_n__bge_reranker_text | recall@5 | 0.2114 | 0.6428 | +0.4315 | Not reported; see reason below | 180 / 3 / 32 |
| document__b_n__laya_text | document__b_n__bge_reranker_text | ndcg@10 | 0.2773 | 0.7388 | +0.4615 | Not reported; see reason below | 193 / 10 / 12 |
| document__b_s__none | document__b_s__bge_reranker_text | hit@5 | 0.9628 | 0.9674 | +0.0047 | Not reported; see reason below | 3 / 2 / 210 |
| document__b_s__none | document__b_s__bge_reranker_text | recall@5 | 0.6171 | 0.6380 | +0.0209 | Not reported; see reason below | 47 / 35 / 133 |
| document__b_s__none | document__b_s__bge_reranker_text | ndcg@10 | 0.7288 | 0.7377 | +0.0089 | Not reported; see reason below | 91 / 82 / 42 |
| document__b_s__laya_text | document__b_s__bge_reranker_text | hit@5 | 0.5674 | 0.9674 | +0.4000 | Not reported; see reason below | 87 / 1 / 127 |
| document__b_s__laya_text | document__b_s__bge_reranker_text | recall@5 | 0.2071 | 0.6380 | +0.4309 | Not reported; see reason below | 178 / 3 / 34 |
| document__b_s__laya_text | document__b_s__bge_reranker_text | ndcg@10 | 0.2788 | 0.7377 | +0.4589 | Not reported; see reason below | 192 / 9 / 14 |
| document__b_j__none | document__b_j__bge_reranker_text | hit@5 | 0.9581 | 0.9721 | +0.0140 | Not reported; see reason below | 4 / 1 / 210 |
| document__b_j__none | document__b_j__bge_reranker_text | recall@5 | 0.5832 | 0.6399 | +0.0567 | Not reported; see reason below | 59 / 28 / 128 |
| document__b_j__none | document__b_j__bge_reranker_text | ndcg@10 | 0.6961 | 0.7355 | +0.0394 | Not reported; see reason below | 109 / 58 / 48 |
| document__b_j__laya_text | document__b_j__bge_reranker_text | hit@5 | 0.5488 | 0.9721 | +0.4233 | Not reported; see reason below | 92 / 1 / 122 |
| document__b_j__laya_text | document__b_j__bge_reranker_text | recall@5 | 0.1975 | 0.6399 | +0.4424 | Not reported; see reason below | 182 / 3 / 30 |
| document__b_j__laya_text | document__b_j__bge_reranker_text | ndcg@10 | 0.2805 | 0.7355 | +0.4550 | Not reported; see reason below | 195 / 7 / 13 |
| document__g_e__none | document__g_e__bge_reranker_text | hit@5 | 0.9535 | 0.9721 | +0.0186 | Not reported; see reason below | 7 / 3 / 205 |
| document__g_e__none | document__g_e__bge_reranker_text | recall@5 | 0.5587 | 0.6368 | +0.0781 | Not reported; see reason below | 64 / 33 / 118 |
| document__g_e__none | document__g_e__bge_reranker_text | ndcg@10 | 0.6866 | 0.7358 | +0.0492 | Not reported; see reason below | 111 / 67 / 37 |
| document__g_e__laya_text | document__g_e__bge_reranker_text | hit@5 | 0.5767 | 0.9721 | +0.3953 | Not reported; see reason below | 86 / 1 / 128 |
| document__g_e__laya_text | document__g_e__bge_reranker_text | recall@5 | 0.2108 | 0.6368 | +0.4260 | Not reported; see reason below | 176 / 4 / 35 |
| document__g_e__laya_text | document__g_e__bge_reranker_text | ndcg@10 | 0.2883 | 0.7358 | +0.4475 | Not reported; see reason below | 193 / 12 / 10 |
| document__g_n__none | document__g_n__bge_reranker_text | hit@5 | 0.9581 | 0.9674 | +0.0093 | Not reported; see reason below | 6 / 4 / 205 |
| document__g_n__none | document__g_n__bge_reranker_text | recall@5 | 0.5611 | 0.6322 | +0.0711 | Not reported; see reason below | 68 / 27 / 120 |
| document__g_n__none | document__g_n__bge_reranker_text | ndcg@10 | 0.6764 | 0.7389 | +0.0625 | Not reported; see reason below | 123 / 52 / 40 |
| document__g_n__laya_text | document__g_n__bge_reranker_text | hit@5 | 0.5628 | 0.9674 | +0.4047 | Not reported; see reason below | 88 / 1 / 126 |
| document__g_n__laya_text | document__g_n__bge_reranker_text | recall@5 | 0.2113 | 0.6322 | +0.4209 | Not reported; see reason below | 176 / 6 / 33 |
| document__g_n__laya_text | document__g_n__bge_reranker_text | ndcg@10 | 0.2748 | 0.7389 | +0.4641 | Not reported; see reason below | 194 / 9 / 12 |
| document__g_s__none | document__g_s__bge_reranker_text | hit@5 | 0.9581 | 0.9674 | +0.0093 | Not reported; see reason below | 4 / 2 / 209 |
| document__g_s__none | document__g_s__bge_reranker_text | recall@5 | 0.6220 | 0.6349 | +0.0129 | Not reported; see reason below | 40 / 37 / 138 |
| document__g_s__none | document__g_s__bge_reranker_text | ndcg@10 | 0.7233 | 0.7391 | +0.0159 | Not reported; see reason below | 92 / 83 / 40 |
| document__g_s__laya_text | document__g_s__bge_reranker_text | hit@5 | 0.5535 | 0.9674 | +0.4140 | Not reported; see reason below | 90 / 1 / 124 |
| document__g_s__laya_text | document__g_s__bge_reranker_text | recall@5 | 0.2027 | 0.6349 | +0.4322 | Not reported; see reason below | 180 / 4 / 31 |
| document__g_s__laya_text | document__g_s__bge_reranker_text | ndcg@10 | 0.2689 | 0.7391 | +0.4703 | Not reported; see reason below | 196 / 8 / 11 |
| document__g_j__none | document__g_j__bge_reranker_text | hit@5 | 0.9535 | 0.9721 | +0.0186 | Not reported; see reason below | 7 / 3 / 205 |
| document__g_j__none | document__g_j__bge_reranker_text | recall@5 | 0.5639 | 0.6394 | +0.0756 | Not reported; see reason below | 70 / 30 / 115 |
| document__g_j__none | document__g_j__bge_reranker_text | ndcg@10 | 0.6899 | 0.7380 | +0.0481 | Not reported; see reason below | 117 / 61 / 37 |
| document__g_j__laya_text | document__g_j__bge_reranker_text | hit@5 | 0.5628 | 0.9721 | +0.4093 | Not reported; see reason below | 89 / 1 / 125 |
| document__g_j__laya_text | document__g_j__bge_reranker_text | recall@5 | 0.2035 | 0.6394 | +0.4360 | Not reported; see reason below | 178 / 5 / 32 |
| document__g_j__laya_text | document__g_j__bge_reranker_text | ndcg@10 | 0.2832 | 0.7380 | +0.4548 | Not reported; see reason below | 192 / 11 / 12 |
| document__e_n__none | document__e_n__bge_reranker_text | hit@5 | 0.9628 | 0.9674 | +0.0047 | Not reported; see reason below | 4 / 3 / 208 |
| document__e_n__none | document__e_n__bge_reranker_text | recall@5 | 0.5600 | 0.6385 | +0.0786 | Not reported; see reason below | 74 / 29 / 112 |
| document__e_n__none | document__e_n__bge_reranker_text | ndcg@10 | 0.6666 | 0.7390 | +0.0724 | Not reported; see reason below | 119 / 59 / 37 |
| document__e_n__laya_text | document__e_n__bge_reranker_text | hit@5 | 0.5535 | 0.9674 | +0.4140 | Not reported; see reason below | 90 / 1 / 124 |
| document__e_n__laya_text | document__e_n__bge_reranker_text | recall@5 | 0.2073 | 0.6385 | +0.4312 | Not reported; see reason below | 178 / 3 / 34 |
| document__e_n__laya_text | document__e_n__bge_reranker_text | ndcg@10 | 0.2800 | 0.7390 | +0.4590 | Not reported; see reason below | 195 / 8 / 12 |
| document__e_s__none | document__e_s__bge_reranker_text | hit@5 | 0.9674 | 0.9674 | +0.0000 | Not reported; see reason below | 2 / 2 / 211 |
| document__e_s__none | document__e_s__bge_reranker_text | recall@5 | 0.6112 | 0.6400 | +0.0288 | Not reported; see reason below | 48 / 39 / 128 |
| document__e_s__none | document__e_s__bge_reranker_text | ndcg@10 | 0.7399 | 0.7419 | +0.0020 | Not reported; see reason below | 98 / 78 / 39 |
| document__e_s__laya_text | document__e_s__bge_reranker_text | hit@5 | 0.5442 | 0.9674 | +0.4233 | Not reported; see reason below | 92 / 1 / 122 |
| document__e_s__laya_text | document__e_s__bge_reranker_text | recall@5 | 0.2031 | 0.6400 | +0.4369 | Not reported; see reason below | 180 / 2 / 33 |
| document__e_s__laya_text | document__e_s__bge_reranker_text | ndcg@10 | 0.2780 | 0.7419 | +0.4639 | Not reported; see reason below | 193 / 10 / 12 |
| document__e_j__none | document__e_j__bge_reranker_text | hit@5 | 0.9628 | 0.9721 | +0.0093 | Not reported; see reason below | 4 / 2 / 209 |
| document__e_j__none | document__e_j__bge_reranker_text | recall@5 | 0.5497 | 0.6431 | +0.0934 | Not reported; see reason below | 79 / 30 / 106 |
| document__e_j__none | document__e_j__bge_reranker_text | ndcg@10 | 0.6722 | 0.7388 | +0.0666 | Not reported; see reason below | 119 / 58 / 38 |
| document__e_j__laya_text | document__e_j__bge_reranker_text | hit@5 | 0.5349 | 0.9721 | +0.4372 | Not reported; see reason below | 95 / 1 / 119 |
| document__e_j__laya_text | document__e_j__bge_reranker_text | recall@5 | 0.1987 | 0.6431 | +0.4444 | Not reported; see reason below | 179 / 3 / 33 |
| document__e_j__laya_text | document__e_j__bge_reranker_text | ndcg@10 | 0.2832 | 0.7388 | +0.4556 | Not reported; see reason below | 192 / 12 / 11 |
| document__n_s__none | document__n_s__bge_reranker_text | hit@5 | 0.9674 | 0.9721 | +0.0047 | Not reported; see reason below | 2 / 1 / 212 |
| document__n_s__none | document__n_s__bge_reranker_text | recall@5 | 0.6163 | 0.6435 | +0.0272 | Not reported; see reason below | 52 / 38 / 125 |
| document__n_s__none | document__n_s__bge_reranker_text | ndcg@10 | 0.7264 | 0.7425 | +0.0160 | Not reported; see reason below | 97 / 82 / 36 |
| document__n_s__laya_text | document__n_s__bge_reranker_text | hit@5 | 0.5581 | 0.9721 | +0.4140 | Not reported; see reason below | 90 / 1 / 124 |
| document__n_s__laya_text | document__n_s__bge_reranker_text | recall@5 | 0.2070 | 0.6435 | +0.4365 | Not reported; see reason below | 179 / 3 / 33 |
| document__n_s__laya_text | document__n_s__bge_reranker_text | ndcg@10 | 0.2718 | 0.7425 | +0.4706 | Not reported; see reason below | 193 / 10 / 12 |
| document__n_j__none | document__n_j__bge_reranker_text | hit@5 | 0.9302 | 0.9721 | +0.0419 | Not reported; see reason below | 11 / 2 / 202 |
| document__n_j__none | document__n_j__bge_reranker_text | recall@5 | 0.5429 | 0.6428 | +0.0998 | Not reported; see reason below | 80 / 30 / 105 |
| document__n_j__none | document__n_j__bge_reranker_text | ndcg@10 | 0.6629 | 0.7393 | +0.0765 | Not reported; see reason below | 121 / 60 / 34 |
| document__n_j__laya_text | document__n_j__bge_reranker_text | hit@5 | 0.5535 | 0.9721 | +0.4186 | Not reported; see reason below | 91 / 1 / 123 |
| document__n_j__laya_text | document__n_j__bge_reranker_text | recall@5 | 0.2063 | 0.6428 | +0.4365 | Not reported; see reason below | 179 / 4 / 32 |
| document__n_j__laya_text | document__n_j__bge_reranker_text | ndcg@10 | 0.2774 | 0.7393 | +0.4619 | Not reported; see reason below | 194 / 8 / 13 |
| document__s_j__none | document__s_j__bge_reranker_text | hit@5 | 0.9721 | 0.9721 | +0.0000 | Not reported; see reason below | 1 / 1 / 213 |
| document__s_j__none | document__s_j__bge_reranker_text | recall@5 | 0.6215 | 0.6412 | +0.0197 | Not reported; see reason below | 50 / 40 / 125 |
| document__s_j__none | document__s_j__bge_reranker_text | ndcg@10 | 0.7409 | 0.7400 | -0.0008 | Not reported; see reason below | 94 / 77 / 44 |
| document__s_j__laya_text | document__s_j__bge_reranker_text | hit@5 | 0.5256 | 0.9721 | +0.4465 | Not reported; see reason below | 97 / 1 / 117 |
| document__s_j__laya_text | document__s_j__bge_reranker_text | recall@5 | 0.1927 | 0.6412 | +0.4484 | Not reported; see reason below | 182 / 2 / 31 |
| document__s_j__laya_text | document__s_j__bge_reranker_text | ndcg@10 | 0.2657 | 0.7400 | +0.4743 | Not reported; see reason below | 195 / 8 / 12 |
| document__b_g_e__none | document__b_g_e__bge_reranker_text | hit@5 | 0.9674 | 0.9674 | +0.0000 | Not reported; see reason below | 2 / 2 / 211 |
| document__b_g_e__none | document__b_g_e__bge_reranker_text | recall@5 | 0.5941 | 0.6370 | +0.0429 | Not reported; see reason below | 54 / 29 / 132 |
| document__b_g_e__none | document__b_g_e__bge_reranker_text | ndcg@10 | 0.7005 | 0.7380 | +0.0375 | Not reported; see reason below | 110 / 63 / 42 |
| document__b_g_e__laya_text | document__b_g_e__bge_reranker_text | hit@5 | 0.5581 | 0.9674 | +0.4093 | Not reported; see reason below | 88 / 0 / 127 |
| document__b_g_e__laya_text | document__b_g_e__bge_reranker_text | recall@5 | 0.1996 | 0.6370 | +0.4373 | Not reported; see reason below | 181 / 3 / 31 |
| document__b_g_e__laya_text | document__b_g_e__bge_reranker_text | ndcg@10 | 0.2722 | 0.7380 | +0.4658 | Not reported; see reason below | 195 / 8 / 12 |
| document__b_g_n__none | document__b_g_n__bge_reranker_text | hit@5 | 0.9767 | 0.9721 | -0.0047 | Not reported; see reason below | 1 / 2 / 212 |
| document__b_g_n__none | document__b_g_n__bge_reranker_text | recall@5 | 0.5970 | 0.6414 | +0.0444 | Not reported; see reason below | 58 / 30 / 127 |
| document__b_g_n__none | document__b_g_n__bge_reranker_text | ndcg@10 | 0.6988 | 0.7372 | +0.0384 | Not reported; see reason below | 105 / 62 / 48 |
| document__b_g_n__laya_text | document__b_g_n__bge_reranker_text | hit@5 | 0.5395 | 0.9721 | +0.4326 | Not reported; see reason below | 93 / 0 / 122 |
| document__b_g_n__laya_text | document__b_g_n__bge_reranker_text | recall@5 | 0.1926 | 0.6414 | +0.4488 | Not reported; see reason below | 182 / 3 / 30 |
| document__b_g_n__laya_text | document__b_g_n__bge_reranker_text | ndcg@10 | 0.2640 | 0.7372 | +0.4732 | Not reported; see reason below | 194 / 8 / 13 |
| document__b_g_s__none | document__b_g_s__bge_reranker_text | hit@5 | 0.9767 | 0.9674 | -0.0093 | Not reported; see reason below | 0 / 2 / 213 |
| document__b_g_s__none | document__b_g_s__bge_reranker_text | recall@5 | 0.6279 | 0.6373 | +0.0094 | Not reported; see reason below | 40 / 35 / 140 |
| document__b_g_s__none | document__b_g_s__bge_reranker_text | ndcg@10 | 0.7349 | 0.7362 | +0.0013 | Not reported; see reason below | 93 / 71 / 51 |
| document__b_g_s__laya_text | document__b_g_s__bge_reranker_text | hit@5 | 0.5674 | 0.9674 | +0.4000 | Not reported; see reason below | 86 / 0 / 129 |
| document__b_g_s__laya_text | document__b_g_s__bge_reranker_text | recall@5 | 0.2070 | 0.6373 | +0.4303 | Not reported; see reason below | 177 / 5 / 33 |
| document__b_g_s__laya_text | document__b_g_s__bge_reranker_text | ndcg@10 | 0.2764 | 0.7362 | +0.4598 | Not reported; see reason below | 193 / 9 / 13 |
| document__b_g_j__none | document__b_g_j__bge_reranker_text | hit@5 | 0.9674 | 0.9674 | +0.0000 | Not reported; see reason below | 2 / 2 / 211 |
| document__b_g_j__none | document__b_g_j__bge_reranker_text | recall@5 | 0.5981 | 0.6375 | +0.0394 | Not reported; see reason below | 57 / 29 / 129 |
| document__b_g_j__none | document__b_g_j__bge_reranker_text | ndcg@10 | 0.7057 | 0.7367 | +0.0310 | Not reported; see reason below | 105 / 64 / 46 |
| document__b_g_j__laya_text | document__b_g_j__bge_reranker_text | hit@5 | 0.5442 | 0.9674 | +0.4233 | Not reported; see reason below | 91 / 0 / 124 |
| document__b_g_j__laya_text | document__b_g_j__bge_reranker_text | recall@5 | 0.1952 | 0.6375 | +0.4423 | Not reported; see reason below | 182 / 3 / 30 |
| document__b_g_j__laya_text | document__b_g_j__bge_reranker_text | ndcg@10 | 0.2681 | 0.7367 | +0.4686 | Not reported; see reason below | 194 / 7 / 14 |
| document__b_e_n__none | document__b_e_n__bge_reranker_text | hit@5 | 0.9628 | 0.9674 | +0.0047 | Not reported; see reason below | 3 / 2 / 210 |
| document__b_e_n__none | document__b_e_n__bge_reranker_text | recall@5 | 0.5884 | 0.6378 | +0.0494 | Not reported; see reason below | 63 / 31 / 121 |
| document__b_e_n__none | document__b_e_n__bge_reranker_text | ndcg@10 | 0.6995 | 0.7397 | +0.0402 | Not reported; see reason below | 109 / 58 / 48 |
| document__b_e_n__laya_text | document__b_e_n__bge_reranker_text | hit@5 | 0.5628 | 0.9674 | +0.4047 | Not reported; see reason below | 88 / 1 / 126 |
| document__b_e_n__laya_text | document__b_e_n__bge_reranker_text | recall@5 | 0.2023 | 0.6378 | +0.4356 | Not reported; see reason below | 182 / 3 / 30 |
| document__b_e_n__laya_text | document__b_e_n__bge_reranker_text | ndcg@10 | 0.2770 | 0.7397 | +0.4628 | Not reported; see reason below | 194 / 9 / 12 |
| document__b_e_s__none | document__b_e_s__bge_reranker_text | hit@5 | 0.9721 | 0.9674 | -0.0047 | Not reported; see reason below | 1 / 2 / 212 |
| document__b_e_s__none | document__b_e_s__bge_reranker_text | recall@5 | 0.6152 | 0.6391 | +0.0239 | Not reported; see reason below | 51 / 37 / 127 |
| document__b_e_s__none | document__b_e_s__bge_reranker_text | ndcg@10 | 0.7312 | 0.7388 | +0.0077 | Not reported; see reason below | 95 / 77 / 43 |
| document__b_e_s__laya_text | document__b_e_s__bge_reranker_text | hit@5 | 0.5535 | 0.9674 | +0.4140 | Not reported; see reason below | 90 / 1 / 124 |
| document__b_e_s__laya_text | document__b_e_s__bge_reranker_text | recall@5 | 0.2033 | 0.6391 | +0.4358 | Not reported; see reason below | 179 / 3 / 33 |
| document__b_e_s__laya_text | document__b_e_s__bge_reranker_text | ndcg@10 | 0.2763 | 0.7388 | +0.4625 | Not reported; see reason below | 192 / 10 / 13 |
| document__b_e_j__none | document__b_e_j__bge_reranker_text | hit@5 | 0.9628 | 0.9721 | +0.0093 | Not reported; see reason below | 3 / 1 / 211 |
| document__b_e_j__none | document__b_e_j__bge_reranker_text | recall@5 | 0.5894 | 0.6401 | +0.0507 | Not reported; see reason below | 61 / 35 / 119 |
| document__b_e_j__none | document__b_e_j__bge_reranker_text | ndcg@10 | 0.7042 | 0.7376 | +0.0334 | Not reported; see reason below | 109 / 63 / 43 |
| document__b_e_j__laya_text | document__b_e_j__bge_reranker_text | hit@5 | 0.5628 | 0.9721 | +0.4093 | Not reported; see reason below | 89 / 1 / 125 |
| document__b_e_j__laya_text | document__b_e_j__bge_reranker_text | recall@5 | 0.2035 | 0.6401 | +0.4366 | Not reported; see reason below | 181 / 4 / 30 |
| document__b_e_j__laya_text | document__b_e_j__bge_reranker_text | ndcg@10 | 0.2792 | 0.7376 | +0.4584 | Not reported; see reason below | 195 / 9 / 11 |
| document__b_n_s__none | document__b_n_s__bge_reranker_text | hit@5 | 0.9814 | 0.9674 | -0.0140 | Not reported; see reason below | 0 / 3 / 212 |
| document__b_n_s__none | document__b_n_s__bge_reranker_text | recall@5 | 0.6258 | 0.6390 | +0.0132 | Not reported; see reason below | 48 / 37 / 130 |
| document__b_n_s__none | document__b_n_s__bge_reranker_text | ndcg@10 | 0.7301 | 0.7411 | +0.0110 | Not reported; see reason below | 97 / 77 / 41 |
| document__b_n_s__laya_text | document__b_n_s__bge_reranker_text | hit@5 | 0.5674 | 0.9674 | +0.4000 | Not reported; see reason below | 87 / 1 / 127 |
| document__b_n_s__laya_text | document__b_n_s__bge_reranker_text | recall@5 | 0.2054 | 0.6390 | +0.4336 | Not reported; see reason below | 180 / 3 / 32 |
| document__b_n_s__laya_text | document__b_n_s__bge_reranker_text | ndcg@10 | 0.2692 | 0.7411 | +0.4719 | Not reported; see reason below | 193 / 10 / 12 |
| document__b_n_j__none | document__b_n_j__bge_reranker_text | hit@5 | 0.9581 | 0.9721 | +0.0140 | Not reported; see reason below | 4 / 1 / 210 |
| document__b_n_j__none | document__b_n_j__bge_reranker_text | recall@5 | 0.5888 | 0.6425 | +0.0537 | Not reported; see reason below | 63 / 34 / 118 |
| document__b_n_j__none | document__b_n_j__bge_reranker_text | ndcg@10 | 0.6999 | 0.7380 | +0.0380 | Not reported; see reason below | 106 / 65 / 44 |
| document__b_n_j__laya_text | document__b_n_j__bge_reranker_text | hit@5 | 0.5442 | 0.9721 | +0.4279 | Not reported; see reason below | 93 / 1 / 121 |
| document__b_n_j__laya_text | document__b_n_j__bge_reranker_text | recall@5 | 0.1982 | 0.6425 | +0.4443 | Not reported; see reason below | 180 / 4 / 31 |
| document__b_n_j__laya_text | document__b_n_j__bge_reranker_text | ndcg@10 | 0.2706 | 0.7380 | +0.4674 | Not reported; see reason below | 194 / 9 / 12 |
| document__b_s_j__none | document__b_s_j__bge_reranker_text | hit@5 | 0.9721 | 0.9674 | -0.0047 | Not reported; see reason below | 1 / 2 / 212 |
| document__b_s_j__none | document__b_s_j__bge_reranker_text | recall@5 | 0.6244 | 0.6391 | +0.0147 | Not reported; see reason below | 47 / 40 / 128 |
| document__b_s_j__none | document__b_s_j__bge_reranker_text | ndcg@10 | 0.7440 | 0.7393 | -0.0047 | Not reported; see reason below | 89 / 81 / 45 |
| document__b_s_j__laya_text | document__b_s_j__bge_reranker_text | hit@5 | 0.5535 | 0.9674 | +0.4140 | Not reported; see reason below | 90 / 1 / 124 |
| document__b_s_j__laya_text | document__b_s_j__bge_reranker_text | recall@5 | 0.2004 | 0.6391 | +0.4387 | Not reported; see reason below | 179 / 3 / 33 |
| document__b_s_j__laya_text | document__b_s_j__bge_reranker_text | ndcg@10 | 0.2748 | 0.7393 | +0.4645 | Not reported; see reason below | 194 / 9 / 12 |
| document__g_e_n__none | document__g_e_n__bge_reranker_text | hit@5 | 0.9628 | 0.9674 | +0.0047 | Not reported; see reason below | 5 / 4 / 206 |
| document__g_e_n__none | document__g_e_n__bge_reranker_text | recall@5 | 0.5792 | 0.6371 | +0.0579 | Not reported; see reason below | 68 / 35 / 112 |
| document__g_e_n__none | document__g_e_n__bge_reranker_text | ndcg@10 | 0.6987 | 0.7395 | +0.0409 | Not reported; see reason below | 115 / 64 / 36 |
| document__g_e_n__laya_text | document__g_e_n__bge_reranker_text | hit@5 | 0.5674 | 0.9674 | +0.4000 | Not reported; see reason below | 87 / 1 / 127 |
| document__g_e_n__laya_text | document__g_e_n__bge_reranker_text | recall@5 | 0.2074 | 0.6371 | +0.4297 | Not reported; see reason below | 178 / 5 / 32 |
| document__g_e_n__laya_text | document__g_e_n__bge_reranker_text | ndcg@10 | 0.2833 | 0.7395 | +0.4563 | Not reported; see reason below | 193 / 12 / 10 |
| document__g_e_s__none | document__g_e_s__bge_reranker_text | hit@5 | 0.9581 | 0.9674 | +0.0093 | Not reported; see reason below | 4 / 2 / 209 |
| document__g_e_s__none | document__g_e_s__bge_reranker_text | recall@5 | 0.6049 | 0.6378 | +0.0329 | Not reported; see reason below | 49 / 36 / 130 |
| document__g_e_s__none | document__g_e_s__bge_reranker_text | ndcg@10 | 0.7276 | 0.7412 | +0.0136 | Not reported; see reason below | 99 / 79 / 37 |
| document__g_e_s__laya_text | document__g_e_s__bge_reranker_text | hit@5 | 0.5628 | 0.9674 | +0.4047 | Not reported; see reason below | 88 / 1 / 126 |
| document__g_e_s__laya_text | document__g_e_s__bge_reranker_text | recall@5 | 0.2073 | 0.6378 | +0.4304 | Not reported; see reason below | 179 / 3 / 33 |
| document__g_e_s__laya_text | document__g_e_s__bge_reranker_text | ndcg@10 | 0.2807 | 0.7412 | +0.4605 | Not reported; see reason below | 194 / 10 / 11 |
| document__g_e_j__none | document__g_e_j__bge_reranker_text | hit@5 | 0.9628 | 0.9721 | +0.0093 | Not reported; see reason below | 5 / 3 / 207 |
| document__g_e_j__none | document__g_e_j__bge_reranker_text | recall@5 | 0.5771 | 0.6404 | +0.0632 | Not reported; see reason below | 63 / 34 / 118 |
| document__g_e_j__none | document__g_e_j__bge_reranker_text | ndcg@10 | 0.6970 | 0.7385 | +0.0415 | Not reported; see reason below | 105 / 69 / 41 |
| document__g_e_j__laya_text | document__g_e_j__bge_reranker_text | hit@5 | 0.5767 | 0.9721 | +0.3953 | Not reported; see reason below | 86 / 1 / 128 |
| document__g_e_j__laya_text | document__g_e_j__bge_reranker_text | recall@5 | 0.2096 | 0.6404 | +0.4308 | Not reported; see reason below | 176 / 5 / 34 |
| document__g_e_j__laya_text | document__g_e_j__bge_reranker_text | ndcg@10 | 0.2820 | 0.7385 | +0.4565 | Not reported; see reason below | 193 / 11 / 11 |
| document__g_n_s__none | document__g_n_s__bge_reranker_text | hit@5 | 0.9721 | 0.9721 | +0.0000 | Not reported; see reason below | 2 / 2 / 211 |
| document__g_n_s__none | document__g_n_s__bge_reranker_text | recall@5 | 0.6167 | 0.6428 | +0.0262 | Not reported; see reason below | 49 / 34 / 132 |
| document__g_n_s__none | document__g_n_s__bge_reranker_text | ndcg@10 | 0.7327 | 0.7398 | +0.0071 | Not reported; see reason below | 93 / 79 / 43 |
| document__g_n_s__laya_text | document__g_n_s__bge_reranker_text | hit@5 | 0.5535 | 0.9721 | +0.4186 | Not reported; see reason below | 91 / 1 / 123 |
| document__g_n_s__laya_text | document__g_n_s__bge_reranker_text | recall@5 | 0.2037 | 0.6428 | +0.4391 | Not reported; see reason below | 181 / 3 / 31 |
| document__g_n_s__laya_text | document__g_n_s__bge_reranker_text | ndcg@10 | 0.2693 | 0.7398 | +0.4705 | Not reported; see reason below | 195 / 9 / 11 |
| document__g_n_j__none | document__g_n_j__bge_reranker_text | hit@5 | 0.9535 | 0.9721 | +0.0186 | Not reported; see reason below | 6 / 2 / 207 |
| document__g_n_j__none | document__g_n_j__bge_reranker_text | recall@5 | 0.5746 | 0.6413 | +0.0667 | Not reported; see reason below | 68 / 31 / 116 |
| document__g_n_j__none | document__g_n_j__bge_reranker_text | ndcg@10 | 0.6886 | 0.7402 | +0.0516 | Not reported; see reason below | 112 / 61 / 42 |
| document__g_n_j__laya_text | document__g_n_j__bge_reranker_text | hit@5 | 0.5535 | 0.9721 | +0.4186 | Not reported; see reason below | 91 / 1 / 123 |
| document__g_n_j__laya_text | document__g_n_j__bge_reranker_text | recall@5 | 0.2087 | 0.6413 | +0.4326 | Not reported; see reason below | 178 / 6 / 31 |
| document__g_n_j__laya_text | document__g_n_j__bge_reranker_text | ndcg@10 | 0.2758 | 0.7402 | +0.4644 | Not reported; see reason below | 193 / 10 / 12 |
| document__g_s_j__none | document__g_s_j__bge_reranker_text | hit@5 | 0.9581 | 0.9721 | +0.0140 | Not reported; see reason below | 4 / 1 / 210 |
| document__g_s_j__none | document__g_s_j__bge_reranker_text | recall@5 | 0.6123 | 0.6399 | +0.0276 | Not reported; see reason below | 50 / 37 / 128 |
| document__g_s_j__none | document__g_s_j__bge_reranker_text | ndcg@10 | 0.7303 | 0.7390 | +0.0087 | Not reported; see reason below | 93 / 78 / 44 |
| document__g_s_j__laya_text | document__g_s_j__bge_reranker_text | hit@5 | 0.5442 | 0.9721 | +0.4279 | Not reported; see reason below | 93 / 1 / 121 |
| document__g_s_j__laya_text | document__g_s_j__bge_reranker_text | recall@5 | 0.1997 | 0.6399 | +0.4402 | Not reported; see reason below | 181 / 4 / 30 |
| document__g_s_j__laya_text | document__g_s_j__bge_reranker_text | ndcg@10 | 0.2705 | 0.7390 | +0.4685 | Not reported; see reason below | 195 / 9 / 11 |
| document__e_n_s__none | document__e_n_s__bge_reranker_text | hit@5 | 0.9628 | 0.9674 | +0.0047 | Not reported; see reason below | 2 / 1 / 212 |
| document__e_n_s__none | document__e_n_s__bge_reranker_text | recall@5 | 0.6023 | 0.6391 | +0.0368 | Not reported; see reason below | 59 / 37 / 119 |
| document__e_n_s__none | document__e_n_s__bge_reranker_text | ndcg@10 | 0.7330 | 0.7423 | +0.0093 | Not reported; see reason below | 90 / 81 / 44 |
| document__e_n_s__laya_text | document__e_n_s__bge_reranker_text | hit@5 | 0.5535 | 0.9674 | +0.4140 | Not reported; see reason below | 90 / 1 / 124 |
| document__e_n_s__laya_text | document__e_n_s__bge_reranker_text | recall@5 | 0.2040 | 0.6391 | +0.4351 | Not reported; see reason below | 178 / 4 / 33 |
| document__e_n_s__laya_text | document__e_n_s__bge_reranker_text | ndcg@10 | 0.2722 | 0.7423 | +0.4701 | Not reported; see reason below | 194 / 10 / 11 |
| document__e_n_j__none | document__e_n_j__bge_reranker_text | hit@5 | 0.9628 | 0.9721 | +0.0093 | Not reported; see reason below | 4 / 2 / 209 |
| document__e_n_j__none | document__e_n_j__bge_reranker_text | recall@5 | 0.5652 | 0.6449 | +0.0797 | Not reported; see reason below | 75 / 32 / 108 |
| document__e_n_j__none | document__e_n_j__bge_reranker_text | ndcg@10 | 0.6764 | 0.7394 | +0.0630 | Not reported; see reason below | 118 / 57 / 40 |
| document__e_n_j__laya_text | document__e_n_j__bge_reranker_text | hit@5 | 0.5488 | 0.9721 | +0.4233 | Not reported; see reason below | 92 / 1 / 122 |
| document__e_n_j__laya_text | document__e_n_j__bge_reranker_text | recall@5 | 0.2048 | 0.6449 | +0.4401 | Not reported; see reason below | 179 / 4 / 32 |
| document__e_n_j__laya_text | document__e_n_j__bge_reranker_text | ndcg@10 | 0.2810 | 0.7394 | +0.4584 | Not reported; see reason below | 195 / 9 / 11 |
| document__e_s_j__none | document__e_s_j__bge_reranker_text | hit@5 | 0.9674 | 0.9674 | +0.0000 | Not reported; see reason below | 2 / 2 / 211 |
| document__e_s_j__none | document__e_s_j__bge_reranker_text | recall@5 | 0.5981 | 0.6359 | +0.0378 | Not reported; see reason below | 55 / 39 / 121 |
| document__e_s_j__none | document__e_s_j__bge_reranker_text | ndcg@10 | 0.7327 | 0.7405 | +0.0078 | Not reported; see reason below | 97 / 78 / 40 |
| document__e_s_j__laya_text | document__e_s_j__bge_reranker_text | hit@5 | 0.5349 | 0.9674 | +0.4326 | Not reported; see reason below | 94 / 1 / 120 |
| document__e_s_j__laya_text | document__e_s_j__bge_reranker_text | recall@5 | 0.1999 | 0.6359 | +0.4360 | Not reported; see reason below | 180 / 3 / 32 |
| document__e_s_j__laya_text | document__e_s_j__bge_reranker_text | ndcg@10 | 0.2775 | 0.7405 | +0.4630 | Not reported; see reason below | 194 / 10 / 11 |
| document__n_s_j__none | document__n_s_j__bge_reranker_text | hit@5 | 0.9721 | 0.9721 | +0.0000 | Not reported; see reason below | 2 / 2 / 211 |
| document__n_s_j__none | document__n_s_j__bge_reranker_text | recall@5 | 0.6113 | 0.6438 | +0.0325 | Not reported; see reason below | 57 / 40 / 118 |
| document__n_s_j__none | document__n_s_j__bge_reranker_text | ndcg@10 | 0.7258 | 0.7425 | +0.0167 | Not reported; see reason below | 99 / 73 / 43 |
| document__n_s_j__laya_text | document__n_s_j__bge_reranker_text | hit@5 | 0.5488 | 0.9721 | +0.4233 | Not reported; see reason below | 92 / 1 / 122 |
| document__n_s_j__laya_text | document__n_s_j__bge_reranker_text | recall@5 | 0.2024 | 0.6438 | +0.4414 | Not reported; see reason below | 179 / 2 / 34 |
| document__n_s_j__laya_text | document__n_s_j__bge_reranker_text | ndcg@10 | 0.2704 | 0.7425 | +0.4722 | Not reported; see reason below | 196 / 9 / 10 |
| document__b_g_e_n__none | document__b_g_e_n__bge_reranker_text | hit@5 | 0.9721 | 0.9674 | -0.0047 | Not reported; see reason below | 2 / 3 / 210 |
| document__b_g_e_n__none | document__b_g_e_n__bge_reranker_text | recall@5 | 0.6003 | 0.6398 | +0.0395 | Not reported; see reason below | 60 / 35 / 120 |
| document__b_g_e_n__none | document__b_g_e_n__bge_reranker_text | ndcg@10 | 0.7063 | 0.7393 | +0.0331 | Not reported; see reason below | 108 / 59 / 48 |
| document__b_g_e_n__laya_text | document__b_g_e_n__bge_reranker_text | hit@5 | 0.5488 | 0.9674 | +0.4186 | Not reported; see reason below | 91 / 1 / 123 |
| document__b_g_e_n__laya_text | document__b_g_e_n__bge_reranker_text | recall@5 | 0.2005 | 0.6398 | +0.4393 | Not reported; see reason below | 181 / 4 / 30 |
| document__b_g_e_n__laya_text | document__b_g_e_n__bge_reranker_text | ndcg@10 | 0.2731 | 0.7393 | +0.4663 | Not reported; see reason below | 193 / 11 / 11 |
| document__b_g_e_s__none | document__b_g_e_s__bge_reranker_text | hit@5 | 0.9767 | 0.9674 | -0.0093 | Not reported; see reason below | 0 / 2 / 213 |
| document__b_g_e_s__none | document__b_g_e_s__bge_reranker_text | recall@5 | 0.6260 | 0.6383 | +0.0123 | Not reported; see reason below | 46 / 39 / 130 |
| document__b_g_e_s__none | document__b_g_e_s__bge_reranker_text | ndcg@10 | 0.7357 | 0.7379 | +0.0022 | Not reported; see reason below | 91 / 75 / 49 |
| document__b_g_e_s__laya_text | document__b_g_e_s__bge_reranker_text | hit@5 | 0.5442 | 0.9674 | +0.4233 | Not reported; see reason below | 92 / 1 / 122 |
| document__b_g_e_s__laya_text | document__b_g_e_s__bge_reranker_text | recall@5 | 0.2016 | 0.6383 | +0.4367 | Not reported; see reason below | 181 / 5 / 29 |
| document__b_g_e_s__laya_text | document__b_g_e_s__bge_reranker_text | ndcg@10 | 0.2770 | 0.7379 | +0.4609 | Not reported; see reason below | 193 / 11 / 11 |
| document__b_g_e_j__none | document__b_g_e_j__bge_reranker_text | hit@5 | 0.9674 | 0.9674 | +0.0000 | Not reported; see reason below | 2 / 2 / 211 |
| document__b_g_e_j__none | document__b_g_e_j__bge_reranker_text | recall@5 | 0.5967 | 0.6355 | +0.0387 | Not reported; see reason below | 55 / 34 / 126 |
| document__b_g_e_j__none | document__b_g_e_j__bge_reranker_text | ndcg@10 | 0.7105 | 0.7383 | +0.0278 | Not reported; see reason below | 107 / 60 / 48 |
| document__b_g_e_j__laya_text | document__b_g_e_j__bge_reranker_text | hit@5 | 0.5535 | 0.9674 | +0.4140 | Not reported; see reason below | 90 / 1 / 124 |
| document__b_g_e_j__laya_text | document__b_g_e_j__bge_reranker_text | recall@5 | 0.2030 | 0.6355 | +0.4324 | Not reported; see reason below | 181 / 4 / 30 |
| document__b_g_e_j__laya_text | document__b_g_e_j__bge_reranker_text | ndcg@10 | 0.2761 | 0.7383 | +0.4622 | Not reported; see reason below | 195 / 9 / 11 |
| document__b_g_n_s__none | document__b_g_n_s__bge_reranker_text | hit@5 | 0.9767 | 0.9674 | -0.0093 | Not reported; see reason below | 1 / 3 / 211 |
| document__b_g_n_s__none | document__b_g_n_s__bge_reranker_text | recall@5 | 0.6212 | 0.6387 | +0.0176 | Not reported; see reason below | 46 / 36 / 133 |
| document__b_g_n_s__none | document__b_g_n_s__bge_reranker_text | ndcg@10 | 0.7304 | 0.7388 | +0.0083 | Not reported; see reason below | 94 / 75 / 46 |
| document__b_g_n_s__laya_text | document__b_g_n_s__bge_reranker_text | hit@5 | 0.5488 | 0.9674 | +0.4186 | Not reported; see reason below | 91 / 1 / 123 |
| document__b_g_n_s__laya_text | document__b_g_n_s__bge_reranker_text | recall@5 | 0.2043 | 0.6387 | +0.4345 | Not reported; see reason below | 180 / 5 / 30 |
| document__b_g_n_s__laya_text | document__b_g_n_s__bge_reranker_text | ndcg@10 | 0.2698 | 0.7388 | +0.4690 | Not reported; see reason below | 194 / 10 / 11 |
| document__b_g_n_j__none | document__b_g_n_j__bge_reranker_text | hit@5 | 0.9721 | 0.9721 | +0.0000 | Not reported; see reason below | 1 / 1 / 213 |
| document__b_g_n_j__none | document__b_g_n_j__bge_reranker_text | recall@5 | 0.6016 | 0.6444 | +0.0428 | Not reported; see reason below | 60 / 32 / 123 |
| document__b_g_n_j__none | document__b_g_n_j__bge_reranker_text | ndcg@10 | 0.7110 | 0.7388 | +0.0279 | Not reported; see reason below | 104 / 60 / 51 |
| document__b_g_n_j__laya_text | document__b_g_n_j__bge_reranker_text | hit@5 | 0.5442 | 0.9721 | +0.4279 | Not reported; see reason below | 93 / 1 / 121 |
| document__b_g_n_j__laya_text | document__b_g_n_j__bge_reranker_text | recall@5 | 0.1980 | 0.6444 | +0.4465 | Not reported; see reason below | 181 / 3 / 31 |
| document__b_g_n_j__laya_text | document__b_g_n_j__bge_reranker_text | ndcg@10 | 0.2742 | 0.7388 | +0.4647 | Not reported; see reason below | 193 / 11 / 11 |
| document__b_g_s_j__none | document__b_g_s_j__bge_reranker_text | hit@5 | 0.9721 | 0.9721 | +0.0000 | Not reported; see reason below | 1 / 1 / 213 |
| document__b_g_s_j__none | document__b_g_s_j__bge_reranker_text | recall@5 | 0.6241 | 0.6434 | +0.0193 | Not reported; see reason below | 47 / 35 / 133 |
| document__b_g_s_j__none | document__b_g_s_j__bge_reranker_text | ndcg@10 | 0.7371 | 0.7379 | +0.0008 | Not reported; see reason below | 86 / 76 / 53 |
| document__b_g_s_j__laya_text | document__b_g_s_j__bge_reranker_text | hit@5 | 0.5535 | 0.9721 | +0.4186 | Not reported; see reason below | 91 / 1 / 123 |
| document__b_g_s_j__laya_text | document__b_g_s_j__bge_reranker_text | recall@5 | 0.2041 | 0.6434 | +0.4393 | Not reported; see reason below | 180 / 6 / 29 |
| document__b_g_s_j__laya_text | document__b_g_s_j__bge_reranker_text | ndcg@10 | 0.2736 | 0.7379 | +0.4643 | Not reported; see reason below | 195 / 8 / 12 |
| document__b_e_n_s__none | document__b_e_n_s__bge_reranker_text | hit@5 | 0.9767 | 0.9721 | -0.0047 | Not reported; see reason below | 0 / 1 / 214 |
| document__b_e_n_s__none | document__b_e_n_s__bge_reranker_text | recall@5 | 0.6201 | 0.6440 | +0.0239 | Not reported; see reason below | 54 / 37 / 124 |
| document__b_e_n_s__none | document__b_e_n_s__bge_reranker_text | ndcg@10 | 0.7332 | 0.7401 | +0.0068 | Not reported; see reason below | 92 / 75 / 48 |
| document__b_e_n_s__laya_text | document__b_e_n_s__bge_reranker_text | hit@5 | 0.5581 | 0.9721 | +0.4140 | Not reported; see reason below | 90 / 1 / 124 |
| document__b_e_n_s__laya_text | document__b_e_n_s__bge_reranker_text | recall@5 | 0.2061 | 0.6440 | +0.4379 | Not reported; see reason below | 181 / 4 / 30 |
| document__b_e_n_s__laya_text | document__b_e_n_s__bge_reranker_text | ndcg@10 | 0.2772 | 0.7401 | +0.4628 | Not reported; see reason below | 194 / 10 / 11 |
| document__b_e_n_j__none | document__b_e_n_j__bge_reranker_text | hit@5 | 0.9628 | 0.9674 | +0.0047 | Not reported; see reason below | 3 / 2 / 210 |
| document__b_e_n_j__none | document__b_e_n_j__bge_reranker_text | recall@5 | 0.5933 | 0.6364 | +0.0431 | Not reported; see reason below | 63 / 38 / 114 |
| document__b_e_n_j__none | document__b_e_n_j__bge_reranker_text | ndcg@10 | 0.7095 | 0.7389 | +0.0294 | Not reported; see reason below | 103 / 66 / 46 |
| document__b_e_n_j__laya_text | document__b_e_n_j__bge_reranker_text | hit@5 | 0.5535 | 0.9674 | +0.4140 | Not reported; see reason below | 90 / 1 / 124 |
| document__b_e_n_j__laya_text | document__b_e_n_j__bge_reranker_text | recall@5 | 0.2016 | 0.6364 | +0.4348 | Not reported; see reason below | 180 / 4 / 31 |
| document__b_e_n_j__laya_text | document__b_e_n_j__bge_reranker_text | ndcg@10 | 0.2758 | 0.7389 | +0.4630 | Not reported; see reason below | 194 / 10 / 11 |
| document__b_e_s_j__none | document__b_e_s_j__bge_reranker_text | hit@5 | 0.9767 | 0.9721 | -0.0047 | Not reported; see reason below | 0 / 1 / 214 |
| document__b_e_s_j__none | document__b_e_s_j__bge_reranker_text | recall@5 | 0.6145 | 0.6434 | +0.0289 | Not reported; see reason below | 54 / 41 / 120 |
| document__b_e_s_j__none | document__b_e_s_j__bge_reranker_text | ndcg@10 | 0.7374 | 0.7398 | +0.0025 | Not reported; see reason below | 95 / 76 / 44 |
| document__b_e_s_j__laya_text | document__b_e_s_j__bge_reranker_text | hit@5 | 0.5442 | 0.9721 | +0.4279 | Not reported; see reason below | 93 / 1 / 121 |
| document__b_e_s_j__laya_text | document__b_e_s_j__bge_reranker_text | recall@5 | 0.1994 | 0.6434 | +0.4439 | Not reported; see reason below | 180 / 3 / 32 |
| document__b_e_s_j__laya_text | document__b_e_s_j__bge_reranker_text | ndcg@10 | 0.2820 | 0.7398 | +0.4578 | Not reported; see reason below | 195 / 9 / 11 |
| document__b_n_s_j__none | document__b_n_s_j__bge_reranker_text | hit@5 | 0.9628 | 0.9721 | +0.0093 | Not reported; see reason below | 3 / 1 / 211 |
| document__b_n_s_j__none | document__b_n_s_j__bge_reranker_text | recall@5 | 0.6113 | 0.6447 | +0.0334 | Not reported; see reason below | 56 / 37 / 122 |
| document__b_n_s_j__none | document__b_n_s_j__bge_reranker_text | ndcg@10 | 0.7298 | 0.7409 | +0.0111 | Not reported; see reason below | 96 / 76 / 43 |
| document__b_n_s_j__laya_text | document__b_n_s_j__bge_reranker_text | hit@5 | 0.5535 | 0.9721 | +0.4186 | Not reported; see reason below | 91 / 1 / 123 |
| document__b_n_s_j__laya_text | document__b_n_s_j__bge_reranker_text | recall@5 | 0.2026 | 0.6447 | +0.4420 | Not reported; see reason below | 179 / 3 / 33 |
| document__b_n_s_j__laya_text | document__b_n_s_j__bge_reranker_text | ndcg@10 | 0.2729 | 0.7409 | +0.4680 | Not reported; see reason below | 194 / 9 / 12 |
| document__g_e_n_s__none | document__g_e_n_s__bge_reranker_text | hit@5 | 0.9721 | 0.9674 | -0.0047 | Not reported; see reason below | 2 / 3 / 210 |
| document__g_e_n_s__none | document__g_e_n_s__bge_reranker_text | recall@5 | 0.6181 | 0.6392 | +0.0211 | Not reported; see reason below | 51 / 36 / 128 |
| document__g_e_n_s__none | document__g_e_n_s__bge_reranker_text | ndcg@10 | 0.7267 | 0.7423 | +0.0156 | Not reported; see reason below | 96 / 74 / 45 |
| document__g_e_n_s__laya_text | document__g_e_n_s__bge_reranker_text | hit@5 | 0.5442 | 0.9674 | +0.4233 | Not reported; see reason below | 92 / 1 / 122 |
| document__g_e_n_s__laya_text | document__g_e_n_s__bge_reranker_text | recall@5 | 0.2031 | 0.6392 | +0.4361 | Not reported; see reason below | 180 / 4 / 31 |
| document__g_e_n_s__laya_text | document__g_e_n_s__bge_reranker_text | ndcg@10 | 0.2745 | 0.7423 | +0.4678 | Not reported; see reason below | 195 / 8 / 12 |
| document__g_e_n_j__none | document__g_e_n_j__bge_reranker_text | hit@5 | 0.9581 | 0.9721 | +0.0140 | Not reported; see reason below | 5 / 2 / 208 |
| document__g_e_n_j__none | document__g_e_n_j__bge_reranker_text | recall@5 | 0.5803 | 0.6389 | +0.0586 | Not reported; see reason below | 59 / 32 / 124 |
| document__g_e_n_j__none | document__g_e_n_j__bge_reranker_text | ndcg@10 | 0.6928 | 0.7386 | +0.0458 | Not reported; see reason below | 112 / 64 / 39 |
| document__g_e_n_j__laya_text | document__g_e_n_j__bge_reranker_text | hit@5 | 0.5721 | 0.9721 | +0.4000 | Not reported; see reason below | 87 / 1 / 127 |
| document__g_e_n_j__laya_text | document__g_e_n_j__bge_reranker_text | recall@5 | 0.2042 | 0.6389 | +0.4347 | Not reported; see reason below | 178 / 3 / 34 |
| document__g_e_n_j__laya_text | document__g_e_n_j__bge_reranker_text | ndcg@10 | 0.2815 | 0.7386 | +0.4571 | Not reported; see reason below | 195 / 9 / 11 |
| document__g_e_s_j__none | document__g_e_s_j__bge_reranker_text | hit@5 | 0.9581 | 0.9721 | +0.0140 | Not reported; see reason below | 3 / 0 / 212 |
| document__g_e_s_j__none | document__g_e_s_j__bge_reranker_text | recall@5 | 0.6101 | 0.6424 | +0.0323 | Not reported; see reason below | 50 / 38 / 127 |
| document__g_e_s_j__none | document__g_e_s_j__bge_reranker_text | ndcg@10 | 0.7273 | 0.7413 | +0.0140 | Not reported; see reason below | 90 / 79 / 46 |
| document__g_e_s_j__laya_text | document__g_e_s_j__bge_reranker_text | hit@5 | 0.5581 | 0.9721 | +0.4140 | Not reported; see reason below | 90 / 1 / 124 |
| document__g_e_s_j__laya_text | document__g_e_s_j__bge_reranker_text | recall@5 | 0.2072 | 0.6424 | +0.4353 | Not reported; see reason below | 178 / 3 / 34 |
| document__g_e_s_j__laya_text | document__g_e_s_j__bge_reranker_text | ndcg@10 | 0.2763 | 0.7413 | +0.4651 | Not reported; see reason below | 194 / 9 / 12 |
| document__g_n_s_j__none | document__g_n_s_j__bge_reranker_text | hit@5 | 0.9721 | 0.9721 | +0.0000 | Not reported; see reason below | 2 / 2 / 211 |
| document__g_n_s_j__none | document__g_n_s_j__bge_reranker_text | recall@5 | 0.6194 | 0.6429 | +0.0235 | Not reported; see reason below | 48 / 37 / 130 |
| document__g_n_s_j__none | document__g_n_s_j__bge_reranker_text | ndcg@10 | 0.7268 | 0.7417 | +0.0149 | Not reported; see reason below | 92 / 76 / 47 |
| document__g_n_s_j__laya_text | document__g_n_s_j__bge_reranker_text | hit@5 | 0.5442 | 0.9721 | +0.4279 | Not reported; see reason below | 93 / 1 / 121 |
| document__g_n_s_j__laya_text | document__g_n_s_j__bge_reranker_text | recall@5 | 0.2036 | 0.6429 | +0.4393 | Not reported; see reason below | 179 / 5 / 31 |
| document__g_n_s_j__laya_text | document__g_n_s_j__bge_reranker_text | ndcg@10 | 0.2761 | 0.7417 | +0.4656 | Not reported; see reason below | 195 / 8 / 12 |
| document__e_n_s_j__none | document__e_n_s_j__bge_reranker_text | hit@5 | 0.9767 | 0.9674 | -0.0093 | Not reported; see reason below | 1 / 3 / 211 |
| document__e_n_s_j__none | document__e_n_s_j__bge_reranker_text | recall@5 | 0.5973 | 0.6364 | +0.0391 | Not reported; see reason below | 60 / 36 / 119 |
| document__e_n_s_j__none | document__e_n_s_j__bge_reranker_text | ndcg@10 | 0.7258 | 0.7408 | +0.0151 | Not reported; see reason below | 98 / 72 / 45 |
| document__e_n_s_j__laya_text | document__e_n_s_j__bge_reranker_text | hit@5 | 0.5628 | 0.9674 | +0.4047 | Not reported; see reason below | 88 / 1 / 126 |
| document__e_n_s_j__laya_text | document__e_n_s_j__bge_reranker_text | recall@5 | 0.2067 | 0.6364 | +0.4297 | Not reported; see reason below | 178 / 3 / 34 |
| document__e_n_s_j__laya_text | document__e_n_s_j__bge_reranker_text | ndcg@10 | 0.2751 | 0.7408 | +0.4657 | Not reported; see reason below | 194 / 10 / 11 |
| document__b_g_e_n_s__none | document__b_g_e_n_s__bge_reranker_text | hit@5 | 0.9674 | 0.9721 | +0.0047 | Not reported; see reason below | 2 / 1 / 212 |
| document__b_g_e_n_s__none | document__b_g_e_n_s__bge_reranker_text | recall@5 | 0.6199 | 0.6434 | +0.0235 | Not reported; see reason below | 50 / 35 / 130 |
| document__b_g_e_n_s__none | document__b_g_e_n_s__bge_reranker_text | ndcg@10 | 0.7301 | 0.7392 | +0.0092 | Not reported; see reason below | 93 / 72 / 50 |
| document__b_g_e_n_s__laya_text | document__b_g_e_n_s__bge_reranker_text | hit@5 | 0.5488 | 0.9721 | +0.4233 | Not reported; see reason below | 92 / 1 / 122 |
| document__b_g_e_n_s__laya_text | document__b_g_e_n_s__bge_reranker_text | recall@5 | 0.2026 | 0.6434 | +0.4408 | Not reported; see reason below | 182 / 4 / 29 |
| document__b_g_e_n_s__laya_text | document__b_g_e_n_s__bge_reranker_text | ndcg@10 | 0.2712 | 0.7392 | +0.4680 | Not reported; see reason below | 195 / 9 / 11 |
| document__b_g_e_n_j__none | document__b_g_e_n_j__bge_reranker_text | hit@5 | 0.9674 | 0.9674 | +0.0000 | Not reported; see reason below | 3 / 3 / 209 |
| document__b_g_e_n_j__none | document__b_g_e_n_j__bge_reranker_text | recall@5 | 0.5931 | 0.6389 | +0.0458 | Not reported; see reason below | 63 / 38 / 114 |
| document__b_g_e_n_j__none | document__b_g_e_n_j__bge_reranker_text | ndcg@10 | 0.7132 | 0.7402 | +0.0271 | Not reported; see reason below | 101 / 66 / 48 |
| document__b_g_e_n_j__laya_text | document__b_g_e_n_j__bge_reranker_text | hit@5 | 0.5488 | 0.9674 | +0.4186 | Not reported; see reason below | 91 / 1 / 123 |
| document__b_g_e_n_j__laya_text | document__b_g_e_n_j__bge_reranker_text | recall@5 | 0.2030 | 0.6389 | +0.4358 | Not reported; see reason below | 179 / 4 / 32 |
| document__b_g_e_n_j__laya_text | document__b_g_e_n_j__bge_reranker_text | ndcg@10 | 0.2775 | 0.7402 | +0.4628 | Not reported; see reason below | 195 / 9 / 11 |
| document__b_g_e_s_j__none | document__b_g_e_s_j__bge_reranker_text | hit@5 | 0.9721 | 0.9674 | -0.0047 | Not reported; see reason below | 1 / 2 / 212 |
| document__b_g_e_s_j__none | document__b_g_e_s_j__bge_reranker_text | recall@5 | 0.6212 | 0.6392 | +0.0181 | Not reported; see reason below | 51 / 37 / 127 |
| document__b_g_e_s_j__none | document__b_g_e_s_j__bge_reranker_text | ndcg@10 | 0.7314 | 0.7393 | +0.0079 | Not reported; see reason below | 92 / 73 / 50 |
| document__b_g_e_s_j__laya_text | document__b_g_e_s_j__bge_reranker_text | hit@5 | 0.5581 | 0.9674 | +0.4093 | Not reported; see reason below | 89 / 1 / 125 |
| document__b_g_e_s_j__laya_text | document__b_g_e_s_j__bge_reranker_text | recall@5 | 0.2078 | 0.6392 | +0.4315 | Not reported; see reason below | 181 / 6 / 28 |
| document__b_g_e_s_j__laya_text | document__b_g_e_s_j__bge_reranker_text | ndcg@10 | 0.2784 | 0.7393 | +0.4610 | Not reported; see reason below | 195 / 8 / 12 |
| document__b_g_n_s_j__none | document__b_g_n_s_j__bge_reranker_text | hit@5 | 0.9721 | 0.9721 | +0.0000 | Not reported; see reason below | 1 / 1 / 213 |
| document__b_g_n_s_j__none | document__b_g_n_s_j__bge_reranker_text | recall@5 | 0.6196 | 0.6439 | +0.0244 | Not reported; see reason below | 47 / 36 / 132 |
| document__b_g_n_s_j__none | document__b_g_n_s_j__bge_reranker_text | ndcg@10 | 0.7339 | 0.7394 | +0.0056 | Not reported; see reason below | 90 / 73 / 52 |
| document__b_g_n_s_j__laya_text | document__b_g_n_s_j__bge_reranker_text | hit@5 | 0.5442 | 0.9721 | +0.4279 | Not reported; see reason below | 93 / 1 / 121 |
| document__b_g_n_s_j__laya_text | document__b_g_n_s_j__bge_reranker_text | recall@5 | 0.2031 | 0.6439 | +0.4409 | Not reported; see reason below | 181 / 5 / 29 |
| document__b_g_n_s_j__laya_text | document__b_g_n_s_j__bge_reranker_text | ndcg@10 | 0.2714 | 0.7394 | +0.4680 | Not reported; see reason below | 195 / 8 / 12 |
| document__b_e_n_s_j__none | document__b_e_n_s_j__bge_reranker_text | hit@5 | 0.9674 | 0.9721 | +0.0047 | Not reported; see reason below | 2 / 1 / 212 |
| document__b_e_n_s_j__none | document__b_e_n_s_j__bge_reranker_text | recall@5 | 0.6149 | 0.6438 | +0.0289 | Not reported; see reason below | 52 / 38 / 125 |
| document__b_e_n_s_j__none | document__b_e_n_s_j__bge_reranker_text | ndcg@10 | 0.7349 | 0.7408 | +0.0059 | Not reported; see reason below | 92 / 77 / 46 |
| document__b_e_n_s_j__laya_text | document__b_e_n_s_j__bge_reranker_text | hit@5 | 0.5535 | 0.9721 | +0.4186 | Not reported; see reason below | 91 / 1 / 123 |
| document__b_e_n_s_j__laya_text | document__b_e_n_s_j__bge_reranker_text | recall@5 | 0.2060 | 0.6438 | +0.4378 | Not reported; see reason below | 178 / 3 / 34 |
| document__b_e_n_s_j__laya_text | document__b_e_n_s_j__bge_reranker_text | ndcg@10 | 0.2761 | 0.7408 | +0.4646 | Not reported; see reason below | 194 / 9 / 12 |
| document__g_e_n_s_j__none | document__g_e_n_s_j__bge_reranker_text | hit@5 | 0.9721 | 0.9674 | -0.0047 | Not reported; see reason below | 2 / 3 / 210 |
| document__g_e_n_s_j__none | document__g_e_n_s_j__bge_reranker_text | recall@5 | 0.6150 | 0.6387 | +0.0237 | Not reported; see reason below | 53 / 41 / 121 |
| document__g_e_n_s_j__none | document__g_e_n_s_j__bge_reranker_text | ndcg@10 | 0.7240 | 0.7423 | +0.0183 | Not reported; see reason below | 95 / 73 / 47 |
| document__g_e_n_s_j__laya_text | document__g_e_n_s_j__bge_reranker_text | hit@5 | 0.5535 | 0.9674 | +0.4140 | Not reported; see reason below | 90 / 1 / 124 |
| document__g_e_n_s_j__laya_text | document__g_e_n_s_j__bge_reranker_text | recall@5 | 0.2063 | 0.6387 | +0.4325 | Not reported; see reason below | 177 / 3 / 35 |
| document__g_e_n_s_j__laya_text | document__g_e_n_s_j__bge_reranker_text | ndcg@10 | 0.2731 | 0.7423 | +0.4691 | Not reported; see reason below | 196 / 8 / 11 |
| document__b_g_e_n_s_j__none | document__b_g_e_n_s_j__bge_reranker_text | hit@5 | 0.9674 | 0.9674 | +0.0000 | Not reported; see reason below | 2 / 2 / 211 |
| document__b_g_e_n_s_j__none | document__b_g_e_n_s_j__bge_reranker_text | recall@5 | 0.6124 | 0.6402 | +0.0277 | Not reported; see reason below | 55 / 39 / 121 |
| document__b_g_e_n_s_j__none | document__b_g_e_n_s_j__bge_reranker_text | ndcg@10 | 0.7351 | 0.7416 | +0.0064 | Not reported; see reason below | 95 / 73 / 47 |
| document__b_g_e_n_s_j__laya_text | document__b_g_e_n_s_j__bge_reranker_text | hit@5 | 0.5488 | 0.9674 | +0.4186 | Not reported; see reason below | 91 / 1 / 123 |
| document__b_g_e_n_s_j__laya_text | document__b_g_e_n_s_j__bge_reranker_text | recall@5 | 0.2033 | 0.6402 | +0.4368 | Not reported; see reason below | 179 / 5 / 31 |
| document__b_g_e_n_s_j__laya_text | document__b_g_e_n_s_j__bge_reranker_text | ndcg@10 | 0.2774 | 0.7416 | +0.4642 | Not reported; see reason below | 195 / 8 / 12 |

document__b__none → document__b__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__b__laya_text → document__b__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__g__none → document__g__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__g__laya_text → document__g__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__e__none → document__e__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__e__laya_text → document__e__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__n__none → document__n__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__n__laya_text → document__n__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__s__none → document__s__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__s__laya_text → document__s__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__j__none → document__j__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__j__laya_text → document__j__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_g__none → document__b_g__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_g__laya_text → document__b_g__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_e__none → document__b_e__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_e__laya_text → document__b_e__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_n__none → document__b_n__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_n__laya_text → document__b_n__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_s__none → document__b_s__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_s__laya_text → document__b_s__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_j__none → document__b_j__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_j__laya_text → document__b_j__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__g_e__none → document__g_e__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__g_e__laya_text → document__g_e__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__g_n__none → document__g_n__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__g_n__laya_text → document__g_n__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__g_s__none → document__g_s__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__g_s__laya_text → document__g_s__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__g_j__none → document__g_j__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__g_j__laya_text → document__g_j__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__e_n__none → document__e_n__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__e_n__laya_text → document__e_n__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__e_s__none → document__e_s__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__e_s__laya_text → document__e_s__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__e_j__none → document__e_j__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__e_j__laya_text → document__e_j__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__n_s__none → document__n_s__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__n_s__laya_text → document__n_s__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__n_j__none → document__n_j__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__n_j__laya_text → document__n_j__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__s_j__none → document__s_j__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__s_j__laya_text → document__s_j__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_g_e__none → document__b_g_e__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_g_e__laya_text → document__b_g_e__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_g_n__none → document__b_g_n__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_g_n__laya_text → document__b_g_n__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_g_s__none → document__b_g_s__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_g_s__laya_text → document__b_g_s__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_g_j__none → document__b_g_j__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_g_j__laya_text → document__b_g_j__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_e_n__none → document__b_e_n__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_e_n__laya_text → document__b_e_n__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_e_s__none → document__b_e_s__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_e_s__laya_text → document__b_e_s__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_e_j__none → document__b_e_j__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_e_j__laya_text → document__b_e_j__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_n_s__none → document__b_n_s__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_n_s__laya_text → document__b_n_s__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_n_j__none → document__b_n_j__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_n_j__laya_text → document__b_n_j__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_s_j__none → document__b_s_j__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_s_j__laya_text → document__b_s_j__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__g_e_n__none → document__g_e_n__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__g_e_n__laya_text → document__g_e_n__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__g_e_s__none → document__g_e_s__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__g_e_s__laya_text → document__g_e_s__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__g_e_j__none → document__g_e_j__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__g_e_j__laya_text → document__g_e_j__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__g_n_s__none → document__g_n_s__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__g_n_s__laya_text → document__g_n_s__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__g_n_j__none → document__g_n_j__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__g_n_j__laya_text → document__g_n_j__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__g_s_j__none → document__g_s_j__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__g_s_j__laya_text → document__g_s_j__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__e_n_s__none → document__e_n_s__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__e_n_s__laya_text → document__e_n_s__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__e_n_j__none → document__e_n_j__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__e_n_j__laya_text → document__e_n_j__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__e_s_j__none → document__e_s_j__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__e_s_j__laya_text → document__e_s_j__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__n_s_j__none → document__n_s_j__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__n_s_j__laya_text → document__n_s_j__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_g_e_n__none → document__b_g_e_n__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_g_e_n__laya_text → document__b_g_e_n__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_g_e_s__none → document__b_g_e_s__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_g_e_s__laya_text → document__b_g_e_s__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_g_e_j__none → document__b_g_e_j__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_g_e_j__laya_text → document__b_g_e_j__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_g_n_s__none → document__b_g_n_s__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_g_n_s__laya_text → document__b_g_n_s__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_g_n_j__none → document__b_g_n_j__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_g_n_j__laya_text → document__b_g_n_j__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_g_s_j__none → document__b_g_s_j__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_g_s_j__laya_text → document__b_g_s_j__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_e_n_s__none → document__b_e_n_s__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_e_n_s__laya_text → document__b_e_n_s__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_e_n_j__none → document__b_e_n_j__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_e_n_j__laya_text → document__b_e_n_j__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_e_s_j__none → document__b_e_s_j__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_e_s_j__laya_text → document__b_e_s_j__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_n_s_j__none → document__b_n_s_j__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_n_s_j__laya_text → document__b_n_s_j__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__g_e_n_s__none → document__g_e_n_s__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__g_e_n_s__laya_text → document__g_e_n_s__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__g_e_n_j__none → document__g_e_n_j__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__g_e_n_j__laya_text → document__g_e_n_j__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__g_e_s_j__none → document__g_e_s_j__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__g_e_s_j__laya_text → document__g_e_s_j__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__g_n_s_j__none → document__g_n_s_j__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__g_n_s_j__laya_text → document__g_n_s_j__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__e_n_s_j__none → document__e_n_s_j__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__e_n_s_j__laya_text → document__e_n_s_j__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_g_e_n_s__none → document__b_g_e_n_s__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_g_e_n_s__laya_text → document__b_g_e_n_s__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_g_e_n_j__none → document__b_g_e_n_j__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_g_e_n_j__laya_text → document__b_g_e_n_j__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_g_e_s_j__none → document__b_g_e_s_j__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_g_e_s_j__laya_text → document__b_g_e_s_j__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_g_n_s_j__none → document__b_g_n_s_j__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_g_n_s_j__laya_text → document__b_g_n_s_j__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_e_n_s_j__none → document__b_e_n_s_j__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_e_n_s_j__laya_text → document__b_e_n_s_j__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__g_e_n_s_j__none → document__g_e_n_s_j__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__g_e_n_s_j__laya_text → document__g_e_n_s_j__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_g_e_n_s_j__none → document__b_g_e_n_s_j__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_g_e_n_s_j__laya_text → document__b_g_e_n_s_j__bge_reranker_text: Only one connected source group; a resampling confidence interval is not reported.

Bootstrap settings for eligible comparisons: 5000 samples; seed: 42. No model inference or generated-answer judging was performed.

Pairing follows [SciPy's paired resampling definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html); source groups follow [cluster resampling](https://www.stata.com/support/faqs/statistics/bootstrap-with-panel-data/).

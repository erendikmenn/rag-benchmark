# Paired retrieval comparisons

Dataset: vidore-v3-computer_science-en.

Differences are right minus left, in metric units. Positive values favor the right method. Where reported, intervals are 95% paired source-group percentile intervals; methods use the same frozen queries and gallery. These are marginal exploratory comparisons without a multiple-comparison correction.

Grouping: Positive pages map to corpus.metadata.doc_id. Queries touching the same source document are connected transitively across multi-document positives.

Source groups: 1; queries: 215; largest group: 215 queries (100.0%).

Conservative reporting safeguard: withhold intervals when one source group contains more than 50% of queries. This is not a formal statistical threshold.

| Left | Right | Metric | Left mean | Right mean | Delta | 95% interval | Wins / losses / ties |
|---|---|---|---:|---:|---:|---|---|
| document__b__none | document__b__laya_text | hit@5 | 0.9209 | 0.5116 | -0.4093 | Not reported; see reason below | 6 / 94 / 115 |
| document__b__none | document__b__laya_text | recall@5 | 0.5353 | 0.1947 | -0.3406 | Not reported; see reason below | 13 / 154 / 48 |
| document__b__none | document__b__laya_text | ndcg@10 | 0.6334 | 0.2638 | -0.3696 | Not reported; see reason below | 26 / 175 / 14 |
| document__g__none | document__g__laya_text | hit@5 | 0.9535 | 0.5488 | -0.4047 | Not reported; see reason below | 1 / 88 / 126 |
| document__g__none | document__g__laya_text | recall@5 | 0.5245 | 0.2024 | -0.3221 | Not reported; see reason below | 6 / 153 / 56 |
| document__g__none | document__g__laya_text | ndcg@10 | 0.6356 | 0.2829 | -0.3527 | Not reported; see reason below | 19 / 182 / 14 |
| document__e__none | document__e__laya_text | hit@5 | 0.9442 | 0.5628 | -0.3814 | Not reported; see reason below | 2 / 84 / 129 |
| document__e__none | document__e__laya_text | recall@5 | 0.5440 | 0.2063 | -0.3377 | Not reported; see reason below | 8 / 164 / 43 |
| document__e__none | document__e__laya_text | ndcg@10 | 0.6623 | 0.2873 | -0.3750 | Not reported; see reason below | 17 / 187 / 11 |
| document__n__none | document__n__laya_text | hit@5 | 0.9163 | 0.5535 | -0.3628 | Not reported; see reason below | 5 / 83 / 127 |
| document__n__none | document__n__laya_text | recall@5 | 0.5270 | 0.1987 | -0.3283 | Not reported; see reason below | 12 / 150 / 53 |
| document__n__none | document__n__laya_text | ndcg@10 | 0.6296 | 0.2620 | -0.3676 | Not reported; see reason below | 21 / 179 / 15 |
| document__s__none | document__s__laya_text | hit@5 | 0.9721 | 0.5442 | -0.4279 | Not reported; see reason below | 3 / 95 / 117 |
| document__s__none | document__s__laya_text | recall@5 | 0.6534 | 0.2043 | -0.4491 | Not reported; see reason below | 7 / 180 / 28 |
| document__s__none | document__s__laya_text | ndcg@10 | 0.7585 | 0.2688 | -0.4897 | Not reported; see reason below | 11 / 197 / 7 |
| document__j__none | document__j__laya_text | hit@5 | 0.9442 | 0.5535 | -0.3907 | Not reported; see reason below | 2 / 86 / 127 |
| document__j__none | document__j__laya_text | recall@5 | 0.5454 | 0.2055 | -0.3399 | Not reported; see reason below | 9 / 161 / 45 |
| document__j__none | document__j__laya_text | ndcg@10 | 0.6753 | 0.2776 | -0.3977 | Not reported; see reason below | 12 / 190 / 13 |
| document__b_g__none | document__b_g__laya_text | hit@5 | 0.9488 | 0.5535 | -0.3953 | Not reported; see reason below | 4 / 89 / 122 |
| document__b_g__none | document__b_g__laya_text | recall@5 | 0.5733 | 0.2079 | -0.3655 | Not reported; see reason below | 9 / 161 / 45 |
| document__b_g__none | document__b_g__laya_text | ndcg@10 | 0.6771 | 0.2744 | -0.4028 | Not reported; see reason below | 17 / 187 / 11 |
| document__b_e__none | document__b_e__laya_text | hit@5 | 0.9535 | 0.5628 | -0.3907 | Not reported; see reason below | 3 / 87 / 125 |
| document__b_e__none | document__b_e__laya_text | recall@5 | 0.5788 | 0.2101 | -0.3687 | Not reported; see reason below | 7 / 162 / 46 |
| document__b_e__none | document__b_e__laya_text | ndcg@10 | 0.6956 | 0.2824 | -0.4132 | Not reported; see reason below | 12 / 188 / 15 |
| document__b_n__none | document__b_n__laya_text | hit@5 | 0.9442 | 0.5628 | -0.3814 | Not reported; see reason below | 5 / 87 / 123 |
| document__b_n__none | document__b_n__laya_text | recall@5 | 0.5781 | 0.2114 | -0.3667 | Not reported; see reason below | 8 / 160 / 47 |
| document__b_n__none | document__b_n__laya_text | ndcg@10 | 0.6843 | 0.2773 | -0.4070 | Not reported; see reason below | 16 / 186 / 13 |
| document__b_s__none | document__b_s__laya_text | hit@5 | 0.9628 | 0.5674 | -0.3953 | Not reported; see reason below | 1 / 86 / 128 |
| document__b_s__none | document__b_s__laya_text | recall@5 | 0.6171 | 0.2071 | -0.4100 | Not reported; see reason below | 6 / 168 / 41 |
| document__b_s__none | document__b_s__laya_text | ndcg@10 | 0.7288 | 0.2788 | -0.4500 | Not reported; see reason below | 11 / 193 / 11 |
| document__b_j__none | document__b_j__laya_text | hit@5 | 0.9581 | 0.5488 | -0.4093 | Not reported; see reason below | 3 / 91 / 121 |
| document__b_j__none | document__b_j__laya_text | recall@5 | 0.5832 | 0.1975 | -0.3857 | Not reported; see reason below | 7 / 165 / 43 |
| document__b_j__none | document__b_j__laya_text | ndcg@10 | 0.6961 | 0.2805 | -0.4155 | Not reported; see reason below | 11 / 191 / 13 |
| document__g_e__none | document__g_e__laya_text | hit@5 | 0.9535 | 0.5767 | -0.3767 | Not reported; see reason below | 1 / 82 / 132 |
| document__g_e__none | document__g_e__laya_text | recall@5 | 0.5587 | 0.2108 | -0.3479 | Not reported; see reason below | 4 / 162 / 49 |
| document__g_e__none | document__g_e__laya_text | ndcg@10 | 0.6866 | 0.2883 | -0.3983 | Not reported; see reason below | 14 / 190 / 11 |
| document__g_n__none | document__g_n__laya_text | hit@5 | 0.9581 | 0.5628 | -0.3953 | Not reported; see reason below | 1 / 86 / 128 |
| document__g_n__none | document__g_n__laya_text | recall@5 | 0.5611 | 0.2113 | -0.3498 | Not reported; see reason below | 6 / 155 / 54 |
| document__g_n__none | document__g_n__laya_text | ndcg@10 | 0.6764 | 0.2748 | -0.4015 | Not reported; see reason below | 18 / 185 / 12 |
| document__g_s__none | document__g_s__laya_text | hit@5 | 0.9581 | 0.5535 | -0.4047 | Not reported; see reason below | 2 / 89 / 124 |
| document__g_s__none | document__g_s__laya_text | recall@5 | 0.6220 | 0.2027 | -0.4193 | Not reported; see reason below | 4 / 177 / 34 |
| document__g_s__none | document__g_s__laya_text | ndcg@10 | 0.7233 | 0.2689 | -0.4544 | Not reported; see reason below | 9 / 196 / 10 |
| document__g_j__none | document__g_j__laya_text | hit@5 | 0.9535 | 0.5628 | -0.3907 | Not reported; see reason below | 1 / 85 / 129 |
| document__g_j__none | document__g_j__laya_text | recall@5 | 0.5639 | 0.2035 | -0.3604 | Not reported; see reason below | 6 / 162 / 47 |
| document__g_j__none | document__g_j__laya_text | ndcg@10 | 0.6899 | 0.2832 | -0.4067 | Not reported; see reason below | 14 / 190 / 11 |
| document__e_n__none | document__e_n__laya_text | hit@5 | 0.9628 | 0.5535 | -0.4093 | Not reported; see reason below | 1 / 89 / 125 |
| document__e_n__none | document__e_n__laya_text | recall@5 | 0.5600 | 0.2073 | -0.3527 | Not reported; see reason below | 5 / 158 / 52 |
| document__e_n__none | document__e_n__laya_text | ndcg@10 | 0.6666 | 0.2800 | -0.3866 | Not reported; see reason below | 17 / 187 / 11 |
| document__e_s__none | document__e_s__laya_text | hit@5 | 0.9674 | 0.5442 | -0.4233 | Not reported; see reason below | 1 / 92 / 122 |
| document__e_s__none | document__e_s__laya_text | recall@5 | 0.6112 | 0.2031 | -0.4082 | Not reported; see reason below | 4 / 178 / 33 |
| document__e_s__none | document__e_s__laya_text | ndcg@10 | 0.7399 | 0.2780 | -0.4619 | Not reported; see reason below | 7 / 197 / 11 |
| document__e_j__none | document__e_j__laya_text | hit@5 | 0.9628 | 0.5349 | -0.4279 | Not reported; see reason below | 0 / 92 / 123 |
| document__e_j__none | document__e_j__laya_text | recall@5 | 0.5497 | 0.1987 | -0.3510 | Not reported; see reason below | 6 / 162 / 47 |
| document__e_j__none | document__e_j__laya_text | ndcg@10 | 0.6722 | 0.2832 | -0.3890 | Not reported; see reason below | 16 / 188 / 11 |
| document__n_s__none | document__n_s__laya_text | hit@5 | 0.9674 | 0.5581 | -0.4093 | Not reported; see reason below | 2 / 90 / 123 |
| document__n_s__none | document__n_s__laya_text | recall@5 | 0.6163 | 0.2070 | -0.4093 | Not reported; see reason below | 6 / 171 / 38 |
| document__n_s__none | document__n_s__laya_text | ndcg@10 | 0.7264 | 0.2718 | -0.4546 | Not reported; see reason below | 4 / 203 / 8 |
| document__n_j__none | document__n_j__laya_text | hit@5 | 0.9302 | 0.5535 | -0.3767 | Not reported; see reason below | 5 / 86 / 124 |
| document__n_j__none | document__n_j__laya_text | recall@5 | 0.5429 | 0.2063 | -0.3367 | Not reported; see reason below | 11 / 153 / 51 |
| document__n_j__none | document__n_j__laya_text | ndcg@10 | 0.6629 | 0.2774 | -0.3855 | Not reported; see reason below | 16 / 189 / 10 |
| document__s_j__none | document__s_j__laya_text | hit@5 | 0.9721 | 0.5256 | -0.4465 | Not reported; see reason below | 0 / 96 / 119 |
| document__s_j__none | document__s_j__laya_text | recall@5 | 0.6215 | 0.1927 | -0.4287 | Not reported; see reason below | 1 / 176 / 38 |
| document__s_j__none | document__s_j__laya_text | ndcg@10 | 0.7409 | 0.2657 | -0.4752 | Not reported; see reason below | 7 / 197 / 11 |
| document__b_g_e__none | document__b_g_e__laya_text | hit@5 | 0.9674 | 0.5581 | -0.4093 | Not reported; see reason below | 0 / 88 / 127 |
| document__b_g_e__none | document__b_g_e__laya_text | recall@5 | 0.5941 | 0.1996 | -0.3945 | Not reported; see reason below | 3 / 169 / 43 |
| document__b_g_e__none | document__b_g_e__laya_text | ndcg@10 | 0.7005 | 0.2722 | -0.4283 | Not reported; see reason below | 12 / 190 / 13 |
| document__b_g_n__none | document__b_g_n__laya_text | hit@5 | 0.9767 | 0.5395 | -0.4372 | Not reported; see reason below | 0 / 94 / 121 |
| document__b_g_n__none | document__b_g_n__laya_text | recall@5 | 0.5970 | 0.1926 | -0.4044 | Not reported; see reason below | 3 / 172 / 40 |
| document__b_g_n__none | document__b_g_n__laya_text | ndcg@10 | 0.6988 | 0.2640 | -0.4348 | Not reported; see reason below | 12 / 193 / 10 |
| document__b_g_s__none | document__b_g_s__laya_text | hit@5 | 0.9767 | 0.5674 | -0.4093 | Not reported; see reason below | 0 / 88 / 127 |
| document__b_g_s__none | document__b_g_s__laya_text | recall@5 | 0.6279 | 0.2070 | -0.4209 | Not reported; see reason below | 4 / 176 / 35 |
| document__b_g_s__none | document__b_g_s__laya_text | ndcg@10 | 0.7349 | 0.2764 | -0.4584 | Not reported; see reason below | 6 / 198 / 11 |
| document__b_g_j__none | document__b_g_j__laya_text | hit@5 | 0.9674 | 0.5442 | -0.4233 | Not reported; see reason below | 1 / 92 / 122 |
| document__b_g_j__none | document__b_g_j__laya_text | recall@5 | 0.5981 | 0.1952 | -0.4029 | Not reported; see reason below | 3 / 170 / 42 |
| document__b_g_j__none | document__b_g_j__laya_text | ndcg@10 | 0.7057 | 0.2681 | -0.4376 | Not reported; see reason below | 10 / 194 / 11 |
| document__b_e_n__none | document__b_e_n__laya_text | hit@5 | 0.9628 | 0.5628 | -0.4000 | Not reported; see reason below | 2 / 88 / 125 |
| document__b_e_n__none | document__b_e_n__laya_text | recall@5 | 0.5884 | 0.2023 | -0.3861 | Not reported; see reason below | 4 / 167 / 44 |
| document__b_e_n__none | document__b_e_n__laya_text | ndcg@10 | 0.6995 | 0.2770 | -0.4225 | Not reported; see reason below | 11 / 193 / 11 |
| document__b_e_s__none | document__b_e_s__laya_text | hit@5 | 0.9721 | 0.5535 | -0.4186 | Not reported; see reason below | 1 / 91 / 123 |
| document__b_e_s__none | document__b_e_s__laya_text | recall@5 | 0.6152 | 0.2033 | -0.4119 | Not reported; see reason below | 5 / 174 / 36 |
| document__b_e_s__none | document__b_e_s__laya_text | ndcg@10 | 0.7312 | 0.2763 | -0.4548 | Not reported; see reason below | 6 / 196 / 13 |
| document__b_e_j__none | document__b_e_j__laya_text | hit@5 | 0.9628 | 0.5628 | -0.4000 | Not reported; see reason below | 2 / 88 / 125 |
| document__b_e_j__none | document__b_e_j__laya_text | recall@5 | 0.5894 | 0.2035 | -0.3859 | Not reported; see reason below | 6 / 168 / 41 |
| document__b_e_j__none | document__b_e_j__laya_text | ndcg@10 | 0.7042 | 0.2792 | -0.4250 | Not reported; see reason below | 10 / 193 / 12 |
| document__b_n_s__none | document__b_n_s__laya_text | hit@5 | 0.9814 | 0.5674 | -0.4140 | Not reported; see reason below | 0 / 89 / 126 |
| document__b_n_s__none | document__b_n_s__laya_text | recall@5 | 0.6258 | 0.2054 | -0.4204 | Not reported; see reason below | 4 / 175 / 36 |
| document__b_n_s__none | document__b_n_s__laya_text | ndcg@10 | 0.7301 | 0.2692 | -0.4609 | Not reported; see reason below | 7 / 200 / 8 |
| document__b_n_j__none | document__b_n_j__laya_text | hit@5 | 0.9581 | 0.5442 | -0.4140 | Not reported; see reason below | 3 / 92 / 120 |
| document__b_n_j__none | document__b_n_j__laya_text | recall@5 | 0.5888 | 0.1982 | -0.3906 | Not reported; see reason below | 5 / 166 / 44 |
| document__b_n_j__none | document__b_n_j__laya_text | ndcg@10 | 0.6999 | 0.2706 | -0.4294 | Not reported; see reason below | 14 / 190 / 11 |
| document__b_s_j__none | document__b_s_j__laya_text | hit@5 | 0.9721 | 0.5535 | -0.4186 | Not reported; see reason below | 1 / 91 / 123 |
| document__b_s_j__none | document__b_s_j__laya_text | recall@5 | 0.6244 | 0.2004 | -0.4240 | Not reported; see reason below | 4 / 176 / 35 |
| document__b_s_j__none | document__b_s_j__laya_text | ndcg@10 | 0.7440 | 0.2748 | -0.4692 | Not reported; see reason below | 5 / 198 / 12 |
| document__g_e_n__none | document__g_e_n__laya_text | hit@5 | 0.9628 | 0.5674 | -0.3953 | Not reported; see reason below | 0 / 85 / 130 |
| document__g_e_n__none | document__g_e_n__laya_text | recall@5 | 0.5792 | 0.2074 | -0.3718 | Not reported; see reason below | 6 / 162 / 47 |
| document__g_e_n__none | document__g_e_n__laya_text | ndcg@10 | 0.6987 | 0.2833 | -0.4154 | Not reported; see reason below | 9 / 193 / 13 |
| document__g_e_s__none | document__g_e_s__laya_text | hit@5 | 0.9581 | 0.5628 | -0.3953 | Not reported; see reason below | 1 / 86 / 128 |
| document__g_e_s__none | document__g_e_s__laya_text | recall@5 | 0.6049 | 0.2073 | -0.3976 | Not reported; see reason below | 2 / 172 / 41 |
| document__g_e_s__none | document__g_e_s__laya_text | ndcg@10 | 0.7276 | 0.2807 | -0.4469 | Not reported; see reason below | 3 / 200 / 12 |
| document__g_e_j__none | document__g_e_j__laya_text | hit@5 | 0.9628 | 0.5767 | -0.3860 | Not reported; see reason below | 1 / 84 / 130 |
| document__g_e_j__none | document__g_e_j__laya_text | recall@5 | 0.5771 | 0.2096 | -0.3675 | Not reported; see reason below | 6 / 167 / 42 |
| document__g_e_j__none | document__g_e_j__laya_text | ndcg@10 | 0.6970 | 0.2820 | -0.4150 | Not reported; see reason below | 11 / 193 / 11 |
| document__g_n_s__none | document__g_n_s__laya_text | hit@5 | 0.9721 | 0.5535 | -0.4186 | Not reported; see reason below | 0 / 90 / 125 |
| document__g_n_s__none | document__g_n_s__laya_text | recall@5 | 0.6167 | 0.2037 | -0.4130 | Not reported; see reason below | 1 / 175 / 39 |
| document__g_n_s__none | document__g_n_s__laya_text | ndcg@10 | 0.7327 | 0.2693 | -0.4635 | Not reported; see reason below | 6 / 201 / 8 |
| document__g_n_j__none | document__g_n_j__laya_text | hit@5 | 0.9535 | 0.5535 | -0.4000 | Not reported; see reason below | 1 / 87 / 127 |
| document__g_n_j__none | document__g_n_j__laya_text | recall@5 | 0.5746 | 0.2087 | -0.3659 | Not reported; see reason below | 6 / 164 / 45 |
| document__g_n_j__none | document__g_n_j__laya_text | ndcg@10 | 0.6886 | 0.2758 | -0.4128 | Not reported; see reason below | 11 / 191 / 13 |
| document__g_s_j__none | document__g_s_j__laya_text | hit@5 | 0.9581 | 0.5442 | -0.4140 | Not reported; see reason below | 1 / 90 / 124 |
| document__g_s_j__none | document__g_s_j__laya_text | recall@5 | 0.6123 | 0.1997 | -0.4126 | Not reported; see reason below | 2 / 174 / 39 |
| document__g_s_j__none | document__g_s_j__laya_text | ndcg@10 | 0.7303 | 0.2705 | -0.4598 | Not reported; see reason below | 6 / 199 / 10 |
| document__e_n_s__none | document__e_n_s__laya_text | hit@5 | 0.9628 | 0.5535 | -0.4093 | Not reported; see reason below | 2 / 90 / 123 |
| document__e_n_s__none | document__e_n_s__laya_text | recall@5 | 0.6023 | 0.2040 | -0.3983 | Not reported; see reason below | 7 / 171 / 37 |
| document__e_n_s__none | document__e_n_s__laya_text | ndcg@10 | 0.7330 | 0.2722 | -0.4608 | Not reported; see reason below | 8 / 197 / 10 |
| document__e_n_j__none | document__e_n_j__laya_text | hit@5 | 0.9628 | 0.5488 | -0.4140 | Not reported; see reason below | 1 / 90 / 124 |
| document__e_n_j__none | document__e_n_j__laya_text | recall@5 | 0.5652 | 0.2048 | -0.3604 | Not reported; see reason below | 5 / 160 / 50 |
| document__e_n_j__none | document__e_n_j__laya_text | ndcg@10 | 0.6764 | 0.2810 | -0.3954 | Not reported; see reason below | 11 / 194 / 10 |
| document__e_s_j__none | document__e_s_j__laya_text | hit@5 | 0.9674 | 0.5349 | -0.4326 | Not reported; see reason below | 0 / 93 / 122 |
| document__e_s_j__none | document__e_s_j__laya_text | recall@5 | 0.5981 | 0.1999 | -0.3982 | Not reported; see reason below | 3 / 172 / 40 |
| document__e_s_j__none | document__e_s_j__laya_text | ndcg@10 | 0.7327 | 0.2775 | -0.4552 | Not reported; see reason below | 6 / 198 / 11 |
| document__n_s_j__none | document__n_s_j__laya_text | hit@5 | 0.9721 | 0.5488 | -0.4233 | Not reported; see reason below | 1 / 92 / 122 |
| document__n_s_j__none | document__n_s_j__laya_text | recall@5 | 0.6113 | 0.2024 | -0.4089 | Not reported; see reason below | 5 / 172 / 38 |
| document__n_s_j__none | document__n_s_j__laya_text | ndcg@10 | 0.7258 | 0.2704 | -0.4554 | Not reported; see reason below | 8 / 197 / 10 |
| document__b_g_e_n__none | document__b_g_e_n__laya_text | hit@5 | 0.9721 | 0.5488 | -0.4233 | Not reported; see reason below | 0 / 91 / 124 |
| document__b_g_e_n__none | document__b_g_e_n__laya_text | recall@5 | 0.6003 | 0.2005 | -0.3998 | Not reported; see reason below | 4 / 167 / 44 |
| document__b_g_e_n__none | document__b_g_e_n__laya_text | ndcg@10 | 0.7063 | 0.2731 | -0.4332 | Not reported; see reason below | 12 / 192 / 11 |
| document__b_g_e_s__none | document__b_g_e_s__laya_text | hit@5 | 0.9767 | 0.5442 | -0.4326 | Not reported; see reason below | 0 / 93 / 122 |
| document__b_g_e_s__none | document__b_g_e_s__laya_text | recall@5 | 0.6260 | 0.2016 | -0.4244 | Not reported; see reason below | 4 / 181 / 30 |
| document__b_g_e_s__none | document__b_g_e_s__laya_text | ndcg@10 | 0.7357 | 0.2770 | -0.4587 | Not reported; see reason below | 6 / 198 / 11 |
| document__b_g_e_j__none | document__b_g_e_j__laya_text | hit@5 | 0.9674 | 0.5535 | -0.4140 | Not reported; see reason below | 0 / 89 / 126 |
| document__b_g_e_j__none | document__b_g_e_j__laya_text | recall@5 | 0.5967 | 0.2030 | -0.3937 | Not reported; see reason below | 5 / 169 / 41 |
| document__b_g_e_j__none | document__b_g_e_j__laya_text | ndcg@10 | 0.7105 | 0.2761 | -0.4344 | Not reported; see reason below | 8 / 195 / 12 |
| document__b_g_n_s__none | document__b_g_n_s__laya_text | hit@5 | 0.9767 | 0.5488 | -0.4279 | Not reported; see reason below | 0 / 92 / 123 |
| document__b_g_n_s__none | document__b_g_n_s__laya_text | recall@5 | 0.6212 | 0.2043 | -0.4169 | Not reported; see reason below | 3 / 176 / 36 |
| document__b_g_n_s__none | document__b_g_n_s__laya_text | ndcg@10 | 0.7304 | 0.2698 | -0.4607 | Not reported; see reason below | 10 / 196 / 9 |
| document__b_g_n_j__none | document__b_g_n_j__laya_text | hit@5 | 0.9721 | 0.5442 | -0.4279 | Not reported; see reason below | 0 / 92 / 123 |
| document__b_g_n_j__none | document__b_g_n_j__laya_text | recall@5 | 0.6016 | 0.1980 | -0.4036 | Not reported; see reason below | 2 / 168 / 45 |
| document__b_g_n_j__none | document__b_g_n_j__laya_text | ndcg@10 | 0.7110 | 0.2742 | -0.4368 | Not reported; see reason below | 10 / 195 / 10 |
| document__b_g_s_j__none | document__b_g_s_j__laya_text | hit@5 | 0.9721 | 0.5535 | -0.4186 | Not reported; see reason below | 0 / 90 / 125 |
| document__b_g_s_j__none | document__b_g_s_j__laya_text | recall@5 | 0.6241 | 0.2041 | -0.4200 | Not reported; see reason below | 3 / 179 / 33 |
| document__b_g_s_j__none | document__b_g_s_j__laya_text | ndcg@10 | 0.7371 | 0.2736 | -0.4634 | Not reported; see reason below | 6 / 197 / 12 |
| document__b_e_n_s__none | document__b_e_n_s__laya_text | hit@5 | 0.9767 | 0.5581 | -0.4186 | Not reported; see reason below | 0 / 90 / 125 |
| document__b_e_n_s__none | document__b_e_n_s__laya_text | recall@5 | 0.6201 | 0.2061 | -0.4140 | Not reported; see reason below | 4 / 174 / 37 |
| document__b_e_n_s__none | document__b_e_n_s__laya_text | ndcg@10 | 0.7332 | 0.2772 | -0.4560 | Not reported; see reason below | 5 / 200 / 10 |
| document__b_e_n_j__none | document__b_e_n_j__laya_text | hit@5 | 0.9628 | 0.5535 | -0.4093 | Not reported; see reason below | 2 / 90 / 123 |
| document__b_e_n_j__none | document__b_e_n_j__laya_text | recall@5 | 0.5933 | 0.2016 | -0.3917 | Not reported; see reason below | 5 / 169 / 41 |
| document__b_e_n_j__none | document__b_e_n_j__laya_text | ndcg@10 | 0.7095 | 0.2758 | -0.4337 | Not reported; see reason below | 8 / 195 / 12 |
| document__b_e_s_j__none | document__b_e_s_j__laya_text | hit@5 | 0.9767 | 0.5442 | -0.4326 | Not reported; see reason below | 0 / 93 / 122 |
| document__b_e_s_j__none | document__b_e_s_j__laya_text | recall@5 | 0.6145 | 0.1994 | -0.4150 | Not reported; see reason below | 4 / 176 / 35 |
| document__b_e_s_j__none | document__b_e_s_j__laya_text | ndcg@10 | 0.7374 | 0.2820 | -0.4554 | Not reported; see reason below | 4 / 200 / 11 |
| document__b_n_s_j__none | document__b_n_s_j__laya_text | hit@5 | 0.9628 | 0.5535 | -0.4093 | Not reported; see reason below | 2 / 90 / 123 |
| document__b_n_s_j__none | document__b_n_s_j__laya_text | recall@5 | 0.6113 | 0.2026 | -0.4086 | Not reported; see reason below | 4 / 170 / 41 |
| document__b_n_s_j__none | document__b_n_s_j__laya_text | ndcg@10 | 0.7298 | 0.2729 | -0.4569 | Not reported; see reason below | 6 / 199 / 10 |
| document__g_e_n_s__none | document__g_e_n_s__laya_text | hit@5 | 0.9721 | 0.5442 | -0.4279 | Not reported; see reason below | 0 / 92 / 123 |
| document__g_e_n_s__none | document__g_e_n_s__laya_text | recall@5 | 0.6181 | 0.2031 | -0.4150 | Not reported; see reason below | 4 / 173 / 38 |
| document__g_e_n_s__none | document__g_e_n_s__laya_text | ndcg@10 | 0.7267 | 0.2745 | -0.4522 | Not reported; see reason below | 6 / 198 / 11 |
| document__g_e_n_j__none | document__g_e_n_j__laya_text | hit@5 | 0.9581 | 0.5721 | -0.3860 | Not reported; see reason below | 0 / 83 / 132 |
| document__g_e_n_j__none | document__g_e_n_j__laya_text | recall@5 | 0.5803 | 0.2042 | -0.3761 | Not reported; see reason below | 5 / 166 / 44 |
| document__g_e_n_j__none | document__g_e_n_j__laya_text | ndcg@10 | 0.6928 | 0.2815 | -0.4113 | Not reported; see reason below | 10 / 191 / 14 |
| document__g_e_s_j__none | document__g_e_s_j__laya_text | hit@5 | 0.9581 | 0.5581 | -0.4000 | Not reported; see reason below | 1 / 87 / 127 |
| document__g_e_s_j__none | document__g_e_s_j__laya_text | recall@5 | 0.6101 | 0.2072 | -0.4030 | Not reported; see reason below | 3 / 169 / 43 |
| document__g_e_s_j__none | document__g_e_s_j__laya_text | ndcg@10 | 0.7273 | 0.2763 | -0.4511 | Not reported; see reason below | 6 / 196 / 13 |
| document__g_n_s_j__none | document__g_n_s_j__laya_text | hit@5 | 0.9721 | 0.5442 | -0.4279 | Not reported; see reason below | 0 / 92 / 123 |
| document__g_n_s_j__none | document__g_n_s_j__laya_text | recall@5 | 0.6194 | 0.2036 | -0.4158 | Not reported; see reason below | 2 / 170 / 43 |
| document__g_n_s_j__none | document__g_n_s_j__laya_text | ndcg@10 | 0.7268 | 0.2761 | -0.4507 | Not reported; see reason below | 8 / 196 / 11 |
| document__e_n_s_j__none | document__e_n_s_j__laya_text | hit@5 | 0.9767 | 0.5628 | -0.4140 | Not reported; see reason below | 0 / 89 / 126 |
| document__e_n_s_j__none | document__e_n_s_j__laya_text | recall@5 | 0.5973 | 0.2067 | -0.3906 | Not reported; see reason below | 4 / 169 / 42 |
| document__e_n_s_j__none | document__e_n_s_j__laya_text | ndcg@10 | 0.7258 | 0.2751 | -0.4507 | Not reported; see reason below | 8 / 196 / 11 |
| document__b_g_e_n_s__none | document__b_g_e_n_s__laya_text | hit@5 | 0.9674 | 0.5488 | -0.4186 | Not reported; see reason below | 0 / 90 / 125 |
| document__b_g_e_n_s__none | document__b_g_e_n_s__laya_text | recall@5 | 0.6199 | 0.2026 | -0.4173 | Not reported; see reason below | 2 / 177 / 36 |
| document__b_g_e_n_s__none | document__b_g_e_n_s__laya_text | ndcg@10 | 0.7301 | 0.2712 | -0.4589 | Not reported; see reason below | 8 / 196 / 11 |
| document__b_g_e_n_j__none | document__b_g_e_n_j__laya_text | hit@5 | 0.9674 | 0.5488 | -0.4186 | Not reported; see reason below | 0 / 90 / 125 |
| document__b_g_e_n_j__none | document__b_g_e_n_j__laya_text | recall@5 | 0.5931 | 0.2030 | -0.3900 | Not reported; see reason below | 5 / 166 / 44 |
| document__b_g_e_n_j__none | document__b_g_e_n_j__laya_text | ndcg@10 | 0.7132 | 0.2775 | -0.4357 | Not reported; see reason below | 11 / 191 / 13 |
| document__b_g_e_s_j__none | document__b_g_e_s_j__laya_text | hit@5 | 0.9721 | 0.5581 | -0.4140 | Not reported; see reason below | 0 / 89 / 126 |
| document__b_g_e_s_j__none | document__b_g_e_s_j__laya_text | recall@5 | 0.6212 | 0.2078 | -0.4134 | Not reported; see reason below | 4 / 178 / 33 |
| document__b_g_e_s_j__none | document__b_g_e_s_j__laya_text | ndcg@10 | 0.7314 | 0.2784 | -0.4530 | Not reported; see reason below | 7 / 196 / 12 |
| document__b_g_n_s_j__none | document__b_g_n_s_j__laya_text | hit@5 | 0.9721 | 0.5442 | -0.4279 | Not reported; see reason below | 0 / 92 / 123 |
| document__b_g_n_s_j__none | document__b_g_n_s_j__laya_text | recall@5 | 0.6196 | 0.2031 | -0.4165 | Not reported; see reason below | 3 / 175 / 37 |
| document__b_g_n_s_j__none | document__b_g_n_s_j__laya_text | ndcg@10 | 0.7339 | 0.2714 | -0.4624 | Not reported; see reason below | 10 / 194 / 11 |
| document__b_e_n_s_j__none | document__b_e_n_s_j__laya_text | hit@5 | 0.9674 | 0.5535 | -0.4140 | Not reported; see reason below | 1 / 90 / 124 |
| document__b_e_n_s_j__none | document__b_e_n_s_j__laya_text | recall@5 | 0.6149 | 0.2060 | -0.4089 | Not reported; see reason below | 4 / 172 / 39 |
| document__b_e_n_s_j__none | document__b_e_n_s_j__laya_text | ndcg@10 | 0.7349 | 0.2761 | -0.4588 | Not reported; see reason below | 6 / 199 / 10 |
| document__g_e_n_s_j__none | document__g_e_n_s_j__laya_text | hit@5 | 0.9721 | 0.5535 | -0.4186 | Not reported; see reason below | 0 / 90 / 125 |
| document__g_e_n_s_j__none | document__g_e_n_s_j__laya_text | recall@5 | 0.6150 | 0.2063 | -0.4087 | Not reported; see reason below | 4 / 171 / 40 |
| document__g_e_n_s_j__none | document__g_e_n_s_j__laya_text | ndcg@10 | 0.7240 | 0.2731 | -0.4509 | Not reported; see reason below | 6 / 197 / 12 |
| document__b_g_e_n_s_j__none | document__b_g_e_n_s_j__laya_text | hit@5 | 0.9674 | 0.5488 | -0.4186 | Not reported; see reason below | 0 / 90 / 125 |
| document__b_g_e_n_s_j__none | document__b_g_e_n_s_j__laya_text | recall@5 | 0.6124 | 0.2033 | -0.4091 | Not reported; see reason below | 4 / 172 / 39 |
| document__b_g_e_n_s_j__none | document__b_g_e_n_s_j__laya_text | ndcg@10 | 0.7351 | 0.2774 | -0.4578 | Not reported; see reason below | 7 / 197 / 11 |

document__b__none → document__b__laya_text: Only one connected source group; a resampling confidence interval is not reported.

document__g__none → document__g__laya_text: Only one connected source group; a resampling confidence interval is not reported.

document__e__none → document__e__laya_text: Only one connected source group; a resampling confidence interval is not reported.

document__n__none → document__n__laya_text: Only one connected source group; a resampling confidence interval is not reported.

document__s__none → document__s__laya_text: Only one connected source group; a resampling confidence interval is not reported.

document__j__none → document__j__laya_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_g__none → document__b_g__laya_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_e__none → document__b_e__laya_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_n__none → document__b_n__laya_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_s__none → document__b_s__laya_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_j__none → document__b_j__laya_text: Only one connected source group; a resampling confidence interval is not reported.

document__g_e__none → document__g_e__laya_text: Only one connected source group; a resampling confidence interval is not reported.

document__g_n__none → document__g_n__laya_text: Only one connected source group; a resampling confidence interval is not reported.

document__g_s__none → document__g_s__laya_text: Only one connected source group; a resampling confidence interval is not reported.

document__g_j__none → document__g_j__laya_text: Only one connected source group; a resampling confidence interval is not reported.

document__e_n__none → document__e_n__laya_text: Only one connected source group; a resampling confidence interval is not reported.

document__e_s__none → document__e_s__laya_text: Only one connected source group; a resampling confidence interval is not reported.

document__e_j__none → document__e_j__laya_text: Only one connected source group; a resampling confidence interval is not reported.

document__n_s__none → document__n_s__laya_text: Only one connected source group; a resampling confidence interval is not reported.

document__n_j__none → document__n_j__laya_text: Only one connected source group; a resampling confidence interval is not reported.

document__s_j__none → document__s_j__laya_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_g_e__none → document__b_g_e__laya_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_g_n__none → document__b_g_n__laya_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_g_s__none → document__b_g_s__laya_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_g_j__none → document__b_g_j__laya_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_e_n__none → document__b_e_n__laya_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_e_s__none → document__b_e_s__laya_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_e_j__none → document__b_e_j__laya_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_n_s__none → document__b_n_s__laya_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_n_j__none → document__b_n_j__laya_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_s_j__none → document__b_s_j__laya_text: Only one connected source group; a resampling confidence interval is not reported.

document__g_e_n__none → document__g_e_n__laya_text: Only one connected source group; a resampling confidence interval is not reported.

document__g_e_s__none → document__g_e_s__laya_text: Only one connected source group; a resampling confidence interval is not reported.

document__g_e_j__none → document__g_e_j__laya_text: Only one connected source group; a resampling confidence interval is not reported.

document__g_n_s__none → document__g_n_s__laya_text: Only one connected source group; a resampling confidence interval is not reported.

document__g_n_j__none → document__g_n_j__laya_text: Only one connected source group; a resampling confidence interval is not reported.

document__g_s_j__none → document__g_s_j__laya_text: Only one connected source group; a resampling confidence interval is not reported.

document__e_n_s__none → document__e_n_s__laya_text: Only one connected source group; a resampling confidence interval is not reported.

document__e_n_j__none → document__e_n_j__laya_text: Only one connected source group; a resampling confidence interval is not reported.

document__e_s_j__none → document__e_s_j__laya_text: Only one connected source group; a resampling confidence interval is not reported.

document__n_s_j__none → document__n_s_j__laya_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_g_e_n__none → document__b_g_e_n__laya_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_g_e_s__none → document__b_g_e_s__laya_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_g_e_j__none → document__b_g_e_j__laya_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_g_n_s__none → document__b_g_n_s__laya_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_g_n_j__none → document__b_g_n_j__laya_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_g_s_j__none → document__b_g_s_j__laya_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_e_n_s__none → document__b_e_n_s__laya_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_e_n_j__none → document__b_e_n_j__laya_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_e_s_j__none → document__b_e_s_j__laya_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_n_s_j__none → document__b_n_s_j__laya_text: Only one connected source group; a resampling confidence interval is not reported.

document__g_e_n_s__none → document__g_e_n_s__laya_text: Only one connected source group; a resampling confidence interval is not reported.

document__g_e_n_j__none → document__g_e_n_j__laya_text: Only one connected source group; a resampling confidence interval is not reported.

document__g_e_s_j__none → document__g_e_s_j__laya_text: Only one connected source group; a resampling confidence interval is not reported.

document__g_n_s_j__none → document__g_n_s_j__laya_text: Only one connected source group; a resampling confidence interval is not reported.

document__e_n_s_j__none → document__e_n_s_j__laya_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_g_e_n_s__none → document__b_g_e_n_s__laya_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_g_e_n_j__none → document__b_g_e_n_j__laya_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_g_e_s_j__none → document__b_g_e_s_j__laya_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_g_n_s_j__none → document__b_g_n_s_j__laya_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_e_n_s_j__none → document__b_e_n_s_j__laya_text: Only one connected source group; a resampling confidence interval is not reported.

document__g_e_n_s_j__none → document__g_e_n_s_j__laya_text: Only one connected source group; a resampling confidence interval is not reported.

document__b_g_e_n_s_j__none → document__b_g_e_n_s_j__laya_text: Only one connected source group; a resampling confidence interval is not reported.

Bootstrap settings for eligible comparisons: 5000 samples; seed: 42. No model inference or generated-answer judging was performed.

Pairing follows [SciPy's paired resampling definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html); source groups follow [cluster resampling](https://www.stata.com/support/faqs/statistics/bootstrap-with-panel-data/).

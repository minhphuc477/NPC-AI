# Proposal Alignment Evaluation Report

- Run ID: `20260305T134411Z`
- Generated: `2026-03-05T13:48:05.826681+00:00`
- Scenarios: `artifacts\proposal_control_tuning\tuning_matrix\m1_l2\20260305T134411Z\scenarios.jsonl`
- Scenario count: `20`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off
- `baseline_no_context`: model `phi3:mini`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2850 (0.2332, 0.3415) | 0.3497 (0.2742, 0.4396) | 0.8374 (0.7958, 0.8766) | 0.3838 (0.3531, 0.4176) | 0.0856 |
| proposed_contextual | 0.0758 (0.0310, 0.1314) | 0.1292 (0.0939, 0.1682) | 0.7744 (0.7583, 0.7929) | 0.2132 (0.1868, 0.2436) | 0.0648 |
| candidate_no_context | 0.0287 (0.0155, 0.0458) | 0.1716 (0.1075, 0.2467) | 0.8180 (0.7834, 0.8557) | 0.2132 (0.1842, 0.2475) | 0.0418 |
| baseline_no_context | 0.0349 (0.0200, 0.0526) | 0.1956 (0.1530, 0.2401) | 0.9000 (0.8719, 0.9270) | 0.2391 (0.2207, 0.2580) | 0.0563 |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0471 | 1.6404 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0424 | -0.2469 |
| proposed_vs_candidate_no_context | naturalness | -0.0436 | -0.0533 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0625 | 2.7500 |
| proposed_vs_candidate_no_context | context_overlap | 0.0112 | 0.2623 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0524 | -0.6286 |
| proposed_vs_candidate_no_context | persona_style | -0.0023 | -0.0044 |
| proposed_vs_candidate_no_context | distinct1 | -0.0032 | -0.0035 |
| proposed_vs_candidate_no_context | length_score | -0.2117 | -0.5935 |
| proposed_vs_candidate_no_context | sentence_score | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0230 | 0.5511 |
| proposed_vs_candidate_no_context | overall_quality | -0.0000 | -0.0001 |
| proposed_vs_baseline_no_context | context_relevance | 0.0410 | 1.1748 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0663 | -0.3392 |
| proposed_vs_baseline_no_context | naturalness | -0.1256 | -0.1396 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0542 | 1.7439 |
| proposed_vs_baseline_no_context | context_overlap | 0.0101 | 0.2319 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0614 | -0.6649 |
| proposed_vs_baseline_no_context | persona_style | -0.0859 | -0.1413 |
| proposed_vs_baseline_no_context | distinct1 | -0.0436 | -0.0446 |
| proposed_vs_baseline_no_context | length_score | -0.4533 | -0.7577 |
| proposed_vs_baseline_no_context | sentence_score | -0.1750 | -0.1955 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0085 | 0.1513 |
| proposed_vs_baseline_no_context | overall_quality | -0.0259 | -0.1085 |
| controlled_vs_proposed_raw | context_relevance | 0.2092 | 2.7590 |
| controlled_vs_proposed_raw | persona_consistency | 0.2205 | 1.7061 |
| controlled_vs_proposed_raw | naturalness | 0.0630 | 0.0813 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2718 | 3.1893 |
| controlled_vs_proposed_raw | context_overlap | 0.0631 | 1.1713 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2548 | 8.2308 |
| controlled_vs_proposed_raw | persona_style | 0.0834 | 0.1597 |
| controlled_vs_proposed_raw | distinct1 | -0.0143 | -0.0153 |
| controlled_vs_proposed_raw | length_score | 0.2583 | 1.7816 |
| controlled_vs_proposed_raw | sentence_score | 0.2100 | 0.2917 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0208 | 0.3208 |
| controlled_vs_proposed_raw | overall_quality | 0.1706 | 0.8003 |
| controlled_vs_candidate_no_context | context_relevance | 0.2563 | 8.9255 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1781 | 1.0381 |
| controlled_vs_candidate_no_context | naturalness | 0.0194 | 0.0237 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3343 | 14.7100 |
| controlled_vs_candidate_no_context | context_overlap | 0.0743 | 1.7410 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2024 | 2.4286 |
| controlled_vs_candidate_no_context | persona_style | 0.0811 | 0.1547 |
| controlled_vs_candidate_no_context | distinct1 | -0.0175 | -0.0187 |
| controlled_vs_candidate_no_context | length_score | 0.0467 | 0.1308 |
| controlled_vs_candidate_no_context | sentence_score | 0.2100 | 0.2917 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0438 | 1.0487 |
| controlled_vs_candidate_no_context | overall_quality | 0.1706 | 0.8002 |
| controlled_vs_baseline_no_context | context_relevance | 0.2502 | 7.1749 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1542 | 0.7883 |
| controlled_vs_baseline_no_context | naturalness | -0.0626 | -0.0696 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3260 | 10.4951 |
| controlled_vs_baseline_no_context | context_overlap | 0.0733 | 1.6750 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1933 | 2.0928 |
| controlled_vs_baseline_no_context | persona_style | -0.0025 | -0.0041 |
| controlled_vs_baseline_no_context | distinct1 | -0.0579 | -0.0592 |
| controlled_vs_baseline_no_context | length_score | -0.1950 | -0.3259 |
| controlled_vs_baseline_no_context | sentence_score | 0.0350 | 0.0391 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0293 | 0.5207 |
| controlled_vs_baseline_no_context | overall_quality | 0.1447 | 0.6050 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2502 | 7.1749 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1542 | 0.7883 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0626 | -0.0696 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3260 | 10.4951 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0733 | 1.6750 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1933 | 2.0928 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | -0.0025 | -0.0041 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0579 | -0.0592 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | -0.1950 | -0.3259 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0350 | 0.0391 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0293 | 0.5207 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1447 | 0.6050 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0471 | (0.0030, 0.1011) | 0.0180 | 0.0471 | (-0.0097, 0.1361) | 0.0847 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0424 | (-0.1034, 0.0147) | 0.9207 | -0.0424 | (-0.1241, 0.0074) | 0.9417 |
| proposed_vs_candidate_no_context | naturalness | -0.0436 | (-0.0760, -0.0146) | 0.9997 | -0.0436 | (-0.0932, -0.0112) | 0.9987 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0625 | (0.0030, 0.1352) | 0.0247 | 0.0625 | (-0.0145, 0.1806) | 0.1080 |
| proposed_vs_candidate_no_context | context_overlap | 0.0112 | (-0.0000, 0.0243) | 0.0253 | 0.0112 | (0.0015, 0.0243) | 0.0000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0524 | (-0.1310, 0.0179) | 0.9303 | -0.0524 | (-0.1524, 0.0064) | 0.9577 |
| proposed_vs_candidate_no_context | persona_style | -0.0023 | (-0.0356, 0.0269) | 0.5613 | -0.0023 | (-0.0417, 0.0263) | 0.5670 |
| proposed_vs_candidate_no_context | distinct1 | -0.0032 | (-0.0199, 0.0144) | 0.6500 | -0.0032 | (-0.0263, 0.0156) | 0.6253 |
| proposed_vs_candidate_no_context | length_score | -0.2117 | (-0.3467, -0.0916) | 1.0000 | -0.2117 | (-0.4157, -0.0639) | 1.0000 |
| proposed_vs_candidate_no_context | sentence_score | 0.0000 | (-0.0875, 0.0875) | 0.5813 | 0.0000 | (-0.0618, 0.0457) | 0.6383 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0230 | (-0.0011, 0.0523) | 0.0370 | 0.0230 | (-0.0014, 0.0727) | 0.0620 |
| proposed_vs_candidate_no_context | overall_quality | -0.0000 | (-0.0342, 0.0363) | 0.5267 | -0.0000 | (-0.0504, 0.0449) | 0.5357 |
| proposed_vs_baseline_no_context | context_relevance | 0.0410 | (-0.0073, 0.0975) | 0.0537 | 0.0410 | (-0.0258, 0.1398) | 0.1917 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0663 | (-0.1212, -0.0066) | 0.9853 | -0.0663 | (-0.1469, 0.0095) | 0.9577 |
| proposed_vs_baseline_no_context | naturalness | -0.1256 | (-0.1601, -0.0865) | 1.0000 | -0.1256 | (-0.1662, -0.0739) | 1.0000 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0542 | (-0.0099, 0.1314) | 0.0703 | 0.0542 | (-0.0369, 0.1910) | 0.2117 |
| proposed_vs_baseline_no_context | context_overlap | 0.0101 | (-0.0049, 0.0244) | 0.0897 | 0.0101 | (-0.0071, 0.0297) | 0.1407 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0614 | (-0.1167, -0.0021) | 0.9793 | -0.0614 | (-0.1357, 0.0125) | 0.9553 |
| proposed_vs_baseline_no_context | persona_style | -0.0859 | (-0.2181, 0.0313) | 0.9150 | -0.0859 | (-0.2872, 0.0760) | 0.7917 |
| proposed_vs_baseline_no_context | distinct1 | -0.0436 | (-0.0612, -0.0238) | 1.0000 | -0.0436 | (-0.0613, -0.0214) | 1.0000 |
| proposed_vs_baseline_no_context | length_score | -0.4533 | (-0.5850, -0.3116) | 1.0000 | -0.4533 | (-0.5789, -0.2912) | 1.0000 |
| proposed_vs_baseline_no_context | sentence_score | -0.1750 | (-0.2625, -0.0700) | 1.0000 | -0.1750 | (-0.2763, -0.0412) | 0.9983 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0085 | (-0.0178, 0.0394) | 0.2887 | 0.0085 | (-0.0232, 0.0475) | 0.3260 |
| proposed_vs_baseline_no_context | overall_quality | -0.0259 | (-0.0646, 0.0153) | 0.8970 | -0.0259 | (-0.0834, 0.0422) | 0.7923 |
| controlled_vs_proposed_raw | context_relevance | 0.2092 | (0.1453, 0.2745) | 0.0000 | 0.2092 | (0.1315, 0.2886) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.2205 | (0.1502, 0.3030) | 0.0000 | 0.2205 | (0.1550, 0.2785) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0630 | (0.0136, 0.1075) | 0.0067 | 0.0630 | (0.0127, 0.1008) | 0.0113 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2718 | (0.1877, 0.3606) | 0.0000 | 0.2718 | (0.1678, 0.3806) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0631 | (0.0387, 0.0938) | 0.0000 | 0.0631 | (0.0366, 0.0915) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2548 | (0.1643, 0.3631) | 0.0000 | 0.2548 | (0.1784, 0.3389) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.0834 | (-0.0336, 0.2052) | 0.0890 | 0.0834 | (-0.0767, 0.3004) | 0.2347 |
| controlled_vs_proposed_raw | distinct1 | -0.0143 | (-0.0454, 0.0143) | 0.8157 | -0.0143 | (-0.0403, 0.0086) | 0.8960 |
| controlled_vs_proposed_raw | length_score | 0.2583 | (0.0683, 0.4468) | 0.0040 | 0.2583 | (0.0490, 0.3889) | 0.0107 |
| controlled_vs_proposed_raw | sentence_score | 0.2100 | (0.1225, 0.2975) | 0.0000 | 0.2100 | (0.0833, 0.3000) | 0.0010 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0208 | (-0.0119, 0.0587) | 0.1143 | 0.0208 | (-0.0108, 0.0590) | 0.1127 |
| controlled_vs_proposed_raw | overall_quality | 0.1706 | (0.1279, 0.2130) | 0.0000 | 0.1706 | (0.1171, 0.2231) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2563 | (0.2071, 0.3042) | 0.0000 | 0.2563 | (0.1982, 0.3139) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1781 | (0.0951, 0.2736) | 0.0000 | 0.1781 | (0.0979, 0.2339) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.0194 | (-0.0392, 0.0766) | 0.2600 | 0.0194 | (-0.0620, 0.0741) | 0.3097 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3343 | (0.2694, 0.4002) | 0.0000 | 0.3343 | (0.2552, 0.4106) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0743 | (0.0518, 0.1021) | 0.0000 | 0.0743 | (0.0508, 0.1026) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2024 | (0.1059, 0.3157) | 0.0000 | 0.2024 | (0.1235, 0.2658) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.0811 | (-0.0491, 0.2181) | 0.1123 | 0.0811 | (-0.1098, 0.2934) | 0.2500 |
| controlled_vs_candidate_no_context | distinct1 | -0.0175 | (-0.0486, 0.0114) | 0.8783 | -0.0175 | (-0.0462, 0.0038) | 0.9477 |
| controlled_vs_candidate_no_context | length_score | 0.0467 | (-0.1867, 0.2917) | 0.3357 | 0.0467 | (-0.2745, 0.2834) | 0.3920 |
| controlled_vs_candidate_no_context | sentence_score | 0.2100 | (0.1225, 0.2975) | 0.0000 | 0.2100 | (0.0833, 0.2891) | 0.0023 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0438 | (0.0061, 0.0870) | 0.0107 | 0.0438 | (0.0022, 0.0982) | 0.0163 |
| controlled_vs_candidate_no_context | overall_quality | 0.1706 | (0.1294, 0.2113) | 0.0000 | 0.1706 | (0.1144, 0.2081) | 0.0000 |
| controlled_vs_baseline_no_context | context_relevance | 0.2502 | (0.2009, 0.3001) | 0.0000 | 0.2502 | (0.1972, 0.3036) | 0.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1542 | (0.0794, 0.2438) | 0.0000 | 0.1542 | (0.0889, 0.2066) | 0.0000 |
| controlled_vs_baseline_no_context | naturalness | -0.0626 | (-0.1095, -0.0149) | 0.9957 | -0.0626 | (-0.1086, -0.0199) | 0.9990 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3260 | (0.2614, 0.3896) | 0.0000 | 0.3260 | (0.2552, 0.3982) | 0.0000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0733 | (0.0491, 0.1033) | 0.0000 | 0.0733 | (0.0522, 0.1021) | 0.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1933 | (0.1012, 0.3043) | 0.0000 | 0.1933 | (0.1190, 0.2638) | 0.0000 |
| controlled_vs_baseline_no_context | persona_style | -0.0025 | (-0.0725, 0.0688) | 0.5293 | -0.0025 | (-0.0275, 0.0167) | 0.5997 |
| controlled_vs_baseline_no_context | distinct1 | -0.0579 | (-0.0847, -0.0340) | 1.0000 | -0.0579 | (-0.0753, -0.0370) | 1.0000 |
| controlled_vs_baseline_no_context | length_score | -0.1950 | (-0.3983, 0.0167) | 0.9657 | -0.1950 | (-0.4000, -0.0392) | 0.9967 |
| controlled_vs_baseline_no_context | sentence_score | 0.0350 | (-0.0700, 0.1225) | 0.2883 | 0.0350 | (-0.0412, 0.1225) | 0.2683 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0293 | (-0.0124, 0.0773) | 0.0913 | 0.0293 | (-0.0151, 0.0804) | 0.1177 |
| controlled_vs_baseline_no_context | overall_quality | 0.1447 | (0.1110, 0.1787) | 0.0000 | 0.1447 | (0.1130, 0.1810) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2502 | (0.2016, 0.2991) | 0.0000 | 0.2502 | (0.1935, 0.3050) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1542 | (0.0762, 0.2469) | 0.0000 | 0.1542 | (0.0900, 0.2072) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0626 | (-0.1141, -0.0136) | 0.9927 | -0.0626 | (-0.1124, -0.0196) | 0.9983 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3260 | (0.2623, 0.3904) | 0.0000 | 0.3260 | (0.2538, 0.3971) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0733 | (0.0477, 0.1036) | 0.0000 | 0.0733 | (0.0520, 0.1004) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1933 | (0.1010, 0.3038) | 0.0000 | 0.1933 | (0.1177, 0.2611) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | -0.0025 | (-0.0721, 0.0679) | 0.5293 | -0.0025 | (-0.0267, 0.0172) | 0.6017 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0579 | (-0.0851, -0.0347) | 1.0000 | -0.0579 | (-0.0755, -0.0371) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | -0.1950 | (-0.3933, 0.0117) | 0.9677 | -0.1950 | (-0.4019, -0.0421) | 0.9947 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0350 | (-0.0700, 0.1225) | 0.2870 | 0.0350 | (-0.0437, 0.1167) | 0.2857 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0293 | (-0.0121, 0.0747) | 0.0917 | 0.0293 | (-0.0170, 0.0814) | 0.1133 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1447 | (0.1111, 0.1794) | 0.0000 | 0.1447 | (0.1108, 0.1800) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 9 | 2 | 9 | 0.6750 | 0.8182 |
| proposed_vs_candidate_no_context | persona_consistency | 2 | 5 | 13 | 0.4250 | 0.2857 |
| proposed_vs_candidate_no_context | naturalness | 2 | 9 | 9 | 0.3250 | 0.1818 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 6 | 2 | 12 | 0.6000 | 0.7500 |
| proposed_vs_candidate_no_context | context_overlap | 9 | 2 | 9 | 0.6750 | 0.8182 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 1 | 5 | 14 | 0.4000 | 0.1667 |
| proposed_vs_candidate_no_context | persona_style | 2 | 1 | 17 | 0.5250 | 0.6667 |
| proposed_vs_candidate_no_context | distinct1 | 6 | 4 | 10 | 0.5500 | 0.6000 |
| proposed_vs_candidate_no_context | length_score | 2 | 9 | 9 | 0.3250 | 0.1818 |
| proposed_vs_candidate_no_context | sentence_score | 3 | 3 | 14 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | bertscore_f1 | 9 | 4 | 7 | 0.6250 | 0.6923 |
| proposed_vs_candidate_no_context | overall_quality | 6 | 7 | 7 | 0.4750 | 0.4615 |
| proposed_vs_baseline_no_context | context_relevance | 10 | 10 | 0 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context | persona_consistency | 3 | 9 | 8 | 0.3500 | 0.2500 |
| proposed_vs_baseline_no_context | naturalness | 2 | 18 | 0 | 0.1000 | 0.1000 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 5 | 5 | 10 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context | context_overlap | 12 | 8 | 0 | 0.6000 | 0.6000 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 1 | 8 | 11 | 0.3250 | 0.1111 |
| proposed_vs_baseline_no_context | persona_style | 3 | 5 | 12 | 0.4500 | 0.3750 |
| proposed_vs_baseline_no_context | distinct1 | 2 | 16 | 2 | 0.1500 | 0.1111 |
| proposed_vs_baseline_no_context | length_score | 2 | 18 | 0 | 0.1000 | 0.1000 |
| proposed_vs_baseline_no_context | sentence_score | 2 | 12 | 6 | 0.2500 | 0.1429 |
| proposed_vs_baseline_no_context | bertscore_f1 | 10 | 10 | 0 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context | overall_quality | 5 | 15 | 0 | 0.2500 | 0.2500 |
| controlled_vs_proposed_raw | context_relevance | 18 | 2 | 0 | 0.9000 | 0.9000 |
| controlled_vs_proposed_raw | persona_consistency | 17 | 0 | 3 | 0.9250 | 1.0000 |
| controlled_vs_proposed_raw | naturalness | 14 | 6 | 0 | 0.7000 | 0.7000 |
| controlled_vs_proposed_raw | context_keyword_coverage | 17 | 1 | 2 | 0.9000 | 0.9444 |
| controlled_vs_proposed_raw | context_overlap | 18 | 2 | 0 | 0.9000 | 0.9000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 17 | 0 | 3 | 0.9250 | 1.0000 |
| controlled_vs_proposed_raw | persona_style | 5 | 3 | 12 | 0.5500 | 0.6250 |
| controlled_vs_proposed_raw | distinct1 | 10 | 10 | 0 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | length_score | 13 | 6 | 1 | 0.6750 | 0.6842 |
| controlled_vs_proposed_raw | sentence_score | 13 | 1 | 6 | 0.8000 | 0.9286 |
| controlled_vs_proposed_raw | bertscore_f1 | 11 | 9 | 0 | 0.5500 | 0.5500 |
| controlled_vs_proposed_raw | overall_quality | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_relevance | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 16 | 1 | 3 | 0.8750 | 0.9412 |
| controlled_vs_candidate_no_context | naturalness | 11 | 9 | 0 | 0.5500 | 0.5500 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 19 | 0 | 1 | 0.9750 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 16 | 1 | 3 | 0.8750 | 0.9412 |
| controlled_vs_candidate_no_context | persona_style | 5 | 3 | 12 | 0.5500 | 0.6250 |
| controlled_vs_candidate_no_context | distinct1 | 9 | 11 | 0 | 0.4500 | 0.4500 |
| controlled_vs_candidate_no_context | length_score | 10 | 10 | 0 | 0.5000 | 0.5000 |
| controlled_vs_candidate_no_context | sentence_score | 13 | 1 | 6 | 0.8000 | 0.9286 |
| controlled_vs_candidate_no_context | bertscore_f1 | 13 | 7 | 0 | 0.6500 | 0.6500 |
| controlled_vs_candidate_no_context | overall_quality | 18 | 2 | 0 | 0.9000 | 0.9000 |
| controlled_vs_baseline_no_context | context_relevance | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_vs_baseline_no_context | persona_consistency | 15 | 2 | 3 | 0.8250 | 0.8824 |
| controlled_vs_baseline_no_context | naturalness | 6 | 14 | 0 | 0.3000 | 0.3000 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 19 | 0 | 1 | 0.9750 | 1.0000 |
| controlled_vs_baseline_no_context | context_overlap | 18 | 2 | 0 | 0.9000 | 0.9000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 14 | 1 | 5 | 0.8250 | 0.9333 |
| controlled_vs_baseline_no_context | persona_style | 3 | 3 | 14 | 0.5000 | 0.5000 |
| controlled_vs_baseline_no_context | distinct1 | 2 | 18 | 0 | 0.1000 | 0.1000 |
| controlled_vs_baseline_no_context | length_score | 5 | 13 | 2 | 0.3000 | 0.2778 |
| controlled_vs_baseline_no_context | sentence_score | 5 | 3 | 12 | 0.5500 | 0.6250 |
| controlled_vs_baseline_no_context | bertscore_f1 | 9 | 11 | 0 | 0.4500 | 0.4500 |
| controlled_vs_baseline_no_context | overall_quality | 20 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 19 | 1 | 0 | 0.9500 | 0.9500 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 15 | 2 | 3 | 0.8250 | 0.8824 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 6 | 14 | 0 | 0.3000 | 0.3000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 19 | 0 | 1 | 0.9750 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 18 | 2 | 0 | 0.9000 | 0.9000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 14 | 1 | 5 | 0.8250 | 0.9333 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 3 | 3 | 14 | 0.5000 | 0.5000 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | 2 | 18 | 0 | 0.1000 | 0.1000 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 5 | 13 | 2 | 0.3000 | 0.2778 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 5 | 3 | 12 | 0.5500 | 0.6250 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 9 | 11 | 0 | 0.4500 | 0.4500 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 20 | 0 | 0 | 1.0000 | 1.0000 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.2500 | 0.5000 | 0.5000 |
| proposed_contextual | 0.0000 | 0.0000 | 0.5500 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.5500 | 0.0000 | 0.0000 |
| baseline_no_context | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `8`
- Unique template signatures: `18`
- Template signature ratio: `0.9000`
- Effective sample size by source clustering: `6.67`
- Effective sample size by template-signature clustering: `16.67`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: enabled.

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
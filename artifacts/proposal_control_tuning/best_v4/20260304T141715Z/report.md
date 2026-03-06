# Proposal Alignment Evaluation Report

- Run ID: `20260304T141715Z`
- Generated: `2026-03-04T14:24:21.880313+00:00`
- Scenarios: `artifacts\proposal_control_tuning\best_v4\20260304T141715Z\scenarios.jsonl`
- Scenario count: `40`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off
- `baseline_no_context`: model `phi3:mini`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2910 (0.2653, 0.3179) | 0.3536 (0.3055, 0.4061) | 0.8826 (0.8609, 0.9024) | 0.3961 (0.3787, 0.4145) | 0.0862 |
| proposed_contextual | 0.0765 (0.0483, 0.1102) | 0.1374 (0.1004, 0.1816) | 0.7973 (0.7778, 0.8180) | 0.2204 (0.2000, 0.2433) | 0.0628 |
| candidate_no_context | 0.0206 (0.0123, 0.0300) | 0.1526 (0.1134, 0.1959) | 0.7969 (0.7729, 0.8241) | 0.1991 (0.1822, 0.2186) | 0.0318 |
| baseline_no_context | 0.0400 (0.0263, 0.0557) | 0.2052 (0.1682, 0.2422) | 0.8834 (0.8660, 0.9007) | 0.2411 (0.2266, 0.2571) | 0.0506 |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0559 | 2.7146 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0152 | -0.0999 |
| proposed_vs_candidate_no_context | naturalness | 0.0004 | 0.0005 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0711 | 5.3629 |
| proposed_vs_candidate_no_context | context_overlap | 0.0205 | 0.5427 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0163 | -0.2395 |
| proposed_vs_candidate_no_context | persona_style | -0.0110 | -0.0224 |
| proposed_vs_candidate_no_context | distinct1 | 0.0007 | 0.0007 |
| proposed_vs_candidate_no_context | length_score | -0.0125 | -0.0487 |
| proposed_vs_candidate_no_context | sentence_score | 0.0262 | 0.0360 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0310 | 0.9757 |
| proposed_vs_candidate_no_context | overall_quality | 0.0213 | 0.1071 |
| proposed_vs_baseline_no_context | context_relevance | 0.0365 | 0.9127 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0678 | -0.3306 |
| proposed_vs_baseline_no_context | naturalness | -0.0860 | -0.0974 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0459 | 1.1919 |
| proposed_vs_baseline_no_context | context_overlap | 0.0147 | 0.3367 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0598 | -0.5358 |
| proposed_vs_baseline_no_context | persona_style | -0.1001 | -0.1726 |
| proposed_vs_baseline_no_context | distinct1 | -0.0472 | -0.0481 |
| proposed_vs_baseline_no_context | length_score | -0.2833 | -0.5371 |
| proposed_vs_baseline_no_context | sentence_score | -0.1050 | -0.1221 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0121 | 0.2394 |
| proposed_vs_baseline_no_context | overall_quality | -0.0207 | -0.0859 |
| controlled_vs_proposed_raw | context_relevance | 0.2145 | 2.8038 |
| controlled_vs_proposed_raw | persona_consistency | 0.2162 | 1.5742 |
| controlled_vs_proposed_raw | naturalness | 0.0852 | 0.1069 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2812 | 3.3341 |
| controlled_vs_proposed_raw | context_overlap | 0.0588 | 1.0098 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2449 | 4.7287 |
| controlled_vs_proposed_raw | persona_style | 0.1016 | 0.2119 |
| controlled_vs_proposed_raw | distinct1 | 0.0039 | 0.0041 |
| controlled_vs_proposed_raw | length_score | 0.3283 | 1.3447 |
| controlled_vs_proposed_raw | sentence_score | 0.1837 | 0.2434 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0234 | 0.3730 |
| controlled_vs_proposed_raw | overall_quality | 0.1756 | 0.7968 |
| controlled_vs_candidate_no_context | context_relevance | 0.2704 | 13.1296 |
| controlled_vs_candidate_no_context | persona_consistency | 0.2010 | 1.3170 |
| controlled_vs_candidate_no_context | naturalness | 0.0856 | 0.1075 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3523 | 26.5771 |
| controlled_vs_candidate_no_context | context_overlap | 0.0792 | 2.1005 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2286 | 3.3566 |
| controlled_vs_candidate_no_context | persona_style | 0.0906 | 0.1847 |
| controlled_vs_candidate_no_context | distinct1 | 0.0045 | 0.0049 |
| controlled_vs_candidate_no_context | length_score | 0.3158 | 1.2305 |
| controlled_vs_candidate_no_context | sentence_score | 0.2100 | 0.2882 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0544 | 1.7125 |
| controlled_vs_candidate_no_context | overall_quality | 0.1969 | 0.9891 |
| controlled_vs_baseline_no_context | context_relevance | 0.2510 | 6.2755 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1484 | 0.7233 |
| controlled_vs_baseline_no_context | naturalness | -0.0008 | -0.0009 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3271 | 8.5000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0734 | 1.6865 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1851 | 1.6596 |
| controlled_vs_baseline_no_context | persona_style | 0.0015 | 0.0027 |
| controlled_vs_baseline_no_context | distinct1 | -0.0433 | -0.0442 |
| controlled_vs_baseline_no_context | length_score | 0.0450 | 0.0853 |
| controlled_vs_baseline_no_context | sentence_score | 0.0787 | 0.0916 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0355 | 0.7017 |
| controlled_vs_baseline_no_context | overall_quality | 0.1549 | 0.6424 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2510 | 6.2755 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1484 | 0.7233 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0008 | -0.0009 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3271 | 8.5000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0734 | 1.6865 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1851 | 1.6596 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0015 | 0.0027 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0433 | -0.0442 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.0450 | 0.0853 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0787 | 0.0916 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0355 | 0.7017 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1549 | 0.6424 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0559 | (0.0254, 0.0907) | 0.0000 | 0.0559 | (0.0171, 0.1004) | 0.0007 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0152 | (-0.0551, 0.0227) | 0.7807 | -0.0152 | (-0.0494, 0.0188) | 0.8157 |
| proposed_vs_candidate_no_context | naturalness | 0.0004 | (-0.0304, 0.0312) | 0.5040 | 0.0004 | (-0.0456, 0.0462) | 0.5147 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0711 | (0.0304, 0.1157) | 0.0003 | 0.0711 | (0.0189, 0.1288) | 0.0010 |
| proposed_vs_candidate_no_context | context_overlap | 0.0205 | (0.0111, 0.0313) | 0.0000 | 0.0205 | (0.0085, 0.0352) | 0.0000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0163 | (-0.0566, 0.0208) | 0.7980 | -0.0163 | (-0.0603, 0.0246) | 0.7857 |
| proposed_vs_candidate_no_context | persona_style | -0.0110 | (-0.0738, 0.0513) | 0.6123 | -0.0110 | (-0.0505, 0.0219) | 0.7023 |
| proposed_vs_candidate_no_context | distinct1 | 0.0007 | (-0.0140, 0.0150) | 0.4717 | 0.0007 | (-0.0195, 0.0203) | 0.4703 |
| proposed_vs_candidate_no_context | length_score | -0.0125 | (-0.1359, 0.1050) | 0.5847 | -0.0125 | (-0.1717, 0.1631) | 0.5507 |
| proposed_vs_candidate_no_context | sentence_score | 0.0262 | (-0.0350, 0.0875) | 0.2310 | 0.0262 | (-0.0362, 0.0977) | 0.2987 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0310 | (0.0167, 0.0469) | 0.0000 | 0.0310 | (0.0148, 0.0555) | 0.0000 |
| proposed_vs_candidate_no_context | overall_quality | 0.0213 | (-0.0021, 0.0451) | 0.0373 | 0.0213 | (-0.0079, 0.0557) | 0.0970 |
| proposed_vs_baseline_no_context | context_relevance | 0.0365 | (0.0031, 0.0718) | 0.0183 | 0.0365 | (-0.0048, 0.0885) | 0.0710 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0678 | (-0.1195, -0.0149) | 0.9953 | -0.0678 | (-0.1318, -0.0070) | 0.9847 |
| proposed_vs_baseline_no_context | naturalness | -0.0860 | (-0.1153, -0.0548) | 1.0000 | -0.0860 | (-0.1306, -0.0242) | 0.9950 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0459 | (0.0005, 0.0955) | 0.0237 | 0.0459 | (-0.0104, 0.1141) | 0.0923 |
| proposed_vs_baseline_no_context | context_overlap | 0.0147 | (0.0035, 0.0266) | 0.0040 | 0.0147 | (-0.0013, 0.0328) | 0.0350 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0598 | (-0.1193, -0.0044) | 0.9827 | -0.0598 | (-0.1184, -0.0026) | 0.9793 |
| proposed_vs_baseline_no_context | persona_style | -0.1001 | (-0.1863, -0.0191) | 0.9933 | -0.1001 | (-0.2712, 0.0386) | 0.8843 |
| proposed_vs_baseline_no_context | distinct1 | -0.0472 | (-0.0598, -0.0338) | 1.0000 | -0.0472 | (-0.0628, -0.0281) | 1.0000 |
| proposed_vs_baseline_no_context | length_score | -0.2833 | (-0.3950, -0.1633) | 1.0000 | -0.2833 | (-0.4378, -0.0722) | 0.9930 |
| proposed_vs_baseline_no_context | sentence_score | -0.1050 | (-0.1837, -0.0262) | 0.9950 | -0.1050 | (-0.2279, 0.0553) | 0.9147 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0121 | (-0.0060, 0.0297) | 0.1007 | 0.0121 | (-0.0128, 0.0434) | 0.1947 |
| proposed_vs_baseline_no_context | overall_quality | -0.0207 | (-0.0484, 0.0078) | 0.9237 | -0.0207 | (-0.0609, 0.0269) | 0.8187 |
| controlled_vs_proposed_raw | context_relevance | 0.2145 | (0.1843, 0.2467) | 0.0000 | 0.2145 | (0.1889, 0.2415) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.2162 | (0.1585, 0.2704) | 0.0000 | 0.2162 | (0.1565, 0.2713) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0852 | (0.0560, 0.1152) | 0.0000 | 0.0852 | (0.0532, 0.1115) | 0.0000 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2812 | (0.2393, 0.3227) | 0.0000 | 0.2812 | (0.2489, 0.3149) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0588 | (0.0430, 0.0756) | 0.0000 | 0.0588 | (0.0412, 0.0746) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2449 | (0.1786, 0.3119) | 0.0000 | 0.2449 | (0.1759, 0.3224) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.1016 | (0.0295, 0.1784) | 0.0023 | 0.1016 | (-0.0081, 0.2674) | 0.0567 |
| controlled_vs_proposed_raw | distinct1 | 0.0039 | (-0.0099, 0.0175) | 0.2897 | 0.0039 | (-0.0134, 0.0195) | 0.3273 |
| controlled_vs_proposed_raw | length_score | 0.3283 | (0.2175, 0.4450) | 0.0000 | 0.3283 | (0.2184, 0.4404) | 0.0000 |
| controlled_vs_proposed_raw | sentence_score | 0.1837 | (0.1137, 0.2537) | 0.0000 | 0.1837 | (0.1077, 0.2442) | 0.0000 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0234 | (-0.0034, 0.0511) | 0.0437 | 0.0234 | (-0.0121, 0.0557) | 0.0793 |
| controlled_vs_proposed_raw | overall_quality | 0.1756 | (0.1491, 0.2013) | 0.0000 | 0.1756 | (0.1446, 0.1987) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2704 | (0.2444, 0.2978) | 0.0000 | 0.2704 | (0.2393, 0.3042) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.2010 | (0.1469, 0.2554) | 0.0000 | 0.2010 | (0.1437, 0.2517) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.0856 | (0.0551, 0.1159) | 0.0000 | 0.0856 | (0.0492, 0.1138) | 0.0003 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3523 | (0.3189, 0.3883) | 0.0000 | 0.3523 | (0.3104, 0.3965) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0792 | (0.0669, 0.0936) | 0.0000 | 0.0792 | (0.0660, 0.0928) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2286 | (0.1611, 0.2975) | 0.0000 | 0.2286 | (0.1540, 0.3040) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.0906 | (0.0331, 0.1568) | 0.0007 | 0.0906 | (0.0000, 0.2171) | 0.1017 |
| controlled_vs_candidate_no_context | distinct1 | 0.0045 | (-0.0094, 0.0181) | 0.2650 | 0.0045 | (-0.0124, 0.0205) | 0.3233 |
| controlled_vs_candidate_no_context | length_score | 0.3158 | (0.1875, 0.4350) | 0.0000 | 0.3158 | (0.1728, 0.4334) | 0.0003 |
| controlled_vs_candidate_no_context | sentence_score | 0.2100 | (0.1487, 0.2625) | 0.0000 | 0.2100 | (0.1448, 0.2606) | 0.0000 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0544 | (0.0335, 0.0770) | 0.0000 | 0.0544 | (0.0314, 0.0783) | 0.0000 |
| controlled_vs_candidate_no_context | overall_quality | 0.1969 | (0.1729, 0.2202) | 0.0000 | 0.1969 | (0.1783, 0.2099) | 0.0000 |
| controlled_vs_baseline_no_context | context_relevance | 0.2510 | (0.2215, 0.2813) | 0.0000 | 0.2510 | (0.2190, 0.2911) | 0.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1484 | (0.0941, 0.2056) | 0.0000 | 0.1484 | (0.0784, 0.2150) | 0.0000 |
| controlled_vs_baseline_no_context | naturalness | -0.0008 | (-0.0316, 0.0308) | 0.5083 | -0.0008 | (-0.0362, 0.0418) | 0.5187 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3271 | (0.2890, 0.3680) | 0.0000 | 0.3271 | (0.2852, 0.3817) | 0.0000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0734 | (0.0609, 0.0876) | 0.0000 | 0.0734 | (0.0642, 0.0830) | 0.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1851 | (0.1171, 0.2532) | 0.0000 | 0.1851 | (0.1027, 0.2686) | 0.0000 |
| controlled_vs_baseline_no_context | persona_style | 0.0015 | (-0.0381, 0.0406) | 0.4870 | 0.0015 | (-0.0417, 0.0507) | 0.4577 |
| controlled_vs_baseline_no_context | distinct1 | -0.0433 | (-0.0572, -0.0294) | 1.0000 | -0.0433 | (-0.0547, -0.0303) | 1.0000 |
| controlled_vs_baseline_no_context | length_score | 0.0450 | (-0.0900, 0.1783) | 0.2697 | 0.0450 | (-0.1080, 0.2268) | 0.3033 |
| controlled_vs_baseline_no_context | sentence_score | 0.0788 | (0.0087, 0.1487) | 0.0210 | 0.0788 | (-0.0071, 0.1956) | 0.0447 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0355 | (0.0092, 0.0603) | 0.0033 | 0.0355 | (0.0057, 0.0621) | 0.0123 |
| controlled_vs_baseline_no_context | overall_quality | 0.1549 | (0.1336, 0.1753) | 0.0000 | 0.1549 | (0.1315, 0.1775) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2510 | (0.2213, 0.2809) | 0.0000 | 0.2510 | (0.2199, 0.2904) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1484 | (0.0934, 0.2056) | 0.0000 | 0.1484 | (0.0774, 0.2158) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0008 | (-0.0317, 0.0298) | 0.5297 | -0.0008 | (-0.0345, 0.0422) | 0.5180 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3271 | (0.2876, 0.3682) | 0.0000 | 0.3271 | (0.2850, 0.3820) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0734 | (0.0608, 0.0878) | 0.0000 | 0.0734 | (0.0641, 0.0830) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1851 | (0.1156, 0.2524) | 0.0000 | 0.1851 | (0.1009, 0.2682) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0015 | (-0.0375, 0.0421) | 0.4497 | 0.0015 | (-0.0417, 0.0486) | 0.4483 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0433 | (-0.0565, -0.0305) | 1.0000 | -0.0433 | (-0.0544, -0.0294) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.0450 | (-0.0859, 0.1833) | 0.2430 | 0.0450 | (-0.1021, 0.2334) | 0.2840 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0788 | (0.0087, 0.1487) | 0.0187 | 0.0788 | (0.0000, 0.2032) | 0.0453 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0355 | (0.0084, 0.0610) | 0.0080 | 0.0355 | (0.0061, 0.0619) | 0.0100 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1549 | (0.1332, 0.1759) | 0.0000 | 0.1549 | (0.1313, 0.1779) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 18 | 5 | 17 | 0.6625 | 0.7826 |
| proposed_vs_candidate_no_context | persona_consistency | 7 | 7 | 26 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | naturalness | 14 | 9 | 17 | 0.5625 | 0.6087 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 13 | 2 | 25 | 0.6375 | 0.8667 |
| proposed_vs_candidate_no_context | context_overlap | 18 | 5 | 17 | 0.6625 | 0.7826 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 5 | 6 | 29 | 0.4875 | 0.4545 |
| proposed_vs_candidate_no_context | persona_style | 3 | 4 | 33 | 0.4875 | 0.4286 |
| proposed_vs_candidate_no_context | distinct1 | 11 | 10 | 19 | 0.5125 | 0.5238 |
| proposed_vs_candidate_no_context | length_score | 11 | 11 | 18 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | sentence_score | 8 | 5 | 27 | 0.5375 | 0.6154 |
| proposed_vs_candidate_no_context | bertscore_f1 | 28 | 4 | 8 | 0.8000 | 0.8750 |
| proposed_vs_candidate_no_context | overall_quality | 25 | 7 | 8 | 0.7250 | 0.7812 |
| proposed_vs_baseline_no_context | context_relevance | 18 | 22 | 0 | 0.4500 | 0.4500 |
| proposed_vs_baseline_no_context | persona_consistency | 5 | 19 | 16 | 0.3250 | 0.2083 |
| proposed_vs_baseline_no_context | naturalness | 8 | 32 | 0 | 0.2000 | 0.2000 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 12 | 8 | 20 | 0.5500 | 0.6000 |
| proposed_vs_baseline_no_context | context_overlap | 23 | 17 | 0 | 0.5750 | 0.5750 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 4 | 14 | 22 | 0.3750 | 0.2222 |
| proposed_vs_baseline_no_context | persona_style | 2 | 11 | 27 | 0.3875 | 0.1538 |
| proposed_vs_baseline_no_context | distinct1 | 4 | 33 | 3 | 0.1375 | 0.1081 |
| proposed_vs_baseline_no_context | length_score | 7 | 30 | 3 | 0.2125 | 0.1892 |
| proposed_vs_baseline_no_context | sentence_score | 7 | 19 | 14 | 0.3500 | 0.2692 |
| proposed_vs_baseline_no_context | bertscore_f1 | 23 | 17 | 0 | 0.5750 | 0.5750 |
| proposed_vs_baseline_no_context | overall_quality | 15 | 25 | 0 | 0.3750 | 0.3750 |
| controlled_vs_proposed_raw | context_relevance | 39 | 1 | 0 | 0.9750 | 0.9750 |
| controlled_vs_proposed_raw | persona_consistency | 35 | 2 | 3 | 0.9125 | 0.9459 |
| controlled_vs_proposed_raw | naturalness | 31 | 8 | 1 | 0.7875 | 0.7949 |
| controlled_vs_proposed_raw | context_keyword_coverage | 38 | 1 | 1 | 0.9625 | 0.9744 |
| controlled_vs_proposed_raw | context_overlap | 37 | 3 | 0 | 0.9250 | 0.9250 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 35 | 2 | 3 | 0.9125 | 0.9459 |
| controlled_vs_proposed_raw | persona_style | 10 | 1 | 29 | 0.6125 | 0.9091 |
| controlled_vs_proposed_raw | distinct1 | 21 | 18 | 1 | 0.5375 | 0.5385 |
| controlled_vs_proposed_raw | length_score | 28 | 9 | 3 | 0.7375 | 0.7568 |
| controlled_vs_proposed_raw | sentence_score | 24 | 3 | 13 | 0.7625 | 0.8889 |
| controlled_vs_proposed_raw | bertscore_f1 | 27 | 13 | 0 | 0.6750 | 0.6750 |
| controlled_vs_proposed_raw | overall_quality | 38 | 2 | 0 | 0.9500 | 0.9500 |
| controlled_vs_candidate_no_context | context_relevance | 40 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 34 | 4 | 2 | 0.8750 | 0.8947 |
| controlled_vs_candidate_no_context | naturalness | 31 | 9 | 0 | 0.7750 | 0.7750 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 40 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 40 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 34 | 4 | 2 | 0.8750 | 0.8947 |
| controlled_vs_candidate_no_context | persona_style | 8 | 0 | 32 | 0.6000 | 1.0000 |
| controlled_vs_candidate_no_context | distinct1 | 19 | 17 | 4 | 0.5250 | 0.5278 |
| controlled_vs_candidate_no_context | length_score | 30 | 8 | 2 | 0.7750 | 0.7895 |
| controlled_vs_candidate_no_context | sentence_score | 25 | 1 | 14 | 0.8000 | 0.9615 |
| controlled_vs_candidate_no_context | bertscore_f1 | 33 | 7 | 0 | 0.8250 | 0.8250 |
| controlled_vs_candidate_no_context | overall_quality | 40 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | context_relevance | 40 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 29 | 4 | 7 | 0.8125 | 0.8788 |
| controlled_vs_baseline_no_context | naturalness | 21 | 19 | 0 | 0.5250 | 0.5250 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 40 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | context_overlap | 40 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 29 | 3 | 8 | 0.8250 | 0.9062 |
| controlled_vs_baseline_no_context | persona_style | 3 | 3 | 34 | 0.5000 | 0.5000 |
| controlled_vs_baseline_no_context | distinct1 | 5 | 33 | 2 | 0.1500 | 0.1316 |
| controlled_vs_baseline_no_context | length_score | 23 | 17 | 0 | 0.5750 | 0.5750 |
| controlled_vs_baseline_no_context | sentence_score | 14 | 5 | 21 | 0.6125 | 0.7368 |
| controlled_vs_baseline_no_context | bertscore_f1 | 28 | 12 | 0 | 0.7000 | 0.7000 |
| controlled_vs_baseline_no_context | overall_quality | 39 | 1 | 0 | 0.9750 | 0.9750 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 40 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 29 | 4 | 7 | 0.8125 | 0.8788 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 21 | 19 | 0 | 0.5250 | 0.5250 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 40 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 40 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 29 | 3 | 8 | 0.8250 | 0.9062 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 3 | 3 | 34 | 0.5000 | 0.5000 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | 5 | 33 | 2 | 0.1500 | 0.1316 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 23 | 17 | 0 | 0.5750 | 0.5750 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 14 | 5 | 21 | 0.6125 | 0.7368 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 28 | 12 | 0 | 0.7000 | 0.7000 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 39 | 1 | 0 | 0.9750 | 0.9750 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.4000 | 0.2750 | 0.7250 |
| proposed_contextual | 0.0000 | 0.0000 | 0.5250 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.6500 | 0.0000 | 0.0000 |
| baseline_no_context | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `8`
- Unique template signatures: `33`
- Template signature ratio: `0.8250`
- Effective sample size by source clustering: `7.02`
- Effective sample size by template-signature clustering: `28.57`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: enabled.

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
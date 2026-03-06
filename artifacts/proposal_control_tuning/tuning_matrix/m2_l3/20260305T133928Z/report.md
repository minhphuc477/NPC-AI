# Proposal Alignment Evaluation Report

- Run ID: `20260305T133928Z`
- Generated: `2026-03-05T13:44:10.134837+00:00`
- Scenarios: `artifacts\proposal_control_tuning\tuning_matrix\m2_l3\20260305T133928Z\scenarios.jsonl`
- Scenario count: `20`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off
- `baseline_no_context`: model `phi3:mini`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2726 (0.2377, 0.3110) | 0.2859 (0.2370, 0.3377) | 0.8858 (0.8525, 0.9177) | 0.3656 (0.3453, 0.3853) | 0.0808 |
| proposed_contextual | 0.0690 (0.0347, 0.1083) | 0.1094 (0.0811, 0.1418) | 0.8051 (0.7762, 0.8373) | 0.2095 (0.1841, 0.2374) | 0.0670 |
| candidate_no_context | 0.0381 (0.0165, 0.0648) | 0.1410 (0.0953, 0.2043) | 0.8115 (0.7785, 0.8476) | 0.2070 (0.1825, 0.2359) | 0.0537 |
| baseline_no_context | 0.0388 (0.0209, 0.0598) | 0.1851 (0.1373, 0.2374) | 0.8875 (0.8639, 0.9114) | 0.2348 (0.2143, 0.2573) | 0.0530 |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0309 | 0.8097 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0316 | -0.2244 |
| proposed_vs_candidate_no_context | naturalness | -0.0064 | -0.0079 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0405 | 1.1889 |
| proposed_vs_candidate_no_context | context_overlap | 0.0084 | 0.1759 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0369 | -0.8378 |
| proposed_vs_candidate_no_context | persona_style | -0.0106 | -0.0200 |
| proposed_vs_candidate_no_context | distinct1 | -0.0027 | -0.0028 |
| proposed_vs_candidate_no_context | length_score | -0.0267 | -0.0874 |
| proposed_vs_candidate_no_context | sentence_score | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0134 | 0.2496 |
| proposed_vs_candidate_no_context | overall_quality | 0.0026 | 0.0124 |
| proposed_vs_baseline_no_context | context_relevance | 0.0303 | 0.7802 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0758 | -0.4093 |
| proposed_vs_baseline_no_context | naturalness | -0.0823 | -0.0928 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0383 | 1.0521 |
| proposed_vs_baseline_no_context | context_overlap | 0.0116 | 0.2607 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0769 | -0.9150 |
| proposed_vs_baseline_no_context | persona_style | -0.0713 | -0.1209 |
| proposed_vs_baseline_no_context | distinct1 | -0.0477 | -0.0485 |
| proposed_vs_baseline_no_context | length_score | -0.2550 | -0.4781 |
| proposed_vs_baseline_no_context | sentence_score | -0.1225 | -0.1396 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0140 | 0.2651 |
| proposed_vs_baseline_no_context | overall_quality | -0.0253 | -0.1077 |
| controlled_vs_proposed_raw | context_relevance | 0.2036 | 2.9489 |
| controlled_vs_proposed_raw | persona_consistency | 0.1765 | 1.6139 |
| controlled_vs_proposed_raw | naturalness | 0.0806 | 0.1002 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2683 | 3.5959 |
| controlled_vs_proposed_raw | context_overlap | 0.0524 | 0.9361 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1945 | 27.2333 |
| controlled_vs_proposed_raw | persona_style | 0.1044 | 0.2014 |
| controlled_vs_proposed_raw | distinct1 | -0.0074 | -0.0079 |
| controlled_vs_proposed_raw | length_score | 0.3667 | 1.3174 |
| controlled_vs_proposed_raw | sentence_score | 0.1050 | 0.1391 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0138 | 0.2056 |
| controlled_vs_proposed_raw | overall_quality | 0.1561 | 0.7449 |
| controlled_vs_candidate_no_context | context_relevance | 0.2344 | 6.1462 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1449 | 1.0273 |
| controlled_vs_candidate_no_context | naturalness | 0.0742 | 0.0915 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3089 | 9.0600 |
| controlled_vs_candidate_no_context | context_overlap | 0.0608 | 1.2768 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1576 | 3.5784 |
| controlled_vs_candidate_no_context | persona_style | 0.0938 | 0.1774 |
| controlled_vs_candidate_no_context | distinct1 | -0.0100 | -0.0107 |
| controlled_vs_candidate_no_context | length_score | 0.3400 | 1.1148 |
| controlled_vs_candidate_no_context | sentence_score | 0.1050 | 0.1391 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0272 | 0.5065 |
| controlled_vs_candidate_no_context | overall_quality | 0.1587 | 0.7666 |
| controlled_vs_baseline_no_context | context_relevance | 0.2338 | 6.0299 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1007 | 0.5440 |
| controlled_vs_baseline_no_context | naturalness | -0.0017 | -0.0019 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3066 | 8.4312 |
| controlled_vs_baseline_no_context | context_overlap | 0.0640 | 1.4408 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1176 | 1.3994 |
| controlled_vs_baseline_no_context | persona_style | 0.0331 | 0.0562 |
| controlled_vs_baseline_no_context | distinct1 | -0.0551 | -0.0560 |
| controlled_vs_baseline_no_context | length_score | 0.1117 | 0.2094 |
| controlled_vs_baseline_no_context | sentence_score | -0.0175 | -0.0199 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0278 | 0.5251 |
| controlled_vs_baseline_no_context | overall_quality | 0.1308 | 0.5569 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2338 | 6.0299 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1007 | 0.5440 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0017 | -0.0019 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3066 | 8.4312 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0640 | 1.4408 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1176 | 1.3994 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0331 | 0.0562 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0551 | -0.0560 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.1117 | 0.2094 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | -0.0175 | -0.0199 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0278 | 0.5251 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1308 | 0.5569 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0309 | (0.0104, 0.0552) | 0.0003 | 0.0309 | (0.0066, 0.0627) | 0.0040 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0316 | (-0.0970, 0.0138) | 0.8483 | -0.0316 | (-0.1206, 0.0123) | 0.8607 |
| proposed_vs_candidate_no_context | naturalness | -0.0064 | (-0.0417, 0.0276) | 0.6530 | -0.0064 | (-0.0479, 0.0350) | 0.6257 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0405 | (0.0133, 0.0723) | 0.0003 | 0.0405 | (0.0098, 0.0808) | 0.0033 |
| proposed_vs_candidate_no_context | context_overlap | 0.0084 | (-0.0037, 0.0219) | 0.0963 | 0.0084 | (-0.0044, 0.0211) | 0.0953 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0369 | (-0.1167, 0.0143) | 0.8693 | -0.0369 | (-0.1444, 0.0143) | 0.8900 |
| proposed_vs_candidate_no_context | persona_style | -0.0106 | (-0.0984, 0.0556) | 0.5610 | -0.0106 | (-0.0929, 0.0588) | 0.5740 |
| proposed_vs_candidate_no_context | distinct1 | -0.0027 | (-0.0180, 0.0130) | 0.6273 | -0.0027 | (-0.0176, 0.0178) | 0.6377 |
| proposed_vs_candidate_no_context | length_score | -0.0267 | (-0.1550, 0.1150) | 0.6820 | -0.0267 | (-0.1800, 0.1350) | 0.6457 |
| proposed_vs_candidate_no_context | sentence_score | 0.0000 | (-0.0700, 0.0700) | 0.5913 | 0.0000 | (-0.0737, 0.0750) | 0.5993 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0134 | (-0.0057, 0.0361) | 0.1020 | 0.0134 | (-0.0082, 0.0438) | 0.1297 |
| proposed_vs_candidate_no_context | overall_quality | 0.0026 | (-0.0278, 0.0276) | 0.4063 | 0.0026 | (-0.0334, 0.0285) | 0.4087 |
| proposed_vs_baseline_no_context | context_relevance | 0.0303 | (-0.0162, 0.0785) | 0.0973 | 0.0303 | (-0.0294, 0.0989) | 0.1773 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0758 | (-0.1438, -0.0163) | 0.9963 | -0.0758 | (-0.1545, -0.0075) | 0.9860 |
| proposed_vs_baseline_no_context | naturalness | -0.0823 | (-0.1262, -0.0328) | 0.9983 | -0.0823 | (-0.1359, -0.0049) | 0.9797 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0383 | (-0.0284, 0.1023) | 0.1373 | 0.0383 | (-0.0455, 0.1372) | 0.1940 |
| proposed_vs_baseline_no_context | context_overlap | 0.0116 | (-0.0045, 0.0273) | 0.0693 | 0.0116 | (-0.0064, 0.0313) | 0.1100 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0769 | (-0.1483, -0.0195) | 0.9960 | -0.0769 | (-0.1577, -0.0175) | 1.0000 |
| proposed_vs_baseline_no_context | persona_style | -0.0713 | (-0.2223, 0.0638) | 0.8353 | -0.0713 | (-0.2892, 0.1173) | 0.7420 |
| proposed_vs_baseline_no_context | distinct1 | -0.0477 | (-0.0640, -0.0309) | 1.0000 | -0.0477 | (-0.0654, -0.0295) | 1.0000 |
| proposed_vs_baseline_no_context | length_score | -0.2550 | (-0.4333, -0.0567) | 0.9930 | -0.2550 | (-0.4649, 0.0444) | 0.9630 |
| proposed_vs_baseline_no_context | sentence_score | -0.1225 | (-0.2450, 0.0000) | 0.9800 | -0.1225 | (-0.2546, 0.0700) | 0.9183 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0140 | (-0.0242, 0.0557) | 0.2503 | 0.0140 | (-0.0319, 0.0674) | 0.3000 |
| proposed_vs_baseline_no_context | overall_quality | -0.0253 | (-0.0676, 0.0173) | 0.8760 | -0.0253 | (-0.0761, 0.0352) | 0.8123 |
| controlled_vs_proposed_raw | context_relevance | 0.2036 | (0.1663, 0.2348) | 0.0000 | 0.2036 | (0.1584, 0.2433) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.1765 | (0.1329, 0.2177) | 0.0000 | 0.1765 | (0.1446, 0.2196) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0806 | (0.0243, 0.1322) | 0.0020 | 0.0806 | (-0.0019, 0.1430) | 0.0280 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2683 | (0.2191, 0.3100) | 0.0000 | 0.2683 | (0.2085, 0.3167) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0524 | (0.0362, 0.0690) | 0.0000 | 0.0524 | (0.0331, 0.0734) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1945 | (0.1445, 0.2417) | 0.0000 | 0.1945 | (0.1706, 0.2319) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.1044 | (0.0083, 0.2112) | 0.0147 | 0.1044 | (-0.0048, 0.3171) | 0.0963 |
| controlled_vs_proposed_raw | distinct1 | -0.0074 | (-0.0349, 0.0157) | 0.7083 | -0.0074 | (-0.0378, 0.0161) | 0.7167 |
| controlled_vs_proposed_raw | length_score | 0.3667 | (0.1783, 0.5434) | 0.0003 | 0.3667 | (0.0944, 0.5803) | 0.0030 |
| controlled_vs_proposed_raw | sentence_score | 0.1050 | (-0.0175, 0.2275) | 0.0643 | 0.1050 | (-0.0656, 0.2333) | 0.1373 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0138 | (-0.0199, 0.0433) | 0.1820 | 0.0138 | (-0.0244, 0.0442) | 0.1973 |
| controlled_vs_proposed_raw | overall_quality | 0.1561 | (0.1291, 0.1808) | 0.0000 | 0.1561 | (0.1214, 0.1848) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2344 | (0.1941, 0.2726) | 0.0000 | 0.2344 | (0.1870, 0.2875) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1449 | (0.0685, 0.2078) | 0.0000 | 0.1449 | (0.0635, 0.2026) | 0.0033 |
| controlled_vs_candidate_no_context | naturalness | 0.0742 | (0.0268, 0.1236) | 0.0020 | 0.0742 | (0.0011, 0.1259) | 0.0240 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3089 | (0.2555, 0.3614) | 0.0000 | 0.3089 | (0.2498, 0.3758) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0608 | (0.0433, 0.0819) | 0.0000 | 0.0608 | (0.0439, 0.0839) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1576 | (0.0690, 0.2307) | 0.0003 | 0.1576 | (0.0592, 0.2274) | 0.0050 |
| controlled_vs_candidate_no_context | persona_style | 0.0938 | (0.0143, 0.1974) | 0.0050 | 0.0938 | (0.0023, 0.2244) | 0.0240 |
| controlled_vs_candidate_no_context | distinct1 | -0.0100 | (-0.0336, 0.0120) | 0.8113 | -0.0100 | (-0.0358, 0.0091) | 0.8223 |
| controlled_vs_candidate_no_context | length_score | 0.3400 | (0.1816, 0.5017) | 0.0000 | 0.3400 | (0.0925, 0.5243) | 0.0037 |
| controlled_vs_candidate_no_context | sentence_score | 0.1050 | (-0.0175, 0.2100) | 0.0437 | 0.1050 | (-0.0219, 0.2068) | 0.0780 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0272 | (0.0092, 0.0423) | 0.0020 | 0.0272 | (0.0158, 0.0390) | 0.0000 |
| controlled_vs_candidate_no_context | overall_quality | 0.1587 | (0.1234, 0.1893) | 0.0000 | 0.1587 | (0.1134, 0.1931) | 0.0000 |
| controlled_vs_baseline_no_context | context_relevance | 0.2338 | (0.1932, 0.2783) | 0.0000 | 0.2338 | (0.1806, 0.2946) | 0.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1007 | (0.0353, 0.1611) | 0.0007 | 0.1007 | (0.0317, 0.1684) | 0.0023 |
| controlled_vs_baseline_no_context | naturalness | -0.0017 | (-0.0366, 0.0375) | 0.5320 | -0.0017 | (-0.0413, 0.0344) | 0.5420 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3066 | (0.2511, 0.3648) | 0.0000 | 0.3066 | (0.2315, 0.3850) | 0.0000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0640 | (0.0502, 0.0802) | 0.0000 | 0.0640 | (0.0534, 0.0803) | 0.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1176 | (0.0469, 0.1829) | 0.0010 | 0.1176 | (0.0426, 0.1910) | 0.0010 |
| controlled_vs_baseline_no_context | persona_style | 0.0331 | (-0.0417, 0.1171) | 0.2120 | 0.0331 | (-0.0395, 0.1491) | 0.3553 |
| controlled_vs_baseline_no_context | distinct1 | -0.0551 | (-0.0728, -0.0396) | 1.0000 | -0.0551 | (-0.0779, -0.0353) | 1.0000 |
| controlled_vs_baseline_no_context | length_score | 0.1117 | (-0.0317, 0.2600) | 0.0717 | 0.1117 | (-0.0667, 0.2627) | 0.0940 |
| controlled_vs_baseline_no_context | sentence_score | -0.0175 | (-0.1050, 0.0700) | 0.7180 | -0.0175 | (-0.1105, 0.0737) | 0.7073 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0278 | (-0.0003, 0.0566) | 0.0273 | 0.0278 | (-0.0007, 0.0524) | 0.0293 |
| controlled_vs_baseline_no_context | overall_quality | 0.1308 | (0.0981, 0.1611) | 0.0000 | 0.1308 | (0.0993, 0.1651) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2338 | (0.1944, 0.2769) | 0.0000 | 0.2338 | (0.1784, 0.2900) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1007 | (0.0354, 0.1598) | 0.0010 | 0.1007 | (0.0317, 0.1692) | 0.0027 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0017 | (-0.0397, 0.0358) | 0.5383 | -0.0017 | (-0.0427, 0.0345) | 0.5313 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3066 | (0.2507, 0.3682) | 0.0000 | 0.3066 | (0.2298, 0.3850) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0640 | (0.0493, 0.0797) | 0.0000 | 0.0640 | (0.0538, 0.0799) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1176 | (0.0452, 0.1831) | 0.0000 | 0.1176 | (0.0490, 0.1905) | 0.0013 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0331 | (-0.0428, 0.1173) | 0.2223 | 0.0331 | (-0.0375, 0.1491) | 0.3447 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0551 | (-0.0729, -0.0393) | 1.0000 | -0.0551 | (-0.0765, -0.0344) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.1117 | (-0.0317, 0.2550) | 0.0707 | 0.1117 | (-0.0569, 0.2603) | 0.0917 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | -0.0175 | (-0.1050, 0.0700) | 0.7100 | -0.0175 | (-0.1105, 0.0778) | 0.7043 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0278 | (-0.0021, 0.0554) | 0.0357 | 0.0278 | (-0.0002, 0.0531) | 0.0253 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1308 | (0.1014, 0.1593) | 0.0000 | 0.1308 | (0.0994, 0.1657) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 8 | 4 | 8 | 0.6000 | 0.6667 |
| proposed_vs_candidate_no_context | persona_consistency | 3 | 3 | 14 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | naturalness | 5 | 7 | 8 | 0.4500 | 0.4167 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 6 | 0 | 14 | 0.6500 | 1.0000 |
| proposed_vs_candidate_no_context | context_overlap | 6 | 6 | 8 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 1 | 2 | 17 | 0.4750 | 0.3333 |
| proposed_vs_candidate_no_context | persona_style | 2 | 1 | 17 | 0.5250 | 0.6667 |
| proposed_vs_candidate_no_context | distinct1 | 5 | 6 | 9 | 0.4750 | 0.4545 |
| proposed_vs_candidate_no_context | length_score | 5 | 7 | 8 | 0.4500 | 0.4167 |
| proposed_vs_candidate_no_context | sentence_score | 2 | 2 | 16 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | bertscore_f1 | 9 | 5 | 6 | 0.6000 | 0.6429 |
| proposed_vs_candidate_no_context | overall_quality | 9 | 5 | 6 | 0.6000 | 0.6429 |
| proposed_vs_baseline_no_context | context_relevance | 10 | 10 | 0 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context | persona_consistency | 4 | 9 | 7 | 0.3750 | 0.3077 |
| proposed_vs_baseline_no_context | naturalness | 4 | 16 | 0 | 0.2000 | 0.2000 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 7 | 5 | 8 | 0.5500 | 0.5833 |
| proposed_vs_baseline_no_context | context_overlap | 12 | 8 | 0 | 0.6000 | 0.6000 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 1 | 7 | 12 | 0.3500 | 0.1250 |
| proposed_vs_baseline_no_context | persona_style | 3 | 5 | 12 | 0.4500 | 0.3750 |
| proposed_vs_baseline_no_context | distinct1 | 1 | 16 | 3 | 0.1250 | 0.0588 |
| proposed_vs_baseline_no_context | length_score | 4 | 16 | 0 | 0.2000 | 0.2000 |
| proposed_vs_baseline_no_context | sentence_score | 4 | 11 | 5 | 0.3250 | 0.2667 |
| proposed_vs_baseline_no_context | bertscore_f1 | 11 | 9 | 0 | 0.5500 | 0.5500 |
| proposed_vs_baseline_no_context | overall_quality | 8 | 12 | 0 | 0.4000 | 0.4000 |
| controlled_vs_proposed_raw | context_relevance | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_vs_proposed_raw | persona_consistency | 17 | 0 | 3 | 0.9250 | 1.0000 |
| controlled_vs_proposed_raw | naturalness | 15 | 5 | 0 | 0.7500 | 0.7500 |
| controlled_vs_proposed_raw | context_keyword_coverage | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_vs_proposed_raw | context_overlap | 18 | 1 | 1 | 0.9250 | 0.9474 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 17 | 0 | 3 | 0.9250 | 1.0000 |
| controlled_vs_proposed_raw | persona_style | 4 | 1 | 15 | 0.5750 | 0.8000 |
| controlled_vs_proposed_raw | distinct1 | 11 | 9 | 0 | 0.5500 | 0.5500 |
| controlled_vs_proposed_raw | length_score | 15 | 4 | 1 | 0.7750 | 0.7895 |
| controlled_vs_proposed_raw | sentence_score | 10 | 4 | 6 | 0.6500 | 0.7143 |
| controlled_vs_proposed_raw | bertscore_f1 | 15 | 5 | 0 | 0.7500 | 0.7500 |
| controlled_vs_proposed_raw | overall_quality | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_vs_candidate_no_context | context_relevance | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_vs_candidate_no_context | persona_consistency | 15 | 2 | 3 | 0.8250 | 0.8824 |
| controlled_vs_candidate_no_context | naturalness | 13 | 7 | 0 | 0.6500 | 0.6500 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 19 | 0 | 1 | 0.9750 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 15 | 1 | 4 | 0.8500 | 0.9375 |
| controlled_vs_candidate_no_context | persona_style | 5 | 1 | 14 | 0.6000 | 0.8333 |
| controlled_vs_candidate_no_context | distinct1 | 9 | 10 | 1 | 0.4750 | 0.4737 |
| controlled_vs_candidate_no_context | length_score | 14 | 6 | 0 | 0.7000 | 0.7000 |
| controlled_vs_candidate_no_context | sentence_score | 9 | 3 | 8 | 0.6500 | 0.7500 |
| controlled_vs_candidate_no_context | bertscore_f1 | 17 | 3 | 0 | 0.8500 | 0.8500 |
| controlled_vs_candidate_no_context | overall_quality | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_vs_baseline_no_context | context_relevance | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 14 | 3 | 3 | 0.7750 | 0.8235 |
| controlled_vs_baseline_no_context | naturalness | 9 | 11 | 0 | 0.4500 | 0.4500 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | context_overlap | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 14 | 2 | 4 | 0.8000 | 0.8750 |
| controlled_vs_baseline_no_context | persona_style | 3 | 2 | 15 | 0.5250 | 0.6000 |
| controlled_vs_baseline_no_context | distinct1 | 2 | 18 | 0 | 0.1000 | 0.1000 |
| controlled_vs_baseline_no_context | length_score | 12 | 8 | 0 | 0.6000 | 0.6000 |
| controlled_vs_baseline_no_context | sentence_score | 3 | 4 | 13 | 0.4750 | 0.4286 |
| controlled_vs_baseline_no_context | bertscore_f1 | 14 | 6 | 0 | 0.7000 | 0.7000 |
| controlled_vs_baseline_no_context | overall_quality | 19 | 1 | 0 | 0.9500 | 0.9500 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 20 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 14 | 3 | 3 | 0.7750 | 0.8235 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 9 | 11 | 0 | 0.4500 | 0.4500 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 20 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 20 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 14 | 2 | 4 | 0.8000 | 0.8750 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 3 | 2 | 15 | 0.5250 | 0.6000 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | 2 | 18 | 0 | 0.1000 | 0.1000 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 12 | 8 | 0 | 0.6000 | 0.6000 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 3 | 4 | 13 | 0.4750 | 0.4286 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 14 | 6 | 0 | 0.7000 | 0.7000 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 19 | 1 | 0 | 0.9500 | 0.9500 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.4000 | 0.4000 | 0.6000 |
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
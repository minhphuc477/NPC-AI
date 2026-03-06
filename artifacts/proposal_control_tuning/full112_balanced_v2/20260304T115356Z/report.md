# Proposal Alignment Evaluation Report

- Run ID: `20260304T115356Z`
- Generated: `2026-03-04T12:18:46.277917+00:00`
- Scenarios: `artifacts\proposal_control_tuning\full112_balanced_v2\20260304T115356Z\scenarios.jsonl`
- Scenario count: `112`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off
- `baseline_no_context`: model `phi3:mini`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2394 (0.2271, 0.2522) | 0.3663 (0.3376, 0.3977) | 0.9133 (0.9023, 0.9233) | 0.3836 (0.3731, 0.3946) | 0.0746 |
| proposed_contextual | 0.0875 (0.0656, 0.1116) | 0.1545 (0.1314, 0.1785) | 0.7999 (0.7871, 0.8138) | 0.2321 (0.2163, 0.2498) | 0.0741 |
| candidate_no_context | 0.0338 (0.0258, 0.0429) | 0.1733 (0.1457, 0.2036) | 0.8121 (0.7982, 0.8279) | 0.2146 (0.2024, 0.2274) | 0.0377 |
| baseline_no_context | 0.0503 (0.0415, 0.0596) | 0.1920 (0.1726, 0.2125) | 0.8868 (0.8752, 0.8983) | 0.2422 (0.2336, 0.2508) | 0.0559 |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0537 | 1.5900 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0188 | -0.1083 |
| proposed_vs_candidate_no_context | naturalness | -0.0122 | -0.0151 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0685 | 2.1997 |
| proposed_vs_candidate_no_context | context_overlap | 0.0192 | 0.4800 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0240 | -0.2675 |
| proposed_vs_candidate_no_context | persona_style | 0.0022 | 0.0044 |
| proposed_vs_candidate_no_context | distinct1 | -0.0060 | -0.0064 |
| proposed_vs_candidate_no_context | length_score | -0.0408 | -0.1371 |
| proposed_vs_candidate_no_context | sentence_score | -0.0156 | -0.0202 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0364 | 0.9654 |
| proposed_vs_candidate_no_context | overall_quality | 0.0175 | 0.0815 |
| proposed_vs_baseline_no_context | context_relevance | 0.0373 | 0.7412 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0375 | -0.1952 |
| proposed_vs_baseline_no_context | naturalness | -0.0869 | -0.0980 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0463 | 0.8656 |
| proposed_vs_baseline_no_context | context_overlap | 0.0163 | 0.3793 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0267 | -0.2887 |
| proposed_vs_baseline_no_context | persona_style | -0.0807 | -0.1366 |
| proposed_vs_baseline_no_context | distinct1 | -0.0462 | -0.0473 |
| proposed_vs_baseline_no_context | length_score | -0.2863 | -0.5274 |
| proposed_vs_baseline_no_context | sentence_score | -0.1161 | -0.1326 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0182 | 0.3247 |
| proposed_vs_baseline_no_context | overall_quality | -0.0100 | -0.0415 |
| controlled_vs_proposed_raw | context_relevance | 0.1519 | 1.7353 |
| controlled_vs_proposed_raw | persona_consistency | 0.2117 | 1.3699 |
| controlled_vs_proposed_raw | naturalness | 0.1134 | 0.1418 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2014 | 2.0202 |
| controlled_vs_proposed_raw | context_overlap | 0.0363 | 0.6138 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2432 | 3.6968 |
| controlled_vs_proposed_raw | persona_style | 0.0859 | 0.1686 |
| controlled_vs_proposed_raw | distinct1 | 0.0070 | 0.0075 |
| controlled_vs_proposed_raw | length_score | 0.4509 | 1.7575 |
| controlled_vs_proposed_raw | sentence_score | 0.2031 | 0.2675 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0005 | 0.0065 |
| controlled_vs_proposed_raw | overall_quality | 0.1515 | 0.6525 |
| controlled_vs_candidate_no_context | context_relevance | 0.2056 | 6.0846 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1929 | 1.1132 |
| controlled_vs_candidate_no_context | naturalness | 0.1012 | 0.1246 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.2699 | 8.6639 |
| controlled_vs_candidate_no_context | context_overlap | 0.0554 | 1.3885 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2191 | 2.4403 |
| controlled_vs_candidate_no_context | persona_style | 0.0881 | 0.1737 |
| controlled_vs_candidate_no_context | distinct1 | 0.0010 | 0.0011 |
| controlled_vs_candidate_no_context | length_score | 0.4101 | 1.3794 |
| controlled_vs_candidate_no_context | sentence_score | 0.1875 | 0.2419 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0369 | 0.9783 |
| controlled_vs_candidate_no_context | overall_quality | 0.1690 | 0.7872 |
| controlled_vs_baseline_no_context | context_relevance | 0.1891 | 3.7628 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1742 | 0.9072 |
| controlled_vs_baseline_no_context | naturalness | 0.0265 | 0.0299 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.2476 | 4.6344 |
| controlled_vs_baseline_no_context | context_overlap | 0.0525 | 1.2260 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2165 | 2.3407 |
| controlled_vs_baseline_no_context | persona_style | 0.0053 | 0.0089 |
| controlled_vs_baseline_no_context | distinct1 | -0.0392 | -0.0401 |
| controlled_vs_baseline_no_context | length_score | 0.1646 | 0.3032 |
| controlled_vs_baseline_no_context | sentence_score | 0.0871 | 0.0994 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0186 | 0.3333 |
| controlled_vs_baseline_no_context | overall_quality | 0.1414 | 0.5840 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.1891 | 3.7628 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1742 | 0.9072 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 0.0265 | 0.0299 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.2476 | 4.6344 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0525 | 1.2260 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2165 | 2.3407 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0053 | 0.0089 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0392 | -0.0401 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.1646 | 0.3032 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0871 | 0.0994 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0186 | 0.3333 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1414 | 0.5840 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0537 | (0.0315, 0.0790) | 0.0000 | 0.0537 | (0.0283, 0.0768) | 0.0000 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0188 | (-0.0492, 0.0098) | 0.9017 | -0.0188 | (-0.0483, 0.0076) | 0.9067 |
| proposed_vs_candidate_no_context | naturalness | -0.0122 | (-0.0275, 0.0037) | 0.9380 | -0.0122 | (-0.0285, 0.0019) | 0.9457 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0685 | (0.0378, 0.1001) | 0.0000 | 0.0685 | (0.0361, 0.0982) | 0.0000 |
| proposed_vs_candidate_no_context | context_overlap | 0.0192 | (0.0106, 0.0289) | 0.0000 | 0.0192 | (0.0082, 0.0314) | 0.0000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0240 | (-0.0602, 0.0089) | 0.9163 | -0.0240 | (-0.0566, 0.0048) | 0.9410 |
| proposed_vs_candidate_no_context | persona_style | 0.0022 | (-0.0273, 0.0294) | 0.4333 | 0.0022 | (-0.0328, 0.0310) | 0.4333 |
| proposed_vs_candidate_no_context | distinct1 | -0.0060 | (-0.0131, 0.0011) | 0.9453 | -0.0060 | (-0.0141, 0.0014) | 0.9400 |
| proposed_vs_candidate_no_context | length_score | -0.0408 | (-0.1015, 0.0197) | 0.9153 | -0.0408 | (-0.0932, 0.0095) | 0.9510 |
| proposed_vs_candidate_no_context | sentence_score | -0.0156 | (-0.0500, 0.0219) | 0.8247 | -0.0156 | (-0.0563, 0.0250) | 0.7803 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0364 | (0.0236, 0.0501) | 0.0000 | 0.0364 | (0.0230, 0.0487) | 0.0000 |
| proposed_vs_candidate_no_context | overall_quality | 0.0175 | (0.0005, 0.0343) | 0.0227 | 0.0175 | (0.0072, 0.0298) | 0.0000 |
| proposed_vs_baseline_no_context | context_relevance | 0.0373 | (0.0138, 0.0627) | 0.0007 | 0.0373 | (0.0030, 0.0707) | 0.0143 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0375 | (-0.0652, -0.0083) | 0.9940 | -0.0375 | (-0.0823, 0.0038) | 0.9630 |
| proposed_vs_baseline_no_context | naturalness | -0.0869 | (-0.1047, -0.0682) | 1.0000 | -0.0869 | (-0.1129, -0.0560) | 1.0000 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0463 | (0.0153, 0.0802) | 0.0003 | 0.0463 | (0.0012, 0.0934) | 0.0203 |
| proposed_vs_baseline_no_context | context_overlap | 0.0163 | (0.0072, 0.0270) | 0.0000 | 0.0163 | (0.0053, 0.0292) | 0.0017 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0267 | (-0.0591, 0.0049) | 0.9520 | -0.0267 | (-0.0640, 0.0106) | 0.9257 |
| proposed_vs_baseline_no_context | persona_style | -0.0807 | (-0.1242, -0.0405) | 1.0000 | -0.0807 | (-0.2145, 0.0113) | 0.9263 |
| proposed_vs_baseline_no_context | distinct1 | -0.0462 | (-0.0557, -0.0359) | 1.0000 | -0.0462 | (-0.0603, -0.0297) | 1.0000 |
| proposed_vs_baseline_no_context | length_score | -0.2863 | (-0.3554, -0.2179) | 1.0000 | -0.2863 | (-0.3705, -0.1810) | 1.0000 |
| proposed_vs_baseline_no_context | sentence_score | -0.1161 | (-0.1567, -0.0728) | 1.0000 | -0.1161 | (-0.1946, -0.0344) | 0.9970 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0182 | (0.0058, 0.0306) | 0.0017 | 0.0182 | (-0.0044, 0.0420) | 0.0583 |
| proposed_vs_baseline_no_context | overall_quality | -0.0100 | (-0.0281, 0.0087) | 0.8703 | -0.0100 | (-0.0405, 0.0177) | 0.7537 |
| controlled_vs_proposed_raw | context_relevance | 0.1519 | (0.1253, 0.1759) | 0.0000 | 0.1519 | (0.1182, 0.1876) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.2117 | (0.1815, 0.2428) | 0.0000 | 0.2117 | (0.1720, 0.2554) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.1134 | (0.0948, 0.1310) | 0.0000 | 0.1134 | (0.0768, 0.1503) | 0.0000 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2014 | (0.1675, 0.2325) | 0.0000 | 0.2014 | (0.1568, 0.2467) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0363 | (0.0248, 0.0462) | 0.0000 | 0.0363 | (0.0237, 0.0509) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2432 | (0.2094, 0.2787) | 0.0000 | 0.2432 | (0.1925, 0.3027) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.0859 | (0.0457, 0.1294) | 0.0000 | 0.0859 | (0.0056, 0.2171) | 0.0133 |
| controlled_vs_proposed_raw | distinct1 | 0.0070 | (-0.0014, 0.0152) | 0.0503 | 0.0070 | (-0.0065, 0.0193) | 0.1567 |
| controlled_vs_proposed_raw | length_score | 0.4509 | (0.3768, 0.5205) | 0.0000 | 0.4509 | (0.3116, 0.5893) | 0.0000 |
| controlled_vs_proposed_raw | sentence_score | 0.2031 | (0.1656, 0.2406) | 0.0000 | 0.2031 | (0.1375, 0.2625) | 0.0000 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0005 | (-0.0151, 0.0157) | 0.4673 | 0.0005 | (-0.0330, 0.0275) | 0.4653 |
| controlled_vs_proposed_raw | overall_quality | 0.1515 | (0.1343, 0.1682) | 0.0000 | 0.1515 | (0.1283, 0.1775) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2056 | (0.1907, 0.2207) | 0.0000 | 0.2056 | (0.1914, 0.2197) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1929 | (0.1577, 0.2302) | 0.0000 | 0.1929 | (0.1360, 0.2513) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.1012 | (0.0821, 0.1199) | 0.0000 | 0.1012 | (0.0539, 0.1455) | 0.0000 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.2699 | (0.2500, 0.2910) | 0.0000 | 0.2699 | (0.2489, 0.2879) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0554 | (0.0493, 0.0619) | 0.0000 | 0.0554 | (0.0484, 0.0648) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2191 | (0.1786, 0.2627) | 0.0000 | 0.2191 | (0.1535, 0.2933) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.0881 | (0.0491, 0.1292) | 0.0000 | 0.0881 | (0.0117, 0.1906) | 0.0057 |
| controlled_vs_candidate_no_context | distinct1 | 0.0010 | (-0.0072, 0.0088) | 0.4003 | 0.0010 | (-0.0140, 0.0175) | 0.4467 |
| controlled_vs_candidate_no_context | length_score | 0.4101 | (0.3324, 0.4860) | 0.0000 | 0.4101 | (0.2420, 0.5697) | 0.0000 |
| controlled_vs_candidate_no_context | sentence_score | 0.1875 | (0.1469, 0.2250) | 0.0000 | 0.1875 | (0.1062, 0.2656) | 0.0000 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0369 | (0.0244, 0.0497) | 0.0000 | 0.0369 | (0.0105, 0.0588) | 0.0040 |
| controlled_vs_candidate_no_context | overall_quality | 0.1690 | (0.1535, 0.1842) | 0.0000 | 0.1690 | (0.1443, 0.1930) | 0.0000 |
| controlled_vs_baseline_no_context | context_relevance | 0.1891 | (0.1759, 0.2031) | 0.0000 | 0.1891 | (0.1730, 0.2102) | 0.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1742 | (0.1463, 0.2039) | 0.0000 | 0.1742 | (0.1167, 0.2347) | 0.0000 |
| controlled_vs_baseline_no_context | naturalness | 0.0265 | (0.0095, 0.0427) | 0.0003 | 0.0265 | (-0.0025, 0.0541) | 0.0383 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.2476 | (0.2282, 0.2664) | 0.0000 | 0.2476 | (0.2258, 0.2780) | 0.0000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0525 | (0.0464, 0.0589) | 0.0000 | 0.0525 | (0.0455, 0.0617) | 0.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2165 | (0.1823, 0.2513) | 0.0000 | 0.2165 | (0.1461, 0.2931) | 0.0000 |
| controlled_vs_baseline_no_context | persona_style | 0.0053 | (-0.0160, 0.0268) | 0.2943 | 0.0053 | (-0.0202, 0.0323) | 0.3660 |
| controlled_vs_baseline_no_context | distinct1 | -0.0392 | (-0.0470, -0.0304) | 1.0000 | -0.0392 | (-0.0499, -0.0275) | 1.0000 |
| controlled_vs_baseline_no_context | length_score | 0.1646 | (0.0938, 0.2345) | 0.0000 | 0.1646 | (0.0256, 0.2890) | 0.0130 |
| controlled_vs_baseline_no_context | sentence_score | 0.0871 | (0.0491, 0.1246) | 0.0000 | 0.0871 | (0.0281, 0.1589) | 0.0007 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0186 | (0.0058, 0.0314) | 0.0033 | 0.0186 | (-0.0007, 0.0375) | 0.0303 |
| controlled_vs_baseline_no_context | overall_quality | 0.1414 | (0.1306, 0.1525) | 0.0000 | 0.1414 | (0.1226, 0.1613) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.1891 | (0.1754, 0.2037) | 0.0000 | 0.1891 | (0.1729, 0.2093) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1742 | (0.1446, 0.2043) | 0.0000 | 0.1742 | (0.1166, 0.2333) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 0.0265 | (0.0093, 0.0428) | 0.0013 | 0.0265 | (-0.0028, 0.0543) | 0.0420 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.2476 | (0.2290, 0.2669) | 0.0000 | 0.2476 | (0.2240, 0.2793) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0525 | (0.0463, 0.0590) | 0.0000 | 0.0525 | (0.0452, 0.0614) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2165 | (0.1829, 0.2537) | 0.0000 | 0.2165 | (0.1472, 0.2881) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0053 | (-0.0163, 0.0253) | 0.3073 | 0.0053 | (-0.0202, 0.0323) | 0.3500 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0392 | (-0.0473, -0.0304) | 1.0000 | -0.0392 | (-0.0501, -0.0273) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.1646 | (0.0931, 0.2336) | 0.0000 | 0.1646 | (0.0256, 0.2851) | 0.0133 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0871 | (0.0500, 0.1250) | 0.0000 | 0.0871 | (0.0281, 0.1585) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0186 | (0.0056, 0.0315) | 0.0027 | 0.0186 | (-0.0013, 0.0376) | 0.0313 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1414 | (0.1308, 0.1520) | 0.0000 | 0.1414 | (0.1225, 0.1627) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 46 | 18 | 48 | 0.6250 | 0.7188 |
| proposed_vs_candidate_no_context | persona_consistency | 24 | 24 | 64 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | naturalness | 27 | 37 | 48 | 0.4554 | 0.4219 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 36 | 12 | 64 | 0.6071 | 0.7500 |
| proposed_vs_candidate_no_context | context_overlap | 46 | 18 | 48 | 0.6250 | 0.7188 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 18 | 19 | 75 | 0.4955 | 0.4865 |
| proposed_vs_candidate_no_context | persona_style | 12 | 9 | 91 | 0.5134 | 0.5714 |
| proposed_vs_candidate_no_context | distinct1 | 23 | 36 | 53 | 0.4420 | 0.3898 |
| proposed_vs_candidate_no_context | length_score | 25 | 38 | 49 | 0.4420 | 0.3968 |
| proposed_vs_candidate_no_context | sentence_score | 15 | 20 | 77 | 0.4777 | 0.4286 |
| proposed_vs_candidate_no_context | bertscore_f1 | 66 | 20 | 26 | 0.7054 | 0.7674 |
| proposed_vs_candidate_no_context | overall_quality | 59 | 27 | 26 | 0.6429 | 0.6860 |
| proposed_vs_baseline_no_context | context_relevance | 58 | 52 | 2 | 0.5268 | 0.5273 |
| proposed_vs_baseline_no_context | persona_consistency | 22 | 44 | 46 | 0.4018 | 0.3333 |
| proposed_vs_baseline_no_context | naturalness | 19 | 93 | 0 | 0.1696 | 0.1696 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 36 | 31 | 45 | 0.5223 | 0.5373 |
| proposed_vs_baseline_no_context | context_overlap | 67 | 43 | 2 | 0.6071 | 0.6091 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 17 | 31 | 64 | 0.4375 | 0.3542 |
| proposed_vs_baseline_no_context | persona_style | 9 | 26 | 77 | 0.4241 | 0.2571 |
| proposed_vs_baseline_no_context | distinct1 | 13 | 86 | 13 | 0.1741 | 0.1313 |
| proposed_vs_baseline_no_context | length_score | 23 | 85 | 4 | 0.2232 | 0.2130 |
| proposed_vs_baseline_no_context | sentence_score | 10 | 47 | 55 | 0.3348 | 0.1754 |
| proposed_vs_baseline_no_context | bertscore_f1 | 63 | 49 | 0 | 0.5625 | 0.5625 |
| proposed_vs_baseline_no_context | overall_quality | 39 | 73 | 0 | 0.3482 | 0.3482 |
| controlled_vs_proposed_raw | context_relevance | 99 | 13 | 0 | 0.8839 | 0.8839 |
| controlled_vs_proposed_raw | persona_consistency | 99 | 6 | 7 | 0.9152 | 0.9429 |
| controlled_vs_proposed_raw | naturalness | 96 | 16 | 0 | 0.8571 | 0.8571 |
| controlled_vs_proposed_raw | context_keyword_coverage | 97 | 10 | 5 | 0.8884 | 0.9065 |
| controlled_vs_proposed_raw | context_overlap | 93 | 19 | 0 | 0.8304 | 0.8304 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 98 | 3 | 11 | 0.9241 | 0.9703 |
| controlled_vs_proposed_raw | persona_style | 27 | 8 | 77 | 0.5848 | 0.7714 |
| controlled_vs_proposed_raw | distinct1 | 69 | 38 | 5 | 0.6384 | 0.6449 |
| controlled_vs_proposed_raw | length_score | 91 | 17 | 4 | 0.8304 | 0.8426 |
| controlled_vs_proposed_raw | sentence_score | 69 | 4 | 39 | 0.7902 | 0.9452 |
| controlled_vs_proposed_raw | bertscore_f1 | 69 | 43 | 0 | 0.6161 | 0.6161 |
| controlled_vs_proposed_raw | overall_quality | 104 | 8 | 0 | 0.9286 | 0.9286 |
| controlled_vs_candidate_no_context | context_relevance | 110 | 2 | 0 | 0.9821 | 0.9821 |
| controlled_vs_candidate_no_context | persona_consistency | 96 | 10 | 6 | 0.8839 | 0.9057 |
| controlled_vs_candidate_no_context | naturalness | 88 | 24 | 0 | 0.7857 | 0.7857 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 108 | 0 | 4 | 0.9821 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 106 | 6 | 0 | 0.9464 | 0.9464 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 94 | 9 | 9 | 0.8795 | 0.9126 |
| controlled_vs_candidate_no_context | persona_style | 31 | 7 | 74 | 0.6071 | 0.8158 |
| controlled_vs_candidate_no_context | distinct1 | 62 | 45 | 5 | 0.5759 | 0.5794 |
| controlled_vs_candidate_no_context | length_score | 87 | 24 | 1 | 0.7812 | 0.7838 |
| controlled_vs_candidate_no_context | sentence_score | 67 | 7 | 38 | 0.7679 | 0.9054 |
| controlled_vs_candidate_no_context | bertscore_f1 | 88 | 24 | 0 | 0.7857 | 0.7857 |
| controlled_vs_candidate_no_context | overall_quality | 106 | 6 | 0 | 0.9464 | 0.9464 |
| controlled_vs_baseline_no_context | context_relevance | 110 | 2 | 0 | 0.9821 | 0.9821 |
| controlled_vs_baseline_no_context | persona_consistency | 93 | 9 | 10 | 0.8750 | 0.9118 |
| controlled_vs_baseline_no_context | naturalness | 71 | 41 | 0 | 0.6339 | 0.6339 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 110 | 0 | 2 | 0.9911 | 1.0000 |
| controlled_vs_baseline_no_context | context_overlap | 109 | 3 | 0 | 0.9732 | 0.9732 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 89 | 4 | 19 | 0.8795 | 0.9570 |
| controlled_vs_baseline_no_context | persona_style | 14 | 10 | 88 | 0.5179 | 0.5833 |
| controlled_vs_baseline_no_context | distinct1 | 14 | 97 | 1 | 0.1295 | 0.1261 |
| controlled_vs_baseline_no_context | length_score | 79 | 29 | 4 | 0.7232 | 0.7315 |
| controlled_vs_baseline_no_context | sentence_score | 35 | 8 | 69 | 0.6205 | 0.8140 |
| controlled_vs_baseline_no_context | bertscore_f1 | 71 | 41 | 0 | 0.6339 | 0.6339 |
| controlled_vs_baseline_no_context | overall_quality | 112 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 110 | 2 | 0 | 0.9821 | 0.9821 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 93 | 9 | 10 | 0.8750 | 0.9118 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 71 | 41 | 0 | 0.6339 | 0.6339 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 110 | 0 | 2 | 0.9911 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 109 | 3 | 0 | 0.9732 | 0.9732 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 89 | 4 | 19 | 0.8795 | 0.9570 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 14 | 10 | 88 | 0.5179 | 0.5833 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | 14 | 97 | 1 | 0.1295 | 0.1261 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 79 | 29 | 4 | 0.7232 | 0.7315 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 35 | 8 | 69 | 0.6205 | 0.8140 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 71 | 41 | 0 | 0.6339 | 0.6339 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 112 | 0 | 0 | 1.0000 | 1.0000 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.7054 | 0.1339 | 0.8661 |
| proposed_contextual | 0.0000 | 0.0000 | 0.5357 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.4911 | 0.0000 | 0.0000 |
| baseline_no_context | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `8`
- Unique template signatures: `71`
- Template signature ratio: `0.6339`
- Effective sample size by source clustering: `8.00`
- Effective sample size by template-signature clustering: `56.00`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: enabled.

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
# Proposal Alignment Evaluation Report

- Run ID: `20260305T223517Z`
- Generated: `2026-03-05T22:40:58.809858+00:00`
- Scenarios: `artifacts\proposal_control_tuning\preflight_hybrid\20260305T222831Z\seed_runs\seed_31\20260305T223517Z\scenarios.jsonl`
- Scenario count: `24`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_hb`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off
- `baseline_no_context`: model `phi3:mini`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2514 (0.2139, 0.2883) | 0.3302 (0.2779, 0.4011) | 0.8999 (0.8760, 0.9210) | 0.3744 (0.3554, 0.3993) | 0.0812 |
| proposed_contextual_controlled_hb | 0.2836 (0.2308, 0.3351) | 0.3422 (0.2778, 0.4211) | 0.8756 (0.8494, 0.9003) | 0.3891 (0.3630, 0.4175) | 0.0936 |
| proposed_contextual | 0.0797 (0.0403, 0.1267) | 0.1414 (0.1016, 0.1850) | 0.8208 (0.7893, 0.8516) | 0.2272 (0.1974, 0.2617) | 0.0674 |
| candidate_no_context | 0.0237 (0.0137, 0.0348) | 0.1535 (0.1088, 0.2058) | 0.8344 (0.7986, 0.8690) | 0.2081 (0.1873, 0.2319) | 0.0395 |
| baseline_no_context | 0.0430 (0.0257, 0.0621) | 0.1603 (0.1236, 0.2054) | 0.9031 (0.8804, 0.9250) | 0.2312 (0.2163, 0.2472) | 0.0531 |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0560 | 2.3602 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0121 | -0.0788 |
| proposed_vs_candidate_no_context | naturalness | -0.0135 | -0.0162 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0685 | 3.6167 |
| proposed_vs_candidate_no_context | context_overlap | 0.0267 | 0.7668 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0179 | -0.2857 |
| proposed_vs_candidate_no_context | persona_style | 0.0109 | 0.0211 |
| proposed_vs_candidate_no_context | distinct1 | -0.0047 | -0.0050 |
| proposed_vs_candidate_no_context | length_score | -0.0583 | -0.1448 |
| proposed_vs_candidate_no_context | sentence_score | -0.0146 | -0.0190 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0279 | 0.7050 |
| proposed_vs_candidate_no_context | overall_quality | 0.0190 | 0.0914 |
| proposed_vs_baseline_no_context | context_relevance | 0.0367 | 0.8520 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0189 | -0.1177 |
| proposed_vs_baseline_no_context | naturalness | -0.0823 | -0.0911 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0419 | 0.9183 |
| proposed_vs_baseline_no_context | context_overlap | 0.0245 | 0.6617 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0030 | -0.0625 |
| proposed_vs_baseline_no_context | persona_style | -0.0825 | -0.1349 |
| proposed_vs_baseline_no_context | distinct1 | -0.0361 | -0.0369 |
| proposed_vs_baseline_no_context | length_score | -0.2736 | -0.4427 |
| proposed_vs_baseline_no_context | sentence_score | -0.1312 | -0.1486 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0143 | 0.2692 |
| proposed_vs_baseline_no_context | overall_quality | -0.0040 | -0.0173 |
| controlled_vs_proposed_raw | context_relevance | 0.1717 | 2.1549 |
| controlled_vs_proposed_raw | persona_consistency | 0.1887 | 1.3342 |
| controlled_vs_proposed_raw | naturalness | 0.0790 | 0.0963 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2271 | 2.5971 |
| controlled_vs_proposed_raw | context_overlap | 0.0425 | 0.6898 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2153 | 4.8222 |
| controlled_vs_proposed_raw | persona_style | 0.0825 | 0.1560 |
| controlled_vs_proposed_raw | distinct1 | -0.0067 | -0.0071 |
| controlled_vs_proposed_raw | length_score | 0.3153 | 0.9153 |
| controlled_vs_proposed_raw | sentence_score | 0.1896 | 0.2521 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0138 | 0.2051 |
| controlled_vs_proposed_raw | overall_quality | 0.1473 | 0.6484 |
| controlled_vs_candidate_no_context | context_relevance | 0.2277 | 9.6013 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1766 | 1.1502 |
| controlled_vs_candidate_no_context | naturalness | 0.0655 | 0.0785 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.2956 | 15.6067 |
| controlled_vs_candidate_no_context | context_overlap | 0.0692 | 1.9856 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1974 | 3.1587 |
| controlled_vs_candidate_no_context | persona_style | 0.0934 | 0.1804 |
| controlled_vs_candidate_no_context | distinct1 | -0.0114 | -0.0121 |
| controlled_vs_candidate_no_context | length_score | 0.2569 | 0.6379 |
| controlled_vs_candidate_no_context | sentence_score | 0.1750 | 0.2283 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0417 | 1.0548 |
| controlled_vs_candidate_no_context | overall_quality | 0.1663 | 0.7990 |
| controlled_vs_baseline_no_context | context_relevance | 0.2084 | 4.8430 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1698 | 1.0594 |
| controlled_vs_baseline_no_context | naturalness | -0.0033 | -0.0036 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.2689 | 5.9003 |
| controlled_vs_baseline_no_context | context_overlap | 0.0670 | 1.8080 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2123 | 4.4583 |
| controlled_vs_baseline_no_context | persona_style | 0.0000 | 0.0000 |
| controlled_vs_baseline_no_context | distinct1 | -0.0429 | -0.0438 |
| controlled_vs_baseline_no_context | length_score | 0.0417 | 0.0674 |
| controlled_vs_baseline_no_context | sentence_score | 0.0583 | 0.0660 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0281 | 0.5295 |
| controlled_vs_baseline_no_context | overall_quality | 0.1433 | 0.6198 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0322 | 0.1281 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0121 | 0.0366 |
| controlled_alt_vs_controlled_default | naturalness | -0.0243 | -0.0270 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0459 | 0.1459 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0002 | 0.0021 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0190 | 0.0733 |
| controlled_alt_vs_controlled_default | persona_style | -0.0158 | -0.0258 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0101 | 0.0108 |
| controlled_alt_vs_controlled_default | length_score | -0.1139 | -0.1726 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0583 | -0.0619 |
| controlled_alt_vs_controlled_default | bertscore_f1 | 0.0124 | 0.1530 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0147 | 0.0392 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.2039 | 2.5590 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.2008 | 1.4196 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0548 | 0.0667 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2730 | 3.1220 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0427 | 0.6933 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.2343 | 5.2489 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0667 | 0.1261 |
| controlled_alt_vs_proposed_raw | distinct1 | 0.0034 | 0.0036 |
| controlled_alt_vs_proposed_raw | length_score | 0.2014 | 0.5847 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1312 | 0.1745 |
| controlled_alt_vs_proposed_raw | bertscore_f1 | 0.0262 | 0.3896 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1619 | 0.7129 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2599 | 10.9590 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1887 | 1.2289 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0412 | 0.0494 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3415 | 18.0300 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0694 | 1.9917 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.2165 | 3.4635 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0776 | 0.1499 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0013 | -0.0014 |
| controlled_alt_vs_candidate_no_context | length_score | 0.1431 | 0.3552 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1167 | 0.1522 |
| controlled_alt_vs_candidate_no_context | bertscore_f1 | 0.0541 | 1.3693 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1810 | 0.8694 |
| controlled_alt_vs_baseline_no_context | context_relevance | 0.2405 | 5.5912 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 0.1819 | 1.1348 |
| controlled_alt_vs_baseline_no_context | naturalness | -0.0276 | -0.0305 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 0.3148 | 6.9072 |
| controlled_alt_vs_baseline_no_context | context_overlap | 0.0672 | 1.8137 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 0.2313 | 4.8583 |
| controlled_alt_vs_baseline_no_context | persona_style | -0.0158 | -0.0258 |
| controlled_alt_vs_baseline_no_context | distinct1 | -0.0328 | -0.0335 |
| controlled_alt_vs_baseline_no_context | length_score | -0.0722 | -0.1169 |
| controlled_alt_vs_baseline_no_context | sentence_score | 0.0000 | 0.0000 |
| controlled_alt_vs_baseline_no_context | bertscore_f1 | 0.0405 | 0.7636 |
| controlled_alt_vs_baseline_no_context | overall_quality | 0.1579 | 0.6833 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2084 | 4.8430 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1698 | 1.0594 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0033 | -0.0036 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.2689 | 5.9003 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0670 | 1.8080 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2123 | 4.4583 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0000 | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0429 | -0.0438 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.0417 | 0.0674 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0583 | 0.0660 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0281 | 0.5295 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1433 | 0.6198 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0560 | (0.0147, 0.1013) | 0.0023 | 0.0560 | (0.0119, 0.1162) | 0.0000 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0121 | (-0.0864, 0.0540) | 0.6500 | -0.0121 | (-0.0545, 0.0333) | 0.7033 |
| proposed_vs_candidate_no_context | naturalness | -0.0135 | (-0.0599, 0.0337) | 0.7110 | -0.0135 | (-0.0550, 0.0240) | 0.7677 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0685 | (0.0152, 0.1288) | 0.0030 | 0.0685 | (0.0109, 0.1473) | 0.0227 |
| proposed_vs_candidate_no_context | context_overlap | 0.0267 | (0.0092, 0.0471) | 0.0020 | 0.0267 | (0.0106, 0.0490) | 0.0000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0179 | (-0.0963, 0.0625) | 0.6960 | -0.0179 | (-0.0605, 0.0152) | 0.8600 |
| proposed_vs_candidate_no_context | persona_style | 0.0109 | (-0.0672, 0.0892) | 0.3763 | 0.0109 | (-0.0924, 0.1264) | 0.4747 |
| proposed_vs_candidate_no_context | distinct1 | -0.0047 | (-0.0281, 0.0187) | 0.6463 | -0.0047 | (-0.0247, 0.0167) | 0.6953 |
| proposed_vs_candidate_no_context | length_score | -0.0583 | (-0.2292, 0.1070) | 0.7483 | -0.0583 | (-0.2348, 0.0798) | 0.7683 |
| proposed_vs_candidate_no_context | sentence_score | -0.0146 | (-0.1021, 0.0729) | 0.6780 | -0.0146 | (-0.0833, 0.0457) | 0.7423 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0279 | (-0.0003, 0.0561) | 0.0260 | 0.0279 | (0.0080, 0.0503) | 0.0017 |
| proposed_vs_candidate_no_context | overall_quality | 0.0190 | (-0.0223, 0.0601) | 0.1937 | 0.0190 | (-0.0190, 0.0614) | 0.1903 |
| proposed_vs_baseline_no_context | context_relevance | 0.0367 | (-0.0046, 0.0826) | 0.0407 | 0.0367 | (-0.0094, 0.0915) | 0.0627 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0189 | (-0.0682, 0.0195) | 0.7820 | -0.0189 | (-0.0815, 0.0198) | 0.7637 |
| proposed_vs_baseline_no_context | naturalness | -0.0823 | (-0.1153, -0.0476) | 1.0000 | -0.0823 | (-0.1279, -0.0456) | 1.0000 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0419 | (-0.0114, 0.0989) | 0.0627 | 0.0419 | (-0.0223, 0.1136) | 0.1110 |
| proposed_vs_baseline_no_context | context_overlap | 0.0245 | (0.0080, 0.0438) | 0.0013 | 0.0245 | (0.0056, 0.0452) | 0.0047 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0030 | (-0.0645, 0.0417) | 0.5233 | -0.0030 | (-0.0737, 0.0345) | 0.5197 |
| proposed_vs_baseline_no_context | persona_style | -0.0825 | (-0.1649, -0.0139) | 1.0000 | -0.0825 | (-0.2313, 0.0000) | 1.0000 |
| proposed_vs_baseline_no_context | distinct1 | -0.0361 | (-0.0544, -0.0176) | 1.0000 | -0.0361 | (-0.0541, -0.0191) | 1.0000 |
| proposed_vs_baseline_no_context | length_score | -0.2736 | (-0.3931, -0.1528) | 1.0000 | -0.2736 | (-0.4463, -0.1402) | 1.0000 |
| proposed_vs_baseline_no_context | sentence_score | -0.1313 | (-0.2042, -0.0437) | 1.0000 | -0.1313 | (-0.2587, -0.0241) | 1.0000 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0143 | (-0.0179, 0.0511) | 0.2070 | 0.0143 | (-0.0001, 0.0360) | 0.0267 |
| proposed_vs_baseline_no_context | overall_quality | -0.0040 | (-0.0346, 0.0253) | 0.5890 | -0.0040 | (-0.0406, 0.0286) | 0.6127 |
| controlled_vs_proposed_raw | context_relevance | 0.1717 | (0.1115, 0.2290) | 0.0000 | 0.1717 | (0.1153, 0.2270) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.1887 | (0.1438, 0.2469) | 0.0000 | 0.1887 | (0.1349, 0.2731) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0790 | (0.0469, 0.1118) | 0.0000 | 0.0790 | (0.0354, 0.1298) | 0.0000 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2271 | (0.1519, 0.2918) | 0.0000 | 0.2271 | (0.1493, 0.2998) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0425 | (0.0133, 0.0722) | 0.0040 | 0.0425 | (0.0138, 0.0747) | 0.0007 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2153 | (0.1599, 0.2869) | 0.0000 | 0.2153 | (0.1568, 0.3193) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.0825 | (0.0069, 0.1654) | 0.0130 | 0.0825 | (0.0000, 0.2423) | 0.0993 |
| controlled_vs_proposed_raw | distinct1 | -0.0067 | (-0.0229, 0.0099) | 0.7887 | -0.0067 | (-0.0274, 0.0207) | 0.7060 |
| controlled_vs_proposed_raw | length_score | 0.3153 | (0.1847, 0.4472) | 0.0000 | 0.3153 | (0.1360, 0.4879) | 0.0000 |
| controlled_vs_proposed_raw | sentence_score | 0.1896 | (0.1021, 0.2625) | 0.0000 | 0.1896 | (0.0903, 0.2975) | 0.0000 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0138 | (-0.0109, 0.0385) | 0.1417 | 0.0138 | (-0.0066, 0.0352) | 0.0737 |
| controlled_vs_proposed_raw | overall_quality | 0.1473 | (0.1094, 0.1840) | 0.0000 | 0.1473 | (0.1083, 0.1946) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2277 | (0.1892, 0.2682) | 0.0000 | 0.2277 | (0.1829, 0.2729) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1766 | (0.1080, 0.2447) | 0.0000 | 0.1766 | (0.1318, 0.2372) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.0655 | (0.0227, 0.1065) | 0.0010 | 0.0655 | (0.0225, 0.1056) | 0.0023 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.2956 | (0.2419, 0.3486) | 0.0000 | 0.2956 | (0.2348, 0.3621) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0692 | (0.0507, 0.0915) | 0.0000 | 0.0692 | (0.0491, 0.0933) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1974 | (0.1206, 0.2824) | 0.0000 | 0.1974 | (0.1512, 0.2608) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.0934 | (0.0153, 0.1806) | 0.0070 | 0.0934 | (-0.0035, 0.2256) | 0.0323 |
| controlled_vs_candidate_no_context | distinct1 | -0.0114 | (-0.0354, 0.0118) | 0.8347 | -0.0114 | (-0.0309, 0.0098) | 0.8633 |
| controlled_vs_candidate_no_context | length_score | 0.2569 | (0.1097, 0.4042) | 0.0007 | 0.2569 | (0.0803, 0.4111) | 0.0020 |
| controlled_vs_candidate_no_context | sentence_score | 0.1750 | (0.0729, 0.2625) | 0.0000 | 0.1750 | (0.0955, 0.2587) | 0.0000 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0417 | (0.0169, 0.0636) | 0.0013 | 0.0417 | (0.0192, 0.0665) | 0.0000 |
| controlled_vs_candidate_no_context | overall_quality | 0.1663 | (0.1371, 0.1933) | 0.0000 | 0.1663 | (0.1392, 0.1935) | 0.0000 |
| controlled_vs_baseline_no_context | context_relevance | 0.2084 | (0.1676, 0.2467) | 0.0000 | 0.2084 | (0.1637, 0.2556) | 0.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1698 | (0.1139, 0.2301) | 0.0000 | 0.1698 | (0.0987, 0.2514) | 0.0000 |
| controlled_vs_baseline_no_context | naturalness | -0.0033 | (-0.0302, 0.0243) | 0.5920 | -0.0033 | (-0.0386, 0.0287) | 0.5533 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.2689 | (0.2121, 0.3232) | 0.0000 | 0.2689 | (0.2009, 0.3273) | 0.0000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0670 | (0.0454, 0.0931) | 0.0000 | 0.0670 | (0.0414, 0.0983) | 0.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2123 | (0.1486, 0.2877) | 0.0000 | 0.2123 | (0.1240, 0.3145) | 0.0000 |
| controlled_vs_baseline_no_context | persona_style | 0.0000 | (-0.0208, 0.0208) | 0.4707 | 0.0000 | (0.0000, 0.0000) | 0.3457 |
| controlled_vs_baseline_no_context | distinct1 | -0.0429 | (-0.0612, -0.0254) | 1.0000 | -0.0429 | (-0.0591, -0.0270) | 1.0000 |
| controlled_vs_baseline_no_context | length_score | 0.0417 | (-0.0833, 0.1764) | 0.2687 | 0.0417 | (-0.1384, 0.1959) | 0.3193 |
| controlled_vs_baseline_no_context | sentence_score | 0.0583 | (0.0146, 0.1167) | 0.0147 | 0.0583 | (0.0000, 0.1333) | 0.0253 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0281 | (0.0024, 0.0542) | 0.0147 | 0.0281 | (0.0110, 0.0503) | 0.0000 |
| controlled_vs_baseline_no_context | overall_quality | 0.1433 | (0.1174, 0.1701) | 0.0000 | 0.1433 | (0.1090, 0.1796) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0322 | (-0.0194, 0.0790) | 0.0987 | 0.0322 | (-0.0101, 0.0706) | 0.0717 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0121 | (-0.0363, 0.0806) | 0.3980 | 0.0121 | (-0.0329, 0.0625) | 0.3167 |
| controlled_alt_vs_controlled_default | naturalness | -0.0243 | (-0.0531, 0.0017) | 0.9680 | -0.0243 | (-0.0527, 0.0056) | 0.9470 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0459 | (-0.0227, 0.1057) | 0.0897 | 0.0459 | (-0.0113, 0.0943) | 0.0587 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0002 | (-0.0240, 0.0202) | 0.4847 | 0.0002 | (-0.0241, 0.0178) | 0.4817 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0190 | (-0.0403, 0.1060) | 0.3453 | 0.0190 | (-0.0331, 0.0834) | 0.2580 |
| controlled_alt_vs_controlled_default | persona_style | -0.0158 | (-0.0487, 0.0139) | 0.8733 | -0.0158 | (-0.0407, -0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0101 | (-0.0076, 0.0292) | 0.1473 | 0.0101 | (-0.0101, 0.0344) | 0.1880 |
| controlled_alt_vs_controlled_default | length_score | -0.1139 | (-0.2403, 0.0042) | 0.9727 | -0.1139 | (-0.2433, 0.0307) | 0.9377 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0583 | (-0.1313, 0.0146) | 0.9733 | -0.0583 | (-0.1203, 0.0146) | 0.9703 |
| controlled_alt_vs_controlled_default | bertscore_f1 | 0.0124 | (-0.0170, 0.0399) | 0.1943 | 0.0124 | (-0.0197, 0.0396) | 0.2240 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0147 | (-0.0139, 0.0480) | 0.1900 | 0.0147 | (-0.0106, 0.0365) | 0.1420 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.2039 | (0.1434, 0.2622) | 0.0000 | 0.2039 | (0.1076, 0.2767) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.2008 | (0.1350, 0.2766) | 0.0000 | 0.2008 | (0.1377, 0.2758) | 0.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0548 | (0.0156, 0.0937) | 0.0027 | 0.0548 | (0.0050, 0.1039) | 0.0173 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2730 | (0.1922, 0.3485) | 0.0000 | 0.2730 | (0.1529, 0.3692) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0427 | (0.0195, 0.0636) | 0.0000 | 0.0427 | (0.0105, 0.0693) | 0.0047 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.2343 | (0.1526, 0.3278) | 0.0000 | 0.2343 | (0.1621, 0.3261) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0667 | (-0.0199, 0.1719) | 0.0787 | 0.0667 | (-0.0253, 0.2192) | 0.1257 |
| controlled_alt_vs_proposed_raw | distinct1 | 0.0034 | (-0.0132, 0.0208) | 0.3420 | 0.0034 | (-0.0137, 0.0308) | 0.3757 |
| controlled_alt_vs_proposed_raw | length_score | 0.2014 | (0.0431, 0.3583) | 0.0053 | 0.2014 | (-0.0000, 0.3786) | 0.0260 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1313 | (0.0437, 0.2188) | 0.0053 | 0.1313 | (0.0167, 0.2450) | 0.0137 |
| controlled_alt_vs_proposed_raw | bertscore_f1 | 0.0262 | (-0.0052, 0.0597) | 0.0553 | 0.0262 | (-0.0171, 0.0616) | 0.1087 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1619 | (0.1233, 0.2041) | 0.0000 | 0.1619 | (0.1091, 0.2092) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2599 | (0.2049, 0.3151) | 0.0000 | 0.2599 | (0.1818, 0.3223) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1887 | (0.1130, 0.2760) | 0.0000 | 0.1887 | (0.1452, 0.2450) | 0.0000 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0412 | (-0.0042, 0.0857) | 0.0377 | 0.0412 | (-0.0080, 0.0817) | 0.0517 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3415 | (0.2657, 0.4132) | 0.0000 | 0.3415 | (0.2364, 0.4275) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0694 | (0.0522, 0.0857) | 0.0000 | 0.0694 | (0.0520, 0.0854) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.2165 | (0.1258, 0.3268) | 0.0000 | 0.2165 | (0.1634, 0.2869) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0776 | (0.0025, 0.1638) | 0.0203 | 0.0776 | (-0.0100, 0.1968) | 0.0467 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0013 | (-0.0246, 0.0224) | 0.5440 | -0.0013 | (-0.0260, 0.0243) | 0.5273 |
| controlled_alt_vs_candidate_no_context | length_score | 0.1431 | (-0.0375, 0.3139) | 0.0637 | 0.1431 | (-0.0727, 0.3120) | 0.0903 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1167 | (0.0146, 0.2188) | 0.0193 | 0.1167 | (0.0437, 0.2042) | 0.0007 |
| controlled_alt_vs_candidate_no_context | bertscore_f1 | 0.0541 | (0.0194, 0.0872) | 0.0020 | 0.0541 | (0.0228, 0.0840) | 0.0003 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1810 | (0.1428, 0.2184) | 0.0000 | 0.1810 | (0.1443, 0.2115) | 0.0000 |
| controlled_alt_vs_baseline_no_context | context_relevance | 0.2405 | (0.1881, 0.2922) | 0.0000 | 0.2405 | (0.1681, 0.2998) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 0.1819 | (0.1186, 0.2572) | 0.0000 | 0.1819 | (0.1176, 0.2450) | 0.0000 |
| controlled_alt_vs_baseline_no_context | naturalness | -0.0276 | (-0.0609, 0.0045) | 0.9513 | -0.0276 | (-0.0692, 0.0053) | 0.9470 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 0.3148 | (0.2456, 0.3827) | 0.0000 | 0.3148 | (0.2165, 0.3995) | 0.0000 |
| controlled_alt_vs_baseline_no_context | context_overlap | 0.0672 | (0.0512, 0.0834) | 0.0000 | 0.0672 | (0.0468, 0.0847) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 0.2313 | (0.1512, 0.3258) | 0.0000 | 0.2313 | (0.1528, 0.3207) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_style | -0.0158 | (-0.0603, 0.0278) | 0.7700 | -0.0158 | (-0.0432, 0.0000) | 1.0000 |
| controlled_alt_vs_baseline_no_context | distinct1 | -0.0328 | (-0.0504, -0.0126) | 0.9997 | -0.0328 | (-0.0504, -0.0123) | 0.9980 |
| controlled_alt_vs_baseline_no_context | length_score | -0.0722 | (-0.2236, 0.0750) | 0.8340 | -0.0722 | (-0.2815, 0.0945) | 0.7863 |
| controlled_alt_vs_baseline_no_context | sentence_score | 0.0000 | (-0.0729, 0.0729) | 0.5637 | 0.0000 | (-0.0667, 0.0955) | 0.5677 |
| controlled_alt_vs_baseline_no_context | bertscore_f1 | 0.0405 | (0.0104, 0.0730) | 0.0040 | 0.0405 | (0.0136, 0.0681) | 0.0017 |
| controlled_alt_vs_baseline_no_context | overall_quality | 0.1579 | (0.1242, 0.1915) | 0.0000 | 0.1579 | (0.1114, 0.1899) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2084 | (0.1658, 0.2482) | 0.0000 | 0.2084 | (0.1627, 0.2545) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1698 | (0.1151, 0.2314) | 0.0000 | 0.1698 | (0.0983, 0.2517) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0033 | (-0.0298, 0.0240) | 0.6097 | -0.0033 | (-0.0386, 0.0291) | 0.5557 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.2689 | (0.2121, 0.3232) | 0.0000 | 0.2689 | (0.2034, 0.3306) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0670 | (0.0455, 0.0938) | 0.0000 | 0.0670 | (0.0427, 0.0984) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2123 | (0.1458, 0.2889) | 0.0000 | 0.2123 | (0.1272, 0.3137) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0000 | (-0.0208, 0.0208) | 0.4597 | 0.0000 | (0.0000, 0.0000) | 0.3663 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0429 | (-0.0616, -0.0243) | 1.0000 | -0.0429 | (-0.0588, -0.0266) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.0417 | (-0.0917, 0.1778) | 0.2670 | 0.0417 | (-0.1300, 0.2012) | 0.3173 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0583 | (0.0146, 0.1167) | 0.0187 | 0.0583 | (0.0121, 0.1333) | 0.0247 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0281 | (0.0020, 0.0532) | 0.0157 | 0.0281 | (0.0111, 0.0516) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1433 | (0.1175, 0.1708) | 0.0000 | 0.1433 | (0.1076, 0.1767) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 12 | 6 | 6 | 0.6250 | 0.6667 |
| proposed_vs_candidate_no_context | persona_consistency | 6 | 7 | 11 | 0.4792 | 0.4615 |
| proposed_vs_candidate_no_context | naturalness | 8 | 10 | 6 | 0.4583 | 0.4444 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 9 | 3 | 12 | 0.6250 | 0.7500 |
| proposed_vs_candidate_no_context | context_overlap | 13 | 5 | 6 | 0.6667 | 0.7222 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 4 | 6 | 14 | 0.4583 | 0.4000 |
| proposed_vs_candidate_no_context | persona_style | 3 | 2 | 19 | 0.5208 | 0.6000 |
| proposed_vs_candidate_no_context | distinct1 | 7 | 9 | 8 | 0.4583 | 0.4375 |
| proposed_vs_candidate_no_context | length_score | 9 | 9 | 6 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | sentence_score | 5 | 6 | 13 | 0.4792 | 0.4545 |
| proposed_vs_candidate_no_context | bertscore_f1 | 12 | 6 | 6 | 0.6250 | 0.6667 |
| proposed_vs_candidate_no_context | overall_quality | 9 | 9 | 6 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context | context_relevance | 11 | 11 | 2 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context | persona_consistency | 3 | 5 | 16 | 0.4583 | 0.3750 |
| proposed_vs_baseline_no_context | naturalness | 2 | 21 | 1 | 0.1042 | 0.0870 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 8 | 6 | 10 | 0.5417 | 0.5714 |
| proposed_vs_baseline_no_context | context_overlap | 14 | 8 | 2 | 0.6250 | 0.6364 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 3 | 1 | 20 | 0.5417 | 0.7500 |
| proposed_vs_baseline_no_context | persona_style | 0 | 4 | 20 | 0.4167 | 0.0000 |
| proposed_vs_baseline_no_context | distinct1 | 4 | 17 | 3 | 0.2292 | 0.1905 |
| proposed_vs_baseline_no_context | length_score | 3 | 20 | 1 | 0.1458 | 0.1304 |
| proposed_vs_baseline_no_context | sentence_score | 1 | 10 | 13 | 0.3125 | 0.0909 |
| proposed_vs_baseline_no_context | bertscore_f1 | 12 | 12 | 0 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context | overall_quality | 10 | 14 | 0 | 0.4167 | 0.4167 |
| controlled_vs_proposed_raw | context_relevance | 22 | 2 | 0 | 0.9167 | 0.9167 |
| controlled_vs_proposed_raw | persona_consistency | 22 | 0 | 2 | 0.9583 | 1.0000 |
| controlled_vs_proposed_raw | naturalness | 20 | 4 | 0 | 0.8333 | 0.8333 |
| controlled_vs_proposed_raw | context_keyword_coverage | 20 | 1 | 3 | 0.8958 | 0.9524 |
| controlled_vs_proposed_raw | context_overlap | 20 | 4 | 0 | 0.8333 | 0.8333 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 22 | 0 | 2 | 0.9583 | 1.0000 |
| controlled_vs_proposed_raw | persona_style | 4 | 0 | 20 | 0.5833 | 1.0000 |
| controlled_vs_proposed_raw | distinct1 | 7 | 16 | 1 | 0.3125 | 0.3043 |
| controlled_vs_proposed_raw | length_score | 19 | 5 | 0 | 0.7917 | 0.7917 |
| controlled_vs_proposed_raw | sentence_score | 14 | 1 | 9 | 0.7708 | 0.9333 |
| controlled_vs_proposed_raw | bertscore_f1 | 15 | 9 | 0 | 0.6250 | 0.6250 |
| controlled_vs_proposed_raw | overall_quality | 22 | 2 | 0 | 0.9167 | 0.9167 |
| controlled_vs_candidate_no_context | context_relevance | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 21 | 2 | 1 | 0.8958 | 0.9130 |
| controlled_vs_candidate_no_context | naturalness | 15 | 9 | 0 | 0.6250 | 0.6250 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 23 | 0 | 1 | 0.9792 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 21 | 2 | 1 | 0.8958 | 0.9130 |
| controlled_vs_candidate_no_context | persona_style | 7 | 1 | 16 | 0.6250 | 0.8750 |
| controlled_vs_candidate_no_context | distinct1 | 9 | 15 | 0 | 0.3750 | 0.3750 |
| controlled_vs_candidate_no_context | length_score | 16 | 8 | 0 | 0.6667 | 0.6667 |
| controlled_vs_candidate_no_context | sentence_score | 14 | 2 | 8 | 0.7500 | 0.8750 |
| controlled_vs_candidate_no_context | bertscore_f1 | 20 | 4 | 0 | 0.8333 | 0.8333 |
| controlled_vs_candidate_no_context | overall_quality | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_vs_baseline_no_context | context_relevance | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_vs_baseline_no_context | naturalness | 13 | 11 | 0 | 0.5417 | 0.5417 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 22 | 0 | 2 | 0.9583 | 1.0000 |
| controlled_vs_baseline_no_context | context_overlap | 22 | 2 | 0 | 0.9167 | 0.9167 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_vs_baseline_no_context | persona_style | 1 | 1 | 22 | 0.5000 | 0.5000 |
| controlled_vs_baseline_no_context | distinct1 | 2 | 20 | 2 | 0.1250 | 0.0909 |
| controlled_vs_baseline_no_context | length_score | 14 | 9 | 1 | 0.6042 | 0.6087 |
| controlled_vs_baseline_no_context | sentence_score | 4 | 0 | 20 | 0.5833 | 1.0000 |
| controlled_vs_baseline_no_context | bertscore_f1 | 14 | 10 | 0 | 0.5833 | 0.5833 |
| controlled_vs_baseline_no_context | overall_quality | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 16 | 5 | 3 | 0.7292 | 0.7619 |
| controlled_alt_vs_controlled_default | persona_consistency | 4 | 6 | 14 | 0.4583 | 0.4000 |
| controlled_alt_vs_controlled_default | naturalness | 6 | 15 | 3 | 0.3125 | 0.2857 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 12 | 4 | 8 | 0.6667 | 0.7500 |
| controlled_alt_vs_controlled_default | context_overlap | 13 | 8 | 3 | 0.6042 | 0.6190 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 3 | 4 | 17 | 0.4792 | 0.4286 |
| controlled_alt_vs_controlled_default | persona_style | 1 | 3 | 20 | 0.4583 | 0.2500 |
| controlled_alt_vs_controlled_default | distinct1 | 11 | 9 | 4 | 0.5417 | 0.5500 |
| controlled_alt_vs_controlled_default | length_score | 8 | 12 | 4 | 0.4167 | 0.4000 |
| controlled_alt_vs_controlled_default | sentence_score | 1 | 5 | 18 | 0.4167 | 0.1667 |
| controlled_alt_vs_controlled_default | bertscore_f1 | 12 | 10 | 2 | 0.5417 | 0.5455 |
| controlled_alt_vs_controlled_default | overall_quality | 12 | 10 | 2 | 0.5417 | 0.5455 |
| controlled_alt_vs_proposed_raw | context_relevance | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_alt_vs_proposed_raw | persona_consistency | 20 | 0 | 4 | 0.9167 | 1.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 17 | 7 | 0 | 0.7083 | 0.7083 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 22 | 1 | 1 | 0.9375 | 0.9565 |
| controlled_alt_vs_proposed_raw | context_overlap | 20 | 4 | 0 | 0.8333 | 0.8333 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 20 | 0 | 4 | 0.9167 | 1.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 3 | 2 | 19 | 0.5208 | 0.6000 |
| controlled_alt_vs_proposed_raw | distinct1 | 10 | 11 | 3 | 0.4792 | 0.4762 |
| controlled_alt_vs_proposed_raw | length_score | 16 | 8 | 0 | 0.6667 | 0.6667 |
| controlled_alt_vs_proposed_raw | sentence_score | 11 | 2 | 11 | 0.6875 | 0.8462 |
| controlled_alt_vs_proposed_raw | bertscore_f1 | 14 | 10 | 0 | 0.5833 | 0.5833 |
| controlled_alt_vs_proposed_raw | overall_quality | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_alt_vs_candidate_no_context | context_relevance | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 19 | 2 | 3 | 0.8542 | 0.9048 |
| controlled_alt_vs_candidate_no_context | naturalness | 15 | 9 | 0 | 0.6250 | 0.6250 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 21 | 0 | 3 | 0.9375 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 19 | 2 | 3 | 0.8542 | 0.9048 |
| controlled_alt_vs_candidate_no_context | persona_style | 6 | 2 | 16 | 0.5833 | 0.7500 |
| controlled_alt_vs_candidate_no_context | distinct1 | 11 | 11 | 2 | 0.5000 | 0.5000 |
| controlled_alt_vs_candidate_no_context | length_score | 15 | 9 | 0 | 0.6250 | 0.6250 |
| controlled_alt_vs_candidate_no_context | sentence_score | 12 | 4 | 8 | 0.6667 | 0.7500 |
| controlled_alt_vs_candidate_no_context | bertscore_f1 | 18 | 6 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_candidate_no_context | overall_quality | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_alt_vs_baseline_no_context | context_relevance | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 22 | 1 | 1 | 0.9375 | 0.9565 |
| controlled_alt_vs_baseline_no_context | naturalness | 11 | 13 | 0 | 0.4583 | 0.4583 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 21 | 0 | 3 | 0.9375 | 1.0000 |
| controlled_alt_vs_baseline_no_context | context_overlap | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 22 | 1 | 1 | 0.9375 | 0.9565 |
| controlled_alt_vs_baseline_no_context | persona_style | 1 | 3 | 20 | 0.4583 | 0.2500 |
| controlled_alt_vs_baseline_no_context | distinct1 | 3 | 19 | 2 | 0.1667 | 0.1364 |
| controlled_alt_vs_baseline_no_context | length_score | 10 | 14 | 0 | 0.4167 | 0.4167 |
| controlled_alt_vs_baseline_no_context | sentence_score | 3 | 3 | 18 | 0.5000 | 0.5000 |
| controlled_alt_vs_baseline_no_context | bertscore_f1 | 18 | 6 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_baseline_no_context | overall_quality | 23 | 1 | 0 | 0.9583 | 0.9583 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 24 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 23 | 1 | 0 | 0.9583 | 0.9583 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 13 | 11 | 0 | 0.5417 | 0.5417 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 22 | 0 | 2 | 0.9583 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 22 | 2 | 0 | 0.9167 | 0.9167 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 23 | 1 | 0 | 0.9583 | 0.9583 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 1 | 1 | 22 | 0.5000 | 0.5000 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | 2 | 20 | 2 | 0.1250 | 0.0909 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 14 | 9 | 1 | 0.6042 | 0.6087 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 4 | 0 | 20 | 0.5833 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 14 | 10 | 0 | 0.5833 | 0.5833 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 24 | 0 | 0 | 1.0000 | 1.0000 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.3750 | 0.3333 | 0.6667 |
| proposed_contextual_controlled_hb | 0.0000 | 0.0000 | 0.1667 | 0.5833 | 0.4167 |
| proposed_contextual | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.4583 | 0.0000 | 0.0000 |
| baseline_no_context | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `8`
- Unique template signatures: `23`
- Template signature ratio: `0.9583`
- Effective sample size by source clustering: `7.20`
- Effective sample size by template-signature clustering: `22.15`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: enabled.

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
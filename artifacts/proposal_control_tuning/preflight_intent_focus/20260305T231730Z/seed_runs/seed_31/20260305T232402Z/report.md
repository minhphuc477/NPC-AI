# Proposal Alignment Evaluation Report

- Run ID: `20260305T232402Z`
- Generated: `2026-03-05T23:29:58.111433+00:00`
- Scenarios: `artifacts\proposal_control_tuning\preflight_intent_focus\20260305T231730Z\seed_runs\seed_31\20260305T232402Z\scenarios.jsonl`
- Scenario count: `24`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_alt`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off
- `baseline_no_context`: model `phi3:mini`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2881 (0.2399, 0.3386) | 0.3448 (0.3007, 0.3947) | 0.8789 (0.8479, 0.9067) | 0.3908 (0.3740, 0.4091) | 0.0848 |
| proposed_contextual_controlled_alt | 0.2906 (0.2430, 0.3371) | 0.3130 (0.2805, 0.3490) | 0.8834 (0.8492, 0.9162) | 0.3835 (0.3660, 0.4029) | 0.0959 |
| proposed_contextual | 0.0583 (0.0267, 0.0979) | 0.1645 (0.1064, 0.2361) | 0.8007 (0.7760, 0.8275) | 0.2201 (0.1948, 0.2490) | 0.0453 |
| candidate_no_context | 0.0277 (0.0155, 0.0422) | 0.1593 (0.1113, 0.2146) | 0.8201 (0.7913, 0.8517) | 0.2097 (0.1894, 0.2337) | 0.0482 |
| baseline_no_context | 0.0429 (0.0272, 0.0606) | 0.1716 (0.1269, 0.2222) | 0.9010 (0.8758, 0.9257) | 0.2326 (0.2168, 0.2495) | 0.0348 |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0306 | 1.1038 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0051 | 0.0322 |
| proposed_vs_candidate_no_context | naturalness | -0.0195 | -0.0237 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0407 | 1.8090 |
| proposed_vs_candidate_no_context | context_overlap | 0.0070 | 0.1765 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0079 | 0.1250 |
| proposed_vs_candidate_no_context | persona_style | -0.0061 | -0.0112 |
| proposed_vs_candidate_no_context | distinct1 | 0.0092 | 0.0098 |
| proposed_vs_candidate_no_context | length_score | -0.0792 | -0.2405 |
| proposed_vs_candidate_no_context | sentence_score | -0.0729 | -0.0916 |
| proposed_vs_candidate_no_context | bertscore_f1 | -0.0030 | -0.0615 |
| proposed_vs_candidate_no_context | overall_quality | 0.0103 | 0.0493 |
| proposed_vs_baseline_no_context | context_relevance | 0.0154 | 0.3580 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0071 | -0.0416 |
| proposed_vs_baseline_no_context | naturalness | -0.1003 | -0.1113 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0176 | 0.3870 |
| proposed_vs_baseline_no_context | context_overlap | 0.0101 | 0.2745 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 0.0030 | 0.0435 |
| proposed_vs_baseline_no_context | persona_style | -0.0476 | -0.0815 |
| proposed_vs_baseline_no_context | distinct1 | -0.0299 | -0.0307 |
| proposed_vs_baseline_no_context | length_score | -0.3542 | -0.5862 |
| proposed_vs_baseline_no_context | sentence_score | -0.1750 | -0.1949 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0105 | 0.3010 |
| proposed_vs_baseline_no_context | overall_quality | -0.0126 | -0.0540 |
| controlled_vs_proposed_raw | context_relevance | 0.2298 | 3.9445 |
| controlled_vs_proposed_raw | persona_consistency | 0.1803 | 1.0962 |
| controlled_vs_proposed_raw | naturalness | 0.0782 | 0.0977 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.3028 | 4.7970 |
| controlled_vs_proposed_raw | context_overlap | 0.0595 | 1.2683 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2022 | 2.8306 |
| controlled_vs_proposed_raw | persona_style | 0.0927 | 0.1727 |
| controlled_vs_proposed_raw | distinct1 | -0.0109 | -0.0115 |
| controlled_vs_proposed_raw | length_score | 0.3278 | 1.3111 |
| controlled_vs_proposed_raw | sentence_score | 0.1750 | 0.2421 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0395 | 0.8722 |
| controlled_vs_proposed_raw | overall_quality | 0.1708 | 0.7761 |
| controlled_vs_candidate_no_context | context_relevance | 0.2604 | 9.4021 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1854 | 1.1637 |
| controlled_vs_candidate_no_context | naturalness | 0.0588 | 0.0716 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3435 | 15.2837 |
| controlled_vs_candidate_no_context | context_overlap | 0.0666 | 1.6687 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2101 | 3.3094 |
| controlled_vs_candidate_no_context | persona_style | 0.0866 | 0.1596 |
| controlled_vs_candidate_no_context | distinct1 | -0.0017 | -0.0018 |
| controlled_vs_candidate_no_context | length_score | 0.2486 | 0.7553 |
| controlled_vs_candidate_no_context | sentence_score | 0.1021 | 0.1283 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0365 | 0.7571 |
| controlled_vs_candidate_no_context | overall_quality | 0.1811 | 0.8637 |
| controlled_vs_baseline_no_context | context_relevance | 0.2452 | 5.7148 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1731 | 1.0089 |
| controlled_vs_baseline_no_context | naturalness | -0.0221 | -0.0245 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3205 | 7.0402 |
| controlled_vs_baseline_no_context | context_overlap | 0.0696 | 1.8911 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2052 | 2.9971 |
| controlled_vs_baseline_no_context | persona_style | 0.0451 | 0.0771 |
| controlled_vs_baseline_no_context | distinct1 | -0.0408 | -0.0418 |
| controlled_vs_baseline_no_context | length_score | -0.0264 | -0.0437 |
| controlled_vs_baseline_no_context | sentence_score | 0.0000 | 0.0000 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0500 | 1.4358 |
| controlled_vs_baseline_no_context | overall_quality | 0.1582 | 0.6802 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0025 | 0.0086 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0318 | -0.0922 |
| controlled_alt_vs_controlled_default | naturalness | 0.0045 | 0.0051 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0013 | 0.0035 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0053 | 0.0501 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0345 | -0.1262 |
| controlled_alt_vs_controlled_default | persona_style | -0.0208 | -0.0331 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0089 | 0.0095 |
| controlled_alt_vs_controlled_default | length_score | -0.0125 | -0.0216 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0292 | 0.0325 |
| controlled_alt_vs_controlled_default | bertscore_f1 | 0.0111 | 0.1311 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0073 | -0.0187 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.2323 | 3.9872 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1485 | 0.9029 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0827 | 0.1033 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.3041 | 4.8170 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0648 | 1.3820 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1677 | 2.3472 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0718 | 0.1339 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0020 | -0.0021 |
| controlled_alt_vs_proposed_raw | length_score | 0.3153 | 1.2611 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.2042 | 0.2824 |
| controlled_alt_vs_proposed_raw | bertscore_f1 | 0.0506 | 1.1176 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1635 | 0.7428 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2629 | 9.4918 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1536 | 0.9642 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0632 | 0.0771 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3448 | 15.3399 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0719 | 1.8024 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1756 | 2.7656 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0657 | 0.1211 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0072 | 0.0077 |
| controlled_alt_vs_candidate_no_context | length_score | 0.2361 | 0.7173 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1312 | 0.1649 |
| controlled_alt_vs_candidate_no_context | bertscore_f1 | 0.0476 | 0.9874 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1738 | 0.8288 |
| controlled_alt_vs_baseline_no_context | context_relevance | 0.2477 | 5.7727 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 0.1414 | 0.8237 |
| controlled_alt_vs_baseline_no_context | naturalness | -0.0176 | -0.0196 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 0.3217 | 7.0680 |
| controlled_alt_vs_baseline_no_context | context_overlap | 0.0750 | 2.0360 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 0.1706 | 2.4928 |
| controlled_alt_vs_baseline_no_context | persona_style | 0.0242 | 0.0415 |
| controlled_alt_vs_baseline_no_context | distinct1 | -0.0319 | -0.0327 |
| controlled_alt_vs_baseline_no_context | length_score | -0.0389 | -0.0644 |
| controlled_alt_vs_baseline_no_context | sentence_score | 0.0292 | 0.0325 |
| controlled_alt_vs_baseline_no_context | bertscore_f1 | 0.0611 | 1.7550 |
| controlled_alt_vs_baseline_no_context | overall_quality | 0.1509 | 0.6487 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2452 | 5.7148 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1731 | 1.0089 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0221 | -0.0245 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3205 | 7.0402 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0696 | 1.8911 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2052 | 2.9971 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0451 | 0.0771 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0408 | -0.0418 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | -0.0264 | -0.0437 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0000 | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0500 | 1.4358 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1582 | 0.6802 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0306 | (-0.0084, 0.0722) | 0.0650 | 0.0306 | (-0.0035, 0.0645) | 0.0430 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0051 | (-0.0549, 0.0679) | 0.4540 | 0.0051 | (-0.0496, 0.0944) | 0.4910 |
| proposed_vs_candidate_no_context | naturalness | -0.0195 | (-0.0540, 0.0132) | 0.8720 | -0.0195 | (-0.0490, 0.0031) | 0.9490 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0407 | (-0.0098, 0.0947) | 0.0633 | 0.0407 | (-0.0051, 0.0839) | 0.0490 |
| proposed_vs_candidate_no_context | context_overlap | 0.0070 | (-0.0056, 0.0212) | 0.1540 | 0.0070 | (-0.0005, 0.0145) | 0.0367 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0079 | (-0.0694, 0.0833) | 0.4490 | 0.0079 | (-0.0595, 0.1145) | 0.4563 |
| proposed_vs_candidate_no_context | persona_style | -0.0061 | (-0.0769, 0.0429) | 0.5293 | -0.0061 | (-0.0804, 0.0425) | 0.5437 |
| proposed_vs_candidate_no_context | distinct1 | 0.0092 | (-0.0069, 0.0264) | 0.1507 | 0.0092 | (-0.0103, 0.0278) | 0.2100 |
| proposed_vs_candidate_no_context | length_score | -0.0792 | (-0.2167, 0.0639) | 0.8550 | -0.0792 | (-0.1907, 0.0116) | 0.9550 |
| proposed_vs_candidate_no_context | sentence_score | -0.0729 | (-0.1462, 0.0146) | 0.9713 | -0.0729 | (-0.1604, 0.0305) | 0.9423 |
| proposed_vs_candidate_no_context | bertscore_f1 | -0.0030 | (-0.0326, 0.0256) | 0.5667 | -0.0030 | (-0.0457, 0.0324) | 0.5283 |
| proposed_vs_candidate_no_context | overall_quality | 0.0103 | (-0.0213, 0.0436) | 0.2867 | 0.0103 | (-0.0215, 0.0458) | 0.2757 |
| proposed_vs_baseline_no_context | context_relevance | 0.0154 | (-0.0213, 0.0566) | 0.2203 | 0.0154 | (-0.0301, 0.0592) | 0.2703 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0071 | (-0.0489, 0.0371) | 0.6420 | -0.0071 | (-0.0501, 0.0379) | 0.6350 |
| proposed_vs_baseline_no_context | naturalness | -0.1003 | (-0.1395, -0.0573) | 1.0000 | -0.1003 | (-0.1493, -0.0562) | 1.0000 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0176 | (-0.0311, 0.0725) | 0.2673 | 0.0176 | (-0.0473, 0.0775) | 0.3130 |
| proposed_vs_baseline_no_context | context_overlap | 0.0101 | (-0.0032, 0.0237) | 0.0660 | 0.0101 | (-0.0006, 0.0227) | 0.0307 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 0.0030 | (-0.0417, 0.0476) | 0.4340 | 0.0030 | (-0.0420, 0.0544) | 0.4583 |
| proposed_vs_baseline_no_context | persona_style | -0.0476 | (-0.1436, 0.0339) | 0.8647 | -0.0476 | (-0.1994, 0.0478) | 0.7693 |
| proposed_vs_baseline_no_context | distinct1 | -0.0299 | (-0.0515, -0.0079) | 0.9957 | -0.0299 | (-0.0563, -0.0114) | 0.9990 |
| proposed_vs_baseline_no_context | length_score | -0.3542 | (-0.5014, -0.1931) | 1.0000 | -0.3542 | (-0.5467, -0.2073) | 1.0000 |
| proposed_vs_baseline_no_context | sentence_score | -0.1750 | (-0.2625, -0.0729) | 0.9993 | -0.1750 | (-0.2864, -0.0159) | 0.9863 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0105 | (-0.0176, 0.0386) | 0.2280 | 0.0105 | (-0.0197, 0.0400) | 0.2513 |
| proposed_vs_baseline_no_context | overall_quality | -0.0126 | (-0.0359, 0.0114) | 0.8413 | -0.0126 | (-0.0426, 0.0152) | 0.8137 |
| controlled_vs_proposed_raw | context_relevance | 0.2298 | (0.1674, 0.2882) | 0.0000 | 0.2298 | (0.2009, 0.2698) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.1803 | (0.1076, 0.2499) | 0.0000 | 0.1803 | (0.0970, 0.2559) | 0.0003 |
| controlled_vs_proposed_raw | naturalness | 0.0782 | (0.0346, 0.1211) | 0.0000 | 0.0782 | (0.0230, 0.1386) | 0.0027 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.3028 | (0.2273, 0.3785) | 0.0000 | 0.3028 | (0.2625, 0.3598) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0595 | (0.0396, 0.0785) | 0.0000 | 0.0595 | (0.0416, 0.0759) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2022 | (0.1135, 0.2794) | 0.0000 | 0.2022 | (0.0974, 0.2748) | 0.0007 |
| controlled_vs_proposed_raw | persona_style | 0.0927 | (0.0119, 0.1875) | 0.0113 | 0.0927 | (-0.0068, 0.2604) | 0.0993 |
| controlled_vs_proposed_raw | distinct1 | -0.0109 | (-0.0327, 0.0096) | 0.8507 | -0.0109 | (-0.0375, 0.0187) | 0.7690 |
| controlled_vs_proposed_raw | length_score | 0.3278 | (0.1583, 0.4917) | 0.0000 | 0.3278 | (0.1303, 0.5320) | 0.0003 |
| controlled_vs_proposed_raw | sentence_score | 0.1750 | (0.1021, 0.2479) | 0.0000 | 0.1750 | (0.0942, 0.2587) | 0.0000 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0395 | (0.0121, 0.0716) | 0.0020 | 0.0395 | (0.0109, 0.0754) | 0.0037 |
| controlled_vs_proposed_raw | overall_quality | 0.1708 | (0.1424, 0.1970) | 0.0000 | 0.1708 | (0.1404, 0.2053) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2604 | (0.2146, 0.3137) | 0.0000 | 0.2604 | (0.2206, 0.2989) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1854 | (0.1120, 0.2609) | 0.0000 | 0.1854 | (0.1512, 0.2470) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.0588 | (0.0041, 0.1096) | 0.0180 | 0.0588 | (-0.0007, 0.1190) | 0.0263 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3435 | (0.2763, 0.4164) | 0.0000 | 0.3435 | (0.2874, 0.3965) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0666 | (0.0540, 0.0790) | 0.0000 | 0.0666 | (0.0537, 0.0769) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2101 | (0.1200, 0.3004) | 0.0000 | 0.2101 | (0.1796, 0.2567) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.0866 | (0.0185, 0.1730) | 0.0040 | 0.0866 | (0.0136, 0.2132) | 0.0043 |
| controlled_vs_candidate_no_context | distinct1 | -0.0017 | (-0.0265, 0.0229) | 0.5583 | -0.0017 | (-0.0367, 0.0269) | 0.5343 |
| controlled_vs_candidate_no_context | length_score | 0.2486 | (0.0431, 0.4431) | 0.0077 | 0.2486 | (0.0185, 0.4635) | 0.0163 |
| controlled_vs_candidate_no_context | sentence_score | 0.1021 | (0.0000, 0.2042) | 0.0367 | 0.1021 | (-0.0125, 0.2042) | 0.0430 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0365 | (0.0028, 0.0674) | 0.0173 | 0.0365 | (0.0014, 0.0795) | 0.0200 |
| controlled_vs_candidate_no_context | overall_quality | 0.1811 | (0.1536, 0.2067) | 0.0000 | 0.1811 | (0.1585, 0.2062) | 0.0000 |
| controlled_vs_baseline_no_context | context_relevance | 0.2452 | (0.1915, 0.3039) | 0.0000 | 0.2452 | (0.1925, 0.2991) | 0.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1731 | (0.1163, 0.2324) | 0.0000 | 0.1731 | (0.1039, 0.2330) | 0.0000 |
| controlled_vs_baseline_no_context | naturalness | -0.0221 | (-0.0623, 0.0153) | 0.8663 | -0.0221 | (-0.0618, 0.0095) | 0.9030 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3205 | (0.2470, 0.3987) | 0.0000 | 0.3205 | (0.2423, 0.3955) | 0.0000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0696 | (0.0531, 0.0868) | 0.0000 | 0.0696 | (0.0564, 0.0825) | 0.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2052 | (0.1365, 0.2712) | 0.0000 | 0.2052 | (0.1192, 0.2778) | 0.0000 |
| controlled_vs_baseline_no_context | persona_style | 0.0451 | (-0.0061, 0.1110) | 0.0540 | 0.0451 | (0.0086, 0.1077) | 0.0210 |
| controlled_vs_baseline_no_context | distinct1 | -0.0408 | (-0.0629, -0.0186) | 0.9993 | -0.0408 | (-0.0630, -0.0227) | 1.0000 |
| controlled_vs_baseline_no_context | length_score | -0.0264 | (-0.1875, 0.1195) | 0.6313 | -0.0264 | (-0.1767, 0.1120) | 0.6300 |
| controlled_vs_baseline_no_context | sentence_score | 0.0000 | (-0.0875, 0.0875) | 0.5480 | 0.0000 | (-0.0913, 0.1114) | 0.5643 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0500 | (0.0288, 0.0694) | 0.0000 | 0.0500 | (0.0300, 0.0738) | 0.0000 |
| controlled_vs_baseline_no_context | overall_quality | 0.1582 | (0.1402, 0.1776) | 0.0000 | 0.1582 | (0.1390, 0.1762) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0025 | (-0.0623, 0.0654) | 0.4647 | 0.0025 | (-0.0661, 0.0512) | 0.4643 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0318 | (-0.0811, 0.0147) | 0.9107 | -0.0318 | (-0.0731, -0.0057) | 0.9960 |
| controlled_alt_vs_controlled_default | naturalness | 0.0045 | (-0.0349, 0.0467) | 0.4183 | 0.0045 | (-0.0330, 0.0602) | 0.4057 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0013 | (-0.0859, 0.0878) | 0.4737 | 0.0013 | (-0.0909, 0.0663) | 0.4803 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0053 | (-0.0143, 0.0241) | 0.2760 | 0.0053 | (-0.0116, 0.0244) | 0.3027 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0345 | (-0.0915, 0.0179) | 0.8880 | -0.0345 | (-0.0844, -0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | persona_style | -0.0208 | (-0.0625, 0.0166) | 0.8397 | -0.0208 | (-0.0526, -0.0029) | 1.0000 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0089 | (-0.0150, 0.0324) | 0.2253 | 0.0089 | (-0.0174, 0.0407) | 0.2573 |
| controlled_alt_vs_controlled_default | length_score | -0.0125 | (-0.1778, 0.1473) | 0.5770 | -0.0125 | (-0.1522, 0.1930) | 0.5360 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0292 | (-0.0292, 0.0875) | 0.2147 | 0.0292 | (0.0000, 0.0700) | 0.0997 |
| controlled_alt_vs_controlled_default | bertscore_f1 | 0.0111 | (-0.0160, 0.0404) | 0.2220 | 0.0111 | (-0.0113, 0.0325) | 0.1790 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0073 | (-0.0332, 0.0165) | 0.7020 | -0.0073 | (-0.0382, 0.0130) | 0.7193 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.2323 | (0.1695, 0.2873) | 0.0000 | 0.2323 | (0.1693, 0.2731) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1485 | (0.0725, 0.2095) | 0.0000 | 0.1485 | (0.0513, 0.2102) | 0.0057 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0827 | (0.0292, 0.1348) | 0.0007 | 0.0827 | (0.0143, 0.1591) | 0.0050 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.3041 | (0.2222, 0.3758) | 0.0000 | 0.3041 | (0.2198, 0.3572) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0648 | (0.0413, 0.0866) | 0.0000 | 0.0648 | (0.0427, 0.0850) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1677 | (0.0863, 0.2413) | 0.0000 | 0.1677 | (0.0578, 0.2320) | 0.0013 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0718 | (-0.0092, 0.1688) | 0.0487 | 0.0718 | (-0.0292, 0.2330) | 0.1223 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0020 | (-0.0282, 0.0212) | 0.5623 | -0.0020 | (-0.0335, 0.0339) | 0.5093 |
| controlled_alt_vs_proposed_raw | length_score | 0.3153 | (0.0972, 0.5278) | 0.0050 | 0.3153 | (0.0595, 0.6233) | 0.0057 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.2042 | (0.1313, 0.2771) | 0.0000 | 0.2042 | (0.1260, 0.2833) | 0.0000 |
| controlled_alt_vs_proposed_raw | bertscore_f1 | 0.0506 | (0.0164, 0.0803) | 0.0030 | 0.0506 | (0.0097, 0.0951) | 0.0043 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1635 | (0.1270, 0.1949) | 0.0000 | 0.1635 | (0.1128, 0.1981) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2629 | (0.2096, 0.3129) | 0.0000 | 0.2629 | (0.2005, 0.3121) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1536 | (0.0962, 0.2094) | 0.0000 | 0.1536 | (0.1350, 0.1783) | 0.0000 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0632 | (0.0186, 0.1074) | 0.0013 | 0.0632 | (0.0085, 0.1297) | 0.0133 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3448 | (0.2741, 0.4139) | 0.0000 | 0.3448 | (0.2600, 0.4130) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0719 | (0.0530, 0.0899) | 0.0000 | 0.0719 | (0.0552, 0.0863) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1756 | (0.1048, 0.2442) | 0.0000 | 0.1756 | (0.1584, 0.1911) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0657 | (0.0009, 0.1408) | 0.0237 | 0.0657 | (-0.0034, 0.1733) | 0.0400 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0072 | (-0.0149, 0.0275) | 0.2490 | 0.0072 | (-0.0120, 0.0304) | 0.2210 |
| controlled_alt_vs_candidate_no_context | length_score | 0.2361 | (0.0500, 0.4181) | 0.0050 | 0.2361 | (-0.0051, 0.5015) | 0.0267 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1313 | (0.0437, 0.2188) | 0.0047 | 0.1313 | (0.0389, 0.2240) | 0.0040 |
| controlled_alt_vs_candidate_no_context | bertscore_f1 | 0.0476 | (0.0159, 0.0795) | 0.0027 | 0.0476 | (0.0224, 0.0840) | 0.0000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1738 | (0.1409, 0.2010) | 0.0000 | 0.1738 | (0.1486, 0.1960) | 0.0000 |
| controlled_alt_vs_baseline_no_context | context_relevance | 0.2477 | (0.1968, 0.3012) | 0.0000 | 0.2477 | (0.1875, 0.3035) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 0.1414 | (0.0880, 0.1887) | 0.0000 | 0.1414 | (0.0737, 0.1850) | 0.0000 |
| controlled_alt_vs_baseline_no_context | naturalness | -0.0176 | (-0.0537, 0.0158) | 0.8313 | -0.0176 | (-0.0494, 0.0130) | 0.8027 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 0.3217 | (0.2520, 0.3930) | 0.0000 | 0.3217 | (0.2373, 0.4010) | 0.0000 |
| controlled_alt_vs_baseline_no_context | context_overlap | 0.0750 | (0.0558, 0.0952) | 0.0000 | 0.0750 | (0.0596, 0.0903) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 0.1706 | (0.1030, 0.2304) | 0.0000 | 0.1706 | (0.0781, 0.2278) | 0.0013 |
| controlled_alt_vs_baseline_no_context | persona_style | 0.0242 | (-0.0013, 0.0556) | 0.0423 | 0.0242 | (0.0000, 0.0624) | 0.1037 |
| controlled_alt_vs_baseline_no_context | distinct1 | -0.0319 | (-0.0486, -0.0148) | 1.0000 | -0.0319 | (-0.0501, -0.0133) | 1.0000 |
| controlled_alt_vs_baseline_no_context | length_score | -0.0389 | (-0.1875, 0.1069) | 0.6883 | -0.0389 | (-0.1702, 0.1091) | 0.6650 |
| controlled_alt_vs_baseline_no_context | sentence_score | 0.0292 | (-0.0729, 0.1313) | 0.3250 | 0.0292 | (-0.0389, 0.1217) | 0.2913 |
| controlled_alt_vs_baseline_no_context | bertscore_f1 | 0.0611 | (0.0313, 0.0928) | 0.0000 | 0.0611 | (0.0316, 0.0869) | 0.0000 |
| controlled_alt_vs_baseline_no_context | overall_quality | 0.1509 | (0.1242, 0.1760) | 0.0000 | 0.1509 | (0.1118, 0.1748) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2452 | (0.1901, 0.3015) | 0.0000 | 0.2452 | (0.1936, 0.2991) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1731 | (0.1136, 0.2308) | 0.0000 | 0.1731 | (0.1102, 0.2342) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0221 | (-0.0628, 0.0171) | 0.8470 | -0.0221 | (-0.0611, 0.0111) | 0.9023 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3205 | (0.2485, 0.3984) | 0.0000 | 0.3205 | (0.2459, 0.3989) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0696 | (0.0535, 0.0868) | 0.0000 | 0.0696 | (0.0565, 0.0831) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2052 | (0.1341, 0.2754) | 0.0000 | 0.2052 | (0.1193, 0.2793) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0451 | (-0.0061, 0.1110) | 0.0543 | 0.0451 | (0.0086, 0.1030) | 0.0200 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0408 | (-0.0617, -0.0178) | 1.0000 | -0.0408 | (-0.0639, -0.0226) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | -0.0264 | (-0.1973, 0.1223) | 0.6457 | -0.0264 | (-0.1803, 0.1104) | 0.6400 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0000 | (-0.0875, 0.0875) | 0.5623 | 0.0000 | (-0.0913, 0.1105) | 0.5620 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0500 | (0.0287, 0.0688) | 0.0000 | 0.0500 | (0.0293, 0.0740) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1582 | (0.1396, 0.1771) | 0.0000 | 0.1582 | (0.1391, 0.1781) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 7 | 10 | 7 | 0.4375 | 0.4118 |
| proposed_vs_candidate_no_context | persona_consistency | 4 | 5 | 15 | 0.4792 | 0.4444 |
| proposed_vs_candidate_no_context | naturalness | 7 | 10 | 7 | 0.4375 | 0.4118 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 7 | 5 | 12 | 0.5417 | 0.5833 |
| proposed_vs_candidate_no_context | context_overlap | 7 | 10 | 7 | 0.4375 | 0.4118 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 3 | 3 | 18 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | persona_style | 3 | 2 | 19 | 0.5208 | 0.6000 |
| proposed_vs_candidate_no_context | distinct1 | 8 | 7 | 9 | 0.5208 | 0.5333 |
| proposed_vs_candidate_no_context | length_score | 6 | 10 | 8 | 0.4167 | 0.3750 |
| proposed_vs_candidate_no_context | sentence_score | 2 | 7 | 15 | 0.3958 | 0.2222 |
| proposed_vs_candidate_no_context | bertscore_f1 | 9 | 11 | 4 | 0.4583 | 0.4500 |
| proposed_vs_candidate_no_context | overall_quality | 13 | 7 | 4 | 0.6250 | 0.6500 |
| proposed_vs_baseline_no_context | context_relevance | 11 | 13 | 0 | 0.4583 | 0.4583 |
| proposed_vs_baseline_no_context | persona_consistency | 3 | 6 | 15 | 0.4375 | 0.3333 |
| proposed_vs_baseline_no_context | naturalness | 5 | 19 | 0 | 0.2083 | 0.2083 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 6 | 8 | 10 | 0.4583 | 0.4286 |
| proposed_vs_baseline_no_context | context_overlap | 11 | 13 | 0 | 0.4583 | 0.4583 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 2 | 2 | 20 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context | persona_style | 2 | 5 | 17 | 0.4375 | 0.2857 |
| proposed_vs_baseline_no_context | distinct1 | 6 | 14 | 4 | 0.3333 | 0.3000 |
| proposed_vs_baseline_no_context | length_score | 5 | 19 | 0 | 0.2083 | 0.2083 |
| proposed_vs_baseline_no_context | sentence_score | 3 | 15 | 6 | 0.2500 | 0.1667 |
| proposed_vs_baseline_no_context | bertscore_f1 | 12 | 12 | 0 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context | overall_quality | 9 | 15 | 0 | 0.3750 | 0.3750 |
| controlled_vs_proposed_raw | context_relevance | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_vs_proposed_raw | persona_consistency | 22 | 1 | 1 | 0.9375 | 0.9565 |
| controlled_vs_proposed_raw | naturalness | 19 | 5 | 0 | 0.7917 | 0.7917 |
| controlled_vs_proposed_raw | context_keyword_coverage | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_vs_proposed_raw | context_overlap | 21 | 3 | 0 | 0.8750 | 0.8750 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 22 | 1 | 1 | 0.9375 | 0.9565 |
| controlled_vs_proposed_raw | persona_style | 5 | 1 | 18 | 0.5833 | 0.8333 |
| controlled_vs_proposed_raw | distinct1 | 8 | 13 | 3 | 0.3958 | 0.3810 |
| controlled_vs_proposed_raw | length_score | 19 | 4 | 1 | 0.8125 | 0.8261 |
| controlled_vs_proposed_raw | sentence_score | 12 | 0 | 12 | 0.7500 | 1.0000 |
| controlled_vs_proposed_raw | bertscore_f1 | 16 | 8 | 0 | 0.6667 | 0.6667 |
| controlled_vs_proposed_raw | overall_quality | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_relevance | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 21 | 2 | 1 | 0.8958 | 0.9130 |
| controlled_vs_candidate_no_context | naturalness | 15 | 9 | 0 | 0.6250 | 0.6250 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 23 | 0 | 1 | 0.9792 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 21 | 2 | 1 | 0.8958 | 0.9130 |
| controlled_vs_candidate_no_context | persona_style | 5 | 1 | 18 | 0.5833 | 0.8333 |
| controlled_vs_candidate_no_context | distinct1 | 11 | 11 | 2 | 0.5000 | 0.5000 |
| controlled_vs_candidate_no_context | length_score | 17 | 7 | 0 | 0.7083 | 0.7083 |
| controlled_vs_candidate_no_context | sentence_score | 11 | 4 | 9 | 0.6458 | 0.7333 |
| controlled_vs_candidate_no_context | bertscore_f1 | 18 | 6 | 0 | 0.7500 | 0.7500 |
| controlled_vs_candidate_no_context | overall_quality | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | context_relevance | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_vs_baseline_no_context | persona_consistency | 22 | 2 | 0 | 0.9167 | 0.9167 |
| controlled_vs_baseline_no_context | naturalness | 10 | 14 | 0 | 0.4167 | 0.4167 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 23 | 0 | 1 | 0.9792 | 1.0000 |
| controlled_vs_baseline_no_context | context_overlap | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 22 | 2 | 0 | 0.9167 | 0.9167 |
| controlled_vs_baseline_no_context | persona_style | 3 | 1 | 20 | 0.5417 | 0.7500 |
| controlled_vs_baseline_no_context | distinct1 | 6 | 18 | 0 | 0.2500 | 0.2500 |
| controlled_vs_baseline_no_context | length_score | 12 | 11 | 1 | 0.5208 | 0.5217 |
| controlled_vs_baseline_no_context | sentence_score | 5 | 5 | 14 | 0.5000 | 0.5000 |
| controlled_vs_baseline_no_context | bertscore_f1 | 21 | 3 | 0 | 0.8750 | 0.8750 |
| controlled_vs_baseline_no_context | overall_quality | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 12 | 8 | 4 | 0.5833 | 0.6000 |
| controlled_alt_vs_controlled_default | persona_consistency | 4 | 8 | 12 | 0.4167 | 0.3333 |
| controlled_alt_vs_controlled_default | naturalness | 11 | 9 | 4 | 0.5417 | 0.5500 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 9 | 8 | 7 | 0.5208 | 0.5294 |
| controlled_alt_vs_controlled_default | context_overlap | 13 | 7 | 4 | 0.6250 | 0.6500 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 2 | 6 | 16 | 0.4167 | 0.2500 |
| controlled_alt_vs_controlled_default | persona_style | 2 | 3 | 19 | 0.4792 | 0.4000 |
| controlled_alt_vs_controlled_default | distinct1 | 12 | 8 | 4 | 0.5833 | 0.6000 |
| controlled_alt_vs_controlled_default | length_score | 9 | 8 | 7 | 0.5208 | 0.5294 |
| controlled_alt_vs_controlled_default | sentence_score | 3 | 1 | 20 | 0.5417 | 0.7500 |
| controlled_alt_vs_controlled_default | bertscore_f1 | 10 | 10 | 4 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | overall_quality | 9 | 11 | 4 | 0.4583 | 0.4500 |
| controlled_alt_vs_proposed_raw | context_relevance | 22 | 2 | 0 | 0.9167 | 0.9167 |
| controlled_alt_vs_proposed_raw | persona_consistency | 21 | 2 | 1 | 0.8958 | 0.9130 |
| controlled_alt_vs_proposed_raw | naturalness | 16 | 8 | 0 | 0.6667 | 0.6667 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 22 | 2 | 0 | 0.9167 | 0.9167 |
| controlled_alt_vs_proposed_raw | context_overlap | 21 | 3 | 0 | 0.8750 | 0.8750 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 21 | 2 | 1 | 0.8958 | 0.9130 |
| controlled_alt_vs_proposed_raw | persona_style | 5 | 3 | 16 | 0.5417 | 0.6250 |
| controlled_alt_vs_proposed_raw | distinct1 | 13 | 9 | 2 | 0.5833 | 0.5909 |
| controlled_alt_vs_proposed_raw | length_score | 15 | 7 | 2 | 0.6667 | 0.6818 |
| controlled_alt_vs_proposed_raw | sentence_score | 14 | 0 | 10 | 0.7917 | 1.0000 |
| controlled_alt_vs_proposed_raw | bertscore_f1 | 19 | 5 | 0 | 0.7917 | 0.7917 |
| controlled_alt_vs_proposed_raw | overall_quality | 22 | 2 | 0 | 0.9167 | 0.9167 |
| controlled_alt_vs_candidate_no_context | context_relevance | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 21 | 2 | 1 | 0.8958 | 0.9130 |
| controlled_alt_vs_candidate_no_context | naturalness | 15 | 9 | 0 | 0.6250 | 0.6250 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 22 | 0 | 2 | 0.9583 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 21 | 2 | 1 | 0.8958 | 0.9130 |
| controlled_alt_vs_candidate_no_context | persona_style | 5 | 1 | 18 | 0.5833 | 0.8333 |
| controlled_alt_vs_candidate_no_context | distinct1 | 13 | 10 | 1 | 0.5625 | 0.5652 |
| controlled_alt_vs_candidate_no_context | length_score | 15 | 8 | 1 | 0.6458 | 0.6522 |
| controlled_alt_vs_candidate_no_context | sentence_score | 11 | 2 | 11 | 0.6875 | 0.8462 |
| controlled_alt_vs_candidate_no_context | bertscore_f1 | 20 | 4 | 0 | 0.8333 | 0.8333 |
| controlled_alt_vs_candidate_no_context | overall_quality | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_alt_vs_baseline_no_context | context_relevance | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 21 | 2 | 1 | 0.8958 | 0.9130 |
| controlled_alt_vs_baseline_no_context | naturalness | 10 | 14 | 0 | 0.4167 | 0.4167 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 22 | 0 | 2 | 0.9583 | 1.0000 |
| controlled_alt_vs_baseline_no_context | context_overlap | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 21 | 2 | 1 | 0.8958 | 0.9130 |
| controlled_alt_vs_baseline_no_context | persona_style | 3 | 1 | 20 | 0.5417 | 0.7500 |
| controlled_alt_vs_baseline_no_context | distinct1 | 5 | 17 | 2 | 0.2500 | 0.2273 |
| controlled_alt_vs_baseline_no_context | length_score | 12 | 11 | 1 | 0.5208 | 0.5217 |
| controlled_alt_vs_baseline_no_context | sentence_score | 7 | 5 | 12 | 0.5417 | 0.5833 |
| controlled_alt_vs_baseline_no_context | bertscore_f1 | 20 | 4 | 0 | 0.8333 | 0.8333 |
| controlled_alt_vs_baseline_no_context | overall_quality | 24 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 23 | 1 | 0 | 0.9583 | 0.9583 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 22 | 2 | 0 | 0.9167 | 0.9167 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 10 | 14 | 0 | 0.4167 | 0.4167 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 23 | 0 | 1 | 0.9792 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 23 | 1 | 0 | 0.9583 | 0.9583 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 22 | 2 | 0 | 0.9167 | 0.9167 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 3 | 1 | 20 | 0.5417 | 0.7500 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | 6 | 18 | 0 | 0.2500 | 0.2500 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 12 | 11 | 1 | 0.5208 | 0.5217 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 5 | 5 | 14 | 0.5000 | 0.5000 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 21 | 3 | 0 | 0.8750 | 0.8750 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 24 | 0 | 0 | 1.0000 | 1.0000 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.2500 | 0.4167 | 0.5833 |
| proposed_contextual_controlled_alt | 0.0000 | 0.0000 | 0.2917 | 0.5833 | 0.4167 |
| proposed_contextual | 0.0000 | 0.0000 | 0.4583 | 0.0000 | 0.0000 |
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
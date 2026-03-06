# Proposal Alignment Evaluation Report

- Run ID: `20260306T093645Z`
- Generated: `2026-03-06T10:09:41.117857+00:00`
- Scenarios: `artifacts\proposal_control_tuning\preflight_blend_full112\20260306T093644Z\seed_runs\seed_29\20260306T093645Z\scenarios.jsonl`
- Scenario count: `112`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_alt`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off
- `baseline_no_context`: model `phi3:mini`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2913 (0.2725, 0.3108) | 0.3440 (0.3152, 0.3743) | 0.8688 (0.8541, 0.8836) | 0.3916 (0.3795, 0.4038) | 0.0972 |
| proposed_contextual_controlled_alt | 0.2665 (0.2486, 0.2867) | 0.3455 (0.3174, 0.3765) | 0.8589 (0.8453, 0.8728) | 0.3787 (0.3687, 0.3892) | 0.0824 |
| proposed_contextual | 0.0827 (0.0630, 0.1026) | 0.1639 (0.1382, 0.1920) | 0.8025 (0.7900, 0.8172) | 0.2331 (0.2185, 0.2487) | 0.0678 |
| candidate_no_context | 0.0283 (0.0219, 0.0353) | 0.1648 (0.1364, 0.1954) | 0.8062 (0.7928, 0.8218) | 0.2086 (0.1972, 0.2201) | 0.0385 |
| baseline_no_context | 0.0528 (0.0438, 0.0623) | 0.2023 (0.1811, 0.2244) | 0.8837 (0.8726, 0.8949) | 0.2461 (0.2367, 0.2558) | 0.0571 |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0544 | 1.9210 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0009 | -0.0055 |
| proposed_vs_candidate_no_context | naturalness | -0.0038 | -0.0047 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0709 | 3.0530 |
| proposed_vs_candidate_no_context | context_overlap | 0.0159 | 0.3946 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0017 | -0.0219 |
| proposed_vs_candidate_no_context | persona_style | 0.0025 | 0.0049 |
| proposed_vs_candidate_no_context | distinct1 | 0.0045 | 0.0048 |
| proposed_vs_candidate_no_context | length_score | -0.0167 | -0.0594 |
| proposed_vs_candidate_no_context | sentence_score | -0.0219 | -0.0283 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0293 | 0.7599 |
| proposed_vs_candidate_no_context | overall_quality | 0.0245 | 0.1175 |
| proposed_vs_baseline_no_context | context_relevance | 0.0299 | 0.5672 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0384 | -0.1896 |
| proposed_vs_baseline_no_context | naturalness | -0.0812 | -0.0919 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0373 | 0.6557 |
| proposed_vs_baseline_no_context | context_overlap | 0.0128 | 0.2957 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0270 | -0.2574 |
| proposed_vs_baseline_no_context | persona_style | -0.0839 | -0.1417 |
| proposed_vs_baseline_no_context | distinct1 | -0.0420 | -0.0429 |
| proposed_vs_baseline_no_context | length_score | -0.2622 | -0.4986 |
| proposed_vs_baseline_no_context | sentence_score | -0.1188 | -0.1367 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0107 | 0.1868 |
| proposed_vs_baseline_no_context | overall_quality | -0.0131 | -0.0531 |
| controlled_vs_proposed_raw | context_relevance | 0.2086 | 2.5220 |
| controlled_vs_proposed_raw | persona_consistency | 0.1801 | 1.0986 |
| controlled_vs_proposed_raw | naturalness | 0.0664 | 0.0827 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2725 | 2.8941 |
| controlled_vs_proposed_raw | context_overlap | 0.0596 | 1.0635 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2008 | 2.5823 |
| controlled_vs_proposed_raw | persona_style | 0.0971 | 0.1909 |
| controlled_vs_proposed_raw | distinct1 | 0.0014 | 0.0015 |
| controlled_vs_proposed_raw | length_score | 0.2548 | 0.9661 |
| controlled_vs_proposed_raw | sentence_score | 0.1509 | 0.2012 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0294 | 0.4337 |
| controlled_vs_proposed_raw | overall_quality | 0.1585 | 0.6801 |
| controlled_vs_candidate_no_context | context_relevance | 0.2630 | 9.2876 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1792 | 1.0871 |
| controlled_vs_candidate_no_context | naturalness | 0.0626 | 0.0777 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3434 | 14.7828 |
| controlled_vs_candidate_no_context | context_overlap | 0.0755 | 1.8777 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1991 | 2.5037 |
| controlled_vs_candidate_no_context | persona_style | 0.0996 | 0.1967 |
| controlled_vs_candidate_no_context | distinct1 | 0.0059 | 0.0063 |
| controlled_vs_candidate_no_context | length_score | 0.2381 | 0.8493 |
| controlled_vs_candidate_no_context | sentence_score | 0.1290 | 0.1671 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0587 | 1.5232 |
| controlled_vs_candidate_no_context | overall_quality | 0.1830 | 0.8774 |
| controlled_vs_baseline_no_context | context_relevance | 0.2385 | 4.5196 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1417 | 0.7007 |
| controlled_vs_baseline_no_context | naturalness | -0.0148 | -0.0168 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3097 | 5.4475 |
| controlled_vs_baseline_no_context | context_overlap | 0.0724 | 1.6738 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1739 | 1.6602 |
| controlled_vs_baseline_no_context | persona_style | 0.0131 | 0.0222 |
| controlled_vs_baseline_no_context | distinct1 | -0.0406 | -0.0415 |
| controlled_vs_baseline_no_context | length_score | -0.0074 | -0.0141 |
| controlled_vs_baseline_no_context | sentence_score | 0.0321 | 0.0370 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0401 | 0.7015 |
| controlled_vs_baseline_no_context | overall_quality | 0.1454 | 0.5908 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0248 | -0.0851 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0015 | 0.0044 |
| controlled_alt_vs_controlled_default | naturalness | -0.0099 | -0.0114 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0316 | -0.0861 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0090 | -0.0779 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0054 | 0.0194 |
| controlled_alt_vs_controlled_default | persona_style | -0.0140 | -0.0232 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0059 | -0.0063 |
| controlled_alt_vs_controlled_default | length_score | -0.0381 | -0.0735 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0022 | 0.0025 |
| controlled_alt_vs_controlled_default | bertscore_f1 | -0.0147 | -0.1516 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0128 | -0.0328 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1838 | 2.2222 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1816 | 1.1078 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0564 | 0.0703 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2409 | 2.5588 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0506 | 0.9028 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.2062 | 2.6517 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0831 | 0.1633 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0045 | -0.0048 |
| controlled_alt_vs_proposed_raw | length_score | 0.2167 | 0.8217 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1531 | 0.2042 |
| controlled_alt_vs_proposed_raw | bertscore_f1 | 0.0147 | 0.2163 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1457 | 0.6250 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2382 | 8.4120 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1807 | 1.0963 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0527 | 0.0653 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3118 | 13.4240 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0665 | 1.6535 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.2045 | 2.5717 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0855 | 0.1690 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0000 | 0.0000 |
| controlled_alt_vs_candidate_no_context | length_score | 0.2000 | 0.7134 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1313 | 0.1700 |
| controlled_alt_vs_candidate_no_context | bertscore_f1 | 0.0439 | 1.1405 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1702 | 0.8159 |
| controlled_alt_vs_baseline_no_context | context_relevance | 0.2137 | 4.0498 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 0.1432 | 0.7081 |
| controlled_alt_vs_baseline_no_context | naturalness | -0.0248 | -0.0280 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 0.2782 | 4.8925 |
| controlled_alt_vs_baseline_no_context | context_overlap | 0.0634 | 1.4655 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 0.1793 | 1.7117 |
| controlled_alt_vs_baseline_no_context | persona_style | -0.0009 | -0.0015 |
| controlled_alt_vs_baseline_no_context | distinct1 | -0.0465 | -0.0475 |
| controlled_alt_vs_baseline_no_context | length_score | -0.0455 | -0.0866 |
| controlled_alt_vs_baseline_no_context | sentence_score | 0.0344 | 0.0396 |
| controlled_alt_vs_baseline_no_context | bertscore_f1 | 0.0253 | 0.4435 |
| controlled_alt_vs_baseline_no_context | overall_quality | 0.1326 | 0.5387 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2385 | 4.5196 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1417 | 0.7007 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0148 | -0.0168 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3097 | 5.4475 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0724 | 1.6738 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1739 | 1.6602 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0131 | 0.0222 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0406 | -0.0415 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | -0.0074 | -0.0141 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0321 | 0.0370 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0401 | 0.7015 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1454 | 0.5908 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0544 | (0.0360, 0.0726) | 0.0000 | 0.0544 | (0.0265, 0.0842) | 0.0000 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0009 | (-0.0269, 0.0253) | 0.5253 | -0.0009 | (-0.0292, 0.0178) | 0.4920 |
| proposed_vs_candidate_no_context | naturalness | -0.0038 | (-0.0185, 0.0113) | 0.6803 | -0.0038 | (-0.0203, 0.0120) | 0.6890 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0709 | (0.0466, 0.0972) | 0.0000 | 0.0709 | (0.0369, 0.1112) | 0.0000 |
| proposed_vs_candidate_no_context | context_overlap | 0.0159 | (0.0096, 0.0227) | 0.0000 | 0.0159 | (0.0073, 0.0246) | 0.0000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0017 | (-0.0328, 0.0265) | 0.5420 | -0.0017 | (-0.0332, 0.0185) | 0.5330 |
| proposed_vs_candidate_no_context | persona_style | 0.0025 | (-0.0254, 0.0272) | 0.4253 | 0.0025 | (-0.0257, 0.0355) | 0.4577 |
| proposed_vs_candidate_no_context | distinct1 | 0.0045 | (-0.0027, 0.0120) | 0.1170 | 0.0045 | (-0.0024, 0.0109) | 0.1003 |
| proposed_vs_candidate_no_context | length_score | -0.0167 | (-0.0759, 0.0432) | 0.6990 | -0.0167 | (-0.0837, 0.0461) | 0.6883 |
| proposed_vs_candidate_no_context | sentence_score | -0.0219 | (-0.0531, 0.0094) | 0.9173 | -0.0219 | (-0.0500, 0.0032) | 0.9543 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0293 | (0.0180, 0.0410) | 0.0000 | 0.0293 | (0.0105, 0.0537) | 0.0000 |
| proposed_vs_candidate_no_context | overall_quality | 0.0245 | (0.0102, 0.0393) | 0.0003 | 0.0245 | (0.0101, 0.0413) | 0.0000 |
| proposed_vs_baseline_no_context | context_relevance | 0.0299 | (0.0087, 0.0520) | 0.0033 | 0.0299 | (-0.0102, 0.0718) | 0.0797 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0384 | (-0.0652, -0.0109) | 0.9977 | -0.0384 | (-0.0873, 0.0096) | 0.9447 |
| proposed_vs_baseline_no_context | naturalness | -0.0812 | (-0.0989, -0.0638) | 1.0000 | -0.0812 | (-0.1028, -0.0581) | 1.0000 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0373 | (0.0099, 0.0660) | 0.0023 | 0.0373 | (-0.0189, 0.0933) | 0.0987 |
| proposed_vs_baseline_no_context | context_overlap | 0.0128 | (0.0056, 0.0206) | 0.0000 | 0.0128 | (0.0024, 0.0226) | 0.0097 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0270 | (-0.0551, 0.0018) | 0.9680 | -0.0270 | (-0.0654, 0.0158) | 0.8973 |
| proposed_vs_baseline_no_context | persona_style | -0.0839 | (-0.1291, -0.0434) | 1.0000 | -0.0839 | (-0.2209, 0.0205) | 0.9033 |
| proposed_vs_baseline_no_context | distinct1 | -0.0420 | (-0.0510, -0.0325) | 1.0000 | -0.0420 | (-0.0551, -0.0251) | 1.0000 |
| proposed_vs_baseline_no_context | length_score | -0.2622 | (-0.3307, -0.1943) | 1.0000 | -0.2622 | (-0.3369, -0.1723) | 1.0000 |
| proposed_vs_baseline_no_context | sentence_score | -0.1187 | (-0.1594, -0.0781) | 1.0000 | -0.1187 | (-0.2000, -0.0375) | 0.9980 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0107 | (-0.0031, 0.0247) | 0.0590 | 0.0107 | (-0.0164, 0.0371) | 0.2157 |
| proposed_vs_baseline_no_context | overall_quality | -0.0131 | (-0.0303, 0.0044) | 0.9333 | -0.0131 | (-0.0493, 0.0205) | 0.7683 |
| controlled_vs_proposed_raw | context_relevance | 0.2086 | (0.1803, 0.2361) | 0.0000 | 0.2086 | (0.1795, 0.2388) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.1801 | (0.1467, 0.2156) | 0.0000 | 0.1801 | (0.1340, 0.2301) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0664 | (0.0456, 0.0865) | 0.0000 | 0.0664 | (0.0286, 0.1000) | 0.0000 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2725 | (0.2358, 0.3088) | 0.0000 | 0.2725 | (0.2345, 0.3126) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0596 | (0.0486, 0.0710) | 0.0000 | 0.0596 | (0.0480, 0.0719) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2008 | (0.1610, 0.2440) | 0.0000 | 0.2008 | (0.1524, 0.2564) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.0971 | (0.0570, 0.1393) | 0.0000 | 0.0971 | (-0.0000, 0.2268) | 0.0253 |
| controlled_vs_proposed_raw | distinct1 | 0.0014 | (-0.0076, 0.0105) | 0.3743 | 0.0014 | (-0.0140, 0.0150) | 0.4313 |
| controlled_vs_proposed_raw | length_score | 0.2548 | (0.1684, 0.3381) | 0.0000 | 0.2548 | (0.1036, 0.3905) | 0.0003 |
| controlled_vs_proposed_raw | sentence_score | 0.1509 | (0.1112, 0.1911) | 0.0000 | 0.1509 | (0.1098, 0.1893) | 0.0000 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0294 | (0.0154, 0.0437) | 0.0000 | 0.0294 | (0.0084, 0.0501) | 0.0023 |
| controlled_vs_proposed_raw | overall_quality | 0.1585 | (0.1368, 0.1783) | 0.0000 | 0.1585 | (0.1277, 0.1920) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2630 | (0.2448, 0.2820) | 0.0000 | 0.2630 | (0.2454, 0.2813) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1792 | (0.1438, 0.2160) | 0.0000 | 0.1792 | (0.1193, 0.2351) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.0626 | (0.0413, 0.0837) | 0.0000 | 0.0626 | (0.0144, 0.1042) | 0.0067 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3434 | (0.3187, 0.3677) | 0.0000 | 0.3434 | (0.3186, 0.3674) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0755 | (0.0663, 0.0863) | 0.0000 | 0.0755 | (0.0694, 0.0820) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1991 | (0.1571, 0.2443) | 0.0000 | 0.1991 | (0.1368, 0.2608) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.0996 | (0.0606, 0.1411) | 0.0000 | 0.0996 | (0.0024, 0.2196) | 0.0240 |
| controlled_vs_candidate_no_context | distinct1 | 0.0059 | (-0.0036, 0.0145) | 0.1120 | 0.0059 | (-0.0090, 0.0207) | 0.2110 |
| controlled_vs_candidate_no_context | length_score | 0.2381 | (0.1536, 0.3238) | 0.0000 | 0.2381 | (0.0520, 0.3932) | 0.0037 |
| controlled_vs_candidate_no_context | sentence_score | 0.1290 | (0.0862, 0.1705) | 0.0000 | 0.1290 | (0.0723, 0.1848) | 0.0000 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0587 | (0.0449, 0.0723) | 0.0000 | 0.0587 | (0.0362, 0.0804) | 0.0000 |
| controlled_vs_candidate_no_context | overall_quality | 0.1830 | (0.1677, 0.1985) | 0.0000 | 0.1830 | (0.1590, 0.2088) | 0.0000 |
| controlled_vs_baseline_no_context | context_relevance | 0.2385 | (0.2190, 0.2601) | 0.0000 | 0.2385 | (0.2155, 0.2589) | 0.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1417 | (0.1095, 0.1738) | 0.0000 | 0.1417 | (0.1094, 0.1719) | 0.0000 |
| controlled_vs_baseline_no_context | naturalness | -0.0148 | (-0.0343, 0.0036) | 0.9443 | -0.0148 | (-0.0436, 0.0132) | 0.8540 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3097 | (0.2834, 0.3352) | 0.0000 | 0.3097 | (0.2788, 0.3380) | 0.0000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0724 | (0.0634, 0.0826) | 0.0000 | 0.0724 | (0.0671, 0.0779) | 0.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1739 | (0.1365, 0.2123) | 0.0000 | 0.1739 | (0.1301, 0.2117) | 0.0000 |
| controlled_vs_baseline_no_context | persona_style | 0.0131 | (-0.0096, 0.0364) | 0.1247 | 0.0131 | (-0.0124, 0.0423) | 0.1823 |
| controlled_vs_baseline_no_context | distinct1 | -0.0406 | (-0.0481, -0.0326) | 1.0000 | -0.0406 | (-0.0473, -0.0342) | 1.0000 |
| controlled_vs_baseline_no_context | length_score | -0.0074 | (-0.0890, 0.0729) | 0.5747 | -0.0074 | (-0.1524, 0.1194) | 0.5297 |
| controlled_vs_baseline_no_context | sentence_score | 0.0321 | (-0.0121, 0.0759) | 0.0683 | 0.0321 | (-0.0407, 0.1081) | 0.2137 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0401 | (0.0262, 0.0536) | 0.0000 | 0.0401 | (0.0232, 0.0569) | 0.0000 |
| controlled_vs_baseline_no_context | overall_quality | 0.1454 | (0.1325, 0.1587) | 0.0000 | 0.1454 | (0.1316, 0.1568) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0248 | (-0.0514, 0.0031) | 0.9637 | -0.0248 | (-0.0552, 0.0037) | 0.9517 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0015 | (-0.0330, 0.0355) | 0.4693 | 0.0015 | (-0.0405, 0.0425) | 0.4907 |
| controlled_alt_vs_controlled_default | naturalness | -0.0099 | (-0.0280, 0.0086) | 0.8527 | -0.0099 | (-0.0211, 0.0013) | 0.9583 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0316 | (-0.0656, 0.0040) | 0.9590 | -0.0316 | (-0.0748, 0.0088) | 0.9317 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0090 | (-0.0201, 0.0014) | 0.9533 | -0.0090 | (-0.0196, 0.0015) | 0.9540 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0054 | (-0.0391, 0.0488) | 0.3973 | 0.0054 | (-0.0440, 0.0566) | 0.4147 |
| controlled_alt_vs_controlled_default | persona_style | -0.0140 | (-0.0339, 0.0052) | 0.9133 | -0.0140 | (-0.0342, 0.0036) | 0.9243 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0059 | (-0.0147, 0.0028) | 0.9067 | -0.0059 | (-0.0162, 0.0019) | 0.9083 |
| controlled_alt_vs_controlled_default | length_score | -0.0381 | (-0.1211, 0.0426) | 0.8160 | -0.0381 | (-0.0839, 0.0128) | 0.9200 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0022 | (-0.0348, 0.0420) | 0.4697 | 0.0022 | (-0.0321, 0.0339) | 0.4470 |
| controlled_alt_vs_controlled_default | bertscore_f1 | -0.0147 | (-0.0289, 0.0000) | 0.9743 | -0.0147 | (-0.0251, -0.0047) | 0.9983 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0128 | (-0.0271, 0.0019) | 0.9563 | -0.0128 | (-0.0308, 0.0069) | 0.9130 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1838 | (0.1573, 0.2093) | 0.0000 | 0.1838 | (0.1518, 0.2161) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1816 | (0.1466, 0.2161) | 0.0000 | 0.1816 | (0.1358, 0.2193) | 0.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0564 | (0.0360, 0.0766) | 0.0000 | 0.0564 | (0.0190, 0.0927) | 0.0010 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2409 | (0.2061, 0.2747) | 0.0000 | 0.2409 | (0.1976, 0.2833) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0506 | (0.0419, 0.0596) | 0.0000 | 0.0506 | (0.0404, 0.0623) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.2062 | (0.1669, 0.2483) | 0.0000 | 0.2062 | (0.1634, 0.2446) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0831 | (0.0421, 0.1277) | 0.0000 | 0.0831 | (-0.0223, 0.2234) | 0.1107 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0045 | (-0.0136, 0.0049) | 0.8277 | -0.0045 | (-0.0222, 0.0115) | 0.6880 |
| controlled_alt_vs_proposed_raw | length_score | 0.2167 | (0.1333, 0.2961) | 0.0000 | 0.2167 | (0.0914, 0.3491) | 0.0007 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1531 | (0.1125, 0.1938) | 0.0000 | 0.1531 | (0.0844, 0.2219) | 0.0000 |
| controlled_alt_vs_proposed_raw | bertscore_f1 | 0.0147 | (0.0005, 0.0294) | 0.0203 | 0.0147 | (-0.0037, 0.0318) | 0.0590 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1457 | (0.1263, 0.1641) | 0.0000 | 0.1457 | (0.1146, 0.1728) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2382 | (0.2181, 0.2577) | 0.0000 | 0.2382 | (0.2121, 0.2659) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1807 | (0.1440, 0.2153) | 0.0000 | 0.1807 | (0.1294, 0.2279) | 0.0000 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0527 | (0.0315, 0.0732) | 0.0000 | 0.0527 | (0.0086, 0.0948) | 0.0083 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3118 | (0.2855, 0.3382) | 0.0000 | 0.3118 | (0.2773, 0.3509) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0665 | (0.0594, 0.0734) | 0.0000 | 0.0665 | (0.0593, 0.0732) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.2045 | (0.1610, 0.2482) | 0.0000 | 0.2045 | (0.1501, 0.2528) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0855 | (0.0422, 0.1261) | 0.0000 | 0.0855 | (-0.0158, 0.2141) | 0.0897 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0000 | (-0.0100, 0.0099) | 0.5053 | 0.0000 | (-0.0165, 0.0151) | 0.4900 |
| controlled_alt_vs_candidate_no_context | length_score | 0.2000 | (0.1095, 0.2893) | 0.0000 | 0.2000 | (0.0449, 0.3533) | 0.0043 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1313 | (0.0844, 0.1750) | 0.0000 | 0.1313 | (0.0531, 0.2094) | 0.0003 |
| controlled_alt_vs_candidate_no_context | bertscore_f1 | 0.0439 | (0.0309, 0.0567) | 0.0000 | 0.0439 | (0.0245, 0.0649) | 0.0000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1702 | (0.1544, 0.1849) | 0.0000 | 0.1702 | (0.1465, 0.1922) | 0.0000 |
| controlled_alt_vs_baseline_no_context | context_relevance | 0.2137 | (0.1936, 0.2375) | 0.0000 | 0.2137 | (0.1827, 0.2482) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 0.1432 | (0.1146, 0.1730) | 0.0000 | 0.1432 | (0.1220, 0.1641) | 0.0000 |
| controlled_alt_vs_baseline_no_context | naturalness | -0.0248 | (-0.0430, -0.0062) | 0.9960 | -0.0248 | (-0.0512, 0.0013) | 0.9677 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 0.2782 | (0.2482, 0.3075) | 0.0000 | 0.2782 | (0.2361, 0.3293) | 0.0000 |
| controlled_alt_vs_baseline_no_context | context_overlap | 0.0634 | (0.0558, 0.0713) | 0.0000 | 0.0634 | (0.0581, 0.0690) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 0.1793 | (0.1440, 0.2145) | 0.0000 | 0.1793 | (0.1542, 0.2040) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_style | -0.0009 | (-0.0258, 0.0246) | 0.5300 | -0.0009 | (-0.0116, 0.0118) | 0.5767 |
| controlled_alt_vs_baseline_no_context | distinct1 | -0.0465 | (-0.0554, -0.0374) | 1.0000 | -0.0465 | (-0.0550, -0.0376) | 1.0000 |
| controlled_alt_vs_baseline_no_context | length_score | -0.0455 | (-0.1289, 0.0346) | 0.8770 | -0.0455 | (-0.1625, 0.0762) | 0.7647 |
| controlled_alt_vs_baseline_no_context | sentence_score | 0.0344 | (-0.0094, 0.0781) | 0.0623 | 0.0344 | (-0.0407, 0.1187) | 0.2233 |
| controlled_alt_vs_baseline_no_context | bertscore_f1 | 0.0253 | (0.0105, 0.0392) | 0.0000 | 0.0253 | (0.0032, 0.0432) | 0.0137 |
| controlled_alt_vs_baseline_no_context | overall_quality | 0.1326 | (0.1193, 0.1464) | 0.0000 | 0.1326 | (0.1206, 0.1507) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2385 | (0.2183, 0.2580) | 0.0000 | 0.2385 | (0.2176, 0.2586) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1417 | (0.1106, 0.1735) | 0.0000 | 0.1417 | (0.1081, 0.1726) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0148 | (-0.0341, 0.0035) | 0.9433 | -0.0148 | (-0.0447, 0.0136) | 0.8447 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3097 | (0.2844, 0.3360) | 0.0000 | 0.3097 | (0.2797, 0.3373) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0724 | (0.0637, 0.0822) | 0.0000 | 0.0724 | (0.0668, 0.0780) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1739 | (0.1353, 0.2119) | 0.0000 | 0.1739 | (0.1327, 0.2124) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0131 | (-0.0103, 0.0358) | 0.1247 | 0.0131 | (-0.0126, 0.0431) | 0.1760 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0406 | (-0.0481, -0.0331) | 1.0000 | -0.0406 | (-0.0473, -0.0342) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | -0.0074 | (-0.0878, 0.0687) | 0.5790 | -0.0074 | (-0.1533, 0.1238) | 0.5157 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0321 | (-0.0121, 0.0754) | 0.0783 | 0.0321 | (-0.0482, 0.1107) | 0.2190 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0401 | (0.0258, 0.0535) | 0.0000 | 0.0401 | (0.0234, 0.0570) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1454 | (0.1327, 0.1579) | 0.0000 | 0.1454 | (0.1316, 0.1565) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 48 | 15 | 49 | 0.6473 | 0.7619 |
| proposed_vs_candidate_no_context | persona_consistency | 21 | 17 | 74 | 0.5179 | 0.5526 |
| proposed_vs_candidate_no_context | naturalness | 27 | 36 | 49 | 0.4598 | 0.4286 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 41 | 7 | 64 | 0.6518 | 0.8542 |
| proposed_vs_candidate_no_context | context_overlap | 42 | 20 | 50 | 0.5982 | 0.6774 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 14 | 11 | 87 | 0.5134 | 0.5600 |
| proposed_vs_candidate_no_context | persona_style | 12 | 8 | 92 | 0.5179 | 0.6000 |
| proposed_vs_candidate_no_context | distinct1 | 30 | 25 | 57 | 0.5223 | 0.5455 |
| proposed_vs_candidate_no_context | length_score | 30 | 31 | 51 | 0.4955 | 0.4918 |
| proposed_vs_candidate_no_context | sentence_score | 11 | 18 | 83 | 0.4688 | 0.3793 |
| proposed_vs_candidate_no_context | bertscore_f1 | 60 | 27 | 25 | 0.6473 | 0.6897 |
| proposed_vs_candidate_no_context | overall_quality | 56 | 31 | 25 | 0.6116 | 0.6437 |
| proposed_vs_baseline_no_context | context_relevance | 57 | 53 | 2 | 0.5179 | 0.5182 |
| proposed_vs_baseline_no_context | persona_consistency | 21 | 51 | 40 | 0.3661 | 0.2917 |
| proposed_vs_baseline_no_context | naturalness | 24 | 88 | 0 | 0.2143 | 0.2143 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 37 | 30 | 45 | 0.5312 | 0.5522 |
| proposed_vs_baseline_no_context | context_overlap | 62 | 47 | 3 | 0.5670 | 0.5688 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 16 | 37 | 59 | 0.4062 | 0.3019 |
| proposed_vs_baseline_no_context | persona_style | 9 | 26 | 77 | 0.4241 | 0.2571 |
| proposed_vs_baseline_no_context | distinct1 | 17 | 81 | 14 | 0.2143 | 0.1735 |
| proposed_vs_baseline_no_context | length_score | 26 | 84 | 2 | 0.2411 | 0.2364 |
| proposed_vs_baseline_no_context | sentence_score | 10 | 48 | 54 | 0.3304 | 0.1724 |
| proposed_vs_baseline_no_context | bertscore_f1 | 64 | 48 | 0 | 0.5714 | 0.5714 |
| proposed_vs_baseline_no_context | overall_quality | 44 | 68 | 0 | 0.3929 | 0.3929 |
| controlled_vs_proposed_raw | context_relevance | 102 | 10 | 0 | 0.9107 | 0.9107 |
| controlled_vs_proposed_raw | persona_consistency | 88 | 11 | 13 | 0.8438 | 0.8889 |
| controlled_vs_proposed_raw | naturalness | 82 | 30 | 0 | 0.7321 | 0.7321 |
| controlled_vs_proposed_raw | context_keyword_coverage | 97 | 9 | 6 | 0.8929 | 0.9151 |
| controlled_vs_proposed_raw | context_overlap | 101 | 11 | 0 | 0.9018 | 0.9018 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 87 | 10 | 15 | 0.8438 | 0.8969 |
| controlled_vs_proposed_raw | persona_style | 32 | 9 | 71 | 0.6027 | 0.7805 |
| controlled_vs_proposed_raw | distinct1 | 58 | 50 | 4 | 0.5357 | 0.5370 |
| controlled_vs_proposed_raw | length_score | 73 | 33 | 6 | 0.6786 | 0.6887 |
| controlled_vs_proposed_raw | sentence_score | 57 | 9 | 46 | 0.7143 | 0.8636 |
| controlled_vs_proposed_raw | bertscore_f1 | 69 | 43 | 0 | 0.6161 | 0.6161 |
| controlled_vs_proposed_raw | overall_quality | 103 | 9 | 0 | 0.9196 | 0.9196 |
| controlled_vs_candidate_no_context | context_relevance | 112 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 89 | 11 | 12 | 0.8482 | 0.8900 |
| controlled_vs_candidate_no_context | naturalness | 78 | 34 | 0 | 0.6964 | 0.6964 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 112 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 110 | 2 | 0 | 0.9821 | 0.9821 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 88 | 9 | 15 | 0.8527 | 0.9072 |
| controlled_vs_candidate_no_context | persona_style | 34 | 9 | 69 | 0.6116 | 0.7907 |
| controlled_vs_candidate_no_context | distinct1 | 59 | 49 | 4 | 0.5446 | 0.5463 |
| controlled_vs_candidate_no_context | length_score | 76 | 32 | 4 | 0.6964 | 0.7037 |
| controlled_vs_candidate_no_context | sentence_score | 53 | 12 | 47 | 0.6830 | 0.8154 |
| controlled_vs_candidate_no_context | bertscore_f1 | 87 | 25 | 0 | 0.7768 | 0.7768 |
| controlled_vs_candidate_no_context | overall_quality | 111 | 1 | 0 | 0.9911 | 0.9911 |
| controlled_vs_baseline_no_context | context_relevance | 111 | 1 | 0 | 0.9911 | 0.9911 |
| controlled_vs_baseline_no_context | persona_consistency | 83 | 13 | 16 | 0.8125 | 0.8646 |
| controlled_vs_baseline_no_context | naturalness | 50 | 62 | 0 | 0.4464 | 0.4464 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 110 | 1 | 1 | 0.9866 | 0.9910 |
| controlled_vs_baseline_no_context | context_overlap | 111 | 0 | 1 | 0.9955 | 1.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 79 | 9 | 24 | 0.8125 | 0.8977 |
| controlled_vs_baseline_no_context | persona_style | 17 | 11 | 84 | 0.5268 | 0.6071 |
| controlled_vs_baseline_no_context | distinct1 | 16 | 90 | 6 | 0.1696 | 0.1509 |
| controlled_vs_baseline_no_context | length_score | 56 | 53 | 3 | 0.5134 | 0.5138 |
| controlled_vs_baseline_no_context | sentence_score | 32 | 21 | 59 | 0.5491 | 0.6038 |
| controlled_vs_baseline_no_context | bertscore_f1 | 81 | 31 | 0 | 0.7232 | 0.7232 |
| controlled_vs_baseline_no_context | overall_quality | 110 | 2 | 0 | 0.9821 | 0.9821 |
| controlled_alt_vs_controlled_default | context_relevance | 44 | 57 | 11 | 0.4420 | 0.4356 |
| controlled_alt_vs_controlled_default | persona_consistency | 33 | 34 | 45 | 0.4955 | 0.4925 |
| controlled_alt_vs_controlled_default | naturalness | 42 | 59 | 11 | 0.4241 | 0.4158 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 33 | 41 | 38 | 0.4643 | 0.4459 |
| controlled_alt_vs_controlled_default | context_overlap | 43 | 58 | 11 | 0.4330 | 0.4257 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 27 | 24 | 61 | 0.5134 | 0.5294 |
| controlled_alt_vs_controlled_default | persona_style | 13 | 19 | 80 | 0.4732 | 0.4062 |
| controlled_alt_vs_controlled_default | distinct1 | 43 | 55 | 14 | 0.4464 | 0.4388 |
| controlled_alt_vs_controlled_default | length_score | 42 | 51 | 19 | 0.4598 | 0.4516 |
| controlled_alt_vs_controlled_default | sentence_score | 17 | 18 | 77 | 0.4955 | 0.4857 |
| controlled_alt_vs_controlled_default | bertscore_f1 | 44 | 59 | 9 | 0.4330 | 0.4272 |
| controlled_alt_vs_controlled_default | overall_quality | 49 | 54 | 9 | 0.4777 | 0.4757 |
| controlled_alt_vs_proposed_raw | context_relevance | 99 | 13 | 0 | 0.8839 | 0.8839 |
| controlled_alt_vs_proposed_raw | persona_consistency | 89 | 10 | 13 | 0.8527 | 0.8990 |
| controlled_alt_vs_proposed_raw | naturalness | 77 | 35 | 0 | 0.6875 | 0.6875 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 92 | 9 | 11 | 0.8705 | 0.9109 |
| controlled_alt_vs_proposed_raw | context_overlap | 95 | 17 | 0 | 0.8482 | 0.8482 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 88 | 6 | 18 | 0.8661 | 0.9362 |
| controlled_alt_vs_proposed_raw | persona_style | 25 | 13 | 74 | 0.5536 | 0.6579 |
| controlled_alt_vs_proposed_raw | distinct1 | 50 | 59 | 3 | 0.4598 | 0.4587 |
| controlled_alt_vs_proposed_raw | length_score | 70 | 35 | 7 | 0.6562 | 0.6667 |
| controlled_alt_vs_proposed_raw | sentence_score | 59 | 10 | 43 | 0.7188 | 0.8551 |
| controlled_alt_vs_proposed_raw | bertscore_f1 | 63 | 49 | 0 | 0.5625 | 0.5625 |
| controlled_alt_vs_proposed_raw | overall_quality | 103 | 9 | 0 | 0.9196 | 0.9196 |
| controlled_alt_vs_candidate_no_context | context_relevance | 112 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 89 | 10 | 13 | 0.8527 | 0.8990 |
| controlled_alt_vs_candidate_no_context | naturalness | 76 | 36 | 0 | 0.6786 | 0.6786 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 109 | 0 | 3 | 0.9866 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 109 | 2 | 1 | 0.9777 | 0.9820 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 88 | 8 | 16 | 0.8571 | 0.9167 |
| controlled_alt_vs_candidate_no_context | persona_style | 29 | 11 | 72 | 0.5804 | 0.7250 |
| controlled_alt_vs_candidate_no_context | distinct1 | 57 | 52 | 3 | 0.5223 | 0.5229 |
| controlled_alt_vs_candidate_no_context | length_score | 67 | 39 | 6 | 0.6250 | 0.6321 |
| controlled_alt_vs_candidate_no_context | sentence_score | 57 | 15 | 40 | 0.6875 | 0.7917 |
| controlled_alt_vs_candidate_no_context | bertscore_f1 | 85 | 27 | 0 | 0.7589 | 0.7589 |
| controlled_alt_vs_candidate_no_context | overall_quality | 110 | 2 | 0 | 0.9821 | 0.9821 |
| controlled_alt_vs_baseline_no_context | context_relevance | 109 | 3 | 0 | 0.9732 | 0.9732 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 84 | 13 | 15 | 0.8170 | 0.8660 |
| controlled_alt_vs_baseline_no_context | naturalness | 48 | 64 | 0 | 0.4286 | 0.4286 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 106 | 2 | 4 | 0.9643 | 0.9815 |
| controlled_alt_vs_baseline_no_context | context_overlap | 109 | 3 | 0 | 0.9732 | 0.9732 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 82 | 5 | 25 | 0.8438 | 0.9425 |
| controlled_alt_vs_baseline_no_context | persona_style | 15 | 15 | 82 | 0.5000 | 0.5000 |
| controlled_alt_vs_baseline_no_context | distinct1 | 13 | 93 | 6 | 0.1429 | 0.1226 |
| controlled_alt_vs_baseline_no_context | length_score | 58 | 52 | 2 | 0.5268 | 0.5273 |
| controlled_alt_vs_baseline_no_context | sentence_score | 31 | 20 | 61 | 0.5491 | 0.6078 |
| controlled_alt_vs_baseline_no_context | bertscore_f1 | 74 | 38 | 0 | 0.6607 | 0.6607 |
| controlled_alt_vs_baseline_no_context | overall_quality | 109 | 3 | 0 | 0.9732 | 0.9732 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 111 | 1 | 0 | 0.9911 | 0.9911 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 83 | 13 | 16 | 0.8125 | 0.8646 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 50 | 62 | 0 | 0.4464 | 0.4464 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 110 | 1 | 1 | 0.9866 | 0.9910 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 111 | 0 | 1 | 0.9955 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 79 | 9 | 24 | 0.8125 | 0.8977 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 17 | 11 | 84 | 0.5268 | 0.6071 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | 16 | 90 | 6 | 0.1696 | 0.1509 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 56 | 53 | 3 | 0.5134 | 0.5138 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 32 | 21 | 59 | 0.5491 | 0.6038 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 81 | 31 | 0 | 0.7232 | 0.7232 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 110 | 2 | 0 | 0.9821 | 0.9821 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.2768 | 0.4464 | 0.5446 |
| proposed_contextual_controlled_alt | 0.0000 | 0.0000 | 0.1964 | 0.4643 | 0.5268 |
| proposed_contextual | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.5357 | 0.0000 | 0.0000 |
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
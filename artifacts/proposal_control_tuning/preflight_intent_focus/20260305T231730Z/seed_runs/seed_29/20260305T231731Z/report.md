# Proposal Alignment Evaluation Report

- Run ID: `20260305T231731Z`
- Generated: `2026-03-05T23:24:01.094126+00:00`
- Scenarios: `artifacts\proposal_control_tuning\preflight_intent_focus\20260305T231730Z\seed_runs\seed_29\20260305T231731Z\scenarios.jsonl`
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
| proposed_contextual_controlled | 0.3204 (0.2675, 0.3714) | 0.3390 (0.2950, 0.3848) | 0.8327 (0.7984, 0.8689) | 0.3942 (0.3744, 0.4140) | 0.0837 |
| proposed_contextual_controlled_alt | 0.2662 (0.2373, 0.2965) | 0.3585 (0.2891, 0.4289) | 0.8718 (0.8393, 0.9044) | 0.3832 (0.3594, 0.4073) | 0.0657 |
| proposed_contextual | 0.0782 (0.0346, 0.1304) | 0.1268 (0.0882, 0.1738) | 0.7762 (0.7622, 0.7924) | 0.2123 (0.1855, 0.2411) | 0.0476 |
| candidate_no_context | 0.0252 (0.0143, 0.0404) | 0.1840 (0.1213, 0.2533) | 0.7838 (0.7617, 0.8109) | 0.2092 (0.1832, 0.2369) | 0.0318 |
| baseline_no_context | 0.0253 (0.0141, 0.0405) | 0.1521 (0.1159, 0.1878) | 0.8873 (0.8626, 0.9120) | 0.2173 (0.2025, 0.2330) | 0.0394 |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0530 | 2.1024 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0572 | -0.3111 |
| proposed_vs_candidate_no_context | naturalness | -0.0076 | -0.0098 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0720 | 3.9310 |
| proposed_vs_candidate_no_context | context_overlap | 0.0088 | 0.2134 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0740 | -0.6602 |
| proposed_vs_candidate_no_context | persona_style | 0.0098 | 0.0208 |
| proposed_vs_candidate_no_context | distinct1 | -0.0016 | -0.0018 |
| proposed_vs_candidate_no_context | length_score | -0.0222 | -0.1212 |
| proposed_vs_candidate_no_context | sentence_score | -0.0146 | -0.0190 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0158 | 0.4985 |
| proposed_vs_candidate_no_context | overall_quality | 0.0030 | 0.0145 |
| proposed_vs_baseline_no_context | context_relevance | 0.0529 | 2.0871 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0253 | -0.1665 |
| proposed_vs_baseline_no_context | naturalness | -0.1111 | -0.1252 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0706 | 3.5833 |
| proposed_vs_baseline_no_context | context_overlap | 0.0116 | 0.3023 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0143 | -0.2727 |
| proposed_vs_baseline_no_context | persona_style | -0.0695 | -0.1261 |
| proposed_vs_baseline_no_context | distinct1 | -0.0528 | -0.0541 |
| proposed_vs_baseline_no_context | length_score | -0.3861 | -0.7056 |
| proposed_vs_baseline_no_context | sentence_score | -0.1167 | -0.1343 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0083 | 0.2097 |
| proposed_vs_baseline_no_context | overall_quality | -0.0050 | -0.0232 |
| controlled_vs_proposed_raw | context_relevance | 0.2422 | 3.0953 |
| controlled_vs_proposed_raw | persona_consistency | 0.2122 | 1.6740 |
| controlled_vs_proposed_raw | naturalness | 0.0565 | 0.0728 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.3176 | 3.5175 |
| controlled_vs_proposed_raw | context_overlap | 0.0664 | 1.3230 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2425 | 6.3646 |
| controlled_vs_proposed_raw | persona_style | 0.0912 | 0.1895 |
| controlled_vs_proposed_raw | distinct1 | 0.0012 | 0.0013 |
| controlled_vs_proposed_raw | length_score | 0.1972 | 1.2241 |
| controlled_vs_proposed_raw | sentence_score | 0.1625 | 0.2161 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0360 | 0.7561 |
| controlled_vs_proposed_raw | overall_quality | 0.1820 | 0.8574 |
| controlled_vs_candidate_no_context | context_relevance | 0.2952 | 11.7051 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1550 | 0.8422 |
| controlled_vs_candidate_no_context | naturalness | 0.0489 | 0.0624 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3895 | 21.2759 |
| controlled_vs_candidate_no_context | context_overlap | 0.0752 | 1.8187 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1685 | 1.5027 |
| controlled_vs_candidate_no_context | persona_style | 0.1010 | 0.2143 |
| controlled_vs_candidate_no_context | distinct1 | -0.0004 | -0.0005 |
| controlled_vs_candidate_no_context | length_score | 0.1750 | 0.9545 |
| controlled_vs_candidate_no_context | sentence_score | 0.1479 | 0.1929 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0519 | 1.6315 |
| controlled_vs_candidate_no_context | overall_quality | 0.1850 | 0.8843 |
| controlled_vs_baseline_no_context | context_relevance | 0.2951 | 11.6426 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1869 | 1.2288 |
| controlled_vs_baseline_no_context | naturalness | -0.0546 | -0.0615 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3881 | 19.7051 |
| controlled_vs_baseline_no_context | context_overlap | 0.0780 | 2.0253 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2282 | 4.3561 |
| controlled_vs_baseline_no_context | persona_style | 0.0218 | 0.0395 |
| controlled_vs_baseline_no_context | distinct1 | -0.0516 | -0.0528 |
| controlled_vs_baseline_no_context | length_score | -0.1889 | -0.3452 |
| controlled_vs_baseline_no_context | sentence_score | 0.0458 | 0.0528 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0443 | 1.1244 |
| controlled_vs_baseline_no_context | overall_quality | 0.1769 | 0.8143 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0543 | -0.1694 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0196 | 0.0577 |
| controlled_alt_vs_controlled_default | naturalness | 0.0391 | 0.0470 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0763 | -0.1870 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0030 | -0.0259 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0234 | 0.0835 |
| controlled_alt_vs_controlled_default | persona_style | 0.0041 | 0.0072 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0021 | 0.0023 |
| controlled_alt_vs_controlled_default | length_score | 0.1931 | 0.5388 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0021 | -0.0023 |
| controlled_alt_vs_controlled_default | bertscore_f1 | -0.0180 | -0.2152 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0111 | -0.0280 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1879 | 2.4015 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.2318 | 1.8282 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0957 | 0.1233 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2413 | 2.6727 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0634 | 1.2629 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.2659 | 6.9792 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0953 | 0.1980 |
| controlled_alt_vs_proposed_raw | distinct1 | 0.0033 | 0.0036 |
| controlled_alt_vs_proposed_raw | length_score | 0.3903 | 2.4224 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1604 | 0.2133 |
| controlled_alt_vs_proposed_raw | bertscore_f1 | 0.0180 | 0.3782 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1709 | 0.8053 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2409 | 9.5527 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1745 | 0.9484 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0880 | 0.1123 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3133 | 17.1103 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0722 | 1.7458 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1919 | 1.7115 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.1052 | 0.2230 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0017 | 0.0018 |
| controlled_alt_vs_candidate_no_context | length_score | 0.3681 | 2.0076 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1458 | 0.1902 |
| controlled_alt_vs_candidate_no_context | bertscore_f1 | 0.0339 | 1.0653 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1740 | 0.8315 |
| controlled_alt_vs_baseline_no_context | context_relevance | 0.2408 | 9.5008 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 0.2064 | 1.3574 |
| controlled_alt_vs_baseline_no_context | naturalness | -0.0154 | -0.0174 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 0.3119 | 15.8333 |
| controlled_alt_vs_baseline_no_context | context_overlap | 0.0750 | 1.9471 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 0.2516 | 4.8030 |
| controlled_alt_vs_baseline_no_context | persona_style | 0.0259 | 0.0469 |
| controlled_alt_vs_baseline_no_context | distinct1 | -0.0495 | -0.0506 |
| controlled_alt_vs_baseline_no_context | length_score | 0.0042 | 0.0076 |
| controlled_alt_vs_baseline_no_context | sentence_score | 0.0437 | 0.0504 |
| controlled_alt_vs_baseline_no_context | bertscore_f1 | 0.0263 | 0.6673 |
| controlled_alt_vs_baseline_no_context | overall_quality | 0.1659 | 0.7634 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2951 | 11.6426 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1869 | 1.2288 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0546 | -0.0615 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3881 | 19.7051 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0780 | 2.0253 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2282 | 4.3561 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0218 | 0.0395 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0516 | -0.0528 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | -0.1889 | -0.3452 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0458 | 0.0528 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0443 | 1.1244 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1769 | 0.8143 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0530 | (0.0073, 0.1078) | 0.0097 | 0.0530 | (0.0066, 0.1058) | 0.0100 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0572 | (-0.1309, 0.0110) | 0.9443 | -0.0572 | (-0.1731, 0.0256) | 0.8647 |
| proposed_vs_candidate_no_context | naturalness | -0.0076 | (-0.0389, 0.0175) | 0.6870 | -0.0076 | (-0.0444, 0.0236) | 0.6630 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0720 | (0.0082, 0.1471) | 0.0120 | 0.0720 | (0.0094, 0.1429) | 0.0150 |
| proposed_vs_candidate_no_context | context_overlap | 0.0088 | (-0.0011, 0.0184) | 0.0403 | 0.0088 | (-0.0013, 0.0173) | 0.0400 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0740 | (-0.1583, 0.0000) | 0.9750 | -0.0740 | (-0.1991, 0.0168) | 0.9243 |
| proposed_vs_candidate_no_context | persona_style | 0.0098 | (-0.0859, 0.1098) | 0.4287 | 0.0098 | (-0.0572, 0.0862) | 0.3747 |
| proposed_vs_candidate_no_context | distinct1 | -0.0016 | (-0.0263, 0.0200) | 0.5447 | -0.0016 | (-0.0259, 0.0252) | 0.5720 |
| proposed_vs_candidate_no_context | length_score | -0.0222 | (-0.1292, 0.0736) | 0.6310 | -0.0222 | (-0.1222, 0.0873) | 0.6257 |
| proposed_vs_candidate_no_context | sentence_score | -0.0146 | (-0.1021, 0.0729) | 0.6933 | -0.0146 | (-0.1000, 0.0500) | 0.7070 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0158 | (-0.0079, 0.0368) | 0.0850 | 0.0158 | (-0.0043, 0.0370) | 0.0667 |
| proposed_vs_candidate_no_context | overall_quality | 0.0030 | (-0.0332, 0.0431) | 0.4333 | 0.0030 | (-0.0494, 0.0427) | 0.4177 |
| proposed_vs_baseline_no_context | context_relevance | 0.0529 | (0.0073, 0.1077) | 0.0090 | 0.0529 | (0.0160, 0.1042) | 0.0010 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0253 | (-0.0813, 0.0350) | 0.8007 | -0.0253 | (-0.0594, 0.0166) | 0.8807 |
| proposed_vs_baseline_no_context | naturalness | -0.1111 | (-0.1411, -0.0809) | 1.0000 | -0.1111 | (-0.1430, -0.0714) | 1.0000 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0706 | (0.0061, 0.1477) | 0.0113 | 0.0706 | (0.0169, 0.1421) | 0.0023 |
| proposed_vs_baseline_no_context | context_overlap | 0.0116 | (-0.0007, 0.0243) | 0.0360 | 0.0116 | (0.0025, 0.0228) | 0.0043 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0143 | (-0.0821, 0.0560) | 0.6840 | -0.0143 | (-0.0452, 0.0208) | 0.7863 |
| proposed_vs_baseline_no_context | persona_style | -0.0695 | (-0.1719, 0.0213) | 0.9160 | -0.0695 | (-0.2362, 0.0400) | 0.7577 |
| proposed_vs_baseline_no_context | distinct1 | -0.0528 | (-0.0757, -0.0298) | 1.0000 | -0.0528 | (-0.0729, -0.0230) | 0.9997 |
| proposed_vs_baseline_no_context | length_score | -0.3861 | (-0.5167, -0.2458) | 1.0000 | -0.3861 | (-0.5370, -0.2037) | 0.9997 |
| proposed_vs_baseline_no_context | sentence_score | -0.1167 | (-0.2042, -0.0292) | 0.9977 | -0.1167 | (-0.1680, -0.0553) | 1.0000 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0083 | (-0.0152, 0.0314) | 0.2530 | 0.0083 | (-0.0183, 0.0380) | 0.2983 |
| proposed_vs_baseline_no_context | overall_quality | -0.0050 | (-0.0375, 0.0303) | 0.6127 | -0.0050 | (-0.0289, 0.0288) | 0.6340 |
| controlled_vs_proposed_raw | context_relevance | 0.2422 | (0.1883, 0.2987) | 0.0000 | 0.2422 | (0.1933, 0.3118) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.2122 | (0.1471, 0.2825) | 0.0000 | 0.2122 | (0.1510, 0.2605) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0565 | (0.0209, 0.0944) | 0.0003 | 0.0565 | (0.0208, 0.0849) | 0.0017 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.3176 | (0.2393, 0.3927) | 0.0000 | 0.3176 | (0.2503, 0.4075) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0664 | (0.0523, 0.0810) | 0.0000 | 0.0664 | (0.0549, 0.0806) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2425 | (0.1726, 0.3157) | 0.0000 | 0.2425 | (0.1910, 0.2854) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.0912 | (-0.0080, 0.1923) | 0.0403 | 0.0912 | (-0.0473, 0.2521) | 0.1327 |
| controlled_vs_proposed_raw | distinct1 | 0.0012 | (-0.0210, 0.0247) | 0.4687 | 0.0012 | (-0.0115, 0.0142) | 0.4080 |
| controlled_vs_proposed_raw | length_score | 0.1972 | (0.0556, 0.3514) | 0.0043 | 0.1972 | (0.0222, 0.3508) | 0.0140 |
| controlled_vs_proposed_raw | sentence_score | 0.1625 | (0.0812, 0.2375) | 0.0003 | 0.1625 | (0.0763, 0.2250) | 0.0000 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0360 | (0.0056, 0.0669) | 0.0117 | 0.0360 | (0.0131, 0.0646) | 0.0000 |
| controlled_vs_proposed_raw | overall_quality | 0.1820 | (0.1488, 0.2142) | 0.0000 | 0.1820 | (0.1569, 0.2039) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2952 | (0.2449, 0.3496) | 0.0000 | 0.2952 | (0.2411, 0.3612) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1550 | (0.0786, 0.2302) | 0.0000 | 0.1550 | (0.0432, 0.2384) | 0.0033 |
| controlled_vs_candidate_no_context | naturalness | 0.0489 | (0.0046, 0.0914) | 0.0120 | 0.0489 | (0.0023, 0.0940) | 0.0200 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3895 | (0.3210, 0.4586) | 0.0000 | 0.3895 | (0.3120, 0.4827) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0752 | (0.0617, 0.0886) | 0.0000 | 0.0752 | (0.0658, 0.0863) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1685 | (0.0807, 0.2552) | 0.0000 | 0.1685 | (0.0524, 0.2581) | 0.0040 |
| controlled_vs_candidate_no_context | persona_style | 0.1010 | (-0.0046, 0.2140) | 0.0287 | 0.1010 | (-0.0528, 0.2582) | 0.1440 |
| controlled_vs_candidate_no_context | distinct1 | -0.0004 | (-0.0193, 0.0174) | 0.5120 | -0.0004 | (-0.0190, 0.0181) | 0.5283 |
| controlled_vs_candidate_no_context | length_score | 0.1750 | (-0.0042, 0.3528) | 0.0270 | 0.1750 | (0.0035, 0.3834) | 0.0223 |
| controlled_vs_candidate_no_context | sentence_score | 0.1479 | (0.0458, 0.2479) | 0.0030 | 0.1479 | (0.0190, 0.2518) | 0.0117 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0519 | (0.0260, 0.0801) | 0.0000 | 0.0519 | (0.0232, 0.0821) | 0.0000 |
| controlled_vs_candidate_no_context | overall_quality | 0.1850 | (0.1501, 0.2174) | 0.0000 | 0.1850 | (0.1375, 0.2238) | 0.0000 |
| controlled_vs_baseline_no_context | context_relevance | 0.2951 | (0.2441, 0.3467) | 0.0000 | 0.2951 | (0.2471, 0.3561) | 0.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1869 | (0.1381, 0.2385) | 0.0000 | 0.1869 | (0.1453, 0.2224) | 0.0000 |
| controlled_vs_baseline_no_context | naturalness | -0.0546 | (-0.0940, -0.0152) | 0.9977 | -0.0546 | (-0.0943, -0.0080) | 0.9910 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3881 | (0.3217, 0.4548) | 0.0000 | 0.3881 | (0.3203, 0.4707) | 0.0000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0780 | (0.0621, 0.0944) | 0.0000 | 0.0780 | (0.0675, 0.0913) | 0.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2282 | (0.1692, 0.2883) | 0.0000 | 0.2282 | (0.1925, 0.2546) | 0.0000 |
| controlled_vs_baseline_no_context | persona_style | 0.0218 | (-0.0377, 0.0799) | 0.2217 | 0.0218 | (-0.0631, 0.1042) | 0.2993 |
| controlled_vs_baseline_no_context | distinct1 | -0.0516 | (-0.0735, -0.0305) | 1.0000 | -0.0516 | (-0.0692, -0.0274) | 1.0000 |
| controlled_vs_baseline_no_context | length_score | -0.1889 | (-0.3514, -0.0319) | 0.9910 | -0.1889 | (-0.3508, -0.0014) | 0.9750 |
| controlled_vs_baseline_no_context | sentence_score | 0.0458 | (-0.0396, 0.1313) | 0.1320 | 0.0458 | (-0.0471, 0.1188) | 0.1547 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0443 | (0.0177, 0.0707) | 0.0003 | 0.0443 | (0.0213, 0.0698) | 0.0000 |
| controlled_vs_baseline_no_context | overall_quality | 0.1769 | (0.1556, 0.1962) | 0.0000 | 0.1769 | (0.1581, 0.1959) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0543 | (-0.1079, 0.0043) | 0.9637 | -0.0543 | (-0.1142, -0.0070) | 0.9840 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0196 | (-0.0621, 0.0987) | 0.3103 | 0.0196 | (-0.0739, 0.1360) | 0.3317 |
| controlled_alt_vs_controlled_default | naturalness | 0.0391 | (-0.0049, 0.0827) | 0.0407 | 0.0391 | (-0.0163, 0.0802) | 0.0767 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0763 | (-0.1491, -0.0038) | 0.9790 | -0.0763 | (-0.1549, -0.0129) | 0.9893 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0030 | (-0.0273, 0.0248) | 0.6163 | -0.0030 | (-0.0167, 0.0075) | 0.7047 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0234 | (-0.0732, 0.1206) | 0.3257 | 0.0234 | (-0.0963, 0.1704) | 0.3613 |
| controlled_alt_vs_controlled_default | persona_style | 0.0041 | (-0.0315, 0.0411) | 0.3953 | 0.0041 | (-0.0245, 0.0367) | 0.4150 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0021 | (-0.0195, 0.0256) | 0.4280 | 0.0021 | (-0.0184, 0.0187) | 0.4000 |
| controlled_alt_vs_controlled_default | length_score | 0.1931 | (-0.0069, 0.3819) | 0.0280 | 0.1931 | (-0.0565, 0.4000) | 0.0660 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0021 | (-0.0583, 0.0667) | 0.6227 | -0.0021 | (-0.0875, 0.0848) | 0.6690 |
| controlled_alt_vs_controlled_default | bertscore_f1 | -0.0180 | (-0.0480, 0.0091) | 0.9007 | -0.0180 | (-0.0438, 0.0049) | 0.9380 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0111 | (-0.0342, 0.0145) | 0.8073 | -0.0111 | (-0.0264, 0.0096) | 0.8600 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1879 | (0.1377, 0.2345) | 0.0000 | 0.1879 | (0.1438, 0.2262) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.2318 | (0.1721, 0.2953) | 0.0000 | 0.2318 | (0.1499, 0.3287) | 0.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0957 | (0.0607, 0.1284) | 0.0000 | 0.0957 | (0.0388, 0.1448) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2413 | (0.1761, 0.3014) | 0.0000 | 0.2413 | (0.1847, 0.2909) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0634 | (0.0433, 0.0873) | 0.0000 | 0.0634 | (0.0469, 0.0780) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.2659 | (0.1932, 0.3401) | 0.0000 | 0.2659 | (0.1706, 0.3944) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0953 | (0.0108, 0.1960) | 0.0123 | 0.0953 | (-0.0264, 0.2546) | 0.1013 |
| controlled_alt_vs_proposed_raw | distinct1 | 0.0033 | (-0.0134, 0.0206) | 0.3487 | 0.0033 | (-0.0167, 0.0187) | 0.3527 |
| controlled_alt_vs_proposed_raw | length_score | 0.3903 | (0.2444, 0.5319) | 0.0000 | 0.3903 | (0.1450, 0.6000) | 0.0003 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1604 | (0.0729, 0.2333) | 0.0007 | 0.1604 | (0.0636, 0.2375) | 0.0000 |
| controlled_alt_vs_proposed_raw | bertscore_f1 | 0.0180 | (-0.0101, 0.0462) | 0.1113 | 0.0180 | (-0.0124, 0.0450) | 0.1200 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1709 | (0.1426, 0.1993) | 0.0000 | 0.1709 | (0.1442, 0.2032) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2409 | (0.2106, 0.2762) | 0.0000 | 0.2409 | (0.2165, 0.2679) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1745 | (0.1062, 0.2414) | 0.0000 | 0.1745 | (0.0985, 0.2713) | 0.0000 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0880 | (0.0451, 0.1264) | 0.0000 | 0.0880 | (0.0262, 0.1328) | 0.0033 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3133 | (0.2723, 0.3580) | 0.0000 | 0.3133 | (0.2799, 0.3486) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0722 | (0.0546, 0.0955) | 0.0000 | 0.0722 | (0.0586, 0.0855) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1919 | (0.1181, 0.2744) | 0.0000 | 0.1919 | (0.1147, 0.3121) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.1052 | (-0.0004, 0.2190) | 0.0263 | 0.1052 | (-0.0438, 0.2724) | 0.1083 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0017 | (-0.0171, 0.0196) | 0.4180 | 0.0017 | (-0.0197, 0.0156) | 0.4263 |
| controlled_alt_vs_candidate_no_context | length_score | 0.3681 | (0.1944, 0.5236) | 0.0000 | 0.3681 | (0.1368, 0.5346) | 0.0013 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1458 | (0.0292, 0.2479) | 0.0103 | 0.1458 | (-0.0175, 0.2722) | 0.0540 |
| controlled_alt_vs_candidate_no_context | bertscore_f1 | 0.0339 | (0.0090, 0.0595) | 0.0027 | 0.0339 | (0.0092, 0.0573) | 0.0043 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1740 | (0.1452, 0.2049) | 0.0000 | 0.1740 | (0.1412, 0.2104) | 0.0000 |
| controlled_alt_vs_baseline_no_context | context_relevance | 0.2408 | (0.2041, 0.2745) | 0.0000 | 0.2408 | (0.2128, 0.2705) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 0.2064 | (0.1367, 0.2781) | 0.0000 | 0.2064 | (0.1115, 0.3171) | 0.0000 |
| controlled_alt_vs_baseline_no_context | naturalness | -0.0154 | (-0.0509, 0.0219) | 0.7983 | -0.0154 | (-0.0607, 0.0210) | 0.7857 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 0.3119 | (0.2614, 0.3561) | 0.0000 | 0.3119 | (0.2705, 0.3539) | 0.0000 |
| controlled_alt_vs_baseline_no_context | context_overlap | 0.0750 | (0.0586, 0.0958) | 0.0000 | 0.0750 | (0.0652, 0.0826) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 0.2516 | (0.1685, 0.3454) | 0.0000 | 0.2516 | (0.1399, 0.3917) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_style | 0.0259 | (-0.0199, 0.0837) | 0.1553 | 0.0259 | (-0.0365, 0.0950) | 0.2163 |
| controlled_alt_vs_baseline_no_context | distinct1 | -0.0495 | (-0.0673, -0.0323) | 1.0000 | -0.0495 | (-0.0660, -0.0333) | 1.0000 |
| controlled_alt_vs_baseline_no_context | length_score | 0.0042 | (-0.1653, 0.1598) | 0.4970 | 0.0042 | (-0.2158, 0.1857) | 0.4727 |
| controlled_alt_vs_baseline_no_context | sentence_score | 0.0437 | (-0.0292, 0.1167) | 0.1560 | 0.0437 | (-0.0500, 0.1289) | 0.2067 |
| controlled_alt_vs_baseline_no_context | bertscore_f1 | 0.0263 | (0.0075, 0.0461) | 0.0030 | 0.0263 | (0.0067, 0.0428) | 0.0053 |
| controlled_alt_vs_baseline_no_context | overall_quality | 0.1659 | (0.1407, 0.1919) | 0.0000 | 0.1659 | (0.1387, 0.1962) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2951 | (0.2446, 0.3483) | 0.0000 | 0.2951 | (0.2450, 0.3587) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1869 | (0.1372, 0.2360) | 0.0000 | 0.1869 | (0.1439, 0.2231) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0546 | (-0.0925, -0.0151) | 0.9977 | -0.0546 | (-0.0946, -0.0086) | 0.9887 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3881 | (0.3211, 0.4563) | 0.0000 | 0.3881 | (0.3204, 0.4716) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0780 | (0.0621, 0.0943) | 0.0000 | 0.0780 | (0.0675, 0.0913) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2282 | (0.1698, 0.2847) | 0.0000 | 0.2282 | (0.1930, 0.2557) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0218 | (-0.0375, 0.0828) | 0.2203 | 0.0218 | (-0.0569, 0.1024) | 0.3027 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0516 | (-0.0728, -0.0303) | 1.0000 | -0.0516 | (-0.0689, -0.0271) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | -0.1889 | (-0.3569, -0.0347) | 0.9943 | -0.1889 | (-0.3485, 0.0213) | 0.9650 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0458 | (-0.0396, 0.1313) | 0.1340 | 0.0458 | (-0.0478, 0.1185) | 0.1653 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0443 | (0.0173, 0.0715) | 0.0010 | 0.0443 | (0.0213, 0.0699) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1769 | (0.1564, 0.1960) | 0.0000 | 0.1769 | (0.1581, 0.1956) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 10 | 5 | 9 | 0.6042 | 0.6667 |
| proposed_vs_candidate_no_context | persona_consistency | 4 | 6 | 14 | 0.4583 | 0.4000 |
| proposed_vs_candidate_no_context | naturalness | 8 | 7 | 9 | 0.5208 | 0.5333 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 7 | 3 | 14 | 0.5833 | 0.7000 |
| proposed_vs_candidate_no_context | context_overlap | 10 | 5 | 9 | 0.6042 | 0.6667 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 2 | 5 | 17 | 0.4375 | 0.2857 |
| proposed_vs_candidate_no_context | persona_style | 3 | 3 | 18 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | distinct1 | 10 | 5 | 9 | 0.6042 | 0.6667 |
| proposed_vs_candidate_no_context | length_score | 7 | 7 | 10 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | sentence_score | 4 | 5 | 15 | 0.4792 | 0.4444 |
| proposed_vs_candidate_no_context | bertscore_f1 | 12 | 7 | 5 | 0.6042 | 0.6316 |
| proposed_vs_candidate_no_context | overall_quality | 8 | 11 | 5 | 0.4375 | 0.4211 |
| proposed_vs_baseline_no_context | context_relevance | 12 | 12 | 0 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context | persona_consistency | 2 | 9 | 13 | 0.3542 | 0.1818 |
| proposed_vs_baseline_no_context | naturalness | 1 | 23 | 0 | 0.0417 | 0.0417 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 7 | 3 | 14 | 0.5833 | 0.7000 |
| proposed_vs_baseline_no_context | context_overlap | 14 | 10 | 0 | 0.5833 | 0.5833 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 2 | 4 | 18 | 0.4583 | 0.3333 |
| proposed_vs_baseline_no_context | persona_style | 1 | 5 | 18 | 0.4167 | 0.1667 |
| proposed_vs_baseline_no_context | distinct1 | 5 | 19 | 0 | 0.2083 | 0.2083 |
| proposed_vs_baseline_no_context | length_score | 2 | 22 | 0 | 0.0833 | 0.0833 |
| proposed_vs_baseline_no_context | sentence_score | 2 | 10 | 12 | 0.3333 | 0.1667 |
| proposed_vs_baseline_no_context | bertscore_f1 | 12 | 12 | 0 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context | overall_quality | 8 | 16 | 0 | 0.3333 | 0.3333 |
| controlled_vs_proposed_raw | context_relevance | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_vs_proposed_raw | persona_consistency | 21 | 2 | 1 | 0.8958 | 0.9130 |
| controlled_vs_proposed_raw | naturalness | 19 | 5 | 0 | 0.7917 | 0.7917 |
| controlled_vs_proposed_raw | context_keyword_coverage | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_vs_proposed_raw | context_overlap | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 21 | 1 | 2 | 0.9167 | 0.9545 |
| controlled_vs_proposed_raw | persona_style | 8 | 4 | 12 | 0.5833 | 0.6667 |
| controlled_vs_proposed_raw | distinct1 | 13 | 10 | 1 | 0.5625 | 0.5652 |
| controlled_vs_proposed_raw | length_score | 13 | 8 | 3 | 0.6042 | 0.6190 |
| controlled_vs_proposed_raw | sentence_score | 12 | 1 | 11 | 0.7292 | 0.9231 |
| controlled_vs_proposed_raw | bertscore_f1 | 18 | 6 | 0 | 0.7500 | 0.7500 |
| controlled_vs_proposed_raw | overall_quality | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_relevance | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 19 | 4 | 1 | 0.8125 | 0.8261 |
| controlled_vs_candidate_no_context | naturalness | 18 | 6 | 0 | 0.7500 | 0.7500 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 19 | 4 | 1 | 0.8125 | 0.8261 |
| controlled_vs_candidate_no_context | persona_style | 8 | 4 | 12 | 0.5833 | 0.6667 |
| controlled_vs_candidate_no_context | distinct1 | 13 | 10 | 1 | 0.5625 | 0.5652 |
| controlled_vs_candidate_no_context | length_score | 14 | 9 | 1 | 0.6042 | 0.6087 |
| controlled_vs_candidate_no_context | sentence_score | 13 | 3 | 8 | 0.7083 | 0.8125 |
| controlled_vs_candidate_no_context | bertscore_f1 | 19 | 5 | 0 | 0.7917 | 0.7917 |
| controlled_vs_candidate_no_context | overall_quality | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_vs_baseline_no_context | context_relevance | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_vs_baseline_no_context | naturalness | 8 | 16 | 0 | 0.3333 | 0.3333 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | context_overlap | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_vs_baseline_no_context | persona_style | 6 | 3 | 15 | 0.5625 | 0.6667 |
| controlled_vs_baseline_no_context | distinct1 | 2 | 21 | 1 | 0.1042 | 0.0870 |
| controlled_vs_baseline_no_context | length_score | 8 | 16 | 0 | 0.3333 | 0.3333 |
| controlled_vs_baseline_no_context | sentence_score | 6 | 3 | 15 | 0.5625 | 0.6667 |
| controlled_vs_baseline_no_context | bertscore_f1 | 17 | 7 | 0 | 0.7083 | 0.7083 |
| controlled_vs_baseline_no_context | overall_quality | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 6 | 15 | 3 | 0.3125 | 0.2857 |
| controlled_alt_vs_controlled_default | persona_consistency | 10 | 8 | 6 | 0.5417 | 0.5556 |
| controlled_alt_vs_controlled_default | naturalness | 15 | 6 | 3 | 0.6875 | 0.7143 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 4 | 12 | 8 | 0.3333 | 0.2500 |
| controlled_alt_vs_controlled_default | context_overlap | 7 | 12 | 5 | 0.3958 | 0.3684 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 9 | 7 | 8 | 0.5417 | 0.5625 |
| controlled_alt_vs_controlled_default | persona_style | 3 | 3 | 18 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | distinct1 | 12 | 9 | 3 | 0.5625 | 0.5714 |
| controlled_alt_vs_controlled_default | length_score | 14 | 5 | 5 | 0.6875 | 0.7368 |
| controlled_alt_vs_controlled_default | sentence_score | 1 | 2 | 21 | 0.4792 | 0.3333 |
| controlled_alt_vs_controlled_default | bertscore_f1 | 10 | 12 | 2 | 0.4583 | 0.4545 |
| controlled_alt_vs_controlled_default | overall_quality | 8 | 14 | 2 | 0.3750 | 0.3636 |
| controlled_alt_vs_proposed_raw | context_relevance | 22 | 2 | 0 | 0.9167 | 0.9167 |
| controlled_alt_vs_proposed_raw | persona_consistency | 21 | 0 | 3 | 0.9375 | 1.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 20 | 4 | 0 | 0.8333 | 0.8333 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 22 | 2 | 0 | 0.9167 | 0.9167 |
| controlled_alt_vs_proposed_raw | context_overlap | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 21 | 0 | 3 | 0.9375 | 1.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 7 | 3 | 14 | 0.5833 | 0.7000 |
| controlled_alt_vs_proposed_raw | distinct1 | 14 | 10 | 0 | 0.5833 | 0.5833 |
| controlled_alt_vs_proposed_raw | length_score | 19 | 3 | 2 | 0.8333 | 0.8636 |
| controlled_alt_vs_proposed_raw | sentence_score | 12 | 1 | 11 | 0.7292 | 0.9231 |
| controlled_alt_vs_proposed_raw | bertscore_f1 | 16 | 8 | 0 | 0.6667 | 0.6667 |
| controlled_alt_vs_proposed_raw | overall_quality | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 19 | 2 | 3 | 0.8542 | 0.9048 |
| controlled_alt_vs_candidate_no_context | naturalness | 22 | 2 | 0 | 0.9167 | 0.9167 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 18 | 2 | 4 | 0.8333 | 0.9000 |
| controlled_alt_vs_candidate_no_context | persona_style | 8 | 5 | 11 | 0.5625 | 0.6154 |
| controlled_alt_vs_candidate_no_context | distinct1 | 15 | 9 | 0 | 0.6250 | 0.6250 |
| controlled_alt_vs_candidate_no_context | length_score | 20 | 4 | 0 | 0.8333 | 0.8333 |
| controlled_alt_vs_candidate_no_context | sentence_score | 14 | 4 | 6 | 0.7083 | 0.7778 |
| controlled_alt_vs_candidate_no_context | bertscore_f1 | 17 | 7 | 0 | 0.7083 | 0.7083 |
| controlled_alt_vs_candidate_no_context | overall_quality | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context | context_relevance | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 20 | 1 | 3 | 0.8958 | 0.9524 |
| controlled_alt_vs_baseline_no_context | naturalness | 11 | 13 | 0 | 0.4583 | 0.4583 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 23 | 0 | 1 | 0.9792 | 1.0000 |
| controlled_alt_vs_baseline_no_context | context_overlap | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 20 | 1 | 3 | 0.8958 | 0.9524 |
| controlled_alt_vs_baseline_no_context | persona_style | 4 | 3 | 17 | 0.5208 | 0.5714 |
| controlled_alt_vs_baseline_no_context | distinct1 | 3 | 21 | 0 | 0.1250 | 0.1250 |
| controlled_alt_vs_baseline_no_context | length_score | 11 | 12 | 1 | 0.4792 | 0.4783 |
| controlled_alt_vs_baseline_no_context | sentence_score | 5 | 2 | 17 | 0.5625 | 0.7143 |
| controlled_alt_vs_baseline_no_context | bertscore_f1 | 14 | 10 | 0 | 0.5833 | 0.5833 |
| controlled_alt_vs_baseline_no_context | overall_quality | 24 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 24 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 23 | 1 | 0 | 0.9583 | 0.9583 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 8 | 16 | 0 | 0.3333 | 0.3333 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 24 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 23 | 1 | 0 | 0.9583 | 0.9583 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 23 | 1 | 0 | 0.9583 | 0.9583 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 6 | 3 | 15 | 0.5625 | 0.6667 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | 2 | 21 | 1 | 0.1042 | 0.0870 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 8 | 16 | 0 | 0.3333 | 0.3333 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 6 | 3 | 15 | 0.5625 | 0.6667 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 17 | 7 | 0 | 0.7083 | 0.7083 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 24 | 0 | 0 | 1.0000 | 1.0000 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.2083 | 0.3750 | 0.6250 |
| proposed_contextual_controlled_alt | 0.0000 | 0.0000 | 0.3333 | 0.2917 | 0.7083 |
| proposed_contextual | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.6250 | 0.0000 | 0.0000 |
| baseline_no_context | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `8`
- Unique template signatures: `22`
- Template signature ratio: `0.9167`
- Effective sample size by source clustering: `7.20`
- Effective sample size by template-signature clustering: `20.57`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: enabled.

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
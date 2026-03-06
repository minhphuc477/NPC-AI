# Proposal Alignment Evaluation Report

- Run ID: `20260304T114443Z`
- Generated: `2026-03-04T11:53:29.556793+00:00`
- Scenarios: `artifacts\proposal_control_tuning\balanced_v2\20260304T114443Z\scenarios.jsonl`
- Scenario count: `40`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off
- `baseline_no_context`: model `phi3:mini`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2537 (0.2335, 0.2738) | 0.4297 (0.3680, 0.4927) | 0.9012 (0.8767, 0.9231) | 0.4076 (0.3880, 0.4270) | 0.0674 |
| proposed_contextual | 0.0669 (0.0401, 0.0972) | 0.1376 (0.1086, 0.1712) | 0.7989 (0.7795, 0.8208) | 0.2166 (0.1958, 0.2394) | 0.0614 |
| candidate_no_context | 0.0280 (0.0183, 0.0397) | 0.1834 (0.1338, 0.2423) | 0.7932 (0.7748, 0.8152) | 0.2127 (0.1927, 0.2340) | 0.0406 |
| baseline_no_context | 0.0394 (0.0269, 0.0523) | 0.1785 (0.1473, 0.2147) | 0.8895 (0.8703, 0.9080) | 0.2337 (0.2199, 0.2478) | 0.0537 |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0389 | 1.3917 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0458 | -0.2496 |
| proposed_vs_candidate_no_context | naturalness | 0.0056 | 0.0071 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0500 | 2.1980 |
| proposed_vs_candidate_no_context | context_overlap | 0.0131 | 0.3252 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0599 | -0.5589 |
| proposed_vs_candidate_no_context | persona_style | 0.0107 | 0.0219 |
| proposed_vs_candidate_no_context | distinct1 | 0.0022 | 0.0023 |
| proposed_vs_candidate_no_context | length_score | 0.0325 | 0.1388 |
| proposed_vs_candidate_no_context | sentence_score | -0.0175 | -0.0232 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0208 | 0.5129 |
| proposed_vs_candidate_no_context | overall_quality | 0.0038 | 0.0181 |
| proposed_vs_baseline_no_context | context_relevance | 0.0275 | 0.6974 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0408 | -0.2289 |
| proposed_vs_baseline_no_context | naturalness | -0.0907 | -0.1019 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0345 | 0.9011 |
| proposed_vs_baseline_no_context | context_overlap | 0.0111 | 0.2647 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0340 | -0.4187 |
| proposed_vs_baseline_no_context | persona_style | -0.0681 | -0.1200 |
| proposed_vs_baseline_no_context | distinct1 | -0.0510 | -0.0520 |
| proposed_vs_baseline_no_context | length_score | -0.2858 | -0.5173 |
| proposed_vs_baseline_no_context | sentence_score | -0.1312 | -0.1511 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0077 | 0.1433 |
| proposed_vs_baseline_no_context | overall_quality | -0.0171 | -0.0733 |
| controlled_vs_proposed_raw | context_relevance | 0.1868 | 2.7910 |
| controlled_vs_proposed_raw | persona_consistency | 0.2921 | 2.1225 |
| controlled_vs_proposed_raw | naturalness | 0.1024 | 0.1281 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2475 | 3.3996 |
| controlled_vs_proposed_raw | context_overlap | 0.0451 | 0.8481 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.3435 | 7.2670 |
| controlled_vs_proposed_raw | persona_style | 0.0865 | 0.1733 |
| controlled_vs_proposed_raw | distinct1 | 0.0065 | 0.0070 |
| controlled_vs_proposed_raw | length_score | 0.3800 | 1.4250 |
| controlled_vs_proposed_raw | sentence_score | 0.2375 | 0.3220 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0060 | 0.0972 |
| controlled_vs_proposed_raw | overall_quality | 0.1910 | 0.8819 |
| controlled_vs_candidate_no_context | context_relevance | 0.2257 | 8.0669 |
| controlled_vs_candidate_no_context | persona_consistency | 0.2463 | 1.3431 |
| controlled_vs_candidate_no_context | naturalness | 0.1080 | 0.1361 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.2975 | 13.0699 |
| controlled_vs_candidate_no_context | context_overlap | 0.0582 | 1.4491 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2836 | 2.6467 |
| controlled_vs_candidate_no_context | persona_style | 0.0971 | 0.1989 |
| controlled_vs_candidate_no_context | distinct1 | 0.0087 | 0.0094 |
| controlled_vs_candidate_no_context | length_score | 0.4125 | 1.7616 |
| controlled_vs_candidate_no_context | sentence_score | 0.2200 | 0.2914 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0268 | 0.6599 |
| controlled_vs_candidate_no_context | overall_quality | 0.1948 | 0.9159 |
| controlled_vs_baseline_no_context | context_relevance | 0.2143 | 5.4346 |
| controlled_vs_baseline_no_context | persona_consistency | 0.2512 | 1.4077 |
| controlled_vs_baseline_no_context | naturalness | 0.0117 | 0.0131 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.2820 | 7.3640 |
| controlled_vs_baseline_no_context | context_overlap | 0.0563 | 1.3373 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.3094 | 3.8053 |
| controlled_vs_baseline_no_context | persona_style | 0.0184 | 0.0325 |
| controlled_vs_baseline_no_context | distinct1 | -0.0444 | -0.0453 |
| controlled_vs_baseline_no_context | length_score | 0.0942 | 0.1704 |
| controlled_vs_baseline_no_context | sentence_score | 0.1062 | 0.1223 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0137 | 0.2543 |
| controlled_vs_baseline_no_context | overall_quality | 0.1739 | 0.7440 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2143 | 5.4346 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.2512 | 1.4077 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 0.0117 | 0.0131 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.2820 | 7.3640 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0563 | 1.3373 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.3094 | 3.8053 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0184 | 0.0325 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0444 | -0.0453 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.0942 | 0.1704 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.1062 | 0.1223 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0137 | 0.2543 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1739 | 0.7440 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0389 | (0.0128, 0.0706) | 0.0013 | 0.0389 | (0.0058, 0.0794) | 0.0020 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0458 | (-0.1003, 0.0031) | 0.9620 | -0.0458 | (-0.1046, 0.0100) | 0.9443 |
| proposed_vs_candidate_no_context | naturalness | 0.0056 | (-0.0186, 0.0287) | 0.3207 | 0.0056 | (-0.0237, 0.0348) | 0.3947 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0500 | (0.0180, 0.0887) | 0.0010 | 0.0500 | (0.0062, 0.1022) | 0.0067 |
| proposed_vs_candidate_no_context | context_overlap | 0.0131 | (0.0049, 0.0222) | 0.0000 | 0.0131 | (0.0021, 0.0266) | 0.0037 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0599 | (-0.1224, -0.0033) | 0.9830 | -0.0599 | (-0.1242, 0.0052) | 0.9660 |
| proposed_vs_candidate_no_context | persona_style | 0.0107 | (-0.0370, 0.0586) | 0.3463 | 0.0107 | (-0.0488, 0.0629) | 0.3343 |
| proposed_vs_candidate_no_context | distinct1 | 0.0022 | (-0.0112, 0.0155) | 0.3770 | 0.0022 | (-0.0163, 0.0178) | 0.4140 |
| proposed_vs_candidate_no_context | length_score | 0.0325 | (-0.0542, 0.1242) | 0.2287 | 0.0325 | (-0.0847, 0.1476) | 0.3070 |
| proposed_vs_candidate_no_context | sentence_score | -0.0175 | (-0.0700, 0.0350) | 0.7810 | -0.0175 | (-0.0721, 0.0368) | 0.7877 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0208 | (0.0036, 0.0378) | 0.0087 | 0.0208 | (-0.0002, 0.0421) | 0.0263 |
| proposed_vs_candidate_no_context | overall_quality | 0.0038 | (-0.0174, 0.0267) | 0.3780 | 0.0038 | (-0.0193, 0.0235) | 0.3600 |
| proposed_vs_baseline_no_context | context_relevance | 0.0275 | (-0.0051, 0.0630) | 0.0560 | 0.0275 | (-0.0028, 0.0714) | 0.0563 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0408 | (-0.0896, 0.0062) | 0.9520 | -0.0408 | (-0.0935, 0.0182) | 0.9080 |
| proposed_vs_baseline_no_context | naturalness | -0.0907 | (-0.1184, -0.0600) | 1.0000 | -0.0907 | (-0.1220, -0.0567) | 1.0000 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0345 | (-0.0069, 0.0797) | 0.0603 | 0.0345 | (-0.0051, 0.0900) | 0.0783 |
| proposed_vs_baseline_no_context | context_overlap | 0.0111 | (-0.0006, 0.0237) | 0.0310 | 0.0111 | (-0.0051, 0.0289) | 0.1013 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0340 | (-0.0935, 0.0229) | 0.8843 | -0.0340 | (-0.1050, 0.0368) | 0.8280 |
| proposed_vs_baseline_no_context | persona_style | -0.0681 | (-0.1482, 0.0040) | 0.9663 | -0.0681 | (-0.2044, 0.0529) | 0.8250 |
| proposed_vs_baseline_no_context | distinct1 | -0.0510 | (-0.0634, -0.0364) | 1.0000 | -0.0510 | (-0.0628, -0.0374) | 1.0000 |
| proposed_vs_baseline_no_context | length_score | -0.2858 | (-0.3933, -0.1775) | 1.0000 | -0.2858 | (-0.3943, -0.1737) | 1.0000 |
| proposed_vs_baseline_no_context | sentence_score | -0.1313 | (-0.2012, -0.0612) | 1.0000 | -0.1313 | (-0.2000, -0.0362) | 0.9973 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0077 | (-0.0100, 0.0245) | 0.1907 | 0.0077 | (-0.0211, 0.0316) | 0.2863 |
| proposed_vs_baseline_no_context | overall_quality | -0.0171 | (-0.0474, 0.0122) | 0.8747 | -0.0171 | (-0.0495, 0.0227) | 0.8203 |
| controlled_vs_proposed_raw | context_relevance | 0.1868 | (0.1570, 0.2175) | 0.0000 | 0.1868 | (0.1531, 0.2223) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.2921 | (0.2284, 0.3598) | 0.0000 | 0.2921 | (0.2170, 0.3620) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.1024 | (0.0674, 0.1341) | 0.0000 | 0.1024 | (0.0638, 0.1322) | 0.0000 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2475 | (0.2073, 0.2882) | 0.0000 | 0.2475 | (0.2062, 0.2907) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0451 | (0.0332, 0.0576) | 0.0000 | 0.0451 | (0.0280, 0.0623) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.3435 | (0.2667, 0.4254) | 0.0000 | 0.3435 | (0.2414, 0.4424) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.0865 | (0.0268, 0.1542) | 0.0007 | 0.0865 | (0.0000, 0.2188) | 0.0943 |
| controlled_vs_proposed_raw | distinct1 | 0.0065 | (-0.0059, 0.0182) | 0.1457 | 0.0065 | (-0.0079, 0.0191) | 0.1870 |
| controlled_vs_proposed_raw | length_score | 0.3800 | (0.2516, 0.5092) | 0.0000 | 0.3800 | (0.2378, 0.4978) | 0.0000 |
| controlled_vs_proposed_raw | sentence_score | 0.2375 | (0.1625, 0.2975) | 0.0000 | 0.2375 | (0.1456, 0.3073) | 0.0000 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0060 | (-0.0120, 0.0236) | 0.2577 | 0.0060 | (-0.0178, 0.0359) | 0.3610 |
| controlled_vs_proposed_raw | overall_quality | 0.1910 | (0.1658, 0.2163) | 0.0000 | 0.1910 | (0.1555, 0.2214) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2257 | (0.2043, 0.2476) | 0.0000 | 0.2257 | (0.2007, 0.2513) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.2463 | (0.1657, 0.3246) | 0.0000 | 0.2463 | (0.1300, 0.3443) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.1080 | (0.0746, 0.1384) | 0.0000 | 0.1080 | (0.0654, 0.1361) | 0.0000 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.2975 | (0.2684, 0.3282) | 0.0000 | 0.2975 | (0.2631, 0.3299) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0582 | (0.0505, 0.0660) | 0.0000 | 0.0582 | (0.0500, 0.0693) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2836 | (0.1851, 0.3814) | 0.0000 | 0.2836 | (0.1581, 0.4173) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.0971 | (0.0219, 0.1852) | 0.0057 | 0.0971 | (-0.0311, 0.2615) | 0.1107 |
| controlled_vs_candidate_no_context | distinct1 | 0.0087 | (-0.0052, 0.0216) | 0.0960 | 0.0087 | (-0.0095, 0.0223) | 0.1527 |
| controlled_vs_candidate_no_context | length_score | 0.4125 | (0.2816, 0.5334) | 0.0000 | 0.4125 | (0.2667, 0.5207) | 0.0000 |
| controlled_vs_candidate_no_context | sentence_score | 0.2200 | (0.1600, 0.2725) | 0.0000 | 0.2200 | (0.1559, 0.2657) | 0.0000 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0268 | (0.0038, 0.0491) | 0.0103 | 0.0268 | (-0.0088, 0.0583) | 0.0630 |
| controlled_vs_candidate_no_context | overall_quality | 0.1948 | (0.1699, 0.2185) | 0.0000 | 0.1948 | (0.1650, 0.2194) | 0.0000 |
| controlled_vs_baseline_no_context | context_relevance | 0.2143 | (0.1907, 0.2389) | 0.0000 | 0.2143 | (0.1947, 0.2353) | 0.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 0.2512 | (0.1882, 0.3180) | 0.0000 | 0.2512 | (0.1707, 0.3139) | 0.0000 |
| controlled_vs_baseline_no_context | naturalness | 0.0117 | (-0.0181, 0.0412) | 0.2250 | 0.0117 | (-0.0115, 0.0394) | 0.1987 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.2820 | (0.2482, 0.3174) | 0.0000 | 0.2820 | (0.2550, 0.3091) | 0.0000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0563 | (0.0472, 0.0657) | 0.0000 | 0.0563 | (0.0473, 0.0665) | 0.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.3094 | (0.2314, 0.3900) | 0.0000 | 0.3094 | (0.2110, 0.3887) | 0.0000 |
| controlled_vs_baseline_no_context | persona_style | 0.0184 | (-0.0208, 0.0600) | 0.1757 | 0.0184 | (-0.0163, 0.0870) | 0.2387 |
| controlled_vs_baseline_no_context | distinct1 | -0.0444 | (-0.0553, -0.0336) | 1.0000 | -0.0444 | (-0.0548, -0.0338) | 1.0000 |
| controlled_vs_baseline_no_context | length_score | 0.0942 | (-0.0442, 0.2275) | 0.0887 | 0.0942 | (-0.0200, 0.2267) | 0.0740 |
| controlled_vs_baseline_no_context | sentence_score | 0.1062 | (0.0462, 0.1662) | 0.0000 | 0.1062 | (0.0400, 0.1764) | 0.0007 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0137 | (-0.0061, 0.0324) | 0.0853 | 0.0137 | (-0.0196, 0.0384) | 0.1847 |
| controlled_vs_baseline_no_context | overall_quality | 0.1739 | (0.1510, 0.1970) | 0.0000 | 0.1739 | (0.1485, 0.1933) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2143 | (0.1918, 0.2385) | 0.0000 | 0.2143 | (0.1941, 0.2344) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.2512 | (0.1882, 0.3168) | 0.0000 | 0.2512 | (0.1699, 0.3134) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 0.0117 | (-0.0181, 0.0416) | 0.2197 | 0.0117 | (-0.0108, 0.0404) | 0.1860 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.2820 | (0.2508, 0.3161) | 0.0000 | 0.2820 | (0.2531, 0.3100) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0563 | (0.0471, 0.0657) | 0.0000 | 0.0563 | (0.0474, 0.0673) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.3094 | (0.2339, 0.3892) | 0.0000 | 0.3094 | (0.2095, 0.3880) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0184 | (-0.0187, 0.0610) | 0.1803 | 0.0184 | (-0.0167, 0.0821) | 0.2427 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0444 | (-0.0552, -0.0337) | 1.0000 | -0.0444 | (-0.0547, -0.0342) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.0942 | (-0.0333, 0.2233) | 0.0780 | 0.0942 | (-0.0182, 0.2295) | 0.0637 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.1062 | (0.0462, 0.1663) | 0.0000 | 0.1062 | (0.0350, 0.1763) | 0.0017 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0137 | (-0.0067, 0.0327) | 0.0837 | 0.0137 | (-0.0185, 0.0388) | 0.1780 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1739 | (0.1494, 0.1956) | 0.0000 | 0.1739 | (0.1489, 0.1939) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 17 | 4 | 19 | 0.6625 | 0.8095 |
| proposed_vs_candidate_no_context | persona_consistency | 6 | 10 | 24 | 0.4500 | 0.3750 |
| proposed_vs_candidate_no_context | naturalness | 11 | 10 | 19 | 0.5125 | 0.5238 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 12 | 2 | 26 | 0.6250 | 0.8571 |
| proposed_vs_candidate_no_context | context_overlap | 16 | 5 | 19 | 0.6375 | 0.7619 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 4 | 9 | 27 | 0.4375 | 0.3077 |
| proposed_vs_candidate_no_context | persona_style | 4 | 3 | 33 | 0.5125 | 0.5714 |
| proposed_vs_candidate_no_context | distinct1 | 10 | 11 | 19 | 0.4875 | 0.4762 |
| proposed_vs_candidate_no_context | length_score | 12 | 9 | 19 | 0.5375 | 0.5714 |
| proposed_vs_candidate_no_context | sentence_score | 4 | 6 | 30 | 0.4750 | 0.4000 |
| proposed_vs_candidate_no_context | bertscore_f1 | 16 | 15 | 9 | 0.5125 | 0.5161 |
| proposed_vs_candidate_no_context | overall_quality | 11 | 20 | 9 | 0.3875 | 0.3548 |
| proposed_vs_baseline_no_context | context_relevance | 18 | 22 | 0 | 0.4500 | 0.4500 |
| proposed_vs_baseline_no_context | persona_consistency | 8 | 18 | 14 | 0.3750 | 0.3077 |
| proposed_vs_baseline_no_context | naturalness | 7 | 33 | 0 | 0.1750 | 0.1750 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 11 | 11 | 18 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context | context_overlap | 21 | 18 | 1 | 0.5375 | 0.5385 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 6 | 11 | 23 | 0.4375 | 0.3529 |
| proposed_vs_baseline_no_context | persona_style | 3 | 8 | 29 | 0.4375 | 0.2727 |
| proposed_vs_baseline_no_context | distinct1 | 3 | 35 | 2 | 0.1000 | 0.0789 |
| proposed_vs_baseline_no_context | length_score | 8 | 32 | 0 | 0.2000 | 0.2000 |
| proposed_vs_baseline_no_context | sentence_score | 4 | 19 | 17 | 0.3125 | 0.1739 |
| proposed_vs_baseline_no_context | bertscore_f1 | 21 | 19 | 0 | 0.5250 | 0.5250 |
| proposed_vs_baseline_no_context | overall_quality | 15 | 25 | 0 | 0.3750 | 0.3750 |
| controlled_vs_proposed_raw | context_relevance | 37 | 3 | 0 | 0.9250 | 0.9250 |
| controlled_vs_proposed_raw | persona_consistency | 38 | 1 | 1 | 0.9625 | 0.9744 |
| controlled_vs_proposed_raw | naturalness | 33 | 7 | 0 | 0.8250 | 0.8250 |
| controlled_vs_proposed_raw | context_keyword_coverage | 36 | 0 | 4 | 0.9500 | 1.0000 |
| controlled_vs_proposed_raw | context_overlap | 33 | 7 | 0 | 0.8250 | 0.8250 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 37 | 1 | 2 | 0.9500 | 0.9737 |
| controlled_vs_proposed_raw | persona_style | 8 | 1 | 31 | 0.5875 | 0.8889 |
| controlled_vs_proposed_raw | distinct1 | 24 | 15 | 1 | 0.6125 | 0.6154 |
| controlled_vs_proposed_raw | length_score | 29 | 9 | 2 | 0.7500 | 0.7632 |
| controlled_vs_proposed_raw | sentence_score | 30 | 2 | 8 | 0.8500 | 0.9375 |
| controlled_vs_proposed_raw | bertscore_f1 | 22 | 18 | 0 | 0.5500 | 0.5500 |
| controlled_vs_proposed_raw | overall_quality | 39 | 1 | 0 | 0.9750 | 0.9750 |
| controlled_vs_candidate_no_context | context_relevance | 40 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 34 | 5 | 1 | 0.8625 | 0.8718 |
| controlled_vs_candidate_no_context | naturalness | 33 | 7 | 0 | 0.8250 | 0.8250 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 40 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 40 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 34 | 5 | 1 | 0.8625 | 0.8718 |
| controlled_vs_candidate_no_context | persona_style | 10 | 3 | 27 | 0.5875 | 0.7692 |
| controlled_vs_candidate_no_context | distinct1 | 23 | 16 | 1 | 0.5875 | 0.5897 |
| controlled_vs_candidate_no_context | length_score | 29 | 9 | 2 | 0.7500 | 0.7632 |
| controlled_vs_candidate_no_context | sentence_score | 26 | 1 | 13 | 0.8125 | 0.9630 |
| controlled_vs_candidate_no_context | bertscore_f1 | 26 | 14 | 0 | 0.6500 | 0.6500 |
| controlled_vs_candidate_no_context | overall_quality | 40 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | context_relevance | 40 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 35 | 2 | 3 | 0.9125 | 0.9459 |
| controlled_vs_baseline_no_context | naturalness | 23 | 17 | 0 | 0.5750 | 0.5750 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 40 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | context_overlap | 40 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 35 | 2 | 3 | 0.9125 | 0.9459 |
| controlled_vs_baseline_no_context | persona_style | 4 | 3 | 33 | 0.5125 | 0.5714 |
| controlled_vs_baseline_no_context | distinct1 | 6 | 34 | 0 | 0.1500 | 0.1500 |
| controlled_vs_baseline_no_context | length_score | 22 | 15 | 3 | 0.5875 | 0.5946 |
| controlled_vs_baseline_no_context | sentence_score | 14 | 2 | 24 | 0.6500 | 0.8750 |
| controlled_vs_baseline_no_context | bertscore_f1 | 25 | 15 | 0 | 0.6250 | 0.6250 |
| controlled_vs_baseline_no_context | overall_quality | 39 | 1 | 0 | 0.9750 | 0.9750 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 40 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 35 | 2 | 3 | 0.9125 | 0.9459 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 23 | 17 | 0 | 0.5750 | 0.5750 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 40 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 40 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 35 | 2 | 3 | 0.9125 | 0.9459 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 4 | 3 | 33 | 0.5125 | 0.5714 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | 6 | 34 | 0 | 0.1500 | 0.1500 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 22 | 15 | 3 | 0.5875 | 0.5946 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 14 | 2 | 24 | 0.6500 | 0.8750 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 25 | 15 | 0 | 0.6250 | 0.6250 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 39 | 1 | 0 | 0.9750 | 0.9750 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.6250 | 0.0750 | 0.9250 |
| proposed_contextual | 0.0000 | 0.0000 | 0.5750 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.5500 | 0.0000 | 0.0000 |
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
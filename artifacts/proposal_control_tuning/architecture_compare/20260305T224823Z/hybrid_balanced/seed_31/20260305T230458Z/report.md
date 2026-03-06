# Proposal Alignment Evaluation Report

- Run ID: `20260305T230458Z`
- Generated: `2026-03-05T23:10:50.338303+00:00`
- Scenarios: `artifacts\proposal_control_tuning\architecture_compare\20260305T224823Z\hybrid_balanced\seed_31\20260305T230458Z\scenarios.jsonl`
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
| proposed_contextual_controlled | 0.2872 (0.2512, 0.3261) | 0.3798 (0.3200, 0.4467) | 0.8739 (0.8453, 0.9025) | 0.4404 (0.4176, 0.4664) | n/a |
| proposed_contextual_controlled_alt | 0.2693 (0.2437, 0.2982) | 0.3488 (0.2990, 0.4019) | 0.9021 (0.8794, 0.9255) | 0.4261 (0.4107, 0.4416) | n/a |
| proposed_contextual | 0.0850 (0.0443, 0.1303) | 0.1707 (0.1219, 0.2274) | 0.8277 (0.7964, 0.8592) | 0.2602 (0.2299, 0.2924) | n/a |
| candidate_no_context | 0.0316 (0.0191, 0.0477) | 0.1487 (0.1078, 0.1941) | 0.8333 (0.7979, 0.8693) | 0.2281 (0.2044, 0.2532) | n/a |
| baseline_no_context | 0.0468 (0.0277, 0.0674) | 0.1756 (0.1339, 0.2231) | 0.8925 (0.8718, 0.9134) | 0.2561 (0.2362, 0.2764) | n/a |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0534 | 1.6896 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0221 | 0.1483 |
| proposed_vs_candidate_no_context | naturalness | -0.0056 | -0.0067 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0682 | 2.3226 |
| proposed_vs_candidate_no_context | context_overlap | 0.0190 | 0.5142 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0327 | 0.6346 |
| proposed_vs_candidate_no_context | persona_style | -0.0207 | -0.0385 |
| proposed_vs_candidate_no_context | distinct1 | 0.0116 | 0.0123 |
| proposed_vs_candidate_no_context | length_score | -0.0583 | -0.1438 |
| proposed_vs_candidate_no_context | sentence_score | 0.0146 | 0.0190 |
| proposed_vs_candidate_no_context | overall_quality | 0.0322 | 0.1410 |
| proposed_vs_baseline_no_context | context_relevance | 0.0382 | 0.8165 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0049 | -0.0279 |
| proposed_vs_baseline_no_context | naturalness | -0.0648 | -0.0727 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0491 | 1.0143 |
| proposed_vs_baseline_no_context | context_overlap | 0.0128 | 0.2972 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 0.0141 | 0.2006 |
| proposed_vs_baseline_no_context | persona_style | -0.0808 | -0.1353 |
| proposed_vs_baseline_no_context | distinct1 | -0.0213 | -0.0219 |
| proposed_vs_baseline_no_context | length_score | -0.2306 | -0.3990 |
| proposed_vs_baseline_no_context | sentence_score | -0.1021 | -0.1156 |
| proposed_vs_baseline_no_context | overall_quality | 0.0041 | 0.0160 |
| controlled_vs_proposed_raw | context_relevance | 0.2022 | 2.3779 |
| controlled_vs_proposed_raw | persona_consistency | 0.2091 | 1.2245 |
| controlled_vs_proposed_raw | naturalness | 0.0462 | 0.0558 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2656 | 2.7230 |
| controlled_vs_proposed_raw | context_overlap | 0.0543 | 0.9720 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2359 | 2.7976 |
| controlled_vs_proposed_raw | persona_style | 0.1017 | 0.1969 |
| controlled_vs_proposed_raw | distinct1 | -0.0113 | -0.0119 |
| controlled_vs_proposed_raw | length_score | 0.2097 | 0.6040 |
| controlled_vs_proposed_raw | sentence_score | 0.0875 | 0.1120 |
| controlled_vs_proposed_raw | overall_quality | 0.1802 | 0.6923 |
| controlled_vs_candidate_no_context | context_relevance | 0.2556 | 8.0853 |
| controlled_vs_candidate_no_context | persona_consistency | 0.2311 | 1.5544 |
| controlled_vs_candidate_no_context | naturalness | 0.0406 | 0.0487 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3338 | 11.3699 |
| controlled_vs_candidate_no_context | context_overlap | 0.0733 | 1.9859 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2687 | 5.2077 |
| controlled_vs_candidate_no_context | persona_style | 0.0810 | 0.1508 |
| controlled_vs_candidate_no_context | distinct1 | 0.0003 | 0.0003 |
| controlled_vs_candidate_no_context | length_score | 0.1514 | 0.3733 |
| controlled_vs_candidate_no_context | sentence_score | 0.1021 | 0.1332 |
| controlled_vs_candidate_no_context | overall_quality | 0.2123 | 0.9308 |
| controlled_vs_baseline_no_context | context_relevance | 0.2404 | 5.1359 |
| controlled_vs_baseline_no_context | persona_consistency | 0.2042 | 1.1624 |
| controlled_vs_baseline_no_context | naturalness | -0.0187 | -0.0209 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3147 | 6.4993 |
| controlled_vs_baseline_no_context | context_overlap | 0.0671 | 1.5580 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2500 | 3.5593 |
| controlled_vs_baseline_no_context | persona_style | 0.0208 | 0.0349 |
| controlled_vs_baseline_no_context | distinct1 | -0.0326 | -0.0335 |
| controlled_vs_baseline_no_context | length_score | -0.0208 | -0.0361 |
| controlled_vs_baseline_no_context | sentence_score | -0.0146 | -0.0165 |
| controlled_vs_baseline_no_context | overall_quality | 0.1842 | 0.7193 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0179 | -0.0623 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0310 | -0.0816 |
| controlled_alt_vs_controlled_default | naturalness | 0.0282 | 0.0323 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0221 | -0.0608 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0081 | -0.0733 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0405 | -0.1264 |
| controlled_alt_vs_controlled_default | persona_style | 0.0069 | 0.0112 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0081 | 0.0086 |
| controlled_alt_vs_controlled_default | length_score | 0.0958 | 0.1721 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0583 | 0.0671 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0143 | -0.0324 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1843 | 2.1676 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1781 | 1.0429 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0744 | 0.0899 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2435 | 2.4964 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0462 | 0.8274 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1954 | 2.3176 |
| controlled_alt_vs_proposed_raw | persona_style | 0.1086 | 0.2103 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0032 | -0.0034 |
| controlled_alt_vs_proposed_raw | length_score | 0.3056 | 0.8800 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1458 | 0.1867 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1659 | 0.6374 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2377 | 7.5194 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.2001 | 1.3460 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0688 | 0.0826 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3117 | 10.6172 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0652 | 1.7670 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.2282 | 4.4231 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0879 | 0.1637 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0084 | 0.0089 |
| controlled_alt_vs_candidate_no_context | length_score | 0.2472 | 0.6096 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1604 | 0.2092 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1980 | 0.8682 |
| controlled_alt_vs_baseline_no_context | context_relevance | 0.2225 | 4.7537 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 0.1732 | 0.9860 |
| controlled_alt_vs_baseline_no_context | naturalness | 0.0096 | 0.0107 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 0.2926 | 6.0430 |
| controlled_alt_vs_baseline_no_context | context_overlap | 0.0590 | 1.3705 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 0.2095 | 2.9831 |
| controlled_alt_vs_baseline_no_context | persona_style | 0.0278 | 0.0465 |
| controlled_alt_vs_baseline_no_context | distinct1 | -0.0245 | -0.0252 |
| controlled_alt_vs_baseline_no_context | length_score | 0.0750 | 0.1298 |
| controlled_alt_vs_baseline_no_context | sentence_score | 0.0438 | 0.0495 |
| controlled_alt_vs_baseline_no_context | overall_quality | 0.1700 | 0.6635 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2404 | 5.1359 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.2042 | 1.1624 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0187 | -0.0209 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3147 | 6.4993 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0671 | 1.5580 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2500 | 3.5593 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0208 | 0.0349 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0326 | -0.0335 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | -0.0208 | -0.0361 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | -0.0146 | -0.0165 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1842 | 0.7193 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0534 | (0.0135, 0.0994) | 0.0020 | 0.0534 | (0.0099, 0.1007) | 0.0020 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0221 | (-0.0236, 0.0691) | 0.1723 | 0.0221 | (-0.0145, 0.0570) | 0.1283 |
| proposed_vs_candidate_no_context | naturalness | -0.0056 | (-0.0411, 0.0309) | 0.6140 | -0.0056 | (-0.0484, 0.0304) | 0.6207 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0682 | (0.0155, 0.1291) | 0.0033 | 0.0682 | (0.0119, 0.1299) | 0.0037 |
| proposed_vs_candidate_no_context | context_overlap | 0.0190 | (0.0057, 0.0333) | 0.0013 | 0.0190 | (0.0057, 0.0307) | 0.0010 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0327 | (-0.0228, 0.0893) | 0.1383 | 0.0327 | (-0.0076, 0.0696) | 0.0690 |
| proposed_vs_candidate_no_context | persona_style | -0.0207 | (-0.0558, 0.0040) | 0.9203 | -0.0207 | (-0.0523, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | distinct1 | 0.0116 | (-0.0066, 0.0290) | 0.1087 | 0.0116 | (-0.0090, 0.0286) | 0.1270 |
| proposed_vs_candidate_no_context | length_score | -0.0583 | (-0.1861, 0.0722) | 0.8153 | -0.0583 | (-0.2087, 0.0681) | 0.8020 |
| proposed_vs_candidate_no_context | sentence_score | 0.0146 | (-0.0729, 0.1021) | 0.4267 | 0.0146 | (-0.0921, 0.0933) | 0.4507 |
| proposed_vs_candidate_no_context | overall_quality | 0.0322 | (-0.0017, 0.0684) | 0.0353 | 0.0322 | (-0.0045, 0.0636) | 0.0480 |
| proposed_vs_baseline_no_context | context_relevance | 0.0382 | (-0.0106, 0.0912) | 0.0683 | 0.0382 | (-0.0257, 0.1040) | 0.1623 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0049 | (-0.0674, 0.0520) | 0.5647 | -0.0049 | (-0.0857, 0.0613) | 0.5720 |
| proposed_vs_baseline_no_context | naturalness | -0.0648 | (-0.1043, -0.0236) | 0.9997 | -0.0648 | (-0.1160, -0.0241) | 1.0000 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0491 | (-0.0159, 0.1181) | 0.0747 | 0.0491 | (-0.0365, 0.1336) | 0.1800 |
| proposed_vs_baseline_no_context | context_overlap | 0.0128 | (-0.0029, 0.0295) | 0.0620 | 0.0128 | (-0.0015, 0.0259) | 0.0443 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 0.0141 | (-0.0591, 0.0800) | 0.3360 | 0.0141 | (-0.0803, 0.0925) | 0.4050 |
| proposed_vs_baseline_no_context | persona_style | -0.0808 | (-0.1629, -0.0191) | 1.0000 | -0.0808 | (-0.2193, -0.0095) | 1.0000 |
| proposed_vs_baseline_no_context | distinct1 | -0.0213 | (-0.0403, -0.0028) | 0.9867 | -0.0213 | (-0.0367, -0.0093) | 0.9997 |
| proposed_vs_baseline_no_context | length_score | -0.2306 | (-0.3819, -0.0722) | 0.9983 | -0.2306 | (-0.4281, -0.0620) | 0.9987 |
| proposed_vs_baseline_no_context | sentence_score | -0.1021 | (-0.1750, -0.0292) | 0.9973 | -0.1021 | (-0.2227, -0.0241) | 1.0000 |
| proposed_vs_baseline_no_context | overall_quality | 0.0041 | (-0.0397, 0.0473) | 0.4213 | 0.0041 | (-0.0584, 0.0548) | 0.4770 |
| controlled_vs_proposed_raw | context_relevance | 0.2022 | (0.1444, 0.2598) | 0.0000 | 0.2022 | (0.1631, 0.2450) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.2091 | (0.1351, 0.2894) | 0.0000 | 0.2091 | (0.1288, 0.3117) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0462 | (0.0036, 0.0898) | 0.0163 | 0.0462 | (-0.0021, 0.1128) | 0.0323 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2656 | (0.1920, 0.3434) | 0.0000 | 0.2656 | (0.2182, 0.3162) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0543 | (0.0306, 0.0760) | 0.0000 | 0.0543 | (0.0318, 0.0781) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2359 | (0.1446, 0.3317) | 0.0000 | 0.2359 | (0.1436, 0.3522) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.1017 | (0.0273, 0.1911) | 0.0037 | 0.1017 | (0.0092, 0.2640) | 0.0233 |
| controlled_vs_proposed_raw | distinct1 | -0.0113 | (-0.0299, 0.0075) | 0.8893 | -0.0113 | (-0.0341, 0.0160) | 0.7980 |
| controlled_vs_proposed_raw | length_score | 0.2097 | (0.0542, 0.3681) | 0.0040 | 0.2097 | (0.0319, 0.4348) | 0.0073 |
| controlled_vs_proposed_raw | sentence_score | 0.0875 | (-0.0146, 0.1750) | 0.0500 | 0.0875 | (-0.0438, 0.2100) | 0.0963 |
| controlled_vs_proposed_raw | overall_quality | 0.1802 | (0.1412, 0.2203) | 0.0000 | 0.1802 | (0.1347, 0.2380) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2556 | (0.2169, 0.2946) | 0.0000 | 0.2556 | (0.2331, 0.2727) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.2311 | (0.1658, 0.3035) | 0.0000 | 0.2311 | (0.1614, 0.3218) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.0406 | (-0.0011, 0.0861) | 0.0290 | 0.0406 | (0.0031, 0.0859) | 0.0157 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3338 | (0.2856, 0.3876) | 0.0000 | 0.3338 | (0.3036, 0.3596) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0733 | (0.0550, 0.0921) | 0.0000 | 0.0733 | (0.0572, 0.0888) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2687 | (0.1915, 0.3564) | 0.0000 | 0.2687 | (0.1862, 0.3704) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.0810 | (0.0169, 0.1595) | 0.0037 | 0.0810 | (0.0016, 0.2263) | 0.0213 |
| controlled_vs_candidate_no_context | distinct1 | 0.0003 | (-0.0152, 0.0152) | 0.4933 | 0.0003 | (-0.0133, 0.0151) | 0.4870 |
| controlled_vs_candidate_no_context | length_score | 0.1514 | (-0.0209, 0.3278) | 0.0450 | 0.1514 | (0.0000, 0.3386) | 0.0247 |
| controlled_vs_candidate_no_context | sentence_score | 0.1021 | (-0.0146, 0.2042) | 0.0447 | 0.1021 | (0.0140, 0.1826) | 0.0200 |
| controlled_vs_candidate_no_context | overall_quality | 0.2123 | (0.1834, 0.2430) | 0.0000 | 0.2123 | (0.1828, 0.2502) | 0.0000 |
| controlled_vs_baseline_no_context | context_relevance | 0.2404 | (0.1969, 0.2836) | 0.0000 | 0.2404 | (0.2007, 0.2713) | 0.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 0.2042 | (0.1388, 0.2712) | 0.0000 | 0.2042 | (0.1397, 0.2709) | 0.0000 |
| controlled_vs_baseline_no_context | naturalness | -0.0187 | (-0.0544, 0.0164) | 0.8427 | -0.0187 | (-0.0426, 0.0072) | 0.9197 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3147 | (0.2576, 0.3700) | 0.0000 | 0.3147 | (0.2595, 0.3600) | 0.0000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0671 | (0.0478, 0.0897) | 0.0000 | 0.0671 | (0.0521, 0.0830) | 0.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2500 | (0.1720, 0.3333) | 0.0000 | 0.2500 | (0.1746, 0.3261) | 0.0000 |
| controlled_vs_baseline_no_context | persona_style | 0.0208 | (0.0000, 0.0556) | 0.1237 | 0.0208 | (0.0000, 0.0750) | 0.3457 |
| controlled_vs_baseline_no_context | distinct1 | -0.0326 | (-0.0517, -0.0121) | 0.9990 | -0.0326 | (-0.0494, -0.0140) | 1.0000 |
| controlled_vs_baseline_no_context | length_score | -0.0208 | (-0.1792, 0.1292) | 0.6117 | -0.0208 | (-0.1244, 0.0825) | 0.6560 |
| controlled_vs_baseline_no_context | sentence_score | -0.0146 | (-0.1021, 0.0729) | 0.6837 | -0.0146 | (-0.1273, 0.0750) | 0.6367 |
| controlled_vs_baseline_no_context | overall_quality | 0.1842 | (0.1541, 0.2166) | 0.0000 | 0.1842 | (0.1530, 0.2139) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0179 | (-0.0527, 0.0177) | 0.8290 | -0.0179 | (-0.0465, 0.0068) | 0.9057 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0310 | (-0.0954, 0.0302) | 0.8357 | -0.0310 | (-0.0726, 0.0039) | 0.9553 |
| controlled_alt_vs_controlled_default | naturalness | 0.0282 | (-0.0068, 0.0643) | 0.0587 | 0.0282 | (-0.0002, 0.0552) | 0.0260 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0221 | (-0.0695, 0.0227) | 0.8277 | -0.0221 | (-0.0599, 0.0070) | 0.9233 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0081 | (-0.0263, 0.0090) | 0.8183 | -0.0081 | (-0.0242, 0.0069) | 0.8547 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0405 | (-0.1177, 0.0294) | 0.8543 | -0.0405 | (-0.0947, 0.0011) | 0.9690 |
| controlled_alt_vs_controlled_default | persona_style | 0.0069 | (0.0000, 0.0208) | 0.3640 | 0.0069 | (0.0000, 0.0250) | 0.3340 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0081 | (-0.0107, 0.0263) | 0.1900 | 0.0081 | (-0.0070, 0.0202) | 0.1383 |
| controlled_alt_vs_controlled_default | length_score | 0.0958 | (-0.0611, 0.2556) | 0.1250 | 0.0958 | (-0.0286, 0.2217) | 0.0703 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0583 | (-0.0146, 0.1313) | 0.0917 | 0.0583 | (-0.0121, 0.1260) | 0.0770 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0143 | (-0.0389, 0.0083) | 0.8827 | -0.0143 | (-0.0361, 0.0032) | 0.9380 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1843 | (0.1356, 0.2266) | 0.0000 | 0.1843 | (0.1501, 0.2152) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1781 | (0.1090, 0.2454) | 0.0000 | 0.1781 | (0.1005, 0.2786) | 0.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0744 | (0.0343, 0.1161) | 0.0003 | 0.0744 | (0.0299, 0.1299) | 0.0003 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2435 | (0.1812, 0.2972) | 0.0000 | 0.2435 | (0.2024, 0.2809) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0462 | (0.0258, 0.0661) | 0.0000 | 0.0462 | (0.0251, 0.0609) | 0.0003 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1954 | (0.1173, 0.2762) | 0.0000 | 0.1954 | (0.1100, 0.2983) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 0.1086 | (0.0273, 0.2062) | 0.0037 | 0.1086 | (0.0000, 0.2718) | 0.0267 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0032 | (-0.0217, 0.0165) | 0.6263 | -0.0032 | (-0.0210, 0.0146) | 0.6280 |
| controlled_alt_vs_proposed_raw | length_score | 0.3056 | (0.1472, 0.4583) | 0.0000 | 0.3056 | (0.1347, 0.5088) | 0.0007 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1458 | (0.0583, 0.2333) | 0.0013 | 0.1458 | (0.0389, 0.2800) | 0.0023 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1659 | (0.1288, 0.2003) | 0.0000 | 0.1659 | (0.1323, 0.2059) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2377 | (0.2128, 0.2664) | 0.0000 | 0.2377 | (0.2075, 0.2670) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.2001 | (0.1434, 0.2614) | 0.0000 | 0.2001 | (0.1394, 0.2837) | 0.0000 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0688 | (0.0282, 0.1092) | 0.0007 | 0.0688 | (0.0182, 0.1150) | 0.0047 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3117 | (0.2782, 0.3488) | 0.0000 | 0.3117 | (0.2745, 0.3504) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0652 | (0.0508, 0.0798) | 0.0000 | 0.0652 | (0.0493, 0.0756) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.2282 | (0.1661, 0.2976) | 0.0000 | 0.2282 | (0.1631, 0.3125) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0879 | (0.0169, 0.1727) | 0.0040 | 0.0879 | (0.0015, 0.2358) | 0.0233 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0084 | (-0.0095, 0.0266) | 0.1823 | 0.0084 | (-0.0090, 0.0246) | 0.1717 |
| controlled_alt_vs_candidate_no_context | length_score | 0.2472 | (0.0833, 0.4069) | 0.0007 | 0.2472 | (0.0364, 0.4453) | 0.0127 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1604 | (0.0729, 0.2479) | 0.0013 | 0.1604 | (0.1167, 0.2283) | 0.0000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1980 | (0.1744, 0.2225) | 0.0000 | 0.1980 | (0.1724, 0.2270) | 0.0000 |
| controlled_alt_vs_baseline_no_context | context_relevance | 0.2225 | (0.1884, 0.2561) | 0.0000 | 0.2225 | (0.1732, 0.2694) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 0.1732 | (0.1032, 0.2430) | 0.0000 | 0.1732 | (0.0883, 0.2530) | 0.0000 |
| controlled_alt_vs_baseline_no_context | naturalness | 0.0096 | (-0.0187, 0.0423) | 0.2717 | 0.0096 | (-0.0274, 0.0430) | 0.3007 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 0.2926 | (0.2481, 0.3390) | 0.0000 | 0.2926 | (0.2282, 0.3558) | 0.0000 |
| controlled_alt_vs_baseline_no_context | context_overlap | 0.0590 | (0.0448, 0.0738) | 0.0000 | 0.0590 | (0.0422, 0.0707) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 0.2095 | (0.1250, 0.2879) | 0.0000 | 0.2095 | (0.0968, 0.2899) | 0.0007 |
| controlled_alt_vs_baseline_no_context | persona_style | 0.0278 | (0.0000, 0.0764) | 0.1230 | 0.0278 | (0.0000, 0.0952) | 0.3683 |
| controlled_alt_vs_baseline_no_context | distinct1 | -0.0245 | (-0.0390, -0.0092) | 0.9993 | -0.0245 | (-0.0373, -0.0126) | 1.0000 |
| controlled_alt_vs_baseline_no_context | length_score | 0.0750 | (-0.0625, 0.2181) | 0.1577 | 0.0750 | (-0.1197, 0.2458) | 0.2310 |
| controlled_alt_vs_baseline_no_context | sentence_score | 0.0437 | (-0.0437, 0.1313) | 0.2063 | 0.0437 | (-0.0159, 0.1273) | 0.1473 |
| controlled_alt_vs_baseline_no_context | overall_quality | 0.1700 | (0.1384, 0.1984) | 0.0000 | 0.1700 | (0.1278, 0.2042) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2404 | (0.1985, 0.2843) | 0.0000 | 0.2404 | (0.2009, 0.2714) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.2042 | (0.1383, 0.2722) | 0.0000 | 0.2042 | (0.1444, 0.2698) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0187 | (-0.0556, 0.0139) | 0.8510 | -0.0187 | (-0.0427, 0.0087) | 0.9163 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3147 | (0.2560, 0.3731) | 0.0000 | 0.3147 | (0.2597, 0.3608) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0671 | (0.0465, 0.0905) | 0.0000 | 0.0671 | (0.0517, 0.0833) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2500 | (0.1708, 0.3325) | 0.0000 | 0.2500 | (0.1800, 0.3254) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0208 | (0.0000, 0.0556) | 0.1233 | 0.0208 | (0.0000, 0.0714) | 0.3663 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0326 | (-0.0513, -0.0122) | 0.9980 | -0.0326 | (-0.0502, -0.0136) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | -0.0208 | (-0.1694, 0.1250) | 0.6053 | -0.0208 | (-0.1139, 0.0875) | 0.6610 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | -0.0146 | (-0.1021, 0.0729) | 0.7037 | -0.0146 | (-0.1212, 0.0700) | 0.6373 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1842 | (0.1521, 0.2163) | 0.0000 | 0.1842 | (0.1533, 0.2141) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 12 | 4 | 8 | 0.6667 | 0.7500 |
| proposed_vs_candidate_no_context | persona_consistency | 8 | 5 | 11 | 0.5625 | 0.6154 |
| proposed_vs_candidate_no_context | naturalness | 7 | 9 | 8 | 0.4583 | 0.4375 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 9 | 3 | 12 | 0.6250 | 0.7500 |
| proposed_vs_candidate_no_context | context_overlap | 12 | 4 | 8 | 0.6667 | 0.7500 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 7 | 3 | 14 | 0.5833 | 0.7000 |
| proposed_vs_candidate_no_context | persona_style | 1 | 2 | 21 | 0.4792 | 0.3333 |
| proposed_vs_candidate_no_context | distinct1 | 12 | 4 | 8 | 0.6667 | 0.7500 |
| proposed_vs_candidate_no_context | length_score | 7 | 8 | 9 | 0.4792 | 0.4667 |
| proposed_vs_candidate_no_context | sentence_score | 5 | 4 | 15 | 0.5208 | 0.5556 |
| proposed_vs_candidate_no_context | overall_quality | 10 | 6 | 8 | 0.5833 | 0.6250 |
| proposed_vs_baseline_no_context | context_relevance | 12 | 12 | 0 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context | persona_consistency | 7 | 9 | 8 | 0.4583 | 0.4375 |
| proposed_vs_baseline_no_context | naturalness | 9 | 15 | 0 | 0.3750 | 0.3750 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 8 | 7 | 9 | 0.5208 | 0.5333 |
| proposed_vs_baseline_no_context | context_overlap | 11 | 13 | 0 | 0.4583 | 0.4583 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 7 | 4 | 13 | 0.5625 | 0.6364 |
| proposed_vs_baseline_no_context | persona_style | 0 | 5 | 19 | 0.3958 | 0.0000 |
| proposed_vs_baseline_no_context | distinct1 | 6 | 14 | 4 | 0.3333 | 0.3000 |
| proposed_vs_baseline_no_context | length_score | 9 | 15 | 0 | 0.3750 | 0.3750 |
| proposed_vs_baseline_no_context | sentence_score | 1 | 8 | 15 | 0.3542 | 0.1111 |
| proposed_vs_baseline_no_context | overall_quality | 11 | 13 | 0 | 0.4583 | 0.4583 |
| controlled_vs_proposed_raw | context_relevance | 22 | 2 | 0 | 0.9167 | 0.9167 |
| controlled_vs_proposed_raw | persona_consistency | 19 | 1 | 4 | 0.8750 | 0.9500 |
| controlled_vs_proposed_raw | naturalness | 13 | 11 | 0 | 0.5417 | 0.5417 |
| controlled_vs_proposed_raw | context_keyword_coverage | 20 | 1 | 3 | 0.8958 | 0.9524 |
| controlled_vs_proposed_raw | context_overlap | 21 | 3 | 0 | 0.8750 | 0.8750 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 19 | 1 | 4 | 0.8750 | 0.9500 |
| controlled_vs_proposed_raw | persona_style | 5 | 0 | 19 | 0.6042 | 1.0000 |
| controlled_vs_proposed_raw | distinct1 | 8 | 13 | 3 | 0.3958 | 0.3810 |
| controlled_vs_proposed_raw | length_score | 14 | 10 | 0 | 0.5833 | 0.5833 |
| controlled_vs_proposed_raw | sentence_score | 9 | 3 | 12 | 0.6250 | 0.7500 |
| controlled_vs_proposed_raw | overall_quality | 22 | 2 | 0 | 0.9167 | 0.9167 |
| controlled_vs_candidate_no_context | context_relevance | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 21 | 0 | 3 | 0.9375 | 1.0000 |
| controlled_vs_candidate_no_context | naturalness | 14 | 10 | 0 | 0.5833 | 0.5833 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 21 | 0 | 3 | 0.9375 | 1.0000 |
| controlled_vs_candidate_no_context | persona_style | 5 | 0 | 19 | 0.6042 | 1.0000 |
| controlled_vs_candidate_no_context | distinct1 | 12 | 10 | 2 | 0.5417 | 0.5455 |
| controlled_vs_candidate_no_context | length_score | 12 | 12 | 0 | 0.5000 | 0.5000 |
| controlled_vs_candidate_no_context | sentence_score | 11 | 4 | 9 | 0.6458 | 0.7333 |
| controlled_vs_candidate_no_context | overall_quality | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | context_relevance | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_vs_baseline_no_context | persona_consistency | 22 | 1 | 1 | 0.9375 | 0.9565 |
| controlled_vs_baseline_no_context | naturalness | 11 | 13 | 0 | 0.4583 | 0.4583 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 23 | 0 | 1 | 0.9792 | 1.0000 |
| controlled_vs_baseline_no_context | context_overlap | 22 | 2 | 0 | 0.9167 | 0.9167 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 22 | 1 | 1 | 0.9375 | 0.9565 |
| controlled_vs_baseline_no_context | persona_style | 2 | 0 | 22 | 0.5417 | 1.0000 |
| controlled_vs_baseline_no_context | distinct1 | 7 | 17 | 0 | 0.2917 | 0.2917 |
| controlled_vs_baseline_no_context | length_score | 12 | 12 | 0 | 0.5000 | 0.5000 |
| controlled_vs_baseline_no_context | sentence_score | 4 | 5 | 15 | 0.4792 | 0.4444 |
| controlled_vs_baseline_no_context | overall_quality | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 8 | 12 | 4 | 0.4167 | 0.4000 |
| controlled_alt_vs_controlled_default | persona_consistency | 3 | 6 | 15 | 0.4375 | 0.3333 |
| controlled_alt_vs_controlled_default | naturalness | 11 | 10 | 3 | 0.5208 | 0.5238 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 4 | 8 | 12 | 0.4167 | 0.3333 |
| controlled_alt_vs_controlled_default | context_overlap | 9 | 11 | 4 | 0.4583 | 0.4500 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 3 | 6 | 15 | 0.4375 | 0.3333 |
| controlled_alt_vs_controlled_default | persona_style | 1 | 0 | 23 | 0.5208 | 1.0000 |
| controlled_alt_vs_controlled_default | distinct1 | 14 | 7 | 3 | 0.6458 | 0.6667 |
| controlled_alt_vs_controlled_default | length_score | 11 | 10 | 3 | 0.5208 | 0.5238 |
| controlled_alt_vs_controlled_default | sentence_score | 6 | 2 | 16 | 0.5833 | 0.7500 |
| controlled_alt_vs_controlled_default | overall_quality | 11 | 10 | 3 | 0.5208 | 0.5238 |
| controlled_alt_vs_proposed_raw | context_relevance | 22 | 2 | 0 | 0.9167 | 0.9167 |
| controlled_alt_vs_proposed_raw | persona_consistency | 18 | 2 | 4 | 0.8333 | 0.9000 |
| controlled_alt_vs_proposed_raw | naturalness | 17 | 7 | 0 | 0.7083 | 0.7083 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 22 | 2 | 0 | 0.9167 | 0.9167 |
| controlled_alt_vs_proposed_raw | context_overlap | 21 | 3 | 0 | 0.8750 | 0.8750 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 18 | 2 | 4 | 0.8333 | 0.9000 |
| controlled_alt_vs_proposed_raw | persona_style | 5 | 0 | 19 | 0.6042 | 1.0000 |
| controlled_alt_vs_proposed_raw | distinct1 | 10 | 12 | 2 | 0.4583 | 0.4545 |
| controlled_alt_vs_proposed_raw | length_score | 19 | 5 | 0 | 0.7917 | 0.7917 |
| controlled_alt_vs_proposed_raw | sentence_score | 12 | 2 | 10 | 0.7083 | 0.8571 |
| controlled_alt_vs_proposed_raw | overall_quality | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_alt_vs_candidate_no_context | context_relevance | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 21 | 0 | 3 | 0.9375 | 1.0000 |
| controlled_alt_vs_candidate_no_context | naturalness | 16 | 8 | 0 | 0.6667 | 0.6667 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 21 | 0 | 3 | 0.9375 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_style | 5 | 0 | 19 | 0.6042 | 1.0000 |
| controlled_alt_vs_candidate_no_context | distinct1 | 13 | 10 | 1 | 0.5625 | 0.5652 |
| controlled_alt_vs_candidate_no_context | length_score | 17 | 7 | 0 | 0.7083 | 0.7083 |
| controlled_alt_vs_candidate_no_context | sentence_score | 13 | 2 | 9 | 0.7292 | 0.8667 |
| controlled_alt_vs_candidate_no_context | overall_quality | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context | context_relevance | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 20 | 1 | 3 | 0.8958 | 0.9524 |
| controlled_alt_vs_baseline_no_context | naturalness | 11 | 13 | 0 | 0.4583 | 0.4583 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 23 | 0 | 1 | 0.9792 | 1.0000 |
| controlled_alt_vs_baseline_no_context | context_overlap | 22 | 2 | 0 | 0.9167 | 0.9167 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 20 | 1 | 3 | 0.8958 | 0.9524 |
| controlled_alt_vs_baseline_no_context | persona_style | 2 | 0 | 22 | 0.5417 | 1.0000 |
| controlled_alt_vs_baseline_no_context | distinct1 | 5 | 18 | 1 | 0.2292 | 0.2174 |
| controlled_alt_vs_baseline_no_context | length_score | 11 | 11 | 2 | 0.5000 | 0.5000 |
| controlled_alt_vs_baseline_no_context | sentence_score | 6 | 3 | 15 | 0.5625 | 0.6667 |
| controlled_alt_vs_baseline_no_context | overall_quality | 23 | 1 | 0 | 0.9583 | 0.9583 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 23 | 1 | 0 | 0.9583 | 0.9583 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 22 | 1 | 1 | 0.9375 | 0.9565 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 11 | 13 | 0 | 0.4583 | 0.4583 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 23 | 0 | 1 | 0.9792 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 22 | 2 | 0 | 0.9167 | 0.9167 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 22 | 1 | 1 | 0.9375 | 0.9565 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 2 | 0 | 22 | 0.5417 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | 7 | 17 | 0 | 0.2917 | 0.2917 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 12 | 12 | 0 | 0.5000 | 0.5000 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 4 | 5 | 15 | 0.4792 | 0.4444 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 24 | 0 | 0 | 1.0000 | 1.0000 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.1667 | 0.5833 | 0.4167 |
| proposed_contextual_controlled_alt | 0.0000 | 0.0000 | 0.2083 | 0.5000 | 0.5000 |
| proposed_contextual | 0.0000 | 0.0000 | 0.4167 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 |
| baseline_no_context | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `8`
- Unique template signatures: `23`
- Template signature ratio: `0.9583`
- Effective sample size by source clustering: `7.20`
- Effective sample size by template-signature clustering: `22.15`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: unavailable (disabled_by_flag).

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
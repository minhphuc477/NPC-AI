# Proposal Alignment Evaluation Report

- Run ID: `20260309T091617Z`
- Generated: `2026-03-09T09:17:12.108510+00:00`
- Scenarios: `artifacts\proposal\20260309T091617Z\scenarios.jsonl`
- Scenario count: `144`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off
- `baseline_no_context`: model `phi3:mini`, dynamic_context=off, response_control=off
- `baseline_no_context_phi3_latest`: model `phi3:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2735 (0.2568, 0.2915) | 0.3137 (0.2934, 0.3358) | 0.9015 (0.8913, 0.9110) | 0.4156 (0.4048, 0.4271) | n/a |
| proposed_contextual | 0.0869 (0.0699, 0.1054) | 0.1531 (0.1339, 0.1737) | 0.8203 (0.8074, 0.8330) | 0.2529 (0.2395, 0.2676) | n/a |
| candidate_no_context | 0.0261 (0.0209, 0.0319) | 0.1609 (0.1413, 0.1837) | 0.8244 (0.8119, 0.8376) | 0.2284 (0.2189, 0.2398) | n/a |
| baseline_no_context | 0.0367 (0.0300, 0.0436) | 0.1636 (0.1474, 0.1815) | 0.8929 (0.8833, 0.9019) | 0.2476 (0.2395, 0.2563) | n/a |
| baseline_no_context_phi3_latest | 0.0362 (0.0296, 0.0431) | 0.1605 (0.1431, 0.1797) | 0.8830 (0.8738, 0.8916) | 0.2443 (0.2364, 0.2531) | n/a |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0608 | 2.3271 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0078 | -0.0483 |
| proposed_vs_candidate_no_context | naturalness | -0.0040 | -0.0049 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0790 | 3.6163 |
| proposed_vs_candidate_no_context | context_overlap | 0.0183 | 0.5059 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0109 | -0.1579 |
| proposed_vs_candidate_no_context | persona_style | 0.0048 | 0.0091 |
| proposed_vs_candidate_no_context | distinct1 | -0.0021 | -0.0023 |
| proposed_vs_candidate_no_context | length_score | -0.0037 | -0.0110 |
| proposed_vs_candidate_no_context | sentence_score | -0.0243 | -0.0303 |
| proposed_vs_candidate_no_context | overall_quality | 0.0244 | 0.1069 |
| proposed_vs_baseline_no_context | context_relevance | 0.0502 | 1.3701 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0106 | -0.0645 |
| proposed_vs_baseline_no_context | naturalness | -0.0725 | -0.0812 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0641 | 1.7433 |
| proposed_vs_baseline_no_context | context_overlap | 0.0179 | 0.4914 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0047 | -0.0742 |
| proposed_vs_baseline_no_context | persona_style | -0.0341 | -0.0602 |
| proposed_vs_baseline_no_context | distinct1 | -0.0437 | -0.0445 |
| proposed_vs_baseline_no_context | length_score | -0.2218 | -0.3992 |
| proposed_vs_baseline_no_context | sentence_score | -0.1069 | -0.1207 |
| proposed_vs_baseline_no_context | overall_quality | 0.0053 | 0.0213 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 0.0508 | 1.4030 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | -0.0074 | -0.0459 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | -0.0626 | -0.0709 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.0640 | 1.7379 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 0.0198 | 0.5708 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | -0.0020 | -0.0324 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | -0.0290 | -0.0517 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | -0.0380 | -0.0389 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | -0.1933 | -0.3667 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | -0.0875 | -0.1010 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | 0.0085 | 0.0350 |
| controlled_vs_proposed_raw | context_relevance | 0.1865 | 2.1458 |
| controlled_vs_proposed_raw | persona_consistency | 0.1606 | 1.0492 |
| controlled_vs_proposed_raw | naturalness | 0.0812 | 0.0990 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2441 | 2.4196 |
| controlled_vs_proposed_raw | context_overlap | 0.0522 | 0.9602 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1934 | 3.3233 |
| controlled_vs_proposed_raw | persona_style | 0.0295 | 0.0553 |
| controlled_vs_proposed_raw | distinct1 | -0.0011 | -0.0012 |
| controlled_vs_proposed_raw | length_score | 0.3197 | 0.9577 |
| controlled_vs_proposed_raw | sentence_score | 0.1774 | 0.2278 |
| controlled_vs_proposed_raw | overall_quality | 0.1627 | 0.6435 |
| controlled_vs_candidate_no_context | context_relevance | 0.2473 | 9.4665 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1529 | 0.9503 |
| controlled_vs_candidate_no_context | naturalness | 0.0772 | 0.0936 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3231 | 14.7857 |
| controlled_vs_candidate_no_context | context_overlap | 0.0705 | 1.9519 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1825 | 2.6407 |
| controlled_vs_candidate_no_context | persona_style | 0.0343 | 0.0649 |
| controlled_vs_candidate_no_context | distinct1 | -0.0032 | -0.0034 |
| controlled_vs_candidate_no_context | length_score | 0.3160 | 0.9362 |
| controlled_vs_candidate_no_context | sentence_score | 0.1531 | 0.1907 |
| controlled_vs_candidate_no_context | overall_quality | 0.1871 | 0.8192 |
| controlled_vs_baseline_no_context | context_relevance | 0.2368 | 6.4559 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1501 | 0.9171 |
| controlled_vs_baseline_no_context | naturalness | 0.0087 | 0.0097 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3082 | 8.3811 |
| controlled_vs_baseline_no_context | context_overlap | 0.0701 | 1.9235 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1888 | 3.0026 |
| controlled_vs_baseline_no_context | persona_style | -0.0046 | -0.0082 |
| controlled_vs_baseline_no_context | distinct1 | -0.0448 | -0.0456 |
| controlled_vs_baseline_no_context | length_score | 0.0979 | 0.1762 |
| controlled_vs_baseline_no_context | sentence_score | 0.0705 | 0.0796 |
| controlled_vs_baseline_no_context | overall_quality | 0.1680 | 0.6785 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.2373 | 6.5592 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1533 | 0.9552 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | 0.0186 | 0.0210 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.3081 | 8.3624 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0719 | 2.0791 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1915 | 3.1831 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | 0.0005 | 0.0008 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0391 | -0.0400 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | 0.1264 | 0.2398 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.0899 | 0.1038 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1713 | 0.7010 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2368 | 6.4559 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1501 | 0.9171 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 0.0087 | 0.0097 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3082 | 8.3811 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0701 | 1.9235 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1888 | 3.0026 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | -0.0046 | -0.0082 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0448 | -0.0456 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.0979 | 0.1762 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0705 | 0.0796 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1680 | 0.6785 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.2373 | 6.5592 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1533 | 0.9552 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | 0.0186 | 0.0210 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.3081 | 8.3624 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0719 | 2.0791 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1915 | 3.1831 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | 0.0005 | 0.0008 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0391 | -0.0400 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | 0.1264 | 0.2398 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.0899 | 0.1038 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1713 | 0.7010 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0608 | (0.0432, 0.0780) | 0.0000 | 0.0608 | (0.0311, 0.0927) | 0.0000 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0078 | (-0.0256, 0.0084) | 0.8103 | -0.0078 | (-0.0201, 0.0026) | 0.9160 |
| proposed_vs_candidate_no_context | naturalness | -0.0040 | (-0.0172, 0.0094) | 0.7170 | -0.0040 | (-0.0196, 0.0097) | 0.6937 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0790 | (0.0565, 0.1028) | 0.0000 | 0.0790 | (0.0379, 0.1192) | 0.0000 |
| proposed_vs_candidate_no_context | context_overlap | 0.0183 | (0.0123, 0.0246) | 0.0000 | 0.0183 | (0.0101, 0.0256) | 0.0000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0109 | (-0.0322, 0.0083) | 0.8560 | -0.0109 | (-0.0232, -0.0002) | 0.9767 |
| proposed_vs_candidate_no_context | persona_style | 0.0048 | (-0.0097, 0.0212) | 0.2760 | 0.0048 | (-0.0104, 0.0213) | 0.2777 |
| proposed_vs_candidate_no_context | distinct1 | -0.0021 | (-0.0086, 0.0042) | 0.7437 | -0.0021 | (-0.0081, 0.0041) | 0.7697 |
| proposed_vs_candidate_no_context | length_score | -0.0037 | (-0.0546, 0.0486) | 0.5463 | -0.0037 | (-0.0646, 0.0509) | 0.5283 |
| proposed_vs_candidate_no_context | sentence_score | -0.0243 | (-0.0583, 0.0097) | 0.9237 | -0.0243 | (-0.0486, 0.0049) | 0.9610 |
| proposed_vs_candidate_no_context | overall_quality | 0.0244 | (0.0114, 0.0364) | 0.0000 | 0.0244 | (0.0055, 0.0431) | 0.0043 |
| proposed_vs_baseline_no_context | context_relevance | 0.0502 | (0.0338, 0.0687) | 0.0000 | 0.0502 | (0.0158, 0.0838) | 0.0027 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0106 | (-0.0332, 0.0116) | 0.8280 | -0.0106 | (-0.0552, 0.0329) | 0.6923 |
| proposed_vs_baseline_no_context | naturalness | -0.0725 | (-0.0891, -0.0564) | 1.0000 | -0.0725 | (-0.1041, -0.0441) | 1.0000 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0641 | (0.0410, 0.0886) | 0.0000 | 0.0641 | (0.0211, 0.1078) | 0.0007 |
| proposed_vs_baseline_no_context | context_overlap | 0.0179 | (0.0110, 0.0252) | 0.0000 | 0.0179 | (0.0078, 0.0276) | 0.0003 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0047 | (-0.0301, 0.0197) | 0.6660 | -0.0047 | (-0.0455, 0.0416) | 0.5970 |
| proposed_vs_baseline_no_context | persona_style | -0.0341 | (-0.0572, -0.0122) | 0.9993 | -0.0341 | (-0.0883, 0.0104) | 0.9007 |
| proposed_vs_baseline_no_context | distinct1 | -0.0437 | (-0.0507, -0.0360) | 1.0000 | -0.0437 | (-0.0550, -0.0323) | 1.0000 |
| proposed_vs_baseline_no_context | length_score | -0.2218 | (-0.2822, -0.1613) | 1.0000 | -0.2218 | (-0.3245, -0.1217) | 1.0000 |
| proposed_vs_baseline_no_context | sentence_score | -0.1069 | (-0.1434, -0.0681) | 1.0000 | -0.1069 | (-0.1750, -0.0462) | 1.0000 |
| proposed_vs_baseline_no_context | overall_quality | 0.0053 | (-0.0096, 0.0207) | 0.2410 | 0.0053 | (-0.0282, 0.0344) | 0.3723 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 0.0508 | (0.0334, 0.0685) | 0.0000 | 0.0508 | (0.0170, 0.0831) | 0.0010 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | -0.0074 | (-0.0295, 0.0155) | 0.7320 | -0.0074 | (-0.0513, 0.0502) | 0.6333 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | -0.0626 | (-0.0774, -0.0481) | 1.0000 | -0.0626 | (-0.0885, -0.0382) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.0640 | (0.0411, 0.0880) | 0.0000 | 0.0640 | (0.0224, 0.1060) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 0.0198 | (0.0137, 0.0263) | 0.0000 | 0.0198 | (0.0100, 0.0290) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | -0.0020 | (-0.0295, 0.0255) | 0.5793 | -0.0020 | (-0.0534, 0.0599) | 0.5463 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | -0.0290 | (-0.0527, -0.0085) | 0.9983 | -0.0290 | (-0.0762, 0.0116) | 0.8927 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | -0.0380 | (-0.0451, -0.0307) | 1.0000 | -0.0380 | (-0.0514, -0.0257) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | -0.1933 | (-0.2507, -0.1336) | 1.0000 | -0.1933 | (-0.2799, -0.1072) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | -0.0875 | (-0.1240, -0.0510) | 1.0000 | -0.0875 | (-0.1312, -0.0437) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | 0.0085 | (-0.0064, 0.0241) | 0.1337 | 0.0085 | (-0.0263, 0.0405) | 0.3130 |
| controlled_vs_proposed_raw | context_relevance | 0.1865 | (0.1649, 0.2088) | 0.0000 | 0.1865 | (0.1452, 0.2259) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.1606 | (0.1367, 0.1833) | 0.0000 | 0.1606 | (0.1306, 0.1859) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0812 | (0.0643, 0.0975) | 0.0000 | 0.0812 | (0.0427, 0.1181) | 0.0000 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2441 | (0.2147, 0.2729) | 0.0000 | 0.2441 | (0.1888, 0.2962) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0522 | (0.0428, 0.0623) | 0.0000 | 0.0522 | (0.0430, 0.0621) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1934 | (0.1653, 0.2204) | 0.0000 | 0.1934 | (0.1582, 0.2248) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.0295 | (0.0099, 0.0518) | 0.0000 | 0.0295 | (0.0029, 0.0642) | 0.0027 |
| controlled_vs_proposed_raw | distinct1 | -0.0011 | (-0.0084, 0.0066) | 0.6023 | -0.0011 | (-0.0185, 0.0181) | 0.5407 |
| controlled_vs_proposed_raw | length_score | 0.3197 | (0.2528, 0.3861) | 0.0000 | 0.3197 | (0.1799, 0.4505) | 0.0000 |
| controlled_vs_proposed_raw | sentence_score | 0.1774 | (0.1458, 0.2090) | 0.0000 | 0.1774 | (0.1191, 0.2358) | 0.0000 |
| controlled_vs_proposed_raw | overall_quality | 0.1627 | (0.1463, 0.1790) | 0.0000 | 0.1627 | (0.1337, 0.1922) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2473 | (0.2296, 0.2662) | 0.0000 | 0.2473 | (0.2105, 0.2828) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1529 | (0.1267, 0.1783) | 0.0000 | 0.1529 | (0.1231, 0.1769) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.0772 | (0.0603, 0.0928) | 0.0000 | 0.0772 | (0.0463, 0.1099) | 0.0000 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3231 | (0.3001, 0.3459) | 0.0000 | 0.3231 | (0.2736, 0.3695) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0705 | (0.0626, 0.0792) | 0.0000 | 0.0705 | (0.0642, 0.0754) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1825 | (0.1505, 0.2122) | 0.0000 | 0.1825 | (0.1487, 0.2107) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.0343 | (0.0106, 0.0593) | 0.0033 | 0.0343 | (0.0055, 0.0803) | 0.0243 |
| controlled_vs_candidate_no_context | distinct1 | -0.0032 | (-0.0108, 0.0041) | 0.8020 | -0.0032 | (-0.0177, 0.0137) | 0.6697 |
| controlled_vs_candidate_no_context | length_score | 0.3160 | (0.2484, 0.3799) | 0.0000 | 0.3160 | (0.2053, 0.4375) | 0.0000 |
| controlled_vs_candidate_no_context | sentence_score | 0.1531 | (0.1215, 0.1847) | 0.0000 | 0.1531 | (0.1045, 0.2066) | 0.0000 |
| controlled_vs_candidate_no_context | overall_quality | 0.1871 | (0.1727, 0.2019) | 0.0000 | 0.1871 | (0.1608, 0.2085) | 0.0000 |
| controlled_vs_baseline_no_context | context_relevance | 0.2368 | (0.2203, 0.2543) | 0.0000 | 0.2368 | (0.2040, 0.2688) | 0.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1501 | (0.1254, 0.1749) | 0.0000 | 0.1501 | (0.1218, 0.1747) | 0.0000 |
| controlled_vs_baseline_no_context | naturalness | 0.0087 | (-0.0047, 0.0223) | 0.1103 | 0.0087 | (-0.0077, 0.0282) | 0.1850 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3082 | (0.2874, 0.3307) | 0.0000 | 0.3082 | (0.2621, 0.3504) | 0.0000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0701 | (0.0618, 0.0798) | 0.0000 | 0.0701 | (0.0596, 0.0788) | 0.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1888 | (0.1576, 0.2180) | 0.0000 | 0.1888 | (0.1590, 0.2163) | 0.0000 |
| controlled_vs_baseline_no_context | persona_style | -0.0046 | (-0.0203, 0.0102) | 0.7267 | -0.0046 | (-0.0378, 0.0197) | 0.5903 |
| controlled_vs_baseline_no_context | distinct1 | -0.0448 | (-0.0508, -0.0381) | 1.0000 | -0.0448 | (-0.0545, -0.0343) | 1.0000 |
| controlled_vs_baseline_no_context | length_score | 0.0979 | (0.0368, 0.1593) | 0.0017 | 0.0979 | (0.0194, 0.1829) | 0.0043 |
| controlled_vs_baseline_no_context | sentence_score | 0.0705 | (0.0389, 0.1021) | 0.0000 | 0.0705 | (0.0340, 0.1142) | 0.0000 |
| controlled_vs_baseline_no_context | overall_quality | 0.1680 | (0.1554, 0.1800) | 0.0000 | 0.1680 | (0.1501, 0.1846) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.2373 | (0.2200, 0.2553) | 0.0000 | 0.2373 | (0.2037, 0.2688) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1533 | (0.1283, 0.1785) | 0.0000 | 0.1533 | (0.1248, 0.1812) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | 0.0186 | (0.0040, 0.0327) | 0.0073 | 0.0186 | (0.0027, 0.0346) | 0.0107 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.3081 | (0.2851, 0.3320) | 0.0000 | 0.3081 | (0.2636, 0.3518) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0719 | (0.0636, 0.0816) | 0.0000 | 0.0719 | (0.0640, 0.0791) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1915 | (0.1588, 0.2224) | 0.0000 | 0.1915 | (0.1599, 0.2235) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | 0.0005 | (-0.0138, 0.0161) | 0.4840 | 0.0005 | (-0.0252, 0.0217) | 0.4617 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0391 | (-0.0456, -0.0331) | 1.0000 | -0.0391 | (-0.0480, -0.0303) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | 0.1264 | (0.0604, 0.1907) | 0.0000 | 0.1264 | (0.0634, 0.1891) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.0899 | (0.0583, 0.1215) | 0.0000 | 0.0899 | (0.0608, 0.1288) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1713 | (0.1582, 0.1846) | 0.0000 | 0.1713 | (0.1583, 0.1850) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2368 | (0.2202, 0.2552) | 0.0000 | 0.2368 | (0.2034, 0.2676) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1501 | (0.1240, 0.1755) | 0.0000 | 0.1501 | (0.1218, 0.1756) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 0.0087 | (-0.0049, 0.0217) | 0.1027 | 0.0087 | (-0.0085, 0.0276) | 0.1870 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3082 | (0.2867, 0.3302) | 0.0000 | 0.3082 | (0.2618, 0.3520) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0701 | (0.0620, 0.0799) | 0.0000 | 0.0701 | (0.0595, 0.0789) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1888 | (0.1575, 0.2201) | 0.0000 | 0.1888 | (0.1593, 0.2148) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | -0.0046 | (-0.0202, 0.0101) | 0.7227 | -0.0046 | (-0.0378, 0.0197) | 0.5843 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0448 | (-0.0509, -0.0384) | 1.0000 | -0.0448 | (-0.0550, -0.0344) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.0979 | (0.0350, 0.1595) | 0.0017 | 0.0979 | (0.0171, 0.1852) | 0.0087 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0705 | (0.0389, 0.1021) | 0.0000 | 0.0705 | (0.0340, 0.1142) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1680 | (0.1550, 0.1802) | 0.0000 | 0.1680 | (0.1499, 0.1852) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.2373 | (0.2198, 0.2557) | 0.0000 | 0.2373 | (0.2050, 0.2678) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1533 | (0.1287, 0.1788) | 0.0000 | 0.1533 | (0.1260, 0.1813) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | 0.0186 | (0.0043, 0.0327) | 0.0053 | 0.0186 | (0.0031, 0.0347) | 0.0100 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.3081 | (0.2860, 0.3321) | 0.0000 | 0.3081 | (0.2646, 0.3511) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0719 | (0.0636, 0.0819) | 0.0000 | 0.0719 | (0.0643, 0.0790) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1915 | (0.1621, 0.2228) | 0.0000 | 0.1915 | (0.1603, 0.2231) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | 0.0005 | (-0.0147, 0.0150) | 0.4940 | 0.0005 | (-0.0257, 0.0217) | 0.4527 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0391 | (-0.0454, -0.0326) | 1.0000 | -0.0391 | (-0.0481, -0.0306) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | 0.1264 | (0.0604, 0.1901) | 0.0000 | 0.1264 | (0.0616, 0.1857) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.0899 | (0.0583, 0.1215) | 0.0000 | 0.0899 | (0.0632, 0.1288) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1713 | (0.1583, 0.1839) | 0.0000 | 0.1713 | (0.1582, 0.1846) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 70 | 25 | 49 | 0.6562 | 0.7368 |
| proposed_vs_candidate_no_context | persona_consistency | 22 | 20 | 102 | 0.5069 | 0.5238 |
| proposed_vs_candidate_no_context | naturalness | 45 | 50 | 49 | 0.4826 | 0.4737 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 58 | 10 | 76 | 0.6667 | 0.8529 |
| proposed_vs_candidate_no_context | context_overlap | 71 | 24 | 49 | 0.6632 | 0.7474 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 16 | 16 | 112 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | persona_style | 9 | 8 | 127 | 0.5035 | 0.5294 |
| proposed_vs_candidate_no_context | distinct1 | 39 | 46 | 59 | 0.4757 | 0.4588 |
| proposed_vs_candidate_no_context | length_score | 45 | 47 | 52 | 0.4931 | 0.4891 |
| proposed_vs_candidate_no_context | sentence_score | 21 | 31 | 92 | 0.4653 | 0.4038 |
| proposed_vs_candidate_no_context | overall_quality | 64 | 32 | 48 | 0.6111 | 0.6667 |
| proposed_vs_baseline_no_context | context_relevance | 85 | 59 | 0 | 0.5903 | 0.5903 |
| proposed_vs_baseline_no_context | persona_consistency | 26 | 35 | 83 | 0.4688 | 0.4262 |
| proposed_vs_baseline_no_context | naturalness | 35 | 109 | 0 | 0.2431 | 0.2431 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 59 | 21 | 64 | 0.6319 | 0.7375 |
| proposed_vs_baseline_no_context | context_overlap | 92 | 52 | 0 | 0.6389 | 0.6389 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 22 | 25 | 97 | 0.4896 | 0.4681 |
| proposed_vs_baseline_no_context | persona_style | 8 | 19 | 117 | 0.4618 | 0.2963 |
| proposed_vs_baseline_no_context | distinct1 | 18 | 111 | 15 | 0.1771 | 0.1395 |
| proposed_vs_baseline_no_context | length_score | 36 | 107 | 1 | 0.2535 | 0.2517 |
| proposed_vs_baseline_no_context | sentence_score | 17 | 61 | 66 | 0.3472 | 0.2179 |
| proposed_vs_baseline_no_context | overall_quality | 60 | 84 | 0 | 0.4167 | 0.4167 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 90 | 53 | 1 | 0.6285 | 0.6294 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | 24 | 37 | 83 | 0.4549 | 0.3934 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | 39 | 104 | 1 | 0.2743 | 0.2727 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 56 | 21 | 67 | 0.6215 | 0.7273 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 95 | 48 | 1 | 0.6632 | 0.6643 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 20 | 25 | 99 | 0.4826 | 0.4444 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | 11 | 17 | 116 | 0.4792 | 0.3929 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | 26 | 106 | 12 | 0.2222 | 0.1970 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | 38 | 101 | 5 | 0.2812 | 0.2734 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | 17 | 53 | 74 | 0.3750 | 0.2429 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | 62 | 82 | 0 | 0.4306 | 0.4306 |
| controlled_vs_proposed_raw | context_relevance | 134 | 10 | 0 | 0.9306 | 0.9306 |
| controlled_vs_proposed_raw | persona_consistency | 117 | 6 | 21 | 0.8854 | 0.9512 |
| controlled_vs_proposed_raw | naturalness | 110 | 34 | 0 | 0.7639 | 0.7639 |
| controlled_vs_proposed_raw | context_keyword_coverage | 127 | 7 | 10 | 0.9167 | 0.9478 |
| controlled_vs_proposed_raw | context_overlap | 130 | 14 | 0 | 0.9028 | 0.9028 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 116 | 4 | 24 | 0.8889 | 0.9667 |
| controlled_vs_proposed_raw | persona_style | 16 | 5 | 123 | 0.5382 | 0.7619 |
| controlled_vs_proposed_raw | distinct1 | 80 | 60 | 4 | 0.5694 | 0.5714 |
| controlled_vs_proposed_raw | length_score | 108 | 34 | 2 | 0.7569 | 0.7606 |
| controlled_vs_proposed_raw | sentence_score | 78 | 5 | 61 | 0.7535 | 0.9398 |
| controlled_vs_proposed_raw | overall_quality | 135 | 9 | 0 | 0.9375 | 0.9375 |
| controlled_vs_candidate_no_context | context_relevance | 144 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 116 | 12 | 16 | 0.8611 | 0.9062 |
| controlled_vs_candidate_no_context | naturalness | 110 | 34 | 0 | 0.7639 | 0.7639 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 141 | 0 | 3 | 0.9896 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 143 | 1 | 0 | 0.9931 | 0.9931 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 114 | 10 | 20 | 0.8611 | 0.9194 |
| controlled_vs_candidate_no_context | persona_style | 18 | 6 | 120 | 0.5417 | 0.7500 |
| controlled_vs_candidate_no_context | distinct1 | 76 | 66 | 2 | 0.5347 | 0.5352 |
| controlled_vs_candidate_no_context | length_score | 110 | 31 | 3 | 0.7743 | 0.7801 |
| controlled_vs_candidate_no_context | sentence_score | 67 | 4 | 73 | 0.7188 | 0.9437 |
| controlled_vs_candidate_no_context | overall_quality | 139 | 5 | 0 | 0.9653 | 0.9653 |
| controlled_vs_baseline_no_context | context_relevance | 144 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 117 | 11 | 16 | 0.8681 | 0.9141 |
| controlled_vs_baseline_no_context | naturalness | 79 | 65 | 0 | 0.5486 | 0.5486 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 141 | 0 | 3 | 0.9896 | 1.0000 |
| controlled_vs_baseline_no_context | context_overlap | 140 | 4 | 0 | 0.9722 | 0.9722 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 117 | 9 | 18 | 0.8750 | 0.9286 |
| controlled_vs_baseline_no_context | persona_style | 8 | 9 | 127 | 0.4965 | 0.4706 |
| controlled_vs_baseline_no_context | distinct1 | 16 | 121 | 7 | 0.1354 | 0.1168 |
| controlled_vs_baseline_no_context | length_score | 85 | 51 | 8 | 0.6181 | 0.6250 |
| controlled_vs_baseline_no_context | sentence_score | 40 | 11 | 93 | 0.6007 | 0.7843 |
| controlled_vs_baseline_no_context | overall_quality | 141 | 3 | 0 | 0.9792 | 0.9792 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 142 | 2 | 0 | 0.9861 | 0.9861 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 118 | 8 | 18 | 0.8819 | 0.9365 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | 94 | 50 | 0 | 0.6528 | 0.6528 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 140 | 1 | 3 | 0.9826 | 0.9929 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 142 | 2 | 0 | 0.9861 | 0.9861 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 117 | 6 | 21 | 0.8854 | 0.9512 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | 11 | 9 | 124 | 0.5069 | 0.5500 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | 17 | 122 | 5 | 0.1354 | 0.1223 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | 95 | 46 | 3 | 0.6701 | 0.6738 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 45 | 8 | 91 | 0.6285 | 0.8491 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 141 | 3 | 0 | 0.9792 | 0.9792 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 144 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 117 | 11 | 16 | 0.8681 | 0.9141 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 79 | 65 | 0 | 0.5486 | 0.5486 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 141 | 0 | 3 | 0.9896 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 140 | 4 | 0 | 0.9722 | 0.9722 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 117 | 9 | 18 | 0.8750 | 0.9286 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 8 | 9 | 127 | 0.4965 | 0.4706 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | 16 | 121 | 7 | 0.1354 | 0.1168 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 85 | 51 | 8 | 0.6181 | 0.6250 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 40 | 11 | 93 | 0.6007 | 0.7843 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 141 | 3 | 0 | 0.9792 | 0.9792 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 142 | 2 | 0 | 0.9861 | 0.9861 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 118 | 8 | 18 | 0.8819 | 0.9365 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | 94 | 50 | 0 | 0.6528 | 0.6528 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 140 | 1 | 3 | 0.9826 | 0.9929 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 142 | 2 | 0 | 0.9861 | 0.9861 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 117 | 6 | 21 | 0.8854 | 0.9512 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | 11 | 9 | 124 | 0.5069 | 0.5500 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | 17 | 122 | 5 | 0.1354 | 0.1223 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | 95 | 46 | 3 | 0.6701 | 0.6738 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | 45 | 8 | 91 | 0.6285 | 0.8491 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 141 | 3 | 0 | 0.9792 | 0.9792 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.0139 | 0.4583 | 0.1528 |
| proposed_contextual | 0.0000 | 0.0000 | 0.4444 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.4375 | 0.0000 | 0.0000 |
| baseline_no_context | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline_no_context_phi3_latest | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `8`
- Unique template signatures: `143`
- Template signature ratio: `0.9931`
- Effective sample size by source clustering: `8.00`
- Effective sample size by template-signature clustering: `142.03`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: unavailable (No BERTScore values found in merged scores.).

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
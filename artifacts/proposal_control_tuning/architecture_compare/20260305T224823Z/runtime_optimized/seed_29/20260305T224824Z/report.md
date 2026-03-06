# Proposal Alignment Evaluation Report

- Run ID: `20260305T224824Z`
- Generated: `2026-03-05T22:54:03.879901+00:00`
- Scenarios: `artifacts\proposal_control_tuning\architecture_compare\20260305T224823Z\runtime_optimized\seed_29\20260305T224824Z\scenarios.jsonl`
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
| proposed_contextual_controlled | 0.2615 (0.2267, 0.3007) | 0.3738 (0.2995, 0.4544) | 0.8831 (0.8547, 0.9107) | 0.4281 (0.4003, 0.4575) | n/a |
| proposed_contextual_controlled_alt | 0.2640 (0.2208, 0.3126) | 0.3208 (0.2785, 0.3706) | 0.8771 (0.8410, 0.9107) | 0.4083 (0.3864, 0.4317) | n/a |
| proposed_contextual | 0.0729 (0.0401, 0.1146) | 0.1424 (0.0933, 0.2065) | 0.8169 (0.7878, 0.8472) | 0.2427 (0.2107, 0.2772) | n/a |
| candidate_no_context | 0.0224 (0.0133, 0.0335) | 0.1827 (0.1174, 0.2575) | 0.8011 (0.7761, 0.8273) | 0.2313 (0.2009, 0.2652) | n/a |
| baseline_no_context | 0.0406 (0.0216, 0.0625) | 0.1680 (0.1272, 0.2116) | 0.8893 (0.8670, 0.9121) | 0.2501 (0.2302, 0.2717) | n/a |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0505 | 2.2498 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0403 | -0.2206 |
| proposed_vs_candidate_no_context | naturalness | 0.0158 | 0.0197 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0663 | 4.4681 |
| proposed_vs_candidate_no_context | context_overlap | 0.0136 | 0.3393 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0532 | -0.4542 |
| proposed_vs_candidate_no_context | persona_style | 0.0111 | 0.0249 |
| proposed_vs_candidate_no_context | distinct1 | 0.0032 | 0.0034 |
| proposed_vs_candidate_no_context | length_score | 0.0653 | 0.2304 |
| proposed_vs_candidate_no_context | sentence_score | 0.0146 | 0.0198 |
| proposed_vs_candidate_no_context | overall_quality | 0.0114 | 0.0491 |
| proposed_vs_baseline_no_context | context_relevance | 0.0324 | 0.7985 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0255 | -0.1521 |
| proposed_vs_baseline_no_context | naturalness | -0.0723 | -0.0813 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0401 | 0.9769 |
| proposed_vs_baseline_no_context | context_overlap | 0.0144 | 0.3652 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0024 | -0.0359 |
| proposed_vs_baseline_no_context | persona_style | -0.1182 | -0.2056 |
| proposed_vs_baseline_no_context | distinct1 | -0.0390 | -0.0402 |
| proposed_vs_baseline_no_context | length_score | -0.2181 | -0.3848 |
| proposed_vs_baseline_no_context | sentence_score | -0.1312 | -0.1486 |
| proposed_vs_baseline_no_context | overall_quality | -0.0074 | -0.0296 |
| controlled_vs_proposed_raw | context_relevance | 0.1886 | 2.5858 |
| controlled_vs_proposed_raw | persona_consistency | 0.2313 | 1.6242 |
| controlled_vs_proposed_raw | naturalness | 0.0661 | 0.0809 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2457 | 3.0288 |
| controlled_vs_proposed_raw | context_overlap | 0.0553 | 1.0281 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2538 | 3.9720 |
| controlled_vs_proposed_raw | persona_style | 0.1416 | 0.3102 |
| controlled_vs_proposed_raw | distinct1 | -0.0026 | -0.0028 |
| controlled_vs_proposed_raw | length_score | 0.2264 | 0.6494 |
| controlled_vs_proposed_raw | sentence_score | 0.2188 | 0.2909 |
| controlled_vs_proposed_raw | overall_quality | 0.1854 | 0.7639 |
| controlled_vs_candidate_no_context | context_relevance | 0.2391 | 10.6533 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1910 | 1.0453 |
| controlled_vs_candidate_no_context | naturalness | 0.0819 | 0.1022 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3120 | 21.0298 |
| controlled_vs_candidate_no_context | context_overlap | 0.0690 | 1.7164 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2006 | 1.7136 |
| controlled_vs_candidate_no_context | persona_style | 0.1527 | 0.3428 |
| controlled_vs_candidate_no_context | distinct1 | 0.0006 | 0.0006 |
| controlled_vs_candidate_no_context | length_score | 0.2917 | 1.0294 |
| controlled_vs_candidate_no_context | sentence_score | 0.2333 | 0.3164 |
| controlled_vs_candidate_no_context | overall_quality | 0.1968 | 0.8505 |
| controlled_vs_baseline_no_context | context_relevance | 0.2210 | 5.4491 |
| controlled_vs_baseline_no_context | persona_consistency | 0.2058 | 1.2252 |
| controlled_vs_baseline_no_context | naturalness | -0.0062 | -0.0070 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.2858 | 6.9646 |
| controlled_vs_baseline_no_context | context_overlap | 0.0697 | 1.7689 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2514 | 3.7934 |
| controlled_vs_baseline_no_context | persona_style | 0.0234 | 0.0408 |
| controlled_vs_baseline_no_context | distinct1 | -0.0415 | -0.0429 |
| controlled_vs_baseline_no_context | length_score | 0.0083 | 0.0147 |
| controlled_vs_baseline_no_context | sentence_score | 0.0875 | 0.0991 |
| controlled_vs_baseline_no_context | overall_quality | 0.1780 | 0.7117 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0025 | 0.0095 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0529 | -0.1416 |
| controlled_alt_vs_controlled_default | naturalness | -0.0059 | -0.0067 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0000 | 0.0000 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0083 | 0.0762 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0607 | -0.1911 |
| controlled_alt_vs_controlled_default | persona_style | -0.0218 | -0.0365 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0088 | -0.0095 |
| controlled_alt_vs_controlled_default | length_score | 0.0167 | 0.0290 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0292 | -0.0300 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0198 | -0.0463 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1911 | 2.6200 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1784 | 1.2526 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0602 | 0.0737 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2457 | 3.0288 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0637 | 1.1828 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1931 | 3.0217 |
| controlled_alt_vs_proposed_raw | persona_style | 0.1198 | 0.2624 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0114 | -0.0122 |
| controlled_alt_vs_proposed_raw | length_score | 0.2431 | 0.6972 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1896 | 0.2521 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1656 | 0.6823 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2416 | 10.7646 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1381 | 0.7556 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0760 | 0.0949 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3120 | 21.0298 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0773 | 1.9235 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1399 | 1.1949 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.1309 | 0.2938 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0082 | -0.0088 |
| controlled_alt_vs_candidate_no_context | length_score | 0.3083 | 1.0882 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.2042 | 0.2768 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1769 | 0.7649 |
| controlled_alt_vs_baseline_no_context | context_relevance | 0.2235 | 5.5106 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 0.1529 | 0.9100 |
| controlled_alt_vs_baseline_no_context | naturalness | -0.0121 | -0.0136 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 0.2858 | 6.9646 |
| controlled_alt_vs_baseline_no_context | context_overlap | 0.0781 | 1.9800 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 0.1907 | 2.8772 |
| controlled_alt_vs_baseline_no_context | persona_style | 0.0016 | 0.0028 |
| controlled_alt_vs_baseline_no_context | distinct1 | -0.0503 | -0.0519 |
| controlled_alt_vs_baseline_no_context | length_score | 0.0250 | 0.0441 |
| controlled_alt_vs_baseline_no_context | sentence_score | 0.0583 | 0.0660 |
| controlled_alt_vs_baseline_no_context | overall_quality | 0.1582 | 0.6325 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2210 | 5.4491 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.2058 | 1.2252 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0062 | -0.0070 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.2858 | 6.9646 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0697 | 1.7689 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2514 | 3.7934 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0234 | 0.0408 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0415 | -0.0429 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.0083 | 0.0147 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0875 | 0.0991 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1780 | 0.7117 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0505 | (0.0149, 0.0886) | 0.0003 | 0.0505 | (0.0142, 0.1017) | 0.0013 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0403 | (-0.1214, 0.0326) | 0.8430 | -0.0403 | (-0.1425, 0.0261) | 0.7923 |
| proposed_vs_candidate_no_context | naturalness | 0.0158 | (-0.0129, 0.0414) | 0.1243 | 0.0158 | (-0.0226, 0.0455) | 0.1820 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0663 | (0.0208, 0.1165) | 0.0007 | 0.0663 | (0.0192, 0.1342) | 0.0007 |
| proposed_vs_candidate_no_context | context_overlap | 0.0136 | (0.0015, 0.0265) | 0.0130 | 0.0136 | (0.0023, 0.0274) | 0.0090 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0532 | (-0.1405, 0.0258) | 0.8953 | -0.0532 | (-0.1633, 0.0182) | 0.8767 |
| proposed_vs_candidate_no_context | persona_style | 0.0111 | (-0.0583, 0.0854) | 0.3797 | 0.0111 | (-0.0518, 0.0889) | 0.3757 |
| proposed_vs_candidate_no_context | distinct1 | 0.0032 | (-0.0142, 0.0193) | 0.3553 | 0.0032 | (-0.0089, 0.0129) | 0.2717 |
| proposed_vs_candidate_no_context | length_score | 0.0653 | (-0.0500, 0.1819) | 0.1313 | 0.0653 | (-0.0841, 0.2050) | 0.1853 |
| proposed_vs_candidate_no_context | sentence_score | 0.0146 | (-0.0729, 0.1021) | 0.4330 | 0.0146 | (-0.1021, 0.1273) | 0.4393 |
| proposed_vs_candidate_no_context | overall_quality | 0.0114 | (-0.0307, 0.0505) | 0.2903 | 0.0114 | (-0.0406, 0.0571) | 0.2933 |
| proposed_vs_baseline_no_context | context_relevance | 0.0324 | (-0.0067, 0.0724) | 0.0473 | 0.0324 | (-0.0016, 0.0747) | 0.0297 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0255 | (-0.0950, 0.0492) | 0.7603 | -0.0255 | (-0.0900, 0.0446) | 0.7470 |
| proposed_vs_baseline_no_context | naturalness | -0.0723 | (-0.1119, -0.0310) | 0.9993 | -0.0723 | (-0.1028, -0.0263) | 0.9970 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0401 | (-0.0117, 0.0931) | 0.0700 | 0.0401 | (-0.0109, 0.0981) | 0.0610 |
| proposed_vs_baseline_no_context | context_overlap | 0.0144 | (0.0031, 0.0258) | 0.0070 | 0.0144 | (0.0025, 0.0260) | 0.0083 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0024 | (-0.0849, 0.0812) | 0.5450 | -0.0024 | (-0.0484, 0.0610) | 0.5320 |
| proposed_vs_baseline_no_context | persona_style | -0.1182 | (-0.2300, -0.0176) | 0.9897 | -0.1182 | (-0.3236, 0.0106) | 0.9107 |
| proposed_vs_baseline_no_context | distinct1 | -0.0390 | (-0.0557, -0.0207) | 1.0000 | -0.0390 | (-0.0520, -0.0235) | 1.0000 |
| proposed_vs_baseline_no_context | length_score | -0.2181 | (-0.3639, -0.0556) | 0.9960 | -0.2181 | (-0.3423, -0.0193) | 0.9813 |
| proposed_vs_baseline_no_context | sentence_score | -0.1313 | (-0.2188, -0.0437) | 0.9983 | -0.1313 | (-0.2167, -0.0500) | 1.0000 |
| proposed_vs_baseline_no_context | overall_quality | -0.0074 | (-0.0475, 0.0332) | 0.6480 | -0.0074 | (-0.0492, 0.0371) | 0.6180 |
| controlled_vs_proposed_raw | context_relevance | 0.1886 | (0.1500, 0.2320) | 0.0000 | 0.1886 | (0.1408, 0.2279) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.2313 | (0.1441, 0.3240) | 0.0000 | 0.2313 | (0.1385, 0.2994) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0661 | (0.0241, 0.1038) | 0.0007 | 0.0661 | (0.0186, 0.0981) | 0.0060 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2457 | (0.1936, 0.2997) | 0.0000 | 0.2457 | (0.1785, 0.3010) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0553 | (0.0369, 0.0758) | 0.0000 | 0.0553 | (0.0407, 0.0664) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2538 | (0.1488, 0.3694) | 0.0000 | 0.2538 | (0.1500, 0.3386) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.1416 | (0.0427, 0.2485) | 0.0003 | 0.1416 | (-0.0053, 0.3500) | 0.0393 |
| controlled_vs_proposed_raw | distinct1 | -0.0026 | (-0.0212, 0.0151) | 0.6207 | -0.0026 | (-0.0231, 0.0136) | 0.6387 |
| controlled_vs_proposed_raw | length_score | 0.2264 | (0.0583, 0.3861) | 0.0043 | 0.2264 | (0.0383, 0.3529) | 0.0117 |
| controlled_vs_proposed_raw | sentence_score | 0.2188 | (0.1458, 0.2771) | 0.0000 | 0.2188 | (0.1474, 0.2833) | 0.0000 |
| controlled_vs_proposed_raw | overall_quality | 0.1854 | (0.1393, 0.2312) | 0.0000 | 0.1854 | (0.1359, 0.2197) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2391 | (0.2011, 0.2784) | 0.0000 | 0.2391 | (0.1783, 0.2982) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1910 | (0.1125, 0.2750) | 0.0000 | 0.1910 | (0.0870, 0.2745) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.0819 | (0.0398, 0.1224) | 0.0000 | 0.0819 | (0.0250, 0.1243) | 0.0030 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3120 | (0.2610, 0.3630) | 0.0000 | 0.3120 | (0.2322, 0.3890) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0690 | (0.0507, 0.0889) | 0.0000 | 0.0690 | (0.0480, 0.0884) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2006 | (0.1071, 0.3024) | 0.0000 | 0.2006 | (0.0957, 0.2952) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.1527 | (0.0344, 0.2757) | 0.0043 | 0.1527 | (-0.0096, 0.3516) | 0.0430 |
| controlled_vs_candidate_no_context | distinct1 | 0.0006 | (-0.0206, 0.0203) | 0.4723 | 0.0006 | (-0.0211, 0.0205) | 0.4867 |
| controlled_vs_candidate_no_context | length_score | 0.2917 | (0.1000, 0.4750) | 0.0020 | 0.2917 | (0.0551, 0.4692) | 0.0100 |
| controlled_vs_candidate_no_context | sentence_score | 0.2333 | (0.1604, 0.2917) | 0.0000 | 0.2333 | (0.1333, 0.3080) | 0.0000 |
| controlled_vs_candidate_no_context | overall_quality | 0.1968 | (0.1557, 0.2394) | 0.0000 | 0.1968 | (0.1356, 0.2408) | 0.0000 |
| controlled_vs_baseline_no_context | context_relevance | 0.2210 | (0.1803, 0.2654) | 0.0000 | 0.2210 | (0.1600, 0.2843) | 0.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 0.2058 | (0.1447, 0.2715) | 0.0000 | 0.2058 | (0.1485, 0.2652) | 0.0000 |
| controlled_vs_baseline_no_context | naturalness | -0.0062 | (-0.0485, 0.0334) | 0.5963 | -0.0062 | (-0.0438, 0.0283) | 0.6240 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.2858 | (0.2291, 0.3434) | 0.0000 | 0.2858 | (0.2025, 0.3707) | 0.0000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0697 | (0.0521, 0.0875) | 0.0000 | 0.0697 | (0.0561, 0.0827) | 0.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2514 | (0.1720, 0.3379) | 0.0000 | 0.2514 | (0.1848, 0.3213) | 0.0000 |
| controlled_vs_baseline_no_context | persona_style | 0.0234 | (-0.0139, 0.0642) | 0.1233 | 0.0234 | (-0.0019, 0.0585) | 0.1050 |
| controlled_vs_baseline_no_context | distinct1 | -0.0415 | (-0.0588, -0.0245) | 1.0000 | -0.0415 | (-0.0571, -0.0276) | 1.0000 |
| controlled_vs_baseline_no_context | length_score | 0.0083 | (-0.1750, 0.1847) | 0.4543 | 0.0083 | (-0.1580, 0.1741) | 0.4470 |
| controlled_vs_baseline_no_context | sentence_score | 0.0875 | (0.0146, 0.1604) | 0.0150 | 0.0875 | (0.0375, 0.1500) | 0.0000 |
| controlled_vs_baseline_no_context | overall_quality | 0.1780 | (0.1481, 0.2115) | 0.0000 | 0.1780 | (0.1432, 0.2077) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0025 | (-0.0440, 0.0533) | 0.4663 | 0.0025 | (-0.0499, 0.0708) | 0.4607 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0529 | (-0.1186, 0.0041) | 0.9650 | -0.0529 | (-0.1184, 0.0014) | 0.9590 |
| controlled_alt_vs_controlled_default | naturalness | -0.0059 | (-0.0501, 0.0353) | 0.6180 | -0.0059 | (-0.0424, 0.0380) | 0.6140 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0000 | (-0.0638, 0.0676) | 0.5133 | -0.0000 | (-0.0727, 0.0949) | 0.5247 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0083 | (-0.0089, 0.0253) | 0.1667 | 0.0083 | (-0.0042, 0.0242) | 0.1107 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0607 | (-0.1357, 0.0048) | 0.9610 | -0.0607 | (-0.1321, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | persona_style | -0.0218 | (-0.0556, 0.0065) | 0.9140 | -0.0218 | (-0.0714, 0.0138) | 0.8803 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0088 | (-0.0409, 0.0195) | 0.7033 | -0.0088 | (-0.0449, 0.0278) | 0.6573 |
| controlled_alt_vs_controlled_default | length_score | 0.0167 | (-0.1556, 0.1875) | 0.4043 | 0.0167 | (-0.1071, 0.1725) | 0.4387 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0292 | (-0.0875, 0.0292) | 0.9103 | -0.0292 | (-0.0833, 0.0280) | 0.9033 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0198 | (-0.0499, 0.0054) | 0.9340 | -0.0198 | (-0.0535, 0.0106) | 0.8393 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1911 | (0.1440, 0.2384) | 0.0000 | 0.1911 | (0.1454, 0.2392) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1784 | (0.1049, 0.2443) | 0.0000 | 0.1784 | (0.1057, 0.2400) | 0.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0602 | (0.0131, 0.1054) | 0.0047 | 0.0602 | (0.0168, 0.1096) | 0.0013 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2457 | (0.1807, 0.3087) | 0.0000 | 0.2457 | (0.1788, 0.3091) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0637 | (0.0463, 0.0823) | 0.0000 | 0.0637 | (0.0488, 0.0806) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1931 | (0.1105, 0.2683) | 0.0000 | 0.1931 | (0.1272, 0.2425) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 0.1198 | (0.0200, 0.2331) | 0.0080 | 0.1198 | (-0.0238, 0.3350) | 0.1570 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0114 | (-0.0363, 0.0104) | 0.8277 | -0.0114 | (-0.0333, 0.0138) | 0.8110 |
| controlled_alt_vs_proposed_raw | length_score | 0.2431 | (0.0569, 0.4264) | 0.0047 | 0.2431 | (0.0412, 0.4309) | 0.0080 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1896 | (0.1021, 0.2625) | 0.0000 | 0.1896 | (0.0955, 0.2520) | 0.0003 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1656 | (0.1290, 0.1995) | 0.0000 | 0.1656 | (0.1290, 0.1982) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2416 | (0.1968, 0.2896) | 0.0000 | 0.2416 | (0.1940, 0.2965) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1381 | (0.0550, 0.2152) | 0.0003 | 0.1381 | (-0.0022, 0.2455) | 0.0267 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0760 | (0.0311, 0.1185) | 0.0007 | 0.0760 | (0.0240, 0.1252) | 0.0020 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3120 | (0.2520, 0.3765) | 0.0000 | 0.3120 | (0.2494, 0.3831) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0773 | (0.0597, 0.0961) | 0.0000 | 0.0773 | (0.0608, 0.0929) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1399 | (0.0541, 0.2228) | 0.0013 | 0.1399 | (0.0041, 0.2404) | 0.0223 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.1309 | (0.0095, 0.2542) | 0.0177 | 0.1309 | (-0.0472, 0.3497) | 0.1177 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0082 | (-0.0311, 0.0117) | 0.7653 | -0.0082 | (-0.0251, 0.0132) | 0.7990 |
| controlled_alt_vs_candidate_no_context | length_score | 0.3083 | (0.1111, 0.4820) | 0.0017 | 0.3083 | (0.0936, 0.5044) | 0.0027 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.2042 | (0.1313, 0.2771) | 0.0000 | 0.2042 | (0.1105, 0.2827) | 0.0000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1769 | (0.1424, 0.2122) | 0.0000 | 0.1769 | (0.1216, 0.2193) | 0.0000 |
| controlled_alt_vs_baseline_no_context | context_relevance | 0.2235 | (0.1709, 0.2794) | 0.0000 | 0.2235 | (0.1758, 0.2742) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 0.1529 | (0.1217, 0.1854) | 0.0000 | 0.1529 | (0.1288, 0.1855) | 0.0000 |
| controlled_alt_vs_baseline_no_context | naturalness | -0.0121 | (-0.0415, 0.0164) | 0.7793 | -0.0121 | (-0.0438, 0.0247) | 0.7330 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 0.2858 | (0.2178, 0.3578) | 0.0000 | 0.2858 | (0.2212, 0.3595) | 0.0000 |
| controlled_alt_vs_baseline_no_context | context_overlap | 0.0781 | (0.0600, 0.0973) | 0.0000 | 0.0781 | (0.0701, 0.0878) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 0.1907 | (0.1506, 0.2316) | 0.0000 | 0.1907 | (0.1634, 0.2290) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_style | 0.0016 | (-0.0262, 0.0275) | 0.4507 | 0.0016 | (-0.0273, 0.0280) | 0.4510 |
| controlled_alt_vs_baseline_no_context | distinct1 | -0.0503 | (-0.0756, -0.0273) | 1.0000 | -0.0503 | (-0.0754, -0.0247) | 1.0000 |
| controlled_alt_vs_baseline_no_context | length_score | 0.0250 | (-0.1056, 0.1417) | 0.3680 | 0.0250 | (-0.1281, 0.1911) | 0.3917 |
| controlled_alt_vs_baseline_no_context | sentence_score | 0.0583 | (-0.0292, 0.1604) | 0.1437 | 0.0583 | (-0.0292, 0.1432) | 0.1230 |
| controlled_alt_vs_baseline_no_context | overall_quality | 0.1582 | (0.1332, 0.1852) | 0.0000 | 0.1582 | (0.1394, 0.1816) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2210 | (0.1767, 0.2652) | 0.0000 | 0.2210 | (0.1575, 0.2882) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.2058 | (0.1437, 0.2730) | 0.0000 | 0.2058 | (0.1520, 0.2614) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0062 | (-0.0482, 0.0354) | 0.6053 | -0.0062 | (-0.0431, 0.0280) | 0.6323 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.2858 | (0.2278, 0.3428) | 0.0000 | 0.2858 | (0.2015, 0.3722) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0697 | (0.0529, 0.0871) | 0.0000 | 0.0697 | (0.0561, 0.0825) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2514 | (0.1716, 0.3375) | 0.0000 | 0.2514 | (0.1840, 0.3202) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0234 | (-0.0148, 0.0660) | 0.1347 | 0.0234 | (-0.0018, 0.0585) | 0.0973 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0415 | (-0.0587, -0.0243) | 1.0000 | -0.0415 | (-0.0570, -0.0283) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.0083 | (-0.1833, 0.1875) | 0.4690 | 0.0083 | (-0.1617, 0.1788) | 0.4390 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0875 | (0.0146, 0.1604) | 0.0143 | 0.0875 | (0.0362, 0.1522) | 0.0003 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1780 | (0.1466, 0.2107) | 0.0000 | 0.1780 | (0.1446, 0.2064) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 10 | 3 | 11 | 0.6458 | 0.7692 |
| proposed_vs_candidate_no_context | persona_consistency | 6 | 5 | 13 | 0.5208 | 0.5455 |
| proposed_vs_candidate_no_context | naturalness | 8 | 5 | 11 | 0.5625 | 0.6154 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 8 | 1 | 15 | 0.6458 | 0.8889 |
| proposed_vs_candidate_no_context | context_overlap | 10 | 3 | 11 | 0.6458 | 0.7692 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 3 | 5 | 16 | 0.4583 | 0.3750 |
| proposed_vs_candidate_no_context | persona_style | 3 | 2 | 19 | 0.5208 | 0.6000 |
| proposed_vs_candidate_no_context | distinct1 | 7 | 6 | 11 | 0.5208 | 0.5385 |
| proposed_vs_candidate_no_context | length_score | 7 | 6 | 11 | 0.5208 | 0.5385 |
| proposed_vs_candidate_no_context | sentence_score | 5 | 4 | 15 | 0.5208 | 0.5556 |
| proposed_vs_candidate_no_context | overall_quality | 9 | 4 | 11 | 0.6042 | 0.6923 |
| proposed_vs_baseline_no_context | context_relevance | 16 | 8 | 0 | 0.6667 | 0.6667 |
| proposed_vs_baseline_no_context | persona_consistency | 6 | 9 | 9 | 0.4375 | 0.4000 |
| proposed_vs_baseline_no_context | naturalness | 5 | 19 | 0 | 0.2083 | 0.2083 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 7 | 3 | 14 | 0.5833 | 0.7000 |
| proposed_vs_baseline_no_context | context_overlap | 16 | 8 | 0 | 0.6667 | 0.6667 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 4 | 5 | 15 | 0.4792 | 0.4444 |
| proposed_vs_baseline_no_context | persona_style | 3 | 7 | 14 | 0.4167 | 0.3000 |
| proposed_vs_baseline_no_context | distinct1 | 3 | 21 | 0 | 0.1250 | 0.1250 |
| proposed_vs_baseline_no_context | length_score | 6 | 17 | 1 | 0.2708 | 0.2609 |
| proposed_vs_baseline_no_context | sentence_score | 2 | 11 | 11 | 0.3125 | 0.1538 |
| proposed_vs_baseline_no_context | overall_quality | 10 | 14 | 0 | 0.4167 | 0.4167 |
| controlled_vs_proposed_raw | context_relevance | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_proposed_raw | persona_consistency | 20 | 2 | 2 | 0.8750 | 0.9091 |
| controlled_vs_proposed_raw | naturalness | 18 | 6 | 0 | 0.7500 | 0.7500 |
| controlled_vs_proposed_raw | context_keyword_coverage | 23 | 0 | 1 | 0.9792 | 1.0000 |
| controlled_vs_proposed_raw | context_overlap | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 19 | 2 | 3 | 0.8542 | 0.9048 |
| controlled_vs_proposed_raw | persona_style | 7 | 1 | 16 | 0.6250 | 0.8750 |
| controlled_vs_proposed_raw | distinct1 | 11 | 13 | 0 | 0.4583 | 0.4583 |
| controlled_vs_proposed_raw | length_score | 17 | 7 | 0 | 0.7083 | 0.7083 |
| controlled_vs_proposed_raw | sentence_score | 15 | 0 | 9 | 0.8125 | 1.0000 |
| controlled_vs_proposed_raw | overall_quality | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_vs_candidate_no_context | context_relevance | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 18 | 4 | 2 | 0.7917 | 0.8182 |
| controlled_vs_candidate_no_context | naturalness | 17 | 7 | 0 | 0.7083 | 0.7083 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 18 | 3 | 3 | 0.8125 | 0.8571 |
| controlled_vs_candidate_no_context | persona_style | 9 | 2 | 13 | 0.6458 | 0.8182 |
| controlled_vs_candidate_no_context | distinct1 | 14 | 10 | 0 | 0.5833 | 0.5833 |
| controlled_vs_candidate_no_context | length_score | 16 | 8 | 0 | 0.6667 | 0.6667 |
| controlled_vs_candidate_no_context | sentence_score | 16 | 0 | 8 | 0.8333 | 1.0000 |
| controlled_vs_candidate_no_context | overall_quality | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | context_relevance | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 21 | 1 | 2 | 0.9167 | 0.9545 |
| controlled_vs_baseline_no_context | naturalness | 13 | 11 | 0 | 0.5417 | 0.5417 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 23 | 0 | 1 | 0.9792 | 1.0000 |
| controlled_vs_baseline_no_context | context_overlap | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 21 | 1 | 2 | 0.9167 | 0.9545 |
| controlled_vs_baseline_no_context | persona_style | 4 | 2 | 18 | 0.5417 | 0.6667 |
| controlled_vs_baseline_no_context | distinct1 | 4 | 20 | 0 | 0.1667 | 0.1667 |
| controlled_vs_baseline_no_context | length_score | 13 | 11 | 0 | 0.5417 | 0.5417 |
| controlled_vs_baseline_no_context | sentence_score | 7 | 1 | 16 | 0.6250 | 0.8750 |
| controlled_vs_baseline_no_context | overall_quality | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 10 | 9 | 5 | 0.5208 | 0.5263 |
| controlled_alt_vs_controlled_default | persona_consistency | 3 | 6 | 15 | 0.4375 | 0.3333 |
| controlled_alt_vs_controlled_default | naturalness | 10 | 9 | 5 | 0.5208 | 0.5263 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 6 | 7 | 11 | 0.4792 | 0.4615 |
| controlled_alt_vs_controlled_default | context_overlap | 12 | 7 | 5 | 0.6042 | 0.6316 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 2 | 5 | 17 | 0.4375 | 0.2857 |
| controlled_alt_vs_controlled_default | persona_style | 1 | 3 | 20 | 0.4583 | 0.2500 |
| controlled_alt_vs_controlled_default | distinct1 | 11 | 8 | 5 | 0.5625 | 0.5789 |
| controlled_alt_vs_controlled_default | length_score | 9 | 8 | 7 | 0.5208 | 0.5294 |
| controlled_alt_vs_controlled_default | sentence_score | 1 | 3 | 20 | 0.4583 | 0.2500 |
| controlled_alt_vs_controlled_default | overall_quality | 8 | 11 | 5 | 0.4375 | 0.4211 |
| controlled_alt_vs_proposed_raw | context_relevance | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_alt_vs_proposed_raw | persona_consistency | 22 | 2 | 0 | 0.9167 | 0.9167 |
| controlled_alt_vs_proposed_raw | naturalness | 16 | 8 | 0 | 0.6667 | 0.6667 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 22 | 1 | 1 | 0.9375 | 0.9565 |
| controlled_alt_vs_proposed_raw | context_overlap | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 21 | 2 | 1 | 0.8958 | 0.9130 |
| controlled_alt_vs_proposed_raw | persona_style | 6 | 2 | 16 | 0.5833 | 0.7500 |
| controlled_alt_vs_proposed_raw | distinct1 | 10 | 14 | 0 | 0.4167 | 0.4167 |
| controlled_alt_vs_proposed_raw | length_score | 16 | 8 | 0 | 0.6667 | 0.6667 |
| controlled_alt_vs_proposed_raw | sentence_score | 14 | 1 | 9 | 0.7708 | 0.9333 |
| controlled_alt_vs_proposed_raw | overall_quality | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_alt_vs_candidate_no_context | context_relevance | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 19 | 5 | 0 | 0.7917 | 0.7917 |
| controlled_alt_vs_candidate_no_context | naturalness | 17 | 7 | 0 | 0.7083 | 0.7083 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 23 | 0 | 1 | 0.9792 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 19 | 4 | 1 | 0.8125 | 0.8261 |
| controlled_alt_vs_candidate_no_context | persona_style | 7 | 2 | 15 | 0.6042 | 0.7778 |
| controlled_alt_vs_candidate_no_context | distinct1 | 13 | 10 | 1 | 0.5625 | 0.5652 |
| controlled_alt_vs_candidate_no_context | length_score | 15 | 7 | 2 | 0.6667 | 0.6818 |
| controlled_alt_vs_candidate_no_context | sentence_score | 14 | 0 | 10 | 0.7917 | 1.0000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_alt_vs_baseline_no_context | context_relevance | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 21 | 0 | 3 | 0.9375 | 1.0000 |
| controlled_alt_vs_baseline_no_context | naturalness | 11 | 13 | 0 | 0.4583 | 0.4583 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 22 | 0 | 2 | 0.9583 | 1.0000 |
| controlled_alt_vs_baseline_no_context | context_overlap | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 21 | 0 | 3 | 0.9375 | 1.0000 |
| controlled_alt_vs_baseline_no_context | persona_style | 2 | 1 | 21 | 0.5208 | 0.6667 |
| controlled_alt_vs_baseline_no_context | distinct1 | 3 | 21 | 0 | 0.1250 | 0.1250 |
| controlled_alt_vs_baseline_no_context | length_score | 14 | 9 | 1 | 0.6042 | 0.6087 |
| controlled_alt_vs_baseline_no_context | sentence_score | 8 | 4 | 12 | 0.5833 | 0.6667 |
| controlled_alt_vs_baseline_no_context | overall_quality | 24 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 24 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 21 | 1 | 2 | 0.9167 | 0.9545 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 13 | 11 | 0 | 0.5417 | 0.5417 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 23 | 0 | 1 | 0.9792 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 23 | 1 | 0 | 0.9583 | 0.9583 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 21 | 1 | 2 | 0.9167 | 0.9545 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 4 | 2 | 18 | 0.5417 | 0.6667 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | 4 | 20 | 0 | 0.1667 | 0.1667 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 13 | 11 | 0 | 0.5417 | 0.5417 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 7 | 1 | 16 | 0.6250 | 0.8750 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 24 | 0 | 0 | 1.0000 | 1.0000 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.3333 | 0.2917 | 0.7083 |
| proposed_contextual_controlled_alt | 0.0000 | 0.0000 | 0.2917 | 0.2917 | 0.7083 |
| proposed_contextual | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.5833 | 0.0000 | 0.0000 |
| baseline_no_context | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `8`
- Unique template signatures: `22`
- Template signature ratio: `0.9167`
- Effective sample size by source clustering: `7.20`
- Effective sample size by template-signature clustering: `20.57`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: unavailable (disabled_by_flag).

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
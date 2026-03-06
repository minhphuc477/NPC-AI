# Proposal Alignment Evaluation Report

- Run ID: `20260306T162110Z`
- Generated: `2026-03-06T16:24:45.006084+00:00`
- Scenarios: `artifacts\proposal_control_tuning\architecture_compare_fast\20260306T162109Z\runtime_optimized\seed_29\attempt_1\20260306T162110Z\scenarios.jsonl`
- Scenario count: `20`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_alt`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2658 (0.2294, 0.2988) | 0.2837 (0.2354, 0.3323) | 0.9084 (0.8869, 0.9300) | 0.4011 (0.3769, 0.4251) | n/a |
| proposed_contextual_controlled_alt | 0.2832 (0.2580, 0.3103) | 0.2630 (0.2198, 0.3045) | 0.9064 (0.8888, 0.9240) | 0.4008 (0.3784, 0.4231) | n/a |
| proposed_contextual | 0.0754 (0.0339, 0.1250) | 0.1163 (0.0851, 0.1486) | 0.8189 (0.7841, 0.8554) | 0.2331 (0.2037, 0.2673) | n/a |
| candidate_no_context | 0.0192 (0.0119, 0.0326) | 0.1507 (0.1040, 0.2026) | 0.8036 (0.7697, 0.8412) | 0.2168 (0.1944, 0.2423) | n/a |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0562 | 2.9184 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0343 | -0.2279 |
| proposed_vs_candidate_no_context | naturalness | 0.0153 | 0.0191 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0761 | 9.1364 |
| proposed_vs_candidate_no_context | context_overlap | 0.0095 | 0.2132 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0464 | -0.8478 |
| proposed_vs_candidate_no_context | persona_style | 0.0140 | 0.0263 |
| proposed_vs_candidate_no_context | distinct1 | 0.0065 | 0.0069 |
| proposed_vs_candidate_no_context | length_score | 0.0550 | 0.2012 |
| proposed_vs_candidate_no_context | sentence_score | 0.0175 | 0.0232 |
| proposed_vs_candidate_no_context | overall_quality | 0.0163 | 0.0751 |
| controlled_vs_proposed_raw | context_relevance | 0.1904 | 2.5257 |
| controlled_vs_proposed_raw | persona_consistency | 0.1673 | 1.4381 |
| controlled_vs_proposed_raw | naturalness | 0.0895 | 0.1093 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2498 | 2.9578 |
| controlled_vs_proposed_raw | context_overlap | 0.0518 | 0.9550 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1900 | 22.8000 |
| controlled_vs_proposed_raw | persona_style | 0.0766 | 0.1397 |
| controlled_vs_proposed_raw | distinct1 | 0.0024 | 0.0026 |
| controlled_vs_proposed_raw | length_score | 0.3550 | 1.0812 |
| controlled_vs_proposed_raw | sentence_score | 0.1750 | 0.2265 |
| controlled_vs_proposed_raw | overall_quality | 0.1680 | 0.7209 |
| controlled_vs_candidate_no_context | context_relevance | 0.2466 | 12.8152 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1330 | 0.8825 |
| controlled_vs_candidate_no_context | naturalness | 0.1048 | 0.1304 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3260 | 39.1182 |
| controlled_vs_candidate_no_context | context_overlap | 0.0613 | 1.3717 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1436 | 2.6217 |
| controlled_vs_candidate_no_context | persona_style | 0.0906 | 0.1696 |
| controlled_vs_candidate_no_context | distinct1 | 0.0089 | 0.0095 |
| controlled_vs_candidate_no_context | length_score | 0.4100 | 1.5000 |
| controlled_vs_candidate_no_context | sentence_score | 0.1925 | 0.2550 |
| controlled_vs_candidate_no_context | overall_quality | 0.1843 | 0.8502 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0173 | 0.0653 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0207 | -0.0730 |
| controlled_alt_vs_controlled_default | naturalness | -0.0020 | -0.0022 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0182 | 0.0544 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0154 | 0.1453 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0238 | -0.1200 |
| controlled_alt_vs_controlled_default | persona_style | -0.0083 | -0.0133 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0066 | -0.0070 |
| controlled_alt_vs_controlled_default | length_score | 0.0383 | 0.0561 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0700 | -0.0739 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0003 | -0.0008 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.2078 | 2.7558 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1466 | 1.2600 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0875 | 0.1068 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2680 | 3.1731 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0672 | 1.2391 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1662 | 19.9429 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0683 | 0.1245 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0042 | -0.0045 |
| controlled_alt_vs_proposed_raw | length_score | 0.3933 | 1.1980 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1050 | 0.1359 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1677 | 0.7194 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2639 | 13.7168 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1123 | 0.7451 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.1028 | 0.1280 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3442 | 41.3000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0767 | 1.7165 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1198 | 2.1870 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0823 | 0.1540 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0023 | 0.0024 |
| controlled_alt_vs_candidate_no_context | length_score | 0.4483 | 1.6402 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1225 | 0.1623 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1840 | 0.8486 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0562 | (0.0194, 0.1063) | 0.0000 | 0.0562 | (0.0205, 0.1057) | 0.0000 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0343 | (-0.0792, 0.0023) | 0.9640 | -0.0343 | (-0.0868, 0.0000) | 0.9750 |
| proposed_vs_candidate_no_context | naturalness | 0.0153 | (-0.0238, 0.0523) | 0.2157 | 0.0153 | (-0.0277, 0.0430) | 0.2010 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0761 | (0.0265, 0.1394) | 0.0000 | 0.0761 | (0.0273, 0.1452) | 0.0000 |
| proposed_vs_candidate_no_context | context_overlap | 0.0095 | (-0.0031, 0.0227) | 0.0730 | 0.0095 | (-0.0017, 0.0211) | 0.0477 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0464 | (-0.1024, 0.0000) | 1.0000 | -0.0464 | (-0.1124, -0.0064) | 1.0000 |
| proposed_vs_candidate_no_context | persona_style | 0.0140 | (-0.0079, 0.0500) | 0.3683 | 0.0140 | (-0.0079, 0.0557) | 0.3327 |
| proposed_vs_candidate_no_context | distinct1 | 0.0065 | (-0.0081, 0.0214) | 0.1960 | 0.0065 | (-0.0165, 0.0278) | 0.2837 |
| proposed_vs_candidate_no_context | length_score | 0.0550 | (-0.1134, 0.2117) | 0.2360 | 0.0550 | (-0.0870, 0.1637) | 0.1920 |
| proposed_vs_candidate_no_context | sentence_score | 0.0175 | (-0.0525, 0.0875) | 0.4003 | 0.0175 | (-0.0467, 0.0667) | 0.3743 |
| proposed_vs_candidate_no_context | overall_quality | 0.0163 | (-0.0139, 0.0483) | 0.1410 | 0.0163 | (-0.0151, 0.0392) | 0.1277 |
| controlled_vs_proposed_raw | context_relevance | 0.1904 | (0.1415, 0.2350) | 0.0000 | 0.1904 | (0.1494, 0.2244) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.1673 | (0.1213, 0.2164) | 0.0000 | 0.1673 | (0.1158, 0.2221) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0895 | (0.0436, 0.1355) | 0.0000 | 0.0895 | (0.0205, 0.1442) | 0.0077 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2498 | (0.1841, 0.3097) | 0.0000 | 0.2498 | (0.1937, 0.2943) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0518 | (0.0348, 0.0682) | 0.0000 | 0.0518 | (0.0391, 0.0661) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1900 | (0.1340, 0.2486) | 0.0000 | 0.1900 | (0.1295, 0.2689) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.0766 | (0.0026, 0.1750) | 0.0153 | 0.0766 | (0.0020, 0.2079) | 0.0217 |
| controlled_vs_proposed_raw | distinct1 | 0.0024 | (-0.0164, 0.0215) | 0.3980 | 0.0024 | (-0.0219, 0.0234) | 0.4087 |
| controlled_vs_proposed_raw | length_score | 0.3550 | (0.1550, 0.5450) | 0.0010 | 0.3550 | (0.0823, 0.5614) | 0.0083 |
| controlled_vs_proposed_raw | sentence_score | 0.1750 | (0.1050, 0.2450) | 0.0000 | 0.1750 | (0.0656, 0.2625) | 0.0003 |
| controlled_vs_proposed_raw | overall_quality | 0.1680 | (0.1397, 0.1979) | 0.0000 | 0.1680 | (0.1432, 0.1964) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2466 | (0.2125, 0.2788) | 0.0000 | 0.2466 | (0.2028, 0.2820) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1330 | (0.0736, 0.1926) | 0.0000 | 0.1330 | (0.0919, 0.1823) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.1048 | (0.0644, 0.1417) | 0.0000 | 0.1048 | (0.0363, 0.1495) | 0.0027 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3260 | (0.2789, 0.3667) | 0.0000 | 0.3260 | (0.2710, 0.3743) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0613 | (0.0484, 0.0739) | 0.0000 | 0.0613 | (0.0472, 0.0761) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1436 | (0.0750, 0.2121) | 0.0000 | 0.1436 | (0.0929, 0.2087) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.0906 | (0.0083, 0.1891) | 0.0150 | 0.0906 | (0.0000, 0.2336) | 0.0943 |
| controlled_vs_candidate_no_context | distinct1 | 0.0089 | (-0.0093, 0.0264) | 0.1583 | 0.0089 | (-0.0167, 0.0254) | 0.2260 |
| controlled_vs_candidate_no_context | length_score | 0.4100 | (0.2450, 0.5550) | 0.0000 | 0.4100 | (0.1684, 0.5747) | 0.0003 |
| controlled_vs_candidate_no_context | sentence_score | 0.1925 | (0.1050, 0.2800) | 0.0000 | 0.1925 | (0.0467, 0.2982) | 0.0037 |
| controlled_vs_candidate_no_context | overall_quality | 0.1843 | (0.1501, 0.2159) | 0.0000 | 0.1843 | (0.1497, 0.2135) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0173 | (-0.0207, 0.0582) | 0.1847 | 0.0173 | (-0.0228, 0.0558) | 0.2090 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0207 | (-0.0584, 0.0107) | 0.8860 | -0.0207 | (-0.0526, -0.0013) | 1.0000 |
| controlled_alt_vs_controlled_default | naturalness | -0.0020 | (-0.0221, 0.0180) | 0.5693 | -0.0020 | (-0.0185, 0.0080) | 0.6133 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0182 | (-0.0273, 0.0682) | 0.2330 | 0.0182 | (-0.0321, 0.0699) | 0.2900 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0154 | (-0.0018, 0.0350) | 0.0393 | 0.0154 | (-0.0001, 0.0303) | 0.0263 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0238 | (-0.0705, 0.0157) | 0.8767 | -0.0238 | (-0.0635, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | persona_style | -0.0083 | (-0.0250, 0.0000) | 1.0000 | -0.0083 | (-0.0294, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0066 | (-0.0236, 0.0109) | 0.7827 | -0.0066 | (-0.0257, 0.0129) | 0.7690 |
| controlled_alt_vs_controlled_default | length_score | 0.0383 | (-0.0367, 0.1217) | 0.1723 | 0.0383 | (-0.0333, 0.0867) | 0.1283 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0700 | (-0.1575, 0.0175) | 0.9550 | -0.0700 | (-0.1235, -0.0194) | 1.0000 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0003 | (-0.0265, 0.0238) | 0.5063 | -0.0003 | (-0.0260, 0.0204) | 0.4900 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.2078 | (0.1439, 0.2625) | 0.0000 | 0.2078 | (0.1483, 0.2580) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1466 | (0.1076, 0.1839) | 0.0000 | 0.1466 | (0.1060, 0.1933) | 0.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0875 | (0.0364, 0.1337) | 0.0003 | 0.0875 | (0.0122, 0.1390) | 0.0113 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2680 | (0.1862, 0.3408) | 0.0000 | 0.2680 | (0.1897, 0.3318) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0672 | (0.0474, 0.0871) | 0.0000 | 0.0672 | (0.0545, 0.0787) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1662 | (0.1217, 0.2091) | 0.0000 | 0.1662 | (0.1185, 0.2330) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0683 | (-0.0057, 0.1724) | 0.0553 | 0.0683 | (0.0000, 0.2073) | 0.1013 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0042 | (-0.0258, 0.0162) | 0.6387 | -0.0042 | (-0.0389, 0.0222) | 0.5923 |
| controlled_alt_vs_proposed_raw | length_score | 0.3933 | (0.2000, 0.5717) | 0.0000 | 0.3933 | (0.1244, 0.5767) | 0.0037 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1050 | (-0.0175, 0.2275) | 0.0617 | 0.1050 | (0.0000, 0.1919) | 0.0427 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1677 | (0.1261, 0.2055) | 0.0000 | 0.1677 | (0.1323, 0.2014) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2639 | (0.2351, 0.2948) | 0.0000 | 0.2639 | (0.2448, 0.2808) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1123 | (0.0582, 0.1652) | 0.0000 | 0.1123 | (0.0639, 0.1688) | 0.0000 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.1028 | (0.0523, 0.1473) | 0.0003 | 0.1028 | (0.0286, 0.1529) | 0.0060 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3442 | (0.3057, 0.3832) | 0.0000 | 0.3442 | (0.3178, 0.3636) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0767 | (0.0605, 0.0958) | 0.0000 | 0.0767 | (0.0661, 0.0863) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1198 | (0.0533, 0.1795) | 0.0007 | 0.1198 | (0.0640, 0.1898) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0823 | (0.0083, 0.1731) | 0.0107 | 0.0823 | (0.0000, 0.2248) | 0.0980 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0023 | (-0.0214, 0.0248) | 0.4353 | 0.0023 | (-0.0326, 0.0272) | 0.4227 |
| controlled_alt_vs_candidate_no_context | length_score | 0.4483 | (0.2700, 0.6067) | 0.0000 | 0.4483 | (0.1833, 0.6133) | 0.0003 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1225 | (0.0175, 0.2275) | 0.0247 | 0.1225 | (0.0000, 0.2188) | 0.0437 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1840 | (0.1500, 0.2153) | 0.0000 | 0.1840 | (0.1523, 0.2110) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 8 | 3 | 9 | 0.6250 | 0.7273 |
| proposed_vs_candidate_no_context | persona_consistency | 1 | 4 | 15 | 0.4250 | 0.2000 |
| proposed_vs_candidate_no_context | naturalness | 7 | 4 | 9 | 0.5750 | 0.6364 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 8 | 0 | 12 | 0.7000 | 1.0000 |
| proposed_vs_candidate_no_context | context_overlap | 8 | 3 | 9 | 0.6250 | 0.7273 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0 | 3 | 17 | 0.4250 | 0.0000 |
| proposed_vs_candidate_no_context | persona_style | 1 | 1 | 18 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | distinct1 | 7 | 3 | 10 | 0.6000 | 0.7000 |
| proposed_vs_candidate_no_context | length_score | 6 | 5 | 9 | 0.5250 | 0.5455 |
| proposed_vs_candidate_no_context | sentence_score | 3 | 2 | 15 | 0.5250 | 0.6000 |
| proposed_vs_candidate_no_context | overall_quality | 8 | 3 | 9 | 0.6250 | 0.7273 |
| controlled_vs_proposed_raw | context_relevance | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_vs_proposed_raw | persona_consistency | 16 | 0 | 4 | 0.9000 | 1.0000 |
| controlled_vs_proposed_raw | naturalness | 15 | 5 | 0 | 0.7500 | 0.7500 |
| controlled_vs_proposed_raw | context_keyword_coverage | 18 | 1 | 1 | 0.9250 | 0.9474 |
| controlled_vs_proposed_raw | context_overlap | 18 | 2 | 0 | 0.9000 | 0.9000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 16 | 0 | 4 | 0.9000 | 1.0000 |
| controlled_vs_proposed_raw | persona_style | 4 | 0 | 16 | 0.6000 | 1.0000 |
| controlled_vs_proposed_raw | distinct1 | 10 | 10 | 0 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | length_score | 15 | 4 | 1 | 0.7750 | 0.7895 |
| controlled_vs_proposed_raw | sentence_score | 10 | 0 | 10 | 0.7500 | 1.0000 |
| controlled_vs_proposed_raw | overall_quality | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_relevance | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_vs_candidate_no_context | persona_consistency | 14 | 2 | 4 | 0.8000 | 0.8750 |
| controlled_vs_candidate_no_context | naturalness | 17 | 3 | 0 | 0.8500 | 0.8500 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 19 | 0 | 1 | 0.9750 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 14 | 2 | 4 | 0.8000 | 0.8750 |
| controlled_vs_candidate_no_context | persona_style | 4 | 0 | 16 | 0.6000 | 1.0000 |
| controlled_vs_candidate_no_context | distinct1 | 12 | 8 | 0 | 0.6000 | 0.6000 |
| controlled_vs_candidate_no_context | length_score | 17 | 3 | 0 | 0.8500 | 0.8500 |
| controlled_vs_candidate_no_context | sentence_score | 12 | 1 | 7 | 0.7750 | 0.9231 |
| controlled_vs_candidate_no_context | overall_quality | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 9 | 5 | 6 | 0.6000 | 0.6429 |
| controlled_alt_vs_controlled_default | persona_consistency | 1 | 4 | 15 | 0.4250 | 0.2000 |
| controlled_alt_vs_controlled_default | naturalness | 7 | 7 | 6 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 5 | 4 | 11 | 0.5250 | 0.5556 |
| controlled_alt_vs_controlled_default | context_overlap | 9 | 5 | 6 | 0.6000 | 0.6429 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 1 | 3 | 16 | 0.4500 | 0.2500 |
| controlled_alt_vs_controlled_default | persona_style | 0 | 1 | 19 | 0.4750 | 0.0000 |
| controlled_alt_vs_controlled_default | distinct1 | 6 | 8 | 6 | 0.4500 | 0.4286 |
| controlled_alt_vs_controlled_default | length_score | 9 | 5 | 6 | 0.6000 | 0.6429 |
| controlled_alt_vs_controlled_default | sentence_score | 2 | 6 | 12 | 0.4000 | 0.2500 |
| controlled_alt_vs_controlled_default | overall_quality | 7 | 7 | 6 | 0.5000 | 0.5000 |
| controlled_alt_vs_proposed_raw | context_relevance | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_alt_vs_proposed_raw | persona_consistency | 16 | 0 | 4 | 0.9000 | 1.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 13 | 7 | 0 | 0.6500 | 0.6500 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 18 | 1 | 1 | 0.9250 | 0.9474 |
| controlled_alt_vs_proposed_raw | context_overlap | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 16 | 0 | 4 | 0.9000 | 1.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 4 | 1 | 15 | 0.5750 | 0.8000 |
| controlled_alt_vs_proposed_raw | distinct1 | 11 | 8 | 1 | 0.5750 | 0.5789 |
| controlled_alt_vs_proposed_raw | length_score | 15 | 4 | 1 | 0.7750 | 0.7895 |
| controlled_alt_vs_proposed_raw | sentence_score | 10 | 4 | 6 | 0.6500 | 0.7143 |
| controlled_alt_vs_proposed_raw | overall_quality | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_alt_vs_candidate_no_context | context_relevance | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 14 | 3 | 3 | 0.7750 | 0.8235 |
| controlled_alt_vs_candidate_no_context | naturalness | 15 | 5 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 14 | 3 | 3 | 0.7750 | 0.8235 |
| controlled_alt_vs_candidate_no_context | persona_style | 4 | 0 | 16 | 0.6000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | distinct1 | 12 | 7 | 1 | 0.6250 | 0.6316 |
| controlled_alt_vs_candidate_no_context | length_score | 17 | 3 | 0 | 0.8500 | 0.8500 |
| controlled_alt_vs_candidate_no_context | sentence_score | 10 | 3 | 7 | 0.6750 | 0.7692 |
| controlled_alt_vs_candidate_no_context | overall_quality | 20 | 0 | 0 | 1.0000 | 1.0000 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.0000 | 0.5500 | 0.2000 |
| proposed_contextual_controlled_alt | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.2500 |
| proposed_contextual | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.6500 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `8`
- Unique template signatures: `18`
- Template signature ratio: `0.9000`
- Effective sample size by source clustering: `6.67`
- Effective sample size by template-signature clustering: `16.67`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: unavailable (disabled_by_flag).

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
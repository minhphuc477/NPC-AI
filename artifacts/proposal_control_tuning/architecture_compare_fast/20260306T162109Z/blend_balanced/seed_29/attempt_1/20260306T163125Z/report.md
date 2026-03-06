# Proposal Alignment Evaluation Report

- Run ID: `20260306T163125Z`
- Generated: `2026-03-06T16:34:51.865344+00:00`
- Scenarios: `artifacts\proposal_control_tuning\architecture_compare_fast\20260306T162109Z\blend_balanced\seed_29\attempt_1\20260306T163125Z\scenarios.jsonl`
- Scenario count: `20`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_alt`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2661 (0.2361, 0.2974) | 0.2468 (0.2013, 0.2918) | 0.9133 (0.8925, 0.9336) | 0.3885 (0.3615, 0.4153) | n/a |
| proposed_contextual_controlled_alt | 0.2622 (0.2242, 0.3053) | 0.2745 (0.2255, 0.3209) | 0.9058 (0.8839, 0.9274) | 0.3956 (0.3698, 0.4220) | n/a |
| proposed_contextual | 0.0744 (0.0336, 0.1278) | 0.1360 (0.0944, 0.1896) | 0.8105 (0.7772, 0.8455) | 0.2388 (0.2076, 0.2752) | n/a |
| candidate_no_context | 0.0248 (0.0122, 0.0404) | 0.1922 (0.1265, 0.2665) | 0.8245 (0.7902, 0.8590) | 0.2389 (0.2080, 0.2738) | n/a |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0497 | 2.0060 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0562 | -0.2924 |
| proposed_vs_candidate_no_context | naturalness | -0.0140 | -0.0170 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0625 | 3.4375 |
| proposed_vs_candidate_no_context | context_overlap | 0.0197 | 0.4915 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0560 | -0.5595 |
| proposed_vs_candidate_no_context | persona_style | -0.0571 | -0.1019 |
| proposed_vs_candidate_no_context | distinct1 | -0.0087 | -0.0093 |
| proposed_vs_candidate_no_context | length_score | -0.0350 | -0.1010 |
| proposed_vs_candidate_no_context | sentence_score | -0.0350 | -0.0443 |
| proposed_vs_candidate_no_context | overall_quality | -0.0001 | -0.0006 |
| controlled_vs_proposed_raw | context_relevance | 0.1917 | 2.5760 |
| controlled_vs_proposed_raw | persona_consistency | 0.1108 | 0.8144 |
| controlled_vs_proposed_raw | naturalness | 0.1028 | 0.1269 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2529 | 3.1343 |
| controlled_vs_proposed_raw | context_overlap | 0.0490 | 0.8187 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1105 | 2.5081 |
| controlled_vs_proposed_raw | persona_style | 0.1119 | 0.2221 |
| controlled_vs_proposed_raw | distinct1 | 0.0021 | 0.0022 |
| controlled_vs_proposed_raw | length_score | 0.4050 | 1.2995 |
| controlled_vs_proposed_raw | sentence_score | 0.2100 | 0.2781 |
| controlled_vs_proposed_raw | overall_quality | 0.1497 | 0.6268 |
| controlled_vs_candidate_no_context | context_relevance | 0.2414 | 9.7493 |
| controlled_vs_candidate_no_context | persona_consistency | 0.0546 | 0.2839 |
| controlled_vs_candidate_no_context | naturalness | 0.0888 | 0.1078 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3154 | 17.3458 |
| controlled_vs_candidate_no_context | context_overlap | 0.0687 | 1.7126 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.0545 | 0.5452 |
| controlled_vs_candidate_no_context | persona_style | 0.0547 | 0.0976 |
| controlled_vs_candidate_no_context | distinct1 | -0.0066 | -0.0071 |
| controlled_vs_candidate_no_context | length_score | 0.3700 | 1.0673 |
| controlled_vs_candidate_no_context | sentence_score | 0.1750 | 0.2215 |
| controlled_vs_candidate_no_context | overall_quality | 0.1495 | 0.6259 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0039 | -0.0148 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0277 | 0.1123 |
| controlled_alt_vs_controlled_default | naturalness | -0.0075 | -0.0082 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0045 | -0.0136 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0025 | -0.0230 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0355 | 0.2296 |
| controlled_alt_vs_controlled_default | persona_style | -0.0034 | -0.0055 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0021 | 0.0022 |
| controlled_alt_vs_controlled_default | length_score | -0.0317 | -0.0442 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0175 | -0.0181 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0072 | 0.0184 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1878 | 2.5231 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1385 | 1.0181 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0953 | 0.1176 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2483 | 3.0779 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0465 | 0.7769 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1460 | 3.3135 |
| controlled_alt_vs_proposed_raw | persona_style | 0.1085 | 0.2153 |
| controlled_alt_vs_proposed_raw | distinct1 | 0.0042 | 0.0045 |
| controlled_alt_vs_proposed_raw | length_score | 0.3733 | 1.1979 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1925 | 0.2550 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1568 | 0.6568 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2374 | 9.5905 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.0823 | 0.4281 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0813 | 0.0987 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3108 | 17.0958 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0662 | 1.6502 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.0900 | 0.9000 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0513 | 0.0915 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0045 | -0.0048 |
| controlled_alt_vs_candidate_no_context | length_score | 0.3383 | 0.9760 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1575 | 0.1994 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1567 | 0.6558 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0497 | (0.0084, 0.1059) | 0.0057 | 0.0497 | (0.0012, 0.1144) | 0.0207 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0562 | (-0.1240, 0.0015) | 0.9690 | -0.0562 | (-0.1333, 0.0044) | 0.9630 |
| proposed_vs_candidate_no_context | naturalness | -0.0140 | (-0.0411, 0.0121) | 0.8450 | -0.0140 | (-0.0457, 0.0109) | 0.8413 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0625 | (0.0083, 0.1307) | 0.0117 | 0.0625 | (-0.0010, 0.1490) | 0.0330 |
| proposed_vs_candidate_no_context | context_overlap | 0.0197 | (0.0062, 0.0349) | 0.0003 | 0.0197 | (0.0054, 0.0364) | 0.0010 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0560 | (-0.1226, 0.0048) | 0.9697 | -0.0560 | (-0.1317, 0.0048) | 0.9727 |
| proposed_vs_candidate_no_context | persona_style | -0.0571 | (-0.1672, 0.0376) | 0.8747 | -0.0571 | (-0.2010, 0.0264) | 0.7027 |
| proposed_vs_candidate_no_context | distinct1 | -0.0087 | (-0.0257, 0.0086) | 0.8403 | -0.0087 | (-0.0283, 0.0093) | 0.8283 |
| proposed_vs_candidate_no_context | length_score | -0.0350 | (-0.1267, 0.0533) | 0.7693 | -0.0350 | (-0.1439, 0.0550) | 0.7333 |
| proposed_vs_candidate_no_context | sentence_score | -0.0350 | (-0.1050, 0.0350) | 0.9047 | -0.0350 | (-0.1000, 0.0368) | 0.9057 |
| proposed_vs_candidate_no_context | overall_quality | -0.0001 | (-0.0302, 0.0285) | 0.4947 | -0.0001 | (-0.0431, 0.0373) | 0.4883 |
| controlled_vs_proposed_raw | context_relevance | 0.1917 | (0.1420, 0.2320) | 0.0000 | 0.1917 | (0.1513, 0.2267) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.1108 | (0.0582, 0.1618) | 0.0000 | 0.1108 | (0.0394, 0.1869) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.1028 | (0.0579, 0.1464) | 0.0000 | 0.1028 | (0.0258, 0.1584) | 0.0037 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2529 | (0.1879, 0.3074) | 0.0000 | 0.2529 | (0.1925, 0.3008) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0490 | (0.0305, 0.0670) | 0.0000 | 0.0490 | (0.0380, 0.0623) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1105 | (0.0583, 0.1619) | 0.0000 | 0.1105 | (0.0484, 0.1773) | 0.0003 |
| controlled_vs_proposed_raw | persona_style | 0.1119 | (0.0058, 0.2366) | 0.0183 | 0.1119 | (-0.0213, 0.3145) | 0.0943 |
| controlled_vs_proposed_raw | distinct1 | 0.0021 | (-0.0132, 0.0165) | 0.3650 | 0.0021 | (-0.0123, 0.0159) | 0.4123 |
| controlled_vs_proposed_raw | length_score | 0.4050 | (0.2250, 0.5750) | 0.0000 | 0.4050 | (0.1136, 0.6158) | 0.0040 |
| controlled_vs_proposed_raw | sentence_score | 0.2100 | (0.1225, 0.2975) | 0.0000 | 0.2100 | (0.0808, 0.3023) | 0.0003 |
| controlled_vs_proposed_raw | overall_quality | 0.1497 | (0.1135, 0.1825) | 0.0000 | 0.1497 | (0.1128, 0.1926) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2414 | (0.2104, 0.2763) | 0.0000 | 0.2414 | (0.2067, 0.2855) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.0546 | (-0.0193, 0.1220) | 0.0667 | 0.0546 | (-0.0212, 0.1199) | 0.0667 |
| controlled_vs_candidate_no_context | naturalness | 0.0888 | (0.0408, 0.1348) | 0.0003 | 0.0888 | (0.0196, 0.1376) | 0.0080 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3154 | (0.2718, 0.3586) | 0.0000 | 0.3154 | (0.2682, 0.3704) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0687 | (0.0553, 0.0839) | 0.0000 | 0.0687 | (0.0539, 0.0904) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.0545 | (-0.0224, 0.1267) | 0.0813 | 0.0545 | (-0.0262, 0.1280) | 0.0800 |
| controlled_vs_candidate_no_context | persona_style | 0.0547 | (-0.0156, 0.1334) | 0.0727 | 0.0547 | (-0.0164, 0.1416) | 0.0817 |
| controlled_vs_candidate_no_context | distinct1 | -0.0066 | (-0.0266, 0.0115) | 0.7543 | -0.0066 | (-0.0270, 0.0090) | 0.7947 |
| controlled_vs_candidate_no_context | length_score | 0.3700 | (0.1983, 0.5250) | 0.0000 | 0.3700 | (0.1244, 0.5367) | 0.0013 |
| controlled_vs_candidate_no_context | sentence_score | 0.1750 | (0.0700, 0.2800) | 0.0020 | 0.1750 | (0.0233, 0.2827) | 0.0137 |
| controlled_vs_candidate_no_context | overall_quality | 0.1495 | (0.1157, 0.1814) | 0.0000 | 0.1495 | (0.1152, 0.1774) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0039 | (-0.0462, 0.0374) | 0.5617 | -0.0039 | (-0.0179, 0.0205) | 0.6600 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0277 | (-0.0067, 0.0699) | 0.0653 | 0.0277 | (-0.0100, 0.0660) | 0.0787 |
| controlled_alt_vs_controlled_default | naturalness | -0.0075 | (-0.0319, 0.0158) | 0.7243 | -0.0075 | (-0.0315, 0.0175) | 0.7280 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0045 | (-0.0591, 0.0538) | 0.5633 | -0.0045 | (-0.0245, 0.0284) | 0.7067 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0025 | (-0.0163, 0.0116) | 0.6657 | -0.0025 | (-0.0185, 0.0087) | 0.6530 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0355 | (-0.0083, 0.0876) | 0.0800 | 0.0355 | (-0.0106, 0.0843) | 0.0867 |
| controlled_alt_vs_controlled_default | persona_style | -0.0034 | (-0.0294, 0.0236) | 0.5993 | -0.0034 | (-0.0276, 0.0152) | 0.6483 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0021 | (-0.0140, 0.0187) | 0.4063 | 0.0021 | (-0.0029, 0.0082) | 0.2043 |
| controlled_alt_vs_controlled_default | length_score | -0.0317 | (-0.1467, 0.0767) | 0.7033 | -0.0317 | (-0.1333, 0.0870) | 0.7087 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0175 | (-0.0700, 0.0350) | 0.8177 | -0.0175 | (-0.0618, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0072 | (-0.0156, 0.0287) | 0.2533 | 0.0072 | (-0.0064, 0.0234) | 0.1580 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1878 | (0.1296, 0.2365) | 0.0000 | 0.1878 | (0.1427, 0.2308) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1385 | (0.0903, 0.1876) | 0.0000 | 0.1385 | (0.0892, 0.2009) | 0.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0953 | (0.0561, 0.1339) | 0.0000 | 0.0953 | (0.0380, 0.1357) | 0.0003 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2483 | (0.1697, 0.3146) | 0.0000 | 0.2483 | (0.1837, 0.3085) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0465 | (0.0291, 0.0613) | 0.0000 | 0.0465 | (0.0329, 0.0617) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1460 | (0.0905, 0.1995) | 0.0000 | 0.1460 | (0.1071, 0.1902) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 0.1085 | (0.0105, 0.2278) | 0.0110 | 0.1085 | (-0.0198, 0.3179) | 0.1033 |
| controlled_alt_vs_proposed_raw | distinct1 | 0.0042 | (-0.0089, 0.0160) | 0.2547 | 0.0042 | (-0.0077, 0.0140) | 0.2490 |
| controlled_alt_vs_proposed_raw | length_score | 0.3733 | (0.2083, 0.5267) | 0.0000 | 0.3733 | (0.1611, 0.5188) | 0.0003 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1925 | (0.1225, 0.2625) | 0.0000 | 0.1925 | (0.0700, 0.2864) | 0.0000 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1568 | (0.1245, 0.1875) | 0.0000 | 0.1568 | (0.1257, 0.1942) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2374 | (0.1933, 0.2818) | 0.0000 | 0.2374 | (0.2030, 0.2795) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.0823 | (0.0158, 0.1486) | 0.0073 | 0.0823 | (0.0235, 0.1381) | 0.0070 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0813 | (0.0385, 0.1248) | 0.0000 | 0.0813 | (0.0267, 0.1198) | 0.0023 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3108 | (0.2500, 0.3733) | 0.0000 | 0.3108 | (0.2641, 0.3671) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0662 | (0.0544, 0.0773) | 0.0000 | 0.0662 | (0.0549, 0.0815) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.0900 | (0.0138, 0.1617) | 0.0107 | 0.0900 | (0.0219, 0.1572) | 0.0090 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0513 | (-0.0192, 0.1354) | 0.0963 | 0.0513 | (-0.0202, 0.1324) | 0.0697 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0045 | (-0.0253, 0.0151) | 0.6610 | -0.0045 | (-0.0252, 0.0149) | 0.6983 |
| controlled_alt_vs_candidate_no_context | length_score | 0.3383 | (0.1783, 0.4984) | 0.0000 | 0.3383 | (0.1604, 0.4600) | 0.0000 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1575 | (0.0525, 0.2625) | 0.0037 | 0.1575 | (0.0184, 0.2625) | 0.0207 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1567 | (0.1217, 0.1919) | 0.0000 | 0.1567 | (0.1257, 0.1854) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 9 | 2 | 9 | 0.6750 | 0.8182 |
| proposed_vs_candidate_no_context | persona_consistency | 4 | 6 | 10 | 0.4500 | 0.4000 |
| proposed_vs_candidate_no_context | naturalness | 4 | 7 | 9 | 0.4250 | 0.3636 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 7 | 2 | 11 | 0.6250 | 0.7778 |
| proposed_vs_candidate_no_context | context_overlap | 9 | 2 | 9 | 0.6750 | 0.8182 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 2 | 5 | 13 | 0.4250 | 0.2857 |
| proposed_vs_candidate_no_context | persona_style | 2 | 4 | 14 | 0.4500 | 0.3333 |
| proposed_vs_candidate_no_context | distinct1 | 3 | 8 | 9 | 0.3750 | 0.2727 |
| proposed_vs_candidate_no_context | length_score | 4 | 7 | 9 | 0.4250 | 0.3636 |
| proposed_vs_candidate_no_context | sentence_score | 1 | 3 | 16 | 0.4500 | 0.2500 |
| proposed_vs_candidate_no_context | overall_quality | 7 | 4 | 9 | 0.5750 | 0.6364 |
| controlled_vs_proposed_raw | context_relevance | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_vs_proposed_raw | persona_consistency | 12 | 2 | 6 | 0.7500 | 0.8571 |
| controlled_vs_proposed_raw | naturalness | 15 | 5 | 0 | 0.7500 | 0.7500 |
| controlled_vs_proposed_raw | context_keyword_coverage | 18 | 1 | 1 | 0.9250 | 0.9474 |
| controlled_vs_proposed_raw | context_overlap | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 12 | 1 | 7 | 0.7750 | 0.9231 |
| controlled_vs_proposed_raw | persona_style | 7 | 2 | 11 | 0.6250 | 0.7778 |
| controlled_vs_proposed_raw | distinct1 | 12 | 8 | 0 | 0.6000 | 0.6000 |
| controlled_vs_proposed_raw | length_score | 16 | 4 | 0 | 0.8000 | 0.8000 |
| controlled_vs_proposed_raw | sentence_score | 13 | 1 | 6 | 0.8000 | 0.9286 |
| controlled_vs_proposed_raw | overall_quality | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_vs_candidate_no_context | context_relevance | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 9 | 3 | 8 | 0.6500 | 0.7500 |
| controlled_vs_candidate_no_context | naturalness | 15 | 5 | 0 | 0.7500 | 0.7500 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 9 | 3 | 8 | 0.6500 | 0.7500 |
| controlled_vs_candidate_no_context | persona_style | 4 | 1 | 15 | 0.5750 | 0.8000 |
| controlled_vs_candidate_no_context | distinct1 | 10 | 10 | 0 | 0.5000 | 0.5000 |
| controlled_vs_candidate_no_context | length_score | 16 | 3 | 1 | 0.8250 | 0.8421 |
| controlled_vs_candidate_no_context | sentence_score | 12 | 2 | 6 | 0.7500 | 0.8571 |
| controlled_vs_candidate_no_context | overall_quality | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_alt_vs_controlled_default | context_relevance | 5 | 5 | 10 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | persona_consistency | 5 | 3 | 12 | 0.5500 | 0.6250 |
| controlled_alt_vs_controlled_default | naturalness | 5 | 5 | 10 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 4 | 4 | 12 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | context_overlap | 5 | 5 | 10 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 4 | 1 | 15 | 0.5750 | 0.8000 |
| controlled_alt_vs_controlled_default | persona_style | 1 | 2 | 17 | 0.4750 | 0.3333 |
| controlled_alt_vs_controlled_default | distinct1 | 5 | 5 | 10 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | length_score | 3 | 7 | 10 | 0.4000 | 0.3000 |
| controlled_alt_vs_controlled_default | sentence_score | 1 | 2 | 17 | 0.4750 | 0.3333 |
| controlled_alt_vs_controlled_default | overall_quality | 7 | 3 | 10 | 0.6000 | 0.7000 |
| controlled_alt_vs_proposed_raw | context_relevance | 18 | 2 | 0 | 0.9000 | 0.9000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 15 | 1 | 4 | 0.8500 | 0.9375 |
| controlled_alt_vs_proposed_raw | naturalness | 15 | 4 | 1 | 0.7750 | 0.7895 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 18 | 1 | 1 | 0.9250 | 0.9474 |
| controlled_alt_vs_proposed_raw | context_overlap | 18 | 2 | 0 | 0.9000 | 0.9000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 15 | 1 | 4 | 0.8500 | 0.9375 |
| controlled_alt_vs_proposed_raw | persona_style | 6 | 1 | 13 | 0.6250 | 0.8571 |
| controlled_alt_vs_proposed_raw | distinct1 | 12 | 6 | 2 | 0.6500 | 0.6667 |
| controlled_alt_vs_proposed_raw | length_score | 16 | 2 | 2 | 0.8500 | 0.8889 |
| controlled_alt_vs_proposed_raw | sentence_score | 11 | 0 | 9 | 0.7750 | 1.0000 |
| controlled_alt_vs_proposed_raw | overall_quality | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 11 | 3 | 6 | 0.7000 | 0.7857 |
| controlled_alt_vs_candidate_no_context | naturalness | 13 | 7 | 0 | 0.6500 | 0.6500 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 19 | 0 | 1 | 0.9750 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 11 | 3 | 6 | 0.7000 | 0.7857 |
| controlled_alt_vs_candidate_no_context | persona_style | 4 | 2 | 14 | 0.5500 | 0.6667 |
| controlled_alt_vs_candidate_no_context | distinct1 | 11 | 9 | 0 | 0.5500 | 0.5500 |
| controlled_alt_vs_candidate_no_context | length_score | 16 | 4 | 0 | 0.8000 | 0.8000 |
| controlled_alt_vs_candidate_no_context | sentence_score | 11 | 2 | 7 | 0.7250 | 0.8462 |
| controlled_alt_vs_candidate_no_context | overall_quality | 19 | 1 | 0 | 0.9500 | 0.9500 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.0000 | 0.3000 | 0.2500 |
| proposed_contextual_controlled_alt | 0.0000 | 0.0000 | 0.0000 | 0.4500 | 0.2500 |
| proposed_contextual | 0.0000 | 0.0000 | 0.5500 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `8`
- Unique template signatures: `18`
- Template signature ratio: `0.9000`
- Effective sample size by source clustering: `6.67`
- Effective sample size by template-signature clustering: `16.67`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: unavailable (disabled_by_flag).

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
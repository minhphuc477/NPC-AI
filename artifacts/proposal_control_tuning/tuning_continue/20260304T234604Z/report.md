# Proposal Alignment Evaluation Report

- Run ID: `20260304T234604Z`
- Generated: `2026-03-04T23:50:47.467928+00:00`
- Scenarios: `artifacts\proposal_control_tuning\tuning_continue\20260304T234604Z\scenarios.jsonl`
- Scenario count: `20`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off
- `baseline_no_context`: model `phi3:mini`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2952 (0.2573, 0.3409) | 0.3181 (0.2580, 0.3798) | 0.8892 (0.8586, 0.9185) | 0.3867 (0.3591, 0.4145) | 0.0850 |
| proposed_contextual | 0.0794 (0.0386, 0.1245) | 0.1746 (0.1019, 0.2618) | 0.7711 (0.7561, 0.7910) | 0.2294 (0.1896, 0.2724) | 0.0685 |
| candidate_no_context | 0.0385 (0.0208, 0.0603) | 0.1789 (0.1170, 0.2482) | 0.7970 (0.7679, 0.8294) | 0.2143 (0.1881, 0.2436) | 0.0308 |
| baseline_no_context | 0.0286 (0.0172, 0.0413) | 0.2091 (0.1534, 0.2689) | 0.8716 (0.8429, 0.9001) | 0.2366 (0.2163, 0.2599) | 0.0641 |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0409 | 1.0635 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0043 | -0.0241 |
| proposed_vs_candidate_no_context | naturalness | -0.0260 | -0.0326 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0538 | 1.5106 |
| proposed_vs_candidate_no_context | context_overlap | 0.0110 | 0.2423 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0033 | -0.0374 |
| proposed_vs_candidate_no_context | persona_style | -0.0082 | -0.0153 |
| proposed_vs_candidate_no_context | distinct1 | -0.0103 | -0.0111 |
| proposed_vs_candidate_no_context | length_score | -0.0917 | -0.3846 |
| proposed_vs_candidate_no_context | sentence_score | -0.0350 | -0.0453 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0376 | 1.2207 |
| proposed_vs_candidate_no_context | overall_quality | 0.0151 | 0.0704 |
| proposed_vs_baseline_no_context | context_relevance | 0.0509 | 1.7807 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0344 | -0.1647 |
| proposed_vs_baseline_no_context | naturalness | -0.1005 | -0.1153 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0667 | 2.9333 |
| proposed_vs_baseline_no_context | context_overlap | 0.0140 | 0.3320 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0183 | -0.1762 |
| proposed_vs_baseline_no_context | persona_style | -0.0988 | -0.1571 |
| proposed_vs_baseline_no_context | distinct1 | -0.0518 | -0.0533 |
| proposed_vs_baseline_no_context | length_score | -0.3117 | -0.6800 |
| proposed_vs_baseline_no_context | sentence_score | -0.1750 | -0.1918 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0044 | 0.0682 |
| proposed_vs_baseline_no_context | overall_quality | -0.0072 | -0.0304 |
| controlled_vs_proposed_raw | context_relevance | 0.2158 | 2.7166 |
| controlled_vs_proposed_raw | persona_consistency | 0.1435 | 0.8218 |
| controlled_vs_proposed_raw | naturalness | 0.1181 | 0.1531 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2858 | 3.1975 |
| controlled_vs_proposed_raw | context_overlap | 0.0524 | 0.9320 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1540 | 1.7972 |
| controlled_vs_proposed_raw | persona_style | 0.1013 | 0.1911 |
| controlled_vs_proposed_raw | distinct1 | 0.0200 | 0.0217 |
| controlled_vs_proposed_raw | length_score | 0.4367 | 2.9773 |
| controlled_vs_proposed_raw | sentence_score | 0.2275 | 0.3085 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0165 | 0.2416 |
| controlled_vs_proposed_raw | overall_quality | 0.1573 | 0.6856 |
| controlled_vs_candidate_no_context | context_relevance | 0.2567 | 6.6693 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1392 | 0.7779 |
| controlled_vs_candidate_no_context | naturalness | 0.0921 | 0.1156 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3396 | 9.5383 |
| controlled_vs_candidate_no_context | context_overlap | 0.0633 | 1.4001 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1507 | 1.6925 |
| controlled_vs_candidate_no_context | persona_style | 0.0931 | 0.1729 |
| controlled_vs_candidate_no_context | distinct1 | 0.0097 | 0.0104 |
| controlled_vs_candidate_no_context | length_score | 0.3450 | 1.4476 |
| controlled_vs_candidate_no_context | sentence_score | 0.1925 | 0.2492 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0542 | 1.7572 |
| controlled_vs_candidate_no_context | overall_quality | 0.1724 | 0.8043 |
| controlled_vs_baseline_no_context | context_relevance | 0.2667 | 9.3349 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1091 | 0.5217 |
| controlled_vs_baseline_no_context | naturalness | 0.0175 | 0.0201 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3525 | 15.5100 |
| controlled_vs_baseline_no_context | context_overlap | 0.0664 | 1.5733 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1357 | 1.3043 |
| controlled_vs_baseline_no_context | persona_style | 0.0025 | 0.0040 |
| controlled_vs_baseline_no_context | distinct1 | -0.0318 | -0.0327 |
| controlled_vs_baseline_no_context | length_score | 0.1250 | 0.2727 |
| controlled_vs_baseline_no_context | sentence_score | 0.0525 | 0.0575 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0209 | 0.3263 |
| controlled_vs_baseline_no_context | overall_quality | 0.1501 | 0.6343 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2667 | 9.3349 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1091 | 0.5217 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 0.0175 | 0.0201 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3525 | 15.5100 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0664 | 1.5733 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1357 | 1.3043 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0025 | 0.0040 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0318 | -0.0327 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.1250 | 0.2727 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0525 | 0.0575 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0209 | 0.3263 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1501 | 0.6343 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0409 | (0.0029, 0.0828) | 0.0180 | 0.0409 | (-0.0001, 0.0941) | 0.0253 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0043 | (-0.0653, 0.0565) | 0.5647 | -0.0043 | (-0.0561, 0.0738) | 0.5743 |
| proposed_vs_candidate_no_context | naturalness | -0.0260 | (-0.0633, 0.0083) | 0.9287 | -0.0260 | (-0.0550, 0.0033) | 0.9563 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0538 | (-0.0000, 0.1068) | 0.0253 | 0.0538 | (-0.0006, 0.1212) | 0.0290 |
| proposed_vs_candidate_no_context | context_overlap | 0.0110 | (-0.0006, 0.0249) | 0.0340 | 0.0110 | (0.0005, 0.0281) | 0.0187 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0033 | (-0.0779, 0.0738) | 0.5403 | -0.0033 | (-0.0671, 0.0952) | 0.5387 |
| proposed_vs_candidate_no_context | persona_style | -0.0082 | (-0.0381, 0.0135) | 0.7293 | -0.0082 | (-0.0290, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | distinct1 | -0.0103 | (-0.0263, 0.0048) | 0.9080 | -0.0103 | (-0.0256, 0.0023) | 0.9397 |
| proposed_vs_candidate_no_context | length_score | -0.0917 | (-0.2283, 0.0283) | 0.9247 | -0.0917 | (-0.1926, 0.0104) | 0.9557 |
| proposed_vs_candidate_no_context | sentence_score | -0.0350 | (-0.1225, 0.0525) | 0.8700 | -0.0350 | (-0.0972, 0.0319) | 0.9147 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0376 | (0.0154, 0.0625) | 0.0003 | 0.0376 | (0.0180, 0.0713) | 0.0000 |
| proposed_vs_candidate_no_context | overall_quality | 0.0151 | (-0.0190, 0.0506) | 0.1993 | 0.0151 | (-0.0196, 0.0630) | 0.2543 |
| proposed_vs_baseline_no_context | context_relevance | 0.0509 | (0.0101, 0.0927) | 0.0063 | 0.0509 | (0.0038, 0.1058) | 0.0107 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0344 | (-0.1064, 0.0583) | 0.7837 | -0.0344 | (-0.1209, 0.0936) | 0.7457 |
| proposed_vs_baseline_no_context | naturalness | -0.1005 | (-0.1326, -0.0676) | 1.0000 | -0.1005 | (-0.1431, -0.0573) | 1.0000 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0667 | (0.0136, 0.1220) | 0.0063 | 0.0667 | (0.0073, 0.1414) | 0.0160 |
| proposed_vs_baseline_no_context | context_overlap | 0.0140 | (0.0002, 0.0302) | 0.0247 | 0.0140 | (-0.0021, 0.0365) | 0.0453 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0183 | (-0.1067, 0.0950) | 0.6833 | -0.0183 | (-0.1282, 0.1404) | 0.6473 |
| proposed_vs_baseline_no_context | persona_style | -0.0988 | (-0.1979, -0.0166) | 1.0000 | -0.0988 | (-0.2316, -0.0127) | 1.0000 |
| proposed_vs_baseline_no_context | distinct1 | -0.0518 | (-0.0675, -0.0349) | 1.0000 | -0.0518 | (-0.0722, -0.0308) | 1.0000 |
| proposed_vs_baseline_no_context | length_score | -0.3117 | (-0.4517, -0.1717) | 1.0000 | -0.3117 | (-0.4822, -0.1437) | 1.0000 |
| proposed_vs_baseline_no_context | sentence_score | -0.1750 | (-0.2625, -0.0875) | 1.0000 | -0.1750 | (-0.2800, -0.0437) | 1.0000 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0044 | (-0.0132, 0.0223) | 0.3183 | 0.0044 | (-0.0161, 0.0256) | 0.3433 |
| proposed_vs_baseline_no_context | overall_quality | -0.0072 | (-0.0458, 0.0385) | 0.6483 | -0.0072 | (-0.0561, 0.0549) | 0.5940 |
| controlled_vs_proposed_raw | context_relevance | 0.2158 | (0.1542, 0.2792) | 0.0000 | 0.2158 | (0.1473, 0.2875) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.1435 | (0.0393, 0.2277) | 0.0050 | 0.1435 | (0.0509, 0.2306) | 0.0010 |
| controlled_vs_proposed_raw | naturalness | 0.1181 | (0.0842, 0.1509) | 0.0000 | 0.1181 | (0.0727, 0.1470) | 0.0000 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2858 | (0.2034, 0.3690) | 0.0000 | 0.2858 | (0.1950, 0.3828) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0524 | (0.0343, 0.0713) | 0.0000 | 0.0524 | (0.0319, 0.0698) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1540 | (0.0362, 0.2593) | 0.0070 | 0.1540 | (0.0429, 0.2625) | 0.0057 |
| controlled_vs_proposed_raw | persona_style | 0.1013 | (0.0005, 0.2134) | 0.0230 | 0.1013 | (-0.0259, 0.2613) | 0.0677 |
| controlled_vs_proposed_raw | distinct1 | 0.0200 | (0.0048, 0.0359) | 0.0063 | 0.0200 | (0.0029, 0.0353) | 0.0147 |
| controlled_vs_proposed_raw | length_score | 0.4367 | (0.2650, 0.5917) | 0.0000 | 0.4367 | (0.2380, 0.5653) | 0.0000 |
| controlled_vs_proposed_raw | sentence_score | 0.2275 | (0.1575, 0.2975) | 0.0000 | 0.2275 | (0.1615, 0.2834) | 0.0000 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0165 | (-0.0141, 0.0432) | 0.1387 | 0.0165 | (-0.0191, 0.0508) | 0.1683 |
| controlled_vs_proposed_raw | overall_quality | 0.1573 | (0.1103, 0.2031) | 0.0000 | 0.1573 | (0.1039, 0.2155) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2567 | (0.2125, 0.3077) | 0.0000 | 0.2567 | (0.2151, 0.3153) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1392 | (0.0526, 0.2189) | 0.0003 | 0.1392 | (0.0859, 0.2092) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.0921 | (0.0418, 0.1374) | 0.0003 | 0.0921 | (0.0262, 0.1426) | 0.0043 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3396 | (0.2792, 0.4044) | 0.0000 | 0.3396 | (0.2833, 0.4162) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0633 | (0.0474, 0.0792) | 0.0000 | 0.0633 | (0.0519, 0.0780) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1507 | (0.0517, 0.2479) | 0.0013 | 0.1507 | (0.0896, 0.2363) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.0931 | (-0.0058, 0.2113) | 0.0343 | 0.0931 | (-0.0342, 0.2457) | 0.0973 |
| controlled_vs_candidate_no_context | distinct1 | 0.0097 | (-0.0105, 0.0295) | 0.1697 | 0.0097 | (-0.0168, 0.0346) | 0.2340 |
| controlled_vs_candidate_no_context | length_score | 0.3450 | (0.1383, 0.5400) | 0.0007 | 0.3450 | (0.0896, 0.5433) | 0.0053 |
| controlled_vs_candidate_no_context | sentence_score | 0.1925 | (0.1050, 0.2625) | 0.0000 | 0.1925 | (0.1235, 0.2450) | 0.0000 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0542 | (0.0308, 0.0778) | 0.0000 | 0.0542 | (0.0277, 0.0901) | 0.0000 |
| controlled_vs_candidate_no_context | overall_quality | 0.1724 | (0.1300, 0.2120) | 0.0000 | 0.1724 | (0.1420, 0.2157) | 0.0000 |
| controlled_vs_baseline_no_context | context_relevance | 0.2667 | (0.2252, 0.3143) | 0.0000 | 0.2667 | (0.2304, 0.3177) | 0.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1091 | (0.0448, 0.1777) | 0.0007 | 0.1091 | (0.0255, 0.2047) | 0.0037 |
| controlled_vs_baseline_no_context | naturalness | 0.0175 | (-0.0243, 0.0583) | 0.2023 | 0.0175 | (-0.0467, 0.0703) | 0.2817 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3525 | (0.2968, 0.4177) | 0.0000 | 0.3525 | (0.3012, 0.4220) | 0.0000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0664 | (0.0512, 0.0826) | 0.0000 | 0.0664 | (0.0555, 0.0820) | 0.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1357 | (0.0502, 0.2207) | 0.0010 | 0.1357 | (0.0304, 0.2641) | 0.0033 |
| controlled_vs_baseline_no_context | persona_style | 0.0025 | (-0.0450, 0.0500) | 0.4683 | 0.0025 | (-0.0553, 0.0714) | 0.4963 |
| controlled_vs_baseline_no_context | distinct1 | -0.0318 | (-0.0487, -0.0136) | 0.9997 | -0.0318 | (-0.0545, -0.0046) | 0.9903 |
| controlled_vs_baseline_no_context | length_score | 0.1250 | (-0.0833, 0.3067) | 0.1103 | 0.1250 | (-0.1834, 0.3407) | 0.1693 |
| controlled_vs_baseline_no_context | sentence_score | 0.0525 | (-0.0175, 0.1225) | 0.1120 | 0.0525 | (-0.0250, 0.1556) | 0.1647 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0209 | (-0.0025, 0.0439) | 0.0387 | 0.0209 | (-0.0012, 0.0396) | 0.0317 |
| controlled_vs_baseline_no_context | overall_quality | 0.1501 | (0.1133, 0.1867) | 0.0000 | 0.1501 | (0.1056, 0.2032) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2667 | (0.2251, 0.3125) | 0.0000 | 0.2667 | (0.2311, 0.3203) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1091 | (0.0400, 0.1745) | 0.0020 | 0.1091 | (0.0308, 0.2064) | 0.0047 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 0.0175 | (-0.0266, 0.0569) | 0.2050 | 0.0175 | (-0.0508, 0.0718) | 0.2700 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3525 | (0.2967, 0.4194) | 0.0000 | 0.3525 | (0.3015, 0.4255) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0664 | (0.0509, 0.0819) | 0.0000 | 0.0664 | (0.0561, 0.0820) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1357 | (0.0540, 0.2210) | 0.0007 | 0.1357 | (0.0342, 0.2565) | 0.0020 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0025 | (-0.0450, 0.0500) | 0.4753 | 0.0025 | (-0.0579, 0.0714) | 0.5023 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0318 | (-0.0480, -0.0138) | 0.9990 | -0.0318 | (-0.0546, -0.0042) | 0.9887 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.1250 | (-0.0750, 0.3083) | 0.1080 | 0.1250 | (-0.1900, 0.3386) | 0.1747 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0525 | (-0.0175, 0.1225) | 0.1210 | 0.0525 | (-0.0269, 0.1575) | 0.1527 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0209 | (-0.0021, 0.0444) | 0.0423 | 0.0209 | (-0.0038, 0.0408) | 0.0427 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1501 | (0.1146, 0.1866) | 0.0000 | 0.1501 | (0.1065, 0.2033) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 7 | 3 | 10 | 0.6000 | 0.7000 |
| proposed_vs_candidate_no_context | persona_consistency | 4 | 4 | 12 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | naturalness | 3 | 7 | 10 | 0.4000 | 0.3000 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 6 | 2 | 12 | 0.6000 | 0.7500 |
| proposed_vs_candidate_no_context | context_overlap | 7 | 3 | 10 | 0.6000 | 0.7000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 3 | 3 | 14 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | persona_style | 1 | 1 | 18 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | distinct1 | 3 | 7 | 10 | 0.4000 | 0.3000 |
| proposed_vs_candidate_no_context | length_score | 3 | 6 | 11 | 0.4250 | 0.3333 |
| proposed_vs_candidate_no_context | sentence_score | 2 | 4 | 14 | 0.4500 | 0.3333 |
| proposed_vs_candidate_no_context | bertscore_f1 | 10 | 3 | 7 | 0.6750 | 0.7692 |
| proposed_vs_candidate_no_context | overall_quality | 6 | 7 | 7 | 0.4750 | 0.4615 |
| proposed_vs_baseline_no_context | context_relevance | 11 | 9 | 0 | 0.5500 | 0.5500 |
| proposed_vs_baseline_no_context | persona_consistency | 2 | 10 | 8 | 0.3000 | 0.1667 |
| proposed_vs_baseline_no_context | naturalness | 2 | 18 | 0 | 0.1000 | 0.1000 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 7 | 3 | 10 | 0.6000 | 0.7000 |
| proposed_vs_baseline_no_context | context_overlap | 12 | 8 | 0 | 0.6000 | 0.6000 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 2 | 6 | 12 | 0.4000 | 0.2500 |
| proposed_vs_baseline_no_context | persona_style | 0 | 5 | 15 | 0.3750 | 0.0000 |
| proposed_vs_baseline_no_context | distinct1 | 3 | 17 | 0 | 0.1500 | 0.1500 |
| proposed_vs_baseline_no_context | length_score | 4 | 15 | 1 | 0.2250 | 0.2105 |
| proposed_vs_baseline_no_context | sentence_score | 1 | 11 | 8 | 0.2500 | 0.0833 |
| proposed_vs_baseline_no_context | bertscore_f1 | 10 | 10 | 0 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context | overall_quality | 7 | 13 | 0 | 0.3500 | 0.3500 |
| controlled_vs_proposed_raw | context_relevance | 18 | 2 | 0 | 0.9000 | 0.9000 |
| controlled_vs_proposed_raw | persona_consistency | 15 | 2 | 3 | 0.8250 | 0.8824 |
| controlled_vs_proposed_raw | naturalness | 18 | 2 | 0 | 0.9000 | 0.9000 |
| controlled_vs_proposed_raw | context_keyword_coverage | 17 | 1 | 2 | 0.9000 | 0.9444 |
| controlled_vs_proposed_raw | context_overlap | 18 | 2 | 0 | 0.9000 | 0.9000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 15 | 2 | 3 | 0.8250 | 0.8824 |
| controlled_vs_proposed_raw | persona_style | 6 | 2 | 12 | 0.6000 | 0.7500 |
| controlled_vs_proposed_raw | distinct1 | 15 | 5 | 0 | 0.7500 | 0.7500 |
| controlled_vs_proposed_raw | length_score | 15 | 4 | 1 | 0.7750 | 0.7895 |
| controlled_vs_proposed_raw | sentence_score | 13 | 0 | 7 | 0.8250 | 1.0000 |
| controlled_vs_proposed_raw | bertscore_f1 | 14 | 6 | 0 | 0.7000 | 0.7000 |
| controlled_vs_proposed_raw | overall_quality | 18 | 2 | 0 | 0.9000 | 0.9000 |
| controlled_vs_candidate_no_context | context_relevance | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 14 | 4 | 2 | 0.7500 | 0.7778 |
| controlled_vs_candidate_no_context | naturalness | 16 | 4 | 0 | 0.8000 | 0.8000 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 14 | 3 | 3 | 0.7750 | 0.8235 |
| controlled_vs_candidate_no_context | persona_style | 5 | 2 | 13 | 0.5750 | 0.7143 |
| controlled_vs_candidate_no_context | distinct1 | 12 | 8 | 0 | 0.6000 | 0.6000 |
| controlled_vs_candidate_no_context | length_score | 14 | 5 | 1 | 0.7250 | 0.7368 |
| controlled_vs_candidate_no_context | sentence_score | 11 | 0 | 9 | 0.7750 | 1.0000 |
| controlled_vs_candidate_no_context | bertscore_f1 | 17 | 3 | 0 | 0.8500 | 0.8500 |
| controlled_vs_candidate_no_context | overall_quality | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_vs_baseline_no_context | context_relevance | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 14 | 3 | 3 | 0.7750 | 0.8235 |
| controlled_vs_baseline_no_context | naturalness | 13 | 7 | 0 | 0.6500 | 0.6500 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | context_overlap | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 13 | 3 | 4 | 0.7500 | 0.8125 |
| controlled_vs_baseline_no_context | persona_style | 2 | 2 | 16 | 0.5000 | 0.5000 |
| controlled_vs_baseline_no_context | distinct1 | 3 | 17 | 0 | 0.1500 | 0.1500 |
| controlled_vs_baseline_no_context | length_score | 13 | 7 | 0 | 0.6500 | 0.6500 |
| controlled_vs_baseline_no_context | sentence_score | 4 | 1 | 15 | 0.5750 | 0.8000 |
| controlled_vs_baseline_no_context | bertscore_f1 | 12 | 8 | 0 | 0.6000 | 0.6000 |
| controlled_vs_baseline_no_context | overall_quality | 20 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 20 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 14 | 3 | 3 | 0.7750 | 0.8235 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 13 | 7 | 0 | 0.6500 | 0.6500 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 20 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 20 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 13 | 3 | 4 | 0.7500 | 0.8125 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 2 | 2 | 16 | 0.5000 | 0.5000 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | 3 | 17 | 0 | 0.1500 | 0.1500 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 13 | 7 | 0 | 0.6500 | 0.6500 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 4 | 1 | 15 | 0.5750 | 0.8000 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 12 | 8 | 0 | 0.6000 | 0.6000 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 20 | 0 | 0 | 1.0000 | 1.0000 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.3500 | 0.4000 | 0.6000 |
| proposed_contextual | 0.0000 | 0.0000 | 0.6000 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.5500 | 0.0000 | 0.0000 |
| baseline_no_context | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `8`
- Unique template signatures: `18`
- Template signature ratio: `0.9000`
- Effective sample size by source clustering: `6.67`
- Effective sample size by template-signature clustering: `16.67`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: enabled.

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
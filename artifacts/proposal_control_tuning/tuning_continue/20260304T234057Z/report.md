# Proposal Alignment Evaluation Report

- Run ID: `20260304T234057Z`
- Generated: `2026-03-04T23:45:45.080682+00:00`
- Scenarios: `artifacts\proposal_control_tuning\tuning_continue\20260304T234057Z\scenarios.jsonl`
- Scenario count: `20`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off
- `baseline_no_context`: model `phi3:mini`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2650 (0.2384, 0.2939) | 0.3772 (0.3049, 0.4659) | 0.8630 (0.8277, 0.8981) | 0.3885 (0.3631, 0.4182) | 0.0747 |
| proposed_contextual | 0.0952 (0.0487, 0.1471) | 0.1566 (0.1079, 0.2100) | 0.7936 (0.7685, 0.8215) | 0.2337 (0.1997, 0.2723) | 0.0673 |
| candidate_no_context | 0.0274 (0.0137, 0.0459) | 0.1761 (0.1217, 0.2350) | 0.8199 (0.7836, 0.8585) | 0.2135 (0.1893, 0.2402) | 0.0365 |
| baseline_no_context | 0.0438 (0.0245, 0.0661) | 0.1912 (0.1460, 0.2394) | 0.9029 (0.8781, 0.9274) | 0.2413 (0.2242, 0.2605) | 0.0514 |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0679 | 2.4795 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0194 | -0.1103 |
| proposed_vs_candidate_no_context | naturalness | -0.0263 | -0.0321 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0883 | 3.8833 |
| proposed_vs_candidate_no_context | context_overlap | 0.0203 | 0.5310 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0195 | -0.2433 |
| proposed_vs_candidate_no_context | persona_style | -0.0190 | -0.0341 |
| proposed_vs_candidate_no_context | distinct1 | -0.0097 | -0.0102 |
| proposed_vs_candidate_no_context | length_score | -0.1033 | -0.3229 |
| proposed_vs_candidate_no_context | sentence_score | -0.0175 | -0.0227 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0308 | 0.8427 |
| proposed_vs_candidate_no_context | overall_quality | 0.0202 | 0.0944 |
| proposed_vs_baseline_no_context | context_relevance | 0.0514 | 1.1722 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0346 | -0.1807 |
| proposed_vs_baseline_no_context | naturalness | -0.1093 | -0.1210 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0670 | 1.5259 |
| proposed_vs_baseline_no_context | context_overlap | 0.0149 | 0.3409 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0262 | -0.3014 |
| proposed_vs_baseline_no_context | persona_style | -0.0680 | -0.1118 |
| proposed_vs_baseline_no_context | distinct1 | -0.0432 | -0.0441 |
| proposed_vs_baseline_no_context | length_score | -0.3900 | -0.6429 |
| proposed_vs_baseline_no_context | sentence_score | -0.1400 | -0.1564 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0159 | 0.3088 |
| proposed_vs_baseline_no_context | overall_quality | -0.0076 | -0.0316 |
| controlled_vs_proposed_raw | context_relevance | 0.1698 | 1.7826 |
| controlled_vs_proposed_raw | persona_consistency | 0.2206 | 1.4083 |
| controlled_vs_proposed_raw | naturalness | 0.0694 | 0.0874 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2233 | 2.0116 |
| controlled_vs_proposed_raw | context_overlap | 0.0450 | 0.7690 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2607 | 4.2941 |
| controlled_vs_proposed_raw | persona_style | 0.0601 | 0.1112 |
| controlled_vs_proposed_raw | distinct1 | -0.0059 | -0.0063 |
| controlled_vs_proposed_raw | length_score | 0.2800 | 1.2923 |
| controlled_vs_proposed_raw | sentence_score | 0.1575 | 0.2086 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0074 | 0.1102 |
| controlled_vs_proposed_raw | overall_quality | 0.1548 | 0.6622 |
| controlled_vs_candidate_no_context | context_relevance | 0.2376 | 8.6821 |
| controlled_vs_candidate_no_context | persona_consistency | 0.2012 | 1.1425 |
| controlled_vs_candidate_no_context | naturalness | 0.0431 | 0.0526 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3115 | 13.7067 |
| controlled_vs_candidate_no_context | context_overlap | 0.0653 | 1.7084 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2412 | 3.0059 |
| controlled_vs_candidate_no_context | persona_style | 0.0410 | 0.0734 |
| controlled_vs_candidate_no_context | distinct1 | -0.0155 | -0.0164 |
| controlled_vs_candidate_no_context | length_score | 0.1767 | 0.5521 |
| controlled_vs_candidate_no_context | sentence_score | 0.1400 | 0.1812 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0382 | 1.0457 |
| controlled_vs_candidate_no_context | overall_quality | 0.1749 | 0.8192 |
| controlled_vs_baseline_no_context | context_relevance | 0.2212 | 5.0444 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1860 | 0.9730 |
| controlled_vs_baseline_no_context | naturalness | -0.0399 | -0.0442 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.2903 | 6.6069 |
| controlled_vs_baseline_no_context | context_overlap | 0.0599 | 1.3721 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2345 | 2.6986 |
| controlled_vs_baseline_no_context | persona_style | -0.0079 | -0.0130 |
| controlled_vs_baseline_no_context | distinct1 | -0.0491 | -0.0501 |
| controlled_vs_baseline_no_context | length_score | -0.1100 | -0.1813 |
| controlled_vs_baseline_no_context | sentence_score | 0.0175 | 0.0196 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0233 | 0.4530 |
| controlled_vs_baseline_no_context | overall_quality | 0.1471 | 0.6097 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2212 | 5.0444 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1860 | 0.9730 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0399 | -0.0442 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.2903 | 6.6069 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0599 | 1.3721 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2345 | 2.6986 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | -0.0079 | -0.0130 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0491 | -0.0501 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | -0.1100 | -0.1813 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0175 | 0.0196 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0233 | 0.4530 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1471 | 0.6097 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0679 | (0.0205, 0.1253) | 0.0010 | 0.0679 | (0.0213, 0.1266) | 0.0000 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0194 | (-0.0492, 0.0083) | 0.9040 | -0.0194 | (-0.0399, 0.0095) | 0.9153 |
| proposed_vs_candidate_no_context | naturalness | -0.0263 | (-0.0664, 0.0134) | 0.9030 | -0.0263 | (-0.0618, 0.0130) | 0.9147 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0883 | (0.0273, 0.1640) | 0.0003 | 0.0883 | (0.0245, 0.1715) | 0.0003 |
| proposed_vs_candidate_no_context | context_overlap | 0.0203 | (0.0055, 0.0382) | 0.0007 | 0.0203 | (0.0068, 0.0431) | 0.0000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0195 | (-0.0600, 0.0202) | 0.8557 | -0.0195 | (-0.0470, 0.0233) | 0.8553 |
| proposed_vs_candidate_no_context | persona_style | -0.0190 | (-0.0774, 0.0226) | 0.7730 | -0.0190 | (-0.0966, 0.0215) | 0.7510 |
| proposed_vs_candidate_no_context | distinct1 | -0.0097 | (-0.0266, 0.0070) | 0.8727 | -0.0097 | (-0.0273, 0.0059) | 0.8933 |
| proposed_vs_candidate_no_context | length_score | -0.1033 | (-0.2733, 0.0534) | 0.8997 | -0.1033 | (-0.2167, 0.0438) | 0.9277 |
| proposed_vs_candidate_no_context | sentence_score | -0.0175 | (-0.0875, 0.0525) | 0.7490 | -0.0175 | (-0.0955, 0.0657) | 0.7583 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0308 | (0.0054, 0.0576) | 0.0087 | 0.0308 | (0.0140, 0.0523) | 0.0000 |
| proposed_vs_candidate_no_context | overall_quality | 0.0202 | (-0.0049, 0.0484) | 0.0590 | 0.0202 | (-0.0066, 0.0570) | 0.0863 |
| proposed_vs_baseline_no_context | context_relevance | 0.0514 | (0.0127, 0.0948) | 0.0020 | 0.0514 | (0.0095, 0.0972) | 0.0060 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0346 | (-0.0893, 0.0190) | 0.8853 | -0.0346 | (-0.0877, 0.0349) | 0.8380 |
| proposed_vs_baseline_no_context | naturalness | -0.1093 | (-0.1492, -0.0633) | 1.0000 | -0.1093 | (-0.1540, -0.0341) | 0.9953 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0670 | (0.0170, 0.1258) | 0.0033 | 0.0670 | (0.0148, 0.1256) | 0.0060 |
| proposed_vs_baseline_no_context | context_overlap | 0.0149 | (0.0011, 0.0306) | 0.0183 | 0.0149 | (-0.0024, 0.0365) | 0.0463 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0262 | (-0.0900, 0.0331) | 0.7857 | -0.0262 | (-0.0945, 0.0476) | 0.7060 |
| proposed_vs_baseline_no_context | persona_style | -0.0680 | (-0.1724, 0.0060) | 0.9443 | -0.0680 | (-0.2010, 0.0119) | 0.8730 |
| proposed_vs_baseline_no_context | distinct1 | -0.0432 | (-0.0601, -0.0250) | 1.0000 | -0.0432 | (-0.0659, -0.0223) | 1.0000 |
| proposed_vs_baseline_no_context | length_score | -0.3900 | (-0.5400, -0.2133) | 1.0000 | -0.3900 | (-0.5458, -0.1356) | 0.9967 |
| proposed_vs_baseline_no_context | sentence_score | -0.1400 | (-0.2450, -0.0350) | 0.9953 | -0.1400 | (-0.2693, 0.0584) | 0.9353 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0159 | (-0.0097, 0.0410) | 0.1187 | 0.0159 | (-0.0145, 0.0461) | 0.1350 |
| proposed_vs_baseline_no_context | overall_quality | -0.0076 | (-0.0439, 0.0313) | 0.6650 | -0.0076 | (-0.0470, 0.0409) | 0.5980 |
| controlled_vs_proposed_raw | context_relevance | 0.1698 | (0.1235, 0.2116) | 0.0000 | 0.1698 | (0.1022, 0.2197) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.2206 | (0.1400, 0.3128) | 0.0000 | 0.2206 | (0.1343, 0.3003) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0694 | (0.0206, 0.1138) | 0.0017 | 0.0694 | (0.0125, 0.1323) | 0.0067 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2233 | (0.1629, 0.2787) | 0.0000 | 0.2233 | (0.1408, 0.2848) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0450 | (0.0271, 0.0610) | 0.0000 | 0.0450 | (0.0154, 0.0656) | 0.0020 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2607 | (0.1609, 0.3743) | 0.0000 | 0.2607 | (0.1508, 0.3741) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.0601 | (-0.0329, 0.1627) | 0.1137 | 0.0601 | (-0.0472, 0.2084) | 0.2223 |
| controlled_vs_proposed_raw | distinct1 | -0.0059 | (-0.0305, 0.0181) | 0.6780 | -0.0059 | (-0.0454, 0.0235) | 0.6443 |
| controlled_vs_proposed_raw | length_score | 0.2800 | (0.0816, 0.4700) | 0.0050 | 0.2800 | (0.0765, 0.5393) | 0.0010 |
| controlled_vs_proposed_raw | sentence_score | 0.1575 | (0.0700, 0.2450) | 0.0000 | 0.1575 | (0.0437, 0.2423) | 0.0047 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0074 | (-0.0175, 0.0322) | 0.2780 | 0.0074 | (-0.0142, 0.0231) | 0.2283 |
| controlled_vs_proposed_raw | overall_quality | 0.1548 | (0.1092, 0.1998) | 0.0000 | 0.1548 | (0.0980, 0.1988) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2376 | (0.2124, 0.2654) | 0.0000 | 0.2376 | (0.2239, 0.2505) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.2012 | (0.1063, 0.3030) | 0.0000 | 0.2012 | (0.1207, 0.2832) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.0431 | (-0.0064, 0.0930) | 0.0480 | 0.0431 | (-0.0135, 0.1001) | 0.0620 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3115 | (0.2770, 0.3470) | 0.0000 | 0.3115 | (0.2909, 0.3319) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0653 | (0.0540, 0.0756) | 0.0000 | 0.0653 | (0.0547, 0.0746) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2412 | (0.1312, 0.3755) | 0.0000 | 0.2412 | (0.1315, 0.3519) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.0410 | (-0.0709, 0.1646) | 0.2450 | 0.0410 | (-0.1119, 0.1915) | 0.3420 |
| controlled_vs_candidate_no_context | distinct1 | -0.0155 | (-0.0379, 0.0055) | 0.9160 | -0.0155 | (-0.0507, 0.0111) | 0.8610 |
| controlled_vs_candidate_no_context | length_score | 0.1767 | (-0.0367, 0.3800) | 0.0487 | 0.1767 | (-0.0271, 0.4118) | 0.0453 |
| controlled_vs_candidate_no_context | sentence_score | 0.1400 | (0.0350, 0.2450) | 0.0097 | 0.1400 | (0.0269, 0.2407) | 0.0123 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0382 | (0.0158, 0.0645) | 0.0000 | 0.0382 | (0.0180, 0.0631) | 0.0000 |
| controlled_vs_candidate_no_context | overall_quality | 0.1749 | (0.1399, 0.2128) | 0.0000 | 0.1749 | (0.1449, 0.1995) | 0.0000 |
| controlled_vs_baseline_no_context | context_relevance | 0.2212 | (0.1931, 0.2514) | 0.0000 | 0.2212 | (0.1841, 0.2559) | 0.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1860 | (0.1288, 0.2505) | 0.0000 | 0.1860 | (0.1355, 0.2270) | 0.0000 |
| controlled_vs_baseline_no_context | naturalness | -0.0399 | (-0.0875, 0.0073) | 0.9493 | -0.0399 | (-0.0957, 0.0283) | 0.8750 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.2903 | (0.2498, 0.3303) | 0.0000 | 0.2903 | (0.2417, 0.3418) | 0.0000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0599 | (0.0473, 0.0720) | 0.0000 | 0.0599 | (0.0448, 0.0705) | 0.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2345 | (0.1650, 0.3121) | 0.0000 | 0.2345 | (0.1736, 0.2861) | 0.0000 |
| controlled_vs_baseline_no_context | persona_style | -0.0079 | (-0.0621, 0.0333) | 0.5673 | -0.0079 | (-0.0455, 0.0213) | 0.6727 |
| controlled_vs_baseline_no_context | distinct1 | -0.0491 | (-0.0716, -0.0256) | 1.0000 | -0.0491 | (-0.0729, -0.0293) | 1.0000 |
| controlled_vs_baseline_no_context | length_score | -0.1100 | (-0.3184, 0.0984) | 0.8453 | -0.1100 | (-0.3644, 0.2033) | 0.7543 |
| controlled_vs_baseline_no_context | sentence_score | 0.0175 | (-0.1050, 0.1225) | 0.4270 | 0.0175 | (-0.0840, 0.1750) | 0.4490 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0233 | (-0.0012, 0.0468) | 0.0303 | 0.0233 | (-0.0081, 0.0500) | 0.0643 |
| controlled_vs_baseline_no_context | overall_quality | 0.1471 | (0.1216, 0.1710) | 0.0000 | 0.1471 | (0.1243, 0.1630) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2212 | (0.1913, 0.2508) | 0.0000 | 0.2212 | (0.1818, 0.2556) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1860 | (0.1265, 0.2509) | 0.0000 | 0.1860 | (0.1375, 0.2259) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0399 | (-0.0863, 0.0051) | 0.9547 | -0.0399 | (-0.0935, 0.0283) | 0.8707 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.2903 | (0.2515, 0.3307) | 0.0000 | 0.2903 | (0.2406, 0.3429) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0599 | (0.0471, 0.0720) | 0.0000 | 0.0599 | (0.0457, 0.0702) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2345 | (0.1621, 0.3088) | 0.0000 | 0.2345 | (0.1743, 0.2850) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | -0.0079 | (-0.0583, 0.0333) | 0.5860 | -0.0079 | (-0.0434, 0.0208) | 0.6767 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0491 | (-0.0706, -0.0261) | 1.0000 | -0.0491 | (-0.0731, -0.0286) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | -0.1100 | (-0.3217, 0.0967) | 0.8553 | -0.1100 | (-0.3608, 0.1926) | 0.7370 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0175 | (-0.1050, 0.1225) | 0.4563 | 0.0175 | (-0.0840, 0.1750) | 0.4443 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0233 | (-0.0018, 0.0469) | 0.0337 | 0.0233 | (-0.0084, 0.0486) | 0.0663 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1471 | (0.1228, 0.1717) | 0.0000 | 0.1471 | (0.1226, 0.1635) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 9 | 3 | 8 | 0.6500 | 0.7500 |
| proposed_vs_candidate_no_context | persona_consistency | 3 | 5 | 12 | 0.4500 | 0.3750 |
| proposed_vs_candidate_no_context | naturalness | 5 | 6 | 9 | 0.4750 | 0.4545 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 6 | 0 | 14 | 0.6500 | 1.0000 |
| proposed_vs_candidate_no_context | context_overlap | 9 | 3 | 8 | 0.6500 | 0.7500 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 2 | 4 | 14 | 0.4500 | 0.3333 |
| proposed_vs_candidate_no_context | persona_style | 1 | 2 | 17 | 0.4750 | 0.3333 |
| proposed_vs_candidate_no_context | distinct1 | 2 | 7 | 11 | 0.3750 | 0.2222 |
| proposed_vs_candidate_no_context | length_score | 5 | 6 | 9 | 0.4750 | 0.4545 |
| proposed_vs_candidate_no_context | sentence_score | 2 | 3 | 15 | 0.4750 | 0.4000 |
| proposed_vs_candidate_no_context | bertscore_f1 | 12 | 4 | 4 | 0.7000 | 0.7500 |
| proposed_vs_candidate_no_context | overall_quality | 11 | 5 | 4 | 0.6500 | 0.6875 |
| proposed_vs_baseline_no_context | context_relevance | 12 | 8 | 0 | 0.6000 | 0.6000 |
| proposed_vs_baseline_no_context | persona_consistency | 5 | 6 | 9 | 0.4750 | 0.4545 |
| proposed_vs_baseline_no_context | naturalness | 2 | 18 | 0 | 0.1000 | 0.1000 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 8 | 2 | 10 | 0.6500 | 0.8000 |
| proposed_vs_baseline_no_context | context_overlap | 11 | 9 | 0 | 0.5500 | 0.5500 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 4 | 5 | 11 | 0.4750 | 0.4444 |
| proposed_vs_baseline_no_context | persona_style | 1 | 3 | 16 | 0.4500 | 0.2500 |
| proposed_vs_baseline_no_context | distinct1 | 2 | 15 | 3 | 0.1750 | 0.1176 |
| proposed_vs_baseline_no_context | length_score | 2 | 18 | 0 | 0.1000 | 0.1000 |
| proposed_vs_baseline_no_context | sentence_score | 3 | 11 | 6 | 0.3000 | 0.2143 |
| proposed_vs_baseline_no_context | bertscore_f1 | 13 | 7 | 0 | 0.6500 | 0.6500 |
| proposed_vs_baseline_no_context | overall_quality | 9 | 11 | 0 | 0.4500 | 0.4500 |
| controlled_vs_proposed_raw | context_relevance | 18 | 2 | 0 | 0.9000 | 0.9000 |
| controlled_vs_proposed_raw | persona_consistency | 16 | 2 | 2 | 0.8500 | 0.8889 |
| controlled_vs_proposed_raw | naturalness | 15 | 5 | 0 | 0.7500 | 0.7500 |
| controlled_vs_proposed_raw | context_keyword_coverage | 17 | 2 | 1 | 0.8750 | 0.8947 |
| controlled_vs_proposed_raw | context_overlap | 18 | 2 | 0 | 0.9000 | 0.9000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 16 | 0 | 4 | 0.9000 | 1.0000 |
| controlled_vs_proposed_raw | persona_style | 5 | 2 | 13 | 0.5750 | 0.7143 |
| controlled_vs_proposed_raw | distinct1 | 9 | 11 | 0 | 0.4500 | 0.4500 |
| controlled_vs_proposed_raw | length_score | 13 | 6 | 1 | 0.6750 | 0.6842 |
| controlled_vs_proposed_raw | sentence_score | 10 | 1 | 9 | 0.7250 | 0.9091 |
| controlled_vs_proposed_raw | bertscore_f1 | 10 | 10 | 0 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | overall_quality | 18 | 2 | 0 | 0.9000 | 0.9000 |
| controlled_vs_candidate_no_context | context_relevance | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 15 | 3 | 2 | 0.8000 | 0.8333 |
| controlled_vs_candidate_no_context | naturalness | 12 | 8 | 0 | 0.6000 | 0.6000 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 15 | 2 | 3 | 0.8250 | 0.8824 |
| controlled_vs_candidate_no_context | persona_style | 4 | 3 | 13 | 0.5250 | 0.5714 |
| controlled_vs_candidate_no_context | distinct1 | 8 | 10 | 2 | 0.4500 | 0.4444 |
| controlled_vs_candidate_no_context | length_score | 12 | 7 | 1 | 0.6250 | 0.6316 |
| controlled_vs_candidate_no_context | sentence_score | 10 | 2 | 8 | 0.7000 | 0.8333 |
| controlled_vs_candidate_no_context | bertscore_f1 | 14 | 6 | 0 | 0.7000 | 0.7000 |
| controlled_vs_candidate_no_context | overall_quality | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | context_relevance | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 17 | 0 | 3 | 0.9250 | 1.0000 |
| controlled_vs_baseline_no_context | naturalness | 7 | 13 | 0 | 0.3500 | 0.3500 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | context_overlap | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 17 | 0 | 3 | 0.9250 | 1.0000 |
| controlled_vs_baseline_no_context | persona_style | 2 | 2 | 16 | 0.5000 | 0.5000 |
| controlled_vs_baseline_no_context | distinct1 | 3 | 17 | 0 | 0.1500 | 0.1500 |
| controlled_vs_baseline_no_context | length_score | 9 | 11 | 0 | 0.4500 | 0.4500 |
| controlled_vs_baseline_no_context | sentence_score | 6 | 5 | 9 | 0.5250 | 0.5455 |
| controlled_vs_baseline_no_context | bertscore_f1 | 14 | 6 | 0 | 0.7000 | 0.7000 |
| controlled_vs_baseline_no_context | overall_quality | 20 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 20 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 17 | 0 | 3 | 0.9250 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 7 | 13 | 0 | 0.3500 | 0.3500 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 20 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 19 | 1 | 0 | 0.9500 | 0.9500 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 17 | 0 | 3 | 0.9250 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 2 | 2 | 16 | 0.5000 | 0.5000 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | 3 | 17 | 0 | 0.1500 | 0.1500 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 9 | 11 | 0 | 0.4500 | 0.4500 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 6 | 5 | 9 | 0.5250 | 0.5455 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 14 | 6 | 0 | 0.7000 | 0.7000 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 20 | 0 | 0 | 1.0000 | 1.0000 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.2500 | 0.5000 | 0.5000 |
| proposed_contextual | 0.0000 | 0.0000 | 0.4500 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 |
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
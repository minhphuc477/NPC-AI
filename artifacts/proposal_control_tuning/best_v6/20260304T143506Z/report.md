# Proposal Alignment Evaluation Report

- Run ID: `20260304T143506Z`
- Generated: `2026-03-04T14:42:09.927439+00:00`
- Scenarios: `artifacts\proposal_control_tuning\best_v6\20260304T143506Z\scenarios.jsonl`
- Scenario count: `40`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off
- `baseline_no_context`: model `phi3:mini`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2846 (0.2556, 0.3190) | 0.3146 (0.2670, 0.3647) | 0.8988 (0.8731, 0.9235) | 0.3842 (0.3646, 0.4050) | 0.0928 |
| proposed_contextual | 0.0765 (0.0429, 0.1159) | 0.1212 (0.0911, 0.1535) | 0.7933 (0.7727, 0.8154) | 0.2153 (0.1912, 0.2432) | 0.0695 |
| candidate_no_context | 0.0267 (0.0160, 0.0399) | 0.1816 (0.1336, 0.2346) | 0.8052 (0.7801, 0.8313) | 0.2138 (0.1939, 0.2367) | 0.0406 |
| baseline_no_context | 0.0317 (0.0222, 0.0427) | 0.2014 (0.1634, 0.2415) | 0.8691 (0.8490, 0.8902) | 0.2346 (0.2194, 0.2504) | 0.0515 |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0498 | 1.8641 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0605 | -0.3330 |
| proposed_vs_candidate_no_context | naturalness | -0.0119 | -0.0148 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0612 | 2.9579 |
| proposed_vs_candidate_no_context | context_overlap | 0.0232 | 0.5686 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0667 | -0.6313 |
| proposed_vs_candidate_no_context | persona_style | -0.0357 | -0.0736 |
| proposed_vs_candidate_no_context | distinct1 | -0.0018 | -0.0019 |
| proposed_vs_candidate_no_context | length_score | -0.0342 | -0.1195 |
| proposed_vs_candidate_no_context | sentence_score | -0.0438 | -0.0579 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0289 | 0.7120 |
| proposed_vs_candidate_no_context | overall_quality | 0.0015 | 0.0070 |
| proposed_vs_baseline_no_context | context_relevance | 0.0448 | 1.4134 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0802 | -0.3984 |
| proposed_vs_baseline_no_context | naturalness | -0.0758 | -0.0872 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0523 | 1.7670 |
| proposed_vs_baseline_no_context | context_overlap | 0.0273 | 0.7465 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0748 | -0.6576 |
| proposed_vs_baseline_no_context | persona_style | -0.1021 | -0.1849 |
| proposed_vs_baseline_no_context | distinct1 | -0.0420 | -0.0432 |
| proposed_vs_baseline_no_context | length_score | -0.2250 | -0.4720 |
| proposed_vs_baseline_no_context | sentence_score | -0.1400 | -0.1645 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0180 | 0.3491 |
| proposed_vs_baseline_no_context | overall_quality | -0.0193 | -0.0821 |
| controlled_vs_proposed_raw | context_relevance | 0.2081 | 2.7218 |
| controlled_vs_proposed_raw | persona_consistency | 0.1935 | 1.5966 |
| controlled_vs_proposed_raw | naturalness | 0.1055 | 0.1330 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2745 | 3.3535 |
| controlled_vs_proposed_raw | context_overlap | 0.0533 | 0.8335 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2090 | 5.3700 |
| controlled_vs_proposed_raw | persona_style | 0.1311 | 0.2912 |
| controlled_vs_proposed_raw | distinct1 | 0.0019 | 0.0020 |
| controlled_vs_proposed_raw | length_score | 0.4008 | 1.5927 |
| controlled_vs_proposed_raw | sentence_score | 0.2625 | 0.3691 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0233 | 0.3347 |
| controlled_vs_proposed_raw | overall_quality | 0.1689 | 0.7844 |
| controlled_vs_candidate_no_context | context_relevance | 0.2579 | 9.6596 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1330 | 0.7320 |
| controlled_vs_candidate_no_context | naturalness | 0.0936 | 0.1162 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3357 | 16.2308 |
| controlled_vs_candidate_no_context | context_overlap | 0.0764 | 1.8759 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1424 | 1.3484 |
| controlled_vs_candidate_no_context | persona_style | 0.0953 | 0.1962 |
| controlled_vs_candidate_no_context | distinct1 | 0.0001 | 0.0001 |
| controlled_vs_candidate_no_context | length_score | 0.3667 | 1.2828 |
| controlled_vs_candidate_no_context | sentence_score | 0.2188 | 0.2897 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0522 | 1.2850 |
| controlled_vs_candidate_no_context | overall_quality | 0.1704 | 0.7968 |
| controlled_vs_baseline_no_context | context_relevance | 0.2529 | 7.9822 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1132 | 0.5622 |
| controlled_vs_baseline_no_context | naturalness | 0.0297 | 0.0342 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3268 | 11.0461 |
| controlled_vs_baseline_no_context | context_overlap | 0.0806 | 2.2021 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1343 | 1.1812 |
| controlled_vs_baseline_no_context | persona_style | 0.0290 | 0.0525 |
| controlled_vs_baseline_no_context | distinct1 | -0.0401 | -0.0412 |
| controlled_vs_baseline_no_context | length_score | 0.1758 | 0.3689 |
| controlled_vs_baseline_no_context | sentence_score | 0.1225 | 0.1439 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0413 | 0.8005 |
| controlled_vs_baseline_no_context | overall_quality | 0.1496 | 0.6378 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2529 | 7.9822 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1132 | 0.5622 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 0.0297 | 0.0342 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3268 | 11.0461 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0806 | 2.2021 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1343 | 1.1812 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0290 | 0.0525 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0401 | -0.0412 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.1758 | 0.3689 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.1225 | 0.1439 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0413 | 0.8005 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1496 | 0.6378 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0498 | (0.0168, 0.0884) | 0.0007 | 0.0498 | (0.0057, 0.1126) | 0.0053 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0605 | (-0.1154, -0.0098) | 0.9927 | -0.0605 | (-0.1727, 0.0007) | 0.9710 |
| proposed_vs_candidate_no_context | naturalness | -0.0119 | (-0.0373, 0.0132) | 0.8253 | -0.0119 | (-0.0518, 0.0157) | 0.7477 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0612 | (0.0199, 0.1093) | 0.0013 | 0.0612 | (0.0065, 0.1404) | 0.0077 |
| proposed_vs_candidate_no_context | context_overlap | 0.0232 | (0.0071, 0.0466) | 0.0000 | 0.0232 | (0.0050, 0.0489) | 0.0000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0667 | (-0.1327, -0.0086) | 0.9893 | -0.0667 | (-0.1987, 0.0065) | 0.9457 |
| proposed_vs_candidate_no_context | persona_style | -0.0357 | (-0.0822, -0.0000) | 0.9807 | -0.0357 | (-0.0856, 0.0000) | 0.9757 |
| proposed_vs_candidate_no_context | distinct1 | -0.0018 | (-0.0147, 0.0109) | 0.6053 | -0.0018 | (-0.0185, 0.0136) | 0.5667 |
| proposed_vs_candidate_no_context | length_score | -0.0342 | (-0.1325, 0.0584) | 0.7607 | -0.0342 | (-0.1586, 0.0642) | 0.7077 |
| proposed_vs_candidate_no_context | sentence_score | -0.0437 | (-0.0962, 0.0087) | 0.9700 | -0.0437 | (-0.1207, 0.0152) | 0.9507 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0289 | (0.0130, 0.0468) | 0.0000 | 0.0289 | (0.0101, 0.0535) | 0.0003 |
| proposed_vs_candidate_no_context | overall_quality | 0.0015 | (-0.0245, 0.0268) | 0.4553 | 0.0015 | (-0.0436, 0.0379) | 0.4610 |
| proposed_vs_baseline_no_context | context_relevance | 0.0448 | (0.0100, 0.0883) | 0.0047 | 0.0448 | (-0.0055, 0.1117) | 0.0520 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0802 | (-0.1303, -0.0326) | 0.9997 | -0.0802 | (-0.1831, 0.0027) | 0.9683 |
| proposed_vs_baseline_no_context | naturalness | -0.0758 | (-0.1076, -0.0440) | 1.0000 | -0.0758 | (-0.1162, -0.0211) | 0.9930 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0523 | (0.0058, 0.1048) | 0.0117 | 0.0523 | (-0.0149, 0.1381) | 0.0803 |
| proposed_vs_baseline_no_context | context_overlap | 0.0273 | (0.0097, 0.0506) | 0.0010 | 0.0273 | (0.0031, 0.0607) | 0.0077 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0748 | (-0.1348, -0.0231) | 0.9983 | -0.0748 | (-0.1979, 0.0121) | 0.9447 |
| proposed_vs_baseline_no_context | persona_style | -0.1021 | (-0.1874, -0.0264) | 0.9980 | -0.1021 | (-0.3093, 0.0182) | 0.8463 |
| proposed_vs_baseline_no_context | distinct1 | -0.0420 | (-0.0542, -0.0287) | 1.0000 | -0.0420 | (-0.0570, -0.0244) | 1.0000 |
| proposed_vs_baseline_no_context | length_score | -0.2250 | (-0.3509, -0.1000) | 0.9990 | -0.2250 | (-0.3733, -0.0211) | 0.9817 |
| proposed_vs_baseline_no_context | sentence_score | -0.1400 | (-0.2100, -0.0700) | 1.0000 | -0.1400 | (-0.2261, -0.0350) | 0.9983 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0180 | (-0.0037, 0.0392) | 0.0517 | 0.0180 | (-0.0067, 0.0486) | 0.0767 |
| proposed_vs_baseline_no_context | overall_quality | -0.0193 | (-0.0490, 0.0120) | 0.8917 | -0.0193 | (-0.0694, 0.0380) | 0.7617 |
| controlled_vs_proposed_raw | context_relevance | 0.2081 | (0.1767, 0.2394) | 0.0000 | 0.2081 | (0.1823, 0.2252) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.1935 | (0.1457, 0.2474) | 0.0000 | 0.1935 | (0.1234, 0.2937) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.1055 | (0.0750, 0.1362) | 0.0000 | 0.1055 | (0.0582, 0.1515) | 0.0000 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2745 | (0.2330, 0.3174) | 0.0000 | 0.2745 | (0.2396, 0.2980) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0533 | (0.0418, 0.0646) | 0.0000 | 0.0533 | (0.0413, 0.0659) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2090 | (0.1545, 0.2706) | 0.0000 | 0.2090 | (0.1381, 0.3162) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.1311 | (0.0625, 0.2076) | 0.0000 | 0.1311 | (0.0125, 0.3321) | 0.0040 |
| controlled_vs_proposed_raw | distinct1 | 0.0019 | (-0.0132, 0.0147) | 0.3747 | 0.0019 | (-0.0090, 0.0144) | 0.4107 |
| controlled_vs_proposed_raw | length_score | 0.4008 | (0.2783, 0.5225) | 0.0000 | 0.4008 | (0.2074, 0.5917) | 0.0000 |
| controlled_vs_proposed_raw | sentence_score | 0.2625 | (0.2185, 0.3063) | 0.0000 | 0.2625 | (0.2019, 0.3120) | 0.0000 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0233 | (0.0068, 0.0392) | 0.0047 | 0.0233 | (0.0002, 0.0506) | 0.0240 |
| controlled_vs_proposed_raw | overall_quality | 0.1689 | (0.1439, 0.1939) | 0.0000 | 0.1689 | (0.1404, 0.2032) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2579 | (0.2245, 0.2921) | 0.0000 | 0.2579 | (0.2094, 0.3214) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1330 | (0.0793, 0.1865) | 0.0000 | 0.1330 | (0.0673, 0.2043) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.0936 | (0.0524, 0.1281) | 0.0000 | 0.0936 | (0.0264, 0.1405) | 0.0053 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3357 | (0.2947, 0.3805) | 0.0000 | 0.3357 | (0.2727, 0.4137) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0764 | (0.0617, 0.0966) | 0.0000 | 0.0764 | (0.0577, 0.0988) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1424 | (0.0813, 0.2025) | 0.0000 | 0.1424 | (0.0784, 0.2138) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.0953 | (0.0294, 0.1702) | 0.0017 | 0.0953 | (-0.0077, 0.2531) | 0.1023 |
| controlled_vs_candidate_no_context | distinct1 | 0.0001 | (-0.0187, 0.0159) | 0.4763 | 0.0001 | (-0.0192, 0.0158) | 0.4810 |
| controlled_vs_candidate_no_context | length_score | 0.3667 | (0.2142, 0.5009) | 0.0000 | 0.3667 | (0.1252, 0.5511) | 0.0033 |
| controlled_vs_candidate_no_context | sentence_score | 0.2188 | (0.1575, 0.2712) | 0.0000 | 0.2188 | (0.1166, 0.2988) | 0.0000 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0522 | (0.0316, 0.0735) | 0.0000 | 0.0522 | (0.0249, 0.0833) | 0.0000 |
| controlled_vs_candidate_no_context | overall_quality | 0.1704 | (0.1464, 0.1933) | 0.0000 | 0.1704 | (0.1374, 0.2008) | 0.0000 |
| controlled_vs_baseline_no_context | context_relevance | 0.2529 | (0.2205, 0.2874) | 0.0000 | 0.2529 | (0.2074, 0.3187) | 0.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1132 | (0.0692, 0.1588) | 0.0000 | 0.1132 | (0.0520, 0.1792) | 0.0000 |
| controlled_vs_baseline_no_context | naturalness | 0.0297 | (-0.0033, 0.0616) | 0.0380 | 0.0297 | (-0.0068, 0.0713) | 0.0593 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3268 | (0.2826, 0.3705) | 0.0000 | 0.3268 | (0.2685, 0.4074) | 0.0000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0806 | (0.0658, 0.1014) | 0.0000 | 0.0806 | (0.0627, 0.1051) | 0.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1343 | (0.0786, 0.1900) | 0.0000 | 0.1343 | (0.0539, 0.2188) | 0.0000 |
| controlled_vs_baseline_no_context | persona_style | 0.0290 | (-0.0019, 0.0639) | 0.0337 | 0.0290 | (0.0029, 0.0782) | 0.0207 |
| controlled_vs_baseline_no_context | distinct1 | -0.0401 | (-0.0559, -0.0256) | 1.0000 | -0.0401 | (-0.0563, -0.0228) | 1.0000 |
| controlled_vs_baseline_no_context | length_score | 0.1758 | (0.0400, 0.3109) | 0.0087 | 0.1758 | (0.0012, 0.3615) | 0.0243 |
| controlled_vs_baseline_no_context | sentence_score | 0.1225 | (0.0700, 0.1750) | 0.0000 | 0.1225 | (0.0700, 0.1803) | 0.0000 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0413 | (0.0239, 0.0591) | 0.0000 | 0.0413 | (0.0259, 0.0649) | 0.0000 |
| controlled_vs_baseline_no_context | overall_quality | 0.1496 | (0.1282, 0.1719) | 0.0000 | 0.1496 | (0.1143, 0.1872) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2529 | (0.2196, 0.2886) | 0.0000 | 0.2529 | (0.2075, 0.3176) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1132 | (0.0666, 0.1586) | 0.0000 | 0.1132 | (0.0532, 0.1791) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 0.0297 | (-0.0049, 0.0628) | 0.0463 | 0.0297 | (-0.0068, 0.0734) | 0.0613 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3268 | (0.2849, 0.3739) | 0.0000 | 0.3268 | (0.2689, 0.4077) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0806 | (0.0651, 0.1022) | 0.0000 | 0.0806 | (0.0621, 0.1038) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1343 | (0.0796, 0.1886) | 0.0000 | 0.1343 | (0.0553, 0.2161) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0290 | (-0.0010, 0.0648) | 0.0323 | 0.0290 | (0.0029, 0.0792) | 0.0217 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0401 | (-0.0559, -0.0254) | 1.0000 | -0.0401 | (-0.0562, -0.0233) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.1758 | (0.0383, 0.3109) | 0.0063 | 0.1758 | (0.0024, 0.3602) | 0.0233 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.1225 | (0.0700, 0.1750) | 0.0000 | 0.1225 | (0.0662, 0.1842) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0413 | (0.0240, 0.0605) | 0.0000 | 0.0413 | (0.0263, 0.0656) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1496 | (0.1272, 0.1719) | 0.0000 | 0.1496 | (0.1141, 0.1882) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 13 | 7 | 20 | 0.5750 | 0.6500 |
| proposed_vs_candidate_no_context | persona_consistency | 4 | 10 | 26 | 0.4250 | 0.2857 |
| proposed_vs_candidate_no_context | naturalness | 8 | 12 | 20 | 0.4500 | 0.4000 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 10 | 4 | 26 | 0.5750 | 0.7143 |
| proposed_vs_candidate_no_context | context_overlap | 14 | 6 | 20 | 0.6000 | 0.7000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 4 | 8 | 28 | 0.4500 | 0.3333 |
| proposed_vs_candidate_no_context | persona_style | 1 | 4 | 35 | 0.4625 | 0.2000 |
| proposed_vs_candidate_no_context | distinct1 | 11 | 8 | 21 | 0.5375 | 0.5789 |
| proposed_vs_candidate_no_context | length_score | 10 | 10 | 20 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | sentence_score | 2 | 7 | 31 | 0.4375 | 0.2222 |
| proposed_vs_candidate_no_context | bertscore_f1 | 16 | 5 | 19 | 0.6375 | 0.7619 |
| proposed_vs_candidate_no_context | overall_quality | 10 | 11 | 19 | 0.4875 | 0.4762 |
| proposed_vs_baseline_no_context | context_relevance | 19 | 21 | 0 | 0.4750 | 0.4750 |
| proposed_vs_baseline_no_context | persona_consistency | 5 | 18 | 17 | 0.3375 | 0.2174 |
| proposed_vs_baseline_no_context | naturalness | 10 | 30 | 0 | 0.2500 | 0.2500 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 10 | 8 | 22 | 0.5250 | 0.5556 |
| proposed_vs_baseline_no_context | context_overlap | 24 | 16 | 0 | 0.6000 | 0.6000 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 3 | 12 | 25 | 0.3875 | 0.2000 |
| proposed_vs_baseline_no_context | persona_style | 3 | 9 | 28 | 0.4250 | 0.2500 |
| proposed_vs_baseline_no_context | distinct1 | 4 | 33 | 3 | 0.1375 | 0.1081 |
| proposed_vs_baseline_no_context | length_score | 13 | 27 | 0 | 0.3250 | 0.3250 |
| proposed_vs_baseline_no_context | sentence_score | 3 | 19 | 18 | 0.3000 | 0.1364 |
| proposed_vs_baseline_no_context | bertscore_f1 | 21 | 19 | 0 | 0.5250 | 0.5250 |
| proposed_vs_baseline_no_context | overall_quality | 13 | 27 | 0 | 0.3250 | 0.3250 |
| controlled_vs_proposed_raw | context_relevance | 38 | 2 | 0 | 0.9500 | 0.9500 |
| controlled_vs_proposed_raw | persona_consistency | 31 | 0 | 9 | 0.8875 | 1.0000 |
| controlled_vs_proposed_raw | naturalness | 33 | 7 | 0 | 0.8250 | 0.8250 |
| controlled_vs_proposed_raw | context_keyword_coverage | 37 | 1 | 2 | 0.9500 | 0.9737 |
| controlled_vs_proposed_raw | context_overlap | 36 | 4 | 0 | 0.9000 | 0.9000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 31 | 0 | 9 | 0.8875 | 1.0000 |
| controlled_vs_proposed_raw | persona_style | 12 | 0 | 28 | 0.6500 | 1.0000 |
| controlled_vs_proposed_raw | distinct1 | 24 | 14 | 2 | 0.6250 | 0.6316 |
| controlled_vs_proposed_raw | length_score | 28 | 9 | 3 | 0.7375 | 0.7568 |
| controlled_vs_proposed_raw | sentence_score | 30 | 0 | 10 | 0.8750 | 1.0000 |
| controlled_vs_proposed_raw | bertscore_f1 | 26 | 14 | 0 | 0.6500 | 0.6500 |
| controlled_vs_proposed_raw | overall_quality | 38 | 2 | 0 | 0.9500 | 0.9500 |
| controlled_vs_candidate_no_context | context_relevance | 40 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 27 | 5 | 8 | 0.7750 | 0.8438 |
| controlled_vs_candidate_no_context | naturalness | 31 | 9 | 0 | 0.7750 | 0.7750 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 39 | 0 | 1 | 0.9875 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 40 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 27 | 5 | 8 | 0.7750 | 0.8438 |
| controlled_vs_candidate_no_context | persona_style | 10 | 2 | 28 | 0.6000 | 0.8333 |
| controlled_vs_candidate_no_context | distinct1 | 23 | 16 | 1 | 0.5875 | 0.5897 |
| controlled_vs_candidate_no_context | length_score | 31 | 8 | 1 | 0.7875 | 0.7949 |
| controlled_vs_candidate_no_context | sentence_score | 26 | 1 | 13 | 0.8125 | 0.9630 |
| controlled_vs_candidate_no_context | bertscore_f1 | 34 | 6 | 0 | 0.8500 | 0.8500 |
| controlled_vs_candidate_no_context | overall_quality | 39 | 1 | 0 | 0.9750 | 0.9750 |
| controlled_vs_baseline_no_context | context_relevance | 40 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 27 | 4 | 9 | 0.7875 | 0.8710 |
| controlled_vs_baseline_no_context | naturalness | 28 | 11 | 1 | 0.7125 | 0.7179 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 40 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | context_overlap | 40 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 25 | 4 | 11 | 0.7625 | 0.8621 |
| controlled_vs_baseline_no_context | persona_style | 6 | 2 | 32 | 0.5500 | 0.7500 |
| controlled_vs_baseline_no_context | distinct1 | 7 | 30 | 3 | 0.2125 | 0.1892 |
| controlled_vs_baseline_no_context | length_score | 27 | 11 | 2 | 0.7000 | 0.7105 |
| controlled_vs_baseline_no_context | sentence_score | 14 | 0 | 26 | 0.6750 | 1.0000 |
| controlled_vs_baseline_no_context | bertscore_f1 | 30 | 10 | 0 | 0.7500 | 0.7500 |
| controlled_vs_baseline_no_context | overall_quality | 40 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 40 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 27 | 4 | 9 | 0.7875 | 0.8710 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 28 | 11 | 1 | 0.7125 | 0.7179 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 40 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 40 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 25 | 4 | 11 | 0.7625 | 0.8621 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 6 | 2 | 32 | 0.5500 | 0.7500 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | 7 | 30 | 3 | 0.2125 | 0.1892 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 27 | 11 | 2 | 0.7000 | 0.7105 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 14 | 0 | 26 | 0.6750 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 30 | 10 | 0 | 0.7500 | 0.7500 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 40 | 0 | 0 | 1.0000 | 1.0000 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.5250 | 0.3250 | 0.6750 |
| proposed_contextual | 0.0000 | 0.0000 | 0.6250 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.6000 | 0.0000 | 0.0000 |
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
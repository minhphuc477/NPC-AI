# Proposal Alignment Evaluation Report

- Run ID: `20260304T104722Z`
- Generated: `2026-03-04T11:00:30.551905+00:00`
- Scenarios: `artifacts\proposal_control_tuning\conservative\20260304T104722Z\scenarios.jsonl`
- Scenario count: `40`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off
- `baseline_no_context`: model `phi3:mini`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2570 (0.2420, 0.2731) | 0.3882 (0.3367, 0.4437) | 0.9023 (0.8826, 0.9202) | 0.3959 (0.3788, 0.4161) | 0.0750 |
| proposed_contextual | 0.0756 (0.0457, 0.1099) | 0.1298 (0.0953, 0.1730) | 0.7968 (0.7785, 0.8188) | 0.2178 (0.1956, 0.2431) | 0.0629 |
| candidate_no_context | 0.0167 (0.0115, 0.0227) | 0.1642 (0.1191, 0.2150) | 0.7986 (0.7767, 0.8239) | 0.2023 (0.1845, 0.2224) | 0.0369 |
| baseline_no_context | 0.0385 (0.0267, 0.0524) | 0.1960 (0.1542, 0.2450) | 0.9016 (0.8807, 0.9217) | 0.2406 (0.2239, 0.2573) | 0.0462 |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0590 | 3.5407 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0344 | -0.2097 |
| proposed_vs_candidate_no_context | naturalness | -0.0017 | -0.0022 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0784 | 11.5000 |
| proposed_vs_candidate_no_context | context_overlap | 0.0137 | 0.3451 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0399 | -0.4653 |
| proposed_vs_candidate_no_context | persona_style | -0.0127 | -0.0265 |
| proposed_vs_candidate_no_context | distinct1 | 0.0052 | 0.0055 |
| proposed_vs_candidate_no_context | length_score | -0.0050 | -0.0199 |
| proposed_vs_candidate_no_context | sentence_score | -0.0262 | -0.0348 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0260 | 0.7061 |
| proposed_vs_candidate_no_context | overall_quality | 0.0155 | 0.0766 |
| proposed_vs_baseline_no_context | context_relevance | 0.0371 | 0.9633 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0662 | -0.3378 |
| proposed_vs_baseline_no_context | naturalness | -0.1047 | -0.1162 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0467 | 1.2146 |
| proposed_vs_baseline_no_context | context_overlap | 0.0147 | 0.3794 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0575 | -0.5565 |
| proposed_vs_baseline_no_context | persona_style | -0.1010 | -0.1783 |
| proposed_vs_baseline_no_context | distinct1 | -0.0397 | -0.0406 |
| proposed_vs_baseline_no_context | length_score | -0.3517 | -0.5877 |
| proposed_vs_baseline_no_context | sentence_score | -0.1837 | -0.2014 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0167 | 0.3608 |
| proposed_vs_baseline_no_context | overall_quality | -0.0227 | -0.0945 |
| controlled_vs_proposed_raw | context_relevance | 0.1813 | 2.3970 |
| controlled_vs_proposed_raw | persona_consistency | 0.2584 | 1.9911 |
| controlled_vs_proposed_raw | naturalness | 0.1055 | 0.1324 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2398 | 2.8142 |
| controlled_vs_proposed_raw | context_overlap | 0.0448 | 0.8405 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2863 | 6.2468 |
| controlled_vs_proposed_raw | persona_style | 0.1469 | 0.3154 |
| controlled_vs_proposed_raw | distinct1 | -0.0045 | -0.0048 |
| controlled_vs_proposed_raw | length_score | 0.4217 | 1.7095 |
| controlled_vs_proposed_raw | sentence_score | 0.2275 | 0.3122 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0120 | 0.1914 |
| controlled_vs_proposed_raw | overall_quality | 0.1781 | 0.8176 |
| controlled_vs_candidate_no_context | context_relevance | 0.2403 | 14.4248 |
| controlled_vs_candidate_no_context | persona_consistency | 0.2240 | 1.3639 |
| controlled_vs_candidate_no_context | naturalness | 0.1037 | 0.1299 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3183 | 46.6778 |
| controlled_vs_candidate_no_context | context_overlap | 0.0585 | 1.4757 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2464 | 2.8750 |
| controlled_vs_candidate_no_context | persona_style | 0.1342 | 0.2806 |
| controlled_vs_candidate_no_context | distinct1 | 0.0007 | 0.0007 |
| controlled_vs_candidate_no_context | length_score | 0.4167 | 1.6556 |
| controlled_vs_candidate_no_context | sentence_score | 0.2013 | 0.2666 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0381 | 1.0326 |
| controlled_vs_candidate_no_context | overall_quality | 0.1936 | 0.9568 |
| controlled_vs_baseline_no_context | context_relevance | 0.2185 | 5.6695 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1922 | 0.9807 |
| controlled_vs_baseline_no_context | naturalness | 0.0007 | 0.0008 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.2866 | 7.4469 |
| controlled_vs_baseline_no_context | context_overlap | 0.0595 | 1.5389 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2288 | 2.2143 |
| controlled_vs_baseline_no_context | persona_style | 0.0458 | 0.0809 |
| controlled_vs_baseline_no_context | distinct1 | -0.0441 | -0.0452 |
| controlled_vs_baseline_no_context | length_score | 0.0700 | 0.1170 |
| controlled_vs_baseline_no_context | sentence_score | 0.0438 | 0.0479 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0287 | 0.6212 |
| controlled_vs_baseline_no_context | overall_quality | 0.1554 | 0.6459 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2185 | 5.6695 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1922 | 0.9807 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 0.0007 | 0.0008 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.2866 | 7.4469 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0595 | 1.5389 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2288 | 2.2143 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0458 | 0.0809 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0441 | -0.0452 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.0700 | 0.1170 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0438 | 0.0479 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0287 | 0.6212 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1554 | 0.6459 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0590 | (0.0281, 0.0953) | 0.0000 | 0.0590 | (0.0249, 0.1095) | 0.0000 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0344 | (-0.0861, 0.0127) | 0.9177 | -0.0344 | (-0.1223, 0.0247) | 0.8197 |
| proposed_vs_candidate_no_context | naturalness | -0.0017 | (-0.0283, 0.0231) | 0.5570 | -0.0017 | (-0.0371, 0.0230) | 0.5407 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0784 | (0.0375, 0.1254) | 0.0000 | 0.0784 | (0.0316, 0.1461) | 0.0000 |
| proposed_vs_candidate_no_context | context_overlap | 0.0137 | (0.0040, 0.0251) | 0.0013 | 0.0137 | (0.0024, 0.0294) | 0.0070 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0399 | (-0.1024, 0.0179) | 0.9130 | -0.0399 | (-0.1455, 0.0306) | 0.8303 |
| proposed_vs_candidate_no_context | persona_style | -0.0127 | (-0.0580, 0.0255) | 0.7077 | -0.0127 | (-0.0497, 0.0252) | 0.7510 |
| proposed_vs_candidate_no_context | distinct1 | 0.0052 | (-0.0090, 0.0196) | 0.2347 | 0.0052 | (-0.0098, 0.0185) | 0.2527 |
| proposed_vs_candidate_no_context | length_score | -0.0050 | (-0.0983, 0.0808) | 0.5523 | -0.0050 | (-0.1051, 0.0762) | 0.5420 |
| proposed_vs_candidate_no_context | sentence_score | -0.0262 | (-0.0875, 0.0350) | 0.8390 | -0.0262 | (-0.1030, 0.0298) | 0.8220 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0260 | (0.0036, 0.0515) | 0.0087 | 0.0260 | (0.0032, 0.0633) | 0.0083 |
| proposed_vs_candidate_no_context | overall_quality | 0.0155 | (-0.0122, 0.0427) | 0.1330 | 0.0155 | (-0.0251, 0.0534) | 0.2210 |
| proposed_vs_baseline_no_context | context_relevance | 0.0371 | (0.0036, 0.0741) | 0.0127 | 0.0371 | (-0.0038, 0.0903) | 0.0467 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0662 | (-0.1255, -0.0127) | 0.9927 | -0.0662 | (-0.1372, 0.0051) | 0.9657 |
| proposed_vs_baseline_no_context | naturalness | -0.1047 | (-0.1347, -0.0732) | 1.0000 | -0.1047 | (-0.1504, -0.0542) | 1.0000 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0467 | (0.0017, 0.0970) | 0.0210 | 0.0467 | (-0.0101, 0.1188) | 0.0713 |
| proposed_vs_baseline_no_context | context_overlap | 0.0147 | (0.0027, 0.0285) | 0.0060 | 0.0147 | (0.0016, 0.0332) | 0.0093 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0575 | (-0.1258, 0.0055) | 0.9627 | -0.0575 | (-0.1310, 0.0210) | 0.9260 |
| proposed_vs_baseline_no_context | persona_style | -0.1010 | (-0.1781, -0.0370) | 1.0000 | -0.1010 | (-0.2882, 0.0135) | 0.9130 |
| proposed_vs_baseline_no_context | distinct1 | -0.0397 | (-0.0558, -0.0225) | 1.0000 | -0.0397 | (-0.0547, -0.0217) | 1.0000 |
| proposed_vs_baseline_no_context | length_score | -0.3517 | (-0.4617, -0.2333) | 1.0000 | -0.3517 | (-0.5236, -0.1575) | 0.9997 |
| proposed_vs_baseline_no_context | sentence_score | -0.1837 | (-0.2450, -0.1225) | 1.0000 | -0.1837 | (-0.2603, -0.0921) | 1.0000 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0167 | (-0.0101, 0.0454) | 0.1213 | 0.0167 | (-0.0117, 0.0583) | 0.1450 |
| proposed_vs_baseline_no_context | overall_quality | -0.0227 | (-0.0515, 0.0075) | 0.9277 | -0.0227 | (-0.0567, 0.0227) | 0.8467 |
| controlled_vs_proposed_raw | context_relevance | 0.1813 | (0.1488, 0.2108) | 0.0000 | 0.1813 | (0.1449, 0.2108) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.2584 | (0.2000, 0.3217) | 0.0000 | 0.2584 | (0.1912, 0.3213) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.1055 | (0.0788, 0.1310) | 0.0000 | 0.1055 | (0.0782, 0.1318) | 0.0000 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2398 | (0.1961, 0.2800) | 0.0000 | 0.2398 | (0.1917, 0.2784) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0448 | (0.0327, 0.0555) | 0.0000 | 0.0448 | (0.0279, 0.0592) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2863 | (0.2214, 0.3571) | 0.0000 | 0.2863 | (0.2237, 0.3463) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.1469 | (0.0708, 0.2318) | 0.0000 | 0.1469 | (0.0000, 0.3357) | 0.0253 |
| controlled_vs_proposed_raw | distinct1 | -0.0045 | (-0.0193, 0.0097) | 0.7337 | -0.0045 | (-0.0213, 0.0104) | 0.7203 |
| controlled_vs_proposed_raw | length_score | 0.4217 | (0.3133, 0.5267) | 0.0000 | 0.4217 | (0.3252, 0.5238) | 0.0000 |
| controlled_vs_proposed_raw | sentence_score | 0.2275 | (0.1662, 0.2887) | 0.0000 | 0.2275 | (0.1709, 0.2800) | 0.0000 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0120 | (-0.0148, 0.0371) | 0.1850 | 0.0120 | (-0.0387, 0.0457) | 0.2717 |
| controlled_vs_proposed_raw | overall_quality | 0.1781 | (0.1488, 0.2077) | 0.0000 | 0.1781 | (0.1399, 0.2132) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2403 | (0.2235, 0.2581) | 0.0000 | 0.2403 | (0.2149, 0.2682) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.2240 | (0.1629, 0.2856) | 0.0000 | 0.2240 | (0.1328, 0.2977) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.1037 | (0.0702, 0.1339) | 0.0000 | 0.1037 | (0.0578, 0.1364) | 0.0000 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3183 | (0.2947, 0.3413) | 0.0000 | 0.3183 | (0.2816, 0.3588) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0585 | (0.0516, 0.0653) | 0.0000 | 0.0585 | (0.0528, 0.0664) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2464 | (0.1734, 0.3218) | 0.0000 | 0.2464 | (0.1397, 0.3375) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.1342 | (0.0619, 0.2209) | 0.0000 | 0.1342 | (0.0075, 0.2945) | 0.0240 |
| controlled_vs_candidate_no_context | distinct1 | 0.0007 | (-0.0112, 0.0115) | 0.4363 | 0.0007 | (-0.0140, 0.0118) | 0.4490 |
| controlled_vs_candidate_no_context | length_score | 0.4167 | (0.2900, 0.5350) | 0.0000 | 0.4167 | (0.2568, 0.5445) | 0.0000 |
| controlled_vs_candidate_no_context | sentence_score | 0.2012 | (0.1313, 0.2625) | 0.0000 | 0.2012 | (0.1016, 0.2833) | 0.0000 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0381 | (0.0159, 0.0588) | 0.0000 | 0.0381 | (0.0058, 0.0631) | 0.0113 |
| controlled_vs_candidate_no_context | overall_quality | 0.1936 | (0.1703, 0.2159) | 0.0000 | 0.1936 | (0.1522, 0.2229) | 0.0000 |
| controlled_vs_baseline_no_context | context_relevance | 0.2185 | (0.1961, 0.2416) | 0.0000 | 0.2185 | (0.1883, 0.2537) | 0.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1922 | (0.1306, 0.2489) | 0.0000 | 0.1922 | (0.1297, 0.2501) | 0.0000 |
| controlled_vs_baseline_no_context | naturalness | 0.0007 | (-0.0297, 0.0328) | 0.4607 | 0.0007 | (-0.0431, 0.0516) | 0.4897 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.2866 | (0.2553, 0.3178) | 0.0000 | 0.2866 | (0.2448, 0.3361) | 0.0000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0595 | (0.0519, 0.0668) | 0.0000 | 0.0595 | (0.0521, 0.0693) | 0.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2288 | (0.1518, 0.3000) | 0.0000 | 0.2288 | (0.1547, 0.2931) | 0.0000 |
| controlled_vs_baseline_no_context | persona_style | 0.0458 | (0.0042, 0.0917) | 0.0140 | 0.0458 | (0.0000, 0.1250) | 0.1050 |
| controlled_vs_baseline_no_context | distinct1 | -0.0441 | (-0.0559, -0.0316) | 1.0000 | -0.0441 | (-0.0589, -0.0262) | 1.0000 |
| controlled_vs_baseline_no_context | length_score | 0.0700 | (-0.0642, 0.1967) | 0.1500 | 0.0700 | (-0.1125, 0.2889) | 0.2440 |
| controlled_vs_baseline_no_context | sentence_score | 0.0437 | (-0.0262, 0.1137) | 0.1187 | 0.0437 | (-0.0333, 0.1346) | 0.1697 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0287 | (0.0120, 0.0459) | 0.0000 | 0.0287 | (0.0078, 0.0491) | 0.0057 |
| controlled_vs_baseline_no_context | overall_quality | 0.1554 | (0.1312, 0.1781) | 0.0000 | 0.1554 | (0.1276, 0.1820) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2185 | (0.1962, 0.2407) | 0.0000 | 0.2185 | (0.1894, 0.2520) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1922 | (0.1283, 0.2500) | 0.0000 | 0.1922 | (0.1318, 0.2490) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 0.0007 | (-0.0319, 0.0297) | 0.4717 | 0.0007 | (-0.0411, 0.0517) | 0.4843 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.2866 | (0.2548, 0.3186) | 0.0000 | 0.2866 | (0.2437, 0.3366) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0595 | (0.0522, 0.0667) | 0.0000 | 0.0595 | (0.0519, 0.0697) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2288 | (0.1518, 0.2973) | 0.0000 | 0.2288 | (0.1537, 0.2926) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0458 | (0.0042, 0.0958) | 0.0147 | 0.0458 | (0.0000, 0.1250) | 0.0987 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0441 | (-0.0561, -0.0318) | 1.0000 | -0.0441 | (-0.0594, -0.0256) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.0700 | (-0.0584, 0.2033) | 0.1527 | 0.0700 | (-0.1094, 0.2986) | 0.2370 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0437 | (-0.0262, 0.1137) | 0.1140 | 0.0437 | (-0.0318, 0.1458) | 0.1767 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0287 | (0.0117, 0.0464) | 0.0003 | 0.0287 | (0.0082, 0.0495) | 0.0053 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1554 | (0.1316, 0.1789) | 0.0000 | 0.1554 | (0.1269, 0.1831) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 16 | 7 | 17 | 0.6125 | 0.6957 |
| proposed_vs_candidate_no_context | persona_consistency | 4 | 7 | 29 | 0.4625 | 0.3636 |
| proposed_vs_candidate_no_context | naturalness | 12 | 11 | 17 | 0.5125 | 0.5217 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 13 | 1 | 26 | 0.6500 | 0.9286 |
| proposed_vs_candidate_no_context | context_overlap | 16 | 7 | 17 | 0.6125 | 0.6957 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 3 | 7 | 30 | 0.4500 | 0.3000 |
| proposed_vs_candidate_no_context | persona_style | 3 | 3 | 34 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | distinct1 | 11 | 8 | 21 | 0.5375 | 0.5789 |
| proposed_vs_candidate_no_context | length_score | 11 | 12 | 17 | 0.4875 | 0.4783 |
| proposed_vs_candidate_no_context | sentence_score | 5 | 8 | 27 | 0.4625 | 0.3846 |
| proposed_vs_candidate_no_context | bertscore_f1 | 17 | 6 | 17 | 0.6375 | 0.7391 |
| proposed_vs_candidate_no_context | overall_quality | 13 | 10 | 17 | 0.5375 | 0.5652 |
| proposed_vs_baseline_no_context | context_relevance | 19 | 21 | 0 | 0.4750 | 0.4750 |
| proposed_vs_baseline_no_context | persona_consistency | 4 | 18 | 18 | 0.3250 | 0.1818 |
| proposed_vs_baseline_no_context | naturalness | 7 | 33 | 0 | 0.1750 | 0.1750 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 13 | 10 | 17 | 0.5375 | 0.5652 |
| proposed_vs_baseline_no_context | context_overlap | 25 | 15 | 0 | 0.6250 | 0.6250 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 3 | 12 | 25 | 0.3875 | 0.2000 |
| proposed_vs_baseline_no_context | persona_style | 1 | 9 | 30 | 0.4000 | 0.1000 |
| proposed_vs_baseline_no_context | distinct1 | 5 | 29 | 6 | 0.2000 | 0.1471 |
| proposed_vs_baseline_no_context | length_score | 8 | 31 | 1 | 0.2125 | 0.2051 |
| proposed_vs_baseline_no_context | sentence_score | 1 | 22 | 17 | 0.2375 | 0.0435 |
| proposed_vs_baseline_no_context | bertscore_f1 | 23 | 17 | 0 | 0.5750 | 0.5750 |
| proposed_vs_baseline_no_context | overall_quality | 13 | 27 | 0 | 0.3250 | 0.3250 |
| controlled_vs_proposed_raw | context_relevance | 37 | 3 | 0 | 0.9250 | 0.9250 |
| controlled_vs_proposed_raw | persona_consistency | 37 | 1 | 2 | 0.9500 | 0.9737 |
| controlled_vs_proposed_raw | naturalness | 36 | 4 | 0 | 0.9000 | 0.9000 |
| controlled_vs_proposed_raw | context_keyword_coverage | 36 | 3 | 1 | 0.9125 | 0.9231 |
| controlled_vs_proposed_raw | context_overlap | 38 | 2 | 0 | 0.9500 | 0.9500 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 37 | 1 | 2 | 0.9500 | 0.9737 |
| controlled_vs_proposed_raw | persona_style | 11 | 1 | 28 | 0.6250 | 0.9167 |
| controlled_vs_proposed_raw | distinct1 | 22 | 18 | 0 | 0.5500 | 0.5500 |
| controlled_vs_proposed_raw | length_score | 34 | 5 | 1 | 0.8625 | 0.8718 |
| controlled_vs_proposed_raw | sentence_score | 28 | 2 | 10 | 0.8250 | 0.9333 |
| controlled_vs_proposed_raw | bertscore_f1 | 30 | 10 | 0 | 0.7500 | 0.7500 |
| controlled_vs_proposed_raw | overall_quality | 39 | 1 | 0 | 0.9750 | 0.9750 |
| controlled_vs_candidate_no_context | context_relevance | 40 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 35 | 4 | 1 | 0.8875 | 0.8974 |
| controlled_vs_candidate_no_context | naturalness | 35 | 5 | 0 | 0.8750 | 0.8750 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 40 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 40 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 34 | 4 | 2 | 0.8750 | 0.8947 |
| controlled_vs_candidate_no_context | persona_style | 11 | 1 | 28 | 0.6250 | 0.9167 |
| controlled_vs_candidate_no_context | distinct1 | 27 | 12 | 1 | 0.6875 | 0.6923 |
| controlled_vs_candidate_no_context | length_score | 33 | 6 | 1 | 0.8375 | 0.8462 |
| controlled_vs_candidate_no_context | sentence_score | 26 | 3 | 11 | 0.7875 | 0.8966 |
| controlled_vs_candidate_no_context | bertscore_f1 | 33 | 7 | 0 | 0.8250 | 0.8250 |
| controlled_vs_candidate_no_context | overall_quality | 40 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | context_relevance | 40 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 33 | 1 | 6 | 0.9000 | 0.9706 |
| controlled_vs_baseline_no_context | naturalness | 21 | 19 | 0 | 0.5250 | 0.5250 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 40 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | context_overlap | 40 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 32 | 1 | 7 | 0.8875 | 0.9697 |
| controlled_vs_baseline_no_context | persona_style | 5 | 1 | 34 | 0.5500 | 0.8333 |
| controlled_vs_baseline_no_context | distinct1 | 6 | 34 | 0 | 0.1500 | 0.1500 |
| controlled_vs_baseline_no_context | length_score | 25 | 15 | 0 | 0.6250 | 0.6250 |
| controlled_vs_baseline_no_context | sentence_score | 10 | 5 | 25 | 0.5625 | 0.6667 |
| controlled_vs_baseline_no_context | bertscore_f1 | 29 | 11 | 0 | 0.7250 | 0.7250 |
| controlled_vs_baseline_no_context | overall_quality | 39 | 1 | 0 | 0.9750 | 0.9750 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 40 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 33 | 1 | 6 | 0.9000 | 0.9706 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 21 | 19 | 0 | 0.5250 | 0.5250 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 40 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 40 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 32 | 1 | 7 | 0.8875 | 0.9697 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 5 | 1 | 34 | 0.5500 | 0.8333 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | 6 | 34 | 0 | 0.1500 | 0.1500 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 25 | 15 | 0 | 0.6250 | 0.6250 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 10 | 5 | 25 | 0.5625 | 0.6667 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 29 | 11 | 0 | 0.7250 | 0.7250 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 39 | 1 | 0 | 0.9750 | 0.9750 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.7000 | 0.0500 | 0.9500 |
| proposed_contextual | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.6500 | 0.0000 | 0.0000 |
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
# Proposal Alignment Evaluation Report

- Run ID: `20260304T111002Z`
- Generated: `2026-03-04T11:18:35.921942+00:00`
- Scenarios: `artifacts\proposal_control_tuning\balanced\20260304T111002Z\scenarios.jsonl`
- Scenario count: `40`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off
- `baseline_no_context`: model `phi3:mini`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2515 (0.2320, 0.2725) | 0.3613 (0.3198, 0.4033) | 0.9130 (0.8948, 0.9290) | 0.3882 (0.3711, 0.4080) | 0.0866 |
| proposed_contextual | 0.0878 (0.0488, 0.1304) | 0.1344 (0.1032, 0.1680) | 0.7928 (0.7717, 0.8156) | 0.2231 (0.1981, 0.2509) | 0.0629 |
| candidate_no_context | 0.0263 (0.0167, 0.0376) | 0.1608 (0.1189, 0.2095) | 0.7903 (0.7683, 0.8142) | 0.2041 (0.1864, 0.2245) | 0.0401 |
| baseline_no_context | 0.0421 (0.0282, 0.0582) | 0.1735 (0.1436, 0.2045) | 0.8912 (0.8726, 0.9111) | 0.2345 (0.2220, 0.2475) | 0.0620 |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0615 | 2.3340 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0264 | -0.1641 |
| proposed_vs_candidate_no_context | naturalness | 0.0025 | 0.0032 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0771 | 3.7704 |
| proposed_vs_candidate_no_context | context_overlap | 0.0250 | 0.6238 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0337 | -0.4131 |
| proposed_vs_candidate_no_context | persona_style | 0.0028 | 0.0059 |
| proposed_vs_candidate_no_context | distinct1 | 0.0005 | 0.0005 |
| proposed_vs_candidate_no_context | length_score | 0.0117 | 0.0513 |
| proposed_vs_candidate_no_context | sentence_score | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0228 | 0.5680 |
| proposed_vs_candidate_no_context | overall_quality | 0.0190 | 0.0931 |
| proposed_vs_baseline_no_context | context_relevance | 0.0457 | 1.0840 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0391 | -0.2254 |
| proposed_vs_baseline_no_context | naturalness | -0.0984 | -0.1104 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0545 | 1.2676 |
| proposed_vs_baseline_no_context | context_overlap | 0.0250 | 0.6240 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0262 | -0.3537 |
| proposed_vs_baseline_no_context | persona_style | -0.0908 | -0.1589 |
| proposed_vs_baseline_no_context | distinct1 | -0.0459 | -0.0471 |
| proposed_vs_baseline_no_context | length_score | -0.3258 | -0.5767 |
| proposed_vs_baseline_no_context | sentence_score | -0.1488 | -0.1678 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0009 | 0.0144 |
| proposed_vs_baseline_no_context | overall_quality | -0.0114 | -0.0486 |
| controlled_vs_proposed_raw | context_relevance | 0.1636 | 1.8630 |
| controlled_vs_proposed_raw | persona_consistency | 0.2269 | 1.6887 |
| controlled_vs_proposed_raw | naturalness | 0.1201 | 0.1515 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2186 | 2.2403 |
| controlled_vs_proposed_raw | context_overlap | 0.0353 | 0.5430 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2569 | 5.3682 |
| controlled_vs_proposed_raw | persona_style | 0.1070 | 0.2228 |
| controlled_vs_proposed_raw | distinct1 | 0.0092 | 0.0100 |
| controlled_vs_proposed_raw | length_score | 0.4508 | 1.8850 |
| controlled_vs_proposed_raw | sentence_score | 0.2625 | 0.3559 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0237 | 0.3760 |
| controlled_vs_proposed_raw | overall_quality | 0.1652 | 0.7403 |
| controlled_vs_candidate_no_context | context_relevance | 0.2251 | 8.5452 |
| controlled_vs_candidate_no_context | persona_consistency | 0.2005 | 1.2474 |
| controlled_vs_candidate_no_context | naturalness | 0.1227 | 0.1552 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.2957 | 14.4574 |
| controlled_vs_candidate_no_context | context_overlap | 0.0603 | 1.5056 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2232 | 2.7372 |
| controlled_vs_candidate_no_context | persona_style | 0.1098 | 0.2300 |
| controlled_vs_candidate_no_context | distinct1 | 0.0098 | 0.0105 |
| controlled_vs_candidate_no_context | length_score | 0.4625 | 2.0330 |
| controlled_vs_candidate_no_context | sentence_score | 0.2625 | 0.3559 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0464 | 1.1576 |
| controlled_vs_candidate_no_context | overall_quality | 0.1841 | 0.9023 |
| controlled_vs_baseline_no_context | context_relevance | 0.2093 | 4.9664 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1878 | 1.0827 |
| controlled_vs_baseline_no_context | naturalness | 0.0217 | 0.0244 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.2731 | 6.3477 |
| controlled_vs_baseline_no_context | context_overlap | 0.0604 | 1.5059 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2307 | 3.1158 |
| controlled_vs_baseline_no_context | persona_style | 0.0163 | 0.0285 |
| controlled_vs_baseline_no_context | distinct1 | -0.0366 | -0.0376 |
| controlled_vs_baseline_no_context | length_score | 0.1250 | 0.2212 |
| controlled_vs_baseline_no_context | sentence_score | 0.1137 | 0.1283 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0246 | 0.3959 |
| controlled_vs_baseline_no_context | overall_quality | 0.1537 | 0.6557 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2093 | 4.9664 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1878 | 1.0827 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 0.0217 | 0.0244 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.2731 | 6.3477 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0604 | 1.5059 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2307 | 3.1158 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0163 | 0.0285 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0366 | -0.0376 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.1250 | 0.2212 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.1137 | 0.1283 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0246 | 0.3959 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1537 | 0.6557 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0615 | (0.0219, 0.1088) | 0.0007 | 0.0615 | (0.0064, 0.1349) | 0.0127 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0264 | (-0.0800, 0.0201) | 0.8480 | -0.0264 | (-0.1323, 0.0374) | 0.6860 |
| proposed_vs_candidate_no_context | naturalness | 0.0025 | (-0.0255, 0.0298) | 0.4350 | 0.0025 | (-0.0391, 0.0294) | 0.4303 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0771 | (0.0288, 0.1327) | 0.0010 | 0.0771 | (0.0083, 0.1696) | 0.0133 |
| proposed_vs_candidate_no_context | context_overlap | 0.0250 | (0.0047, 0.0511) | 0.0027 | 0.0250 | (0.0025, 0.0531) | 0.0127 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0337 | (-0.0980, 0.0239) | 0.8653 | -0.0337 | (-0.1611, 0.0422) | 0.7503 |
| proposed_vs_candidate_no_context | persona_style | 0.0028 | (-0.0229, 0.0272) | 0.4510 | 0.0028 | (-0.0349, 0.0393) | 0.4460 |
| proposed_vs_candidate_no_context | distinct1 | 0.0005 | (-0.0107, 0.0112) | 0.4613 | 0.0005 | (-0.0192, 0.0166) | 0.4717 |
| proposed_vs_candidate_no_context | length_score | 0.0117 | (-0.0983, 0.1183) | 0.4190 | 0.0117 | (-0.0991, 0.0800) | 0.3680 |
| proposed_vs_candidate_no_context | sentence_score | 0.0000 | (-0.0700, 0.0700) | 0.5487 | 0.0000 | (-0.1300, 0.1102) | 0.5250 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0228 | (0.0034, 0.0450) | 0.0113 | 0.0228 | (0.0031, 0.0469) | 0.0100 |
| proposed_vs_candidate_no_context | overall_quality | 0.0190 | (-0.0126, 0.0519) | 0.1183 | 0.0190 | (-0.0343, 0.0690) | 0.2450 |
| proposed_vs_baseline_no_context | context_relevance | 0.0457 | (0.0044, 0.0933) | 0.0157 | 0.0457 | (-0.0109, 0.1186) | 0.0730 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0391 | (-0.0894, 0.0082) | 0.9463 | -0.0391 | (-0.0888, 0.0164) | 0.9193 |
| proposed_vs_baseline_no_context | naturalness | -0.0984 | (-0.1279, -0.0690) | 1.0000 | -0.0984 | (-0.1279, -0.0579) | 1.0000 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0545 | (-0.0005, 0.1148) | 0.0273 | 0.0545 | (-0.0203, 0.1527) | 0.1050 |
| proposed_vs_baseline_no_context | context_overlap | 0.0250 | (0.0057, 0.0512) | 0.0030 | 0.0250 | (0.0053, 0.0499) | 0.0020 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0262 | (-0.0858, 0.0308) | 0.8113 | -0.0262 | (-0.0849, 0.0357) | 0.8153 |
| proposed_vs_baseline_no_context | persona_style | -0.0908 | (-0.1689, -0.0210) | 0.9960 | -0.0908 | (-0.2542, 0.0255) | 0.8770 |
| proposed_vs_baseline_no_context | distinct1 | -0.0459 | (-0.0582, -0.0328) | 1.0000 | -0.0459 | (-0.0602, -0.0291) | 1.0000 |
| proposed_vs_baseline_no_context | length_score | -0.3258 | (-0.4492, -0.2000) | 1.0000 | -0.3258 | (-0.4535, -0.1494) | 0.9997 |
| proposed_vs_baseline_no_context | sentence_score | -0.1487 | (-0.2012, -0.0962) | 1.0000 | -0.1487 | (-0.2118, -0.0721) | 1.0000 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0009 | (-0.0194, 0.0206) | 0.4693 | 0.0009 | (-0.0222, 0.0226) | 0.4573 |
| proposed_vs_baseline_no_context | overall_quality | -0.0114 | (-0.0435, 0.0224) | 0.7513 | -0.0114 | (-0.0486, 0.0395) | 0.6977 |
| controlled_vs_proposed_raw | context_relevance | 0.1636 | (0.1200, 0.2047) | 0.0000 | 0.1636 | (0.1015, 0.2166) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.2269 | (0.1818, 0.2760) | 0.0000 | 0.2269 | (0.1610, 0.3123) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.1201 | (0.0907, 0.1481) | 0.0000 | 0.1201 | (0.0905, 0.1430) | 0.0000 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2186 | (0.1624, 0.2729) | 0.0000 | 0.2186 | (0.1432, 0.2850) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0353 | (0.0102, 0.0557) | 0.0040 | 0.0353 | (0.0073, 0.0604) | 0.0080 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2569 | (0.2034, 0.3120) | 0.0000 | 0.2569 | (0.1869, 0.3607) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.1070 | (0.0411, 0.1815) | 0.0000 | 0.1070 | (0.0000, 0.2722) | 0.0253 |
| controlled_vs_proposed_raw | distinct1 | 0.0092 | (-0.0032, 0.0211) | 0.0690 | 0.0092 | (-0.0049, 0.0231) | 0.0980 |
| controlled_vs_proposed_raw | length_score | 0.4508 | (0.3258, 0.5742) | 0.0000 | 0.4508 | (0.3274, 0.5453) | 0.0000 |
| controlled_vs_proposed_raw | sentence_score | 0.2625 | (0.2100, 0.3063) | 0.0000 | 0.2625 | (0.1978, 0.3111) | 0.0000 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0237 | (-0.0020, 0.0471) | 0.0363 | 0.0237 | (-0.0123, 0.0583) | 0.0887 |
| controlled_vs_proposed_raw | overall_quality | 0.1652 | (0.1364, 0.1937) | 0.0000 | 0.1652 | (0.1173, 0.2135) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2251 | (0.2028, 0.2480) | 0.0000 | 0.2251 | (0.2032, 0.2481) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.2005 | (0.1564, 0.2454) | 0.0000 | 0.2005 | (0.1331, 0.2630) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.1227 | (0.0917, 0.1517) | 0.0000 | 0.1227 | (0.0691, 0.1573) | 0.0003 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.2957 | (0.2663, 0.3267) | 0.0000 | 0.2957 | (0.2671, 0.3269) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0603 | (0.0531, 0.0684) | 0.0000 | 0.0603 | (0.0497, 0.0749) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2232 | (0.1737, 0.2726) | 0.0000 | 0.2232 | (0.1524, 0.3002) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.1098 | (0.0422, 0.1862) | 0.0000 | 0.1098 | (0.0000, 0.2672) | 0.0267 |
| controlled_vs_candidate_no_context | distinct1 | 0.0098 | (-0.0032, 0.0221) | 0.0723 | 0.0098 | (-0.0040, 0.0201) | 0.0717 |
| controlled_vs_candidate_no_context | length_score | 0.4625 | (0.3316, 0.5825) | 0.0000 | 0.4625 | (0.2613, 0.6026) | 0.0000 |
| controlled_vs_candidate_no_context | sentence_score | 0.2625 | (0.2100, 0.3063) | 0.0000 | 0.2625 | (0.1647, 0.3306) | 0.0000 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0464 | (0.0311, 0.0633) | 0.0000 | 0.0464 | (0.0220, 0.0803) | 0.0000 |
| controlled_vs_candidate_no_context | overall_quality | 0.1841 | (0.1642, 0.2049) | 0.0000 | 0.1841 | (0.1564, 0.2063) | 0.0000 |
| controlled_vs_baseline_no_context | context_relevance | 0.2093 | (0.1848, 0.2350) | 0.0000 | 0.2093 | (0.1791, 0.2414) | 0.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1878 | (0.1438, 0.2352) | 0.0000 | 0.1878 | (0.1336, 0.2602) | 0.0000 |
| controlled_vs_baseline_no_context | naturalness | 0.0217 | (-0.0050, 0.0502) | 0.0567 | 0.0217 | (-0.0127, 0.0614) | 0.1243 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.2731 | (0.2418, 0.3076) | 0.0000 | 0.2731 | (0.2309, 0.3160) | 0.0000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0604 | (0.0510, 0.0696) | 0.0000 | 0.0604 | (0.0509, 0.0730) | 0.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2307 | (0.1780, 0.2866) | 0.0000 | 0.2307 | (0.1633, 0.3181) | 0.0000 |
| controlled_vs_baseline_no_context | persona_style | 0.0163 | (-0.0250, 0.0626) | 0.2430 | 0.0163 | (-0.0152, 0.0661) | 0.2140 |
| controlled_vs_baseline_no_context | distinct1 | -0.0366 | (-0.0460, -0.0270) | 1.0000 | -0.0366 | (-0.0465, -0.0256) | 1.0000 |
| controlled_vs_baseline_no_context | length_score | 0.1250 | (-0.0025, 0.2509) | 0.0263 | 0.1250 | (-0.0504, 0.3208) | 0.0913 |
| controlled_vs_baseline_no_context | sentence_score | 0.1137 | (0.0700, 0.1662) | 0.0000 | 0.1137 | (0.0512, 0.1909) | 0.0000 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0246 | (0.0070, 0.0423) | 0.0033 | 0.0246 | (-0.0024, 0.0489) | 0.0367 |
| controlled_vs_baseline_no_context | overall_quality | 0.1537 | (0.1339, 0.1741) | 0.0000 | 0.1537 | (0.1317, 0.1828) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2093 | (0.1859, 0.2356) | 0.0000 | 0.2093 | (0.1785, 0.2415) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1878 | (0.1439, 0.2343) | 0.0000 | 0.1878 | (0.1351, 0.2601) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 0.0217 | (-0.0068, 0.0485) | 0.0620 | 0.0217 | (-0.0117, 0.0633) | 0.1153 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.2731 | (0.2415, 0.3059) | 0.0000 | 0.2731 | (0.2301, 0.3163) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0604 | (0.0510, 0.0695) | 0.0000 | 0.0604 | (0.0511, 0.0734) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2307 | (0.1799, 0.2848) | 0.0000 | 0.2307 | (0.1668, 0.3136) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0163 | (-0.0233, 0.0638) | 0.2277 | 0.0163 | (-0.0155, 0.0635) | 0.2083 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0366 | (-0.0460, -0.0275) | 1.0000 | -0.0366 | (-0.0464, -0.0255) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.1250 | (-0.0050, 0.2492) | 0.0293 | 0.1250 | (-0.0476, 0.3200) | 0.0830 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.1137 | (0.0700, 0.1662) | 0.0000 | 0.1137 | (0.0525, 0.1956) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0246 | (0.0071, 0.0418) | 0.0023 | 0.0246 | (-0.0028, 0.0507) | 0.0377 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1537 | (0.1350, 0.1735) | 0.0000 | 0.1537 | (0.1313, 0.1840) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 14 | 8 | 18 | 0.5750 | 0.6364 |
| proposed_vs_candidate_no_context | persona_consistency | 8 | 8 | 24 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | naturalness | 11 | 11 | 18 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 11 | 4 | 25 | 0.5875 | 0.7333 |
| proposed_vs_candidate_no_context | context_overlap | 15 | 7 | 18 | 0.6000 | 0.6818 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 7 | 7 | 26 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | persona_style | 3 | 2 | 35 | 0.5125 | 0.6000 |
| proposed_vs_candidate_no_context | distinct1 | 9 | 10 | 21 | 0.4875 | 0.4737 |
| proposed_vs_candidate_no_context | length_score | 9 | 13 | 18 | 0.4500 | 0.4091 |
| proposed_vs_candidate_no_context | sentence_score | 8 | 8 | 24 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | bertscore_f1 | 19 | 13 | 8 | 0.5750 | 0.5938 |
| proposed_vs_candidate_no_context | overall_quality | 20 | 12 | 8 | 0.6000 | 0.6250 |
| proposed_vs_baseline_no_context | context_relevance | 19 | 21 | 0 | 0.4750 | 0.4750 |
| proposed_vs_baseline_no_context | persona_consistency | 8 | 16 | 16 | 0.4000 | 0.3333 |
| proposed_vs_baseline_no_context | naturalness | 6 | 34 | 0 | 0.1500 | 0.1500 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 11 | 11 | 18 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context | context_overlap | 24 | 15 | 1 | 0.6125 | 0.6154 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 7 | 10 | 23 | 0.4625 | 0.4118 |
| proposed_vs_baseline_no_context | persona_style | 1 | 8 | 31 | 0.4125 | 0.1111 |
| proposed_vs_baseline_no_context | distinct1 | 5 | 33 | 2 | 0.1500 | 0.1316 |
| proposed_vs_baseline_no_context | length_score | 9 | 30 | 1 | 0.2375 | 0.2308 |
| proposed_vs_baseline_no_context | sentence_score | 0 | 17 | 23 | 0.2875 | 0.0000 |
| proposed_vs_baseline_no_context | bertscore_f1 | 20 | 20 | 0 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context | overall_quality | 13 | 27 | 0 | 0.3250 | 0.3250 |
| controlled_vs_proposed_raw | context_relevance | 34 | 6 | 0 | 0.8500 | 0.8500 |
| controlled_vs_proposed_raw | persona_consistency | 37 | 2 | 1 | 0.9375 | 0.9487 |
| controlled_vs_proposed_raw | naturalness | 35 | 5 | 0 | 0.8750 | 0.8750 |
| controlled_vs_proposed_raw | context_keyword_coverage | 32 | 5 | 3 | 0.8375 | 0.8649 |
| controlled_vs_proposed_raw | context_overlap | 35 | 5 | 0 | 0.8750 | 0.8750 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 37 | 1 | 2 | 0.9500 | 0.9737 |
| controlled_vs_proposed_raw | persona_style | 9 | 1 | 30 | 0.6000 | 0.9000 |
| controlled_vs_proposed_raw | distinct1 | 25 | 14 | 1 | 0.6375 | 0.6410 |
| controlled_vs_proposed_raw | length_score | 33 | 5 | 2 | 0.8500 | 0.8684 |
| controlled_vs_proposed_raw | sentence_score | 30 | 0 | 10 | 0.8750 | 1.0000 |
| controlled_vs_proposed_raw | bertscore_f1 | 30 | 10 | 0 | 0.7500 | 0.7500 |
| controlled_vs_proposed_raw | overall_quality | 36 | 4 | 0 | 0.9000 | 0.9000 |
| controlled_vs_candidate_no_context | context_relevance | 40 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 37 | 2 | 1 | 0.9375 | 0.9487 |
| controlled_vs_candidate_no_context | naturalness | 33 | 7 | 0 | 0.8250 | 0.8250 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 40 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 40 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 37 | 2 | 1 | 0.9375 | 0.9487 |
| controlled_vs_candidate_no_context | persona_style | 10 | 1 | 29 | 0.6125 | 0.9091 |
| controlled_vs_candidate_no_context | distinct1 | 26 | 12 | 2 | 0.6750 | 0.6842 |
| controlled_vs_candidate_no_context | length_score | 32 | 8 | 0 | 0.8000 | 0.8000 |
| controlled_vs_candidate_no_context | sentence_score | 30 | 0 | 10 | 0.8750 | 1.0000 |
| controlled_vs_candidate_no_context | bertscore_f1 | 34 | 6 | 0 | 0.8500 | 0.8500 |
| controlled_vs_candidate_no_context | overall_quality | 40 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | context_relevance | 40 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 32 | 2 | 6 | 0.8750 | 0.9412 |
| controlled_vs_baseline_no_context | naturalness | 22 | 18 | 0 | 0.5500 | 0.5500 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 40 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | context_overlap | 40 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 32 | 0 | 8 | 0.9000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_style | 4 | 4 | 32 | 0.5000 | 0.5000 |
| controlled_vs_baseline_no_context | distinct1 | 5 | 35 | 0 | 0.1250 | 0.1250 |
| controlled_vs_baseline_no_context | length_score | 24 | 13 | 3 | 0.6375 | 0.6486 |
| controlled_vs_baseline_no_context | sentence_score | 13 | 0 | 27 | 0.6625 | 1.0000 |
| controlled_vs_baseline_no_context | bertscore_f1 | 25 | 15 | 0 | 0.6250 | 0.6250 |
| controlled_vs_baseline_no_context | overall_quality | 40 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 40 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 32 | 2 | 6 | 0.8750 | 0.9412 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 22 | 18 | 0 | 0.5500 | 0.5500 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 40 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 40 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 32 | 0 | 8 | 0.9000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 4 | 4 | 32 | 0.5000 | 0.5000 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | 5 | 35 | 0 | 0.1250 | 0.1250 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 24 | 13 | 3 | 0.6375 | 0.6486 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 13 | 0 | 27 | 0.6625 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 25 | 15 | 0 | 0.6250 | 0.6250 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 40 | 0 | 0 | 1.0000 | 1.0000 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.7000 | 0.1250 | 0.8750 |
| proposed_contextual | 0.0000 | 0.0000 | 0.5500 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.6750 | 0.0000 | 0.0000 |
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
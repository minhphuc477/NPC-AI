# Proposal Alignment Evaluation Report

- Run ID: `20260306T180248Z`
- Generated: `2026-03-06T18:03:39.855209+00:00`
- Scenarios: `artifacts\proposal_control_tuning\full112_postupgrade_nogate\20260306T180248Z\scenarios.jsonl`
- Scenario count: `112`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_alt`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off
- `baseline_no_context`: model `phi3:mini`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2633 (0.2504, 0.2775) | 0.3111 (0.2891, 0.3315) | 0.9071 (0.8955, 0.9181) | 0.3768 (0.3666, 0.3864) | 0.0999 |
| proposed_contextual_controlled_alt | 0.2615 (0.2442, 0.2808) | 0.3049 (0.2836, 0.3274) | 0.8970 (0.8845, 0.9097) | 0.3711 (0.3608, 0.3808) | 0.0915 |
| proposed_contextual | 0.0711 (0.0544, 0.0900) | 0.1448 (0.1209, 0.1699) | 0.8070 (0.7934, 0.8209) | 0.2229 (0.2093, 0.2385) | 0.0687 |
| candidate_no_context | 0.0238 (0.0180, 0.0303) | 0.1779 (0.1493, 0.2067) | 0.8103 (0.7972, 0.8249) | 0.2125 (0.2014, 0.2250) | 0.0450 |
| baseline_no_context | 0.0417 (0.0335, 0.0509) | 0.1984 (0.1764, 0.2204) | 0.8815 (0.8696, 0.8925) | 0.2398 (0.2313, 0.2492) | 0.0528 |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0473 | 1.9815 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0331 | -0.1860 |
| proposed_vs_candidate_no_context | naturalness | -0.0032 | -0.0040 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0605 | 3.6235 |
| proposed_vs_candidate_no_context | context_overlap | 0.0162 | 0.4012 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0400 | -0.4187 |
| proposed_vs_candidate_no_context | persona_style | -0.0056 | -0.0110 |
| proposed_vs_candidate_no_context | distinct1 | 0.0024 | 0.0025 |
| proposed_vs_candidate_no_context | length_score | -0.0128 | -0.0431 |
| proposed_vs_candidate_no_context | sentence_score | -0.0250 | -0.0320 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0237 | 0.5264 |
| proposed_vs_candidate_no_context | overall_quality | 0.0103 | 0.0487 |
| proposed_vs_baseline_no_context | context_relevance | 0.0294 | 0.7038 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0536 | -0.2704 |
| proposed_vs_baseline_no_context | naturalness | -0.0745 | -0.0845 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0347 | 0.8156 |
| proposed_vs_baseline_no_context | context_overlap | 0.0169 | 0.4251 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0511 | -0.4795 |
| proposed_vs_baseline_no_context | persona_style | -0.0638 | -0.1128 |
| proposed_vs_baseline_no_context | distinct1 | -0.0457 | -0.0465 |
| proposed_vs_baseline_no_context | length_score | -0.2310 | -0.4486 |
| proposed_vs_baseline_no_context | sentence_score | -0.1000 | -0.1168 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0159 | 0.3019 |
| proposed_vs_baseline_no_context | overall_quality | -0.0169 | -0.0706 |
| controlled_vs_proposed_raw | context_relevance | 0.1922 | 2.7037 |
| controlled_vs_proposed_raw | persona_consistency | 0.1663 | 1.1485 |
| controlled_vs_proposed_raw | naturalness | 0.1001 | 0.1240 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2534 | 3.2804 |
| controlled_vs_proposed_raw | context_overlap | 0.0495 | 0.8722 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1871 | 3.3716 |
| controlled_vs_proposed_raw | persona_style | 0.0831 | 0.1656 |
| controlled_vs_proposed_raw | distinct1 | 0.0005 | 0.0005 |
| controlled_vs_proposed_raw | length_score | 0.3949 | 1.3910 |
| controlled_vs_proposed_raw | sentence_score | 0.2094 | 0.2769 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0312 | 0.4540 |
| controlled_vs_proposed_raw | overall_quality | 0.1539 | 0.6907 |
| controlled_vs_candidate_no_context | context_relevance | 0.2395 | 10.0426 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1332 | 0.7488 |
| controlled_vs_candidate_no_context | naturalness | 0.0969 | 0.1195 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3139 | 18.7903 |
| controlled_vs_candidate_no_context | context_overlap | 0.0657 | 1.6233 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1471 | 1.5412 |
| controlled_vs_candidate_no_context | persona_style | 0.0775 | 0.1527 |
| controlled_vs_candidate_no_context | distinct1 | 0.0028 | 0.0030 |
| controlled_vs_candidate_no_context | length_score | 0.3821 | 1.2879 |
| controlled_vs_candidate_no_context | sentence_score | 0.1844 | 0.2360 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0549 | 1.2194 |
| controlled_vs_candidate_no_context | overall_quality | 0.1643 | 0.7730 |
| controlled_vs_baseline_no_context | context_relevance | 0.2216 | 5.3103 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1126 | 0.5676 |
| controlled_vs_baseline_no_context | naturalness | 0.0256 | 0.0291 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.2881 | 6.7714 |
| controlled_vs_baseline_no_context | context_overlap | 0.0664 | 1.6681 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1360 | 1.2756 |
| controlled_vs_baseline_no_context | persona_style | 0.0193 | 0.0341 |
| controlled_vs_baseline_no_context | distinct1 | -0.0452 | -0.0460 |
| controlled_vs_baseline_no_context | length_score | 0.1640 | 0.3185 |
| controlled_vs_baseline_no_context | sentence_score | 0.1094 | 0.1277 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0471 | 0.8930 |
| controlled_vs_baseline_no_context | overall_quality | 0.1370 | 0.5713 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0018 | -0.0069 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0062 | -0.0198 |
| controlled_alt_vs_controlled_default | naturalness | -0.0102 | -0.0112 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0019 | -0.0059 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0015 | -0.0143 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0101 | -0.0415 |
| controlled_alt_vs_controlled_default | persona_style | 0.0095 | 0.0163 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0004 | 0.0004 |
| controlled_alt_vs_controlled_default | length_score | -0.0357 | -0.0526 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0312 | -0.0324 |
| controlled_alt_vs_controlled_default | bertscore_f1 | -0.0084 | -0.0844 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0057 | -0.0151 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1904 | 2.6781 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1601 | 1.1060 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0899 | 0.1114 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2514 | 3.2552 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0480 | 0.8455 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1770 | 3.1900 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0926 | 0.1845 |
| controlled_alt_vs_proposed_raw | distinct1 | 0.0009 | 0.0009 |
| controlled_alt_vs_proposed_raw | length_score | 0.3592 | 1.2652 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1781 | 0.2355 |
| controlled_alt_vs_proposed_raw | bertscore_f1 | 0.0228 | 0.3313 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1482 | 0.6651 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2377 | 9.9663 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1270 | 0.7142 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0867 | 0.1070 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3120 | 18.6737 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0642 | 1.5858 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1370 | 1.4356 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0870 | 0.1715 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0032 | 0.0035 |
| controlled_alt_vs_candidate_no_context | length_score | 0.3464 | 1.1675 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1531 | 0.1960 |
| controlled_alt_vs_candidate_no_context | bertscore_f1 | 0.0465 | 1.0322 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1586 | 0.7462 |
| controlled_alt_vs_baseline_no_context | context_relevance | 0.2198 | 5.2667 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 0.1065 | 0.5366 |
| controlled_alt_vs_baseline_no_context | naturalness | 0.0155 | 0.0176 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 0.2861 | 6.7256 |
| controlled_alt_vs_baseline_no_context | context_overlap | 0.0649 | 1.6299 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 0.1259 | 1.1811 |
| controlled_alt_vs_baseline_no_context | persona_style | 0.0288 | 0.0509 |
| controlled_alt_vs_baseline_no_context | distinct1 | -0.0448 | -0.0456 |
| controlled_alt_vs_baseline_no_context | length_score | 0.1283 | 0.2491 |
| controlled_alt_vs_baseline_no_context | sentence_score | 0.0781 | 0.0912 |
| controlled_alt_vs_baseline_no_context | bertscore_f1 | 0.0387 | 0.7333 |
| controlled_alt_vs_baseline_no_context | overall_quality | 0.1313 | 0.5475 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2216 | 5.3103 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1126 | 0.5676 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 0.0256 | 0.0291 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.2881 | 6.7714 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0664 | 1.6681 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1360 | 1.2756 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0193 | 0.0341 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0452 | -0.0460 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.1640 | 0.3185 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.1094 | 0.1277 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0471 | 0.8930 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1370 | 0.5713 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0473 | (0.0303, 0.0653) | 0.0000 | 0.0473 | (0.0190, 0.0790) | 0.0000 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0331 | (-0.0610, -0.0051) | 0.9910 | -0.0331 | (-0.0896, 0.0087) | 0.9070 |
| proposed_vs_candidate_no_context | naturalness | -0.0032 | (-0.0208, 0.0150) | 0.6293 | -0.0032 | (-0.0338, 0.0202) | 0.5627 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0605 | (0.0399, 0.0845) | 0.0000 | 0.0605 | (0.0261, 0.1004) | 0.0000 |
| proposed_vs_candidate_no_context | context_overlap | 0.0162 | (0.0087, 0.0253) | 0.0000 | 0.0162 | (0.0052, 0.0283) | 0.0000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0400 | (-0.0750, -0.0077) | 0.9913 | -0.0400 | (-0.1114, 0.0121) | 0.9007 |
| proposed_vs_candidate_no_context | persona_style | -0.0056 | (-0.0302, 0.0198) | 0.6843 | -0.0056 | (-0.0213, 0.0090) | 0.7683 |
| proposed_vs_candidate_no_context | distinct1 | 0.0024 | (-0.0069, 0.0120) | 0.2973 | 0.0024 | (-0.0087, 0.0114) | 0.3057 |
| proposed_vs_candidate_no_context | length_score | -0.0128 | (-0.0744, 0.0506) | 0.6550 | -0.0128 | (-0.1077, 0.0631) | 0.5660 |
| proposed_vs_candidate_no_context | sentence_score | -0.0250 | (-0.0625, 0.0125) | 0.9117 | -0.0250 | (-0.1062, 0.0437) | 0.7607 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0237 | (0.0110, 0.0368) | 0.0003 | 0.0237 | (0.0005, 0.0454) | 0.0207 |
| proposed_vs_candidate_no_context | overall_quality | 0.0103 | (-0.0053, 0.0258) | 0.0913 | 0.0103 | (-0.0172, 0.0318) | 0.2130 |
| proposed_vs_baseline_no_context | context_relevance | 0.0294 | (0.0109, 0.0489) | 0.0007 | 0.0294 | (-0.0047, 0.0636) | 0.0433 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0536 | (-0.0825, -0.0250) | 0.9997 | -0.0536 | (-0.0981, -0.0103) | 0.9943 |
| proposed_vs_baseline_no_context | naturalness | -0.0745 | (-0.0912, -0.0560) | 1.0000 | -0.0745 | (-0.1021, -0.0478) | 1.0000 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0347 | (0.0116, 0.0606) | 0.0017 | 0.0347 | (-0.0110, 0.0801) | 0.0710 |
| proposed_vs_baseline_no_context | context_overlap | 0.0169 | (0.0087, 0.0260) | 0.0000 | 0.0169 | (0.0066, 0.0274) | 0.0000 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0511 | (-0.0859, -0.0178) | 0.9980 | -0.0511 | (-0.0982, -0.0109) | 0.9937 |
| proposed_vs_baseline_no_context | persona_style | -0.0638 | (-0.1066, -0.0215) | 0.9990 | -0.0638 | (-0.1965, 0.0353) | 0.8307 |
| proposed_vs_baseline_no_context | distinct1 | -0.0457 | (-0.0544, -0.0372) | 1.0000 | -0.0457 | (-0.0559, -0.0359) | 1.0000 |
| proposed_vs_baseline_no_context | length_score | -0.2310 | (-0.2946, -0.1655) | 1.0000 | -0.2310 | (-0.3286, -0.1327) | 1.0000 |
| proposed_vs_baseline_no_context | sentence_score | -0.1000 | (-0.1406, -0.0563) | 1.0000 | -0.1000 | (-0.1906, -0.0187) | 0.9927 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0159 | (0.0033, 0.0283) | 0.0050 | 0.0159 | (-0.0054, 0.0385) | 0.0780 |
| proposed_vs_baseline_no_context | overall_quality | -0.0169 | (-0.0336, -0.0007) | 0.9813 | -0.0169 | (-0.0458, 0.0137) | 0.8720 |
| controlled_vs_proposed_raw | context_relevance | 0.1922 | (0.1708, 0.2136) | 0.0000 | 0.1922 | (0.1653, 0.2206) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.1663 | (0.1389, 0.1944) | 0.0000 | 0.1663 | (0.1344, 0.1956) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.1001 | (0.0810, 0.1190) | 0.0000 | 0.1001 | (0.0554, 0.1429) | 0.0000 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2534 | (0.2256, 0.2795) | 0.0000 | 0.2534 | (0.2151, 0.2903) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0495 | (0.0399, 0.0583) | 0.0000 | 0.0495 | (0.0401, 0.0626) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1871 | (0.1552, 0.2179) | 0.0000 | 0.1871 | (0.1533, 0.2143) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.0831 | (0.0452, 0.1240) | 0.0000 | 0.0831 | (-0.0046, 0.2137) | 0.0820 |
| controlled_vs_proposed_raw | distinct1 | 0.0005 | (-0.0079, 0.0087) | 0.4457 | 0.0005 | (-0.0128, 0.0153) | 0.4717 |
| controlled_vs_proposed_raw | length_score | 0.3949 | (0.3173, 0.4703) | 0.0000 | 0.3949 | (0.2348, 0.5464) | 0.0000 |
| controlled_vs_proposed_raw | sentence_score | 0.2094 | (0.1718, 0.2469) | 0.0000 | 0.2094 | (0.1375, 0.2781) | 0.0000 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0312 | (0.0170, 0.0460) | 0.0000 | 0.0312 | (0.0060, 0.0645) | 0.0023 |
| controlled_vs_proposed_raw | overall_quality | 0.1539 | (0.1371, 0.1696) | 0.0000 | 0.1539 | (0.1317, 0.1782) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2395 | (0.2245, 0.2547) | 0.0000 | 0.2395 | (0.2099, 0.2647) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1332 | (0.1043, 0.1606) | 0.0000 | 0.1332 | (0.0598, 0.1924) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.0969 | (0.0765, 0.1178) | 0.0000 | 0.0969 | (0.0455, 0.1523) | 0.0000 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3139 | (0.2944, 0.3339) | 0.0000 | 0.3139 | (0.2739, 0.3470) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0657 | (0.0595, 0.0721) | 0.0000 | 0.0657 | (0.0544, 0.0766) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1471 | (0.1132, 0.1775) | 0.0000 | 0.1471 | (0.0662, 0.2094) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.0775 | (0.0374, 0.1212) | 0.0000 | 0.0775 | (-0.0247, 0.2249) | 0.1190 |
| controlled_vs_candidate_no_context | distinct1 | 0.0028 | (-0.0063, 0.0120) | 0.2777 | 0.0028 | (-0.0131, 0.0214) | 0.4117 |
| controlled_vs_candidate_no_context | length_score | 0.3821 | (0.3009, 0.4607) | 0.0000 | 0.3821 | (0.1929, 0.5765) | 0.0000 |
| controlled_vs_candidate_no_context | sentence_score | 0.1844 | (0.1469, 0.2219) | 0.0000 | 0.1844 | (0.0875, 0.2781) | 0.0000 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0549 | (0.0425, 0.0670) | 0.0000 | 0.0549 | (0.0367, 0.0718) | 0.0000 |
| controlled_vs_candidate_no_context | overall_quality | 0.1643 | (0.1505, 0.1779) | 0.0000 | 0.1643 | (0.1249, 0.1953) | 0.0000 |
| controlled_vs_baseline_no_context | context_relevance | 0.2216 | (0.2078, 0.2368) | 0.0000 | 0.2216 | (0.1990, 0.2432) | 0.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1126 | (0.0839, 0.1401) | 0.0000 | 0.1126 | (0.0529, 0.1642) | 0.0000 |
| controlled_vs_baseline_no_context | naturalness | 0.0256 | (0.0079, 0.0428) | 0.0030 | 0.0256 | (-0.0068, 0.0557) | 0.0587 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.2881 | (0.2693, 0.3084) | 0.0000 | 0.2881 | (0.2578, 0.3165) | 0.0000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0664 | (0.0604, 0.0726) | 0.0000 | 0.0664 | (0.0548, 0.0767) | 0.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1360 | (0.1036, 0.1682) | 0.0000 | 0.1360 | (0.0614, 0.1994) | 0.0000 |
| controlled_vs_baseline_no_context | persona_style | 0.0193 | (-0.0036, 0.0413) | 0.0513 | 0.0193 | (0.0005, 0.0460) | 0.0170 |
| controlled_vs_baseline_no_context | distinct1 | -0.0452 | (-0.0519, -0.0383) | 1.0000 | -0.0452 | (-0.0569, -0.0308) | 1.0000 |
| controlled_vs_baseline_no_context | length_score | 0.1640 | (0.0887, 0.2402) | 0.0000 | 0.1640 | (0.0083, 0.2985) | 0.0187 |
| controlled_vs_baseline_no_context | sentence_score | 0.1094 | (0.0750, 0.1437) | 0.0000 | 0.1094 | (0.0405, 0.1906) | 0.0000 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0471 | (0.0334, 0.0608) | 0.0000 | 0.0471 | (0.0274, 0.0691) | 0.0000 |
| controlled_vs_baseline_no_context | overall_quality | 0.1370 | (0.1252, 0.1486) | 0.0000 | 0.1370 | (0.1079, 0.1626) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0018 | (-0.0214, 0.0165) | 0.5683 | -0.0018 | (-0.0100, 0.0084) | 0.6647 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0062 | (-0.0229, 0.0109) | 0.7450 | -0.0062 | (-0.0193, 0.0073) | 0.8190 |
| controlled_alt_vs_controlled_default | naturalness | -0.0102 | (-0.0219, 0.0010) | 0.9633 | -0.0102 | (-0.0239, 0.0018) | 0.9443 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0019 | (-0.0282, 0.0226) | 0.5573 | -0.0019 | (-0.0122, 0.0104) | 0.6647 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0015 | (-0.0079, 0.0047) | 0.6737 | -0.0015 | (-0.0078, 0.0052) | 0.6880 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0101 | (-0.0311, 0.0106) | 0.8210 | -0.0101 | (-0.0279, 0.0078) | 0.8570 |
| controlled_alt_vs_controlled_default | persona_style | 0.0095 | (-0.0007, 0.0207) | 0.0373 | 0.0095 | (-0.0020, 0.0227) | 0.0747 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0004 | (-0.0059, 0.0069) | 0.4513 | 0.0004 | (-0.0045, 0.0070) | 0.4757 |
| controlled_alt_vs_controlled_default | length_score | -0.0357 | (-0.0857, 0.0134) | 0.9163 | -0.0357 | (-0.1060, 0.0208) | 0.8507 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0312 | (-0.0594, -0.0031) | 0.9877 | -0.0312 | (-0.0594, -0.0062) | 0.9967 |
| controlled_alt_vs_controlled_default | bertscore_f1 | -0.0084 | (-0.0179, 0.0011) | 0.9543 | -0.0084 | (-0.0150, 0.0007) | 0.9627 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0057 | (-0.0147, 0.0034) | 0.8897 | -0.0057 | (-0.0112, -0.0003) | 0.9793 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1904 | (0.1679, 0.2116) | 0.0000 | 0.1904 | (0.1566, 0.2205) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1601 | (0.1362, 0.1840) | 0.0000 | 0.1601 | (0.1236, 0.1990) | 0.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0899 | (0.0677, 0.1120) | 0.0000 | 0.0899 | (0.0417, 0.1368) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2514 | (0.2214, 0.2808) | 0.0000 | 0.2514 | (0.2088, 0.2922) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0480 | (0.0379, 0.0566) | 0.0000 | 0.0480 | (0.0386, 0.0589) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1770 | (0.1510, 0.2031) | 0.0000 | 0.1770 | (0.1450, 0.1997) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0926 | (0.0581, 0.1326) | 0.0000 | 0.0926 | (0.0051, 0.2192) | 0.0003 |
| controlled_alt_vs_proposed_raw | distinct1 | 0.0009 | (-0.0088, 0.0102) | 0.4153 | 0.0009 | (-0.0105, 0.0134) | 0.4720 |
| controlled_alt_vs_proposed_raw | length_score | 0.3592 | (0.2726, 0.4458) | 0.0000 | 0.3592 | (0.1839, 0.5381) | 0.0000 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1781 | (0.1313, 0.2188) | 0.0000 | 0.1781 | (0.0969, 0.2594) | 0.0000 |
| controlled_alt_vs_proposed_raw | bertscore_f1 | 0.0228 | (0.0089, 0.0364) | 0.0010 | 0.0228 | (-0.0002, 0.0533) | 0.0257 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1482 | (0.1325, 0.1636) | 0.0000 | 0.1482 | (0.1254, 0.1741) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2377 | (0.2197, 0.2557) | 0.0000 | 0.2377 | (0.2023, 0.2702) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1270 | (0.0979, 0.1565) | 0.0000 | 0.1270 | (0.0527, 0.1928) | 0.0007 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0867 | (0.0651, 0.1075) | 0.0000 | 0.0867 | (0.0337, 0.1435) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3120 | (0.2877, 0.3352) | 0.0000 | 0.3120 | (0.2614, 0.3551) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0642 | (0.0577, 0.0709) | 0.0000 | 0.0642 | (0.0525, 0.0759) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1370 | (0.1040, 0.1687) | 0.0000 | 0.1370 | (0.0578, 0.2018) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0870 | (0.0484, 0.1277) | 0.0000 | 0.0870 | (-0.0076, 0.2271) | 0.0783 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0032 | (-0.0065, 0.0131) | 0.2540 | 0.0032 | (-0.0144, 0.0221) | 0.3760 |
| controlled_alt_vs_candidate_no_context | length_score | 0.3464 | (0.2595, 0.4259) | 0.0000 | 0.3464 | (0.1485, 0.5565) | 0.0000 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1531 | (0.1094, 0.1938) | 0.0000 | 0.1531 | (0.0437, 0.2594) | 0.0020 |
| controlled_alt_vs_candidate_no_context | bertscore_f1 | 0.0465 | (0.0344, 0.0590) | 0.0000 | 0.0465 | (0.0289, 0.0650) | 0.0000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1586 | (0.1426, 0.1734) | 0.0000 | 0.1586 | (0.1157, 0.1913) | 0.0000 |
| controlled_alt_vs_baseline_no_context | context_relevance | 0.2198 | (0.2012, 0.2382) | 0.0000 | 0.2198 | (0.1926, 0.2468) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 0.1065 | (0.0779, 0.1333) | 0.0000 | 0.1065 | (0.0541, 0.1541) | 0.0000 |
| controlled_alt_vs_baseline_no_context | naturalness | 0.0155 | (-0.0023, 0.0327) | 0.0457 | 0.0155 | (-0.0146, 0.0441) | 0.1610 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 0.2861 | (0.2624, 0.3109) | 0.0000 | 0.2861 | (0.2498, 0.3222) | 0.0000 |
| controlled_alt_vs_baseline_no_context | context_overlap | 0.0649 | (0.0582, 0.0717) | 0.0000 | 0.0649 | (0.0526, 0.0756) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 0.1259 | (0.0929, 0.1615) | 0.0000 | 0.1259 | (0.0561, 0.1847) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_style | 0.0288 | (0.0056, 0.0548) | 0.0070 | 0.0288 | (0.0001, 0.0690) | 0.0240 |
| controlled_alt_vs_baseline_no_context | distinct1 | -0.0448 | (-0.0523, -0.0368) | 1.0000 | -0.0448 | (-0.0561, -0.0320) | 1.0000 |
| controlled_alt_vs_baseline_no_context | length_score | 0.1283 | (0.0503, 0.2039) | 0.0013 | 0.1283 | (-0.0080, 0.2611) | 0.0323 |
| controlled_alt_vs_baseline_no_context | sentence_score | 0.0781 | (0.0406, 0.1187) | 0.0000 | 0.0781 | (0.0094, 0.1562) | 0.0153 |
| controlled_alt_vs_baseline_no_context | bertscore_f1 | 0.0387 | (0.0252, 0.0530) | 0.0000 | 0.0387 | (0.0171, 0.0612) | 0.0000 |
| controlled_alt_vs_baseline_no_context | overall_quality | 0.1313 | (0.1183, 0.1445) | 0.0000 | 0.1313 | (0.1044, 0.1555) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2216 | (0.2080, 0.2370) | 0.0000 | 0.2216 | (0.1999, 0.2421) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1126 | (0.0857, 0.1396) | 0.0000 | 0.1126 | (0.0549, 0.1646) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 0.0256 | (0.0087, 0.0426) | 0.0017 | 0.0256 | (-0.0068, 0.0567) | 0.0573 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.2881 | (0.2690, 0.3086) | 0.0000 | 0.2881 | (0.2586, 0.3172) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0664 | (0.0602, 0.0723) | 0.0000 | 0.0664 | (0.0550, 0.0775) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1360 | (0.1019, 0.1682) | 0.0000 | 0.1360 | (0.0676, 0.1994) | 0.0003 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0193 | (-0.0022, 0.0428) | 0.0367 | 0.0193 | (0.0003, 0.0463) | 0.0200 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0452 | (-0.0520, -0.0382) | 1.0000 | -0.0452 | (-0.0567, -0.0302) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.1640 | (0.0830, 0.2390) | 0.0000 | 0.1640 | (0.0131, 0.2997) | 0.0130 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.1094 | (0.0719, 0.1437) | 0.0000 | 0.1094 | (0.0406, 0.1875) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0471 | (0.0331, 0.0604) | 0.0000 | 0.0471 | (0.0278, 0.0691) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1370 | (0.1245, 0.1495) | 0.0000 | 0.1370 | (0.1088, 0.1629) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 50 | 19 | 43 | 0.6384 | 0.7246 |
| proposed_vs_candidate_no_context | persona_consistency | 11 | 26 | 75 | 0.4330 | 0.2973 |
| proposed_vs_candidate_no_context | naturalness | 31 | 38 | 43 | 0.4688 | 0.4493 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 38 | 6 | 68 | 0.6429 | 0.8636 |
| proposed_vs_candidate_no_context | context_overlap | 46 | 23 | 43 | 0.6027 | 0.6667 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 9 | 19 | 84 | 0.4554 | 0.3214 |
| proposed_vs_candidate_no_context | persona_style | 6 | 14 | 92 | 0.4643 | 0.3000 |
| proposed_vs_candidate_no_context | distinct1 | 32 | 27 | 53 | 0.5223 | 0.5424 |
| proposed_vs_candidate_no_context | length_score | 31 | 37 | 44 | 0.4732 | 0.4559 |
| proposed_vs_candidate_no_context | sentence_score | 16 | 24 | 72 | 0.4643 | 0.4000 |
| proposed_vs_candidate_no_context | bertscore_f1 | 59 | 29 | 24 | 0.6339 | 0.6705 |
| proposed_vs_candidate_no_context | overall_quality | 55 | 33 | 24 | 0.5982 | 0.6250 |
| proposed_vs_baseline_no_context | context_relevance | 63 | 49 | 0 | 0.5625 | 0.5625 |
| proposed_vs_baseline_no_context | persona_consistency | 18 | 52 | 42 | 0.3482 | 0.2571 |
| proposed_vs_baseline_no_context | naturalness | 23 | 88 | 1 | 0.2098 | 0.2072 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 32 | 23 | 57 | 0.5402 | 0.5818 |
| proposed_vs_baseline_no_context | context_overlap | 70 | 42 | 0 | 0.6250 | 0.6250 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 10 | 36 | 66 | 0.3839 | 0.2174 |
| proposed_vs_baseline_no_context | persona_style | 12 | 27 | 73 | 0.4330 | 0.3077 |
| proposed_vs_baseline_no_context | distinct1 | 16 | 83 | 13 | 0.2009 | 0.1616 |
| proposed_vs_baseline_no_context | length_score | 27 | 83 | 2 | 0.2500 | 0.2455 |
| proposed_vs_baseline_no_context | sentence_score | 12 | 44 | 56 | 0.3571 | 0.2143 |
| proposed_vs_baseline_no_context | bertscore_f1 | 64 | 48 | 0 | 0.5714 | 0.5714 |
| proposed_vs_baseline_no_context | overall_quality | 41 | 71 | 0 | 0.3661 | 0.3661 |
| controlled_vs_proposed_raw | context_relevance | 104 | 8 | 0 | 0.9286 | 0.9286 |
| controlled_vs_proposed_raw | persona_consistency | 90 | 10 | 12 | 0.8571 | 0.9000 |
| controlled_vs_proposed_raw | naturalness | 89 | 23 | 0 | 0.7946 | 0.7946 |
| controlled_vs_proposed_raw | context_keyword_coverage | 102 | 4 | 6 | 0.9375 | 0.9623 |
| controlled_vs_proposed_raw | context_overlap | 102 | 10 | 0 | 0.9107 | 0.9107 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 90 | 9 | 13 | 0.8616 | 0.9091 |
| controlled_vs_proposed_raw | persona_style | 28 | 7 | 77 | 0.5938 | 0.8000 |
| controlled_vs_proposed_raw | distinct1 | 66 | 45 | 1 | 0.5938 | 0.5946 |
| controlled_vs_proposed_raw | length_score | 83 | 23 | 6 | 0.7679 | 0.7830 |
| controlled_vs_proposed_raw | sentence_score | 73 | 6 | 33 | 0.7991 | 0.9241 |
| controlled_vs_proposed_raw | bertscore_f1 | 82 | 30 | 0 | 0.7321 | 0.7321 |
| controlled_vs_proposed_raw | overall_quality | 105 | 7 | 0 | 0.9375 | 0.9375 |
| controlled_vs_candidate_no_context | context_relevance | 111 | 1 | 0 | 0.9911 | 0.9911 |
| controlled_vs_candidate_no_context | persona_consistency | 81 | 12 | 19 | 0.8080 | 0.8710 |
| controlled_vs_candidate_no_context | naturalness | 90 | 22 | 0 | 0.8036 | 0.8036 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 109 | 0 | 3 | 0.9866 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 110 | 2 | 0 | 0.9821 | 0.9821 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 80 | 9 | 23 | 0.8170 | 0.8989 |
| controlled_vs_candidate_no_context | persona_style | 26 | 9 | 77 | 0.5759 | 0.7429 |
| controlled_vs_candidate_no_context | distinct1 | 67 | 44 | 1 | 0.6027 | 0.6036 |
| controlled_vs_candidate_no_context | length_score | 90 | 19 | 3 | 0.8170 | 0.8257 |
| controlled_vs_candidate_no_context | sentence_score | 63 | 4 | 45 | 0.7634 | 0.9403 |
| controlled_vs_candidate_no_context | bertscore_f1 | 97 | 15 | 0 | 0.8661 | 0.8661 |
| controlled_vs_candidate_no_context | overall_quality | 108 | 4 | 0 | 0.9643 | 0.9643 |
| controlled_vs_baseline_no_context | context_relevance | 111 | 1 | 0 | 0.9911 | 0.9911 |
| controlled_vs_baseline_no_context | persona_consistency | 79 | 14 | 19 | 0.7902 | 0.8495 |
| controlled_vs_baseline_no_context | naturalness | 77 | 35 | 0 | 0.6875 | 0.6875 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 111 | 0 | 1 | 0.9955 | 1.0000 |
| controlled_vs_baseline_no_context | context_overlap | 110 | 1 | 1 | 0.9866 | 0.9910 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 77 | 11 | 24 | 0.7946 | 0.8750 |
| controlled_vs_baseline_no_context | persona_style | 14 | 10 | 88 | 0.5179 | 0.5833 |
| controlled_vs_baseline_no_context | distinct1 | 14 | 98 | 0 | 0.1250 | 0.1250 |
| controlled_vs_baseline_no_context | length_score | 82 | 27 | 3 | 0.7455 | 0.7523 |
| controlled_vs_baseline_no_context | sentence_score | 40 | 5 | 67 | 0.6562 | 0.8889 |
| controlled_vs_baseline_no_context | bertscore_f1 | 83 | 29 | 0 | 0.7411 | 0.7411 |
| controlled_vs_baseline_no_context | overall_quality | 111 | 1 | 0 | 0.9911 | 0.9911 |
| controlled_alt_vs_controlled_default | context_relevance | 30 | 33 | 49 | 0.4866 | 0.4762 |
| controlled_alt_vs_controlled_default | persona_consistency | 16 | 18 | 78 | 0.4911 | 0.4706 |
| controlled_alt_vs_controlled_default | naturalness | 21 | 42 | 49 | 0.4062 | 0.3333 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 26 | 26 | 60 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | context_overlap | 30 | 33 | 49 | 0.4866 | 0.4762 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 12 | 18 | 82 | 0.4732 | 0.4000 |
| controlled_alt_vs_controlled_default | persona_style | 9 | 2 | 101 | 0.5312 | 0.8182 |
| controlled_alt_vs_controlled_default | distinct1 | 29 | 33 | 50 | 0.4821 | 0.4677 |
| controlled_alt_vs_controlled_default | length_score | 21 | 35 | 56 | 0.4375 | 0.3750 |
| controlled_alt_vs_controlled_default | sentence_score | 6 | 16 | 90 | 0.4554 | 0.2727 |
| controlled_alt_vs_controlled_default | bertscore_f1 | 25 | 42 | 45 | 0.4241 | 0.3731 |
| controlled_alt_vs_controlled_default | overall_quality | 32 | 35 | 45 | 0.4866 | 0.4776 |
| controlled_alt_vs_proposed_raw | context_relevance | 104 | 8 | 0 | 0.9286 | 0.9286 |
| controlled_alt_vs_proposed_raw | persona_consistency | 91 | 7 | 14 | 0.8750 | 0.9286 |
| controlled_alt_vs_proposed_raw | naturalness | 87 | 25 | 0 | 0.7768 | 0.7768 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 99 | 7 | 6 | 0.9107 | 0.9340 |
| controlled_alt_vs_proposed_raw | context_overlap | 101 | 11 | 0 | 0.9018 | 0.9018 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 89 | 5 | 18 | 0.8750 | 0.9468 |
| controlled_alt_vs_proposed_raw | persona_style | 29 | 3 | 80 | 0.6161 | 0.9062 |
| controlled_alt_vs_proposed_raw | distinct1 | 67 | 40 | 5 | 0.6205 | 0.6262 |
| controlled_alt_vs_proposed_raw | length_score | 86 | 24 | 2 | 0.7768 | 0.7818 |
| controlled_alt_vs_proposed_raw | sentence_score | 68 | 11 | 33 | 0.7545 | 0.8608 |
| controlled_alt_vs_proposed_raw | bertscore_f1 | 78 | 34 | 0 | 0.6964 | 0.6964 |
| controlled_alt_vs_proposed_raw | overall_quality | 104 | 8 | 0 | 0.9286 | 0.9286 |
| controlled_alt_vs_candidate_no_context | context_relevance | 109 | 3 | 0 | 0.9732 | 0.9732 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 80 | 14 | 18 | 0.7946 | 0.8511 |
| controlled_alt_vs_candidate_no_context | naturalness | 81 | 31 | 0 | 0.7232 | 0.7232 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 107 | 0 | 5 | 0.9777 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 109 | 3 | 0 | 0.9732 | 0.9732 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 79 | 13 | 20 | 0.7946 | 0.8587 |
| controlled_alt_vs_candidate_no_context | persona_style | 25 | 6 | 81 | 0.5848 | 0.8065 |
| controlled_alt_vs_candidate_no_context | distinct1 | 71 | 38 | 3 | 0.6473 | 0.6514 |
| controlled_alt_vs_candidate_no_context | length_score | 81 | 26 | 5 | 0.7455 | 0.7570 |
| controlled_alt_vs_candidate_no_context | sentence_score | 60 | 11 | 41 | 0.7188 | 0.8451 |
| controlled_alt_vs_candidate_no_context | bertscore_f1 | 90 | 22 | 0 | 0.8036 | 0.8036 |
| controlled_alt_vs_candidate_no_context | overall_quality | 106 | 6 | 0 | 0.9464 | 0.9464 |
| controlled_alt_vs_baseline_no_context | context_relevance | 109 | 3 | 0 | 0.9732 | 0.9732 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 77 | 18 | 17 | 0.7634 | 0.8105 |
| controlled_alt_vs_baseline_no_context | naturalness | 70 | 42 | 0 | 0.6250 | 0.6250 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 106 | 1 | 5 | 0.9688 | 0.9907 |
| controlled_alt_vs_baseline_no_context | context_overlap | 108 | 3 | 1 | 0.9688 | 0.9730 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 74 | 14 | 24 | 0.7679 | 0.8409 |
| controlled_alt_vs_baseline_no_context | persona_style | 17 | 6 | 89 | 0.5491 | 0.7391 |
| controlled_alt_vs_baseline_no_context | distinct1 | 13 | 93 | 6 | 0.1429 | 0.1226 |
| controlled_alt_vs_baseline_no_context | length_score | 73 | 32 | 7 | 0.6830 | 0.6952 |
| controlled_alt_vs_baseline_no_context | sentence_score | 36 | 11 | 65 | 0.6116 | 0.7660 |
| controlled_alt_vs_baseline_no_context | bertscore_f1 | 79 | 33 | 0 | 0.7054 | 0.7054 |
| controlled_alt_vs_baseline_no_context | overall_quality | 109 | 3 | 0 | 0.9732 | 0.9732 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 111 | 1 | 0 | 0.9911 | 0.9911 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 79 | 14 | 19 | 0.7902 | 0.8495 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 77 | 35 | 0 | 0.6875 | 0.6875 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 111 | 0 | 1 | 0.9955 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 110 | 1 | 1 | 0.9866 | 0.9910 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 77 | 11 | 24 | 0.7946 | 0.8750 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 14 | 10 | 88 | 0.5179 | 0.5833 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | 14 | 98 | 0 | 0.1250 | 0.1250 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 82 | 27 | 3 | 0.7455 | 0.7523 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 40 | 5 | 67 | 0.6562 | 0.8889 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 83 | 29 | 0 | 0.7411 | 0.7411 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 111 | 1 | 0 | 0.9911 | 0.9911 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.0000 | 0.3571 | 0.0982 |
| proposed_contextual_controlled_alt | 0.0000 | 0.0000 | 0.0000 | 0.4643 | 0.0982 |
| proposed_contextual | 0.0000 | 0.0000 | 0.4911 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.5089 | 0.0000 | 0.0000 |
| baseline_no_context | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `8`
- Unique template signatures: `71`
- Template signature ratio: `0.6339`
- Effective sample size by source clustering: `8.00`
- Effective sample size by template-signature clustering: `56.00`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: enabled.

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
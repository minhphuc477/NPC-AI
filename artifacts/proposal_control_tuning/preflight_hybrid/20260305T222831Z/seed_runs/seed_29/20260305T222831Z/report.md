# Proposal Alignment Evaluation Report

- Run ID: `20260305T222831Z`
- Generated: `2026-03-05T22:35:15.885521+00:00`
- Scenarios: `artifacts\proposal_control_tuning\preflight_hybrid\20260305T222831Z\seed_runs\seed_29\20260305T222831Z\scenarios.jsonl`
- Scenario count: `24`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_hb`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off
- `baseline_no_context`: model `phi3:mini`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.3069 (0.2558, 0.3608) | 0.3570 (0.2926, 0.4251) | 0.8381 (0.8122, 0.8639) | 0.3962 (0.3685, 0.4266) | 0.0911 |
| proposed_contextual_controlled_hb | 0.2689 (0.2420, 0.2971) | 0.3459 (0.2861, 0.4136) | 0.8731 (0.8453, 0.8987) | 0.3820 (0.3605, 0.4017) | 0.0823 |
| proposed_contextual | 0.0484 (0.0221, 0.0796) | 0.1232 (0.0875, 0.1621) | 0.8148 (0.7828, 0.8484) | 0.2047 (0.1799, 0.2327) | 0.0371 |
| candidate_no_context | 0.0287 (0.0153, 0.0449) | 0.1937 (0.1258, 0.2680) | 0.8137 (0.7814, 0.8470) | 0.2180 (0.1897, 0.2487) | 0.0245 |
| baseline_no_context | 0.0367 (0.0199, 0.0562) | 0.1782 (0.1337, 0.2272) | 0.8956 (0.8736, 0.9191) | 0.2313 (0.2134, 0.2500) | 0.0351 |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0197 | 0.6846 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0705 | -0.3638 |
| proposed_vs_candidate_no_context | naturalness | 0.0011 | 0.0014 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0265 | 1.1667 |
| proposed_vs_candidate_no_context | context_overlap | 0.0037 | 0.0872 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0815 | -0.6908 |
| proposed_vs_candidate_no_context | persona_style | -0.0261 | -0.0526 |
| proposed_vs_candidate_no_context | distinct1 | 0.0009 | 0.0010 |
| proposed_vs_candidate_no_context | length_score | 0.0111 | 0.0356 |
| proposed_vs_candidate_no_context | sentence_score | -0.0146 | -0.0187 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0126 | 0.5147 |
| proposed_vs_candidate_no_context | overall_quality | -0.0134 | -0.0613 |
| proposed_vs_baseline_no_context | context_relevance | 0.0117 | 0.3198 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0550 | -0.3084 |
| proposed_vs_baseline_no_context | naturalness | -0.0808 | -0.0902 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0144 | 0.4130 |
| proposed_vs_baseline_no_context | context_overlap | 0.0055 | 0.1348 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0429 | -0.5400 |
| proposed_vs_baseline_no_context | persona_style | -0.1034 | -0.1803 |
| proposed_vs_baseline_no_context | distinct1 | -0.0523 | -0.0530 |
| proposed_vs_baseline_no_context | length_score | -0.2556 | -0.4412 |
| proposed_vs_baseline_no_context | sentence_score | -0.0875 | -0.1024 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0020 | 0.0558 |
| proposed_vs_baseline_no_context | overall_quality | -0.0266 | -0.1151 |
| controlled_vs_proposed_raw | context_relevance | 0.2585 | 5.3388 |
| controlled_vs_proposed_raw | persona_consistency | 0.2338 | 1.8972 |
| controlled_vs_proposed_raw | naturalness | 0.0233 | 0.0286 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.3372 | 6.8474 |
| controlled_vs_proposed_raw | context_overlap | 0.0750 | 1.6120 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2643 | 7.2391 |
| controlled_vs_proposed_raw | persona_style | 0.1117 | 0.2377 |
| controlled_vs_proposed_raw | distinct1 | -0.0031 | -0.0033 |
| controlled_vs_proposed_raw | length_score | 0.0389 | 0.1202 |
| controlled_vs_proposed_raw | sentence_score | 0.1750 | 0.2283 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0540 | 1.4573 |
| controlled_vs_proposed_raw | overall_quality | 0.1915 | 0.9358 |
| controlled_vs_candidate_no_context | context_relevance | 0.2782 | 9.6785 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1633 | 0.8432 |
| controlled_vs_candidate_no_context | naturalness | 0.0244 | 0.0300 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3637 | 16.0028 |
| controlled_vs_candidate_no_context | context_overlap | 0.0787 | 1.8397 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1827 | 1.5479 |
| controlled_vs_candidate_no_context | persona_style | 0.0856 | 0.1725 |
| controlled_vs_candidate_no_context | distinct1 | -0.0022 | -0.0024 |
| controlled_vs_candidate_no_context | length_score | 0.0500 | 0.1600 |
| controlled_vs_candidate_no_context | sentence_score | 0.1604 | 0.2053 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0666 | 2.7220 |
| controlled_vs_candidate_no_context | overall_quality | 0.1782 | 0.8172 |
| controlled_vs_baseline_no_context | context_relevance | 0.2703 | 7.3658 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1788 | 1.0036 |
| controlled_vs_baseline_no_context | naturalness | -0.0575 | -0.0642 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3516 | 10.0888 |
| controlled_vs_baseline_no_context | context_overlap | 0.0805 | 1.9640 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2214 | 2.7900 |
| controlled_vs_baseline_no_context | persona_style | 0.0084 | 0.0146 |
| controlled_vs_baseline_no_context | distinct1 | -0.0554 | -0.0562 |
| controlled_vs_baseline_no_context | length_score | -0.2167 | -0.3741 |
| controlled_vs_baseline_no_context | sentence_score | 0.0875 | 0.1024 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0560 | 1.5945 |
| controlled_vs_baseline_no_context | overall_quality | 0.1649 | 0.7131 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0381 | -0.1240 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0111 | -0.0312 |
| controlled_alt_vs_controlled_default | naturalness | 0.0350 | 0.0417 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0513 | -0.1328 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0071 | -0.0585 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0179 | -0.0594 |
| controlled_alt_vs_controlled_default | persona_style | 0.0158 | 0.0272 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0088 | 0.0095 |
| controlled_alt_vs_controlled_default | length_score | 0.1681 | 0.4636 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0292 | -0.0310 |
| controlled_alt_vs_controlled_default | bertscore_f1 | -0.0088 | -0.0966 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0142 | -0.0358 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.2205 | 4.5528 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.2226 | 1.8070 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0583 | 0.0715 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2859 | 5.8051 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0679 | 1.4591 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.2464 | 6.7500 |
| controlled_alt_vs_proposed_raw | persona_style | 0.1275 | 0.2713 |
| controlled_alt_vs_proposed_raw | distinct1 | 0.0057 | 0.0061 |
| controlled_alt_vs_proposed_raw | length_score | 0.2069 | 0.6395 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1458 | 0.1902 |
| controlled_alt_vs_proposed_raw | bertscore_f1 | 0.0452 | 1.2200 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1774 | 0.8665 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2401 | 8.3544 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1522 | 0.7858 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0594 | 0.0730 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3124 | 13.7444 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0716 | 1.6734 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1649 | 1.3966 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.1014 | 0.2044 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0066 | 0.0071 |
| controlled_alt_vs_candidate_no_context | length_score | 0.2181 | 0.6978 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1312 | 0.1680 |
| controlled_alt_vs_candidate_no_context | bertscore_f1 | 0.0578 | 2.3626 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1640 | 0.7521 |
| controlled_alt_vs_baseline_no_context | context_relevance | 0.2322 | 6.3284 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 0.1677 | 0.9412 |
| controlled_alt_vs_baseline_no_context | naturalness | -0.0225 | -0.0251 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 0.3003 | 8.6159 |
| controlled_alt_vs_baseline_no_context | context_overlap | 0.0734 | 1.7905 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 0.2036 | 2.5650 |
| controlled_alt_vs_baseline_no_context | persona_style | 0.0242 | 0.0422 |
| controlled_alt_vs_baseline_no_context | distinct1 | -0.0466 | -0.0472 |
| controlled_alt_vs_baseline_no_context | length_score | -0.0486 | -0.0839 |
| controlled_alt_vs_baseline_no_context | sentence_score | 0.0583 | 0.0683 |
| controlled_alt_vs_baseline_no_context | bertscore_f1 | 0.0472 | 1.3439 |
| controlled_alt_vs_baseline_no_context | overall_quality | 0.1507 | 0.6517 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2703 | 7.3658 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1788 | 1.0036 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0575 | -0.0642 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3516 | 10.0888 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0805 | 1.9640 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2214 | 2.7900 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0084 | 0.0146 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0554 | -0.0562 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | -0.2167 | -0.3741 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0875 | 0.1024 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0560 | 1.5945 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1649 | 0.7131 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0197 | (0.0010, 0.0424) | 0.0187 | 0.0197 | (0.0023, 0.0371) | 0.0110 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0705 | (-0.1592, 0.0086) | 0.9553 | -0.0705 | (-0.2207, 0.0317) | 0.8533 |
| proposed_vs_candidate_no_context | naturalness | 0.0011 | (-0.0337, 0.0367) | 0.4847 | 0.0011 | (-0.0540, 0.0433) | 0.4543 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0265 | (0.0000, 0.0568) | 0.0263 | 0.0265 | (0.0040, 0.0492) | 0.0163 |
| proposed_vs_candidate_no_context | context_overlap | 0.0037 | (-0.0054, 0.0131) | 0.2210 | 0.0037 | (-0.0062, 0.0127) | 0.2277 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0815 | (-0.1736, 0.0026) | 0.9677 | -0.0815 | (-0.2501, 0.0265) | 0.8780 |
| proposed_vs_candidate_no_context | persona_style | -0.0261 | (-0.1133, 0.0621) | 0.7237 | -0.0261 | (-0.1163, 0.0704) | 0.6980 |
| proposed_vs_candidate_no_context | distinct1 | 0.0009 | (-0.0109, 0.0151) | 0.4407 | 0.0009 | (-0.0156, 0.0148) | 0.4407 |
| proposed_vs_candidate_no_context | length_score | 0.0111 | (-0.1028, 0.1319) | 0.4183 | 0.0111 | (-0.1654, 0.1667) | 0.4193 |
| proposed_vs_candidate_no_context | sentence_score | -0.0146 | (-0.0875, 0.0583) | 0.7243 | -0.0146 | (-0.1313, 0.0729) | 0.6320 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0126 | (-0.0098, 0.0335) | 0.1377 | 0.0126 | (-0.0037, 0.0302) | 0.0660 |
| proposed_vs_candidate_no_context | overall_quality | -0.0134 | (-0.0499, 0.0223) | 0.7407 | -0.0134 | (-0.0726, 0.0284) | 0.6683 |
| proposed_vs_baseline_no_context | context_relevance | 0.0117 | (-0.0168, 0.0417) | 0.2150 | 0.0117 | (-0.0051, 0.0319) | 0.0930 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0550 | (-0.1288, 0.0107) | 0.9463 | -0.0550 | (-0.1794, 0.0394) | 0.8090 |
| proposed_vs_baseline_no_context | naturalness | -0.0808 | (-0.1223, -0.0328) | 0.9997 | -0.0808 | (-0.1314, -0.0097) | 0.9850 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0144 | (-0.0242, 0.0531) | 0.2440 | 0.0144 | (-0.0095, 0.0413) | 0.1453 |
| proposed_vs_baseline_no_context | context_overlap | 0.0055 | (-0.0056, 0.0173) | 0.1730 | 0.0055 | (-0.0070, 0.0203) | 0.2110 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0429 | (-0.1310, 0.0314) | 0.8477 | -0.0429 | (-0.1973, 0.0576) | 0.6980 |
| proposed_vs_baseline_no_context | persona_style | -0.1034 | (-0.2062, -0.0114) | 0.9883 | -0.1034 | (-0.2562, -0.0004) | 1.0000 |
| proposed_vs_baseline_no_context | distinct1 | -0.0523 | (-0.0663, -0.0387) | 1.0000 | -0.0523 | (-0.0665, -0.0325) | 1.0000 |
| proposed_vs_baseline_no_context | length_score | -0.2556 | (-0.4153, -0.0806) | 0.9970 | -0.2556 | (-0.4500, 0.0070) | 0.9727 |
| proposed_vs_baseline_no_context | sentence_score | -0.0875 | (-0.1896, 0.0146) | 0.9547 | -0.0875 | (-0.1815, 0.0553) | 0.9187 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0020 | (-0.0248, 0.0273) | 0.4297 | 0.0020 | (-0.0255, 0.0327) | 0.4653 |
| proposed_vs_baseline_no_context | overall_quality | -0.0266 | (-0.0617, 0.0075) | 0.9290 | -0.0266 | (-0.0765, 0.0198) | 0.8483 |
| controlled_vs_proposed_raw | context_relevance | 0.2585 | (0.2047, 0.3138) | 0.0000 | 0.2585 | (0.2131, 0.3191) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.2338 | (0.1550, 0.3256) | 0.0000 | 0.2338 | (0.1394, 0.3970) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0233 | (-0.0120, 0.0586) | 0.0950 | 0.0233 | (-0.0197, 0.0589) | 0.1620 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.3372 | (0.2662, 0.4107) | 0.0000 | 0.3372 | (0.2781, 0.4127) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0750 | (0.0515, 0.1010) | 0.0000 | 0.0750 | (0.0482, 0.1054) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2643 | (0.1730, 0.3732) | 0.0000 | 0.2643 | (0.1615, 0.4578) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.1117 | (0.0073, 0.2215) | 0.0177 | 0.1117 | (-0.0042, 0.2656) | 0.0427 |
| controlled_vs_proposed_raw | distinct1 | -0.0031 | (-0.0200, 0.0127) | 0.6480 | -0.0031 | (-0.0223, 0.0095) | 0.6467 |
| controlled_vs_proposed_raw | length_score | 0.0389 | (-0.1000, 0.1819) | 0.2953 | 0.0389 | (-0.1276, 0.1917) | 0.3277 |
| controlled_vs_proposed_raw | sentence_score | 0.1750 | (0.0875, 0.2625) | 0.0003 | 0.1750 | (0.0553, 0.2534) | 0.0040 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0540 | (0.0231, 0.0842) | 0.0003 | 0.0540 | (0.0225, 0.0848) | 0.0000 |
| controlled_vs_proposed_raw | overall_quality | 0.1915 | (0.1539, 0.2337) | 0.0000 | 0.1915 | (0.1485, 0.2510) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2782 | (0.2287, 0.3306) | 0.0000 | 0.2782 | (0.2297, 0.3417) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1633 | (0.1017, 0.2183) | 0.0000 | 0.1633 | (0.1350, 0.1895) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.0244 | (-0.0129, 0.0629) | 0.1027 | 0.0244 | (-0.0396, 0.0795) | 0.2497 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3637 | (0.3008, 0.4272) | 0.0000 | 0.3637 | (0.2986, 0.4441) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0787 | (0.0565, 0.1040) | 0.0000 | 0.0787 | (0.0559, 0.1050) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1827 | (0.1097, 0.2472) | 0.0000 | 0.1827 | (0.1361, 0.2263) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.0856 | (-0.0102, 0.1899) | 0.0457 | 0.0856 | (-0.0448, 0.2238) | 0.1057 |
| controlled_vs_candidate_no_context | distinct1 | -0.0022 | (-0.0203, 0.0164) | 0.5963 | -0.0022 | (-0.0263, 0.0150) | 0.5720 |
| controlled_vs_candidate_no_context | length_score | 0.0500 | (-0.1042, 0.2069) | 0.2647 | 0.0500 | (-0.1921, 0.2825) | 0.3670 |
| controlled_vs_candidate_no_context | sentence_score | 0.1604 | (0.0729, 0.2333) | 0.0000 | 0.1604 | (0.0389, 0.2567) | 0.0063 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0666 | (0.0339, 0.1018) | 0.0000 | 0.0666 | (0.0383, 0.1003) | 0.0000 |
| controlled_vs_candidate_no_context | overall_quality | 0.1782 | (0.1461, 0.2080) | 0.0000 | 0.1782 | (0.1566, 0.2093) | 0.0000 |
| controlled_vs_baseline_no_context | context_relevance | 0.2703 | (0.2183, 0.3262) | 0.0000 | 0.2703 | (0.2279, 0.3322) | 0.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1788 | (0.1346, 0.2233) | 0.0000 | 0.1788 | (0.1422, 0.2241) | 0.0000 |
| controlled_vs_baseline_no_context | naturalness | -0.0575 | (-0.0908, -0.0244) | 0.9997 | -0.0575 | (-0.0962, -0.0134) | 0.9950 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3516 | (0.2856, 0.4227) | 0.0000 | 0.3516 | (0.2963, 0.4307) | 0.0000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0805 | (0.0602, 0.1023) | 0.0000 | 0.0805 | (0.0629, 0.1012) | 0.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2214 | (0.1675, 0.2786) | 0.0000 | 0.2214 | (0.1773, 0.2775) | 0.0000 |
| controlled_vs_baseline_no_context | persona_style | 0.0084 | (-0.0359, 0.0506) | 0.3530 | 0.0084 | (-0.0332, 0.0463) | 0.3723 |
| controlled_vs_baseline_no_context | distinct1 | -0.0554 | (-0.0731, -0.0373) | 1.0000 | -0.0554 | (-0.0695, -0.0382) | 1.0000 |
| controlled_vs_baseline_no_context | length_score | -0.2167 | (-0.3611, -0.0694) | 0.9990 | -0.2167 | (-0.3969, -0.0190) | 0.9870 |
| controlled_vs_baseline_no_context | sentence_score | 0.0875 | (-0.0146, 0.1750) | 0.0497 | 0.0875 | (-0.0250, 0.2000) | 0.0750 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0560 | (0.0218, 0.0940) | 0.0000 | 0.0560 | (0.0242, 0.0904) | 0.0007 |
| controlled_vs_baseline_no_context | overall_quality | 0.1649 | (0.1373, 0.1904) | 0.0000 | 0.1649 | (0.1407, 0.1961) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0381 | (-0.0861, 0.0111) | 0.9357 | -0.0381 | (-0.0899, 0.0075) | 0.9513 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0111 | (-0.0984, 0.0768) | 0.6087 | -0.0111 | (-0.1412, 0.0693) | 0.5607 |
| controlled_alt_vs_controlled_default | naturalness | 0.0350 | (-0.0031, 0.0720) | 0.0333 | 0.0350 | (0.0147, 0.0561) | 0.0000 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0513 | (-0.1154, 0.0131) | 0.9380 | -0.0513 | (-0.1159, 0.0091) | 0.9470 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0071 | (-0.0302, 0.0171) | 0.7120 | -0.0071 | (-0.0312, 0.0154) | 0.7130 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0179 | (-0.1246, 0.0976) | 0.6340 | -0.0179 | (-0.1783, 0.0859) | 0.5543 |
| controlled_alt_vs_controlled_default | persona_style | 0.0158 | (-0.0243, 0.0622) | 0.2450 | 0.0158 | (-0.0137, 0.0545) | 0.1893 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0088 | (-0.0114, 0.0294) | 0.1887 | 0.0088 | (-0.0022, 0.0194) | 0.0567 |
| controlled_alt_vs_controlled_default | length_score | 0.1681 | (0.0083, 0.3181) | 0.0207 | 0.1681 | (0.0580, 0.2928) | 0.0000 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0292 | (-0.1021, 0.0437) | 0.8597 | -0.0292 | (-0.1077, 0.0477) | 0.8103 |
| controlled_alt_vs_controlled_default | bertscore_f1 | -0.0088 | (-0.0422, 0.0239) | 0.6987 | -0.0088 | (-0.0439, 0.0207) | 0.6747 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0142 | (-0.0440, 0.0153) | 0.8220 | -0.0142 | (-0.0561, 0.0180) | 0.7767 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.2205 | (0.1844, 0.2560) | 0.0000 | 0.2205 | (0.1927, 0.2515) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.2226 | (0.1462, 0.3017) | 0.0000 | 0.2226 | (0.1611, 0.2719) | 0.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0583 | (0.0184, 0.0977) | 0.0033 | 0.0583 | (0.0116, 0.0900) | 0.0080 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2859 | (0.2398, 0.3318) | 0.0000 | 0.2859 | (0.2468, 0.3325) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0679 | (0.0496, 0.0895) | 0.0000 | 0.0679 | (0.0528, 0.0876) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.2464 | (0.1579, 0.3379) | 0.0000 | 0.2464 | (0.1861, 0.3013) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 0.1275 | (0.0343, 0.2335) | 0.0027 | 0.1275 | (0.0185, 0.2763) | 0.0037 |
| controlled_alt_vs_proposed_raw | distinct1 | 0.0057 | (-0.0138, 0.0254) | 0.2873 | 0.0057 | (-0.0119, 0.0177) | 0.2287 |
| controlled_alt_vs_proposed_raw | length_score | 0.2069 | (0.0639, 0.3542) | 0.0017 | 0.2069 | (0.0667, 0.3104) | 0.0043 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1458 | (0.0437, 0.2333) | 0.0047 | 0.1458 | (0.0389, 0.2240) | 0.0067 |
| controlled_alt_vs_proposed_raw | bertscore_f1 | 0.0452 | (0.0156, 0.0739) | 0.0013 | 0.0452 | (0.0246, 0.0723) | 0.0000 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1774 | (0.1401, 0.2128) | 0.0000 | 0.1774 | (0.1523, 0.2009) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2401 | (0.2128, 0.2691) | 0.0000 | 0.2401 | (0.2135, 0.2728) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1522 | (0.0666, 0.2454) | 0.0000 | 0.1522 | (0.0364, 0.2354) | 0.0050 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0594 | (0.0097, 0.1069) | 0.0097 | 0.0594 | (-0.0017, 0.1076) | 0.0277 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3124 | (0.2765, 0.3513) | 0.0000 | 0.3124 | (0.2765, 0.3578) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0716 | (0.0535, 0.0923) | 0.0000 | 0.0716 | (0.0626, 0.0837) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1649 | (0.0658, 0.2788) | 0.0010 | 0.1649 | (0.0310, 0.2729) | 0.0070 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.1014 | (0.0104, 0.1992) | 0.0150 | 0.1014 | (-0.0068, 0.2257) | 0.0417 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0066 | (-0.0140, 0.0262) | 0.2567 | 0.0066 | (-0.0192, 0.0274) | 0.3070 |
| controlled_alt_vs_candidate_no_context | length_score | 0.2181 | (0.0306, 0.3944) | 0.0123 | 0.2181 | (0.0188, 0.3816) | 0.0167 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1313 | (0.0292, 0.2188) | 0.0063 | 0.1313 | (0.0000, 0.2293) | 0.0297 |
| controlled_alt_vs_candidate_no_context | bertscore_f1 | 0.0578 | (0.0291, 0.0867) | 0.0000 | 0.0578 | (0.0307, 0.0963) | 0.0000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1640 | (0.1274, 0.2006) | 0.0000 | 0.1640 | (0.1220, 0.1949) | 0.0000 |
| controlled_alt_vs_baseline_no_context | context_relevance | 0.2322 | (0.1994, 0.2651) | 0.0000 | 0.2322 | (0.2033, 0.2668) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 0.1677 | (0.0950, 0.2458) | 0.0000 | 0.1677 | (0.0742, 0.2424) | 0.0007 |
| controlled_alt_vs_baseline_no_context | naturalness | -0.0225 | (-0.0516, 0.0092) | 0.9190 | -0.0225 | (-0.0541, 0.0232) | 0.8503 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 0.3003 | (0.2564, 0.3479) | 0.0000 | 0.3003 | (0.2598, 0.3519) | 0.0000 |
| controlled_alt_vs_baseline_no_context | context_overlap | 0.0734 | (0.0571, 0.0951) | 0.0000 | 0.0734 | (0.0598, 0.0934) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 0.2036 | (0.1121, 0.3020) | 0.0000 | 0.2036 | (0.0769, 0.2975) | 0.0003 |
| controlled_alt_vs_baseline_no_context | persona_style | 0.0242 | (-0.0164, 0.0653) | 0.1187 | 0.0242 | (0.0000, 0.0718) | 0.0227 |
| controlled_alt_vs_baseline_no_context | distinct1 | -0.0466 | (-0.0638, -0.0289) | 1.0000 | -0.0466 | (-0.0608, -0.0255) | 1.0000 |
| controlled_alt_vs_baseline_no_context | length_score | -0.0486 | (-0.1958, 0.0917) | 0.7580 | -0.0486 | (-0.1711, 0.1246) | 0.7250 |
| controlled_alt_vs_baseline_no_context | sentence_score | 0.0583 | (-0.0292, 0.1458) | 0.1207 | 0.0583 | (-0.0280, 0.1575) | 0.1320 |
| controlled_alt_vs_baseline_no_context | bertscore_f1 | 0.0472 | (0.0127, 0.0807) | 0.0027 | 0.0472 | (0.0171, 0.0852) | 0.0010 |
| controlled_alt_vs_baseline_no_context | overall_quality | 0.1507 | (0.1229, 0.1774) | 0.0000 | 0.1507 | (0.1214, 0.1769) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2703 | (0.2212, 0.3229) | 0.0000 | 0.2703 | (0.2283, 0.3323) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1788 | (0.1352, 0.2217) | 0.0000 | 0.1788 | (0.1444, 0.2234) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0575 | (-0.0899, -0.0248) | 0.9997 | -0.0575 | (-0.0952, -0.0114) | 0.9937 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3516 | (0.2866, 0.4211) | 0.0000 | 0.3516 | (0.2959, 0.4348) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0805 | (0.0608, 0.1019) | 0.0000 | 0.0805 | (0.0626, 0.1034) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2214 | (0.1675, 0.2780) | 0.0000 | 0.2214 | (0.1778, 0.2740) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0084 | (-0.0330, 0.0506) | 0.3387 | 0.0084 | (-0.0303, 0.0482) | 0.3573 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0554 | (-0.0734, -0.0380) | 1.0000 | -0.0554 | (-0.0699, -0.0371) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | -0.2167 | (-0.3722, -0.0694) | 1.0000 | -0.2167 | (-0.3880, -0.0185) | 0.9847 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0875 | (0.0000, 0.1750) | 0.0467 | 0.0875 | (-0.0304, 0.2027) | 0.0717 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0560 | (0.0214, 0.0918) | 0.0007 | 0.0560 | (0.0237, 0.0917) | 0.0003 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1649 | (0.1382, 0.1904) | 0.0000 | 0.1649 | (0.1409, 0.1952) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 10 | 4 | 10 | 0.6250 | 0.7143 |
| proposed_vs_candidate_no_context | persona_consistency | 4 | 7 | 13 | 0.4375 | 0.3636 |
| proposed_vs_candidate_no_context | naturalness | 7 | 7 | 10 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 5 | 1 | 18 | 0.5833 | 0.8333 |
| proposed_vs_candidate_no_context | context_overlap | 9 | 4 | 11 | 0.6042 | 0.6923 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 3 | 6 | 15 | 0.4375 | 0.3333 |
| proposed_vs_candidate_no_context | persona_style | 2 | 5 | 17 | 0.4375 | 0.2857 |
| proposed_vs_candidate_no_context | distinct1 | 6 | 7 | 11 | 0.4792 | 0.4615 |
| proposed_vs_candidate_no_context | length_score | 8 | 6 | 10 | 0.5417 | 0.5714 |
| proposed_vs_candidate_no_context | sentence_score | 3 | 4 | 17 | 0.4792 | 0.4286 |
| proposed_vs_candidate_no_context | bertscore_f1 | 10 | 7 | 7 | 0.5625 | 0.5882 |
| proposed_vs_candidate_no_context | overall_quality | 9 | 8 | 7 | 0.5208 | 0.5294 |
| proposed_vs_baseline_no_context | context_relevance | 13 | 11 | 0 | 0.5417 | 0.5417 |
| proposed_vs_baseline_no_context | persona_consistency | 6 | 9 | 9 | 0.4375 | 0.4000 |
| proposed_vs_baseline_no_context | naturalness | 6 | 18 | 0 | 0.2500 | 0.2500 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 5 | 3 | 16 | 0.5417 | 0.6250 |
| proposed_vs_baseline_no_context | context_overlap | 12 | 11 | 1 | 0.5208 | 0.5217 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 5 | 5 | 14 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context | persona_style | 2 | 6 | 16 | 0.4167 | 0.2500 |
| proposed_vs_baseline_no_context | distinct1 | 2 | 21 | 1 | 0.1042 | 0.0870 |
| proposed_vs_baseline_no_context | length_score | 6 | 18 | 0 | 0.2500 | 0.2500 |
| proposed_vs_baseline_no_context | sentence_score | 5 | 11 | 8 | 0.3750 | 0.3125 |
| proposed_vs_baseline_no_context | bertscore_f1 | 14 | 10 | 0 | 0.5833 | 0.5833 |
| proposed_vs_baseline_no_context | overall_quality | 9 | 15 | 0 | 0.3750 | 0.3750 |
| controlled_vs_proposed_raw | context_relevance | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_vs_proposed_raw | persona_consistency | 19 | 1 | 4 | 0.8750 | 0.9500 |
| controlled_vs_proposed_raw | naturalness | 14 | 9 | 1 | 0.6042 | 0.6087 |
| controlled_vs_proposed_raw | context_keyword_coverage | 23 | 0 | 1 | 0.9792 | 1.0000 |
| controlled_vs_proposed_raw | context_overlap | 22 | 2 | 0 | 0.9167 | 0.9167 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 19 | 0 | 5 | 0.8958 | 1.0000 |
| controlled_vs_proposed_raw | persona_style | 7 | 2 | 15 | 0.6042 | 0.7778 |
| controlled_vs_proposed_raw | distinct1 | 10 | 12 | 2 | 0.4583 | 0.4545 |
| controlled_vs_proposed_raw | length_score | 10 | 12 | 2 | 0.4583 | 0.4545 |
| controlled_vs_proposed_raw | sentence_score | 14 | 2 | 8 | 0.7500 | 0.8750 |
| controlled_vs_proposed_raw | bertscore_f1 | 18 | 6 | 0 | 0.7500 | 0.7500 |
| controlled_vs_proposed_raw | overall_quality | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_relevance | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 20 | 2 | 2 | 0.8750 | 0.9091 |
| controlled_vs_candidate_no_context | naturalness | 13 | 11 | 0 | 0.5417 | 0.5417 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 20 | 2 | 2 | 0.8750 | 0.9091 |
| controlled_vs_candidate_no_context | persona_style | 6 | 3 | 15 | 0.5625 | 0.6667 |
| controlled_vs_candidate_no_context | distinct1 | 11 | 13 | 0 | 0.4583 | 0.4583 |
| controlled_vs_candidate_no_context | length_score | 11 | 12 | 1 | 0.4792 | 0.4783 |
| controlled_vs_candidate_no_context | sentence_score | 12 | 1 | 11 | 0.7292 | 0.9231 |
| controlled_vs_candidate_no_context | bertscore_f1 | 18 | 6 | 0 | 0.7500 | 0.7500 |
| controlled_vs_candidate_no_context | overall_quality | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_vs_baseline_no_context | context_relevance | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 20 | 0 | 4 | 0.9167 | 1.0000 |
| controlled_vs_baseline_no_context | naturalness | 6 | 18 | 0 | 0.2500 | 0.2500 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | context_overlap | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 20 | 0 | 4 | 0.9167 | 1.0000 |
| controlled_vs_baseline_no_context | persona_style | 3 | 2 | 19 | 0.5208 | 0.6000 |
| controlled_vs_baseline_no_context | distinct1 | 2 | 21 | 1 | 0.1042 | 0.0870 |
| controlled_vs_baseline_no_context | length_score | 8 | 15 | 1 | 0.3542 | 0.3478 |
| controlled_vs_baseline_no_context | sentence_score | 9 | 3 | 12 | 0.6250 | 0.7500 |
| controlled_vs_baseline_no_context | bertscore_f1 | 17 | 7 | 0 | 0.7083 | 0.7083 |
| controlled_vs_baseline_no_context | overall_quality | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 10 | 13 | 1 | 0.4375 | 0.4348 |
| controlled_alt_vs_controlled_default | persona_consistency | 5 | 7 | 12 | 0.4583 | 0.4167 |
| controlled_alt_vs_controlled_default | naturalness | 15 | 8 | 1 | 0.6458 | 0.6522 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 6 | 11 | 7 | 0.3958 | 0.3529 |
| controlled_alt_vs_controlled_default | context_overlap | 10 | 13 | 1 | 0.4375 | 0.4348 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 5 | 6 | 13 | 0.4792 | 0.4545 |
| controlled_alt_vs_controlled_default | persona_style | 3 | 2 | 19 | 0.5208 | 0.6000 |
| controlled_alt_vs_controlled_default | distinct1 | 12 | 11 | 1 | 0.5208 | 0.5217 |
| controlled_alt_vs_controlled_default | length_score | 16 | 6 | 2 | 0.7083 | 0.7273 |
| controlled_alt_vs_controlled_default | sentence_score | 2 | 4 | 18 | 0.4583 | 0.3333 |
| controlled_alt_vs_controlled_default | bertscore_f1 | 11 | 12 | 1 | 0.4792 | 0.4783 |
| controlled_alt_vs_controlled_default | overall_quality | 10 | 13 | 1 | 0.4375 | 0.4348 |
| controlled_alt_vs_proposed_raw | context_relevance | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_alt_vs_proposed_raw | persona_consistency | 19 | 2 | 3 | 0.8542 | 0.9048 |
| controlled_alt_vs_proposed_raw | naturalness | 17 | 7 | 0 | 0.7083 | 0.7083 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 23 | 0 | 1 | 0.9792 | 1.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 22 | 2 | 0 | 0.9167 | 0.9167 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 19 | 1 | 4 | 0.8750 | 0.9500 |
| controlled_alt_vs_proposed_raw | persona_style | 8 | 1 | 15 | 0.6458 | 0.8889 |
| controlled_alt_vs_proposed_raw | distinct1 | 13 | 11 | 0 | 0.5417 | 0.5417 |
| controlled_alt_vs_proposed_raw | length_score | 14 | 10 | 0 | 0.5833 | 0.5833 |
| controlled_alt_vs_proposed_raw | sentence_score | 13 | 3 | 8 | 0.7083 | 0.8125 |
| controlled_alt_vs_proposed_raw | bertscore_f1 | 16 | 8 | 0 | 0.6667 | 0.6667 |
| controlled_alt_vs_proposed_raw | overall_quality | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_alt_vs_candidate_no_context | context_relevance | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 16 | 5 | 3 | 0.7292 | 0.7619 |
| controlled_alt_vs_candidate_no_context | naturalness | 17 | 7 | 0 | 0.7083 | 0.7083 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 16 | 5 | 3 | 0.7292 | 0.7619 |
| controlled_alt_vs_candidate_no_context | persona_style | 6 | 2 | 16 | 0.5833 | 0.7500 |
| controlled_alt_vs_candidate_no_context | distinct1 | 15 | 9 | 0 | 0.6250 | 0.6250 |
| controlled_alt_vs_candidate_no_context | length_score | 15 | 9 | 0 | 0.6250 | 0.6250 |
| controlled_alt_vs_candidate_no_context | sentence_score | 12 | 3 | 9 | 0.6875 | 0.8000 |
| controlled_alt_vs_candidate_no_context | bertscore_f1 | 17 | 7 | 0 | 0.7083 | 0.7083 |
| controlled_alt_vs_candidate_no_context | overall_quality | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context | context_relevance | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 19 | 2 | 3 | 0.8542 | 0.9048 |
| controlled_alt_vs_baseline_no_context | naturalness | 10 | 14 | 0 | 0.4167 | 0.4167 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context | context_overlap | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 18 | 2 | 4 | 0.8333 | 0.9000 |
| controlled_alt_vs_baseline_no_context | persona_style | 4 | 2 | 18 | 0.5417 | 0.6667 |
| controlled_alt_vs_baseline_no_context | distinct1 | 3 | 20 | 1 | 0.1458 | 0.1304 |
| controlled_alt_vs_baseline_no_context | length_score | 12 | 11 | 1 | 0.5208 | 0.5217 |
| controlled_alt_vs_baseline_no_context | sentence_score | 7 | 3 | 14 | 0.5833 | 0.7000 |
| controlled_alt_vs_baseline_no_context | bertscore_f1 | 18 | 6 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_baseline_no_context | overall_quality | 24 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 24 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 20 | 0 | 4 | 0.9167 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 6 | 18 | 0 | 0.2500 | 0.2500 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 24 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 24 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 20 | 0 | 4 | 0.9167 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 3 | 2 | 19 | 0.5208 | 0.6000 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | 2 | 21 | 1 | 0.1042 | 0.0870 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 8 | 15 | 1 | 0.3542 | 0.3478 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 9 | 3 | 12 | 0.6250 | 0.7500 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 17 | 7 | 0 | 0.7083 | 0.7083 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 24 | 0 | 0 | 1.0000 | 1.0000 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.1250 | 0.3750 | 0.6250 |
| proposed_contextual_controlled_hb | 0.0000 | 0.0000 | 0.2083 | 0.3750 | 0.6250 |
| proposed_contextual | 0.0000 | 0.0000 | 0.5833 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.5833 | 0.0000 | 0.0000 |
| baseline_no_context | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `8`
- Unique template signatures: `22`
- Template signature ratio: `0.9167`
- Effective sample size by source clustering: `7.20`
- Effective sample size by template-signature clustering: `20.57`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: enabled.

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
# Proposal Alignment Evaluation Report

- Run ID: `20260306T002050Z`
- Generated: `2026-03-06T00:27:03.586535+00:00`
- Scenarios: `artifacts\proposal_control_tuning\preflight_hybrid_v2\20260306T001357Z\seed_runs\seed_31\20260306T002050Z\scenarios.jsonl`
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
| proposed_contextual_controlled | 0.2381 (0.2037, 0.2727) | 0.3336 (0.2863, 0.3960) | 0.8769 (0.8348, 0.9120) | 0.3664 (0.3466, 0.3896) | 0.0883 |
| proposed_contextual_controlled_alt | 0.2889 (0.2436, 0.3329) | 0.3169 (0.2697, 0.3826) | 0.8609 (0.8302, 0.8912) | 0.3804 (0.3611, 0.4001) | 0.0959 |
| proposed_contextual | 0.1130 (0.0647, 0.1673) | 0.1506 (0.1042, 0.2006) | 0.8092 (0.7802, 0.8396) | 0.2421 (0.2108, 0.2771) | 0.0712 |
| candidate_no_context | 0.0220 (0.0131, 0.0325) | 0.1696 (0.1249, 0.2229) | 0.8341 (0.7958, 0.8723) | 0.2119 (0.1931, 0.2328) | 0.0395 |
| baseline_no_context | 0.0347 (0.0192, 0.0536) | 0.1878 (0.1422, 0.2381) | 0.8986 (0.8782, 0.9186) | 0.2358 (0.2170, 0.2542) | 0.0486 |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0909 | 4.1310 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0190 | -0.1119 |
| proposed_vs_candidate_no_context | naturalness | -0.0249 | -0.0299 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.1189 | 7.6585 |
| proposed_vs_candidate_no_context | context_overlap | 0.0256 | 0.6894 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0143 | -0.2136 |
| proposed_vs_candidate_no_context | persona_style | -0.0378 | -0.0651 |
| proposed_vs_candidate_no_context | distinct1 | -0.0047 | -0.0049 |
| proposed_vs_candidate_no_context | length_score | -0.0944 | -0.2455 |
| proposed_vs_candidate_no_context | sentence_score | -0.0417 | -0.0524 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0317 | 0.8014 |
| proposed_vs_candidate_no_context | overall_quality | 0.0303 | 0.1429 |
| proposed_vs_baseline_no_context | context_relevance | 0.0782 | 2.2531 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0372 | -0.1982 |
| proposed_vs_baseline_no_context | naturalness | -0.0894 | -0.0995 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.1010 | 3.0189 |
| proposed_vs_baseline_no_context | context_overlap | 0.0251 | 0.6658 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0361 | -0.4072 |
| proposed_vs_baseline_no_context | persona_style | -0.0417 | -0.0714 |
| proposed_vs_baseline_no_context | distinct1 | -0.0321 | -0.0331 |
| proposed_vs_baseline_no_context | length_score | -0.3111 | -0.5173 |
| proposed_vs_baseline_no_context | sentence_score | -0.1437 | -0.1601 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0226 | 0.4664 |
| proposed_vs_baseline_no_context | overall_quality | 0.0063 | 0.0268 |
| controlled_vs_proposed_raw | context_relevance | 0.1251 | 1.1078 |
| controlled_vs_proposed_raw | persona_consistency | 0.1830 | 1.2157 |
| controlled_vs_proposed_raw | naturalness | 0.0677 | 0.0836 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.1576 | 1.1723 |
| controlled_vs_proposed_raw | context_overlap | 0.0493 | 0.7855 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2087 | 3.9698 |
| controlled_vs_proposed_raw | persona_style | 0.0803 | 0.1480 |
| controlled_vs_proposed_raw | distinct1 | -0.0000 | -0.0000 |
| controlled_vs_proposed_raw | length_score | 0.2861 | 0.9856 |
| controlled_vs_proposed_raw | sentence_score | 0.1292 | 0.1713 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0171 | 0.2401 |
| controlled_vs_proposed_raw | overall_quality | 0.1242 | 0.5131 |
| controlled_vs_candidate_no_context | context_relevance | 0.2161 | 9.8153 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1641 | 0.9676 |
| controlled_vs_candidate_no_context | naturalness | 0.0427 | 0.0512 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.2766 | 17.8089 |
| controlled_vs_candidate_no_context | context_overlap | 0.0749 | 2.0164 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1944 | 2.9080 |
| controlled_vs_candidate_no_context | persona_style | 0.0425 | 0.0733 |
| controlled_vs_candidate_no_context | distinct1 | -0.0047 | -0.0050 |
| controlled_vs_candidate_no_context | length_score | 0.1917 | 0.4982 |
| controlled_vs_candidate_no_context | sentence_score | 0.0875 | 0.1099 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0488 | 1.2340 |
| controlled_vs_candidate_no_context | overall_quality | 0.1545 | 0.7293 |
| controlled_vs_baseline_no_context | context_relevance | 0.2034 | 5.8570 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1458 | 0.7764 |
| controlled_vs_baseline_no_context | naturalness | -0.0218 | -0.0242 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.2586 | 7.7302 |
| controlled_vs_baseline_no_context | context_overlap | 0.0744 | 1.9743 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1726 | 1.9463 |
| controlled_vs_baseline_no_context | persona_style | 0.0386 | 0.0661 |
| controlled_vs_baseline_no_context | distinct1 | -0.0322 | -0.0331 |
| controlled_vs_baseline_no_context | length_score | -0.0250 | -0.0416 |
| controlled_vs_baseline_no_context | sentence_score | -0.0146 | -0.0162 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0397 | 0.8184 |
| controlled_vs_baseline_no_context | overall_quality | 0.1305 | 0.5536 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0508 | 0.2134 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0167 | -0.0501 |
| controlled_alt_vs_controlled_default | naturalness | -0.0160 | -0.0182 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0750 | 0.2568 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0056 | -0.0501 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0147 | -0.0562 |
| controlled_alt_vs_controlled_default | persona_style | -0.0248 | -0.0398 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0010 | 0.0010 |
| controlled_alt_vs_controlled_default | length_score | -0.1000 | -0.1735 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0146 | 0.0165 |
| controlled_alt_vs_controlled_default | bertscore_f1 | 0.0076 | 0.0862 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0140 | 0.0383 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1759 | 1.5577 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1663 | 1.1047 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0517 | 0.0639 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2326 | 1.7300 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0437 | 0.6959 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1940 | 3.6906 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0555 | 0.1023 |
| controlled_alt_vs_proposed_raw | distinct1 | 0.0009 | 0.0010 |
| controlled_alt_vs_proposed_raw | length_score | 0.1861 | 0.6411 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1437 | 0.1906 |
| controlled_alt_vs_proposed_raw | bertscore_f1 | 0.0247 | 0.3470 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1383 | 0.5711 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2669 | 12.1236 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1474 | 0.8691 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0268 | 0.0321 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3516 | 22.6382 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0693 | 1.8651 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1798 | 2.6884 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0177 | 0.0306 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0037 | -0.0040 |
| controlled_alt_vs_candidate_no_context | length_score | 0.0917 | 0.2383 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1021 | 0.1283 |
| controlled_alt_vs_candidate_no_context | bertscore_f1 | 0.0564 | 1.4265 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1685 | 0.7956 |
| controlled_alt_vs_baseline_no_context | context_relevance | 0.2542 | 7.3204 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 0.1291 | 0.6875 |
| controlled_alt_vs_baseline_no_context | naturalness | -0.0378 | -0.0420 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 0.3336 | 9.9717 |
| controlled_alt_vs_baseline_no_context | context_overlap | 0.0687 | 1.8252 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 0.1579 | 1.7808 |
| controlled_alt_vs_baseline_no_context | persona_style | 0.0138 | 0.0236 |
| controlled_alt_vs_baseline_no_context | distinct1 | -0.0312 | -0.0321 |
| controlled_alt_vs_baseline_no_context | length_score | -0.1250 | -0.2079 |
| controlled_alt_vs_baseline_no_context | sentence_score | 0.0000 | 0.0000 |
| controlled_alt_vs_baseline_no_context | bertscore_f1 | 0.0474 | 0.9752 |
| controlled_alt_vs_baseline_no_context | overall_quality | 0.1446 | 0.6131 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2034 | 5.8570 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1458 | 0.7764 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0218 | -0.0242 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.2586 | 7.7302 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0744 | 1.9743 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1726 | 1.9463 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0386 | 0.0661 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0322 | -0.0331 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | -0.0250 | -0.0416 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | -0.0146 | -0.0162 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0397 | 0.8184 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1305 | 0.5536 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0909 | (0.0395, 0.1468) | 0.0000 | 0.0909 | (0.0150, 0.1619) | 0.0073 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0190 | (-0.0945, 0.0527) | 0.6873 | -0.0190 | (-0.0923, 0.0457) | 0.7113 |
| proposed_vs_candidate_no_context | naturalness | -0.0249 | (-0.0658, 0.0153) | 0.8880 | -0.0249 | (-0.0785, 0.0192) | 0.8520 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.1189 | (0.0561, 0.1919) | 0.0000 | 0.1189 | (0.0187, 0.2101) | 0.0080 |
| proposed_vs_candidate_no_context | context_overlap | 0.0256 | (0.0107, 0.0438) | 0.0000 | 0.0256 | (0.0065, 0.0422) | 0.0000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0143 | (-0.0992, 0.0681) | 0.6197 | -0.0143 | (-0.0976, 0.0650) | 0.6543 |
| proposed_vs_candidate_no_context | persona_style | -0.0378 | (-0.1029, 0.0000) | 1.0000 | -0.0378 | (-0.1109, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | distinct1 | -0.0047 | (-0.0250, 0.0171) | 0.6630 | -0.0047 | (-0.0264, 0.0167) | 0.6830 |
| proposed_vs_candidate_no_context | length_score | -0.0944 | (-0.2347, 0.0528) | 0.8997 | -0.0944 | (-0.2695, 0.0636) | 0.8720 |
| proposed_vs_candidate_no_context | sentence_score | -0.0417 | (-0.1271, 0.0437) | 0.8413 | -0.0417 | (-0.1546, 0.0519) | 0.7877 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0317 | (0.0055, 0.0602) | 0.0080 | 0.0317 | (-0.0088, 0.0759) | 0.0753 |
| proposed_vs_candidate_no_context | overall_quality | 0.0303 | (-0.0063, 0.0692) | 0.0590 | 0.0303 | (-0.0203, 0.0638) | 0.1187 |
| proposed_vs_baseline_no_context | context_relevance | 0.0782 | (0.0300, 0.1341) | 0.0000 | 0.0782 | (0.0087, 0.1510) | 0.0117 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0372 | (-0.1051, 0.0223) | 0.8623 | -0.0372 | (-0.1508, 0.0328) | 0.8027 |
| proposed_vs_baseline_no_context | naturalness | -0.0894 | (-0.1269, -0.0494) | 1.0000 | -0.0894 | (-0.1292, -0.0539) | 1.0000 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.1010 | (0.0363, 0.1758) | 0.0000 | 0.1010 | (0.0051, 0.1984) | 0.0170 |
| proposed_vs_baseline_no_context | context_overlap | 0.0251 | (0.0083, 0.0444) | 0.0013 | 0.0251 | (0.0043, 0.0428) | 0.0063 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0361 | (-0.1189, 0.0337) | 0.8227 | -0.0361 | (-0.1752, 0.0473) | 0.7303 |
| proposed_vs_baseline_no_context | persona_style | -0.0417 | (-0.1312, 0.0329) | 0.8583 | -0.0417 | (-0.1866, 0.0495) | 0.7453 |
| proposed_vs_baseline_no_context | distinct1 | -0.0321 | (-0.0473, -0.0155) | 1.0000 | -0.0321 | (-0.0483, -0.0205) | 1.0000 |
| proposed_vs_baseline_no_context | length_score | -0.3111 | (-0.4514, -0.1652) | 1.0000 | -0.3111 | (-0.4470, -0.1936) | 1.0000 |
| proposed_vs_baseline_no_context | sentence_score | -0.1437 | (-0.2458, -0.0396) | 0.9957 | -0.1437 | (-0.2860, -0.0146) | 0.9923 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0226 | (-0.0107, 0.0592) | 0.1067 | 0.0226 | (-0.0275, 0.0723) | 0.1907 |
| proposed_vs_baseline_no_context | overall_quality | 0.0063 | (-0.0337, 0.0450) | 0.3793 | 0.0063 | (-0.0556, 0.0535) | 0.4237 |
| controlled_vs_proposed_raw | context_relevance | 0.1251 | (0.0682, 0.1747) | 0.0000 | 0.1251 | (0.0685, 0.1969) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.1830 | (0.1085, 0.2574) | 0.0000 | 0.1830 | (0.1038, 0.2848) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0677 | (0.0205, 0.1144) | 0.0020 | 0.0677 | (0.0096, 0.1319) | 0.0110 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.1576 | (0.0888, 0.2251) | 0.0000 | 0.1576 | (0.0818, 0.2495) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0493 | (0.0245, 0.0757) | 0.0000 | 0.0493 | (0.0245, 0.0843) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2087 | (0.1188, 0.2982) | 0.0000 | 0.2087 | (0.1167, 0.3291) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.0803 | (-0.0035, 0.1745) | 0.0337 | 0.0803 | (-0.0008, 0.2396) | 0.0347 |
| controlled_vs_proposed_raw | distinct1 | -0.0000 | (-0.0314, 0.0275) | 0.4810 | -0.0000 | (-0.0272, 0.0218) | 0.5050 |
| controlled_vs_proposed_raw | length_score | 0.2861 | (0.1208, 0.4486) | 0.0003 | 0.2861 | (0.0567, 0.5111) | 0.0097 |
| controlled_vs_proposed_raw | sentence_score | 0.1292 | (0.0417, 0.2167) | 0.0037 | 0.1292 | (0.0269, 0.2425) | 0.0113 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0171 | (-0.0118, 0.0451) | 0.1180 | 0.0171 | (-0.0121, 0.0580) | 0.1523 |
| controlled_vs_proposed_raw | overall_quality | 0.1242 | (0.0814, 0.1647) | 0.0000 | 0.1242 | (0.0762, 0.1856) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2161 | (0.1845, 0.2504) | 0.0000 | 0.2161 | (0.1740, 0.2508) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1641 | (0.0924, 0.2389) | 0.0000 | 0.1641 | (0.1091, 0.2415) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.0427 | (-0.0147, 0.1014) | 0.0783 | 0.0427 | (-0.0303, 0.1092) | 0.1313 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.2766 | (0.2293, 0.3227) | 0.0000 | 0.2766 | (0.2201, 0.3224) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0749 | (0.0540, 0.0982) | 0.0000 | 0.0749 | (0.0549, 0.0974) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1944 | (0.1079, 0.2827) | 0.0000 | 0.1944 | (0.1237, 0.2900) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.0425 | (-0.0265, 0.1315) | 0.1567 | 0.0425 | (-0.0108, 0.1339) | 0.1020 |
| controlled_vs_candidate_no_context | distinct1 | -0.0047 | (-0.0399, 0.0241) | 0.5910 | -0.0047 | (-0.0370, 0.0174) | 0.6257 |
| controlled_vs_candidate_no_context | length_score | 0.1917 | (-0.0167, 0.3875) | 0.0357 | 0.1917 | (-0.0970, 0.4526) | 0.0903 |
| controlled_vs_candidate_no_context | sentence_score | 0.0875 | (0.0000, 0.1750) | 0.0413 | 0.0875 | (-0.0140, 0.1896) | 0.0590 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0488 | (0.0205, 0.0781) | 0.0000 | 0.0488 | (0.0207, 0.0868) | 0.0000 |
| controlled_vs_candidate_no_context | overall_quality | 0.1545 | (0.1231, 0.1867) | 0.0000 | 0.1545 | (0.1212, 0.1892) | 0.0000 |
| controlled_vs_baseline_no_context | context_relevance | 0.2034 | (0.1693, 0.2403) | 0.0000 | 0.2034 | (0.1614, 0.2415) | 0.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1458 | (0.1001, 0.1986) | 0.0000 | 0.1458 | (0.1080, 0.1880) | 0.0000 |
| controlled_vs_baseline_no_context | naturalness | -0.0218 | (-0.0726, 0.0235) | 0.8107 | -0.0218 | (-0.0696, 0.0152) | 0.8533 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.2586 | (0.2112, 0.3071) | 0.0000 | 0.2586 | (0.1950, 0.3122) | 0.0000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0744 | (0.0526, 0.1003) | 0.0000 | 0.0744 | (0.0572, 0.0971) | 0.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1726 | (0.1163, 0.2409) | 0.0000 | 0.1726 | (0.1221, 0.2268) | 0.0000 |
| controlled_vs_baseline_no_context | persona_style | 0.0386 | (-0.0147, 0.1058) | 0.0887 | 0.0386 | (0.0086, 0.0882) | 0.0210 |
| controlled_vs_baseline_no_context | distinct1 | -0.0322 | (-0.0574, -0.0086) | 0.9983 | -0.0322 | (-0.0562, -0.0134) | 0.9997 |
| controlled_vs_baseline_no_context | length_score | -0.0250 | (-0.2070, 0.1472) | 0.6017 | -0.0250 | (-0.1971, 0.1218) | 0.5963 |
| controlled_vs_baseline_no_context | sentence_score | -0.0146 | (-0.1021, 0.0729) | 0.6783 | -0.0146 | (-0.1217, 0.0875) | 0.6497 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0397 | (0.0114, 0.0675) | 0.0023 | 0.0397 | (0.0110, 0.0817) | 0.0003 |
| controlled_vs_baseline_no_context | overall_quality | 0.1305 | (0.1075, 0.1557) | 0.0000 | 0.1305 | (0.1033, 0.1522) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0508 | (0.0045, 0.0986) | 0.0147 | 0.0508 | (0.0116, 0.1008) | 0.0043 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0167 | (-0.0977, 0.0654) | 0.6823 | -0.0167 | (-0.1046, 0.0609) | 0.6450 |
| controlled_alt_vs_controlled_default | naturalness | -0.0160 | (-0.0529, 0.0249) | 0.7697 | -0.0160 | (-0.0400, 0.0092) | 0.9000 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0750 | (0.0100, 0.1426) | 0.0123 | 0.0750 | (0.0211, 0.1412) | 0.0030 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0056 | (-0.0282, 0.0143) | 0.6817 | -0.0056 | (-0.0287, 0.0137) | 0.6723 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0147 | (-0.1083, 0.0827) | 0.6300 | -0.0147 | (-0.1235, 0.0821) | 0.6163 |
| controlled_alt_vs_controlled_default | persona_style | -0.0248 | (-0.0701, 0.0045) | 0.9113 | -0.0248 | (-0.0639, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0010 | (-0.0207, 0.0294) | 0.5050 | 0.0010 | (-0.0162, 0.0253) | 0.5037 |
| controlled_alt_vs_controlled_default | length_score | -0.1000 | (-0.2445, 0.0361) | 0.9210 | -0.1000 | (-0.2084, -0.0060) | 0.9817 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0146 | (-0.0875, 0.1167) | 0.4400 | 0.0146 | (-0.0648, 0.1167) | 0.4423 |
| controlled_alt_vs_controlled_default | bertscore_f1 | 0.0076 | (-0.0180, 0.0348) | 0.2917 | 0.0076 | (-0.0182, 0.0353) | 0.2883 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0140 | (-0.0124, 0.0387) | 0.1473 | 0.0140 | (-0.0089, 0.0318) | 0.0960 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1759 | (0.1117, 0.2409) | 0.0000 | 0.1759 | (0.0999, 0.2729) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1663 | (0.0962, 0.2417) | 0.0000 | 0.1663 | (0.1078, 0.2526) | 0.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0517 | (0.0104, 0.0923) | 0.0077 | 0.0517 | (-0.0060, 0.0991) | 0.0380 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2326 | (0.1474, 0.3172) | 0.0000 | 0.2326 | (0.1305, 0.3585) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0437 | (0.0257, 0.0623) | 0.0000 | 0.0437 | (0.0235, 0.0719) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1940 | (0.1133, 0.2863) | 0.0000 | 0.1940 | (0.1337, 0.2910) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0555 | (-0.0397, 0.1652) | 0.1500 | 0.0555 | (-0.0463, 0.2105) | 0.1990 |
| controlled_alt_vs_proposed_raw | distinct1 | 0.0009 | (-0.0187, 0.0194) | 0.4463 | 0.0009 | (-0.0245, 0.0222) | 0.4590 |
| controlled_alt_vs_proposed_raw | length_score | 0.1861 | (0.0124, 0.3514) | 0.0183 | 0.1861 | (-0.0533, 0.3883) | 0.0597 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1437 | (0.0417, 0.2437) | 0.0043 | 0.1437 | (0.0654, 0.2425) | 0.0000 |
| controlled_alt_vs_proposed_raw | bertscore_f1 | 0.0247 | (-0.0022, 0.0529) | 0.0367 | 0.0247 | (-0.0001, 0.0611) | 0.0253 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1383 | (0.0993, 0.1771) | 0.0000 | 0.1383 | (0.0908, 0.2011) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2669 | (0.2230, 0.3119) | 0.0000 | 0.2669 | (0.2054, 0.3264) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1474 | (0.0693, 0.2301) | 0.0000 | 0.1474 | (0.0794, 0.2309) | 0.0000 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0268 | (-0.0273, 0.0771) | 0.1670 | 0.0268 | (-0.0483, 0.0876) | 0.2357 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3516 | (0.2907, 0.4123) | 0.0000 | 0.3516 | (0.2671, 0.4320) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0693 | (0.0573, 0.0816) | 0.0000 | 0.0693 | (0.0568, 0.0816) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1798 | (0.0881, 0.2786) | 0.0000 | 0.1798 | (0.0919, 0.2785) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0177 | (-0.0650, 0.1143) | 0.3607 | 0.0177 | (-0.0549, 0.1176) | 0.3347 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0037 | (-0.0236, 0.0146) | 0.6443 | -0.0037 | (-0.0347, 0.0181) | 0.5977 |
| controlled_alt_vs_candidate_no_context | length_score | 0.0917 | (-0.1459, 0.3042) | 0.2077 | 0.0917 | (-0.2208, 0.3470) | 0.2707 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1021 | (-0.0146, 0.2042) | 0.0480 | 0.1021 | (0.0211, 0.1833) | 0.0103 |
| controlled_alt_vs_candidate_no_context | bertscore_f1 | 0.0564 | (0.0308, 0.0822) | 0.0000 | 0.0564 | (0.0322, 0.0890) | 0.0000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1685 | (0.1375, 0.1981) | 0.0000 | 0.1685 | (0.1426, 0.1943) | 0.0000 |
| controlled_alt_vs_baseline_no_context | context_relevance | 0.2542 | (0.2013, 0.3071) | 0.0000 | 0.2542 | (0.1850, 0.3151) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 0.1291 | (0.0564, 0.2136) | 0.0003 | 0.1291 | (0.0367, 0.1993) | 0.0043 |
| controlled_alt_vs_baseline_no_context | naturalness | -0.0378 | (-0.0739, -0.0027) | 0.9833 | -0.0378 | (-0.0804, -0.0071) | 0.9967 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 0.3336 | (0.2601, 0.4097) | 0.0000 | 0.3336 | (0.2363, 0.4222) | 0.0000 |
| controlled_alt_vs_baseline_no_context | context_overlap | 0.0687 | (0.0565, 0.0812) | 0.0000 | 0.0687 | (0.0590, 0.0795) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 0.1579 | (0.0682, 0.2568) | 0.0000 | 0.1579 | (0.0329, 0.2493) | 0.0087 |
| controlled_alt_vs_baseline_no_context | persona_style | 0.0138 | (-0.0573, 0.0859) | 0.3647 | 0.0138 | (-0.0408, 0.0685) | 0.2917 |
| controlled_alt_vs_baseline_no_context | distinct1 | -0.0312 | (-0.0501, -0.0127) | 0.9993 | -0.0312 | (-0.0559, -0.0112) | 1.0000 |
| controlled_alt_vs_baseline_no_context | length_score | -0.1250 | (-0.2833, 0.0264) | 0.9487 | -0.1250 | (-0.3226, 0.0269) | 0.9403 |
| controlled_alt_vs_baseline_no_context | sentence_score | 0.0000 | (-0.0729, 0.0729) | 0.5813 | 0.0000 | (-0.0778, 0.0833) | 0.5453 |
| controlled_alt_vs_baseline_no_context | bertscore_f1 | 0.0474 | (0.0160, 0.0819) | 0.0007 | 0.0474 | (0.0186, 0.0854) | 0.0003 |
| controlled_alt_vs_baseline_no_context | overall_quality | 0.1446 | (0.1154, 0.1728) | 0.0000 | 0.1446 | (0.1150, 0.1691) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2034 | (0.1669, 0.2390) | 0.0000 | 0.2034 | (0.1610, 0.2428) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1458 | (0.0978, 0.2001) | 0.0000 | 0.1458 | (0.1083, 0.1891) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0218 | (-0.0715, 0.0239) | 0.8060 | -0.0218 | (-0.0704, 0.0164) | 0.8450 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.2586 | (0.2112, 0.3096) | 0.0000 | 0.2586 | (0.1971, 0.3141) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0744 | (0.0522, 0.1006) | 0.0000 | 0.0744 | (0.0576, 0.0963) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1726 | (0.1169, 0.2409) | 0.0000 | 0.1726 | (0.1221, 0.2294) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0386 | (-0.0139, 0.1061) | 0.0923 | 0.0386 | (0.0086, 0.0875) | 0.0200 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0322 | (-0.0579, -0.0081) | 0.9977 | -0.0322 | (-0.0566, -0.0132) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | -0.0250 | (-0.2139, 0.1389) | 0.6160 | -0.0250 | (-0.1855, 0.1278) | 0.5987 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | -0.0146 | (-0.1021, 0.0729) | 0.6960 | -0.0146 | (-0.1260, 0.0875) | 0.6723 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0397 | (0.0118, 0.0685) | 0.0023 | 0.0397 | (0.0114, 0.0822) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1305 | (0.1079, 0.1550) | 0.0000 | 0.1305 | (0.1033, 0.1522) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 13 | 3 | 8 | 0.7083 | 0.8125 |
| proposed_vs_candidate_no_context | persona_consistency | 5 | 6 | 13 | 0.4792 | 0.4545 |
| proposed_vs_candidate_no_context | naturalness | 6 | 10 | 8 | 0.4167 | 0.3750 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 12 | 1 | 11 | 0.7292 | 0.9231 |
| proposed_vs_candidate_no_context | context_overlap | 12 | 4 | 8 | 0.6667 | 0.7500 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 5 | 5 | 14 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | persona_style | 0 | 2 | 22 | 0.4583 | 0.0000 |
| proposed_vs_candidate_no_context | distinct1 | 8 | 8 | 8 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | length_score | 4 | 10 | 10 | 0.3750 | 0.2857 |
| proposed_vs_candidate_no_context | sentence_score | 3 | 6 | 15 | 0.4375 | 0.3333 |
| proposed_vs_candidate_no_context | bertscore_f1 | 12 | 6 | 6 | 0.6250 | 0.6667 |
| proposed_vs_candidate_no_context | overall_quality | 12 | 6 | 6 | 0.6250 | 0.6667 |
| proposed_vs_baseline_no_context | context_relevance | 15 | 9 | 0 | 0.6250 | 0.6250 |
| proposed_vs_baseline_no_context | persona_consistency | 5 | 7 | 12 | 0.4583 | 0.4167 |
| proposed_vs_baseline_no_context | naturalness | 4 | 19 | 1 | 0.1875 | 0.1739 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 10 | 2 | 12 | 0.6667 | 0.8333 |
| proposed_vs_baseline_no_context | context_overlap | 15 | 9 | 0 | 0.6250 | 0.6250 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 4 | 5 | 15 | 0.4792 | 0.4444 |
| proposed_vs_baseline_no_context | persona_style | 1 | 3 | 20 | 0.4583 | 0.2500 |
| proposed_vs_baseline_no_context | distinct1 | 3 | 17 | 4 | 0.2083 | 0.1500 |
| proposed_vs_baseline_no_context | length_score | 4 | 19 | 1 | 0.1875 | 0.1739 |
| proposed_vs_baseline_no_context | sentence_score | 3 | 12 | 9 | 0.3125 | 0.2000 |
| proposed_vs_baseline_no_context | bertscore_f1 | 14 | 10 | 0 | 0.5833 | 0.5833 |
| proposed_vs_baseline_no_context | overall_quality | 10 | 14 | 0 | 0.4167 | 0.4167 |
| controlled_vs_proposed_raw | context_relevance | 19 | 5 | 0 | 0.7917 | 0.7917 |
| controlled_vs_proposed_raw | persona_consistency | 20 | 2 | 2 | 0.8750 | 0.9091 |
| controlled_vs_proposed_raw | naturalness | 16 | 8 | 0 | 0.6667 | 0.6667 |
| controlled_vs_proposed_raw | context_keyword_coverage | 18 | 5 | 1 | 0.7708 | 0.7826 |
| controlled_vs_proposed_raw | context_overlap | 20 | 4 | 0 | 0.8333 | 0.8333 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 20 | 2 | 2 | 0.8750 | 0.9091 |
| controlled_vs_proposed_raw | persona_style | 4 | 2 | 18 | 0.5417 | 0.6667 |
| controlled_vs_proposed_raw | distinct1 | 12 | 11 | 1 | 0.5208 | 0.5217 |
| controlled_vs_proposed_raw | length_score | 16 | 7 | 1 | 0.6875 | 0.6957 |
| controlled_vs_proposed_raw | sentence_score | 11 | 2 | 11 | 0.6875 | 0.8462 |
| controlled_vs_proposed_raw | bertscore_f1 | 16 | 8 | 0 | 0.6667 | 0.6667 |
| controlled_vs_proposed_raw | overall_quality | 20 | 4 | 0 | 0.8333 | 0.8333 |
| controlled_vs_candidate_no_context | context_relevance | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 19 | 3 | 2 | 0.8333 | 0.8636 |
| controlled_vs_candidate_no_context | naturalness | 15 | 9 | 0 | 0.6250 | 0.6250 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 23 | 0 | 1 | 0.9792 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 19 | 1 | 4 | 0.8750 | 0.9500 |
| controlled_vs_candidate_no_context | persona_style | 3 | 3 | 18 | 0.5000 | 0.5000 |
| controlled_vs_candidate_no_context | distinct1 | 11 | 11 | 2 | 0.5000 | 0.5000 |
| controlled_vs_candidate_no_context | length_score | 14 | 10 | 0 | 0.5833 | 0.5833 |
| controlled_vs_candidate_no_context | sentence_score | 9 | 3 | 12 | 0.6250 | 0.7500 |
| controlled_vs_candidate_no_context | bertscore_f1 | 20 | 4 | 0 | 0.8333 | 0.8333 |
| controlled_vs_candidate_no_context | overall_quality | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | context_relevance | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 18 | 0 | 6 | 0.8750 | 1.0000 |
| controlled_vs_baseline_no_context | naturalness | 13 | 11 | 0 | 0.5417 | 0.5417 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 23 | 0 | 1 | 0.9792 | 1.0000 |
| controlled_vs_baseline_no_context | context_overlap | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 18 | 0 | 6 | 0.8750 | 1.0000 |
| controlled_vs_baseline_no_context | persona_style | 3 | 2 | 19 | 0.5208 | 0.6000 |
| controlled_vs_baseline_no_context | distinct1 | 6 | 17 | 1 | 0.2708 | 0.2609 |
| controlled_vs_baseline_no_context | length_score | 13 | 8 | 3 | 0.6042 | 0.6190 |
| controlled_vs_baseline_no_context | sentence_score | 4 | 5 | 15 | 0.4792 | 0.4444 |
| controlled_vs_baseline_no_context | bertscore_f1 | 15 | 9 | 0 | 0.6250 | 0.6250 |
| controlled_vs_baseline_no_context | overall_quality | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 12 | 7 | 5 | 0.6042 | 0.6316 |
| controlled_alt_vs_controlled_default | persona_consistency | 5 | 6 | 13 | 0.4792 | 0.4545 |
| controlled_alt_vs_controlled_default | naturalness | 7 | 12 | 5 | 0.3958 | 0.3684 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 9 | 5 | 10 | 0.5833 | 0.6429 |
| controlled_alt_vs_controlled_default | context_overlap | 9 | 9 | 6 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 4 | 5 | 15 | 0.4792 | 0.4444 |
| controlled_alt_vs_controlled_default | persona_style | 1 | 2 | 21 | 0.4792 | 0.3333 |
| controlled_alt_vs_controlled_default | distinct1 | 6 | 12 | 6 | 0.3750 | 0.3333 |
| controlled_alt_vs_controlled_default | length_score | 7 | 9 | 8 | 0.4583 | 0.4375 |
| controlled_alt_vs_controlled_default | sentence_score | 7 | 6 | 11 | 0.5208 | 0.5385 |
| controlled_alt_vs_controlled_default | bertscore_f1 | 10 | 10 | 4 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | overall_quality | 10 | 10 | 4 | 0.5000 | 0.5000 |
| controlled_alt_vs_proposed_raw | context_relevance | 20 | 4 | 0 | 0.8333 | 0.8333 |
| controlled_alt_vs_proposed_raw | persona_consistency | 21 | 2 | 1 | 0.8958 | 0.9130 |
| controlled_alt_vs_proposed_raw | naturalness | 18 | 6 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 20 | 3 | 1 | 0.8542 | 0.8696 |
| controlled_alt_vs_proposed_raw | context_overlap | 21 | 3 | 0 | 0.8750 | 0.8750 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 21 | 2 | 1 | 0.8958 | 0.9130 |
| controlled_alt_vs_proposed_raw | persona_style | 4 | 4 | 16 | 0.5000 | 0.5000 |
| controlled_alt_vs_proposed_raw | distinct1 | 14 | 9 | 1 | 0.6042 | 0.6087 |
| controlled_alt_vs_proposed_raw | length_score | 15 | 9 | 0 | 0.6250 | 0.6250 |
| controlled_alt_vs_proposed_raw | sentence_score | 12 | 3 | 9 | 0.6875 | 0.8000 |
| controlled_alt_vs_proposed_raw | bertscore_f1 | 15 | 9 | 0 | 0.6250 | 0.6250 |
| controlled_alt_vs_proposed_raw | overall_quality | 22 | 2 | 0 | 0.9167 | 0.9167 |
| controlled_alt_vs_candidate_no_context | context_relevance | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 20 | 4 | 0 | 0.8333 | 0.8333 |
| controlled_alt_vs_candidate_no_context | naturalness | 12 | 12 | 0 | 0.5000 | 0.5000 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 20 | 2 | 2 | 0.8750 | 0.9091 |
| controlled_alt_vs_candidate_no_context | persona_style | 3 | 5 | 16 | 0.4583 | 0.3750 |
| controlled_alt_vs_candidate_no_context | distinct1 | 11 | 10 | 3 | 0.5208 | 0.5238 |
| controlled_alt_vs_candidate_no_context | length_score | 11 | 13 | 0 | 0.4583 | 0.4583 |
| controlled_alt_vs_candidate_no_context | sentence_score | 12 | 5 | 7 | 0.6458 | 0.7059 |
| controlled_alt_vs_candidate_no_context | bertscore_f1 | 19 | 5 | 0 | 0.7917 | 0.7917 |
| controlled_alt_vs_candidate_no_context | overall_quality | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context | context_relevance | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 17 | 3 | 4 | 0.7917 | 0.8500 |
| controlled_alt_vs_baseline_no_context | naturalness | 7 | 16 | 1 | 0.3125 | 0.3043 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context | context_overlap | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 17 | 2 | 5 | 0.8125 | 0.8947 |
| controlled_alt_vs_baseline_no_context | persona_style | 3 | 4 | 17 | 0.4792 | 0.4286 |
| controlled_alt_vs_baseline_no_context | distinct1 | 6 | 16 | 2 | 0.2917 | 0.2727 |
| controlled_alt_vs_baseline_no_context | length_score | 7 | 14 | 3 | 0.3542 | 0.3333 |
| controlled_alt_vs_baseline_no_context | sentence_score | 3 | 3 | 18 | 0.5000 | 0.5000 |
| controlled_alt_vs_baseline_no_context | bertscore_f1 | 15 | 9 | 0 | 0.6250 | 0.6250 |
| controlled_alt_vs_baseline_no_context | overall_quality | 24 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 24 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 18 | 0 | 6 | 0.8750 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 13 | 11 | 0 | 0.5417 | 0.5417 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 23 | 0 | 1 | 0.9792 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 23 | 1 | 0 | 0.9583 | 0.9583 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 18 | 0 | 6 | 0.8750 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 3 | 2 | 19 | 0.5208 | 0.6000 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | 6 | 17 | 1 | 0.2708 | 0.2609 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 13 | 8 | 3 | 0.6042 | 0.6190 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 4 | 5 | 15 | 0.4792 | 0.4444 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 15 | 9 | 0 | 0.6250 | 0.6250 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 24 | 0 | 0 | 1.0000 | 1.0000 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.2500 | 0.4583 | 0.5417 |
| proposed_contextual_controlled_alt | 0.0000 | 0.0000 | 0.2083 | 0.4583 | 0.5417 |
| proposed_contextual | 0.0000 | 0.0000 | 0.4583 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 |
| baseline_no_context | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `8`
- Unique template signatures: `23`
- Template signature ratio: `0.9583`
- Effective sample size by source clustering: `7.20`
- Effective sample size by template-signature clustering: `22.15`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: enabled.

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
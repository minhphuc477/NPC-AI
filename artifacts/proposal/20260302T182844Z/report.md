# Proposal Alignment Evaluation Report

- Run ID: `20260302T182844Z`
- Generated: `2026-03-02T18:29:57.627820+00:00`
- Scenarios: `artifacts\proposal\20260302T182844Z\scenarios.jsonl`
- Scenario count: `112`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off
- `baseline_no_context`: model `phi3:mini`, dynamic_context=off, response_control=off
- `baseline_no_context_phi3_latest`: model `phi3:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2647 (0.2534, 0.2764) | 0.3568 (0.3270, 0.3862) | 0.9288 (0.9213, 0.9361) | 0.3951 (0.3831, 0.4076) | 0.0871 |
| proposed_contextual | 0.0516 (0.0378, 0.0682) | 0.1419 (0.1172, 0.1691) | 0.8131 (0.7983, 0.8296) | 0.2143 (0.2003, 0.2296) | 0.0594 |
| candidate_no_context | 0.0267 (0.0205, 0.0332) | 0.1379 (0.1138, 0.1642) | 0.8234 (0.8065, 0.8414) | 0.2019 (0.1905, 0.2137) | 0.0365 |
| baseline_no_context | 0.0398 (0.0319, 0.0480) | 0.1615 (0.1457, 0.1774) | 0.8977 (0.8868, 0.9087) | 0.2279 (0.2206, 0.2347) | 0.0393 |
| baseline_no_context_phi3_latest | 0.0351 (0.0272, 0.0435) | 0.1617 (0.1426, 0.1808) | 0.8982 (0.8868, 0.9102) | 0.2266 (0.2193, 0.2344) | 0.0441 |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0249 | 0.9354 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0040 | 0.0288 |
| proposed_vs_candidate_no_context | naturalness | -0.0103 | -0.0126 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0340 | 1.6937 |
| proposed_vs_candidate_no_context | context_overlap | 0.0039 | 0.0926 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0040 | 0.0805 |
| proposed_vs_candidate_no_context | persona_style | 0.0037 | 0.0076 |
| proposed_vs_candidate_no_context | distinct1 | -0.0049 | -0.0052 |
| proposed_vs_candidate_no_context | length_score | -0.0295 | -0.0831 |
| proposed_vs_candidate_no_context | sentence_score | -0.0156 | -0.0206 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0229 | 0.6285 |
| proposed_vs_candidate_no_context | overall_quality | 0.0124 | 0.0612 |
| proposed_vs_baseline_no_context | context_relevance | 0.0118 | 0.2967 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0196 | -0.1213 |
| proposed_vs_baseline_no_context | naturalness | -0.0846 | -0.0943 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0134 | 0.3293 |
| proposed_vs_baseline_no_context | context_overlap | 0.0081 | 0.2149 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0011 | -0.0200 |
| proposed_vs_baseline_no_context | persona_style | -0.0935 | -0.1596 |
| proposed_vs_baseline_no_context | distinct1 | -0.0466 | -0.0474 |
| proposed_vs_baseline_no_context | length_score | -0.2580 | -0.4426 |
| proposed_vs_baseline_no_context | sentence_score | -0.1344 | -0.1530 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0200 | 0.5097 |
| proposed_vs_baseline_no_context | overall_quality | -0.0136 | -0.0595 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 0.0165 | 0.4718 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | -0.0198 | -0.1225 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | -0.0851 | -0.0948 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.0212 | 0.6455 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 0.0057 | 0.1417 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | -0.0025 | -0.0442 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | -0.0890 | -0.1530 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | -0.0524 | -0.0529 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | -0.2598 | -0.4443 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | -0.1125 | -0.1314 |
| proposed_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0153 | 0.3457 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | -0.0123 | -0.0543 |
| controlled_vs_proposed_raw | context_relevance | 0.2131 | 4.1291 |
| controlled_vs_proposed_raw | persona_consistency | 0.2150 | 1.5149 |
| controlled_vs_proposed_raw | naturalness | 0.1158 | 0.1424 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2801 | 5.1861 |
| controlled_vs_proposed_raw | context_overlap | 0.0567 | 1.2327 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2447 | 4.5145 |
| controlled_vs_proposed_raw | persona_style | 0.0958 | 0.1946 |
| controlled_vs_proposed_raw | distinct1 | 0.0072 | 0.0077 |
| controlled_vs_proposed_raw | length_score | 0.4393 | 1.3516 |
| controlled_vs_proposed_raw | sentence_score | 0.2411 | 0.3241 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0277 | 0.4662 |
| controlled_vs_proposed_raw | overall_quality | 0.1808 | 0.8435 |
| controlled_vs_candidate_no_context | context_relevance | 0.2380 | 8.9268 |
| controlled_vs_candidate_no_context | persona_consistency | 0.2189 | 1.5873 |
| controlled_vs_candidate_no_context | naturalness | 0.1054 | 0.1281 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3140 | 15.6633 |
| controlled_vs_candidate_no_context | context_overlap | 0.0606 | 1.4395 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2488 | 4.9585 |
| controlled_vs_candidate_no_context | persona_style | 0.0996 | 0.2036 |
| controlled_vs_candidate_no_context | distinct1 | 0.0023 | 0.0025 |
| controlled_vs_candidate_no_context | length_score | 0.4098 | 1.1562 |
| controlled_vs_candidate_no_context | sentence_score | 0.2254 | 0.2969 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0506 | 1.3878 |
| controlled_vs_candidate_no_context | overall_quality | 0.1931 | 0.9563 |
| controlled_vs_baseline_no_context | context_relevance | 0.2249 | 5.6508 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1954 | 1.2098 |
| controlled_vs_baseline_no_context | naturalness | 0.0311 | 0.0347 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.2935 | 7.2234 |
| controlled_vs_baseline_no_context | context_overlap | 0.0648 | 1.7125 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2436 | 4.4043 |
| controlled_vs_baseline_no_context | persona_style | 0.0023 | 0.0039 |
| controlled_vs_baseline_no_context | distinct1 | -0.0394 | -0.0401 |
| controlled_vs_baseline_no_context | length_score | 0.1813 | 0.3109 |
| controlled_vs_baseline_no_context | sentence_score | 0.1067 | 0.1215 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0477 | 1.2135 |
| controlled_vs_baseline_no_context | overall_quality | 0.1672 | 0.7338 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.2296 | 6.5492 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1951 | 1.2068 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | 0.0307 | 0.0342 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.3013 | 9.1793 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0624 | 1.5490 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.2422 | 4.2706 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | 0.0068 | 0.0117 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0452 | -0.0457 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | 0.1795 | 0.3069 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.1286 | 0.1502 |
| controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0429 | 0.9731 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1685 | 0.7434 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2249 | 5.6508 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1954 | 1.2098 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 0.0311 | 0.0347 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.2935 | 7.2234 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0648 | 1.7125 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2436 | 4.4043 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0023 | 0.0039 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0394 | -0.0401 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.1813 | 0.3109 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.1067 | 0.1215 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0477 | 1.2135 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1672 | 0.7338 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.2296 | 6.5492 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1951 | 1.2068 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | 0.0307 | 0.0342 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.3013 | 9.1793 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0624 | 1.5490 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.2422 | 4.2706 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | 0.0068 | 0.0117 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0452 | -0.0457 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | 0.1795 | 0.3069 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.1286 | 0.1502 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0429 | 0.9731 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1685 | 0.7434 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) |
|---|---|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0249 | (0.0101, 0.0417) | 0.0000 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0040 | (-0.0248, 0.0314) | 0.3933 |
| proposed_vs_candidate_no_context | naturalness | -0.0103 | (-0.0274, 0.0065) | 0.8877 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0340 | (0.0138, 0.0545) | 0.0003 |
| proposed_vs_candidate_no_context | context_overlap | 0.0039 | (-0.0018, 0.0104) | 0.0917 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0040 | (-0.0295, 0.0381) | 0.3997 |
| proposed_vs_candidate_no_context | persona_style | 0.0037 | (-0.0204, 0.0293) | 0.3960 |
| proposed_vs_candidate_no_context | distinct1 | -0.0049 | (-0.0147, 0.0040) | 0.8447 |
| proposed_vs_candidate_no_context | length_score | -0.0295 | (-0.0958, 0.0384) | 0.8093 |
| proposed_vs_candidate_no_context | sentence_score | -0.0156 | (-0.0469, 0.0156) | 0.8677 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0229 | (0.0094, 0.0370) | 0.0007 |
| proposed_vs_candidate_no_context | overall_quality | 0.0124 | (-0.0019, 0.0280) | 0.0437 |
| proposed_vs_baseline_no_context | context_relevance | 0.0118 | (-0.0037, 0.0299) | 0.0807 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0196 | (-0.0462, 0.0078) | 0.9240 |
| proposed_vs_baseline_no_context | naturalness | -0.0846 | (-0.1042, -0.0658) | 1.0000 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0134 | (-0.0078, 0.0375) | 0.1100 |
| proposed_vs_baseline_no_context | context_overlap | 0.0081 | (0.0018, 0.0152) | 0.0073 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0011 | (-0.0312, 0.0313) | 0.5340 |
| proposed_vs_baseline_no_context | persona_style | -0.0935 | (-0.1388, -0.0517) | 1.0000 |
| proposed_vs_baseline_no_context | distinct1 | -0.0466 | (-0.0589, -0.0355) | 1.0000 |
| proposed_vs_baseline_no_context | length_score | -0.2580 | (-0.3322, -0.1848) | 1.0000 |
| proposed_vs_baseline_no_context | sentence_score | -0.1344 | (-0.1719, -0.0969) | 1.0000 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0200 | (0.0047, 0.0365) | 0.0033 |
| proposed_vs_baseline_no_context | overall_quality | -0.0136 | (-0.0284, 0.0032) | 0.9477 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 0.0165 | (-0.0003, 0.0353) | 0.0287 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | -0.0198 | (-0.0500, 0.0112) | 0.8997 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | -0.0851 | (-0.1046, -0.0651) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.0212 | (-0.0005, 0.0464) | 0.0283 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 0.0057 | (-0.0003, 0.0123) | 0.0343 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | -0.0025 | (-0.0346, 0.0302) | 0.5867 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | -0.0890 | (-0.1349, -0.0435) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | -0.0524 | (-0.0642, -0.0418) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | -0.2598 | (-0.3384, -0.1818) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | -0.1125 | (-0.1531, -0.0687) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0153 | (-0.0008, 0.0314) | 0.0307 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | -0.0123 | (-0.0293, 0.0049) | 0.9227 |
| controlled_vs_proposed_raw | context_relevance | 0.2131 | (0.1962, 0.2285) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.2150 | (0.1878, 0.2419) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.1158 | (0.0964, 0.1337) | 0.0000 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2801 | (0.2587, 0.3009) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0567 | (0.0492, 0.0642) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2447 | (0.2106, 0.2771) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.0958 | (0.0564, 0.1366) | 0.0000 |
| controlled_vs_proposed_raw | distinct1 | 0.0072 | (-0.0029, 0.0185) | 0.0853 |
| controlled_vs_proposed_raw | length_score | 0.4393 | (0.3652, 0.5072) | 0.0000 |
| controlled_vs_proposed_raw | sentence_score | 0.2411 | (0.2098, 0.2719) | 0.0000 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0277 | (0.0131, 0.0422) | 0.0000 |
| controlled_vs_proposed_raw | overall_quality | 0.1808 | (0.1666, 0.1953) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2380 | (0.2260, 0.2505) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.2189 | (0.1873, 0.2495) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.1054 | (0.0865, 0.1238) | 0.0000 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3140 | (0.2977, 0.3312) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0606 | (0.0543, 0.0670) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2488 | (0.2116, 0.2857) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.0996 | (0.0602, 0.1409) | 0.0000 |
| controlled_vs_candidate_no_context | distinct1 | 0.0023 | (-0.0054, 0.0101) | 0.2630 |
| controlled_vs_candidate_no_context | length_score | 0.4098 | (0.3411, 0.4804) | 0.0000 |
| controlled_vs_candidate_no_context | sentence_score | 0.2254 | (0.1942, 0.2562) | 0.0000 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0506 | (0.0378, 0.0635) | 0.0000 |
| controlled_vs_candidate_no_context | overall_quality | 0.1931 | (0.1776, 0.2078) | 0.0000 |
| controlled_vs_baseline_no_context | context_relevance | 0.2249 | (0.2124, 0.2373) | 0.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1954 | (0.1697, 0.2214) | 0.0000 |
| controlled_vs_baseline_no_context | naturalness | 0.0311 | (0.0199, 0.0426) | 0.0000 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.2935 | (0.2773, 0.3103) | 0.0000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0648 | (0.0587, 0.0712) | 0.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2436 | (0.2139, 0.2746) | 0.0000 |
| controlled_vs_baseline_no_context | persona_style | 0.0023 | (-0.0163, 0.0199) | 0.4000 |
| controlled_vs_baseline_no_context | distinct1 | -0.0394 | (-0.0452, -0.0338) | 1.0000 |
| controlled_vs_baseline_no_context | length_score | 0.1812 | (0.1345, 0.2286) | 0.0000 |
| controlled_vs_baseline_no_context | sentence_score | 0.1067 | (0.0696, 0.1437) | 0.0000 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0477 | (0.0342, 0.0609) | 0.0000 |
| controlled_vs_baseline_no_context | overall_quality | 0.1672 | (0.1556, 0.1791) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.2296 | (0.2169, 0.2429) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1951 | (0.1679, 0.2226) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | 0.0307 | (0.0186, 0.0427) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.3013 | (0.2838, 0.3188) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0624 | (0.0561, 0.0685) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.2422 | (0.2101, 0.2761) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | 0.0068 | (-0.0168, 0.0305) | 0.2940 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0452 | (-0.0504, -0.0403) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | 0.1795 | (0.1310, 0.2318) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.1286 | (0.0920, 0.1656) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0429 | (0.0292, 0.0566) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1685 | (0.1558, 0.1818) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2249 | (0.2128, 0.2378) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1954 | (0.1693, 0.2218) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 0.0311 | (0.0196, 0.0425) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.2935 | (0.2774, 0.3101) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0648 | (0.0584, 0.0715) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2436 | (0.2140, 0.2744) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0023 | (-0.0166, 0.0197) | 0.4013 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0394 | (-0.0451, -0.0335) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.1812 | (0.1342, 0.2262) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.1067 | (0.0692, 0.1437) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0477 | (0.0340, 0.0611) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1672 | (0.1559, 0.1796) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.2296 | (0.2167, 0.2427) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1951 | (0.1676, 0.2226) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | 0.0307 | (0.0189, 0.0428) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.3013 | (0.2833, 0.3190) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0624 | (0.0562, 0.0688) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.2422 | (0.2085, 0.2748) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | 0.0068 | (-0.0164, 0.0325) | 0.2810 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0452 | (-0.0503, -0.0403) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | 0.1795 | (0.1277, 0.2316) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.1286 | (0.0942, 0.1629) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0429 | (0.0293, 0.0566) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1685 | (0.1558, 0.1816) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 36 | 24 | 52 | 0.5536 | 0.6000 |
| proposed_vs_candidate_no_context | persona_consistency | 16 | 17 | 79 | 0.4955 | 0.4848 |
| proposed_vs_candidate_no_context | naturalness | 28 | 32 | 52 | 0.4821 | 0.4667 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 27 | 11 | 74 | 0.5714 | 0.7105 |
| proposed_vs_candidate_no_context | context_overlap | 32 | 28 | 52 | 0.5179 | 0.5333 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 10 | 12 | 90 | 0.4911 | 0.4545 |
| proposed_vs_candidate_no_context | persona_style | 8 | 7 | 97 | 0.5045 | 0.5333 |
| proposed_vs_candidate_no_context | distinct1 | 21 | 24 | 67 | 0.4866 | 0.4667 |
| proposed_vs_candidate_no_context | length_score | 25 | 33 | 54 | 0.4643 | 0.4310 |
| proposed_vs_candidate_no_context | sentence_score | 10 | 15 | 87 | 0.4777 | 0.4000 |
| proposed_vs_candidate_no_context | bertscore_f1 | 46 | 22 | 44 | 0.6071 | 0.6765 |
| proposed_vs_candidate_no_context | overall_quality | 41 | 27 | 44 | 0.5625 | 0.6029 |
| proposed_vs_baseline_no_context | context_relevance | 59 | 51 | 2 | 0.5357 | 0.5364 |
| proposed_vs_baseline_no_context | persona_consistency | 17 | 48 | 47 | 0.3616 | 0.2615 |
| proposed_vs_baseline_no_context | naturalness | 20 | 90 | 2 | 0.1875 | 0.1818 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 22 | 27 | 63 | 0.4777 | 0.4490 |
| proposed_vs_baseline_no_context | context_overlap | 69 | 41 | 2 | 0.6250 | 0.6273 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 13 | 25 | 74 | 0.4464 | 0.3421 |
| proposed_vs_baseline_no_context | persona_style | 6 | 28 | 78 | 0.4018 | 0.1765 |
| proposed_vs_baseline_no_context | distinct1 | 18 | 76 | 18 | 0.2411 | 0.1915 |
| proposed_vs_baseline_no_context | length_score | 22 | 88 | 2 | 0.2054 | 0.2000 |
| proposed_vs_baseline_no_context | sentence_score | 6 | 49 | 57 | 0.3080 | 0.1091 |
| proposed_vs_baseline_no_context | bertscore_f1 | 63 | 49 | 0 | 0.5625 | 0.5625 |
| proposed_vs_baseline_no_context | overall_quality | 34 | 78 | 0 | 0.3036 | 0.3036 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 56 | 54 | 2 | 0.5089 | 0.5091 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | 17 | 48 | 47 | 0.3616 | 0.2615 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | 25 | 87 | 0 | 0.2232 | 0.2232 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 26 | 23 | 63 | 0.5134 | 0.5306 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 62 | 48 | 2 | 0.5625 | 0.5636 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 14 | 28 | 70 | 0.4375 | 0.3333 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | 7 | 31 | 74 | 0.3929 | 0.1842 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | 12 | 80 | 20 | 0.1964 | 0.1304 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | 26 | 84 | 2 | 0.2411 | 0.2364 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | 11 | 47 | 54 | 0.3393 | 0.1897 |
| proposed_vs_baseline_no_context_phi3_latest | bertscore_f1 | 63 | 49 | 0 | 0.5625 | 0.5625 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | 34 | 78 | 0 | 0.3036 | 0.3036 |
| controlled_vs_proposed_raw | context_relevance | 109 | 3 | 0 | 0.9732 | 0.9732 |
| controlled_vs_proposed_raw | persona_consistency | 102 | 7 | 3 | 0.9241 | 0.9358 |
| controlled_vs_proposed_raw | naturalness | 92 | 20 | 0 | 0.8214 | 0.8214 |
| controlled_vs_proposed_raw | context_keyword_coverage | 107 | 2 | 3 | 0.9688 | 0.9817 |
| controlled_vs_proposed_raw | context_overlap | 105 | 6 | 1 | 0.9420 | 0.9459 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 102 | 6 | 4 | 0.9286 | 0.9444 |
| controlled_vs_proposed_raw | persona_style | 26 | 3 | 83 | 0.6027 | 0.8966 |
| controlled_vs_proposed_raw | distinct1 | 67 | 44 | 1 | 0.6027 | 0.6036 |
| controlled_vs_proposed_raw | length_score | 92 | 18 | 2 | 0.8304 | 0.8364 |
| controlled_vs_proposed_raw | sentence_score | 78 | 1 | 33 | 0.8438 | 0.9873 |
| controlled_vs_proposed_raw | bertscore_f1 | 82 | 30 | 0 | 0.7321 | 0.7321 |
| controlled_vs_proposed_raw | overall_quality | 110 | 2 | 0 | 0.9821 | 0.9821 |
| controlled_vs_candidate_no_context | context_relevance | 112 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 101 | 7 | 4 | 0.9196 | 0.9352 |
| controlled_vs_candidate_no_context | naturalness | 86 | 25 | 1 | 0.7723 | 0.7748 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 112 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 109 | 3 | 0 | 0.9732 | 0.9732 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 101 | 7 | 4 | 0.9196 | 0.9352 |
| controlled_vs_candidate_no_context | persona_style | 27 | 3 | 82 | 0.6071 | 0.9000 |
| controlled_vs_candidate_no_context | distinct1 | 64 | 46 | 2 | 0.5804 | 0.5818 |
| controlled_vs_candidate_no_context | length_score | 86 | 23 | 3 | 0.7812 | 0.7890 |
| controlled_vs_candidate_no_context | sentence_score | 73 | 1 | 38 | 0.8214 | 0.9865 |
| controlled_vs_candidate_no_context | bertscore_f1 | 95 | 17 | 0 | 0.8482 | 0.8482 |
| controlled_vs_candidate_no_context | overall_quality | 108 | 4 | 0 | 0.9643 | 0.9643 |
| controlled_vs_baseline_no_context | context_relevance | 112 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 97 | 4 | 11 | 0.9152 | 0.9604 |
| controlled_vs_baseline_no_context | naturalness | 78 | 33 | 1 | 0.7009 | 0.7027 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 112 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | context_overlap | 108 | 4 | 0 | 0.9643 | 0.9643 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 97 | 2 | 13 | 0.9241 | 0.9798 |
| controlled_vs_baseline_no_context | persona_style | 8 | 7 | 97 | 0.5045 | 0.5333 |
| controlled_vs_baseline_no_context | distinct1 | 15 | 96 | 1 | 0.1384 | 0.1351 |
| controlled_vs_baseline_no_context | length_score | 80 | 27 | 5 | 0.7366 | 0.7477 |
| controlled_vs_baseline_no_context | sentence_score | 39 | 4 | 69 | 0.6562 | 0.9070 |
| controlled_vs_baseline_no_context | bertscore_f1 | 82 | 30 | 0 | 0.7321 | 0.7321 |
| controlled_vs_baseline_no_context | overall_quality | 112 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 112 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 95 | 4 | 13 | 0.9062 | 0.9596 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | 74 | 38 | 0 | 0.6607 | 0.6607 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 112 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 108 | 3 | 1 | 0.9688 | 0.9730 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 95 | 2 | 15 | 0.9152 | 0.9794 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | 11 | 12 | 89 | 0.4955 | 0.4783 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | 7 | 105 | 0 | 0.0625 | 0.0625 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | 73 | 31 | 8 | 0.6875 | 0.7019 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 45 | 4 | 63 | 0.6830 | 0.9184 |
| controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 83 | 29 | 0 | 0.7411 | 0.7411 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 112 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 112 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 97 | 4 | 11 | 0.9152 | 0.9604 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 78 | 33 | 1 | 0.7009 | 0.7027 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 112 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 108 | 4 | 0 | 0.9643 | 0.9643 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 97 | 2 | 13 | 0.9241 | 0.9798 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 8 | 7 | 97 | 0.5045 | 0.5333 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | 15 | 96 | 1 | 0.1384 | 0.1351 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 80 | 27 | 5 | 0.7366 | 0.7477 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 39 | 4 | 69 | 0.6562 | 0.9070 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 82 | 30 | 0 | 0.7321 | 0.7321 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 112 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 112 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 95 | 4 | 13 | 0.9062 | 0.9596 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | 74 | 38 | 0 | 0.6607 | 0.6607 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 112 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 108 | 3 | 1 | 0.9688 | 0.9730 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 95 | 2 | 15 | 0.9152 | 0.9794 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | 11 | 12 | 89 | 0.4955 | 0.4783 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | 7 | 105 | 0 | 0.0625 | 0.0625 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | 73 | 31 | 8 | 0.6875 | 0.7019 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | 45 | 4 | 63 | 0.6830 | 0.9184 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 83 | 29 | 0 | 0.7411 | 0.7411 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 112 | 0 | 0 | 1.0000 | 1.0000 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

- BERTScore status: enabled.

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
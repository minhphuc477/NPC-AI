# Proposal Alignment Evaluation Report

- Run ID: `20260306T091907Z`
- Generated: `2026-03-06T09:22:06.134443+00:00`
- Scenarios: `artifacts\proposal_control_tuning\preflight_blend_gatearm_smoke\20260306T091907Z\seed_runs\seed_31\20260306T091907Z\scenarios.jsonl`
- Scenario count: `8`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_alt`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off
- `baseline_no_context`: model `phi3:mini`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2835 (0.2304, 0.3506) | 0.2799 (0.2181, 0.3602) | 0.8826 (0.8219, 0.9358) | 0.3713 (0.3432, 0.4019) | 0.0895 |
| proposed_contextual_controlled_alt | 0.3114 (0.2707, 0.3670) | 0.2608 (0.2284, 0.2991) | 0.8418 (0.7793, 0.9050) | 0.3660 (0.3456, 0.3881) | 0.0721 |
| proposed_contextual | 0.0739 (0.0091, 0.1640) | 0.1351 (0.0643, 0.2077) | 0.8142 (0.7616, 0.8824) | 0.2231 (0.1748, 0.2741) | 0.0638 |
| candidate_no_context | 0.0245 (0.0062, 0.0506) | 0.2342 (0.1125, 0.3601) | 0.8096 (0.7678, 0.8585) | 0.2326 (0.1798, 0.2915) | 0.0472 |
| baseline_no_context | 0.0362 (0.0137, 0.0600) | 0.1488 (0.0833, 0.2464) | 0.8837 (0.8409, 0.9205) | 0.2265 (0.2049, 0.2546) | 0.0844 |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0494 | 2.0183 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0991 | -0.4231 |
| proposed_vs_candidate_no_context | naturalness | 0.0046 | 0.0056 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0682 | 3.0000 |
| proposed_vs_candidate_no_context | context_overlap | 0.0057 | 0.1983 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.1161 | -0.6610 |
| proposed_vs_candidate_no_context | persona_style | -0.0312 | -0.0667 |
| proposed_vs_candidate_no_context | distinct1 | -0.0089 | -0.0094 |
| proposed_vs_candidate_no_context | length_score | 0.0625 | 0.2459 |
| proposed_vs_candidate_no_context | sentence_score | -0.0437 | -0.0560 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0166 | 0.3505 |
| proposed_vs_candidate_no_context | overall_quality | -0.0095 | -0.0407 |
| proposed_vs_baseline_no_context | context_relevance | 0.0377 | 1.0426 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0137 | -0.0920 |
| proposed_vs_baseline_no_context | naturalness | -0.0695 | -0.0786 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0568 | 1.6667 |
| proposed_vs_baseline_no_context | context_overlap | -0.0068 | -0.1657 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0119 | -0.1667 |
| proposed_vs_baseline_no_context | persona_style | -0.0208 | -0.0455 |
| proposed_vs_baseline_no_context | distinct1 | -0.0346 | -0.0354 |
| proposed_vs_baseline_no_context | length_score | -0.2125 | -0.4016 |
| proposed_vs_baseline_no_context | sentence_score | -0.1312 | -0.1511 |
| proposed_vs_baseline_no_context | bertscore_f1 | -0.0207 | -0.2448 |
| proposed_vs_baseline_no_context | overall_quality | -0.0034 | -0.0151 |
| controlled_vs_proposed_raw | context_relevance | 0.2096 | 2.8350 |
| controlled_vs_proposed_raw | persona_consistency | 0.1447 | 1.0713 |
| controlled_vs_proposed_raw | naturalness | 0.0684 | 0.0841 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2727 | 3.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0622 | 1.8135 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1798 | 3.0200 |
| controlled_vs_proposed_raw | persona_style | 0.0047 | 0.0107 |
| controlled_vs_proposed_raw | distinct1 | 0.0039 | 0.0042 |
| controlled_vs_proposed_raw | length_score | 0.2250 | 0.7105 |
| controlled_vs_proposed_raw | sentence_score | 0.2188 | 0.2966 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0257 | 0.4026 |
| controlled_vs_proposed_raw | overall_quality | 0.1482 | 0.6640 |
| controlled_vs_candidate_no_context | context_relevance | 0.2590 | 10.5751 |
| controlled_vs_candidate_no_context | persona_consistency | 0.0456 | 0.1949 |
| controlled_vs_candidate_no_context | naturalness | 0.0730 | 0.0902 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3409 | 15.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0678 | 2.3715 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.0637 | 0.3627 |
| controlled_vs_candidate_no_context | persona_style | -0.0266 | -0.0567 |
| controlled_vs_candidate_no_context | distinct1 | -0.0050 | -0.0053 |
| controlled_vs_candidate_no_context | length_score | 0.2875 | 1.1311 |
| controlled_vs_candidate_no_context | sentence_score | 0.1750 | 0.2240 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0422 | 0.8942 |
| controlled_vs_candidate_no_context | overall_quality | 0.1387 | 0.5963 |
| controlled_vs_baseline_no_context | context_relevance | 0.2473 | 6.8334 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1311 | 0.8807 |
| controlled_vs_baseline_no_context | naturalness | -0.0010 | -0.0012 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3295 | 9.6667 |
| controlled_vs_baseline_no_context | context_overlap | 0.0554 | 1.3474 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1679 | 2.3500 |
| controlled_vs_baseline_no_context | persona_style | -0.0161 | -0.0352 |
| controlled_vs_baseline_no_context | distinct1 | -0.0307 | -0.0314 |
| controlled_vs_baseline_no_context | length_score | 0.0125 | 0.0236 |
| controlled_vs_baseline_no_context | sentence_score | 0.0875 | 0.1007 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0050 | 0.0593 |
| controlled_vs_baseline_no_context | overall_quality | 0.1447 | 0.6390 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0280 | 0.0987 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0191 | -0.0682 |
| controlled_alt_vs_controlled_default | naturalness | -0.0408 | -0.0462 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0341 | 0.0937 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0137 | 0.1420 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0357 | -0.1493 |
| controlled_alt_vs_controlled_default | persona_style | 0.0474 | 0.1072 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0296 | -0.0313 |
| controlled_alt_vs_controlled_default | length_score | -0.0625 | -0.1154 |
| controlled_alt_vs_controlled_default | sentence_score | -0.1313 | -0.1373 |
| controlled_alt_vs_controlled_default | bertscore_f1 | -0.0173 | -0.1939 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0052 | -0.0141 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.2375 | 3.2134 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1257 | 0.9300 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0277 | 0.0340 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.3068 | 3.3750 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0759 | 2.2132 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1440 | 2.4200 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0521 | 0.1190 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0257 | -0.0272 |
| controlled_alt_vs_proposed_raw | length_score | 0.1625 | 0.5132 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0875 | 0.1186 |
| controlled_alt_vs_proposed_raw | bertscore_f1 | 0.0083 | 0.1306 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1429 | 0.6405 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2870 | 11.7173 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.0265 | 0.1133 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0322 | 0.0398 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3750 | 16.5000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0815 | 2.8504 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.0280 | 0.1593 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0208 | 0.0444 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0346 | -0.0364 |
| controlled_alt_vs_candidate_no_context | length_score | 0.2250 | 0.8852 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.0437 | 0.0560 |
| controlled_alt_vs_candidate_no_context | bertscore_f1 | 0.0249 | 0.5269 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1334 | 0.5737 |
| controlled_alt_vs_baseline_no_context | context_relevance | 0.2753 | 7.6064 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 0.1120 | 0.7524 |
| controlled_alt_vs_baseline_no_context | naturalness | -0.0418 | -0.0473 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 0.3636 | 10.6667 |
| controlled_alt_vs_baseline_no_context | context_overlap | 0.0691 | 1.6808 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 0.1321 | 1.8500 |
| controlled_alt_vs_baseline_no_context | persona_style | 0.0312 | 0.0682 |
| controlled_alt_vs_baseline_no_context | distinct1 | -0.0603 | -0.0617 |
| controlled_alt_vs_baseline_no_context | length_score | -0.0500 | -0.0945 |
| controlled_alt_vs_baseline_no_context | sentence_score | -0.0438 | -0.0504 |
| controlled_alt_vs_baseline_no_context | bertscore_f1 | -0.0123 | -0.1461 |
| controlled_alt_vs_baseline_no_context | overall_quality | 0.1395 | 0.6158 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2473 | 6.8334 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1311 | 0.8807 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0010 | -0.0012 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3295 | 9.6667 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0554 | 1.3474 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1679 | 2.3500 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | -0.0161 | -0.0352 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0307 | -0.0314 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.0125 | 0.0236 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0875 | 0.1007 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0050 | 0.0593 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1447 | 0.6390 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0494 | (-0.0252, 0.1468) | 0.1117 | 0.0494 | (0.0121, 0.0751) | 0.0000 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0991 | (-0.2426, 0.0333) | 0.9297 | -0.0991 | (-0.2790, 0.0136) | 0.9607 |
| proposed_vs_candidate_no_context | naturalness | 0.0046 | (-0.0597, 0.0631) | 0.4360 | 0.0046 | (-0.0743, 0.0610) | 0.3770 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0682 | (-0.0227, 0.1932) | 0.1470 | 0.0682 | (0.0152, 0.1074) | 0.0100 |
| proposed_vs_candidate_no_context | context_overlap | 0.0057 | (-0.0059, 0.0183) | 0.1747 | 0.0057 | (-0.0013, 0.0161) | 0.0710 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.1161 | (-0.2917, 0.0447) | 0.9127 | -0.1161 | (-0.3175, 0.0159) | 0.9643 |
| proposed_vs_candidate_no_context | persona_style | -0.0312 | (-0.0938, 0.0000) | 1.0000 | -0.0312 | (-0.1250, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | distinct1 | -0.0089 | (-0.0378, 0.0140) | 0.7340 | -0.0089 | (-0.0509, 0.0116) | 0.7177 |
| proposed_vs_candidate_no_context | length_score | 0.0625 | (-0.1542, 0.2917) | 0.2753 | 0.0625 | (-0.2333, 0.3458) | 0.3363 |
| proposed_vs_candidate_no_context | sentence_score | -0.0437 | (-0.2188, 0.1312) | 0.7473 | -0.0437 | (-0.1750, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0166 | (0.0022, 0.0347) | 0.0043 | 0.0166 | (0.0003, 0.0288) | 0.0157 |
| proposed_vs_candidate_no_context | overall_quality | -0.0095 | (-0.0887, 0.0747) | 0.5673 | -0.0095 | (-0.0892, 0.0328) | 0.6200 |
| proposed_vs_baseline_no_context | context_relevance | 0.0377 | (-0.0196, 0.1278) | 0.2150 | 0.0377 | (-0.0031, 0.0849) | 0.3180 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0137 | (-0.1488, 0.0833) | 0.5423 | -0.0137 | (-0.2119, 0.0738) | 0.5493 |
| proposed_vs_baseline_no_context | naturalness | -0.0695 | (-0.1444, 0.0014) | 0.9700 | -0.0695 | (-0.1464, 0.0295) | 0.9043 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0568 | (-0.0227, 0.1818) | 0.2173 | 0.0568 | (0.0000, 0.1240) | 0.3233 |
| proposed_vs_baseline_no_context | context_overlap | -0.0068 | (-0.0151, 0.0024) | 0.9313 | -0.0068 | (-0.0104, -0.0047) | 1.0000 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0119 | (-0.1935, 0.1161) | 0.5190 | -0.0119 | (-0.2579, 0.0884) | 0.5457 |
| proposed_vs_baseline_no_context | persona_style | -0.0208 | (-0.0625, 0.0000) | 1.0000 | -0.0208 | (-0.0833, 0.0000) | 1.0000 |
| proposed_vs_baseline_no_context | distinct1 | -0.0346 | (-0.0555, -0.0146) | 1.0000 | -0.0346 | (-0.0544, -0.0076) | 1.0000 |
| proposed_vs_baseline_no_context | length_score | -0.2125 | (-0.5126, 0.0958) | 0.9053 | -0.2125 | (-0.5556, 0.2286) | 0.8223 |
| proposed_vs_baseline_no_context | sentence_score | -0.1312 | (-0.3062, 0.0437) | 0.9583 | -0.1312 | (-0.2188, -0.0500) | 1.0000 |
| proposed_vs_baseline_no_context | bertscore_f1 | -0.0207 | (-0.0510, 0.0195) | 0.8570 | -0.0207 | (-0.0513, 0.0207) | 0.8283 |
| proposed_vs_baseline_no_context | overall_quality | -0.0034 | (-0.0663, 0.0628) | 0.5360 | -0.0034 | (-0.0802, 0.0330) | 0.5287 |
| controlled_vs_proposed_raw | context_relevance | 0.2096 | (0.1017, 0.3174) | 0.0000 | 0.2096 | (0.1700, 0.2489) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.1447 | (0.0521, 0.2602) | 0.0000 | 0.1447 | (0.0507, 0.2989) | 0.0033 |
| controlled_vs_proposed_raw | naturalness | 0.0684 | (-0.0104, 0.1431) | 0.0450 | 0.0684 | (-0.0469, 0.1581) | 0.1537 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2727 | (0.1250, 0.4091) | 0.0000 | 0.2727 | (0.2159, 0.3247) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0622 | (0.0471, 0.0796) | 0.0000 | 0.0622 | (0.0489, 0.0718) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1798 | (0.0833, 0.3065) | 0.0000 | 0.1798 | (0.0889, 0.3742) | 0.0007 |
| controlled_vs_proposed_raw | persona_style | 0.0047 | (-0.0797, 0.0938) | 0.4533 | 0.0047 | (-0.0911, 0.1071) | 0.4397 |
| controlled_vs_proposed_raw | distinct1 | 0.0039 | (-0.0257, 0.0322) | 0.3870 | 0.0039 | (-0.0379, 0.0254) | 0.3680 |
| controlled_vs_proposed_raw | length_score | 0.2250 | (-0.1168, 0.5626) | 0.1073 | 0.2250 | (-0.2619, 0.6037) | 0.2123 |
| controlled_vs_proposed_raw | sentence_score | 0.2188 | (0.0875, 0.3062) | 0.0000 | 0.2188 | (0.1312, 0.3000) | 0.0003 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0257 | (-0.0020, 0.0512) | 0.0343 | 0.0257 | (0.0062, 0.0631) | 0.0033 |
| controlled_vs_proposed_raw | overall_quality | 0.1482 | (0.0781, 0.2166) | 0.0000 | 0.1482 | (0.1000, 0.2169) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2590 | (0.2066, 0.3319) | 0.0000 | 0.2590 | (0.1964, 0.3043) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.0456 | (-0.0333, 0.1203) | 0.1183 | 0.0456 | (0.0103, 0.0979) | 0.0003 |
| controlled_vs_candidate_no_context | naturalness | 0.0730 | (-0.0195, 0.1565) | 0.0603 | 0.0730 | (-0.0691, 0.1500) | 0.1177 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3409 | (0.2614, 0.4318) | 0.0000 | 0.3409 | (0.2576, 0.4050) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0678 | (0.0537, 0.0830) | 0.0000 | 0.0678 | (0.0585, 0.0747) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.0637 | (-0.0417, 0.1619) | 0.1173 | 0.0637 | (0.0179, 0.1371) | 0.0143 |
| controlled_vs_candidate_no_context | persona_style | -0.0266 | (-0.0797, 0.0000) | 1.0000 | -0.0266 | (-0.1063, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | distinct1 | -0.0050 | (-0.0443, 0.0319) | 0.5963 | -0.0050 | (-0.0628, 0.0320) | 0.5477 |
| controlled_vs_candidate_no_context | length_score | 0.2875 | (-0.0750, 0.5917) | 0.0583 | 0.2875 | (-0.2533, 0.5733) | 0.1277 |
| controlled_vs_candidate_no_context | sentence_score | 0.1750 | (0.0437, 0.3062) | 0.0043 | 0.1750 | (0.0583, 0.2500) | 0.0087 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0422 | (0.0243, 0.0570) | 0.0000 | 0.0422 | (0.0326, 0.0634) | 0.0000 |
| controlled_vs_candidate_no_context | overall_quality | 0.1387 | (0.0927, 0.1888) | 0.0000 | 0.1387 | (0.0943, 0.1732) | 0.0000 |
| controlled_vs_baseline_no_context | context_relevance | 0.2473 | (0.1810, 0.3217) | 0.0000 | 0.2473 | (0.1669, 0.3131) | 0.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1311 | (0.0940, 0.1756) | 0.0000 | 0.1311 | (0.0784, 0.1721) | 0.0000 |
| controlled_vs_baseline_no_context | naturalness | -0.0010 | (-0.0417, 0.0450) | 0.5260 | -0.0010 | (-0.0475, 0.0205) | 0.5440 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3295 | (0.2384, 0.4432) | 0.0000 | 0.3295 | (0.2159, 0.4215) | 0.0000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0554 | (0.0466, 0.0648) | 0.0000 | 0.0554 | (0.0434, 0.0627) | 0.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1679 | (0.1054, 0.2262) | 0.0000 | 0.1679 | (0.0857, 0.2204) | 0.0013 |
| controlled_vs_baseline_no_context | persona_style | -0.0161 | (-0.1005, 0.0729) | 0.6480 | -0.0161 | (-0.1264, 0.0833) | 0.6460 |
| controlled_vs_baseline_no_context | distinct1 | -0.0307 | (-0.0515, -0.0109) | 1.0000 | -0.0307 | (-0.0513, -0.0094) | 1.0000 |
| controlled_vs_baseline_no_context | length_score | 0.0125 | (-0.1917, 0.2292) | 0.4527 | 0.0125 | (-0.2400, 0.1273) | 0.4453 |
| controlled_vs_baseline_no_context | sentence_score | 0.0875 | (0.0000, 0.1750) | 0.1130 | 0.0875 | (0.0000, 0.2000) | 0.0757 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0050 | (-0.0301, 0.0389) | 0.3833 | 0.0050 | (-0.0291, 0.0516) | 0.3890 |
| controlled_vs_baseline_no_context | overall_quality | 0.1447 | (0.1178, 0.1752) | 0.0000 | 0.1447 | (0.1117, 0.1702) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0280 | (-0.0625, 0.1261) | 0.2873 | 0.0280 | (-0.0543, 0.1680) | 0.2973 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0191 | (-0.0815, 0.0201) | 0.6830 | -0.0191 | (-0.1072, 0.0253) | 0.6763 |
| controlled_alt_vs_controlled_default | naturalness | -0.0408 | (-0.1316, 0.0599) | 0.7980 | -0.0408 | (-0.1099, 0.0875) | 0.7950 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0341 | (-0.0909, 0.1705) | 0.3360 | 0.0341 | (-0.0808, 0.2182) | 0.3707 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0137 | (-0.0068, 0.0359) | 0.1067 | 0.0137 | (0.0025, 0.0305) | 0.0107 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0357 | (-0.1071, 0.0000) | 1.0000 | -0.0357 | (-0.1429, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | persona_style | 0.0474 | (0.0000, 0.1099) | 0.0923 | 0.0474 | (0.0000, 0.1517) | 0.0880 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0296 | (-0.0786, 0.0191) | 0.8663 | -0.0296 | (-0.0587, 0.0157) | 0.9083 |
| controlled_alt_vs_controlled_default | length_score | -0.0625 | (-0.4000, 0.2959) | 0.6507 | -0.0625 | (-0.3208, 0.3556) | 0.6900 |
| controlled_alt_vs_controlled_default | sentence_score | -0.1312 | (-0.2625, 0.0437) | 0.9600 | -0.1312 | (-0.2693, 0.1400) | 0.8937 |
| controlled_alt_vs_controlled_default | bertscore_f1 | -0.0173 | (-0.0517, 0.0131) | 0.8480 | -0.0173 | (-0.0527, 0.0065) | 0.9247 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0052 | (-0.0471, 0.0344) | 0.5837 | -0.0052 | (-0.0424, 0.0492) | 0.5933 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.2375 | (0.1356, 0.3296) | 0.0003 | 0.2375 | (0.1716, 0.3650) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1257 | (0.0667, 0.1861) | 0.0000 | 0.1257 | (0.0667, 0.2017) | 0.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0277 | (-0.0625, 0.1103) | 0.2517 | 0.0277 | (-0.0803, 0.1104) | 0.2830 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.3068 | (0.1591, 0.4318) | 0.0000 | 0.3068 | (0.2121, 0.4698) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0759 | (0.0498, 0.1003) | 0.0000 | 0.0759 | (0.0627, 0.0863) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1440 | (0.0833, 0.2048) | 0.0000 | 0.1440 | (0.0833, 0.2175) | 0.0007 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0521 | (0.0000, 0.1250) | 0.1047 | 0.0521 | (0.0000, 0.1667) | 0.0697 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0257 | (-0.0568, 0.0059) | 0.9483 | -0.0257 | (-0.0408, -0.0138) | 1.0000 |
| controlled_alt_vs_proposed_raw | length_score | 0.1625 | (-0.2209, 0.5042) | 0.1867 | 0.1625 | (-0.3143, 0.5333) | 0.2673 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0875 | (-0.0875, 0.2625) | 0.1947 | 0.0875 | (-0.0700, 0.3500) | 0.1927 |
| controlled_alt_vs_proposed_raw | bertscore_f1 | 0.0083 | (-0.0427, 0.0482) | 0.3560 | 0.0083 | (-0.0186, 0.0483) | 0.3040 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1429 | (0.0899, 0.1934) | 0.0000 | 0.1429 | (0.1114, 0.2122) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2870 | (0.2313, 0.3518) | 0.0000 | 0.2870 | (0.2470, 0.3631) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.0265 | (-0.0810, 0.1201) | 0.2977 | 0.0265 | (-0.0758, 0.1157) | 0.2437 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0322 | (-0.0473, 0.1061) | 0.2060 | 0.0322 | (-0.0666, 0.1069) | 0.2470 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3750 | (0.2955, 0.4659) | 0.0000 | 0.3750 | (0.3182, 0.5000) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0815 | (0.0663, 0.0958) | 0.0000 | 0.0815 | (0.0762, 0.0883) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.0280 | (-0.1089, 0.1375) | 0.3210 | 0.0280 | (-0.1095, 0.1238) | 0.2820 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0208 | (0.0000, 0.0625) | 0.3517 | 0.0208 | (0.0000, 0.0833) | 0.3333 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0346 | (-0.0712, 0.0062) | 0.9573 | -0.0346 | (-0.0771, -0.0091) | 1.0000 |
| controlled_alt_vs_candidate_no_context | length_score | 0.2250 | (-0.0917, 0.5083) | 0.0743 | 0.2250 | (-0.2000, 0.5185) | 0.1473 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.0437 | (-0.1312, 0.2188) | 0.4013 | 0.0437 | (-0.0875, 0.2800) | 0.3683 |
| controlled_alt_vs_candidate_no_context | bertscore_f1 | 0.0249 | (-0.0119, 0.0589) | 0.0963 | 0.0249 | (0.0006, 0.0530) | 0.0157 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1334 | (0.0809, 0.1876) | 0.0000 | 0.1334 | (0.0850, 0.1959) | 0.0000 |
| controlled_alt_vs_baseline_no_context | context_relevance | 0.2753 | (0.2339, 0.3337) | 0.0000 | 0.2753 | (0.2328, 0.3533) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 0.1120 | (0.0173, 0.1810) | 0.0090 | 0.1120 | (-0.0036, 0.1763) | 0.0387 |
| controlled_alt_vs_baseline_no_context | naturalness | -0.0418 | (-0.1264, 0.0509) | 0.8210 | -0.0418 | (-0.1057, 0.0541) | 0.8517 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 0.3636 | (0.3068, 0.4545) | 0.0000 | 0.3636 | (0.3117, 0.4848) | 0.0000 |
| controlled_alt_vs_baseline_no_context | context_overlap | 0.0691 | (0.0472, 0.0905) | 0.0000 | 0.0691 | (0.0562, 0.0785) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 0.1321 | (-0.0018, 0.2274) | 0.0270 | 0.1321 | (-0.0306, 0.2204) | 0.0630 |
| controlled_alt_vs_baseline_no_context | persona_style | 0.0312 | (0.0000, 0.0938) | 0.3283 | 0.0312 | (0.0000, 0.1250) | 0.3417 |
| controlled_alt_vs_baseline_no_context | distinct1 | -0.0603 | (-0.1035, -0.0250) | 1.0000 | -0.0603 | (-0.0796, -0.0264) | 1.0000 |
| controlled_alt_vs_baseline_no_context | length_score | -0.0500 | (-0.3667, 0.3000) | 0.6317 | -0.0500 | (-0.3893, 0.2667) | 0.6653 |
| controlled_alt_vs_baseline_no_context | sentence_score | -0.0437 | (-0.2625, 0.1750) | 0.7353 | -0.0437 | (-0.2188, 0.2800) | 0.6987 |
| controlled_alt_vs_baseline_no_context | bertscore_f1 | -0.0123 | (-0.0417, 0.0197) | 0.7767 | -0.0123 | (-0.0405, 0.0308) | 0.7087 |
| controlled_alt_vs_baseline_no_context | overall_quality | 0.1395 | (0.1053, 0.1718) | 0.0000 | 0.1395 | (0.0998, 0.1727) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2473 | (0.1818, 0.3282) | 0.0000 | 0.2473 | (0.1669, 0.3130) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1311 | (0.0944, 0.1748) | 0.0000 | 0.1311 | (0.0740, 0.1721) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0010 | (-0.0410, 0.0450) | 0.5280 | -0.0010 | (-0.0475, 0.0205) | 0.5523 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3295 | (0.2386, 0.4318) | 0.0000 | 0.3295 | (0.2159, 0.4242) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0554 | (0.0467, 0.0650) | 0.0000 | 0.0554 | (0.0434, 0.0627) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1679 | (0.1083, 0.2292) | 0.0000 | 0.1679 | (0.0918, 0.2204) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | -0.0161 | (-0.1005, 0.0729) | 0.6480 | -0.0161 | (-0.1264, 0.0897) | 0.6257 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0307 | (-0.0518, -0.0105) | 1.0000 | -0.0307 | (-0.0511, -0.0094) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.0125 | (-0.1917, 0.2292) | 0.4480 | 0.0125 | (-0.2400, 0.1273) | 0.4390 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0875 | (0.0000, 0.1761) | 0.1030 | 0.0875 | (0.0000, 0.2000) | 0.0757 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0050 | (-0.0299, 0.0393) | 0.3880 | 0.0050 | (-0.0285, 0.0505) | 0.3733 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1447 | (0.1173, 0.1747) | 0.0000 | 0.1447 | (0.1137, 0.1702) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 4 | 2 | 2 | 0.6250 | 0.6667 |
| proposed_vs_candidate_no_context | persona_consistency | 2 | 3 | 3 | 0.4375 | 0.4000 |
| proposed_vs_candidate_no_context | naturalness | 4 | 2 | 2 | 0.6250 | 0.6667 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 3 | 2 | 3 | 0.5625 | 0.6000 |
| proposed_vs_candidate_no_context | context_overlap | 4 | 2 | 2 | 0.6250 | 0.6667 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 2 | 3 | 3 | 0.4375 | 0.4000 |
| proposed_vs_candidate_no_context | persona_style | 0 | 1 | 7 | 0.4375 | 0.0000 |
| proposed_vs_candidate_no_context | distinct1 | 2 | 2 | 4 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | length_score | 3 | 3 | 2 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | sentence_score | 2 | 3 | 3 | 0.4375 | 0.4000 |
| proposed_vs_candidate_no_context | bertscore_f1 | 5 | 1 | 2 | 0.7500 | 0.8333 |
| proposed_vs_candidate_no_context | overall_quality | 3 | 3 | 2 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context | context_relevance | 2 | 6 | 0 | 0.2500 | 0.2500 |
| proposed_vs_baseline_no_context | persona_consistency | 3 | 2 | 3 | 0.5625 | 0.6000 |
| proposed_vs_baseline_no_context | naturalness | 3 | 5 | 0 | 0.3750 | 0.3750 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 2 | 1 | 5 | 0.5625 | 0.6667 |
| proposed_vs_baseline_no_context | context_overlap | 1 | 7 | 0 | 0.1250 | 0.1250 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 3 | 1 | 4 | 0.6250 | 0.7500 |
| proposed_vs_baseline_no_context | persona_style | 0 | 1 | 7 | 0.4375 | 0.0000 |
| proposed_vs_baseline_no_context | distinct1 | 0 | 6 | 2 | 0.1250 | 0.0000 |
| proposed_vs_baseline_no_context | length_score | 3 | 5 | 0 | 0.3750 | 0.3750 |
| proposed_vs_baseline_no_context | sentence_score | 1 | 4 | 3 | 0.3125 | 0.2000 |
| proposed_vs_baseline_no_context | bertscore_f1 | 1 | 7 | 0 | 0.1250 | 0.1250 |
| proposed_vs_baseline_no_context | overall_quality | 4 | 4 | 0 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | context_relevance | 7 | 1 | 0 | 0.8750 | 0.8750 |
| controlled_vs_proposed_raw | persona_consistency | 6 | 1 | 1 | 0.8125 | 0.8571 |
| controlled_vs_proposed_raw | naturalness | 5 | 2 | 1 | 0.6875 | 0.7143 |
| controlled_vs_proposed_raw | context_keyword_coverage | 6 | 1 | 1 | 0.8125 | 0.8571 |
| controlled_vs_proposed_raw | context_overlap | 8 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 6 | 0 | 2 | 0.8750 | 1.0000 |
| controlled_vs_proposed_raw | persona_style | 1 | 1 | 6 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | distinct1 | 4 | 3 | 1 | 0.5625 | 0.5714 |
| controlled_vs_proposed_raw | length_score | 4 | 3 | 1 | 0.5625 | 0.5714 |
| controlled_vs_proposed_raw | sentence_score | 5 | 0 | 3 | 0.8125 | 1.0000 |
| controlled_vs_proposed_raw | bertscore_f1 | 6 | 2 | 0 | 0.7500 | 0.7500 |
| controlled_vs_proposed_raw | overall_quality | 8 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_relevance | 8 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 5 | 2 | 1 | 0.6875 | 0.7143 |
| controlled_vs_candidate_no_context | naturalness | 5 | 3 | 0 | 0.6250 | 0.6250 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 8 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 8 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 5 | 2 | 1 | 0.6875 | 0.7143 |
| controlled_vs_candidate_no_context | persona_style | 0 | 1 | 7 | 0.4375 | 0.0000 |
| controlled_vs_candidate_no_context | distinct1 | 4 | 3 | 1 | 0.5625 | 0.5714 |
| controlled_vs_candidate_no_context | length_score | 5 | 3 | 0 | 0.6250 | 0.6250 |
| controlled_vs_candidate_no_context | sentence_score | 4 | 0 | 4 | 0.7500 | 1.0000 |
| controlled_vs_candidate_no_context | bertscore_f1 | 7 | 1 | 0 | 0.8750 | 0.8750 |
| controlled_vs_candidate_no_context | overall_quality | 8 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | context_relevance | 8 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 8 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | naturalness | 3 | 4 | 1 | 0.4375 | 0.4286 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 8 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | context_overlap | 8 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 7 | 0 | 1 | 0.9375 | 1.0000 |
| controlled_vs_baseline_no_context | persona_style | 1 | 2 | 5 | 0.4375 | 0.3333 |
| controlled_vs_baseline_no_context | distinct1 | 0 | 6 | 2 | 0.1250 | 0.0000 |
| controlled_vs_baseline_no_context | length_score | 3 | 4 | 1 | 0.4375 | 0.4286 |
| controlled_vs_baseline_no_context | sentence_score | 2 | 0 | 6 | 0.6250 | 1.0000 |
| controlled_vs_baseline_no_context | bertscore_f1 | 5 | 3 | 0 | 0.6250 | 0.6250 |
| controlled_vs_baseline_no_context | overall_quality | 8 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 4 | 3 | 1 | 0.5625 | 0.5714 |
| controlled_alt_vs_controlled_default | persona_consistency | 2 | 1 | 5 | 0.5625 | 0.6667 |
| controlled_alt_vs_controlled_default | naturalness | 2 | 5 | 1 | 0.3125 | 0.2857 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 2 | 2 | 4 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | context_overlap | 4 | 3 | 1 | 0.5625 | 0.5714 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0 | 1 | 7 | 0.4375 | 0.0000 |
| controlled_alt_vs_controlled_default | persona_style | 2 | 0 | 6 | 0.6250 | 1.0000 |
| controlled_alt_vs_controlled_default | distinct1 | 2 | 5 | 1 | 0.3125 | 0.2857 |
| controlled_alt_vs_controlled_default | length_score | 2 | 4 | 2 | 0.3750 | 0.3333 |
| controlled_alt_vs_controlled_default | sentence_score | 1 | 4 | 3 | 0.3125 | 0.2000 |
| controlled_alt_vs_controlled_default | bertscore_f1 | 3 | 4 | 1 | 0.4375 | 0.4286 |
| controlled_alt_vs_controlled_default | overall_quality | 4 | 3 | 1 | 0.5625 | 0.5714 |
| controlled_alt_vs_proposed_raw | context_relevance | 7 | 1 | 0 | 0.8750 | 0.8750 |
| controlled_alt_vs_proposed_raw | persona_consistency | 6 | 0 | 2 | 0.8750 | 1.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 5 | 3 | 0 | 0.6250 | 0.6250 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 7 | 1 | 0 | 0.8750 | 0.8750 |
| controlled_alt_vs_proposed_raw | context_overlap | 8 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 6 | 0 | 2 | 0.8750 | 1.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 2 | 0 | 6 | 0.6250 | 1.0000 |
| controlled_alt_vs_proposed_raw | distinct1 | 1 | 6 | 1 | 0.1875 | 0.1429 |
| controlled_alt_vs_proposed_raw | length_score | 5 | 3 | 0 | 0.6250 | 0.6250 |
| controlled_alt_vs_proposed_raw | sentence_score | 3 | 1 | 4 | 0.6250 | 0.7500 |
| controlled_alt_vs_proposed_raw | bertscore_f1 | 5 | 3 | 0 | 0.6250 | 0.6250 |
| controlled_alt_vs_proposed_raw | overall_quality | 8 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 8 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 5 | 3 | 0 | 0.6250 | 0.6250 |
| controlled_alt_vs_candidate_no_context | naturalness | 4 | 4 | 0 | 0.5000 | 0.5000 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 8 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 8 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 5 | 3 | 0 | 0.6250 | 0.6250 |
| controlled_alt_vs_candidate_no_context | persona_style | 1 | 0 | 7 | 0.5625 | 1.0000 |
| controlled_alt_vs_candidate_no_context | distinct1 | 2 | 5 | 1 | 0.3125 | 0.2857 |
| controlled_alt_vs_candidate_no_context | length_score | 6 | 2 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_candidate_no_context | sentence_score | 3 | 2 | 3 | 0.5625 | 0.6000 |
| controlled_alt_vs_candidate_no_context | bertscore_f1 | 5 | 3 | 0 | 0.6250 | 0.6250 |
| controlled_alt_vs_candidate_no_context | overall_quality | 8 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context | context_relevance | 8 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 7 | 1 | 0 | 0.8750 | 0.8750 |
| controlled_alt_vs_baseline_no_context | naturalness | 2 | 6 | 0 | 0.2500 | 0.2500 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 8 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context | context_overlap | 8 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 7 | 1 | 0 | 0.8750 | 0.8750 |
| controlled_alt_vs_baseline_no_context | persona_style | 1 | 0 | 7 | 0.5625 | 1.0000 |
| controlled_alt_vs_baseline_no_context | distinct1 | 1 | 6 | 1 | 0.1875 | 0.1429 |
| controlled_alt_vs_baseline_no_context | length_score | 4 | 4 | 0 | 0.5000 | 0.5000 |
| controlled_alt_vs_baseline_no_context | sentence_score | 3 | 4 | 1 | 0.4375 | 0.4286 |
| controlled_alt_vs_baseline_no_context | bertscore_f1 | 3 | 5 | 0 | 0.3750 | 0.3750 |
| controlled_alt_vs_baseline_no_context | overall_quality | 8 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 8 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 8 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 3 | 4 | 1 | 0.4375 | 0.4286 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 8 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 8 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 7 | 0 | 1 | 0.9375 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 1 | 2 | 5 | 0.4375 | 0.3333 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | 0 | 6 | 2 | 0.1250 | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 3 | 4 | 1 | 0.4375 | 0.4286 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 2 | 0 | 6 | 0.6250 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 5 | 3 | 0 | 0.6250 | 0.6250 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 8 | 0 | 0 | 1.0000 | 1.0000 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.3750 | 0.5000 | 0.5000 |
| proposed_contextual_controlled_alt | 0.0000 | 0.0000 | 0.1250 | 0.5000 | 0.5000 |
| proposed_contextual | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.3750 | 0.0000 | 0.0000 |
| baseline_no_context | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `5`
- Unique template signatures: `8`
- Template signature ratio: `1.0000`
- Effective sample size by source clustering: `4.00`
- Effective sample size by template-signature clustering: `8.00`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: enabled.

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
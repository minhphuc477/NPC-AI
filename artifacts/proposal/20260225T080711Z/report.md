# Proposal Alignment Evaluation Report

- Run ID: `20260225T080711Z`
- Generated: `2026-02-25T08:08:18.180819+00:00`
- Scenarios: `artifacts\proposal\20260225T080711Z\scenarios.jsonl`
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
| proposed_contextual_controlled | 0.2855 (0.2701, 0.3015) | 0.2412 (0.2116, 0.2701) | 0.8891 (0.8787, 0.8994) | 0.3606 (0.3469, 0.3742) | 0.1083 |
| proposed_contextual | 0.0624 (0.0480, 0.0796) | 0.1463 (0.1238, 0.1687) | 0.8253 (0.8091, 0.8418) | 0.2228 (0.2087, 0.2383) | 0.0633 |
| candidate_no_context | 0.0213 (0.0161, 0.0273) | 0.1378 (0.1165, 0.1616) | 0.8170 (0.7996, 0.8351) | 0.1993 (0.1892, 0.2095) | 0.0421 |
| baseline_no_context | 0.0368 (0.0295, 0.0444) | 0.1735 (0.1548, 0.1930) | 0.8970 (0.8854, 0.9085) | 0.2313 (0.2233, 0.2393) | 0.0450 |
| baseline_no_context_phi3_latest | 0.0394 (0.0307, 0.0488) | 0.1578 (0.1406, 0.1748) | 0.8925 (0.8803, 0.9039) | 0.2270 (0.2191, 0.2350) | 0.0518 |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0412 | 1.9353 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0085 | 0.0617 |
| proposed_vs_candidate_no_context | naturalness | 0.0083 | 0.0102 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0541 | 3.9970 |
| proposed_vs_candidate_no_context | context_overlap | 0.0111 | 0.2811 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0087 | 0.1715 |
| proposed_vs_candidate_no_context | persona_style | 0.0077 | 0.0158 |
| proposed_vs_candidate_no_context | distinct1 | 0.0125 | 0.0134 |
| proposed_vs_candidate_no_context | length_score | 0.0149 | 0.0439 |
| proposed_vs_candidate_no_context | sentence_score | 0.0031 | 0.0042 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0212 | 0.5029 |
| proposed_vs_candidate_no_context | overall_quality | 0.0234 | 0.1176 |
| proposed_vs_baseline_no_context | context_relevance | 0.0256 | 0.6958 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0272 | -0.1567 |
| proposed_vs_baseline_no_context | naturalness | -0.0718 | -0.0800 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0305 | 0.8244 |
| proposed_vs_baseline_no_context | context_overlap | 0.0141 | 0.3893 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0113 | -0.1592 |
| proposed_vs_baseline_no_context | persona_style | -0.0909 | -0.1555 |
| proposed_vs_baseline_no_context | distinct1 | -0.0371 | -0.0377 |
| proposed_vs_baseline_no_context | length_score | -0.2268 | -0.3904 |
| proposed_vs_baseline_no_context | sentence_score | -0.1156 | -0.1331 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0183 | 0.4066 |
| proposed_vs_baseline_no_context | overall_quality | -0.0085 | -0.0367 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 0.0231 | 0.5859 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | -0.0115 | -0.0726 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | -0.0672 | -0.0753 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.0284 | 0.7267 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 0.0105 | 0.2636 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.0074 | 0.1419 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | -0.0869 | -0.1497 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | -0.0339 | -0.0346 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | -0.2086 | -0.3707 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | -0.1188 | -0.1362 |
| proposed_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0115 | 0.2210 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | -0.0042 | -0.0187 |
| controlled_vs_proposed_raw | context_relevance | 0.2231 | 3.5723 |
| controlled_vs_proposed_raw | persona_consistency | 0.0949 | 0.6483 |
| controlled_vs_proposed_raw | naturalness | 0.0639 | 0.0774 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2977 | 4.4036 |
| controlled_vs_proposed_raw | context_overlap | 0.0489 | 0.9708 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.0924 | 1.5521 |
| controlled_vs_proposed_raw | persona_style | 0.1048 | 0.2123 |
| controlled_vs_proposed_raw | distinct1 | -0.0074 | -0.0078 |
| controlled_vs_proposed_raw | length_score | 0.2196 | 0.6202 |
| controlled_vs_proposed_raw | sentence_score | 0.2344 | 0.3112 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0450 | 0.7120 |
| controlled_vs_proposed_raw | overall_quality | 0.1379 | 0.6189 |
| controlled_vs_candidate_no_context | context_relevance | 0.2642 | 12.4210 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1034 | 0.7500 |
| controlled_vs_candidate_no_context | naturalness | 0.0722 | 0.0883 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3518 | 26.0020 |
| controlled_vs_candidate_no_context | context_overlap | 0.0600 | 1.5249 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1011 | 1.9900 |
| controlled_vs_candidate_no_context | persona_style | 0.1125 | 0.2314 |
| controlled_vs_candidate_no_context | distinct1 | 0.0051 | 0.0055 |
| controlled_vs_candidate_no_context | length_score | 0.2345 | 0.6912 |
| controlled_vs_candidate_no_context | sentence_score | 0.2375 | 0.3167 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0662 | 1.5730 |
| controlled_vs_candidate_no_context | overall_quality | 0.1613 | 0.8093 |
| controlled_vs_baseline_no_context | context_relevance | 0.2487 | 6.7537 |
| controlled_vs_baseline_no_context | persona_consistency | 0.0677 | 0.3900 |
| controlled_vs_baseline_no_context | naturalness | -0.0079 | -0.0088 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3282 | 8.8583 |
| controlled_vs_baseline_no_context | context_overlap | 0.0631 | 1.7381 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.0811 | 1.1459 |
| controlled_vs_baseline_no_context | persona_style | 0.0139 | 0.0238 |
| controlled_vs_baseline_no_context | distinct1 | -0.0445 | -0.0452 |
| controlled_vs_baseline_no_context | length_score | -0.0071 | -0.0123 |
| controlled_vs_baseline_no_context | sentence_score | 0.1187 | 0.1367 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0633 | 1.4081 |
| controlled_vs_baseline_no_context | overall_quality | 0.1294 | 0.5594 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.2461 | 6.2513 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.0834 | 0.5286 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | -0.0033 | -0.0037 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.3261 | 8.3303 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0594 | 1.4903 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.0998 | 1.9144 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | 0.0179 | 0.0308 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0414 | -0.0422 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | 0.0110 | 0.0196 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.1156 | 0.1326 |
| controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0565 | 1.0904 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1336 | 0.5886 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2487 | 6.7537 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.0677 | 0.3900 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0079 | -0.0088 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3282 | 8.8583 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0631 | 1.7381 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.0811 | 1.1459 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0139 | 0.0238 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0445 | -0.0452 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | -0.0071 | -0.0123 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.1187 | 0.1367 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0633 | 1.4081 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1294 | 0.5594 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.2461 | 6.2513 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.0834 | 0.5286 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | -0.0033 | -0.0037 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.3261 | 8.3303 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0594 | 1.4903 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.0998 | 1.9144 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | 0.0179 | 0.0308 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0414 | -0.0422 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | 0.0110 | 0.0196 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.1156 | 0.1326 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0565 | 1.0904 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1336 | 0.5886 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) |
|---|---|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0412 | (0.0257, 0.0581) | 0.0000 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0085 | (-0.0187, 0.0360) | 0.2730 |
| proposed_vs_candidate_no_context | naturalness | 0.0083 | (-0.0082, 0.0261) | 0.1763 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0541 | (0.0344, 0.0757) | 0.0000 |
| proposed_vs_candidate_no_context | context_overlap | 0.0111 | (0.0050, 0.0182) | 0.0000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0087 | (-0.0236, 0.0408) | 0.2860 |
| proposed_vs_candidate_no_context | persona_style | 0.0077 | (-0.0187, 0.0333) | 0.2757 |
| proposed_vs_candidate_no_context | distinct1 | 0.0125 | (0.0063, 0.0187) | 0.0000 |
| proposed_vs_candidate_no_context | length_score | 0.0149 | (-0.0557, 0.0813) | 0.3383 |
| proposed_vs_candidate_no_context | sentence_score | 0.0031 | (-0.0312, 0.0375) | 0.4587 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0212 | (0.0087, 0.0343) | 0.0003 |
| proposed_vs_candidate_no_context | overall_quality | 0.0234 | (0.0079, 0.0397) | 0.0013 |
| proposed_vs_baseline_no_context | context_relevance | 0.0256 | (0.0088, 0.0440) | 0.0013 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0272 | (-0.0539, -0.0002) | 0.9753 |
| proposed_vs_baseline_no_context | naturalness | -0.0718 | (-0.0906, -0.0528) | 1.0000 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0305 | (0.0089, 0.0538) | 0.0010 |
| proposed_vs_baseline_no_context | context_overlap | 0.0141 | (0.0070, 0.0219) | 0.0000 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0113 | (-0.0400, 0.0193) | 0.7780 |
| proposed_vs_baseline_no_context | persona_style | -0.0909 | (-0.1382, -0.0467) | 1.0000 |
| proposed_vs_baseline_no_context | distinct1 | -0.0371 | (-0.0458, -0.0282) | 1.0000 |
| proposed_vs_baseline_no_context | length_score | -0.2268 | (-0.3003, -0.1503) | 1.0000 |
| proposed_vs_baseline_no_context | sentence_score | -0.1156 | (-0.1562, -0.0750) | 1.0000 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0183 | (0.0037, 0.0334) | 0.0053 |
| proposed_vs_baseline_no_context | overall_quality | -0.0085 | (-0.0247, 0.0083) | 0.8417 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 0.0231 | (0.0049, 0.0412) | 0.0057 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | -0.0115 | (-0.0373, 0.0145) | 0.8070 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | -0.0672 | (-0.0870, -0.0469) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.0284 | (0.0058, 0.0527) | 0.0050 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 0.0105 | (0.0034, 0.0182) | 0.0023 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.0074 | (-0.0193, 0.0355) | 0.3077 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | -0.0869 | (-0.1294, -0.0450) | 0.9997 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | -0.0339 | (-0.0429, -0.0248) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | -0.2086 | (-0.2887, -0.1313) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | -0.1187 | (-0.1594, -0.0750) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0115 | (-0.0032, 0.0263) | 0.0690 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | -0.0042 | (-0.0213, 0.0127) | 0.6917 |
| controlled_vs_proposed_raw | context_relevance | 0.2231 | (0.2021, 0.2431) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.0949 | (0.0648, 0.1251) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0639 | (0.0435, 0.0828) | 0.0000 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2977 | (0.2718, 0.3244) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0489 | (0.0406, 0.0567) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.0924 | (0.0574, 0.1293) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.1048 | (0.0631, 0.1455) | 0.0000 |
| controlled_vs_proposed_raw | distinct1 | -0.0074 | (-0.0164, 0.0020) | 0.9427 |
| controlled_vs_proposed_raw | length_score | 0.2196 | (0.1420, 0.2973) | 0.0000 |
| controlled_vs_proposed_raw | sentence_score | 0.2344 | (0.2000, 0.2656) | 0.0000 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0450 | (0.0290, 0.0605) | 0.0000 |
| controlled_vs_proposed_raw | overall_quality | 0.1379 | (0.1214, 0.1558) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2642 | (0.2491, 0.2807) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1034 | (0.0725, 0.1345) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.0722 | (0.0502, 0.0937) | 0.0000 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3518 | (0.3323, 0.3734) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0600 | (0.0530, 0.0667) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1011 | (0.0653, 0.1378) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.1125 | (0.0720, 0.1531) | 0.0000 |
| controlled_vs_candidate_no_context | distinct1 | 0.0051 | (-0.0036, 0.0137) | 0.1133 |
| controlled_vs_candidate_no_context | length_score | 0.2345 | (0.1515, 0.3173) | 0.0000 |
| controlled_vs_candidate_no_context | sentence_score | 0.2375 | (0.2062, 0.2687) | 0.0000 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0662 | (0.0529, 0.0791) | 0.0000 |
| controlled_vs_candidate_no_context | overall_quality | 0.1613 | (0.1468, 0.1769) | 0.0000 |
| controlled_vs_baseline_no_context | context_relevance | 0.2487 | (0.2325, 0.2654) | 0.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 0.0677 | (0.0370, 0.1003) | 0.0000 |
| controlled_vs_baseline_no_context | naturalness | -0.0079 | (-0.0222, 0.0059) | 0.8630 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3282 | (0.3072, 0.3504) | 0.0000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0631 | (0.0566, 0.0697) | 0.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.0811 | (0.0430, 0.1203) | 0.0000 |
| controlled_vs_baseline_no_context | persona_style | 0.0139 | (-0.0139, 0.0427) | 0.1687 |
| controlled_vs_baseline_no_context | distinct1 | -0.0445 | (-0.0508, -0.0384) | 1.0000 |
| controlled_vs_baseline_no_context | length_score | -0.0071 | (-0.0717, 0.0568) | 0.5927 |
| controlled_vs_baseline_no_context | sentence_score | 0.1187 | (0.0844, 0.1531) | 0.0000 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0633 | (0.0478, 0.0781) | 0.0000 |
| controlled_vs_baseline_no_context | overall_quality | 0.1294 | (0.1139, 0.1450) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.2461 | (0.2291, 0.2644) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.0834 | (0.0569, 0.1107) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | -0.0033 | (-0.0179, 0.0115) | 0.6557 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.3261 | (0.3035, 0.3497) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0594 | (0.0523, 0.0666) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.0998 | (0.0682, 0.1322) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | 0.0179 | (-0.0095, 0.0456) | 0.0950 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0414 | (-0.0480, -0.0346) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | 0.0110 | (-0.0572, 0.0780) | 0.3723 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.1156 | (0.0844, 0.1500) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0565 | (0.0417, 0.0710) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1336 | (0.1195, 0.1477) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2487 | (0.2321, 0.2658) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.0677 | (0.0358, 0.0991) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0079 | (-0.0220, 0.0064) | 0.8517 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3282 | (0.3063, 0.3510) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0631 | (0.0568, 0.0695) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.0811 | (0.0422, 0.1202) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0139 | (-0.0144, 0.0412) | 0.1647 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0445 | (-0.0510, -0.0383) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | -0.0071 | (-0.0720, 0.0533) | 0.5950 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.1187 | (0.0874, 0.1531) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0633 | (0.0487, 0.0786) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1294 | (0.1136, 0.1452) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.2461 | (0.2293, 0.2634) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.0834 | (0.0547, 0.1123) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | -0.0033 | (-0.0185, 0.0114) | 0.6713 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.3261 | (0.3037, 0.3496) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0594 | (0.0526, 0.0663) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.0998 | (0.0679, 0.1337) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | 0.0179 | (-0.0089, 0.0463) | 0.0983 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0414 | (-0.0482, -0.0347) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | 0.0110 | (-0.0572, 0.0789) | 0.3690 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.1156 | (0.0844, 0.1500) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0565 | (0.0412, 0.0709) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1336 | (0.1199, 0.1475) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 43 | 15 | 54 | 0.6250 | 0.7414 |
| proposed_vs_candidate_no_context | persona_consistency | 25 | 18 | 69 | 0.5312 | 0.5814 |
| proposed_vs_candidate_no_context | naturalness | 30 | 28 | 54 | 0.5089 | 0.5172 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 37 | 6 | 69 | 0.6384 | 0.8605 |
| proposed_vs_candidate_no_context | context_overlap | 38 | 19 | 55 | 0.5848 | 0.6667 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 19 | 13 | 80 | 0.5268 | 0.5938 |
| proposed_vs_candidate_no_context | persona_style | 12 | 8 | 92 | 0.5179 | 0.6000 |
| proposed_vs_candidate_no_context | distinct1 | 35 | 7 | 70 | 0.6250 | 0.8333 |
| proposed_vs_candidate_no_context | length_score | 29 | 27 | 56 | 0.5089 | 0.5179 |
| proposed_vs_candidate_no_context | sentence_score | 16 | 15 | 81 | 0.5045 | 0.5161 |
| proposed_vs_candidate_no_context | bertscore_f1 | 43 | 24 | 45 | 0.5848 | 0.6418 |
| proposed_vs_candidate_no_context | overall_quality | 45 | 22 | 45 | 0.6027 | 0.6716 |
| proposed_vs_baseline_no_context | context_relevance | 66 | 46 | 0 | 0.5893 | 0.5893 |
| proposed_vs_baseline_no_context | persona_consistency | 23 | 46 | 43 | 0.3973 | 0.3333 |
| proposed_vs_baseline_no_context | naturalness | 29 | 83 | 0 | 0.2589 | 0.2589 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 32 | 26 | 54 | 0.5268 | 0.5517 |
| proposed_vs_baseline_no_context | context_overlap | 72 | 40 | 0 | 0.6429 | 0.6429 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 19 | 29 | 64 | 0.4554 | 0.3958 |
| proposed_vs_baseline_no_context | persona_style | 7 | 31 | 74 | 0.3929 | 0.1842 |
| proposed_vs_baseline_no_context | distinct1 | 18 | 70 | 24 | 0.2679 | 0.2045 |
| proposed_vs_baseline_no_context | length_score | 26 | 83 | 3 | 0.2455 | 0.2385 |
| proposed_vs_baseline_no_context | sentence_score | 8 | 45 | 59 | 0.3348 | 0.1509 |
| proposed_vs_baseline_no_context | bertscore_f1 | 65 | 47 | 0 | 0.5804 | 0.5804 |
| proposed_vs_baseline_no_context | overall_quality | 38 | 74 | 0 | 0.3393 | 0.3393 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 66 | 46 | 0 | 0.5893 | 0.5893 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | 24 | 41 | 47 | 0.4241 | 0.3692 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | 29 | 83 | 0 | 0.2589 | 0.2589 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 32 | 23 | 57 | 0.5402 | 0.5818 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 69 | 41 | 2 | 0.6250 | 0.6273 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 21 | 24 | 67 | 0.4866 | 0.4667 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | 8 | 28 | 76 | 0.4107 | 0.2222 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | 21 | 68 | 23 | 0.2902 | 0.2360 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | 32 | 80 | 0 | 0.2857 | 0.2857 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | 11 | 49 | 52 | 0.3304 | 0.1833 |
| proposed_vs_baseline_no_context_phi3_latest | bertscore_f1 | 58 | 54 | 0 | 0.5179 | 0.5179 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | 42 | 70 | 0 | 0.3750 | 0.3750 |
| controlled_vs_proposed_raw | context_relevance | 106 | 6 | 0 | 0.9464 | 0.9464 |
| controlled_vs_proposed_raw | persona_consistency | 62 | 12 | 38 | 0.7232 | 0.8378 |
| controlled_vs_proposed_raw | naturalness | 78 | 34 | 0 | 0.6964 | 0.6964 |
| controlled_vs_proposed_raw | context_keyword_coverage | 106 | 1 | 5 | 0.9688 | 0.9907 |
| controlled_vs_proposed_raw | context_overlap | 99 | 12 | 1 | 0.8884 | 0.8919 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 46 | 11 | 55 | 0.6562 | 0.8070 |
| controlled_vs_proposed_raw | persona_style | 32 | 6 | 74 | 0.6161 | 0.8421 |
| controlled_vs_proposed_raw | distinct1 | 58 | 52 | 2 | 0.5268 | 0.5273 |
| controlled_vs_proposed_raw | length_score | 71 | 38 | 3 | 0.6473 | 0.6514 |
| controlled_vs_proposed_raw | sentence_score | 76 | 1 | 35 | 0.8348 | 0.9870 |
| controlled_vs_proposed_raw | bertscore_f1 | 82 | 30 | 0 | 0.7321 | 0.7321 |
| controlled_vs_proposed_raw | overall_quality | 105 | 7 | 0 | 0.9375 | 0.9375 |
| controlled_vs_candidate_no_context | context_relevance | 112 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 72 | 9 | 31 | 0.7812 | 0.8889 |
| controlled_vs_candidate_no_context | naturalness | 82 | 30 | 0 | 0.7321 | 0.7321 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 112 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 109 | 3 | 0 | 0.9732 | 0.9732 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 51 | 9 | 52 | 0.6875 | 0.8500 |
| controlled_vs_candidate_no_context | persona_style | 34 | 4 | 74 | 0.6339 | 0.8947 |
| controlled_vs_candidate_no_context | distinct1 | 71 | 40 | 1 | 0.6384 | 0.6396 |
| controlled_vs_candidate_no_context | length_score | 74 | 36 | 2 | 0.6696 | 0.6727 |
| controlled_vs_candidate_no_context | sentence_score | 78 | 2 | 32 | 0.8393 | 0.9750 |
| controlled_vs_candidate_no_context | bertscore_f1 | 95 | 17 | 0 | 0.8482 | 0.8482 |
| controlled_vs_candidate_no_context | overall_quality | 111 | 1 | 0 | 0.9911 | 0.9911 |
| controlled_vs_baseline_no_context | context_relevance | 112 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 55 | 24 | 33 | 0.6384 | 0.6962 |
| controlled_vs_baseline_no_context | naturalness | 43 | 69 | 0 | 0.3839 | 0.3839 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 112 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | context_overlap | 109 | 2 | 1 | 0.9777 | 0.9820 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 49 | 21 | 42 | 0.6250 | 0.7000 |
| controlled_vs_baseline_no_context | persona_style | 16 | 10 | 86 | 0.5268 | 0.6154 |
| controlled_vs_baseline_no_context | distinct1 | 9 | 101 | 2 | 0.0893 | 0.0818 |
| controlled_vs_baseline_no_context | length_score | 50 | 59 | 3 | 0.4598 | 0.4587 |
| controlled_vs_baseline_no_context | sentence_score | 40 | 2 | 70 | 0.6696 | 0.9524 |
| controlled_vs_baseline_no_context | bertscore_f1 | 88 | 24 | 0 | 0.7857 | 0.7857 |
| controlled_vs_baseline_no_context | overall_quality | 106 | 6 | 0 | 0.9464 | 0.9464 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 112 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 52 | 14 | 46 | 0.6696 | 0.7879 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | 47 | 65 | 0 | 0.4196 | 0.4196 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 111 | 0 | 1 | 0.9955 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 106 | 5 | 1 | 0.9509 | 0.9550 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 47 | 10 | 55 | 0.6652 | 0.8246 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | 15 | 10 | 87 | 0.5223 | 0.6000 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | 16 | 96 | 0 | 0.1429 | 0.1429 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | 52 | 54 | 6 | 0.4911 | 0.4906 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 39 | 2 | 71 | 0.6652 | 0.9512 |
| controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 85 | 27 | 0 | 0.7589 | 0.7589 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 111 | 1 | 0 | 0.9911 | 0.9911 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 112 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 55 | 24 | 33 | 0.6384 | 0.6962 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 43 | 69 | 0 | 0.3839 | 0.3839 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 112 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 109 | 2 | 1 | 0.9777 | 0.9820 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 49 | 21 | 42 | 0.6250 | 0.7000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 16 | 10 | 86 | 0.5268 | 0.6154 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | 9 | 101 | 2 | 0.0893 | 0.0818 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 50 | 59 | 3 | 0.4598 | 0.4587 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 40 | 2 | 70 | 0.6696 | 0.9524 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 88 | 24 | 0 | 0.7857 | 0.7857 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 106 | 6 | 0 | 0.9464 | 0.9464 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 112 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 52 | 14 | 46 | 0.6696 | 0.7879 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | 47 | 65 | 0 | 0.4196 | 0.4196 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 111 | 0 | 1 | 0.9955 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 106 | 5 | 1 | 0.9509 | 0.9550 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 47 | 10 | 55 | 0.6652 | 0.8246 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | 15 | 10 | 87 | 0.5223 | 0.6000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | 16 | 96 | 0 | 0.1429 | 0.1429 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | 52 | 54 | 6 | 0.4911 | 0.4906 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | 39 | 2 | 71 | 0.6652 | 0.9512 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 85 | 27 | 0 | 0.7589 | 0.7589 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 111 | 1 | 0 | 0.9911 | 0.9911 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

- BERTScore status: enabled.

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
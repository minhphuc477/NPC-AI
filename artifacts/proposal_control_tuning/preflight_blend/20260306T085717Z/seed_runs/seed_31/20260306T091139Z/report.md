# Proposal Alignment Evaluation Report

- Run ID: `20260306T091139Z`
- Generated: `2026-03-06T09:18:10.746786+00:00`
- Scenarios: `artifacts\proposal_control_tuning\preflight_blend\20260306T085717Z\seed_runs\seed_31\20260306T091139Z\scenarios.jsonl`
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
| proposed_contextual_controlled | 0.2641 (0.2163, 0.3188) | 0.3201 (0.2737, 0.3760) | 0.8749 (0.8455, 0.9016) | 0.3720 (0.3480, 0.3977) | 0.0838 |
| proposed_contextual_controlled_alt | 0.2594 (0.2107, 0.3092) | 0.3556 (0.2867, 0.4379) | 0.8654 (0.8370, 0.8929) | 0.3777 (0.3536, 0.4016) | 0.0635 |
| proposed_contextual | 0.1069 (0.0636, 0.1593) | 0.1684 (0.1211, 0.2198) | 0.7936 (0.7723, 0.8171) | 0.2426 (0.2129, 0.2758) | 0.0682 |
| candidate_no_context | 0.0241 (0.0128, 0.0376) | 0.1426 (0.1067, 0.1810) | 0.8271 (0.7945, 0.8620) | 0.2032 (0.1847, 0.2243) | 0.0427 |
| baseline_no_context | 0.0493 (0.0313, 0.0695) | 0.1899 (0.1349, 0.2581) | 0.8891 (0.8644, 0.9117) | 0.2409 (0.2224, 0.2639) | 0.0521 |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0828 | 3.4385 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0259 | 0.1814 |
| proposed_vs_candidate_no_context | naturalness | -0.0335 | -0.0405 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.1066 | 5.4103 |
| proposed_vs_candidate_no_context | context_overlap | 0.0274 | 0.7986 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0407 | 1.0513 |
| proposed_vs_candidate_no_context | persona_style | -0.0334 | -0.0598 |
| proposed_vs_candidate_no_context | distinct1 | -0.0147 | -0.0155 |
| proposed_vs_candidate_no_context | length_score | -0.1528 | -0.4264 |
| proposed_vs_candidate_no_context | sentence_score | 0.0292 | 0.0388 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0255 | 0.5976 |
| proposed_vs_candidate_no_context | overall_quality | 0.0394 | 0.1939 |
| proposed_vs_baseline_no_context | context_relevance | 0.0576 | 1.1696 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0214 | -0.1129 |
| proposed_vs_baseline_no_context | naturalness | -0.0955 | -0.1074 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0739 | 1.4096 |
| proposed_vs_baseline_no_context | context_overlap | 0.0198 | 0.4705 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0073 | -0.0847 |
| proposed_vs_baseline_no_context | persona_style | -0.0778 | -0.1291 |
| proposed_vs_baseline_no_context | distinct1 | -0.0415 | -0.0425 |
| proposed_vs_baseline_no_context | length_score | -0.3361 | -0.6205 |
| proposed_vs_baseline_no_context | sentence_score | -0.1167 | -0.1299 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0161 | 0.3091 |
| proposed_vs_baseline_no_context | overall_quality | 0.0018 | 0.0074 |
| controlled_vs_proposed_raw | context_relevance | 0.1572 | 1.4701 |
| controlled_vs_proposed_raw | persona_consistency | 0.1516 | 0.9002 |
| controlled_vs_proposed_raw | naturalness | 0.0813 | 0.1024 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2050 | 1.6235 |
| controlled_vs_proposed_raw | context_overlap | 0.0456 | 0.7383 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1673 | 2.1075 |
| controlled_vs_proposed_raw | persona_style | 0.0891 | 0.1697 |
| controlled_vs_proposed_raw | distinct1 | 0.0070 | 0.0074 |
| controlled_vs_proposed_raw | length_score | 0.3194 | 1.5541 |
| controlled_vs_proposed_raw | sentence_score | 0.1458 | 0.1867 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0156 | 0.2294 |
| controlled_vs_proposed_raw | overall_quality | 0.1294 | 0.5331 |
| controlled_vs_candidate_no_context | context_relevance | 0.2400 | 9.9636 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1775 | 1.2449 |
| controlled_vs_candidate_no_context | naturalness | 0.0477 | 0.0577 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3116 | 15.8173 |
| controlled_vs_candidate_no_context | context_overlap | 0.0730 | 2.1265 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2079 | 5.3744 |
| controlled_vs_candidate_no_context | persona_style | 0.0557 | 0.0998 |
| controlled_vs_candidate_no_context | distinct1 | -0.0077 | -0.0081 |
| controlled_vs_candidate_no_context | length_score | 0.1667 | 0.4651 |
| controlled_vs_candidate_no_context | sentence_score | 0.1750 | 0.2327 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0411 | 0.9642 |
| controlled_vs_candidate_no_context | overall_quality | 0.1688 | 0.8304 |
| controlled_vs_baseline_no_context | context_relevance | 0.2148 | 4.3592 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1302 | 0.6857 |
| controlled_vs_baseline_no_context | naturalness | -0.0142 | -0.0160 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.2789 | 5.3217 |
| controlled_vs_baseline_no_context | context_overlap | 0.0653 | 1.5562 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1599 | 1.8444 |
| controlled_vs_baseline_no_context | persona_style | 0.0113 | 0.0187 |
| controlled_vs_baseline_no_context | distinct1 | -0.0346 | -0.0354 |
| controlled_vs_baseline_no_context | length_score | -0.0167 | -0.0308 |
| controlled_vs_baseline_no_context | sentence_score | 0.0292 | 0.0325 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0317 | 0.6094 |
| controlled_vs_baseline_no_context | overall_quality | 0.1311 | 0.5445 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0047 | -0.0177 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0356 | 0.1112 |
| controlled_alt_vs_controlled_default | naturalness | -0.0095 | -0.0108 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0041 | -0.0124 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0060 | -0.0556 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0395 | 0.1601 |
| controlled_alt_vs_controlled_default | persona_style | 0.0200 | 0.0325 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0049 | -0.0052 |
| controlled_alt_vs_controlled_default | length_score | -0.0208 | -0.0397 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0292 | -0.0315 |
| controlled_alt_vs_controlled_default | bertscore_f1 | -0.0203 | -0.2427 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0057 | 0.0153 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1525 | 1.4265 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1872 | 1.1114 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0718 | 0.0905 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2009 | 1.5910 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0396 | 0.6417 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.2067 | 2.6050 |
| controlled_alt_vs_proposed_raw | persona_style | 0.1090 | 0.2078 |
| controlled_alt_vs_proposed_raw | distinct1 | 0.0020 | 0.0022 |
| controlled_alt_vs_proposed_raw | length_score | 0.2986 | 1.4527 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1167 | 0.1493 |
| controlled_alt_vs_proposed_raw | bertscore_f1 | -0.0047 | -0.0689 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1351 | 0.5566 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2353 | 9.7701 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.2131 | 1.4945 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0383 | 0.0463 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3074 | 15.6090 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0670 | 1.9527 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.2474 | 6.3949 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0757 | 0.1356 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0127 | -0.0133 |
| controlled_alt_vs_candidate_no_context | length_score | 0.1458 | 0.4070 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1458 | 0.1939 |
| controlled_alt_vs_candidate_no_context | bertscore_f1 | 0.0208 | 0.4875 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1745 | 0.8585 |
| controlled_alt_vs_baseline_no_context | context_relevance | 0.2101 | 4.2645 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 0.1658 | 0.8731 |
| controlled_alt_vs_baseline_no_context | naturalness | -0.0237 | -0.0267 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 0.2747 | 5.2434 |
| controlled_alt_vs_baseline_no_context | context_overlap | 0.0594 | 1.4141 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 0.1994 | 2.2998 |
| controlled_alt_vs_baseline_no_context | persona_style | 0.0312 | 0.0518 |
| controlled_alt_vs_baseline_no_context | distinct1 | -0.0395 | -0.0404 |
| controlled_alt_vs_baseline_no_context | length_score | -0.0375 | -0.0692 |
| controlled_alt_vs_baseline_no_context | sentence_score | 0.0000 | 0.0000 |
| controlled_alt_vs_baseline_no_context | bertscore_f1 | 0.0114 | 0.2188 |
| controlled_alt_vs_baseline_no_context | overall_quality | 0.1368 | 0.5682 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2148 | 4.3592 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1302 | 0.6857 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0142 | -0.0160 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.2789 | 5.3217 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0653 | 1.5562 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1599 | 1.8444 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0113 | 0.0187 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0346 | -0.0354 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | -0.0167 | -0.0308 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0292 | 0.0325 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0317 | 0.6094 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1311 | 0.5445 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0828 | (0.0342, 0.1357) | 0.0003 | 0.0828 | (0.0257, 0.1275) | 0.0013 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0259 | (-0.0424, 0.0920) | 0.2270 | 0.0259 | (-0.0210, 0.0706) | 0.1440 |
| proposed_vs_candidate_no_context | naturalness | -0.0335 | (-0.0671, 0.0006) | 0.9730 | -0.0335 | (-0.0678, -0.0050) | 0.9940 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.1066 | (0.0403, 0.1815) | 0.0003 | 0.1066 | (0.0364, 0.1643) | 0.0013 |
| proposed_vs_candidate_no_context | context_overlap | 0.0274 | (0.0156, 0.0399) | 0.0000 | 0.0274 | (0.0124, 0.0408) | 0.0000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0407 | (-0.0327, 0.1200) | 0.1563 | 0.0407 | (-0.0023, 0.0893) | 0.0563 |
| proposed_vs_candidate_no_context | persona_style | -0.0334 | (-0.1073, 0.0213) | 0.8560 | -0.0334 | (-0.1086, 0.0141) | 0.8873 |
| proposed_vs_candidate_no_context | distinct1 | -0.0147 | (-0.0349, 0.0034) | 0.9397 | -0.0147 | (-0.0315, -0.0013) | 0.9840 |
| proposed_vs_candidate_no_context | length_score | -0.1528 | (-0.2681, -0.0389) | 0.9947 | -0.1528 | (-0.2619, -0.0555) | 1.0000 |
| proposed_vs_candidate_no_context | sentence_score | 0.0292 | (-0.0729, 0.1313) | 0.3413 | 0.0292 | (-0.0795, 0.1217) | 0.3253 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0255 | (0.0076, 0.0470) | 0.0013 | 0.0255 | (0.0006, 0.0487) | 0.0213 |
| proposed_vs_candidate_no_context | overall_quality | 0.0394 | (-0.0024, 0.0818) | 0.0343 | 0.0394 | (-0.0006, 0.0682) | 0.0263 |
| proposed_vs_baseline_no_context | context_relevance | 0.0576 | (0.0179, 0.1051) | 0.0020 | 0.0576 | (0.0101, 0.0957) | 0.0060 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0214 | (-0.1045, 0.0486) | 0.6980 | -0.0214 | (-0.1351, 0.0627) | 0.6647 |
| proposed_vs_baseline_no_context | naturalness | -0.0955 | (-0.1177, -0.0729) | 1.0000 | -0.0955 | (-0.1128, -0.0807) | 1.0000 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0739 | (0.0180, 0.1392) | 0.0040 | 0.0739 | (0.0114, 0.1261) | 0.0113 |
| proposed_vs_baseline_no_context | context_overlap | 0.0198 | (0.0071, 0.0326) | 0.0027 | 0.0198 | (0.0016, 0.0352) | 0.0190 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0073 | (-0.1046, 0.0736) | 0.5550 | -0.0073 | (-0.1326, 0.0901) | 0.5687 |
| proposed_vs_baseline_no_context | persona_style | -0.0778 | (-0.1614, -0.0125) | 0.9957 | -0.0778 | (-0.2139, -0.0068) | 1.0000 |
| proposed_vs_baseline_no_context | distinct1 | -0.0415 | (-0.0597, -0.0200) | 1.0000 | -0.0415 | (-0.0669, -0.0216) | 1.0000 |
| proposed_vs_baseline_no_context | length_score | -0.3361 | (-0.4306, -0.2458) | 1.0000 | -0.3361 | (-0.3985, -0.2787) | 1.0000 |
| proposed_vs_baseline_no_context | sentence_score | -0.1167 | (-0.2042, -0.0146) | 0.9920 | -0.1167 | (-0.1969, -0.0333) | 1.0000 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0161 | (-0.0070, 0.0402) | 0.0903 | 0.0161 | (-0.0152, 0.0451) | 0.1503 |
| proposed_vs_baseline_no_context | overall_quality | 0.0018 | (-0.0332, 0.0347) | 0.4653 | 0.0018 | (-0.0529, 0.0396) | 0.4690 |
| controlled_vs_proposed_raw | context_relevance | 0.1572 | (0.0864, 0.2292) | 0.0000 | 0.1572 | (0.1016, 0.2061) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.1516 | (0.0949, 0.2103) | 0.0000 | 0.1516 | (0.0906, 0.2433) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0813 | (0.0403, 0.1206) | 0.0000 | 0.0813 | (0.0342, 0.1422) | 0.0000 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2050 | (0.1146, 0.2959) | 0.0000 | 0.2050 | (0.1285, 0.2641) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0456 | (0.0186, 0.0780) | 0.0000 | 0.0456 | (0.0119, 0.0861) | 0.0010 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1673 | (0.1016, 0.2395) | 0.0000 | 0.1673 | (0.0981, 0.2679) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.0891 | (0.0183, 0.1759) | 0.0023 | 0.0891 | (0.0096, 0.2466) | 0.0037 |
| controlled_vs_proposed_raw | distinct1 | 0.0070 | (-0.0186, 0.0324) | 0.2973 | 0.0070 | (-0.0172, 0.0405) | 0.3010 |
| controlled_vs_proposed_raw | length_score | 0.3194 | (0.1639, 0.4750) | 0.0000 | 0.3194 | (0.1422, 0.5371) | 0.0000 |
| controlled_vs_proposed_raw | sentence_score | 0.1458 | (0.0437, 0.2333) | 0.0047 | 0.1458 | (0.0795, 0.2395) | 0.0000 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0156 | (-0.0115, 0.0436) | 0.1437 | 0.0156 | (-0.0129, 0.0543) | 0.1703 |
| controlled_vs_proposed_raw | overall_quality | 0.1294 | (0.0860, 0.1734) | 0.0000 | 0.1294 | (0.0832, 0.1850) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2400 | (0.1904, 0.2928) | 0.0000 | 0.2400 | (0.1797, 0.3016) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1775 | (0.1203, 0.2367) | 0.0000 | 0.1775 | (0.1348, 0.2452) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.0477 | (-0.0004, 0.0953) | 0.0263 | 0.0477 | (-0.0038, 0.1090) | 0.0363 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3116 | (0.2437, 0.3813) | 0.0000 | 0.3116 | (0.2352, 0.3945) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0730 | (0.0516, 0.0983) | 0.0000 | 0.0730 | (0.0446, 0.1051) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2079 | (0.1377, 0.2863) | 0.0000 | 0.2079 | (0.1500, 0.2952) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.0557 | (0.0018, 0.1276) | 0.0200 | 0.0557 | (-0.0015, 0.1572) | 0.0387 |
| controlled_vs_candidate_no_context | distinct1 | -0.0077 | (-0.0328, 0.0179) | 0.7197 | -0.0077 | (-0.0300, 0.0193) | 0.7313 |
| controlled_vs_candidate_no_context | length_score | 0.1667 | (-0.0223, 0.3486) | 0.0393 | 0.1667 | (-0.0253, 0.4080) | 0.0467 |
| controlled_vs_candidate_no_context | sentence_score | 0.1750 | (0.0875, 0.2625) | 0.0003 | 0.1750 | (0.0667, 0.2739) | 0.0017 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0411 | (0.0161, 0.0674) | 0.0017 | 0.0411 | (0.0136, 0.0688) | 0.0007 |
| controlled_vs_candidate_no_context | overall_quality | 0.1688 | (0.1398, 0.1976) | 0.0000 | 0.1688 | (0.1315, 0.2069) | 0.0000 |
| controlled_vs_baseline_no_context | context_relevance | 0.2148 | (0.1710, 0.2652) | 0.0000 | 0.2148 | (0.1586, 0.2734) | 0.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1302 | (0.0493, 0.2003) | 0.0013 | 0.1302 | (0.0385, 0.2053) | 0.0050 |
| controlled_vs_baseline_no_context | naturalness | -0.0142 | (-0.0569, 0.0280) | 0.7477 | -0.0142 | (-0.0573, 0.0419) | 0.7047 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.2789 | (0.2216, 0.3418) | 0.0000 | 0.2789 | (0.2080, 0.3571) | 0.0000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0653 | (0.0419, 0.0971) | 0.0000 | 0.0653 | (0.0355, 0.1003) | 0.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1599 | (0.0665, 0.2460) | 0.0007 | 0.1599 | (0.0426, 0.2569) | 0.0060 |
| controlled_vs_baseline_no_context | persona_style | 0.0113 | (-0.0200, 0.0528) | 0.2970 | 0.0113 | (-0.0105, 0.0433) | 0.1883 |
| controlled_vs_baseline_no_context | distinct1 | -0.0346 | (-0.0578, -0.0100) | 0.9980 | -0.0346 | (-0.0523, -0.0079) | 0.9917 |
| controlled_vs_baseline_no_context | length_score | -0.0167 | (-0.1722, 0.1403) | 0.5863 | -0.0167 | (-0.1958, 0.1972) | 0.5467 |
| controlled_vs_baseline_no_context | sentence_score | 0.0292 | (-0.0437, 0.1024) | 0.2943 | 0.0292 | (-0.0673, 0.1432) | 0.3330 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0317 | (0.0054, 0.0568) | 0.0110 | 0.0317 | (0.0013, 0.0573) | 0.0203 |
| controlled_vs_baseline_no_context | overall_quality | 0.1311 | (0.0992, 0.1646) | 0.0000 | 0.1311 | (0.0898, 0.1703) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0047 | (-0.0486, 0.0380) | 0.5777 | -0.0047 | (-0.0390, 0.0288) | 0.5933 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0356 | (-0.0271, 0.1142) | 0.1697 | 0.0356 | (-0.0227, 0.1172) | 0.1587 |
| controlled_alt_vs_controlled_default | naturalness | -0.0095 | (-0.0440, 0.0220) | 0.6940 | -0.0095 | (-0.0365, 0.0144) | 0.7713 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0041 | (-0.0581, 0.0527) | 0.5470 | -0.0041 | (-0.0455, 0.0379) | 0.5940 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0060 | (-0.0320, 0.0151) | 0.6533 | -0.0060 | (-0.0391, 0.0162) | 0.6267 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0395 | (-0.0391, 0.1383) | 0.1990 | 0.0395 | (-0.0344, 0.1441) | 0.1850 |
| controlled_alt_vs_controlled_default | persona_style | 0.0200 | (0.0010, 0.0457) | 0.0103 | 0.0200 | (0.0022, 0.0502) | 0.0027 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0049 | (-0.0256, 0.0162) | 0.6830 | -0.0049 | (-0.0231, 0.0115) | 0.7200 |
| controlled_alt_vs_controlled_default | length_score | -0.0208 | (-0.1764, 0.1292) | 0.5890 | -0.0208 | (-0.1333, 0.0800) | 0.6690 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0292 | (-0.1021, 0.0437) | 0.8090 | -0.0292 | (-0.0875, 0.0280) | 0.9147 |
| controlled_alt_vs_controlled_default | bertscore_f1 | -0.0203 | (-0.0502, 0.0066) | 0.9193 | -0.0203 | (-0.0394, -0.0015) | 0.9870 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0057 | (-0.0202, 0.0324) | 0.3600 | 0.0057 | (-0.0201, 0.0362) | 0.3303 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1525 | (0.0827, 0.2211) | 0.0000 | 0.1525 | (0.0956, 0.2055) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1872 | (0.0990, 0.2830) | 0.0000 | 0.1872 | (0.0973, 0.3017) | 0.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0718 | (0.0279, 0.1137) | 0.0007 | 0.0718 | (0.0322, 0.1219) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2009 | (0.1115, 0.2953) | 0.0000 | 0.2009 | (0.1261, 0.2727) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0396 | (0.0207, 0.0596) | 0.0000 | 0.0396 | (0.0220, 0.0577) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.2067 | (0.1071, 0.3183) | 0.0000 | 0.2067 | (0.1054, 0.3384) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 0.1090 | (0.0295, 0.2015) | 0.0007 | 0.1090 | (0.0172, 0.2609) | 0.0053 |
| controlled_alt_vs_proposed_raw | distinct1 | 0.0020 | (-0.0241, 0.0248) | 0.4400 | 0.0020 | (-0.0233, 0.0323) | 0.4197 |
| controlled_alt_vs_proposed_raw | length_score | 0.2986 | (0.1167, 0.4625) | 0.0013 | 0.2986 | (0.1518, 0.4727) | 0.0000 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1167 | (0.0292, 0.2042) | 0.0083 | 0.1167 | (0.0625, 0.1925) | 0.0000 |
| controlled_alt_vs_proposed_raw | bertscore_f1 | -0.0047 | (-0.0314, 0.0222) | 0.6223 | -0.0047 | (-0.0238, 0.0239) | 0.6557 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1351 | (0.0866, 0.1843) | 0.0000 | 0.1351 | (0.0840, 0.1905) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2353 | (0.1868, 0.2846) | 0.0000 | 0.2353 | (0.1678, 0.3103) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.2131 | (0.1361, 0.3045) | 0.0000 | 0.2131 | (0.1465, 0.3022) | 0.0000 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0383 | (-0.0199, 0.0887) | 0.0897 | 0.0383 | (-0.0211, 0.0939) | 0.1007 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3074 | (0.2411, 0.3725) | 0.0000 | 0.3074 | (0.2208, 0.4046) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0670 | (0.0517, 0.0837) | 0.0000 | 0.0670 | (0.0474, 0.0830) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.2474 | (0.1556, 0.3611) | 0.0000 | 0.2474 | (0.1735, 0.3464) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0757 | (0.0154, 0.1488) | 0.0047 | 0.0757 | (0.0079, 0.1852) | 0.0043 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0127 | (-0.0363, 0.0102) | 0.8450 | -0.0127 | (-0.0303, 0.0099) | 0.8713 |
| controlled_alt_vs_candidate_no_context | length_score | 0.1458 | (-0.0764, 0.3528) | 0.0970 | 0.1458 | (-0.0747, 0.3611) | 0.0907 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1458 | (0.0583, 0.2333) | 0.0017 | 0.1458 | (0.0456, 0.2188) | 0.0037 |
| controlled_alt_vs_candidate_no_context | bertscore_f1 | 0.0208 | (-0.0029, 0.0435) | 0.0403 | 0.0208 | (0.0000, 0.0391) | 0.0243 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1745 | (0.1454, 0.2053) | 0.0000 | 0.1745 | (0.1364, 0.2124) | 0.0000 |
| controlled_alt_vs_baseline_no_context | context_relevance | 0.2101 | (0.1594, 0.2613) | 0.0000 | 0.2101 | (0.1512, 0.2829) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 0.1658 | (0.0631, 0.2688) | 0.0003 | 0.1658 | (0.1361, 0.2005) | 0.0000 |
| controlled_alt_vs_baseline_no_context | naturalness | -0.0237 | (-0.0635, 0.0140) | 0.8947 | -0.0237 | (-0.0609, 0.0203) | 0.8413 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 0.2747 | (0.2076, 0.3429) | 0.0000 | 0.2747 | (0.1952, 0.3745) | 0.0000 |
| controlled_alt_vs_baseline_no_context | context_overlap | 0.0594 | (0.0409, 0.0796) | 0.0000 | 0.0594 | (0.0352, 0.0776) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 0.1994 | (0.0819, 0.3242) | 0.0003 | 0.1994 | (0.1569, 0.2488) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_style | 0.0312 | (-0.0174, 0.0959) | 0.1563 | 0.0312 | (-0.0064, 0.0941) | 0.1083 |
| controlled_alt_vs_baseline_no_context | distinct1 | -0.0395 | (-0.0637, -0.0164) | 1.0000 | -0.0395 | (-0.0490, -0.0242) | 1.0000 |
| controlled_alt_vs_baseline_no_context | length_score | -0.0375 | (-0.1986, 0.1098) | 0.6920 | -0.0375 | (-0.1940, 0.1359) | 0.6677 |
| controlled_alt_vs_baseline_no_context | sentence_score | 0.0000 | (-0.0875, 0.0875) | 0.5643 | 0.0000 | (-0.0729, 0.0875) | 0.5530 |
| controlled_alt_vs_baseline_no_context | bertscore_f1 | 0.0114 | (-0.0178, 0.0412) | 0.2220 | 0.0114 | (-0.0127, 0.0345) | 0.1740 |
| controlled_alt_vs_baseline_no_context | overall_quality | 0.1368 | (0.0975, 0.1740) | 0.0000 | 0.1368 | (0.1091, 0.1635) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2148 | (0.1714, 0.2654) | 0.0000 | 0.2148 | (0.1592, 0.2734) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1302 | (0.0486, 0.2012) | 0.0017 | 0.1302 | (0.0445, 0.2051) | 0.0020 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0142 | (-0.0559, 0.0276) | 0.7353 | -0.0142 | (-0.0589, 0.0450) | 0.6870 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.2789 | (0.2206, 0.3428) | 0.0000 | 0.2789 | (0.2074, 0.3636) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0653 | (0.0417, 0.0961) | 0.0000 | 0.0653 | (0.0361, 0.1000) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1599 | (0.0682, 0.2461) | 0.0010 | 0.1599 | (0.0420, 0.2558) | 0.0060 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0113 | (-0.0207, 0.0518) | 0.2900 | 0.0113 | (-0.0108, 0.0435) | 0.2093 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0346 | (-0.0581, -0.0109) | 0.9977 | -0.0346 | (-0.0532, -0.0078) | 0.9917 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | -0.0167 | (-0.1750, 0.1306) | 0.5833 | -0.0167 | (-0.1926, 0.2096) | 0.5357 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0292 | (-0.0437, 0.1167) | 0.3040 | 0.0292 | (-0.0604, 0.1432) | 0.3197 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0317 | (0.0049, 0.0574) | 0.0120 | 0.0317 | (0.0014, 0.0593) | 0.0210 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1311 | (0.0976, 0.1646) | 0.0000 | 0.1311 | (0.0899, 0.1702) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 14 | 3 | 7 | 0.7292 | 0.8235 |
| proposed_vs_candidate_no_context | persona_consistency | 7 | 5 | 12 | 0.5417 | 0.5833 |
| proposed_vs_candidate_no_context | naturalness | 5 | 12 | 7 | 0.3542 | 0.2941 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 12 | 2 | 10 | 0.7083 | 0.8571 |
| proposed_vs_candidate_no_context | context_overlap | 15 | 2 | 7 | 0.7708 | 0.8824 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 5 | 3 | 16 | 0.5417 | 0.6250 |
| proposed_vs_candidate_no_context | persona_style | 2 | 4 | 18 | 0.4583 | 0.3333 |
| proposed_vs_candidate_no_context | distinct1 | 7 | 8 | 9 | 0.4792 | 0.4667 |
| proposed_vs_candidate_no_context | length_score | 3 | 13 | 8 | 0.2917 | 0.1875 |
| proposed_vs_candidate_no_context | sentence_score | 8 | 6 | 10 | 0.5417 | 0.5714 |
| proposed_vs_candidate_no_context | bertscore_f1 | 14 | 5 | 5 | 0.6875 | 0.7368 |
| proposed_vs_candidate_no_context | overall_quality | 12 | 7 | 5 | 0.6042 | 0.6316 |
| proposed_vs_baseline_no_context | context_relevance | 14 | 10 | 0 | 0.5833 | 0.5833 |
| proposed_vs_baseline_no_context | persona_consistency | 6 | 7 | 11 | 0.4792 | 0.4615 |
| proposed_vs_baseline_no_context | naturalness | 1 | 23 | 0 | 0.0417 | 0.0417 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 12 | 4 | 8 | 0.6667 | 0.7500 |
| proposed_vs_baseline_no_context | context_overlap | 18 | 6 | 0 | 0.7500 | 0.7500 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 5 | 5 | 14 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context | persona_style | 1 | 5 | 18 | 0.4167 | 0.1667 |
| proposed_vs_baseline_no_context | distinct1 | 3 | 19 | 2 | 0.1667 | 0.1364 |
| proposed_vs_baseline_no_context | length_score | 1 | 22 | 1 | 0.0625 | 0.0435 |
| proposed_vs_baseline_no_context | sentence_score | 3 | 11 | 10 | 0.3333 | 0.2143 |
| proposed_vs_baseline_no_context | bertscore_f1 | 14 | 10 | 0 | 0.5833 | 0.5833 |
| proposed_vs_baseline_no_context | overall_quality | 10 | 14 | 0 | 0.4167 | 0.4167 |
| controlled_vs_proposed_raw | context_relevance | 21 | 3 | 0 | 0.8750 | 0.8750 |
| controlled_vs_proposed_raw | persona_consistency | 17 | 1 | 6 | 0.8333 | 0.9444 |
| controlled_vs_proposed_raw | naturalness | 18 | 6 | 0 | 0.7500 | 0.7500 |
| controlled_vs_proposed_raw | context_keyword_coverage | 21 | 3 | 0 | 0.8750 | 0.8750 |
| controlled_vs_proposed_raw | context_overlap | 15 | 9 | 0 | 0.6250 | 0.6250 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 17 | 1 | 6 | 0.8333 | 0.9444 |
| controlled_vs_proposed_raw | persona_style | 7 | 2 | 15 | 0.6042 | 0.7778 |
| controlled_vs_proposed_raw | distinct1 | 15 | 9 | 0 | 0.6250 | 0.6250 |
| controlled_vs_proposed_raw | length_score | 18 | 6 | 0 | 0.7500 | 0.7500 |
| controlled_vs_proposed_raw | sentence_score | 13 | 3 | 8 | 0.7083 | 0.8125 |
| controlled_vs_proposed_raw | bertscore_f1 | 14 | 10 | 0 | 0.5833 | 0.5833 |
| controlled_vs_proposed_raw | overall_quality | 21 | 3 | 0 | 0.8750 | 0.8750 |
| controlled_vs_candidate_no_context | context_relevance | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 19 | 1 | 4 | 0.8750 | 0.9500 |
| controlled_vs_candidate_no_context | naturalness | 14 | 10 | 0 | 0.5833 | 0.5833 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 22 | 0 | 2 | 0.9583 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 19 | 1 | 4 | 0.8750 | 0.9500 |
| controlled_vs_candidate_no_context | persona_style | 4 | 2 | 18 | 0.5417 | 0.6667 |
| controlled_vs_candidate_no_context | distinct1 | 9 | 13 | 2 | 0.4167 | 0.4091 |
| controlled_vs_candidate_no_context | length_score | 16 | 8 | 0 | 0.6667 | 0.6667 |
| controlled_vs_candidate_no_context | sentence_score | 14 | 2 | 8 | 0.7500 | 0.8750 |
| controlled_vs_candidate_no_context | bertscore_f1 | 19 | 5 | 0 | 0.7917 | 0.7917 |
| controlled_vs_candidate_no_context | overall_quality | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | context_relevance | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 19 | 2 | 3 | 0.8542 | 0.9048 |
| controlled_vs_baseline_no_context | naturalness | 10 | 13 | 1 | 0.4375 | 0.4348 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 23 | 0 | 1 | 0.9792 | 1.0000 |
| controlled_vs_baseline_no_context | context_overlap | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 19 | 2 | 3 | 0.8542 | 0.9048 |
| controlled_vs_baseline_no_context | persona_style | 2 | 4 | 18 | 0.4583 | 0.3333 |
| controlled_vs_baseline_no_context | distinct1 | 3 | 16 | 5 | 0.2292 | 0.1579 |
| controlled_vs_baseline_no_context | length_score | 10 | 12 | 2 | 0.4583 | 0.4545 |
| controlled_vs_baseline_no_context | sentence_score | 5 | 3 | 16 | 0.5417 | 0.6250 |
| controlled_vs_baseline_no_context | bertscore_f1 | 16 | 8 | 0 | 0.6667 | 0.6667 |
| controlled_vs_baseline_no_context | overall_quality | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_alt_vs_controlled_default | context_relevance | 9 | 11 | 4 | 0.4583 | 0.4500 |
| controlled_alt_vs_controlled_default | persona_consistency | 7 | 4 | 13 | 0.5625 | 0.6364 |
| controlled_alt_vs_controlled_default | naturalness | 11 | 8 | 5 | 0.5625 | 0.5789 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 7 | 8 | 9 | 0.4792 | 0.4667 |
| controlled_alt_vs_controlled_default | context_overlap | 10 | 10 | 4 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 4 | 4 | 16 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | persona_style | 5 | 1 | 18 | 0.5833 | 0.8333 |
| controlled_alt_vs_controlled_default | distinct1 | 8 | 9 | 7 | 0.4792 | 0.4706 |
| controlled_alt_vs_controlled_default | length_score | 9 | 8 | 7 | 0.5208 | 0.5294 |
| controlled_alt_vs_controlled_default | sentence_score | 3 | 5 | 16 | 0.4583 | 0.3750 |
| controlled_alt_vs_controlled_default | bertscore_f1 | 10 | 10 | 4 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | overall_quality | 10 | 10 | 4 | 0.5000 | 0.5000 |
| controlled_alt_vs_proposed_raw | context_relevance | 19 | 5 | 0 | 0.7917 | 0.7917 |
| controlled_alt_vs_proposed_raw | persona_consistency | 17 | 2 | 5 | 0.8125 | 0.8947 |
| controlled_alt_vs_proposed_raw | naturalness | 18 | 6 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 19 | 4 | 1 | 0.8125 | 0.8261 |
| controlled_alt_vs_proposed_raw | context_overlap | 19 | 5 | 0 | 0.7917 | 0.7917 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 17 | 2 | 5 | 0.8125 | 0.8947 |
| controlled_alt_vs_proposed_raw | persona_style | 7 | 2 | 15 | 0.6042 | 0.7778 |
| controlled_alt_vs_proposed_raw | distinct1 | 13 | 9 | 2 | 0.5833 | 0.5909 |
| controlled_alt_vs_proposed_raw | length_score | 18 | 6 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_proposed_raw | sentence_score | 10 | 2 | 12 | 0.6667 | 0.8333 |
| controlled_alt_vs_proposed_raw | bertscore_f1 | 11 | 13 | 0 | 0.4583 | 0.4583 |
| controlled_alt_vs_proposed_raw | overall_quality | 19 | 5 | 0 | 0.7917 | 0.7917 |
| controlled_alt_vs_candidate_no_context | context_relevance | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 20 | 1 | 3 | 0.8958 | 0.9524 |
| controlled_alt_vs_candidate_no_context | naturalness | 14 | 10 | 0 | 0.5833 | 0.5833 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 22 | 0 | 2 | 0.9583 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 20 | 1 | 3 | 0.8958 | 0.9524 |
| controlled_alt_vs_candidate_no_context | persona_style | 5 | 2 | 17 | 0.5625 | 0.7143 |
| controlled_alt_vs_candidate_no_context | distinct1 | 8 | 11 | 5 | 0.4375 | 0.4211 |
| controlled_alt_vs_candidate_no_context | length_score | 16 | 8 | 0 | 0.6667 | 0.6667 |
| controlled_alt_vs_candidate_no_context | sentence_score | 12 | 2 | 10 | 0.7083 | 0.8571 |
| controlled_alt_vs_candidate_no_context | bertscore_f1 | 15 | 9 | 0 | 0.6250 | 0.6250 |
| controlled_alt_vs_candidate_no_context | overall_quality | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context | context_relevance | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 18 | 2 | 4 | 0.8333 | 0.9000 |
| controlled_alt_vs_baseline_no_context | naturalness | 12 | 11 | 1 | 0.5208 | 0.5217 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 22 | 0 | 2 | 0.9583 | 1.0000 |
| controlled_alt_vs_baseline_no_context | context_overlap | 22 | 2 | 0 | 0.9167 | 0.9167 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 18 | 2 | 4 | 0.8333 | 0.9000 |
| controlled_alt_vs_baseline_no_context | persona_style | 2 | 4 | 18 | 0.4583 | 0.3333 |
| controlled_alt_vs_baseline_no_context | distinct1 | 5 | 15 | 4 | 0.2917 | 0.2500 |
| controlled_alt_vs_baseline_no_context | length_score | 13 | 10 | 1 | 0.5625 | 0.5652 |
| controlled_alt_vs_baseline_no_context | sentence_score | 5 | 5 | 14 | 0.5000 | 0.5000 |
| controlled_alt_vs_baseline_no_context | bertscore_f1 | 13 | 11 | 0 | 0.5417 | 0.5417 |
| controlled_alt_vs_baseline_no_context | overall_quality | 23 | 1 | 0 | 0.9583 | 0.9583 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 24 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 19 | 2 | 3 | 0.8542 | 0.9048 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 10 | 13 | 1 | 0.4375 | 0.4348 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 23 | 0 | 1 | 0.9792 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 23 | 1 | 0 | 0.9583 | 0.9583 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 19 | 2 | 3 | 0.8542 | 0.9048 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 2 | 4 | 18 | 0.4583 | 0.3333 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | 3 | 16 | 5 | 0.2292 | 0.1579 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 10 | 12 | 2 | 0.4583 | 0.4545 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 5 | 3 | 16 | 0.5417 | 0.6250 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 16 | 8 | 0 | 0.6667 | 0.6667 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 23 | 1 | 0 | 0.9583 | 0.9583 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.2917 | 0.5417 | 0.4583 |
| proposed_contextual_controlled_alt | 0.0000 | 0.0000 | 0.2083 | 0.5833 | 0.4167 |
| proposed_contextual | 0.0000 | 0.0000 | 0.4167 | 0.0000 | 0.0000 |
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
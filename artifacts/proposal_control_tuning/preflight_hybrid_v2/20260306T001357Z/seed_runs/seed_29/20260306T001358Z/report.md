# Proposal Alignment Evaluation Report

- Run ID: `20260306T001358Z`
- Generated: `2026-03-06T00:20:48.405790+00:00`
- Scenarios: `artifacts\proposal_control_tuning\preflight_hybrid_v2\20260306T001357Z\seed_runs\seed_29\20260306T001358Z\scenarios.jsonl`
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
| proposed_contextual_controlled | 0.3163 (0.2767, 0.3575) | 0.4065 (0.3320, 0.4900) | 0.8569 (0.8253, 0.8892) | 0.4188 (0.3936, 0.4456) | 0.0816 |
| proposed_contextual_controlled_alt | 0.2576 (0.2143, 0.3014) | 0.3890 (0.3259, 0.4550) | 0.8751 (0.8465, 0.9039) | 0.3901 (0.3652, 0.4150) | 0.0639 |
| proposed_contextual | 0.1016 (0.0552, 0.1571) | 0.1834 (0.1188, 0.2620) | 0.7983 (0.7737, 0.8245) | 0.2471 (0.2074, 0.2937) | 0.0725 |
| candidate_no_context | 0.0222 (0.0122, 0.0354) | 0.1604 (0.0967, 0.2434) | 0.8019 (0.7730, 0.8341) | 0.2035 (0.1798, 0.2309) | 0.0295 |
| baseline_no_context | 0.0355 (0.0180, 0.0578) | 0.2300 (0.1757, 0.2885) | 0.8918 (0.8655, 0.9158) | 0.2476 (0.2266, 0.2690) | 0.0344 |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0794 | 3.5750 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0229 | 0.1430 |
| proposed_vs_candidate_no_context | naturalness | -0.0037 | -0.0046 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.1042 | 6.8750 |
| proposed_vs_candidate_no_context | context_overlap | 0.0215 | 0.5554 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0153 | 0.1730 |
| proposed_vs_candidate_no_context | persona_style | 0.0536 | 0.1194 |
| proposed_vs_candidate_no_context | distinct1 | 0.0060 | 0.0064 |
| proposed_vs_candidate_no_context | length_score | -0.0375 | -0.1452 |
| proposed_vs_candidate_no_context | sentence_score | 0.0146 | 0.0187 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0430 | 1.4567 |
| proposed_vs_candidate_no_context | overall_quality | 0.0436 | 0.2142 |
| proposed_vs_baseline_no_context | context_relevance | 0.0660 | 1.8575 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0466 | -0.2025 |
| proposed_vs_baseline_no_context | naturalness | -0.0936 | -0.1049 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0855 | 2.5327 |
| proposed_vs_baseline_no_context | context_overlap | 0.0204 | 0.5156 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0456 | -0.3059 |
| proposed_vs_baseline_no_context | persona_style | -0.0503 | -0.0910 |
| proposed_vs_baseline_no_context | distinct1 | -0.0396 | -0.0406 |
| proposed_vs_baseline_no_context | length_score | -0.3375 | -0.6045 |
| proposed_vs_baseline_no_context | sentence_score | -0.1021 | -0.1137 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0381 | 1.1055 |
| proposed_vs_baseline_no_context | overall_quality | -0.0005 | -0.0019 |
| controlled_vs_proposed_raw | context_relevance | 0.2148 | 2.1149 |
| controlled_vs_proposed_raw | persona_consistency | 0.2231 | 1.2165 |
| controlled_vs_proposed_raw | naturalness | 0.0586 | 0.0734 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2807 | 2.3529 |
| controlled_vs_proposed_raw | context_overlap | 0.0608 | 1.0121 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2587 | 2.4981 |
| controlled_vs_proposed_raw | persona_style | 0.0805 | 0.1602 |
| controlled_vs_proposed_raw | distinct1 | -0.0064 | -0.0068 |
| controlled_vs_proposed_raw | length_score | 0.2375 | 1.0755 |
| controlled_vs_proposed_raw | sentence_score | 0.1458 | 0.1832 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0091 | 0.1255 |
| controlled_vs_proposed_raw | overall_quality | 0.1717 | 0.6949 |
| controlled_vs_candidate_no_context | context_relevance | 0.2941 | 13.2507 |
| controlled_vs_candidate_no_context | persona_consistency | 0.2460 | 1.5335 |
| controlled_vs_candidate_no_context | naturalness | 0.0550 | 0.0685 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3849 | 25.4042 |
| controlled_vs_candidate_no_context | context_overlap | 0.0823 | 2.1298 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2740 | 3.1034 |
| controlled_vs_candidate_no_context | persona_style | 0.1341 | 0.2987 |
| controlled_vs_candidate_no_context | distinct1 | -0.0004 | -0.0004 |
| controlled_vs_candidate_no_context | length_score | 0.2000 | 0.7742 |
| controlled_vs_candidate_no_context | sentence_score | 0.1604 | 0.2053 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0521 | 1.7649 |
| controlled_vs_candidate_no_context | overall_quality | 0.2153 | 1.0580 |
| controlled_vs_baseline_no_context | context_relevance | 0.2808 | 7.9009 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1765 | 0.7676 |
| controlled_vs_baseline_no_context | naturalness | -0.0349 | -0.0392 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3663 | 10.8449 |
| controlled_vs_baseline_no_context | context_overlap | 0.0813 | 2.0496 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2131 | 1.4282 |
| controlled_vs_baseline_no_context | persona_style | 0.0302 | 0.0546 |
| controlled_vs_baseline_no_context | distinct1 | -0.0460 | -0.0471 |
| controlled_vs_baseline_no_context | length_score | -0.1000 | -0.1791 |
| controlled_vs_baseline_no_context | sentence_score | 0.0438 | 0.0487 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0471 | 1.3696 |
| controlled_vs_baseline_no_context | overall_quality | 0.1712 | 0.6916 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0587 | -0.1855 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0175 | -0.0429 |
| controlled_alt_vs_controlled_default | naturalness | 0.0182 | 0.0212 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0770 | -0.1925 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0159 | -0.1312 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0276 | -0.0761 |
| controlled_alt_vs_controlled_default | persona_style | 0.0231 | 0.0395 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0063 | 0.0068 |
| controlled_alt_vs_controlled_default | length_score | 0.0736 | 0.1606 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0000 | 0.0000 |
| controlled_alt_vs_controlled_default | bertscore_f1 | -0.0177 | -0.2171 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0287 | -0.0686 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1561 | 1.5371 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.2056 | 1.1213 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0768 | 0.0962 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2037 | 1.7074 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0450 | 0.7481 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.2312 | 2.2318 |
| controlled_alt_vs_proposed_raw | persona_style | 0.1036 | 0.2061 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0001 | -0.0001 |
| controlled_alt_vs_proposed_raw | length_score | 0.3111 | 1.4088 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1458 | 0.1832 |
| controlled_alt_vs_proposed_raw | bertscore_f1 | -0.0086 | -0.1188 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1430 | 0.5787 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2354 | 10.6073 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.2286 | 1.4247 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0731 | 0.0912 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3079 | 20.3208 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0664 | 1.7191 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.2464 | 2.7910 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.1572 | 0.3501 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0059 | 0.0063 |
| controlled_alt_vs_candidate_no_context | length_score | 0.2736 | 1.0591 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1604 | 0.2053 |
| controlled_alt_vs_candidate_no_context | bertscore_f1 | 0.0344 | 1.1648 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1866 | 0.9169 |
| controlled_alt_vs_baseline_no_context | context_relevance | 0.2221 | 6.2498 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 0.1591 | 0.6917 |
| controlled_alt_vs_baseline_no_context | naturalness | -0.0168 | -0.0188 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 0.2893 | 8.5645 |
| controlled_alt_vs_baseline_no_context | context_overlap | 0.0654 | 1.6494 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 0.1855 | 1.2434 |
| controlled_alt_vs_baseline_no_context | persona_style | 0.0533 | 0.0963 |
| controlled_alt_vs_baseline_no_context | distinct1 | -0.0397 | -0.0406 |
| controlled_alt_vs_baseline_no_context | length_score | -0.0264 | -0.0473 |
| controlled_alt_vs_baseline_no_context | sentence_score | 0.0438 | 0.0487 |
| controlled_alt_vs_baseline_no_context | bertscore_f1 | 0.0294 | 0.8553 |
| controlled_alt_vs_baseline_no_context | overall_quality | 0.1425 | 0.5756 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2808 | 7.9009 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1765 | 0.7676 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0349 | -0.0392 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3663 | 10.8449 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0813 | 2.0496 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2131 | 1.4282 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0302 | 0.0546 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0460 | -0.0471 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | -0.1000 | -0.1791 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0438 | 0.0487 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0471 | 1.3696 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1712 | 0.6916 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0794 | (0.0316, 0.1376) | 0.0000 | 0.0794 | (0.0265, 0.1435) | 0.0007 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0229 | (-0.0603, 0.1072) | 0.2877 | 0.0229 | (0.0002, 0.0425) | 0.0243 |
| proposed_vs_candidate_no_context | naturalness | -0.0037 | (-0.0404, 0.0329) | 0.5800 | -0.0037 | (-0.0407, 0.0285) | 0.6087 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.1042 | (0.0404, 0.1806) | 0.0007 | 0.1042 | (0.0356, 0.1877) | 0.0007 |
| proposed_vs_candidate_no_context | context_overlap | 0.0215 | (0.0091, 0.0351) | 0.0000 | 0.0215 | (0.0076, 0.0388) | 0.0020 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0153 | (-0.0875, 0.1208) | 0.3587 | 0.0153 | (-0.0119, 0.0375) | 0.0950 |
| proposed_vs_candidate_no_context | persona_style | 0.0536 | (0.0138, 0.1023) | 0.0030 | 0.0536 | (0.0081, 0.1222) | 0.0240 |
| proposed_vs_candidate_no_context | distinct1 | 0.0060 | (-0.0128, 0.0240) | 0.2447 | 0.0060 | (-0.0059, 0.0188) | 0.1643 |
| proposed_vs_candidate_no_context | length_score | -0.0375 | (-0.1847, 0.1111) | 0.6880 | -0.0375 | (-0.1945, 0.0821) | 0.7113 |
| proposed_vs_candidate_no_context | sentence_score | 0.0146 | (-0.0583, 0.0875) | 0.4360 | 0.0146 | (-0.0539, 0.0955) | 0.4490 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0430 | (0.0185, 0.0703) | 0.0000 | 0.0430 | (0.0197, 0.0734) | 0.0000 |
| proposed_vs_candidate_no_context | overall_quality | 0.0436 | (0.0071, 0.0860) | 0.0113 | 0.0436 | (0.0129, 0.0730) | 0.0033 |
| proposed_vs_baseline_no_context | context_relevance | 0.0660 | (0.0102, 0.1260) | 0.0110 | 0.0660 | (0.0143, 0.1296) | 0.0087 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0466 | (-0.1129, 0.0266) | 0.9010 | -0.0466 | (-0.1189, 0.0510) | 0.8360 |
| proposed_vs_baseline_no_context | naturalness | -0.0936 | (-0.1306, -0.0558) | 1.0000 | -0.0936 | (-0.1264, -0.0393) | 0.9987 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0855 | (0.0110, 0.1701) | 0.0110 | 0.0855 | (0.0121, 0.1739) | 0.0097 |
| proposed_vs_baseline_no_context | context_overlap | 0.0204 | (0.0077, 0.0337) | 0.0007 | 0.0204 | (0.0078, 0.0364) | 0.0000 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0456 | (-0.1204, 0.0361) | 0.8777 | -0.0456 | (-0.1187, 0.0684) | 0.8177 |
| proposed_vs_baseline_no_context | persona_style | -0.0503 | (-0.1550, 0.0466) | 0.8257 | -0.0503 | (-0.2176, 0.0669) | 0.6643 |
| proposed_vs_baseline_no_context | distinct1 | -0.0396 | (-0.0592, -0.0203) | 1.0000 | -0.0396 | (-0.0543, -0.0222) | 1.0000 |
| proposed_vs_baseline_no_context | length_score | -0.3375 | (-0.4792, -0.1806) | 1.0000 | -0.3375 | (-0.4364, -0.1476) | 0.9973 |
| proposed_vs_baseline_no_context | sentence_score | -0.1021 | (-0.1896, 0.0000) | 0.9860 | -0.1021 | (-0.2052, 0.0368) | 0.9377 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0381 | (0.0161, 0.0633) | 0.0003 | 0.0381 | (0.0085, 0.0766) | 0.0063 |
| proposed_vs_baseline_no_context | overall_quality | -0.0005 | (-0.0443, 0.0454) | 0.5177 | -0.0005 | (-0.0442, 0.0569) | 0.5093 |
| controlled_vs_proposed_raw | context_relevance | 0.2148 | (0.1490, 0.2775) | 0.0000 | 0.2148 | (0.1362, 0.2905) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.2231 | (0.1215, 0.3246) | 0.0000 | 0.2231 | (0.0848, 0.3282) | 0.0013 |
| controlled_vs_proposed_raw | naturalness | 0.0586 | (0.0164, 0.1002) | 0.0040 | 0.0586 | (0.0073, 0.0971) | 0.0140 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2807 | (0.1903, 0.3619) | 0.0000 | 0.2807 | (0.1834, 0.3741) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0608 | (0.0412, 0.0805) | 0.0000 | 0.0608 | (0.0328, 0.0845) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2587 | (0.1403, 0.3750) | 0.0000 | 0.2587 | (0.1160, 0.3767) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.0805 | (-0.0244, 0.1877) | 0.0717 | 0.0805 | (-0.0748, 0.2543) | 0.1917 |
| controlled_vs_proposed_raw | distinct1 | -0.0064 | (-0.0307, 0.0156) | 0.7090 | -0.0064 | (-0.0331, 0.0161) | 0.7107 |
| controlled_vs_proposed_raw | length_score | 0.2375 | (0.0708, 0.4028) | 0.0033 | 0.2375 | (0.0614, 0.3748) | 0.0033 |
| controlled_vs_proposed_raw | sentence_score | 0.1458 | (0.0583, 0.2333) | 0.0010 | 0.1458 | (0.0667, 0.2074) | 0.0000 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0091 | (-0.0122, 0.0302) | 0.2067 | 0.0091 | (-0.0223, 0.0294) | 0.2493 |
| controlled_vs_proposed_raw | overall_quality | 0.1717 | (0.1190, 0.2192) | 0.0000 | 0.1717 | (0.1069, 0.2194) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2941 | (0.2589, 0.3331) | 0.0000 | 0.2941 | (0.2608, 0.3346) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.2460 | (0.1196, 0.3706) | 0.0000 | 0.2460 | (0.1165, 0.3510) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.0550 | (0.0010, 0.1062) | 0.0237 | 0.0550 | (-0.0205, 0.1042) | 0.0670 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3849 | (0.3390, 0.4335) | 0.0000 | 0.3849 | (0.3424, 0.4428) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0823 | (0.0634, 0.1015) | 0.0000 | 0.0823 | (0.0601, 0.0992) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2740 | (0.1293, 0.4179) | 0.0000 | 0.2740 | (0.1366, 0.3853) | 0.0003 |
| controlled_vs_candidate_no_context | persona_style | 0.1341 | (0.0281, 0.2499) | 0.0063 | 0.1341 | (-0.0402, 0.3154) | 0.0693 |
| controlled_vs_candidate_no_context | distinct1 | -0.0004 | (-0.0220, 0.0191) | 0.5050 | -0.0004 | (-0.0266, 0.0179) | 0.5070 |
| controlled_vs_candidate_no_context | length_score | 0.2000 | (-0.0194, 0.4139) | 0.0387 | 0.2000 | (-0.1111, 0.4128) | 0.0843 |
| controlled_vs_candidate_no_context | sentence_score | 0.1604 | (0.0729, 0.2333) | 0.0000 | 0.1604 | (0.0875, 0.2125) | 0.0000 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0521 | (0.0295, 0.0771) | 0.0000 | 0.0521 | (0.0333, 0.0700) | 0.0000 |
| controlled_vs_candidate_no_context | overall_quality | 0.2153 | (0.1734, 0.2545) | 0.0000 | 0.2153 | (0.1732, 0.2459) | 0.0000 |
| controlled_vs_baseline_no_context | context_relevance | 0.2808 | (0.2425, 0.3196) | 0.0000 | 0.2808 | (0.2520, 0.3161) | 0.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1765 | (0.0681, 0.2920) | 0.0010 | 0.1765 | (0.0274, 0.3200) | 0.0120 |
| controlled_vs_baseline_no_context | naturalness | -0.0349 | (-0.0731, 0.0041) | 0.9610 | -0.0349 | (-0.0699, -0.0009) | 0.9773 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3663 | (0.3134, 0.4235) | 0.0000 | 0.3663 | (0.3311, 0.4141) | 0.0000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0813 | (0.0657, 0.0977) | 0.0000 | 0.0813 | (0.0597, 0.0980) | 0.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2131 | (0.0913, 0.3417) | 0.0000 | 0.2131 | (0.0433, 0.3795) | 0.0087 |
| controlled_vs_baseline_no_context | persona_style | 0.0302 | (-0.0361, 0.1047) | 0.2057 | 0.0302 | (-0.0595, 0.1437) | 0.2943 |
| controlled_vs_baseline_no_context | distinct1 | -0.0460 | (-0.0588, -0.0317) | 1.0000 | -0.0460 | (-0.0605, -0.0327) | 1.0000 |
| controlled_vs_baseline_no_context | length_score | -0.1000 | (-0.2653, 0.0695) | 0.8737 | -0.1000 | (-0.2730, 0.0611) | 0.8900 |
| controlled_vs_baseline_no_context | sentence_score | 0.0437 | (-0.0292, 0.1167) | 0.1753 | 0.0437 | (-0.0125, 0.1114) | 0.1257 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0471 | (0.0222, 0.0716) | 0.0003 | 0.0471 | (0.0146, 0.0770) | 0.0013 |
| controlled_vs_baseline_no_context | overall_quality | 0.1712 | (0.1315, 0.2108) | 0.0000 | 0.1712 | (0.1232, 0.2120) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0587 | (-0.1231, 0.0019) | 0.9700 | -0.0587 | (-0.1345, 0.0161) | 0.9503 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0175 | (-0.1263, 0.0859) | 0.6260 | -0.0175 | (-0.1362, 0.1242) | 0.6237 |
| controlled_alt_vs_controlled_default | naturalness | 0.0182 | (-0.0236, 0.0624) | 0.2217 | 0.0182 | (-0.0202, 0.0647) | 0.1767 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0770 | (-0.1660, 0.0076) | 0.9643 | -0.0770 | (-0.1686, 0.0179) | 0.9453 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0159 | (-0.0358, 0.0044) | 0.9417 | -0.0159 | (-0.0312, 0.0078) | 0.9167 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0276 | (-0.1619, 0.0931) | 0.6553 | -0.0276 | (-0.1607, 0.1345) | 0.6443 |
| controlled_alt_vs_controlled_default | persona_style | 0.0231 | (-0.0327, 0.0810) | 0.2093 | 0.0231 | (-0.0319, 0.0940) | 0.2343 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0063 | (-0.0155, 0.0280) | 0.2907 | 0.0063 | (-0.0116, 0.0281) | 0.2707 |
| controlled_alt_vs_controlled_default | length_score | 0.0736 | (-0.1209, 0.2667) | 0.2207 | 0.0736 | (-0.0972, 0.2914) | 0.2037 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0000 | (-0.0729, 0.0729) | 0.5803 | 0.0000 | (-0.0913, 0.0761) | 0.5483 |
| controlled_alt_vs_controlled_default | bertscore_f1 | -0.0177 | (-0.0475, 0.0108) | 0.8823 | -0.0177 | (-0.0503, 0.0106) | 0.8793 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0287 | (-0.0668, 0.0059) | 0.9447 | -0.0287 | (-0.0623, 0.0080) | 0.9367 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1561 | (0.0916, 0.2163) | 0.0000 | 0.1561 | (0.1059, 0.1892) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.2056 | (0.1074, 0.2951) | 0.0000 | 0.2056 | (0.1561, 0.2485) | 0.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0768 | (0.0383, 0.1167) | 0.0000 | 0.0768 | (0.0274, 0.1210) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2037 | (0.1208, 0.2838) | 0.0000 | 0.2037 | (0.1379, 0.2490) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0450 | (0.0305, 0.0593) | 0.0000 | 0.0450 | (0.0293, 0.0577) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.2312 | (0.1157, 0.3405) | 0.0000 | 0.2312 | (0.1783, 0.2865) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 0.1036 | (0.0111, 0.2083) | 0.0120 | 0.1036 | (-0.0142, 0.2617) | 0.1067 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0001 | (-0.0195, 0.0192) | 0.4953 | -0.0001 | (-0.0220, 0.0152) | 0.4983 |
| controlled_alt_vs_proposed_raw | length_score | 0.3111 | (0.1402, 0.4708) | 0.0000 | 0.3111 | (0.1077, 0.5083) | 0.0007 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1458 | (0.0437, 0.2479) | 0.0050 | 0.1458 | (0.0350, 0.2333) | 0.0050 |
| controlled_alt_vs_proposed_raw | bertscore_f1 | -0.0086 | (-0.0356, 0.0168) | 0.7540 | -0.0086 | (-0.0420, 0.0192) | 0.7073 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1430 | (0.0896, 0.1920) | 0.0000 | 0.1430 | (0.1102, 0.1731) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2354 | (0.1869, 0.2851) | 0.0000 | 0.2354 | (0.1828, 0.2992) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.2286 | (0.1300, 0.3133) | 0.0000 | 0.2286 | (0.1770, 0.2634) | 0.0000 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0731 | (0.0314, 0.1138) | 0.0003 | 0.0731 | (0.0257, 0.1217) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3079 | (0.2415, 0.3734) | 0.0000 | 0.3079 | (0.2363, 0.3950) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0664 | (0.0555, 0.0791) | 0.0000 | 0.0664 | (0.0536, 0.0796) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.2464 | (0.1357, 0.3508) | 0.0000 | 0.2464 | (0.1855, 0.3000) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.1572 | (0.0600, 0.2713) | 0.0000 | 0.1572 | (0.0188, 0.3138) | 0.0017 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0059 | (-0.0103, 0.0211) | 0.2297 | 0.0059 | (-0.0153, 0.0207) | 0.2760 |
| controlled_alt_vs_candidate_no_context | length_score | 0.2736 | (0.0944, 0.4431) | 0.0017 | 0.2736 | (0.0963, 0.4833) | 0.0000 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1604 | (0.0583, 0.2479) | 0.0020 | 0.1604 | (0.0333, 0.2558) | 0.0100 |
| controlled_alt_vs_candidate_no_context | bertscore_f1 | 0.0344 | (0.0148, 0.0529) | 0.0003 | 0.0344 | (0.0055, 0.0605) | 0.0087 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1866 | (0.1549, 0.2204) | 0.0000 | 0.1866 | (0.1679, 0.2019) | 0.0000 |
| controlled_alt_vs_baseline_no_context | context_relevance | 0.2221 | (0.1730, 0.2716) | 0.0000 | 0.2221 | (0.1645, 0.2862) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 0.1591 | (0.0933, 0.2302) | 0.0000 | 0.1591 | (0.1124, 0.2278) | 0.0000 |
| controlled_alt_vs_baseline_no_context | naturalness | -0.0168 | (-0.0545, 0.0214) | 0.7933 | -0.0168 | (-0.0599, 0.0349) | 0.7387 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 0.2893 | (0.2232, 0.3576) | 0.0000 | 0.2893 | (0.2131, 0.3801) | 0.0000 |
| controlled_alt_vs_baseline_no_context | context_overlap | 0.0654 | (0.0510, 0.0815) | 0.0000 | 0.0654 | (0.0499, 0.0786) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 0.1855 | (0.1079, 0.2730) | 0.0000 | 0.1855 | (0.1253, 0.2735) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_style | 0.0533 | (-0.0132, 0.1340) | 0.0727 | 0.0533 | (0.0067, 0.1338) | 0.0227 |
| controlled_alt_vs_baseline_no_context | distinct1 | -0.0397 | (-0.0579, -0.0208) | 1.0000 | -0.0397 | (-0.0522, -0.0270) | 1.0000 |
| controlled_alt_vs_baseline_no_context | length_score | -0.0264 | (-0.2139, 0.1542) | 0.6140 | -0.0264 | (-0.2440, 0.2377) | 0.5877 |
| controlled_alt_vs_baseline_no_context | sentence_score | 0.0437 | (-0.0437, 0.1313) | 0.2027 | 0.0437 | (-0.0483, 0.1501) | 0.2090 |
| controlled_alt_vs_baseline_no_context | bertscore_f1 | 0.0294 | (0.0018, 0.0586) | 0.0193 | 0.0294 | (-0.0102, 0.0680) | 0.0697 |
| controlled_alt_vs_baseline_no_context | overall_quality | 0.1425 | (0.1120, 0.1738) | 0.0000 | 0.1425 | (0.1139, 0.1830) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2808 | (0.2408, 0.3235) | 0.0000 | 0.2808 | (0.2517, 0.3154) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1765 | (0.0702, 0.2923) | 0.0010 | 0.1765 | (0.0299, 0.3222) | 0.0103 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0349 | (-0.0725, 0.0054) | 0.9563 | -0.0349 | (-0.0708, 0.0001) | 0.9740 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3663 | (0.3117, 0.4220) | 0.0000 | 0.3663 | (0.3295, 0.4148) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0813 | (0.0659, 0.0974) | 0.0000 | 0.0813 | (0.0597, 0.0980) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2131 | (0.0937, 0.3427) | 0.0000 | 0.2131 | (0.0441, 0.3814) | 0.0113 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0302 | (-0.0394, 0.1048) | 0.2017 | 0.0302 | (-0.0595, 0.1424) | 0.2977 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0460 | (-0.0593, -0.0324) | 1.0000 | -0.0460 | (-0.0611, -0.0330) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | -0.1000 | (-0.2528, 0.0778) | 0.8707 | -0.1000 | (-0.2667, 0.0617) | 0.8823 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0437 | (-0.0292, 0.1167) | 0.1607 | 0.0437 | (-0.0121, 0.1167) | 0.1043 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0471 | (0.0220, 0.0717) | 0.0003 | 0.0471 | (0.0142, 0.0765) | 0.0013 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1712 | (0.1313, 0.2106) | 0.0000 | 0.1712 | (0.1236, 0.2116) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 12 | 2 | 10 | 0.7083 | 0.8571 |
| proposed_vs_candidate_no_context | persona_consistency | 8 | 2 | 14 | 0.6250 | 0.8000 |
| proposed_vs_candidate_no_context | naturalness | 7 | 7 | 10 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 11 | 1 | 12 | 0.7083 | 0.9167 |
| proposed_vs_candidate_no_context | context_overlap | 11 | 3 | 10 | 0.6667 | 0.7857 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 4 | 2 | 18 | 0.5417 | 0.6667 |
| proposed_vs_candidate_no_context | persona_style | 5 | 0 | 19 | 0.6042 | 1.0000 |
| proposed_vs_candidate_no_context | distinct1 | 9 | 5 | 10 | 0.5833 | 0.6429 |
| proposed_vs_candidate_no_context | length_score | 4 | 9 | 11 | 0.3958 | 0.3077 |
| proposed_vs_candidate_no_context | sentence_score | 4 | 3 | 17 | 0.5208 | 0.5714 |
| proposed_vs_candidate_no_context | bertscore_f1 | 11 | 5 | 8 | 0.6250 | 0.6875 |
| proposed_vs_candidate_no_context | overall_quality | 12 | 4 | 8 | 0.6667 | 0.7500 |
| proposed_vs_baseline_no_context | context_relevance | 16 | 8 | 0 | 0.6667 | 0.6667 |
| proposed_vs_baseline_no_context | persona_consistency | 5 | 11 | 8 | 0.3750 | 0.3125 |
| proposed_vs_baseline_no_context | naturalness | 4 | 20 | 0 | 0.1667 | 0.1667 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 11 | 4 | 9 | 0.6458 | 0.7333 |
| proposed_vs_baseline_no_context | context_overlap | 16 | 8 | 0 | 0.6667 | 0.6667 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 2 | 8 | 14 | 0.3750 | 0.2000 |
| proposed_vs_baseline_no_context | persona_style | 4 | 5 | 15 | 0.4792 | 0.4444 |
| proposed_vs_baseline_no_context | distinct1 | 4 | 18 | 2 | 0.2083 | 0.1818 |
| proposed_vs_baseline_no_context | length_score | 5 | 19 | 0 | 0.2083 | 0.2083 |
| proposed_vs_baseline_no_context | sentence_score | 3 | 10 | 11 | 0.3542 | 0.2308 |
| proposed_vs_baseline_no_context | bertscore_f1 | 17 | 7 | 0 | 0.7083 | 0.7083 |
| proposed_vs_baseline_no_context | overall_quality | 11 | 13 | 0 | 0.4583 | 0.4583 |
| controlled_vs_proposed_raw | context_relevance | 22 | 2 | 0 | 0.9167 | 0.9167 |
| controlled_vs_proposed_raw | persona_consistency | 20 | 2 | 2 | 0.8750 | 0.9091 |
| controlled_vs_proposed_raw | naturalness | 16 | 8 | 0 | 0.6667 | 0.6667 |
| controlled_vs_proposed_raw | context_keyword_coverage | 22 | 2 | 0 | 0.9167 | 0.9167 |
| controlled_vs_proposed_raw | context_overlap | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 20 | 2 | 2 | 0.8750 | 0.9091 |
| controlled_vs_proposed_raw | persona_style | 7 | 4 | 13 | 0.5625 | 0.6364 |
| controlled_vs_proposed_raw | distinct1 | 13 | 11 | 0 | 0.5417 | 0.5417 |
| controlled_vs_proposed_raw | length_score | 17 | 4 | 3 | 0.7708 | 0.8095 |
| controlled_vs_proposed_raw | sentence_score | 12 | 2 | 10 | 0.7083 | 0.8571 |
| controlled_vs_proposed_raw | bertscore_f1 | 13 | 11 | 0 | 0.5417 | 0.5417 |
| controlled_vs_proposed_raw | overall_quality | 21 | 3 | 0 | 0.8750 | 0.8750 |
| controlled_vs_candidate_no_context | context_relevance | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 19 | 4 | 1 | 0.8125 | 0.8261 |
| controlled_vs_candidate_no_context | naturalness | 18 | 6 | 0 | 0.7500 | 0.7500 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 19 | 3 | 2 | 0.8333 | 0.8636 |
| controlled_vs_candidate_no_context | persona_style | 8 | 3 | 13 | 0.6042 | 0.7273 |
| controlled_vs_candidate_no_context | distinct1 | 14 | 9 | 1 | 0.6042 | 0.6087 |
| controlled_vs_candidate_no_context | length_score | 17 | 7 | 0 | 0.7083 | 0.7083 |
| controlled_vs_candidate_no_context | sentence_score | 12 | 1 | 11 | 0.7292 | 0.9231 |
| controlled_vs_candidate_no_context | bertscore_f1 | 20 | 4 | 0 | 0.8333 | 0.8333 |
| controlled_vs_candidate_no_context | overall_quality | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_vs_baseline_no_context | context_relevance | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 18 | 4 | 2 | 0.7917 | 0.8182 |
| controlled_vs_baseline_no_context | naturalness | 8 | 16 | 0 | 0.3333 | 0.3333 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | context_overlap | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 18 | 4 | 2 | 0.7917 | 0.8182 |
| controlled_vs_baseline_no_context | persona_style | 4 | 3 | 17 | 0.5208 | 0.5714 |
| controlled_vs_baseline_no_context | distinct1 | 2 | 20 | 2 | 0.1250 | 0.0909 |
| controlled_vs_baseline_no_context | length_score | 8 | 16 | 0 | 0.3333 | 0.3333 |
| controlled_vs_baseline_no_context | sentence_score | 5 | 2 | 17 | 0.5625 | 0.7143 |
| controlled_vs_baseline_no_context | bertscore_f1 | 19 | 5 | 0 | 0.7917 | 0.7917 |
| controlled_vs_baseline_no_context | overall_quality | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_alt_vs_controlled_default | context_relevance | 7 | 16 | 1 | 0.3125 | 0.3043 |
| controlled_alt_vs_controlled_default | persona_consistency | 8 | 8 | 8 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | naturalness | 14 | 9 | 1 | 0.6042 | 0.6087 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 6 | 14 | 4 | 0.3333 | 0.3000 |
| controlled_alt_vs_controlled_default | context_overlap | 9 | 14 | 1 | 0.3958 | 0.3913 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 6 | 6 | 12 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | persona_style | 6 | 2 | 16 | 0.5833 | 0.7500 |
| controlled_alt_vs_controlled_default | distinct1 | 10 | 11 | 3 | 0.4792 | 0.4762 |
| controlled_alt_vs_controlled_default | length_score | 13 | 9 | 2 | 0.5833 | 0.5909 |
| controlled_alt_vs_controlled_default | sentence_score | 3 | 3 | 18 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | bertscore_f1 | 10 | 13 | 1 | 0.4375 | 0.4348 |
| controlled_alt_vs_controlled_default | overall_quality | 9 | 14 | 1 | 0.3958 | 0.3913 |
| controlled_alt_vs_proposed_raw | context_relevance | 22 | 2 | 0 | 0.9167 | 0.9167 |
| controlled_alt_vs_proposed_raw | persona_consistency | 19 | 2 | 3 | 0.8542 | 0.9048 |
| controlled_alt_vs_proposed_raw | naturalness | 19 | 5 | 0 | 0.7917 | 0.7917 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 21 | 2 | 1 | 0.8958 | 0.9130 |
| controlled_alt_vs_proposed_raw | context_overlap | 22 | 2 | 0 | 0.9167 | 0.9167 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 19 | 1 | 4 | 0.8750 | 0.9500 |
| controlled_alt_vs_proposed_raw | persona_style | 7 | 2 | 15 | 0.6042 | 0.7778 |
| controlled_alt_vs_proposed_raw | distinct1 | 13 | 11 | 0 | 0.5417 | 0.5417 |
| controlled_alt_vs_proposed_raw | length_score | 17 | 6 | 1 | 0.7292 | 0.7391 |
| controlled_alt_vs_proposed_raw | sentence_score | 13 | 3 | 8 | 0.7083 | 0.8125 |
| controlled_alt_vs_proposed_raw | bertscore_f1 | 13 | 11 | 0 | 0.5417 | 0.5417 |
| controlled_alt_vs_proposed_raw | overall_quality | 22 | 2 | 0 | 0.9167 | 0.9167 |
| controlled_alt_vs_candidate_no_context | context_relevance | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 22 | 1 | 1 | 0.9375 | 0.9565 |
| controlled_alt_vs_candidate_no_context | naturalness | 19 | 5 | 0 | 0.7917 | 0.7917 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_alt_vs_candidate_no_context | context_overlap | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 22 | 1 | 1 | 0.9375 | 0.9565 |
| controlled_alt_vs_candidate_no_context | persona_style | 9 | 0 | 15 | 0.6875 | 1.0000 |
| controlled_alt_vs_candidate_no_context | distinct1 | 15 | 7 | 2 | 0.6667 | 0.6818 |
| controlled_alt_vs_candidate_no_context | length_score | 17 | 7 | 0 | 0.7083 | 0.7083 |
| controlled_alt_vs_candidate_no_context | sentence_score | 14 | 3 | 7 | 0.7292 | 0.8235 |
| controlled_alt_vs_candidate_no_context | bertscore_f1 | 18 | 6 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_candidate_no_context | overall_quality | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context | context_relevance | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 18 | 2 | 4 | 0.8333 | 0.9000 |
| controlled_alt_vs_baseline_no_context | naturalness | 9 | 15 | 0 | 0.3750 | 0.3750 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_alt_vs_baseline_no_context | context_overlap | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 17 | 1 | 6 | 0.8333 | 0.9444 |
| controlled_alt_vs_baseline_no_context | persona_style | 5 | 2 | 17 | 0.5625 | 0.7143 |
| controlled_alt_vs_baseline_no_context | distinct1 | 5 | 19 | 0 | 0.2083 | 0.2083 |
| controlled_alt_vs_baseline_no_context | length_score | 10 | 14 | 0 | 0.4167 | 0.4167 |
| controlled_alt_vs_baseline_no_context | sentence_score | 6 | 3 | 15 | 0.5625 | 0.6667 |
| controlled_alt_vs_baseline_no_context | bertscore_f1 | 16 | 8 | 0 | 0.6667 | 0.6667 |
| controlled_alt_vs_baseline_no_context | overall_quality | 24 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 24 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 18 | 4 | 2 | 0.7917 | 0.8182 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 8 | 16 | 0 | 0.3333 | 0.3333 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 24 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 24 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 18 | 4 | 2 | 0.7917 | 0.8182 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 4 | 3 | 17 | 0.5208 | 0.5714 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | 2 | 20 | 2 | 0.1250 | 0.0909 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 8 | 16 | 0 | 0.3333 | 0.3333 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 5 | 2 | 17 | 0.5625 | 0.7143 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 19 | 5 | 0 | 0.7917 | 0.7917 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 23 | 1 | 0 | 0.9583 | 0.9583 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.2500 | 0.3333 | 0.6667 |
| proposed_contextual_controlled_alt | 0.0000 | 0.0000 | 0.2083 | 0.5417 | 0.4583 |
| proposed_contextual | 0.0000 | 0.0000 | 0.4583 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.5417 | 0.0000 | 0.0000 |
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
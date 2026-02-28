# Proposal Alignment Evaluation Report

- Run ID: `20260228T080406Z`
- Generated: `2026-02-28T08:04:43.706895+00:00`
- Scenarios: `artifacts\proposal\20260228T080406Z\scenarios.jsonl`
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
| proposed_contextual_controlled | 0.2877 (0.2716, 0.3040) | 0.2565 (0.2197, 0.2940) | 0.8932 (0.8835, 0.9017) | 0.3679 (0.3541, 0.3816) | 0.1129 |
| proposed_contextual | 0.0678 (0.0498, 0.0880) | 0.1368 (0.1138, 0.1602) | 0.8035 (0.7900, 0.8167) | 0.2189 (0.2047, 0.2353) | 0.0727 |
| candidate_no_context | 0.0259 (0.0191, 0.0338) | 0.1675 (0.1378, 0.2003) | 0.8133 (0.7986, 0.8288) | 0.2103 (0.1968, 0.2240) | 0.0398 |
| baseline_no_context | 0.0452 (0.0359, 0.0558) | 0.1702 (0.1519, 0.1912) | 0.8760 (0.8650, 0.8869) | 0.2305 (0.2214, 0.2399) | 0.0526 |
| baseline_no_context_phi3_latest | 0.0456 (0.0366, 0.0551) | 0.1838 (0.1620, 0.2047) | 0.8858 (0.8753, 0.8962) | 0.2371 (0.2284, 0.2456) | 0.0528 |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0419 | 1.6144 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0307 | -0.1834 |
| proposed_vs_candidate_no_context | naturalness | -0.0098 | -0.0120 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0544 | 2.7162 |
| proposed_vs_candidate_no_context | context_overlap | 0.0126 | 0.3177 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0436 | -0.4801 |
| proposed_vs_candidate_no_context | persona_style | 0.0207 | 0.0437 |
| proposed_vs_candidate_no_context | distinct1 | -0.0021 | -0.0023 |
| proposed_vs_candidate_no_context | length_score | -0.0411 | -0.1278 |
| proposed_vs_candidate_no_context | sentence_score | -0.0071 | -0.0097 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0329 | 0.8272 |
| proposed_vs_candidate_no_context | overall_quality | 0.0086 | 0.0410 |
| proposed_vs_baseline_no_context | context_relevance | 0.0226 | 0.4994 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0334 | -0.1962 |
| proposed_vs_baseline_no_context | naturalness | -0.0726 | -0.0828 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0279 | 0.6007 |
| proposed_vs_baseline_no_context | context_overlap | 0.0101 | 0.2391 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0191 | -0.2880 |
| proposed_vs_baseline_no_context | persona_style | -0.0906 | -0.1547 |
| proposed_vs_baseline_no_context | distinct1 | -0.0445 | -0.0454 |
| proposed_vs_baseline_no_context | length_score | -0.2173 | -0.4366 |
| proposed_vs_baseline_no_context | sentence_score | -0.1129 | -0.1338 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0201 | 0.3821 |
| proposed_vs_baseline_no_context | overall_quality | -0.0116 | -0.0504 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 0.0222 | 0.4868 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | -0.0470 | -0.2557 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | -0.0824 | -0.0930 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.0263 | 0.5476 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 0.0125 | 0.3153 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | -0.0358 | -0.4311 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | -0.0919 | -0.1566 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | -0.0489 | -0.0497 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | -0.2420 | -0.4632 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | -0.1442 | -0.1647 |
| proposed_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0199 | 0.3777 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | -0.0181 | -0.0765 |
| controlled_vs_proposed_raw | context_relevance | 0.2199 | 3.2449 |
| controlled_vs_proposed_raw | persona_consistency | 0.1197 | 0.8753 |
| controlled_vs_proposed_raw | naturalness | 0.0897 | 0.1116 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2921 | 3.9256 |
| controlled_vs_proposed_raw | context_overlap | 0.0515 | 0.9855 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1279 | 2.7095 |
| controlled_vs_proposed_raw | persona_style | 0.0871 | 0.1759 |
| controlled_vs_proposed_raw | distinct1 | 0.0080 | 0.0085 |
| controlled_vs_proposed_raw | length_score | 0.3045 | 1.0861 |
| controlled_vs_proposed_raw | sentence_score | 0.2560 | 0.3501 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0402 | 0.5532 |
| controlled_vs_proposed_raw | overall_quality | 0.1490 | 0.6807 |
| controlled_vs_candidate_no_context | context_relevance | 0.2618 | 10.0979 |
| controlled_vs_candidate_no_context | persona_consistency | 0.0890 | 0.5314 |
| controlled_vs_candidate_no_context | naturalness | 0.0799 | 0.0983 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3465 | 17.3046 |
| controlled_vs_candidate_no_context | context_overlap | 0.0642 | 1.6162 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.0843 | 0.9286 |
| controlled_vs_candidate_no_context | persona_style | 0.1078 | 0.2273 |
| controlled_vs_candidate_no_context | distinct1 | 0.0058 | 0.0062 |
| controlled_vs_candidate_no_context | length_score | 0.2634 | 0.8195 |
| controlled_vs_candidate_no_context | sentence_score | 0.2489 | 0.3371 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0731 | 1.8380 |
| controlled_vs_candidate_no_context | overall_quality | 0.1577 | 0.7497 |
| controlled_vs_baseline_no_context | context_relevance | 0.2425 | 5.3648 |
| controlled_vs_baseline_no_context | persona_consistency | 0.0863 | 0.5073 |
| controlled_vs_baseline_no_context | naturalness | 0.0171 | 0.0195 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3200 | 6.8844 |
| controlled_vs_baseline_no_context | context_overlap | 0.0616 | 1.4603 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1088 | 1.6411 |
| controlled_vs_baseline_no_context | persona_style | -0.0035 | -0.0060 |
| controlled_vs_baseline_no_context | distinct1 | -0.0366 | -0.0373 |
| controlled_vs_baseline_no_context | length_score | 0.0872 | 0.1753 |
| controlled_vs_baseline_no_context | sentence_score | 0.1431 | 0.1695 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0603 | 1.1466 |
| controlled_vs_baseline_no_context | overall_quality | 0.1374 | 0.5961 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.2421 | 5.3112 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.0727 | 0.3958 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | 0.0073 | 0.0083 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.3184 | 6.6226 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0641 | 1.6115 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.0921 | 1.1105 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | -0.0048 | -0.0082 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0409 | -0.0416 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | 0.0625 | 0.1197 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.1118 | 0.1277 |
| controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0601 | 1.1399 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1309 | 0.5521 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2425 | 5.3648 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.0863 | 0.5073 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 0.0171 | 0.0195 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3200 | 6.8844 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0616 | 1.4603 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1088 | 1.6411 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | -0.0035 | -0.0060 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0366 | -0.0373 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.0872 | 0.1753 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.1431 | 0.1695 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0603 | 1.1466 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1374 | 0.5961 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.2421 | 5.3112 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.0727 | 0.3958 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | 0.0073 | 0.0083 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.3184 | 6.6226 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0641 | 1.6115 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.0921 | 1.1105 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | -0.0048 | -0.0082 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0409 | -0.0416 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | 0.0625 | 0.1197 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.1118 | 0.1277 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0601 | 1.1399 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1309 | 0.5521 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) |
|---|---|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0419 | (0.0233, 0.0624) | 0.0000 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0307 | (-0.0615, -0.0019) | 0.9813 |
| proposed_vs_candidate_no_context | naturalness | -0.0098 | (-0.0237, 0.0044) | 0.9047 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0544 | (0.0289, 0.0821) | 0.0000 |
| proposed_vs_candidate_no_context | context_overlap | 0.0126 | (0.0059, 0.0199) | 0.0000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0436 | (-0.0782, -0.0096) | 0.9940 |
| proposed_vs_candidate_no_context | persona_style | 0.0207 | (-0.0048, 0.0527) | 0.0643 |
| proposed_vs_candidate_no_context | distinct1 | -0.0021 | (-0.0093, 0.0048) | 0.7360 |
| proposed_vs_candidate_no_context | length_score | -0.0411 | (-0.0961, 0.0122) | 0.9347 |
| proposed_vs_candidate_no_context | sentence_score | -0.0071 | (-0.0442, 0.0295) | 0.6587 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0329 | (0.0202, 0.0468) | 0.0000 |
| proposed_vs_candidate_no_context | overall_quality | 0.0086 | (-0.0074, 0.0258) | 0.1527 |
| proposed_vs_baseline_no_context | context_relevance | 0.0226 | (0.0012, 0.0455) | 0.0203 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0334 | (-0.0612, -0.0048) | 0.9890 |
| proposed_vs_baseline_no_context | naturalness | -0.0726 | (-0.0890, -0.0553) | 1.0000 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0279 | (-0.0002, 0.0573) | 0.0263 |
| proposed_vs_baseline_no_context | context_overlap | 0.0101 | (0.0026, 0.0181) | 0.0040 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0191 | (-0.0499, 0.0115) | 0.8863 |
| proposed_vs_baseline_no_context | persona_style | -0.0906 | (-0.1378, -0.0472) | 1.0000 |
| proposed_vs_baseline_no_context | distinct1 | -0.0445 | (-0.0534, -0.0356) | 1.0000 |
| proposed_vs_baseline_no_context | length_score | -0.2173 | (-0.2830, -0.1521) | 1.0000 |
| proposed_vs_baseline_no_context | sentence_score | -0.1129 | (-0.1531, -0.0705) | 1.0000 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0201 | (0.0062, 0.0345) | 0.0037 |
| proposed_vs_baseline_no_context | overall_quality | -0.0116 | (-0.0298, 0.0066) | 0.8947 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 0.0222 | (0.0019, 0.0440) | 0.0160 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | -0.0470 | (-0.0782, -0.0158) | 0.9977 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | -0.0824 | (-0.0991, -0.0649) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.0263 | (-0.0014, 0.0564) | 0.0323 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 0.0125 | (0.0055, 0.0203) | 0.0003 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | -0.0358 | (-0.0685, -0.0020) | 0.9800 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | -0.0919 | (-0.1350, -0.0492) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | -0.0489 | (-0.0568, -0.0406) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | -0.2420 | (-0.3066, -0.1723) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | -0.1442 | (-0.1817, -0.1040) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0199 | (0.0070, 0.0338) | 0.0007 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | -0.0181 | (-0.0351, 0.0004) | 0.9727 |
| controlled_vs_proposed_raw | context_relevance | 0.2220 | (0.1958, 0.2474) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.1215 | (0.0873, 0.1579) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0900 | (0.0720, 0.1073) | 0.0000 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2949 | (0.2601, 0.3287) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0522 | (0.0424, 0.0618) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1300 | (0.0885, 0.1739) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.0872 | (0.0444, 0.1327) | 0.0000 |
| controlled_vs_proposed_raw | distinct1 | 0.0081 | (0.0001, 0.0164) | 0.0233 |
| controlled_vs_proposed_raw | length_score | 0.3048 | (0.2318, 0.3727) | 0.0000 |
| controlled_vs_proposed_raw | sentence_score | 0.2577 | (0.2259, 0.2895) | 0.0000 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0403 | (0.0246, 0.0546) | 0.0000 |
| controlled_vs_proposed_raw | overall_quality | 0.1505 | (0.1311, 0.1687) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2615 | (0.2446, 0.2789) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.0878 | (0.0463, 0.1303) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.0806 | (0.0614, 0.1003) | 0.0000 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3461 | (0.3242, 0.3688) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0642 | (0.0575, 0.0711) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.0826 | (0.0324, 0.1336) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.1083 | (0.0631, 0.1589) | 0.0000 |
| controlled_vs_candidate_no_context | distinct1 | 0.0057 | (-0.0024, 0.0138) | 0.0827 |
| controlled_vs_candidate_no_context | length_score | 0.2664 | (0.1864, 0.3391) | 0.0000 |
| controlled_vs_candidate_no_context | sentence_score | 0.2505 | (0.2159, 0.2850) | 0.0000 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0736 | (0.0583, 0.0882) | 0.0000 |
| controlled_vs_candidate_no_context | overall_quality | 0.1573 | (0.1410, 0.1736) | 0.0000 |
| controlled_vs_baseline_no_context | context_relevance | 0.2419 | (0.2244, 0.2603) | 0.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 0.0850 | (0.0490, 0.1220) | 0.0000 |
| controlled_vs_baseline_no_context | naturalness | 0.0166 | (0.0029, 0.0297) | 0.0073 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3192 | (0.2960, 0.3430) | 0.0000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0617 | (0.0545, 0.0690) | 0.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1076 | (0.0631, 0.1547) | 0.0000 |
| controlled_vs_baseline_no_context | persona_style | -0.0051 | (-0.0230, 0.0129) | 0.7173 |
| controlled_vs_baseline_no_context | distinct1 | -0.0362 | (-0.0434, -0.0292) | 1.0000 |
| controlled_vs_baseline_no_context | length_score | 0.0839 | (0.0209, 0.1427) | 0.0030 |
| controlled_vs_baseline_no_context | sentence_score | 0.1427 | (0.1050, 0.1777) | 0.0000 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0617 | (0.0464, 0.0764) | 0.0000 |
| controlled_vs_baseline_no_context | overall_quality | 0.1368 | (0.1219, 0.1528) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.2415 | (0.2238, 0.2589) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.0712 | (0.0321, 0.1111) | 0.0007 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | 0.0070 | (-0.0059, 0.0202) | 0.1383 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.3175 | (0.2933, 0.3416) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0640 | (0.0567, 0.0714) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.0906 | (0.0405, 0.1397) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | -0.0064 | (-0.0262, 0.0137) | 0.7300 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0406 | (-0.0473, -0.0337) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | 0.0609 | (0.0042, 0.1176) | 0.0203 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.1109 | (0.0764, 0.1459) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0610 | (0.0454, 0.0761) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1302 | (0.1153, 0.1457) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2419 | (0.2237, 0.2607) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.0850 | (0.0484, 0.1227) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 0.0166 | (0.0033, 0.0307) | 0.0063 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3192 | (0.2966, 0.3426) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0617 | (0.0547, 0.0690) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1076 | (0.0626, 0.1540) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | -0.0051 | (-0.0237, 0.0134) | 0.7140 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0362 | (-0.0428, -0.0293) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.0839 | (0.0206, 0.1439) | 0.0040 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.1427 | (0.1050, 0.1800) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0617 | (0.0463, 0.0763) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1368 | (0.1217, 0.1524) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.2415 | (0.2232, 0.2597) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.0712 | (0.0332, 0.1102) | 0.0007 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | 0.0070 | (-0.0061, 0.0202) | 0.1497 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.3175 | (0.2941, 0.3415) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0640 | (0.0567, 0.0717) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.0906 | (0.0443, 0.1419) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | -0.0064 | (-0.0259, 0.0144) | 0.7120 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0406 | (-0.0473, -0.0339) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | 0.0609 | (0.0024, 0.1206) | 0.0203 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.1109 | (0.0755, 0.1482) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0610 | (0.0461, 0.0768) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1302 | (0.1149, 0.1458) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 41 | 17 | 54 | 0.6071 | 0.7069 |
| proposed_vs_candidate_no_context | persona_consistency | 16 | 19 | 77 | 0.4866 | 0.4571 |
| proposed_vs_candidate_no_context | naturalness | 24 | 34 | 54 | 0.4554 | 0.4138 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 33 | 9 | 70 | 0.6071 | 0.7857 |
| proposed_vs_candidate_no_context | context_overlap | 41 | 17 | 54 | 0.6071 | 0.7069 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 10 | 17 | 85 | 0.4688 | 0.3704 |
| proposed_vs_candidate_no_context | persona_style | 10 | 3 | 99 | 0.5312 | 0.7692 |
| proposed_vs_candidate_no_context | distinct1 | 23 | 29 | 60 | 0.4732 | 0.4423 |
| proposed_vs_candidate_no_context | length_score | 23 | 34 | 55 | 0.4509 | 0.4035 |
| proposed_vs_candidate_no_context | sentence_score | 15 | 18 | 79 | 0.4866 | 0.4545 |
| proposed_vs_candidate_no_context | bertscore_f1 | 48 | 19 | 45 | 0.6295 | 0.7164 |
| proposed_vs_candidate_no_context | overall_quality | 35 | 32 | 45 | 0.5134 | 0.5224 |
| proposed_vs_baseline_no_context | context_relevance | 58 | 54 | 0 | 0.5179 | 0.5179 |
| proposed_vs_baseline_no_context | persona_consistency | 16 | 53 | 43 | 0.3348 | 0.2319 |
| proposed_vs_baseline_no_context | naturalness | 24 | 88 | 0 | 0.2143 | 0.2143 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 30 | 27 | 55 | 0.5134 | 0.5263 |
| proposed_vs_baseline_no_context | context_overlap | 68 | 44 | 0 | 0.6071 | 0.6071 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 15 | 32 | 65 | 0.4241 | 0.3191 |
| proposed_vs_baseline_no_context | persona_style | 8 | 29 | 75 | 0.4062 | 0.2162 |
| proposed_vs_baseline_no_context | distinct1 | 16 | 85 | 11 | 0.1920 | 0.1584 |
| proposed_vs_baseline_no_context | length_score | 27 | 82 | 3 | 0.2545 | 0.2477 |
| proposed_vs_baseline_no_context | sentence_score | 11 | 47 | 54 | 0.3393 | 0.1897 |
| proposed_vs_baseline_no_context | bertscore_f1 | 67 | 45 | 0 | 0.5982 | 0.5982 |
| proposed_vs_baseline_no_context | overall_quality | 39 | 73 | 0 | 0.3482 | 0.3482 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 56 | 55 | 1 | 0.5045 | 0.5045 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | 16 | 59 | 37 | 0.3080 | 0.2133 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | 25 | 87 | 0 | 0.2232 | 0.2232 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 30 | 30 | 52 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 66 | 43 | 3 | 0.6027 | 0.6055 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 14 | 37 | 61 | 0.3973 | 0.2745 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | 8 | 28 | 76 | 0.4107 | 0.2222 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | 14 | 84 | 14 | 0.1875 | 0.1429 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | 25 | 86 | 1 | 0.2277 | 0.2252 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | 7 | 53 | 52 | 0.2946 | 0.1167 |
| proposed_vs_baseline_no_context_phi3_latest | bertscore_f1 | 62 | 50 | 0 | 0.5536 | 0.5536 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | 35 | 77 | 0 | 0.3125 | 0.3125 |
| controlled_vs_proposed_raw | context_relevance | 101 | 9 | 0 | 0.9182 | 0.9182 |
| controlled_vs_proposed_raw | persona_consistency | 68 | 6 | 36 | 0.7818 | 0.9189 |
| controlled_vs_proposed_raw | naturalness | 90 | 20 | 0 | 0.8182 | 0.8182 |
| controlled_vs_proposed_raw | context_keyword_coverage | 101 | 5 | 4 | 0.9364 | 0.9528 |
| controlled_vs_proposed_raw | context_overlap | 98 | 12 | 0 | 0.8909 | 0.8909 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 52 | 6 | 52 | 0.7091 | 0.8966 |
| controlled_vs_proposed_raw | persona_style | 24 | 8 | 78 | 0.5727 | 0.7500 |
| controlled_vs_proposed_raw | distinct1 | 68 | 41 | 1 | 0.6227 | 0.6239 |
| controlled_vs_proposed_raw | length_score | 81 | 27 | 2 | 0.7455 | 0.7500 |
| controlled_vs_proposed_raw | sentence_score | 83 | 2 | 25 | 0.8682 | 0.9765 |
| controlled_vs_proposed_raw | bertscore_f1 | 76 | 34 | 0 | 0.6909 | 0.6909 |
| controlled_vs_proposed_raw | overall_quality | 100 | 10 | 0 | 0.9091 | 0.9091 |
| controlled_vs_candidate_no_context | context_relevance | 109 | 1 | 0 | 0.9909 | 0.9909 |
| controlled_vs_candidate_no_context | persona_consistency | 68 | 15 | 27 | 0.7409 | 0.8193 |
| controlled_vs_candidate_no_context | naturalness | 81 | 29 | 0 | 0.7364 | 0.7364 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 109 | 0 | 1 | 0.9955 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 108 | 2 | 0 | 0.9818 | 0.9818 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 48 | 15 | 47 | 0.6500 | 0.7619 |
| controlled_vs_candidate_no_context | persona_style | 29 | 5 | 76 | 0.6091 | 0.8529 |
| controlled_vs_candidate_no_context | distinct1 | 66 | 42 | 2 | 0.6091 | 0.6111 |
| controlled_vs_candidate_no_context | length_score | 77 | 33 | 0 | 0.7000 | 0.7000 |
| controlled_vs_candidate_no_context | sentence_score | 79 | 2 | 29 | 0.8500 | 0.9753 |
| controlled_vs_candidate_no_context | bertscore_f1 | 89 | 21 | 0 | 0.8091 | 0.8091 |
| controlled_vs_candidate_no_context | overall_quality | 104 | 6 | 0 | 0.9455 | 0.9455 |
| controlled_vs_baseline_no_context | context_relevance | 108 | 2 | 0 | 0.9818 | 0.9818 |
| controlled_vs_baseline_no_context | persona_consistency | 52 | 22 | 36 | 0.6364 | 0.7027 |
| controlled_vs_baseline_no_context | naturalness | 62 | 48 | 0 | 0.5636 | 0.5636 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 108 | 2 | 0 | 0.9818 | 0.9818 |
| controlled_vs_baseline_no_context | context_overlap | 108 | 2 | 0 | 0.9818 | 0.9818 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 47 | 18 | 45 | 0.6318 | 0.7231 |
| controlled_vs_baseline_no_context | persona_style | 9 | 13 | 88 | 0.4818 | 0.4091 |
| controlled_vs_baseline_no_context | distinct1 | 17 | 89 | 4 | 0.1727 | 0.1604 |
| controlled_vs_baseline_no_context | length_score | 64 | 43 | 3 | 0.5955 | 0.5981 |
| controlled_vs_baseline_no_context | sentence_score | 47 | 3 | 60 | 0.7000 | 0.9400 |
| controlled_vs_baseline_no_context | bertscore_f1 | 91 | 19 | 0 | 0.8273 | 0.8273 |
| controlled_vs_baseline_no_context | overall_quality | 106 | 4 | 0 | 0.9636 | 0.9636 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 109 | 1 | 0 | 0.9909 | 0.9909 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 47 | 25 | 38 | 0.6000 | 0.6528 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | 55 | 54 | 1 | 0.5045 | 0.5046 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 108 | 1 | 1 | 0.9864 | 0.9908 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 106 | 3 | 1 | 0.9682 | 0.9725 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 43 | 19 | 48 | 0.6091 | 0.6935 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | 9 | 12 | 89 | 0.4864 | 0.4286 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | 13 | 95 | 2 | 0.1273 | 0.1204 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | 57 | 45 | 8 | 0.5545 | 0.5588 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 38 | 4 | 68 | 0.6545 | 0.9048 |
| controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 87 | 23 | 0 | 0.7909 | 0.7909 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 105 | 5 | 0 | 0.9545 | 0.9545 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 108 | 2 | 0 | 0.9818 | 0.9818 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 52 | 22 | 36 | 0.6364 | 0.7027 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 62 | 48 | 0 | 0.5636 | 0.5636 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 108 | 2 | 0 | 0.9818 | 0.9818 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 108 | 2 | 0 | 0.9818 | 0.9818 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 47 | 18 | 45 | 0.6318 | 0.7231 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 9 | 13 | 88 | 0.4818 | 0.4091 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | 17 | 89 | 4 | 0.1727 | 0.1604 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 64 | 43 | 3 | 0.5955 | 0.5981 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 47 | 3 | 60 | 0.7000 | 0.9400 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 91 | 19 | 0 | 0.8273 | 0.8273 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 106 | 4 | 0 | 0.9636 | 0.9636 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 109 | 1 | 0 | 0.9909 | 0.9909 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 47 | 25 | 38 | 0.6000 | 0.6528 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | 55 | 54 | 1 | 0.5045 | 0.5046 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 108 | 1 | 1 | 0.9864 | 0.9908 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 106 | 3 | 1 | 0.9682 | 0.9725 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 43 | 19 | 48 | 0.6091 | 0.6935 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | 9 | 12 | 89 | 0.4864 | 0.4286 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | 13 | 95 | 2 | 0.1273 | 0.1204 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | 57 | 45 | 8 | 0.5545 | 0.5588 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | 38 | 4 | 68 | 0.6545 | 0.9048 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 87 | 23 | 0 | 0.7909 | 0.7909 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 105 | 5 | 0 | 0.9545 | 0.9545 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

- BERTScore status: enabled.

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
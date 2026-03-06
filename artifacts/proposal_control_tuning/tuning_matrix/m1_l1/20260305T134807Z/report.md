# Proposal Alignment Evaluation Report

- Run ID: `20260305T134807Z`
- Generated: `2026-03-05T13:51:28.510992+00:00`
- Scenarios: `artifacts\proposal_control_tuning\tuning_matrix\m1_l1\20260305T134807Z\scenarios.jsonl`
- Scenario count: `20`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off
- `baseline_no_context`: model `phi3:mini`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2782 (0.2404, 0.3237) | 0.2832 (0.2313, 0.3365) | 0.8822 (0.8383, 0.9209) | 0.3683 (0.3424, 0.3946) | 0.0930 |
| proposed_contextual | 0.0646 (0.0246, 0.1137) | 0.1114 (0.0830, 0.1411) | 0.7932 (0.7706, 0.8185) | 0.2058 (0.1798, 0.2370) | 0.0623 |
| candidate_no_context | 0.0266 (0.0137, 0.0428) | 0.1940 (0.1279, 0.2714) | 0.8425 (0.8045, 0.8837) | 0.2229 (0.1971, 0.2525) | 0.0320 |
| baseline_no_context | 0.0305 (0.0161, 0.0485) | 0.1517 (0.1090, 0.1987) | 0.8587 (0.8305, 0.8867) | 0.2147 (0.1987, 0.2327) | 0.0470 |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0379 | 1.4223 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0826 | -0.4256 |
| proposed_vs_candidate_no_context | naturalness | -0.0493 | -0.0585 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0485 | 2.1333 |
| proposed_vs_candidate_no_context | context_overlap | 0.0132 | 0.3690 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0957 | -0.9199 |
| proposed_vs_candidate_no_context | persona_style | -0.0299 | -0.0540 |
| proposed_vs_candidate_no_context | distinct1 | 0.0003 | 0.0003 |
| proposed_vs_candidate_no_context | length_score | -0.2033 | -0.4784 |
| proposed_vs_candidate_no_context | sentence_score | -0.0875 | -0.1061 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0303 | 0.9448 |
| proposed_vs_candidate_no_context | overall_quality | -0.0171 | -0.0766 |
| proposed_vs_baseline_no_context | context_relevance | 0.0340 | 1.1157 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0403 | -0.2657 |
| proposed_vs_baseline_no_context | naturalness | -0.0655 | -0.0763 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0443 | 1.6479 |
| proposed_vs_baseline_no_context | context_overlap | 0.0101 | 0.2582 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0355 | -0.8098 |
| proposed_vs_baseline_no_context | persona_style | -0.0597 | -0.1023 |
| proposed_vs_baseline_no_context | distinct1 | -0.0292 | -0.0302 |
| proposed_vs_baseline_no_context | length_score | -0.2167 | -0.4943 |
| proposed_vs_baseline_no_context | sentence_score | -0.1050 | -0.1246 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0153 | 0.3264 |
| proposed_vs_baseline_no_context | overall_quality | -0.0089 | -0.0412 |
| controlled_vs_proposed_raw | context_relevance | 0.2137 | 3.3105 |
| controlled_vs_proposed_raw | persona_consistency | 0.1718 | 1.5419 |
| controlled_vs_proposed_raw | naturalness | 0.0890 | 0.1122 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2808 | 3.9436 |
| controlled_vs_proposed_raw | context_overlap | 0.0571 | 1.1641 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1971 | 23.6571 |
| controlled_vs_proposed_raw | persona_style | 0.0702 | 0.1341 |
| controlled_vs_proposed_raw | distinct1 | -0.0186 | -0.0198 |
| controlled_vs_proposed_raw | length_score | 0.3817 | 1.7218 |
| controlled_vs_proposed_raw | sentence_score | 0.2450 | 0.3322 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0307 | 0.4922 |
| controlled_vs_proposed_raw | overall_quality | 0.1625 | 0.7896 |
| controlled_vs_candidate_no_context | context_relevance | 0.2516 | 9.4415 |
| controlled_vs_candidate_no_context | persona_consistency | 0.0892 | 0.4600 |
| controlled_vs_candidate_no_context | naturalness | 0.0397 | 0.0471 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3293 | 14.4900 |
| controlled_vs_candidate_no_context | context_overlap | 0.0703 | 1.9627 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1014 | 0.9748 |
| controlled_vs_candidate_no_context | persona_style | 0.0403 | 0.0729 |
| controlled_vs_candidate_no_context | distinct1 | -0.0183 | -0.0195 |
| controlled_vs_candidate_no_context | length_score | 0.1783 | 0.4196 |
| controlled_vs_candidate_no_context | sentence_score | 0.1575 | 0.1909 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0609 | 1.9020 |
| controlled_vs_candidate_no_context | overall_quality | 0.1454 | 0.6526 |
| controlled_vs_baseline_no_context | context_relevance | 0.2477 | 8.1197 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1314 | 0.8664 |
| controlled_vs_baseline_no_context | naturalness | 0.0235 | 0.0274 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3252 | 12.0901 |
| controlled_vs_baseline_no_context | context_overlap | 0.0671 | 1.7229 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1617 | 3.6902 |
| controlled_vs_baseline_no_context | persona_style | 0.0106 | 0.0181 |
| controlled_vs_baseline_no_context | distinct1 | -0.0478 | -0.0494 |
| controlled_vs_baseline_no_context | length_score | 0.1650 | 0.3764 |
| controlled_vs_baseline_no_context | sentence_score | 0.1400 | 0.1662 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0460 | 0.9793 |
| controlled_vs_baseline_no_context | overall_quality | 0.1537 | 0.7158 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2477 | 8.1197 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1314 | 0.8664 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 0.0235 | 0.0274 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3252 | 12.0901 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0671 | 1.7229 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1617 | 3.6902 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0106 | 0.0181 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0478 | -0.0494 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.1650 | 0.3764 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.1400 | 0.1662 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0460 | 0.9793 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1537 | 0.7158 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0379 | (0.0041, 0.0824) | 0.0107 | 0.0379 | (-0.0046, 0.0933) | 0.0493 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0826 | (-0.1626, -0.0134) | 0.9947 | -0.0826 | (-0.1795, -0.0196) | 0.9970 |
| proposed_vs_candidate_no_context | naturalness | -0.0493 | (-0.0942, -0.0093) | 0.9927 | -0.0493 | (-0.0745, -0.0185) | 0.9993 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0485 | (0.0045, 0.1030) | 0.0140 | 0.0485 | (-0.0051, 0.1199) | 0.0600 |
| proposed_vs_candidate_no_context | context_overlap | 0.0132 | (-0.0022, 0.0292) | 0.0453 | 0.0132 | (-0.0086, 0.0335) | 0.1150 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0957 | (-0.1938, -0.0100) | 0.9920 | -0.0957 | (-0.2126, -0.0318) | 1.0000 |
| proposed_vs_candidate_no_context | persona_style | -0.0299 | (-0.1151, 0.0390) | 0.7617 | -0.0299 | (-0.1200, 0.0545) | 0.7653 |
| proposed_vs_candidate_no_context | distinct1 | 0.0003 | (-0.0218, 0.0239) | 0.4833 | 0.0003 | (-0.0276, 0.0307) | 0.4897 |
| proposed_vs_candidate_no_context | length_score | -0.2033 | (-0.3617, -0.0600) | 0.9973 | -0.2033 | (-0.2923, -0.1125) | 1.0000 |
| proposed_vs_candidate_no_context | sentence_score | -0.0875 | (-0.1750, 0.0000) | 0.9877 | -0.0875 | (-0.1556, -0.0159) | 0.9920 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0303 | (0.0087, 0.0527) | 0.0020 | 0.0303 | (0.0048, 0.0606) | 0.0087 |
| proposed_vs_candidate_no_context | overall_quality | -0.0171 | (-0.0521, 0.0176) | 0.8333 | -0.0171 | (-0.0622, 0.0193) | 0.8230 |
| proposed_vs_baseline_no_context | context_relevance | 0.0340 | (-0.0054, 0.0862) | 0.0517 | 0.0340 | (-0.0111, 0.1009) | 0.1083 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0403 | (-0.0886, 0.0012) | 0.9713 | -0.0403 | (-0.0935, 0.0106) | 0.9287 |
| proposed_vs_baseline_no_context | naturalness | -0.0655 | (-0.1002, -0.0318) | 1.0000 | -0.0655 | (-0.0941, -0.0288) | 0.9997 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0443 | (-0.0091, 0.1121) | 0.0647 | 0.0443 | (-0.0170, 0.1383) | 0.1553 |
| proposed_vs_baseline_no_context | context_overlap | 0.0101 | (-0.0015, 0.0234) | 0.0493 | 0.0101 | (-0.0039, 0.0254) | 0.0850 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0355 | (-0.0876, 0.0083) | 0.9480 | -0.0355 | (-0.0756, 0.0116) | 0.9283 |
| proposed_vs_baseline_no_context | persona_style | -0.0597 | (-0.1688, 0.0226) | 0.8790 | -0.0597 | (-0.1969, 0.0170) | 0.7583 |
| proposed_vs_baseline_no_context | distinct1 | -0.0292 | (-0.0465, -0.0107) | 0.9990 | -0.0292 | (-0.0410, -0.0162) | 1.0000 |
| proposed_vs_baseline_no_context | length_score | -0.2167 | (-0.3383, -0.0983) | 1.0000 | -0.2167 | (-0.3178, -0.0912) | 0.9997 |
| proposed_vs_baseline_no_context | sentence_score | -0.1050 | (-0.2100, 0.0000) | 0.9790 | -0.1050 | (-0.2240, 0.0778) | 0.9240 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0153 | (-0.0185, 0.0494) | 0.1890 | 0.0153 | (-0.0319, 0.0560) | 0.2300 |
| proposed_vs_baseline_no_context | overall_quality | -0.0089 | (-0.0372, 0.0223) | 0.7343 | -0.0089 | (-0.0416, 0.0345) | 0.6970 |
| controlled_vs_proposed_raw | context_relevance | 0.2137 | (0.1648, 0.2641) | 0.0000 | 0.2137 | (0.1576, 0.2812) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.1718 | (0.1194, 0.2260) | 0.0000 | 0.1718 | (0.1163, 0.2525) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0890 | (0.0389, 0.1352) | 0.0003 | 0.0890 | (0.0453, 0.1221) | 0.0000 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2808 | (0.2104, 0.3523) | 0.0000 | 0.2808 | (0.2027, 0.3687) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0571 | (0.0407, 0.0746) | 0.0000 | 0.0571 | (0.0380, 0.0843) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1971 | (0.1352, 0.2626) | 0.0000 | 0.1971 | (0.1349, 0.2944) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.0702 | (-0.0083, 0.1687) | 0.0493 | 0.0702 | (-0.0220, 0.2146) | 0.1347 |
| controlled_vs_proposed_raw | distinct1 | -0.0186 | (-0.0535, 0.0088) | 0.8987 | -0.0186 | (-0.0495, 0.0033) | 0.9483 |
| controlled_vs_proposed_raw | length_score | 0.3817 | (0.1900, 0.5600) | 0.0000 | 0.3817 | (0.2370, 0.5000) | 0.0000 |
| controlled_vs_proposed_raw | sentence_score | 0.2450 | (0.1750, 0.3150) | 0.0000 | 0.2450 | (0.1658, 0.3033) | 0.0000 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0307 | (0.0062, 0.0559) | 0.0050 | 0.0307 | (-0.0024, 0.0648) | 0.0353 |
| controlled_vs_proposed_raw | overall_quality | 0.1625 | (0.1285, 0.1948) | 0.0000 | 0.1625 | (0.1275, 0.2115) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2516 | (0.2160, 0.2907) | 0.0000 | 0.2516 | (0.2198, 0.2978) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.0892 | (0.0185, 0.1617) | 0.0073 | 0.0892 | (0.0393, 0.1449) | 0.0013 |
| controlled_vs_candidate_no_context | naturalness | 0.0397 | (-0.0208, 0.0963) | 0.0960 | 0.0397 | (-0.0157, 0.0808) | 0.0810 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3293 | (0.2806, 0.3822) | 0.0000 | 0.3293 | (0.2862, 0.3914) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0703 | (0.0596, 0.0814) | 0.0000 | 0.0703 | (0.0602, 0.0867) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1014 | (0.0150, 0.1843) | 0.0100 | 0.1014 | (0.0467, 0.1692) | 0.0003 |
| controlled_vs_candidate_no_context | persona_style | 0.0403 | (-0.0235, 0.1266) | 0.1420 | 0.0403 | (-0.0276, 0.1132) | 0.1317 |
| controlled_vs_candidate_no_context | distinct1 | -0.0183 | (-0.0577, 0.0125) | 0.8633 | -0.0183 | (-0.0626, 0.0141) | 0.8337 |
| controlled_vs_candidate_no_context | length_score | 0.1783 | (-0.0384, 0.3834) | 0.0477 | 0.1783 | (0.0250, 0.2921) | 0.0143 |
| controlled_vs_candidate_no_context | sentence_score | 0.1575 | (0.0875, 0.2275) | 0.0000 | 0.1575 | (0.0920, 0.2139) | 0.0000 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0609 | (0.0405, 0.0825) | 0.0000 | 0.0609 | (0.0419, 0.0837) | 0.0000 |
| controlled_vs_candidate_no_context | overall_quality | 0.1454 | (0.1099, 0.1793) | 0.0000 | 0.1454 | (0.1177, 0.1831) | 0.0000 |
| controlled_vs_baseline_no_context | context_relevance | 0.2477 | (0.2140, 0.2861) | 0.0000 | 0.2477 | (0.2130, 0.2886) | 0.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1314 | (0.0575, 0.1986) | 0.0007 | 0.1314 | (0.0641, 0.2213) | 0.0000 |
| controlled_vs_baseline_no_context | naturalness | 0.0235 | (-0.0315, 0.0745) | 0.1937 | 0.0235 | (-0.0104, 0.0563) | 0.0883 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3252 | (0.2758, 0.3727) | 0.0000 | 0.3252 | (0.2800, 0.3778) | 0.0000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0671 | (0.0540, 0.0809) | 0.0000 | 0.0671 | (0.0533, 0.0877) | 0.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1617 | (0.0700, 0.2421) | 0.0000 | 0.1617 | (0.0742, 0.2730) | 0.0000 |
| controlled_vs_baseline_no_context | persona_style | 0.0106 | (-0.0479, 0.0643) | 0.3487 | 0.0106 | (-0.0145, 0.0549) | 0.3423 |
| controlled_vs_baseline_no_context | distinct1 | -0.0478 | (-0.0777, -0.0213) | 1.0000 | -0.0478 | (-0.0792, -0.0261) | 1.0000 |
| controlled_vs_baseline_no_context | length_score | 0.1650 | (-0.0650, 0.3867) | 0.0783 | 0.1650 | (-0.0000, 0.3177) | 0.0257 |
| controlled_vs_baseline_no_context | sentence_score | 0.1400 | (0.0700, 0.2100) | 0.0000 | 0.1400 | (0.0457, 0.2625) | 0.0000 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0460 | (0.0150, 0.0786) | 0.0007 | 0.0460 | (0.0027, 0.0818) | 0.0180 |
| controlled_vs_baseline_no_context | overall_quality | 0.1537 | (0.1236, 0.1834) | 0.0000 | 0.1537 | (0.1202, 0.2002) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2477 | (0.2119, 0.2844) | 0.0000 | 0.2477 | (0.2122, 0.2897) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1314 | (0.0603, 0.1963) | 0.0000 | 0.1314 | (0.0677, 0.2248) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 0.0235 | (-0.0285, 0.0753) | 0.1897 | 0.0235 | (-0.0104, 0.0557) | 0.0807 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3252 | (0.2791, 0.3744) | 0.0000 | 0.3252 | (0.2781, 0.3797) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0671 | (0.0540, 0.0805) | 0.0000 | 0.0671 | (0.0534, 0.0886) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1617 | (0.0757, 0.2414) | 0.0007 | 0.1617 | (0.0792, 0.2750) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0106 | (-0.0441, 0.0667) | 0.3540 | 0.0106 | (-0.0141, 0.0533) | 0.3447 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0478 | (-0.0779, -0.0212) | 1.0000 | -0.0478 | (-0.0782, -0.0266) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.1650 | (-0.0617, 0.3883) | 0.0807 | 0.1650 | (-0.0042, 0.3187) | 0.0290 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.1400 | (0.0700, 0.2100) | 0.0000 | 0.1400 | (0.0457, 0.2625) | 0.0003 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0460 | (0.0131, 0.0785) | 0.0030 | 0.0460 | (0.0040, 0.0837) | 0.0163 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1537 | (0.1233, 0.1845) | 0.0000 | 0.1537 | (0.1197, 0.1999) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 8 | 6 | 6 | 0.5500 | 0.5714 |
| proposed_vs_candidate_no_context | persona_consistency | 3 | 7 | 10 | 0.4000 | 0.3000 |
| proposed_vs_candidate_no_context | naturalness | 4 | 10 | 6 | 0.3500 | 0.2857 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 5 | 1 | 14 | 0.6000 | 0.8333 |
| proposed_vs_candidate_no_context | context_overlap | 8 | 6 | 6 | 0.5500 | 0.5714 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 1 | 5 | 14 | 0.4000 | 0.1667 |
| proposed_vs_candidate_no_context | persona_style | 2 | 3 | 15 | 0.4750 | 0.4000 |
| proposed_vs_candidate_no_context | distinct1 | 6 | 8 | 6 | 0.4500 | 0.4286 |
| proposed_vs_candidate_no_context | length_score | 4 | 10 | 6 | 0.3500 | 0.2857 |
| proposed_vs_candidate_no_context | sentence_score | 1 | 6 | 13 | 0.3750 | 0.1429 |
| proposed_vs_candidate_no_context | bertscore_f1 | 10 | 9 | 1 | 0.5250 | 0.5263 |
| proposed_vs_candidate_no_context | overall_quality | 7 | 12 | 1 | 0.3750 | 0.3684 |
| proposed_vs_baseline_no_context | context_relevance | 10 | 9 | 1 | 0.5250 | 0.5263 |
| proposed_vs_baseline_no_context | persona_consistency | 2 | 6 | 12 | 0.4000 | 0.2500 |
| proposed_vs_baseline_no_context | naturalness | 4 | 16 | 0 | 0.2000 | 0.2000 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 5 | 3 | 12 | 0.5500 | 0.6250 |
| proposed_vs_baseline_no_context | context_overlap | 11 | 8 | 1 | 0.5750 | 0.5789 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 1 | 4 | 15 | 0.4250 | 0.2000 |
| proposed_vs_baseline_no_context | persona_style | 1 | 3 | 16 | 0.4500 | 0.2500 |
| proposed_vs_baseline_no_context | distinct1 | 3 | 14 | 3 | 0.2250 | 0.1765 |
| proposed_vs_baseline_no_context | length_score | 5 | 14 | 1 | 0.2750 | 0.2632 |
| proposed_vs_baseline_no_context | sentence_score | 3 | 9 | 8 | 0.3500 | 0.2500 |
| proposed_vs_baseline_no_context | bertscore_f1 | 15 | 5 | 0 | 0.7500 | 0.7500 |
| proposed_vs_baseline_no_context | overall_quality | 6 | 14 | 0 | 0.3000 | 0.3000 |
| controlled_vs_proposed_raw | context_relevance | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_vs_proposed_raw | persona_consistency | 16 | 0 | 4 | 0.9000 | 1.0000 |
| controlled_vs_proposed_raw | naturalness | 16 | 4 | 0 | 0.8000 | 0.8000 |
| controlled_vs_proposed_raw | context_keyword_coverage | 18 | 1 | 1 | 0.9250 | 0.9474 |
| controlled_vs_proposed_raw | context_overlap | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 16 | 0 | 4 | 0.9000 | 1.0000 |
| controlled_vs_proposed_raw | persona_style | 4 | 2 | 14 | 0.5500 | 0.6667 |
| controlled_vs_proposed_raw | distinct1 | 10 | 10 | 0 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | length_score | 15 | 4 | 1 | 0.7750 | 0.7895 |
| controlled_vs_proposed_raw | sentence_score | 14 | 0 | 6 | 0.8500 | 1.0000 |
| controlled_vs_proposed_raw | bertscore_f1 | 14 | 6 | 0 | 0.7000 | 0.7000 |
| controlled_vs_proposed_raw | overall_quality | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_relevance | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 12 | 5 | 3 | 0.6750 | 0.7059 |
| controlled_vs_candidate_no_context | naturalness | 12 | 8 | 0 | 0.6000 | 0.6000 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 12 | 5 | 3 | 0.6750 | 0.7059 |
| controlled_vs_candidate_no_context | persona_style | 4 | 2 | 14 | 0.5500 | 0.6667 |
| controlled_vs_candidate_no_context | distinct1 | 9 | 11 | 0 | 0.4500 | 0.4500 |
| controlled_vs_candidate_no_context | length_score | 12 | 6 | 2 | 0.6500 | 0.6667 |
| controlled_vs_candidate_no_context | sentence_score | 9 | 0 | 11 | 0.7250 | 1.0000 |
| controlled_vs_candidate_no_context | bertscore_f1 | 18 | 2 | 0 | 0.9000 | 0.9000 |
| controlled_vs_candidate_no_context | overall_quality | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_vs_baseline_no_context | context_relevance | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 14 | 1 | 5 | 0.8250 | 0.9333 |
| controlled_vs_baseline_no_context | naturalness | 13 | 7 | 0 | 0.6500 | 0.6500 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | context_overlap | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 14 | 1 | 5 | 0.8250 | 0.9333 |
| controlled_vs_baseline_no_context | persona_style | 3 | 2 | 15 | 0.5250 | 0.6000 |
| controlled_vs_baseline_no_context | distinct1 | 3 | 17 | 0 | 0.1500 | 0.1500 |
| controlled_vs_baseline_no_context | length_score | 14 | 6 | 0 | 0.7000 | 0.7000 |
| controlled_vs_baseline_no_context | sentence_score | 8 | 0 | 12 | 0.7000 | 1.0000 |
| controlled_vs_baseline_no_context | bertscore_f1 | 15 | 5 | 0 | 0.7500 | 0.7500 |
| controlled_vs_baseline_no_context | overall_quality | 19 | 1 | 0 | 0.9500 | 0.9500 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 20 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 14 | 1 | 5 | 0.8250 | 0.9333 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 13 | 7 | 0 | 0.6500 | 0.6500 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 20 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 20 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 14 | 1 | 5 | 0.8250 | 0.9333 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 3 | 2 | 15 | 0.5250 | 0.6000 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | 3 | 17 | 0 | 0.1500 | 0.1500 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 14 | 6 | 0 | 0.7000 | 0.7000 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 8 | 0 | 12 | 0.7000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 15 | 5 | 0 | 0.7500 | 0.7500 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 19 | 1 | 0 | 0.9500 | 0.9500 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.4500 | 0.4500 | 0.5500 |
| proposed_contextual | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.4000 | 0.0000 | 0.0000 |
| baseline_no_context | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `8`
- Unique template signatures: `18`
- Template signature ratio: `0.9000`
- Effective sample size by source clustering: `6.67`
- Effective sample size by template-signature clustering: `16.67`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: enabled.

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
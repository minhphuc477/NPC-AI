# Proposal Alignment Evaluation Report

- Run ID: `20260304T140701Z`
- Generated: `2026-03-04T14:14:35.379801+00:00`
- Scenarios: `artifacts\proposal_control_tuning\best_v3\20260304T140701Z\scenarios.jsonl`
- Scenario count: `40`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off
- `baseline_no_context`: model `phi3:mini`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.3007 (0.2751, 0.3276) | 0.3774 (0.3316, 0.4323) | 0.8571 (0.8300, 0.8842) | 0.4020 (0.3833, 0.4218) | 0.0742 |
| proposed_contextual | 0.0805 (0.0481, 0.1197) | 0.1654 (0.1191, 0.2205) | 0.8011 (0.7812, 0.8226) | 0.2326 (0.2025, 0.2684) | 0.0677 |
| candidate_no_context | 0.0266 (0.0141, 0.0422) | 0.1639 (0.1212, 0.2152) | 0.8077 (0.7831, 0.8350) | 0.2069 (0.1870, 0.2278) | 0.0290 |
| baseline_no_context | 0.0545 (0.0397, 0.0703) | 0.1968 (0.1622, 0.2375) | 0.8886 (0.8716, 0.9053) | 0.2454 (0.2321, 0.2597) | 0.0537 |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0539 | 2.0276 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0015 | 0.0094 |
| proposed_vs_candidate_no_context | naturalness | -0.0065 | -0.0081 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0717 | 3.3943 |
| proposed_vs_candidate_no_context | context_overlap | 0.0123 | 0.3129 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0018 | -0.0216 |
| proposed_vs_candidate_no_context | persona_style | 0.0148 | 0.0303 |
| proposed_vs_candidate_no_context | distinct1 | 0.0049 | 0.0053 |
| proposed_vs_candidate_no_context | length_score | -0.0250 | -0.0875 |
| proposed_vs_candidate_no_context | sentence_score | -0.0350 | -0.0458 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0386 | 1.3311 |
| proposed_vs_candidate_no_context | overall_quality | 0.0257 | 0.1242 |
| proposed_vs_baseline_no_context | context_relevance | 0.0260 | 0.4780 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0314 | -0.1596 |
| proposed_vs_baseline_no_context | naturalness | -0.0875 | -0.0985 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0341 | 0.5799 |
| proposed_vs_baseline_no_context | context_overlap | 0.0072 | 0.1629 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0167 | -0.1707 |
| proposed_vs_baseline_no_context | persona_style | -0.0903 | -0.1522 |
| proposed_vs_baseline_no_context | distinct1 | -0.0407 | -0.0415 |
| proposed_vs_baseline_no_context | length_score | -0.2817 | -0.5192 |
| proposed_vs_baseline_no_context | sentence_score | -0.1488 | -0.1695 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0140 | 0.2614 |
| proposed_vs_baseline_no_context | overall_quality | -0.0128 | -0.0523 |
| controlled_vs_proposed_raw | context_relevance | 0.2202 | 2.7353 |
| controlled_vs_proposed_raw | persona_consistency | 0.2120 | 1.2819 |
| controlled_vs_proposed_raw | naturalness | 0.0560 | 0.0698 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2867 | 3.0869 |
| controlled_vs_proposed_raw | context_overlap | 0.0650 | 1.2590 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2460 | 3.0382 |
| controlled_vs_proposed_raw | persona_style | 0.0762 | 0.1516 |
| controlled_vs_proposed_raw | distinct1 | -0.0189 | -0.0201 |
| controlled_vs_proposed_raw | length_score | 0.2275 | 0.8722 |
| controlled_vs_proposed_raw | sentence_score | 0.2013 | 0.2762 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0065 | 0.0964 |
| controlled_vs_proposed_raw | overall_quality | 0.1694 | 0.7282 |
| controlled_vs_candidate_no_context | context_relevance | 0.2741 | 10.3092 |
| controlled_vs_candidate_no_context | persona_consistency | 0.2135 | 1.3033 |
| controlled_vs_candidate_no_context | naturalness | 0.0494 | 0.0612 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3584 | 16.9588 |
| controlled_vs_candidate_no_context | context_overlap | 0.0773 | 1.9658 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2442 | 2.9511 |
| controlled_vs_candidate_no_context | persona_style | 0.0911 | 0.1865 |
| controlled_vs_candidate_no_context | distinct1 | -0.0139 | -0.0149 |
| controlled_vs_candidate_no_context | length_score | 0.2025 | 0.7085 |
| controlled_vs_candidate_no_context | sentence_score | 0.1663 | 0.2177 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0452 | 1.5557 |
| controlled_vs_candidate_no_context | overall_quality | 0.1951 | 0.9429 |
| controlled_vs_baseline_no_context | context_relevance | 0.2462 | 4.5207 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1806 | 0.9178 |
| controlled_vs_baseline_no_context | naturalness | -0.0315 | -0.0355 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3208 | 5.4568 |
| controlled_vs_baseline_no_context | context_overlap | 0.0722 | 1.6270 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2293 | 2.3488 |
| controlled_vs_baseline_no_context | persona_style | -0.0141 | -0.0237 |
| controlled_vs_baseline_no_context | distinct1 | -0.0596 | -0.0607 |
| controlled_vs_baseline_no_context | length_score | -0.0542 | -0.0998 |
| controlled_vs_baseline_no_context | sentence_score | 0.0525 | 0.0598 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0206 | 0.3830 |
| controlled_vs_baseline_no_context | overall_quality | 0.1565 | 0.6378 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2462 | 4.5207 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1806 | 0.9178 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0315 | -0.0355 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3208 | 5.4568 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0722 | 1.6270 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2293 | 2.3488 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | -0.0141 | -0.0237 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0596 | -0.0607 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | -0.0542 | -0.0998 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0525 | 0.0598 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0206 | 0.3830 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1565 | 0.6378 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0539 | (0.0197, 0.0928) | 0.0000 | 0.0539 | (0.0175, 0.0996) | 0.0010 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0015 | (-0.0620, 0.0666) | 0.4807 | 0.0015 | (-0.0404, 0.0451) | 0.4857 |
| proposed_vs_candidate_no_context | naturalness | -0.0065 | (-0.0307, 0.0166) | 0.7043 | -0.0065 | (-0.0320, 0.0134) | 0.7443 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0717 | (0.0270, 0.1246) | 0.0000 | 0.0717 | (0.0275, 0.1273) | 0.0003 |
| proposed_vs_candidate_no_context | context_overlap | 0.0123 | (-0.0003, 0.0267) | 0.0273 | 0.0123 | (-0.0058, 0.0318) | 0.0877 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0018 | (-0.0768, 0.0750) | 0.5200 | -0.0018 | (-0.0488, 0.0481) | 0.5630 |
| proposed_vs_candidate_no_context | persona_style | 0.0148 | (-0.0422, 0.0646) | 0.2770 | 0.0148 | (-0.0402, 0.0863) | 0.3123 |
| proposed_vs_candidate_no_context | distinct1 | 0.0049 | (-0.0054, 0.0147) | 0.1713 | 0.0049 | (-0.0091, 0.0172) | 0.2283 |
| proposed_vs_candidate_no_context | length_score | -0.0250 | (-0.1275, 0.0650) | 0.7040 | -0.0250 | (-0.1261, 0.0702) | 0.7017 |
| proposed_vs_candidate_no_context | sentence_score | -0.0350 | (-0.0962, 0.0350) | 0.8743 | -0.0350 | (-0.1016, 0.0167) | 0.9153 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0386 | (0.0155, 0.0632) | 0.0007 | 0.0386 | (0.0128, 0.0719) | 0.0003 |
| proposed_vs_candidate_no_context | overall_quality | 0.0257 | (-0.0074, 0.0625) | 0.0650 | 0.0257 | (0.0059, 0.0473) | 0.0057 |
| proposed_vs_baseline_no_context | context_relevance | 0.0260 | (-0.0061, 0.0649) | 0.0683 | 0.0260 | (-0.0108, 0.0702) | 0.0830 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0314 | (-0.0912, 0.0295) | 0.8427 | -0.0314 | (-0.1014, 0.0558) | 0.7860 |
| proposed_vs_baseline_no_context | naturalness | -0.0875 | (-0.1155, -0.0588) | 1.0000 | -0.0875 | (-0.1291, -0.0414) | 1.0000 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0341 | (-0.0096, 0.0842) | 0.0680 | 0.0341 | (-0.0136, 0.0943) | 0.0877 |
| proposed_vs_baseline_no_context | context_overlap | 0.0072 | (-0.0038, 0.0196) | 0.1080 | 0.0072 | (-0.0081, 0.0248) | 0.1790 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0167 | (-0.0835, 0.0560) | 0.6927 | -0.0167 | (-0.0906, 0.0873) | 0.6653 |
| proposed_vs_baseline_no_context | persona_style | -0.0903 | (-0.1667, -0.0218) | 0.9927 | -0.0903 | (-0.2207, 0.0141) | 0.9367 |
| proposed_vs_baseline_no_context | distinct1 | -0.0407 | (-0.0550, -0.0249) | 1.0000 | -0.0407 | (-0.0543, -0.0258) | 1.0000 |
| proposed_vs_baseline_no_context | length_score | -0.2817 | (-0.3917, -0.1650) | 1.0000 | -0.2817 | (-0.4488, -0.0851) | 0.9980 |
| proposed_vs_baseline_no_context | sentence_score | -0.1487 | (-0.2188, -0.0788) | 1.0000 | -0.1487 | (-0.2170, -0.0583) | 1.0000 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0140 | (-0.0054, 0.0339) | 0.0870 | 0.0140 | (-0.0038, 0.0370) | 0.0707 |
| proposed_vs_baseline_no_context | overall_quality | -0.0128 | (-0.0436, 0.0195) | 0.7870 | -0.0128 | (-0.0448, 0.0271) | 0.7580 |
| controlled_vs_proposed_raw | context_relevance | 0.2202 | (0.1762, 0.2623) | 0.0000 | 0.2202 | (0.1694, 0.2663) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.2120 | (0.1422, 0.2841) | 0.0000 | 0.2120 | (0.1105, 0.2977) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0560 | (0.0225, 0.0905) | 0.0003 | 0.0560 | (0.0257, 0.0821) | 0.0000 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2867 | (0.2261, 0.3407) | 0.0000 | 0.2867 | (0.2182, 0.3477) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0650 | (0.0489, 0.0829) | 0.0000 | 0.0650 | (0.0441, 0.0875) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2460 | (0.1599, 0.3316) | 0.0000 | 0.2460 | (0.1271, 0.3553) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.0763 | (0.0145, 0.1482) | 0.0063 | 0.0763 | (-0.0256, 0.2142) | 0.1183 |
| controlled_vs_proposed_raw | distinct1 | -0.0189 | (-0.0376, -0.0024) | 0.9883 | -0.0189 | (-0.0414, -0.0009) | 0.9803 |
| controlled_vs_proposed_raw | length_score | 0.2275 | (0.1000, 0.3617) | 0.0007 | 0.2275 | (0.1096, 0.3273) | 0.0007 |
| controlled_vs_proposed_raw | sentence_score | 0.2012 | (0.1313, 0.2625) | 0.0000 | 0.2012 | (0.1544, 0.2360) | 0.0000 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0065 | (-0.0180, 0.0290) | 0.3120 | 0.0065 | (-0.0169, 0.0291) | 0.2830 |
| controlled_vs_proposed_raw | overall_quality | 0.1694 | (0.1335, 0.2057) | 0.0000 | 0.1694 | (0.1239, 0.2085) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2741 | (0.2461, 0.3014) | 0.0000 | 0.2741 | (0.2486, 0.3072) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.2135 | (0.1512, 0.2766) | 0.0000 | 0.2135 | (0.1304, 0.2891) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.0494 | (0.0138, 0.0845) | 0.0030 | 0.0494 | (0.0010, 0.0851) | 0.0233 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3584 | (0.3205, 0.4003) | 0.0000 | 0.3584 | (0.3177, 0.4046) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0773 | (0.0633, 0.0936) | 0.0000 | 0.0773 | (0.0632, 0.0949) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2442 | (0.1681, 0.3215) | 0.0000 | 0.2442 | (0.1502, 0.3442) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.0911 | (0.0223, 0.1705) | 0.0047 | 0.0911 | (-0.0138, 0.2195) | 0.0510 |
| controlled_vs_candidate_no_context | distinct1 | -0.0139 | (-0.0324, 0.0038) | 0.9367 | -0.0139 | (-0.0420, 0.0068) | 0.8860 |
| controlled_vs_candidate_no_context | length_score | 0.2025 | (0.0608, 0.3400) | 0.0030 | 0.2025 | (0.0264, 0.3293) | 0.0133 |
| controlled_vs_candidate_no_context | sentence_score | 0.1662 | (0.0962, 0.2275) | 0.0000 | 0.1662 | (0.0683, 0.2423) | 0.0003 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0452 | (0.0230, 0.0666) | 0.0000 | 0.0452 | (0.0217, 0.0789) | 0.0000 |
| controlled_vs_candidate_no_context | overall_quality | 0.1951 | (0.1696, 0.2220) | 0.0000 | 0.1951 | (0.1673, 0.2196) | 0.0000 |
| controlled_vs_baseline_no_context | context_relevance | 0.2462 | (0.2197, 0.2739) | 0.0000 | 0.2462 | (0.2135, 0.2796) | 0.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1806 | (0.1295, 0.2360) | 0.0000 | 0.1806 | (0.1078, 0.2428) | 0.0000 |
| controlled_vs_baseline_no_context | naturalness | -0.0315 | (-0.0671, 0.0051) | 0.9500 | -0.0315 | (-0.0563, -0.0009) | 0.9783 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3208 | (0.2847, 0.3592) | 0.0000 | 0.3208 | (0.2728, 0.3673) | 0.0000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0722 | (0.0581, 0.0884) | 0.0000 | 0.0722 | (0.0596, 0.0882) | 0.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2293 | (0.1682, 0.2956) | 0.0000 | 0.2293 | (0.1444, 0.3064) | 0.0000 |
| controlled_vs_baseline_no_context | persona_style | -0.0141 | (-0.0535, 0.0241) | 0.7607 | -0.0141 | (-0.0505, 0.0248) | 0.7870 |
| controlled_vs_baseline_no_context | distinct1 | -0.0596 | (-0.0766, -0.0435) | 1.0000 | -0.0596 | (-0.0734, -0.0463) | 1.0000 |
| controlled_vs_baseline_no_context | length_score | -0.0542 | (-0.2117, 0.0992) | 0.7567 | -0.0542 | (-0.1856, 0.1060) | 0.7810 |
| controlled_vs_baseline_no_context | sentence_score | 0.0525 | (-0.0175, 0.1225) | 0.0907 | 0.0525 | (0.0000, 0.1203) | 0.0487 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0206 | (-0.0010, 0.0429) | 0.0307 | 0.0206 | (-0.0044, 0.0442) | 0.0540 |
| controlled_vs_baseline_no_context | overall_quality | 0.1565 | (0.1354, 0.1771) | 0.0000 | 0.1565 | (0.1319, 0.1752) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2462 | (0.2199, 0.2748) | 0.0000 | 0.2462 | (0.2140, 0.2788) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1806 | (0.1288, 0.2372) | 0.0000 | 0.1806 | (0.1055, 0.2419) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0315 | (-0.0677, 0.0051) | 0.9537 | -0.0315 | (-0.0567, -0.0001) | 0.9750 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3208 | (0.2846, 0.3602) | 0.0000 | 0.3208 | (0.2729, 0.3690) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0722 | (0.0580, 0.0884) | 0.0000 | 0.0722 | (0.0601, 0.0885) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2293 | (0.1694, 0.2932) | 0.0000 | 0.2293 | (0.1452, 0.3030) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | -0.0141 | (-0.0535, 0.0256) | 0.7650 | -0.0141 | (-0.0513, 0.0252) | 0.7880 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0596 | (-0.0769, -0.0431) | 1.0000 | -0.0596 | (-0.0740, -0.0468) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | -0.0542 | (-0.2100, 0.1008) | 0.7420 | -0.0542 | (-0.1911, 0.1065) | 0.7477 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0525 | (-0.0175, 0.1225) | 0.0900 | 0.0525 | (0.0000, 0.1242) | 0.0450 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0206 | (-0.0013, 0.0417) | 0.0330 | 0.0206 | (-0.0037, 0.0460) | 0.0490 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1565 | (0.1351, 0.1778) | 0.0000 | 0.1565 | (0.1333, 0.1756) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 19 | 5 | 16 | 0.6750 | 0.7917 |
| proposed_vs_candidate_no_context | persona_consistency | 9 | 8 | 23 | 0.5125 | 0.5294 |
| proposed_vs_candidate_no_context | naturalness | 15 | 9 | 16 | 0.5750 | 0.6250 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 15 | 3 | 22 | 0.6500 | 0.8333 |
| proposed_vs_candidate_no_context | context_overlap | 19 | 5 | 16 | 0.6750 | 0.7917 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 5 | 7 | 28 | 0.4750 | 0.4167 |
| proposed_vs_candidate_no_context | persona_style | 6 | 2 | 32 | 0.5500 | 0.7500 |
| proposed_vs_candidate_no_context | distinct1 | 14 | 8 | 18 | 0.5750 | 0.6364 |
| proposed_vs_candidate_no_context | length_score | 13 | 11 | 16 | 0.5250 | 0.5417 |
| proposed_vs_candidate_no_context | sentence_score | 6 | 10 | 24 | 0.4500 | 0.3750 |
| proposed_vs_candidate_no_context | bertscore_f1 | 23 | 7 | 10 | 0.7000 | 0.7667 |
| proposed_vs_candidate_no_context | overall_quality | 18 | 12 | 10 | 0.5750 | 0.6000 |
| proposed_vs_baseline_no_context | context_relevance | 19 | 21 | 0 | 0.4750 | 0.4750 |
| proposed_vs_baseline_no_context | persona_consistency | 7 | 16 | 17 | 0.3875 | 0.3043 |
| proposed_vs_baseline_no_context | naturalness | 8 | 32 | 0 | 0.2000 | 0.2000 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 13 | 9 | 18 | 0.5500 | 0.5909 |
| proposed_vs_baseline_no_context | context_overlap | 21 | 19 | 0 | 0.5250 | 0.5250 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 6 | 11 | 23 | 0.4375 | 0.3529 |
| proposed_vs_baseline_no_context | persona_style | 2 | 10 | 28 | 0.4000 | 0.1667 |
| proposed_vs_baseline_no_context | distinct1 | 6 | 29 | 5 | 0.2125 | 0.1714 |
| proposed_vs_baseline_no_context | length_score | 9 | 31 | 0 | 0.2250 | 0.2250 |
| proposed_vs_baseline_no_context | sentence_score | 3 | 20 | 17 | 0.2875 | 0.1304 |
| proposed_vs_baseline_no_context | bertscore_f1 | 26 | 14 | 0 | 0.6500 | 0.6500 |
| proposed_vs_baseline_no_context | overall_quality | 14 | 26 | 0 | 0.3500 | 0.3500 |
| controlled_vs_proposed_raw | context_relevance | 37 | 3 | 0 | 0.9250 | 0.9250 |
| controlled_vs_proposed_raw | persona_consistency | 34 | 4 | 2 | 0.8750 | 0.8947 |
| controlled_vs_proposed_raw | naturalness | 29 | 11 | 0 | 0.7250 | 0.7250 |
| controlled_vs_proposed_raw | context_keyword_coverage | 36 | 2 | 2 | 0.9250 | 0.9474 |
| controlled_vs_proposed_raw | context_overlap | 37 | 3 | 0 | 0.9250 | 0.9250 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 34 | 4 | 2 | 0.8750 | 0.8947 |
| controlled_vs_proposed_raw | persona_style | 8 | 2 | 30 | 0.5750 | 0.8000 |
| controlled_vs_proposed_raw | distinct1 | 18 | 19 | 3 | 0.4875 | 0.4865 |
| controlled_vs_proposed_raw | length_score | 25 | 13 | 2 | 0.6500 | 0.6579 |
| controlled_vs_proposed_raw | sentence_score | 25 | 2 | 13 | 0.7875 | 0.9259 |
| controlled_vs_proposed_raw | bertscore_f1 | 24 | 16 | 0 | 0.6000 | 0.6000 |
| controlled_vs_proposed_raw | overall_quality | 36 | 4 | 0 | 0.9000 | 0.9000 |
| controlled_vs_candidate_no_context | context_relevance | 40 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 33 | 4 | 3 | 0.8625 | 0.8919 |
| controlled_vs_candidate_no_context | naturalness | 27 | 13 | 0 | 0.6750 | 0.6750 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 40 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 40 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 33 | 3 | 4 | 0.8750 | 0.9167 |
| controlled_vs_candidate_no_context | persona_style | 10 | 2 | 28 | 0.6000 | 0.8333 |
| controlled_vs_candidate_no_context | distinct1 | 20 | 20 | 0 | 0.5000 | 0.5000 |
| controlled_vs_candidate_no_context | length_score | 24 | 14 | 2 | 0.6250 | 0.6316 |
| controlled_vs_candidate_no_context | sentence_score | 22 | 3 | 15 | 0.7375 | 0.8800 |
| controlled_vs_candidate_no_context | bertscore_f1 | 32 | 8 | 0 | 0.8000 | 0.8000 |
| controlled_vs_candidate_no_context | overall_quality | 40 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | context_relevance | 40 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 34 | 3 | 3 | 0.8875 | 0.9189 |
| controlled_vs_baseline_no_context | naturalness | 17 | 23 | 0 | 0.4250 | 0.4250 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 40 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | context_overlap | 39 | 1 | 0 | 0.9750 | 0.9750 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 34 | 2 | 4 | 0.9000 | 0.9444 |
| controlled_vs_baseline_no_context | persona_style | 3 | 5 | 32 | 0.4750 | 0.3750 |
| controlled_vs_baseline_no_context | distinct1 | 3 | 36 | 1 | 0.0875 | 0.0769 |
| controlled_vs_baseline_no_context | length_score | 19 | 21 | 0 | 0.4750 | 0.4750 |
| controlled_vs_baseline_no_context | sentence_score | 12 | 6 | 22 | 0.5750 | 0.6667 |
| controlled_vs_baseline_no_context | bertscore_f1 | 24 | 16 | 0 | 0.6000 | 0.6000 |
| controlled_vs_baseline_no_context | overall_quality | 39 | 1 | 0 | 0.9750 | 0.9750 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 40 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 34 | 3 | 3 | 0.8875 | 0.9189 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 17 | 23 | 0 | 0.4250 | 0.4250 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 40 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 39 | 1 | 0 | 0.9750 | 0.9750 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 34 | 2 | 4 | 0.9000 | 0.9444 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 3 | 5 | 32 | 0.4750 | 0.3750 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | 3 | 36 | 1 | 0.0875 | 0.0769 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 19 | 21 | 0 | 0.4750 | 0.4750 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 12 | 6 | 22 | 0.5750 | 0.6667 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 24 | 16 | 0 | 0.6000 | 0.6000 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 39 | 1 | 0 | 0.9750 | 0.9750 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.3250 | 0.0250 | 0.7250 |
| proposed_contextual | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.6000 | 0.0000 | 0.0000 |
| baseline_no_context | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `8`
- Unique template signatures: `33`
- Template signature ratio: `0.8250`
- Effective sample size by source clustering: `7.02`
- Effective sample size by template-signature clustering: `28.57`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: enabled.

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
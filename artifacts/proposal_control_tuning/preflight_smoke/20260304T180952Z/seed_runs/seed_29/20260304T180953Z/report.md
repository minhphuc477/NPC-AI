# Proposal Alignment Evaluation Report

- Run ID: `20260304T180953Z`
- Generated: `2026-03-04T18:13:54.611885+00:00`
- Scenarios: `artifacts\proposal_control_tuning\preflight_smoke\20260304T180952Z\seed_runs\seed_29\20260304T180953Z\scenarios.jsonl`
- Scenario count: `8`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off
- `baseline_no_context`: model `phi3:mini`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2842 (0.2356, 0.3426) | 0.2558 (0.1966, 0.3150) | 0.8514 (0.7847, 0.9130) | 0.3538 (0.3232, 0.3854) | 0.0630 |
| proposed_contextual | 0.0606 (0.0193, 0.1236) | 0.1717 (0.0600, 0.3558) | 0.8252 (0.7779, 0.8757) | 0.2304 (0.1784, 0.3002) | 0.0635 |
| candidate_no_context | 0.0446 (0.0197, 0.0766) | 0.1492 (0.0875, 0.2283) | 0.8101 (0.7652, 0.8642) | 0.2122 (0.1762, 0.2545) | 0.0495 |
| baseline_no_context | 0.0302 (0.0132, 0.0523) | 0.1442 (0.1117, 0.1875) | 0.8603 (0.8214, 0.9028) | 0.2146 (0.1948, 0.2354) | 0.0622 |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0160 | 0.3592 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0225 | 0.1508 |
| proposed_vs_candidate_no_context | naturalness | 0.0151 | 0.0186 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0123 | 0.2826 |
| proposed_vs_candidate_no_context | context_overlap | 0.0247 | 0.5252 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0375 | 0.5625 |
| proposed_vs_candidate_no_context | persona_style | -0.0375 | -0.0783 |
| proposed_vs_candidate_no_context | distinct1 | 0.0028 | 0.0030 |
| proposed_vs_candidate_no_context | length_score | 0.0917 | 0.3188 |
| proposed_vs_candidate_no_context | sentence_score | -0.0437 | -0.0560 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0140 | 0.2827 |
| proposed_vs_candidate_no_context | overall_quality | 0.0181 | 0.0854 |
| proposed_vs_baseline_no_context | context_relevance | 0.0304 | 1.0095 |
| proposed_vs_baseline_no_context | persona_consistency | 0.0275 | 0.1908 |
| proposed_vs_baseline_no_context | naturalness | -0.0351 | -0.0408 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0331 | 1.4583 |
| proposed_vs_baseline_no_context | context_overlap | 0.0241 | 0.5082 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 0.0542 | 1.0833 |
| proposed_vs_baseline_no_context | persona_style | -0.0792 | -0.1520 |
| proposed_vs_baseline_no_context | distinct1 | -0.0368 | -0.0377 |
| proposed_vs_baseline_no_context | length_score | -0.0583 | -0.1333 |
| proposed_vs_baseline_no_context | sentence_score | -0.0875 | -0.1061 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0013 | 0.0209 |
| proposed_vs_baseline_no_context | overall_quality | 0.0157 | 0.0734 |
| controlled_vs_proposed_raw | context_relevance | 0.2236 | 3.6895 |
| controlled_vs_proposed_raw | persona_consistency | 0.0842 | 0.4903 |
| controlled_vs_proposed_raw | naturalness | 0.0262 | 0.0318 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.3021 | 5.4068 |
| controlled_vs_proposed_raw | context_overlap | 0.0403 | 0.5633 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.0958 | 0.9200 |
| controlled_vs_proposed_raw | persona_style | 0.0375 | 0.0849 |
| controlled_vs_proposed_raw | distinct1 | -0.0006 | -0.0007 |
| controlled_vs_proposed_raw | length_score | 0.0667 | 0.1758 |
| controlled_vs_proposed_raw | sentence_score | 0.1312 | 0.1780 |
| controlled_vs_proposed_raw | bertscore_f1 | -0.0004 | -0.0066 |
| controlled_vs_proposed_raw | overall_quality | 0.1234 | 0.5358 |
| controlled_vs_candidate_no_context | context_relevance | 0.2396 | 5.3742 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1067 | 0.7151 |
| controlled_vs_candidate_no_context | naturalness | 0.0413 | 0.0510 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3144 | 7.2174 |
| controlled_vs_candidate_no_context | context_overlap | 0.0650 | 1.3842 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1333 | 2.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.0000 | 0.0000 |
| controlled_vs_candidate_no_context | distinct1 | 0.0022 | 0.0023 |
| controlled_vs_candidate_no_context | length_score | 0.1583 | 0.5507 |
| controlled_vs_candidate_no_context | sentence_score | 0.0875 | 0.1120 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0136 | 0.2742 |
| controlled_vs_candidate_no_context | overall_quality | 0.1415 | 0.6670 |
| controlled_vs_baseline_no_context | context_relevance | 0.2540 | 8.4237 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1117 | 0.7746 |
| controlled_vs_baseline_no_context | naturalness | -0.0089 | -0.0104 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3352 | 14.7500 |
| controlled_vs_baseline_no_context | context_overlap | 0.0645 | 1.3578 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1500 | 3.0000 |
| controlled_vs_baseline_no_context | persona_style | -0.0417 | -0.0800 |
| controlled_vs_baseline_no_context | distinct1 | -0.0374 | -0.0383 |
| controlled_vs_baseline_no_context | length_score | 0.0083 | 0.0190 |
| controlled_vs_baseline_no_context | sentence_score | 0.0438 | 0.0530 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0009 | 0.0142 |
| controlled_vs_baseline_no_context | overall_quality | 0.1392 | 0.6485 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2540 | 8.4237 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1117 | 0.7746 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0089 | -0.0104 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3352 | 14.7500 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0645 | 1.3578 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1500 | 3.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | -0.0417 | -0.0800 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0374 | -0.0383 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.0083 | 0.0190 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0438 | 0.0530 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0009 | 0.0142 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1392 | 0.6485 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0160 | (-0.0195, 0.0688) | 0.3163 | 0.0160 | (-0.0226, 0.0609) | 0.3273 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0225 | (-0.0725, 0.1500) | 0.4070 | 0.0225 | (-0.0667, 0.1714) | 0.3980 |
| proposed_vs_candidate_no_context | naturalness | 0.0151 | (-0.0137, 0.0461) | 0.1643 | 0.0151 | (-0.0126, 0.0382) | 0.1323 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0123 | (-0.0312, 0.0682) | 0.3647 | 0.0123 | (-0.0357, 0.0578) | 0.3350 |
| proposed_vs_candidate_no_context | context_overlap | 0.0247 | (-0.0081, 0.0774) | 0.1640 | 0.0247 | (-0.0058, 0.0660) | 0.1267 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0375 | (-0.0750, 0.1875) | 0.3750 | 0.0375 | (-0.0600, 0.2500) | 0.3547 |
| proposed_vs_candidate_no_context | persona_style | -0.0375 | (-0.0875, 0.0000) | 1.0000 | -0.0375 | (-0.0857, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | distinct1 | 0.0028 | (-0.0385, 0.0322) | 0.4083 | 0.0028 | (-0.0396, 0.0394) | 0.4070 |
| proposed_vs_candidate_no_context | length_score | 0.0917 | (-0.0542, 0.2417) | 0.1187 | 0.0917 | (-0.0222, 0.1958) | 0.0443 |
| proposed_vs_candidate_no_context | sentence_score | -0.0437 | (-0.1312, 0.0000) | 1.0000 | -0.0437 | (-0.1750, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0140 | (-0.0004, 0.0314) | 0.0330 | 0.0140 | (-0.0031, 0.0285) | 0.0583 |
| proposed_vs_candidate_no_context | overall_quality | 0.0181 | (-0.0051, 0.0513) | 0.1013 | 0.0181 | (-0.0051, 0.0589) | 0.0610 |
| proposed_vs_baseline_no_context | context_relevance | 0.0304 | (-0.0265, 0.1005) | 0.1837 | 0.0304 | (-0.0065, 0.0765) | 0.0623 |
| proposed_vs_baseline_no_context | persona_consistency | 0.0275 | (-0.1042, 0.2234) | 0.3943 | 0.0275 | (-0.0963, 0.3022) | 0.3940 |
| proposed_vs_baseline_no_context | naturalness | -0.0351 | (-0.1142, 0.0471) | 0.8043 | -0.0351 | (-0.1228, 0.0540) | 0.7897 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0331 | (-0.0350, 0.1136) | 0.2500 | 0.0331 | (-0.0182, 0.0898) | 0.1440 |
| proposed_vs_baseline_no_context | context_overlap | 0.0241 | (-0.0100, 0.0715) | 0.1387 | 0.0241 | (-0.0024, 0.0556) | 0.0440 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 0.0542 | (-0.1000, 0.2875) | 0.3310 | 0.0542 | (-0.0889, 0.3833) | 0.3337 |
| proposed_vs_baseline_no_context | persona_style | -0.0792 | (-0.1667, 0.0000) | 1.0000 | -0.0792 | (-0.2000, 0.0000) | 1.0000 |
| proposed_vs_baseline_no_context | distinct1 | -0.0368 | (-0.0729, 0.0030) | 0.9657 | -0.0368 | (-0.0789, 0.0157) | 0.9130 |
| proposed_vs_baseline_no_context | length_score | -0.0583 | (-0.3667, 0.2667) | 0.6383 | -0.0583 | (-0.3926, 0.2625) | 0.6087 |
| proposed_vs_baseline_no_context | sentence_score | -0.0875 | (-0.2188, 0.0000) | 1.0000 | -0.0875 | (-0.1556, 0.0000) | 1.0000 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0013 | (-0.0390, 0.0409) | 0.4827 | 0.0013 | (-0.0297, 0.0374) | 0.4293 |
| proposed_vs_baseline_no_context | overall_quality | 0.0157 | (-0.0527, 0.1001) | 0.3810 | 0.0157 | (-0.0341, 0.1085) | 0.3480 |
| controlled_vs_proposed_raw | context_relevance | 0.2236 | (0.1401, 0.2873) | 0.0000 | 0.2236 | (0.1676, 0.2970) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.0842 | (-0.1067, 0.2125) | 0.1743 | 0.0842 | (-0.1811, 0.2175) | 0.2310 |
| controlled_vs_proposed_raw | naturalness | 0.0262 | (-0.0663, 0.1179) | 0.2997 | 0.0262 | (-0.0914, 0.1255) | 0.3153 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.3021 | (0.1998, 0.3816) | 0.0000 | 0.3021 | (0.2323, 0.3902) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0403 | (-0.0053, 0.0757) | 0.0430 | 0.0403 | (0.0121, 0.0730) | 0.0037 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.0958 | (-0.1500, 0.2625) | 0.1947 | 0.0958 | (-0.2000, 0.2600) | 0.2260 |
| controlled_vs_proposed_raw | persona_style | 0.0375 | (0.0000, 0.0875) | 0.0950 | 0.0375 | (0.0000, 0.0857) | 0.0810 |
| controlled_vs_proposed_raw | distinct1 | -0.0006 | (-0.0597, 0.0554) | 0.5160 | -0.0006 | (-0.0803, 0.0580) | 0.5183 |
| controlled_vs_proposed_raw | length_score | 0.0667 | (-0.2708, 0.3958) | 0.3757 | 0.0667 | (-0.3286, 0.4125) | 0.4023 |
| controlled_vs_proposed_raw | sentence_score | 0.1312 | (-0.0437, 0.3062) | 0.1057 | 0.1312 | (0.0000, 0.2917) | 0.0847 |
| controlled_vs_proposed_raw | bertscore_f1 | -0.0004 | (-0.0549, 0.0509) | 0.4970 | -0.0004 | (-0.0579, 0.0597) | 0.5020 |
| controlled_vs_proposed_raw | overall_quality | 0.1234 | (0.0695, 0.1720) | 0.0000 | 0.1234 | (0.0685, 0.1533) | 0.0003 |
| controlled_vs_candidate_no_context | context_relevance | 0.2396 | (0.2022, 0.2783) | 0.0000 | 0.2396 | (0.2152, 0.2908) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1067 | (0.0116, 0.1958) | 0.0130 | 0.1067 | (-0.0114, 0.2000) | 0.0310 |
| controlled_vs_candidate_no_context | naturalness | 0.0413 | (-0.0389, 0.1230) | 0.1637 | 0.0413 | (-0.0603, 0.1268) | 0.2513 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3144 | (0.2614, 0.3674) | 0.0000 | 0.3144 | (0.2795, 0.3766) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0650 | (0.0486, 0.0792) | 0.0000 | 0.0650 | (0.0470, 0.0847) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1333 | (0.0083, 0.2375) | 0.0140 | 0.1333 | (-0.0143, 0.2630) | 0.0307 |
| controlled_vs_candidate_no_context | persona_style | 0.0000 | (-0.0625, 0.0625) | 0.6520 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | distinct1 | 0.0022 | (-0.0346, 0.0377) | 0.4457 | 0.0022 | (-0.0396, 0.0381) | 0.4210 |
| controlled_vs_candidate_no_context | length_score | 0.1583 | (-0.1292, 0.4583) | 0.1493 | 0.1583 | (-0.2476, 0.4741) | 0.2270 |
| controlled_vs_candidate_no_context | sentence_score | 0.0875 | (-0.0875, 0.2625) | 0.2070 | 0.0875 | (0.0000, 0.2333) | 0.3270 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0136 | (-0.0262, 0.0512) | 0.2667 | 0.0136 | (-0.0324, 0.0628) | 0.2850 |
| controlled_vs_candidate_no_context | overall_quality | 0.1415 | (0.1084, 0.1739) | 0.0000 | 0.1415 | (0.1255, 0.1538) | 0.0000 |
| controlled_vs_baseline_no_context | context_relevance | 0.2540 | (0.2077, 0.3150) | 0.0000 | 0.2540 | (0.2080, 0.3345) | 0.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1117 | (0.0133, 0.1950) | 0.0117 | 0.1117 | (0.0333, 0.2095) | 0.0003 |
| controlled_vs_baseline_no_context | naturalness | -0.0089 | (-0.0800, 0.0652) | 0.5820 | -0.0089 | (-0.1044, 0.0727) | 0.5880 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3352 | (0.2727, 0.4091) | 0.0000 | 0.3352 | (0.2727, 0.4470) | 0.0000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0645 | (0.0518, 0.0754) | 0.0000 | 0.0645 | (0.0539, 0.0773) | 0.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1500 | (0.0333, 0.2542) | 0.0090 | 0.1500 | (0.0417, 0.2630) | 0.0000 |
| controlled_vs_baseline_no_context | persona_style | -0.0417 | (-0.1250, 0.0000) | 1.0000 | -0.0417 | (-0.1111, 0.0000) | 1.0000 |
| controlled_vs_baseline_no_context | distinct1 | -0.0374 | (-0.0851, 0.0112) | 0.9287 | -0.0374 | (-0.0832, 0.0095) | 0.9403 |
| controlled_vs_baseline_no_context | length_score | 0.0083 | (-0.3000, 0.2833) | 0.4743 | 0.0083 | (-0.4048, 0.3250) | 0.4630 |
| controlled_vs_baseline_no_context | sentence_score | 0.0437 | (-0.0875, 0.1750) | 0.3847 | 0.0437 | (-0.0875, 0.2003) | 0.3603 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0009 | (-0.0677, 0.0644) | 0.4803 | 0.0009 | (-0.0722, 0.0801) | 0.4497 |
| controlled_vs_baseline_no_context | overall_quality | 0.1392 | (0.0982, 0.1762) | 0.0000 | 0.1392 | (0.1064, 0.1884) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2540 | (0.2075, 0.3103) | 0.0000 | 0.2540 | (0.2084, 0.3367) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1117 | (0.0167, 0.1983) | 0.0097 | 0.1117 | (0.0333, 0.2095) | 0.0007 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0089 | (-0.0849, 0.0633) | 0.5860 | -0.0089 | (-0.1044, 0.0727) | 0.5663 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3352 | (0.2727, 0.4205) | 0.0000 | 0.3352 | (0.2727, 0.4470) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0645 | (0.0516, 0.0751) | 0.0000 | 0.0645 | (0.0542, 0.0772) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1500 | (0.0291, 0.2542) | 0.0117 | 0.1500 | (0.0407, 0.2630) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | -0.0417 | (-0.1250, 0.0000) | 1.0000 | -0.0417 | (-0.1111, 0.0000) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0374 | (-0.0857, 0.0159) | 0.9183 | -0.0374 | (-0.0832, 0.0147) | 0.9390 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.0083 | (-0.2917, 0.2917) | 0.4707 | 0.0083 | (-0.3833, 0.3250) | 0.4607 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0437 | (-0.0875, 0.1750) | 0.3877 | 0.0437 | (-0.1000, 0.2100) | 0.3697 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0009 | (-0.0642, 0.0673) | 0.4867 | 0.0009 | (-0.0734, 0.0801) | 0.4433 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1392 | (0.0984, 0.1799) | 0.0000 | 0.1392 | (0.1104, 0.1884) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 2 | 3 | 3 | 0.4375 | 0.4000 |
| proposed_vs_candidate_no_context | persona_consistency | 1 | 2 | 5 | 0.4375 | 0.3333 |
| proposed_vs_candidate_no_context | naturalness | 3 | 2 | 3 | 0.5625 | 0.6000 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 1 | 1 | 6 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | context_overlap | 3 | 2 | 3 | 0.5625 | 0.6000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 1 | 1 | 6 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | persona_style | 0 | 2 | 6 | 0.3750 | 0.0000 |
| proposed_vs_candidate_no_context | distinct1 | 4 | 1 | 3 | 0.6875 | 0.8000 |
| proposed_vs_candidate_no_context | length_score | 3 | 2 | 3 | 0.5625 | 0.6000 |
| proposed_vs_candidate_no_context | sentence_score | 0 | 1 | 7 | 0.4375 | 0.0000 |
| proposed_vs_candidate_no_context | bertscore_f1 | 4 | 1 | 3 | 0.6875 | 0.8000 |
| proposed_vs_candidate_no_context | overall_quality | 3 | 2 | 3 | 0.5625 | 0.6000 |
| proposed_vs_baseline_no_context | context_relevance | 5 | 3 | 0 | 0.6250 | 0.6250 |
| proposed_vs_baseline_no_context | persona_consistency | 1 | 4 | 3 | 0.3125 | 0.2000 |
| proposed_vs_baseline_no_context | naturalness | 3 | 5 | 0 | 0.3750 | 0.3750 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 3 | 2 | 3 | 0.5625 | 0.6000 |
| proposed_vs_baseline_no_context | context_overlap | 4 | 4 | 0 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 1 | 2 | 5 | 0.4375 | 0.3333 |
| proposed_vs_baseline_no_context | persona_style | 0 | 3 | 5 | 0.3125 | 0.0000 |
| proposed_vs_baseline_no_context | distinct1 | 2 | 5 | 1 | 0.3125 | 0.2857 |
| proposed_vs_baseline_no_context | length_score | 3 | 5 | 0 | 0.3750 | 0.3750 |
| proposed_vs_baseline_no_context | sentence_score | 0 | 2 | 6 | 0.3750 | 0.0000 |
| proposed_vs_baseline_no_context | bertscore_f1 | 3 | 5 | 0 | 0.3750 | 0.3750 |
| proposed_vs_baseline_no_context | overall_quality | 4 | 4 | 0 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | context_relevance | 7 | 1 | 0 | 0.8750 | 0.8750 |
| controlled_vs_proposed_raw | persona_consistency | 6 | 1 | 1 | 0.8125 | 0.8571 |
| controlled_vs_proposed_raw | naturalness | 4 | 4 | 0 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | context_keyword_coverage | 7 | 0 | 1 | 0.9375 | 1.0000 |
| controlled_vs_proposed_raw | context_overlap | 7 | 1 | 0 | 0.8750 | 0.8750 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 6 | 1 | 1 | 0.8125 | 0.8571 |
| controlled_vs_proposed_raw | persona_style | 2 | 0 | 6 | 0.6250 | 1.0000 |
| controlled_vs_proposed_raw | distinct1 | 4 | 4 | 0 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | length_score | 3 | 5 | 0 | 0.3750 | 0.3750 |
| controlled_vs_proposed_raw | sentence_score | 4 | 1 | 3 | 0.6875 | 0.8000 |
| controlled_vs_proposed_raw | bertscore_f1 | 5 | 3 | 0 | 0.6250 | 0.6250 |
| controlled_vs_proposed_raw | overall_quality | 7 | 1 | 0 | 0.8750 | 0.8750 |
| controlled_vs_candidate_no_context | context_relevance | 8 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 5 | 2 | 1 | 0.6875 | 0.7143 |
| controlled_vs_candidate_no_context | naturalness | 3 | 5 | 0 | 0.3750 | 0.3750 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 8 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 8 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 5 | 1 | 2 | 0.7500 | 0.8333 |
| controlled_vs_candidate_no_context | persona_style | 1 | 1 | 6 | 0.5000 | 0.5000 |
| controlled_vs_candidate_no_context | distinct1 | 3 | 4 | 1 | 0.4375 | 0.4286 |
| controlled_vs_candidate_no_context | length_score | 4 | 3 | 1 | 0.5625 | 0.5714 |
| controlled_vs_candidate_no_context | sentence_score | 3 | 1 | 4 | 0.6250 | 0.7500 |
| controlled_vs_candidate_no_context | bertscore_f1 | 5 | 3 | 0 | 0.6250 | 0.6250 |
| controlled_vs_candidate_no_context | overall_quality | 8 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | context_relevance | 8 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 6 | 1 | 1 | 0.8125 | 0.8571 |
| controlled_vs_baseline_no_context | naturalness | 4 | 4 | 0 | 0.5000 | 0.5000 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 8 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | context_overlap | 8 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 6 | 1 | 1 | 0.8125 | 0.8571 |
| controlled_vs_baseline_no_context | persona_style | 0 | 1 | 7 | 0.4375 | 0.0000 |
| controlled_vs_baseline_no_context | distinct1 | 2 | 5 | 1 | 0.3125 | 0.2857 |
| controlled_vs_baseline_no_context | length_score | 5 | 3 | 0 | 0.6250 | 0.6250 |
| controlled_vs_baseline_no_context | sentence_score | 2 | 1 | 5 | 0.5625 | 0.6667 |
| controlled_vs_baseline_no_context | bertscore_f1 | 4 | 4 | 0 | 0.5000 | 0.5000 |
| controlled_vs_baseline_no_context | overall_quality | 8 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 8 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 6 | 1 | 1 | 0.8125 | 0.8571 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 4 | 4 | 0 | 0.5000 | 0.5000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 8 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 8 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 6 | 1 | 1 | 0.8125 | 0.8571 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0 | 1 | 7 | 0.4375 | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | 2 | 5 | 1 | 0.3125 | 0.2857 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 5 | 3 | 0 | 0.6250 | 0.6250 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 2 | 1 | 5 | 0.5625 | 0.6667 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 4 | 4 | 0 | 0.5000 | 0.5000 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 8 | 0 | 0 | 1.0000 | 1.0000 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.2500 | 0.6250 | 0.3750 |
| proposed_contextual | 0.0000 | 0.0000 | 0.3750 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.3750 | 0.0000 | 0.0000 |
| baseline_no_context | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `5`
- Unique template signatures: `7`
- Template signature ratio: `0.8750`
- Effective sample size by source clustering: `4.57`
- Effective sample size by template-signature clustering: `6.40`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: enabled.

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
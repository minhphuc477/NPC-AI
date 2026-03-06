# Proposal Alignment Evaluation Report

- Run ID: `20260306T182550Z`
- Generated: `2026-03-06T18:27:32.430395+00:00`
- Scenarios: `artifacts\proposal_control_tuning\smoke_with_phi3latest\20260306T182550Z\scenarios.jsonl`
- Scenario count: `4`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_alt`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off
- `baseline_no_context`: model `phi3:mini`, dynamic_context=off, response_control=off
- `baseline_no_context_phi3_latest`: model `phi3:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2715 (0.2307, 0.2970) | 0.3323 (0.1967, 0.4679) | 0.8968 (0.8193, 0.9512) | 0.3853 (0.3432, 0.4275) | 0.0847 |
| proposed_contextual_controlled_alt | 0.2146 (0.1571, 0.2680) | 0.3007 (0.1943, 0.4321) | 0.8669 (0.7916, 0.9422) | 0.3476 (0.3142, 0.3980) | 0.0834 |
| proposed_contextual | 0.1089 (0.0079, 0.2796) | 0.1369 (0.0000, 0.2738) | 0.7747 (0.7486, 0.8009) | 0.2315 (0.1414, 0.3217) | 0.0641 |
| candidate_no_context | 0.0248 (0.0081, 0.0574) | 0.2589 (0.0500, 0.4679) | 0.8378 (0.7486, 0.9270) | 0.2470 (0.1549, 0.3390) | 0.0525 |
| baseline_no_context | 0.0280 (0.0124, 0.0566) | 0.1893 (0.0500, 0.3929) | 0.8748 (0.8163, 0.9333) | 0.2361 (0.1928, 0.2919) | 0.0848 |
| baseline_no_context_phi3_latest | 0.0421 (0.0115, 0.0727) | 0.2460 (0.1317, 0.3929) | 0.8617 (0.7935, 0.9298) | 0.2576 (0.2199, 0.3103) | 0.0853 |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0841 | 3.3944 |
| proposed_vs_candidate_no_context | persona_consistency | -0.1220 | -0.4713 |
| proposed_vs_candidate_no_context | naturalness | -0.0632 | -0.0754 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.1136 | 5.0000 |
| proposed_vs_candidate_no_context | context_overlap | 0.0152 | 0.5153 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.1369 | -0.6389 |
| proposed_vs_candidate_no_context | persona_style | -0.0625 | -0.1429 |
| proposed_vs_candidate_no_context | distinct1 | 0.0015 | 0.0016 |
| proposed_vs_candidate_no_context | length_score | -0.2750 | -0.6735 |
| proposed_vs_candidate_no_context | sentence_score | -0.0875 | -0.1061 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0116 | 0.2215 |
| proposed_vs_candidate_no_context | overall_quality | -0.0154 | -0.0625 |
| proposed_vs_baseline_no_context | context_relevance | 0.0809 | 2.8842 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0524 | -0.2767 |
| proposed_vs_baseline_no_context | naturalness | -0.1002 | -0.1145 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.1136 | 5.0000 |
| proposed_vs_baseline_no_context | context_overlap | 0.0044 | 0.1086 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0655 | -0.4583 |
| proposed_vs_baseline_no_context | persona_style | 0.0000 | 0.0000 |
| proposed_vs_baseline_no_context | distinct1 | -0.0535 | -0.0541 |
| proposed_vs_baseline_no_context | length_score | -0.3500 | -0.7241 |
| proposed_vs_baseline_no_context | sentence_score | -0.0875 | -0.1061 |
| proposed_vs_baseline_no_context | bertscore_f1 | -0.0206 | -0.2436 |
| proposed_vs_baseline_no_context | overall_quality | -0.0046 | -0.0194 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 0.0668 | 1.5887 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | -0.1090 | -0.4434 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | -0.0870 | -0.1010 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.0909 | 2.0000 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 0.0106 | 0.3117 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | -0.1155 | -0.5988 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | -0.0833 | -0.1818 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | -0.0362 | -0.0373 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | -0.2750 | -0.6735 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | -0.1750 | -0.1918 |
| proposed_vs_baseline_no_context_phi3_latest | bertscore_f1 | -0.0212 | -0.2486 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | -0.0261 | -0.1013 |
| controlled_vs_proposed_raw | context_relevance | 0.1626 | 1.4935 |
| controlled_vs_proposed_raw | persona_consistency | 0.1954 | 1.4270 |
| controlled_vs_proposed_raw | naturalness | 0.1221 | 0.1577 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2045 | 1.5000 |
| controlled_vs_proposed_raw | context_overlap | 0.0649 | 1.4476 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2286 | 2.9538 |
| controlled_vs_proposed_raw | persona_style | 0.0625 | 0.1667 |
| controlled_vs_proposed_raw | distinct1 | 0.0158 | 0.0169 |
| controlled_vs_proposed_raw | length_score | 0.4917 | 3.6875 |
| controlled_vs_proposed_raw | sentence_score | 0.1750 | 0.2373 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0206 | 0.3214 |
| controlled_vs_proposed_raw | overall_quality | 0.1538 | 0.6642 |
| controlled_vs_candidate_no_context | context_relevance | 0.2468 | 9.9576 |
| controlled_vs_candidate_no_context | persona_consistency | 0.0733 | 0.2832 |
| controlled_vs_candidate_no_context | naturalness | 0.0590 | 0.0704 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3182 | 14.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0801 | 2.7088 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.0917 | 0.4278 |
| controlled_vs_candidate_no_context | persona_style | 0.0000 | 0.0000 |
| controlled_vs_candidate_no_context | distinct1 | 0.0173 | 0.0185 |
| controlled_vs_candidate_no_context | length_score | 0.2167 | 0.5306 |
| controlled_vs_candidate_no_context | sentence_score | 0.0875 | 0.1061 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0322 | 0.6141 |
| controlled_vs_candidate_no_context | overall_quality | 0.1384 | 0.5602 |
| controlled_vs_baseline_no_context | context_relevance | 0.2435 | 8.6854 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1430 | 0.7553 |
| controlled_vs_baseline_no_context | naturalness | 0.0220 | 0.0251 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3182 | 14.0000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0693 | 1.7134 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1631 | 1.1417 |
| controlled_vs_baseline_no_context | persona_style | 0.0625 | 0.1667 |
| controlled_vs_baseline_no_context | distinct1 | -0.0377 | -0.0382 |
| controlled_vs_baseline_no_context | length_score | 0.1417 | 0.2931 |
| controlled_vs_baseline_no_context | sentence_score | 0.0875 | 0.1061 |
| controlled_vs_baseline_no_context | bertscore_f1 | -0.0000 | -0.0005 |
| controlled_vs_baseline_no_context | overall_quality | 0.1492 | 0.6320 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.2295 | 5.4549 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.0863 | 0.3509 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | 0.0352 | 0.0408 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.2955 | 6.5000 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0755 | 2.2105 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1131 | 0.5864 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | -0.0208 | -0.0455 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0205 | -0.0210 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | 0.2167 | 0.5306 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.0000 | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | -0.0006 | -0.0070 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1277 | 0.4956 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0569 | -0.2096 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0315 | -0.0949 |
| controlled_alt_vs_controlled_default | naturalness | -0.0299 | -0.0333 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0682 | -0.2000 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0306 | -0.2791 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0238 | -0.0778 |
| controlled_alt_vs_controlled_default | persona_style | -0.0625 | -0.1429 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0123 | -0.0129 |
| controlled_alt_vs_controlled_default | length_score | -0.1250 | -0.2000 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0000 | 0.0000 |
| controlled_alt_vs_controlled_default | bertscore_f1 | -0.0014 | -0.0160 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0378 | -0.0980 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1057 | 0.9709 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1638 | 1.1965 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0922 | 0.1191 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.1364 | 1.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0343 | 0.7644 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.2048 | 2.6462 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0000 | 0.0000 |
| controlled_alt_vs_proposed_raw | distinct1 | 0.0035 | 0.0038 |
| controlled_alt_vs_proposed_raw | length_score | 0.3667 | 2.7500 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1750 | 0.2373 |
| controlled_alt_vs_proposed_raw | bertscore_f1 | 0.0192 | 0.3002 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1160 | 0.5010 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.1898 | 7.6610 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.0418 | 0.1614 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0291 | 0.0347 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.2500 | 11.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0495 | 1.6736 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.0679 | 0.3167 |
| controlled_alt_vs_candidate_no_context | persona_style | -0.0625 | -0.1429 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0050 | 0.0054 |
| controlled_alt_vs_candidate_no_context | length_score | 0.0917 | 0.2245 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.0875 | 0.1061 |
| controlled_alt_vs_candidate_no_context | bertscore_f1 | 0.0309 | 0.5882 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1006 | 0.4072 |
| controlled_alt_vs_baseline_no_context | context_relevance | 0.1866 | 6.6554 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 0.1114 | 0.5887 |
| controlled_alt_vs_baseline_no_context | naturalness | -0.0079 | -0.0090 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 0.2500 | 11.0000 |
| controlled_alt_vs_baseline_no_context | context_overlap | 0.0386 | 0.9560 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 0.1393 | 0.9750 |
| controlled_alt_vs_baseline_no_context | persona_style | 0.0000 | 0.0000 |
| controlled_alt_vs_baseline_no_context | distinct1 | -0.0500 | -0.0505 |
| controlled_alt_vs_baseline_no_context | length_score | 0.0167 | 0.0345 |
| controlled_alt_vs_baseline_no_context | sentence_score | 0.0875 | 0.1061 |
| controlled_alt_vs_baseline_no_context | bertscore_f1 | -0.0014 | -0.0165 |
| controlled_alt_vs_baseline_no_context | overall_quality | 0.1114 | 0.4720 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_relevance | 0.1726 | 4.1020 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_consistency | 0.0548 | 0.2227 |
| controlled_alt_vs_baseline_no_context_phi3_latest | naturalness | 0.0053 | 0.0061 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.2273 | 5.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_overlap | 0.0449 | 1.3144 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.0893 | 0.4630 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_style | -0.0833 | -0.1818 |
| controlled_alt_vs_baseline_no_context_phi3_latest | distinct1 | -0.0327 | -0.0337 |
| controlled_alt_vs_baseline_no_context_phi3_latest | length_score | 0.0917 | 0.2245 |
| controlled_alt_vs_baseline_no_context_phi3_latest | sentence_score | 0.0000 | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | bertscore_f1 | -0.0020 | -0.0230 |
| controlled_alt_vs_baseline_no_context_phi3_latest | overall_quality | 0.0899 | 0.3490 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2435 | 8.6854 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1430 | 0.7553 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 0.0220 | 0.0251 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3182 | 14.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0693 | 1.7134 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1631 | 1.1417 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0625 | 0.1667 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0377 | -0.0382 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.1417 | 0.2931 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0875 | 0.1061 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | -0.0000 | -0.0005 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1492 | 0.6320 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.2295 | 5.4549 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.0863 | 0.3509 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | 0.0352 | 0.0408 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.2955 | 6.5000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0755 | 2.2105 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1131 | 0.5864 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | -0.0208 | -0.0455 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0205 | -0.0210 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | 0.2167 | 0.5306 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.0000 | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | -0.0006 | -0.0070 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1277 | 0.4956 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0841 | (-0.0025, 0.2551) | 0.3113 | 0.0841 | (-0.0025, 0.2551) | 0.3310 |
| proposed_vs_candidate_no_context | persona_consistency | -0.1220 | (-0.3804, 0.0714) | 0.8170 | -0.1220 | (-0.3804, 0.0714) | 0.8027 |
| proposed_vs_candidate_no_context | naturalness | -0.0632 | (-0.1435, 0.0172) | 0.9377 | -0.0632 | (-0.1435, 0.0172) | 0.9430 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.1136 | (0.0000, 0.3409) | 0.3160 | 0.1136 | (0.0000, 0.3409) | 0.3193 |
| proposed_vs_candidate_no_context | context_overlap | 0.0152 | (-0.0084, 0.0548) | 0.3253 | 0.0152 | (-0.0084, 0.0548) | 0.3373 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.1369 | (-0.4286, 0.0893) | 0.8180 | -0.1369 | (-0.4286, 0.0893) | 0.8167 |
| proposed_vs_candidate_no_context | persona_style | -0.0625 | (-0.1875, 0.0000) | 1.0000 | -0.0625 | (-0.1875, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | distinct1 | 0.0015 | (-0.0070, 0.0114) | 0.4330 | 0.0015 | (-0.0070, 0.0114) | 0.4083 |
| proposed_vs_candidate_no_context | length_score | -0.2750 | (-0.5333, -0.0167) | 1.0000 | -0.2750 | (-0.5333, -0.0167) | 1.0000 |
| proposed_vs_candidate_no_context | sentence_score | -0.0875 | (-0.3500, 0.1750) | 0.8127 | -0.0875 | (-0.3500, 0.1750) | 0.8170 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0116 | (-0.0503, 0.0678) | 0.3467 | 0.0116 | (-0.0503, 0.0676) | 0.3630 |
| proposed_vs_candidate_no_context | overall_quality | -0.0154 | (-0.1308, 0.1313) | 0.5650 | -0.0154 | (-0.1407, 0.1313) | 0.5827 |
| proposed_vs_baseline_no_context | context_relevance | 0.0809 | (-0.0064, 0.2516) | 0.3297 | 0.0809 | (-0.0064, 0.2516) | 0.3053 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0524 | (-0.3143, 0.1239) | 0.6817 | -0.0524 | (-0.3143, 0.1238) | 0.6847 |
| proposed_vs_baseline_no_context | naturalness | -0.1002 | (-0.1733, -0.0092) | 0.9827 | -0.1002 | (-0.1733, -0.0092) | 0.9837 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.1136 | (0.0000, 0.3409) | 0.3247 | 0.1136 | (0.0000, 0.3409) | 0.3117 |
| proposed_vs_baseline_no_context | context_overlap | 0.0044 | (-0.0212, 0.0430) | 0.3583 | 0.0044 | (-0.0212, 0.0430) | 0.3767 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0655 | (-0.3929, 0.1548) | 0.6963 | -0.0655 | (-0.3929, 0.1548) | 0.6800 |
| proposed_vs_baseline_no_context | persona_style | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_baseline_no_context | distinct1 | -0.0535 | (-0.0833, -0.0189) | 1.0000 | -0.0535 | (-0.0833, -0.0189) | 1.0000 |
| proposed_vs_baseline_no_context | length_score | -0.3500 | (-0.5833, -0.0083) | 0.9833 | -0.3500 | (-0.5833, -0.0083) | 0.9853 |
| proposed_vs_baseline_no_context | sentence_score | -0.0875 | (-0.2625, 0.0000) | 1.0000 | -0.0875 | (-0.2625, 0.0000) | 1.0000 |
| proposed_vs_baseline_no_context | bertscore_f1 | -0.0206 | (-0.0597, 0.0184) | 0.8730 | -0.0206 | (-0.0597, 0.0184) | 0.8630 |
| proposed_vs_baseline_no_context | overall_quality | -0.0046 | (-0.1243, 0.1108) | 0.5843 | -0.0046 | (-0.1243, 0.1108) | 0.5663 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 0.0668 | (-0.0499, 0.2523) | 0.3130 | 0.0668 | (-0.0499, 0.2523) | 0.3077 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | -0.1090 | (-0.3419, 0.1238) | 0.8180 | -0.1090 | (-0.3419, 0.1238) | 0.8003 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | -0.0870 | (-0.1639, -0.0101) | 0.9803 | -0.0870 | (-0.1639, -0.0101) | 0.9823 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.0909 | (-0.0682, 0.3409) | 0.3130 | 0.0909 | (-0.0682, 0.3409) | 0.3193 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 0.0106 | (-0.0095, 0.0457) | 0.3237 | 0.0106 | (-0.0095, 0.0457) | 0.3183 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | -0.1155 | (-0.3929, 0.1548) | 0.8123 | -0.1155 | (-0.3929, 0.1548) | 0.8113 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | -0.0833 | (-0.2500, 0.0000) | 1.0000 | -0.0833 | (-0.2500, 0.0000) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | -0.0362 | (-0.0714, -0.0064) | 1.0000 | -0.0362 | (-0.0714, -0.0064) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | -0.2750 | (-0.6000, 0.0500) | 0.9373 | -0.2750 | (-0.6000, 0.0500) | 0.9460 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | -0.1750 | (-0.3500, 0.0000) | 1.0000 | -0.1750 | (-0.3500, 0.0000) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | bertscore_f1 | -0.0212 | (-0.0607, 0.0182) | 0.8567 | -0.0212 | (-0.0607, 0.0182) | 0.8567 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | -0.0261 | (-0.1472, 0.0950) | 0.6393 | -0.0261 | (-0.1472, 0.0950) | 0.6563 |
| controlled_vs_proposed_raw | context_relevance | 0.1626 | (0.0061, 0.2708) | 0.0033 | 0.1626 | (0.0061, 0.2708) | 0.0010 |
| controlled_vs_proposed_raw | persona_consistency | 0.1954 | (0.0400, 0.4089) | 0.0033 | 0.1954 | (0.0400, 0.4089) | 0.0030 |
| controlled_vs_proposed_raw | naturalness | 0.1221 | (0.0589, 0.1854) | 0.0000 | 0.1221 | (0.0589, 0.1854) | 0.0000 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2045 | (0.0000, 0.3409) | 0.0027 | 0.2045 | (0.0000, 0.3409) | 0.0050 |
| controlled_vs_proposed_raw | context_overlap | 0.0649 | (0.0188, 0.1109) | 0.0027 | 0.0649 | (0.0188, 0.1109) | 0.0027 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2286 | (0.0500, 0.4643) | 0.0033 | 0.2286 | (0.0357, 0.4643) | 0.0053 |
| controlled_vs_proposed_raw | persona_style | 0.0625 | (0.0000, 0.1875) | 0.3130 | 0.0625 | (0.0000, 0.1875) | 0.3133 |
| controlled_vs_proposed_raw | distinct1 | 0.0158 | (-0.0062, 0.0378) | 0.0777 | 0.0158 | (-0.0062, 0.0378) | 0.0827 |
| controlled_vs_proposed_raw | length_score | 0.4917 | (0.1250, 0.8167) | 0.0037 | 0.4917 | (0.1250, 0.8167) | 0.0030 |
| controlled_vs_proposed_raw | sentence_score | 0.1750 | (0.0000, 0.3500) | 0.0567 | 0.1750 | (0.0000, 0.3500) | 0.0660 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0206 | (-0.0151, 0.0526) | 0.1447 | 0.0206 | (-0.0151, 0.0526) | 0.1380 |
| controlled_vs_proposed_raw | overall_quality | 0.1538 | (0.0428, 0.2577) | 0.0030 | 0.1538 | (0.0428, 0.2577) | 0.0050 |
| controlled_vs_candidate_no_context | context_relevance | 0.2468 | (0.2094, 0.2842) | 0.0000 | 0.2468 | (0.2094, 0.2842) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.0733 | (0.0000, 0.1467) | 0.0507 | 0.0733 | (0.0000, 0.1467) | 0.0610 |
| controlled_vs_candidate_no_context | naturalness | 0.0590 | (-0.0846, 0.2026) | 0.2650 | 0.0590 | (-0.0846, 0.2026) | 0.2900 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3182 | (0.2727, 0.3636) | 0.0000 | 0.3182 | (0.2727, 0.3636) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0801 | (0.0515, 0.1133) | 0.0000 | 0.0801 | (0.0515, 0.1133) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.0917 | (0.0000, 0.1833) | 0.0580 | 0.0917 | (0.0000, 0.1833) | 0.0663 |
| controlled_vs_candidate_no_context | persona_style | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | distinct1 | 0.0173 | (0.0013, 0.0332) | 0.0037 | 0.0173 | (0.0013, 0.0332) | 0.0037 |
| controlled_vs_candidate_no_context | length_score | 0.2167 | (-0.3667, 0.8000) | 0.2940 | 0.2167 | (-0.3667, 0.8000) | 0.2747 |
| controlled_vs_candidate_no_context | sentence_score | 0.0875 | (-0.1750, 0.3500) | 0.3843 | 0.0875 | (-0.1750, 0.3500) | 0.3870 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0322 | (-0.0372, 0.0860) | 0.1440 | 0.0322 | (-0.0372, 0.0860) | 0.1433 |
| controlled_vs_candidate_no_context | overall_quality | 0.1384 | (0.0884, 0.1883) | 0.0000 | 0.1384 | (0.0884, 0.1883) | 0.0000 |
| controlled_vs_baseline_no_context | context_relevance | 0.2435 | (0.2065, 0.2805) | 0.0000 | 0.2435 | (0.2065, 0.2805) | 0.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1430 | (0.0775, 0.2048) | 0.0000 | 0.1430 | (0.0775, 0.2048) | 0.0000 |
| controlled_vs_baseline_no_context | naturalness | 0.0220 | (-0.0464, 0.0944) | 0.2637 | 0.0220 | (-0.0464, 0.0944) | 0.2687 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3182 | (0.2727, 0.3636) | 0.0000 | 0.3182 | (0.2727, 0.3636) | 0.0000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0693 | (0.0331, 0.1022) | 0.0000 | 0.0693 | (0.0331, 0.1022) | 0.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1631 | (0.0500, 0.2560) | 0.0050 | 0.1631 | (0.0500, 0.2560) | 0.0023 |
| controlled_vs_baseline_no_context | persona_style | 0.0625 | (0.0000, 0.1875) | 0.3183 | 0.0625 | (0.0000, 0.1875) | 0.3147 |
| controlled_vs_baseline_no_context | distinct1 | -0.0377 | (-0.0719, -0.0018) | 1.0000 | -0.0377 | (-0.0719, -0.0036) | 1.0000 |
| controlled_vs_baseline_no_context | length_score | 0.1417 | (-0.3083, 0.5000) | 0.2497 | 0.1417 | (-0.3083, 0.5000) | 0.2287 |
| controlled_vs_baseline_no_context | sentence_score | 0.0875 | (0.0000, 0.2625) | 0.3133 | 0.0875 | (0.0000, 0.2625) | 0.3177 |
| controlled_vs_baseline_no_context | bertscore_f1 | -0.0000 | (-0.0374, 0.0365) | 0.5423 | -0.0000 | (-0.0374, 0.0365) | 0.5540 |
| controlled_vs_baseline_no_context | overall_quality | 0.1492 | (0.1281, 0.1695) | 0.0000 | 0.1492 | (0.1281, 0.1695) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.2295 | (0.1709, 0.2814) | 0.0000 | 0.2295 | (0.1709, 0.2814) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.0863 | (-0.0167, 0.1839) | 0.0673 | 0.0863 | (-0.0167, 0.1839) | 0.0517 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | 0.0352 | (-0.0869, 0.1412) | 0.3047 | 0.0352 | (-0.0869, 0.1412) | 0.2887 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.2955 | (0.2273, 0.3636) | 0.0000 | 0.2955 | (0.2273, 0.3636) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0755 | (0.0463, 0.1048) | 0.0000 | 0.0755 | (0.0463, 0.1048) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1131 | (0.0000, 0.2262) | 0.0643 | 0.1131 | (0.0000, 0.2262) | 0.0637 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | -0.0208 | (-0.2500, 0.1875) | 0.6447 | -0.0208 | (-0.2500, 0.1875) | 0.6480 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0205 | (-0.0522, 0.0113) | 0.9170 | -0.0205 | (-0.0522, 0.0113) | 0.9187 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | 0.2167 | (-0.3583, 0.6833) | 0.2067 | 0.2167 | (-0.3583, 0.6833) | 0.1973 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | -0.0006 | (-0.0475, 0.0463) | 0.5323 | -0.0006 | (-0.0475, 0.0463) | 0.5423 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1277 | (0.0906, 0.1648) | 0.0000 | 0.1277 | (0.0906, 0.1648) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0569 | (-0.1163, 0.0012) | 0.9447 | -0.0569 | (-0.1163, 0.0012) | 0.9410 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0315 | (-0.2375, 0.1714) | 0.6507 | -0.0315 | (-0.2089, 0.1714) | 0.6600 |
| controlled_alt_vs_controlled_default | naturalness | -0.0299 | (-0.1299, 0.0835) | 0.7053 | -0.0299 | (-0.1299, 0.0835) | 0.6917 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0682 | (-0.1364, 0.0000) | 1.0000 | -0.0682 | (-0.1364, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0306 | (-0.0695, 0.0041) | 0.9447 | -0.0306 | (-0.0695, 0.0041) | 0.9467 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0238 | (-0.2143, 0.2143) | 0.5770 | -0.0238 | (-0.2143, 0.2143) | 0.5837 |
| controlled_alt_vs_controlled_default | persona_style | -0.0625 | (-0.1875, 0.0000) | 1.0000 | -0.0625 | (-0.1875, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0123 | (-0.0247, 0.0000) | 1.0000 | -0.0123 | (-0.0247, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | length_score | -0.1250 | (-0.6167, 0.4417) | 0.7137 | -0.1250 | (-0.6167, 0.4417) | 0.7007 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | bertscore_f1 | -0.0014 | (-0.0368, 0.0340) | 0.6283 | -0.0014 | (-0.0368, 0.0340) | 0.6290 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0378 | (-0.1045, 0.0290) | 0.8203 | -0.0378 | (-0.1045, 0.0290) | 0.8280 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1057 | (0.0006, 0.2108) | 0.0213 | 0.1057 | (0.0006, 0.2108) | 0.0207 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1638 | (0.0571, 0.2476) | 0.0030 | 0.1638 | (0.0571, 0.2476) | 0.0030 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0922 | (-0.0091, 0.1936) | 0.0593 | 0.0922 | (-0.0091, 0.1936) | 0.0570 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.1364 | (0.0000, 0.2727) | 0.0193 | 0.1364 | (0.0000, 0.2727) | 0.0170 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0343 | (0.0021, 0.0696) | 0.0033 | 0.0343 | (0.0021, 0.0696) | 0.0027 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.2048 | (0.0714, 0.3095) | 0.0050 | 0.2048 | (0.0714, 0.3095) | 0.0040 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_proposed_raw | distinct1 | 0.0035 | (-0.0280, 0.0298) | 0.3927 | 0.0035 | (-0.0280, 0.0298) | 0.3687 |
| controlled_alt_vs_proposed_raw | length_score | 0.3667 | (0.0000, 0.7333) | 0.0607 | 0.3667 | (0.0000, 0.7333) | 0.0660 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1750 | (0.0000, 0.3500) | 0.0680 | 0.1750 | (0.0000, 0.3500) | 0.0640 |
| controlled_alt_vs_proposed_raw | bertscore_f1 | 0.0192 | (-0.0170, 0.0475) | 0.1163 | 0.0192 | (-0.0170, 0.0475) | 0.1257 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1160 | (0.0441, 0.1879) | 0.0000 | 0.1160 | (0.0441, 0.1879) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.1898 | (0.0997, 0.2593) | 0.0000 | 0.1898 | (0.0997, 0.2593) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.0418 | (-0.1964, 0.2800) | 0.3690 | 0.0418 | (-0.1964, 0.2800) | 0.3880 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0291 | (-0.0998, 0.1659) | 0.3677 | 0.0291 | (-0.0998, 0.1659) | 0.3743 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.2500 | (0.1364, 0.3409) | 0.0000 | 0.2500 | (0.1364, 0.3409) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0495 | (0.0111, 0.0838) | 0.0047 | 0.0495 | (0.0111, 0.0838) | 0.0037 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.0679 | (-0.2143, 0.3500) | 0.3747 | 0.0679 | (-0.2143, 0.3500) | 0.3510 |
| controlled_alt_vs_candidate_no_context | persona_style | -0.0625 | (-0.1875, 0.0000) | 1.0000 | -0.0625 | (-0.1875, 0.0000) | 1.0000 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0050 | (-0.0190, 0.0265) | 0.3537 | 0.0050 | (-0.0190, 0.0265) | 0.3437 |
| controlled_alt_vs_candidate_no_context | length_score | 0.0917 | (-0.3750, 0.6167) | 0.3730 | 0.0917 | (-0.3750, 0.6167) | 0.3547 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.0875 | (-0.1750, 0.3500) | 0.3617 | 0.0875 | (-0.1750, 0.3500) | 0.3720 |
| controlled_alt_vs_candidate_no_context | bertscore_f1 | 0.0309 | (-0.0037, 0.0537) | 0.0483 | 0.0309 | (-0.0037, 0.0537) | 0.0547 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1006 | (-0.0161, 0.2172) | 0.0623 | 0.1006 | (-0.0161, 0.2172) | 0.0573 |
| controlled_alt_vs_baseline_no_context | context_relevance | 0.1866 | (0.1006, 0.2547) | 0.0000 | 0.1866 | (0.1006, 0.2547) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 0.1114 | (-0.1314, 0.3286) | 0.1447 | 0.1114 | (-0.1314, 0.3286) | 0.1390 |
| controlled_alt_vs_baseline_no_context | naturalness | -0.0079 | (-0.1080, 0.0587) | 0.5630 | -0.0079 | (-0.1080, 0.0587) | 0.5630 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 0.2500 | (0.1364, 0.3409) | 0.0000 | 0.2500 | (0.1364, 0.3409) | 0.0000 |
| controlled_alt_vs_baseline_no_context | context_overlap | 0.0386 | (0.0057, 0.0716) | 0.0027 | 0.0386 | (0.0057, 0.0716) | 0.0027 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 0.1393 | (-0.1643, 0.4107) | 0.1350 | 0.1393 | (-0.1643, 0.4107) | 0.1577 |
| controlled_alt_vs_baseline_no_context | persona_style | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_baseline_no_context | distinct1 | -0.0500 | (-0.0967, -0.0116) | 1.0000 | -0.0500 | (-0.0967, -0.0116) | 1.0000 |
| controlled_alt_vs_baseline_no_context | length_score | 0.0167 | (-0.3500, 0.2167) | 0.2720 | 0.0167 | (-0.3500, 0.2167) | 0.2737 |
| controlled_alt_vs_baseline_no_context | sentence_score | 0.0875 | (0.0000, 0.2625) | 0.3067 | 0.0875 | (0.0000, 0.2625) | 0.3150 |
| controlled_alt_vs_baseline_no_context | bertscore_f1 | -0.0014 | (-0.0304, 0.0358) | 0.6130 | -0.0014 | (-0.0304, 0.0358) | 0.6157 |
| controlled_alt_vs_baseline_no_context | overall_quality | 0.1114 | (0.0438, 0.1790) | 0.0000 | 0.1114 | (0.0438, 0.1790) | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_relevance | 0.1726 | (0.1000, 0.2451) | 0.0000 | 0.1726 | (0.1000, 0.2451) | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_consistency | 0.0548 | (-0.1476, 0.2833) | 0.3210 | 0.0548 | (-0.1476, 0.2833) | 0.3317 |
| controlled_alt_vs_baseline_no_context_phi3_latest | naturalness | 0.0053 | (-0.1144, 0.1234) | 0.4427 | 0.0053 | (-0.1144, 0.1234) | 0.4153 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.2273 | (0.1364, 0.3182) | 0.0000 | 0.2273 | (0.1364, 0.3182) | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_overlap | 0.0449 | (0.0151, 0.0747) | 0.0040 | 0.0449 | (0.0151, 0.0747) | 0.0040 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.0893 | (-0.1786, 0.3750) | 0.2973 | 0.0893 | (-0.1786, 0.3750) | 0.3133 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_style | -0.0833 | (-0.2500, 0.0000) | 1.0000 | -0.0833 | (-0.2500, 0.0000) | 1.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | distinct1 | -0.0327 | (-0.0767, 0.0113) | 0.9227 | -0.0327 | (-0.0767, 0.0113) | 0.9257 |
| controlled_alt_vs_baseline_no_context_phi3_latest | length_score | 0.0917 | (-0.4417, 0.6167) | 0.3590 | 0.0917 | (-0.4417, 0.6167) | 0.3660 |
| controlled_alt_vs_baseline_no_context_phi3_latest | sentence_score | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | bertscore_f1 | -0.0020 | (-0.0285, 0.0459) | 0.6870 | -0.0020 | (-0.0285, 0.0459) | 0.6670 |
| controlled_alt_vs_baseline_no_context_phi3_latest | overall_quality | 0.0899 | (0.0227, 0.1755) | 0.0000 | 0.0899 | (0.0227, 0.1755) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2435 | (0.2065, 0.2805) | 0.0000 | 0.2435 | (0.2065, 0.2805) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1430 | (0.0775, 0.2048) | 0.0000 | 0.1430 | (0.0775, 0.2048) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 0.0220 | (-0.0464, 0.0944) | 0.2753 | 0.0220 | (-0.0464, 0.0944) | 0.2697 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3182 | (0.2727, 0.3636) | 0.0000 | 0.3182 | (0.2727, 0.3636) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0693 | (0.0331, 0.1022) | 0.0000 | 0.0693 | (0.0331, 0.1022) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1631 | (0.0500, 0.2560) | 0.0030 | 0.1631 | (0.0500, 0.2560) | 0.0027 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0625 | (0.0000, 0.1875) | 0.3117 | 0.0625 | (0.0000, 0.1875) | 0.3007 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0377 | (-0.0719, -0.0036) | 1.0000 | -0.0377 | (-0.0719, -0.0036) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.1417 | (-0.3083, 0.5000) | 0.2423 | 0.1417 | (-0.3083, 0.5000) | 0.2460 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0875 | (0.0000, 0.2625) | 0.3183 | 0.0875 | (0.0000, 0.2625) | 0.3167 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | -0.0000 | (-0.0374, 0.0365) | 0.5353 | -0.0000 | (-0.0374, 0.0365) | 0.5463 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1492 | (0.1281, 0.1695) | 0.0000 | 0.1492 | (0.1281, 0.1695) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.2295 | (0.1709, 0.2814) | 0.0000 | 0.2295 | (0.1709, 0.2814) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.0863 | (-0.0167, 0.1839) | 0.0570 | 0.0863 | (-0.0167, 0.1839) | 0.0547 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | 0.0352 | (-0.0869, 0.1412) | 0.3010 | 0.0352 | (-0.0869, 0.1412) | 0.2983 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.2955 | (0.2273, 0.3636) | 0.0000 | 0.2955 | (0.2273, 0.3636) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0755 | (0.0463, 0.1048) | 0.0000 | 0.0755 | (0.0463, 0.1048) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1131 | (0.0000, 0.2262) | 0.0697 | 0.1131 | (0.0000, 0.2262) | 0.0610 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | -0.0208 | (-0.2500, 0.1875) | 0.6503 | -0.0208 | (-0.2500, 0.1875) | 0.6320 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0205 | (-0.0522, 0.0113) | 0.9003 | -0.0205 | (-0.0522, 0.0113) | 0.9137 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | 0.2167 | (-0.3583, 0.6833) | 0.2160 | 0.2167 | (-0.3583, 0.6833) | 0.2123 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | -0.0006 | (-0.0475, 0.0463) | 0.5477 | -0.0006 | (-0.0475, 0.0463) | 0.5393 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1277 | (0.0906, 0.1648) | 0.0000 | 0.1277 | (0.0906, 0.1648) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 1 | 2 | 1 | 0.3750 | 0.3333 |
| proposed_vs_candidate_no_context | persona_consistency | 1 | 2 | 1 | 0.3750 | 0.3333 |
| proposed_vs_candidate_no_context | naturalness | 1 | 2 | 1 | 0.3750 | 0.3333 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 1 | 0 | 3 | 0.6250 | 1.0000 |
| proposed_vs_candidate_no_context | context_overlap | 1 | 2 | 1 | 0.3750 | 0.3333 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 1 | 2 | 1 | 0.3750 | 0.3333 |
| proposed_vs_candidate_no_context | persona_style | 0 | 1 | 3 | 0.3750 | 0.0000 |
| proposed_vs_candidate_no_context | distinct1 | 1 | 1 | 2 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | length_score | 0 | 3 | 1 | 0.1250 | 0.0000 |
| proposed_vs_candidate_no_context | sentence_score | 1 | 2 | 1 | 0.3750 | 0.3333 |
| proposed_vs_candidate_no_context | bertscore_f1 | 2 | 1 | 1 | 0.6250 | 0.6667 |
| proposed_vs_candidate_no_context | overall_quality | 1 | 2 | 1 | 0.3750 | 0.3333 |
| proposed_vs_baseline_no_context | context_relevance | 1 | 3 | 0 | 0.2500 | 0.2500 |
| proposed_vs_baseline_no_context | persona_consistency | 2 | 1 | 1 | 0.6250 | 0.6667 |
| proposed_vs_baseline_no_context | naturalness | 1 | 3 | 0 | 0.2500 | 0.2500 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 1 | 0 | 3 | 0.6250 | 1.0000 |
| proposed_vs_baseline_no_context | context_overlap | 1 | 3 | 0 | 0.2500 | 0.2500 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 2 | 1 | 1 | 0.6250 | 0.6667 |
| proposed_vs_baseline_no_context | persona_style | 0 | 0 | 4 | 0.5000 | nan |
| proposed_vs_baseline_no_context | distinct1 | 0 | 3 | 1 | 0.1250 | 0.0000 |
| proposed_vs_baseline_no_context | length_score | 1 | 3 | 0 | 0.2500 | 0.2500 |
| proposed_vs_baseline_no_context | sentence_score | 0 | 1 | 3 | 0.3750 | 0.0000 |
| proposed_vs_baseline_no_context | bertscore_f1 | 2 | 2 | 0 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context | overall_quality | 2 | 2 | 0 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 1 | 3 | 0 | 0.2500 | 0.2500 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | 2 | 2 | 0 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | 1 | 3 | 0 | 0.2500 | 0.2500 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 1 | 1 | 2 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 1 | 3 | 0 | 0.2500 | 0.2500 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 2 | 2 | 0 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | 0 | 1 | 3 | 0.3750 | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | 0 | 3 | 1 | 0.1250 | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | 1 | 3 | 0 | 0.2500 | 0.2500 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | 0 | 2 | 2 | 0.2500 | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | bertscore_f1 | 2 | 2 | 0 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | 2 | 2 | 0 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | context_relevance | 3 | 1 | 0 | 0.7500 | 0.7500 |
| controlled_vs_proposed_raw | persona_consistency | 3 | 0 | 1 | 0.8750 | 1.0000 |
| controlled_vs_proposed_raw | naturalness | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_proposed_raw | context_keyword_coverage | 3 | 1 | 0 | 0.7500 | 0.7500 |
| controlled_vs_proposed_raw | context_overlap | 3 | 1 | 0 | 0.7500 | 0.7500 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 3 | 0 | 1 | 0.8750 | 1.0000 |
| controlled_vs_proposed_raw | persona_style | 1 | 0 | 3 | 0.6250 | 1.0000 |
| controlled_vs_proposed_raw | distinct1 | 2 | 1 | 1 | 0.6250 | 0.6667 |
| controlled_vs_proposed_raw | length_score | 3 | 1 | 0 | 0.7500 | 0.7500 |
| controlled_vs_proposed_raw | sentence_score | 2 | 0 | 2 | 0.7500 | 1.0000 |
| controlled_vs_proposed_raw | bertscore_f1 | 3 | 1 | 0 | 0.7500 | 0.7500 |
| controlled_vs_proposed_raw | overall_quality | 3 | 1 | 0 | 0.7500 | 0.7500 |
| controlled_vs_candidate_no_context | context_relevance | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 2 | 0 | 2 | 0.7500 | 1.0000 |
| controlled_vs_candidate_no_context | naturalness | 2 | 2 | 0 | 0.5000 | 0.5000 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 2 | 0 | 2 | 0.7500 | 1.0000 |
| controlled_vs_candidate_no_context | persona_style | 0 | 0 | 4 | 0.5000 | nan |
| controlled_vs_candidate_no_context | distinct1 | 3 | 0 | 1 | 0.8750 | 1.0000 |
| controlled_vs_candidate_no_context | length_score | 2 | 2 | 0 | 0.5000 | 0.5000 |
| controlled_vs_candidate_no_context | sentence_score | 2 | 1 | 1 | 0.6250 | 0.6667 |
| controlled_vs_candidate_no_context | bertscore_f1 | 3 | 1 | 0 | 0.7500 | 0.7500 |
| controlled_vs_candidate_no_context | overall_quality | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | context_relevance | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | naturalness | 3 | 1 | 0 | 0.7500 | 0.7500 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | context_overlap | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 3 | 0 | 1 | 0.8750 | 1.0000 |
| controlled_vs_baseline_no_context | persona_style | 1 | 0 | 3 | 0.6250 | 1.0000 |
| controlled_vs_baseline_no_context | distinct1 | 0 | 3 | 1 | 0.1250 | 0.0000 |
| controlled_vs_baseline_no_context | length_score | 3 | 1 | 0 | 0.7500 | 0.7500 |
| controlled_vs_baseline_no_context | sentence_score | 1 | 0 | 3 | 0.6250 | 1.0000 |
| controlled_vs_baseline_no_context | bertscore_f1 | 2 | 2 | 0 | 0.5000 | 0.5000 |
| controlled_vs_baseline_no_context | overall_quality | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 3 | 1 | 0 | 0.7500 | 0.7500 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | 3 | 1 | 0 | 0.7500 | 0.7500 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 2 | 0 | 2 | 0.7500 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | 1 | 1 | 2 | 0.5000 | 0.5000 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | 1 | 2 | 1 | 0.3750 | 0.3333 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | 3 | 1 | 0 | 0.7500 | 0.7500 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0 | 0 | 4 | 0.5000 | nan |
| controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 2 | 2 | 0 | 0.5000 | 0.5000 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 1 | 2 | 1 | 0.3750 | 0.3333 |
| controlled_alt_vs_controlled_default | persona_consistency | 1 | 2 | 1 | 0.3750 | 0.3333 |
| controlled_alt_vs_controlled_default | naturalness | 1 | 2 | 1 | 0.3750 | 0.3333 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0 | 2 | 2 | 0.2500 | 0.0000 |
| controlled_alt_vs_controlled_default | context_overlap | 1 | 2 | 1 | 0.3750 | 0.3333 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 1 | 2 | 1 | 0.3750 | 0.3333 |
| controlled_alt_vs_controlled_default | persona_style | 0 | 1 | 3 | 0.3750 | 0.0000 |
| controlled_alt_vs_controlled_default | distinct1 | 0 | 2 | 2 | 0.2500 | 0.0000 |
| controlled_alt_vs_controlled_default | length_score | 1 | 2 | 1 | 0.3750 | 0.3333 |
| controlled_alt_vs_controlled_default | sentence_score | 0 | 0 | 4 | 0.5000 | nan |
| controlled_alt_vs_controlled_default | bertscore_f1 | 1 | 2 | 1 | 0.3750 | 0.3333 |
| controlled_alt_vs_controlled_default | overall_quality | 1 | 2 | 1 | 0.3750 | 0.3333 |
| controlled_alt_vs_proposed_raw | context_relevance | 3 | 1 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_proposed_raw | persona_consistency | 3 | 0 | 1 | 0.8750 | 1.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 2 | 1 | 1 | 0.6250 | 0.6667 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 3 | 1 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_proposed_raw | context_overlap | 3 | 0 | 1 | 0.8750 | 1.0000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 3 | 0 | 1 | 0.8750 | 1.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 0 | 0 | 4 | 0.5000 | nan |
| controlled_alt_vs_proposed_raw | distinct1 | 2 | 1 | 1 | 0.6250 | 0.6667 |
| controlled_alt_vs_proposed_raw | length_score | 2 | 0 | 2 | 0.7500 | 1.0000 |
| controlled_alt_vs_proposed_raw | sentence_score | 2 | 0 | 2 | 0.7500 | 1.0000 |
| controlled_alt_vs_proposed_raw | bertscore_f1 | 3 | 1 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_proposed_raw | overall_quality | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 2 | 2 | 0 | 0.5000 | 0.5000 |
| controlled_alt_vs_candidate_no_context | naturalness | 3 | 1 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 3 | 1 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 2 | 2 | 0 | 0.5000 | 0.5000 |
| controlled_alt_vs_candidate_no_context | persona_style | 0 | 1 | 3 | 0.3750 | 0.0000 |
| controlled_alt_vs_candidate_no_context | distinct1 | 2 | 1 | 1 | 0.6250 | 0.6667 |
| controlled_alt_vs_candidate_no_context | length_score | 2 | 2 | 0 | 0.5000 | 0.5000 |
| controlled_alt_vs_candidate_no_context | sentence_score | 2 | 1 | 1 | 0.6250 | 0.6667 |
| controlled_alt_vs_candidate_no_context | bertscore_f1 | 3 | 1 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_candidate_no_context | overall_quality | 3 | 1 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_baseline_no_context | context_relevance | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 3 | 1 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_baseline_no_context | naturalness | 3 | 1 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context | context_overlap | 3 | 1 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 3 | 1 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_baseline_no_context | persona_style | 0 | 0 | 4 | 0.5000 | nan |
| controlled_alt_vs_baseline_no_context | distinct1 | 0 | 3 | 1 | 0.1250 | 0.0000 |
| controlled_alt_vs_baseline_no_context | length_score | 3 | 1 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_baseline_no_context | sentence_score | 1 | 0 | 3 | 0.6250 | 1.0000 |
| controlled_alt_vs_baseline_no_context | bertscore_f1 | 1 | 3 | 0 | 0.2500 | 0.2500 |
| controlled_alt_vs_baseline_no_context | overall_quality | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_relevance | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_consistency | 2 | 2 | 0 | 0.5000 | 0.5000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | naturalness | 2 | 2 | 0 | 0.5000 | 0.5000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_overlap | 3 | 1 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 2 | 1 | 1 | 0.6250 | 0.6667 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_style | 0 | 1 | 3 | 0.3750 | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | distinct1 | 1 | 2 | 1 | 0.3750 | 0.3333 |
| controlled_alt_vs_baseline_no_context_phi3_latest | length_score | 3 | 1 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_baseline_no_context_phi3_latest | sentence_score | 0 | 0 | 4 | 0.5000 | nan |
| controlled_alt_vs_baseline_no_context_phi3_latest | bertscore_f1 | 1 | 3 | 0 | 0.2500 | 0.2500 |
| controlled_alt_vs_baseline_no_context_phi3_latest | overall_quality | 4 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 4 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 4 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 3 | 1 | 0 | 0.7500 | 0.7500 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 4 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 4 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 3 | 0 | 1 | 0.8750 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 1 | 0 | 3 | 0.6250 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | 0 | 3 | 1 | 0.1250 | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 3 | 1 | 0 | 0.7500 | 0.7500 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 1 | 0 | 3 | 0.6250 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 2 | 2 | 0 | 0.5000 | 0.5000 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 4 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 4 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 3 | 1 | 0 | 0.7500 | 0.7500 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | 3 | 1 | 0 | 0.7500 | 0.7500 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 4 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 4 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 2 | 0 | 2 | 0.7500 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | 1 | 1 | 2 | 0.5000 | 0.5000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | 1 | 2 | 1 | 0.3750 | 0.3333 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | 3 | 1 | 0 | 0.7500 | 0.7500 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0 | 0 | 4 | 0.5000 | nan |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 2 | 2 | 0 | 0.5000 | 0.5000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 4 | 0 | 0 | 1.0000 | 1.0000 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.0000 |
| proposed_contextual_controlled_alt | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.0000 |
| proposed_contextual | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 |
| baseline_no_context | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline_no_context_phi3_latest | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `4`
- Unique template signatures: `4`
- Template signature ratio: `1.0000`
- Effective sample size by source clustering: `4.00`
- Effective sample size by template-signature clustering: `4.00`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: enabled.

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
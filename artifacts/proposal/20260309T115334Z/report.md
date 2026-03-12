# Proposal Alignment Evaluation Report

- Run ID: `20260309T115334Z`
- Generated: `2026-03-09T11:54:55.195958+00:00`
- Scenarios: `artifacts\proposal\20260309T115334Z\scenarios.jsonl`
- Scenario count: `4`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off
- `baseline_no_context`: model `phi3:mini`, dynamic_context=off, response_control=off
- `baseline_no_context_phi3_latest`: model `phi3:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2069 (0.1789, 0.2298) | 0.2995 (0.2333, 0.3798) | 0.8975 (0.8558, 0.9329) | 0.3765 (0.3488, 0.4148) | n/a |
| proposed_contextual | 0.0934 (0.0153, 0.2089) | 0.1833 (0.1250, 0.2250) | 0.9086 (0.8065, 0.9833) | 0.2819 (0.2315, 0.3267) | n/a |
| candidate_no_context | 0.0065 (0.0048, 0.0086) | 0.2167 (0.1250, 0.3250) | 0.8887 (0.7915, 0.9549) | 0.2501 (0.2140, 0.2920) | n/a |
| baseline_no_context | 0.0391 (0.0028, 0.0755) | 0.1500 (0.1000, 0.2000) | 0.9185 (0.8700, 0.9683) | 0.2466 (0.2050, 0.2882) | n/a |
| baseline_no_context_phi3_latest | 0.0266 (0.0050, 0.0621) | 0.1500 (0.1000, 0.2000) | 0.9150 (0.8933, 0.9367) | 0.2396 (0.2171, 0.2627) | n/a |

## Game-facing Outcome Metrics (mean, 95% CI)
| Arm | Quest-state Correctness | Lore Consistency | Contradiction Safety | Objective Completion Support | Gameplay Usefulness | Time-pressure Acceptability |
|---|---:|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2663 (0.2308, 0.3026) | 0.1251 (0.0650, 0.1853) | 1.0000 (1.0000, 1.0000) | 0.0862 (0.0000, 0.1975) | 0.3543 (0.3215, 0.4090) | 0.2944 (0.2500, 0.3795) |
| proposed_contextual | 0.1783 (0.1228, 0.2719) | 0.0262 (0.0000, 0.0694) | 1.0000 (1.0000, 1.0000) | 0.0335 (0.0000, 0.0671) | 0.3020 (0.2490, 0.3505) | 0.2834 (0.2500, 0.3168) |
| candidate_no_context | 0.1074 (0.1017, 0.1173) | 0.0030 (0.0000, 0.0091) | 1.0000 (1.0000, 1.0000) | 0.0198 (0.0000, 0.0594) | 0.2673 (0.2416, 0.2881) | 0.2879 (0.2607, 0.3102) |
| baseline_no_context | 0.1369 (0.1010, 0.1728) | 0.0183 (0.0000, 0.0474) | 1.0000 (1.0000, 1.0000) | 0.0544 (0.0104, 0.0879) | 0.2950 (0.2466, 0.3336) | 0.2928 (0.2594, 0.3261) |
| baseline_no_context_phi3_latest | 0.1237 (0.1018, 0.1649) | 0.0071 (0.0000, 0.0142) | 1.0000 (1.0000, 1.0000) | 0.0335 (0.0000, 0.0725) | 0.2839 (0.2789, 0.2890) | 0.2981 (0.2658, 0.3255) |

- Multi-turn contradiction rate is reported as `1 - contradiction_safety` in row-level outputs.

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0869 | 13.2747 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0333 | -0.1538 |
| proposed_vs_candidate_no_context | naturalness | 0.0199 | 0.0224 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.0709 | 0.6607 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0231 | 7.6663 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | 0.0137 | 0.6947 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 0.0347 | 0.1298 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | -0.0045 | -0.0157 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.1136 | nan |
| proposed_vs_candidate_no_context | context_overlap | 0.0244 | 1.1167 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0417 | -0.5000 |
| proposed_vs_candidate_no_context | persona_style | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | distinct1 | -0.0014 | -0.0014 |
| proposed_vs_candidate_no_context | length_score | 0.0583 | 0.0986 |
| proposed_vs_candidate_no_context | sentence_score | 0.0875 | 0.1061 |
| proposed_vs_candidate_no_context | overall_quality | 0.0318 | 0.1273 |
| proposed_vs_baseline_no_context | context_relevance | 0.0543 | 1.3861 |
| proposed_vs_baseline_no_context | persona_consistency | 0.0333 | 0.2222 |
| proposed_vs_baseline_no_context | naturalness | -0.0099 | -0.0108 |
| proposed_vs_baseline_no_context | quest_state_correctness | 0.0414 | 0.3021 |
| proposed_vs_baseline_no_context | lore_consistency | 0.0078 | 0.4277 |
| proposed_vs_baseline_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| proposed_vs_baseline_no_context | objective_completion_support | -0.0208 | -0.3831 |
| proposed_vs_baseline_no_context | gameplay_usefulness | 0.0070 | 0.0236 |
| proposed_vs_baseline_no_context | time_pressure_acceptability | -0.0094 | -0.0320 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0659 | 1.3810 |
| proposed_vs_baseline_no_context | context_overlap | 0.0271 | 1.4161 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 0.0417 | nan |
| proposed_vs_baseline_no_context | persona_style | 0.0000 | 0.0000 |
| proposed_vs_baseline_no_context | distinct1 | -0.0247 | -0.0249 |
| proposed_vs_baseline_no_context | length_score | 0.0000 | 0.0000 |
| proposed_vs_baseline_no_context | sentence_score | 0.0000 | 0.0000 |
| proposed_vs_baseline_no_context | overall_quality | 0.0353 | 0.1433 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 0.0668 | 2.5120 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | 0.0333 | 0.2222 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | -0.0064 | -0.0070 |
| proposed_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.0545 | 0.4408 |
| proposed_vs_baseline_no_context_phi3_latest | lore_consistency | 0.0191 | 2.6820 |
| proposed_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | objective_completion_support | -0.0000 | -0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 0.0180 | 0.0635 |
| proposed_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | -0.0147 | -0.0494 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.0886 | 3.5455 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 0.0159 | 0.5230 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.0417 | nan |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | 0.0000 | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | -0.0119 | -0.0122 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | -0.0083 | -0.0127 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | 0.0000 | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | 0.0424 | 0.1768 |
| controlled_vs_proposed_raw | context_relevance | 0.1135 | 1.2157 |
| controlled_vs_proposed_raw | persona_consistency | 0.1161 | 0.6334 |
| controlled_vs_proposed_raw | naturalness | -0.0111 | -0.0122 |
| controlled_vs_proposed_raw | quest_state_correctness | 0.0880 | 0.4938 |
| controlled_vs_proposed_raw | lore_consistency | 0.0990 | 3.7835 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0527 | 1.5714 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0523 | 0.1733 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 0.0110 | 0.0389 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.1477 | 1.3000 |
| controlled_vs_proposed_raw | context_overlap | 0.0338 | 0.7314 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1548 | 3.7143 |
| controlled_vs_proposed_raw | persona_style | -0.0385 | -0.0513 |
| controlled_vs_proposed_raw | distinct1 | -0.0146 | -0.0151 |
| controlled_vs_proposed_raw | length_score | 0.0250 | 0.0385 |
| controlled_vs_proposed_raw | sentence_score | -0.0875 | -0.0959 |
| controlled_vs_proposed_raw | overall_quality | 0.0946 | 0.3355 |
| controlled_vs_candidate_no_context | context_relevance | 0.2004 | 30.6284 |
| controlled_vs_candidate_no_context | persona_consistency | 0.0828 | 0.3821 |
| controlled_vs_candidate_no_context | naturalness | 0.0088 | 0.0099 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.1590 | 1.4808 |
| controlled_vs_candidate_no_context | lore_consistency | 0.1221 | 40.4550 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | 0.0665 | 3.3579 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0870 | 0.3256 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 0.0065 | 0.0226 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.2614 | nan |
| controlled_vs_candidate_no_context | context_overlap | 0.0581 | 2.6649 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1131 | 1.3571 |
| controlled_vs_candidate_no_context | persona_style | -0.0385 | -0.0513 |
| controlled_vs_candidate_no_context | distinct1 | -0.0160 | -0.0165 |
| controlled_vs_candidate_no_context | length_score | 0.0833 | 0.1408 |
| controlled_vs_candidate_no_context | sentence_score | 0.0000 | 0.0000 |
| controlled_vs_candidate_no_context | overall_quality | 0.1264 | 0.5055 |
| controlled_vs_baseline_no_context | context_relevance | 0.1678 | 4.2869 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1495 | 0.9963 |
| controlled_vs_baseline_no_context | naturalness | -0.0210 | -0.0228 |
| controlled_vs_baseline_no_context | quest_state_correctness | 0.1294 | 0.9451 |
| controlled_vs_baseline_no_context | lore_consistency | 0.1068 | 5.8293 |
| controlled_vs_baseline_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_baseline_no_context | objective_completion_support | 0.0319 | 0.5862 |
| controlled_vs_baseline_no_context | gameplay_usefulness | 0.0593 | 0.2010 |
| controlled_vs_baseline_no_context | time_pressure_acceptability | 0.0017 | 0.0057 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.2136 | 4.4762 |
| controlled_vs_baseline_no_context | context_overlap | 0.0608 | 3.1833 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1964 | nan |
| controlled_vs_baseline_no_context | persona_style | -0.0385 | -0.0513 |
| controlled_vs_baseline_no_context | distinct1 | -0.0393 | -0.0396 |
| controlled_vs_baseline_no_context | length_score | 0.0250 | 0.0385 |
| controlled_vs_baseline_no_context | sentence_score | -0.0875 | -0.0959 |
| controlled_vs_baseline_no_context | overall_quality | 0.1299 | 0.5268 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.1803 | 6.7814 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1495 | 0.9963 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | -0.0175 | -0.0192 |
| controlled_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.1426 | 1.1523 |
| controlled_vs_baseline_no_context_phi3_latest | lore_consistency | 0.1180 | 16.6129 |
| controlled_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.0527 | 1.5714 |
| controlled_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 0.0704 | 0.2479 |
| controlled_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | -0.0037 | -0.0124 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.2364 | 9.4545 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0496 | 1.6370 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1964 | nan |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | -0.0385 | -0.0513 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0265 | -0.0271 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | 0.0167 | 0.0253 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | -0.0875 | -0.0959 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1369 | 0.5716 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.1678 | 4.2869 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1495 | 0.9963 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0210 | -0.0228 |
| proposed_contextual_controlled_vs_baseline_no_context | quest_state_correctness | 0.1294 | 0.9451 |
| proposed_contextual_controlled_vs_baseline_no_context | lore_consistency | 0.1068 | 5.8293 |
| proposed_contextual_controlled_vs_baseline_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | objective_completion_support | 0.0319 | 0.5862 |
| proposed_contextual_controlled_vs_baseline_no_context | gameplay_usefulness | 0.0593 | 0.2010 |
| proposed_contextual_controlled_vs_baseline_no_context | time_pressure_acceptability | 0.0017 | 0.0057 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.2136 | 4.4762 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0608 | 3.1833 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1964 | nan |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | -0.0385 | -0.0513 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0393 | -0.0396 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.0250 | 0.0385 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | -0.0875 | -0.0959 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1299 | 0.5268 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.1803 | 6.7814 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1495 | 0.9963 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | -0.0175 | -0.0192 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.1426 | 1.1523 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | lore_consistency | 0.1180 | 16.6129 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.0527 | 1.5714 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 0.0704 | 0.2479 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | -0.0037 | -0.0124 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.2364 | 9.4545 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0496 | 1.6370 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1964 | nan |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | -0.0385 | -0.0513 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0265 | -0.0271 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | 0.0167 | 0.0253 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | -0.0875 | -0.0959 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1369 | 0.5716 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0869 | (0.0093, 0.2003) | 0.0040 | 0.0869 | (0.0000, 0.1397) | 0.0337 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0333 | (-0.2000, 0.1000) | 0.7307 | -0.0333 | (-0.0667, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | naturalness | 0.0199 | (-0.0002, 0.0500) | 0.0633 | 0.0199 | (0.0000, 0.0331) | 0.0300 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.0709 | (0.0107, 0.1688) | 0.0027 | 0.0709 | (0.0000, 0.1196) | 0.0340 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0231 | (0.0000, 0.0694) | 0.3267 | 0.0231 | (0.0000, 0.0463) | 0.2823 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | 0.0138 | (0.0000, 0.0413) | 0.3127 | 0.0138 | (0.0000, 0.0275) | 0.2987 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 0.0347 | (0.0084, 0.0681) | 0.0013 | 0.0347 | (0.0000, 0.0585) | 0.0337 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | -0.0045 | (-0.0321, 0.0186) | 0.6333 | -0.0045 | (-0.0091, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.1136 | (0.0000, 0.2955) | 0.0623 | 0.1136 | (0.0000, 0.1818) | 0.0397 |
| proposed_vs_candidate_no_context | context_overlap | 0.0244 | (0.0053, 0.0499) | 0.0057 | 0.0244 | (0.0000, 0.0416) | 0.0350 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0417 | (-0.2500, 0.1250) | 0.7313 | -0.0417 | (-0.0833, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | persona_style | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | distinct1 | -0.0014 | (-0.0041, 0.0000) | 1.0000 | -0.0014 | (-0.0027, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | length_score | 0.0583 | (-0.1083, 0.2500) | 0.2590 | 0.0583 | (0.0000, 0.0833) | 0.0457 |
| proposed_vs_candidate_no_context | sentence_score | 0.0875 | (0.0000, 0.2625) | 0.3260 | 0.0875 | (0.0000, 0.1750) | 0.2943 |
| proposed_vs_candidate_no_context | overall_quality | 0.0318 | (0.0086, 0.0582) | 0.0037 | 0.0318 | (0.0000, 0.0465) | 0.0367 |
| proposed_vs_baseline_no_context | context_relevance | 0.0543 | (-0.0479, 0.1987) | 0.2593 | 0.0543 | (-0.0718, 0.1444) | 0.3040 |
| proposed_vs_baseline_no_context | persona_consistency | 0.0333 | (0.0000, 0.1000) | 0.3280 | 0.0333 | (0.0000, 0.0667) | 0.3023 |
| proposed_vs_baseline_no_context | naturalness | -0.0099 | (-0.1737, 0.1114) | 0.5723 | -0.0099 | (-0.2447, 0.0829) | 0.5827 |
| proposed_vs_baseline_no_context | quest_state_correctness | 0.0414 | (-0.0411, 0.1609) | 0.2647 | 0.0414 | (-0.0625, 0.1213) | 0.2960 |
| proposed_vs_baseline_no_context | lore_consistency | 0.0078 | (-0.0469, 0.0694) | 0.3720 | 0.0078 | (-0.0631, 0.0463) | 0.3823 |
| proposed_vs_baseline_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_baseline_no_context | objective_completion_support | -0.0208 | (-0.0417, 0.0000) | 1.0000 | -0.0208 | (-0.0417, 0.0000) | 1.0000 |
| proposed_vs_baseline_no_context | gameplay_usefulness | 0.0070 | (-0.0701, 0.0770) | 0.3713 | 0.0070 | (-0.1190, 0.0770) | 0.4043 |
| proposed_vs_baseline_no_context | time_pressure_acceptability | -0.0094 | (-0.0187, 0.0000) | 1.0000 | -0.0094 | (-0.0187, 0.0000) | 1.0000 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0659 | (-0.0750, 0.2727) | 0.3083 | 0.0659 | (-0.1000, 0.1818) | 0.2990 |
| proposed_vs_baseline_no_context | context_overlap | 0.0271 | (-0.0029, 0.0595) | 0.0513 | 0.0271 | (-0.0058, 0.0570) | 0.2953 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 0.0417 | (0.0000, 0.1250) | 0.3117 | 0.0417 | (0.0000, 0.0833) | 0.3017 |
| proposed_vs_baseline_no_context | persona_style | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_baseline_no_context | distinct1 | -0.0247 | (-0.0682, 0.0139) | 0.8727 | -0.0247 | (-0.0909, 0.0278) | 0.8463 |
| proposed_vs_baseline_no_context | length_score | 0.0000 | (-0.6000, 0.5167) | 0.4407 | 0.0000 | (-0.8667, 0.4500) | 0.4077 |
| proposed_vs_baseline_no_context | sentence_score | 0.0000 | (-0.2625, 0.2625) | 0.6250 | 0.0000 | (-0.3500, 0.3500) | 0.6327 |
| proposed_vs_baseline_no_context | overall_quality | 0.0353 | (-0.0404, 0.1081) | 0.1757 | 0.0353 | (-0.0828, 0.1081) | 0.2640 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 0.0668 | (-0.0386, 0.2008) | 0.1520 | 0.0668 | (-0.0741, 0.1366) | 0.1563 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | 0.0333 | (0.0000, 0.1000) | 0.3147 | 0.0333 | (0.0000, 0.0667) | 0.2987 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | -0.0064 | (-0.0969, 0.0466) | 0.6717 | -0.0064 | (-0.1447, 0.0466) | 0.5900 |
| proposed_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.0545 | (-0.0364, 0.1690) | 0.1523 | 0.0545 | (-0.0633, 0.1185) | 0.1557 |
| proposed_vs_baseline_no_context_phi3_latest | lore_consistency | 0.0191 | (-0.0116, 0.0692) | 0.3123 | 0.0191 | (-0.0154, 0.0463) | 0.2920 |
| proposed_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | objective_completion_support | -0.0000 | (-0.0312, 0.0312) | 0.6250 | -0.0000 | (-0.0208, 0.0417) | 0.6260 |
| proposed_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 0.0180 | (-0.0358, 0.0624) | 0.2393 | 0.0180 | (-0.0597, 0.0480) | 0.2543 |
| proposed_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | -0.0147 | (-0.0321, 0.0014) | 0.9373 | -0.0147 | (-0.0308, 0.0027) | 0.7590 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.0886 | (-0.0523, 0.2727) | 0.1993 | 0.0886 | (-0.1000, 0.1818) | 0.1623 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 0.0159 | (-0.0033, 0.0312) | 0.0563 | 0.0159 | (-0.0136, 0.0312) | 0.1527 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.0417 | (0.0000, 0.1250) | 0.3193 | 0.0417 | (0.0000, 0.0833) | 0.2943 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | -0.0119 | (-0.0682, 0.0503) | 0.6633 | -0.0119 | (-0.0909, 0.0789) | 0.6310 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | -0.0083 | (-0.2333, 0.2167) | 0.5603 | -0.0083 | (-0.3667, 0.2167) | 0.6230 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | 0.0000 | (-0.2625, 0.2625) | 0.6330 | 0.0000 | (-0.3500, 0.3500) | 0.6337 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | 0.0424 | (-0.0321, 0.1075) | 0.1443 | 0.0424 | (-0.0636, 0.0960) | 0.1593 |
| controlled_vs_proposed_raw | context_relevance | 0.1135 | (-0.0121, 0.2087) | 0.0377 | 0.1135 | (0.0387, 0.2310) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.1161 | (0.0333, 0.1971) | 0.0023 | 0.1161 | (0.0667, 0.2286) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | -0.0111 | (-0.1090, 0.0957) | 0.6223 | -0.0111 | (-0.1451, 0.1406) | 0.6333 |
| controlled_vs_proposed_raw | quest_state_correctness | 0.0880 | (-0.0130, 0.1688) | 0.0407 | 0.0880 | (0.0186, 0.1931) | 0.0000 |
| controlled_vs_proposed_raw | lore_consistency | 0.0990 | (0.0187, 0.1792) | 0.0000 | 0.0990 | (0.0187, 0.1901) | 0.0000 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0527 | (-0.0275, 0.1381) | 0.1187 | 0.0527 | (-0.0275, 0.1842) | 0.1433 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0523 | (-0.0174, 0.1566) | 0.1330 | 0.0523 | (-0.0010, 0.2076) | 0.1570 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 0.0110 | (-0.0495, 0.0801) | 0.3823 | 0.0110 | (-0.0338, 0.1068) | 0.3627 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.1477 | (-0.0182, 0.2705) | 0.0370 | 0.1477 | (0.0545, 0.3000) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0338 | (0.0016, 0.0659) | 0.0213 | 0.0338 | (0.0016, 0.0699) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1548 | (0.0417, 0.2560) | 0.0033 | 0.1548 | (0.0833, 0.2857) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | -0.0385 | (-0.1154, 0.0000) | 1.0000 | -0.0385 | (-0.1538, 0.0000) | 1.0000 |
| controlled_vs_proposed_raw | distinct1 | -0.0146 | (-0.0542, 0.0250) | 0.7200 | -0.0146 | (-0.0769, 0.0140) | 0.7060 |
| controlled_vs_proposed_raw | length_score | 0.0250 | (-0.2500, 0.3417) | 0.4593 | 0.0250 | (-0.3667, 0.5000) | 0.4023 |
| controlled_vs_proposed_raw | sentence_score | -0.0875 | (-0.3500, 0.1750) | 0.8137 | -0.0875 | (-0.3500, 0.3500) | 0.7363 |
| controlled_vs_proposed_raw | overall_quality | 0.0946 | (0.0342, 0.1806) | 0.0000 | 0.0946 | (0.0393, 0.2210) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2004 | (0.1739, 0.2224) | 0.0000 | 0.2004 | (0.1784, 0.2310) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.0828 | (-0.0667, 0.1971) | 0.1443 | 0.0828 | (-0.0000, 0.2286) | 0.0360 |
| controlled_vs_candidate_no_context | naturalness | 0.0088 | (-0.0920, 0.1051) | 0.3710 | 0.0088 | (-0.1318, 0.1406) | 0.3850 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.1590 | (0.1289, 0.1849) | 0.0000 | 0.1590 | (0.1382, 0.1931) | 0.0000 |
| controlled_vs_candidate_no_context | lore_consistency | 0.1221 | (0.0650, 0.1792) | 0.0000 | 0.1221 | (0.0650, 0.1901) | 0.0000 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | 0.0665 | (0.0000, 0.1381) | 0.0627 | 0.0665 | (0.0000, 0.1842) | 0.0397 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0870 | (0.0353, 0.1682) | 0.0000 | 0.0870 | (0.0253, 0.2076) | 0.0000 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 0.0065 | (-0.0429, 0.0694) | 0.4327 | 0.0065 | (-0.0429, 0.1068) | 0.3640 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.2614 | (0.2182, 0.2932) | 0.0000 | 0.2614 | (0.2364, 0.3000) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0581 | (0.0308, 0.0743) | 0.0000 | 0.0581 | (0.0432, 0.0762) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1131 | (-0.0833, 0.2560) | 0.1400 | 0.1131 | (0.0000, 0.2857) | 0.0373 |
| controlled_vs_candidate_no_context | persona_style | -0.0385 | (-0.1154, 0.0000) | 1.0000 | -0.0385 | (-0.1538, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | distinct1 | -0.0160 | (-0.0542, 0.0221) | 0.8077 | -0.0160 | (-0.0769, 0.0140) | 0.7320 |
| controlled_vs_candidate_no_context | length_score | 0.0833 | (-0.1833, 0.3583) | 0.2763 | 0.0833 | (-0.3000, 0.5000) | 0.3673 |
| controlled_vs_candidate_no_context | sentence_score | 0.0000 | (-0.2625, 0.2625) | 0.6433 | 0.0000 | (-0.3500, 0.3500) | 0.6353 |
| controlled_vs_candidate_no_context | overall_quality | 0.1264 | (0.0640, 0.1940) | 0.0000 | 0.1264 | (0.0858, 0.2210) | 0.0000 |
| controlled_vs_baseline_no_context | context_relevance | 0.1678 | (0.1509, 0.1900) | 0.0000 | 0.1678 | (0.1458, 0.1831) | 0.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1495 | (0.1103, 0.2048) | 0.0000 | 0.1495 | (0.1026, 0.2286) | 0.0000 |
| controlled_vs_baseline_no_context | naturalness | -0.0210 | (-0.1049, 0.0629) | 0.6997 | -0.0210 | (-0.1057, 0.0629) | 0.7417 |
| controlled_vs_baseline_no_context | quest_state_correctness | 0.1294 | (0.1128, 0.1507) | 0.0000 | 0.1294 | (0.1072, 0.1399) | 0.0000 |
| controlled_vs_baseline_no_context | lore_consistency | 0.1068 | (0.0558, 0.1522) | 0.0000 | 0.1068 | (0.0650, 0.1703) | 0.0000 |
| controlled_vs_baseline_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_baseline_no_context | objective_completion_support | 0.0319 | (-0.0625, 0.1381) | 0.3150 | 0.0319 | (-0.0483, 0.1842) | 0.3660 |
| controlled_vs_baseline_no_context | gameplay_usefulness | 0.0593 | (0.0195, 0.0989) | 0.0070 | 0.0593 | (-0.0035, 0.0886) | 0.0347 |
| controlled_vs_baseline_no_context | time_pressure_acceptability | 0.0017 | (-0.0648, 0.0767) | 0.4640 | 0.0017 | (-0.0432, 0.1068) | 0.4133 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.2136 | (0.1864, 0.2545) | 0.0000 | 0.2136 | (0.1818, 0.2364) | 0.0000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0608 | (0.0394, 0.0801) | 0.0000 | 0.0608 | (0.0587, 0.0640) | 0.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1964 | (0.1667, 0.2560) | 0.0000 | 0.1964 | (0.1667, 0.2857) | 0.0000 |
| controlled_vs_baseline_no_context | persona_style | -0.0385 | (-0.1154, 0.0000) | 1.0000 | -0.0385 | (-0.1538, 0.0000) | 1.0000 |
| controlled_vs_baseline_no_context | distinct1 | -0.0393 | (-0.0655, -0.0123) | 1.0000 | -0.0393 | (-0.0769, -0.0156) | 1.0000 |
| controlled_vs_baseline_no_context | length_score | 0.0250 | (-0.3833, 0.4333) | 0.3997 | 0.0250 | (-0.4000, 0.4333) | 0.3020 |
| controlled_vs_baseline_no_context | sentence_score | -0.0875 | (-0.2625, 0.0000) | 1.0000 | -0.0875 | (-0.1750, 0.0000) | 1.0000 |
| controlled_vs_baseline_no_context | overall_quality | 0.1299 | (0.1015, 0.1474) | 0.0000 | 0.1299 | (0.0867, 0.1474) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.1803 | (0.1536, 0.2071) | 0.0000 | 0.1803 | (0.1569, 0.2139) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1495 | (0.1103, 0.2048) | 0.0000 | 0.1495 | (0.1026, 0.2286) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | -0.0175 | (-0.0720, 0.0177) | 0.7137 | -0.0175 | (-0.0985, 0.0162) | 0.7353 |
| controlled_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.1426 | (0.1213, 0.1639) | 0.0000 | 0.1426 | (0.1298, 0.1663) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | lore_consistency | 0.1180 | (0.0650, 0.1711) | 0.0000 | 0.1180 | (0.0650, 0.1747) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.0527 | (-0.0521, 0.1694) | 0.2080 | 0.0527 | (-0.0483, 0.2258) | 0.1473 |
| controlled_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 0.0704 | (0.0412, 0.1217) | 0.0000 | 0.0704 | (0.0395, 0.1480) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | -0.0037 | (-0.0646, 0.0715) | 0.5907 | -0.0037 | (-0.0646, 0.1096) | 0.5927 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.2364 | (0.2000, 0.2727) | 0.0000 | 0.2364 | (0.2000, 0.2727) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0496 | (0.0328, 0.0664) | 0.0000 | 0.0496 | (0.0328, 0.0766) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1964 | (0.1667, 0.2560) | 0.0000 | 0.1964 | (0.1667, 0.2857) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | -0.0385 | (-0.1154, 0.0000) | 1.0000 | -0.0385 | (-0.1538, 0.0000) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0265 | (-0.0577, 0.0010) | 0.9447 | -0.0265 | (-0.0769, 0.0020) | 0.9653 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | 0.0167 | (-0.3167, 0.2500) | 0.3913 | 0.0167 | (-0.4667, 0.2000) | 0.3720 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | -0.0875 | (-0.2625, 0.0000) | 1.0000 | -0.0875 | (-0.1750, 0.0000) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1369 | (0.1208, 0.1531) | 0.0000 | 0.1369 | (0.1199, 0.1574) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.1678 | (0.1509, 0.1900) | 0.0000 | 0.1678 | (0.1458, 0.1831) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1495 | (0.1103, 0.2048) | 0.0000 | 0.1495 | (0.1026, 0.2286) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0210 | (-0.1049, 0.0629) | 0.7120 | -0.0210 | (-0.1057, 0.0629) | 0.7347 |
| proposed_contextual_controlled_vs_baseline_no_context | quest_state_correctness | 0.1294 | (0.1128, 0.1507) | 0.0000 | 0.1294 | (0.1072, 0.1399) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | lore_consistency | 0.1068 | (0.0558, 0.1522) | 0.0000 | 0.1068 | (0.0650, 0.1703) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | objective_completion_support | 0.0319 | (-0.0625, 0.1381) | 0.2873 | 0.0319 | (-0.0483, 0.1842) | 0.3607 |
| proposed_contextual_controlled_vs_baseline_no_context | gameplay_usefulness | 0.0593 | (0.0195, 0.0989) | 0.0037 | 0.0593 | (-0.0035, 0.0886) | 0.0410 |
| proposed_contextual_controlled_vs_baseline_no_context | time_pressure_acceptability | 0.0017 | (-0.0648, 0.0767) | 0.4723 | 0.0017 | (-0.0432, 0.1068) | 0.4010 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.2136 | (0.1864, 0.2545) | 0.0000 | 0.2136 | (0.1818, 0.2364) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0608 | (0.0394, 0.0801) | 0.0000 | 0.0608 | (0.0587, 0.0640) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1964 | (0.1667, 0.2560) | 0.0000 | 0.1964 | (0.1667, 0.2857) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | -0.0385 | (-0.1154, 0.0000) | 1.0000 | -0.0385 | (-0.1538, 0.0000) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0393 | (-0.0655, -0.0123) | 1.0000 | -0.0393 | (-0.0769, -0.0156) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.0250 | (-0.3833, 0.4333) | 0.4073 | 0.0250 | (-0.4000, 0.4333) | 0.2917 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | -0.0875 | (-0.2625, 0.0000) | 1.0000 | -0.0875 | (-0.1750, 0.0000) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1299 | (0.1015, 0.1474) | 0.0000 | 0.1299 | (0.0867, 0.1474) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.1803 | (0.1536, 0.2071) | 0.0000 | 0.1803 | (0.1569, 0.2139) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1495 | (0.1103, 0.2048) | 0.0000 | 0.1495 | (0.1026, 0.2286) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | -0.0175 | (-0.0720, 0.0177) | 0.7003 | -0.0175 | (-0.0985, 0.0162) | 0.7400 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.1426 | (0.1213, 0.1639) | 0.0000 | 0.1426 | (0.1298, 0.1663) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | lore_consistency | 0.1180 | (0.0650, 0.1711) | 0.0000 | 0.1180 | (0.0650, 0.1747) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.0527 | (-0.0521, 0.1694) | 0.2203 | 0.0527 | (-0.0483, 0.2258) | 0.1467 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 0.0704 | (0.0412, 0.1217) | 0.0000 | 0.0704 | (0.0395, 0.1480) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | -0.0037 | (-0.0646, 0.0715) | 0.5683 | -0.0037 | (-0.0646, 0.1096) | 0.5893 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.2364 | (0.2000, 0.2727) | 0.0000 | 0.2364 | (0.2000, 0.2727) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0496 | (0.0328, 0.0664) | 0.0000 | 0.0496 | (0.0328, 0.0766) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1964 | (0.1667, 0.2560) | 0.0000 | 0.1964 | (0.1667, 0.2857) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | -0.0385 | (-0.1154, 0.0000) | 1.0000 | -0.0385 | (-0.1538, 0.0000) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0265 | (-0.0577, 0.0010) | 0.9400 | -0.0265 | (-0.0769, 0.0020) | 0.9623 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | 0.0167 | (-0.3167, 0.2500) | 0.3797 | 0.0167 | (-0.4667, 0.2000) | 0.3740 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | -0.0875 | (-0.2625, 0.0000) | 1.0000 | -0.0875 | (-0.1750, 0.0000) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1369 | (0.1208, 0.1531) | 0.0000 | 0.1369 | (0.1199, 0.1574) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 3 | 0 | 1 | 0.8750 | 1.0000 |
| proposed_vs_candidate_no_context | persona_consistency | 1 | 1 | 2 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | naturalness | 2 | 1 | 1 | 0.6250 | 0.6667 |
| proposed_vs_candidate_no_context | quest_state_correctness | 3 | 0 | 1 | 0.8750 | 1.0000 |
| proposed_vs_candidate_no_context | lore_consistency | 1 | 0 | 3 | 0.6250 | 1.0000 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 4 | 0.5000 | nan |
| proposed_vs_candidate_no_context | objective_completion_support | 1 | 0 | 3 | 0.6250 | 1.0000 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 3 | 0 | 1 | 0.8750 | 1.0000 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | 1 | 1 | 2 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 2 | 0 | 2 | 0.7500 | 1.0000 |
| proposed_vs_candidate_no_context | context_overlap | 3 | 0 | 1 | 0.8750 | 1.0000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 1 | 1 | 2 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | persona_style | 0 | 0 | 4 | 0.5000 | nan |
| proposed_vs_candidate_no_context | distinct1 | 0 | 1 | 3 | 0.3750 | 0.0000 |
| proposed_vs_candidate_no_context | length_score | 2 | 1 | 1 | 0.6250 | 0.6667 |
| proposed_vs_candidate_no_context | sentence_score | 1 | 0 | 3 | 0.6250 | 1.0000 |
| proposed_vs_candidate_no_context | overall_quality | 3 | 0 | 1 | 0.8750 | 1.0000 |
| proposed_vs_baseline_no_context | context_relevance | 2 | 1 | 1 | 0.6250 | 0.6667 |
| proposed_vs_baseline_no_context | persona_consistency | 1 | 0 | 3 | 0.6250 | 1.0000 |
| proposed_vs_baseline_no_context | naturalness | 3 | 1 | 0 | 0.7500 | 0.7500 |
| proposed_vs_baseline_no_context | quest_state_correctness | 2 | 2 | 0 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context | lore_consistency | 2 | 1 | 1 | 0.6250 | 0.6667 |
| proposed_vs_baseline_no_context | multi_turn_contradiction_safety | 0 | 0 | 4 | 0.5000 | nan |
| proposed_vs_baseline_no_context | objective_completion_support | 0 | 2 | 2 | 0.2500 | 0.0000 |
| proposed_vs_baseline_no_context | gameplay_usefulness | 2 | 2 | 0 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context | time_pressure_acceptability | 0 | 2 | 2 | 0.2500 | 0.0000 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 1 | 1 | 2 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context | context_overlap | 2 | 1 | 1 | 0.6250 | 0.6667 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 1 | 0 | 3 | 0.6250 | 1.0000 |
| proposed_vs_baseline_no_context | persona_style | 0 | 0 | 4 | 0.5000 | nan |
| proposed_vs_baseline_no_context | distinct1 | 1 | 2 | 1 | 0.3750 | 0.3333 |
| proposed_vs_baseline_no_context | length_score | 2 | 2 | 0 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context | sentence_score | 1 | 1 | 2 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context | overall_quality | 3 | 1 | 0 | 0.7500 | 0.7500 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 3 | 1 | 0 | 0.7500 | 0.7500 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | 1 | 0 | 3 | 0.6250 | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | 3 | 1 | 0 | 0.7500 | 0.7500 |
| proposed_vs_baseline_no_context_phi3_latest | quest_state_correctness | 3 | 1 | 0 | 0.7500 | 0.7500 |
| proposed_vs_baseline_no_context_phi3_latest | lore_consistency | 1 | 2 | 1 | 0.3750 | 0.3333 |
| proposed_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0 | 0 | 4 | 0.5000 | nan |
| proposed_vs_baseline_no_context_phi3_latest | objective_completion_support | 1 | 1 | 2 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 3 | 1 | 0 | 0.7500 | 0.7500 |
| proposed_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 1 | 2 | 1 | 0.3750 | 0.3333 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 2 | 1 | 1 | 0.6250 | 0.6667 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 3 | 1 | 0 | 0.7500 | 0.7500 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 1 | 0 | 3 | 0.6250 | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | 0 | 0 | 4 | 0.5000 | nan |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | 1 | 2 | 1 | 0.3750 | 0.3333 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | 2 | 2 | 0 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | 1 | 1 | 2 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | 3 | 1 | 0 | 0.7500 | 0.7500 |
| controlled_vs_proposed_raw | context_relevance | 3 | 1 | 0 | 0.7500 | 0.7500 |
| controlled_vs_proposed_raw | persona_consistency | 3 | 0 | 1 | 0.8750 | 1.0000 |
| controlled_vs_proposed_raw | naturalness | 1 | 3 | 0 | 0.2500 | 0.2500 |
| controlled_vs_proposed_raw | quest_state_correctness | 3 | 1 | 0 | 0.7500 | 0.7500 |
| controlled_vs_proposed_raw | lore_consistency | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0 | 0 | 4 | 0.5000 | nan |
| controlled_vs_proposed_raw | objective_completion_support | 2 | 1 | 1 | 0.6250 | 0.6667 |
| controlled_vs_proposed_raw | gameplay_usefulness | 3 | 1 | 0 | 0.7500 | 0.7500 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 2 | 1 | 1 | 0.6250 | 0.6667 |
| controlled_vs_proposed_raw | context_keyword_coverage | 3 | 1 | 0 | 0.7500 | 0.7500 |
| controlled_vs_proposed_raw | context_overlap | 3 | 1 | 0 | 0.7500 | 0.7500 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 3 | 0 | 1 | 0.8750 | 1.0000 |
| controlled_vs_proposed_raw | persona_style | 0 | 1 | 3 | 0.3750 | 0.0000 |
| controlled_vs_proposed_raw | distinct1 | 2 | 2 | 0 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | length_score | 2 | 2 | 0 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | sentence_score | 1 | 2 | 1 | 0.3750 | 0.3333 |
| controlled_vs_proposed_raw | overall_quality | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_relevance | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 3 | 1 | 0 | 0.7500 | 0.7500 |
| controlled_vs_candidate_no_context | naturalness | 2 | 2 | 0 | 0.5000 | 0.5000 |
| controlled_vs_candidate_no_context | quest_state_correctness | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | lore_consistency | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 4 | 0.5000 | nan |
| controlled_vs_candidate_no_context | objective_completion_support | 2 | 0 | 2 | 0.7500 | 1.0000 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 2 | 2 | 0 | 0.5000 | 0.5000 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 3 | 1 | 0 | 0.7500 | 0.7500 |
| controlled_vs_candidate_no_context | persona_style | 0 | 1 | 3 | 0.3750 | 0.0000 |
| controlled_vs_candidate_no_context | distinct1 | 2 | 2 | 0 | 0.5000 | 0.5000 |
| controlled_vs_candidate_no_context | length_score | 2 | 2 | 0 | 0.5000 | 0.5000 |
| controlled_vs_candidate_no_context | sentence_score | 1 | 1 | 2 | 0.5000 | 0.5000 |
| controlled_vs_candidate_no_context | overall_quality | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | context_relevance | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | naturalness | 2 | 2 | 0 | 0.5000 | 0.5000 |
| controlled_vs_baseline_no_context | quest_state_correctness | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | lore_consistency | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | multi_turn_contradiction_safety | 0 | 0 | 4 | 0.5000 | nan |
| controlled_vs_baseline_no_context | objective_completion_support | 2 | 1 | 1 | 0.6250 | 0.6667 |
| controlled_vs_baseline_no_context | gameplay_usefulness | 3 | 1 | 0 | 0.7500 | 0.7500 |
| controlled_vs_baseline_no_context | time_pressure_acceptability | 1 | 2 | 1 | 0.3750 | 0.3333 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | context_overlap | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_style | 0 | 1 | 3 | 0.3750 | 0.0000 |
| controlled_vs_baseline_no_context | distinct1 | 0 | 3 | 1 | 0.1250 | 0.0000 |
| controlled_vs_baseline_no_context | length_score | 2 | 2 | 0 | 0.5000 | 0.5000 |
| controlled_vs_baseline_no_context | sentence_score | 0 | 1 | 3 | 0.3750 | 0.0000 |
| controlled_vs_baseline_no_context | overall_quality | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | 2 | 2 | 0 | 0.5000 | 0.5000 |
| controlled_vs_baseline_no_context_phi3_latest | quest_state_correctness | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | lore_consistency | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0 | 0 | 4 | 0.5000 | nan |
| controlled_vs_baseline_no_context_phi3_latest | objective_completion_support | 2 | 1 | 1 | 0.6250 | 0.6667 |
| controlled_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 2 | 2 | 0 | 0.5000 | 0.5000 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | 0 | 1 | 3 | 0.3750 | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | 1 | 2 | 1 | 0.3750 | 0.3333 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | 3 | 1 | 0 | 0.7500 | 0.7500 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0 | 1 | 3 | 0.3750 | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 4 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 4 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 4 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 2 | 2 | 0 | 0.5000 | 0.5000 |
| proposed_contextual_controlled_vs_baseline_no_context | quest_state_correctness | 4 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | lore_consistency | 4 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | multi_turn_contradiction_safety | 0 | 0 | 4 | 0.5000 | nan |
| proposed_contextual_controlled_vs_baseline_no_context | objective_completion_support | 2 | 1 | 1 | 0.6250 | 0.6667 |
| proposed_contextual_controlled_vs_baseline_no_context | gameplay_usefulness | 3 | 1 | 0 | 0.7500 | 0.7500 |
| proposed_contextual_controlled_vs_baseline_no_context | time_pressure_acceptability | 1 | 2 | 1 | 0.3750 | 0.3333 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 4 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 4 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 4 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0 | 1 | 3 | 0.3750 | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | 0 | 3 | 1 | 0.1250 | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 2 | 2 | 0 | 0.5000 | 0.5000 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0 | 1 | 3 | 0.3750 | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 4 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 4 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 4 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | 2 | 2 | 0 | 0.5000 | 0.5000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | quest_state_correctness | 4 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | lore_consistency | 4 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0 | 0 | 4 | 0.5000 | nan |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | objective_completion_support | 2 | 1 | 1 | 0.6250 | 0.6667 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 4 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 2 | 2 | 0 | 0.5000 | 0.5000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 4 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 4 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 4 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | 0 | 1 | 3 | 0.3750 | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | 1 | 2 | 1 | 0.3750 | 0.3333 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | 3 | 1 | 0 | 0.7500 | 0.7500 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0 | 1 | 3 | 0.3750 | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 4 | 0 | 0 | 1.0000 | 1.0000 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.0000 | 0.7500 | 0.0000 |
| proposed_contextual | 0.0000 | 0.0000 | 0.2500 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.2500 | 0.0000 | 0.0000 |
| baseline_no_context | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline_no_context_phi3_latest | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `3`
- Unique template signatures: `4`
- Template signature ratio: `1.0000`
- Effective sample size by source clustering: `2.67`
- Effective sample size by template-signature clustering: `4.00`
- Detailed diagnostics are published in `scenario_dependence.json`.

## Multi-turn Contradiction
| Arm | Contradiction Rate | Contradiction Safety | Contradicted Sources | Source Count |
|---|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 1.0000 | 0 | 3 |
| proposed_contextual | 0.0000 | 1.0000 | 0 | 3 |
| candidate_no_context | 0.0000 | 1.0000 | 0 | 3 |
| baseline_no_context | 0.0000 | 1.0000 | 0 | 3 |
| baseline_no_context_phi3_latest | 0.0000 | 1.0000 | 0 | 3 |
- Detailed source-level values are published in `multi_turn_contradictions.json`.

- BERTScore status: unavailable (No module named 'bert_score').

This report covers proposal RO5 metrics plus game-facing outcomes: quest-state correctness, lore consistency, contradiction safety, objective completion support, gameplay usefulness, and time-pressure acceptability.
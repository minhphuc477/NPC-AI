# Proposal Alignment Evaluation Report

- Run ID: `20260312T114229Z`
- Generated: `2026-03-12T11:44:47.552426+00:00`
- Scenarios: `artifacts\proposal\20260312T114229Z\scenarios.jsonl`
- Scenario count: `144`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off
- `baseline_no_context`: model `phi3:mini`, dynamic_context=off, response_control=off
- `baseline_no_context_phi3_latest`: model `phi3:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.1121 (0.0914, 0.1348) | 0.2815 (0.2598, 0.3054) | 0.8688 (0.8618, 0.8758) | 0.3223 (0.3073, 0.3369) | n/a |
| proposed_contextual | 0.0875 (0.0706, 0.1057) | 0.2040 (0.1830, 0.2252) | 0.8717 (0.8644, 0.8783) | 0.2821 (0.2711, 0.2935) | n/a |
| candidate_no_context | 0.0272 (0.0209, 0.0336) | 0.2160 (0.1963, 0.2353) | 0.8697 (0.8635, 0.8766) | 0.2586 (0.2504, 0.2681) | n/a |
| baseline_no_context | 0.0382 (0.0306, 0.0462) | 0.1522 (0.1377, 0.1671) | 0.8920 (0.8832, 0.9003) | 0.2438 (0.2364, 0.2511) | n/a |
| baseline_no_context_phi3_latest | 0.0353 (0.0282, 0.0420) | 0.1584 (0.1438, 0.1734) | 0.8911 (0.8823, 0.8996) | 0.2447 (0.2372, 0.2524) | n/a |

## Game-facing Outcome Metrics (mean, 95% CI)
| Arm | Quest-state Correctness | Lore Consistency | Contradiction Safety | Objective Completion Support | Gameplay Usefulness | Time-pressure Acceptability |
|---|---:|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.1974 (0.1788, 0.2156) | 0.0549 (0.0394, 0.0726) | 1.0000 (1.0000, 1.0000) | 0.0847 (0.0723, 0.0967) | 0.3050 (0.2985, 0.3118) | 0.2965 (0.2876, 0.3058) |
| proposed_contextual | 0.1750 (0.1608, 0.1899) | 0.0453 (0.0336, 0.0585) | 0.8750 (0.8194, 0.9236) | 0.0822 (0.0705, 0.0944) | 0.3004 (0.2934, 0.3076) | 0.2972 (0.2893, 0.3052) |
| candidate_no_context | 0.1236 (0.1178, 0.1298) | 0.0059 (0.0033, 0.0090) | 1.0000 (1.0000, 1.0000) | 0.0723 (0.0611, 0.0845) | 0.2776 (0.2711, 0.2842) | 0.2946 (0.2867, 0.3032) |
| baseline_no_context | 0.1325 (0.1256, 0.1401) | 0.0140 (0.0100, 0.0186) | 0.9021 (0.8531, 0.9510) | 0.0414 (0.0344, 0.0490) | 0.2770 (0.2715, 0.2825) | 0.2854 (0.2796, 0.2915) |
| baseline_no_context_phi3_latest | 0.1296 (0.1233, 0.1359) | 0.0136 (0.0098, 0.0179) | 1.0000 (1.0000, 1.0000) | 0.0386 (0.0312, 0.0460) | 0.2754 (0.2701, 0.2808) | 0.2892 (0.2835, 0.2950) |

- Multi-turn contradiction rate is reported as `1 - contradiction_safety` in row-level outputs.

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0603 | 2.2225 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0120 | -0.0556 |
| proposed_vs_candidate_no_context | naturalness | 0.0020 | 0.0023 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.0515 | 0.4166 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0394 | 6.6647 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | -0.1250 | -0.1250 |
| proposed_vs_candidate_no_context | objective_completion_support | 0.0100 | 0.1380 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 0.0228 | 0.0822 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | 0.0026 | 0.0089 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0773 | 3.1348 |
| proposed_vs_candidate_no_context | context_overlap | 0.0207 | 0.6278 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0198 | -0.1470 |
| proposed_vs_candidate_no_context | persona_style | 0.0191 | 0.0352 |
| proposed_vs_candidate_no_context | distinct1 | -0.0018 | -0.0019 |
| proposed_vs_candidate_no_context | length_score | 0.0243 | 0.0496 |
| proposed_vs_candidate_no_context | sentence_score | -0.0219 | -0.0228 |
| proposed_vs_candidate_no_context | overall_quality | 0.0235 | 0.0910 |
| proposed_vs_baseline_no_context | context_relevance | 0.0493 | 1.2879 |
| proposed_vs_baseline_no_context | persona_consistency | 0.0517 | 0.3396 |
| proposed_vs_baseline_no_context | naturalness | -0.0202 | -0.0227 |
| proposed_vs_baseline_no_context | quest_state_correctness | 0.0426 | 0.3214 |
| proposed_vs_baseline_no_context | lore_consistency | 0.0312 | 2.2246 |
| proposed_vs_baseline_no_context | multi_turn_contradiction_safety | -0.0271 | -0.0300 |
| proposed_vs_baseline_no_context | objective_completion_support | 0.0408 | 0.9847 |
| proposed_vs_baseline_no_context | gameplay_usefulness | 0.0234 | 0.0845 |
| proposed_vs_baseline_no_context | time_pressure_acceptability | 0.0118 | 0.0414 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0636 | 1.6555 |
| proposed_vs_baseline_no_context | context_overlap | 0.0158 | 0.4171 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 0.0658 | 1.3416 |
| proposed_vs_baseline_no_context | persona_style | -0.0045 | -0.0080 |
| proposed_vs_baseline_no_context | distinct1 | -0.0408 | -0.0417 |
| proposed_vs_baseline_no_context | length_score | -0.0465 | -0.0829 |
| proposed_vs_baseline_no_context | sentence_score | 0.0536 | 0.0605 |
| proposed_vs_baseline_no_context | overall_quality | 0.0382 | 0.1568 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 0.0522 | 1.4811 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | 0.0455 | 0.2872 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | -0.0194 | -0.0218 |
| proposed_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.0454 | 0.3507 |
| proposed_vs_baseline_no_context_phi3_latest | lore_consistency | 0.0316 | 2.3192 |
| proposed_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | -0.1250 | -0.1250 |
| proposed_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.0437 | 1.1314 |
| proposed_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 0.0250 | 0.0906 |
| proposed_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 0.0081 | 0.0279 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.0668 | 1.9004 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 0.0181 | 0.5111 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.0572 | 0.9948 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | -0.0014 | -0.0025 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | -0.0452 | -0.0460 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | -0.0396 | -0.0714 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | 0.0656 | 0.0751 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | 0.0374 | 0.1526 |
| controlled_vs_proposed_raw | context_relevance | 0.0246 | 0.2810 |
| controlled_vs_proposed_raw | persona_consistency | 0.0775 | 0.3800 |
| controlled_vs_proposed_raw | naturalness | -0.0029 | -0.0034 |
| controlled_vs_proposed_raw | quest_state_correctness | 0.0224 | 0.1277 |
| controlled_vs_proposed_raw | lore_consistency | 0.0096 | 0.2128 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0.1250 | 0.1429 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0025 | 0.0298 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0046 | 0.0153 |
| controlled_vs_proposed_raw | time_pressure_acceptability | -0.0007 | -0.0024 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.0329 | 0.3222 |
| controlled_vs_proposed_raw | context_overlap | 0.0053 | 0.0982 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1008 | 0.8784 |
| controlled_vs_proposed_raw | persona_style | -0.0158 | -0.0281 |
| controlled_vs_proposed_raw | distinct1 | -0.0012 | -0.0013 |
| controlled_vs_proposed_raw | length_score | -0.0338 | -0.0657 |
| controlled_vs_proposed_raw | sentence_score | 0.0438 | 0.0466 |
| controlled_vs_proposed_raw | overall_quality | 0.0402 | 0.1425 |
| controlled_vs_candidate_no_context | context_relevance | 0.0849 | 3.1281 |
| controlled_vs_candidate_no_context | persona_consistency | 0.0655 | 0.3033 |
| controlled_vs_candidate_no_context | naturalness | -0.0010 | -0.0011 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.0738 | 0.5975 |
| controlled_vs_candidate_no_context | lore_consistency | 0.0490 | 8.2958 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | 0.0124 | 0.1719 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0274 | 0.0987 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 0.0019 | 0.0064 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.1102 | 4.4670 |
| controlled_vs_candidate_no_context | context_overlap | 0.0259 | 0.7877 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.0811 | 0.6024 |
| controlled_vs_candidate_no_context | persona_style | 0.0033 | 0.0061 |
| controlled_vs_candidate_no_context | distinct1 | -0.0030 | -0.0032 |
| controlled_vs_candidate_no_context | length_score | -0.0095 | -0.0194 |
| controlled_vs_candidate_no_context | sentence_score | 0.0219 | 0.0228 |
| controlled_vs_candidate_no_context | overall_quality | 0.0637 | 0.2464 |
| controlled_vs_baseline_no_context | context_relevance | 0.0738 | 1.9309 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1292 | 0.8487 |
| controlled_vs_baseline_no_context | naturalness | -0.0232 | -0.0260 |
| controlled_vs_baseline_no_context | quest_state_correctness | 0.0649 | 0.4901 |
| controlled_vs_baseline_no_context | lore_consistency | 0.0409 | 2.9108 |
| controlled_vs_baseline_no_context | multi_turn_contradiction_safety | 0.0979 | 0.1085 |
| controlled_vs_baseline_no_context | objective_completion_support | 0.0433 | 1.0439 |
| controlled_vs_baseline_no_context | gameplay_usefulness | 0.0280 | 0.1011 |
| controlled_vs_baseline_no_context | time_pressure_acceptability | 0.0111 | 0.0389 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.0965 | 2.5110 |
| controlled_vs_baseline_no_context | context_overlap | 0.0210 | 0.5563 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1666 | 3.3986 |
| controlled_vs_baseline_no_context | persona_style | -0.0203 | -0.0359 |
| controlled_vs_baseline_no_context | distinct1 | -0.0420 | -0.0429 |
| controlled_vs_baseline_no_context | length_score | -0.0803 | -0.1431 |
| controlled_vs_baseline_no_context | sentence_score | 0.0973 | 0.1099 |
| controlled_vs_baseline_no_context | overall_quality | 0.0784 | 0.3216 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.0768 | 2.1784 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1230 | 0.7764 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | -0.0223 | -0.0251 |
| controlled_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.0678 | 0.5232 |
| controlled_vs_baseline_no_context_phi3_latest | lore_consistency | 0.0413 | 3.0255 |
| controlled_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.0461 | 1.1950 |
| controlled_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 0.0295 | 0.1073 |
| controlled_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 0.0073 | 0.0254 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.0997 | 2.8349 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0234 | 0.6596 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1581 | 2.7471 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | -0.0172 | -0.0306 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0464 | -0.0472 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | -0.0734 | -0.1324 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.1094 | 0.1252 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.0776 | 0.3169 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.0738 | 1.9309 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1292 | 0.8487 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0232 | -0.0260 |
| proposed_contextual_controlled_vs_baseline_no_context | quest_state_correctness | 0.0649 | 0.4901 |
| proposed_contextual_controlled_vs_baseline_no_context | lore_consistency | 0.0409 | 2.9108 |
| proposed_contextual_controlled_vs_baseline_no_context | multi_turn_contradiction_safety | 0.0979 | 0.1085 |
| proposed_contextual_controlled_vs_baseline_no_context | objective_completion_support | 0.0433 | 1.0439 |
| proposed_contextual_controlled_vs_baseline_no_context | gameplay_usefulness | 0.0280 | 0.1011 |
| proposed_contextual_controlled_vs_baseline_no_context | time_pressure_acceptability | 0.0111 | 0.0389 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.0965 | 2.5110 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0210 | 0.5563 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1666 | 3.3986 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | -0.0203 | -0.0359 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0420 | -0.0429 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | -0.0803 | -0.1431 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0973 | 0.1099 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.0784 | 0.3216 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.0768 | 2.1784 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1230 | 0.7764 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | -0.0223 | -0.0251 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.0678 | 0.5232 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | lore_consistency | 0.0413 | 3.0255 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.0461 | 1.1950 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 0.0295 | 0.1073 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 0.0073 | 0.0254 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.0997 | 2.8349 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0234 | 0.6596 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1581 | 2.7471 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | -0.0172 | -0.0306 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0464 | -0.0472 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | -0.0734 | -0.1324 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.1094 | 0.1252 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.0776 | 0.3169 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0603 | (0.0430, 0.0781) | 0.0000 | 0.0603 | (0.0316, 0.0947) | 0.0000 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0120 | (-0.0298, 0.0071) | 0.8903 | -0.0120 | (-0.0213, -0.0012) | 0.9840 |
| proposed_vs_candidate_no_context | naturalness | 0.0020 | (-0.0067, 0.0110) | 0.3310 | 0.0020 | (-0.0093, 0.0106) | 0.3270 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.0515 | (0.0369, 0.0680) | 0.0000 | 0.0515 | (0.0255, 0.0799) | 0.0000 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0394 | (0.0276, 0.0528) | 0.0000 | 0.0394 | (0.0183, 0.0620) | 0.0000 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | -0.1250 | (-0.1806, -0.0764) | 1.0000 | -0.1250 | (-0.3750, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | 0.0100 | (-0.0008, 0.0205) | 0.0317 | 0.0100 | (0.0009, 0.0185) | 0.0157 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 0.0228 | (0.0158, 0.0303) | 0.0000 | 0.0228 | (0.0132, 0.0321) | 0.0000 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | 0.0026 | (-0.0052, 0.0104) | 0.2503 | 0.0026 | (-0.0051, 0.0097) | 0.2377 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0773 | (0.0547, 0.1017) | 0.0000 | 0.0773 | (0.0407, 0.1189) | 0.0000 |
| proposed_vs_candidate_no_context | context_overlap | 0.0207 | (0.0141, 0.0276) | 0.0000 | 0.0207 | (0.0099, 0.0328) | 0.0000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0198 | (-0.0435, 0.0036) | 0.9507 | -0.0198 | (-0.0311, -0.0056) | 0.9953 |
| proposed_vs_candidate_no_context | persona_style | 0.0191 | (0.0059, 0.0336) | 0.0037 | 0.0191 | (0.0020, 0.0408) | 0.0143 |
| proposed_vs_candidate_no_context | distinct1 | -0.0018 | (-0.0093, 0.0059) | 0.6713 | -0.0018 | (-0.0088, 0.0037) | 0.6933 |
| proposed_vs_candidate_no_context | length_score | 0.0243 | (-0.0171, 0.0644) | 0.1247 | 0.0243 | (-0.0310, 0.0678) | 0.1607 |
| proposed_vs_candidate_no_context | sentence_score | -0.0219 | (-0.0462, 0.0024) | 0.9727 | -0.0219 | (-0.0340, -0.0097) | 1.0000 |
| proposed_vs_candidate_no_context | overall_quality | 0.0235 | (0.0128, 0.0346) | 0.0000 | 0.0235 | (0.0112, 0.0380) | 0.0000 |
| proposed_vs_baseline_no_context | context_relevance | 0.0483 | (0.0308, 0.0656) | 0.0000 | 0.0483 | (0.0217, 0.0832) | 0.0000 |
| proposed_vs_baseline_no_context | persona_consistency | 0.0517 | (0.0307, 0.0740) | 0.0000 | 0.0517 | (0.0142, 0.0899) | 0.0027 |
| proposed_vs_baseline_no_context | naturalness | -0.0203 | (-0.0307, -0.0097) | 1.0000 | -0.0203 | (-0.0278, -0.0123) | 1.0000 |
| proposed_vs_baseline_no_context | quest_state_correctness | 0.0416 | (0.0276, 0.0560) | 0.0000 | 0.0416 | (0.0201, 0.0681) | 0.0000 |
| proposed_vs_baseline_no_context | lore_consistency | 0.0311 | (0.0186, 0.0441) | 0.0000 | 0.0311 | (0.0130, 0.0527) | 0.0000 |
| proposed_vs_baseline_no_context | multi_turn_contradiction_safety | -0.0210 | (-0.0979, 0.0559) | 0.7277 | -0.0210 | (-0.3617, 0.2917) | 0.6403 |
| proposed_vs_baseline_no_context | objective_completion_support | 0.0400 | (0.0275, 0.0524) | 0.0000 | 0.0400 | (0.0243, 0.0599) | 0.0000 |
| proposed_vs_baseline_no_context | gameplay_usefulness | 0.0228 | (0.0143, 0.0317) | 0.0000 | 0.0228 | (0.0121, 0.0331) | 0.0003 |
| proposed_vs_baseline_no_context | time_pressure_acceptability | 0.0112 | (0.0027, 0.0201) | 0.0053 | 0.0112 | (-0.0004, 0.0234) | 0.0293 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0624 | (0.0397, 0.0849) | 0.0000 | 0.0624 | (0.0314, 0.1043) | 0.0000 |
| proposed_vs_baseline_no_context | context_overlap | 0.0154 | (0.0081, 0.0232) | 0.0000 | 0.0154 | (-0.0006, 0.0306) | 0.0313 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 0.0666 | (0.0385, 0.0937) | 0.0000 | 0.0666 | (0.0190, 0.1109) | 0.0043 |
| proposed_vs_baseline_no_context | persona_style | -0.0076 | (-0.0273, 0.0107) | 0.7910 | -0.0076 | (-0.0559, 0.0369) | 0.6133 |
| proposed_vs_baseline_no_context | distinct1 | -0.0412 | (-0.0506, -0.0319) | 1.0000 | -0.0412 | (-0.0581, -0.0272) | 1.0000 |
| proposed_vs_baseline_no_context | length_score | -0.0457 | (-0.0897, -0.0021) | 0.9793 | -0.0457 | (-0.0766, -0.0138) | 0.9987 |
| proposed_vs_baseline_no_context | sentence_score | 0.0531 | (0.0220, 0.0850) | 0.0007 | 0.0531 | (0.0111, 0.0954) | 0.0050 |
| proposed_vs_baseline_no_context | overall_quality | 0.0378 | (0.0260, 0.0498) | 0.0000 | 0.0378 | (0.0173, 0.0592) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 0.0522 | (0.0353, 0.0690) | 0.0000 | 0.0522 | (0.0254, 0.0859) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | 0.0455 | (0.0237, 0.0690) | 0.0000 | 0.0455 | (0.0173, 0.0735) | 0.0020 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | -0.0194 | (-0.0296, -0.0097) | 1.0000 | -0.0194 | (-0.0320, -0.0056) | 0.9977 |
| proposed_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.0454 | (0.0321, 0.0600) | 0.0000 | 0.0454 | (0.0232, 0.0729) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | lore_consistency | 0.0316 | (0.0192, 0.0451) | 0.0000 | 0.0316 | (0.0119, 0.0552) | 0.0003 |
| proposed_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | -0.1250 | (-0.1806, -0.0764) | 1.0000 | -0.1250 | (-0.3750, 0.0000) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.0437 | (0.0312, 0.0559) | 0.0000 | 0.0437 | (0.0278, 0.0635) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 0.0250 | (0.0165, 0.0332) | 0.0000 | 0.0250 | (0.0130, 0.0385) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 0.0081 | (-0.0009, 0.0171) | 0.0350 | 0.0081 | (-0.0040, 0.0230) | 0.1307 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.0668 | (0.0454, 0.0890) | 0.0000 | 0.0668 | (0.0344, 0.1086) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 0.0181 | (0.0113, 0.0258) | 0.0000 | 0.0181 | (0.0044, 0.0314) | 0.0027 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.0572 | (0.0291, 0.0851) | 0.0000 | 0.0572 | (0.0194, 0.0914) | 0.0023 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | -0.0014 | (-0.0192, 0.0169) | 0.5517 | -0.0014 | (-0.0353, 0.0331) | 0.5397 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | -0.0452 | (-0.0538, -0.0365) | 1.0000 | -0.0452 | (-0.0624, -0.0312) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | -0.0396 | (-0.0815, 0.0053) | 0.9597 | -0.0396 | (-0.0880, 0.0113) | 0.9387 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | 0.0656 | (0.0316, 0.0997) | 0.0000 | 0.0656 | (0.0073, 0.1215) | 0.0123 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | 0.0374 | (0.0255, 0.0494) | 0.0000 | 0.0374 | (0.0174, 0.0559) | 0.0000 |
| controlled_vs_proposed_raw | context_relevance | 0.0246 | (-0.0004, 0.0518) | 0.0277 | 0.0246 | (0.0086, 0.0418) | 0.0013 |
| controlled_vs_proposed_raw | persona_consistency | 0.0775 | (0.0520, 0.1023) | 0.0000 | 0.0775 | (0.0433, 0.1141) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | -0.0029 | (-0.0132, 0.0062) | 0.7267 | -0.0029 | (-0.0122, 0.0076) | 0.7113 |
| controlled_vs_proposed_raw | quest_state_correctness | 0.0224 | (0.0003, 0.0449) | 0.0237 | 0.0224 | (0.0072, 0.0394) | 0.0007 |
| controlled_vs_proposed_raw | lore_consistency | 0.0096 | (-0.0083, 0.0285) | 0.1497 | 0.0096 | (0.0001, 0.0193) | 0.0237 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0.1250 | (0.0764, 0.1806) | 0.0000 | 0.1250 | (0.0000, 0.3750) | 0.3403 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0025 | (-0.0088, 0.0141) | 0.3357 | 0.0025 | (-0.0081, 0.0165) | 0.3737 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0046 | (-0.0031, 0.0129) | 0.1263 | 0.0046 | (-0.0006, 0.0111) | 0.0510 |
| controlled_vs_proposed_raw | time_pressure_acceptability | -0.0007 | (-0.0097, 0.0081) | 0.5570 | -0.0007 | (-0.0092, 0.0083) | 0.5697 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.0329 | (-0.0013, 0.0648) | 0.0303 | 0.0329 | (0.0119, 0.0544) | 0.0007 |
| controlled_vs_proposed_raw | context_overlap | 0.0053 | (-0.0043, 0.0163) | 0.1570 | 0.0053 | (-0.0028, 0.0145) | 0.1280 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1008 | (0.0697, 0.1310) | 0.0000 | 0.1008 | (0.0617, 0.1435) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | -0.0158 | (-0.0287, -0.0032) | 0.9920 | -0.0158 | (-0.0373, 0.0009) | 0.9653 |
| controlled_vs_proposed_raw | distinct1 | -0.0012 | (-0.0099, 0.0073) | 0.6143 | -0.0012 | (-0.0068, 0.0038) | 0.6617 |
| controlled_vs_proposed_raw | length_score | -0.0338 | (-0.0775, 0.0083) | 0.9360 | -0.0338 | (-0.0831, 0.0164) | 0.9083 |
| controlled_vs_proposed_raw | sentence_score | 0.0437 | (0.0194, 0.0681) | 0.0000 | 0.0437 | (0.0219, 0.0681) | 0.0000 |
| controlled_vs_proposed_raw | overall_quality | 0.0402 | (0.0239, 0.0568) | 0.0000 | 0.0402 | (0.0247, 0.0594) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.0849 | (0.0632, 0.1083) | 0.0000 | 0.0849 | (0.0608, 0.1092) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.0655 | (0.0418, 0.0894) | 0.0000 | 0.0655 | (0.0336, 0.0988) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | -0.0010 | (-0.0097, 0.0081) | 0.5797 | -0.0010 | (-0.0080, 0.0069) | 0.6217 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.0738 | (0.0548, 0.0939) | 0.0000 | 0.0738 | (0.0525, 0.0950) | 0.0000 |
| controlled_vs_candidate_no_context | lore_consistency | 0.0490 | (0.0324, 0.0665) | 0.0000 | 0.0490 | (0.0334, 0.0672) | 0.0000 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | 0.0124 | (0.0017, 0.0230) | 0.0150 | 0.0124 | (0.0018, 0.0235) | 0.0130 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0274 | (0.0192, 0.0357) | 0.0000 | 0.0274 | (0.0161, 0.0375) | 0.0000 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 0.0019 | (-0.0085, 0.0119) | 0.3640 | 0.0019 | (-0.0050, 0.0077) | 0.2837 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.1102 | (0.0825, 0.1402) | 0.0000 | 0.1102 | (0.0786, 0.1424) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0259 | (0.0177, 0.0354) | 0.0000 | 0.0259 | (0.0165, 0.0362) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.0811 | (0.0510, 0.1108) | 0.0000 | 0.0811 | (0.0422, 0.1212) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.0033 | (-0.0089, 0.0155) | 0.2993 | 0.0033 | (0.0012, 0.0057) | 0.0033 |
| controlled_vs_candidate_no_context | distinct1 | -0.0030 | (-0.0111, 0.0051) | 0.7670 | -0.0030 | (-0.0116, 0.0054) | 0.7403 |
| controlled_vs_candidate_no_context | length_score | -0.0095 | (-0.0528, 0.0326) | 0.6753 | -0.0095 | (-0.0486, 0.0331) | 0.6657 |
| controlled_vs_candidate_no_context | sentence_score | 0.0219 | (0.0024, 0.0413) | 0.0207 | 0.0219 | (-0.0001, 0.0462) | 0.0393 |
| controlled_vs_candidate_no_context | overall_quality | 0.0637 | (0.0481, 0.0797) | 0.0000 | 0.0637 | (0.0459, 0.0835) | 0.0000 |
| controlled_vs_baseline_no_context | context_relevance | 0.0741 | (0.0510, 0.0971) | 0.0000 | 0.0741 | (0.0508, 0.1001) | 0.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1282 | (0.1044, 0.1531) | 0.0000 | 0.1282 | (0.0863, 0.1594) | 0.0000 |
| controlled_vs_baseline_no_context | naturalness | -0.0232 | (-0.0333, -0.0128) | 1.0000 | -0.0232 | (-0.0308, -0.0151) | 1.0000 |
| controlled_vs_baseline_no_context | quest_state_correctness | 0.0650 | (0.0458, 0.0856) | 0.0000 | 0.0650 | (0.0440, 0.0873) | 0.0000 |
| controlled_vs_baseline_no_context | lore_consistency | 0.0412 | (0.0252, 0.0584) | 0.0000 | 0.0412 | (0.0243, 0.0596) | 0.0000 |
| controlled_vs_baseline_no_context | multi_turn_contradiction_safety | 0.0979 | (0.0490, 0.1469) | 0.0000 | 0.0979 | (0.0000, 0.2937) | 0.3590 |
| controlled_vs_baseline_no_context | objective_completion_support | 0.0418 | (0.0299, 0.0542) | 0.0000 | 0.0418 | (0.0210, 0.0634) | 0.0000 |
| controlled_vs_baseline_no_context | gameplay_usefulness | 0.0274 | (0.0189, 0.0356) | 0.0000 | 0.0274 | (0.0148, 0.0372) | 0.0000 |
| controlled_vs_baseline_no_context | time_pressure_acceptability | 0.0101 | (0.0008, 0.0195) | 0.0183 | 0.0101 | (-0.0022, 0.0220) | 0.0540 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.0968 | (0.0680, 0.1272) | 0.0000 | 0.0968 | (0.0668, 0.1298) | 0.0000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0211 | (0.0120, 0.0309) | 0.0000 | 0.0211 | (0.0092, 0.0333) | 0.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1661 | (0.1372, 0.1938) | 0.0000 | 0.1661 | (0.1130, 0.2043) | 0.0000 |
| controlled_vs_baseline_no_context | persona_style | -0.0235 | (-0.0436, -0.0051) | 0.9947 | -0.0235 | (-0.0713, 0.0112) | 0.8520 |
| controlled_vs_baseline_no_context | distinct1 | -0.0418 | (-0.0505, -0.0332) | 1.0000 | -0.0418 | (-0.0562, -0.0311) | 1.0000 |
| controlled_vs_baseline_no_context | length_score | -0.0807 | (-0.1226, -0.0364) | 0.9997 | -0.0807 | (-0.1196, -0.0375) | 1.0000 |
| controlled_vs_baseline_no_context | sentence_score | 0.0972 | (0.0682, 0.1266) | 0.0000 | 0.0972 | (0.0654, 0.1271) | 0.0000 |
| controlled_vs_baseline_no_context | overall_quality | 0.0782 | (0.0640, 0.0935) | 0.0000 | 0.0782 | (0.0559, 0.0975) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.0768 | (0.0567, 0.0990) | 0.0000 | 0.0768 | (0.0540, 0.1031) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1230 | (0.0985, 0.1482) | 0.0000 | 0.1230 | (0.0798, 0.1590) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | -0.0223 | (-0.0324, -0.0117) | 1.0000 | -0.0223 | (-0.0309, -0.0141) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.0678 | (0.0494, 0.0875) | 0.0000 | 0.0678 | (0.0479, 0.0892) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | lore_consistency | 0.0413 | (0.0252, 0.0582) | 0.0000 | 0.0413 | (0.0228, 0.0610) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.0461 | (0.0343, 0.0584) | 0.0000 | 0.0461 | (0.0269, 0.0672) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 0.0295 | (0.0214, 0.0377) | 0.0000 | 0.0295 | (0.0166, 0.0416) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 0.0073 | (-0.0019, 0.0165) | 0.0627 | 0.0073 | (-0.0055, 0.0198) | 0.1347 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.0997 | (0.0724, 0.1288) | 0.0000 | 0.0997 | (0.0719, 0.1306) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0234 | (0.0147, 0.0327) | 0.0000 | 0.0234 | (0.0128, 0.0352) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1581 | (0.1293, 0.1867) | 0.0000 | 0.1581 | (0.1025, 0.2030) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | -0.0172 | (-0.0358, 0.0001) | 0.9743 | -0.0172 | (-0.0508, 0.0087) | 0.8680 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0464 | (-0.0550, -0.0381) | 1.0000 | -0.0464 | (-0.0615, -0.0337) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | -0.0734 | (-0.1229, -0.0262) | 1.0000 | -0.0734 | (-0.1262, -0.0238) | 0.9990 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.1094 | (0.0802, 0.1385) | 0.0000 | 0.1094 | (0.0656, 0.1507) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.0776 | (0.0630, 0.0929) | 0.0000 | 0.0776 | (0.0549, 0.0985) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.0741 | (0.0515, 0.0991) | 0.0000 | 0.0741 | (0.0503, 0.0986) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1282 | (0.1048, 0.1529) | 0.0000 | 0.1282 | (0.0864, 0.1598) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0232 | (-0.0334, -0.0129) | 1.0000 | -0.0232 | (-0.0305, -0.0147) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | quest_state_correctness | 0.0650 | (0.0459, 0.0853) | 0.0000 | 0.0650 | (0.0441, 0.0874) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | lore_consistency | 0.0412 | (0.0248, 0.0589) | 0.0000 | 0.0412 | (0.0238, 0.0609) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | multi_turn_contradiction_safety | 0.0979 | (0.0490, 0.1469) | 0.0000 | 0.0979 | (0.0000, 0.2937) | 0.3260 |
| proposed_contextual_controlled_vs_baseline_no_context | objective_completion_support | 0.0418 | (0.0297, 0.0537) | 0.0000 | 0.0418 | (0.0215, 0.0626) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | gameplay_usefulness | 0.0274 | (0.0191, 0.0355) | 0.0000 | 0.0274 | (0.0148, 0.0369) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | time_pressure_acceptability | 0.0101 | (0.0004, 0.0197) | 0.0220 | 0.0101 | (-0.0023, 0.0221) | 0.0553 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.0968 | (0.0672, 0.1259) | 0.0000 | 0.0968 | (0.0679, 0.1290) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0211 | (0.0124, 0.0306) | 0.0000 | 0.0211 | (0.0089, 0.0326) | 0.0003 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1661 | (0.1389, 0.1958) | 0.0000 | 0.1661 | (0.1130, 0.2044) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | -0.0235 | (-0.0435, -0.0050) | 0.9917 | -0.0235 | (-0.0713, 0.0104) | 0.8507 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0418 | (-0.0507, -0.0335) | 1.0000 | -0.0418 | (-0.0554, -0.0301) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | -0.0807 | (-0.1235, -0.0389) | 1.0000 | -0.0807 | (-0.1200, -0.0370) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0972 | (0.0682, 0.1273) | 0.0000 | 0.0972 | (0.0674, 0.1262) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.0782 | (0.0635, 0.0936) | 0.0000 | 0.0782 | (0.0554, 0.0970) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.0768 | (0.0563, 0.0993) | 0.0000 | 0.0768 | (0.0533, 0.1004) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1230 | (0.1003, 0.1485) | 0.0000 | 0.1230 | (0.0817, 0.1592) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | -0.0223 | (-0.0330, -0.0116) | 1.0000 | -0.0223 | (-0.0316, -0.0142) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.0678 | (0.0493, 0.0869) | 0.0000 | 0.0678 | (0.0477, 0.0896) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | lore_consistency | 0.0413 | (0.0253, 0.0575) | 0.0000 | 0.0413 | (0.0229, 0.0603) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.0461 | (0.0346, 0.0575) | 0.0000 | 0.0461 | (0.0273, 0.0671) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 0.0295 | (0.0215, 0.0378) | 0.0000 | 0.0295 | (0.0167, 0.0411) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 0.0073 | (-0.0015, 0.0169) | 0.0537 | 0.0073 | (-0.0060, 0.0205) | 0.1417 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.0997 | (0.0730, 0.1280) | 0.0000 | 0.0997 | (0.0707, 0.1323) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0234 | (0.0149, 0.0331) | 0.0000 | 0.0234 | (0.0128, 0.0351) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1581 | (0.1293, 0.1870) | 0.0000 | 0.1581 | (0.1060, 0.2021) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | -0.0172 | (-0.0365, 0.0003) | 0.9713 | -0.0172 | (-0.0522, 0.0087) | 0.8623 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0464 | (-0.0551, -0.0375) | 1.0000 | -0.0464 | (-0.0609, -0.0333) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | -0.0734 | (-0.1211, -0.0255) | 0.9987 | -0.0734 | (-0.1241, -0.0245) | 0.9997 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.1094 | (0.0802, 0.1385) | 0.0000 | 0.1094 | (0.0681, 0.1507) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.0776 | (0.0625, 0.0925) | 0.0000 | 0.0776 | (0.0550, 0.0984) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 72 | 25 | 47 | 0.6632 | 0.7423 |
| proposed_vs_candidate_no_context | persona_consistency | 23 | 31 | 90 | 0.4722 | 0.4259 |
| proposed_vs_candidate_no_context | naturalness | 49 | 48 | 47 | 0.5035 | 0.5052 |
| proposed_vs_candidate_no_context | quest_state_correctness | 69 | 28 | 47 | 0.6424 | 0.7113 |
| proposed_vs_candidate_no_context | lore_consistency | 56 | 10 | 78 | 0.6597 | 0.8485 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 18 | 126 | 0.4375 | 0.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | 50 | 31 | 63 | 0.5660 | 0.6173 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 70 | 27 | 47 | 0.6493 | 0.7216 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | 42 | 47 | 55 | 0.4826 | 0.4719 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 60 | 10 | 74 | 0.6736 | 0.8571 |
| proposed_vs_candidate_no_context | context_overlap | 73 | 24 | 47 | 0.6701 | 0.7526 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 18 | 29 | 97 | 0.4618 | 0.3830 |
| proposed_vs_candidate_no_context | persona_style | 15 | 5 | 124 | 0.5347 | 0.7500 |
| proposed_vs_candidate_no_context | distinct1 | 38 | 45 | 61 | 0.4757 | 0.4578 |
| proposed_vs_candidate_no_context | length_score | 51 | 44 | 49 | 0.5243 | 0.5368 |
| proposed_vs_candidate_no_context | sentence_score | 9 | 18 | 117 | 0.4688 | 0.3333 |
| proposed_vs_candidate_no_context | overall_quality | 63 | 34 | 47 | 0.6007 | 0.6495 |
| proposed_vs_baseline_no_context | context_relevance | 84 | 58 | 1 | 0.5909 | 0.5915 |
| proposed_vs_baseline_no_context | persona_consistency | 59 | 24 | 60 | 0.6224 | 0.7108 |
| proposed_vs_baseline_no_context | naturalness | 49 | 93 | 1 | 0.3462 | 0.3451 |
| proposed_vs_baseline_no_context | quest_state_correctness | 85 | 58 | 0 | 0.5944 | 0.5944 |
| proposed_vs_baseline_no_context | lore_consistency | 55 | 32 | 56 | 0.5804 | 0.6322 |
| proposed_vs_baseline_no_context | multi_turn_contradiction_safety | 14 | 17 | 112 | 0.4895 | 0.4516 |
| proposed_vs_baseline_no_context | objective_completion_support | 80 | 32 | 31 | 0.6678 | 0.7143 |
| proposed_vs_baseline_no_context | gameplay_usefulness | 97 | 46 | 0 | 0.6783 | 0.6783 |
| proposed_vs_baseline_no_context | time_pressure_acceptability | 75 | 53 | 15 | 0.5769 | 0.5859 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 56 | 21 | 66 | 0.6224 | 0.7273 |
| proposed_vs_baseline_no_context | context_overlap | 87 | 53 | 3 | 0.6189 | 0.6214 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 53 | 17 | 73 | 0.6259 | 0.7571 |
| proposed_vs_baseline_no_context | persona_style | 12 | 18 | 113 | 0.4790 | 0.4000 |
| proposed_vs_baseline_no_context | distinct1 | 25 | 95 | 23 | 0.2552 | 0.2083 |
| proposed_vs_baseline_no_context | length_score | 54 | 80 | 9 | 0.4091 | 0.4030 |
| proposed_vs_baseline_no_context | sentence_score | 33 | 12 | 98 | 0.5734 | 0.7333 |
| proposed_vs_baseline_no_context | overall_quality | 93 | 50 | 0 | 0.6503 | 0.6503 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 84 | 57 | 3 | 0.5938 | 0.5957 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | 56 | 24 | 64 | 0.6111 | 0.7000 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | 51 | 92 | 1 | 0.3576 | 0.3566 |
| proposed_vs_baseline_no_context_phi3_latest | quest_state_correctness | 85 | 56 | 3 | 0.6007 | 0.6028 |
| proposed_vs_baseline_no_context_phi3_latest | lore_consistency | 55 | 35 | 54 | 0.5694 | 0.6111 |
| proposed_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0 | 18 | 126 | 0.4375 | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | objective_completion_support | 79 | 34 | 31 | 0.6562 | 0.6991 |
| proposed_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 94 | 50 | 0 | 0.6528 | 0.6528 |
| proposed_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 73 | 56 | 15 | 0.5590 | 0.5659 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 57 | 18 | 69 | 0.6354 | 0.7600 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 85 | 56 | 3 | 0.6007 | 0.6028 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 50 | 18 | 76 | 0.6111 | 0.7353 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | 12 | 16 | 116 | 0.4861 | 0.4286 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | 22 | 102 | 20 | 0.2222 | 0.1774 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | 60 | 76 | 8 | 0.4444 | 0.4412 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | 41 | 14 | 89 | 0.5938 | 0.7455 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | 100 | 44 | 0 | 0.6944 | 0.6944 |
| controlled_vs_proposed_raw | context_relevance | 64 | 52 | 28 | 0.5417 | 0.5517 |
| controlled_vs_proposed_raw | persona_consistency | 77 | 14 | 53 | 0.7188 | 0.8462 |
| controlled_vs_proposed_raw | naturalness | 53 | 63 | 28 | 0.4653 | 0.4569 |
| controlled_vs_proposed_raw | quest_state_correctness | 63 | 53 | 28 | 0.5347 | 0.5431 |
| controlled_vs_proposed_raw | lore_consistency | 38 | 45 | 61 | 0.4757 | 0.4578 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 18 | 0 | 126 | 0.5625 | 1.0000 |
| controlled_vs_proposed_raw | objective_completion_support | 40 | 45 | 59 | 0.4826 | 0.4706 |
| controlled_vs_proposed_raw | gameplay_usefulness | 62 | 54 | 28 | 0.5278 | 0.5345 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 45 | 54 | 45 | 0.4688 | 0.4545 |
| controlled_vs_proposed_raw | context_keyword_coverage | 45 | 35 | 64 | 0.5347 | 0.5625 |
| controlled_vs_proposed_raw | context_overlap | 60 | 56 | 28 | 0.5139 | 0.5172 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 76 | 11 | 57 | 0.7257 | 0.8736 |
| controlled_vs_proposed_raw | persona_style | 6 | 17 | 121 | 0.4618 | 0.2609 |
| controlled_vs_proposed_raw | distinct1 | 61 | 47 | 36 | 0.5486 | 0.5648 |
| controlled_vs_proposed_raw | length_score | 46 | 66 | 32 | 0.4306 | 0.4107 |
| controlled_vs_proposed_raw | sentence_score | 23 | 5 | 116 | 0.5625 | 0.8214 |
| controlled_vs_proposed_raw | overall_quality | 79 | 37 | 28 | 0.6458 | 0.6810 |
| controlled_vs_candidate_no_context | context_relevance | 82 | 34 | 28 | 0.6667 | 0.7069 |
| controlled_vs_candidate_no_context | persona_consistency | 69 | 14 | 61 | 0.6910 | 0.8313 |
| controlled_vs_candidate_no_context | naturalness | 62 | 54 | 28 | 0.5278 | 0.5345 |
| controlled_vs_candidate_no_context | quest_state_correctness | 80 | 36 | 28 | 0.6528 | 0.6897 |
| controlled_vs_candidate_no_context | lore_consistency | 48 | 13 | 83 | 0.6215 | 0.7869 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 144 | 0.5000 | nan |
| controlled_vs_candidate_no_context | objective_completion_support | 51 | 32 | 61 | 0.5660 | 0.6145 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 86 | 30 | 28 | 0.6944 | 0.7414 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 46 | 51 | 47 | 0.4826 | 0.4742 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 61 | 11 | 72 | 0.6736 | 0.8472 |
| controlled_vs_candidate_no_context | context_overlap | 81 | 35 | 28 | 0.6597 | 0.6983 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 66 | 14 | 64 | 0.6806 | 0.8250 |
| controlled_vs_candidate_no_context | persona_style | 11 | 9 | 124 | 0.5069 | 0.5500 |
| controlled_vs_candidate_no_context | distinct1 | 56 | 54 | 34 | 0.5069 | 0.5091 |
| controlled_vs_candidate_no_context | length_score | 60 | 53 | 31 | 0.5243 | 0.5310 |
| controlled_vs_candidate_no_context | sentence_score | 14 | 5 | 125 | 0.5312 | 0.7368 |
| controlled_vs_candidate_no_context | overall_quality | 93 | 23 | 28 | 0.7431 | 0.8017 |
| controlled_vs_baseline_no_context | context_relevance | 89 | 53 | 1 | 0.6259 | 0.6268 |
| controlled_vs_baseline_no_context | persona_consistency | 107 | 12 | 24 | 0.8322 | 0.8992 |
| controlled_vs_baseline_no_context | naturalness | 47 | 95 | 1 | 0.3322 | 0.3310 |
| controlled_vs_baseline_no_context | quest_state_correctness | 87 | 55 | 1 | 0.6119 | 0.6127 |
| controlled_vs_baseline_no_context | lore_consistency | 48 | 39 | 56 | 0.5315 | 0.5517 |
| controlled_vs_baseline_no_context | multi_turn_contradiction_safety | 14 | 0 | 129 | 0.5490 | 1.0000 |
| controlled_vs_baseline_no_context | objective_completion_support | 83 | 29 | 31 | 0.6888 | 0.7411 |
| controlled_vs_baseline_no_context | gameplay_usefulness | 99 | 44 | 0 | 0.6923 | 0.6923 |
| controlled_vs_baseline_no_context | time_pressure_acceptability | 72 | 58 | 13 | 0.5490 | 0.5538 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 60 | 21 | 62 | 0.6364 | 0.7407 |
| controlled_vs_baseline_no_context | context_overlap | 86 | 55 | 2 | 0.6084 | 0.6099 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 107 | 7 | 29 | 0.8497 | 0.9386 |
| controlled_vs_baseline_no_context | persona_style | 8 | 18 | 117 | 0.4650 | 0.3077 |
| controlled_vs_baseline_no_context | distinct1 | 26 | 108 | 9 | 0.2133 | 0.1940 |
| controlled_vs_baseline_no_context | length_score | 48 | 89 | 6 | 0.3566 | 0.3504 |
| controlled_vs_baseline_no_context | sentence_score | 42 | 4 | 97 | 0.6329 | 0.9130 |
| controlled_vs_baseline_no_context | overall_quality | 117 | 26 | 0 | 0.8182 | 0.8182 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 91 | 50 | 3 | 0.6424 | 0.6454 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 107 | 12 | 25 | 0.8299 | 0.8992 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | 45 | 99 | 0 | 0.3125 | 0.3125 |
| controlled_vs_baseline_no_context_phi3_latest | quest_state_correctness | 93 | 48 | 3 | 0.6562 | 0.6596 |
| controlled_vs_baseline_no_context_phi3_latest | lore_consistency | 47 | 38 | 59 | 0.5312 | 0.5529 |
| controlled_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0 | 0 | 144 | 0.5000 | nan |
| controlled_vs_baseline_no_context_phi3_latest | objective_completion_support | 83 | 28 | 33 | 0.6910 | 0.7477 |
| controlled_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 107 | 37 | 0 | 0.7431 | 0.7431 |
| controlled_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 71 | 60 | 13 | 0.5382 | 0.5420 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 62 | 16 | 66 | 0.6597 | 0.7949 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 86 | 54 | 4 | 0.6111 | 0.6143 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 107 | 8 | 29 | 0.8438 | 0.9304 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | 8 | 16 | 120 | 0.4722 | 0.3333 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | 22 | 109 | 13 | 0.1979 | 0.1679 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | 58 | 80 | 6 | 0.4236 | 0.4203 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 49 | 4 | 91 | 0.6562 | 0.9245 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 117 | 27 | 0 | 0.8125 | 0.8125 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 89 | 53 | 1 | 0.6259 | 0.6268 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 107 | 12 | 24 | 0.8322 | 0.8992 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 47 | 95 | 1 | 0.3322 | 0.3310 |
| proposed_contextual_controlled_vs_baseline_no_context | quest_state_correctness | 87 | 55 | 1 | 0.6119 | 0.6127 |
| proposed_contextual_controlled_vs_baseline_no_context | lore_consistency | 48 | 39 | 56 | 0.5315 | 0.5517 |
| proposed_contextual_controlled_vs_baseline_no_context | multi_turn_contradiction_safety | 14 | 0 | 129 | 0.5490 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | objective_completion_support | 83 | 29 | 31 | 0.6888 | 0.7411 |
| proposed_contextual_controlled_vs_baseline_no_context | gameplay_usefulness | 99 | 44 | 0 | 0.6923 | 0.6923 |
| proposed_contextual_controlled_vs_baseline_no_context | time_pressure_acceptability | 72 | 58 | 13 | 0.5490 | 0.5538 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 60 | 21 | 62 | 0.6364 | 0.7407 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 86 | 55 | 2 | 0.6084 | 0.6099 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 107 | 7 | 29 | 0.8497 | 0.9386 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 8 | 18 | 117 | 0.4650 | 0.3077 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | 26 | 108 | 9 | 0.2133 | 0.1940 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 48 | 89 | 6 | 0.3566 | 0.3504 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 42 | 4 | 97 | 0.6329 | 0.9130 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 117 | 26 | 0 | 0.8182 | 0.8182 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 91 | 50 | 3 | 0.6424 | 0.6454 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 107 | 12 | 25 | 0.8299 | 0.8992 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | 45 | 99 | 0 | 0.3125 | 0.3125 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | quest_state_correctness | 93 | 48 | 3 | 0.6562 | 0.6596 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | lore_consistency | 47 | 38 | 59 | 0.5312 | 0.5529 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0 | 0 | 144 | 0.5000 | nan |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | objective_completion_support | 83 | 28 | 33 | 0.6910 | 0.7477 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 107 | 37 | 0 | 0.7431 | 0.7431 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 71 | 60 | 13 | 0.5382 | 0.5420 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 62 | 16 | 66 | 0.6597 | 0.7949 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 86 | 54 | 4 | 0.6111 | 0.6143 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 107 | 8 | 29 | 0.8438 | 0.9304 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | 8 | 16 | 120 | 0.4722 | 0.3333 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | 22 | 109 | 13 | 0.1979 | 0.1679 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | 58 | 80 | 6 | 0.4236 | 0.4203 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | 49 | 4 | 91 | 0.6562 | 0.9245 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 117 | 27 | 0 | 0.8125 | 0.8125 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.2361 | 0.1944 | 0.8056 |
| proposed_contextual | 0.0000 | 0.0000 | 0.4444 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.4653 | 0.0000 | 0.0000 |
| baseline_no_context | 0.0069 | 0.0069 | 0.0069 | 0.0000 | 0.0000 |
| baseline_no_context_phi3_latest | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `8`
- Unique template signatures: `143`
- Template signature ratio: `0.9931`
- Effective sample size by source clustering: `8.00`
- Effective sample size by template-signature clustering: `142.03`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: unavailable (No BERTScore values found in merged scores.).

This report covers proposal RO5 metrics plus game-facing outcomes: quest-state correctness, lore consistency, contradiction safety, objective completion support, gameplay usefulness, and time-pressure acceptability.
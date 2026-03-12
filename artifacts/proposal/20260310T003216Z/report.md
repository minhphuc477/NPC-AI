# Proposal Alignment Evaluation Report

- Run ID: `20260310T003216Z`
- Generated: `2026-03-10T00:33:59.461647+00:00`
- Scenarios: `artifacts\proposal\20260310T003216Z\scenarios.jsonl`
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
| proposed_contextual_controlled | 0.2706 (0.2558, 0.2860) | 0.2972 (0.2794, 0.3151) | 0.9003 (0.8900, 0.9100) | 0.4079 (0.3990, 0.4170) | n/a |
| proposed_contextual | 0.0830 (0.0661, 0.1014) | 0.1442 (0.1259, 0.1637) | 0.8193 (0.8072, 0.8313) | 0.2475 (0.2338, 0.2629) | n/a |
| candidate_no_context | 0.0268 (0.0206, 0.0333) | 0.1631 (0.1419, 0.1875) | 0.8289 (0.8157, 0.8426) | 0.2306 (0.2201, 0.2424) | n/a |
| baseline_no_context | 0.0360 (0.0292, 0.0426) | 0.1585 (0.1435, 0.1742) | 0.8936 (0.8836, 0.9035) | 0.2455 (0.2385, 0.2528) | n/a |
| baseline_no_context_phi3_latest | 0.0313 (0.0252, 0.0377) | 0.1530 (0.1396, 0.1668) | 0.8913 (0.8819, 0.9000) | 0.2409 (0.2340, 0.2481) | n/a |

## Game-facing Outcome Metrics (mean, 95% CI)
| Arm | Quest-state Correctness | Lore Consistency | Contradiction Safety | Objective Completion Support | Gameplay Usefulness | Time-pressure Acceptability |
|---|---:|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.3241 (0.3110, 0.3377) | 0.1897 (0.1760, 0.2036) | 1.0000 (1.0000, 1.0000) | 0.0594 (0.0487, 0.0712) | 0.3558 (0.3481, 0.3638) | 0.2886 (0.2798, 0.2973) |
| proposed_contextual | 0.1715 (0.1573, 0.1866) | 0.0458 (0.0358, 0.0567) | 1.0000 (1.0000, 1.0000) | 0.0430 (0.0377, 0.0490) | 0.2575 (0.2495, 0.2656) | 0.2784 (0.2730, 0.2837) |
| candidate_no_context | 0.1229 (0.1176, 0.1284) | 0.0140 (0.0107, 0.0179) | 1.0000 (1.0000, 1.0000) | 0.0435 (0.0376, 0.0497) | 0.2461 (0.2401, 0.2524) | 0.2821 (0.2774, 0.2870) |
| baseline_no_context | 0.1306 (0.1246, 0.1371) | 0.0128 (0.0095, 0.0166) | 1.0000 (1.0000, 1.0000) | 0.0398 (0.0334, 0.0465) | 0.2766 (0.2712, 0.2823) | 0.2881 (0.2827, 0.2939) |
| baseline_no_context_phi3_latest | 0.1268 (0.1211, 0.1328) | 0.0154 (0.0111, 0.0201) | 0.8750 (0.8194, 0.9236) | 0.0413 (0.0336, 0.0494) | 0.2749 (0.2690, 0.2809) | 0.2894 (0.2836, 0.2957) |

- Multi-turn contradiction rate is reported as `1 - contradiction_safety` in row-level outputs.

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0562 | 2.0939 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0189 | -0.1161 |
| proposed_vs_candidate_no_context | naturalness | -0.0097 | -0.0117 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.0486 | 0.3958 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0319 | 2.2813 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | -0.0005 | -0.0105 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 0.0114 | 0.0462 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | -0.0038 | -0.0134 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0723 | 3.0987 |
| proposed_vs_candidate_no_context | context_overlap | 0.0185 | 0.5297 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0255 | -0.3488 |
| proposed_vs_candidate_no_context | persona_style | 0.0074 | 0.0142 |
| proposed_vs_candidate_no_context | distinct1 | -0.0064 | -0.0068 |
| proposed_vs_candidate_no_context | length_score | -0.0245 | -0.0705 |
| proposed_vs_candidate_no_context | sentence_score | -0.0219 | -0.0272 |
| proposed_vs_candidate_no_context | overall_quality | 0.0169 | 0.0732 |
| proposed_vs_baseline_no_context | context_relevance | 0.0470 | 1.3037 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0143 | -0.0903 |
| proposed_vs_baseline_no_context | naturalness | -0.0744 | -0.0832 |
| proposed_vs_baseline_no_context | quest_state_correctness | 0.0409 | 0.3130 |
| proposed_vs_baseline_no_context | lore_consistency | 0.0330 | 2.5674 |
| proposed_vs_baseline_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| proposed_vs_baseline_no_context | objective_completion_support | 0.0032 | 0.0814 |
| proposed_vs_baseline_no_context | gameplay_usefulness | -0.0191 | -0.0690 |
| proposed_vs_baseline_no_context | time_pressure_acceptability | -0.0097 | -0.0338 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0598 | 1.6695 |
| proposed_vs_baseline_no_context | context_overlap | 0.0170 | 0.4658 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0081 | -0.1458 |
| proposed_vs_baseline_no_context | persona_style | -0.0390 | -0.0685 |
| proposed_vs_baseline_no_context | distinct1 | -0.0371 | -0.0380 |
| proposed_vs_baseline_no_context | length_score | -0.2498 | -0.4356 |
| proposed_vs_baseline_no_context | sentence_score | -0.0955 | -0.1086 |
| proposed_vs_baseline_no_context | overall_quality | 0.0021 | 0.0084 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 0.0517 | 1.6521 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | -0.0088 | -0.0574 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | -0.0720 | -0.0808 |
| proposed_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.0447 | 0.3530 |
| proposed_vs_baseline_no_context_phi3_latest | lore_consistency | 0.0304 | 1.9764 |
| proposed_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0.1250 | 0.1429 |
| proposed_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.0017 | 0.0422 |
| proposed_vs_baseline_no_context_phi3_latest | gameplay_usefulness | -0.0175 | -0.0635 |
| proposed_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | -0.0111 | -0.0383 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.0660 | 2.2183 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 0.0185 | 0.5295 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | -0.0034 | -0.0667 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | -0.0303 | -0.0540 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | -0.0409 | -0.0417 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | -0.2310 | -0.4165 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | -0.0948 | -0.1079 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | 0.0067 | 0.0276 |
| controlled_vs_proposed_raw | context_relevance | 0.1876 | 2.2589 |
| controlled_vs_proposed_raw | persona_consistency | 0.1530 | 1.0613 |
| controlled_vs_proposed_raw | naturalness | 0.0810 | 0.0989 |
| controlled_vs_proposed_raw | quest_state_correctness | 0.1526 | 0.8896 |
| controlled_vs_proposed_raw | lore_consistency | 0.1438 | 3.1374 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0163 | 0.3799 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0983 | 0.3819 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 0.0103 | 0.0369 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2454 | 2.5645 |
| controlled_vs_proposed_raw | context_overlap | 0.0527 | 0.9842 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1845 | 3.8723 |
| controlled_vs_proposed_raw | persona_style | 0.0270 | 0.0508 |
| controlled_vs_proposed_raw | distinct1 | -0.0044 | -0.0047 |
| controlled_vs_proposed_raw | length_score | 0.3426 | 1.0587 |
| controlled_vs_proposed_raw | sentence_score | 0.1438 | 0.1834 |
| controlled_vs_proposed_raw | overall_quality | 0.1603 | 0.6478 |
| controlled_vs_candidate_no_context | context_relevance | 0.2438 | 9.0828 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1341 | 0.8219 |
| controlled_vs_candidate_no_context | naturalness | 0.0713 | 0.0861 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.2012 | 1.6374 |
| controlled_vs_candidate_no_context | lore_consistency | 0.1757 | 12.5762 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | 0.0159 | 0.3654 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.1097 | 0.4457 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 0.0065 | 0.0230 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3178 | 13.6097 |
| controlled_vs_candidate_no_context | context_overlap | 0.0712 | 2.0351 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1590 | 2.1726 |
| controlled_vs_candidate_no_context | persona_style | 0.0344 | 0.0657 |
| controlled_vs_candidate_no_context | distinct1 | -0.0108 | -0.0114 |
| controlled_vs_candidate_no_context | length_score | 0.3181 | 0.9136 |
| controlled_vs_candidate_no_context | sentence_score | 0.1219 | 0.1513 |
| controlled_vs_candidate_no_context | overall_quality | 0.1772 | 0.7684 |
| controlled_vs_baseline_no_context | context_relevance | 0.2346 | 6.5075 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1387 | 0.8752 |
| controlled_vs_baseline_no_context | naturalness | 0.0066 | 0.0074 |
| controlled_vs_baseline_no_context | quest_state_correctness | 0.1934 | 1.4810 |
| controlled_vs_baseline_no_context | lore_consistency | 0.1768 | 13.7598 |
| controlled_vs_baseline_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_baseline_no_context | objective_completion_support | 0.0196 | 0.4922 |
| controlled_vs_baseline_no_context | gameplay_usefulness | 0.0792 | 0.2865 |
| controlled_vs_baseline_no_context | time_pressure_acceptability | 0.0005 | 0.0019 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3053 | 8.5154 |
| controlled_vs_baseline_no_context | context_overlap | 0.0697 | 1.9084 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1764 | 3.1618 |
| controlled_vs_baseline_no_context | persona_style | -0.0120 | -0.0212 |
| controlled_vs_baseline_no_context | distinct1 | -0.0415 | -0.0425 |
| controlled_vs_baseline_no_context | length_score | 0.0928 | 0.1619 |
| controlled_vs_baseline_no_context | sentence_score | 0.0483 | 0.0549 |
| controlled_vs_baseline_no_context | overall_quality | 0.1624 | 0.6616 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.2393 | 7.6428 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1442 | 0.9430 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | 0.0090 | 0.0101 |
| controlled_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.1973 | 1.5567 |
| controlled_vs_baseline_no_context_phi3_latest | lore_consistency | 0.1743 | 11.3148 |
| controlled_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0.1250 | 0.1429 |
| controlled_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.0181 | 0.4381 |
| controlled_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 0.0809 | 0.2941 |
| controlled_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | -0.0008 | -0.0028 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.3114 | 10.4717 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0712 | 2.0348 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1811 | 3.5473 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | -0.0033 | -0.0059 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0453 | -0.0461 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | 0.1116 | 0.2012 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.0490 | 0.0557 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1670 | 0.6933 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2346 | 6.5075 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1387 | 0.8752 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 0.0066 | 0.0074 |
| proposed_contextual_controlled_vs_baseline_no_context | quest_state_correctness | 0.1934 | 1.4810 |
| proposed_contextual_controlled_vs_baseline_no_context | lore_consistency | 0.1768 | 13.7598 |
| proposed_contextual_controlled_vs_baseline_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | objective_completion_support | 0.0196 | 0.4922 |
| proposed_contextual_controlled_vs_baseline_no_context | gameplay_usefulness | 0.0792 | 0.2865 |
| proposed_contextual_controlled_vs_baseline_no_context | time_pressure_acceptability | 0.0005 | 0.0019 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3053 | 8.5154 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0697 | 1.9084 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1764 | 3.1618 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | -0.0120 | -0.0212 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0415 | -0.0425 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.0928 | 0.1619 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0483 | 0.0549 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1624 | 0.6616 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.2393 | 7.6428 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1442 | 0.9430 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | 0.0090 | 0.0101 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.1973 | 1.5567 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | lore_consistency | 0.1743 | 11.3148 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0.1250 | 0.1429 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.0181 | 0.4381 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 0.0809 | 0.2941 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | -0.0008 | -0.0028 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.3114 | 10.4717 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0712 | 2.0348 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1811 | 3.5473 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | -0.0033 | -0.0059 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0453 | -0.0461 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | 0.1116 | 0.2012 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.0490 | 0.0557 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1670 | 0.6933 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0562 | (0.0386, 0.0741) | 0.0000 | 0.0562 | (0.0318, 0.0854) | 0.0000 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0189 | (-0.0415, 0.0020) | 0.9580 | -0.0189 | (-0.0563, 0.0106) | 0.8807 |
| proposed_vs_candidate_no_context | naturalness | -0.0097 | (-0.0243, 0.0052) | 0.8953 | -0.0097 | (-0.0210, 0.0016) | 0.9513 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.0486 | (0.0335, 0.0642) | 0.0000 | 0.0486 | (0.0240, 0.0729) | 0.0000 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0319 | (0.0215, 0.0427) | 0.0000 | 0.0319 | (0.0188, 0.0456) | 0.0000 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | -0.0005 | (-0.0078, 0.0067) | 0.5517 | -0.0005 | (-0.0100, 0.0083) | 0.5260 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 0.0114 | (0.0024, 0.0199) | 0.0090 | 0.0114 | (0.0034, 0.0209) | 0.0010 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | -0.0038 | (-0.0096, 0.0021) | 0.8950 | -0.0038 | (-0.0094, 0.0024) | 0.8917 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0723 | (0.0508, 0.0960) | 0.0000 | 0.0723 | (0.0399, 0.1107) | 0.0000 |
| proposed_vs_candidate_no_context | context_overlap | 0.0185 | (0.0126, 0.0241) | 0.0000 | 0.0185 | (0.0119, 0.0248) | 0.0000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0255 | (-0.0521, -0.0003) | 0.9763 | -0.0255 | (-0.0679, 0.0103) | 0.8960 |
| proposed_vs_candidate_no_context | persona_style | 0.0074 | (-0.0083, 0.0231) | 0.1797 | 0.0074 | (-0.0046, 0.0227) | 0.1677 |
| proposed_vs_candidate_no_context | distinct1 | -0.0064 | (-0.0139, 0.0012) | 0.9553 | -0.0064 | (-0.0098, -0.0035) | 1.0000 |
| proposed_vs_candidate_no_context | length_score | -0.0245 | (-0.0782, 0.0294) | 0.8127 | -0.0245 | (-0.0708, 0.0195) | 0.8550 |
| proposed_vs_candidate_no_context | sentence_score | -0.0219 | (-0.0559, 0.0122) | 0.9163 | -0.0219 | (-0.0681, 0.0316) | 0.8167 |
| proposed_vs_candidate_no_context | overall_quality | 0.0169 | (0.0025, 0.0308) | 0.0123 | 0.0169 | (-0.0002, 0.0401) | 0.0270 |
| proposed_vs_baseline_no_context | context_relevance | 0.0470 | (0.0276, 0.0650) | 0.0000 | 0.0470 | (0.0129, 0.0843) | 0.0010 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0143 | (-0.0349, 0.0070) | 0.9120 | -0.0143 | (-0.0425, 0.0143) | 0.8407 |
| proposed_vs_baseline_no_context | naturalness | -0.0744 | (-0.0883, -0.0605) | 1.0000 | -0.0744 | (-0.0989, -0.0545) | 1.0000 |
| proposed_vs_baseline_no_context | quest_state_correctness | 0.0409 | (0.0256, 0.0575) | 0.0000 | 0.0409 | (0.0114, 0.0720) | 0.0013 |
| proposed_vs_baseline_no_context | lore_consistency | 0.0330 | (0.0235, 0.0437) | 0.0000 | 0.0330 | (0.0214, 0.0449) | 0.0000 |
| proposed_vs_baseline_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_baseline_no_context | objective_completion_support | 0.0032 | (-0.0041, 0.0101) | 0.1867 | 0.0032 | (-0.0028, 0.0104) | 0.1770 |
| proposed_vs_baseline_no_context | gameplay_usefulness | -0.0191 | (-0.0283, -0.0096) | 1.0000 | -0.0191 | (-0.0347, -0.0053) | 1.0000 |
| proposed_vs_baseline_no_context | time_pressure_acceptability | -0.0097 | (-0.0165, -0.0031) | 0.9997 | -0.0097 | (-0.0174, -0.0007) | 0.9827 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0598 | (0.0363, 0.0874) | 0.0000 | 0.0598 | (0.0156, 0.1095) | 0.0023 |
| proposed_vs_baseline_no_context | context_overlap | 0.0170 | (0.0109, 0.0233) | 0.0000 | 0.0170 | (0.0085, 0.0258) | 0.0000 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0081 | (-0.0318, 0.0153) | 0.7637 | -0.0081 | (-0.0351, 0.0242) | 0.7123 |
| proposed_vs_baseline_no_context | persona_style | -0.0390 | (-0.0666, -0.0136) | 0.9997 | -0.0390 | (-0.0928, 0.0087) | 0.9217 |
| proposed_vs_baseline_no_context | distinct1 | -0.0371 | (-0.0447, -0.0297) | 1.0000 | -0.0371 | (-0.0498, -0.0250) | 1.0000 |
| proposed_vs_baseline_no_context | length_score | -0.2498 | (-0.3005, -0.1981) | 1.0000 | -0.2498 | (-0.3322, -0.1824) | 1.0000 |
| proposed_vs_baseline_no_context | sentence_score | -0.0955 | (-0.1316, -0.0569) | 1.0000 | -0.0955 | (-0.1531, -0.0455) | 1.0000 |
| proposed_vs_baseline_no_context | overall_quality | 0.0021 | (-0.0127, 0.0168) | 0.3960 | 0.0021 | (-0.0246, 0.0301) | 0.4537 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 0.0517 | (0.0332, 0.0717) | 0.0000 | 0.0517 | (0.0167, 0.0873) | 0.0010 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | -0.0088 | (-0.0279, 0.0116) | 0.8083 | -0.0088 | (-0.0380, 0.0223) | 0.7037 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | -0.0720 | (-0.0871, -0.0572) | 1.0000 | -0.0720 | (-0.0996, -0.0464) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.0447 | (0.0297, 0.0606) | 0.0000 | 0.0447 | (0.0150, 0.0744) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | lore_consistency | 0.0304 | (0.0198, 0.0413) | 0.0000 | 0.0304 | (0.0160, 0.0460) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0.1250 | (0.0764, 0.1806) | 0.0000 | 0.1250 | (0.0000, 0.3750) | 0.3337 |
| proposed_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.0017 | (-0.0066, 0.0099) | 0.3403 | 0.0017 | (-0.0056, 0.0085) | 0.3123 |
| proposed_vs_baseline_no_context_phi3_latest | gameplay_usefulness | -0.0175 | (-0.0272, -0.0076) | 1.0000 | -0.0175 | (-0.0339, -0.0023) | 0.9860 |
| proposed_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | -0.0111 | (-0.0186, -0.0038) | 0.9990 | -0.0111 | (-0.0166, -0.0062) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.0660 | (0.0423, 0.0912) | 0.0000 | 0.0660 | (0.0228, 0.1111) | 0.0017 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 0.0185 | (0.0123, 0.0247) | 0.0000 | 0.0185 | (0.0082, 0.0285) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | -0.0034 | (-0.0253, 0.0198) | 0.6173 | -0.0034 | (-0.0375, 0.0293) | 0.5843 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | -0.0303 | (-0.0571, -0.0051) | 0.9910 | -0.0303 | (-0.0872, 0.0167) | 0.8597 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | -0.0409 | (-0.0476, -0.0340) | 1.0000 | -0.0409 | (-0.0528, -0.0289) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | -0.2310 | (-0.2894, -0.1738) | 1.0000 | -0.2310 | (-0.3181, -0.1507) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | -0.0948 | (-0.1337, -0.0559) | 1.0000 | -0.0948 | (-0.1726, -0.0146) | 0.9893 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | 0.0067 | (-0.0086, 0.0224) | 0.2043 | 0.0067 | (-0.0215, 0.0364) | 0.3390 |
| controlled_vs_proposed_raw | context_relevance | 0.1876 | (0.1670, 0.2089) | 0.0000 | 0.1876 | (0.1560, 0.2173) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.1530 | (0.1325, 0.1733) | 0.0000 | 0.1530 | (0.1186, 0.1839) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0810 | (0.0648, 0.0977) | 0.0000 | 0.0810 | (0.0495, 0.1136) | 0.0000 |
| controlled_vs_proposed_raw | quest_state_correctness | 0.1526 | (0.1338, 0.1702) | 0.0000 | 0.1526 | (0.1236, 0.1805) | 0.0000 |
| controlled_vs_proposed_raw | lore_consistency | 0.1438 | (0.1280, 0.1594) | 0.0000 | 0.1438 | (0.1265, 0.1619) | 0.0000 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0163 | (0.0064, 0.0274) | 0.0010 | 0.0163 | (0.0022, 0.0341) | 0.0073 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0983 | (0.0867, 0.1102) | 0.0000 | 0.0983 | (0.0803, 0.1187) | 0.0000 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 0.0103 | (0.0013, 0.0188) | 0.0113 | 0.0103 | (0.0004, 0.0223) | 0.0200 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2454 | (0.2179, 0.2733) | 0.0000 | 0.2454 | (0.2045, 0.2853) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0527 | (0.0448, 0.0606) | 0.0000 | 0.0527 | (0.0445, 0.0602) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1845 | (0.1601, 0.2094) | 0.0000 | 0.1845 | (0.1429, 0.2242) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.0270 | (0.0078, 0.0492) | 0.0017 | 0.0270 | (0.0017, 0.0669) | 0.0053 |
| controlled_vs_proposed_raw | distinct1 | -0.0044 | (-0.0119, 0.0026) | 0.8810 | -0.0044 | (-0.0155, 0.0078) | 0.7560 |
| controlled_vs_proposed_raw | length_score | 0.3426 | (0.2808, 0.4044) | 0.0000 | 0.3426 | (0.2303, 0.4488) | 0.0000 |
| controlled_vs_proposed_raw | sentence_score | 0.1437 | (0.1076, 0.1799) | 0.0000 | 0.1437 | (0.0684, 0.2215) | 0.0000 |
| controlled_vs_proposed_raw | overall_quality | 0.1603 | (0.1452, 0.1752) | 0.0000 | 0.1603 | (0.1384, 0.1807) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2438 | (0.2295, 0.2578) | 0.0000 | 0.2438 | (0.2228, 0.2653) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1341 | (0.1123, 0.1537) | 0.0000 | 0.1341 | (0.1030, 0.1647) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.0713 | (0.0520, 0.0897) | 0.0000 | 0.0713 | (0.0367, 0.1067) | 0.0000 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.2012 | (0.1892, 0.2136) | 0.0000 | 0.2012 | (0.1836, 0.2182) | 0.0000 |
| controlled_vs_candidate_no_context | lore_consistency | 0.1757 | (0.1618, 0.1893) | 0.0000 | 0.1757 | (0.1514, 0.2012) | 0.0000 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | 0.0159 | (0.0045, 0.0278) | 0.0033 | 0.0159 | (-0.0046, 0.0423) | 0.0807 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.1097 | (0.0994, 0.1200) | 0.0000 | 0.1097 | (0.0916, 0.1285) | 0.0000 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 0.0065 | (-0.0023, 0.0153) | 0.0780 | 0.0065 | (-0.0047, 0.0221) | 0.1943 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3178 | (0.2988, 0.3375) | 0.0000 | 0.3178 | (0.2876, 0.3458) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0712 | (0.0648, 0.0780) | 0.0000 | 0.0712 | (0.0653, 0.0769) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1590 | (0.1343, 0.1818) | 0.0000 | 0.1590 | (0.1215, 0.1977) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.0344 | (0.0096, 0.0610) | 0.0013 | 0.0344 | (0.0041, 0.0795) | 0.0043 |
| controlled_vs_candidate_no_context | distinct1 | -0.0108 | (-0.0194, -0.0025) | 0.9940 | -0.0108 | (-0.0233, 0.0020) | 0.9510 |
| controlled_vs_candidate_no_context | length_score | 0.3181 | (0.2447, 0.3907) | 0.0000 | 0.3181 | (0.1873, 0.4403) | 0.0000 |
| controlled_vs_candidate_no_context | sentence_score | 0.1219 | (0.0830, 0.1604) | 0.0000 | 0.1219 | (0.0583, 0.1826) | 0.0003 |
| controlled_vs_candidate_no_context | overall_quality | 0.1772 | (0.1661, 0.1882) | 0.0000 | 0.1772 | (0.1589, 0.1940) | 0.0000 |
| controlled_vs_baseline_no_context | context_relevance | 0.2346 | (0.2203, 0.2497) | 0.0000 | 0.2346 | (0.2176, 0.2532) | 0.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1387 | (0.1175, 0.1593) | 0.0000 | 0.1387 | (0.0950, 0.1785) | 0.0000 |
| controlled_vs_baseline_no_context | naturalness | 0.0066 | (-0.0071, 0.0210) | 0.1883 | 0.0066 | (-0.0102, 0.0238) | 0.2323 |
| controlled_vs_baseline_no_context | quest_state_correctness | 0.1934 | (0.1807, 0.2067) | 0.0000 | 0.1934 | (0.1789, 0.2075) | 0.0000 |
| controlled_vs_baseline_no_context | lore_consistency | 0.1768 | (0.1636, 0.1906) | 0.0000 | 0.1768 | (0.1545, 0.1999) | 0.0000 |
| controlled_vs_baseline_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_baseline_no_context | objective_completion_support | 0.0196 | (0.0081, 0.0314) | 0.0003 | 0.0196 | (0.0009, 0.0443) | 0.0180 |
| controlled_vs_baseline_no_context | gameplay_usefulness | 0.0792 | (0.0700, 0.0883) | 0.0000 | 0.0792 | (0.0661, 0.0962) | 0.0000 |
| controlled_vs_baseline_no_context | time_pressure_acceptability | 0.0005 | (-0.0091, 0.0102) | 0.4550 | 0.0005 | (-0.0136, 0.0185) | 0.5000 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3053 | (0.2860, 0.3247) | 0.0000 | 0.3053 | (0.2811, 0.3302) | 0.0000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0697 | (0.0629, 0.0768) | 0.0000 | 0.0697 | (0.0624, 0.0765) | 0.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1764 | (0.1522, 0.2000) | 0.0000 | 0.1764 | (0.1262, 0.2209) | 0.0000 |
| controlled_vs_baseline_no_context | persona_style | -0.0120 | (-0.0347, 0.0072) | 0.8713 | -0.0120 | (-0.0552, 0.0183) | 0.6697 |
| controlled_vs_baseline_no_context | distinct1 | -0.0415 | (-0.0482, -0.0351) | 1.0000 | -0.0415 | (-0.0503, -0.0331) | 1.0000 |
| controlled_vs_baseline_no_context | length_score | 0.0928 | (0.0326, 0.1519) | 0.0017 | 0.0928 | (0.0215, 0.1604) | 0.0070 |
| controlled_vs_baseline_no_context | sentence_score | 0.0483 | (0.0125, 0.0851) | 0.0033 | 0.0483 | (-0.0024, 0.0997) | 0.0363 |
| controlled_vs_baseline_no_context | overall_quality | 0.1624 | (0.1521, 0.1731) | 0.0000 | 0.1624 | (0.1461, 0.1768) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.2393 | (0.2245, 0.2542) | 0.0000 | 0.2393 | (0.2188, 0.2608) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1442 | (0.1243, 0.1633) | 0.0000 | 0.1442 | (0.0944, 0.1882) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | 0.0090 | (-0.0054, 0.0228) | 0.1037 | 0.0090 | (-0.0015, 0.0192) | 0.0493 |
| controlled_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.1973 | (0.1847, 0.2098) | 0.0000 | 0.1973 | (0.1808, 0.2135) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | lore_consistency | 0.1743 | (0.1604, 0.1886) | 0.0000 | 0.1743 | (0.1500, 0.1995) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0.1250 | (0.0764, 0.1806) | 0.0000 | 0.1250 | (0.0000, 0.3750) | 0.3320 |
| controlled_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.0181 | (0.0057, 0.0302) | 0.0010 | 0.0181 | (-0.0011, 0.0399) | 0.0343 |
| controlled_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 0.0809 | (0.0718, 0.0904) | 0.0000 | 0.0809 | (0.0718, 0.0921) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | -0.0008 | (-0.0095, 0.0083) | 0.5627 | -0.0008 | (-0.0097, 0.0114) | 0.5927 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.3114 | (0.2921, 0.3309) | 0.0000 | 0.3114 | (0.2839, 0.3389) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0712 | (0.0644, 0.0785) | 0.0000 | 0.0712 | (0.0628, 0.0790) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1811 | (0.1573, 0.2031) | 0.0000 | 0.1811 | (0.1209, 0.2298) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | -0.0033 | (-0.0216, 0.0156) | 0.6387 | -0.0033 | (-0.0426, 0.0268) | 0.5663 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0453 | (-0.0513, -0.0389) | 1.0000 | -0.0453 | (-0.0530, -0.0364) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | 0.1116 | (0.0463, 0.1727) | 0.0003 | 0.1116 | (0.0567, 0.1597) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.0490 | (0.0149, 0.0854) | 0.0033 | 0.0490 | (0.0000, 0.0997) | 0.0263 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1670 | (0.1562, 0.1772) | 0.0000 | 0.1670 | (0.1461, 0.1838) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2346 | (0.2199, 0.2499) | 0.0000 | 0.2346 | (0.2173, 0.2526) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1387 | (0.1179, 0.1581) | 0.0000 | 0.1387 | (0.0984, 0.1788) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 0.0066 | (-0.0071, 0.0204) | 0.1703 | 0.0066 | (-0.0107, 0.0239) | 0.2203 |
| proposed_contextual_controlled_vs_baseline_no_context | quest_state_correctness | 0.1934 | (0.1809, 0.2068) | 0.0000 | 0.1934 | (0.1787, 0.2077) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | lore_consistency | 0.1768 | (0.1630, 0.1903) | 0.0000 | 0.1768 | (0.1543, 0.1999) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | objective_completion_support | 0.0196 | (0.0081, 0.0320) | 0.0000 | 0.0196 | (0.0008, 0.0432) | 0.0207 |
| proposed_contextual_controlled_vs_baseline_no_context | gameplay_usefulness | 0.0792 | (0.0698, 0.0888) | 0.0000 | 0.0792 | (0.0660, 0.0964) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | time_pressure_acceptability | 0.0005 | (-0.0092, 0.0100) | 0.4627 | 0.0005 | (-0.0132, 0.0183) | 0.5063 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3053 | (0.2861, 0.3245) | 0.0000 | 0.3053 | (0.2816, 0.3303) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0697 | (0.0631, 0.0769) | 0.0000 | 0.0697 | (0.0627, 0.0767) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1764 | (0.1517, 0.2002) | 0.0000 | 0.1764 | (0.1275, 0.2216) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | -0.0120 | (-0.0331, 0.0075) | 0.8777 | -0.0120 | (-0.0552, 0.0183) | 0.6687 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0415 | (-0.0478, -0.0349) | 1.0000 | -0.0415 | (-0.0504, -0.0336) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.0928 | (0.0310, 0.1535) | 0.0010 | 0.0928 | (0.0206, 0.1581) | 0.0050 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0483 | (0.0125, 0.0837) | 0.0047 | 0.0483 | (-0.0024, 0.0993) | 0.0350 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1624 | (0.1517, 0.1734) | 0.0000 | 0.1624 | (0.1448, 0.1762) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.2393 | (0.2253, 0.2547) | 0.0000 | 0.2393 | (0.2184, 0.2595) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1442 | (0.1250, 0.1631) | 0.0000 | 0.1442 | (0.0944, 0.1886) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | 0.0090 | (-0.0056, 0.0229) | 0.1120 | 0.0090 | (-0.0015, 0.0193) | 0.0460 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.1973 | (0.1850, 0.2099) | 0.0000 | 0.1973 | (0.1815, 0.2134) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | lore_consistency | 0.1743 | (0.1605, 0.1879) | 0.0000 | 0.1743 | (0.1495, 0.1990) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0.1250 | (0.0764, 0.1806) | 0.0000 | 0.1250 | (0.0000, 0.3750) | 0.3363 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.0181 | (0.0060, 0.0301) | 0.0020 | 0.0181 | (-0.0003, 0.0391) | 0.0297 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 0.0809 | (0.0714, 0.0904) | 0.0000 | 0.0809 | (0.0724, 0.0914) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | -0.0008 | (-0.0099, 0.0075) | 0.5743 | -0.0008 | (-0.0098, 0.0118) | 0.6007 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.3114 | (0.2918, 0.3313) | 0.0000 | 0.3114 | (0.2830, 0.3403) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0712 | (0.0643, 0.0783) | 0.0000 | 0.0712 | (0.0630, 0.0787) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1811 | (0.1588, 0.2024) | 0.0000 | 0.1811 | (0.1245, 0.2312) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | -0.0033 | (-0.0214, 0.0154) | 0.6317 | -0.0033 | (-0.0426, 0.0268) | 0.5497 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0453 | (-0.0514, -0.0389) | 1.0000 | -0.0453 | (-0.0529, -0.0366) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | 0.1116 | (0.0472, 0.1718) | 0.0003 | 0.1116 | (0.0581, 0.1607) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.0490 | (0.0146, 0.0851) | 0.0050 | 0.0490 | (0.0003, 0.0976) | 0.0250 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1670 | (0.1563, 0.1775) | 0.0000 | 0.1670 | (0.1463, 0.1835) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 76 | 25 | 43 | 0.6771 | 0.7525 |
| proposed_vs_candidate_no_context | persona_consistency | 18 | 24 | 102 | 0.4792 | 0.4286 |
| proposed_vs_candidate_no_context | naturalness | 44 | 57 | 43 | 0.4549 | 0.4356 |
| proposed_vs_candidate_no_context | quest_state_correctness | 75 | 26 | 43 | 0.6701 | 0.7426 |
| proposed_vs_candidate_no_context | lore_consistency | 68 | 17 | 59 | 0.6771 | 0.8000 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 144 | 0.5000 | nan |
| proposed_vs_candidate_no_context | objective_completion_support | 34 | 33 | 77 | 0.5035 | 0.5075 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 57 | 44 | 43 | 0.5451 | 0.5644 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | 38 | 55 | 51 | 0.4410 | 0.4086 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 54 | 12 | 78 | 0.6458 | 0.8182 |
| proposed_vs_candidate_no_context | context_overlap | 77 | 24 | 43 | 0.6840 | 0.7624 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 10 | 19 | 115 | 0.4688 | 0.3448 |
| proposed_vs_candidate_no_context | persona_style | 13 | 8 | 123 | 0.5174 | 0.6190 |
| proposed_vs_candidate_no_context | distinct1 | 40 | 49 | 55 | 0.4688 | 0.4494 |
| proposed_vs_candidate_no_context | length_score | 44 | 55 | 45 | 0.4618 | 0.4444 |
| proposed_vs_candidate_no_context | sentence_score | 20 | 29 | 95 | 0.4688 | 0.4082 |
| proposed_vs_candidate_no_context | overall_quality | 64 | 37 | 43 | 0.5938 | 0.6337 |
| proposed_vs_baseline_no_context | context_relevance | 83 | 61 | 0 | 0.5764 | 0.5764 |
| proposed_vs_baseline_no_context | persona_consistency | 17 | 37 | 90 | 0.4306 | 0.3148 |
| proposed_vs_baseline_no_context | naturalness | 27 | 115 | 2 | 0.1944 | 0.1901 |
| proposed_vs_baseline_no_context | quest_state_correctness | 80 | 64 | 0 | 0.5556 | 0.5556 |
| proposed_vs_baseline_no_context | lore_consistency | 95 | 32 | 17 | 0.7188 | 0.7480 |
| proposed_vs_baseline_no_context | multi_turn_contradiction_safety | 0 | 0 | 144 | 0.5000 | nan |
| proposed_vs_baseline_no_context | objective_completion_support | 54 | 44 | 46 | 0.5347 | 0.5510 |
| proposed_vs_baseline_no_context | gameplay_usefulness | 48 | 96 | 0 | 0.3333 | 0.3333 |
| proposed_vs_baseline_no_context | time_pressure_acceptability | 56 | 72 | 16 | 0.4444 | 0.4375 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 48 | 24 | 72 | 0.5833 | 0.6667 |
| proposed_vs_baseline_no_context | context_overlap | 87 | 57 | 0 | 0.6042 | 0.6042 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 14 | 25 | 105 | 0.4618 | 0.3590 |
| proposed_vs_baseline_no_context | persona_style | 7 | 18 | 119 | 0.4618 | 0.2800 |
| proposed_vs_baseline_no_context | distinct1 | 27 | 106 | 11 | 0.2257 | 0.2030 |
| proposed_vs_baseline_no_context | length_score | 27 | 111 | 6 | 0.2083 | 0.1957 |
| proposed_vs_baseline_no_context | sentence_score | 15 | 55 | 74 | 0.3611 | 0.2143 |
| proposed_vs_baseline_no_context | overall_quality | 57 | 87 | 0 | 0.3958 | 0.3958 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 87 | 56 | 1 | 0.6076 | 0.6084 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | 19 | 43 | 82 | 0.4167 | 0.3065 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | 32 | 110 | 2 | 0.2292 | 0.2254 |
| proposed_vs_baseline_no_context_phi3_latest | quest_state_correctness | 86 | 57 | 1 | 0.6007 | 0.6014 |
| proposed_vs_baseline_no_context_phi3_latest | lore_consistency | 92 | 33 | 19 | 0.7049 | 0.7360 |
| proposed_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 18 | 0 | 126 | 0.5625 | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | objective_completion_support | 55 | 43 | 46 | 0.5417 | 0.5612 |
| proposed_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 48 | 96 | 0 | 0.3333 | 0.3333 |
| proposed_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 54 | 64 | 26 | 0.4653 | 0.4576 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 51 | 21 | 72 | 0.6042 | 0.7083 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 93 | 49 | 2 | 0.6528 | 0.6549 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 15 | 25 | 104 | 0.4653 | 0.3750 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | 7 | 20 | 117 | 0.4549 | 0.2593 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | 23 | 105 | 16 | 0.2153 | 0.1797 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | 36 | 105 | 3 | 0.2604 | 0.2553 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | 17 | 56 | 71 | 0.3646 | 0.2329 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | 58 | 85 | 1 | 0.4062 | 0.4056 |
| controlled_vs_proposed_raw | context_relevance | 131 | 13 | 0 | 0.9097 | 0.9097 |
| controlled_vs_proposed_raw | persona_consistency | 119 | 9 | 16 | 0.8819 | 0.9297 |
| controlled_vs_proposed_raw | naturalness | 113 | 31 | 0 | 0.7847 | 0.7847 |
| controlled_vs_proposed_raw | quest_state_correctness | 131 | 13 | 0 | 0.9097 | 0.9097 |
| controlled_vs_proposed_raw | lore_consistency | 127 | 13 | 4 | 0.8958 | 0.9071 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0 | 0 | 144 | 0.5000 | nan |
| controlled_vs_proposed_raw | objective_completion_support | 55 | 60 | 29 | 0.4826 | 0.4783 |
| controlled_vs_proposed_raw | gameplay_usefulness | 130 | 14 | 0 | 0.9028 | 0.9028 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 78 | 58 | 8 | 0.5694 | 0.5735 |
| controlled_vs_proposed_raw | context_keyword_coverage | 125 | 9 | 10 | 0.9028 | 0.9328 |
| controlled_vs_proposed_raw | context_overlap | 129 | 14 | 1 | 0.8993 | 0.9021 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 119 | 7 | 18 | 0.8889 | 0.9444 |
| controlled_vs_proposed_raw | persona_style | 15 | 6 | 123 | 0.5312 | 0.7143 |
| controlled_vs_proposed_raw | distinct1 | 73 | 63 | 8 | 0.5347 | 0.5368 |
| controlled_vs_proposed_raw | length_score | 114 | 29 | 1 | 0.7951 | 0.7972 |
| controlled_vs_proposed_raw | sentence_score | 72 | 13 | 59 | 0.7049 | 0.8471 |
| controlled_vs_proposed_raw | overall_quality | 136 | 8 | 0 | 0.9444 | 0.9444 |
| controlled_vs_candidate_no_context | context_relevance | 143 | 1 | 0 | 0.9931 | 0.9931 |
| controlled_vs_candidate_no_context | persona_consistency | 117 | 9 | 18 | 0.8750 | 0.9286 |
| controlled_vs_candidate_no_context | naturalness | 103 | 40 | 1 | 0.7188 | 0.7203 |
| controlled_vs_candidate_no_context | quest_state_correctness | 143 | 1 | 0 | 0.9931 | 0.9931 |
| controlled_vs_candidate_no_context | lore_consistency | 136 | 4 | 4 | 0.9583 | 0.9714 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 144 | 0.5000 | nan |
| controlled_vs_candidate_no_context | objective_completion_support | 61 | 57 | 26 | 0.5139 | 0.5169 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 136 | 8 | 0 | 0.9444 | 0.9444 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 67 | 72 | 5 | 0.4826 | 0.4820 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 143 | 0 | 1 | 0.9965 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 142 | 2 | 0 | 0.9861 | 0.9861 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 114 | 8 | 22 | 0.8681 | 0.9344 |
| controlled_vs_candidate_no_context | persona_style | 18 | 6 | 120 | 0.5417 | 0.7500 |
| controlled_vs_candidate_no_context | distinct1 | 62 | 76 | 6 | 0.4514 | 0.4493 |
| controlled_vs_candidate_no_context | length_score | 106 | 33 | 5 | 0.7535 | 0.7626 |
| controlled_vs_candidate_no_context | sentence_score | 66 | 16 | 62 | 0.6736 | 0.8049 |
| controlled_vs_candidate_no_context | overall_quality | 140 | 4 | 0 | 0.9722 | 0.9722 |
| controlled_vs_baseline_no_context | context_relevance | 144 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 112 | 8 | 24 | 0.8611 | 0.9333 |
| controlled_vs_baseline_no_context | naturalness | 84 | 60 | 0 | 0.5833 | 0.5833 |
| controlled_vs_baseline_no_context | quest_state_correctness | 144 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | lore_consistency | 138 | 1 | 5 | 0.9757 | 0.9928 |
| controlled_vs_baseline_no_context | multi_turn_contradiction_safety | 0 | 0 | 144 | 0.5000 | nan |
| controlled_vs_baseline_no_context | objective_completion_support | 61 | 52 | 31 | 0.5312 | 0.5398 |
| controlled_vs_baseline_no_context | gameplay_usefulness | 130 | 14 | 0 | 0.9028 | 0.9028 |
| controlled_vs_baseline_no_context | time_pressure_acceptability | 70 | 64 | 10 | 0.5208 | 0.5224 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 143 | 0 | 1 | 0.9965 | 1.0000 |
| controlled_vs_baseline_no_context | context_overlap | 142 | 2 | 0 | 0.9861 | 0.9861 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 112 | 4 | 28 | 0.8750 | 0.9655 |
| controlled_vs_baseline_no_context | persona_style | 7 | 9 | 128 | 0.4931 | 0.4375 |
| controlled_vs_baseline_no_context | distinct1 | 15 | 121 | 8 | 0.1319 | 0.1103 |
| controlled_vs_baseline_no_context | length_score | 89 | 50 | 5 | 0.6354 | 0.6403 |
| controlled_vs_baseline_no_context | sentence_score | 38 | 19 | 87 | 0.5660 | 0.6667 |
| controlled_vs_baseline_no_context | overall_quality | 143 | 1 | 0 | 0.9931 | 0.9931 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 143 | 1 | 0 | 0.9931 | 0.9931 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 117 | 8 | 19 | 0.8785 | 0.9360 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | 81 | 63 | 0 | 0.5625 | 0.5625 |
| controlled_vs_baseline_no_context_phi3_latest | quest_state_correctness | 143 | 1 | 0 | 0.9931 | 0.9931 |
| controlled_vs_baseline_no_context_phi3_latest | lore_consistency | 138 | 2 | 4 | 0.9722 | 0.9857 |
| controlled_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 18 | 0 | 126 | 0.5625 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | objective_completion_support | 66 | 47 | 31 | 0.5660 | 0.5841 |
| controlled_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 129 | 15 | 0 | 0.8958 | 0.8958 |
| controlled_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 59 | 73 | 12 | 0.4514 | 0.4470 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 143 | 0 | 1 | 0.9965 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 140 | 4 | 0 | 0.9722 | 0.9722 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 116 | 5 | 23 | 0.8854 | 0.9587 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | 9 | 12 | 123 | 0.4896 | 0.4286 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | 13 | 124 | 7 | 0.1146 | 0.0949 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | 95 | 47 | 2 | 0.6667 | 0.6690 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 40 | 20 | 84 | 0.5694 | 0.6667 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 144 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 144 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 112 | 8 | 24 | 0.8611 | 0.9333 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 84 | 60 | 0 | 0.5833 | 0.5833 |
| proposed_contextual_controlled_vs_baseline_no_context | quest_state_correctness | 144 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | lore_consistency | 138 | 1 | 5 | 0.9757 | 0.9928 |
| proposed_contextual_controlled_vs_baseline_no_context | multi_turn_contradiction_safety | 0 | 0 | 144 | 0.5000 | nan |
| proposed_contextual_controlled_vs_baseline_no_context | objective_completion_support | 61 | 52 | 31 | 0.5312 | 0.5398 |
| proposed_contextual_controlled_vs_baseline_no_context | gameplay_usefulness | 130 | 14 | 0 | 0.9028 | 0.9028 |
| proposed_contextual_controlled_vs_baseline_no_context | time_pressure_acceptability | 70 | 64 | 10 | 0.5208 | 0.5224 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 143 | 0 | 1 | 0.9965 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 142 | 2 | 0 | 0.9861 | 0.9861 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 112 | 4 | 28 | 0.8750 | 0.9655 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 7 | 9 | 128 | 0.4931 | 0.4375 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | 15 | 121 | 8 | 0.1319 | 0.1103 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 89 | 50 | 5 | 0.6354 | 0.6403 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 38 | 19 | 87 | 0.5660 | 0.6667 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 143 | 1 | 0 | 0.9931 | 0.9931 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 143 | 1 | 0 | 0.9931 | 0.9931 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 117 | 8 | 19 | 0.8785 | 0.9360 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | 81 | 63 | 0 | 0.5625 | 0.5625 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | quest_state_correctness | 143 | 1 | 0 | 0.9931 | 0.9931 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | lore_consistency | 138 | 2 | 4 | 0.9722 | 0.9857 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 18 | 0 | 126 | 0.5625 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | objective_completion_support | 66 | 47 | 31 | 0.5660 | 0.5841 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 129 | 15 | 0 | 0.8958 | 0.8958 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 59 | 73 | 12 | 0.4514 | 0.4470 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 143 | 0 | 1 | 0.9965 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 140 | 4 | 0 | 0.9722 | 0.9722 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 116 | 5 | 23 | 0.8854 | 0.9587 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | 9 | 12 | 123 | 0.4896 | 0.4286 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | 13 | 124 | 7 | 0.1146 | 0.0949 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | 95 | 47 | 2 | 0.6667 | 0.6690 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | 40 | 20 | 84 | 0.5694 | 0.6667 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 144 | 0 | 0 | 1.0000 | 1.0000 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.0139 | 0.4375 | 0.1667 |
| proposed_contextual | 0.0000 | 0.0000 | 0.4514 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.4167 | 0.0000 | 0.0000 |
| baseline_no_context | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
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
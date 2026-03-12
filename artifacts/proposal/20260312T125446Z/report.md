# Proposal Alignment Evaluation Report

- Run ID: `20260312T125446Z`
- Generated: `2026-03-12T12:56:37.030548+00:00`
- Scenarios: `artifacts\proposal\20260312T125446Z\scenarios.jsonl`
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
| proposed_contextual_controlled | 0.1231 (0.1024, 0.1453) | 0.2806 (0.2588, 0.3051) | 0.8666 (0.8595, 0.8736) | 0.3265 (0.3111, 0.3410) | n/a |
| proposed_contextual | 0.0745 (0.0599, 0.0894) | 0.1962 (0.1772, 0.2159) | 0.8724 (0.8650, 0.8801) | 0.2736 (0.2637, 0.2842) | n/a |
| candidate_no_context | 0.0344 (0.0284, 0.0410) | 0.2258 (0.2051, 0.2483) | 0.8737 (0.8658, 0.8816) | 0.2665 (0.2574, 0.2764) | n/a |
| baseline_no_context | 0.0318 (0.0258, 0.0386) | 0.1556 (0.1410, 0.1708) | 0.8886 (0.8788, 0.8978) | 0.2416 (0.2342, 0.2493) | n/a |
| baseline_no_context_phi3_latest | 0.0355 (0.0291, 0.0427) | 0.1578 (0.1418, 0.1741) | 0.8963 (0.8870, 0.9052) | 0.2456 (0.2376, 0.2536) | n/a |

## Game-facing Outcome Metrics (mean, 95% CI)
| Arm | Quest-state Correctness | Lore Consistency | Contradiction Safety | Objective Completion Support | Gameplay Usefulness | Time-pressure Acceptability |
|---|---:|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2070 (0.1881, 0.2266) | 0.0602 (0.0456, 0.0767) | 0.8750 (0.8194, 0.9236) | 0.0865 (0.0745, 0.0985) | 0.3077 (0.3018, 0.3143) | 0.2954 (0.2860, 0.3045) |
| proposed_contextual | 0.1641 (0.1513, 0.1775) | 0.0353 (0.0256, 0.0467) | 1.0000 (1.0000, 1.0000) | 0.0739 (0.0619, 0.0864) | 0.2952 (0.2877, 0.3030) | 0.2977 (0.2901, 0.3056) |
| candidate_no_context | 0.1299 (0.1241, 0.1360) | 0.0067 (0.0040, 0.0099) | 1.0000 (1.0000, 1.0000) | 0.0741 (0.0618, 0.0867) | 0.2806 (0.2743, 0.2871) | 0.2962 (0.2887, 0.3045) |
| baseline_no_context | 0.1268 (0.1213, 0.1327) | 0.0139 (0.0103, 0.0180) | 1.0000 (1.0000, 1.0000) | 0.0375 (0.0311, 0.0448) | 0.2724 (0.2665, 0.2782) | 0.2879 (0.2824, 0.2930) |
| baseline_no_context_phi3_latest | 0.1292 (0.1233, 0.1351) | 0.0115 (0.0080, 0.0157) | 0.8750 (0.8194, 0.9236) | 0.0329 (0.0277, 0.0384) | 0.2746 (0.2689, 0.2801) | 0.2859 (0.2812, 0.2907) |

- Multi-turn contradiction rate is reported as `1 - contradiction_safety` in row-level outputs.

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0401 | 1.1639 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0296 | -0.1313 |
| proposed_vs_candidate_no_context | naturalness | -0.0013 | -0.0015 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.0342 | 0.2632 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0286 | 4.2686 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | -0.0001 | -0.0015 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 0.0146 | 0.0521 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | 0.0015 | 0.0050 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0507 | 1.4576 |
| proposed_vs_candidate_no_context | context_overlap | 0.0154 | 0.4563 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0376 | -0.2554 |
| proposed_vs_candidate_no_context | persona_style | 0.0023 | 0.0043 |
| proposed_vs_candidate_no_context | distinct1 | -0.0045 | -0.0047 |
| proposed_vs_candidate_no_context | length_score | 0.0146 | 0.0289 |
| proposed_vs_candidate_no_context | sentence_score | -0.0243 | -0.0254 |
| proposed_vs_candidate_no_context | overall_quality | 0.0071 | 0.0265 |
| proposed_vs_baseline_no_context | context_relevance | 0.0427 | 1.3414 |
| proposed_vs_baseline_no_context | persona_consistency | 0.0405 | 0.2603 |
| proposed_vs_baseline_no_context | naturalness | -0.0162 | -0.0183 |
| proposed_vs_baseline_no_context | quest_state_correctness | 0.0372 | 0.2937 |
| proposed_vs_baseline_no_context | lore_consistency | 0.0215 | 1.5479 |
| proposed_vs_baseline_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| proposed_vs_baseline_no_context | objective_completion_support | 0.0364 | 0.9699 |
| proposed_vs_baseline_no_context | gameplay_usefulness | 0.0229 | 0.0840 |
| proposed_vs_baseline_no_context | time_pressure_acceptability | 0.0098 | 0.0340 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0545 | 1.7572 |
| proposed_vs_baseline_no_context | context_overlap | 0.0152 | 0.4514 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 0.0555 | 1.0232 |
| proposed_vs_baseline_no_context | persona_style | -0.0194 | -0.0345 |
| proposed_vs_baseline_no_context | distinct1 | -0.0409 | -0.0418 |
| proposed_vs_baseline_no_context | length_score | -0.0271 | -0.0496 |
| proposed_vs_baseline_no_context | sentence_score | 0.0556 | 0.0632 |
| proposed_vs_baseline_no_context | overall_quality | 0.0320 | 0.1323 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 0.0391 | 1.1016 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | 0.0384 | 0.2433 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | -0.0239 | -0.0267 |
| proposed_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.0348 | 0.2696 |
| proposed_vs_baseline_no_context_phi3_latest | lore_consistency | 0.0238 | 2.0645 |
| proposed_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0.1250 | 0.1429 |
| proposed_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.0410 | 1.2447 |
| proposed_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 0.0206 | 0.0750 |
| proposed_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 0.0118 | 0.0413 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.0502 | 1.4268 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 0.0130 | 0.3605 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.0521 | 0.9025 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | -0.0163 | -0.0292 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | -0.0449 | -0.0457 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | -0.0542 | -0.0945 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | 0.0486 | 0.0549 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | 0.0279 | 0.1137 |
| controlled_vs_proposed_raw | context_relevance | 0.0485 | 0.6514 |
| controlled_vs_proposed_raw | persona_consistency | 0.0845 | 0.4306 |
| controlled_vs_proposed_raw | naturalness | -0.0058 | -0.0066 |
| controlled_vs_proposed_raw | quest_state_correctness | 0.0429 | 0.2617 |
| controlled_vs_proposed_raw | lore_consistency | 0.0249 | 0.7043 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | -0.1250 | -0.1250 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0126 | 0.1702 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0125 | 0.0423 |
| controlled_vs_proposed_raw | time_pressure_acceptability | -0.0023 | -0.0077 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.0640 | 0.7485 |
| controlled_vs_proposed_raw | context_overlap | 0.0126 | 0.2564 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1029 | 0.9376 |
| controlled_vs_proposed_raw | persona_style | 0.0108 | 0.0200 |
| controlled_vs_proposed_raw | distinct1 | -0.0046 | -0.0049 |
| controlled_vs_proposed_raw | length_score | -0.0431 | -0.0830 |
| controlled_vs_proposed_raw | sentence_score | 0.0486 | 0.0520 |
| controlled_vs_proposed_raw | overall_quality | 0.0529 | 0.1936 |
| controlled_vs_candidate_no_context | context_relevance | 0.0886 | 2.5735 |
| controlled_vs_candidate_no_context | persona_consistency | 0.0548 | 0.2428 |
| controlled_vs_candidate_no_context | naturalness | -0.0071 | -0.0081 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.0771 | 0.5938 |
| controlled_vs_candidate_no_context | lore_consistency | 0.0535 | 7.9792 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | -0.1250 | -0.1250 |
| controlled_vs_candidate_no_context | objective_completion_support | 0.0125 | 0.1685 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0271 | 0.0966 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | -0.0008 | -0.0027 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.1146 | 3.2972 |
| controlled_vs_candidate_no_context | context_overlap | 0.0279 | 0.8298 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.0652 | 0.4428 |
| controlled_vs_candidate_no_context | persona_style | 0.0132 | 0.0244 |
| controlled_vs_candidate_no_context | distinct1 | -0.0091 | -0.0096 |
| controlled_vs_candidate_no_context | length_score | -0.0285 | -0.0565 |
| controlled_vs_candidate_no_context | sentence_score | 0.0243 | 0.0254 |
| controlled_vs_candidate_no_context | overall_quality | 0.0600 | 0.2252 |
| controlled_vs_baseline_no_context | context_relevance | 0.0912 | 2.8665 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1250 | 0.8030 |
| controlled_vs_baseline_no_context | naturalness | -0.0220 | -0.0247 |
| controlled_vs_baseline_no_context | quest_state_correctness | 0.0802 | 0.6323 |
| controlled_vs_baseline_no_context | lore_consistency | 0.0464 | 3.3422 |
| controlled_vs_baseline_no_context | multi_turn_contradiction_safety | -0.1250 | -0.1250 |
| controlled_vs_baseline_no_context | objective_completion_support | 0.0490 | 1.3053 |
| controlled_vs_baseline_no_context | gameplay_usefulness | 0.0354 | 0.1299 |
| controlled_vs_baseline_no_context | time_pressure_acceptability | 0.0075 | 0.0260 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.1184 | 3.8211 |
| controlled_vs_baseline_no_context | context_overlap | 0.0278 | 0.8235 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1584 | 2.9201 |
| controlled_vs_baseline_no_context | persona_style | -0.0085 | -0.0152 |
| controlled_vs_baseline_no_context | distinct1 | -0.0455 | -0.0465 |
| controlled_vs_baseline_no_context | length_score | -0.0701 | -0.1285 |
| controlled_vs_baseline_no_context | sentence_score | 0.1042 | 0.1185 |
| controlled_vs_baseline_no_context | overall_quality | 0.0849 | 0.3515 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.0876 | 2.4706 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1229 | 0.7787 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | -0.0297 | -0.0331 |
| controlled_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.0778 | 0.6019 |
| controlled_vs_baseline_no_context_phi3_latest | lore_consistency | 0.0487 | 4.2227 |
| controlled_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.0536 | 1.6268 |
| controlled_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 0.0331 | 0.1206 |
| controlled_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 0.0095 | 0.0333 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.1142 | 3.2433 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0256 | 0.7093 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1549 | 2.6864 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | -0.0055 | -0.0098 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0495 | -0.0504 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | -0.0972 | -0.1697 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.0972 | 0.1098 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.0809 | 0.3293 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.0912 | 2.8665 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1250 | 0.8030 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0220 | -0.0247 |
| proposed_contextual_controlled_vs_baseline_no_context | quest_state_correctness | 0.0802 | 0.6323 |
| proposed_contextual_controlled_vs_baseline_no_context | lore_consistency | 0.0464 | 3.3422 |
| proposed_contextual_controlled_vs_baseline_no_context | multi_turn_contradiction_safety | -0.1250 | -0.1250 |
| proposed_contextual_controlled_vs_baseline_no_context | objective_completion_support | 0.0490 | 1.3053 |
| proposed_contextual_controlled_vs_baseline_no_context | gameplay_usefulness | 0.0354 | 0.1299 |
| proposed_contextual_controlled_vs_baseline_no_context | time_pressure_acceptability | 0.0075 | 0.0260 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.1184 | 3.8211 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0278 | 0.8235 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1584 | 2.9201 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | -0.0085 | -0.0152 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0455 | -0.0465 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | -0.0701 | -0.1285 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.1042 | 0.1185 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.0849 | 0.3515 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.0876 | 2.4706 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1229 | 0.7787 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | -0.0297 | -0.0331 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.0778 | 0.6019 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | lore_consistency | 0.0487 | 4.2227 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.0536 | 1.6268 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 0.0331 | 0.1206 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 0.0095 | 0.0333 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.1142 | 3.2433 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0256 | 0.7093 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1549 | 2.6864 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | -0.0055 | -0.0098 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0495 | -0.0504 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | -0.0972 | -0.1697 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.0972 | 0.1098 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.0809 | 0.3293 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0401 | (0.0250, 0.0566) | 0.0000 | 0.0401 | (0.0215, 0.0608) | 0.0000 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0296 | (-0.0515, -0.0101) | 0.9993 | -0.0296 | (-0.0521, -0.0073) | 0.9977 |
| proposed_vs_candidate_no_context | naturalness | -0.0013 | (-0.0104, 0.0076) | 0.6200 | -0.0013 | (-0.0067, 0.0039) | 0.6873 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.0342 | (0.0210, 0.0490) | 0.0000 | 0.0342 | (0.0168, 0.0519) | 0.0000 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0286 | (0.0177, 0.0403) | 0.0000 | 0.0286 | (0.0135, 0.0446) | 0.0000 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | -0.0001 | (-0.0119, 0.0117) | 0.5137 | -0.0001 | (-0.0161, 0.0163) | 0.5183 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 0.0146 | (0.0075, 0.0217) | 0.0000 | 0.0146 | (0.0057, 0.0253) | 0.0000 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | 0.0015 | (-0.0072, 0.0096) | 0.3787 | 0.0015 | (-0.0118, 0.0171) | 0.4523 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0507 | (0.0305, 0.0732) | 0.0000 | 0.0507 | (0.0266, 0.0764) | 0.0000 |
| proposed_vs_candidate_no_context | context_overlap | 0.0154 | (0.0092, 0.0213) | 0.0000 | 0.0154 | (0.0062, 0.0260) | 0.0000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0376 | (-0.0630, -0.0114) | 0.9967 | -0.0376 | (-0.0674, -0.0103) | 0.9967 |
| proposed_vs_candidate_no_context | persona_style | 0.0023 | (-0.0160, 0.0205) | 0.3830 | 0.0023 | (-0.0254, 0.0294) | 0.4153 |
| proposed_vs_candidate_no_context | distinct1 | -0.0045 | (-0.0121, 0.0037) | 0.8543 | -0.0045 | (-0.0119, 0.0037) | 0.8613 |
| proposed_vs_candidate_no_context | length_score | 0.0146 | (-0.0211, 0.0498) | 0.2190 | 0.0146 | (-0.0137, 0.0461) | 0.1770 |
| proposed_vs_candidate_no_context | sentence_score | -0.0243 | (-0.0486, 0.0000) | 0.9797 | -0.0243 | (-0.0487, 0.0000) | 0.9863 |
| proposed_vs_candidate_no_context | overall_quality | 0.0071 | (-0.0042, 0.0186) | 0.1153 | 0.0071 | (-0.0061, 0.0199) | 0.1443 |
| proposed_vs_baseline_no_context | context_relevance | 0.0427 | (0.0271, 0.0599) | 0.0000 | 0.0427 | (0.0201, 0.0693) | 0.0000 |
| proposed_vs_baseline_no_context | persona_consistency | 0.0405 | (0.0191, 0.0623) | 0.0000 | 0.0405 | (0.0141, 0.0639) | 0.0053 |
| proposed_vs_baseline_no_context | naturalness | -0.0162 | (-0.0287, -0.0047) | 0.9980 | -0.0162 | (-0.0377, 0.0041) | 0.9370 |
| proposed_vs_baseline_no_context | quest_state_correctness | 0.0372 | (0.0238, 0.0517) | 0.0000 | 0.0372 | (0.0193, 0.0578) | 0.0000 |
| proposed_vs_baseline_no_context | lore_consistency | 0.0215 | (0.0110, 0.0328) | 0.0000 | 0.0215 | (0.0059, 0.0384) | 0.0030 |
| proposed_vs_baseline_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_baseline_no_context | objective_completion_support | 0.0364 | (0.0238, 0.0497) | 0.0000 | 0.0364 | (0.0116, 0.0637) | 0.0010 |
| proposed_vs_baseline_no_context | gameplay_usefulness | 0.0229 | (0.0137, 0.0319) | 0.0000 | 0.0229 | (0.0048, 0.0385) | 0.0060 |
| proposed_vs_baseline_no_context | time_pressure_acceptability | 0.0098 | (0.0008, 0.0188) | 0.0157 | 0.0098 | (-0.0069, 0.0271) | 0.1390 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0545 | (0.0338, 0.0754) | 0.0000 | 0.0545 | (0.0272, 0.0890) | 0.0000 |
| proposed_vs_baseline_no_context | context_overlap | 0.0152 | (0.0089, 0.0217) | 0.0000 | 0.0152 | (0.0043, 0.0264) | 0.0020 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 0.0555 | (0.0285, 0.0837) | 0.0000 | 0.0555 | (0.0192, 0.0854) | 0.0027 |
| proposed_vs_baseline_no_context | persona_style | -0.0194 | (-0.0389, -0.0002) | 0.9767 | -0.0194 | (-0.0666, 0.0208) | 0.7877 |
| proposed_vs_baseline_no_context | distinct1 | -0.0409 | (-0.0506, -0.0316) | 1.0000 | -0.0409 | (-0.0598, -0.0252) | 1.0000 |
| proposed_vs_baseline_no_context | length_score | -0.0271 | (-0.0782, 0.0213) | 0.8740 | -0.0271 | (-0.1086, 0.0528) | 0.7243 |
| proposed_vs_baseline_no_context | sentence_score | 0.0556 | (0.0219, 0.0892) | 0.0003 | 0.0556 | (-0.0126, 0.1261) | 0.0580 |
| proposed_vs_baseline_no_context | overall_quality | 0.0320 | (0.0207, 0.0437) | 0.0000 | 0.0320 | (0.0137, 0.0489) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 0.0391 | (0.0235, 0.0557) | 0.0000 | 0.0391 | (0.0154, 0.0684) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | 0.0384 | (0.0158, 0.0601) | 0.0003 | 0.0384 | (0.0047, 0.0636) | 0.0153 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | -0.0239 | (-0.0351, -0.0127) | 1.0000 | -0.0239 | (-0.0362, -0.0115) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.0348 | (0.0217, 0.0494) | 0.0000 | 0.0348 | (0.0150, 0.0574) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | lore_consistency | 0.0238 | (0.0135, 0.0354) | 0.0000 | 0.0238 | (0.0064, 0.0431) | 0.0027 |
| proposed_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0.1250 | (0.0764, 0.1806) | 0.0000 | 0.1250 | (0.0000, 0.3750) | 0.3337 |
| proposed_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.0410 | (0.0294, 0.0528) | 0.0000 | 0.0410 | (0.0166, 0.0699) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 0.0206 | (0.0115, 0.0299) | 0.0000 | 0.0206 | (0.0065, 0.0341) | 0.0017 |
| proposed_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 0.0118 | (0.0037, 0.0203) | 0.0033 | 0.0118 | (-0.0051, 0.0302) | 0.1010 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.0502 | (0.0290, 0.0727) | 0.0000 | 0.0502 | (0.0207, 0.0886) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 0.0130 | (0.0064, 0.0197) | 0.0003 | 0.0130 | (-0.0011, 0.0259) | 0.0353 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.0521 | (0.0253, 0.0792) | 0.0000 | 0.0521 | (0.0082, 0.0892) | 0.0107 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | -0.0163 | (-0.0365, 0.0053) | 0.9327 | -0.0163 | (-0.0549, 0.0174) | 0.8193 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | -0.0449 | (-0.0533, -0.0358) | 1.0000 | -0.0449 | (-0.0618, -0.0312) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | -0.0542 | (-0.1012, -0.0060) | 0.9893 | -0.0542 | (-0.1000, -0.0065) | 0.9857 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | 0.0486 | (0.0170, 0.0802) | 0.0033 | 0.0486 | (0.0122, 0.0826) | 0.0073 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | 0.0279 | (0.0157, 0.0403) | 0.0000 | 0.0279 | (0.0070, 0.0471) | 0.0047 |
| controlled_vs_proposed_raw | context_relevance | 0.0485 | (0.0248, 0.0728) | 0.0000 | 0.0485 | (0.0320, 0.0697) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.0845 | (0.0591, 0.1111) | 0.0000 | 0.0845 | (0.0598, 0.1190) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | -0.0058 | (-0.0157, 0.0040) | 0.8760 | -0.0058 | (-0.0150, 0.0017) | 0.9290 |
| controlled_vs_proposed_raw | quest_state_correctness | 0.0429 | (0.0230, 0.0646) | 0.0000 | 0.0429 | (0.0286, 0.0594) | 0.0000 |
| controlled_vs_proposed_raw | lore_consistency | 0.0249 | (0.0073, 0.0425) | 0.0017 | 0.0249 | (0.0160, 0.0343) | 0.0000 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | -0.1250 | (-0.1806, -0.0764) | 1.0000 | -0.1250 | (-0.3750, 0.0000) | 1.0000 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0126 | (0.0012, 0.0243) | 0.0123 | 0.0126 | (-0.0046, 0.0290) | 0.0833 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0125 | (0.0051, 0.0200) | 0.0000 | 0.0125 | (0.0044, 0.0205) | 0.0007 |
| controlled_vs_proposed_raw | time_pressure_acceptability | -0.0023 | (-0.0111, 0.0059) | 0.7033 | -0.0023 | (-0.0157, 0.0100) | 0.6237 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.0640 | (0.0312, 0.0966) | 0.0000 | 0.0640 | (0.0410, 0.0914) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0126 | (0.0037, 0.0216) | 0.0013 | 0.0126 | (0.0059, 0.0216) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1029 | (0.0713, 0.1340) | 0.0000 | 0.1029 | (0.0750, 0.1422) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.0108 | (-0.0052, 0.0282) | 0.1003 | 0.0108 | (-0.0104, 0.0368) | 0.1840 |
| controlled_vs_proposed_raw | distinct1 | -0.0046 | (-0.0125, 0.0030) | 0.8870 | -0.0046 | (-0.0150, 0.0064) | 0.7960 |
| controlled_vs_proposed_raw | length_score | -0.0431 | (-0.0840, -0.0016) | 0.9780 | -0.0431 | (-0.0752, -0.0109) | 0.9950 |
| controlled_vs_proposed_raw | sentence_score | 0.0486 | (0.0243, 0.0729) | 0.0000 | 0.0486 | (0.0194, 0.0753) | 0.0000 |
| controlled_vs_proposed_raw | overall_quality | 0.0529 | (0.0373, 0.0691) | 0.0000 | 0.0529 | (0.0405, 0.0710) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.0886 | (0.0663, 0.1114) | 0.0000 | 0.0886 | (0.0629, 0.1179) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.0548 | (0.0304, 0.0813) | 0.0000 | 0.0548 | (0.0179, 0.0901) | 0.0020 |
| controlled_vs_candidate_no_context | naturalness | -0.0071 | (-0.0171, 0.0029) | 0.9177 | -0.0071 | (-0.0134, -0.0005) | 0.9813 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.0771 | (0.0578, 0.0965) | 0.0000 | 0.0771 | (0.0544, 0.1021) | 0.0000 |
| controlled_vs_candidate_no_context | lore_consistency | 0.0535 | (0.0381, 0.0707) | 0.0000 | 0.0535 | (0.0329, 0.0751) | 0.0000 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | -0.1250 | (-0.1806, -0.0764) | 1.0000 | -0.1250 | (-0.3750, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | 0.0125 | (0.0008, 0.0236) | 0.0177 | 0.0125 | (0.0058, 0.0186) | 0.0003 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0271 | (0.0202, 0.0341) | 0.0000 | 0.0271 | (0.0194, 0.0343) | 0.0000 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | -0.0008 | (-0.0098, 0.0084) | 0.5657 | -0.0008 | (-0.0093, 0.0071) | 0.5660 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.1146 | (0.0855, 0.1439) | 0.0000 | 0.1146 | (0.0805, 0.1521) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0279 | (0.0195, 0.0366) | 0.0000 | 0.0279 | (0.0181, 0.0400) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.0652 | (0.0326, 0.0965) | 0.0000 | 0.0652 | (0.0190, 0.1087) | 0.0017 |
| controlled_vs_candidate_no_context | persona_style | 0.0132 | (0.0015, 0.0269) | 0.0133 | 0.0132 | (0.0033, 0.0241) | 0.0010 |
| controlled_vs_candidate_no_context | distinct1 | -0.0091 | (-0.0174, -0.0009) | 0.9857 | -0.0091 | (-0.0196, 0.0002) | 0.9690 |
| controlled_vs_candidate_no_context | length_score | -0.0285 | (-0.0704, 0.0141) | 0.9027 | -0.0285 | (-0.0569, 0.0005) | 0.9717 |
| controlled_vs_candidate_no_context | sentence_score | 0.0243 | (0.0024, 0.0462) | 0.0160 | 0.0243 | (0.0122, 0.0365) | 0.0000 |
| controlled_vs_candidate_no_context | overall_quality | 0.0600 | (0.0448, 0.0767) | 0.0000 | 0.0600 | (0.0396, 0.0821) | 0.0000 |
| controlled_vs_baseline_no_context | context_relevance | 0.0912 | (0.0690, 0.1153) | 0.0000 | 0.0912 | (0.0625, 0.1231) | 0.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1250 | (0.1004, 0.1501) | 0.0000 | 0.1250 | (0.0865, 0.1616) | 0.0000 |
| controlled_vs_baseline_no_context | naturalness | -0.0220 | (-0.0340, -0.0102) | 1.0000 | -0.0220 | (-0.0368, -0.0068) | 0.9980 |
| controlled_vs_baseline_no_context | quest_state_correctness | 0.0802 | (0.0606, 0.1012) | 0.0000 | 0.0802 | (0.0551, 0.1056) | 0.0000 |
| controlled_vs_baseline_no_context | lore_consistency | 0.0464 | (0.0309, 0.0626) | 0.0000 | 0.0464 | (0.0246, 0.0684) | 0.0000 |
| controlled_vs_baseline_no_context | multi_turn_contradiction_safety | -0.1250 | (-0.1806, -0.0764) | 1.0000 | -0.1250 | (-0.3750, 0.0000) | 1.0000 |
| controlled_vs_baseline_no_context | objective_completion_support | 0.0490 | (0.0361, 0.0618) | 0.0000 | 0.0490 | (0.0262, 0.0756) | 0.0000 |
| controlled_vs_baseline_no_context | gameplay_usefulness | 0.0354 | (0.0268, 0.0435) | 0.0000 | 0.0354 | (0.0201, 0.0482) | 0.0000 |
| controlled_vs_baseline_no_context | time_pressure_acceptability | 0.0075 | (-0.0018, 0.0171) | 0.0600 | 0.0075 | (-0.0048, 0.0244) | 0.1660 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.1184 | (0.0887, 0.1489) | 0.0000 | 0.1184 | (0.0799, 0.1598) | 0.0000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0278 | (0.0192, 0.0367) | 0.0000 | 0.0278 | (0.0167, 0.0403) | 0.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1584 | (0.1288, 0.1886) | 0.0000 | 0.1584 | (0.1118, 0.1984) | 0.0000 |
| controlled_vs_baseline_no_context | persona_style | -0.0085 | (-0.0263, 0.0088) | 0.8403 | -0.0085 | (-0.0516, 0.0272) | 0.6447 |
| controlled_vs_baseline_no_context | distinct1 | -0.0455 | (-0.0538, -0.0373) | 1.0000 | -0.0455 | (-0.0628, -0.0313) | 1.0000 |
| controlled_vs_baseline_no_context | length_score | -0.0701 | (-0.1213, -0.0169) | 0.9973 | -0.0701 | (-0.1292, -0.0141) | 0.9933 |
| controlled_vs_baseline_no_context | sentence_score | 0.1042 | (0.0750, 0.1354) | 0.0000 | 0.1042 | (0.0604, 0.1524) | 0.0000 |
| controlled_vs_baseline_no_context | overall_quality | 0.0849 | (0.0685, 0.1016) | 0.0000 | 0.0849 | (0.0596, 0.1085) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.0876 | (0.0660, 0.1104) | 0.0000 | 0.0876 | (0.0570, 0.1206) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1229 | (0.0998, 0.1474) | 0.0000 | 0.1229 | (0.0787, 0.1642) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | -0.0297 | (-0.0412, -0.0182) | 1.0000 | -0.0297 | (-0.0410, -0.0191) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.0778 | (0.0590, 0.0970) | 0.0000 | 0.0778 | (0.0519, 0.1039) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | lore_consistency | 0.0487 | (0.0335, 0.0645) | 0.0000 | 0.0487 | (0.0246, 0.0726) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.0536 | (0.0417, 0.0660) | 0.0000 | 0.0536 | (0.0297, 0.0811) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 0.0331 | (0.0251, 0.0408) | 0.0000 | 0.0331 | (0.0210, 0.0449) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 0.0095 | (-0.0002, 0.0195) | 0.0267 | 0.0095 | (-0.0022, 0.0264) | 0.0907 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.1142 | (0.0867, 0.1430) | 0.0000 | 0.1142 | (0.0745, 0.1555) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0256 | (0.0168, 0.0344) | 0.0000 | 0.0256 | (0.0123, 0.0399) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1549 | (0.1238, 0.1835) | 0.0000 | 0.1549 | (0.0972, 0.2016) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | -0.0055 | (-0.0208, 0.0110) | 0.7543 | -0.0055 | (-0.0382, 0.0231) | 0.6417 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0495 | (-0.0577, -0.0414) | 1.0000 | -0.0495 | (-0.0647, -0.0361) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | -0.0972 | (-0.1444, -0.0486) | 1.0000 | -0.0972 | (-0.1324, -0.0567) | 0.9997 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.0972 | (0.0656, 0.1288) | 0.0000 | 0.0972 | (0.0729, 0.1167) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.0809 | (0.0649, 0.0959) | 0.0000 | 0.0809 | (0.0536, 0.1073) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.0912 | (0.0686, 0.1145) | 0.0000 | 0.0912 | (0.0614, 0.1233) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1250 | (0.1004, 0.1497) | 0.0000 | 0.1250 | (0.0874, 0.1617) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0220 | (-0.0335, -0.0099) | 1.0000 | -0.0220 | (-0.0367, -0.0063) | 0.9967 |
| proposed_contextual_controlled_vs_baseline_no_context | quest_state_correctness | 0.0802 | (0.0603, 0.0998) | 0.0000 | 0.0802 | (0.0554, 0.1069) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | lore_consistency | 0.0464 | (0.0316, 0.0622) | 0.0000 | 0.0464 | (0.0249, 0.0689) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | multi_turn_contradiction_safety | -0.1250 | (-0.1806, -0.0764) | 1.0000 | -0.1250 | (-0.3750, 0.0000) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | objective_completion_support | 0.0490 | (0.0365, 0.0616) | 0.0000 | 0.0490 | (0.0267, 0.0745) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | gameplay_usefulness | 0.0354 | (0.0266, 0.0440) | 0.0000 | 0.0354 | (0.0208, 0.0487) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | time_pressure_acceptability | 0.0075 | (-0.0016, 0.0172) | 0.0597 | 0.0075 | (-0.0044, 0.0241) | 0.1757 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.1184 | (0.0900, 0.1479) | 0.0000 | 0.1184 | (0.0806, 0.1603) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0278 | (0.0193, 0.0368) | 0.0000 | 0.0278 | (0.0170, 0.0404) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1584 | (0.1287, 0.1887) | 0.0000 | 0.1584 | (0.1113, 0.1984) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | -0.0085 | (-0.0258, 0.0096) | 0.8287 | -0.0085 | (-0.0516, 0.0272) | 0.6513 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0455 | (-0.0536, -0.0375) | 1.0000 | -0.0455 | (-0.0616, -0.0311) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | -0.0701 | (-0.1213, -0.0199) | 0.9967 | -0.0701 | (-0.1285, -0.0130) | 0.9943 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.1042 | (0.0750, 0.1354) | 0.0000 | 0.1042 | (0.0583, 0.1524) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.0849 | (0.0681, 0.1012) | 0.0000 | 0.0849 | (0.0586, 0.1087) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.0876 | (0.0664, 0.1095) | 0.0000 | 0.0876 | (0.0571, 0.1203) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1229 | (0.0980, 0.1471) | 0.0000 | 0.1229 | (0.0798, 0.1649) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | -0.0297 | (-0.0414, -0.0180) | 1.0000 | -0.0297 | (-0.0412, -0.0189) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.0778 | (0.0598, 0.0966) | 0.0000 | 0.0778 | (0.0533, 0.1034) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | lore_consistency | 0.0487 | (0.0341, 0.0653) | 0.0000 | 0.0487 | (0.0252, 0.0719) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.0536 | (0.0411, 0.0650) | 0.0000 | 0.0536 | (0.0306, 0.0803) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 0.0331 | (0.0253, 0.0407) | 0.0000 | 0.0331 | (0.0207, 0.0443) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 0.0095 | (-0.0003, 0.0196) | 0.0290 | 0.0095 | (-0.0023, 0.0270) | 0.0903 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.1142 | (0.0868, 0.1423) | 0.0000 | 0.1142 | (0.0753, 0.1567) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0256 | (0.0171, 0.0345) | 0.0000 | 0.0256 | (0.0125, 0.0395) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1549 | (0.1249, 0.1847) | 0.0000 | 0.1549 | (0.1007, 0.1992) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | -0.0055 | (-0.0213, 0.0105) | 0.7623 | -0.0055 | (-0.0385, 0.0228) | 0.6313 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0495 | (-0.0579, -0.0409) | 1.0000 | -0.0495 | (-0.0642, -0.0359) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | -0.0972 | (-0.1449, -0.0493) | 1.0000 | -0.0972 | (-0.1324, -0.0593) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.0972 | (0.0681, 0.1288) | 0.0000 | 0.0972 | (0.0729, 0.1191) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.0809 | (0.0653, 0.0962) | 0.0000 | 0.0809 | (0.0528, 0.1074) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 62 | 35 | 47 | 0.5938 | 0.6392 |
| proposed_vs_candidate_no_context | persona_consistency | 21 | 39 | 84 | 0.4375 | 0.3500 |
| proposed_vs_candidate_no_context | naturalness | 46 | 50 | 48 | 0.4861 | 0.4792 |
| proposed_vs_candidate_no_context | quest_state_correctness | 62 | 35 | 47 | 0.5938 | 0.6392 |
| proposed_vs_candidate_no_context | lore_consistency | 46 | 16 | 82 | 0.6042 | 0.7419 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 144 | 0.5000 | nan |
| proposed_vs_candidate_no_context | objective_completion_support | 40 | 34 | 70 | 0.5208 | 0.5405 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 66 | 31 | 47 | 0.6215 | 0.6804 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | 42 | 39 | 63 | 0.5104 | 0.5185 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 47 | 20 | 77 | 0.5938 | 0.7015 |
| proposed_vs_candidate_no_context | context_overlap | 66 | 30 | 48 | 0.6250 | 0.6875 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 19 | 34 | 91 | 0.4479 | 0.3585 |
| proposed_vs_candidate_no_context | persona_style | 10 | 8 | 126 | 0.5069 | 0.5556 |
| proposed_vs_candidate_no_context | distinct1 | 38 | 46 | 60 | 0.4722 | 0.4524 |
| proposed_vs_candidate_no_context | length_score | 47 | 46 | 51 | 0.5035 | 0.5054 |
| proposed_vs_candidate_no_context | sentence_score | 9 | 19 | 116 | 0.4653 | 0.3214 |
| proposed_vs_candidate_no_context | overall_quality | 54 | 43 | 47 | 0.5382 | 0.5567 |
| proposed_vs_baseline_no_context | context_relevance | 89 | 53 | 2 | 0.6250 | 0.6268 |
| proposed_vs_baseline_no_context | persona_consistency | 61 | 26 | 57 | 0.6215 | 0.7011 |
| proposed_vs_baseline_no_context | naturalness | 61 | 82 | 1 | 0.4271 | 0.4266 |
| proposed_vs_baseline_no_context | quest_state_correctness | 89 | 53 | 2 | 0.6250 | 0.6268 |
| proposed_vs_baseline_no_context | lore_consistency | 47 | 46 | 51 | 0.5035 | 0.5054 |
| proposed_vs_baseline_no_context | multi_turn_contradiction_safety | 0 | 0 | 144 | 0.5000 | nan |
| proposed_vs_baseline_no_context | objective_completion_support | 74 | 31 | 39 | 0.6493 | 0.7048 |
| proposed_vs_baseline_no_context | gameplay_usefulness | 95 | 49 | 0 | 0.6597 | 0.6597 |
| proposed_vs_baseline_no_context | time_pressure_acceptability | 77 | 51 | 16 | 0.5903 | 0.6016 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 51 | 17 | 76 | 0.6181 | 0.7500 |
| proposed_vs_baseline_no_context | context_overlap | 90 | 52 | 2 | 0.6319 | 0.6338 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 58 | 20 | 66 | 0.6319 | 0.7436 |
| proposed_vs_baseline_no_context | persona_style | 10 | 22 | 112 | 0.4583 | 0.3125 |
| proposed_vs_baseline_no_context | distinct1 | 27 | 97 | 20 | 0.2569 | 0.2177 |
| proposed_vs_baseline_no_context | length_score | 64 | 71 | 9 | 0.4757 | 0.4741 |
| proposed_vs_baseline_no_context | sentence_score | 36 | 14 | 94 | 0.5764 | 0.7200 |
| proposed_vs_baseline_no_context | overall_quality | 102 | 42 | 0 | 0.7083 | 0.7083 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 89 | 55 | 0 | 0.6181 | 0.6181 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | 61 | 28 | 55 | 0.6146 | 0.6854 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | 55 | 88 | 1 | 0.3854 | 0.3846 |
| proposed_vs_baseline_no_context_phi3_latest | quest_state_correctness | 88 | 56 | 0 | 0.6111 | 0.6111 |
| proposed_vs_baseline_no_context_phi3_latest | lore_consistency | 47 | 40 | 57 | 0.5243 | 0.5402 |
| proposed_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 18 | 0 | 126 | 0.5625 | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | objective_completion_support | 78 | 35 | 31 | 0.6493 | 0.6903 |
| proposed_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 90 | 54 | 0 | 0.6250 | 0.6250 |
| proposed_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 72 | 55 | 17 | 0.5590 | 0.5669 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 53 | 22 | 69 | 0.6076 | 0.7067 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 92 | 52 | 0 | 0.6389 | 0.6389 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 55 | 19 | 70 | 0.6250 | 0.7432 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | 10 | 19 | 115 | 0.4688 | 0.3448 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | 18 | 104 | 22 | 0.2014 | 0.1475 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | 58 | 81 | 5 | 0.4201 | 0.4173 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | 34 | 14 | 96 | 0.5694 | 0.7083 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | 101 | 43 | 0 | 0.7014 | 0.7014 |
| controlled_vs_proposed_raw | context_relevance | 67 | 45 | 32 | 0.5764 | 0.5982 |
| controlled_vs_proposed_raw | persona_consistency | 68 | 14 | 62 | 0.6875 | 0.8293 |
| controlled_vs_proposed_raw | naturalness | 50 | 63 | 31 | 0.4549 | 0.4425 |
| controlled_vs_proposed_raw | quest_state_correctness | 69 | 44 | 31 | 0.5868 | 0.6106 |
| controlled_vs_proposed_raw | lore_consistency | 43 | 31 | 70 | 0.5417 | 0.5811 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0 | 18 | 126 | 0.4375 | 0.0000 |
| controlled_vs_proposed_raw | objective_completion_support | 47 | 34 | 63 | 0.5451 | 0.5802 |
| controlled_vs_proposed_raw | gameplay_usefulness | 64 | 49 | 31 | 0.5521 | 0.5664 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 44 | 47 | 53 | 0.4896 | 0.4835 |
| controlled_vs_proposed_raw | context_keyword_coverage | 50 | 26 | 68 | 0.5833 | 0.6579 |
| controlled_vs_proposed_raw | context_overlap | 61 | 51 | 32 | 0.5347 | 0.5446 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 68 | 11 | 65 | 0.6979 | 0.8608 |
| controlled_vs_proposed_raw | persona_style | 11 | 8 | 125 | 0.5104 | 0.5789 |
| controlled_vs_proposed_raw | distinct1 | 61 | 45 | 38 | 0.5556 | 0.5755 |
| controlled_vs_proposed_raw | length_score | 46 | 62 | 36 | 0.4444 | 0.4259 |
| controlled_vs_proposed_raw | sentence_score | 26 | 6 | 112 | 0.5694 | 0.8125 |
| controlled_vs_proposed_raw | overall_quality | 81 | 32 | 31 | 0.6701 | 0.7168 |
| controlled_vs_candidate_no_context | context_relevance | 78 | 38 | 28 | 0.6389 | 0.6724 |
| controlled_vs_candidate_no_context | persona_consistency | 63 | 17 | 64 | 0.6597 | 0.7875 |
| controlled_vs_candidate_no_context | naturalness | 49 | 67 | 28 | 0.4375 | 0.4224 |
| controlled_vs_candidate_no_context | quest_state_correctness | 78 | 38 | 28 | 0.6389 | 0.6724 |
| controlled_vs_candidate_no_context | lore_consistency | 52 | 14 | 78 | 0.6319 | 0.7879 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 18 | 126 | 0.4375 | 0.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | 52 | 39 | 53 | 0.5451 | 0.5714 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 88 | 28 | 28 | 0.7083 | 0.7586 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 46 | 53 | 45 | 0.4757 | 0.4646 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 67 | 14 | 63 | 0.6840 | 0.8272 |
| controlled_vs_candidate_no_context | context_overlap | 77 | 38 | 29 | 0.6354 | 0.6696 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 62 | 17 | 65 | 0.6562 | 0.7848 |
| controlled_vs_candidate_no_context | persona_style | 11 | 5 | 128 | 0.5208 | 0.6875 |
| controlled_vs_candidate_no_context | distinct1 | 52 | 58 | 34 | 0.4792 | 0.4727 |
| controlled_vs_candidate_no_context | length_score | 49 | 64 | 31 | 0.4479 | 0.4336 |
| controlled_vs_candidate_no_context | sentence_score | 16 | 6 | 122 | 0.5347 | 0.7273 |
| controlled_vs_candidate_no_context | overall_quality | 93 | 23 | 28 | 0.7431 | 0.8017 |
| controlled_vs_baseline_no_context | context_relevance | 94 | 49 | 1 | 0.6562 | 0.6573 |
| controlled_vs_baseline_no_context | persona_consistency | 108 | 13 | 23 | 0.8299 | 0.8926 |
| controlled_vs_baseline_no_context | naturalness | 56 | 88 | 0 | 0.3889 | 0.3889 |
| controlled_vs_baseline_no_context | quest_state_correctness | 94 | 49 | 1 | 0.6562 | 0.6573 |
| controlled_vs_baseline_no_context | lore_consistency | 52 | 39 | 53 | 0.5451 | 0.5714 |
| controlled_vs_baseline_no_context | multi_turn_contradiction_safety | 0 | 18 | 126 | 0.4375 | 0.0000 |
| controlled_vs_baseline_no_context | objective_completion_support | 86 | 27 | 31 | 0.7049 | 0.7611 |
| controlled_vs_baseline_no_context | gameplay_usefulness | 105 | 39 | 0 | 0.7292 | 0.7292 |
| controlled_vs_baseline_no_context | time_pressure_acceptability | 77 | 55 | 12 | 0.5764 | 0.5833 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 65 | 11 | 68 | 0.6875 | 0.8553 |
| controlled_vs_baseline_no_context | context_overlap | 96 | 47 | 1 | 0.6701 | 0.6713 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 108 | 9 | 27 | 0.8438 | 0.9231 |
| controlled_vs_baseline_no_context | persona_style | 11 | 14 | 119 | 0.4896 | 0.4400 |
| controlled_vs_baseline_no_context | distinct1 | 25 | 109 | 10 | 0.2083 | 0.1866 |
| controlled_vs_baseline_no_context | length_score | 58 | 82 | 4 | 0.4167 | 0.4143 |
| controlled_vs_baseline_no_context | sentence_score | 47 | 5 | 92 | 0.6458 | 0.9038 |
| controlled_vs_baseline_no_context | overall_quality | 118 | 26 | 0 | 0.8194 | 0.8194 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 99 | 45 | 0 | 0.6875 | 0.6875 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 111 | 12 | 21 | 0.8438 | 0.9024 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | 51 | 93 | 0 | 0.3542 | 0.3542 |
| controlled_vs_baseline_no_context_phi3_latest | quest_state_correctness | 98 | 46 | 0 | 0.6806 | 0.6806 |
| controlled_vs_baseline_no_context_phi3_latest | lore_consistency | 54 | 37 | 53 | 0.5590 | 0.5934 |
| controlled_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0 | 0 | 144 | 0.5000 | nan |
| controlled_vs_baseline_no_context_phi3_latest | objective_completion_support | 85 | 26 | 33 | 0.7049 | 0.7658 |
| controlled_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 106 | 38 | 0 | 0.7361 | 0.7361 |
| controlled_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 74 | 53 | 17 | 0.5729 | 0.5827 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 72 | 15 | 57 | 0.6979 | 0.8276 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 97 | 47 | 0 | 0.6736 | 0.6736 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 111 | 9 | 24 | 0.8542 | 0.9250 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | 7 | 12 | 125 | 0.4826 | 0.3684 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | 20 | 118 | 6 | 0.1597 | 0.1449 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | 50 | 88 | 6 | 0.3681 | 0.3623 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 46 | 6 | 92 | 0.6389 | 0.8846 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 123 | 21 | 0 | 0.8542 | 0.8542 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 94 | 49 | 1 | 0.6562 | 0.6573 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 108 | 13 | 23 | 0.8299 | 0.8926 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 56 | 88 | 0 | 0.3889 | 0.3889 |
| proposed_contextual_controlled_vs_baseline_no_context | quest_state_correctness | 94 | 49 | 1 | 0.6562 | 0.6573 |
| proposed_contextual_controlled_vs_baseline_no_context | lore_consistency | 52 | 39 | 53 | 0.5451 | 0.5714 |
| proposed_contextual_controlled_vs_baseline_no_context | multi_turn_contradiction_safety | 0 | 18 | 126 | 0.4375 | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | objective_completion_support | 86 | 27 | 31 | 0.7049 | 0.7611 |
| proposed_contextual_controlled_vs_baseline_no_context | gameplay_usefulness | 105 | 39 | 0 | 0.7292 | 0.7292 |
| proposed_contextual_controlled_vs_baseline_no_context | time_pressure_acceptability | 77 | 55 | 12 | 0.5764 | 0.5833 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 65 | 11 | 68 | 0.6875 | 0.8553 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 96 | 47 | 1 | 0.6701 | 0.6713 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 108 | 9 | 27 | 0.8438 | 0.9231 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 11 | 14 | 119 | 0.4896 | 0.4400 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | 25 | 109 | 10 | 0.2083 | 0.1866 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 58 | 82 | 4 | 0.4167 | 0.4143 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 47 | 5 | 92 | 0.6458 | 0.9038 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 118 | 26 | 0 | 0.8194 | 0.8194 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 99 | 45 | 0 | 0.6875 | 0.6875 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 111 | 12 | 21 | 0.8438 | 0.9024 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | 51 | 93 | 0 | 0.3542 | 0.3542 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | quest_state_correctness | 98 | 46 | 0 | 0.6806 | 0.6806 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | lore_consistency | 54 | 37 | 53 | 0.5590 | 0.5934 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0 | 0 | 144 | 0.5000 | nan |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | objective_completion_support | 85 | 26 | 33 | 0.7049 | 0.7658 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 106 | 38 | 0 | 0.7361 | 0.7361 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 74 | 53 | 17 | 0.5729 | 0.5827 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 72 | 15 | 57 | 0.6979 | 0.8276 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 97 | 47 | 0 | 0.6736 | 0.6736 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 111 | 9 | 24 | 0.8542 | 0.9250 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | 7 | 12 | 125 | 0.4826 | 0.3684 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | 20 | 118 | 6 | 0.1597 | 0.1449 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | 50 | 88 | 6 | 0.3681 | 0.3623 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | 46 | 6 | 92 | 0.6389 | 0.8846 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 123 | 21 | 0 | 0.8542 | 0.8542 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.2014 | 0.2569 | 0.7431 |
| proposed_contextual | 0.0000 | 0.0000 | 0.4792 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.4306 | 0.0000 | 0.0000 |
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
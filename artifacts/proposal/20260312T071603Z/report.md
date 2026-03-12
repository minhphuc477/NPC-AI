# Proposal Alignment Evaluation Report

- Run ID: `20260312T071603Z`
- Generated: `2026-03-12T07:18:10.173373+00:00`
- Scenarios: `artifacts\proposal\20260312T071603Z\scenarios.jsonl`
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
| proposed_contextual_controlled | 0.1194 (0.0989, 0.1420) | 0.3005 (0.2755, 0.3259) | 0.8656 (0.8584, 0.8732) | 0.3321 (0.3165, 0.3488) | n/a |
| proposed_contextual | 0.0852 (0.0679, 0.1034) | 0.2123 (0.1926, 0.2325) | 0.8634 (0.8569, 0.8700) | 0.2827 (0.2716, 0.2946) | n/a |
| candidate_no_context | 0.0330 (0.0267, 0.0394) | 0.2113 (0.1902, 0.2329) | 0.8777 (0.8709, 0.8851) | 0.2612 (0.2523, 0.2708) | n/a |
| baseline_no_context | 0.0370 (0.0300, 0.0445) | 0.1625 (0.1473, 0.1800) | 0.8875 (0.8777, 0.8970) | 0.2462 (0.2383, 0.2546) | n/a |
| baseline_no_context_phi3_latest | 0.0339 (0.0274, 0.0410) | 0.1550 (0.1399, 0.1708) | 0.8901 (0.8810, 0.8988) | 0.2425 (0.2346, 0.2509) | n/a |

## Game-facing Outcome Metrics (mean, 95% CI)
| Arm | Quest-state Correctness | Lore Consistency | Contradiction Safety | Objective Completion Support | Gameplay Usefulness | Time-pressure Acceptability |
|---|---:|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2018 (0.1836, 0.2198) | 0.0582 (0.0436, 0.0734) | 1.0000 (1.0000, 1.0000) | 0.0863 (0.0743, 0.0978) | 0.3069 (0.3006, 0.3134) | 0.2948 (0.2856, 0.3043) |
| proposed_contextual | 0.1730 (0.1588, 0.1882) | 0.0394 (0.0280, 0.0511) | 1.0000 (1.0000, 1.0000) | 0.0733 (0.0611, 0.0854) | 0.2920 (0.2853, 0.2992) | 0.2931 (0.2849, 0.3016) |
| candidate_no_context | 0.1283 (0.1228, 0.1345) | 0.0102 (0.0067, 0.0140) | 1.0000 (1.0000, 1.0000) | 0.0679 (0.0571, 0.0804) | 0.2821 (0.2757, 0.2884) | 0.2985 (0.2913, 0.3066) |
| baseline_no_context | 0.1311 (0.1246, 0.1380) | 0.0136 (0.0097, 0.0181) | 1.0000 (1.0000, 1.0000) | 0.0461 (0.0385, 0.0546) | 0.2766 (0.2705, 0.2831) | 0.2917 (0.2854, 0.2985) |
| baseline_no_context_phi3_latest | 0.1290 (0.1229, 0.1355) | 0.0149 (0.0110, 0.0194) | 1.0000 (1.0000, 1.0000) | 0.0391 (0.0326, 0.0461) | 0.2743 (0.2689, 0.2797) | 0.2847 (0.2798, 0.2896) |

- Multi-turn contradiction rate is reported as `1 - contradiction_safety` in row-level outputs.

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0522 | 1.5808 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0010 | 0.0049 |
| proposed_vs_candidate_no_context | naturalness | -0.0143 | -0.0163 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.0448 | 0.3490 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0292 | 2.8556 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | 0.0054 | 0.0794 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 0.0099 | 0.0350 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | -0.0054 | -0.0182 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0670 | 2.0411 |
| proposed_vs_candidate_no_context | context_overlap | 0.0177 | 0.5287 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0021 | -0.0163 |
| proposed_vs_candidate_no_context | persona_style | 0.0137 | 0.0255 |
| proposed_vs_candidate_no_context | distinct1 | -0.0092 | -0.0097 |
| proposed_vs_candidate_no_context | length_score | -0.0447 | -0.0857 |
| proposed_vs_candidate_no_context | sentence_score | -0.0170 | -0.0179 |
| proposed_vs_candidate_no_context | overall_quality | 0.0215 | 0.0822 |
| proposed_vs_baseline_no_context | context_relevance | 0.0482 | 1.3007 |
| proposed_vs_baseline_no_context | persona_consistency | 0.0498 | 0.3065 |
| proposed_vs_baseline_no_context | naturalness | -0.0241 | -0.0272 |
| proposed_vs_baseline_no_context | quest_state_correctness | 0.0420 | 0.3201 |
| proposed_vs_baseline_no_context | lore_consistency | 0.0258 | 1.9028 |
| proposed_vs_baseline_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| proposed_vs_baseline_no_context | objective_completion_support | 0.0272 | 0.5896 |
| proposed_vs_baseline_no_context | gameplay_usefulness | 0.0154 | 0.0556 |
| proposed_vs_baseline_no_context | time_pressure_acceptability | 0.0013 | 0.0046 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0623 | 1.6620 |
| proposed_vs_baseline_no_context | context_overlap | 0.0152 | 0.4226 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 0.0676 | 1.1194 |
| proposed_vs_baseline_no_context | persona_style | -0.0214 | -0.0374 |
| proposed_vs_baseline_no_context | distinct1 | -0.0400 | -0.0410 |
| proposed_vs_baseline_no_context | length_score | -0.0757 | -0.1370 |
| proposed_vs_baseline_no_context | sentence_score | 0.0701 | 0.0814 |
| proposed_vs_baseline_no_context | overall_quality | 0.0365 | 0.1482 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 0.0513 | 1.5098 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | 0.0573 | 0.3699 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | -0.0267 | -0.0300 |
| proposed_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.0440 | 0.3411 |
| proposed_vs_baseline_no_context_phi3_latest | lore_consistency | 0.0245 | 1.6440 |
| proposed_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.0342 | 0.8735 |
| proposed_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 0.0177 | 0.0644 |
| proposed_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 0.0084 | 0.0293 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.0671 | 2.0577 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 0.0142 | 0.3829 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.0756 | 1.4447 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | -0.0159 | -0.0281 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | -0.0407 | -0.0416 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | -0.0810 | -0.1452 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | 0.0580 | 0.0663 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | 0.0402 | 0.1659 |
| controlled_vs_proposed_raw | context_relevance | 0.0342 | 0.4018 |
| controlled_vs_proposed_raw | persona_consistency | 0.0882 | 0.4155 |
| controlled_vs_proposed_raw | naturalness | 0.0022 | 0.0026 |
| controlled_vs_proposed_raw | quest_state_correctness | 0.0287 | 0.1660 |
| controlled_vs_proposed_raw | lore_consistency | 0.0188 | 0.4780 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0131 | 0.1783 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0149 | 0.0510 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 0.0017 | 0.0059 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.0449 | 0.4498 |
| controlled_vs_proposed_raw | context_overlap | 0.0094 | 0.1834 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1098 | 0.8576 |
| controlled_vs_proposed_raw | persona_style | 0.0020 | 0.0037 |
| controlled_vs_proposed_raw | distinct1 | -0.0051 | -0.0054 |
| controlled_vs_proposed_raw | length_score | -0.0002 | -0.0003 |
| controlled_vs_proposed_raw | sentence_score | 0.0443 | 0.0475 |
| controlled_vs_proposed_raw | overall_quality | 0.0494 | 0.1748 |
| controlled_vs_candidate_no_context | context_relevance | 0.0864 | 2.6178 |
| controlled_vs_candidate_no_context | persona_consistency | 0.0892 | 0.4225 |
| controlled_vs_candidate_no_context | naturalness | -0.0121 | -0.0138 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.0735 | 0.5729 |
| controlled_vs_candidate_no_context | lore_consistency | 0.0480 | 4.6983 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | 0.0185 | 0.2718 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0248 | 0.0879 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | -0.0037 | -0.0123 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.1118 | 3.4091 |
| controlled_vs_candidate_no_context | context_overlap | 0.0271 | 0.8090 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1076 | 0.8274 |
| controlled_vs_candidate_no_context | persona_style | 0.0157 | 0.0293 |
| controlled_vs_candidate_no_context | distinct1 | -0.0143 | -0.0151 |
| controlled_vs_candidate_no_context | length_score | -0.0448 | -0.0860 |
| controlled_vs_candidate_no_context | sentence_score | 0.0273 | 0.0287 |
| controlled_vs_candidate_no_context | overall_quality | 0.0709 | 0.2713 |
| controlled_vs_baseline_no_context | context_relevance | 0.0824 | 2.2251 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1380 | 0.8493 |
| controlled_vs_baseline_no_context | naturalness | -0.0219 | -0.0247 |
| controlled_vs_baseline_no_context | quest_state_correctness | 0.0707 | 0.5392 |
| controlled_vs_baseline_no_context | lore_consistency | 0.0447 | 3.2902 |
| controlled_vs_baseline_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_baseline_no_context | objective_completion_support | 0.0402 | 0.8730 |
| controlled_vs_baseline_no_context | gameplay_usefulness | 0.0303 | 0.1094 |
| controlled_vs_baseline_no_context | time_pressure_acceptability | 0.0031 | 0.0105 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.1072 | 2.8595 |
| controlled_vs_baseline_no_context | context_overlap | 0.0246 | 0.6835 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1773 | 2.9370 |
| controlled_vs_baseline_no_context | persona_style | -0.0193 | -0.0339 |
| controlled_vs_baseline_no_context | distinct1 | -0.0451 | -0.0462 |
| controlled_vs_baseline_no_context | length_score | -0.0759 | -0.1373 |
| controlled_vs_baseline_no_context | sentence_score | 0.1144 | 0.1328 |
| controlled_vs_baseline_no_context | overall_quality | 0.0859 | 0.3489 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.0855 | 2.5182 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1455 | 0.9391 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | -0.0244 | -0.0275 |
| controlled_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.0727 | 0.5637 |
| controlled_vs_baseline_no_context_phi3_latest | lore_consistency | 0.0433 | 2.9078 |
| controlled_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.0472 | 1.2075 |
| controlled_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 0.0326 | 0.1187 |
| controlled_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 0.0101 | 0.0354 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.1120 | 3.4332 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0236 | 0.6365 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1854 | 3.5413 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | -0.0139 | -0.0245 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0458 | -0.0468 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | -0.0812 | -0.1455 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.1023 | 0.1170 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.0896 | 0.3696 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.0824 | 2.2251 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1380 | 0.8493 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0219 | -0.0247 |
| proposed_contextual_controlled_vs_baseline_no_context | quest_state_correctness | 0.0707 | 0.5392 |
| proposed_contextual_controlled_vs_baseline_no_context | lore_consistency | 0.0447 | 3.2902 |
| proposed_contextual_controlled_vs_baseline_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | objective_completion_support | 0.0402 | 0.8730 |
| proposed_contextual_controlled_vs_baseline_no_context | gameplay_usefulness | 0.0303 | 0.1094 |
| proposed_contextual_controlled_vs_baseline_no_context | time_pressure_acceptability | 0.0031 | 0.0105 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.1072 | 2.8595 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0246 | 0.6835 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1773 | 2.9370 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | -0.0193 | -0.0339 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0451 | -0.0462 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | -0.0759 | -0.1373 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.1144 | 0.1328 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.0859 | 0.3489 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.0855 | 2.5182 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1455 | 0.9391 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | -0.0244 | -0.0275 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.0727 | 0.5637 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | lore_consistency | 0.0433 | 2.9078 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.0472 | 1.2075 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 0.0326 | 0.1187 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 0.0101 | 0.0354 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.1120 | 3.4332 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0236 | 0.6365 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1854 | 3.5413 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | -0.0139 | -0.0245 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0458 | -0.0468 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | -0.0812 | -0.1455 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.1023 | 0.1170 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.0896 | 0.3696 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0522 | (0.0357, 0.0705) | 0.0000 | 0.0522 | (0.0321, 0.0720) | 0.0000 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0010 | (-0.0182, 0.0191) | 0.4780 | 0.0010 | (-0.0119, 0.0142) | 0.4343 |
| proposed_vs_candidate_no_context | naturalness | -0.0143 | (-0.0234, -0.0051) | 1.0000 | -0.0143 | (-0.0250, -0.0056) | 1.0000 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.0448 | (0.0298, 0.0602) | 0.0000 | 0.0448 | (0.0270, 0.0610) | 0.0000 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0292 | (0.0185, 0.0401) | 0.0000 | 0.0292 | (0.0175, 0.0406) | 0.0000 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | 0.0054 | (-0.0028, 0.0137) | 0.1043 | 0.0054 | (-0.0052, 0.0143) | 0.1453 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 0.0099 | (0.0035, 0.0162) | 0.0013 | 0.0099 | (0.0039, 0.0163) | 0.0007 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | -0.0054 | (-0.0120, 0.0016) | 0.9303 | -0.0054 | (-0.0123, 0.0015) | 0.9400 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0670 | (0.0451, 0.0906) | 0.0000 | 0.0670 | (0.0409, 0.0920) | 0.0000 |
| proposed_vs_candidate_no_context | context_overlap | 0.0177 | (0.0108, 0.0245) | 0.0000 | 0.0177 | (0.0081, 0.0282) | 0.0000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0021 | (-0.0248, 0.0200) | 0.5677 | -0.0021 | (-0.0174, 0.0127) | 0.6043 |
| proposed_vs_candidate_no_context | persona_style | 0.0137 | (-0.0022, 0.0305) | 0.0480 | 0.0137 | (-0.0018, 0.0316) | 0.0453 |
| proposed_vs_candidate_no_context | distinct1 | -0.0092 | (-0.0169, -0.0015) | 0.9903 | -0.0092 | (-0.0144, -0.0043) | 1.0000 |
| proposed_vs_candidate_no_context | length_score | -0.0447 | (-0.0870, -0.0037) | 0.9833 | -0.0447 | (-0.0935, -0.0088) | 0.9993 |
| proposed_vs_candidate_no_context | sentence_score | -0.0170 | (-0.0413, 0.0073) | 0.9347 | -0.0170 | (-0.0316, -0.0024) | 0.9960 |
| proposed_vs_candidate_no_context | overall_quality | 0.0215 | (0.0098, 0.0330) | 0.0000 | 0.0215 | (0.0089, 0.0333) | 0.0000 |
| proposed_vs_baseline_no_context | context_relevance | 0.0482 | (0.0314, 0.0668) | 0.0000 | 0.0482 | (0.0226, 0.0706) | 0.0000 |
| proposed_vs_baseline_no_context | persona_consistency | 0.0498 | (0.0279, 0.0708) | 0.0000 | 0.0498 | (0.0181, 0.0797) | 0.0010 |
| proposed_vs_baseline_no_context | naturalness | -0.0241 | (-0.0352, -0.0123) | 1.0000 | -0.0241 | (-0.0376, -0.0116) | 1.0000 |
| proposed_vs_baseline_no_context | quest_state_correctness | 0.0420 | (0.0272, 0.0578) | 0.0000 | 0.0420 | (0.0230, 0.0601) | 0.0000 |
| proposed_vs_baseline_no_context | lore_consistency | 0.0258 | (0.0150, 0.0371) | 0.0000 | 0.0258 | (0.0124, 0.0386) | 0.0000 |
| proposed_vs_baseline_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_baseline_no_context | objective_completion_support | 0.0272 | (0.0137, 0.0403) | 0.0003 | 0.0272 | (0.0080, 0.0475) | 0.0023 |
| proposed_vs_baseline_no_context | gameplay_usefulness | 0.0154 | (0.0060, 0.0247) | 0.0010 | 0.0154 | (0.0022, 0.0273) | 0.0117 |
| proposed_vs_baseline_no_context | time_pressure_acceptability | 0.0013 | (-0.0090, 0.0116) | 0.3953 | 0.0013 | (-0.0098, 0.0136) | 0.4363 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0623 | (0.0402, 0.0861) | 0.0000 | 0.0623 | (0.0329, 0.0894) | 0.0000 |
| proposed_vs_baseline_no_context | context_overlap | 0.0152 | (0.0075, 0.0229) | 0.0000 | 0.0152 | (0.0003, 0.0306) | 0.0223 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 0.0676 | (0.0416, 0.0926) | 0.0000 | 0.0676 | (0.0277, 0.0987) | 0.0023 |
| proposed_vs_baseline_no_context | persona_style | -0.0214 | (-0.0405, -0.0026) | 0.9857 | -0.0214 | (-0.0685, 0.0121) | 0.8013 |
| proposed_vs_baseline_no_context | distinct1 | -0.0400 | (-0.0503, -0.0304) | 1.0000 | -0.0400 | (-0.0588, -0.0253) | 1.0000 |
| proposed_vs_baseline_no_context | length_score | -0.0757 | (-0.1236, -0.0264) | 0.9993 | -0.0757 | (-0.1322, -0.0218) | 0.9983 |
| proposed_vs_baseline_no_context | sentence_score | 0.0701 | (0.0365, 0.1042) | 0.0000 | 0.0701 | (0.0194, 0.1236) | 0.0013 |
| proposed_vs_baseline_no_context | overall_quality | 0.0365 | (0.0238, 0.0492) | 0.0000 | 0.0365 | (0.0158, 0.0554) | 0.0003 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 0.0513 | (0.0333, 0.0698) | 0.0000 | 0.0513 | (0.0255, 0.0752) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | 0.0573 | (0.0354, 0.0796) | 0.0000 | 0.0573 | (0.0286, 0.0884) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | -0.0267 | (-0.0375, -0.0160) | 1.0000 | -0.0267 | (-0.0384, -0.0144) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.0440 | (0.0295, 0.0600) | 0.0000 | 0.0440 | (0.0237, 0.0631) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | lore_consistency | 0.0245 | (0.0135, 0.0365) | 0.0000 | 0.0245 | (0.0087, 0.0398) | 0.0020 |
| proposed_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.0342 | (0.0215, 0.0465) | 0.0000 | 0.0342 | (0.0131, 0.0584) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 0.0177 | (0.0092, 0.0258) | 0.0000 | 0.0177 | (0.0059, 0.0309) | 0.0013 |
| proposed_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 0.0084 | (-0.0011, 0.0175) | 0.0370 | 0.0084 | (-0.0049, 0.0226) | 0.1227 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.0671 | (0.0446, 0.0908) | 0.0000 | 0.0671 | (0.0366, 0.0959) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 0.0142 | (0.0072, 0.0222) | 0.0000 | 0.0142 | (0.0010, 0.0279) | 0.0190 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.0756 | (0.0488, 0.1032) | 0.0000 | 0.0756 | (0.0403, 0.1084) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | -0.0159 | (-0.0338, 0.0015) | 0.9627 | -0.0159 | (-0.0612, 0.0196) | 0.7873 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | -0.0407 | (-0.0512, -0.0303) | 1.0000 | -0.0407 | (-0.0595, -0.0251) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | -0.0810 | (-0.1287, -0.0317) | 0.9997 | -0.0810 | (-0.1250, -0.0329) | 0.9993 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | 0.0580 | (0.0240, 0.0941) | 0.0007 | 0.0580 | (0.0049, 0.1139) | 0.0130 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | 0.0402 | (0.0274, 0.0530) | 0.0000 | 0.0402 | (0.0202, 0.0590) | 0.0003 |
| controlled_vs_proposed_raw | context_relevance | 0.0347 | (0.0102, 0.0592) | 0.0013 | 0.0347 | (0.0034, 0.0681) | 0.0127 |
| controlled_vs_proposed_raw | persona_consistency | 0.0874 | (0.0630, 0.1125) | 0.0000 | 0.0874 | (0.0549, 0.1249) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0023 | (-0.0073, 0.0118) | 0.3277 | 0.0023 | (-0.0092, 0.0155) | 0.3710 |
| controlled_vs_proposed_raw | quest_state_correctness | 0.0292 | (0.0078, 0.0505) | 0.0030 | 0.0292 | (0.0038, 0.0554) | 0.0097 |
| controlled_vs_proposed_raw | lore_consistency | 0.0186 | (0.0006, 0.0367) | 0.0223 | 0.0186 | (-0.0084, 0.0459) | 0.0977 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0126 | (0.0011, 0.0248) | 0.0170 | 0.0126 | (-0.0050, 0.0322) | 0.0863 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0147 | (0.0068, 0.0230) | 0.0003 | 0.0147 | (0.0045, 0.0264) | 0.0010 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 0.0013 | (-0.0086, 0.0113) | 0.4087 | 0.0013 | (-0.0076, 0.0097) | 0.3727 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.0456 | (0.0130, 0.0784) | 0.0033 | 0.0456 | (0.0048, 0.0863) | 0.0150 |
| controlled_vs_proposed_raw | context_overlap | 0.0095 | (0.0005, 0.0191) | 0.0197 | 0.0095 | (-0.0050, 0.0221) | 0.1003 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1089 | (0.0780, 0.1402) | 0.0000 | 0.1089 | (0.0699, 0.1547) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.0017 | (-0.0106, 0.0143) | 0.3757 | 0.0017 | (-0.0197, 0.0241) | 0.4530 |
| controlled_vs_proposed_raw | distinct1 | -0.0049 | (-0.0131, 0.0036) | 0.8617 | -0.0049 | (-0.0151, 0.0046) | 0.8347 |
| controlled_vs_proposed_raw | length_score | -0.0007 | (-0.0452, 0.0436) | 0.4993 | -0.0007 | (-0.0688, 0.0627) | 0.5003 |
| controlled_vs_proposed_raw | sentence_score | 0.0448 | (0.0171, 0.0720) | 0.0007 | 0.0448 | (0.0194, 0.0743) | 0.0000 |
| controlled_vs_proposed_raw | overall_quality | 0.0494 | (0.0326, 0.0657) | 0.0000 | 0.0494 | (0.0303, 0.0676) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.0867 | (0.0644, 0.1094) | 0.0000 | 0.0867 | (0.0510, 0.1237) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.0894 | (0.0598, 0.1188) | 0.0000 | 0.0894 | (0.0481, 0.1303) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | -0.0123 | (-0.0219, -0.0031) | 0.9950 | -0.0123 | (-0.0212, -0.0033) | 0.9950 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.0738 | (0.0557, 0.0928) | 0.0000 | 0.0738 | (0.0438, 0.1043) | 0.0000 |
| controlled_vs_candidate_no_context | lore_consistency | 0.0480 | (0.0321, 0.0645) | 0.0000 | 0.0480 | (0.0176, 0.0764) | 0.0003 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | 0.0183 | (0.0068, 0.0299) | 0.0007 | 0.0183 | (0.0073, 0.0292) | 0.0000 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0245 | (0.0169, 0.0321) | 0.0000 | 0.0245 | (0.0138, 0.0357) | 0.0000 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | -0.0040 | (-0.0133, 0.0052) | 0.8120 | -0.0040 | (-0.0102, 0.0017) | 0.9173 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.1123 | (0.0841, 0.1419) | 0.0000 | 0.1123 | (0.0633, 0.1607) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0270 | (0.0192, 0.0347) | 0.0000 | 0.0270 | (0.0172, 0.0380) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1079 | (0.0721, 0.1446) | 0.0000 | 0.1079 | (0.0620, 0.1556) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.0155 | (-0.0018, 0.0341) | 0.0447 | 0.0155 | (-0.0047, 0.0350) | 0.0753 |
| controlled_vs_candidate_no_context | distinct1 | -0.0140 | (-0.0221, -0.0059) | 0.9997 | -0.0140 | (-0.0230, -0.0062) | 1.0000 |
| controlled_vs_candidate_no_context | length_score | -0.0469 | (-0.0881, -0.0030) | 0.9850 | -0.0469 | (-0.0984, -0.0000) | 0.9753 |
| controlled_vs_candidate_no_context | sentence_score | 0.0276 | (0.0031, 0.0538) | 0.0117 | 0.0276 | (0.0073, 0.0535) | 0.0003 |
| controlled_vs_candidate_no_context | overall_quality | 0.0710 | (0.0546, 0.0871) | 0.0000 | 0.0710 | (0.0495, 0.0949) | 0.0000 |
| controlled_vs_baseline_no_context | context_relevance | 0.0832 | (0.0620, 0.1056) | 0.0000 | 0.0832 | (0.0513, 0.1193) | 0.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1376 | (0.1118, 0.1640) | 0.0000 | 0.1376 | (0.0896, 0.1805) | 0.0000 |
| controlled_vs_baseline_no_context | naturalness | -0.0217 | (-0.0341, -0.0099) | 1.0000 | -0.0217 | (-0.0338, -0.0102) | 1.0000 |
| controlled_vs_baseline_no_context | quest_state_correctness | 0.0714 | (0.0532, 0.0903) | 0.0000 | 0.0714 | (0.0452, 0.0998) | 0.0000 |
| controlled_vs_baseline_no_context | lore_consistency | 0.0448 | (0.0298, 0.0603) | 0.0000 | 0.0448 | (0.0203, 0.0703) | 0.0000 |
| controlled_vs_baseline_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_baseline_no_context | objective_completion_support | 0.0399 | (0.0276, 0.0524) | 0.0000 | 0.0399 | (0.0195, 0.0609) | 0.0000 |
| controlled_vs_baseline_no_context | gameplay_usefulness | 0.0304 | (0.0213, 0.0387) | 0.0000 | 0.0304 | (0.0238, 0.0372) | 0.0000 |
| controlled_vs_baseline_no_context | time_pressure_acceptability | 0.0030 | (-0.0069, 0.0123) | 0.2750 | 0.0030 | (-0.0112, 0.0172) | 0.3467 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.1083 | (0.0792, 0.1378) | 0.0000 | 0.1083 | (0.0673, 0.1544) | 0.0000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0247 | (0.0165, 0.0334) | 0.0000 | 0.0247 | (0.0113, 0.0372) | 0.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1769 | (0.1451, 0.2104) | 0.0000 | 0.1769 | (0.1190, 0.2271) | 0.0000 |
| controlled_vs_baseline_no_context | persona_style | -0.0198 | (-0.0380, -0.0020) | 0.9850 | -0.0198 | (-0.0505, 0.0047) | 0.9280 |
| controlled_vs_baseline_no_context | distinct1 | -0.0454 | (-0.0545, -0.0366) | 1.0000 | -0.0454 | (-0.0620, -0.0332) | 1.0000 |
| controlled_vs_baseline_no_context | length_score | -0.0751 | (-0.1249, -0.0242) | 0.9963 | -0.0751 | (-0.1331, -0.0223) | 0.9987 |
| controlled_vs_baseline_no_context | sentence_score | 0.1154 | (0.0836, 0.1476) | 0.0000 | 0.1154 | (0.0712, 0.1549) | 0.0000 |
| controlled_vs_baseline_no_context | overall_quality | 0.0862 | (0.0693, 0.1031) | 0.0000 | 0.0862 | (0.0598, 0.1111) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.0854 | (0.0629, 0.1083) | 0.0000 | 0.0854 | (0.0522, 0.1235) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1451 | (0.1176, 0.1740) | 0.0000 | 0.1451 | (0.1049, 0.1836) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | -0.0246 | (-0.0366, -0.0128) | 1.0000 | -0.0246 | (-0.0361, -0.0129) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.0727 | (0.0533, 0.0925) | 0.0000 | 0.0727 | (0.0425, 0.1042) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | lore_consistency | 0.0432 | (0.0281, 0.0582) | 0.0000 | 0.0432 | (0.0202, 0.0670) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.0470 | (0.0353, 0.0598) | 0.0000 | 0.0470 | (0.0316, 0.0661) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 0.0325 | (0.0244, 0.0406) | 0.0000 | 0.0325 | (0.0246, 0.0405) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 0.0100 | (0.0007, 0.0195) | 0.0183 | 0.0100 | (-0.0012, 0.0245) | 0.0467 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.1118 | (0.0832, 0.1417) | 0.0000 | 0.1118 | (0.0660, 0.1596) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0237 | (0.0155, 0.0320) | 0.0000 | 0.0237 | (0.0132, 0.0343) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1850 | (0.1491, 0.2190) | 0.0000 | 0.1850 | (0.1344, 0.2289) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | -0.0143 | (-0.0282, -0.0010) | 0.9823 | -0.0143 | (-0.0384, 0.0063) | 0.8963 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0459 | (-0.0548, -0.0366) | 1.0000 | -0.0459 | (-0.0635, -0.0306) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | -0.0809 | (-0.1298, -0.0324) | 0.9990 | -0.0809 | (-0.1345, -0.0326) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.1007 | (0.0710, 0.1322) | 0.0000 | 0.1007 | (0.0564, 0.1483) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.0894 | (0.0723, 0.1070) | 0.0000 | 0.0894 | (0.0651, 0.1123) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.0832 | (0.0616, 0.1056) | 0.0000 | 0.0832 | (0.0513, 0.1198) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1376 | (0.1115, 0.1650) | 0.0000 | 0.1376 | (0.0925, 0.1821) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0217 | (-0.0342, -0.0097) | 1.0000 | -0.0217 | (-0.0345, -0.0106) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | quest_state_correctness | 0.0714 | (0.0530, 0.0903) | 0.0000 | 0.0714 | (0.0461, 0.0999) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | lore_consistency | 0.0448 | (0.0294, 0.0606) | 0.0000 | 0.0448 | (0.0212, 0.0683) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | objective_completion_support | 0.0399 | (0.0279, 0.0521) | 0.0000 | 0.0399 | (0.0199, 0.0592) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | gameplay_usefulness | 0.0304 | (0.0213, 0.0396) | 0.0000 | 0.0304 | (0.0239, 0.0370) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | time_pressure_acceptability | 0.0030 | (-0.0066, 0.0129) | 0.2670 | 0.0030 | (-0.0113, 0.0169) | 0.3547 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.1083 | (0.0805, 0.1372) | 0.0000 | 0.1083 | (0.0663, 0.1546) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0247 | (0.0163, 0.0333) | 0.0000 | 0.0247 | (0.0114, 0.0376) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1769 | (0.1457, 0.2099) | 0.0000 | 0.1769 | (0.1204, 0.2274) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | -0.0198 | (-0.0377, -0.0022) | 0.9857 | -0.0198 | (-0.0512, 0.0043) | 0.9313 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0454 | (-0.0547, -0.0366) | 1.0000 | -0.0454 | (-0.0612, -0.0331) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | -0.0751 | (-0.1245, -0.0256) | 0.9983 | -0.0751 | (-0.1340, -0.0221) | 0.9990 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.1154 | (0.0829, 0.1479) | 0.0000 | 0.1154 | (0.0700, 0.1549) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.0862 | (0.0697, 0.1029) | 0.0000 | 0.0862 | (0.0586, 0.1099) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.0854 | (0.0630, 0.1074) | 0.0000 | 0.0854 | (0.0500, 0.1231) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1451 | (0.1168, 0.1728) | 0.0000 | 0.1451 | (0.1059, 0.1835) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | -0.0246 | (-0.0365, -0.0129) | 1.0000 | -0.0246 | (-0.0366, -0.0129) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.0727 | (0.0549, 0.0923) | 0.0000 | 0.0727 | (0.0439, 0.1048) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | lore_consistency | 0.0432 | (0.0279, 0.0589) | 0.0000 | 0.0432 | (0.0191, 0.0664) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.0470 | (0.0346, 0.0588) | 0.0000 | 0.0470 | (0.0312, 0.0662) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 0.0325 | (0.0247, 0.0406) | 0.0000 | 0.0325 | (0.0247, 0.0403) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 0.0100 | (0.0005, 0.0196) | 0.0217 | 0.0100 | (-0.0014, 0.0236) | 0.0497 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.1118 | (0.0826, 0.1406) | 0.0000 | 0.1118 | (0.0661, 0.1629) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0237 | (0.0157, 0.0319) | 0.0000 | 0.0237 | (0.0132, 0.0337) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1850 | (0.1518, 0.2206) | 0.0000 | 0.1850 | (0.1365, 0.2263) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | -0.0143 | (-0.0288, -0.0011) | 0.9843 | -0.0143 | (-0.0393, 0.0058) | 0.8910 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0459 | (-0.0552, -0.0367) | 1.0000 | -0.0459 | (-0.0628, -0.0304) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | -0.0809 | (-0.1315, -0.0319) | 0.9997 | -0.0809 | (-0.1345, -0.0313) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.1007 | (0.0696, 0.1304) | 0.0000 | 0.1007 | (0.0573, 0.1472) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.0894 | (0.0728, 0.1074) | 0.0000 | 0.0894 | (0.0654, 0.1122) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 68 | 28 | 48 | 0.6389 | 0.7083 |
| proposed_vs_candidate_no_context | persona_consistency | 34 | 25 | 85 | 0.5312 | 0.5763 |
| proposed_vs_candidate_no_context | naturalness | 35 | 61 | 48 | 0.4097 | 0.3646 |
| proposed_vs_candidate_no_context | quest_state_correctness | 68 | 28 | 48 | 0.6389 | 0.7083 |
| proposed_vs_candidate_no_context | lore_consistency | 55 | 13 | 76 | 0.6458 | 0.8088 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 144 | 0.5000 | nan |
| proposed_vs_candidate_no_context | objective_completion_support | 36 | 27 | 81 | 0.5312 | 0.5714 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 62 | 34 | 48 | 0.5972 | 0.6458 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | 32 | 47 | 65 | 0.4479 | 0.4051 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 55 | 12 | 77 | 0.6493 | 0.8209 |
| proposed_vs_candidate_no_context | context_overlap | 65 | 30 | 49 | 0.6215 | 0.6842 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 26 | 23 | 95 | 0.5104 | 0.5306 |
| proposed_vs_candidate_no_context | persona_style | 14 | 5 | 125 | 0.5312 | 0.7368 |
| proposed_vs_candidate_no_context | distinct1 | 31 | 49 | 64 | 0.4375 | 0.3875 |
| proposed_vs_candidate_no_context | length_score | 39 | 53 | 52 | 0.4514 | 0.4239 |
| proposed_vs_candidate_no_context | sentence_score | 9 | 16 | 119 | 0.4757 | 0.3600 |
| proposed_vs_candidate_no_context | overall_quality | 66 | 30 | 48 | 0.6250 | 0.6875 |
| proposed_vs_baseline_no_context | context_relevance | 82 | 57 | 5 | 0.5868 | 0.5899 |
| proposed_vs_baseline_no_context | persona_consistency | 66 | 24 | 54 | 0.6458 | 0.7333 |
| proposed_vs_baseline_no_context | naturalness | 52 | 92 | 0 | 0.3611 | 0.3611 |
| proposed_vs_baseline_no_context | quest_state_correctness | 81 | 58 | 5 | 0.5799 | 0.5827 |
| proposed_vs_baseline_no_context | lore_consistency | 54 | 40 | 50 | 0.5486 | 0.5745 |
| proposed_vs_baseline_no_context | multi_turn_contradiction_safety | 0 | 0 | 144 | 0.5000 | nan |
| proposed_vs_baseline_no_context | objective_completion_support | 66 | 45 | 33 | 0.5729 | 0.5946 |
| proposed_vs_baseline_no_context | gameplay_usefulness | 91 | 53 | 0 | 0.6319 | 0.6319 |
| proposed_vs_baseline_no_context | time_pressure_acceptability | 62 | 70 | 12 | 0.4722 | 0.4697 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 57 | 17 | 70 | 0.6389 | 0.7703 |
| proposed_vs_baseline_no_context | context_overlap | 84 | 54 | 6 | 0.6042 | 0.6087 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 62 | 18 | 64 | 0.6528 | 0.7750 |
| proposed_vs_baseline_no_context | persona_style | 7 | 20 | 117 | 0.4549 | 0.2593 |
| proposed_vs_baseline_no_context | distinct1 | 27 | 97 | 20 | 0.2569 | 0.2177 |
| proposed_vs_baseline_no_context | length_score | 54 | 85 | 5 | 0.3924 | 0.3885 |
| proposed_vs_baseline_no_context | sentence_score | 41 | 13 | 90 | 0.5972 | 0.7593 |
| proposed_vs_baseline_no_context | overall_quality | 97 | 47 | 0 | 0.6736 | 0.6736 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 88 | 53 | 3 | 0.6215 | 0.6241 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | 67 | 26 | 51 | 0.6424 | 0.7204 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | 49 | 94 | 1 | 0.3438 | 0.3427 |
| proposed_vs_baseline_no_context_phi3_latest | quest_state_correctness | 88 | 53 | 3 | 0.6215 | 0.6241 |
| proposed_vs_baseline_no_context_phi3_latest | lore_consistency | 53 | 40 | 51 | 0.5451 | 0.5699 |
| proposed_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0 | 0 | 144 | 0.5000 | nan |
| proposed_vs_baseline_no_context_phi3_latest | objective_completion_support | 73 | 36 | 35 | 0.6285 | 0.6697 |
| proposed_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 95 | 49 | 0 | 0.6597 | 0.6597 |
| proposed_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 72 | 53 | 19 | 0.5660 | 0.5760 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 58 | 14 | 72 | 0.6528 | 0.8056 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 84 | 55 | 5 | 0.6007 | 0.6043 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 65 | 17 | 62 | 0.6667 | 0.7927 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | 7 | 19 | 118 | 0.4583 | 0.2692 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | 35 | 93 | 16 | 0.2986 | 0.2734 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | 50 | 85 | 9 | 0.3785 | 0.3704 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | 39 | 16 | 89 | 0.5799 | 0.7091 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | 103 | 41 | 0 | 0.7153 | 0.7153 |
| controlled_vs_proposed_raw | context_relevance | 70 | 46 | 27 | 0.5839 | 0.6034 |
| controlled_vs_proposed_raw | persona_consistency | 70 | 12 | 61 | 0.7028 | 0.8537 |
| controlled_vs_proposed_raw | naturalness | 59 | 57 | 27 | 0.5070 | 0.5086 |
| controlled_vs_proposed_raw | quest_state_correctness | 68 | 48 | 27 | 0.5699 | 0.5862 |
| controlled_vs_proposed_raw | lore_consistency | 42 | 40 | 61 | 0.5070 | 0.5122 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0 | 0 | 143 | 0.5000 | nan |
| controlled_vs_proposed_raw | objective_completion_support | 44 | 39 | 60 | 0.5175 | 0.5301 |
| controlled_vs_proposed_raw | gameplay_usefulness | 67 | 49 | 27 | 0.5629 | 0.5776 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 45 | 48 | 50 | 0.4895 | 0.4839 |
| controlled_vs_proposed_raw | context_keyword_coverage | 52 | 31 | 60 | 0.5734 | 0.6265 |
| controlled_vs_proposed_raw | context_overlap | 68 | 48 | 27 | 0.5699 | 0.5862 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 67 | 9 | 67 | 0.7028 | 0.8816 |
| controlled_vs_proposed_raw | persona_style | 12 | 11 | 120 | 0.5035 | 0.5217 |
| controlled_vs_proposed_raw | distinct1 | 55 | 55 | 33 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | length_score | 53 | 61 | 29 | 0.4720 | 0.4649 |
| controlled_vs_proposed_raw | sentence_score | 26 | 7 | 110 | 0.5664 | 0.7879 |
| controlled_vs_proposed_raw | overall_quality | 84 | 32 | 27 | 0.6818 | 0.7241 |
| controlled_vs_candidate_no_context | context_relevance | 87 | 33 | 23 | 0.6888 | 0.7250 |
| controlled_vs_candidate_no_context | persona_consistency | 76 | 14 | 53 | 0.7168 | 0.8444 |
| controlled_vs_candidate_no_context | naturalness | 51 | 68 | 24 | 0.4406 | 0.4286 |
| controlled_vs_candidate_no_context | quest_state_correctness | 89 | 31 | 23 | 0.7028 | 0.7417 |
| controlled_vs_candidate_no_context | lore_consistency | 50 | 23 | 70 | 0.5944 | 0.6849 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 143 | 0.5000 | nan |
| controlled_vs_candidate_no_context | objective_completion_support | 54 | 32 | 57 | 0.5769 | 0.6279 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 83 | 37 | 23 | 0.6608 | 0.6917 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 45 | 50 | 48 | 0.4825 | 0.4737 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 66 | 15 | 62 | 0.6783 | 0.8148 |
| controlled_vs_candidate_no_context | context_overlap | 82 | 37 | 24 | 0.6573 | 0.6891 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 69 | 13 | 61 | 0.6958 | 0.8415 |
| controlled_vs_candidate_no_context | persona_style | 19 | 8 | 116 | 0.5385 | 0.7037 |
| controlled_vs_candidate_no_context | distinct1 | 45 | 70 | 28 | 0.4126 | 0.3913 |
| controlled_vs_candidate_no_context | length_score | 45 | 72 | 26 | 0.4056 | 0.3846 |
| controlled_vs_candidate_no_context | sentence_score | 19 | 7 | 117 | 0.5420 | 0.7308 |
| controlled_vs_candidate_no_context | overall_quality | 94 | 26 | 23 | 0.7378 | 0.7833 |
| controlled_vs_baseline_no_context | context_relevance | 86 | 55 | 2 | 0.6084 | 0.6099 |
| controlled_vs_baseline_no_context | persona_consistency | 106 | 13 | 24 | 0.8252 | 0.8908 |
| controlled_vs_baseline_no_context | naturalness | 54 | 89 | 0 | 0.3776 | 0.3776 |
| controlled_vs_baseline_no_context | quest_state_correctness | 88 | 53 | 2 | 0.6224 | 0.6241 |
| controlled_vs_baseline_no_context | lore_consistency | 52 | 39 | 52 | 0.5455 | 0.5714 |
| controlled_vs_baseline_no_context | multi_turn_contradiction_safety | 0 | 0 | 143 | 0.5000 | nan |
| controlled_vs_baseline_no_context | objective_completion_support | 80 | 36 | 27 | 0.6538 | 0.6897 |
| controlled_vs_baseline_no_context | gameplay_usefulness | 98 | 45 | 0 | 0.6853 | 0.6853 |
| controlled_vs_baseline_no_context | time_pressure_acceptability | 67 | 66 | 10 | 0.5035 | 0.5038 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 67 | 21 | 55 | 0.6608 | 0.7614 |
| controlled_vs_baseline_no_context | context_overlap | 93 | 48 | 2 | 0.6573 | 0.6596 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 105 | 6 | 32 | 0.8462 | 0.9459 |
| controlled_vs_baseline_no_context | persona_style | 5 | 19 | 119 | 0.4510 | 0.2083 |
| controlled_vs_baseline_no_context | distinct1 | 30 | 106 | 7 | 0.2343 | 0.2206 |
| controlled_vs_baseline_no_context | length_score | 53 | 84 | 6 | 0.3916 | 0.3869 |
| controlled_vs_baseline_no_context | sentence_score | 53 | 7 | 83 | 0.6608 | 0.8833 |
| controlled_vs_baseline_no_context | overall_quality | 118 | 25 | 0 | 0.8252 | 0.8252 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 91 | 50 | 2 | 0.6434 | 0.6454 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 110 | 12 | 21 | 0.8427 | 0.9016 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | 51 | 92 | 0 | 0.3566 | 0.3566 |
| controlled_vs_baseline_no_context_phi3_latest | quest_state_correctness | 91 | 50 | 2 | 0.6434 | 0.6454 |
| controlled_vs_baseline_no_context_phi3_latest | lore_consistency | 51 | 40 | 52 | 0.5385 | 0.5604 |
| controlled_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0 | 0 | 143 | 0.5000 | nan |
| controlled_vs_baseline_no_context_phi3_latest | objective_completion_support | 88 | 30 | 25 | 0.7028 | 0.7458 |
| controlled_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 108 | 35 | 0 | 0.7552 | 0.7552 |
| controlled_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 77 | 55 | 11 | 0.5769 | 0.5833 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 67 | 19 | 57 | 0.6678 | 0.7791 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 89 | 52 | 2 | 0.6294 | 0.6312 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 109 | 9 | 25 | 0.8497 | 0.9237 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | 6 | 15 | 122 | 0.4685 | 0.2857 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | 29 | 110 | 4 | 0.2168 | 0.2086 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | 54 | 84 | 5 | 0.3951 | 0.3913 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 45 | 5 | 93 | 0.6399 | 0.9000 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 119 | 24 | 0 | 0.8322 | 0.8322 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 86 | 55 | 2 | 0.6084 | 0.6099 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 106 | 13 | 24 | 0.8252 | 0.8908 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 54 | 89 | 0 | 0.3776 | 0.3776 |
| proposed_contextual_controlled_vs_baseline_no_context | quest_state_correctness | 88 | 53 | 2 | 0.6224 | 0.6241 |
| proposed_contextual_controlled_vs_baseline_no_context | lore_consistency | 52 | 39 | 52 | 0.5455 | 0.5714 |
| proposed_contextual_controlled_vs_baseline_no_context | multi_turn_contradiction_safety | 0 | 0 | 143 | 0.5000 | nan |
| proposed_contextual_controlled_vs_baseline_no_context | objective_completion_support | 80 | 36 | 27 | 0.6538 | 0.6897 |
| proposed_contextual_controlled_vs_baseline_no_context | gameplay_usefulness | 98 | 45 | 0 | 0.6853 | 0.6853 |
| proposed_contextual_controlled_vs_baseline_no_context | time_pressure_acceptability | 67 | 66 | 10 | 0.5035 | 0.5038 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 67 | 21 | 55 | 0.6608 | 0.7614 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 93 | 48 | 2 | 0.6573 | 0.6596 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 105 | 6 | 32 | 0.8462 | 0.9459 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 5 | 19 | 119 | 0.4510 | 0.2083 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | 30 | 106 | 7 | 0.2343 | 0.2206 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 53 | 84 | 6 | 0.3916 | 0.3869 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 53 | 7 | 83 | 0.6608 | 0.8833 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 118 | 25 | 0 | 0.8252 | 0.8252 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 91 | 50 | 2 | 0.6434 | 0.6454 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 110 | 12 | 21 | 0.8427 | 0.9016 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | 51 | 92 | 0 | 0.3566 | 0.3566 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | quest_state_correctness | 91 | 50 | 2 | 0.6434 | 0.6454 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | lore_consistency | 51 | 40 | 52 | 0.5385 | 0.5604 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0 | 0 | 143 | 0.5000 | nan |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | objective_completion_support | 88 | 30 | 25 | 0.7028 | 0.7458 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 108 | 35 | 0 | 0.7552 | 0.7552 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 77 | 55 | 11 | 0.5769 | 0.5833 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 67 | 19 | 57 | 0.6678 | 0.7791 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 89 | 52 | 2 | 0.6294 | 0.6312 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 109 | 9 | 25 | 0.8497 | 0.9237 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | 6 | 15 | 122 | 0.4685 | 0.2857 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | 29 | 110 | 4 | 0.2168 | 0.2086 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | 54 | 84 | 5 | 0.3951 | 0.3913 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | 45 | 5 | 93 | 0.6399 | 0.9000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 119 | 24 | 0 | 0.8322 | 0.8322 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0069 | 0.2222 | 0.2431 | 0.7569 |
| proposed_contextual | 0.0000 | 0.0000 | 0.4583 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.4444 | 0.0000 | 0.0000 |
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
# Proposal Alignment Evaluation Report

- Run ID: `20260313T052529Z`
- Generated: `2026-03-13T05:30:43.245652+00:00`
- Scenarios: `artifacts\proposal_control_tuning\auto_tune\20260313T051946Z\train_runs\trial_000\seed_23\20260313T052529Z\scenarios.jsonl`
- Scenario count: `24`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_tuned`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0997 (0.0607, 0.1402) | 0.3003 (0.2401, 0.3730) | 0.8843 (0.8648, 0.9028) | 0.3277 (0.2899, 0.3682) | n/a |
| proposed_contextual_controlled_tuned | 0.1350 (0.0814, 0.1943) | 0.2885 (0.2317, 0.3527) | 0.8794 (0.8605, 0.8973) | 0.3385 (0.3029, 0.3753) | n/a |
| proposed_contextual | 0.0905 (0.0567, 0.1290) | 0.2285 (0.1752, 0.2899) | 0.8663 (0.8510, 0.8817) | 0.2921 (0.2605, 0.3259) | n/a |
| candidate_no_context | 0.0353 (0.0220, 0.0492) | 0.2343 (0.1733, 0.3048) | 0.8802 (0.8631, 0.8972) | 0.2714 (0.2457, 0.2993) | n/a |

## Game-facing Outcome Metrics (mean, 95% CI)
| Arm | Quest-state Correctness | Lore Consistency | Contradiction Safety | Objective Completion Support | Gameplay Usefulness | Time-pressure Acceptability |
|---|---:|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.1827 (0.1483, 0.2198) | 0.0241 (0.0080, 0.0452) | 1.0000 (1.0000, 1.0000) | 0.0809 (0.0574, 0.1044) | 0.3090 (0.2935, 0.3250) | 0.2938 (0.2817, 0.3057) |
| proposed_contextual_controlled_tuned | 0.2158 (0.1707, 0.2671) | 0.0630 (0.0263, 0.1092) | 1.0000 (1.0000, 1.0000) | 0.0601 (0.0380, 0.0837) | 0.3067 (0.2934, 0.3192) | 0.2799 (0.2589, 0.2969) |
| proposed_contextual | 0.1759 (0.1474, 0.2085) | 0.0376 (0.0165, 0.0621) | 1.0000 (1.0000, 1.0000) | 0.0590 (0.0356, 0.0854) | 0.2936 (0.2792, 0.3075) | 0.2865 (0.2719, 0.3013) |
| candidate_no_context | 0.1271 (0.1174, 0.1382) | 0.0074 (0.0021, 0.0142) | 1.0000 (1.0000, 1.0000) | 0.0643 (0.0410, 0.0867) | 0.2883 (0.2760, 0.2998) | 0.2877 (0.2773, 0.2981) |

- Multi-turn contradiction rate is reported as `1 - contradiction_safety` in row-level outputs.

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0552 | 1.5631 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0058 | -0.0246 |
| proposed_vs_candidate_no_context | naturalness | -0.0139 | -0.0158 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.0488 | 0.3842 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0302 | 4.0938 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | -0.0053 | -0.0832 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 0.0053 | 0.0185 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | -0.0012 | -0.0040 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0682 | 2.0000 |
| proposed_vs_candidate_no_context | context_overlap | 0.0249 | 0.6526 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0020 | -0.0123 |
| proposed_vs_candidate_no_context | persona_style | -0.0208 | -0.0394 |
| proposed_vs_candidate_no_context | distinct1 | 0.0048 | 0.0051 |
| proposed_vs_candidate_no_context | length_score | -0.0792 | -0.1360 |
| proposed_vs_candidate_no_context | sentence_score | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | overall_quality | 0.0207 | 0.0763 |
| controlled_vs_proposed_raw | context_relevance | 0.0092 | 0.1019 |
| controlled_vs_proposed_raw | persona_consistency | 0.0718 | 0.3141 |
| controlled_vs_proposed_raw | naturalness | 0.0180 | 0.0208 |
| controlled_vs_proposed_raw | quest_state_correctness | 0.0068 | 0.0387 |
| controlled_vs_proposed_raw | lore_consistency | -0.0135 | -0.3593 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0219 | 0.3722 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0154 | 0.0525 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 0.0072 | 0.0252 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.0163 | 0.1593 |
| controlled_vs_proposed_raw | context_overlap | -0.0073 | -0.1153 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.0962 | 0.6062 |
| controlled_vs_proposed_raw | persona_style | -0.0260 | -0.0513 |
| controlled_vs_proposed_raw | distinct1 | -0.0037 | -0.0040 |
| controlled_vs_proposed_raw | length_score | 0.0611 | 0.1215 |
| controlled_vs_proposed_raw | sentence_score | 0.0729 | 0.0812 |
| controlled_vs_proposed_raw | overall_quality | 0.0355 | 0.1216 |
| controlled_vs_candidate_no_context | context_relevance | 0.0644 | 1.8243 |
| controlled_vs_candidate_no_context | persona_consistency | 0.0660 | 0.2818 |
| controlled_vs_candidate_no_context | naturalness | 0.0041 | 0.0047 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.0556 | 0.4378 |
| controlled_vs_candidate_no_context | lore_consistency | 0.0167 | 2.2637 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | 0.0166 | 0.2581 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0207 | 0.0719 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 0.0061 | 0.0211 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.0845 | 2.4778 |
| controlled_vs_candidate_no_context | context_overlap | 0.0176 | 0.4621 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.0942 | 0.5864 |
| controlled_vs_candidate_no_context | persona_style | -0.0469 | -0.0887 |
| controlled_vs_candidate_no_context | distinct1 | 0.0011 | 0.0012 |
| controlled_vs_candidate_no_context | length_score | -0.0181 | -0.0310 |
| controlled_vs_candidate_no_context | sentence_score | 0.0729 | 0.0812 |
| controlled_vs_candidate_no_context | overall_quality | 0.0562 | 0.2072 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0353 | 0.3538 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0118 | -0.0393 |
| controlled_alt_vs_controlled_default | naturalness | -0.0049 | -0.0056 |
| controlled_alt_vs_controlled_default | quest_state_correctness | 0.0330 | 0.1809 |
| controlled_alt_vs_controlled_default | lore_consistency | 0.0389 | 1.6179 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_controlled_default | objective_completion_support | -0.0208 | -0.2575 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | -0.0023 | -0.0073 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | -0.0139 | -0.0472 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0447 | 0.3770 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0133 | 0.2387 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0151 | -0.0591 |
| controlled_alt_vs_controlled_default | persona_style | 0.0013 | 0.0027 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0028 | 0.0030 |
| controlled_alt_vs_controlled_default | length_score | -0.0375 | -0.0665 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0146 | 0.0150 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0108 | 0.0331 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.0445 | 0.4917 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.0600 | 0.2624 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0131 | 0.0151 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 0.0399 | 0.2266 |
| controlled_alt_vs_proposed_raw | lore_consistency | 0.0254 | 0.6773 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_proposed_raw | objective_completion_support | 0.0011 | 0.0188 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 0.0132 | 0.0448 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | -0.0066 | -0.0231 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.0610 | 0.5963 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0060 | 0.0959 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.0812 | 0.5113 |
| controlled_alt_vs_proposed_raw | persona_style | -0.0247 | -0.0487 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0009 | -0.0010 |
| controlled_alt_vs_proposed_raw | length_score | 0.0236 | 0.0470 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0875 | 0.0974 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.0464 | 0.1588 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.0997 | 2.8234 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.0542 | 0.2314 |
| controlled_alt_vs_candidate_no_context | naturalness | -0.0008 | -0.0009 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 0.0887 | 0.6979 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 0.0556 | 7.5441 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_candidate_no_context | objective_completion_support | -0.0042 | -0.0659 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 0.0185 | 0.0641 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | -0.0078 | -0.0271 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.1292 | 3.7889 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0310 | 0.8111 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.0792 | 0.4926 |
| controlled_alt_vs_candidate_no_context | persona_style | -0.0456 | -0.0862 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0039 | 0.0041 |
| controlled_alt_vs_candidate_no_context | length_score | -0.0556 | -0.0955 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.0875 | 0.0974 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.0671 | 0.2471 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0552 | (0.0249, 0.0895) | 0.0000 | 0.0552 | (0.0079, 0.0890) | 0.0083 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0058 | (-0.0571, 0.0450) | 0.5617 | -0.0058 | (-0.0855, 0.0462) | 0.5727 |
| proposed_vs_candidate_no_context | naturalness | -0.0139 | (-0.0333, 0.0041) | 0.9290 | -0.0139 | (-0.0321, -0.0004) | 1.0000 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.0488 | (0.0215, 0.0792) | 0.0000 | 0.0488 | (0.0085, 0.0776) | 0.0113 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0302 | (0.0104, 0.0548) | 0.0000 | 0.0302 | (0.0000, 0.0589) | 0.0783 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | -0.0053 | (-0.0240, 0.0157) | 0.7177 | -0.0053 | (-0.0239, 0.0222) | 0.6317 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 0.0053 | (-0.0083, 0.0189) | 0.2140 | 0.0053 | (0.0007, 0.0114) | 0.0077 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | -0.0012 | (-0.0148, 0.0143) | 0.5910 | -0.0012 | (-0.0140, 0.0150) | 0.6523 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0682 | (0.0303, 0.1136) | 0.0000 | 0.0682 | (0.0091, 0.1104) | 0.0103 |
| proposed_vs_candidate_no_context | context_overlap | 0.0249 | (0.0090, 0.0422) | 0.0003 | 0.0249 | (0.0052, 0.0390) | 0.0097 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0020 | (-0.0734, 0.0575) | 0.5167 | -0.0020 | (-0.0898, 0.0577) | 0.5133 |
| proposed_vs_candidate_no_context | persona_style | -0.0208 | (-0.0729, 0.0208) | 0.8593 | -0.0208 | (-0.0682, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | distinct1 | 0.0048 | (-0.0097, 0.0213) | 0.2610 | 0.0048 | (-0.0114, 0.0221) | 0.3387 |
| proposed_vs_candidate_no_context | length_score | -0.0792 | (-0.1722, 0.0097) | 0.9547 | -0.0792 | (-0.1455, -0.0136) | 1.0000 |
| proposed_vs_candidate_no_context | sentence_score | 0.0000 | (-0.0583, 0.0583) | 0.6113 | 0.0000 | (-0.0404, 0.0404) | 0.6417 |
| proposed_vs_candidate_no_context | overall_quality | 0.0207 | (-0.0072, 0.0474) | 0.0713 | 0.0207 | (-0.0211, 0.0551) | 0.1783 |
| controlled_vs_proposed_raw | context_relevance | 0.0092 | (-0.0382, 0.0582) | 0.3487 | 0.0092 | (-0.0249, 0.0604) | 0.3527 |
| controlled_vs_proposed_raw | persona_consistency | 0.0718 | (-0.0041, 0.1551) | 0.0327 | 0.0718 | (-0.0480, 0.1542) | 0.0990 |
| controlled_vs_proposed_raw | naturalness | 0.0180 | (-0.0060, 0.0420) | 0.0640 | 0.0180 | (0.0010, 0.0336) | 0.0207 |
| controlled_vs_proposed_raw | quest_state_correctness | 0.0068 | (-0.0339, 0.0484) | 0.3583 | 0.0068 | (-0.0201, 0.0423) | 0.3170 |
| controlled_vs_proposed_raw | lore_consistency | -0.0135 | (-0.0419, 0.0153) | 0.8237 | -0.0135 | (-0.0368, 0.0052) | 0.8023 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0219 | (-0.0113, 0.0547) | 0.0940 | 0.0219 | (-0.0448, 0.0727) | 0.2643 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0154 | (-0.0054, 0.0392) | 0.0870 | 0.0154 | (-0.0133, 0.0373) | 0.1320 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 0.0072 | (-0.0115, 0.0257) | 0.2160 | 0.0072 | (-0.0219, 0.0302) | 0.2990 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.0163 | (-0.0432, 0.0754) | 0.2997 | 0.0163 | (-0.0254, 0.0839) | 0.3413 |
| controlled_vs_proposed_raw | context_overlap | -0.0073 | (-0.0296, 0.0143) | 0.7497 | -0.0073 | (-0.0237, 0.0048) | 0.8783 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.0962 | (0.0050, 0.2004) | 0.0177 | 0.0962 | (-0.0429, 0.2013) | 0.0733 |
| controlled_vs_proposed_raw | persona_style | -0.0260 | (-0.0534, -0.0078) | 1.0000 | -0.0260 | (-0.0625, 0.0000) | 1.0000 |
| controlled_vs_proposed_raw | distinct1 | -0.0037 | (-0.0247, 0.0156) | 0.6307 | -0.0037 | (-0.0282, 0.0204) | 0.5950 |
| controlled_vs_proposed_raw | length_score | 0.0611 | (-0.0375, 0.1667) | 0.1230 | 0.0611 | (-0.0258, 0.1381) | 0.0847 |
| controlled_vs_proposed_raw | sentence_score | 0.0729 | (-0.0146, 0.1462) | 0.0593 | 0.0729 | (-0.0350, 0.2000) | 0.2110 |
| controlled_vs_proposed_raw | overall_quality | 0.0355 | (-0.0099, 0.0856) | 0.0753 | 0.0355 | (-0.0261, 0.0915) | 0.1273 |
| controlled_vs_candidate_no_context | context_relevance | 0.0644 | (0.0235, 0.1058) | 0.0000 | 0.0644 | (0.0026, 0.1129) | 0.0247 |
| controlled_vs_candidate_no_context | persona_consistency | 0.0660 | (-0.0065, 0.1456) | 0.0390 | 0.0660 | (-0.0608, 0.1497) | 0.1387 |
| controlled_vs_candidate_no_context | naturalness | 0.0041 | (-0.0151, 0.0266) | 0.3670 | 0.0041 | (-0.0118, 0.0263) | 0.3377 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.0556 | (0.0216, 0.0923) | 0.0007 | 0.0556 | (0.0001, 0.0969) | 0.0203 |
| controlled_vs_candidate_no_context | lore_consistency | 0.0167 | (0.0039, 0.0335) | 0.0007 | 0.0167 | (0.0026, 0.0277) | 0.0127 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | 0.0166 | (-0.0098, 0.0454) | 0.1150 | 0.0166 | (-0.0242, 0.0457) | 0.1947 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0207 | (0.0026, 0.0423) | 0.0110 | 0.0207 | (-0.0073, 0.0437) | 0.0810 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 0.0061 | (-0.0060, 0.0182) | 0.1633 | 0.0061 | (-0.0095, 0.0172) | 0.1867 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.0845 | (0.0341, 0.1364) | 0.0003 | 0.0845 | (0.0095, 0.1455) | 0.0153 |
| controlled_vs_candidate_no_context | context_overlap | 0.0176 | (0.0009, 0.0343) | 0.0200 | 0.0176 | (-0.0128, 0.0376) | 0.0987 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.0942 | (0.0079, 0.1915) | 0.0143 | 0.0942 | (-0.0452, 0.1939) | 0.0950 |
| controlled_vs_candidate_no_context | persona_style | -0.0469 | (-0.0990, -0.0117) | 1.0000 | -0.0469 | (-0.1193, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | distinct1 | 0.0011 | (-0.0117, 0.0130) | 0.4247 | 0.0011 | (-0.0086, 0.0162) | 0.4587 |
| controlled_vs_candidate_no_context | length_score | -0.0181 | (-0.1056, 0.0903) | 0.6677 | -0.0181 | (-0.0875, 0.0654) | 0.6973 |
| controlled_vs_candidate_no_context | sentence_score | 0.0729 | (0.0000, 0.1458) | 0.0290 | 0.0729 | (-0.0350, 0.1625) | 0.1273 |
| controlled_vs_candidate_no_context | overall_quality | 0.0562 | (0.0127, 0.1018) | 0.0037 | 0.0562 | (-0.0128, 0.1055) | 0.0467 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0353 | (-0.0264, 0.1075) | 0.1640 | 0.0353 | (-0.0458, 0.1172) | 0.2050 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0118 | (-0.0911, 0.0701) | 0.6123 | -0.0118 | (-0.0748, 0.0512) | 0.6540 |
| controlled_alt_vs_controlled_default | naturalness | -0.0049 | (-0.0253, 0.0135) | 0.6827 | -0.0049 | (-0.0257, 0.0124) | 0.6967 |
| controlled_alt_vs_controlled_default | quest_state_correctness | 0.0330 | (-0.0228, 0.0895) | 0.1290 | 0.0330 | (-0.0317, 0.0996) | 0.1553 |
| controlled_alt_vs_controlled_default | lore_consistency | 0.0389 | (-0.0028, 0.0908) | 0.0377 | 0.0389 | (0.0064, 0.0774) | 0.0073 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | objective_completion_support | -0.0208 | (-0.0443, 0.0038) | 0.9553 | -0.0208 | (-0.0460, 0.0144) | 0.8567 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | -0.0023 | (-0.0169, 0.0118) | 0.6307 | -0.0023 | (-0.0163, 0.0129) | 0.6170 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | -0.0139 | (-0.0345, 0.0028) | 0.9403 | -0.0139 | (-0.0293, 0.0039) | 0.9533 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0447 | (-0.0356, 0.1364) | 0.1663 | 0.0447 | (-0.0628, 0.1488) | 0.1973 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0133 | (-0.0120, 0.0425) | 0.1887 | 0.0133 | (-0.0122, 0.0435) | 0.1730 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0151 | (-0.1121, 0.0821) | 0.6153 | -0.0151 | (-0.0913, 0.0611) | 0.6540 |
| controlled_alt_vs_controlled_default | persona_style | 0.0013 | (-0.0286, 0.0274) | 0.4773 | 0.0013 | (-0.0312, 0.0352) | 0.4277 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0028 | (-0.0134, 0.0195) | 0.3750 | 0.0028 | (-0.0166, 0.0214) | 0.3947 |
| controlled_alt_vs_controlled_default | length_score | -0.0375 | (-0.1319, 0.0500) | 0.7990 | -0.0375 | (-0.1367, 0.0333) | 0.8080 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0146 | (-0.0292, 0.0729) | 0.3747 | 0.0146 | (0.0000, 0.0404) | 0.3370 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0108 | (-0.0372, 0.0603) | 0.3290 | 0.0108 | (-0.0504, 0.0686) | 0.3227 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.0445 | (-0.0139, 0.1060) | 0.0673 | 0.0445 | (-0.0027, 0.1002) | 0.0387 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.0600 | (-0.0058, 0.1359) | 0.0427 | 0.0600 | (-0.0432, 0.1415) | 0.1147 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0131 | (-0.0100, 0.0343) | 0.1227 | 0.0131 | (-0.0224, 0.0445) | 0.2590 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 0.0399 | (-0.0142, 0.0978) | 0.0797 | 0.0399 | (0.0011, 0.0865) | 0.0163 |
| controlled_alt_vs_proposed_raw | lore_consistency | 0.0254 | (-0.0194, 0.0760) | 0.1470 | 0.0254 | (-0.0167, 0.0784) | 0.1513 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_proposed_raw | objective_completion_support | 0.0011 | (-0.0231, 0.0240) | 0.4543 | 0.0011 | (-0.0386, 0.0295) | 0.4563 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 0.0132 | (-0.0045, 0.0328) | 0.0830 | 0.0132 | (-0.0133, 0.0357) | 0.1677 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | -0.0066 | (-0.0301, 0.0148) | 0.6953 | -0.0066 | (-0.0353, 0.0151) | 0.6950 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.0610 | (-0.0148, 0.1455) | 0.0537 | 0.0610 | (0.0007, 0.1322) | 0.0117 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0060 | (-0.0184, 0.0363) | 0.3397 | 0.0060 | (-0.0105, 0.0307) | 0.3320 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.0812 | (-0.0010, 0.1707) | 0.0290 | 0.0812 | (-0.0509, 0.1873) | 0.1027 |
| controlled_alt_vs_proposed_raw | persona_style | -0.0247 | (-0.0586, 0.0039) | 0.9577 | -0.0247 | (-0.0724, 0.0000) | 1.0000 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0009 | (-0.0168, 0.0153) | 0.5463 | -0.0009 | (-0.0175, 0.0223) | 0.5490 |
| controlled_alt_vs_proposed_raw | length_score | 0.0236 | (-0.0764, 0.1250) | 0.3197 | 0.0236 | (-0.1167, 0.1381) | 0.3923 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0875 | (0.0146, 0.1604) | 0.0133 | 0.0875 | (-0.0350, 0.2125) | 0.1270 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.0464 | (0.0043, 0.0883) | 0.0130 | 0.0464 | (0.0081, 0.0847) | 0.0017 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.0997 | (0.0530, 0.1560) | 0.0000 | 0.0997 | (0.0603, 0.1272) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.0542 | (-0.0271, 0.1385) | 0.0930 | 0.0542 | (-0.0601, 0.1405) | 0.1600 |
| controlled_alt_vs_candidate_no_context | naturalness | -0.0008 | (-0.0244, 0.0231) | 0.5367 | -0.0008 | (-0.0363, 0.0363) | 0.5593 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 0.0887 | (0.0464, 0.1391) | 0.0000 | 0.0887 | (0.0605, 0.1128) | 0.0000 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 0.0556 | (0.0183, 0.1034) | 0.0000 | 0.0556 | (0.0263, 0.0872) | 0.0003 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_candidate_no_context | objective_completion_support | -0.0042 | (-0.0313, 0.0212) | 0.6130 | -0.0042 | (-0.0201, 0.0094) | 0.7217 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 0.0185 | (0.0029, 0.0338) | 0.0087 | 0.0185 | (-0.0058, 0.0386) | 0.0763 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | -0.0078 | (-0.0303, 0.0118) | 0.7480 | -0.0078 | (-0.0282, 0.0068) | 0.8280 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.1292 | (0.0686, 0.2008) | 0.0000 | 0.1292 | (0.0748, 0.1629) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0310 | (0.0093, 0.0560) | 0.0010 | 0.0310 | (0.0168, 0.0470) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.0792 | (-0.0159, 0.1827) | 0.0507 | 0.0792 | (-0.0600, 0.1885) | 0.1023 |
| controlled_alt_vs_candidate_no_context | persona_style | -0.0456 | (-0.1094, 0.0013) | 0.9717 | -0.0456 | (-0.1406, 0.0000) | 1.0000 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0039 | (-0.0127, 0.0202) | 0.3323 | 0.0039 | (-0.0149, 0.0179) | 0.2827 |
| controlled_alt_vs_candidate_no_context | length_score | -0.0556 | (-0.1653, 0.0556) | 0.8407 | -0.0556 | (-0.1933, 0.0750) | 0.7900 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.0875 | (0.0146, 0.1604) | 0.0143 | 0.0875 | (-0.0350, 0.1875) | 0.0940 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.0671 | (0.0212, 0.1150) | 0.0010 | 0.0671 | (0.0096, 0.1166) | 0.0103 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 12 | 2 | 10 | 0.7083 | 0.8571 |
| proposed_vs_candidate_no_context | persona_consistency | 5 | 3 | 16 | 0.5417 | 0.6250 |
| proposed_vs_candidate_no_context | naturalness | 6 | 7 | 11 | 0.4792 | 0.4615 |
| proposed_vs_candidate_no_context | quest_state_correctness | 12 | 2 | 10 | 0.7083 | 0.8571 |
| proposed_vs_candidate_no_context | lore_consistency | 10 | 1 | 13 | 0.6875 | 0.9091 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| proposed_vs_candidate_no_context | objective_completion_support | 4 | 6 | 14 | 0.4583 | 0.4000 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 9 | 5 | 10 | 0.5833 | 0.6429 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | 5 | 7 | 12 | 0.4583 | 0.4167 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 9 | 0 | 15 | 0.6875 | 1.0000 |
| proposed_vs_candidate_no_context | context_overlap | 12 | 2 | 10 | 0.7083 | 0.8571 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 5 | 3 | 16 | 0.5417 | 0.6250 |
| proposed_vs_candidate_no_context | persona_style | 1 | 2 | 21 | 0.4792 | 0.3333 |
| proposed_vs_candidate_no_context | distinct1 | 6 | 4 | 14 | 0.5417 | 0.6000 |
| proposed_vs_candidate_no_context | length_score | 5 | 6 | 13 | 0.4792 | 0.4545 |
| proposed_vs_candidate_no_context | sentence_score | 2 | 2 | 20 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | overall_quality | 11 | 3 | 10 | 0.6667 | 0.7857 |
| controlled_vs_proposed_raw | context_relevance | 8 | 12 | 4 | 0.4167 | 0.4000 |
| controlled_vs_proposed_raw | persona_consistency | 9 | 6 | 9 | 0.5625 | 0.6000 |
| controlled_vs_proposed_raw | naturalness | 12 | 8 | 4 | 0.5833 | 0.6000 |
| controlled_vs_proposed_raw | quest_state_correctness | 8 | 12 | 4 | 0.4167 | 0.4000 |
| controlled_vs_proposed_raw | lore_consistency | 6 | 8 | 10 | 0.4583 | 0.4286 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_vs_proposed_raw | objective_completion_support | 11 | 4 | 9 | 0.6458 | 0.7333 |
| controlled_vs_proposed_raw | gameplay_usefulness | 9 | 11 | 4 | 0.4583 | 0.4500 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 10 | 5 | 9 | 0.6042 | 0.6667 |
| controlled_vs_proposed_raw | context_keyword_coverage | 8 | 6 | 10 | 0.5417 | 0.5714 |
| controlled_vs_proposed_raw | context_overlap | 7 | 13 | 4 | 0.3750 | 0.3500 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 9 | 6 | 9 | 0.5625 | 0.6000 |
| controlled_vs_proposed_raw | persona_style | 0 | 5 | 19 | 0.3958 | 0.0000 |
| controlled_vs_proposed_raw | distinct1 | 9 | 10 | 5 | 0.4792 | 0.4737 |
| controlled_vs_proposed_raw | length_score | 10 | 10 | 4 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | sentence_score | 7 | 2 | 15 | 0.6042 | 0.7778 |
| controlled_vs_proposed_raw | overall_quality | 12 | 8 | 4 | 0.5833 | 0.6000 |
| controlled_vs_candidate_no_context | context_relevance | 11 | 7 | 6 | 0.5833 | 0.6111 |
| controlled_vs_candidate_no_context | persona_consistency | 10 | 6 | 8 | 0.5833 | 0.6250 |
| controlled_vs_candidate_no_context | naturalness | 9 | 9 | 6 | 0.5000 | 0.5000 |
| controlled_vs_candidate_no_context | quest_state_correctness | 11 | 7 | 6 | 0.5833 | 0.6111 |
| controlled_vs_candidate_no_context | lore_consistency | 9 | 1 | 14 | 0.6667 | 0.9000 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_vs_candidate_no_context | objective_completion_support | 9 | 3 | 12 | 0.6250 | 0.7500 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 11 | 7 | 6 | 0.5833 | 0.6111 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 9 | 3 | 12 | 0.6250 | 0.7500 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 11 | 3 | 10 | 0.6667 | 0.7857 |
| controlled_vs_candidate_no_context | context_overlap | 11 | 7 | 6 | 0.5833 | 0.6111 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 10 | 5 | 9 | 0.6042 | 0.6667 |
| controlled_vs_candidate_no_context | persona_style | 0 | 6 | 18 | 0.3750 | 0.0000 |
| controlled_vs_candidate_no_context | distinct1 | 10 | 6 | 8 | 0.5833 | 0.6250 |
| controlled_vs_candidate_no_context | length_score | 6 | 12 | 6 | 0.3750 | 0.3333 |
| controlled_vs_candidate_no_context | sentence_score | 6 | 1 | 17 | 0.6042 | 0.8571 |
| controlled_vs_candidate_no_context | overall_quality | 14 | 4 | 6 | 0.7083 | 0.7778 |
| controlled_alt_vs_controlled_default | context_relevance | 7 | 10 | 7 | 0.4375 | 0.4118 |
| controlled_alt_vs_controlled_default | persona_consistency | 5 | 4 | 15 | 0.5208 | 0.5556 |
| controlled_alt_vs_controlled_default | naturalness | 7 | 10 | 7 | 0.4375 | 0.4118 |
| controlled_alt_vs_controlled_default | quest_state_correctness | 9 | 8 | 7 | 0.5208 | 0.5294 |
| controlled_alt_vs_controlled_default | lore_consistency | 8 | 7 | 9 | 0.5208 | 0.5333 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_alt_vs_controlled_default | objective_completion_support | 4 | 10 | 10 | 0.3750 | 0.2857 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | 9 | 8 | 7 | 0.5208 | 0.5294 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | 6 | 10 | 8 | 0.4167 | 0.3750 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 7 | 8 | 9 | 0.4792 | 0.4667 |
| controlled_alt_vs_controlled_default | context_overlap | 8 | 9 | 7 | 0.4792 | 0.4706 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 3 | 4 | 17 | 0.4792 | 0.4286 |
| controlled_alt_vs_controlled_default | persona_style | 2 | 1 | 21 | 0.5208 | 0.6667 |
| controlled_alt_vs_controlled_default | distinct1 | 8 | 8 | 8 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | length_score | 9 | 8 | 7 | 0.5208 | 0.5294 |
| controlled_alt_vs_controlled_default | sentence_score | 2 | 1 | 21 | 0.5208 | 0.6667 |
| controlled_alt_vs_controlled_default | overall_quality | 9 | 8 | 7 | 0.5208 | 0.5294 |
| controlled_alt_vs_proposed_raw | context_relevance | 11 | 10 | 3 | 0.5208 | 0.5238 |
| controlled_alt_vs_proposed_raw | persona_consistency | 9 | 6 | 9 | 0.5625 | 0.6000 |
| controlled_alt_vs_proposed_raw | naturalness | 13 | 8 | 3 | 0.6042 | 0.6190 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 11 | 10 | 3 | 0.5208 | 0.5238 |
| controlled_alt_vs_proposed_raw | lore_consistency | 9 | 6 | 9 | 0.5625 | 0.6000 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_alt_vs_proposed_raw | objective_completion_support | 7 | 6 | 11 | 0.5208 | 0.5385 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 12 | 9 | 3 | 0.5625 | 0.5714 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | 8 | 6 | 10 | 0.5417 | 0.5714 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 10 | 5 | 9 | 0.6042 | 0.6667 |
| controlled_alt_vs_proposed_raw | context_overlap | 11 | 10 | 3 | 0.5208 | 0.5238 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 9 | 5 | 10 | 0.5833 | 0.6429 |
| controlled_alt_vs_proposed_raw | persona_style | 1 | 5 | 18 | 0.4167 | 0.1667 |
| controlled_alt_vs_proposed_raw | distinct1 | 10 | 10 | 4 | 0.5000 | 0.5000 |
| controlled_alt_vs_proposed_raw | length_score | 11 | 10 | 3 | 0.5208 | 0.5238 |
| controlled_alt_vs_proposed_raw | sentence_score | 7 | 1 | 16 | 0.6250 | 0.8750 |
| controlled_alt_vs_proposed_raw | overall_quality | 15 | 6 | 3 | 0.6875 | 0.7143 |
| controlled_alt_vs_candidate_no_context | context_relevance | 16 | 5 | 3 | 0.7292 | 0.7619 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 11 | 6 | 7 | 0.6042 | 0.6471 |
| controlled_alt_vs_candidate_no_context | naturalness | 9 | 12 | 3 | 0.4375 | 0.4286 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 17 | 4 | 3 | 0.7708 | 0.8095 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 11 | 2 | 11 | 0.6875 | 0.8462 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_alt_vs_candidate_no_context | objective_completion_support | 8 | 7 | 9 | 0.5208 | 0.5333 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 16 | 5 | 3 | 0.7292 | 0.7619 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | 10 | 6 | 8 | 0.5833 | 0.6250 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 14 | 1 | 9 | 0.7708 | 0.9333 |
| controlled_alt_vs_candidate_no_context | context_overlap | 15 | 6 | 3 | 0.6875 | 0.7143 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 11 | 6 | 7 | 0.6042 | 0.6471 |
| controlled_alt_vs_candidate_no_context | persona_style | 1 | 5 | 18 | 0.4167 | 0.1667 |
| controlled_alt_vs_candidate_no_context | distinct1 | 13 | 7 | 4 | 0.6250 | 0.6500 |
| controlled_alt_vs_candidate_no_context | length_score | 9 | 12 | 3 | 0.4375 | 0.4286 |
| controlled_alt_vs_candidate_no_context | sentence_score | 7 | 1 | 16 | 0.6250 | 0.8750 |
| controlled_alt_vs_candidate_no_context | overall_quality | 17 | 4 | 3 | 0.7708 | 0.8095 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.0417 | 0.1667 | 0.8333 |
| proposed_contextual_controlled_tuned | 0.0000 | 0.0000 | 0.0833 | 0.2917 | 0.7083 |
| proposed_contextual | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.5417 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `5`
- Unique template signatures: `19`
- Template signature ratio: `0.7917`
- Effective sample size by source clustering: `4.80`
- Effective sample size by template-signature clustering: `16.94`
- Detailed diagnostics are published in `scenario_dependence.json`.

## Multi-turn Contradiction
| Arm | Contradiction Rate | Contradiction Safety | Contradicted Sources | Source Count |
|---|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 1.0000 | 0 | 5 |
| proposed_contextual_controlled_tuned | 0.0000 | 1.0000 | 0 | 5 |
| proposed_contextual | 0.0000 | 1.0000 | 0 | 5 |
| candidate_no_context | 0.0000 | 1.0000 | 0 | 5 |
- Detailed source-level values are published in `multi_turn_contradictions.json`.

- BERTScore status: unavailable (disabled_by_flag).

This report covers proposal RO5 metrics plus game-facing outcomes: quest-state correctness, lore consistency, contradiction safety, objective completion support, gameplay usefulness, and time-pressure acceptability.
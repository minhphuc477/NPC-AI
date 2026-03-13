# Proposal Alignment Evaluation Report

- Run ID: `20260313T051947Z`
- Generated: `2026-03-13T05:25:28.616298+00:00`
- Scenarios: `artifacts\proposal_control_tuning\auto_tune\20260313T051946Z\train_runs\trial_000\seed_19\20260313T051947Z\scenarios.jsonl`
- Scenario count: `24`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_tuned`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0976 (0.0613, 0.1366) | 0.2548 (0.2036, 0.3096) | 0.8690 (0.8411, 0.8937) | 0.3061 (0.2751, 0.3366) | n/a |
| proposed_contextual_controlled_tuned | 0.1274 (0.0642, 0.1995) | 0.2838 (0.2108, 0.3698) | 0.8713 (0.8494, 0.8922) | 0.3313 (0.2843, 0.3834) | n/a |
| proposed_contextual | 0.0905 (0.0485, 0.1335) | 0.2396 (0.1768, 0.3070) | 0.8710 (0.8570, 0.8861) | 0.2973 (0.2654, 0.3323) | n/a |
| candidate_no_context | 0.0354 (0.0207, 0.0526) | 0.2470 (0.1929, 0.3103) | 0.8897 (0.8710, 0.9083) | 0.2780 (0.2515, 0.3060) | n/a |

## Game-facing Outcome Metrics (mean, 95% CI)
| Arm | Quest-state Correctness | Lore Consistency | Contradiction Safety | Objective Completion Support | Gameplay Usefulness | Time-pressure Acceptability |
|---|---:|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.1844 (0.1496, 0.2182) | 0.0326 (0.0130, 0.0552) | 1.0000 (1.0000, 1.0000) | 0.0735 (0.0473, 0.0994) | 0.3002 (0.2864, 0.3141) | 0.2866 (0.2672, 0.3040) |
| proposed_contextual_controlled_tuned | 0.2072 (0.1546, 0.2694) | 0.0693 (0.0247, 0.1246) | 1.0000 (1.0000, 1.0000) | 0.0715 (0.0465, 0.0948) | 0.3071 (0.2931, 0.3206) | 0.2865 (0.2705, 0.3018) |
| proposed_contextual | 0.1734 (0.1405, 0.2127) | 0.0373 (0.0161, 0.0624) | 1.0000 (1.0000, 1.0000) | 0.0538 (0.0308, 0.0770) | 0.2915 (0.2768, 0.3057) | 0.2853 (0.2693, 0.3019) |
| candidate_no_context | 0.1263 (0.1145, 0.1397) | 0.0056 (0.0000, 0.0143) | 1.0000 (1.0000, 1.0000) | 0.0759 (0.0515, 0.1006) | 0.2920 (0.2844, 0.2997) | 0.2990 (0.2862, 0.3118) |

- Multi-turn contradiction rate is reported as `1 - contradiction_safety` in row-level outputs.

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0551 | 1.5561 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0074 | -0.0299 |
| proposed_vs_candidate_no_context | naturalness | -0.0187 | -0.0210 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.0471 | 0.3729 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0317 | 5.6546 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | -0.0221 | -0.2909 |
| proposed_vs_candidate_no_context | gameplay_usefulness | -0.0005 | -0.0017 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | -0.0136 | -0.0457 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0731 | 2.0978 |
| proposed_vs_candidate_no_context | context_overlap | 0.0131 | 0.3571 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0079 | -0.0455 |
| proposed_vs_candidate_no_context | persona_style | -0.0052 | -0.0097 |
| proposed_vs_candidate_no_context | distinct1 | -0.0015 | -0.0016 |
| proposed_vs_candidate_no_context | length_score | -0.0611 | -0.1063 |
| proposed_vs_candidate_no_context | sentence_score | -0.0583 | -0.0583 |
| proposed_vs_candidate_no_context | overall_quality | 0.0193 | 0.0693 |
| controlled_vs_proposed_raw | context_relevance | 0.0071 | 0.0783 |
| controlled_vs_proposed_raw | persona_consistency | 0.0152 | 0.0635 |
| controlled_vs_proposed_raw | naturalness | -0.0020 | -0.0023 |
| controlled_vs_proposed_raw | quest_state_correctness | 0.0110 | 0.0632 |
| controlled_vs_proposed_raw | lore_consistency | -0.0046 | -0.1237 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0197 | 0.3652 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0087 | 0.0298 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 0.0012 | 0.0044 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.0061 | 0.0561 |
| controlled_vs_proposed_raw | context_overlap | 0.0095 | 0.1904 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.0262 | 0.1571 |
| controlled_vs_proposed_raw | persona_style | -0.0286 | -0.0539 |
| controlled_vs_proposed_raw | distinct1 | -0.0163 | -0.0174 |
| controlled_vs_proposed_raw | length_score | 0.0000 | 0.0000 |
| controlled_vs_proposed_raw | sentence_score | 0.0583 | 0.0619 |
| controlled_vs_proposed_raw | overall_quality | 0.0088 | 0.0295 |
| controlled_vs_candidate_no_context | context_relevance | 0.0622 | 1.7563 |
| controlled_vs_candidate_no_context | persona_consistency | 0.0078 | 0.0317 |
| controlled_vs_candidate_no_context | naturalness | -0.0207 | -0.0232 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.0581 | 0.4596 |
| controlled_vs_candidate_no_context | lore_consistency | 0.0270 | 4.8312 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | -0.0024 | -0.0320 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0082 | 0.0281 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | -0.0124 | -0.0415 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.0792 | 2.2717 |
| controlled_vs_candidate_no_context | context_overlap | 0.0226 | 0.6155 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.0183 | 0.1045 |
| controlled_vs_candidate_no_context | persona_style | -0.0339 | -0.0631 |
| controlled_vs_candidate_no_context | distinct1 | -0.0178 | -0.0190 |
| controlled_vs_candidate_no_context | length_score | -0.0611 | -0.1063 |
| controlled_vs_candidate_no_context | sentence_score | 0.0000 | 0.0000 |
| controlled_vs_candidate_no_context | overall_quality | 0.0280 | 0.1008 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0298 | 0.3054 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0290 | 0.1138 |
| controlled_alt_vs_controlled_default | naturalness | 0.0023 | 0.0027 |
| controlled_alt_vs_controlled_default | quest_state_correctness | 0.0228 | 0.1239 |
| controlled_alt_vs_controlled_default | lore_consistency | 0.0366 | 1.1221 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_controlled_default | objective_completion_support | -0.0019 | -0.0265 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | 0.0068 | 0.0228 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | -0.0001 | -0.0004 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0424 | 0.3721 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0004 | 0.0066 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0323 | 0.1677 |
| controlled_alt_vs_controlled_default | persona_style | 0.0156 | 0.0311 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0091 | 0.0100 |
| controlled_alt_vs_controlled_default | length_score | -0.0083 | -0.0162 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0000 | 0.0000 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0252 | 0.0824 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.0369 | 0.4077 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.0442 | 0.1846 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0003 | 0.0004 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 0.0338 | 0.1949 |
| controlled_alt_vs_proposed_raw | lore_consistency | 0.0320 | 0.8595 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_proposed_raw | objective_completion_support | 0.0177 | 0.3290 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 0.0155 | 0.0533 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | 0.0011 | 0.0039 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.0485 | 0.4491 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0099 | 0.1982 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.0585 | 0.3512 |
| controlled_alt_vs_proposed_raw | persona_style | -0.0130 | -0.0245 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0072 | -0.0077 |
| controlled_alt_vs_proposed_raw | length_score | -0.0083 | -0.0162 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0583 | 0.0619 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.0340 | 0.1144 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.0920 | 2.5982 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.0368 | 0.1491 |
| controlled_alt_vs_candidate_no_context | naturalness | -0.0184 | -0.0206 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 0.0809 | 0.6404 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 0.0637 | 11.3742 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_candidate_no_context | objective_completion_support | -0.0044 | -0.0576 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 0.0150 | 0.0515 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | -0.0125 | -0.0419 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.1216 | 3.4891 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0230 | 0.6261 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.0506 | 0.2898 |
| controlled_alt_vs_candidate_no_context | persona_style | -0.0182 | -0.0340 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0087 | -0.0093 |
| controlled_alt_vs_candidate_no_context | length_score | -0.0694 | -0.1208 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.0000 | 0.0000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.0533 | 0.1916 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0551 | (0.0234, 0.0893) | 0.0000 | 0.0551 | (0.0238, 0.0842) | 0.0010 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0074 | (-0.0516, 0.0358) | 0.6227 | -0.0074 | (-0.0518, 0.0230) | 0.6610 |
| proposed_vs_candidate_no_context | naturalness | -0.0187 | (-0.0397, 0.0007) | 0.9697 | -0.0187 | (-0.0478, -0.0013) | 1.0000 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.0471 | (0.0191, 0.0763) | 0.0000 | 0.0471 | (0.0203, 0.0705) | 0.0003 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0317 | (0.0111, 0.0543) | 0.0007 | 0.0317 | (0.0120, 0.0529) | 0.0007 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | -0.0221 | (-0.0449, -0.0015) | 0.9817 | -0.0221 | (-0.0323, -0.0091) | 1.0000 |
| proposed_vs_candidate_no_context | gameplay_usefulness | -0.0005 | (-0.0156, 0.0156) | 0.5413 | -0.0005 | (-0.0166, 0.0089) | 0.5257 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | -0.0136 | (-0.0256, -0.0020) | 0.9870 | -0.0136 | (-0.0289, -0.0008) | 0.9897 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0731 | (0.0303, 0.1197) | 0.0000 | 0.0731 | (0.0325, 0.1118) | 0.0000 |
| proposed_vs_candidate_no_context | context_overlap | 0.0131 | (0.0032, 0.0240) | 0.0030 | 0.0131 | (0.0047, 0.0229) | 0.0000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0079 | (-0.0635, 0.0476) | 0.6377 | -0.0079 | (-0.0617, 0.0258) | 0.6510 |
| proposed_vs_candidate_no_context | persona_style | -0.0052 | (-0.0326, 0.0247) | 0.6827 | -0.0052 | (-0.0402, 0.0341) | 0.6380 |
| proposed_vs_candidate_no_context | distinct1 | -0.0015 | (-0.0155, 0.0143) | 0.5907 | -0.0015 | (-0.0147, 0.0167) | 0.6173 |
| proposed_vs_candidate_no_context | length_score | -0.0611 | (-0.1458, 0.0236) | 0.9143 | -0.0611 | (-0.1697, 0.0024) | 0.9610 |
| proposed_vs_candidate_no_context | sentence_score | -0.0583 | (-0.1167, -0.0146) | 1.0000 | -0.0583 | (-0.1225, -0.0125) | 1.0000 |
| proposed_vs_candidate_no_context | overall_quality | 0.0193 | (-0.0026, 0.0407) | 0.0400 | 0.0193 | (0.0045, 0.0342) | 0.0140 |
| controlled_vs_proposed_raw | context_relevance | 0.0071 | (-0.0394, 0.0517) | 0.3650 | 0.0071 | (-0.0215, 0.0357) | 0.3220 |
| controlled_vs_proposed_raw | persona_consistency | 0.0152 | (-0.0463, 0.0773) | 0.3337 | 0.0152 | (-0.0595, 0.0800) | 0.3073 |
| controlled_vs_proposed_raw | naturalness | -0.0020 | (-0.0295, 0.0238) | 0.5720 | -0.0020 | (-0.0376, 0.0386) | 0.5423 |
| controlled_vs_proposed_raw | quest_state_correctness | 0.0110 | (-0.0326, 0.0506) | 0.2850 | 0.0110 | (-0.0172, 0.0418) | 0.2497 |
| controlled_vs_proposed_raw | lore_consistency | -0.0046 | (-0.0286, 0.0230) | 0.6357 | -0.0046 | (-0.0320, 0.0214) | 0.6827 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0197 | (-0.0033, 0.0463) | 0.0473 | 0.0197 | (0.0026, 0.0398) | 0.0080 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0087 | (-0.0073, 0.0280) | 0.1783 | 0.0087 | (-0.0083, 0.0324) | 0.1860 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 0.0012 | (-0.0135, 0.0172) | 0.4450 | 0.0012 | (-0.0071, 0.0132) | 0.4387 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.0061 | (-0.0538, 0.0644) | 0.4100 | 0.0061 | (-0.0295, 0.0417) | 0.3633 |
| controlled_vs_proposed_raw | context_overlap | 0.0095 | (-0.0126, 0.0324) | 0.2067 | 0.0095 | (-0.0077, 0.0260) | 0.1520 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.0262 | (-0.0508, 0.1006) | 0.2543 | 0.0262 | (-0.0552, 0.1000) | 0.2223 |
| controlled_vs_proposed_raw | persona_style | -0.0286 | (-0.0612, -0.0026) | 0.9883 | -0.0286 | (-0.0750, 0.0000) | 1.0000 |
| controlled_vs_proposed_raw | distinct1 | -0.0163 | (-0.0402, 0.0034) | 0.9417 | -0.0163 | (-0.0520, 0.0047) | 0.8897 |
| controlled_vs_proposed_raw | length_score | -0.0000 | (-0.0986, 0.1028) | 0.5037 | -0.0000 | (-0.1333, 0.1583) | 0.5440 |
| controlled_vs_proposed_raw | sentence_score | 0.0583 | (0.0146, 0.1167) | 0.0107 | 0.0583 | (0.0125, 0.1225) | 0.0103 |
| controlled_vs_proposed_raw | overall_quality | 0.0088 | (-0.0320, 0.0501) | 0.3323 | 0.0088 | (-0.0269, 0.0451) | 0.2997 |
| controlled_vs_candidate_no_context | context_relevance | 0.0622 | (0.0268, 0.1003) | 0.0000 | 0.0622 | (0.0325, 0.1038) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.0078 | (-0.0376, 0.0526) | 0.3557 | 0.0078 | (-0.0463, 0.0575) | 0.3507 |
| controlled_vs_candidate_no_context | naturalness | -0.0207 | (-0.0569, 0.0112) | 0.8800 | -0.0207 | (-0.0807, 0.0276) | 0.7733 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.0581 | (0.0240, 0.0918) | 0.0000 | 0.0581 | (0.0323, 0.0958) | 0.0000 |
| controlled_vs_candidate_no_context | lore_consistency | 0.0270 | (0.0074, 0.0495) | 0.0020 | 0.0270 | (0.0133, 0.0411) | 0.0007 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | -0.0024 | (-0.0299, 0.0251) | 0.5913 | -0.0024 | (-0.0177, 0.0148) | 0.5950 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0082 | (-0.0082, 0.0242) | 0.1590 | 0.0082 | (-0.0143, 0.0351) | 0.2810 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | -0.0124 | (-0.0284, 0.0037) | 0.9343 | -0.0124 | (-0.0290, 0.0063) | 0.9077 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.0792 | (0.0333, 0.1288) | 0.0003 | 0.0792 | (0.0386, 0.1405) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0226 | (0.0060, 0.0429) | 0.0033 | 0.0226 | (0.0086, 0.0347) | 0.0013 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.0183 | (-0.0353, 0.0708) | 0.2640 | 0.0183 | (-0.0397, 0.0764) | 0.2437 |
| controlled_vs_candidate_no_context | persona_style | -0.0339 | (-0.0651, -0.0091) | 1.0000 | -0.0339 | (-0.0745, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | distinct1 | -0.0178 | (-0.0457, 0.0082) | 0.9003 | -0.0178 | (-0.0629, 0.0103) | 0.8273 |
| controlled_vs_candidate_no_context | length_score | -0.0611 | (-0.1945, 0.0639) | 0.8170 | -0.0611 | (-0.2667, 0.1242) | 0.7183 |
| controlled_vs_candidate_no_context | sentence_score | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | overall_quality | 0.0280 | (-0.0009, 0.0579) | 0.0290 | 0.0280 | (-0.0014, 0.0666) | 0.0363 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0298 | (-0.0293, 0.1009) | 0.1853 | 0.0298 | (-0.0210, 0.0807) | 0.1437 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0290 | (-0.0410, 0.1084) | 0.2460 | 0.0290 | (-0.0113, 0.0995) | 0.1203 |
| controlled_alt_vs_controlled_default | naturalness | 0.0023 | (-0.0319, 0.0342) | 0.4457 | 0.0023 | (-0.0297, 0.0322) | 0.3747 |
| controlled_alt_vs_controlled_default | quest_state_correctness | 0.0228 | (-0.0293, 0.0870) | 0.2247 | 0.0228 | (-0.0207, 0.0674) | 0.1577 |
| controlled_alt_vs_controlled_default | lore_consistency | 0.0366 | (-0.0080, 0.0922) | 0.0583 | 0.0366 | (-0.0123, 0.0855) | 0.0633 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | objective_completion_support | -0.0019 | (-0.0226, 0.0161) | 0.5607 | -0.0019 | (-0.0133, 0.0092) | 0.5850 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | 0.0068 | (-0.0050, 0.0195) | 0.1217 | 0.0068 | (-0.0074, 0.0206) | 0.1500 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | -0.0001 | (-0.0192, 0.0198) | 0.5177 | -0.0001 | (-0.0146, 0.0153) | 0.5437 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0424 | (-0.0371, 0.1379) | 0.1680 | 0.0424 | (-0.0227, 0.1076) | 0.0850 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0004 | (-0.0244, 0.0232) | 0.4617 | 0.0004 | (-0.0237, 0.0245) | 0.4690 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0323 | (-0.0577, 0.1345) | 0.2637 | 0.0323 | (-0.0163, 0.1136) | 0.1970 |
| controlled_alt_vs_controlled_default | persona_style | 0.0156 | (0.0000, 0.0417) | 0.1277 | 0.0156 | (0.0000, 0.0375) | 0.0857 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0091 | (-0.0107, 0.0314) | 0.1997 | 0.0091 | (-0.0110, 0.0442) | 0.2963 |
| controlled_alt_vs_controlled_default | length_score | -0.0083 | (-0.1389, 0.1153) | 0.5467 | -0.0083 | (-0.1182, 0.0682) | 0.5617 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0252 | (-0.0155, 0.0759) | 0.1283 | 0.0252 | (-0.0025, 0.0442) | 0.0430 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.0369 | (-0.0236, 0.1066) | 0.1427 | 0.0369 | (0.0146, 0.0592) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.0442 | (-0.0477, 0.1447) | 0.1860 | 0.0442 | (-0.0360, 0.1604) | 0.2040 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0003 | (-0.0215, 0.0229) | 0.4837 | 0.0003 | (-0.0190, 0.0287) | 0.4777 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 0.0338 | (-0.0157, 0.0929) | 0.1060 | 0.0338 | (0.0145, 0.0502) | 0.0003 |
| controlled_alt_vs_proposed_raw | lore_consistency | 0.0320 | (-0.0047, 0.0791) | 0.0503 | 0.0320 | (0.0086, 0.0594) | 0.0003 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_proposed_raw | objective_completion_support | 0.0177 | (-0.0001, 0.0403) | 0.0283 | 0.0177 | (-0.0032, 0.0446) | 0.0820 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 0.0155 | (-0.0011, 0.0343) | 0.0347 | 0.0155 | (0.0030, 0.0331) | 0.0110 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | 0.0011 | (-0.0207, 0.0235) | 0.4443 | 0.0011 | (-0.0111, 0.0186) | 0.4600 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.0485 | (-0.0318, 0.1356) | 0.1347 | 0.0485 | (0.0168, 0.0780) | 0.0003 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0099 | (-0.0080, 0.0277) | 0.1310 | 0.0099 | (0.0023, 0.0174) | 0.0143 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.0585 | (-0.0605, 0.1895) | 0.1847 | 0.0585 | (-0.0357, 0.2045) | 0.1323 |
| controlled_alt_vs_proposed_raw | persona_style | -0.0130 | (-0.0560, 0.0299) | 0.7350 | -0.0130 | (-0.0369, 0.0000) | 1.0000 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0072 | (-0.0282, 0.0116) | 0.7627 | -0.0072 | (-0.0132, -0.0026) | 1.0000 |
| controlled_alt_vs_proposed_raw | length_score | -0.0083 | (-0.1056, 0.0848) | 0.5740 | -0.0083 | (-0.0903, 0.1125) | 0.5737 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0583 | (0.0146, 0.1167) | 0.0130 | 0.0583 | (0.0125, 0.1273) | 0.0077 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.0340 | (-0.0189, 0.0927) | 0.1177 | 0.0340 | (-0.0000, 0.0823) | 0.0257 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.0920 | (0.0362, 0.1606) | 0.0000 | 0.0920 | (0.0404, 0.1363) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.0368 | (-0.0391, 0.1202) | 0.1857 | 0.0368 | (-0.0267, 0.1455) | 0.2610 |
| controlled_alt_vs_candidate_no_context | naturalness | -0.0184 | (-0.0458, 0.0083) | 0.9110 | -0.0184 | (-0.0533, 0.0174) | 0.8350 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 0.0809 | (0.0295, 0.1386) | 0.0000 | 0.0809 | (0.0353, 0.1226) | 0.0000 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 0.0637 | (0.0213, 0.1189) | 0.0000 | 0.0637 | (0.0212, 0.1069) | 0.0007 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_candidate_no_context | objective_completion_support | -0.0044 | (-0.0265, 0.0182) | 0.6627 | -0.0044 | (-0.0281, 0.0236) | 0.6210 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 0.0150 | (-0.0003, 0.0313) | 0.0263 | 0.0150 | (0.0020, 0.0322) | 0.0007 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | -0.0125 | (-0.0286, 0.0019) | 0.9530 | -0.0125 | (-0.0280, 0.0011) | 0.9557 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.1216 | (0.0458, 0.2053) | 0.0000 | 0.1216 | (0.0519, 0.1789) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0230 | (0.0055, 0.0406) | 0.0050 | 0.0230 | (0.0080, 0.0384) | 0.0003 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.0506 | (-0.0487, 0.1578) | 0.1630 | 0.0506 | (-0.0288, 0.1818) | 0.2077 |
| controlled_alt_vs_candidate_no_context | persona_style | -0.0182 | (-0.0534, 0.0169) | 0.8610 | -0.0182 | (-0.0505, 0.0000) | 1.0000 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0087 | (-0.0323, 0.0153) | 0.7707 | -0.0087 | (-0.0242, 0.0086) | 0.8320 |
| controlled_alt_vs_candidate_no_context | length_score | -0.0694 | (-0.1736, 0.0361) | 0.9090 | -0.0694 | (-0.2258, 0.0708) | 0.8403 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.0533 | (0.0026, 0.1078) | 0.0207 | 0.0533 | (0.0132, 0.1059) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 11 | 4 | 9 | 0.6458 | 0.7333 |
| proposed_vs_candidate_no_context | persona_consistency | 5 | 5 | 14 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | naturalness | 5 | 10 | 9 | 0.3958 | 0.3333 |
| proposed_vs_candidate_no_context | quest_state_correctness | 12 | 3 | 9 | 0.6875 | 0.8000 |
| proposed_vs_candidate_no_context | lore_consistency | 9 | 1 | 14 | 0.6667 | 0.9000 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| proposed_vs_candidate_no_context | objective_completion_support | 2 | 7 | 15 | 0.3958 | 0.2222 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 5 | 10 | 9 | 0.3958 | 0.3333 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | 3 | 11 | 10 | 0.3333 | 0.2143 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 9 | 1 | 14 | 0.6667 | 0.9000 |
| proposed_vs_candidate_no_context | context_overlap | 10 | 5 | 9 | 0.6042 | 0.6667 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 4 | 4 | 16 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | persona_style | 1 | 3 | 20 | 0.4583 | 0.2500 |
| proposed_vs_candidate_no_context | distinct1 | 4 | 7 | 13 | 0.4375 | 0.3636 |
| proposed_vs_candidate_no_context | length_score | 4 | 9 | 11 | 0.3958 | 0.3077 |
| proposed_vs_candidate_no_context | sentence_score | 0 | 4 | 20 | 0.4167 | 0.0000 |
| proposed_vs_candidate_no_context | overall_quality | 9 | 6 | 9 | 0.5625 | 0.6000 |
| controlled_vs_proposed_raw | context_relevance | 11 | 8 | 5 | 0.5625 | 0.5789 |
| controlled_vs_proposed_raw | persona_consistency | 8 | 8 | 8 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | naturalness | 6 | 13 | 5 | 0.3542 | 0.3158 |
| controlled_vs_proposed_raw | quest_state_correctness | 11 | 8 | 5 | 0.5625 | 0.5789 |
| controlled_vs_proposed_raw | lore_consistency | 4 | 7 | 13 | 0.4375 | 0.3636 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_vs_proposed_raw | objective_completion_support | 6 | 5 | 13 | 0.5208 | 0.5455 |
| controlled_vs_proposed_raw | gameplay_usefulness | 9 | 10 | 5 | 0.4792 | 0.4737 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 6 | 6 | 12 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | context_keyword_coverage | 6 | 5 | 13 | 0.5208 | 0.5455 |
| controlled_vs_proposed_raw | context_overlap | 11 | 8 | 5 | 0.5625 | 0.5789 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 8 | 6 | 10 | 0.5417 | 0.5714 |
| controlled_vs_proposed_raw | persona_style | 1 | 5 | 18 | 0.4167 | 0.1667 |
| controlled_vs_proposed_raw | distinct1 | 8 | 10 | 6 | 0.4583 | 0.4444 |
| controlled_vs_proposed_raw | length_score | 7 | 10 | 7 | 0.4375 | 0.4118 |
| controlled_vs_proposed_raw | sentence_score | 4 | 0 | 20 | 0.5833 | 1.0000 |
| controlled_vs_proposed_raw | overall_quality | 12 | 7 | 5 | 0.6042 | 0.6316 |
| controlled_vs_candidate_no_context | context_relevance | 14 | 4 | 6 | 0.7083 | 0.7778 |
| controlled_vs_candidate_no_context | persona_consistency | 7 | 7 | 10 | 0.5000 | 0.5000 |
| controlled_vs_candidate_no_context | naturalness | 7 | 11 | 6 | 0.4167 | 0.3889 |
| controlled_vs_candidate_no_context | quest_state_correctness | 14 | 4 | 6 | 0.7083 | 0.7778 |
| controlled_vs_candidate_no_context | lore_consistency | 7 | 2 | 15 | 0.6042 | 0.7778 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_vs_candidate_no_context | objective_completion_support | 6 | 8 | 10 | 0.4583 | 0.4286 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 9 | 9 | 6 | 0.5000 | 0.5000 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 4 | 10 | 10 | 0.3750 | 0.2857 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 10 | 2 | 12 | 0.6667 | 0.8333 |
| controlled_vs_candidate_no_context | context_overlap | 13 | 5 | 6 | 0.6667 | 0.7222 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 7 | 5 | 12 | 0.5417 | 0.5833 |
| controlled_vs_candidate_no_context | persona_style | 0 | 5 | 19 | 0.3958 | 0.0000 |
| controlled_vs_candidate_no_context | distinct1 | 7 | 10 | 7 | 0.4375 | 0.4118 |
| controlled_vs_candidate_no_context | length_score | 7 | 11 | 6 | 0.4167 | 0.3889 |
| controlled_vs_candidate_no_context | sentence_score | 0 | 0 | 24 | 0.5000 | nan |
| controlled_vs_candidate_no_context | overall_quality | 13 | 5 | 6 | 0.6667 | 0.7222 |
| controlled_alt_vs_controlled_default | context_relevance | 8 | 5 | 11 | 0.5625 | 0.6154 |
| controlled_alt_vs_controlled_default | persona_consistency | 5 | 5 | 14 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | naturalness | 6 | 7 | 11 | 0.4792 | 0.4615 |
| controlled_alt_vs_controlled_default | quest_state_correctness | 6 | 7 | 11 | 0.4792 | 0.4615 |
| controlled_alt_vs_controlled_default | lore_consistency | 7 | 4 | 13 | 0.5625 | 0.6364 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_alt_vs_controlled_default | objective_completion_support | 6 | 4 | 14 | 0.5417 | 0.6000 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | 8 | 5 | 11 | 0.5625 | 0.6154 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | 4 | 9 | 11 | 0.3958 | 0.3077 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 5 | 4 | 15 | 0.5208 | 0.5556 |
| controlled_alt_vs_controlled_default | context_overlap | 9 | 4 | 11 | 0.6042 | 0.6923 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 5 | 5 | 14 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | persona_style | 2 | 0 | 22 | 0.5417 | 1.0000 |
| controlled_alt_vs_controlled_default | distinct1 | 6 | 7 | 11 | 0.4792 | 0.4615 |
| controlled_alt_vs_controlled_default | length_score | 6 | 7 | 11 | 0.4792 | 0.4615 |
| controlled_alt_vs_controlled_default | sentence_score | 0 | 0 | 24 | 0.5000 | nan |
| controlled_alt_vs_controlled_default | overall_quality | 7 | 6 | 11 | 0.5208 | 0.5385 |
| controlled_alt_vs_proposed_raw | context_relevance | 8 | 10 | 6 | 0.4583 | 0.4444 |
| controlled_alt_vs_proposed_raw | persona_consistency | 8 | 7 | 9 | 0.5208 | 0.5333 |
| controlled_alt_vs_proposed_raw | naturalness | 8 | 9 | 7 | 0.4792 | 0.4706 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 8 | 10 | 6 | 0.4583 | 0.4444 |
| controlled_alt_vs_proposed_raw | lore_consistency | 7 | 4 | 13 | 0.5625 | 0.6364 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_alt_vs_proposed_raw | objective_completion_support | 5 | 2 | 17 | 0.5625 | 0.7143 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 11 | 7 | 6 | 0.5833 | 0.6111 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | 6 | 4 | 14 | 0.5417 | 0.6000 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 5 | 6 | 13 | 0.4792 | 0.4545 |
| controlled_alt_vs_proposed_raw | context_overlap | 8 | 10 | 6 | 0.4583 | 0.4444 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 8 | 6 | 10 | 0.5417 | 0.5714 |
| controlled_alt_vs_proposed_raw | persona_style | 2 | 5 | 17 | 0.4375 | 0.2857 |
| controlled_alt_vs_proposed_raw | distinct1 | 8 | 8 | 8 | 0.5000 | 0.5000 |
| controlled_alt_vs_proposed_raw | length_score | 8 | 8 | 8 | 0.5000 | 0.5000 |
| controlled_alt_vs_proposed_raw | sentence_score | 4 | 0 | 20 | 0.5833 | 1.0000 |
| controlled_alt_vs_proposed_raw | overall_quality | 10 | 8 | 6 | 0.5417 | 0.5556 |
| controlled_alt_vs_candidate_no_context | context_relevance | 12 | 5 | 7 | 0.6458 | 0.7059 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 7 | 5 | 12 | 0.5417 | 0.5833 |
| controlled_alt_vs_candidate_no_context | naturalness | 6 | 11 | 7 | 0.3958 | 0.3529 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 12 | 5 | 7 | 0.6458 | 0.7059 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 9 | 0 | 15 | 0.6875 | 1.0000 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_alt_vs_candidate_no_context | objective_completion_support | 5 | 7 | 12 | 0.4583 | 0.4167 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 10 | 7 | 7 | 0.5625 | 0.5882 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | 4 | 10 | 10 | 0.3750 | 0.2857 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 10 | 1 | 13 | 0.6875 | 0.9091 |
| controlled_alt_vs_candidate_no_context | context_overlap | 12 | 5 | 7 | 0.6458 | 0.7059 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 7 | 4 | 13 | 0.5625 | 0.6364 |
| controlled_alt_vs_candidate_no_context | persona_style | 1 | 4 | 19 | 0.4375 | 0.2000 |
| controlled_alt_vs_candidate_no_context | distinct1 | 7 | 8 | 9 | 0.4792 | 0.4667 |
| controlled_alt_vs_candidate_no_context | length_score | 6 | 11 | 7 | 0.3958 | 0.3529 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0 | 0 | 24 | 0.5000 | nan |
| controlled_alt_vs_candidate_no_context | overall_quality | 11 | 6 | 7 | 0.6042 | 0.6471 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.1250 | 0.2083 | 0.7917 |
| proposed_contextual_controlled_tuned | 0.0000 | 0.0000 | 0.0833 | 0.2500 | 0.7500 |
| proposed_contextual | 0.0000 | 0.0000 | 0.4583 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.4583 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `5`
- Unique template signatures: `21`
- Template signature ratio: `0.8750`
- Effective sample size by source clustering: `4.80`
- Effective sample size by template-signature clustering: `19.20`
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
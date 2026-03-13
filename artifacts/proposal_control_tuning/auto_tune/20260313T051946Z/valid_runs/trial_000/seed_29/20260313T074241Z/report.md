# Proposal Alignment Evaluation Report

- Run ID: `20260313T074241Z`
- Generated: `2026-03-13T07:49:08.598668+00:00`
- Scenarios: `artifacts\proposal_control_tuning\auto_tune\20260313T051946Z\valid_runs\trial_000\seed_29\20260313T074241Z\scenarios.jsonl`
- Scenario count: `24`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_tuned`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.1345 (0.0877, 0.1925) | 0.3505 (0.3029, 0.3943) | 0.8623 (0.8498, 0.8726) | 0.3545 (0.3293, 0.3816) | n/a |
| proposed_contextual_controlled_tuned | 0.1731 (0.1062, 0.2503) | 0.3337 (0.2896, 0.3776) | 0.8697 (0.8527, 0.8858) | 0.3672 (0.3372, 0.4001) | n/a |
| proposed_contextual | 0.0608 (0.0419, 0.0817) | 0.2790 (0.2268, 0.3323) | 0.8776 (0.8602, 0.8953) | 0.2973 (0.2737, 0.3207) | n/a |
| candidate_no_context | 0.0569 (0.0421, 0.0732) | 0.3096 (0.2568, 0.3599) | 0.8835 (0.8692, 0.9004) | 0.3084 (0.2843, 0.3297) | n/a |

## Game-facing Outcome Metrics (mean, 95% CI)
| Arm | Quest-state Correctness | Lore Consistency | Contradiction Safety | Objective Completion Support | Gameplay Usefulness | Time-pressure Acceptability |
|---|---:|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2236 (0.1825, 0.2718) | 0.0775 (0.0295, 0.1409) | 1.0000 (1.0000, 1.0000) | 0.1338 (0.0921, 0.1760) | 0.3273 (0.3086, 0.3463) | 0.3277 (0.2976, 0.3580) |
| proposed_contextual_controlled_tuned | 0.2521 (0.1959, 0.3142) | 0.1053 (0.0463, 0.1729) | 1.0000 (1.0000, 1.0000) | 0.1162 (0.0704, 0.1620) | 0.3361 (0.3122, 0.3581) | 0.3210 (0.2905, 0.3536) |
| proposed_contextual | 0.1602 (0.1399, 0.1809) | 0.0139 (0.0034, 0.0274) | 1.0000 (1.0000, 1.0000) | 0.1194 (0.0724, 0.1669) | 0.3111 (0.2857, 0.3380) | 0.3309 (0.3018, 0.3605) |
| candidate_no_context | 0.1542 (0.1387, 0.1686) | 0.0050 (0.0000, 0.0120) | 1.0000 (1.0000, 1.0000) | 0.1201 (0.0717, 0.1677) | 0.3154 (0.2895, 0.3414) | 0.3326 (0.3029, 0.3646) |

- Multi-turn contradiction rate is reported as `1 - contradiction_safety` in row-level outputs.

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0039 | 0.0688 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0307 | -0.0990 |
| proposed_vs_candidate_no_context | naturalness | -0.0059 | -0.0067 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.0061 | 0.0393 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0089 | 1.7991 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | -0.0007 | -0.0061 |
| proposed_vs_candidate_no_context | gameplay_usefulness | -0.0043 | -0.0135 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | -0.0017 | -0.0052 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0041 | 0.0660 |
| proposed_vs_candidate_no_context | context_overlap | 0.0035 | 0.0779 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0411 | -0.1847 |
| proposed_vs_candidate_no_context | persona_style | 0.0110 | 0.0167 |
| proposed_vs_candidate_no_context | distinct1 | 0.0048 | 0.0051 |
| proposed_vs_candidate_no_context | length_score | -0.0319 | -0.0587 |
| proposed_vs_candidate_no_context | sentence_score | -0.0146 | -0.0148 |
| proposed_vs_candidate_no_context | overall_quality | -0.0111 | -0.0360 |
| controlled_vs_proposed_raw | context_relevance | 0.0737 | 1.2124 |
| controlled_vs_proposed_raw | persona_consistency | 0.0715 | 0.2564 |
| controlled_vs_proposed_raw | naturalness | -0.0152 | -0.0174 |
| controlled_vs_proposed_raw | quest_state_correctness | 0.0634 | 0.3956 |
| controlled_vs_proposed_raw | lore_consistency | 0.0636 | 4.5703 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0144 | 0.1207 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0161 | 0.0519 |
| controlled_vs_proposed_raw | time_pressure_acceptability | -0.0031 | -0.0094 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.0960 | 1.4476 |
| controlled_vs_proposed_raw | context_overlap | 0.0217 | 0.4533 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.0762 | 0.4201 |
| controlled_vs_proposed_raw | persona_style | 0.0529 | 0.0790 |
| controlled_vs_proposed_raw | distinct1 | 0.0006 | 0.0007 |
| controlled_vs_proposed_raw | length_score | -0.0847 | -0.1653 |
| controlled_vs_proposed_raw | sentence_score | 0.0146 | 0.0150 |
| controlled_vs_proposed_raw | overall_quality | 0.0572 | 0.1924 |
| controlled_vs_candidate_no_context | context_relevance | 0.0776 | 1.3646 |
| controlled_vs_candidate_no_context | persona_consistency | 0.0409 | 0.1320 |
| controlled_vs_candidate_no_context | naturalness | -0.0212 | -0.0240 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.0695 | 0.4505 |
| controlled_vs_candidate_no_context | lore_consistency | 0.0725 | 14.5919 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | 0.0137 | 0.1139 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0119 | 0.0377 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | -0.0048 | -0.0146 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.1001 | 1.6091 |
| controlled_vs_candidate_no_context | context_overlap | 0.0252 | 0.5665 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.0351 | 0.1579 |
| controlled_vs_candidate_no_context | persona_style | 0.0639 | 0.0970 |
| controlled_vs_candidate_no_context | distinct1 | 0.0054 | 0.0057 |
| controlled_vs_candidate_no_context | length_score | -0.1167 | -0.2143 |
| controlled_vs_candidate_no_context | sentence_score | 0.0000 | 0.0000 |
| controlled_vs_candidate_no_context | overall_quality | 0.0461 | 0.1494 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0386 | 0.2872 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0168 | -0.0479 |
| controlled_alt_vs_controlled_default | naturalness | 0.0073 | 0.0085 |
| controlled_alt_vs_controlled_default | quest_state_correctness | 0.0285 | 0.1273 |
| controlled_alt_vs_controlled_default | lore_consistency | 0.0277 | 0.3579 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_controlled_default | objective_completion_support | -0.0176 | -0.1318 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | 0.0088 | 0.0270 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | -0.0068 | -0.0207 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0464 | 0.2860 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0205 | 0.2939 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0218 | -0.0847 |
| controlled_alt_vs_controlled_default | persona_style | 0.0033 | 0.0046 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0106 | -0.0112 |
| controlled_alt_vs_controlled_default | length_score | 0.0597 | 0.1396 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0000 | 0.0000 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0127 | 0.0359 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1123 | 1.8479 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.0547 | 0.1962 |
| controlled_alt_vs_proposed_raw | naturalness | -0.0079 | -0.0090 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 0.0919 | 0.5733 |
| controlled_alt_vs_proposed_raw | lore_consistency | 0.0913 | 6.5637 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_proposed_raw | objective_completion_support | -0.0032 | -0.0270 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 0.0250 | 0.0803 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | -0.0099 | -0.0299 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.1424 | 2.1476 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0422 | 0.8804 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.0544 | 0.2998 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0562 | 0.0839 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0099 | -0.0105 |
| controlled_alt_vs_proposed_raw | length_score | -0.0250 | -0.0488 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0146 | 0.0150 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.0699 | 0.2352 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.1162 | 2.0438 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.0241 | 0.0778 |
| controlled_alt_vs_candidate_no_context | naturalness | -0.0138 | -0.0157 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 0.0979 | 0.6352 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 0.1003 | 20.1715 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_candidate_no_context | objective_completion_support | -0.0040 | -0.0329 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 0.0207 | 0.0657 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | -0.0116 | -0.0350 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.1465 | 2.3553 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0456 | 1.0268 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.0133 | 0.0598 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0672 | 0.1021 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0052 | -0.0055 |
| controlled_alt_vs_candidate_no_context | length_score | -0.0569 | -0.1046 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.0000 | 0.0000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.0588 | 0.1907 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0039 | (-0.0125, 0.0197) | 0.3043 | 0.0039 | (-0.0084, 0.0149) | 0.2957 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0307 | (-0.0634, -0.0053) | 0.9953 | -0.0307 | (-0.0634, -0.0158) | 1.0000 |
| proposed_vs_candidate_no_context | naturalness | -0.0059 | (-0.0303, 0.0181) | 0.6697 | -0.0059 | (-0.0139, 0.0122) | 0.7333 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.0061 | (-0.0064, 0.0188) | 0.1543 | 0.0061 | (-0.0033, 0.0167) | 0.3103 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0089 | (-0.0012, 0.0214) | 0.0450 | 0.0089 | (0.0057, 0.0164) | 0.0000 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | -0.0007 | (-0.0183, 0.0150) | 0.5200 | -0.0007 | (-0.0029, 0.0019) | 0.7430 |
| proposed_vs_candidate_no_context | gameplay_usefulness | -0.0043 | (-0.0217, 0.0112) | 0.6947 | -0.0043 | (-0.0058, 0.0001) | 0.9643 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | -0.0017 | (-0.0104, 0.0057) | 0.6347 | -0.0017 | (-0.0083, 0.0068) | 0.7467 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0041 | (-0.0139, 0.0262) | 0.3540 | 0.0041 | (-0.0139, 0.0182) | 0.2920 |
| proposed_vs_candidate_no_context | context_overlap | 0.0035 | (-0.0021, 0.0104) | 0.1333 | 0.0035 | (-0.0020, 0.0073) | 0.1490 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0411 | (-0.0835, -0.0069) | 1.0000 | -0.0411 | (-0.0833, -0.0250) | 1.0000 |
| proposed_vs_candidate_no_context | persona_style | 0.0110 | (-0.0085, 0.0361) | 0.2050 | 0.0110 | (0.0000, 0.0208) | 0.0417 |
| proposed_vs_candidate_no_context | distinct1 | 0.0048 | (-0.0074, 0.0168) | 0.2017 | 0.0048 | (-0.0064, 0.0139) | 0.2643 |
| proposed_vs_candidate_no_context | length_score | -0.0319 | (-0.1333, 0.0667) | 0.7397 | -0.0319 | (-0.0567, 0.0333) | 0.9623 |
| proposed_vs_candidate_no_context | sentence_score | -0.0146 | (-0.0729, 0.0292) | 0.8177 | -0.0146 | (-0.0437, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | overall_quality | -0.0111 | (-0.0282, 0.0047) | 0.9013 | -0.0111 | (-0.0257, -0.0045) | 1.0000 |
| controlled_vs_proposed_raw | context_relevance | 0.0737 | (0.0262, 0.1344) | 0.0003 | 0.0737 | (0.0219, 0.1323) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.0715 | (0.0290, 0.1182) | 0.0000 | 0.0715 | (0.0343, 0.1367) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | -0.0152 | (-0.0307, -0.0009) | 0.9813 | -0.0152 | (-0.0391, -0.0016) | 1.0000 |
| controlled_vs_proposed_raw | quest_state_correctness | 0.0634 | (0.0201, 0.1113) | 0.0017 | 0.0634 | (0.0153, 0.1218) | 0.0000 |
| controlled_vs_proposed_raw | lore_consistency | 0.0636 | (0.0125, 0.1240) | 0.0040 | 0.0636 | (0.0026, 0.1287) | 0.0000 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0144 | (-0.0134, 0.0476) | 0.1787 | 0.0144 | (0.0016, 0.0250) | 0.0000 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0161 | (-0.0007, 0.0352) | 0.0310 | 0.0161 | (-0.0021, 0.0371) | 0.0423 |
| controlled_vs_proposed_raw | time_pressure_acceptability | -0.0031 | (-0.0195, 0.0151) | 0.6580 | -0.0031 | (-0.0133, 0.0026) | 0.7030 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.0960 | (0.0341, 0.1657) | 0.0003 | 0.0960 | (0.0273, 0.1705) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0217 | (0.0001, 0.0466) | 0.0240 | 0.0217 | (0.0093, 0.0433) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.0762 | (0.0236, 0.1327) | 0.0017 | 0.0762 | (0.0429, 0.1667) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.0529 | (0.0112, 0.1084) | 0.0050 | 0.0529 | (0.0000, 0.1458) | 0.0403 |
| controlled_vs_proposed_raw | distinct1 | 0.0006 | (-0.0183, 0.0181) | 0.4777 | 0.0006 | (-0.0107, 0.0138) | 0.4070 |
| controlled_vs_proposed_raw | length_score | -0.0847 | (-0.1556, -0.0139) | 0.9883 | -0.0847 | (-0.1833, -0.0083) | 1.0000 |
| controlled_vs_proposed_raw | sentence_score | 0.0146 | (-0.0292, 0.0729) | 0.3967 | 0.0146 | (0.0000, 0.0437) | 0.3070 |
| controlled_vs_proposed_raw | overall_quality | 0.0572 | (0.0277, 0.0865) | 0.0000 | 0.0572 | (0.0206, 0.0847) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.0776 | (0.0256, 0.1351) | 0.0000 | 0.0776 | (0.0368, 0.1317) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.0409 | (0.0092, 0.0764) | 0.0047 | 0.0409 | (0.0114, 0.0733) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | -0.0212 | (-0.0388, -0.0046) | 0.9957 | -0.0212 | (-0.0269, -0.0112) | 1.0000 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.0695 | (0.0259, 0.1209) | 0.0003 | 0.0695 | (0.0320, 0.1216) | 0.0000 |
| controlled_vs_candidate_no_context | lore_consistency | 0.0725 | (0.0245, 0.1282) | 0.0000 | 0.0725 | (0.0097, 0.1344) | 0.0000 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | 0.0137 | (-0.0094, 0.0365) | 0.1200 | 0.0137 | (-0.0013, 0.0269) | 0.0370 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0119 | (-0.0015, 0.0272) | 0.0460 | 0.0119 | (-0.0077, 0.0313) | 0.1617 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | -0.0048 | (-0.0188, 0.0092) | 0.7537 | -0.0048 | (-0.0133, 0.0094) | 0.7427 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.1001 | (0.0366, 0.1695) | 0.0010 | 0.1001 | (0.0455, 0.1705) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0252 | (0.0042, 0.0494) | 0.0070 | 0.0252 | (0.0166, 0.0413) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.0351 | (-0.0010, 0.0758) | 0.0363 | 0.0351 | (0.0143, 0.0833) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.0639 | (0.0139, 0.1222) | 0.0053 | 0.0639 | (0.0000, 0.1667) | 0.0343 |
| controlled_vs_candidate_no_context | distinct1 | 0.0054 | (-0.0108, 0.0213) | 0.2387 | 0.0054 | (0.0012, 0.0078) | 0.0000 |
| controlled_vs_candidate_no_context | length_score | -0.1167 | (-0.2028, -0.0277) | 0.9960 | -0.1167 | (-0.1500, -0.0583) | 1.0000 |
| controlled_vs_candidate_no_context | sentence_score | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | overall_quality | 0.0461 | (0.0215, 0.0730) | 0.0000 | 0.0461 | (0.0160, 0.0763) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0386 | (-0.0578, 0.1376) | 0.2180 | 0.0386 | (0.0055, 0.1094) | 0.0000 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0168 | (-0.0597, 0.0234) | 0.7960 | -0.0168 | (-0.0889, 0.0097) | 0.7217 |
| controlled_alt_vs_controlled_default | naturalness | 0.0073 | (-0.0152, 0.0319) | 0.2783 | 0.0073 | (-0.0023, 0.0224) | 0.1437 |
| controlled_alt_vs_controlled_default | quest_state_correctness | 0.0285 | (-0.0550, 0.1129) | 0.2520 | 0.0285 | (0.0054, 0.0892) | 0.0000 |
| controlled_alt_vs_controlled_default | lore_consistency | 0.0277 | (-0.0599, 0.1190) | 0.2790 | 0.0277 | (0.0032, 0.0750) | 0.0000 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | objective_completion_support | -0.0176 | (-0.0560, 0.0215) | 0.8357 | -0.0176 | (-0.0725, 0.0108) | 0.8510 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | 0.0088 | (-0.0094, 0.0299) | 0.2063 | 0.0088 | (0.0031, 0.0195) | 0.0000 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | -0.0068 | (-0.0353, 0.0215) | 0.6783 | -0.0068 | (-0.0366, 0.0074) | 0.7433 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0464 | (-0.0751, 0.1717) | 0.2363 | 0.0464 | (0.0091, 0.1250) | 0.0000 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0205 | (-0.0201, 0.0682) | 0.1777 | 0.0205 | (-0.0029, 0.0731) | 0.0377 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0218 | (-0.0708, 0.0252) | 0.8160 | -0.0218 | (-0.1111, 0.0143) | 0.7500 |
| controlled_alt_vs_controlled_default | persona_style | 0.0033 | (-0.0490, 0.0658) | 0.4970 | 0.0033 | (-0.0088, 0.0208) | 0.4267 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0106 | (-0.0321, 0.0088) | 0.8450 | -0.0106 | (-0.0361, 0.0021) | 0.9620 |
| controlled_alt_vs_controlled_default | length_score | 0.0597 | (-0.0389, 0.1695) | 0.1257 | 0.0597 | (-0.0333, 0.1542) | 0.1487 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0000 | (-0.0437, 0.0437) | 0.6593 | 0.0000 | (-0.0437, 0.0350) | 0.6310 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0127 | (-0.0251, 0.0509) | 0.2707 | 0.0127 | (0.0060, 0.0182) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1123 | (0.0427, 0.1904) | 0.0000 | 0.1123 | (0.0274, 0.1913) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.0547 | (0.0183, 0.0964) | 0.0010 | 0.0547 | (0.0440, 0.0733) | 0.0000 |
| controlled_alt_vs_proposed_raw | naturalness | -0.0079 | (-0.0350, 0.0202) | 0.7220 | -0.0079 | (-0.0358, 0.0208) | 0.7457 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 0.0919 | (0.0343, 0.1578) | 0.0000 | 0.0919 | (0.0208, 0.1549) | 0.0000 |
| controlled_alt_vs_proposed_raw | lore_consistency | 0.0913 | (0.0324, 0.1590) | 0.0000 | 0.0913 | (0.0217, 0.1534) | 0.0000 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_proposed_raw | objective_completion_support | -0.0032 | (-0.0307, 0.0260) | 0.5963 | -0.0032 | (-0.0475, 0.0124) | 0.6953 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 0.0250 | (0.0001, 0.0527) | 0.0240 | 0.0250 | (0.0010, 0.0566) | 0.0000 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | -0.0099 | (-0.0342, 0.0135) | 0.7777 | -0.0099 | (-0.0340, 0.0013) | 0.9640 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.1424 | (0.0568, 0.2415) | 0.0000 | 0.1424 | (0.0364, 0.2361) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0422 | (0.0122, 0.0819) | 0.0013 | 0.0422 | (0.0064, 0.0867) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.0544 | (0.0119, 0.1028) | 0.0093 | 0.0544 | (0.0500, 0.0571) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0562 | (0.0102, 0.1117) | 0.0040 | 0.0562 | (-0.0088, 0.1667) | 0.1503 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0099 | (-0.0297, 0.0084) | 0.8470 | -0.0099 | (-0.0422, 0.0158) | 0.7533 |
| controlled_alt_vs_proposed_raw | length_score | -0.0250 | (-0.1347, 0.0931) | 0.6770 | -0.0250 | (-0.1200, 0.1458) | 0.6183 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0146 | (-0.0292, 0.0583) | 0.4127 | 0.0146 | (0.0000, 0.0350) | 0.3067 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.0699 | (0.0353, 0.1077) | 0.0000 | 0.0699 | (0.0265, 0.1029) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.1162 | (0.0525, 0.1901) | 0.0000 | 0.1162 | (0.0423, 0.1829) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.0241 | (-0.0162, 0.0602) | 0.1073 | 0.0241 | (-0.0156, 0.0575) | 0.0353 |
| controlled_alt_vs_candidate_no_context | naturalness | -0.0138 | (-0.0350, 0.0080) | 0.8903 | -0.0138 | (-0.0281, 0.0112) | 0.8520 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 0.0979 | (0.0439, 0.1631) | 0.0000 | 0.0979 | (0.0375, 0.1516) | 0.0000 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 0.1003 | (0.0390, 0.1737) | 0.0000 | 0.1003 | (0.0287, 0.1699) | 0.0000 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_candidate_no_context | objective_completion_support | -0.0040 | (-0.0347, 0.0325) | 0.6180 | -0.0040 | (-0.0456, 0.0104) | 0.7033 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 0.0207 | (-0.0026, 0.0460) | 0.0423 | 0.0207 | (-0.0045, 0.0507) | 0.0353 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | -0.0116 | (-0.0351, 0.0120) | 0.8353 | -0.0116 | (-0.0272, -0.0058) | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.1465 | (0.0619, 0.2412) | 0.0000 | 0.1465 | (0.0545, 0.2222) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0456 | (0.0157, 0.0855) | 0.0000 | 0.0456 | (0.0137, 0.0912) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.0133 | (-0.0304, 0.0518) | 0.2697 | 0.0133 | (-0.0278, 0.0286) | 0.2623 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0672 | (0.0139, 0.1264) | 0.0047 | 0.0672 | (-0.0088, 0.1875) | 0.0353 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0052 | (-0.0243, 0.0139) | 0.7110 | -0.0052 | (-0.0283, 0.0094) | 0.7430 |
| controlled_alt_vs_candidate_no_context | length_score | -0.0569 | (-0.1431, 0.0334) | 0.8990 | -0.0569 | (-0.1767, 0.0958) | 0.7407 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.0000 | (-0.0437, 0.0437) | 0.6570 | 0.0000 | (-0.0437, 0.0350) | 0.6333 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.0588 | (0.0286, 0.0923) | 0.0000 | 0.0588 | (0.0220, 0.0945) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 5 | 3 | 16 | 0.5417 | 0.6250 |
| proposed_vs_candidate_no_context | persona_consistency | 1 | 5 | 18 | 0.4167 | 0.1667 |
| proposed_vs_candidate_no_context | naturalness | 3 | 5 | 16 | 0.4583 | 0.3750 |
| proposed_vs_candidate_no_context | quest_state_correctness | 6 | 2 | 16 | 0.5833 | 0.7500 |
| proposed_vs_candidate_no_context | lore_consistency | 5 | 1 | 18 | 0.5833 | 0.8333 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| proposed_vs_candidate_no_context | objective_completion_support | 4 | 2 | 18 | 0.5417 | 0.6667 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 2 | 6 | 16 | 0.4167 | 0.2500 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | 3 | 4 | 17 | 0.4792 | 0.4286 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 2 | 1 | 21 | 0.5208 | 0.6667 |
| proposed_vs_candidate_no_context | context_overlap | 5 | 3 | 16 | 0.5417 | 0.6250 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0 | 4 | 20 | 0.4167 | 0.0000 |
| proposed_vs_candidate_no_context | persona_style | 2 | 1 | 21 | 0.5208 | 0.6667 |
| proposed_vs_candidate_no_context | distinct1 | 5 | 1 | 18 | 0.5833 | 0.8333 |
| proposed_vs_candidate_no_context | length_score | 3 | 5 | 16 | 0.4583 | 0.3750 |
| proposed_vs_candidate_no_context | sentence_score | 1 | 2 | 21 | 0.4792 | 0.3333 |
| proposed_vs_candidate_no_context | overall_quality | 2 | 6 | 16 | 0.4167 | 0.2500 |
| controlled_vs_proposed_raw | context_relevance | 18 | 5 | 1 | 0.7708 | 0.7826 |
| controlled_vs_proposed_raw | persona_consistency | 10 | 1 | 13 | 0.6875 | 0.9091 |
| controlled_vs_proposed_raw | naturalness | 7 | 16 | 1 | 0.3125 | 0.3043 |
| controlled_vs_proposed_raw | quest_state_correctness | 18 | 5 | 1 | 0.7708 | 0.7826 |
| controlled_vs_proposed_raw | lore_consistency | 8 | 3 | 13 | 0.6042 | 0.7273 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_vs_proposed_raw | objective_completion_support | 9 | 10 | 5 | 0.4792 | 0.4737 |
| controlled_vs_proposed_raw | gameplay_usefulness | 12 | 11 | 1 | 0.5208 | 0.5217 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 5 | 14 | 5 | 0.3125 | 0.2632 |
| controlled_vs_proposed_raw | context_keyword_coverage | 8 | 1 | 15 | 0.6458 | 0.8889 |
| controlled_vs_proposed_raw | context_overlap | 17 | 6 | 1 | 0.7292 | 0.7391 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 8 | 1 | 15 | 0.6458 | 0.8889 |
| controlled_vs_proposed_raw | persona_style | 5 | 0 | 19 | 0.6042 | 1.0000 |
| controlled_vs_proposed_raw | distinct1 | 15 | 6 | 3 | 0.6875 | 0.7143 |
| controlled_vs_proposed_raw | length_score | 5 | 17 | 2 | 0.2500 | 0.2273 |
| controlled_vs_proposed_raw | sentence_score | 2 | 1 | 21 | 0.5208 | 0.6667 |
| controlled_vs_proposed_raw | overall_quality | 14 | 9 | 1 | 0.6042 | 0.6087 |
| controlled_vs_candidate_no_context | context_relevance | 17 | 6 | 1 | 0.7292 | 0.7391 |
| controlled_vs_candidate_no_context | persona_consistency | 8 | 1 | 15 | 0.6458 | 0.8889 |
| controlled_vs_candidate_no_context | naturalness | 6 | 17 | 1 | 0.2708 | 0.2609 |
| controlled_vs_candidate_no_context | quest_state_correctness | 17 | 6 | 1 | 0.7292 | 0.7391 |
| controlled_vs_candidate_no_context | lore_consistency | 8 | 1 | 15 | 0.6458 | 0.8889 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_vs_candidate_no_context | objective_completion_support | 10 | 9 | 5 | 0.5208 | 0.5263 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 14 | 9 | 1 | 0.6042 | 0.6087 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 6 | 13 | 5 | 0.3542 | 0.3158 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 8 | 2 | 14 | 0.6250 | 0.8000 |
| controlled_vs_candidate_no_context | context_overlap | 17 | 6 | 1 | 0.7292 | 0.7391 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 5 | 1 | 18 | 0.5833 | 0.8333 |
| controlled_vs_candidate_no_context | persona_style | 5 | 0 | 19 | 0.6042 | 1.0000 |
| controlled_vs_candidate_no_context | distinct1 | 15 | 7 | 2 | 0.6667 | 0.6818 |
| controlled_vs_candidate_no_context | length_score | 5 | 17 | 2 | 0.2500 | 0.2273 |
| controlled_vs_candidate_no_context | sentence_score | 0 | 0 | 24 | 0.5000 | nan |
| controlled_vs_candidate_no_context | overall_quality | 14 | 9 | 1 | 0.6042 | 0.6087 |
| controlled_alt_vs_controlled_default | context_relevance | 10 | 7 | 7 | 0.5625 | 0.5882 |
| controlled_alt_vs_controlled_default | persona_consistency | 6 | 8 | 10 | 0.4583 | 0.4286 |
| controlled_alt_vs_controlled_default | naturalness | 8 | 9 | 7 | 0.4792 | 0.4706 |
| controlled_alt_vs_controlled_default | quest_state_correctness | 9 | 8 | 7 | 0.5208 | 0.5294 |
| controlled_alt_vs_controlled_default | lore_consistency | 9 | 7 | 8 | 0.5417 | 0.5625 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_alt_vs_controlled_default | objective_completion_support | 5 | 10 | 9 | 0.3958 | 0.3333 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | 9 | 8 | 7 | 0.5208 | 0.5294 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | 7 | 9 | 8 | 0.4583 | 0.4375 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 9 | 6 | 9 | 0.5625 | 0.6000 |
| controlled_alt_vs_controlled_default | context_overlap | 10 | 7 | 7 | 0.5625 | 0.5882 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 4 | 6 | 14 | 0.4583 | 0.4000 |
| controlled_alt_vs_controlled_default | persona_style | 2 | 4 | 18 | 0.4583 | 0.3333 |
| controlled_alt_vs_controlled_default | distinct1 | 7 | 9 | 8 | 0.4583 | 0.4375 |
| controlled_alt_vs_controlled_default | length_score | 11 | 6 | 7 | 0.6042 | 0.6471 |
| controlled_alt_vs_controlled_default | sentence_score | 1 | 1 | 22 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | overall_quality | 9 | 8 | 7 | 0.5208 | 0.5294 |
| controlled_alt_vs_proposed_raw | context_relevance | 17 | 6 | 1 | 0.7292 | 0.7391 |
| controlled_alt_vs_proposed_raw | persona_consistency | 10 | 1 | 13 | 0.6875 | 0.9091 |
| controlled_alt_vs_proposed_raw | naturalness | 9 | 14 | 1 | 0.3958 | 0.3913 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 17 | 6 | 1 | 0.7292 | 0.7391 |
| controlled_alt_vs_proposed_raw | lore_consistency | 10 | 2 | 12 | 0.6667 | 0.8333 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_alt_vs_proposed_raw | objective_completion_support | 5 | 10 | 9 | 0.3958 | 0.3333 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 12 | 11 | 1 | 0.5208 | 0.5217 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | 6 | 12 | 6 | 0.3750 | 0.3333 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 10 | 1 | 13 | 0.6875 | 0.9091 |
| controlled_alt_vs_proposed_raw | context_overlap | 16 | 7 | 1 | 0.6875 | 0.6957 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 7 | 1 | 16 | 0.6250 | 0.8750 |
| controlled_alt_vs_proposed_raw | persona_style | 5 | 1 | 18 | 0.5833 | 0.8333 |
| controlled_alt_vs_proposed_raw | distinct1 | 13 | 10 | 1 | 0.5625 | 0.5652 |
| controlled_alt_vs_proposed_raw | length_score | 9 | 13 | 2 | 0.4167 | 0.4091 |
| controlled_alt_vs_proposed_raw | sentence_score | 2 | 1 | 21 | 0.5208 | 0.6667 |
| controlled_alt_vs_proposed_raw | overall_quality | 16 | 7 | 1 | 0.6875 | 0.6957 |
| controlled_alt_vs_candidate_no_context | context_relevance | 16 | 7 | 1 | 0.6875 | 0.6957 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 8 | 2 | 14 | 0.6250 | 0.8000 |
| controlled_alt_vs_candidate_no_context | naturalness | 8 | 15 | 1 | 0.3542 | 0.3478 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 17 | 6 | 1 | 0.7292 | 0.7391 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 9 | 2 | 13 | 0.6458 | 0.8182 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_alt_vs_candidate_no_context | objective_completion_support | 5 | 12 | 7 | 0.3542 | 0.2941 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 12 | 11 | 1 | 0.5208 | 0.5217 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | 5 | 14 | 5 | 0.3125 | 0.2632 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 9 | 0 | 15 | 0.6875 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 16 | 7 | 1 | 0.6875 | 0.6957 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 5 | 2 | 17 | 0.5625 | 0.7143 |
| controlled_alt_vs_candidate_no_context | persona_style | 5 | 1 | 18 | 0.5833 | 0.8333 |
| controlled_alt_vs_candidate_no_context | distinct1 | 14 | 9 | 1 | 0.6042 | 0.6087 |
| controlled_alt_vs_candidate_no_context | length_score | 9 | 13 | 2 | 0.4167 | 0.4091 |
| controlled_alt_vs_candidate_no_context | sentence_score | 1 | 1 | 22 | 0.5000 | 0.5000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 15 | 8 | 1 | 0.6458 | 0.6522 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.2083 | 0.0833 | 0.9167 |
| proposed_contextual_controlled_tuned | 0.0000 | 0.0000 | 0.0833 | 0.1667 | 0.8333 |
| proposed_contextual | 0.0000 | 0.0000 | 0.6667 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.7083 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `3`
- Unique template signatures: `18`
- Template signature ratio: `0.7500`
- Effective sample size by source clustering: `2.88`
- Effective sample size by template-signature clustering: `16.00`
- Detailed diagnostics are published in `scenario_dependence.json`.

## Multi-turn Contradiction
| Arm | Contradiction Rate | Contradiction Safety | Contradicted Sources | Source Count |
|---|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 1.0000 | 0 | 3 |
| proposed_contextual_controlled_tuned | 0.0000 | 1.0000 | 0 | 3 |
| proposed_contextual | 0.0000 | 1.0000 | 0 | 3 |
| candidate_no_context | 0.0000 | 1.0000 | 0 | 3 |
- Detailed source-level values are published in `multi_turn_contradictions.json`.

- BERTScore status: unavailable (disabled_by_flag).

This report covers proposal RO5 metrics plus game-facing outcomes: quest-state correctness, lore consistency, contradiction safety, objective completion support, gameplay usefulness, and time-pressure acceptability.
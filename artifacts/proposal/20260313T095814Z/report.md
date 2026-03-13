# Proposal Alignment Evaluation Report

- Run ID: `20260313T095814Z`
- Generated: `2026-03-13T10:02:49.412793+00:00`
- Scenarios: `artifacts\proposal\20260313T095814Z\scenarios.jsonl`
- Scenario count: `144`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_tuned`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off
- `baseline_no_context`: model `phi3:mini`, dynamic_context=off, response_control=off
- `baseline_no_context_phi3_latest`: model `phi3:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.1266 (0.1048, 0.1503) | 0.3062 (0.2796, 0.3333) | 0.8707 (0.8625, 0.8787) | 0.3384 (0.3229, 0.3535) | n/a |
| proposed_contextual_controlled_tuned | 0.1126 (0.0898, 0.1359) | 0.2951 (0.2687, 0.3234) | 0.8631 (0.8551, 0.8713) | 0.3263 (0.3103, 0.3424) | n/a |
| proposed_contextual | 0.0865 (0.0696, 0.1028) | 0.2091 (0.1881, 0.2311) | 0.8727 (0.8656, 0.8799) | 0.2841 (0.2721, 0.2967) | n/a |
| candidate_no_context | 0.0314 (0.0251, 0.0384) | 0.2204 (0.1971, 0.2444) | 0.8743 (0.8680, 0.8810) | 0.2631 (0.2538, 0.2734) | n/a |
| baseline_no_context | 0.0363 (0.0290, 0.0442) | 0.1651 (0.1475, 0.1824) | 0.8915 (0.8836, 0.8997) | 0.2477 (0.2397, 0.2560) | n/a |
| baseline_no_context_phi3_latest | 0.0366 (0.0293, 0.0439) | 0.1548 (0.1400, 0.1714) | 0.8901 (0.8810, 0.8984) | 0.2437 (0.2358, 0.2522) | n/a |

## Game-facing Outcome Metrics (mean, 95% CI)
| Arm | Quest-state Correctness | Lore Consistency | Contradiction Safety | Objective Completion Support | Gameplay Usefulness | Time-pressure Acceptability |
|---|---:|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2080 (0.1892, 0.2285) | 0.0558 (0.0402, 0.0727) | 1.0000 (1.0000, 1.0000) | 0.0869 (0.0747, 0.0995) | 0.3109 (0.3042, 0.3179) | 0.2980 (0.2893, 0.3070) |
| proposed_contextual_controlled_tuned | 0.1982 (0.1776, 0.2187) | 0.0557 (0.0381, 0.0740) | 1.0000 (1.0000, 1.0000) | 0.0879 (0.0747, 0.1013) | 0.3039 (0.2967, 0.3119) | 0.2974 (0.2874, 0.3070) |
| proposed_contextual | 0.1744 (0.1614, 0.1887) | 0.0424 (0.0315, 0.0544) | 1.0000 (1.0000, 1.0000) | 0.0680 (0.0565, 0.0799) | 0.2944 (0.2873, 0.3018) | 0.2937 (0.2859, 0.3019) |
| candidate_no_context | 0.1272 (0.1216, 0.1335) | 0.0064 (0.0039, 0.0093) | 0.8750 (0.8193, 0.9236) | 0.0728 (0.0604, 0.0858) | 0.2817 (0.2756, 0.2883) | 0.2994 (0.2914, 0.3075) |
| baseline_no_context | 0.1302 (0.1240, 0.1367) | 0.0152 (0.0112, 0.0197) | 1.0000 (1.0000, 1.0000) | 0.0413 (0.0352, 0.0478) | 0.2764 (0.2714, 0.2815) | 0.2863 (0.2810, 0.2919) |
| baseline_no_context_phi3_latest | 0.1314 (0.1251, 0.1381) | 0.0155 (0.0113, 0.0199) | 1.0000 (1.0000, 1.0000) | 0.0385 (0.0319, 0.0456) | 0.2751 (0.2692, 0.2811) | 0.2863 (0.2804, 0.2924) |

- Multi-turn contradiction rate is reported as `1 - contradiction_safety` in row-level outputs.

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0551 | 1.7552 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0113 | -0.0511 |
| proposed_vs_candidate_no_context | naturalness | -0.0016 | -0.0018 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.0472 | 0.3712 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0360 | 5.6644 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0.1250 | 0.1429 |
| proposed_vs_candidate_no_context | objective_completion_support | -0.0048 | -0.0662 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 0.0128 | 0.0454 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | -0.0057 | -0.0191 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0717 | 2.3802 |
| proposed_vs_candidate_no_context | context_overlap | 0.0164 | 0.4759 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0150 | -0.1071 |
| proposed_vs_candidate_no_context | persona_style | 0.0038 | 0.0071 |
| proposed_vs_candidate_no_context | distinct1 | 0.0068 | 0.0073 |
| proposed_vs_candidate_no_context | length_score | -0.0028 | -0.0054 |
| proposed_vs_candidate_no_context | sentence_score | -0.0385 | -0.0404 |
| proposed_vs_candidate_no_context | overall_quality | 0.0210 | 0.0798 |
| proposed_vs_baseline_no_context | context_relevance | 0.0503 | 1.3860 |
| proposed_vs_baseline_no_context | persona_consistency | 0.0440 | 0.2665 |
| proposed_vs_baseline_no_context | naturalness | -0.0188 | -0.0211 |
| proposed_vs_baseline_no_context | quest_state_correctness | 0.0442 | 0.3391 |
| proposed_vs_baseline_no_context | lore_consistency | 0.0272 | 1.7946 |
| proposed_vs_baseline_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| proposed_vs_baseline_no_context | objective_completion_support | 0.0266 | 0.6437 |
| proposed_vs_baseline_no_context | gameplay_usefulness | 0.0181 | 0.0653 |
| proposed_vs_baseline_no_context | time_pressure_acceptability | 0.0073 | 0.0256 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0660 | 1.8440 |
| proposed_vs_baseline_no_context | context_overlap | 0.0134 | 0.3598 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 0.0592 | 0.8928 |
| proposed_vs_baseline_no_context | persona_style | -0.0167 | -0.0299 |
| proposed_vs_baseline_no_context | distinct1 | -0.0344 | -0.0351 |
| proposed_vs_baseline_no_context | length_score | -0.0377 | -0.0682 |
| proposed_vs_baseline_no_context | sentence_score | 0.0247 | 0.0277 |
| proposed_vs_baseline_no_context | overall_quality | 0.0364 | 0.1468 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 0.0500 | 1.3668 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | 0.0543 | 0.3510 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | -0.0173 | -0.0195 |
| proposed_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.0430 | 0.3275 |
| proposed_vs_baseline_no_context_phi3_latest | lore_consistency | 0.0269 | 1.7359 |
| proposed_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.0295 | 0.7650 |
| proposed_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 0.0193 | 0.0703 |
| proposed_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 0.0073 | 0.0256 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.0659 | 1.8340 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 0.0127 | 0.3352 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.0714 | 1.3183 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | -0.0138 | -0.0248 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | -0.0339 | -0.0347 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | -0.0373 | -0.0674 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | 0.0368 | 0.0419 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | 0.0404 | 0.1656 |
| controlled_vs_proposed_raw | context_relevance | 0.0401 | 0.4636 |
| controlled_vs_proposed_raw | persona_consistency | 0.0971 | 0.4643 |
| controlled_vs_proposed_raw | naturalness | -0.0020 | -0.0023 |
| controlled_vs_proposed_raw | quest_state_correctness | 0.0336 | 0.1927 |
| controlled_vs_proposed_raw | lore_consistency | 0.0134 | 0.3159 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0190 | 0.2791 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0165 | 0.0559 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 0.0043 | 0.0148 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.0511 | 0.5021 |
| controlled_vs_proposed_raw | context_overlap | 0.0143 | 0.2828 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1182 | 0.9418 |
| controlled_vs_proposed_raw | persona_style | 0.0126 | 0.0232 |
| controlled_vs_proposed_raw | distinct1 | -0.0076 | -0.0081 |
| controlled_vs_proposed_raw | length_score | -0.0162 | -0.0314 |
| controlled_vs_proposed_raw | sentence_score | 0.0462 | 0.0505 |
| controlled_vs_proposed_raw | overall_quality | 0.0543 | 0.1913 |
| controlled_vs_candidate_no_context | context_relevance | 0.0952 | 3.0323 |
| controlled_vs_candidate_no_context | persona_consistency | 0.0858 | 0.3894 |
| controlled_vs_candidate_no_context | naturalness | -0.0036 | -0.0041 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.0808 | 0.6354 |
| controlled_vs_candidate_no_context | lore_consistency | 0.0494 | 7.7694 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0.1250 | 0.1429 |
| controlled_vs_candidate_no_context | objective_completion_support | 0.0142 | 0.1945 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0292 | 0.1038 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | -0.0014 | -0.0046 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.1229 | 4.0775 |
| controlled_vs_candidate_no_context | context_overlap | 0.0307 | 0.8934 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1031 | 0.7339 |
| controlled_vs_candidate_no_context | persona_style | 0.0165 | 0.0305 |
| controlled_vs_candidate_no_context | distinct1 | -0.0008 | -0.0009 |
| controlled_vs_candidate_no_context | length_score | -0.0190 | -0.0366 |
| controlled_vs_candidate_no_context | sentence_score | 0.0076 | 0.0080 |
| controlled_vs_candidate_no_context | overall_quality | 0.0753 | 0.2863 |
| controlled_vs_baseline_no_context | context_relevance | 0.0904 | 2.4921 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1411 | 0.8546 |
| controlled_vs_baseline_no_context | naturalness | -0.0208 | -0.0234 |
| controlled_vs_baseline_no_context | quest_state_correctness | 0.0778 | 0.5972 |
| controlled_vs_baseline_no_context | lore_consistency | 0.0406 | 2.6774 |
| controlled_vs_baseline_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_baseline_no_context | objective_completion_support | 0.0456 | 1.1024 |
| controlled_vs_baseline_no_context | gameplay_usefulness | 0.0345 | 0.1249 |
| controlled_vs_baseline_no_context | time_pressure_acceptability | 0.0117 | 0.0407 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.1172 | 3.2720 |
| controlled_vs_baseline_no_context | context_overlap | 0.0278 | 0.7444 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1774 | 2.6753 |
| controlled_vs_baseline_no_context | persona_style | -0.0041 | -0.0073 |
| controlled_vs_baseline_no_context | distinct1 | -0.0421 | -0.0429 |
| controlled_vs_baseline_no_context | length_score | -0.0539 | -0.0975 |
| controlled_vs_baseline_no_context | sentence_score | 0.0708 | 0.0795 |
| controlled_vs_baseline_no_context | overall_quality | 0.0907 | 0.3661 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.0901 | 2.4640 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1514 | 0.9782 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | -0.0193 | -0.0217 |
| controlled_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.0766 | 0.5833 |
| controlled_vs_baseline_no_context_phi3_latest | lore_consistency | 0.0403 | 2.6000 |
| controlled_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.0484 | 1.2576 |
| controlled_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 0.0358 | 0.1301 |
| controlled_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 0.0117 | 0.0408 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.1171 | 3.2570 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0271 | 0.7129 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1896 | 3.5015 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | -0.0012 | -0.0021 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0416 | -0.0425 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | -0.0535 | -0.0967 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.0830 | 0.0945 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.0947 | 0.3886 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0140 | -0.1104 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0111 | -0.0363 |
| controlled_alt_vs_controlled_default | naturalness | -0.0076 | -0.0088 |
| controlled_alt_vs_controlled_default | quest_state_correctness | -0.0098 | -0.0469 |
| controlled_alt_vs_controlled_default | lore_consistency | -0.0001 | -0.0017 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_controlled_default | objective_completion_support | 0.0010 | 0.0115 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | -0.0070 | -0.0226 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | -0.0006 | -0.0021 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0184 | -0.1205 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0036 | -0.0553 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0132 | -0.0543 |
| controlled_alt_vs_controlled_default | persona_style | -0.0027 | -0.0048 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0057 | -0.0061 |
| controlled_alt_vs_controlled_default | length_score | -0.0347 | -0.0695 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0191 | 0.0199 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0122 | -0.0359 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.0261 | 0.3019 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.0860 | 0.4111 |
| controlled_alt_vs_proposed_raw | naturalness | -0.0096 | -0.0110 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 0.0238 | 0.1367 |
| controlled_alt_vs_proposed_raw | lore_consistency | 0.0133 | 0.3136 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_proposed_raw | objective_completion_support | 0.0200 | 0.2939 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 0.0094 | 0.0320 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | 0.0037 | 0.0126 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.0327 | 0.3211 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0107 | 0.2120 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1050 | 0.8364 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0100 | 0.0183 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0133 | -0.0141 |
| controlled_alt_vs_proposed_raw | length_score | -0.0509 | -0.0988 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0653 | 0.0713 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.0422 | 0.1485 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.0812 | 2.5871 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.0747 | 0.3389 |
| controlled_alt_vs_candidate_no_context | naturalness | -0.0112 | -0.0128 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 0.0711 | 0.5587 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 0.0493 | 7.7546 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0.1250 | 0.1429 |
| controlled_alt_vs_candidate_no_context | objective_completion_support | 0.0152 | 0.2083 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 0.0222 | 0.0789 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | -0.0020 | -0.0068 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.1044 | 3.4658 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0271 | 0.7888 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.0899 | 0.6398 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0138 | 0.0256 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0065 | -0.0070 |
| controlled_alt_vs_candidate_no_context | length_score | -0.0537 | -0.1036 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.0267 | 0.0280 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.0632 | 0.2401 |
| controlled_alt_vs_baseline_no_context | context_relevance | 0.0764 | 2.1065 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 0.1300 | 0.7872 |
| controlled_alt_vs_baseline_no_context | naturalness | -0.0285 | -0.0319 |
| controlled_alt_vs_baseline_no_context | quest_state_correctness | 0.0680 | 0.5222 |
| controlled_alt_vs_baseline_no_context | lore_consistency | 0.0405 | 2.6711 |
| controlled_alt_vs_baseline_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_baseline_no_context | objective_completion_support | 0.0466 | 1.1267 |
| controlled_alt_vs_baseline_no_context | gameplay_usefulness | 0.0275 | 0.0995 |
| controlled_alt_vs_baseline_no_context | time_pressure_acceptability | 0.0110 | 0.0385 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 0.0988 | 2.7573 |
| controlled_alt_vs_baseline_no_context | context_overlap | 0.0242 | 0.6480 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 0.1642 | 2.4758 |
| controlled_alt_vs_baseline_no_context | persona_style | -0.0068 | -0.0121 |
| controlled_alt_vs_baseline_no_context | distinct1 | -0.0477 | -0.0487 |
| controlled_alt_vs_baseline_no_context | length_score | -0.0887 | -0.1603 |
| controlled_alt_vs_baseline_no_context | sentence_score | 0.0899 | 0.1010 |
| controlled_alt_vs_baseline_no_context | overall_quality | 0.0785 | 0.3170 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_relevance | 0.0761 | 2.0815 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1403 | 0.9064 |
| controlled_alt_vs_baseline_no_context_phi3_latest | naturalness | -0.0270 | -0.0303 |
| controlled_alt_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.0669 | 0.5090 |
| controlled_alt_vs_baseline_no_context_phi3_latest | lore_consistency | 0.0402 | 2.5939 |
| controlled_alt_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.0494 | 1.2836 |
| controlled_alt_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 0.0288 | 0.1046 |
| controlled_alt_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 0.0110 | 0.0385 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.0986 | 2.7441 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_overlap | 0.0235 | 0.6182 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1763 | 3.2572 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_style | -0.0039 | -0.0069 |
| controlled_alt_vs_baseline_no_context_phi3_latest | distinct1 | -0.0473 | -0.0483 |
| controlled_alt_vs_baseline_no_context_phi3_latest | length_score | -0.0882 | -0.1595 |
| controlled_alt_vs_baseline_no_context_phi3_latest | sentence_score | 0.1021 | 0.1162 |
| controlled_alt_vs_baseline_no_context_phi3_latest | overall_quality | 0.0826 | 0.3387 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | context_relevance | 0.0764 | 2.1065 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | persona_consistency | 0.1300 | 0.7872 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | naturalness | -0.0285 | -0.0319 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | quest_state_correctness | 0.0680 | 0.5222 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | lore_consistency | 0.0405 | 2.6711 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | objective_completion_support | 0.0466 | 1.1267 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | gameplay_usefulness | 0.0275 | 0.0995 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | time_pressure_acceptability | 0.0110 | 0.0385 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | context_keyword_coverage | 0.0988 | 2.7573 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | context_overlap | 0.0242 | 0.6480 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | persona_keyword_coverage | 0.1642 | 2.4758 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | persona_style | -0.0068 | -0.0121 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | distinct1 | -0.0477 | -0.0487 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | length_score | -0.0887 | -0.1603 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | sentence_score | 0.0899 | 0.1010 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | overall_quality | 0.0785 | 0.3170 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | context_relevance | 0.0761 | 2.0815 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1403 | 0.9064 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | naturalness | -0.0270 | -0.0303 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.0669 | 0.5090 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | lore_consistency | 0.0402 | 2.5939 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.0494 | 1.2836 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 0.0288 | 0.1046 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 0.0110 | 0.0385 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.0986 | 2.7441 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | context_overlap | 0.0235 | 0.6182 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1763 | 3.2572 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | persona_style | -0.0039 | -0.0069 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | distinct1 | -0.0473 | -0.0483 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | length_score | -0.0882 | -0.1595 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | sentence_score | 0.1021 | 0.1162 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | overall_quality | 0.0826 | 0.3387 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0551 | (0.0388, 0.0711) | 0.0000 | 0.0551 | (0.0317, 0.0822) | 0.0000 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0113 | (-0.0313, 0.0081) | 0.8723 | -0.0113 | (-0.0200, -0.0014) | 0.9897 |
| proposed_vs_candidate_no_context | naturalness | -0.0016 | (-0.0102, 0.0070) | 0.6400 | -0.0016 | (-0.0090, 0.0068) | 0.6673 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.0472 | (0.0337, 0.0614) | 0.0000 | 0.0472 | (0.0260, 0.0691) | 0.0000 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0360 | (0.0256, 0.0481) | 0.0000 | 0.0360 | (0.0183, 0.0583) | 0.0000 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0.1250 | (0.0764, 0.1806) | 0.0000 | 0.1250 | (0.0000, 0.3750) | 0.3577 |
| proposed_vs_candidate_no_context | objective_completion_support | -0.0048 | (-0.0156, 0.0052) | 0.8257 | -0.0048 | (-0.0155, 0.0056) | 0.8183 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 0.0128 | (0.0046, 0.0213) | 0.0007 | 0.0128 | (0.0043, 0.0206) | 0.0027 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | -0.0057 | (-0.0126, 0.0015) | 0.9377 | -0.0057 | (-0.0129, 0.0007) | 0.9543 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0717 | (0.0509, 0.0942) | 0.0000 | 0.0717 | (0.0411, 0.1058) | 0.0000 |
| proposed_vs_candidate_no_context | context_overlap | 0.0164 | (0.0103, 0.0228) | 0.0000 | 0.0164 | (0.0085, 0.0239) | 0.0000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0150 | (-0.0396, 0.0091) | 0.8763 | -0.0150 | (-0.0285, -0.0019) | 0.9877 |
| proposed_vs_candidate_no_context | persona_style | 0.0038 | (-0.0104, 0.0188) | 0.3140 | 0.0038 | (-0.0039, 0.0159) | 0.3153 |
| proposed_vs_candidate_no_context | distinct1 | 0.0068 | (0.0005, 0.0135) | 0.0160 | 0.0068 | (0.0024, 0.0114) | 0.0010 |
| proposed_vs_candidate_no_context | length_score | -0.0028 | (-0.0417, 0.0375) | 0.5740 | -0.0028 | (-0.0345, 0.0292) | 0.5810 |
| proposed_vs_candidate_no_context | sentence_score | -0.0385 | (-0.0656, -0.0118) | 0.9990 | -0.0385 | (-0.0556, -0.0188) | 1.0000 |
| proposed_vs_candidate_no_context | overall_quality | 0.0210 | (0.0102, 0.0321) | 0.0000 | 0.0210 | (0.0103, 0.0336) | 0.0000 |
| proposed_vs_baseline_no_context | context_relevance | 0.0503 | (0.0337, 0.0674) | 0.0000 | 0.0503 | (0.0242, 0.0755) | 0.0000 |
| proposed_vs_baseline_no_context | persona_consistency | 0.0440 | (0.0213, 0.0673) | 0.0000 | 0.0440 | (0.0125, 0.0777) | 0.0070 |
| proposed_vs_baseline_no_context | naturalness | -0.0188 | (-0.0282, -0.0095) | 1.0000 | -0.0188 | (-0.0330, -0.0071) | 1.0000 |
| proposed_vs_baseline_no_context | quest_state_correctness | 0.0442 | (0.0303, 0.0593) | 0.0000 | 0.0442 | (0.0228, 0.0645) | 0.0000 |
| proposed_vs_baseline_no_context | lore_consistency | 0.0272 | (0.0157, 0.0391) | 0.0000 | 0.0272 | (0.0104, 0.0469) | 0.0000 |
| proposed_vs_baseline_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_baseline_no_context | objective_completion_support | 0.0266 | (0.0159, 0.0384) | 0.0000 | 0.0266 | (0.0085, 0.0511) | 0.0000 |
| proposed_vs_baseline_no_context | gameplay_usefulness | 0.0181 | (0.0097, 0.0263) | 0.0000 | 0.0181 | (0.0040, 0.0309) | 0.0057 |
| proposed_vs_baseline_no_context | time_pressure_acceptability | 0.0073 | (-0.0002, 0.0153) | 0.0277 | 0.0073 | (-0.0071, 0.0244) | 0.1957 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0660 | (0.0450, 0.0892) | 0.0000 | 0.0660 | (0.0351, 0.0971) | 0.0000 |
| proposed_vs_baseline_no_context | context_overlap | 0.0134 | (0.0067, 0.0207) | 0.0000 | 0.0134 | (-0.0015, 0.0245) | 0.0340 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 0.0592 | (0.0287, 0.0878) | 0.0000 | 0.0592 | (0.0157, 0.0993) | 0.0057 |
| proposed_vs_baseline_no_context | persona_style | -0.0167 | (-0.0323, -0.0020) | 0.9870 | -0.0167 | (-0.0507, 0.0107) | 0.8557 |
| proposed_vs_baseline_no_context | distinct1 | -0.0344 | (-0.0443, -0.0250) | 1.0000 | -0.0344 | (-0.0573, -0.0186) | 1.0000 |
| proposed_vs_baseline_no_context | length_score | -0.0377 | (-0.0815, 0.0037) | 0.9640 | -0.0377 | (-0.1074, 0.0178) | 0.8917 |
| proposed_vs_baseline_no_context | sentence_score | 0.0247 | (-0.0118, 0.0583) | 0.0867 | 0.0247 | (-0.0264, 0.0753) | 0.1850 |
| proposed_vs_baseline_no_context | overall_quality | 0.0364 | (0.0238, 0.0492) | 0.0000 | 0.0364 | (0.0141, 0.0570) | 0.0003 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 0.0500 | (0.0323, 0.0677) | 0.0000 | 0.0500 | (0.0245, 0.0736) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | 0.0543 | (0.0350, 0.0734) | 0.0000 | 0.0543 | (0.0250, 0.0822) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | -0.0173 | (-0.0280, -0.0073) | 1.0000 | -0.0173 | (-0.0318, -0.0044) | 0.9997 |
| proposed_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.0430 | (0.0292, 0.0580) | 0.0000 | 0.0430 | (0.0230, 0.0624) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | lore_consistency | 0.0269 | (0.0159, 0.0383) | 0.0000 | 0.0269 | (0.0102, 0.0479) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.0295 | (0.0183, 0.0413) | 0.0000 | 0.0295 | (0.0136, 0.0493) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 0.0193 | (0.0107, 0.0279) | 0.0000 | 0.0193 | (0.0089, 0.0299) | 0.0003 |
| proposed_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 0.0073 | (-0.0003, 0.0153) | 0.0283 | 0.0073 | (-0.0038, 0.0206) | 0.1280 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.0659 | (0.0449, 0.0885) | 0.0000 | 0.0659 | (0.0367, 0.0944) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 0.0127 | (0.0058, 0.0201) | 0.0003 | 0.0127 | (-0.0005, 0.0242) | 0.0293 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.0714 | (0.0474, 0.0952) | 0.0000 | 0.0714 | (0.0345, 0.1048) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | -0.0138 | (-0.0330, 0.0059) | 0.9177 | -0.0138 | (-0.0580, 0.0216) | 0.7337 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | -0.0339 | (-0.0433, -0.0246) | 1.0000 | -0.0339 | (-0.0541, -0.0176) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | -0.0373 | (-0.0843, 0.0079) | 0.9403 | -0.0373 | (-0.0870, 0.0123) | 0.9257 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | 0.0368 | (0.0031, 0.0705) | 0.0157 | 0.0368 | (-0.0038, 0.0826) | 0.0450 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | 0.0404 | (0.0285, 0.0517) | 0.0000 | 0.0404 | (0.0208, 0.0583) | 0.0000 |
| controlled_vs_proposed_raw | context_relevance | 0.0401 | (0.0160, 0.0650) | 0.0003 | 0.0401 | (0.0093, 0.0677) | 0.0037 |
| controlled_vs_proposed_raw | persona_consistency | 0.0971 | (0.0699, 0.1258) | 0.0000 | 0.0971 | (0.0617, 0.1376) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | -0.0020 | (-0.0124, 0.0087) | 0.6500 | -0.0020 | (-0.0157, 0.0123) | 0.5987 |
| controlled_vs_proposed_raw | quest_state_correctness | 0.0336 | (0.0129, 0.0557) | 0.0007 | 0.0336 | (0.0077, 0.0575) | 0.0057 |
| controlled_vs_proposed_raw | lore_consistency | 0.0134 | (-0.0028, 0.0314) | 0.0560 | 0.0134 | (-0.0093, 0.0351) | 0.1330 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0190 | (0.0075, 0.0301) | 0.0003 | 0.0190 | (0.0043, 0.0369) | 0.0047 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0165 | (0.0077, 0.0250) | 0.0000 | 0.0165 | (0.0089, 0.0240) | 0.0000 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 0.0043 | (-0.0030, 0.0121) | 0.1380 | 0.0043 | (-0.0041, 0.0138) | 0.1680 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.0511 | (0.0196, 0.0853) | 0.0010 | 0.0511 | (0.0148, 0.0857) | 0.0030 |
| controlled_vs_proposed_raw | context_overlap | 0.0143 | (0.0036, 0.0253) | 0.0017 | 0.0143 | (0.0021, 0.0270) | 0.0090 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1182 | (0.0841, 0.1531) | 0.0000 | 0.1182 | (0.0752, 0.1667) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.0126 | (0.0000, 0.0268) | 0.0250 | 0.0126 | (0.0002, 0.0298) | 0.0250 |
| controlled_vs_proposed_raw | distinct1 | -0.0076 | (-0.0154, 0.0002) | 0.9717 | -0.0076 | (-0.0141, -0.0001) | 0.9767 |
| controlled_vs_proposed_raw | length_score | -0.0162 | (-0.0620, 0.0278) | 0.7567 | -0.0162 | (-0.0931, 0.0572) | 0.6520 |
| controlled_vs_proposed_raw | sentence_score | 0.0462 | (0.0149, 0.0771) | 0.0013 | 0.0462 | (0.0149, 0.0816) | 0.0010 |
| controlled_vs_proposed_raw | overall_quality | 0.0543 | (0.0383, 0.0706) | 0.0000 | 0.0543 | (0.0344, 0.0702) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.0952 | (0.0736, 0.1191) | 0.0000 | 0.0952 | (0.0643, 0.1295) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.0858 | (0.0564, 0.1165) | 0.0000 | 0.0858 | (0.0491, 0.1269) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | -0.0036 | (-0.0142, 0.0064) | 0.7463 | -0.0036 | (-0.0187, 0.0128) | 0.6830 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.0808 | (0.0622, 0.1004) | 0.0000 | 0.0808 | (0.0542, 0.1082) | 0.0000 |
| controlled_vs_candidate_no_context | lore_consistency | 0.0494 | (0.0331, 0.0673) | 0.0000 | 0.0494 | (0.0267, 0.0710) | 0.0000 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0.1250 | (0.0694, 0.1806) | 0.0000 | 0.1250 | (0.0000, 0.3750) | 0.3440 |
| controlled_vs_candidate_no_context | objective_completion_support | 0.0142 | (0.0033, 0.0253) | 0.0077 | 0.0142 | (0.0016, 0.0279) | 0.0117 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0292 | (0.0217, 0.0366) | 0.0000 | 0.0292 | (0.0170, 0.0413) | 0.0000 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | -0.0014 | (-0.0106, 0.0076) | 0.6163 | -0.0014 | (-0.0104, 0.0072) | 0.6263 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.1229 | (0.0951, 0.1532) | 0.0000 | 0.1229 | (0.0824, 0.1656) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0307 | (0.0216, 0.0408) | 0.0000 | 0.0307 | (0.0206, 0.0426) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1031 | (0.0666, 0.1407) | 0.0000 | 0.1031 | (0.0583, 0.1493) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.0165 | (0.0004, 0.0336) | 0.0207 | 0.0165 | (0.0000, 0.0381) | 0.0210 |
| controlled_vs_candidate_no_context | distinct1 | -0.0008 | (-0.0092, 0.0078) | 0.5633 | -0.0008 | (-0.0070, 0.0056) | 0.6013 |
| controlled_vs_candidate_no_context | length_score | -0.0190 | (-0.0650, 0.0262) | 0.7933 | -0.0190 | (-0.0958, 0.0607) | 0.6830 |
| controlled_vs_candidate_no_context | sentence_score | 0.0076 | (-0.0215, 0.0365) | 0.2933 | 0.0076 | (-0.0361, 0.0583) | 0.3847 |
| controlled_vs_candidate_no_context | overall_quality | 0.0753 | (0.0588, 0.0916) | 0.0000 | 0.0753 | (0.0520, 0.0957) | 0.0000 |
| controlled_vs_baseline_no_context | context_relevance | 0.0904 | (0.0671, 0.1143) | 0.0000 | 0.0904 | (0.0602, 0.1225) | 0.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1411 | (0.1153, 0.1695) | 0.0000 | 0.1411 | (0.1070, 0.1768) | 0.0000 |
| controlled_vs_baseline_no_context | naturalness | -0.0208 | (-0.0324, -0.0096) | 0.9997 | -0.0208 | (-0.0329, -0.0093) | 1.0000 |
| controlled_vs_baseline_no_context | quest_state_correctness | 0.0778 | (0.0579, 0.0982) | 0.0000 | 0.0778 | (0.0521, 0.1039) | 0.0000 |
| controlled_vs_baseline_no_context | lore_consistency | 0.0406 | (0.0237, 0.0589) | 0.0000 | 0.0406 | (0.0192, 0.0622) | 0.0000 |
| controlled_vs_baseline_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_baseline_no_context | objective_completion_support | 0.0456 | (0.0317, 0.0589) | 0.0000 | 0.0456 | (0.0210, 0.0716) | 0.0000 |
| controlled_vs_baseline_no_context | gameplay_usefulness | 0.0345 | (0.0255, 0.0431) | 0.0000 | 0.0345 | (0.0208, 0.0476) | 0.0000 |
| controlled_vs_baseline_no_context | time_pressure_acceptability | 0.0117 | (0.0022, 0.0206) | 0.0080 | 0.0117 | (-0.0028, 0.0271) | 0.0577 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.1172 | (0.0878, 0.1475) | 0.0000 | 0.1172 | (0.0796, 0.1570) | 0.0000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0278 | (0.0185, 0.0377) | 0.0000 | 0.0278 | (0.0165, 0.0415) | 0.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1774 | (0.1431, 0.2112) | 0.0000 | 0.1774 | (0.1346, 0.2205) | 0.0000 |
| controlled_vs_baseline_no_context | persona_style | -0.0041 | (-0.0199, 0.0110) | 0.6923 | -0.0041 | (-0.0277, 0.0189) | 0.6137 |
| controlled_vs_baseline_no_context | distinct1 | -0.0421 | (-0.0508, -0.0331) | 1.0000 | -0.0421 | (-0.0606, -0.0284) | 1.0000 |
| controlled_vs_baseline_no_context | length_score | -0.0539 | (-0.1042, -0.0042) | 0.9850 | -0.0539 | (-0.1352, 0.0144) | 0.9290 |
| controlled_vs_baseline_no_context | sentence_score | 0.0708 | (0.0392, 0.1028) | 0.0000 | 0.0708 | (0.0292, 0.1101) | 0.0010 |
| controlled_vs_baseline_no_context | overall_quality | 0.0907 | (0.0741, 0.1068) | 0.0000 | 0.0907 | (0.0705, 0.1091) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.0901 | (0.0676, 0.1135) | 0.0000 | 0.0901 | (0.0630, 0.1212) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1514 | (0.1262, 0.1777) | 0.0000 | 0.1514 | (0.1205, 0.1868) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | -0.0193 | (-0.0325, -0.0070) | 0.9983 | -0.0193 | (-0.0325, -0.0068) | 0.9983 |
| controlled_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.0766 | (0.0567, 0.0971) | 0.0000 | 0.0766 | (0.0525, 0.1018) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | lore_consistency | 0.0403 | (0.0249, 0.0571) | 0.0000 | 0.0403 | (0.0208, 0.0592) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.0484 | (0.0361, 0.0604) | 0.0000 | 0.0484 | (0.0280, 0.0687) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 0.0358 | (0.0272, 0.0445) | 0.0000 | 0.0358 | (0.0245, 0.0473) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 0.0117 | (0.0020, 0.0213) | 0.0090 | 0.0117 | (0.0001, 0.0236) | 0.0247 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.1171 | (0.0877, 0.1457) | 0.0000 | 0.1171 | (0.0819, 0.1555) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0271 | (0.0169, 0.0369) | 0.0000 | 0.0271 | (0.0146, 0.0418) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1896 | (0.1594, 0.2207) | 0.0000 | 0.1896 | (0.1530, 0.2287) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | -0.0012 | (-0.0214, 0.0189) | 0.5500 | -0.0012 | (-0.0330, 0.0313) | 0.5230 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0416 | (-0.0501, -0.0331) | 1.0000 | -0.0416 | (-0.0572, -0.0291) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | -0.0535 | (-0.1056, -0.0016) | 0.9773 | -0.0535 | (-0.1262, 0.0104) | 0.9480 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.0830 | (0.0514, 0.1146) | 0.0000 | 0.0830 | (0.0365, 0.1288) | 0.0007 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.0947 | (0.0783, 0.1114) | 0.0000 | 0.0947 | (0.0762, 0.1118) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0140 | (-0.0425, 0.0157) | 0.8320 | -0.0140 | (-0.0350, 0.0085) | 0.8813 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0111 | (-0.0429, 0.0199) | 0.7617 | -0.0111 | (-0.0267, 0.0051) | 0.9087 |
| controlled_alt_vs_controlled_default | naturalness | -0.0076 | (-0.0185, 0.0031) | 0.9267 | -0.0076 | (-0.0218, 0.0048) | 0.8693 |
| controlled_alt_vs_controlled_default | quest_state_correctness | -0.0098 | (-0.0354, 0.0145) | 0.7873 | -0.0098 | (-0.0292, 0.0110) | 0.8147 |
| controlled_alt_vs_controlled_default | lore_consistency | -0.0001 | (-0.0215, 0.0224) | 0.4870 | -0.0001 | (-0.0135, 0.0143) | 0.5197 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | objective_completion_support | 0.0010 | (-0.0093, 0.0120) | 0.4223 | 0.0010 | (-0.0097, 0.0130) | 0.4363 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | -0.0070 | (-0.0149, 0.0008) | 0.9587 | -0.0070 | (-0.0155, 0.0017) | 0.9460 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | -0.0006 | (-0.0095, 0.0084) | 0.5500 | -0.0006 | (-0.0067, 0.0057) | 0.5897 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0184 | (-0.0549, 0.0170) | 0.8430 | -0.0184 | (-0.0448, 0.0093) | 0.9030 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0036 | (-0.0168, 0.0112) | 0.7037 | -0.0036 | (-0.0164, 0.0119) | 0.7027 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0132 | (-0.0504, 0.0218) | 0.7720 | -0.0132 | (-0.0329, 0.0072) | 0.8933 |
| controlled_alt_vs_controlled_default | persona_style | -0.0027 | (-0.0157, 0.0110) | 0.6667 | -0.0027 | (-0.0053, -0.0004) | 1.0000 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0057 | (-0.0137, 0.0016) | 0.9327 | -0.0057 | (-0.0123, -0.0003) | 0.9823 |
| controlled_alt_vs_controlled_default | length_score | -0.0347 | (-0.0792, 0.0090) | 0.9387 | -0.0347 | (-0.0970, 0.0190) | 0.8800 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0191 | (-0.0028, 0.0427) | 0.0610 | 0.0191 | (-0.0028, 0.0451) | 0.0703 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0122 | (-0.0295, 0.0057) | 0.9057 | -0.0122 | (-0.0220, -0.0028) | 0.9927 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.0261 | (0.0023, 0.0510) | 0.0153 | 0.0261 | (0.0069, 0.0443) | 0.0043 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.0860 | (0.0566, 0.1157) | 0.0000 | 0.0860 | (0.0408, 0.1347) | 0.0000 |
| controlled_alt_vs_proposed_raw | naturalness | -0.0096 | (-0.0194, 0.0000) | 0.9747 | -0.0096 | (-0.0183, -0.0013) | 0.9923 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 0.0238 | (0.0024, 0.0457) | 0.0153 | 0.0238 | (0.0062, 0.0398) | 0.0040 |
| controlled_alt_vs_proposed_raw | lore_consistency | 0.0133 | (-0.0040, 0.0319) | 0.0680 | 0.0133 | (-0.0087, 0.0280) | 0.0853 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_proposed_raw | objective_completion_support | 0.0200 | (0.0079, 0.0326) | 0.0010 | 0.0200 | (0.0021, 0.0366) | 0.0143 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 0.0094 | (0.0014, 0.0183) | 0.0113 | 0.0094 | (0.0027, 0.0169) | 0.0003 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | 0.0037 | (-0.0048, 0.0123) | 0.1930 | 0.0037 | (-0.0058, 0.0130) | 0.2273 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.0327 | (0.0013, 0.0652) | 0.0203 | 0.0327 | (0.0076, 0.0564) | 0.0070 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0107 | (-0.0001, 0.0235) | 0.0273 | 0.0107 | (0.0013, 0.0200) | 0.0147 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1050 | (0.0701, 0.1418) | 0.0000 | 0.1050 | (0.0478, 0.1682) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0100 | (-0.0044, 0.0264) | 0.0880 | 0.0100 | (-0.0007, 0.0245) | 0.0433 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0133 | (-0.0221, -0.0049) | 0.9993 | -0.0133 | (-0.0246, -0.0023) | 0.9917 |
| controlled_alt_vs_proposed_raw | length_score | -0.0509 | (-0.0917, -0.0095) | 0.9937 | -0.0509 | (-0.0949, -0.0137) | 0.9987 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0653 | (0.0365, 0.0941) | 0.0000 | 0.0653 | (0.0316, 0.1059) | 0.0000 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.0422 | (0.0256, 0.0596) | 0.0000 | 0.0422 | (0.0246, 0.0606) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.0812 | (0.0584, 0.1051) | 0.0000 | 0.0812 | (0.0623, 0.1005) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.0747 | (0.0453, 0.1028) | 0.0000 | 0.0747 | (0.0234, 0.1222) | 0.0033 |
| controlled_alt_vs_candidate_no_context | naturalness | -0.0112 | (-0.0214, -0.0011) | 0.9853 | -0.0112 | (-0.0206, -0.0022) | 0.9950 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 0.0711 | (0.0518, 0.0917) | 0.0000 | 0.0711 | (0.0554, 0.0869) | 0.0000 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 0.0493 | (0.0330, 0.0667) | 0.0000 | 0.0493 | (0.0370, 0.0623) | 0.0000 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0.1250 | (0.0764, 0.1806) | 0.0000 | 0.1250 | (0.0000, 0.3750) | 0.3380 |
| controlled_alt_vs_candidate_no_context | objective_completion_support | 0.0152 | (0.0040, 0.0267) | 0.0037 | 0.0152 | (0.0011, 0.0283) | 0.0177 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 0.0222 | (0.0144, 0.0305) | 0.0000 | 0.0222 | (0.0132, 0.0313) | 0.0000 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | -0.0020 | (-0.0101, 0.0066) | 0.6753 | -0.0020 | (-0.0085, 0.0052) | 0.7160 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.1044 | (0.0767, 0.1348) | 0.0000 | 0.1044 | (0.0792, 0.1314) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0271 | (0.0175, 0.0390) | 0.0000 | 0.0271 | (0.0196, 0.0365) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.0899 | (0.0533, 0.1249) | 0.0000 | 0.0899 | (0.0298, 0.1492) | 0.0017 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0138 | (-0.0003, 0.0291) | 0.0277 | 0.0138 | (-0.0009, 0.0338) | 0.0930 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0065 | (-0.0167, 0.0026) | 0.9187 | -0.0065 | (-0.0182, 0.0040) | 0.8807 |
| controlled_alt_vs_candidate_no_context | length_score | -0.0537 | (-0.0963, -0.0106) | 0.9910 | -0.0537 | (-0.0991, -0.0171) | 1.0000 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.0267 | (0.0049, 0.0486) | 0.0117 | 0.0267 | (-0.0122, 0.0705) | 0.1100 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.0632 | (0.0481, 0.0794) | 0.0000 | 0.0632 | (0.0418, 0.0842) | 0.0000 |
| controlled_alt_vs_baseline_no_context | context_relevance | 0.0764 | (0.0521, 0.1005) | 0.0000 | 0.0764 | (0.0537, 0.0985) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 0.1300 | (0.1014, 0.1593) | 0.0000 | 0.1300 | (0.0874, 0.1749) | 0.0000 |
| controlled_alt_vs_baseline_no_context | naturalness | -0.0285 | (-0.0394, -0.0178) | 1.0000 | -0.0285 | (-0.0414, -0.0157) | 1.0000 |
| controlled_alt_vs_baseline_no_context | quest_state_correctness | 0.0680 | (0.0469, 0.0891) | 0.0000 | 0.0680 | (0.0489, 0.0874) | 0.0000 |
| controlled_alt_vs_baseline_no_context | lore_consistency | 0.0405 | (0.0233, 0.0599) | 0.0000 | 0.0405 | (0.0260, 0.0560) | 0.0000 |
| controlled_alt_vs_baseline_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_baseline_no_context | objective_completion_support | 0.0466 | (0.0325, 0.0609) | 0.0000 | 0.0466 | (0.0217, 0.0770) | 0.0000 |
| controlled_alt_vs_baseline_no_context | gameplay_usefulness | 0.0275 | (0.0192, 0.0362) | 0.0000 | 0.0275 | (0.0105, 0.0439) | 0.0013 |
| controlled_alt_vs_baseline_no_context | time_pressure_acceptability | 0.0110 | (0.0004, 0.0219) | 0.0197 | 0.0110 | (-0.0055, 0.0307) | 0.1170 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 0.0988 | (0.0695, 0.1298) | 0.0000 | 0.0988 | (0.0709, 0.1274) | 0.0000 |
| controlled_alt_vs_baseline_no_context | context_overlap | 0.0242 | (0.0138, 0.0363) | 0.0000 | 0.0242 | (0.0110, 0.0384) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 0.1642 | (0.1281, 0.2034) | 0.0000 | 0.1642 | (0.1110, 0.2193) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_style | -0.0068 | (-0.0216, 0.0076) | 0.8000 | -0.0068 | (-0.0317, 0.0156) | 0.7027 |
| controlled_alt_vs_baseline_no_context | distinct1 | -0.0477 | (-0.0574, -0.0382) | 1.0000 | -0.0477 | (-0.0638, -0.0338) | 1.0000 |
| controlled_alt_vs_baseline_no_context | length_score | -0.0887 | (-0.1338, -0.0433) | 1.0000 | -0.0887 | (-0.1658, -0.0164) | 0.9943 |
| controlled_alt_vs_baseline_no_context | sentence_score | 0.0899 | (0.0607, 0.1191) | 0.0000 | 0.0899 | (0.0535, 0.1264) | 0.0000 |
| controlled_alt_vs_baseline_no_context | overall_quality | 0.0785 | (0.0624, 0.0949) | 0.0000 | 0.0785 | (0.0546, 0.0980) | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_relevance | 0.0761 | (0.0543, 0.0992) | 0.0000 | 0.0761 | (0.0552, 0.0987) | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1403 | (0.1130, 0.1689) | 0.0000 | 0.1403 | (0.0983, 0.1871) | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | naturalness | -0.0270 | (-0.0382, -0.0161) | 1.0000 | -0.0270 | (-0.0371, -0.0162) | 1.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.0669 | (0.0463, 0.0871) | 0.0000 | 0.0669 | (0.0493, 0.0864) | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | lore_consistency | 0.0402 | (0.0239, 0.0573) | 0.0000 | 0.0402 | (0.0285, 0.0526) | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.0494 | (0.0375, 0.0620) | 0.0000 | 0.0494 | (0.0291, 0.0737) | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 0.0288 | (0.0201, 0.0376) | 0.0000 | 0.0288 | (0.0147, 0.0436) | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 0.0110 | (0.0012, 0.0208) | 0.0150 | 0.0110 | (-0.0023, 0.0274) | 0.0547 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.0986 | (0.0708, 0.1272) | 0.0000 | 0.0986 | (0.0715, 0.1280) | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_overlap | 0.0235 | (0.0130, 0.0356) | 0.0000 | 0.0235 | (0.0107, 0.0371) | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1763 | (0.1437, 0.2085) | 0.0000 | 0.1763 | (0.1267, 0.2306) | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_style | -0.0039 | (-0.0240, 0.0167) | 0.6433 | -0.0039 | (-0.0376, 0.0294) | 0.5697 |
| controlled_alt_vs_baseline_no_context_phi3_latest | distinct1 | -0.0473 | (-0.0577, -0.0377) | 1.0000 | -0.0473 | (-0.0597, -0.0350) | 1.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | length_score | -0.0882 | (-0.1338, -0.0428) | 1.0000 | -0.0882 | (-0.1442, -0.0345) | 1.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | sentence_score | 0.1021 | (0.0729, 0.1337) | 0.0000 | 0.1021 | (0.0632, 0.1434) | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | overall_quality | 0.0826 | (0.0666, 0.0990) | 0.0000 | 0.0826 | (0.0625, 0.1025) | 0.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | context_relevance | 0.0764 | (0.0529, 0.1014) | 0.0000 | 0.0764 | (0.0536, 0.0989) | 0.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | persona_consistency | 0.1300 | (0.1012, 0.1588) | 0.0000 | 0.1300 | (0.0882, 0.1739) | 0.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | naturalness | -0.0285 | (-0.0393, -0.0177) | 1.0000 | -0.0285 | (-0.0419, -0.0155) | 1.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | quest_state_correctness | 0.0680 | (0.0475, 0.0898) | 0.0000 | 0.0680 | (0.0496, 0.0866) | 0.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | lore_consistency | 0.0405 | (0.0231, 0.0597) | 0.0000 | 0.0405 | (0.0260, 0.0560) | 0.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | objective_completion_support | 0.0466 | (0.0328, 0.0607) | 0.0000 | 0.0466 | (0.0216, 0.0783) | 0.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | gameplay_usefulness | 0.0275 | (0.0187, 0.0360) | 0.0000 | 0.0275 | (0.0106, 0.0444) | 0.0007 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | time_pressure_acceptability | 0.0110 | (0.0004, 0.0214) | 0.0190 | 0.0110 | (-0.0055, 0.0310) | 0.1097 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | context_keyword_coverage | 0.0988 | (0.0689, 0.1316) | 0.0000 | 0.0988 | (0.0703, 0.1287) | 0.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | context_overlap | 0.0242 | (0.0137, 0.0366) | 0.0000 | 0.0242 | (0.0110, 0.0397) | 0.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | persona_keyword_coverage | 0.1642 | (0.1268, 0.2012) | 0.0000 | 0.1642 | (0.1099, 0.2166) | 0.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | persona_style | -0.0068 | (-0.0220, 0.0081) | 0.8147 | -0.0068 | (-0.0317, 0.0174) | 0.6853 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | distinct1 | -0.0477 | (-0.0581, -0.0381) | 1.0000 | -0.0477 | (-0.0630, -0.0336) | 1.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | length_score | -0.0887 | (-0.1347, -0.0419) | 1.0000 | -0.0887 | (-0.1681, -0.0171) | 0.9963 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | sentence_score | 0.0899 | (0.0608, 0.1215) | 0.0000 | 0.0899 | (0.0510, 0.1264) | 0.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | overall_quality | 0.0785 | (0.0619, 0.0951) | 0.0000 | 0.0785 | (0.0556, 0.0983) | 0.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | context_relevance | 0.0761 | (0.0540, 0.0996) | 0.0000 | 0.0761 | (0.0543, 0.0984) | 0.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1403 | (0.1131, 0.1687) | 0.0000 | 0.1403 | (0.0983, 0.1823) | 0.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | naturalness | -0.0270 | (-0.0384, -0.0161) | 1.0000 | -0.0270 | (-0.0372, -0.0162) | 1.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.0669 | (0.0472, 0.0872) | 0.0000 | 0.0669 | (0.0497, 0.0858) | 0.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | lore_consistency | 0.0402 | (0.0243, 0.0577) | 0.0000 | 0.0402 | (0.0286, 0.0521) | 0.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.0494 | (0.0371, 0.0620) | 0.0000 | 0.0494 | (0.0287, 0.0737) | 0.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 0.0288 | (0.0202, 0.0375) | 0.0000 | 0.0288 | (0.0142, 0.0437) | 0.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 0.0110 | (0.0017, 0.0206) | 0.0097 | 0.0110 | (-0.0027, 0.0268) | 0.0623 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.0986 | (0.0713, 0.1278) | 0.0000 | 0.0986 | (0.0721, 0.1283) | 0.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | context_overlap | 0.0235 | (0.0130, 0.0362) | 0.0000 | 0.0235 | (0.0105, 0.0366) | 0.0003 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1763 | (0.1408, 0.2118) | 0.0000 | 0.1763 | (0.1279, 0.2292) | 0.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | persona_style | -0.0039 | (-0.0246, 0.0171) | 0.6530 | -0.0039 | (-0.0379, 0.0283) | 0.5697 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | distinct1 | -0.0473 | (-0.0577, -0.0378) | 1.0000 | -0.0473 | (-0.0601, -0.0355) | 1.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | length_score | -0.0882 | (-0.1384, -0.0428) | 1.0000 | -0.0882 | (-0.1445, -0.0368) | 1.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | sentence_score | 0.1021 | (0.0705, 0.1312) | 0.0000 | 0.1021 | (0.0632, 0.1434) | 0.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | overall_quality | 0.0826 | (0.0669, 0.0986) | 0.0000 | 0.0826 | (0.0619, 0.1030) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 71 | 22 | 51 | 0.6701 | 0.7634 |
| proposed_vs_candidate_no_context | persona_consistency | 24 | 27 | 93 | 0.4896 | 0.4706 |
| proposed_vs_candidate_no_context | naturalness | 41 | 49 | 54 | 0.4722 | 0.4556 |
| proposed_vs_candidate_no_context | quest_state_correctness | 71 | 22 | 51 | 0.6701 | 0.7634 |
| proposed_vs_candidate_no_context | lore_consistency | 60 | 8 | 76 | 0.6806 | 0.8824 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 18 | 0 | 126 | 0.5625 | 1.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | 36 | 35 | 73 | 0.5035 | 0.5070 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 55 | 38 | 51 | 0.5590 | 0.5914 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | 34 | 47 | 63 | 0.4549 | 0.4198 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 54 | 11 | 79 | 0.6493 | 0.8308 |
| proposed_vs_candidate_no_context | context_overlap | 66 | 27 | 51 | 0.6354 | 0.7097 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 17 | 23 | 104 | 0.4792 | 0.4250 |
| proposed_vs_candidate_no_context | persona_style | 11 | 10 | 123 | 0.5035 | 0.5238 |
| proposed_vs_candidate_no_context | distinct1 | 44 | 33 | 67 | 0.5382 | 0.5714 |
| proposed_vs_candidate_no_context | length_score | 43 | 43 | 58 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | sentence_score | 8 | 24 | 112 | 0.4444 | 0.2500 |
| proposed_vs_candidate_no_context | overall_quality | 59 | 34 | 51 | 0.5868 | 0.6344 |
| proposed_vs_baseline_no_context | context_relevance | 81 | 60 | 3 | 0.5729 | 0.5745 |
| proposed_vs_baseline_no_context | persona_consistency | 60 | 25 | 59 | 0.6215 | 0.7059 |
| proposed_vs_baseline_no_context | naturalness | 54 | 89 | 1 | 0.3785 | 0.3776 |
| proposed_vs_baseline_no_context | quest_state_correctness | 84 | 57 | 3 | 0.5938 | 0.5957 |
| proposed_vs_baseline_no_context | lore_consistency | 57 | 41 | 46 | 0.5556 | 0.5816 |
| proposed_vs_baseline_no_context | multi_turn_contradiction_safety | 0 | 0 | 144 | 0.5000 | nan |
| proposed_vs_baseline_no_context | objective_completion_support | 65 | 39 | 40 | 0.5903 | 0.6250 |
| proposed_vs_baseline_no_context | gameplay_usefulness | 95 | 49 | 0 | 0.6597 | 0.6597 |
| proposed_vs_baseline_no_context | time_pressure_acceptability | 67 | 56 | 21 | 0.5382 | 0.5447 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 60 | 18 | 66 | 0.6458 | 0.7692 |
| proposed_vs_baseline_no_context | context_overlap | 81 | 59 | 4 | 0.5764 | 0.5786 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 57 | 18 | 69 | 0.6354 | 0.7600 |
| proposed_vs_baseline_no_context | persona_style | 6 | 18 | 120 | 0.4583 | 0.2500 |
| proposed_vs_baseline_no_context | distinct1 | 31 | 94 | 19 | 0.2812 | 0.2480 |
| proposed_vs_baseline_no_context | length_score | 64 | 72 | 8 | 0.4722 | 0.4706 |
| proposed_vs_baseline_no_context | sentence_score | 32 | 21 | 91 | 0.5382 | 0.6038 |
| proposed_vs_baseline_no_context | overall_quality | 105 | 39 | 0 | 0.7292 | 0.7292 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 82 | 61 | 1 | 0.5729 | 0.5734 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | 64 | 23 | 57 | 0.6424 | 0.7356 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | 59 | 84 | 1 | 0.4132 | 0.4126 |
| proposed_vs_baseline_no_context_phi3_latest | quest_state_correctness | 83 | 61 | 0 | 0.5764 | 0.5764 |
| proposed_vs_baseline_no_context_phi3_latest | lore_consistency | 55 | 39 | 50 | 0.5556 | 0.5851 |
| proposed_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0 | 0 | 144 | 0.5000 | nan |
| proposed_vs_baseline_no_context_phi3_latest | objective_completion_support | 67 | 34 | 43 | 0.6146 | 0.6634 |
| proposed_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 95 | 49 | 0 | 0.6597 | 0.6597 |
| proposed_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 67 | 53 | 24 | 0.5486 | 0.5583 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 61 | 18 | 65 | 0.6493 | 0.7722 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 81 | 61 | 2 | 0.5694 | 0.5704 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 60 | 14 | 70 | 0.6597 | 0.8108 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | 10 | 21 | 113 | 0.4618 | 0.3226 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | 39 | 89 | 16 | 0.3264 | 0.3047 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | 66 | 70 | 8 | 0.4861 | 0.4853 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | 33 | 17 | 94 | 0.5556 | 0.6600 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | 101 | 43 | 0 | 0.7014 | 0.7014 |
| controlled_vs_proposed_raw | context_relevance | 68 | 48 | 28 | 0.5694 | 0.5862 |
| controlled_vs_proposed_raw | persona_consistency | 71 | 12 | 61 | 0.7049 | 0.8554 |
| controlled_vs_proposed_raw | naturalness | 60 | 55 | 29 | 0.5174 | 0.5217 |
| controlled_vs_proposed_raw | quest_state_correctness | 67 | 49 | 28 | 0.5625 | 0.5776 |
| controlled_vs_proposed_raw | lore_consistency | 40 | 38 | 66 | 0.5069 | 0.5128 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0 | 0 | 144 | 0.5000 | nan |
| controlled_vs_proposed_raw | objective_completion_support | 47 | 35 | 62 | 0.5417 | 0.5732 |
| controlled_vs_proposed_raw | gameplay_usefulness | 72 | 44 | 28 | 0.5972 | 0.6207 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 48 | 48 | 48 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | context_keyword_coverage | 47 | 27 | 70 | 0.5694 | 0.6351 |
| controlled_vs_proposed_raw | context_overlap | 70 | 46 | 28 | 0.5833 | 0.6034 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 69 | 10 | 65 | 0.7049 | 0.8734 |
| controlled_vs_proposed_raw | persona_style | 14 | 5 | 125 | 0.5312 | 0.7368 |
| controlled_vs_proposed_raw | distinct1 | 52 | 57 | 35 | 0.4826 | 0.4771 |
| controlled_vs_proposed_raw | length_score | 61 | 51 | 32 | 0.5347 | 0.5446 |
| controlled_vs_proposed_raw | sentence_score | 29 | 10 | 105 | 0.5660 | 0.7436 |
| controlled_vs_proposed_raw | overall_quality | 86 | 30 | 28 | 0.6944 | 0.7414 |
| controlled_vs_candidate_no_context | context_relevance | 89 | 29 | 26 | 0.7083 | 0.7542 |
| controlled_vs_candidate_no_context | persona_consistency | 71 | 14 | 59 | 0.6979 | 0.8353 |
| controlled_vs_candidate_no_context | naturalness | 62 | 56 | 26 | 0.5208 | 0.5254 |
| controlled_vs_candidate_no_context | quest_state_correctness | 88 | 30 | 26 | 0.7014 | 0.7458 |
| controlled_vs_candidate_no_context | lore_consistency | 50 | 17 | 77 | 0.6146 | 0.7463 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 18 | 0 | 126 | 0.5625 | 1.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | 45 | 40 | 59 | 0.5174 | 0.5294 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 90 | 28 | 26 | 0.7153 | 0.7627 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 44 | 55 | 45 | 0.4618 | 0.4444 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 73 | 8 | 63 | 0.7257 | 0.9012 |
| controlled_vs_candidate_no_context | context_overlap | 82 | 36 | 26 | 0.6597 | 0.6949 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 66 | 14 | 64 | 0.6806 | 0.8250 |
| controlled_vs_candidate_no_context | persona_style | 13 | 7 | 124 | 0.5208 | 0.6500 |
| controlled_vs_candidate_no_context | distinct1 | 62 | 50 | 32 | 0.5417 | 0.5536 |
| controlled_vs_candidate_no_context | length_score | 62 | 51 | 31 | 0.5382 | 0.5487 |
| controlled_vs_candidate_no_context | sentence_score | 19 | 15 | 110 | 0.5139 | 0.5588 |
| controlled_vs_candidate_no_context | overall_quality | 98 | 20 | 26 | 0.7708 | 0.8305 |
| controlled_vs_baseline_no_context | context_relevance | 96 | 45 | 3 | 0.6771 | 0.6809 |
| controlled_vs_baseline_no_context | persona_consistency | 112 | 14 | 18 | 0.8403 | 0.8889 |
| controlled_vs_baseline_no_context | naturalness | 61 | 83 | 0 | 0.4236 | 0.4236 |
| controlled_vs_baseline_no_context | quest_state_correctness | 93 | 48 | 3 | 0.6562 | 0.6596 |
| controlled_vs_baseline_no_context | lore_consistency | 52 | 44 | 48 | 0.5278 | 0.5417 |
| controlled_vs_baseline_no_context | multi_turn_contradiction_safety | 0 | 0 | 144 | 0.5000 | nan |
| controlled_vs_baseline_no_context | objective_completion_support | 84 | 32 | 28 | 0.6806 | 0.7241 |
| controlled_vs_baseline_no_context | gameplay_usefulness | 106 | 38 | 0 | 0.7361 | 0.7361 |
| controlled_vs_baseline_no_context | time_pressure_acceptability | 80 | 52 | 12 | 0.5972 | 0.6061 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 73 | 17 | 54 | 0.6944 | 0.8111 |
| controlled_vs_baseline_no_context | context_overlap | 90 | 50 | 4 | 0.6389 | 0.6429 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 112 | 11 | 21 | 0.8507 | 0.9106 |
| controlled_vs_baseline_no_context | persona_style | 11 | 11 | 122 | 0.5000 | 0.5000 |
| controlled_vs_baseline_no_context | distinct1 | 27 | 110 | 7 | 0.2118 | 0.1971 |
| controlled_vs_baseline_no_context | length_score | 64 | 73 | 7 | 0.4688 | 0.4672 |
| controlled_vs_baseline_no_context | sentence_score | 40 | 11 | 93 | 0.6007 | 0.7843 |
| controlled_vs_baseline_no_context | overall_quality | 121 | 23 | 0 | 0.8403 | 0.8403 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 94 | 50 | 0 | 0.6528 | 0.6528 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 116 | 8 | 20 | 0.8750 | 0.9355 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | 56 | 88 | 0 | 0.3889 | 0.3889 |
| controlled_vs_baseline_no_context_phi3_latest | quest_state_correctness | 93 | 51 | 0 | 0.6458 | 0.6458 |
| controlled_vs_baseline_no_context_phi3_latest | lore_consistency | 53 | 36 | 55 | 0.5590 | 0.5955 |
| controlled_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0 | 0 | 144 | 0.5000 | nan |
| controlled_vs_baseline_no_context_phi3_latest | objective_completion_support | 84 | 24 | 36 | 0.7083 | 0.7778 |
| controlled_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 105 | 39 | 0 | 0.7292 | 0.7292 |
| controlled_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 78 | 48 | 18 | 0.6042 | 0.6190 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 74 | 19 | 51 | 0.6910 | 0.7957 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 89 | 54 | 1 | 0.6215 | 0.6224 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 116 | 6 | 22 | 0.8819 | 0.9508 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | 12 | 13 | 119 | 0.4965 | 0.4800 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | 31 | 107 | 6 | 0.2361 | 0.2246 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | 66 | 73 | 5 | 0.4757 | 0.4748 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 44 | 10 | 90 | 0.6181 | 0.8148 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 121 | 23 | 0 | 0.8403 | 0.8403 |
| controlled_alt_vs_controlled_default | context_relevance | 40 | 53 | 51 | 0.4549 | 0.4301 |
| controlled_alt_vs_controlled_default | persona_consistency | 21 | 33 | 90 | 0.4583 | 0.3889 |
| controlled_alt_vs_controlled_default | naturalness | 40 | 53 | 51 | 0.4549 | 0.4301 |
| controlled_alt_vs_controlled_default | quest_state_correctness | 40 | 53 | 51 | 0.4549 | 0.4301 |
| controlled_alt_vs_controlled_default | lore_consistency | 35 | 34 | 75 | 0.5035 | 0.5072 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0 | 0 | 144 | 0.5000 | nan |
| controlled_alt_vs_controlled_default | objective_completion_support | 39 | 37 | 68 | 0.5069 | 0.5132 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | 40 | 53 | 51 | 0.4549 | 0.4301 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | 38 | 45 | 61 | 0.4757 | 0.4578 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 32 | 42 | 70 | 0.4653 | 0.4324 |
| controlled_alt_vs_controlled_default | context_overlap | 40 | 53 | 51 | 0.4549 | 0.4301 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 18 | 29 | 97 | 0.4618 | 0.3830 |
| controlled_alt_vs_controlled_default | persona_style | 4 | 8 | 132 | 0.4861 | 0.3333 |
| controlled_alt_vs_controlled_default | distinct1 | 43 | 44 | 57 | 0.4965 | 0.4943 |
| controlled_alt_vs_controlled_default | length_score | 38 | 53 | 53 | 0.4479 | 0.4176 |
| controlled_alt_vs_controlled_default | sentence_score | 14 | 7 | 123 | 0.5243 | 0.6667 |
| controlled_alt_vs_controlled_default | overall_quality | 44 | 49 | 51 | 0.4826 | 0.4731 |
| controlled_alt_vs_proposed_raw | context_relevance | 63 | 54 | 27 | 0.5312 | 0.5385 |
| controlled_alt_vs_proposed_raw | persona_consistency | 73 | 12 | 59 | 0.7118 | 0.8588 |
| controlled_alt_vs_proposed_raw | naturalness | 49 | 68 | 27 | 0.4340 | 0.4188 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 64 | 53 | 27 | 0.5382 | 0.5470 |
| controlled_alt_vs_proposed_raw | lore_consistency | 38 | 37 | 69 | 0.5035 | 0.5067 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0 | 0 | 144 | 0.5000 | nan |
| controlled_alt_vs_proposed_raw | objective_completion_support | 50 | 42 | 52 | 0.5278 | 0.5435 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 67 | 50 | 27 | 0.5590 | 0.5726 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | 46 | 52 | 46 | 0.4792 | 0.4694 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 48 | 35 | 61 | 0.5451 | 0.5783 |
| controlled_alt_vs_proposed_raw | context_overlap | 62 | 54 | 28 | 0.5278 | 0.5345 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 67 | 11 | 66 | 0.6944 | 0.8590 |
| controlled_alt_vs_proposed_raw | persona_style | 13 | 8 | 123 | 0.5174 | 0.6190 |
| controlled_alt_vs_proposed_raw | distinct1 | 42 | 67 | 35 | 0.4132 | 0.3853 |
| controlled_alt_vs_proposed_raw | length_score | 47 | 67 | 30 | 0.4306 | 0.4123 |
| controlled_alt_vs_proposed_raw | sentence_score | 32 | 6 | 106 | 0.5903 | 0.8421 |
| controlled_alt_vs_proposed_raw | overall_quality | 83 | 34 | 27 | 0.6701 | 0.7094 |
| controlled_alt_vs_candidate_no_context | context_relevance | 86 | 31 | 27 | 0.6910 | 0.7350 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 70 | 13 | 61 | 0.6979 | 0.8434 |
| controlled_alt_vs_candidate_no_context | naturalness | 50 | 67 | 27 | 0.4410 | 0.4274 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 85 | 32 | 27 | 0.6840 | 0.7265 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 46 | 12 | 86 | 0.6181 | 0.7931 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 18 | 0 | 126 | 0.5625 | 1.0000 |
| controlled_alt_vs_candidate_no_context | objective_completion_support | 50 | 35 | 59 | 0.5521 | 0.5882 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 82 | 35 | 27 | 0.6632 | 0.7009 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | 41 | 58 | 45 | 0.4410 | 0.4141 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 64 | 11 | 69 | 0.6840 | 0.8533 |
| controlled_alt_vs_candidate_no_context | context_overlap | 83 | 34 | 27 | 0.6701 | 0.7094 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 65 | 13 | 66 | 0.6806 | 0.8333 |
| controlled_alt_vs_candidate_no_context | persona_style | 14 | 8 | 122 | 0.5208 | 0.6364 |
| controlled_alt_vs_candidate_no_context | distinct1 | 55 | 56 | 33 | 0.4965 | 0.4955 |
| controlled_alt_vs_candidate_no_context | length_score | 51 | 63 | 30 | 0.4583 | 0.4474 |
| controlled_alt_vs_candidate_no_context | sentence_score | 17 | 6 | 121 | 0.5382 | 0.7391 |
| controlled_alt_vs_candidate_no_context | overall_quality | 92 | 25 | 27 | 0.7326 | 0.7863 |
| controlled_alt_vs_baseline_no_context | context_relevance | 85 | 56 | 3 | 0.6007 | 0.6028 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 108 | 12 | 24 | 0.8333 | 0.9000 |
| controlled_alt_vs_baseline_no_context | naturalness | 49 | 95 | 0 | 0.3403 | 0.3403 |
| controlled_alt_vs_baseline_no_context | quest_state_correctness | 86 | 55 | 3 | 0.6076 | 0.6099 |
| controlled_alt_vs_baseline_no_context | lore_consistency | 47 | 44 | 53 | 0.5104 | 0.5165 |
| controlled_alt_vs_baseline_no_context | multi_turn_contradiction_safety | 0 | 0 | 144 | 0.5000 | nan |
| controlled_alt_vs_baseline_no_context | objective_completion_support | 83 | 35 | 26 | 0.6667 | 0.7034 |
| controlled_alt_vs_baseline_no_context | gameplay_usefulness | 101 | 43 | 0 | 0.7014 | 0.7014 |
| controlled_alt_vs_baseline_no_context | time_pressure_acceptability | 75 | 58 | 11 | 0.5590 | 0.5639 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 63 | 20 | 61 | 0.6493 | 0.7590 |
| controlled_alt_vs_baseline_no_context | context_overlap | 84 | 57 | 3 | 0.5938 | 0.5957 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 106 | 11 | 27 | 0.8299 | 0.9060 |
| controlled_alt_vs_baseline_no_context | persona_style | 9 | 12 | 123 | 0.4896 | 0.4286 |
| controlled_alt_vs_baseline_no_context | distinct1 | 24 | 113 | 7 | 0.1910 | 0.1752 |
| controlled_alt_vs_baseline_no_context | length_score | 51 | 80 | 13 | 0.3993 | 0.3893 |
| controlled_alt_vs_baseline_no_context | sentence_score | 43 | 6 | 95 | 0.6285 | 0.8776 |
| controlled_alt_vs_baseline_no_context | overall_quality | 116 | 28 | 0 | 0.8056 | 0.8056 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_relevance | 90 | 54 | 0 | 0.6250 | 0.6250 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_consistency | 111 | 11 | 22 | 0.8472 | 0.9098 |
| controlled_alt_vs_baseline_no_context_phi3_latest | naturalness | 47 | 96 | 1 | 0.3299 | 0.3287 |
| controlled_alt_vs_baseline_no_context_phi3_latest | quest_state_correctness | 89 | 55 | 0 | 0.6181 | 0.6181 |
| controlled_alt_vs_baseline_no_context_phi3_latest | lore_consistency | 47 | 39 | 58 | 0.5278 | 0.5465 |
| controlled_alt_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0 | 0 | 144 | 0.5000 | nan |
| controlled_alt_vs_baseline_no_context_phi3_latest | objective_completion_support | 87 | 28 | 29 | 0.7049 | 0.7565 |
| controlled_alt_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 101 | 43 | 0 | 0.7014 | 0.7014 |
| controlled_alt_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 80 | 48 | 16 | 0.6111 | 0.6250 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 64 | 17 | 63 | 0.6632 | 0.7901 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_overlap | 87 | 57 | 0 | 0.6042 | 0.6042 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 110 | 7 | 27 | 0.8576 | 0.9402 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_style | 12 | 16 | 116 | 0.4861 | 0.4286 |
| controlled_alt_vs_baseline_no_context_phi3_latest | distinct1 | 27 | 112 | 5 | 0.2049 | 0.1942 |
| controlled_alt_vs_baseline_no_context_phi3_latest | length_score | 58 | 81 | 5 | 0.4201 | 0.4173 |
| controlled_alt_vs_baseline_no_context_phi3_latest | sentence_score | 48 | 6 | 90 | 0.6458 | 0.8889 |
| controlled_alt_vs_baseline_no_context_phi3_latest | overall_quality | 118 | 26 | 0 | 0.8194 | 0.8194 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | context_relevance | 85 | 56 | 3 | 0.6007 | 0.6028 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | persona_consistency | 108 | 12 | 24 | 0.8333 | 0.9000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | naturalness | 49 | 95 | 0 | 0.3403 | 0.3403 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | quest_state_correctness | 86 | 55 | 3 | 0.6076 | 0.6099 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | lore_consistency | 47 | 44 | 53 | 0.5104 | 0.5165 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | multi_turn_contradiction_safety | 0 | 0 | 144 | 0.5000 | nan |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | objective_completion_support | 83 | 35 | 26 | 0.6667 | 0.7034 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | gameplay_usefulness | 101 | 43 | 0 | 0.7014 | 0.7014 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | time_pressure_acceptability | 75 | 58 | 11 | 0.5590 | 0.5639 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | context_keyword_coverage | 63 | 20 | 61 | 0.6493 | 0.7590 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | context_overlap | 84 | 57 | 3 | 0.5938 | 0.5957 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | persona_keyword_coverage | 106 | 11 | 27 | 0.8299 | 0.9060 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | persona_style | 9 | 12 | 123 | 0.4896 | 0.4286 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | distinct1 | 24 | 113 | 7 | 0.1910 | 0.1752 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | length_score | 51 | 80 | 13 | 0.3993 | 0.3893 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | sentence_score | 43 | 6 | 95 | 0.6285 | 0.8776 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | overall_quality | 116 | 28 | 0 | 0.8056 | 0.8056 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | context_relevance | 90 | 54 | 0 | 0.6250 | 0.6250 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | persona_consistency | 111 | 11 | 22 | 0.8472 | 0.9098 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | naturalness | 47 | 96 | 1 | 0.3299 | 0.3287 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | quest_state_correctness | 89 | 55 | 0 | 0.6181 | 0.6181 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | lore_consistency | 47 | 39 | 58 | 0.5278 | 0.5465 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0 | 0 | 144 | 0.5000 | nan |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | objective_completion_support | 87 | 28 | 29 | 0.7049 | 0.7565 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 101 | 43 | 0 | 0.7014 | 0.7014 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 80 | 48 | 16 | 0.6111 | 0.6250 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 64 | 17 | 63 | 0.6632 | 0.7901 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | context_overlap | 87 | 57 | 0 | 0.6042 | 0.6042 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 110 | 7 | 27 | 0.8576 | 0.9402 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | persona_style | 12 | 16 | 116 | 0.4861 | 0.4286 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | distinct1 | 27 | 112 | 5 | 0.2049 | 0.1942 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | length_score | 58 | 81 | 5 | 0.4201 | 0.4173 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | sentence_score | 48 | 6 | 90 | 0.6458 | 0.8889 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | overall_quality | 118 | 26 | 0 | 0.8194 | 0.8194 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.1944 | 0.1875 | 0.8125 |
| proposed_contextual_controlled_tuned | 0.0000 | 0.0000 | 0.2014 | 0.1875 | 0.7986 |
| proposed_contextual | 0.0000 | 0.0000 | 0.4028 | 0.0000 | 0.0000 |
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
# Proposal Alignment Evaluation Report

- Run ID: `20260313T061450Z`
- Generated: `2026-03-13T06:21:31.360280+00:00`
- Scenarios: `artifacts\proposal_control_tuning\auto_tune\20260313T051946Z\train_runs\trial_005\seed_19\20260313T061450Z\scenarios.jsonl`
- Scenario count: `24`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_tuned`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0963 (0.0501, 0.1509) | 0.2914 (0.2305, 0.3591) | 0.8686 (0.8540, 0.8822) | 0.3191 (0.2821, 0.3619) | n/a |
| proposed_contextual_controlled_tuned | 0.0760 (0.0426, 0.1174) | 0.2592 (0.1999, 0.3239) | 0.8801 (0.8662, 0.8958) | 0.2999 (0.2662, 0.3349) | n/a |
| proposed_contextual | 0.0609 (0.0317, 0.0942) | 0.2030 (0.1536, 0.2573) | 0.8745 (0.8601, 0.8893) | 0.2699 (0.2480, 0.2940) | n/a |
| candidate_no_context | 0.0402 (0.0262, 0.0541) | 0.2605 (0.1953, 0.3308) | 0.8836 (0.8672, 0.9011) | 0.2841 (0.2571, 0.3139) | n/a |

## Game-facing Outcome Metrics (mean, 95% CI)
| Arm | Quest-state Correctness | Lore Consistency | Contradiction Safety | Objective Completion Support | Gameplay Usefulness | Time-pressure Acceptability |
|---|---:|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.1763 (0.1379, 0.2190) | 0.0387 (0.0112, 0.0746) | 1.0000 (1.0000, 1.0000) | 0.0769 (0.0539, 0.0997) | 0.2991 (0.2857, 0.3130) | 0.2947 (0.2752, 0.3136) |
| proposed_contextual_controlled_tuned | 0.1661 (0.1340, 0.2059) | 0.0287 (0.0096, 0.0514) | 0.8333 (0.6667, 0.9583) | 0.0804 (0.0567, 0.1024) | 0.3034 (0.2914, 0.3160) | 0.3030 (0.2861, 0.3190) |
| proposed_contextual | 0.1519 (0.1254, 0.1851) | 0.0240 (0.0072, 0.0448) | 1.0000 (1.0000, 1.0000) | 0.0729 (0.0480, 0.0989) | 0.2951 (0.2834, 0.3059) | 0.2973 (0.2803, 0.3142) |
| candidate_no_context | 0.1326 (0.1208, 0.1450) | 0.0036 (0.0005, 0.0073) | 1.0000 (1.0000, 1.0000) | 0.0731 (0.0505, 0.0938) | 0.2887 (0.2806, 0.2972) | 0.3030 (0.2878, 0.3194) |

- Multi-turn contradiction rate is reported as `1 - contradiction_safety` in row-level outputs.

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0208 | 0.5165 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0575 | -0.2207 |
| proposed_vs_candidate_no_context | naturalness | -0.0091 | -0.0103 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.0193 | 0.1458 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0203 | 5.5795 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | -0.0001 | -0.0019 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 0.0064 | 0.0223 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | -0.0057 | -0.0188 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0261 | 0.6216 |
| proposed_vs_candidate_no_context | context_overlap | 0.0082 | 0.2286 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0784 | -0.4115 |
| proposed_vs_candidate_no_context | persona_style | 0.0260 | 0.0482 |
| proposed_vs_candidate_no_context | distinct1 | -0.0027 | -0.0029 |
| proposed_vs_candidate_no_context | length_score | -0.0111 | -0.0200 |
| proposed_vs_candidate_no_context | sentence_score | -0.0583 | -0.0583 |
| proposed_vs_candidate_no_context | overall_quality | -0.0142 | -0.0500 |
| controlled_vs_proposed_raw | context_relevance | 0.0354 | 0.5812 |
| controlled_vs_proposed_raw | persona_consistency | 0.0885 | 0.4359 |
| controlled_vs_proposed_raw | naturalness | -0.0059 | -0.0067 |
| controlled_vs_proposed_raw | quest_state_correctness | 0.0244 | 0.1606 |
| controlled_vs_proposed_raw | lore_consistency | 0.0148 | 0.6160 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0040 | 0.0552 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0040 | 0.0135 |
| controlled_vs_proposed_raw | time_pressure_acceptability | -0.0027 | -0.0090 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.0481 | 0.7056 |
| controlled_vs_proposed_raw | context_overlap | 0.0058 | 0.1319 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1200 | 1.0708 |
| controlled_vs_proposed_raw | persona_style | -0.0378 | -0.0667 |
| controlled_vs_proposed_raw | distinct1 | -0.0031 | -0.0034 |
| controlled_vs_proposed_raw | length_score | -0.0306 | -0.0561 |
| controlled_vs_proposed_raw | sentence_score | 0.0146 | 0.0155 |
| controlled_vs_proposed_raw | overall_quality | 0.0492 | 0.1822 |
| controlled_vs_candidate_no_context | context_relevance | 0.0562 | 1.3979 |
| controlled_vs_candidate_no_context | persona_consistency | 0.0310 | 0.1190 |
| controlled_vs_candidate_no_context | naturalness | -0.0150 | -0.0170 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.0437 | 0.3299 |
| controlled_vs_candidate_no_context | lore_consistency | 0.0351 | 9.6324 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | 0.0039 | 0.0532 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0104 | 0.0360 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | -0.0084 | -0.0277 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.0742 | 1.7658 |
| controlled_vs_candidate_no_context | context_overlap | 0.0140 | 0.3906 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.0417 | 0.2187 |
| controlled_vs_candidate_no_context | persona_style | -0.0117 | -0.0217 |
| controlled_vs_candidate_no_context | distinct1 | -0.0058 | -0.0062 |
| controlled_vs_candidate_no_context | length_score | -0.0417 | -0.0750 |
| controlled_vs_candidate_no_context | sentence_score | -0.0438 | -0.0438 |
| controlled_vs_candidate_no_context | overall_quality | 0.0350 | 0.1230 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0203 | -0.2110 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0322 | -0.1105 |
| controlled_alt_vs_controlled_default | naturalness | 0.0115 | 0.0132 |
| controlled_alt_vs_controlled_default | quest_state_correctness | -0.0102 | -0.0578 |
| controlled_alt_vs_controlled_default | lore_consistency | -0.0100 | -0.2582 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | -0.1667 | -0.1667 |
| controlled_alt_vs_controlled_default | objective_completion_support | 0.0035 | 0.0451 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | 0.0043 | 0.0144 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | 0.0083 | 0.0282 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0277 | -0.2378 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0032 | -0.0651 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0383 | -0.1650 |
| controlled_alt_vs_controlled_default | persona_style | -0.0078 | -0.0148 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0028 | 0.0030 |
| controlled_alt_vs_controlled_default | length_score | 0.0444 | 0.0865 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0146 | 0.0153 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0192 | -0.0602 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.0151 | 0.2476 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.0563 | 0.2773 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0056 | 0.0064 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 0.0142 | 0.0935 |
| controlled_alt_vs_proposed_raw | lore_consistency | 0.0048 | 0.1988 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | -0.1667 | -0.1667 |
| controlled_alt_vs_proposed_raw | objective_completion_support | 0.0075 | 0.1029 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 0.0083 | 0.0281 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | 0.0056 | 0.0189 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.0205 | 0.3000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0026 | 0.0582 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.0817 | 0.7292 |
| controlled_alt_vs_proposed_raw | persona_style | -0.0456 | -0.0805 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0003 | -0.0004 |
| controlled_alt_vs_proposed_raw | length_score | 0.0139 | 0.0255 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0292 | 0.0310 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.0300 | 0.1110 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.0358 | 0.8919 |
| controlled_alt_vs_candidate_no_context | persona_consistency | -0.0012 | -0.0046 |
| controlled_alt_vs_candidate_no_context | naturalness | -0.0036 | -0.0040 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 0.0335 | 0.2530 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 0.0251 | 6.8872 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | -0.1667 | -0.1667 |
| controlled_alt_vs_candidate_no_context | objective_completion_support | 0.0074 | 0.1008 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 0.0147 | 0.0509 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | -0.0001 | -0.0002 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.0466 | 1.1081 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0108 | 0.3001 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.0034 | 0.0177 |
| controlled_alt_vs_candidate_no_context | persona_style | -0.0195 | -0.0361 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0030 | -0.0032 |
| controlled_alt_vs_candidate_no_context | length_score | 0.0028 | 0.0050 |
| controlled_alt_vs_candidate_no_context | sentence_score | -0.0292 | -0.0292 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.0158 | 0.0555 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0208 | (-0.0054, 0.0487) | 0.0663 | 0.0208 | (-0.0002, 0.0458) | 0.0343 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0575 | (-0.1279, 0.0082) | 0.9583 | -0.0575 | (-0.1844, 0.0217) | 0.8567 |
| proposed_vs_candidate_no_context | naturalness | -0.0091 | (-0.0300, 0.0115) | 0.7983 | -0.0091 | (-0.0307, 0.0041) | 0.8623 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.0193 | (-0.0032, 0.0435) | 0.0467 | 0.0193 | (-0.0013, 0.0445) | 0.0297 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0203 | (0.0039, 0.0391) | 0.0043 | 0.0203 | (0.0034, 0.0388) | 0.0117 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | -0.0001 | (-0.0305, 0.0297) | 0.5047 | -0.0001 | (-0.0261, 0.0268) | 0.5210 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 0.0064 | (-0.0080, 0.0204) | 0.1803 | 0.0064 | (-0.0098, 0.0219) | 0.1897 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | -0.0057 | (-0.0237, 0.0118) | 0.7470 | -0.0057 | (-0.0309, 0.0221) | 0.6747 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0261 | (-0.0076, 0.0644) | 0.0743 | 0.0261 | (-0.0003, 0.0598) | 0.0437 |
| proposed_vs_candidate_no_context | context_overlap | 0.0082 | (-0.0020, 0.0200) | 0.0613 | 0.0082 | (0.0002, 0.0176) | 0.0190 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0784 | (-0.1677, 0.0089) | 0.9613 | -0.0784 | (-0.2489, 0.0238) | 0.9027 |
| proposed_vs_candidate_no_context | persona_style | 0.0260 | (-0.0183, 0.0716) | 0.1307 | 0.0260 | (0.0000, 0.0739) | 0.0723 |
| proposed_vs_candidate_no_context | distinct1 | -0.0027 | (-0.0281, 0.0233) | 0.5783 | -0.0027 | (-0.0296, 0.0208) | 0.5707 |
| proposed_vs_candidate_no_context | length_score | -0.0111 | (-0.0917, 0.0681) | 0.6157 | -0.0111 | (-0.1273, 0.0694) | 0.5637 |
| proposed_vs_candidate_no_context | sentence_score | -0.0583 | (-0.1167, -0.0146) | 1.0000 | -0.0583 | (-0.1400, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | overall_quality | -0.0142 | (-0.0405, 0.0105) | 0.8663 | -0.0142 | (-0.0694, 0.0200) | 0.6997 |
| controlled_vs_proposed_raw | context_relevance | 0.0354 | (-0.0124, 0.0903) | 0.0863 | 0.0354 | (-0.0096, 0.0655) | 0.0410 |
| controlled_vs_proposed_raw | persona_consistency | 0.0885 | (0.0186, 0.1645) | 0.0050 | 0.0885 | (0.0269, 0.1305) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | -0.0059 | (-0.0226, 0.0113) | 0.7463 | -0.0059 | (-0.0233, 0.0103) | 0.7043 |
| controlled_vs_proposed_raw | quest_state_correctness | 0.0244 | (-0.0177, 0.0733) | 0.1363 | 0.0244 | (-0.0160, 0.0495) | 0.0800 |
| controlled_vs_proposed_raw | lore_consistency | 0.0148 | (-0.0136, 0.0495) | 0.1703 | 0.0148 | (-0.0100, 0.0312) | 0.1057 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0040 | (-0.0215, 0.0308) | 0.3733 | 0.0040 | (-0.0160, 0.0402) | 0.3950 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0040 | (-0.0110, 0.0215) | 0.3190 | 0.0040 | (-0.0132, 0.0243) | 0.3147 |
| controlled_vs_proposed_raw | time_pressure_acceptability | -0.0027 | (-0.0193, 0.0129) | 0.6253 | -0.0027 | (-0.0172, 0.0168) | 0.6017 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.0481 | (-0.0122, 0.1189) | 0.0647 | 0.0481 | (0.0041, 0.0832) | 0.0243 |
| controlled_vs_proposed_raw | context_overlap | 0.0058 | (-0.0149, 0.0280) | 0.2973 | 0.0058 | (-0.0174, 0.0240) | 0.2637 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1200 | (0.0347, 0.2123) | 0.0017 | 0.1200 | (0.0467, 0.1710) | 0.0007 |
| controlled_vs_proposed_raw | persona_style | -0.0378 | (-0.0703, -0.0117) | 1.0000 | -0.0378 | (-0.0794, 0.0000) | 1.0000 |
| controlled_vs_proposed_raw | distinct1 | -0.0031 | (-0.0260, 0.0182) | 0.6060 | -0.0031 | (-0.0364, 0.0227) | 0.5547 |
| controlled_vs_proposed_raw | length_score | -0.0306 | (-0.1014, 0.0417) | 0.7923 | -0.0306 | (-0.0782, 0.0217) | 0.8720 |
| controlled_vs_proposed_raw | sentence_score | 0.0146 | (-0.0437, 0.0729) | 0.4097 | 0.0146 | (-0.0437, 0.0875) | 0.4377 |
| controlled_vs_proposed_raw | overall_quality | 0.0492 | (0.0107, 0.0921) | 0.0043 | 0.0492 | (0.0190, 0.0799) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.0562 | (0.0103, 0.1095) | 0.0077 | 0.0562 | (0.0131, 0.0967) | 0.0117 |
| controlled_vs_candidate_no_context | persona_consistency | 0.0310 | (-0.0461, 0.1131) | 0.2013 | 0.0310 | (-0.0867, 0.1167) | 0.2573 |
| controlled_vs_candidate_no_context | naturalness | -0.0150 | (-0.0337, 0.0041) | 0.9290 | -0.0150 | (-0.0315, -0.0004) | 0.9853 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.0437 | (0.0050, 0.0916) | 0.0120 | 0.0437 | (0.0097, 0.0772) | 0.0120 |
| controlled_vs_candidate_no_context | lore_consistency | 0.0351 | (0.0075, 0.0688) | 0.0010 | 0.0351 | (0.0055, 0.0602) | 0.0100 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | 0.0039 | (-0.0187, 0.0278) | 0.3713 | 0.0039 | (-0.0177, 0.0332) | 0.3263 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0104 | (-0.0034, 0.0244) | 0.0700 | 0.0104 | (0.0023, 0.0217) | 0.0007 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | -0.0084 | (-0.0270, 0.0093) | 0.8313 | -0.0084 | (-0.0288, 0.0210) | 0.7683 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.0742 | (0.0144, 0.1447) | 0.0043 | 0.0742 | (0.0227, 0.1252) | 0.0130 |
| controlled_vs_candidate_no_context | context_overlap | 0.0140 | (-0.0047, 0.0341) | 0.0780 | 0.0140 | (-0.0094, 0.0331) | 0.0987 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.0417 | (-0.0556, 0.1431) | 0.2110 | 0.0417 | (-0.0837, 0.1458) | 0.2340 |
| controlled_vs_candidate_no_context | persona_style | -0.0117 | (-0.0703, 0.0417) | 0.6537 | -0.0117 | (-0.0325, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | distinct1 | -0.0058 | (-0.0271, 0.0149) | 0.7020 | -0.0058 | (-0.0328, 0.0139) | 0.6830 |
| controlled_vs_candidate_no_context | length_score | -0.0417 | (-0.1306, 0.0458) | 0.8190 | -0.0417 | (-0.1409, 0.0515) | 0.8307 |
| controlled_vs_candidate_no_context | sentence_score | -0.0437 | (-0.1021, 0.0000) | 1.0000 | -0.0437 | (-0.1114, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | overall_quality | 0.0350 | (-0.0046, 0.0781) | 0.0440 | 0.0350 | (-0.0206, 0.0728) | 0.0780 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0203 | (-0.0636, 0.0195) | 0.8317 | -0.0203 | (-0.0430, 0.0081) | 0.9077 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0322 | (-0.1100, 0.0450) | 0.7953 | -0.0322 | (-0.0846, 0.0122) | 0.8747 |
| controlled_alt_vs_controlled_default | naturalness | 0.0115 | (-0.0034, 0.0284) | 0.0760 | 0.0115 | (-0.0044, 0.0275) | 0.0557 |
| controlled_alt_vs_controlled_default | quest_state_correctness | -0.0102 | (-0.0520, 0.0274) | 0.6837 | -0.0102 | (-0.0306, 0.0175) | 0.7593 |
| controlled_alt_vs_controlled_default | lore_consistency | -0.0100 | (-0.0359, 0.0128) | 0.7823 | -0.0100 | (-0.0276, 0.0071) | 0.8653 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | -0.1667 | (-0.3333, -0.0417) | 1.0000 | -0.1667 | (-0.5455, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | objective_completion_support | 0.0035 | (-0.0167, 0.0239) | 0.3593 | 0.0035 | (-0.0094, 0.0149) | 0.2913 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | 0.0043 | (-0.0036, 0.0132) | 0.1563 | 0.0043 | (-0.0038, 0.0127) | 0.1407 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | 0.0083 | (-0.0076, 0.0261) | 0.1610 | 0.0083 | (-0.0052, 0.0172) | 0.0847 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0277 | (-0.0864, 0.0235) | 0.8390 | -0.0277 | (-0.0545, 0.0045) | 0.9120 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0032 | (-0.0168, 0.0106) | 0.6830 | -0.0032 | (-0.0184, 0.0168) | 0.6523 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0383 | (-0.1359, 0.0554) | 0.7923 | -0.0383 | (-0.0923, 0.0139) | 0.8617 |
| controlled_alt_vs_controlled_default | persona_style | -0.0078 | (-0.0391, 0.0234) | 0.7290 | -0.0078 | (-0.0341, 0.0067) | 0.7457 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0028 | (-0.0138, 0.0195) | 0.3700 | 0.0028 | (-0.0082, 0.0169) | 0.3540 |
| controlled_alt_vs_controlled_default | length_score | 0.0444 | (-0.0250, 0.1264) | 0.1217 | 0.0444 | (-0.0136, 0.0847) | 0.0597 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0146 | (-0.0437, 0.0729) | 0.4170 | 0.0146 | (-0.0375, 0.0955) | 0.4330 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0192 | (-0.0543, 0.0169) | 0.8497 | -0.0192 | (-0.0393, 0.0011) | 0.9700 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.0151 | (-0.0280, 0.0567) | 0.2377 | 0.0151 | (-0.0049, 0.0340) | 0.0713 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.0563 | (-0.0018, 0.1198) | 0.0317 | 0.0563 | (0.0229, 0.1030) | 0.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0056 | (-0.0090, 0.0198) | 0.2057 | 0.0056 | (-0.0039, 0.0162) | 0.1520 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 0.0142 | (-0.0244, 0.0522) | 0.2280 | 0.0142 | (-0.0030, 0.0304) | 0.0507 |
| controlled_alt_vs_proposed_raw | lore_consistency | 0.0048 | (-0.0230, 0.0327) | 0.3940 | 0.0048 | (-0.0129, 0.0203) | 0.3273 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | -0.1667 | (-0.3333, -0.0417) | 1.0000 | -0.1667 | (-0.5455, 0.0000) | 1.0000 |
| controlled_alt_vs_proposed_raw | objective_completion_support | 0.0075 | (-0.0181, 0.0334) | 0.2827 | 0.0075 | (-0.0185, 0.0417) | 0.3393 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 0.0083 | (-0.0064, 0.0244) | 0.1517 | 0.0083 | (-0.0027, 0.0259) | 0.1263 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | 0.0056 | (-0.0074, 0.0176) | 0.1963 | 0.0056 | (-0.0093, 0.0233) | 0.2340 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.0205 | (-0.0371, 0.0742) | 0.2443 | 0.0205 | (-0.0029, 0.0420) | 0.0343 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0026 | (-0.0157, 0.0197) | 0.3857 | 0.0026 | (-0.0142, 0.0166) | 0.3813 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.0817 | (0.0097, 0.1562) | 0.0120 | 0.0817 | (0.0415, 0.1381) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_style | -0.0456 | (-0.0872, -0.0078) | 0.9930 | -0.0456 | (-0.1125, 0.0000) | 1.0000 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0003 | (-0.0205, 0.0197) | 0.4957 | -0.0003 | (-0.0270, 0.0203) | 0.5183 |
| controlled_alt_vs_proposed_raw | length_score | 0.0139 | (-0.0375, 0.0639) | 0.2963 | 0.0139 | (-0.0278, 0.0683) | 0.2910 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0292 | (-0.0437, 0.1021) | 0.2630 | 0.0292 | (-0.0538, 0.1273) | 0.3020 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.0300 | (-0.0006, 0.0630) | 0.0293 | 0.0300 | (0.0151, 0.0477) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.0358 | (0.0010, 0.0751) | 0.0230 | 0.0358 | (0.0191, 0.0594) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | -0.0012 | (-0.0850, 0.0855) | 0.5190 | -0.0012 | (-0.1285, 0.0826) | 0.5243 |
| controlled_alt_vs_candidate_no_context | naturalness | -0.0036 | (-0.0238, 0.0176) | 0.6373 | -0.0036 | (-0.0184, 0.0090) | 0.6860 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 0.0335 | (0.0023, 0.0694) | 0.0177 | 0.0335 | (0.0203, 0.0579) | 0.0000 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 0.0251 | (0.0060, 0.0472) | 0.0033 | 0.0251 | (0.0115, 0.0358) | 0.0000 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | -0.1667 | (-0.3333, -0.0417) | 1.0000 | -0.1667 | (-0.5455, 0.0000) | 1.0000 |
| controlled_alt_vs_candidate_no_context | objective_completion_support | 0.0074 | (-0.0215, 0.0356) | 0.3080 | 0.0074 | (-0.0111, 0.0319) | 0.2393 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 0.0147 | (0.0017, 0.0280) | 0.0133 | 0.0147 | (0.0085, 0.0234) | 0.0000 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | -0.0001 | (-0.0163, 0.0181) | 0.5210 | -0.0001 | (-0.0121, 0.0160) | 0.5313 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.0466 | (0.0008, 0.0970) | 0.0207 | 0.0466 | (0.0256, 0.0786) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0108 | (-0.0037, 0.0262) | 0.0720 | 0.0108 | (-0.0022, 0.0185) | 0.0443 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.0034 | (-0.0984, 0.1032) | 0.4703 | 0.0034 | (-0.1512, 0.1076) | 0.4433 |
| controlled_alt_vs_candidate_no_context | persona_style | -0.0195 | (-0.0742, 0.0260) | 0.7723 | -0.0195 | (-0.0440, 0.0000) | 1.0000 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0030 | (-0.0239, 0.0174) | 0.6023 | -0.0030 | (-0.0232, 0.0114) | 0.6277 |
| controlled_alt_vs_candidate_no_context | length_score | 0.0028 | (-0.0708, 0.0750) | 0.4853 | 0.0028 | (-0.0611, 0.0667) | 0.4583 |
| controlled_alt_vs_candidate_no_context | sentence_score | -0.0292 | (-0.0729, 0.0000) | 1.0000 | -0.0292 | (-0.0625, 0.0000) | 1.0000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.0158 | (-0.0217, 0.0540) | 0.2037 | 0.0158 | (-0.0349, 0.0477) | 0.2607 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 9 | 7 | 8 | 0.5417 | 0.5625 |
| proposed_vs_candidate_no_context | persona_consistency | 4 | 8 | 12 | 0.4167 | 0.3333 |
| proposed_vs_candidate_no_context | naturalness | 7 | 9 | 8 | 0.4583 | 0.4375 |
| proposed_vs_candidate_no_context | quest_state_correctness | 8 | 8 | 8 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | lore_consistency | 7 | 2 | 15 | 0.6042 | 0.7778 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| proposed_vs_candidate_no_context | objective_completion_support | 6 | 8 | 10 | 0.4583 | 0.4286 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 9 | 7 | 8 | 0.5417 | 0.5625 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | 6 | 9 | 9 | 0.4375 | 0.4000 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 5 | 3 | 16 | 0.5417 | 0.6250 |
| proposed_vs_candidate_no_context | context_overlap | 9 | 7 | 8 | 0.5417 | 0.5625 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 3 | 8 | 13 | 0.3958 | 0.2727 |
| proposed_vs_candidate_no_context | persona_style | 5 | 2 | 17 | 0.5625 | 0.7143 |
| proposed_vs_candidate_no_context | distinct1 | 8 | 8 | 8 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | length_score | 7 | 9 | 8 | 0.4583 | 0.4375 |
| proposed_vs_candidate_no_context | sentence_score | 0 | 4 | 20 | 0.4167 | 0.0000 |
| proposed_vs_candidate_no_context | overall_quality | 7 | 9 | 8 | 0.4583 | 0.4375 |
| controlled_vs_proposed_raw | context_relevance | 12 | 7 | 5 | 0.6042 | 0.6316 |
| controlled_vs_proposed_raw | persona_consistency | 12 | 5 | 7 | 0.6458 | 0.7059 |
| controlled_vs_proposed_raw | naturalness | 7 | 12 | 5 | 0.3958 | 0.3684 |
| controlled_vs_proposed_raw | quest_state_correctness | 11 | 8 | 5 | 0.5625 | 0.5789 |
| controlled_vs_proposed_raw | lore_consistency | 6 | 5 | 13 | 0.5208 | 0.5455 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_vs_proposed_raw | objective_completion_support | 6 | 7 | 11 | 0.4792 | 0.4615 |
| controlled_vs_proposed_raw | gameplay_usefulness | 7 | 12 | 5 | 0.3958 | 0.3684 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 6 | 8 | 10 | 0.4583 | 0.4286 |
| controlled_vs_proposed_raw | context_keyword_coverage | 9 | 4 | 11 | 0.6042 | 0.6923 |
| controlled_vs_proposed_raw | context_overlap | 10 | 9 | 5 | 0.5208 | 0.5263 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 12 | 4 | 8 | 0.6667 | 0.7500 |
| controlled_vs_proposed_raw | persona_style | 0 | 6 | 18 | 0.3750 | 0.0000 |
| controlled_vs_proposed_raw | distinct1 | 9 | 10 | 5 | 0.4792 | 0.4737 |
| controlled_vs_proposed_raw | length_score | 8 | 10 | 6 | 0.4583 | 0.4444 |
| controlled_vs_proposed_raw | sentence_score | 3 | 2 | 19 | 0.5208 | 0.6000 |
| controlled_vs_proposed_raw | overall_quality | 13 | 6 | 5 | 0.6458 | 0.6842 |
| controlled_vs_candidate_no_context | context_relevance | 10 | 9 | 5 | 0.5208 | 0.5263 |
| controlled_vs_candidate_no_context | persona_consistency | 10 | 5 | 9 | 0.6042 | 0.6667 |
| controlled_vs_candidate_no_context | naturalness | 6 | 13 | 5 | 0.3542 | 0.3158 |
| controlled_vs_candidate_no_context | quest_state_correctness | 10 | 9 | 5 | 0.5208 | 0.5263 |
| controlled_vs_candidate_no_context | lore_consistency | 7 | 3 | 14 | 0.5833 | 0.7000 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_vs_candidate_no_context | objective_completion_support | 7 | 7 | 10 | 0.5000 | 0.5000 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 12 | 7 | 5 | 0.6042 | 0.6316 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 5 | 11 | 8 | 0.3750 | 0.3125 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 8 | 3 | 13 | 0.6042 | 0.7273 |
| controlled_vs_candidate_no_context | context_overlap | 9 | 10 | 5 | 0.4792 | 0.4737 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 9 | 4 | 11 | 0.6042 | 0.6923 |
| controlled_vs_candidate_no_context | persona_style | 4 | 4 | 16 | 0.5000 | 0.5000 |
| controlled_vs_candidate_no_context | distinct1 | 10 | 9 | 5 | 0.5208 | 0.5263 |
| controlled_vs_candidate_no_context | length_score | 9 | 9 | 6 | 0.5000 | 0.5000 |
| controlled_vs_candidate_no_context | sentence_score | 0 | 3 | 21 | 0.4375 | 0.0000 |
| controlled_vs_candidate_no_context | overall_quality | 14 | 5 | 5 | 0.6875 | 0.7368 |
| controlled_alt_vs_controlled_default | context_relevance | 7 | 7 | 10 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | persona_consistency | 3 | 8 | 13 | 0.3958 | 0.2727 |
| controlled_alt_vs_controlled_default | naturalness | 9 | 5 | 10 | 0.5833 | 0.6429 |
| controlled_alt_vs_controlled_default | quest_state_correctness | 7 | 7 | 10 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | lore_consistency | 5 | 4 | 15 | 0.5208 | 0.5556 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0 | 4 | 20 | 0.4167 | 0.0000 |
| controlled_alt_vs_controlled_default | objective_completion_support | 7 | 5 | 12 | 0.5417 | 0.5833 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | 8 | 6 | 10 | 0.5417 | 0.5714 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | 7 | 7 | 10 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 6 | 7 | 11 | 0.4792 | 0.4615 |
| controlled_alt_vs_controlled_default | context_overlap | 5 | 8 | 11 | 0.4375 | 0.3846 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 3 | 7 | 14 | 0.4167 | 0.3000 |
| controlled_alt_vs_controlled_default | persona_style | 1 | 3 | 20 | 0.4583 | 0.2500 |
| controlled_alt_vs_controlled_default | distinct1 | 7 | 7 | 10 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | length_score | 7 | 7 | 10 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | sentence_score | 3 | 2 | 19 | 0.5208 | 0.6000 |
| controlled_alt_vs_controlled_default | overall_quality | 5 | 9 | 10 | 0.4167 | 0.3571 |
| controlled_alt_vs_proposed_raw | context_relevance | 12 | 6 | 6 | 0.6250 | 0.6667 |
| controlled_alt_vs_proposed_raw | persona_consistency | 9 | 7 | 8 | 0.5417 | 0.5625 |
| controlled_alt_vs_proposed_raw | naturalness | 10 | 8 | 6 | 0.5417 | 0.5556 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 11 | 7 | 6 | 0.5833 | 0.6111 |
| controlled_alt_vs_proposed_raw | lore_consistency | 6 | 4 | 14 | 0.5417 | 0.6000 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0 | 4 | 20 | 0.4167 | 0.0000 |
| controlled_alt_vs_proposed_raw | objective_completion_support | 6 | 4 | 14 | 0.5417 | 0.6000 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 9 | 9 | 6 | 0.5000 | 0.5000 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | 8 | 4 | 12 | 0.5833 | 0.6667 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 8 | 5 | 11 | 0.5625 | 0.6154 |
| controlled_alt_vs_proposed_raw | context_overlap | 10 | 8 | 6 | 0.5417 | 0.5556 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 9 | 4 | 11 | 0.6042 | 0.6923 |
| controlled_alt_vs_proposed_raw | persona_style | 1 | 7 | 16 | 0.3750 | 0.1250 |
| controlled_alt_vs_proposed_raw | distinct1 | 10 | 8 | 6 | 0.5417 | 0.5556 |
| controlled_alt_vs_proposed_raw | length_score | 10 | 8 | 6 | 0.5417 | 0.5556 |
| controlled_alt_vs_proposed_raw | sentence_score | 4 | 2 | 18 | 0.5417 | 0.6667 |
| controlled_alt_vs_proposed_raw | overall_quality | 10 | 8 | 6 | 0.5417 | 0.5556 |
| controlled_alt_vs_candidate_no_context | context_relevance | 11 | 7 | 6 | 0.5833 | 0.6111 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 7 | 8 | 9 | 0.4792 | 0.4667 |
| controlled_alt_vs_candidate_no_context | naturalness | 7 | 10 | 7 | 0.4375 | 0.4118 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 10 | 8 | 6 | 0.5417 | 0.5556 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 7 | 2 | 15 | 0.6042 | 0.7778 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 4 | 20 | 0.4167 | 0.0000 |
| controlled_alt_vs_candidate_no_context | objective_completion_support | 7 | 6 | 11 | 0.5208 | 0.5385 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 13 | 5 | 6 | 0.6667 | 0.7222 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | 7 | 9 | 8 | 0.4583 | 0.4375 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 8 | 5 | 11 | 0.5625 | 0.6154 |
| controlled_alt_vs_candidate_no_context | context_overlap | 10 | 8 | 6 | 0.5417 | 0.5556 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 7 | 7 | 10 | 0.5000 | 0.5000 |
| controlled_alt_vs_candidate_no_context | persona_style | 3 | 4 | 17 | 0.4792 | 0.4286 |
| controlled_alt_vs_candidate_no_context | distinct1 | 9 | 8 | 7 | 0.5208 | 0.5294 |
| controlled_alt_vs_candidate_no_context | length_score | 9 | 7 | 8 | 0.5417 | 0.5625 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0 | 2 | 22 | 0.4583 | 0.0000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 13 | 5 | 6 | 0.6667 | 0.7222 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.1250 | 0.2083 | 0.7917 |
| proposed_contextual_controlled_tuned | 0.0000 | 0.0000 | 0.1667 | 0.1667 | 0.7917 |
| proposed_contextual | 0.0000 | 0.0000 | 0.5833 | 0.0000 | 0.0000 |
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
| proposed_contextual_controlled_tuned | 0.2000 | 0.8000 | 1 | 5 |
| proposed_contextual | 0.0000 | 1.0000 | 0 | 5 |
| candidate_no_context | 0.0000 | 1.0000 | 0 | 5 |
- Detailed source-level values are published in `multi_turn_contradictions.json`.

- BERTScore status: unavailable (disabled_by_flag).

This report covers proposal RO5 metrics plus game-facing outcomes: quest-state correctness, lore consistency, contradiction safety, objective completion support, gameplay usefulness, and time-pressure acceptability.
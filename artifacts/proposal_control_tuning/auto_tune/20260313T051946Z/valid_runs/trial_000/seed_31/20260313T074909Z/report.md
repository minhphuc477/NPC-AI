# Proposal Alignment Evaluation Report

- Run ID: `20260313T074909Z`
- Generated: `2026-03-13T07:55:08.314121+00:00`
- Scenarios: `artifacts\proposal_control_tuning\auto_tune\20260313T051946Z\valid_runs\trial_000\seed_31\20260313T074909Z\scenarios.jsonl`
- Scenario count: `24`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_tuned`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.1375 (0.0857, 0.2022) | 0.3268 (0.2769, 0.3752) | 0.8728 (0.8562, 0.8891) | 0.3500 (0.3188, 0.3853) | n/a |
| proposed_contextual_controlled_tuned | 0.1103 (0.0724, 0.1546) | 0.3438 (0.2925, 0.3958) | 0.8660 (0.8559, 0.8745) | 0.3425 (0.3137, 0.3723) | n/a |
| proposed_contextual | 0.0907 (0.0618, 0.1238) | 0.2418 (0.1869, 0.2990) | 0.8720 (0.8571, 0.8866) | 0.2964 (0.2731, 0.3190) | n/a |
| candidate_no_context | 0.0427 (0.0260, 0.0611) | 0.2885 (0.2285, 0.3589) | 0.8746 (0.8646, 0.8858) | 0.2928 (0.2654, 0.3255) | n/a |

## Game-facing Outcome Metrics (mean, 95% CI)
| Arm | Quest-state Correctness | Lore Consistency | Contradiction Safety | Objective Completion Support | Gameplay Usefulness | Time-pressure Acceptability |
|---|---:|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2294 (0.1794, 0.2845) | 0.0832 (0.0361, 0.1406) | 1.0000 (1.0000, 1.0000) | 0.1051 (0.0612, 0.1473) | 0.3207 (0.3013, 0.3391) | 0.3118 (0.2826, 0.3411) |
| proposed_contextual_controlled_tuned | 0.2019 (0.1676, 0.2403) | 0.0546 (0.0189, 0.0938) | 0.6667 (0.4583, 0.8333) | 0.1159 (0.0746, 0.1604) | 0.3160 (0.2982, 0.3339) | 0.3221 (0.2929, 0.3511) |
| proposed_contextual | 0.1799 (0.1538, 0.2080) | 0.0497 (0.0199, 0.0832) | 1.0000 (1.0000, 1.0000) | 0.0908 (0.0427, 0.1409) | 0.3048 (0.2802, 0.3302) | 0.3130 (0.2830, 0.3457) |
| candidate_no_context | 0.1380 (0.1216, 0.1550) | 0.0063 (0.0007, 0.0148) | 1.0000 (1.0000, 1.0000) | 0.0892 (0.0482, 0.1361) | 0.2907 (0.2685, 0.3155) | 0.3097 (0.2834, 0.3387) |

- Multi-turn contradiction rate is reported as `1 - contradiction_safety` in row-level outputs.

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0480 | 1.1225 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0467 | -0.1619 |
| proposed_vs_candidate_no_context | naturalness | -0.0027 | -0.0031 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.0419 | 0.3037 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0434 | 6.8702 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | 0.0016 | 0.0175 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 0.0141 | 0.0484 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | 0.0033 | 0.0107 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0600 | 1.3571 |
| proposed_vs_candidate_no_context | context_overlap | 0.0199 | 0.5067 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0647 | -0.3141 |
| proposed_vs_candidate_no_context | persona_style | 0.0252 | 0.0407 |
| proposed_vs_candidate_no_context | distinct1 | 0.0027 | 0.0028 |
| proposed_vs_candidate_no_context | length_score | -0.0042 | -0.0084 |
| proposed_vs_candidate_no_context | sentence_score | -0.0292 | -0.0300 |
| proposed_vs_candidate_no_context | overall_quality | 0.0036 | 0.0124 |
| controlled_vs_proposed_raw | context_relevance | 0.0468 | 0.5160 |
| controlled_vs_proposed_raw | persona_consistency | 0.0849 | 0.3512 |
| controlled_vs_proposed_raw | naturalness | 0.0008 | 0.0009 |
| controlled_vs_proposed_raw | quest_state_correctness | 0.0494 | 0.2748 |
| controlled_vs_proposed_raw | lore_consistency | 0.0335 | 0.6739 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0144 | 0.1584 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0160 | 0.0524 |
| controlled_vs_proposed_raw | time_pressure_acceptability | -0.0012 | -0.0039 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.0616 | 0.5909 |
| controlled_vs_proposed_raw | context_overlap | 0.0123 | 0.2086 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1032 | 0.7303 |
| controlled_vs_proposed_raw | persona_style | 0.0120 | 0.0187 |
| controlled_vs_proposed_raw | distinct1 | 0.0064 | 0.0067 |
| controlled_vs_proposed_raw | length_score | -0.0306 | -0.0621 |
| controlled_vs_proposed_raw | sentence_score | 0.0437 | 0.0465 |
| controlled_vs_proposed_raw | overall_quality | 0.0536 | 0.1809 |
| controlled_vs_candidate_no_context | context_relevance | 0.0947 | 2.2178 |
| controlled_vs_candidate_no_context | persona_consistency | 0.0382 | 0.1325 |
| controlled_vs_candidate_no_context | naturalness | -0.0019 | -0.0021 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.0914 | 0.6620 |
| controlled_vs_candidate_no_context | lore_consistency | 0.0769 | 12.1738 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | 0.0159 | 0.1787 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0300 | 0.1033 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 0.0021 | 0.0068 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.1215 | 2.7500 |
| controlled_vs_candidate_no_context | context_overlap | 0.0323 | 0.8210 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.0385 | 0.1869 |
| controlled_vs_candidate_no_context | persona_style | 0.0372 | 0.0601 |
| controlled_vs_candidate_no_context | distinct1 | 0.0091 | 0.0096 |
| controlled_vs_candidate_no_context | length_score | -0.0347 | -0.0700 |
| controlled_vs_candidate_no_context | sentence_score | 0.0146 | 0.0150 |
| controlled_vs_candidate_no_context | overall_quality | 0.0572 | 0.1955 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0271 | -0.1973 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0170 | 0.0522 |
| controlled_alt_vs_controlled_default | naturalness | -0.0068 | -0.0078 |
| controlled_alt_vs_controlled_default | quest_state_correctness | -0.0274 | -0.1196 |
| controlled_alt_vs_controlled_default | lore_consistency | -0.0286 | -0.3436 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | -0.3333 | -0.3333 |
| controlled_alt_vs_controlled_default | objective_completion_support | 0.0107 | 0.1020 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | -0.0048 | -0.0149 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | 0.0103 | 0.0331 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0294 | -0.1771 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0219 | -0.3065 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0139 | 0.0568 |
| controlled_alt_vs_controlled_default | persona_style | 0.0297 | 0.0452 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0025 | -0.0026 |
| controlled_alt_vs_controlled_default | length_score | -0.0292 | -0.0633 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0000 | 0.0000 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0075 | -0.0215 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.0197 | 0.2169 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1020 | 0.4217 |
| controlled_alt_vs_proposed_raw | naturalness | -0.0060 | -0.0069 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 0.0220 | 0.1223 |
| controlled_alt_vs_proposed_raw | lore_consistency | 0.0049 | 0.0987 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | -0.3333 | -0.3333 |
| controlled_alt_vs_proposed_raw | objective_completion_support | 0.0251 | 0.2766 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 0.0112 | 0.0368 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | 0.0091 | 0.0291 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.0322 | 0.3091 |
| controlled_alt_vs_proposed_raw | context_overlap | -0.0096 | -0.1618 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1171 | 0.8287 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0417 | 0.0648 |
| controlled_alt_vs_proposed_raw | distinct1 | 0.0039 | 0.0041 |
| controlled_alt_vs_proposed_raw | length_score | -0.0597 | -0.1215 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0437 | 0.0465 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.0461 | 0.1555 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.0676 | 1.5829 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.0553 | 0.1916 |
| controlled_alt_vs_candidate_no_context | naturalness | -0.0087 | -0.0099 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 0.0639 | 0.4631 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 0.0483 | 7.6467 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | -0.3333 | -0.3333 |
| controlled_alt_vs_candidate_no_context | objective_completion_support | 0.0267 | 0.2989 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 0.0253 | 0.0869 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | 0.0124 | 0.0401 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.0922 | 2.0857 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0103 | 0.2629 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.0524 | 0.2543 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0669 | 0.1081 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0066 | 0.0070 |
| controlled_alt_vs_candidate_no_context | length_score | -0.0639 | -0.1289 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.0146 | 0.0150 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.0497 | 0.1698 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0480 | (0.0151, 0.0835) | 0.0013 | 0.0480 | (-0.0087, 0.1323) | 0.0407 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0467 | (-0.1139, 0.0089) | 0.9470 | -0.0467 | (-0.0780, -0.0286) | 1.0000 |
| proposed_vs_candidate_no_context | naturalness | -0.0027 | (-0.0186, 0.0138) | 0.6153 | -0.0027 | (-0.0127, 0.0025) | 0.7120 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.0419 | (0.0138, 0.0757) | 0.0007 | 0.0419 | (-0.0079, 0.1157) | 0.0380 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0434 | (0.0152, 0.0748) | 0.0000 | 0.0434 | (0.0079, 0.1097) | 0.0000 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | 0.0016 | (-0.0165, 0.0251) | 0.4690 | 0.0016 | (-0.0183, 0.0207) | 0.3997 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 0.0141 | (0.0005, 0.0295) | 0.0223 | 0.0141 | (0.0031, 0.0341) | 0.0000 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | 0.0033 | (-0.0079, 0.0152) | 0.2947 | 0.0033 | (-0.0063, 0.0155) | 0.3023 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0600 | (0.0174, 0.1089) | 0.0010 | 0.0600 | (-0.0114, 0.1667) | 0.0397 |
| proposed_vs_candidate_no_context | context_overlap | 0.0199 | (0.0050, 0.0365) | 0.0040 | 0.0199 | (-0.0025, 0.0520) | 0.0337 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0647 | (-0.1472, 0.0056) | 0.9663 | -0.0647 | (-0.0952, -0.0357) | 1.0000 |
| proposed_vs_candidate_no_context | persona_style | 0.0252 | (-0.0032, 0.0624) | 0.0527 | 0.0252 | (-0.0089, 0.0741) | 0.3000 |
| proposed_vs_candidate_no_context | distinct1 | 0.0027 | (-0.0086, 0.0134) | 0.3060 | 0.0027 | (-0.0110, 0.0097) | 0.2550 |
| proposed_vs_candidate_no_context | length_score | -0.0042 | (-0.0806, 0.0722) | 0.5550 | -0.0042 | (-0.0417, 0.0190) | 0.7087 |
| proposed_vs_candidate_no_context | sentence_score | -0.0292 | (-0.0875, 0.0292) | 0.9197 | -0.0292 | (-0.0500, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | overall_quality | 0.0036 | (-0.0273, 0.0299) | 0.3833 | 0.0036 | (-0.0173, 0.0323) | 0.3947 |
| controlled_vs_proposed_raw | context_relevance | 0.0468 | (-0.0119, 0.1138) | 0.0700 | 0.0468 | (-0.0320, 0.0940) | 0.0343 |
| controlled_vs_proposed_raw | persona_consistency | 0.0849 | (0.0353, 0.1398) | 0.0003 | 0.0849 | (0.0090, 0.1572) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0008 | (-0.0206, 0.0223) | 0.4710 | 0.0008 | (-0.0238, 0.0199) | 0.3647 |
| controlled_vs_proposed_raw | quest_state_correctness | 0.0494 | (-0.0075, 0.1170) | 0.0467 | 0.0494 | (-0.0206, 0.0866) | 0.0343 |
| controlled_vs_proposed_raw | lore_consistency | 0.0335 | (-0.0193, 0.0936) | 0.1317 | 0.0335 | (-0.0602, 0.0882) | 0.2627 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0144 | (-0.0267, 0.0583) | 0.2613 | 0.0144 | (-0.0027, 0.0288) | 0.0337 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0160 | (-0.0091, 0.0430) | 0.1130 | 0.0160 | (-0.0052, 0.0391) | 0.1530 |
| controlled_vs_proposed_raw | time_pressure_acceptability | -0.0012 | (-0.0306, 0.0267) | 0.5233 | -0.0012 | (-0.0155, 0.0206) | 0.6073 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.0616 | (-0.0123, 0.1490) | 0.0560 | 0.0616 | (-0.0357, 0.1212) | 0.0367 |
| controlled_vs_proposed_raw | context_overlap | 0.0123 | (-0.0127, 0.0406) | 0.1807 | 0.0123 | (-0.0233, 0.0305) | 0.2660 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1032 | (0.0391, 0.1728) | 0.0003 | 0.1032 | (0.0179, 0.1905) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.0120 | (-0.0315, 0.0639) | 0.3230 | 0.0120 | (-0.0266, 0.0370) | 0.2603 |
| controlled_vs_proposed_raw | distinct1 | 0.0064 | (-0.0096, 0.0218) | 0.1983 | 0.0064 | (-0.0100, 0.0205) | 0.1430 |
| controlled_vs_proposed_raw | length_score | -0.0306 | (-0.1278, 0.0708) | 0.7147 | -0.0306 | (-0.1238, 0.0667) | 0.7360 |
| controlled_vs_proposed_raw | sentence_score | 0.0437 | (-0.0146, 0.1021) | 0.1267 | 0.0437 | (0.0389, 0.0500) | 0.0000 |
| controlled_vs_proposed_raw | overall_quality | 0.0536 | (0.0236, 0.0902) | 0.0000 | 0.0536 | (0.0328, 0.0831) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.0947 | (0.0382, 0.1631) | 0.0003 | 0.0947 | (0.0539, 0.1267) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.0382 | (-0.0216, 0.0957) | 0.1050 | 0.0382 | (-0.0196, 0.0792) | 0.0377 |
| controlled_vs_candidate_no_context | naturalness | -0.0019 | (-0.0226, 0.0177) | 0.5740 | -0.0019 | (-0.0213, 0.0221) | 0.6173 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.0914 | (0.0405, 0.1532) | 0.0000 | 0.0914 | (0.0610, 0.1154) | 0.0000 |
| controlled_vs_candidate_no_context | lore_consistency | 0.0769 | (0.0282, 0.1358) | 0.0000 | 0.0769 | (0.0495, 0.1116) | 0.0000 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | 0.0159 | (-0.0265, 0.0600) | 0.2127 | 0.0159 | (0.0105, 0.0183) | 0.0000 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0300 | (0.0087, 0.0512) | 0.0023 | 0.0300 | (0.0115, 0.0473) | 0.0000 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 0.0021 | (-0.0284, 0.0326) | 0.4503 | 0.0021 | (-0.0055, 0.0143) | 0.2580 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.1215 | (0.0489, 0.2061) | 0.0000 | 0.1215 | (0.0682, 0.1616) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0323 | (0.0106, 0.0591) | 0.0000 | 0.0323 | (0.0205, 0.0454) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.0385 | (-0.0397, 0.1066) | 0.1457 | 0.0385 | (-0.0179, 0.0952) | 0.0300 |
| controlled_vs_candidate_no_context | persona_style | 0.0372 | (-0.0072, 0.0898) | 0.0577 | 0.0372 | (-0.0266, 0.1111) | 0.2623 |
| controlled_vs_candidate_no_context | distinct1 | 0.0091 | (-0.0056, 0.0241) | 0.1080 | 0.0091 | (-0.0008, 0.0163) | 0.0363 |
| controlled_vs_candidate_no_context | length_score | -0.0347 | (-0.1292, 0.0583) | 0.7590 | -0.0347 | (-0.1048, 0.0778) | 0.7327 |
| controlled_vs_candidate_no_context | sentence_score | 0.0146 | (-0.0292, 0.0583) | 0.4000 | 0.0146 | (0.0000, 0.0437) | 0.2847 |
| controlled_vs_candidate_no_context | overall_quality | 0.0572 | (0.0189, 0.0978) | 0.0010 | 0.0572 | (0.0155, 0.0830) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0271 | (-0.0865, 0.0319) | 0.8147 | -0.0271 | (-0.0313, -0.0221) | 1.0000 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0170 | (-0.0360, 0.0727) | 0.2677 | 0.0170 | (0.0053, 0.0381) | 0.0000 |
| controlled_alt_vs_controlled_default | naturalness | -0.0068 | (-0.0249, 0.0094) | 0.7783 | -0.0068 | (-0.0181, 0.0053) | 0.8527 |
| controlled_alt_vs_controlled_default | quest_state_correctness | -0.0274 | (-0.0816, 0.0253) | 0.8423 | -0.0274 | (-0.0290, -0.0262) | 1.0000 |
| controlled_alt_vs_controlled_default | lore_consistency | -0.0286 | (-0.0834, 0.0270) | 0.8457 | -0.0286 | (-0.0456, -0.0066) | 1.0000 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | -0.3333 | (-0.5417, -0.1667) | 1.0000 | -0.3333 | (-1.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | objective_completion_support | 0.0107 | (-0.0196, 0.0422) | 0.2473 | 0.0107 | (-0.0046, 0.0407) | 0.2607 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | -0.0048 | (-0.0195, 0.0089) | 0.7503 | -0.0048 | (-0.0151, 0.0064) | 0.8497 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | 0.0103 | (-0.0123, 0.0356) | 0.2023 | 0.0103 | (-0.0003, 0.0178) | 0.0357 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0294 | (-0.1004, 0.0521) | 0.7713 | -0.0294 | (-0.0357, -0.0202) | 1.0000 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0219 | (-0.0458, -0.0019) | 0.9853 | -0.0219 | (-0.0265, -0.0176) | 1.0000 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0139 | (-0.0389, 0.0661) | 0.3053 | 0.0139 | (0.0000, 0.0476) | 0.3197 |
| controlled_alt_vs_controlled_default | persona_style | 0.0297 | (-0.0347, 0.1068) | 0.2117 | 0.0297 | (0.0000, 0.0556) | 0.0373 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0025 | (-0.0165, 0.0125) | 0.6220 | -0.0025 | (-0.0119, 0.0084) | 0.6277 |
| controlled_alt_vs_controlled_default | length_score | -0.0292 | (-0.1139, 0.0528) | 0.7547 | -0.0292 | (-0.0667, 0.0292) | 0.8460 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0000 | (-0.0437, 0.0437) | 0.6403 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0075 | (-0.0404, 0.0293) | 0.6857 | -0.0075 | (-0.0106, -0.0011) | 1.0000 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.0197 | (-0.0216, 0.0697) | 0.1920 | 0.0197 | (-0.0633, 0.0719) | 0.2623 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1020 | (0.0461, 0.1653) | 0.0000 | 0.1020 | (0.0143, 0.1953) | 0.0000 |
| controlled_alt_vs_proposed_raw | naturalness | -0.0060 | (-0.0265, 0.0116) | 0.7123 | -0.0060 | (-0.0299, 0.0062) | 0.6967 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 0.0220 | (-0.0162, 0.0625) | 0.1357 | 0.0220 | (-0.0467, 0.0595) | 0.2560 |
| controlled_alt_vs_proposed_raw | lore_consistency | 0.0049 | (-0.0334, 0.0463) | 0.4290 | 0.0049 | (-0.0668, 0.0577) | 0.3747 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | -0.3333 | (-0.5417, -0.1667) | 1.0000 | -0.3333 | (-1.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_proposed_raw | objective_completion_support | 0.0251 | (-0.0127, 0.0618) | 0.0903 | 0.0251 | (-0.0009, 0.0695) | 0.0370 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 0.0112 | (-0.0061, 0.0307) | 0.1180 | 0.0112 | (0.0012, 0.0240) | 0.0000 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | 0.0091 | (-0.0130, 0.0318) | 0.2160 | 0.0091 | (-0.0058, 0.0384) | 0.2623 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.0322 | (-0.0231, 0.0963) | 0.1497 | 0.0322 | (-0.0714, 0.1010) | 0.2597 |
| controlled_alt_vs_proposed_raw | context_overlap | -0.0096 | (-0.0298, 0.0057) | 0.8657 | -0.0096 | (-0.0443, 0.0055) | 0.7003 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1171 | (0.0504, 0.1905) | 0.0000 | 0.1171 | (0.0179, 0.2381) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0417 | (-0.0175, 0.1178) | 0.1030 | 0.0417 | (0.0000, 0.0926) | 0.0357 |
| controlled_alt_vs_proposed_raw | distinct1 | 0.0039 | (-0.0105, 0.0162) | 0.2903 | 0.0039 | (-0.0053, 0.0192) | 0.2947 |
| controlled_alt_vs_proposed_raw | length_score | -0.0597 | (-0.1611, 0.0306) | 0.8997 | -0.0597 | (-0.1714, -0.0000) | 1.0000 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0437 | (-0.0146, 0.1021) | 0.1187 | 0.0437 | (0.0389, 0.0500) | 0.0000 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.0461 | (0.0196, 0.0798) | 0.0000 | 0.0461 | (0.0223, 0.0732) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.0676 | (0.0230, 0.1215) | 0.0000 | 0.0676 | (0.0248, 0.1047) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.0553 | (-0.0067, 0.1199) | 0.0400 | 0.0553 | (-0.0143, 0.1173) | 0.0350 |
| controlled_alt_vs_candidate_no_context | naturalness | -0.0087 | (-0.0239, 0.0058) | 0.8543 | -0.0087 | (-0.0274, 0.0040) | 0.8537 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 0.0639 | (0.0259, 0.1066) | 0.0003 | 0.0639 | (0.0320, 0.0883) | 0.0000 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 0.0483 | (0.0099, 0.0909) | 0.0050 | 0.0483 | (0.0161, 0.0811) | 0.0000 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | -0.3333 | (-0.5417, -0.1250) | 1.0000 | -0.3333 | (-1.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_candidate_no_context | objective_completion_support | 0.0267 | (0.0014, 0.0556) | 0.0200 | 0.0267 | (0.0137, 0.0512) | 0.0000 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 0.0253 | (0.0100, 0.0411) | 0.0010 | 0.0253 | (0.0086, 0.0353) | 0.0000 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | 0.0124 | (-0.0065, 0.0341) | 0.1077 | 0.0124 | (-0.0058, 0.0321) | 0.0377 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.0922 | (0.0328, 0.1616) | 0.0003 | 0.0922 | (0.0341, 0.1414) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0103 | (-0.0026, 0.0233) | 0.0607 | 0.0103 | (0.0030, 0.0189) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.0524 | (-0.0250, 0.1236) | 0.0853 | 0.0524 | (-0.0179, 0.1429) | 0.0413 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0669 | (0.0046, 0.1494) | 0.0103 | 0.0669 | (0.0000, 0.1667) | 0.0410 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0066 | (-0.0053, 0.0176) | 0.1200 | 0.0066 | (0.0044, 0.0083) | 0.0000 |
| controlled_alt_vs_candidate_no_context | length_score | -0.0639 | (-0.1473, 0.0097) | 0.9540 | -0.0639 | (-0.1524, 0.0111) | 0.9610 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.0146 | (0.0000, 0.0437) | 0.3470 | 0.0146 | (0.0000, 0.0437) | 0.3003 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.0497 | (0.0138, 0.0912) | 0.0033 | 0.0497 | (0.0050, 0.0731) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 10 | 3 | 11 | 0.6458 | 0.7692 |
| proposed_vs_candidate_no_context | persona_consistency | 2 | 6 | 16 | 0.4167 | 0.2500 |
| proposed_vs_candidate_no_context | naturalness | 5 | 7 | 12 | 0.4583 | 0.4167 |
| proposed_vs_candidate_no_context | quest_state_correctness | 10 | 3 | 11 | 0.6458 | 0.7692 |
| proposed_vs_candidate_no_context | lore_consistency | 9 | 1 | 14 | 0.6667 | 0.9000 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| proposed_vs_candidate_no_context | objective_completion_support | 2 | 4 | 18 | 0.4583 | 0.3333 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 10 | 3 | 11 | 0.6458 | 0.7692 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | 5 | 3 | 16 | 0.5417 | 0.6250 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 9 | 2 | 13 | 0.6458 | 0.8182 |
| proposed_vs_candidate_no_context | context_overlap | 10 | 3 | 11 | 0.6458 | 0.7692 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 1 | 6 | 17 | 0.3958 | 0.1429 |
| proposed_vs_candidate_no_context | persona_style | 5 | 1 | 18 | 0.5833 | 0.8333 |
| proposed_vs_candidate_no_context | distinct1 | 8 | 3 | 13 | 0.6042 | 0.7273 |
| proposed_vs_candidate_no_context | length_score | 5 | 7 | 12 | 0.4583 | 0.4167 |
| proposed_vs_candidate_no_context | sentence_score | 1 | 3 | 20 | 0.4583 | 0.2500 |
| proposed_vs_candidate_no_context | overall_quality | 9 | 4 | 11 | 0.6042 | 0.6923 |
| controlled_vs_proposed_raw | context_relevance | 13 | 11 | 0 | 0.5417 | 0.5417 |
| controlled_vs_proposed_raw | persona_consistency | 13 | 2 | 9 | 0.7292 | 0.8667 |
| controlled_vs_proposed_raw | naturalness | 11 | 13 | 0 | 0.4583 | 0.4583 |
| controlled_vs_proposed_raw | quest_state_correctness | 13 | 11 | 0 | 0.5417 | 0.5417 |
| controlled_vs_proposed_raw | lore_consistency | 6 | 7 | 11 | 0.4792 | 0.4615 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_vs_proposed_raw | objective_completion_support | 7 | 10 | 7 | 0.4375 | 0.4118 |
| controlled_vs_proposed_raw | gameplay_usefulness | 13 | 11 | 0 | 0.5417 | 0.5417 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 7 | 10 | 7 | 0.4375 | 0.4118 |
| controlled_vs_proposed_raw | context_keyword_coverage | 7 | 7 | 10 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | context_overlap | 14 | 10 | 0 | 0.5833 | 0.5833 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 11 | 2 | 11 | 0.6875 | 0.8462 |
| controlled_vs_proposed_raw | persona_style | 4 | 4 | 16 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | distinct1 | 17 | 6 | 1 | 0.7292 | 0.7391 |
| controlled_vs_proposed_raw | length_score | 11 | 13 | 0 | 0.4583 | 0.4583 |
| controlled_vs_proposed_raw | sentence_score | 4 | 1 | 19 | 0.5625 | 0.8000 |
| controlled_vs_proposed_raw | overall_quality | 18 | 6 | 0 | 0.7500 | 0.7500 |
| controlled_vs_candidate_no_context | context_relevance | 15 | 9 | 0 | 0.6250 | 0.6250 |
| controlled_vs_candidate_no_context | persona_consistency | 9 | 3 | 12 | 0.6250 | 0.7500 |
| controlled_vs_candidate_no_context | naturalness | 13 | 11 | 0 | 0.5417 | 0.5417 |
| controlled_vs_candidate_no_context | quest_state_correctness | 15 | 9 | 0 | 0.6250 | 0.6250 |
| controlled_vs_candidate_no_context | lore_consistency | 8 | 2 | 14 | 0.6250 | 0.8000 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_vs_candidate_no_context | objective_completion_support | 9 | 8 | 7 | 0.5208 | 0.5294 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 18 | 6 | 0 | 0.7500 | 0.7500 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 8 | 9 | 7 | 0.4792 | 0.4706 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 11 | 2 | 11 | 0.6875 | 0.8462 |
| controlled_vs_candidate_no_context | context_overlap | 15 | 9 | 0 | 0.6250 | 0.6250 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 7 | 3 | 14 | 0.5833 | 0.7000 |
| controlled_vs_candidate_no_context | persona_style | 5 | 2 | 17 | 0.5625 | 0.7143 |
| controlled_vs_candidate_no_context | distinct1 | 16 | 7 | 1 | 0.6875 | 0.6957 |
| controlled_vs_candidate_no_context | length_score | 13 | 11 | 0 | 0.5417 | 0.5417 |
| controlled_vs_candidate_no_context | sentence_score | 2 | 1 | 21 | 0.5208 | 0.6667 |
| controlled_vs_candidate_no_context | overall_quality | 20 | 4 | 0 | 0.8333 | 0.8333 |
| controlled_alt_vs_controlled_default | context_relevance | 5 | 7 | 12 | 0.4583 | 0.4167 |
| controlled_alt_vs_controlled_default | persona_consistency | 6 | 4 | 14 | 0.5417 | 0.6000 |
| controlled_alt_vs_controlled_default | naturalness | 5 | 6 | 13 | 0.4792 | 0.4545 |
| controlled_alt_vs_controlled_default | quest_state_correctness | 5 | 7 | 12 | 0.4583 | 0.4167 |
| controlled_alt_vs_controlled_default | lore_consistency | 4 | 7 | 13 | 0.4375 | 0.3636 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0 | 8 | 16 | 0.3333 | 0.0000 |
| controlled_alt_vs_controlled_default | objective_completion_support | 6 | 3 | 15 | 0.5625 | 0.6667 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | 5 | 7 | 12 | 0.4583 | 0.4167 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | 6 | 5 | 13 | 0.5208 | 0.5455 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 5 | 7 | 12 | 0.4583 | 0.4167 |
| controlled_alt_vs_controlled_default | context_overlap | 3 | 9 | 12 | 0.3750 | 0.2500 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 5 | 3 | 16 | 0.5417 | 0.6250 |
| controlled_alt_vs_controlled_default | persona_style | 3 | 2 | 19 | 0.5208 | 0.6000 |
| controlled_alt_vs_controlled_default | distinct1 | 5 | 6 | 13 | 0.4792 | 0.4545 |
| controlled_alt_vs_controlled_default | length_score | 3 | 7 | 14 | 0.4167 | 0.3000 |
| controlled_alt_vs_controlled_default | sentence_score | 1 | 1 | 22 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | overall_quality | 4 | 8 | 12 | 0.4167 | 0.3333 |
| controlled_alt_vs_proposed_raw | context_relevance | 13 | 11 | 0 | 0.5417 | 0.5417 |
| controlled_alt_vs_proposed_raw | persona_consistency | 11 | 1 | 12 | 0.7083 | 0.9167 |
| controlled_alt_vs_proposed_raw | naturalness | 10 | 14 | 0 | 0.4167 | 0.4167 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 15 | 9 | 0 | 0.6250 | 0.6250 |
| controlled_alt_vs_proposed_raw | lore_consistency | 4 | 6 | 14 | 0.4583 | 0.4000 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0 | 8 | 16 | 0.3333 | 0.0000 |
| controlled_alt_vs_proposed_raw | objective_completion_support | 8 | 8 | 8 | 0.5000 | 0.5000 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 14 | 10 | 0 | 0.5833 | 0.5833 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | 7 | 10 | 7 | 0.4375 | 0.4118 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 6 | 4 | 14 | 0.5417 | 0.6000 |
| controlled_alt_vs_proposed_raw | context_overlap | 11 | 12 | 1 | 0.4792 | 0.4783 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 10 | 1 | 13 | 0.6875 | 0.9091 |
| controlled_alt_vs_proposed_raw | persona_style | 4 | 3 | 17 | 0.5208 | 0.5714 |
| controlled_alt_vs_proposed_raw | distinct1 | 16 | 6 | 2 | 0.7083 | 0.7273 |
| controlled_alt_vs_proposed_raw | length_score | 10 | 14 | 0 | 0.4167 | 0.4167 |
| controlled_alt_vs_proposed_raw | sentence_score | 4 | 1 | 19 | 0.5625 | 0.8000 |
| controlled_alt_vs_proposed_raw | overall_quality | 17 | 7 | 0 | 0.7083 | 0.7083 |
| controlled_alt_vs_candidate_no_context | context_relevance | 14 | 10 | 0 | 0.5833 | 0.5833 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 9 | 2 | 13 | 0.6458 | 0.8182 |
| controlled_alt_vs_candidate_no_context | naturalness | 11 | 13 | 0 | 0.4583 | 0.4583 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 15 | 9 | 0 | 0.6250 | 0.6250 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 6 | 3 | 15 | 0.5625 | 0.6667 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 8 | 16 | 0.3333 | 0.0000 |
| controlled_alt_vs_candidate_no_context | objective_completion_support | 9 | 8 | 7 | 0.5208 | 0.5294 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 18 | 6 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | 8 | 9 | 7 | 0.4792 | 0.4706 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 8 | 1 | 15 | 0.6458 | 0.8889 |
| controlled_alt_vs_candidate_no_context | context_overlap | 14 | 10 | 0 | 0.5833 | 0.5833 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 7 | 2 | 15 | 0.6042 | 0.7778 |
| controlled_alt_vs_candidate_no_context | persona_style | 5 | 1 | 18 | 0.5833 | 0.8333 |
| controlled_alt_vs_candidate_no_context | distinct1 | 16 | 6 | 2 | 0.7083 | 0.7273 |
| controlled_alt_vs_candidate_no_context | length_score | 10 | 14 | 0 | 0.4167 | 0.4167 |
| controlled_alt_vs_candidate_no_context | sentence_score | 1 | 0 | 23 | 0.5208 | 1.0000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 17 | 7 | 0 | 0.7083 | 0.7083 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.1250 | 0.1667 | 0.8333 |
| proposed_contextual_controlled_tuned | 0.0000 | 0.0000 | 0.2083 | 0.2083 | 0.7917 |
| proposed_contextual | 0.0000 | 0.0000 | 0.5417 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.5833 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `3`
- Unique template signatures: `18`
- Template signature ratio: `0.7500`
- Effective sample size by source clustering: `2.97`
- Effective sample size by template-signature clustering: `16.00`
- Detailed diagnostics are published in `scenario_dependence.json`.

## Multi-turn Contradiction
| Arm | Contradiction Rate | Contradiction Safety | Contradicted Sources | Source Count |
|---|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 1.0000 | 0 | 3 |
| proposed_contextual_controlled_tuned | 0.3333 | 0.6667 | 1 | 3 |
| proposed_contextual | 0.0000 | 1.0000 | 0 | 3 |
| candidate_no_context | 0.0000 | 1.0000 | 0 | 3 |
- Detailed source-level values are published in `multi_turn_contradictions.json`.

- BERTScore status: unavailable (disabled_by_flag).

This report covers proposal RO5 metrics plus game-facing outcomes: quest-state correctness, lore consistency, contradiction safety, objective completion support, gameplay usefulness, and time-pressure acceptability.
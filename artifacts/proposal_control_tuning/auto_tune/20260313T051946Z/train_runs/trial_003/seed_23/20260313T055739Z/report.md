# Proposal Alignment Evaluation Report

- Run ID: `20260313T055739Z`
- Generated: `2026-03-13T06:03:51.448092+00:00`
- Scenarios: `artifacts\proposal_control_tuning\auto_tune\20260313T051946Z\train_runs\trial_003\seed_23\20260313T055739Z\scenarios.jsonl`
- Scenario count: `24`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_tuned`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0735 (0.0335, 0.1183) | 0.2355 (0.1916, 0.2880) | 0.8702 (0.8592, 0.8823) | 0.2885 (0.2606, 0.3184) | n/a |
| proposed_contextual_controlled_tuned | 0.1361 (0.0809, 0.2006) | 0.2564 (0.2098, 0.3075) | 0.8818 (0.8589, 0.9037) | 0.3273 (0.2917, 0.3643) | n/a |
| proposed_contextual | 0.1057 (0.0649, 0.1493) | 0.2349 (0.1699, 0.3059) | 0.8780 (0.8588, 0.8984) | 0.3040 (0.2706, 0.3394) | n/a |
| candidate_no_context | 0.0353 (0.0228, 0.0497) | 0.2679 (0.2013, 0.3412) | 0.8842 (0.8667, 0.9011) | 0.2850 (0.2547, 0.3159) | n/a |

## Game-facing Outcome Metrics (mean, 95% CI)
| Arm | Quest-state Correctness | Lore Consistency | Contradiction Safety | Objective Completion Support | Gameplay Usefulness | Time-pressure Acceptability |
|---|---:|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.1623 (0.1290, 0.1990) | 0.0289 (0.0055, 0.0604) | 1.0000 (1.0000, 1.0000) | 0.0899 (0.0666, 0.1126) | 0.3006 (0.2882, 0.3141) | 0.2973 (0.2826, 0.3115) |
| proposed_contextual_controlled_tuned | 0.2139 (0.1661, 0.2679) | 0.0760 (0.0329, 0.1307) | 1.0000 (1.0000, 1.0000) | 0.0686 (0.0486, 0.0896) | 0.3126 (0.2959, 0.3295) | 0.2857 (0.2722, 0.2990) |
| proposed_contextual | 0.1893 (0.1561, 0.2263) | 0.0404 (0.0173, 0.0675) | 1.0000 (1.0000, 1.0000) | 0.0385 (0.0188, 0.0605) | 0.2914 (0.2760, 0.3059) | 0.2717 (0.2580, 0.2856) |
| candidate_no_context | 0.1251 (0.1162, 0.1353) | 0.0058 (0.0011, 0.0119) | 1.0000 (1.0000, 1.0000) | 0.0532 (0.0299, 0.0779) | 0.2814 (0.2686, 0.2935) | 0.2808 (0.2703, 0.2922) |

- Multi-turn contradiction rate is reported as `1 - contradiction_safety` in row-level outputs.

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0703 | 1.9909 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0330 | -0.1231 |
| proposed_vs_candidate_no_context | naturalness | -0.0063 | -0.0071 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.0642 | 0.5133 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0347 | 5.9848 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | -0.0147 | -0.2768 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 0.0101 | 0.0358 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | -0.0091 | -0.0324 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0871 | 2.5556 |
| proposed_vs_candidate_no_context | context_overlap | 0.0311 | 0.8152 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0347 | -0.1691 |
| proposed_vs_candidate_no_context | persona_style | -0.0260 | -0.0503 |
| proposed_vs_candidate_no_context | distinct1 | -0.0049 | -0.0052 |
| proposed_vs_candidate_no_context | length_score | -0.0069 | -0.0126 |
| proposed_vs_candidate_no_context | sentence_score | -0.0292 | -0.0305 |
| proposed_vs_candidate_no_context | overall_quality | 0.0191 | 0.0669 |
| controlled_vs_proposed_raw | context_relevance | -0.0321 | -0.3041 |
| controlled_vs_proposed_raw | persona_consistency | 0.0006 | 0.0025 |
| controlled_vs_proposed_raw | naturalness | -0.0077 | -0.0088 |
| controlled_vs_proposed_raw | quest_state_correctness | -0.0271 | -0.1430 |
| controlled_vs_proposed_raw | lore_consistency | -0.0116 | -0.2866 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0514 | 1.3357 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0092 | 0.0314 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 0.0256 | 0.0943 |
| controlled_vs_proposed_raw | context_keyword_coverage | -0.0375 | -0.3094 |
| controlled_vs_proposed_raw | context_overlap | -0.0196 | -0.2827 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.0060 | 0.0349 |
| controlled_vs_proposed_raw | persona_style | -0.0208 | -0.0423 |
| controlled_vs_proposed_raw | distinct1 | -0.0209 | -0.0222 |
| controlled_vs_proposed_raw | length_score | -0.0333 | -0.0614 |
| controlled_vs_proposed_raw | sentence_score | 0.0729 | 0.0787 |
| controlled_vs_proposed_raw | overall_quality | -0.0156 | -0.0512 |
| controlled_vs_candidate_no_context | context_relevance | 0.0382 | 1.0813 |
| controlled_vs_candidate_no_context | persona_consistency | -0.0324 | -0.1209 |
| controlled_vs_candidate_no_context | naturalness | -0.0140 | -0.0158 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.0372 | 0.2970 |
| controlled_vs_candidate_no_context | lore_consistency | 0.0231 | 3.9828 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | 0.0367 | 0.6893 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0192 | 0.0683 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 0.0165 | 0.0588 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.0496 | 1.4556 |
| controlled_vs_candidate_no_context | context_overlap | 0.0115 | 0.3020 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | -0.0288 | -0.1401 |
| controlled_vs_candidate_no_context | persona_style | -0.0469 | -0.0905 |
| controlled_vs_candidate_no_context | distinct1 | -0.0258 | -0.0273 |
| controlled_vs_candidate_no_context | length_score | -0.0403 | -0.0732 |
| controlled_vs_candidate_no_context | sentence_score | 0.0438 | 0.0458 |
| controlled_vs_candidate_no_context | overall_quality | 0.0035 | 0.0122 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0626 | 0.8508 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0209 | 0.0887 |
| controlled_alt_vs_controlled_default | naturalness | 0.0116 | 0.0133 |
| controlled_alt_vs_controlled_default | quest_state_correctness | 0.0516 | 0.3180 |
| controlled_alt_vs_controlled_default | lore_consistency | 0.0471 | 1.6336 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_controlled_default | objective_completion_support | -0.0213 | -0.2365 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | 0.0120 | 0.0401 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | -0.0116 | -0.0391 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0807 | 0.9638 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0202 | 0.4071 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0212 | 0.1202 |
| controlled_alt_vs_controlled_default | persona_style | 0.0195 | 0.0414 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0096 | 0.0105 |
| controlled_alt_vs_controlled_default | length_score | 0.0458 | 0.0899 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0146 | -0.0146 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0388 | 0.1346 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.0304 | 0.2879 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.0215 | 0.0914 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0038 | 0.0043 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 0.0245 | 0.1296 |
| controlled_alt_vs_proposed_raw | lore_consistency | 0.0355 | 0.8787 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_proposed_raw | objective_completion_support | 0.0301 | 0.7834 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 0.0212 | 0.0728 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | 0.0140 | 0.0516 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.0432 | 0.3563 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0006 | 0.0093 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.0272 | 0.1593 |
| controlled_alt_vs_proposed_raw | persona_style | -0.0013 | -0.0026 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0113 | -0.0120 |
| controlled_alt_vs_proposed_raw | length_score | 0.0125 | 0.0230 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0583 | 0.0629 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.0232 | 0.0765 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.1007 | 2.8521 |
| controlled_alt_vs_candidate_no_context | persona_consistency | -0.0115 | -0.0429 |
| controlled_alt_vs_candidate_no_context | naturalness | -0.0024 | -0.0028 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 0.0888 | 0.7094 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 0.0702 | 12.1226 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_candidate_no_context | objective_completion_support | 0.0154 | 0.2898 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 0.0313 | 0.1112 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | 0.0049 | 0.0175 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.1303 | 3.8222 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0318 | 0.8320 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | -0.0075 | -0.0367 |
| controlled_alt_vs_candidate_no_context | persona_style | -0.0273 | -0.0528 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0162 | -0.0171 |
| controlled_alt_vs_candidate_no_context | length_score | 0.0056 | 0.0101 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.0292 | 0.0305 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.0423 | 0.1484 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0703 | (0.0294, 0.1128) | 0.0000 | 0.0703 | (0.0346, 0.0976) | 0.0003 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0330 | (-0.0750, -0.0021) | 1.0000 | -0.0330 | (-0.0587, -0.0106) | 1.0000 |
| proposed_vs_candidate_no_context | naturalness | -0.0063 | (-0.0328, 0.0198) | 0.6900 | -0.0063 | (-0.0348, 0.0151) | 0.6773 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.0642 | (0.0310, 0.0993) | 0.0000 | 0.0642 | (0.0313, 0.0870) | 0.0003 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0347 | (0.0103, 0.0625) | 0.0010 | 0.0347 | (0.0102, 0.0642) | 0.0000 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | -0.0147 | (-0.0320, -0.0004) | 0.9777 | -0.0147 | (-0.0285, -0.0019) | 1.0000 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 0.0101 | (-0.0080, 0.0273) | 0.1363 | 0.0101 | (-0.0038, 0.0240) | 0.1050 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | -0.0091 | (-0.0186, -0.0007) | 0.9833 | -0.0091 | (-0.0144, -0.0028) | 1.0000 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0871 | (0.0379, 0.1439) | 0.0000 | 0.0871 | (0.0455, 0.1212) | 0.0007 |
| proposed_vs_candidate_no_context | context_overlap | 0.0311 | (0.0106, 0.0540) | 0.0007 | 0.0311 | (0.0099, 0.0475) | 0.0000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0347 | (-0.0764, 0.0000) | 1.0000 | -0.0347 | (-0.0606, -0.0076) | 1.0000 |
| proposed_vs_candidate_no_context | persona_style | -0.0260 | (-0.0677, 0.0000) | 1.0000 | -0.0260 | (-0.0625, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | distinct1 | -0.0049 | (-0.0168, 0.0063) | 0.7967 | -0.0049 | (-0.0211, 0.0044) | 0.7463 |
| proposed_vs_candidate_no_context | length_score | -0.0069 | (-0.1306, 0.1153) | 0.5530 | -0.0069 | (-0.1200, 0.0893) | 0.5670 |
| proposed_vs_candidate_no_context | sentence_score | -0.0292 | (-0.0875, 0.0292) | 0.9143 | -0.0292 | (-0.0636, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | overall_quality | 0.0191 | (0.0009, 0.0384) | 0.0190 | 0.0191 | (0.0030, 0.0326) | 0.0117 |
| controlled_vs_proposed_raw | context_relevance | -0.0321 | (-0.0798, 0.0131) | 0.9127 | -0.0321 | (-0.0729, -0.0062) | 1.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.0006 | (-0.0638, 0.0558) | 0.4710 | 0.0006 | (-0.1052, 0.0762) | 0.4530 |
| controlled_vs_proposed_raw | naturalness | -0.0077 | (-0.0295, 0.0149) | 0.7463 | -0.0077 | (-0.0323, 0.0124) | 0.7043 |
| controlled_vs_proposed_raw | quest_state_correctness | -0.0271 | (-0.0657, 0.0111) | 0.9147 | -0.0271 | (-0.0567, -0.0040) | 1.0000 |
| controlled_vs_proposed_raw | lore_consistency | -0.0116 | (-0.0442, 0.0176) | 0.7857 | -0.0116 | (-0.0437, 0.0219) | 0.7350 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0514 | (0.0265, 0.0780) | 0.0000 | 0.0514 | (0.0122, 0.0794) | 0.0103 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0092 | (-0.0058, 0.0278) | 0.1410 | 0.0092 | (-0.0092, 0.0235) | 0.1463 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 0.0256 | (0.0093, 0.0423) | 0.0017 | 0.0256 | (0.0060, 0.0410) | 0.0090 |
| controlled_vs_proposed_raw | context_keyword_coverage | -0.0375 | (-0.1023, 0.0242) | 0.8820 | -0.0375 | (-0.0905, -0.0031) | 1.0000 |
| controlled_vs_proposed_raw | context_overlap | -0.0196 | (-0.0455, 0.0074) | 0.9283 | -0.0196 | (-0.0318, -0.0087) | 1.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.0060 | (-0.0645, 0.0694) | 0.4343 | 0.0060 | (-0.1190, 0.0952) | 0.4537 |
| controlled_vs_proposed_raw | persona_style | -0.0208 | (-0.0599, 0.0169) | 0.8770 | -0.0208 | (-0.0682, 0.0000) | 1.0000 |
| controlled_vs_proposed_raw | distinct1 | -0.0209 | (-0.0415, -0.0022) | 0.9850 | -0.0209 | (-0.0513, 0.0139) | 0.8617 |
| controlled_vs_proposed_raw | length_score | -0.0333 | (-0.1181, 0.0459) | 0.7957 | -0.0333 | (-0.1115, 0.0321) | 0.7820 |
| controlled_vs_proposed_raw | sentence_score | 0.0729 | (0.0292, 0.1313) | 0.0040 | 0.0729 | (0.0175, 0.1125) | 0.0103 |
| controlled_vs_proposed_raw | overall_quality | -0.0156 | (-0.0543, 0.0157) | 0.8160 | -0.0156 | (-0.0674, 0.0214) | 0.7470 |
| controlled_vs_candidate_no_context | context_relevance | 0.0382 | (0.0011, 0.0784) | 0.0230 | 0.0382 | (-0.0086, 0.0710) | 0.0617 |
| controlled_vs_candidate_no_context | persona_consistency | -0.0324 | (-0.0952, 0.0236) | 0.8530 | -0.0324 | (-0.1444, 0.0476) | 0.7497 |
| controlled_vs_candidate_no_context | naturalness | -0.0140 | (-0.0332, 0.0053) | 0.9257 | -0.0140 | (-0.0378, 0.0032) | 0.9363 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.0372 | (0.0037, 0.0754) | 0.0140 | 0.0372 | (-0.0057, 0.0682) | 0.0363 |
| controlled_vs_candidate_no_context | lore_consistency | 0.0231 | (0.0014, 0.0506) | 0.0163 | 0.0231 | (0.0000, 0.0466) | 0.0753 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | 0.0367 | (0.0113, 0.0622) | 0.0017 | 0.0367 | (0.0073, 0.0679) | 0.0113 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0192 | (0.0032, 0.0370) | 0.0113 | 0.0192 | (-0.0077, 0.0384) | 0.0753 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 0.0165 | (0.0021, 0.0308) | 0.0093 | 0.0165 | (0.0020, 0.0308) | 0.0113 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.0496 | (0.0076, 0.0992) | 0.0107 | 0.0496 | (-0.0045, 0.0881) | 0.0470 |
| controlled_vs_candidate_no_context | context_overlap | 0.0115 | (-0.0095, 0.0385) | 0.1767 | 0.0115 | (-0.0163, 0.0325) | 0.1577 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | -0.0288 | (-0.0953, 0.0308) | 0.8110 | -0.0288 | (-0.1524, 0.0577) | 0.7387 |
| controlled_vs_candidate_no_context | persona_style | -0.0469 | (-0.0990, -0.0117) | 1.0000 | -0.0469 | (-0.1193, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | distinct1 | -0.0258 | (-0.0456, -0.0059) | 0.9937 | -0.0258 | (-0.0581, 0.0096) | 0.9197 |
| controlled_vs_candidate_no_context | length_score | -0.0403 | (-0.1222, 0.0375) | 0.8483 | -0.0403 | (-0.1450, 0.0361) | 0.8423 |
| controlled_vs_candidate_no_context | sentence_score | 0.0437 | (0.0000, 0.0875) | 0.0393 | 0.0437 | (0.0000, 0.0817) | 0.0727 |
| controlled_vs_candidate_no_context | overall_quality | 0.0035 | (-0.0294, 0.0336) | 0.3967 | 0.0035 | (-0.0556, 0.0432) | 0.4330 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0626 | (-0.0026, 0.1304) | 0.0313 | 0.0626 | (0.0377, 0.0980) | 0.0000 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0209 | (-0.0189, 0.0648) | 0.1530 | 0.0209 | (-0.0074, 0.0426) | 0.0687 |
| controlled_alt_vs_controlled_default | naturalness | 0.0116 | (-0.0097, 0.0325) | 0.1390 | 0.0116 | (-0.0196, 0.0491) | 0.2800 |
| controlled_alt_vs_controlled_default | quest_state_correctness | 0.0516 | (-0.0050, 0.1111) | 0.0353 | 0.0516 | (0.0274, 0.0836) | 0.0000 |
| controlled_alt_vs_controlled_default | lore_consistency | 0.0471 | (-0.0039, 0.1067) | 0.0343 | 0.0471 | (0.0167, 0.0833) | 0.0000 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | objective_completion_support | -0.0213 | (-0.0433, 0.0015) | 0.9680 | -0.0213 | (-0.0465, 0.0031) | 0.9430 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | 0.0120 | (-0.0005, 0.0252) | 0.0313 | 0.0120 | (-0.0021, 0.0262) | 0.0580 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | -0.0116 | (-0.0246, 0.0012) | 0.9593 | -0.0116 | (-0.0166, -0.0063) | 1.0000 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0807 | (-0.0038, 0.1780) | 0.0307 | 0.0807 | (0.0515, 0.1281) | 0.0000 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0202 | (-0.0071, 0.0471) | 0.0767 | 0.0202 | (-0.0010, 0.0394) | 0.0293 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0212 | (-0.0258, 0.0683) | 0.1940 | 0.0212 | (-0.0082, 0.0504) | 0.0877 |
| controlled_alt_vs_controlled_default | persona_style | 0.0195 | (-0.0208, 0.0599) | 0.2253 | 0.0195 | (0.0000, 0.0469) | 0.0727 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0096 | (-0.0039, 0.0259) | 0.0967 | 0.0096 | (-0.0060, 0.0292) | 0.2263 |
| controlled_alt_vs_controlled_default | length_score | 0.0458 | (-0.0500, 0.1403) | 0.1843 | 0.0458 | (-0.0917, 0.1861) | 0.3037 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0146 | (-0.0437, 0.0000) | 1.0000 | -0.0146 | (-0.0404, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0388 | (-0.0005, 0.0780) | 0.0267 | 0.0388 | (0.0316, 0.0498) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.0304 | (-0.0253, 0.0901) | 0.1467 | 0.0304 | (0.0133, 0.0481) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.0215 | (-0.0439, 0.0879) | 0.2533 | 0.0215 | (-0.0774, 0.0914) | 0.3007 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0038 | (-0.0209, 0.0308) | 0.3760 | 0.0038 | (-0.0114, 0.0161) | 0.3133 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 0.0245 | (-0.0277, 0.0774) | 0.1770 | 0.0245 | (0.0097, 0.0363) | 0.0003 |
| controlled_alt_vs_proposed_raw | lore_consistency | 0.0355 | (-0.0015, 0.0770) | 0.0307 | 0.0355 | (0.0065, 0.0645) | 0.0000 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_proposed_raw | objective_completion_support | 0.0301 | (0.0119, 0.0498) | 0.0000 | 0.0301 | (0.0083, 0.0446) | 0.0037 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 0.0212 | (0.0027, 0.0425) | 0.0120 | 0.0212 | (0.0010, 0.0370) | 0.0150 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | 0.0140 | (-0.0018, 0.0307) | 0.0443 | 0.0140 | (-0.0066, 0.0293) | 0.0747 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.0432 | (-0.0288, 0.1189) | 0.1200 | 0.0432 | (0.0182, 0.0630) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0006 | (-0.0266, 0.0289) | 0.4873 | 0.0006 | (-0.0192, 0.0197) | 0.4823 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.0272 | (-0.0456, 0.0996) | 0.2380 | 0.0272 | (-0.0948, 0.1143) | 0.2947 |
| controlled_alt_vs_proposed_raw | persona_style | -0.0013 | (-0.0508, 0.0586) | 0.5340 | -0.0013 | (-0.0312, 0.0273) | 0.6240 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0113 | (-0.0322, 0.0069) | 0.8880 | -0.0113 | (-0.0330, 0.0162) | 0.7870 |
| controlled_alt_vs_proposed_raw | length_score | 0.0125 | (-0.0944, 0.1236) | 0.4093 | 0.0125 | (-0.0900, 0.0893) | 0.3893 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0583 | (0.0146, 0.1167) | 0.0110 | 0.0583 | (0.0159, 0.0942) | 0.0090 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.0232 | (-0.0202, 0.0686) | 0.1560 | 0.0232 | (-0.0211, 0.0545) | 0.1387 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.1007 | (0.0452, 0.1668) | 0.0000 | 0.1007 | (0.0697, 0.1206) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | -0.0115 | (-0.0788, 0.0515) | 0.6350 | -0.0115 | (-0.1103, 0.0636) | 0.6190 |
| controlled_alt_vs_candidate_no_context | naturalness | -0.0024 | (-0.0288, 0.0253) | 0.5697 | -0.0024 | (-0.0361, 0.0302) | 0.5777 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 0.0888 | (0.0372, 0.1427) | 0.0000 | 0.0888 | (0.0581, 0.1083) | 0.0000 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 0.0702 | (0.0285, 0.1215) | 0.0000 | 0.0702 | (0.0336, 0.1010) | 0.0000 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_candidate_no_context | objective_completion_support | 0.0154 | (-0.0094, 0.0400) | 0.1143 | 0.0154 | (0.0022, 0.0260) | 0.0097 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 0.0313 | (0.0106, 0.0534) | 0.0003 | 0.0313 | (0.0036, 0.0570) | 0.0130 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | 0.0049 | (-0.0096, 0.0190) | 0.2447 | 0.0049 | (-0.0093, 0.0170) | 0.2743 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.1303 | (0.0568, 0.2137) | 0.0003 | 0.1303 | (0.0909, 0.1527) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0318 | (0.0083, 0.0588) | 0.0027 | 0.0318 | (0.0051, 0.0540) | 0.0073 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | -0.0075 | (-0.0833, 0.0704) | 0.5640 | -0.0075 | (-0.1281, 0.0795) | 0.5540 |
| controlled_alt_vs_candidate_no_context | persona_style | -0.0273 | (-0.0716, 0.0143) | 0.9050 | -0.0273 | (-0.0753, 0.0000) | 1.0000 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0162 | (-0.0355, 0.0014) | 0.9610 | -0.0162 | (-0.0342, 0.0097) | 0.8993 |
| controlled_alt_vs_candidate_no_context | length_score | 0.0056 | (-0.1222, 0.1389) | 0.4630 | 0.0056 | (-0.1833, 0.1658) | 0.4907 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.0292 | (-0.0292, 0.0875) | 0.2307 | 0.0292 | (0.0000, 0.0808) | 0.3327 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.0423 | (-0.0003, 0.0846) | 0.0263 | 0.0423 | (-0.0087, 0.0787) | 0.0387 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 12 | 3 | 9 | 0.6875 | 0.8000 |
| proposed_vs_candidate_no_context | persona_consistency | 0 | 4 | 20 | 0.4167 | 0.0000 |
| proposed_vs_candidate_no_context | naturalness | 7 | 8 | 9 | 0.4792 | 0.4667 |
| proposed_vs_candidate_no_context | quest_state_correctness | 13 | 2 | 9 | 0.7292 | 0.8667 |
| proposed_vs_candidate_no_context | lore_consistency | 9 | 2 | 13 | 0.6458 | 0.8182 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| proposed_vs_candidate_no_context | objective_completion_support | 1 | 5 | 18 | 0.4167 | 0.1667 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 11 | 4 | 9 | 0.6458 | 0.7333 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | 2 | 7 | 15 | 0.3958 | 0.2222 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 10 | 1 | 13 | 0.6875 | 0.9091 |
| proposed_vs_candidate_no_context | context_overlap | 11 | 4 | 9 | 0.6458 | 0.7333 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0 | 3 | 21 | 0.4375 | 0.0000 |
| proposed_vs_candidate_no_context | persona_style | 0 | 2 | 22 | 0.4583 | 0.0000 |
| proposed_vs_candidate_no_context | distinct1 | 4 | 6 | 14 | 0.4583 | 0.4000 |
| proposed_vs_candidate_no_context | length_score | 7 | 8 | 9 | 0.4792 | 0.4667 |
| proposed_vs_candidate_no_context | sentence_score | 1 | 3 | 20 | 0.4583 | 0.2500 |
| proposed_vs_candidate_no_context | overall_quality | 10 | 5 | 9 | 0.6042 | 0.6667 |
| controlled_vs_proposed_raw | context_relevance | 5 | 15 | 4 | 0.2917 | 0.2500 |
| controlled_vs_proposed_raw | persona_consistency | 9 | 5 | 10 | 0.5833 | 0.6429 |
| controlled_vs_proposed_raw | naturalness | 9 | 11 | 4 | 0.4583 | 0.4500 |
| controlled_vs_proposed_raw | quest_state_correctness | 5 | 15 | 4 | 0.2917 | 0.2500 |
| controlled_vs_proposed_raw | lore_consistency | 5 | 6 | 13 | 0.4792 | 0.4545 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_vs_proposed_raw | objective_completion_support | 11 | 1 | 12 | 0.7083 | 0.9167 |
| controlled_vs_proposed_raw | gameplay_usefulness | 9 | 11 | 4 | 0.4583 | 0.4500 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 13 | 1 | 10 | 0.7500 | 0.9286 |
| controlled_vs_proposed_raw | context_keyword_coverage | 5 | 10 | 9 | 0.3958 | 0.3333 |
| controlled_vs_proposed_raw | context_overlap | 4 | 15 | 5 | 0.2708 | 0.2105 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 9 | 5 | 10 | 0.5833 | 0.6429 |
| controlled_vs_proposed_raw | persona_style | 1 | 5 | 18 | 0.4167 | 0.1667 |
| controlled_vs_proposed_raw | distinct1 | 9 | 11 | 4 | 0.4583 | 0.4500 |
| controlled_vs_proposed_raw | length_score | 7 | 13 | 4 | 0.3750 | 0.3500 |
| controlled_vs_proposed_raw | sentence_score | 5 | 0 | 19 | 0.6042 | 1.0000 |
| controlled_vs_proposed_raw | overall_quality | 10 | 10 | 4 | 0.5000 | 0.5000 |
| controlled_vs_candidate_no_context | context_relevance | 10 | 9 | 5 | 0.5208 | 0.5263 |
| controlled_vs_candidate_no_context | persona_consistency | 6 | 7 | 11 | 0.4792 | 0.4615 |
| controlled_vs_candidate_no_context | naturalness | 6 | 14 | 4 | 0.3333 | 0.3000 |
| controlled_vs_candidate_no_context | quest_state_correctness | 12 | 8 | 4 | 0.5833 | 0.6000 |
| controlled_vs_candidate_no_context | lore_consistency | 6 | 2 | 16 | 0.5833 | 0.7500 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_vs_candidate_no_context | objective_completion_support | 10 | 3 | 11 | 0.6458 | 0.7692 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 13 | 7 | 4 | 0.6250 | 0.6500 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 10 | 4 | 10 | 0.6250 | 0.7143 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 7 | 3 | 14 | 0.5833 | 0.7000 |
| controlled_vs_candidate_no_context | context_overlap | 8 | 11 | 5 | 0.4375 | 0.4211 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 6 | 7 | 11 | 0.4792 | 0.4615 |
| controlled_vs_candidate_no_context | persona_style | 0 | 6 | 18 | 0.3750 | 0.0000 |
| controlled_vs_candidate_no_context | distinct1 | 8 | 12 | 4 | 0.4167 | 0.4000 |
| controlled_vs_candidate_no_context | length_score | 7 | 13 | 4 | 0.3750 | 0.3500 |
| controlled_vs_candidate_no_context | sentence_score | 3 | 0 | 21 | 0.5625 | 1.0000 |
| controlled_vs_candidate_no_context | overall_quality | 12 | 8 | 4 | 0.5833 | 0.6000 |
| controlled_alt_vs_controlled_default | context_relevance | 10 | 4 | 10 | 0.6250 | 0.7143 |
| controlled_alt_vs_controlled_default | persona_consistency | 8 | 4 | 12 | 0.5833 | 0.6667 |
| controlled_alt_vs_controlled_default | naturalness | 9 | 5 | 10 | 0.5833 | 0.6429 |
| controlled_alt_vs_controlled_default | quest_state_correctness | 10 | 4 | 10 | 0.6250 | 0.7143 |
| controlled_alt_vs_controlled_default | lore_consistency | 10 | 2 | 12 | 0.6667 | 0.8333 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_alt_vs_controlled_default | objective_completion_support | 3 | 9 | 12 | 0.3750 | 0.2500 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | 10 | 4 | 10 | 0.6250 | 0.7143 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | 4 | 9 | 11 | 0.3958 | 0.3077 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 9 | 3 | 12 | 0.6250 | 0.7500 |
| controlled_alt_vs_controlled_default | context_overlap | 11 | 3 | 10 | 0.6667 | 0.7857 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 6 | 4 | 14 | 0.5417 | 0.6000 |
| controlled_alt_vs_controlled_default | persona_style | 3 | 1 | 20 | 0.5417 | 0.7500 |
| controlled_alt_vs_controlled_default | distinct1 | 7 | 7 | 10 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | length_score | 9 | 5 | 10 | 0.5833 | 0.6429 |
| controlled_alt_vs_controlled_default | sentence_score | 0 | 1 | 23 | 0.4792 | 0.0000 |
| controlled_alt_vs_controlled_default | overall_quality | 11 | 3 | 10 | 0.6667 | 0.7857 |
| controlled_alt_vs_proposed_raw | context_relevance | 10 | 11 | 3 | 0.4792 | 0.4762 |
| controlled_alt_vs_proposed_raw | persona_consistency | 10 | 7 | 7 | 0.5625 | 0.5882 |
| controlled_alt_vs_proposed_raw | naturalness | 11 | 10 | 3 | 0.5208 | 0.5238 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 10 | 11 | 3 | 0.4792 | 0.4762 |
| controlled_alt_vs_proposed_raw | lore_consistency | 10 | 3 | 11 | 0.6458 | 0.7692 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_alt_vs_proposed_raw | objective_completion_support | 10 | 1 | 13 | 0.6875 | 0.9091 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 13 | 8 | 3 | 0.6042 | 0.6190 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | 10 | 6 | 8 | 0.5833 | 0.6250 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 9 | 8 | 7 | 0.5208 | 0.5294 |
| controlled_alt_vs_proposed_raw | context_overlap | 9 | 12 | 3 | 0.4375 | 0.4286 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 9 | 7 | 8 | 0.5417 | 0.5625 |
| controlled_alt_vs_proposed_raw | persona_style | 2 | 5 | 17 | 0.4375 | 0.2857 |
| controlled_alt_vs_proposed_raw | distinct1 | 12 | 8 | 4 | 0.5833 | 0.6000 |
| controlled_alt_vs_proposed_raw | length_score | 12 | 9 | 3 | 0.5625 | 0.5714 |
| controlled_alt_vs_proposed_raw | sentence_score | 4 | 0 | 20 | 0.5833 | 1.0000 |
| controlled_alt_vs_proposed_raw | overall_quality | 13 | 8 | 3 | 0.6042 | 0.6190 |
| controlled_alt_vs_candidate_no_context | context_relevance | 14 | 7 | 3 | 0.6458 | 0.6667 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 9 | 9 | 6 | 0.5000 | 0.5000 |
| controlled_alt_vs_candidate_no_context | naturalness | 8 | 13 | 3 | 0.3958 | 0.3810 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 16 | 5 | 3 | 0.7292 | 0.7619 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 11 | 0 | 13 | 0.7292 | 1.0000 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_alt_vs_candidate_no_context | objective_completion_support | 10 | 4 | 10 | 0.6250 | 0.7143 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 16 | 5 | 3 | 0.7292 | 0.7619 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | 10 | 6 | 8 | 0.5833 | 0.6250 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 11 | 2 | 11 | 0.6875 | 0.8462 |
| controlled_alt_vs_candidate_no_context | context_overlap | 14 | 7 | 3 | 0.6458 | 0.6667 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 8 | 9 | 7 | 0.4792 | 0.4706 |
| controlled_alt_vs_candidate_no_context | persona_style | 2 | 6 | 16 | 0.4167 | 0.2500 |
| controlled_alt_vs_candidate_no_context | distinct1 | 11 | 9 | 4 | 0.5417 | 0.5500 |
| controlled_alt_vs_candidate_no_context | length_score | 9 | 12 | 3 | 0.4375 | 0.4286 |
| controlled_alt_vs_candidate_no_context | sentence_score | 3 | 1 | 20 | 0.5417 | 0.7500 |
| controlled_alt_vs_candidate_no_context | overall_quality | 14 | 7 | 3 | 0.6458 | 0.6667 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.2083 | 0.1667 | 0.8333 |
| proposed_contextual_controlled_tuned | 0.0000 | 0.0000 | 0.0417 | 0.2500 | 0.7500 |
| proposed_contextual | 0.0000 | 0.0000 | 0.3750 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.4583 | 0.0000 | 0.0000 |

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
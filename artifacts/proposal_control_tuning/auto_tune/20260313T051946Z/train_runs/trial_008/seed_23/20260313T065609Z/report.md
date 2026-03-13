# Proposal Alignment Evaluation Report

- Run ID: `20260313T065609Z`
- Generated: `2026-03-13T07:02:25.658335+00:00`
- Scenarios: `artifacts\proposal_control_tuning\auto_tune\20260313T051946Z\train_runs\trial_008\seed_23\20260313T065609Z\scenarios.jsonl`
- Scenario count: `24`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_tuned`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.1205 (0.0716, 0.1695) | 0.3048 (0.2400, 0.3811) | 0.8889 (0.8673, 0.9106) | 0.3396 (0.2995, 0.3814) | n/a |
| proposed_contextual_controlled_tuned | 0.1087 (0.0538, 0.1705) | 0.2771 (0.2226, 0.3402) | 0.8703 (0.8569, 0.8835) | 0.3203 (0.2772, 0.3647) | n/a |
| proposed_contextual | 0.1097 (0.0614, 0.1646) | 0.2415 (0.1808, 0.3073) | 0.8752 (0.8620, 0.8889) | 0.3077 (0.2714, 0.3458) | n/a |
| candidate_no_context | 0.0297 (0.0179, 0.0427) | 0.2479 (0.1851, 0.3140) | 0.8784 (0.8608, 0.8938) | 0.2740 (0.2459, 0.3044) | n/a |

## Game-facing Outcome Metrics (mean, 95% CI)
| Arm | Quest-state Correctness | Lore Consistency | Contradiction Safety | Objective Completion Support | Gameplay Usefulness | Time-pressure Acceptability |
|---|---:|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.1983 (0.1585, 0.2436) | 0.0518 (0.0188, 0.0906) | 1.0000 (1.0000, 1.0000) | 0.0783 (0.0541, 0.1011) | 0.3147 (0.2996, 0.3302) | 0.2903 (0.2760, 0.3040) |
| proposed_contextual_controlled_tuned | 0.1937 (0.1437, 0.2465) | 0.0383 (0.0129, 0.0693) | 1.0000 (1.0000, 1.0000) | 0.0877 (0.0640, 0.1099) | 0.3100 (0.2931, 0.3276) | 0.2937 (0.2780, 0.3093) |
| proposed_contextual | 0.1933 (0.1516, 0.2431) | 0.0445 (0.0154, 0.0796) | 1.0000 (1.0000, 1.0000) | 0.0562 (0.0335, 0.0799) | 0.2980 (0.2833, 0.3138) | 0.2806 (0.2691, 0.2932) |
| candidate_no_context | 0.1208 (0.1130, 0.1294) | 0.0076 (0.0027, 0.0139) | 1.0000 (1.0000, 1.0000) | 0.0447 (0.0238, 0.0672) | 0.2763 (0.2649, 0.2876) | 0.2857 (0.2727, 0.3005) |

- Multi-turn contradiction rate is reported as `1 - contradiction_safety` in row-level outputs.

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0800 | 2.6915 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0064 | -0.0258 |
| proposed_vs_candidate_no_context | naturalness | -0.0032 | -0.0036 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.0725 | 0.6002 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0369 | 4.8281 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | 0.0115 | 0.2582 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 0.0217 | 0.0785 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | -0.0051 | -0.0177 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.1023 | 3.8571 |
| proposed_vs_candidate_no_context | context_overlap | 0.0279 | 0.7517 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0119 | -0.0642 |
| proposed_vs_candidate_no_context | persona_style | 0.0156 | 0.0314 |
| proposed_vs_candidate_no_context | distinct1 | 0.0038 | 0.0041 |
| proposed_vs_candidate_no_context | length_score | -0.0236 | -0.0429 |
| proposed_vs_candidate_no_context | sentence_score | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | overall_quality | 0.0337 | 0.1230 |
| controlled_vs_proposed_raw | context_relevance | 0.0109 | 0.0990 |
| controlled_vs_proposed_raw | persona_consistency | 0.0633 | 0.2620 |
| controlled_vs_proposed_raw | naturalness | 0.0136 | 0.0156 |
| controlled_vs_proposed_raw | quest_state_correctness | 0.0050 | 0.0259 |
| controlled_vs_proposed_raw | lore_consistency | 0.0073 | 0.1643 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0221 | 0.3931 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0167 | 0.0561 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 0.0097 | 0.0344 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.0155 | 0.1206 |
| controlled_vs_proposed_raw | context_overlap | -0.0000 | -0.0006 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.0833 | 0.4800 |
| controlled_vs_proposed_raw | persona_style | -0.0169 | -0.0330 |
| controlled_vs_proposed_raw | distinct1 | -0.0057 | -0.0060 |
| controlled_vs_proposed_raw | length_score | 0.0431 | 0.0818 |
| controlled_vs_proposed_raw | sentence_score | 0.0729 | 0.0787 |
| controlled_vs_proposed_raw | overall_quality | 0.0319 | 0.1036 |
| controlled_vs_candidate_no_context | context_relevance | 0.0908 | 3.0570 |
| controlled_vs_candidate_no_context | persona_consistency | 0.0569 | 0.2295 |
| controlled_vs_candidate_no_context | naturalness | 0.0104 | 0.0119 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.0775 | 0.6415 |
| controlled_vs_candidate_no_context | lore_consistency | 0.0442 | 5.7858 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | 0.0336 | 0.7527 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0384 | 0.1390 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 0.0046 | 0.0161 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.1178 | 4.4429 |
| controlled_vs_candidate_no_context | context_overlap | 0.0279 | 0.7507 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.0714 | 0.3850 |
| controlled_vs_candidate_no_context | persona_style | -0.0013 | -0.0026 |
| controlled_vs_candidate_no_context | distinct1 | -0.0018 | -0.0019 |
| controlled_vs_candidate_no_context | length_score | 0.0194 | 0.0354 |
| controlled_vs_candidate_no_context | sentence_score | 0.0729 | 0.0787 |
| controlled_vs_candidate_no_context | overall_quality | 0.0656 | 0.2393 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0118 | -0.0982 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0277 | -0.0908 |
| controlled_alt_vs_controlled_default | naturalness | -0.0186 | -0.0209 |
| controlled_alt_vs_controlled_default | quest_state_correctness | -0.0046 | -0.0230 |
| controlled_alt_vs_controlled_default | lore_consistency | -0.0135 | -0.2602 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_controlled_default | objective_completion_support | 0.0094 | 0.1207 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | -0.0047 | -0.0150 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | 0.0034 | 0.0118 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0152 | -0.1050 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0041 | -0.0631 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0294 | -0.1143 |
| controlled_alt_vs_controlled_default | persona_style | -0.0208 | -0.0420 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0158 | -0.0168 |
| controlled_alt_vs_controlled_default | length_score | -0.0542 | -0.0951 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0146 | -0.0146 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0193 | -0.0567 |
| controlled_alt_vs_proposed_raw | context_relevance | -0.0010 | -0.0089 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.0356 | 0.1475 |
| controlled_alt_vs_proposed_raw | naturalness | -0.0050 | -0.0057 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 0.0004 | 0.0022 |
| controlled_alt_vs_proposed_raw | lore_consistency | -0.0062 | -0.1387 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_proposed_raw | objective_completion_support | 0.0315 | 0.5612 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 0.0120 | 0.0402 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | 0.0131 | 0.0466 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.0004 | 0.0029 |
| controlled_alt_vs_proposed_raw | context_overlap | -0.0041 | -0.0637 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.0540 | 0.3109 |
| controlled_alt_vs_proposed_raw | persona_style | -0.0378 | -0.0736 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0214 | -0.0227 |
| controlled_alt_vs_proposed_raw | length_score | -0.0111 | -0.0211 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0583 | 0.0629 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.0126 | 0.0410 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.0790 | 2.6586 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.0292 | 0.1179 |
| controlled_alt_vs_candidate_no_context | naturalness | -0.0082 | -0.0093 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 0.0729 | 0.6037 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 0.0307 | 4.0201 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_candidate_no_context | objective_completion_support | 0.0431 | 0.9642 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 0.0337 | 0.1219 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | 0.0080 | 0.0280 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.1027 | 3.8714 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0238 | 0.6401 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.0421 | 0.2267 |
| controlled_alt_vs_candidate_no_context | persona_style | -0.0221 | -0.0445 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0176 | -0.0187 |
| controlled_alt_vs_candidate_no_context | length_score | -0.0347 | -0.0631 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.0583 | 0.0629 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.0463 | 0.1690 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0800 | (0.0342, 0.1325) | 0.0000 | 0.0800 | (0.0406, 0.1070) | 0.0003 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0064 | (-0.0546, 0.0376) | 0.5987 | -0.0064 | (-0.0452, 0.0369) | 0.6617 |
| proposed_vs_candidate_no_context | naturalness | -0.0032 | (-0.0235, 0.0152) | 0.6240 | -0.0032 | (-0.0288, 0.0154) | 0.6320 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.0725 | (0.0345, 0.1148) | 0.0000 | 0.0725 | (0.0363, 0.0981) | 0.0003 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0369 | (0.0076, 0.0742) | 0.0023 | 0.0369 | (0.0056, 0.0592) | 0.0107 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | 0.0115 | (-0.0099, 0.0320) | 0.1367 | 0.0115 | (-0.0043, 0.0307) | 0.0930 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 0.0217 | (0.0054, 0.0390) | 0.0027 | 0.0217 | (0.0073, 0.0377) | 0.0000 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | -0.0051 | (-0.0196, 0.0072) | 0.7483 | -0.0051 | (-0.0175, 0.0057) | 0.7083 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.1023 | (0.0417, 0.1705) | 0.0000 | 0.1023 | (0.0537, 0.1331) | 0.0007 |
| proposed_vs_candidate_no_context | context_overlap | 0.0279 | (0.0065, 0.0554) | 0.0010 | 0.0279 | (0.0029, 0.0525) | 0.0000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0119 | (-0.0744, 0.0417) | 0.6607 | -0.0119 | (-0.0639, 0.0467) | 0.6910 |
| proposed_vs_candidate_no_context | persona_style | 0.0156 | (0.0000, 0.0417) | 0.1283 | 0.0156 | (0.0000, 0.0398) | 0.0757 |
| proposed_vs_candidate_no_context | distinct1 | 0.0038 | (-0.0094, 0.0182) | 0.3037 | 0.0038 | (-0.0184, 0.0197) | 0.3400 |
| proposed_vs_candidate_no_context | length_score | -0.0236 | (-0.1097, 0.0542) | 0.7250 | -0.0236 | (-0.1050, 0.0389) | 0.7400 |
| proposed_vs_candidate_no_context | sentence_score | 0.0000 | (-0.0729, 0.0729) | 0.5883 | 0.0000 | (-0.0477, 0.0375) | 0.6393 |
| proposed_vs_candidate_no_context | overall_quality | 0.0337 | (0.0058, 0.0610) | 0.0073 | 0.0337 | (0.0039, 0.0624) | 0.0007 |
| controlled_vs_proposed_raw | context_relevance | 0.0109 | (-0.0456, 0.0687) | 0.3417 | 0.0109 | (-0.0600, 0.0632) | 0.3513 |
| controlled_vs_proposed_raw | persona_consistency | 0.0633 | (-0.0198, 0.1548) | 0.0720 | 0.0633 | (-0.0050, 0.1333) | 0.0373 |
| controlled_vs_proposed_raw | naturalness | 0.0136 | (-0.0158, 0.0410) | 0.1720 | 0.0136 | (-0.0048, 0.0287) | 0.0743 |
| controlled_vs_proposed_raw | quest_state_correctness | 0.0050 | (-0.0410, 0.0521) | 0.4213 | 0.0050 | (-0.0480, 0.0509) | 0.4117 |
| controlled_vs_proposed_raw | lore_consistency | 0.0073 | (-0.0316, 0.0470) | 0.3470 | 0.0073 | (-0.0169, 0.0333) | 0.3427 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0221 | (-0.0046, 0.0503) | 0.0553 | 0.0221 | (-0.0067, 0.0417) | 0.0557 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0167 | (-0.0007, 0.0350) | 0.0303 | 0.0167 | (-0.0098, 0.0376) | 0.1123 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 0.0097 | (-0.0049, 0.0253) | 0.0983 | 0.0097 | (-0.0024, 0.0183) | 0.0573 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.0155 | (-0.0564, 0.0875) | 0.3240 | 0.0155 | (-0.0781, 0.0833) | 0.3303 |
| controlled_vs_proposed_raw | context_overlap | -0.0000 | (-0.0273, 0.0252) | 0.4920 | -0.0000 | (-0.0222, 0.0186) | 0.4913 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.0833 | (-0.0079, 0.1984) | 0.0423 | 0.0833 | (0.0076, 0.1667) | 0.0187 |
| controlled_vs_proposed_raw | persona_style | -0.0169 | (-0.0495, 0.0169) | 0.8487 | -0.0169 | (-0.0554, 0.0000) | 1.0000 |
| controlled_vs_proposed_raw | distinct1 | -0.0057 | (-0.0224, 0.0137) | 0.7183 | -0.0057 | (-0.0250, 0.0215) | 0.6467 |
| controlled_vs_proposed_raw | length_score | 0.0431 | (-0.0778, 0.1584) | 0.2377 | 0.0431 | (-0.0550, 0.1256) | 0.2077 |
| controlled_vs_proposed_raw | sentence_score | 0.0729 | (0.0146, 0.1313) | 0.0040 | 0.0729 | (0.0175, 0.1125) | 0.0103 |
| controlled_vs_proposed_raw | overall_quality | 0.0319 | (-0.0128, 0.0827) | 0.0963 | 0.0319 | (-0.0216, 0.0743) | 0.0920 |
| controlled_vs_candidate_no_context | context_relevance | 0.0908 | (0.0455, 0.1389) | 0.0000 | 0.0908 | (0.0213, 0.1567) | 0.0097 |
| controlled_vs_candidate_no_context | persona_consistency | 0.0569 | (-0.0161, 0.1420) | 0.0667 | 0.0569 | (-0.0456, 0.1436) | 0.1453 |
| controlled_vs_candidate_no_context | naturalness | 0.0104 | (-0.0180, 0.0396) | 0.2383 | 0.0104 | (-0.0325, 0.0419) | 0.2927 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.0775 | (0.0398, 0.1206) | 0.0000 | 0.0775 | (0.0178, 0.1379) | 0.0067 |
| controlled_vs_candidate_no_context | lore_consistency | 0.0442 | (0.0146, 0.0761) | 0.0000 | 0.0442 | (0.0000, 0.0875) | 0.0753 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | 0.0336 | (0.0087, 0.0617) | 0.0047 | 0.0336 | (-0.0011, 0.0658) | 0.0410 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0384 | (0.0188, 0.0608) | 0.0000 | 0.0384 | (0.0028, 0.0734) | 0.0193 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 0.0046 | (-0.0162, 0.0222) | 0.3027 | 0.0046 | (-0.0097, 0.0189) | 0.2953 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.1178 | (0.0606, 0.1818) | 0.0000 | 0.1178 | (0.0273, 0.1892) | 0.0073 |
| controlled_vs_candidate_no_context | context_overlap | 0.0279 | (0.0081, 0.0499) | 0.0023 | 0.0279 | (-0.0081, 0.0602) | 0.0457 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.0714 | (-0.0139, 0.1756) | 0.0570 | 0.0714 | (-0.0563, 0.1795) | 0.1320 |
| controlled_vs_candidate_no_context | persona_style | -0.0013 | (-0.0378, 0.0378) | 0.5427 | -0.0013 | (-0.0352, 0.0312) | 0.6437 |
| controlled_vs_candidate_no_context | distinct1 | -0.0018 | (-0.0215, 0.0191) | 0.5727 | -0.0018 | (-0.0221, 0.0221) | 0.5890 |
| controlled_vs_candidate_no_context | length_score | 0.0194 | (-0.0986, 0.1389) | 0.4147 | 0.0194 | (-0.1545, 0.1692) | 0.4037 |
| controlled_vs_candidate_no_context | sentence_score | 0.0729 | (0.0288, 0.1313) | 0.0053 | 0.0729 | (0.0000, 0.1375) | 0.0727 |
| controlled_vs_candidate_no_context | overall_quality | 0.0656 | (0.0185, 0.1144) | 0.0023 | 0.0656 | (0.0007, 0.1265) | 0.0247 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0118 | (-0.0611, 0.0425) | 0.6753 | -0.0118 | (-0.0673, 0.0396) | 0.6470 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0277 | (-0.1276, 0.0567) | 0.7253 | -0.0277 | (-0.1197, 0.0381) | 0.7533 |
| controlled_alt_vs_controlled_default | naturalness | -0.0186 | (-0.0433, 0.0054) | 0.9280 | -0.0186 | (-0.0386, -0.0015) | 0.9883 |
| controlled_alt_vs_controlled_default | quest_state_correctness | -0.0046 | (-0.0463, 0.0392) | 0.5940 | -0.0046 | (-0.0511, 0.0342) | 0.5660 |
| controlled_alt_vs_controlled_default | lore_consistency | -0.0135 | (-0.0480, 0.0215) | 0.7717 | -0.0135 | (-0.0550, 0.0337) | 0.6997 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | objective_completion_support | 0.0094 | (-0.0139, 0.0349) | 0.2350 | 0.0094 | (0.0000, 0.0197) | 0.0713 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | -0.0047 | (-0.0216, 0.0114) | 0.7193 | -0.0047 | (-0.0241, 0.0177) | 0.6483 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | 0.0034 | (-0.0114, 0.0174) | 0.3297 | 0.0034 | (-0.0008, 0.0105) | 0.0843 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0152 | (-0.0795, 0.0492) | 0.7077 | -0.0152 | (-0.0844, 0.0490) | 0.6453 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0041 | (-0.0277, 0.0240) | 0.6360 | -0.0041 | (-0.0317, 0.0212) | 0.6007 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0294 | (-0.1488, 0.0750) | 0.6870 | -0.0294 | (-0.1371, 0.0500) | 0.7110 |
| controlled_alt_vs_controlled_default | persona_style | -0.0208 | (-0.0521, 0.0000) | 1.0000 | -0.0208 | (-0.0682, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0158 | (-0.0377, 0.0057) | 0.9197 | -0.0158 | (-0.0363, -0.0001) | 1.0000 |
| controlled_alt_vs_controlled_default | length_score | -0.0542 | (-0.1431, 0.0292) | 0.8960 | -0.0542 | (-0.1051, -0.0030) | 0.9847 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0146 | (-0.0437, 0.0000) | 1.0000 | -0.0146 | (-0.0404, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0193 | (-0.0704, 0.0252) | 0.7753 | -0.0193 | (-0.0461, 0.0055) | 0.9370 |
| controlled_alt_vs_proposed_raw | context_relevance | -0.0010 | (-0.0692, 0.0708) | 0.5393 | -0.0010 | (-0.0619, 0.0660) | 0.5047 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.0356 | (-0.0506, 0.1277) | 0.2140 | 0.0356 | (-0.0708, 0.1114) | 0.2217 |
| controlled_alt_vs_proposed_raw | naturalness | -0.0050 | (-0.0230, 0.0137) | 0.6893 | -0.0050 | (-0.0182, 0.0094) | 0.7027 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 0.0004 | (-0.0554, 0.0594) | 0.5127 | 0.0004 | (-0.0523, 0.0515) | 0.4543 |
| controlled_alt_vs_proposed_raw | lore_consistency | -0.0062 | (-0.0450, 0.0326) | 0.5997 | -0.0062 | (-0.0315, 0.0278) | 0.6507 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_proposed_raw | objective_completion_support | 0.0315 | (0.0069, 0.0567) | 0.0070 | 0.0315 | (-0.0043, 0.0572) | 0.0340 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 0.0120 | (-0.0096, 0.0340) | 0.1503 | 0.0120 | (-0.0128, 0.0325) | 0.1240 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | 0.0131 | (-0.0057, 0.0316) | 0.0830 | 0.0131 | (-0.0000, 0.0239) | 0.0260 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.0004 | (-0.0833, 0.0951) | 0.4987 | 0.0004 | (-0.0781, 0.0826) | 0.4807 |
| controlled_alt_vs_proposed_raw | context_overlap | -0.0041 | (-0.0373, 0.0275) | 0.6207 | -0.0041 | (-0.0341, 0.0313) | 0.6230 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.0540 | (-0.0456, 0.1661) | 0.1557 | 0.0540 | (-0.0638, 0.1429) | 0.1780 |
| controlled_alt_vs_proposed_raw | persona_style | -0.0378 | (-0.0703, -0.0117) | 1.0000 | -0.0378 | (-0.0906, 0.0000) | 1.0000 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0214 | (-0.0454, 0.0029) | 0.9597 | -0.0214 | (-0.0603, 0.0214) | 0.8090 |
| controlled_alt_vs_proposed_raw | length_score | -0.0111 | (-0.0792, 0.0597) | 0.6337 | -0.0111 | (-0.0750, 0.0345) | 0.6573 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0583 | (0.0000, 0.1313) | 0.0577 | 0.0583 | (0.0159, 0.0942) | 0.0090 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.0126 | (-0.0356, 0.0597) | 0.3113 | 0.0126 | (-0.0306, 0.0383) | 0.2640 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.0790 | (0.0262, 0.1385) | 0.0010 | 0.0790 | (0.0173, 0.1379) | 0.0067 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.0292 | (-0.0445, 0.1106) | 0.2227 | 0.0292 | (-0.1029, 0.1408) | 0.3463 |
| controlled_alt_vs_candidate_no_context | naturalness | -0.0082 | (-0.0292, 0.0149) | 0.7677 | -0.0082 | (-0.0379, 0.0085) | 0.6963 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 0.0729 | (0.0255, 0.1280) | 0.0003 | 0.0729 | (0.0143, 0.1188) | 0.0067 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 0.0307 | (0.0066, 0.0600) | 0.0027 | 0.0307 | (0.0160, 0.0440) | 0.0000 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_candidate_no_context | objective_completion_support | 0.0431 | (0.0128, 0.0714) | 0.0020 | 0.0431 | (-0.0011, 0.0836) | 0.0343 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 0.0337 | (0.0150, 0.0551) | 0.0000 | 0.0337 | (0.0047, 0.0582) | 0.0097 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | 0.0080 | (-0.0121, 0.0272) | 0.2080 | 0.0080 | (-0.0105, 0.0265) | 0.2237 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.1027 | (0.0345, 0.1780) | 0.0000 | 0.1027 | (0.0207, 0.1822) | 0.0110 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0238 | (-0.0007, 0.0505) | 0.0287 | 0.0238 | (-0.0050, 0.0423) | 0.0633 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.0421 | (-0.0446, 0.1373) | 0.1820 | 0.0421 | (-0.1257, 0.1808) | 0.3097 |
| controlled_alt_vs_candidate_no_context | persona_style | -0.0221 | (-0.0469, -0.0039) | 1.0000 | -0.0221 | (-0.0531, 0.0000) | 1.0000 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0176 | (-0.0411, 0.0057) | 0.9303 | -0.0176 | (-0.0553, 0.0193) | 0.7780 |
| controlled_alt_vs_candidate_no_context | length_score | -0.0347 | (-0.1208, 0.0569) | 0.7757 | -0.0347 | (-0.1617, 0.0611) | 0.7157 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.0583 | (0.0146, 0.1167) | 0.0120 | 0.0583 | (0.0000, 0.1000) | 0.0760 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.0463 | (0.0035, 0.0904) | 0.0177 | 0.0463 | (-0.0166, 0.0962) | 0.0623 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 13 | 2 | 9 | 0.7292 | 0.8667 |
| proposed_vs_candidate_no_context | persona_consistency | 4 | 3 | 17 | 0.5208 | 0.5714 |
| proposed_vs_candidate_no_context | naturalness | 8 | 7 | 9 | 0.5208 | 0.5333 |
| proposed_vs_candidate_no_context | quest_state_correctness | 14 | 1 | 9 | 0.7708 | 0.9333 |
| proposed_vs_candidate_no_context | lore_consistency | 8 | 3 | 13 | 0.6042 | 0.7273 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| proposed_vs_candidate_no_context | objective_completion_support | 6 | 3 | 15 | 0.5625 | 0.6667 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 11 | 4 | 9 | 0.6458 | 0.7333 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | 5 | 6 | 13 | 0.4792 | 0.4545 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 9 | 0 | 15 | 0.6875 | 1.0000 |
| proposed_vs_candidate_no_context | context_overlap | 11 | 4 | 9 | 0.6458 | 0.7333 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 3 | 3 | 18 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | persona_style | 2 | 0 | 22 | 0.5417 | 1.0000 |
| proposed_vs_candidate_no_context | distinct1 | 8 | 4 | 12 | 0.5833 | 0.6667 |
| proposed_vs_candidate_no_context | length_score | 6 | 7 | 11 | 0.4792 | 0.4615 |
| proposed_vs_candidate_no_context | sentence_score | 3 | 3 | 18 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | overall_quality | 11 | 4 | 9 | 0.6458 | 0.7333 |
| controlled_vs_proposed_raw | context_relevance | 9 | 12 | 3 | 0.4375 | 0.4286 |
| controlled_vs_proposed_raw | persona_consistency | 11 | 6 | 7 | 0.6042 | 0.6471 |
| controlled_vs_proposed_raw | naturalness | 13 | 8 | 3 | 0.6042 | 0.6190 |
| controlled_vs_proposed_raw | quest_state_correctness | 9 | 12 | 3 | 0.4375 | 0.4286 |
| controlled_vs_proposed_raw | lore_consistency | 6 | 6 | 12 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_vs_proposed_raw | objective_completion_support | 9 | 4 | 11 | 0.6042 | 0.6923 |
| controlled_vs_proposed_raw | gameplay_usefulness | 12 | 9 | 3 | 0.5625 | 0.5714 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 9 | 5 | 10 | 0.5833 | 0.6429 |
| controlled_vs_proposed_raw | context_keyword_coverage | 8 | 6 | 10 | 0.5417 | 0.5714 |
| controlled_vs_proposed_raw | context_overlap | 9 | 12 | 3 | 0.4375 | 0.4286 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 11 | 6 | 7 | 0.6042 | 0.6471 |
| controlled_vs_proposed_raw | persona_style | 1 | 5 | 18 | 0.4167 | 0.1667 |
| controlled_vs_proposed_raw | distinct1 | 9 | 11 | 4 | 0.4583 | 0.4500 |
| controlled_vs_proposed_raw | length_score | 12 | 7 | 5 | 0.6042 | 0.6316 |
| controlled_vs_proposed_raw | sentence_score | 5 | 0 | 19 | 0.6042 | 1.0000 |
| controlled_vs_proposed_raw | overall_quality | 12 | 9 | 3 | 0.5625 | 0.5714 |
| controlled_vs_candidate_no_context | context_relevance | 13 | 5 | 6 | 0.6667 | 0.7222 |
| controlled_vs_candidate_no_context | persona_consistency | 10 | 5 | 9 | 0.6042 | 0.6667 |
| controlled_vs_candidate_no_context | naturalness | 10 | 9 | 5 | 0.5208 | 0.5263 |
| controlled_vs_candidate_no_context | quest_state_correctness | 14 | 5 | 5 | 0.6875 | 0.7368 |
| controlled_vs_candidate_no_context | lore_consistency | 8 | 2 | 14 | 0.6250 | 0.8000 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_vs_candidate_no_context | objective_completion_support | 9 | 2 | 13 | 0.6458 | 0.8182 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 16 | 3 | 5 | 0.7708 | 0.8421 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 8 | 4 | 12 | 0.5833 | 0.6667 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 12 | 2 | 10 | 0.7083 | 0.8571 |
| controlled_vs_candidate_no_context | context_overlap | 13 | 5 | 6 | 0.6667 | 0.7222 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 10 | 5 | 9 | 0.6042 | 0.6667 |
| controlled_vs_candidate_no_context | persona_style | 2 | 4 | 18 | 0.4583 | 0.3333 |
| controlled_vs_candidate_no_context | distinct1 | 10 | 9 | 5 | 0.5208 | 0.5263 |
| controlled_vs_candidate_no_context | length_score | 10 | 9 | 5 | 0.5208 | 0.5263 |
| controlled_vs_candidate_no_context | sentence_score | 5 | 0 | 19 | 0.6042 | 1.0000 |
| controlled_vs_candidate_no_context | overall_quality | 15 | 4 | 5 | 0.7292 | 0.7895 |
| controlled_alt_vs_controlled_default | context_relevance | 5 | 8 | 11 | 0.4375 | 0.3846 |
| controlled_alt_vs_controlled_default | persona_consistency | 3 | 4 | 17 | 0.4792 | 0.4286 |
| controlled_alt_vs_controlled_default | naturalness | 6 | 7 | 11 | 0.4792 | 0.4615 |
| controlled_alt_vs_controlled_default | quest_state_correctness | 6 | 7 | 11 | 0.4792 | 0.4615 |
| controlled_alt_vs_controlled_default | lore_consistency | 4 | 7 | 13 | 0.4375 | 0.3636 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_alt_vs_controlled_default | objective_completion_support | 6 | 7 | 11 | 0.4792 | 0.4615 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | 4 | 9 | 11 | 0.3958 | 0.3077 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | 7 | 6 | 11 | 0.5208 | 0.5385 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 5 | 8 | 11 | 0.4375 | 0.3846 |
| controlled_alt_vs_controlled_default | context_overlap | 4 | 8 | 12 | 0.4167 | 0.3333 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 3 | 3 | 18 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | persona_style | 0 | 2 | 22 | 0.4583 | 0.0000 |
| controlled_alt_vs_controlled_default | distinct1 | 5 | 8 | 11 | 0.4375 | 0.3846 |
| controlled_alt_vs_controlled_default | length_score | 5 | 7 | 12 | 0.4583 | 0.4167 |
| controlled_alt_vs_controlled_default | sentence_score | 0 | 1 | 23 | 0.4792 | 0.0000 |
| controlled_alt_vs_controlled_default | overall_quality | 5 | 8 | 11 | 0.4375 | 0.3846 |
| controlled_alt_vs_proposed_raw | context_relevance | 8 | 13 | 3 | 0.3958 | 0.3810 |
| controlled_alt_vs_proposed_raw | persona_consistency | 10 | 7 | 7 | 0.5625 | 0.5882 |
| controlled_alt_vs_proposed_raw | naturalness | 9 | 12 | 3 | 0.4375 | 0.4286 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 8 | 13 | 3 | 0.3958 | 0.3810 |
| controlled_alt_vs_proposed_raw | lore_consistency | 6 | 8 | 10 | 0.4583 | 0.4286 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_alt_vs_proposed_raw | objective_completion_support | 10 | 3 | 11 | 0.6458 | 0.7692 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 12 | 9 | 3 | 0.5625 | 0.5714 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | 11 | 5 | 8 | 0.6250 | 0.6875 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 6 | 10 | 8 | 0.4167 | 0.3750 |
| controlled_alt_vs_proposed_raw | context_overlap | 7 | 14 | 3 | 0.3542 | 0.3333 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 10 | 7 | 7 | 0.5625 | 0.5882 |
| controlled_alt_vs_proposed_raw | persona_style | 0 | 6 | 18 | 0.3750 | 0.0000 |
| controlled_alt_vs_proposed_raw | distinct1 | 9 | 12 | 3 | 0.4375 | 0.4286 |
| controlled_alt_vs_proposed_raw | length_score | 9 | 11 | 4 | 0.4583 | 0.4500 |
| controlled_alt_vs_proposed_raw | sentence_score | 5 | 1 | 18 | 0.5833 | 0.8333 |
| controlled_alt_vs_proposed_raw | overall_quality | 11 | 10 | 3 | 0.5208 | 0.5238 |
| controlled_alt_vs_candidate_no_context | context_relevance | 11 | 8 | 5 | 0.5625 | 0.5789 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 10 | 6 | 8 | 0.5833 | 0.6250 |
| controlled_alt_vs_candidate_no_context | naturalness | 7 | 13 | 4 | 0.3750 | 0.3500 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 13 | 7 | 4 | 0.6250 | 0.6500 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 7 | 6 | 11 | 0.5208 | 0.5385 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_alt_vs_candidate_no_context | objective_completion_support | 12 | 3 | 9 | 0.6875 | 0.8000 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 15 | 5 | 4 | 0.7083 | 0.7500 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | 9 | 6 | 9 | 0.5625 | 0.6000 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 9 | 2 | 13 | 0.6458 | 0.8182 |
| controlled_alt_vs_candidate_no_context | context_overlap | 11 | 8 | 5 | 0.5625 | 0.5789 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 10 | 6 | 8 | 0.5833 | 0.6250 |
| controlled_alt_vs_candidate_no_context | persona_style | 0 | 4 | 20 | 0.4167 | 0.0000 |
| controlled_alt_vs_candidate_no_context | distinct1 | 7 | 13 | 4 | 0.3750 | 0.3500 |
| controlled_alt_vs_candidate_no_context | length_score | 9 | 11 | 4 | 0.4583 | 0.4500 |
| controlled_alt_vs_candidate_no_context | sentence_score | 4 | 0 | 20 | 0.5833 | 1.0000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 15 | 5 | 4 | 0.7083 | 0.7500 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.1250 | 0.2500 | 0.7500 |
| proposed_contextual_controlled_tuned | 0.0000 | 0.0000 | 0.2083 | 0.1250 | 0.8750 |
| proposed_contextual | 0.0000 | 0.0000 | 0.4167 | 0.0000 | 0.0000 |
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
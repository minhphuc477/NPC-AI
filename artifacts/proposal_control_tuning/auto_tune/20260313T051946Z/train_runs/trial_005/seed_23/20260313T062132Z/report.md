# Proposal Alignment Evaluation Report

- Run ID: `20260313T062132Z`
- Generated: `2026-03-13T06:27:27.783498+00:00`
- Scenarios: `artifacts\proposal_control_tuning\auto_tune\20260313T051946Z\train_runs\trial_005\seed_23\20260313T062132Z\scenarios.jsonl`
- Scenario count: `24`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_tuned`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0874 (0.0458, 0.1359) | 0.2225 (0.1810, 0.2649) | 0.8840 (0.8696, 0.8989) | 0.2927 (0.2626, 0.3260) | n/a |
| proposed_contextual_controlled_tuned | 0.0917 (0.0474, 0.1379) | 0.2425 (0.1936, 0.2970) | 0.8744 (0.8543, 0.8942) | 0.3008 (0.2704, 0.3332) | n/a |
| proposed_contextual | 0.0731 (0.0350, 0.1200) | 0.2539 (0.1947, 0.3184) | 0.8788 (0.8621, 0.8969) | 0.2963 (0.2603, 0.3377) | n/a |
| candidate_no_context | 0.0296 (0.0173, 0.0427) | 0.2576 (0.1926, 0.3274) | 0.8742 (0.8552, 0.8924) | 0.2764 (0.2471, 0.3071) | n/a |

## Game-facing Outcome Metrics (mean, 95% CI)
| Arm | Quest-state Correctness | Lore Consistency | Contradiction Safety | Objective Completion Support | Gameplay Usefulness | Time-pressure Acceptability |
|---|---:|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.1746 (0.1378, 0.2151) | 0.0417 (0.0113, 0.0780) | 1.0000 (1.0000, 1.0000) | 0.0746 (0.0514, 0.0972) | 0.3056 (0.2926, 0.3194) | 0.2953 (0.2814, 0.3098) |
| proposed_contextual_controlled_tuned | 0.1756 (0.1381, 0.2156) | 0.0315 (0.0084, 0.0610) | 1.0000 (1.0000, 1.0000) | 0.0832 (0.0576, 0.1063) | 0.3039 (0.2886, 0.3194) | 0.2918 (0.2732, 0.3091) |
| proposed_contextual | 0.1613 (0.1284, 0.2038) | 0.0235 (0.0068, 0.0430) | 1.0000 (1.0000, 1.0000) | 0.0647 (0.0411, 0.0886) | 0.2958 (0.2812, 0.3094) | 0.2886 (0.2740, 0.3036) |
| candidate_no_context | 0.1220 (0.1138, 0.1308) | 0.0069 (0.0018, 0.0134) | 1.0000 (1.0000, 1.0000) | 0.0594 (0.0367, 0.0826) | 0.2786 (0.2652, 0.2913) | 0.2900 (0.2761, 0.3059) |

- Multi-turn contradiction rate is reported as `1 - contradiction_safety` in row-level outputs.

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0435 | 1.4701 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0037 | -0.0142 |
| proposed_vs_candidate_no_context | naturalness | 0.0045 | 0.0052 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.0393 | 0.3221 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0166 | 2.3967 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | 0.0052 | 0.0876 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 0.0171 | 0.0615 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | -0.0014 | -0.0048 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0568 | 2.1429 |
| proposed_vs_candidate_no_context | context_overlap | 0.0124 | 0.3365 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0020 | -0.0103 |
| proposed_vs_candidate_no_context | persona_style | -0.0104 | -0.0201 |
| proposed_vs_candidate_no_context | distinct1 | 0.0027 | 0.0028 |
| proposed_vs_candidate_no_context | length_score | 0.0028 | 0.0051 |
| proposed_vs_candidate_no_context | sentence_score | 0.0292 | 0.0320 |
| proposed_vs_candidate_no_context | overall_quality | 0.0199 | 0.0719 |
| controlled_vs_proposed_raw | context_relevance | 0.0143 | 0.1961 |
| controlled_vs_proposed_raw | persona_consistency | -0.0314 | -0.1236 |
| controlled_vs_proposed_raw | naturalness | 0.0052 | 0.0059 |
| controlled_vs_proposed_raw | quest_state_correctness | 0.0133 | 0.0825 |
| controlled_vs_proposed_raw | lore_consistency | 0.0182 | 0.7731 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0099 | 0.1536 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0098 | 0.0333 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 0.0067 | 0.0232 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.0193 | 0.2318 |
| controlled_vs_proposed_raw | context_overlap | 0.0027 | 0.0546 |
| controlled_vs_proposed_raw | persona_keyword_coverage | -0.0327 | -0.1719 |
| controlled_vs_proposed_raw | persona_style | -0.0260 | -0.0513 |
| controlled_vs_proposed_raw | distinct1 | -0.0113 | -0.0120 |
| controlled_vs_proposed_raw | length_score | 0.0194 | 0.0358 |
| controlled_vs_proposed_raw | sentence_score | 0.0583 | 0.0619 |
| controlled_vs_proposed_raw | overall_quality | -0.0036 | -0.0123 |
| controlled_vs_candidate_no_context | context_relevance | 0.0578 | 1.9545 |
| controlled_vs_candidate_no_context | persona_consistency | -0.0351 | -0.1361 |
| controlled_vs_candidate_no_context | naturalness | 0.0097 | 0.0111 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.0526 | 0.4312 |
| controlled_vs_candidate_no_context | lore_consistency | 0.0348 | 5.0229 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | 0.0151 | 0.2547 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0270 | 0.0969 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 0.0053 | 0.0183 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.0761 | 2.8714 |
| controlled_vs_candidate_no_context | context_overlap | 0.0150 | 0.4095 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | -0.0347 | -0.1804 |
| controlled_vs_candidate_no_context | persona_style | -0.0365 | -0.0704 |
| controlled_vs_candidate_no_context | distinct1 | -0.0087 | -0.0092 |
| controlled_vs_candidate_no_context | length_score | 0.0222 | 0.0411 |
| controlled_vs_candidate_no_context | sentence_score | 0.0875 | 0.0959 |
| controlled_vs_candidate_no_context | overall_quality | 0.0162 | 0.0588 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0043 | 0.0494 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0200 | 0.0897 |
| controlled_alt_vs_controlled_default | naturalness | -0.0096 | -0.0108 |
| controlled_alt_vs_controlled_default | quest_state_correctness | 0.0010 | 0.0057 |
| controlled_alt_vs_controlled_default | lore_consistency | -0.0102 | -0.2453 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_controlled_default | objective_completion_support | 0.0086 | 0.1155 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | -0.0017 | -0.0055 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | -0.0035 | -0.0119 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0083 | 0.0812 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0051 | -0.0976 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0302 | 0.1912 |
| controlled_alt_vs_controlled_default | persona_style | -0.0208 | -0.0432 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0027 | 0.0029 |
| controlled_alt_vs_controlled_default | length_score | -0.0458 | -0.0815 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0146 | -0.0146 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0081 | 0.0277 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.0186 | 0.2552 |
| controlled_alt_vs_proposed_raw | persona_consistency | -0.0114 | -0.0450 |
| controlled_alt_vs_proposed_raw | naturalness | -0.0044 | -0.0050 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 0.0143 | 0.0886 |
| controlled_alt_vs_proposed_raw | lore_consistency | 0.0080 | 0.3383 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_proposed_raw | objective_completion_support | 0.0185 | 0.2868 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 0.0082 | 0.0276 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | 0.0032 | 0.0110 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.0277 | 0.3318 |
| controlled_alt_vs_proposed_raw | context_overlap | -0.0024 | -0.0483 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | -0.0026 | -0.0135 |
| controlled_alt_vs_proposed_raw | persona_style | -0.0469 | -0.0923 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0087 | -0.0092 |
| controlled_alt_vs_proposed_raw | length_score | -0.0264 | -0.0486 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0437 | 0.0465 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.0045 | 0.0151 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.0621 | 2.1005 |
| controlled_alt_vs_candidate_no_context | persona_consistency | -0.0151 | -0.0587 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0002 | 0.0002 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 0.0536 | 0.4393 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 0.0246 | 3.5457 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_candidate_no_context | objective_completion_support | 0.0238 | 0.3995 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 0.0253 | 0.0909 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | 0.0018 | 0.0062 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.0845 | 3.1857 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0100 | 0.2720 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | -0.0046 | -0.0237 |
| controlled_alt_vs_candidate_no_context | persona_style | -0.0573 | -0.1106 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0060 | -0.0064 |
| controlled_alt_vs_candidate_no_context | length_score | -0.0236 | -0.0437 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.0729 | 0.0799 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.0244 | 0.0881 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0435 | (0.0058, 0.0919) | 0.0063 | 0.0435 | (0.0124, 0.0698) | 0.0083 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0037 | (-0.0622, 0.0484) | 0.5313 | -0.0037 | (-0.0908, 0.0667) | 0.5590 |
| proposed_vs_candidate_no_context | naturalness | 0.0045 | (-0.0242, 0.0340) | 0.3887 | 0.0045 | (-0.0401, 0.0526) | 0.4680 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.0393 | (0.0067, 0.0807) | 0.0057 | 0.0393 | (0.0128, 0.0618) | 0.0113 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0166 | (-0.0011, 0.0384) | 0.0353 | 0.0166 | (-0.0027, 0.0360) | 0.0810 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | 0.0052 | (-0.0147, 0.0252) | 0.3133 | 0.0052 | (-0.0035, 0.0185) | 0.1467 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 0.0171 | (-0.0015, 0.0370) | 0.0387 | 0.0171 | (-0.0006, 0.0391) | 0.0700 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | -0.0014 | (-0.0213, 0.0165) | 0.5323 | -0.0014 | (-0.0186, 0.0127) | 0.5643 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0568 | (0.0114, 0.1174) | 0.0050 | 0.0568 | (0.0165, 0.0909) | 0.0103 |
| proposed_vs_candidate_no_context | context_overlap | 0.0124 | (-0.0020, 0.0302) | 0.0520 | 0.0124 | (0.0029, 0.0214) | 0.0097 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0020 | (-0.0715, 0.0655) | 0.5087 | -0.0020 | (-0.1050, 0.0833) | 0.5570 |
| proposed_vs_candidate_no_context | persona_style | -0.0104 | (-0.0417, 0.0208) | 0.8110 | -0.0104 | (-0.0341, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | distinct1 | 0.0027 | (-0.0140, 0.0211) | 0.3937 | 0.0027 | (-0.0188, 0.0178) | 0.3790 |
| proposed_vs_candidate_no_context | length_score | 0.0028 | (-0.1222, 0.1250) | 0.4907 | 0.0028 | (-0.1731, 0.2115) | 0.4913 |
| proposed_vs_candidate_no_context | sentence_score | 0.0292 | (-0.0292, 0.0875) | 0.2240 | 0.0292 | (0.0000, 0.0808) | 0.3277 |
| proposed_vs_candidate_no_context | overall_quality | 0.0199 | (-0.0139, 0.0556) | 0.1237 | 0.0199 | (-0.0111, 0.0556) | 0.1957 |
| controlled_vs_proposed_raw | context_relevance | 0.0143 | (-0.0192, 0.0523) | 0.2200 | 0.0143 | (-0.0150, 0.0342) | 0.1593 |
| controlled_vs_proposed_raw | persona_consistency | -0.0314 | (-0.0691, 0.0067) | 0.9503 | -0.0314 | (-0.0913, 0.0178) | 0.8870 |
| controlled_vs_proposed_raw | naturalness | 0.0052 | (-0.0080, 0.0189) | 0.2223 | 0.0052 | (-0.0121, 0.0256) | 0.3400 |
| controlled_vs_proposed_raw | quest_state_correctness | 0.0133 | (-0.0180, 0.0460) | 0.2123 | 0.0133 | (-0.0121, 0.0336) | 0.1320 |
| controlled_vs_proposed_raw | lore_consistency | 0.0182 | (-0.0069, 0.0469) | 0.0930 | 0.0182 | (0.0033, 0.0299) | 0.0140 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0099 | (-0.0053, 0.0292) | 0.1223 | 0.0099 | (-0.0042, 0.0273) | 0.2103 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0098 | (-0.0040, 0.0263) | 0.0913 | 0.0098 | (-0.0106, 0.0328) | 0.2410 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 0.0067 | (0.0001, 0.0145) | 0.0237 | 0.0067 | (-0.0001, 0.0172) | 0.0770 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.0193 | (-0.0227, 0.0655) | 0.1967 | 0.0193 | (-0.0136, 0.0429) | 0.1217 |
| controlled_vs_proposed_raw | context_overlap | 0.0027 | (-0.0156, 0.0232) | 0.4050 | 0.0027 | (-0.0218, 0.0182) | 0.3873 |
| controlled_vs_proposed_raw | persona_keyword_coverage | -0.0327 | (-0.0764, 0.0139) | 0.9167 | -0.0327 | (-0.1000, 0.0222) | 0.8637 |
| controlled_vs_proposed_raw | persona_style | -0.0260 | (-0.0638, 0.0117) | 0.9250 | -0.0260 | (-0.0625, 0.0000) | 1.0000 |
| controlled_vs_proposed_raw | distinct1 | -0.0113 | (-0.0300, 0.0070) | 0.8827 | -0.0113 | (-0.0364, 0.0230) | 0.7177 |
| controlled_vs_proposed_raw | length_score | 0.0194 | (-0.0458, 0.0889) | 0.2967 | 0.0194 | (-0.0600, 0.1024) | 0.3580 |
| controlled_vs_proposed_raw | sentence_score | 0.0583 | (0.0146, 0.1167) | 0.0097 | 0.0583 | (0.0000, 0.1250) | 0.0830 |
| controlled_vs_proposed_raw | overall_quality | -0.0036 | (-0.0275, 0.0208) | 0.6143 | -0.0036 | (-0.0428, 0.0250) | 0.6160 |
| controlled_vs_candidate_no_context | context_relevance | 0.0578 | (0.0165, 0.1075) | 0.0013 | 0.0578 | (-0.0013, 0.1002) | 0.0310 |
| controlled_vs_candidate_no_context | persona_consistency | -0.0351 | (-0.1006, 0.0246) | 0.8633 | -0.0351 | (-0.1508, 0.0476) | 0.7587 |
| controlled_vs_candidate_no_context | naturalness | 0.0097 | (-0.0140, 0.0365) | 0.2280 | 0.0097 | (-0.0254, 0.0462) | 0.3313 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.0526 | (0.0163, 0.0939) | 0.0017 | 0.0526 | (-0.0007, 0.0907) | 0.0260 |
| controlled_vs_candidate_no_context | lore_consistency | 0.0348 | (0.0061, 0.0706) | 0.0037 | 0.0348 | (0.0048, 0.0658) | 0.0127 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | 0.0151 | (-0.0033, 0.0364) | 0.0607 | 0.0151 | (0.0037, 0.0262) | 0.0113 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0270 | (0.0097, 0.0464) | 0.0017 | 0.0270 | (-0.0038, 0.0490) | 0.0330 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 0.0053 | (-0.0105, 0.0210) | 0.2390 | 0.0053 | (-0.0030, 0.0141) | 0.1087 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.0761 | (0.0227, 0.1405) | 0.0020 | 0.0761 | (0.0041, 0.1292) | 0.0223 |
| controlled_vs_candidate_no_context | context_overlap | 0.0150 | (-0.0048, 0.0365) | 0.0670 | 0.0150 | (-0.0139, 0.0355) | 0.1157 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | -0.0347 | (-0.1111, 0.0337) | 0.8370 | -0.0347 | (-0.1667, 0.0577) | 0.7543 |
| controlled_vs_candidate_no_context | persona_style | -0.0365 | (-0.0859, -0.0039) | 1.0000 | -0.0365 | (-0.0875, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | distinct1 | -0.0087 | (-0.0230, 0.0049) | 0.8913 | -0.0087 | (-0.0243, 0.0129) | 0.7997 |
| controlled_vs_candidate_no_context | length_score | 0.0222 | (-0.0833, 0.1431) | 0.3717 | 0.0222 | (-0.1200, 0.1962) | 0.4163 |
| controlled_vs_candidate_no_context | sentence_score | 0.0875 | (0.0292, 0.1458) | 0.0010 | 0.0875 | (0.0000, 0.1500) | 0.0727 |
| controlled_vs_candidate_no_context | overall_quality | 0.0162 | (-0.0214, 0.0521) | 0.1897 | 0.0162 | (-0.0472, 0.0619) | 0.2980 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0043 | (-0.0561, 0.0561) | 0.4423 | 0.0043 | (-0.0598, 0.0504) | 0.4080 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0200 | (-0.0361, 0.0741) | 0.2293 | 0.0200 | (-0.0208, 0.0495) | 0.1590 |
| controlled_alt_vs_controlled_default | naturalness | -0.0096 | (-0.0270, 0.0065) | 0.8680 | -0.0096 | (-0.0286, 0.0064) | 0.8240 |
| controlled_alt_vs_controlled_default | quest_state_correctness | 0.0010 | (-0.0469, 0.0517) | 0.4840 | 0.0010 | (-0.0518, 0.0401) | 0.3990 |
| controlled_alt_vs_controlled_default | lore_consistency | -0.0102 | (-0.0500, 0.0232) | 0.7030 | -0.0102 | (-0.0502, 0.0297) | 0.6813 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | objective_completion_support | 0.0086 | (-0.0150, 0.0340) | 0.2247 | 0.0086 | (-0.0036, 0.0242) | 0.0893 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | -0.0017 | (-0.0104, 0.0068) | 0.6433 | -0.0017 | (-0.0053, 0.0015) | 0.8240 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | -0.0035 | (-0.0251, 0.0146) | 0.6353 | -0.0035 | (-0.0208, 0.0140) | 0.6230 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0083 | (-0.0645, 0.0811) | 0.4010 | 0.0083 | (-0.0694, 0.0669) | 0.3793 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0051 | (-0.0255, 0.0161) | 0.6840 | -0.0051 | (-0.0244, 0.0142) | 0.6840 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0302 | (-0.0383, 0.0980) | 0.1803 | 0.0302 | (-0.0071, 0.0654) | 0.0593 |
| controlled_alt_vs_controlled_default | persona_style | -0.0208 | (-0.0521, 0.0000) | 1.0000 | -0.0208 | (-0.0682, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0027 | (-0.0119, 0.0186) | 0.3667 | 0.0027 | (-0.0027, 0.0081) | 0.1840 |
| controlled_alt_vs_controlled_default | length_score | -0.0458 | (-0.1320, 0.0306) | 0.8663 | -0.0458 | (-0.1269, 0.0306) | 0.8350 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0146 | (-0.0437, 0.0000) | 1.0000 | -0.0146 | (-0.0404, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0081 | (-0.0351, 0.0474) | 0.3343 | 0.0081 | (-0.0322, 0.0360) | 0.2893 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.0186 | (-0.0395, 0.0759) | 0.2507 | 0.0186 | (-0.0447, 0.0721) | 0.2850 |
| controlled_alt_vs_proposed_raw | persona_consistency | -0.0114 | (-0.0807, 0.0503) | 0.6230 | -0.0114 | (-0.1021, 0.0533) | 0.6407 |
| controlled_alt_vs_proposed_raw | naturalness | -0.0044 | (-0.0248, 0.0176) | 0.6590 | -0.0044 | (-0.0378, 0.0310) | 0.5880 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 0.0143 | (-0.0379, 0.0653) | 0.2857 | 0.0143 | (-0.0407, 0.0602) | 0.3213 |
| controlled_alt_vs_proposed_raw | lore_consistency | 0.0080 | (-0.0199, 0.0401) | 0.3043 | 0.0080 | (-0.0261, 0.0444) | 0.3697 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_proposed_raw | objective_completion_support | 0.0185 | (-0.0041, 0.0440) | 0.0633 | 0.0185 | (-0.0019, 0.0405) | 0.0523 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 0.0082 | (-0.0075, 0.0278) | 0.1953 | 0.0082 | (-0.0131, 0.0331) | 0.3377 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | 0.0032 | (-0.0187, 0.0226) | 0.3793 | 0.0032 | (-0.0199, 0.0262) | 0.3857 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.0277 | (-0.0481, 0.1034) | 0.2377 | 0.0277 | (-0.0690, 0.0955) | 0.2577 |
| controlled_alt_vs_proposed_raw | context_overlap | -0.0024 | (-0.0211, 0.0183) | 0.6217 | -0.0024 | (-0.0259, 0.0175) | 0.5830 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | -0.0026 | (-0.0810, 0.0722) | 0.5177 | -0.0026 | (-0.0995, 0.0667) | 0.5087 |
| controlled_alt_vs_proposed_raw | persona_style | -0.0469 | (-0.0846, -0.0156) | 1.0000 | -0.0469 | (-0.1193, 0.0000) | 1.0000 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0087 | (-0.0304, 0.0101) | 0.7923 | -0.0087 | (-0.0325, 0.0218) | 0.6923 |
| controlled_alt_vs_proposed_raw | length_score | -0.0264 | (-0.1222, 0.0778) | 0.7050 | -0.0264 | (-0.1500, 0.1223) | 0.6427 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0437 | (0.0000, 0.0875) | 0.0423 | 0.0437 | (0.0000, 0.1212) | 0.3197 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.0045 | (-0.0441, 0.0493) | 0.4397 | 0.0045 | (-0.0570, 0.0483) | 0.3627 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.0621 | (0.0170, 0.1103) | 0.0030 | 0.0621 | (-0.0089, 0.1129) | 0.0353 |
| controlled_alt_vs_candidate_no_context | persona_consistency | -0.0151 | (-0.0936, 0.0665) | 0.6483 | -0.0151 | (-0.1643, 0.0944) | 0.5787 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0002 | (-0.0219, 0.0259) | 0.5090 | 0.0002 | (-0.0254, 0.0207) | 0.4847 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 0.0536 | (0.0129, 0.0960) | 0.0037 | 0.0536 | (-0.0081, 0.0994) | 0.0450 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 0.0246 | (0.0025, 0.0527) | 0.0093 | 0.0246 | (0.0028, 0.0489) | 0.0087 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_candidate_no_context | objective_completion_support | 0.0238 | (-0.0042, 0.0516) | 0.0487 | 0.0238 | (0.0001, 0.0458) | 0.0183 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 0.0253 | (0.0073, 0.0468) | 0.0020 | 0.0253 | (-0.0060, 0.0471) | 0.0453 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | 0.0018 | (-0.0206, 0.0225) | 0.4380 | 0.0018 | (-0.0168, 0.0248) | 0.4610 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.0845 | (0.0265, 0.1439) | 0.0007 | 0.0845 | (-0.0091, 0.1513) | 0.0367 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0100 | (-0.0073, 0.0282) | 0.1383 | 0.0100 | (-0.0140, 0.0270) | 0.1977 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | -0.0046 | (-0.0945, 0.0891) | 0.5447 | -0.0046 | (-0.1710, 0.1179) | 0.5013 |
| controlled_alt_vs_candidate_no_context | persona_style | -0.0573 | (-0.1081, -0.0182) | 1.0000 | -0.0573 | (-0.1534, 0.0000) | 1.0000 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0060 | (-0.0266, 0.0134) | 0.7133 | -0.0060 | (-0.0230, 0.0141) | 0.7020 |
| controlled_alt_vs_candidate_no_context | length_score | -0.0236 | (-0.1306, 0.1014) | 0.6813 | -0.0236 | (-0.1300, 0.0857) | 0.6850 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.0729 | (0.0000, 0.1458) | 0.0360 | 0.0729 | (0.0000, 0.1375) | 0.0760 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.0244 | (-0.0185, 0.0683) | 0.1357 | 0.0244 | (-0.0681, 0.0919) | 0.2983 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 8 | 5 | 11 | 0.5625 | 0.6154 |
| proposed_vs_candidate_no_context | persona_consistency | 5 | 4 | 15 | 0.5208 | 0.5556 |
| proposed_vs_candidate_no_context | naturalness | 6 | 8 | 10 | 0.4583 | 0.4286 |
| proposed_vs_candidate_no_context | quest_state_correctness | 8 | 6 | 10 | 0.5417 | 0.5714 |
| proposed_vs_candidate_no_context | lore_consistency | 5 | 5 | 14 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| proposed_vs_candidate_no_context | objective_completion_support | 5 | 5 | 14 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 10 | 4 | 10 | 0.6250 | 0.7143 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | 7 | 6 | 11 | 0.5208 | 0.5385 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 5 | 1 | 18 | 0.5833 | 0.8333 |
| proposed_vs_candidate_no_context | context_overlap | 6 | 7 | 11 | 0.4792 | 0.4615 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 5 | 4 | 15 | 0.5208 | 0.5556 |
| proposed_vs_candidate_no_context | persona_style | 1 | 2 | 21 | 0.4792 | 0.3333 |
| proposed_vs_candidate_no_context | distinct1 | 6 | 4 | 14 | 0.5417 | 0.6000 |
| proposed_vs_candidate_no_context | length_score | 8 | 6 | 10 | 0.5417 | 0.5714 |
| proposed_vs_candidate_no_context | sentence_score | 3 | 1 | 20 | 0.5417 | 0.7500 |
| proposed_vs_candidate_no_context | overall_quality | 8 | 6 | 10 | 0.5417 | 0.5714 |
| controlled_vs_proposed_raw | context_relevance | 8 | 10 | 6 | 0.4583 | 0.4444 |
| controlled_vs_proposed_raw | persona_consistency | 4 | 11 | 9 | 0.3542 | 0.2667 |
| controlled_vs_proposed_raw | naturalness | 9 | 9 | 6 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | quest_state_correctness | 9 | 9 | 6 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | lore_consistency | 6 | 3 | 15 | 0.5625 | 0.6667 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_vs_proposed_raw | objective_completion_support | 4 | 3 | 17 | 0.5208 | 0.5714 |
| controlled_vs_proposed_raw | gameplay_usefulness | 8 | 10 | 6 | 0.4583 | 0.4444 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 6 | 2 | 16 | 0.5833 | 0.7500 |
| controlled_vs_proposed_raw | context_keyword_coverage | 6 | 6 | 12 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | context_overlap | 8 | 10 | 6 | 0.4583 | 0.4444 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 4 | 9 | 11 | 0.3958 | 0.3077 |
| controlled_vs_proposed_raw | persona_style | 1 | 6 | 17 | 0.3958 | 0.1429 |
| controlled_vs_proposed_raw | distinct1 | 10 | 7 | 7 | 0.5625 | 0.5882 |
| controlled_vs_proposed_raw | length_score | 7 | 10 | 7 | 0.4375 | 0.4118 |
| controlled_vs_proposed_raw | sentence_score | 4 | 0 | 20 | 0.5833 | 1.0000 |
| controlled_vs_proposed_raw | overall_quality | 8 | 10 | 6 | 0.4583 | 0.4444 |
| controlled_vs_candidate_no_context | context_relevance | 11 | 6 | 7 | 0.6042 | 0.6471 |
| controlled_vs_candidate_no_context | persona_consistency | 6 | 8 | 10 | 0.4583 | 0.4286 |
| controlled_vs_candidate_no_context | naturalness | 6 | 12 | 6 | 0.3750 | 0.3333 |
| controlled_vs_candidate_no_context | quest_state_correctness | 12 | 6 | 6 | 0.6250 | 0.6667 |
| controlled_vs_candidate_no_context | lore_consistency | 8 | 4 | 12 | 0.5833 | 0.6667 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_vs_candidate_no_context | objective_completion_support | 5 | 3 | 16 | 0.5417 | 0.6250 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 12 | 6 | 6 | 0.6250 | 0.6667 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 7 | 5 | 12 | 0.5417 | 0.5833 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 10 | 3 | 11 | 0.6458 | 0.7692 |
| controlled_vs_candidate_no_context | context_overlap | 9 | 8 | 7 | 0.5208 | 0.5294 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 6 | 8 | 10 | 0.4583 | 0.4286 |
| controlled_vs_candidate_no_context | persona_style | 0 | 5 | 19 | 0.3958 | 0.0000 |
| controlled_vs_candidate_no_context | distinct1 | 7 | 10 | 7 | 0.4375 | 0.4118 |
| controlled_vs_candidate_no_context | length_score | 6 | 12 | 6 | 0.3750 | 0.3333 |
| controlled_vs_candidate_no_context | sentence_score | 6 | 0 | 18 | 0.6250 | 1.0000 |
| controlled_vs_candidate_no_context | overall_quality | 11 | 7 | 6 | 0.5833 | 0.6111 |
| controlled_alt_vs_controlled_default | context_relevance | 6 | 7 | 11 | 0.4792 | 0.4615 |
| controlled_alt_vs_controlled_default | persona_consistency | 6 | 3 | 15 | 0.5625 | 0.6667 |
| controlled_alt_vs_controlled_default | naturalness | 5 | 8 | 11 | 0.4375 | 0.3846 |
| controlled_alt_vs_controlled_default | quest_state_correctness | 5 | 8 | 11 | 0.4375 | 0.3846 |
| controlled_alt_vs_controlled_default | lore_consistency | 5 | 5 | 14 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_alt_vs_controlled_default | objective_completion_support | 5 | 5 | 14 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | 7 | 6 | 11 | 0.5208 | 0.5385 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | 6 | 5 | 13 | 0.5208 | 0.5455 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 6 | 5 | 13 | 0.5208 | 0.5455 |
| controlled_alt_vs_controlled_default | context_overlap | 5 | 8 | 11 | 0.4375 | 0.3846 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 6 | 3 | 15 | 0.5625 | 0.6667 |
| controlled_alt_vs_controlled_default | persona_style | 0 | 2 | 22 | 0.4583 | 0.0000 |
| controlled_alt_vs_controlled_default | distinct1 | 7 | 5 | 12 | 0.5417 | 0.5833 |
| controlled_alt_vs_controlled_default | length_score | 5 | 8 | 11 | 0.4375 | 0.3846 |
| controlled_alt_vs_controlled_default | sentence_score | 0 | 1 | 23 | 0.4792 | 0.0000 |
| controlled_alt_vs_controlled_default | overall_quality | 9 | 4 | 11 | 0.6042 | 0.6923 |
| controlled_alt_vs_proposed_raw | context_relevance | 9 | 12 | 3 | 0.4375 | 0.4286 |
| controlled_alt_vs_proposed_raw | persona_consistency | 7 | 9 | 8 | 0.4583 | 0.4375 |
| controlled_alt_vs_proposed_raw | naturalness | 11 | 10 | 3 | 0.5208 | 0.5238 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 9 | 12 | 3 | 0.4375 | 0.4286 |
| controlled_alt_vs_proposed_raw | lore_consistency | 6 | 5 | 13 | 0.5208 | 0.5455 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_alt_vs_proposed_raw | objective_completion_support | 6 | 5 | 13 | 0.5208 | 0.5455 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 10 | 11 | 3 | 0.4792 | 0.4762 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | 8 | 5 | 11 | 0.5625 | 0.6154 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 6 | 5 | 13 | 0.5208 | 0.5455 |
| controlled_alt_vs_proposed_raw | context_overlap | 9 | 12 | 3 | 0.4375 | 0.4286 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 7 | 7 | 10 | 0.5000 | 0.5000 |
| controlled_alt_vs_proposed_raw | persona_style | 0 | 7 | 17 | 0.3542 | 0.0000 |
| controlled_alt_vs_proposed_raw | distinct1 | 11 | 9 | 4 | 0.5417 | 0.5500 |
| controlled_alt_vs_proposed_raw | length_score | 8 | 12 | 4 | 0.4167 | 0.4000 |
| controlled_alt_vs_proposed_raw | sentence_score | 3 | 0 | 21 | 0.5625 | 1.0000 |
| controlled_alt_vs_proposed_raw | overall_quality | 12 | 9 | 3 | 0.5625 | 0.5714 |
| controlled_alt_vs_candidate_no_context | context_relevance | 11 | 10 | 3 | 0.5208 | 0.5238 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 8 | 9 | 7 | 0.4792 | 0.4706 |
| controlled_alt_vs_candidate_no_context | naturalness | 9 | 11 | 4 | 0.4583 | 0.4500 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 12 | 9 | 3 | 0.5625 | 0.5714 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 7 | 4 | 13 | 0.5625 | 0.6364 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_alt_vs_candidate_no_context | objective_completion_support | 9 | 4 | 11 | 0.6042 | 0.6923 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 13 | 8 | 3 | 0.6042 | 0.6190 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | 8 | 7 | 9 | 0.5208 | 0.5333 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 9 | 3 | 12 | 0.6250 | 0.7500 |
| controlled_alt_vs_candidate_no_context | context_overlap | 10 | 11 | 3 | 0.4792 | 0.4762 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 8 | 8 | 8 | 0.5000 | 0.5000 |
| controlled_alt_vs_candidate_no_context | persona_style | 0 | 7 | 17 | 0.3542 | 0.0000 |
| controlled_alt_vs_candidate_no_context | distinct1 | 9 | 11 | 4 | 0.4583 | 0.4500 |
| controlled_alt_vs_candidate_no_context | length_score | 7 | 12 | 5 | 0.3958 | 0.3684 |
| controlled_alt_vs_candidate_no_context | sentence_score | 6 | 1 | 17 | 0.6042 | 0.8571 |
| controlled_alt_vs_candidate_no_context | overall_quality | 13 | 8 | 3 | 0.6042 | 0.6190 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.1250 | 0.2083 | 0.7917 |
| proposed_contextual_controlled_tuned | 0.0000 | 0.0000 | 0.1667 | 0.1250 | 0.8333 |
| proposed_contextual | 0.0000 | 0.0000 | 0.5417 | 0.0000 | 0.0000 |
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
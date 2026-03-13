# Proposal Alignment Evaluation Report

- Run ID: `20260313T005245Z`
- Generated: `2026-03-13T00:56:11.731567+00:00`
- Scenarios: `artifacts\proposal_control_tuning\auto_tune\20260313T004330Z\train_runs\trial_003\seed_19\20260313T005245Z\scenarios.jsonl`
- Scenario count: `12`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_tuned`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.1258 (0.0698, 0.1916) | 0.2947 (0.2163, 0.3747) | 0.8702 (0.8334, 0.9071) | 0.3321 (0.2884, 0.3800) | n/a |
| proposed_contextual_controlled_tuned | 0.1630 (0.1100, 0.2298) | 0.3649 (0.2524, 0.4765) | 0.8996 (0.8681, 0.9307) | 0.3819 (0.3203, 0.4453) | n/a |
| proposed_contextual | 0.0975 (0.0646, 0.1349) | 0.2333 (0.1607, 0.3111) | 0.8652 (0.8486, 0.8818) | 0.2947 (0.2638, 0.3240) | n/a |
| candidate_no_context | 0.0375 (0.0201, 0.0550) | 0.2540 (0.1630, 0.3608) | 0.8730 (0.8573, 0.8885) | 0.2764 (0.2355, 0.3225) | n/a |

## Game-facing Outcome Metrics (mean, 95% CI)
| Arm | Quest-state Correctness | Lore Consistency | Contradiction Safety | Objective Completion Support | Gameplay Usefulness | Time-pressure Acceptability |
|---|---:|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2107 (0.1635, 0.2674) | 0.0594 (0.0000, 0.1259) | 1.0000 (1.0000, 1.0000) | 0.1391 (0.0818, 0.1958) | 0.3373 (0.3187, 0.3562) | 0.3350 (0.3011, 0.3718) |
| proposed_contextual_controlled_tuned | 0.2449 (0.1936, 0.2997) | 0.0572 (0.0069, 0.1326) | 1.0000 (1.0000, 1.0000) | 0.1069 (0.0651, 0.1499) | 0.3417 (0.3252, 0.3590) | 0.3244 (0.3035, 0.3476) |
| proposed_contextual | 0.1867 (0.1534, 0.2210) | 0.0455 (0.0130, 0.0831) | 1.0000 (1.0000, 1.0000) | 0.1331 (0.0755, 0.1925) | 0.3260 (0.2984, 0.3531) | 0.3384 (0.3019, 0.3767) |
| candidate_no_context | 0.1346 (0.1179, 0.1527) | 0.0044 (0.0000, 0.0113) | 1.0000 (1.0000, 1.0000) | 0.1098 (0.0545, 0.1699) | 0.3035 (0.2732, 0.3356) | 0.3221 (0.2855, 0.3612) |

- Multi-turn contradiction rate is reported as `1 - contradiction_safety` in row-level outputs.

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0600 | 1.5990 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0206 | -0.0813 |
| proposed_vs_candidate_no_context | naturalness | -0.0078 | -0.0089 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.0521 | 0.3869 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0411 | 9.3484 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | 0.0233 | 0.2125 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 0.0225 | 0.0743 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | 0.0163 | 0.0506 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0758 | 2.0000 |
| proposed_vs_candidate_no_context | context_overlap | 0.0232 | 0.6319 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0258 | -0.1711 |
| proposed_vs_candidate_no_context | persona_style | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | distinct1 | -0.0100 | -0.0106 |
| proposed_vs_candidate_no_context | length_score | -0.0333 | -0.0642 |
| proposed_vs_candidate_no_context | sentence_score | 0.0292 | 0.0320 |
| proposed_vs_candidate_no_context | overall_quality | 0.0184 | 0.0664 |
| controlled_vs_proposed_raw | context_relevance | 0.0284 | 0.2909 |
| controlled_vs_proposed_raw | persona_consistency | 0.0614 | 0.2632 |
| controlled_vs_proposed_raw | naturalness | 0.0049 | 0.0057 |
| controlled_vs_proposed_raw | quest_state_correctness | 0.0240 | 0.1285 |
| controlled_vs_proposed_raw | lore_consistency | 0.0138 | 0.3032 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0060 | 0.0449 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0112 | 0.0344 |
| controlled_vs_proposed_raw | time_pressure_acceptability | -0.0034 | -0.0099 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.0379 | 0.3333 |
| controlled_vs_proposed_raw | context_overlap | 0.0061 | 0.1025 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.0794 | 0.6349 |
| controlled_vs_proposed_raw | persona_style | -0.0104 | -0.0156 |
| controlled_vs_proposed_raw | distinct1 | -0.0159 | -0.0170 |
| controlled_vs_proposed_raw | length_score | 0.0417 | 0.0857 |
| controlled_vs_proposed_raw | sentence_score | 0.0292 | 0.0310 |
| controlled_vs_proposed_raw | overall_quality | 0.0374 | 0.1269 |
| controlled_vs_candidate_no_context | context_relevance | 0.0883 | 2.3550 |
| controlled_vs_candidate_no_context | persona_consistency | 0.0408 | 0.1605 |
| controlled_vs_candidate_no_context | naturalness | -0.0028 | -0.0033 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.0761 | 0.5651 |
| controlled_vs_candidate_no_context | lore_consistency | 0.0549 | 12.4859 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | 0.0293 | 0.2669 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0338 | 0.1112 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 0.0129 | 0.0402 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.1136 | 3.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0293 | 0.7992 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.0536 | 0.3553 |
| controlled_vs_candidate_no_context | persona_style | -0.0104 | -0.0156 |
| controlled_vs_candidate_no_context | distinct1 | -0.0259 | -0.0274 |
| controlled_vs_candidate_no_context | length_score | 0.0083 | 0.0160 |
| controlled_vs_candidate_no_context | sentence_score | 0.0583 | 0.0639 |
| controlled_vs_candidate_no_context | overall_quality | 0.0558 | 0.2018 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0371 | 0.2952 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0701 | 0.2380 |
| controlled_alt_vs_controlled_default | naturalness | 0.0294 | 0.0338 |
| controlled_alt_vs_controlled_default | quest_state_correctness | 0.0343 | 0.1626 |
| controlled_alt_vs_controlled_default | lore_consistency | -0.0022 | -0.0364 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_controlled_default | objective_completion_support | -0.0322 | -0.2317 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | 0.0044 | 0.0131 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | -0.0106 | -0.0316 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0530 | 0.3500 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0001 | 0.0013 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0869 | 0.4252 |
| controlled_alt_vs_controlled_default | persona_style | 0.0031 | 0.0048 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0319 | 0.0347 |
| controlled_alt_vs_controlled_default | length_score | 0.0833 | 0.1579 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0000 | 0.0000 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0498 | 0.1498 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.0655 | 0.6719 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1316 | 0.5638 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0343 | 0.0397 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 0.0582 | 0.3120 |
| controlled_alt_vs_proposed_raw | lore_consistency | 0.0117 | 0.2558 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_proposed_raw | objective_completion_support | -0.0262 | -0.1972 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 0.0156 | 0.0480 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | -0.0140 | -0.0412 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.0909 | 0.8000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0062 | 0.1039 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1663 | 1.3302 |
| controlled_alt_vs_proposed_raw | persona_style | -0.0073 | -0.0109 |
| controlled_alt_vs_proposed_raw | distinct1 | 0.0160 | 0.0171 |
| controlled_alt_vs_proposed_raw | length_score | 0.1250 | 0.2571 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0292 | 0.0310 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.0872 | 0.2957 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.1255 | 3.3453 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1109 | 0.4368 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0266 | 0.0304 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 0.1103 | 0.8196 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 0.0528 | 11.9956 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_candidate_no_context | objective_completion_support | -0.0029 | -0.0266 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 0.0382 | 0.1258 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | 0.0024 | 0.0073 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.1667 | 4.4000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0294 | 0.8015 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1405 | 0.9316 |
| controlled_alt_vs_candidate_no_context | persona_style | -0.0073 | -0.0109 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0060 | 0.0064 |
| controlled_alt_vs_candidate_no_context | length_score | 0.0917 | 0.1765 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.0583 | 0.0639 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1055 | 0.3818 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0600 | (0.0176, 0.1060) | 0.0007 | 0.0600 | (0.0238, 0.0970) | 0.0037 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0206 | (-0.0683, 0.0190) | 0.8583 | -0.0206 | (-0.0364, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | naturalness | -0.0078 | (-0.0200, 0.0028) | 0.8977 | -0.0078 | (-0.0201, 0.0014) | 0.6810 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.0521 | (0.0145, 0.0925) | 0.0003 | 0.0521 | (0.0193, 0.0862) | 0.0027 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0411 | (0.0103, 0.0818) | 0.0003 | 0.0411 | (0.0184, 0.0639) | 0.0033 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | 0.0233 | (-0.0099, 0.0709) | 0.1437 | 0.0233 | (0.0000, 0.0421) | 0.0650 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 0.0225 | (0.0038, 0.0438) | 0.0013 | 0.0225 | (0.0078, 0.0344) | 0.0043 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | 0.0163 | (-0.0075, 0.0477) | 0.1380 | 0.0163 | (0.0000, 0.0280) | 0.0553 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0758 | (0.0227, 0.1364) | 0.0017 | 0.0758 | (0.0303, 0.1240) | 0.0033 |
| proposed_vs_candidate_no_context | context_overlap | 0.0232 | (0.0034, 0.0456) | 0.0070 | 0.0232 | (0.0053, 0.0375) | 0.0053 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0258 | (-0.0873, 0.0238) | 0.8647 | -0.0258 | (-0.0455, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | persona_style | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | distinct1 | -0.0100 | (-0.0274, 0.0051) | 0.8923 | -0.0100 | (-0.0186, 0.0023) | 0.9400 |
| proposed_vs_candidate_no_context | length_score | -0.0333 | (-0.1111, 0.0361) | 0.7963 | -0.0333 | (-0.0714, 0.0200) | 0.8840 |
| proposed_vs_candidate_no_context | sentence_score | 0.0292 | (-0.0583, 0.1167) | 0.3900 | 0.0292 | (0.0000, 0.0875) | 0.3073 |
| proposed_vs_candidate_no_context | overall_quality | 0.0184 | (-0.0101, 0.0474) | 0.1030 | 0.0184 | (0.0005, 0.0362) | 0.0030 |
| controlled_vs_proposed_raw | context_relevance | 0.0284 | (-0.0170, 0.0912) | 0.1713 | 0.0284 | (0.0016, 0.0751) | 0.0063 |
| controlled_vs_proposed_raw | persona_consistency | 0.0614 | (0.0186, 0.1111) | 0.0013 | 0.0614 | (0.0204, 0.1091) | 0.0043 |
| controlled_vs_proposed_raw | naturalness | 0.0049 | (-0.0283, 0.0364) | 0.3780 | 0.0049 | (-0.0220, 0.0439) | 0.4213 |
| controlled_vs_proposed_raw | quest_state_correctness | 0.0240 | (-0.0168, 0.0813) | 0.1793 | 0.0240 | (0.0030, 0.0614) | 0.0043 |
| controlled_vs_proposed_raw | lore_consistency | 0.0138 | (-0.0324, 0.0770) | 0.3253 | 0.0138 | (-0.0240, 0.0786) | 0.3160 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0060 | (-0.0280, 0.0517) | 0.3900 | 0.0060 | (-0.0077, 0.0157) | 0.1933 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0112 | (-0.0122, 0.0333) | 0.1700 | 0.0112 | (-0.0071, 0.0386) | 0.2077 |
| controlled_vs_proposed_raw | time_pressure_acceptability | -0.0034 | (-0.0331, 0.0266) | 0.5927 | -0.0034 | (-0.0089, 0.0044) | 0.8173 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.0379 | (-0.0303, 0.1288) | 0.1937 | 0.0379 | (0.0000, 0.0985) | 0.0703 |
| controlled_vs_proposed_raw | context_overlap | 0.0061 | (-0.0087, 0.0212) | 0.2093 | 0.0061 | (-0.0121, 0.0261) | 0.2653 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.0794 | (0.0258, 0.1429) | 0.0020 | 0.0794 | (0.0286, 0.1374) | 0.0037 |
| controlled_vs_proposed_raw | persona_style | -0.0104 | (-0.0312, 0.0000) | 1.0000 | -0.0104 | (-0.0250, 0.0000) | 1.0000 |
| controlled_vs_proposed_raw | distinct1 | -0.0159 | (-0.0443, 0.0095) | 0.8767 | -0.0159 | (-0.0386, -0.0024) | 1.0000 |
| controlled_vs_proposed_raw | length_score | 0.0417 | (-0.1167, 0.1972) | 0.3180 | 0.0417 | (-0.0978, 0.2528) | 0.3353 |
| controlled_vs_proposed_raw | sentence_score | 0.0292 | (-0.0583, 0.1167) | 0.4207 | 0.0292 | (0.0000, 0.0875) | 0.3173 |
| controlled_vs_proposed_raw | overall_quality | 0.0374 | (0.0130, 0.0710) | 0.0000 | 0.0374 | (0.0102, 0.0758) | 0.0037 |
| controlled_vs_candidate_no_context | context_relevance | 0.0883 | (0.0326, 0.1553) | 0.0000 | 0.0883 | (0.0254, 0.1512) | 0.0037 |
| controlled_vs_candidate_no_context | persona_consistency | 0.0408 | (-0.0259, 0.1111) | 0.1430 | 0.0408 | (-0.0000, 0.1000) | 0.0667 |
| controlled_vs_candidate_no_context | naturalness | -0.0028 | (-0.0440, 0.0321) | 0.5347 | -0.0028 | (-0.0420, 0.0454) | 0.6407 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.0761 | (0.0289, 0.1300) | 0.0003 | 0.0761 | (0.0256, 0.1266) | 0.0033 |
| controlled_vs_candidate_no_context | lore_consistency | 0.0549 | (-0.0025, 0.1163) | 0.0280 | 0.0549 | (0.0000, 0.1463) | 0.0590 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | 0.0293 | (-0.0083, 0.0731) | 0.0767 | 0.0293 | (-0.0077, 0.0575) | 0.0633 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0338 | (0.0089, 0.0607) | 0.0030 | 0.0338 | (0.0057, 0.0725) | 0.0037 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 0.0129 | (-0.0028, 0.0302) | 0.0547 | 0.0129 | (-0.0054, 0.0282) | 0.0777 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.1136 | (0.0379, 0.1970) | 0.0003 | 0.1136 | (0.0303, 0.1970) | 0.0023 |
| controlled_vs_candidate_no_context | context_overlap | 0.0293 | (0.0065, 0.0568) | 0.0027 | 0.0293 | (0.0084, 0.0552) | 0.0020 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.0536 | (-0.0258, 0.1369) | 0.1057 | 0.0536 | (0.0000, 0.1264) | 0.0670 |
| controlled_vs_candidate_no_context | persona_style | -0.0104 | (-0.0312, 0.0000) | 1.0000 | -0.0104 | (-0.0250, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | distinct1 | -0.0259 | (-0.0538, -0.0017) | 0.9823 | -0.0259 | (-0.0399, -0.0111) | 1.0000 |
| controlled_vs_candidate_no_context | length_score | 0.0083 | (-0.1694, 0.1722) | 0.4287 | 0.0083 | (-0.1578, 0.2194) | 0.3667 |
| controlled_vs_candidate_no_context | sentence_score | 0.0583 | (0.0000, 0.1458) | 0.1137 | 0.0583 | (0.0000, 0.1750) | 0.3227 |
| controlled_vs_candidate_no_context | overall_quality | 0.0558 | (0.0158, 0.0990) | 0.0017 | 0.0558 | (0.0106, 0.1036) | 0.0030 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0371 | (-0.0398, 0.1121) | 0.1753 | 0.0371 | (-0.0741, 0.1135) | 0.2183 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0701 | (-0.0178, 0.1710) | 0.0600 | 0.0701 | (0.0364, 0.1215) | 0.0000 |
| controlled_alt_vs_controlled_default | naturalness | 0.0294 | (-0.0181, 0.0806) | 0.1260 | 0.0294 | (0.0082, 0.0435) | 0.0063 |
| controlled_alt_vs_controlled_default | quest_state_correctness | 0.0343 | (-0.0276, 0.0972) | 0.1387 | 0.0343 | (-0.0528, 0.0913) | 0.1450 |
| controlled_alt_vs_controlled_default | lore_consistency | -0.0022 | (-0.0982, 0.1026) | 0.5280 | -0.0022 | (-0.1313, 0.0953) | 0.5333 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | objective_completion_support | -0.0322 | (-0.0955, 0.0275) | 0.8380 | -0.0322 | (-0.0983, 0.0206) | 0.6760 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | 0.0044 | (-0.0179, 0.0236) | 0.3280 | 0.0044 | (-0.0010, 0.0125) | 0.0677 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | -0.0106 | (-0.0462, 0.0205) | 0.7303 | -0.0106 | (-0.0486, 0.0258) | 0.6737 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0530 | (-0.0455, 0.1515) | 0.1440 | 0.0530 | (-0.0826, 0.1515) | 0.1560 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0001 | (-0.0412, 0.0334) | 0.4747 | 0.0001 | (-0.0524, 0.0347) | 0.4287 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0869 | (-0.0198, 0.2111) | 0.0613 | 0.0869 | (0.0442, 0.1519) | 0.0000 |
| controlled_alt_vs_controlled_default | persona_style | 0.0031 | (-0.0219, 0.0312) | 0.4693 | 0.0031 | (0.0000, 0.0075) | 0.3190 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0319 | (0.0059, 0.0576) | 0.0073 | 0.0319 | (0.0137, 0.0465) | 0.0000 |
| controlled_alt_vs_controlled_default | length_score | 0.0833 | (-0.1083, 0.3000) | 0.2237 | 0.0833 | (0.0200, 0.1528) | 0.0027 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0000 | (-0.0875, 0.0875) | 0.6393 | 0.0000 | (-0.1167, 0.0700) | 0.6377 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0498 | (0.0011, 0.0982) | 0.0217 | 0.0498 | (0.0108, 0.0860) | 0.0057 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.0655 | (-0.0014, 0.1371) | 0.0267 | 0.0655 | (0.0024, 0.1170) | 0.0043 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1316 | (0.0509, 0.2330) | 0.0000 | 0.1316 | (0.0846, 0.1973) | 0.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0343 | (-0.0003, 0.0717) | 0.0267 | 0.0343 | (0.0059, 0.0659) | 0.0033 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 0.0582 | (-0.0001, 0.1169) | 0.0253 | 0.0582 | (0.0086, 0.0953) | 0.0060 |
| controlled_alt_vs_proposed_raw | lore_consistency | 0.0117 | (-0.0608, 0.1012) | 0.4130 | 0.0117 | (-0.0527, 0.0694) | 0.3857 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_proposed_raw | objective_completion_support | -0.0262 | (-0.0787, 0.0225) | 0.8403 | -0.0262 | (-0.0851, 0.0220) | 0.6973 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 0.0156 | (-0.0027, 0.0386) | 0.0537 | 0.0156 | (-0.0061, 0.0429) | 0.1203 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | -0.0140 | (-0.0477, 0.0178) | 0.8077 | -0.0140 | (-0.0574, 0.0286) | 0.6677 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.0909 | (0.0076, 0.1894) | 0.0183 | 0.0909 | (0.0165, 0.1576) | 0.0020 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0062 | (-0.0244, 0.0359) | 0.3423 | 0.0062 | (-0.0304, 0.0303) | 0.2703 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1663 | (0.0690, 0.2849) | 0.0003 | 0.1663 | (0.1088, 0.2467) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_style | -0.0073 | (-0.0219, 0.0000) | 1.0000 | -0.0073 | (-0.0175, 0.0000) | 1.0000 |
| controlled_alt_vs_proposed_raw | distinct1 | 0.0160 | (-0.0086, 0.0407) | 0.1007 | 0.0160 | (-0.0086, 0.0400) | 0.1183 |
| controlled_alt_vs_proposed_raw | length_score | 0.1250 | (-0.0500, 0.3056) | 0.0927 | 0.1250 | (-0.0278, 0.3152) | 0.0623 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0292 | (-0.0583, 0.1167) | 0.3967 | 0.0292 | (-0.0778, 0.1000) | 0.3827 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.0872 | (0.0378, 0.1384) | 0.0000 | 0.0872 | (0.0751, 0.0962) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.1255 | (0.0736, 0.1796) | 0.0000 | 0.1255 | (0.0661, 0.1679) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1109 | (0.0252, 0.2193) | 0.0030 | 0.1109 | (0.0492, 0.1973) | 0.0000 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0266 | (-0.0039, 0.0648) | 0.0507 | 0.0266 | (-0.0065, 0.0673) | 0.0607 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 0.1103 | (0.0654, 0.1563) | 0.0000 | 0.1103 | (0.0594, 0.1504) | 0.0000 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 0.0528 | (0.0048, 0.1263) | 0.0057 | 0.0528 | (0.0000, 0.0973) | 0.0627 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_candidate_no_context | objective_completion_support | -0.0029 | (-0.0507, 0.0417) | 0.5390 | -0.0029 | (-0.0449, 0.0387) | 0.5403 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 0.0382 | (0.0133, 0.0640) | 0.0003 | 0.0382 | (0.0127, 0.0795) | 0.0000 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | 0.0024 | (-0.0318, 0.0355) | 0.4453 | 0.0024 | (-0.0346, 0.0490) | 0.3617 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.1667 | (0.0985, 0.2424) | 0.0000 | 0.1667 | (0.0909, 0.2208) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0294 | (0.0103, 0.0484) | 0.0017 | 0.0294 | (0.0029, 0.0483) | 0.0197 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1405 | (0.0317, 0.2691) | 0.0043 | 0.1405 | (0.0646, 0.2467) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_style | -0.0073 | (-0.0219, 0.0000) | 1.0000 | -0.0073 | (-0.0175, 0.0000) | 1.0000 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0060 | (-0.0196, 0.0324) | 0.3280 | 0.0060 | (-0.0135, 0.0255) | 0.3167 |
| controlled_alt_vs_candidate_no_context | length_score | 0.0917 | (-0.0556, 0.2556) | 0.1237 | 0.0917 | (-0.0833, 0.2697) | 0.1463 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.0583 | (-0.0583, 0.1750) | 0.2140 | 0.0583 | (-0.0700, 0.1750) | 0.2580 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1055 | (0.0604, 0.1585) | 0.0000 | 0.1055 | (0.0967, 0.1144) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 6 | 0 | 6 | 0.7500 | 1.0000 |
| proposed_vs_candidate_no_context | persona_consistency | 1 | 2 | 9 | 0.4583 | 0.3333 |
| proposed_vs_candidate_no_context | naturalness | 2 | 3 | 7 | 0.4583 | 0.4000 |
| proposed_vs_candidate_no_context | quest_state_correctness | 6 | 0 | 6 | 0.7500 | 1.0000 |
| proposed_vs_candidate_no_context | lore_consistency | 5 | 0 | 7 | 0.7083 | 1.0000 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 12 | 0.5000 | nan |
| proposed_vs_candidate_no_context | objective_completion_support | 2 | 1 | 9 | 0.5417 | 0.6667 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 5 | 1 | 6 | 0.6667 | 0.8333 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | 2 | 2 | 8 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 5 | 0 | 7 | 0.7083 | 1.0000 |
| proposed_vs_candidate_no_context | context_overlap | 5 | 1 | 6 | 0.6667 | 0.8333 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 1 | 2 | 9 | 0.4583 | 0.3333 |
| proposed_vs_candidate_no_context | persona_style | 0 | 0 | 12 | 0.5000 | nan |
| proposed_vs_candidate_no_context | distinct1 | 1 | 3 | 8 | 0.4167 | 0.2500 |
| proposed_vs_candidate_no_context | length_score | 2 | 3 | 7 | 0.4583 | 0.4000 |
| proposed_vs_candidate_no_context | sentence_score | 2 | 1 | 9 | 0.5417 | 0.6667 |
| proposed_vs_candidate_no_context | overall_quality | 5 | 1 | 6 | 0.6667 | 0.8333 |
| controlled_vs_proposed_raw | context_relevance | 6 | 4 | 2 | 0.5833 | 0.6000 |
| controlled_vs_proposed_raw | persona_consistency | 5 | 0 | 7 | 0.7083 | 1.0000 |
| controlled_vs_proposed_raw | naturalness | 6 | 4 | 2 | 0.5833 | 0.6000 |
| controlled_vs_proposed_raw | quest_state_correctness | 6 | 4 | 2 | 0.5833 | 0.6000 |
| controlled_vs_proposed_raw | lore_consistency | 3 | 3 | 6 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0 | 0 | 12 | 0.5000 | nan |
| controlled_vs_proposed_raw | objective_completion_support | 2 | 5 | 5 | 0.3750 | 0.2857 |
| controlled_vs_proposed_raw | gameplay_usefulness | 7 | 3 | 2 | 0.6667 | 0.7000 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 3 | 5 | 4 | 0.4167 | 0.3750 |
| controlled_vs_proposed_raw | context_keyword_coverage | 4 | 3 | 5 | 0.5417 | 0.5714 |
| controlled_vs_proposed_raw | context_overlap | 5 | 5 | 2 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 5 | 0 | 7 | 0.7083 | 1.0000 |
| controlled_vs_proposed_raw | persona_style | 0 | 1 | 11 | 0.4583 | 0.0000 |
| controlled_vs_proposed_raw | distinct1 | 6 | 4 | 2 | 0.5833 | 0.6000 |
| controlled_vs_proposed_raw | length_score | 6 | 4 | 2 | 0.5833 | 0.6000 |
| controlled_vs_proposed_raw | sentence_score | 2 | 1 | 9 | 0.5417 | 0.6667 |
| controlled_vs_proposed_raw | overall_quality | 8 | 2 | 2 | 0.7500 | 0.8000 |
| controlled_vs_candidate_no_context | context_relevance | 8 | 2 | 2 | 0.7500 | 0.8000 |
| controlled_vs_candidate_no_context | persona_consistency | 4 | 2 | 6 | 0.5833 | 0.6667 |
| controlled_vs_candidate_no_context | naturalness | 5 | 5 | 2 | 0.5000 | 0.5000 |
| controlled_vs_candidate_no_context | quest_state_correctness | 8 | 2 | 2 | 0.7500 | 0.8000 |
| controlled_vs_candidate_no_context | lore_consistency | 3 | 2 | 7 | 0.5417 | 0.6000 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 12 | 0.5000 | nan |
| controlled_vs_candidate_no_context | objective_completion_support | 3 | 3 | 6 | 0.5000 | 0.5000 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 8 | 2 | 2 | 0.7500 | 0.8000 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 5 | 3 | 4 | 0.5833 | 0.6250 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 6 | 0 | 6 | 0.7500 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 8 | 2 | 2 | 0.7500 | 0.8000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 4 | 2 | 6 | 0.5833 | 0.6667 |
| controlled_vs_candidate_no_context | persona_style | 0 | 1 | 11 | 0.4583 | 0.0000 |
| controlled_vs_candidate_no_context | distinct1 | 4 | 6 | 2 | 0.4167 | 0.4000 |
| controlled_vs_candidate_no_context | length_score | 6 | 4 | 2 | 0.5833 | 0.6000 |
| controlled_vs_candidate_no_context | sentence_score | 2 | 0 | 10 | 0.5833 | 1.0000 |
| controlled_vs_candidate_no_context | overall_quality | 7 | 3 | 2 | 0.6667 | 0.7000 |
| controlled_alt_vs_controlled_default | context_relevance | 6 | 3 | 3 | 0.6250 | 0.6667 |
| controlled_alt_vs_controlled_default | persona_consistency | 5 | 2 | 5 | 0.6250 | 0.7143 |
| controlled_alt_vs_controlled_default | naturalness | 5 | 4 | 3 | 0.5417 | 0.5556 |
| controlled_alt_vs_controlled_default | quest_state_correctness | 6 | 3 | 3 | 0.6250 | 0.6667 |
| controlled_alt_vs_controlled_default | lore_consistency | 4 | 3 | 5 | 0.5417 | 0.5714 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0 | 0 | 12 | 0.5000 | nan |
| controlled_alt_vs_controlled_default | objective_completion_support | 4 | 5 | 3 | 0.4583 | 0.4444 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | 6 | 3 | 3 | 0.6250 | 0.6667 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | 4 | 5 | 3 | 0.4583 | 0.4444 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 5 | 3 | 4 | 0.5833 | 0.6250 |
| controlled_alt_vs_controlled_default | context_overlap | 5 | 4 | 3 | 0.5417 | 0.5556 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 4 | 2 | 6 | 0.5833 | 0.6667 |
| controlled_alt_vs_controlled_default | persona_style | 1 | 1 | 10 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | distinct1 | 7 | 2 | 3 | 0.7083 | 0.7778 |
| controlled_alt_vs_controlled_default | length_score | 4 | 4 | 4 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | sentence_score | 1 | 1 | 10 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | overall_quality | 7 | 2 | 3 | 0.7083 | 0.7778 |
| controlled_alt_vs_proposed_raw | context_relevance | 7 | 4 | 1 | 0.6250 | 0.6364 |
| controlled_alt_vs_proposed_raw | persona_consistency | 7 | 0 | 5 | 0.7917 | 1.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 6 | 5 | 1 | 0.5417 | 0.5455 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 7 | 4 | 1 | 0.6250 | 0.6364 |
| controlled_alt_vs_proposed_raw | lore_consistency | 3 | 4 | 5 | 0.4583 | 0.4286 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0 | 0 | 12 | 0.5000 | nan |
| controlled_alt_vs_proposed_raw | objective_completion_support | 3 | 6 | 3 | 0.3750 | 0.3333 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 6 | 5 | 1 | 0.5417 | 0.5455 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | 5 | 5 | 2 | 0.5000 | 0.5000 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 6 | 1 | 5 | 0.7083 | 0.8571 |
| controlled_alt_vs_proposed_raw | context_overlap | 6 | 5 | 1 | 0.5417 | 0.5455 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 7 | 0 | 5 | 0.7917 | 1.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 0 | 1 | 11 | 0.4583 | 0.0000 |
| controlled_alt_vs_proposed_raw | distinct1 | 7 | 3 | 2 | 0.6667 | 0.7000 |
| controlled_alt_vs_proposed_raw | length_score | 6 | 4 | 2 | 0.5833 | 0.6000 |
| controlled_alt_vs_proposed_raw | sentence_score | 2 | 1 | 9 | 0.5417 | 0.6667 |
| controlled_alt_vs_proposed_raw | overall_quality | 9 | 2 | 1 | 0.7917 | 0.8182 |
| controlled_alt_vs_candidate_no_context | context_relevance | 10 | 1 | 1 | 0.8750 | 0.9091 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 6 | 1 | 5 | 0.7083 | 0.8571 |
| controlled_alt_vs_candidate_no_context | naturalness | 6 | 5 | 1 | 0.5417 | 0.5455 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 10 | 1 | 1 | 0.8750 | 0.9091 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 4 | 1 | 7 | 0.6250 | 0.8000 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 12 | 0.5000 | nan |
| controlled_alt_vs_candidate_no_context | objective_completion_support | 5 | 4 | 3 | 0.5417 | 0.5556 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 8 | 3 | 1 | 0.7083 | 0.7273 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | 6 | 4 | 2 | 0.5833 | 0.6000 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 9 | 0 | 3 | 0.8750 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 8 | 3 | 1 | 0.7083 | 0.7273 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 6 | 1 | 5 | 0.7083 | 0.8571 |
| controlled_alt_vs_candidate_no_context | persona_style | 0 | 1 | 11 | 0.4583 | 0.0000 |
| controlled_alt_vs_candidate_no_context | distinct1 | 6 | 4 | 2 | 0.5833 | 0.6000 |
| controlled_alt_vs_candidate_no_context | length_score | 6 | 5 | 1 | 0.5417 | 0.5455 |
| controlled_alt_vs_candidate_no_context | sentence_score | 3 | 1 | 8 | 0.5833 | 0.7500 |
| controlled_alt_vs_candidate_no_context | overall_quality | 10 | 1 | 1 | 0.8750 | 0.9091 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.1667 | 0.1667 | 0.8333 |
| proposed_contextual_controlled_tuned | 0.0000 | 0.0000 | 0.0833 | 0.0833 | 0.9167 |
| proposed_contextual | 0.0000 | 0.0000 | 0.5833 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `4`
- Unique template signatures: `11`
- Template signature ratio: `0.9167`
- Effective sample size by source clustering: `3.79`
- Effective sample size by template-signature clustering: `10.29`
- Detailed diagnostics are published in `scenario_dependence.json`.

## Multi-turn Contradiction
| Arm | Contradiction Rate | Contradiction Safety | Contradicted Sources | Source Count |
|---|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 1.0000 | 0 | 4 |
| proposed_contextual_controlled_tuned | 0.0000 | 1.0000 | 0 | 4 |
| proposed_contextual | 0.0000 | 1.0000 | 0 | 4 |
| candidate_no_context | 0.0000 | 1.0000 | 0 | 4 |
- Detailed source-level values are published in `multi_turn_contradictions.json`.

- BERTScore status: unavailable (disabled_by_flag).

This report covers proposal RO5 metrics plus game-facing outcomes: quest-state correctness, lore consistency, contradiction safety, objective completion support, gameplay usefulness, and time-pressure acceptability.
# Proposal Alignment Evaluation Report

- Run ID: `20260313T054007Z`
- Generated: `2026-03-13T05:44:40.187633+00:00`
- Scenarios: `artifacts\proposal_control_tuning\auto_tune\20260313T051946Z\train_runs\trial_002\seed_19\20260313T054007Z\scenarios.jsonl`
- Scenario count: `24`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_tuned`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.1062 (0.0538, 0.1644) | 0.2846 (0.2148, 0.3559) | 0.8739 (0.8569, 0.8907) | 0.3222 (0.2759, 0.3719) | n/a |
| proposed_contextual_controlled_tuned | 0.0764 (0.0377, 0.1224) | 0.2414 (0.1916, 0.2940) | 0.8755 (0.8587, 0.8924) | 0.2922 (0.2637, 0.3234) | n/a |
| proposed_contextual | 0.0911 (0.0464, 0.1401) | 0.1865 (0.1350, 0.2452) | 0.8626 (0.8481, 0.8769) | 0.2761 (0.2439, 0.3129) | n/a |
| candidate_no_context | 0.0287 (0.0156, 0.0446) | 0.2419 (0.1821, 0.3102) | 0.8813 (0.8641, 0.8999) | 0.2712 (0.2456, 0.3013) | n/a |

## Game-facing Outcome Metrics (mean, 95% CI)
| Arm | Quest-state Correctness | Lore Consistency | Contradiction Safety | Objective Completion Support | Gameplay Usefulness | Time-pressure Acceptability |
|---|---:|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.1953 (0.1508, 0.2495) | 0.0334 (0.0093, 0.0603) | 0.8333 (0.6667, 0.9583) | 0.0776 (0.0538, 0.1004) | 0.3068 (0.2883, 0.3274) | 0.3007 (0.2832, 0.3169) |
| proposed_contextual_controlled_tuned | 0.1645 (0.1310, 0.2040) | 0.0287 (0.0093, 0.0535) | 1.0000 (1.0000, 1.0000) | 0.0662 (0.0408, 0.0906) | 0.2948 (0.2848, 0.3052) | 0.2905 (0.2728, 0.3061) |
| proposed_contextual | 0.1760 (0.1393, 0.2178) | 0.0396 (0.0101, 0.0801) | 1.0000 (1.0000, 1.0000) | 0.0663 (0.0403, 0.0917) | 0.2927 (0.2763, 0.3080) | 0.2921 (0.2724, 0.3119) |
| candidate_no_context | 0.1232 (0.1113, 0.1395) | 0.0034 (0.0000, 0.0092) | 1.0000 (1.0000, 1.0000) | 0.0712 (0.0458, 0.0961) | 0.2876 (0.2767, 0.2971) | 0.2950 (0.2794, 0.3114) |

- Multi-turn contradiction rate is reported as `1 - contradiction_safety` in row-level outputs.

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0623 | 2.1711 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0555 | -0.2292 |
| proposed_vs_candidate_no_context | naturalness | -0.0187 | -0.0213 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.0528 | 0.4289 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0362 | 10.7687 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | -0.0049 | -0.0693 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 0.0051 | 0.0176 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | -0.0029 | -0.0100 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0814 | 3.0714 |
| proposed_vs_candidate_no_context | context_overlap | 0.0178 | 0.5254 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0615 | -0.3713 |
| proposed_vs_candidate_no_context | persona_style | -0.0312 | -0.0571 |
| proposed_vs_candidate_no_context | distinct1 | 0.0063 | 0.0068 |
| proposed_vs_candidate_no_context | length_score | -0.0917 | -0.1590 |
| proposed_vs_candidate_no_context | sentence_score | -0.0292 | -0.0300 |
| proposed_vs_candidate_no_context | overall_quality | 0.0049 | 0.0180 |
| controlled_vs_proposed_raw | context_relevance | 0.0151 | 0.1658 |
| controlled_vs_proposed_raw | persona_consistency | 0.0982 | 0.5264 |
| controlled_vs_proposed_raw | naturalness | 0.0113 | 0.0131 |
| controlled_vs_proposed_raw | quest_state_correctness | 0.0193 | 0.1097 |
| controlled_vs_proposed_raw | lore_consistency | -0.0062 | -0.1565 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | -0.1667 | -0.1667 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0114 | 0.1719 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0141 | 0.0481 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 0.0086 | 0.0295 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.0197 | 0.1825 |
| controlled_vs_proposed_raw | context_overlap | 0.0044 | 0.0845 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1230 | 1.1810 |
| controlled_vs_proposed_raw | persona_style | -0.0013 | -0.0025 |
| controlled_vs_proposed_raw | distinct1 | -0.0105 | -0.0113 |
| controlled_vs_proposed_raw | length_score | 0.0556 | 0.1146 |
| controlled_vs_proposed_raw | sentence_score | 0.0437 | 0.0465 |
| controlled_vs_proposed_raw | overall_quality | 0.0461 | 0.1669 |
| controlled_vs_candidate_no_context | context_relevance | 0.0774 | 2.6969 |
| controlled_vs_candidate_no_context | persona_consistency | 0.0427 | 0.1765 |
| controlled_vs_candidate_no_context | naturalness | -0.0074 | -0.0084 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.0721 | 0.5856 |
| controlled_vs_candidate_no_context | lore_consistency | 0.0300 | 8.9271 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | -0.1667 | -0.1667 |
| controlled_vs_candidate_no_context | objective_completion_support | 0.0065 | 0.0907 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0192 | 0.0666 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 0.0057 | 0.0193 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.1011 | 3.8143 |
| controlled_vs_candidate_no_context | context_overlap | 0.0221 | 0.6543 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.0615 | 0.3713 |
| controlled_vs_candidate_no_context | persona_style | -0.0326 | -0.0595 |
| controlled_vs_candidate_no_context | distinct1 | -0.0042 | -0.0045 |
| controlled_vs_candidate_no_context | length_score | -0.0361 | -0.0627 |
| controlled_vs_candidate_no_context | sentence_score | 0.0146 | 0.0150 |
| controlled_vs_candidate_no_context | overall_quality | 0.0510 | 0.1880 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0298 | -0.2805 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0432 | -0.1517 |
| controlled_alt_vs_controlled_default | naturalness | 0.0016 | 0.0018 |
| controlled_alt_vs_controlled_default | quest_state_correctness | -0.0309 | -0.1581 |
| controlled_alt_vs_controlled_default | lore_consistency | -0.0047 | -0.1404 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0.1667 | 0.2000 |
| controlled_alt_vs_controlled_default | objective_completion_support | -0.0115 | -0.1476 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | -0.0120 | -0.0391 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | -0.0102 | -0.0339 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0402 | -0.3145 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0056 | -0.0996 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0575 | -0.2533 |
| controlled_alt_vs_controlled_default | persona_style | 0.0143 | 0.0278 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0130 | 0.0141 |
| controlled_alt_vs_controlled_default | length_score | 0.0111 | 0.0206 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0583 | -0.0592 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0300 | -0.0932 |
| controlled_alt_vs_proposed_raw | context_relevance | -0.0147 | -0.1612 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.0550 | 0.2949 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0129 | 0.0149 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | -0.0116 | -0.0658 |
| controlled_alt_vs_proposed_raw | lore_consistency | -0.0109 | -0.2749 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_proposed_raw | objective_completion_support | -0.0001 | -0.0010 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 0.0021 | 0.0071 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | -0.0016 | -0.0054 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | -0.0205 | -0.1895 |
| controlled_alt_vs_proposed_raw | context_overlap | -0.0012 | -0.0235 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.0655 | 0.6286 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0130 | 0.0253 |
| controlled_alt_vs_proposed_raw | distinct1 | 0.0025 | 0.0027 |
| controlled_alt_vs_proposed_raw | length_score | 0.0667 | 0.1375 |
| controlled_alt_vs_proposed_raw | sentence_score | -0.0146 | -0.0155 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.0161 | 0.0582 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.0477 | 1.6598 |
| controlled_alt_vs_candidate_no_context | persona_consistency | -0.0005 | -0.0019 |
| controlled_alt_vs_candidate_no_context | naturalness | -0.0059 | -0.0066 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 0.0413 | 0.3350 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 0.0253 | 7.5330 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_candidate_no_context | objective_completion_support | -0.0050 | -0.0702 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 0.0072 | 0.0249 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | -0.0045 | -0.0153 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.0610 | 2.3000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0166 | 0.4895 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.0040 | 0.0240 |
| controlled_alt_vs_candidate_no_context | persona_style | -0.0182 | -0.0333 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0088 | 0.0095 |
| controlled_alt_vs_candidate_no_context | length_score | -0.0250 | -0.0434 |
| controlled_alt_vs_candidate_no_context | sentence_score | -0.0437 | -0.0451 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.0210 | 0.0773 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0623 | (0.0178, 0.1163) | 0.0013 | 0.0623 | (0.0252, 0.1020) | 0.0010 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0555 | (-0.1228, 0.0064) | 0.9597 | -0.0555 | (-0.1288, 0.0007) | 0.9610 |
| proposed_vs_candidate_no_context | naturalness | -0.0187 | (-0.0409, 0.0027) | 0.9547 | -0.0187 | (-0.0459, 0.0113) | 0.9130 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.0528 | (0.0122, 0.0994) | 0.0020 | 0.0528 | (0.0202, 0.0891) | 0.0003 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0362 | (0.0052, 0.0754) | 0.0043 | 0.0362 | (0.0071, 0.0653) | 0.0117 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | -0.0049 | (-0.0263, 0.0171) | 0.6730 | -0.0049 | (-0.0211, 0.0162) | 0.7083 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 0.0051 | (-0.0122, 0.0212) | 0.2700 | 0.0051 | (-0.0013, 0.0135) | 0.0793 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | -0.0029 | (-0.0179, 0.0123) | 0.6617 | -0.0029 | (-0.0146, 0.0082) | 0.7087 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0814 | (0.0227, 0.1515) | 0.0010 | 0.0814 | (0.0325, 0.1364) | 0.0000 |
| proposed_vs_candidate_no_context | context_overlap | 0.0178 | (0.0017, 0.0351) | 0.0130 | 0.0178 | (0.0063, 0.0330) | 0.0000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0615 | (-0.1458, 0.0169) | 0.9313 | -0.0615 | (-0.1786, 0.0069) | 0.9640 |
| proposed_vs_candidate_no_context | persona_style | -0.0312 | (-0.1094, 0.0339) | 0.7950 | -0.0312 | (-0.1339, 0.0682) | 0.7247 |
| proposed_vs_candidate_no_context | distinct1 | 0.0063 | (-0.0174, 0.0296) | 0.2997 | 0.0063 | (-0.0281, 0.0396) | 0.3677 |
| proposed_vs_candidate_no_context | length_score | -0.0917 | (-0.1764, -0.0083) | 0.9847 | -0.0917 | (-0.1742, -0.0125) | 0.9887 |
| proposed_vs_candidate_no_context | sentence_score | -0.0292 | (-0.0729, 0.0000) | 1.0000 | -0.0292 | (-0.0636, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | overall_quality | 0.0049 | (-0.0303, 0.0411) | 0.3923 | 0.0049 | (-0.0268, 0.0321) | 0.3510 |
| controlled_vs_proposed_raw | context_relevance | 0.0151 | (-0.0563, 0.0896) | 0.3543 | 0.0151 | (-0.0164, 0.0586) | 0.1897 |
| controlled_vs_proposed_raw | persona_consistency | 0.0982 | (0.0229, 0.1823) | 0.0027 | 0.0982 | (0.0198, 0.2079) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0113 | (-0.0094, 0.0343) | 0.1457 | 0.0113 | (-0.0040, 0.0291) | 0.0713 |
| controlled_vs_proposed_raw | quest_state_correctness | 0.0193 | (-0.0429, 0.0845) | 0.2863 | 0.0193 | (-0.0052, 0.0592) | 0.1070 |
| controlled_vs_proposed_raw | lore_consistency | -0.0062 | (-0.0507, 0.0352) | 0.5843 | -0.0062 | (-0.0322, 0.0169) | 0.6903 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | -0.1667 | (-0.3333, -0.0417) | 1.0000 | -0.1667 | (-0.5455, 0.0000) | 1.0000 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0114 | (-0.0088, 0.0304) | 0.1387 | 0.0114 | (-0.0074, 0.0345) | 0.1197 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0141 | (-0.0107, 0.0430) | 0.1513 | 0.0141 | (0.0016, 0.0315) | 0.0123 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 0.0086 | (-0.0096, 0.0243) | 0.1613 | 0.0086 | (-0.0107, 0.0265) | 0.1673 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.0197 | (-0.0758, 0.1189) | 0.3560 | 0.0197 | (-0.0197, 0.0764) | 0.2200 |
| controlled_vs_proposed_raw | context_overlap | 0.0044 | (-0.0178, 0.0278) | 0.3537 | 0.0044 | (-0.0094, 0.0154) | 0.2417 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1230 | (0.0308, 0.2232) | 0.0023 | 0.1230 | (0.0119, 0.2786) | 0.0107 |
| controlled_vs_proposed_raw | persona_style | -0.0013 | (-0.0742, 0.0951) | 0.5480 | -0.0013 | (-0.1023, 0.0770) | 0.6333 |
| controlled_vs_proposed_raw | distinct1 | -0.0105 | (-0.0297, 0.0060) | 0.8797 | -0.0105 | (-0.0422, 0.0105) | 0.7727 |
| controlled_vs_proposed_raw | length_score | 0.0556 | (-0.0319, 0.1569) | 0.1170 | 0.0556 | (-0.0167, 0.1242) | 0.0623 |
| controlled_vs_proposed_raw | sentence_score | 0.0437 | (0.0000, 0.0875) | 0.0353 | 0.0437 | (0.0000, 0.1050) | 0.0773 |
| controlled_vs_proposed_raw | overall_quality | 0.0461 | (-0.0082, 0.1053) | 0.0507 | 0.0461 | (0.0124, 0.0933) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.0774 | (0.0242, 0.1387) | 0.0003 | 0.0774 | (0.0303, 0.1367) | 0.0007 |
| controlled_vs_candidate_no_context | persona_consistency | 0.0427 | (-0.0118, 0.1128) | 0.0757 | 0.0427 | (-0.0148, 0.1275) | 0.1090 |
| controlled_vs_candidate_no_context | naturalness | -0.0074 | (-0.0273, 0.0113) | 0.7647 | -0.0074 | (-0.0323, 0.0135) | 0.7297 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.0721 | (0.0252, 0.1252) | 0.0003 | 0.0721 | (0.0281, 0.1288) | 0.0003 |
| controlled_vs_candidate_no_context | lore_consistency | 0.0300 | (0.0046, 0.0607) | 0.0097 | 0.0300 | (0.0019, 0.0563) | 0.0200 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | -0.1667 | (-0.3333, -0.0417) | 1.0000 | -0.1667 | (-0.5455, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | 0.0065 | (-0.0168, 0.0286) | 0.2817 | 0.0065 | (-0.0041, 0.0209) | 0.1470 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0192 | (-0.0017, 0.0428) | 0.0380 | 0.0192 | (0.0066, 0.0360) | 0.0020 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 0.0057 | (-0.0116, 0.0234) | 0.2740 | 0.0057 | (-0.0018, 0.0135) | 0.0693 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.1011 | (0.0318, 0.1803) | 0.0003 | 0.1011 | (0.0399, 0.1806) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0221 | (0.0063, 0.0404) | 0.0030 | 0.0221 | (0.0093, 0.0358) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.0615 | (-0.0070, 0.1458) | 0.0523 | 0.0615 | (-0.0068, 0.1591) | 0.0597 |
| controlled_vs_candidate_no_context | persona_style | -0.0326 | (-0.0768, 0.0130) | 0.9227 | -0.0326 | (-0.0709, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | distinct1 | -0.0042 | (-0.0185, 0.0102) | 0.7267 | -0.0042 | (-0.0188, 0.0058) | 0.7253 |
| controlled_vs_candidate_no_context | length_score | -0.0361 | (-0.1125, 0.0347) | 0.8433 | -0.0361 | (-0.1306, 0.0569) | 0.7623 |
| controlled_vs_candidate_no_context | sentence_score | 0.0146 | (0.0000, 0.0437) | 0.3693 | 0.0146 | (0.0000, 0.0477) | 0.3270 |
| controlled_vs_candidate_no_context | overall_quality | 0.0510 | (0.0162, 0.0892) | 0.0010 | 0.0510 | (0.0167, 0.0986) | 0.0007 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0298 | (-0.0808, 0.0295) | 0.8493 | -0.0298 | (-0.0639, -0.0034) | 0.9893 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0432 | (-0.1152, 0.0096) | 0.9177 | -0.0432 | (-0.1121, 0.0061) | 0.9500 |
| controlled_alt_vs_controlled_default | naturalness | 0.0016 | (-0.0190, 0.0225) | 0.4567 | 0.0016 | (-0.0096, 0.0163) | 0.4410 |
| controlled_alt_vs_controlled_default | quest_state_correctness | -0.0309 | (-0.0810, 0.0213) | 0.8953 | -0.0309 | (-0.0637, -0.0040) | 0.9920 |
| controlled_alt_vs_controlled_default | lore_consistency | -0.0047 | (-0.0351, 0.0260) | 0.6190 | -0.0047 | (-0.0156, 0.0088) | 0.7617 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0.1667 | (0.0417, 0.3333) | 0.0133 | 0.1667 | (0.0000, 0.5455) | 0.3247 |
| controlled_alt_vs_controlled_default | objective_completion_support | -0.0115 | (-0.0330, 0.0086) | 0.8697 | -0.0115 | (-0.0255, -0.0001) | 0.9770 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | -0.0120 | (-0.0293, 0.0029) | 0.9400 | -0.0120 | (-0.0227, -0.0036) | 1.0000 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | -0.0102 | (-0.0283, 0.0087) | 0.8643 | -0.0102 | (-0.0259, 0.0030) | 0.9390 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0402 | (-0.1099, 0.0341) | 0.8650 | -0.0402 | (-0.0836, -0.0049) | 1.0000 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0056 | (-0.0225, 0.0111) | 0.7337 | -0.0056 | (-0.0144, 0.0001) | 0.9713 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0575 | (-0.1488, 0.0149) | 0.9167 | -0.0575 | (-0.1440, 0.0060) | 0.9693 |
| controlled_alt_vs_controlled_default | persona_style | 0.0143 | (-0.0260, 0.0547) | 0.2750 | 0.0143 | (0.0000, 0.0375) | 0.0857 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0130 | (-0.0011, 0.0289) | 0.0377 | 0.0130 | (-0.0022, 0.0351) | 0.0647 |
| controlled_alt_vs_controlled_default | length_score | 0.0111 | (-0.0708, 0.0944) | 0.3887 | 0.0111 | (-0.0455, 0.0682) | 0.3020 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0583 | (-0.1167, -0.0146) | 1.0000 | -0.0583 | (-0.1225, -0.0135) | 1.0000 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0300 | (-0.0692, 0.0060) | 0.9483 | -0.0300 | (-0.0634, -0.0062) | 0.9910 |
| controlled_alt_vs_proposed_raw | context_relevance | -0.0147 | (-0.0642, 0.0302) | 0.7330 | -0.0147 | (-0.0366, 0.0020) | 0.9447 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.0550 | (-0.0015, 0.1141) | 0.0300 | 0.0550 | (0.0216, 0.1024) | 0.0003 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0129 | (-0.0084, 0.0334) | 0.1180 | 0.0129 | (-0.0070, 0.0321) | 0.0713 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | -0.0116 | (-0.0543, 0.0292) | 0.7120 | -0.0116 | (-0.0354, 0.0067) | 0.8883 |
| controlled_alt_vs_proposed_raw | lore_consistency | -0.0109 | (-0.0368, 0.0107) | 0.8093 | -0.0109 | (-0.0284, 0.0097) | 0.8747 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_proposed_raw | objective_completion_support | -0.0001 | (-0.0199, 0.0197) | 0.4990 | -0.0001 | (-0.0090, 0.0100) | 0.5667 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 0.0021 | (-0.0104, 0.0155) | 0.3987 | 0.0021 | (-0.0055, 0.0104) | 0.2703 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | -0.0016 | (-0.0166, 0.0128) | 0.5810 | -0.0016 | (-0.0071, 0.0048) | 0.7083 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | -0.0205 | (-0.0848, 0.0379) | 0.7460 | -0.0205 | (-0.0509, 0.0041) | 0.9573 |
| controlled_alt_vs_proposed_raw | context_overlap | -0.0012 | (-0.0203, 0.0171) | 0.5507 | -0.0012 | (-0.0124, 0.0064) | 0.5803 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.0655 | (0.0050, 0.1300) | 0.0193 | 0.0655 | (0.0179, 0.1321) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0130 | (-0.0508, 0.0964) | 0.4020 | 0.0130 | (-0.0682, 0.0871) | 0.4203 |
| controlled_alt_vs_proposed_raw | distinct1 | 0.0025 | (-0.0155, 0.0200) | 0.3930 | 0.0025 | (-0.0122, 0.0246) | 0.3897 |
| controlled_alt_vs_proposed_raw | length_score | 0.0667 | (-0.0264, 0.1694) | 0.0957 | 0.0667 | (0.0097, 0.1470) | 0.0007 |
| controlled_alt_vs_proposed_raw | sentence_score | -0.0146 | (-0.0729, 0.0437) | 0.7563 | -0.0146 | (-0.0875, 0.0477) | 0.7190 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.0161 | (-0.0183, 0.0493) | 0.1817 | 0.0161 | (0.0025, 0.0341) | 0.0083 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.0477 | (0.0115, 0.0921) | 0.0030 | 0.0477 | (0.0197, 0.0745) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | -0.0005 | (-0.0526, 0.0418) | 0.4787 | -0.0005 | (-0.0542, 0.0485) | 0.5293 |
| controlled_alt_vs_candidate_no_context | naturalness | -0.0059 | (-0.0293, 0.0147) | 0.6900 | -0.0059 | (-0.0176, 0.0066) | 0.8180 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 0.0413 | (0.0098, 0.0772) | 0.0013 | 0.0413 | (0.0163, 0.0656) | 0.0000 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 0.0253 | (0.0050, 0.0494) | 0.0063 | 0.0253 | (0.0082, 0.0424) | 0.0007 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_candidate_no_context | objective_completion_support | -0.0050 | (-0.0303, 0.0197) | 0.6580 | -0.0050 | (-0.0192, 0.0089) | 0.7350 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 0.0072 | (-0.0048, 0.0197) | 0.1297 | 0.0072 | (0.0013, 0.0153) | 0.0007 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | -0.0045 | (-0.0203, 0.0104) | 0.7153 | -0.0045 | (-0.0147, 0.0031) | 0.8513 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.0610 | (0.0117, 0.1216) | 0.0030 | 0.0610 | (0.0263, 0.0955) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0166 | (0.0027, 0.0308) | 0.0073 | 0.0166 | (0.0071, 0.0249) | 0.0003 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.0040 | (-0.0645, 0.0556) | 0.4220 | 0.0040 | (-0.0628, 0.0606) | 0.4017 |
| controlled_alt_vs_candidate_no_context | persona_style | -0.0182 | (-0.0612, 0.0208) | 0.8330 | -0.0182 | (-0.0505, 0.0000) | 1.0000 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0088 | (-0.0094, 0.0283) | 0.1807 | 0.0088 | (-0.0038, 0.0318) | 0.2527 |
| controlled_alt_vs_candidate_no_context | length_score | -0.0250 | (-0.1111, 0.0583) | 0.7227 | -0.0250 | (-0.0750, 0.0242) | 0.8490 |
| controlled_alt_vs_candidate_no_context | sentence_score | -0.0437 | (-0.1021, 0.0000) | 1.0000 | -0.0437 | (-0.1050, 0.0000) | 1.0000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.0210 | (0.0005, 0.0400) | 0.0220 | 0.0210 | (0.0008, 0.0448) | 0.0187 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 11 | 4 | 9 | 0.6458 | 0.7333 |
| proposed_vs_candidate_no_context | persona_consistency | 3 | 9 | 12 | 0.3750 | 0.2500 |
| proposed_vs_candidate_no_context | naturalness | 4 | 11 | 9 | 0.3542 | 0.2667 |
| proposed_vs_candidate_no_context | quest_state_correctness | 11 | 4 | 9 | 0.6458 | 0.7333 |
| proposed_vs_candidate_no_context | lore_consistency | 7 | 1 | 16 | 0.6250 | 0.8750 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| proposed_vs_candidate_no_context | objective_completion_support | 4 | 5 | 15 | 0.4792 | 0.4444 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 8 | 7 | 9 | 0.5208 | 0.5333 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | 5 | 7 | 12 | 0.4583 | 0.4167 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 7 | 2 | 15 | 0.6042 | 0.7778 |
| proposed_vs_candidate_no_context | context_overlap | 12 | 3 | 9 | 0.6875 | 0.8000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 3 | 8 | 13 | 0.3958 | 0.2727 |
| proposed_vs_candidate_no_context | persona_style | 3 | 4 | 17 | 0.4792 | 0.4286 |
| proposed_vs_candidate_no_context | distinct1 | 8 | 7 | 9 | 0.5208 | 0.5333 |
| proposed_vs_candidate_no_context | length_score | 4 | 10 | 10 | 0.3750 | 0.2857 |
| proposed_vs_candidate_no_context | sentence_score | 0 | 2 | 22 | 0.4583 | 0.0000 |
| proposed_vs_candidate_no_context | overall_quality | 9 | 6 | 9 | 0.5625 | 0.6000 |
| controlled_vs_proposed_raw | context_relevance | 9 | 9 | 6 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | persona_consistency | 12 | 5 | 7 | 0.6458 | 0.7059 |
| controlled_vs_proposed_raw | naturalness | 10 | 8 | 6 | 0.5417 | 0.5556 |
| controlled_vs_proposed_raw | quest_state_correctness | 8 | 10 | 6 | 0.4583 | 0.4444 |
| controlled_vs_proposed_raw | lore_consistency | 5 | 4 | 15 | 0.5208 | 0.5556 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0 | 4 | 20 | 0.4167 | 0.0000 |
| controlled_vs_proposed_raw | objective_completion_support | 6 | 4 | 14 | 0.5417 | 0.6000 |
| controlled_vs_proposed_raw | gameplay_usefulness | 8 | 10 | 6 | 0.4583 | 0.4444 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 9 | 4 | 11 | 0.6042 | 0.6923 |
| controlled_vs_proposed_raw | context_keyword_coverage | 6 | 7 | 11 | 0.4792 | 0.4615 |
| controlled_vs_proposed_raw | context_overlap | 9 | 9 | 6 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 12 | 4 | 8 | 0.6667 | 0.7500 |
| controlled_vs_proposed_raw | persona_style | 2 | 7 | 15 | 0.3958 | 0.2222 |
| controlled_vs_proposed_raw | distinct1 | 8 | 9 | 7 | 0.4792 | 0.4706 |
| controlled_vs_proposed_raw | length_score | 10 | 7 | 7 | 0.5625 | 0.5882 |
| controlled_vs_proposed_raw | sentence_score | 3 | 0 | 21 | 0.5625 | 1.0000 |
| controlled_vs_proposed_raw | overall_quality | 8 | 10 | 6 | 0.4583 | 0.4444 |
| controlled_vs_candidate_no_context | context_relevance | 11 | 5 | 8 | 0.6250 | 0.6875 |
| controlled_vs_candidate_no_context | persona_consistency | 8 | 6 | 10 | 0.5417 | 0.5714 |
| controlled_vs_candidate_no_context | naturalness | 6 | 10 | 8 | 0.4167 | 0.3750 |
| controlled_vs_candidate_no_context | quest_state_correctness | 12 | 4 | 8 | 0.6667 | 0.7500 |
| controlled_vs_candidate_no_context | lore_consistency | 5 | 1 | 18 | 0.5833 | 0.8333 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 4 | 20 | 0.4167 | 0.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | 6 | 3 | 15 | 0.5625 | 0.6667 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 8 | 8 | 8 | 0.5000 | 0.5000 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 6 | 4 | 14 | 0.5417 | 0.6000 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 9 | 2 | 13 | 0.6458 | 0.8182 |
| controlled_vs_candidate_no_context | context_overlap | 11 | 5 | 8 | 0.6250 | 0.6875 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 7 | 3 | 14 | 0.5833 | 0.7000 |
| controlled_vs_candidate_no_context | persona_style | 2 | 6 | 16 | 0.4167 | 0.2500 |
| controlled_vs_candidate_no_context | distinct1 | 7 | 9 | 8 | 0.4583 | 0.4375 |
| controlled_vs_candidate_no_context | length_score | 7 | 8 | 9 | 0.4792 | 0.4667 |
| controlled_vs_candidate_no_context | sentence_score | 1 | 0 | 23 | 0.5208 | 1.0000 |
| controlled_vs_candidate_no_context | overall_quality | 12 | 4 | 8 | 0.6667 | 0.7500 |
| controlled_alt_vs_controlled_default | context_relevance | 4 | 8 | 12 | 0.4167 | 0.3333 |
| controlled_alt_vs_controlled_default | persona_consistency | 3 | 5 | 16 | 0.4583 | 0.3750 |
| controlled_alt_vs_controlled_default | naturalness | 7 | 5 | 12 | 0.5417 | 0.5833 |
| controlled_alt_vs_controlled_default | quest_state_correctness | 4 | 8 | 12 | 0.4167 | 0.3333 |
| controlled_alt_vs_controlled_default | lore_consistency | 4 | 5 | 15 | 0.4792 | 0.4444 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 4 | 0 | 20 | 0.5833 | 1.0000 |
| controlled_alt_vs_controlled_default | objective_completion_support | 4 | 5 | 15 | 0.4792 | 0.4444 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | 5 | 7 | 12 | 0.4583 | 0.4167 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | 4 | 8 | 12 | 0.4167 | 0.3333 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 2 | 7 | 15 | 0.3958 | 0.2222 |
| controlled_alt_vs_controlled_default | context_overlap | 5 | 7 | 12 | 0.4583 | 0.4167 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 2 | 4 | 18 | 0.4583 | 0.3333 |
| controlled_alt_vs_controlled_default | persona_style | 3 | 2 | 19 | 0.5208 | 0.6000 |
| controlled_alt_vs_controlled_default | distinct1 | 8 | 4 | 12 | 0.5833 | 0.6667 |
| controlled_alt_vs_controlled_default | length_score | 6 | 5 | 13 | 0.5208 | 0.5455 |
| controlled_alt_vs_controlled_default | sentence_score | 0 | 4 | 20 | 0.4167 | 0.0000 |
| controlled_alt_vs_controlled_default | overall_quality | 3 | 9 | 12 | 0.3750 | 0.2500 |
| controlled_alt_vs_proposed_raw | context_relevance | 11 | 7 | 6 | 0.5833 | 0.6111 |
| controlled_alt_vs_proposed_raw | persona_consistency | 10 | 5 | 9 | 0.6042 | 0.6667 |
| controlled_alt_vs_proposed_raw | naturalness | 8 | 10 | 6 | 0.4583 | 0.4444 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 10 | 8 | 6 | 0.5417 | 0.5556 |
| controlled_alt_vs_proposed_raw | lore_consistency | 5 | 3 | 16 | 0.5417 | 0.6250 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_alt_vs_proposed_raw | objective_completion_support | 5 | 4 | 15 | 0.5208 | 0.5556 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 7 | 11 | 6 | 0.4167 | 0.3889 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | 5 | 5 | 14 | 0.5000 | 0.5000 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 4 | 5 | 15 | 0.4792 | 0.4444 |
| controlled_alt_vs_proposed_raw | context_overlap | 11 | 7 | 6 | 0.5833 | 0.6111 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 10 | 4 | 10 | 0.6250 | 0.7143 |
| controlled_alt_vs_proposed_raw | persona_style | 2 | 5 | 17 | 0.4375 | 0.2857 |
| controlled_alt_vs_proposed_raw | distinct1 | 9 | 7 | 8 | 0.5417 | 0.5625 |
| controlled_alt_vs_proposed_raw | length_score | 9 | 9 | 6 | 0.5000 | 0.5000 |
| controlled_alt_vs_proposed_raw | sentence_score | 2 | 3 | 19 | 0.4792 | 0.4000 |
| controlled_alt_vs_proposed_raw | overall_quality | 11 | 7 | 6 | 0.5833 | 0.6111 |
| controlled_alt_vs_candidate_no_context | context_relevance | 13 | 4 | 7 | 0.6875 | 0.7647 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 7 | 5 | 12 | 0.5417 | 0.5833 |
| controlled_alt_vs_candidate_no_context | naturalness | 8 | 9 | 7 | 0.4792 | 0.4706 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 14 | 3 | 7 | 0.7292 | 0.8235 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 8 | 1 | 15 | 0.6458 | 0.8889 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_alt_vs_candidate_no_context | objective_completion_support | 4 | 4 | 16 | 0.5000 | 0.5000 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 11 | 6 | 7 | 0.6042 | 0.6471 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | 6 | 6 | 12 | 0.5000 | 0.5000 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 8 | 2 | 14 | 0.6250 | 0.8000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 13 | 4 | 7 | 0.6875 | 0.7647 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 6 | 3 | 15 | 0.5625 | 0.6667 |
| controlled_alt_vs_candidate_no_context | persona_style | 2 | 4 | 18 | 0.4583 | 0.3333 |
| controlled_alt_vs_candidate_no_context | distinct1 | 9 | 8 | 7 | 0.5208 | 0.5294 |
| controlled_alt_vs_candidate_no_context | length_score | 9 | 8 | 7 | 0.5208 | 0.5294 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0 | 3 | 21 | 0.4375 | 0.0000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 12 | 5 | 7 | 0.6458 | 0.7059 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.0833 | 0.1667 | 0.8333 |
| proposed_contextual_controlled_tuned | 0.0000 | 0.0000 | 0.0833 | 0.2917 | 0.7083 |
| proposed_contextual | 0.0000 | 0.0000 | 0.5417 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.5833 | 0.0000 | 0.0000 |

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
| proposed_contextual_controlled | 0.2000 | 0.8000 | 1 | 5 |
| proposed_contextual_controlled_tuned | 0.0000 | 1.0000 | 0 | 5 |
| proposed_contextual | 0.0000 | 1.0000 | 0 | 5 |
| candidate_no_context | 0.0000 | 1.0000 | 0 | 5 |
- Detailed source-level values are published in `multi_turn_contradictions.json`.

- BERTScore status: unavailable (disabled_by_flag).

This report covers proposal RO5 metrics plus game-facing outcomes: quest-state correctness, lore consistency, contradiction safety, objective completion support, gameplay usefulness, and time-pressure acceptability.
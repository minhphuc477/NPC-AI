# Proposal Alignment Evaluation Report

- Run ID: `20260313T004644Z`
- Generated: `2026-03-13T00:49:35.892839+00:00`
- Scenarios: `artifacts\proposal_control_tuning\auto_tune\20260313T004330Z\train_runs\trial_001\seed_19\20260313T004644Z\scenarios.jsonl`
- Scenario count: `12`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_tuned`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.1466 (0.0784, 0.2207) | 0.2984 (0.2325, 0.3643) | 0.8578 (0.8300, 0.8830) | 0.3408 (0.2960, 0.3837) | n/a |
| proposed_contextual_controlled_tuned | 0.1157 (0.0543, 0.1922) | 0.3079 (0.2310, 0.3944) | 0.8801 (0.8572, 0.9088) | 0.3341 (0.2836, 0.3885) | n/a |
| proposed_contextual | 0.1214 (0.0551, 0.1980) | 0.2571 (0.1825, 0.3341) | 0.8700 (0.8476, 0.8982) | 0.3156 (0.2628, 0.3713) | n/a |
| candidate_no_context | 0.0425 (0.0244, 0.0599) | 0.2381 (0.1528, 0.3385) | 0.8642 (0.8466, 0.8834) | 0.2711 (0.2313, 0.3184) | n/a |

## Game-facing Outcome Metrics (mean, 95% CI)
| Arm | Quest-state Correctness | Lore Consistency | Contradiction Safety | Objective Completion Support | Gameplay Usefulness | Time-pressure Acceptability |
|---|---:|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2305 (0.1734, 0.2937) | 0.0663 (0.0174, 0.1282) | 1.0000 (1.0000, 1.0000) | 0.1347 (0.0731, 0.1951) | 0.3311 (0.3057, 0.3568) | 0.3310 (0.2807, 0.3788) |
| proposed_contextual_controlled_tuned | 0.2096 (0.1534, 0.2778) | 0.0482 (0.0000, 0.1196) | 1.0000 (1.0000, 1.0000) | 0.1391 (0.0960, 0.1847) | 0.3412 (0.3233, 0.3587) | 0.3472 (0.3192, 0.3758) |
| proposed_contextual | 0.2103 (0.1534, 0.2831) | 0.0466 (0.0069, 0.0943) | 1.0000 (1.0000, 1.0000) | 0.1138 (0.0510, 0.1808) | 0.3266 (0.2946, 0.3588) | 0.3238 (0.2854, 0.3660) |
| candidate_no_context | 0.1405 (0.1228, 0.1591) | 0.0068 (0.0018, 0.0129) | 1.0000 (1.0000, 1.0000) | 0.1014 (0.0483, 0.1626) | 0.2977 (0.2682, 0.3291) | 0.3244 (0.2958, 0.3605) |

- Multi-turn contradiction rate is reported as `1 - contradiction_safety` in row-level outputs.

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0789 | 1.8564 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0190 | 0.0800 |
| proposed_vs_candidate_no_context | naturalness | 0.0058 | 0.0067 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.0698 | 0.4965 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0398 | 5.8348 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | 0.0124 | 0.1226 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 0.0289 | 0.0970 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | -0.0006 | -0.0017 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0985 | 2.1667 |
| proposed_vs_candidate_no_context | context_overlap | 0.0332 | 0.9326 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0238 | 0.1818 |
| proposed_vs_candidate_no_context | persona_style | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | distinct1 | -0.0068 | -0.0072 |
| proposed_vs_candidate_no_context | length_score | 0.0278 | 0.0562 |
| proposed_vs_candidate_no_context | sentence_score | 0.0292 | 0.0330 |
| proposed_vs_candidate_no_context | overall_quality | 0.0446 | 0.1645 |
| controlled_vs_proposed_raw | context_relevance | 0.0252 | 0.2077 |
| controlled_vs_proposed_raw | persona_consistency | 0.0413 | 0.1605 |
| controlled_vs_proposed_raw | naturalness | -0.0122 | -0.0140 |
| controlled_vs_proposed_raw | quest_state_correctness | 0.0202 | 0.0960 |
| controlled_vs_proposed_raw | lore_consistency | 0.0198 | 0.4245 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0209 | 0.1836 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0044 | 0.0136 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 0.0071 | 0.0221 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.0379 | 0.2632 |
| controlled_vs_proposed_raw | context_overlap | -0.0043 | -0.0628 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.0516 | 0.3333 |
| controlled_vs_proposed_raw | persona_style | 0.0000 | 0.0000 |
| controlled_vs_proposed_raw | distinct1 | -0.0117 | -0.0126 |
| controlled_vs_proposed_raw | length_score | -0.0667 | -0.1277 |
| controlled_vs_proposed_raw | sentence_score | 0.0583 | 0.0639 |
| controlled_vs_proposed_raw | overall_quality | 0.0251 | 0.0797 |
| controlled_vs_candidate_no_context | context_relevance | 0.1041 | 2.4497 |
| controlled_vs_candidate_no_context | persona_consistency | 0.0603 | 0.2533 |
| controlled_vs_candidate_no_context | naturalness | -0.0064 | -0.0074 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.0900 | 0.6401 |
| controlled_vs_candidate_no_context | lore_consistency | 0.0595 | 8.7363 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | 0.0333 | 0.3288 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0333 | 0.1119 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 0.0066 | 0.0203 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.1364 | 3.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0289 | 0.8112 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.0754 | 0.5758 |
| controlled_vs_candidate_no_context | persona_style | 0.0000 | 0.0000 |
| controlled_vs_candidate_no_context | distinct1 | -0.0185 | -0.0196 |
| controlled_vs_candidate_no_context | length_score | -0.0389 | -0.0787 |
| controlled_vs_candidate_no_context | sentence_score | 0.0875 | 0.0991 |
| controlled_vs_candidate_no_context | overall_quality | 0.0697 | 0.2572 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0309 | -0.2111 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0095 | 0.0319 |
| controlled_alt_vs_controlled_default | naturalness | 0.0224 | 0.0261 |
| controlled_alt_vs_controlled_default | quest_state_correctness | -0.0209 | -0.0905 |
| controlled_alt_vs_controlled_default | lore_consistency | -0.0181 | -0.2730 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_controlled_default | objective_completion_support | 0.0044 | 0.0325 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | 0.0101 | 0.0306 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | 0.0162 | 0.0489 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0455 | -0.2500 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0029 | 0.0449 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0119 | 0.0577 |
| controlled_alt_vs_controlled_default | persona_style | 0.0000 | 0.0000 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0090 | 0.0098 |
| controlled_alt_vs_controlled_default | length_score | 0.1083 | 0.2378 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0292 | -0.0300 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0067 | -0.0198 |
| controlled_alt_vs_proposed_raw | context_relevance | -0.0057 | -0.0472 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.0508 | 0.1975 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0102 | 0.0117 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | -0.0007 | -0.0032 |
| controlled_alt_vs_proposed_raw | lore_consistency | 0.0017 | 0.0357 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_proposed_raw | objective_completion_support | 0.0253 | 0.2221 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 0.0146 | 0.0447 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | 0.0233 | 0.0721 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | -0.0076 | -0.0526 |
| controlled_alt_vs_proposed_raw | context_overlap | -0.0014 | -0.0207 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.0635 | 0.4103 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0000 | 0.0000 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0027 | -0.0029 |
| controlled_alt_vs_proposed_raw | length_score | 0.0417 | 0.0798 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0292 | 0.0320 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.0184 | 0.0583 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.0732 | 1.7216 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.0698 | 0.2933 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0159 | 0.0184 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 0.0691 | 0.4917 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 0.0414 | 6.0786 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_candidate_no_context | objective_completion_support | 0.0377 | 0.3719 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 0.0435 | 0.1460 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | 0.0228 | 0.0702 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.0909 | 2.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0318 | 0.8925 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.0873 | 0.6667 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0000 | 0.0000 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0095 | -0.0100 |
| controlled_alt_vs_candidate_no_context | length_score | 0.0694 | 0.1404 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.0583 | 0.0660 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.0630 | 0.2324 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0789 | (0.0174, 0.1495) | 0.0040 | 0.0789 | (0.0118, 0.1591) | 0.0037 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0190 | (-0.0667, 0.0937) | 0.2843 | 0.0190 | (-0.0533, 0.0727) | 0.2953 |
| proposed_vs_candidate_no_context | naturalness | 0.0058 | (-0.0201, 0.0294) | 0.3160 | 0.0058 | (-0.0011, 0.0164) | 0.0863 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.0698 | (0.0141, 0.1341) | 0.0040 | 0.0698 | (0.0100, 0.1494) | 0.0027 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0398 | (0.0004, 0.0887) | 0.0220 | 0.0398 | (0.0002, 0.1090) | 0.0033 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | 0.0124 | (-0.0323, 0.0696) | 0.3290 | 0.0124 | (-0.0400, 0.0548) | 0.3433 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 0.0289 | (0.0086, 0.0543) | 0.0013 | 0.0289 | (0.0124, 0.0380) | 0.0043 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | -0.0006 | (-0.0241, 0.0320) | 0.5540 | -0.0006 | (-0.0376, 0.0292) | 0.5667 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0985 | (0.0152, 0.1894) | 0.0107 | 0.0985 | (0.0152, 0.1983) | 0.0033 |
| proposed_vs_candidate_no_context | context_overlap | 0.0332 | (0.0071, 0.0651) | 0.0030 | 0.0332 | (0.0025, 0.0677) | 0.0053 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0238 | (-0.0972, 0.1171) | 0.3360 | 0.0238 | (-0.0667, 0.0909) | 0.3723 |
| proposed_vs_candidate_no_context | persona_style | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | distinct1 | -0.0068 | (-0.0316, 0.0141) | 0.7020 | -0.0068 | (-0.0186, 0.0085) | 0.7913 |
| proposed_vs_candidate_no_context | length_score | 0.0278 | (-0.0583, 0.1140) | 0.2673 | 0.0278 | (-0.0487, 0.1282) | 0.3377 |
| proposed_vs_candidate_no_context | sentence_score | 0.0292 | (-0.0875, 0.1458) | 0.4210 | 0.0292 | (-0.1400, 0.1885) | 0.4297 |
| proposed_vs_candidate_no_context | overall_quality | 0.0446 | (-0.0149, 0.1038) | 0.0657 | 0.0446 | (0.0075, 0.0982) | 0.0030 |
| controlled_vs_proposed_raw | context_relevance | 0.0252 | (-0.0617, 0.1243) | 0.3253 | 0.0252 | (0.0010, 0.0535) | 0.0063 |
| controlled_vs_proposed_raw | persona_consistency | 0.0413 | (-0.0111, 0.0937) | 0.0817 | 0.0413 | (-0.0000, 0.0727) | 0.0607 |
| controlled_vs_proposed_raw | naturalness | -0.0122 | (-0.0364, 0.0128) | 0.8327 | -0.0122 | (-0.0302, 0.0058) | 0.9107 |
| controlled_vs_proposed_raw | quest_state_correctness | 0.0202 | (-0.0608, 0.1062) | 0.3567 | 0.0202 | (0.0003, 0.0516) | 0.0043 |
| controlled_vs_proposed_raw | lore_consistency | 0.0198 | (-0.0455, 0.0839) | 0.2650 | 0.0198 | (-0.0020, 0.0415) | 0.0600 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0209 | (-0.0165, 0.0702) | 0.1910 | 0.0209 | (-0.0160, 0.0473) | 0.1377 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0044 | (-0.0225, 0.0382) | 0.4230 | 0.0044 | (-0.0149, 0.0205) | 0.3550 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 0.0071 | (-0.0289, 0.0463) | 0.3497 | 0.0071 | (-0.0182, 0.0253) | 0.3083 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.0379 | (-0.0758, 0.1667) | 0.3010 | 0.0379 | (0.0000, 0.0758) | 0.0703 |
| controlled_vs_proposed_raw | context_overlap | -0.0043 | (-0.0436, 0.0256) | 0.5930 | -0.0043 | (-0.0208, 0.0053) | 0.6820 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.0516 | (-0.0139, 0.1210) | 0.0790 | 0.0516 | (0.0000, 0.0909) | 0.0633 |
| controlled_vs_proposed_raw | persona_style | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_proposed_raw | distinct1 | -0.0117 | (-0.0297, 0.0046) | 0.9223 | -0.0117 | (-0.0178, -0.0049) | 1.0000 |
| controlled_vs_proposed_raw | length_score | -0.0667 | (-0.1778, 0.0472) | 0.8837 | -0.0667 | (-0.1455, 0.0111) | 0.9343 |
| controlled_vs_proposed_raw | sentence_score | 0.0583 | (-0.0583, 0.1750) | 0.2240 | 0.0583 | (0.0000, 0.1000) | 0.0667 |
| controlled_vs_proposed_raw | overall_quality | 0.0251 | (-0.0292, 0.0766) | 0.1917 | 0.0251 | (0.0063, 0.0478) | 0.0037 |
| controlled_vs_candidate_no_context | context_relevance | 0.1041 | (0.0359, 0.1823) | 0.0000 | 0.1041 | (0.0128, 0.1954) | 0.0037 |
| controlled_vs_candidate_no_context | persona_consistency | 0.0603 | (-0.0444, 0.1429) | 0.1097 | 0.0603 | (0.0000, 0.1034) | 0.0033 |
| controlled_vs_candidate_no_context | naturalness | -0.0064 | (-0.0487, 0.0293) | 0.6083 | -0.0064 | (-0.0188, 0.0047) | 0.8753 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.0900 | (0.0316, 0.1575) | 0.0000 | 0.0900 | (0.0104, 0.1695) | 0.0033 |
| controlled_vs_candidate_no_context | lore_consistency | 0.0595 | (0.0102, 0.1170) | 0.0003 | 0.0595 | (-0.0017, 0.1452) | 0.0590 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | 0.0333 | (-0.0193, 0.0988) | 0.1340 | 0.0333 | (-0.0129, 0.0903) | 0.3130 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0333 | (0.0015, 0.0673) | 0.0193 | 0.0333 | (0.0111, 0.0492) | 0.0037 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 0.0066 | (-0.0362, 0.0499) | 0.3983 | 0.0066 | (-0.0305, 0.0467) | 0.4300 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.1364 | (0.0455, 0.2348) | 0.0003 | 0.1364 | (0.0152, 0.2576) | 0.0023 |
| controlled_vs_candidate_no_context | context_overlap | 0.0289 | (0.0124, 0.0479) | 0.0000 | 0.0289 | (0.0074, 0.0504) | 0.0020 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.0754 | (-0.0556, 0.1825) | 0.1190 | 0.0754 | (-0.0000, 0.1293) | 0.0670 |
| controlled_vs_candidate_no_context | persona_style | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | distinct1 | -0.0185 | (-0.0458, 0.0077) | 0.9110 | -0.0185 | (-0.0289, -0.0040) | 1.0000 |
| controlled_vs_candidate_no_context | length_score | -0.0389 | (-0.2139, 0.0944) | 0.6773 | -0.0389 | (-0.1306, 0.0095) | 0.6897 |
| controlled_vs_candidate_no_context | sentence_score | 0.0875 | (-0.0292, 0.2042) | 0.1083 | 0.0875 | (-0.0700, 0.2000) | 0.1850 |
| controlled_vs_candidate_no_context | overall_quality | 0.0697 | (0.0151, 0.1221) | 0.0043 | 0.0697 | (0.0237, 0.1123) | 0.0030 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0309 | (-0.0891, 0.0169) | 0.8673 | -0.0309 | (-0.0880, 0.0124) | 0.8737 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0095 | (-0.0444, 0.0730) | 0.4340 | 0.0095 | (-0.0267, 0.0364) | 0.3740 |
| controlled_alt_vs_controlled_default | naturalness | 0.0224 | (-0.0102, 0.0574) | 0.1010 | 0.0224 | (0.0056, 0.0444) | 0.0063 |
| controlled_alt_vs_controlled_default | quest_state_correctness | -0.0209 | (-0.0780, 0.0278) | 0.7750 | -0.0209 | (-0.0798, 0.0176) | 0.7957 |
| controlled_alt_vs_controlled_default | lore_consistency | -0.0181 | (-0.0764, 0.0410) | 0.7390 | -0.0181 | (-0.0726, 0.0179) | 0.6937 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | objective_completion_support | 0.0044 | (-0.0398, 0.0425) | 0.3987 | 0.0044 | (-0.0343, 0.0441) | 0.4200 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | 0.0101 | (-0.0110, 0.0364) | 0.2140 | 0.0101 | (-0.0114, 0.0422) | 0.2537 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | 0.0162 | (-0.0132, 0.0496) | 0.1687 | 0.0162 | (-0.0160, 0.0626) | 0.2043 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0455 | (-0.1212, 0.0152) | 0.9287 | -0.0455 | (-0.1240, 0.0152) | 0.9260 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0029 | (-0.0269, 0.0340) | 0.4150 | 0.0029 | (-0.0031, 0.0079) | 0.2053 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0119 | (-0.0556, 0.0833) | 0.3880 | 0.0119 | (-0.0333, 0.0455) | 0.3000 |
| controlled_alt_vs_controlled_default | persona_style | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0090 | (-0.0266, 0.0423) | 0.2983 | 0.0090 | (-0.0058, 0.0199) | 0.1170 |
| controlled_alt_vs_controlled_default | length_score | 0.1083 | (-0.0306, 0.2611) | 0.0690 | 0.1083 | (0.0267, 0.2361) | 0.0027 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0292 | (-0.1167, 0.0583) | 0.8277 | -0.0292 | (-0.1000, 0.0700) | 0.8193 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0067 | (-0.0554, 0.0392) | 0.6190 | -0.0067 | (-0.0462, 0.0189) | 0.6220 |
| controlled_alt_vs_proposed_raw | context_relevance | -0.0057 | (-0.1217, 0.1121) | 0.5530 | -0.0057 | (-0.0495, 0.0219) | 0.5770 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.0508 | (-0.0222, 0.1365) | 0.1173 | 0.0508 | (-0.0267, 0.1091) | 0.0960 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0102 | (-0.0168, 0.0399) | 0.2593 | 0.0102 | (-0.0093, 0.0241) | 0.1750 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | -0.0007 | (-0.1033, 0.1044) | 0.5183 | -0.0007 | (-0.0513, 0.0366) | 0.5120 |
| controlled_alt_vs_proposed_raw | lore_consistency | 0.0017 | (-0.0761, 0.0859) | 0.4997 | 0.0017 | (-0.0443, 0.0477) | 0.3783 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_proposed_raw | objective_completion_support | 0.0253 | (-0.0138, 0.0633) | 0.1053 | 0.0253 | (-0.0011, 0.0623) | 0.0713 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 0.0146 | (-0.0164, 0.0491) | 0.2047 | 0.0146 | (0.0043, 0.0267) | 0.0043 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | 0.0233 | (0.0049, 0.0412) | 0.0053 | 0.0233 | (0.0029, 0.0437) | 0.0027 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | -0.0076 | (-0.1515, 0.1364) | 0.5817 | -0.0076 | (-0.0606, 0.0260) | 0.6820 |
| controlled_alt_vs_proposed_raw | context_overlap | -0.0014 | (-0.0509, 0.0530) | 0.5303 | -0.0014 | (-0.0235, 0.0125) | 0.5270 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.0635 | (-0.0278, 0.1627) | 0.1060 | 0.0635 | (-0.0333, 0.1364) | 0.0750 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0027 | (-0.0366, 0.0315) | 0.5730 | -0.0027 | (-0.0227, 0.0117) | 0.6370 |
| controlled_alt_vs_proposed_raw | length_score | 0.0417 | (-0.0833, 0.1972) | 0.3170 | 0.0417 | (-0.0800, 0.1545) | 0.2750 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0292 | (-0.0583, 0.1167) | 0.3823 | 0.0292 | (-0.0808, 0.1615) | 0.4167 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.0184 | (-0.0589, 0.1006) | 0.3200 | 0.0184 | (-0.0322, 0.0546) | 0.1937 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.0732 | (0.0118, 0.1519) | 0.0027 | 0.0732 | (0.0253, 0.1211) | 0.0023 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.0698 | (0.0000, 0.1397) | 0.0170 | 0.0698 | (0.0148, 0.1232) | 0.0023 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0159 | (-0.0057, 0.0431) | 0.0887 | 0.0159 | (0.0028, 0.0256) | 0.0043 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 0.0691 | (0.0145, 0.1349) | 0.0020 | 0.0691 | (0.0280, 0.1102) | 0.0063 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 0.0414 | (-0.0049, 0.1086) | 0.0983 | 0.0414 | (0.0028, 0.0800) | 0.0033 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_candidate_no_context | objective_completion_support | 0.0377 | (0.0026, 0.0815) | 0.0163 | 0.0377 | (0.0083, 0.0587) | 0.0030 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 0.0435 | (0.0137, 0.0732) | 0.0013 | 0.0435 | (0.0166, 0.0639) | 0.0043 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | 0.0228 | (-0.0007, 0.0504) | 0.0310 | 0.0228 | (-0.0048, 0.0425) | 0.0590 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.0909 | (0.0152, 0.1818) | 0.0010 | 0.0909 | (0.0303, 0.1515) | 0.0043 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0318 | (0.0001, 0.0735) | 0.0233 | 0.0318 | (0.0135, 0.0501) | 0.0030 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.0873 | (0.0079, 0.1806) | 0.0137 | 0.0873 | (0.0185, 0.1540) | 0.0040 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0095 | (-0.0308, 0.0096) | 0.8140 | -0.0095 | (-0.0142, -0.0040) | 1.0000 |
| controlled_alt_vs_candidate_no_context | length_score | 0.0694 | (-0.0278, 0.2028) | 0.1203 | 0.0694 | (0.0300, 0.1056) | 0.0063 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.0583 | (-0.0583, 0.1750) | 0.2093 | 0.0583 | (0.0000, 0.1000) | 0.0620 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.0630 | (0.0236, 0.1087) | 0.0000 | 0.0630 | (0.0263, 0.0774) | 0.0047 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 6 | 2 | 4 | 0.6667 | 0.7500 |
| proposed_vs_candidate_no_context | persona_consistency | 4 | 1 | 7 | 0.6250 | 0.8000 |
| proposed_vs_candidate_no_context | naturalness | 5 | 3 | 4 | 0.5833 | 0.6250 |
| proposed_vs_candidate_no_context | quest_state_correctness | 6 | 2 | 4 | 0.6667 | 0.7500 |
| proposed_vs_candidate_no_context | lore_consistency | 5 | 3 | 4 | 0.5833 | 0.6250 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 12 | 0.5000 | nan |
| proposed_vs_candidate_no_context | objective_completion_support | 2 | 3 | 7 | 0.4583 | 0.4000 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 6 | 2 | 4 | 0.6667 | 0.7500 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | 2 | 4 | 6 | 0.4167 | 0.3333 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 5 | 1 | 6 | 0.6667 | 0.8333 |
| proposed_vs_candidate_no_context | context_overlap | 7 | 1 | 4 | 0.7500 | 0.8750 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 4 | 1 | 7 | 0.6250 | 0.8000 |
| proposed_vs_candidate_no_context | persona_style | 0 | 0 | 12 | 0.5000 | nan |
| proposed_vs_candidate_no_context | distinct1 | 3 | 2 | 7 | 0.5417 | 0.6000 |
| proposed_vs_candidate_no_context | length_score | 6 | 2 | 4 | 0.6667 | 0.7500 |
| proposed_vs_candidate_no_context | sentence_score | 3 | 2 | 7 | 0.5417 | 0.6000 |
| proposed_vs_candidate_no_context | overall_quality | 5 | 3 | 4 | 0.5833 | 0.6250 |
| controlled_vs_proposed_raw | context_relevance | 5 | 4 | 3 | 0.5417 | 0.5556 |
| controlled_vs_proposed_raw | persona_consistency | 4 | 1 | 7 | 0.6250 | 0.8000 |
| controlled_vs_proposed_raw | naturalness | 3 | 6 | 3 | 0.3750 | 0.3333 |
| controlled_vs_proposed_raw | quest_state_correctness | 5 | 4 | 3 | 0.5417 | 0.5556 |
| controlled_vs_proposed_raw | lore_consistency | 4 | 3 | 5 | 0.5417 | 0.5714 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0 | 0 | 12 | 0.5000 | nan |
| controlled_vs_proposed_raw | objective_completion_support | 4 | 3 | 5 | 0.5417 | 0.5714 |
| controlled_vs_proposed_raw | gameplay_usefulness | 3 | 6 | 3 | 0.3750 | 0.3333 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 4 | 4 | 4 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | context_keyword_coverage | 2 | 2 | 8 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | context_overlap | 6 | 3 | 3 | 0.6250 | 0.6667 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 4 | 1 | 7 | 0.6250 | 0.8000 |
| controlled_vs_proposed_raw | persona_style | 0 | 0 | 12 | 0.5000 | nan |
| controlled_vs_proposed_raw | distinct1 | 3 | 5 | 4 | 0.4167 | 0.3750 |
| controlled_vs_proposed_raw | length_score | 2 | 7 | 3 | 0.2917 | 0.2222 |
| controlled_vs_proposed_raw | sentence_score | 3 | 1 | 8 | 0.5833 | 0.7500 |
| controlled_vs_proposed_raw | overall_quality | 5 | 4 | 3 | 0.5417 | 0.5556 |
| controlled_vs_candidate_no_context | context_relevance | 9 | 1 | 2 | 0.8333 | 0.9000 |
| controlled_vs_candidate_no_context | persona_consistency | 6 | 1 | 5 | 0.7083 | 0.8571 |
| controlled_vs_candidate_no_context | naturalness | 6 | 4 | 2 | 0.5833 | 0.6000 |
| controlled_vs_candidate_no_context | quest_state_correctness | 9 | 1 | 2 | 0.8333 | 0.9000 |
| controlled_vs_candidate_no_context | lore_consistency | 5 | 2 | 5 | 0.6250 | 0.7143 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 12 | 0.5000 | nan |
| controlled_vs_candidate_no_context | objective_completion_support | 4 | 4 | 4 | 0.5000 | 0.5000 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 6 | 4 | 2 | 0.5833 | 0.6000 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 4 | 4 | 4 | 0.5000 | 0.5000 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 6 | 0 | 6 | 0.7500 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 9 | 1 | 2 | 0.8333 | 0.9000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 6 | 1 | 5 | 0.7083 | 0.8571 |
| controlled_vs_candidate_no_context | persona_style | 0 | 0 | 12 | 0.5000 | nan |
| controlled_vs_candidate_no_context | distinct1 | 5 | 5 | 2 | 0.5000 | 0.5000 |
| controlled_vs_candidate_no_context | length_score | 5 | 5 | 2 | 0.5000 | 0.5000 |
| controlled_vs_candidate_no_context | sentence_score | 4 | 1 | 7 | 0.6250 | 0.8000 |
| controlled_vs_candidate_no_context | overall_quality | 7 | 3 | 2 | 0.6667 | 0.7000 |
| controlled_alt_vs_controlled_default | context_relevance | 3 | 4 | 5 | 0.4583 | 0.4286 |
| controlled_alt_vs_controlled_default | persona_consistency | 2 | 2 | 8 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | naturalness | 5 | 2 | 5 | 0.6250 | 0.7143 |
| controlled_alt_vs_controlled_default | quest_state_correctness | 3 | 4 | 5 | 0.4583 | 0.4286 |
| controlled_alt_vs_controlled_default | lore_consistency | 2 | 4 | 6 | 0.4167 | 0.3333 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0 | 0 | 12 | 0.5000 | nan |
| controlled_alt_vs_controlled_default | objective_completion_support | 3 | 3 | 6 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | 3 | 4 | 5 | 0.4583 | 0.4286 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | 4 | 3 | 5 | 0.5417 | 0.5714 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 2 | 4 | 6 | 0.4167 | 0.3333 |
| controlled_alt_vs_controlled_default | context_overlap | 4 | 3 | 5 | 0.5417 | 0.5714 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 2 | 2 | 8 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | persona_style | 0 | 0 | 12 | 0.5000 | nan |
| controlled_alt_vs_controlled_default | distinct1 | 4 | 3 | 5 | 0.5417 | 0.5714 |
| controlled_alt_vs_controlled_default | length_score | 4 | 3 | 5 | 0.5417 | 0.5714 |
| controlled_alt_vs_controlled_default | sentence_score | 1 | 2 | 9 | 0.4583 | 0.3333 |
| controlled_alt_vs_controlled_default | overall_quality | 2 | 5 | 5 | 0.3750 | 0.2857 |
| controlled_alt_vs_proposed_raw | context_relevance | 6 | 4 | 2 | 0.5833 | 0.6000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 4 | 2 | 6 | 0.5833 | 0.6667 |
| controlled_alt_vs_proposed_raw | naturalness | 5 | 5 | 2 | 0.5000 | 0.5000 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 6 | 4 | 2 | 0.5833 | 0.6000 |
| controlled_alt_vs_proposed_raw | lore_consistency | 3 | 4 | 5 | 0.4583 | 0.4286 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0 | 0 | 12 | 0.5000 | nan |
| controlled_alt_vs_proposed_raw | objective_completion_support | 6 | 4 | 2 | 0.5833 | 0.6000 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 4 | 6 | 2 | 0.4167 | 0.4000 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | 7 | 3 | 2 | 0.6667 | 0.7000 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 3 | 4 | 5 | 0.4583 | 0.4286 |
| controlled_alt_vs_proposed_raw | context_overlap | 6 | 4 | 2 | 0.5833 | 0.6000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 4 | 2 | 6 | 0.5833 | 0.6667 |
| controlled_alt_vs_proposed_raw | persona_style | 0 | 0 | 12 | 0.5000 | nan |
| controlled_alt_vs_proposed_raw | distinct1 | 4 | 5 | 3 | 0.4583 | 0.4444 |
| controlled_alt_vs_proposed_raw | length_score | 4 | 5 | 3 | 0.4583 | 0.4444 |
| controlled_alt_vs_proposed_raw | sentence_score | 2 | 1 | 9 | 0.5417 | 0.6667 |
| controlled_alt_vs_proposed_raw | overall_quality | 3 | 7 | 2 | 0.3333 | 0.3000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 7 | 3 | 2 | 0.6667 | 0.7000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 5 | 1 | 6 | 0.6667 | 0.8333 |
| controlled_alt_vs_candidate_no_context | naturalness | 6 | 4 | 2 | 0.5833 | 0.6000 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 7 | 3 | 2 | 0.6667 | 0.7000 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 3 | 3 | 6 | 0.5000 | 0.5000 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 12 | 0.5000 | nan |
| controlled_alt_vs_candidate_no_context | objective_completion_support | 5 | 2 | 5 | 0.6250 | 0.7143 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 8 | 2 | 2 | 0.7500 | 0.8000 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | 5 | 3 | 4 | 0.5833 | 0.6250 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 5 | 0 | 7 | 0.7083 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 6 | 4 | 2 | 0.5833 | 0.6000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 5 | 1 | 6 | 0.6667 | 0.8333 |
| controlled_alt_vs_candidate_no_context | persona_style | 0 | 0 | 12 | 0.5000 | nan |
| controlled_alt_vs_candidate_no_context | distinct1 | 5 | 4 | 3 | 0.5417 | 0.5556 |
| controlled_alt_vs_candidate_no_context | length_score | 6 | 4 | 2 | 0.5833 | 0.6000 |
| controlled_alt_vs_candidate_no_context | sentence_score | 3 | 1 | 8 | 0.5833 | 0.7500 |
| controlled_alt_vs_candidate_no_context | overall_quality | 8 | 2 | 2 | 0.7500 | 0.8000 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.1667 | 0.1667 | 0.8333 |
| proposed_contextual_controlled_tuned | 0.0000 | 0.0000 | 0.0833 | 0.0833 | 0.9167 |
| proposed_contextual | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 |
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
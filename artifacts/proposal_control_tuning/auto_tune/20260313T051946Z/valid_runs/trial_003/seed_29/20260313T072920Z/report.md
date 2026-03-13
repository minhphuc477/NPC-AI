# Proposal Alignment Evaluation Report

- Run ID: `20260313T072920Z`
- Generated: `2026-03-13T07:35:52.716220+00:00`
- Scenarios: `artifacts\proposal_control_tuning\auto_tune\20260313T051946Z\valid_runs\trial_003\seed_29\20260313T072920Z\scenarios.jsonl`
- Scenario count: `24`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_tuned`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0931 (0.0619, 0.1351) | 0.3299 (0.2796, 0.3771) | 0.8763 (0.8651, 0.8888) | 0.3311 (0.3073, 0.3552) | n/a |
| proposed_contextual_controlled_tuned | 0.1581 (0.0928, 0.2298) | 0.3315 (0.2868, 0.3738) | 0.8682 (0.8545, 0.8817) | 0.3593 (0.3314, 0.3905) | n/a |
| proposed_contextual | 0.0702 (0.0434, 0.0986) | 0.2703 (0.2206, 0.3236) | 0.8730 (0.8545, 0.8887) | 0.2975 (0.2733, 0.3206) | n/a |
| candidate_no_context | 0.0483 (0.0348, 0.0635) | 0.2843 (0.2311, 0.3376) | 0.8814 (0.8704, 0.8936) | 0.2945 (0.2734, 0.3154) | n/a |

## Game-facing Outcome Metrics (mean, 95% CI)
| Arm | Quest-state Correctness | Lore Consistency | Contradiction Safety | Objective Completion Support | Gameplay Usefulness | Time-pressure Acceptability |
|---|---:|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.1850 (0.1562, 0.2193) | 0.0342 (0.0052, 0.0708) | 1.0000 (1.0000, 1.0000) | 0.1273 (0.0829, 0.1755) | 0.3242 (0.3034, 0.3454) | 0.3372 (0.3071, 0.3662) |
| proposed_contextual_controlled_tuned | 0.2445 (0.1923, 0.3054) | 0.0840 (0.0271, 0.1542) | 1.0000 (1.0000, 1.0000) | 0.1415 (0.0970, 0.1872) | 0.3399 (0.3186, 0.3596) | 0.3350 (0.3026, 0.3677) |
| proposed_contextual | 0.1672 (0.1424, 0.1964) | 0.0336 (0.0102, 0.0646) | 1.0000 (1.0000, 1.0000) | 0.1054 (0.0603, 0.1546) | 0.3078 (0.2807, 0.3343) | 0.3278 (0.2996, 0.3564) |
| candidate_no_context | 0.1474 (0.1327, 0.1623) | 0.0068 (0.0015, 0.0136) | 1.0000 (1.0000, 1.0000) | 0.1179 (0.0690, 0.1654) | 0.3126 (0.2867, 0.3375) | 0.3334 (0.3037, 0.3642) |

- Multi-turn contradiction rate is reported as `1 - contradiction_safety` in row-level outputs.

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0219 | 0.4526 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0141 | -0.0494 |
| proposed_vs_candidate_no_context | naturalness | -0.0084 | -0.0095 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.0198 | 0.1343 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0268 | 3.9302 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | -0.0125 | -0.1060 |
| proposed_vs_candidate_no_context | gameplay_usefulness | -0.0049 | -0.0156 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | -0.0055 | -0.0166 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0284 | 0.5521 |
| proposed_vs_candidate_no_context | context_overlap | 0.0066 | 0.1612 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0202 | -0.1067 |
| proposed_vs_candidate_no_context | persona_style | 0.0107 | 0.0161 |
| proposed_vs_candidate_no_context | distinct1 | 0.0009 | 0.0009 |
| proposed_vs_candidate_no_context | length_score | -0.0292 | -0.0538 |
| proposed_vs_candidate_no_context | sentence_score | -0.0292 | -0.0305 |
| proposed_vs_candidate_no_context | overall_quality | 0.0030 | 0.0103 |
| controlled_vs_proposed_raw | context_relevance | 0.0229 | 0.3269 |
| controlled_vs_proposed_raw | persona_consistency | 0.0596 | 0.2207 |
| controlled_vs_proposed_raw | naturalness | 0.0033 | 0.0038 |
| controlled_vs_proposed_raw | quest_state_correctness | 0.0178 | 0.1064 |
| controlled_vs_proposed_raw | lore_consistency | 0.0006 | 0.0174 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0219 | 0.2082 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0164 | 0.0533 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 0.0094 | 0.0286 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.0297 | 0.3715 |
| controlled_vs_proposed_raw | context_overlap | 0.0073 | 0.1523 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.0728 | 0.4297 |
| controlled_vs_proposed_raw | persona_style | 0.0069 | 0.0103 |
| controlled_vs_proposed_raw | distinct1 | -0.0111 | -0.0118 |
| controlled_vs_proposed_raw | length_score | 0.0097 | 0.0190 |
| controlled_vs_proposed_raw | sentence_score | 0.0583 | 0.0629 |
| controlled_vs_proposed_raw | overall_quality | 0.0336 | 0.1128 |
| controlled_vs_candidate_no_context | context_relevance | 0.0448 | 0.9275 |
| controlled_vs_candidate_no_context | persona_consistency | 0.0456 | 0.1603 |
| controlled_vs_candidate_no_context | naturalness | -0.0051 | -0.0058 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.0376 | 0.2550 |
| controlled_vs_candidate_no_context | lore_consistency | 0.0274 | 4.0161 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | 0.0094 | 0.0801 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0115 | 0.0369 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 0.0039 | 0.0116 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.0581 | 1.1288 |
| controlled_vs_candidate_no_context | context_overlap | 0.0139 | 0.3380 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.0526 | 0.2772 |
| controlled_vs_candidate_no_context | persona_style | 0.0176 | 0.0266 |
| controlled_vs_candidate_no_context | distinct1 | -0.0103 | -0.0109 |
| controlled_vs_candidate_no_context | length_score | -0.0194 | -0.0359 |
| controlled_vs_candidate_no_context | sentence_score | 0.0292 | 0.0305 |
| controlled_vs_candidate_no_context | overall_quality | 0.0366 | 0.1243 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0650 | 0.6976 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0016 | 0.0049 |
| controlled_alt_vs_controlled_default | naturalness | -0.0081 | -0.0093 |
| controlled_alt_vs_controlled_default | quest_state_correctness | 0.0594 | 0.3212 |
| controlled_alt_vs_controlled_default | lore_consistency | 0.0498 | 1.4579 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_controlled_default | objective_completion_support | 0.0141 | 0.1110 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | 0.0157 | 0.0485 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | -0.0023 | -0.0067 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0805 | 0.7349 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0288 | 0.5241 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0079 | -0.0328 |
| controlled_alt_vs_controlled_default | persona_style | 0.0399 | 0.0586 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0062 | 0.0067 |
| controlled_alt_vs_controlled_default | length_score | -0.0458 | -0.0878 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0146 | -0.0148 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0282 | 0.0852 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.0879 | 1.2526 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.0613 | 0.2267 |
| controlled_alt_vs_proposed_raw | naturalness | -0.0048 | -0.0055 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 0.0772 | 0.4618 |
| controlled_alt_vs_proposed_raw | lore_consistency | 0.0504 | 1.5007 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_proposed_raw | objective_completion_support | 0.0361 | 0.3423 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 0.0321 | 0.1044 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | 0.0071 | 0.0217 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.1102 | 1.3794 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0360 | 0.7562 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.0649 | 0.3829 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0468 | 0.0695 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0049 | -0.0052 |
| controlled_alt_vs_proposed_raw | length_score | -0.0361 | -0.0705 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0437 | 0.0472 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.0618 | 0.2077 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.1098 | 2.2722 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.0472 | 0.1660 |
| controlled_alt_vs_candidate_no_context | naturalness | -0.0132 | -0.0150 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 0.0970 | 0.6581 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 0.0772 | 11.3288 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_candidate_no_context | objective_completion_support | 0.0236 | 0.2000 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 0.0273 | 0.0872 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | 0.0016 | 0.0048 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.1386 | 2.6933 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0426 | 1.0393 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.0446 | 0.2354 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0575 | 0.0867 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0040 | -0.0043 |
| controlled_alt_vs_candidate_no_context | length_score | -0.0653 | -0.1205 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.0146 | 0.0153 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.0648 | 0.2201 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0219 | (-0.0034, 0.0531) | 0.0490 | 0.0219 | (-0.0006, 0.0699) | 0.0353 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0141 | (-0.0419, 0.0090) | 0.8593 | -0.0141 | (-0.0229, 0.0030) | 0.9640 |
| proposed_vs_candidate_no_context | naturalness | -0.0084 | (-0.0231, 0.0060) | 0.8643 | -0.0084 | (-0.0138, 0.0047) | 0.9593 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.0198 | (-0.0028, 0.0475) | 0.0533 | 0.0198 | (-0.0002, 0.0613) | 0.0337 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0268 | (0.0034, 0.0574) | 0.0080 | 0.0268 | (0.0049, 0.0593) | 0.0000 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | -0.0125 | (-0.0511, 0.0211) | 0.7537 | -0.0125 | (-0.0272, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | gameplay_usefulness | -0.0049 | (-0.0269, 0.0133) | 0.6583 | -0.0049 | (-0.0177, 0.0185) | 0.6327 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | -0.0055 | (-0.0259, 0.0123) | 0.6897 | -0.0055 | (-0.0162, 0.0050) | 0.7467 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0284 | (-0.0076, 0.0682) | 0.0657 | 0.0284 | (0.0000, 0.0833) | 0.0357 |
| proposed_vs_candidate_no_context | context_overlap | 0.0066 | (-0.0040, 0.0186) | 0.1187 | 0.0066 | (-0.0056, 0.0384) | 0.2980 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0202 | (-0.0560, 0.0060) | 0.9350 | -0.0202 | (-0.0286, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | persona_style | 0.0107 | (0.0000, 0.0283) | 0.0403 | 0.0107 | (0.0000, 0.0208) | 0.0417 |
| proposed_vs_candidate_no_context | distinct1 | 0.0009 | (-0.0137, 0.0152) | 0.4627 | 0.0009 | (-0.0006, 0.0041) | 0.2603 |
| proposed_vs_candidate_no_context | length_score | -0.0292 | (-0.1070, 0.0500) | 0.7663 | -0.0292 | (-0.0600, 0.0444) | 0.8533 |
| proposed_vs_candidate_no_context | sentence_score | -0.0292 | (-0.1167, 0.0583) | 0.8260 | -0.0292 | (-0.0583, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | overall_quality | 0.0030 | (-0.0161, 0.0214) | 0.3583 | 0.0030 | (-0.0092, 0.0336) | 0.2973 |
| controlled_vs_proposed_raw | context_relevance | 0.0229 | (-0.0123, 0.0651) | 0.1167 | 0.0229 | (0.0037, 0.0510) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.0596 | (0.0270, 0.0951) | 0.0000 | 0.0596 | (0.0242, 0.1111) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0033 | (-0.0160, 0.0226) | 0.3750 | 0.0033 | (-0.0100, 0.0236) | 0.2960 |
| controlled_vs_proposed_raw | quest_state_correctness | 0.0178 | (-0.0137, 0.0557) | 0.1477 | 0.0178 | (-0.0001, 0.0444) | 0.0423 |
| controlled_vs_proposed_raw | lore_consistency | 0.0006 | (-0.0390, 0.0449) | 0.4997 | 0.0006 | (-0.0264, 0.0350) | 0.3920 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0219 | (-0.0087, 0.0592) | 0.0907 | 0.0219 | (0.0050, 0.0445) | 0.0000 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0164 | (-0.0021, 0.0379) | 0.0430 | 0.0164 | (-0.0004, 0.0273) | 0.0317 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 0.0094 | (-0.0081, 0.0297) | 0.1817 | 0.0094 | (-0.0047, 0.0263) | 0.2980 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.0297 | (-0.0189, 0.0865) | 0.1247 | 0.0297 | (0.0000, 0.0682) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0073 | (-0.0042, 0.0200) | 0.1110 | 0.0073 | (-0.0058, 0.0122) | 0.0320 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.0728 | (0.0317, 0.1189) | 0.0000 | 0.0728 | (0.0250, 0.1389) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.0069 | (-0.0208, 0.0417) | 0.4650 | 0.0069 | (0.0000, 0.0208) | 0.2937 |
| controlled_vs_proposed_raw | distinct1 | -0.0111 | (-0.0256, 0.0016) | 0.9500 | -0.0111 | (-0.0348, 0.0006) | 0.9700 |
| controlled_vs_proposed_raw | length_score | 0.0097 | (-0.0833, 0.1042) | 0.4223 | 0.0097 | (-0.0467, 0.1167) | 0.2780 |
| controlled_vs_proposed_raw | sentence_score | 0.0583 | (-0.0146, 0.1313) | 0.0653 | 0.0583 | (0.0000, 0.1167) | 0.0383 |
| controlled_vs_proposed_raw | overall_quality | 0.0336 | (0.0143, 0.0546) | 0.0000 | 0.0336 | (0.0218, 0.0483) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.0448 | (0.0112, 0.0882) | 0.0010 | 0.0448 | (0.0147, 0.0876) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.0456 | (0.0164, 0.0809) | 0.0000 | 0.0456 | (0.0083, 0.1141) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | -0.0051 | (-0.0232, 0.0131) | 0.7153 | -0.0051 | (-0.0168, 0.0098) | 0.7343 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.0376 | (0.0086, 0.0765) | 0.0020 | 0.0376 | (0.0108, 0.0736) | 0.0000 |
| controlled_vs_candidate_no_context | lore_consistency | 0.0274 | (0.0013, 0.0615) | 0.0180 | 0.0274 | (0.0139, 0.0400) | 0.0000 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | 0.0094 | (-0.0164, 0.0400) | 0.2510 | 0.0094 | (0.0022, 0.0173) | 0.0000 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0115 | (-0.0019, 0.0258) | 0.0497 | 0.0115 | (0.0000, 0.0210) | 0.0000 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 0.0039 | (-0.0094, 0.0187) | 0.3160 | 0.0039 | (-0.0012, 0.0101) | 0.2447 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.0581 | (0.0145, 0.1149) | 0.0013 | 0.0581 | (0.0182, 0.1111) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0139 | (0.0035, 0.0262) | 0.0017 | 0.0139 | (0.0066, 0.0326) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.0526 | (0.0179, 0.0943) | 0.0013 | 0.0526 | (0.0000, 0.1389) | 0.0397 |
| controlled_vs_candidate_no_context | persona_style | 0.0176 | (0.0000, 0.0478) | 0.0393 | 0.0176 | (0.0000, 0.0417) | 0.0343 |
| controlled_vs_candidate_no_context | distinct1 | -0.0103 | (-0.0272, 0.0048) | 0.8913 | -0.0103 | (-0.0307, 0.0000) | 0.9677 |
| controlled_vs_candidate_no_context | length_score | -0.0194 | (-0.1084, 0.0806) | 0.6677 | -0.0194 | (-0.1067, 0.0708) | 0.5990 |
| controlled_vs_candidate_no_context | sentence_score | 0.0292 | (-0.0292, 0.0875) | 0.2067 | 0.0292 | (-0.0437, 0.0700) | 0.2490 |
| controlled_vs_candidate_no_context | overall_quality | 0.0366 | (0.0160, 0.0608) | 0.0000 | 0.0366 | (0.0162, 0.0819) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0650 | (-0.0155, 0.1457) | 0.0533 | 0.0650 | (-0.0076, 0.1274) | 0.0357 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0016 | (-0.0215, 0.0247) | 0.4423 | 0.0016 | (-0.0459, 0.0250) | 0.3760 |
| controlled_alt_vs_controlled_default | naturalness | -0.0081 | (-0.0266, 0.0090) | 0.8190 | -0.0081 | (-0.0184, 0.0092) | 0.8520 |
| controlled_alt_vs_controlled_default | quest_state_correctness | 0.0594 | (-0.0044, 0.1270) | 0.0367 | 0.0594 | (-0.0047, 0.1175) | 0.0410 |
| controlled_alt_vs_controlled_default | lore_consistency | 0.0498 | (-0.0204, 0.1269) | 0.0900 | 0.0498 | (-0.0217, 0.1251) | 0.0390 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | objective_completion_support | 0.0141 | (-0.0074, 0.0409) | 0.1127 | 0.0141 | (-0.0042, 0.0210) | 0.0360 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | 0.0157 | (-0.0009, 0.0345) | 0.0347 | 0.0157 | (0.0003, 0.0272) | 0.0000 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | -0.0023 | (-0.0167, 0.0129) | 0.6310 | -0.0023 | (-0.0142, 0.0092) | 0.7560 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0805 | (-0.0177, 0.1821) | 0.0570 | 0.0805 | (-0.0091, 0.1591) | 0.0307 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0288 | (0.0040, 0.0587) | 0.0080 | 0.0288 | (-0.0040, 0.0535) | 0.0377 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0079 | (-0.0347, 0.0119) | 0.8127 | -0.0079 | (-0.0556, 0.0143) | 0.7500 |
| controlled_alt_vs_controlled_default | persona_style | 0.0399 | (-0.0217, 0.1145) | 0.1217 | 0.0399 | (-0.0071, 0.1250) | 0.3040 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0062 | (-0.0059, 0.0188) | 0.1580 | 0.0062 | (-0.0018, 0.0120) | 0.0397 |
| controlled_alt_vs_controlled_default | length_score | -0.0458 | (-0.1236, 0.0278) | 0.8780 | -0.0458 | (-0.0700, 0.0222) | 0.9610 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0146 | (-0.0729, 0.0292) | 0.8107 | -0.0146 | (-0.0437, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0282 | (-0.0062, 0.0622) | 0.0473 | 0.0282 | (-0.0012, 0.0624) | 0.0353 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.0879 | (0.0168, 0.1667) | 0.0053 | 0.0879 | (-0.0039, 0.1784) | 0.0350 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.0613 | (0.0194, 0.1047) | 0.0033 | 0.0613 | (0.0492, 0.0686) | 0.0000 |
| controlled_alt_vs_proposed_raw | naturalness | -0.0048 | (-0.0244, 0.0150) | 0.6920 | -0.0048 | (-0.0152, 0.0051) | 0.7413 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 0.0772 | (0.0172, 0.1459) | 0.0043 | 0.0772 | (-0.0049, 0.1619) | 0.0360 |
| controlled_alt_vs_proposed_raw | lore_consistency | 0.0504 | (-0.0141, 0.1260) | 0.0773 | 0.0504 | (-0.0325, 0.1177) | 0.1580 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_proposed_raw | objective_completion_support | 0.0361 | (-0.0031, 0.0788) | 0.0353 | 0.0361 | (0.0028, 0.0641) | 0.0000 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 0.0321 | (0.0077, 0.0558) | 0.0017 | 0.0321 | (0.0180, 0.0538) | 0.0000 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | 0.0071 | (-0.0144, 0.0309) | 0.2737 | 0.0071 | (-0.0190, 0.0355) | 0.2973 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.1102 | (0.0262, 0.2131) | 0.0050 | 0.1102 | (-0.0091, 0.2273) | 0.0343 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0360 | (0.0099, 0.0671) | 0.0013 | 0.0360 | (0.0082, 0.0643) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.0649 | (0.0143, 0.1165) | 0.0083 | 0.0649 | (0.0250, 0.0857) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0468 | (0.0051, 0.1032) | 0.0097 | 0.0468 | (-0.0071, 0.1458) | 0.2987 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0049 | (-0.0198, 0.0099) | 0.7340 | -0.0049 | (-0.0228, 0.0029) | 0.7400 |
| controlled_alt_vs_proposed_raw | length_score | -0.0361 | (-0.1361, 0.0597) | 0.7613 | -0.0361 | (-0.1167, 0.0500) | 0.7353 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0437 | (-0.0146, 0.1021) | 0.1163 | 0.0437 | (-0.0437, 0.1167) | 0.2670 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.0618 | (0.0323, 0.0952) | 0.0000 | 0.0618 | (0.0206, 0.0997) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.1098 | (0.0469, 0.1851) | 0.0000 | 0.1098 | (0.0072, 0.1902) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.0472 | (0.0064, 0.0890) | 0.0137 | 0.0472 | (0.0333, 0.0682) | 0.0000 |
| controlled_alt_vs_candidate_no_context | naturalness | -0.0132 | (-0.0300, 0.0007) | 0.9680 | -0.0132 | (-0.0271, 0.0039) | 0.9627 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 0.0970 | (0.0429, 0.1602) | 0.0000 | 0.0970 | (0.0060, 0.1625) | 0.0000 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 0.0772 | (0.0184, 0.1447) | 0.0013 | 0.0772 | (-0.0077, 0.1580) | 0.0377 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_candidate_no_context | objective_completion_support | 0.0236 | (-0.0072, 0.0569) | 0.0710 | 0.0236 | (-0.0019, 0.0369) | 0.0423 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 0.0273 | (0.0092, 0.0467) | 0.0000 | 0.0273 | (0.0003, 0.0474) | 0.0000 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | 0.0016 | (-0.0157, 0.0211) | 0.4457 | 0.0016 | (-0.0140, 0.0193) | 0.3047 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.1386 | (0.0584, 0.2320) | 0.0000 | 0.1386 | (0.0091, 0.2361) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0426 | (0.0179, 0.0722) | 0.0003 | 0.0426 | (0.0026, 0.0830) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.0446 | (-0.0020, 0.0962) | 0.0410 | 0.0446 | (0.0000, 0.0833) | 0.0327 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0575 | (0.0089, 0.1188) | 0.0020 | 0.0575 | (0.0000, 0.1667) | 0.0353 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0040 | (-0.0179, 0.0095) | 0.7180 | -0.0040 | (-0.0187, 0.0030) | 0.7607 |
| controlled_alt_vs_candidate_no_context | length_score | -0.0653 | (-0.1431, 0.0056) | 0.9617 | -0.0653 | (-0.1767, 0.0278) | 0.7070 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.0146 | (-0.0437, 0.0729) | 0.4030 | 0.0146 | (-0.0875, 0.0700) | 0.3690 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.0648 | (0.0356, 0.0987) | 0.0000 | 0.0648 | (0.0150, 0.1136) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 6 | 4 | 14 | 0.5417 | 0.6000 |
| proposed_vs_candidate_no_context | persona_consistency | 3 | 3 | 18 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | naturalness | 3 | 7 | 14 | 0.4167 | 0.3000 |
| proposed_vs_candidate_no_context | quest_state_correctness | 5 | 5 | 14 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | lore_consistency | 8 | 2 | 14 | 0.6250 | 0.8000 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| proposed_vs_candidate_no_context | objective_completion_support | 4 | 4 | 16 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 6 | 4 | 14 | 0.5417 | 0.6000 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | 5 | 4 | 15 | 0.5208 | 0.5556 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 4 | 2 | 18 | 0.5417 | 0.6667 |
| proposed_vs_candidate_no_context | context_overlap | 6 | 4 | 14 | 0.5417 | 0.6000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 1 | 3 | 20 | 0.4583 | 0.2500 |
| proposed_vs_candidate_no_context | persona_style | 3 | 0 | 21 | 0.5625 | 1.0000 |
| proposed_vs_candidate_no_context | distinct1 | 5 | 5 | 14 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | length_score | 4 | 5 | 15 | 0.4792 | 0.4444 |
| proposed_vs_candidate_no_context | sentence_score | 3 | 5 | 16 | 0.4583 | 0.3750 |
| proposed_vs_candidate_no_context | overall_quality | 7 | 3 | 14 | 0.5833 | 0.7000 |
| controlled_vs_proposed_raw | context_relevance | 13 | 11 | 0 | 0.5417 | 0.5417 |
| controlled_vs_proposed_raw | persona_consistency | 9 | 0 | 15 | 0.6875 | 1.0000 |
| controlled_vs_proposed_raw | naturalness | 12 | 12 | 0 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | quest_state_correctness | 13 | 11 | 0 | 0.5417 | 0.5417 |
| controlled_vs_proposed_raw | lore_consistency | 2 | 8 | 14 | 0.3750 | 0.2000 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_vs_proposed_raw | objective_completion_support | 6 | 9 | 9 | 0.4375 | 0.4000 |
| controlled_vs_proposed_raw | gameplay_usefulness | 14 | 10 | 0 | 0.5833 | 0.5833 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 6 | 11 | 7 | 0.3958 | 0.3529 |
| controlled_vs_proposed_raw | context_keyword_coverage | 4 | 1 | 19 | 0.5625 | 0.8000 |
| controlled_vs_proposed_raw | context_overlap | 13 | 11 | 0 | 0.5417 | 0.5417 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 8 | 0 | 16 | 0.6667 | 1.0000 |
| controlled_vs_proposed_raw | persona_style | 1 | 1 | 22 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | distinct1 | 13 | 9 | 2 | 0.5833 | 0.5909 |
| controlled_vs_proposed_raw | length_score | 12 | 12 | 0 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | sentence_score | 5 | 1 | 18 | 0.5833 | 0.8333 |
| controlled_vs_proposed_raw | overall_quality | 16 | 8 | 0 | 0.6667 | 0.6667 |
| controlled_vs_candidate_no_context | context_relevance | 14 | 8 | 2 | 0.6250 | 0.6364 |
| controlled_vs_candidate_no_context | persona_consistency | 7 | 0 | 17 | 0.6458 | 1.0000 |
| controlled_vs_candidate_no_context | naturalness | 8 | 14 | 2 | 0.3750 | 0.3636 |
| controlled_vs_candidate_no_context | quest_state_correctness | 13 | 9 | 2 | 0.5833 | 0.5909 |
| controlled_vs_candidate_no_context | lore_consistency | 5 | 2 | 17 | 0.5625 | 0.7143 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_vs_candidate_no_context | objective_completion_support | 5 | 9 | 10 | 0.4167 | 0.3571 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 14 | 8 | 2 | 0.6250 | 0.6364 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 6 | 10 | 8 | 0.4167 | 0.3750 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 6 | 0 | 18 | 0.6250 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 14 | 8 | 2 | 0.6250 | 0.6364 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 6 | 0 | 18 | 0.6250 | 1.0000 |
| controlled_vs_candidate_no_context | persona_style | 3 | 0 | 21 | 0.5625 | 1.0000 |
| controlled_vs_candidate_no_context | distinct1 | 13 | 8 | 3 | 0.6042 | 0.6190 |
| controlled_vs_candidate_no_context | length_score | 9 | 13 | 2 | 0.4167 | 0.4091 |
| controlled_vs_candidate_no_context | sentence_score | 3 | 1 | 20 | 0.5417 | 0.7500 |
| controlled_vs_candidate_no_context | overall_quality | 15 | 7 | 2 | 0.6667 | 0.6818 |
| controlled_alt_vs_controlled_default | context_relevance | 9 | 3 | 12 | 0.6250 | 0.7500 |
| controlled_alt_vs_controlled_default | persona_consistency | 5 | 5 | 14 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | naturalness | 6 | 6 | 12 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | quest_state_correctness | 9 | 3 | 12 | 0.6250 | 0.7500 |
| controlled_alt_vs_controlled_default | lore_consistency | 7 | 4 | 13 | 0.5625 | 0.6364 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_alt_vs_controlled_default | objective_completion_support | 6 | 4 | 14 | 0.5417 | 0.6000 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | 9 | 3 | 12 | 0.6250 | 0.7500 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | 6 | 6 | 12 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 7 | 3 | 14 | 0.5833 | 0.7000 |
| controlled_alt_vs_controlled_default | context_overlap | 9 | 3 | 12 | 0.6250 | 0.7500 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 1 | 2 | 21 | 0.4792 | 0.3333 |
| controlled_alt_vs_controlled_default | persona_style | 4 | 3 | 17 | 0.5208 | 0.5714 |
| controlled_alt_vs_controlled_default | distinct1 | 7 | 5 | 12 | 0.5417 | 0.5833 |
| controlled_alt_vs_controlled_default | length_score | 6 | 6 | 12 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | sentence_score | 1 | 2 | 21 | 0.4792 | 0.3333 |
| controlled_alt_vs_controlled_default | overall_quality | 8 | 4 | 12 | 0.5833 | 0.6667 |
| controlled_alt_vs_proposed_raw | context_relevance | 16 | 8 | 0 | 0.6667 | 0.6667 |
| controlled_alt_vs_proposed_raw | persona_consistency | 12 | 2 | 10 | 0.7083 | 0.8571 |
| controlled_alt_vs_proposed_raw | naturalness | 12 | 12 | 0 | 0.5000 | 0.5000 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 16 | 8 | 0 | 0.6667 | 0.6667 |
| controlled_alt_vs_proposed_raw | lore_consistency | 6 | 7 | 11 | 0.4792 | 0.4615 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_alt_vs_proposed_raw | objective_completion_support | 9 | 10 | 5 | 0.4792 | 0.4737 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 16 | 8 | 0 | 0.6667 | 0.6667 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | 6 | 14 | 4 | 0.3333 | 0.3000 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 9 | 2 | 13 | 0.6458 | 0.8182 |
| controlled_alt_vs_proposed_raw | context_overlap | 16 | 8 | 0 | 0.6667 | 0.6667 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 9 | 2 | 13 | 0.6458 | 0.8182 |
| controlled_alt_vs_proposed_raw | persona_style | 4 | 2 | 18 | 0.5417 | 0.6667 |
| controlled_alt_vs_proposed_raw | distinct1 | 13 | 10 | 1 | 0.5625 | 0.5652 |
| controlled_alt_vs_proposed_raw | length_score | 12 | 12 | 0 | 0.5000 | 0.5000 |
| controlled_alt_vs_proposed_raw | sentence_score | 4 | 1 | 19 | 0.5625 | 0.8000 |
| controlled_alt_vs_proposed_raw | overall_quality | 16 | 8 | 0 | 0.6667 | 0.6667 |
| controlled_alt_vs_candidate_no_context | context_relevance | 16 | 6 | 2 | 0.7083 | 0.7273 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 10 | 2 | 12 | 0.6667 | 0.8333 |
| controlled_alt_vs_candidate_no_context | naturalness | 12 | 11 | 1 | 0.5208 | 0.5217 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 16 | 6 | 2 | 0.7083 | 0.7273 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 8 | 4 | 12 | 0.5833 | 0.6667 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_alt_vs_candidate_no_context | objective_completion_support | 8 | 8 | 8 | 0.5000 | 0.5000 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 17 | 6 | 1 | 0.7292 | 0.7391 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | 7 | 12 | 5 | 0.3958 | 0.3684 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 10 | 0 | 14 | 0.7083 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 16 | 6 | 2 | 0.7083 | 0.7273 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 6 | 2 | 16 | 0.5833 | 0.7500 |
| controlled_alt_vs_candidate_no_context | persona_style | 6 | 1 | 17 | 0.6042 | 0.8571 |
| controlled_alt_vs_candidate_no_context | distinct1 | 12 | 11 | 1 | 0.5208 | 0.5217 |
| controlled_alt_vs_candidate_no_context | length_score | 11 | 12 | 1 | 0.4792 | 0.4783 |
| controlled_alt_vs_candidate_no_context | sentence_score | 3 | 2 | 19 | 0.5208 | 0.6000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 17 | 6 | 1 | 0.7292 | 0.7391 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.1667 | 0.2083 | 0.7917 |
| proposed_contextual_controlled_tuned | 0.0000 | 0.0000 | 0.0833 | 0.1667 | 0.8333 |
| proposed_contextual | 0.0000 | 0.0000 | 0.6250 | 0.0000 | 0.0000 |
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
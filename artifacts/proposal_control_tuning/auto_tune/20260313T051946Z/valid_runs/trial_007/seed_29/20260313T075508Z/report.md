# Proposal Alignment Evaluation Report

- Run ID: `20260313T075508Z`
- Generated: `2026-03-13T08:02:02.236430+00:00`
- Scenarios: `artifacts\proposal_control_tuning\auto_tune\20260313T051946Z\valid_runs\trial_007\seed_29\20260313T075508Z\scenarios.jsonl`
- Scenario count: `24`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_tuned`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.1036 (0.0691, 0.1436) | 0.3355 (0.2806, 0.3880) | 0.8822 (0.8661, 0.8993) | 0.3390 (0.3140, 0.3643) | n/a |
| proposed_contextual_controlled_tuned | 0.1060 (0.0724, 0.1408) | 0.3871 (0.3134, 0.4648) | 0.8680 (0.8573, 0.8797) | 0.3566 (0.3192, 0.3981) | n/a |
| proposed_contextual | 0.0797 (0.0553, 0.1067) | 0.2959 (0.2443, 0.3515) | 0.8779 (0.8678, 0.8902) | 0.3121 (0.2889, 0.3346) | n/a |
| candidate_no_context | 0.0409 (0.0266, 0.0574) | 0.3045 (0.2361, 0.3879) | 0.8761 (0.8656, 0.8882) | 0.2977 (0.2706, 0.3300) | n/a |

## Game-facing Outcome Metrics (mean, 95% CI)
| Arm | Quest-state Correctness | Lore Consistency | Contradiction Safety | Objective Completion Support | Gameplay Usefulness | Time-pressure Acceptability |
|---|---:|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.1929 (0.1647, 0.2270) | 0.0421 (0.0099, 0.0813) | 1.0000 (1.0000, 1.0000) | 0.1216 (0.0770, 0.1679) | 0.3261 (0.3064, 0.3446) | 0.3344 (0.3056, 0.3618) |
| proposed_contextual_controlled_tuned | 0.1950 (0.1652, 0.2247) | 0.0465 (0.0183, 0.0796) | 1.0000 (1.0000, 1.0000) | 0.1230 (0.0788, 0.1695) | 0.3197 (0.2996, 0.3398) | 0.3302 (0.2992, 0.3597) |
| proposed_contextual | 0.1738 (0.1531, 0.1960) | 0.0333 (0.0097, 0.0601) | 1.0000 (1.0000, 1.0000) | 0.1197 (0.0691, 0.1702) | 0.3200 (0.2959, 0.3448) | 0.3358 (0.3051, 0.3669) |
| candidate_no_context | 0.1407 (0.1249, 0.1560) | 0.0107 (0.0013, 0.0233) | 1.0000 (1.0000, 1.0000) | 0.1149 (0.0644, 0.1664) | 0.3065 (0.2811, 0.3334) | 0.3312 (0.2994, 0.3631) |

- Multi-turn contradiction rate is reported as `1 - contradiction_safety` in row-level outputs.

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0388 | 0.9480 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0087 | -0.0285 |
| proposed_vs_candidate_no_context | naturalness | 0.0017 | 0.0020 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.0331 | 0.2352 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0226 | 2.1076 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | 0.0049 | 0.0423 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 0.0136 | 0.0443 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | 0.0046 | 0.0138 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0508 | 1.2481 |
| proposed_vs_candidate_no_context | context_overlap | 0.0107 | 0.2591 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0238 | -0.1085 |
| proposed_vs_candidate_no_context | persona_style | 0.0519 | 0.0805 |
| proposed_vs_candidate_no_context | distinct1 | -0.0048 | -0.0051 |
| proposed_vs_candidate_no_context | length_score | 0.0153 | 0.0293 |
| proposed_vs_candidate_no_context | sentence_score | 0.0146 | 0.0155 |
| proposed_vs_candidate_no_context | overall_quality | 0.0143 | 0.0482 |
| controlled_vs_proposed_raw | context_relevance | 0.0239 | 0.2995 |
| controlled_vs_proposed_raw | persona_consistency | 0.0396 | 0.1339 |
| controlled_vs_proposed_raw | naturalness | 0.0043 | 0.0049 |
| controlled_vs_proposed_raw | quest_state_correctness | 0.0192 | 0.1104 |
| controlled_vs_proposed_raw | lore_consistency | 0.0088 | 0.2644 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0019 | 0.0157 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0060 | 0.0189 |
| controlled_vs_proposed_raw | time_pressure_acceptability | -0.0014 | -0.0043 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.0344 | 0.3759 |
| controlled_vs_proposed_raw | context_overlap | -0.0007 | -0.0134 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.0476 | 0.2434 |
| controlled_vs_proposed_raw | persona_style | 0.0076 | 0.0110 |
| controlled_vs_proposed_raw | distinct1 | 0.0116 | 0.0123 |
| controlled_vs_proposed_raw | length_score | -0.0278 | -0.0518 |
| controlled_vs_proposed_raw | sentence_score | 0.0438 | 0.0458 |
| controlled_vs_proposed_raw | overall_quality | 0.0270 | 0.0864 |
| controlled_vs_candidate_no_context | context_relevance | 0.0627 | 1.5315 |
| controlled_vs_candidate_no_context | persona_consistency | 0.0310 | 0.1016 |
| controlled_vs_candidate_no_context | naturalness | 0.0061 | 0.0069 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.0523 | 0.3716 |
| controlled_vs_candidate_no_context | lore_consistency | 0.0313 | 2.9294 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | 0.0067 | 0.0586 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0196 | 0.0640 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 0.0031 | 0.0095 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.0852 | 2.0930 |
| controlled_vs_candidate_no_context | context_overlap | 0.0100 | 0.2422 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.0238 | 0.1085 |
| controlled_vs_candidate_no_context | persona_style | 0.0595 | 0.0923 |
| controlled_vs_candidate_no_context | distinct1 | 0.0068 | 0.0072 |
| controlled_vs_candidate_no_context | length_score | -0.0125 | -0.0240 |
| controlled_vs_candidate_no_context | sentence_score | 0.0583 | 0.0619 |
| controlled_vs_candidate_no_context | overall_quality | 0.0413 | 0.1387 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0024 | 0.0231 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0516 | 0.1540 |
| controlled_alt_vs_controlled_default | naturalness | -0.0142 | -0.0161 |
| controlled_alt_vs_controlled_default | quest_state_correctness | 0.0020 | 0.0106 |
| controlled_alt_vs_controlled_default | lore_consistency | 0.0044 | 0.1054 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_controlled_default | objective_completion_support | 0.0014 | 0.0117 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | -0.0064 | -0.0195 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | -0.0042 | -0.0124 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0022 | 0.0175 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0028 | 0.0551 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0641 | 0.2635 |
| controlled_alt_vs_controlled_default | persona_style | 0.0019 | 0.0027 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0024 | -0.0025 |
| controlled_alt_vs_controlled_default | length_score | -0.0444 | -0.0874 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0438 | -0.0438 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0175 | 0.0517 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.0263 | 0.3296 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.0913 | 0.3085 |
| controlled_alt_vs_proposed_raw | naturalness | -0.0099 | -0.0113 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 0.0212 | 0.1221 |
| controlled_alt_vs_proposed_raw | lore_consistency | 0.0132 | 0.3977 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_proposed_raw | objective_completion_support | 0.0033 | 0.0276 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | -0.0003 | -0.0010 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | -0.0056 | -0.0167 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.0366 | 0.4000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0021 | 0.0410 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1117 | 0.5710 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0095 | 0.0137 |
| controlled_alt_vs_proposed_raw | distinct1 | 0.0092 | 0.0098 |
| controlled_alt_vs_proposed_raw | length_score | -0.0722 | -0.1347 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0000 | 0.0000 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.0445 | 0.1426 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.0651 | 1.5901 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.0826 | 0.2712 |
| controlled_alt_vs_candidate_no_context | naturalness | -0.0082 | -0.0093 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 0.0543 | 0.3861 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 0.0358 | 3.3434 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_candidate_no_context | objective_completion_support | 0.0082 | 0.0710 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 0.0132 | 0.0432 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | -0.0010 | -0.0031 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.0874 | 2.1473 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0129 | 0.3107 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.0879 | 0.4005 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0614 | 0.0952 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0044 | 0.0047 |
| controlled_alt_vs_candidate_no_context | length_score | -0.0569 | -0.1093 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.0146 | 0.0155 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.0588 | 0.1976 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0388 | (0.0157, 0.0665) | 0.0000 | 0.0388 | (0.0129, 0.0777) | 0.0000 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0087 | (-0.0709, 0.0433) | 0.5913 | -0.0087 | (-0.1252, 0.0343) | 0.6903 |
| proposed_vs_candidate_no_context | naturalness | 0.0017 | (-0.0083, 0.0121) | 0.3527 | 0.0017 | (-0.0007, 0.0087) | 0.2940 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.0331 | (0.0138, 0.0564) | 0.0000 | 0.0331 | (0.0116, 0.0714) | 0.0000 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0226 | (0.0007, 0.0510) | 0.0210 | 0.0226 | (-0.0039, 0.0500) | 0.0347 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | 0.0049 | (-0.0294, 0.0388) | 0.3990 | 0.0049 | (-0.0133, 0.0155) | 0.2603 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 0.0136 | (-0.0007, 0.0294) | 0.0333 | 0.0136 | (0.0080, 0.0225) | 0.0000 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | 0.0046 | (-0.0087, 0.0221) | 0.3010 | 0.0046 | (0.0001, 0.0090) | 0.0000 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0508 | (0.0211, 0.0862) | 0.0000 | 0.0508 | (0.0182, 0.0972) | 0.0000 |
| proposed_vs_candidate_no_context | context_overlap | 0.0107 | (0.0025, 0.0205) | 0.0050 | 0.0107 | (0.0005, 0.0322) | 0.0000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0238 | (-0.0964, 0.0371) | 0.7340 | -0.0238 | (-0.1667, 0.0429) | 0.7470 |
| proposed_vs_candidate_no_context | persona_style | 0.0519 | (-0.0076, 0.1250) | 0.0557 | 0.0519 | (0.0000, 0.1250) | 0.0417 |
| proposed_vs_candidate_no_context | distinct1 | -0.0048 | (-0.0170, 0.0056) | 0.8107 | -0.0048 | (-0.0089, -0.0012) | 1.0000 |
| proposed_vs_candidate_no_context | length_score | 0.0153 | (-0.0403, 0.0694) | 0.2873 | 0.0153 | (-0.0267, 0.0611) | 0.2523 |
| proposed_vs_candidate_no_context | sentence_score | 0.0146 | (-0.0437, 0.0729) | 0.4067 | 0.0146 | (-0.0437, 0.0700) | 0.4087 |
| proposed_vs_candidate_no_context | overall_quality | 0.0143 | (-0.0096, 0.0380) | 0.1083 | 0.0143 | (-0.0101, 0.0270) | 0.0363 |
| controlled_vs_proposed_raw | context_relevance | 0.0239 | (-0.0146, 0.0679) | 0.1177 | 0.0239 | (-0.0183, 0.0411) | 0.0323 |
| controlled_vs_proposed_raw | persona_consistency | 0.0396 | (-0.0260, 0.1078) | 0.1207 | 0.0396 | (0.0083, 0.1283) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0043 | (-0.0125, 0.0219) | 0.3213 | 0.0043 | (-0.0231, 0.0213) | 0.2520 |
| controlled_vs_proposed_raw | quest_state_correctness | 0.0192 | (-0.0126, 0.0567) | 0.1320 | 0.0192 | (-0.0196, 0.0383) | 0.0360 |
| controlled_vs_proposed_raw | lore_consistency | 0.0088 | (-0.0254, 0.0506) | 0.3467 | 0.0088 | (-0.0399, 0.0382) | 0.3670 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0019 | (-0.0326, 0.0380) | 0.4560 | 0.0019 | (-0.0240, 0.0406) | 0.3780 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0060 | (-0.0057, 0.0198) | 0.1743 | 0.0060 | (-0.0015, 0.0189) | 0.2550 |
| controlled_vs_proposed_raw | time_pressure_acceptability | -0.0014 | (-0.0196, 0.0168) | 0.5653 | -0.0014 | (-0.0118, 0.0215) | 0.6253 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.0344 | (-0.0152, 0.0909) | 0.0947 | 0.0344 | (-0.0139, 0.0568) | 0.0347 |
| controlled_vs_proposed_raw | context_overlap | -0.0007 | (-0.0156, 0.0150) | 0.5223 | -0.0007 | (-0.0286, 0.0119) | 0.5933 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.0476 | (-0.0250, 0.1194) | 0.0970 | 0.0476 | (0.0000, 0.1667) | 0.0377 |
| controlled_vs_proposed_raw | persona_style | 0.0076 | (-0.0554, 0.0672) | 0.3893 | 0.0076 | (-0.0250, 0.0417) | 0.2937 |
| controlled_vs_proposed_raw | distinct1 | 0.0116 | (-0.0043, 0.0280) | 0.0780 | 0.0116 | (0.0082, 0.0158) | 0.0000 |
| controlled_vs_proposed_raw | length_score | -0.0278 | (-0.1125, 0.0597) | 0.7250 | -0.0278 | (-0.1611, 0.0542) | 0.7323 |
| controlled_vs_proposed_raw | sentence_score | 0.0437 | (0.0000, 0.0875) | 0.0453 | 0.0437 | (0.0350, 0.0583) | 0.0000 |
| controlled_vs_proposed_raw | overall_quality | 0.0270 | (-0.0014, 0.0561) | 0.0340 | 0.0270 | (0.0221, 0.0362) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.0627 | (0.0247, 0.1085) | 0.0000 | 0.0627 | (0.0483, 0.0831) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.0310 | (-0.0556, 0.1047) | 0.2117 | 0.0310 | (0.0032, 0.0457) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.0061 | (-0.0113, 0.0247) | 0.2637 | 0.0061 | (-0.0145, 0.0206) | 0.2580 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.0523 | (0.0217, 0.0892) | 0.0000 | 0.0523 | (0.0387, 0.0696) | 0.0000 |
| controlled_vs_candidate_no_context | lore_consistency | 0.0313 | (-0.0013, 0.0699) | 0.0333 | 0.0313 | (0.0101, 0.0436) | 0.0000 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | 0.0067 | (-0.0140, 0.0312) | 0.3230 | 0.0067 | (-0.0085, 0.0272) | 0.2730 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0196 | (0.0015, 0.0395) | 0.0187 | 0.0196 | (0.0083, 0.0327) | 0.0000 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 0.0031 | (-0.0111, 0.0173) | 0.3283 | 0.0031 | (-0.0033, 0.0216) | 0.2890 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.0852 | (0.0335, 0.1411) | 0.0000 | 0.0852 | (0.0636, 0.1136) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0100 | (0.0006, 0.0202) | 0.0177 | 0.0100 | (0.0036, 0.0124) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.0238 | (-0.0718, 0.0980) | 0.2787 | 0.0238 | (-0.0000, 0.0571) | 0.2897 |
| controlled_vs_candidate_no_context | persona_style | 0.0595 | (-0.0218, 0.1468) | 0.0810 | 0.0595 | (0.0000, 0.1667) | 0.0343 |
| controlled_vs_candidate_no_context | distinct1 | 0.0068 | (-0.0046, 0.0177) | 0.1157 | 0.0068 | (-0.0007, 0.0106) | 0.0380 |
| controlled_vs_candidate_no_context | length_score | -0.0125 | (-0.1000, 0.0722) | 0.6167 | -0.0125 | (-0.1000, 0.0875) | 0.7437 |
| controlled_vs_candidate_no_context | sentence_score | 0.0583 | (0.0146, 0.1167) | 0.0110 | 0.0583 | (0.0000, 0.1050) | 0.0357 |
| controlled_vs_candidate_no_context | overall_quality | 0.0413 | (0.0092, 0.0769) | 0.0067 | 0.0413 | (0.0261, 0.0531) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0024 | (-0.0356, 0.0390) | 0.4633 | 0.0024 | (-0.0230, 0.0529) | 0.4257 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0516 | (0.0071, 0.1099) | 0.0093 | 0.0516 | (0.0365, 0.0717) | 0.0000 |
| controlled_alt_vs_controlled_default | naturalness | -0.0142 | (-0.0337, 0.0040) | 0.9317 | -0.0142 | (-0.0250, -0.0029) | 1.0000 |
| controlled_alt_vs_controlled_default | quest_state_correctness | 0.0020 | (-0.0280, 0.0322) | 0.4350 | 0.0020 | (-0.0158, 0.0396) | 0.3977 |
| controlled_alt_vs_controlled_default | lore_consistency | 0.0044 | (-0.0472, 0.0533) | 0.4163 | 0.0044 | (-0.0257, 0.0733) | 0.4117 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | objective_completion_support | 0.0014 | (-0.0241, 0.0293) | 0.4517 | 0.0014 | (-0.0314, 0.0184) | 0.3910 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | -0.0064 | (-0.0177, 0.0033) | 0.8970 | -0.0064 | (-0.0094, 0.0015) | 0.9583 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | -0.0042 | (-0.0220, 0.0153) | 0.6597 | -0.0042 | (-0.0289, 0.0071) | 0.7123 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0022 | (-0.0480, 0.0518) | 0.4690 | 0.0022 | (-0.0341, 0.0694) | 0.4020 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0028 | (-0.0109, 0.0173) | 0.3493 | 0.0028 | (-0.0041, 0.0143) | 0.1570 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0641 | (0.0073, 0.1308) | 0.0073 | 0.0641 | (0.0278, 0.1000) | 0.0000 |
| controlled_alt_vs_controlled_default | persona_style | 0.0019 | (-0.0458, 0.0536) | 0.4877 | 0.0019 | (-0.0417, 0.0714) | 0.4197 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0024 | (-0.0152, 0.0096) | 0.6337 | -0.0024 | (-0.0175, 0.0034) | 0.7097 |
| controlled_alt_vs_controlled_default | length_score | -0.0444 | (-0.1403, 0.0444) | 0.8330 | -0.0444 | (-0.0967, 0.0500) | 0.8503 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0437 | (-0.0875, 0.0000) | 1.0000 | -0.0437 | (-0.0700, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0175 | (-0.0074, 0.0446) | 0.0887 | 0.0175 | (0.0084, 0.0367) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.0263 | (-0.0111, 0.0626) | 0.0810 | 0.0263 | (0.0181, 0.0346) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.0913 | (0.0087, 0.1898) | 0.0120 | 0.0913 | (0.0561, 0.1648) | 0.0000 |
| controlled_alt_vs_proposed_raw | naturalness | -0.0099 | (-0.0236, 0.0038) | 0.9130 | -0.0099 | (-0.0260, 0.0120) | 0.7457 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 0.0212 | (-0.0094, 0.0539) | 0.0870 | 0.0212 | (0.0200, 0.0225) | 0.0000 |
| controlled_alt_vs_proposed_raw | lore_consistency | 0.0132 | (-0.0205, 0.0485) | 0.2207 | 0.0132 | (-0.0011, 0.0334) | 0.0383 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_proposed_raw | objective_completion_support | 0.0033 | (-0.0245, 0.0334) | 0.4323 | 0.0033 | (-0.0056, 0.0100) | 0.2607 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | -0.0003 | (-0.0135, 0.0154) | 0.5183 | -0.0003 | (-0.0084, 0.0095) | 0.5837 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | -0.0056 | (-0.0220, 0.0118) | 0.7473 | -0.0056 | (-0.0101, 0.0014) | 0.9637 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.0366 | (-0.0107, 0.0868) | 0.0613 | 0.0366 | (0.0227, 0.0556) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0021 | (-0.0103, 0.0148) | 0.3757 | 0.0021 | (-0.0143, 0.0078) | 0.2613 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1117 | (0.0234, 0.2173) | 0.0063 | 0.1117 | (0.0714, 0.1944) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0095 | (-0.0698, 0.0756) | 0.3563 | 0.0095 | (-0.0050, 0.0464) | 0.2607 |
| controlled_alt_vs_proposed_raw | distinct1 | 0.0092 | (-0.0059, 0.0226) | 0.1137 | 0.0092 | (-0.0094, 0.0192) | 0.1520 |
| controlled_alt_vs_proposed_raw | length_score | -0.0722 | (-0.1445, 0.0069) | 0.9677 | -0.0722 | (-0.1111, 0.0042) | 0.9663 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0000 | (-0.0437, 0.0437) | 0.6510 | 0.0000 | (-0.0350, 0.0437) | 0.6177 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.0445 | (0.0050, 0.0884) | 0.0137 | 0.0445 | (0.0305, 0.0728) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.0651 | (0.0308, 0.1027) | 0.0000 | 0.0651 | (0.0407, 0.1123) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.0826 | (-0.0198, 0.1826) | 0.0537 | 0.0826 | (0.0397, 0.1050) | 0.0000 |
| controlled_alt_vs_candidate_no_context | naturalness | -0.0082 | (-0.0252, 0.0086) | 0.8230 | -0.0082 | (-0.0182, 0.0113) | 0.8520 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 0.0543 | (0.0262, 0.0849) | 0.0000 | 0.0543 | (0.0325, 0.0913) | 0.0000 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 0.0358 | (0.0110, 0.0631) | 0.0007 | 0.0358 | (0.0087, 0.0834) | 0.0000 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_candidate_no_context | objective_completion_support | 0.0082 | (-0.0228, 0.0403) | 0.3167 | 0.0082 | (-0.0042, 0.0152) | 0.0423 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 0.0132 | (-0.0048, 0.0311) | 0.0797 | 0.0132 | (-0.0003, 0.0233) | 0.0353 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | -0.0010 | (-0.0194, 0.0179) | 0.5607 | -0.0010 | (-0.0073, 0.0038) | 0.6313 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.0874 | (0.0432, 0.1345) | 0.0000 | 0.0874 | (0.0545, 0.1528) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0129 | (0.0030, 0.0243) | 0.0023 | 0.0129 | (0.0083, 0.0179) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.0879 | (-0.0258, 0.2002) | 0.0660 | 0.0879 | (0.0278, 0.1143) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0614 | (-0.0021, 0.1467) | 0.0423 | 0.0614 | (-0.0050, 0.1250) | 0.0353 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0044 | (-0.0088, 0.0173) | 0.2513 | 0.0044 | (-0.0183, 0.0140) | 0.2697 |
| controlled_alt_vs_candidate_no_context | length_score | -0.0569 | (-0.1347, 0.0194) | 0.9300 | -0.0569 | (-0.1367, 0.0375) | 0.8487 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.0146 | (-0.0437, 0.0729) | 0.4220 | 0.0146 | (0.0000, 0.0350) | 0.2967 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.0588 | (0.0147, 0.1053) | 0.0027 | 0.0588 | (0.0493, 0.0678) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 9 | 1 | 14 | 0.6667 | 0.9000 |
| proposed_vs_candidate_no_context | persona_consistency | 4 | 5 | 15 | 0.4792 | 0.4444 |
| proposed_vs_candidate_no_context | naturalness | 6 | 4 | 14 | 0.5417 | 0.6000 |
| proposed_vs_candidate_no_context | quest_state_correctness | 9 | 1 | 14 | 0.6667 | 0.9000 |
| proposed_vs_candidate_no_context | lore_consistency | 5 | 4 | 15 | 0.5208 | 0.5556 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| proposed_vs_candidate_no_context | objective_completion_support | 4 | 3 | 17 | 0.5208 | 0.5714 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 7 | 3 | 14 | 0.5833 | 0.7000 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | 4 | 3 | 17 | 0.5208 | 0.5714 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 8 | 0 | 16 | 0.6667 | 1.0000 |
| proposed_vs_candidate_no_context | context_overlap | 7 | 3 | 14 | 0.5833 | 0.7000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 3 | 4 | 17 | 0.4792 | 0.4286 |
| proposed_vs_candidate_no_context | persona_style | 3 | 1 | 20 | 0.5417 | 0.7500 |
| proposed_vs_candidate_no_context | distinct1 | 4 | 5 | 15 | 0.4792 | 0.4444 |
| proposed_vs_candidate_no_context | length_score | 5 | 5 | 14 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | sentence_score | 3 | 2 | 19 | 0.5208 | 0.6000 |
| proposed_vs_candidate_no_context | overall_quality | 9 | 1 | 14 | 0.6667 | 0.9000 |
| controlled_vs_proposed_raw | context_relevance | 14 | 10 | 0 | 0.5833 | 0.5833 |
| controlled_vs_proposed_raw | persona_consistency | 7 | 3 | 14 | 0.5833 | 0.7000 |
| controlled_vs_proposed_raw | naturalness | 11 | 13 | 0 | 0.4583 | 0.4583 |
| controlled_vs_proposed_raw | quest_state_correctness | 14 | 10 | 0 | 0.5833 | 0.5833 |
| controlled_vs_proposed_raw | lore_consistency | 4 | 5 | 15 | 0.4792 | 0.4444 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_vs_proposed_raw | objective_completion_support | 5 | 10 | 9 | 0.3958 | 0.3333 |
| controlled_vs_proposed_raw | gameplay_usefulness | 13 | 11 | 0 | 0.5417 | 0.5417 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 7 | 11 | 6 | 0.4167 | 0.3889 |
| controlled_vs_proposed_raw | context_keyword_coverage | 6 | 4 | 14 | 0.5417 | 0.6000 |
| controlled_vs_proposed_raw | context_overlap | 12 | 12 | 0 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 6 | 3 | 15 | 0.5625 | 0.6667 |
| controlled_vs_proposed_raw | persona_style | 4 | 2 | 18 | 0.5417 | 0.6667 |
| controlled_vs_proposed_raw | distinct1 | 16 | 6 | 2 | 0.7083 | 0.7273 |
| controlled_vs_proposed_raw | length_score | 11 | 13 | 0 | 0.4583 | 0.4583 |
| controlled_vs_proposed_raw | sentence_score | 3 | 0 | 21 | 0.5625 | 1.0000 |
| controlled_vs_proposed_raw | overall_quality | 14 | 10 | 0 | 0.5833 | 0.5833 |
| controlled_vs_candidate_no_context | context_relevance | 18 | 6 | 0 | 0.7500 | 0.7500 |
| controlled_vs_candidate_no_context | persona_consistency | 8 | 2 | 14 | 0.6250 | 0.8000 |
| controlled_vs_candidate_no_context | naturalness | 12 | 12 | 0 | 0.5000 | 0.5000 |
| controlled_vs_candidate_no_context | quest_state_correctness | 18 | 6 | 0 | 0.7500 | 0.7500 |
| controlled_vs_candidate_no_context | lore_consistency | 6 | 4 | 14 | 0.5417 | 0.6000 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_vs_candidate_no_context | objective_completion_support | 6 | 10 | 8 | 0.4167 | 0.3750 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 15 | 9 | 0 | 0.6250 | 0.6250 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 8 | 11 | 5 | 0.4375 | 0.4211 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 10 | 1 | 13 | 0.6875 | 0.9091 |
| controlled_vs_candidate_no_context | context_overlap | 17 | 7 | 0 | 0.7083 | 0.7083 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 7 | 2 | 15 | 0.6042 | 0.7778 |
| controlled_vs_candidate_no_context | persona_style | 4 | 1 | 19 | 0.5625 | 0.8000 |
| controlled_vs_candidate_no_context | distinct1 | 16 | 6 | 2 | 0.7083 | 0.7273 |
| controlled_vs_candidate_no_context | length_score | 11 | 13 | 0 | 0.4583 | 0.4583 |
| controlled_vs_candidate_no_context | sentence_score | 4 | 0 | 20 | 0.5833 | 1.0000 |
| controlled_vs_candidate_no_context | overall_quality | 15 | 9 | 0 | 0.6250 | 0.6250 |
| controlled_alt_vs_controlled_default | context_relevance | 4 | 8 | 12 | 0.4167 | 0.3333 |
| controlled_alt_vs_controlled_default | persona_consistency | 7 | 2 | 15 | 0.6042 | 0.7778 |
| controlled_alt_vs_controlled_default | naturalness | 5 | 7 | 12 | 0.4583 | 0.4167 |
| controlled_alt_vs_controlled_default | quest_state_correctness | 4 | 8 | 12 | 0.4167 | 0.3333 |
| controlled_alt_vs_controlled_default | lore_consistency | 6 | 5 | 13 | 0.5208 | 0.5455 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_alt_vs_controlled_default | objective_completion_support | 5 | 5 | 14 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | 5 | 7 | 12 | 0.4583 | 0.4167 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | 5 | 7 | 12 | 0.4583 | 0.4167 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 4 | 6 | 14 | 0.4583 | 0.4000 |
| controlled_alt_vs_controlled_default | context_overlap | 6 | 6 | 12 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 6 | 1 | 17 | 0.6042 | 0.8571 |
| controlled_alt_vs_controlled_default | persona_style | 2 | 3 | 19 | 0.4792 | 0.4000 |
| controlled_alt_vs_controlled_default | distinct1 | 6 | 6 | 12 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | length_score | 6 | 6 | 12 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | sentence_score | 0 | 3 | 21 | 0.4375 | 0.0000 |
| controlled_alt_vs_controlled_default | overall_quality | 6 | 6 | 12 | 0.5000 | 0.5000 |
| controlled_alt_vs_proposed_raw | context_relevance | 16 | 8 | 0 | 0.6667 | 0.6667 |
| controlled_alt_vs_proposed_raw | persona_consistency | 8 | 2 | 14 | 0.6250 | 0.8000 |
| controlled_alt_vs_proposed_raw | naturalness | 10 | 14 | 0 | 0.4167 | 0.4167 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 16 | 8 | 0 | 0.6667 | 0.6667 |
| controlled_alt_vs_proposed_raw | lore_consistency | 7 | 4 | 13 | 0.5625 | 0.6364 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_alt_vs_proposed_raw | objective_completion_support | 6 | 12 | 6 | 0.3750 | 0.3333 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 10 | 14 | 0 | 0.4167 | 0.4167 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | 4 | 14 | 6 | 0.2917 | 0.2222 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 8 | 2 | 14 | 0.6250 | 0.8000 |
| controlled_alt_vs_proposed_raw | context_overlap | 14 | 10 | 0 | 0.5833 | 0.5833 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 7 | 2 | 15 | 0.6042 | 0.7778 |
| controlled_alt_vs_proposed_raw | persona_style | 4 | 2 | 18 | 0.5417 | 0.6667 |
| controlled_alt_vs_proposed_raw | distinct1 | 17 | 5 | 2 | 0.7500 | 0.7727 |
| controlled_alt_vs_proposed_raw | length_score | 8 | 16 | 0 | 0.3333 | 0.3333 |
| controlled_alt_vs_proposed_raw | sentence_score | 1 | 1 | 22 | 0.5000 | 0.5000 |
| controlled_alt_vs_proposed_raw | overall_quality | 14 | 10 | 0 | 0.5833 | 0.5833 |
| controlled_alt_vs_candidate_no_context | context_relevance | 17 | 6 | 1 | 0.7292 | 0.7391 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 10 | 1 | 13 | 0.6875 | 0.9091 |
| controlled_alt_vs_candidate_no_context | naturalness | 10 | 14 | 0 | 0.4167 | 0.4167 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 17 | 6 | 1 | 0.7292 | 0.7391 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 8 | 2 | 14 | 0.6250 | 0.8000 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_alt_vs_candidate_no_context | objective_completion_support | 6 | 9 | 9 | 0.4375 | 0.4000 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 14 | 10 | 0 | 0.5833 | 0.5833 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | 6 | 11 | 7 | 0.3958 | 0.3529 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 10 | 0 | 14 | 0.7083 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 16 | 7 | 1 | 0.6875 | 0.6957 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 9 | 1 | 14 | 0.6667 | 0.9000 |
| controlled_alt_vs_candidate_no_context | persona_style | 3 | 1 | 20 | 0.5417 | 0.7500 |
| controlled_alt_vs_candidate_no_context | distinct1 | 16 | 6 | 2 | 0.7083 | 0.7273 |
| controlled_alt_vs_candidate_no_context | length_score | 10 | 14 | 0 | 0.4167 | 0.4167 |
| controlled_alt_vs_candidate_no_context | sentence_score | 3 | 2 | 19 | 0.5208 | 0.6000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 16 | 8 | 0 | 0.6667 | 0.6667 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.1250 | 0.0833 | 0.9167 |
| proposed_contextual_controlled_tuned | 0.0000 | 0.0000 | 0.0833 | 0.1667 | 0.8333 |
| proposed_contextual | 0.0000 | 0.0000 | 0.6667 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.6667 | 0.0000 | 0.0000 |

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
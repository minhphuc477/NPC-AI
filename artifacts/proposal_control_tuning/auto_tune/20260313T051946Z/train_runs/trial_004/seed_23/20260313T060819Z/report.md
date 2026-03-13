# Proposal Alignment Evaluation Report

- Run ID: `20260313T060819Z`
- Generated: `2026-03-13T06:14:49.649879+00:00`
- Scenarios: `artifacts\proposal_control_tuning\auto_tune\20260313T051946Z\train_runs\trial_004\seed_23\20260313T060819Z\scenarios.jsonl`
- Scenario count: `24`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_tuned`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.1095 (0.0644, 0.1614) | 0.2643 (0.2005, 0.3406) | 0.8694 (0.8492, 0.8899) | 0.3159 (0.2779, 0.3572) | n/a |
| proposed_contextual_controlled_tuned | 0.1115 (0.0604, 0.1665) | 0.2271 (0.1898, 0.2681) | 0.8782 (0.8587, 0.8987) | 0.3041 (0.2728, 0.3370) | n/a |
| proposed_contextual | 0.0851 (0.0519, 0.1259) | 0.2285 (0.1725, 0.2899) | 0.8688 (0.8503, 0.8877) | 0.2902 (0.2602, 0.3222) | n/a |
| candidate_no_context | 0.0330 (0.0187, 0.0498) | 0.2555 (0.1979, 0.3200) | 0.8723 (0.8564, 0.8889) | 0.2768 (0.2498, 0.3063) | n/a |

## Game-facing Outcome Metrics (mean, 95% CI)
| Arm | Quest-state Correctness | Lore Consistency | Contradiction Safety | Objective Completion Support | Gameplay Usefulness | Time-pressure Acceptability |
|---|---:|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.1908 (0.1510, 0.2313) | 0.0489 (0.0162, 0.0890) | 1.0000 (1.0000, 1.0000) | 0.0793 (0.0529, 0.1047) | 0.3048 (0.2911, 0.3200) | 0.2901 (0.2708, 0.3094) |
| proposed_contextual_controlled_tuned | 0.1926 (0.1494, 0.2420) | 0.0462 (0.0159, 0.0821) | 1.0000 (1.0000, 1.0000) | 0.0803 (0.0555, 0.1044) | 0.3109 (0.2980, 0.3251) | 0.2928 (0.2763, 0.3091) |
| proposed_contextual | 0.1747 (0.1437, 0.2083) | 0.0285 (0.0095, 0.0509) | 1.0000 (1.0000, 1.0000) | 0.0633 (0.0374, 0.0894) | 0.2943 (0.2811, 0.3066) | 0.2901 (0.2754, 0.3039) |
| candidate_no_context | 0.1243 (0.1135, 0.1384) | 0.0056 (0.0011, 0.0116) | 1.0000 (1.0000, 1.0000) | 0.0880 (0.0573, 0.1193) | 0.2891 (0.2754, 0.3015) | 0.3003 (0.2850, 0.3163) |

- Multi-turn contradiction rate is reported as `1 - contradiction_safety` in row-level outputs.

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0521 | 1.5786 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0269 | -0.1054 |
| proposed_vs_candidate_no_context | naturalness | -0.0035 | -0.0040 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.0503 | 0.4050 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0229 | 4.0876 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | -0.0247 | -0.2802 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 0.0052 | 0.0180 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | -0.0102 | -0.0340 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0655 | 2.1098 |
| proposed_vs_candidate_no_context | context_overlap | 0.0208 | 0.5545 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0298 | -0.1579 |
| proposed_vs_candidate_no_context | persona_style | -0.0156 | -0.0299 |
| proposed_vs_candidate_no_context | distinct1 | -0.0125 | -0.0132 |
| proposed_vs_candidate_no_context | length_score | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | sentence_score | 0.0146 | 0.0157 |
| proposed_vs_candidate_no_context | overall_quality | 0.0134 | 0.0485 |
| controlled_vs_proposed_raw | context_relevance | 0.0244 | 0.2866 |
| controlled_vs_proposed_raw | persona_consistency | 0.0358 | 0.1566 |
| controlled_vs_proposed_raw | naturalness | 0.0007 | 0.0007 |
| controlled_vs_proposed_raw | quest_state_correctness | 0.0162 | 0.0925 |
| controlled_vs_proposed_raw | lore_consistency | 0.0204 | 0.7140 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0160 | 0.2522 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0105 | 0.0358 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 0.0000 | 0.0001 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.0371 | 0.3843 |
| controlled_vs_proposed_raw | context_overlap | -0.0053 | -0.0905 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.0516 | 0.3250 |
| controlled_vs_proposed_raw | persona_style | -0.0273 | -0.0538 |
| controlled_vs_proposed_raw | distinct1 | -0.0088 | -0.0094 |
| controlled_vs_proposed_raw | length_score | 0.0097 | 0.0190 |
| controlled_vs_proposed_raw | sentence_score | 0.0312 | 0.0332 |
| controlled_vs_proposed_raw | overall_quality | 0.0257 | 0.0886 |
| controlled_vs_candidate_no_context | context_relevance | 0.0765 | 2.3175 |
| controlled_vs_candidate_no_context | persona_consistency | 0.0089 | 0.0347 |
| controlled_vs_candidate_no_context | naturalness | -0.0029 | -0.0033 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.0665 | 0.5351 |
| controlled_vs_candidate_no_context | lore_consistency | 0.0433 | 7.7199 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | -0.0087 | -0.0987 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0157 | 0.0545 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | -0.0102 | -0.0339 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.1027 | 3.3049 |
| controlled_vs_candidate_no_context | context_overlap | 0.0156 | 0.4139 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.0218 | 0.1158 |
| controlled_vs_candidate_no_context | persona_style | -0.0430 | -0.0821 |
| controlled_vs_candidate_no_context | distinct1 | -0.0213 | -0.0225 |
| controlled_vs_candidate_no_context | length_score | 0.0097 | 0.0190 |
| controlled_vs_candidate_no_context | sentence_score | 0.0458 | 0.0494 |
| controlled_vs_candidate_no_context | overall_quality | 0.0392 | 0.1415 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0020 | 0.0181 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0373 | -0.1410 |
| controlled_alt_vs_controlled_default | naturalness | 0.0087 | 0.0100 |
| controlled_alt_vs_controlled_default | quest_state_correctness | 0.0018 | 0.0095 |
| controlled_alt_vs_controlled_default | lore_consistency | -0.0027 | -0.0550 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_controlled_default | objective_completion_support | 0.0010 | 0.0131 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | 0.0061 | 0.0199 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | 0.0027 | 0.0094 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0023 | -0.0170 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0119 | 0.2238 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0492 | -0.2340 |
| controlled_alt_vs_controlled_default | persona_style | 0.0104 | 0.0217 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0080 | 0.0087 |
| controlled_alt_vs_controlled_default | length_score | 0.0181 | 0.0346 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0125 | 0.0128 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0119 | -0.0375 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.0264 | 0.3098 |
| controlled_alt_vs_proposed_raw | persona_consistency | -0.0015 | -0.0065 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0094 | 0.0108 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 0.0180 | 0.1029 |
| controlled_alt_vs_proposed_raw | lore_consistency | 0.0177 | 0.6198 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_proposed_raw | objective_completion_support | 0.0170 | 0.2686 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 0.0166 | 0.0565 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | 0.0028 | 0.0095 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.0348 | 0.3608 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0066 | 0.1131 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.0024 | 0.0150 |
| controlled_alt_vs_proposed_raw | persona_style | -0.0169 | -0.0333 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0008 | -0.0008 |
| controlled_alt_vs_proposed_raw | length_score | 0.0278 | 0.0542 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0437 | 0.0465 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.0139 | 0.0478 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.0785 | 2.3774 |
| controlled_alt_vs_candidate_no_context | persona_consistency | -0.0284 | -0.1112 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0059 | 0.0067 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 0.0683 | 0.5497 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 0.0406 | 7.2406 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_candidate_no_context | objective_completion_support | -0.0076 | -0.0868 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 0.0218 | 0.0755 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | -0.0075 | -0.0248 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.1004 | 3.2317 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0275 | 0.7303 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | -0.0274 | -0.1453 |
| controlled_alt_vs_candidate_no_context | persona_style | -0.0326 | -0.0622 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0133 | -0.0141 |
| controlled_alt_vs_candidate_no_context | length_score | 0.0278 | 0.0542 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.0583 | 0.0629 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.0273 | 0.0986 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0521 | (0.0220, 0.0876) | 0.0000 | 0.0521 | (0.0046, 0.0861) | 0.0003 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0269 | (-0.0804, 0.0245) | 0.8287 | -0.0269 | (-0.0940, 0.0308) | 0.7983 |
| proposed_vs_candidate_no_context | naturalness | -0.0035 | (-0.0232, 0.0171) | 0.6520 | -0.0035 | (-0.0349, 0.0201) | 0.6033 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.0503 | (0.0217, 0.0814) | 0.0000 | 0.0503 | (0.0063, 0.0818) | 0.0003 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0229 | (0.0070, 0.0423) | 0.0000 | 0.0229 | (0.0000, 0.0506) | 0.0783 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | -0.0247 | (-0.0488, -0.0050) | 0.9907 | -0.0247 | (-0.0534, -0.0013) | 1.0000 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 0.0052 | (-0.0109, 0.0190) | 0.2363 | 0.0052 | (-0.0125, 0.0235) | 0.3330 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | -0.0102 | (-0.0244, 0.0012) | 0.9583 | -0.0102 | (-0.0207, 0.0005) | 0.9707 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0655 | (0.0227, 0.1121) | 0.0003 | 0.0655 | (0.0045, 0.1081) | 0.0103 |
| proposed_vs_candidate_no_context | context_overlap | 0.0208 | (0.0102, 0.0326) | 0.0000 | 0.0208 | (0.0046, 0.0333) | 0.0000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0298 | (-0.0992, 0.0347) | 0.7990 | -0.0298 | (-0.1073, 0.0417) | 0.7847 |
| proposed_vs_candidate_no_context | persona_style | -0.0156 | (-0.0417, 0.0000) | 1.0000 | -0.0156 | (-0.0398, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | distinct1 | -0.0125 | (-0.0273, 0.0013) | 0.9613 | -0.0125 | (-0.0306, -0.0008) | 0.9917 |
| proposed_vs_candidate_no_context | length_score | 0.0000 | (-0.0861, 0.0806) | 0.5163 | 0.0000 | (-0.1182, 0.1000) | 0.5577 |
| proposed_vs_candidate_no_context | sentence_score | 0.0146 | (-0.0437, 0.0729) | 0.3997 | 0.0146 | (-0.0292, 0.0625) | 0.3687 |
| proposed_vs_candidate_no_context | overall_quality | 0.0134 | (-0.0136, 0.0430) | 0.1650 | 0.0134 | (-0.0277, 0.0482) | 0.2370 |
| controlled_vs_proposed_raw | context_relevance | 0.0244 | (-0.0166, 0.0717) | 0.1430 | 0.0244 | (-0.0046, 0.0710) | 0.0790 |
| controlled_vs_proposed_raw | persona_consistency | 0.0358 | (-0.0322, 0.1170) | 0.1770 | 0.0358 | (-0.0565, 0.1060) | 0.2047 |
| controlled_vs_proposed_raw | naturalness | 0.0007 | (-0.0263, 0.0255) | 0.4713 | 0.0007 | (-0.0367, 0.0368) | 0.4293 |
| controlled_vs_proposed_raw | quest_state_correctness | 0.0162 | (-0.0186, 0.0579) | 0.2063 | 0.0162 | (-0.0092, 0.0500) | 0.1097 |
| controlled_vs_proposed_raw | lore_consistency | 0.0204 | (-0.0127, 0.0556) | 0.1283 | 0.0204 | (0.0001, 0.0407) | 0.0197 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0160 | (-0.0083, 0.0397) | 0.0923 | 0.0160 | (-0.0163, 0.0458) | 0.2230 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0105 | (-0.0053, 0.0283) | 0.1083 | 0.0105 | (-0.0068, 0.0313) | 0.1887 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 0.0000 | (-0.0198, 0.0178) | 0.4800 | 0.0000 | (-0.0150, 0.0137) | 0.4827 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.0371 | (-0.0167, 0.1015) | 0.1280 | 0.0371 | (-0.0050, 0.0950) | 0.0437 |
| controlled_vs_proposed_raw | context_overlap | -0.0053 | (-0.0180, 0.0080) | 0.7767 | -0.0053 | (-0.0178, 0.0089) | 0.7980 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.0516 | (-0.0317, 0.1538) | 0.1300 | 0.0516 | (-0.0556, 0.1374) | 0.1637 |
| controlled_vs_proposed_raw | persona_style | -0.0273 | (-0.0638, 0.0013) | 0.9720 | -0.0273 | (-0.0753, 0.0000) | 1.0000 |
| controlled_vs_proposed_raw | distinct1 | -0.0088 | (-0.0299, 0.0107) | 0.8070 | -0.0088 | (-0.0255, 0.0096) | 0.8003 |
| controlled_vs_proposed_raw | length_score | 0.0097 | (-0.1097, 0.1278) | 0.4443 | 0.0097 | (-0.1244, 0.1423) | 0.4137 |
| controlled_vs_proposed_raw | sentence_score | 0.0312 | (-0.0229, 0.0875) | 0.1010 | 0.0312 | (0.0000, 0.0808) | 0.0830 |
| controlled_vs_proposed_raw | overall_quality | 0.0257 | (-0.0128, 0.0673) | 0.1120 | 0.0257 | (-0.0090, 0.0618) | 0.0807 |
| controlled_vs_candidate_no_context | context_relevance | 0.0765 | (0.0300, 0.1279) | 0.0003 | 0.0765 | (0.0277, 0.1144) | 0.0007 |
| controlled_vs_candidate_no_context | persona_consistency | 0.0089 | (-0.0644, 0.0901) | 0.4163 | 0.0089 | (-0.0803, 0.1055) | 0.4313 |
| controlled_vs_candidate_no_context | naturalness | -0.0029 | (-0.0257, 0.0201) | 0.5930 | -0.0029 | (-0.0319, 0.0234) | 0.5783 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.0665 | (0.0251, 0.1101) | 0.0000 | 0.0665 | (0.0262, 0.1019) | 0.0000 |
| controlled_vs_candidate_no_context | lore_consistency | 0.0433 | (0.0072, 0.0834) | 0.0043 | 0.0433 | (0.0150, 0.0635) | 0.0000 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | -0.0087 | (-0.0357, 0.0122) | 0.7483 | -0.0087 | (-0.0400, 0.0111) | 0.7460 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0157 | (0.0016, 0.0333) | 0.0117 | 0.0157 | (-0.0019, 0.0302) | 0.0450 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | -0.0102 | (-0.0333, 0.0107) | 0.8297 | -0.0102 | (-0.0269, 0.0004) | 0.9703 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.1027 | (0.0417, 0.1742) | 0.0000 | 0.1027 | (0.0455, 0.1507) | 0.0003 |
| controlled_vs_candidate_no_context | context_overlap | 0.0156 | (0.0007, 0.0308) | 0.0200 | 0.0156 | (-0.0055, 0.0303) | 0.0563 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.0218 | (-0.0655, 0.1210) | 0.3400 | 0.0218 | (-0.0764, 0.1346) | 0.3560 |
| controlled_vs_candidate_no_context | persona_style | -0.0430 | (-0.0807, -0.0117) | 1.0000 | -0.0430 | (-0.1151, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | distinct1 | -0.0213 | (-0.0421, -0.0033) | 0.9910 | -0.0213 | (-0.0462, 0.0051) | 0.9317 |
| controlled_vs_candidate_no_context | length_score | 0.0097 | (-0.0694, 0.0944) | 0.4290 | 0.0097 | (-0.0909, 0.1107) | 0.4383 |
| controlled_vs_candidate_no_context | sentence_score | 0.0458 | (-0.0104, 0.1042) | 0.0490 | 0.0458 | (0.0159, 0.0712) | 0.0073 |
| controlled_vs_candidate_no_context | overall_quality | 0.0392 | (-0.0019, 0.0827) | 0.0313 | 0.0392 | (-0.0046, 0.0825) | 0.0427 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0020 | (-0.0747, 0.0791) | 0.4760 | 0.0020 | (-0.0468, 0.0629) | 0.4780 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0373 | (-0.1062, 0.0229) | 0.8787 | -0.0373 | (-0.0949, 0.0191) | 0.9203 |
| controlled_alt_vs_controlled_default | naturalness | 0.0087 | (-0.0169, 0.0333) | 0.2653 | 0.0087 | (-0.0055, 0.0256) | 0.1947 |
| controlled_alt_vs_controlled_default | quest_state_correctness | 0.0018 | (-0.0599, 0.0709) | 0.4947 | 0.0018 | (-0.0364, 0.0525) | 0.4807 |
| controlled_alt_vs_controlled_default | lore_consistency | -0.0027 | (-0.0579, 0.0533) | 0.5363 | -0.0027 | (-0.0359, 0.0418) | 0.5473 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | objective_completion_support | 0.0010 | (-0.0290, 0.0315) | 0.4640 | 0.0010 | (-0.0215, 0.0236) | 0.4493 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | 0.0061 | (-0.0147, 0.0260) | 0.2900 | 0.0061 | (-0.0094, 0.0235) | 0.2423 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | 0.0027 | (-0.0239, 0.0287) | 0.4143 | 0.0027 | (-0.0147, 0.0211) | 0.3993 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0023 | (-0.1045, 0.0947) | 0.5270 | -0.0023 | (-0.0661, 0.0736) | 0.5433 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0119 | (-0.0137, 0.0396) | 0.1943 | 0.0119 | (-0.0056, 0.0326) | 0.0883 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0492 | (-0.1320, 0.0252) | 0.9040 | -0.0492 | (-0.1186, 0.0182) | 0.9243 |
| controlled_alt_vs_controlled_default | persona_style | 0.0104 | (-0.0299, 0.0521) | 0.3413 | 0.0104 | (0.0000, 0.0341) | 0.3183 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0080 | (-0.0117, 0.0270) | 0.2103 | 0.0080 | (-0.0066, 0.0273) | 0.1743 |
| controlled_alt_vs_controlled_default | length_score | 0.0181 | (-0.0819, 0.1250) | 0.3633 | 0.0181 | (-0.0733, 0.1107) | 0.3963 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0125 | (-0.0437, 0.0813) | 0.4503 | 0.0125 | (0.0000, 0.0346) | 0.3307 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0119 | (-0.0554, 0.0306) | 0.6910 | -0.0119 | (-0.0492, 0.0311) | 0.7023 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.0264 | (-0.0322, 0.0854) | 0.1987 | 0.0264 | (-0.0113, 0.0811) | 0.1353 |
| controlled_alt_vs_proposed_raw | persona_consistency | -0.0015 | (-0.0426, 0.0442) | 0.5447 | -0.0015 | (-0.0678, 0.0438) | 0.5210 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0094 | (-0.0178, 0.0361) | 0.2343 | 0.0094 | (-0.0111, 0.0382) | 0.2920 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 0.0180 | (-0.0313, 0.0706) | 0.2430 | 0.0180 | (-0.0142, 0.0677) | 0.1863 |
| controlled_alt_vs_proposed_raw | lore_consistency | 0.0177 | (-0.0180, 0.0565) | 0.1750 | 0.0177 | (-0.0064, 0.0500) | 0.1113 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_proposed_raw | objective_completion_support | 0.0170 | (-0.0056, 0.0397) | 0.0823 | 0.0170 | (-0.0163, 0.0412) | 0.1420 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 0.0166 | (-0.0009, 0.0361) | 0.0347 | 0.0166 | (0.0034, 0.0278) | 0.0050 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | 0.0028 | (-0.0114, 0.0165) | 0.3423 | 0.0028 | (-0.0124, 0.0144) | 0.3530 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.0348 | (-0.0402, 0.1189) | 0.1933 | 0.0348 | (-0.0182, 0.1083) | 0.1573 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0066 | (-0.0114, 0.0279) | 0.2640 | 0.0066 | (-0.0074, 0.0219) | 0.1983 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.0024 | (-0.0486, 0.0593) | 0.4707 | 0.0024 | (-0.0823, 0.0548) | 0.4143 |
| controlled_alt_vs_proposed_raw | persona_style | -0.0169 | (-0.0625, 0.0286) | 0.7770 | -0.0169 | (-0.0682, 0.0117) | 0.7470 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0008 | (-0.0265, 0.0233) | 0.5347 | -0.0008 | (-0.0309, 0.0313) | 0.5040 |
| controlled_alt_vs_proposed_raw | length_score | 0.0278 | (-0.0736, 0.1348) | 0.3000 | 0.0278 | (-0.0717, 0.1500) | 0.3470 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0437 | (-0.0146, 0.1021) | 0.1247 | 0.0437 | (0.0000, 0.0875) | 0.0737 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.0139 | (-0.0167, 0.0456) | 0.2017 | 0.0139 | (0.0047, 0.0234) | 0.0007 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.0785 | (0.0350, 0.1271) | 0.0003 | 0.0785 | (0.0544, 0.1092) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | -0.0284 | (-0.0917, 0.0272) | 0.8377 | -0.0284 | (-0.1190, 0.0482) | 0.7487 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0059 | (-0.0229, 0.0342) | 0.3527 | 0.0059 | (-0.0278, 0.0311) | 0.3370 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 0.0683 | (0.0290, 0.1123) | 0.0000 | 0.0683 | (0.0438, 0.0966) | 0.0000 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 0.0406 | (0.0127, 0.0753) | 0.0000 | 0.0406 | (0.0179, 0.0659) | 0.0003 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_candidate_no_context | objective_completion_support | -0.0076 | (-0.0368, 0.0196) | 0.6930 | -0.0076 | (-0.0338, 0.0235) | 0.6980 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 0.0218 | (0.0055, 0.0396) | 0.0023 | 0.0218 | (-0.0045, 0.0421) | 0.0507 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | -0.0075 | (-0.0242, 0.0085) | 0.8207 | -0.0075 | (-0.0185, 0.0079) | 0.8213 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.1004 | (0.0436, 0.1644) | 0.0000 | 0.1004 | (0.0665, 0.1417) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0275 | (0.0081, 0.0473) | 0.0027 | 0.0275 | (0.0208, 0.0335) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | -0.0274 | (-0.0962, 0.0361) | 0.7850 | -0.0274 | (-0.1310, 0.0654) | 0.6717 |
| controlled_alt_vs_candidate_no_context | persona_style | -0.0326 | (-0.0768, 0.0065) | 0.9490 | -0.0326 | (-0.1037, 0.0000) | 1.0000 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0133 | (-0.0368, 0.0097) | 0.8620 | -0.0133 | (-0.0463, 0.0215) | 0.7513 |
| controlled_alt_vs_candidate_no_context | length_score | 0.0278 | (-0.0820, 0.1417) | 0.3170 | 0.0278 | (-0.1467, 0.1524) | 0.3757 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.0583 | (-0.0146, 0.1167) | 0.0687 | 0.0583 | (0.0159, 0.0942) | 0.0090 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.0273 | (-0.0055, 0.0583) | 0.0497 | 0.0273 | (-0.0200, 0.0610) | 0.1180 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 12 | 1 | 11 | 0.7292 | 0.9231 |
| proposed_vs_candidate_no_context | persona_consistency | 4 | 5 | 15 | 0.4792 | 0.4444 |
| proposed_vs_candidate_no_context | naturalness | 6 | 6 | 12 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | quest_state_correctness | 12 | 1 | 11 | 0.7292 | 0.9231 |
| proposed_vs_candidate_no_context | lore_consistency | 8 | 1 | 15 | 0.6458 | 0.8889 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| proposed_vs_candidate_no_context | objective_completion_support | 2 | 7 | 15 | 0.3958 | 0.2222 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 10 | 3 | 11 | 0.6458 | 0.7692 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | 3 | 7 | 14 | 0.4167 | 0.3000 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 9 | 1 | 14 | 0.6667 | 0.9000 |
| proposed_vs_candidate_no_context | context_overlap | 13 | 0 | 11 | 0.7708 | 1.0000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 4 | 5 | 15 | 0.4792 | 0.4444 |
| proposed_vs_candidate_no_context | persona_style | 0 | 2 | 22 | 0.4583 | 0.0000 |
| proposed_vs_candidate_no_context | distinct1 | 2 | 8 | 14 | 0.3750 | 0.2000 |
| proposed_vs_candidate_no_context | length_score | 5 | 6 | 13 | 0.4792 | 0.4545 |
| proposed_vs_candidate_no_context | sentence_score | 3 | 2 | 19 | 0.5208 | 0.6000 |
| proposed_vs_candidate_no_context | overall_quality | 8 | 5 | 11 | 0.5625 | 0.6154 |
| controlled_vs_proposed_raw | context_relevance | 10 | 8 | 6 | 0.5417 | 0.5556 |
| controlled_vs_proposed_raw | persona_consistency | 7 | 7 | 10 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | naturalness | 8 | 10 | 6 | 0.4583 | 0.4444 |
| controlled_vs_proposed_raw | quest_state_correctness | 10 | 8 | 6 | 0.5417 | 0.5556 |
| controlled_vs_proposed_raw | lore_consistency | 6 | 4 | 14 | 0.5417 | 0.6000 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_vs_proposed_raw | objective_completion_support | 6 | 1 | 17 | 0.6042 | 0.8571 |
| controlled_vs_proposed_raw | gameplay_usefulness | 9 | 9 | 6 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 6 | 4 | 14 | 0.5417 | 0.6000 |
| controlled_vs_proposed_raw | context_keyword_coverage | 6 | 5 | 13 | 0.5208 | 0.5455 |
| controlled_vs_proposed_raw | context_overlap | 8 | 10 | 6 | 0.4583 | 0.4444 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 7 | 5 | 12 | 0.5417 | 0.5833 |
| controlled_vs_proposed_raw | persona_style | 1 | 5 | 18 | 0.4167 | 0.1667 |
| controlled_vs_proposed_raw | distinct1 | 8 | 8 | 8 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | length_score | 8 | 10 | 6 | 0.4583 | 0.4444 |
| controlled_vs_proposed_raw | sentence_score | 3 | 1 | 20 | 0.5417 | 0.7500 |
| controlled_vs_proposed_raw | overall_quality | 10 | 8 | 6 | 0.5417 | 0.5556 |
| controlled_vs_candidate_no_context | context_relevance | 13 | 4 | 7 | 0.6875 | 0.7647 |
| controlled_vs_candidate_no_context | persona_consistency | 5 | 6 | 13 | 0.4792 | 0.4545 |
| controlled_vs_candidate_no_context | naturalness | 7 | 10 | 7 | 0.4375 | 0.4118 |
| controlled_vs_candidate_no_context | quest_state_correctness | 13 | 4 | 7 | 0.6875 | 0.7647 |
| controlled_vs_candidate_no_context | lore_consistency | 6 | 3 | 15 | 0.5625 | 0.6667 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_vs_candidate_no_context | objective_completion_support | 2 | 4 | 18 | 0.4583 | 0.3333 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 10 | 7 | 7 | 0.5625 | 0.5882 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 5 | 6 | 13 | 0.4792 | 0.4545 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 10 | 2 | 12 | 0.6667 | 0.8333 |
| controlled_vs_candidate_no_context | context_overlap | 13 | 4 | 7 | 0.6875 | 0.7647 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 5 | 5 | 14 | 0.5000 | 0.5000 |
| controlled_vs_candidate_no_context | persona_style | 0 | 6 | 18 | 0.3750 | 0.0000 |
| controlled_vs_candidate_no_context | distinct1 | 7 | 8 | 9 | 0.4792 | 0.4667 |
| controlled_vs_candidate_no_context | length_score | 7 | 10 | 7 | 0.4375 | 0.4118 |
| controlled_vs_candidate_no_context | sentence_score | 4 | 1 | 19 | 0.5625 | 0.8000 |
| controlled_vs_candidate_no_context | overall_quality | 11 | 6 | 7 | 0.6042 | 0.6471 |
| controlled_alt_vs_controlled_default | context_relevance | 7 | 9 | 8 | 0.4583 | 0.4375 |
| controlled_alt_vs_controlled_default | persona_consistency | 5 | 7 | 12 | 0.4583 | 0.4167 |
| controlled_alt_vs_controlled_default | naturalness | 10 | 6 | 8 | 0.5833 | 0.6250 |
| controlled_alt_vs_controlled_default | quest_state_correctness | 8 | 8 | 8 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | lore_consistency | 8 | 6 | 10 | 0.5417 | 0.5714 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_alt_vs_controlled_default | objective_completion_support | 5 | 6 | 13 | 0.4792 | 0.4545 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | 10 | 6 | 8 | 0.5833 | 0.6250 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | 7 | 6 | 11 | 0.5208 | 0.5385 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 7 | 7 | 10 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | context_overlap | 7 | 9 | 8 | 0.4583 | 0.4375 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 3 | 6 | 15 | 0.4375 | 0.3333 |
| controlled_alt_vs_controlled_default | persona_style | 3 | 2 | 19 | 0.5208 | 0.6000 |
| controlled_alt_vs_controlled_default | distinct1 | 11 | 5 | 8 | 0.6250 | 0.6875 |
| controlled_alt_vs_controlled_default | length_score | 8 | 8 | 8 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | sentence_score | 1 | 1 | 22 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | overall_quality | 7 | 9 | 8 | 0.4583 | 0.4375 |
| controlled_alt_vs_proposed_raw | context_relevance | 10 | 10 | 4 | 0.5000 | 0.5000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 6 | 10 | 8 | 0.4167 | 0.3750 |
| controlled_alt_vs_proposed_raw | naturalness | 12 | 8 | 4 | 0.5833 | 0.6000 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 10 | 10 | 4 | 0.5000 | 0.5000 |
| controlled_alt_vs_proposed_raw | lore_consistency | 7 | 5 | 12 | 0.5417 | 0.5833 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_alt_vs_proposed_raw | objective_completion_support | 7 | 4 | 13 | 0.5625 | 0.6364 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 11 | 9 | 4 | 0.5417 | 0.5500 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | 7 | 5 | 12 | 0.5417 | 0.5833 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 7 | 7 | 10 | 0.5000 | 0.5000 |
| controlled_alt_vs_proposed_raw | context_overlap | 10 | 10 | 4 | 0.5000 | 0.5000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 6 | 7 | 11 | 0.4792 | 0.4615 |
| controlled_alt_vs_proposed_raw | persona_style | 3 | 5 | 16 | 0.4583 | 0.3750 |
| controlled_alt_vs_proposed_raw | distinct1 | 12 | 8 | 4 | 0.5833 | 0.6000 |
| controlled_alt_vs_proposed_raw | length_score | 10 | 10 | 4 | 0.5000 | 0.5000 |
| controlled_alt_vs_proposed_raw | sentence_score | 4 | 1 | 19 | 0.5625 | 0.8000 |
| controlled_alt_vs_proposed_raw | overall_quality | 10 | 10 | 4 | 0.5000 | 0.5000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 14 | 6 | 4 | 0.6667 | 0.7000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 6 | 9 | 9 | 0.4375 | 0.4000 |
| controlled_alt_vs_candidate_no_context | naturalness | 12 | 8 | 4 | 0.5833 | 0.6000 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 14 | 6 | 4 | 0.6667 | 0.7000 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 8 | 2 | 14 | 0.6250 | 0.8000 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_alt_vs_candidate_no_context | objective_completion_support | 6 | 6 | 12 | 0.5000 | 0.5000 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 13 | 7 | 4 | 0.6250 | 0.6500 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | 6 | 8 | 10 | 0.4583 | 0.4286 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 10 | 1 | 13 | 0.6875 | 0.9091 |
| controlled_alt_vs_candidate_no_context | context_overlap | 13 | 7 | 4 | 0.6250 | 0.6500 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 6 | 7 | 11 | 0.4792 | 0.4615 |
| controlled_alt_vs_candidate_no_context | persona_style | 2 | 5 | 17 | 0.4375 | 0.2857 |
| controlled_alt_vs_candidate_no_context | distinct1 | 11 | 8 | 5 | 0.5625 | 0.5789 |
| controlled_alt_vs_candidate_no_context | length_score | 11 | 9 | 4 | 0.5417 | 0.5500 |
| controlled_alt_vs_candidate_no_context | sentence_score | 5 | 1 | 18 | 0.5833 | 0.8333 |
| controlled_alt_vs_candidate_no_context | overall_quality | 15 | 5 | 4 | 0.7083 | 0.7500 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.1250 | 0.1667 | 0.8333 |
| proposed_contextual_controlled_tuned | 0.0000 | 0.0000 | 0.1667 | 0.1667 | 0.8333 |
| proposed_contextual | 0.0000 | 0.0000 | 0.5833 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 |

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
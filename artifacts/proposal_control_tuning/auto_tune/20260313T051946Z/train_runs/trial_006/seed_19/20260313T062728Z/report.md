# Proposal Alignment Evaluation Report

- Run ID: `20260313T062728Z`
- Generated: `2026-03-13T06:33:34.678789+00:00`
- Scenarios: `artifacts\proposal_control_tuning\auto_tune\20260313T051946Z\train_runs\trial_006\seed_19\20260313T062728Z\scenarios.jsonl`
- Scenario count: `24`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_tuned`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.1002 (0.0593, 0.1436) | 0.2973 (0.2344, 0.3656) | 0.8736 (0.8563, 0.8915) | 0.3237 (0.2870, 0.3599) | n/a |
| proposed_contextual_controlled_tuned | 0.1772 (0.1203, 0.2372) | 0.2897 (0.2404, 0.3497) | 0.8634 (0.8324, 0.8917) | 0.3548 (0.3112, 0.3963) | n/a |
| proposed_contextual | 0.0733 (0.0398, 0.1156) | 0.2322 (0.1740, 0.2930) | 0.8706 (0.8513, 0.8892) | 0.2863 (0.2552, 0.3173) | n/a |
| candidate_no_context | 0.0351 (0.0185, 0.0552) | 0.2517 (0.1949, 0.3132) | 0.8811 (0.8661, 0.8979) | 0.2780 (0.2514, 0.3067) | n/a |

## Game-facing Outcome Metrics (mean, 95% CI)
| Arm | Quest-state Correctness | Lore Consistency | Contradiction Safety | Objective Completion Support | Gameplay Usefulness | Time-pressure Acceptability |
|---|---:|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.1879 (0.1496, 0.2304) | 0.0375 (0.0132, 0.0653) | 1.0000 (1.0000, 1.0000) | 0.0837 (0.0571, 0.1101) | 0.3063 (0.2907, 0.3226) | 0.2960 (0.2775, 0.3134) |
| proposed_contextual_controlled_tuned | 0.2435 (0.1959, 0.2980) | 0.0853 (0.0496, 0.1247) | 1.0000 (1.0000, 1.0000) | 0.0603 (0.0392, 0.0819) | 0.3100 (0.2880, 0.3327) | 0.2771 (0.2581, 0.2962) |
| proposed_contextual | 0.1596 (0.1309, 0.1939) | 0.0233 (0.0034, 0.0532) | 1.0000 (1.0000, 1.0000) | 0.0721 (0.0510, 0.0929) | 0.2903 (0.2739, 0.3062) | 0.2994 (0.2865, 0.3132) |
| candidate_no_context | 0.1279 (0.1151, 0.1449) | 0.0091 (0.0009, 0.0216) | 1.0000 (1.0000, 1.0000) | 0.0817 (0.0593, 0.1022) | 0.2925 (0.2825, 0.3029) | 0.3050 (0.2920, 0.3186) |

- Multi-turn contradiction rate is reported as `1 - contradiction_safety` in row-level outputs.

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0382 | 1.0884 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0196 | -0.0778 |
| proposed_vs_candidate_no_context | naturalness | -0.0105 | -0.0119 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.0317 | 0.2481 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0141 | 1.5456 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | -0.0096 | -0.1173 |
| proposed_vs_candidate_no_context | gameplay_usefulness | -0.0022 | -0.0075 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | -0.0056 | -0.0184 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0500 | 1.4505 |
| proposed_vs_candidate_no_context | context_overlap | 0.0108 | 0.2935 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0258 | -0.1421 |
| proposed_vs_candidate_no_context | persona_style | 0.0052 | 0.0098 |
| proposed_vs_candidate_no_context | distinct1 | 0.0105 | 0.0113 |
| proposed_vs_candidate_no_context | length_score | -0.0833 | -0.1460 |
| proposed_vs_candidate_no_context | sentence_score | 0.0146 | 0.0150 |
| proposed_vs_candidate_no_context | overall_quality | 0.0082 | 0.0296 |
| controlled_vs_proposed_raw | context_relevance | 0.0269 | 0.3664 |
| controlled_vs_proposed_raw | persona_consistency | 0.0652 | 0.2807 |
| controlled_vs_proposed_raw | naturalness | 0.0030 | 0.0034 |
| controlled_vs_proposed_raw | quest_state_correctness | 0.0283 | 0.1775 |
| controlled_vs_proposed_raw | lore_consistency | 0.0142 | 0.6119 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0116 | 0.1609 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0159 | 0.0548 |
| controlled_vs_proposed_raw | time_pressure_acceptability | -0.0035 | -0.0115 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.0345 | 0.4081 |
| controlled_vs_proposed_raw | context_overlap | 0.0092 | 0.1934 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.0802 | 0.5146 |
| controlled_vs_proposed_raw | persona_style | 0.0052 | 0.0097 |
| controlled_vs_proposed_raw | distinct1 | -0.0162 | -0.0173 |
| controlled_vs_proposed_raw | length_score | 0.0681 | 0.1396 |
| controlled_vs_proposed_raw | sentence_score | -0.0417 | -0.0423 |
| controlled_vs_proposed_raw | overall_quality | 0.0374 | 0.1306 |
| controlled_vs_candidate_no_context | context_relevance | 0.0651 | 1.8537 |
| controlled_vs_candidate_no_context | persona_consistency | 0.0456 | 0.1810 |
| controlled_vs_candidate_no_context | naturalness | -0.0075 | -0.0085 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.0600 | 0.4696 |
| controlled_vs_candidate_no_context | lore_consistency | 0.0284 | 3.1031 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | 0.0020 | 0.0247 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0137 | 0.0469 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | -0.0091 | -0.0297 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.0845 | 2.4505 |
| controlled_vs_candidate_no_context | context_overlap | 0.0199 | 0.5436 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.0544 | 0.2995 |
| controlled_vs_candidate_no_context | persona_style | 0.0104 | 0.0196 |
| controlled_vs_candidate_no_context | distinct1 | -0.0057 | -0.0061 |
| controlled_vs_candidate_no_context | length_score | -0.0153 | -0.0268 |
| controlled_vs_candidate_no_context | sentence_score | -0.0271 | -0.0279 |
| controlled_vs_candidate_no_context | overall_quality | 0.0456 | 0.1641 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0769 | 0.7677 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0076 | -0.0256 |
| controlled_alt_vs_controlled_default | naturalness | -0.0102 | -0.0117 |
| controlled_alt_vs_controlled_default | quest_state_correctness | 0.0555 | 0.2956 |
| controlled_alt_vs_controlled_default | lore_consistency | 0.0478 | 1.2744 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_controlled_default | objective_completion_support | -0.0233 | -0.2788 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | 0.0038 | 0.0123 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | -0.0188 | -0.0637 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.1004 | 0.8439 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0223 | 0.3938 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0040 | -0.0168 |
| controlled_alt_vs_controlled_default | persona_style | -0.0221 | -0.0408 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0052 | 0.0056 |
| controlled_alt_vs_controlled_default | length_score | -0.0542 | -0.0975 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0146 | 0.0155 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0312 | 0.0963 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1038 | 1.4155 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.0576 | 0.2480 |
| controlled_alt_vs_proposed_raw | naturalness | -0.0072 | -0.0083 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 0.0839 | 0.5255 |
| controlled_alt_vs_proposed_raw | lore_consistency | 0.0620 | 2.6660 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_proposed_raw | objective_completion_support | -0.0117 | -0.1628 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 0.0197 | 0.0678 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | -0.0223 | -0.0745 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.1348 | 1.5964 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0314 | 0.6633 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.0762 | 0.4892 |
| controlled_alt_vs_proposed_raw | persona_style | -0.0169 | -0.0315 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0110 | -0.0118 |
| controlled_alt_vs_proposed_raw | length_score | 0.0139 | 0.0285 |
| controlled_alt_vs_proposed_raw | sentence_score | -0.0271 | -0.0275 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.0686 | 0.2395 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.1421 | 4.0446 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.0380 | 0.1508 |
| controlled_alt_vs_candidate_no_context | naturalness | -0.0177 | -0.0201 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 0.1156 | 0.9040 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 0.0762 | 8.3320 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_candidate_no_context | objective_completion_support | -0.0213 | -0.2611 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 0.0175 | 0.0598 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | -0.0279 | -0.0915 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.1848 | 5.3626 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0422 | 1.1514 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.0504 | 0.2776 |
| controlled_alt_vs_candidate_no_context | persona_style | -0.0117 | -0.0220 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0005 | -0.0006 |
| controlled_alt_vs_candidate_no_context | length_score | -0.0694 | -0.1217 |
| controlled_alt_vs_candidate_no_context | sentence_score | -0.0125 | -0.0129 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.0768 | 0.2762 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0382 | (-0.0012, 0.0857) | 0.0300 | 0.0382 | (0.0099, 0.0666) | 0.0010 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0196 | (-0.0814, 0.0357) | 0.7300 | -0.0196 | (-0.0522, 0.0080) | 0.9190 |
| proposed_vs_candidate_no_context | naturalness | -0.0105 | (-0.0316, 0.0094) | 0.8370 | -0.0105 | (-0.0222, -0.0021) | 0.9887 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.0317 | (0.0016, 0.0677) | 0.0187 | 0.0317 | (0.0075, 0.0570) | 0.0003 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0141 | (-0.0091, 0.0464) | 0.1750 | 0.0141 | (-0.0022, 0.0303) | 0.0740 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | -0.0096 | (-0.0344, 0.0153) | 0.7873 | -0.0096 | (-0.0364, 0.0169) | 0.7557 |
| proposed_vs_candidate_no_context | gameplay_usefulness | -0.0022 | (-0.0171, 0.0125) | 0.6133 | -0.0022 | (-0.0156, 0.0063) | 0.6583 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | -0.0056 | (-0.0190, 0.0087) | 0.7820 | -0.0056 | (-0.0184, 0.0080) | 0.8030 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0500 | (0.0000, 0.1102) | 0.0240 | 0.0500 | (0.0152, 0.0876) | 0.0000 |
| proposed_vs_candidate_no_context | context_overlap | 0.0108 | (-0.0004, 0.0240) | 0.0297 | 0.0108 | (-0.0027, 0.0292) | 0.0853 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0258 | (-0.0992, 0.0427) | 0.7607 | -0.0258 | (-0.0639, 0.0076) | 0.9400 |
| proposed_vs_candidate_no_context | persona_style | 0.0052 | (-0.0234, 0.0286) | 0.3540 | 0.0052 | (-0.0341, 0.0402) | 0.4377 |
| proposed_vs_candidate_no_context | distinct1 | 0.0105 | (-0.0084, 0.0298) | 0.1530 | 0.0105 | (-0.0036, 0.0296) | 0.0953 |
| proposed_vs_candidate_no_context | length_score | -0.0833 | (-0.1778, 0.0111) | 0.9617 | -0.0833 | (-0.1783, -0.0107) | 0.9897 |
| proposed_vs_candidate_no_context | sentence_score | 0.0146 | (-0.0292, 0.0583) | 0.3847 | 0.0146 | (-0.0437, 0.0955) | 0.4333 |
| proposed_vs_candidate_no_context | overall_quality | 0.0082 | (-0.0207, 0.0344) | 0.2833 | 0.0082 | (-0.0033, 0.0191) | 0.1007 |
| controlled_vs_proposed_raw | context_relevance | 0.0269 | (-0.0187, 0.0755) | 0.1357 | 0.0269 | (-0.0270, 0.0656) | 0.1300 |
| controlled_vs_proposed_raw | persona_consistency | 0.0652 | (0.0198, 0.1126) | 0.0023 | 0.0652 | (0.0387, 0.1083) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0030 | (-0.0202, 0.0254) | 0.3830 | 0.0030 | (-0.0020, 0.0077) | 0.1380 |
| controlled_vs_proposed_raw | quest_state_correctness | 0.0283 | (-0.0143, 0.0717) | 0.0917 | 0.0283 | (-0.0194, 0.0690) | 0.1260 |
| controlled_vs_proposed_raw | lore_consistency | 0.0142 | (-0.0118, 0.0435) | 0.1553 | 0.0142 | (-0.0085, 0.0321) | 0.0927 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0116 | (-0.0124, 0.0376) | 0.1823 | 0.0116 | (-0.0078, 0.0288) | 0.0940 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0159 | (-0.0027, 0.0366) | 0.0533 | 0.0159 | (0.0034, 0.0219) | 0.0110 |
| controlled_vs_proposed_raw | time_pressure_acceptability | -0.0035 | (-0.0188, 0.0113) | 0.6717 | -0.0035 | (-0.0148, 0.0070) | 0.6933 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.0345 | (-0.0265, 0.0985) | 0.1430 | 0.0345 | (-0.0326, 0.0789) | 0.0967 |
| controlled_vs_proposed_raw | context_overlap | 0.0092 | (-0.0088, 0.0301) | 0.1650 | 0.0092 | (-0.0200, 0.0336) | 0.2120 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.0802 | (0.0206, 0.1405) | 0.0047 | 0.0802 | (0.0385, 0.1452) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.0052 | (-0.0404, 0.0534) | 0.4460 | 0.0052 | (-0.0341, 0.0402) | 0.4357 |
| controlled_vs_proposed_raw | distinct1 | -0.0162 | (-0.0318, -0.0007) | 0.9793 | -0.0162 | (-0.0381, 0.0007) | 0.9560 |
| controlled_vs_proposed_raw | length_score | 0.0681 | (-0.0403, 0.1750) | 0.1090 | 0.0681 | (0.0400, 0.1030) | 0.0000 |
| controlled_vs_proposed_raw | sentence_score | -0.0417 | (-0.1125, 0.0167) | 0.9127 | -0.0417 | (-0.1146, 0.0318) | 0.8607 |
| controlled_vs_proposed_raw | overall_quality | 0.0374 | (0.0051, 0.0713) | 0.0087 | 0.0374 | (0.0096, 0.0593) | 0.0090 |
| controlled_vs_candidate_no_context | context_relevance | 0.0651 | (0.0262, 0.1080) | 0.0000 | 0.0651 | (0.0320, 0.0952) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.0456 | (-0.0244, 0.1130) | 0.0917 | 0.0456 | (0.0068, 0.0815) | 0.0107 |
| controlled_vs_candidate_no_context | naturalness | -0.0075 | (-0.0251, 0.0113) | 0.7757 | -0.0075 | (-0.0208, 0.0026) | 0.9207 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.0600 | (0.0242, 0.0988) | 0.0003 | 0.0600 | (0.0259, 0.0916) | 0.0000 |
| controlled_vs_candidate_no_context | lore_consistency | 0.0284 | (0.0066, 0.0542) | 0.0043 | 0.0284 | (0.0104, 0.0435) | 0.0007 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | 0.0020 | (-0.0281, 0.0308) | 0.4567 | 0.0020 | (-0.0120, 0.0160) | 0.4027 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0137 | (-0.0015, 0.0304) | 0.0390 | 0.0137 | (-0.0001, 0.0248) | 0.0310 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | -0.0091 | (-0.0274, 0.0089) | 0.8453 | -0.0091 | (-0.0185, 0.0013) | 0.9473 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.0845 | (0.0356, 0.1417) | 0.0000 | 0.0845 | (0.0413, 0.1240) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0199 | (0.0052, 0.0372) | 0.0023 | 0.0199 | (0.0098, 0.0322) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.0544 | (-0.0270, 0.1425) | 0.1003 | 0.0544 | (0.0000, 0.1156) | 0.0277 |
| controlled_vs_candidate_no_context | persona_style | 0.0104 | (-0.0469, 0.0678) | 0.3853 | 0.0104 | (-0.0682, 0.0804) | 0.4503 |
| controlled_vs_candidate_no_context | distinct1 | -0.0057 | (-0.0259, 0.0156) | 0.7150 | -0.0057 | (-0.0164, 0.0065) | 0.8327 |
| controlled_vs_candidate_no_context | length_score | -0.0153 | (-0.0944, 0.0681) | 0.6647 | -0.0153 | (-0.0950, 0.0476) | 0.6860 |
| controlled_vs_candidate_no_context | sentence_score | -0.0271 | (-0.0958, 0.0292) | 0.8237 | -0.0271 | (-0.0750, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | overall_quality | 0.0456 | (0.0107, 0.0845) | 0.0047 | 0.0456 | (0.0212, 0.0665) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0769 | (-0.0031, 0.1577) | 0.0323 | 0.0769 | (0.0108, 0.1552) | 0.0063 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0076 | (-0.0887, 0.0751) | 0.5773 | -0.0076 | (-0.0536, 0.0333) | 0.6350 |
| controlled_alt_vs_controlled_default | naturalness | -0.0102 | (-0.0480, 0.0274) | 0.6877 | -0.0102 | (-0.0395, 0.0190) | 0.7337 |
| controlled_alt_vs_controlled_default | quest_state_correctness | 0.0555 | (-0.0129, 0.1263) | 0.0550 | 0.0555 | (-0.0050, 0.1336) | 0.0730 |
| controlled_alt_vs_controlled_default | lore_consistency | 0.0478 | (0.0059, 0.0921) | 0.0150 | 0.0478 | (0.0174, 0.0956) | 0.0000 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | objective_completion_support | -0.0233 | (-0.0547, 0.0072) | 0.9303 | -0.0233 | (-0.0430, -0.0033) | 0.9923 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | 0.0038 | (-0.0220, 0.0304) | 0.3900 | 0.0038 | (-0.0138, 0.0214) | 0.3390 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | -0.0188 | (-0.0400, 0.0043) | 0.9443 | -0.0188 | (-0.0284, -0.0091) | 1.0000 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.1004 | (-0.0004, 0.2034) | 0.0267 | 0.1004 | (0.0185, 0.1977) | 0.0043 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0223 | (-0.0068, 0.0571) | 0.0783 | 0.0223 | (-0.0073, 0.0518) | 0.0797 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0040 | (-0.1064, 0.0909) | 0.5287 | -0.0040 | (-0.0536, 0.0417) | 0.6440 |
| controlled_alt_vs_controlled_default | persona_style | -0.0221 | (-0.0612, 0.0156) | 0.9043 | -0.0221 | (-0.0443, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0052 | (-0.0224, 0.0297) | 0.3337 | 0.0052 | (-0.0323, 0.0257) | 0.2830 |
| controlled_alt_vs_controlled_default | length_score | -0.0542 | (-0.1889, 0.0639) | 0.7990 | -0.0542 | (-0.1488, 0.0449) | 0.8307 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0146 | (-0.0813, 0.1104) | 0.3993 | 0.0146 | (-0.0437, 0.0875) | 0.4460 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0312 | (-0.0225, 0.0861) | 0.1307 | 0.0312 | (-0.0026, 0.0793) | 0.0437 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1038 | (0.0392, 0.1660) | 0.0003 | 0.1038 | (0.0621, 0.1372) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.0576 | (-0.0077, 0.1336) | 0.0480 | 0.0576 | (0.0372, 0.0812) | 0.0000 |
| controlled_alt_vs_proposed_raw | naturalness | -0.0072 | (-0.0374, 0.0240) | 0.6797 | -0.0072 | (-0.0366, 0.0241) | 0.6737 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 0.0839 | (0.0320, 0.1413) | 0.0010 | 0.0839 | (0.0475, 0.1143) | 0.0000 |
| controlled_alt_vs_proposed_raw | lore_consistency | 0.0620 | (0.0252, 0.1013) | 0.0000 | 0.0620 | (0.0384, 0.0897) | 0.0000 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_proposed_raw | objective_completion_support | -0.0117 | (-0.0310, 0.0083) | 0.8693 | -0.0117 | (-0.0336, 0.0090) | 0.9317 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 0.0197 | (-0.0024, 0.0427) | 0.0420 | 0.0197 | (0.0064, 0.0330) | 0.0000 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | -0.0223 | (-0.0390, -0.0063) | 0.9970 | -0.0223 | (-0.0417, -0.0102) | 1.0000 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.1348 | (0.0530, 0.2159) | 0.0007 | 0.1348 | (0.0864, 0.1826) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0314 | (0.0044, 0.0644) | 0.0070 | 0.0314 | (0.0068, 0.0572) | 0.0067 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.0762 | (-0.0040, 0.1700) | 0.0320 | 0.0762 | (0.0474, 0.1091) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_style | -0.0169 | (-0.0521, 0.0117) | 0.8710 | -0.0169 | (-0.0682, 0.0100) | 0.7560 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0110 | (-0.0442, 0.0170) | 0.7457 | -0.0110 | (-0.0646, 0.0244) | 0.6710 |
| controlled_alt_vs_proposed_raw | length_score | 0.0139 | (-0.1111, 0.1389) | 0.4110 | 0.0139 | (-0.0569, 0.0897) | 0.3827 |
| controlled_alt_vs_proposed_raw | sentence_score | -0.0271 | (-0.0813, 0.0000) | 1.0000 | -0.0271 | (-0.0750, 0.0000) | 1.0000 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.0686 | (0.0268, 0.1131) | 0.0013 | 0.0686 | (0.0414, 0.0925) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.1421 | (0.0820, 0.2024) | 0.0000 | 0.1421 | (0.1082, 0.1968) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.0380 | (-0.0092, 0.0885) | 0.0577 | 0.0380 | (0.0104, 0.0675) | 0.0033 |
| controlled_alt_vs_candidate_no_context | naturalness | -0.0177 | (-0.0568, 0.0191) | 0.8143 | -0.0177 | (-0.0579, 0.0215) | 0.8330 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 0.1156 | (0.0668, 0.1722) | 0.0000 | 0.1156 | (0.0829, 0.1615) | 0.0000 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 0.0762 | (0.0386, 0.1156) | 0.0000 | 0.0762 | (0.0524, 0.1109) | 0.0000 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_candidate_no_context | objective_completion_support | -0.0213 | (-0.0439, 0.0008) | 0.9700 | -0.0213 | (-0.0384, -0.0043) | 0.9953 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 0.0175 | (-0.0002, 0.0384) | 0.0273 | 0.0175 | (0.0030, 0.0324) | 0.0050 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | -0.0279 | (-0.0425, -0.0135) | 1.0000 | -0.0279 | (-0.0382, -0.0194) | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.1848 | (0.1091, 0.2636) | 0.0000 | 0.1848 | (0.1422, 0.2537) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0422 | (0.0180, 0.0727) | 0.0000 | 0.0422 | (0.0218, 0.0625) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.0504 | (-0.0060, 0.1123) | 0.0453 | 0.0504 | (0.0188, 0.0905) | 0.0007 |
| controlled_alt_vs_candidate_no_context | persona_style | -0.0117 | (-0.0534, 0.0286) | 0.7017 | -0.0117 | (-0.1023, 0.0502) | 0.6287 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0005 | (-0.0313, 0.0276) | 0.4920 | -0.0005 | (-0.0379, 0.0256) | 0.5267 |
| controlled_alt_vs_candidate_no_context | length_score | -0.0694 | (-0.2028, 0.0639) | 0.8490 | -0.0694 | (-0.1917, 0.0560) | 0.8620 |
| controlled_alt_vs_candidate_no_context | sentence_score | -0.0125 | (-0.0833, 0.0583) | 0.6370 | -0.0125 | (-0.0833, 0.0875) | 0.6253 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.0768 | (0.0409, 0.1142) | 0.0000 | 0.0768 | (0.0543, 0.1008) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 10 | 6 | 8 | 0.5833 | 0.6250 |
| proposed_vs_candidate_no_context | persona_consistency | 6 | 6 | 12 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | naturalness | 6 | 10 | 8 | 0.4167 | 0.3750 |
| proposed_vs_candidate_no_context | quest_state_correctness | 10 | 6 | 8 | 0.5833 | 0.6250 |
| proposed_vs_candidate_no_context | lore_consistency | 5 | 3 | 16 | 0.5417 | 0.6250 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| proposed_vs_candidate_no_context | objective_completion_support | 6 | 9 | 9 | 0.4375 | 0.4000 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 8 | 8 | 8 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | 5 | 11 | 8 | 0.3750 | 0.3125 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 8 | 3 | 13 | 0.6042 | 0.7273 |
| proposed_vs_candidate_no_context | context_overlap | 11 | 5 | 8 | 0.6250 | 0.6875 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 4 | 6 | 14 | 0.4583 | 0.4000 |
| proposed_vs_candidate_no_context | persona_style | 4 | 1 | 19 | 0.5625 | 0.8000 |
| proposed_vs_candidate_no_context | distinct1 | 9 | 5 | 10 | 0.5833 | 0.6429 |
| proposed_vs_candidate_no_context | length_score | 5 | 11 | 8 | 0.3750 | 0.3125 |
| proposed_vs_candidate_no_context | sentence_score | 2 | 1 | 21 | 0.5208 | 0.6667 |
| proposed_vs_candidate_no_context | overall_quality | 10 | 6 | 8 | 0.5833 | 0.6250 |
| controlled_vs_proposed_raw | context_relevance | 9 | 8 | 7 | 0.5208 | 0.5294 |
| controlled_vs_proposed_raw | persona_consistency | 12 | 3 | 9 | 0.6875 | 0.8000 |
| controlled_vs_proposed_raw | naturalness | 8 | 9 | 7 | 0.4792 | 0.4706 |
| controlled_vs_proposed_raw | quest_state_correctness | 8 | 9 | 7 | 0.4792 | 0.4706 |
| controlled_vs_proposed_raw | lore_consistency | 6 | 4 | 14 | 0.5417 | 0.6000 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_vs_proposed_raw | objective_completion_support | 6 | 4 | 14 | 0.5417 | 0.6000 |
| controlled_vs_proposed_raw | gameplay_usefulness | 8 | 9 | 7 | 0.4792 | 0.4706 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 6 | 8 | 10 | 0.4583 | 0.4286 |
| controlled_vs_proposed_raw | context_keyword_coverage | 7 | 4 | 13 | 0.5625 | 0.6364 |
| controlled_vs_proposed_raw | context_overlap | 8 | 9 | 7 | 0.4792 | 0.4706 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 12 | 3 | 9 | 0.6875 | 0.8000 |
| controlled_vs_proposed_raw | persona_style | 3 | 3 | 18 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | distinct1 | 5 | 11 | 8 | 0.3750 | 0.3125 |
| controlled_vs_proposed_raw | length_score | 12 | 5 | 7 | 0.6458 | 0.7059 |
| controlled_vs_proposed_raw | sentence_score | 1 | 3 | 20 | 0.4583 | 0.2500 |
| controlled_vs_proposed_raw | overall_quality | 14 | 3 | 7 | 0.7292 | 0.8235 |
| controlled_vs_candidate_no_context | context_relevance | 13 | 5 | 6 | 0.6667 | 0.7222 |
| controlled_vs_candidate_no_context | persona_consistency | 9 | 4 | 11 | 0.6042 | 0.6923 |
| controlled_vs_candidate_no_context | naturalness | 7 | 11 | 6 | 0.4167 | 0.3889 |
| controlled_vs_candidate_no_context | quest_state_correctness | 13 | 5 | 6 | 0.6667 | 0.7222 |
| controlled_vs_candidate_no_context | lore_consistency | 7 | 1 | 16 | 0.6250 | 0.8750 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_vs_candidate_no_context | objective_completion_support | 7 | 8 | 9 | 0.4792 | 0.4667 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 13 | 5 | 6 | 0.6667 | 0.7222 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 5 | 11 | 8 | 0.3750 | 0.3125 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 11 | 2 | 11 | 0.6875 | 0.8462 |
| controlled_vs_candidate_no_context | context_overlap | 12 | 5 | 7 | 0.6458 | 0.7059 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 8 | 4 | 12 | 0.5833 | 0.6667 |
| controlled_vs_candidate_no_context | persona_style | 5 | 4 | 15 | 0.5208 | 0.5556 |
| controlled_vs_candidate_no_context | distinct1 | 8 | 10 | 6 | 0.4583 | 0.4444 |
| controlled_vs_candidate_no_context | length_score | 7 | 10 | 7 | 0.4375 | 0.4118 |
| controlled_vs_candidate_no_context | sentence_score | 1 | 2 | 21 | 0.4792 | 0.3333 |
| controlled_vs_candidate_no_context | overall_quality | 15 | 3 | 6 | 0.7500 | 0.8333 |
| controlled_alt_vs_controlled_default | context_relevance | 12 | 8 | 4 | 0.5833 | 0.6000 |
| controlled_alt_vs_controlled_default | persona_consistency | 6 | 8 | 10 | 0.4583 | 0.4286 |
| controlled_alt_vs_controlled_default | naturalness | 10 | 10 | 4 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | quest_state_correctness | 12 | 8 | 4 | 0.5833 | 0.6000 |
| controlled_alt_vs_controlled_default | lore_consistency | 10 | 5 | 9 | 0.6042 | 0.6667 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_alt_vs_controlled_default | objective_completion_support | 5 | 9 | 10 | 0.4167 | 0.3571 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | 9 | 11 | 4 | 0.4583 | 0.4500 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | 7 | 11 | 6 | 0.4167 | 0.3889 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 12 | 6 | 6 | 0.6250 | 0.6667 |
| controlled_alt_vs_controlled_default | context_overlap | 12 | 8 | 4 | 0.5833 | 0.6000 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 6 | 5 | 13 | 0.5208 | 0.5455 |
| controlled_alt_vs_controlled_default | persona_style | 1 | 4 | 19 | 0.4375 | 0.2000 |
| controlled_alt_vs_controlled_default | distinct1 | 13 | 7 | 4 | 0.6250 | 0.6500 |
| controlled_alt_vs_controlled_default | length_score | 7 | 11 | 6 | 0.4167 | 0.3889 |
| controlled_alt_vs_controlled_default | sentence_score | 3 | 2 | 19 | 0.5208 | 0.6000 |
| controlled_alt_vs_controlled_default | overall_quality | 12 | 8 | 4 | 0.5833 | 0.6000 |
| controlled_alt_vs_proposed_raw | context_relevance | 15 | 5 | 4 | 0.7083 | 0.7500 |
| controlled_alt_vs_proposed_raw | persona_consistency | 11 | 4 | 9 | 0.6458 | 0.7333 |
| controlled_alt_vs_proposed_raw | naturalness | 10 | 10 | 4 | 0.5000 | 0.5000 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 15 | 5 | 4 | 0.7083 | 0.7500 |
| controlled_alt_vs_proposed_raw | lore_consistency | 12 | 2 | 10 | 0.7083 | 0.8571 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_alt_vs_proposed_raw | objective_completion_support | 5 | 8 | 11 | 0.4375 | 0.3846 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 13 | 7 | 4 | 0.6250 | 0.6500 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | 4 | 12 | 8 | 0.3333 | 0.2500 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 15 | 4 | 5 | 0.7292 | 0.7895 |
| controlled_alt_vs_proposed_raw | context_overlap | 13 | 7 | 4 | 0.6250 | 0.6500 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 10 | 4 | 10 | 0.6250 | 0.7143 |
| controlled_alt_vs_proposed_raw | persona_style | 2 | 4 | 18 | 0.4583 | 0.3333 |
| controlled_alt_vs_proposed_raw | distinct1 | 9 | 11 | 4 | 0.4583 | 0.4500 |
| controlled_alt_vs_proposed_raw | length_score | 11 | 9 | 4 | 0.5417 | 0.5500 |
| controlled_alt_vs_proposed_raw | sentence_score | 0 | 1 | 23 | 0.4792 | 0.0000 |
| controlled_alt_vs_proposed_raw | overall_quality | 17 | 3 | 4 | 0.7917 | 0.8500 |
| controlled_alt_vs_candidate_no_context | context_relevance | 17 | 3 | 4 | 0.7917 | 0.8500 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 8 | 6 | 10 | 0.5417 | 0.5714 |
| controlled_alt_vs_candidate_no_context | naturalness | 8 | 12 | 4 | 0.4167 | 0.4000 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 17 | 3 | 4 | 0.7917 | 0.8500 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 12 | 1 | 11 | 0.7292 | 0.9231 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_alt_vs_candidate_no_context | objective_completion_support | 5 | 10 | 9 | 0.3958 | 0.3333 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 12 | 8 | 4 | 0.5833 | 0.6000 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | 3 | 14 | 7 | 0.2708 | 0.1765 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 15 | 1 | 8 | 0.7917 | 0.9375 |
| controlled_alt_vs_candidate_no_context | context_overlap | 15 | 5 | 4 | 0.7083 | 0.7500 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 8 | 4 | 12 | 0.5833 | 0.6667 |
| controlled_alt_vs_candidate_no_context | persona_style | 4 | 5 | 15 | 0.4792 | 0.4444 |
| controlled_alt_vs_candidate_no_context | distinct1 | 11 | 9 | 4 | 0.5417 | 0.5500 |
| controlled_alt_vs_candidate_no_context | length_score | 9 | 11 | 4 | 0.4583 | 0.4500 |
| controlled_alt_vs_candidate_no_context | sentence_score | 2 | 2 | 20 | 0.5000 | 0.5000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 17 | 3 | 4 | 0.7917 | 0.8500 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.0833 | 0.3750 | 0.6250 |
| proposed_contextual_controlled_tuned | 0.0000 | 0.0000 | 0.0000 | 0.2500 | 0.7500 |
| proposed_contextual | 0.0000 | 0.0000 | 0.4583 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.5417 | 0.0000 | 0.0000 |

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
| proposed_contextual_controlled | 0.0000 | 1.0000 | 0 | 5 |
| proposed_contextual_controlled_tuned | 0.0000 | 1.0000 | 0 | 5 |
| proposed_contextual | 0.0000 | 1.0000 | 0 | 5 |
| candidate_no_context | 0.0000 | 1.0000 | 0 | 5 |
- Detailed source-level values are published in `multi_turn_contradictions.json`.

- BERTScore status: unavailable (disabled_by_flag).

This report covers proposal RO5 metrics plus game-facing outcomes: quest-state correctness, lore consistency, contradiction safety, objective completion support, gameplay usefulness, and time-pressure acceptability.
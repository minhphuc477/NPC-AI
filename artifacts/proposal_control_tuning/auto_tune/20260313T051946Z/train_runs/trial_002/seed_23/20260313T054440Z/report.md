# Proposal Alignment Evaluation Report

- Run ID: `20260313T054440Z`
- Generated: `2026-03-13T05:51:50.096906+00:00`
- Scenarios: `artifacts\proposal_control_tuning\auto_tune\20260313T051946Z\train_runs\trial_002\seed_23\20260313T054440Z\scenarios.jsonl`
- Scenario count: `24`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_tuned`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0898 (0.0557, 0.1278) | 0.2516 (0.2015, 0.3031) | 0.8601 (0.8287, 0.8855) | 0.2999 (0.2688, 0.3296) | n/a |
| proposed_contextual_controlled_tuned | 0.1367 (0.0873, 0.1951) | 0.2397 (0.1771, 0.3083) | 0.8708 (0.8550, 0.8870) | 0.3192 (0.2825, 0.3608) | n/a |
| proposed_contextual | 0.0969 (0.0543, 0.1462) | 0.2270 (0.1703, 0.2905) | 0.8670 (0.8527, 0.8820) | 0.2946 (0.2601, 0.3288) | n/a |
| candidate_no_context | 0.0349 (0.0204, 0.0516) | 0.2446 (0.1804, 0.3169) | 0.8826 (0.8616, 0.9021) | 0.2756 (0.2474, 0.3064) | n/a |

## Game-facing Outcome Metrics (mean, 95% CI)
| Arm | Quest-state Correctness | Lore Consistency | Contradiction Safety | Objective Completion Support | Gameplay Usefulness | Time-pressure Acceptability |
|---|---:|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.1766 (0.1463, 0.2102) | 0.0299 (0.0096, 0.0572) | 1.0000 (1.0000, 1.0000) | 0.0860 (0.0613, 0.1097) | 0.2991 (0.2879, 0.3104) | 0.2929 (0.2775, 0.3081) |
| proposed_contextual_controlled_tuned | 0.2206 (0.1710, 0.2704) | 0.0699 (0.0318, 0.1148) | 1.0000 (1.0000, 1.0000) | 0.0841 (0.0552, 0.1117) | 0.3105 (0.2950, 0.3264) | 0.2904 (0.2728, 0.3088) |
| proposed_contextual | 0.1832 (0.1453, 0.2279) | 0.0413 (0.0116, 0.0779) | 1.0000 (1.0000, 1.0000) | 0.0658 (0.0413, 0.0911) | 0.2964 (0.2819, 0.3118) | 0.2850 (0.2706, 0.2998) |
| candidate_no_context | 0.1250 (0.1138, 0.1396) | 0.0060 (0.0013, 0.0121) | 1.0000 (1.0000, 1.0000) | 0.0708 (0.0458, 0.0959) | 0.2887 (0.2745, 0.3014) | 0.2969 (0.2844, 0.3108) |

- Multi-turn contradiction rate is reported as `1 - contradiction_safety` in row-level outputs.

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0620 | 1.7760 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0177 | -0.0722 |
| proposed_vs_candidate_no_context | naturalness | -0.0155 | -0.0176 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.0583 | 0.4662 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0353 | 5.8490 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | -0.0051 | -0.0716 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 0.0076 | 0.0265 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | -0.0120 | -0.0403 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0761 | 2.2333 |
| proposed_vs_candidate_no_context | context_overlap | 0.0290 | 0.7876 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0169 | -0.0971 |
| proposed_vs_candidate_no_context | persona_style | -0.0208 | -0.0394 |
| proposed_vs_candidate_no_context | distinct1 | -0.0128 | -0.0135 |
| proposed_vs_candidate_no_context | length_score | -0.0375 | -0.0685 |
| proposed_vs_candidate_no_context | sentence_score | -0.0292 | -0.0305 |
| proposed_vs_candidate_no_context | overall_quality | 0.0190 | 0.0689 |
| controlled_vs_proposed_raw | context_relevance | -0.0071 | -0.0730 |
| controlled_vs_proposed_raw | persona_consistency | 0.0247 | 0.1088 |
| controlled_vs_proposed_raw | naturalness | -0.0069 | -0.0080 |
| controlled_vs_proposed_raw | quest_state_correctness | -0.0066 | -0.0362 |
| controlled_vs_proposed_raw | lore_consistency | -0.0114 | -0.2769 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0203 | 0.3083 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0028 | 0.0093 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 0.0079 | 0.0278 |
| controlled_vs_proposed_raw | context_keyword_coverage | -0.0057 | -0.0515 |
| controlled_vs_proposed_raw | context_overlap | -0.0103 | -0.1566 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.0377 | 0.2405 |
| controlled_vs_proposed_raw | persona_style | -0.0273 | -0.0538 |
| controlled_vs_proposed_raw | distinct1 | -0.0155 | -0.0166 |
| controlled_vs_proposed_raw | length_score | -0.0097 | -0.0191 |
| controlled_vs_proposed_raw | sentence_score | 0.0583 | 0.0629 |
| controlled_vs_proposed_raw | overall_quality | 0.0053 | 0.0179 |
| controlled_vs_candidate_no_context | context_relevance | 0.0549 | 1.5735 |
| controlled_vs_candidate_no_context | persona_consistency | 0.0070 | 0.0287 |
| controlled_vs_candidate_no_context | naturalness | -0.0224 | -0.0254 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.0516 | 0.4131 |
| controlled_vs_candidate_no_context | lore_consistency | 0.0238 | 3.9522 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | 0.0152 | 0.2147 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0104 | 0.0360 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | -0.0041 | -0.0137 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.0705 | 2.0667 |
| controlled_vs_candidate_no_context | context_overlap | 0.0187 | 0.5076 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.0208 | 0.1200 |
| controlled_vs_candidate_no_context | persona_style | -0.0482 | -0.0911 |
| controlled_vs_candidate_no_context | distinct1 | -0.0283 | -0.0300 |
| controlled_vs_candidate_no_context | length_score | -0.0472 | -0.0863 |
| controlled_vs_candidate_no_context | sentence_score | 0.0292 | 0.0305 |
| controlled_vs_candidate_no_context | overall_quality | 0.0243 | 0.0881 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0469 | 0.5220 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0119 | -0.0473 |
| controlled_alt_vs_controlled_default | naturalness | 0.0107 | 0.0124 |
| controlled_alt_vs_controlled_default | quest_state_correctness | 0.0440 | 0.2494 |
| controlled_alt_vs_controlled_default | lore_consistency | 0.0400 | 1.3394 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_controlled_default | objective_completion_support | -0.0019 | -0.0226 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | 0.0113 | 0.0378 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | -0.0025 | -0.0085 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0595 | 0.5688 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0176 | 0.3164 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0149 | -0.0765 |
| controlled_alt_vs_controlled_default | persona_style | 0.0000 | 0.0000 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0297 | 0.0325 |
| controlled_alt_vs_controlled_default | length_score | -0.0292 | -0.0583 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0000 | 0.0000 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0193 | 0.0644 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.0398 | 0.4110 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.0128 | 0.0563 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0038 | 0.0043 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 0.0374 | 0.2042 |
| controlled_alt_vs_proposed_raw | lore_consistency | 0.0286 | 0.6915 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_proposed_raw | objective_completion_support | 0.0183 | 0.2788 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 0.0141 | 0.0475 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | 0.0054 | 0.0191 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.0538 | 0.4880 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0073 | 0.1102 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.0228 | 0.1456 |
| controlled_alt_vs_proposed_raw | persona_style | -0.0273 | -0.0538 |
| controlled_alt_vs_proposed_raw | distinct1 | 0.0142 | 0.0153 |
| controlled_alt_vs_proposed_raw | length_score | -0.0389 | -0.0763 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0583 | 0.0629 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.0246 | 0.0835 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.1018 | 2.9169 |
| controlled_alt_vs_candidate_no_context | persona_consistency | -0.0049 | -0.0199 |
| controlled_alt_vs_candidate_no_context | naturalness | -0.0118 | -0.0133 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 0.0957 | 0.7656 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 0.0638 | 10.5854 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_candidate_no_context | objective_completion_support | 0.0133 | 0.1873 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 0.0217 | 0.0752 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | -0.0065 | -0.0220 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.1299 | 3.8111 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0362 | 0.9846 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.0060 | 0.0343 |
| controlled_alt_vs_candidate_no_context | persona_style | -0.0482 | -0.0911 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0015 | 0.0016 |
| controlled_alt_vs_candidate_no_context | length_score | -0.0764 | -0.1396 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.0292 | 0.0305 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.0436 | 0.1582 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0620 | (0.0198, 0.1088) | 0.0010 | 0.0620 | (0.0021, 0.1381) | 0.0083 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0177 | (-0.0745, 0.0384) | 0.7167 | -0.0177 | (-0.0820, 0.0238) | 0.7813 |
| proposed_vs_candidate_no_context | naturalness | -0.0155 | (-0.0301, -0.0022) | 0.9910 | -0.0155 | (-0.0354, -0.0021) | 1.0000 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.0583 | (0.0205, 0.0999) | 0.0007 | 0.0583 | (0.0054, 0.1173) | 0.0113 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0353 | (0.0068, 0.0705) | 0.0003 | 0.0353 | (0.0017, 0.0877) | 0.0097 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | -0.0051 | (-0.0274, 0.0157) | 0.6567 | -0.0051 | (-0.0308, 0.0211) | 0.6310 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 0.0076 | (-0.0035, 0.0183) | 0.0867 | 0.0076 | (-0.0021, 0.0160) | 0.0737 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | -0.0120 | (-0.0289, 0.0031) | 0.9337 | -0.0120 | (-0.0319, 0.0064) | 0.8713 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0761 | (0.0227, 0.1367) | 0.0010 | 0.0761 | (0.0000, 0.1759) | 0.0733 |
| proposed_vs_candidate_no_context | context_overlap | 0.0290 | (0.0130, 0.0460) | 0.0000 | 0.0290 | (0.0071, 0.0498) | 0.0097 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0169 | (-0.0913, 0.0496) | 0.6877 | -0.0169 | (-0.0855, 0.0298) | 0.7353 |
| proposed_vs_candidate_no_context | persona_style | -0.0208 | (-0.0521, 0.0000) | 1.0000 | -0.0208 | (-0.0682, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | distinct1 | -0.0128 | (-0.0312, 0.0033) | 0.9273 | -0.0128 | (-0.0350, 0.0075) | 0.8783 |
| proposed_vs_candidate_no_context | length_score | -0.0375 | (-0.0959, 0.0181) | 0.9023 | -0.0375 | (-0.1100, 0.0056) | 0.9273 |
| proposed_vs_candidate_no_context | sentence_score | -0.0292 | (-0.0729, 0.0000) | 1.0000 | -0.0292 | (-0.0625, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | overall_quality | 0.0190 | (-0.0091, 0.0462) | 0.0940 | 0.0190 | (-0.0078, 0.0503) | 0.1263 |
| controlled_vs_proposed_raw | context_relevance | -0.0071 | (-0.0655, 0.0474) | 0.5897 | -0.0071 | (-0.0783, 0.0632) | 0.5630 |
| controlled_vs_proposed_raw | persona_consistency | 0.0247 | (-0.0388, 0.0890) | 0.2413 | 0.0247 | (-0.0874, 0.1048) | 0.2990 |
| controlled_vs_proposed_raw | naturalness | -0.0069 | (-0.0457, 0.0252) | 0.6377 | -0.0069 | (-0.0368, 0.0246) | 0.6280 |
| controlled_vs_proposed_raw | quest_state_correctness | -0.0066 | (-0.0628, 0.0426) | 0.5817 | -0.0066 | (-0.0695, 0.0562) | 0.5510 |
| controlled_vs_proposed_raw | lore_consistency | -0.0114 | (-0.0567, 0.0311) | 0.6877 | -0.0114 | (-0.0645, 0.0434) | 0.6097 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0203 | (-0.0011, 0.0456) | 0.0313 | 0.0203 | (0.0013, 0.0403) | 0.0160 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0028 | (-0.0154, 0.0205) | 0.3863 | 0.0028 | (-0.0153, 0.0221) | 0.3870 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 0.0079 | (-0.0073, 0.0240) | 0.1580 | 0.0079 | (-0.0003, 0.0203) | 0.0300 |
| controlled_vs_proposed_raw | context_keyword_coverage | -0.0057 | (-0.0852, 0.0686) | 0.5470 | -0.0057 | (-0.0924, 0.0856) | 0.5563 |
| controlled_vs_proposed_raw | context_overlap | -0.0103 | (-0.0289, 0.0074) | 0.8600 | -0.0103 | (-0.0314, 0.0108) | 0.7910 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.0377 | (-0.0367, 0.1181) | 0.1810 | 0.0377 | (-0.0929, 0.1310) | 0.2777 |
| controlled_vs_proposed_raw | persona_style | -0.0273 | (-0.0690, 0.0104) | 0.9253 | -0.0273 | (-0.1023, 0.0117) | 0.7490 |
| controlled_vs_proposed_raw | distinct1 | -0.0155 | (-0.0437, 0.0100) | 0.8580 | -0.0155 | (-0.0404, 0.0046) | 0.8820 |
| controlled_vs_proposed_raw | length_score | -0.0097 | (-0.1292, 0.1000) | 0.5567 | -0.0097 | (-0.1064, 0.1051) | 0.5793 |
| controlled_vs_proposed_raw | sentence_score | 0.0583 | (0.0000, 0.1313) | 0.0537 | 0.0583 | (0.0159, 0.0955) | 0.0107 |
| controlled_vs_proposed_raw | overall_quality | 0.0053 | (-0.0332, 0.0459) | 0.4087 | 0.0053 | (-0.0464, 0.0553) | 0.4483 |
| controlled_vs_candidate_no_context | context_relevance | 0.0549 | (0.0171, 0.0951) | 0.0007 | 0.0549 | (0.0233, 0.0789) | 0.0007 |
| controlled_vs_candidate_no_context | persona_consistency | 0.0070 | (-0.0728, 0.0829) | 0.4413 | 0.0070 | (-0.1431, 0.1197) | 0.4377 |
| controlled_vs_candidate_no_context | naturalness | -0.0224 | (-0.0623, 0.0151) | 0.8593 | -0.0224 | (-0.0517, 0.0108) | 0.9083 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.0516 | (0.0162, 0.0874) | 0.0010 | 0.0516 | (0.0211, 0.0759) | 0.0000 |
| controlled_vs_candidate_no_context | lore_consistency | 0.0238 | (0.0034, 0.0520) | 0.0033 | 0.0238 | (0.0057, 0.0473) | 0.0043 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | 0.0152 | (-0.0012, 0.0339) | 0.0363 | 0.0152 | (0.0062, 0.0242) | 0.0007 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0104 | (-0.0065, 0.0275) | 0.1087 | 0.0104 | (-0.0049, 0.0308) | 0.2557 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | -0.0041 | (-0.0221, 0.0098) | 0.6680 | -0.0041 | (-0.0254, 0.0124) | 0.6090 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.0705 | (0.0212, 0.1216) | 0.0023 | 0.0705 | (0.0289, 0.1021) | 0.0003 |
| controlled_vs_candidate_no_context | context_overlap | 0.0187 | (0.0045, 0.0333) | 0.0030 | 0.0187 | (0.0019, 0.0350) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.0208 | (-0.0685, 0.1082) | 0.3310 | 0.0208 | (-0.1500, 0.1429) | 0.4483 |
| controlled_vs_candidate_no_context | persona_style | -0.0482 | (-0.1068, 0.0013) | 0.9740 | -0.0482 | (-0.1705, 0.0117) | 0.7443 |
| controlled_vs_candidate_no_context | distinct1 | -0.0283 | (-0.0577, -0.0044) | 0.9913 | -0.0283 | (-0.0387, -0.0135) | 1.0000 |
| controlled_vs_candidate_no_context | length_score | -0.0472 | (-0.1848, 0.0917) | 0.7673 | -0.0472 | (-0.1708, 0.0949) | 0.7710 |
| controlled_vs_candidate_no_context | sentence_score | 0.0292 | (-0.0292, 0.0875) | 0.2260 | 0.0292 | (0.0000, 0.0500) | 0.0727 |
| controlled_vs_candidate_no_context | overall_quality | 0.0243 | (-0.0154, 0.0640) | 0.1180 | 0.0243 | (-0.0403, 0.0783) | 0.2663 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0469 | (-0.0102, 0.1042) | 0.0553 | 0.0469 | (-0.0212, 0.0983) | 0.0793 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0119 | (-0.0801, 0.0552) | 0.6707 | -0.0119 | (-0.0777, 0.0470) | 0.6180 |
| controlled_alt_vs_controlled_default | naturalness | 0.0107 | (-0.0158, 0.0409) | 0.2427 | 0.0107 | (0.0019, 0.0183) | 0.0123 |
| controlled_alt_vs_controlled_default | quest_state_correctness | 0.0440 | (-0.0085, 0.0988) | 0.0540 | 0.0440 | (-0.0128, 0.0914) | 0.0723 |
| controlled_alt_vs_controlled_default | lore_consistency | 0.0400 | (-0.0042, 0.0827) | 0.0340 | 0.0400 | (-0.0213, 0.0901) | 0.1417 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | objective_completion_support | -0.0019 | (-0.0258, 0.0222) | 0.5647 | -0.0019 | (-0.0061, 0.0003) | 0.9087 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | 0.0113 | (-0.0053, 0.0292) | 0.0940 | 0.0113 | (-0.0070, 0.0244) | 0.0783 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | -0.0025 | (-0.0250, 0.0217) | 0.5887 | -0.0025 | (-0.0133, 0.0083) | 0.6540 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0595 | (-0.0174, 0.1352) | 0.0623 | 0.0595 | (-0.0286, 0.1299) | 0.0700 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0176 | (-0.0039, 0.0399) | 0.0520 | 0.0176 | (-0.0018, 0.0343) | 0.0440 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0149 | (-0.0913, 0.0724) | 0.6570 | -0.0149 | (-0.0944, 0.0568) | 0.6183 |
| controlled_alt_vs_controlled_default | persona_style | 0.0000 | (-0.0391, 0.0391) | 0.5440 | 0.0000 | (-0.0312, 0.0312) | 0.6450 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0297 | (0.0067, 0.0585) | 0.0017 | 0.0297 | (0.0122, 0.0446) | 0.0000 |
| controlled_alt_vs_controlled_default | length_score | -0.0292 | (-0.1250, 0.0653) | 0.7217 | -0.0292 | (-0.0514, -0.0069) | 1.0000 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0000 | (-0.0437, 0.0437) | 0.6433 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0193 | (-0.0207, 0.0623) | 0.1753 | 0.0193 | (-0.0149, 0.0588) | 0.2533 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.0398 | (-0.0092, 0.0959) | 0.0590 | 0.0398 | (-0.0352, 0.1123) | 0.1873 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.0128 | (-0.0568, 0.0902) | 0.3783 | 0.0128 | (-0.0893, 0.1073) | 0.4377 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0038 | (-0.0196, 0.0270) | 0.3577 | 0.0038 | (-0.0227, 0.0336) | 0.4523 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 0.0374 | (-0.0077, 0.0864) | 0.0513 | 0.0374 | (-0.0285, 0.0974) | 0.1527 |
| controlled_alt_vs_proposed_raw | lore_consistency | 0.0286 | (-0.0081, 0.0654) | 0.0570 | 0.0286 | (0.0019, 0.0583) | 0.0143 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_proposed_raw | objective_completion_support | 0.0183 | (-0.0067, 0.0451) | 0.0810 | 0.0183 | (-0.0002, 0.0369) | 0.0283 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 0.0141 | (-0.0074, 0.0362) | 0.1037 | 0.0141 | (-0.0058, 0.0405) | 0.2413 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | 0.0054 | (-0.0139, 0.0267) | 0.3090 | 0.0054 | (-0.0033, 0.0138) | 0.0960 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.0538 | (-0.0144, 0.1250) | 0.0560 | 0.0538 | (-0.0529, 0.1469) | 0.1540 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0073 | (-0.0117, 0.0276) | 0.2457 | 0.0073 | (-0.0147, 0.0297) | 0.3147 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.0228 | (-0.0595, 0.1171) | 0.3150 | 0.0228 | (-0.0963, 0.1429) | 0.3753 |
| controlled_alt_vs_proposed_raw | persona_style | -0.0273 | (-0.0690, 0.0157) | 0.8977 | -0.0273 | (-0.0753, 0.0000) | 1.0000 |
| controlled_alt_vs_proposed_raw | distinct1 | 0.0142 | (-0.0013, 0.0316) | 0.0420 | 0.0142 | (-0.0036, 0.0323) | 0.0693 |
| controlled_alt_vs_proposed_raw | length_score | -0.0389 | (-0.1486, 0.0639) | 0.7610 | -0.0389 | (-0.1528, 0.0861) | 0.7257 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0583 | (-0.0146, 0.1313) | 0.0587 | 0.0583 | (0.0159, 0.0942) | 0.0087 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.0246 | (-0.0114, 0.0612) | 0.0977 | 0.0246 | (-0.0415, 0.0718) | 0.1993 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.1018 | (0.0498, 0.1581) | 0.0003 | 0.1018 | (0.0107, 0.1683) | 0.0003 |
| controlled_alt_vs_candidate_no_context | persona_consistency | -0.0049 | (-0.0806, 0.0684) | 0.5613 | -0.0049 | (-0.1329, 0.0952) | 0.5520 |
| controlled_alt_vs_candidate_no_context | naturalness | -0.0118 | (-0.0371, 0.0138) | 0.8203 | -0.0118 | (-0.0415, 0.0182) | 0.7590 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 0.0957 | (0.0481, 0.1434) | 0.0000 | 0.0957 | (0.0142, 0.1539) | 0.0003 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 0.0638 | (0.0291, 0.1032) | 0.0000 | 0.0638 | (0.0042, 0.1165) | 0.0103 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_candidate_no_context | objective_completion_support | 0.0133 | (-0.0029, 0.0319) | 0.0657 | 0.0133 | (0.0045, 0.0231) | 0.0003 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 0.0217 | (-0.0004, 0.0448) | 0.0280 | 0.0217 | (-0.0074, 0.0519) | 0.0943 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | -0.0065 | (-0.0272, 0.0144) | 0.7370 | -0.0065 | (-0.0358, 0.0168) | 0.6570 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.1299 | (0.0644, 0.1981) | 0.0000 | 0.1299 | (0.0136, 0.2140) | 0.0080 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0362 | (0.0128, 0.0589) | 0.0003 | 0.0362 | (-0.0000, 0.0621) | 0.0253 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.0060 | (-0.0784, 0.0903) | 0.4493 | 0.0060 | (-0.1524, 0.1278) | 0.4640 |
| controlled_alt_vs_candidate_no_context | persona_style | -0.0482 | (-0.1029, -0.0052) | 0.9887 | -0.0482 | (-0.1435, 0.0000) | 1.0000 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0015 | (-0.0114, 0.0139) | 0.4097 | 0.0015 | (-0.0106, 0.0159) | 0.4050 |
| controlled_alt_vs_candidate_no_context | length_score | -0.0764 | (-0.1792, 0.0403) | 0.9057 | -0.0764 | (-0.2194, 0.0885) | 0.8320 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.0292 | (-0.0292, 0.0875) | 0.2183 | 0.0292 | (0.0000, 0.0500) | 0.0760 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.0436 | (-0.0014, 0.0892) | 0.0287 | 0.0436 | (-0.0493, 0.1125) | 0.1570 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 14 | 1 | 9 | 0.7708 | 0.9333 |
| proposed_vs_candidate_no_context | persona_consistency | 4 | 5 | 15 | 0.4792 | 0.4444 |
| proposed_vs_candidate_no_context | naturalness | 6 | 9 | 9 | 0.4375 | 0.4000 |
| proposed_vs_candidate_no_context | quest_state_correctness | 14 | 1 | 9 | 0.7708 | 0.9333 |
| proposed_vs_candidate_no_context | lore_consistency | 8 | 1 | 15 | 0.6458 | 0.8889 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| proposed_vs_candidate_no_context | objective_completion_support | 5 | 5 | 14 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 10 | 5 | 9 | 0.6042 | 0.6667 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | 4 | 8 | 12 | 0.4167 | 0.3333 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 9 | 1 | 14 | 0.6667 | 0.9000 |
| proposed_vs_candidate_no_context | context_overlap | 14 | 1 | 9 | 0.7708 | 0.9333 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 4 | 5 | 15 | 0.4792 | 0.4444 |
| proposed_vs_candidate_no_context | persona_style | 0 | 2 | 22 | 0.4583 | 0.0000 |
| proposed_vs_candidate_no_context | distinct1 | 6 | 6 | 12 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | length_score | 5 | 9 | 10 | 0.4167 | 0.3571 |
| proposed_vs_candidate_no_context | sentence_score | 0 | 2 | 22 | 0.4583 | 0.0000 |
| proposed_vs_candidate_no_context | overall_quality | 12 | 3 | 9 | 0.6875 | 0.8000 |
| controlled_vs_proposed_raw | context_relevance | 11 | 8 | 5 | 0.5625 | 0.5789 |
| controlled_vs_proposed_raw | persona_consistency | 9 | 7 | 8 | 0.5417 | 0.5625 |
| controlled_vs_proposed_raw | naturalness | 10 | 9 | 5 | 0.5208 | 0.5263 |
| controlled_vs_proposed_raw | quest_state_correctness | 11 | 8 | 5 | 0.5625 | 0.5789 |
| controlled_vs_proposed_raw | lore_consistency | 6 | 6 | 12 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_vs_proposed_raw | objective_completion_support | 6 | 4 | 14 | 0.5417 | 0.6000 |
| controlled_vs_proposed_raw | gameplay_usefulness | 8 | 11 | 5 | 0.4375 | 0.4211 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 9 | 4 | 11 | 0.6042 | 0.6923 |
| controlled_vs_proposed_raw | context_keyword_coverage | 9 | 6 | 9 | 0.5625 | 0.6000 |
| controlled_vs_proposed_raw | context_overlap | 10 | 9 | 5 | 0.5208 | 0.5263 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 8 | 5 | 11 | 0.5625 | 0.6154 |
| controlled_vs_proposed_raw | persona_style | 2 | 5 | 17 | 0.4375 | 0.2857 |
| controlled_vs_proposed_raw | distinct1 | 8 | 11 | 5 | 0.4375 | 0.4211 |
| controlled_vs_proposed_raw | length_score | 8 | 11 | 5 | 0.4375 | 0.4211 |
| controlled_vs_proposed_raw | sentence_score | 5 | 1 | 18 | 0.5833 | 0.8333 |
| controlled_vs_proposed_raw | overall_quality | 10 | 9 | 5 | 0.5208 | 0.5263 |
| controlled_vs_candidate_no_context | context_relevance | 14 | 5 | 5 | 0.6875 | 0.7368 |
| controlled_vs_candidate_no_context | persona_consistency | 9 | 8 | 7 | 0.5208 | 0.5294 |
| controlled_vs_candidate_no_context | naturalness | 8 | 11 | 5 | 0.4375 | 0.4211 |
| controlled_vs_candidate_no_context | quest_state_correctness | 14 | 5 | 5 | 0.6875 | 0.7368 |
| controlled_vs_candidate_no_context | lore_consistency | 7 | 3 | 14 | 0.5833 | 0.7000 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_vs_candidate_no_context | objective_completion_support | 6 | 3 | 15 | 0.5625 | 0.6667 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 11 | 8 | 5 | 0.5625 | 0.5789 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 7 | 5 | 12 | 0.5417 | 0.5833 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 10 | 2 | 12 | 0.6667 | 0.8333 |
| controlled_vs_candidate_no_context | context_overlap | 14 | 5 | 5 | 0.6875 | 0.7368 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 8 | 7 | 9 | 0.5208 | 0.5333 |
| controlled_vs_candidate_no_context | persona_style | 2 | 6 | 16 | 0.4167 | 0.2500 |
| controlled_vs_candidate_no_context | distinct1 | 7 | 12 | 5 | 0.3958 | 0.3684 |
| controlled_vs_candidate_no_context | length_score | 8 | 11 | 5 | 0.4375 | 0.4211 |
| controlled_vs_candidate_no_context | sentence_score | 3 | 1 | 20 | 0.5417 | 0.7500 |
| controlled_vs_candidate_no_context | overall_quality | 11 | 8 | 5 | 0.5625 | 0.5789 |
| controlled_alt_vs_controlled_default | context_relevance | 9 | 6 | 9 | 0.5625 | 0.6000 |
| controlled_alt_vs_controlled_default | persona_consistency | 5 | 7 | 12 | 0.4583 | 0.4167 |
| controlled_alt_vs_controlled_default | naturalness | 8 | 7 | 9 | 0.5208 | 0.5333 |
| controlled_alt_vs_controlled_default | quest_state_correctness | 10 | 5 | 9 | 0.6042 | 0.6667 |
| controlled_alt_vs_controlled_default | lore_consistency | 12 | 2 | 10 | 0.7083 | 0.8571 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_alt_vs_controlled_default | objective_completion_support | 5 | 6 | 13 | 0.4792 | 0.4545 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | 9 | 6 | 9 | 0.5625 | 0.6000 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | 6 | 7 | 11 | 0.4792 | 0.4615 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 8 | 3 | 13 | 0.6042 | 0.7273 |
| controlled_alt_vs_controlled_default | context_overlap | 10 | 5 | 9 | 0.6042 | 0.6667 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 4 | 6 | 14 | 0.4583 | 0.4000 |
| controlled_alt_vs_controlled_default | persona_style | 2 | 2 | 20 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | distinct1 | 10 | 5 | 9 | 0.6042 | 0.6667 |
| controlled_alt_vs_controlled_default | length_score | 5 | 10 | 9 | 0.3958 | 0.3333 |
| controlled_alt_vs_controlled_default | sentence_score | 1 | 1 | 22 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | overall_quality | 9 | 6 | 9 | 0.5625 | 0.6000 |
| controlled_alt_vs_proposed_raw | context_relevance | 12 | 8 | 4 | 0.5833 | 0.6000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 5 | 7 | 12 | 0.4583 | 0.4167 |
| controlled_alt_vs_proposed_raw | naturalness | 11 | 9 | 4 | 0.5417 | 0.5500 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 12 | 8 | 4 | 0.5833 | 0.6000 |
| controlled_alt_vs_proposed_raw | lore_consistency | 10 | 3 | 11 | 0.6458 | 0.7692 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_alt_vs_proposed_raw | objective_completion_support | 8 | 4 | 12 | 0.5833 | 0.6667 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 11 | 9 | 4 | 0.5417 | 0.5500 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | 7 | 6 | 11 | 0.5208 | 0.5385 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 9 | 6 | 9 | 0.5625 | 0.6000 |
| controlled_alt_vs_proposed_raw | context_overlap | 10 | 10 | 4 | 0.5000 | 0.5000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 5 | 5 | 14 | 0.5000 | 0.5000 |
| controlled_alt_vs_proposed_raw | persona_style | 2 | 6 | 16 | 0.4167 | 0.2500 |
| controlled_alt_vs_proposed_raw | distinct1 | 12 | 8 | 4 | 0.5833 | 0.6000 |
| controlled_alt_vs_proposed_raw | length_score | 8 | 12 | 4 | 0.4167 | 0.4000 |
| controlled_alt_vs_proposed_raw | sentence_score | 5 | 1 | 18 | 0.5833 | 0.8333 |
| controlled_alt_vs_proposed_raw | overall_quality | 13 | 7 | 4 | 0.6250 | 0.6500 |
| controlled_alt_vs_candidate_no_context | context_relevance | 16 | 4 | 4 | 0.7500 | 0.8000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 7 | 7 | 10 | 0.5000 | 0.5000 |
| controlled_alt_vs_candidate_no_context | naturalness | 7 | 13 | 4 | 0.3750 | 0.3500 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 15 | 5 | 4 | 0.7083 | 0.7500 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 12 | 0 | 12 | 0.7500 | 1.0000 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_alt_vs_candidate_no_context | objective_completion_support | 7 | 5 | 12 | 0.5417 | 0.5833 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 11 | 9 | 4 | 0.5417 | 0.5500 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | 5 | 8 | 11 | 0.4375 | 0.3846 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 11 | 2 | 11 | 0.6875 | 0.8462 |
| controlled_alt_vs_candidate_no_context | context_overlap | 16 | 4 | 4 | 0.7500 | 0.8000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 7 | 6 | 11 | 0.5208 | 0.5385 |
| controlled_alt_vs_candidate_no_context | persona_style | 1 | 6 | 17 | 0.3958 | 0.1429 |
| controlled_alt_vs_candidate_no_context | distinct1 | 9 | 10 | 5 | 0.4792 | 0.4737 |
| controlled_alt_vs_candidate_no_context | length_score | 6 | 14 | 4 | 0.3333 | 0.3000 |
| controlled_alt_vs_candidate_no_context | sentence_score | 3 | 1 | 20 | 0.5417 | 0.7500 |
| controlled_alt_vs_candidate_no_context | overall_quality | 13 | 7 | 4 | 0.6250 | 0.6500 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.0833 | 0.0833 | 0.9167 |
| proposed_contextual_controlled_tuned | 0.0000 | 0.0000 | 0.0000 | 0.2083 | 0.7917 |
| proposed_contextual | 0.0000 | 0.0000 | 0.5417 | 0.0000 | 0.0000 |
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
# Proposal Alignment Evaluation Report

- Run ID: `20260313T055150Z`
- Generated: `2026-03-13T05:57:38.289494+00:00`
- Scenarios: `artifacts\proposal_control_tuning\auto_tune\20260313T051946Z\train_runs\trial_003\seed_19\20260313T055150Z\scenarios.jsonl`
- Scenario count: `24`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_tuned`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0846 (0.0458, 0.1258) | 0.2614 (0.2104, 0.3138) | 0.8656 (0.8500, 0.8806) | 0.3016 (0.2662, 0.3382) | n/a |
| proposed_contextual_controlled_tuned | 0.0892 (0.0462, 0.1384) | 0.2484 (0.1959, 0.3034) | 0.8714 (0.8586, 0.8852) | 0.3003 (0.2667, 0.3382) | n/a |
| proposed_contextual | 0.0825 (0.0403, 0.1343) | 0.2420 (0.1818, 0.3045) | 0.8659 (0.8528, 0.8804) | 0.2933 (0.2557, 0.3300) | n/a |
| candidate_no_context | 0.0234 (0.0133, 0.0352) | 0.2453 (0.1886, 0.3062) | 0.8798 (0.8616, 0.9004) | 0.2692 (0.2445, 0.2959) | n/a |

## Game-facing Outcome Metrics (mean, 95% CI)
| Arm | Quest-state Correctness | Lore Consistency | Contradiction Safety | Objective Completion Support | Gameplay Usefulness | Time-pressure Acceptability |
|---|---:|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.1726 (0.1384, 0.2122) | 0.0369 (0.0113, 0.0681) | 1.0000 (1.0000, 1.0000) | 0.0856 (0.0601, 0.1095) | 0.3000 (0.2867, 0.3127) | 0.3031 (0.2812, 0.3240) |
| proposed_contextual_controlled_tuned | 0.1807 (0.1406, 0.2281) | 0.0343 (0.0095, 0.0651) | 1.0000 (1.0000, 1.0000) | 0.0872 (0.0613, 0.1138) | 0.3050 (0.2906, 0.3211) | 0.3052 (0.2888, 0.3211) |
| proposed_contextual | 0.1683 (0.1337, 0.2091) | 0.0227 (0.0032, 0.0547) | 1.0000 (1.0000, 1.0000) | 0.0844 (0.0611, 0.1112) | 0.2987 (0.2835, 0.3155) | 0.3024 (0.2854, 0.3198) |
| candidate_no_context | 0.1169 (0.1081, 0.1277) | 0.0054 (0.0002, 0.0137) | 1.0000 (1.0000, 1.0000) | 0.0822 (0.0594, 0.1032) | 0.2865 (0.2727, 0.2986) | 0.3020 (0.2874, 0.3167) |

- Multi-turn contradiction rate is reported as `1 - contradiction_safety` in row-level outputs.

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0591 | 2.5247 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0033 | -0.0136 |
| proposed_vs_candidate_no_context | naturalness | -0.0139 | -0.0158 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.0515 | 0.4404 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0173 | 3.2144 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | 0.0023 | 0.0279 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 0.0122 | 0.0427 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | 0.0004 | 0.0013 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0761 | 4.0200 |
| proposed_vs_candidate_no_context | context_overlap | 0.0194 | 0.5723 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0085 | 0.0526 |
| proposed_vs_candidate_no_context | persona_style | -0.0508 | -0.0878 |
| proposed_vs_candidate_no_context | distinct1 | -0.0106 | -0.0114 |
| proposed_vs_candidate_no_context | length_score | -0.0556 | -0.1013 |
| proposed_vs_candidate_no_context | sentence_score | 0.0146 | 0.0153 |
| proposed_vs_candidate_no_context | overall_quality | 0.0241 | 0.0895 |
| controlled_vs_proposed_raw | context_relevance | 0.0021 | 0.0252 |
| controlled_vs_proposed_raw | persona_consistency | 0.0194 | 0.0804 |
| controlled_vs_proposed_raw | naturalness | -0.0003 | -0.0004 |
| controlled_vs_proposed_raw | quest_state_correctness | 0.0042 | 0.0252 |
| controlled_vs_proposed_raw | lore_consistency | 0.0141 | 0.6225 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0011 | 0.0132 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0013 | 0.0043 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 0.0008 | 0.0025 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.0019 | 0.0199 |
| controlled_vs_proposed_raw | context_overlap | 0.0025 | 0.0474 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.0266 | 0.1558 |
| controlled_vs_proposed_raw | persona_style | -0.0091 | -0.0173 |
| controlled_vs_proposed_raw | distinct1 | -0.0086 | -0.0093 |
| controlled_vs_proposed_raw | length_score | 0.0083 | 0.0169 |
| controlled_vs_proposed_raw | sentence_score | 0.0146 | 0.0150 |
| controlled_vs_proposed_raw | overall_quality | 0.0083 | 0.0282 |
| controlled_vs_candidate_no_context | context_relevance | 0.0612 | 2.6136 |
| controlled_vs_candidate_no_context | persona_consistency | 0.0161 | 0.0657 |
| controlled_vs_candidate_no_context | naturalness | -0.0142 | -0.0162 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.0557 | 0.4767 |
| controlled_vs_candidate_no_context | lore_consistency | 0.0315 | 5.8379 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | 0.0034 | 0.0414 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0135 | 0.0472 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 0.0012 | 0.0038 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.0780 | 4.1200 |
| controlled_vs_candidate_no_context | context_overlap | 0.0219 | 0.6468 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.0351 | 0.2166 |
| controlled_vs_candidate_no_context | persona_style | -0.0599 | -0.1036 |
| controlled_vs_candidate_no_context | distinct1 | -0.0193 | -0.0206 |
| controlled_vs_candidate_no_context | length_score | -0.0472 | -0.0861 |
| controlled_vs_candidate_no_context | sentence_score | 0.0292 | 0.0305 |
| controlled_vs_candidate_no_context | overall_quality | 0.0324 | 0.1202 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0046 | 0.0541 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0130 | -0.0499 |
| controlled_alt_vs_controlled_default | naturalness | 0.0059 | 0.0068 |
| controlled_alt_vs_controlled_default | quest_state_correctness | 0.0081 | 0.0471 |
| controlled_alt_vs_controlled_default | lore_consistency | -0.0026 | -0.0701 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_controlled_default | objective_completion_support | 0.0017 | 0.0195 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | 0.0049 | 0.0164 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | 0.0021 | 0.0068 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0076 | 0.0781 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0024 | -0.0435 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0137 | -0.0694 |
| controlled_alt_vs_controlled_default | persona_style | -0.0104 | -0.0201 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0069 | 0.0075 |
| controlled_alt_vs_controlled_default | length_score | 0.0083 | 0.0166 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0146 | 0.0148 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0013 | -0.0043 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.0067 | 0.0807 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.0064 | 0.0265 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0056 | 0.0064 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 0.0124 | 0.0735 |
| controlled_alt_vs_proposed_raw | lore_consistency | 0.0116 | 0.5087 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_proposed_raw | objective_completion_support | 0.0028 | 0.0329 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 0.0062 | 0.0208 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | 0.0028 | 0.0093 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.0095 | 0.0996 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0001 | 0.0018 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.0129 | 0.0756 |
| controlled_alt_vs_proposed_raw | persona_style | -0.0195 | -0.0370 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0017 | -0.0019 |
| controlled_alt_vs_proposed_raw | length_score | 0.0167 | 0.0338 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0292 | 0.0300 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.0070 | 0.0237 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.0658 | 2.8091 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.0031 | 0.0126 |
| controlled_alt_vs_candidate_no_context | naturalness | -0.0083 | -0.0095 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 0.0639 | 0.5464 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 0.0289 | 5.3583 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_candidate_no_context | objective_completion_support | 0.0051 | 0.0617 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 0.0185 | 0.0644 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | 0.0032 | 0.0107 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.0856 | 4.5200 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0195 | 0.5752 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.0214 | 0.1322 |
| controlled_alt_vs_candidate_no_context | persona_style | -0.0703 | -0.1216 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0124 | -0.0132 |
| controlled_alt_vs_candidate_no_context | length_score | -0.0389 | -0.0709 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.0438 | 0.0458 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.0311 | 0.1154 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0591 | (0.0137, 0.1147) | 0.0010 | 0.0591 | (0.0247, 0.0912) | 0.0010 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0033 | (-0.0641, 0.0595) | 0.5520 | -0.0033 | (-0.0429, 0.0435) | 0.5803 |
| proposed_vs_candidate_no_context | naturalness | -0.0139 | (-0.0362, 0.0095) | 0.8753 | -0.0139 | (-0.0441, 0.0137) | 0.8147 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.0515 | (0.0160, 0.0968) | 0.0013 | 0.0515 | (0.0222, 0.0777) | 0.0003 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0173 | (-0.0052, 0.0505) | 0.1193 | 0.0173 | (-0.0039, 0.0431) | 0.1097 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | 0.0023 | (-0.0210, 0.0251) | 0.4077 | 0.0023 | (-0.0242, 0.0237) | 0.4080 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 0.0122 | (-0.0077, 0.0351) | 0.1243 | 0.0122 | (-0.0001, 0.0298) | 0.0357 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | 0.0004 | (-0.0147, 0.0151) | 0.4740 | 0.0004 | (-0.0142, 0.0142) | 0.4160 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0761 | (0.0193, 0.1443) | 0.0017 | 0.0761 | (0.0325, 0.1157) | 0.0000 |
| proposed_vs_candidate_no_context | context_overlap | 0.0194 | (0.0003, 0.0400) | 0.0223 | 0.0194 | (0.0045, 0.0341) | 0.0010 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0085 | (-0.0623, 0.0816) | 0.4127 | 0.0085 | (-0.0524, 0.0682) | 0.3973 |
| proposed_vs_candidate_no_context | persona_style | -0.0508 | (-0.1198, -0.0052) | 1.0000 | -0.0508 | (-0.1406, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | distinct1 | -0.0106 | (-0.0321, 0.0115) | 0.8233 | -0.0106 | (-0.0367, 0.0161) | 0.8143 |
| proposed_vs_candidate_no_context | length_score | -0.0556 | (-0.1375, 0.0347) | 0.8890 | -0.0556 | (-0.1778, 0.0773) | 0.8280 |
| proposed_vs_candidate_no_context | sentence_score | 0.0146 | (-0.0437, 0.0729) | 0.4117 | 0.0146 | (0.0000, 0.0404) | 0.3327 |
| proposed_vs_candidate_no_context | overall_quality | 0.0241 | (-0.0053, 0.0538) | 0.0537 | 0.0241 | (0.0018, 0.0496) | 0.0240 |
| controlled_vs_proposed_raw | context_relevance | 0.0021 | (-0.0513, 0.0547) | 0.4830 | 0.0021 | (-0.0392, 0.0410) | 0.4450 |
| controlled_vs_proposed_raw | persona_consistency | 0.0194 | (-0.0560, 0.0922) | 0.3100 | 0.0194 | (-0.0473, 0.0552) | 0.2487 |
| controlled_vs_proposed_raw | naturalness | -0.0003 | (-0.0181, 0.0171) | 0.5057 | -0.0003 | (-0.0136, 0.0154) | 0.5387 |
| controlled_vs_proposed_raw | quest_state_correctness | 0.0042 | (-0.0384, 0.0491) | 0.4193 | 0.0042 | (-0.0335, 0.0403) | 0.3843 |
| controlled_vs_proposed_raw | lore_consistency | 0.0141 | (-0.0144, 0.0466) | 0.1837 | 0.0141 | (-0.0098, 0.0382) | 0.1163 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0011 | (-0.0199, 0.0235) | 0.4727 | 0.0011 | (-0.0082, 0.0136) | 0.4553 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0013 | (-0.0133, 0.0167) | 0.4273 | 0.0013 | (-0.0081, 0.0145) | 0.3853 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 0.0008 | (-0.0217, 0.0228) | 0.4677 | 0.0008 | (-0.0199, 0.0240) | 0.4443 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.0019 | (-0.0644, 0.0674) | 0.4690 | 0.0019 | (-0.0488, 0.0492) | 0.4383 |
| controlled_vs_proposed_raw | context_overlap | 0.0025 | (-0.0231, 0.0293) | 0.4300 | 0.0025 | (-0.0219, 0.0228) | 0.3883 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.0266 | (-0.0617, 0.1069) | 0.2737 | 0.0266 | (-0.0393, 0.0681) | 0.1743 |
| controlled_vs_proposed_raw | persona_style | -0.0091 | (-0.0690, 0.0742) | 0.6417 | -0.0091 | (-0.1023, 0.0569) | 0.6333 |
| controlled_vs_proposed_raw | distinct1 | -0.0086 | (-0.0307, 0.0134) | 0.8017 | -0.0086 | (-0.0300, 0.0138) | 0.7867 |
| controlled_vs_proposed_raw | length_score | 0.0083 | (-0.0764, 0.0903) | 0.4203 | 0.0083 | (-0.0583, 0.0933) | 0.4297 |
| controlled_vs_proposed_raw | sentence_score | 0.0146 | (-0.0292, 0.0583) | 0.3810 | 0.0146 | (-0.0318, 0.0636) | 0.3763 |
| controlled_vs_proposed_raw | overall_quality | 0.0083 | (-0.0358, 0.0520) | 0.3540 | 0.0083 | (-0.0302, 0.0376) | 0.2937 |
| controlled_vs_candidate_no_context | context_relevance | 0.0612 | (0.0203, 0.1090) | 0.0010 | 0.0612 | (0.0340, 0.0939) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.0161 | (-0.0563, 0.0858) | 0.3237 | 0.0161 | (-0.0885, 0.0968) | 0.3223 |
| controlled_vs_candidate_no_context | naturalness | -0.0142 | (-0.0328, 0.0040) | 0.9337 | -0.0142 | (-0.0364, 0.0119) | 0.8750 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.0557 | (0.0163, 0.1007) | 0.0007 | 0.0557 | (0.0290, 0.0905) | 0.0000 |
| controlled_vs_candidate_no_context | lore_consistency | 0.0315 | (0.0043, 0.0641) | 0.0113 | 0.0315 | (0.0225, 0.0402) | 0.0000 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | 0.0034 | (-0.0188, 0.0247) | 0.3680 | 0.0034 | (-0.0235, 0.0312) | 0.3690 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0135 | (-0.0047, 0.0317) | 0.0700 | 0.0135 | (-0.0062, 0.0411) | 0.1183 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 0.0012 | (-0.0204, 0.0202) | 0.4517 | 0.0012 | (-0.0126, 0.0204) | 0.4383 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.0780 | (0.0265, 0.1386) | 0.0003 | 0.0780 | (0.0427, 0.1218) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0219 | (0.0006, 0.0466) | 0.0210 | 0.0219 | (0.0062, 0.0433) | 0.0003 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.0351 | (-0.0583, 0.1240) | 0.2247 | 0.0351 | (-0.0790, 0.1227) | 0.2433 |
| controlled_vs_candidate_no_context | persona_style | -0.0599 | (-0.1016, -0.0208) | 1.0000 | -0.0599 | (-0.1250, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | distinct1 | -0.0193 | (-0.0370, -0.0027) | 0.9910 | -0.0193 | (-0.0300, -0.0080) | 1.0000 |
| controlled_vs_candidate_no_context | length_score | -0.0472 | (-0.1333, 0.0319) | 0.8617 | -0.0472 | (-0.1385, 0.0606) | 0.8033 |
| controlled_vs_candidate_no_context | sentence_score | 0.0292 | (-0.0292, 0.0875) | 0.2110 | 0.0292 | (-0.0292, 0.0795) | 0.2000 |
| controlled_vs_candidate_no_context | overall_quality | 0.0324 | (-0.0103, 0.0756) | 0.0700 | 0.0324 | (-0.0153, 0.0791) | 0.0870 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0046 | (-0.0334, 0.0435) | 0.4200 | 0.0046 | (-0.0240, 0.0332) | 0.3850 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0130 | (-0.0738, 0.0574) | 0.6583 | -0.0130 | (-0.0502, 0.0478) | 0.7000 |
| controlled_alt_vs_controlled_default | naturalness | 0.0059 | (-0.0061, 0.0205) | 0.1933 | 0.0059 | (-0.0003, 0.0123) | 0.0377 |
| controlled_alt_vs_controlled_default | quest_state_correctness | 0.0081 | (-0.0249, 0.0460) | 0.3280 | 0.0081 | (-0.0154, 0.0388) | 0.2617 |
| controlled_alt_vs_controlled_default | lore_consistency | -0.0026 | (-0.0317, 0.0218) | 0.5807 | -0.0026 | (-0.0263, 0.0211) | 0.5923 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | objective_completion_support | 0.0017 | (-0.0175, 0.0191) | 0.4223 | 0.0017 | (-0.0179, 0.0125) | 0.3420 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | 0.0049 | (-0.0077, 0.0172) | 0.2123 | 0.0049 | (-0.0109, 0.0151) | 0.2200 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | 0.0021 | (-0.0168, 0.0212) | 0.4163 | 0.0021 | (-0.0194, 0.0195) | 0.4143 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0076 | (-0.0417, 0.0568) | 0.4283 | 0.0076 | (-0.0265, 0.0417) | 0.3283 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0024 | (-0.0222, 0.0148) | 0.5923 | -0.0024 | (-0.0228, 0.0195) | 0.5907 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0137 | (-0.0938, 0.0675) | 0.6373 | -0.0137 | (-0.0583, 0.0628) | 0.6633 |
| controlled_alt_vs_controlled_default | persona_style | -0.0104 | (-0.0312, 0.0000) | 1.0000 | -0.0104 | (-0.0288, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0069 | (-0.0088, 0.0229) | 0.2070 | 0.0069 | (-0.0031, 0.0225) | 0.1637 |
| controlled_alt_vs_controlled_default | length_score | 0.0083 | (-0.0514, 0.0764) | 0.4103 | 0.0083 | (-0.0300, 0.0474) | 0.3787 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0146 | (0.0000, 0.0437) | 0.3500 | 0.0146 | (0.0000, 0.0477) | 0.3300 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0013 | (-0.0356, 0.0329) | 0.5310 | -0.0013 | (-0.0180, 0.0184) | 0.5370 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.0067 | (-0.0382, 0.0497) | 0.3797 | 0.0067 | (-0.0471, 0.0560) | 0.3760 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.0064 | (-0.0347, 0.0490) | 0.4187 | 0.0064 | (-0.0094, 0.0364) | 0.3230 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0056 | (-0.0076, 0.0186) | 0.2173 | 0.0056 | (-0.0060, 0.0187) | 0.1950 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 0.0124 | (-0.0268, 0.0517) | 0.2583 | 0.0124 | (-0.0365, 0.0596) | 0.2967 |
| controlled_alt_vs_proposed_raw | lore_consistency | 0.0116 | (-0.0029, 0.0292) | 0.0660 | 0.0116 | (0.0008, 0.0279) | 0.0113 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_proposed_raw | objective_completion_support | 0.0028 | (-0.0143, 0.0210) | 0.3817 | 0.0028 | (-0.0055, 0.0119) | 0.2467 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 0.0062 | (-0.0077, 0.0212) | 0.2027 | 0.0062 | (-0.0094, 0.0263) | 0.2077 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | 0.0028 | (-0.0082, 0.0144) | 0.3237 | 0.0028 | (-0.0024, 0.0074) | 0.1290 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.0095 | (-0.0447, 0.0636) | 0.3670 | 0.0095 | (-0.0613, 0.0789) | 0.3820 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0001 | (-0.0210, 0.0202) | 0.4890 | 0.0001 | (-0.0200, 0.0138) | 0.4547 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.0129 | (-0.0347, 0.0605) | 0.3220 | 0.0129 | (-0.0137, 0.0500) | 0.2673 |
| controlled_alt_vs_proposed_raw | persona_style | -0.0195 | (-0.0703, 0.0404) | 0.7680 | -0.0195 | (-0.1023, 0.0301) | 0.7353 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0017 | (-0.0214, 0.0157) | 0.5677 | -0.0017 | (-0.0188, 0.0213) | 0.6097 |
| controlled_alt_vs_proposed_raw | length_score | 0.0167 | (-0.0528, 0.0903) | 0.3373 | 0.0167 | (-0.0550, 0.0806) | 0.2790 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0292 | (0.0000, 0.0729) | 0.1247 | 0.0292 | (0.0000, 0.0700) | 0.0743 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.0070 | (-0.0185, 0.0357) | 0.3223 | 0.0070 | (-0.0181, 0.0361) | 0.3113 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.0658 | (0.0222, 0.1148) | 0.0003 | 0.0658 | (0.0170, 0.1233) | 0.0077 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.0031 | (-0.0500, 0.0589) | 0.4567 | 0.0031 | (-0.0497, 0.0763) | 0.4620 |
| controlled_alt_vs_candidate_no_context | naturalness | -0.0083 | (-0.0297, 0.0125) | 0.7820 | -0.0083 | (-0.0298, 0.0170) | 0.7503 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 0.0639 | (0.0219, 0.1098) | 0.0000 | 0.0639 | (0.0171, 0.1191) | 0.0000 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 0.0289 | (0.0053, 0.0615) | 0.0017 | 0.0289 | (0.0014, 0.0564) | 0.0097 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_candidate_no_context | objective_completion_support | 0.0051 | (-0.0194, 0.0306) | 0.3527 | 0.0051 | (-0.0221, 0.0318) | 0.4103 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 0.0185 | (-0.0046, 0.0405) | 0.0573 | 0.0185 | (-0.0052, 0.0465) | 0.0587 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | 0.0032 | (-0.0123, 0.0199) | 0.3517 | 0.0032 | (-0.0076, 0.0140) | 0.2963 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.0856 | (0.0280, 0.1545) | 0.0003 | 0.0856 | (0.0227, 0.1612) | 0.0087 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0195 | (0.0041, 0.0362) | 0.0063 | 0.0195 | (0.0028, 0.0357) | 0.0083 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.0214 | (-0.0455, 0.0889) | 0.2667 | 0.0214 | (-0.0374, 0.1024) | 0.2857 |
| controlled_alt_vs_candidate_no_context | persona_style | -0.0703 | (-0.1133, -0.0326) | 1.0000 | -0.0703 | (-0.1449, 0.0000) | 1.0000 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0124 | (-0.0277, 0.0021) | 0.9470 | -0.0124 | (-0.0224, 0.0010) | 0.9587 |
| controlled_alt_vs_candidate_no_context | length_score | -0.0389 | (-0.1264, 0.0556) | 0.7910 | -0.0389 | (-0.1250, 0.0545) | 0.7520 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.0437 | (0.0000, 0.0875) | 0.0400 | 0.0437 | (0.0135, 0.0795) | 0.0083 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.0311 | (-0.0020, 0.0673) | 0.0343 | 0.0311 | (-0.0135, 0.0837) | 0.0950 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 11 | 5 | 8 | 0.6250 | 0.6875 |
| proposed_vs_candidate_no_context | persona_consistency | 6 | 6 | 12 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | naturalness | 5 | 11 | 8 | 0.3750 | 0.3125 |
| proposed_vs_candidate_no_context | quest_state_correctness | 11 | 5 | 8 | 0.6250 | 0.6875 |
| proposed_vs_candidate_no_context | lore_consistency | 8 | 1 | 15 | 0.6458 | 0.8889 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| proposed_vs_candidate_no_context | objective_completion_support | 9 | 7 | 8 | 0.5417 | 0.5625 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 9 | 7 | 8 | 0.5417 | 0.5625 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | 7 | 9 | 8 | 0.4583 | 0.4375 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 7 | 1 | 16 | 0.6250 | 0.8750 |
| proposed_vs_candidate_no_context | context_overlap | 10 | 6 | 8 | 0.5833 | 0.6250 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 6 | 6 | 12 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | persona_style | 0 | 4 | 20 | 0.4167 | 0.0000 |
| proposed_vs_candidate_no_context | distinct1 | 7 | 8 | 9 | 0.4792 | 0.4667 |
| proposed_vs_candidate_no_context | length_score | 6 | 10 | 8 | 0.4167 | 0.3750 |
| proposed_vs_candidate_no_context | sentence_score | 3 | 2 | 19 | 0.5208 | 0.6000 |
| proposed_vs_candidate_no_context | overall_quality | 12 | 4 | 8 | 0.6667 | 0.7500 |
| controlled_vs_proposed_raw | context_relevance | 10 | 9 | 5 | 0.5208 | 0.5263 |
| controlled_vs_proposed_raw | persona_consistency | 9 | 7 | 8 | 0.5417 | 0.5625 |
| controlled_vs_proposed_raw | naturalness | 9 | 10 | 5 | 0.4792 | 0.4737 |
| controlled_vs_proposed_raw | quest_state_correctness | 8 | 11 | 5 | 0.4375 | 0.4211 |
| controlled_vs_proposed_raw | lore_consistency | 5 | 7 | 12 | 0.4583 | 0.4167 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_vs_proposed_raw | objective_completion_support | 5 | 5 | 14 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | gameplay_usefulness | 11 | 8 | 5 | 0.5625 | 0.5789 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 7 | 6 | 11 | 0.5208 | 0.5385 |
| controlled_vs_proposed_raw | context_keyword_coverage | 7 | 7 | 10 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | context_overlap | 11 | 8 | 5 | 0.5625 | 0.5789 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 9 | 6 | 9 | 0.5625 | 0.6000 |
| controlled_vs_proposed_raw | persona_style | 2 | 6 | 16 | 0.4167 | 0.2500 |
| controlled_vs_proposed_raw | distinct1 | 9 | 10 | 5 | 0.4792 | 0.4737 |
| controlled_vs_proposed_raw | length_score | 9 | 10 | 5 | 0.4792 | 0.4737 |
| controlled_vs_proposed_raw | sentence_score | 2 | 1 | 21 | 0.5208 | 0.6667 |
| controlled_vs_proposed_raw | overall_quality | 10 | 9 | 5 | 0.5208 | 0.5263 |
| controlled_vs_candidate_no_context | context_relevance | 13 | 5 | 6 | 0.6667 | 0.7222 |
| controlled_vs_candidate_no_context | persona_consistency | 10 | 7 | 7 | 0.5625 | 0.5882 |
| controlled_vs_candidate_no_context | naturalness | 7 | 11 | 6 | 0.4167 | 0.3889 |
| controlled_vs_candidate_no_context | quest_state_correctness | 13 | 5 | 6 | 0.6667 | 0.7222 |
| controlled_vs_candidate_no_context | lore_consistency | 7 | 1 | 16 | 0.6250 | 0.8750 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_vs_candidate_no_context | objective_completion_support | 8 | 5 | 11 | 0.5625 | 0.6154 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 11 | 7 | 6 | 0.5833 | 0.6111 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 7 | 7 | 10 | 0.5000 | 0.5000 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 8 | 2 | 14 | 0.6250 | 0.8000 |
| controlled_vs_candidate_no_context | context_overlap | 11 | 7 | 6 | 0.5833 | 0.6111 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 10 | 5 | 9 | 0.6042 | 0.6667 |
| controlled_vs_candidate_no_context | persona_style | 0 | 7 | 17 | 0.3542 | 0.0000 |
| controlled_vs_candidate_no_context | distinct1 | 7 | 11 | 6 | 0.4167 | 0.3889 |
| controlled_vs_candidate_no_context | length_score | 9 | 9 | 6 | 0.5000 | 0.5000 |
| controlled_vs_candidate_no_context | sentence_score | 3 | 1 | 20 | 0.5417 | 0.7500 |
| controlled_vs_candidate_no_context | overall_quality | 12 | 6 | 6 | 0.6250 | 0.6667 |
| controlled_alt_vs_controlled_default | context_relevance | 7 | 4 | 13 | 0.5625 | 0.6364 |
| controlled_alt_vs_controlled_default | persona_consistency | 3 | 6 | 15 | 0.4375 | 0.3333 |
| controlled_alt_vs_controlled_default | naturalness | 5 | 6 | 13 | 0.4792 | 0.4545 |
| controlled_alt_vs_controlled_default | quest_state_correctness | 7 | 4 | 13 | 0.5625 | 0.6364 |
| controlled_alt_vs_controlled_default | lore_consistency | 4 | 5 | 15 | 0.4792 | 0.4444 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_alt_vs_controlled_default | objective_completion_support | 6 | 4 | 14 | 0.5417 | 0.6000 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | 6 | 5 | 13 | 0.5208 | 0.5455 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | 5 | 6 | 13 | 0.4792 | 0.4545 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 3 | 3 | 18 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | context_overlap | 7 | 4 | 13 | 0.5625 | 0.6364 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 3 | 6 | 15 | 0.4375 | 0.3333 |
| controlled_alt_vs_controlled_default | persona_style | 0 | 1 | 23 | 0.4792 | 0.0000 |
| controlled_alt_vs_controlled_default | distinct1 | 6 | 5 | 13 | 0.5208 | 0.5455 |
| controlled_alt_vs_controlled_default | length_score | 4 | 7 | 13 | 0.4375 | 0.3636 |
| controlled_alt_vs_controlled_default | sentence_score | 1 | 0 | 23 | 0.5208 | 1.0000 |
| controlled_alt_vs_controlled_default | overall_quality | 6 | 5 | 13 | 0.5208 | 0.5455 |
| controlled_alt_vs_proposed_raw | context_relevance | 11 | 7 | 6 | 0.5833 | 0.6111 |
| controlled_alt_vs_proposed_raw | persona_consistency | 5 | 7 | 12 | 0.4583 | 0.4167 |
| controlled_alt_vs_proposed_raw | naturalness | 8 | 10 | 6 | 0.4583 | 0.4444 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 9 | 9 | 6 | 0.5000 | 0.5000 |
| controlled_alt_vs_proposed_raw | lore_consistency | 6 | 5 | 13 | 0.5208 | 0.5455 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_alt_vs_proposed_raw | objective_completion_support | 6 | 4 | 14 | 0.5417 | 0.6000 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 9 | 9 | 6 | 0.5000 | 0.5000 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | 7 | 4 | 13 | 0.5625 | 0.6364 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 6 | 5 | 13 | 0.5208 | 0.5455 |
| controlled_alt_vs_proposed_raw | context_overlap | 11 | 7 | 6 | 0.5833 | 0.6111 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 5 | 4 | 15 | 0.5208 | 0.5556 |
| controlled_alt_vs_proposed_raw | persona_style | 2 | 6 | 16 | 0.4167 | 0.2500 |
| controlled_alt_vs_proposed_raw | distinct1 | 10 | 7 | 7 | 0.5625 | 0.5882 |
| controlled_alt_vs_proposed_raw | length_score | 8 | 9 | 7 | 0.4792 | 0.4706 |
| controlled_alt_vs_proposed_raw | sentence_score | 2 | 0 | 22 | 0.5417 | 1.0000 |
| controlled_alt_vs_proposed_raw | overall_quality | 9 | 9 | 6 | 0.5000 | 0.5000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 13 | 4 | 7 | 0.6875 | 0.7647 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 6 | 8 | 10 | 0.4583 | 0.4286 |
| controlled_alt_vs_candidate_no_context | naturalness | 7 | 10 | 7 | 0.4375 | 0.4118 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 13 | 4 | 7 | 0.6875 | 0.7647 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 6 | 1 | 17 | 0.6042 | 0.8571 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_alt_vs_candidate_no_context | objective_completion_support | 7 | 6 | 11 | 0.5208 | 0.5385 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 11 | 6 | 7 | 0.6042 | 0.6471 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | 6 | 7 | 11 | 0.4792 | 0.4615 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 7 | 1 | 16 | 0.6250 | 0.8750 |
| controlled_alt_vs_candidate_no_context | context_overlap | 12 | 5 | 7 | 0.6458 | 0.7059 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 6 | 4 | 14 | 0.5417 | 0.6000 |
| controlled_alt_vs_candidate_no_context | persona_style | 0 | 8 | 16 | 0.3333 | 0.0000 |
| controlled_alt_vs_candidate_no_context | distinct1 | 6 | 10 | 8 | 0.4167 | 0.3750 |
| controlled_alt_vs_candidate_no_context | length_score | 9 | 8 | 7 | 0.5208 | 0.5294 |
| controlled_alt_vs_candidate_no_context | sentence_score | 3 | 0 | 21 | 0.5625 | 1.0000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 10 | 7 | 7 | 0.5625 | 0.5882 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.1250 | 0.2083 | 0.7917 |
| proposed_contextual_controlled_tuned | 0.0000 | 0.0000 | 0.1250 | 0.1667 | 0.7917 |
| proposed_contextual | 0.0000 | 0.0000 | 0.5417 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.4167 | 0.0000 | 0.0000 |

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
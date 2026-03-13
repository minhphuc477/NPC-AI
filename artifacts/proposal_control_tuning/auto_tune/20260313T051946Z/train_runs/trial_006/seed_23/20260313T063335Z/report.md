# Proposal Alignment Evaluation Report

- Run ID: `20260313T063335Z`
- Generated: `2026-03-13T06:39:20.511497+00:00`
- Scenarios: `artifacts\proposal_control_tuning\auto_tune\20260313T051946Z\train_runs\trial_006\seed_23\20260313T063335Z\scenarios.jsonl`
- Scenario count: `24`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_tuned`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.1170 (0.0651, 0.1779) | 0.2297 (0.1875, 0.2767) | 0.8938 (0.8728, 0.9141) | 0.3109 (0.2731, 0.3500) | n/a |
| proposed_contextual_controlled_tuned | 0.0864 (0.0457, 0.1366) | 0.2160 (0.1747, 0.2595) | 0.8744 (0.8590, 0.8909) | 0.2883 (0.2574, 0.3251) | n/a |
| proposed_contextual | 0.0684 (0.0366, 0.1056) | 0.2058 (0.1538, 0.2691) | 0.8777 (0.8616, 0.8953) | 0.2758 (0.2477, 0.3068) | n/a |
| candidate_no_context | 0.0271 (0.0155, 0.0405) | 0.2576 (0.1922, 0.3287) | 0.8767 (0.8606, 0.8925) | 0.2757 (0.2474, 0.3059) | n/a |

## Game-facing Outcome Metrics (mean, 95% CI)
| Arm | Quest-state Correctness | Lore Consistency | Contradiction Safety | Objective Completion Support | Gameplay Usefulness | Time-pressure Acceptability |
|---|---:|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.1978 (0.1533, 0.2448) | 0.0579 (0.0198, 0.1036) | 1.0000 (1.0000, 1.0000) | 0.0718 (0.0457, 0.0967) | 0.3155 (0.3005, 0.3314) | 0.2916 (0.2797, 0.3043) |
| proposed_contextual_controlled_tuned | 0.1754 (0.1379, 0.2204) | 0.0383 (0.0145, 0.0709) | 1.0000 (1.0000, 1.0000) | 0.0962 (0.0714, 0.1188) | 0.3093 (0.2978, 0.3210) | 0.3055 (0.2889, 0.3218) |
| proposed_contextual | 0.1558 (0.1295, 0.1880) | 0.0264 (0.0081, 0.0542) | 1.0000 (1.0000, 1.0000) | 0.0679 (0.0413, 0.0949) | 0.2946 (0.2789, 0.3089) | 0.2885 (0.2763, 0.3016) |
| candidate_no_context | 0.1203 (0.1117, 0.1300) | 0.0053 (0.0009, 0.0115) | 1.0000 (1.0000, 1.0000) | 0.0599 (0.0383, 0.0818) | 0.2797 (0.2683, 0.2912) | 0.2890 (0.2781, 0.3002) |

- Multi-turn contradiction rate is reported as `1 - contradiction_safety` in row-level outputs.

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0413 | 1.5220 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0518 | -0.2011 |
| proposed_vs_candidate_no_context | naturalness | 0.0011 | 0.0012 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.0355 | 0.2953 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0212 | 4.0303 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | 0.0081 | 0.1346 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 0.0148 | 0.0531 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | -0.0006 | -0.0020 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0530 | 2.3333 |
| proposed_vs_candidate_no_context | context_overlap | 0.0138 | 0.3695 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0615 | -0.3196 |
| proposed_vs_candidate_no_context | persona_style | -0.0130 | -0.0251 |
| proposed_vs_candidate_no_context | distinct1 | 0.0012 | 0.0013 |
| proposed_vs_candidate_no_context | length_score | -0.0042 | -0.0078 |
| proposed_vs_candidate_no_context | sentence_score | 0.0146 | 0.0155 |
| proposed_vs_candidate_no_context | overall_quality | 0.0001 | 0.0003 |
| controlled_vs_proposed_raw | context_relevance | 0.0486 | 0.7115 |
| controlled_vs_proposed_raw | persona_consistency | 0.0239 | 0.1161 |
| controlled_vs_proposed_raw | naturalness | 0.0160 | 0.0183 |
| controlled_vs_proposed_raw | quest_state_correctness | 0.0420 | 0.2695 |
| controlled_vs_proposed_raw | lore_consistency | 0.0315 | 1.1928 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0039 | 0.0573 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0210 | 0.0712 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 0.0032 | 0.0110 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.0636 | 0.8400 |
| controlled_vs_proposed_raw | context_overlap | 0.0137 | 0.2672 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.0357 | 0.2727 |
| controlled_vs_proposed_raw | persona_style | -0.0234 | -0.0464 |
| controlled_vs_proposed_raw | distinct1 | 0.0001 | 0.0001 |
| controlled_vs_proposed_raw | length_score | 0.0611 | 0.1155 |
| controlled_vs_proposed_raw | sentence_score | 0.0438 | 0.0458 |
| controlled_vs_proposed_raw | overall_quality | 0.0351 | 0.1273 |
| controlled_vs_candidate_no_context | context_relevance | 0.0899 | 3.3163 |
| controlled_vs_candidate_no_context | persona_consistency | -0.0279 | -0.1084 |
| controlled_vs_candidate_no_context | naturalness | 0.0171 | 0.0196 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.0775 | 0.6444 |
| controlled_vs_candidate_no_context | lore_consistency | 0.0527 | 10.0305 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | 0.0119 | 0.1995 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0358 | 0.1280 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 0.0026 | 0.0090 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.1167 | 5.1333 |
| controlled_vs_candidate_no_context | context_overlap | 0.0275 | 0.7353 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | -0.0258 | -0.1340 |
| controlled_vs_candidate_no_context | persona_style | -0.0365 | -0.0704 |
| controlled_vs_candidate_no_context | distinct1 | 0.0013 | 0.0014 |
| controlled_vs_candidate_no_context | length_score | 0.0569 | 0.1068 |
| controlled_vs_candidate_no_context | sentence_score | 0.0583 | 0.0619 |
| controlled_vs_candidate_no_context | overall_quality | 0.0352 | 0.1277 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0306 | -0.2616 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0137 | -0.0596 |
| controlled_alt_vs_controlled_default | naturalness | -0.0194 | -0.0217 |
| controlled_alt_vs_controlled_default | quest_state_correctness | -0.0223 | -0.1129 |
| controlled_alt_vs_controlled_default | lore_consistency | -0.0196 | -0.3389 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_controlled_default | objective_completion_support | 0.0244 | 0.3395 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | -0.0063 | -0.0199 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | 0.0139 | 0.0475 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0367 | -0.2636 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0163 | -0.2514 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0109 | -0.0655 |
| controlled_alt_vs_controlled_default | persona_style | -0.0247 | -0.0514 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0132 | -0.0140 |
| controlled_alt_vs_controlled_default | length_score | -0.0736 | -0.1247 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0000 | 0.0000 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0226 | -0.0727 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.0180 | 0.2638 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.0102 | 0.0496 |
| controlled_alt_vs_proposed_raw | naturalness | -0.0034 | -0.0038 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 0.0197 | 0.1262 |
| controlled_alt_vs_proposed_raw | lore_consistency | 0.0119 | 0.4497 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_proposed_raw | objective_completion_support | 0.0283 | 0.4162 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 0.0147 | 0.0499 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | 0.0170 | 0.0590 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.0269 | 0.3550 |
| controlled_alt_vs_proposed_raw | context_overlap | -0.0026 | -0.0514 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.0248 | 0.1894 |
| controlled_alt_vs_proposed_raw | persona_style | -0.0482 | -0.0954 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0131 | -0.0139 |
| controlled_alt_vs_proposed_raw | length_score | -0.0125 | -0.0236 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0438 | 0.0458 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.0125 | 0.0454 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.0593 | 2.1873 |
| controlled_alt_vs_candidate_no_context | persona_consistency | -0.0416 | -0.1615 |
| controlled_alt_vs_candidate_no_context | naturalness | -0.0023 | -0.0026 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 0.0552 | 0.4588 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 0.0330 | 6.2926 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_candidate_no_context | objective_completion_support | 0.0363 | 0.6067 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 0.0295 | 0.1056 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | 0.0165 | 0.0570 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.0799 | 3.5167 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0112 | 0.2991 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | -0.0367 | -0.1907 |
| controlled_alt_vs_candidate_no_context | persona_style | -0.0612 | -0.1181 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0119 | -0.0127 |
| controlled_alt_vs_candidate_no_context | length_score | -0.0167 | -0.0312 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.0583 | 0.0619 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.0126 | 0.0457 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0413 | (0.0121, 0.0774) | 0.0000 | 0.0413 | (0.0002, 0.0793) | 0.0087 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0518 | (-0.1140, 0.0051) | 0.9613 | -0.0518 | (-0.1371, 0.0051) | 0.9600 |
| proposed_vs_candidate_no_context | naturalness | 0.0011 | (-0.0208, 0.0237) | 0.4827 | 0.0011 | (-0.0347, 0.0381) | 0.5093 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.0355 | (0.0103, 0.0661) | 0.0007 | 0.0355 | (0.0001, 0.0651) | 0.0130 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0212 | (0.0033, 0.0475) | 0.0010 | 0.0212 | (0.0000, 0.0509) | 0.0783 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | 0.0081 | (-0.0088, 0.0262) | 0.1800 | 0.0081 | (-0.0054, 0.0332) | 0.3217 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 0.0148 | (-0.0016, 0.0307) | 0.0397 | 0.0148 | (-0.0015, 0.0321) | 0.0333 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | -0.0006 | (-0.0103, 0.0105) | 0.5677 | -0.0006 | (-0.0119, 0.0133) | 0.6457 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0530 | (0.0152, 0.0985) | 0.0020 | 0.0530 | (0.0000, 0.1039) | 0.0853 |
| proposed_vs_candidate_no_context | context_overlap | 0.0138 | (0.0009, 0.0295) | 0.0163 | 0.0138 | (0.0007, 0.0228) | 0.0110 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0615 | (-0.1359, 0.0010) | 0.9737 | -0.0615 | (-0.1714, 0.0064) | 0.9583 |
| proposed_vs_candidate_no_context | persona_style | -0.0130 | (-0.0521, 0.0208) | 0.8123 | -0.0130 | (-0.0426, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | distinct1 | 0.0012 | (-0.0117, 0.0142) | 0.4400 | 0.0012 | (-0.0252, 0.0175) | 0.4097 |
| proposed_vs_candidate_no_context | length_score | -0.0042 | (-0.1028, 0.0847) | 0.5403 | -0.0042 | (-0.1121, 0.1359) | 0.5773 |
| proposed_vs_candidate_no_context | sentence_score | 0.0146 | (-0.0292, 0.0583) | 0.3807 | 0.0146 | (-0.0404, 0.0808) | 0.4377 |
| proposed_vs_candidate_no_context | overall_quality | 0.0001 | (-0.0315, 0.0283) | 0.4977 | 0.0001 | (-0.0509, 0.0365) | 0.4693 |
| controlled_vs_proposed_raw | context_relevance | 0.0486 | (0.0010, 0.1032) | 0.0213 | 0.0486 | (0.0162, 0.0761) | 0.0003 |
| controlled_vs_proposed_raw | persona_consistency | 0.0239 | (-0.0204, 0.0728) | 0.1643 | 0.0239 | (-0.0493, 0.0775) | 0.2593 |
| controlled_vs_proposed_raw | naturalness | 0.0160 | (-0.0053, 0.0376) | 0.0630 | 0.0160 | (0.0034, 0.0332) | 0.0007 |
| controlled_vs_proposed_raw | quest_state_correctness | 0.0420 | (0.0026, 0.0861) | 0.0163 | 0.0420 | (0.0136, 0.0642) | 0.0007 |
| controlled_vs_proposed_raw | lore_consistency | 0.0315 | (-0.0087, 0.0760) | 0.0667 | 0.0315 | (0.0084, 0.0564) | 0.0127 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0039 | (-0.0119, 0.0202) | 0.3033 | 0.0039 | (-0.0094, 0.0202) | 0.3553 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0210 | (0.0005, 0.0437) | 0.0207 | 0.0210 | (0.0044, 0.0330) | 0.0083 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 0.0032 | (-0.0047, 0.0109) | 0.2053 | 0.0032 | (0.0003, 0.0069) | 0.0093 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.0636 | (0.0042, 0.1409) | 0.0177 | 0.0636 | (0.0211, 0.1028) | 0.0100 |
| controlled_vs_proposed_raw | context_overlap | 0.0137 | (-0.0088, 0.0391) | 0.1327 | 0.0137 | (-0.0055, 0.0352) | 0.1233 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.0357 | (-0.0198, 0.0982) | 0.0930 | 0.0357 | (-0.0476, 0.0962) | 0.1877 |
| controlled_vs_proposed_raw | persona_style | -0.0234 | (-0.0677, 0.0182) | 0.8727 | -0.0234 | (-0.1023, 0.0234) | 0.7490 |
| controlled_vs_proposed_raw | distinct1 | 0.0001 | (-0.0166, 0.0169) | 0.4843 | 0.0001 | (-0.0218, 0.0207) | 0.4490 |
| controlled_vs_proposed_raw | length_score | 0.0611 | (-0.0264, 0.1514) | 0.0920 | 0.0611 | (-0.0100, 0.1381) | 0.0547 |
| controlled_vs_proposed_raw | sentence_score | 0.0437 | (0.0000, 0.1021) | 0.0370 | 0.0437 | (0.0000, 0.0875) | 0.0830 |
| controlled_vs_proposed_raw | overall_quality | 0.0351 | (0.0041, 0.0662) | 0.0133 | 0.0351 | (-0.0039, 0.0630) | 0.0343 |
| controlled_vs_candidate_no_context | context_relevance | 0.0899 | (0.0360, 0.1505) | 0.0000 | 0.0899 | (0.0188, 0.1407) | 0.0007 |
| controlled_vs_candidate_no_context | persona_consistency | -0.0279 | (-0.1036, 0.0368) | 0.7673 | -0.0279 | (-0.1737, 0.0762) | 0.6977 |
| controlled_vs_candidate_no_context | naturalness | 0.0171 | (-0.0106, 0.0463) | 0.1230 | 0.0171 | (-0.0284, 0.0712) | 0.3253 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.0775 | (0.0326, 0.1284) | 0.0000 | 0.0775 | (0.0143, 0.1229) | 0.0000 |
| controlled_vs_candidate_no_context | lore_consistency | 0.0527 | (0.0157, 0.0977) | 0.0000 | 0.0527 | (0.0139, 0.0855) | 0.0043 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | 0.0119 | (-0.0083, 0.0319) | 0.1213 | 0.0119 | (-0.0068, 0.0365) | 0.1410 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0358 | (0.0147, 0.0582) | 0.0000 | 0.0358 | (0.0060, 0.0623) | 0.0007 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 0.0026 | (-0.0090, 0.0152) | 0.3403 | 0.0026 | (-0.0053, 0.0138) | 0.3043 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.1167 | (0.0492, 0.2000) | 0.0007 | 0.1167 | (0.0259, 0.1815) | 0.0097 |
| controlled_vs_candidate_no_context | context_overlap | 0.0275 | (0.0074, 0.0492) | 0.0030 | 0.0275 | (-0.0038, 0.0564) | 0.0497 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | -0.0258 | (-0.1111, 0.0506) | 0.7247 | -0.0258 | (-0.1952, 0.0952) | 0.5977 |
| controlled_vs_candidate_no_context | persona_style | -0.0365 | (-0.0951, 0.0156) | 0.9107 | -0.0365 | (-0.1080, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | distinct1 | 0.0013 | (-0.0141, 0.0176) | 0.4537 | 0.0013 | (-0.0237, 0.0263) | 0.4543 |
| controlled_vs_candidate_no_context | length_score | 0.0569 | (-0.0625, 0.1764) | 0.1897 | 0.0569 | (-0.1150, 0.2690) | 0.3380 |
| controlled_vs_candidate_no_context | sentence_score | 0.0583 | (0.0146, 0.1167) | 0.0127 | 0.0583 | (0.0000, 0.1212) | 0.0727 |
| controlled_vs_candidate_no_context | overall_quality | 0.0352 | (-0.0119, 0.0785) | 0.0720 | 0.0352 | (-0.0504, 0.0979) | 0.2043 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0306 | (-0.0947, 0.0250) | 0.8557 | -0.0306 | (-0.0526, -0.0098) | 1.0000 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0137 | (-0.0652, 0.0326) | 0.6970 | -0.0137 | (-0.0567, 0.0119) | 0.7910 |
| controlled_alt_vs_controlled_default | naturalness | -0.0194 | (-0.0417, 0.0024) | 0.9597 | -0.0194 | (-0.0321, -0.0074) | 1.0000 |
| controlled_alt_vs_controlled_default | quest_state_correctness | -0.0223 | (-0.0793, 0.0268) | 0.8073 | -0.0223 | (-0.0444, -0.0038) | 1.0000 |
| controlled_alt_vs_controlled_default | lore_consistency | -0.0196 | (-0.0685, 0.0214) | 0.7937 | -0.0196 | (-0.0460, 0.0058) | 0.9173 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | objective_completion_support | 0.0244 | (0.0047, 0.0481) | 0.0057 | 0.0244 | (0.0043, 0.0440) | 0.0103 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | -0.0063 | (-0.0234, 0.0086) | 0.7770 | -0.0063 | (-0.0153, 0.0023) | 0.9120 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | 0.0139 | (-0.0032, 0.0299) | 0.0500 | 0.0139 | (0.0016, 0.0283) | 0.0083 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0367 | (-0.1281, 0.0353) | 0.8090 | -0.0367 | (-0.0707, -0.0083) | 1.0000 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0163 | (-0.0374, 0.0040) | 0.9473 | -0.0163 | (-0.0337, -0.0038) | 1.0000 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0109 | (-0.0704, 0.0417) | 0.6700 | -0.0109 | (-0.0584, 0.0179) | 0.7267 |
| controlled_alt_vs_controlled_default | persona_style | -0.0247 | (-0.0560, 0.0000) | 1.0000 | -0.0247 | (-0.0594, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0132 | (-0.0291, 0.0011) | 0.9670 | -0.0132 | (-0.0291, -0.0013) | 1.0000 |
| controlled_alt_vs_controlled_default | length_score | -0.0736 | (-0.1611, 0.0181) | 0.9453 | -0.0736 | (-0.1036, -0.0350) | 1.0000 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0226 | (-0.0619, 0.0133) | 0.8870 | -0.0226 | (-0.0360, -0.0113) | 1.0000 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.0180 | (-0.0281, 0.0657) | 0.2180 | 0.0180 | (0.0015, 0.0331) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.0102 | (-0.0574, 0.0736) | 0.3747 | 0.0102 | (-0.0986, 0.0857) | 0.4110 |
| controlled_alt_vs_proposed_raw | naturalness | -0.0034 | (-0.0230, 0.0169) | 0.6153 | -0.0034 | (-0.0097, 0.0026) | 0.8603 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 0.0197 | (-0.0230, 0.0640) | 0.1807 | 0.0197 | (0.0001, 0.0337) | 0.0003 |
| controlled_alt_vs_proposed_raw | lore_consistency | 0.0119 | (-0.0221, 0.0447) | 0.2330 | 0.0119 | (-0.0178, 0.0416) | 0.2670 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_proposed_raw | objective_completion_support | 0.0283 | (0.0019, 0.0542) | 0.0207 | 0.0283 | (0.0082, 0.0428) | 0.0137 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 0.0147 | (-0.0014, 0.0332) | 0.0373 | 0.0147 | (0.0017, 0.0241) | 0.0107 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | 0.0170 | (-0.0013, 0.0350) | 0.0343 | 0.0170 | (0.0037, 0.0299) | 0.0103 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.0269 | (-0.0341, 0.0875) | 0.1890 | 0.0269 | (0.0045, 0.0458) | 0.0127 |
| controlled_alt_vs_proposed_raw | context_overlap | -0.0026 | (-0.0209, 0.0154) | 0.6193 | -0.0026 | (-0.0141, 0.0034) | 0.6950 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.0248 | (-0.0536, 0.0972) | 0.2480 | 0.0248 | (-0.0905, 0.1071) | 0.3100 |
| controlled_alt_vs_proposed_raw | persona_style | -0.0482 | (-0.0911, -0.0091) | 0.9930 | -0.0482 | (-0.1435, 0.0000) | 1.0000 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0131 | (-0.0344, 0.0063) | 0.9060 | -0.0131 | (-0.0345, 0.0157) | 0.8080 |
| controlled_alt_vs_proposed_raw | length_score | -0.0125 | (-0.0903, 0.0708) | 0.6330 | -0.0125 | (-0.0717, 0.0372) | 0.7003 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0437 | (0.0000, 0.0875) | 0.0357 | 0.0437 | (0.0000, 0.0875) | 0.0737 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.0125 | (-0.0232, 0.0497) | 0.2503 | 0.0125 | (-0.0334, 0.0453) | 0.2843 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.0593 | (0.0193, 0.1056) | 0.0010 | 0.0593 | (0.0026, 0.1041) | 0.0003 |
| controlled_alt_vs_candidate_no_context | persona_consistency | -0.0416 | (-0.1217, 0.0364) | 0.8537 | -0.0416 | (-0.2186, 0.0810) | 0.6907 |
| controlled_alt_vs_candidate_no_context | naturalness | -0.0023 | (-0.0259, 0.0233) | 0.5967 | -0.0023 | (-0.0457, 0.0399) | 0.5720 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 0.0552 | (0.0173, 0.0980) | 0.0003 | 0.0552 | (0.0003, 0.0988) | 0.0003 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 0.0330 | (0.0108, 0.0605) | 0.0000 | 0.0330 | (0.0096, 0.0523) | 0.0103 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_candidate_no_context | objective_completion_support | 0.0363 | (0.0093, 0.0629) | 0.0050 | 0.0363 | (0.0100, 0.0666) | 0.0117 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 0.0295 | (0.0122, 0.0484) | 0.0000 | 0.0295 | (0.0020, 0.0545) | 0.0150 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | 0.0165 | (-0.0029, 0.0345) | 0.0467 | 0.0165 | (-0.0008, 0.0338) | 0.0837 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.0799 | (0.0302, 0.1402) | 0.0003 | 0.0799 | (0.0045, 0.1373) | 0.0080 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0112 | (-0.0051, 0.0271) | 0.0843 | 0.0112 | (-0.0103, 0.0265) | 0.1643 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | -0.0367 | (-0.1270, 0.0506) | 0.7810 | -0.0367 | (-0.2381, 0.1071) | 0.6223 |
| controlled_alt_vs_candidate_no_context | persona_style | -0.0612 | (-0.1146, -0.0195) | 1.0000 | -0.0612 | (-0.1577, 0.0000) | 1.0000 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0119 | (-0.0324, 0.0082) | 0.8727 | -0.0119 | (-0.0309, 0.0168) | 0.8347 |
| controlled_alt_vs_candidate_no_context | length_score | -0.0167 | (-0.1125, 0.0875) | 0.6443 | -0.0167 | (-0.1850, 0.1731) | 0.5983 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.0583 | (0.0146, 0.1167) | 0.0190 | 0.0583 | (0.0000, 0.1250) | 0.0760 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.0126 | (-0.0299, 0.0566) | 0.2807 | 0.0126 | (-0.0901, 0.0818) | 0.3860 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 10 | 3 | 11 | 0.6458 | 0.7692 |
| proposed_vs_candidate_no_context | persona_consistency | 2 | 5 | 17 | 0.4375 | 0.2857 |
| proposed_vs_candidate_no_context | naturalness | 8 | 5 | 11 | 0.5625 | 0.6154 |
| proposed_vs_candidate_no_context | quest_state_correctness | 10 | 4 | 10 | 0.6250 | 0.7143 |
| proposed_vs_candidate_no_context | lore_consistency | 7 | 1 | 16 | 0.6250 | 0.8750 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| proposed_vs_candidate_no_context | objective_completion_support | 5 | 4 | 15 | 0.5208 | 0.5556 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 9 | 4 | 11 | 0.6042 | 0.6923 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | 3 | 6 | 15 | 0.4375 | 0.3333 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 6 | 0 | 18 | 0.6250 | 1.0000 |
| proposed_vs_candidate_no_context | context_overlap | 8 | 5 | 11 | 0.5625 | 0.6154 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 2 | 5 | 17 | 0.4375 | 0.2857 |
| proposed_vs_candidate_no_context | persona_style | 1 | 2 | 21 | 0.4792 | 0.3333 |
| proposed_vs_candidate_no_context | distinct1 | 6 | 4 | 14 | 0.5417 | 0.6000 |
| proposed_vs_candidate_no_context | length_score | 6 | 7 | 11 | 0.4792 | 0.4615 |
| proposed_vs_candidate_no_context | sentence_score | 2 | 1 | 21 | 0.5208 | 0.6667 |
| proposed_vs_candidate_no_context | overall_quality | 8 | 5 | 11 | 0.5625 | 0.6154 |
| controlled_vs_proposed_raw | context_relevance | 12 | 8 | 4 | 0.5833 | 0.6000 |
| controlled_vs_proposed_raw | persona_consistency | 7 | 7 | 10 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | naturalness | 14 | 6 | 4 | 0.6667 | 0.7000 |
| controlled_vs_proposed_raw | quest_state_correctness | 12 | 8 | 4 | 0.5833 | 0.6000 |
| controlled_vs_proposed_raw | lore_consistency | 8 | 3 | 13 | 0.6042 | 0.7273 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_vs_proposed_raw | objective_completion_support | 3 | 2 | 19 | 0.5208 | 0.6000 |
| controlled_vs_proposed_raw | gameplay_usefulness | 12 | 8 | 4 | 0.5833 | 0.6000 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 5 | 2 | 17 | 0.5625 | 0.7143 |
| controlled_vs_proposed_raw | context_keyword_coverage | 8 | 4 | 12 | 0.5833 | 0.6667 |
| controlled_vs_proposed_raw | context_overlap | 11 | 9 | 4 | 0.5417 | 0.5500 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 7 | 4 | 13 | 0.5625 | 0.6364 |
| controlled_vs_proposed_raw | persona_style | 2 | 5 | 17 | 0.4375 | 0.2857 |
| controlled_vs_proposed_raw | distinct1 | 10 | 9 | 5 | 0.5208 | 0.5263 |
| controlled_vs_proposed_raw | length_score | 12 | 8 | 4 | 0.5833 | 0.6000 |
| controlled_vs_proposed_raw | sentence_score | 3 | 0 | 21 | 0.5625 | 1.0000 |
| controlled_vs_proposed_raw | overall_quality | 14 | 6 | 4 | 0.6667 | 0.7000 |
| controlled_vs_candidate_no_context | context_relevance | 15 | 5 | 4 | 0.7083 | 0.7500 |
| controlled_vs_candidate_no_context | persona_consistency | 8 | 7 | 9 | 0.5208 | 0.5333 |
| controlled_vs_candidate_no_context | naturalness | 11 | 9 | 4 | 0.5417 | 0.5500 |
| controlled_vs_candidate_no_context | quest_state_correctness | 15 | 5 | 4 | 0.7083 | 0.7500 |
| controlled_vs_candidate_no_context | lore_consistency | 10 | 1 | 13 | 0.6875 | 0.9091 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_vs_candidate_no_context | objective_completion_support | 7 | 4 | 13 | 0.5625 | 0.6364 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 14 | 6 | 4 | 0.6667 | 0.7000 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 7 | 6 | 11 | 0.5208 | 0.5385 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 11 | 2 | 11 | 0.6875 | 0.8462 |
| controlled_vs_candidate_no_context | context_overlap | 15 | 5 | 4 | 0.7083 | 0.7500 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 7 | 7 | 10 | 0.5000 | 0.5000 |
| controlled_vs_candidate_no_context | persona_style | 2 | 6 | 16 | 0.4167 | 0.2500 |
| controlled_vs_candidate_no_context | distinct1 | 10 | 10 | 4 | 0.5000 | 0.5000 |
| controlled_vs_candidate_no_context | length_score | 11 | 9 | 4 | 0.5417 | 0.5500 |
| controlled_vs_candidate_no_context | sentence_score | 4 | 0 | 20 | 0.5833 | 1.0000 |
| controlled_vs_candidate_no_context | overall_quality | 14 | 6 | 4 | 0.6667 | 0.7000 |
| controlled_alt_vs_controlled_default | context_relevance | 6 | 7 | 11 | 0.4792 | 0.4615 |
| controlled_alt_vs_controlled_default | persona_consistency | 3 | 5 | 16 | 0.4583 | 0.3750 |
| controlled_alt_vs_controlled_default | naturalness | 3 | 10 | 11 | 0.3542 | 0.2308 |
| controlled_alt_vs_controlled_default | quest_state_correctness | 6 | 7 | 11 | 0.4792 | 0.4615 |
| controlled_alt_vs_controlled_default | lore_consistency | 6 | 5 | 13 | 0.5208 | 0.5455 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_alt_vs_controlled_default | objective_completion_support | 7 | 2 | 15 | 0.6042 | 0.7778 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | 6 | 7 | 11 | 0.4792 | 0.4615 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | 8 | 3 | 13 | 0.6042 | 0.7273 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 4 | 6 | 14 | 0.4583 | 0.4000 |
| controlled_alt_vs_controlled_default | context_overlap | 4 | 9 | 11 | 0.3958 | 0.3077 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 3 | 3 | 18 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | persona_style | 0 | 3 | 21 | 0.4375 | 0.0000 |
| controlled_alt_vs_controlled_default | distinct1 | 4 | 9 | 11 | 0.3958 | 0.3077 |
| controlled_alt_vs_controlled_default | length_score | 3 | 10 | 11 | 0.3542 | 0.2308 |
| controlled_alt_vs_controlled_default | sentence_score | 0 | 0 | 24 | 0.5000 | nan |
| controlled_alt_vs_controlled_default | overall_quality | 5 | 8 | 11 | 0.4375 | 0.3846 |
| controlled_alt_vs_proposed_raw | context_relevance | 12 | 8 | 4 | 0.5833 | 0.6000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 9 | 8 | 7 | 0.5208 | 0.5294 |
| controlled_alt_vs_proposed_raw | naturalness | 9 | 11 | 4 | 0.4583 | 0.4500 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 12 | 8 | 4 | 0.5833 | 0.6000 |
| controlled_alt_vs_proposed_raw | lore_consistency | 9 | 4 | 11 | 0.6042 | 0.6923 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_alt_vs_proposed_raw | objective_completion_support | 9 | 2 | 13 | 0.6458 | 0.8182 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 10 | 10 | 4 | 0.5000 | 0.5000 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | 10 | 2 | 12 | 0.6667 | 0.8333 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 7 | 5 | 12 | 0.5417 | 0.5833 |
| controlled_alt_vs_proposed_raw | context_overlap | 11 | 9 | 4 | 0.5417 | 0.5500 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 9 | 5 | 10 | 0.5833 | 0.6429 |
| controlled_alt_vs_proposed_raw | persona_style | 1 | 7 | 16 | 0.3750 | 0.1250 |
| controlled_alt_vs_proposed_raw | distinct1 | 8 | 11 | 5 | 0.4375 | 0.4211 |
| controlled_alt_vs_proposed_raw | length_score | 8 | 12 | 4 | 0.4167 | 0.4000 |
| controlled_alt_vs_proposed_raw | sentence_score | 3 | 0 | 21 | 0.5625 | 1.0000 |
| controlled_alt_vs_proposed_raw | overall_quality | 10 | 10 | 4 | 0.5000 | 0.5000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 12 | 7 | 5 | 0.6042 | 0.6316 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 7 | 8 | 9 | 0.4792 | 0.4667 |
| controlled_alt_vs_candidate_no_context | naturalness | 6 | 13 | 5 | 0.3542 | 0.3158 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 13 | 6 | 5 | 0.6458 | 0.6842 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 9 | 2 | 13 | 0.6458 | 0.8182 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_alt_vs_candidate_no_context | objective_completion_support | 10 | 2 | 12 | 0.6667 | 0.8333 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 14 | 5 | 5 | 0.6875 | 0.7368 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | 11 | 3 | 10 | 0.6667 | 0.7857 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 9 | 2 | 13 | 0.6458 | 0.8182 |
| controlled_alt_vs_candidate_no_context | context_overlap | 11 | 8 | 5 | 0.5625 | 0.5789 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 7 | 8 | 9 | 0.4792 | 0.4667 |
| controlled_alt_vs_candidate_no_context | persona_style | 0 | 7 | 17 | 0.3542 | 0.0000 |
| controlled_alt_vs_candidate_no_context | distinct1 | 9 | 10 | 5 | 0.4792 | 0.4737 |
| controlled_alt_vs_candidate_no_context | length_score | 8 | 11 | 5 | 0.4375 | 0.4211 |
| controlled_alt_vs_candidate_no_context | sentence_score | 4 | 0 | 20 | 0.5833 | 1.0000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 12 | 7 | 5 | 0.6042 | 0.6316 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.0417 | 0.3333 | 0.6667 |
| proposed_contextual_controlled_tuned | 0.0000 | 0.0000 | 0.1667 | 0.1250 | 0.8750 |
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
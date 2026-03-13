# Proposal Alignment Evaluation Report

- Run ID: `20260313T065023Z`
- Generated: `2026-03-13T06:56:09.151624+00:00`
- Scenarios: `artifacts\proposal_control_tuning\auto_tune\20260313T051946Z\train_runs\trial_008\seed_19\20260313T065023Z\scenarios.jsonl`
- Scenario count: `24`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_tuned`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0546 (0.0218, 0.0967) | 0.2334 (0.1865, 0.2814) | 0.8663 (0.8565, 0.8762) | 0.2780 (0.2494, 0.3112) | n/a |
| proposed_contextual_controlled_tuned | 0.0674 (0.0375, 0.1010) | 0.2169 (0.1744, 0.2611) | 0.8708 (0.8532, 0.8888) | 0.2778 (0.2508, 0.3065) | n/a |
| proposed_contextual | 0.0840 (0.0437, 0.1310) | 0.2293 (0.1726, 0.2908) | 0.8785 (0.8622, 0.8942) | 0.2919 (0.2574, 0.3260) | n/a |
| candidate_no_context | 0.0344 (0.0203, 0.0499) | 0.2419 (0.1829, 0.3067) | 0.8836 (0.8671, 0.9008) | 0.2746 (0.2506, 0.2996) | n/a |

## Game-facing Outcome Metrics (mean, 95% CI)
| Arm | Quest-state Correctness | Lore Consistency | Contradiction Safety | Objective Completion Support | Gameplay Usefulness | Time-pressure Acceptability |
|---|---:|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.1419 (0.1165, 0.1755) | 0.0167 (0.0008, 0.0381) | 1.0000 (1.0000, 1.0000) | 0.0864 (0.0626, 0.1094) | 0.2969 (0.2862, 0.3081) | 0.3047 (0.2891, 0.3193) |
| proposed_contextual_controlled_tuned | 0.1587 (0.1292, 0.1950) | 0.0225 (0.0068, 0.0417) | 1.0000 (1.0000, 1.0000) | 0.0853 (0.0609, 0.1090) | 0.3009 (0.2890, 0.3144) | 0.3074 (0.2883, 0.3247) |
| proposed_contextual | 0.1677 (0.1343, 0.2091) | 0.0307 (0.0087, 0.0560) | 1.0000 (1.0000, 1.0000) | 0.0577 (0.0353, 0.0826) | 0.2954 (0.2793, 0.3127) | 0.2974 (0.2852, 0.3106) |
| candidate_no_context | 0.1278 (0.1153, 0.1409) | 0.0070 (0.0000, 0.0188) | 1.0000 (1.0000, 1.0000) | 0.0712 (0.0469, 0.0945) | 0.2880 (0.2770, 0.2987) | 0.3045 (0.2908, 0.3189) |

- Multi-turn contradiction rate is reported as `1 - contradiction_safety` in row-level outputs.

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0495 | 1.4382 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0127 | -0.0523 |
| proposed_vs_candidate_no_context | naturalness | -0.0051 | -0.0058 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.0399 | 0.3124 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0238 | 3.4035 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | -0.0135 | -0.1901 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 0.0075 | 0.0259 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | -0.0070 | -0.0231 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0640 | 1.8370 |
| proposed_vs_candidate_no_context | context_overlap | 0.0157 | 0.4694 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0129 | -0.0769 |
| proposed_vs_candidate_no_context | persona_style | -0.0117 | -0.0217 |
| proposed_vs_candidate_no_context | distinct1 | 0.0103 | 0.0111 |
| proposed_vs_candidate_no_context | length_score | -0.0181 | -0.0319 |
| proposed_vs_candidate_no_context | sentence_score | -0.0562 | -0.0571 |
| proposed_vs_candidate_no_context | overall_quality | 0.0174 | 0.0632 |
| controlled_vs_proposed_raw | context_relevance | -0.0294 | -0.3501 |
| controlled_vs_proposed_raw | persona_consistency | 0.0041 | 0.0178 |
| controlled_vs_proposed_raw | naturalness | -0.0122 | -0.0139 |
| controlled_vs_proposed_raw | quest_state_correctness | -0.0258 | -0.1537 |
| controlled_vs_proposed_raw | lore_consistency | -0.0141 | -0.4577 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0287 | 0.4970 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0015 | 0.0050 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 0.0073 | 0.0246 |
| controlled_vs_proposed_raw | context_keyword_coverage | -0.0371 | -0.3755 |
| controlled_vs_proposed_raw | context_overlap | -0.0114 | -0.2312 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.0149 | 0.0962 |
| controlled_vs_proposed_raw | persona_style | -0.0391 | -0.0741 |
| controlled_vs_proposed_raw | distinct1 | -0.0286 | -0.0305 |
| controlled_vs_proposed_raw | length_score | -0.0319 | -0.0582 |
| controlled_vs_proposed_raw | sentence_score | 0.0562 | 0.0605 |
| controlled_vs_proposed_raw | overall_quality | -0.0139 | -0.0475 |
| controlled_vs_candidate_no_context | context_relevance | 0.0201 | 0.5845 |
| controlled_vs_candidate_no_context | persona_consistency | -0.0086 | -0.0354 |
| controlled_vs_candidate_no_context | naturalness | -0.0173 | -0.0196 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.0141 | 0.1107 |
| controlled_vs_candidate_no_context | lore_consistency | 0.0097 | 1.3882 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | 0.0151 | 0.2125 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0089 | 0.0311 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 0.0003 | 0.0009 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.0269 | 0.7717 |
| controlled_vs_candidate_no_context | context_overlap | 0.0043 | 0.1297 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.0020 | 0.0118 |
| controlled_vs_candidate_no_context | persona_style | -0.0508 | -0.0942 |
| controlled_vs_candidate_no_context | distinct1 | -0.0183 | -0.0197 |
| controlled_vs_candidate_no_context | length_score | -0.0500 | -0.0882 |
| controlled_vs_candidate_no_context | sentence_score | 0.0000 | 0.0000 |
| controlled_vs_candidate_no_context | overall_quality | 0.0035 | 0.0127 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0128 | 0.2351 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0165 | -0.0708 |
| controlled_alt_vs_controlled_default | naturalness | 0.0045 | 0.0052 |
| controlled_alt_vs_controlled_default | quest_state_correctness | 0.0167 | 0.1179 |
| controlled_alt_vs_controlled_default | lore_consistency | 0.0058 | 0.3471 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_controlled_default | objective_completion_support | -0.0010 | -0.0121 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | 0.0040 | 0.0133 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | 0.0027 | 0.0088 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0121 | 0.1963 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0145 | 0.3828 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0298 | -0.1754 |
| controlled_alt_vs_controlled_default | persona_style | 0.0365 | 0.0747 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0045 | 0.0049 |
| controlled_alt_vs_controlled_default | length_score | 0.0208 | 0.0403 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0146 | -0.0148 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0003 | -0.0009 |
| controlled_alt_vs_proposed_raw | context_relevance | -0.0166 | -0.1973 |
| controlled_alt_vs_proposed_raw | persona_consistency | -0.0124 | -0.0542 |
| controlled_alt_vs_proposed_raw | naturalness | -0.0077 | -0.0088 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | -0.0090 | -0.0539 |
| controlled_alt_vs_proposed_raw | lore_consistency | -0.0083 | -0.2694 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_proposed_raw | objective_completion_support | 0.0276 | 0.4789 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 0.0054 | 0.0184 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | 0.0100 | 0.0336 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | -0.0250 | -0.2529 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0031 | 0.0631 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | -0.0149 | -0.0962 |
| controlled_alt_vs_proposed_raw | persona_style | -0.0026 | -0.0049 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0241 | -0.0257 |
| controlled_alt_vs_proposed_raw | length_score | -0.0111 | -0.0203 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0417 | 0.0448 |
| controlled_alt_vs_proposed_raw | overall_quality | -0.0141 | -0.0484 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.0330 | 0.9570 |
| controlled_alt_vs_candidate_no_context | persona_consistency | -0.0251 | -0.1037 |
| controlled_alt_vs_candidate_no_context | naturalness | -0.0128 | -0.0145 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 0.0309 | 0.2416 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 0.0155 | 2.2172 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_candidate_no_context | objective_completion_support | 0.0141 | 0.1979 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 0.0129 | 0.0448 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | 0.0030 | 0.0097 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.0390 | 1.1196 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0188 | 0.5621 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | -0.0278 | -0.1657 |
| controlled_alt_vs_candidate_no_context | persona_style | -0.0143 | -0.0266 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0138 | -0.0149 |
| controlled_alt_vs_candidate_no_context | length_score | -0.0292 | -0.0515 |
| controlled_alt_vs_candidate_no_context | sentence_score | -0.0146 | -0.0148 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.0032 | 0.0118 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0495 | (0.0076, 0.0982) | 0.0080 | 0.0495 | (0.0176, 0.0832) | 0.0010 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0127 | (-0.0768, 0.0524) | 0.6593 | -0.0127 | (-0.0469, 0.0260) | 0.8077 |
| proposed_vs_candidate_no_context | naturalness | -0.0051 | (-0.0281, 0.0154) | 0.6710 | -0.0051 | (-0.0243, 0.0143) | 0.6873 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.0399 | (0.0057, 0.0788) | 0.0087 | 0.0399 | (0.0154, 0.0683) | 0.0003 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0238 | (-0.0016, 0.0517) | 0.0337 | 0.0238 | (-0.0041, 0.0539) | 0.0480 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | -0.0135 | (-0.0339, 0.0087) | 0.8900 | -0.0135 | (-0.0294, -0.0017) | 1.0000 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 0.0075 | (-0.0120, 0.0282) | 0.2183 | 0.0075 | (0.0003, 0.0180) | 0.0150 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | -0.0070 | (-0.0227, 0.0098) | 0.8053 | -0.0070 | (-0.0170, 0.0007) | 0.9410 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0640 | (0.0068, 0.1295) | 0.0153 | 0.0640 | (0.0189, 0.1145) | 0.0000 |
| proposed_vs_candidate_no_context | context_overlap | 0.0157 | (0.0040, 0.0296) | 0.0023 | 0.0157 | (0.0061, 0.0270) | 0.0000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0129 | (-0.0903, 0.0695) | 0.6403 | -0.0129 | (-0.0519, 0.0314) | 0.7450 |
| proposed_vs_candidate_no_context | persona_style | -0.0117 | (-0.0560, 0.0326) | 0.6983 | -0.0117 | (-0.0837, 0.0682) | 0.6380 |
| proposed_vs_candidate_no_context | distinct1 | 0.0103 | (-0.0087, 0.0304) | 0.1467 | 0.0103 | (-0.0037, 0.0285) | 0.0737 |
| proposed_vs_candidate_no_context | length_score | -0.0181 | (-0.1208, 0.0736) | 0.6410 | -0.0181 | (-0.1182, 0.0939) | 0.6647 |
| proposed_vs_candidate_no_context | sentence_score | -0.0563 | (-0.1396, 0.0146) | 0.9593 | -0.0563 | (-0.1819, 0.0269) | 0.8970 |
| proposed_vs_candidate_no_context | overall_quality | 0.0174 | (-0.0107, 0.0482) | 0.1173 | 0.0174 | (0.0063, 0.0303) | 0.0000 |
| controlled_vs_proposed_raw | context_relevance | -0.0294 | (-0.0609, 0.0014) | 0.9690 | -0.0294 | (-0.0540, -0.0079) | 1.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.0041 | (-0.0501, 0.0549) | 0.4383 | 0.0041 | (-0.0317, 0.0401) | 0.3950 |
| controlled_vs_proposed_raw | naturalness | -0.0122 | (-0.0265, 0.0015) | 0.9593 | -0.0122 | (-0.0341, 0.0066) | 0.8987 |
| controlled_vs_proposed_raw | quest_state_correctness | -0.0258 | (-0.0535, 0.0008) | 0.9723 | -0.0258 | (-0.0455, -0.0084) | 1.0000 |
| controlled_vs_proposed_raw | lore_consistency | -0.0141 | (-0.0281, -0.0028) | 0.9997 | -0.0141 | (-0.0264, -0.0051) | 1.0000 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0287 | (0.0031, 0.0540) | 0.0123 | 0.0287 | (-0.0030, 0.0734) | 0.0507 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0015 | (-0.0115, 0.0171) | 0.4313 | 0.0015 | (-0.0099, 0.0151) | 0.4233 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 0.0073 | (-0.0060, 0.0213) | 0.1493 | 0.0073 | (-0.0060, 0.0251) | 0.1677 |
| controlled_vs_proposed_raw | context_keyword_coverage | -0.0371 | (-0.0795, 0.0015) | 0.9673 | -0.0371 | (-0.0636, -0.0106) | 1.0000 |
| controlled_vs_proposed_raw | context_overlap | -0.0114 | (-0.0270, 0.0024) | 0.9433 | -0.0114 | (-0.0276, -0.0013) | 0.9897 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.0149 | (-0.0486, 0.0724) | 0.3207 | 0.0149 | (-0.0119, 0.0512) | 0.1513 |
| controlled_vs_proposed_raw | persona_style | -0.0391 | (-0.0807, 0.0013) | 0.9740 | -0.0391 | (-0.1108, 0.0000) | 1.0000 |
| controlled_vs_proposed_raw | distinct1 | -0.0286 | (-0.0500, -0.0086) | 0.9983 | -0.0286 | (-0.0731, 0.0011) | 0.9513 |
| controlled_vs_proposed_raw | length_score | -0.0319 | (-0.0903, 0.0278) | 0.8617 | -0.0319 | (-0.1136, 0.0530) | 0.8063 |
| controlled_vs_proposed_raw | sentence_score | 0.0563 | (-0.0146, 0.1375) | 0.0820 | 0.0563 | (-0.0269, 0.1682) | 0.1467 |
| controlled_vs_proposed_raw | overall_quality | -0.0139 | (-0.0444, 0.0142) | 0.8180 | -0.0139 | (-0.0321, 0.0056) | 0.9297 |
| controlled_vs_candidate_no_context | context_relevance | 0.0201 | (-0.0147, 0.0642) | 0.1620 | 0.0201 | (-0.0174, 0.0596) | 0.1343 |
| controlled_vs_candidate_no_context | persona_consistency | -0.0086 | (-0.0502, 0.0337) | 0.6510 | -0.0086 | (-0.0505, 0.0461) | 0.6307 |
| controlled_vs_candidate_no_context | naturalness | -0.0173 | (-0.0319, -0.0043) | 0.9977 | -0.0173 | (-0.0257, -0.0069) | 1.0000 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.0141 | (-0.0147, 0.0498) | 0.1993 | 0.0141 | (-0.0098, 0.0440) | 0.1277 |
| controlled_vs_candidate_no_context | lore_consistency | 0.0097 | (-0.0109, 0.0338) | 0.2020 | 0.0097 | (-0.0119, 0.0294) | 0.1903 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | 0.0151 | (-0.0093, 0.0416) | 0.1183 | 0.0151 | (-0.0096, 0.0517) | 0.1540 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0089 | (-0.0048, 0.0223) | 0.1023 | 0.0089 | (-0.0016, 0.0236) | 0.0597 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 0.0003 | (-0.0173, 0.0167) | 0.4783 | 0.0003 | (-0.0141, 0.0147) | 0.4797 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.0269 | (-0.0205, 0.0879) | 0.1807 | 0.0269 | (-0.0210, 0.0818) | 0.1357 |
| controlled_vs_candidate_no_context | context_overlap | 0.0043 | (-0.0058, 0.0155) | 0.2080 | 0.0043 | (-0.0037, 0.0117) | 0.1410 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.0020 | (-0.0496, 0.0536) | 0.4597 | 0.0020 | (-0.0412, 0.0607) | 0.4920 |
| controlled_vs_candidate_no_context | persona_style | -0.0508 | (-0.0859, -0.0182) | 1.0000 | -0.0508 | (-0.1127, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | distinct1 | -0.0183 | (-0.0345, -0.0044) | 0.9973 | -0.0183 | (-0.0477, -0.0015) | 0.9880 |
| controlled_vs_candidate_no_context | length_score | -0.0500 | (-0.1097, 0.0056) | 0.9610 | -0.0500 | (-0.1090, -0.0000) | 0.9793 |
| controlled_vs_candidate_no_context | sentence_score | 0.0000 | (-0.0437, 0.0437) | 0.6537 | 0.0000 | (-0.0477, 0.0375) | 0.6360 |
| controlled_vs_candidate_no_context | overall_quality | 0.0035 | (-0.0164, 0.0255) | 0.3620 | 0.0035 | (-0.0094, 0.0219) | 0.3630 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0128 | (-0.0286, 0.0494) | 0.2353 | 0.0128 | (-0.0175, 0.0542) | 0.2537 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0165 | (-0.0553, 0.0167) | 0.8247 | -0.0165 | (-0.0643, 0.0189) | 0.7730 |
| controlled_alt_vs_controlled_default | naturalness | 0.0045 | (-0.0099, 0.0195) | 0.2677 | 0.0045 | (-0.0140, 0.0208) | 0.2740 |
| controlled_alt_vs_controlled_default | quest_state_correctness | 0.0167 | (-0.0191, 0.0486) | 0.1567 | 0.0167 | (-0.0106, 0.0551) | 0.1397 |
| controlled_alt_vs_controlled_default | lore_consistency | 0.0058 | (-0.0156, 0.0278) | 0.2807 | 0.0058 | (-0.0054, 0.0249) | 0.2440 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | objective_completion_support | -0.0010 | (-0.0185, 0.0175) | 0.5540 | -0.0010 | (-0.0237, 0.0152) | 0.5360 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | 0.0040 | (-0.0034, 0.0116) | 0.1517 | 0.0040 | (0.0013, 0.0067) | 0.0000 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | 0.0027 | (-0.0102, 0.0153) | 0.3467 | 0.0027 | (-0.0089, 0.0185) | 0.3467 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0121 | (-0.0375, 0.0549) | 0.2653 | 0.0121 | (-0.0314, 0.0620) | 0.3073 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0145 | (-0.0013, 0.0330) | 0.0400 | 0.0145 | (-0.0009, 0.0371) | 0.0350 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0298 | (-0.0833, 0.0159) | 0.8963 | -0.0298 | (-0.1050, 0.0128) | 0.8793 |
| controlled_alt_vs_controlled_default | persona_style | 0.0365 | (0.0091, 0.0729) | 0.0090 | 0.0365 | (0.0000, 0.0781) | 0.0857 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0045 | (-0.0152, 0.0244) | 0.3330 | 0.0045 | (-0.0224, 0.0379) | 0.3833 |
| controlled_alt_vs_controlled_default | length_score | 0.0208 | (-0.0236, 0.0709) | 0.2013 | 0.0208 | (-0.0303, 0.0783) | 0.2360 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0146 | (-0.0583, 0.0292) | 0.8167 | -0.0146 | (-0.0477, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0003 | (-0.0271, 0.0239) | 0.4833 | -0.0003 | (-0.0254, 0.0192) | 0.5127 |
| controlled_alt_vs_proposed_raw | context_relevance | -0.0166 | (-0.0646, 0.0330) | 0.7343 | -0.0166 | (-0.0385, 0.0042) | 0.9373 |
| controlled_alt_vs_proposed_raw | persona_consistency | -0.0124 | (-0.0751, 0.0477) | 0.6750 | -0.0124 | (-0.0926, 0.0365) | 0.6537 |
| controlled_alt_vs_proposed_raw | naturalness | -0.0077 | (-0.0285, 0.0152) | 0.7487 | -0.0077 | (-0.0417, 0.0232) | 0.7247 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | -0.0090 | (-0.0502, 0.0337) | 0.6473 | -0.0090 | (-0.0307, 0.0116) | 0.8183 |
| controlled_alt_vs_proposed_raw | lore_consistency | -0.0083 | (-0.0363, 0.0177) | 0.7390 | -0.0083 | (-0.0243, 0.0120) | 0.8030 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_proposed_raw | objective_completion_support | 0.0276 | (0.0083, 0.0485) | 0.0013 | 0.0276 | (0.0018, 0.0636) | 0.0160 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 0.0054 | (-0.0093, 0.0229) | 0.2553 | 0.0054 | (-0.0044, 0.0191) | 0.1677 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | 0.0100 | (-0.0044, 0.0242) | 0.0967 | 0.0100 | (-0.0053, 0.0278) | 0.0837 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | -0.0250 | (-0.0902, 0.0364) | 0.7770 | -0.0250 | (-0.0545, 0.0038) | 0.9580 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0031 | (-0.0146, 0.0247) | 0.3823 | 0.0031 | (-0.0064, 0.0151) | 0.2963 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | -0.0149 | (-0.0853, 0.0556) | 0.6683 | -0.0149 | (-0.1119, 0.0455) | 0.6470 |
| controlled_alt_vs_proposed_raw | persona_style | -0.0026 | (-0.0508, 0.0560) | 0.5693 | -0.0026 | (-0.0341, 0.0201) | 0.6483 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0241 | (-0.0446, -0.0051) | 0.9943 | -0.0241 | (-0.0449, -0.0081) | 1.0000 |
| controlled_alt_vs_proposed_raw | length_score | -0.0111 | (-0.0972, 0.0847) | 0.6010 | -0.0111 | (-0.1364, 0.1273) | 0.6067 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0417 | (-0.0292, 0.1147) | 0.1657 | 0.0417 | (-0.0269, 0.1250) | 0.1833 |
| controlled_alt_vs_proposed_raw | overall_quality | -0.0141 | (-0.0537, 0.0243) | 0.7557 | -0.0141 | (-0.0493, 0.0042) | 0.8043 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.0330 | (0.0052, 0.0642) | 0.0097 | 0.0330 | (0.0144, 0.0508) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | -0.0251 | (-0.0808, 0.0255) | 0.8187 | -0.0251 | (-0.0992, 0.0284) | 0.7517 |
| controlled_alt_vs_candidate_no_context | naturalness | -0.0128 | (-0.0308, 0.0059) | 0.9140 | -0.0128 | (-0.0300, 0.0032) | 0.9397 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 0.0309 | (0.0034, 0.0607) | 0.0137 | 0.0309 | (0.0121, 0.0499) | 0.0000 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 0.0155 | (0.0039, 0.0302) | 0.0000 | 0.0155 | (0.0019, 0.0299) | 0.0097 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_candidate_no_context | objective_completion_support | 0.0141 | (-0.0101, 0.0406) | 0.1370 | 0.0141 | (-0.0054, 0.0444) | 0.1280 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 0.0129 | (-0.0007, 0.0273) | 0.0313 | 0.0129 | (0.0024, 0.0273) | 0.0100 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | 0.0030 | (-0.0145, 0.0210) | 0.3793 | 0.0030 | (-0.0095, 0.0191) | 0.3573 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.0390 | (0.0034, 0.0769) | 0.0153 | 0.0390 | (0.0169, 0.0599) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0188 | (0.0021, 0.0384) | 0.0140 | 0.0188 | (0.0056, 0.0372) | 0.0003 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | -0.0278 | (-0.1012, 0.0377) | 0.7840 | -0.0278 | (-0.1333, 0.0417) | 0.7537 |
| controlled_alt_vs_candidate_no_context | persona_style | -0.0143 | (-0.0612, 0.0326) | 0.7307 | -0.0143 | (-0.0636, 0.0341) | 0.7407 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0138 | (-0.0326, 0.0066) | 0.9130 | -0.0138 | (-0.0289, -0.0039) | 1.0000 |
| controlled_alt_vs_candidate_no_context | length_score | -0.0292 | (-0.0945, 0.0458) | 0.8023 | -0.0292 | (-0.0910, 0.0367) | 0.8103 |
| controlled_alt_vs_candidate_no_context | sentence_score | -0.0146 | (-0.0583, 0.0292) | 0.8120 | -0.0146 | (-0.0700, 0.0269) | 0.8083 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.0032 | (-0.0232, 0.0294) | 0.3820 | 0.0032 | (-0.0287, 0.0290) | 0.3830 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 12 | 4 | 8 | 0.6667 | 0.7500 |
| proposed_vs_candidate_no_context | persona_consistency | 4 | 8 | 12 | 0.4167 | 0.3333 |
| proposed_vs_candidate_no_context | naturalness | 6 | 10 | 8 | 0.4167 | 0.3750 |
| proposed_vs_candidate_no_context | quest_state_correctness | 12 | 4 | 8 | 0.6667 | 0.7500 |
| proposed_vs_candidate_no_context | lore_consistency | 8 | 1 | 15 | 0.6458 | 0.8889 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| proposed_vs_candidate_no_context | objective_completion_support | 2 | 8 | 14 | 0.3750 | 0.2000 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 9 | 7 | 8 | 0.5417 | 0.5625 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | 4 | 8 | 12 | 0.4167 | 0.3333 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 8 | 3 | 13 | 0.6042 | 0.7273 |
| proposed_vs_candidate_no_context | context_overlap | 12 | 4 | 8 | 0.6667 | 0.7500 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 4 | 8 | 12 | 0.4167 | 0.3333 |
| proposed_vs_candidate_no_context | persona_style | 2 | 4 | 18 | 0.4583 | 0.3333 |
| proposed_vs_candidate_no_context | distinct1 | 8 | 8 | 8 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | length_score | 6 | 7 | 11 | 0.4792 | 0.4615 |
| proposed_vs_candidate_no_context | sentence_score | 1 | 4 | 19 | 0.4375 | 0.2000 |
| proposed_vs_candidate_no_context | overall_quality | 9 | 7 | 8 | 0.5417 | 0.5625 |
| controlled_vs_proposed_raw | context_relevance | 7 | 11 | 6 | 0.4167 | 0.3889 |
| controlled_vs_proposed_raw | persona_consistency | 8 | 6 | 10 | 0.5417 | 0.5714 |
| controlled_vs_proposed_raw | naturalness | 5 | 13 | 6 | 0.3333 | 0.2778 |
| controlled_vs_proposed_raw | quest_state_correctness | 6 | 12 | 6 | 0.3750 | 0.3333 |
| controlled_vs_proposed_raw | lore_consistency | 1 | 7 | 16 | 0.3750 | 0.1250 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_vs_proposed_raw | objective_completion_support | 8 | 2 | 14 | 0.6250 | 0.8000 |
| controlled_vs_proposed_raw | gameplay_usefulness | 8 | 10 | 6 | 0.4583 | 0.4444 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 7 | 3 | 14 | 0.5833 | 0.7000 |
| controlled_vs_proposed_raw | context_keyword_coverage | 3 | 8 | 13 | 0.3958 | 0.2727 |
| controlled_vs_proposed_raw | context_overlap | 8 | 10 | 6 | 0.4583 | 0.4444 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 8 | 5 | 11 | 0.5625 | 0.6154 |
| controlled_vs_proposed_raw | persona_style | 2 | 7 | 15 | 0.3958 | 0.2222 |
| controlled_vs_proposed_raw | distinct1 | 8 | 10 | 6 | 0.4583 | 0.4444 |
| controlled_vs_proposed_raw | length_score | 6 | 11 | 7 | 0.3958 | 0.3529 |
| controlled_vs_proposed_raw | sentence_score | 4 | 1 | 19 | 0.5625 | 0.8000 |
| controlled_vs_proposed_raw | overall_quality | 7 | 11 | 6 | 0.4167 | 0.3889 |
| controlled_vs_candidate_no_context | context_relevance | 7 | 9 | 8 | 0.4583 | 0.4375 |
| controlled_vs_candidate_no_context | persona_consistency | 7 | 8 | 9 | 0.4792 | 0.4667 |
| controlled_vs_candidate_no_context | naturalness | 5 | 11 | 8 | 0.3750 | 0.3125 |
| controlled_vs_candidate_no_context | quest_state_correctness | 7 | 9 | 8 | 0.4583 | 0.4375 |
| controlled_vs_candidate_no_context | lore_consistency | 3 | 1 | 20 | 0.5417 | 0.7500 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_vs_candidate_no_context | objective_completion_support | 5 | 4 | 15 | 0.5208 | 0.5556 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 10 | 6 | 8 | 0.5833 | 0.6250 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 6 | 6 | 12 | 0.5000 | 0.5000 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 4 | 4 | 16 | 0.5000 | 0.5000 |
| controlled_vs_candidate_no_context | context_overlap | 9 | 7 | 8 | 0.5417 | 0.5625 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 7 | 6 | 11 | 0.5208 | 0.5385 |
| controlled_vs_candidate_no_context | persona_style | 0 | 7 | 17 | 0.3542 | 0.0000 |
| controlled_vs_candidate_no_context | distinct1 | 4 | 11 | 9 | 0.3542 | 0.2667 |
| controlled_vs_candidate_no_context | length_score | 5 | 11 | 8 | 0.3750 | 0.3125 |
| controlled_vs_candidate_no_context | sentence_score | 1 | 1 | 22 | 0.5000 | 0.5000 |
| controlled_vs_candidate_no_context | overall_quality | 9 | 7 | 8 | 0.5417 | 0.5625 |
| controlled_alt_vs_controlled_default | context_relevance | 7 | 2 | 15 | 0.6042 | 0.7778 |
| controlled_alt_vs_controlled_default | persona_consistency | 3 | 4 | 17 | 0.4792 | 0.4286 |
| controlled_alt_vs_controlled_default | naturalness | 4 | 5 | 15 | 0.4792 | 0.4444 |
| controlled_alt_vs_controlled_default | quest_state_correctness | 8 | 1 | 15 | 0.6458 | 0.8889 |
| controlled_alt_vs_controlled_default | lore_consistency | 5 | 2 | 17 | 0.5625 | 0.7143 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_alt_vs_controlled_default | objective_completion_support | 3 | 4 | 17 | 0.4792 | 0.4286 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | 6 | 3 | 15 | 0.5625 | 0.6667 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | 5 | 4 | 15 | 0.5208 | 0.5556 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 5 | 1 | 18 | 0.5833 | 0.8333 |
| controlled_alt_vs_controlled_default | context_overlap | 6 | 3 | 15 | 0.5625 | 0.6667 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 2 | 4 | 18 | 0.4583 | 0.3333 |
| controlled_alt_vs_controlled_default | persona_style | 4 | 0 | 20 | 0.5833 | 1.0000 |
| controlled_alt_vs_controlled_default | distinct1 | 4 | 5 | 15 | 0.4792 | 0.4444 |
| controlled_alt_vs_controlled_default | length_score | 5 | 4 | 15 | 0.5208 | 0.5556 |
| controlled_alt_vs_controlled_default | sentence_score | 1 | 2 | 21 | 0.4792 | 0.3333 |
| controlled_alt_vs_controlled_default | overall_quality | 6 | 3 | 15 | 0.5625 | 0.6667 |
| controlled_alt_vs_proposed_raw | context_relevance | 7 | 11 | 6 | 0.4167 | 0.3889 |
| controlled_alt_vs_proposed_raw | persona_consistency | 7 | 8 | 9 | 0.4792 | 0.4667 |
| controlled_alt_vs_proposed_raw | naturalness | 6 | 12 | 6 | 0.3750 | 0.3333 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 6 | 12 | 6 | 0.3750 | 0.3333 |
| controlled_alt_vs_proposed_raw | lore_consistency | 4 | 7 | 13 | 0.4375 | 0.3636 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_alt_vs_proposed_raw | objective_completion_support | 8 | 3 | 13 | 0.6042 | 0.7273 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 8 | 10 | 6 | 0.4583 | 0.4444 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | 7 | 5 | 12 | 0.5417 | 0.5833 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 5 | 8 | 11 | 0.4375 | 0.3846 |
| controlled_alt_vs_proposed_raw | context_overlap | 7 | 11 | 6 | 0.4167 | 0.3889 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 7 | 7 | 10 | 0.5000 | 0.5000 |
| controlled_alt_vs_proposed_raw | persona_style | 3 | 5 | 16 | 0.4583 | 0.3750 |
| controlled_alt_vs_proposed_raw | distinct1 | 6 | 11 | 7 | 0.3958 | 0.3529 |
| controlled_alt_vs_proposed_raw | length_score | 7 | 10 | 7 | 0.4375 | 0.4118 |
| controlled_alt_vs_proposed_raw | sentence_score | 3 | 1 | 20 | 0.5417 | 0.7500 |
| controlled_alt_vs_proposed_raw | overall_quality | 6 | 12 | 6 | 0.3750 | 0.3333 |
| controlled_alt_vs_candidate_no_context | context_relevance | 11 | 6 | 7 | 0.6042 | 0.6471 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 7 | 7 | 10 | 0.5000 | 0.5000 |
| controlled_alt_vs_candidate_no_context | naturalness | 4 | 13 | 7 | 0.3125 | 0.2353 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 11 | 6 | 7 | 0.6042 | 0.6471 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 6 | 0 | 18 | 0.6250 | 1.0000 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_alt_vs_candidate_no_context | objective_completion_support | 6 | 5 | 13 | 0.5208 | 0.5455 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 12 | 5 | 7 | 0.6458 | 0.7059 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | 6 | 7 | 11 | 0.4792 | 0.4615 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 8 | 3 | 13 | 0.6042 | 0.7273 |
| controlled_alt_vs_candidate_no_context | context_overlap | 12 | 5 | 7 | 0.6458 | 0.7059 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 6 | 6 | 12 | 0.5000 | 0.5000 |
| controlled_alt_vs_candidate_no_context | persona_style | 3 | 5 | 16 | 0.4583 | 0.3750 |
| controlled_alt_vs_candidate_no_context | distinct1 | 5 | 11 | 8 | 0.3750 | 0.3125 |
| controlled_alt_vs_candidate_no_context | length_score | 5 | 12 | 7 | 0.3542 | 0.2941 |
| controlled_alt_vs_candidate_no_context | sentence_score | 1 | 2 | 21 | 0.4792 | 0.3333 |
| controlled_alt_vs_candidate_no_context | overall_quality | 10 | 7 | 7 | 0.5625 | 0.5882 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.1250 | 0.1667 | 0.8333 |
| proposed_contextual_controlled_tuned | 0.0000 | 0.0000 | 0.1250 | 0.0833 | 0.8333 |
| proposed_contextual | 0.0000 | 0.0000 | 0.4583 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 |

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
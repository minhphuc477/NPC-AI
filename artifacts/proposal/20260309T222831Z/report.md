# Proposal Alignment Evaluation Report

- Run ID: `20260309T222831Z`
- Generated: `2026-03-09T22:35:40.546457+00:00`
- Scenarios: `artifacts\proposal\20260309T222831Z\scenarios.jsonl`
- Scenario count: `144`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off
- `baseline_no_context`: model `phi3:mini`, dynamic_context=off, response_control=off
- `baseline_no_context_phi3_latest`: model `phi3:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2707 (0.2573, 0.2854) | 0.2883 (0.2717, 0.3054) | 0.8953 (0.8844, 0.9048) | 0.4037 (0.3946, 0.4121) | n/a |
| proposed_contextual | 0.0782 (0.0622, 0.0937) | 0.1565 (0.1375, 0.1783) | 0.8197 (0.8078, 0.8318) | 0.2500 (0.2374, 0.2634) | n/a |
| candidate_no_context | 0.0242 (0.0195, 0.0294) | 0.1620 (0.1422, 0.1848) | 0.8234 (0.8109, 0.8369) | 0.2279 (0.2182, 0.2382) | n/a |
| baseline_no_context | 0.0317 (0.0258, 0.0376) | 0.1464 (0.1323, 0.1615) | 0.8875 (0.8777, 0.8973) | 0.2379 (0.2313, 0.2450) | n/a |
| baseline_no_context_phi3_latest | 0.0342 (0.0279, 0.0409) | 0.1564 (0.1419, 0.1706) | 0.8918 (0.8829, 0.9010) | 0.2435 (0.2367, 0.2507) | n/a |

## Game-facing Outcome Metrics (mean, 95% CI)
| Arm | Quest-state Correctness | Lore Consistency | Contradiction Safety | Objective Completion Support | Gameplay Usefulness | Time-pressure Acceptability |
|---|---:|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.3236 (0.3110, 0.3371) | 0.1929 (0.1786, 0.2083) | 0.7500 (0.6806, 0.8194) | 0.0688 (0.0579, 0.0801) | 0.3570 (0.3495, 0.3649) | 0.2871 (0.2776, 0.2962) |
| proposed_contextual | 0.1664 (0.1541, 0.1801) | 0.0428 (0.0335, 0.0520) | 1.0000 (1.0000, 1.0000) | 0.0457 (0.0389, 0.0532) | 0.2572 (0.2490, 0.2658) | 0.2763 (0.2712, 0.2813) |
| candidate_no_context | 0.1208 (0.1161, 0.1257) | 0.0120 (0.0094, 0.0147) | 1.0000 (1.0000, 1.0000) | 0.0468 (0.0412, 0.0527) | 0.2440 (0.2381, 0.2498) | 0.2817 (0.2772, 0.2862) |
| baseline_no_context | 0.1270 (0.1218, 0.1322) | 0.0121 (0.0087, 0.0160) | 1.0000 (1.0000, 1.0000) | 0.0373 (0.0312, 0.0439) | 0.2708 (0.2651, 0.2765) | 0.2854 (0.2797, 0.2911) |
| baseline_no_context_phi3_latest | 0.1292 (0.1235, 0.1351) | 0.0137 (0.0099, 0.0178) | 1.0000 (1.0000, 1.0000) | 0.0428 (0.0345, 0.0516) | 0.2757 (0.2698, 0.2818) | 0.2882 (0.2824, 0.2939) |

- Multi-turn contradiction rate is reported as `1 - contradiction_safety` in row-level outputs.

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0541 | 2.2391 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0056 | -0.0343 |
| proposed_vs_candidate_no_context | naturalness | -0.0037 | -0.0045 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.0457 | 0.3780 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0308 | 2.5702 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | -0.0011 | -0.0234 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 0.0132 | 0.0542 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | -0.0054 | -0.0190 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0701 | 3.5066 |
| proposed_vs_candidate_no_context | context_overlap | 0.0166 | 0.4915 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0091 | -0.1265 |
| proposed_vs_candidate_no_context | persona_style | 0.0087 | 0.0166 |
| proposed_vs_candidate_no_context | distinct1 | 0.0018 | 0.0019 |
| proposed_vs_candidate_no_context | length_score | -0.0083 | -0.0251 |
| proposed_vs_candidate_no_context | sentence_score | -0.0267 | -0.0333 |
| proposed_vs_candidate_no_context | overall_quality | 0.0221 | 0.0971 |
| proposed_vs_baseline_no_context | context_relevance | 0.0465 | 1.4679 |
| proposed_vs_baseline_no_context | persona_consistency | 0.0101 | 0.0690 |
| proposed_vs_baseline_no_context | naturalness | -0.0678 | -0.0763 |
| proposed_vs_baseline_no_context | quest_state_correctness | 0.0395 | 0.3108 |
| proposed_vs_baseline_no_context | lore_consistency | 0.0307 | 2.5418 |
| proposed_vs_baseline_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| proposed_vs_baseline_no_context | objective_completion_support | 0.0085 | 0.2268 |
| proposed_vs_baseline_no_context | gameplay_usefulness | -0.0135 | -0.0500 |
| proposed_vs_baseline_no_context | time_pressure_acceptability | -0.0091 | -0.0318 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0594 | 1.9279 |
| proposed_vs_baseline_no_context | context_overlap | 0.0166 | 0.4916 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 0.0189 | 0.4291 |
| proposed_vs_baseline_no_context | persona_style | -0.0252 | -0.0453 |
| proposed_vs_baseline_no_context | distinct1 | -0.0381 | -0.0388 |
| proposed_vs_baseline_no_context | length_score | -0.2125 | -0.3960 |
| proposed_vs_baseline_no_context | sentence_score | -0.0997 | -0.1137 |
| proposed_vs_baseline_no_context | overall_quality | 0.0121 | 0.0509 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 0.0441 | 1.2896 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | 0.0001 | 0.0003 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | -0.0721 | -0.0808 |
| proposed_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.0372 | 0.2879 |
| proposed_vs_baseline_no_context_phi3_latest | lore_consistency | 0.0292 | 2.1355 |
| proposed_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.0029 | 0.0681 |
| proposed_vs_baseline_no_context_phi3_latest | gameplay_usefulness | -0.0185 | -0.0670 |
| proposed_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | -0.0119 | -0.0413 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.0570 | 1.7171 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 0.0140 | 0.3830 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.0082 | 0.1490 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | -0.0324 | -0.0576 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | -0.0330 | -0.0338 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | -0.2336 | -0.4188 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | -0.1212 | -0.1350 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | 0.0065 | 0.0267 |
| controlled_vs_proposed_raw | context_relevance | 0.1924 | 2.4592 |
| controlled_vs_proposed_raw | persona_consistency | 0.1319 | 0.8427 |
| controlled_vs_proposed_raw | naturalness | 0.0756 | 0.0922 |
| controlled_vs_proposed_raw | quest_state_correctness | 0.1571 | 0.9442 |
| controlled_vs_proposed_raw | lore_consistency | 0.1501 | 3.5056 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | -0.2500 | -0.2500 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0231 | 0.5044 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0998 | 0.3881 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 0.0108 | 0.0390 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2533 | 2.8104 |
| controlled_vs_proposed_raw | context_overlap | 0.0503 | 0.9959 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1578 | 2.5055 |
| controlled_vs_proposed_raw | persona_style | 0.0280 | 0.0527 |
| controlled_vs_proposed_raw | distinct1 | -0.0125 | -0.0132 |
| controlled_vs_proposed_raw | length_score | 0.3144 | 0.9700 |
| controlled_vs_proposed_raw | sentence_score | 0.1774 | 0.2284 |
| controlled_vs_proposed_raw | overall_quality | 0.1537 | 0.6146 |
| controlled_vs_candidate_no_context | context_relevance | 0.2465 | 10.2045 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1263 | 0.7794 |
| controlled_vs_candidate_no_context | naturalness | 0.0719 | 0.0873 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.2028 | 1.6792 |
| controlled_vs_candidate_no_context | lore_consistency | 0.1809 | 15.0859 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | -0.2500 | -0.2500 |
| controlled_vs_candidate_no_context | objective_completion_support | 0.0220 | 0.4692 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.1130 | 0.4633 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 0.0054 | 0.0193 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3235 | 16.1720 |
| controlled_vs_candidate_no_context | context_overlap | 0.0669 | 1.9769 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1487 | 2.0619 |
| controlled_vs_candidate_no_context | persona_style | 0.0366 | 0.0702 |
| controlled_vs_candidate_no_context | distinct1 | -0.0107 | -0.0113 |
| controlled_vs_candidate_no_context | length_score | 0.3060 | 0.9206 |
| controlled_vs_candidate_no_context | sentence_score | 0.1507 | 0.1876 |
| controlled_vs_candidate_no_context | overall_quality | 0.1758 | 0.7714 |
| controlled_vs_baseline_no_context | context_relevance | 0.2390 | 7.5369 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1420 | 0.9698 |
| controlled_vs_baseline_no_context | naturalness | 0.0078 | 0.0088 |
| controlled_vs_baseline_no_context | quest_state_correctness | 0.1966 | 1.5485 |
| controlled_vs_baseline_no_context | lore_consistency | 0.1808 | 14.9580 |
| controlled_vs_baseline_no_context | multi_turn_contradiction_safety | -0.2500 | -0.2500 |
| controlled_vs_baseline_no_context | objective_completion_support | 0.0315 | 0.8455 |
| controlled_vs_baseline_no_context | gameplay_usefulness | 0.0863 | 0.3187 |
| controlled_vs_baseline_no_context | time_pressure_acceptability | 0.0017 | 0.0060 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3127 | 10.1565 |
| controlled_vs_baseline_no_context | context_overlap | 0.0669 | 1.9772 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1768 | 4.0098 |
| controlled_vs_baseline_no_context | persona_style | 0.0028 | 0.0050 |
| controlled_vs_baseline_no_context | distinct1 | -0.0505 | -0.0515 |
| controlled_vs_baseline_no_context | length_score | 0.1019 | 0.1898 |
| controlled_vs_baseline_no_context | sentence_score | 0.0778 | 0.0887 |
| controlled_vs_baseline_no_context | overall_quality | 0.1658 | 0.6968 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.2365 | 6.9201 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1319 | 0.8433 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | 0.0035 | 0.0039 |
| controlled_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.1944 | 1.5040 |
| controlled_vs_baseline_no_context_phi3_latest | lore_consistency | 0.1793 | 13.1276 |
| controlled_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | -0.2500 | -0.2500 |
| controlled_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.0260 | 0.6068 |
| controlled_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 0.0814 | 0.2951 |
| controlled_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | -0.0011 | -0.0038 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.3103 | 9.3533 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0643 | 1.7604 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1660 | 3.0277 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | -0.0045 | -0.0079 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0455 | -0.0466 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | 0.0808 | 0.1449 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.0563 | 0.0626 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1602 | 0.6578 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2390 | 7.5369 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1420 | 0.9698 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 0.0078 | 0.0088 |
| proposed_contextual_controlled_vs_baseline_no_context | quest_state_correctness | 0.1966 | 1.5485 |
| proposed_contextual_controlled_vs_baseline_no_context | lore_consistency | 0.1808 | 14.9580 |
| proposed_contextual_controlled_vs_baseline_no_context | multi_turn_contradiction_safety | -0.2500 | -0.2500 |
| proposed_contextual_controlled_vs_baseline_no_context | objective_completion_support | 0.0315 | 0.8455 |
| proposed_contextual_controlled_vs_baseline_no_context | gameplay_usefulness | 0.0863 | 0.3187 |
| proposed_contextual_controlled_vs_baseline_no_context | time_pressure_acceptability | 0.0017 | 0.0060 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3127 | 10.1565 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0669 | 1.9772 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1768 | 4.0098 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0028 | 0.0050 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0505 | -0.0515 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.1019 | 0.1898 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0778 | 0.0887 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1658 | 0.6968 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.2365 | 6.9201 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1319 | 0.8433 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | 0.0035 | 0.0039 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.1944 | 1.5040 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | lore_consistency | 0.1793 | 13.1276 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | -0.2500 | -0.2500 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.0260 | 0.6068 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 0.0814 | 0.2951 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | -0.0011 | -0.0038 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.3103 | 9.3533 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0643 | 1.7604 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1660 | 3.0277 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | -0.0045 | -0.0079 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0455 | -0.0466 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | 0.0808 | 0.1449 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.0563 | 0.0626 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1602 | 0.6578 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0541 | (0.0378, 0.0716) | 0.0000 | 0.0541 | (0.0305, 0.0818) | 0.0000 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0056 | (-0.0246, 0.0137) | 0.7187 | -0.0056 | (-0.0176, 0.0095) | 0.7960 |
| proposed_vs_candidate_no_context | naturalness | -0.0037 | (-0.0175, 0.0106) | 0.6880 | -0.0037 | (-0.0223, 0.0142) | 0.6417 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.0457 | (0.0314, 0.0600) | 0.0000 | 0.0457 | (0.0226, 0.0686) | 0.0000 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0308 | (0.0216, 0.0411) | 0.0000 | 0.0308 | (0.0154, 0.0464) | 0.0000 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | -0.0011 | (-0.0089, 0.0070) | 0.6080 | -0.0011 | (-0.0090, 0.0073) | 0.6077 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 0.0132 | (0.0042, 0.0221) | 0.0003 | 0.0132 | (0.0011, 0.0238) | 0.0170 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | -0.0054 | (-0.0111, 0.0005) | 0.9600 | -0.0054 | (-0.0124, 0.0004) | 0.9603 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0701 | (0.0496, 0.0934) | 0.0000 | 0.0701 | (0.0373, 0.1066) | 0.0000 |
| proposed_vs_candidate_no_context | context_overlap | 0.0166 | (0.0110, 0.0224) | 0.0000 | 0.0166 | (0.0102, 0.0227) | 0.0000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0091 | (-0.0328, 0.0126) | 0.7887 | -0.0091 | (-0.0232, 0.0083) | 0.8590 |
| proposed_vs_candidate_no_context | persona_style | 0.0087 | (-0.0025, 0.0220) | 0.0667 | 0.0087 | (0.0017, 0.0175) | 0.0063 |
| proposed_vs_candidate_no_context | distinct1 | 0.0018 | (-0.0046, 0.0082) | 0.2913 | 0.0018 | (-0.0061, 0.0105) | 0.3670 |
| proposed_vs_candidate_no_context | length_score | -0.0083 | (-0.0600, 0.0452) | 0.6317 | -0.0083 | (-0.0785, 0.0593) | 0.5907 |
| proposed_vs_candidate_no_context | sentence_score | -0.0267 | (-0.0580, 0.0052) | 0.9490 | -0.0267 | (-0.0486, -0.0024) | 0.9913 |
| proposed_vs_candidate_no_context | overall_quality | 0.0221 | (0.0094, 0.0352) | 0.0000 | 0.0221 | (0.0094, 0.0350) | 0.0003 |
| proposed_vs_baseline_no_context | context_relevance | 0.0465 | (0.0316, 0.0627) | 0.0000 | 0.0465 | (0.0216, 0.0721) | 0.0000 |
| proposed_vs_baseline_no_context | persona_consistency | 0.0101 | (-0.0098, 0.0307) | 0.1573 | 0.0101 | (-0.0234, 0.0452) | 0.2913 |
| proposed_vs_baseline_no_context | naturalness | -0.0678 | (-0.0826, -0.0525) | 1.0000 | -0.0678 | (-0.0929, -0.0455) | 1.0000 |
| proposed_vs_baseline_no_context | quest_state_correctness | 0.0395 | (0.0266, 0.0533) | 0.0000 | 0.0395 | (0.0168, 0.0609) | 0.0000 |
| proposed_vs_baseline_no_context | lore_consistency | 0.0307 | (0.0210, 0.0403) | 0.0000 | 0.0307 | (0.0165, 0.0456) | 0.0000 |
| proposed_vs_baseline_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_baseline_no_context | objective_completion_support | 0.0085 | (-0.0005, 0.0174) | 0.0343 | 0.0085 | (0.0016, 0.0178) | 0.0033 |
| proposed_vs_baseline_no_context | gameplay_usefulness | -0.0135 | (-0.0233, -0.0039) | 0.9967 | -0.0135 | (-0.0254, -0.0025) | 0.9930 |
| proposed_vs_baseline_no_context | time_pressure_acceptability | -0.0091 | (-0.0158, -0.0022) | 0.9937 | -0.0091 | (-0.0162, -0.0025) | 1.0000 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0594 | (0.0392, 0.0800) | 0.0000 | 0.0594 | (0.0270, 0.0921) | 0.0000 |
| proposed_vs_baseline_no_context | context_overlap | 0.0166 | (0.0109, 0.0223) | 0.0000 | 0.0166 | (0.0087, 0.0243) | 0.0000 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 0.0189 | (-0.0052, 0.0438) | 0.0640 | 0.0189 | (-0.0157, 0.0567) | 0.1603 |
| proposed_vs_baseline_no_context | persona_style | -0.0252 | (-0.0485, -0.0031) | 0.9873 | -0.0252 | (-0.0674, 0.0098) | 0.9103 |
| proposed_vs_baseline_no_context | distinct1 | -0.0381 | (-0.0447, -0.0310) | 1.0000 | -0.0381 | (-0.0493, -0.0264) | 1.0000 |
| proposed_vs_baseline_no_context | length_score | -0.2125 | (-0.2667, -0.1549) | 1.0000 | -0.2125 | (-0.2942, -0.1359) | 1.0000 |
| proposed_vs_baseline_no_context | sentence_score | -0.0997 | (-0.1361, -0.0635) | 1.0000 | -0.0997 | (-0.1487, -0.0510) | 1.0000 |
| proposed_vs_baseline_no_context | overall_quality | 0.0121 | (-0.0013, 0.0259) | 0.0387 | 0.0121 | (-0.0100, 0.0334) | 0.1483 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 0.0441 | (0.0286, 0.0599) | 0.0000 | 0.0441 | (0.0188, 0.0692) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | 0.0001 | (-0.0207, 0.0243) | 0.4943 | 0.0001 | (-0.0439, 0.0492) | 0.5113 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | -0.0721 | (-0.0863, -0.0577) | 1.0000 | -0.0721 | (-0.0963, -0.0492) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.0372 | (0.0239, 0.0513) | 0.0000 | 0.0372 | (0.0131, 0.0601) | 0.0010 |
| proposed_vs_baseline_no_context_phi3_latest | lore_consistency | 0.0292 | (0.0193, 0.0396) | 0.0000 | 0.0292 | (0.0150, 0.0442) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.0029 | (-0.0073, 0.0138) | 0.2883 | 0.0029 | (-0.0033, 0.0090) | 0.1853 |
| proposed_vs_baseline_no_context_phi3_latest | gameplay_usefulness | -0.0185 | (-0.0285, -0.0088) | 1.0000 | -0.0185 | (-0.0315, -0.0067) | 0.9997 |
| proposed_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | -0.0119 | (-0.0188, -0.0054) | 1.0000 | -0.0119 | (-0.0200, -0.0048) | 0.9997 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.0570 | (0.0359, 0.0793) | 0.0000 | 0.0570 | (0.0237, 0.0910) | 0.0003 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 0.0140 | (0.0087, 0.0196) | 0.0000 | 0.0140 | (0.0067, 0.0208) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.0082 | (-0.0169, 0.0339) | 0.2740 | 0.0082 | (-0.0420, 0.0622) | 0.3893 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | -0.0324 | (-0.0575, -0.0087) | 0.9990 | -0.0324 | (-0.0885, 0.0137) | 0.9027 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | -0.0330 | (-0.0404, -0.0256) | 1.0000 | -0.0330 | (-0.0458, -0.0188) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | -0.2336 | (-0.2910, -0.1778) | 1.0000 | -0.2336 | (-0.3169, -0.1532) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | -0.1212 | (-0.1580, -0.0847) | 1.0000 | -0.1212 | (-0.1629, -0.0799) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | 0.0065 | (-0.0067, 0.0206) | 0.1687 | 0.0065 | (-0.0193, 0.0309) | 0.3127 |
| controlled_vs_proposed_raw | context_relevance | 0.1924 | (0.1725, 0.2107) | 0.0000 | 0.1924 | (0.1690, 0.2165) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.1319 | (0.1089, 0.1537) | 0.0000 | 0.1319 | (0.0988, 0.1663) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0756 | (0.0573, 0.0935) | 0.0000 | 0.0756 | (0.0430, 0.1098) | 0.0000 |
| controlled_vs_proposed_raw | quest_state_correctness | 0.1571 | (0.1412, 0.1741) | 0.0000 | 0.1571 | (0.1340, 0.1809) | 0.0000 |
| controlled_vs_proposed_raw | lore_consistency | 0.1501 | (0.1341, 0.1647) | 0.0000 | 0.1501 | (0.1297, 0.1708) | 0.0000 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | -0.2500 | (-0.3196, -0.1806) | 1.0000 | -0.2500 | (-0.6250, 0.0000) | 1.0000 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0231 | (0.0105, 0.0355) | 0.0003 | 0.0231 | (0.0017, 0.0461) | 0.0183 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0998 | (0.0868, 0.1122) | 0.0000 | 0.0998 | (0.0804, 0.1202) | 0.0000 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 0.0108 | (0.0008, 0.0207) | 0.0160 | 0.0108 | (-0.0023, 0.0247) | 0.0587 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2533 | (0.2286, 0.2775) | 0.0000 | 0.2533 | (0.2234, 0.2842) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0503 | (0.0439, 0.0565) | 0.0000 | 0.0503 | (0.0433, 0.0577) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1578 | (0.1314, 0.1828) | 0.0000 | 0.1578 | (0.1199, 0.1994) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.0280 | (0.0081, 0.0516) | 0.0017 | 0.0280 | (-0.0025, 0.0654) | 0.0520 |
| controlled_vs_proposed_raw | distinct1 | -0.0125 | (-0.0199, -0.0047) | 1.0000 | -0.0125 | (-0.0276, 0.0048) | 0.9257 |
| controlled_vs_proposed_raw | length_score | 0.3144 | (0.2398, 0.3901) | 0.0000 | 0.3144 | (0.1843, 0.4317) | 0.0000 |
| controlled_vs_proposed_raw | sentence_score | 0.1774 | (0.1417, 0.2135) | 0.0000 | 0.1774 | (0.1354, 0.2236) | 0.0000 |
| controlled_vs_proposed_raw | overall_quality | 0.1537 | (0.1393, 0.1672) | 0.0000 | 0.1537 | (0.1324, 0.1757) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2465 | (0.2317, 0.2618) | 0.0000 | 0.2465 | (0.2222, 0.2720) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1263 | (0.1021, 0.1487) | 0.0000 | 0.1263 | (0.0878, 0.1632) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.0719 | (0.0530, 0.0896) | 0.0000 | 0.0719 | (0.0384, 0.1103) | 0.0000 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.2028 | (0.1902, 0.2159) | 0.0000 | 0.2028 | (0.1826, 0.2260) | 0.0000 |
| controlled_vs_candidate_no_context | lore_consistency | 0.1809 | (0.1660, 0.1955) | 0.0000 | 0.1809 | (0.1533, 0.2119) | 0.0000 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | -0.2500 | (-0.3194, -0.1806) | 1.0000 | -0.2500 | (-0.5000, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | 0.0220 | (0.0110, 0.0340) | 0.0000 | 0.0220 | (0.0056, 0.0427) | 0.0020 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.1130 | (0.1024, 0.1235) | 0.0000 | 0.1130 | (0.0951, 0.1306) | 0.0000 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 0.0054 | (-0.0043, 0.0150) | 0.1287 | 0.0054 | (-0.0082, 0.0203) | 0.2327 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3235 | (0.3041, 0.3436) | 0.0000 | 0.3235 | (0.2893, 0.3576) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0669 | (0.0619, 0.0720) | 0.0000 | 0.0669 | (0.0602, 0.0721) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1487 | (0.1200, 0.1764) | 0.0000 | 0.1487 | (0.1073, 0.1901) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.0366 | (0.0151, 0.0612) | 0.0000 | 0.0366 | (0.0012, 0.0821) | 0.0210 |
| controlled_vs_candidate_no_context | distinct1 | -0.0107 | (-0.0191, -0.0023) | 0.9947 | -0.0107 | (-0.0249, 0.0056) | 0.9070 |
| controlled_vs_candidate_no_context | length_score | 0.3060 | (0.2384, 0.3748) | 0.0000 | 0.3060 | (0.1859, 0.4401) | 0.0000 |
| controlled_vs_candidate_no_context | sentence_score | 0.1507 | (0.1153, 0.1851) | 0.0000 | 0.1507 | (0.0986, 0.2042) | 0.0000 |
| controlled_vs_candidate_no_context | overall_quality | 0.1758 | (0.1633, 0.1886) | 0.0000 | 0.1758 | (0.1519, 0.1994) | 0.0000 |
| controlled_vs_baseline_no_context | context_relevance | 0.2390 | (0.2256, 0.2531) | 0.0000 | 0.2390 | (0.2207, 0.2590) | 0.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1420 | (0.1232, 0.1601) | 0.0000 | 0.1420 | (0.1078, 0.1674) | 0.0000 |
| controlled_vs_baseline_no_context | naturalness | 0.0078 | (-0.0067, 0.0215) | 0.1390 | 0.0078 | (-0.0114, 0.0256) | 0.2083 |
| controlled_vs_baseline_no_context | quest_state_correctness | 0.1966 | (0.1847, 0.2088) | 0.0000 | 0.1966 | (0.1812, 0.2140) | 0.0000 |
| controlled_vs_baseline_no_context | lore_consistency | 0.1808 | (0.1668, 0.1957) | 0.0000 | 0.1808 | (0.1547, 0.2125) | 0.0000 |
| controlled_vs_baseline_no_context | multi_turn_contradiction_safety | -0.2500 | (-0.3194, -0.1806) | 1.0000 | -0.2500 | (-0.6250, 0.0000) | 1.0000 |
| controlled_vs_baseline_no_context | objective_completion_support | 0.0315 | (0.0198, 0.0429) | 0.0000 | 0.0315 | (0.0097, 0.0541) | 0.0017 |
| controlled_vs_baseline_no_context | gameplay_usefulness | 0.0863 | (0.0774, 0.0955) | 0.0000 | 0.0863 | (0.0758, 0.0986) | 0.0000 |
| controlled_vs_baseline_no_context | time_pressure_acceptability | 0.0017 | (-0.0082, 0.0112) | 0.3600 | 0.0017 | (-0.0107, 0.0159) | 0.4313 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3127 | (0.2946, 0.3315) | 0.0000 | 0.3127 | (0.2879, 0.3402) | 0.0000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0669 | (0.0619, 0.0720) | 0.0000 | 0.0669 | (0.0604, 0.0730) | 0.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1768 | (0.1539, 0.1994) | 0.0000 | 0.1768 | (0.1346, 0.2081) | 0.0000 |
| controlled_vs_baseline_no_context | persona_style | 0.0028 | (-0.0096, 0.0162) | 0.3367 | 0.0028 | (-0.0061, 0.0123) | 0.2723 |
| controlled_vs_baseline_no_context | distinct1 | -0.0505 | (-0.0570, -0.0439) | 1.0000 | -0.0505 | (-0.0590, -0.0412) | 1.0000 |
| controlled_vs_baseline_no_context | length_score | 0.1019 | (0.0380, 0.1639) | 0.0013 | 0.1019 | (0.0208, 0.1708) | 0.0067 |
| controlled_vs_baseline_no_context | sentence_score | 0.0778 | (0.0448, 0.1108) | 0.0000 | 0.0778 | (0.0267, 0.1281) | 0.0010 |
| controlled_vs_baseline_no_context | overall_quality | 0.1658 | (0.1563, 0.1758) | 0.0000 | 0.1658 | (0.1506, 0.1795) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.2365 | (0.2227, 0.2516) | 0.0000 | 0.2365 | (0.2203, 0.2535) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1319 | (0.1120, 0.1527) | 0.0000 | 0.1319 | (0.0803, 0.1687) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | 0.0035 | (-0.0114, 0.0178) | 0.3090 | 0.0035 | (-0.0127, 0.0202) | 0.3370 |
| controlled_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.1944 | (0.1817, 0.2075) | 0.0000 | 0.1944 | (0.1804, 0.2099) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | lore_consistency | 0.1793 | (0.1651, 0.1942) | 0.0000 | 0.1793 | (0.1532, 0.2091) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | -0.2500 | (-0.3194, -0.1806) | 1.0000 | -0.2500 | (-0.5000, 0.0000) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.0260 | (0.0143, 0.0371) | 0.0000 | 0.0260 | (0.0079, 0.0451) | 0.0010 |
| controlled_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 0.0814 | (0.0713, 0.0909) | 0.0000 | 0.0814 | (0.0706, 0.0934) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | -0.0011 | (-0.0111, 0.0088) | 0.5770 | -0.0011 | (-0.0106, 0.0100) | 0.5807 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.3103 | (0.2918, 0.3301) | 0.0000 | 0.3103 | (0.2874, 0.3343) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0643 | (0.0590, 0.0698) | 0.0000 | 0.0643 | (0.0584, 0.0695) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1660 | (0.1394, 0.1900) | 0.0000 | 0.1660 | (0.1021, 0.2099) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | -0.0045 | (-0.0197, 0.0099) | 0.7137 | -0.0045 | (-0.0351, 0.0178) | 0.6017 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0455 | (-0.0525, -0.0385) | 1.0000 | -0.0455 | (-0.0544, -0.0355) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | 0.0808 | (0.0146, 0.1451) | 0.0080 | 0.0808 | (0.0171, 0.1461) | 0.0040 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.0562 | (0.0250, 0.0878) | 0.0000 | 0.0562 | (0.0052, 0.1003) | 0.0153 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1602 | (0.1496, 0.1707) | 0.0000 | 0.1602 | (0.1376, 0.1763) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2390 | (0.2255, 0.2524) | 0.0000 | 0.2390 | (0.2209, 0.2586) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1420 | (0.1227, 0.1597) | 0.0000 | 0.1420 | (0.1086, 0.1691) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 0.0078 | (-0.0061, 0.0222) | 0.1303 | 0.0078 | (-0.0125, 0.0259) | 0.2060 |
| proposed_contextual_controlled_vs_baseline_no_context | quest_state_correctness | 0.1966 | (0.1848, 0.2093) | 0.0000 | 0.1966 | (0.1812, 0.2151) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | lore_consistency | 0.1808 | (0.1660, 0.1966) | 0.0000 | 0.1808 | (0.1550, 0.2100) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | multi_turn_contradiction_safety | -0.2500 | (-0.3194, -0.1806) | 1.0000 | -0.2500 | (-0.5000, 0.0000) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | objective_completion_support | 0.0315 | (0.0202, 0.0433) | 0.0000 | 0.0315 | (0.0107, 0.0537) | 0.0007 |
| proposed_contextual_controlled_vs_baseline_no_context | gameplay_usefulness | 0.0863 | (0.0773, 0.0958) | 0.0000 | 0.0863 | (0.0759, 0.0988) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | time_pressure_acceptability | 0.0017 | (-0.0081, 0.0114) | 0.3790 | 0.0017 | (-0.0112, 0.0157) | 0.4157 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3127 | (0.2952, 0.3305) | 0.0000 | 0.3127 | (0.2870, 0.3399) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0669 | (0.0618, 0.0719) | 0.0000 | 0.0669 | (0.0603, 0.0728) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1768 | (0.1531, 0.1978) | 0.0000 | 0.1768 | (0.1343, 0.2077) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0028 | (-0.0097, 0.0159) | 0.3287 | 0.0028 | (-0.0062, 0.0120) | 0.3003 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0505 | (-0.0570, -0.0441) | 1.0000 | -0.0505 | (-0.0591, -0.0411) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.1019 | (0.0417, 0.1644) | 0.0000 | 0.1019 | (0.0208, 0.1738) | 0.0083 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0778 | (0.0462, 0.1108) | 0.0000 | 0.0778 | (0.0243, 0.1264) | 0.0010 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1658 | (0.1558, 0.1755) | 0.0000 | 0.1658 | (0.1498, 0.1798) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.2365 | (0.2226, 0.2515) | 0.0000 | 0.2365 | (0.2206, 0.2533) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1319 | (0.1120, 0.1516) | 0.0000 | 0.1319 | (0.0800, 0.1690) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | 0.0035 | (-0.0109, 0.0179) | 0.3077 | 0.0035 | (-0.0137, 0.0206) | 0.3530 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.1944 | (0.1817, 0.2081) | 0.0000 | 0.1944 | (0.1807, 0.2098) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | lore_consistency | 0.1793 | (0.1657, 0.1931) | 0.0000 | 0.1793 | (0.1534, 0.2076) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | -0.2500 | (-0.3194, -0.1806) | 1.0000 | -0.2500 | (-0.6250, 0.0000) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.0260 | (0.0146, 0.0378) | 0.0000 | 0.0260 | (0.0079, 0.0446) | 0.0003 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 0.0814 | (0.0708, 0.0916) | 0.0000 | 0.0814 | (0.0698, 0.0932) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | -0.0011 | (-0.0113, 0.0087) | 0.5833 | -0.0011 | (-0.0107, 0.0099) | 0.6010 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.3103 | (0.2905, 0.3301) | 0.0000 | 0.3103 | (0.2879, 0.3340) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0643 | (0.0590, 0.0699) | 0.0000 | 0.0643 | (0.0584, 0.0695) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1660 | (0.1411, 0.1904) | 0.0000 | 0.1660 | (0.1039, 0.2100) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | -0.0045 | (-0.0201, 0.0096) | 0.7220 | -0.0045 | (-0.0357, 0.0178) | 0.5833 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0455 | (-0.0523, -0.0381) | 1.0000 | -0.0455 | (-0.0545, -0.0358) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | 0.0808 | (0.0127, 0.1461) | 0.0110 | 0.0808 | (0.0155, 0.1465) | 0.0050 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.0562 | (0.0257, 0.0865) | 0.0000 | 0.0562 | (0.0097, 0.1003) | 0.0147 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1602 | (0.1492, 0.1711) | 0.0000 | 0.1602 | (0.1374, 0.1769) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 68 | 31 | 45 | 0.6285 | 0.6869 |
| proposed_vs_candidate_no_context | persona_consistency | 25 | 27 | 92 | 0.4931 | 0.4808 |
| proposed_vs_candidate_no_context | naturalness | 45 | 53 | 46 | 0.4722 | 0.4592 |
| proposed_vs_candidate_no_context | quest_state_correctness | 67 | 32 | 45 | 0.6215 | 0.6768 |
| proposed_vs_candidate_no_context | lore_consistency | 66 | 14 | 64 | 0.6806 | 0.8250 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 144 | 0.5000 | nan |
| proposed_vs_candidate_no_context | objective_completion_support | 33 | 38 | 73 | 0.4826 | 0.4648 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 57 | 42 | 45 | 0.5521 | 0.5758 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | 35 | 56 | 53 | 0.4271 | 0.3846 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 55 | 15 | 74 | 0.6389 | 0.7857 |
| proposed_vs_candidate_no_context | context_overlap | 69 | 30 | 45 | 0.6354 | 0.6970 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 19 | 23 | 102 | 0.4861 | 0.4524 |
| proposed_vs_candidate_no_context | persona_style | 9 | 6 | 129 | 0.5104 | 0.6000 |
| proposed_vs_candidate_no_context | distinct1 | 45 | 44 | 55 | 0.5035 | 0.5056 |
| proposed_vs_candidate_no_context | length_score | 40 | 57 | 47 | 0.4410 | 0.4124 |
| proposed_vs_candidate_no_context | sentence_score | 17 | 27 | 100 | 0.4653 | 0.3864 |
| proposed_vs_candidate_no_context | overall_quality | 64 | 35 | 45 | 0.6007 | 0.6465 |
| proposed_vs_baseline_no_context | context_relevance | 96 | 47 | 1 | 0.6701 | 0.6713 |
| proposed_vs_baseline_no_context | persona_consistency | 30 | 29 | 85 | 0.5035 | 0.5085 |
| proposed_vs_baseline_no_context | naturalness | 30 | 114 | 0 | 0.2083 | 0.2083 |
| proposed_vs_baseline_no_context | quest_state_correctness | 94 | 49 | 1 | 0.6562 | 0.6573 |
| proposed_vs_baseline_no_context | lore_consistency | 98 | 30 | 16 | 0.7361 | 0.7656 |
| proposed_vs_baseline_no_context | multi_turn_contradiction_safety | 0 | 0 | 144 | 0.5000 | nan |
| proposed_vs_baseline_no_context | objective_completion_support | 64 | 40 | 40 | 0.5833 | 0.6154 |
| proposed_vs_baseline_no_context | gameplay_usefulness | 55 | 89 | 0 | 0.3819 | 0.3819 |
| proposed_vs_baseline_no_context | time_pressure_acceptability | 64 | 61 | 19 | 0.5104 | 0.5120 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 54 | 18 | 72 | 0.6250 | 0.7500 |
| proposed_vs_baseline_no_context | context_overlap | 93 | 49 | 2 | 0.6528 | 0.6549 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 27 | 17 | 100 | 0.5347 | 0.6136 |
| proposed_vs_baseline_no_context | persona_style | 6 | 17 | 121 | 0.4618 | 0.2609 |
| proposed_vs_baseline_no_context | distinct1 | 22 | 106 | 16 | 0.2083 | 0.1719 |
| proposed_vs_baseline_no_context | length_score | 30 | 111 | 3 | 0.2188 | 0.2128 |
| proposed_vs_baseline_no_context | sentence_score | 15 | 55 | 74 | 0.3611 | 0.2143 |
| proposed_vs_baseline_no_context | overall_quality | 68 | 76 | 0 | 0.4722 | 0.4722 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 94 | 49 | 1 | 0.6562 | 0.6573 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | 27 | 43 | 74 | 0.4444 | 0.3857 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | 31 | 112 | 1 | 0.2188 | 0.2168 |
| proposed_vs_baseline_no_context_phi3_latest | quest_state_correctness | 91 | 52 | 1 | 0.6354 | 0.6364 |
| proposed_vs_baseline_no_context_phi3_latest | lore_consistency | 92 | 33 | 19 | 0.7049 | 0.7360 |
| proposed_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0 | 0 | 144 | 0.5000 | nan |
| proposed_vs_baseline_no_context_phi3_latest | objective_completion_support | 63 | 40 | 41 | 0.5799 | 0.6117 |
| proposed_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 49 | 95 | 0 | 0.3403 | 0.3403 |
| proposed_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 52 | 67 | 25 | 0.4479 | 0.4370 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 54 | 20 | 70 | 0.6181 | 0.7297 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 94 | 49 | 1 | 0.6562 | 0.6573 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 23 | 28 | 93 | 0.4826 | 0.4510 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | 8 | 19 | 117 | 0.4618 | 0.2963 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | 28 | 98 | 18 | 0.2569 | 0.2222 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | 33 | 106 | 5 | 0.2465 | 0.2374 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | 12 | 61 | 71 | 0.3299 | 0.1644 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | 59 | 85 | 0 | 0.4097 | 0.4097 |
| controlled_vs_proposed_raw | context_relevance | 136 | 7 | 1 | 0.9479 | 0.9510 |
| controlled_vs_proposed_raw | persona_consistency | 112 | 12 | 20 | 0.8472 | 0.9032 |
| controlled_vs_proposed_raw | naturalness | 106 | 37 | 1 | 0.7396 | 0.7413 |
| controlled_vs_proposed_raw | quest_state_correctness | 132 | 12 | 0 | 0.9167 | 0.9167 |
| controlled_vs_proposed_raw | lore_consistency | 133 | 8 | 3 | 0.9340 | 0.9433 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0 | 36 | 108 | 0.3750 | 0.0000 |
| controlled_vs_proposed_raw | objective_completion_support | 67 | 50 | 27 | 0.5590 | 0.5726 |
| controlled_vs_proposed_raw | gameplay_usefulness | 124 | 20 | 0 | 0.8611 | 0.8611 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 74 | 64 | 6 | 0.5347 | 0.5362 |
| controlled_vs_proposed_raw | context_keyword_coverage | 127 | 5 | 12 | 0.9236 | 0.9621 |
| controlled_vs_proposed_raw | context_overlap | 130 | 13 | 1 | 0.9062 | 0.9091 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 112 | 10 | 22 | 0.8542 | 0.9180 |
| controlled_vs_proposed_raw | persona_style | 17 | 6 | 121 | 0.5382 | 0.7391 |
| controlled_vs_proposed_raw | distinct1 | 65 | 76 | 3 | 0.4618 | 0.4610 |
| controlled_vs_proposed_raw | length_score | 104 | 38 | 2 | 0.7292 | 0.7324 |
| controlled_vs_proposed_raw | sentence_score | 81 | 8 | 55 | 0.7535 | 0.9101 |
| controlled_vs_proposed_raw | overall_quality | 136 | 8 | 0 | 0.9444 | 0.9444 |
| controlled_vs_candidate_no_context | context_relevance | 143 | 1 | 0 | 0.9931 | 0.9931 |
| controlled_vs_candidate_no_context | persona_consistency | 106 | 14 | 24 | 0.8194 | 0.8833 |
| controlled_vs_candidate_no_context | naturalness | 103 | 41 | 0 | 0.7153 | 0.7153 |
| controlled_vs_candidate_no_context | quest_state_correctness | 143 | 1 | 0 | 0.9931 | 0.9931 |
| controlled_vs_candidate_no_context | lore_consistency | 138 | 2 | 4 | 0.9722 | 0.9857 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 36 | 108 | 0.3750 | 0.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | 66 | 48 | 30 | 0.5625 | 0.5789 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 134 | 10 | 0 | 0.9306 | 0.9306 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 71 | 65 | 8 | 0.5208 | 0.5221 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 143 | 1 | 0 | 0.9931 | 0.9931 |
| controlled_vs_candidate_no_context | context_overlap | 142 | 2 | 0 | 0.9861 | 0.9861 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 105 | 12 | 27 | 0.8229 | 0.8974 |
| controlled_vs_candidate_no_context | persona_style | 20 | 6 | 118 | 0.5486 | 0.7692 |
| controlled_vs_candidate_no_context | distinct1 | 70 | 73 | 1 | 0.4896 | 0.4895 |
| controlled_vs_candidate_no_context | length_score | 109 | 34 | 1 | 0.7604 | 0.7622 |
| controlled_vs_candidate_no_context | sentence_score | 72 | 10 | 62 | 0.7153 | 0.8780 |
| controlled_vs_candidate_no_context | overall_quality | 139 | 5 | 0 | 0.9653 | 0.9653 |
| controlled_vs_baseline_no_context | context_relevance | 144 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 120 | 7 | 17 | 0.8924 | 0.9449 |
| controlled_vs_baseline_no_context | naturalness | 82 | 62 | 0 | 0.5694 | 0.5694 |
| controlled_vs_baseline_no_context | quest_state_correctness | 144 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | lore_consistency | 138 | 3 | 3 | 0.9688 | 0.9787 |
| controlled_vs_baseline_no_context | multi_turn_contradiction_safety | 0 | 36 | 108 | 0.3750 | 0.0000 |
| controlled_vs_baseline_no_context | objective_completion_support | 73 | 38 | 33 | 0.6215 | 0.6577 |
| controlled_vs_baseline_no_context | gameplay_usefulness | 135 | 9 | 0 | 0.9375 | 0.9375 |
| controlled_vs_baseline_no_context | time_pressure_acceptability | 69 | 62 | 13 | 0.5243 | 0.5267 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 144 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | context_overlap | 143 | 1 | 0 | 0.9931 | 0.9931 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 120 | 7 | 17 | 0.8924 | 0.9449 |
| controlled_vs_baseline_no_context | persona_style | 9 | 12 | 123 | 0.4896 | 0.4286 |
| controlled_vs_baseline_no_context | distinct1 | 13 | 129 | 2 | 0.0972 | 0.0915 |
| controlled_vs_baseline_no_context | length_score | 92 | 51 | 1 | 0.6424 | 0.6434 |
| controlled_vs_baseline_no_context | sentence_score | 41 | 10 | 93 | 0.6076 | 0.8039 |
| controlled_vs_baseline_no_context | overall_quality | 144 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 143 | 0 | 1 | 0.9965 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 114 | 11 | 19 | 0.8576 | 0.9120 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | 84 | 59 | 1 | 0.5868 | 0.5874 |
| controlled_vs_baseline_no_context_phi3_latest | quest_state_correctness | 143 | 0 | 1 | 0.9965 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | lore_consistency | 139 | 1 | 4 | 0.9792 | 0.9929 |
| controlled_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0 | 36 | 108 | 0.3750 | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | objective_completion_support | 68 | 41 | 35 | 0.5938 | 0.6239 |
| controlled_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 125 | 19 | 0 | 0.8681 | 0.8681 |
| controlled_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 63 | 69 | 12 | 0.4792 | 0.4773 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 143 | 0 | 1 | 0.9965 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 141 | 2 | 1 | 0.9826 | 0.9860 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 113 | 9 | 22 | 0.8611 | 0.9262 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | 12 | 11 | 121 | 0.5035 | 0.5217 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | 26 | 116 | 2 | 0.1875 | 0.1831 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | 93 | 49 | 2 | 0.6528 | 0.6549 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 34 | 10 | 100 | 0.5833 | 0.7727 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 142 | 2 | 0 | 0.9861 | 0.9861 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 144 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 120 | 7 | 17 | 0.8924 | 0.9449 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 82 | 62 | 0 | 0.5694 | 0.5694 |
| proposed_contextual_controlled_vs_baseline_no_context | quest_state_correctness | 144 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | lore_consistency | 138 | 3 | 3 | 0.9688 | 0.9787 |
| proposed_contextual_controlled_vs_baseline_no_context | multi_turn_contradiction_safety | 0 | 36 | 108 | 0.3750 | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | objective_completion_support | 73 | 38 | 33 | 0.6215 | 0.6577 |
| proposed_contextual_controlled_vs_baseline_no_context | gameplay_usefulness | 135 | 9 | 0 | 0.9375 | 0.9375 |
| proposed_contextual_controlled_vs_baseline_no_context | time_pressure_acceptability | 69 | 62 | 13 | 0.5243 | 0.5267 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 144 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 143 | 1 | 0 | 0.9931 | 0.9931 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 120 | 7 | 17 | 0.8924 | 0.9449 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 9 | 12 | 123 | 0.4896 | 0.4286 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | 13 | 129 | 2 | 0.0972 | 0.0915 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 92 | 51 | 1 | 0.6424 | 0.6434 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 41 | 10 | 93 | 0.6076 | 0.8039 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 144 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 143 | 0 | 1 | 0.9965 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 114 | 11 | 19 | 0.8576 | 0.9120 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | 84 | 59 | 1 | 0.5868 | 0.5874 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | quest_state_correctness | 143 | 0 | 1 | 0.9965 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | lore_consistency | 139 | 1 | 4 | 0.9792 | 0.9929 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0 | 36 | 108 | 0.3750 | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | objective_completion_support | 68 | 41 | 35 | 0.5938 | 0.6239 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 125 | 19 | 0 | 0.8681 | 0.8681 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 63 | 69 | 12 | 0.4792 | 0.4773 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 143 | 0 | 1 | 0.9965 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 141 | 2 | 1 | 0.9826 | 0.9860 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 113 | 9 | 22 | 0.8611 | 0.9262 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | 12 | 11 | 121 | 0.5035 | 0.5217 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | 26 | 116 | 2 | 0.1875 | 0.1831 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | 93 | 49 | 2 | 0.6528 | 0.6549 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | 34 | 10 | 100 | 0.5833 | 0.7727 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 142 | 2 | 0 | 0.9861 | 0.9861 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.0069 | 0.4444 | 0.1458 |
| proposed_contextual | 0.0000 | 0.0000 | 0.4375 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.4444 | 0.0000 | 0.0000 |
| baseline_no_context | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline_no_context_phi3_latest | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `8`
- Unique template signatures: `143`
- Template signature ratio: `0.9931`
- Effective sample size by source clustering: `8.00`
- Effective sample size by template-signature clustering: `142.03`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: unavailable (No BERTScore values found in merged scores.).

This report covers proposal RO5 metrics plus game-facing outcomes: quest-state correctness, lore consistency, contradiction safety, objective completion support, gameplay usefulness, and time-pressure acceptability.
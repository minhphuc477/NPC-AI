# Proposal Alignment Evaluation Report

- Run ID: `20260310T015736Z`
- Generated: `2026-03-10T01:58:22.143652+00:00`
- Scenarios: `tmp\proposal_smoke_fix\20260310T015736Z\scenarios.jsonl`
- Scenario count: `3`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off
- `baseline_no_context`: model `phi3:mini`, dynamic_context=off, response_control=off
- `baseline_no_context_phi3_latest`: model `phi3:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.3086 (0.2296, 0.4148) | 0.2667 (0.2333, 0.3333) | 0.8314 (0.7678, 0.9380) | 0.3996 (0.3443, 0.4565) | n/a |
| proposed_contextual | 0.2669 (0.2177, 0.3017) | 0.3111 (0.2333, 0.3667) | 0.9142 (0.8598, 0.9424) | 0.4132 (0.3983, 0.4426) | n/a |
| candidate_no_context | 0.0537 (0.0111, 0.0753) | 0.2667 (0.2333, 0.3333) | 0.9047 (0.9046, 0.9048) | 0.2954 (0.2949, 0.2960) | n/a |
| baseline_no_context | 0.0296 (0.0060, 0.0742) | 0.1333 (0.1000, 0.2000) | 0.9279 (0.8733, 0.9702) | 0.2387 (0.2068, 0.2572) | n/a |
| baseline_no_context_phi3_latest | 0.0079 (0.0051, 0.0094) | 0.1778 (0.1000, 0.3333) | 0.9571 (0.9262, 0.9825) | 0.2509 (0.2171, 0.3058) | n/a |

## Game-facing Outcome Metrics (mean, 95% CI)
| Arm | Quest-state Correctness | Lore Consistency | Contradiction Safety | Objective Completion Support | Gameplay Usefulness | Time-pressure Acceptability |
|---|---:|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.3584 (0.3073, 0.4573) | 0.1933 (0.0278, 0.3184) | 1.0000 (1.0000, 1.0000) | 0.1378 (0.0833, 0.2350) | 0.3489 (0.2719, 0.4063) | 0.2613 (0.2054, 0.3356) |
| proposed_contextual | 0.3112 (0.2824, 0.3406) | 0.1665 (0.1252, 0.1907) | 1.0000 (1.0000, 1.0000) | 0.1322 (0.0950, 0.1783) | 0.3898 (0.3713, 0.4047) | 0.3503 (0.3356, 0.3630) |
| candidate_no_context | 0.1382 (0.1187, 0.1491) | 0.0000 (0.0000, 0.0000) | 1.0000 (1.0000, 1.0000) | 0.1044 (0.0950, 0.1233) | 0.3232 (0.3158, 0.3269) | 0.3484 (0.3356, 0.3630) |
| baseline_no_context | 0.1173 (0.1022, 0.1467) | 0.0089 (0.0000, 0.0266) | 1.0000 (1.0000, 1.0000) | 0.0411 (0.0400, 0.0417) | 0.2886 (0.2563, 0.3131) | 0.2854 (0.2680, 0.2942) |
| baseline_no_context_phi3_latest | 0.1028 (0.1018, 0.1034) | 0.0000 (0.0000, 0.0000) | 1.0000 (1.0000, 1.0000) | 0.0272 (0.0000, 0.0817) | 0.2942 (0.2826, 0.3085) | 0.2791 (0.2500, 0.3174) |

- Multi-turn contradiction rate is reported as `1 - contradiction_safety` in row-level outputs.

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.2132 | 3.9696 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0444 | 0.1667 |
| proposed_vs_candidate_no_context | naturalness | 0.0095 | 0.0105 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.1730 | 1.2514 |
| proposed_vs_candidate_no_context | lore_consistency | 0.1665 | nan |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | 0.0278 | 0.2660 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 0.0666 | 0.2061 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | 0.0020 | 0.0057 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.2818 | 4.4286 |
| proposed_vs_candidate_no_context | context_overlap | 0.0531 | 1.7381 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0556 | 0.3333 |
| proposed_vs_candidate_no_context | persona_style | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | distinct1 | 0.0237 | 0.0253 |
| proposed_vs_candidate_no_context | length_score | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | sentence_score | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | overall_quality | 0.1179 | 0.3990 |
| proposed_vs_baseline_no_context | context_relevance | 0.2373 | 8.0111 |
| proposed_vs_baseline_no_context | persona_consistency | 0.1778 | 1.3333 |
| proposed_vs_baseline_no_context | naturalness | -0.0137 | -0.0147 |
| proposed_vs_baseline_no_context | quest_state_correctness | 0.1939 | 1.6523 |
| proposed_vs_baseline_no_context | lore_consistency | 0.1576 | 17.7908 |
| proposed_vs_baseline_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| proposed_vs_baseline_no_context | objective_completion_support | 0.0911 | 2.2162 |
| proposed_vs_baseline_no_context | gameplay_usefulness | 0.1011 | 0.3504 |
| proposed_vs_baseline_no_context | time_pressure_acceptability | 0.0649 | 0.2274 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.3152 | 10.4000 |
| proposed_vs_baseline_no_context | context_overlap | 0.0556 | 1.9838 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 0.2222 | nan |
| proposed_vs_baseline_no_context | persona_style | 0.0000 | 0.0000 |
| proposed_vs_baseline_no_context | distinct1 | -0.0286 | -0.0288 |
| proposed_vs_baseline_no_context | length_score | -0.0111 | -0.0169 |
| proposed_vs_baseline_no_context | sentence_score | 0.0000 | 0.0000 |
| proposed_vs_baseline_no_context | overall_quality | 0.1746 | 0.7315 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 0.2590 | 32.7989 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1333 | 0.7500 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | -0.0429 | -0.0448 |
| proposed_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.2084 | 2.0259 |
| proposed_vs_baseline_no_context_phi3_latest | lore_consistency | 0.1665 | nan |
| proposed_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.1050 | 3.8571 |
| proposed_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 0.0955 | 0.3247 |
| proposed_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 0.0712 | 0.2552 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.3455 | nan |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 0.0573 | 2.1767 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1667 | 3.0000 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | 0.0000 | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | 0.0094 | 0.0099 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | -0.2333 | -0.2658 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | 0.0000 | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | 0.1623 | 0.6468 |
| controlled_vs_proposed_raw | context_relevance | 0.0417 | 0.1561 |
| controlled_vs_proposed_raw | persona_consistency | -0.0444 | -0.1429 |
| controlled_vs_proposed_raw | naturalness | -0.0828 | -0.0906 |
| controlled_vs_proposed_raw | quest_state_correctness | 0.0472 | 0.1517 |
| controlled_vs_proposed_raw | lore_consistency | 0.0268 | 0.1610 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0056 | 0.0420 |
| controlled_vs_proposed_raw | gameplay_usefulness | -0.0409 | -0.1049 |
| controlled_vs_proposed_raw | time_pressure_acceptability | -0.0890 | -0.2541 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.0576 | 0.1667 |
| controlled_vs_proposed_raw | context_overlap | 0.0045 | 0.0544 |
| controlled_vs_proposed_raw | persona_keyword_coverage | -0.0556 | -0.2500 |
| controlled_vs_proposed_raw | persona_style | 0.0000 | 0.0000 |
| controlled_vs_proposed_raw | distinct1 | -0.0293 | -0.0305 |
| controlled_vs_proposed_raw | length_score | -0.3556 | -0.5517 |
| controlled_vs_proposed_raw | sentence_score | 0.0000 | 0.0000 |
| controlled_vs_proposed_raw | overall_quality | -0.0136 | -0.0330 |
| controlled_vs_candidate_no_context | context_relevance | 0.2549 | 4.7454 |
| controlled_vs_candidate_no_context | persona_consistency | 0.0000 | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | -0.0733 | -0.0811 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.2202 | 1.5930 |
| controlled_vs_candidate_no_context | lore_consistency | 0.1933 | nan |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | 0.0333 | 0.3191 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0257 | 0.0795 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | -0.0870 | -0.2498 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3394 | 5.3333 |
| controlled_vs_candidate_no_context | context_overlap | 0.0576 | 1.8870 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.0000 | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.0000 | 0.0000 |
| controlled_vs_candidate_no_context | distinct1 | -0.0056 | -0.0060 |
| controlled_vs_candidate_no_context | length_score | -0.3556 | -0.5517 |
| controlled_vs_candidate_no_context | sentence_score | 0.0000 | 0.0000 |
| controlled_vs_candidate_no_context | overall_quality | 0.1042 | 0.3529 |
| controlled_vs_baseline_no_context | context_relevance | 0.2790 | 9.4179 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1333 | 1.0000 |
| controlled_vs_baseline_no_context | naturalness | -0.0965 | -0.1040 |
| controlled_vs_baseline_no_context | quest_state_correctness | 0.2411 | 2.0548 |
| controlled_vs_baseline_no_context | lore_consistency | 0.1844 | 20.8164 |
| controlled_vs_baseline_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_baseline_no_context | objective_completion_support | 0.0967 | 2.3514 |
| controlled_vs_baseline_no_context | gameplay_usefulness | 0.0602 | 0.2087 |
| controlled_vs_baseline_no_context | time_pressure_acceptability | -0.0241 | -0.0845 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3727 | 12.3000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0601 | 2.1461 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1667 | nan |
| controlled_vs_baseline_no_context | persona_style | 0.0000 | 0.0000 |
| controlled_vs_baseline_no_context | distinct1 | -0.0579 | -0.0584 |
| controlled_vs_baseline_no_context | length_score | -0.3667 | -0.5593 |
| controlled_vs_baseline_no_context | sentence_score | 0.0000 | 0.0000 |
| controlled_vs_baseline_no_context | overall_quality | 0.1609 | 0.6744 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.3007 | 38.0753 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.0889 | 0.5000 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | -0.1257 | -0.1314 |
| controlled_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.2556 | 2.4851 |
| controlled_vs_baseline_no_context_phi3_latest | lore_consistency | 0.1933 | nan |
| controlled_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.1106 | 4.0612 |
| controlled_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 0.0546 | 0.1857 |
| controlled_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | -0.0178 | -0.0637 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.4030 | nan |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0618 | 2.3495 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1111 | 2.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | 0.0000 | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0199 | -0.0209 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | -0.5889 | -0.6709 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.0000 | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1487 | 0.5925 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2790 | 9.4179 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1333 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0965 | -0.1040 |
| proposed_contextual_controlled_vs_baseline_no_context | quest_state_correctness | 0.2411 | 2.0548 |
| proposed_contextual_controlled_vs_baseline_no_context | lore_consistency | 0.1844 | 20.8164 |
| proposed_contextual_controlled_vs_baseline_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | objective_completion_support | 0.0967 | 2.3514 |
| proposed_contextual_controlled_vs_baseline_no_context | gameplay_usefulness | 0.0602 | 0.2087 |
| proposed_contextual_controlled_vs_baseline_no_context | time_pressure_acceptability | -0.0241 | -0.0845 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3727 | 12.3000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0601 | 2.1461 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1667 | nan |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0000 | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0579 | -0.0584 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | -0.3667 | -0.5593 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0000 | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1609 | 0.6744 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.3007 | 38.0753 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.0889 | 0.5000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | -0.1257 | -0.1314 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.2556 | 2.4851 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | lore_consistency | 0.1933 | nan |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.1106 | 4.0612 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 0.0546 | 0.1857 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | -0.0178 | -0.0637 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.4030 | nan |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0618 | 2.3495 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1111 | 2.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | 0.0000 | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0199 | -0.0209 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | -0.5889 | -0.6709 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.0000 | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1487 | 0.5925 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.2132 | (0.2066, 0.2264) | 0.0000 | 0.2132 | (0.2132, 0.2132) | 0.0000 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0444 | (0.0000, 0.1333) | 0.3000 | 0.0444 | (0.0444, 0.0444) | 0.0000 |
| proposed_vs_candidate_no_context | naturalness | 0.0095 | (-0.0450, 0.0377) | 0.2433 | 0.0095 | (0.0095, 0.0095) | 0.0000 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.1730 | (0.1637, 0.1915) | 0.0000 | 0.1730 | (0.1730, 0.1730) | 0.0000 |
| proposed_vs_candidate_no_context | lore_consistency | 0.1665 | (0.1252, 0.1907) | 0.0000 | 0.1665 | (0.1665, 0.1665) | 0.0000 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | 0.0278 | (0.0000, 0.0833) | 0.3033 | 0.0278 | (0.0278, 0.0278) | 0.0000 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 0.0666 | (0.0444, 0.0779) | 0.0000 | 0.0666 | (0.0666, 0.0666) | 0.0000 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | 0.0020 | (0.0000, 0.0059) | 0.2937 | 0.0020 | (0.0020, 0.0020) | 0.0000 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.2818 | (0.2727, 0.3000) | 0.0000 | 0.2818 | (0.2818, 0.2818) | 0.0000 |
| proposed_vs_candidate_no_context | context_overlap | 0.0531 | (0.0522, 0.0547) | 0.0000 | 0.0531 | (0.0531, 0.0531) | 0.0000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0556 | (0.0000, 0.1667) | 0.2957 | 0.0556 | (0.0556, 0.0556) | 0.0000 |
| proposed_vs_candidate_no_context | persona_style | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | distinct1 | 0.0237 | (0.0062, 0.0542) | 0.0000 | 0.0237 | (0.0237, 0.0237) | 0.0000 |
| proposed_vs_candidate_no_context | length_score | -0.0000 | (-0.3333, 0.1667) | 0.6977 | -0.0000 | (-0.0000, -0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | sentence_score | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | overall_quality | 0.1179 | (0.1034, 0.1465) | 0.0000 | 0.1179 | (0.1179, 0.1179) | 0.0000 |
| proposed_vs_baseline_no_context | context_relevance | 0.2373 | (0.2072, 0.2957) | 0.0000 | 0.2373 | (0.2373, 0.2373) | 0.0000 |
| proposed_vs_baseline_no_context | persona_consistency | 0.1778 | (0.1333, 0.2667) | 0.0000 | 0.1778 | (0.1778, 0.1778) | 0.0000 |
| proposed_vs_baseline_no_context | naturalness | -0.0137 | (-0.0298, 0.0024) | 0.9663 | -0.0137 | (-0.0137, -0.0137) | 1.0000 |
| proposed_vs_baseline_no_context | quest_state_correctness | 0.1939 | (0.1639, 0.2384) | 0.0000 | 0.1939 | (0.1939, 0.1939) | 0.0000 |
| proposed_vs_baseline_no_context | lore_consistency | 0.1576 | (0.1252, 0.1907) | 0.0000 | 0.1576 | (0.1576, 0.1576) | 0.0000 |
| proposed_vs_baseline_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_baseline_no_context | objective_completion_support | 0.0911 | (0.0550, 0.1367) | 0.0000 | 0.0911 | (0.0911, 0.0911) | 0.0000 |
| proposed_vs_baseline_no_context | gameplay_usefulness | 0.1011 | (0.0916, 0.1150) | 0.0000 | 0.1011 | (0.1011, 0.1011) | 0.0000 |
| proposed_vs_baseline_no_context | time_pressure_acceptability | 0.0649 | (0.0583, 0.0688) | 0.0000 | 0.0649 | (0.0649, 0.0649) | 0.0000 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.3152 | (0.2727, 0.4000) | 0.0000 | 0.3152 | (0.3152, 0.3152) | 0.0000 |
| proposed_vs_baseline_no_context | context_overlap | 0.0556 | (0.0523, 0.0603) | 0.0000 | 0.0556 | (0.0556, 0.0556) | 0.0000 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 0.2222 | (0.1667, 0.3333) | 0.0000 | 0.2222 | (0.2222, 0.2222) | 0.0000 |
| proposed_vs_baseline_no_context | persona_style | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_baseline_no_context | distinct1 | -0.0286 | (-0.0606, -0.0079) | 1.0000 | -0.0286 | (-0.0286, -0.0286) | 1.0000 |
| proposed_vs_baseline_no_context | length_score | -0.0111 | (-0.1333, 0.1333) | 0.6310 | -0.0111 | (-0.0111, -0.0111) | 1.0000 |
| proposed_vs_baseline_no_context | sentence_score | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_baseline_no_context | overall_quality | 0.1746 | (0.1411, 0.2357) | 0.0000 | 0.1746 | (0.1746, 0.1746) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 0.2590 | (0.2085, 0.2966) | 0.0000 | 0.2590 | (0.2590, 0.2590) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1333 | (0.0000, 0.2667) | 0.0347 | 0.1333 | (0.1333, 0.1333) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | -0.0429 | (-0.0664, -0.0221) | 1.0000 | -0.0429 | (-0.0429, -0.0429) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.2084 | (0.1790, 0.2388) | 0.0000 | 0.2084 | (0.2084, 0.2084) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | lore_consistency | 0.1665 | (0.1252, 0.1907) | 0.0000 | 0.1665 | (0.1665, 0.1665) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.1050 | (0.0950, 0.1233) | 0.0000 | 0.1050 | (0.1050, 0.1050) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 0.0955 | (0.0629, 0.1130) | 0.0000 | 0.0955 | (0.0955, 0.0955) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 0.0712 | (0.0351, 0.0930) | 0.0000 | 0.0712 | (0.0712, 0.0712) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.3455 | (0.2727, 0.4000) | 0.0000 | 0.3455 | (0.3455, 0.3455) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 0.0573 | (0.0553, 0.0585) | 0.0000 | 0.0573 | (0.0573, 0.0573) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1667 | (0.0000, 0.3333) | 0.0373 | 0.1667 | (0.1667, 0.1667) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | 0.0094 | (-0.0336, 0.0447) | 0.2633 | 0.0094 | (0.0094, 0.0094) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | -0.2333 | (-0.3667, -0.1333) | 1.0000 | -0.2333 | (-0.2333, -0.2333) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | 0.1623 | (0.0924, 0.2255) | 0.0000 | 0.1623 | (0.1623, 0.1623) | 0.0000 |
| controlled_vs_proposed_raw | context_relevance | 0.0417 | (-0.0721, 0.1971) | 0.2837 | 0.0417 | (0.0417, 0.0417) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | -0.0444 | (-0.1333, 0.0000) | 1.0000 | -0.0444 | (-0.0444, -0.0444) | 1.0000 |
| controlled_vs_proposed_raw | naturalness | -0.0828 | (-0.1726, -0.0044) | 1.0000 | -0.0828 | (-0.0828, -0.0828) | 1.0000 |
| controlled_vs_proposed_raw | quest_state_correctness | 0.0472 | (-0.0333, 0.1750) | 0.2890 | 0.0472 | (0.0472, 0.0472) | 0.0000 |
| controlled_vs_proposed_raw | lore_consistency | 0.0268 | (-0.1628, 0.1933) | 0.3627 | 0.0268 | (0.0268, 0.0268) | 0.0000 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0056 | (-0.0950, 0.1117) | 0.4140 | 0.0056 | (0.0056, 0.0056) | 0.0000 |
| controlled_vs_proposed_raw | gameplay_usefulness | -0.0409 | (-0.0994, 0.0016) | 0.9583 | -0.0409 | (-0.0409, -0.0409) | 1.0000 |
| controlled_vs_proposed_raw | time_pressure_acceptability | -0.0890 | (-0.1470, 0.0000) | 1.0000 | -0.0890 | (-0.0890, -0.0890) | 1.0000 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.0576 | (-0.1000, 0.2727) | 0.2843 | 0.0576 | (0.0576, 0.0576) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0045 | (-0.0071, 0.0207) | 0.2953 | 0.0045 | (0.0045, 0.0045) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | -0.0556 | (-0.1667, 0.0000) | 1.0000 | -0.0556 | (-0.0556, -0.0556) | 1.0000 |
| controlled_vs_proposed_raw | persona_style | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_proposed_raw | distinct1 | -0.0293 | (-0.0482, -0.0122) | 1.0000 | -0.0293 | (-0.0293, -0.0293) | 1.0000 |
| controlled_vs_proposed_raw | length_score | -0.3556 | (-0.7667, 0.0333) | 0.9597 | -0.3556 | (-0.3556, -0.3556) | 1.0000 |
| controlled_vs_proposed_raw | sentence_score | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_proposed_raw | overall_quality | -0.0136 | (-0.0983, 0.0583) | 0.6230 | -0.0136 | (-0.0136, -0.0136) | 1.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2549 | (0.1543, 0.4037) | 0.0000 | 0.2549 | (0.2549, 0.2549) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | naturalness | -0.0733 | (-0.1368, 0.0333) | 0.9610 | -0.0733 | (-0.0733, -0.0733) | 1.0000 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.2202 | (0.1582, 0.3387) | 0.0000 | 0.2202 | (0.2202, 0.2202) | 0.0000 |
| controlled_vs_candidate_no_context | lore_consistency | 0.1933 | (0.0278, 0.3184) | 0.0000 | 0.1933 | (0.1933, 0.1933) | 0.0000 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | 0.0333 | (-0.0117, 0.1117) | 0.2910 | 0.0333 | (0.0333, 0.0333) | 0.0000 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0257 | (-0.0550, 0.0795) | 0.2600 | 0.0257 | (0.0257, 0.0257) | 0.0000 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | -0.0870 | (-0.1411, 0.0000) | 1.0000 | -0.0870 | (-0.0870, -0.0870) | 1.0000 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3394 | (0.2000, 0.5455) | 0.0000 | 0.3394 | (0.3394, 0.3394) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0576 | (0.0477, 0.0730) | 0.0000 | 0.0576 | (0.0576, 0.0576) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | distinct1 | -0.0056 | (-0.0420, 0.0420) | 0.6370 | -0.0056 | (-0.0056, -0.0056) | 1.0000 |
| controlled_vs_candidate_no_context | length_score | -0.3556 | (-0.6667, 0.2000) | 0.9610 | -0.3556 | (-0.3556, -0.3556) | 1.0000 |
| controlled_vs_candidate_no_context | sentence_score | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | overall_quality | 0.1042 | (0.0482, 0.1616) | 0.0000 | 0.1042 | (0.1042, 0.1042) | 0.0000 |
| controlled_vs_baseline_no_context | context_relevance | 0.2790 | (0.2072, 0.4061) | 0.0000 | 0.2790 | (0.2790, 0.2790) | 0.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1333 | (0.1333, 0.1333) | 0.0000 | 0.1333 | (0.1333, 0.1333) | 0.0000 |
| controlled_vs_baseline_no_context | naturalness | -0.0965 | (-0.2024, -0.0020) | 1.0000 | -0.0965 | (-0.0965, -0.0965) | 1.0000 |
| controlled_vs_baseline_no_context | quest_state_correctness | 0.2411 | (0.1639, 0.3542) | 0.0000 | 0.2411 | (0.2411, 0.2411) | 0.0000 |
| controlled_vs_baseline_no_context | lore_consistency | 0.1844 | (0.0278, 0.3184) | 0.0000 | 0.1844 | (0.1844, 0.1844) | 0.0000 |
| controlled_vs_baseline_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_baseline_no_context | objective_completion_support | 0.0967 | (0.0417, 0.1933) | 0.0000 | 0.0967 | (0.0967, 0.0967) | 0.0000 |
| controlled_vs_baseline_no_context | gameplay_usefulness | 0.0602 | (0.0156, 0.0932) | 0.0000 | 0.0602 | (0.0602, 0.0602) | 0.0000 |
| controlled_vs_baseline_no_context | time_pressure_acceptability | -0.0241 | (-0.0887, 0.0676) | 0.7383 | -0.0241 | (-0.0241, -0.0241) | 1.0000 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3727 | (0.2727, 0.5455) | 0.0000 | 0.3727 | (0.3727, 0.3727) | 0.0000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0601 | (0.0452, 0.0810) | 0.0000 | 0.0601 | (0.0601, 0.0601) | 0.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1667 | (0.1667, 0.1667) | 0.0000 | 0.1667 | (0.1667, 0.1667) | 0.0000 |
| controlled_vs_baseline_no_context | persona_style | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_baseline_no_context | distinct1 | -0.0579 | (-0.0882, -0.0294) | 1.0000 | -0.0579 | (-0.0579, -0.0579) | 1.0000 |
| controlled_vs_baseline_no_context | length_score | -0.3667 | (-0.9000, 0.1667) | 0.9607 | -0.3667 | (-0.3667, -0.3667) | 1.0000 |
| controlled_vs_baseline_no_context | sentence_score | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_baseline_no_context | overall_quality | 0.1609 | (0.1375, 0.1994) | 0.0000 | 0.1609 | (0.1609, 0.1609) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.3007 | (0.2245, 0.4056) | 0.0000 | 0.3007 | (0.3007, 0.3007) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.0889 | (0.0000, 0.1333) | 0.0367 | 0.0889 | (0.0889, 0.0889) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | -0.1257 | (-0.1947, -0.0445) | 1.0000 | -0.1257 | (-0.1257, -0.1257) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.2556 | (0.2055, 0.3540) | 0.0000 | 0.2556 | (0.2556, 0.2556) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | lore_consistency | 0.1933 | (0.0278, 0.3184) | 0.0000 | 0.1933 | (0.1933, 0.1933) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.1106 | (0.0017, 0.2350) | 0.0000 | 0.1106 | (0.1106, 0.1106) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 0.0546 | (-0.0365, 0.1147) | 0.0377 | 0.0546 | (0.0546, 0.0546) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | -0.0178 | (-0.1119, 0.0856) | 0.6427 | -0.0178 | (-0.0178, -0.0178) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.4030 | (0.3000, 0.5455) | 0.0000 | 0.4030 | (0.4030, 0.4030) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0618 | (0.0483, 0.0792) | 0.0000 | 0.0618 | (0.0618, 0.0618) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1111 | (0.0000, 0.1667) | 0.0450 | 0.1111 | (0.1111, 0.1111) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0199 | (-0.0612, 0.0051) | 0.8480 | -0.0199 | (-0.0199, -0.0199) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | -0.5889 | (-0.9667, -0.1000) | 1.0000 | -0.5889 | (-0.5889, -0.5889) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1487 | (0.1272, 0.1681) | 0.0000 | 0.1487 | (0.1487, 0.1487) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2790 | (0.2072, 0.4061) | 0.0000 | 0.2790 | (0.2790, 0.2790) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1333 | (0.1333, 0.1333) | 0.0000 | 0.1333 | (0.1333, 0.1333) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0965 | (-0.2024, -0.0020) | 1.0000 | -0.0965 | (-0.0965, -0.0965) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | quest_state_correctness | 0.2411 | (0.1639, 0.3542) | 0.0000 | 0.2411 | (0.2411, 0.2411) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | lore_consistency | 0.1844 | (0.0278, 0.3184) | 0.0000 | 0.1844 | (0.1844, 0.1844) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | objective_completion_support | 0.0967 | (0.0417, 0.1933) | 0.0000 | 0.0967 | (0.0967, 0.0967) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | gameplay_usefulness | 0.0602 | (0.0156, 0.0932) | 0.0000 | 0.0602 | (0.0602, 0.0602) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | time_pressure_acceptability | -0.0241 | (-0.0887, 0.0676) | 0.7370 | -0.0241 | (-0.0241, -0.0241) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3727 | (0.2727, 0.5455) | 0.0000 | 0.3727 | (0.3727, 0.3727) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0601 | (0.0452, 0.0810) | 0.0000 | 0.0601 | (0.0601, 0.0601) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1667 | (0.1667, 0.1667) | 0.0000 | 0.1667 | (0.1667, 0.1667) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0579 | (-0.0882, -0.0294) | 1.0000 | -0.0579 | (-0.0579, -0.0579) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | -0.3667 | (-0.9000, 0.1667) | 0.9610 | -0.3667 | (-0.3667, -0.3667) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1609 | (0.1375, 0.1994) | 0.0000 | 0.1609 | (0.1609, 0.1609) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.3007 | (0.2245, 0.4056) | 0.0000 | 0.3007 | (0.3007, 0.3007) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.0889 | (0.0000, 0.1333) | 0.0390 | 0.0889 | (0.0889, 0.0889) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | -0.1257 | (-0.1947, -0.0445) | 1.0000 | -0.1257 | (-0.1257, -0.1257) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.2556 | (0.2055, 0.3540) | 0.0000 | 0.2556 | (0.2556, 0.2556) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | lore_consistency | 0.1933 | (0.0278, 0.3184) | 0.0000 | 0.1933 | (0.1933, 0.1933) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.1106 | (0.0017, 0.2350) | 0.0000 | 0.1106 | (0.1106, 0.1106) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 0.0546 | (-0.0365, 0.1147) | 0.0473 | 0.0546 | (0.0546, 0.0546) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | -0.0178 | (-0.1119, 0.0856) | 0.6347 | -0.0178 | (-0.0178, -0.0178) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.4030 | (0.3000, 0.5455) | 0.0000 | 0.4030 | (0.4030, 0.4030) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0618 | (0.0483, 0.0792) | 0.0000 | 0.0618 | (0.0618, 0.0618) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1111 | (0.0000, 0.1667) | 0.0363 | 0.1111 | (0.1111, 0.1111) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0199 | (-0.0612, 0.0051) | 0.8550 | -0.0199 | (-0.0199, -0.0199) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | -0.5889 | (-0.9667, -0.1000) | 1.0000 | -0.5889 | (-0.5889, -0.5889) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1487 | (0.1272, 0.1681) | 0.0000 | 0.1487 | (0.1487, 0.1487) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 3 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_vs_candidate_no_context | persona_consistency | 1 | 0 | 2 | 0.6667 | 1.0000 |
| proposed_vs_candidate_no_context | naturalness | 2 | 1 | 0 | 0.6667 | 0.6667 |
| proposed_vs_candidate_no_context | quest_state_correctness | 3 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_vs_candidate_no_context | lore_consistency | 3 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 3 | 0.5000 | nan |
| proposed_vs_candidate_no_context | objective_completion_support | 1 | 0 | 2 | 0.6667 | 1.0000 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 3 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | 1 | 0 | 2 | 0.6667 | 1.0000 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 3 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_vs_candidate_no_context | context_overlap | 3 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 1 | 0 | 2 | 0.6667 | 1.0000 |
| proposed_vs_candidate_no_context | persona_style | 0 | 0 | 3 | 0.5000 | nan |
| proposed_vs_candidate_no_context | distinct1 | 3 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_vs_candidate_no_context | length_score | 2 | 1 | 0 | 0.6667 | 0.6667 |
| proposed_vs_candidate_no_context | sentence_score | 0 | 0 | 3 | 0.5000 | nan |
| proposed_vs_candidate_no_context | overall_quality | 3 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_vs_baseline_no_context | context_relevance | 3 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_vs_baseline_no_context | persona_consistency | 3 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_vs_baseline_no_context | naturalness | 1 | 2 | 0 | 0.3333 | 0.3333 |
| proposed_vs_baseline_no_context | quest_state_correctness | 3 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_vs_baseline_no_context | lore_consistency | 3 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_vs_baseline_no_context | multi_turn_contradiction_safety | 0 | 0 | 3 | 0.5000 | nan |
| proposed_vs_baseline_no_context | objective_completion_support | 3 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_vs_baseline_no_context | gameplay_usefulness | 3 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_vs_baseline_no_context | time_pressure_acceptability | 3 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 3 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_vs_baseline_no_context | context_overlap | 3 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 3 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_vs_baseline_no_context | persona_style | 0 | 0 | 3 | 0.5000 | nan |
| proposed_vs_baseline_no_context | distinct1 | 0 | 3 | 0 | 0.0000 | 0.0000 |
| proposed_vs_baseline_no_context | length_score | 1 | 2 | 0 | 0.3333 | 0.3333 |
| proposed_vs_baseline_no_context | sentence_score | 0 | 0 | 3 | 0.5000 | nan |
| proposed_vs_baseline_no_context | overall_quality | 3 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 3 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | 2 | 0 | 1 | 0.8333 | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | 0 | 3 | 0 | 0.0000 | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | quest_state_correctness | 3 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | lore_consistency | 3 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0 | 0 | 3 | 0.5000 | nan |
| proposed_vs_baseline_no_context_phi3_latest | objective_completion_support | 3 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 3 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 3 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 3 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 3 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 2 | 0 | 1 | 0.8333 | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | 0 | 0 | 3 | 0.5000 | nan |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | 2 | 1 | 0 | 0.6667 | 0.6667 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | 0 | 3 | 0 | 0.0000 | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | 0 | 0 | 3 | 0.5000 | nan |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | 3 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_proposed_raw | context_relevance | 1 | 1 | 1 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | persona_consistency | 0 | 1 | 2 | 0.3333 | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0 | 3 | 0 | 0.0000 | 0.0000 |
| controlled_vs_proposed_raw | quest_state_correctness | 1 | 1 | 1 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | lore_consistency | 2 | 1 | 0 | 0.6667 | 0.6667 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0 | 0 | 3 | 0.5000 | nan |
| controlled_vs_proposed_raw | objective_completion_support | 1 | 1 | 1 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | gameplay_usefulness | 1 | 2 | 0 | 0.3333 | 0.3333 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 0 | 2 | 1 | 0.1667 | 0.0000 |
| controlled_vs_proposed_raw | context_keyword_coverage | 1 | 1 | 1 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | context_overlap | 1 | 1 | 1 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0 | 1 | 2 | 0.3333 | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0 | 0 | 3 | 0.5000 | nan |
| controlled_vs_proposed_raw | distinct1 | 0 | 3 | 0 | 0.0000 | 0.0000 |
| controlled_vs_proposed_raw | length_score | 1 | 2 | 0 | 0.3333 | 0.3333 |
| controlled_vs_proposed_raw | sentence_score | 0 | 0 | 3 | 0.5000 | nan |
| controlled_vs_proposed_raw | overall_quality | 1 | 2 | 0 | 0.3333 | 0.3333 |
| controlled_vs_candidate_no_context | context_relevance | 3 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0 | 0 | 3 | 0.5000 | nan |
| controlled_vs_candidate_no_context | naturalness | 1 | 2 | 0 | 0.3333 | 0.3333 |
| controlled_vs_candidate_no_context | quest_state_correctness | 3 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | lore_consistency | 3 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 3 | 0.5000 | nan |
| controlled_vs_candidate_no_context | objective_completion_support | 1 | 1 | 1 | 0.5000 | 0.5000 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 2 | 1 | 0 | 0.6667 | 0.6667 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 0 | 2 | 1 | 0.1667 | 0.0000 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 3 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 3 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0 | 0 | 3 | 0.5000 | nan |
| controlled_vs_candidate_no_context | persona_style | 0 | 0 | 3 | 0.5000 | nan |
| controlled_vs_candidate_no_context | distinct1 | 1 | 2 | 0 | 0.3333 | 0.3333 |
| controlled_vs_candidate_no_context | length_score | 1 | 2 | 0 | 0.3333 | 0.3333 |
| controlled_vs_candidate_no_context | sentence_score | 0 | 0 | 3 | 0.5000 | nan |
| controlled_vs_candidate_no_context | overall_quality | 3 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | context_relevance | 3 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 3 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | naturalness | 0 | 3 | 0 | 0.0000 | 0.0000 |
| controlled_vs_baseline_no_context | quest_state_correctness | 3 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | lore_consistency | 3 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | multi_turn_contradiction_safety | 0 | 0 | 3 | 0.5000 | nan |
| controlled_vs_baseline_no_context | objective_completion_support | 3 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | gameplay_usefulness | 3 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | time_pressure_acceptability | 1 | 2 | 0 | 0.3333 | 0.3333 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 3 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | context_overlap | 3 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 3 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_style | 0 | 0 | 3 | 0.5000 | nan |
| controlled_vs_baseline_no_context | distinct1 | 0 | 3 | 0 | 0.0000 | 0.0000 |
| controlled_vs_baseline_no_context | length_score | 1 | 2 | 0 | 0.3333 | 0.3333 |
| controlled_vs_baseline_no_context | sentence_score | 0 | 0 | 3 | 0.5000 | nan |
| controlled_vs_baseline_no_context | overall_quality | 3 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 3 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 2 | 0 | 1 | 0.8333 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | 0 | 3 | 0 | 0.0000 | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | quest_state_correctness | 3 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | lore_consistency | 3 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0 | 0 | 3 | 0.5000 | nan |
| controlled_vs_baseline_no_context_phi3_latest | objective_completion_support | 3 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 2 | 1 | 0 | 0.6667 | 0.6667 |
| controlled_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 1 | 2 | 0 | 0.3333 | 0.3333 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 3 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 3 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 2 | 0 | 1 | 0.8333 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | 0 | 0 | 3 | 0.5000 | nan |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | 1 | 2 | 0 | 0.3333 | 0.3333 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | 0 | 3 | 0 | 0.0000 | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0 | 0 | 3 | 0.5000 | nan |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 3 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 3 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 3 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 0 | 3 | 0 | 0.0000 | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | quest_state_correctness | 3 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | lore_consistency | 3 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | multi_turn_contradiction_safety | 0 | 0 | 3 | 0.5000 | nan |
| proposed_contextual_controlled_vs_baseline_no_context | objective_completion_support | 3 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | gameplay_usefulness | 3 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | time_pressure_acceptability | 1 | 2 | 0 | 0.3333 | 0.3333 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 3 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 3 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 3 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0 | 0 | 3 | 0.5000 | nan |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | 0 | 3 | 0 | 0.0000 | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 1 | 2 | 0 | 0.3333 | 0.3333 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0 | 0 | 3 | 0.5000 | nan |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 3 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 3 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 2 | 0 | 1 | 0.8333 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | 0 | 3 | 0 | 0.0000 | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | quest_state_correctness | 3 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | lore_consistency | 3 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0 | 0 | 3 | 0.5000 | nan |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | objective_completion_support | 3 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 2 | 1 | 0 | 0.6667 | 0.6667 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 1 | 2 | 0 | 0.3333 | 0.3333 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 3 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 3 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 2 | 0 | 1 | 0.8333 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | 0 | 0 | 3 | 0.5000 | nan |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | 1 | 2 | 0 | 0.3333 | 0.3333 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | 0 | 3 | 0 | 0.0000 | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0 | 0 | 3 | 0.5000 | nan |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 3 | 0 | 0 | 1.0000 | 1.0000 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.0000 | 0.6667 | 0.0000 |
| proposed_contextual | 0.0000 | 0.0000 | 0.6667 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| baseline_no_context | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline_no_context_phi3_latest | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `1`
- Unique template signatures: `3`
- Template signature ratio: `1.0000`
- Effective sample size by source clustering: `1.00`
- Effective sample size by template-signature clustering: `3.00`
- Detailed diagnostics are published in `scenario_dependence.json`.

## Multi-turn Contradiction
| Arm | Contradiction Rate | Contradiction Safety | Contradicted Sources | Source Count |
|---|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 1.0000 | 0 | 1 |
| proposed_contextual | 0.0000 | 1.0000 | 0 | 1 |
| candidate_no_context | 0.0000 | 1.0000 | 0 | 1 |
| baseline_no_context | 0.0000 | 1.0000 | 0 | 1 |
| baseline_no_context_phi3_latest | 0.0000 | 1.0000 | 0 | 1 |
- Detailed source-level values are published in `multi_turn_contradictions.json`.

- BERTScore status: unavailable (No module named 'bert_score').

This report covers proposal RO5 metrics plus game-facing outcomes: quest-state correctness, lore consistency, contradiction safety, objective completion support, gameplay usefulness, and time-pressure acceptability.
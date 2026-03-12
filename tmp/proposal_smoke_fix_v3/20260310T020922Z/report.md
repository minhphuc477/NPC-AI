# Proposal Alignment Evaluation Report

- Run ID: `20260310T020922Z`
- Generated: `2026-03-10T02:12:08.745317+00:00`
- Scenarios: `tmp\proposal_smoke_fix_v3\20260310T020922Z\scenarios.jsonl`
- Scenario count: `5`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off
- `baseline_no_context`: model `phi3:mini`, dynamic_context=off, response_control=off
- `baseline_no_context_phi3_latest`: model `phi3:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0830 (0.0125, 0.1978) | 0.3496 (0.2533, 0.4458) | 0.8769 (0.8627, 0.8911) | 0.3340 (0.2752, 0.4230) | n/a |
| proposed_contextual | 0.1984 (0.0762, 0.3089) | 0.2138 (0.1467, 0.2938) | 0.8630 (0.7964, 0.9295) | 0.3347 (0.2655, 0.3934) | n/a |
| candidate_no_context | 0.0210 (0.0021, 0.0517) | 0.1871 (0.1138, 0.2667) | 0.8686 (0.8294, 0.9001) | 0.2429 (0.2093, 0.2779) | n/a |
| baseline_no_context | 0.0075 (0.0034, 0.0108) | 0.1667 (0.1000, 0.2600) | 0.8920 (0.8337, 0.9293) | 0.2331 (0.2032, 0.2718) | n/a |
| baseline_no_context_phi3_latest | 0.0072 (0.0034, 0.0105) | 0.1758 (0.1000, 0.2692) | 0.8959 (0.8563, 0.9352) | 0.2385 (0.2080, 0.2724) | n/a |

## Game-facing Outcome Metrics (mean, 95% CI)
| Arm | Quest-state Correctness | Lore Consistency | Contradiction Safety | Objective Completion Support | Gameplay Usefulness | Time-pressure Acceptability |
|---|---:|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.1691 (0.1133, 0.2632) | 0.0127 (0.0000, 0.0265) | 1.0000 (1.0000, 1.0000) | 0.1277 (0.0853, 0.1680) | 0.3257 (0.3026, 0.3489) | 0.3341 (0.2995, 0.3607) |
| proposed_contextual | 0.2542 (0.1566, 0.3432) | 0.0935 (0.0291, 0.1578) | 1.0000 (1.0000, 1.0000) | 0.0873 (0.0277, 0.1463) | 0.3183 (0.2546, 0.3686) | 0.2770 (0.2260, 0.3279) |
| candidate_no_context | 0.1145 (0.1008, 0.1335) | 0.0093 (0.0000, 0.0192) | 1.0000 (1.0000, 1.0000) | 0.0327 (0.0000, 0.0980) | 0.2628 (0.2291, 0.2914) | 0.2852 (0.2540, 0.3271) |
| baseline_no_context | 0.1027 (0.1012, 0.1039) | 0.0065 (0.0000, 0.0150) | 1.0000 (1.0000, 1.0000) | 0.0193 (0.0000, 0.0413) | 0.2606 (0.2271, 0.2882) | 0.2776 (0.2620, 0.2942) |
| baseline_no_context_phi3_latest | 0.1026 (0.1012, 0.1037) | 0.0041 (0.0000, 0.0122) | 1.0000 (1.0000, 1.0000) | 0.0000 (0.0000, 0.0000) | 0.2522 (0.2341, 0.2703) | 0.2706 (0.2540, 0.2871) |

- Multi-turn contradiction rate is reported as `1 - contradiction_safety` in row-level outputs.

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.1774 | 8.4284 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0267 | 0.1425 |
| proposed_vs_candidate_no_context | naturalness | -0.0056 | -0.0065 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.1397 | 1.2201 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0841 | 9.0126 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | 0.0547 | 1.6735 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 0.0555 | 0.2111 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | -0.0083 | -0.0290 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.2382 | 13.1000 |
| proposed_vs_candidate_no_context | context_overlap | 0.0355 | 1.2790 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0333 | 0.5000 |
| proposed_vs_candidate_no_context | persona_style | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | distinct1 | -0.0150 | -0.0156 |
| proposed_vs_candidate_no_context | length_score | -0.0333 | -0.0725 |
| proposed_vs_candidate_no_context | sentence_score | 0.0700 | 0.0753 |
| proposed_vs_candidate_no_context | overall_quality | 0.0918 | 0.3779 |
| proposed_vs_baseline_no_context | context_relevance | 0.1909 | 25.5680 |
| proposed_vs_baseline_no_context | persona_consistency | 0.0471 | 0.2825 |
| proposed_vs_baseline_no_context | naturalness | -0.0290 | -0.0325 |
| proposed_vs_baseline_no_context | quest_state_correctness | 0.1515 | 1.4757 |
| proposed_vs_baseline_no_context | lore_consistency | 0.0870 | 13.4378 |
| proposed_vs_baseline_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| proposed_vs_baseline_no_context | objective_completion_support | 0.0680 | 3.5172 |
| proposed_vs_baseline_no_context | gameplay_usefulness | 0.0577 | 0.2215 |
| proposed_vs_baseline_no_context | time_pressure_acceptability | -0.0007 | -0.0024 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.2564 | nan |
| proposed_vs_baseline_no_context | context_overlap | 0.0383 | 1.5379 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 0.0667 | 2.0000 |
| proposed_vs_baseline_no_context | persona_style | -0.0312 | -0.0446 |
| proposed_vs_baseline_no_context | distinct1 | -0.0475 | -0.0479 |
| proposed_vs_baseline_no_context | length_score | -0.1200 | -0.2195 |
| proposed_vs_baseline_no_context | sentence_score | 0.1400 | 0.1628 |
| proposed_vs_baseline_no_context | overall_quality | 0.1016 | 0.4358 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 0.1912 | 26.4301 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | 0.0379 | 0.2156 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | -0.0330 | -0.0368 |
| proposed_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.1516 | 1.4777 |
| proposed_vs_baseline_no_context_phi3_latest | lore_consistency | 0.0894 | 22.0009 |
| proposed_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.0873 | nan |
| proposed_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 0.0661 | 0.2621 |
| proposed_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 0.0064 | 0.0236 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.2564 | nan |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 0.0391 | 1.6203 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.0333 | 0.5000 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | 0.0562 | 0.0918 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | -0.0466 | -0.0470 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | -0.1067 | -0.2000 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | 0.0700 | 0.0753 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | 0.0961 | 0.4029 |
| controlled_vs_proposed_raw | context_relevance | -0.1154 | -0.5818 |
| controlled_vs_proposed_raw | persona_consistency | 0.1358 | 0.6355 |
| controlled_vs_proposed_raw | naturalness | 0.0139 | 0.0161 |
| controlled_vs_proposed_raw | quest_state_correctness | -0.0852 | -0.3350 |
| controlled_vs_proposed_raw | lore_consistency | -0.0807 | -0.8639 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0403 | 0.4618 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0074 | 0.0233 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 0.0572 | 0.2064 |
| controlled_vs_proposed_raw | context_keyword_coverage | -0.1582 | -0.6170 |
| controlled_vs_proposed_raw | context_overlap | -0.0157 | -0.2483 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1667 | 1.6667 |
| controlled_vs_proposed_raw | persona_style | 0.0125 | 0.0187 |
| controlled_vs_proposed_raw | distinct1 | -0.0352 | -0.0373 |
| controlled_vs_proposed_raw | length_score | 0.1400 | 0.3281 |
| controlled_vs_proposed_raw | sentence_score | 0.0000 | 0.0000 |
| controlled_vs_proposed_raw | overall_quality | -0.0007 | -0.0020 |
| controlled_vs_candidate_no_context | context_relevance | 0.0619 | 2.9430 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1625 | 0.8686 |
| controlled_vs_candidate_no_context | naturalness | 0.0083 | 0.0095 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.0546 | 0.4765 |
| controlled_vs_candidate_no_context | lore_consistency | 0.0034 | 0.3627 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | 0.0950 | 2.9082 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0629 | 0.2393 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 0.0489 | 0.1714 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.0800 | 4.4000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0198 | 0.7132 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2000 | 3.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.0125 | 0.0187 |
| controlled_vs_candidate_no_context | distinct1 | -0.0502 | -0.0523 |
| controlled_vs_candidate_no_context | length_score | 0.1067 | 0.2319 |
| controlled_vs_candidate_no_context | sentence_score | 0.0700 | 0.0753 |
| controlled_vs_candidate_no_context | overall_quality | 0.0911 | 0.3751 |
| controlled_vs_baseline_no_context | context_relevance | 0.0755 | 10.1108 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1829 | 1.0975 |
| controlled_vs_baseline_no_context | naturalness | -0.0151 | -0.0169 |
| controlled_vs_baseline_no_context | quest_state_correctness | 0.0664 | 0.6464 |
| controlled_vs_baseline_no_context | lore_consistency | 0.0062 | 0.9650 |
| controlled_vs_baseline_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_baseline_no_context | objective_completion_support | 0.1083 | 5.6034 |
| controlled_vs_baseline_no_context | gameplay_usefulness | 0.0651 | 0.2499 |
| controlled_vs_baseline_no_context | time_pressure_acceptability | 0.0565 | 0.2035 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.0982 | nan |
| controlled_vs_baseline_no_context | context_overlap | 0.0226 | 0.9078 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2333 | 7.0000 |
| controlled_vs_baseline_no_context | persona_style | -0.0187 | -0.0268 |
| controlled_vs_baseline_no_context | distinct1 | -0.0827 | -0.0834 |
| controlled_vs_baseline_no_context | length_score | 0.0200 | 0.0366 |
| controlled_vs_baseline_no_context | sentence_score | 0.1400 | 0.1628 |
| controlled_vs_baseline_no_context | overall_quality | 0.1009 | 0.4329 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.0757 | 10.4714 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1738 | 0.9882 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | -0.0190 | -0.0213 |
| controlled_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.0665 | 0.6478 |
| controlled_vs_baseline_no_context_phi3_latest | lore_consistency | 0.0087 | 2.1304 |
| controlled_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.1277 | nan |
| controlled_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 0.0735 | 0.2915 |
| controlled_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 0.0635 | 0.2349 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.0982 | nan |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0234 | 0.9697 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.2000 | 3.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | 0.0687 | 0.1122 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0818 | -0.0825 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | 0.0333 | 0.0625 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.0700 | 0.0753 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.0954 | 0.4001 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.0755 | 10.1108 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1829 | 1.0975 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0151 | -0.0169 |
| proposed_contextual_controlled_vs_baseline_no_context | quest_state_correctness | 0.0664 | 0.6464 |
| proposed_contextual_controlled_vs_baseline_no_context | lore_consistency | 0.0062 | 0.9650 |
| proposed_contextual_controlled_vs_baseline_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | objective_completion_support | 0.1083 | 5.6034 |
| proposed_contextual_controlled_vs_baseline_no_context | gameplay_usefulness | 0.0651 | 0.2499 |
| proposed_contextual_controlled_vs_baseline_no_context | time_pressure_acceptability | 0.0565 | 0.2035 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.0982 | nan |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0226 | 0.9078 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2333 | 7.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | -0.0187 | -0.0268 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0827 | -0.0834 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.0200 | 0.0366 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.1400 | 0.1628 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1009 | 0.4329 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.0757 | 10.4714 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1738 | 0.9882 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | -0.0190 | -0.0213 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.0665 | 0.6478 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | lore_consistency | 0.0087 | 2.1304 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.1277 | nan |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 0.0735 | 0.2915 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 0.0635 | 0.2349 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.0982 | nan |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0234 | 0.9697 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.2000 | 3.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | 0.0687 | 0.1122 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0818 | -0.0825 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | 0.0333 | 0.0625 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.0700 | 0.0753 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.0954 | 0.4001 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.1774 | (0.0643, 0.2904) | 0.0013 | 0.1774 | (0.0875, 0.3122) | 0.0000 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0267 | (-0.0800, 0.1067) | 0.3813 | 0.0267 | (0.0000, 0.0444) | 0.2577 |
| proposed_vs_candidate_no_context | naturalness | -0.0056 | (-0.0830, 0.0859) | 0.5810 | -0.0056 | (-0.0756, 0.0992) | 0.7660 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.1397 | (0.0436, 0.2325) | 0.0023 | 0.1397 | (0.0686, 0.2464) | 0.0000 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0841 | (0.0291, 0.1391) | 0.0003 | 0.0841 | (0.0609, 0.1189) | 0.0000 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | 0.0547 | (-0.0400, 0.1470) | 0.1693 | 0.0547 | (0.0275, 0.0728) | 0.0000 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 0.0555 | (-0.0367, 0.1387) | 0.1183 | 0.0555 | (0.0008, 0.1376) | 0.0000 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | -0.0083 | (-0.0935, 0.0534) | 0.5630 | -0.0083 | (-0.0316, 0.0267) | 0.7467 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.2382 | (0.0909, 0.3818) | 0.0007 | 0.2382 | (0.1242, 0.4091) | 0.0000 |
| proposed_vs_candidate_no_context | context_overlap | 0.0355 | (-0.0062, 0.0771) | 0.0570 | 0.0355 | (0.0017, 0.0861) | 0.0000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0333 | (-0.0667, 0.1333) | 0.3693 | 0.0333 | (0.0000, 0.0556) | 0.2470 |
| proposed_vs_candidate_no_context | persona_style | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | distinct1 | -0.0150 | (-0.0462, 0.0281) | 0.7370 | -0.0150 | (-0.0334, 0.0126) | 0.7470 |
| proposed_vs_candidate_no_context | length_score | -0.0333 | (-0.3667, 0.3933) | 0.6033 | -0.0333 | (-0.3111, 0.3833) | 0.7433 |
| proposed_vs_candidate_no_context | sentence_score | 0.0700 | (0.0000, 0.2100) | 0.3273 | 0.0700 | (0.0000, 0.1750) | 0.2513 |
| proposed_vs_candidate_no_context | overall_quality | 0.0918 | (0.0084, 0.1591) | 0.0163 | 0.0918 | (0.0430, 0.1650) | 0.0000 |
| proposed_vs_baseline_no_context | context_relevance | 0.1909 | (0.0672, 0.3023) | 0.0003 | 0.1909 | (0.1139, 0.3065) | 0.0000 |
| proposed_vs_baseline_no_context | persona_consistency | 0.0471 | (-0.0658, 0.1804) | 0.2770 | 0.0471 | (-0.0156, 0.0889) | 0.2437 |
| proposed_vs_baseline_no_context | naturalness | -0.0290 | (-0.1296, 0.0804) | 0.6823 | -0.0290 | (-0.1124, 0.0961) | 0.7447 |
| proposed_vs_baseline_no_context | quest_state_correctness | 0.1515 | (0.0622, 0.2408) | 0.0000 | 0.1515 | (0.0897, 0.2443) | 0.0000 |
| proposed_vs_baseline_no_context | lore_consistency | 0.0870 | (0.0206, 0.1534) | 0.0043 | 0.0870 | (0.0620, 0.1244) | 0.0000 |
| proposed_vs_baseline_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_baseline_no_context | objective_completion_support | 0.0680 | (-0.0110, 0.1470) | 0.0537 | 0.0680 | (0.0000, 0.1133) | 0.2557 |
| proposed_vs_baseline_no_context | gameplay_usefulness | 0.0577 | (-0.0375, 0.1455) | 0.1183 | 0.0577 | (0.0062, 0.1350) | 0.0000 |
| proposed_vs_baseline_no_context | time_pressure_acceptability | -0.0007 | (-0.0630, 0.0617) | 0.5427 | -0.0007 | (-0.0009, -0.0003) | 1.0000 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.2564 | (0.0927, 0.4000) | 0.0000 | 0.2564 | (0.1545, 0.4091) | 0.0000 |
| proposed_vs_baseline_no_context | context_overlap | 0.0383 | (0.0022, 0.0761) | 0.0113 | 0.0383 | (0.0190, 0.0671) | 0.0000 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 0.0667 | (-0.0667, 0.2333) | 0.2580 | 0.0667 | (0.0000, 0.1111) | 0.2697 |
| proposed_vs_baseline_no_context | persona_style | -0.0312 | (-0.0938, 0.0000) | 1.0000 | -0.0312 | (-0.0781, 0.0000) | 1.0000 |
| proposed_vs_baseline_no_context | distinct1 | -0.0475 | (-0.0810, -0.0014) | 0.9807 | -0.0475 | (-0.0769, -0.0036) | 1.0000 |
| proposed_vs_baseline_no_context | length_score | -0.1200 | (-0.5800, 0.3733) | 0.6313 | -0.1200 | (-0.4667, 0.4000) | 0.7580 |
| proposed_vs_baseline_no_context | sentence_score | 0.1400 | (0.0000, 0.2800) | 0.0800 | 0.1400 | (0.1167, 0.1750) | 0.0000 |
| proposed_vs_baseline_no_context | overall_quality | 0.1016 | (0.0061, 0.1815) | 0.0193 | 0.1016 | (0.0642, 0.1576) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 0.1912 | (0.0667, 0.3030) | 0.0003 | 0.1912 | (0.1124, 0.3094) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | 0.0379 | (-0.0801, 0.1713) | 0.3143 | 0.0379 | (-0.0385, 0.0889) | 0.2410 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | -0.0330 | (-0.1308, 0.0688) | 0.7410 | -0.0330 | (-0.1201, 0.0977) | 0.7527 |
| proposed_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.1516 | (0.0622, 0.2411) | 0.0003 | 0.1516 | (0.0891, 0.2454) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | lore_consistency | 0.0894 | (0.0291, 0.1516) | 0.0000 | 0.0894 | (0.0691, 0.1198) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.0873 | (0.0277, 0.1470) | 0.0007 | 0.0873 | (0.0275, 0.1272) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 0.0661 | (-0.0097, 0.1302) | 0.0317 | 0.0661 | (0.0159, 0.1415) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 0.0064 | (-0.0412, 0.0478) | 0.3840 | 0.0064 | (0.0000, 0.0159) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.2564 | (0.0927, 0.4000) | 0.0013 | 0.2564 | (0.1545, 0.4091) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 0.0391 | (0.0013, 0.0768) | 0.0240 | 0.0391 | (0.0140, 0.0767) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.0333 | (-0.1333, 0.2000) | 0.4420 | 0.0333 | (-0.0833, 0.1111) | 0.2553 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | 0.0563 | (0.0000, 0.1688) | 0.3113 | 0.0563 | (0.0000, 0.1406) | 0.2620 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | -0.0466 | (-0.0686, -0.0210) | 1.0000 | -0.0466 | (-0.0614, -0.0244) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | -0.1067 | (-0.5400, 0.3267) | 0.6737 | -0.1067 | (-0.4778, 0.4500) | 0.7447 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | 0.0700 | (0.0000, 0.2100) | 0.3190 | 0.0700 | (0.0000, 0.1750) | 0.2503 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | 0.0961 | (0.0006, 0.1779) | 0.0210 | 0.0961 | (0.0621, 0.1471) | 0.0000 |
| controlled_vs_proposed_raw | context_relevance | -0.1154 | (-0.2812, 0.0686) | 0.9070 | -0.1154 | (-0.2990, 0.0070) | 0.7577 |
| controlled_vs_proposed_raw | persona_consistency | 0.1358 | (-0.0000, 0.2475) | 0.0267 | 0.1358 | (0.0889, 0.2062) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0139 | (-0.0584, 0.0793) | 0.3603 | 0.0139 | (-0.0659, 0.0672) | 0.2497 |
| controlled_vs_proposed_raw | quest_state_correctness | -0.0852 | (-0.2234, 0.0736) | 0.8580 | -0.0852 | (-0.2343, 0.0143) | 0.7597 |
| controlled_vs_proposed_raw | lore_consistency | -0.0807 | (-0.1461, -0.0154) | 0.9900 | -0.0807 | (-0.1300, -0.0479) | 1.0000 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0403 | (-0.0233, 0.1000) | 0.1080 | 0.0403 | (0.0178, 0.0742) | 0.0000 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0074 | (-0.0503, 0.0653) | 0.4107 | 0.0074 | (-0.0670, 0.0570) | 0.2493 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 0.0572 | (0.0033, 0.1238) | 0.0187 | 0.0572 | (0.0297, 0.0755) | 0.0000 |
| controlled_vs_proposed_raw | context_keyword_coverage | -0.1582 | (-0.3818, 0.1073) | 0.8970 | -0.1582 | (-0.4091, 0.0091) | 0.7517 |
| controlled_vs_proposed_raw | context_overlap | -0.0157 | (-0.0559, 0.0236) | 0.7773 | -0.0157 | (-0.0422, 0.0020) | 0.7437 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1667 | (0.0000, 0.3000) | 0.0323 | 0.1667 | (0.1111, 0.2500) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.0125 | (0.0000, 0.0375) | 0.3170 | 0.0125 | (0.0000, 0.0312) | 0.2463 |
| controlled_vs_proposed_raw | distinct1 | -0.0352 | (-0.0760, 0.0060) | 0.9377 | -0.0352 | (-0.0898, 0.0012) | 0.7393 |
| controlled_vs_proposed_raw | length_score | 0.1400 | (-0.1400, 0.4400) | 0.2070 | 0.1400 | (-0.1500, 0.3333) | 0.2450 |
| controlled_vs_proposed_raw | sentence_score | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_proposed_raw | overall_quality | -0.0007 | (-0.1065, 0.1126) | 0.5363 | -0.0007 | (-0.0767, 0.0500) | 0.7477 |
| controlled_vs_candidate_no_context | context_relevance | 0.0619 | (-0.0010, 0.1771) | 0.0340 | 0.0619 | (0.0132, 0.0944) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1625 | (0.0267, 0.2983) | 0.0130 | 0.1625 | (0.1333, 0.2062) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.0083 | (-0.0175, 0.0393) | 0.2797 | 0.0083 | (-0.0084, 0.0333) | 0.2617 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.0546 | (-0.0002, 0.1542) | 0.0337 | 0.0546 | (0.0121, 0.0829) | 0.0000 |
| controlled_vs_candidate_no_context | lore_consistency | 0.0034 | (-0.0123, 0.0216) | 0.3790 | 0.0034 | (-0.0110, 0.0130) | 0.2483 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | 0.0950 | (0.0353, 0.1523) | 0.0007 | 0.0950 | (0.0906, 0.1017) | 0.0000 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0629 | (0.0239, 0.0992) | 0.0007 | 0.0629 | (0.0578, 0.0705) | 0.0000 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 0.0489 | (0.0178, 0.0800) | 0.0010 | 0.0489 | (0.0439, 0.0564) | 0.0000 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.0800 | (0.0000, 0.2400) | 0.3110 | 0.0800 | (0.0000, 0.1333) | 0.2510 |
| controlled_vs_candidate_no_context | context_overlap | 0.0198 | (-0.0063, 0.0419) | 0.0590 | 0.0198 | (0.0037, 0.0439) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2000 | (0.0333, 0.3667) | 0.0120 | 0.2000 | (0.1667, 0.2500) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.0125 | (0.0000, 0.0375) | 0.3273 | 0.0125 | (0.0000, 0.0312) | 0.2513 |
| controlled_vs_candidate_no_context | distinct1 | -0.0502 | (-0.0878, -0.0184) | 1.0000 | -0.0502 | (-0.0772, -0.0321) | 1.0000 |
| controlled_vs_candidate_no_context | length_score | 0.1067 | (0.0000, 0.2600) | 0.0287 | 0.1067 | (0.0222, 0.2333) | 0.0000 |
| controlled_vs_candidate_no_context | sentence_score | 0.0700 | (0.0000, 0.2100) | 0.3337 | 0.0700 | (0.0000, 0.1750) | 0.2540 |
| controlled_vs_candidate_no_context | overall_quality | 0.0911 | (0.0129, 0.1900) | 0.0083 | 0.0911 | (0.0883, 0.0930) | 0.0000 |
| controlled_vs_baseline_no_context | context_relevance | 0.0755 | (0.0049, 0.1906) | 0.0000 | 0.0755 | (0.0075, 0.1209) | 0.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1829 | (0.0762, 0.2933) | 0.0003 | 0.1829 | (0.1778, 0.1906) | 0.0000 |
| controlled_vs_baseline_no_context | naturalness | -0.0151 | (-0.0551, 0.0342) | 0.7547 | -0.0151 | (-0.0453, 0.0301) | 0.7617 |
| controlled_vs_baseline_no_context | quest_state_correctness | 0.0664 | (0.0111, 0.1612) | 0.0000 | 0.0664 | (0.0100, 0.1040) | 0.0000 |
| controlled_vs_baseline_no_context | lore_consistency | 0.0062 | (-0.0128, 0.0254) | 0.2633 | 0.0062 | (-0.0055, 0.0141) | 0.2430 |
| controlled_vs_baseline_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_baseline_no_context | objective_completion_support | 0.1083 | (0.0453, 0.1597) | 0.0000 | 0.1083 | (0.0742, 0.1311) | 0.0000 |
| controlled_vs_baseline_no_context | gameplay_usefulness | 0.0651 | (0.0230, 0.1072) | 0.0000 | 0.0651 | (0.0632, 0.0680) | 0.0000 |
| controlled_vs_baseline_no_context | time_pressure_acceptability | 0.0565 | (0.0090, 0.0866) | 0.0047 | 0.0565 | (0.0294, 0.0746) | 0.0000 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.0982 | (0.0000, 0.2582) | 0.0737 | 0.0982 | (0.0000, 0.1636) | 0.2530 |
| controlled_vs_baseline_no_context | context_overlap | 0.0226 | (0.0081, 0.0398) | 0.0000 | 0.0226 | (0.0210, 0.0250) | 0.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2333 | (0.1000, 0.3667) | 0.0007 | 0.2333 | (0.2222, 0.2500) | 0.0000 |
| controlled_vs_baseline_no_context | persona_style | -0.0187 | (-0.0563, 0.0000) | 1.0000 | -0.0187 | (-0.0469, 0.0000) | 1.0000 |
| controlled_vs_baseline_no_context | distinct1 | -0.0827 | (-0.1058, -0.0601) | 1.0000 | -0.0827 | (-0.0934, -0.0756) | 1.0000 |
| controlled_vs_baseline_no_context | length_score | 0.0200 | (-0.2467, 0.2733) | 0.4447 | 0.0200 | (-0.1333, 0.2500) | 0.2517 |
| controlled_vs_baseline_no_context | sentence_score | 0.1400 | (0.0000, 0.2800) | 0.0750 | 0.1400 | (0.1167, 0.1750) | 0.0000 |
| controlled_vs_baseline_no_context | overall_quality | 0.1009 | (0.0248, 0.1992) | 0.0000 | 0.1009 | (0.0809, 0.1142) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.0757 | (0.0079, 0.1791) | 0.0000 | 0.0757 | (0.0104, 0.1193) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1738 | (0.0671, 0.2933) | 0.0003 | 0.1738 | (0.1677, 0.1778) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | -0.0190 | (-0.0614, 0.0233) | 0.8120 | -0.0190 | (-0.0529, 0.0318) | 0.7507 |
| controlled_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.0665 | (0.0117, 0.1601) | 0.0000 | 0.0665 | (0.0111, 0.1034) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | lore_consistency | 0.0087 | (-0.0081, 0.0255) | 0.1860 | 0.0087 | (-0.0102, 0.0212) | 0.2723 |
| controlled_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.1277 | (0.0783, 0.1680) | 0.0000 | 0.1277 | (0.1017, 0.1450) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 0.0735 | (0.0535, 0.0934) | 0.0000 | 0.0735 | (0.0729, 0.0745) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 0.0635 | (0.0390, 0.0841) | 0.0000 | 0.0635 | (0.0457, 0.0755) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.0982 | (0.0000, 0.2582) | 0.0817 | 0.0982 | (0.0000, 0.1636) | 0.2437 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0234 | (0.0093, 0.0351) | 0.0003 | 0.0234 | (0.0160, 0.0345) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.2000 | (0.0667, 0.3667) | 0.0003 | 0.2000 | (0.1667, 0.2222) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | 0.0688 | (0.0000, 0.2062) | 0.3310 | 0.0688 | (0.0000, 0.1719) | 0.2483 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0818 | (-0.1111, -0.0382) | 1.0000 | -0.0818 | (-0.1142, -0.0601) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | 0.0333 | (-0.2333, 0.2600) | 0.3847 | 0.0333 | (-0.1444, 0.3000) | 0.2503 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.0700 | (0.0000, 0.2100) | 0.3287 | 0.0700 | (0.0000, 0.1750) | 0.2497 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.0954 | (0.0220, 0.1889) | 0.0023 | 0.0954 | (0.0704, 0.1121) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.0755 | (0.0049, 0.1906) | 0.0000 | 0.0755 | (0.0075, 0.1209) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1829 | (0.0762, 0.2933) | 0.0003 | 0.1829 | (0.1778, 0.1906) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0151 | (-0.0551, 0.0346) | 0.7520 | -0.0151 | (-0.0453, 0.0301) | 0.7477 |
| proposed_contextual_controlled_vs_baseline_no_context | quest_state_correctness | 0.0664 | (0.0111, 0.1636) | 0.0000 | 0.0664 | (0.0100, 0.1040) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | lore_consistency | 0.0062 | (-0.0129, 0.0254) | 0.2940 | 0.0062 | (-0.0055, 0.0141) | 0.2563 |
| proposed_contextual_controlled_vs_baseline_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | objective_completion_support | 0.1083 | (0.0430, 0.1597) | 0.0000 | 0.1083 | (0.0742, 0.1311) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | gameplay_usefulness | 0.0651 | (0.0230, 0.1045) | 0.0000 | 0.0651 | (0.0632, 0.0680) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | time_pressure_acceptability | 0.0565 | (0.0083, 0.0858) | 0.0057 | 0.0565 | (0.0294, 0.0746) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.0982 | (0.0000, 0.2582) | 0.0790 | 0.0982 | (0.0000, 0.1636) | 0.2583 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0226 | (0.0074, 0.0398) | 0.0000 | 0.0226 | (0.0210, 0.0250) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2333 | (0.1000, 0.4000) | 0.0010 | 0.2333 | (0.2222, 0.2500) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | -0.0187 | (-0.0563, 0.0000) | 1.0000 | -0.0187 | (-0.0469, 0.0000) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0827 | (-0.1074, -0.0601) | 1.0000 | -0.0827 | (-0.0934, -0.0756) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.0200 | (-0.2467, 0.2867) | 0.4280 | 0.0200 | (-0.1333, 0.2500) | 0.2550 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.1400 | (0.0000, 0.2800) | 0.0800 | 0.1400 | (0.1167, 0.1750) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1009 | (0.0248, 0.1993) | 0.0000 | 0.1009 | (0.0809, 0.1142) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.0757 | (0.0076, 0.1879) | 0.0000 | 0.0757 | (0.0104, 0.1193) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1738 | (0.0800, 0.3071) | 0.0000 | 0.1738 | (0.1677, 0.1778) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | -0.0190 | (-0.0614, 0.0233) | 0.8193 | -0.0190 | (-0.0529, 0.0318) | 0.7540 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.0665 | (0.0117, 0.1626) | 0.0000 | 0.0665 | (0.0111, 0.1034) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | lore_consistency | 0.0087 | (-0.0094, 0.0265) | 0.1863 | 0.0087 | (-0.0102, 0.0212) | 0.2517 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.1277 | (0.0783, 0.1680) | 0.0000 | 0.1277 | (0.1017, 0.1450) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 0.0735 | (0.0537, 0.0934) | 0.0000 | 0.0735 | (0.0729, 0.0745) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 0.0635 | (0.0371, 0.0841) | 0.0000 | 0.0635 | (0.0457, 0.0755) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.0982 | (0.0000, 0.2582) | 0.0680 | 0.0982 | (0.0000, 0.1636) | 0.2553 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0234 | (0.0093, 0.0354) | 0.0007 | 0.0234 | (0.0160, 0.0345) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.2000 | (0.0667, 0.3667) | 0.0000 | 0.2000 | (0.1667, 0.2222) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | 0.0688 | (0.0000, 0.2062) | 0.3420 | 0.0688 | (0.0000, 0.1719) | 0.2440 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0818 | (-0.1114, -0.0421) | 1.0000 | -0.0818 | (-0.1142, -0.0601) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | 0.0333 | (-0.2200, 0.2600) | 0.3887 | 0.0333 | (-0.1444, 0.3000) | 0.2487 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.0700 | (0.0000, 0.2100) | 0.3207 | 0.0700 | (0.0000, 0.1750) | 0.2583 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.0954 | (0.0184, 0.1924) | 0.0017 | 0.0954 | (0.0704, 0.1121) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 4 | 1 | 0 | 0.8000 | 0.8000 |
| proposed_vs_candidate_no_context | persona_consistency | 2 | 1 | 2 | 0.6000 | 0.6667 |
| proposed_vs_candidate_no_context | naturalness | 2 | 3 | 0 | 0.4000 | 0.4000 |
| proposed_vs_candidate_no_context | quest_state_correctness | 4 | 1 | 0 | 0.8000 | 0.8000 |
| proposed_vs_candidate_no_context | lore_consistency | 4 | 0 | 1 | 0.9000 | 1.0000 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 5 | 0.5000 | nan |
| proposed_vs_candidate_no_context | objective_completion_support | 3 | 1 | 1 | 0.7000 | 0.7500 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 4 | 1 | 0 | 0.8000 | 0.8000 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | 2 | 2 | 1 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 4 | 0 | 1 | 0.9000 | 1.0000 |
| proposed_vs_candidate_no_context | context_overlap | 3 | 2 | 0 | 0.6000 | 0.6000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 2 | 1 | 2 | 0.6000 | 0.6667 |
| proposed_vs_candidate_no_context | persona_style | 0 | 0 | 5 | 0.5000 | nan |
| proposed_vs_candidate_no_context | distinct1 | 1 | 4 | 0 | 0.2000 | 0.2000 |
| proposed_vs_candidate_no_context | length_score | 1 | 3 | 1 | 0.3000 | 0.2500 |
| proposed_vs_candidate_no_context | sentence_score | 1 | 0 | 4 | 0.6000 | 1.0000 |
| proposed_vs_candidate_no_context | overall_quality | 4 | 1 | 0 | 0.8000 | 0.8000 |
| proposed_vs_baseline_no_context | context_relevance | 4 | 1 | 0 | 0.8000 | 0.8000 |
| proposed_vs_baseline_no_context | persona_consistency | 2 | 2 | 1 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context | naturalness | 2 | 3 | 0 | 0.4000 | 0.4000 |
| proposed_vs_baseline_no_context | quest_state_correctness | 4 | 1 | 0 | 0.8000 | 0.8000 |
| proposed_vs_baseline_no_context | lore_consistency | 4 | 1 | 0 | 0.8000 | 0.8000 |
| proposed_vs_baseline_no_context | multi_turn_contradiction_safety | 0 | 0 | 5 | 0.5000 | nan |
| proposed_vs_baseline_no_context | objective_completion_support | 3 | 1 | 1 | 0.7000 | 0.7500 |
| proposed_vs_baseline_no_context | gameplay_usefulness | 4 | 1 | 0 | 0.8000 | 0.8000 |
| proposed_vs_baseline_no_context | time_pressure_acceptability | 2 | 3 | 0 | 0.4000 | 0.4000 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 4 | 0 | 1 | 0.9000 | 1.0000 |
| proposed_vs_baseline_no_context | context_overlap | 4 | 1 | 0 | 0.8000 | 0.8000 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 2 | 1 | 2 | 0.6000 | 0.6667 |
| proposed_vs_baseline_no_context | persona_style | 0 | 1 | 4 | 0.4000 | 0.0000 |
| proposed_vs_baseline_no_context | distinct1 | 1 | 4 | 0 | 0.2000 | 0.2000 |
| proposed_vs_baseline_no_context | length_score | 2 | 3 | 0 | 0.4000 | 0.4000 |
| proposed_vs_baseline_no_context | sentence_score | 2 | 0 | 3 | 0.7000 | 1.0000 |
| proposed_vs_baseline_no_context | overall_quality | 4 | 1 | 0 | 0.8000 | 0.8000 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 4 | 1 | 0 | 0.8000 | 0.8000 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | 2 | 2 | 1 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | 2 | 3 | 0 | 0.4000 | 0.4000 |
| proposed_vs_baseline_no_context_phi3_latest | quest_state_correctness | 4 | 1 | 0 | 0.8000 | 0.8000 |
| proposed_vs_baseline_no_context_phi3_latest | lore_consistency | 4 | 0 | 1 | 0.9000 | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0 | 0 | 5 | 0.5000 | nan |
| proposed_vs_baseline_no_context_phi3_latest | objective_completion_support | 4 | 0 | 1 | 0.9000 | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 4 | 1 | 0 | 0.8000 | 0.8000 |
| proposed_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 2 | 2 | 1 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 4 | 0 | 1 | 0.9000 | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 3 | 2 | 0 | 0.6000 | 0.6000 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 2 | 2 | 1 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | 1 | 0 | 4 | 0.6000 | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | 0 | 4 | 1 | 0.1000 | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | 2 | 3 | 0 | 0.4000 | 0.4000 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | 1 | 0 | 4 | 0.6000 | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | 4 | 1 | 0 | 0.8000 | 0.8000 |
| controlled_vs_proposed_raw | context_relevance | 2 | 3 | 0 | 0.4000 | 0.4000 |
| controlled_vs_proposed_raw | persona_consistency | 4 | 1 | 0 | 0.8000 | 0.8000 |
| controlled_vs_proposed_raw | naturalness | 3 | 2 | 0 | 0.6000 | 0.6000 |
| controlled_vs_proposed_raw | quest_state_correctness | 2 | 3 | 0 | 0.4000 | 0.4000 |
| controlled_vs_proposed_raw | lore_consistency | 1 | 3 | 1 | 0.3000 | 0.2500 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0 | 0 | 5 | 0.5000 | nan |
| controlled_vs_proposed_raw | objective_completion_support | 4 | 1 | 0 | 0.8000 | 0.8000 |
| controlled_vs_proposed_raw | gameplay_usefulness | 3 | 2 | 0 | 0.6000 | 0.6000 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 4 | 1 | 0 | 0.8000 | 0.8000 |
| controlled_vs_proposed_raw | context_keyword_coverage | 1 | 3 | 1 | 0.3000 | 0.2500 |
| controlled_vs_proposed_raw | context_overlap | 2 | 3 | 0 | 0.4000 | 0.4000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 4 | 1 | 0 | 0.8000 | 0.8000 |
| controlled_vs_proposed_raw | persona_style | 1 | 0 | 4 | 0.6000 | 1.0000 |
| controlled_vs_proposed_raw | distinct1 | 1 | 4 | 0 | 0.2000 | 0.2000 |
| controlled_vs_proposed_raw | length_score | 4 | 1 | 0 | 0.8000 | 0.8000 |
| controlled_vs_proposed_raw | sentence_score | 0 | 0 | 5 | 0.5000 | nan |
| controlled_vs_proposed_raw | overall_quality | 2 | 3 | 0 | 0.4000 | 0.4000 |
| controlled_vs_candidate_no_context | context_relevance | 3 | 1 | 1 | 0.7000 | 0.7500 |
| controlled_vs_candidate_no_context | persona_consistency | 3 | 0 | 2 | 0.8000 | 1.0000 |
| controlled_vs_candidate_no_context | naturalness | 3 | 1 | 1 | 0.7000 | 0.7500 |
| controlled_vs_candidate_no_context | quest_state_correctness | 3 | 1 | 1 | 0.7000 | 0.7500 |
| controlled_vs_candidate_no_context | lore_consistency | 2 | 1 | 2 | 0.6000 | 0.6667 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 5 | 0.5000 | nan |
| controlled_vs_candidate_no_context | objective_completion_support | 4 | 0 | 1 | 0.9000 | 1.0000 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 4 | 0 | 1 | 0.9000 | 1.0000 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 4 | 0 | 1 | 0.9000 | 1.0000 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 1 | 0 | 4 | 0.6000 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 3 | 1 | 1 | 0.7000 | 0.7500 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 3 | 0 | 2 | 0.8000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_style | 1 | 0 | 4 | 0.6000 | 1.0000 |
| controlled_vs_candidate_no_context | distinct1 | 0 | 4 | 1 | 0.1000 | 0.0000 |
| controlled_vs_candidate_no_context | length_score | 3 | 1 | 1 | 0.7000 | 0.7500 |
| controlled_vs_candidate_no_context | sentence_score | 1 | 0 | 4 | 0.6000 | 1.0000 |
| controlled_vs_candidate_no_context | overall_quality | 3 | 1 | 1 | 0.7000 | 0.7500 |
| controlled_vs_baseline_no_context | context_relevance | 5 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 4 | 0 | 1 | 0.9000 | 1.0000 |
| controlled_vs_baseline_no_context | naturalness | 1 | 4 | 0 | 0.2000 | 0.2000 |
| controlled_vs_baseline_no_context | quest_state_correctness | 5 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | lore_consistency | 2 | 2 | 1 | 0.5000 | 0.5000 |
| controlled_vs_baseline_no_context | multi_turn_contradiction_safety | 0 | 0 | 5 | 0.5000 | nan |
| controlled_vs_baseline_no_context | objective_completion_support | 4 | 1 | 0 | 0.8000 | 0.8000 |
| controlled_vs_baseline_no_context | gameplay_usefulness | 5 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | time_pressure_acceptability | 4 | 1 | 0 | 0.8000 | 0.8000 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 2 | 0 | 3 | 0.7000 | 1.0000 |
| controlled_vs_baseline_no_context | context_overlap | 5 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 4 | 0 | 1 | 0.9000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_style | 0 | 1 | 4 | 0.4000 | 0.0000 |
| controlled_vs_baseline_no_context | distinct1 | 0 | 5 | 0 | 0.0000 | 0.0000 |
| controlled_vs_baseline_no_context | length_score | 3 | 2 | 0 | 0.6000 | 0.6000 |
| controlled_vs_baseline_no_context | sentence_score | 2 | 0 | 3 | 0.7000 | 1.0000 |
| controlled_vs_baseline_no_context | overall_quality | 4 | 1 | 0 | 0.8000 | 0.8000 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 5 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 4 | 0 | 1 | 0.9000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | 2 | 3 | 0 | 0.4000 | 0.4000 |
| controlled_vs_baseline_no_context_phi3_latest | quest_state_correctness | 5 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | lore_consistency | 2 | 1 | 2 | 0.6000 | 0.6667 |
| controlled_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0 | 0 | 5 | 0.5000 | nan |
| controlled_vs_baseline_no_context_phi3_latest | objective_completion_support | 5 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 5 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 5 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 2 | 0 | 3 | 0.7000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 4 | 1 | 0 | 0.8000 | 0.8000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 4 | 0 | 1 | 0.9000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | 1 | 0 | 4 | 0.6000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | 0 | 5 | 0 | 0.0000 | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | 3 | 2 | 0 | 0.6000 | 0.6000 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 1 | 0 | 4 | 0.6000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 4 | 1 | 0 | 0.8000 | 0.8000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 5 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 4 | 0 | 1 | 0.9000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 1 | 4 | 0 | 0.2000 | 0.2000 |
| proposed_contextual_controlled_vs_baseline_no_context | quest_state_correctness | 5 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | lore_consistency | 2 | 2 | 1 | 0.5000 | 0.5000 |
| proposed_contextual_controlled_vs_baseline_no_context | multi_turn_contradiction_safety | 0 | 0 | 5 | 0.5000 | nan |
| proposed_contextual_controlled_vs_baseline_no_context | objective_completion_support | 4 | 1 | 0 | 0.8000 | 0.8000 |
| proposed_contextual_controlled_vs_baseline_no_context | gameplay_usefulness | 5 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | time_pressure_acceptability | 4 | 1 | 0 | 0.8000 | 0.8000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 2 | 0 | 3 | 0.7000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 5 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 4 | 0 | 1 | 0.9000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0 | 1 | 4 | 0.4000 | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | 0 | 5 | 0 | 0.0000 | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 3 | 2 | 0 | 0.6000 | 0.6000 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 2 | 0 | 3 | 0.7000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 4 | 1 | 0 | 0.8000 | 0.8000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 5 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 4 | 0 | 1 | 0.9000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | 2 | 3 | 0 | 0.4000 | 0.4000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | quest_state_correctness | 5 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | lore_consistency | 2 | 1 | 2 | 0.6000 | 0.6667 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0 | 0 | 5 | 0.5000 | nan |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | objective_completion_support | 5 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 5 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 5 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 2 | 0 | 3 | 0.7000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 4 | 1 | 0 | 0.8000 | 0.8000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 4 | 0 | 1 | 0.9000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | 1 | 0 | 4 | 0.6000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | 0 | 5 | 0 | 0.0000 | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | 3 | 2 | 0 | 0.6000 | 0.6000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | 1 | 0 | 4 | 0.6000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 4 | 1 | 0 | 0.8000 | 0.8000 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.6000 | 0.0000 | 1.0000 |
| proposed_contextual | 0.0000 | 0.0000 | 0.2000 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.2000 | 0.0000 | 0.0000 |
| baseline_no_context | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline_no_context_phi3_latest | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `2`
- Unique template signatures: `5`
- Template signature ratio: `1.0000`
- Effective sample size by source clustering: `1.92`
- Effective sample size by template-signature clustering: `5.00`
- Detailed diagnostics are published in `scenario_dependence.json`.

## Multi-turn Contradiction
| Arm | Contradiction Rate | Contradiction Safety | Contradicted Sources | Source Count |
|---|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 1.0000 | 0 | 2 |
| proposed_contextual | 0.0000 | 1.0000 | 0 | 2 |
| candidate_no_context | 0.0000 | 1.0000 | 0 | 2 |
| baseline_no_context | 0.0000 | 1.0000 | 0 | 2 |
| baseline_no_context_phi3_latest | 0.0000 | 1.0000 | 0 | 2 |
- Detailed source-level values are published in `multi_turn_contradictions.json`.

- BERTScore status: unavailable (No module named 'bert_score').

This report covers proposal RO5 metrics plus game-facing outcomes: quest-state correctness, lore consistency, contradiction safety, objective completion support, gameplay usefulness, and time-pressure acceptability.
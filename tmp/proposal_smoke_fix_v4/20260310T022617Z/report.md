# Proposal Alignment Evaluation Report

- Run ID: `20260310T022617Z`
- Generated: `2026-03-10T02:28:23.165352+00:00`
- Scenarios: `tmp\proposal_smoke_fix_v4\20260310T022617Z\scenarios.jsonl`
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
| proposed_contextual_controlled | 0.1525 (0.0247, 0.3344) | 0.2733 (0.2333, 0.3133) | 0.8467 (0.7865, 0.8849) | 0.3311 (0.2772, 0.4093) | n/a |
| proposed_contextual | 0.1479 (0.1055, 0.1924) | 0.2200 (0.1533, 0.3000) | 0.9097 (0.8821, 0.9381) | 0.3219 (0.2917, 0.3686) | n/a |
| candidate_no_context | 0.0477 (0.0050, 0.1287) | 0.1667 (0.1200, 0.2133) | 0.8588 (0.7879, 0.9297) | 0.2453 (0.2250, 0.2655) | n/a |
| baseline_no_context | 0.0345 (0.0047, 0.0880) | 0.1933 (0.1000, 0.2867) | 0.9068 (0.8483, 0.9547) | 0.2587 (0.2171, 0.3060) | n/a |
| baseline_no_context_phi3_latest | 0.0086 (0.0043, 0.0118) | 0.1351 (0.1000, 0.1751) | 0.9277 (0.8784, 0.9692) | 0.2292 (0.2100, 0.2499) | n/a |

## Game-facing Outcome Metrics (mean, 95% CI)
| Arm | Quest-state Correctness | Lore Consistency | Contradiction Safety | Objective Completion Support | Gameplay Usefulness | Time-pressure Acceptability |
|---|---:|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2338 (0.1212, 0.3898) | 0.0767 (0.0000, 0.2300) | 1.0000 (1.0000, 1.0000) | 0.1170 (0.0740, 0.1577) | 0.3147 (0.3044, 0.3251) | 0.2898 (0.2150, 0.3539) |
| proposed_contextual | 0.2200 (0.1816, 0.2568) | 0.0786 (0.0180, 0.1393) | 1.0000 (1.0000, 1.0000) | 0.0630 (0.0167, 0.1093) | 0.3248 (0.3079, 0.3363) | 0.3004 (0.2804, 0.3248) |
| candidate_no_context | 0.1350 (0.1015, 0.1999) | 0.0054 (0.0000, 0.0135) | 1.0000 (1.0000, 1.0000) | 0.0247 (0.0000, 0.0573) | 0.2480 (0.2177, 0.2806) | 0.2599 (0.1948, 0.3058) |
| baseline_no_context | 0.1263 (0.1015, 0.1734) | 0.0049 (0.0000, 0.0147) | 1.0000 (1.0000, 1.0000) | 0.0490 (0.0243, 0.0733) | 0.2821 (0.2637, 0.3002) | 0.2734 (0.2342, 0.3030) |
| baseline_no_context_phi3_latest | 0.1060 (0.1015, 0.1130) | 0.0030 (0.0000, 0.0089) | 1.0000 (1.0000, 1.0000) | 0.0273 (0.0000, 0.0600) | 0.2784 (0.2529, 0.3048) | 0.2893 (0.2663, 0.3123) |

- Multi-turn contradiction rate is reported as `1 - contradiction_safety` in row-level outputs.

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.1002 | 2.1021 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0533 | 0.3200 |
| proposed_vs_candidate_no_context | naturalness | 0.0509 | 0.0593 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.0850 | 0.6295 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0732 | 13.5936 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | 0.0383 | 1.5541 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 0.0768 | 0.3099 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | 0.0404 | 0.1556 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.1291 | 2.3667 |
| proposed_vs_candidate_no_context | context_overlap | 0.0328 | 1.0366 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0667 | 2.0000 |
| proposed_vs_candidate_no_context | persona_style | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | distinct1 | 0.0064 | 0.0067 |
| proposed_vs_candidate_no_context | length_score | 0.2067 | 0.4769 |
| proposed_vs_candidate_no_context | sentence_score | 0.0700 | 0.0814 |
| proposed_vs_candidate_no_context | overall_quality | 0.0766 | 0.3122 |
| proposed_vs_baseline_no_context | context_relevance | 0.1134 | 3.2918 |
| proposed_vs_baseline_no_context | persona_consistency | 0.0267 | 0.1379 |
| proposed_vs_baseline_no_context | naturalness | 0.0029 | 0.0032 |
| proposed_vs_baseline_no_context | quest_state_correctness | 0.0938 | 0.7425 |
| proposed_vs_baseline_no_context | lore_consistency | 0.0737 | 15.0120 |
| proposed_vs_baseline_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| proposed_vs_baseline_no_context | objective_completion_support | 0.0140 | 0.2857 |
| proposed_vs_baseline_no_context | gameplay_usefulness | 0.0427 | 0.1514 |
| proposed_vs_baseline_no_context | time_pressure_acceptability | 0.0270 | 0.0987 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.1473 | 4.0500 |
| proposed_vs_baseline_no_context | context_overlap | 0.0344 | 1.1462 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 0.0333 | 0.5000 |
| proposed_vs_baseline_no_context | persona_style | 0.0000 | 0.0000 |
| proposed_vs_baseline_no_context | distinct1 | -0.0186 | -0.0187 |
| proposed_vs_baseline_no_context | length_score | 0.0867 | 0.1566 |
| proposed_vs_baseline_no_context | sentence_score | -0.0700 | -0.0700 |
| proposed_vs_baseline_no_context | overall_quality | 0.0632 | 0.2443 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 0.1393 | 16.1905 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | 0.0849 | 0.6282 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | -0.0180 | -0.0194 |
| proposed_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.1140 | 1.0751 |
| proposed_vs_baseline_no_context_phi3_latest | lore_consistency | 0.0757 | 25.6204 |
| proposed_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.0357 | 1.3049 |
| proposed_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 0.0464 | 0.1666 |
| proposed_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 0.0111 | 0.0383 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.1836 | nan |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 0.0357 | 1.2448 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1000 | nan |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | 0.0244 | 0.0361 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | -0.0183 | -0.0185 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | -0.0533 | -0.0769 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | 0.0000 | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | 0.0927 | 0.4045 |
| controlled_vs_proposed_raw | context_relevance | 0.0046 | 0.0311 |
| controlled_vs_proposed_raw | persona_consistency | 0.0533 | 0.2424 |
| controlled_vs_proposed_raw | naturalness | -0.0630 | -0.0692 |
| controlled_vs_proposed_raw | quest_state_correctness | 0.0138 | 0.0627 |
| controlled_vs_proposed_raw | lore_consistency | -0.0020 | -0.0250 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0540 | 0.8571 |
| controlled_vs_proposed_raw | gameplay_usefulness | -0.0101 | -0.0312 |
| controlled_vs_proposed_raw | time_pressure_acceptability | -0.0105 | -0.0350 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.0018 | 0.0099 |
| controlled_vs_proposed_raw | context_overlap | 0.0111 | 0.1726 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.0667 | 0.6667 |
| controlled_vs_proposed_raw | persona_style | 0.0000 | 0.0000 |
| controlled_vs_proposed_raw | distinct1 | -0.0241 | -0.0248 |
| controlled_vs_proposed_raw | length_score | -0.2667 | -0.4167 |
| controlled_vs_proposed_raw | sentence_score | 0.0000 | 0.0000 |
| controlled_vs_proposed_raw | overall_quality | 0.0092 | 0.0286 |
| controlled_vs_candidate_no_context | context_relevance | 0.1048 | 2.1987 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1067 | 0.6400 |
| controlled_vs_candidate_no_context | naturalness | -0.0121 | -0.0141 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.0988 | 0.7317 |
| controlled_vs_candidate_no_context | lore_consistency | 0.0713 | 13.2287 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | 0.0923 | 3.7432 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0667 | 0.2690 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 0.0299 | 0.1152 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.1309 | 2.4000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0439 | 1.3881 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1333 | 4.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.0000 | 0.0000 |
| controlled_vs_candidate_no_context | distinct1 | -0.0177 | -0.0183 |
| controlled_vs_candidate_no_context | length_score | -0.0600 | -0.1385 |
| controlled_vs_candidate_no_context | sentence_score | 0.0700 | 0.0814 |
| controlled_vs_candidate_no_context | overall_quality | 0.0858 | 0.3497 |
| controlled_vs_baseline_no_context | context_relevance | 0.1180 | 3.4254 |
| controlled_vs_baseline_no_context | persona_consistency | 0.0800 | 0.4138 |
| controlled_vs_baseline_no_context | naturalness | -0.0601 | -0.0662 |
| controlled_vs_baseline_no_context | quest_state_correctness | 0.1075 | 0.8517 |
| controlled_vs_baseline_no_context | lore_consistency | 0.0718 | 14.6117 |
| controlled_vs_baseline_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_baseline_no_context | objective_completion_support | 0.0680 | 1.3878 |
| controlled_vs_baseline_no_context | gameplay_usefulness | 0.0326 | 0.1155 |
| controlled_vs_baseline_no_context | time_pressure_acceptability | 0.0165 | 0.0603 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.1491 | 4.1000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0455 | 1.5167 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1000 | 1.5000 |
| controlled_vs_baseline_no_context | persona_style | 0.0000 | 0.0000 |
| controlled_vs_baseline_no_context | distinct1 | -0.0427 | -0.0431 |
| controlled_vs_baseline_no_context | length_score | -0.1800 | -0.3253 |
| controlled_vs_baseline_no_context | sentence_score | -0.0700 | -0.0700 |
| controlled_vs_baseline_no_context | overall_quality | 0.0724 | 0.2799 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.1439 | 16.7260 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1382 | 1.0229 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | -0.0810 | -0.0873 |
| controlled_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.1278 | 1.2052 |
| controlled_vs_baseline_no_context_phi3_latest | lore_consistency | 0.0737 | 24.9549 |
| controlled_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.0897 | 3.2805 |
| controlled_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 0.0362 | 0.1302 |
| controlled_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 0.0006 | 0.0020 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.1855 | nan |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0468 | 1.6323 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1667 | nan |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | 0.0244 | 0.0361 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0424 | -0.0428 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | -0.3200 | -0.4615 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.0000 | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1019 | 0.4446 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.1180 | 3.4254 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.0800 | 0.4138 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0601 | -0.0662 |
| proposed_contextual_controlled_vs_baseline_no_context | quest_state_correctness | 0.1075 | 0.8517 |
| proposed_contextual_controlled_vs_baseline_no_context | lore_consistency | 0.0718 | 14.6117 |
| proposed_contextual_controlled_vs_baseline_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | objective_completion_support | 0.0680 | 1.3878 |
| proposed_contextual_controlled_vs_baseline_no_context | gameplay_usefulness | 0.0326 | 0.1155 |
| proposed_contextual_controlled_vs_baseline_no_context | time_pressure_acceptability | 0.0165 | 0.0603 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.1491 | 4.1000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0455 | 1.5167 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1000 | 1.5000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0000 | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0427 | -0.0431 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | -0.1800 | -0.3253 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | -0.0700 | -0.0700 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.0724 | 0.2799 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.1439 | 16.7260 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1382 | 1.0229 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | -0.0810 | -0.0873 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.1278 | 1.2052 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | lore_consistency | 0.0737 | 24.9549 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.0897 | 3.2805 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 0.0362 | 0.1302 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 0.0006 | 0.0020 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.1855 | nan |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0468 | 1.6323 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1667 | nan |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | 0.0244 | 0.0361 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0424 | -0.0428 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | -0.3200 | -0.4615 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.0000 | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1019 | 0.4446 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.1002 | (0.0153, 0.1725) | 0.0090 | 0.1002 | (0.0354, 0.1434) | 0.0000 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0533 | (0.0000, 0.1600) | 0.3257 | 0.0533 | (0.0000, 0.0889) | 0.2577 |
| proposed_vs_candidate_no_context | naturalness | 0.0509 | (-0.0154, 0.1207) | 0.0823 | 0.0509 | (0.0332, 0.0774) | 0.0000 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.0850 | (0.0099, 0.1510) | 0.0127 | 0.0850 | (0.0374, 0.1167) | 0.0000 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0732 | (0.0073, 0.1395) | 0.0137 | 0.0732 | (0.0377, 0.0969) | 0.0000 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | 0.0383 | (-0.0137, 0.0977) | 0.0893 | 0.0383 | (0.0208, 0.0500) | 0.0000 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 0.0768 | (0.0562, 0.0967) | 0.0000 | 0.0768 | (0.0754, 0.0778) | 0.0000 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | 0.0404 | (-0.0122, 0.1063) | 0.0757 | 0.0404 | (0.0193, 0.0722) | 0.0000 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.1291 | (0.0182, 0.2200) | 0.0123 | 0.1291 | (0.0455, 0.1848) | 0.0000 |
| proposed_vs_candidate_no_context | context_overlap | 0.0328 | (-0.0070, 0.0725) | 0.0480 | 0.0328 | (0.0118, 0.0467) | 0.0000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0667 | (0.0000, 0.2000) | 0.3237 | 0.0667 | (0.0000, 0.1111) | 0.2470 |
| proposed_vs_candidate_no_context | persona_style | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | distinct1 | 0.0064 | (-0.0223, 0.0320) | 0.3190 | 0.0064 | (-0.0127, 0.0352) | 0.2567 |
| proposed_vs_candidate_no_context | length_score | 0.2067 | (-0.1400, 0.5600) | 0.1323 | 0.2067 | (0.1333, 0.3167) | 0.0000 |
| proposed_vs_candidate_no_context | sentence_score | 0.0700 | (0.0000, 0.2100) | 0.3370 | 0.0700 | (0.0000, 0.1167) | 0.2440 |
| proposed_vs_candidate_no_context | overall_quality | 0.0766 | (0.0282, 0.1389) | 0.0000 | 0.0766 | (0.0318, 0.1064) | 0.0000 |
| proposed_vs_baseline_no_context | context_relevance | 0.1134 | (0.0547, 0.1693) | 0.0000 | 0.1134 | (0.0742, 0.1396) | 0.0000 |
| proposed_vs_baseline_no_context | persona_consistency | 0.0267 | (-0.1067, 0.1600) | 0.4393 | 0.0267 | (-0.0667, 0.0889) | 0.2437 |
| proposed_vs_baseline_no_context | naturalness | 0.0029 | (-0.0630, 0.0736) | 0.4550 | 0.0029 | (-0.0449, 0.0746) | 0.2557 |
| proposed_vs_baseline_no_context | quest_state_correctness | 0.0938 | (0.0464, 0.1413) | 0.0000 | 0.0938 | (0.0614, 0.1154) | 0.0000 |
| proposed_vs_baseline_no_context | lore_consistency | 0.0737 | (0.0131, 0.1356) | 0.0003 | 0.0737 | (0.0288, 0.1037) | 0.0000 |
| proposed_vs_baseline_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_baseline_no_context | objective_completion_support | 0.0140 | (-0.0487, 0.0837) | 0.3553 | 0.0140 | (-0.0392, 0.0494) | 0.2557 |
| proposed_vs_baseline_no_context | gameplay_usefulness | 0.0427 | (0.0136, 0.0718) | 0.0003 | 0.0427 | (0.0375, 0.0505) | 0.0000 |
| proposed_vs_baseline_no_context | time_pressure_acceptability | 0.0270 | (-0.0120, 0.0660) | 0.0840 | 0.0270 | (0.0243, 0.0310) | 0.0000 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.1473 | (0.0727, 0.2200) | 0.0000 | 0.1473 | (0.0909, 0.1848) | 0.0000 |
| proposed_vs_baseline_no_context | context_overlap | 0.0344 | (0.0094, 0.0555) | 0.0020 | 0.0344 | (0.0339, 0.0351) | 0.0000 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 0.0333 | (-0.1333, 0.2000) | 0.4343 | 0.0333 | (-0.0833, 0.1111) | 0.2697 |
| proposed_vs_baseline_no_context | persona_style | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_baseline_no_context | distinct1 | -0.0186 | (-0.0488, 0.0116) | 0.8947 | -0.0186 | (-0.0217, -0.0165) | 1.0000 |
| proposed_vs_baseline_no_context | length_score | 0.0867 | (-0.2133, 0.3867) | 0.3057 | 0.0867 | (-0.1333, 0.4167) | 0.2670 |
| proposed_vs_baseline_no_context | sentence_score | -0.0700 | (-0.2100, 0.0000) | 1.0000 | -0.0700 | (-0.1167, 0.0000) | 1.0000 |
| proposed_vs_baseline_no_context | overall_quality | 0.0632 | (0.0050, 0.1368) | 0.0087 | 0.0632 | (0.0239, 0.0894) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 0.1393 | (0.0949, 0.1835) | 0.0000 | 0.1393 | (0.1388, 0.1399) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | 0.0849 | (0.0049, 0.1867) | 0.0140 | 0.0849 | (0.0000, 0.1415) | 0.2410 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | -0.0180 | (-0.0737, 0.0563) | 0.7510 | -0.0180 | (-0.0572, 0.0409) | 0.7527 |
| proposed_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.1140 | (0.0749, 0.1527) | 0.0000 | 0.1140 | (0.1102, 0.1197) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | lore_consistency | 0.0757 | (0.0121, 0.1392) | 0.0137 | 0.0757 | (0.0337, 0.1037) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.0357 | (-0.0271, 0.0983) | 0.1600 | 0.0357 | (0.0142, 0.0500) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 0.0464 | (0.0113, 0.0810) | 0.0087 | 0.0464 | (0.0343, 0.0645) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 0.0111 | (-0.0303, 0.0532) | 0.3187 | 0.0111 | (-0.0173, 0.0300) | 0.2407 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.1836 | (0.1327, 0.2364) | 0.0000 | 0.1836 | (0.1818, 0.1848) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 0.0357 | (0.0065, 0.0601) | 0.0063 | 0.0357 | (0.0315, 0.0420) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1000 | (0.0000, 0.2333) | 0.0803 | 0.1000 | (0.0000, 0.1667) | 0.2553 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | 0.0244 | (0.0000, 0.0732) | 0.3220 | 0.0244 | (0.0000, 0.0407) | 0.2450 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | -0.0183 | (-0.0488, 0.0122) | 0.9113 | -0.0183 | (-0.0194, -0.0166) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | -0.0533 | (-0.2733, 0.2467) | 0.6800 | -0.0533 | (-0.1889, 0.1500) | 0.7447 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | 0.0000 | (-0.2100, 0.2100) | 0.6290 | 0.0000 | (-0.1167, 0.1750) | 0.7583 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | 0.0927 | (0.0580, 0.1439) | 0.0000 | 0.0927 | (0.0731, 0.1058) | 0.0000 |
| controlled_vs_proposed_raw | context_relevance | 0.0046 | (-0.1405, 0.1988) | 0.4423 | 0.0046 | (-0.0659, 0.1103) | 0.2423 |
| controlled_vs_proposed_raw | persona_consistency | 0.0533 | (-0.0533, 0.1333) | 0.1863 | 0.0533 | (0.0000, 0.1333) | 0.2297 |
| controlled_vs_proposed_raw | naturalness | -0.0630 | (-0.1458, -0.0027) | 0.9910 | -0.0630 | (-0.1500, -0.0050) | 1.0000 |
| controlled_vs_proposed_raw | quest_state_correctness | 0.0138 | (-0.1172, 0.1848) | 0.4387 | 0.0138 | (-0.0479, 0.1064) | 0.2403 |
| controlled_vs_proposed_raw | lore_consistency | -0.0020 | (-0.1260, 0.1576) | 0.5853 | -0.0020 | (-0.1037, 0.1506) | 0.7473 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0540 | (-0.0303, 0.1383) | 0.1130 | 0.0540 | (0.0494, 0.0608) | 0.0000 |
| controlled_vs_proposed_raw | gameplay_usefulness | -0.0101 | (-0.0235, 0.0045) | 0.9273 | -0.0101 | (-0.0161, -0.0062) | 1.0000 |
| controlled_vs_proposed_raw | time_pressure_acceptability | -0.0105 | (-0.0886, 0.0675) | 0.6047 | -0.0105 | (-0.0295, 0.0022) | 0.7597 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.0018 | (-0.1818, 0.2564) | 0.4523 | 0.0018 | (-0.0879, 0.1364) | 0.2483 |
| controlled_vs_proposed_raw | context_overlap | 0.0111 | (-0.0465, 0.0718) | 0.3853 | 0.0111 | (-0.0145, 0.0495) | 0.2563 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.0667 | (-0.0667, 0.1667) | 0.1920 | 0.0667 | (0.0000, 0.1667) | 0.2437 |
| controlled_vs_proposed_raw | persona_style | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_proposed_raw | distinct1 | -0.0241 | (-0.0584, 0.0102) | 0.9133 | -0.0241 | (-0.0728, 0.0084) | 0.7393 |
| controlled_vs_proposed_raw | length_score | -0.2667 | (-0.5267, -0.0867) | 1.0000 | -0.2667 | (-0.5167, -0.1000) | 1.0000 |
| controlled_vs_proposed_raw | sentence_score | 0.0000 | (-0.2100, 0.2100) | 0.6397 | 0.0000 | (-0.1750, 0.1167) | 0.7457 |
| controlled_vs_proposed_raw | overall_quality | 0.0092 | (-0.0665, 0.0933) | 0.4323 | 0.0092 | (-0.0317, 0.0706) | 0.2523 |
| controlled_vs_candidate_no_context | context_relevance | 0.1048 | (-0.0856, 0.3203) | 0.1530 | 0.1048 | (0.0776, 0.1457) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1067 | (0.0533, 0.1333) | 0.0007 | 0.1067 | (0.0889, 0.1333) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | -0.0121 | (-0.1333, 0.0792) | 0.5450 | -0.0121 | (-0.0726, 0.0283) | 0.7383 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.0988 | (-0.0589, 0.2815) | 0.1393 | 0.0988 | (0.0688, 0.1438) | 0.0000 |
| controlled_vs_candidate_no_context | lore_consistency | 0.0713 | (-0.0122, 0.2287) | 0.3320 | 0.0713 | (-0.0068, 0.1884) | 0.2533 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | 0.0923 | (0.0567, 0.1307) | 0.0000 | 0.0923 | (0.0817, 0.0994) | 0.0000 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0667 | (0.0392, 0.0942) | 0.0000 | 0.0667 | (0.0593, 0.0716) | 0.0000 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 0.0299 | (-0.0702, 0.1518) | 0.2710 | 0.0299 | (0.0214, 0.0427) | 0.0000 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.1309 | (-0.1091, 0.4218) | 0.1560 | 0.1309 | (0.0970, 0.1818) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0439 | (0.0025, 0.0988) | 0.0247 | 0.0439 | (0.0322, 0.0613) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1333 | (0.0667, 0.1667) | 0.0000 | 0.1333 | (0.1111, 0.1667) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | distinct1 | -0.0177 | (-0.0336, -0.0017) | 0.9893 | -0.0177 | (-0.0377, -0.0044) | 1.0000 |
| controlled_vs_candidate_no_context | length_score | -0.0600 | (-0.5667, 0.3533) | 0.6063 | -0.0600 | (-0.2000, 0.0333) | 0.7573 |
| controlled_vs_candidate_no_context | sentence_score | 0.0700 | (-0.1400, 0.2800) | 0.3800 | 0.0700 | (-0.1750, 0.2333) | 0.2547 |
| controlled_vs_candidate_no_context | overall_quality | 0.0858 | (0.0161, 0.1661) | 0.0040 | 0.0858 | (0.0747, 0.1025) | 0.0000 |
| controlled_vs_baseline_no_context | context_relevance | 0.1180 | (0.0177, 0.2484) | 0.0000 | 0.1180 | (0.0737, 0.1845) | 0.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 0.0800 | (0.0267, 0.1333) | 0.0087 | 0.0800 | (0.0667, 0.0889) | 0.0000 |
| controlled_vs_baseline_no_context | naturalness | -0.0601 | (-0.0979, -0.0101) | 0.9940 | -0.0601 | (-0.0753, -0.0499) | 1.0000 |
| controlled_vs_baseline_no_context | quest_state_correctness | 0.1075 | (0.0185, 0.2309) | 0.0000 | 0.1075 | (0.0674, 0.1677) | 0.0000 |
| controlled_vs_baseline_no_context | lore_consistency | 0.0718 | (0.0000, 0.2153) | 0.3220 | 0.0718 | (0.0000, 0.1794) | 0.2470 |
| controlled_vs_baseline_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_baseline_no_context | objective_completion_support | 0.0680 | (0.0120, 0.1080) | 0.0060 | 0.0680 | (0.0217, 0.0989) | 0.0000 |
| controlled_vs_baseline_no_context | gameplay_usefulness | 0.0326 | (0.0088, 0.0590) | 0.0000 | 0.0326 | (0.0314, 0.0344) | 0.0000 |
| controlled_vs_baseline_no_context | time_pressure_acceptability | 0.0165 | (-0.0238, 0.0556) | 0.2400 | 0.0165 | (0.0014, 0.0265) | 0.0000 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.1491 | (0.0182, 0.3018) | 0.0103 | 0.1491 | (0.0970, 0.2273) | 0.0000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0455 | (0.0081, 0.0899) | 0.0043 | 0.0455 | (0.0194, 0.0846) | 0.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1000 | (0.0333, 0.1667) | 0.0103 | 0.1000 | (0.0833, 0.1111) | 0.0000 |
| controlled_vs_baseline_no_context | persona_style | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_baseline_no_context | distinct1 | -0.0427 | (-0.0880, 0.0027) | 0.9723 | -0.0427 | (-0.0946, -0.0081) | 1.0000 |
| controlled_vs_baseline_no_context | length_score | -0.1800 | (-0.3937, 0.0467) | 0.9280 | -0.1800 | (-0.2333, -0.1000) | 1.0000 |
| controlled_vs_baseline_no_context | sentence_score | -0.0700 | (-0.2100, 0.0000) | 1.0000 | -0.0700 | (-0.1750, 0.0000) | 1.0000 |
| controlled_vs_baseline_no_context | overall_quality | 0.0724 | (0.0210, 0.1170) | 0.0010 | 0.0724 | (0.0576, 0.0945) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.1439 | (0.0169, 0.3246) | 0.0000 | 0.1439 | (0.0730, 0.2502) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1382 | (0.1333, 0.1480) | 0.0000 | 0.1382 | (0.1333, 0.1415) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | -0.0810 | (-0.1764, -0.0050) | 0.9840 | -0.0810 | (-0.1091, -0.0622) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.1278 | (0.0109, 0.2977) | 0.0003 | 0.1278 | (0.0623, 0.2261) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | lore_consistency | 0.0737 | (-0.0089, 0.2300) | 0.3160 | 0.0737 | (0.0000, 0.1843) | 0.2400 |
| controlled_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.0897 | (0.0353, 0.1413) | 0.0000 | 0.0897 | (0.0750, 0.0994) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 0.0362 | (0.0035, 0.0705) | 0.0097 | 0.0362 | (0.0281, 0.0485) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 0.0006 | (-0.0879, 0.0631) | 0.4353 | 0.0006 | (-0.0469, 0.0322) | 0.2543 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.1855 | (0.0182, 0.4218) | 0.0123 | 0.1855 | (0.0970, 0.3182) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0468 | (0.0080, 0.0978) | 0.0047 | 0.0468 | (0.0170, 0.0915) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1667 | (0.1667, 0.1667) | 0.0000 | 0.1667 | (0.1667, 0.1667) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | 0.0244 | (0.0000, 0.0732) | 0.3353 | 0.0244 | (0.0000, 0.0407) | 0.2393 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0424 | (-0.0873, -0.0044) | 0.9840 | -0.0424 | (-0.0894, -0.0111) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | -0.3200 | (-0.7000, 0.0200) | 0.9650 | -0.3200 | (-0.3667, -0.2889) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.0000 | (-0.2100, 0.2100) | 0.6350 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1019 | (0.0542, 0.1655) | 0.0000 | 0.1019 | (0.0740, 0.1437) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.1180 | (0.0177, 0.2466) | 0.0000 | 0.1180 | (0.0737, 0.1845) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.0800 | (0.0267, 0.1333) | 0.0093 | 0.0800 | (0.0667, 0.0889) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0601 | (-0.0994, -0.0101) | 0.9940 | -0.0601 | (-0.0753, -0.0499) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | quest_state_correctness | 0.1075 | (0.0162, 0.2286) | 0.0000 | 0.1075 | (0.0674, 0.1677) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | lore_consistency | 0.0718 | (0.0000, 0.2153) | 0.3233 | 0.0718 | (0.0000, 0.1794) | 0.2323 |
| proposed_contextual_controlled_vs_baseline_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | objective_completion_support | 0.0680 | (0.0120, 0.1057) | 0.0063 | 0.0680 | (0.0217, 0.0989) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | gameplay_usefulness | 0.0326 | (0.0088, 0.0611) | 0.0000 | 0.0326 | (0.0314, 0.0344) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | time_pressure_acceptability | 0.0165 | (-0.0238, 0.0556) | 0.2530 | 0.0165 | (0.0014, 0.0265) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.1491 | (0.0182, 0.3091) | 0.0100 | 0.1491 | (0.0970, 0.2273) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0455 | (0.0081, 0.0923) | 0.0067 | 0.0455 | (0.0194, 0.0846) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1000 | (0.0333, 0.1667) | 0.0107 | 0.1000 | (0.0833, 0.1111) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0427 | (-0.0893, 0.0027) | 0.9663 | -0.0427 | (-0.0946, -0.0081) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | -0.1800 | (-0.3933, 0.0467) | 0.9337 | -0.1800 | (-0.2333, -0.1000) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | -0.0700 | (-0.2100, 0.0000) | 1.0000 | -0.0700 | (-0.1750, 0.0000) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.0724 | (0.0210, 0.1170) | 0.0007 | 0.0724 | (0.0576, 0.0945) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.1439 | (0.0169, 0.3246) | 0.0003 | 0.1439 | (0.0730, 0.2502) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1382 | (0.1333, 0.1480) | 0.0000 | 0.1382 | (0.1333, 0.1415) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | -0.0810 | (-0.1764, -0.0050) | 0.9830 | -0.0810 | (-0.1091, -0.0622) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.1278 | (0.0101, 0.2977) | 0.0000 | 0.1278 | (0.0623, 0.2261) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | lore_consistency | 0.0737 | (-0.0089, 0.2300) | 0.3417 | 0.0737 | (0.0000, 0.1843) | 0.2540 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.0897 | (0.0353, 0.1407) | 0.0000 | 0.0897 | (0.0750, 0.0994) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 0.0362 | (0.0047, 0.0705) | 0.0070 | 0.0362 | (0.0281, 0.0485) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 0.0006 | (-0.0904, 0.0631) | 0.4357 | 0.0006 | (-0.0469, 0.0322) | 0.2467 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.1855 | (0.0182, 0.4218) | 0.0073 | 0.1855 | (0.0970, 0.3182) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0468 | (0.0071, 0.0987) | 0.0063 | 0.0468 | (0.0170, 0.0915) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1667 | (0.1667, 0.1667) | 0.0000 | 0.1667 | (0.1667, 0.1667) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | 0.0244 | (0.0000, 0.0732) | 0.3463 | 0.0244 | (0.0000, 0.0407) | 0.2463 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0424 | (-0.0873, -0.0014) | 0.9847 | -0.0424 | (-0.0894, -0.0111) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | -0.3200 | (-0.7000, 0.0200) | 0.9633 | -0.3200 | (-0.3667, -0.2889) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.0000 | (-0.2100, 0.2100) | 0.6353 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1019 | (0.0542, 0.1660) | 0.0000 | 0.1019 | (0.0740, 0.1437) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 4 | 1 | 0 | 0.8000 | 0.8000 |
| proposed_vs_candidate_no_context | persona_consistency | 1 | 0 | 4 | 0.6000 | 1.0000 |
| proposed_vs_candidate_no_context | naturalness | 3 | 2 | 0 | 0.6000 | 0.6000 |
| proposed_vs_candidate_no_context | quest_state_correctness | 4 | 1 | 0 | 0.8000 | 0.8000 |
| proposed_vs_candidate_no_context | lore_consistency | 4 | 1 | 0 | 0.8000 | 0.8000 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 5 | 0.5000 | nan |
| proposed_vs_candidate_no_context | objective_completion_support | 3 | 1 | 1 | 0.7000 | 0.7500 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 5 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | 2 | 2 | 1 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 4 | 1 | 0 | 0.8000 | 0.8000 |
| proposed_vs_candidate_no_context | context_overlap | 3 | 2 | 0 | 0.6000 | 0.6000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 1 | 0 | 4 | 0.6000 | 1.0000 |
| proposed_vs_candidate_no_context | persona_style | 0 | 0 | 5 | 0.5000 | nan |
| proposed_vs_candidate_no_context | distinct1 | 3 | 1 | 1 | 0.7000 | 0.7500 |
| proposed_vs_candidate_no_context | length_score | 3 | 2 | 0 | 0.6000 | 0.6000 |
| proposed_vs_candidate_no_context | sentence_score | 1 | 0 | 4 | 0.6000 | 1.0000 |
| proposed_vs_candidate_no_context | overall_quality | 5 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_vs_baseline_no_context | context_relevance | 5 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_vs_baseline_no_context | persona_consistency | 2 | 2 | 1 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context | naturalness | 3 | 2 | 0 | 0.6000 | 0.6000 |
| proposed_vs_baseline_no_context | quest_state_correctness | 5 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_vs_baseline_no_context | lore_consistency | 4 | 0 | 1 | 0.9000 | 1.0000 |
| proposed_vs_baseline_no_context | multi_turn_contradiction_safety | 0 | 0 | 5 | 0.5000 | nan |
| proposed_vs_baseline_no_context | objective_completion_support | 3 | 2 | 0 | 0.6000 | 0.6000 |
| proposed_vs_baseline_no_context | gameplay_usefulness | 4 | 1 | 0 | 0.8000 | 0.8000 |
| proposed_vs_baseline_no_context | time_pressure_acceptability | 3 | 2 | 0 | 0.6000 | 0.6000 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 4 | 0 | 1 | 0.9000 | 1.0000 |
| proposed_vs_baseline_no_context | context_overlap | 4 | 1 | 0 | 0.8000 | 0.8000 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 2 | 2 | 1 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context | persona_style | 0 | 0 | 5 | 0.5000 | nan |
| proposed_vs_baseline_no_context | distinct1 | 2 | 3 | 0 | 0.4000 | 0.4000 |
| proposed_vs_baseline_no_context | length_score | 3 | 2 | 0 | 0.6000 | 0.6000 |
| proposed_vs_baseline_no_context | sentence_score | 0 | 1 | 4 | 0.4000 | 0.0000 |
| proposed_vs_baseline_no_context | overall_quality | 4 | 1 | 0 | 0.8000 | 0.8000 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 5 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | 3 | 0 | 2 | 0.8000 | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | 1 | 4 | 0 | 0.2000 | 0.2000 |
| proposed_vs_baseline_no_context_phi3_latest | quest_state_correctness | 5 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | lore_consistency | 3 | 1 | 1 | 0.7000 | 0.7500 |
| proposed_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0 | 0 | 5 | 0.5000 | nan |
| proposed_vs_baseline_no_context_phi3_latest | objective_completion_support | 3 | 1 | 1 | 0.7000 | 0.7500 |
| proposed_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 4 | 1 | 0 | 0.8000 | 0.8000 |
| proposed_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 2 | 2 | 1 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 5 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 4 | 1 | 0 | 0.8000 | 0.8000 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 2 | 0 | 3 | 0.7000 | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | 1 | 0 | 4 | 0.6000 | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | 2 | 3 | 0 | 0.4000 | 0.4000 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | 1 | 4 | 0 | 0.2000 | 0.2000 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | 1 | 1 | 3 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | 5 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_proposed_raw | context_relevance | 2 | 3 | 0 | 0.4000 | 0.4000 |
| controlled_vs_proposed_raw | persona_consistency | 3 | 1 | 1 | 0.7000 | 0.7500 |
| controlled_vs_proposed_raw | naturalness | 1 | 4 | 0 | 0.2000 | 0.2000 |
| controlled_vs_proposed_raw | quest_state_correctness | 2 | 3 | 0 | 0.4000 | 0.4000 |
| controlled_vs_proposed_raw | lore_consistency | 1 | 3 | 1 | 0.3000 | 0.2500 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0 | 0 | 5 | 0.5000 | nan |
| controlled_vs_proposed_raw | objective_completion_support | 3 | 2 | 0 | 0.6000 | 0.6000 |
| controlled_vs_proposed_raw | gameplay_usefulness | 1 | 4 | 0 | 0.2000 | 0.2000 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 3 | 2 | 0 | 0.6000 | 0.6000 |
| controlled_vs_proposed_raw | context_keyword_coverage | 2 | 3 | 0 | 0.4000 | 0.4000 |
| controlled_vs_proposed_raw | context_overlap | 2 | 3 | 0 | 0.4000 | 0.4000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 3 | 1 | 1 | 0.7000 | 0.7500 |
| controlled_vs_proposed_raw | persona_style | 0 | 0 | 5 | 0.5000 | nan |
| controlled_vs_proposed_raw | distinct1 | 1 | 3 | 1 | 0.3000 | 0.2500 |
| controlled_vs_proposed_raw | length_score | 0 | 4 | 1 | 0.1000 | 0.0000 |
| controlled_vs_proposed_raw | sentence_score | 1 | 1 | 3 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | overall_quality | 2 | 3 | 0 | 0.4000 | 0.4000 |
| controlled_vs_candidate_no_context | context_relevance | 4 | 1 | 0 | 0.8000 | 0.8000 |
| controlled_vs_candidate_no_context | persona_consistency | 4 | 0 | 1 | 0.9000 | 1.0000 |
| controlled_vs_candidate_no_context | naturalness | 3 | 2 | 0 | 0.6000 | 0.6000 |
| controlled_vs_candidate_no_context | quest_state_correctness | 4 | 1 | 0 | 0.8000 | 0.8000 |
| controlled_vs_candidate_no_context | lore_consistency | 1 | 2 | 2 | 0.4000 | 0.3333 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 5 | 0.5000 | nan |
| controlled_vs_candidate_no_context | objective_completion_support | 5 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 5 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 2 | 3 | 0 | 0.4000 | 0.4000 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 3 | 1 | 1 | 0.7000 | 0.7500 |
| controlled_vs_candidate_no_context | context_overlap | 4 | 1 | 0 | 0.8000 | 0.8000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 4 | 0 | 1 | 0.9000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_style | 0 | 0 | 5 | 0.5000 | nan |
| controlled_vs_candidate_no_context | distinct1 | 1 | 3 | 1 | 0.3000 | 0.2500 |
| controlled_vs_candidate_no_context | length_score | 3 | 2 | 0 | 0.6000 | 0.6000 |
| controlled_vs_candidate_no_context | sentence_score | 2 | 1 | 2 | 0.6000 | 0.6667 |
| controlled_vs_candidate_no_context | overall_quality | 4 | 1 | 0 | 0.8000 | 0.8000 |
| controlled_vs_baseline_no_context | context_relevance | 5 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 3 | 0 | 2 | 0.8000 | 1.0000 |
| controlled_vs_baseline_no_context | naturalness | 1 | 4 | 0 | 0.2000 | 0.2000 |
| controlled_vs_baseline_no_context | quest_state_correctness | 5 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | lore_consistency | 1 | 0 | 4 | 0.6000 | 1.0000 |
| controlled_vs_baseline_no_context | multi_turn_contradiction_safety | 0 | 0 | 5 | 0.5000 | nan |
| controlled_vs_baseline_no_context | objective_completion_support | 4 | 1 | 0 | 0.8000 | 0.8000 |
| controlled_vs_baseline_no_context | gameplay_usefulness | 5 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | time_pressure_acceptability | 3 | 2 | 0 | 0.6000 | 0.6000 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 3 | 0 | 2 | 0.8000 | 1.0000 |
| controlled_vs_baseline_no_context | context_overlap | 4 | 1 | 0 | 0.8000 | 0.8000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 3 | 0 | 2 | 0.8000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_style | 0 | 0 | 5 | 0.5000 | nan |
| controlled_vs_baseline_no_context | distinct1 | 1 | 4 | 0 | 0.2000 | 0.2000 |
| controlled_vs_baseline_no_context | length_score | 1 | 4 | 0 | 0.2000 | 0.2000 |
| controlled_vs_baseline_no_context | sentence_score | 0 | 1 | 4 | 0.4000 | 0.0000 |
| controlled_vs_baseline_no_context | overall_quality | 4 | 1 | 0 | 0.8000 | 0.8000 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 4 | 1 | 0 | 0.8000 | 0.8000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 5 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | 1 | 4 | 0 | 0.2000 | 0.2000 |
| controlled_vs_baseline_no_context_phi3_latest | quest_state_correctness | 4 | 1 | 0 | 0.8000 | 0.8000 |
| controlled_vs_baseline_no_context_phi3_latest | lore_consistency | 1 | 1 | 3 | 0.5000 | 0.5000 |
| controlled_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0 | 0 | 5 | 0.5000 | nan |
| controlled_vs_baseline_no_context_phi3_latest | objective_completion_support | 4 | 1 | 0 | 0.8000 | 0.8000 |
| controlled_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 4 | 1 | 0 | 0.8000 | 0.8000 |
| controlled_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 3 | 2 | 0 | 0.6000 | 0.6000 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 3 | 0 | 2 | 0.8000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 4 | 1 | 0 | 0.8000 | 0.8000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 5 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | 1 | 0 | 4 | 0.6000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | 1 | 4 | 0 | 0.2000 | 0.2000 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | 1 | 4 | 0 | 0.2000 | 0.2000 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 1 | 1 | 3 | 0.5000 | 0.5000 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 5 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 5 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 3 | 0 | 2 | 0.8000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 1 | 4 | 0 | 0.2000 | 0.2000 |
| proposed_contextual_controlled_vs_baseline_no_context | quest_state_correctness | 5 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | lore_consistency | 1 | 0 | 4 | 0.6000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | multi_turn_contradiction_safety | 0 | 0 | 5 | 0.5000 | nan |
| proposed_contextual_controlled_vs_baseline_no_context | objective_completion_support | 4 | 1 | 0 | 0.8000 | 0.8000 |
| proposed_contextual_controlled_vs_baseline_no_context | gameplay_usefulness | 5 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | time_pressure_acceptability | 3 | 2 | 0 | 0.6000 | 0.6000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 3 | 0 | 2 | 0.8000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 4 | 1 | 0 | 0.8000 | 0.8000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 3 | 0 | 2 | 0.8000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0 | 0 | 5 | 0.5000 | nan |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | 1 | 4 | 0 | 0.2000 | 0.2000 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 1 | 4 | 0 | 0.2000 | 0.2000 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0 | 1 | 4 | 0.4000 | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 4 | 1 | 0 | 0.8000 | 0.8000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 4 | 1 | 0 | 0.8000 | 0.8000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 5 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | 1 | 4 | 0 | 0.2000 | 0.2000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | quest_state_correctness | 4 | 1 | 0 | 0.8000 | 0.8000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | lore_consistency | 1 | 1 | 3 | 0.5000 | 0.5000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0 | 0 | 5 | 0.5000 | nan |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | objective_completion_support | 4 | 1 | 0 | 0.8000 | 0.8000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 4 | 1 | 0 | 0.8000 | 0.8000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 3 | 2 | 0 | 0.6000 | 0.6000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 3 | 0 | 2 | 0.8000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 4 | 1 | 0 | 0.8000 | 0.8000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 5 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | 1 | 0 | 4 | 0.6000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | 1 | 4 | 0 | 0.2000 | 0.2000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | 1 | 4 | 0 | 0.2000 | 0.2000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | 1 | 1 | 3 | 0.5000 | 0.5000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 5 | 0 | 0 | 1.0000 | 1.0000 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.4000 | 0.2000 | 0.8000 |
| proposed_contextual | 0.0000 | 0.0000 | 0.2000 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
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
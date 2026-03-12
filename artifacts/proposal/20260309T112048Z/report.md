# Proposal Alignment Evaluation Report

- Run ID: `20260309T112048Z`
- Generated: `2026-03-09T11:22:07.021797+00:00`
- Scenarios: `artifacts\proposal\20260309T112048Z\scenarios.jsonl`
- Scenario count: `4`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off
- `baseline_no_context`: model `phi3:mini`, dynamic_context=off, response_control=off
- `baseline_no_context_phi3_latest`: model `phi3:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2109 (0.1842, 0.2322) | 0.3738 (0.2821, 0.4643) | 0.9116 (0.8959, 0.9282) | 0.4081 (0.3644, 0.4471) | n/a |
| proposed_contextual | 0.0485 (0.0109, 0.0860) | 0.1833 (0.1000, 0.2750) | 0.8493 (0.7486, 0.9500) | 0.2486 (0.1987, 0.3026) | n/a |
| candidate_no_context | 0.0065 (0.0017, 0.0123) | 0.1500 (0.1000, 0.2000) | 0.8222 (0.7486, 0.9322) | 0.2118 (0.1874, 0.2425) | n/a |
| baseline_no_context | 0.0196 (0.0014, 0.0521) | 0.1786 (0.1000, 0.2607) | 0.9181 (0.8733, 0.9628) | 0.2482 (0.2057, 0.2907) | n/a |
| baseline_no_context_phi3_latest | 0.0194 (0.0000, 0.0542) | 0.1500 (0.1000, 0.2000) | 0.9138 (0.8616, 0.9636) | 0.2366 (0.2056, 0.2707) | n/a |

## Game-facing Outcome Metrics (mean, 95% CI)
| Arm | Quest-state Correctness | Lore Consistency | Contradiction Safety | Objective Completion Support | Gameplay Usefulness | Time-pressure Acceptability |
|---|---:|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | nan (nan, nan) | nan (nan, nan) | nan (nan, nan) | nan (nan, nan) | nan (nan, nan) | nan (nan, nan) |
| proposed_contextual | nan (nan, nan) | nan (nan, nan) | nan (nan, nan) | nan (nan, nan) | nan (nan, nan) | nan (nan, nan) |
| candidate_no_context | nan (nan, nan) | nan (nan, nan) | nan (nan, nan) | nan (nan, nan) | nan (nan, nan) | nan (nan, nan) |
| baseline_no_context | nan (nan, nan) | nan (nan, nan) | nan (nan, nan) | nan (nan, nan) | nan (nan, nan) | nan (nan, nan) |
| baseline_no_context_phi3_latest | nan (nan, nan) | nan (nan, nan) | nan (nan, nan) | nan (nan, nan) | nan (nan, nan) | nan (nan, nan) |

- Multi-turn contradiction rate is reported as `1 - contradiction_safety` in row-level outputs.

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0420 | 6.4616 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0333 | 0.2222 |
| proposed_vs_candidate_no_context | naturalness | 0.0271 | 0.0329 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0477 | nan |
| proposed_vs_candidate_no_context | context_overlap | 0.0286 | 1.3187 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0417 | nan |
| proposed_vs_candidate_no_context | persona_style | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | distinct1 | -0.0074 | -0.0077 |
| proposed_vs_candidate_no_context | length_score | 0.1500 | 0.4500 |
| proposed_vs_candidate_no_context | sentence_score | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | overall_quality | 0.0368 | 0.1740 |
| proposed_vs_baseline_no_context | context_relevance | 0.0289 | 1.4791 |
| proposed_vs_baseline_no_context | persona_consistency | 0.0048 | 0.0267 |
| proposed_vs_baseline_no_context | naturalness | -0.0688 | -0.0749 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0250 | 1.1000 |
| proposed_vs_baseline_no_context | context_overlap | 0.0381 | 3.1344 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 0.0060 | 0.1667 |
| proposed_vs_baseline_no_context | persona_style | 0.0000 | 0.0000 |
| proposed_vs_baseline_no_context | distinct1 | -0.0396 | -0.0402 |
| proposed_vs_baseline_no_context | length_score | -0.1333 | -0.2162 |
| proposed_vs_baseline_no_context | sentence_score | -0.2625 | -0.2625 |
| proposed_vs_baseline_no_context | overall_quality | 0.0004 | 0.0018 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 0.0291 | 1.4965 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | 0.0333 | 0.2222 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | -0.0646 | -0.0706 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.0250 | 1.1000 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 0.0385 | 3.2957 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.0417 | nan |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | 0.0000 | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | -0.0374 | -0.0380 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | -0.1167 | -0.1944 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | -0.2625 | -0.2625 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | 0.0120 | 0.0509 |
| controlled_vs_proposed_raw | context_relevance | 0.1624 | 3.3514 |
| controlled_vs_proposed_raw | persona_consistency | 0.1905 | 1.0390 |
| controlled_vs_proposed_raw | naturalness | 0.0623 | 0.0734 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2136 | 4.4762 |
| controlled_vs_proposed_raw | context_overlap | 0.0430 | 0.8565 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2381 | 5.7143 |
| controlled_vs_proposed_raw | persona_style | 0.0000 | 0.0000 |
| controlled_vs_proposed_raw | distinct1 | -0.0171 | -0.0181 |
| controlled_vs_proposed_raw | length_score | 0.2583 | 0.5345 |
| controlled_vs_proposed_raw | sentence_score | 0.1750 | 0.2373 |
| controlled_vs_proposed_raw | overall_quality | 0.1595 | 0.6414 |
| controlled_vs_candidate_no_context | context_relevance | 0.2044 | 31.4679 |
| controlled_vs_candidate_no_context | persona_consistency | 0.2238 | 1.4921 |
| controlled_vs_candidate_no_context | naturalness | 0.0894 | 0.1087 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.2614 | nan |
| controlled_vs_candidate_no_context | context_overlap | 0.0716 | 3.3046 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2798 | nan |
| controlled_vs_candidate_no_context | persona_style | 0.0000 | 0.0000 |
| controlled_vs_candidate_no_context | distinct1 | -0.0245 | -0.0256 |
| controlled_vs_candidate_no_context | length_score | 0.4083 | 1.2250 |
| controlled_vs_candidate_no_context | sentence_score | 0.1750 | 0.2373 |
| controlled_vs_candidate_no_context | overall_quality | 0.1963 | 0.9269 |
| controlled_vs_baseline_no_context | context_relevance | 0.1914 | 9.7873 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1952 | 1.0933 |
| controlled_vs_baseline_no_context | naturalness | -0.0065 | -0.0070 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.2386 | 10.5000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0811 | 6.6754 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2440 | 6.8333 |
| controlled_vs_baseline_no_context | persona_style | 0.0000 | 0.0000 |
| controlled_vs_baseline_no_context | distinct1 | -0.0568 | -0.0575 |
| controlled_vs_baseline_no_context | length_score | 0.1250 | 0.2027 |
| controlled_vs_baseline_no_context | sentence_score | -0.0875 | -0.0875 |
| controlled_vs_baseline_no_context | overall_quality | 0.1599 | 0.6443 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.1915 | 9.8634 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.2238 | 1.4921 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | -0.0022 | -0.0024 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.2386 | 10.5000 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0815 | 6.9749 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.2798 | nan |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | 0.0000 | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0545 | -0.0554 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | 0.1417 | 0.2361 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | -0.0875 | -0.0875 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1715 | 0.7250 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.1914 | 9.7873 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1952 | 1.0933 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0065 | -0.0070 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.2386 | 10.5000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0811 | 6.6754 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2440 | 6.8333 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0000 | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0568 | -0.0575 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.1250 | 0.2027 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | -0.0875 | -0.0875 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1599 | 0.6443 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.1915 | 9.8634 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.2238 | 1.4921 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | -0.0022 | -0.0024 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.2386 | 10.5000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0815 | 6.9749 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.2798 | nan |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | 0.0000 | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0545 | -0.0554 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | 0.1417 | 0.2361 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | -0.0875 | -0.0875 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1715 | 0.7250 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0420 | (0.0000, 0.0840) | 0.0693 | 0.0420 | (0.0000, 0.0729) | 0.0337 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0333 | (0.0000, 0.1000) | 0.3263 | 0.0333 | (0.0000, 0.1333) | 0.3027 |
| proposed_vs_candidate_no_context | naturalness | 0.0271 | (-0.0413, 0.1224) | 0.3210 | 0.0271 | (-0.0550, 0.0816) | 0.2947 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.0329 | (0.0000, 0.0659) | 0.0660 | 0.0329 | (0.0000, 0.0609) | 0.0340 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0112 | (0.0000, 0.0266) | 0.0667 | 0.0112 | (0.0000, 0.0177) | 0.0353 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | -0.0104 | (-0.0312, 0.0000) | 1.0000 | -0.0104 | (-0.0417, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 0.0278 | (-0.0004, 0.0838) | 0.3237 | 0.0278 | (-0.0006, 0.0559) | 0.2913 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | 0.0060 | (-0.0141, 0.0321) | 0.3437 | 0.0060 | (-0.0187, 0.0214) | 0.2967 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0477 | (0.0000, 0.0955) | 0.0653 | 0.0477 | (0.0000, 0.0909) | 0.0397 |
| proposed_vs_candidate_no_context | context_overlap | 0.0286 | (0.0000, 0.0625) | 0.0533 | 0.0286 | (0.0000, 0.0417) | 0.0350 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0417 | (0.0000, 0.1250) | 0.3327 | 0.0417 | (0.0000, 0.1667) | 0.3023 |
| proposed_vs_candidate_no_context | persona_style | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | distinct1 | -0.0074 | (-0.0221, 0.0000) | 1.0000 | -0.0074 | (-0.0147, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | length_score | 0.1500 | (-0.0750, 0.5250) | 0.3220 | 0.1500 | (-0.1000, 0.3500) | 0.3037 |
| proposed_vs_candidate_no_context | sentence_score | 0.0000 | (-0.2625, 0.2625) | 0.6480 | 0.0000 | (-0.3500, 0.1750) | 0.6350 |
| proposed_vs_candidate_no_context | overall_quality | 0.0368 | (0.0000, 0.0737) | 0.0587 | 0.0368 | (0.0000, 0.0725) | 0.0367 |
| proposed_vs_baseline_no_context | context_relevance | 0.0289 | (0.0039, 0.0736) | 0.0000 | 0.0289 | (0.0021, 0.0522) | 0.0000 |
| proposed_vs_baseline_no_context | persona_consistency | 0.0048 | (-0.0857, 0.1000) | 0.4200 | 0.0048 | (-0.1143, 0.1333) | 0.4007 |
| proposed_vs_baseline_no_context | naturalness | -0.0688 | (-0.1714, 0.0450) | 0.8823 | -0.0688 | (-0.1980, -0.0182) | 1.0000 |
| proposed_vs_baseline_no_context | quest_state_correctness | 0.0269 | (0.0051, 0.0577) | 0.0000 | 0.0269 | (0.0008, 0.0444) | 0.0000 |
| proposed_vs_baseline_no_context | lore_consistency | 0.0158 | (0.0001, 0.0296) | 0.0177 | 0.0158 | (-0.0078, 0.0296) | 0.1537 |
| proposed_vs_baseline_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_baseline_no_context | objective_completion_support | -0.0250 | (-0.0500, 0.0000) | 1.0000 | -0.0250 | (-0.0450, 0.0000) | 1.0000 |
| proposed_vs_baseline_no_context | gameplay_usefulness | -0.0212 | (-0.0983, 0.0507) | 0.7013 | -0.0212 | (-0.1054, 0.0082) | 0.6997 |
| proposed_vs_baseline_no_context | time_pressure_acceptability | -0.0234 | (-0.0507, 0.0000) | 1.0000 | -0.0234 | (-0.0338, 0.0000) | 1.0000 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0250 | (0.0000, 0.0750) | 0.3210 | 0.0250 | (0.0000, 0.0500) | 0.2990 |
| proposed_vs_baseline_no_context | context_overlap | 0.0381 | (0.0131, 0.0702) | 0.0000 | 0.0381 | (0.0071, 0.0572) | 0.0000 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 0.0060 | (-0.1071, 0.1250) | 0.4303 | 0.0060 | (-0.1429, 0.1667) | 0.4213 |
| proposed_vs_baseline_no_context | persona_style | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_baseline_no_context | distinct1 | -0.0396 | (-0.0909, 0.0167) | 0.9197 | -0.0396 | (-0.0909, 0.0526) | 0.8447 |
| proposed_vs_baseline_no_context | length_score | -0.1333 | (-0.5083, 0.3583) | 0.7450 | -0.1333 | (-0.6333, 0.1167) | 0.8490 |
| proposed_vs_baseline_no_context | sentence_score | -0.2625 | (-0.3500, -0.0875) | 1.0000 | -0.2625 | (-0.3500, -0.1750) | 1.0000 |
| proposed_vs_baseline_no_context | overall_quality | 0.0004 | (-0.0536, 0.0545) | 0.4517 | 0.0004 | (-0.0817, 0.0452) | 0.3630 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 0.0291 | (0.0058, 0.0730) | 0.0000 | 0.0291 | (0.0049, 0.0523) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | 0.0333 | (0.0000, 0.1000) | 0.3260 | 0.0333 | (0.0000, 0.1333) | 0.2820 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | -0.0646 | (-0.1678, 0.0517) | 0.8317 | -0.0646 | (-0.1776, -0.0182) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.0196 | (-0.0052, 0.0538) | 0.1213 | 0.0196 | (-0.0129, 0.0445) | 0.2523 |
| proposed_vs_baseline_no_context_phi3_latest | lore_consistency | 0.0169 | (0.0032, 0.0296) | 0.0037 | 0.0169 | (-0.0037, 0.0296) | 0.0353 |
| proposed_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | objective_completion_support | -0.0008 | (-0.0312, 0.0304) | 0.6297 | -0.0008 | (-0.0417, 0.0208) | 0.6340 |
| proposed_vs_baseline_no_context_phi3_latest | gameplay_usefulness | -0.0101 | (-0.0644, 0.0647) | 0.6700 | -0.0101 | (-0.0800, 0.0285) | 0.6323 |
| proposed_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 0.0049 | (-0.0211, 0.0308) | 0.4007 | 0.0049 | (-0.0234, 0.0308) | 0.2820 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.0250 | (0.0000, 0.0750) | 0.2957 | 0.0250 | (0.0000, 0.0500) | 0.3180 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 0.0385 | (0.0195, 0.0682) | 0.0000 | 0.0385 | (0.0162, 0.0576) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.0417 | (0.0000, 0.1250) | 0.3187 | 0.0417 | (0.0000, 0.1667) | 0.2980 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | -0.0374 | (-0.0755, 0.0062) | 0.9717 | -0.0374 | (-0.0602, 0.0270) | 0.9633 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | -0.1167 | (-0.5167, 0.3917) | 0.6837 | -0.1167 | (-0.6000, 0.1167) | 0.7347 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | -0.2625 | (-0.3500, -0.0875) | 1.0000 | -0.2625 | (-0.3500, -0.1750) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | 0.0120 | (-0.0306, 0.0547) | 0.3233 | 0.0120 | (-0.0331, 0.0429) | 0.2680 |
| controlled_vs_proposed_raw | context_relevance | 0.1624 | (0.1084, 0.2165) | 0.0000 | 0.1624 | (0.1385, 0.2310) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.1905 | (0.0571, 0.3333) | 0.0043 | 0.1905 | (0.0000, 0.2667) | 0.0360 |
| controlled_vs_proposed_raw | naturalness | 0.0623 | (-0.0243, 0.1490) | 0.0630 | 0.0623 | (-0.0028, 0.1406) | 0.0373 |
| controlled_vs_proposed_raw | quest_state_correctness | 0.1311 | (0.0748, 0.1776) | 0.0000 | 0.1311 | (0.1045, 0.1931) | 0.0000 |
| controlled_vs_proposed_raw | lore_consistency | 0.1306 | (0.1017, 0.1594) | 0.0000 | 0.1306 | (0.1082, 0.1683) | 0.0000 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0877 | (0.0208, 0.1546) | 0.0030 | 0.0877 | (0.0208, 0.1842) | 0.0000 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0992 | (0.0291, 0.1715) | 0.0030 | 0.0992 | (0.0629, 0.2076) | 0.0000 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 0.0432 | (0.0048, 0.0825) | 0.0040 | 0.0432 | (0.0048, 0.1068) | 0.0000 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2136 | (0.1409, 0.2864) | 0.0000 | 0.2136 | (0.1818, 0.3000) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0430 | (0.0302, 0.0616) | 0.0000 | 0.0430 | (0.0324, 0.0699) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2381 | (0.0714, 0.4167) | 0.0027 | 0.2381 | (0.0000, 0.3333) | 0.0360 |
| controlled_vs_proposed_raw | persona_style | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_proposed_raw | distinct1 | -0.0171 | (-0.0441, 0.0099) | 0.9330 | -0.0171 | (-0.0444, 0.0140) | 0.9620 |
| controlled_vs_proposed_raw | length_score | 0.2583 | (-0.0333, 0.5500) | 0.0577 | 0.2583 | (-0.1000, 0.5000) | 0.0373 |
| controlled_vs_proposed_raw | sentence_score | 0.1750 | (-0.1750, 0.3500) | 0.2517 | 0.1750 | (0.0000, 0.3500) | 0.0397 |
| controlled_vs_proposed_raw | overall_quality | 0.1595 | (0.0704, 0.2485) | 0.0000 | 0.1595 | (0.0638, 0.2210) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2044 | (0.1828, 0.2237) | 0.0000 | 0.2044 | (0.1877, 0.2310) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.2238 | (0.1333, 0.3571) | 0.0000 | 0.2238 | (0.1333, 0.2667) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.0894 | (-0.0082, 0.1490) | 0.0517 | 0.0894 | (-0.0578, 0.1406) | 0.0433 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.1641 | (0.1432, 0.1854) | 0.0000 | 0.1641 | (0.1489, 0.1931) | 0.0000 |
| controlled_vs_candidate_no_context | lore_consistency | 0.1417 | (0.1063, 0.1771) | 0.0000 | 0.1417 | (0.1175, 0.1683) | 0.0000 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | 0.0773 | (0.0208, 0.1485) | 0.0023 | 0.0773 | (0.0208, 0.1842) | 0.0000 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.1270 | (0.0796, 0.1825) | 0.0000 | 0.1270 | (0.0626, 0.2076) | 0.0000 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 0.0492 | (0.0180, 0.0895) | 0.0000 | 0.0492 | (0.0263, 0.1068) | 0.0000 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.2614 | (0.2182, 0.2932) | 0.0000 | 0.2614 | (0.2364, 0.3000) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0716 | (0.0452, 0.1004) | 0.0000 | 0.0716 | (0.0683, 0.0740) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2798 | (0.1667, 0.4167) | 0.0000 | 0.2798 | (0.1667, 0.3333) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | distinct1 | -0.0245 | (-0.0588, 0.0099) | 0.9427 | -0.0245 | (-0.0444, 0.0140) | 0.9657 |
| controlled_vs_candidate_no_context | length_score | 0.4083 | (-0.0000, 0.6750) | 0.0373 | 0.4083 | (-0.2000, 0.6667) | 0.0350 |
| controlled_vs_candidate_no_context | sentence_score | 0.1750 | (0.0000, 0.3500) | 0.0607 | 0.1750 | (0.0000, 0.3500) | 0.0367 |
| controlled_vs_candidate_no_context | overall_quality | 0.1963 | (0.1441, 0.2485) | 0.0000 | 0.1963 | (0.1364, 0.2210) | 0.0000 |
| controlled_vs_baseline_no_context | context_relevance | 0.1914 | (0.1605, 0.2222) | 0.0000 | 0.1914 | (0.1477, 0.2331) | 0.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1952 | (0.1190, 0.3333) | 0.0000 | 0.1952 | (0.1143, 0.2667) | 0.0000 |
| controlled_vs_baseline_no_context | naturalness | -0.0065 | (-0.0504, 0.0375) | 0.6127 | -0.0065 | (-0.0574, 0.0375) | 0.7417 |
| controlled_vs_baseline_no_context | quest_state_correctness | 0.1581 | (0.1291, 0.1870) | 0.0000 | 0.1581 | (0.1225, 0.1939) | 0.0000 |
| controlled_vs_baseline_no_context | lore_consistency | 0.1464 | (0.1097, 0.1832) | 0.0000 | 0.1464 | (0.1004, 0.1804) | 0.0000 |
| controlled_vs_baseline_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_baseline_no_context | objective_completion_support | 0.0627 | (-0.0067, 0.1321) | 0.0620 | 0.0627 | (-0.0067, 0.1392) | 0.0377 |
| controlled_vs_baseline_no_context | gameplay_usefulness | 0.0780 | (0.0606, 0.0954) | 0.0000 | 0.0780 | (0.0677, 0.1022) | 0.0000 |
| controlled_vs_baseline_no_context | time_pressure_acceptability | 0.0198 | (-0.0294, 0.0685) | 0.2510 | 0.0198 | (-0.0290, 0.0807) | 0.2580 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.2386 | (0.1909, 0.2932) | 0.0000 | 0.2386 | (0.1818, 0.3000) | 0.0000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0811 | (0.0681, 0.1004) | 0.0000 | 0.0811 | (0.0681, 0.0896) | 0.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2440 | (0.1488, 0.4167) | 0.0000 | 0.2440 | (0.1429, 0.3333) | 0.0000 |
| controlled_vs_baseline_no_context | persona_style | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_baseline_no_context | distinct1 | -0.0568 | (-0.0821, -0.0131) | 0.9967 | -0.0568 | (-0.0791, 0.0082) | 0.9617 |
| controlled_vs_baseline_no_context | length_score | 0.1250 | (-0.1833, 0.4417) | 0.2270 | 0.1250 | (-0.2333, 0.4333) | 0.3020 |
| controlled_vs_baseline_no_context | sentence_score | -0.0875 | (-0.2625, 0.0000) | 1.0000 | -0.0875 | (-0.1750, 0.0000) | 1.0000 |
| controlled_vs_baseline_no_context | overall_quality | 0.1599 | (0.1170, 0.2226) | 0.0000 | 0.1599 | (0.1090, 0.1957) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.1915 | (0.1583, 0.2247) | 0.0000 | 0.1915 | (0.1434, 0.2378) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.2238 | (0.1333, 0.3333) | 0.0000 | 0.2238 | (0.1333, 0.2667) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | -0.0022 | (-0.0420, 0.0476) | 0.5800 | -0.0022 | (-0.0470, 0.0375) | 0.6237 |
| controlled_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.1508 | (0.1137, 0.1879) | 0.0000 | 0.1508 | (0.0916, 0.1956) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | lore_consistency | 0.1475 | (0.1117, 0.1832) | 0.0000 | 0.1475 | (0.1046, 0.1804) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.0869 | (0.0208, 0.1565) | 0.0033 | 0.0869 | (0.0417, 0.1808) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 0.0890 | (0.0584, 0.1174) | 0.0000 | 0.0890 | (0.0457, 0.1276) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 0.0481 | (0.0320, 0.0720) | 0.0000 | 0.0481 | (0.0356, 0.0835) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.2386 | (0.1909, 0.2864) | 0.0000 | 0.2386 | (0.1818, 0.3000) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0815 | (0.0612, 0.1019) | 0.0000 | 0.0815 | (0.0536, 0.0926) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.2798 | (0.1667, 0.4167) | 0.0000 | 0.2798 | (0.1667, 0.3333) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0545 | (-0.0791, -0.0299) | 1.0000 | -0.0545 | (-0.0791, -0.0174) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | 0.1417 | (-0.1500, 0.5000) | 0.2117 | 0.1417 | (-0.2000, 0.4333) | 0.2947 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | -0.0875 | (-0.2625, 0.0000) | 1.0000 | -0.0875 | (-0.1750, 0.0000) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1715 | (0.1252, 0.2218) | 0.0000 | 0.1715 | (0.1068, 0.1957) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.1914 | (0.1605, 0.2222) | 0.0000 | 0.1914 | (0.1477, 0.2331) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1952 | (0.1190, 0.3333) | 0.0000 | 0.1952 | (0.1143, 0.2667) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0065 | (-0.0504, 0.0375) | 0.6200 | -0.0065 | (-0.0574, 0.0375) | 0.7347 |
| proposed_contextual_controlled_vs_baseline_no_context | quest_state_correctness | 0.1581 | (0.1291, 0.1870) | 0.0000 | 0.1581 | (0.1225, 0.1939) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | lore_consistency | 0.1464 | (0.1097, 0.1832) | 0.0000 | 0.1464 | (0.1004, 0.1804) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | objective_completion_support | 0.0627 | (-0.0067, 0.1321) | 0.0547 | 0.0627 | (-0.0067, 0.1392) | 0.0367 |
| proposed_contextual_controlled_vs_baseline_no_context | gameplay_usefulness | 0.0780 | (0.0606, 0.0954) | 0.0000 | 0.0780 | (0.0677, 0.1022) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | time_pressure_acceptability | 0.0198 | (-0.0294, 0.0685) | 0.2563 | 0.0198 | (-0.0290, 0.0807) | 0.2537 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.2386 | (0.1909, 0.2864) | 0.0000 | 0.2386 | (0.1818, 0.3000) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0811 | (0.0681, 0.1004) | 0.0000 | 0.0811 | (0.0681, 0.0896) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2440 | (0.1488, 0.4167) | 0.0000 | 0.2440 | (0.1429, 0.3333) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0568 | (-0.0821, -0.0131) | 0.9953 | -0.0568 | (-0.0791, 0.0082) | 0.9690 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.1250 | (-0.1833, 0.4417) | 0.2203 | 0.1250 | (-0.2333, 0.4333) | 0.2917 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | -0.0875 | (-0.2625, 0.0000) | 1.0000 | -0.0875 | (-0.1750, 0.0000) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1599 | (0.1170, 0.2226) | 0.0000 | 0.1599 | (0.1090, 0.1957) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.1915 | (0.1583, 0.2247) | 0.0000 | 0.1915 | (0.1434, 0.2378) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.2238 | (0.1333, 0.3333) | 0.0000 | 0.2238 | (0.1333, 0.2667) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | -0.0022 | (-0.0420, 0.0476) | 0.5790 | -0.0022 | (-0.0470, 0.0375) | 0.6340 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.1508 | (0.1137, 0.1879) | 0.0000 | 0.1508 | (0.0916, 0.1956) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | lore_consistency | 0.1475 | (0.1117, 0.1832) | 0.0000 | 0.1475 | (0.1046, 0.1804) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.0869 | (0.0208, 0.1565) | 0.0027 | 0.0869 | (0.0417, 0.1808) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 0.0890 | (0.0584, 0.1174) | 0.0000 | 0.0890 | (0.0457, 0.1276) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 0.0481 | (0.0320, 0.0720) | 0.0000 | 0.0481 | (0.0356, 0.0835) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.2386 | (0.1909, 0.2932) | 0.0000 | 0.2386 | (0.1818, 0.3000) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0815 | (0.0612, 0.1019) | 0.0000 | 0.0815 | (0.0536, 0.0926) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.2798 | (0.1667, 0.4167) | 0.0000 | 0.2798 | (0.1667, 0.3333) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0545 | (-0.0791, -0.0299) | 1.0000 | -0.0545 | (-0.0791, -0.0174) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | 0.1417 | (-0.1500, 0.5000) | 0.2170 | 0.1417 | (-0.2000, 0.4333) | 0.3077 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | -0.0875 | (-0.2625, 0.0000) | 1.0000 | -0.0875 | (-0.1750, 0.0000) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1715 | (0.1252, 0.2218) | 0.0000 | 0.1715 | (0.1068, 0.1957) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 2 | 0 | 2 | 0.7500 | 1.0000 |
| proposed_vs_candidate_no_context | persona_consistency | 1 | 0 | 3 | 0.6250 | 1.0000 |
| proposed_vs_candidate_no_context | naturalness | 1 | 1 | 2 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | quest_state_correctness | 2 | 0 | 2 | 0.7500 | 1.0000 |
| proposed_vs_candidate_no_context | lore_consistency | 2 | 0 | 2 | 0.7500 | 1.0000 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 4 | 0.5000 | nan |
| proposed_vs_candidate_no_context | objective_completion_support | 0 | 1 | 3 | 0.3750 | 0.0000 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 1 | 1 | 2 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | 1 | 1 | 2 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 2 | 0 | 2 | 0.7500 | 1.0000 |
| proposed_vs_candidate_no_context | context_overlap | 2 | 0 | 2 | 0.7500 | 1.0000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 1 | 0 | 3 | 0.6250 | 1.0000 |
| proposed_vs_candidate_no_context | persona_style | 0 | 0 | 4 | 0.5000 | nan |
| proposed_vs_candidate_no_context | distinct1 | 0 | 1 | 3 | 0.3750 | 0.0000 |
| proposed_vs_candidate_no_context | length_score | 1 | 1 | 2 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | sentence_score | 1 | 1 | 2 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | overall_quality | 2 | 0 | 2 | 0.7500 | 1.0000 |
| proposed_vs_baseline_no_context | context_relevance | 4 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_vs_baseline_no_context | persona_consistency | 1 | 1 | 2 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context | naturalness | 1 | 3 | 0 | 0.2500 | 0.2500 |
| proposed_vs_baseline_no_context | quest_state_correctness | 4 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_vs_baseline_no_context | lore_consistency | 3 | 1 | 0 | 0.7500 | 0.7500 |
| proposed_vs_baseline_no_context | multi_turn_contradiction_safety | 0 | 0 | 4 | 0.5000 | nan |
| proposed_vs_baseline_no_context | objective_completion_support | 0 | 2 | 2 | 0.2500 | 0.0000 |
| proposed_vs_baseline_no_context | gameplay_usefulness | 2 | 2 | 0 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context | time_pressure_acceptability | 0 | 2 | 2 | 0.2500 | 0.0000 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 1 | 0 | 3 | 0.6250 | 1.0000 |
| proposed_vs_baseline_no_context | context_overlap | 4 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 1 | 1 | 2 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context | persona_style | 0 | 0 | 4 | 0.5000 | nan |
| proposed_vs_baseline_no_context | distinct1 | 1 | 3 | 0 | 0.2500 | 0.2500 |
| proposed_vs_baseline_no_context | length_score | 1 | 3 | 0 | 0.2500 | 0.2500 |
| proposed_vs_baseline_no_context | sentence_score | 0 | 3 | 1 | 0.1250 | 0.0000 |
| proposed_vs_baseline_no_context | overall_quality | 2 | 2 | 0 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 4 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | 1 | 0 | 3 | 0.6250 | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | 1 | 3 | 0 | 0.2500 | 0.2500 |
| proposed_vs_baseline_no_context_phi3_latest | quest_state_correctness | 3 | 1 | 0 | 0.7500 | 0.7500 |
| proposed_vs_baseline_no_context_phi3_latest | lore_consistency | 3 | 1 | 0 | 0.7500 | 0.7500 |
| proposed_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0 | 0 | 4 | 0.5000 | nan |
| proposed_vs_baseline_no_context_phi3_latest | objective_completion_support | 1 | 2 | 1 | 0.3750 | 0.3333 |
| proposed_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 1 | 3 | 0 | 0.2500 | 0.2500 |
| proposed_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 2 | 2 | 0 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 1 | 0 | 3 | 0.6250 | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 4 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 1 | 0 | 3 | 0.6250 | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | 0 | 0 | 4 | 0.5000 | nan |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | 1 | 3 | 0 | 0.2500 | 0.2500 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | 1 | 3 | 0 | 0.2500 | 0.2500 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | 0 | 3 | 1 | 0.1250 | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | 2 | 2 | 0 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | context_relevance | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_proposed_raw | persona_consistency | 3 | 0 | 1 | 0.8750 | 1.0000 |
| controlled_vs_proposed_raw | naturalness | 2 | 2 | 0 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | quest_state_correctness | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_proposed_raw | lore_consistency | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0 | 0 | 4 | 0.5000 | nan |
| controlled_vs_proposed_raw | objective_completion_support | 3 | 0 | 1 | 0.8750 | 1.0000 |
| controlled_vs_proposed_raw | gameplay_usefulness | 3 | 1 | 0 | 0.7500 | 0.7500 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 3 | 0 | 1 | 0.8750 | 1.0000 |
| controlled_vs_proposed_raw | context_keyword_coverage | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_proposed_raw | context_overlap | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 3 | 0 | 1 | 0.8750 | 1.0000 |
| controlled_vs_proposed_raw | persona_style | 0 | 0 | 4 | 0.5000 | nan |
| controlled_vs_proposed_raw | distinct1 | 2 | 2 | 0 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | length_score | 3 | 1 | 0 | 0.7500 | 0.7500 |
| controlled_vs_proposed_raw | sentence_score | 3 | 1 | 0 | 0.7500 | 0.7500 |
| controlled_vs_proposed_raw | overall_quality | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_relevance | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | naturalness | 3 | 1 | 0 | 0.7500 | 0.7500 |
| controlled_vs_candidate_no_context | quest_state_correctness | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | lore_consistency | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 4 | 0.5000 | nan |
| controlled_vs_candidate_no_context | objective_completion_support | 3 | 0 | 1 | 0.8750 | 1.0000 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_style | 0 | 0 | 4 | 0.5000 | nan |
| controlled_vs_candidate_no_context | distinct1 | 2 | 2 | 0 | 0.5000 | 0.5000 |
| controlled_vs_candidate_no_context | length_score | 3 | 1 | 0 | 0.7500 | 0.7500 |
| controlled_vs_candidate_no_context | sentence_score | 2 | 0 | 2 | 0.7500 | 1.0000 |
| controlled_vs_candidate_no_context | overall_quality | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | context_relevance | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | naturalness | 2 | 2 | 0 | 0.5000 | 0.5000 |
| controlled_vs_baseline_no_context | quest_state_correctness | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | lore_consistency | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | multi_turn_contradiction_safety | 0 | 0 | 4 | 0.5000 | nan |
| controlled_vs_baseline_no_context | objective_completion_support | 2 | 1 | 1 | 0.6250 | 0.6667 |
| controlled_vs_baseline_no_context | gameplay_usefulness | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | time_pressure_acceptability | 2 | 1 | 1 | 0.6250 | 0.6667 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | context_overlap | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_style | 0 | 0 | 4 | 0.5000 | nan |
| controlled_vs_baseline_no_context | distinct1 | 1 | 3 | 0 | 0.2500 | 0.2500 |
| controlled_vs_baseline_no_context | length_score | 2 | 2 | 0 | 0.5000 | 0.5000 |
| controlled_vs_baseline_no_context | sentence_score | 0 | 1 | 3 | 0.3750 | 0.0000 |
| controlled_vs_baseline_no_context | overall_quality | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | 1 | 3 | 0 | 0.2500 | 0.2500 |
| controlled_vs_baseline_no_context_phi3_latest | quest_state_correctness | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | lore_consistency | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0 | 0 | 4 | 0.5000 | nan |
| controlled_vs_baseline_no_context_phi3_latest | objective_completion_support | 3 | 0 | 1 | 0.8750 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | 0 | 0 | 4 | 0.5000 | nan |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | 0 | 4 | 0 | 0.0000 | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | 2 | 2 | 0 | 0.5000 | 0.5000 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0 | 1 | 3 | 0.3750 | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 4 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 4 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 4 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 2 | 2 | 0 | 0.5000 | 0.5000 |
| proposed_contextual_controlled_vs_baseline_no_context | quest_state_correctness | 4 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | lore_consistency | 4 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | multi_turn_contradiction_safety | 0 | 0 | 4 | 0.5000 | nan |
| proposed_contextual_controlled_vs_baseline_no_context | objective_completion_support | 2 | 1 | 1 | 0.6250 | 0.6667 |
| proposed_contextual_controlled_vs_baseline_no_context | gameplay_usefulness | 4 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | time_pressure_acceptability | 2 | 1 | 1 | 0.6250 | 0.6667 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 4 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 4 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 4 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0 | 0 | 4 | 0.5000 | nan |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | 1 | 3 | 0 | 0.2500 | 0.2500 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 2 | 2 | 0 | 0.5000 | 0.5000 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0 | 1 | 3 | 0.3750 | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 4 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 4 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 4 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | 1 | 3 | 0 | 0.2500 | 0.2500 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | quest_state_correctness | 4 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | lore_consistency | 4 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0 | 0 | 4 | 0.5000 | nan |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | objective_completion_support | 3 | 0 | 1 | 0.8750 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 4 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 4 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 4 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 4 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 4 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | 0 | 0 | 4 | 0.5000 | nan |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | 0 | 4 | 0 | 0.0000 | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | 2 | 2 | 0 | 0.5000 | 0.5000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0 | 1 | 3 | 0.3750 | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 4 | 0 | 0 | 1.0000 | 1.0000 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.0000 | 0.7500 | 0.0000 |
| proposed_contextual | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 |
| baseline_no_context | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline_no_context_phi3_latest | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `3`
- Unique template signatures: `4`
- Template signature ratio: `1.0000`
- Effective sample size by source clustering: `2.67`
- Effective sample size by template-signature clustering: `4.00`
- Detailed diagnostics are published in `scenario_dependence.json`.

## Multi-turn Contradiction
| Arm | Contradiction Rate | Contradiction Safety | Contradicted Sources | Source Count |
|---|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 1.0000 | 0 | 3 |
| proposed_contextual | 0.0000 | 1.0000 | 0 | 3 |
| candidate_no_context | 0.0000 | 1.0000 | 0 | 3 |
| baseline_no_context | 0.0000 | 1.0000 | 0 | 3 |
| baseline_no_context_phi3_latest | 0.0000 | 1.0000 | 0 | 3 |
- Detailed source-level values are published in `multi_turn_contradictions.json`.

- BERTScore status: unavailable (No module named 'bert_score').

This report covers proposal RO5 metrics plus game-facing outcomes: quest-state correctness, lore consistency, contradiction safety, objective completion support, gameplay usefulness, and time-pressure acceptability.
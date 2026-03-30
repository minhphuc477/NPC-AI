# Proposal Alignment Evaluation Report

- Run ID: `20260310T020702Z`
- Generated: `2026-03-10T02:08:47.072927+00:00`
- Scenarios: `tmp\proposal_smoke_fix_v2\20260310T020702Z\scenarios.jsonl`
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
| proposed_contextual_controlled | 0.1791 (0.0766, 0.2817) | 0.2733 (0.2333, 0.3133) | 0.8893 (0.8642, 0.9148) | 0.3522 (0.2915, 0.4130) | n/a |
| proposed_contextual | 0.2459 (0.1167, 0.3452) | 0.2467 (0.2067, 0.3067) | 0.8509 (0.7897, 0.9099) | 0.3662 (0.2931, 0.4289) | n/a |
| candidate_no_context | 0.0229 (0.0072, 0.0501) | 0.2200 (0.1533, 0.2867) | 0.9007 (0.8613, 0.9540) | 0.2618 (0.2383, 0.2826) | n/a |
| baseline_no_context | 0.0100 (0.0048, 0.0141) | 0.1375 (0.1000, 0.1775) | 0.8628 (0.7842, 0.9292) | 0.2174 (0.2010, 0.2338) | n/a |
| baseline_no_context_phi3_latest | 0.0500 (0.0094, 0.1011) | 0.1657 (0.1000, 0.2590) | 0.9041 (0.8557, 0.9528) | 0.2550 (0.2152, 0.3005) | n/a |

## Game-facing Outcome Metrics (mean, 95% CI)
| Arm | Quest-state Correctness | Lore Consistency | Contradiction Safety | Objective Completion Support | Gameplay Usefulness | Time-pressure Acceptability |
|---|---:|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2401 (0.1482, 0.3237) | 0.1061 (0.0354, 0.1768) | 1.0000 (1.0000, 1.0000) | 0.0843 (0.0250, 0.1437) | 0.3429 (0.3195, 0.3645) | 0.3053 (0.2743, 0.3386) |
| proposed_contextual | 0.2996 (0.1989, 0.3771) | 0.1150 (0.0578, 0.1646) | 1.0000 (1.0000, 1.0000) | 0.0663 (0.0110, 0.1300) | 0.3214 (0.2622, 0.3842) | 0.2860 (0.2646, 0.3126) |
| candidate_no_context | 0.1152 (0.1030, 0.1325) | 0.0049 (0.0000, 0.0148) | 1.0000 (1.0000, 1.0000) | 0.1307 (0.0797, 0.1680) | 0.3179 (0.3090, 0.3319) | 0.3493 (0.3280, 0.3695) |
| baseline_no_context | 0.1065 (0.1019, 0.1131) | 0.0021 (0.0000, 0.0063) | 1.0000 (1.0000, 1.0000) | 0.0467 (0.0080, 0.0940) | 0.2595 (0.2265, 0.2905) | 0.2779 (0.2580, 0.2992) |
| baseline_no_context_phi3_latest | 0.1392 (0.1063, 0.1824) | 0.0211 (0.0029, 0.0416) | 1.0000 (1.0000, 1.0000) | 0.0323 (0.0000, 0.0650) | 0.2794 (0.2410, 0.3189) | 0.2739 (0.2374, 0.3075) |

- Multi-turn contradiction rate is reported as `1 - contradiction_safety` in row-level outputs.

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.2230 | 9.7584 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0267 | 0.1212 |
| proposed_vs_candidate_no_context | naturalness | -0.0497 | -0.0552 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.1844 | 1.6014 |
| proposed_vs_candidate_no_context | lore_consistency | 0.1101 | 22.3748 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | -0.0643 | -0.4923 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 0.0035 | 0.0109 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | -0.0633 | -0.1813 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.3000 | 16.5000 |
| proposed_vs_candidate_no_context | context_overlap | 0.0434 | 1.2848 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0333 | 0.3333 |
| proposed_vs_candidate_no_context | persona_style | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | distinct1 | 0.0007 | 0.0007 |
| proposed_vs_candidate_no_context | length_score | -0.1800 | -0.2967 |
| proposed_vs_candidate_no_context | sentence_score | -0.1400 | -0.1400 |
| proposed_vs_candidate_no_context | overall_quality | 0.1043 | 0.3985 |
| proposed_vs_baseline_no_context | context_relevance | 0.2359 | 23.5773 |
| proposed_vs_baseline_no_context | persona_consistency | 0.1092 | 0.7939 |
| proposed_vs_baseline_no_context | naturalness | -0.0119 | -0.0138 |
| proposed_vs_baseline_no_context | quest_state_correctness | 0.1930 | 1.8120 |
| proposed_vs_baseline_no_context | lore_consistency | 0.1129 | 53.8158 |
| proposed_vs_baseline_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| proposed_vs_baseline_no_context | objective_completion_support | 0.0197 | 0.4214 |
| proposed_vs_baseline_no_context | gameplay_usefulness | 0.0619 | 0.2386 |
| proposed_vs_baseline_no_context | time_pressure_acceptability | 0.0080 | 0.0289 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.3182 | nan |
| proposed_vs_baseline_no_context | context_overlap | 0.0438 | 1.3127 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 0.1333 | nan |
| proposed_vs_baseline_no_context | persona_style | 0.0125 | 0.0182 |
| proposed_vs_baseline_no_context | distinct1 | -0.0198 | -0.0204 |
| proposed_vs_baseline_no_context | length_score | -0.0200 | -0.0448 |
| proposed_vs_baseline_no_context | sentence_score | 0.0000 | 0.0000 |
| proposed_vs_baseline_no_context | overall_quality | 0.1487 | 0.6841 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 0.1959 | 3.9219 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | 0.0810 | 0.4887 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | -0.0531 | -0.0588 |
| proposed_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.1604 | 1.1524 |
| proposed_vs_baseline_no_context_phi3_latest | lore_consistency | 0.0939 | 4.4490 |
| proposed_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.0340 | 1.0515 |
| proposed_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 0.0420 | 0.1504 |
| proposed_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 0.0121 | 0.0441 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.2618 | 4.6452 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 0.0421 | 1.2037 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1000 | 3.0000 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | 0.0049 | 0.0070 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | -0.0354 | -0.0359 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | -0.1600 | -0.2727 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | -0.0700 | -0.0753 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | 0.1111 | 0.4358 |
| controlled_vs_proposed_raw | context_relevance | -0.0668 | -0.2717 |
| controlled_vs_proposed_raw | persona_consistency | 0.0267 | 0.1081 |
| controlled_vs_proposed_raw | naturalness | 0.0384 | 0.0451 |
| controlled_vs_proposed_raw | quest_state_correctness | -0.0595 | -0.1986 |
| controlled_vs_proposed_raw | lore_consistency | -0.0089 | -0.0774 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0180 | 0.2714 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0215 | 0.0670 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 0.0193 | 0.0676 |
| controlled_vs_proposed_raw | context_keyword_coverage | -0.0945 | -0.2971 |
| controlled_vs_proposed_raw | context_overlap | -0.0020 | -0.0263 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.0333 | 0.2500 |
| controlled_vs_proposed_raw | persona_style | 0.0000 | 0.0000 |
| controlled_vs_proposed_raw | distinct1 | -0.0098 | -0.0103 |
| controlled_vs_proposed_raw | length_score | 0.2467 | 0.5781 |
| controlled_vs_proposed_raw | sentence_score | -0.0700 | -0.0814 |
| controlled_vs_proposed_raw | overall_quality | -0.0139 | -0.0380 |
| controlled_vs_candidate_no_context | context_relevance | 0.1562 | 6.8358 |
| controlled_vs_candidate_no_context | persona_consistency | 0.0533 | 0.2424 |
| controlled_vs_candidate_no_context | naturalness | -0.0113 | -0.0126 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.1249 | 1.0846 |
| controlled_vs_candidate_no_context | lore_consistency | 0.1012 | 20.5662 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | -0.0463 | -0.3546 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0250 | 0.0787 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | -0.0440 | -0.1260 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.2055 | 11.3000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0413 | 1.2248 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.0667 | 0.6667 |
| controlled_vs_candidate_no_context | persona_style | 0.0000 | 0.0000 |
| controlled_vs_candidate_no_context | distinct1 | -0.0091 | -0.0096 |
| controlled_vs_candidate_no_context | length_score | 0.0667 | 0.1099 |
| controlled_vs_candidate_no_context | sentence_score | -0.2100 | -0.2100 |
| controlled_vs_candidate_no_context | overall_quality | 0.0904 | 0.3453 |
| controlled_vs_baseline_no_context | context_relevance | 0.1691 | 16.9007 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1358 | 0.9879 |
| controlled_vs_baseline_no_context | naturalness | 0.0265 | 0.0307 |
| controlled_vs_baseline_no_context | quest_state_correctness | 0.1335 | 1.2534 |
| controlled_vs_baseline_no_context | lore_consistency | 0.1040 | 49.5744 |
| controlled_vs_baseline_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_baseline_no_context | objective_completion_support | 0.0377 | 0.8071 |
| controlled_vs_baseline_no_context | gameplay_usefulness | 0.0835 | 0.3217 |
| controlled_vs_baseline_no_context | time_pressure_acceptability | 0.0274 | 0.0985 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.2236 | nan |
| controlled_vs_baseline_no_context | context_overlap | 0.0417 | 1.2520 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1667 | nan |
| controlled_vs_baseline_no_context | persona_style | 0.0125 | 0.0182 |
| controlled_vs_baseline_no_context | distinct1 | -0.0296 | -0.0305 |
| controlled_vs_baseline_no_context | length_score | 0.2267 | 0.5075 |
| controlled_vs_baseline_no_context | sentence_score | -0.0700 | -0.0814 |
| controlled_vs_baseline_no_context | overall_quality | 0.1348 | 0.6200 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.1291 | 2.5848 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1076 | 0.6497 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | -0.0147 | -0.0163 |
| controlled_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.1009 | 0.7248 |
| controlled_vs_baseline_no_context_phi3_latest | lore_consistency | 0.0850 | 4.0274 |
| controlled_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.0520 | 1.6082 |
| controlled_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 0.0635 | 0.2275 |
| controlled_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 0.0314 | 0.1147 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.1673 | 2.9677 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0401 | 1.1458 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1333 | 4.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | 0.0049 | 0.0070 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0452 | -0.0459 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | 0.0867 | 0.1477 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | -0.1400 | -0.1505 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.0972 | 0.3812 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.1691 | 16.9007 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1358 | 0.9879 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 0.0265 | 0.0307 |
| proposed_contextual_controlled_vs_baseline_no_context | quest_state_correctness | 0.1335 | 1.2534 |
| proposed_contextual_controlled_vs_baseline_no_context | lore_consistency | 0.1040 | 49.5744 |
| proposed_contextual_controlled_vs_baseline_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | objective_completion_support | 0.0377 | 0.8071 |
| proposed_contextual_controlled_vs_baseline_no_context | gameplay_usefulness | 0.0835 | 0.3217 |
| proposed_contextual_controlled_vs_baseline_no_context | time_pressure_acceptability | 0.0274 | 0.0985 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.2236 | nan |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0417 | 1.2520 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1667 | nan |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0125 | 0.0182 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0296 | -0.0305 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.2267 | 0.5075 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | -0.0700 | -0.0814 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1348 | 0.6200 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.1291 | 2.5848 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1076 | 0.6497 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | -0.0147 | -0.0163 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.1009 | 0.7248 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | lore_consistency | 0.0850 | 4.0274 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.0520 | 1.6082 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 0.0635 | 0.2275 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 0.0314 | 0.1147 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.1673 | 2.9677 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0401 | 1.1458 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1333 | 4.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | 0.0049 | 0.0070 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0452 | -0.0459 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | 0.0867 | 0.1477 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | -0.1400 | -0.1505 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.0972 | 0.3812 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.2230 | (0.1067, 0.3292) | 0.0000 | 0.2230 | (0.1681, 0.2596) | 0.0000 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0267 | (-0.0800, 0.1067) | 0.3813 | 0.0267 | (0.0000, 0.0444) | 0.2577 |
| proposed_vs_candidate_no_context | naturalness | -0.0497 | (-0.0809, -0.0156) | 0.9987 | -0.0497 | (-0.0603, -0.0427) | 1.0000 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.1844 | (0.0909, 0.2675) | 0.0000 | 0.1844 | (0.1472, 0.2092) | 0.0000 |
| proposed_vs_candidate_no_context | lore_consistency | 0.1101 | (0.0555, 0.1570) | 0.0000 | 0.1101 | (0.0838, 0.1277) | 0.0000 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | -0.0643 | (-0.1354, 0.0123) | 0.9477 | -0.0643 | (-0.0711, -0.0542) | 1.0000 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 0.0035 | (-0.0605, 0.0683) | 0.4420 | 0.0035 | (-0.0027, 0.0076) | 0.2503 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | -0.0633 | (-0.1046, -0.0132) | 0.9917 | -0.0633 | (-0.0822, -0.0350) | 1.0000 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.3000 | (0.1455, 0.4455) | 0.0007 | 0.3000 | (0.2273, 0.3485) | 0.0000 |
| proposed_vs_candidate_no_context | context_overlap | 0.0434 | (0.0119, 0.0664) | 0.0047 | 0.0434 | (0.0300, 0.0522) | 0.0000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0333 | (-0.0667, 0.1333) | 0.3693 | 0.0333 | (0.0000, 0.0556) | 0.2470 |
| proposed_vs_candidate_no_context | persona_style | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | distinct1 | 0.0007 | (-0.0480, 0.0678) | 0.5033 | 0.0007 | (-0.0235, 0.0369) | 0.2567 |
| proposed_vs_candidate_no_context | length_score | -0.1800 | (-0.3533, 0.0000) | 0.9770 | -0.1800 | (-0.2000, -0.1667) | 1.0000 |
| proposed_vs_candidate_no_context | sentence_score | -0.1400 | (-0.2800, 0.0000) | 1.0000 | -0.1400 | (-0.3500, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | overall_quality | 0.1043 | (0.0265, 0.1760) | 0.0023 | 0.1043 | (0.0666, 0.1295) | 0.0000 |
| proposed_vs_baseline_no_context | context_relevance | 0.2359 | (0.1182, 0.3339) | 0.0000 | 0.2359 | (0.1710, 0.2791) | 0.0000 |
| proposed_vs_baseline_no_context | persona_consistency | 0.1092 | (0.0292, 0.1892) | 0.0007 | 0.1092 | (0.0729, 0.1333) | 0.0000 |
| proposed_vs_baseline_no_context | naturalness | -0.0119 | (-0.0554, 0.0178) | 0.6700 | -0.0119 | (-0.0214, 0.0022) | 0.7447 |
| proposed_vs_baseline_no_context | quest_state_correctness | 0.1930 | (0.0958, 0.2714) | 0.0000 | 0.1930 | (0.1482, 0.2229) | 0.0000 |
| proposed_vs_baseline_no_context | lore_consistency | 0.1129 | (0.0554, 0.1646) | 0.0000 | 0.1129 | (0.0785, 0.1359) | 0.0000 |
| proposed_vs_baseline_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_baseline_no_context | objective_completion_support | 0.0197 | (-0.0157, 0.0470) | 0.0963 | 0.0197 | (0.0189, 0.0208) | 0.0000 |
| proposed_vs_baseline_no_context | gameplay_usefulness | 0.0619 | (0.0271, 0.0940) | 0.0000 | 0.0619 | (0.0558, 0.0660) | 0.0000 |
| proposed_vs_baseline_no_context | time_pressure_acceptability | 0.0080 | (-0.0106, 0.0300) | 0.2450 | 0.0080 | (0.0072, 0.0094) | 0.0000 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.3182 | (0.1455, 0.4457) | 0.0000 | 0.3182 | (0.2273, 0.3788) | 0.0000 |
| proposed_vs_baseline_no_context | context_overlap | 0.0438 | (0.0309, 0.0555) | 0.0000 | 0.0438 | (0.0395, 0.0466) | 0.0000 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 0.1333 | (0.0333, 0.2333) | 0.0083 | 0.1333 | (0.0833, 0.1667) | 0.0000 |
| proposed_vs_baseline_no_context | persona_style | 0.0125 | (0.0000, 0.0375) | 0.3257 | 0.0125 | (0.0000, 0.0312) | 0.2550 |
| proposed_vs_baseline_no_context | distinct1 | -0.0198 | (-0.0443, -0.0026) | 1.0000 | -0.0198 | (-0.0270, -0.0090) | 1.0000 |
| proposed_vs_baseline_no_context | length_score | -0.0200 | (-0.2000, 0.1400) | 0.6320 | -0.0200 | (-0.1111, 0.1167) | 0.7580 |
| proposed_vs_baseline_no_context | sentence_score | 0.0000 | (-0.2100, 0.2100) | 0.6477 | 0.0000 | (-0.1750, 0.1167) | 0.7507 |
| proposed_vs_baseline_no_context | overall_quality | 0.1487 | (0.0929, 0.2035) | 0.0000 | 0.1487 | (0.1070, 0.1765) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 0.1959 | (0.0955, 0.2737) | 0.0000 | 0.1959 | (0.1029, 0.2579) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | 0.0810 | (-0.0504, 0.1876) | 0.0760 | 0.0810 | (-0.0000, 0.1350) | 0.2410 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | -0.0531 | (-0.1366, 0.0333) | 0.8747 | -0.0531 | (-0.1078, 0.0289) | 0.7527 |
| proposed_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.1604 | (0.0770, 0.2147) | 0.0000 | 0.1604 | (0.0890, 0.2080) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | lore_consistency | 0.0939 | (0.0466, 0.1405) | 0.0000 | 0.0939 | (0.0473, 0.1250) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.0340 | (-0.0287, 0.0890) | 0.1430 | 0.0340 | (0.0244, 0.0483) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 0.0420 | (-0.0082, 0.1088) | 0.0607 | 0.0420 | (0.0193, 0.0761) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 0.0121 | (-0.0429, 0.0697) | 0.3707 | 0.0121 | (-0.0201, 0.0604) | 0.2527 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.2618 | (0.1273, 0.3673) | 0.0000 | 0.2618 | (0.1364, 0.3455) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 0.0421 | (0.0214, 0.0552) | 0.0000 | 0.0421 | (0.0247, 0.0537) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1000 | (-0.0333, 0.2333) | 0.1363 | 0.1000 | (0.0000, 0.1667) | 0.2553 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | 0.0049 | (0.0000, 0.0146) | 0.3220 | 0.0049 | (0.0000, 0.0081) | 0.2450 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | -0.0354 | (-0.0584, -0.0115) | 1.0000 | -0.0354 | (-0.0418, -0.0256) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | -0.1600 | (-0.6133, 0.3600) | 0.7550 | -0.1600 | (-0.4556, 0.2833) | 0.7447 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | -0.0700 | (-0.2100, 0.0000) | 1.0000 | -0.0700 | (-0.1750, 0.0000) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | 0.1111 | (0.0528, 0.1732) | 0.0000 | 0.1111 | (0.0537, 0.1494) | 0.0000 |
| controlled_vs_proposed_raw | context_relevance | -0.0668 | (-0.1502, 0.0053) | 0.9400 | -0.0668 | (-0.1172, 0.0088) | 0.7487 |
| controlled_vs_proposed_raw | persona_consistency | 0.0267 | (-0.0533, 0.1067) | 0.3660 | 0.0267 | (0.0000, 0.0667) | 0.2297 |
| controlled_vs_proposed_raw | naturalness | 0.0384 | (-0.0334, 0.1087) | 0.1687 | 0.0384 | (0.0168, 0.0528) | 0.0000 |
| controlled_vs_proposed_raw | quest_state_correctness | -0.0595 | (-0.1217, -0.0084) | 0.9997 | -0.0595 | (-0.0964, -0.0042) | 1.0000 |
| controlled_vs_proposed_raw | lore_consistency | -0.0089 | (-0.0768, 0.0511) | 0.5820 | -0.0089 | (-0.0244, 0.0014) | 0.7410 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0180 | (-0.0773, 0.1133) | 0.3613 | 0.0180 | (-0.0061, 0.0542) | 0.2533 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0215 | (-0.0322, 0.0753) | 0.2400 | 0.0215 | (0.0110, 0.0373) | 0.0000 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 0.0193 | (-0.0271, 0.0632) | 0.2147 | 0.0193 | (0.0136, 0.0232) | 0.0000 |
| controlled_vs_proposed_raw | context_keyword_coverage | -0.0945 | (-0.2036, 0.0000) | 1.0000 | -0.0945 | (-0.1576, 0.0000) | 1.0000 |
| controlled_vs_proposed_raw | context_overlap | -0.0020 | (-0.0325, 0.0257) | 0.5673 | -0.0020 | (-0.0229, 0.0293) | 0.7400 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.0333 | (-0.0667, 0.1333) | 0.3710 | 0.0333 | (0.0000, 0.0833) | 0.2437 |
| controlled_vs_proposed_raw | persona_style | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_proposed_raw | distinct1 | -0.0098 | (-0.0699, 0.0331) | 0.6227 | -0.0098 | (-0.0601, 0.0238) | 0.7393 |
| controlled_vs_proposed_raw | length_score | 0.2467 | (-0.0867, 0.5933) | 0.0803 | 0.2467 | (0.1167, 0.3333) | 0.0000 |
| controlled_vs_proposed_raw | sentence_score | -0.0700 | (-0.2800, 0.1400) | 0.8127 | -0.0700 | (-0.2333, 0.1750) | 0.7420 |
| controlled_vs_proposed_raw | overall_quality | -0.0139 | (-0.0867, 0.0549) | 0.6560 | -0.0139 | (-0.0442, 0.0315) | 0.7527 |
| controlled_vs_candidate_no_context | context_relevance | 0.1562 | (0.0394, 0.2793) | 0.0090 | 0.1562 | (0.1424, 0.1769) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.0533 | (0.0000, 0.1067) | 0.0930 | 0.0533 | (0.0444, 0.0667) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | -0.0113 | (-0.0568, 0.0341) | 0.6957 | -0.0113 | (-0.0435, 0.0101) | 0.7383 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.1249 | (0.0321, 0.2177) | 0.0090 | 0.1249 | (0.1128, 0.1430) | 0.0000 |
| controlled_vs_candidate_no_context | lore_consistency | 0.1012 | (0.0256, 0.1788) | 0.0000 | 0.1012 | (0.0594, 0.1291) | 0.0000 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | -0.0463 | (-0.0950, 0.0000) | 1.0000 | -0.0463 | (-0.0772, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0250 | (0.0015, 0.0512) | 0.0093 | 0.0250 | (0.0186, 0.0346) | 0.0000 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | -0.0440 | (-0.0795, -0.0086) | 1.0000 | -0.0440 | (-0.0591, -0.0214) | 1.0000 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.2055 | (0.0545, 0.3564) | 0.0117 | 0.2055 | (0.1909, 0.2273) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0413 | (0.0042, 0.0843) | 0.0083 | 0.0413 | (0.0293, 0.0593) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.0667 | (0.0000, 0.1333) | 0.0790 | 0.0667 | (0.0556, 0.0833) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | distinct1 | -0.0091 | (-0.0279, 0.0027) | 0.8030 | -0.0091 | (-0.0233, 0.0003) | 0.7387 |
| controlled_vs_candidate_no_context | length_score | 0.0667 | (-0.1067, 0.3000) | 0.3083 | 0.0667 | (-0.0833, 0.1667) | 0.2460 |
| controlled_vs_candidate_no_context | sentence_score | -0.2100 | (-0.3500, -0.0700) | 1.0000 | -0.2100 | (-0.2333, -0.1750) | 1.0000 |
| controlled_vs_candidate_no_context | overall_quality | 0.0904 | (0.0226, 0.1582) | 0.0000 | 0.0904 | (0.0852, 0.0982) | 0.0000 |
| controlled_vs_baseline_no_context | context_relevance | 0.1691 | (0.0644, 0.2776) | 0.0000 | 0.1691 | (0.1620, 0.1797) | 0.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1358 | (0.1333, 0.1408) | 0.0000 | 0.1358 | (0.1333, 0.1396) | 0.0000 |
| controlled_vs_baseline_no_context | naturalness | 0.0265 | (-0.0462, 0.1179) | 0.2713 | 0.0265 | (0.0190, 0.0315) | 0.0000 |
| controlled_vs_baseline_no_context | quest_state_correctness | 0.1335 | (0.0501, 0.2206) | 0.0000 | 0.1335 | (0.1265, 0.1440) | 0.0000 |
| controlled_vs_baseline_no_context | lore_consistency | 0.1040 | (0.0312, 0.1768) | 0.0007 | 0.1040 | (0.0541, 0.1373) | 0.0000 |
| controlled_vs_baseline_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_baseline_no_context | objective_completion_support | 0.0377 | (-0.0573, 0.1327) | 0.2613 | 0.0377 | (0.0128, 0.0750) | 0.0000 |
| controlled_vs_baseline_no_context | gameplay_usefulness | 0.0835 | (0.0472, 0.1156) | 0.0000 | 0.0835 | (0.0770, 0.0931) | 0.0000 |
| controlled_vs_baseline_no_context | time_pressure_acceptability | 0.0274 | (-0.0217, 0.0765) | 0.1320 | 0.0274 | (0.0229, 0.0303) | 0.0000 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.2236 | (0.0782, 0.3509) | 0.0000 | 0.2236 | (0.2212, 0.2273) | 0.0000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0417 | (0.0074, 0.0752) | 0.0067 | 0.0417 | (0.0237, 0.0688) | 0.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1667 | (0.1667, 0.1667) | 0.0000 | 0.1667 | (0.1667, 0.1667) | 0.0000 |
| controlled_vs_baseline_no_context | persona_style | 0.0125 | (0.0000, 0.0375) | 0.3253 | 0.0125 | (0.0000, 0.0312) | 0.2463 |
| controlled_vs_baseline_no_context | distinct1 | -0.0296 | (-0.0835, 0.0242) | 0.8340 | -0.0296 | (-0.0691, -0.0033) | 1.0000 |
| controlled_vs_baseline_no_context | length_score | 0.2267 | (-0.0867, 0.6267) | 0.1027 | 0.2267 | (0.2222, 0.2333) | 0.0000 |
| controlled_vs_baseline_no_context | sentence_score | -0.0700 | (-0.2800, 0.1400) | 0.8120 | -0.0700 | (-0.1167, 0.0000) | 1.0000 |
| controlled_vs_baseline_no_context | overall_quality | 0.1348 | (0.0833, 0.1842) | 0.0000 | 0.1348 | (0.1323, 0.1386) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.1291 | (0.0555, 0.1982) | 0.0000 | 0.1291 | (0.1116, 0.1408) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1076 | (0.0533, 0.1363) | 0.0007 | 0.1076 | (0.0667, 0.1350) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | -0.0147 | (-0.0644, 0.0379) | 0.7313 | -0.0147 | (-0.0550, 0.0457) | 0.7507 |
| controlled_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.1009 | (0.0419, 0.1575) | 0.0000 | 0.1009 | (0.0849, 0.1116) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | lore_consistency | 0.0850 | (0.0179, 0.1522) | 0.0017 | 0.0850 | (0.0229, 0.1264) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.0520 | (-0.0397, 0.1437) | 0.1270 | 0.0520 | (0.0183, 0.1025) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 0.0635 | (0.0104, 0.1106) | 0.0103 | 0.0635 | (0.0303, 0.1135) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 0.0314 | (-0.0172, 0.0800) | 0.1003 | 0.0314 | (0.0030, 0.0739) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.1673 | (0.0727, 0.2582) | 0.0007 | 0.1673 | (0.1364, 0.1879) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0401 | (0.0120, 0.0682) | 0.0000 | 0.0401 | (0.0308, 0.0540) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1333 | (0.0667, 0.1667) | 0.0000 | 0.1333 | (0.0833, 0.1667) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | 0.0049 | (0.0000, 0.0146) | 0.3353 | 0.0049 | (0.0000, 0.0081) | 0.2393 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0452 | (-0.0917, -0.0002) | 0.9750 | -0.0452 | (-0.0858, -0.0181) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | 0.0867 | (-0.2000, 0.3933) | 0.3120 | 0.0867 | (-0.1222, 0.4000) | 0.2503 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | -0.1400 | (-0.3500, 0.1400) | 0.9107 | -0.1400 | (-0.2333, 0.0000) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.0972 | (0.0726, 0.1218) | 0.0000 | 0.0972 | (0.0852, 0.1052) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.1691 | (0.0673, 0.2709) | 0.0000 | 0.1691 | (0.1620, 0.1797) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1358 | (0.1333, 0.1408) | 0.0000 | 0.1358 | (0.1333, 0.1396) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 0.0265 | (-0.0462, 0.1179) | 0.2877 | 0.0265 | (0.0190, 0.0315) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | quest_state_correctness | 0.1335 | (0.0463, 0.2170) | 0.0000 | 0.1335 | (0.1265, 0.1440) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | lore_consistency | 0.1040 | (0.0312, 0.1768) | 0.0030 | 0.1040 | (0.0541, 0.1373) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | objective_completion_support | 0.0377 | (-0.0573, 0.1327) | 0.2407 | 0.0377 | (0.0128, 0.0750) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | gameplay_usefulness | 0.0835 | (0.0472, 0.1141) | 0.0000 | 0.0835 | (0.0770, 0.0931) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | time_pressure_acceptability | 0.0274 | (-0.0217, 0.0765) | 0.1377 | 0.0274 | (0.0229, 0.0303) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.2236 | (0.0782, 0.3564) | 0.0003 | 0.2236 | (0.2212, 0.2273) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0417 | (0.0074, 0.0780) | 0.0100 | 0.0417 | (0.0237, 0.0688) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1667 | (0.1667, 0.1667) | 0.0000 | 0.1667 | (0.1667, 0.1667) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0125 | (0.0000, 0.0375) | 0.3337 | 0.0125 | (0.0000, 0.0312) | 0.2400 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0296 | (-0.0838, 0.0242) | 0.8123 | -0.0296 | (-0.0691, -0.0033) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.2267 | (-0.0867, 0.6000) | 0.1107 | 0.2267 | (0.2222, 0.2333) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | -0.0700 | (-0.2800, 0.1400) | 0.8097 | -0.0700 | (-0.1167, 0.0000) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1348 | (0.0843, 0.1863) | 0.0000 | 0.1348 | (0.1323, 0.1386) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.1291 | (0.0523, 0.1982) | 0.0000 | 0.1291 | (0.1116, 0.1408) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1076 | (0.0533, 0.1363) | 0.0003 | 0.1076 | (0.0667, 0.1350) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | -0.0147 | (-0.0644, 0.0402) | 0.7507 | -0.0147 | (-0.0550, 0.0457) | 0.7540 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.1009 | (0.0419, 0.1575) | 0.0000 | 0.1009 | (0.0849, 0.1116) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | lore_consistency | 0.0850 | (0.0179, 0.1521) | 0.0027 | 0.0850 | (0.0229, 0.1264) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.0520 | (-0.0397, 0.1437) | 0.1293 | 0.0520 | (0.0183, 0.1025) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 0.0635 | (0.0106, 0.1104) | 0.0107 | 0.0635 | (0.0303, 0.1135) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 0.0314 | (-0.0172, 0.0800) | 0.1063 | 0.0314 | (0.0030, 0.0739) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.1673 | (0.0727, 0.2582) | 0.0007 | 0.1673 | (0.1364, 0.1879) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0401 | (0.0120, 0.0716) | 0.0000 | 0.0401 | (0.0308, 0.0540) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1333 | (0.0667, 0.1667) | 0.0000 | 0.1333 | (0.0833, 0.1667) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | 0.0049 | (0.0000, 0.0146) | 0.3463 | 0.0049 | (0.0000, 0.0081) | 0.2463 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0452 | (-0.0917, 0.0009) | 0.9717 | -0.0452 | (-0.0858, -0.0181) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | 0.0867 | (-0.2000, 0.3868) | 0.3117 | 0.0867 | (-0.1222, 0.4000) | 0.2487 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | -0.1400 | (-0.3500, 0.1400) | 0.9047 | -0.1400 | (-0.2333, 0.0000) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.0972 | (0.0712, 0.1218) | 0.0000 | 0.0972 | (0.0852, 0.1052) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 4 | 1 | 0 | 0.8000 | 0.8000 |
| proposed_vs_candidate_no_context | persona_consistency | 2 | 1 | 2 | 0.6000 | 0.6667 |
| proposed_vs_candidate_no_context | naturalness | 1 | 4 | 0 | 0.2000 | 0.2000 |
| proposed_vs_candidate_no_context | quest_state_correctness | 4 | 1 | 0 | 0.8000 | 0.8000 |
| proposed_vs_candidate_no_context | lore_consistency | 5 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 5 | 0.5000 | nan |
| proposed_vs_candidate_no_context | objective_completion_support | 2 | 3 | 0 | 0.4000 | 0.4000 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 2 | 3 | 0 | 0.4000 | 0.4000 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | 1 | 4 | 0 | 0.2000 | 0.2000 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 4 | 0 | 1 | 0.9000 | 1.0000 |
| proposed_vs_candidate_no_context | context_overlap | 4 | 1 | 0 | 0.8000 | 0.8000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 2 | 1 | 2 | 0.6000 | 0.6667 |
| proposed_vs_candidate_no_context | persona_style | 0 | 0 | 5 | 0.5000 | nan |
| proposed_vs_candidate_no_context | distinct1 | 2 | 3 | 0 | 0.4000 | 0.4000 |
| proposed_vs_candidate_no_context | length_score | 1 | 4 | 0 | 0.2000 | 0.2000 |
| proposed_vs_candidate_no_context | sentence_score | 0 | 2 | 3 | 0.3000 | 0.0000 |
| proposed_vs_candidate_no_context | overall_quality | 4 | 1 | 0 | 0.8000 | 0.8000 |
| proposed_vs_baseline_no_context | context_relevance | 5 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_vs_baseline_no_context | persona_consistency | 4 | 0 | 1 | 0.9000 | 1.0000 |
| proposed_vs_baseline_no_context | naturalness | 2 | 1 | 2 | 0.6000 | 0.6667 |
| proposed_vs_baseline_no_context | quest_state_correctness | 5 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_vs_baseline_no_context | lore_consistency | 5 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_vs_baseline_no_context | multi_turn_contradiction_safety | 0 | 0 | 5 | 0.5000 | nan |
| proposed_vs_baseline_no_context | objective_completion_support | 3 | 1 | 1 | 0.7000 | 0.7500 |
| proposed_vs_baseline_no_context | gameplay_usefulness | 5 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_vs_baseline_no_context | time_pressure_acceptability | 2 | 2 | 1 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 4 | 0 | 1 | 0.9000 | 1.0000 |
| proposed_vs_baseline_no_context | context_overlap | 5 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 3 | 0 | 2 | 0.8000 | 1.0000 |
| proposed_vs_baseline_no_context | persona_style | 1 | 0 | 4 | 0.6000 | 1.0000 |
| proposed_vs_baseline_no_context | distinct1 | 0 | 3 | 2 | 0.2000 | 0.0000 |
| proposed_vs_baseline_no_context | length_score | 1 | 1 | 3 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context | sentence_score | 1 | 1 | 3 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context | overall_quality | 5 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 5 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | 4 | 1 | 0 | 0.8000 | 0.8000 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | 1 | 4 | 0 | 0.2000 | 0.2000 |
| proposed_vs_baseline_no_context_phi3_latest | quest_state_correctness | 5 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | lore_consistency | 5 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0 | 0 | 5 | 0.5000 | nan |
| proposed_vs_baseline_no_context_phi3_latest | objective_completion_support | 3 | 1 | 1 | 0.7000 | 0.7500 |
| proposed_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 3 | 2 | 0 | 0.6000 | 0.6000 |
| proposed_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 2 | 2 | 1 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 4 | 0 | 1 | 0.9000 | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 5 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 3 | 1 | 1 | 0.7000 | 0.7500 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | 1 | 0 | 4 | 0.6000 | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | 0 | 4 | 1 | 0.1000 | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | 1 | 4 | 0 | 0.2000 | 0.2000 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | 0 | 1 | 4 | 0.4000 | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | 5 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_proposed_raw | context_relevance | 2 | 3 | 0 | 0.4000 | 0.4000 |
| controlled_vs_proposed_raw | persona_consistency | 2 | 1 | 2 | 0.6000 | 0.6667 |
| controlled_vs_proposed_raw | naturalness | 3 | 2 | 0 | 0.6000 | 0.6000 |
| controlled_vs_proposed_raw | quest_state_correctness | 1 | 4 | 0 | 0.2000 | 0.2000 |
| controlled_vs_proposed_raw | lore_consistency | 2 | 3 | 0 | 0.4000 | 0.4000 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0 | 0 | 5 | 0.5000 | nan |
| controlled_vs_proposed_raw | objective_completion_support | 2 | 2 | 1 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | gameplay_usefulness | 3 | 2 | 0 | 0.6000 | 0.6000 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 3 | 1 | 1 | 0.7000 | 0.7500 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0 | 2 | 3 | 0.3000 | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 3 | 2 | 0 | 0.6000 | 0.6000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 2 | 1 | 2 | 0.6000 | 0.6667 |
| controlled_vs_proposed_raw | persona_style | 0 | 0 | 5 | 0.5000 | nan |
| controlled_vs_proposed_raw | distinct1 | 3 | 2 | 0 | 0.6000 | 0.6000 |
| controlled_vs_proposed_raw | length_score | 3 | 2 | 0 | 0.6000 | 0.6000 |
| controlled_vs_proposed_raw | sentence_score | 1 | 2 | 2 | 0.4000 | 0.3333 |
| controlled_vs_proposed_raw | overall_quality | 3 | 2 | 0 | 0.6000 | 0.6000 |
| controlled_vs_candidate_no_context | context_relevance | 3 | 1 | 1 | 0.7000 | 0.7500 |
| controlled_vs_candidate_no_context | persona_consistency | 2 | 0 | 3 | 0.7000 | 1.0000 |
| controlled_vs_candidate_no_context | naturalness | 2 | 2 | 1 | 0.5000 | 0.5000 |
| controlled_vs_candidate_no_context | quest_state_correctness | 3 | 1 | 1 | 0.7000 | 0.7500 |
| controlled_vs_candidate_no_context | lore_consistency | 4 | 0 | 1 | 0.9000 | 1.0000 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 5 | 0.5000 | nan |
| controlled_vs_candidate_no_context | objective_completion_support | 0 | 2 | 3 | 0.3000 | 0.0000 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 3 | 1 | 1 | 0.7000 | 0.7500 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 0 | 3 | 2 | 0.2000 | 0.0000 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 3 | 0 | 2 | 0.8000 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 3 | 1 | 1 | 0.7000 | 0.7500 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 2 | 0 | 3 | 0.7000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_style | 0 | 0 | 5 | 0.5000 | nan |
| controlled_vs_candidate_no_context | distinct1 | 1 | 2 | 2 | 0.4000 | 0.3333 |
| controlled_vs_candidate_no_context | length_score | 2 | 2 | 1 | 0.5000 | 0.5000 |
| controlled_vs_candidate_no_context | sentence_score | 0 | 3 | 2 | 0.2000 | 0.0000 |
| controlled_vs_candidate_no_context | overall_quality | 4 | 0 | 1 | 0.9000 | 1.0000 |
| controlled_vs_baseline_no_context | context_relevance | 5 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 5 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | naturalness | 2 | 3 | 0 | 0.4000 | 0.4000 |
| controlled_vs_baseline_no_context | quest_state_correctness | 5 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | lore_consistency | 4 | 1 | 0 | 0.8000 | 0.8000 |
| controlled_vs_baseline_no_context | multi_turn_contradiction_safety | 0 | 0 | 5 | 0.5000 | nan |
| controlled_vs_baseline_no_context | objective_completion_support | 2 | 3 | 0 | 0.4000 | 0.4000 |
| controlled_vs_baseline_no_context | gameplay_usefulness | 5 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | time_pressure_acceptability | 3 | 2 | 0 | 0.6000 | 0.6000 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 4 | 0 | 1 | 0.9000 | 1.0000 |
| controlled_vs_baseline_no_context | context_overlap | 4 | 1 | 0 | 0.8000 | 0.8000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 5 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_style | 1 | 0 | 4 | 0.6000 | 1.0000 |
| controlled_vs_baseline_no_context | distinct1 | 2 | 3 | 0 | 0.4000 | 0.4000 |
| controlled_vs_baseline_no_context | length_score | 3 | 2 | 0 | 0.6000 | 0.6000 |
| controlled_vs_baseline_no_context | sentence_score | 1 | 2 | 2 | 0.4000 | 0.3333 |
| controlled_vs_baseline_no_context | overall_quality | 5 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 5 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 4 | 0 | 1 | 0.9000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | 2 | 3 | 0 | 0.4000 | 0.4000 |
| controlled_vs_baseline_no_context_phi3_latest | quest_state_correctness | 5 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | lore_consistency | 4 | 1 | 0 | 0.8000 | 0.8000 |
| controlled_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0 | 0 | 5 | 0.5000 | nan |
| controlled_vs_baseline_no_context_phi3_latest | objective_completion_support | 3 | 2 | 0 | 0.6000 | 0.6000 |
| controlled_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 4 | 1 | 0 | 0.8000 | 0.8000 |
| controlled_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 3 | 2 | 0 | 0.6000 | 0.6000 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 4 | 0 | 1 | 0.9000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 5 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 4 | 0 | 1 | 0.9000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | 1 | 0 | 4 | 0.6000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | 1 | 4 | 0 | 0.2000 | 0.2000 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | 3 | 1 | 1 | 0.7000 | 0.7500 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 1 | 3 | 1 | 0.3000 | 0.2500 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 5 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 5 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 5 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 2 | 3 | 0 | 0.4000 | 0.4000 |
| proposed_contextual_controlled_vs_baseline_no_context | quest_state_correctness | 5 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | lore_consistency | 4 | 1 | 0 | 0.8000 | 0.8000 |
| proposed_contextual_controlled_vs_baseline_no_context | multi_turn_contradiction_safety | 0 | 0 | 5 | 0.5000 | nan |
| proposed_contextual_controlled_vs_baseline_no_context | objective_completion_support | 2 | 3 | 0 | 0.4000 | 0.4000 |
| proposed_contextual_controlled_vs_baseline_no_context | gameplay_usefulness | 5 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | time_pressure_acceptability | 3 | 2 | 0 | 0.6000 | 0.6000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 4 | 0 | 1 | 0.9000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 4 | 1 | 0 | 0.8000 | 0.8000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 5 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 1 | 0 | 4 | 0.6000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | 2 | 3 | 0 | 0.4000 | 0.4000 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 3 | 2 | 0 | 0.6000 | 0.6000 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 1 | 2 | 2 | 0.4000 | 0.3333 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 5 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 5 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 4 | 0 | 1 | 0.9000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | 2 | 3 | 0 | 0.4000 | 0.4000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | quest_state_correctness | 5 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | lore_consistency | 4 | 1 | 0 | 0.8000 | 0.8000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0 | 0 | 5 | 0.5000 | nan |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | objective_completion_support | 3 | 2 | 0 | 0.6000 | 0.6000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 4 | 1 | 0 | 0.8000 | 0.8000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 3 | 2 | 0 | 0.6000 | 0.6000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 4 | 0 | 1 | 0.9000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 5 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 4 | 0 | 1 | 0.9000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | 1 | 0 | 4 | 0.6000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | 1 | 4 | 0 | 0.2000 | 0.2000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | 3 | 1 | 1 | 0.7000 | 0.7500 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | 1 | 3 | 1 | 0.3000 | 0.2500 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 5 | 0 | 0 | 1.0000 | 1.0000 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.2000 | 0.6000 | 0.4000 |
| proposed_contextual | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.6000 | 0.0000 | 0.0000 |
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
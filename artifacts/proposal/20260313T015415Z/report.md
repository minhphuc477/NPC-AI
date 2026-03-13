# Proposal Alignment Evaluation Report

- Run ID: `20260313T015415Z`
- Generated: `2026-03-13T01:56:10.830860+00:00`
- Scenarios: `artifacts\proposal\20260313T015415Z\scenarios.jsonl`
- Scenario count: `144`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_tuned`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off
- `baseline_no_context`: model `phi3:mini`, dynamic_context=off, response_control=off
- `baseline_no_context_phi3_latest`: model `phi3:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.1198 (0.0974, 0.1444) | 0.2881 (0.2676, 0.3101) | 0.8681 (0.8601, 0.8759) | 0.3280 (0.3143, 0.3418) | n/a |
| proposed_contextual_controlled_tuned | 0.1254 (0.1005, 0.1502) | 0.3001 (0.2754, 0.3275) | 0.8563 (0.8484, 0.8639) | 0.3327 (0.3158, 0.3500) | n/a |
| proposed_contextual | 0.0914 (0.0732, 0.1109) | 0.2191 (0.1988, 0.2414) | 0.8761 (0.8694, 0.8829) | 0.2907 (0.2784, 0.3035) | n/a |
| candidate_no_context | 0.0306 (0.0244, 0.0372) | 0.2131 (0.1918, 0.2344) | 0.8772 (0.8707, 0.8845) | 0.2607 (0.2516, 0.2708) | n/a |
| baseline_no_context | 0.0377 (0.0305, 0.0453) | 0.1578 (0.1422, 0.1741) | 0.8926 (0.8839, 0.9010) | 0.2460 (0.2387, 0.2537) | n/a |
| baseline_no_context_phi3_latest | 0.0323 (0.0259, 0.0397) | 0.1589 (0.1437, 0.1759) | 0.8966 (0.8873, 0.9054) | 0.2446 (0.2376, 0.2520) | n/a |

## Game-facing Outcome Metrics (mean, 95% CI)
| Arm | Quest-state Correctness | Lore Consistency | Contradiction Safety | Objective Completion Support | Gameplay Usefulness | Time-pressure Acceptability |
|---|---:|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2049 (0.1849, 0.2261) | 0.0593 (0.0430, 0.0771) | 1.0000 (1.0000, 1.0000) | 0.0917 (0.0784, 0.1040) | 0.3099 (0.3029, 0.3170) | 0.3004 (0.2912, 0.3098) |
| proposed_contextual_controlled_tuned | 0.2097 (0.1887, 0.2316) | 0.0632 (0.0443, 0.0837) | 1.0000 (1.0000, 1.0000) | 0.0890 (0.0772, 0.1011) | 0.3052 (0.2983, 0.3125) | 0.2957 (0.2858, 0.3058) |
| proposed_contextual | 0.1774 (0.1625, 0.1941) | 0.0455 (0.0323, 0.0605) | 0.8750 (0.8194, 0.9236) | 0.0720 (0.0612, 0.0835) | 0.2990 (0.2918, 0.3065) | 0.2959 (0.2887, 0.3035) |
| candidate_no_context | 0.1266 (0.1213, 0.1322) | 0.0060 (0.0036, 0.0088) | 1.0000 (1.0000, 1.0000) | 0.0716 (0.0598, 0.0841) | 0.2824 (0.2761, 0.2889) | 0.3014 (0.2931, 0.3102) |
| baseline_no_context | 0.1323 (0.1260, 0.1388) | 0.0156 (0.0109, 0.0208) | 1.0000 (1.0000, 1.0000) | 0.0453 (0.0372, 0.0545) | 0.2785 (0.2727, 0.2845) | 0.2911 (0.2851, 0.2975) |
| baseline_no_context_phi3_latest | 0.1270 (0.1211, 0.1335) | 0.0165 (0.0121, 0.0215) | 1.0000 (1.0000, 1.0000) | 0.0397 (0.0332, 0.0470) | 0.2780 (0.2720, 0.2839) | 0.2893 (0.2840, 0.2952) |

- Multi-turn contradiction rate is reported as `1 - contradiction_safety` in row-level outputs.

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0608 | 1.9838 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0059 | 0.0279 |
| proposed_vs_candidate_no_context | naturalness | -0.0011 | -0.0013 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.0508 | 0.4016 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0395 | 6.5675 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | -0.1250 | -0.1250 |
| proposed_vs_candidate_no_context | objective_completion_support | 0.0004 | 0.0052 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 0.0165 | 0.0586 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | -0.0055 | -0.0183 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0786 | 2.6962 |
| proposed_vs_candidate_no_context | context_overlap | 0.0191 | 0.5600 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0038 | 0.0289 |
| proposed_vs_candidate_no_context | persona_style | 0.0144 | 0.0268 |
| proposed_vs_candidate_no_context | distinct1 | -0.0024 | -0.0026 |
| proposed_vs_candidate_no_context | length_score | 0.0032 | 0.0061 |
| proposed_vs_candidate_no_context | sentence_score | -0.0146 | -0.0153 |
| proposed_vs_candidate_no_context | overall_quality | 0.0300 | 0.1149 |
| proposed_vs_baseline_no_context | context_relevance | 0.0537 | 1.4252 |
| proposed_vs_baseline_no_context | persona_consistency | 0.0613 | 0.3887 |
| proposed_vs_baseline_no_context | naturalness | -0.0165 | -0.0185 |
| proposed_vs_baseline_no_context | quest_state_correctness | 0.0451 | 0.3412 |
| proposed_vs_baseline_no_context | lore_consistency | 0.0299 | 1.9199 |
| proposed_vs_baseline_no_context | multi_turn_contradiction_safety | -0.1250 | -0.1250 |
| proposed_vs_baseline_no_context | objective_completion_support | 0.0266 | 0.5871 |
| proposed_vs_baseline_no_context | gameplay_usefulness | 0.0205 | 0.0735 |
| proposed_vs_baseline_no_context | time_pressure_acceptability | 0.0048 | 0.0163 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0697 | 1.8257 |
| proposed_vs_baseline_no_context | context_overlap | 0.0165 | 0.4512 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 0.0784 | 1.3485 |
| proposed_vs_baseline_no_context | persona_style | -0.0072 | -0.0129 |
| proposed_vs_baseline_no_context | distinct1 | -0.0394 | -0.0402 |
| proposed_vs_baseline_no_context | length_score | -0.0271 | -0.0484 |
| proposed_vs_baseline_no_context | sentence_score | 0.0462 | 0.0517 |
| proposed_vs_baseline_no_context | overall_quality | 0.0447 | 0.1817 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 0.0591 | 1.8278 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | 0.0602 | 0.3792 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | -0.0205 | -0.0229 |
| proposed_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.0504 | 0.3971 |
| proposed_vs_baseline_no_context_phi3_latest | lore_consistency | 0.0290 | 1.7623 |
| proposed_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | -0.1250 | -0.1250 |
| proposed_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.0322 | 0.8108 |
| proposed_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 0.0210 | 0.0755 |
| proposed_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 0.0066 | 0.0228 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.0775 | 2.5626 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 0.0160 | 0.4307 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.0776 | 1.3169 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | -0.0094 | -0.0169 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | -0.0407 | -0.0415 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | -0.0525 | -0.0898 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | 0.0628 | 0.0717 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | 0.0461 | 0.1885 |
| controlled_vs_proposed_raw | context_relevance | 0.0284 | 0.3108 |
| controlled_vs_proposed_raw | persona_consistency | 0.0691 | 0.3152 |
| controlled_vs_proposed_raw | naturalness | -0.0080 | -0.0091 |
| controlled_vs_proposed_raw | quest_state_correctness | 0.0274 | 0.1547 |
| controlled_vs_proposed_raw | lore_consistency | 0.0138 | 0.3033 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0.1250 | 0.1429 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0197 | 0.2740 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0109 | 0.0366 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 0.0046 | 0.0155 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.0370 | 0.3428 |
| controlled_vs_proposed_raw | context_overlap | 0.0085 | 0.1595 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.0839 | 0.6139 |
| controlled_vs_proposed_raw | persona_style | 0.0098 | 0.0179 |
| controlled_vs_proposed_raw | distinct1 | -0.0059 | -0.0062 |
| controlled_vs_proposed_raw | length_score | -0.0465 | -0.0874 |
| controlled_vs_proposed_raw | sentence_score | 0.0365 | 0.0388 |
| controlled_vs_proposed_raw | overall_quality | 0.0373 | 0.1284 |
| controlled_vs_candidate_no_context | context_relevance | 0.0892 | 2.9112 |
| controlled_vs_candidate_no_context | persona_consistency | 0.0750 | 0.3518 |
| controlled_vs_candidate_no_context | naturalness | -0.0091 | -0.0104 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.0783 | 0.6184 |
| controlled_vs_candidate_no_context | lore_consistency | 0.0533 | 8.8630 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | 0.0201 | 0.2806 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0275 | 0.0973 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | -0.0009 | -0.0031 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.1156 | 3.9632 |
| controlled_vs_candidate_no_context | context_overlap | 0.0275 | 0.8089 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.0877 | 0.6605 |
| controlled_vs_candidate_no_context | persona_style | 0.0242 | 0.0452 |
| controlled_vs_candidate_no_context | distinct1 | -0.0083 | -0.0088 |
| controlled_vs_candidate_no_context | length_score | -0.0433 | -0.0818 |
| controlled_vs_candidate_no_context | sentence_score | 0.0219 | 0.0229 |
| controlled_vs_candidate_no_context | overall_quality | 0.0673 | 0.2581 |
| controlled_vs_baseline_no_context | context_relevance | 0.0821 | 2.1790 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1304 | 0.8263 |
| controlled_vs_baseline_no_context | naturalness | -0.0246 | -0.0275 |
| controlled_vs_baseline_no_context | quest_state_correctness | 0.0726 | 0.5487 |
| controlled_vs_baseline_no_context | lore_consistency | 0.0437 | 2.8056 |
| controlled_vs_baseline_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_baseline_no_context | objective_completion_support | 0.0463 | 1.0220 |
| controlled_vs_baseline_no_context | gameplay_usefulness | 0.0314 | 0.1127 |
| controlled_vs_baseline_no_context | time_pressure_acceptability | 0.0093 | 0.0320 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.1066 | 2.7943 |
| controlled_vs_baseline_no_context | context_overlap | 0.0250 | 0.6827 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1623 | 2.7902 |
| controlled_vs_baseline_no_context | persona_style | 0.0027 | 0.0048 |
| controlled_vs_baseline_no_context | distinct1 | -0.0452 | -0.0462 |
| controlled_vs_baseline_no_context | length_score | -0.0736 | -0.1316 |
| controlled_vs_baseline_no_context | sentence_score | 0.0826 | 0.0925 |
| controlled_vs_baseline_no_context | overall_quality | 0.0820 | 0.3335 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.0875 | 2.7068 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1293 | 0.8138 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | -0.0285 | -0.0318 |
| controlled_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.0779 | 0.6133 |
| controlled_vs_baseline_no_context_phi3_latest | lore_consistency | 0.0428 | 2.6002 |
| controlled_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.0519 | 1.3070 |
| controlled_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 0.0319 | 0.1149 |
| controlled_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 0.0112 | 0.0386 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.1145 | 3.7837 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0245 | 0.6589 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1615 | 2.7392 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | 0.0004 | 0.0007 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0465 | -0.0475 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | -0.0991 | -0.1694 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.0993 | 0.1133 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.0834 | 0.3411 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0056 | 0.0466 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0120 | 0.0416 |
| controlled_alt_vs_controlled_default | naturalness | -0.0118 | -0.0135 |
| controlled_alt_vs_controlled_default | quest_state_correctness | 0.0048 | 0.0236 |
| controlled_alt_vs_controlled_default | lore_consistency | 0.0039 | 0.0655 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_controlled_default | objective_completion_support | -0.0027 | -0.0295 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | -0.0047 | -0.0151 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | -0.0047 | -0.0157 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0070 | 0.0483 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0023 | 0.0374 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0156 | 0.0706 |
| controlled_alt_vs_controlled_default | persona_style | -0.0024 | -0.0042 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0064 | -0.0069 |
| controlled_alt_vs_controlled_default | length_score | -0.0419 | -0.0862 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0021 | -0.0021 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0047 | 0.0145 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.0340 | 0.3719 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.0810 | 0.3699 |
| controlled_alt_vs_proposed_raw | naturalness | -0.0198 | -0.0226 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 0.0323 | 0.1820 |
| controlled_alt_vs_proposed_raw | lore_consistency | 0.0177 | 0.3887 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0.1250 | 0.1429 |
| controlled_alt_vs_proposed_raw | objective_completion_support | 0.0170 | 0.2364 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 0.0062 | 0.0209 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | -0.0001 | -0.0005 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.0439 | 0.4076 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0108 | 0.2029 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.0994 | 0.7279 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0075 | 0.0136 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0123 | -0.0131 |
| controlled_alt_vs_proposed_raw | length_score | -0.0884 | -0.1661 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0344 | 0.0366 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.0421 | 0.1448 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.0948 | 3.0934 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.0870 | 0.4081 |
| controlled_alt_vs_candidate_no_context | naturalness | -0.0209 | -0.0238 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 0.0831 | 0.6567 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 0.0572 | 9.5089 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_candidate_no_context | objective_completion_support | 0.0174 | 0.2428 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 0.0228 | 0.0807 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | -0.0056 | -0.0187 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.1226 | 4.2027 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0298 | 0.8765 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1033 | 0.7778 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0218 | 0.0408 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0147 | -0.0156 |
| controlled_alt_vs_candidate_no_context | length_score | -0.0852 | -0.1610 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.0198 | 0.0207 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.0720 | 0.2763 |
| controlled_alt_vs_baseline_no_context | context_relevance | 0.0877 | 2.3270 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 0.1424 | 0.9023 |
| controlled_alt_vs_baseline_no_context | naturalness | -0.0363 | -0.0407 |
| controlled_alt_vs_baseline_no_context | quest_state_correctness | 0.0774 | 0.5853 |
| controlled_alt_vs_baseline_no_context | lore_consistency | 0.0476 | 3.0548 |
| controlled_alt_vs_baseline_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_baseline_no_context | objective_completion_support | 0.0436 | 0.9622 |
| controlled_alt_vs_baseline_no_context | gameplay_usefulness | 0.0267 | 0.0959 |
| controlled_alt_vs_baseline_no_context | time_pressure_acceptability | 0.0046 | 0.0158 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 0.1136 | 2.9774 |
| controlled_alt_vs_baseline_no_context | context_overlap | 0.0273 | 0.7456 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 0.1779 | 3.0580 |
| controlled_alt_vs_baseline_no_context | persona_style | 0.0003 | 0.0005 |
| controlled_alt_vs_baseline_no_context | distinct1 | -0.0517 | -0.0528 |
| controlled_alt_vs_baseline_no_context | length_score | -0.1155 | -0.2065 |
| controlled_alt_vs_baseline_no_context | sentence_score | 0.0806 | 0.0902 |
| controlled_alt_vs_baseline_no_context | overall_quality | 0.0868 | 0.3528 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_relevance | 0.0931 | 2.8794 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1413 | 0.8893 |
| controlled_alt_vs_baseline_no_context_phi3_latest | naturalness | -0.0403 | -0.0449 |
| controlled_alt_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.0827 | 0.6514 |
| controlled_alt_vs_baseline_no_context_phi3_latest | lore_consistency | 0.0467 | 2.8360 |
| controlled_alt_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.0492 | 1.2389 |
| controlled_alt_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 0.0272 | 0.0980 |
| controlled_alt_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 0.0064 | 0.0223 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.1215 | 4.0146 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_overlap | 0.0268 | 0.7209 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1771 | 3.0034 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_style | -0.0020 | -0.0035 |
| controlled_alt_vs_baseline_no_context_phi3_latest | distinct1 | -0.0530 | -0.0541 |
| controlled_alt_vs_baseline_no_context_phi3_latest | length_score | -0.1410 | -0.2410 |
| controlled_alt_vs_baseline_no_context_phi3_latest | sentence_score | 0.0972 | 0.1109 |
| controlled_alt_vs_baseline_no_context_phi3_latest | overall_quality | 0.0882 | 0.3606 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | context_relevance | 0.0877 | 2.3270 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | persona_consistency | 0.1424 | 0.9023 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | naturalness | -0.0363 | -0.0407 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | quest_state_correctness | 0.0774 | 0.5853 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | lore_consistency | 0.0476 | 3.0548 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | objective_completion_support | 0.0436 | 0.9622 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | gameplay_usefulness | 0.0267 | 0.0959 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | time_pressure_acceptability | 0.0046 | 0.0158 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | context_keyword_coverage | 0.1136 | 2.9774 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | context_overlap | 0.0273 | 0.7456 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | persona_keyword_coverage | 0.1779 | 3.0580 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | persona_style | 0.0003 | 0.0005 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | distinct1 | -0.0517 | -0.0528 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | length_score | -0.1155 | -0.2065 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | sentence_score | 0.0806 | 0.0902 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | overall_quality | 0.0868 | 0.3528 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | context_relevance | 0.0931 | 2.8794 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1413 | 0.8893 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | naturalness | -0.0403 | -0.0449 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.0827 | 0.6514 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | lore_consistency | 0.0467 | 2.8360 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.0492 | 1.2389 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 0.0272 | 0.0980 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 0.0064 | 0.0223 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.1215 | 4.0146 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | context_overlap | 0.0268 | 0.7209 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1771 | 3.0034 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | persona_style | -0.0020 | -0.0035 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | distinct1 | -0.0530 | -0.0541 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | length_score | -0.1410 | -0.2410 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | sentence_score | 0.0972 | 0.1109 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | overall_quality | 0.0882 | 0.3606 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0608 | (0.0416, 0.0808) | 0.0000 | 0.0608 | (0.0287, 0.0953) | 0.0000 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0059 | (-0.0092, 0.0221) | 0.2327 | 0.0059 | (-0.0119, 0.0268) | 0.2740 |
| proposed_vs_candidate_no_context | naturalness | -0.0011 | (-0.0099, 0.0072) | 0.5867 | -0.0011 | (-0.0100, 0.0081) | 0.6013 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.0508 | (0.0349, 0.0674) | 0.0000 | 0.0508 | (0.0213, 0.0785) | 0.0003 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0395 | (0.0262, 0.0544) | 0.0000 | 0.0395 | (0.0169, 0.0612) | 0.0000 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | -0.1250 | (-0.1875, -0.0694) | 1.0000 | -0.1250 | (-0.3750, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | 0.0004 | (-0.0100, 0.0102) | 0.4700 | 0.0004 | (-0.0082, 0.0091) | 0.4650 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 0.0165 | (0.0085, 0.0254) | 0.0000 | 0.0165 | (0.0014, 0.0309) | 0.0157 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | -0.0055 | (-0.0128, 0.0017) | 0.9350 | -0.0055 | (-0.0102, 0.0020) | 0.9323 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0786 | (0.0540, 0.1049) | 0.0000 | 0.0786 | (0.0339, 0.1229) | 0.0000 |
| proposed_vs_candidate_no_context | context_overlap | 0.0191 | (0.0122, 0.0263) | 0.0000 | 0.0191 | (0.0087, 0.0278) | 0.0003 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0038 | (-0.0139, 0.0229) | 0.3437 | 0.0038 | (-0.0195, 0.0296) | 0.4047 |
| proposed_vs_candidate_no_context | persona_style | 0.0144 | (-0.0020, 0.0329) | 0.0437 | 0.0144 | (0.0000, 0.0286) | 0.0257 |
| proposed_vs_candidate_no_context | distinct1 | -0.0024 | (-0.0103, 0.0060) | 0.7353 | -0.0024 | (-0.0074, 0.0032) | 0.8123 |
| proposed_vs_candidate_no_context | length_score | 0.0032 | (-0.0306, 0.0352) | 0.4167 | 0.0032 | (-0.0370, 0.0424) | 0.4373 |
| proposed_vs_candidate_no_context | sentence_score | -0.0146 | (-0.0413, 0.0122) | 0.8760 | -0.0146 | (-0.0681, 0.0413) | 0.7090 |
| proposed_vs_candidate_no_context | overall_quality | 0.0300 | (0.0183, 0.0425) | 0.0000 | 0.0300 | (0.0064, 0.0535) | 0.0093 |
| proposed_vs_baseline_no_context | context_relevance | 0.0537 | (0.0353, 0.0724) | 0.0000 | 0.0537 | (0.0190, 0.0888) | 0.0010 |
| proposed_vs_baseline_no_context | persona_consistency | 0.0613 | (0.0382, 0.0843) | 0.0000 | 0.0613 | (0.0266, 0.0923) | 0.0007 |
| proposed_vs_baseline_no_context | naturalness | -0.0165 | (-0.0257, -0.0073) | 0.9997 | -0.0165 | (-0.0249, -0.0084) | 1.0000 |
| proposed_vs_baseline_no_context | quest_state_correctness | 0.0451 | (0.0310, 0.0608) | 0.0000 | 0.0451 | (0.0171, 0.0725) | 0.0000 |
| proposed_vs_baseline_no_context | lore_consistency | 0.0299 | (0.0164, 0.0454) | 0.0000 | 0.0299 | (0.0093, 0.0510) | 0.0017 |
| proposed_vs_baseline_no_context | multi_turn_contradiction_safety | -0.1250 | (-0.1806, -0.0764) | 1.0000 | -0.1250 | (-0.3750, 0.0000) | 1.0000 |
| proposed_vs_baseline_no_context | objective_completion_support | 0.0266 | (0.0132, 0.0394) | 0.0000 | 0.0266 | (0.0150, 0.0406) | 0.0000 |
| proposed_vs_baseline_no_context | gameplay_usefulness | 0.0205 | (0.0113, 0.0300) | 0.0000 | 0.0205 | (0.0105, 0.0303) | 0.0000 |
| proposed_vs_baseline_no_context | time_pressure_acceptability | 0.0048 | (-0.0041, 0.0133) | 0.1313 | 0.0048 | (-0.0068, 0.0164) | 0.2183 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0697 | (0.0461, 0.0940) | 0.0000 | 0.0697 | (0.0256, 0.1137) | 0.0013 |
| proposed_vs_baseline_no_context | context_overlap | 0.0165 | (0.0094, 0.0243) | 0.0000 | 0.0165 | (0.0041, 0.0275) | 0.0043 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 0.0784 | (0.0517, 0.1066) | 0.0000 | 0.0784 | (0.0331, 0.1177) | 0.0010 |
| proposed_vs_baseline_no_context | persona_style | -0.0072 | (-0.0244, 0.0096) | 0.7943 | -0.0072 | (-0.0389, 0.0207) | 0.6560 |
| proposed_vs_baseline_no_context | distinct1 | -0.0394 | (-0.0487, -0.0299) | 1.0000 | -0.0394 | (-0.0596, -0.0257) | 1.0000 |
| proposed_vs_baseline_no_context | length_score | -0.0271 | (-0.0662, 0.0107) | 0.9170 | -0.0271 | (-0.0549, -0.0028) | 0.9867 |
| proposed_vs_baseline_no_context | sentence_score | 0.0462 | (0.0145, 0.0778) | 0.0027 | 0.0462 | (-0.0049, 0.0826) | 0.0490 |
| proposed_vs_baseline_no_context | overall_quality | 0.0447 | (0.0315, 0.0579) | 0.0000 | 0.0447 | (0.0181, 0.0693) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 0.0591 | (0.0398, 0.0803) | 0.0000 | 0.0591 | (0.0232, 0.0947) | 0.0003 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | 0.0602 | (0.0371, 0.0840) | 0.0000 | 0.0602 | (0.0167, 0.1044) | 0.0030 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | -0.0205 | (-0.0308, -0.0100) | 1.0000 | -0.0205 | (-0.0378, -0.0027) | 0.9873 |
| proposed_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.0504 | (0.0347, 0.0676) | 0.0000 | 0.0504 | (0.0206, 0.0785) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | lore_consistency | 0.0290 | (0.0158, 0.0437) | 0.0000 | 0.0290 | (0.0071, 0.0530) | 0.0030 |
| proposed_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | -0.1250 | (-0.1806, -0.0694) | 1.0000 | -0.1250 | (-0.3750, 0.0000) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.0322 | (0.0208, 0.0440) | 0.0000 | 0.0322 | (0.0153, 0.0512) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 0.0210 | (0.0118, 0.0298) | 0.0000 | 0.0210 | (0.0073, 0.0345) | 0.0007 |
| proposed_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 0.0066 | (-0.0014, 0.0146) | 0.0510 | 0.0066 | (-0.0051, 0.0193) | 0.1533 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.0775 | (0.0534, 0.1031) | 0.0000 | 0.0775 | (0.0331, 0.1238) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 0.0160 | (0.0089, 0.0236) | 0.0000 | 0.0160 | (0.0015, 0.0293) | 0.0150 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.0776 | (0.0502, 0.1070) | 0.0000 | 0.0776 | (0.0239, 0.1271) | 0.0023 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | -0.0094 | (-0.0281, 0.0096) | 0.8387 | -0.0094 | (-0.0482, 0.0230) | 0.7203 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | -0.0407 | (-0.0496, -0.0318) | 1.0000 | -0.0407 | (-0.0575, -0.0286) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | -0.0525 | (-0.0977, -0.0079) | 0.9883 | -0.0525 | (-0.1160, 0.0095) | 0.9460 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | 0.0628 | (0.0292, 0.0986) | 0.0003 | 0.0628 | (0.0073, 0.1163) | 0.0187 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | 0.0461 | (0.0330, 0.0593) | 0.0000 | 0.0461 | (0.0164, 0.0741) | 0.0023 |
| controlled_vs_proposed_raw | context_relevance | 0.0284 | (0.0038, 0.0538) | 0.0140 | 0.0284 | (0.0019, 0.0590) | 0.0143 |
| controlled_vs_proposed_raw | persona_consistency | 0.0691 | (0.0443, 0.0936) | 0.0000 | 0.0691 | (0.0325, 0.1050) | 0.0003 |
| controlled_vs_proposed_raw | naturalness | -0.0080 | (-0.0166, 0.0008) | 0.9627 | -0.0080 | (-0.0206, 0.0035) | 0.9060 |
| controlled_vs_proposed_raw | quest_state_correctness | 0.0274 | (0.0059, 0.0495) | 0.0053 | 0.0274 | (0.0033, 0.0549) | 0.0083 |
| controlled_vs_proposed_raw | lore_consistency | 0.0138 | (-0.0066, 0.0330) | 0.0873 | 0.0138 | (-0.0089, 0.0371) | 0.1200 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0.1250 | (0.0764, 0.1806) | 0.0000 | 0.1250 | (0.0000, 0.3750) | 0.3390 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0197 | (0.0079, 0.0315) | 0.0007 | 0.0197 | (0.0076, 0.0318) | 0.0003 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0109 | (0.0032, 0.0192) | 0.0030 | 0.0109 | (0.0031, 0.0192) | 0.0027 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 0.0046 | (-0.0034, 0.0128) | 0.1317 | 0.0046 | (-0.0046, 0.0135) | 0.1550 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.0370 | (0.0052, 0.0706) | 0.0103 | 0.0370 | (0.0025, 0.0766) | 0.0163 |
| controlled_vs_proposed_raw | context_overlap | 0.0085 | (0.0001, 0.0171) | 0.0233 | 0.0085 | (-0.0013, 0.0176) | 0.0423 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.0839 | (0.0544, 0.1141) | 0.0000 | 0.0839 | (0.0425, 0.1270) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.0098 | (-0.0087, 0.0296) | 0.1577 | 0.0098 | (-0.0091, 0.0329) | 0.1767 |
| controlled_vs_proposed_raw | distinct1 | -0.0059 | (-0.0139, 0.0025) | 0.9190 | -0.0059 | (-0.0120, 0.0024) | 0.9333 |
| controlled_vs_proposed_raw | length_score | -0.0465 | (-0.0836, -0.0090) | 0.9927 | -0.0465 | (-0.1012, 0.0007) | 0.9737 |
| controlled_vs_proposed_raw | sentence_score | 0.0365 | (0.0097, 0.0632) | 0.0030 | 0.0365 | (0.0097, 0.0656) | 0.0017 |
| controlled_vs_proposed_raw | overall_quality | 0.0373 | (0.0230, 0.0520) | 0.0000 | 0.0373 | (0.0224, 0.0534) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.0892 | (0.0667, 0.1128) | 0.0000 | 0.0892 | (0.0574, 0.1218) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.0750 | (0.0502, 0.0988) | 0.0000 | 0.0750 | (0.0348, 0.1123) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | -0.0091 | (-0.0186, 0.0006) | 0.9677 | -0.0091 | (-0.0203, 0.0013) | 0.9527 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.0783 | (0.0586, 0.0995) | 0.0000 | 0.0783 | (0.0507, 0.1044) | 0.0000 |
| controlled_vs_candidate_no_context | lore_consistency | 0.0533 | (0.0368, 0.0709) | 0.0000 | 0.0533 | (0.0339, 0.0755) | 0.0000 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | 0.0201 | (0.0075, 0.0326) | 0.0010 | 0.0201 | (0.0066, 0.0336) | 0.0013 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0275 | (0.0199, 0.0352) | 0.0000 | 0.0275 | (0.0168, 0.0375) | 0.0000 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | -0.0009 | (-0.0103, 0.0080) | 0.5933 | -0.0009 | (-0.0094, 0.0068) | 0.5770 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.1156 | (0.0859, 0.1459) | 0.0000 | 0.1156 | (0.0728, 0.1598) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0275 | (0.0205, 0.0348) | 0.0000 | 0.0275 | (0.0193, 0.0361) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.0877 | (0.0571, 0.1168) | 0.0000 | 0.0877 | (0.0448, 0.1301) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.0242 | (0.0059, 0.0444) | 0.0050 | 0.0242 | (-0.0039, 0.0557) | 0.0953 |
| controlled_vs_candidate_no_context | distinct1 | -0.0083 | (-0.0180, 0.0020) | 0.9480 | -0.0083 | (-0.0155, -0.0005) | 0.9833 |
| controlled_vs_candidate_no_context | length_score | -0.0433 | (-0.0836, -0.0035) | 0.9837 | -0.0433 | (-0.0863, -0.0046) | 0.9823 |
| controlled_vs_candidate_no_context | sentence_score | 0.0219 | (-0.0024, 0.0462) | 0.0457 | 0.0219 | (-0.0122, 0.0559) | 0.1227 |
| controlled_vs_candidate_no_context | overall_quality | 0.0673 | (0.0520, 0.0827) | 0.0000 | 0.0673 | (0.0425, 0.0910) | 0.0000 |
| controlled_vs_baseline_no_context | context_relevance | 0.0821 | (0.0597, 0.1058) | 0.0000 | 0.0821 | (0.0534, 0.1093) | 0.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1304 | (0.1070, 0.1541) | 0.0000 | 0.1304 | (0.0784, 0.1778) | 0.0000 |
| controlled_vs_baseline_no_context | naturalness | -0.0246 | (-0.0350, -0.0138) | 1.0000 | -0.0246 | (-0.0389, -0.0126) | 1.0000 |
| controlled_vs_baseline_no_context | quest_state_correctness | 0.0726 | (0.0528, 0.0931) | 0.0000 | 0.0726 | (0.0466, 0.0976) | 0.0000 |
| controlled_vs_baseline_no_context | lore_consistency | 0.0437 | (0.0279, 0.0610) | 0.0000 | 0.0437 | (0.0239, 0.0650) | 0.0000 |
| controlled_vs_baseline_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_baseline_no_context | objective_completion_support | 0.0463 | (0.0343, 0.0588) | 0.0000 | 0.0463 | (0.0296, 0.0641) | 0.0000 |
| controlled_vs_baseline_no_context | gameplay_usefulness | 0.0314 | (0.0222, 0.0403) | 0.0000 | 0.0314 | (0.0237, 0.0384) | 0.0000 |
| controlled_vs_baseline_no_context | time_pressure_acceptability | 0.0093 | (-0.0004, 0.0188) | 0.0317 | 0.0093 | (-0.0061, 0.0249) | 0.1103 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.1066 | (0.0769, 0.1377) | 0.0000 | 0.1066 | (0.0688, 0.1434) | 0.0000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0250 | (0.0172, 0.0329) | 0.0000 | 0.0250 | (0.0148, 0.0339) | 0.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1623 | (0.1341, 0.1914) | 0.0000 | 0.1623 | (0.1013, 0.2204) | 0.0000 |
| controlled_vs_baseline_no_context | persona_style | 0.0027 | (-0.0141, 0.0189) | 0.3837 | 0.0027 | (-0.0248, 0.0310) | 0.4340 |
| controlled_vs_baseline_no_context | distinct1 | -0.0452 | (-0.0539, -0.0364) | 1.0000 | -0.0452 | (-0.0576, -0.0356) | 1.0000 |
| controlled_vs_baseline_no_context | length_score | -0.0736 | (-0.1225, -0.0257) | 0.9983 | -0.0736 | (-0.1368, -0.0215) | 0.9970 |
| controlled_vs_baseline_no_context | sentence_score | 0.0826 | (0.0534, 0.1118) | 0.0000 | 0.0826 | (0.0413, 0.1240) | 0.0000 |
| controlled_vs_baseline_no_context | overall_quality | 0.0820 | (0.0674, 0.0964) | 0.0000 | 0.0820 | (0.0560, 0.1057) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.0875 | (0.0646, 0.1113) | 0.0000 | 0.0875 | (0.0585, 0.1169) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1293 | (0.1054, 0.1540) | 0.0000 | 0.1293 | (0.0776, 0.1701) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | -0.0285 | (-0.0411, -0.0167) | 1.0000 | -0.0285 | (-0.0501, -0.0103) | 0.9993 |
| controlled_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.0779 | (0.0577, 0.0987) | 0.0000 | 0.0779 | (0.0516, 0.1036) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | lore_consistency | 0.0428 | (0.0261, 0.0602) | 0.0000 | 0.0428 | (0.0237, 0.0644) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.0519 | (0.0401, 0.0638) | 0.0000 | 0.0519 | (0.0342, 0.0721) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 0.0319 | (0.0235, 0.0404) | 0.0000 | 0.0319 | (0.0192, 0.0429) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 0.0112 | (0.0019, 0.0207) | 0.0080 | 0.0112 | (-0.0047, 0.0280) | 0.0810 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.1145 | (0.0833, 0.1454) | 0.0000 | 0.1145 | (0.0755, 0.1526) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0245 | (0.0169, 0.0326) | 0.0000 | 0.0245 | (0.0148, 0.0335) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1615 | (0.1317, 0.1892) | 0.0000 | 0.1615 | (0.0930, 0.2121) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | 0.0004 | (-0.0154, 0.0172) | 0.4947 | 0.0004 | (-0.0250, 0.0276) | 0.4833 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0465 | (-0.0547, -0.0379) | 1.0000 | -0.0465 | (-0.0557, -0.0387) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | -0.0991 | (-0.1537, -0.0449) | 1.0000 | -0.0991 | (-0.1935, -0.0192) | 0.9973 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.0993 | (0.0677, 0.1312) | 0.0000 | 0.0993 | (0.0486, 0.1479) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.0834 | (0.0683, 0.0989) | 0.0000 | 0.0834 | (0.0555, 0.1069) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0056 | (-0.0244, 0.0373) | 0.3483 | 0.0056 | (-0.0207, 0.0342) | 0.3590 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0120 | (-0.0141, 0.0384) | 0.1830 | 0.0120 | (-0.0161, 0.0384) | 0.2003 |
| controlled_alt_vs_controlled_default | naturalness | -0.0118 | (-0.0225, -0.0021) | 0.9933 | -0.0118 | (-0.0233, -0.0018) | 0.9933 |
| controlled_alt_vs_controlled_default | quest_state_correctness | 0.0048 | (-0.0213, 0.0326) | 0.3617 | 0.0048 | (-0.0183, 0.0283) | 0.3597 |
| controlled_alt_vs_controlled_default | lore_consistency | 0.0039 | (-0.0203, 0.0299) | 0.3810 | 0.0039 | (-0.0188, 0.0283) | 0.3903 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | objective_completion_support | -0.0027 | (-0.0130, 0.0075) | 0.6937 | -0.0027 | (-0.0108, 0.0049) | 0.7570 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | -0.0047 | (-0.0118, 0.0029) | 0.8933 | -0.0047 | (-0.0145, 0.0045) | 0.8207 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | -0.0047 | (-0.0139, 0.0038) | 0.8443 | -0.0047 | (-0.0128, 0.0017) | 0.9040 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0070 | (-0.0330, 0.0477) | 0.3693 | 0.0070 | (-0.0266, 0.0446) | 0.3653 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0023 | (-0.0093, 0.0144) | 0.3513 | 0.0023 | (-0.0091, 0.0136) | 0.3400 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0156 | (-0.0146, 0.0459) | 0.1620 | 0.0156 | (-0.0215, 0.0478) | 0.1910 |
| controlled_alt_vs_controlled_default | persona_style | -0.0024 | (-0.0202, 0.0136) | 0.5910 | -0.0024 | (-0.0105, 0.0080) | 0.7163 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0064 | (-0.0156, 0.0022) | 0.9333 | -0.0064 | (-0.0154, 0.0026) | 0.9203 |
| controlled_alt_vs_controlled_default | length_score | -0.0419 | (-0.0829, -0.0039) | 0.9847 | -0.0419 | (-0.0822, -0.0039) | 0.9870 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0021 | (-0.0208, 0.0149) | 0.5973 | -0.0021 | (-0.0170, 0.0125) | 0.6027 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0047 | (-0.0130, 0.0237) | 0.3037 | 0.0047 | (-0.0129, 0.0236) | 0.3057 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.0340 | (0.0066, 0.0640) | 0.0060 | 0.0340 | (0.0083, 0.0625) | 0.0037 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.0810 | (0.0550, 0.1096) | 0.0000 | 0.0810 | (0.0288, 0.1317) | 0.0007 |
| controlled_alt_vs_proposed_raw | naturalness | -0.0198 | (-0.0303, -0.0102) | 1.0000 | -0.0198 | (-0.0296, -0.0087) | 1.0000 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 0.0323 | (0.0093, 0.0564) | 0.0017 | 0.0323 | (0.0098, 0.0574) | 0.0007 |
| controlled_alt_vs_proposed_raw | lore_consistency | 0.0177 | (-0.0044, 0.0399) | 0.0603 | 0.0177 | (0.0013, 0.0347) | 0.0143 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0.1250 | (0.0764, 0.1806) | 0.0000 | 0.1250 | (0.0000, 0.3750) | 0.3417 |
| controlled_alt_vs_proposed_raw | objective_completion_support | 0.0170 | (0.0063, 0.0284) | 0.0010 | 0.0170 | (0.0113, 0.0239) | 0.0000 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 0.0062 | (-0.0025, 0.0153) | 0.0790 | 0.0062 | (-0.0032, 0.0167) | 0.1090 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | -0.0001 | (-0.0089, 0.0082) | 0.5157 | -0.0001 | (-0.0072, 0.0061) | 0.5110 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.0439 | (0.0087, 0.0796) | 0.0063 | 0.0439 | (0.0119, 0.0784) | 0.0010 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0108 | (-0.0008, 0.0228) | 0.0330 | 0.0108 | (0.0013, 0.0211) | 0.0103 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.0994 | (0.0667, 0.1338) | 0.0000 | 0.0994 | (0.0362, 0.1588) | 0.0007 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0075 | (-0.0079, 0.0237) | 0.1687 | 0.0075 | (-0.0052, 0.0235) | 0.1727 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0123 | (-0.0217, -0.0026) | 0.9947 | -0.0123 | (-0.0205, -0.0043) | 1.0000 |
| controlled_alt_vs_proposed_raw | length_score | -0.0884 | (-0.1289, -0.0479) | 1.0000 | -0.0884 | (-0.1375, -0.0370) | 1.0000 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0344 | (0.0076, 0.0608) | 0.0067 | 0.0344 | (0.0073, 0.0656) | 0.0070 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.0421 | (0.0250, 0.0601) | 0.0000 | 0.0421 | (0.0149, 0.0691) | 0.0007 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.0948 | (0.0701, 0.1207) | 0.0000 | 0.0948 | (0.0592, 0.1258) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.0870 | (0.0594, 0.1148) | 0.0000 | 0.0870 | (0.0362, 0.1331) | 0.0007 |
| controlled_alt_vs_candidate_no_context | naturalness | -0.0209 | (-0.0310, -0.0110) | 1.0000 | -0.0209 | (-0.0316, -0.0103) | 1.0000 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 0.0831 | (0.0616, 0.1057) | 0.0000 | 0.0831 | (0.0502, 0.1104) | 0.0000 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 0.0572 | (0.0389, 0.0773) | 0.0000 | 0.0572 | (0.0303, 0.0787) | 0.0000 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_candidate_no_context | objective_completion_support | 0.0174 | (0.0061, 0.0292) | 0.0017 | 0.0174 | (0.0080, 0.0271) | 0.0000 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 0.0228 | (0.0153, 0.0305) | 0.0000 | 0.0228 | (0.0119, 0.0339) | 0.0000 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | -0.0056 | (-0.0150, 0.0040) | 0.8750 | -0.0056 | (-0.0125, 0.0007) | 0.9587 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.1226 | (0.0899, 0.1569) | 0.0000 | 0.1226 | (0.0753, 0.1641) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0298 | (0.0203, 0.0405) | 0.0000 | 0.0298 | (0.0185, 0.0387) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1033 | (0.0702, 0.1372) | 0.0000 | 0.1033 | (0.0434, 0.1590) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0218 | (0.0057, 0.0390) | 0.0037 | 0.0218 | (-0.0022, 0.0542) | 0.0593 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0147 | (-0.0253, -0.0040) | 0.9977 | -0.0147 | (-0.0205, -0.0091) | 1.0000 |
| controlled_alt_vs_candidate_no_context | length_score | -0.0852 | (-0.1250, -0.0458) | 1.0000 | -0.0852 | (-0.1218, -0.0481) | 1.0000 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.0198 | (-0.0063, 0.0444) | 0.0663 | 0.0198 | (-0.0097, 0.0559) | 0.1300 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.0720 | (0.0545, 0.0906) | 0.0000 | 0.0720 | (0.0381, 0.1008) | 0.0000 |
| controlled_alt_vs_baseline_no_context | context_relevance | 0.0877 | (0.0645, 0.1128) | 0.0000 | 0.0877 | (0.0545, 0.1161) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 0.1424 | (0.1167, 0.1685) | 0.0000 | 0.1424 | (0.0803, 0.1986) | 0.0000 |
| controlled_alt_vs_baseline_no_context | naturalness | -0.0363 | (-0.0476, -0.0251) | 1.0000 | -0.0363 | (-0.0487, -0.0229) | 1.0000 |
| controlled_alt_vs_baseline_no_context | quest_state_correctness | 0.0774 | (0.0568, 0.0984) | 0.0000 | 0.0774 | (0.0468, 0.1028) | 0.0000 |
| controlled_alt_vs_baseline_no_context | lore_consistency | 0.0476 | (0.0287, 0.0670) | 0.0000 | 0.0476 | (0.0253, 0.0666) | 0.0000 |
| controlled_alt_vs_baseline_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_baseline_no_context | objective_completion_support | 0.0436 | (0.0304, 0.0564) | 0.0000 | 0.0436 | (0.0300, 0.0592) | 0.0000 |
| controlled_alt_vs_baseline_no_context | gameplay_usefulness | 0.0267 | (0.0182, 0.0354) | 0.0000 | 0.0267 | (0.0155, 0.0396) | 0.0000 |
| controlled_alt_vs_baseline_no_context | time_pressure_acceptability | 0.0046 | (-0.0056, 0.0153) | 0.1917 | 0.0046 | (-0.0116, 0.0194) | 0.2920 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 0.1136 | (0.0822, 0.1443) | 0.0000 | 0.1136 | (0.0704, 0.1495) | 0.0000 |
| controlled_alt_vs_baseline_no_context | context_overlap | 0.0273 | (0.0174, 0.0383) | 0.0000 | 0.0273 | (0.0132, 0.0375) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 0.1779 | (0.1461, 0.2117) | 0.0000 | 0.1779 | (0.0969, 0.2440) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_style | 0.0003 | (-0.0147, 0.0156) | 0.4663 | 0.0003 | (-0.0238, 0.0243) | 0.4740 |
| controlled_alt_vs_baseline_no_context | distinct1 | -0.0517 | (-0.0611, -0.0423) | 1.0000 | -0.0517 | (-0.0694, -0.0372) | 1.0000 |
| controlled_alt_vs_baseline_no_context | length_score | -0.1155 | (-0.1576, -0.0701) | 1.0000 | -0.1155 | (-0.1745, -0.0560) | 1.0000 |
| controlled_alt_vs_baseline_no_context | sentence_score | 0.0806 | (0.0472, 0.1142) | 0.0000 | 0.0806 | (0.0392, 0.1198) | 0.0000 |
| controlled_alt_vs_baseline_no_context | overall_quality | 0.0868 | (0.0711, 0.1035) | 0.0000 | 0.0868 | (0.0485, 0.1165) | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_relevance | 0.0931 | (0.0678, 0.1179) | 0.0000 | 0.0931 | (0.0576, 0.1236) | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1413 | (0.1157, 0.1684) | 0.0000 | 0.1413 | (0.0739, 0.1925) | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | naturalness | -0.0403 | (-0.0518, -0.0283) | 1.0000 | -0.0403 | (-0.0597, -0.0203) | 1.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.0827 | (0.0613, 0.1056) | 0.0000 | 0.0827 | (0.0505, 0.1090) | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | lore_consistency | 0.0467 | (0.0286, 0.0666) | 0.0000 | 0.0467 | (0.0243, 0.0674) | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.0492 | (0.0369, 0.0616) | 0.0000 | 0.0492 | (0.0325, 0.0685) | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 0.0272 | (0.0187, 0.0362) | 0.0000 | 0.0272 | (0.0119, 0.0433) | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 0.0064 | (-0.0035, 0.0172) | 0.1103 | 0.0064 | (-0.0095, 0.0223) | 0.1990 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.1215 | (0.0905, 0.1539) | 0.0000 | 0.1215 | (0.0737, 0.1611) | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_overlap | 0.0268 | (0.0170, 0.0382) | 0.0000 | 0.0268 | (0.0106, 0.0381) | 0.0007 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1771 | (0.1434, 0.2113) | 0.0000 | 0.1771 | (0.0953, 0.2379) | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_style | -0.0020 | (-0.0182, 0.0139) | 0.5910 | -0.0020 | (-0.0251, 0.0209) | 0.5600 |
| controlled_alt_vs_baseline_no_context_phi3_latest | distinct1 | -0.0530 | (-0.0624, -0.0436) | 1.0000 | -0.0530 | (-0.0676, -0.0400) | 1.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | length_score | -0.1410 | (-0.1884, -0.0912) | 1.0000 | -0.1410 | (-0.2174, -0.0646) | 1.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | sentence_score | 0.0972 | (0.0639, 0.1312) | 0.0000 | 0.0972 | (0.0437, 0.1552) | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | overall_quality | 0.0882 | (0.0722, 0.1053) | 0.0000 | 0.0882 | (0.0493, 0.1165) | 0.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | context_relevance | 0.0877 | (0.0645, 0.1128) | 0.0000 | 0.0877 | (0.0550, 0.1161) | 0.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | persona_consistency | 0.1424 | (0.1162, 0.1695) | 0.0000 | 0.1424 | (0.0798, 0.1978) | 0.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | naturalness | -0.0363 | (-0.0479, -0.0250) | 1.0000 | -0.0363 | (-0.0494, -0.0230) | 1.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | quest_state_correctness | 0.0774 | (0.0568, 0.0989) | 0.0000 | 0.0774 | (0.0470, 0.1018) | 0.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | lore_consistency | 0.0476 | (0.0289, 0.0667) | 0.0000 | 0.0476 | (0.0268, 0.0666) | 0.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | objective_completion_support | 0.0436 | (0.0311, 0.0562) | 0.0000 | 0.0436 | (0.0304, 0.0599) | 0.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | gameplay_usefulness | 0.0267 | (0.0180, 0.0355) | 0.0000 | 0.0267 | (0.0150, 0.0398) | 0.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | time_pressure_acceptability | 0.0046 | (-0.0060, 0.0149) | 0.1930 | 0.0046 | (-0.0116, 0.0194) | 0.2837 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | context_keyword_coverage | 0.1136 | (0.0830, 0.1467) | 0.0000 | 0.1136 | (0.0698, 0.1498) | 0.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | context_overlap | 0.0273 | (0.0177, 0.0382) | 0.0000 | 0.0273 | (0.0131, 0.0372) | 0.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | persona_keyword_coverage | 0.1779 | (0.1470, 0.2105) | 0.0000 | 0.1779 | (0.0957, 0.2438) | 0.0003 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | persona_style | 0.0003 | (-0.0155, 0.0164) | 0.4837 | 0.0003 | (-0.0238, 0.0245) | 0.4563 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | distinct1 | -0.0517 | (-0.0615, -0.0423) | 1.0000 | -0.0517 | (-0.0688, -0.0370) | 1.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | length_score | -0.1155 | (-0.1604, -0.0697) | 1.0000 | -0.1155 | (-0.1720, -0.0544) | 1.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | sentence_score | 0.0806 | (0.0486, 0.1142) | 0.0000 | 0.0806 | (0.0389, 0.1177) | 0.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | overall_quality | 0.0868 | (0.0706, 0.1038) | 0.0000 | 0.0868 | (0.0486, 0.1162) | 0.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | context_relevance | 0.0931 | (0.0680, 0.1190) | 0.0000 | 0.0931 | (0.0552, 0.1234) | 0.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1413 | (0.1156, 0.1696) | 0.0000 | 0.1413 | (0.0745, 0.1884) | 0.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | naturalness | -0.0403 | (-0.0518, -0.0291) | 1.0000 | -0.0403 | (-0.0590, -0.0212) | 1.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.0827 | (0.0618, 0.1051) | 0.0000 | 0.0827 | (0.0506, 0.1090) | 0.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | lore_consistency | 0.0467 | (0.0294, 0.0658) | 0.0000 | 0.0467 | (0.0232, 0.0670) | 0.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.0492 | (0.0371, 0.0614) | 0.0000 | 0.0492 | (0.0316, 0.0693) | 0.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 0.0272 | (0.0185, 0.0367) | 0.0000 | 0.0272 | (0.0114, 0.0432) | 0.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 0.0064 | (-0.0041, 0.0168) | 0.1163 | 0.0064 | (-0.0099, 0.0212) | 0.2070 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.1215 | (0.0880, 0.1546) | 0.0000 | 0.1215 | (0.0782, 0.1612) | 0.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | context_overlap | 0.0268 | (0.0169, 0.0380) | 0.0000 | 0.0268 | (0.0111, 0.0382) | 0.0027 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1771 | (0.1457, 0.2103) | 0.0000 | 0.1771 | (0.0957, 0.2379) | 0.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | persona_style | -0.0020 | (-0.0184, 0.0139) | 0.5953 | -0.0020 | (-0.0254, 0.0200) | 0.5720 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | distinct1 | -0.0530 | (-0.0626, -0.0437) | 1.0000 | -0.0530 | (-0.0677, -0.0402) | 1.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | length_score | -0.1410 | (-0.1896, -0.0963) | 1.0000 | -0.1410 | (-0.2183, -0.0674) | 1.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | sentence_score | 0.0972 | (0.0618, 0.1333) | 0.0000 | 0.0972 | (0.0437, 0.1556) | 0.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | overall_quality | 0.0882 | (0.0714, 0.1050) | 0.0000 | 0.0882 | (0.0501, 0.1160) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 68 | 24 | 52 | 0.6528 | 0.7391 |
| proposed_vs_candidate_no_context | persona_consistency | 27 | 21 | 96 | 0.5208 | 0.5625 |
| proposed_vs_candidate_no_context | naturalness | 48 | 45 | 51 | 0.5104 | 0.5161 |
| proposed_vs_candidate_no_context | quest_state_correctness | 65 | 27 | 52 | 0.6319 | 0.7065 |
| proposed_vs_candidate_no_context | lore_consistency | 49 | 12 | 83 | 0.6285 | 0.8033 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 18 | 126 | 0.4375 | 0.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | 41 | 31 | 72 | 0.5347 | 0.5694 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 66 | 27 | 51 | 0.6354 | 0.7097 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | 37 | 45 | 62 | 0.4722 | 0.4512 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 55 | 10 | 79 | 0.6562 | 0.8462 |
| proposed_vs_candidate_no_context | context_overlap | 65 | 27 | 52 | 0.6319 | 0.7065 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 19 | 19 | 106 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | persona_style | 10 | 4 | 130 | 0.5208 | 0.7143 |
| proposed_vs_candidate_no_context | distinct1 | 35 | 45 | 64 | 0.4653 | 0.4375 |
| proposed_vs_candidate_no_context | length_score | 48 | 39 | 57 | 0.5312 | 0.5517 |
| proposed_vs_candidate_no_context | sentence_score | 14 | 20 | 110 | 0.4792 | 0.4118 |
| proposed_vs_candidate_no_context | overall_quality | 65 | 28 | 51 | 0.6285 | 0.6989 |
| proposed_vs_baseline_no_context | context_relevance | 81 | 60 | 3 | 0.5729 | 0.5745 |
| proposed_vs_baseline_no_context | persona_consistency | 66 | 25 | 53 | 0.6424 | 0.7253 |
| proposed_vs_baseline_no_context | naturalness | 51 | 93 | 0 | 0.3542 | 0.3542 |
| proposed_vs_baseline_no_context | quest_state_correctness | 84 | 57 | 3 | 0.5938 | 0.5957 |
| proposed_vs_baseline_no_context | lore_consistency | 49 | 37 | 58 | 0.5417 | 0.5698 |
| proposed_vs_baseline_no_context | multi_turn_contradiction_safety | 0 | 18 | 126 | 0.4375 | 0.0000 |
| proposed_vs_baseline_no_context | objective_completion_support | 69 | 39 | 36 | 0.6042 | 0.6389 |
| proposed_vs_baseline_no_context | gameplay_usefulness | 93 | 51 | 0 | 0.6458 | 0.6458 |
| proposed_vs_baseline_no_context | time_pressure_acceptability | 65 | 63 | 16 | 0.5069 | 0.5078 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 56 | 22 | 66 | 0.6181 | 0.7179 |
| proposed_vs_baseline_no_context | context_overlap | 86 | 52 | 6 | 0.6181 | 0.6232 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 62 | 19 | 63 | 0.6493 | 0.7654 |
| proposed_vs_baseline_no_context | persona_style | 11 | 16 | 117 | 0.4826 | 0.4074 |
| proposed_vs_baseline_no_context | distinct1 | 34 | 95 | 15 | 0.2882 | 0.2636 |
| proposed_vs_baseline_no_context | length_score | 61 | 77 | 6 | 0.4444 | 0.4420 |
| proposed_vs_baseline_no_context | sentence_score | 34 | 15 | 95 | 0.5660 | 0.6939 |
| proposed_vs_baseline_no_context | overall_quality | 103 | 41 | 0 | 0.7153 | 0.7153 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 86 | 52 | 6 | 0.6181 | 0.6232 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | 68 | 29 | 47 | 0.6354 | 0.7010 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | 54 | 89 | 1 | 0.3785 | 0.3776 |
| proposed_vs_baseline_no_context_phi3_latest | quest_state_correctness | 87 | 51 | 6 | 0.6250 | 0.6304 |
| proposed_vs_baseline_no_context_phi3_latest | lore_consistency | 48 | 44 | 52 | 0.5139 | 0.5217 |
| proposed_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0 | 18 | 126 | 0.4375 | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | objective_completion_support | 76 | 29 | 39 | 0.6632 | 0.7238 |
| proposed_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 90 | 54 | 0 | 0.6250 | 0.6250 |
| proposed_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 66 | 58 | 20 | 0.5278 | 0.5323 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 62 | 16 | 66 | 0.6597 | 0.7949 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 87 | 51 | 6 | 0.6250 | 0.6304 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 63 | 20 | 61 | 0.6493 | 0.7590 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | 9 | 16 | 119 | 0.4757 | 0.3600 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | 29 | 98 | 17 | 0.2604 | 0.2283 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | 59 | 79 | 6 | 0.4306 | 0.4275 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | 38 | 13 | 93 | 0.5868 | 0.7451 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | 108 | 36 | 0 | 0.7500 | 0.7500 |
| controlled_vs_proposed_raw | context_relevance | 60 | 57 | 27 | 0.5104 | 0.5128 |
| controlled_vs_proposed_raw | persona_consistency | 72 | 17 | 55 | 0.6910 | 0.8090 |
| controlled_vs_proposed_raw | naturalness | 53 | 65 | 26 | 0.4583 | 0.4492 |
| controlled_vs_proposed_raw | quest_state_correctness | 62 | 55 | 27 | 0.5243 | 0.5299 |
| controlled_vs_proposed_raw | lore_consistency | 39 | 40 | 65 | 0.4965 | 0.4937 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 18 | 0 | 126 | 0.5625 | 1.0000 |
| controlled_vs_proposed_raw | objective_completion_support | 52 | 37 | 55 | 0.5521 | 0.5843 |
| controlled_vs_proposed_raw | gameplay_usefulness | 67 | 51 | 26 | 0.5556 | 0.5678 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 47 | 52 | 45 | 0.4826 | 0.4747 |
| controlled_vs_proposed_raw | context_keyword_coverage | 39 | 32 | 73 | 0.5243 | 0.5493 |
| controlled_vs_proposed_raw | context_overlap | 63 | 54 | 27 | 0.5312 | 0.5385 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 66 | 14 | 64 | 0.6806 | 0.8250 |
| controlled_vs_proposed_raw | persona_style | 15 | 11 | 118 | 0.5139 | 0.5769 |
| controlled_vs_proposed_raw | distinct1 | 55 | 58 | 31 | 0.4896 | 0.4867 |
| controlled_vs_proposed_raw | length_score | 50 | 64 | 30 | 0.4514 | 0.4386 |
| controlled_vs_proposed_raw | sentence_score | 23 | 8 | 113 | 0.5521 | 0.7419 |
| controlled_vs_proposed_raw | overall_quality | 78 | 40 | 26 | 0.6319 | 0.6610 |
| controlled_vs_candidate_no_context | context_relevance | 85 | 33 | 26 | 0.6806 | 0.7203 |
| controlled_vs_candidate_no_context | persona_consistency | 76 | 17 | 51 | 0.7049 | 0.8172 |
| controlled_vs_candidate_no_context | naturalness | 48 | 71 | 25 | 0.4201 | 0.4034 |
| controlled_vs_candidate_no_context | quest_state_correctness | 81 | 37 | 26 | 0.6528 | 0.6864 |
| controlled_vs_candidate_no_context | lore_consistency | 51 | 11 | 82 | 0.6389 | 0.8226 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 144 | 0.5000 | nan |
| controlled_vs_candidate_no_context | objective_completion_support | 54 | 35 | 55 | 0.5660 | 0.6067 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 87 | 33 | 24 | 0.6875 | 0.7250 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 47 | 53 | 44 | 0.4792 | 0.4700 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 61 | 13 | 70 | 0.6667 | 0.8243 |
| controlled_vs_candidate_no_context | context_overlap | 84 | 34 | 26 | 0.6736 | 0.7119 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 71 | 16 | 57 | 0.6910 | 0.8161 |
| controlled_vs_candidate_no_context | persona_style | 19 | 9 | 116 | 0.5347 | 0.6786 |
| controlled_vs_candidate_no_context | distinct1 | 48 | 63 | 33 | 0.4479 | 0.4324 |
| controlled_vs_candidate_no_context | length_score | 49 | 68 | 27 | 0.4340 | 0.4188 |
| controlled_vs_candidate_no_context | sentence_score | 18 | 9 | 117 | 0.5312 | 0.6667 |
| controlled_vs_candidate_no_context | overall_quality | 94 | 26 | 24 | 0.7361 | 0.7833 |
| controlled_vs_baseline_no_context | context_relevance | 88 | 54 | 2 | 0.6181 | 0.6197 |
| controlled_vs_baseline_no_context | persona_consistency | 109 | 13 | 22 | 0.8333 | 0.8934 |
| controlled_vs_baseline_no_context | naturalness | 53 | 90 | 1 | 0.3715 | 0.3706 |
| controlled_vs_baseline_no_context | quest_state_correctness | 88 | 54 | 2 | 0.6181 | 0.6197 |
| controlled_vs_baseline_no_context | lore_consistency | 51 | 41 | 52 | 0.5347 | 0.5543 |
| controlled_vs_baseline_no_context | multi_turn_contradiction_safety | 0 | 0 | 144 | 0.5000 | nan |
| controlled_vs_baseline_no_context | objective_completion_support | 82 | 33 | 29 | 0.6701 | 0.7130 |
| controlled_vs_baseline_no_context | gameplay_usefulness | 103 | 41 | 0 | 0.7153 | 0.7153 |
| controlled_vs_baseline_no_context | time_pressure_acceptability | 73 | 56 | 15 | 0.5590 | 0.5659 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 63 | 20 | 61 | 0.6493 | 0.7590 |
| controlled_vs_baseline_no_context | context_overlap | 92 | 49 | 3 | 0.6493 | 0.6525 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 108 | 11 | 25 | 0.8368 | 0.9076 |
| controlled_vs_baseline_no_context | persona_style | 13 | 11 | 120 | 0.5069 | 0.5417 |
| controlled_vs_baseline_no_context | distinct1 | 21 | 112 | 11 | 0.1840 | 0.1579 |
| controlled_vs_baseline_no_context | length_score | 55 | 84 | 5 | 0.3993 | 0.3957 |
| controlled_vs_baseline_no_context | sentence_score | 42 | 8 | 94 | 0.6181 | 0.8400 |
| controlled_vs_baseline_no_context | overall_quality | 116 | 28 | 0 | 0.8056 | 0.8056 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 89 | 50 | 5 | 0.6354 | 0.6403 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 109 | 16 | 19 | 0.8229 | 0.8720 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | 61 | 83 | 0 | 0.4236 | 0.4236 |
| controlled_vs_baseline_no_context_phi3_latest | quest_state_correctness | 91 | 49 | 4 | 0.6458 | 0.6500 |
| controlled_vs_baseline_no_context_phi3_latest | lore_consistency | 50 | 45 | 49 | 0.5174 | 0.5263 |
| controlled_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0 | 0 | 144 | 0.5000 | nan |
| controlled_vs_baseline_no_context_phi3_latest | objective_completion_support | 88 | 25 | 31 | 0.7188 | 0.7788 |
| controlled_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 106 | 38 | 0 | 0.7361 | 0.7361 |
| controlled_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 75 | 52 | 17 | 0.5799 | 0.5906 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 66 | 16 | 62 | 0.6736 | 0.8049 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 90 | 49 | 5 | 0.6424 | 0.6475 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 108 | 13 | 23 | 0.8299 | 0.8926 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | 9 | 13 | 122 | 0.4861 | 0.4091 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | 26 | 111 | 7 | 0.2049 | 0.1898 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | 63 | 78 | 3 | 0.4479 | 0.4468 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 49 | 8 | 87 | 0.6424 | 0.8596 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 119 | 25 | 0 | 0.8264 | 0.8264 |
| controlled_alt_vs_controlled_default | context_relevance | 43 | 45 | 56 | 0.4931 | 0.4886 |
| controlled_alt_vs_controlled_default | persona_consistency | 29 | 24 | 91 | 0.5174 | 0.5472 |
| controlled_alt_vs_controlled_default | naturalness | 32 | 56 | 56 | 0.4167 | 0.3636 |
| controlled_alt_vs_controlled_default | quest_state_correctness | 42 | 46 | 56 | 0.4861 | 0.4773 |
| controlled_alt_vs_controlled_default | lore_consistency | 35 | 38 | 71 | 0.4896 | 0.4795 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0 | 0 | 144 | 0.5000 | nan |
| controlled_alt_vs_controlled_default | objective_completion_support | 33 | 36 | 75 | 0.4896 | 0.4783 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | 36 | 52 | 56 | 0.4444 | 0.4091 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | 40 | 41 | 63 | 0.4965 | 0.4938 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 38 | 37 | 69 | 0.5035 | 0.5067 |
| controlled_alt_vs_controlled_default | context_overlap | 42 | 46 | 56 | 0.4861 | 0.4773 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 26 | 20 | 98 | 0.5208 | 0.5652 |
| controlled_alt_vs_controlled_default | persona_style | 11 | 8 | 125 | 0.5104 | 0.5789 |
| controlled_alt_vs_controlled_default | distinct1 | 42 | 43 | 59 | 0.4965 | 0.4941 |
| controlled_alt_vs_controlled_default | length_score | 35 | 45 | 64 | 0.4653 | 0.4375 |
| controlled_alt_vs_controlled_default | sentence_score | 7 | 8 | 129 | 0.4965 | 0.4667 |
| controlled_alt_vs_controlled_default | overall_quality | 44 | 44 | 56 | 0.5000 | 0.5000 |
| controlled_alt_vs_proposed_raw | context_relevance | 65 | 50 | 29 | 0.5521 | 0.5652 |
| controlled_alt_vs_proposed_raw | persona_consistency | 63 | 13 | 68 | 0.6736 | 0.8289 |
| controlled_alt_vs_proposed_raw | naturalness | 41 | 74 | 29 | 0.3854 | 0.3565 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 63 | 52 | 29 | 0.5382 | 0.5478 |
| controlled_alt_vs_proposed_raw | lore_consistency | 40 | 38 | 66 | 0.5069 | 0.5128 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 18 | 0 | 126 | 0.5625 | 1.0000 |
| controlled_alt_vs_proposed_raw | objective_completion_support | 48 | 35 | 61 | 0.5451 | 0.5783 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 59 | 56 | 29 | 0.5104 | 0.5130 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | 51 | 48 | 45 | 0.5104 | 0.5152 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 46 | 35 | 63 | 0.5382 | 0.5679 |
| controlled_alt_vs_proposed_raw | context_overlap | 63 | 52 | 29 | 0.5382 | 0.5478 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 61 | 11 | 72 | 0.6736 | 0.8472 |
| controlled_alt_vs_proposed_raw | persona_style | 13 | 8 | 123 | 0.5174 | 0.6190 |
| controlled_alt_vs_proposed_raw | distinct1 | 51 | 60 | 33 | 0.4688 | 0.4595 |
| controlled_alt_vs_proposed_raw | length_score | 35 | 74 | 35 | 0.3646 | 0.3211 |
| controlled_alt_vs_proposed_raw | sentence_score | 23 | 8 | 113 | 0.5521 | 0.7419 |
| controlled_alt_vs_proposed_raw | overall_quality | 74 | 41 | 29 | 0.6146 | 0.6435 |
| controlled_alt_vs_candidate_no_context | context_relevance | 85 | 30 | 29 | 0.6910 | 0.7391 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 73 | 14 | 57 | 0.7049 | 0.8391 |
| controlled_alt_vs_candidate_no_context | naturalness | 39 | 76 | 29 | 0.3715 | 0.3391 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 81 | 34 | 29 | 0.6632 | 0.7043 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 48 | 12 | 84 | 0.6250 | 0.8000 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 144 | 0.5000 | nan |
| controlled_alt_vs_candidate_no_context | objective_completion_support | 53 | 31 | 60 | 0.5764 | 0.6310 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 85 | 30 | 29 | 0.6910 | 0.7391 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | 44 | 55 | 45 | 0.4618 | 0.4444 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 64 | 14 | 66 | 0.6736 | 0.8205 |
| controlled_alt_vs_candidate_no_context | context_overlap | 80 | 35 | 29 | 0.6562 | 0.6957 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 69 | 13 | 62 | 0.6944 | 0.8415 |
| controlled_alt_vs_candidate_no_context | persona_style | 18 | 6 | 120 | 0.5417 | 0.7500 |
| controlled_alt_vs_candidate_no_context | distinct1 | 49 | 62 | 33 | 0.4549 | 0.4414 |
| controlled_alt_vs_candidate_no_context | length_score | 38 | 75 | 31 | 0.3715 | 0.3363 |
| controlled_alt_vs_candidate_no_context | sentence_score | 18 | 9 | 117 | 0.5312 | 0.6667 |
| controlled_alt_vs_candidate_no_context | overall_quality | 92 | 23 | 29 | 0.7396 | 0.8000 |
| controlled_alt_vs_baseline_no_context | context_relevance | 94 | 50 | 0 | 0.6528 | 0.6528 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 111 | 10 | 23 | 0.8507 | 0.9174 |
| controlled_alt_vs_baseline_no_context | naturalness | 41 | 103 | 0 | 0.2847 | 0.2847 |
| controlled_alt_vs_baseline_no_context | quest_state_correctness | 92 | 52 | 0 | 0.6389 | 0.6389 |
| controlled_alt_vs_baseline_no_context | lore_consistency | 48 | 40 | 56 | 0.5278 | 0.5455 |
| controlled_alt_vs_baseline_no_context | multi_turn_contradiction_safety | 0 | 0 | 144 | 0.5000 | nan |
| controlled_alt_vs_baseline_no_context | objective_completion_support | 82 | 31 | 31 | 0.6771 | 0.7257 |
| controlled_alt_vs_baseline_no_context | gameplay_usefulness | 99 | 45 | 0 | 0.6875 | 0.6875 |
| controlled_alt_vs_baseline_no_context | time_pressure_acceptability | 69 | 64 | 11 | 0.5174 | 0.5188 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 64 | 14 | 66 | 0.6736 | 0.8205 |
| controlled_alt_vs_baseline_no_context | context_overlap | 93 | 51 | 0 | 0.6458 | 0.6458 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 111 | 9 | 24 | 0.8542 | 0.9250 |
| controlled_alt_vs_baseline_no_context | persona_style | 12 | 11 | 121 | 0.5035 | 0.5217 |
| controlled_alt_vs_baseline_no_context | distinct1 | 24 | 115 | 5 | 0.1840 | 0.1727 |
| controlled_alt_vs_baseline_no_context | length_score | 45 | 94 | 5 | 0.3299 | 0.3237 |
| controlled_alt_vs_baseline_no_context | sentence_score | 43 | 9 | 92 | 0.6181 | 0.8269 |
| controlled_alt_vs_baseline_no_context | overall_quality | 118 | 26 | 0 | 0.8194 | 0.8194 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_relevance | 89 | 54 | 1 | 0.6215 | 0.6224 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_consistency | 107 | 13 | 24 | 0.8264 | 0.8917 |
| controlled_alt_vs_baseline_no_context_phi3_latest | naturalness | 45 | 99 | 0 | 0.3125 | 0.3125 |
| controlled_alt_vs_baseline_no_context_phi3_latest | quest_state_correctness | 92 | 52 | 0 | 0.6389 | 0.6389 |
| controlled_alt_vs_baseline_no_context_phi3_latest | lore_consistency | 48 | 45 | 51 | 0.5104 | 0.5161 |
| controlled_alt_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0 | 0 | 144 | 0.5000 | nan |
| controlled_alt_vs_baseline_no_context_phi3_latest | objective_completion_support | 90 | 25 | 29 | 0.7257 | 0.7826 |
| controlled_alt_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 95 | 49 | 0 | 0.6597 | 0.6597 |
| controlled_alt_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 73 | 58 | 13 | 0.5521 | 0.5573 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 67 | 16 | 61 | 0.6771 | 0.8072 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_overlap | 88 | 55 | 1 | 0.6146 | 0.6154 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 107 | 10 | 27 | 0.8368 | 0.9145 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_style | 11 | 11 | 122 | 0.5000 | 0.5000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | distinct1 | 21 | 116 | 7 | 0.1701 | 0.1533 |
| controlled_alt_vs_baseline_no_context_phi3_latest | length_score | 47 | 92 | 5 | 0.3438 | 0.3381 |
| controlled_alt_vs_baseline_no_context_phi3_latest | sentence_score | 49 | 9 | 86 | 0.6389 | 0.8448 |
| controlled_alt_vs_baseline_no_context_phi3_latest | overall_quality | 120 | 24 | 0 | 0.8333 | 0.8333 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | context_relevance | 94 | 50 | 0 | 0.6528 | 0.6528 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | persona_consistency | 111 | 10 | 23 | 0.8507 | 0.9174 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | naturalness | 41 | 103 | 0 | 0.2847 | 0.2847 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | quest_state_correctness | 92 | 52 | 0 | 0.6389 | 0.6389 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | lore_consistency | 48 | 40 | 56 | 0.5278 | 0.5455 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | multi_turn_contradiction_safety | 0 | 0 | 144 | 0.5000 | nan |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | objective_completion_support | 82 | 31 | 31 | 0.6771 | 0.7257 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | gameplay_usefulness | 99 | 45 | 0 | 0.6875 | 0.6875 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | time_pressure_acceptability | 69 | 64 | 11 | 0.5174 | 0.5188 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | context_keyword_coverage | 64 | 14 | 66 | 0.6736 | 0.8205 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | context_overlap | 93 | 51 | 0 | 0.6458 | 0.6458 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | persona_keyword_coverage | 111 | 9 | 24 | 0.8542 | 0.9250 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | persona_style | 12 | 11 | 121 | 0.5035 | 0.5217 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | distinct1 | 24 | 115 | 5 | 0.1840 | 0.1727 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | length_score | 45 | 94 | 5 | 0.3299 | 0.3237 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | sentence_score | 43 | 9 | 92 | 0.6181 | 0.8269 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | overall_quality | 118 | 26 | 0 | 0.8194 | 0.8194 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | context_relevance | 89 | 54 | 1 | 0.6215 | 0.6224 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | persona_consistency | 107 | 13 | 24 | 0.8264 | 0.8917 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | naturalness | 45 | 99 | 0 | 0.3125 | 0.3125 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | quest_state_correctness | 92 | 52 | 0 | 0.6389 | 0.6389 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | lore_consistency | 48 | 45 | 51 | 0.5104 | 0.5161 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0 | 0 | 144 | 0.5000 | nan |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | objective_completion_support | 90 | 25 | 29 | 0.7257 | 0.7826 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 95 | 49 | 0 | 0.6597 | 0.6597 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 73 | 58 | 13 | 0.5521 | 0.5573 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 67 | 16 | 61 | 0.6771 | 0.8072 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | context_overlap | 88 | 55 | 1 | 0.6146 | 0.6154 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 107 | 10 | 27 | 0.8368 | 0.9145 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | persona_style | 11 | 11 | 122 | 0.5000 | 0.5000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | distinct1 | 21 | 116 | 7 | 0.1701 | 0.1533 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | length_score | 47 | 92 | 5 | 0.3438 | 0.3381 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | sentence_score | 49 | 9 | 86 | 0.6389 | 0.8448 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | overall_quality | 120 | 24 | 0 | 0.8333 | 0.8333 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.1944 | 0.2083 | 0.7917 |
| proposed_contextual_controlled_tuned | 0.0000 | 0.0000 | 0.2500 | 0.1597 | 0.8403 |
| proposed_contextual | 0.0000 | 0.0000 | 0.4306 | 0.0000 | 0.0000 |
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
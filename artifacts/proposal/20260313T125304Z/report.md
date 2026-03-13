# Proposal Alignment Evaluation Report

- Run ID: `20260313T125304Z`
- Generated: `2026-03-13T12:56:37.691394+00:00`
- Scenarios: `artifacts\proposal\20260313T125304Z\scenarios.jsonl`
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
| proposed_contextual_controlled | 0.1183 (0.0941, 0.1436) | 0.2817 (0.2606, 0.3035) | 0.8727 (0.8657, 0.8794) | 0.3260 (0.3114, 0.3410) | n/a |
| proposed_contextual_controlled_tuned | 0.1302 (0.1090, 0.1533) | 0.2848 (0.2628, 0.3091) | 0.8693 (0.8622, 0.8764) | 0.3318 (0.3178, 0.3468) | n/a |
| proposed_contextual | 0.0871 (0.0706, 0.1037) | 0.2077 (0.1883, 0.2297) | 0.8725 (0.8665, 0.8787) | 0.2839 (0.2733, 0.2951) | n/a |
| candidate_no_context | 0.0295 (0.0233, 0.0363) | 0.2293 (0.2073, 0.2519) | 0.8762 (0.8696, 0.8830) | 0.2660 (0.2567, 0.2760) | n/a |
| baseline_no_context | 0.0390 (0.0322, 0.0470) | 0.1565 (0.1420, 0.1715) | 0.8962 (0.8869, 0.9052) | 0.2467 (0.2388, 0.2546) | n/a |
| baseline_no_context_phi3_latest | 0.0392 (0.0320, 0.0468) | 0.1575 (0.1425, 0.1736) | 0.8915 (0.8828, 0.9002) | 0.2463 (0.2389, 0.2544) | n/a |

## Game-facing Outcome Metrics (mean, 95% CI)
| Arm | Quest-state Correctness | Lore Consistency | Contradiction Safety | Objective Completion Support | Gameplay Usefulness | Time-pressure Acceptability |
|---|---:|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2026 (0.1828, 0.2234) | 0.0586 (0.0425, 0.0780) | 1.0000 (1.0000, 1.0000) | 0.0842 (0.0719, 0.0965) | 0.3101 (0.3033, 0.3176) | 0.3001 (0.2916, 0.3088) |
| proposed_contextual_controlled_tuned | 0.2120 (0.1935, 0.2314) | 0.0600 (0.0453, 0.0760) | 0.8750 (0.8194, 0.9236) | 0.0809 (0.0689, 0.0938) | 0.3077 (0.3009, 0.3147) | 0.2973 (0.2878, 0.3063) |
| proposed_contextual | 0.1749 (0.1612, 0.1896) | 0.0426 (0.0322, 0.0547) | 0.8750 (0.8194, 0.9236) | 0.0683 (0.0568, 0.0809) | 0.2950 (0.2877, 0.3023) | 0.2924 (0.2843, 0.3010) |
| candidate_no_context | 0.1257 (0.1204, 0.1316) | 0.0053 (0.0028, 0.0083) | 1.0000 (1.0000, 1.0000) | 0.0732 (0.0618, 0.0846) | 0.2808 (0.2755, 0.2867) | 0.2986 (0.2913, 0.3063) |
| baseline_no_context | 0.1326 (0.1263, 0.1395) | 0.0188 (0.0139, 0.0241) | 0.8750 (0.8194, 0.9236) | 0.0388 (0.0324, 0.0453) | 0.2793 (0.2739, 0.2851) | 0.2869 (0.2817, 0.2924) |
| baseline_no_context_phi3_latest | 0.1332 (0.1268, 0.1401) | 0.0173 (0.0129, 0.0219) | 0.8750 (0.8194, 0.9236) | 0.0404 (0.0329, 0.0487) | 0.2775 (0.2718, 0.2833) | 0.2874 (0.2820, 0.2932) |

- Multi-turn contradiction rate is reported as `1 - contradiction_safety` in row-level outputs.

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0576 | 1.9532 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0216 | -0.0943 |
| proposed_vs_candidate_no_context | naturalness | -0.0037 | -0.0042 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.0492 | 0.3912 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0373 | 7.0288 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | -0.1250 | -0.1250 |
| proposed_vs_candidate_no_context | objective_completion_support | -0.0049 | -0.0669 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 0.0142 | 0.0504 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | -0.0062 | -0.0208 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0739 | 2.6400 |
| proposed_vs_candidate_no_context | context_overlap | 0.0196 | 0.5930 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0263 | -0.1731 |
| proposed_vs_candidate_no_context | persona_style | -0.0028 | -0.0052 |
| proposed_vs_candidate_no_context | distinct1 | -0.0022 | -0.0023 |
| proposed_vs_candidate_no_context | length_score | -0.0030 | -0.0058 |
| proposed_vs_candidate_no_context | sentence_score | -0.0219 | -0.0231 |
| proposed_vs_candidate_no_context | overall_quality | 0.0178 | 0.0671 |
| proposed_vs_baseline_no_context | context_relevance | 0.0481 | 1.2326 |
| proposed_vs_baseline_no_context | persona_consistency | 0.0512 | 0.3274 |
| proposed_vs_baseline_no_context | naturalness | -0.0237 | -0.0265 |
| proposed_vs_baseline_no_context | quest_state_correctness | 0.0423 | 0.3189 |
| proposed_vs_baseline_no_context | lore_consistency | 0.0238 | 1.2629 |
| proposed_vs_baseline_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| proposed_vs_baseline_no_context | objective_completion_support | 0.0295 | 0.7605 |
| proposed_vs_baseline_no_context | gameplay_usefulness | 0.0157 | 0.0561 |
| proposed_vs_baseline_no_context | time_pressure_acceptability | 0.0055 | 0.0193 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0631 | 1.6271 |
| proposed_vs_baseline_no_context | context_overlap | 0.0130 | 0.3294 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 0.0697 | 1.2437 |
| proposed_vs_baseline_no_context | persona_style | -0.0228 | -0.0408 |
| proposed_vs_baseline_no_context | distinct1 | -0.0413 | -0.0421 |
| proposed_vs_baseline_no_context | length_score | -0.0579 | -0.1003 |
| proposed_vs_baseline_no_context | sentence_score | 0.0438 | 0.0495 |
| proposed_vs_baseline_no_context | overall_quality | 0.0372 | 0.1506 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 0.0479 | 1.2232 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | 0.0501 | 0.3182 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | -0.0190 | -0.0213 |
| proposed_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.0417 | 0.3130 |
| proposed_vs_baseline_no_context_phi3_latest | lore_consistency | 0.0253 | 1.4643 |
| proposed_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.0279 | 0.6919 |
| proposed_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 0.0175 | 0.0631 |
| proposed_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 0.0050 | 0.0172 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.0625 | 1.5850 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 0.0139 | 0.3611 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.0677 | 1.1645 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | -0.0200 | -0.0360 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | -0.0408 | -0.0416 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | -0.0338 | -0.0611 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | 0.0410 | 0.0462 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | 0.0375 | 0.1525 |
| controlled_vs_proposed_raw | context_relevance | 0.0312 | 0.3584 |
| controlled_vs_proposed_raw | persona_consistency | 0.0740 | 0.3563 |
| controlled_vs_proposed_raw | naturalness | 0.0002 | 0.0003 |
| controlled_vs_proposed_raw | quest_state_correctness | 0.0277 | 0.1583 |
| controlled_vs_proposed_raw | lore_consistency | 0.0160 | 0.3748 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0.1250 | 0.1429 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0159 | 0.2330 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0151 | 0.0511 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 0.0077 | 0.0262 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.0408 | 0.4002 |
| controlled_vs_proposed_raw | context_overlap | 0.0089 | 0.1692 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.0895 | 0.7118 |
| controlled_vs_proposed_raw | persona_style | 0.0119 | 0.0222 |
| controlled_vs_proposed_raw | distinct1 | -0.0068 | -0.0073 |
| controlled_vs_proposed_raw | length_score | -0.0095 | -0.0183 |
| controlled_vs_proposed_raw | sentence_score | 0.0490 | 0.0528 |
| controlled_vs_proposed_raw | overall_quality | 0.0421 | 0.1484 |
| controlled_vs_candidate_no_context | context_relevance | 0.0888 | 3.0115 |
| controlled_vs_candidate_no_context | persona_consistency | 0.0524 | 0.2284 |
| controlled_vs_candidate_no_context | naturalness | -0.0035 | -0.0040 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.0769 | 0.6114 |
| controlled_vs_candidate_no_context | lore_consistency | 0.0533 | 10.0378 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | 0.0110 | 0.1505 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0292 | 0.1041 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 0.0015 | 0.0049 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.1147 | 4.0966 |
| controlled_vs_candidate_no_context | context_overlap | 0.0285 | 0.8626 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.0632 | 0.4155 |
| controlled_vs_candidate_no_context | persona_style | 0.0091 | 0.0169 |
| controlled_vs_candidate_no_context | distinct1 | -0.0090 | -0.0095 |
| controlled_vs_candidate_no_context | length_score | -0.0125 | -0.0239 |
| controlled_vs_candidate_no_context | sentence_score | 0.0271 | 0.0285 |
| controlled_vs_candidate_no_context | overall_quality | 0.0600 | 0.2254 |
| controlled_vs_baseline_no_context | context_relevance | 0.0793 | 2.0327 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1252 | 0.8003 |
| controlled_vs_baseline_no_context | naturalness | -0.0235 | -0.0262 |
| controlled_vs_baseline_no_context | quest_state_correctness | 0.0700 | 0.5277 |
| controlled_vs_baseline_no_context | lore_consistency | 0.0398 | 2.1110 |
| controlled_vs_baseline_no_context | multi_turn_contradiction_safety | 0.1250 | 0.1429 |
| controlled_vs_baseline_no_context | objective_completion_support | 0.0454 | 1.1707 |
| controlled_vs_baseline_no_context | gameplay_usefulness | 0.0307 | 0.1100 |
| controlled_vs_baseline_no_context | time_pressure_acceptability | 0.0132 | 0.0460 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.1039 | 2.6783 |
| controlled_vs_baseline_no_context | context_overlap | 0.0219 | 0.5543 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1592 | 2.8407 |
| controlled_vs_baseline_no_context | persona_style | -0.0109 | -0.0195 |
| controlled_vs_baseline_no_context | distinct1 | -0.0481 | -0.0490 |
| controlled_vs_baseline_no_context | length_score | -0.0674 | -0.1168 |
| controlled_vs_baseline_no_context | sentence_score | 0.0927 | 0.1050 |
| controlled_vs_baseline_no_context | overall_quality | 0.0793 | 0.3213 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.0791 | 2.0199 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1241 | 0.7879 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | -0.0188 | -0.0211 |
| controlled_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.0694 | 0.5209 |
| controlled_vs_baseline_no_context_phi3_latest | lore_consistency | 0.0413 | 2.3879 |
| controlled_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0.1250 | 0.1429 |
| controlled_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.0438 | 1.0860 |
| controlled_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 0.0326 | 0.1174 |
| controlled_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 0.0126 | 0.0439 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.1033 | 2.6194 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0228 | 0.5914 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1572 | 2.7052 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | -0.0081 | -0.0146 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0476 | -0.0485 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | -0.0433 | -0.0783 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.0899 | 0.1015 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.0797 | 0.3234 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0119 | 0.1006 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0031 | 0.0111 |
| controlled_alt_vs_controlled_default | naturalness | -0.0034 | -0.0038 |
| controlled_alt_vs_controlled_default | quest_state_correctness | 0.0094 | 0.0464 |
| controlled_alt_vs_controlled_default | lore_consistency | 0.0014 | 0.0241 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | -0.1250 | -0.1250 |
| controlled_alt_vs_controlled_default | objective_completion_support | -0.0033 | -0.0389 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | -0.0023 | -0.0076 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | -0.0027 | -0.0091 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0139 | 0.0974 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0073 | 0.1180 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0022 | 0.0104 |
| controlled_alt_vs_controlled_default | persona_style | 0.0067 | 0.0122 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0027 | 0.0029 |
| controlled_alt_vs_controlled_default | length_score | -0.0160 | -0.0313 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0125 | -0.0128 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0058 | 0.0178 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.0431 | 0.4950 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.0771 | 0.3714 |
| controlled_alt_vs_proposed_raw | naturalness | -0.0031 | -0.0036 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 0.0371 | 0.2121 |
| controlled_alt_vs_proposed_raw | lore_consistency | 0.0174 | 0.4080 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_proposed_raw | objective_completion_support | 0.0126 | 0.1850 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 0.0127 | 0.0431 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | 0.0049 | 0.0169 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.0547 | 0.5365 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0161 | 0.3072 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.0918 | 0.7297 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0186 | 0.0347 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0041 | -0.0044 |
| controlled_alt_vs_proposed_raw | length_score | -0.0255 | -0.0491 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0365 | 0.0393 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.0479 | 0.1688 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.1007 | 3.4151 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.0555 | 0.2421 |
| controlled_alt_vs_candidate_no_context | naturalness | -0.0068 | -0.0078 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 0.0863 | 0.6862 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 0.0547 | 10.3043 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | -0.1250 | -0.1250 |
| controlled_alt_vs_candidate_no_context | objective_completion_support | 0.0077 | 0.1057 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 0.0269 | 0.0958 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | -0.0013 | -0.0042 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.1286 | 4.5930 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0357 | 1.0824 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.0654 | 0.4303 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0158 | 0.0294 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0063 | -0.0067 |
| controlled_alt_vs_candidate_no_context | length_score | -0.0285 | -0.0545 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.0146 | 0.0154 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.0658 | 0.2472 |
| controlled_alt_vs_baseline_no_context | context_relevance | 0.0912 | 2.3379 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 0.1283 | 0.8203 |
| controlled_alt_vs_baseline_no_context | naturalness | -0.0269 | -0.0300 |
| controlled_alt_vs_baseline_no_context | quest_state_correctness | 0.0794 | 0.5986 |
| controlled_alt_vs_baseline_no_context | lore_consistency | 0.0412 | 2.1861 |
| controlled_alt_vs_baseline_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_baseline_no_context | objective_completion_support | 0.0421 | 1.0862 |
| controlled_alt_vs_baseline_no_context | gameplay_usefulness | 0.0284 | 0.1017 |
| controlled_alt_vs_baseline_no_context | time_pressure_acceptability | 0.0105 | 0.0365 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 0.1178 | 3.0366 |
| controlled_alt_vs_baseline_no_context | context_overlap | 0.0292 | 0.7377 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 0.1615 | 2.8808 |
| controlled_alt_vs_baseline_no_context | persona_style | -0.0042 | -0.0075 |
| controlled_alt_vs_baseline_no_context | distinct1 | -0.0454 | -0.0462 |
| controlled_alt_vs_baseline_no_context | length_score | -0.0833 | -0.1445 |
| controlled_alt_vs_baseline_no_context | sentence_score | 0.0802 | 0.0908 |
| controlled_alt_vs_baseline_no_context | overall_quality | 0.0851 | 0.3449 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_relevance | 0.0910 | 2.3238 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1273 | 0.8078 |
| controlled_alt_vs_baseline_no_context_phi3_latest | naturalness | -0.0221 | -0.0248 |
| controlled_alt_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.0788 | 0.5914 |
| controlled_alt_vs_baseline_no_context_phi3_latest | lore_consistency | 0.0427 | 2.4697 |
| controlled_alt_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.0406 | 1.0049 |
| controlled_alt_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 0.0302 | 0.1089 |
| controlled_alt_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 0.0099 | 0.0344 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.1172 | 2.9720 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_overlap | 0.0301 | 0.7792 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1594 | 2.7439 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_style | -0.0014 | -0.0025 |
| controlled_alt_vs_baseline_no_context_phi3_latest | distinct1 | -0.0449 | -0.0457 |
| controlled_alt_vs_baseline_no_context_phi3_latest | length_score | -0.0593 | -0.1072 |
| controlled_alt_vs_baseline_no_context_phi3_latest | sentence_score | 0.0774 | 0.0874 |
| controlled_alt_vs_baseline_no_context_phi3_latest | overall_quality | 0.0855 | 0.3470 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | context_relevance | 0.0912 | 2.3379 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | persona_consistency | 0.1283 | 0.8203 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | naturalness | -0.0269 | -0.0300 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | quest_state_correctness | 0.0794 | 0.5986 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | lore_consistency | 0.0412 | 2.1861 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | objective_completion_support | 0.0421 | 1.0862 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | gameplay_usefulness | 0.0284 | 0.1017 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | time_pressure_acceptability | 0.0105 | 0.0365 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | context_keyword_coverage | 0.1178 | 3.0366 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | context_overlap | 0.0292 | 0.7377 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | persona_keyword_coverage | 0.1615 | 2.8808 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | persona_style | -0.0042 | -0.0075 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | distinct1 | -0.0454 | -0.0462 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | length_score | -0.0833 | -0.1445 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | sentence_score | 0.0802 | 0.0908 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | overall_quality | 0.0851 | 0.3449 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | context_relevance | 0.0910 | 2.3238 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1273 | 0.8078 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | naturalness | -0.0221 | -0.0248 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.0788 | 0.5914 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | lore_consistency | 0.0427 | 2.4697 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.0406 | 1.0049 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 0.0302 | 0.1089 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 0.0099 | 0.0344 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.1172 | 2.9720 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | context_overlap | 0.0301 | 0.7792 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1594 | 2.7439 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | persona_style | -0.0014 | -0.0025 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | distinct1 | -0.0449 | -0.0457 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | length_score | -0.0593 | -0.1072 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | sentence_score | 0.0774 | 0.0874 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | overall_quality | 0.0855 | 0.3470 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0576 | (0.0405, 0.0751) | 0.0000 | 0.0576 | (0.0339, 0.0809) | 0.0000 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0216 | (-0.0399, -0.0040) | 0.9923 | -0.0216 | (-0.0402, -0.0049) | 0.9953 |
| proposed_vs_candidate_no_context | naturalness | -0.0037 | (-0.0113, 0.0037) | 0.8300 | -0.0037 | (-0.0109, 0.0039) | 0.8357 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.0492 | (0.0345, 0.0653) | 0.0000 | 0.0492 | (0.0276, 0.0700) | 0.0000 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0373 | (0.0265, 0.0493) | 0.0000 | 0.0373 | (0.0212, 0.0540) | 0.0000 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | -0.1250 | (-0.1806, -0.0764) | 1.0000 | -0.1250 | (-0.3750, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | -0.0049 | (-0.0143, 0.0040) | 0.8530 | -0.0049 | (-0.0121, 0.0033) | 0.8807 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 0.0142 | (0.0080, 0.0207) | 0.0000 | 0.0142 | (0.0069, 0.0206) | 0.0000 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | -0.0062 | (-0.0124, 0.0002) | 0.9707 | -0.0062 | (-0.0138, 0.0010) | 0.9530 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0739 | (0.0523, 0.0978) | 0.0000 | 0.0739 | (0.0437, 0.1032) | 0.0000 |
| proposed_vs_candidate_no_context | context_overlap | 0.0196 | (0.0139, 0.0256) | 0.0000 | 0.0196 | (0.0105, 0.0286) | 0.0000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0263 | (-0.0489, -0.0051) | 0.9937 | -0.0263 | (-0.0481, -0.0061) | 0.9967 |
| proposed_vs_candidate_no_context | persona_style | -0.0028 | (-0.0160, 0.0104) | 0.6590 | -0.0028 | (-0.0079, 0.0013) | 0.9003 |
| proposed_vs_candidate_no_context | distinct1 | -0.0022 | (-0.0086, 0.0043) | 0.7347 | -0.0022 | (-0.0078, 0.0032) | 0.7693 |
| proposed_vs_candidate_no_context | length_score | -0.0030 | (-0.0377, 0.0329) | 0.5817 | -0.0030 | (-0.0338, 0.0262) | 0.5610 |
| proposed_vs_candidate_no_context | sentence_score | -0.0219 | (-0.0486, 0.0049) | 0.9577 | -0.0219 | (-0.0340, -0.0073) | 0.9987 |
| proposed_vs_candidate_no_context | overall_quality | 0.0178 | (0.0075, 0.0284) | 0.0003 | 0.0178 | (0.0101, 0.0260) | 0.0000 |
| proposed_vs_baseline_no_context | context_relevance | 0.0481 | (0.0315, 0.0646) | 0.0000 | 0.0481 | (0.0211, 0.0737) | 0.0000 |
| proposed_vs_baseline_no_context | persona_consistency | 0.0512 | (0.0311, 0.0714) | 0.0000 | 0.0512 | (0.0191, 0.0835) | 0.0030 |
| proposed_vs_baseline_no_context | naturalness | -0.0237 | (-0.0338, -0.0133) | 1.0000 | -0.0237 | (-0.0321, -0.0162) | 1.0000 |
| proposed_vs_baseline_no_context | quest_state_correctness | 0.0423 | (0.0285, 0.0578) | 0.0000 | 0.0423 | (0.0204, 0.0627) | 0.0000 |
| proposed_vs_baseline_no_context | lore_consistency | 0.0238 | (0.0130, 0.0354) | 0.0000 | 0.0238 | (0.0080, 0.0401) | 0.0013 |
| proposed_vs_baseline_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_baseline_no_context | objective_completion_support | 0.0295 | (0.0176, 0.0419) | 0.0000 | 0.0295 | (0.0079, 0.0540) | 0.0037 |
| proposed_vs_baseline_no_context | gameplay_usefulness | 0.0157 | (0.0072, 0.0239) | 0.0000 | 0.0157 | (0.0046, 0.0280) | 0.0003 |
| proposed_vs_baseline_no_context | time_pressure_acceptability | 0.0055 | (-0.0027, 0.0140) | 0.0817 | 0.0055 | (-0.0104, 0.0212) | 0.2477 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0631 | (0.0420, 0.0858) | 0.0000 | 0.0631 | (0.0313, 0.0949) | 0.0000 |
| proposed_vs_baseline_no_context | context_overlap | 0.0130 | (0.0069, 0.0192) | 0.0000 | 0.0130 | (-0.0004, 0.0245) | 0.0287 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 0.0697 | (0.0446, 0.0948) | 0.0000 | 0.0697 | (0.0256, 0.1076) | 0.0017 |
| proposed_vs_baseline_no_context | persona_style | -0.0228 | (-0.0411, -0.0048) | 0.9927 | -0.0228 | (-0.0664, 0.0121) | 0.8727 |
| proposed_vs_baseline_no_context | distinct1 | -0.0413 | (-0.0504, -0.0325) | 1.0000 | -0.0413 | (-0.0597, -0.0266) | 1.0000 |
| proposed_vs_baseline_no_context | length_score | -0.0579 | (-0.1005, -0.0139) | 0.9963 | -0.0579 | (-0.0979, -0.0178) | 0.9980 |
| proposed_vs_baseline_no_context | sentence_score | 0.0437 | (0.0097, 0.0802) | 0.0080 | 0.0437 | (-0.0097, 0.0972) | 0.0613 |
| proposed_vs_baseline_no_context | overall_quality | 0.0372 | (0.0263, 0.0481) | 0.0000 | 0.0372 | (0.0167, 0.0555) | 0.0003 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 0.0479 | (0.0314, 0.0652) | 0.0000 | 0.0479 | (0.0217, 0.0712) | 0.0003 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | 0.0501 | (0.0276, 0.0734) | 0.0000 | 0.0501 | (0.0095, 0.0892) | 0.0073 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | -0.0190 | (-0.0284, -0.0100) | 1.0000 | -0.0190 | (-0.0281, -0.0108) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.0417 | (0.0277, 0.0565) | 0.0000 | 0.0417 | (0.0203, 0.0624) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | lore_consistency | 0.0253 | (0.0145, 0.0368) | 0.0000 | 0.0253 | (0.0101, 0.0408) | 0.0003 |
| proposed_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.0279 | (0.0149, 0.0407) | 0.0000 | 0.0279 | (0.0100, 0.0487) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 0.0175 | (0.0088, 0.0262) | 0.0000 | 0.0175 | (0.0077, 0.0275) | 0.0003 |
| proposed_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 0.0050 | (-0.0033, 0.0135) | 0.1267 | 0.0050 | (-0.0078, 0.0185) | 0.2443 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.0625 | (0.0414, 0.0846) | 0.0000 | 0.0625 | (0.0332, 0.0918) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 0.0139 | (0.0077, 0.0201) | 0.0000 | 0.0139 | (-0.0002, 0.0265) | 0.0267 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.0677 | (0.0408, 0.0961) | 0.0000 | 0.0677 | (0.0148, 0.1154) | 0.0037 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | -0.0200 | (-0.0349, -0.0058) | 0.9977 | -0.0200 | (-0.0538, 0.0072) | 0.8940 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | -0.0408 | (-0.0497, -0.0319) | 1.0000 | -0.0408 | (-0.0592, -0.0261) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | -0.0338 | (-0.0731, 0.0049) | 0.9543 | -0.0338 | (-0.0688, -0.0076) | 0.9983 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | 0.0410 | (0.0066, 0.0750) | 0.0113 | 0.0410 | (-0.0097, 0.0969) | 0.0627 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | 0.0375 | (0.0256, 0.0497) | 0.0000 | 0.0375 | (0.0145, 0.0567) | 0.0020 |
| controlled_vs_proposed_raw | context_relevance | 0.0312 | (0.0068, 0.0562) | 0.0063 | 0.0312 | (0.0076, 0.0522) | 0.0050 |
| controlled_vs_proposed_raw | persona_consistency | 0.0740 | (0.0518, 0.0955) | 0.0000 | 0.0740 | (0.0378, 0.1150) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0002 | (-0.0084, 0.0090) | 0.4753 | 0.0002 | (-0.0088, 0.0094) | 0.4803 |
| controlled_vs_proposed_raw | quest_state_correctness | 0.0277 | (0.0067, 0.0494) | 0.0053 | 0.0277 | (0.0037, 0.0487) | 0.0127 |
| controlled_vs_proposed_raw | lore_consistency | 0.0160 | (-0.0030, 0.0358) | 0.0497 | 0.0160 | (0.0024, 0.0309) | 0.0120 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0.1250 | (0.0764, 0.1806) | 0.0000 | 0.1250 | (0.0000, 0.3750) | 0.3403 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0159 | (0.0031, 0.0288) | 0.0097 | 0.0159 | (0.0022, 0.0318) | 0.0093 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0151 | (0.0069, 0.0237) | 0.0003 | 0.0151 | (0.0107, 0.0192) | 0.0000 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 0.0077 | (-0.0013, 0.0168) | 0.0450 | 0.0077 | (-0.0021, 0.0185) | 0.0677 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.0408 | (0.0072, 0.0734) | 0.0077 | 0.0408 | (0.0097, 0.0682) | 0.0067 |
| controlled_vs_proposed_raw | context_overlap | 0.0089 | (-0.0010, 0.0206) | 0.0407 | 0.0089 | (-0.0022, 0.0194) | 0.0567 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.0895 | (0.0620, 0.1152) | 0.0000 | 0.0895 | (0.0480, 0.1368) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.0119 | (-0.0005, 0.0254) | 0.0320 | 0.0119 | (-0.0015, 0.0302) | 0.0503 |
| controlled_vs_proposed_raw | distinct1 | -0.0068 | (-0.0155, 0.0022) | 0.9333 | -0.0068 | (-0.0222, 0.0073) | 0.8050 |
| controlled_vs_proposed_raw | length_score | -0.0095 | (-0.0488, 0.0303) | 0.6847 | -0.0095 | (-0.0620, 0.0285) | 0.6110 |
| controlled_vs_proposed_raw | sentence_score | 0.0490 | (0.0205, 0.0753) | 0.0003 | 0.0490 | (0.0274, 0.0753) | 0.0000 |
| controlled_vs_proposed_raw | overall_quality | 0.0421 | (0.0276, 0.0571) | 0.0000 | 0.0421 | (0.0286, 0.0549) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.0888 | (0.0656, 0.1143) | 0.0000 | 0.0888 | (0.0723, 0.1060) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.0524 | (0.0293, 0.0750) | 0.0000 | 0.0524 | (0.0258, 0.0833) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | -0.0035 | (-0.0126, 0.0061) | 0.7787 | -0.0035 | (-0.0156, 0.0083) | 0.7163 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.0769 | (0.0571, 0.0981) | 0.0000 | 0.0769 | (0.0622, 0.0911) | 0.0000 |
| controlled_vs_candidate_no_context | lore_consistency | 0.0533 | (0.0374, 0.0709) | 0.0000 | 0.0533 | (0.0407, 0.0639) | 0.0000 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | 0.0110 | (-0.0011, 0.0236) | 0.0427 | 0.0110 | (-0.0025, 0.0264) | 0.0617 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0292 | (0.0222, 0.0368) | 0.0000 | 0.0292 | (0.0241, 0.0344) | 0.0000 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 0.0015 | (-0.0070, 0.0103) | 0.3647 | 0.0015 | (-0.0035, 0.0065) | 0.2903 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.1147 | (0.0844, 0.1477) | 0.0000 | 0.1147 | (0.0920, 0.1366) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0285 | (0.0191, 0.0383) | 0.0000 | 0.0285 | (0.0240, 0.0326) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.0632 | (0.0350, 0.0913) | 0.0000 | 0.0632 | (0.0317, 0.0979) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.0091 | (-0.0033, 0.0230) | 0.0790 | 0.0091 | (-0.0023, 0.0235) | 0.0847 |
| controlled_vs_candidate_no_context | distinct1 | -0.0090 | (-0.0173, -0.0011) | 0.9873 | -0.0090 | (-0.0239, 0.0037) | 0.9023 |
| controlled_vs_candidate_no_context | length_score | -0.0125 | (-0.0523, 0.0280) | 0.7187 | -0.0125 | (-0.0722, 0.0426) | 0.6377 |
| controlled_vs_candidate_no_context | sentence_score | 0.0271 | (0.0007, 0.0535) | 0.0207 | 0.0271 | (0.0028, 0.0538) | 0.0127 |
| controlled_vs_candidate_no_context | overall_quality | 0.0600 | (0.0445, 0.0763) | 0.0000 | 0.0600 | (0.0472, 0.0749) | 0.0000 |
| controlled_vs_baseline_no_context | context_relevance | 0.0793 | (0.0543, 0.1057) | 0.0000 | 0.0793 | (0.0590, 0.1047) | 0.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1252 | (0.1042, 0.1457) | 0.0000 | 0.1252 | (0.0737, 0.1667) | 0.0000 |
| controlled_vs_baseline_no_context | naturalness | -0.0235 | (-0.0338, -0.0123) | 1.0000 | -0.0235 | (-0.0354, -0.0117) | 1.0000 |
| controlled_vs_baseline_no_context | quest_state_correctness | 0.0700 | (0.0492, 0.0920) | 0.0000 | 0.0700 | (0.0521, 0.0900) | 0.0000 |
| controlled_vs_baseline_no_context | lore_consistency | 0.0398 | (0.0228, 0.0584) | 0.0000 | 0.0398 | (0.0289, 0.0503) | 0.0000 |
| controlled_vs_baseline_no_context | multi_turn_contradiction_safety | 0.1250 | (0.0764, 0.1806) | 0.0000 | 0.1250 | (0.0000, 0.3750) | 0.3467 |
| controlled_vs_baseline_no_context | objective_completion_support | 0.0454 | (0.0326, 0.0578) | 0.0000 | 0.0454 | (0.0241, 0.0664) | 0.0000 |
| controlled_vs_baseline_no_context | gameplay_usefulness | 0.0307 | (0.0225, 0.0389) | 0.0000 | 0.0307 | (0.0201, 0.0423) | 0.0000 |
| controlled_vs_baseline_no_context | time_pressure_acceptability | 0.0132 | (0.0044, 0.0216) | 0.0013 | 0.0132 | (0.0023, 0.0241) | 0.0083 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.1039 | (0.0705, 0.1373) | 0.0000 | 0.1039 | (0.0764, 0.1362) | 0.0000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0219 | (0.0125, 0.0328) | 0.0000 | 0.0219 | (0.0163, 0.0279) | 0.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1592 | (0.1340, 0.1860) | 0.0000 | 0.1592 | (0.0954, 0.2075) | 0.0000 |
| controlled_vs_baseline_no_context | persona_style | -0.0109 | (-0.0296, 0.0074) | 0.8867 | -0.0109 | (-0.0506, 0.0266) | 0.6997 |
| controlled_vs_baseline_no_context | distinct1 | -0.0481 | (-0.0569, -0.0394) | 1.0000 | -0.0481 | (-0.0633, -0.0350) | 1.0000 |
| controlled_vs_baseline_no_context | length_score | -0.0674 | (-0.1157, -0.0208) | 0.9983 | -0.0674 | (-0.1377, -0.0035) | 0.9797 |
| controlled_vs_baseline_no_context | sentence_score | 0.0927 | (0.0615, 0.1243) | 0.0000 | 0.0927 | (0.0490, 0.1340) | 0.0000 |
| controlled_vs_baseline_no_context | overall_quality | 0.0793 | (0.0638, 0.0947) | 0.0000 | 0.0793 | (0.0562, 0.1005) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.0791 | (0.0550, 0.1048) | 0.0000 | 0.0791 | (0.0622, 0.1025) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1241 | (0.1019, 0.1478) | 0.0000 | 0.1241 | (0.0698, 0.1623) | 0.0003 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | -0.0188 | (-0.0290, -0.0084) | 1.0000 | -0.0188 | (-0.0274, -0.0091) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.0694 | (0.0480, 0.0917) | 0.0000 | 0.0694 | (0.0553, 0.0883) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | lore_consistency | 0.0413 | (0.0237, 0.0600) | 0.0000 | 0.0413 | (0.0306, 0.0500) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0.1250 | (0.0764, 0.1806) | 0.0000 | 0.1250 | (0.0000, 0.3750) | 0.3320 |
| controlled_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.0438 | (0.0312, 0.0558) | 0.0000 | 0.0438 | (0.0235, 0.0629) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 0.0326 | (0.0235, 0.0412) | 0.0000 | 0.0326 | (0.0225, 0.0423) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 0.0126 | (0.0043, 0.0207) | 0.0020 | 0.0126 | (0.0022, 0.0222) | 0.0087 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.1033 | (0.0713, 0.1358) | 0.0000 | 0.1033 | (0.0798, 0.1319) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0228 | (0.0129, 0.0338) | 0.0000 | 0.0228 | (0.0176, 0.0278) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1572 | (0.1290, 0.1849) | 0.0000 | 0.1572 | (0.0868, 0.2037) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | -0.0081 | (-0.0243, 0.0086) | 0.8370 | -0.0081 | (-0.0394, 0.0221) | 0.6910 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0476 | (-0.0562, -0.0386) | 1.0000 | -0.0476 | (-0.0636, -0.0335) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | -0.0433 | (-0.0903, 0.0067) | 0.9567 | -0.0433 | (-0.1030, 0.0086) | 0.9407 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.0899 | (0.0587, 0.1215) | 0.0000 | 0.0899 | (0.0385, 0.1372) | 0.0007 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.0797 | (0.0633, 0.0961) | 0.0000 | 0.0797 | (0.0562, 0.0965) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0119 | (-0.0193, 0.0411) | 0.2177 | 0.0119 | (-0.0195, 0.0468) | 0.2750 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0031 | (-0.0188, 0.0275) | 0.3923 | 0.0031 | (-0.0167, 0.0262) | 0.4257 |
| controlled_alt_vs_controlled_default | naturalness | -0.0034 | (-0.0123, 0.0059) | 0.7787 | -0.0034 | (-0.0117, 0.0051) | 0.7973 |
| controlled_alt_vs_controlled_default | quest_state_correctness | 0.0094 | (-0.0160, 0.0350) | 0.2317 | 0.0094 | (-0.0179, 0.0390) | 0.2827 |
| controlled_alt_vs_controlled_default | lore_consistency | 0.0014 | (-0.0201, 0.0221) | 0.4553 | 0.0014 | (-0.0212, 0.0270) | 0.4747 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | -0.1250 | (-0.1806, -0.0764) | 1.0000 | -0.1250 | (-0.3750, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | objective_completion_support | -0.0033 | (-0.0140, 0.0073) | 0.7183 | -0.0033 | (-0.0125, 0.0068) | 0.7377 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | -0.0023 | (-0.0103, 0.0049) | 0.7203 | -0.0023 | (-0.0125, 0.0093) | 0.6723 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | -0.0027 | (-0.0116, 0.0060) | 0.7173 | -0.0027 | (-0.0108, 0.0062) | 0.7493 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0139 | (-0.0244, 0.0515) | 0.2303 | 0.0139 | (-0.0271, 0.0628) | 0.2580 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0073 | (-0.0058, 0.0204) | 0.1413 | 0.0073 | (-0.0021, 0.0161) | 0.0660 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0022 | (-0.0255, 0.0324) | 0.4440 | 0.0022 | (-0.0229, 0.0318) | 0.4443 |
| controlled_alt_vs_controlled_default | persona_style | 0.0067 | (-0.0063, 0.0192) | 0.1430 | 0.0067 | (0.0012, 0.0145) | 0.0037 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0027 | (-0.0045, 0.0104) | 0.2270 | 0.0027 | (-0.0042, 0.0109) | 0.2423 |
| controlled_alt_vs_controlled_default | length_score | -0.0160 | (-0.0563, 0.0218) | 0.7980 | -0.0160 | (-0.0507, 0.0194) | 0.8173 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0125 | (-0.0344, 0.0090) | 0.8787 | -0.0125 | (-0.0292, 0.0024) | 0.9570 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0058 | (-0.0120, 0.0224) | 0.2703 | 0.0058 | (-0.0075, 0.0203) | 0.2153 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.0431 | (0.0196, 0.0683) | 0.0000 | 0.0431 | (0.0246, 0.0668) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.0771 | (0.0567, 0.1000) | 0.0000 | 0.0771 | (0.0473, 0.1077) | 0.0000 |
| controlled_alt_vs_proposed_raw | naturalness | -0.0031 | (-0.0122, 0.0057) | 0.7657 | -0.0031 | (-0.0110, 0.0047) | 0.7777 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 0.0371 | (0.0177, 0.0570) | 0.0000 | 0.0371 | (0.0218, 0.0573) | 0.0000 |
| controlled_alt_vs_proposed_raw | lore_consistency | 0.0174 | (0.0006, 0.0339) | 0.0190 | 0.0174 | (-0.0008, 0.0425) | 0.0340 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_proposed_raw | objective_completion_support | 0.0126 | (0.0017, 0.0237) | 0.0130 | 0.0126 | (-0.0036, 0.0262) | 0.0580 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 0.0127 | (0.0055, 0.0201) | 0.0000 | 0.0127 | (0.0041, 0.0212) | 0.0013 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | 0.0049 | (-0.0027, 0.0134) | 0.1077 | 0.0049 | (-0.0044, 0.0140) | 0.1387 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.0547 | (0.0229, 0.0858) | 0.0000 | 0.0547 | (0.0321, 0.0860) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0161 | (0.0073, 0.0261) | 0.0000 | 0.0161 | (0.0066, 0.0264) | 0.0007 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.0918 | (0.0654, 0.1188) | 0.0000 | 0.0918 | (0.0574, 0.1257) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0186 | (0.0073, 0.0310) | 0.0007 | 0.0186 | (0.0037, 0.0358) | 0.0057 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0041 | (-0.0123, 0.0039) | 0.8360 | -0.0041 | (-0.0153, 0.0062) | 0.7760 |
| controlled_alt_vs_proposed_raw | length_score | -0.0255 | (-0.0660, 0.0160) | 0.8817 | -0.0255 | (-0.0704, 0.0104) | 0.8897 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0365 | (0.0073, 0.0681) | 0.0083 | 0.0365 | (0.0122, 0.0608) | 0.0000 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.0479 | (0.0345, 0.0620) | 0.0000 | 0.0479 | (0.0350, 0.0601) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.1007 | (0.0774, 0.1233) | 0.0000 | 0.1007 | (0.0682, 0.1377) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.0555 | (0.0305, 0.0804) | 0.0000 | 0.0555 | (0.0280, 0.0839) | 0.0000 |
| controlled_alt_vs_candidate_no_context | naturalness | -0.0068 | (-0.0154, 0.0021) | 0.9320 | -0.0068 | (-0.0143, 0.0004) | 0.9663 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 0.0863 | (0.0672, 0.1060) | 0.0000 | 0.0863 | (0.0598, 0.1175) | 0.0000 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 0.0547 | (0.0385, 0.0706) | 0.0000 | 0.0547 | (0.0266, 0.0865) | 0.0000 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | -0.1250 | (-0.1806, -0.0694) | 1.0000 | -0.1250 | (-0.3750, 0.0000) | 1.0000 |
| controlled_alt_vs_candidate_no_context | objective_completion_support | 0.0077 | (-0.0046, 0.0205) | 0.1003 | 0.0077 | (-0.0091, 0.0243) | 0.1910 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 0.0269 | (0.0200, 0.0337) | 0.0000 | 0.0269 | (0.0157, 0.0367) | 0.0000 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | -0.0013 | (-0.0099, 0.0075) | 0.6280 | -0.0013 | (-0.0086, 0.0083) | 0.6513 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.1286 | (0.0986, 0.1597) | 0.0000 | 0.1286 | (0.0854, 0.1806) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0357 | (0.0262, 0.0459) | 0.0000 | 0.0357 | (0.0266, 0.0442) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.0654 | (0.0343, 0.0952) | 0.0000 | 0.0654 | (0.0326, 0.0992) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0158 | (0.0008, 0.0308) | 0.0197 | 0.0158 | (0.0043, 0.0316) | 0.0193 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0063 | (-0.0138, 0.0014) | 0.9463 | -0.0063 | (-0.0151, 0.0029) | 0.9070 |
| controlled_alt_vs_candidate_no_context | length_score | -0.0285 | (-0.0697, 0.0100) | 0.9160 | -0.0285 | (-0.0720, 0.0081) | 0.9250 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.0146 | (-0.0122, 0.0389) | 0.1630 | 0.0146 | (-0.0122, 0.0389) | 0.1463 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.0658 | (0.0513, 0.0818) | 0.0000 | 0.0658 | (0.0492, 0.0833) | 0.0000 |
| controlled_alt_vs_baseline_no_context | context_relevance | 0.0912 | (0.0682, 0.1155) | 0.0000 | 0.0912 | (0.0542, 0.1338) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 0.1283 | (0.1054, 0.1505) | 0.0000 | 0.1283 | (0.0985, 0.1572) | 0.0000 |
| controlled_alt_vs_baseline_no_context | naturalness | -0.0269 | (-0.0382, -0.0158) | 1.0000 | -0.0269 | (-0.0370, -0.0156) | 1.0000 |
| controlled_alt_vs_baseline_no_context | quest_state_correctness | 0.0794 | (0.0604, 0.0986) | 0.0000 | 0.0794 | (0.0489, 0.1143) | 0.0000 |
| controlled_alt_vs_baseline_no_context | lore_consistency | 0.0412 | (0.0258, 0.0574) | 0.0000 | 0.0412 | (0.0147, 0.0722) | 0.0000 |
| controlled_alt_vs_baseline_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_baseline_no_context | objective_completion_support | 0.0421 | (0.0301, 0.0553) | 0.0000 | 0.0421 | (0.0227, 0.0672) | 0.0000 |
| controlled_alt_vs_baseline_no_context | gameplay_usefulness | 0.0284 | (0.0205, 0.0363) | 0.0000 | 0.0284 | (0.0174, 0.0401) | 0.0000 |
| controlled_alt_vs_baseline_no_context | time_pressure_acceptability | 0.0105 | (0.0014, 0.0196) | 0.0120 | 0.0105 | (-0.0015, 0.0268) | 0.0533 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 0.1178 | (0.0883, 0.1481) | 0.0000 | 0.1178 | (0.0706, 0.1743) | 0.0000 |
| controlled_alt_vs_baseline_no_context | context_overlap | 0.0292 | (0.0201, 0.0395) | 0.0000 | 0.0292 | (0.0177, 0.0408) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 0.1615 | (0.1355, 0.1890) | 0.0000 | 0.1615 | (0.1252, 0.1932) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_style | -0.0042 | (-0.0211, 0.0126) | 0.6713 | -0.0042 | (-0.0419, 0.0285) | 0.5900 |
| controlled_alt_vs_baseline_no_context | distinct1 | -0.0454 | (-0.0534, -0.0372) | 1.0000 | -0.0454 | (-0.0589, -0.0335) | 1.0000 |
| controlled_alt_vs_baseline_no_context | length_score | -0.0833 | (-0.1319, -0.0331) | 0.9993 | -0.0833 | (-0.1412, -0.0301) | 0.9997 |
| controlled_alt_vs_baseline_no_context | sentence_score | 0.0802 | (0.0510, 0.1094) | 0.0000 | 0.0802 | (0.0340, 0.1264) | 0.0007 |
| controlled_alt_vs_baseline_no_context | overall_quality | 0.0851 | (0.0705, 0.1001) | 0.0000 | 0.0851 | (0.0623, 0.1084) | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_relevance | 0.0910 | (0.0695, 0.1137) | 0.0000 | 0.0910 | (0.0575, 0.1297) | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1273 | (0.1021, 0.1513) | 0.0000 | 0.1273 | (0.0940, 0.1550) | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | naturalness | -0.0221 | (-0.0329, -0.0110) | 1.0000 | -0.0221 | (-0.0323, -0.0105) | 1.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.0788 | (0.0605, 0.0970) | 0.0000 | 0.0788 | (0.0491, 0.1104) | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | lore_consistency | 0.0427 | (0.0279, 0.0583) | 0.0000 | 0.0427 | (0.0170, 0.0715) | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.0406 | (0.0281, 0.0528) | 0.0000 | 0.0406 | (0.0227, 0.0616) | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 0.0302 | (0.0220, 0.0382) | 0.0000 | 0.0302 | (0.0181, 0.0419) | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 0.0099 | (-0.0001, 0.0191) | 0.0260 | 0.0099 | (-0.0015, 0.0238) | 0.0483 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.1172 | (0.0897, 0.1461) | 0.0000 | 0.1172 | (0.0716, 0.1679) | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_overlap | 0.0301 | (0.0204, 0.0403) | 0.0000 | 0.0301 | (0.0195, 0.0415) | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1594 | (0.1289, 0.1896) | 0.0000 | 0.1594 | (0.1178, 0.1903) | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_style | -0.0014 | (-0.0166, 0.0143) | 0.5637 | -0.0014 | (-0.0293, 0.0245) | 0.5080 |
| controlled_alt_vs_baseline_no_context_phi3_latest | distinct1 | -0.0449 | (-0.0537, -0.0363) | 1.0000 | -0.0449 | (-0.0586, -0.0329) | 1.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | length_score | -0.0593 | (-0.1069, -0.0130) | 0.9927 | -0.0593 | (-0.1104, -0.0164) | 0.9987 |
| controlled_alt_vs_baseline_no_context_phi3_latest | sentence_score | 0.0774 | (0.0483, 0.1108) | 0.0000 | 0.0774 | (0.0292, 0.1278) | 0.0007 |
| controlled_alt_vs_baseline_no_context_phi3_latest | overall_quality | 0.0855 | (0.0706, 0.1017) | 0.0000 | 0.0855 | (0.0629, 0.1064) | 0.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | context_relevance | 0.0912 | (0.0685, 0.1151) | 0.0000 | 0.0912 | (0.0553, 0.1318) | 0.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | persona_consistency | 0.1283 | (0.1066, 0.1512) | 0.0000 | 0.1283 | (0.0967, 0.1588) | 0.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | naturalness | -0.0269 | (-0.0379, -0.0157) | 1.0000 | -0.0269 | (-0.0372, -0.0155) | 1.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | quest_state_correctness | 0.0794 | (0.0613, 0.0997) | 0.0000 | 0.0794 | (0.0496, 0.1112) | 0.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | lore_consistency | 0.0412 | (0.0262, 0.0572) | 0.0000 | 0.0412 | (0.0144, 0.0742) | 0.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | objective_completion_support | 0.0421 | (0.0302, 0.0550) | 0.0000 | 0.0421 | (0.0231, 0.0685) | 0.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | gameplay_usefulness | 0.0284 | (0.0203, 0.0359) | 0.0000 | 0.0284 | (0.0166, 0.0399) | 0.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | time_pressure_acceptability | 0.0105 | (0.0014, 0.0196) | 0.0117 | 0.0105 | (-0.0019, 0.0269) | 0.0677 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | context_keyword_coverage | 0.1178 | (0.0882, 0.1492) | 0.0000 | 0.1178 | (0.0706, 0.1702) | 0.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | context_overlap | 0.0292 | (0.0198, 0.0391) | 0.0000 | 0.0292 | (0.0179, 0.0409) | 0.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | persona_keyword_coverage | 0.1615 | (0.1353, 0.1895) | 0.0000 | 0.1615 | (0.1254, 0.1921) | 0.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | persona_style | -0.0042 | (-0.0228, 0.0129) | 0.6897 | -0.0042 | (-0.0442, 0.0288) | 0.5703 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | distinct1 | -0.0454 | (-0.0540, -0.0373) | 1.0000 | -0.0454 | (-0.0584, -0.0329) | 1.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | length_score | -0.0833 | (-0.1340, -0.0345) | 0.9997 | -0.0833 | (-0.1435, -0.0299) | 1.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | sentence_score | 0.0802 | (0.0486, 0.1094) | 0.0000 | 0.0802 | (0.0340, 0.1264) | 0.0003 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | overall_quality | 0.0851 | (0.0703, 0.0996) | 0.0000 | 0.0851 | (0.0626, 0.1073) | 0.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | context_relevance | 0.0910 | (0.0686, 0.1139) | 0.0000 | 0.0910 | (0.0563, 0.1303) | 0.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1273 | (0.1032, 0.1525) | 0.0000 | 0.1273 | (0.0943, 0.1533) | 0.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | naturalness | -0.0221 | (-0.0330, -0.0110) | 1.0000 | -0.0221 | (-0.0320, -0.0112) | 1.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.0788 | (0.0606, 0.0979) | 0.0000 | 0.0788 | (0.0497, 0.1114) | 0.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | lore_consistency | 0.0427 | (0.0278, 0.0579) | 0.0000 | 0.0427 | (0.0168, 0.0714) | 0.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.0406 | (0.0281, 0.0531) | 0.0000 | 0.0406 | (0.0223, 0.0616) | 0.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 0.0302 | (0.0221, 0.0381) | 0.0000 | 0.0302 | (0.0178, 0.0420) | 0.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 0.0099 | (0.0005, 0.0193) | 0.0217 | 0.0099 | (-0.0013, 0.0238) | 0.0490 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.1172 | (0.0890, 0.1465) | 0.0000 | 0.1172 | (0.0721, 0.1691) | 0.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | context_overlap | 0.0301 | (0.0203, 0.0412) | 0.0000 | 0.0301 | (0.0194, 0.0418) | 0.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1594 | (0.1293, 0.1908) | 0.0000 | 0.1594 | (0.1178, 0.1900) | 0.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | persona_style | -0.0014 | (-0.0172, 0.0145) | 0.5767 | -0.0014 | (-0.0306, 0.0231) | 0.5150 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | distinct1 | -0.0449 | (-0.0532, -0.0363) | 1.0000 | -0.0449 | (-0.0591, -0.0322) | 1.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | length_score | -0.0593 | (-0.1088, -0.0106) | 0.9913 | -0.0593 | (-0.1104, -0.0171) | 0.9993 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | sentence_score | 0.0774 | (0.0458, 0.1090) | 0.0000 | 0.0774 | (0.0316, 0.1281) | 0.0003 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | overall_quality | 0.0855 | (0.0705, 0.1005) | 0.0000 | 0.0855 | (0.0628, 0.1070) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 69 | 23 | 52 | 0.6597 | 0.7500 |
| proposed_vs_candidate_no_context | persona_consistency | 16 | 33 | 95 | 0.4410 | 0.3265 |
| proposed_vs_candidate_no_context | naturalness | 41 | 50 | 53 | 0.4688 | 0.4505 |
| proposed_vs_candidate_no_context | quest_state_correctness | 68 | 24 | 52 | 0.6528 | 0.7391 |
| proposed_vs_candidate_no_context | lore_consistency | 53 | 10 | 81 | 0.6493 | 0.8413 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 18 | 126 | 0.4375 | 0.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | 30 | 38 | 76 | 0.4722 | 0.4412 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 65 | 27 | 52 | 0.6319 | 0.7065 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | 33 | 50 | 61 | 0.4410 | 0.3976 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 55 | 11 | 78 | 0.6528 | 0.8333 |
| proposed_vs_candidate_no_context | context_overlap | 71 | 21 | 52 | 0.6736 | 0.7717 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 14 | 28 | 102 | 0.4514 | 0.3333 |
| proposed_vs_candidate_no_context | persona_style | 7 | 11 | 126 | 0.4861 | 0.3889 |
| proposed_vs_candidate_no_context | distinct1 | 34 | 47 | 63 | 0.4549 | 0.4198 |
| proposed_vs_candidate_no_context | length_score | 42 | 45 | 57 | 0.4896 | 0.4828 |
| proposed_vs_candidate_no_context | sentence_score | 11 | 20 | 113 | 0.4688 | 0.3548 |
| proposed_vs_candidate_no_context | overall_quality | 59 | 33 | 52 | 0.5903 | 0.6413 |
| proposed_vs_baseline_no_context | context_relevance | 87 | 56 | 1 | 0.6076 | 0.6084 |
| proposed_vs_baseline_no_context | persona_consistency | 59 | 26 | 59 | 0.6146 | 0.6941 |
| proposed_vs_baseline_no_context | naturalness | 49 | 94 | 1 | 0.3438 | 0.3427 |
| proposed_vs_baseline_no_context | quest_state_correctness | 89 | 54 | 1 | 0.6215 | 0.6224 |
| proposed_vs_baseline_no_context | lore_consistency | 51 | 40 | 53 | 0.5382 | 0.5604 |
| proposed_vs_baseline_no_context | multi_turn_contradiction_safety | 0 | 0 | 144 | 0.5000 | nan |
| proposed_vs_baseline_no_context | objective_completion_support | 71 | 40 | 33 | 0.6076 | 0.6396 |
| proposed_vs_baseline_no_context | gameplay_usefulness | 90 | 54 | 0 | 0.6250 | 0.6250 |
| proposed_vs_baseline_no_context | time_pressure_acceptability | 64 | 65 | 15 | 0.4965 | 0.4961 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 58 | 21 | 65 | 0.6285 | 0.7342 |
| proposed_vs_baseline_no_context | context_overlap | 85 | 57 | 2 | 0.5972 | 0.5986 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 56 | 14 | 74 | 0.6458 | 0.8000 |
| proposed_vs_baseline_no_context | persona_style | 7 | 23 | 114 | 0.4444 | 0.2333 |
| proposed_vs_baseline_no_context | distinct1 | 33 | 94 | 17 | 0.2882 | 0.2598 |
| proposed_vs_baseline_no_context | length_score | 53 | 81 | 10 | 0.4028 | 0.3955 |
| proposed_vs_baseline_no_context | sentence_score | 39 | 21 | 84 | 0.5625 | 0.6500 |
| proposed_vs_baseline_no_context | overall_quality | 105 | 39 | 0 | 0.7292 | 0.7292 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 85 | 58 | 1 | 0.5938 | 0.5944 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | 63 | 29 | 52 | 0.6181 | 0.6848 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | 48 | 95 | 1 | 0.3368 | 0.3357 |
| proposed_vs_baseline_no_context_phi3_latest | quest_state_correctness | 85 | 58 | 1 | 0.5938 | 0.5944 |
| proposed_vs_baseline_no_context_phi3_latest | lore_consistency | 53 | 46 | 45 | 0.5243 | 0.5354 |
| proposed_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0 | 0 | 144 | 0.5000 | nan |
| proposed_vs_baseline_no_context_phi3_latest | objective_completion_support | 68 | 37 | 39 | 0.6076 | 0.6476 |
| proposed_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 92 | 52 | 0 | 0.6389 | 0.6389 |
| proposed_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 65 | 56 | 23 | 0.5312 | 0.5372 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 57 | 18 | 69 | 0.6354 | 0.7600 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 83 | 59 | 2 | 0.5833 | 0.5845 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 59 | 20 | 65 | 0.6354 | 0.7468 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | 6 | 22 | 116 | 0.4444 | 0.2143 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | 27 | 93 | 24 | 0.2708 | 0.2250 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | 60 | 75 | 9 | 0.4479 | 0.4444 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | 35 | 18 | 91 | 0.5590 | 0.6604 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | 103 | 41 | 0 | 0.7153 | 0.7153 |
| controlled_vs_proposed_raw | context_relevance | 60 | 58 | 26 | 0.5069 | 0.5085 |
| controlled_vs_proposed_raw | persona_consistency | 74 | 9 | 61 | 0.7257 | 0.8916 |
| controlled_vs_proposed_raw | naturalness | 58 | 61 | 25 | 0.4896 | 0.4874 |
| controlled_vs_proposed_raw | quest_state_correctness | 60 | 58 | 26 | 0.5069 | 0.5085 |
| controlled_vs_proposed_raw | lore_consistency | 38 | 38 | 68 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 18 | 0 | 126 | 0.5625 | 1.0000 |
| controlled_vs_proposed_raw | objective_completion_support | 50 | 38 | 56 | 0.5417 | 0.5682 |
| controlled_vs_proposed_raw | gameplay_usefulness | 72 | 47 | 25 | 0.5868 | 0.6050 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 52 | 43 | 49 | 0.5312 | 0.5474 |
| controlled_vs_proposed_raw | context_keyword_coverage | 41 | 33 | 70 | 0.5278 | 0.5541 |
| controlled_vs_proposed_raw | context_overlap | 64 | 54 | 26 | 0.5347 | 0.5424 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 70 | 8 | 66 | 0.7153 | 0.8974 |
| controlled_vs_proposed_raw | persona_style | 15 | 8 | 121 | 0.5243 | 0.6522 |
| controlled_vs_proposed_raw | distinct1 | 55 | 59 | 30 | 0.4861 | 0.4825 |
| controlled_vs_proposed_raw | length_score | 56 | 61 | 27 | 0.4826 | 0.4786 |
| controlled_vs_proposed_raw | sentence_score | 27 | 6 | 111 | 0.5729 | 0.8182 |
| controlled_vs_proposed_raw | overall_quality | 86 | 33 | 25 | 0.6840 | 0.7227 |
| controlled_vs_candidate_no_context | context_relevance | 82 | 32 | 30 | 0.6736 | 0.7193 |
| controlled_vs_candidate_no_context | persona_consistency | 65 | 14 | 65 | 0.6771 | 0.8228 |
| controlled_vs_candidate_no_context | naturalness | 54 | 61 | 29 | 0.4757 | 0.4696 |
| controlled_vs_candidate_no_context | quest_state_correctness | 81 | 33 | 30 | 0.6667 | 0.7105 |
| controlled_vs_candidate_no_context | lore_consistency | 45 | 15 | 84 | 0.6042 | 0.7500 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 144 | 0.5000 | nan |
| controlled_vs_candidate_no_context | objective_completion_support | 44 | 38 | 62 | 0.5208 | 0.5366 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 90 | 25 | 29 | 0.7257 | 0.7826 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 47 | 48 | 49 | 0.4965 | 0.4947 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 60 | 10 | 74 | 0.6736 | 0.8571 |
| controlled_vs_candidate_no_context | context_overlap | 80 | 34 | 30 | 0.6597 | 0.7018 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 60 | 12 | 72 | 0.6667 | 0.8333 |
| controlled_vs_candidate_no_context | persona_style | 14 | 8 | 122 | 0.5208 | 0.6364 |
| controlled_vs_candidate_no_context | distinct1 | 54 | 56 | 34 | 0.4931 | 0.4909 |
| controlled_vs_candidate_no_context | length_score | 56 | 57 | 31 | 0.4965 | 0.4956 |
| controlled_vs_candidate_no_context | sentence_score | 20 | 8 | 116 | 0.5417 | 0.7143 |
| controlled_vs_candidate_no_context | overall_quality | 93 | 22 | 29 | 0.7465 | 0.8087 |
| controlled_vs_baseline_no_context | context_relevance | 78 | 66 | 0 | 0.5417 | 0.5417 |
| controlled_vs_baseline_no_context | persona_consistency | 112 | 14 | 18 | 0.8403 | 0.8889 |
| controlled_vs_baseline_no_context | naturalness | 50 | 94 | 0 | 0.3472 | 0.3472 |
| controlled_vs_baseline_no_context | quest_state_correctness | 81 | 63 | 0 | 0.5625 | 0.5625 |
| controlled_vs_baseline_no_context | lore_consistency | 47 | 48 | 49 | 0.4965 | 0.4947 |
| controlled_vs_baseline_no_context | multi_turn_contradiction_safety | 18 | 0 | 126 | 0.5625 | 1.0000 |
| controlled_vs_baseline_no_context | objective_completion_support | 85 | 32 | 27 | 0.6840 | 0.7265 |
| controlled_vs_baseline_no_context | gameplay_usefulness | 102 | 42 | 0 | 0.7083 | 0.7083 |
| controlled_vs_baseline_no_context | time_pressure_acceptability | 80 | 47 | 17 | 0.6146 | 0.6299 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 59 | 27 | 58 | 0.6111 | 0.6860 |
| controlled_vs_baseline_no_context | context_overlap | 80 | 63 | 1 | 0.5590 | 0.5594 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 112 | 10 | 22 | 0.8542 | 0.9180 |
| controlled_vs_baseline_no_context | persona_style | 11 | 16 | 117 | 0.4826 | 0.4074 |
| controlled_vs_baseline_no_context | distinct1 | 21 | 112 | 11 | 0.1840 | 0.1579 |
| controlled_vs_baseline_no_context | length_score | 58 | 80 | 6 | 0.4236 | 0.4203 |
| controlled_vs_baseline_no_context | sentence_score | 45 | 7 | 92 | 0.6319 | 0.8654 |
| controlled_vs_baseline_no_context | overall_quality | 119 | 25 | 0 | 0.8264 | 0.8264 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 78 | 64 | 2 | 0.5486 | 0.5493 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 109 | 12 | 23 | 0.8368 | 0.9008 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | 61 | 82 | 1 | 0.4271 | 0.4266 |
| controlled_vs_baseline_no_context_phi3_latest | quest_state_correctness | 80 | 62 | 2 | 0.5625 | 0.5634 |
| controlled_vs_baseline_no_context_phi3_latest | lore_consistency | 48 | 53 | 43 | 0.4826 | 0.4752 |
| controlled_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 18 | 0 | 126 | 0.5625 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | objective_completion_support | 82 | 33 | 29 | 0.6701 | 0.7130 |
| controlled_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 104 | 40 | 0 | 0.7222 | 0.7222 |
| controlled_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 80 | 46 | 18 | 0.6181 | 0.6349 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 59 | 26 | 59 | 0.6146 | 0.6941 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 80 | 62 | 2 | 0.5625 | 0.5634 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 109 | 9 | 26 | 0.8472 | 0.9237 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | 9 | 17 | 118 | 0.4722 | 0.3462 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | 23 | 113 | 8 | 0.1875 | 0.1691 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | 70 | 73 | 1 | 0.4896 | 0.4895 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 44 | 8 | 92 | 0.6250 | 0.8462 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 120 | 24 | 0 | 0.8333 | 0.8333 |
| controlled_alt_vs_controlled_default | context_relevance | 48 | 43 | 53 | 0.5174 | 0.5275 |
| controlled_alt_vs_controlled_default | persona_consistency | 24 | 23 | 97 | 0.5035 | 0.5106 |
| controlled_alt_vs_controlled_default | naturalness | 43 | 48 | 53 | 0.4826 | 0.4725 |
| controlled_alt_vs_controlled_default | quest_state_correctness | 47 | 44 | 53 | 0.5104 | 0.5165 |
| controlled_alt_vs_controlled_default | lore_consistency | 40 | 39 | 65 | 0.5035 | 0.5063 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0 | 18 | 126 | 0.4375 | 0.0000 |
| controlled_alt_vs_controlled_default | objective_completion_support | 31 | 38 | 75 | 0.4757 | 0.4493 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | 44 | 47 | 53 | 0.4896 | 0.4835 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | 37 | 46 | 61 | 0.4688 | 0.4458 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 44 | 34 | 66 | 0.5347 | 0.5641 |
| controlled_alt_vs_controlled_default | context_overlap | 46 | 45 | 53 | 0.5035 | 0.5055 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 18 | 22 | 104 | 0.4861 | 0.4500 |
| controlled_alt_vs_controlled_default | persona_style | 11 | 5 | 128 | 0.5208 | 0.6875 |
| controlled_alt_vs_controlled_default | distinct1 | 48 | 40 | 56 | 0.5278 | 0.5455 |
| controlled_alt_vs_controlled_default | length_score | 41 | 44 | 59 | 0.4896 | 0.4824 |
| controlled_alt_vs_controlled_default | sentence_score | 7 | 13 | 124 | 0.4792 | 0.3500 |
| controlled_alt_vs_controlled_default | overall_quality | 46 | 45 | 53 | 0.5035 | 0.5055 |
| controlled_alt_vs_proposed_raw | context_relevance | 74 | 44 | 26 | 0.6042 | 0.6271 |
| controlled_alt_vs_proposed_raw | persona_consistency | 75 | 11 | 58 | 0.7222 | 0.8721 |
| controlled_alt_vs_proposed_raw | naturalness | 58 | 60 | 26 | 0.4931 | 0.4915 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 73 | 45 | 26 | 0.5972 | 0.6186 |
| controlled_alt_vs_proposed_raw | lore_consistency | 49 | 32 | 63 | 0.5590 | 0.6049 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0 | 0 | 144 | 0.5000 | nan |
| controlled_alt_vs_proposed_raw | objective_completion_support | 46 | 37 | 61 | 0.5312 | 0.5542 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 83 | 36 | 25 | 0.6632 | 0.6975 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | 47 | 48 | 49 | 0.4965 | 0.4947 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 54 | 26 | 64 | 0.5972 | 0.6750 |
| controlled_alt_vs_proposed_raw | context_overlap | 68 | 50 | 26 | 0.5625 | 0.5763 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 70 | 10 | 64 | 0.7083 | 0.8750 |
| controlled_alt_vs_proposed_raw | persona_style | 17 | 5 | 122 | 0.5417 | 0.7727 |
| controlled_alt_vs_proposed_raw | distinct1 | 57 | 54 | 33 | 0.5104 | 0.5135 |
| controlled_alt_vs_proposed_raw | length_score | 54 | 58 | 32 | 0.4861 | 0.4821 |
| controlled_alt_vs_proposed_raw | sentence_score | 27 | 12 | 105 | 0.5521 | 0.6923 |
| controlled_alt_vs_proposed_raw | overall_quality | 91 | 28 | 25 | 0.7188 | 0.7647 |
| controlled_alt_vs_candidate_no_context | context_relevance | 89 | 28 | 27 | 0.7118 | 0.7607 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 66 | 17 | 61 | 0.6701 | 0.7952 |
| controlled_alt_vs_candidate_no_context | naturalness | 45 | 73 | 26 | 0.4028 | 0.3814 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 90 | 27 | 27 | 0.7188 | 0.7692 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 58 | 11 | 75 | 0.6632 | 0.8406 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 18 | 126 | 0.4375 | 0.0000 |
| controlled_alt_vs_candidate_no_context | objective_completion_support | 45 | 43 | 56 | 0.5069 | 0.5114 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 93 | 25 | 26 | 0.7361 | 0.7881 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | 42 | 61 | 41 | 0.4340 | 0.4078 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 75 | 10 | 59 | 0.7257 | 0.8824 |
| controlled_alt_vs_candidate_no_context | context_overlap | 86 | 31 | 27 | 0.6910 | 0.7350 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 59 | 16 | 69 | 0.6493 | 0.7867 |
| controlled_alt_vs_candidate_no_context | persona_style | 17 | 7 | 120 | 0.5347 | 0.7083 |
| controlled_alt_vs_candidate_no_context | distinct1 | 53 | 57 | 34 | 0.4861 | 0.4818 |
| controlled_alt_vs_candidate_no_context | length_score | 51 | 62 | 31 | 0.4618 | 0.4513 |
| controlled_alt_vs_candidate_no_context | sentence_score | 19 | 13 | 112 | 0.5208 | 0.5938 |
| controlled_alt_vs_candidate_no_context | overall_quality | 98 | 20 | 26 | 0.7708 | 0.8305 |
| controlled_alt_vs_baseline_no_context | context_relevance | 91 | 52 | 1 | 0.6354 | 0.6364 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 108 | 13 | 23 | 0.8299 | 0.8926 |
| controlled_alt_vs_baseline_no_context | naturalness | 47 | 97 | 0 | 0.3264 | 0.3264 |
| controlled_alt_vs_baseline_no_context | quest_state_correctness | 94 | 49 | 1 | 0.6562 | 0.6573 |
| controlled_alt_vs_baseline_no_context | lore_consistency | 57 | 45 | 42 | 0.5417 | 0.5588 |
| controlled_alt_vs_baseline_no_context | multi_turn_contradiction_safety | 0 | 0 | 144 | 0.5000 | nan |
| controlled_alt_vs_baseline_no_context | objective_completion_support | 83 | 31 | 30 | 0.6806 | 0.7281 |
| controlled_alt_vs_baseline_no_context | gameplay_usefulness | 107 | 37 | 0 | 0.7431 | 0.7431 |
| controlled_alt_vs_baseline_no_context | time_pressure_acceptability | 74 | 57 | 13 | 0.5590 | 0.5649 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 72 | 19 | 53 | 0.6840 | 0.7912 |
| controlled_alt_vs_baseline_no_context | context_overlap | 90 | 53 | 1 | 0.6285 | 0.6294 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 107 | 8 | 29 | 0.8438 | 0.9304 |
| controlled_alt_vs_baseline_no_context | persona_style | 12 | 13 | 119 | 0.4965 | 0.4800 |
| controlled_alt_vs_baseline_no_context | distinct1 | 21 | 113 | 10 | 0.1806 | 0.1567 |
| controlled_alt_vs_baseline_no_context | length_score | 60 | 79 | 5 | 0.4340 | 0.4317 |
| controlled_alt_vs_baseline_no_context | sentence_score | 40 | 7 | 97 | 0.6146 | 0.8511 |
| controlled_alt_vs_baseline_no_context | overall_quality | 121 | 23 | 0 | 0.8403 | 0.8403 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_relevance | 94 | 49 | 1 | 0.6562 | 0.6573 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_consistency | 109 | 15 | 20 | 0.8264 | 0.8790 |
| controlled_alt_vs_baseline_no_context_phi3_latest | naturalness | 55 | 89 | 0 | 0.3819 | 0.3819 |
| controlled_alt_vs_baseline_no_context_phi3_latest | quest_state_correctness | 95 | 48 | 1 | 0.6632 | 0.6643 |
| controlled_alt_vs_baseline_no_context_phi3_latest | lore_consistency | 60 | 44 | 40 | 0.5556 | 0.5769 |
| controlled_alt_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0 | 0 | 144 | 0.5000 | nan |
| controlled_alt_vs_baseline_no_context_phi3_latest | objective_completion_support | 82 | 30 | 32 | 0.6806 | 0.7321 |
| controlled_alt_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 102 | 42 | 0 | 0.7083 | 0.7083 |
| controlled_alt_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 75 | 53 | 16 | 0.5764 | 0.5859 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 73 | 17 | 54 | 0.6944 | 0.8111 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_overlap | 95 | 48 | 1 | 0.6632 | 0.6643 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 108 | 11 | 25 | 0.8368 | 0.9076 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_style | 9 | 14 | 121 | 0.4826 | 0.3913 |
| controlled_alt_vs_baseline_no_context_phi3_latest | distinct1 | 26 | 109 | 9 | 0.2118 | 0.1926 |
| controlled_alt_vs_baseline_no_context_phi3_latest | length_score | 63 | 71 | 10 | 0.4722 | 0.4701 |
| controlled_alt_vs_baseline_no_context_phi3_latest | sentence_score | 40 | 9 | 95 | 0.6076 | 0.8163 |
| controlled_alt_vs_baseline_no_context_phi3_latest | overall_quality | 118 | 26 | 0 | 0.8194 | 0.8194 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | context_relevance | 91 | 52 | 1 | 0.6354 | 0.6364 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | persona_consistency | 108 | 13 | 23 | 0.8299 | 0.8926 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | naturalness | 47 | 97 | 0 | 0.3264 | 0.3264 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | quest_state_correctness | 94 | 49 | 1 | 0.6562 | 0.6573 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | lore_consistency | 57 | 45 | 42 | 0.5417 | 0.5588 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | multi_turn_contradiction_safety | 0 | 0 | 144 | 0.5000 | nan |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | objective_completion_support | 83 | 31 | 30 | 0.6806 | 0.7281 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | gameplay_usefulness | 107 | 37 | 0 | 0.7431 | 0.7431 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | time_pressure_acceptability | 74 | 57 | 13 | 0.5590 | 0.5649 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | context_keyword_coverage | 72 | 19 | 53 | 0.6840 | 0.7912 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | context_overlap | 90 | 53 | 1 | 0.6285 | 0.6294 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | persona_keyword_coverage | 107 | 8 | 29 | 0.8438 | 0.9304 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | persona_style | 12 | 13 | 119 | 0.4965 | 0.4800 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | distinct1 | 21 | 113 | 10 | 0.1806 | 0.1567 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | length_score | 60 | 79 | 5 | 0.4340 | 0.4317 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | sentence_score | 40 | 7 | 97 | 0.6146 | 0.8511 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | overall_quality | 121 | 23 | 0 | 0.8403 | 0.8403 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | context_relevance | 94 | 49 | 1 | 0.6562 | 0.6573 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | persona_consistency | 109 | 15 | 20 | 0.8264 | 0.8790 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | naturalness | 55 | 89 | 0 | 0.3819 | 0.3819 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | quest_state_correctness | 95 | 48 | 1 | 0.6632 | 0.6643 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | lore_consistency | 60 | 44 | 40 | 0.5556 | 0.5769 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0 | 0 | 144 | 0.5000 | nan |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | objective_completion_support | 82 | 30 | 32 | 0.6806 | 0.7321 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 102 | 42 | 0 | 0.7083 | 0.7083 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 75 | 53 | 16 | 0.5764 | 0.5859 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 73 | 17 | 54 | 0.6944 | 0.8111 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | context_overlap | 95 | 48 | 1 | 0.6632 | 0.6643 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 108 | 11 | 25 | 0.8368 | 0.9076 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | persona_style | 9 | 14 | 121 | 0.4826 | 0.3913 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | distinct1 | 26 | 109 | 9 | 0.2118 | 0.1926 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | length_score | 63 | 71 | 10 | 0.4722 | 0.4701 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | sentence_score | 40 | 9 | 95 | 0.6076 | 0.8163 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | overall_quality | 118 | 26 | 0 | 0.8194 | 0.8194 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.2708 | 0.2153 | 0.7847 |
| proposed_contextual_controlled_tuned | 0.0000 | 0.0000 | 0.1667 | 0.2778 | 0.7153 |
| proposed_contextual | 0.0000 | 0.0000 | 0.4375 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.4583 | 0.0000 | 0.0000 |
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
# Proposal Alignment Evaluation Report

- Run ID: `20260313T064440Z`
- Generated: `2026-03-13T06:50:23.313001+00:00`
- Scenarios: `artifacts\proposal_control_tuning\auto_tune\20260313T051946Z\train_runs\trial_007\seed_23\20260313T064440Z\scenarios.jsonl`
- Scenario count: `24`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_tuned`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0705 (0.0365, 0.1096) | 0.2376 (0.1917, 0.2940) | 0.8682 (0.8481, 0.8846) | 0.2873 (0.2605, 0.3162) | n/a |
| proposed_contextual_controlled_tuned | 0.1094 (0.0603, 0.1671) | 0.2908 (0.2308, 0.3566) | 0.8707 (0.8525, 0.8872) | 0.3260 (0.2865, 0.3717) | n/a |
| proposed_contextual | 0.0631 (0.0352, 0.0947) | 0.2168 (0.1661, 0.2777) | 0.8677 (0.8535, 0.8822) | 0.2751 (0.2486, 0.3038) | n/a |
| candidate_no_context | 0.0245 (0.0134, 0.0368) | 0.2487 (0.1880, 0.3129) | 0.8833 (0.8668, 0.8997) | 0.2728 (0.2476, 0.3008) | n/a |

## Game-facing Outcome Metrics (mean, 95% CI)
| Arm | Quest-state Correctness | Lore Consistency | Contradiction Safety | Objective Completion Support | Gameplay Usefulness | Time-pressure Acceptability |
|---|---:|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.1592 (0.1290, 0.1943) | 0.0265 (0.0060, 0.0536) | 1.0000 (1.0000, 1.0000) | 0.0860 (0.0623, 0.1100) | 0.3000 (0.2858, 0.3155) | 0.2961 (0.2797, 0.3121) |
| proposed_contextual_controlled_tuned | 0.1917 (0.1493, 0.2391) | 0.0522 (0.0202, 0.0876) | 1.0000 (1.0000, 1.0000) | 0.0697 (0.0483, 0.0901) | 0.2976 (0.2838, 0.3124) | 0.2804 (0.2647, 0.2959) |
| proposed_contextual | 0.1510 (0.1286, 0.1759) | 0.0221 (0.0074, 0.0407) | 1.0000 (1.0000, 1.0000) | 0.0731 (0.0447, 0.1019) | 0.2938 (0.2805, 0.3083) | 0.2963 (0.2824, 0.3105) |
| candidate_no_context | 0.1179 (0.1103, 0.1267) | 0.0032 (0.0005, 0.0067) | 1.0000 (1.0000, 1.0000) | 0.0606 (0.0385, 0.0848) | 0.2813 (0.2706, 0.2917) | 0.2859 (0.2753, 0.2966) |

- Multi-turn contradiction rate is reported as `1 - contradiction_safety` in row-level outputs.

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0387 | 1.5811 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0318 | -0.1281 |
| proposed_vs_candidate_no_context | naturalness | -0.0156 | -0.0177 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.0331 | 0.2808 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0190 | 5.9740 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | 0.0124 | 0.2050 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 0.0125 | 0.0445 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | 0.0104 | 0.0363 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0492 | 2.6000 |
| proposed_vs_candidate_no_context | context_overlap | 0.0140 | 0.3749 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0476 | -0.2553 |
| proposed_vs_candidate_no_context | persona_style | 0.0313 | 0.0628 |
| proposed_vs_candidate_no_context | distinct1 | -0.0104 | -0.0110 |
| proposed_vs_candidate_no_context | length_score | -0.0500 | -0.0887 |
| proposed_vs_candidate_no_context | sentence_score | -0.0146 | -0.0155 |
| proposed_vs_candidate_no_context | overall_quality | 0.0023 | 0.0085 |
| controlled_vs_proposed_raw | context_relevance | 0.0074 | 0.1170 |
| controlled_vs_proposed_raw | persona_consistency | 0.0208 | 0.0958 |
| controlled_vs_proposed_raw | naturalness | 0.0004 | 0.0005 |
| controlled_vs_proposed_raw | quest_state_correctness | 0.0083 | 0.0550 |
| controlled_vs_proposed_raw | lore_consistency | 0.0043 | 0.1964 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0129 | 0.1768 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0062 | 0.0212 |
| controlled_vs_proposed_raw | time_pressure_acceptability | -0.0002 | -0.0006 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.0117 | 0.1722 |
| controlled_vs_proposed_raw | context_overlap | -0.0028 | -0.0541 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.0377 | 0.2714 |
| controlled_vs_proposed_raw | persona_style | -0.0469 | -0.0887 |
| controlled_vs_proposed_raw | distinct1 | -0.0108 | -0.0116 |
| controlled_vs_proposed_raw | length_score | 0.0069 | 0.0135 |
| controlled_vs_proposed_raw | sentence_score | 0.0583 | 0.0629 |
| controlled_vs_proposed_raw | overall_quality | 0.0122 | 0.0443 |
| controlled_vs_candidate_no_context | context_relevance | 0.0461 | 1.8831 |
| controlled_vs_candidate_no_context | persona_consistency | -0.0111 | -0.0445 |
| controlled_vs_candidate_no_context | naturalness | -0.0152 | -0.0172 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.0414 | 0.3512 |
| controlled_vs_candidate_no_context | lore_consistency | 0.0233 | 7.3440 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | 0.0253 | 0.4181 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0187 | 0.0666 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 0.0102 | 0.0357 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.0610 | 3.2200 |
| controlled_vs_candidate_no_context | context_overlap | 0.0112 | 0.3005 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | -0.0099 | -0.0532 |
| controlled_vs_candidate_no_context | persona_style | -0.0156 | -0.0314 |
| controlled_vs_candidate_no_context | distinct1 | -0.0212 | -0.0225 |
| controlled_vs_candidate_no_context | length_score | -0.0431 | -0.0764 |
| controlled_vs_candidate_no_context | sentence_score | 0.0437 | 0.0465 |
| controlled_vs_candidate_no_context | overall_quality | 0.0145 | 0.0532 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0389 | 0.5519 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0532 | 0.2238 |
| controlled_alt_vs_controlled_default | naturalness | 0.0025 | 0.0029 |
| controlled_alt_vs_controlled_default | quest_state_correctness | 0.0325 | 0.2040 |
| controlled_alt_vs_controlled_default | lore_consistency | 0.0257 | 0.9707 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_controlled_default | objective_completion_support | -0.0163 | -0.1898 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | -0.0024 | -0.0081 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | -0.0157 | -0.0531 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0534 | 0.6682 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0051 | 0.1051 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0665 | 0.3764 |
| controlled_alt_vs_controlled_default | persona_style | 0.0000 | 0.0000 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0148 | 0.0160 |
| controlled_alt_vs_controlled_default | length_score | -0.0292 | -0.0560 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0000 | 0.0000 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0387 | 0.1348 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.0463 | 0.7335 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.0740 | 0.3411 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0030 | 0.0034 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 0.0408 | 0.2701 |
| controlled_alt_vs_proposed_raw | lore_consistency | 0.0300 | 1.3578 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_proposed_raw | objective_completion_support | -0.0034 | -0.0466 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 0.0038 | 0.0129 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | -0.0159 | -0.0536 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.0652 | 0.9556 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0023 | 0.0453 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1042 | 0.7500 |
| controlled_alt_vs_proposed_raw | persona_style | -0.0469 | -0.0887 |
| controlled_alt_vs_proposed_raw | distinct1 | 0.0039 | 0.0042 |
| controlled_alt_vs_proposed_raw | length_score | -0.0222 | -0.0432 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0583 | 0.0629 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.0509 | 0.1851 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.0850 | 3.4743 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.0421 | 0.1693 |
| controlled_alt_vs_candidate_no_context | naturalness | -0.0127 | -0.0143 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 0.0739 | 0.6267 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 0.0490 | 15.4437 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_candidate_no_context | objective_completion_support | 0.0090 | 0.1489 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 0.0163 | 0.0580 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | -0.0055 | -0.0193 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.1144 | 6.0400 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0163 | 0.4372 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.0565 | 0.3032 |
| controlled_alt_vs_candidate_no_context | persona_style | -0.0156 | -0.0314 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0065 | -0.0069 |
| controlled_alt_vs_candidate_no_context | length_score | -0.0722 | -0.1281 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.0437 | 0.0465 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.0532 | 0.1952 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0387 | (0.0141, 0.0699) | 0.0000 | 0.0387 | (0.0004, 0.0675) | 0.0083 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0318 | (-0.0891, 0.0250) | 0.8587 | -0.0318 | (-0.1103, 0.0111) | 0.8827 |
| proposed_vs_candidate_no_context | naturalness | -0.0156 | (-0.0379, 0.0040) | 0.9360 | -0.0156 | (-0.0391, 0.0050) | 0.9140 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.0331 | (0.0124, 0.0578) | 0.0000 | 0.0331 | (0.0002, 0.0574) | 0.0113 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0190 | (0.0052, 0.0367) | 0.0000 | 0.0190 | (0.0000, 0.0396) | 0.0783 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | 0.0124 | (-0.0171, 0.0431) | 0.2180 | 0.0124 | (0.0002, 0.0365) | 0.0110 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 0.0125 | (-0.0049, 0.0304) | 0.0767 | 0.0125 | (-0.0015, 0.0279) | 0.0700 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | 0.0104 | (-0.0050, 0.0272) | 0.0937 | 0.0104 | (0.0008, 0.0255) | 0.0100 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0492 | (0.0152, 0.0909) | 0.0000 | 0.0492 | (0.0000, 0.0877) | 0.0853 |
| proposed_vs_candidate_no_context | context_overlap | 0.0140 | (0.0031, 0.0265) | 0.0033 | 0.0140 | (0.0015, 0.0231) | 0.0097 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0476 | (-0.1200, 0.0208) | 0.9227 | -0.0476 | (-0.1634, 0.0139) | 0.8747 |
| proposed_vs_candidate_no_context | persona_style | 0.0312 | (-0.0104, 0.0938) | 0.1640 | 0.0312 | (0.0000, 0.1023) | 0.3213 |
| proposed_vs_candidate_no_context | distinct1 | -0.0104 | (-0.0323, 0.0125) | 0.8147 | -0.0104 | (-0.0441, 0.0092) | 0.7593 |
| proposed_vs_candidate_no_context | length_score | -0.0500 | (-0.1500, 0.0361) | 0.8530 | -0.0500 | (-0.1538, 0.0192) | 0.8013 |
| proposed_vs_candidate_no_context | sentence_score | -0.0146 | (-0.0729, 0.0437) | 0.7573 | -0.0146 | (-0.0808, 0.0404) | 0.7350 |
| proposed_vs_candidate_no_context | overall_quality | 0.0023 | (-0.0241, 0.0317) | 0.4560 | 0.0023 | (-0.0368, 0.0291) | 0.4273 |
| controlled_vs_proposed_raw | context_relevance | 0.0074 | (-0.0173, 0.0338) | 0.2863 | 0.0074 | (-0.0179, 0.0262) | 0.2750 |
| controlled_vs_proposed_raw | persona_consistency | 0.0208 | (-0.0246, 0.0678) | 0.1850 | 0.0208 | (-0.0648, 0.0698) | 0.2910 |
| controlled_vs_proposed_raw | naturalness | 0.0004 | (-0.0174, 0.0168) | 0.4620 | 0.0004 | (-0.0181, 0.0223) | 0.4930 |
| controlled_vs_proposed_raw | quest_state_correctness | 0.0083 | (-0.0123, 0.0312) | 0.2297 | 0.0083 | (-0.0150, 0.0275) | 0.2353 |
| controlled_vs_proposed_raw | lore_consistency | 0.0043 | (-0.0097, 0.0222) | 0.3137 | 0.0043 | (-0.0140, 0.0260) | 0.4193 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0129 | (-0.0119, 0.0374) | 0.1467 | 0.0129 | (-0.0205, 0.0377) | 0.1953 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0062 | (-0.0086, 0.0236) | 0.2357 | 0.0062 | (-0.0198, 0.0267) | 0.3180 |
| controlled_vs_proposed_raw | time_pressure_acceptability | -0.0002 | (-0.0152, 0.0147) | 0.5070 | -0.0002 | (-0.0191, 0.0120) | 0.5103 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.0117 | (-0.0189, 0.0455) | 0.2393 | 0.0117 | (-0.0182, 0.0341) | 0.2020 |
| controlled_vs_proposed_raw | context_overlap | -0.0028 | (-0.0151, 0.0099) | 0.6627 | -0.0028 | (-0.0221, 0.0112) | 0.6117 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.0377 | (-0.0149, 0.0972) | 0.0930 | 0.0377 | (-0.0682, 0.1016) | 0.1970 |
| controlled_vs_proposed_raw | persona_style | -0.0469 | (-0.0834, -0.0156) | 1.0000 | -0.0469 | (-0.1193, 0.0000) | 1.0000 |
| controlled_vs_proposed_raw | distinct1 | -0.0108 | (-0.0367, 0.0114) | 0.7843 | -0.0108 | (-0.0373, 0.0259) | 0.7177 |
| controlled_vs_proposed_raw | length_score | 0.0069 | (-0.0528, 0.0708) | 0.4227 | 0.0069 | (-0.0750, 0.0923) | 0.4457 |
| controlled_vs_proposed_raw | sentence_score | 0.0583 | (0.0000, 0.1313) | 0.0580 | 0.0583 | (0.0000, 0.1250) | 0.0830 |
| controlled_vs_proposed_raw | overall_quality | 0.0122 | (-0.0115, 0.0375) | 0.1637 | 0.0122 | (-0.0317, 0.0378) | 0.2470 |
| controlled_vs_candidate_no_context | context_relevance | 0.0461 | (0.0107, 0.0854) | 0.0030 | 0.0461 | (-0.0130, 0.0882) | 0.0627 |
| controlled_vs_candidate_no_context | persona_consistency | -0.0111 | (-0.0742, 0.0458) | 0.6327 | -0.0111 | (-0.1065, 0.0571) | 0.5853 |
| controlled_vs_candidate_no_context | naturalness | -0.0152 | (-0.0352, 0.0037) | 0.9420 | -0.0152 | (-0.0275, -0.0065) | 1.0000 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.0414 | (0.0100, 0.0787) | 0.0023 | 0.0414 | (-0.0106, 0.0794) | 0.0537 |
| controlled_vs_candidate_no_context | lore_consistency | 0.0233 | (0.0031, 0.0477) | 0.0080 | 0.0233 | (0.0000, 0.0452) | 0.0753 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | 0.0253 | (-0.0039, 0.0531) | 0.0480 | 0.0253 | (0.0025, 0.0424) | 0.0113 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0187 | (0.0042, 0.0342) | 0.0060 | 0.0187 | (-0.0085, 0.0393) | 0.0827 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 0.0102 | (-0.0075, 0.0277) | 0.1297 | 0.0102 | (0.0002, 0.0192) | 0.0113 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.0610 | (0.0159, 0.1140) | 0.0017 | 0.0610 | (-0.0136, 0.1143) | 0.0567 |
| controlled_vs_candidate_no_context | context_overlap | 0.0112 | (-0.0043, 0.0273) | 0.0867 | 0.0112 | (-0.0144, 0.0286) | 0.1703 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | -0.0099 | (-0.0774, 0.0585) | 0.6247 | -0.0099 | (-0.1238, 0.0714) | 0.6070 |
| controlled_vs_candidate_no_context | persona_style | -0.0156 | (-0.0716, 0.0339) | 0.7100 | -0.0156 | (-0.0511, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | distinct1 | -0.0212 | (-0.0541, 0.0079) | 0.9260 | -0.0212 | (-0.0466, 0.0104) | 0.9150 |
| controlled_vs_candidate_no_context | length_score | -0.0431 | (-0.1194, 0.0319) | 0.8613 | -0.0431 | (-0.0955, 0.0077) | 0.9463 |
| controlled_vs_candidate_no_context | sentence_score | 0.0437 | (-0.0146, 0.1021) | 0.1193 | 0.0437 | (0.0000, 0.0817) | 0.0727 |
| controlled_vs_candidate_no_context | overall_quality | 0.0145 | (-0.0171, 0.0458) | 0.1910 | 0.0145 | (-0.0469, 0.0560) | 0.3253 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0389 | (0.0035, 0.0842) | 0.0133 | 0.0389 | (-0.0011, 0.0844) | 0.0843 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0532 | (0.0069, 0.1152) | 0.0060 | 0.0532 | (0.0121, 0.0927) | 0.0107 |
| controlled_alt_vs_controlled_default | naturalness | 0.0025 | (-0.0172, 0.0246) | 0.4180 | 0.0025 | (-0.0193, 0.0276) | 0.4493 |
| controlled_alt_vs_controlled_default | quest_state_correctness | 0.0325 | (0.0025, 0.0685) | 0.0153 | 0.0325 | (0.0005, 0.0699) | 0.0113 |
| controlled_alt_vs_controlled_default | lore_consistency | 0.0257 | (0.0003, 0.0554) | 0.0247 | 0.0257 | (-0.0039, 0.0623) | 0.0750 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | objective_completion_support | -0.0163 | (-0.0368, 0.0023) | 0.9583 | -0.0163 | (-0.0342, -0.0020) | 1.0000 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | -0.0024 | (-0.0158, 0.0110) | 0.6327 | -0.0024 | (-0.0090, 0.0025) | 0.7473 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | -0.0157 | (-0.0344, 0.0019) | 0.9580 | -0.0157 | (-0.0323, -0.0001) | 1.0000 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0534 | (0.0042, 0.1144) | 0.0110 | 0.0534 | (0.0000, 0.1129) | 0.0080 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0051 | (-0.0079, 0.0193) | 0.2350 | 0.0051 | (-0.0044, 0.0174) | 0.2547 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0665 | (0.0139, 0.1399) | 0.0037 | 0.0665 | (0.0152, 0.1126) | 0.0133 |
| controlled_alt_vs_controlled_default | persona_style | 0.0000 | (-0.0312, 0.0312) | 0.6490 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0148 | (-0.0029, 0.0377) | 0.0590 | 0.0148 | (-0.0029, 0.0279) | 0.0760 |
| controlled_alt_vs_controlled_default | length_score | -0.0292 | (-0.1153, 0.0431) | 0.7740 | -0.0292 | (-0.1064, 0.0444) | 0.7470 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0000 | (-0.0437, 0.0437) | 0.6303 | 0.0000 | (-0.0404, 0.0404) | 0.6350 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0387 | (0.0112, 0.0730) | 0.0010 | 0.0387 | (0.0068, 0.0699) | 0.0120 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.0463 | (0.0014, 0.0989) | 0.0227 | 0.0463 | (-0.0078, 0.0927) | 0.0403 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.0740 | (0.0046, 0.1530) | 0.0170 | 0.0740 | (-0.0527, 0.1535) | 0.0893 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0030 | (-0.0133, 0.0196) | 0.3547 | 0.0030 | (-0.0153, 0.0161) | 0.3750 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 0.0408 | (0.0012, 0.0849) | 0.0213 | 0.0408 | (-0.0066, 0.0796) | 0.0437 |
| controlled_alt_vs_proposed_raw | lore_consistency | 0.0300 | (0.0051, 0.0590) | 0.0080 | 0.0300 | (0.0081, 0.0495) | 0.0107 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_proposed_raw | objective_completion_support | -0.0034 | (-0.0263, 0.0193) | 0.6163 | -0.0034 | (-0.0322, 0.0156) | 0.5803 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 0.0038 | (-0.0105, 0.0178) | 0.2947 | 0.0038 | (-0.0201, 0.0209) | 0.3667 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | -0.0159 | (-0.0331, -0.0005) | 0.9783 | -0.0159 | (-0.0464, 0.0071) | 0.9170 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.0652 | (0.0045, 0.1379) | 0.0147 | 0.0652 | (-0.0000, 0.1280) | 0.0337 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0023 | (-0.0155, 0.0193) | 0.4073 | 0.0023 | (-0.0205, 0.0205) | 0.3953 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1042 | (0.0248, 0.1974) | 0.0060 | 0.1042 | (-0.0303, 0.2115) | 0.0600 |
| controlled_alt_vs_proposed_raw | persona_style | -0.0469 | (-0.0951, -0.0117) | 1.0000 | -0.0469 | (-0.1193, 0.0000) | 1.0000 |
| controlled_alt_vs_proposed_raw | distinct1 | 0.0039 | (-0.0119, 0.0189) | 0.2927 | 0.0039 | (-0.0103, 0.0222) | 0.3617 |
| controlled_alt_vs_proposed_raw | length_score | -0.0222 | (-0.1000, 0.0514) | 0.7353 | -0.0222 | (-0.1242, 0.0488) | 0.6940 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0583 | (0.0146, 0.1167) | 0.0123 | 0.0583 | (0.0000, 0.1000) | 0.0737 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.0509 | (0.0070, 0.0987) | 0.0117 | 0.0509 | (-0.0220, 0.1036) | 0.0657 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.0850 | (0.0317, 0.1417) | 0.0007 | 0.0850 | (0.0032, 0.1523) | 0.0203 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.0421 | (-0.0222, 0.1164) | 0.1147 | 0.0421 | (-0.0723, 0.1341) | 0.2630 |
| controlled_alt_vs_candidate_no_context | naturalness | -0.0127 | (-0.0379, 0.0108) | 0.8397 | -0.0127 | (-0.0454, 0.0144) | 0.8090 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 0.0739 | (0.0296, 0.1215) | 0.0000 | 0.0739 | (0.0017, 0.1258) | 0.0223 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 0.0490 | (0.0185, 0.0831) | 0.0000 | 0.0490 | (0.0085, 0.0904) | 0.0113 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_candidate_no_context | objective_completion_support | 0.0090 | (-0.0196, 0.0374) | 0.2760 | 0.0090 | (0.0005, 0.0202) | 0.0117 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 0.0163 | (0.0010, 0.0329) | 0.0190 | 0.0163 | (-0.0087, 0.0368) | 0.1200 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | -0.0055 | (-0.0220, 0.0113) | 0.7413 | -0.0055 | (-0.0233, 0.0128) | 0.7033 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.1144 | (0.0455, 0.1879) | 0.0000 | 0.1144 | (0.0091, 0.1986) | 0.0193 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0163 | (-0.0000, 0.0318) | 0.0253 | 0.0163 | (-0.0163, 0.0377) | 0.1443 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.0565 | (-0.0179, 0.1508) | 0.0800 | 0.0565 | (-0.0810, 0.1794) | 0.2150 |
| controlled_alt_vs_candidate_no_context | persona_style | -0.0156 | (-0.0482, 0.0169) | 0.8370 | -0.0156 | (-0.0511, 0.0000) | 1.0000 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0065 | (-0.0261, 0.0118) | 0.7697 | -0.0065 | (-0.0305, 0.0176) | 0.6643 |
| controlled_alt_vs_candidate_no_context | length_score | -0.0722 | (-0.2028, 0.0458) | 0.8823 | -0.0722 | (-0.1889, 0.0487) | 0.8730 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.0437 | (-0.0146, 0.1021) | 0.1193 | 0.0437 | (0.0000, 0.1212) | 0.3327 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.0532 | (0.0112, 0.1012) | 0.0060 | 0.0532 | (-0.0258, 0.1185) | 0.0890 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 11 | 2 | 11 | 0.6875 | 0.8462 |
| proposed_vs_candidate_no_context | persona_consistency | 4 | 6 | 14 | 0.4583 | 0.4000 |
| proposed_vs_candidate_no_context | naturalness | 3 | 11 | 10 | 0.3333 | 0.2143 |
| proposed_vs_candidate_no_context | quest_state_correctness | 12 | 2 | 10 | 0.7083 | 0.8571 |
| proposed_vs_candidate_no_context | lore_consistency | 7 | 0 | 17 | 0.6458 | 1.0000 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| proposed_vs_candidate_no_context | objective_completion_support | 6 | 6 | 12 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 9 | 5 | 10 | 0.5833 | 0.6429 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | 6 | 7 | 11 | 0.4792 | 0.4615 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 7 | 0 | 17 | 0.6458 | 1.0000 |
| proposed_vs_candidate_no_context | context_overlap | 11 | 2 | 11 | 0.6875 | 0.8462 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 3 | 6 | 15 | 0.4375 | 0.3333 |
| proposed_vs_candidate_no_context | persona_style | 3 | 1 | 20 | 0.5417 | 0.7500 |
| proposed_vs_candidate_no_context | distinct1 | 5 | 7 | 12 | 0.4583 | 0.4167 |
| proposed_vs_candidate_no_context | length_score | 7 | 6 | 11 | 0.5208 | 0.5385 |
| proposed_vs_candidate_no_context | sentence_score | 2 | 3 | 19 | 0.4792 | 0.4000 |
| proposed_vs_candidate_no_context | overall_quality | 9 | 5 | 10 | 0.5833 | 0.6429 |
| controlled_vs_proposed_raw | context_relevance | 8 | 10 | 6 | 0.4583 | 0.4444 |
| controlled_vs_proposed_raw | persona_consistency | 8 | 6 | 10 | 0.5417 | 0.5714 |
| controlled_vs_proposed_raw | naturalness | 8 | 10 | 6 | 0.4583 | 0.4444 |
| controlled_vs_proposed_raw | quest_state_correctness | 8 | 10 | 6 | 0.4583 | 0.4444 |
| controlled_vs_proposed_raw | lore_consistency | 3 | 5 | 16 | 0.4583 | 0.3750 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_vs_proposed_raw | objective_completion_support | 5 | 3 | 16 | 0.5417 | 0.6250 |
| controlled_vs_proposed_raw | gameplay_usefulness | 8 | 10 | 6 | 0.4583 | 0.4444 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 4 | 5 | 15 | 0.4792 | 0.4444 |
| controlled_vs_proposed_raw | context_keyword_coverage | 5 | 3 | 16 | 0.5417 | 0.6250 |
| controlled_vs_proposed_raw | context_overlap | 7 | 11 | 6 | 0.4167 | 0.3889 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 8 | 4 | 12 | 0.5833 | 0.6667 |
| controlled_vs_proposed_raw | persona_style | 0 | 7 | 17 | 0.3542 | 0.0000 |
| controlled_vs_proposed_raw | distinct1 | 11 | 5 | 8 | 0.6250 | 0.6875 |
| controlled_vs_proposed_raw | length_score | 8 | 9 | 7 | 0.4792 | 0.4706 |
| controlled_vs_proposed_raw | sentence_score | 5 | 1 | 18 | 0.5833 | 0.8333 |
| controlled_vs_proposed_raw | overall_quality | 10 | 8 | 6 | 0.5417 | 0.5556 |
| controlled_vs_candidate_no_context | context_relevance | 11 | 8 | 5 | 0.5625 | 0.5789 |
| controlled_vs_candidate_no_context | persona_consistency | 8 | 7 | 9 | 0.5208 | 0.5333 |
| controlled_vs_candidate_no_context | naturalness | 7 | 13 | 4 | 0.3750 | 0.3500 |
| controlled_vs_candidate_no_context | quest_state_correctness | 13 | 7 | 4 | 0.6250 | 0.6500 |
| controlled_vs_candidate_no_context | lore_consistency | 6 | 2 | 16 | 0.5833 | 0.7500 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_vs_candidate_no_context | objective_completion_support | 9 | 4 | 11 | 0.6042 | 0.6923 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 14 | 6 | 4 | 0.6667 | 0.7000 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 9 | 5 | 10 | 0.5833 | 0.6429 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 8 | 2 | 14 | 0.6250 | 0.8000 |
| controlled_vs_candidate_no_context | context_overlap | 11 | 8 | 5 | 0.5625 | 0.5789 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 7 | 7 | 10 | 0.5000 | 0.5000 |
| controlled_vs_candidate_no_context | persona_style | 2 | 5 | 17 | 0.4375 | 0.2857 |
| controlled_vs_candidate_no_context | distinct1 | 8 | 11 | 5 | 0.4375 | 0.4211 |
| controlled_vs_candidate_no_context | length_score | 7 | 13 | 4 | 0.3750 | 0.3500 |
| controlled_vs_candidate_no_context | sentence_score | 4 | 1 | 19 | 0.5625 | 0.8000 |
| controlled_vs_candidate_no_context | overall_quality | 14 | 6 | 4 | 0.6667 | 0.7000 |
| controlled_alt_vs_controlled_default | context_relevance | 8 | 4 | 12 | 0.5833 | 0.6667 |
| controlled_alt_vs_controlled_default | persona_consistency | 5 | 1 | 18 | 0.5833 | 0.8333 |
| controlled_alt_vs_controlled_default | naturalness | 6 | 6 | 12 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | quest_state_correctness | 8 | 4 | 12 | 0.5833 | 0.6667 |
| controlled_alt_vs_controlled_default | lore_consistency | 5 | 3 | 16 | 0.5417 | 0.6250 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_alt_vs_controlled_default | objective_completion_support | 3 | 8 | 13 | 0.3958 | 0.2727 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | 7 | 5 | 12 | 0.5417 | 0.5833 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | 4 | 8 | 12 | 0.4167 | 0.3333 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 6 | 2 | 16 | 0.5833 | 0.7500 |
| controlled_alt_vs_controlled_default | context_overlap | 7 | 5 | 12 | 0.5417 | 0.5833 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 5 | 0 | 19 | 0.6042 | 1.0000 |
| controlled_alt_vs_controlled_default | persona_style | 1 | 1 | 22 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | distinct1 | 7 | 4 | 13 | 0.5625 | 0.6364 |
| controlled_alt_vs_controlled_default | length_score | 6 | 6 | 12 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | sentence_score | 1 | 1 | 22 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | overall_quality | 9 | 3 | 12 | 0.6250 | 0.7500 |
| controlled_alt_vs_proposed_raw | context_relevance | 11 | 8 | 5 | 0.5625 | 0.5789 |
| controlled_alt_vs_proposed_raw | persona_consistency | 11 | 5 | 8 | 0.6250 | 0.6875 |
| controlled_alt_vs_proposed_raw | naturalness | 8 | 10 | 6 | 0.4583 | 0.4444 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 12 | 7 | 5 | 0.6042 | 0.6316 |
| controlled_alt_vs_proposed_raw | lore_consistency | 6 | 3 | 15 | 0.5625 | 0.6667 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_alt_vs_proposed_raw | objective_completion_support | 5 | 6 | 13 | 0.4792 | 0.4545 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 9 | 10 | 5 | 0.4792 | 0.4737 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | 3 | 10 | 11 | 0.3542 | 0.2308 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 8 | 4 | 12 | 0.5833 | 0.6667 |
| controlled_alt_vs_proposed_raw | context_overlap | 10 | 9 | 5 | 0.5208 | 0.5263 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 11 | 4 | 9 | 0.6458 | 0.7333 |
| controlled_alt_vs_proposed_raw | persona_style | 0 | 6 | 18 | 0.3750 | 0.0000 |
| controlled_alt_vs_proposed_raw | distinct1 | 11 | 6 | 7 | 0.6042 | 0.6471 |
| controlled_alt_vs_proposed_raw | length_score | 7 | 11 | 6 | 0.4167 | 0.3889 |
| controlled_alt_vs_proposed_raw | sentence_score | 4 | 0 | 20 | 0.5833 | 1.0000 |
| controlled_alt_vs_proposed_raw | overall_quality | 12 | 7 | 5 | 0.6042 | 0.6316 |
| controlled_alt_vs_candidate_no_context | context_relevance | 13 | 7 | 4 | 0.6250 | 0.6500 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 10 | 6 | 8 | 0.5833 | 0.6250 |
| controlled_alt_vs_candidate_no_context | naturalness | 10 | 10 | 4 | 0.5000 | 0.5000 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 14 | 6 | 4 | 0.6667 | 0.7000 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 8 | 1 | 15 | 0.6458 | 0.8889 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_alt_vs_candidate_no_context | objective_completion_support | 8 | 6 | 10 | 0.5417 | 0.5714 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 13 | 7 | 4 | 0.6250 | 0.6500 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | 6 | 9 | 9 | 0.4375 | 0.4000 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 10 | 2 | 12 | 0.6667 | 0.8333 |
| controlled_alt_vs_candidate_no_context | context_overlap | 13 | 7 | 4 | 0.6250 | 0.6500 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 9 | 5 | 10 | 0.5833 | 0.6429 |
| controlled_alt_vs_candidate_no_context | persona_style | 1 | 5 | 18 | 0.4167 | 0.1667 |
| controlled_alt_vs_candidate_no_context | distinct1 | 9 | 10 | 5 | 0.4792 | 0.4737 |
| controlled_alt_vs_candidate_no_context | length_score | 10 | 10 | 4 | 0.5000 | 0.5000 |
| controlled_alt_vs_candidate_no_context | sentence_score | 4 | 1 | 19 | 0.5625 | 0.8000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 14 | 6 | 4 | 0.6667 | 0.7000 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.1250 | 0.1667 | 0.8333 |
| proposed_contextual_controlled_tuned | 0.0000 | 0.0000 | 0.0417 | 0.2917 | 0.7083 |
| proposed_contextual | 0.0000 | 0.0000 | 0.6250 | 0.0000 | 0.0000 |
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
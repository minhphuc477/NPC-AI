# Proposal Alignment Evaluation Report

- Run ID: `20260313T053043Z`
- Generated: `2026-03-13T05:35:31.996393+00:00`
- Scenarios: `artifacts\proposal_control_tuning\auto_tune\20260313T051946Z\train_runs\trial_001\seed_19\20260313T053043Z\scenarios.jsonl`
- Scenario count: `24`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_tuned`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.1357 (0.0784, 0.2023) | 0.2447 (0.1991, 0.2913) | 0.8933 (0.8750, 0.9125) | 0.3246 (0.2846, 0.3633) | n/a |
| proposed_contextual_controlled_tuned | 0.1098 (0.0623, 0.1617) | 0.3026 (0.2261, 0.3843) | 0.8855 (0.8637, 0.9072) | 0.3330 (0.2872, 0.3832) | n/a |
| proposed_contextual | 0.0568 (0.0286, 0.0864) | 0.2030 (0.1556, 0.2590) | 0.8826 (0.8640, 0.9030) | 0.2699 (0.2452, 0.2960) | n/a |
| candidate_no_context | 0.0236 (0.0137, 0.0356) | 0.2250 (0.1665, 0.2884) | 0.8887 (0.8706, 0.9070) | 0.2640 (0.2370, 0.2957) | n/a |

## Game-facing Outcome Metrics (mean, 95% CI)
| Arm | Quest-state Correctness | Lore Consistency | Contradiction Safety | Objective Completion Support | Gameplay Usefulness | Time-pressure Acceptability |
|---|---:|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2145 (0.1623, 0.2715) | 0.0590 (0.0216, 0.0990) | 1.0000 (1.0000, 1.0000) | 0.0624 (0.0375, 0.0868) | 0.3178 (0.2996, 0.3368) | 0.2979 (0.2807, 0.3151) |
| proposed_contextual_controlled_tuned | 0.1964 (0.1524, 0.2440) | 0.0525 (0.0225, 0.0853) | 1.0000 (1.0000, 1.0000) | 0.0686 (0.0431, 0.0945) | 0.3085 (0.2941, 0.3232) | 0.2890 (0.2754, 0.3030) |
| proposed_contextual | 0.1498 (0.1255, 0.1800) | 0.0238 (0.0053, 0.0475) | 1.0000 (1.0000, 1.0000) | 0.0680 (0.0444, 0.0921) | 0.2961 (0.2832, 0.3089) | 0.2980 (0.2826, 0.3143) |
| candidate_no_context | 0.1151 (0.1079, 0.1235) | 0.0000 (0.0000, 0.0000) | 1.0000 (1.0000, 1.0000) | 0.0747 (0.0511, 0.0960) | 0.2898 (0.2787, 0.3006) | 0.3006 (0.2883, 0.3140) |

- Multi-turn contradiction rate is reported as `1 - contradiction_safety` in row-level outputs.

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0332 | 1.4036 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0220 | -0.0976 |
| proposed_vs_candidate_no_context | naturalness | -0.0061 | -0.0069 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.0347 | 0.3012 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0238 | nan |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | -0.0067 | -0.0901 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 0.0063 | 0.0216 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | -0.0026 | -0.0087 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0428 | 2.2600 |
| proposed_vs_candidate_no_context | context_overlap | 0.0107 | 0.3084 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0278 | -0.1905 |
| proposed_vs_candidate_no_context | persona_style | 0.0013 | 0.0024 |
| proposed_vs_candidate_no_context | distinct1 | -0.0150 | -0.0159 |
| proposed_vs_candidate_no_context | length_score | 0.0139 | 0.0243 |
| proposed_vs_candidate_no_context | sentence_score | -0.0292 | -0.0296 |
| proposed_vs_candidate_no_context | overall_quality | 0.0059 | 0.0223 |
| controlled_vs_proposed_raw | context_relevance | 0.0789 | 1.3895 |
| controlled_vs_proposed_raw | persona_consistency | 0.0417 | 0.2054 |
| controlled_vs_proposed_raw | naturalness | 0.0106 | 0.0121 |
| controlled_vs_proposed_raw | quest_state_correctness | 0.0648 | 0.4323 |
| controlled_vs_proposed_raw | lore_consistency | 0.0353 | 1.4846 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_proposed_raw | objective_completion_support | -0.0056 | -0.0827 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0218 | 0.0735 |
| controlled_vs_proposed_raw | time_pressure_acceptability | -0.0002 | -0.0005 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.0992 | 1.6074 |
| controlled_vs_proposed_raw | context_overlap | 0.0314 | 0.6954 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.0619 | 0.5244 |
| controlled_vs_proposed_raw | persona_style | -0.0391 | -0.0719 |
| controlled_vs_proposed_raw | distinct1 | 0.0148 | 0.0160 |
| controlled_vs_proposed_raw | length_score | 0.0236 | 0.0404 |
| controlled_vs_proposed_raw | sentence_score | 0.0000 | 0.0000 |
| controlled_vs_proposed_raw | overall_quality | 0.0547 | 0.2027 |
| controlled_vs_candidate_no_context | context_relevance | 0.1121 | 4.7436 |
| controlled_vs_candidate_no_context | persona_consistency | 0.0197 | 0.0878 |
| controlled_vs_candidate_no_context | naturalness | 0.0045 | 0.0051 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.0994 | 0.8636 |
| controlled_vs_candidate_no_context | lore_consistency | 0.0590 | nan |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | -0.0124 | -0.1654 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0280 | 0.0967 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | -0.0028 | -0.0093 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.1420 | 7.5000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0421 | 1.2183 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.0341 | 0.2340 |
| controlled_vs_candidate_no_context | persona_style | -0.0378 | -0.0697 |
| controlled_vs_candidate_no_context | distinct1 | -0.0001 | -0.0002 |
| controlled_vs_candidate_no_context | length_score | 0.0375 | 0.0657 |
| controlled_vs_candidate_no_context | sentence_score | -0.0292 | -0.0296 |
| controlled_vs_candidate_no_context | overall_quality | 0.0606 | 0.2295 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0259 | -0.1907 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0579 | 0.2365 |
| controlled_alt_vs_controlled_default | naturalness | -0.0078 | -0.0087 |
| controlled_alt_vs_controlled_default | quest_state_correctness | -0.0181 | -0.0844 |
| controlled_alt_vs_controlled_default | lore_consistency | -0.0065 | -0.1110 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_controlled_default | objective_completion_support | 0.0062 | 0.1002 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | -0.0093 | -0.0292 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | -0.0088 | -0.0296 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0307 | -0.1906 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0147 | -0.1913 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0720 | 0.4002 |
| controlled_alt_vs_controlled_default | persona_style | 0.0013 | 0.0026 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0048 | -0.0051 |
| controlled_alt_vs_controlled_default | length_score | -0.0292 | -0.0479 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0000 | 0.0000 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0084 | 0.0258 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.0530 | 0.9338 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.0996 | 0.4905 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0029 | 0.0033 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 0.0466 | 0.3113 |
| controlled_alt_vs_proposed_raw | lore_consistency | 0.0287 | 1.2088 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_proposed_raw | objective_completion_support | 0.0006 | 0.0092 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 0.0125 | 0.0421 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | -0.0090 | -0.0302 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.0686 | 1.1104 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0168 | 0.3710 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1339 | 1.1345 |
| controlled_alt_vs_proposed_raw | persona_style | -0.0378 | -0.0695 |
| controlled_alt_vs_proposed_raw | distinct1 | 0.0100 | 0.0108 |
| controlled_alt_vs_proposed_raw | length_score | -0.0056 | -0.0095 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0000 | 0.0000 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.0631 | 0.2337 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.0862 | 3.6482 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.0776 | 0.3450 |
| controlled_alt_vs_candidate_no_context | naturalness | -0.0032 | -0.0036 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 0.0813 | 0.7063 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 0.0525 | nan |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_candidate_no_context | objective_completion_support | -0.0061 | -0.0818 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 0.0187 | 0.0646 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | -0.0116 | -0.0386 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.1114 | 5.8800 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0274 | 0.7939 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1062 | 0.7279 |
| controlled_alt_vs_candidate_no_context | persona_style | -0.0365 | -0.0673 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0049 | -0.0053 |
| controlled_alt_vs_candidate_no_context | length_score | 0.0083 | 0.0146 |
| controlled_alt_vs_candidate_no_context | sentence_score | -0.0292 | -0.0296 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.0689 | 0.2611 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0332 | (0.0077, 0.0630) | 0.0043 | 0.0332 | (0.0083, 0.0599) | 0.0117 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0220 | (-0.0836, 0.0347) | 0.7697 | -0.0220 | (-0.1091, 0.0292) | 0.7303 |
| proposed_vs_candidate_no_context | naturalness | -0.0061 | (-0.0246, 0.0143) | 0.7337 | -0.0061 | (-0.0315, 0.0110) | 0.7167 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.0347 | (0.0097, 0.0626) | 0.0010 | 0.0347 | (0.0113, 0.0621) | 0.0003 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0238 | (0.0055, 0.0461) | 0.0007 | 0.0238 | (0.0030, 0.0468) | 0.0117 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | -0.0067 | (-0.0267, 0.0133) | 0.7443 | -0.0067 | (-0.0239, 0.0114) | 0.7293 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 0.0063 | (-0.0134, 0.0266) | 0.2633 | 0.0063 | (-0.0023, 0.0164) | 0.0807 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | -0.0026 | (-0.0164, 0.0103) | 0.6527 | -0.0026 | (-0.0100, 0.0060) | 0.7500 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0428 | (0.0083, 0.0822) | 0.0040 | 0.0428 | (0.0124, 0.0764) | 0.0080 |
| proposed_vs_candidate_no_context | context_overlap | 0.0107 | (-0.0004, 0.0246) | 0.0313 | 0.0107 | (-0.0018, 0.0231) | 0.0487 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0278 | (-0.1052, 0.0427) | 0.7610 | -0.0278 | (-0.1364, 0.0357) | 0.7670 |
| proposed_vs_candidate_no_context | persona_style | 0.0013 | (-0.0299, 0.0326) | 0.4920 | 0.0013 | (0.0000, 0.0036) | 0.3273 |
| proposed_vs_candidate_no_context | distinct1 | -0.0150 | (-0.0336, 0.0037) | 0.9400 | -0.0150 | (-0.0375, 0.0030) | 0.9447 |
| proposed_vs_candidate_no_context | length_score | 0.0139 | (-0.0694, 0.1070) | 0.3940 | 0.0139 | (-0.0485, 0.0731) | 0.3650 |
| proposed_vs_candidate_no_context | sentence_score | -0.0292 | (-0.0875, 0.0292) | 0.9033 | -0.0292 | (-0.1114, 0.0318) | 0.8667 |
| proposed_vs_candidate_no_context | overall_quality | 0.0059 | (-0.0219, 0.0327) | 0.3340 | 0.0059 | (-0.0403, 0.0358) | 0.3123 |
| controlled_vs_proposed_raw | context_relevance | 0.0789 | (0.0263, 0.1401) | 0.0003 | 0.0789 | (0.0458, 0.1156) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.0417 | (-0.0173, 0.0999) | 0.0827 | 0.0417 | (-0.0052, 0.0998) | 0.0633 |
| controlled_vs_proposed_raw | naturalness | 0.0106 | (-0.0101, 0.0311) | 0.1580 | 0.0106 | (-0.0195, 0.0424) | 0.2463 |
| controlled_vs_proposed_raw | quest_state_correctness | 0.0648 | (0.0174, 0.1230) | 0.0020 | 0.0648 | (0.0317, 0.1002) | 0.0000 |
| controlled_vs_proposed_raw | lore_consistency | 0.0353 | (0.0029, 0.0719) | 0.0163 | 0.0353 | (0.0186, 0.0511) | 0.0000 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_proposed_raw | objective_completion_support | -0.0056 | (-0.0315, 0.0203) | 0.6643 | -0.0056 | (-0.0311, 0.0174) | 0.6240 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0218 | (0.0044, 0.0424) | 0.0053 | 0.0218 | (0.0004, 0.0493) | 0.0177 |
| controlled_vs_proposed_raw | time_pressure_acceptability | -0.0002 | (-0.0185, 0.0192) | 0.4930 | -0.0002 | (-0.0078, 0.0113) | 0.5380 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.0992 | (0.0311, 0.1750) | 0.0007 | 0.0992 | (0.0601, 0.1455) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0314 | (0.0045, 0.0656) | 0.0097 | 0.0314 | (0.0104, 0.0531) | 0.0007 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.0619 | (-0.0085, 0.1343) | 0.0447 | 0.0619 | (0.0071, 0.1310) | 0.0193 |
| controlled_vs_proposed_raw | persona_style | -0.0391 | (-0.0716, -0.0104) | 0.9997 | -0.0391 | (-0.0807, 0.0000) | 1.0000 |
| controlled_vs_proposed_raw | distinct1 | 0.0148 | (-0.0051, 0.0354) | 0.0870 | 0.0148 | (-0.0018, 0.0330) | 0.0523 |
| controlled_vs_proposed_raw | length_score | 0.0236 | (-0.0569, 0.1083) | 0.2957 | 0.0236 | (-0.1128, 0.1867) | 0.3737 |
| controlled_vs_proposed_raw | sentence_score | 0.0000 | (-0.0729, 0.0729) | 0.5757 | 0.0000 | (-0.0795, 0.0636) | 0.5600 |
| controlled_vs_proposed_raw | overall_quality | 0.0547 | (0.0122, 0.0995) | 0.0053 | 0.0547 | (0.0203, 0.1009) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.1121 | (0.0575, 0.1769) | 0.0003 | 0.1121 | (0.0683, 0.1703) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.0197 | (-0.0429, 0.0767) | 0.2503 | 0.0197 | (-0.0710, 0.0921) | 0.3083 |
| controlled_vs_candidate_no_context | naturalness | 0.0045 | (-0.0162, 0.0273) | 0.3383 | 0.0045 | (-0.0235, 0.0415) | 0.3897 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.0994 | (0.0495, 0.1565) | 0.0000 | 0.0994 | (0.0593, 0.1564) | 0.0000 |
| controlled_vs_candidate_no_context | lore_consistency | 0.0590 | (0.0218, 0.1001) | 0.0000 | 0.0590 | (0.0278, 0.0968) | 0.0007 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | -0.0124 | (-0.0374, 0.0156) | 0.8143 | -0.0124 | (-0.0315, 0.0103) | 0.8610 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0280 | (0.0073, 0.0503) | 0.0027 | 0.0280 | (0.0063, 0.0567) | 0.0007 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | -0.0028 | (-0.0184, 0.0147) | 0.6293 | -0.0028 | (-0.0156, 0.0134) | 0.6113 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.1420 | (0.0758, 0.2159) | 0.0000 | 0.1420 | (0.0881, 0.2194) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0421 | (0.0141, 0.0767) | 0.0000 | 0.0421 | (0.0233, 0.0609) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.0341 | (-0.0383, 0.1026) | 0.1830 | 0.0341 | (-0.0656, 0.1212) | 0.2187 |
| controlled_vs_candidate_no_context | persona_style | -0.0378 | (-0.0703, -0.0091) | 0.9990 | -0.0378 | (-0.0794, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | distinct1 | -0.0001 | (-0.0212, 0.0225) | 0.5130 | -0.0001 | (-0.0239, 0.0236) | 0.5113 |
| controlled_vs_candidate_no_context | length_score | 0.0375 | (-0.0473, 0.1417) | 0.2137 | 0.0375 | (-0.0833, 0.2106) | 0.2903 |
| controlled_vs_candidate_no_context | sentence_score | -0.0292 | (-0.0875, 0.0292) | 0.9007 | -0.0292 | (-0.0700, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | overall_quality | 0.0606 | (0.0184, 0.1042) | 0.0017 | 0.0606 | (0.0090, 0.1166) | 0.0157 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0259 | (-0.1093, 0.0504) | 0.7433 | -0.0259 | (-0.0755, 0.0436) | 0.7850 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0579 | (-0.0124, 0.1423) | 0.0637 | 0.0579 | (0.0001, 0.1492) | 0.0223 |
| controlled_alt_vs_controlled_default | naturalness | -0.0078 | (-0.0349, 0.0159) | 0.7013 | -0.0078 | (-0.0244, 0.0096) | 0.8010 |
| controlled_alt_vs_controlled_default | quest_state_correctness | -0.0181 | (-0.0881, 0.0509) | 0.6797 | -0.0181 | (-0.0649, 0.0473) | 0.7113 |
| controlled_alt_vs_controlled_default | lore_consistency | -0.0065 | (-0.0603, 0.0435) | 0.6043 | -0.0065 | (-0.0401, 0.0405) | 0.6117 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | objective_completion_support | 0.0063 | (-0.0130, 0.0269) | 0.2583 | 0.0063 | (-0.0123, 0.0218) | 0.2237 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | -0.0093 | (-0.0292, 0.0096) | 0.8137 | -0.0093 | (-0.0196, 0.0021) | 0.9450 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | -0.0088 | (-0.0294, 0.0102) | 0.7980 | -0.0088 | (-0.0329, 0.0084) | 0.8213 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0307 | (-0.1333, 0.0652) | 0.7180 | -0.0307 | (-0.0893, 0.0550) | 0.7860 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0147 | (-0.0519, 0.0160) | 0.7940 | -0.0147 | (-0.0413, 0.0169) | 0.8030 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0720 | (-0.0187, 0.1796) | 0.0613 | 0.0720 | (-0.0002, 0.1742) | 0.0303 |
| controlled_alt_vs_controlled_default | persona_style | 0.0013 | (-0.0130, 0.0143) | 0.4433 | 0.0013 | (0.0000, 0.0036) | 0.3373 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0048 | (-0.0227, 0.0105) | 0.6953 | -0.0048 | (-0.0174, 0.0078) | 0.7503 |
| controlled_alt_vs_controlled_default | length_score | -0.0292 | (-0.1500, 0.0708) | 0.6783 | -0.0292 | (-0.1283, 0.0449) | 0.7497 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0000 | (-0.0583, 0.0583) | 0.6093 | 0.0000 | (-0.0795, 0.0636) | 0.5533 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0084 | (-0.0441, 0.0607) | 0.3820 | 0.0084 | (-0.0329, 0.0662) | 0.3593 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.0530 | (-0.0008, 0.1100) | 0.0263 | 0.0530 | (-0.0065, 0.1296) | 0.0510 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.0996 | (0.0216, 0.1846) | 0.0040 | 0.0996 | (0.0110, 0.2236) | 0.0107 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0029 | (-0.0217, 0.0237) | 0.3967 | 0.0029 | (-0.0108, 0.0196) | 0.3267 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 0.0466 | (0.0023, 0.0918) | 0.0197 | 0.0466 | (-0.0045, 0.1128) | 0.0427 |
| controlled_alt_vs_proposed_raw | lore_consistency | 0.0287 | (-0.0031, 0.0605) | 0.0367 | 0.0287 | (-0.0013, 0.0662) | 0.0347 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_proposed_raw | objective_completion_support | 0.0006 | (-0.0203, 0.0217) | 0.4740 | 0.0006 | (-0.0174, 0.0194) | 0.4717 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 0.0125 | (-0.0046, 0.0310) | 0.0840 | 0.0125 | (-0.0062, 0.0372) | 0.0863 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | -0.0090 | (-0.0296, 0.0115) | 0.8097 | -0.0090 | (-0.0264, 0.0009) | 0.9483 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.0686 | (0.0026, 0.1402) | 0.0220 | 0.0686 | (-0.0075, 0.1645) | 0.0450 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0168 | (-0.0030, 0.0380) | 0.0480 | 0.0168 | (-0.0071, 0.0482) | 0.0903 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1339 | (0.0347, 0.2421) | 0.0043 | 0.1339 | (0.0255, 0.2857) | 0.0113 |
| controlled_alt_vs_proposed_raw | persona_style | -0.0378 | (-0.0703, -0.0117) | 1.0000 | -0.0378 | (-0.0853, 0.0000) | 1.0000 |
| controlled_alt_vs_proposed_raw | distinct1 | 0.0100 | (-0.0067, 0.0281) | 0.1310 | 0.0100 | (-0.0054, 0.0255) | 0.0950 |
| controlled_alt_vs_proposed_raw | length_score | -0.0056 | (-0.1181, 0.1014) | 0.5367 | -0.0056 | (-0.0718, 0.0727) | 0.5807 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0000 | (-0.0583, 0.0583) | 0.5970 | 0.0000 | (-0.0636, 0.0797) | 0.6163 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.0631 | (0.0163, 0.1145) | 0.0040 | 0.0631 | (-0.0002, 0.1455) | 0.0270 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.0862 | (0.0350, 0.1417) | 0.0000 | 0.0862 | (0.0230, 0.1746) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.0776 | (-0.0170, 0.1864) | 0.0540 | 0.0776 | (-0.0179, 0.2121) | 0.0673 |
| controlled_alt_vs_candidate_no_context | naturalness | -0.0032 | (-0.0287, 0.0211) | 0.5910 | -0.0032 | (-0.0292, 0.0241) | 0.6353 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 0.0813 | (0.0353, 0.1301) | 0.0000 | 0.0813 | (0.0226, 0.1635) | 0.0000 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 0.0525 | (0.0222, 0.0844) | 0.0000 | 0.0525 | (0.0172, 0.1018) | 0.0007 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_candidate_no_context | objective_completion_support | -0.0061 | (-0.0317, 0.0204) | 0.6887 | -0.0061 | (-0.0187, 0.0088) | 0.7957 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 0.0187 | (0.0024, 0.0372) | 0.0107 | 0.0187 | (-0.0024, 0.0484) | 0.0443 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | -0.0116 | (-0.0286, 0.0051) | 0.9127 | -0.0116 | (-0.0223, -0.0041) | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.1114 | (0.0455, 0.1826) | 0.0000 | 0.1114 | (0.0325, 0.2264) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0274 | (0.0101, 0.0459) | 0.0003 | 0.0274 | (0.0066, 0.0571) | 0.0003 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1062 | (-0.0079, 0.2391) | 0.0343 | 0.1062 | (-0.0092, 0.2774) | 0.0413 |
| controlled_alt_vs_candidate_no_context | persona_style | -0.0365 | (-0.0703, -0.0065) | 0.9943 | -0.0365 | (-0.0817, 0.0000) | 1.0000 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0049 | (-0.0223, 0.0126) | 0.7133 | -0.0049 | (-0.0161, 0.0074) | 0.7917 |
| controlled_alt_vs_candidate_no_context | length_score | 0.0083 | (-0.1181, 0.1403) | 0.4583 | 0.0083 | (-0.0617, 0.1042) | 0.4390 |
| controlled_alt_vs_candidate_no_context | sentence_score | -0.0292 | (-0.0875, 0.0292) | 0.9033 | -0.0292 | (-0.0955, 0.0000) | 1.0000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.0689 | (0.0148, 0.1241) | 0.0027 | 0.0689 | (0.0099, 0.1537) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 11 | 4 | 9 | 0.6458 | 0.7333 |
| proposed_vs_candidate_no_context | persona_consistency | 5 | 6 | 13 | 0.4792 | 0.4545 |
| proposed_vs_candidate_no_context | naturalness | 6 | 8 | 10 | 0.4583 | 0.4286 |
| proposed_vs_candidate_no_context | quest_state_correctness | 10 | 5 | 9 | 0.6042 | 0.6667 |
| proposed_vs_candidate_no_context | lore_consistency | 6 | 0 | 18 | 0.6250 | 1.0000 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| proposed_vs_candidate_no_context | objective_completion_support | 5 | 6 | 13 | 0.4792 | 0.4545 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 9 | 6 | 9 | 0.5625 | 0.6000 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | 6 | 6 | 12 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 6 | 1 | 17 | 0.6042 | 0.8571 |
| proposed_vs_candidate_no_context | context_overlap | 10 | 4 | 10 | 0.6250 | 0.7143 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 4 | 6 | 14 | 0.4583 | 0.4000 |
| proposed_vs_candidate_no_context | persona_style | 2 | 3 | 19 | 0.4792 | 0.4000 |
| proposed_vs_candidate_no_context | distinct1 | 3 | 9 | 12 | 0.3750 | 0.2500 |
| proposed_vs_candidate_no_context | length_score | 6 | 7 | 11 | 0.4792 | 0.4615 |
| proposed_vs_candidate_no_context | sentence_score | 1 | 3 | 20 | 0.4583 | 0.2500 |
| proposed_vs_candidate_no_context | overall_quality | 9 | 6 | 9 | 0.5625 | 0.6000 |
| controlled_vs_proposed_raw | context_relevance | 15 | 4 | 5 | 0.7292 | 0.7895 |
| controlled_vs_proposed_raw | persona_consistency | 10 | 6 | 8 | 0.5833 | 0.6250 |
| controlled_vs_proposed_raw | naturalness | 9 | 10 | 5 | 0.4792 | 0.4737 |
| controlled_vs_proposed_raw | quest_state_correctness | 14 | 5 | 5 | 0.6875 | 0.7368 |
| controlled_vs_proposed_raw | lore_consistency | 8 | 2 | 14 | 0.6250 | 0.8000 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_vs_proposed_raw | objective_completion_support | 4 | 6 | 14 | 0.4583 | 0.4000 |
| controlled_vs_proposed_raw | gameplay_usefulness | 12 | 7 | 5 | 0.6042 | 0.6316 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 6 | 7 | 11 | 0.4792 | 0.4615 |
| controlled_vs_proposed_raw | context_keyword_coverage | 12 | 3 | 9 | 0.6875 | 0.8000 |
| controlled_vs_proposed_raw | context_overlap | 13 | 6 | 5 | 0.6458 | 0.6842 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 10 | 5 | 9 | 0.6042 | 0.6667 |
| controlled_vs_proposed_raw | persona_style | 1 | 7 | 16 | 0.3750 | 0.1250 |
| controlled_vs_proposed_raw | distinct1 | 13 | 5 | 6 | 0.6667 | 0.7222 |
| controlled_vs_proposed_raw | length_score | 9 | 10 | 5 | 0.4792 | 0.4737 |
| controlled_vs_proposed_raw | sentence_score | 3 | 3 | 18 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | overall_quality | 12 | 7 | 5 | 0.6042 | 0.6316 |
| controlled_vs_candidate_no_context | context_relevance | 17 | 2 | 5 | 0.8125 | 0.8947 |
| controlled_vs_candidate_no_context | persona_consistency | 11 | 8 | 5 | 0.5625 | 0.5789 |
| controlled_vs_candidate_no_context | naturalness | 8 | 11 | 5 | 0.4375 | 0.4211 |
| controlled_vs_candidate_no_context | quest_state_correctness | 17 | 2 | 5 | 0.8125 | 0.8947 |
| controlled_vs_candidate_no_context | lore_consistency | 9 | 0 | 15 | 0.6875 | 1.0000 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_vs_candidate_no_context | objective_completion_support | 3 | 10 | 11 | 0.3542 | 0.2308 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 11 | 8 | 5 | 0.5625 | 0.5789 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 4 | 10 | 10 | 0.3750 | 0.2857 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 14 | 2 | 8 | 0.7500 | 0.8750 |
| controlled_vs_candidate_no_context | context_overlap | 16 | 3 | 5 | 0.7708 | 0.8421 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 11 | 7 | 6 | 0.5833 | 0.6111 |
| controlled_vs_candidate_no_context | persona_style | 1 | 6 | 17 | 0.3958 | 0.1429 |
| controlled_vs_candidate_no_context | distinct1 | 8 | 9 | 7 | 0.4792 | 0.4706 |
| controlled_vs_candidate_no_context | length_score | 10 | 9 | 5 | 0.5208 | 0.5263 |
| controlled_vs_candidate_no_context | sentence_score | 1 | 3 | 20 | 0.4583 | 0.2500 |
| controlled_vs_candidate_no_context | overall_quality | 14 | 5 | 5 | 0.6875 | 0.7368 |
| controlled_alt_vs_controlled_default | context_relevance | 8 | 7 | 9 | 0.5208 | 0.5333 |
| controlled_alt_vs_controlled_default | persona_consistency | 7 | 5 | 12 | 0.5417 | 0.5833 |
| controlled_alt_vs_controlled_default | naturalness | 8 | 7 | 9 | 0.5208 | 0.5333 |
| controlled_alt_vs_controlled_default | quest_state_correctness | 7 | 8 | 9 | 0.4792 | 0.4667 |
| controlled_alt_vs_controlled_default | lore_consistency | 7 | 6 | 11 | 0.5208 | 0.5385 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_alt_vs_controlled_default | objective_completion_support | 7 | 4 | 13 | 0.5625 | 0.6364 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | 6 | 9 | 9 | 0.4375 | 0.4000 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | 5 | 8 | 11 | 0.4375 | 0.3846 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 7 | 7 | 10 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | context_overlap | 8 | 7 | 9 | 0.5208 | 0.5333 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 7 | 5 | 12 | 0.5417 | 0.5833 |
| controlled_alt_vs_controlled_default | persona_style | 2 | 1 | 21 | 0.5208 | 0.6667 |
| controlled_alt_vs_controlled_default | distinct1 | 8 | 7 | 9 | 0.5208 | 0.5333 |
| controlled_alt_vs_controlled_default | length_score | 9 | 6 | 9 | 0.5625 | 0.6000 |
| controlled_alt_vs_controlled_default | sentence_score | 2 | 2 | 20 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | overall_quality | 8 | 7 | 9 | 0.5208 | 0.5333 |
| controlled_alt_vs_proposed_raw | context_relevance | 11 | 7 | 6 | 0.5833 | 0.6111 |
| controlled_alt_vs_proposed_raw | persona_consistency | 12 | 4 | 8 | 0.6667 | 0.7500 |
| controlled_alt_vs_proposed_raw | naturalness | 9 | 9 | 6 | 0.5000 | 0.5000 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 10 | 8 | 6 | 0.5417 | 0.5556 |
| controlled_alt_vs_proposed_raw | lore_consistency | 9 | 1 | 14 | 0.6667 | 0.9000 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_alt_vs_proposed_raw | objective_completion_support | 5 | 6 | 13 | 0.4792 | 0.4545 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 9 | 9 | 6 | 0.5000 | 0.5000 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | 5 | 7 | 12 | 0.4583 | 0.4167 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 9 | 5 | 10 | 0.5833 | 0.6429 |
| controlled_alt_vs_proposed_raw | context_overlap | 11 | 7 | 6 | 0.5833 | 0.6111 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 12 | 4 | 8 | 0.6667 | 0.7500 |
| controlled_alt_vs_proposed_raw | persona_style | 0 | 7 | 17 | 0.3542 | 0.0000 |
| controlled_alt_vs_proposed_raw | distinct1 | 11 | 6 | 7 | 0.6042 | 0.6471 |
| controlled_alt_vs_proposed_raw | length_score | 8 | 10 | 6 | 0.4583 | 0.4444 |
| controlled_alt_vs_proposed_raw | sentence_score | 2 | 2 | 20 | 0.5000 | 0.5000 |
| controlled_alt_vs_proposed_raw | overall_quality | 12 | 6 | 6 | 0.6250 | 0.6667 |
| controlled_alt_vs_candidate_no_context | context_relevance | 13 | 5 | 6 | 0.6667 | 0.7222 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 10 | 5 | 9 | 0.6042 | 0.6667 |
| controlled_alt_vs_candidate_no_context | naturalness | 9 | 9 | 6 | 0.5000 | 0.5000 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 13 | 5 | 6 | 0.6667 | 0.7222 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 9 | 0 | 15 | 0.6875 | 1.0000 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_alt_vs_candidate_no_context | objective_completion_support | 6 | 7 | 11 | 0.4792 | 0.4615 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 13 | 5 | 6 | 0.6667 | 0.7222 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | 5 | 8 | 11 | 0.4375 | 0.3846 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 10 | 3 | 11 | 0.6458 | 0.7692 |
| controlled_alt_vs_candidate_no_context | context_overlap | 14 | 4 | 6 | 0.7083 | 0.7778 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 9 | 4 | 11 | 0.6042 | 0.6923 |
| controlled_alt_vs_candidate_no_context | persona_style | 1 | 7 | 16 | 0.3750 | 0.1250 |
| controlled_alt_vs_candidate_no_context | distinct1 | 7 | 9 | 8 | 0.4583 | 0.4375 |
| controlled_alt_vs_candidate_no_context | length_score | 10 | 8 | 6 | 0.5417 | 0.5556 |
| controlled_alt_vs_candidate_no_context | sentence_score | 1 | 3 | 20 | 0.4583 | 0.2500 |
| controlled_alt_vs_candidate_no_context | overall_quality | 14 | 4 | 6 | 0.7083 | 0.7778 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.0833 | 0.1250 | 0.8750 |
| proposed_contextual_controlled_tuned | 0.0000 | 0.0000 | 0.0417 | 0.2917 | 0.7083 |
| proposed_contextual | 0.0000 | 0.0000 | 0.5417 | 0.0000 | 0.0000 |
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
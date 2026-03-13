# Proposal Alignment Evaluation Report

- Run ID: `20260313T060352Z`
- Generated: `2026-03-13T06:08:19.150491+00:00`
- Scenarios: `artifacts\proposal_control_tuning\auto_tune\20260313T051946Z\train_runs\trial_004\seed_19\20260313T060352Z\scenarios.jsonl`
- Scenario count: `24`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_tuned`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0787 (0.0394, 0.1241) | 0.2265 (0.1846, 0.2715) | 0.8844 (0.8699, 0.8995) | 0.2899 (0.2587, 0.3215) | n/a |
| proposed_contextual_controlled_tuned | 0.0683 (0.0388, 0.0996) | 0.2553 (0.1974, 0.3183) | 0.8719 (0.8620, 0.8828) | 0.2932 (0.2638, 0.3241) | n/a |
| proposed_contextual | 0.0653 (0.0352, 0.0989) | 0.1815 (0.1358, 0.2327) | 0.8810 (0.8623, 0.9001) | 0.2659 (0.2396, 0.2902) | n/a |
| candidate_no_context | 0.0381 (0.0234, 0.0544) | 0.2499 (0.1940, 0.3081) | 0.8860 (0.8680, 0.9053) | 0.2795 (0.2537, 0.3071) | n/a |

## Game-facing Outcome Metrics (mean, 95% CI)
| Arm | Quest-state Correctness | Lore Consistency | Contradiction Safety | Objective Completion Support | Gameplay Usefulness | Time-pressure Acceptability |
|---|---:|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.1664 (0.1319, 0.2074) | 0.0307 (0.0089, 0.0566) | 1.0000 (1.0000, 1.0000) | 0.0826 (0.0560, 0.1080) | 0.3086 (0.2939, 0.3246) | 0.3081 (0.2930, 0.3226) |
| proposed_contextual_controlled_tuned | 0.1556 (0.1297, 0.1859) | 0.0182 (0.0037, 0.0374) | 1.0000 (1.0000, 1.0000) | 0.0781 (0.0509, 0.1034) | 0.2974 (0.2864, 0.3089) | 0.2977 (0.2824, 0.3129) |
| proposed_contextual | 0.1525 (0.1266, 0.1824) | 0.0243 (0.0102, 0.0408) | 1.0000 (1.0000, 1.0000) | 0.0742 (0.0493, 0.0988) | 0.3002 (0.2852, 0.3137) | 0.3025 (0.2884, 0.3168) |
| candidate_no_context | 0.1287 (0.1164, 0.1419) | 0.0078 (0.0009, 0.0172) | 1.0000 (1.0000, 1.0000) | 0.0764 (0.0550, 0.0966) | 0.2930 (0.2834, 0.3026) | 0.3022 (0.2906, 0.3160) |

- Multi-turn contradiction rate is reported as `1 - contradiction_safety` in row-level outputs.

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0272 | 0.7146 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0684 | -0.2737 |
| proposed_vs_candidate_no_context | naturalness | -0.0050 | -0.0056 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.0238 | 0.1846 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0166 | 2.1320 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | -0.0022 | -0.0282 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 0.0073 | 0.0248 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | 0.0002 | 0.0007 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0330 | 0.8447 |
| proposed_vs_candidate_no_context | context_overlap | 0.0138 | 0.3842 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0754 | -0.4294 |
| proposed_vs_candidate_no_context | persona_style | -0.0404 | -0.0738 |
| proposed_vs_candidate_no_context | distinct1 | 0.0055 | 0.0060 |
| proposed_vs_candidate_no_context | length_score | -0.0069 | -0.0120 |
| proposed_vs_candidate_no_context | sentence_score | -0.0583 | -0.0583 |
| proposed_vs_candidate_no_context | overall_quality | -0.0136 | -0.0487 |
| controlled_vs_proposed_raw | context_relevance | 0.0134 | 0.2056 |
| controlled_vs_proposed_raw | persona_consistency | 0.0450 | 0.2482 |
| controlled_vs_proposed_raw | naturalness | 0.0035 | 0.0039 |
| controlled_vs_proposed_raw | quest_state_correctness | 0.0140 | 0.0916 |
| controlled_vs_proposed_raw | lore_consistency | 0.0064 | 0.2616 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0083 | 0.1123 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0084 | 0.0279 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 0.0056 | 0.0186 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.0193 | 0.2684 |
| controlled_vs_proposed_raw | context_overlap | -0.0004 | -0.0071 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.0569 | 0.5683 |
| controlled_vs_proposed_raw | persona_style | -0.0026 | -0.0051 |
| controlled_vs_proposed_raw | distinct1 | -0.0051 | -0.0054 |
| controlled_vs_proposed_raw | length_score | 0.0056 | 0.0097 |
| controlled_vs_proposed_raw | sentence_score | 0.0437 | 0.0465 |
| controlled_vs_proposed_raw | overall_quality | 0.0240 | 0.0902 |
| controlled_vs_candidate_no_context | context_relevance | 0.0406 | 1.0670 |
| controlled_vs_candidate_no_context | persona_consistency | -0.0234 | -0.0935 |
| controlled_vs_candidate_no_context | naturalness | -0.0015 | -0.0017 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.0377 | 0.2932 |
| controlled_vs_candidate_no_context | lore_consistency | 0.0229 | 2.9513 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | 0.0062 | 0.0809 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0156 | 0.0534 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 0.0058 | 0.0193 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.0523 | 1.3398 |
| controlled_vs_candidate_no_context | context_overlap | 0.0134 | 0.3744 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | -0.0185 | -0.1051 |
| controlled_vs_candidate_no_context | persona_style | -0.0430 | -0.0786 |
| controlled_vs_candidate_no_context | distinct1 | 0.0005 | 0.0005 |
| controlled_vs_candidate_no_context | length_score | -0.0014 | -0.0024 |
| controlled_vs_candidate_no_context | sentence_score | -0.0146 | -0.0146 |
| controlled_vs_candidate_no_context | overall_quality | 0.0104 | 0.0371 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0104 | -0.1317 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0288 | 0.1272 |
| controlled_alt_vs_controlled_default | naturalness | -0.0125 | -0.0142 |
| controlled_alt_vs_controlled_default | quest_state_correctness | -0.0109 | -0.0653 |
| controlled_alt_vs_controlled_default | lore_consistency | -0.0125 | -0.4082 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_controlled_default | objective_completion_support | -0.0045 | -0.0547 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | -0.0112 | -0.0362 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | -0.0104 | -0.0337 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0144 | -0.1577 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0009 | -0.0192 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0337 | 0.2146 |
| controlled_alt_vs_controlled_default | persona_style | 0.0091 | 0.0181 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0020 | -0.0022 |
| controlled_alt_vs_controlled_default | length_score | -0.0514 | -0.0889 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0146 | -0.0148 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0033 | 0.0113 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.0031 | 0.0468 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.0738 | 0.4069 |
| controlled_alt_vs_proposed_raw | naturalness | -0.0091 | -0.0103 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 0.0031 | 0.0203 |
| controlled_alt_vs_proposed_raw | lore_consistency | -0.0062 | -0.2533 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_proposed_raw | objective_completion_support | 0.0038 | 0.0514 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | -0.0028 | -0.0093 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | -0.0048 | -0.0158 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.0049 | 0.0684 |
| controlled_alt_vs_proposed_raw | context_overlap | -0.0013 | -0.0262 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.0907 | 0.9050 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0065 | 0.0129 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0071 | -0.0076 |
| controlled_alt_vs_proposed_raw | length_score | -0.0458 | -0.0801 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0292 | 0.0310 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.0273 | 0.1026 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.0303 | 0.7949 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.0055 | 0.0218 |
| controlled_alt_vs_candidate_no_context | naturalness | -0.0141 | -0.0159 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 0.0269 | 0.2087 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 0.0104 | 1.3386 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_candidate_no_context | objective_completion_support | 0.0017 | 0.0218 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 0.0045 | 0.0153 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | -0.0046 | -0.0151 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.0379 | 0.9709 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0125 | 0.3479 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.0153 | 0.0870 |
| controlled_alt_vs_candidate_no_context | persona_style | -0.0339 | -0.0619 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0015 | -0.0017 |
| controlled_alt_vs_candidate_no_context | length_score | -0.0528 | -0.0911 |
| controlled_alt_vs_candidate_no_context | sentence_score | -0.0292 | -0.0292 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.0137 | 0.0489 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0272 | (-0.0019, 0.0577) | 0.0357 | 0.0272 | (-0.0013, 0.0642) | 0.0713 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0684 | (-0.1394, 0.0008) | 0.9737 | -0.0684 | (-0.1430, -0.0046) | 0.9850 |
| proposed_vs_candidate_no_context | naturalness | -0.0050 | (-0.0216, 0.0103) | 0.7313 | -0.0050 | (-0.0238, 0.0179) | 0.6547 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.0238 | (-0.0010, 0.0487) | 0.0310 | 0.0238 | (-0.0020, 0.0593) | 0.0703 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0166 | (0.0026, 0.0322) | 0.0120 | 0.0166 | (0.0036, 0.0336) | 0.0117 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | -0.0022 | (-0.0213, 0.0165) | 0.5763 | -0.0022 | (-0.0156, 0.0181) | 0.6410 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 0.0073 | (-0.0071, 0.0218) | 0.1563 | 0.0073 | (-0.0105, 0.0294) | 0.2443 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | 0.0002 | (-0.0121, 0.0132) | 0.4990 | 0.0002 | (-0.0154, 0.0179) | 0.5097 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0330 | (-0.0053, 0.0716) | 0.0467 | 0.0330 | (-0.0014, 0.0809) | 0.0797 |
| proposed_vs_candidate_no_context | context_overlap | 0.0138 | (0.0013, 0.0278) | 0.0137 | 0.0138 | (-0.0029, 0.0334) | 0.0467 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0754 | (-0.1627, 0.0080) | 0.9627 | -0.0754 | (-0.1786, 0.0000) | 0.9833 |
| proposed_vs_candidate_no_context | persona_style | -0.0404 | (-0.1198, 0.0156) | 0.9000 | -0.0404 | (-0.1310, 0.0341) | 0.7450 |
| proposed_vs_candidate_no_context | distinct1 | 0.0055 | (-0.0116, 0.0226) | 0.2530 | 0.0055 | (-0.0113, 0.0209) | 0.2293 |
| proposed_vs_candidate_no_context | length_score | -0.0069 | (-0.0848, 0.0750) | 0.5593 | -0.0069 | (-0.0987, 0.1091) | 0.5960 |
| proposed_vs_candidate_no_context | sentence_score | -0.0583 | (-0.1167, -0.0146) | 1.0000 | -0.0583 | (-0.1225, -0.0135) | 1.0000 |
| proposed_vs_candidate_no_context | overall_quality | -0.0136 | (-0.0454, 0.0182) | 0.7970 | -0.0136 | (-0.0524, 0.0237) | 0.7537 |
| controlled_vs_proposed_raw | context_relevance | 0.0134 | (-0.0131, 0.0409) | 0.1697 | 0.0134 | (-0.0170, 0.0395) | 0.1903 |
| controlled_vs_proposed_raw | persona_consistency | 0.0450 | (-0.0056, 0.1013) | 0.0470 | 0.0450 | (0.0108, 0.0767) | 0.0067 |
| controlled_vs_proposed_raw | naturalness | 0.0035 | (-0.0179, 0.0262) | 0.4023 | 0.0035 | (-0.0113, 0.0137) | 0.3120 |
| controlled_vs_proposed_raw | quest_state_correctness | 0.0140 | (-0.0088, 0.0389) | 0.1167 | 0.0140 | (-0.0141, 0.0385) | 0.1753 |
| controlled_vs_proposed_raw | lore_consistency | 0.0064 | (-0.0118, 0.0282) | 0.2657 | 0.0064 | (-0.0089, 0.0216) | 0.2557 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0083 | (-0.0054, 0.0247) | 0.1413 | 0.0083 | (-0.0015, 0.0208) | 0.0463 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0084 | (-0.0037, 0.0225) | 0.0970 | 0.0084 | (0.0009, 0.0158) | 0.0083 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 0.0056 | (-0.0037, 0.0158) | 0.1203 | 0.0056 | (-0.0036, 0.0159) | 0.1100 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.0193 | (-0.0152, 0.0572) | 0.1370 | 0.0193 | (-0.0210, 0.0534) | 0.2020 |
| controlled_vs_proposed_raw | context_overlap | -0.0004 | (-0.0128, 0.0127) | 0.5280 | -0.0004 | (-0.0105, 0.0098) | 0.5357 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.0569 | (0.0014, 0.1153) | 0.0243 | 0.0569 | (0.0278, 0.0924) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | -0.0026 | (-0.0625, 0.0846) | 0.5830 | -0.0026 | (-0.0682, 0.0469) | 0.6333 |
| controlled_vs_proposed_raw | distinct1 | -0.0051 | (-0.0262, 0.0141) | 0.6937 | -0.0051 | (-0.0270, 0.0124) | 0.6493 |
| controlled_vs_proposed_raw | length_score | 0.0056 | (-0.0806, 0.0986) | 0.4573 | 0.0056 | (-0.0697, 0.0449) | 0.3840 |
| controlled_vs_proposed_raw | sentence_score | 0.0437 | (-0.0146, 0.1021) | 0.1230 | 0.0437 | (0.0135, 0.0795) | 0.0123 |
| controlled_vs_proposed_raw | overall_quality | 0.0240 | (-0.0035, 0.0549) | 0.0447 | 0.0240 | (0.0040, 0.0447) | 0.0073 |
| controlled_vs_candidate_no_context | context_relevance | 0.0406 | (0.0019, 0.0829) | 0.0210 | 0.0406 | (-0.0094, 0.0996) | 0.0800 |
| controlled_vs_candidate_no_context | persona_consistency | -0.0234 | (-0.0874, 0.0367) | 0.7523 | -0.0234 | (-0.1303, 0.0667) | 0.6867 |
| controlled_vs_candidate_no_context | naturalness | -0.0015 | (-0.0226, 0.0199) | 0.5477 | -0.0015 | (-0.0240, 0.0209) | 0.5500 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.0377 | (0.0017, 0.0775) | 0.0187 | 0.0377 | (-0.0088, 0.0969) | 0.0637 |
| controlled_vs_candidate_no_context | lore_consistency | 0.0229 | (-0.0006, 0.0487) | 0.0300 | 0.0229 | (0.0063, 0.0421) | 0.0083 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | 0.0062 | (-0.0174, 0.0303) | 0.3173 | 0.0062 | (-0.0137, 0.0383) | 0.3473 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0156 | (0.0013, 0.0301) | 0.0170 | 0.0156 | (-0.0082, 0.0401) | 0.0657 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 0.0058 | (-0.0086, 0.0208) | 0.2307 | 0.0058 | (-0.0098, 0.0298) | 0.2877 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.0523 | (0.0030, 0.1061) | 0.0170 | 0.0523 | (-0.0119, 0.1364) | 0.0797 |
| controlled_vs_candidate_no_context | context_overlap | 0.0134 | (-0.0004, 0.0282) | 0.0297 | 0.0134 | (-0.0032, 0.0333) | 0.0627 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | -0.0185 | (-0.0954, 0.0506) | 0.6733 | -0.0185 | (-0.1429, 0.0833) | 0.5950 |
| controlled_vs_candidate_no_context | persona_style | -0.0430 | (-0.0768, -0.0130) | 1.0000 | -0.0430 | (-0.0964, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | distinct1 | 0.0005 | (-0.0196, 0.0228) | 0.4997 | 0.0005 | (-0.0188, 0.0165) | 0.4813 |
| controlled_vs_candidate_no_context | length_score | -0.0014 | (-0.0875, 0.0833) | 0.5183 | -0.0014 | (-0.1014, 0.1121) | 0.5247 |
| controlled_vs_candidate_no_context | sentence_score | -0.0146 | (-0.0437, 0.0000) | 1.0000 | -0.0146 | (-0.0477, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | overall_quality | 0.0104 | (-0.0284, 0.0523) | 0.3023 | 0.0104 | (-0.0490, 0.0662) | 0.3563 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0104 | (-0.0560, 0.0349) | 0.6763 | -0.0104 | (-0.0627, 0.0233) | 0.6557 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0288 | (-0.0267, 0.0927) | 0.1723 | 0.0288 | (-0.0364, 0.1206) | 0.2733 |
| controlled_alt_vs_controlled_default | naturalness | -0.0125 | (-0.0312, 0.0035) | 0.9343 | -0.0125 | (-0.0250, -0.0034) | 0.9980 |
| controlled_alt_vs_controlled_default | quest_state_correctness | -0.0109 | (-0.0527, 0.0284) | 0.7013 | -0.0109 | (-0.0605, 0.0217) | 0.6583 |
| controlled_alt_vs_controlled_default | lore_consistency | -0.0125 | (-0.0370, 0.0108) | 0.8400 | -0.0125 | (-0.0326, -0.0010) | 1.0000 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | objective_completion_support | -0.0045 | (-0.0235, 0.0145) | 0.6883 | -0.0045 | (-0.0207, 0.0125) | 0.7263 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | -0.0112 | (-0.0255, 0.0017) | 0.9423 | -0.0112 | (-0.0231, -0.0005) | 0.9813 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | -0.0104 | (-0.0248, 0.0026) | 0.9450 | -0.0104 | (-0.0278, 0.0022) | 0.9390 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0144 | (-0.0743, 0.0417) | 0.6657 | -0.0144 | (-0.0818, 0.0294) | 0.6617 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0009 | (-0.0201, 0.0182) | 0.5540 | -0.0009 | (-0.0180, 0.0103) | 0.5367 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0337 | (-0.0306, 0.1149) | 0.1760 | 0.0337 | (-0.0455, 0.1498) | 0.2943 |
| controlled_alt_vs_controlled_default | persona_style | 0.0091 | (0.0000, 0.0234) | 0.1210 | 0.0091 | (0.0000, 0.0252) | 0.3373 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0020 | (-0.0225, 0.0183) | 0.5490 | -0.0020 | (-0.0191, 0.0158) | 0.5773 |
| controlled_alt_vs_controlled_default | length_score | -0.0514 | (-0.1097, 0.0042) | 0.9640 | -0.0514 | (-0.1133, -0.0013) | 0.9837 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0146 | (-0.0437, 0.0000) | 1.0000 | -0.0146 | (-0.0477, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0033 | (-0.0353, 0.0445) | 0.4443 | 0.0033 | (-0.0416, 0.0489) | 0.4593 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.0031 | (-0.0393, 0.0449) | 0.4307 | 0.0031 | (-0.0376, 0.0340) | 0.4180 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.0738 | (0.0039, 0.1491) | 0.0207 | 0.0738 | (0.0300, 0.1355) | 0.0000 |
| controlled_alt_vs_proposed_raw | naturalness | -0.0091 | (-0.0281, 0.0117) | 0.8157 | -0.0091 | (-0.0328, 0.0077) | 0.8437 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 0.0031 | (-0.0323, 0.0366) | 0.4270 | 0.0031 | (-0.0311, 0.0286) | 0.4080 |
| controlled_alt_vs_proposed_raw | lore_consistency | -0.0062 | (-0.0250, 0.0134) | 0.7307 | -0.0062 | (-0.0242, 0.0101) | 0.7750 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_proposed_raw | objective_completion_support | 0.0038 | (-0.0169, 0.0248) | 0.3527 | 0.0038 | (-0.0061, 0.0210) | 0.3073 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | -0.0028 | (-0.0180, 0.0138) | 0.6613 | -0.0028 | (-0.0144, 0.0055) | 0.6937 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | -0.0048 | (-0.0198, 0.0086) | 0.7467 | -0.0048 | (-0.0167, 0.0058) | 0.8183 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.0049 | (-0.0508, 0.0583) | 0.4137 | 0.0049 | (-0.0434, 0.0417) | 0.3773 |
| controlled_alt_vs_proposed_raw | context_overlap | -0.0013 | (-0.0186, 0.0165) | 0.5703 | -0.0013 | (-0.0242, 0.0165) | 0.5313 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.0907 | (0.0083, 0.1776) | 0.0113 | 0.0907 | (0.0375, 0.1861) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0065 | (-0.0586, 0.1029) | 0.4783 | 0.0065 | (-0.0682, 0.0703) | 0.4247 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0071 | (-0.0282, 0.0141) | 0.7453 | -0.0071 | (-0.0250, 0.0085) | 0.8103 |
| controlled_alt_vs_proposed_raw | length_score | -0.0458 | (-0.1347, 0.0556) | 0.8390 | -0.0458 | (-0.1617, 0.0369) | 0.8220 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0292 | (-0.0292, 0.0875) | 0.2200 | 0.0292 | (0.0000, 0.0583) | 0.0773 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.0273 | (-0.0101, 0.0680) | 0.0827 | 0.0273 | (-0.0027, 0.0587) | 0.0253 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.0303 | (-0.0038, 0.0704) | 0.0447 | 0.0303 | (-0.0001, 0.0693) | 0.0283 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.0055 | (-0.0552, 0.0691) | 0.4307 | 0.0055 | (-0.0314, 0.0409) | 0.3750 |
| controlled_alt_vs_candidate_no_context | naturalness | -0.0141 | (-0.0316, 0.0013) | 0.9607 | -0.0141 | (-0.0352, 0.0031) | 0.9437 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 0.0269 | (-0.0030, 0.0609) | 0.0423 | 0.0269 | (0.0003, 0.0646) | 0.0240 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 0.0104 | (-0.0041, 0.0300) | 0.1170 | 0.0104 | (-0.0045, 0.0275) | 0.0993 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_candidate_no_context | objective_completion_support | 0.0017 | (-0.0173, 0.0219) | 0.4517 | 0.0017 | (-0.0138, 0.0211) | 0.4373 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 0.0045 | (-0.0069, 0.0162) | 0.2280 | 0.0045 | (-0.0088, 0.0226) | 0.2537 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | -0.0046 | (-0.0182, 0.0090) | 0.7437 | -0.0046 | (-0.0104, 0.0020) | 0.9027 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.0379 | (-0.0083, 0.0864) | 0.0577 | 0.0379 | (0.0000, 0.0909) | 0.0343 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0125 | (-0.0015, 0.0286) | 0.0430 | 0.0125 | (-0.0004, 0.0278) | 0.0370 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.0153 | (-0.0615, 0.0923) | 0.3517 | 0.0153 | (-0.0364, 0.0530) | 0.2220 |
| controlled_alt_vs_candidate_no_context | persona_style | -0.0339 | (-0.0716, 0.0000) | 0.9797 | -0.0339 | (-0.0745, 0.0000) | 1.0000 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0015 | (-0.0207, 0.0178) | 0.5777 | -0.0015 | (-0.0134, 0.0087) | 0.6180 |
| controlled_alt_vs_candidate_no_context | length_score | -0.0528 | (-0.1222, 0.0139) | 0.9420 | -0.0528 | (-0.1400, 0.0264) | 0.9037 |
| controlled_alt_vs_candidate_no_context | sentence_score | -0.0292 | (-0.0729, 0.0000) | 1.0000 | -0.0292 | (-0.0955, 0.0000) | 1.0000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.0137 | (-0.0217, 0.0494) | 0.2280 | 0.0137 | (-0.0128, 0.0438) | 0.1703 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 9 | 6 | 9 | 0.5625 | 0.6000 |
| proposed_vs_candidate_no_context | persona_consistency | 5 | 9 | 10 | 0.4167 | 0.3571 |
| proposed_vs_candidate_no_context | naturalness | 5 | 10 | 9 | 0.3958 | 0.3333 |
| proposed_vs_candidate_no_context | quest_state_correctness | 9 | 6 | 9 | 0.5625 | 0.6000 |
| proposed_vs_candidate_no_context | lore_consistency | 9 | 1 | 14 | 0.6667 | 0.9000 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| proposed_vs_candidate_no_context | objective_completion_support | 4 | 6 | 14 | 0.4583 | 0.4000 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 9 | 6 | 9 | 0.5625 | 0.6000 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | 5 | 6 | 13 | 0.4792 | 0.4545 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 6 | 2 | 16 | 0.5833 | 0.7500 |
| proposed_vs_candidate_no_context | context_overlap | 11 | 4 | 9 | 0.6458 | 0.7333 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 5 | 9 | 10 | 0.4167 | 0.3571 |
| proposed_vs_candidate_no_context | persona_style | 1 | 4 | 19 | 0.4375 | 0.2000 |
| proposed_vs_candidate_no_context | distinct1 | 8 | 6 | 10 | 0.5417 | 0.5714 |
| proposed_vs_candidate_no_context | length_score | 7 | 7 | 10 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | sentence_score | 0 | 4 | 20 | 0.4167 | 0.0000 |
| proposed_vs_candidate_no_context | overall_quality | 7 | 8 | 9 | 0.4792 | 0.4667 |
| controlled_vs_proposed_raw | context_relevance | 12 | 6 | 6 | 0.6250 | 0.6667 |
| controlled_vs_proposed_raw | persona_consistency | 8 | 5 | 11 | 0.5625 | 0.6154 |
| controlled_vs_proposed_raw | naturalness | 8 | 10 | 6 | 0.4583 | 0.4444 |
| controlled_vs_proposed_raw | quest_state_correctness | 11 | 7 | 6 | 0.5833 | 0.6111 |
| controlled_vs_proposed_raw | lore_consistency | 6 | 5 | 13 | 0.5208 | 0.5455 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_vs_proposed_raw | objective_completion_support | 4 | 2 | 18 | 0.5417 | 0.6667 |
| controlled_vs_proposed_raw | gameplay_usefulness | 9 | 9 | 6 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 5 | 2 | 17 | 0.5625 | 0.7143 |
| controlled_vs_proposed_raw | context_keyword_coverage | 5 | 3 | 16 | 0.5417 | 0.6250 |
| controlled_vs_proposed_raw | context_overlap | 9 | 9 | 6 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 8 | 3 | 13 | 0.6042 | 0.7273 |
| controlled_vs_proposed_raw | persona_style | 2 | 6 | 16 | 0.4167 | 0.2500 |
| controlled_vs_proposed_raw | distinct1 | 10 | 7 | 7 | 0.5625 | 0.5882 |
| controlled_vs_proposed_raw | length_score | 7 | 10 | 7 | 0.4375 | 0.4118 |
| controlled_vs_proposed_raw | sentence_score | 4 | 1 | 19 | 0.5625 | 0.8000 |
| controlled_vs_proposed_raw | overall_quality | 10 | 8 | 6 | 0.5417 | 0.5556 |
| controlled_vs_candidate_no_context | context_relevance | 9 | 8 | 7 | 0.5208 | 0.5294 |
| controlled_vs_candidate_no_context | persona_consistency | 8 | 7 | 9 | 0.5208 | 0.5333 |
| controlled_vs_candidate_no_context | naturalness | 8 | 9 | 7 | 0.4792 | 0.4706 |
| controlled_vs_candidate_no_context | quest_state_correctness | 9 | 8 | 7 | 0.5208 | 0.5294 |
| controlled_vs_candidate_no_context | lore_consistency | 5 | 4 | 15 | 0.5208 | 0.5556 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_vs_candidate_no_context | objective_completion_support | 7 | 7 | 10 | 0.5000 | 0.5000 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 11 | 6 | 7 | 0.6042 | 0.6471 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 7 | 7 | 10 | 0.5000 | 0.5000 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 6 | 2 | 16 | 0.5833 | 0.7500 |
| controlled_vs_candidate_no_context | context_overlap | 9 | 8 | 7 | 0.5208 | 0.5294 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 8 | 6 | 10 | 0.5417 | 0.5714 |
| controlled_vs_candidate_no_context | persona_style | 0 | 6 | 18 | 0.3750 | 0.0000 |
| controlled_vs_candidate_no_context | distinct1 | 9 | 8 | 7 | 0.5208 | 0.5294 |
| controlled_vs_candidate_no_context | length_score | 7 | 9 | 8 | 0.4583 | 0.4375 |
| controlled_vs_candidate_no_context | sentence_score | 0 | 1 | 23 | 0.4792 | 0.0000 |
| controlled_vs_candidate_no_context | overall_quality | 10 | 7 | 7 | 0.5625 | 0.5882 |
| controlled_alt_vs_controlled_default | context_relevance | 7 | 5 | 12 | 0.5417 | 0.5833 |
| controlled_alt_vs_controlled_default | persona_consistency | 5 | 4 | 15 | 0.5208 | 0.5556 |
| controlled_alt_vs_controlled_default | naturalness | 4 | 8 | 12 | 0.4167 | 0.3333 |
| controlled_alt_vs_controlled_default | quest_state_correctness | 6 | 6 | 12 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | lore_consistency | 4 | 4 | 16 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_alt_vs_controlled_default | objective_completion_support | 3 | 6 | 15 | 0.4375 | 0.3333 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | 5 | 7 | 12 | 0.4583 | 0.4167 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | 3 | 8 | 13 | 0.3958 | 0.2727 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 4 | 5 | 15 | 0.4792 | 0.4444 |
| controlled_alt_vs_controlled_default | context_overlap | 7 | 5 | 12 | 0.5417 | 0.5833 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 4 | 4 | 16 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | persona_style | 2 | 0 | 22 | 0.5417 | 1.0000 |
| controlled_alt_vs_controlled_default | distinct1 | 7 | 5 | 12 | 0.5417 | 0.5833 |
| controlled_alt_vs_controlled_default | length_score | 5 | 7 | 12 | 0.4583 | 0.4167 |
| controlled_alt_vs_controlled_default | sentence_score | 0 | 1 | 23 | 0.4792 | 0.0000 |
| controlled_alt_vs_controlled_default | overall_quality | 6 | 6 | 12 | 0.5000 | 0.5000 |
| controlled_alt_vs_proposed_raw | context_relevance | 11 | 8 | 5 | 0.5625 | 0.5789 |
| controlled_alt_vs_proposed_raw | persona_consistency | 10 | 6 | 8 | 0.5833 | 0.6250 |
| controlled_alt_vs_proposed_raw | naturalness | 7 | 12 | 5 | 0.3958 | 0.3684 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 10 | 9 | 5 | 0.5208 | 0.5263 |
| controlled_alt_vs_proposed_raw | lore_consistency | 4 | 7 | 13 | 0.4375 | 0.3636 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_alt_vs_proposed_raw | objective_completion_support | 5 | 5 | 14 | 0.5000 | 0.5000 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 7 | 12 | 5 | 0.3958 | 0.3684 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | 5 | 6 | 13 | 0.4792 | 0.4545 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 7 | 5 | 12 | 0.5417 | 0.5833 |
| controlled_alt_vs_proposed_raw | context_overlap | 8 | 11 | 5 | 0.4375 | 0.4211 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 10 | 5 | 9 | 0.6042 | 0.6667 |
| controlled_alt_vs_proposed_raw | persona_style | 2 | 6 | 16 | 0.4167 | 0.2500 |
| controlled_alt_vs_proposed_raw | distinct1 | 10 | 9 | 5 | 0.5208 | 0.5263 |
| controlled_alt_vs_proposed_raw | length_score | 7 | 12 | 5 | 0.3958 | 0.3684 |
| controlled_alt_vs_proposed_raw | sentence_score | 3 | 1 | 20 | 0.5417 | 0.7500 |
| controlled_alt_vs_proposed_raw | overall_quality | 10 | 9 | 5 | 0.5208 | 0.5263 |
| controlled_alt_vs_candidate_no_context | context_relevance | 8 | 8 | 8 | 0.5000 | 0.5000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 8 | 5 | 11 | 0.5625 | 0.6154 |
| controlled_alt_vs_candidate_no_context | naturalness | 7 | 9 | 8 | 0.4583 | 0.4375 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 9 | 7 | 8 | 0.5417 | 0.5625 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 3 | 3 | 18 | 0.5000 | 0.5000 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_alt_vs_candidate_no_context | objective_completion_support | 4 | 7 | 13 | 0.4375 | 0.3636 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 9 | 7 | 8 | 0.5417 | 0.5625 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | 4 | 8 | 12 | 0.4167 | 0.3333 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 6 | 3 | 15 | 0.5625 | 0.6667 |
| controlled_alt_vs_candidate_no_context | context_overlap | 8 | 8 | 8 | 0.5000 | 0.5000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 7 | 4 | 13 | 0.5625 | 0.6364 |
| controlled_alt_vs_candidate_no_context | persona_style | 1 | 5 | 18 | 0.4167 | 0.1667 |
| controlled_alt_vs_candidate_no_context | distinct1 | 8 | 7 | 9 | 0.5208 | 0.5333 |
| controlled_alt_vs_candidate_no_context | length_score | 6 | 9 | 9 | 0.4375 | 0.4000 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0 | 2 | 22 | 0.4583 | 0.0000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 11 | 5 | 8 | 0.6250 | 0.6875 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.1667 | 0.1250 | 0.8750 |
| proposed_contextual_controlled_tuned | 0.0000 | 0.0000 | 0.2083 | 0.2083 | 0.7917 |
| proposed_contextual | 0.0000 | 0.0000 | 0.5417 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.5833 | 0.0000 | 0.0000 |

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
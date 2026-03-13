# Proposal Alignment Evaluation Report

- Run ID: `20260313T063921Z`
- Generated: `2026-03-13T06:44:40.199070+00:00`
- Scenarios: `artifacts\proposal_control_tuning\auto_tune\20260313T051946Z\train_runs\trial_007\seed_19\20260313T063921Z\scenarios.jsonl`
- Scenario count: `24`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_tuned`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.1395 (0.0785, 0.2117) | 0.2388 (0.1921, 0.2900) | 0.8778 (0.8617, 0.8940) | 0.3209 (0.2760, 0.3697) | n/a |
| proposed_contextual_controlled_tuned | 0.0701 (0.0356, 0.1114) | 0.2488 (0.1970, 0.3075) | 0.8748 (0.8568, 0.8916) | 0.2919 (0.2629, 0.3228) | n/a |
| proposed_contextual | 0.0797 (0.0427, 0.1218) | 0.2282 (0.1696, 0.2932) | 0.8771 (0.8607, 0.8951) | 0.2889 (0.2585, 0.3211) | n/a |
| candidate_no_context | 0.0203 (0.0108, 0.0320) | 0.2594 (0.1997, 0.3263) | 0.8799 (0.8636, 0.8971) | 0.2736 (0.2469, 0.3024) | n/a |

## Game-facing Outcome Metrics (mean, 95% CI)
| Arm | Quest-state Correctness | Lore Consistency | Contradiction Safety | Objective Completion Support | Gameplay Usefulness | Time-pressure Acceptability |
|---|---:|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2177 (0.1609, 0.2755) | 0.0750 (0.0326, 0.1233) | 1.0000 (1.0000, 1.0000) | 0.0798 (0.0544, 0.1041) | 0.3160 (0.2991, 0.3333) | 0.3005 (0.2824, 0.3169) |
| proposed_contextual_controlled_tuned | 0.1592 (0.1265, 0.1969) | 0.0303 (0.0091, 0.0596) | 1.0000 (1.0000, 1.0000) | 0.0734 (0.0461, 0.1013) | 0.2964 (0.2814, 0.3106) | 0.2973 (0.2793, 0.3150) |
| proposed_contextual | 0.1631 (0.1328, 0.1972) | 0.0281 (0.0091, 0.0501) | 1.0000 (1.0000, 1.0000) | 0.0618 (0.0394, 0.0858) | 0.2943 (0.2830, 0.3054) | 0.2885 (0.2720, 0.3054) |
| candidate_no_context | 0.1138 (0.1066, 0.1219) | 0.0014 (0.0000, 0.0037) | 1.0000 (1.0000, 1.0000) | 0.0701 (0.0474, 0.0917) | 0.2824 (0.2715, 0.2929) | 0.2932 (0.2794, 0.3090) |

- Multi-turn contradiction rate is reported as `1 - contradiction_safety` in row-level outputs.

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0594 | 2.9195 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0312 | -0.1203 |
| proposed_vs_candidate_no_context | naturalness | -0.0027 | -0.0031 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.0493 | 0.4332 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0268 | 19.3884 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | -0.0083 | -0.1188 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 0.0120 | 0.0423 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | -0.0046 | -0.0158 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0769 | 4.9512 |
| proposed_vs_candidate_no_context | context_overlap | 0.0185 | 0.5868 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0387 | -0.2063 |
| proposed_vs_candidate_no_context | persona_style | -0.0013 | -0.0024 |
| proposed_vs_candidate_no_context | distinct1 | 0.0102 | 0.0110 |
| proposed_vs_candidate_no_context | length_score | -0.0194 | -0.0347 |
| proposed_vs_candidate_no_context | sentence_score | -0.0292 | -0.0296 |
| proposed_vs_candidate_no_context | overall_quality | 0.0153 | 0.0559 |
| controlled_vs_proposed_raw | context_relevance | 0.0598 | 0.7504 |
| controlled_vs_proposed_raw | persona_consistency | 0.0106 | 0.0464 |
| controlled_vs_proposed_raw | naturalness | 0.0007 | 0.0008 |
| controlled_vs_proposed_raw | quest_state_correctness | 0.0546 | 0.3350 |
| controlled_vs_proposed_raw | lore_consistency | 0.0468 | 1.6638 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0180 | 0.2910 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0216 | 0.0735 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 0.0120 | 0.0416 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.0788 | 0.8525 |
| controlled_vs_proposed_raw | context_overlap | 0.0156 | 0.3110 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.0139 | 0.0933 |
| controlled_vs_proposed_raw | persona_style | -0.0026 | -0.0048 |
| controlled_vs_proposed_raw | distinct1 | -0.0090 | -0.0097 |
| controlled_vs_proposed_raw | length_score | 0.0069 | 0.0128 |
| controlled_vs_proposed_raw | sentence_score | 0.0292 | 0.0305 |
| controlled_vs_proposed_raw | overall_quality | 0.0320 | 0.1107 |
| controlled_vs_candidate_no_context | context_relevance | 0.1192 | 5.8607 |
| controlled_vs_candidate_no_context | persona_consistency | -0.0206 | -0.0795 |
| controlled_vs_candidate_no_context | naturalness | -0.0021 | -0.0023 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.1039 | 0.9132 |
| controlled_vs_candidate_no_context | lore_consistency | 0.0736 | 53.3110 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | 0.0097 | 0.1376 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0336 | 0.1189 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 0.0074 | 0.0251 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.1557 | 10.0244 |
| controlled_vs_candidate_no_context | context_overlap | 0.0341 | 1.0803 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | -0.0248 | -0.1323 |
| controlled_vs_candidate_no_context | persona_style | -0.0039 | -0.0071 |
| controlled_vs_candidate_no_context | distinct1 | 0.0011 | 0.0012 |
| controlled_vs_candidate_no_context | length_score | -0.0125 | -0.0223 |
| controlled_vs_candidate_no_context | sentence_score | 0.0000 | 0.0000 |
| controlled_vs_candidate_no_context | overall_quality | 0.0473 | 0.1728 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0695 | -0.4979 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0101 | 0.0423 |
| controlled_alt_vs_controlled_default | naturalness | -0.0030 | -0.0034 |
| controlled_alt_vs_controlled_default | quest_state_correctness | -0.0585 | -0.2686 |
| controlled_alt_vs_controlled_default | lore_consistency | -0.0447 | -0.5957 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_controlled_default | objective_completion_support | -0.0064 | -0.0801 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | -0.0195 | -0.0618 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | -0.0032 | -0.0107 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0913 | -0.5332 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0186 | -0.2830 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0159 | 0.0976 |
| controlled_alt_vs_controlled_default | persona_style | -0.0130 | -0.0240 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0022 | 0.0023 |
| controlled_alt_vs_controlled_default | length_score | -0.0194 | -0.0354 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0000 | 0.0000 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0289 | -0.0902 |
| controlled_alt_vs_proposed_raw | context_relevance | -0.0097 | -0.1211 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.0207 | 0.0907 |
| controlled_alt_vs_proposed_raw | naturalness | -0.0023 | -0.0027 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | -0.0038 | -0.0236 |
| controlled_alt_vs_proposed_raw | lore_consistency | 0.0022 | 0.0769 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_proposed_raw | objective_completion_support | 0.0116 | 0.1876 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 0.0021 | 0.0072 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | 0.0088 | 0.0305 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | -0.0125 | -0.1352 |
| controlled_alt_vs_proposed_raw | context_overlap | -0.0030 | -0.0600 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.0298 | 0.2000 |
| controlled_alt_vs_proposed_raw | persona_style | -0.0156 | -0.0286 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0069 | -0.0074 |
| controlled_alt_vs_proposed_raw | length_score | -0.0125 | -0.0231 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0292 | 0.0305 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.0030 | 0.0106 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.0497 | 2.4450 |
| controlled_alt_vs_candidate_no_context | persona_consistency | -0.0105 | -0.0406 |
| controlled_alt_vs_candidate_no_context | naturalness | -0.0051 | -0.0058 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 0.0454 | 0.3994 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 0.0289 | 20.9568 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_candidate_no_context | objective_completion_support | 0.0033 | 0.0465 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 0.0141 | 0.0498 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | 0.0042 | 0.0142 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.0644 | 4.1463 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0155 | 0.4916 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | -0.0089 | -0.0476 |
| controlled_alt_vs_candidate_no_context | persona_style | -0.0169 | -0.0310 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0033 | 0.0035 |
| controlled_alt_vs_candidate_no_context | length_score | -0.0319 | -0.0569 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.0000 | 0.0000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.0183 | 0.0670 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0594 | (0.0233, 0.0999) | 0.0000 | 0.0594 | (0.0149, 0.1046) | 0.0117 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0312 | (-0.0768, 0.0138) | 0.9163 | -0.0312 | (-0.0743, -0.0002) | 1.0000 |
| proposed_vs_candidate_no_context | naturalness | -0.0027 | (-0.0205, 0.0173) | 0.6237 | -0.0027 | (-0.0246, 0.0269) | 0.6033 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.0493 | (0.0194, 0.0812) | 0.0000 | 0.0493 | (0.0125, 0.0876) | 0.0077 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0268 | (0.0096, 0.0459) | 0.0000 | 0.0268 | (0.0053, 0.0483) | 0.0117 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | -0.0083 | (-0.0303, 0.0131) | 0.7690 | -0.0083 | (-0.0275, 0.0064) | 0.8127 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 0.0120 | (-0.0052, 0.0306) | 0.0853 | 0.0120 | (0.0008, 0.0257) | 0.0243 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | -0.0046 | (-0.0189, 0.0111) | 0.7403 | -0.0046 | (-0.0213, 0.0117) | 0.7323 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0769 | (0.0292, 0.1322) | 0.0000 | 0.0769 | (0.0186, 0.1352) | 0.0080 |
| proposed_vs_candidate_no_context | context_overlap | 0.0185 | (0.0057, 0.0333) | 0.0003 | 0.0185 | (0.0009, 0.0383) | 0.0167 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0387 | (-0.1022, 0.0238) | 0.9043 | -0.0387 | (-0.0929, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | persona_style | -0.0013 | (-0.0352, 0.0312) | 0.5627 | -0.0013 | (-0.0036, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | distinct1 | 0.0102 | (-0.0033, 0.0245) | 0.0720 | 0.0102 | (-0.0081, 0.0313) | 0.1697 |
| proposed_vs_candidate_no_context | length_score | -0.0194 | (-0.1069, 0.0681) | 0.6613 | -0.0194 | (-0.1167, 0.0950) | 0.6310 |
| proposed_vs_candidate_no_context | sentence_score | -0.0292 | (-0.0729, 0.0000) | 1.0000 | -0.0292 | (-0.0636, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | overall_quality | 0.0153 | (-0.0117, 0.0439) | 0.1217 | 0.0153 | (-0.0125, 0.0430) | 0.1770 |
| controlled_vs_proposed_raw | context_relevance | 0.0598 | (0.0101, 0.1166) | 0.0087 | 0.0598 | (0.0082, 0.1105) | 0.0003 |
| controlled_vs_proposed_raw | persona_consistency | 0.0106 | (-0.0595, 0.0812) | 0.3910 | 0.0106 | (-0.0975, 0.1206) | 0.4107 |
| controlled_vs_proposed_raw | naturalness | 0.0007 | (-0.0187, 0.0197) | 0.4873 | 0.0007 | (-0.0246, 0.0220) | 0.4740 |
| controlled_vs_proposed_raw | quest_state_correctness | 0.0546 | (0.0114, 0.1045) | 0.0050 | 0.0546 | (0.0091, 0.1004) | 0.0070 |
| controlled_vs_proposed_raw | lore_consistency | 0.0468 | (0.0092, 0.0883) | 0.0047 | 0.0468 | (0.0069, 0.0868) | 0.0083 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0180 | (-0.0001, 0.0388) | 0.0270 | 0.0180 | (0.0024, 0.0398) | 0.0093 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0216 | (0.0023, 0.0436) | 0.0137 | 0.0216 | (0.0029, 0.0403) | 0.0103 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 0.0120 | (-0.0014, 0.0273) | 0.0427 | 0.0120 | (-0.0005, 0.0250) | 0.0280 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.0788 | (0.0114, 0.1527) | 0.0100 | 0.0788 | (0.0124, 0.1462) | 0.0093 |
| controlled_vs_proposed_raw | context_overlap | 0.0156 | (-0.0025, 0.0358) | 0.0467 | 0.0156 | (-0.0007, 0.0384) | 0.0390 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.0139 | (-0.0823, 0.1071) | 0.3853 | 0.0139 | (-0.1212, 0.1500) | 0.4047 |
| controlled_vs_proposed_raw | persona_style | -0.0026 | (-0.0456, 0.0391) | 0.5663 | -0.0026 | (-0.0072, 0.0000) | 1.0000 |
| controlled_vs_proposed_raw | distinct1 | -0.0090 | (-0.0284, 0.0069) | 0.8620 | -0.0090 | (-0.0304, 0.0086) | 0.8123 |
| controlled_vs_proposed_raw | length_score | 0.0069 | (-0.0681, 0.0944) | 0.4367 | 0.0069 | (-0.0758, 0.0769) | 0.4463 |
| controlled_vs_proposed_raw | sentence_score | 0.0292 | (0.0000, 0.0729) | 0.1167 | 0.0292 | (0.0000, 0.0955) | 0.3277 |
| controlled_vs_proposed_raw | overall_quality | 0.0320 | (-0.0099, 0.0769) | 0.0723 | 0.0320 | (-0.0248, 0.0825) | 0.0960 |
| controlled_vs_candidate_no_context | context_relevance | 0.1192 | (0.0594, 0.1849) | 0.0000 | 0.1192 | (0.0280, 0.2105) | 0.0123 |
| controlled_vs_candidate_no_context | persona_consistency | -0.0206 | (-0.0885, 0.0455) | 0.7217 | -0.0206 | (-0.1555, 0.1022) | 0.6387 |
| controlled_vs_candidate_no_context | naturalness | -0.0021 | (-0.0159, 0.0116) | 0.6103 | -0.0021 | (-0.0136, 0.0087) | 0.6417 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.1039 | (0.0523, 0.1634) | 0.0000 | 0.1039 | (0.0263, 0.1815) | 0.0077 |
| controlled_vs_candidate_no_context | lore_consistency | 0.0736 | (0.0308, 0.1209) | 0.0000 | 0.0736 | (0.0121, 0.1351) | 0.0103 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | 0.0097 | (-0.0132, 0.0317) | 0.2153 | 0.0097 | (-0.0111, 0.0380) | 0.2633 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0336 | (0.0159, 0.0530) | 0.0000 | 0.0336 | (0.0113, 0.0536) | 0.0020 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 0.0074 | (-0.0122, 0.0267) | 0.2287 | 0.0074 | (-0.0188, 0.0351) | 0.3020 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.1557 | (0.0795, 0.2402) | 0.0000 | 0.1557 | (0.0399, 0.2735) | 0.0070 |
| controlled_vs_candidate_no_context | context_overlap | 0.0341 | (0.0166, 0.0548) | 0.0000 | 0.0341 | (0.0048, 0.0634) | 0.0077 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | -0.0248 | (-0.1121, 0.0585) | 0.7000 | -0.0248 | (-0.1855, 0.1286) | 0.6340 |
| controlled_vs_candidate_no_context | persona_style | -0.0039 | (-0.0443, 0.0365) | 0.5780 | -0.0039 | (-0.0108, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | distinct1 | 0.0011 | (-0.0202, 0.0201) | 0.4417 | 0.0011 | (-0.0256, 0.0185) | 0.4220 |
| controlled_vs_candidate_no_context | length_score | -0.0125 | (-0.0764, 0.0486) | 0.6520 | -0.0125 | (-0.0515, 0.0317) | 0.7513 |
| controlled_vs_candidate_no_context | sentence_score | 0.0000 | (-0.0437, 0.0437) | 0.6443 | 0.0000 | (-0.0375, 0.0477) | 0.6477 |
| controlled_vs_candidate_no_context | overall_quality | 0.0473 | (0.0022, 0.0917) | 0.0200 | 0.0473 | (-0.0330, 0.1170) | 0.1077 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0695 | (-0.1430, -0.0076) | 0.9877 | -0.0695 | (-0.1575, 0.0090) | 0.9193 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0101 | (-0.0426, 0.0701) | 0.3683 | 0.0101 | (-0.0364, 0.0796) | 0.4050 |
| controlled_alt_vs_controlled_default | naturalness | -0.0030 | (-0.0203, 0.0151) | 0.6467 | -0.0030 | (-0.0104, 0.0056) | 0.7487 |
| controlled_alt_vs_controlled_default | quest_state_correctness | -0.0585 | (-0.1192, -0.0045) | 0.9870 | -0.0585 | (-0.1372, 0.0051) | 0.9310 |
| controlled_alt_vs_controlled_default | lore_consistency | -0.0447 | (-0.0965, 0.0058) | 0.9600 | -0.0447 | (-0.1031, -0.0025) | 1.0000 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | objective_completion_support | -0.0064 | (-0.0269, 0.0129) | 0.7307 | -0.0064 | (-0.0244, 0.0152) | 0.7370 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | -0.0195 | (-0.0432, 0.0024) | 0.9580 | -0.0195 | (-0.0468, 0.0001) | 0.9703 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | -0.0032 | (-0.0194, 0.0123) | 0.6463 | -0.0032 | (-0.0178, 0.0153) | 0.6490 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0913 | (-0.1856, -0.0114) | 0.9880 | -0.0913 | (-0.2144, 0.0083) | 0.9303 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0186 | (-0.0426, 0.0015) | 0.9643 | -0.0186 | (-0.0514, 0.0107) | 0.8457 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0159 | (-0.0486, 0.0863) | 0.3223 | 0.0159 | (-0.0417, 0.0905) | 0.3440 |
| controlled_alt_vs_controlled_default | persona_style | -0.0130 | (-0.0469, 0.0208) | 0.7747 | -0.0130 | (-0.0603, 0.0341) | 0.7293 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0022 | (-0.0141, 0.0220) | 0.4323 | 0.0022 | (-0.0146, 0.0287) | 0.4500 |
| controlled_alt_vs_controlled_default | length_score | -0.0194 | (-0.1014, 0.0583) | 0.6813 | -0.0194 | (-0.0487, 0.0061) | 0.8733 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0000 | (-0.0437, 0.0437) | 0.6580 | 0.0000 | (-0.0477, 0.0375) | 0.6290 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0289 | (-0.0741, 0.0109) | 0.9160 | -0.0289 | (-0.0821, 0.0253) | 0.8200 |
| controlled_alt_vs_proposed_raw | context_relevance | -0.0097 | (-0.0463, 0.0289) | 0.6797 | -0.0097 | (-0.0554, 0.0368) | 0.6433 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.0207 | (-0.0312, 0.0794) | 0.2200 | 0.0207 | (-0.0331, 0.1034) | 0.3010 |
| controlled_alt_vs_proposed_raw | naturalness | -0.0023 | (-0.0276, 0.0198) | 0.5633 | -0.0023 | (-0.0253, 0.0141) | 0.5973 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | -0.0038 | (-0.0361, 0.0313) | 0.5997 | -0.0038 | (-0.0400, 0.0369) | 0.5760 |
| controlled_alt_vs_proposed_raw | lore_consistency | 0.0022 | (-0.0226, 0.0293) | 0.4610 | 0.0022 | (-0.0166, 0.0240) | 0.3933 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_proposed_raw | objective_completion_support | 0.0116 | (-0.0049, 0.0318) | 0.1060 | 0.0116 | (-0.0083, 0.0363) | 0.1490 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 0.0021 | (-0.0145, 0.0186) | 0.4043 | 0.0021 | (-0.0114, 0.0197) | 0.3457 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | 0.0088 | (-0.0030, 0.0228) | 0.0777 | 0.0088 | (-0.0020, 0.0253) | 0.0703 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | -0.0125 | (-0.0652, 0.0379) | 0.6797 | -0.0125 | (-0.0721, 0.0496) | 0.5740 |
| controlled_alt_vs_proposed_raw | context_overlap | -0.0030 | (-0.0163, 0.0101) | 0.6713 | -0.0030 | (-0.0209, 0.0171) | 0.6053 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.0298 | (-0.0338, 0.0982) | 0.1867 | 0.0298 | (-0.0357, 0.1439) | 0.2777 |
| controlled_alt_vs_proposed_raw | persona_style | -0.0156 | (-0.0690, 0.0469) | 0.7200 | -0.0156 | (-0.0670, 0.0341) | 0.7393 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0069 | (-0.0241, 0.0081) | 0.8007 | -0.0069 | (-0.0263, 0.0095) | 0.7747 |
| controlled_alt_vs_proposed_raw | length_score | -0.0125 | (-0.1139, 0.0861) | 0.5857 | -0.0125 | (-0.0970, 0.0577) | 0.6160 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0292 | (-0.0292, 0.0875) | 0.2277 | 0.0292 | (-0.0350, 0.1022) | 0.2597 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.0030 | (-0.0260, 0.0305) | 0.4377 | 0.0030 | (-0.0309, 0.0519) | 0.4353 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.0497 | (0.0135, 0.0916) | 0.0003 | 0.0497 | (0.0047, 0.1228) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | -0.0105 | (-0.0678, 0.0573) | 0.6377 | -0.0105 | (-0.0837, 0.0890) | 0.6063 |
| controlled_alt_vs_candidate_no_context | naturalness | -0.0051 | (-0.0243, 0.0128) | 0.6927 | -0.0051 | (-0.0137, 0.0037) | 0.8547 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 0.0454 | (0.0117, 0.0820) | 0.0030 | 0.0454 | (0.0033, 0.1129) | 0.0000 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 0.0289 | (0.0083, 0.0575) | 0.0000 | 0.0289 | (0.0030, 0.0662) | 0.0097 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_candidate_no_context | objective_completion_support | 0.0033 | (-0.0162, 0.0249) | 0.3737 | 0.0033 | (-0.0096, 0.0214) | 0.3397 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 0.0141 | (-0.0027, 0.0311) | 0.0523 | 0.0141 | (0.0008, 0.0326) | 0.0110 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | 0.0042 | (-0.0099, 0.0202) | 0.3007 | 0.0042 | (-0.0110, 0.0245) | 0.3463 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.0644 | (0.0152, 0.1212) | 0.0020 | 0.0644 | (0.0038, 0.1612) | 0.0087 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0155 | (0.0042, 0.0265) | 0.0020 | 0.0155 | (0.0035, 0.0332) | 0.0003 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | -0.0089 | (-0.0814, 0.0715) | 0.5947 | -0.0089 | (-0.1032, 0.1167) | 0.6150 |
| controlled_alt_vs_candidate_no_context | persona_style | -0.0169 | (-0.0638, 0.0404) | 0.7737 | -0.0169 | (-0.0703, 0.0341) | 0.7407 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0033 | (-0.0147, 0.0213) | 0.3770 | 0.0033 | (-0.0007, 0.0080) | 0.0833 |
| controlled_alt_vs_candidate_no_context | length_score | -0.0319 | (-0.1042, 0.0389) | 0.8123 | -0.0319 | (-0.0722, 0.0106) | 0.9310 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.0000 | (-0.0437, 0.0437) | 0.6557 | 0.0000 | (-0.0437, 0.0437) | 0.6350 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.0183 | (-0.0144, 0.0531) | 0.1413 | 0.0183 | (-0.0258, 0.0922) | 0.2877 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 9 | 4 | 11 | 0.6042 | 0.6923 |
| proposed_vs_candidate_no_context | persona_consistency | 3 | 6 | 15 | 0.4375 | 0.3333 |
| proposed_vs_candidate_no_context | naturalness | 6 | 8 | 10 | 0.4583 | 0.4286 |
| proposed_vs_candidate_no_context | quest_state_correctness | 10 | 3 | 11 | 0.6458 | 0.7692 |
| proposed_vs_candidate_no_context | lore_consistency | 9 | 0 | 15 | 0.6875 | 1.0000 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| proposed_vs_candidate_no_context | objective_completion_support | 5 | 5 | 14 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 8 | 6 | 10 | 0.5417 | 0.5714 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | 3 | 8 | 13 | 0.3958 | 0.2727 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 8 | 1 | 15 | 0.6458 | 0.8889 |
| proposed_vs_candidate_no_context | context_overlap | 10 | 3 | 11 | 0.6458 | 0.7692 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 2 | 5 | 17 | 0.4375 | 0.2857 |
| proposed_vs_candidate_no_context | persona_style | 2 | 2 | 20 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | distinct1 | 8 | 5 | 11 | 0.5625 | 0.6154 |
| proposed_vs_candidate_no_context | length_score | 5 | 9 | 10 | 0.4167 | 0.3571 |
| proposed_vs_candidate_no_context | sentence_score | 0 | 2 | 22 | 0.4583 | 0.0000 |
| proposed_vs_candidate_no_context | overall_quality | 10 | 4 | 10 | 0.6250 | 0.7143 |
| controlled_vs_proposed_raw | context_relevance | 12 | 6 | 6 | 0.6250 | 0.6667 |
| controlled_vs_proposed_raw | persona_consistency | 7 | 8 | 9 | 0.4792 | 0.4667 |
| controlled_vs_proposed_raw | naturalness | 7 | 11 | 6 | 0.4167 | 0.3889 |
| controlled_vs_proposed_raw | quest_state_correctness | 12 | 6 | 6 | 0.6250 | 0.6667 |
| controlled_vs_proposed_raw | lore_consistency | 8 | 2 | 14 | 0.6250 | 0.8000 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_vs_proposed_raw | objective_completion_support | 5 | 3 | 16 | 0.5417 | 0.6250 |
| controlled_vs_proposed_raw | gameplay_usefulness | 9 | 9 | 6 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 7 | 5 | 12 | 0.5417 | 0.5833 |
| controlled_vs_proposed_raw | context_keyword_coverage | 7 | 2 | 15 | 0.6042 | 0.7778 |
| controlled_vs_proposed_raw | context_overlap | 12 | 6 | 6 | 0.6250 | 0.6667 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 7 | 6 | 11 | 0.5208 | 0.5385 |
| controlled_vs_proposed_raw | persona_style | 3 | 4 | 17 | 0.4792 | 0.4286 |
| controlled_vs_proposed_raw | distinct1 | 9 | 7 | 8 | 0.5417 | 0.5625 |
| controlled_vs_proposed_raw | length_score | 7 | 11 | 6 | 0.4167 | 0.3889 |
| controlled_vs_proposed_raw | sentence_score | 2 | 0 | 22 | 0.5417 | 1.0000 |
| controlled_vs_proposed_raw | overall_quality | 10 | 8 | 6 | 0.5417 | 0.5556 |
| controlled_vs_candidate_no_context | context_relevance | 13 | 5 | 6 | 0.6667 | 0.7222 |
| controlled_vs_candidate_no_context | persona_consistency | 6 | 7 | 11 | 0.4792 | 0.4615 |
| controlled_vs_candidate_no_context | naturalness | 8 | 10 | 6 | 0.4583 | 0.4444 |
| controlled_vs_candidate_no_context | quest_state_correctness | 13 | 5 | 6 | 0.6667 | 0.7222 |
| controlled_vs_candidate_no_context | lore_consistency | 10 | 0 | 14 | 0.7083 | 1.0000 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_vs_candidate_no_context | objective_completion_support | 6 | 4 | 14 | 0.5417 | 0.6000 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 14 | 4 | 6 | 0.7083 | 0.7778 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 7 | 6 | 11 | 0.5208 | 0.5385 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 10 | 0 | 14 | 0.7083 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 13 | 5 | 6 | 0.6667 | 0.7222 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 6 | 6 | 12 | 0.5000 | 0.5000 |
| controlled_vs_candidate_no_context | persona_style | 3 | 3 | 18 | 0.5000 | 0.5000 |
| controlled_vs_candidate_no_context | distinct1 | 12 | 6 | 6 | 0.6250 | 0.6667 |
| controlled_vs_candidate_no_context | length_score | 9 | 9 | 6 | 0.5000 | 0.5000 |
| controlled_vs_candidate_no_context | sentence_score | 1 | 1 | 22 | 0.5000 | 0.5000 |
| controlled_vs_candidate_no_context | overall_quality | 13 | 5 | 6 | 0.6667 | 0.7222 |
| controlled_alt_vs_controlled_default | context_relevance | 4 | 9 | 11 | 0.3958 | 0.3077 |
| controlled_alt_vs_controlled_default | persona_consistency | 3 | 7 | 14 | 0.4167 | 0.3000 |
| controlled_alt_vs_controlled_default | naturalness | 6 | 7 | 11 | 0.4792 | 0.4615 |
| controlled_alt_vs_controlled_default | quest_state_correctness | 5 | 8 | 11 | 0.4375 | 0.3846 |
| controlled_alt_vs_controlled_default | lore_consistency | 3 | 8 | 13 | 0.3958 | 0.2727 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_alt_vs_controlled_default | objective_completion_support | 4 | 4 | 16 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | 5 | 8 | 11 | 0.4375 | 0.3846 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | 7 | 5 | 12 | 0.5417 | 0.5833 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 2 | 6 | 16 | 0.4167 | 0.2500 |
| controlled_alt_vs_controlled_default | context_overlap | 5 | 8 | 11 | 0.4375 | 0.3846 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 3 | 4 | 17 | 0.4792 | 0.4286 |
| controlled_alt_vs_controlled_default | persona_style | 1 | 3 | 20 | 0.4583 | 0.2500 |
| controlled_alt_vs_controlled_default | distinct1 | 6 | 6 | 12 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | length_score | 6 | 7 | 11 | 0.4792 | 0.4615 |
| controlled_alt_vs_controlled_default | sentence_score | 1 | 1 | 22 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | overall_quality | 4 | 9 | 11 | 0.3958 | 0.3077 |
| controlled_alt_vs_proposed_raw | context_relevance | 9 | 9 | 6 | 0.5000 | 0.5000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 8 | 9 | 7 | 0.4792 | 0.4706 |
| controlled_alt_vs_proposed_raw | naturalness | 7 | 11 | 6 | 0.4167 | 0.3889 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 8 | 10 | 6 | 0.4583 | 0.4444 |
| controlled_alt_vs_proposed_raw | lore_consistency | 5 | 5 | 14 | 0.5000 | 0.5000 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_alt_vs_proposed_raw | objective_completion_support | 5 | 4 | 15 | 0.5208 | 0.5556 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 7 | 11 | 6 | 0.4167 | 0.3889 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | 5 | 4 | 15 | 0.5208 | 0.5556 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 4 | 5 | 15 | 0.4792 | 0.4444 |
| controlled_alt_vs_proposed_raw | context_overlap | 9 | 9 | 6 | 0.5000 | 0.5000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 8 | 7 | 9 | 0.5208 | 0.5333 |
| controlled_alt_vs_proposed_raw | persona_style | 2 | 7 | 15 | 0.3958 | 0.2222 |
| controlled_alt_vs_proposed_raw | distinct1 | 9 | 9 | 6 | 0.5000 | 0.5000 |
| controlled_alt_vs_proposed_raw | length_score | 9 | 9 | 6 | 0.5000 | 0.5000 |
| controlled_alt_vs_proposed_raw | sentence_score | 3 | 1 | 20 | 0.5417 | 0.7500 |
| controlled_alt_vs_proposed_raw | overall_quality | 8 | 10 | 6 | 0.4583 | 0.4444 |
| controlled_alt_vs_candidate_no_context | context_relevance | 13 | 4 | 7 | 0.6875 | 0.7647 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 5 | 10 | 9 | 0.3958 | 0.3333 |
| controlled_alt_vs_candidate_no_context | naturalness | 7 | 10 | 7 | 0.4375 | 0.4118 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 12 | 5 | 7 | 0.6458 | 0.7059 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 7 | 0 | 17 | 0.6458 | 1.0000 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_alt_vs_candidate_no_context | objective_completion_support | 5 | 4 | 15 | 0.5208 | 0.5556 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 10 | 7 | 7 | 0.5625 | 0.5882 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | 4 | 5 | 15 | 0.4792 | 0.4444 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 6 | 1 | 17 | 0.6042 | 0.8571 |
| controlled_alt_vs_candidate_no_context | context_overlap | 13 | 4 | 7 | 0.6875 | 0.7647 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 5 | 8 | 11 | 0.4375 | 0.3846 |
| controlled_alt_vs_candidate_no_context | persona_style | 1 | 6 | 17 | 0.3958 | 0.1429 |
| controlled_alt_vs_candidate_no_context | distinct1 | 10 | 7 | 7 | 0.5625 | 0.5882 |
| controlled_alt_vs_candidate_no_context | length_score | 7 | 10 | 7 | 0.4375 | 0.4118 |
| controlled_alt_vs_candidate_no_context | sentence_score | 1 | 1 | 22 | 0.5000 | 0.5000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 8 | 9 | 7 | 0.4792 | 0.4706 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.0833 | 0.1250 | 0.8750 |
| proposed_contextual_controlled_tuned | 0.0000 | 0.0000 | 0.0833 | 0.2917 | 0.7083 |
| proposed_contextual | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.5417 | 0.0000 | 0.0000 |

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
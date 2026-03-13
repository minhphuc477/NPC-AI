# Proposal Alignment Evaluation Report

- Run ID: `20260313T070226Z`
- Generated: `2026-03-13T07:08:40.706431+00:00`
- Scenarios: `artifacts\proposal_control_tuning\auto_tune\20260313T051946Z\train_runs\trial_009\seed_19\20260313T070226Z\scenarios.jsonl`
- Scenario count: `24`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_tuned`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.1313 (0.0740, 0.1888) | 0.2710 (0.2080, 0.3415) | 0.8551 (0.8388, 0.8714) | 0.3246 (0.2814, 0.3741) | n/a |
| proposed_contextual_controlled_tuned | 0.0669 (0.0301, 0.1091) | 0.2456 (0.1920, 0.3078) | 0.8669 (0.8530, 0.8812) | 0.2882 (0.2546, 0.3274) | n/a |
| proposed_contextual | 0.0753 (0.0376, 0.1255) | 0.2377 (0.1805, 0.2985) | 0.8678 (0.8535, 0.8843) | 0.2887 (0.2540, 0.3295) | n/a |
| candidate_no_context | 0.0238 (0.0137, 0.0352) | 0.2435 (0.1840, 0.3121) | 0.8870 (0.8698, 0.9072) | 0.2704 (0.2456, 0.2985) | n/a |

## Game-facing Outcome Metrics (mean, 95% CI)
| Arm | Quest-state Correctness | Lore Consistency | Contradiction Safety | Objective Completion Support | Gameplay Usefulness | Time-pressure Acceptability |
|---|---:|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2117 (0.1626, 0.2648) | 0.0597 (0.0280, 0.0946) | 1.0000 (1.0000, 1.0000) | 0.0874 (0.0638, 0.1104) | 0.3085 (0.2961, 0.3210) | 0.3012 (0.2839, 0.3179) |
| proposed_contextual_controlled_tuned | 0.1535 (0.1231, 0.1923) | 0.0181 (0.0047, 0.0340) | 1.0000 (1.0000, 1.0000) | 0.0803 (0.0548, 0.1040) | 0.2948 (0.2831, 0.3071) | 0.2948 (0.2764, 0.3130) |
| proposed_contextual | 0.1624 (0.1284, 0.2045) | 0.0232 (0.0053, 0.0459) | 1.0000 (1.0000, 1.0000) | 0.0776 (0.0549, 0.1003) | 0.2948 (0.2804, 0.3082) | 0.2948 (0.2767, 0.3130) |
| candidate_no_context | 0.1186 (0.1107, 0.1279) | 0.0018 (0.0000, 0.0044) | 1.0000 (1.0000, 1.0000) | 0.0863 (0.0621, 0.1082) | 0.2906 (0.2822, 0.2989) | 0.3003 (0.2872, 0.3128) |

- Multi-turn contradiction rate is reported as `1 - contradiction_safety` in row-level outputs.

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0515 | 2.1628 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0057 | -0.0235 |
| proposed_vs_candidate_no_context | naturalness | -0.0192 | -0.0216 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.0438 | 0.3693 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0215 | 12.1871 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | -0.0086 | -0.0998 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 0.0041 | 0.0143 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | -0.0056 | -0.0185 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0663 | 3.4314 |
| proposed_vs_candidate_no_context | context_overlap | 0.0169 | 0.4942 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0010 | 0.0061 |
| proposed_vs_candidate_no_context | persona_style | -0.0326 | -0.0579 |
| proposed_vs_candidate_no_context | distinct1 | -0.0098 | -0.0105 |
| proposed_vs_candidate_no_context | length_score | -0.0764 | -0.1300 |
| proposed_vs_candidate_no_context | sentence_score | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | overall_quality | 0.0183 | 0.0678 |
| controlled_vs_proposed_raw | context_relevance | 0.0560 | 0.7436 |
| controlled_vs_proposed_raw | persona_consistency | 0.0332 | 0.1398 |
| controlled_vs_proposed_raw | naturalness | -0.0128 | -0.0147 |
| controlled_vs_proposed_raw | quest_state_correctness | 0.0492 | 0.3031 |
| controlled_vs_proposed_raw | lore_consistency | 0.0365 | 1.5701 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0097 | 0.1252 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0138 | 0.0467 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 0.0064 | 0.0218 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.0746 | 0.8717 |
| controlled_vs_proposed_raw | context_overlap | 0.0125 | 0.2439 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.0383 | 0.2325 |
| controlled_vs_proposed_raw | persona_style | 0.0130 | 0.0246 |
| controlled_vs_proposed_raw | distinct1 | -0.0036 | -0.0039 |
| controlled_vs_proposed_raw | length_score | -0.0347 | -0.0679 |
| controlled_vs_proposed_raw | sentence_score | -0.0437 | -0.0451 |
| controlled_vs_proposed_raw | overall_quality | 0.0358 | 0.1241 |
| controlled_vs_candidate_no_context | context_relevance | 0.1075 | 4.5146 |
| controlled_vs_candidate_no_context | persona_consistency | 0.0275 | 0.1131 |
| controlled_vs_candidate_no_context | naturalness | -0.0320 | -0.0360 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.0930 | 0.7843 |
| controlled_vs_candidate_no_context | lore_consistency | 0.0580 | 32.8918 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | 0.0011 | 0.0129 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0179 | 0.0616 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 0.0009 | 0.0029 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.1409 | 7.2941 |
| controlled_vs_candidate_no_context | context_overlap | 0.0294 | 0.8586 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.0393 | 0.2400 |
| controlled_vs_candidate_no_context | persona_style | -0.0195 | -0.0347 |
| controlled_vs_candidate_no_context | distinct1 | -0.0134 | -0.0144 |
| controlled_vs_candidate_no_context | length_score | -0.1111 | -0.1891 |
| controlled_vs_candidate_no_context | sentence_score | -0.0437 | -0.0451 |
| controlled_vs_candidate_no_context | overall_quality | 0.0542 | 0.2004 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0644 | -0.4906 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0253 | -0.0936 |
| controlled_alt_vs_controlled_default | naturalness | 0.0119 | 0.0139 |
| controlled_alt_vs_controlled_default | quest_state_correctness | -0.0582 | -0.2748 |
| controlled_alt_vs_controlled_default | lore_consistency | -0.0416 | -0.6970 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_controlled_default | objective_completion_support | -0.0071 | -0.0811 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | -0.0137 | -0.0445 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | -0.0064 | -0.0211 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0841 | -0.5248 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0184 | -0.2894 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0196 | -0.0968 |
| controlled_alt_vs_controlled_default | persona_style | -0.0482 | -0.0887 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0047 | 0.0051 |
| controlled_alt_vs_controlled_default | length_score | 0.0208 | 0.0437 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0583 | 0.0629 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0363 | -0.1119 |
| controlled_alt_vs_proposed_raw | context_relevance | -0.0084 | -0.1117 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.0079 | 0.0332 |
| controlled_alt_vs_proposed_raw | naturalness | -0.0009 | -0.0010 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | -0.0089 | -0.0550 |
| controlled_alt_vs_proposed_raw | lore_consistency | -0.0051 | -0.2213 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_proposed_raw | objective_completion_support | 0.0026 | 0.0340 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 0.0000 | 0.0001 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | 0.0001 | 0.0002 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | -0.0095 | -0.1106 |
| controlled_alt_vs_proposed_raw | context_overlap | -0.0059 | -0.1161 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.0187 | 0.1133 |
| controlled_alt_vs_proposed_raw | persona_style | -0.0352 | -0.0663 |
| controlled_alt_vs_proposed_raw | distinct1 | 0.0010 | 0.0011 |
| controlled_alt_vs_proposed_raw | length_score | -0.0139 | -0.0272 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0146 | 0.0150 |
| controlled_alt_vs_proposed_raw | overall_quality | -0.0005 | -0.0017 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.0431 | 1.8094 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.0022 | 0.0089 |
| controlled_alt_vs_candidate_no_context | naturalness | -0.0201 | -0.0227 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 0.0349 | 0.2940 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 0.0163 | 9.2684 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_candidate_no_context | objective_completion_support | -0.0060 | -0.0692 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 0.0042 | 0.0144 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | -0.0055 | -0.0183 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.0568 | 2.9412 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0110 | 0.3207 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.0196 | 0.1200 |
| controlled_alt_vs_candidate_no_context | persona_style | -0.0677 | -0.1204 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0087 | -0.0094 |
| controlled_alt_vs_candidate_no_context | length_score | -0.0903 | -0.1537 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.0146 | 0.0150 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.0179 | 0.0661 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0515 | (0.0127, 0.1014) | 0.0020 | 0.0515 | (0.0210, 0.0839) | 0.0010 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0057 | (-0.0709, 0.0582) | 0.5620 | -0.0057 | (-0.1136, 0.0703) | 0.5373 |
| proposed_vs_candidate_no_context | naturalness | -0.0192 | (-0.0426, 0.0028) | 0.9593 | -0.0192 | (-0.0306, -0.0061) | 1.0000 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.0438 | (0.0110, 0.0857) | 0.0010 | 0.0438 | (0.0173, 0.0746) | 0.0003 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0215 | (0.0041, 0.0440) | 0.0010 | 0.0215 | (0.0026, 0.0414) | 0.0117 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | -0.0086 | (-0.0294, 0.0099) | 0.8050 | -0.0086 | (-0.0314, 0.0094) | 0.7977 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 0.0041 | (-0.0136, 0.0220) | 0.3260 | 0.0041 | (-0.0025, 0.0120) | 0.1157 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | -0.0056 | (-0.0238, 0.0110) | 0.7403 | -0.0056 | (-0.0249, 0.0087) | 0.7403 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0663 | (0.0136, 0.1333) | 0.0043 | 0.0663 | (0.0280, 0.1091) | 0.0000 |
| proposed_vs_candidate_no_context | context_overlap | 0.0169 | (0.0043, 0.0307) | 0.0017 | 0.0169 | (0.0051, 0.0337) | 0.0000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0010 | (-0.0853, 0.0844) | 0.4813 | 0.0010 | (-0.1429, 0.1111) | 0.4590 |
| proposed_vs_candidate_no_context | persona_style | -0.0326 | (-0.1016, 0.0195) | 0.8803 | -0.0326 | (-0.0901, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | distinct1 | -0.0098 | (-0.0255, 0.0042) | 0.9033 | -0.0098 | (-0.0208, -0.0016) | 0.9903 |
| proposed_vs_candidate_no_context | length_score | -0.0764 | (-0.1806, 0.0222) | 0.9313 | -0.0764 | (-0.1242, -0.0231) | 1.0000 |
| proposed_vs_candidate_no_context | sentence_score | 0.0000 | (-0.0583, 0.0583) | 0.5973 | 0.0000 | (-0.0375, 0.0477) | 0.6420 |
| proposed_vs_candidate_no_context | overall_quality | 0.0183 | (-0.0148, 0.0537) | 0.1443 | 0.0183 | (-0.0239, 0.0552) | 0.1783 |
| controlled_vs_proposed_raw | context_relevance | 0.0560 | (0.0031, 0.1118) | 0.0170 | 0.0560 | (0.0091, 0.0966) | 0.0160 |
| controlled_vs_proposed_raw | persona_consistency | 0.0332 | (-0.0368, 0.1128) | 0.2063 | 0.0332 | (-0.0621, 0.1089) | 0.1850 |
| controlled_vs_proposed_raw | naturalness | -0.0128 | (-0.0352, 0.0082) | 0.8830 | -0.0128 | (-0.0351, 0.0048) | 0.8637 |
| controlled_vs_proposed_raw | quest_state_correctness | 0.0492 | (-0.0001, 0.1022) | 0.0260 | 0.0492 | (0.0126, 0.0828) | 0.0063 |
| controlled_vs_proposed_raw | lore_consistency | 0.0365 | (0.0067, 0.0689) | 0.0083 | 0.0365 | (0.0187, 0.0552) | 0.0000 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0097 | (-0.0101, 0.0315) | 0.1820 | 0.0097 | (-0.0029, 0.0272) | 0.0800 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0138 | (-0.0014, 0.0302) | 0.0363 | 0.0138 | (0.0027, 0.0288) | 0.0087 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 0.0064 | (-0.0091, 0.0227) | 0.2180 | 0.0064 | (-0.0049, 0.0243) | 0.1980 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.0746 | (0.0045, 0.1485) | 0.0187 | 0.0746 | (0.0182, 0.1289) | 0.0087 |
| controlled_vs_proposed_raw | context_overlap | 0.0125 | (-0.0079, 0.0326) | 0.1123 | 0.0125 | (-0.0002, 0.0249) | 0.0363 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.0383 | (-0.0536, 0.1322) | 0.2173 | 0.0383 | (-0.0655, 0.1361) | 0.1823 |
| controlled_vs_proposed_raw | persona_style | 0.0130 | (-0.0378, 0.0664) | 0.3197 | 0.0130 | (-0.0341, 0.0603) | 0.3487 |
| controlled_vs_proposed_raw | distinct1 | -0.0036 | (-0.0236, 0.0161) | 0.6347 | -0.0036 | (-0.0168, 0.0080) | 0.7247 |
| controlled_vs_proposed_raw | length_score | -0.0347 | (-0.1167, 0.0444) | 0.8140 | -0.0347 | (-0.1179, 0.0347) | 0.7127 |
| controlled_vs_proposed_raw | sentence_score | -0.0437 | (-0.1167, 0.0292) | 0.9197 | -0.0437 | (-0.0875, 0.0000) | 1.0000 |
| controlled_vs_proposed_raw | overall_quality | 0.0358 | (-0.0079, 0.0834) | 0.0603 | 0.0358 | (-0.0180, 0.0805) | 0.0813 |
| controlled_vs_candidate_no_context | context_relevance | 0.1075 | (0.0531, 0.1662) | 0.0000 | 0.1075 | (0.0413, 0.1791) | 0.0003 |
| controlled_vs_candidate_no_context | persona_consistency | 0.0275 | (-0.0676, 0.1292) | 0.2900 | 0.0275 | (-0.1629, 0.1261) | 0.2733 |
| controlled_vs_candidate_no_context | naturalness | -0.0320 | (-0.0587, -0.0050) | 0.9927 | -0.0320 | (-0.0633, -0.0037) | 0.9953 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.0930 | (0.0438, 0.1473) | 0.0000 | 0.0930 | (0.0360, 0.1550) | 0.0000 |
| controlled_vs_candidate_no_context | lore_consistency | 0.0580 | (0.0265, 0.0931) | 0.0000 | 0.0580 | (0.0252, 0.0919) | 0.0007 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | 0.0011 | (-0.0224, 0.0239) | 0.4760 | 0.0011 | (-0.0169, 0.0252) | 0.4810 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0179 | (0.0049, 0.0317) | 0.0020 | 0.0179 | (0.0047, 0.0356) | 0.0003 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 0.0009 | (-0.0165, 0.0193) | 0.4670 | 0.0009 | (-0.0223, 0.0290) | 0.4693 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.1409 | (0.0682, 0.2178) | 0.0000 | 0.1409 | (0.0557, 0.2420) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0294 | (0.0097, 0.0518) | 0.0017 | 0.0294 | (0.0051, 0.0555) | 0.0007 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.0393 | (-0.0814, 0.1615) | 0.2650 | 0.0393 | (-0.1866, 0.1576) | 0.2527 |
| controlled_vs_candidate_no_context | persona_style | -0.0195 | (-0.0521, 0.0065) | 0.9247 | -0.0195 | (-0.0404, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | distinct1 | -0.0134 | (-0.0326, 0.0068) | 0.9070 | -0.0134 | (-0.0329, 0.0029) | 0.9327 |
| controlled_vs_candidate_no_context | length_score | -0.1111 | (-0.2292, -0.0014) | 0.9777 | -0.1111 | (-0.2385, 0.0028) | 0.9710 |
| controlled_vs_candidate_no_context | sentence_score | -0.0437 | (-0.1167, 0.0292) | 0.9120 | -0.0437 | (-0.1212, 0.0350) | 0.8737 |
| controlled_vs_candidate_no_context | overall_quality | 0.0542 | (0.0019, 0.1074) | 0.0207 | 0.0542 | (-0.0376, 0.1191) | 0.0810 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0644 | (-0.1211, -0.0124) | 0.9933 | -0.0644 | (-0.1301, 0.0013) | 0.9707 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0253 | (-0.0864, 0.0350) | 0.7823 | -0.0253 | (-0.0560, 0.0176) | 0.8907 |
| controlled_alt_vs_controlled_default | naturalness | 0.0119 | (-0.0082, 0.0323) | 0.1273 | 0.0119 | (-0.0125, 0.0394) | 0.2770 |
| controlled_alt_vs_controlled_default | quest_state_correctness | -0.0582 | (-0.1121, -0.0106) | 0.9947 | -0.0582 | (-0.1163, -0.0007) | 0.9830 |
| controlled_alt_vs_controlled_default | lore_consistency | -0.0416 | (-0.0755, -0.0154) | 1.0000 | -0.0416 | (-0.0710, -0.0136) | 1.0000 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | objective_completion_support | -0.0071 | (-0.0218, 0.0063) | 0.8230 | -0.0071 | (-0.0204, 0.0096) | 0.8243 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | -0.0137 | (-0.0248, -0.0046) | 0.9997 | -0.0137 | (-0.0265, -0.0052) | 1.0000 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | -0.0064 | (-0.0232, 0.0097) | 0.7680 | -0.0064 | (-0.0239, 0.0110) | 0.7040 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0841 | (-0.1568, -0.0220) | 0.9970 | -0.0841 | (-0.1644, -0.0041) | 0.9900 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0184 | (-0.0424, 0.0049) | 0.9397 | -0.0184 | (-0.0506, 0.0131) | 0.8723 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0196 | (-0.0958, 0.0595) | 0.6843 | -0.0196 | (-0.0583, 0.0433) | 0.7597 |
| controlled_alt_vs_controlled_default | persona_style | -0.0482 | (-0.0911, -0.0130) | 0.9970 | -0.0482 | (-0.0964, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0047 | (-0.0051, 0.0163) | 0.1993 | 0.0047 | (-0.0073, 0.0189) | 0.2543 |
| controlled_alt_vs_controlled_default | length_score | 0.0208 | (-0.0514, 0.1000) | 0.3017 | 0.0208 | (-0.0683, 0.1231) | 0.3620 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0583 | (0.0000, 0.1313) | 0.0553 | 0.0583 | (0.0000, 0.1250) | 0.0803 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0363 | (-0.0746, -0.0004) | 0.9783 | -0.0363 | (-0.0682, 0.0016) | 0.9747 |
| controlled_alt_vs_proposed_raw | context_relevance | -0.0084 | (-0.0610, 0.0415) | 0.6217 | -0.0084 | (-0.0477, 0.0316) | 0.6363 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.0079 | (-0.0557, 0.0724) | 0.4137 | 0.0079 | (-0.0609, 0.0882) | 0.4420 |
| controlled_alt_vs_proposed_raw | naturalness | -0.0009 | (-0.0157, 0.0140) | 0.5467 | -0.0009 | (-0.0127, 0.0075) | 0.5820 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | -0.0089 | (-0.0552, 0.0364) | 0.6517 | -0.0089 | (-0.0426, 0.0266) | 0.6460 |
| controlled_alt_vs_proposed_raw | lore_consistency | -0.0051 | (-0.0313, 0.0176) | 0.6510 | -0.0051 | (-0.0216, 0.0105) | 0.7250 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_proposed_raw | objective_completion_support | 0.0026 | (-0.0210, 0.0269) | 0.4223 | 0.0026 | (-0.0142, 0.0288) | 0.4660 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 0.0000 | (-0.0165, 0.0165) | 0.4900 | 0.0000 | (-0.0099, 0.0110) | 0.4810 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | 0.0001 | (-0.0194, 0.0193) | 0.4983 | 0.0001 | (-0.0143, 0.0156) | 0.4670 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | -0.0095 | (-0.0826, 0.0572) | 0.6070 | -0.0095 | (-0.0584, 0.0388) | 0.6317 |
| controlled_alt_vs_proposed_raw | context_overlap | -0.0059 | (-0.0283, 0.0186) | 0.7110 | -0.0059 | (-0.0280, 0.0153) | 0.6943 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.0187 | (-0.0538, 0.0958) | 0.3247 | 0.0187 | (-0.0585, 0.1083) | 0.3780 |
| controlled_alt_vs_proposed_raw | persona_style | -0.0352 | (-0.0847, 0.0169) | 0.9137 | -0.0352 | (-0.1065, 0.0000) | 1.0000 |
| controlled_alt_vs_proposed_raw | distinct1 | 0.0010 | (-0.0168, 0.0181) | 0.4423 | 0.0010 | (-0.0126, 0.0102) | 0.4253 |
| controlled_alt_vs_proposed_raw | length_score | -0.0139 | (-0.0722, 0.0486) | 0.6973 | -0.0139 | (-0.0433, 0.0071) | 0.9180 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0146 | (-0.0292, 0.0729) | 0.3950 | 0.0146 | (0.0000, 0.0404) | 0.3493 |
| controlled_alt_vs_proposed_raw | overall_quality | -0.0005 | (-0.0414, 0.0399) | 0.5140 | -0.0005 | (-0.0381, 0.0372) | 0.5393 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.0431 | (0.0069, 0.0864) | 0.0073 | 0.0431 | (0.0155, 0.0770) | 0.0003 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.0022 | (-0.0817, 0.0796) | 0.4620 | 0.0022 | (-0.1473, 0.0951) | 0.4527 |
| controlled_alt_vs_candidate_no_context | naturalness | -0.0201 | (-0.0384, -0.0027) | 0.9883 | -0.0201 | (-0.0373, -0.0038) | 0.9937 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 0.0349 | (0.0025, 0.0723) | 0.0143 | 0.0349 | (0.0108, 0.0594) | 0.0000 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 0.0163 | (0.0029, 0.0324) | 0.0107 | 0.0163 | (0.0087, 0.0226) | 0.0000 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_candidate_no_context | objective_completion_support | -0.0060 | (-0.0269, 0.0152) | 0.7160 | -0.0060 | (-0.0174, 0.0101) | 0.7677 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 0.0042 | (-0.0052, 0.0144) | 0.2087 | 0.0042 | (-0.0037, 0.0140) | 0.1837 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | -0.0055 | (-0.0239, 0.0100) | 0.7297 | -0.0055 | (-0.0183, 0.0111) | 0.7827 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.0568 | (0.0106, 0.1152) | 0.0070 | 0.0568 | (0.0203, 0.1005) | 0.0003 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0110 | (-0.0046, 0.0315) | 0.0997 | 0.0110 | (0.0000, 0.0238) | 0.0250 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.0196 | (-0.0776, 0.1115) | 0.3417 | 0.0196 | (-0.1539, 0.1288) | 0.2783 |
| controlled_alt_vs_candidate_no_context | persona_style | -0.0677 | (-0.1081, -0.0299) | 1.0000 | -0.0677 | (-0.1420, 0.0000) | 1.0000 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0087 | (-0.0283, 0.0104) | 0.8083 | -0.0087 | (-0.0262, 0.0037) | 0.8847 |
| controlled_alt_vs_candidate_no_context | length_score | -0.0903 | (-0.1667, -0.0139) | 0.9900 | -0.0903 | (-0.1515, -0.0250) | 0.9980 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.0146 | (-0.0292, 0.0583) | 0.4000 | 0.0146 | (0.0000, 0.0477) | 0.3253 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.0179 | (-0.0250, 0.0632) | 0.2130 | 0.0179 | (-0.0450, 0.0667) | 0.2623 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 11 | 3 | 10 | 0.6667 | 0.7857 |
| proposed_vs_candidate_no_context | persona_consistency | 5 | 6 | 13 | 0.4792 | 0.4545 |
| proposed_vs_candidate_no_context | naturalness | 5 | 10 | 9 | 0.3958 | 0.3333 |
| proposed_vs_candidate_no_context | quest_state_correctness | 11 | 3 | 10 | 0.6667 | 0.7857 |
| proposed_vs_candidate_no_context | lore_consistency | 7 | 1 | 16 | 0.6250 | 0.8750 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| proposed_vs_candidate_no_context | objective_completion_support | 5 | 6 | 13 | 0.4792 | 0.4545 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 9 | 6 | 9 | 0.5625 | 0.6000 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | 6 | 7 | 11 | 0.4792 | 0.4615 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 6 | 1 | 17 | 0.6042 | 0.8571 |
| proposed_vs_candidate_no_context | context_overlap | 12 | 2 | 10 | 0.7083 | 0.8571 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 4 | 4 | 16 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | persona_style | 2 | 4 | 18 | 0.4583 | 0.3333 |
| proposed_vs_candidate_no_context | distinct1 | 5 | 9 | 10 | 0.4167 | 0.3571 |
| proposed_vs_candidate_no_context | length_score | 6 | 9 | 9 | 0.4375 | 0.4000 |
| proposed_vs_candidate_no_context | sentence_score | 2 | 2 | 20 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | overall_quality | 10 | 5 | 9 | 0.6042 | 0.6667 |
| controlled_vs_proposed_raw | context_relevance | 12 | 6 | 6 | 0.6250 | 0.6667 |
| controlled_vs_proposed_raw | persona_consistency | 9 | 7 | 8 | 0.5417 | 0.5625 |
| controlled_vs_proposed_raw | naturalness | 6 | 12 | 6 | 0.3750 | 0.3333 |
| controlled_vs_proposed_raw | quest_state_correctness | 12 | 6 | 6 | 0.6250 | 0.6667 |
| controlled_vs_proposed_raw | lore_consistency | 9 | 3 | 12 | 0.6250 | 0.7500 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_vs_proposed_raw | objective_completion_support | 5 | 3 | 16 | 0.5417 | 0.6250 |
| controlled_vs_proposed_raw | gameplay_usefulness | 8 | 10 | 6 | 0.4583 | 0.4444 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 9 | 4 | 11 | 0.6042 | 0.6923 |
| controlled_vs_proposed_raw | context_keyword_coverage | 8 | 5 | 11 | 0.5625 | 0.6154 |
| controlled_vs_proposed_raw | context_overlap | 13 | 5 | 6 | 0.6667 | 0.7222 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 7 | 5 | 12 | 0.5417 | 0.5833 |
| controlled_vs_proposed_raw | persona_style | 4 | 4 | 16 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | distinct1 | 9 | 9 | 6 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | length_score | 7 | 10 | 7 | 0.4375 | 0.4118 |
| controlled_vs_proposed_raw | sentence_score | 2 | 5 | 17 | 0.4375 | 0.2857 |
| controlled_vs_proposed_raw | overall_quality | 9 | 9 | 6 | 0.5000 | 0.5000 |
| controlled_vs_candidate_no_context | context_relevance | 13 | 5 | 6 | 0.6667 | 0.7222 |
| controlled_vs_candidate_no_context | persona_consistency | 10 | 6 | 8 | 0.5833 | 0.6250 |
| controlled_vs_candidate_no_context | naturalness | 7 | 11 | 6 | 0.4167 | 0.3889 |
| controlled_vs_candidate_no_context | quest_state_correctness | 13 | 5 | 6 | 0.6667 | 0.7222 |
| controlled_vs_candidate_no_context | lore_consistency | 9 | 1 | 14 | 0.6667 | 0.9000 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_vs_candidate_no_context | objective_completion_support | 7 | 6 | 11 | 0.5208 | 0.5385 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 12 | 6 | 6 | 0.6250 | 0.6667 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 8 | 7 | 9 | 0.5208 | 0.5333 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 10 | 1 | 13 | 0.6875 | 0.9091 |
| controlled_vs_candidate_no_context | context_overlap | 13 | 5 | 6 | 0.6667 | 0.7222 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 10 | 6 | 8 | 0.5833 | 0.6250 |
| controlled_vs_candidate_no_context | persona_style | 1 | 4 | 19 | 0.4375 | 0.2000 |
| controlled_vs_candidate_no_context | distinct1 | 4 | 14 | 6 | 0.2917 | 0.2222 |
| controlled_vs_candidate_no_context | length_score | 7 | 10 | 7 | 0.4375 | 0.4118 |
| controlled_vs_candidate_no_context | sentence_score | 2 | 5 | 17 | 0.4375 | 0.2857 |
| controlled_vs_candidate_no_context | overall_quality | 13 | 5 | 6 | 0.6667 | 0.7222 |
| controlled_alt_vs_controlled_default | context_relevance | 4 | 8 | 12 | 0.4167 | 0.3333 |
| controlled_alt_vs_controlled_default | persona_consistency | 3 | 7 | 14 | 0.4167 | 0.3000 |
| controlled_alt_vs_controlled_default | naturalness | 7 | 5 | 12 | 0.5417 | 0.5833 |
| controlled_alt_vs_controlled_default | quest_state_correctness | 4 | 8 | 12 | 0.4167 | 0.3333 |
| controlled_alt_vs_controlled_default | lore_consistency | 0 | 9 | 15 | 0.3125 | 0.0000 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_alt_vs_controlled_default | objective_completion_support | 3 | 4 | 17 | 0.4792 | 0.4286 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | 2 | 10 | 12 | 0.3333 | 0.1667 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | 4 | 6 | 14 | 0.4583 | 0.4000 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 3 | 7 | 14 | 0.4167 | 0.3000 |
| controlled_alt_vs_controlled_default | context_overlap | 4 | 8 | 12 | 0.4167 | 0.3333 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 2 | 5 | 17 | 0.4375 | 0.2857 |
| controlled_alt_vs_controlled_default | persona_style | 1 | 5 | 18 | 0.4167 | 0.1667 |
| controlled_alt_vs_controlled_default | distinct1 | 8 | 4 | 12 | 0.5833 | 0.6667 |
| controlled_alt_vs_controlled_default | length_score | 6 | 6 | 12 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | sentence_score | 5 | 1 | 18 | 0.5833 | 0.8333 |
| controlled_alt_vs_controlled_default | overall_quality | 3 | 9 | 12 | 0.3750 | 0.2500 |
| controlled_alt_vs_proposed_raw | context_relevance | 9 | 9 | 6 | 0.5000 | 0.5000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 7 | 8 | 9 | 0.4792 | 0.4667 |
| controlled_alt_vs_proposed_raw | naturalness | 7 | 11 | 6 | 0.4167 | 0.3889 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 8 | 10 | 6 | 0.4583 | 0.4444 |
| controlled_alt_vs_proposed_raw | lore_consistency | 5 | 5 | 14 | 0.5000 | 0.5000 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_alt_vs_proposed_raw | objective_completion_support | 5 | 5 | 14 | 0.5000 | 0.5000 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 6 | 12 | 6 | 0.3750 | 0.3333 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | 6 | 4 | 14 | 0.5417 | 0.6000 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 5 | 6 | 13 | 0.4792 | 0.4545 |
| controlled_alt_vs_proposed_raw | context_overlap | 8 | 10 | 6 | 0.4583 | 0.4444 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 7 | 6 | 11 | 0.5208 | 0.5385 |
| controlled_alt_vs_proposed_raw | persona_style | 1 | 7 | 16 | 0.3750 | 0.1250 |
| controlled_alt_vs_proposed_raw | distinct1 | 10 | 8 | 6 | 0.5417 | 0.5556 |
| controlled_alt_vs_proposed_raw | length_score | 8 | 10 | 6 | 0.4583 | 0.4444 |
| controlled_alt_vs_proposed_raw | sentence_score | 2 | 1 | 21 | 0.5208 | 0.6667 |
| controlled_alt_vs_proposed_raw | overall_quality | 7 | 11 | 6 | 0.4167 | 0.3889 |
| controlled_alt_vs_candidate_no_context | context_relevance | 11 | 7 | 6 | 0.5833 | 0.6111 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 10 | 6 | 8 | 0.5833 | 0.6250 |
| controlled_alt_vs_candidate_no_context | naturalness | 6 | 12 | 6 | 0.3750 | 0.3333 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 11 | 7 | 6 | 0.5833 | 0.6111 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 5 | 1 | 18 | 0.5833 | 0.8333 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_alt_vs_candidate_no_context | objective_completion_support | 5 | 6 | 13 | 0.4792 | 0.4545 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 10 | 8 | 6 | 0.5417 | 0.5556 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | 6 | 5 | 13 | 0.5208 | 0.5455 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 7 | 2 | 15 | 0.6042 | 0.7778 |
| controlled_alt_vs_candidate_no_context | context_overlap | 11 | 7 | 6 | 0.5833 | 0.6111 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 10 | 5 | 9 | 0.6042 | 0.6667 |
| controlled_alt_vs_candidate_no_context | persona_style | 0 | 8 | 16 | 0.3333 | 0.0000 |
| controlled_alt_vs_candidate_no_context | distinct1 | 7 | 11 | 6 | 0.4167 | 0.3889 |
| controlled_alt_vs_candidate_no_context | length_score | 5 | 12 | 7 | 0.3542 | 0.2941 |
| controlled_alt_vs_candidate_no_context | sentence_score | 2 | 1 | 21 | 0.5208 | 0.6667 |
| controlled_alt_vs_candidate_no_context | overall_quality | 12 | 6 | 6 | 0.6250 | 0.6667 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.2083 | 0.1250 | 0.8750 |
| proposed_contextual_controlled_tuned | 0.0000 | 0.0000 | 0.0833 | 0.1250 | 0.8750 |
| proposed_contextual | 0.0000 | 0.0000 | 0.5833 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.4583 | 0.0000 | 0.0000 |

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
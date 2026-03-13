# Proposal Alignment Evaluation Report

- Run ID: `20260313T005612Z`
- Generated: `2026-03-13T00:59:23.277825+00:00`
- Scenarios: `artifacts\proposal_control_tuning\auto_tune\20260313T004330Z\valid_runs\trial_000\seed_29\20260313T005612Z\scenarios.jsonl`
- Scenario count: `12`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_tuned`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.1169 (0.0567, 0.1908) | 0.3710 (0.2974, 0.4379) | 0.8787 (0.8558, 0.9067) | 0.3591 (0.3131, 0.3994) | n/a |
| proposed_contextual_controlled_tuned | 0.1607 (0.0922, 0.2370) | 0.3765 (0.3036, 0.4413) | 0.8828 (0.8543, 0.9119) | 0.3830 (0.3400, 0.4237) | n/a |
| proposed_contextual | 0.0776 (0.0403, 0.1146) | 0.3430 (0.2654, 0.4196) | 0.9078 (0.8867, 0.9301) | 0.3356 (0.3050, 0.3665) | n/a |
| candidate_no_context | 0.0412 (0.0162, 0.0722) | 0.3066 (0.2141, 0.4041) | 0.8901 (0.8729, 0.9123) | 0.3026 (0.2649, 0.3407) | n/a |

## Game-facing Outcome Metrics (mean, 95% CI)
| Arm | Quest-state Correctness | Lore Consistency | Contradiction Safety | Objective Completion Support | Gameplay Usefulness | Time-pressure Acceptability |
|---|---:|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.1944 (0.1443, 0.2585) | 0.0367 (0.0000, 0.0907) | 1.0000 (1.0000, 1.0000) | 0.0919 (0.0512, 0.1325) | 0.3067 (0.2936, 0.3192) | 0.2860 (0.2536, 0.3174) |
| proposed_contextual_controlled_tuned | 0.2306 (0.1735, 0.3030) | 0.0676 (0.0175, 0.1266) | 1.0000 (1.0000, 1.0000) | 0.0671 (0.0342, 0.1019) | 0.3120 (0.2933, 0.3311) | 0.2796 (0.2505, 0.3094) |
| proposed_contextual | 0.1653 (0.1361, 0.1977) | 0.0300 (0.0000, 0.0624) | 1.0000 (1.0000, 1.0000) | 0.0640 (0.0311, 0.1006) | 0.3021 (0.2890, 0.3159) | 0.2814 (0.2621, 0.3056) |
| candidate_no_context | 0.1328 (0.1117, 0.1598) | 0.0052 (0.0000, 0.0124) | 1.0000 (1.0000, 1.0000) | 0.0550 (0.0278, 0.0886) | 0.2775 (0.2639, 0.2917) | 0.2706 (0.2518, 0.2883) |

- Multi-turn contradiction rate is reported as `1 - contradiction_safety` in row-level outputs.

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0364 | 0.8823 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0364 | 0.1186 |
| proposed_vs_candidate_no_context | naturalness | 0.0177 | 0.0199 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.0325 | 0.2445 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0248 | 4.7394 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | 0.0090 | 0.1641 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 0.0246 | 0.0887 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | 0.0109 | 0.0401 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0492 | 1.3684 |
| proposed_vs_candidate_no_context | context_overlap | 0.0064 | 0.1190 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0258 | 0.1040 |
| proposed_vs_candidate_no_context | persona_style | 0.0787 | 0.1455 |
| proposed_vs_candidate_no_context | distinct1 | 0.0117 | 0.0123 |
| proposed_vs_candidate_no_context | length_score | 0.0944 | 0.1709 |
| proposed_vs_candidate_no_context | sentence_score | -0.0583 | -0.0583 |
| proposed_vs_candidate_no_context | overall_quality | 0.0330 | 0.1092 |
| controlled_vs_proposed_raw | context_relevance | 0.0393 | 0.5061 |
| controlled_vs_proposed_raw | persona_consistency | 0.0280 | 0.0817 |
| controlled_vs_proposed_raw | naturalness | -0.0291 | -0.0321 |
| controlled_vs_proposed_raw | quest_state_correctness | 0.0291 | 0.1759 |
| controlled_vs_proposed_raw | lore_consistency | 0.0067 | 0.2216 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0279 | 0.4360 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0046 | 0.0152 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 0.0046 | 0.0163 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.0582 | 0.6830 |
| controlled_vs_proposed_raw | context_overlap | -0.0049 | -0.0817 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.0337 | 0.1232 |
| controlled_vs_proposed_raw | persona_style | 0.0052 | 0.0084 |
| controlled_vs_proposed_raw | distinct1 | 0.0168 | 0.0175 |
| controlled_vs_proposed_raw | length_score | -0.2083 | -0.3219 |
| controlled_vs_proposed_raw | sentence_score | 0.0583 | 0.0619 |
| controlled_vs_proposed_raw | overall_quality | 0.0234 | 0.0697 |
| controlled_vs_candidate_no_context | context_relevance | 0.0757 | 1.8350 |
| controlled_vs_candidate_no_context | persona_consistency | 0.0644 | 0.2100 |
| controlled_vs_candidate_no_context | naturalness | -0.0114 | -0.0128 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.0616 | 0.4635 |
| controlled_vs_candidate_no_context | lore_consistency | 0.0315 | 6.0110 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | 0.0369 | 0.6717 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0292 | 0.1053 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 0.0154 | 0.0571 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.1074 | 2.9860 |
| controlled_vs_candidate_no_context | context_overlap | 0.0015 | 0.0275 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.0595 | 0.2400 |
| controlled_vs_candidate_no_context | persona_style | 0.0839 | 0.1551 |
| controlled_vs_candidate_no_context | distinct1 | 0.0285 | 0.0300 |
| controlled_vs_candidate_no_context | length_score | -0.1139 | -0.2060 |
| controlled_vs_candidate_no_context | sentence_score | 0.0000 | 0.0000 |
| controlled_vs_candidate_no_context | overall_quality | 0.0564 | 0.1865 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0438 | 0.3749 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0054 | 0.0146 |
| controlled_alt_vs_controlled_default | naturalness | 0.0041 | 0.0047 |
| controlled_alt_vs_controlled_default | quest_state_correctness | 0.0362 | 0.1863 |
| controlled_alt_vs_controlled_default | lore_consistency | 0.0309 | 0.8424 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_controlled_default | objective_completion_support | -0.0249 | -0.2704 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | 0.0054 | 0.0175 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | -0.0064 | -0.0225 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0568 | 0.3961 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0135 | 0.2455 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0198 | 0.0645 |
| controlled_alt_vs_controlled_default | persona_style | -0.0523 | -0.0836 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0244 | -0.0250 |
| controlled_alt_vs_controlled_default | length_score | 0.0694 | 0.1582 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0000 | 0.0000 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0240 | 0.0667 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.0831 | 1.0707 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.0334 | 0.0975 |
| controlled_alt_vs_proposed_raw | naturalness | -0.0250 | -0.0275 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 0.0653 | 0.3950 |
| controlled_alt_vs_proposed_raw | lore_consistency | 0.0376 | 1.2506 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_proposed_raw | objective_completion_support | 0.0031 | 0.0477 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 0.0100 | 0.0330 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | -0.0019 | -0.0066 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.1150 | 1.3496 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0086 | 0.1437 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.0536 | 0.1957 |
| controlled_alt_vs_proposed_raw | persona_style | -0.0471 | -0.0759 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0077 | -0.0080 |
| controlled_alt_vs_proposed_raw | length_score | -0.1389 | -0.2146 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0583 | 0.0619 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.0474 | 0.1411 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.1195 | 2.8978 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.0698 | 0.2277 |
| controlled_alt_vs_candidate_no_context | naturalness | -0.0073 | -0.0082 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 0.0978 | 0.7361 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 0.0624 | 11.9171 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_candidate_no_context | objective_completion_support | 0.0121 | 0.2197 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 0.0346 | 0.1246 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | 0.0090 | 0.0333 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.1643 | 4.5649 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0150 | 0.2797 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.0794 | 0.3200 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0317 | 0.0585 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0040 | 0.0042 |
| controlled_alt_vs_candidate_no_context | length_score | -0.0444 | -0.0804 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.0000 | 0.0000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.0804 | 0.2657 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0364 | (0.0010, 0.0738) | 0.0213 | 0.0364 | (0.0000, 0.0725) | 0.0323 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0364 | (-0.0806, 0.1516) | 0.2700 | 0.0364 | (0.0000, 0.0745) | 0.0403 |
| proposed_vs_candidate_no_context | naturalness | 0.0177 | (-0.0057, 0.0440) | 0.0753 | 0.0177 | (0.0000, 0.0330) | 0.0350 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.0325 | (0.0025, 0.0661) | 0.0187 | 0.0325 | (0.0000, 0.0686) | 0.0410 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0248 | (-0.0011, 0.0561) | 0.0347 | 0.0248 | (0.0000, 0.0595) | 0.3060 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | 0.0090 | (-0.0342, 0.0499) | 0.3240 | 0.0090 | (-0.0243, 0.0460) | 0.4187 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 0.0246 | (0.0053, 0.0459) | 0.0063 | 0.0246 | (0.0000, 0.0440) | 0.0333 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | 0.0109 | (-0.0175, 0.0416) | 0.2390 | 0.0109 | (0.0000, 0.0199) | 0.0363 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0492 | (0.0069, 0.0985) | 0.0133 | 0.0492 | (0.0000, 0.1000) | 0.0330 |
| proposed_vs_candidate_no_context | context_overlap | 0.0064 | (-0.0056, 0.0175) | 0.1360 | 0.0064 | (0.0000, 0.0083) | 0.0327 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0258 | (-0.1250, 0.1706) | 0.3807 | 0.0258 | (0.0000, 0.0333) | 0.0323 |
| proposed_vs_candidate_no_context | persona_style | 0.0787 | (-0.0325, 0.1899) | 0.0927 | 0.0787 | (-0.0500, 0.2389) | 0.2950 |
| proposed_vs_candidate_no_context | distinct1 | 0.0117 | (-0.0150, 0.0401) | 0.2040 | 0.0117 | (0.0000, 0.0248) | 0.0427 |
| proposed_vs_candidate_no_context | length_score | 0.0944 | (-0.0250, 0.2111) | 0.0600 | 0.0944 | (0.0000, 0.1933) | 0.0333 |
| proposed_vs_candidate_no_context | sentence_score | -0.0583 | (-0.1458, 0.0000) | 1.0000 | -0.0583 | (-0.0700, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | overall_quality | 0.0330 | (-0.0164, 0.0788) | 0.0890 | 0.0330 | (0.0000, 0.0603) | 0.0397 |
| controlled_vs_proposed_raw | context_relevance | 0.0393 | (-0.0158, 0.1176) | 0.1410 | 0.0393 | (-0.0173, 0.2121) | 0.1470 |
| controlled_vs_proposed_raw | persona_consistency | 0.0280 | (-0.0544, 0.1185) | 0.2713 | 0.0280 | (-0.1271, 0.1867) | 0.3027 |
| controlled_vs_proposed_raw | naturalness | -0.0291 | (-0.0645, 0.0108) | 0.9257 | -0.0291 | (-0.0496, -0.0010) | 1.0000 |
| controlled_vs_proposed_raw | quest_state_correctness | 0.0291 | (-0.0186, 0.0989) | 0.1980 | 0.0291 | (-0.0216, 0.1698) | 0.1450 |
| controlled_vs_proposed_raw | lore_consistency | 0.0067 | (-0.0405, 0.0613) | 0.4340 | 0.0067 | (-0.0305, 0.1161) | 0.4100 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0279 | (-0.0074, 0.0653) | 0.0657 | 0.0279 | (0.0208, 0.0347) | 0.0000 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0046 | (-0.0110, 0.0212) | 0.3020 | 0.0046 | (-0.0112, 0.0247) | 0.2653 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 0.0046 | (-0.0185, 0.0260) | 0.3360 | 0.0046 | (-0.0370, 0.0139) | 0.2630 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.0582 | (-0.0139, 0.1658) | 0.0893 | 0.0582 | (-0.0167, 0.3000) | 0.0347 |
| controlled_vs_proposed_raw | context_overlap | -0.0049 | (-0.0225, 0.0121) | 0.6930 | -0.0049 | (-0.0189, 0.0069) | 0.7140 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.0337 | (-0.0734, 0.1429) | 0.2917 | 0.0337 | (-0.1667, 0.2333) | 0.2980 |
| controlled_vs_proposed_raw | persona_style | 0.0052 | (-0.0573, 0.0677) | 0.4487 | 0.0052 | (0.0000, 0.0312) | 0.2860 |
| controlled_vs_proposed_raw | distinct1 | 0.0168 | (0.0026, 0.0296) | 0.0127 | 0.0168 | (0.0117, 0.0379) | 0.0000 |
| controlled_vs_proposed_raw | length_score | -0.2083 | (-0.3667, -0.0333) | 0.9877 | -0.2083 | (-0.3167, -0.0667) | 1.0000 |
| controlled_vs_proposed_raw | sentence_score | 0.0583 | (0.0000, 0.1458) | 0.1080 | 0.0583 | (0.0000, 0.0700) | 0.0360 |
| controlled_vs_proposed_raw | overall_quality | 0.0234 | (-0.0087, 0.0573) | 0.0787 | 0.0234 | (-0.0134, 0.0525) | 0.1557 |
| controlled_vs_candidate_no_context | context_relevance | 0.0757 | (0.0078, 0.1557) | 0.0097 | 0.0757 | (0.0416, 0.2121) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.0644 | (-0.0920, 0.1998) | 0.2020 | 0.0644 | (-0.1271, 0.2611) | 0.3057 |
| controlled_vs_candidate_no_context | naturalness | -0.0114 | (-0.0468, 0.0286) | 0.7363 | -0.0114 | (-0.0482, 0.0320) | 0.7423 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.0616 | (0.0066, 0.1277) | 0.0090 | 0.0616 | (0.0328, 0.1698) | 0.0000 |
| controlled_vs_candidate_no_context | lore_consistency | 0.0315 | (-0.0079, 0.0873) | 0.1163 | 0.0315 | (0.0000, 0.1161) | 0.0380 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | 0.0369 | (-0.0101, 0.0808) | 0.0613 | 0.0369 | (-0.0003, 0.0807) | 0.0340 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0292 | (0.0097, 0.0462) | 0.0017 | 0.0292 | (0.0247, 0.0328) | 0.0000 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 0.0154 | (-0.0247, 0.0541) | 0.2303 | 0.0154 | (-0.0370, 0.0318) | 0.1360 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.1074 | (0.0215, 0.2201) | 0.0017 | 0.1074 | (0.0545, 0.3000) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0015 | (-0.0173, 0.0205) | 0.4467 | 0.0015 | (-0.0106, 0.0114) | 0.3883 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.0595 | (-0.1290, 0.2282) | 0.2500 | 0.0595 | (-0.1667, 0.2667) | 0.2913 |
| controlled_vs_candidate_no_context | persona_style | 0.0839 | (-0.0315, 0.2051) | 0.0703 | 0.0839 | (-0.0500, 0.2389) | 0.2693 |
| controlled_vs_candidate_no_context | distinct1 | 0.0285 | (0.0012, 0.0563) | 0.0210 | 0.0285 | (0.0166, 0.0379) | 0.0000 |
| controlled_vs_candidate_no_context | length_score | -0.1139 | (-0.2862, 0.0667) | 0.8867 | -0.1139 | (-0.3167, 0.1267) | 0.8590 |
| controlled_vs_candidate_no_context | sentence_score | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | overall_quality | 0.0564 | (-0.0137, 0.1159) | 0.0550 | 0.0564 | (0.0056, 0.1128) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0438 | (-0.0844, 0.1583) | 0.2510 | 0.0438 | (-0.0051, 0.0786) | 0.0357 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0054 | (-0.0783, 0.0921) | 0.4570 | 0.0054 | (-0.0851, 0.0714) | 0.3760 |
| controlled_alt_vs_controlled_default | naturalness | 0.0041 | (-0.0323, 0.0376) | 0.4033 | 0.0041 | (-0.0140, 0.0289) | 0.2933 |
| controlled_alt_vs_controlled_default | quest_state_correctness | 0.0362 | (-0.0691, 0.1401) | 0.2343 | 0.0362 | (0.0092, 0.0679) | 0.0000 |
| controlled_alt_vs_controlled_default | lore_consistency | 0.0309 | (-0.0537, 0.1136) | 0.2320 | 0.0309 | (-0.0050, 0.0762) | 0.2947 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | objective_completion_support | -0.0249 | (-0.0793, 0.0294) | 0.8017 | -0.0249 | (-0.0460, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | 0.0054 | (-0.0195, 0.0288) | 0.3103 | 0.0054 | (-0.0064, 0.0205) | 0.3003 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | -0.0064 | (-0.0450, 0.0354) | 0.6210 | -0.0064 | (-0.0083, -0.0036) | 1.0000 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0568 | (-0.1057, 0.2117) | 0.2433 | 0.0568 | (0.0000, 0.1000) | 0.0307 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0135 | (-0.0203, 0.0447) | 0.2147 | 0.0135 | (-0.0169, 0.0285) | 0.1570 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0198 | (-0.0873, 0.1270) | 0.3840 | 0.0198 | (-0.1000, 0.1143) | 0.3787 |
| controlled_alt_vs_controlled_default | persona_style | -0.0523 | (-0.1356, 0.0208) | 0.9240 | -0.0523 | (-0.1000, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0244 | (-0.0437, -0.0074) | 0.9993 | -0.0244 | (-0.0344, -0.0145) | 1.0000 |
| controlled_alt_vs_controlled_default | length_score | 0.0694 | (-0.1028, 0.2417) | 0.2223 | 0.0694 | (-0.0333, 0.2133) | 0.2890 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0240 | (-0.0333, 0.0839) | 0.2097 | 0.0240 | (0.0105, 0.0387) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.0831 | (0.0095, 0.1712) | 0.0123 | 0.0831 | (0.0554, 0.2070) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.0334 | (-0.0321, 0.0979) | 0.1547 | 0.0334 | (-0.0604, 0.1016) | 0.2557 |
| controlled_alt_vs_proposed_raw | naturalness | -0.0250 | (-0.0571, 0.0075) | 0.9373 | -0.0250 | (-0.0606, -0.0150) | 1.0000 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 0.0653 | (-0.0001, 0.1443) | 0.0257 | 0.0653 | (0.0389, 0.1790) | 0.0000 |
| controlled_alt_vs_proposed_raw | lore_consistency | 0.0376 | (-0.0109, 0.0951) | 0.1123 | 0.0376 | (0.0000, 0.1111) | 0.0383 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_proposed_raw | objective_completion_support | 0.0031 | (-0.0242, 0.0333) | 0.4457 | 0.0031 | (-0.0113, 0.0208) | 0.2620 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 0.0100 | (-0.0099, 0.0314) | 0.1730 | 0.0100 | (0.0074, 0.0183) | 0.0000 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | -0.0019 | (-0.0288, 0.0260) | 0.5630 | -0.0019 | (-0.0406, 0.0062) | 0.7027 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.1150 | (0.0158, 0.2283) | 0.0080 | 0.1150 | (0.0727, 0.3000) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0086 | (-0.0151, 0.0299) | 0.2320 | 0.0086 | (-0.0101, 0.0150) | 0.0350 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.0536 | (-0.0278, 0.1429) | 0.1063 | 0.0536 | (-0.0833, 0.1333) | 0.1560 |
| controlled_alt_vs_proposed_raw | persona_style | -0.0471 | (-0.1120, 0.0130) | 0.9357 | -0.0471 | (-0.1000, 0.0312) | 0.9560 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0077 | (-0.0349, 0.0160) | 0.6940 | -0.0077 | (-0.0227, 0.0234) | 0.8520 |
| controlled_alt_vs_proposed_raw | length_score | -0.1389 | (-0.2640, 0.0056) | 0.9677 | -0.1389 | (-0.3500, -0.0933) | 1.0000 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0583 | (0.0000, 0.1458) | 0.1040 | 0.0583 | (0.0000, 0.0700) | 0.0393 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.0474 | (0.0126, 0.0837) | 0.0043 | 0.0474 | (0.0253, 0.0633) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.1195 | (0.0459, 0.2067) | 0.0007 | 0.1195 | (0.0702, 0.2070) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.0698 | (-0.0637, 0.1899) | 0.1420 | 0.0698 | (-0.0604, 0.1760) | 0.1480 |
| controlled_alt_vs_candidate_no_context | naturalness | -0.0073 | (-0.0383, 0.0276) | 0.6723 | -0.0073 | (-0.0606, 0.0179) | 0.7403 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 0.0978 | (0.0304, 0.1769) | 0.0010 | 0.0978 | (0.0482, 0.1790) | 0.0000 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 0.0624 | (0.0117, 0.1222) | 0.0080 | 0.0624 | (0.0000, 0.1111) | 0.0403 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_candidate_no_context | objective_completion_support | 0.0121 | (-0.0242, 0.0481) | 0.2760 | 0.0121 | (-0.0140, 0.0347) | 0.1520 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 0.0346 | (0.0094, 0.0609) | 0.0007 | 0.0346 | (0.0183, 0.0533) | 0.0000 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | 0.0090 | (-0.0232, 0.0416) | 0.2920 | 0.0090 | (-0.0406, 0.0261) | 0.2650 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.1643 | (0.0638, 0.2788) | 0.0000 | 0.1643 | (0.0909, 0.3000) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0150 | (-0.0096, 0.0385) | 0.1137 | 0.0150 | (-0.0101, 0.0220) | 0.0380 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.0794 | (-0.0675, 0.2202) | 0.1333 | 0.0794 | (-0.0833, 0.1667) | 0.1450 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0317 | (-0.0875, 0.1562) | 0.3220 | 0.0317 | (-0.1500, 0.2135) | 0.3660 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0040 | (-0.0361, 0.0411) | 0.4227 | 0.0040 | (-0.0018, 0.0234) | 0.1580 |
| controlled_alt_vs_candidate_no_context | length_score | -0.0444 | (-0.2083, 0.1306) | 0.6993 | -0.0444 | (-0.3500, 0.0933) | 0.7340 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.0804 | (0.0192, 0.1344) | 0.0060 | 0.0804 | (0.0443, 0.1234) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 7 | 3 | 2 | 0.6667 | 0.7000 |
| proposed_vs_candidate_no_context | persona_consistency | 7 | 3 | 2 | 0.6667 | 0.7000 |
| proposed_vs_candidate_no_context | naturalness | 5 | 5 | 2 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | quest_state_correctness | 6 | 4 | 2 | 0.5833 | 0.6000 |
| proposed_vs_candidate_no_context | lore_consistency | 3 | 2 | 7 | 0.5417 | 0.6000 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 12 | 0.5000 | nan |
| proposed_vs_candidate_no_context | objective_completion_support | 4 | 3 | 5 | 0.5417 | 0.5714 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 8 | 2 | 2 | 0.7500 | 0.8000 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | 5 | 4 | 3 | 0.5417 | 0.5556 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 5 | 1 | 6 | 0.6667 | 0.8333 |
| proposed_vs_candidate_no_context | context_overlap | 7 | 3 | 2 | 0.6667 | 0.7000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 4 | 3 | 5 | 0.5417 | 0.5714 |
| proposed_vs_candidate_no_context | persona_style | 5 | 2 | 5 | 0.6250 | 0.7143 |
| proposed_vs_candidate_no_context | distinct1 | 5 | 4 | 3 | 0.5417 | 0.5556 |
| proposed_vs_candidate_no_context | length_score | 6 | 3 | 3 | 0.6250 | 0.6667 |
| proposed_vs_candidate_no_context | sentence_score | 0 | 2 | 10 | 0.4167 | 0.0000 |
| proposed_vs_candidate_no_context | overall_quality | 7 | 3 | 2 | 0.6667 | 0.7000 |
| controlled_vs_proposed_raw | context_relevance | 7 | 5 | 0 | 0.5833 | 0.5833 |
| controlled_vs_proposed_raw | persona_consistency | 4 | 6 | 2 | 0.4167 | 0.4000 |
| controlled_vs_proposed_raw | naturalness | 3 | 9 | 0 | 0.2500 | 0.2500 |
| controlled_vs_proposed_raw | quest_state_correctness | 7 | 5 | 0 | 0.5833 | 0.5833 |
| controlled_vs_proposed_raw | lore_consistency | 2 | 2 | 8 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0 | 0 | 12 | 0.5000 | nan |
| controlled_vs_proposed_raw | objective_completion_support | 6 | 4 | 2 | 0.5833 | 0.6000 |
| controlled_vs_proposed_raw | gameplay_usefulness | 5 | 7 | 0 | 0.4167 | 0.4167 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 6 | 4 | 2 | 0.5833 | 0.6000 |
| controlled_vs_proposed_raw | context_keyword_coverage | 4 | 2 | 6 | 0.5833 | 0.6667 |
| controlled_vs_proposed_raw | context_overlap | 6 | 6 | 0 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 4 | 5 | 3 | 0.4583 | 0.4444 |
| controlled_vs_proposed_raw | persona_style | 2 | 2 | 8 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | distinct1 | 7 | 2 | 3 | 0.7083 | 0.7778 |
| controlled_vs_proposed_raw | length_score | 2 | 10 | 0 | 0.1667 | 0.1667 |
| controlled_vs_proposed_raw | sentence_score | 2 | 0 | 10 | 0.5833 | 1.0000 |
| controlled_vs_proposed_raw | overall_quality | 7 | 5 | 0 | 0.5833 | 0.5833 |
| controlled_vs_candidate_no_context | context_relevance | 8 | 4 | 0 | 0.6667 | 0.6667 |
| controlled_vs_candidate_no_context | persona_consistency | 7 | 4 | 1 | 0.6250 | 0.6364 |
| controlled_vs_candidate_no_context | naturalness | 4 | 8 | 0 | 0.3333 | 0.3333 |
| controlled_vs_candidate_no_context | quest_state_correctness | 8 | 4 | 0 | 0.6667 | 0.6667 |
| controlled_vs_candidate_no_context | lore_consistency | 2 | 3 | 7 | 0.4583 | 0.4000 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 12 | 0.5000 | nan |
| controlled_vs_candidate_no_context | objective_completion_support | 7 | 3 | 2 | 0.6667 | 0.7000 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 9 | 3 | 0 | 0.7500 | 0.7500 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 7 | 4 | 1 | 0.6250 | 0.6364 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 6 | 1 | 5 | 0.7083 | 0.8571 |
| controlled_vs_candidate_no_context | context_overlap | 6 | 6 | 0 | 0.5000 | 0.5000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 7 | 4 | 1 | 0.6250 | 0.6364 |
| controlled_vs_candidate_no_context | persona_style | 6 | 3 | 3 | 0.6250 | 0.6667 |
| controlled_vs_candidate_no_context | distinct1 | 8 | 2 | 2 | 0.7500 | 0.8000 |
| controlled_vs_candidate_no_context | length_score | 3 | 8 | 1 | 0.2917 | 0.2727 |
| controlled_vs_candidate_no_context | sentence_score | 0 | 0 | 12 | 0.5000 | nan |
| controlled_vs_candidate_no_context | overall_quality | 9 | 3 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_controlled_default | context_relevance | 6 | 4 | 2 | 0.5833 | 0.6000 |
| controlled_alt_vs_controlled_default | persona_consistency | 4 | 5 | 3 | 0.4583 | 0.4444 |
| controlled_alt_vs_controlled_default | naturalness | 6 | 4 | 2 | 0.5833 | 0.6000 |
| controlled_alt_vs_controlled_default | quest_state_correctness | 6 | 4 | 2 | 0.5833 | 0.6000 |
| controlled_alt_vs_controlled_default | lore_consistency | 4 | 2 | 6 | 0.5833 | 0.6667 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0 | 0 | 12 | 0.5000 | nan |
| controlled_alt_vs_controlled_default | objective_completion_support | 4 | 6 | 2 | 0.4167 | 0.4000 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | 6 | 4 | 2 | 0.5833 | 0.6000 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | 5 | 5 | 2 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 6 | 2 | 4 | 0.6667 | 0.7500 |
| controlled_alt_vs_controlled_default | context_overlap | 6 | 4 | 2 | 0.5833 | 0.6000 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 4 | 3 | 5 | 0.5417 | 0.5714 |
| controlled_alt_vs_controlled_default | persona_style | 1 | 4 | 7 | 0.3750 | 0.2000 |
| controlled_alt_vs_controlled_default | distinct1 | 1 | 8 | 3 | 0.2083 | 0.1111 |
| controlled_alt_vs_controlled_default | length_score | 8 | 2 | 2 | 0.7500 | 0.8000 |
| controlled_alt_vs_controlled_default | sentence_score | 0 | 0 | 12 | 0.5000 | nan |
| controlled_alt_vs_controlled_default | overall_quality | 7 | 3 | 2 | 0.6667 | 0.7000 |
| controlled_alt_vs_proposed_raw | context_relevance | 10 | 2 | 0 | 0.8333 | 0.8333 |
| controlled_alt_vs_proposed_raw | persona_consistency | 5 | 3 | 4 | 0.5833 | 0.6250 |
| controlled_alt_vs_proposed_raw | naturalness | 4 | 8 | 0 | 0.3333 | 0.3333 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 10 | 2 | 0 | 0.8333 | 0.8333 |
| controlled_alt_vs_proposed_raw | lore_consistency | 3 | 2 | 7 | 0.5417 | 0.6000 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0 | 0 | 12 | 0.5000 | nan |
| controlled_alt_vs_proposed_raw | objective_completion_support | 4 | 5 | 3 | 0.4583 | 0.4444 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 6 | 6 | 0 | 0.5000 | 0.5000 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | 5 | 5 | 2 | 0.5000 | 0.5000 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 7 | 1 | 4 | 0.7500 | 0.8750 |
| controlled_alt_vs_proposed_raw | context_overlap | 10 | 2 | 0 | 0.8333 | 0.8333 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 4 | 2 | 6 | 0.5833 | 0.6667 |
| controlled_alt_vs_proposed_raw | persona_style | 1 | 4 | 7 | 0.3750 | 0.2000 |
| controlled_alt_vs_proposed_raw | distinct1 | 7 | 4 | 1 | 0.6250 | 0.6364 |
| controlled_alt_vs_proposed_raw | length_score | 2 | 10 | 0 | 0.1667 | 0.1667 |
| controlled_alt_vs_proposed_raw | sentence_score | 2 | 0 | 10 | 0.5833 | 1.0000 |
| controlled_alt_vs_proposed_raw | overall_quality | 8 | 4 | 0 | 0.6667 | 0.6667 |
| controlled_alt_vs_candidate_no_context | context_relevance | 10 | 2 | 0 | 0.8333 | 0.8333 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 7 | 5 | 0 | 0.5833 | 0.5833 |
| controlled_alt_vs_candidate_no_context | naturalness | 4 | 8 | 0 | 0.3333 | 0.3333 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 9 | 3 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 4 | 1 | 7 | 0.6250 | 0.8000 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 12 | 0.5000 | nan |
| controlled_alt_vs_candidate_no_context | objective_completion_support | 6 | 5 | 1 | 0.5417 | 0.5455 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 7 | 5 | 0 | 0.5833 | 0.5833 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | 6 | 5 | 1 | 0.5417 | 0.5455 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 8 | 1 | 3 | 0.7917 | 0.8889 |
| controlled_alt_vs_candidate_no_context | context_overlap | 9 | 3 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 5 | 3 | 4 | 0.5833 | 0.6250 |
| controlled_alt_vs_candidate_no_context | persona_style | 5 | 5 | 2 | 0.5000 | 0.5000 |
| controlled_alt_vs_candidate_no_context | distinct1 | 7 | 5 | 0 | 0.5833 | 0.5833 |
| controlled_alt_vs_candidate_no_context | length_score | 4 | 7 | 1 | 0.3750 | 0.3636 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0 | 0 | 12 | 0.5000 | nan |
| controlled_alt_vs_candidate_no_context | overall_quality | 9 | 3 | 0 | 0.7500 | 0.7500 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.3333 | 0.2500 | 0.7500 |
| proposed_contextual_controlled_tuned | 0.0000 | 0.0000 | 0.0833 | 0.5000 | 0.5000 |
| proposed_contextual | 0.0000 | 0.0000 | 0.3333 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.3333 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `3`
- Unique template signatures: `10`
- Template signature ratio: `0.8333`
- Effective sample size by source clustering: `2.67`
- Effective sample size by template-signature clustering: `8.00`
- Detailed diagnostics are published in `scenario_dependence.json`.

## Multi-turn Contradiction
| Arm | Contradiction Rate | Contradiction Safety | Contradicted Sources | Source Count |
|---|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 1.0000 | 0 | 3 |
| proposed_contextual_controlled_tuned | 0.0000 | 1.0000 | 0 | 3 |
| proposed_contextual | 0.0000 | 1.0000 | 0 | 3 |
| candidate_no_context | 0.0000 | 1.0000 | 0 | 3 |
- Detailed source-level values are published in `multi_turn_contradictions.json`.

- BERTScore status: unavailable (disabled_by_flag).

This report covers proposal RO5 metrics plus game-facing outcomes: quest-state correctness, lore consistency, contradiction safety, objective completion support, gameplay usefulness, and time-pressure acceptability.
# Proposal Alignment Evaluation Report

- Run ID: `20260313T071520Z`
- Generated: `2026-03-13T07:21:41.648243+00:00`
- Scenarios: `artifacts\proposal_control_tuning\auto_tune\20260313T051946Z\train_runs\trial_010\seed_19\20260313T071520Z\scenarios.jsonl`
- Scenario count: `24`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_tuned`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.1193 (0.0660, 0.1832) | 0.2662 (0.2111, 0.3265) | 0.8583 (0.8386, 0.8758) | 0.3181 (0.2773, 0.3633) | n/a |
| proposed_contextual_controlled_tuned | 0.0692 (0.0326, 0.1164) | 0.2656 (0.2157, 0.3172) | 0.8734 (0.8575, 0.8905) | 0.2978 (0.2687, 0.3283) | n/a |
| proposed_contextual | 0.0773 (0.0405, 0.1212) | 0.2642 (0.1976, 0.3324) | 0.8756 (0.8573, 0.8941) | 0.3013 (0.2654, 0.3381) | n/a |
| candidate_no_context | 0.0387 (0.0223, 0.0557) | 0.2777 (0.2072, 0.3580) | 0.8746 (0.8620, 0.8882) | 0.2882 (0.2560, 0.3242) | n/a |

## Game-facing Outcome Metrics (mean, 95% CI)
| Arm | Quest-state Correctness | Lore Consistency | Contradiction Safety | Objective Completion Support | Gameplay Usefulness | Time-pressure Acceptability |
|---|---:|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2002 (0.1521, 0.2534) | 0.0401 (0.0130, 0.0734) | 0.8333 (0.6667, 0.9583) | 0.0803 (0.0555, 0.1046) | 0.3045 (0.2910, 0.3192) | 0.2928 (0.2715, 0.3124) |
| proposed_contextual_controlled_tuned | 0.1593 (0.1277, 0.1978) | 0.0305 (0.0058, 0.0658) | 1.0000 (1.0000, 1.0000) | 0.0860 (0.0610, 0.1101) | 0.2997 (0.2823, 0.3165) | 0.3013 (0.2852, 0.3156) |
| proposed_contextual | 0.1668 (0.1349, 0.2059) | 0.0323 (0.0141, 0.0538) | 1.0000 (1.0000, 1.0000) | 0.0674 (0.0447, 0.0913) | 0.2955 (0.2806, 0.3101) | 0.2912 (0.2761, 0.3066) |
| candidate_no_context | 0.1284 (0.1155, 0.1431) | 0.0077 (0.0011, 0.0168) | 1.0000 (1.0000, 1.0000) | 0.0743 (0.0542, 0.0933) | 0.2833 (0.2728, 0.2926) | 0.2950 (0.2821, 0.3074) |

- Multi-turn contradiction rate is reported as `1 - contradiction_safety` in row-level outputs.

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0386 | 0.9993 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0134 | -0.0483 |
| proposed_vs_candidate_no_context | naturalness | 0.0011 | 0.0012 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.0384 | 0.2994 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0246 | 3.1874 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | -0.0069 | -0.0935 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 0.0122 | 0.0430 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | -0.0037 | -0.0126 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0458 | 1.0901 |
| proposed_vs_candidate_no_context | context_overlap | 0.0218 | 0.7094 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0109 | -0.0514 |
| proposed_vs_candidate_no_context | persona_style | -0.0234 | -0.0435 |
| proposed_vs_candidate_no_context | distinct1 | -0.0137 | -0.0145 |
| proposed_vs_candidate_no_context | length_score | 0.0472 | 0.0932 |
| proposed_vs_candidate_no_context | sentence_score | -0.0292 | -0.0300 |
| proposed_vs_candidate_no_context | overall_quality | 0.0131 | 0.0455 |
| controlled_vs_proposed_raw | context_relevance | 0.0420 | 0.5437 |
| controlled_vs_proposed_raw | persona_consistency | 0.0020 | 0.0075 |
| controlled_vs_proposed_raw | naturalness | -0.0173 | -0.0197 |
| controlled_vs_proposed_raw | quest_state_correctness | 0.0334 | 0.2002 |
| controlled_vs_proposed_raw | lore_consistency | 0.0078 | 0.2424 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | -0.1667 | -0.1667 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0130 | 0.1928 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0091 | 0.0307 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 0.0016 | 0.0054 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.0557 | 0.6336 |
| controlled_vs_proposed_raw | context_overlap | 0.0101 | 0.1926 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.0012 | 0.0059 |
| controlled_vs_proposed_raw | persona_style | 0.0052 | 0.0101 |
| controlled_vs_proposed_raw | distinct1 | -0.0106 | -0.0115 |
| controlled_vs_proposed_raw | length_score | -0.0750 | -0.1353 |
| controlled_vs_proposed_raw | sentence_score | 0.0437 | 0.0465 |
| controlled_vs_proposed_raw | overall_quality | 0.0168 | 0.0557 |
| controlled_vs_candidate_no_context | context_relevance | 0.0806 | 2.0862 |
| controlled_vs_candidate_no_context | persona_consistency | -0.0114 | -0.0411 |
| controlled_vs_candidate_no_context | naturalness | -0.0162 | -0.0186 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.0718 | 0.5596 |
| controlled_vs_candidate_no_context | lore_consistency | 0.0324 | 4.2023 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | -0.1667 | -0.1667 |
| controlled_vs_candidate_no_context | objective_completion_support | 0.0060 | 0.0813 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0212 | 0.0750 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | -0.0021 | -0.0072 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.1015 | 2.4144 |
| controlled_vs_candidate_no_context | context_overlap | 0.0319 | 1.0386 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | -0.0097 | -0.0458 |
| controlled_vs_candidate_no_context | persona_style | -0.0182 | -0.0338 |
| controlled_vs_candidate_no_context | distinct1 | -0.0243 | -0.0259 |
| controlled_vs_candidate_no_context | length_score | -0.0278 | -0.0548 |
| controlled_vs_candidate_no_context | sentence_score | 0.0146 | 0.0150 |
| controlled_vs_candidate_no_context | overall_quality | 0.0299 | 0.1037 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0501 | -0.4199 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0006 | -0.0022 |
| controlled_alt_vs_controlled_default | naturalness | 0.0151 | 0.0176 |
| controlled_alt_vs_controlled_default | quest_state_correctness | -0.0409 | -0.2041 |
| controlled_alt_vs_controlled_default | lore_consistency | -0.0096 | -0.2386 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0.1667 | 0.2000 |
| controlled_alt_vs_controlled_default | objective_completion_support | 0.0057 | 0.0709 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | -0.0049 | -0.0160 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | 0.0085 | 0.0291 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0663 | -0.4617 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0123 | -0.1963 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0022 | 0.0108 |
| controlled_alt_vs_controlled_default | persona_style | -0.0117 | -0.0225 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0145 | 0.0158 |
| controlled_alt_vs_controlled_default | length_score | 0.0417 | 0.0870 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0146 | -0.0148 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0204 | -0.0640 |
| controlled_alt_vs_proposed_raw | context_relevance | -0.0081 | -0.1045 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.0014 | 0.0053 |
| controlled_alt_vs_proposed_raw | naturalness | -0.0022 | -0.0025 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | -0.0075 | -0.0448 |
| controlled_alt_vs_proposed_raw | lore_consistency | -0.0017 | -0.0541 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_proposed_raw | objective_completion_support | 0.0187 | 0.2773 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 0.0042 | 0.0142 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | 0.0101 | 0.0347 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | -0.0106 | -0.1207 |
| controlled_alt_vs_proposed_raw | context_overlap | -0.0022 | -0.0414 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.0034 | 0.0167 |
| controlled_alt_vs_proposed_raw | persona_style | -0.0065 | -0.0126 |
| controlled_alt_vs_proposed_raw | distinct1 | 0.0039 | 0.0042 |
| controlled_alt_vs_proposed_raw | length_score | -0.0333 | -0.0602 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0292 | 0.0310 |
| controlled_alt_vs_proposed_raw | overall_quality | -0.0036 | -0.0119 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.0305 | 0.7903 |
| controlled_alt_vs_candidate_no_context | persona_consistency | -0.0120 | -0.0433 |
| controlled_alt_vs_candidate_no_context | naturalness | -0.0011 | -0.0013 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 0.0310 | 0.2413 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 0.0228 | 2.9609 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_candidate_no_context | objective_completion_support | 0.0117 | 0.1579 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 0.0164 | 0.0578 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | 0.0064 | 0.0216 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.0352 | 0.8378 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0196 | 0.6385 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | -0.0075 | -0.0355 |
| controlled_alt_vs_candidate_no_context | persona_style | -0.0299 | -0.0556 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0098 | -0.0104 |
| controlled_alt_vs_candidate_no_context | length_score | 0.0139 | 0.0274 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.0000 | 0.0000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.0095 | 0.0331 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0386 | (0.0053, 0.0781) | 0.0093 | 0.0386 | (-0.0021, 0.0900) | 0.0380 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0134 | (-0.0833, 0.0484) | 0.6517 | -0.0134 | (-0.0852, 0.0581) | 0.6913 |
| proposed_vs_candidate_no_context | naturalness | 0.0011 | (-0.0186, 0.0205) | 0.4477 | 0.0011 | (-0.0092, 0.0159) | 0.4807 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.0384 | (0.0086, 0.0723) | 0.0050 | 0.0384 | (-0.0000, 0.0929) | 0.0253 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0246 | (0.0072, 0.0460) | 0.0000 | 0.0246 | (0.0033, 0.0583) | 0.0007 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | -0.0069 | (-0.0278, 0.0135) | 0.7543 | -0.0069 | (-0.0221, 0.0100) | 0.8080 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 0.0122 | (-0.0056, 0.0313) | 0.1023 | 0.0122 | (0.0006, 0.0297) | 0.0197 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | -0.0037 | (-0.0205, 0.0120) | 0.6647 | -0.0037 | (-0.0128, 0.0038) | 0.8113 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0458 | (0.0008, 0.0989) | 0.0200 | 0.0458 | (-0.0032, 0.1148) | 0.0597 |
| proposed_vs_candidate_no_context | context_overlap | 0.0218 | (0.0076, 0.0380) | 0.0010 | 0.0218 | (0.0050, 0.0493) | 0.0000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0109 | (-0.0943, 0.0655) | 0.5990 | -0.0109 | (-0.1000, 0.0714) | 0.6337 |
| proposed_vs_candidate_no_context | persona_style | -0.0234 | (-0.0964, 0.0326) | 0.7330 | -0.0234 | (-0.0871, 0.0341) | 0.7443 |
| proposed_vs_candidate_no_context | distinct1 | -0.0137 | (-0.0315, 0.0048) | 0.9327 | -0.0137 | (-0.0372, 0.0021) | 0.9447 |
| proposed_vs_candidate_no_context | length_score | 0.0472 | (-0.0500, 0.1458) | 0.1687 | 0.0472 | (-0.0205, 0.1317) | 0.0773 |
| proposed_vs_candidate_no_context | sentence_score | -0.0292 | (-0.0875, 0.0292) | 0.9023 | -0.0292 | (-0.0636, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | overall_quality | 0.0131 | (-0.0184, 0.0444) | 0.1977 | 0.0131 | (-0.0060, 0.0495) | 0.2347 |
| controlled_vs_proposed_raw | context_relevance | 0.0420 | (-0.0219, 0.1152) | 0.1133 | 0.0420 | (-0.0370, 0.1228) | 0.1753 |
| controlled_vs_proposed_raw | persona_consistency | 0.0020 | (-0.0634, 0.0774) | 0.4890 | 0.0020 | (-0.0703, 0.0554) | 0.4520 |
| controlled_vs_proposed_raw | naturalness | -0.0173 | (-0.0341, -0.0028) | 0.9927 | -0.0173 | (-0.0296, -0.0042) | 0.9933 |
| controlled_vs_proposed_raw | quest_state_correctness | 0.0334 | (-0.0198, 0.0967) | 0.1307 | 0.0334 | (-0.0340, 0.0974) | 0.1810 |
| controlled_vs_proposed_raw | lore_consistency | 0.0078 | (-0.0231, 0.0475) | 0.3587 | 0.0078 | (-0.0411, 0.0602) | 0.3840 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | -0.1667 | (-0.3333, -0.0417) | 1.0000 | -0.1667 | (-0.5455, 0.0000) | 1.0000 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0130 | (-0.0016, 0.0300) | 0.0437 | 0.0130 | (-0.0044, 0.0390) | 0.1283 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0091 | (-0.0038, 0.0263) | 0.1143 | 0.0091 | (-0.0056, 0.0242) | 0.1377 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 0.0016 | (-0.0137, 0.0177) | 0.4050 | 0.0016 | (-0.0142, 0.0267) | 0.4640 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.0557 | (-0.0231, 0.1500) | 0.0970 | 0.0557 | (-0.0464, 0.1657) | 0.1867 |
| controlled_vs_proposed_raw | context_overlap | 0.0101 | (-0.0125, 0.0360) | 0.2043 | 0.0101 | (-0.0152, 0.0286) | 0.1933 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.0012 | (-0.0724, 0.0819) | 0.5043 | 0.0012 | (-0.0779, 0.0643) | 0.4403 |
| controlled_vs_proposed_raw | persona_style | 0.0052 | (-0.0573, 0.0951) | 0.5017 | 0.0052 | (-0.0682, 0.0670) | 0.4357 |
| controlled_vs_proposed_raw | distinct1 | -0.0106 | (-0.0294, 0.0072) | 0.8833 | -0.0106 | (-0.0294, 0.0097) | 0.8800 |
| controlled_vs_proposed_raw | length_score | -0.0750 | (-0.1569, -0.0125) | 0.9937 | -0.0750 | (-0.1567, 0.0042) | 0.9667 |
| controlled_vs_proposed_raw | sentence_score | 0.0437 | (0.0000, 0.0875) | 0.0330 | 0.0437 | (0.0135, 0.0795) | 0.0093 |
| controlled_vs_proposed_raw | overall_quality | 0.0168 | (-0.0316, 0.0723) | 0.2680 | 0.0168 | (-0.0335, 0.0608) | 0.2550 |
| controlled_vs_candidate_no_context | context_relevance | 0.0806 | (0.0243, 0.1446) | 0.0010 | 0.0806 | (0.0412, 0.1295) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | -0.0114 | (-0.0883, 0.0631) | 0.5970 | -0.0114 | (-0.1275, 0.0816) | 0.5740 |
| controlled_vs_candidate_no_context | naturalness | -0.0162 | (-0.0341, 0.0017) | 0.9610 | -0.0162 | (-0.0314, -0.0025) | 0.9903 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.0718 | (0.0248, 0.1275) | 0.0000 | 0.0718 | (0.0422, 0.1079) | 0.0000 |
| controlled_vs_candidate_no_context | lore_consistency | 0.0324 | (0.0039, 0.0657) | 0.0093 | 0.0324 | (0.0023, 0.0698) | 0.0100 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | -0.1667 | (-0.3333, -0.0417) | 1.0000 | -0.1667 | (-0.5455, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | 0.0060 | (-0.0126, 0.0261) | 0.2710 | 0.0060 | (-0.0118, 0.0311) | 0.2870 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0212 | (0.0055, 0.0383) | 0.0027 | 0.0212 | (0.0081, 0.0332) | 0.0000 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | -0.0021 | (-0.0203, 0.0169) | 0.5950 | -0.0021 | (-0.0123, 0.0145) | 0.6210 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.1015 | (0.0303, 0.1917) | 0.0023 | 0.1015 | (0.0455, 0.1657) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0319 | (0.0119, 0.0544) | 0.0000 | 0.0319 | (0.0193, 0.0449) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | -0.0097 | (-0.1032, 0.0830) | 0.5753 | -0.0097 | (-0.1439, 0.1001) | 0.5293 |
| controlled_vs_candidate_no_context | persona_style | -0.0182 | (-0.0534, 0.0117) | 0.8840 | -0.0182 | (-0.0391, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | distinct1 | -0.0243 | (-0.0427, -0.0081) | 0.9990 | -0.0243 | (-0.0453, -0.0096) | 1.0000 |
| controlled_vs_candidate_no_context | length_score | -0.0278 | (-0.1167, 0.0486) | 0.7323 | -0.0278 | (-0.1056, 0.0470) | 0.7623 |
| controlled_vs_candidate_no_context | sentence_score | 0.0146 | (0.0000, 0.0437) | 0.3530 | 0.0146 | (0.0000, 0.0477) | 0.3253 |
| controlled_vs_candidate_no_context | overall_quality | 0.0299 | (-0.0148, 0.0741) | 0.1010 | 0.0299 | (-0.0141, 0.0642) | 0.0843 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0501 | (-0.1129, 0.0182) | 0.9337 | -0.0501 | (-0.1193, 0.0131) | 0.9423 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0006 | (-0.0611, 0.0533) | 0.4927 | -0.0006 | (-0.0422, 0.0563) | 0.5447 |
| controlled_alt_vs_controlled_default | naturalness | 0.0151 | (-0.0068, 0.0414) | 0.0957 | 0.0151 | (-0.0007, 0.0264) | 0.0307 |
| controlled_alt_vs_controlled_default | quest_state_correctness | -0.0409 | (-0.0932, 0.0153) | 0.9223 | -0.0409 | (-0.0994, 0.0084) | 0.9717 |
| controlled_alt_vs_controlled_default | lore_consistency | -0.0096 | (-0.0443, 0.0302) | 0.7053 | -0.0096 | (-0.0503, 0.0281) | 0.6493 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0.1667 | (0.0417, 0.3333) | 0.0133 | 0.1667 | (0.0000, 0.5455) | 0.3247 |
| controlled_alt_vs_controlled_default | objective_completion_support | 0.0057 | (-0.0129, 0.0233) | 0.2633 | 0.0057 | (-0.0053, 0.0157) | 0.1620 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | -0.0049 | (-0.0186, 0.0103) | 0.7397 | -0.0049 | (-0.0150, 0.0075) | 0.7943 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | 0.0085 | (-0.0084, 0.0250) | 0.1597 | 0.0085 | (-0.0070, 0.0221) | 0.1567 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0663 | (-0.1458, 0.0189) | 0.9417 | -0.0663 | (-0.1556, 0.0140) | 0.9653 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0123 | (-0.0361, 0.0190) | 0.8140 | -0.0123 | (-0.0325, 0.0109) | 0.8637 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0022 | (-0.0720, 0.0702) | 0.4803 | 0.0022 | (-0.0500, 0.0702) | 0.4400 |
| controlled_alt_vs_controlled_default | persona_style | -0.0117 | (-0.0443, 0.0208) | 0.7750 | -0.0117 | (-0.0325, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0145 | (-0.0012, 0.0328) | 0.0353 | 0.0145 | (0.0044, 0.0252) | 0.0000 |
| controlled_alt_vs_controlled_default | length_score | 0.0417 | (-0.0458, 0.1500) | 0.1877 | 0.0417 | (-0.0318, 0.1090) | 0.1573 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0146 | (-0.0437, 0.0000) | 1.0000 | -0.0146 | (-0.0477, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0204 | (-0.0619, 0.0226) | 0.8277 | -0.0204 | (-0.0552, 0.0083) | 0.8553 |
| controlled_alt_vs_proposed_raw | context_relevance | -0.0081 | (-0.0569, 0.0489) | 0.6370 | -0.0081 | (-0.0712, 0.0430) | 0.5940 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.0014 | (-0.0643, 0.0711) | 0.4963 | 0.0014 | (-0.0449, 0.0469) | 0.4797 |
| controlled_alt_vs_proposed_raw | naturalness | -0.0022 | (-0.0245, 0.0240) | 0.5903 | -0.0022 | (-0.0189, 0.0097) | 0.6320 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | -0.0075 | (-0.0483, 0.0392) | 0.6417 | -0.0075 | (-0.0613, 0.0327) | 0.6317 |
| controlled_alt_vs_proposed_raw | lore_consistency | -0.0017 | (-0.0357, 0.0381) | 0.5673 | -0.0017 | (-0.0543, 0.0359) | 0.5543 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_proposed_raw | objective_completion_support | 0.0187 | (-0.0024, 0.0409) | 0.0477 | 0.0187 | (-0.0054, 0.0477) | 0.0627 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 0.0042 | (-0.0142, 0.0283) | 0.3890 | 0.0042 | (-0.0131, 0.0153) | 0.3067 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | 0.0101 | (-0.0022, 0.0217) | 0.0507 | 0.0101 | (-0.0013, 0.0248) | 0.0430 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | -0.0106 | (-0.0727, 0.0614) | 0.6497 | -0.0106 | (-0.0964, 0.0529) | 0.6083 |
| controlled_alt_vs_proposed_raw | context_overlap | -0.0022 | (-0.0245, 0.0259) | 0.5983 | -0.0022 | (-0.0314, 0.0260) | 0.5583 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.0034 | (-0.0740, 0.0841) | 0.4610 | 0.0034 | (-0.0487, 0.0639) | 0.4723 |
| controlled_alt_vs_proposed_raw | persona_style | -0.0065 | (-0.0690, 0.0820) | 0.6093 | -0.0065 | (-0.0682, 0.0368) | 0.6483 |
| controlled_alt_vs_proposed_raw | distinct1 | 0.0039 | (-0.0143, 0.0228) | 0.3437 | 0.0039 | (-0.0134, 0.0198) | 0.2727 |
| controlled_alt_vs_proposed_raw | length_score | -0.0333 | (-0.1375, 0.0833) | 0.7400 | -0.0333 | (-0.1117, 0.0226) | 0.8810 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0292 | (-0.0292, 0.0875) | 0.2327 | 0.0292 | (-0.0318, 0.0795) | 0.2070 |
| controlled_alt_vs_proposed_raw | overall_quality | -0.0036 | (-0.0458, 0.0435) | 0.5720 | -0.0036 | (-0.0416, 0.0261) | 0.6007 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.0305 | (-0.0100, 0.0827) | 0.0850 | 0.0305 | (0.0037, 0.0575) | 0.0050 |
| controlled_alt_vs_candidate_no_context | persona_consistency | -0.0120 | (-0.0872, 0.0632) | 0.6170 | -0.0120 | (-0.1214, 0.1063) | 0.6240 |
| controlled_alt_vs_candidate_no_context | naturalness | -0.0011 | (-0.0195, 0.0162) | 0.5537 | -0.0011 | (-0.0203, 0.0117) | 0.5537 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 0.0310 | (-0.0044, 0.0755) | 0.0397 | 0.0310 | (0.0085, 0.0513) | 0.0000 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 0.0228 | (-0.0045, 0.0573) | 0.0713 | 0.0228 | (-0.0033, 0.0460) | 0.0413 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_candidate_no_context | objective_completion_support | 0.0117 | (-0.0123, 0.0368) | 0.1720 | 0.0117 | (-0.0108, 0.0432) | 0.2087 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 0.0164 | (-0.0019, 0.0357) | 0.0420 | 0.0164 | (0.0145, 0.0189) | 0.0000 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | 0.0064 | (-0.0112, 0.0217) | 0.2190 | 0.0064 | (-0.0025, 0.0148) | 0.0857 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.0352 | (-0.0178, 0.0996) | 0.1173 | 0.0352 | (-0.0029, 0.0682) | 0.0270 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0196 | (0.0000, 0.0446) | 0.0250 | 0.0196 | (0.0054, 0.0333) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | -0.0075 | (-0.1012, 0.0847) | 0.5490 | -0.0075 | (-0.1407, 0.1304) | 0.5930 |
| controlled_alt_vs_candidate_no_context | persona_style | -0.0299 | (-0.0755, 0.0156) | 0.9107 | -0.0299 | (-0.0637, 0.0000) | 1.0000 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0098 | (-0.0244, 0.0045) | 0.9010 | -0.0098 | (-0.0276, 0.0029) | 0.9083 |
| controlled_alt_vs_candidate_no_context | length_score | 0.0139 | (-0.0653, 0.0861) | 0.3580 | 0.0139 | (-0.0409, 0.0597) | 0.3237 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.0000 | (-0.0437, 0.0437) | 0.6627 | 0.0000 | (-0.0437, 0.0437) | 0.6423 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.0095 | (-0.0281, 0.0495) | 0.3347 | 0.0095 | (-0.0353, 0.0503) | 0.3160 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 11 | 5 | 8 | 0.6250 | 0.6875 |
| proposed_vs_candidate_no_context | persona_consistency | 6 | 6 | 12 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | naturalness | 10 | 6 | 8 | 0.5833 | 0.6250 |
| proposed_vs_candidate_no_context | quest_state_correctness | 12 | 4 | 8 | 0.6667 | 0.7500 |
| proposed_vs_candidate_no_context | lore_consistency | 8 | 1 | 15 | 0.6458 | 0.8889 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| proposed_vs_candidate_no_context | objective_completion_support | 6 | 9 | 9 | 0.4375 | 0.4000 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 10 | 6 | 8 | 0.5833 | 0.6250 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | 8 | 7 | 9 | 0.5208 | 0.5333 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 6 | 1 | 17 | 0.6042 | 0.8571 |
| proposed_vs_candidate_no_context | context_overlap | 12 | 4 | 8 | 0.6667 | 0.7500 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 6 | 5 | 13 | 0.5208 | 0.5455 |
| proposed_vs_candidate_no_context | persona_style | 2 | 3 | 19 | 0.4792 | 0.4000 |
| proposed_vs_candidate_no_context | distinct1 | 4 | 10 | 10 | 0.3750 | 0.2857 |
| proposed_vs_candidate_no_context | length_score | 10 | 6 | 8 | 0.5833 | 0.6250 |
| proposed_vs_candidate_no_context | sentence_score | 1 | 3 | 20 | 0.4583 | 0.2500 |
| proposed_vs_candidate_no_context | overall_quality | 10 | 6 | 8 | 0.5833 | 0.6250 |
| controlled_vs_proposed_raw | context_relevance | 8 | 10 | 6 | 0.4583 | 0.4444 |
| controlled_vs_proposed_raw | persona_consistency | 6 | 8 | 10 | 0.4583 | 0.4286 |
| controlled_vs_proposed_raw | naturalness | 5 | 13 | 6 | 0.3333 | 0.2778 |
| controlled_vs_proposed_raw | quest_state_correctness | 7 | 11 | 6 | 0.4167 | 0.3889 |
| controlled_vs_proposed_raw | lore_consistency | 4 | 7 | 13 | 0.4375 | 0.3636 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0 | 4 | 20 | 0.4167 | 0.0000 |
| controlled_vs_proposed_raw | objective_completion_support | 6 | 2 | 16 | 0.5833 | 0.7500 |
| controlled_vs_proposed_raw | gameplay_usefulness | 9 | 9 | 6 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 5 | 6 | 13 | 0.4792 | 0.4545 |
| controlled_vs_proposed_raw | context_keyword_coverage | 6 | 6 | 12 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | context_overlap | 9 | 9 | 6 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 5 | 7 | 12 | 0.4583 | 0.4167 |
| controlled_vs_proposed_raw | persona_style | 2 | 6 | 16 | 0.4167 | 0.2500 |
| controlled_vs_proposed_raw | distinct1 | 8 | 9 | 7 | 0.4792 | 0.4706 |
| controlled_vs_proposed_raw | length_score | 3 | 13 | 8 | 0.2917 | 0.1875 |
| controlled_vs_proposed_raw | sentence_score | 3 | 0 | 21 | 0.5625 | 1.0000 |
| controlled_vs_proposed_raw | overall_quality | 8 | 10 | 6 | 0.4583 | 0.4444 |
| controlled_vs_candidate_no_context | context_relevance | 12 | 6 | 6 | 0.6250 | 0.6667 |
| controlled_vs_candidate_no_context | persona_consistency | 9 | 6 | 9 | 0.5625 | 0.6000 |
| controlled_vs_candidate_no_context | naturalness | 8 | 10 | 6 | 0.4583 | 0.4444 |
| controlled_vs_candidate_no_context | quest_state_correctness | 13 | 5 | 6 | 0.6667 | 0.7222 |
| controlled_vs_candidate_no_context | lore_consistency | 8 | 2 | 14 | 0.6250 | 0.8000 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 4 | 20 | 0.4167 | 0.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | 7 | 6 | 11 | 0.5208 | 0.5385 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 13 | 5 | 6 | 0.6667 | 0.7222 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 6 | 9 | 9 | 0.4375 | 0.4000 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 9 | 2 | 13 | 0.6458 | 0.8182 |
| controlled_vs_candidate_no_context | context_overlap | 13 | 5 | 6 | 0.6667 | 0.7222 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 7 | 6 | 11 | 0.5208 | 0.5385 |
| controlled_vs_candidate_no_context | persona_style | 2 | 4 | 18 | 0.4583 | 0.3333 |
| controlled_vs_candidate_no_context | distinct1 | 5 | 12 | 7 | 0.3542 | 0.2941 |
| controlled_vs_candidate_no_context | length_score | 8 | 7 | 9 | 0.5208 | 0.5333 |
| controlled_vs_candidate_no_context | sentence_score | 1 | 0 | 23 | 0.5208 | 1.0000 |
| controlled_vs_candidate_no_context | overall_quality | 13 | 5 | 6 | 0.6667 | 0.7222 |
| controlled_alt_vs_controlled_default | context_relevance | 2 | 10 | 12 | 0.3333 | 0.1667 |
| controlled_alt_vs_controlled_default | persona_consistency | 6 | 5 | 13 | 0.5208 | 0.5455 |
| controlled_alt_vs_controlled_default | naturalness | 8 | 4 | 12 | 0.5833 | 0.6667 |
| controlled_alt_vs_controlled_default | quest_state_correctness | 2 | 10 | 12 | 0.3333 | 0.1667 |
| controlled_alt_vs_controlled_default | lore_consistency | 2 | 7 | 15 | 0.3958 | 0.2222 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 4 | 0 | 20 | 0.5833 | 1.0000 |
| controlled_alt_vs_controlled_default | objective_completion_support | 7 | 3 | 14 | 0.5833 | 0.7000 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | 4 | 8 | 12 | 0.4167 | 0.3333 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | 7 | 4 | 13 | 0.5625 | 0.6364 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 2 | 9 | 13 | 0.3542 | 0.1818 |
| controlled_alt_vs_controlled_default | context_overlap | 2 | 10 | 12 | 0.3333 | 0.1667 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 6 | 5 | 13 | 0.5208 | 0.5455 |
| controlled_alt_vs_controlled_default | persona_style | 1 | 3 | 20 | 0.4583 | 0.2500 |
| controlled_alt_vs_controlled_default | distinct1 | 7 | 4 | 13 | 0.5625 | 0.6364 |
| controlled_alt_vs_controlled_default | length_score | 6 | 4 | 14 | 0.5417 | 0.6000 |
| controlled_alt_vs_controlled_default | sentence_score | 0 | 1 | 23 | 0.4792 | 0.0000 |
| controlled_alt_vs_controlled_default | overall_quality | 4 | 8 | 12 | 0.4167 | 0.3333 |
| controlled_alt_vs_proposed_raw | context_relevance | 6 | 12 | 6 | 0.3750 | 0.3333 |
| controlled_alt_vs_proposed_raw | persona_consistency | 6 | 9 | 9 | 0.4375 | 0.4000 |
| controlled_alt_vs_proposed_raw | naturalness | 6 | 12 | 6 | 0.3750 | 0.3333 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 5 | 13 | 6 | 0.3333 | 0.2778 |
| controlled_alt_vs_proposed_raw | lore_consistency | 3 | 8 | 13 | 0.3958 | 0.2727 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_alt_vs_proposed_raw | objective_completion_support | 10 | 3 | 11 | 0.6458 | 0.7692 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 7 | 11 | 6 | 0.4167 | 0.3889 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | 9 | 4 | 11 | 0.6042 | 0.6923 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 3 | 8 | 13 | 0.3958 | 0.2727 |
| controlled_alt_vs_proposed_raw | context_overlap | 6 | 12 | 6 | 0.3750 | 0.3333 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 6 | 8 | 10 | 0.4583 | 0.4286 |
| controlled_alt_vs_proposed_raw | persona_style | 1 | 7 | 16 | 0.3750 | 0.1250 |
| controlled_alt_vs_proposed_raw | distinct1 | 11 | 5 | 8 | 0.6250 | 0.6875 |
| controlled_alt_vs_proposed_raw | length_score | 6 | 12 | 6 | 0.3750 | 0.3333 |
| controlled_alt_vs_proposed_raw | sentence_score | 3 | 1 | 20 | 0.5417 | 0.7500 |
| controlled_alt_vs_proposed_raw | overall_quality | 6 | 12 | 6 | 0.3750 | 0.3333 |
| controlled_alt_vs_candidate_no_context | context_relevance | 11 | 7 | 6 | 0.5833 | 0.6111 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 8 | 9 | 7 | 0.4792 | 0.4706 |
| controlled_alt_vs_candidate_no_context | naturalness | 9 | 9 | 6 | 0.5000 | 0.5000 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 11 | 7 | 6 | 0.5833 | 0.6111 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 5 | 4 | 15 | 0.5208 | 0.5556 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_alt_vs_candidate_no_context | objective_completion_support | 8 | 6 | 10 | 0.5417 | 0.5714 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 12 | 6 | 6 | 0.6250 | 0.6667 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | 9 | 7 | 8 | 0.5417 | 0.5625 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 5 | 3 | 16 | 0.5417 | 0.6250 |
| controlled_alt_vs_candidate_no_context | context_overlap | 11 | 7 | 6 | 0.5833 | 0.6111 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 7 | 8 | 9 | 0.4792 | 0.4667 |
| controlled_alt_vs_candidate_no_context | persona_style | 3 | 6 | 15 | 0.4375 | 0.3333 |
| controlled_alt_vs_candidate_no_context | distinct1 | 6 | 10 | 8 | 0.4167 | 0.3750 |
| controlled_alt_vs_candidate_no_context | length_score | 10 | 6 | 8 | 0.5833 | 0.6250 |
| controlled_alt_vs_candidate_no_context | sentence_score | 1 | 1 | 22 | 0.5000 | 0.5000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 12 | 6 | 6 | 0.6250 | 0.6667 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.1250 | 0.2500 | 0.7500 |
| proposed_contextual_controlled_tuned | 0.0000 | 0.0000 | 0.1667 | 0.1250 | 0.8750 |
| proposed_contextual | 0.0000 | 0.0000 | 0.5417 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.4167 | 0.0000 | 0.0000 |

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
| proposed_contextual_controlled | 0.2000 | 0.8000 | 1 | 5 |
| proposed_contextual_controlled_tuned | 0.0000 | 1.0000 | 0 | 5 |
| proposed_contextual | 0.0000 | 1.0000 | 0 | 5 |
| candidate_no_context | 0.0000 | 1.0000 | 0 | 5 |
- Detailed source-level values are published in `multi_turn_contradictions.json`.

- BERTScore status: unavailable (disabled_by_flag).

This report covers proposal RO5 metrics plus game-facing outcomes: quest-state correctness, lore consistency, contradiction safety, objective completion support, gameplay usefulness, and time-pressure acceptability.
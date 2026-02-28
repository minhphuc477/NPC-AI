# Proposal Alignment Evaluation Report

- Run ID: `20260227T173833Z`
- Generated: `2026-02-27T17:39:08.969236+00:00`
- Scenarios: `artifacts\proposal\20260227T173833Z\scenarios.jsonl`
- Scenario count: `112`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off
- `baseline_no_context`: model `phi3:mini`, dynamic_context=off, response_control=off
- `baseline_no_context_phi3_latest`: model `phi3:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2895 (0.2717, 0.3085) | 0.2475 (0.2161, 0.2802) | 0.8872 (0.8761, 0.8971) | 0.3643 (0.3511, 0.3774) | 0.1095 |
| proposed_contextual | 0.0639 (0.0486, 0.0808) | 0.1601 (0.1349, 0.1890) | 0.8071 (0.7942, 0.8194) | 0.2255 (0.2100, 0.2406) | 0.0685 |
| candidate_no_context | 0.0244 (0.0192, 0.0306) | 0.1458 (0.1234, 0.1706) | 0.8069 (0.7927, 0.8219) | 0.2017 (0.1916, 0.2124) | 0.0446 |
| baseline_no_context | 0.0475 (0.0388, 0.0565) | 0.1814 (0.1592, 0.2074) | 0.8792 (0.8689, 0.8891) | 0.2354 (0.2256, 0.2451) | 0.0493 |
| baseline_no_context_phi3_latest | 0.0456 (0.0365, 0.0548) | 0.1632 (0.1448, 0.1811) | 0.8884 (0.8770, 0.8998) | 0.2311 (0.2236, 0.2388) | 0.0545 |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0395 | 1.6154 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0143 | 0.0980 |
| proposed_vs_candidate_no_context | naturalness | 0.0002 | 0.0003 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0519 | 2.9357 |
| proposed_vs_candidate_no_context | context_overlap | 0.0105 | 0.2613 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0209 | 0.3565 |
| proposed_vs_candidate_no_context | persona_style | -0.0122 | -0.0247 |
| proposed_vs_candidate_no_context | distinct1 | 0.0040 | 0.0042 |
| proposed_vs_candidate_no_context | length_score | 0.0012 | 0.0041 |
| proposed_vs_candidate_no_context | sentence_score | -0.0161 | -0.0214 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0240 | 0.5374 |
| proposed_vs_candidate_no_context | overall_quality | 0.0238 | 0.1181 |
| proposed_vs_baseline_no_context | context_relevance | 0.0164 | 0.3455 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0213 | -0.1172 |
| proposed_vs_baseline_no_context | naturalness | -0.0721 | -0.0820 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0193 | 0.3832 |
| proposed_vs_baseline_no_context | context_overlap | 0.0097 | 0.2375 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 0.0003 | 0.0043 |
| proposed_vs_baseline_no_context | persona_style | -0.1077 | -0.1825 |
| proposed_vs_baseline_no_context | distinct1 | -0.0426 | -0.0434 |
| proposed_vs_baseline_no_context | length_score | -0.2128 | -0.4211 |
| proposed_vs_baseline_no_context | sentence_score | -0.1250 | -0.1455 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0192 | 0.3897 |
| proposed_vs_baseline_no_context | overall_quality | -0.0099 | -0.0422 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 0.0183 | 0.4014 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | -0.0031 | -0.0191 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | -0.0813 | -0.0915 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.0212 | 0.4389 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 0.0115 | 0.2937 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.0203 | 0.3429 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | -0.0968 | -0.1672 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | -0.0433 | -0.0441 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | -0.2479 | -0.4587 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | -0.1442 | -0.1641 |
| proposed_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0140 | 0.2573 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | -0.0056 | -0.0242 |
| controlled_vs_proposed_raw | context_relevance | 0.2256 | 3.5279 |
| controlled_vs_proposed_raw | persona_consistency | 0.0874 | 0.5456 |
| controlled_vs_proposed_raw | naturalness | 0.0801 | 0.0992 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.3008 | 4.3229 |
| controlled_vs_proposed_raw | context_overlap | 0.0499 | 0.9837 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.0843 | 1.0588 |
| controlled_vs_proposed_raw | persona_style | 0.0998 | 0.2069 |
| controlled_vs_proposed_raw | distinct1 | 0.0010 | 0.0010 |
| controlled_vs_proposed_raw | length_score | 0.2693 | 0.9207 |
| controlled_vs_proposed_raw | sentence_score | 0.2594 | 0.3532 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0409 | 0.5973 |
| controlled_vs_proposed_raw | overall_quality | 0.1389 | 0.6159 |
| controlled_vs_candidate_no_context | context_relevance | 0.2650 | 10.8420 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1017 | 0.6971 |
| controlled_vs_candidate_no_context | naturalness | 0.0803 | 0.0995 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3527 | 19.9495 |
| controlled_vs_candidate_no_context | context_overlap | 0.0604 | 1.5020 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1052 | 1.7928 |
| controlled_vs_candidate_no_context | persona_style | 0.0876 | 0.1771 |
| controlled_vs_candidate_no_context | distinct1 | 0.0049 | 0.0053 |
| controlled_vs_candidate_no_context | length_score | 0.2705 | 0.9285 |
| controlled_vs_candidate_no_context | sentence_score | 0.2433 | 0.3242 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0649 | 1.4556 |
| controlled_vs_candidate_no_context | overall_quality | 0.1627 | 0.8067 |
| controlled_vs_baseline_no_context | context_relevance | 0.2420 | 5.0921 |
| controlled_vs_baseline_no_context | persona_consistency | 0.0661 | 0.3645 |
| controlled_vs_baseline_no_context | naturalness | 0.0080 | 0.0091 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3201 | 6.3625 |
| controlled_vs_baseline_no_context | context_overlap | 0.0597 | 1.4549 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.0846 | 1.0676 |
| controlled_vs_baseline_no_context | persona_style | -0.0079 | -0.0133 |
| controlled_vs_baseline_no_context | distinct1 | -0.0416 | -0.0424 |
| controlled_vs_baseline_no_context | length_score | 0.0565 | 0.1119 |
| controlled_vs_baseline_no_context | sentence_score | 0.1344 | 0.1564 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0601 | 1.2197 |
| controlled_vs_baseline_no_context | overall_quality | 0.1289 | 0.5477 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.2439 | 5.3454 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.0843 | 0.5162 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | -0.0012 | -0.0014 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.3221 | 6.6590 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0614 | 1.5662 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1046 | 1.7647 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | 0.0029 | 0.0051 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0423 | -0.0431 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | 0.0214 | 0.0396 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.1152 | 0.1311 |
| controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0550 | 1.0082 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1333 | 0.5768 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2420 | 5.0921 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.0661 | 0.3645 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 0.0080 | 0.0091 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3201 | 6.3625 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0597 | 1.4549 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.0846 | 1.0676 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | -0.0079 | -0.0133 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0416 | -0.0424 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.0565 | 0.1119 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.1344 | 0.1564 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0601 | 1.2197 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1289 | 0.5477 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.2439 | 5.3454 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.0843 | 0.5162 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | -0.0012 | -0.0014 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.3221 | 6.6590 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0614 | 1.5662 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1046 | 1.7647 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | 0.0029 | 0.0051 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0423 | -0.0431 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | 0.0214 | 0.0396 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.1152 | 0.1311 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0550 | 1.0082 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1333 | 0.5768 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) |
|---|---|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0395 | (0.0231, 0.0574) | 0.0000 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0143 | (-0.0146, 0.0435) | 0.1700 |
| proposed_vs_candidate_no_context | naturalness | 0.0002 | (-0.0132, 0.0141) | 0.5070 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0519 | (0.0299, 0.0760) | 0.0000 |
| proposed_vs_candidate_no_context | context_overlap | 0.0105 | (0.0045, 0.0171) | 0.0010 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0209 | (-0.0130, 0.0588) | 0.1143 |
| proposed_vs_candidate_no_context | persona_style | -0.0122 | (-0.0335, 0.0071) | 0.8907 |
| proposed_vs_candidate_no_context | distinct1 | 0.0040 | (-0.0033, 0.0113) | 0.1363 |
| proposed_vs_candidate_no_context | length_score | 0.0012 | (-0.0515, 0.0554) | 0.4940 |
| proposed_vs_candidate_no_context | sentence_score | -0.0161 | (-0.0469, 0.0152) | 0.8607 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0240 | (0.0132, 0.0351) | 0.0000 |
| proposed_vs_candidate_no_context | overall_quality | 0.0238 | (0.0084, 0.0398) | 0.0023 |
| proposed_vs_baseline_no_context | context_relevance | 0.0164 | (-0.0032, 0.0367) | 0.0500 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0213 | (-0.0566, 0.0138) | 0.8753 |
| proposed_vs_baseline_no_context | naturalness | -0.0721 | (-0.0902, -0.0536) | 1.0000 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0193 | (-0.0058, 0.0452) | 0.0683 |
| proposed_vs_baseline_no_context | context_overlap | 0.0097 | (0.0029, 0.0167) | 0.0017 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 0.0003 | (-0.0429, 0.0411) | 0.5190 |
| proposed_vs_baseline_no_context | persona_style | -0.1077 | (-0.1515, -0.0635) | 1.0000 |
| proposed_vs_baseline_no_context | distinct1 | -0.0426 | (-0.0514, -0.0337) | 1.0000 |
| proposed_vs_baseline_no_context | length_score | -0.2128 | (-0.2795, -0.1473) | 1.0000 |
| proposed_vs_baseline_no_context | sentence_score | -0.1250 | (-0.1687, -0.0781) | 1.0000 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0192 | (0.0052, 0.0328) | 0.0027 |
| proposed_vs_baseline_no_context | overall_quality | -0.0099 | (-0.0283, 0.0087) | 0.8503 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 0.0183 | (0.0007, 0.0368) | 0.0203 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | -0.0031 | (-0.0333, 0.0275) | 0.5857 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | -0.0813 | (-0.0977, -0.0636) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.0212 | (-0.0014, 0.0458) | 0.0357 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 0.0115 | (0.0049, 0.0186) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.0203 | (-0.0130, 0.0568) | 0.1400 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | -0.0968 | (-0.1414, -0.0543) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | -0.0433 | (-0.0511, -0.0357) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | -0.2479 | (-0.3128, -0.1833) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | -0.1442 | (-0.1875, -0.1018) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0140 | (-0.0001, 0.0277) | 0.0257 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | -0.0056 | (-0.0225, 0.0120) | 0.7477 |
| controlled_vs_proposed_raw | context_relevance | 0.2256 | (0.2021, 0.2504) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.0874 | (0.0501, 0.1230) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0801 | (0.0642, 0.0955) | 0.0000 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.3008 | (0.2707, 0.3332) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0499 | (0.0416, 0.0586) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.0843 | (0.0414, 0.1264) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.0998 | (0.0563, 0.1427) | 0.0000 |
| controlled_vs_proposed_raw | distinct1 | 0.0010 | (-0.0064, 0.0085) | 0.3847 |
| controlled_vs_proposed_raw | length_score | 0.2693 | (0.2065, 0.3316) | 0.0000 |
| controlled_vs_proposed_raw | sentence_score | 0.2594 | (0.2312, 0.2875) | 0.0000 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0409 | (0.0267, 0.0559) | 0.0000 |
| controlled_vs_proposed_raw | overall_quality | 0.1389 | (0.1207, 0.1570) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2650 | (0.2479, 0.2839) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1017 | (0.0675, 0.1352) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.0803 | (0.0634, 0.0969) | 0.0000 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3527 | (0.3287, 0.3772) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0604 | (0.0534, 0.0677) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1052 | (0.0667, 0.1464) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.0876 | (0.0472, 0.1294) | 0.0000 |
| controlled_vs_candidate_no_context | distinct1 | 0.0049 | (-0.0026, 0.0124) | 0.1003 |
| controlled_vs_candidate_no_context | length_score | 0.2705 | (0.2012, 0.3369) | 0.0000 |
| controlled_vs_candidate_no_context | sentence_score | 0.2433 | (0.2125, 0.2719) | 0.0000 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0649 | (0.0521, 0.0778) | 0.0000 |
| controlled_vs_candidate_no_context | overall_quality | 0.1627 | (0.1480, 0.1781) | 0.0000 |
| controlled_vs_baseline_no_context | context_relevance | 0.2420 | (0.2238, 0.2606) | 0.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 0.0661 | (0.0327, 0.1002) | 0.0000 |
| controlled_vs_baseline_no_context | naturalness | 0.0080 | (-0.0059, 0.0216) | 0.1197 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3201 | (0.2966, 0.3443) | 0.0000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0597 | (0.0523, 0.0673) | 0.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.0846 | (0.0415, 0.1275) | 0.0000 |
| controlled_vs_baseline_no_context | persona_style | -0.0079 | (-0.0276, 0.0129) | 0.7863 |
| controlled_vs_baseline_no_context | distinct1 | -0.0416 | (-0.0479, -0.0347) | 1.0000 |
| controlled_vs_baseline_no_context | length_score | 0.0565 | (-0.0092, 0.1217) | 0.0483 |
| controlled_vs_baseline_no_context | sentence_score | 0.1344 | (0.1031, 0.1687) | 0.0000 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0601 | (0.0471, 0.0728) | 0.0000 |
| controlled_vs_baseline_no_context | overall_quality | 0.1289 | (0.1145, 0.1430) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.2439 | (0.2254, 0.2627) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.0843 | (0.0562, 0.1154) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | -0.0012 | (-0.0156, 0.0125) | 0.5803 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.3221 | (0.2976, 0.3478) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0614 | (0.0542, 0.0687) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1046 | (0.0688, 0.1405) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | 0.0029 | (-0.0190, 0.0248) | 0.3997 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0423 | (-0.0484, -0.0361) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | 0.0214 | (-0.0458, 0.0884) | 0.2383 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.1152 | (0.0834, 0.1469) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0550 | (0.0403, 0.0688) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1333 | (0.1199, 0.1479) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2420 | (0.2239, 0.2610) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.0661 | (0.0315, 0.1005) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 0.0080 | (-0.0064, 0.0219) | 0.1407 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3201 | (0.2968, 0.3451) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0597 | (0.0525, 0.0671) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.0846 | (0.0411, 0.1282) | 0.0003 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | -0.0079 | (-0.0283, 0.0130) | 0.7827 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0416 | (-0.0483, -0.0349) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.0565 | (-0.0083, 0.1217) | 0.0447 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.1344 | (0.1031, 0.1656) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0601 | (0.0476, 0.0731) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1289 | (0.1141, 0.1429) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.2439 | (0.2245, 0.2619) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.0843 | (0.0548, 0.1159) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | -0.0012 | (-0.0163, 0.0129) | 0.5690 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.3221 | (0.2971, 0.3471) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0614 | (0.0543, 0.0686) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1046 | (0.0673, 0.1426) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | 0.0029 | (-0.0190, 0.0258) | 0.3927 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0423 | (-0.0482, -0.0359) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | 0.0214 | (-0.0479, 0.0863) | 0.2540 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.1152 | (0.0839, 0.1487) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0550 | (0.0405, 0.0692) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1333 | (0.1202, 0.1473) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 37 | 26 | 49 | 0.5491 | 0.5873 |
| proposed_vs_candidate_no_context | persona_consistency | 22 | 15 | 75 | 0.5312 | 0.5946 |
| proposed_vs_candidate_no_context | naturalness | 29 | 34 | 49 | 0.4777 | 0.4603 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 31 | 9 | 72 | 0.5982 | 0.7750 |
| proposed_vs_candidate_no_context | context_overlap | 36 | 27 | 49 | 0.5402 | 0.5714 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 20 | 10 | 82 | 0.5446 | 0.6667 |
| proposed_vs_candidate_no_context | persona_style | 4 | 8 | 100 | 0.4821 | 0.3333 |
| proposed_vs_candidate_no_context | distinct1 | 31 | 26 | 55 | 0.5223 | 0.5439 |
| proposed_vs_candidate_no_context | length_score | 29 | 34 | 49 | 0.4777 | 0.4603 |
| proposed_vs_candidate_no_context | sentence_score | 11 | 16 | 85 | 0.4777 | 0.4074 |
| proposed_vs_candidate_no_context | bertscore_f1 | 52 | 30 | 30 | 0.5982 | 0.6341 |
| proposed_vs_candidate_no_context | overall_quality | 53 | 29 | 30 | 0.6071 | 0.6463 |
| proposed_vs_baseline_no_context | context_relevance | 51 | 60 | 1 | 0.4598 | 0.4595 |
| proposed_vs_baseline_no_context | persona_consistency | 25 | 54 | 33 | 0.3705 | 0.3165 |
| proposed_vs_baseline_no_context | naturalness | 30 | 82 | 0 | 0.2679 | 0.2679 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 30 | 34 | 48 | 0.4821 | 0.4688 |
| proposed_vs_baseline_no_context | context_overlap | 59 | 52 | 1 | 0.5312 | 0.5315 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 23 | 31 | 58 | 0.4643 | 0.4259 |
| proposed_vs_baseline_no_context | persona_style | 5 | 31 | 76 | 0.3839 | 0.1389 |
| proposed_vs_baseline_no_context | distinct1 | 20 | 82 | 10 | 0.2232 | 0.1961 |
| proposed_vs_baseline_no_context | length_score | 29 | 81 | 2 | 0.2679 | 0.2636 |
| proposed_vs_baseline_no_context | sentence_score | 12 | 52 | 48 | 0.3214 | 0.1875 |
| proposed_vs_baseline_no_context | bertscore_f1 | 68 | 44 | 0 | 0.6071 | 0.6071 |
| proposed_vs_baseline_no_context | overall_quality | 38 | 74 | 0 | 0.3393 | 0.3393 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 57 | 55 | 0 | 0.5089 | 0.5089 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | 27 | 52 | 33 | 0.3884 | 0.3418 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | 26 | 85 | 1 | 0.2366 | 0.2342 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 30 | 28 | 54 | 0.5089 | 0.5172 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 69 | 42 | 1 | 0.6205 | 0.6216 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 23 | 29 | 60 | 0.4732 | 0.4423 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | 6 | 30 | 76 | 0.3929 | 0.1667 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | 17 | 83 | 12 | 0.2054 | 0.1700 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | 23 | 83 | 6 | 0.2321 | 0.2170 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | 10 | 56 | 46 | 0.2946 | 0.1515 |
| proposed_vs_baseline_no_context_phi3_latest | bertscore_f1 | 66 | 46 | 0 | 0.5893 | 0.5893 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | 39 | 73 | 0 | 0.3482 | 0.3482 |
| controlled_vs_proposed_raw | context_relevance | 105 | 7 | 0 | 0.9375 | 0.9375 |
| controlled_vs_proposed_raw | persona_consistency | 67 | 14 | 31 | 0.7366 | 0.8272 |
| controlled_vs_proposed_raw | naturalness | 92 | 20 | 0 | 0.8214 | 0.8214 |
| controlled_vs_proposed_raw | context_keyword_coverage | 104 | 3 | 5 | 0.9509 | 0.9720 |
| controlled_vs_proposed_raw | context_overlap | 95 | 16 | 1 | 0.8527 | 0.8559 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 49 | 14 | 49 | 0.6562 | 0.7778 |
| controlled_vs_proposed_raw | persona_style | 29 | 8 | 75 | 0.5938 | 0.7838 |
| controlled_vs_proposed_raw | distinct1 | 66 | 45 | 1 | 0.5938 | 0.5946 |
| controlled_vs_proposed_raw | length_score | 79 | 29 | 4 | 0.7232 | 0.7315 |
| controlled_vs_proposed_raw | sentence_score | 83 | 0 | 29 | 0.8705 | 1.0000 |
| controlled_vs_proposed_raw | bertscore_f1 | 82 | 30 | 0 | 0.7321 | 0.7321 |
| controlled_vs_proposed_raw | overall_quality | 99 | 13 | 0 | 0.8839 | 0.8839 |
| controlled_vs_candidate_no_context | context_relevance | 112 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 68 | 11 | 33 | 0.7545 | 0.8608 |
| controlled_vs_candidate_no_context | naturalness | 86 | 26 | 0 | 0.7679 | 0.7679 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 112 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 109 | 3 | 0 | 0.9732 | 0.9732 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 51 | 11 | 50 | 0.6786 | 0.8226 |
| controlled_vs_candidate_no_context | persona_style | 24 | 7 | 81 | 0.5759 | 0.7742 |
| controlled_vs_candidate_no_context | distinct1 | 69 | 43 | 0 | 0.6161 | 0.6161 |
| controlled_vs_candidate_no_context | length_score | 75 | 32 | 5 | 0.6920 | 0.7009 |
| controlled_vs_candidate_no_context | sentence_score | 77 | 0 | 35 | 0.8438 | 1.0000 |
| controlled_vs_candidate_no_context | bertscore_f1 | 91 | 21 | 0 | 0.8125 | 0.8125 |
| controlled_vs_candidate_no_context | overall_quality | 108 | 4 | 0 | 0.9643 | 0.9643 |
| controlled_vs_baseline_no_context | context_relevance | 112 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 51 | 22 | 39 | 0.6295 | 0.6986 |
| controlled_vs_baseline_no_context | naturalness | 64 | 48 | 0 | 0.5714 | 0.5714 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 111 | 0 | 1 | 0.9955 | 1.0000 |
| controlled_vs_baseline_no_context | context_overlap | 107 | 5 | 0 | 0.9554 | 0.9554 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 47 | 16 | 49 | 0.6384 | 0.7460 |
| controlled_vs_baseline_no_context | persona_style | 8 | 16 | 88 | 0.4643 | 0.3333 |
| controlled_vs_baseline_no_context | distinct1 | 15 | 96 | 1 | 0.1384 | 0.1351 |
| controlled_vs_baseline_no_context | length_score | 67 | 42 | 3 | 0.6116 | 0.6147 |
| controlled_vs_baseline_no_context | sentence_score | 44 | 1 | 67 | 0.6920 | 0.9778 |
| controlled_vs_baseline_no_context | bertscore_f1 | 89 | 23 | 0 | 0.7946 | 0.7946 |
| controlled_vs_baseline_no_context | overall_quality | 109 | 3 | 0 | 0.9732 | 0.9732 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 112 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 54 | 17 | 41 | 0.6652 | 0.7606 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | 56 | 55 | 1 | 0.5045 | 0.5045 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 112 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 105 | 7 | 0 | 0.9375 | 0.9375 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 47 | 12 | 53 | 0.6562 | 0.7966 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | 12 | 14 | 86 | 0.4911 | 0.4615 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | 16 | 95 | 1 | 0.1473 | 0.1441 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | 62 | 45 | 5 | 0.5759 | 0.5794 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 37 | 1 | 74 | 0.6607 | 0.9737 |
| controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 90 | 22 | 0 | 0.8036 | 0.8036 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 111 | 1 | 0 | 0.9911 | 0.9911 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 112 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 51 | 22 | 39 | 0.6295 | 0.6986 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 64 | 48 | 0 | 0.5714 | 0.5714 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 111 | 0 | 1 | 0.9955 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 107 | 5 | 0 | 0.9554 | 0.9554 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 47 | 16 | 49 | 0.6384 | 0.7460 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 8 | 16 | 88 | 0.4643 | 0.3333 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | 15 | 96 | 1 | 0.1384 | 0.1351 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 67 | 42 | 3 | 0.6116 | 0.6147 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 44 | 1 | 67 | 0.6920 | 0.9778 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 89 | 23 | 0 | 0.7946 | 0.7946 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 109 | 3 | 0 | 0.9732 | 0.9732 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 112 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 54 | 17 | 41 | 0.6652 | 0.7606 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | 56 | 55 | 1 | 0.5045 | 0.5045 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 112 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 105 | 7 | 0 | 0.9375 | 0.9375 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 47 | 12 | 53 | 0.6562 | 0.7966 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | 12 | 14 | 86 | 0.4911 | 0.4615 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | 16 | 95 | 1 | 0.1473 | 0.1441 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | 62 | 45 | 5 | 0.5759 | 0.5794 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | 37 | 1 | 74 | 0.6607 | 0.9737 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 90 | 22 | 0 | 0.8036 | 0.8036 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 111 | 1 | 0 | 0.9911 | 0.9911 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

- BERTScore status: enabled.

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
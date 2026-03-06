# Proposal Alignment Evaluation Report

- Run ID: `20260304T231200Z`
- Generated: `2026-03-04T23:36:30.765136+00:00`
- Scenarios: `artifacts\proposal_control_tuning\final112_tuned\20260304T231200Z\scenarios.jsonl`
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
| proposed_contextual_controlled | 0.2685 (0.2521, 0.2858) | 0.3522 (0.3241, 0.3817) | 0.8822 (0.8672, 0.8969) | 0.3861 (0.3761, 0.3962) | 0.0826 |
| proposed_contextual | 0.0960 (0.0731, 0.1214) | 0.1678 (0.1415, 0.1954) | 0.8005 (0.7880, 0.8135) | 0.2398 (0.2225, 0.2576) | 0.0721 |
| candidate_no_context | 0.0252 (0.0198, 0.0314) | 0.1712 (0.1452, 0.1985) | 0.8159 (0.8016, 0.8321) | 0.2109 (0.2005, 0.2214) | 0.0373 |
| baseline_no_context | 0.0452 (0.0361, 0.0545) | 0.1882 (0.1676, 0.2086) | 0.8876 (0.8765, 0.8981) | 0.2391 (0.2302, 0.2484) | 0.0564 |
| baseline_no_context_phi3_latest | 0.0414 (0.0325, 0.0509) | 0.1903 (0.1679, 0.2143) | 0.8809 (0.8701, 0.8912) | 0.2366 (0.2271, 0.2467) | 0.0533 |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0708 | 2.8092 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0034 | -0.0200 |
| proposed_vs_candidate_no_context | naturalness | -0.0154 | -0.0189 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0925 | 4.7657 |
| proposed_vs_candidate_no_context | context_overlap | 0.0202 | 0.5228 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0025 | -0.0296 |
| proposed_vs_candidate_no_context | persona_style | -0.0073 | -0.0139 |
| proposed_vs_candidate_no_context | distinct1 | -0.0033 | -0.0036 |
| proposed_vs_candidate_no_context | length_score | -0.0649 | -0.2045 |
| proposed_vs_candidate_no_context | sentence_score | -0.0089 | -0.0116 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0348 | 0.9310 |
| proposed_vs_candidate_no_context | overall_quality | 0.0290 | 0.1374 |
| proposed_vs_baseline_no_context | context_relevance | 0.0508 | 1.1219 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0203 | -0.1081 |
| proposed_vs_baseline_no_context | naturalness | -0.0871 | -0.0982 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0652 | 1.3972 |
| proposed_vs_baseline_no_context | context_overlap | 0.0171 | 0.4068 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0090 | -0.1000 |
| proposed_vs_baseline_no_context | persona_style | -0.0658 | -0.1131 |
| proposed_vs_baseline_no_context | distinct1 | -0.0424 | -0.0434 |
| proposed_vs_baseline_no_context | length_score | -0.2937 | -0.5379 |
| proposed_vs_baseline_no_context | sentence_score | -0.1125 | -0.1290 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0157 | 0.2783 |
| proposed_vs_baseline_no_context | overall_quality | 0.0007 | 0.0029 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 0.0546 | 1.3195 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | -0.0225 | -0.1180 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | -0.0804 | -0.0912 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.0706 | 1.7126 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 0.0172 | 0.4133 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | -0.0110 | -0.1196 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | -0.0684 | -0.1171 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | -0.0428 | -0.0437 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | -0.2717 | -0.5185 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | -0.0871 | -0.1028 |
| proposed_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0187 | 0.3510 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | 0.0032 | 0.0135 |
| controlled_vs_proposed_raw | context_relevance | 0.1725 | 1.7973 |
| controlled_vs_proposed_raw | persona_consistency | 0.1844 | 1.0989 |
| controlled_vs_proposed_raw | naturalness | 0.0817 | 0.1021 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2283 | 2.0414 |
| controlled_vs_proposed_raw | context_overlap | 0.0423 | 0.7171 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2100 | 2.6008 |
| controlled_vs_proposed_raw | persona_style | 0.0821 | 0.1591 |
| controlled_vs_proposed_raw | distinct1 | -0.0025 | -0.0027 |
| controlled_vs_proposed_raw | length_score | 0.3205 | 1.2700 |
| controlled_vs_proposed_raw | sentence_score | 0.1844 | 0.2427 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0106 | 0.1465 |
| controlled_vs_proposed_raw | overall_quality | 0.1463 | 0.6100 |
| controlled_vs_candidate_no_context | context_relevance | 0.2433 | 9.6554 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1810 | 1.0569 |
| controlled_vs_candidate_no_context | naturalness | 0.0663 | 0.0813 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3208 | 16.5356 |
| controlled_vs_candidate_no_context | context_overlap | 0.0625 | 1.6148 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2075 | 2.4941 |
| controlled_vs_candidate_no_context | persona_style | 0.0749 | 0.1430 |
| controlled_vs_candidate_no_context | distinct1 | -0.0058 | -0.0062 |
| controlled_vs_candidate_no_context | length_score | 0.2557 | 0.8058 |
| controlled_vs_candidate_no_context | sentence_score | 0.1754 | 0.2282 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0453 | 1.2140 |
| controlled_vs_candidate_no_context | overall_quality | 0.1753 | 0.8312 |
| controlled_vs_baseline_no_context | context_relevance | 0.2233 | 4.9354 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1641 | 0.8720 |
| controlled_vs_baseline_no_context | naturalness | -0.0054 | -0.0061 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.2935 | 6.2908 |
| controlled_vs_baseline_no_context | context_overlap | 0.0594 | 1.4156 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2010 | 2.2408 |
| controlled_vs_baseline_no_context | persona_style | 0.0163 | 0.0281 |
| controlled_vs_baseline_no_context | distinct1 | -0.0449 | -0.0459 |
| controlled_vs_baseline_no_context | length_score | 0.0268 | 0.0490 |
| controlled_vs_baseline_no_context | sentence_score | 0.0719 | 0.0824 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0263 | 0.4656 |
| controlled_vs_baseline_no_context | overall_quality | 0.1470 | 0.6147 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.2271 | 5.4883 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1620 | 0.8512 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | 0.0013 | 0.0015 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.2989 | 7.2500 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0595 | 1.4268 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1990 | 2.1701 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | 0.0137 | 0.0234 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0453 | -0.0463 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | 0.0488 | 0.0931 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.0973 | 0.1149 |
| controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0293 | 0.5490 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1495 | 0.6317 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2233 | 4.9354 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1641 | 0.8720 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0054 | -0.0061 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.2935 | 6.2908 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0594 | 1.4156 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2010 | 2.2408 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0163 | 0.0281 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0449 | -0.0459 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.0268 | 0.0490 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0719 | 0.0824 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0263 | 0.4656 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1470 | 0.6147 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.2271 | 5.4883 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1620 | 0.8512 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | 0.0013 | 0.0015 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.2989 | 7.2500 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0595 | 1.4268 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1990 | 2.1701 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | 0.0137 | 0.0234 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0453 | -0.0463 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | 0.0488 | 0.0931 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.0973 | 0.1149 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0293 | 0.5490 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1495 | 0.6317 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0708 | (0.0482, 0.0972) | 0.0000 | 0.0708 | (0.0377, 0.1032) | 0.0000 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0034 | (-0.0307, 0.0228) | 0.5920 | -0.0034 | (-0.0303, 0.0213) | 0.5987 |
| proposed_vs_candidate_no_context | naturalness | -0.0154 | (-0.0296, -0.0016) | 0.9840 | -0.0154 | (-0.0324, -0.0014) | 0.9853 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0925 | (0.0624, 0.1258) | 0.0000 | 0.0925 | (0.0497, 0.1332) | 0.0000 |
| proposed_vs_candidate_no_context | context_overlap | 0.0202 | (0.0118, 0.0304) | 0.0000 | 0.0202 | (0.0084, 0.0327) | 0.0000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0025 | (-0.0345, 0.0292) | 0.5573 | -0.0025 | (-0.0284, 0.0244) | 0.5800 |
| proposed_vs_candidate_no_context | persona_style | -0.0073 | (-0.0410, 0.0251) | 0.6583 | -0.0073 | (-0.0588, 0.0278) | 0.6037 |
| proposed_vs_candidate_no_context | distinct1 | -0.0033 | (-0.0114, 0.0044) | 0.8030 | -0.0033 | (-0.0134, 0.0064) | 0.7617 |
| proposed_vs_candidate_no_context | length_score | -0.0649 | (-0.1226, -0.0104) | 0.9893 | -0.0649 | (-0.1197, -0.0113) | 0.9913 |
| proposed_vs_candidate_no_context | sentence_score | -0.0089 | (-0.0518, 0.0317) | 0.6793 | -0.0089 | (-0.0522, 0.0344) | 0.6353 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0348 | (0.0240, 0.0460) | 0.0000 | 0.0348 | (0.0183, 0.0544) | 0.0000 |
| proposed_vs_candidate_no_context | overall_quality | 0.0290 | (0.0128, 0.0455) | 0.0003 | 0.0290 | (0.0104, 0.0492) | 0.0000 |
| proposed_vs_baseline_no_context | context_relevance | 0.0508 | (0.0266, 0.0758) | 0.0003 | 0.0508 | (0.0109, 0.0893) | 0.0053 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0203 | (-0.0509, 0.0125) | 0.8940 | -0.0203 | (-0.0673, 0.0212) | 0.8240 |
| proposed_vs_baseline_no_context | naturalness | -0.0871 | (-0.1038, -0.0699) | 1.0000 | -0.0871 | (-0.1136, -0.0602) | 1.0000 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0652 | (0.0335, 0.0993) | 0.0000 | 0.0652 | (0.0117, 0.1175) | 0.0080 |
| proposed_vs_baseline_no_context | context_overlap | 0.0171 | (0.0080, 0.0278) | 0.0000 | 0.0171 | (0.0050, 0.0301) | 0.0043 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0090 | (-0.0444, 0.0258) | 0.6877 | -0.0090 | (-0.0443, 0.0293) | 0.6913 |
| proposed_vs_baseline_no_context | persona_style | -0.0658 | (-0.1069, -0.0270) | 1.0000 | -0.0658 | (-0.1838, 0.0167) | 0.8850 |
| proposed_vs_baseline_no_context | distinct1 | -0.0424 | (-0.0505, -0.0341) | 1.0000 | -0.0424 | (-0.0544, -0.0305) | 1.0000 |
| proposed_vs_baseline_no_context | length_score | -0.2938 | (-0.3595, -0.2262) | 1.0000 | -0.2938 | (-0.3839, -0.1836) | 1.0000 |
| proposed_vs_baseline_no_context | sentence_score | -0.1125 | (-0.1558, -0.0692) | 1.0000 | -0.1125 | (-0.1830, -0.0473) | 1.0000 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0157 | (0.0032, 0.0277) | 0.0083 | 0.0157 | (-0.0015, 0.0357) | 0.0390 |
| proposed_vs_baseline_no_context | overall_quality | 0.0007 | (-0.0173, 0.0209) | 0.4773 | 0.0007 | (-0.0332, 0.0325) | 0.4747 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 0.0546 | (0.0294, 0.0828) | 0.0000 | 0.0546 | (0.0141, 0.0944) | 0.0023 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | -0.0225 | (-0.0533, 0.0082) | 0.9303 | -0.0225 | (-0.0644, 0.0175) | 0.8533 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | -0.0804 | (-0.0964, -0.0643) | 1.0000 | -0.0804 | (-0.1062, -0.0518) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.0706 | (0.0366, 0.1064) | 0.0000 | 0.0706 | (0.0203, 0.1229) | 0.0023 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 0.0172 | (0.0077, 0.0284) | 0.0003 | 0.0172 | (0.0049, 0.0299) | 0.0013 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | -0.0110 | (-0.0463, 0.0245) | 0.7277 | -0.0110 | (-0.0467, 0.0277) | 0.7187 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | -0.0684 | (-0.1092, -0.0301) | 0.9997 | -0.0684 | (-0.1884, 0.0128) | 0.9117 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | -0.0428 | (-0.0507, -0.0349) | 1.0000 | -0.0428 | (-0.0530, -0.0334) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | -0.2717 | (-0.3327, -0.2077) | 1.0000 | -0.2717 | (-0.3756, -0.1411) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | -0.0871 | (-0.1330, -0.0406) | 0.9997 | -0.0871 | (-0.1532, -0.0214) | 0.9980 |
| proposed_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0187 | (0.0060, 0.0315) | 0.0020 | 0.0187 | (0.0022, 0.0418) | 0.0033 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | 0.0032 | (-0.0156, 0.0217) | 0.3683 | 0.0032 | (-0.0251, 0.0327) | 0.4260 |
| controlled_vs_proposed_raw | context_relevance | 0.1725 | (0.1479, 0.1964) | 0.0000 | 0.1725 | (0.1442, 0.2033) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.1844 | (0.1524, 0.2168) | 0.0000 | 0.1844 | (0.1376, 0.2322) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0817 | (0.0605, 0.1023) | 0.0000 | 0.0817 | (0.0367, 0.1249) | 0.0000 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2283 | (0.1956, 0.2586) | 0.0000 | 0.2283 | (0.1903, 0.2675) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0423 | (0.0321, 0.0517) | 0.0000 | 0.0423 | (0.0296, 0.0541) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2100 | (0.1718, 0.2509) | 0.0000 | 0.2100 | (0.1612, 0.2720) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.0821 | (0.0421, 0.1220) | 0.0000 | 0.0821 | (0.0030, 0.1954) | 0.0187 |
| controlled_vs_proposed_raw | distinct1 | -0.0025 | (-0.0121, 0.0069) | 0.7020 | -0.0025 | (-0.0132, 0.0081) | 0.6663 |
| controlled_vs_proposed_raw | length_score | 0.3205 | (0.2330, 0.4045) | 0.0000 | 0.3205 | (0.1562, 0.4894) | 0.0000 |
| controlled_vs_proposed_raw | sentence_score | 0.1844 | (0.1429, 0.2241) | 0.0000 | 0.1844 | (0.1094, 0.2558) | 0.0000 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0106 | (-0.0010, 0.0217) | 0.0363 | 0.0106 | (-0.0078, 0.0273) | 0.1210 |
| controlled_vs_proposed_raw | overall_quality | 0.1463 | (0.1291, 0.1634) | 0.0000 | 0.1463 | (0.1203, 0.1730) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2433 | (0.2262, 0.2604) | 0.0000 | 0.2433 | (0.2205, 0.2783) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1810 | (0.1508, 0.2120) | 0.0000 | 0.1810 | (0.1412, 0.2261) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.0663 | (0.0429, 0.0879) | 0.0000 | 0.0663 | (0.0275, 0.1052) | 0.0003 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3208 | (0.2986, 0.3441) | 0.0000 | 0.3208 | (0.2906, 0.3682) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0625 | (0.0563, 0.0689) | 0.0000 | 0.0625 | (0.0559, 0.0695) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2075 | (0.1710, 0.2426) | 0.0000 | 0.2075 | (0.1583, 0.2654) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.0749 | (0.0380, 0.1133) | 0.0000 | 0.0749 | (0.0080, 0.1546) | 0.0107 |
| controlled_vs_candidate_no_context | distinct1 | -0.0058 | (-0.0147, 0.0026) | 0.9030 | -0.0058 | (-0.0180, 0.0074) | 0.8103 |
| controlled_vs_candidate_no_context | length_score | 0.2557 | (0.1649, 0.3458) | 0.0000 | 0.2557 | (0.1039, 0.4045) | 0.0003 |
| controlled_vs_candidate_no_context | sentence_score | 0.1754 | (0.1353, 0.2156) | 0.0000 | 0.1754 | (0.1196, 0.2411) | 0.0000 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0453 | (0.0342, 0.0560) | 0.0000 | 0.0453 | (0.0264, 0.0623) | 0.0000 |
| controlled_vs_candidate_no_context | overall_quality | 0.1753 | (0.1622, 0.1886) | 0.0000 | 0.1753 | (0.1583, 0.1940) | 0.0000 |
| controlled_vs_baseline_no_context | context_relevance | 0.2233 | (0.2053, 0.2426) | 0.0000 | 0.2233 | (0.1920, 0.2659) | 0.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1641 | (0.1351, 0.1933) | 0.0000 | 0.1641 | (0.1217, 0.2025) | 0.0000 |
| controlled_vs_baseline_no_context | naturalness | -0.0054 | (-0.0243, 0.0132) | 0.7200 | -0.0054 | (-0.0332, 0.0250) | 0.6283 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.2935 | (0.2685, 0.3197) | 0.0000 | 0.2935 | (0.2506, 0.3533) | 0.0000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0594 | (0.0529, 0.0659) | 0.0000 | 0.0594 | (0.0529, 0.0650) | 0.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2010 | (0.1670, 0.2396) | 0.0000 | 0.2010 | (0.1444, 0.2519) | 0.0000 |
| controlled_vs_baseline_no_context | persona_style | 0.0163 | (-0.0074, 0.0414) | 0.0910 | 0.0163 | (-0.0109, 0.0530) | 0.1577 |
| controlled_vs_baseline_no_context | distinct1 | -0.0449 | (-0.0526, -0.0371) | 1.0000 | -0.0449 | (-0.0511, -0.0393) | 1.0000 |
| controlled_vs_baseline_no_context | length_score | 0.0268 | (-0.0539, 0.1083) | 0.2633 | 0.0268 | (-0.0956, 0.1414) | 0.3203 |
| controlled_vs_baseline_no_context | sentence_score | 0.0719 | (0.0348, 0.1089) | 0.0000 | 0.0719 | (0.0125, 0.1406) | 0.0077 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0263 | (0.0155, 0.0373) | 0.0000 | 0.0263 | (0.0145, 0.0380) | 0.0000 |
| controlled_vs_baseline_no_context | overall_quality | 0.1470 | (0.1347, 0.1590) | 0.0000 | 0.1470 | (0.1271, 0.1662) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.2271 | (0.2084, 0.2468) | 0.0000 | 0.2271 | (0.1938, 0.2715) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1620 | (0.1324, 0.1916) | 0.0000 | 0.1620 | (0.1174, 0.2095) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | 0.0013 | (-0.0172, 0.0189) | 0.4450 | 0.0013 | (-0.0268, 0.0302) | 0.4767 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.2989 | (0.2741, 0.3262) | 0.0000 | 0.2989 | (0.2520, 0.3607) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0595 | (0.0530, 0.0664) | 0.0000 | 0.0595 | (0.0528, 0.0657) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1990 | (0.1640, 0.2343) | 0.0000 | 0.1990 | (0.1377, 0.2586) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | 0.0137 | (-0.0097, 0.0391) | 0.1227 | 0.0137 | (-0.0108, 0.0494) | 0.2107 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0453 | (-0.0529, -0.0375) | 1.0000 | -0.0453 | (-0.0524, -0.0373) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | 0.0488 | (-0.0286, 0.1301) | 0.1143 | 0.0488 | (-0.0822, 0.1827) | 0.2410 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.0973 | (0.0576, 0.1375) | 0.0000 | 0.0973 | (0.0500, 0.1451) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0293 | (0.0175, 0.0410) | 0.0000 | 0.0293 | (0.0110, 0.0454) | 0.0013 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1495 | (0.1373, 0.1616) | 0.0000 | 0.1495 | (0.1278, 0.1707) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2233 | (0.2049, 0.2417) | 0.0000 | 0.2233 | (0.1925, 0.2666) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1641 | (0.1322, 0.1939) | 0.0000 | 0.1641 | (0.1169, 0.2033) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0054 | (-0.0245, 0.0130) | 0.7217 | -0.0054 | (-0.0333, 0.0236) | 0.6297 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.2935 | (0.2686, 0.3193) | 0.0000 | 0.2935 | (0.2506, 0.3541) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0594 | (0.0530, 0.0662) | 0.0000 | 0.0594 | (0.0531, 0.0648) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2010 | (0.1650, 0.2366) | 0.0000 | 0.2010 | (0.1441, 0.2511) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0163 | (-0.0076, 0.0412) | 0.0960 | 0.0163 | (-0.0109, 0.0530) | 0.1557 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0449 | (-0.0530, -0.0367) | 1.0000 | -0.0449 | (-0.0509, -0.0393) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.0268 | (-0.0574, 0.1101) | 0.2623 | 0.0268 | (-0.0952, 0.1470) | 0.3357 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0719 | (0.0348, 0.1089) | 0.0000 | 0.0719 | (0.0125, 0.1344) | 0.0063 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0263 | (0.0151, 0.0376) | 0.0000 | 0.0263 | (0.0141, 0.0387) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1470 | (0.1346, 0.1597) | 0.0000 | 0.1470 | (0.1276, 0.1648) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.2271 | (0.2087, 0.2467) | 0.0000 | 0.2271 | (0.1937, 0.2713) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1620 | (0.1330, 0.1907) | 0.0000 | 0.1620 | (0.1145, 0.2089) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | 0.0013 | (-0.0180, 0.0188) | 0.4527 | 0.0013 | (-0.0257, 0.0288) | 0.4550 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.2989 | (0.2742, 0.3244) | 0.0000 | 0.2989 | (0.2528, 0.3615) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0595 | (0.0527, 0.0662) | 0.0000 | 0.0595 | (0.0526, 0.0658) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1990 | (0.1622, 0.2348) | 0.0000 | 0.1990 | (0.1376, 0.2634) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | 0.0137 | (-0.0098, 0.0387) | 0.1327 | 0.0137 | (-0.0089, 0.0494) | 0.1917 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0453 | (-0.0527, -0.0377) | 1.0000 | -0.0453 | (-0.0523, -0.0375) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | 0.0488 | (-0.0324, 0.1301) | 0.1187 | 0.0488 | (-0.0884, 0.1804) | 0.2433 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.0973 | (0.0594, 0.1379) | 0.0000 | 0.0973 | (0.0469, 0.1482) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0293 | (0.0175, 0.0417) | 0.0000 | 0.0293 | (0.0121, 0.0456) | 0.0010 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1495 | (0.1377, 0.1615) | 0.0000 | 0.1495 | (0.1284, 0.1712) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 49 | 20 | 43 | 0.6295 | 0.7101 |
| proposed_vs_candidate_no_context | persona_consistency | 25 | 23 | 64 | 0.5089 | 0.5208 |
| proposed_vs_candidate_no_context | naturalness | 31 | 38 | 43 | 0.4688 | 0.4493 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 41 | 8 | 63 | 0.6473 | 0.8367 |
| proposed_vs_candidate_no_context | context_overlap | 49 | 20 | 43 | 0.6295 | 0.7101 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 17 | 18 | 77 | 0.4955 | 0.4857 |
| proposed_vs_candidate_no_context | persona_style | 12 | 13 | 87 | 0.4955 | 0.4800 |
| proposed_vs_candidate_no_context | distinct1 | 29 | 35 | 48 | 0.4732 | 0.4531 |
| proposed_vs_candidate_no_context | length_score | 23 | 41 | 48 | 0.4196 | 0.3594 |
| proposed_vs_candidate_no_context | sentence_score | 20 | 22 | 70 | 0.4911 | 0.4762 |
| proposed_vs_candidate_no_context | bertscore_f1 | 58 | 12 | 42 | 0.7054 | 0.8286 |
| proposed_vs_candidate_no_context | overall_quality | 51 | 19 | 42 | 0.6429 | 0.7286 |
| proposed_vs_baseline_no_context | context_relevance | 64 | 48 | 0 | 0.5714 | 0.5714 |
| proposed_vs_baseline_no_context | persona_consistency | 25 | 47 | 40 | 0.4018 | 0.3472 |
| proposed_vs_baseline_no_context | naturalness | 19 | 93 | 0 | 0.1696 | 0.1696 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 44 | 23 | 45 | 0.5938 | 0.6567 |
| proposed_vs_baseline_no_context | context_overlap | 72 | 40 | 0 | 0.6429 | 0.6429 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 19 | 29 | 64 | 0.4554 | 0.3958 |
| proposed_vs_baseline_no_context | persona_style | 11 | 26 | 75 | 0.4330 | 0.2973 |
| proposed_vs_baseline_no_context | distinct1 | 17 | 84 | 11 | 0.2009 | 0.1683 |
| proposed_vs_baseline_no_context | length_score | 20 | 91 | 1 | 0.1830 | 0.1802 |
| proposed_vs_baseline_no_context | sentence_score | 13 | 48 | 51 | 0.3438 | 0.2131 |
| proposed_vs_baseline_no_context | bertscore_f1 | 63 | 49 | 0 | 0.5625 | 0.5625 |
| proposed_vs_baseline_no_context | overall_quality | 44 | 68 | 0 | 0.3929 | 0.3929 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 61 | 50 | 1 | 0.5491 | 0.5495 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | 23 | 45 | 44 | 0.4018 | 0.3382 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | 18 | 94 | 0 | 0.1607 | 0.1607 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 44 | 23 | 45 | 0.5938 | 0.6567 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 70 | 41 | 1 | 0.6295 | 0.6306 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 18 | 26 | 68 | 0.4643 | 0.4091 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | 10 | 27 | 75 | 0.4241 | 0.2703 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | 13 | 83 | 16 | 0.1875 | 0.1354 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | 15 | 90 | 7 | 0.1652 | 0.1429 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | 16 | 43 | 53 | 0.3795 | 0.2712 |
| proposed_vs_baseline_no_context_phi3_latest | bertscore_f1 | 68 | 44 | 0 | 0.6071 | 0.6071 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | 45 | 67 | 0 | 0.4018 | 0.4018 |
| controlled_vs_proposed_raw | context_relevance | 103 | 9 | 0 | 0.9196 | 0.9196 |
| controlled_vs_proposed_raw | persona_consistency | 90 | 9 | 13 | 0.8616 | 0.9091 |
| controlled_vs_proposed_raw | naturalness | 78 | 34 | 0 | 0.6964 | 0.6964 |
| controlled_vs_proposed_raw | context_keyword_coverage | 101 | 9 | 2 | 0.9107 | 0.9182 |
| controlled_vs_proposed_raw | context_overlap | 101 | 11 | 0 | 0.9018 | 0.9018 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 89 | 8 | 15 | 0.8616 | 0.9175 |
| controlled_vs_proposed_raw | persona_style | 27 | 7 | 78 | 0.5893 | 0.7941 |
| controlled_vs_proposed_raw | distinct1 | 57 | 52 | 3 | 0.5223 | 0.5229 |
| controlled_vs_proposed_raw | length_score | 75 | 34 | 3 | 0.6830 | 0.6881 |
| controlled_vs_proposed_raw | sentence_score | 66 | 8 | 38 | 0.7589 | 0.8919 |
| controlled_vs_proposed_raw | bertscore_f1 | 71 | 41 | 0 | 0.6339 | 0.6339 |
| controlled_vs_proposed_raw | overall_quality | 103 | 9 | 0 | 0.9196 | 0.9196 |
| controlled_vs_candidate_no_context | context_relevance | 112 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 91 | 7 | 14 | 0.8750 | 0.9286 |
| controlled_vs_candidate_no_context | naturalness | 81 | 31 | 0 | 0.7232 | 0.7232 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 111 | 0 | 1 | 0.9955 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 111 | 1 | 0 | 0.9911 | 0.9911 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 87 | 5 | 20 | 0.8661 | 0.9457 |
| controlled_vs_candidate_no_context | persona_style | 28 | 9 | 75 | 0.5848 | 0.7568 |
| controlled_vs_candidate_no_context | distinct1 | 54 | 53 | 5 | 0.5045 | 0.5047 |
| controlled_vs_candidate_no_context | length_score | 71 | 31 | 10 | 0.6786 | 0.6961 |
| controlled_vs_candidate_no_context | sentence_score | 64 | 8 | 40 | 0.7500 | 0.8889 |
| controlled_vs_candidate_no_context | bertscore_f1 | 90 | 22 | 0 | 0.8036 | 0.8036 |
| controlled_vs_candidate_no_context | overall_quality | 111 | 1 | 0 | 0.9911 | 0.9911 |
| controlled_vs_baseline_no_context | context_relevance | 112 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 89 | 9 | 14 | 0.8571 | 0.9082 |
| controlled_vs_baseline_no_context | naturalness | 59 | 53 | 0 | 0.5268 | 0.5268 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 111 | 0 | 1 | 0.9955 | 1.0000 |
| controlled_vs_baseline_no_context | context_overlap | 108 | 4 | 0 | 0.9643 | 0.9643 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 88 | 6 | 18 | 0.8661 | 0.9362 |
| controlled_vs_baseline_no_context | persona_style | 17 | 11 | 84 | 0.5268 | 0.6071 |
| controlled_vs_baseline_no_context | distinct1 | 13 | 95 | 4 | 0.1339 | 0.1204 |
| controlled_vs_baseline_no_context | length_score | 63 | 47 | 2 | 0.5714 | 0.5727 |
| controlled_vs_baseline_no_context | sentence_score | 30 | 8 | 74 | 0.5982 | 0.7895 |
| controlled_vs_baseline_no_context | bertscore_f1 | 76 | 36 | 0 | 0.6786 | 0.6786 |
| controlled_vs_baseline_no_context | overall_quality | 111 | 1 | 0 | 0.9911 | 0.9911 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 112 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 87 | 7 | 18 | 0.8571 | 0.9255 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | 60 | 52 | 0 | 0.5357 | 0.5357 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 110 | 0 | 2 | 0.9911 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 108 | 4 | 0 | 0.9643 | 0.9643 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 83 | 6 | 23 | 0.8438 | 0.9326 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | 13 | 15 | 84 | 0.4911 | 0.4643 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | 16 | 90 | 6 | 0.1696 | 0.1509 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | 66 | 44 | 2 | 0.5982 | 0.6000 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 41 | 10 | 61 | 0.6384 | 0.8039 |
| controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 78 | 34 | 0 | 0.6964 | 0.6964 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 112 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 112 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 89 | 9 | 14 | 0.8571 | 0.9082 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 59 | 53 | 0 | 0.5268 | 0.5268 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 111 | 0 | 1 | 0.9955 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 108 | 4 | 0 | 0.9643 | 0.9643 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 88 | 6 | 18 | 0.8661 | 0.9362 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 17 | 11 | 84 | 0.5268 | 0.6071 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | 13 | 95 | 4 | 0.1339 | 0.1204 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 63 | 47 | 2 | 0.5714 | 0.5727 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 30 | 8 | 74 | 0.5982 | 0.7895 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 76 | 36 | 0 | 0.6786 | 0.6786 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 111 | 1 | 0 | 0.9911 | 0.9911 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 112 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 87 | 7 | 18 | 0.8571 | 0.9255 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | 60 | 52 | 0 | 0.5357 | 0.5357 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 110 | 0 | 2 | 0.9911 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 108 | 4 | 0 | 0.9643 | 0.9643 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 83 | 6 | 23 | 0.8438 | 0.9326 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | 13 | 15 | 84 | 0.4911 | 0.4643 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | 16 | 90 | 6 | 0.1696 | 0.1509 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | 66 | 44 | 2 | 0.5982 | 0.6000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | 41 | 10 | 61 | 0.6384 | 0.8039 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 78 | 34 | 0 | 0.6964 | 0.6964 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 112 | 0 | 0 | 1.0000 | 1.0000 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.4286 | 0.4196 | 0.5804 |
| proposed_contextual | 0.0000 | 0.0000 | 0.4821 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.4821 | 0.0000 | 0.0000 |
| baseline_no_context | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline_no_context_phi3_latest | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `8`
- Unique template signatures: `71`
- Template signature ratio: `0.6339`
- Effective sample size by source clustering: `8.00`
- Effective sample size by template-signature clustering: `56.00`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: enabled.

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
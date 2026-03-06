# Proposal Alignment Evaluation Report

- Run ID: `20260304T230712Z`
- Generated: `2026-03-04T23:11:30.113449+00:00`
- Scenarios: `artifacts\proposal_control_tuning\tuning_continue\20260304T230712Z\scenarios.jsonl`
- Scenario count: `20`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off
- `baseline_no_context`: model `phi3:mini`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.3033 (0.2659, 0.3414) | 0.3079 (0.2413, 0.3797) | 0.8742 (0.8417, 0.9052) | 0.3848 (0.3588, 0.4116) | 0.0934 |
| proposed_contextual | 0.1179 (0.0592, 0.1851) | 0.1592 (0.1134, 0.2066) | 0.8000 (0.7714, 0.8314) | 0.2472 (0.2099, 0.2885) | 0.0876 |
| candidate_no_context | 0.0252 (0.0127, 0.0416) | 0.1766 (0.1163, 0.2486) | 0.8070 (0.7738, 0.8451) | 0.2107 (0.1855, 0.2388) | 0.0362 |
| baseline_no_context | 0.0215 (0.0119, 0.0324) | 0.1783 (0.1343, 0.2249) | 0.8725 (0.8427, 0.9016) | 0.2222 (0.2052, 0.2406) | 0.0471 |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0928 | 3.6873 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0174 | -0.0984 |
| proposed_vs_candidate_no_context | naturalness | -0.0070 | -0.0087 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.1216 | 6.6875 |
| proposed_vs_candidate_no_context | context_overlap | 0.0255 | 0.6155 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0238 | -0.2725 |
| proposed_vs_candidate_no_context | persona_style | 0.0083 | 0.0156 |
| proposed_vs_candidate_no_context | distinct1 | 0.0054 | 0.0058 |
| proposed_vs_candidate_no_context | length_score | -0.0633 | -0.2043 |
| proposed_vs_candidate_no_context | sentence_score | 0.0350 | 0.0486 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0514 | 1.4213 |
| proposed_vs_candidate_no_context | overall_quality | 0.0366 | 0.1737 |
| proposed_vs_baseline_no_context | context_relevance | 0.0965 | 4.4933 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0191 | -0.1071 |
| proposed_vs_baseline_no_context | naturalness | -0.0724 | -0.0830 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.1261 | 9.2500 |
| proposed_vs_baseline_no_context | context_overlap | 0.0272 | 0.6846 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0133 | -0.1734 |
| proposed_vs_baseline_no_context | persona_style | -0.0422 | -0.0722 |
| proposed_vs_baseline_no_context | distinct1 | -0.0390 | -0.0399 |
| proposed_vs_baseline_no_context | length_score | -0.2567 | -0.5099 |
| proposed_vs_baseline_no_context | sentence_score | -0.0550 | -0.0679 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0405 | 0.8591 |
| proposed_vs_baseline_no_context | overall_quality | 0.0250 | 0.1126 |
| controlled_vs_proposed_raw | context_relevance | 0.1853 | 1.5717 |
| controlled_vs_proposed_raw | persona_consistency | 0.1486 | 0.9335 |
| controlled_vs_proposed_raw | naturalness | 0.0742 | 0.0927 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2442 | 1.7474 |
| controlled_vs_proposed_raw | context_overlap | 0.0479 | 0.7155 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1650 | 2.5955 |
| controlled_vs_proposed_raw | persona_style | 0.0832 | 0.1536 |
| controlled_vs_proposed_raw | distinct1 | -0.0002 | -0.0003 |
| controlled_vs_proposed_raw | length_score | 0.2750 | 1.1149 |
| controlled_vs_proposed_raw | sentence_score | 0.1925 | 0.2550 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0058 | 0.0662 |
| controlled_vs_proposed_raw | overall_quality | 0.1376 | 0.5564 |
| controlled_vs_candidate_no_context | context_relevance | 0.2781 | 11.0544 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1313 | 0.7432 |
| controlled_vs_candidate_no_context | naturalness | 0.0672 | 0.0832 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3658 | 20.1208 |
| controlled_vs_candidate_no_context | context_overlap | 0.0734 | 1.7715 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1412 | 1.6158 |
| controlled_vs_candidate_no_context | persona_style | 0.0916 | 0.1716 |
| controlled_vs_candidate_no_context | distinct1 | 0.0052 | 0.0056 |
| controlled_vs_candidate_no_context | length_score | 0.2117 | 0.6828 |
| controlled_vs_candidate_no_context | sentence_score | 0.2275 | 0.3160 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0572 | 1.5816 |
| controlled_vs_candidate_no_context | overall_quality | 0.1742 | 0.8267 |
| controlled_vs_baseline_no_context | context_relevance | 0.2818 | 13.1272 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1295 | 0.7264 |
| controlled_vs_baseline_no_context | naturalness | 0.0017 | 0.0020 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3704 | 27.1611 |
| controlled_vs_baseline_no_context | context_overlap | 0.0751 | 1.8900 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1517 | 1.9721 |
| controlled_vs_baseline_no_context | persona_style | 0.0411 | 0.0703 |
| controlled_vs_baseline_no_context | distinct1 | -0.0393 | -0.0402 |
| controlled_vs_baseline_no_context | length_score | 0.0183 | 0.0364 |
| controlled_vs_baseline_no_context | sentence_score | 0.1375 | 0.1698 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0463 | 0.9821 |
| controlled_vs_baseline_no_context | overall_quality | 0.1626 | 0.7316 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2818 | 13.1272 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1295 | 0.7264 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 0.0017 | 0.0020 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3704 | 27.1611 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0751 | 1.8900 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1517 | 1.9721 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0411 | 0.0703 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0393 | -0.0402 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.0183 | 0.0364 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.1375 | 0.1698 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0463 | 0.9821 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1626 | 0.7316 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0928 | (0.0341, 0.1571) | 0.0000 | 0.0928 | (0.0270, 0.1821) | 0.0000 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0174 | (-0.0818, 0.0440) | 0.6900 | -0.0174 | (-0.0763, 0.0315) | 0.7477 |
| proposed_vs_candidate_no_context | naturalness | -0.0070 | (-0.0398, 0.0251) | 0.6503 | -0.0070 | (-0.0544, 0.0232) | 0.6243 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.1216 | (0.0447, 0.2083) | 0.0000 | 0.1216 | (0.0360, 0.2364) | 0.0007 |
| proposed_vs_candidate_no_context | context_overlap | 0.0255 | (0.0070, 0.0469) | 0.0020 | 0.0255 | (0.0085, 0.0492) | 0.0000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0238 | (-0.0952, 0.0417) | 0.7697 | -0.0238 | (-0.1000, 0.0390) | 0.7827 |
| proposed_vs_candidate_no_context | persona_style | 0.0083 | (-0.0901, 0.1068) | 0.4607 | 0.0083 | (-0.0143, 0.0438) | 0.3647 |
| proposed_vs_candidate_no_context | distinct1 | 0.0054 | (-0.0102, 0.0209) | 0.2423 | 0.0054 | (-0.0125, 0.0239) | 0.2503 |
| proposed_vs_candidate_no_context | length_score | -0.0633 | (-0.1967, 0.0633) | 0.8500 | -0.0633 | (-0.2700, 0.0653) | 0.8020 |
| proposed_vs_candidate_no_context | sentence_score | 0.0350 | (-0.0525, 0.1225) | 0.2897 | 0.0350 | (-0.0933, 0.1556) | 0.3463 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0514 | (0.0219, 0.0841) | 0.0000 | 0.0514 | (0.0167, 0.0949) | 0.0000 |
| proposed_vs_candidate_no_context | overall_quality | 0.0366 | (-0.0065, 0.0823) | 0.0523 | 0.0366 | (-0.0034, 0.0862) | 0.0433 |
| proposed_vs_baseline_no_context | context_relevance | 0.0965 | (0.0391, 0.1635) | 0.0000 | 0.0965 | (0.0221, 0.1935) | 0.0007 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0191 | (-0.0682, 0.0349) | 0.7653 | -0.0191 | (-0.0803, 0.0580) | 0.7097 |
| proposed_vs_baseline_no_context | naturalness | -0.0724 | (-0.1141, -0.0308) | 0.9997 | -0.0724 | (-0.1100, -0.0171) | 0.9943 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.1261 | (0.0500, 0.2129) | 0.0003 | 0.1261 | (0.0296, 0.2608) | 0.0010 |
| proposed_vs_baseline_no_context | context_overlap | 0.0272 | (0.0045, 0.0522) | 0.0073 | 0.0272 | (-0.0005, 0.0600) | 0.0280 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0133 | (-0.0655, 0.0464) | 0.6920 | -0.0133 | (-0.0787, 0.0715) | 0.6260 |
| proposed_vs_baseline_no_context | persona_style | -0.0422 | (-0.1649, 0.0677) | 0.7650 | -0.0422 | (-0.1927, 0.0963) | 0.6993 |
| proposed_vs_baseline_no_context | distinct1 | -0.0390 | (-0.0580, -0.0195) | 0.9997 | -0.0390 | (-0.0536, -0.0215) | 0.9997 |
| proposed_vs_baseline_no_context | length_score | -0.2567 | (-0.4217, -0.0967) | 1.0000 | -0.2567 | (-0.3948, -0.0532) | 0.9917 |
| proposed_vs_baseline_no_context | sentence_score | -0.0550 | (-0.1600, 0.0525) | 0.8637 | -0.0550 | (-0.1751, 0.0876) | 0.8060 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0405 | (0.0105, 0.0713) | 0.0023 | 0.0405 | (0.0055, 0.0823) | 0.0137 |
| proposed_vs_baseline_no_context | overall_quality | 0.0250 | (-0.0182, 0.0703) | 0.1407 | 0.0250 | (-0.0309, 0.0965) | 0.2333 |
| controlled_vs_proposed_raw | context_relevance | 0.1853 | (0.1194, 0.2467) | 0.0000 | 0.1853 | (0.1058, 0.2404) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.1486 | (0.0811, 0.2119) | 0.0000 | 0.1486 | (0.0788, 0.2454) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0742 | (0.0353, 0.1106) | 0.0000 | 0.0742 | (0.0237, 0.1096) | 0.0053 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2442 | (0.1567, 0.3242) | 0.0000 | 0.2442 | (0.1448, 0.3210) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0479 | (0.0257, 0.0696) | 0.0000 | 0.0479 | (0.0229, 0.0670) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1650 | (0.0874, 0.2329) | 0.0000 | 0.1650 | (0.0917, 0.2574) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.0832 | (0.0059, 0.1776) | 0.0070 | 0.0832 | (-0.0035, 0.2250) | 0.0403 |
| controlled_vs_proposed_raw | distinct1 | -0.0002 | (-0.0234, 0.0204) | 0.5063 | -0.0002 | (-0.0303, 0.0209) | 0.5207 |
| controlled_vs_proposed_raw | length_score | 0.2750 | (0.1000, 0.4400) | 0.0010 | 0.2750 | (0.0521, 0.4439) | 0.0113 |
| controlled_vs_proposed_raw | sentence_score | 0.1925 | (0.0875, 0.2800) | 0.0007 | 0.1925 | (0.0291, 0.3023) | 0.0163 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0058 | (-0.0341, 0.0363) | 0.3580 | 0.0058 | (-0.0355, 0.0351) | 0.3833 |
| controlled_vs_proposed_raw | overall_quality | 0.1376 | (0.0950, 0.1808) | 0.0000 | 0.1376 | (0.0940, 0.1778) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2781 | (0.2498, 0.3072) | 0.0000 | 0.2781 | (0.2514, 0.3132) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1313 | (0.0668, 0.2002) | 0.0000 | 0.1313 | (0.0709, 0.2074) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.0672 | (0.0150, 0.1186) | 0.0050 | 0.0672 | (-0.0263, 0.1249) | 0.0610 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3658 | (0.3285, 0.4056) | 0.0000 | 0.3658 | (0.3296, 0.4090) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0734 | (0.0580, 0.0870) | 0.0000 | 0.0734 | (0.0576, 0.0877) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1412 | (0.0648, 0.2157) | 0.0000 | 0.1412 | (0.0782, 0.2299) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.0916 | (0.0167, 0.1910) | 0.0027 | 0.0916 | (0.0069, 0.2188) | 0.0240 |
| controlled_vs_candidate_no_context | distinct1 | 0.0052 | (-0.0167, 0.0253) | 0.3133 | 0.0052 | (-0.0222, 0.0247) | 0.3213 |
| controlled_vs_candidate_no_context | length_score | 0.2117 | (-0.0050, 0.4233) | 0.0290 | 0.2117 | (-0.1421, 0.4537) | 0.1053 |
| controlled_vs_candidate_no_context | sentence_score | 0.2275 | (0.1575, 0.2975) | 0.0000 | 0.2275 | (0.1474, 0.2962) | 0.0000 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0572 | (0.0339, 0.0812) | 0.0000 | 0.0572 | (0.0447, 0.0773) | 0.0000 |
| controlled_vs_candidate_no_context | overall_quality | 0.1742 | (0.1446, 0.2029) | 0.0000 | 0.1742 | (0.1434, 0.2019) | 0.0000 |
| controlled_vs_baseline_no_context | context_relevance | 0.2818 | (0.2394, 0.3278) | 0.0000 | 0.2818 | (0.2401, 0.3331) | 0.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1295 | (0.0637, 0.1907) | 0.0000 | 0.1295 | (0.0507, 0.2210) | 0.0003 |
| controlled_vs_baseline_no_context | naturalness | 0.0017 | (-0.0462, 0.0516) | 0.4703 | 0.0017 | (-0.0363, 0.0493) | 0.4730 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3704 | (0.3155, 0.4267) | 0.0000 | 0.3704 | (0.3162, 0.4338) | 0.0000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0751 | (0.0561, 0.0936) | 0.0000 | 0.0751 | (0.0594, 0.0965) | 0.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1517 | (0.0690, 0.2281) | 0.0003 | 0.1517 | (0.0526, 0.2675) | 0.0003 |
| controlled_vs_baseline_no_context | persona_style | 0.0411 | (-0.0081, 0.1228) | 0.1240 | 0.0411 | (0.0000, 0.1230) | 0.1067 |
| controlled_vs_baseline_no_context | distinct1 | -0.0393 | (-0.0596, -0.0187) | 0.9997 | -0.0393 | (-0.0650, -0.0203) | 1.0000 |
| controlled_vs_baseline_no_context | length_score | 0.0183 | (-0.1884, 0.2200) | 0.4217 | 0.0183 | (-0.1483, 0.2313) | 0.4297 |
| controlled_vs_baseline_no_context | sentence_score | 0.1375 | (0.0325, 0.2550) | 0.0113 | 0.1375 | (0.0350, 0.2567) | 0.0073 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0463 | (0.0166, 0.0759) | 0.0003 | 0.0463 | (0.0166, 0.0752) | 0.0003 |
| controlled_vs_baseline_no_context | overall_quality | 0.1626 | (0.1292, 0.1916) | 0.0000 | 0.1626 | (0.1242, 0.2004) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2818 | (0.2408, 0.3261) | 0.0000 | 0.2818 | (0.2422, 0.3322) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1295 | (0.0617, 0.1899) | 0.0000 | 0.1295 | (0.0555, 0.2205) | 0.0003 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 0.0017 | (-0.0486, 0.0533) | 0.4730 | 0.0017 | (-0.0389, 0.0514) | 0.4797 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3704 | (0.3155, 0.4283) | 0.0000 | 0.3704 | (0.3176, 0.4362) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0751 | (0.0562, 0.0928) | 0.0000 | 0.0751 | (0.0594, 0.0966) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1517 | (0.0709, 0.2245) | 0.0000 | 0.1517 | (0.0555, 0.2653) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0411 | (-0.0081, 0.1188) | 0.1230 | 0.0411 | (0.0000, 0.1227) | 0.1007 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0393 | (-0.0611, -0.0192) | 0.9997 | -0.0393 | (-0.0643, -0.0207) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.0183 | (-0.1800, 0.2200) | 0.4263 | 0.0183 | (-0.1577, 0.2314) | 0.4323 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.1375 | (0.0325, 0.2425) | 0.0097 | 0.1375 | (0.0350, 0.2567) | 0.0083 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0463 | (0.0163, 0.0753) | 0.0010 | 0.0463 | (0.0182, 0.0757) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1626 | (0.1305, 0.1914) | 0.0000 | 0.1626 | (0.1251, 0.1996) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 11 | 2 | 7 | 0.7250 | 0.8462 |
| proposed_vs_candidate_no_context | persona_consistency | 5 | 4 | 11 | 0.5250 | 0.5556 |
| proposed_vs_candidate_no_context | naturalness | 7 | 6 | 7 | 0.5250 | 0.5385 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 10 | 1 | 9 | 0.7250 | 0.9091 |
| proposed_vs_candidate_no_context | context_overlap | 10 | 3 | 7 | 0.6750 | 0.7692 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 3 | 4 | 13 | 0.4750 | 0.4286 |
| proposed_vs_candidate_no_context | persona_style | 2 | 2 | 16 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | distinct1 | 9 | 3 | 8 | 0.6500 | 0.7500 |
| proposed_vs_candidate_no_context | length_score | 5 | 7 | 8 | 0.4500 | 0.4167 |
| proposed_vs_candidate_no_context | sentence_score | 5 | 3 | 12 | 0.5500 | 0.6250 |
| proposed_vs_candidate_no_context | bertscore_f1 | 14 | 5 | 1 | 0.7250 | 0.7368 |
| proposed_vs_candidate_no_context | overall_quality | 12 | 7 | 1 | 0.6250 | 0.6316 |
| proposed_vs_baseline_no_context | context_relevance | 12 | 8 | 0 | 0.6000 | 0.6000 |
| proposed_vs_baseline_no_context | persona_consistency | 4 | 8 | 8 | 0.4000 | 0.3333 |
| proposed_vs_baseline_no_context | naturalness | 4 | 16 | 0 | 0.2000 | 0.2000 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 10 | 1 | 9 | 0.7250 | 0.9091 |
| proposed_vs_baseline_no_context | context_overlap | 12 | 8 | 0 | 0.6000 | 0.6000 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 3 | 5 | 12 | 0.4500 | 0.3750 |
| proposed_vs_baseline_no_context | persona_style | 2 | 5 | 13 | 0.4250 | 0.2857 |
| proposed_vs_baseline_no_context | distinct1 | 4 | 14 | 2 | 0.2500 | 0.2222 |
| proposed_vs_baseline_no_context | length_score | 5 | 15 | 0 | 0.2500 | 0.2500 |
| proposed_vs_baseline_no_context | sentence_score | 4 | 7 | 9 | 0.4250 | 0.3636 |
| proposed_vs_baseline_no_context | bertscore_f1 | 14 | 6 | 0 | 0.7000 | 0.7000 |
| proposed_vs_baseline_no_context | overall_quality | 10 | 10 | 0 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | context_relevance | 18 | 2 | 0 | 0.9000 | 0.9000 |
| controlled_vs_proposed_raw | persona_consistency | 15 | 2 | 3 | 0.8250 | 0.8824 |
| controlled_vs_proposed_raw | naturalness | 17 | 3 | 0 | 0.8500 | 0.8500 |
| controlled_vs_proposed_raw | context_keyword_coverage | 18 | 2 | 0 | 0.9000 | 0.9000 |
| controlled_vs_proposed_raw | context_overlap | 16 | 4 | 0 | 0.8000 | 0.8000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 15 | 2 | 3 | 0.8250 | 0.8824 |
| controlled_vs_proposed_raw | persona_style | 5 | 1 | 14 | 0.6000 | 0.8333 |
| controlled_vs_proposed_raw | distinct1 | 10 | 8 | 2 | 0.5500 | 0.5556 |
| controlled_vs_proposed_raw | length_score | 13 | 5 | 2 | 0.7000 | 0.7222 |
| controlled_vs_proposed_raw | sentence_score | 13 | 2 | 5 | 0.7750 | 0.8667 |
| controlled_vs_proposed_raw | bertscore_f1 | 15 | 5 | 0 | 0.7500 | 0.7500 |
| controlled_vs_proposed_raw | overall_quality | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_vs_candidate_no_context | context_relevance | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 13 | 2 | 5 | 0.7750 | 0.8667 |
| controlled_vs_candidate_no_context | naturalness | 14 | 6 | 0 | 0.7000 | 0.7000 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 13 | 2 | 5 | 0.7750 | 0.8667 |
| controlled_vs_candidate_no_context | persona_style | 5 | 0 | 15 | 0.6250 | 1.0000 |
| controlled_vs_candidate_no_context | distinct1 | 12 | 6 | 2 | 0.6500 | 0.6667 |
| controlled_vs_candidate_no_context | length_score | 12 | 8 | 0 | 0.6000 | 0.6000 |
| controlled_vs_candidate_no_context | sentence_score | 13 | 0 | 7 | 0.8250 | 1.0000 |
| controlled_vs_candidate_no_context | bertscore_f1 | 18 | 2 | 0 | 0.9000 | 0.9000 |
| controlled_vs_candidate_no_context | overall_quality | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | context_relevance | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 16 | 3 | 1 | 0.8250 | 0.8421 |
| controlled_vs_baseline_no_context | naturalness | 9 | 11 | 0 | 0.4500 | 0.4500 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | context_overlap | 18 | 2 | 0 | 0.9000 | 0.9000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 16 | 3 | 1 | 0.8250 | 0.8421 |
| controlled_vs_baseline_no_context | persona_style | 2 | 1 | 17 | 0.5250 | 0.6667 |
| controlled_vs_baseline_no_context | distinct1 | 3 | 16 | 1 | 0.1750 | 0.1579 |
| controlled_vs_baseline_no_context | length_score | 10 | 9 | 1 | 0.5250 | 0.5263 |
| controlled_vs_baseline_no_context | sentence_score | 9 | 2 | 9 | 0.6750 | 0.8182 |
| controlled_vs_baseline_no_context | bertscore_f1 | 15 | 5 | 0 | 0.7500 | 0.7500 |
| controlled_vs_baseline_no_context | overall_quality | 19 | 1 | 0 | 0.9500 | 0.9500 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 20 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 16 | 3 | 1 | 0.8250 | 0.8421 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 9 | 11 | 0 | 0.4500 | 0.4500 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 20 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 18 | 2 | 0 | 0.9000 | 0.9000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 16 | 3 | 1 | 0.8250 | 0.8421 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 2 | 1 | 17 | 0.5250 | 0.6667 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | 3 | 16 | 1 | 0.1750 | 0.1579 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 10 | 9 | 1 | 0.5250 | 0.5263 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 9 | 2 | 9 | 0.6750 | 0.8182 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 15 | 5 | 0 | 0.7500 | 0.7500 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 19 | 1 | 0 | 0.9500 | 0.9500 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.3500 | 0.4500 | 0.5500 |
| proposed_contextual | 0.0000 | 0.0000 | 0.4000 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.5500 | 0.0000 | 0.0000 |
| baseline_no_context | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `8`
- Unique template signatures: `18`
- Template signature ratio: `0.9000`
- Effective sample size by source clustering: `6.67`
- Effective sample size by template-signature clustering: `16.67`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: enabled.

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
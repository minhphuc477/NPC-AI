# Proposal Alignment Evaluation Report

- Run ID: `20260305T135130Z`
- Generated: `2026-03-05T13:54:48.811495+00:00`
- Scenarios: `artifacts\proposal_control_tuning\tuning_matrix\m2_l2\20260305T135130Z\scenarios.jsonl`
- Scenario count: `20`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off
- `baseline_no_context`: model `phi3:mini`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.3009 (0.2579, 0.3478) | 0.2963 (0.2468, 0.3460) | 0.8915 (0.8595, 0.9218) | 0.3825 (0.3611, 0.4040) | 0.0881 |
| proposed_contextual | 0.0615 (0.0207, 0.1213) | 0.1546 (0.1086, 0.2054) | 0.7827 (0.7631, 0.8070) | 0.2159 (0.1855, 0.2533) | 0.0522 |
| candidate_no_context | 0.0244 (0.0131, 0.0396) | 0.1749 (0.1141, 0.2492) | 0.7767 (0.7573, 0.7998) | 0.2046 (0.1815, 0.2307) | 0.0385 |
| baseline_no_context | 0.0391 (0.0222, 0.0579) | 0.1932 (0.1555, 0.2318) | 0.8736 (0.8459, 0.9014) | 0.2354 (0.2185, 0.2534) | 0.0601 |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0371 | 1.5211 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0203 | -0.1160 |
| proposed_vs_candidate_no_context | naturalness | 0.0061 | 0.0078 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0500 | 2.8696 |
| proposed_vs_candidate_no_context | context_overlap | 0.0070 | 0.1722 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0221 | -0.2583 |
| proposed_vs_candidate_no_context | persona_style | -0.0129 | -0.0243 |
| proposed_vs_candidate_no_context | distinct1 | 0.0120 | 0.0130 |
| proposed_vs_candidate_no_context | length_score | 0.0150 | 0.0978 |
| proposed_vs_candidate_no_context | sentence_score | -0.0175 | -0.0232 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0137 | 0.3569 |
| proposed_vs_candidate_no_context | overall_quality | 0.0113 | 0.0551 |
| proposed_vs_baseline_no_context | context_relevance | 0.0224 | 0.5731 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0386 | -0.1997 |
| proposed_vs_baseline_no_context | naturalness | -0.0908 | -0.1040 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0322 | 0.9140 |
| proposed_vs_baseline_no_context | context_overlap | -0.0005 | -0.0095 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0217 | -0.2542 |
| proposed_vs_baseline_no_context | persona_style | -0.1062 | -0.1700 |
| proposed_vs_baseline_no_context | distinct1 | -0.0400 | -0.0409 |
| proposed_vs_baseline_no_context | length_score | -0.3217 | -0.6565 |
| proposed_vs_baseline_no_context | sentence_score | -0.1050 | -0.1246 |
| proposed_vs_baseline_no_context | bertscore_f1 | -0.0079 | -0.1309 |
| proposed_vs_baseline_no_context | overall_quality | -0.0195 | -0.0828 |
| controlled_vs_proposed_raw | context_relevance | 0.2394 | 3.8936 |
| controlled_vs_proposed_raw | persona_consistency | 0.1417 | 0.9167 |
| controlled_vs_proposed_raw | naturalness | 0.1088 | 0.1390 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.3127 | 4.6371 |
| controlled_vs_proposed_raw | context_overlap | 0.0685 | 1.4388 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1524 | 2.3970 |
| controlled_vs_proposed_raw | persona_style | 0.0991 | 0.1910 |
| controlled_vs_proposed_raw | distinct1 | 0.0035 | 0.0037 |
| controlled_vs_proposed_raw | length_score | 0.4233 | 2.5149 |
| controlled_vs_proposed_raw | sentence_score | 0.2275 | 0.3085 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0359 | 0.6869 |
| controlled_vs_proposed_raw | overall_quality | 0.1666 | 0.7716 |
| controlled_vs_candidate_no_context | context_relevance | 0.2765 | 11.3373 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1214 | 0.6943 |
| controlled_vs_candidate_no_context | naturalness | 0.1149 | 0.1479 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3627 | 20.8130 |
| controlled_vs_candidate_no_context | context_overlap | 0.0755 | 1.8587 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1302 | 1.5194 |
| controlled_vs_candidate_no_context | persona_style | 0.0862 | 0.1621 |
| controlled_vs_candidate_no_context | distinct1 | 0.0155 | 0.0167 |
| controlled_vs_candidate_no_context | length_score | 0.4383 | 2.8587 |
| controlled_vs_candidate_no_context | sentence_score | 0.2100 | 0.2781 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0496 | 1.2891 |
| controlled_vs_candidate_no_context | overall_quality | 0.1779 | 0.8693 |
| controlled_vs_baseline_no_context | context_relevance | 0.2618 | 6.6980 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1031 | 0.5339 |
| controlled_vs_baseline_no_context | naturalness | 0.0180 | 0.0206 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3448 | 9.7892 |
| controlled_vs_baseline_no_context | context_overlap | 0.0681 | 1.4156 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1307 | 1.5335 |
| controlled_vs_baseline_no_context | persona_style | -0.0071 | -0.0114 |
| controlled_vs_baseline_no_context | distinct1 | -0.0366 | -0.0374 |
| controlled_vs_baseline_no_context | length_score | 0.1017 | 0.2075 |
| controlled_vs_baseline_no_context | sentence_score | 0.1225 | 0.1454 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0280 | 0.4661 |
| controlled_vs_baseline_no_context | overall_quality | 0.1471 | 0.6249 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2618 | 6.6980 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1031 | 0.5339 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 0.0180 | 0.0206 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3448 | 9.7892 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0681 | 1.4156 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1307 | 1.5335 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | -0.0071 | -0.0114 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0366 | -0.0374 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.1017 | 0.2075 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.1225 | 0.1454 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0280 | 0.4661 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1471 | 0.6249 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0371 | (-0.0033, 0.1006) | 0.0603 | 0.0371 | (-0.0000, 0.0879) | 0.0280 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0203 | (-0.0761, 0.0345) | 0.7623 | -0.0203 | (-0.0662, 0.0113) | 0.8700 |
| proposed_vs_candidate_no_context | naturalness | 0.0061 | (-0.0151, 0.0269) | 0.3037 | 0.0061 | (-0.0167, 0.0241) | 0.2753 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0500 | (-0.0042, 0.1322) | 0.0883 | 0.0500 | (0.0000, 0.1169) | 0.0957 |
| proposed_vs_candidate_no_context | context_overlap | 0.0070 | (-0.0025, 0.0178) | 0.0813 | 0.0070 | (-0.0014, 0.0126) | 0.0467 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0221 | (-0.0888, 0.0395) | 0.7570 | -0.0221 | (-0.0703, 0.0156) | 0.8530 |
| proposed_vs_candidate_no_context | persona_style | -0.0129 | (-0.0688, 0.0332) | 0.7067 | -0.0129 | (-0.0703, 0.0368) | 0.7110 |
| proposed_vs_candidate_no_context | distinct1 | 0.0120 | (0.0008, 0.0257) | 0.0170 | 0.0120 | (0.0005, 0.0247) | 0.0190 |
| proposed_vs_candidate_no_context | length_score | 0.0150 | (-0.0633, 0.1017) | 0.3740 | 0.0150 | (-0.0722, 0.1106) | 0.4030 |
| proposed_vs_candidate_no_context | sentence_score | -0.0175 | (-0.0875, 0.0525) | 0.7400 | -0.0175 | (-0.1000, 0.0304) | 0.8037 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0137 | (-0.0075, 0.0396) | 0.1277 | 0.0137 | (-0.0071, 0.0379) | 0.1237 |
| proposed_vs_candidate_no_context | overall_quality | 0.0113 | (-0.0186, 0.0541) | 0.3027 | 0.0113 | (-0.0161, 0.0360) | 0.2243 |
| proposed_vs_baseline_no_context | context_relevance | 0.0224 | (-0.0268, 0.0939) | 0.2520 | 0.0224 | (-0.0290, 0.0725) | 0.1970 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0386 | (-0.0937, 0.0219) | 0.8940 | -0.0386 | (-0.1056, 0.0207) | 0.8897 |
| proposed_vs_baseline_no_context | naturalness | -0.0908 | (-0.1244, -0.0561) | 1.0000 | -0.0908 | (-0.1231, -0.0495) | 1.0000 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0322 | (-0.0319, 0.1227) | 0.2227 | 0.0322 | (-0.0374, 0.1010) | 0.1847 |
| proposed_vs_baseline_no_context | context_overlap | -0.0005 | (-0.0174, 0.0191) | 0.5343 | -0.0005 | (-0.0122, 0.0108) | 0.5357 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0217 | (-0.0783, 0.0400) | 0.7697 | -0.0217 | (-0.0796, 0.0381) | 0.7853 |
| proposed_vs_baseline_no_context | persona_style | -0.1062 | (-0.2057, -0.0167) | 1.0000 | -0.1062 | (-0.2708, 0.0000) | 1.0000 |
| proposed_vs_baseline_no_context | distinct1 | -0.0400 | (-0.0586, -0.0203) | 1.0000 | -0.0400 | (-0.0511, -0.0274) | 1.0000 |
| proposed_vs_baseline_no_context | length_score | -0.3217 | (-0.4433, -0.1983) | 1.0000 | -0.3217 | (-0.4375, -0.1745) | 1.0000 |
| proposed_vs_baseline_no_context | sentence_score | -0.1050 | (-0.1929, 0.0000) | 0.9873 | -0.1050 | (-0.2042, 0.0000) | 0.9817 |
| proposed_vs_baseline_no_context | bertscore_f1 | -0.0079 | (-0.0448, 0.0327) | 0.6653 | -0.0079 | (-0.0592, 0.0395) | 0.6483 |
| proposed_vs_baseline_no_context | overall_quality | -0.0195 | (-0.0575, 0.0273) | 0.8210 | -0.0195 | (-0.0629, 0.0210) | 0.8443 |
| controlled_vs_proposed_raw | context_relevance | 0.2394 | (0.1555, 0.3128) | 0.0000 | 0.2394 | (0.1912, 0.2972) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.1417 | (0.0721, 0.2054) | 0.0000 | 0.1417 | (0.0782, 0.2150) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.1088 | (0.0670, 0.1455) | 0.0000 | 0.1088 | (0.0488, 0.1463) | 0.0017 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.3127 | (0.1999, 0.4063) | 0.0000 | 0.3127 | (0.2475, 0.3917) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0685 | (0.0468, 0.0902) | 0.0000 | 0.0685 | (0.0537, 0.0861) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1524 | (0.0712, 0.2269) | 0.0000 | 0.1524 | (0.0836, 0.2364) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.0991 | (0.0119, 0.1974) | 0.0083 | 0.0991 | (-0.0143, 0.2755) | 0.0963 |
| controlled_vs_proposed_raw | distinct1 | 0.0035 | (-0.0179, 0.0228) | 0.3710 | 0.0035 | (-0.0282, 0.0253) | 0.4063 |
| controlled_vs_proposed_raw | length_score | 0.4233 | (0.2333, 0.5967) | 0.0000 | 0.4233 | (0.1667, 0.5958) | 0.0013 |
| controlled_vs_proposed_raw | sentence_score | 0.2275 | (0.1575, 0.2975) | 0.0000 | 0.2275 | (0.1556, 0.2800) | 0.0000 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0359 | (0.0117, 0.0591) | 0.0007 | 0.0359 | (0.0084, 0.0695) | 0.0043 |
| controlled_vs_proposed_raw | overall_quality | 0.1666 | (0.1191, 0.2068) | 0.0000 | 0.1666 | (0.1348, 0.2043) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2765 | (0.2357, 0.3227) | 0.0000 | 0.2765 | (0.2448, 0.3142) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1214 | (0.0470, 0.1955) | 0.0010 | 0.1214 | (0.0445, 0.1912) | 0.0033 |
| controlled_vs_candidate_no_context | naturalness | 0.1149 | (0.0697, 0.1549) | 0.0000 | 0.1149 | (0.0486, 0.1560) | 0.0017 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3627 | (0.3072, 0.4236) | 0.0000 | 0.3627 | (0.3201, 0.4131) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0755 | (0.0587, 0.0926) | 0.0000 | 0.0755 | (0.0616, 0.0898) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1302 | (0.0431, 0.2112) | 0.0023 | 0.1302 | (0.0431, 0.2214) | 0.0030 |
| controlled_vs_candidate_no_context | persona_style | 0.0862 | (-0.0000, 0.1932) | 0.0267 | 0.0862 | (-0.0188, 0.2190) | 0.0863 |
| controlled_vs_candidate_no_context | distinct1 | 0.0155 | (-0.0031, 0.0328) | 0.0487 | 0.0155 | (-0.0117, 0.0338) | 0.1110 |
| controlled_vs_candidate_no_context | length_score | 0.4383 | (0.2483, 0.6067) | 0.0000 | 0.4383 | (0.2110, 0.6000) | 0.0017 |
| controlled_vs_candidate_no_context | sentence_score | 0.2100 | (0.1400, 0.2800) | 0.0000 | 0.2100 | (0.0972, 0.2917) | 0.0000 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0496 | (0.0245, 0.0747) | 0.0000 | 0.0496 | (0.0258, 0.0810) | 0.0000 |
| controlled_vs_candidate_no_context | overall_quality | 0.1779 | (0.1491, 0.2048) | 0.0000 | 0.1779 | (0.1527, 0.2006) | 0.0000 |
| controlled_vs_baseline_no_context | context_relevance | 0.2618 | (0.2212, 0.3068) | 0.0000 | 0.2618 | (0.2270, 0.2977) | 0.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1031 | (0.0516, 0.1566) | 0.0000 | 0.1031 | (0.0479, 0.1700) | 0.0000 |
| controlled_vs_baseline_no_context | naturalness | 0.0180 | (-0.0311, 0.0665) | 0.2200 | 0.0180 | (-0.0292, 0.0662) | 0.2147 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3448 | (0.2924, 0.4082) | 0.0000 | 0.3448 | (0.3004, 0.3963) | 0.0000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0681 | (0.0525, 0.0842) | 0.0000 | 0.0681 | (0.0583, 0.0792) | 0.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1307 | (0.0640, 0.1960) | 0.0000 | 0.1307 | (0.0603, 0.2231) | 0.0000 |
| controlled_vs_baseline_no_context | persona_style | -0.0071 | (-0.0321, 0.0179) | 0.7103 | -0.0071 | (-0.0252, 0.0000) | 1.0000 |
| controlled_vs_baseline_no_context | distinct1 | -0.0366 | (-0.0536, -0.0170) | 1.0000 | -0.0366 | (-0.0610, -0.0167) | 1.0000 |
| controlled_vs_baseline_no_context | length_score | 0.1017 | (-0.1000, 0.3017) | 0.1587 | 0.1017 | (-0.1000, 0.3175) | 0.1493 |
| controlled_vs_baseline_no_context | sentence_score | 0.1225 | (0.0350, 0.2100) | 0.0047 | 0.1225 | (0.0206, 0.2100) | 0.0137 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0280 | (-0.0020, 0.0585) | 0.0357 | 0.0280 | (-0.0116, 0.0582) | 0.0687 |
| controlled_vs_baseline_no_context | overall_quality | 0.1471 | (0.1242, 0.1699) | 0.0000 | 0.1471 | (0.1237, 0.1713) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2618 | (0.2206, 0.3045) | 0.0000 | 0.2618 | (0.2289, 0.2972) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1031 | (0.0496, 0.1548) | 0.0000 | 0.1031 | (0.0468, 0.1753) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 0.0180 | (-0.0300, 0.0634) | 0.2357 | 0.0180 | (-0.0322, 0.0635) | 0.2143 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3448 | (0.2924, 0.4033) | 0.0000 | 0.3448 | (0.2995, 0.3963) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0681 | (0.0528, 0.0844) | 0.0000 | 0.0681 | (0.0586, 0.0789) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1307 | (0.0633, 0.1955) | 0.0000 | 0.1307 | (0.0632, 0.2156) | 0.0003 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | -0.0071 | (-0.0321, 0.0179) | 0.7263 | -0.0071 | (-0.0238, 0.0000) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0366 | (-0.0536, -0.0183) | 0.9993 | -0.0366 | (-0.0598, -0.0173) | 0.9997 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.1017 | (-0.1133, 0.3017) | 0.1773 | 0.1017 | (-0.1111, 0.3156) | 0.1530 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.1225 | (0.0350, 0.2100) | 0.0040 | 0.1225 | (0.0206, 0.2100) | 0.0163 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0280 | (-0.0028, 0.0581) | 0.0387 | 0.0280 | (-0.0109, 0.0584) | 0.0737 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1471 | (0.1253, 0.1688) | 0.0000 | 0.1471 | (0.1230, 0.1711) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 5 | 5 | 10 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | persona_consistency | 3 | 4 | 13 | 0.4750 | 0.4286 |
| proposed_vs_candidate_no_context | naturalness | 6 | 3 | 11 | 0.5750 | 0.6667 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 3 | 1 | 16 | 0.5500 | 0.7500 |
| proposed_vs_candidate_no_context | context_overlap | 6 | 4 | 10 | 0.5500 | 0.6000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 2 | 4 | 14 | 0.4500 | 0.3333 |
| proposed_vs_candidate_no_context | persona_style | 1 | 2 | 17 | 0.4750 | 0.3333 |
| proposed_vs_candidate_no_context | distinct1 | 5 | 3 | 12 | 0.5500 | 0.6250 |
| proposed_vs_candidate_no_context | length_score | 4 | 5 | 11 | 0.4750 | 0.4444 |
| proposed_vs_candidate_no_context | sentence_score | 2 | 3 | 15 | 0.4750 | 0.4000 |
| proposed_vs_candidate_no_context | bertscore_f1 | 11 | 6 | 3 | 0.6250 | 0.6471 |
| proposed_vs_candidate_no_context | overall_quality | 8 | 9 | 3 | 0.4750 | 0.4706 |
| proposed_vs_baseline_no_context | context_relevance | 8 | 11 | 1 | 0.4250 | 0.4211 |
| proposed_vs_baseline_no_context | persona_consistency | 3 | 8 | 9 | 0.3750 | 0.2727 |
| proposed_vs_baseline_no_context | naturalness | 2 | 17 | 1 | 0.1250 | 0.1053 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 4 | 5 | 11 | 0.4750 | 0.4444 |
| proposed_vs_baseline_no_context | context_overlap | 7 | 12 | 1 | 0.3750 | 0.3684 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 3 | 6 | 11 | 0.4250 | 0.3333 |
| proposed_vs_baseline_no_context | persona_style | 0 | 5 | 15 | 0.3750 | 0.0000 |
| proposed_vs_baseline_no_context | distinct1 | 2 | 15 | 3 | 0.1750 | 0.1176 |
| proposed_vs_baseline_no_context | length_score | 2 | 17 | 1 | 0.1250 | 0.1053 |
| proposed_vs_baseline_no_context | sentence_score | 2 | 8 | 10 | 0.3500 | 0.2000 |
| proposed_vs_baseline_no_context | bertscore_f1 | 7 | 13 | 0 | 0.3500 | 0.3500 |
| proposed_vs_baseline_no_context | overall_quality | 4 | 16 | 0 | 0.2000 | 0.2000 |
| controlled_vs_proposed_raw | context_relevance | 18 | 2 | 0 | 0.9000 | 0.9000 |
| controlled_vs_proposed_raw | persona_consistency | 15 | 3 | 2 | 0.8000 | 0.8333 |
| controlled_vs_proposed_raw | naturalness | 17 | 3 | 0 | 0.8500 | 0.8500 |
| controlled_vs_proposed_raw | context_keyword_coverage | 18 | 1 | 1 | 0.9250 | 0.9474 |
| controlled_vs_proposed_raw | context_overlap | 18 | 2 | 0 | 0.9000 | 0.9000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 15 | 3 | 2 | 0.8000 | 0.8333 |
| controlled_vs_proposed_raw | persona_style | 5 | 1 | 14 | 0.6000 | 0.8333 |
| controlled_vs_proposed_raw | distinct1 | 11 | 7 | 2 | 0.6000 | 0.6111 |
| controlled_vs_proposed_raw | length_score | 16 | 4 | 0 | 0.8000 | 0.8000 |
| controlled_vs_proposed_raw | sentence_score | 13 | 0 | 7 | 0.8250 | 1.0000 |
| controlled_vs_proposed_raw | bertscore_f1 | 14 | 6 | 0 | 0.7000 | 0.7000 |
| controlled_vs_proposed_raw | overall_quality | 18 | 2 | 0 | 0.9000 | 0.9000 |
| controlled_vs_candidate_no_context | context_relevance | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 13 | 3 | 4 | 0.7500 | 0.8125 |
| controlled_vs_candidate_no_context | naturalness | 17 | 3 | 0 | 0.8500 | 0.8500 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 13 | 3 | 4 | 0.7500 | 0.8125 |
| controlled_vs_candidate_no_context | persona_style | 5 | 1 | 14 | 0.6000 | 0.8333 |
| controlled_vs_candidate_no_context | distinct1 | 14 | 5 | 1 | 0.7250 | 0.7368 |
| controlled_vs_candidate_no_context | length_score | 16 | 3 | 1 | 0.8250 | 0.8421 |
| controlled_vs_candidate_no_context | sentence_score | 12 | 0 | 8 | 0.8000 | 1.0000 |
| controlled_vs_candidate_no_context | bertscore_f1 | 16 | 4 | 0 | 0.8000 | 0.8000 |
| controlled_vs_candidate_no_context | overall_quality | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | context_relevance | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 15 | 3 | 2 | 0.8000 | 0.8333 |
| controlled_vs_baseline_no_context | naturalness | 13 | 7 | 0 | 0.6500 | 0.6500 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | context_overlap | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 14 | 2 | 4 | 0.8000 | 0.8750 |
| controlled_vs_baseline_no_context | persona_style | 1 | 2 | 17 | 0.4750 | 0.3333 |
| controlled_vs_baseline_no_context | distinct1 | 3 | 15 | 2 | 0.2000 | 0.1667 |
| controlled_vs_baseline_no_context | length_score | 13 | 7 | 0 | 0.6500 | 0.6500 |
| controlled_vs_baseline_no_context | sentence_score | 8 | 1 | 11 | 0.6750 | 0.8889 |
| controlled_vs_baseline_no_context | bertscore_f1 | 12 | 8 | 0 | 0.6000 | 0.6000 |
| controlled_vs_baseline_no_context | overall_quality | 20 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 20 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 15 | 3 | 2 | 0.8000 | 0.8333 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 13 | 7 | 0 | 0.6500 | 0.6500 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 20 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 20 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 14 | 2 | 4 | 0.8000 | 0.8750 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 1 | 2 | 17 | 0.4750 | 0.3333 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | 3 | 15 | 2 | 0.2000 | 0.1667 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 13 | 7 | 0 | 0.6500 | 0.6500 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 8 | 1 | 11 | 0.6750 | 0.8889 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 12 | 8 | 0 | 0.6000 | 0.6000 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 20 | 0 | 0 | 1.0000 | 1.0000 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.3500 | 0.3500 | 0.6500 |
| proposed_contextual | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.6500 | 0.0000 | 0.0000 |
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
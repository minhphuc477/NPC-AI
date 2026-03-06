# Proposal Alignment Evaluation Report

- Run ID: `20260305T234544Z`
- Generated: `2026-03-05T23:50:50.568691+00:00`
- Scenarios: `artifacts\proposal_control_tuning\architecture_compare_seeded\20260305T233017Z\intent_focus_adaptive\seed_31\20260305T234544Z\scenarios.jsonl`
- Scenario count: `24`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_alt`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off
- `baseline_no_context`: model `phi3:mini`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2741 (0.2469, 0.3037) | 0.3077 (0.2622, 0.3552) | 0.8879 (0.8584, 0.9137) | 0.4101 (0.3878, 0.4344) | n/a |
| proposed_contextual_controlled_alt | 0.2743 (0.2440, 0.3072) | 0.3837 (0.3242, 0.4498) | 0.9030 (0.8799, 0.9245) | 0.4413 (0.4191, 0.4632) | n/a |
| proposed_contextual | 0.0923 (0.0552, 0.1372) | 0.1710 (0.1234, 0.2228) | 0.8344 (0.8017, 0.8685) | 0.2644 (0.2311, 0.3013) | n/a |
| candidate_no_context | 0.0189 (0.0115, 0.0278) | 0.1721 (0.1209, 0.2381) | 0.8403 (0.8061, 0.8722) | 0.2319 (0.2084, 0.2591) | n/a |
| baseline_no_context | 0.0431 (0.0261, 0.0621) | 0.1983 (0.1409, 0.2628) | 0.8916 (0.8709, 0.9121) | 0.2631 (0.2402, 0.2890) | n/a |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0734 | 3.8802 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0011 | -0.0066 |
| proposed_vs_candidate_no_context | naturalness | -0.0059 | -0.0070 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0928 | 8.1667 |
| proposed_vs_candidate_no_context | context_overlap | 0.0281 | 0.7693 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0040 | 0.0533 |
| proposed_vs_candidate_no_context | persona_style | -0.0215 | -0.0382 |
| proposed_vs_candidate_no_context | distinct1 | -0.0018 | -0.0020 |
| proposed_vs_candidate_no_context | length_score | -0.0111 | -0.0271 |
| proposed_vs_candidate_no_context | sentence_score | -0.0292 | -0.0366 |
| proposed_vs_candidate_no_context | overall_quality | 0.0326 | 0.1405 |
| proposed_vs_baseline_no_context | context_relevance | 0.0492 | 1.1428 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0273 | -0.1379 |
| proposed_vs_baseline_no_context | naturalness | -0.0572 | -0.0642 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0603 | 1.3741 |
| proposed_vs_baseline_no_context | context_overlap | 0.0234 | 0.5682 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0266 | -0.2533 |
| proposed_vs_baseline_no_context | persona_style | -0.0304 | -0.0532 |
| proposed_vs_baseline_no_context | distinct1 | -0.0321 | -0.0329 |
| proposed_vs_baseline_no_context | length_score | -0.1708 | -0.3000 |
| proposed_vs_baseline_no_context | sentence_score | -0.1021 | -0.1175 |
| proposed_vs_baseline_no_context | overall_quality | 0.0013 | 0.0049 |
| controlled_vs_proposed_raw | context_relevance | 0.1818 | 1.9698 |
| controlled_vs_proposed_raw | persona_consistency | 0.1367 | 0.7997 |
| controlled_vs_proposed_raw | naturalness | 0.0535 | 0.0641 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2417 | 2.3200 |
| controlled_vs_proposed_raw | context_overlap | 0.0422 | 0.6530 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1524 | 1.9443 |
| controlled_vs_proposed_raw | persona_style | 0.0740 | 0.1367 |
| controlled_vs_proposed_raw | distinct1 | -0.0093 | -0.0099 |
| controlled_vs_proposed_raw | length_score | 0.2194 | 0.5505 |
| controlled_vs_proposed_raw | sentence_score | 0.1333 | 0.1739 |
| controlled_vs_proposed_raw | overall_quality | 0.1456 | 0.5508 |
| controlled_vs_candidate_no_context | context_relevance | 0.2552 | 13.4931 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1356 | 0.7879 |
| controlled_vs_candidate_no_context | naturalness | 0.0476 | 0.0567 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3345 | 29.4333 |
| controlled_vs_candidate_no_context | context_overlap | 0.0703 | 1.9247 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1563 | 2.1013 |
| controlled_vs_candidate_no_context | persona_style | 0.0525 | 0.0933 |
| controlled_vs_candidate_no_context | distinct1 | -0.0112 | -0.0118 |
| controlled_vs_candidate_no_context | length_score | 0.2083 | 0.5085 |
| controlled_vs_candidate_no_context | sentence_score | 0.1042 | 0.1309 |
| controlled_vs_candidate_no_context | overall_quality | 0.1782 | 0.7686 |
| controlled_vs_baseline_no_context | context_relevance | 0.2311 | 5.3636 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1094 | 0.5515 |
| controlled_vs_baseline_no_context | naturalness | -0.0037 | -0.0042 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3020 | 6.8820 |
| controlled_vs_baseline_no_context | context_overlap | 0.0656 | 1.5923 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1258 | 1.1985 |
| controlled_vs_baseline_no_context | persona_style | 0.0436 | 0.0763 |
| controlled_vs_baseline_no_context | distinct1 | -0.0414 | -0.0424 |
| controlled_vs_baseline_no_context | length_score | 0.0486 | 0.0854 |
| controlled_vs_baseline_no_context | sentence_score | 0.0312 | 0.0360 |
| controlled_vs_baseline_no_context | overall_quality | 0.1469 | 0.5585 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0001 | 0.0005 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0761 | 0.2473 |
| controlled_alt_vs_controlled_default | naturalness | 0.0151 | 0.0170 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0048 | -0.0139 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0117 | 0.1091 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0956 | 0.4144 |
| controlled_alt_vs_controlled_default | persona_style | -0.0022 | -0.0035 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0050 | 0.0053 |
| controlled_alt_vs_controlled_default | length_score | 0.0667 | 0.1079 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0021 | -0.0023 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0312 | 0.0762 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1820 | 1.9713 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.2128 | 1.2447 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0686 | 0.0822 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2369 | 2.2739 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0539 | 0.8333 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.2480 | 3.1646 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0718 | 0.1327 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0043 | -0.0046 |
| controlled_alt_vs_proposed_raw | length_score | 0.2861 | 0.7178 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1313 | 0.1712 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1769 | 0.6690 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2554 | 13.5003 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.2117 | 1.2300 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0627 | 0.0747 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3297 | 29.0111 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0820 | 2.2437 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.2520 | 3.3867 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0503 | 0.0894 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0062 | -0.0065 |
| controlled_alt_vs_candidate_no_context | length_score | 0.2750 | 0.6712 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1021 | 0.1283 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.2095 | 0.9034 |
| controlled_alt_vs_baseline_no_context | context_relevance | 0.2312 | 5.3668 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 0.1854 | 0.9351 |
| controlled_alt_vs_baseline_no_context | naturalness | 0.0114 | 0.0128 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 0.2972 | 6.7727 |
| controlled_alt_vs_baseline_no_context | context_overlap | 0.0773 | 1.8750 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 0.2214 | 2.1096 |
| controlled_alt_vs_baseline_no_context | persona_style | 0.0414 | 0.0725 |
| controlled_alt_vs_baseline_no_context | distinct1 | -0.0365 | -0.0373 |
| controlled_alt_vs_baseline_no_context | length_score | 0.1153 | 0.2024 |
| controlled_alt_vs_baseline_no_context | sentence_score | 0.0292 | 0.0336 |
| controlled_alt_vs_baseline_no_context | overall_quality | 0.1782 | 0.6772 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2311 | 5.3636 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1094 | 0.5515 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0037 | -0.0042 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3020 | 6.8820 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0656 | 1.5923 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1258 | 1.1985 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0436 | 0.0763 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0414 | -0.0424 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.0486 | 0.0854 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0312 | 0.0360 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1469 | 0.5585 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0734 | (0.0316, 0.1194) | 0.0000 | 0.0734 | (0.0220, 0.1142) | 0.0010 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0011 | (-0.0704, 0.0605) | 0.4943 | -0.0011 | (-0.0927, 0.0634) | 0.4790 |
| proposed_vs_candidate_no_context | naturalness | -0.0059 | (-0.0417, 0.0314) | 0.6147 | -0.0059 | (-0.0440, 0.0259) | 0.6287 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0928 | (0.0401, 0.1496) | 0.0000 | 0.0928 | (0.0303, 0.1446) | 0.0030 |
| proposed_vs_candidate_no_context | context_overlap | 0.0281 | (0.0160, 0.0415) | 0.0000 | 0.0281 | (0.0117, 0.0416) | 0.0007 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0040 | (-0.0734, 0.0784) | 0.4597 | 0.0040 | (-0.0929, 0.0785) | 0.4670 |
| proposed_vs_candidate_no_context | persona_style | -0.0215 | (-0.0851, 0.0351) | 0.7470 | -0.0215 | (-0.1114, 0.0437) | 0.7247 |
| proposed_vs_candidate_no_context | distinct1 | -0.0018 | (-0.0159, 0.0108) | 0.5797 | -0.0018 | (-0.0201, 0.0132) | 0.6070 |
| proposed_vs_candidate_no_context | length_score | -0.0111 | (-0.1417, 0.1306) | 0.5710 | -0.0111 | (-0.1476, 0.1050) | 0.5700 |
| proposed_vs_candidate_no_context | sentence_score | -0.0292 | (-0.1313, 0.0729) | 0.7720 | -0.0292 | (-0.1333, 0.0519) | 0.7893 |
| proposed_vs_candidate_no_context | overall_quality | 0.0326 | (-0.0020, 0.0654) | 0.0353 | 0.0326 | (-0.0134, 0.0720) | 0.0813 |
| proposed_vs_baseline_no_context | context_relevance | 0.0492 | (0.0073, 0.0965) | 0.0090 | 0.0492 | (-0.0110, 0.0991) | 0.0520 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0273 | (-0.1055, 0.0450) | 0.7627 | -0.0273 | (-0.1231, 0.0479) | 0.7597 |
| proposed_vs_baseline_no_context | naturalness | -0.0572 | (-0.0937, -0.0204) | 0.9983 | -0.0572 | (-0.1020, -0.0164) | 0.9973 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0603 | (0.0035, 0.1209) | 0.0203 | 0.0603 | (-0.0216, 0.1221) | 0.0703 |
| proposed_vs_baseline_no_context | context_overlap | 0.0234 | (0.0078, 0.0394) | 0.0020 | 0.0234 | (0.0058, 0.0407) | 0.0043 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0266 | (-0.1060, 0.0498) | 0.7307 | -0.0266 | (-0.1338, 0.0643) | 0.7247 |
| proposed_vs_baseline_no_context | persona_style | -0.0304 | (-0.1377, 0.0614) | 0.7323 | -0.0304 | (-0.1784, 0.0804) | 0.6613 |
| proposed_vs_baseline_no_context | distinct1 | -0.0321 | (-0.0479, -0.0163) | 1.0000 | -0.0321 | (-0.0523, -0.0099) | 0.9983 |
| proposed_vs_baseline_no_context | length_score | -0.1708 | (-0.3097, -0.0292) | 0.9913 | -0.1708 | (-0.3528, -0.0083) | 0.9807 |
| proposed_vs_baseline_no_context | sentence_score | -0.1021 | (-0.1896, -0.0146) | 0.9950 | -0.1021 | (-0.1842, -0.0218) | 0.9960 |
| proposed_vs_baseline_no_context | overall_quality | 0.0013 | (-0.0409, 0.0420) | 0.4817 | 0.0013 | (-0.0584, 0.0522) | 0.5107 |
| controlled_vs_proposed_raw | context_relevance | 0.1818 | (0.1392, 0.2227) | 0.0000 | 0.1818 | (0.1422, 0.2185) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.1367 | (0.0846, 0.1856) | 0.0000 | 0.1367 | (0.0723, 0.2019) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0535 | (0.0088, 0.0979) | 0.0103 | 0.0535 | (-0.0007, 0.1078) | 0.0260 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2417 | (0.1845, 0.2953) | 0.0000 | 0.2417 | (0.1841, 0.2871) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0422 | (0.0257, 0.0590) | 0.0000 | 0.0422 | (0.0279, 0.0597) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1524 | (0.0905, 0.2068) | 0.0000 | 0.1524 | (0.0762, 0.2208) | 0.0003 |
| controlled_vs_proposed_raw | persona_style | 0.0740 | (-0.0015, 0.1589) | 0.0453 | 0.0740 | (-0.0028, 0.2292) | 0.0993 |
| controlled_vs_proposed_raw | distinct1 | -0.0093 | (-0.0230, 0.0047) | 0.9090 | -0.0093 | (-0.0188, 0.0038) | 0.9293 |
| controlled_vs_proposed_raw | length_score | 0.2194 | (0.0361, 0.4056) | 0.0113 | 0.2194 | (-0.0303, 0.4435) | 0.0407 |
| controlled_vs_proposed_raw | sentence_score | 0.1333 | (0.0312, 0.2208) | 0.0043 | 0.1333 | (0.0395, 0.2208) | 0.0000 |
| controlled_vs_proposed_raw | overall_quality | 0.1456 | (0.1094, 0.1791) | 0.0000 | 0.1456 | (0.1003, 0.1884) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2552 | (0.2245, 0.2888) | 0.0000 | 0.2552 | (0.2190, 0.2836) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1356 | (0.0495, 0.2102) | 0.0020 | 0.1356 | (0.0069, 0.2239) | 0.0167 |
| controlled_vs_candidate_no_context | naturalness | 0.0476 | (0.0048, 0.0907) | 0.0130 | 0.0476 | (0.0047, 0.0838) | 0.0143 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3345 | (0.2929, 0.3788) | 0.0000 | 0.3345 | (0.2837, 0.3740) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0703 | (0.0573, 0.0847) | 0.0000 | 0.0703 | (0.0565, 0.0843) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1563 | (0.0591, 0.2393) | 0.0013 | 0.1563 | (0.0109, 0.2501) | 0.0190 |
| controlled_vs_candidate_no_context | persona_style | 0.0525 | (-0.0343, 0.1419) | 0.1217 | 0.0525 | (-0.0620, 0.2010) | 0.1923 |
| controlled_vs_candidate_no_context | distinct1 | -0.0112 | (-0.0266, 0.0044) | 0.9223 | -0.0112 | (-0.0253, 0.0027) | 0.9350 |
| controlled_vs_candidate_no_context | length_score | 0.2083 | (0.0472, 0.3667) | 0.0047 | 0.2083 | (0.0317, 0.3530) | 0.0120 |
| controlled_vs_candidate_no_context | sentence_score | 0.1042 | (-0.0104, 0.2062) | 0.0347 | 0.1042 | (-0.0105, 0.1875) | 0.0317 |
| controlled_vs_candidate_no_context | overall_quality | 0.1782 | (0.1391, 0.2143) | 0.0000 | 0.1782 | (0.1166, 0.2141) | 0.0000 |
| controlled_vs_baseline_no_context | context_relevance | 0.2311 | (0.1984, 0.2649) | 0.0000 | 0.2311 | (0.1822, 0.2719) | 0.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1094 | (0.0357, 0.1778) | 0.0007 | 0.1094 | (-0.0071, 0.1802) | 0.0333 |
| controlled_vs_baseline_no_context | naturalness | -0.0037 | (-0.0415, 0.0310) | 0.5650 | -0.0037 | (-0.0400, 0.0265) | 0.5767 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3020 | (0.2576, 0.3452) | 0.0000 | 0.3020 | (0.2287, 0.3591) | 0.0000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0656 | (0.0524, 0.0811) | 0.0000 | 0.0656 | (0.0504, 0.0809) | 0.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1258 | (0.0426, 0.2071) | 0.0017 | 0.1258 | (-0.0201, 0.2215) | 0.0343 |
| controlled_vs_baseline_no_context | persona_style | 0.0436 | (0.0000, 0.0972) | 0.0437 | 0.0436 | (0.0000, 0.1163) | 0.0943 |
| controlled_vs_baseline_no_context | distinct1 | -0.0414 | (-0.0563, -0.0261) | 1.0000 | -0.0414 | (-0.0569, -0.0239) | 1.0000 |
| controlled_vs_baseline_no_context | length_score | 0.0486 | (-0.1069, 0.1958) | 0.2740 | 0.0486 | (-0.0877, 0.1644) | 0.2267 |
| controlled_vs_baseline_no_context | sentence_score | 0.0312 | (-0.0667, 0.1313) | 0.2463 | 0.0312 | (-0.0800, 0.1273) | 0.2720 |
| controlled_vs_baseline_no_context | overall_quality | 0.1469 | (0.1111, 0.1815) | 0.0000 | 0.1469 | (0.0900, 0.1892) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0001 | (-0.0303, 0.0260) | 0.4893 | 0.0001 | (-0.0221, 0.0275) | 0.4757 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0761 | (0.0146, 0.1523) | 0.0027 | 0.0761 | (0.0064, 0.1747) | 0.0127 |
| controlled_alt_vs_controlled_default | naturalness | 0.0151 | (-0.0260, 0.0565) | 0.2410 | 0.0151 | (-0.0076, 0.0510) | 0.1313 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0048 | (-0.0431, 0.0292) | 0.5867 | -0.0048 | (-0.0364, 0.0339) | 0.6100 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0117 | (-0.0027, 0.0283) | 0.0583 | 0.0117 | (0.0001, 0.0237) | 0.0233 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0956 | (0.0218, 0.1843) | 0.0037 | 0.0956 | (0.0095, 0.2199) | 0.0113 |
| controlled_alt_vs_controlled_default | persona_style | -0.0022 | (-0.0065, 0.0000) | 1.0000 | -0.0022 | (-0.0068, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0050 | (-0.0096, 0.0202) | 0.2617 | 0.0050 | (-0.0133, 0.0173) | 0.2547 |
| controlled_alt_vs_controlled_default | length_score | 0.0667 | (-0.1042, 0.2306) | 0.2297 | 0.0667 | (-0.0519, 0.2369) | 0.1677 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0021 | (-0.1042, 0.1083) | 0.5610 | -0.0021 | (-0.0840, 0.0975) | 0.5700 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0312 | (0.0030, 0.0606) | 0.0137 | 0.0312 | (-0.0016, 0.0752) | 0.0313 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1820 | (0.1323, 0.2263) | 0.0000 | 0.1820 | (0.1441, 0.2287) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.2128 | (0.1429, 0.2887) | 0.0000 | 0.2128 | (0.1378, 0.3139) | 0.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0686 | (0.0330, 0.1040) | 0.0000 | 0.0686 | (0.0333, 0.1096) | 0.0003 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2369 | (0.1739, 0.2945) | 0.0000 | 0.2369 | (0.1840, 0.2992) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0539 | (0.0357, 0.0718) | 0.0000 | 0.0539 | (0.0377, 0.0711) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.2480 | (0.1673, 0.3379) | 0.0000 | 0.2480 | (0.1643, 0.3605) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0718 | (-0.0037, 0.1674) | 0.0440 | 0.0718 | (-0.0065, 0.2113) | 0.1037 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0043 | (-0.0187, 0.0097) | 0.7107 | -0.0043 | (-0.0174, 0.0093) | 0.7557 |
| controlled_alt_vs_proposed_raw | length_score | 0.2861 | (0.1236, 0.4458) | 0.0003 | 0.2861 | (0.1256, 0.4639) | 0.0003 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1313 | (0.0292, 0.2188) | 0.0070 | 0.1313 | (0.0389, 0.2167) | 0.0050 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1769 | (0.1362, 0.2176) | 0.0000 | 0.1769 | (0.1286, 0.2345) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2554 | (0.2231, 0.2908) | 0.0000 | 0.2554 | (0.2293, 0.2847) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.2117 | (0.1270, 0.2968) | 0.0000 | 0.2117 | (0.1093, 0.3154) | 0.0003 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0627 | (0.0258, 0.0994) | 0.0007 | 0.0627 | (0.0271, 0.1013) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3297 | (0.2874, 0.3727) | 0.0000 | 0.3297 | (0.2965, 0.3668) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0820 | (0.0655, 0.1005) | 0.0000 | 0.0820 | (0.0613, 0.1021) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.2520 | (0.1581, 0.3512) | 0.0000 | 0.2520 | (0.1370, 0.3577) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0503 | (-0.0330, 0.1380) | 0.1263 | 0.0503 | (-0.0616, 0.1945) | 0.1847 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0062 | (-0.0227, 0.0091) | 0.7667 | -0.0062 | (-0.0263, 0.0099) | 0.7810 |
| controlled_alt_vs_candidate_no_context | length_score | 0.2750 | (0.1208, 0.4209) | 0.0000 | 0.2750 | (0.1424, 0.4286) | 0.0000 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1021 | (0.0000, 0.1896) | 0.0270 | 0.1021 | (0.0000, 0.1931) | 0.0320 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.2095 | (0.1759, 0.2435) | 0.0000 | 0.2095 | (0.1682, 0.2465) | 0.0000 |
| controlled_alt_vs_baseline_no_context | context_relevance | 0.2312 | (0.2005, 0.2660) | 0.0000 | 0.2312 | (0.2010, 0.2546) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 0.1854 | (0.1064, 0.2642) | 0.0000 | 0.1854 | (0.1079, 0.2497) | 0.0000 |
| controlled_alt_vs_baseline_no_context | naturalness | 0.0114 | (-0.0190, 0.0377) | 0.2067 | 0.0114 | (-0.0122, 0.0337) | 0.1537 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 0.2972 | (0.2562, 0.3434) | 0.0000 | 0.2972 | (0.2556, 0.3325) | 0.0000 |
| controlled_alt_vs_baseline_no_context | context_overlap | 0.0773 | (0.0596, 0.0968) | 0.0000 | 0.0773 | (0.0588, 0.0981) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 0.2214 | (0.1188, 0.3292) | 0.0000 | 0.2214 | (0.1141, 0.3117) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_style | 0.0414 | (0.0000, 0.0911) | 0.0390 | 0.0414 | (0.0000, 0.1102) | 0.1037 |
| controlled_alt_vs_baseline_no_context | distinct1 | -0.0365 | (-0.0511, -0.0220) | 1.0000 | -0.0365 | (-0.0526, -0.0194) | 1.0000 |
| controlled_alt_vs_baseline_no_context | length_score | 0.1153 | (-0.0278, 0.2458) | 0.0560 | 0.1153 | (0.0074, 0.2259) | 0.0210 |
| controlled_alt_vs_baseline_no_context | sentence_score | 0.0292 | (-0.0583, 0.1167) | 0.3117 | 0.0292 | (-0.0648, 0.1226) | 0.3233 |
| controlled_alt_vs_baseline_no_context | overall_quality | 0.1782 | (0.1425, 0.2117) | 0.0000 | 0.1782 | (0.1444, 0.2039) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2311 | (0.1981, 0.2637) | 0.0000 | 0.2311 | (0.1808, 0.2714) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1094 | (0.0384, 0.1729) | 0.0010 | 0.1094 | (-0.0022, 0.1818) | 0.0280 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0037 | (-0.0412, 0.0308) | 0.5653 | -0.0037 | (-0.0410, 0.0266) | 0.5897 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3020 | (0.2574, 0.3477) | 0.0000 | 0.3020 | (0.2324, 0.3636) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0656 | (0.0520, 0.0810) | 0.0000 | 0.0656 | (0.0516, 0.0816) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1258 | (0.0375, 0.2107) | 0.0010 | 0.1258 | (-0.0198, 0.2233) | 0.0357 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0436 | (0.0000, 0.0971) | 0.0400 | 0.0436 | (0.0000, 0.1155) | 0.1030 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0414 | (-0.0558, -0.0261) | 1.0000 | -0.0414 | (-0.0569, -0.0236) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.0486 | (-0.1042, 0.1972) | 0.2753 | 0.0486 | (-0.0797, 0.1611) | 0.2200 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0312 | (-0.0667, 0.1187) | 0.2550 | 0.0312 | (-0.0794, 0.1273) | 0.2733 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1469 | (0.1115, 0.1810) | 0.0000 | 0.1469 | (0.0900, 0.1881) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 15 | 2 | 7 | 0.7708 | 0.8824 |
| proposed_vs_candidate_no_context | persona_consistency | 8 | 5 | 11 | 0.5625 | 0.6154 |
| proposed_vs_candidate_no_context | naturalness | 6 | 11 | 7 | 0.3958 | 0.3529 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 10 | 1 | 13 | 0.6875 | 0.9091 |
| proposed_vs_candidate_no_context | context_overlap | 15 | 2 | 7 | 0.7708 | 0.8824 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 6 | 4 | 14 | 0.5417 | 0.6000 |
| proposed_vs_candidate_no_context | persona_style | 2 | 3 | 19 | 0.4792 | 0.4000 |
| proposed_vs_candidate_no_context | distinct1 | 9 | 6 | 9 | 0.5625 | 0.6000 |
| proposed_vs_candidate_no_context | length_score | 9 | 8 | 7 | 0.5208 | 0.5294 |
| proposed_vs_candidate_no_context | sentence_score | 5 | 7 | 12 | 0.4583 | 0.4167 |
| proposed_vs_candidate_no_context | overall_quality | 11 | 6 | 7 | 0.6042 | 0.6471 |
| proposed_vs_baseline_no_context | context_relevance | 14 | 10 | 0 | 0.5833 | 0.5833 |
| proposed_vs_baseline_no_context | persona_consistency | 8 | 7 | 9 | 0.5208 | 0.5333 |
| proposed_vs_baseline_no_context | naturalness | 8 | 16 | 0 | 0.3333 | 0.3333 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 10 | 5 | 9 | 0.6042 | 0.6667 |
| proposed_vs_baseline_no_context | context_overlap | 17 | 7 | 0 | 0.7083 | 0.7083 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 5 | 6 | 13 | 0.4792 | 0.4545 |
| proposed_vs_baseline_no_context | persona_style | 3 | 3 | 18 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context | distinct1 | 4 | 16 | 4 | 0.2500 | 0.2000 |
| proposed_vs_baseline_no_context | length_score | 8 | 15 | 1 | 0.3542 | 0.3478 |
| proposed_vs_baseline_no_context | sentence_score | 2 | 9 | 13 | 0.3542 | 0.1818 |
| proposed_vs_baseline_no_context | overall_quality | 12 | 12 | 0 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | context_relevance | 22 | 2 | 0 | 0.9167 | 0.9167 |
| controlled_vs_proposed_raw | persona_consistency | 20 | 2 | 2 | 0.8750 | 0.9091 |
| controlled_vs_proposed_raw | naturalness | 17 | 7 | 0 | 0.7083 | 0.7083 |
| controlled_vs_proposed_raw | context_keyword_coverage | 22 | 2 | 0 | 0.9167 | 0.9167 |
| controlled_vs_proposed_raw | context_overlap | 20 | 4 | 0 | 0.8333 | 0.8333 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 20 | 2 | 2 | 0.8750 | 0.9091 |
| controlled_vs_proposed_raw | persona_style | 3 | 1 | 20 | 0.5417 | 0.7500 |
| controlled_vs_proposed_raw | distinct1 | 7 | 15 | 2 | 0.3333 | 0.3182 |
| controlled_vs_proposed_raw | length_score | 17 | 6 | 1 | 0.7292 | 0.7391 |
| controlled_vs_proposed_raw | sentence_score | 12 | 3 | 9 | 0.6875 | 0.8000 |
| controlled_vs_proposed_raw | overall_quality | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_vs_candidate_no_context | context_relevance | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 19 | 3 | 2 | 0.8333 | 0.8636 |
| controlled_vs_candidate_no_context | naturalness | 18 | 6 | 0 | 0.7500 | 0.7500 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 19 | 3 | 2 | 0.8333 | 0.8636 |
| controlled_vs_candidate_no_context | persona_style | 5 | 2 | 17 | 0.5625 | 0.7143 |
| controlled_vs_candidate_no_context | distinct1 | 8 | 14 | 2 | 0.3750 | 0.3636 |
| controlled_vs_candidate_no_context | length_score | 17 | 6 | 1 | 0.7292 | 0.7391 |
| controlled_vs_candidate_no_context | sentence_score | 11 | 3 | 10 | 0.6667 | 0.7857 |
| controlled_vs_candidate_no_context | overall_quality | 22 | 2 | 0 | 0.9167 | 0.9167 |
| controlled_vs_baseline_no_context | context_relevance | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 17 | 4 | 3 | 0.7708 | 0.8095 |
| controlled_vs_baseline_no_context | naturalness | 16 | 8 | 0 | 0.6667 | 0.6667 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | context_overlap | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 17 | 4 | 3 | 0.7708 | 0.8095 |
| controlled_vs_baseline_no_context | persona_style | 3 | 0 | 21 | 0.5625 | 1.0000 |
| controlled_vs_baseline_no_context | distinct1 | 4 | 20 | 0 | 0.1667 | 0.1667 |
| controlled_vs_baseline_no_context | length_score | 14 | 9 | 1 | 0.6042 | 0.6087 |
| controlled_vs_baseline_no_context | sentence_score | 6 | 3 | 15 | 0.5625 | 0.6667 |
| controlled_vs_baseline_no_context | overall_quality | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_alt_vs_controlled_default | context_relevance | 14 | 6 | 4 | 0.6667 | 0.7000 |
| controlled_alt_vs_controlled_default | persona_consistency | 6 | 2 | 16 | 0.5833 | 0.7500 |
| controlled_alt_vs_controlled_default | naturalness | 11 | 9 | 4 | 0.5417 | 0.5500 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 7 | 4 | 13 | 0.5625 | 0.6364 |
| controlled_alt_vs_controlled_default | context_overlap | 12 | 8 | 4 | 0.5833 | 0.6000 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 6 | 1 | 17 | 0.6042 | 0.8571 |
| controlled_alt_vs_controlled_default | persona_style | 0 | 1 | 23 | 0.4792 | 0.0000 |
| controlled_alt_vs_controlled_default | distinct1 | 12 | 8 | 4 | 0.5833 | 0.6000 |
| controlled_alt_vs_controlled_default | length_score | 12 | 8 | 4 | 0.5833 | 0.6000 |
| controlled_alt_vs_controlled_default | sentence_score | 5 | 6 | 13 | 0.4792 | 0.4545 |
| controlled_alt_vs_controlled_default | overall_quality | 14 | 6 | 4 | 0.6667 | 0.7000 |
| controlled_alt_vs_proposed_raw | context_relevance | 22 | 2 | 0 | 0.9167 | 0.9167 |
| controlled_alt_vs_proposed_raw | persona_consistency | 21 | 1 | 2 | 0.9167 | 0.9545 |
| controlled_alt_vs_proposed_raw | naturalness | 19 | 5 | 0 | 0.7917 | 0.7917 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 20 | 2 | 2 | 0.8750 | 0.9091 |
| controlled_alt_vs_proposed_raw | context_overlap | 20 | 4 | 0 | 0.8333 | 0.8333 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 21 | 1 | 2 | 0.9167 | 0.9545 |
| controlled_alt_vs_proposed_raw | persona_style | 3 | 1 | 20 | 0.5417 | 0.7500 |
| controlled_alt_vs_proposed_raw | distinct1 | 10 | 10 | 4 | 0.5000 | 0.5000 |
| controlled_alt_vs_proposed_raw | length_score | 19 | 5 | 0 | 0.7917 | 0.7917 |
| controlled_alt_vs_proposed_raw | sentence_score | 12 | 3 | 9 | 0.6875 | 0.8000 |
| controlled_alt_vs_proposed_raw | overall_quality | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 21 | 3 | 0 | 0.8750 | 0.8750 |
| controlled_alt_vs_candidate_no_context | naturalness | 18 | 6 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 21 | 2 | 1 | 0.8958 | 0.9130 |
| controlled_alt_vs_candidate_no_context | persona_style | 5 | 2 | 17 | 0.5625 | 0.7143 |
| controlled_alt_vs_candidate_no_context | distinct1 | 9 | 12 | 3 | 0.4375 | 0.4286 |
| controlled_alt_vs_candidate_no_context | length_score | 19 | 5 | 0 | 0.7917 | 0.7917 |
| controlled_alt_vs_candidate_no_context | sentence_score | 10 | 3 | 11 | 0.6458 | 0.7692 |
| controlled_alt_vs_candidate_no_context | overall_quality | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context | context_relevance | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 19 | 3 | 2 | 0.8333 | 0.8636 |
| controlled_alt_vs_baseline_no_context | naturalness | 16 | 8 | 0 | 0.6667 | 0.6667 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context | context_overlap | 22 | 2 | 0 | 0.9167 | 0.9167 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 19 | 3 | 2 | 0.8333 | 0.8636 |
| controlled_alt_vs_baseline_no_context | persona_style | 3 | 0 | 21 | 0.5625 | 1.0000 |
| controlled_alt_vs_baseline_no_context | distinct1 | 4 | 19 | 1 | 0.1875 | 0.1739 |
| controlled_alt_vs_baseline_no_context | length_score | 15 | 7 | 2 | 0.6667 | 0.6818 |
| controlled_alt_vs_baseline_no_context | sentence_score | 6 | 4 | 14 | 0.5417 | 0.6000 |
| controlled_alt_vs_baseline_no_context | overall_quality | 24 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 24 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 17 | 4 | 3 | 0.7708 | 0.8095 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 16 | 8 | 0 | 0.6667 | 0.6667 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 24 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 24 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 17 | 4 | 3 | 0.7708 | 0.8095 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 3 | 0 | 21 | 0.5625 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | 4 | 20 | 0 | 0.1667 | 0.1667 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 14 | 9 | 1 | 0.6042 | 0.6087 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 6 | 3 | 15 | 0.5625 | 0.6667 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 23 | 1 | 0 | 0.9583 | 0.9583 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.3750 | 0.4583 | 0.5417 |
| proposed_contextual_controlled_alt | 0.0000 | 0.0000 | 0.3333 | 0.3750 | 0.6250 |
| proposed_contextual | 0.0000 | 0.0000 | 0.4167 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.4167 | 0.0000 | 0.0000 |
| baseline_no_context | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `8`
- Unique template signatures: `23`
- Template signature ratio: `0.9583`
- Effective sample size by source clustering: `7.20`
- Effective sample size by template-signature clustering: `22.15`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: unavailable (disabled_by_flag).

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
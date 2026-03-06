# Proposal Alignment Evaluation Report

- Run ID: `20260306T162445Z`
- Generated: `2026-03-06T16:28:08.704654+00:00`
- Scenarios: `artifacts\proposal_control_tuning\architecture_compare_fast\20260306T162109Z\hybrid_balanced\seed_29\attempt_1\20260306T162445Z\scenarios.jsonl`
- Scenario count: `20`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_alt`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2533 (0.2235, 0.2830) | 0.2814 (0.2309, 0.3326) | 0.9179 (0.8963, 0.9359) | 0.3961 (0.3726, 0.4188) | n/a |
| proposed_contextual_controlled_alt | 0.2600 (0.2272, 0.2937) | 0.2715 (0.2230, 0.3215) | 0.9083 (0.8825, 0.9316) | 0.3941 (0.3695, 0.4189) | n/a |
| proposed_contextual | 0.0667 (0.0308, 0.1097) | 0.1289 (0.0915, 0.1705) | 0.8098 (0.7788, 0.8433) | 0.2319 (0.2025, 0.2653) | n/a |
| candidate_no_context | 0.0321 (0.0169, 0.0514) | 0.1804 (0.1174, 0.2514) | 0.8141 (0.7811, 0.8510) | 0.2357 (0.2067, 0.2702) | n/a |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0347 | 1.0803 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0514 | -0.2852 |
| proposed_vs_candidate_no_context | naturalness | -0.0043 | -0.0053 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0451 | 1.7000 |
| proposed_vs_candidate_no_context | context_overlap | 0.0104 | 0.2298 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0590 | -0.6986 |
| proposed_vs_candidate_no_context | persona_style | -0.0210 | -0.0372 |
| proposed_vs_candidate_no_context | distinct1 | -0.0095 | -0.0101 |
| proposed_vs_candidate_no_context | length_score | 0.0150 | 0.0506 |
| proposed_vs_candidate_no_context | sentence_score | -0.0350 | -0.0453 |
| proposed_vs_candidate_no_context | overall_quality | -0.0037 | -0.0159 |
| controlled_vs_proposed_raw | context_relevance | 0.1866 | 2.7956 |
| controlled_vs_proposed_raw | persona_consistency | 0.1525 | 1.1826 |
| controlled_vs_proposed_raw | naturalness | 0.1081 | 0.1335 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2438 | 3.4053 |
| controlled_vs_proposed_raw | context_overlap | 0.0531 | 0.9583 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1700 | 6.6729 |
| controlled_vs_proposed_raw | persona_style | 0.0823 | 0.1516 |
| controlled_vs_proposed_raw | distinct1 | -0.0116 | -0.0124 |
| controlled_vs_proposed_raw | length_score | 0.4500 | 1.4439 |
| controlled_vs_proposed_raw | sentence_score | 0.2275 | 0.3085 |
| controlled_vs_proposed_raw | overall_quality | 0.1642 | 0.7082 |
| controlled_vs_candidate_no_context | context_relevance | 0.2212 | 6.8962 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1010 | 0.5601 |
| controlled_vs_candidate_no_context | naturalness | 0.1038 | 0.1275 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.2889 | 10.8943 |
| controlled_vs_candidate_no_context | context_overlap | 0.0635 | 1.4083 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1110 | 1.3127 |
| controlled_vs_candidate_no_context | persona_style | 0.0613 | 0.1088 |
| controlled_vs_candidate_no_context | distinct1 | -0.0211 | -0.0224 |
| controlled_vs_candidate_no_context | length_score | 0.4650 | 1.5674 |
| controlled_vs_candidate_no_context | sentence_score | 0.1925 | 0.2492 |
| controlled_vs_candidate_no_context | overall_quality | 0.1605 | 0.6810 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0067 | 0.0264 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0098 | -0.0349 |
| controlled_alt_vs_controlled_default | naturalness | -0.0096 | -0.0105 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0102 | 0.0324 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0016 | -0.0145 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0071 | -0.0365 |
| controlled_alt_vs_controlled_default | persona_style | -0.0206 | -0.0330 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0026 | 0.0028 |
| controlled_alt_vs_controlled_default | length_score | -0.0433 | -0.0569 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0175 | -0.0181 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0021 | -0.0052 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1933 | 2.8958 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1426 | 1.1063 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0985 | 0.1216 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2540 | 3.5481 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0515 | 0.9299 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1629 | 6.3925 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0617 | 0.1137 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0090 | -0.0096 |
| controlled_alt_vs_proposed_raw | length_score | 0.4067 | 1.3048 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.2100 | 0.2847 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1622 | 0.6992 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2279 | 7.1046 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.0912 | 0.5056 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0942 | 0.1157 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.2991 | 11.2800 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0619 | 1.3733 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1038 | 1.2282 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0407 | 0.0722 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0185 | -0.0196 |
| controlled_alt_vs_candidate_no_context | length_score | 0.4217 | 1.4213 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1750 | 0.2265 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1584 | 0.6722 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0347 | (-0.0008, 0.0781) | 0.0310 | 0.0347 | (0.0053, 0.0650) | 0.0057 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0514 | (-0.1308, 0.0110) | 0.9233 | -0.0514 | (-0.1502, 0.0108) | 0.9357 |
| proposed_vs_candidate_no_context | naturalness | -0.0043 | (-0.0439, 0.0333) | 0.5863 | -0.0043 | (-0.0472, 0.0232) | 0.5987 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0451 | (-0.0045, 0.1001) | 0.0410 | 0.0451 | (0.0064, 0.0897) | 0.0147 |
| proposed_vs_candidate_no_context | context_overlap | 0.0104 | (-0.0009, 0.0218) | 0.0360 | 0.0104 | (0.0022, 0.0167) | 0.0093 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0590 | (-0.1476, 0.0117) | 0.9380 | -0.0590 | (-0.1769, 0.0103) | 0.9450 |
| proposed_vs_candidate_no_context | persona_style | -0.0210 | (-0.1034, 0.0410) | 0.6863 | -0.0210 | (-0.1091, 0.0512) | 0.7040 |
| proposed_vs_candidate_no_context | distinct1 | -0.0095 | (-0.0313, 0.0120) | 0.8117 | -0.0095 | (-0.0225, 0.0061) | 0.8937 |
| proposed_vs_candidate_no_context | length_score | 0.0150 | (-0.1300, 0.1550) | 0.4007 | 0.0150 | (-0.1500, 0.1290) | 0.3967 |
| proposed_vs_candidate_no_context | sentence_score | -0.0350 | (-0.1050, 0.0350) | 0.9140 | -0.0350 | (-0.1029, 0.0350) | 0.9043 |
| proposed_vs_candidate_no_context | overall_quality | -0.0037 | (-0.0470, 0.0356) | 0.5363 | -0.0037 | (-0.0506, 0.0222) | 0.5583 |
| controlled_vs_proposed_raw | context_relevance | 0.1866 | (0.1405, 0.2262) | 0.0000 | 0.1866 | (0.1559, 0.2243) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.1525 | (0.0991, 0.2066) | 0.0000 | 0.1525 | (0.0810, 0.2352) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.1081 | (0.0631, 0.1500) | 0.0000 | 0.1081 | (0.0513, 0.1449) | 0.0007 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2438 | (0.1854, 0.2998) | 0.0000 | 0.2438 | (0.2035, 0.2907) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0531 | (0.0387, 0.0679) | 0.0000 | 0.0531 | (0.0402, 0.0761) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1700 | (0.1093, 0.2405) | 0.0000 | 0.1700 | (0.0861, 0.2778) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.0823 | (0.0083, 0.1807) | 0.0173 | 0.0823 | (0.0000, 0.2135) | 0.0977 |
| controlled_vs_proposed_raw | distinct1 | -0.0116 | (-0.0328, 0.0077) | 0.8610 | -0.0116 | (-0.0394, 0.0081) | 0.8323 |
| controlled_vs_proposed_raw | length_score | 0.4500 | (0.2783, 0.6100) | 0.0000 | 0.4500 | (0.2444, 0.5826) | 0.0003 |
| controlled_vs_proposed_raw | sentence_score | 0.2275 | (0.1575, 0.2975) | 0.0000 | 0.2275 | (0.1312, 0.3000) | 0.0000 |
| controlled_vs_proposed_raw | overall_quality | 0.1642 | (0.1307, 0.1976) | 0.0000 | 0.1642 | (0.1382, 0.2012) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2212 | (0.1875, 0.2537) | 0.0000 | 0.2212 | (0.1922, 0.2526) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1010 | (0.0431, 0.1574) | 0.0003 | 0.1010 | (0.0447, 0.1705) | 0.0007 |
| controlled_vs_candidate_no_context | naturalness | 0.1038 | (0.0560, 0.1465) | 0.0003 | 0.1038 | (0.0451, 0.1465) | 0.0007 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.2889 | (0.2409, 0.3309) | 0.0000 | 0.2889 | (0.2500, 0.3305) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0635 | (0.0481, 0.0786) | 0.0000 | 0.0635 | (0.0494, 0.0860) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1110 | (0.0517, 0.1700) | 0.0003 | 0.1110 | (0.0426, 0.1916) | 0.0017 |
| controlled_vs_candidate_no_context | persona_style | 0.0613 | (-0.0055, 0.1448) | 0.0400 | 0.0613 | (-0.0120, 0.1488) | 0.0583 |
| controlled_vs_candidate_no_context | distinct1 | -0.0211 | (-0.0421, -0.0020) | 0.9853 | -0.0211 | (-0.0405, -0.0058) | 0.9977 |
| controlled_vs_candidate_no_context | length_score | 0.4650 | (0.2833, 0.6267) | 0.0000 | 0.4650 | (0.2128, 0.6334) | 0.0007 |
| controlled_vs_candidate_no_context | sentence_score | 0.1925 | (0.1225, 0.2625) | 0.0000 | 0.1925 | (0.1094, 0.2567) | 0.0000 |
| controlled_vs_candidate_no_context | overall_quality | 0.1605 | (0.1235, 0.1954) | 0.0000 | 0.1605 | (0.1296, 0.1959) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0067 | (-0.0243, 0.0413) | 0.3493 | 0.0067 | (-0.0313, 0.0431) | 0.3417 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0098 | (-0.0321, 0.0118) | 0.8223 | -0.0098 | (-0.0416, 0.0126) | 0.7930 |
| controlled_alt_vs_controlled_default | naturalness | -0.0096 | (-0.0259, 0.0061) | 0.8813 | -0.0096 | (-0.0272, 0.0073) | 0.8793 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0102 | (-0.0303, 0.0545) | 0.3223 | 0.0102 | (-0.0366, 0.0574) | 0.3123 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0016 | (-0.0126, 0.0093) | 0.6270 | -0.0016 | (-0.0134, 0.0108) | 0.5763 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0071 | (-0.0381, 0.0179) | 0.6683 | -0.0071 | (-0.0425, 0.0175) | 0.6517 |
| controlled_alt_vs_controlled_default | persona_style | -0.0206 | (-0.0535, 0.0000) | 1.0000 | -0.0206 | (-0.0577, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0026 | (-0.0116, 0.0180) | 0.3643 | 0.0026 | (-0.0160, 0.0225) | 0.3933 |
| controlled_alt_vs_controlled_default | length_score | -0.0433 | (-0.1133, 0.0184) | 0.9070 | -0.0433 | (-0.1016, 0.0136) | 0.9430 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0175 | (-0.0525, 0.0000) | 1.0000 | -0.0175 | (-0.0618, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0021 | (-0.0224, 0.0185) | 0.5840 | -0.0021 | (-0.0299, 0.0210) | 0.5493 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1933 | (0.1519, 0.2313) | 0.0000 | 0.1933 | (0.1679, 0.2279) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1426 | (0.0913, 0.1968) | 0.0000 | 0.1426 | (0.0730, 0.2141) | 0.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0985 | (0.0496, 0.1417) | 0.0003 | 0.0985 | (0.0378, 0.1334) | 0.0003 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2540 | (0.1955, 0.3021) | 0.0000 | 0.2540 | (0.2170, 0.2989) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0515 | (0.0368, 0.0645) | 0.0000 | 0.0515 | (0.0384, 0.0711) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1629 | (0.1029, 0.2286) | 0.0000 | 0.1629 | (0.0774, 0.2587) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0617 | (-0.0162, 0.1641) | 0.0857 | 0.0617 | (-0.0300, 0.2034) | 0.2140 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0090 | (-0.0293, 0.0121) | 0.7993 | -0.0090 | (-0.0319, 0.0081) | 0.8323 |
| controlled_alt_vs_proposed_raw | length_score | 0.4067 | (0.2383, 0.5683) | 0.0000 | 0.4067 | (0.2083, 0.5213) | 0.0007 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.2100 | (0.1225, 0.2975) | 0.0000 | 0.2100 | (0.0875, 0.2975) | 0.0017 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1622 | (0.1299, 0.1941) | 0.0000 | 0.1622 | (0.1296, 0.1973) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2279 | (0.1934, 0.2640) | 0.0000 | 0.2279 | (0.1946, 0.2650) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.0912 | (0.0307, 0.1436) | 0.0037 | 0.0912 | (0.0291, 0.1548) | 0.0037 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0942 | (0.0441, 0.1410) | 0.0000 | 0.0942 | (0.0334, 0.1344) | 0.0010 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.2991 | (0.2495, 0.3476) | 0.0000 | 0.2991 | (0.2523, 0.3496) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0619 | (0.0507, 0.0736) | 0.0000 | 0.0619 | (0.0490, 0.0801) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1038 | (0.0383, 0.1643) | 0.0007 | 0.1038 | (0.0299, 0.1820) | 0.0053 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0407 | (-0.0254, 0.1209) | 0.1400 | 0.0407 | (-0.0263, 0.1188) | 0.1390 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0185 | (-0.0419, 0.0034) | 0.9517 | -0.0185 | (-0.0375, -0.0026) | 0.9897 |
| controlled_alt_vs_candidate_no_context | length_score | 0.4217 | (0.2433, 0.5867) | 0.0000 | 0.4217 | (0.2000, 0.5654) | 0.0000 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1750 | (0.0875, 0.2625) | 0.0017 | 0.1750 | (0.0778, 0.2479) | 0.0000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1584 | (0.1252, 0.1912) | 0.0000 | 0.1584 | (0.1201, 0.1901) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 10 | 3 | 7 | 0.6750 | 0.7692 |
| proposed_vs_candidate_no_context | persona_consistency | 5 | 5 | 10 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | naturalness | 7 | 6 | 7 | 0.5250 | 0.5385 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 6 | 1 | 13 | 0.6250 | 0.8571 |
| proposed_vs_candidate_no_context | context_overlap | 10 | 3 | 7 | 0.6750 | 0.7692 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 2 | 5 | 13 | 0.4250 | 0.2857 |
| proposed_vs_candidate_no_context | persona_style | 3 | 2 | 15 | 0.5250 | 0.6000 |
| proposed_vs_candidate_no_context | distinct1 | 5 | 7 | 8 | 0.4500 | 0.4167 |
| proposed_vs_candidate_no_context | length_score | 8 | 5 | 7 | 0.5750 | 0.6154 |
| proposed_vs_candidate_no_context | sentence_score | 1 | 3 | 16 | 0.4500 | 0.2500 |
| proposed_vs_candidate_no_context | overall_quality | 8 | 5 | 7 | 0.5750 | 0.6154 |
| controlled_vs_proposed_raw | context_relevance | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_vs_proposed_raw | persona_consistency | 14 | 0 | 6 | 0.8500 | 1.0000 |
| controlled_vs_proposed_raw | naturalness | 16 | 4 | 0 | 0.8000 | 0.8000 |
| controlled_vs_proposed_raw | context_keyword_coverage | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_vs_proposed_raw | context_overlap | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 14 | 0 | 6 | 0.8500 | 1.0000 |
| controlled_vs_proposed_raw | persona_style | 4 | 0 | 16 | 0.6000 | 1.0000 |
| controlled_vs_proposed_raw | distinct1 | 10 | 10 | 0 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | length_score | 16 | 4 | 0 | 0.8000 | 0.8000 |
| controlled_vs_proposed_raw | sentence_score | 13 | 0 | 7 | 0.8250 | 1.0000 |
| controlled_vs_proposed_raw | overall_quality | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_relevance | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_vs_candidate_no_context | persona_consistency | 12 | 2 | 6 | 0.7500 | 0.8571 |
| controlled_vs_candidate_no_context | naturalness | 16 | 4 | 0 | 0.8000 | 0.8000 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 19 | 0 | 1 | 0.9750 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 12 | 2 | 6 | 0.7500 | 0.8571 |
| controlled_vs_candidate_no_context | persona_style | 5 | 1 | 14 | 0.6000 | 0.8333 |
| controlled_vs_candidate_no_context | distinct1 | 9 | 11 | 0 | 0.4500 | 0.4500 |
| controlled_vs_candidate_no_context | length_score | 16 | 4 | 0 | 0.8000 | 0.8000 |
| controlled_vs_candidate_no_context | sentence_score | 11 | 0 | 9 | 0.7750 | 1.0000 |
| controlled_vs_candidate_no_context | overall_quality | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_alt_vs_controlled_default | context_relevance | 5 | 6 | 9 | 0.4750 | 0.4545 |
| controlled_alt_vs_controlled_default | persona_consistency | 1 | 4 | 15 | 0.4250 | 0.2000 |
| controlled_alt_vs_controlled_default | naturalness | 3 | 8 | 9 | 0.3750 | 0.2727 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 5 | 5 | 10 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | context_overlap | 4 | 7 | 9 | 0.4250 | 0.3636 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 1 | 2 | 17 | 0.4750 | 0.3333 |
| controlled_alt_vs_controlled_default | persona_style | 0 | 2 | 18 | 0.4500 | 0.0000 |
| controlled_alt_vs_controlled_default | distinct1 | 4 | 7 | 9 | 0.4250 | 0.3636 |
| controlled_alt_vs_controlled_default | length_score | 5 | 6 | 9 | 0.4750 | 0.4545 |
| controlled_alt_vs_controlled_default | sentence_score | 0 | 1 | 19 | 0.4750 | 0.0000 |
| controlled_alt_vs_controlled_default | overall_quality | 5 | 6 | 9 | 0.4750 | 0.4545 |
| controlled_alt_vs_proposed_raw | context_relevance | 18 | 2 | 0 | 0.9000 | 0.9000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 14 | 0 | 6 | 0.8500 | 1.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 15 | 5 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 17 | 0 | 3 | 0.9250 | 1.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 18 | 2 | 0 | 0.9000 | 0.9000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 14 | 0 | 6 | 0.8500 | 1.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 3 | 1 | 16 | 0.5500 | 0.7500 |
| controlled_alt_vs_proposed_raw | distinct1 | 10 | 10 | 0 | 0.5000 | 0.5000 |
| controlled_alt_vs_proposed_raw | length_score | 16 | 3 | 1 | 0.8250 | 0.8421 |
| controlled_alt_vs_proposed_raw | sentence_score | 13 | 1 | 6 | 0.8000 | 0.9286 |
| controlled_alt_vs_proposed_raw | overall_quality | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 13 | 2 | 5 | 0.7750 | 0.8667 |
| controlled_alt_vs_candidate_no_context | naturalness | 16 | 4 | 0 | 0.8000 | 0.8000 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 13 | 2 | 5 | 0.7750 | 0.8667 |
| controlled_alt_vs_candidate_no_context | persona_style | 4 | 2 | 14 | 0.5500 | 0.6667 |
| controlled_alt_vs_candidate_no_context | distinct1 | 9 | 10 | 1 | 0.4750 | 0.4737 |
| controlled_alt_vs_candidate_no_context | length_score | 16 | 4 | 0 | 0.8000 | 0.8000 |
| controlled_alt_vs_candidate_no_context | sentence_score | 11 | 1 | 8 | 0.7500 | 0.9167 |
| controlled_alt_vs_candidate_no_context | overall_quality | 19 | 1 | 0 | 0.9500 | 0.9500 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.0000 | 0.4500 | 0.2000 |
| proposed_contextual_controlled_alt | 0.0000 | 0.0000 | 0.0000 | 0.4500 | 0.2000 |
| proposed_contextual | 0.0000 | 0.0000 | 0.5500 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `8`
- Unique template signatures: `18`
- Template signature ratio: `0.9000`
- Effective sample size by source clustering: `6.67`
- Effective sample size by template-signature clustering: `16.67`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: unavailable (disabled_by_flag).

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
# Proposal Alignment Evaluation Report

- Run ID: `20260305T215929Z`
- Generated: `2026-03-05T22:04:54.152820+00:00`
- Scenarios: `artifacts\proposal_control_tuning\hybrid_compare_seed31\20260305T215929Z\scenarios.jsonl`
- Scenario count: `32`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_hybrid`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2647 (0.2421, 0.2886) | 0.3690 (0.3139, 0.4281) | 0.9026 (0.8783, 0.9234) | 0.4317 (0.4090, 0.4557) | n/a |
| proposed_contextual_controlled_hybrid | 0.2747 (0.2352, 0.3160) | 0.3674 (0.3103, 0.4338) | 0.8902 (0.8653, 0.9126) | 0.4334 (0.4063, 0.4641) | n/a |
| proposed_contextual | 0.0860 (0.0502, 0.1245) | 0.1565 (0.1161, 0.2009) | 0.8228 (0.7948, 0.8524) | 0.2542 (0.2250, 0.2875) | n/a |
| candidate_no_context | 0.0226 (0.0119, 0.0369) | 0.1557 (0.1163, 0.1984) | 0.8247 (0.7957, 0.8535) | 0.2251 (0.2051, 0.2473) | n/a |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0634 | 2.8090 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0008 | 0.0053 |
| proposed_vs_candidate_no_context | naturalness | -0.0019 | -0.0023 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0817 | 4.9286 |
| proposed_vs_candidate_no_context | context_overlap | 0.0207 | 0.5666 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0022 | -0.0337 |
| proposed_vs_candidate_no_context | persona_style | 0.0130 | 0.0253 |
| proposed_vs_candidate_no_context | distinct1 | 0.0010 | 0.0010 |
| proposed_vs_candidate_no_context | length_score | -0.0167 | -0.0475 |
| proposed_vs_candidate_no_context | sentence_score | 0.0109 | 0.0144 |
| proposed_vs_candidate_no_context | overall_quality | 0.0291 | 0.1293 |
| controlled_vs_proposed_raw | context_relevance | 0.1788 | 2.0800 |
| controlled_vs_proposed_raw | persona_consistency | 0.2125 | 1.3576 |
| controlled_vs_proposed_raw | naturalness | 0.0798 | 0.0969 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2319 | 2.3600 |
| controlled_vs_proposed_raw | context_overlap | 0.0549 | 0.9592 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2493 | 3.8953 |
| controlled_vs_proposed_raw | persona_style | 0.0655 | 0.1243 |
| controlled_vs_proposed_raw | distinct1 | -0.0062 | -0.0065 |
| controlled_vs_proposed_raw | length_score | 0.3292 | 0.9844 |
| controlled_vs_proposed_raw | sentence_score | 0.1641 | 0.2130 |
| controlled_vs_proposed_raw | overall_quality | 0.1776 | 0.6986 |
| controlled_vs_candidate_no_context | context_relevance | 0.2422 | 10.7318 |
| controlled_vs_candidate_no_context | persona_consistency | 0.2133 | 1.3700 |
| controlled_vs_candidate_no_context | naturalness | 0.0779 | 0.0945 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3135 | 18.9200 |
| controlled_vs_candidate_no_context | context_overlap | 0.0756 | 2.0694 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2470 | 3.7303 |
| controlled_vs_candidate_no_context | persona_style | 0.0785 | 0.1528 |
| controlled_vs_candidate_no_context | distinct1 | -0.0052 | -0.0055 |
| controlled_vs_candidate_no_context | length_score | 0.3125 | 0.8902 |
| controlled_vs_candidate_no_context | sentence_score | 0.1750 | 0.2305 |
| controlled_vs_candidate_no_context | overall_quality | 0.2067 | 0.9182 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0099 | 0.0375 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0016 | -0.0045 |
| controlled_alt_vs_controlled_default | naturalness | -0.0124 | -0.0137 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0163 | 0.0495 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0050 | -0.0445 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0037 | -0.0119 |
| controlled_alt_vs_controlled_default | persona_style | 0.0067 | 0.0113 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0075 | 0.0080 |
| controlled_alt_vs_controlled_default | length_score | -0.0750 | -0.1130 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0000 | 0.0000 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0017 | 0.0039 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1887 | 2.1957 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.2109 | 1.3471 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0674 | 0.0819 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2482 | 2.5263 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0499 | 0.8721 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.2455 | 3.8372 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0721 | 0.1370 |
| controlled_alt_vs_proposed_raw | distinct1 | 0.0014 | 0.0015 |
| controlled_alt_vs_proposed_raw | length_score | 0.2542 | 0.7601 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1641 | 0.2130 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1793 | 0.7053 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2521 | 11.1722 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.2117 | 1.3594 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0655 | 0.0795 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3299 | 19.9057 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0706 | 1.9329 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.2433 | 3.6742 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0852 | 0.1658 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0023 | 0.0025 |
| controlled_alt_vs_candidate_no_context | length_score | 0.2375 | 0.6766 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1750 | 0.2305 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.2084 | 0.9257 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0634 | (0.0298, 0.0998) | 0.0000 | 0.0634 | (0.0288, 0.0993) | 0.0000 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0008 | (-0.0672, 0.0685) | 0.4803 | 0.0008 | (-0.0642, 0.0636) | 0.4787 |
| proposed_vs_candidate_no_context | naturalness | -0.0019 | (-0.0407, 0.0375) | 0.5397 | -0.0019 | (-0.0333, 0.0277) | 0.5483 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0817 | (0.0360, 0.1359) | 0.0000 | 0.0817 | (0.0362, 0.1242) | 0.0000 |
| proposed_vs_candidate_no_context | context_overlap | 0.0207 | (0.0098, 0.0323) | 0.0000 | 0.0207 | (0.0068, 0.0363) | 0.0000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0022 | (-0.0796, 0.0781) | 0.5040 | -0.0022 | (-0.0804, 0.0694) | 0.5260 |
| proposed_vs_candidate_no_context | persona_style | 0.0130 | (-0.0537, 0.0824) | 0.3797 | 0.0130 | (-0.0194, 0.0538) | 0.2933 |
| proposed_vs_candidate_no_context | distinct1 | 0.0010 | (-0.0154, 0.0166) | 0.4407 | 0.0010 | (-0.0108, 0.0112) | 0.4277 |
| proposed_vs_candidate_no_context | length_score | -0.0167 | (-0.1604, 0.1177) | 0.6063 | -0.0167 | (-0.1309, 0.0937) | 0.6090 |
| proposed_vs_candidate_no_context | sentence_score | 0.0109 | (-0.0766, 0.0875) | 0.4617 | 0.0109 | (-0.0452, 0.0677) | 0.4380 |
| proposed_vs_candidate_no_context | overall_quality | 0.0291 | (-0.0101, 0.0700) | 0.0720 | 0.0291 | (-0.0132, 0.0710) | 0.0937 |
| controlled_vs_proposed_raw | context_relevance | 0.1788 | (0.1337, 0.2141) | 0.0000 | 0.1788 | (0.1284, 0.2185) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.2125 | (0.1436, 0.2790) | 0.0000 | 0.2125 | (0.1414, 0.2975) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0798 | (0.0460, 0.1124) | 0.0000 | 0.0798 | (0.0381, 0.1189) | 0.0000 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2319 | (0.1724, 0.2790) | 0.0000 | 0.2319 | (0.1659, 0.2849) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0549 | (0.0398, 0.0709) | 0.0000 | 0.0549 | (0.0372, 0.0706) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2493 | (0.1679, 0.3362) | 0.0000 | 0.2493 | (0.1658, 0.3456) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.0655 | (0.0104, 0.1286) | 0.0073 | 0.0655 | (-0.0030, 0.1752) | 0.1000 |
| controlled_vs_proposed_raw | distinct1 | -0.0062 | (-0.0223, 0.0106) | 0.7607 | -0.0062 | (-0.0252, 0.0143) | 0.7183 |
| controlled_vs_proposed_raw | length_score | 0.3292 | (0.2083, 0.4490) | 0.0000 | 0.3292 | (0.1874, 0.4677) | 0.0000 |
| controlled_vs_proposed_raw | sentence_score | 0.1641 | (0.0875, 0.2406) | 0.0000 | 0.1641 | (0.0875, 0.2227) | 0.0000 |
| controlled_vs_proposed_raw | overall_quality | 0.1776 | (0.1394, 0.2144) | 0.0000 | 0.1776 | (0.1345, 0.2171) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2422 | (0.2221, 0.2644) | 0.0000 | 0.2422 | (0.2136, 0.2652) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.2133 | (0.1347, 0.2934) | 0.0000 | 0.2133 | (0.1164, 0.3071) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.0779 | (0.0491, 0.1065) | 0.0000 | 0.0779 | (0.0552, 0.1025) | 0.0000 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3135 | (0.2876, 0.3428) | 0.0000 | 0.3135 | (0.2786, 0.3437) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0756 | (0.0623, 0.0901) | 0.0000 | 0.0756 | (0.0548, 0.0928) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2470 | (0.1577, 0.3400) | 0.0000 | 0.2470 | (0.1362, 0.3515) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.0785 | (0.0114, 0.1576) | 0.0080 | 0.0785 | (-0.0064, 0.1984) | 0.0420 |
| controlled_vs_candidate_no_context | distinct1 | -0.0052 | (-0.0223, 0.0120) | 0.7117 | -0.0052 | (-0.0208, 0.0105) | 0.7440 |
| controlled_vs_candidate_no_context | length_score | 0.3125 | (0.2062, 0.4188) | 0.0000 | 0.3125 | (0.2300, 0.3898) | 0.0000 |
| controlled_vs_candidate_no_context | sentence_score | 0.1750 | (0.0984, 0.2406) | 0.0000 | 0.1750 | (0.1167, 0.2368) | 0.0000 |
| controlled_vs_candidate_no_context | overall_quality | 0.2067 | (0.1751, 0.2393) | 0.0000 | 0.2067 | (0.1653, 0.2464) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0099 | (-0.0313, 0.0554) | 0.3423 | 0.0099 | (-0.0415, 0.0662) | 0.3453 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0016 | (-0.0807, 0.0779) | 0.5343 | -0.0016 | (-0.0865, 0.0891) | 0.5240 |
| controlled_alt_vs_controlled_default | naturalness | -0.0124 | (-0.0428, 0.0182) | 0.7950 | -0.0124 | (-0.0427, 0.0225) | 0.7897 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0163 | (-0.0348, 0.0760) | 0.2910 | 0.0163 | (-0.0509, 0.0859) | 0.3227 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0050 | (-0.0234, 0.0128) | 0.7133 | -0.0050 | (-0.0273, 0.0192) | 0.6503 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0037 | (-0.1028, 0.0936) | 0.5377 | -0.0037 | (-0.1076, 0.1102) | 0.5140 |
| controlled_alt_vs_controlled_default | persona_style | 0.0067 | (-0.0211, 0.0396) | 0.3567 | 0.0067 | (-0.0058, 0.0275) | 0.2083 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0075 | (-0.0104, 0.0260) | 0.2043 | 0.0075 | (-0.0133, 0.0275) | 0.2423 |
| controlled_alt_vs_controlled_default | length_score | -0.0750 | (-0.2011, 0.0458) | 0.8797 | -0.0750 | (-0.2049, 0.0774) | 0.8467 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0000 | (-0.0656, 0.0656) | 0.5647 | 0.0000 | (-0.0467, 0.0424) | 0.5957 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0017 | (-0.0352, 0.0384) | 0.4680 | 0.0017 | (-0.0498, 0.0535) | 0.4517 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1887 | (0.1473, 0.2308) | 0.0000 | 0.1887 | (0.1316, 0.2500) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.2109 | (0.1336, 0.2933) | 0.0000 | 0.2109 | (0.1183, 0.3205) | 0.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0674 | (0.0282, 0.1068) | 0.0000 | 0.0674 | (0.0172, 0.1124) | 0.0047 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2482 | (0.1916, 0.3061) | 0.0000 | 0.2482 | (0.1719, 0.3268) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0499 | (0.0343, 0.0653) | 0.0000 | 0.0499 | (0.0312, 0.0691) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.2455 | (0.1531, 0.3500) | 0.0000 | 0.2455 | (0.1341, 0.3751) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0721 | (0.0180, 0.1393) | 0.0023 | 0.0721 | (-0.0003, 0.1827) | 0.0280 |
| controlled_alt_vs_proposed_raw | distinct1 | 0.0014 | (-0.0170, 0.0187) | 0.4310 | 0.0014 | (-0.0220, 0.0230) | 0.4373 |
| controlled_alt_vs_proposed_raw | length_score | 0.2542 | (0.1010, 0.4021) | 0.0007 | 0.2542 | (0.0646, 0.4219) | 0.0037 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1641 | (0.0875, 0.2406) | 0.0000 | 0.1641 | (0.1077, 0.2162) | 0.0000 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1793 | (0.1360, 0.2232) | 0.0000 | 0.1793 | (0.1200, 0.2409) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2521 | (0.2118, 0.2957) | 0.0000 | 0.2521 | (0.2139, 0.2944) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.2117 | (0.1409, 0.2896) | 0.0000 | 0.2117 | (0.1274, 0.3082) | 0.0000 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0655 | (0.0290, 0.1014) | 0.0000 | 0.0655 | (0.0234, 0.1097) | 0.0007 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3299 | (0.2783, 0.3865) | 0.0000 | 0.3299 | (0.2776, 0.3888) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0706 | (0.0575, 0.0844) | 0.0000 | 0.0706 | (0.0575, 0.0850) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.2433 | (0.1606, 0.3394) | 0.0000 | 0.2433 | (0.1436, 0.3576) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0852 | (0.0156, 0.1672) | 0.0040 | 0.0852 | (0.0068, 0.2091) | 0.0217 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0023 | (-0.0157, 0.0187) | 0.4150 | 0.0023 | (-0.0182, 0.0211) | 0.4050 |
| controlled_alt_vs_candidate_no_context | length_score | 0.2375 | (0.0927, 0.3761) | 0.0003 | 0.2375 | (0.0771, 0.4034) | 0.0023 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1750 | (0.1094, 0.2300) | 0.0000 | 0.1750 | (0.1250, 0.2188) | 0.0000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.2084 | (0.1741, 0.2449) | 0.0000 | 0.2084 | (0.1756, 0.2506) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 17 | 5 | 10 | 0.6875 | 0.7727 |
| proposed_vs_candidate_no_context | persona_consistency | 10 | 8 | 14 | 0.5312 | 0.5556 |
| proposed_vs_candidate_no_context | naturalness | 11 | 11 | 10 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 13 | 1 | 18 | 0.6875 | 0.9286 |
| proposed_vs_candidate_no_context | context_overlap | 16 | 6 | 10 | 0.6562 | 0.7273 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 8 | 7 | 17 | 0.5156 | 0.5333 |
| proposed_vs_candidate_no_context | persona_style | 3 | 2 | 27 | 0.5156 | 0.6000 |
| proposed_vs_candidate_no_context | distinct1 | 10 | 8 | 14 | 0.5312 | 0.5556 |
| proposed_vs_candidate_no_context | length_score | 12 | 9 | 11 | 0.5469 | 0.5714 |
| proposed_vs_candidate_no_context | sentence_score | 8 | 7 | 17 | 0.5156 | 0.5333 |
| proposed_vs_candidate_no_context | overall_quality | 14 | 8 | 10 | 0.5938 | 0.6364 |
| controlled_vs_proposed_raw | context_relevance | 31 | 1 | 0 | 0.9688 | 0.9688 |
| controlled_vs_proposed_raw | persona_consistency | 26 | 3 | 3 | 0.8594 | 0.8966 |
| controlled_vs_proposed_raw | naturalness | 25 | 7 | 0 | 0.7812 | 0.7812 |
| controlled_vs_proposed_raw | context_keyword_coverage | 31 | 1 | 0 | 0.9688 | 0.9688 |
| controlled_vs_proposed_raw | context_overlap | 29 | 3 | 0 | 0.9062 | 0.9062 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 26 | 2 | 4 | 0.8750 | 0.9286 |
| controlled_vs_proposed_raw | persona_style | 5 | 1 | 26 | 0.5625 | 0.8333 |
| controlled_vs_proposed_raw | distinct1 | 12 | 18 | 2 | 0.4062 | 0.4000 |
| controlled_vs_proposed_raw | length_score | 25 | 5 | 2 | 0.8125 | 0.8333 |
| controlled_vs_proposed_raw | sentence_score | 17 | 2 | 13 | 0.7344 | 0.8947 |
| controlled_vs_proposed_raw | overall_quality | 30 | 2 | 0 | 0.9375 | 0.9375 |
| controlled_vs_candidate_no_context | context_relevance | 32 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 26 | 3 | 3 | 0.8594 | 0.8966 |
| controlled_vs_candidate_no_context | naturalness | 24 | 8 | 0 | 0.7500 | 0.7500 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 32 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 32 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 26 | 3 | 3 | 0.8594 | 0.8966 |
| controlled_vs_candidate_no_context | persona_style | 6 | 2 | 24 | 0.5625 | 0.7500 |
| controlled_vs_candidate_no_context | distinct1 | 15 | 17 | 0 | 0.4688 | 0.4688 |
| controlled_vs_candidate_no_context | length_score | 24 | 6 | 2 | 0.7812 | 0.8000 |
| controlled_vs_candidate_no_context | sentence_score | 18 | 2 | 12 | 0.7500 | 0.9000 |
| controlled_vs_candidate_no_context | overall_quality | 31 | 1 | 0 | 0.9688 | 0.9688 |
| controlled_alt_vs_controlled_default | context_relevance | 13 | 16 | 3 | 0.4531 | 0.4483 |
| controlled_alt_vs_controlled_default | persona_consistency | 9 | 11 | 12 | 0.4688 | 0.4500 |
| controlled_alt_vs_controlled_default | naturalness | 13 | 16 | 3 | 0.4531 | 0.4483 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 9 | 12 | 11 | 0.4531 | 0.4286 |
| controlled_alt_vs_controlled_default | context_overlap | 14 | 15 | 3 | 0.4844 | 0.4828 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 8 | 11 | 13 | 0.4531 | 0.4211 |
| controlled_alt_vs_controlled_default | persona_style | 3 | 3 | 26 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | distinct1 | 15 | 14 | 3 | 0.5156 | 0.5172 |
| controlled_alt_vs_controlled_default | length_score | 13 | 14 | 5 | 0.4844 | 0.4815 |
| controlled_alt_vs_controlled_default | sentence_score | 4 | 4 | 24 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | overall_quality | 15 | 14 | 3 | 0.5156 | 0.5172 |
| controlled_alt_vs_proposed_raw | context_relevance | 29 | 3 | 0 | 0.9062 | 0.9062 |
| controlled_alt_vs_proposed_raw | persona_consistency | 26 | 2 | 4 | 0.8750 | 0.9286 |
| controlled_alt_vs_proposed_raw | naturalness | 20 | 12 | 0 | 0.6250 | 0.6250 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 27 | 0 | 5 | 0.9219 | 1.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 27 | 5 | 0 | 0.8438 | 0.8438 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 26 | 2 | 4 | 0.8750 | 0.9286 |
| controlled_alt_vs_proposed_raw | persona_style | 6 | 1 | 25 | 0.5781 | 0.8571 |
| controlled_alt_vs_proposed_raw | distinct1 | 17 | 12 | 3 | 0.5781 | 0.5862 |
| controlled_alt_vs_proposed_raw | length_score | 22 | 10 | 0 | 0.6875 | 0.6875 |
| controlled_alt_vs_proposed_raw | sentence_score | 17 | 2 | 13 | 0.7344 | 0.8947 |
| controlled_alt_vs_proposed_raw | overall_quality | 30 | 2 | 0 | 0.9375 | 0.9375 |
| controlled_alt_vs_candidate_no_context | context_relevance | 31 | 1 | 0 | 0.9688 | 0.9688 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 27 | 1 | 4 | 0.9062 | 0.9643 |
| controlled_alt_vs_candidate_no_context | naturalness | 22 | 10 | 0 | 0.6875 | 0.6875 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 31 | 0 | 1 | 0.9844 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 31 | 1 | 0 | 0.9688 | 0.9688 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 26 | 1 | 5 | 0.8906 | 0.9630 |
| controlled_alt_vs_candidate_no_context | persona_style | 7 | 2 | 23 | 0.5781 | 0.7778 |
| controlled_alt_vs_candidate_no_context | distinct1 | 18 | 13 | 1 | 0.5781 | 0.5806 |
| controlled_alt_vs_candidate_no_context | length_score | 21 | 10 | 1 | 0.6719 | 0.6774 |
| controlled_alt_vs_candidate_no_context | sentence_score | 16 | 0 | 16 | 0.7500 | 1.0000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 32 | 0 | 0 | 1.0000 | 1.0000 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.3438 | 0.4062 | 0.5938 |
| proposed_contextual_controlled_hybrid | 0.0000 | 0.0000 | 0.1875 | 0.5000 | 0.5000 |
| proposed_contextual | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.4688 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `8`
- Unique template signatures: `30`
- Template signature ratio: `0.9375`
- Effective sample size by source clustering: `7.42`
- Effective sample size by template-signature clustering: `28.44`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: unavailable (disabled_by_flag).

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
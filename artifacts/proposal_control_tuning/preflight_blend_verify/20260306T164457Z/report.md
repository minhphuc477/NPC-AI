# Proposal Alignment Evaluation Report

- Run ID: `20260306T164457Z`
- Generated: `2026-03-06T16:49:21.373534+00:00`
- Scenarios: `artifacts\proposal_control_tuning\preflight_blend_verify\20260306T164457Z\scenarios.jsonl`
- Scenario count: `24`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_alt`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2611 (0.2256, 0.2974) | 0.3255 (0.2771, 0.3770) | 0.9044 (0.8796, 0.9259) | 0.4129 (0.3948, 0.4294) | n/a |
| proposed_contextual_controlled_alt | 0.2660 (0.2272, 0.3160) | 0.3028 (0.2605, 0.3497) | 0.8875 (0.8612, 0.9109) | 0.4032 (0.3809, 0.4281) | n/a |
| proposed_contextual | 0.0661 (0.0311, 0.1046) | 0.1904 (0.1385, 0.2453) | 0.8095 (0.7818, 0.8402) | 0.2545 (0.2240, 0.2927) | n/a |
| candidate_no_context | 0.0220 (0.0117, 0.0364) | 0.1534 (0.1070, 0.2114) | 0.7936 (0.7678, 0.8227) | 0.2167 (0.1941, 0.2419) | n/a |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0441 | 2.0063 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0370 | 0.2414 |
| proposed_vs_candidate_no_context | naturalness | 0.0159 | 0.0200 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0581 | 4.0000 |
| proposed_vs_candidate_no_context | context_overlap | 0.0115 | 0.2926 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0486 | 1.0000 |
| proposed_vs_candidate_no_context | persona_style | -0.0093 | -0.0162 |
| proposed_vs_candidate_no_context | distinct1 | -0.0005 | -0.0005 |
| proposed_vs_candidate_no_context | length_score | 0.0667 | 0.2909 |
| proposed_vs_candidate_no_context | sentence_score | 0.0292 | 0.0395 |
| proposed_vs_candidate_no_context | overall_quality | 0.0377 | 0.1741 |
| controlled_vs_proposed_raw | context_relevance | 0.1950 | 2.9494 |
| controlled_vs_proposed_raw | persona_consistency | 0.1351 | 0.7092 |
| controlled_vs_proposed_raw | naturalness | 0.0950 | 0.1173 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2523 | 3.4748 |
| controlled_vs_proposed_raw | context_overlap | 0.0613 | 1.2024 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1423 | 1.4633 |
| controlled_vs_proposed_raw | persona_style | 0.1063 | 0.1886 |
| controlled_vs_proposed_raw | distinct1 | 0.0034 | 0.0037 |
| controlled_vs_proposed_raw | length_score | 0.3722 | 1.2582 |
| controlled_vs_proposed_raw | sentence_score | 0.1896 | 0.2473 |
| controlled_vs_proposed_raw | overall_quality | 0.1584 | 0.6226 |
| controlled_vs_candidate_no_context | context_relevance | 0.2391 | 10.8730 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1721 | 1.1219 |
| controlled_vs_candidate_no_context | naturalness | 0.1108 | 0.1397 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3104 | 21.3739 |
| controlled_vs_candidate_no_context | context_overlap | 0.0728 | 1.8469 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1909 | 3.9265 |
| controlled_vs_candidate_no_context | persona_style | 0.0970 | 0.1694 |
| controlled_vs_candidate_no_context | distinct1 | 0.0030 | 0.0032 |
| controlled_vs_candidate_no_context | length_score | 0.4389 | 1.9152 |
| controlled_vs_candidate_no_context | sentence_score | 0.2188 | 0.2966 |
| controlled_vs_candidate_no_context | overall_quality | 0.1961 | 0.9050 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0049 | 0.0188 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0227 | -0.0698 |
| controlled_alt_vs_controlled_default | naturalness | -0.0170 | -0.0187 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0059 | 0.0183 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0025 | 0.0223 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0286 | -0.1193 |
| controlled_alt_vs_controlled_default | persona_style | 0.0008 | 0.0011 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0001 | 0.0001 |
| controlled_alt_vs_controlled_default | length_score | -0.0694 | -0.1040 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0292 | -0.0305 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0097 | -0.0234 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1999 | 3.0236 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1124 | 0.5900 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0780 | 0.0964 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2582 | 3.5565 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0638 | 1.2516 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1137 | 1.1694 |
| controlled_alt_vs_proposed_raw | persona_style | 0.1070 | 0.1900 |
| controlled_alt_vs_proposed_raw | distinct1 | 0.0035 | 0.0038 |
| controlled_alt_vs_proposed_raw | length_score | 0.3028 | 1.0235 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1604 | 0.2092 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1487 | 0.5845 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2440 | 11.0961 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1494 | 0.9739 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0939 | 0.1183 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3163 | 21.7826 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0753 | 1.9105 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1623 | 3.3388 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0977 | 0.1707 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0030 | 0.0033 |
| controlled_alt_vs_candidate_no_context | length_score | 0.3694 | 1.6121 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1896 | 0.2571 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1865 | 0.8604 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0441 | (0.0054, 0.0863) | 0.0097 | 0.0441 | (-0.0042, 0.0911) | 0.0553 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0370 | (-0.0277, 0.1094) | 0.1377 | 0.0370 | (0.0031, 0.0764) | 0.0157 |
| proposed_vs_candidate_no_context | naturalness | 0.0159 | (-0.0149, 0.0439) | 0.1587 | 0.0159 | (-0.0075, 0.0366) | 0.0977 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0581 | (0.0104, 0.1162) | 0.0047 | 0.0581 | (-0.0045, 0.1166) | 0.0633 |
| proposed_vs_candidate_no_context | context_overlap | 0.0115 | (-0.0020, 0.0246) | 0.0517 | 0.0115 | (-0.0069, 0.0280) | 0.1110 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0486 | (-0.0268, 0.1310) | 0.1113 | 0.0486 | (0.0072, 0.0942) | 0.0100 |
| proposed_vs_candidate_no_context | persona_style | -0.0093 | (-0.0421, 0.0173) | 0.7403 | -0.0093 | (-0.0417, 0.0175) | 0.7450 |
| proposed_vs_candidate_no_context | distinct1 | -0.0005 | (-0.0193, 0.0202) | 0.4993 | -0.0005 | (-0.0199, 0.0171) | 0.5083 |
| proposed_vs_candidate_no_context | length_score | 0.0667 | (-0.0472, 0.1681) | 0.1080 | 0.0667 | (-0.0154, 0.1530) | 0.0560 |
| proposed_vs_candidate_no_context | sentence_score | 0.0292 | (-0.0583, 0.1021) | 0.2943 | 0.0292 | (-0.0525, 0.0980) | 0.2940 |
| proposed_vs_candidate_no_context | overall_quality | 0.0377 | (-0.0001, 0.0763) | 0.0253 | 0.0377 | (0.0022, 0.0703) | 0.0177 |
| controlled_vs_proposed_raw | context_relevance | 0.1950 | (0.1527, 0.2305) | 0.0000 | 0.1950 | (0.1667, 0.2252) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.1351 | (0.0839, 0.1793) | 0.0000 | 0.1351 | (0.0901, 0.1848) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0950 | (0.0485, 0.1365) | 0.0000 | 0.0950 | (0.0471, 0.1412) | 0.0000 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2523 | (0.1951, 0.2985) | 0.0000 | 0.2523 | (0.2149, 0.2891) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0613 | (0.0432, 0.0802) | 0.0000 | 0.0613 | (0.0435, 0.0803) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1423 | (0.0829, 0.1931) | 0.0000 | 0.1423 | (0.0992, 0.1841) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.1063 | (0.0363, 0.1922) | 0.0000 | 0.1063 | (0.0133, 0.2694) | 0.0017 |
| controlled_vs_proposed_raw | distinct1 | 0.0034 | (-0.0144, 0.0206) | 0.3497 | 0.0034 | (-0.0129, 0.0187) | 0.3197 |
| controlled_vs_proposed_raw | length_score | 0.3722 | (0.1958, 0.5361) | 0.0000 | 0.3722 | (0.1651, 0.5652) | 0.0007 |
| controlled_vs_proposed_raw | sentence_score | 0.1896 | (0.0875, 0.2771) | 0.0000 | 0.1896 | (0.1105, 0.2852) | 0.0000 |
| controlled_vs_proposed_raw | overall_quality | 0.1584 | (0.1240, 0.1882) | 0.0000 | 0.1584 | (0.1283, 0.1918) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2391 | (0.2053, 0.2734) | 0.0000 | 0.2391 | (0.1978, 0.2738) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1721 | (0.1002, 0.2391) | 0.0000 | 0.1721 | (0.1327, 0.2373) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.1108 | (0.0698, 0.1489) | 0.0000 | 0.1108 | (0.0740, 0.1506) | 0.0000 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3104 | (0.2646, 0.3556) | 0.0000 | 0.3104 | (0.2562, 0.3613) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0728 | (0.0543, 0.0927) | 0.0000 | 0.0728 | (0.0529, 0.0871) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1909 | (0.1145, 0.2696) | 0.0000 | 0.1909 | (0.1645, 0.2294) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.0970 | (0.0147, 0.1933) | 0.0073 | 0.0970 | (0.0009, 0.2848) | 0.0237 |
| controlled_vs_candidate_no_context | distinct1 | 0.0030 | (-0.0154, 0.0209) | 0.3810 | 0.0030 | (-0.0150, 0.0195) | 0.3527 |
| controlled_vs_candidate_no_context | length_score | 0.4389 | (0.2736, 0.5833) | 0.0000 | 0.4389 | (0.2962, 0.6017) | 0.0000 |
| controlled_vs_candidate_no_context | sentence_score | 0.2188 | (0.1458, 0.2771) | 0.0000 | 0.2188 | (0.1685, 0.2625) | 0.0000 |
| controlled_vs_candidate_no_context | overall_quality | 0.1961 | (0.1660, 0.2225) | 0.0000 | 0.1961 | (0.1697, 0.2225) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0049 | (-0.0355, 0.0504) | 0.4300 | 0.0049 | (-0.0461, 0.0517) | 0.4267 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0227 | (-0.0778, 0.0282) | 0.8090 | -0.0227 | (-0.0492, 0.0063) | 0.9440 |
| controlled_alt_vs_controlled_default | naturalness | -0.0170 | (-0.0462, 0.0127) | 0.8633 | -0.0170 | (-0.0407, 0.0119) | 0.8767 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0059 | (-0.0468, 0.0621) | 0.4287 | 0.0059 | (-0.0636, 0.0668) | 0.4333 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0025 | (-0.0215, 0.0375) | 0.4807 | 0.0025 | (-0.0126, 0.0244) | 0.4227 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0286 | (-0.0889, 0.0238) | 0.8427 | -0.0286 | (-0.0627, 0.0073) | 0.9420 |
| controlled_alt_vs_controlled_default | persona_style | 0.0008 | (-0.0411, 0.0553) | 0.5120 | 0.0008 | (-0.0241, 0.0326) | 0.4930 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0001 | (-0.0154, 0.0149) | 0.4880 | 0.0001 | (-0.0107, 0.0092) | 0.4783 |
| controlled_alt_vs_controlled_default | length_score | -0.0694 | (-0.2056, 0.0556) | 0.8510 | -0.0694 | (-0.1743, 0.0533) | 0.8657 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0292 | (-0.0729, 0.0000) | 1.0000 | -0.0292 | (-0.0849, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0097 | (-0.0331, 0.0136) | 0.8063 | -0.0097 | (-0.0301, 0.0072) | 0.8430 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1999 | (0.1492, 0.2559) | 0.0000 | 0.1999 | (0.1456, 0.2526) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1124 | (0.0539, 0.1653) | 0.0000 | 0.1124 | (0.0778, 0.1663) | 0.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0780 | (0.0337, 0.1188) | 0.0000 | 0.0780 | (0.0149, 0.1447) | 0.0087 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2582 | (0.1915, 0.3245) | 0.0000 | 0.2582 | (0.1938, 0.3187) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0638 | (0.0355, 0.1079) | 0.0000 | 0.0638 | (0.0362, 0.1007) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1137 | (0.0468, 0.1722) | 0.0010 | 0.1137 | (0.0826, 0.1471) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 0.1070 | (0.0255, 0.1998) | 0.0030 | 0.1070 | (-0.0017, 0.2784) | 0.0260 |
| controlled_alt_vs_proposed_raw | distinct1 | 0.0035 | (-0.0126, 0.0195) | 0.3463 | 0.0035 | (-0.0156, 0.0214) | 0.3520 |
| controlled_alt_vs_proposed_raw | length_score | 0.3028 | (0.1208, 0.4847) | 0.0010 | 0.3028 | (0.0347, 0.5667) | 0.0140 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1604 | (0.0583, 0.2479) | 0.0030 | 0.1604 | (0.0437, 0.2891) | 0.0050 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1487 | (0.1115, 0.1862) | 0.0000 | 0.1487 | (0.1134, 0.1868) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2440 | (0.2031, 0.2989) | 0.0000 | 0.2440 | (0.1990, 0.2823) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1494 | (0.1045, 0.2016) | 0.0000 | 0.1494 | (0.0960, 0.2252) | 0.0000 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0939 | (0.0472, 0.1348) | 0.0000 | 0.0939 | (0.0434, 0.1480) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3163 | (0.2664, 0.3761) | 0.0000 | 0.3163 | (0.2626, 0.3623) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0753 | (0.0462, 0.1202) | 0.0000 | 0.0753 | (0.0484, 0.1036) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1623 | (0.1123, 0.2101) | 0.0000 | 0.1623 | (0.1192, 0.2160) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0977 | (0.0142, 0.1943) | 0.0067 | 0.0977 | (-0.0092, 0.2971) | 0.0590 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0030 | (-0.0205, 0.0264) | 0.4113 | 0.0030 | (-0.0179, 0.0237) | 0.3973 |
| controlled_alt_vs_candidate_no_context | length_score | 0.3694 | (0.1958, 0.5361) | 0.0000 | 0.3694 | (0.1536, 0.5955) | 0.0000 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1896 | (0.1167, 0.2625) | 0.0000 | 0.1896 | (0.1333, 0.2500) | 0.0000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1865 | (0.1549, 0.2192) | 0.0000 | 0.1865 | (0.1596, 0.2125) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 9 | 6 | 9 | 0.5625 | 0.6000 |
| proposed_vs_candidate_no_context | persona_consistency | 7 | 4 | 13 | 0.5625 | 0.6364 |
| proposed_vs_candidate_no_context | naturalness | 10 | 5 | 9 | 0.6042 | 0.6667 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 6 | 2 | 16 | 0.5833 | 0.7500 |
| proposed_vs_candidate_no_context | context_overlap | 9 | 6 | 9 | 0.5625 | 0.6000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 6 | 3 | 15 | 0.5625 | 0.6667 |
| proposed_vs_candidate_no_context | persona_style | 2 | 2 | 20 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | distinct1 | 7 | 8 | 9 | 0.4792 | 0.4667 |
| proposed_vs_candidate_no_context | length_score | 11 | 4 | 9 | 0.6458 | 0.7333 |
| proposed_vs_candidate_no_context | sentence_score | 5 | 3 | 16 | 0.5417 | 0.6250 |
| proposed_vs_candidate_no_context | overall_quality | 10 | 5 | 9 | 0.6042 | 0.6667 |
| controlled_vs_proposed_raw | context_relevance | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_vs_proposed_raw | persona_consistency | 19 | 1 | 4 | 0.8750 | 0.9500 |
| controlled_vs_proposed_raw | naturalness | 18 | 6 | 0 | 0.7500 | 0.7500 |
| controlled_vs_proposed_raw | context_keyword_coverage | 21 | 1 | 2 | 0.9167 | 0.9545 |
| controlled_vs_proposed_raw | context_overlap | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 19 | 1 | 4 | 0.8750 | 0.9500 |
| controlled_vs_proposed_raw | persona_style | 7 | 1 | 16 | 0.6250 | 0.8750 |
| controlled_vs_proposed_raw | distinct1 | 14 | 9 | 1 | 0.6042 | 0.6087 |
| controlled_vs_proposed_raw | length_score | 19 | 5 | 0 | 0.7917 | 0.7917 |
| controlled_vs_proposed_raw | sentence_score | 15 | 2 | 7 | 0.7708 | 0.8824 |
| controlled_vs_proposed_raw | overall_quality | 22 | 2 | 0 | 0.9167 | 0.9167 |
| controlled_vs_candidate_no_context | context_relevance | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 22 | 1 | 1 | 0.9375 | 0.9565 |
| controlled_vs_candidate_no_context | naturalness | 21 | 3 | 0 | 0.8750 | 0.8750 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 23 | 0 | 1 | 0.9792 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 22 | 2 | 0 | 0.9167 | 0.9167 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 22 | 1 | 1 | 0.9375 | 0.9565 |
| controlled_vs_candidate_no_context | persona_style | 7 | 1 | 16 | 0.6250 | 0.8750 |
| controlled_vs_candidate_no_context | distinct1 | 15 | 9 | 0 | 0.6250 | 0.6250 |
| controlled_vs_candidate_no_context | length_score | 21 | 3 | 0 | 0.8750 | 0.8750 |
| controlled_vs_candidate_no_context | sentence_score | 15 | 0 | 9 | 0.8125 | 1.0000 |
| controlled_vs_candidate_no_context | overall_quality | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_alt_vs_controlled_default | context_relevance | 7 | 10 | 7 | 0.4375 | 0.4118 |
| controlled_alt_vs_controlled_default | persona_consistency | 3 | 5 | 16 | 0.4583 | 0.3750 |
| controlled_alt_vs_controlled_default | naturalness | 7 | 10 | 7 | 0.4375 | 0.4118 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 6 | 7 | 11 | 0.4792 | 0.4615 |
| controlled_alt_vs_controlled_default | context_overlap | 6 | 11 | 7 | 0.3958 | 0.3529 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 2 | 4 | 18 | 0.4583 | 0.3333 |
| controlled_alt_vs_controlled_default | persona_style | 2 | 3 | 19 | 0.4792 | 0.4000 |
| controlled_alt_vs_controlled_default | distinct1 | 8 | 9 | 7 | 0.4792 | 0.4706 |
| controlled_alt_vs_controlled_default | length_score | 7 | 10 | 7 | 0.4375 | 0.4118 |
| controlled_alt_vs_controlled_default | sentence_score | 0 | 2 | 22 | 0.4583 | 0.0000 |
| controlled_alt_vs_controlled_default | overall_quality | 7 | 10 | 7 | 0.4375 | 0.4118 |
| controlled_alt_vs_proposed_raw | context_relevance | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_alt_vs_proposed_raw | persona_consistency | 17 | 2 | 5 | 0.8125 | 0.8947 |
| controlled_alt_vs_proposed_raw | naturalness | 19 | 5 | 0 | 0.7917 | 0.7917 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 21 | 1 | 2 | 0.9167 | 0.9545 |
| controlled_alt_vs_proposed_raw | context_overlap | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 17 | 2 | 5 | 0.8125 | 0.8947 |
| controlled_alt_vs_proposed_raw | persona_style | 7 | 2 | 15 | 0.6042 | 0.7778 |
| controlled_alt_vs_proposed_raw | distinct1 | 13 | 10 | 1 | 0.5625 | 0.5652 |
| controlled_alt_vs_proposed_raw | length_score | 16 | 7 | 1 | 0.6875 | 0.6957 |
| controlled_alt_vs_proposed_raw | sentence_score | 14 | 3 | 7 | 0.7292 | 0.8235 |
| controlled_alt_vs_proposed_raw | overall_quality | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_alt_vs_candidate_no_context | context_relevance | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 19 | 2 | 3 | 0.8542 | 0.9048 |
| controlled_alt_vs_candidate_no_context | naturalness | 18 | 6 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 19 | 1 | 4 | 0.8750 | 0.9500 |
| controlled_alt_vs_candidate_no_context | persona_style | 7 | 2 | 15 | 0.6042 | 0.7778 |
| controlled_alt_vs_candidate_no_context | distinct1 | 14 | 10 | 0 | 0.5833 | 0.5833 |
| controlled_alt_vs_candidate_no_context | length_score | 18 | 5 | 1 | 0.7708 | 0.7826 |
| controlled_alt_vs_candidate_no_context | sentence_score | 13 | 0 | 11 | 0.7708 | 1.0000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 23 | 1 | 0 | 0.9583 | 0.9583 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.0000 | 0.3750 | 0.1250 |
| proposed_contextual_controlled_alt | 0.0000 | 0.0000 | 0.0000 | 0.5833 | 0.1250 |
| proposed_contextual | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.6250 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `8`
- Unique template signatures: `20`
- Template signature ratio: `0.8333`
- Effective sample size by source clustering: `6.86`
- Effective sample size by template-signature clustering: `18.00`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: unavailable (disabled_by_flag).

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
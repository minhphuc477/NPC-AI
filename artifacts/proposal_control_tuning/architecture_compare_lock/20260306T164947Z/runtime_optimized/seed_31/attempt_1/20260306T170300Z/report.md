# Proposal Alignment Evaluation Report

- Run ID: `20260306T170300Z`
- Generated: `2026-03-06T17:07:30.979736+00:00`
- Scenarios: `artifacts\proposal_control_tuning\architecture_compare_lock\20260306T164947Z\runtime_optimized\seed_31\attempt_1\20260306T170300Z\scenarios.jsonl`
- Scenario count: `32`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_alt`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2808 (0.2583, 0.3065) | 0.3158 (0.2686, 0.3729) | 0.9021 (0.8822, 0.9203) | 0.4195 (0.3987, 0.4414) | n/a |
| proposed_contextual_controlled_alt | 0.2438 (0.2185, 0.2697) | 0.2993 (0.2594, 0.3413) | 0.8954 (0.8751, 0.9135) | 0.3946 (0.3763, 0.4129) | n/a |
| proposed_contextual | 0.0461 (0.0271, 0.0667) | 0.1475 (0.1103, 0.1893) | 0.8123 (0.7850, 0.8420) | 0.2303 (0.2083, 0.2540) | n/a |
| candidate_no_context | 0.0127 (0.0096, 0.0177) | 0.1524 (0.1119, 0.1988) | 0.8146 (0.7886, 0.8415) | 0.2172 (0.1990, 0.2378) | n/a |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0334 | 2.6267 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0050 | -0.0325 |
| proposed_vs_candidate_no_context | naturalness | -0.0023 | -0.0028 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0421 | 14.8333 |
| proposed_vs_candidate_no_context | context_overlap | 0.0130 | 0.3643 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0052 | -0.0843 |
| proposed_vs_candidate_no_context | persona_style | -0.0039 | -0.0076 |
| proposed_vs_candidate_no_context | distinct1 | 0.0052 | 0.0055 |
| proposed_vs_candidate_no_context | length_score | -0.0052 | -0.0164 |
| proposed_vs_candidate_no_context | sentence_score | -0.0328 | -0.0432 |
| proposed_vs_candidate_no_context | overall_quality | 0.0131 | 0.0605 |
| controlled_vs_proposed_raw | context_relevance | 0.2347 | 5.0878 |
| controlled_vs_proposed_raw | persona_consistency | 0.1684 | 1.1419 |
| controlled_vs_proposed_raw | naturalness | 0.0898 | 0.1105 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.3105 | 6.9032 |
| controlled_vs_proposed_raw | context_overlap | 0.0577 | 1.1828 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1905 | 3.3684 |
| controlled_vs_proposed_raw | persona_style | 0.0800 | 0.1566 |
| controlled_vs_proposed_raw | distinct1 | -0.0076 | -0.0080 |
| controlled_vs_proposed_raw | length_score | 0.3719 | 1.1940 |
| controlled_vs_proposed_raw | sentence_score | 0.1859 | 0.2559 |
| controlled_vs_proposed_raw | overall_quality | 0.1892 | 0.8216 |
| controlled_vs_candidate_no_context | context_relevance | 0.2681 | 21.0785 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1634 | 1.0723 |
| controlled_vs_candidate_no_context | naturalness | 0.0875 | 0.1075 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3527 | 124.1333 |
| controlled_vs_candidate_no_context | context_overlap | 0.0707 | 1.9779 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1853 | 3.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.0761 | 0.1478 |
| controlled_vs_candidate_no_context | distinct1 | -0.0024 | -0.0026 |
| controlled_vs_candidate_no_context | length_score | 0.3667 | 1.1579 |
| controlled_vs_candidate_no_context | sentence_score | 0.1531 | 0.2016 |
| controlled_vs_candidate_no_context | overall_quality | 0.2024 | 0.9319 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0370 | -0.1317 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0165 | -0.0522 |
| controlled_alt_vs_controlled_default | naturalness | -0.0068 | -0.0075 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0509 | -0.1432 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0045 | -0.0424 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0213 | -0.0861 |
| controlled_alt_vs_controlled_default | persona_style | 0.0026 | 0.0044 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0006 | 0.0007 |
| controlled_alt_vs_controlled_default | length_score | -0.0406 | -0.0595 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0109 | 0.0120 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0250 | -0.0595 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1977 | 4.2859 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1519 | 1.0300 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0830 | 0.1022 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2596 | 5.7716 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0532 | 1.0901 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1692 | 2.9921 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0826 | 0.1617 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0069 | -0.0073 |
| controlled_alt_vs_proposed_raw | length_score | 0.3312 | 1.0635 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1969 | 0.2710 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1643 | 0.7132 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2311 | 18.1704 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1469 | 0.9640 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0808 | 0.0992 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3018 | 106.2167 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0662 | 1.8515 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1640 | 2.6554 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0787 | 0.1528 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0018 | -0.0019 |
| controlled_alt_vs_candidate_no_context | length_score | 0.3260 | 1.0296 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1641 | 0.2160 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1774 | 0.8170 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0334 | (0.0163, 0.0535) | 0.0000 | 0.0334 | (0.0146, 0.0508) | 0.0000 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0050 | (-0.0383, 0.0301) | 0.6090 | -0.0050 | (-0.0275, 0.0209) | 0.6480 |
| proposed_vs_candidate_no_context | naturalness | -0.0023 | (-0.0278, 0.0232) | 0.5570 | -0.0023 | (-0.0272, 0.0247) | 0.5567 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0421 | (0.0199, 0.0677) | 0.0000 | 0.0421 | (0.0184, 0.0649) | 0.0000 |
| proposed_vs_candidate_no_context | context_overlap | 0.0130 | (0.0051, 0.0216) | 0.0000 | 0.0130 | (0.0060, 0.0205) | 0.0000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0052 | (-0.0461, 0.0320) | 0.6217 | -0.0052 | (-0.0286, 0.0206) | 0.6563 |
| proposed_vs_candidate_no_context | persona_style | -0.0039 | (-0.0615, 0.0460) | 0.5663 | -0.0039 | (-0.0379, 0.0322) | 0.6410 |
| proposed_vs_candidate_no_context | distinct1 | 0.0052 | (-0.0043, 0.0143) | 0.1317 | 0.0052 | (0.0007, 0.0093) | 0.0087 |
| proposed_vs_candidate_no_context | length_score | -0.0052 | (-0.1104, 0.0990) | 0.5307 | -0.0052 | (-0.1156, 0.1172) | 0.4977 |
| proposed_vs_candidate_no_context | sentence_score | -0.0328 | (-0.0875, 0.0219) | 0.9160 | -0.0328 | (-0.0778, 0.0212) | 0.9347 |
| proposed_vs_candidate_no_context | overall_quality | 0.0131 | (-0.0021, 0.0306) | 0.0457 | 0.0131 | (0.0014, 0.0286) | 0.0120 |
| controlled_vs_proposed_raw | context_relevance | 0.2347 | (0.2080, 0.2630) | 0.0000 | 0.2347 | (0.2035, 0.2697) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.1684 | (0.1200, 0.2253) | 0.0000 | 0.1684 | (0.1298, 0.2064) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0898 | (0.0511, 0.1258) | 0.0000 | 0.0898 | (0.0387, 0.1368) | 0.0003 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.3105 | (0.2755, 0.3481) | 0.0000 | 0.3105 | (0.2701, 0.3577) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0577 | (0.0464, 0.0692) | 0.0000 | 0.0577 | (0.0432, 0.0714) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1905 | (0.1306, 0.2600) | 0.0000 | 0.1905 | (0.1452, 0.2332) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.0800 | (0.0257, 0.1449) | 0.0010 | 0.0800 | (0.0061, 0.2098) | 0.0217 |
| controlled_vs_proposed_raw | distinct1 | -0.0076 | (-0.0238, 0.0085) | 0.8267 | -0.0076 | (-0.0237, 0.0115) | 0.7953 |
| controlled_vs_proposed_raw | length_score | 0.3719 | (0.2312, 0.5063) | 0.0000 | 0.3719 | (0.1802, 0.5353) | 0.0003 |
| controlled_vs_proposed_raw | sentence_score | 0.1859 | (0.0984, 0.2625) | 0.0000 | 0.1859 | (0.1037, 0.2758) | 0.0003 |
| controlled_vs_proposed_raw | overall_quality | 0.1892 | (0.1637, 0.2180) | 0.0000 | 0.1892 | (0.1636, 0.2127) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2681 | (0.2443, 0.2937) | 0.0000 | 0.2681 | (0.2349, 0.3048) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1634 | (0.1062, 0.2284) | 0.0000 | 0.1634 | (0.1153, 0.2148) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.0875 | (0.0528, 0.1213) | 0.0000 | 0.0875 | (0.0367, 0.1402) | 0.0000 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3527 | (0.3208, 0.3873) | 0.0000 | 0.3527 | (0.3086, 0.3998) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0707 | (0.0609, 0.0804) | 0.0000 | 0.0707 | (0.0536, 0.0841) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1853 | (0.1141, 0.2609) | 0.0000 | 0.1853 | (0.1359, 0.2374) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.0761 | (0.0176, 0.1481) | 0.0043 | 0.0761 | (-0.0033, 0.1949) | 0.0357 |
| controlled_vs_candidate_no_context | distinct1 | -0.0024 | (-0.0176, 0.0116) | 0.6153 | -0.0024 | (-0.0158, 0.0134) | 0.6330 |
| controlled_vs_candidate_no_context | length_score | 0.3667 | (0.2427, 0.4854) | 0.0000 | 0.3667 | (0.1805, 0.5522) | 0.0000 |
| controlled_vs_candidate_no_context | sentence_score | 0.1531 | (0.0766, 0.2297) | 0.0000 | 0.1531 | (0.0565, 0.2676) | 0.0003 |
| controlled_vs_candidate_no_context | overall_quality | 0.2024 | (0.1752, 0.2305) | 0.0000 | 0.2024 | (0.1747, 0.2266) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0370 | (-0.0697, -0.0040) | 0.9850 | -0.0370 | (-0.0782, -0.0019) | 0.9803 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0165 | (-0.0788, 0.0299) | 0.7013 | -0.0165 | (-0.0603, 0.0219) | 0.7667 |
| controlled_alt_vs_controlled_default | naturalness | -0.0068 | (-0.0242, 0.0099) | 0.7813 | -0.0068 | (-0.0208, 0.0079) | 0.8213 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0509 | (-0.0964, -0.0071) | 0.9910 | -0.0509 | (-0.1077, -0.0000) | 0.9767 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0045 | (-0.0153, 0.0054) | 0.7940 | -0.0045 | (-0.0183, 0.0070) | 0.7523 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0213 | (-0.0964, 0.0359) | 0.7017 | -0.0213 | (-0.0739, 0.0230) | 0.7847 |
| controlled_alt_vs_controlled_default | persona_style | 0.0026 | (-0.0156, 0.0234) | 0.4807 | 0.0026 | (-0.0156, 0.0242) | 0.4347 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0006 | (-0.0092, 0.0107) | 0.4300 | 0.0006 | (-0.0114, 0.0132) | 0.4450 |
| controlled_alt_vs_controlled_default | length_score | -0.0406 | (-0.1260, 0.0469) | 0.8237 | -0.0406 | (-0.0939, 0.0056) | 0.9590 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0109 | (-0.0547, 0.0766) | 0.4407 | 0.0109 | (-0.0368, 0.0625) | 0.4113 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0250 | (-0.0528, -0.0030) | 0.9890 | -0.0250 | (-0.0484, -0.0001) | 0.9763 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1977 | (0.1683, 0.2255) | 0.0000 | 0.1977 | (0.1693, 0.2308) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1519 | (0.1181, 0.1819) | 0.0000 | 0.1519 | (0.1168, 0.1915) | 0.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0830 | (0.0476, 0.1197) | 0.0003 | 0.0830 | (0.0383, 0.1257) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2596 | (0.2194, 0.2982) | 0.0000 | 0.2596 | (0.2214, 0.3051) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0532 | (0.0398, 0.0655) | 0.0000 | 0.0532 | (0.0388, 0.0676) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1692 | (0.1317, 0.2051) | 0.0000 | 0.1692 | (0.1327, 0.2087) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0826 | (0.0279, 0.1490) | 0.0003 | 0.0826 | (0.0132, 0.2132) | 0.0030 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0069 | (-0.0254, 0.0105) | 0.7813 | -0.0069 | (-0.0319, 0.0182) | 0.6930 |
| controlled_alt_vs_proposed_raw | length_score | 0.3312 | (0.1906, 0.4657) | 0.0000 | 0.3312 | (0.1560, 0.4656) | 0.0007 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1969 | (0.1312, 0.2625) | 0.0000 | 0.1969 | (0.1264, 0.2750) | 0.0000 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1643 | (0.1415, 0.1857) | 0.0000 | 0.1643 | (0.1410, 0.1937) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2311 | (0.2055, 0.2565) | 0.0000 | 0.2311 | (0.2023, 0.2612) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1469 | (0.1095, 0.1809) | 0.0000 | 0.1469 | (0.1118, 0.1890) | 0.0000 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0808 | (0.0443, 0.1165) | 0.0000 | 0.0808 | (0.0359, 0.1288) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3018 | (0.2669, 0.3366) | 0.0000 | 0.3018 | (0.2648, 0.3424) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0662 | (0.0549, 0.0774) | 0.0000 | 0.0662 | (0.0517, 0.0811) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1640 | (0.1208, 0.2018) | 0.0000 | 0.1640 | (0.1313, 0.2076) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0787 | (0.0192, 0.1508) | 0.0027 | 0.0787 | (0.0128, 0.1898) | 0.0217 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0018 | (-0.0182, 0.0139) | 0.5770 | -0.0018 | (-0.0228, 0.0208) | 0.5570 |
| controlled_alt_vs_candidate_no_context | length_score | 0.3260 | (0.1864, 0.4615) | 0.0000 | 0.3260 | (0.1636, 0.4862) | 0.0000 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1641 | (0.0766, 0.2406) | 0.0007 | 0.1641 | (0.0700, 0.2758) | 0.0000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1774 | (0.1552, 0.1981) | 0.0000 | 0.1774 | (0.1531, 0.2090) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 15 | 2 | 15 | 0.7031 | 0.8824 |
| proposed_vs_candidate_no_context | persona_consistency | 5 | 5 | 22 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | naturalness | 9 | 8 | 15 | 0.5156 | 0.5294 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 10 | 0 | 22 | 0.6562 | 1.0000 |
| proposed_vs_candidate_no_context | context_overlap | 14 | 3 | 15 | 0.6719 | 0.8235 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 3 | 3 | 26 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | persona_style | 3 | 3 | 26 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | distinct1 | 8 | 4 | 20 | 0.5625 | 0.6667 |
| proposed_vs_candidate_no_context | length_score | 10 | 7 | 15 | 0.5469 | 0.5882 |
| proposed_vs_candidate_no_context | sentence_score | 2 | 5 | 25 | 0.4531 | 0.2857 |
| proposed_vs_candidate_no_context | overall_quality | 11 | 6 | 15 | 0.5781 | 0.6471 |
| controlled_vs_proposed_raw | context_relevance | 32 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_proposed_raw | persona_consistency | 27 | 2 | 3 | 0.8906 | 0.9310 |
| controlled_vs_proposed_raw | naturalness | 23 | 9 | 0 | 0.7188 | 0.7188 |
| controlled_vs_proposed_raw | context_keyword_coverage | 32 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_proposed_raw | context_overlap | 31 | 1 | 0 | 0.9688 | 0.9688 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 27 | 2 | 3 | 0.8906 | 0.9310 |
| controlled_vs_proposed_raw | persona_style | 7 | 0 | 25 | 0.6094 | 1.0000 |
| controlled_vs_proposed_raw | distinct1 | 18 | 14 | 0 | 0.5625 | 0.5625 |
| controlled_vs_proposed_raw | length_score | 25 | 6 | 1 | 0.7969 | 0.8065 |
| controlled_vs_proposed_raw | sentence_score | 20 | 3 | 9 | 0.7656 | 0.8696 |
| controlled_vs_proposed_raw | overall_quality | 32 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_relevance | 32 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 26 | 3 | 3 | 0.8594 | 0.8966 |
| controlled_vs_candidate_no_context | naturalness | 25 | 6 | 1 | 0.7969 | 0.8065 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 32 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 32 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 26 | 3 | 3 | 0.8594 | 0.8966 |
| controlled_vs_candidate_no_context | persona_style | 8 | 1 | 23 | 0.6094 | 0.8889 |
| controlled_vs_candidate_no_context | distinct1 | 20 | 11 | 1 | 0.6406 | 0.6452 |
| controlled_vs_candidate_no_context | length_score | 26 | 4 | 2 | 0.8438 | 0.8667 |
| controlled_vs_candidate_no_context | sentence_score | 18 | 4 | 10 | 0.7188 | 0.8182 |
| controlled_vs_candidate_no_context | overall_quality | 32 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 6 | 14 | 12 | 0.3750 | 0.3000 |
| controlled_alt_vs_controlled_default | persona_consistency | 3 | 4 | 25 | 0.4844 | 0.4286 |
| controlled_alt_vs_controlled_default | naturalness | 9 | 10 | 13 | 0.4844 | 0.4737 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 5 | 11 | 16 | 0.4062 | 0.3125 |
| controlled_alt_vs_controlled_default | context_overlap | 11 | 9 | 12 | 0.5312 | 0.5500 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 2 | 3 | 27 | 0.4844 | 0.4000 |
| controlled_alt_vs_controlled_default | persona_style | 1 | 1 | 30 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | distinct1 | 8 | 11 | 13 | 0.4531 | 0.4211 |
| controlled_alt_vs_controlled_default | length_score | 7 | 11 | 14 | 0.4375 | 0.3889 |
| controlled_alt_vs_controlled_default | sentence_score | 5 | 4 | 23 | 0.5156 | 0.5556 |
| controlled_alt_vs_controlled_default | overall_quality | 10 | 10 | 12 | 0.5000 | 0.5000 |
| controlled_alt_vs_proposed_raw | context_relevance | 31 | 1 | 0 | 0.9688 | 0.9688 |
| controlled_alt_vs_proposed_raw | persona_consistency | 27 | 1 | 4 | 0.9062 | 0.9643 |
| controlled_alt_vs_proposed_raw | naturalness | 22 | 10 | 0 | 0.6875 | 0.6875 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 30 | 0 | 2 | 0.9688 | 1.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 29 | 3 | 0 | 0.9062 | 0.9062 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 27 | 1 | 4 | 0.9062 | 0.9643 |
| controlled_alt_vs_proposed_raw | persona_style | 7 | 0 | 25 | 0.6094 | 1.0000 |
| controlled_alt_vs_proposed_raw | distinct1 | 17 | 13 | 2 | 0.5625 | 0.5667 |
| controlled_alt_vs_proposed_raw | length_score | 24 | 8 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_proposed_raw | sentence_score | 19 | 1 | 12 | 0.7812 | 0.9500 |
| controlled_alt_vs_proposed_raw | overall_quality | 32 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 32 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 27 | 2 | 3 | 0.8906 | 0.9310 |
| controlled_alt_vs_candidate_no_context | naturalness | 25 | 7 | 0 | 0.7812 | 0.7812 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 31 | 0 | 1 | 0.9844 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 32 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 27 | 2 | 3 | 0.8906 | 0.9310 |
| controlled_alt_vs_candidate_no_context | persona_style | 8 | 1 | 23 | 0.6094 | 0.8889 |
| controlled_alt_vs_candidate_no_context | distinct1 | 21 | 11 | 0 | 0.6562 | 0.6562 |
| controlled_alt_vs_candidate_no_context | length_score | 26 | 6 | 0 | 0.8125 | 0.8125 |
| controlled_alt_vs_candidate_no_context | sentence_score | 19 | 4 | 9 | 0.7344 | 0.8261 |
| controlled_alt_vs_candidate_no_context | overall_quality | 32 | 0 | 0 | 1.0000 | 1.0000 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.1250 |
| proposed_contextual_controlled_alt | 0.0000 | 0.0000 | 0.0312 | 0.5000 | 0.0938 |
| proposed_contextual | 0.0000 | 0.0000 | 0.5312 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.5312 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `8`
- Unique template signatures: `30`
- Template signature ratio: `0.9375`
- Effective sample size by source clustering: `7.42`
- Effective sample size by template-signature clustering: `28.44`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: unavailable (disabled_by_flag).

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
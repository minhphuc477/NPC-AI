# Proposal Alignment Evaluation Report

- Run ID: `20260305T212950Z`
- Generated: `2026-03-05T21:34:09.534472+00:00`
- Scenarios: `artifacts\proposal_control_tuning\auto_tune_v2b\20260305T210504Z\valid_runs\trial_000\seed_29\20260305T212950Z\scenarios.jsonl`
- Scenario count: `20`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_tuned`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2505 (0.2258, 0.2775) | 0.3879 (0.3296, 0.4546) | 0.8921 (0.8675, 0.9160) | 0.4297 (0.4092, 0.4533) | n/a |
| proposed_contextual_controlled_tuned | 0.2571 (0.2207, 0.2954) | 0.3957 (0.3346, 0.4627) | 0.8760 (0.8461, 0.9082) | 0.4326 (0.4144, 0.4526) | n/a |
| proposed_contextual | 0.1266 (0.0818, 0.1753) | 0.2323 (0.1565, 0.3182) | 0.8478 (0.8174, 0.8794) | 0.3062 (0.2585, 0.3565) | n/a |
| candidate_no_context | 0.0327 (0.0157, 0.0527) | 0.2509 (0.1705, 0.3362) | 0.8529 (0.8171, 0.8855) | 0.2712 (0.2349, 0.3094) | n/a |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0939 | 2.8693 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0185 | -0.0738 |
| proposed_vs_candidate_no_context | naturalness | -0.0050 | -0.0059 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.1182 | 4.0000 |
| proposed_vs_candidate_no_context | context_overlap | 0.0372 | 0.9265 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0274 | -0.1465 |
| proposed_vs_candidate_no_context | persona_style | 0.0169 | 0.0334 |
| proposed_vs_candidate_no_context | distinct1 | -0.0061 | -0.0065 |
| proposed_vs_candidate_no_context | length_score | -0.0217 | -0.0487 |
| proposed_vs_candidate_no_context | sentence_score | 0.0175 | 0.0203 |
| proposed_vs_candidate_no_context | overall_quality | 0.0350 | 0.1292 |
| controlled_vs_proposed_raw | context_relevance | 0.1239 | 0.9786 |
| controlled_vs_proposed_raw | persona_consistency | 0.1555 | 0.6694 |
| controlled_vs_proposed_raw | naturalness | 0.0443 | 0.0522 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.1637 | 1.1082 |
| controlled_vs_proposed_raw | context_overlap | 0.0310 | 0.4005 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1714 | 1.0746 |
| controlled_vs_proposed_raw | persona_style | 0.0920 | 0.1757 |
| controlled_vs_proposed_raw | distinct1 | 0.0002 | 0.0003 |
| controlled_vs_proposed_raw | length_score | 0.1683 | 0.3976 |
| controlled_vs_proposed_raw | sentence_score | 0.1050 | 0.1197 |
| controlled_vs_proposed_raw | overall_quality | 0.1235 | 0.4033 |
| controlled_vs_candidate_no_context | context_relevance | 0.2178 | 6.6557 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1370 | 0.5462 |
| controlled_vs_candidate_no_context | naturalness | 0.0392 | 0.0460 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.2819 | 9.5410 |
| controlled_vs_candidate_no_context | context_overlap | 0.0681 | 1.6980 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1440 | 0.7707 |
| controlled_vs_candidate_no_context | persona_style | 0.1089 | 0.2149 |
| controlled_vs_candidate_no_context | distinct1 | -0.0059 | -0.0062 |
| controlled_vs_candidate_no_context | length_score | 0.1467 | 0.3296 |
| controlled_vs_candidate_no_context | sentence_score | 0.1225 | 0.1424 |
| controlled_vs_candidate_no_context | overall_quality | 0.1585 | 0.5846 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0066 | 0.0264 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0078 | 0.0202 |
| controlled_alt_vs_controlled_default | naturalness | -0.0161 | -0.0181 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0102 | 0.0328 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0018 | -0.0165 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0083 | 0.0252 |
| controlled_alt_vs_controlled_default | persona_style | 0.0058 | 0.0094 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0010 | -0.0010 |
| controlled_alt_vs_controlled_default | length_score | -0.0700 | -0.1183 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0175 | -0.0178 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0028 | 0.0066 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1305 | 1.0309 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1634 | 0.7031 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0281 | 0.0332 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.1739 | 1.1774 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0292 | 0.3773 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1798 | 1.1269 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0978 | 0.1868 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0007 | -0.0008 |
| controlled_alt_vs_proposed_raw | length_score | 0.0983 | 0.2323 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0875 | 0.0997 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1263 | 0.4125 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2244 | 6.8581 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1448 | 0.5774 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0231 | 0.0271 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.2921 | 9.8872 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0663 | 1.6534 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1524 | 0.8153 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.1147 | 0.2264 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0068 | -0.0072 |
| controlled_alt_vs_candidate_no_context | length_score | 0.0767 | 0.1723 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1050 | 0.1221 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1614 | 0.5950 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0939 | (0.0484, 0.1426) | 0.0000 | 0.0939 | (0.0514, 0.1065) | 0.0000 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0185 | (-0.1241, 0.0741) | 0.6330 | -0.0185 | (-0.0652, 0.0426) | 0.7490 |
| proposed_vs_candidate_no_context | naturalness | -0.0050 | (-0.0419, 0.0337) | 0.6100 | -0.0050 | (-0.0485, 0.0342) | 0.5933 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.1182 | (0.0602, 0.1818) | 0.0000 | 0.1182 | (0.0682, 0.1364) | 0.0000 |
| proposed_vs_candidate_no_context | context_overlap | 0.0372 | (0.0222, 0.0526) | 0.0000 | 0.0372 | (0.0122, 0.0499) | 0.0000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0274 | (-0.1440, 0.0857) | 0.6740 | -0.0274 | (-0.0893, 0.0417) | 0.8423 |
| proposed_vs_candidate_no_context | persona_style | 0.0169 | (-0.0603, 0.1047) | 0.3443 | 0.0169 | (-0.0703, 0.0462) | 0.2683 |
| proposed_vs_candidate_no_context | distinct1 | -0.0061 | (-0.0245, 0.0124) | 0.7190 | -0.0061 | (-0.0270, 0.0095) | 0.7100 |
| proposed_vs_candidate_no_context | length_score | -0.0217 | (-0.1434, 0.1150) | 0.6267 | -0.0217 | (-0.1667, 0.1083) | 0.6057 |
| proposed_vs_candidate_no_context | sentence_score | 0.0175 | (-0.0700, 0.1050) | 0.4283 | 0.0175 | (-0.0437, 0.0875) | 0.4057 |
| proposed_vs_candidate_no_context | overall_quality | 0.0350 | (-0.0185, 0.0818) | 0.0890 | 0.0350 | (0.0079, 0.0690) | 0.0000 |
| controlled_vs_proposed_raw | context_relevance | 0.1239 | (0.0731, 0.1695) | 0.0000 | 0.1239 | (0.0857, 0.1601) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.1555 | (0.0653, 0.2458) | 0.0007 | 0.1555 | (0.1293, 0.2458) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0443 | (0.0059, 0.0836) | 0.0087 | 0.0443 | (-0.0268, 0.1509) | 0.0400 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.1637 | (0.0932, 0.2263) | 0.0000 | 0.1637 | (0.1023, 0.2182) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0310 | (0.0133, 0.0478) | 0.0010 | 0.0310 | (0.0180, 0.0471) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1714 | (0.0690, 0.2774) | 0.0010 | 0.1714 | (0.1667, 0.1786) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.0920 | (-0.0372, 0.2215) | 0.0820 | 0.0920 | (-0.0312, 0.5625) | 0.2930 |
| controlled_vs_proposed_raw | distinct1 | 0.0002 | (-0.0139, 0.0145) | 0.4813 | 0.0002 | (-0.0066, 0.0117) | 0.3710 |
| controlled_vs_proposed_raw | length_score | 0.1683 | (0.0116, 0.3300) | 0.0200 | 0.1683 | (-0.1208, 0.6000) | 0.1393 |
| controlled_vs_proposed_raw | sentence_score | 0.1050 | (0.0175, 0.1925) | 0.0113 | 0.1050 | (0.0000, 0.2625) | 0.0347 |
| controlled_vs_proposed_raw | overall_quality | 0.1235 | (0.0742, 0.1701) | 0.0000 | 0.1235 | (0.1031, 0.1890) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2178 | (0.1922, 0.2441) | 0.0000 | 0.2178 | (0.1923, 0.2464) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1370 | (0.0457, 0.2140) | 0.0013 | 0.1370 | (0.0714, 0.1984) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.0392 | (0.0024, 0.0791) | 0.0187 | 0.0392 | (0.0074, 0.1544) | 0.0000 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.2819 | (0.2440, 0.3171) | 0.0000 | 0.2819 | (0.2386, 0.3229) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0681 | (0.0531, 0.0825) | 0.0000 | 0.0681 | (0.0368, 0.0841) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1440 | (0.0500, 0.2250) | 0.0027 | 0.1440 | (0.0893, 0.2083) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.1089 | (-0.0071, 0.2291) | 0.0323 | 0.1089 | (0.0000, 0.4922) | 0.0340 |
| controlled_vs_candidate_no_context | distinct1 | -0.0059 | (-0.0219, 0.0102) | 0.7627 | -0.0059 | (-0.0257, 0.0163) | 0.6927 |
| controlled_vs_candidate_no_context | length_score | 0.1467 | (-0.0033, 0.3084) | 0.0290 | 0.1467 | (-0.0125, 0.6083) | 0.0340 |
| controlled_vs_candidate_no_context | sentence_score | 0.1225 | (0.0350, 0.2100) | 0.0067 | 0.1225 | (0.0875, 0.2625) | 0.0000 |
| controlled_vs_candidate_no_context | overall_quality | 0.1585 | (0.1221, 0.1929) | 0.0000 | 0.1585 | (0.1178, 0.1970) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0066 | (-0.0270, 0.0431) | 0.3773 | 0.0066 | (-0.0217, 0.0376) | 0.3637 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0078 | (-0.1009, 0.1030) | 0.4187 | 0.0078 | (0.0000, 0.0333) | 0.0000 |
| controlled_alt_vs_controlled_default | naturalness | -0.0161 | (-0.0424, 0.0082) | 0.8923 | -0.0161 | (-0.0277, 0.0134) | 0.9673 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0102 | (-0.0375, 0.0621) | 0.3443 | 0.0102 | (-0.0312, 0.0568) | 0.4113 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0018 | (-0.0197, 0.0194) | 0.5793 | -0.0018 | (-0.0073, 0.0048) | 0.6927 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0083 | (-0.1096, 0.1238) | 0.4080 | 0.0083 | (0.0000, 0.0417) | 0.0387 |
| controlled_alt_vs_controlled_default | persona_style | 0.0058 | (-0.0608, 0.0700) | 0.4567 | 0.0058 | (0.0000, 0.0145) | 0.2933 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0010 | (-0.0108, 0.0089) | 0.5943 | -0.0010 | (-0.0193, 0.0126) | 0.5917 |
| controlled_alt_vs_controlled_default | length_score | -0.0700 | (-0.1883, 0.0467) | 0.8763 | -0.0700 | (-0.1000, 0.0500) | 0.9690 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0175 | (-0.0525, 0.0000) | 1.0000 | -0.0175 | (-0.0437, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0028 | (-0.0325, 0.0382) | 0.4270 | 0.0028 | (-0.0150, 0.0157) | 0.2583 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1305 | (0.0657, 0.1973) | 0.0000 | 0.1305 | (0.1222, 0.1615) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1634 | (0.0615, 0.2749) | 0.0007 | 0.1634 | (0.1322, 0.2792) | 0.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0281 | (-0.0215, 0.0777) | 0.1220 | 0.0281 | (-0.0545, 0.1644) | 0.2590 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.1739 | (0.0901, 0.2626) | 0.0000 | 0.1739 | (0.1591, 0.2182) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0292 | (0.0065, 0.0558) | 0.0047 | 0.0292 | (0.0184, 0.0398) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1798 | (0.0690, 0.2941) | 0.0003 | 0.1798 | (0.1667, 0.2083) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0978 | (-0.0280, 0.2339) | 0.0670 | 0.0978 | (-0.0312, 0.5625) | 0.3027 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0007 | (-0.0217, 0.0197) | 0.5180 | -0.0007 | (-0.0258, 0.0203) | 0.5740 |
| controlled_alt_vs_proposed_raw | length_score | 0.0983 | (-0.0883, 0.3017) | 0.1523 | 0.0983 | (-0.2208, 0.6500) | 0.2577 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0875 | (-0.0175, 0.1750) | 0.0553 | 0.0875 | (0.0000, 0.2625) | 0.0440 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1263 | (0.0720, 0.1839) | 0.0000 | 0.1263 | (0.0961, 0.2047) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2244 | (0.1863, 0.2646) | 0.0000 | 0.2244 | (0.2129, 0.2299) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1448 | (0.0399, 0.2360) | 0.0050 | 0.1448 | (0.0714, 0.2318) | 0.0000 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0231 | (-0.0194, 0.0689) | 0.1537 | 0.0231 | (-0.0203, 0.1679) | 0.2943 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.2921 | (0.2428, 0.3454) | 0.0000 | 0.2921 | (0.2864, 0.2955) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0663 | (0.0444, 0.0899) | 0.0000 | 0.0663 | (0.0416, 0.0768) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1524 | (0.0429, 0.2524) | 0.0023 | 0.1524 | (0.0893, 0.2083) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.1147 | (0.0003, 0.2487) | 0.0233 | 0.1147 | (0.0000, 0.4922) | 0.0423 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0068 | (-0.0262, 0.0132) | 0.7510 | -0.0068 | (-0.0164, 0.0250) | 0.7397 |
| controlled_alt_vs_candidate_no_context | length_score | 0.0767 | (-0.0883, 0.2467) | 0.2027 | 0.0767 | (-0.1125, 0.6583) | 0.3047 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1050 | (0.0000, 0.1925) | 0.0263 | 0.1050 | (0.0437, 0.2625) | 0.0000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1614 | (0.1219, 0.2003) | 0.0000 | 0.1614 | (0.1320, 0.2127) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 14 | 2 | 4 | 0.8000 | 0.8750 |
| proposed_vs_candidate_no_context | persona_consistency | 7 | 5 | 8 | 0.5500 | 0.5833 |
| proposed_vs_candidate_no_context | naturalness | 6 | 10 | 4 | 0.4000 | 0.3750 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 12 | 1 | 7 | 0.7750 | 0.9231 |
| proposed_vs_candidate_no_context | context_overlap | 14 | 2 | 4 | 0.8000 | 0.8750 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 3 | 4 | 13 | 0.4750 | 0.4286 |
| proposed_vs_candidate_no_context | persona_style | 5 | 4 | 11 | 0.5250 | 0.5556 |
| proposed_vs_candidate_no_context | distinct1 | 7 | 9 | 4 | 0.4500 | 0.4375 |
| proposed_vs_candidate_no_context | length_score | 7 | 9 | 4 | 0.4500 | 0.4375 |
| proposed_vs_candidate_no_context | sentence_score | 4 | 3 | 13 | 0.5250 | 0.5714 |
| proposed_vs_candidate_no_context | overall_quality | 13 | 3 | 4 | 0.7500 | 0.8125 |
| controlled_vs_proposed_raw | context_relevance | 17 | 3 | 0 | 0.8500 | 0.8500 |
| controlled_vs_proposed_raw | persona_consistency | 15 | 3 | 2 | 0.8000 | 0.8333 |
| controlled_vs_proposed_raw | naturalness | 14 | 6 | 0 | 0.7000 | 0.7000 |
| controlled_vs_proposed_raw | context_keyword_coverage | 15 | 3 | 2 | 0.8000 | 0.8333 |
| controlled_vs_proposed_raw | context_overlap | 16 | 4 | 0 | 0.8000 | 0.8000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 15 | 2 | 3 | 0.8250 | 0.8824 |
| controlled_vs_proposed_raw | persona_style | 8 | 4 | 8 | 0.6000 | 0.6667 |
| controlled_vs_proposed_raw | distinct1 | 11 | 9 | 0 | 0.5500 | 0.5500 |
| controlled_vs_proposed_raw | length_score | 13 | 6 | 1 | 0.6750 | 0.6842 |
| controlled_vs_proposed_raw | sentence_score | 7 | 1 | 12 | 0.6500 | 0.8750 |
| controlled_vs_proposed_raw | overall_quality | 18 | 2 | 0 | 0.9000 | 0.9000 |
| controlled_vs_candidate_no_context | context_relevance | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 16 | 2 | 2 | 0.8500 | 0.8889 |
| controlled_vs_candidate_no_context | naturalness | 12 | 8 | 0 | 0.6000 | 0.6000 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 16 | 2 | 2 | 0.8500 | 0.8889 |
| controlled_vs_candidate_no_context | persona_style | 8 | 3 | 9 | 0.6250 | 0.7273 |
| controlled_vs_candidate_no_context | distinct1 | 9 | 11 | 0 | 0.4500 | 0.4500 |
| controlled_vs_candidate_no_context | length_score | 13 | 7 | 0 | 0.6500 | 0.6500 |
| controlled_vs_candidate_no_context | sentence_score | 8 | 1 | 11 | 0.6750 | 0.8889 |
| controlled_vs_candidate_no_context | overall_quality | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_alt_vs_controlled_default | context_relevance | 6 | 10 | 4 | 0.4000 | 0.3750 |
| controlled_alt_vs_controlled_default | persona_consistency | 8 | 5 | 7 | 0.5750 | 0.6154 |
| controlled_alt_vs_controlled_default | naturalness | 6 | 10 | 4 | 0.4000 | 0.3750 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 3 | 4 | 13 | 0.4750 | 0.4286 |
| controlled_alt_vs_controlled_default | context_overlap | 6 | 10 | 4 | 0.4000 | 0.3750 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 6 | 3 | 11 | 0.5750 | 0.6667 |
| controlled_alt_vs_controlled_default | persona_style | 4 | 3 | 13 | 0.5250 | 0.5714 |
| controlled_alt_vs_controlled_default | distinct1 | 7 | 8 | 5 | 0.4750 | 0.4667 |
| controlled_alt_vs_controlled_default | length_score | 6 | 9 | 5 | 0.4250 | 0.4000 |
| controlled_alt_vs_controlled_default | sentence_score | 0 | 1 | 19 | 0.4750 | 0.0000 |
| controlled_alt_vs_controlled_default | overall_quality | 9 | 7 | 4 | 0.5500 | 0.5625 |
| controlled_alt_vs_proposed_raw | context_relevance | 15 | 5 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_proposed_raw | persona_consistency | 16 | 4 | 0 | 0.8000 | 0.8000 |
| controlled_alt_vs_proposed_raw | naturalness | 12 | 8 | 0 | 0.6000 | 0.6000 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 13 | 3 | 4 | 0.7500 | 0.8125 |
| controlled_alt_vs_proposed_raw | context_overlap | 14 | 6 | 0 | 0.7000 | 0.7000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 16 | 3 | 1 | 0.8250 | 0.8421 |
| controlled_alt_vs_proposed_raw | persona_style | 9 | 4 | 7 | 0.6250 | 0.6923 |
| controlled_alt_vs_proposed_raw | distinct1 | 12 | 8 | 0 | 0.6000 | 0.6000 |
| controlled_alt_vs_proposed_raw | length_score | 12 | 8 | 0 | 0.6000 | 0.6000 |
| controlled_alt_vs_proposed_raw | sentence_score | 7 | 2 | 11 | 0.6250 | 0.7778 |
| controlled_alt_vs_proposed_raw | overall_quality | 16 | 4 | 0 | 0.8000 | 0.8000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 17 | 2 | 1 | 0.8750 | 0.8947 |
| controlled_alt_vs_candidate_no_context | naturalness | 11 | 9 | 0 | 0.5500 | 0.5500 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 17 | 2 | 1 | 0.8750 | 0.8947 |
| controlled_alt_vs_candidate_no_context | persona_style | 7 | 3 | 10 | 0.6000 | 0.7000 |
| controlled_alt_vs_candidate_no_context | distinct1 | 8 | 12 | 0 | 0.4000 | 0.4000 |
| controlled_alt_vs_candidate_no_context | length_score | 9 | 10 | 1 | 0.4750 | 0.4737 |
| controlled_alt_vs_candidate_no_context | sentence_score | 8 | 2 | 10 | 0.6500 | 0.8000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 19 | 1 | 0 | 0.9500 | 0.9500 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.3000 | 0.5500 | 0.4500 |
| proposed_contextual_controlled_tuned | 0.0000 | 0.0000 | 0.3000 | 0.5000 | 0.5000 |
| proposed_contextual | 0.0000 | 0.0000 | 0.3000 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.3500 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `3`
- Unique template signatures: `14`
- Template signature ratio: `0.7000`
- Effective sample size by source clustering: `2.78`
- Effective sample size by template-signature clustering: `10.53`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: unavailable (disabled_by_flag).

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
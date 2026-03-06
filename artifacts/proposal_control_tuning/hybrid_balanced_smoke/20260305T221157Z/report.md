# Proposal Alignment Evaluation Report

- Run ID: `20260305T221157Z`
- Generated: `2026-03-05T22:13:25.037098+00:00`
- Scenarios: `artifacts\proposal_control_tuning\hybrid_balanced_smoke\20260305T221157Z\scenarios.jsonl`
- Scenario count: `8`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_hb`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2818 (0.1795, 0.3755) | 0.4415 (0.3476, 0.5312) | 0.8834 (0.8357, 0.9301) | 0.4620 (0.4244, 0.5075) | n/a |
| proposed_contextual_controlled_hb | 0.1971 (0.1373, 0.2354) | 0.3736 (0.2822, 0.4628) | 0.9121 (0.8937, 0.9307) | 0.4039 (0.3789, 0.4315) | n/a |
| proposed_contextual | 0.1223 (0.0466, 0.2020) | 0.2156 (0.1246, 0.3143) | 0.8378 (0.7876, 0.8861) | 0.2969 (0.2234, 0.3746) | n/a |
| candidate_no_context | 0.0375 (0.0123, 0.0844) | 0.2411 (0.1375, 0.3625) | 0.8245 (0.7778, 0.8736) | 0.2629 (0.2098, 0.3229) | n/a |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0848 | 2.2622 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0255 | -0.1057 |
| proposed_vs_candidate_no_context | naturalness | 0.0133 | 0.0161 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.1127 | 3.6061 |
| proposed_vs_candidate_no_context | context_overlap | 0.0197 | 0.3785 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0089 | -0.0556 |
| proposed_vs_candidate_no_context | persona_style | -0.0916 | -0.1629 |
| proposed_vs_candidate_no_context | distinct1 | 0.0041 | 0.0043 |
| proposed_vs_candidate_no_context | length_score | 0.0583 | 0.1750 |
| proposed_vs_candidate_no_context | sentence_score | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | overall_quality | 0.0340 | 0.1294 |
| controlled_vs_proposed_raw | context_relevance | 0.1595 | 1.3047 |
| controlled_vs_proposed_raw | persona_consistency | 0.2259 | 1.0480 |
| controlled_vs_proposed_raw | naturalness | 0.0456 | 0.0544 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.1903 | 1.3224 |
| controlled_vs_proposed_raw | context_overlap | 0.0876 | 1.2220 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2595 | 1.7098 |
| controlled_vs_proposed_raw | persona_style | 0.0916 | 0.1946 |
| controlled_vs_proposed_raw | distinct1 | 0.0103 | 0.0108 |
| controlled_vs_proposed_raw | length_score | 0.1417 | 0.3617 |
| controlled_vs_proposed_raw | sentence_score | 0.1312 | 0.1680 |
| controlled_vs_proposed_raw | overall_quality | 0.1651 | 0.5562 |
| controlled_vs_candidate_no_context | context_relevance | 0.2443 | 6.5183 |
| controlled_vs_candidate_no_context | persona_consistency | 0.2005 | 0.8316 |
| controlled_vs_candidate_no_context | naturalness | 0.0589 | 0.0714 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3030 | 9.6970 |
| controlled_vs_candidate_no_context | context_overlap | 0.1073 | 2.0630 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2506 | 1.5593 |
| controlled_vs_candidate_no_context | persona_style | 0.0000 | 0.0000 |
| controlled_vs_candidate_no_context | distinct1 | 0.0143 | 0.0151 |
| controlled_vs_candidate_no_context | length_score | 0.2000 | 0.6000 |
| controlled_vs_candidate_no_context | sentence_score | 0.1312 | 0.1680 |
| controlled_vs_candidate_no_context | overall_quality | 0.1991 | 0.7576 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0847 | -0.3005 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0680 | -0.1539 |
| controlled_alt_vs_controlled_default | naturalness | 0.0287 | 0.0325 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0871 | -0.2606 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0790 | -0.4959 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.1006 | -0.2446 |
| controlled_alt_vs_controlled_default | persona_style | 0.0625 | 0.1111 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0079 | -0.0082 |
| controlled_alt_vs_controlled_default | length_score | 0.1375 | 0.2578 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0438 | 0.0479 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0581 | -0.1258 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.0748 | 0.6121 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1580 | 0.7327 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0743 | 0.0887 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.1032 | 0.7171 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0086 | 0.1202 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1589 | 1.0471 |
| controlled_alt_vs_proposed_raw | persona_style | 0.1541 | 0.3274 |
| controlled_alt_vs_proposed_raw | distinct1 | 0.0024 | 0.0025 |
| controlled_alt_vs_proposed_raw | length_score | 0.2792 | 0.7128 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1750 | 0.2240 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1070 | 0.3604 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.1596 | 4.2589 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1325 | 0.5496 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0876 | 0.1062 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.2159 | 6.9091 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0283 | 0.5441 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1500 | 0.9333 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0625 | 0.1111 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0065 | 0.0068 |
| controlled_alt_vs_candidate_no_context | length_score | 0.3375 | 1.0125 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1750 | 0.2240 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1410 | 0.5364 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0848 | (0.0171, 0.1678) | 0.0027 | 0.0848 | (0.0216, 0.1692) | 0.0000 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0255 | (-0.1731, 0.0983) | 0.6360 | -0.0255 | (-0.1420, 0.1099) | 0.6337 |
| proposed_vs_candidate_no_context | naturalness | 0.0133 | (-0.0515, 0.0786) | 0.3527 | 0.0133 | (-0.0498, 0.0841) | 0.3540 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.1127 | (0.0227, 0.2263) | 0.0033 | 0.1127 | (0.0260, 0.2263) | 0.0017 |
| proposed_vs_candidate_no_context | context_overlap | 0.0197 | (0.0041, 0.0354) | 0.0040 | 0.0197 | (0.0083, 0.0335) | 0.0007 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0089 | (-0.1935, 0.1458) | 0.5510 | -0.0089 | (-0.1623, 0.1429) | 0.5223 |
| proposed_vs_candidate_no_context | persona_style | -0.0916 | (-0.2557, 0.0000) | 1.0000 | -0.0916 | (-0.2812, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | distinct1 | 0.0041 | (-0.0287, 0.0327) | 0.3840 | 0.0041 | (-0.0293, 0.0317) | 0.3837 |
| proposed_vs_candidate_no_context | length_score | 0.0583 | (-0.1958, 0.3083) | 0.3093 | 0.0583 | (-0.1750, 0.3286) | 0.3240 |
| proposed_vs_candidate_no_context | sentence_score | 0.0000 | (-0.1312, 0.1312) | 0.6500 | 0.0000 | (-0.1050, 0.1500) | 0.6420 |
| proposed_vs_candidate_no_context | overall_quality | 0.0340 | (-0.0512, 0.1188) | 0.2120 | 0.0340 | (-0.0343, 0.1320) | 0.2210 |
| controlled_vs_proposed_raw | context_relevance | 0.1595 | (0.0565, 0.2643) | 0.0007 | 0.1595 | (0.0560, 0.2906) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.2259 | (0.0886, 0.3643) | 0.0000 | 0.2259 | (0.1088, 0.3534) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0456 | (-0.0128, 0.1143) | 0.0797 | 0.0456 | (-0.0166, 0.1202) | 0.0827 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.1903 | (0.0653, 0.3153) | 0.0017 | 0.1903 | (0.0496, 0.3442) | 0.0007 |
| controlled_vs_proposed_raw | context_overlap | 0.0876 | (0.0293, 0.1588) | 0.0000 | 0.0876 | (0.0291, 0.1708) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2595 | (0.0923, 0.4494) | 0.0007 | 0.2595 | (0.1054, 0.4164) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.0916 | (0.0000, 0.2557) | 0.1003 | 0.0916 | (0.0000, 0.2922) | 0.0910 |
| controlled_vs_proposed_raw | distinct1 | 0.0103 | (-0.0085, 0.0326) | 0.1577 | 0.0103 | (-0.0073, 0.0351) | 0.1487 |
| controlled_vs_proposed_raw | length_score | 0.1417 | (-0.1125, 0.4417) | 0.1610 | 0.1417 | (-0.1333, 0.4381) | 0.1673 |
| controlled_vs_proposed_raw | sentence_score | 0.1312 | (0.0437, 0.2625) | 0.0210 | 0.1312 | (0.0350, 0.2625) | 0.0200 |
| controlled_vs_proposed_raw | overall_quality | 0.1651 | (0.0690, 0.2576) | 0.0003 | 0.1651 | (0.0780, 0.2710) | 0.0003 |
| controlled_vs_candidate_no_context | context_relevance | 0.2443 | (0.1566, 0.3240) | 0.0000 | 0.2443 | (0.1525, 0.3380) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.2005 | (0.0833, 0.3248) | 0.0000 | 0.2005 | (0.0857, 0.3439) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.0589 | (-0.0052, 0.1152) | 0.0350 | 0.0589 | (0.0045, 0.1229) | 0.0140 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3030 | (0.1932, 0.3939) | 0.0000 | 0.3030 | (0.1939, 0.4026) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.1073 | (0.0466, 0.1716) | 0.0000 | 0.1073 | (0.0469, 0.1885) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2506 | (0.1042, 0.4196) | 0.0000 | 0.2506 | (0.1071, 0.4340) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | distinct1 | 0.0143 | (-0.0177, 0.0465) | 0.2063 | 0.0143 | (-0.0192, 0.0463) | 0.2007 |
| controlled_vs_candidate_no_context | length_score | 0.2000 | (-0.0293, 0.4125) | 0.0550 | 0.2000 | (0.0111, 0.4381) | 0.0180 |
| controlled_vs_candidate_no_context | sentence_score | 0.1312 | (-0.0437, 0.2625) | 0.0987 | 0.1312 | (-0.0350, 0.3000) | 0.1070 |
| controlled_vs_candidate_no_context | overall_quality | 0.1991 | (0.1150, 0.2845) | 0.0000 | 0.1991 | (0.1109, 0.2987) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0847 | (-0.1739, 0.0146) | 0.9537 | -0.0847 | (-0.1782, -0.0119) | 0.9917 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0680 | (-0.1842, 0.0426) | 0.8740 | -0.0680 | (-0.1649, 0.0408) | 0.8923 |
| controlled_alt_vs_controlled_default | naturalness | 0.0287 | (-0.0322, 0.0905) | 0.1820 | 0.0287 | (-0.0254, 0.0940) | 0.1480 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0871 | (-0.1970, 0.0341) | 0.9310 | -0.0871 | (-0.1949, -0.0091) | 0.9893 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0790 | (-0.1386, -0.0206) | 0.9983 | -0.0790 | (-0.1480, -0.0263) | 1.0000 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.1006 | (-0.2500, 0.0446) | 0.9080 | -0.1006 | (-0.2244, 0.0510) | 0.9100 |
| controlled_alt_vs_controlled_default | persona_style | 0.0625 | (0.0000, 0.1562) | 0.1073 | 0.0625 | (0.0000, 0.1500) | 0.3640 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0079 | (-0.0411, 0.0247) | 0.6687 | -0.0079 | (-0.0381, 0.0303) | 0.6843 |
| controlled_alt_vs_controlled_default | length_score | 0.1375 | (-0.1625, 0.4125) | 0.1817 | 0.1375 | (-0.1333, 0.4333) | 0.1687 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0437 | (-0.0875, 0.1750) | 0.3810 | 0.0437 | (-0.1000, 0.1750) | 0.3677 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0581 | (-0.0896, -0.0248) | 1.0000 | -0.0581 | (-0.0963, -0.0248) | 0.9990 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.0748 | (-0.0225, 0.1705) | 0.0740 | 0.0748 | (0.0043, 0.1624) | 0.0223 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1580 | (0.0496, 0.2611) | 0.0037 | 0.1580 | (0.0563, 0.2726) | 0.0033 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0743 | (0.0216, 0.1282) | 0.0027 | 0.0743 | (0.0261, 0.1375) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.1032 | (-0.0218, 0.2273) | 0.0533 | 0.1032 | (0.0109, 0.2159) | 0.0227 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0086 | (-0.0229, 0.0380) | 0.2997 | 0.0086 | (-0.0146, 0.0332) | 0.2437 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1589 | (0.0327, 0.2720) | 0.0063 | 0.1589 | (0.0462, 0.2932) | 0.0057 |
| controlled_alt_vs_proposed_raw | persona_style | 0.1541 | (0.0312, 0.3203) | 0.0050 | 0.1541 | (0.0110, 0.3087) | 0.0190 |
| controlled_alt_vs_proposed_raw | distinct1 | 0.0024 | (-0.0271, 0.0343) | 0.4417 | 0.0024 | (-0.0271, 0.0422) | 0.4453 |
| controlled_alt_vs_proposed_raw | length_score | 0.2792 | (0.0916, 0.4708) | 0.0013 | 0.2792 | (0.1074, 0.5048) | 0.0000 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1750 | (0.0437, 0.3062) | 0.0043 | 0.1750 | (0.0500, 0.2722) | 0.0030 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1070 | (0.0250, 0.1850) | 0.0043 | 0.1070 | (0.0288, 0.1999) | 0.0023 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.1596 | (0.0796, 0.2198) | 0.0000 | 0.1596 | (0.0989, 0.2222) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1325 | (-0.0002, 0.2453) | 0.0277 | 0.1325 | (0.0071, 0.2612) | 0.0197 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0876 | (0.0179, 0.1501) | 0.0073 | 0.0876 | (0.0222, 0.1651) | 0.0030 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.2159 | (0.1136, 0.2955) | 0.0000 | 0.2159 | (0.1364, 0.2987) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0283 | (0.0047, 0.0491) | 0.0087 | 0.0283 | (0.0077, 0.0497) | 0.0057 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1500 | (-0.0446, 0.3018) | 0.0563 | 0.1500 | (-0.0358, 0.3231) | 0.0633 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0625 | (0.0000, 0.1562) | 0.1007 | 0.0625 | (0.0000, 0.1500) | 0.3457 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0065 | (-0.0049, 0.0193) | 0.1603 | 0.0065 | (-0.0068, 0.0209) | 0.2017 |
| controlled_alt_vs_candidate_no_context | length_score | 0.3375 | (0.0542, 0.5917) | 0.0093 | 0.3375 | (0.0667, 0.6667) | 0.0127 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1750 | (0.0437, 0.3062) | 0.0060 | 0.1750 | (0.0437, 0.3000) | 0.0040 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1410 | (0.0684, 0.2012) | 0.0000 | 0.1410 | (0.0660, 0.2115) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 5 | 1 | 2 | 0.7500 | 0.8333 |
| proposed_vs_candidate_no_context | persona_consistency | 2 | 2 | 4 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | naturalness | 4 | 2 | 2 | 0.6250 | 0.6667 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 4 | 0 | 4 | 0.7500 | 1.0000 |
| proposed_vs_candidate_no_context | context_overlap | 5 | 1 | 2 | 0.7500 | 0.8333 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 2 | 1 | 5 | 0.5625 | 0.6667 |
| proposed_vs_candidate_no_context | persona_style | 0 | 2 | 6 | 0.3750 | 0.0000 |
| proposed_vs_candidate_no_context | distinct1 | 3 | 2 | 3 | 0.5625 | 0.6000 |
| proposed_vs_candidate_no_context | length_score | 4 | 2 | 2 | 0.6250 | 0.6667 |
| proposed_vs_candidate_no_context | sentence_score | 1 | 1 | 6 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | overall_quality | 4 | 2 | 2 | 0.6250 | 0.6667 |
| controlled_vs_proposed_raw | context_relevance | 5 | 2 | 1 | 0.6875 | 0.7143 |
| controlled_vs_proposed_raw | persona_consistency | 6 | 0 | 2 | 0.8750 | 1.0000 |
| controlled_vs_proposed_raw | naturalness | 5 | 2 | 1 | 0.6875 | 0.7143 |
| controlled_vs_proposed_raw | context_keyword_coverage | 5 | 1 | 2 | 0.7500 | 0.8333 |
| controlled_vs_proposed_raw | context_overlap | 6 | 1 | 1 | 0.8125 | 0.8571 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 5 | 0 | 3 | 0.8125 | 1.0000 |
| controlled_vs_proposed_raw | persona_style | 2 | 0 | 6 | 0.6250 | 1.0000 |
| controlled_vs_proposed_raw | distinct1 | 4 | 2 | 2 | 0.6250 | 0.6667 |
| controlled_vs_proposed_raw | length_score | 4 | 3 | 1 | 0.5625 | 0.5714 |
| controlled_vs_proposed_raw | sentence_score | 3 | 0 | 5 | 0.6875 | 1.0000 |
| controlled_vs_proposed_raw | overall_quality | 6 | 1 | 1 | 0.8125 | 0.8571 |
| controlled_vs_candidate_no_context | context_relevance | 7 | 1 | 0 | 0.8750 | 0.8750 |
| controlled_vs_candidate_no_context | persona_consistency | 6 | 0 | 2 | 0.8750 | 1.0000 |
| controlled_vs_candidate_no_context | naturalness | 6 | 2 | 0 | 0.7500 | 0.7500 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 7 | 0 | 1 | 0.9375 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 7 | 1 | 0 | 0.8750 | 0.8750 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 6 | 0 | 2 | 0.8750 | 1.0000 |
| controlled_vs_candidate_no_context | persona_style | 0 | 0 | 8 | 0.5000 | nan |
| controlled_vs_candidate_no_context | distinct1 | 4 | 3 | 1 | 0.5625 | 0.5714 |
| controlled_vs_candidate_no_context | length_score | 5 | 3 | 0 | 0.6250 | 0.6250 |
| controlled_vs_candidate_no_context | sentence_score | 4 | 1 | 3 | 0.6875 | 0.8000 |
| controlled_vs_candidate_no_context | overall_quality | 8 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 2 | 6 | 0 | 0.2500 | 0.2500 |
| controlled_alt_vs_controlled_default | persona_consistency | 3 | 4 | 1 | 0.4375 | 0.4286 |
| controlled_alt_vs_controlled_default | naturalness | 5 | 3 | 0 | 0.6250 | 0.6250 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 1 | 5 | 2 | 0.2500 | 0.1667 |
| controlled_alt_vs_controlled_default | context_overlap | 2 | 6 | 0 | 0.2500 | 0.2500 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 2 | 4 | 2 | 0.3750 | 0.3333 |
| controlled_alt_vs_controlled_default | persona_style | 2 | 0 | 6 | 0.6250 | 1.0000 |
| controlled_alt_vs_controlled_default | distinct1 | 3 | 4 | 1 | 0.4375 | 0.4286 |
| controlled_alt_vs_controlled_default | length_score | 5 | 2 | 1 | 0.6875 | 0.7143 |
| controlled_alt_vs_controlled_default | sentence_score | 2 | 1 | 5 | 0.5625 | 0.6667 |
| controlled_alt_vs_controlled_default | overall_quality | 1 | 7 | 0 | 0.1250 | 0.1250 |
| controlled_alt_vs_proposed_raw | context_relevance | 5 | 3 | 0 | 0.6250 | 0.6250 |
| controlled_alt_vs_proposed_raw | persona_consistency | 7 | 1 | 0 | 0.8750 | 0.8750 |
| controlled_alt_vs_proposed_raw | naturalness | 7 | 1 | 0 | 0.8750 | 0.8750 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 5 | 3 | 0 | 0.6250 | 0.6250 |
| controlled_alt_vs_proposed_raw | context_overlap | 5 | 3 | 0 | 0.6250 | 0.6250 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 6 | 1 | 1 | 0.8125 | 0.8571 |
| controlled_alt_vs_proposed_raw | persona_style | 4 | 0 | 4 | 0.7500 | 1.0000 |
| controlled_alt_vs_proposed_raw | distinct1 | 3 | 3 | 2 | 0.5000 | 0.5000 |
| controlled_alt_vs_proposed_raw | length_score | 7 | 1 | 0 | 0.8750 | 0.8750 |
| controlled_alt_vs_proposed_raw | sentence_score | 4 | 0 | 4 | 0.7500 | 1.0000 |
| controlled_alt_vs_proposed_raw | overall_quality | 6 | 2 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_candidate_no_context | context_relevance | 6 | 2 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 7 | 1 | 0 | 0.8750 | 0.8750 |
| controlled_alt_vs_candidate_no_context | naturalness | 5 | 3 | 0 | 0.6250 | 0.6250 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 6 | 0 | 2 | 0.8750 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 6 | 2 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 6 | 1 | 1 | 0.8125 | 0.8571 |
| controlled_alt_vs_candidate_no_context | persona_style | 2 | 0 | 6 | 0.6250 | 1.0000 |
| controlled_alt_vs_candidate_no_context | distinct1 | 3 | 2 | 3 | 0.5625 | 0.6000 |
| controlled_alt_vs_candidate_no_context | length_score | 5 | 3 | 0 | 0.6250 | 0.6250 |
| controlled_alt_vs_candidate_no_context | sentence_score | 4 | 0 | 4 | 0.7500 | 1.0000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 7 | 1 | 0 | 0.8750 | 0.8750 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.1250 | 0.6250 | 0.3750 |
| proposed_contextual_controlled_hb | 0.0000 | 0.0000 | 0.3750 | 0.5000 | 0.5000 |
| proposed_contextual | 0.0000 | 0.0000 | 0.3750 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.3750 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `7`
- Unique template signatures: `7`
- Template signature ratio: `0.8750`
- Effective sample size by source clustering: `6.40`
- Effective sample size by template-signature clustering: `6.40`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: unavailable (disabled_by_flag).

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
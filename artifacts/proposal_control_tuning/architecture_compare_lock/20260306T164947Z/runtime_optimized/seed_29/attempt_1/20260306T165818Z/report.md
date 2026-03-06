# Proposal Alignment Evaluation Report

- Run ID: `20260306T165818Z`
- Generated: `2026-03-06T17:02:59.880000+00:00`
- Scenarios: `artifacts\proposal_control_tuning\architecture_compare_lock\20260306T164947Z\runtime_optimized\seed_29\attempt_1\20260306T165818Z\scenarios.jsonl`
- Scenario count: `32`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_alt`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2554 (0.2306, 0.2820) | 0.3019 (0.2629, 0.3451) | 0.9063 (0.8850, 0.9258) | 0.4033 (0.3869, 0.4188) | n/a |
| proposed_contextual_controlled_alt | 0.2543 (0.2274, 0.2791) | 0.3019 (0.2568, 0.3499) | 0.9255 (0.9039, 0.9439) | 0.4065 (0.3879, 0.4251) | n/a |
| proposed_contextual | 0.0571 (0.0309, 0.0898) | 0.1395 (0.0989, 0.1878) | 0.8123 (0.7860, 0.8423) | 0.2332 (0.2070, 0.2641) | n/a |
| candidate_no_context | 0.0213 (0.0144, 0.0302) | 0.1734 (0.1184, 0.2311) | 0.8064 (0.7819, 0.8321) | 0.2281 (0.2022, 0.2549) | n/a |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0358 | 1.6791 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0340 | -0.1958 |
| proposed_vs_candidate_no_context | naturalness | 0.0058 | 0.0072 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0455 | 3.2067 |
| proposed_vs_candidate_no_context | context_overlap | 0.0129 | 0.3419 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0397 | -0.4003 |
| proposed_vs_candidate_no_context | persona_style | -0.0108 | -0.0230 |
| proposed_vs_candidate_no_context | distinct1 | -0.0022 | -0.0024 |
| proposed_vs_candidate_no_context | length_score | 0.0281 | 0.0978 |
| proposed_vs_candidate_no_context | sentence_score | 0.0109 | 0.0146 |
| proposed_vs_candidate_no_context | overall_quality | 0.0051 | 0.0224 |
| controlled_vs_proposed_raw | context_relevance | 0.1983 | 3.4746 |
| controlled_vs_proposed_raw | persona_consistency | 0.1624 | 1.1644 |
| controlled_vs_proposed_raw | naturalness | 0.0940 | 0.1157 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2565 | 4.2924 |
| controlled_vs_proposed_raw | context_overlap | 0.0625 | 1.2302 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1777 | 2.9850 |
| controlled_vs_proposed_raw | persona_style | 0.1013 | 0.2206 |
| controlled_vs_proposed_raw | distinct1 | -0.0005 | -0.0006 |
| controlled_vs_proposed_raw | length_score | 0.3781 | 1.1980 |
| controlled_vs_proposed_raw | sentence_score | 0.1859 | 0.2449 |
| controlled_vs_proposed_raw | overall_quality | 0.1701 | 0.7294 |
| controlled_vs_candidate_no_context | context_relevance | 0.2341 | 10.9881 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1285 | 0.7407 |
| controlled_vs_candidate_no_context | naturalness | 0.0998 | 0.1238 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3020 | 21.2633 |
| controlled_vs_candidate_no_context | context_overlap | 0.0754 | 1.9926 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1379 | 1.3898 |
| controlled_vs_candidate_no_context | persona_style | 0.0905 | 0.1925 |
| controlled_vs_candidate_no_context | distinct1 | -0.0028 | -0.0029 |
| controlled_vs_candidate_no_context | length_score | 0.4062 | 1.4130 |
| controlled_vs_candidate_no_context | sentence_score | 0.1969 | 0.2630 |
| controlled_vs_candidate_no_context | overall_quality | 0.1752 | 0.7682 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0011 | -0.0042 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0000 | 0.0000 |
| controlled_alt_vs_controlled_default | naturalness | 0.0192 | 0.0212 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0013 | -0.0042 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0005 | -0.0045 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0024 | -0.0100 |
| controlled_alt_vs_controlled_default | persona_style | 0.0096 | 0.0170 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0139 | 0.0149 |
| controlled_alt_vs_controlled_default | length_score | 0.0573 | 0.0826 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0219 | 0.0231 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0032 | 0.0079 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1972 | 3.4557 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1624 | 1.1645 |
| controlled_alt_vs_proposed_raw | naturalness | 0.1132 | 0.1394 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2552 | 4.2702 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0620 | 1.2201 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1753 | 2.9450 |
| controlled_alt_vs_proposed_raw | persona_style | 0.1109 | 0.2414 |
| controlled_alt_vs_proposed_raw | distinct1 | 0.0134 | 0.0143 |
| controlled_alt_vs_proposed_raw | length_score | 0.4354 | 1.3795 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.2078 | 0.2737 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1733 | 0.7431 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2330 | 10.9374 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1285 | 0.7407 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.1190 | 0.1476 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3007 | 21.1700 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0749 | 1.9791 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1356 | 1.3658 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.1000 | 0.2128 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0111 | 0.0119 |
| controlled_alt_vs_candidate_no_context | length_score | 0.4635 | 1.6123 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.2188 | 0.2923 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1784 | 0.7822 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0358 | (0.0094, 0.0670) | 0.0010 | 0.0358 | (0.0076, 0.0662) | 0.0040 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0340 | (-0.0698, -0.0005) | 0.9753 | -0.0340 | (-0.0565, -0.0160) | 1.0000 |
| proposed_vs_candidate_no_context | naturalness | 0.0058 | (-0.0129, 0.0252) | 0.2653 | 0.0058 | (-0.0149, 0.0313) | 0.3257 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0455 | (0.0107, 0.0849) | 0.0040 | 0.0455 | (0.0079, 0.0859) | 0.0077 |
| proposed_vs_candidate_no_context | context_overlap | 0.0129 | (0.0033, 0.0238) | 0.0030 | 0.0129 | (0.0033, 0.0249) | 0.0040 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0397 | (-0.0832, 0.0015) | 0.9673 | -0.0397 | (-0.0696, -0.0162) | 1.0000 |
| proposed_vs_candidate_no_context | persona_style | -0.0108 | (-0.0455, 0.0178) | 0.7373 | -0.0108 | (-0.0416, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | distinct1 | -0.0022 | (-0.0110, 0.0068) | 0.6983 | -0.0022 | (-0.0074, 0.0033) | 0.7977 |
| proposed_vs_candidate_no_context | length_score | 0.0281 | (-0.0563, 0.1115) | 0.2463 | 0.0281 | (-0.0710, 0.1513) | 0.2997 |
| proposed_vs_candidate_no_context | sentence_score | 0.0109 | (-0.0328, 0.0656) | 0.4080 | 0.0109 | (0.0000, 0.0420) | 0.3463 |
| proposed_vs_candidate_no_context | overall_quality | 0.0051 | (-0.0102, 0.0231) | 0.2617 | 0.0051 | (-0.0130, 0.0261) | 0.3353 |
| controlled_vs_proposed_raw | context_relevance | 0.1983 | (0.1574, 0.2322) | 0.0000 | 0.1983 | (0.1591, 0.2327) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.1624 | (0.1083, 0.2107) | 0.0000 | 0.1624 | (0.0767, 0.2261) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0940 | (0.0544, 0.1310) | 0.0000 | 0.0940 | (0.0313, 0.1472) | 0.0033 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2565 | (0.2003, 0.3007) | 0.0000 | 0.2565 | (0.2068, 0.2982) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0625 | (0.0469, 0.0762) | 0.0000 | 0.0625 | (0.0423, 0.0790) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1777 | (0.1205, 0.2329) | 0.0000 | 0.1777 | (0.0917, 0.2204) | 0.0010 |
| controlled_vs_proposed_raw | persona_style | 0.1013 | (0.0231, 0.1918) | 0.0047 | 0.1013 | (-0.0256, 0.2678) | 0.0943 |
| controlled_vs_proposed_raw | distinct1 | -0.0005 | (-0.0175, 0.0150) | 0.4900 | -0.0005 | (-0.0233, 0.0194) | 0.5177 |
| controlled_vs_proposed_raw | length_score | 0.3781 | (0.2302, 0.5167) | 0.0000 | 0.3781 | (0.1474, 0.5754) | 0.0013 |
| controlled_vs_proposed_raw | sentence_score | 0.1859 | (0.1094, 0.2516) | 0.0000 | 0.1859 | (0.0729, 0.2579) | 0.0023 |
| controlled_vs_proposed_raw | overall_quality | 0.1701 | (0.1379, 0.2006) | 0.0000 | 0.1701 | (0.1200, 0.2042) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2341 | (0.2058, 0.2633) | 0.0000 | 0.2341 | (0.2046, 0.2635) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1285 | (0.0625, 0.1915) | 0.0000 | 0.1285 | (0.0272, 0.1869) | 0.0103 |
| controlled_vs_candidate_no_context | naturalness | 0.0998 | (0.0650, 0.1330) | 0.0000 | 0.0998 | (0.0513, 0.1427) | 0.0003 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3020 | (0.2644, 0.3413) | 0.0000 | 0.3020 | (0.2646, 0.3396) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0754 | (0.0652, 0.0868) | 0.0000 | 0.0754 | (0.0624, 0.0842) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1379 | (0.0668, 0.2048) | 0.0000 | 0.1379 | (0.0349, 0.1863) | 0.0050 |
| controlled_vs_candidate_no_context | persona_style | 0.0905 | (0.0021, 0.1818) | 0.0233 | 0.0905 | (-0.0383, 0.2579) | 0.1403 |
| controlled_vs_candidate_no_context | distinct1 | -0.0028 | (-0.0187, 0.0125) | 0.6467 | -0.0028 | (-0.0270, 0.0153) | 0.6330 |
| controlled_vs_candidate_no_context | length_score | 0.4062 | (0.2760, 0.5312) | 0.0000 | 0.4062 | (0.2333, 0.5703) | 0.0000 |
| controlled_vs_candidate_no_context | sentence_score | 0.1969 | (0.1203, 0.2734) | 0.0000 | 0.1969 | (0.0955, 0.2600) | 0.0000 |
| controlled_vs_candidate_no_context | overall_quality | 0.1752 | (0.1441, 0.2046) | 0.0000 | 0.1752 | (0.1195, 0.2084) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0011 | (-0.0383, 0.0347) | 0.5217 | -0.0011 | (-0.0342, 0.0249) | 0.5107 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0000 | (-0.0583, 0.0580) | 0.4983 | 0.0000 | (-0.0417, 0.0693) | 0.4973 |
| controlled_alt_vs_controlled_default | naturalness | 0.0192 | (-0.0034, 0.0414) | 0.0543 | 0.0192 | (-0.0056, 0.0398) | 0.0657 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0013 | (-0.0460, 0.0412) | 0.5110 | -0.0013 | (-0.0410, 0.0314) | 0.5250 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0005 | (-0.0195, 0.0215) | 0.5310 | -0.0005 | (-0.0131, 0.0116) | 0.5213 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0024 | (-0.0729, 0.0661) | 0.5240 | -0.0024 | (-0.0506, 0.0762) | 0.5397 |
| controlled_alt_vs_controlled_default | persona_style | 0.0096 | (-0.0202, 0.0401) | 0.2637 | 0.0096 | (-0.0038, 0.0322) | 0.1280 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0139 | (-0.0002, 0.0284) | 0.0270 | 0.0139 | (0.0039, 0.0234) | 0.0027 |
| controlled_alt_vs_controlled_default | length_score | 0.0573 | (-0.0437, 0.1646) | 0.1420 | 0.0573 | (-0.0500, 0.1455) | 0.1493 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0219 | (-0.0328, 0.0766) | 0.2710 | 0.0219 | (0.0000, 0.0560) | 0.0947 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0032 | (-0.0201, 0.0262) | 0.3960 | 0.0032 | (-0.0167, 0.0275) | 0.3703 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1972 | (0.1575, 0.2342) | 0.0000 | 0.1972 | (0.1525, 0.2320) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1624 | (0.1153, 0.2045) | 0.0000 | 0.1624 | (0.1121, 0.2215) | 0.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 0.1132 | (0.0733, 0.1495) | 0.0000 | 0.1132 | (0.0473, 0.1580) | 0.0007 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2552 | (0.2011, 0.3041) | 0.0000 | 0.2552 | (0.2025, 0.2994) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0620 | (0.0456, 0.0797) | 0.0000 | 0.0620 | (0.0420, 0.0799) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1753 | (0.1238, 0.2240) | 0.0000 | 0.1753 | (0.1348, 0.2267) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 0.1109 | (0.0337, 0.2035) | 0.0003 | 0.1109 | (-0.0089, 0.2769) | 0.0463 |
| controlled_alt_vs_proposed_raw | distinct1 | 0.0134 | (-0.0036, 0.0293) | 0.0617 | 0.0134 | (-0.0057, 0.0279) | 0.0787 |
| controlled_alt_vs_proposed_raw | length_score | 0.4354 | (0.2843, 0.5750) | 0.0000 | 0.4354 | (0.1897, 0.6079) | 0.0007 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.2078 | (0.1422, 0.2734) | 0.0000 | 0.2078 | (0.1120, 0.2700) | 0.0007 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1733 | (0.1435, 0.2020) | 0.0000 | 0.1733 | (0.1362, 0.2049) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2330 | (0.2065, 0.2578) | 0.0000 | 0.2330 | (0.2063, 0.2512) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1285 | (0.0804, 0.1785) | 0.0000 | 0.1285 | (0.0795, 0.1801) | 0.0000 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.1190 | (0.0814, 0.1562) | 0.0000 | 0.1190 | (0.0588, 0.1565) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3007 | (0.2648, 0.3320) | 0.0000 | 0.3007 | (0.2663, 0.3229) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0749 | (0.0597, 0.0923) | 0.0000 | 0.0749 | (0.0646, 0.0862) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1356 | (0.0808, 0.1896) | 0.0000 | 0.1356 | (0.0948, 0.1806) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.1000 | (0.0255, 0.1850) | 0.0033 | 0.1000 | (-0.0198, 0.2688) | 0.0947 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0111 | (-0.0059, 0.0278) | 0.0930 | 0.0111 | (-0.0081, 0.0257) | 0.1193 |
| controlled_alt_vs_candidate_no_context | length_score | 0.4635 | (0.3115, 0.6083) | 0.0000 | 0.4635 | (0.2463, 0.6135) | 0.0000 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.2188 | (0.1531, 0.2844) | 0.0000 | 0.2188 | (0.1346, 0.2713) | 0.0000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1784 | (0.1519, 0.2040) | 0.0000 | 0.1784 | (0.1409, 0.2043) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 10 | 4 | 18 | 0.5938 | 0.7143 |
| proposed_vs_candidate_no_context | persona_consistency | 3 | 8 | 21 | 0.4219 | 0.2727 |
| proposed_vs_candidate_no_context | naturalness | 10 | 5 | 17 | 0.5781 | 0.6667 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 9 | 2 | 21 | 0.6094 | 0.8182 |
| proposed_vs_candidate_no_context | context_overlap | 10 | 4 | 18 | 0.5938 | 0.7143 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 1 | 7 | 24 | 0.4062 | 0.1250 |
| proposed_vs_candidate_no_context | persona_style | 2 | 2 | 28 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | distinct1 | 3 | 10 | 19 | 0.3906 | 0.2308 |
| proposed_vs_candidate_no_context | length_score | 9 | 6 | 17 | 0.5469 | 0.6000 |
| proposed_vs_candidate_no_context | sentence_score | 3 | 2 | 27 | 0.5156 | 0.6000 |
| proposed_vs_candidate_no_context | overall_quality | 8 | 7 | 17 | 0.5156 | 0.5333 |
| controlled_vs_proposed_raw | context_relevance | 31 | 1 | 0 | 0.9688 | 0.9688 |
| controlled_vs_proposed_raw | persona_consistency | 25 | 2 | 5 | 0.8594 | 0.9259 |
| controlled_vs_proposed_raw | naturalness | 23 | 9 | 0 | 0.7188 | 0.7188 |
| controlled_vs_proposed_raw | context_keyword_coverage | 30 | 1 | 1 | 0.9531 | 0.9677 |
| controlled_vs_proposed_raw | context_overlap | 30 | 2 | 0 | 0.9375 | 0.9375 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 25 | 2 | 5 | 0.8594 | 0.9259 |
| controlled_vs_proposed_raw | persona_style | 8 | 3 | 21 | 0.5781 | 0.7273 |
| controlled_vs_proposed_raw | distinct1 | 21 | 11 | 0 | 0.6562 | 0.6562 |
| controlled_vs_proposed_raw | length_score | 22 | 10 | 0 | 0.6875 | 0.6875 |
| controlled_vs_proposed_raw | sentence_score | 19 | 2 | 11 | 0.7656 | 0.9048 |
| controlled_vs_proposed_raw | overall_quality | 29 | 3 | 0 | 0.9062 | 0.9062 |
| controlled_vs_candidate_no_context | context_relevance | 32 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 22 | 3 | 7 | 0.7969 | 0.8800 |
| controlled_vs_candidate_no_context | naturalness | 28 | 4 | 0 | 0.8750 | 0.8750 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 31 | 0 | 1 | 0.9844 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 32 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 21 | 3 | 8 | 0.7812 | 0.8750 |
| controlled_vs_candidate_no_context | persona_style | 8 | 4 | 20 | 0.5625 | 0.6667 |
| controlled_vs_candidate_no_context | distinct1 | 18 | 14 | 0 | 0.5625 | 0.5625 |
| controlled_vs_candidate_no_context | length_score | 27 | 5 | 0 | 0.8438 | 0.8438 |
| controlled_vs_candidate_no_context | sentence_score | 20 | 2 | 10 | 0.7812 | 0.9091 |
| controlled_vs_candidate_no_context | overall_quality | 31 | 1 | 0 | 0.9688 | 0.9688 |
| controlled_alt_vs_controlled_default | context_relevance | 11 | 11 | 10 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | persona_consistency | 9 | 8 | 15 | 0.5156 | 0.5294 |
| controlled_alt_vs_controlled_default | naturalness | 14 | 8 | 10 | 0.5938 | 0.6364 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 9 | 8 | 15 | 0.5156 | 0.5294 |
| controlled_alt_vs_controlled_default | context_overlap | 8 | 14 | 10 | 0.4062 | 0.3636 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 6 | 7 | 19 | 0.4844 | 0.4615 |
| controlled_alt_vs_controlled_default | persona_style | 5 | 3 | 24 | 0.5312 | 0.6250 |
| controlled_alt_vs_controlled_default | distinct1 | 15 | 7 | 10 | 0.6250 | 0.6818 |
| controlled_alt_vs_controlled_default | length_score | 12 | 9 | 11 | 0.5469 | 0.5714 |
| controlled_alt_vs_controlled_default | sentence_score | 4 | 2 | 26 | 0.5312 | 0.6667 |
| controlled_alt_vs_controlled_default | overall_quality | 12 | 10 | 10 | 0.5312 | 0.5455 |
| controlled_alt_vs_proposed_raw | context_relevance | 30 | 2 | 0 | 0.9375 | 0.9375 |
| controlled_alt_vs_proposed_raw | persona_consistency | 26 | 2 | 4 | 0.8750 | 0.9286 |
| controlled_alt_vs_proposed_raw | naturalness | 25 | 7 | 0 | 0.7812 | 0.7812 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 30 | 2 | 0 | 0.9375 | 0.9375 |
| controlled_alt_vs_proposed_raw | context_overlap | 31 | 1 | 0 | 0.9688 | 0.9688 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 26 | 2 | 4 | 0.8750 | 0.9286 |
| controlled_alt_vs_proposed_raw | persona_style | 8 | 3 | 21 | 0.5781 | 0.7273 |
| controlled_alt_vs_proposed_raw | distinct1 | 23 | 9 | 0 | 0.7188 | 0.7188 |
| controlled_alt_vs_proposed_raw | length_score | 26 | 6 | 0 | 0.8125 | 0.8125 |
| controlled_alt_vs_proposed_raw | sentence_score | 20 | 1 | 11 | 0.7969 | 0.9524 |
| controlled_alt_vs_proposed_raw | overall_quality | 30 | 2 | 0 | 0.9375 | 0.9375 |
| controlled_alt_vs_candidate_no_context | context_relevance | 32 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 21 | 4 | 7 | 0.7656 | 0.8400 |
| controlled_alt_vs_candidate_no_context | naturalness | 27 | 5 | 0 | 0.8438 | 0.8438 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 31 | 0 | 1 | 0.9844 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 32 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 21 | 3 | 8 | 0.7812 | 0.8750 |
| controlled_alt_vs_candidate_no_context | persona_style | 9 | 2 | 21 | 0.6094 | 0.8182 |
| controlled_alt_vs_candidate_no_context | distinct1 | 24 | 8 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_candidate_no_context | length_score | 26 | 6 | 0 | 0.8125 | 0.8125 |
| controlled_alt_vs_candidate_no_context | sentence_score | 21 | 1 | 10 | 0.8125 | 0.9545 |
| controlled_alt_vs_candidate_no_context | overall_quality | 32 | 0 | 0 | 1.0000 | 1.0000 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.0000 | 0.4062 | 0.1875 |
| proposed_contextual_controlled_alt | 0.0000 | 0.0000 | 0.0000 | 0.4375 | 0.1875 |
| proposed_contextual | 0.0000 | 0.0000 | 0.5625 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.5625 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `8`
- Unique template signatures: `28`
- Template signature ratio: `0.8750`
- Effective sample size by source clustering: `6.83`
- Effective sample size by template-signature clustering: `24.38`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: unavailable (disabled_by_flag).

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
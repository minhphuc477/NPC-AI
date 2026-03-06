# Proposal Alignment Evaluation Report

- Run ID: `20260305T211219Z`
- Generated: `2026-03-05T21:16:23.022985+00:00`
- Scenarios: `artifacts\proposal_control_tuning\auto_tune_v2b\20260305T210504Z\train_runs\trial_002\seed_19\20260305T211219Z\scenarios.jsonl`
- Scenario count: `20`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_tuned`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2953 (0.2646, 0.3292) | 0.3006 (0.2434, 0.3607) | 0.8873 (0.8608, 0.9125) | 0.4173 (0.3935, 0.4409) | n/a |
| proposed_contextual_controlled_tuned | 0.3204 (0.2882, 0.3530) | 0.3568 (0.2908, 0.4314) | 0.8754 (0.8459, 0.9020) | 0.4478 (0.4218, 0.4728) | n/a |
| proposed_contextual | 0.1115 (0.0594, 0.1677) | 0.1700 (0.1166, 0.2350) | 0.8263 (0.7936, 0.8608) | 0.2711 (0.2266, 0.3186) | n/a |
| candidate_no_context | 0.0178 (0.0105, 0.0278) | 0.1407 (0.0950, 0.1907) | 0.7915 (0.7620, 0.8260) | 0.2096 (0.1867, 0.2372) | n/a |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0937 | 5.2608 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0293 | 0.2080 |
| proposed_vs_candidate_no_context | naturalness | 0.0349 | 0.0441 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.1227 | 13.5000 |
| proposed_vs_candidate_no_context | context_overlap | 0.0259 | 0.6798 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0369 | 1.1481 |
| proposed_vs_candidate_no_context | persona_style | -0.0012 | -0.0022 |
| proposed_vs_candidate_no_context | distinct1 | 0.0126 | 0.0135 |
| proposed_vs_candidate_no_context | length_score | 0.0967 | 0.3791 |
| proposed_vs_candidate_no_context | sentence_score | 0.1050 | 0.1533 |
| proposed_vs_candidate_no_context | overall_quality | 0.0615 | 0.2935 |
| controlled_vs_proposed_raw | context_relevance | 0.1838 | 1.6481 |
| controlled_vs_proposed_raw | persona_consistency | 0.1306 | 0.7684 |
| controlled_vs_proposed_raw | naturalness | 0.0609 | 0.0737 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2409 | 1.8276 |
| controlled_vs_proposed_raw | context_overlap | 0.0504 | 0.7867 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1562 | 2.2621 |
| controlled_vs_proposed_raw | persona_style | 0.0283 | 0.0494 |
| controlled_vs_proposed_raw | distinct1 | -0.0002 | -0.0002 |
| controlled_vs_proposed_raw | length_score | 0.2700 | 0.7678 |
| controlled_vs_proposed_raw | sentence_score | 0.0700 | 0.0886 |
| controlled_vs_proposed_raw | overall_quality | 0.1462 | 0.5392 |
| controlled_vs_candidate_no_context | context_relevance | 0.2775 | 15.5791 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1599 | 1.1363 |
| controlled_vs_candidate_no_context | naturalness | 0.0958 | 0.1210 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3636 | 40.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0764 | 2.0013 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1931 | 6.0074 |
| controlled_vs_candidate_no_context | persona_style | 0.0271 | 0.0471 |
| controlled_vs_candidate_no_context | distinct1 | 0.0124 | 0.0134 |
| controlled_vs_candidate_no_context | length_score | 0.3667 | 1.4379 |
| controlled_vs_candidate_no_context | sentence_score | 0.1750 | 0.2555 |
| controlled_vs_candidate_no_context | overall_quality | 0.2077 | 0.9910 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0251 | 0.0851 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0562 | 0.1869 |
| controlled_alt_vs_controlled_default | naturalness | -0.0119 | -0.0134 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0364 | 0.0976 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0011 | -0.0095 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0686 | 0.3044 |
| controlled_alt_vs_controlled_default | persona_style | 0.0067 | 0.0111 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0063 | -0.0067 |
| controlled_alt_vs_controlled_default | length_score | -0.0817 | -0.1314 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0700 | 0.0814 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0306 | 0.0732 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.2089 | 1.8735 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1868 | 1.0990 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0491 | 0.0594 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2773 | 2.1034 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0493 | 0.7697 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.2248 | 3.2552 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0350 | 0.0610 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0065 | -0.0069 |
| controlled_alt_vs_proposed_raw | length_score | 0.1883 | 0.5355 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1400 | 0.1772 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1767 | 0.6519 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.3026 | 16.9901 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.2161 | 1.5356 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0839 | 0.1061 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.4000 | 44.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0753 | 1.9728 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.2617 | 8.1407 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0338 | 0.0587 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0061 | 0.0066 |
| controlled_alt_vs_candidate_no_context | length_score | 0.2850 | 1.1176 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.2450 | 0.3577 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.2382 | 1.1368 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0937 | (0.0436, 0.1483) | 0.0000 | 0.0937 | (0.0222, 0.1597) | 0.0117 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0293 | (-0.0257, 0.0871) | 0.1443 | 0.0293 | (-0.0157, 0.0649) | 0.0910 |
| proposed_vs_candidate_no_context | naturalness | 0.0349 | (0.0135, 0.0589) | 0.0003 | 0.0349 | (0.0103, 0.0574) | 0.0100 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.1227 | (0.0591, 0.1955) | 0.0000 | 0.1227 | (0.0303, 0.2153) | 0.0077 |
| proposed_vs_candidate_no_context | context_overlap | 0.0259 | (0.0051, 0.0550) | 0.0020 | 0.0259 | (0.0032, 0.0513) | 0.0117 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0369 | (-0.0274, 0.1107) | 0.1577 | 0.0369 | (-0.0196, 0.0819) | 0.1213 |
| proposed_vs_candidate_no_context | persona_style | -0.0013 | (-0.0038, 0.0000) | 1.0000 | -0.0013 | (-0.0030, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | distinct1 | 0.0126 | (-0.0045, 0.0318) | 0.0743 | 0.0126 | (0.0020, 0.0301) | 0.0083 |
| proposed_vs_candidate_no_context | length_score | 0.0967 | (0.0233, 0.1767) | 0.0020 | 0.0967 | (0.0250, 0.1526) | 0.0070 |
| proposed_vs_candidate_no_context | sentence_score | 0.1050 | (0.0350, 0.1750) | 0.0007 | 0.1050 | (0.0389, 0.1591) | 0.0080 |
| proposed_vs_candidate_no_context | overall_quality | 0.0615 | (0.0249, 0.1021) | 0.0000 | 0.0615 | (0.0177, 0.1009) | 0.0080 |
| controlled_vs_proposed_raw | context_relevance | 0.1838 | (0.1390, 0.2301) | 0.0000 | 0.1838 | (0.1532, 0.2146) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.1306 | (0.0541, 0.2067) | 0.0007 | 0.1306 | (0.0787, 0.2050) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0609 | (0.0160, 0.1065) | 0.0053 | 0.0609 | (0.0093, 0.1259) | 0.0080 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2409 | (0.1818, 0.2955) | 0.0000 | 0.2409 | (0.2057, 0.2778) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0504 | (0.0227, 0.0732) | 0.0007 | 0.0504 | (0.0290, 0.0690) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1562 | (0.0636, 0.2469) | 0.0007 | 0.1562 | (0.0996, 0.2458) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.0283 | (-0.0112, 0.0833) | 0.1110 | 0.0283 | (-0.0115, 0.1176) | 0.3293 |
| controlled_vs_proposed_raw | distinct1 | -0.0002 | (-0.0204, 0.0192) | 0.4920 | -0.0002 | (-0.0299, 0.0247) | 0.5283 |
| controlled_vs_proposed_raw | length_score | 0.2700 | (0.0883, 0.4350) | 0.0010 | 0.2700 | (0.1313, 0.4875) | 0.0000 |
| controlled_vs_proposed_raw | sentence_score | 0.0700 | (-0.0700, 0.1925) | 0.1823 | 0.0700 | (-0.0636, 0.2625) | 0.1927 |
| controlled_vs_proposed_raw | overall_quality | 0.1462 | (0.0980, 0.1915) | 0.0000 | 0.1462 | (0.1059, 0.1955) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2775 | (0.2453, 0.3127) | 0.0000 | 0.2775 | (0.2367, 0.3158) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1599 | (0.0941, 0.2226) | 0.0000 | 0.1599 | (0.0912, 0.2217) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.0958 | (0.0481, 0.1368) | 0.0000 | 0.0958 | (0.0545, 0.1462) | 0.0000 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3636 | (0.3227, 0.4136) | 0.0000 | 0.3636 | (0.3081, 0.4174) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0764 | (0.0644, 0.0885) | 0.0000 | 0.0764 | (0.0671, 0.0827) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1931 | (0.1098, 0.2731) | 0.0000 | 0.1931 | (0.1140, 0.2682) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.0271 | (-0.0125, 0.0771) | 0.1400 | 0.0271 | (-0.0150, 0.1176) | 0.3403 |
| controlled_vs_candidate_no_context | distinct1 | 0.0124 | (-0.0021, 0.0252) | 0.0460 | 0.0124 | (-0.0030, 0.0267) | 0.0613 |
| controlled_vs_candidate_no_context | length_score | 0.3667 | (0.1867, 0.5367) | 0.0003 | 0.3667 | (0.2350, 0.5426) | 0.0000 |
| controlled_vs_candidate_no_context | sentence_score | 0.1750 | (0.0700, 0.2629) | 0.0013 | 0.1750 | (0.0808, 0.3062) | 0.0007 |
| controlled_vs_candidate_no_context | overall_quality | 0.2077 | (0.1713, 0.2364) | 0.0000 | 0.2077 | (0.1837, 0.2249) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0251 | (-0.0182, 0.0709) | 0.1327 | 0.0251 | (0.0045, 0.0613) | 0.0050 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0562 | (-0.0184, 0.1358) | 0.0730 | 0.0562 | (-0.0376, 0.1262) | 0.1113 |
| controlled_alt_vs_controlled_default | naturalness | -0.0119 | (-0.0475, 0.0256) | 0.7400 | -0.0119 | (-0.0295, 0.0072) | 0.8937 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0364 | (-0.0227, 0.0955) | 0.1073 | 0.0364 | (0.0096, 0.0859) | 0.0000 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0011 | (-0.0191, 0.0175) | 0.5623 | -0.0011 | (-0.0109, 0.0124) | 0.6123 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0686 | (-0.0274, 0.1702) | 0.0797 | 0.0686 | (-0.0471, 0.1614) | 0.1153 |
| controlled_alt_vs_controlled_default | persona_style | 0.0067 | (-0.0317, 0.0415) | 0.3613 | 0.0067 | (-0.0231, 0.0588) | 0.4287 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0063 | (-0.0270, 0.0110) | 0.7327 | -0.0063 | (-0.0133, 0.0019) | 0.9210 |
| controlled_alt_vs_controlled_default | length_score | -0.0817 | (-0.2500, 0.0950) | 0.8193 | -0.0817 | (-0.1364, -0.0148) | 0.9943 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0700 | (-0.0350, 0.1750) | 0.1243 | 0.0700 | (-0.0219, 0.1333) | 0.0907 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0306 | (-0.0022, 0.0618) | 0.0357 | 0.0306 | (-0.0060, 0.0617) | 0.0450 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.2089 | (0.1523, 0.2608) | 0.0000 | 0.2089 | (0.1673, 0.2606) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1868 | (0.1152, 0.2663) | 0.0000 | 0.1868 | (0.1267, 0.2549) | 0.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0491 | (-0.0029, 0.0960) | 0.0330 | 0.0491 | (0.0072, 0.1029) | 0.0090 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2773 | (0.2045, 0.3455) | 0.0000 | 0.2773 | (0.2231, 0.3515) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0493 | (0.0196, 0.0723) | 0.0010 | 0.0493 | (0.0232, 0.0641) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.2248 | (0.1405, 0.3222) | 0.0000 | 0.2248 | (0.1614, 0.2881) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0350 | (-0.0213, 0.1048) | 0.1330 | 0.0350 | (-0.0346, 0.1765) | 0.3377 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0065 | (-0.0333, 0.0173) | 0.6887 | -0.0065 | (-0.0310, 0.0195) | 0.7067 |
| controlled_alt_vs_proposed_raw | length_score | 0.1883 | (-0.0034, 0.3717) | 0.0260 | 0.1883 | (0.0576, 0.3958) | 0.0000 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1400 | (0.0350, 0.2275) | 0.0070 | 0.1400 | (0.0553, 0.2528) | 0.0003 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1767 | (0.1282, 0.2231) | 0.0000 | 0.1767 | (0.1389, 0.2359) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.3026 | (0.2724, 0.3357) | 0.0000 | 0.3026 | (0.2750, 0.3368) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.2161 | (0.1348, 0.3008) | 0.0000 | 0.2161 | (0.1541, 0.2697) | 0.0000 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0839 | (0.0360, 0.1279) | 0.0003 | 0.0839 | (0.0584, 0.1207) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.4000 | (0.3591, 0.4455) | 0.0000 | 0.4000 | (0.3589, 0.4444) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0753 | (0.0614, 0.0903) | 0.0000 | 0.0753 | (0.0617, 0.0880) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.2617 | (0.1567, 0.3593) | 0.0000 | 0.2617 | (0.1896, 0.3182) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0337 | (-0.0246, 0.1040) | 0.1400 | 0.0337 | (-0.0375, 0.1765) | 0.3410 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0061 | (-0.0175, 0.0266) | 0.2817 | 0.0061 | (-0.0079, 0.0247) | 0.2157 |
| controlled_alt_vs_candidate_no_context | length_score | 0.2850 | (0.0983, 0.4567) | 0.0017 | 0.2850 | (0.1885, 0.4208) | 0.0000 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.2450 | (0.1575, 0.3150) | 0.0000 | 0.2450 | (0.2042, 0.2917) | 0.0000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.2382 | (0.2058, 0.2696) | 0.0000 | 0.2382 | (0.2113, 0.2638) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 10 | 1 | 9 | 0.7250 | 0.9091 |
| proposed_vs_candidate_no_context | persona_consistency | 4 | 2 | 14 | 0.5500 | 0.6667 |
| proposed_vs_candidate_no_context | naturalness | 9 | 2 | 9 | 0.6750 | 0.8182 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 10 | 0 | 10 | 0.7500 | 1.0000 |
| proposed_vs_candidate_no_context | context_overlap | 8 | 3 | 9 | 0.6250 | 0.7273 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 4 | 2 | 14 | 0.5500 | 0.6667 |
| proposed_vs_candidate_no_context | persona_style | 0 | 1 | 19 | 0.4750 | 0.0000 |
| proposed_vs_candidate_no_context | distinct1 | 7 | 3 | 10 | 0.6000 | 0.7000 |
| proposed_vs_candidate_no_context | length_score | 8 | 3 | 9 | 0.6250 | 0.7273 |
| proposed_vs_candidate_no_context | sentence_score | 6 | 0 | 14 | 0.6500 | 1.0000 |
| proposed_vs_candidate_no_context | overall_quality | 10 | 1 | 9 | 0.7250 | 0.9091 |
| controlled_vs_proposed_raw | context_relevance | 18 | 1 | 1 | 0.9250 | 0.9474 |
| controlled_vs_proposed_raw | persona_consistency | 14 | 3 | 3 | 0.7750 | 0.8235 |
| controlled_vs_proposed_raw | naturalness | 15 | 5 | 0 | 0.7500 | 0.7500 |
| controlled_vs_proposed_raw | context_keyword_coverage | 18 | 0 | 2 | 0.9500 | 1.0000 |
| controlled_vs_proposed_raw | context_overlap | 16 | 3 | 1 | 0.8250 | 0.8421 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 14 | 3 | 3 | 0.7750 | 0.8235 |
| controlled_vs_proposed_raw | persona_style | 3 | 1 | 16 | 0.5500 | 0.7500 |
| controlled_vs_proposed_raw | distinct1 | 11 | 8 | 1 | 0.5750 | 0.5789 |
| controlled_vs_proposed_raw | length_score | 15 | 4 | 1 | 0.7750 | 0.7895 |
| controlled_vs_proposed_raw | sentence_score | 10 | 6 | 4 | 0.6000 | 0.6250 |
| controlled_vs_proposed_raw | overall_quality | 17 | 3 | 0 | 0.8500 | 0.8500 |
| controlled_vs_candidate_no_context | context_relevance | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 18 | 1 | 1 | 0.9250 | 0.9474 |
| controlled_vs_candidate_no_context | naturalness | 18 | 2 | 0 | 0.9000 | 0.9000 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 18 | 1 | 1 | 0.9250 | 0.9474 |
| controlled_vs_candidate_no_context | persona_style | 2 | 1 | 17 | 0.5250 | 0.6667 |
| controlled_vs_candidate_no_context | distinct1 | 13 | 5 | 2 | 0.7000 | 0.7222 |
| controlled_vs_candidate_no_context | length_score | 15 | 4 | 1 | 0.7750 | 0.7895 |
| controlled_vs_candidate_no_context | sentence_score | 12 | 2 | 6 | 0.7500 | 0.8571 |
| controlled_vs_candidate_no_context | overall_quality | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_alt_vs_controlled_default | context_relevance | 9 | 7 | 4 | 0.5500 | 0.5625 |
| controlled_alt_vs_controlled_default | persona_consistency | 8 | 3 | 9 | 0.6250 | 0.7273 |
| controlled_alt_vs_controlled_default | naturalness | 5 | 11 | 4 | 0.3500 | 0.3125 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 6 | 3 | 11 | 0.5750 | 0.6667 |
| controlled_alt_vs_controlled_default | context_overlap | 7 | 9 | 4 | 0.4500 | 0.4375 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 7 | 3 | 10 | 0.6000 | 0.7000 |
| controlled_alt_vs_controlled_default | persona_style | 3 | 2 | 15 | 0.5250 | 0.6000 |
| controlled_alt_vs_controlled_default | distinct1 | 9 | 6 | 5 | 0.5750 | 0.6000 |
| controlled_alt_vs_controlled_default | length_score | 5 | 9 | 6 | 0.4000 | 0.3571 |
| controlled_alt_vs_controlled_default | sentence_score | 7 | 3 | 10 | 0.6000 | 0.7000 |
| controlled_alt_vs_controlled_default | overall_quality | 10 | 6 | 4 | 0.6000 | 0.6250 |
| controlled_alt_vs_proposed_raw | context_relevance | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_alt_vs_proposed_raw | persona_consistency | 16 | 2 | 2 | 0.8500 | 0.8889 |
| controlled_alt_vs_proposed_raw | naturalness | 13 | 7 | 0 | 0.6500 | 0.6500 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_alt_vs_proposed_raw | context_overlap | 17 | 3 | 0 | 0.8500 | 0.8500 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 16 | 1 | 3 | 0.8750 | 0.9412 |
| controlled_alt_vs_proposed_raw | persona_style | 3 | 2 | 15 | 0.5250 | 0.6000 |
| controlled_alt_vs_proposed_raw | distinct1 | 9 | 10 | 1 | 0.4750 | 0.4737 |
| controlled_alt_vs_proposed_raw | length_score | 12 | 5 | 3 | 0.6750 | 0.7059 |
| controlled_alt_vs_proposed_raw | sentence_score | 10 | 2 | 8 | 0.7000 | 0.8333 |
| controlled_alt_vs_proposed_raw | overall_quality | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_alt_vs_candidate_no_context | context_relevance | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 17 | 1 | 2 | 0.9000 | 0.9444 |
| controlled_alt_vs_candidate_no_context | naturalness | 18 | 2 | 0 | 0.9000 | 0.9000 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 17 | 1 | 2 | 0.9000 | 0.9444 |
| controlled_alt_vs_candidate_no_context | persona_style | 3 | 2 | 15 | 0.5250 | 0.6000 |
| controlled_alt_vs_candidate_no_context | distinct1 | 10 | 9 | 1 | 0.5250 | 0.5263 |
| controlled_alt_vs_candidate_no_context | length_score | 14 | 4 | 2 | 0.7500 | 0.7778 |
| controlled_alt_vs_candidate_no_context | sentence_score | 15 | 1 | 4 | 0.8500 | 0.9375 |
| controlled_alt_vs_candidate_no_context | overall_quality | 20 | 0 | 0 | 1.0000 | 1.0000 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.2500 | 0.3500 | 0.6500 |
| proposed_contextual_controlled_tuned | 0.0000 | 0.0000 | 0.2500 | 0.3500 | 0.6500 |
| proposed_contextual | 0.0000 | 0.0000 | 0.4500 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.7000 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `5`
- Unique template signatures: `17`
- Template signature ratio: `0.8500`
- Effective sample size by source clustering: `4.65`
- Effective sample size by template-signature clustering: `15.38`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: unavailable (disabled_by_flag).

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
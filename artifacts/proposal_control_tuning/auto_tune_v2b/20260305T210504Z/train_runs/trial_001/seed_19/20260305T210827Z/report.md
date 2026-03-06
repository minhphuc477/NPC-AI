# Proposal Alignment Evaluation Report

- Run ID: `20260305T210827Z`
- Generated: `2026-03-05T21:12:18.626301+00:00`
- Scenarios: `artifacts\proposal_control_tuning\auto_tune_v2b\20260305T210504Z\train_runs\trial_001\seed_19\20260305T210827Z\scenarios.jsonl`
- Scenario count: `20`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_tuned`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2756 (0.2399, 0.3199) | 0.3918 (0.3020, 0.4868) | 0.8752 (0.8466, 0.9038) | 0.4395 (0.4075, 0.4757) | n/a |
| proposed_contextual_controlled_tuned | 0.2581 (0.2340, 0.2848) | 0.3363 (0.2637, 0.4126) | 0.8951 (0.8663, 0.9183) | 0.4148 (0.3876, 0.4420) | n/a |
| proposed_contextual | 0.0827 (0.0470, 0.1234) | 0.1589 (0.1170, 0.2026) | 0.8257 (0.7952, 0.8564) | 0.2532 (0.2222, 0.2845) | n/a |
| candidate_no_context | 0.0205 (0.0116, 0.0319) | 0.1274 (0.0907, 0.1621) | 0.7831 (0.7594, 0.8109) | 0.2042 (0.1871, 0.2225) | n/a |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0622 | 3.0308 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0315 | 0.2477 |
| proposed_vs_candidate_no_context | naturalness | 0.0426 | 0.0544 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0818 | 6.0000 |
| proposed_vs_candidate_no_context | context_overlap | 0.0164 | 0.4487 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0338 | 2.1846 |
| proposed_vs_candidate_no_context | persona_style | 0.0225 | 0.0391 |
| proposed_vs_candidate_no_context | distinct1 | 0.0187 | 0.0201 |
| proposed_vs_candidate_no_context | length_score | 0.1317 | 0.6124 |
| proposed_vs_candidate_no_context | sentence_score | 0.0875 | 0.1311 |
| proposed_vs_candidate_no_context | overall_quality | 0.0489 | 0.2396 |
| controlled_vs_proposed_raw | context_relevance | 0.1929 | 2.3320 |
| controlled_vs_proposed_raw | persona_consistency | 0.2328 | 1.4649 |
| controlled_vs_proposed_raw | naturalness | 0.0495 | 0.0599 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2500 | 2.6190 |
| controlled_vs_proposed_raw | context_overlap | 0.0597 | 1.1259 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2857 | 5.7971 |
| controlled_vs_proposed_raw | persona_style | 0.0213 | 0.0356 |
| controlled_vs_proposed_raw | distinct1 | -0.0101 | -0.0106 |
| controlled_vs_proposed_raw | length_score | 0.2083 | 0.6010 |
| controlled_vs_proposed_raw | sentence_score | 0.1225 | 0.1623 |
| controlled_vs_proposed_raw | overall_quality | 0.1863 | 0.7359 |
| controlled_vs_candidate_no_context | context_relevance | 0.2551 | 12.4307 |
| controlled_vs_candidate_no_context | persona_consistency | 0.2644 | 2.0754 |
| controlled_vs_candidate_no_context | naturalness | 0.0921 | 0.1176 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3318 | 24.3333 |
| controlled_vs_candidate_no_context | context_overlap | 0.0761 | 2.0797 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.3195 | 20.6462 |
| controlled_vs_candidate_no_context | persona_style | 0.0438 | 0.0761 |
| controlled_vs_candidate_no_context | distinct1 | 0.0086 | 0.0092 |
| controlled_vs_candidate_no_context | length_score | 0.3400 | 1.5814 |
| controlled_vs_candidate_no_context | sentence_score | 0.2100 | 0.3146 |
| controlled_vs_candidate_no_context | overall_quality | 0.2353 | 1.1519 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0175 | -0.0635 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0555 | -0.1416 |
| controlled_alt_vs_controlled_default | naturalness | 0.0199 | 0.0228 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0182 | -0.0526 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0159 | -0.1411 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0714 | -0.2132 |
| controlled_alt_vs_controlled_default | persona_style | 0.0083 | 0.0135 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0117 | -0.0124 |
| controlled_alt_vs_controlled_default | length_score | 0.0983 | 0.1772 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0525 | 0.0598 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0247 | -0.0562 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1754 | 2.1205 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1773 | 1.1159 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0694 | 0.0841 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2318 | 2.4286 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0438 | 0.8259 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.2143 | 4.3478 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0296 | 0.0495 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0218 | -0.0229 |
| controlled_alt_vs_proposed_raw | length_score | 0.3067 | 0.8846 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1750 | 0.2318 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1616 | 0.6384 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2376 | 11.5781 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.2089 | 1.6399 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.1120 | 0.1430 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3136 | 23.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0602 | 1.6452 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.2481 | 16.0308 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0521 | 0.0906 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0031 | -0.0033 |
| controlled_alt_vs_candidate_no_context | length_score | 0.4383 | 2.0388 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.2625 | 0.3933 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.2106 | 1.0310 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0622 | (0.0230, 0.1056) | 0.0003 | 0.0622 | (0.0276, 0.1053) | 0.0003 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0315 | (-0.0004, 0.0699) | 0.0280 | 0.0315 | (0.0130, 0.0572) | 0.0000 |
| proposed_vs_candidate_no_context | naturalness | 0.0426 | (0.0084, 0.0774) | 0.0060 | 0.0426 | (0.0230, 0.0575) | 0.0003 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0818 | (0.0273, 0.1455) | 0.0017 | 0.0818 | (0.0346, 0.1388) | 0.0000 |
| proposed_vs_candidate_no_context | context_overlap | 0.0164 | (0.0053, 0.0295) | 0.0017 | 0.0164 | (0.0070, 0.0284) | 0.0000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0338 | (0.0000, 0.0721) | 0.0257 | 0.0338 | (0.0190, 0.0500) | 0.0000 |
| proposed_vs_candidate_no_context | persona_style | 0.0225 | (-0.0062, 0.0750) | 0.3613 | 0.0225 | (-0.0058, 0.0882) | 0.3167 |
| proposed_vs_candidate_no_context | distinct1 | 0.0187 | (0.0039, 0.0351) | 0.0083 | 0.0187 | (0.0070, 0.0322) | 0.0000 |
| proposed_vs_candidate_no_context | length_score | 0.1317 | (-0.0083, 0.2734) | 0.0337 | 0.1317 | (0.0741, 0.1848) | 0.0000 |
| proposed_vs_candidate_no_context | sentence_score | 0.0875 | (0.0350, 0.1575) | 0.0037 | 0.0875 | (0.0437, 0.1114) | 0.0000 |
| proposed_vs_candidate_no_context | overall_quality | 0.0489 | (0.0205, 0.0809) | 0.0000 | 0.0489 | (0.0233, 0.0739) | 0.0000 |
| controlled_vs_proposed_raw | context_relevance | 0.1929 | (0.1492, 0.2368) | 0.0000 | 0.1929 | (0.1468, 0.2508) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.2328 | (0.1384, 0.3309) | 0.0000 | 0.2328 | (0.1147, 0.3397) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0495 | (0.0030, 0.0941) | 0.0170 | 0.0495 | (-0.0014, 0.0838) | 0.0270 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2500 | (0.1909, 0.3091) | 0.0000 | 0.2500 | (0.1901, 0.3212) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0597 | (0.0415, 0.0801) | 0.0000 | 0.0597 | (0.0424, 0.0808) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2857 | (0.1779, 0.4126) | 0.0000 | 0.2857 | (0.1368, 0.4204) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.0212 | (-0.0242, 0.0777) | 0.1960 | 0.0212 | (-0.0087, 0.0882) | 0.3293 |
| controlled_vs_proposed_raw | distinct1 | -0.0101 | (-0.0366, 0.0145) | 0.7687 | -0.0101 | (-0.0456, 0.0210) | 0.6957 |
| controlled_vs_proposed_raw | length_score | 0.2083 | (0.0483, 0.3717) | 0.0027 | 0.2083 | (0.0968, 0.2930) | 0.0000 |
| controlled_vs_proposed_raw | sentence_score | 0.1225 | (0.0175, 0.2275) | 0.0160 | 0.1225 | (0.0175, 0.2139) | 0.0167 |
| controlled_vs_proposed_raw | overall_quality | 0.1863 | (0.1408, 0.2336) | 0.0000 | 0.1863 | (0.1199, 0.2464) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2551 | (0.2161, 0.3028) | 0.0000 | 0.2551 | (0.2297, 0.2980) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.2644 | (0.1868, 0.3557) | 0.0000 | 0.2644 | (0.1565, 0.3620) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.0921 | (0.0573, 0.1261) | 0.0000 | 0.0921 | (0.0550, 0.1136) | 0.0000 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3318 | (0.2818, 0.3909) | 0.0000 | 0.3318 | (0.3000, 0.3864) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0761 | (0.0593, 0.0972) | 0.0000 | 0.0761 | (0.0594, 0.0959) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.3195 | (0.2240, 0.4353) | 0.0000 | 0.3195 | (0.1825, 0.4435) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.0437 | (-0.0063, 0.1000) | 0.0550 | 0.0437 | (-0.0150, 0.1765) | 0.3403 |
| controlled_vs_candidate_no_context | distinct1 | 0.0086 | (-0.0140, 0.0312) | 0.2247 | 0.0086 | (-0.0236, 0.0348) | 0.2670 |
| controlled_vs_candidate_no_context | length_score | 0.3400 | (0.2100, 0.4733) | 0.0000 | 0.3400 | (0.2621, 0.4013) | 0.0000 |
| controlled_vs_candidate_no_context | sentence_score | 0.2100 | (0.1225, 0.2975) | 0.0000 | 0.2100 | (0.0921, 0.3000) | 0.0007 |
| controlled_vs_candidate_no_context | overall_quality | 0.2353 | (0.1957, 0.2779) | 0.0000 | 0.2353 | (0.1888, 0.2754) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0175 | (-0.0706, 0.0236) | 0.7347 | -0.0175 | (-0.0689, 0.0159) | 0.8037 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0555 | (-0.1555, 0.0266) | 0.8933 | -0.0555 | (-0.1241, -0.0067) | 1.0000 |
| controlled_alt_vs_controlled_default | naturalness | 0.0199 | (-0.0158, 0.0528) | 0.1300 | 0.0199 | (0.0063, 0.0330) | 0.0030 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0182 | (-0.0865, 0.0318) | 0.7077 | -0.0182 | (-0.0808, 0.0227) | 0.7277 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0159 | (-0.0381, 0.0022) | 0.9553 | -0.0159 | (-0.0395, -0.0002) | 0.9797 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0714 | (-0.1955, 0.0257) | 0.9163 | -0.0714 | (-0.1553, -0.0105) | 1.0000 |
| controlled_alt_vs_controlled_default | persona_style | 0.0083 | (-0.0250, 0.0500) | 0.4713 | 0.0083 | (0.0000, 0.0294) | 0.3280 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0117 | (-0.0335, 0.0127) | 0.8357 | -0.0117 | (-0.0320, 0.0071) | 0.8833 |
| controlled_alt_vs_controlled_default | length_score | 0.0983 | (-0.0284, 0.2283) | 0.0630 | 0.0983 | (0.0159, 0.2175) | 0.0130 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0525 | (-0.0350, 0.1400) | 0.1633 | 0.0525 | (-0.0389, 0.1474) | 0.1837 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0247 | (-0.0658, 0.0089) | 0.9077 | -0.0247 | (-0.0645, 0.0053) | 0.9410 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1754 | (0.1271, 0.2204) | 0.0000 | 0.1754 | (0.1377, 0.2016) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1773 | (0.1076, 0.2499) | 0.0000 | 0.1773 | (0.1081, 0.2528) | 0.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0694 | (0.0350, 0.1031) | 0.0000 | 0.0694 | (0.0193, 0.1115) | 0.0023 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2318 | (0.1682, 0.2909) | 0.0000 | 0.2318 | (0.1869, 0.2618) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0438 | (0.0287, 0.0587) | 0.0000 | 0.0438 | (0.0232, 0.0608) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.2143 | (0.1331, 0.3052) | 0.0000 | 0.2143 | (0.1263, 0.3183) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0296 | (-0.0252, 0.1058) | 0.2343 | 0.0296 | (-0.0087, 0.1176) | 0.3340 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0218 | (-0.0422, -0.0029) | 0.9923 | -0.0218 | (-0.0427, -0.0077) | 1.0000 |
| controlled_alt_vs_proposed_raw | length_score | 0.3067 | (0.1733, 0.4400) | 0.0000 | 0.3067 | (0.1127, 0.4963) | 0.0000 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1750 | (0.1050, 0.2450) | 0.0000 | 0.1750 | (0.1167, 0.2333) | 0.0000 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1616 | (0.1236, 0.1975) | 0.0000 | 0.1616 | (0.1186, 0.2033) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2376 | (0.2134, 0.2621) | 0.0000 | 0.2376 | (0.2126, 0.2517) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.2089 | (0.1522, 0.2692) | 0.0000 | 0.2089 | (0.1375, 0.2800) | 0.0000 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.1120 | (0.0772, 0.1432) | 0.0000 | 0.1120 | (0.0743, 0.1404) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3136 | (0.2818, 0.3455) | 0.0000 | 0.3136 | (0.2784, 0.3349) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0602 | (0.0497, 0.0711) | 0.0000 | 0.0602 | (0.0449, 0.0738) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.2481 | (0.1762, 0.3257) | 0.0000 | 0.2481 | (0.1719, 0.3421) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0521 | (-0.0063, 0.1333) | 0.0603 | 0.0521 | (-0.0144, 0.2059) | 0.3393 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0031 | (-0.0191, 0.0130) | 0.6390 | -0.0031 | (-0.0173, 0.0092) | 0.6430 |
| controlled_alt_vs_candidate_no_context | length_score | 0.4383 | (0.2916, 0.5717) | 0.0000 | 0.4383 | (0.2944, 0.5667) | 0.0000 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.2625 | (0.1925, 0.3325) | 0.0000 | 0.2625 | (0.2059, 0.3150) | 0.0000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.2106 | (0.1876, 0.2333) | 0.0000 | 0.2106 | (0.1808, 0.2404) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 10 | 2 | 8 | 0.7000 | 0.8333 |
| proposed_vs_candidate_no_context | persona_consistency | 5 | 1 | 14 | 0.6000 | 0.8333 |
| proposed_vs_candidate_no_context | naturalness | 9 | 3 | 8 | 0.6500 | 0.7500 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 9 | 2 | 9 | 0.6750 | 0.8182 |
| proposed_vs_candidate_no_context | context_overlap | 11 | 1 | 8 | 0.7500 | 0.9167 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 5 | 1 | 14 | 0.6000 | 0.8333 |
| proposed_vs_candidate_no_context | persona_style | 1 | 2 | 17 | 0.4750 | 0.3333 |
| proposed_vs_candidate_no_context | distinct1 | 7 | 2 | 11 | 0.6250 | 0.7778 |
| proposed_vs_candidate_no_context | length_score | 9 | 3 | 8 | 0.6500 | 0.7500 |
| proposed_vs_candidate_no_context | sentence_score | 5 | 0 | 15 | 0.6250 | 1.0000 |
| proposed_vs_candidate_no_context | overall_quality | 10 | 2 | 8 | 0.7000 | 0.8333 |
| controlled_vs_proposed_raw | context_relevance | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_vs_proposed_raw | persona_consistency | 16 | 2 | 2 | 0.8500 | 0.8889 |
| controlled_vs_proposed_raw | naturalness | 13 | 7 | 0 | 0.6500 | 0.6500 |
| controlled_vs_proposed_raw | context_keyword_coverage | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_vs_proposed_raw | context_overlap | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 16 | 1 | 3 | 0.8750 | 0.9412 |
| controlled_vs_proposed_raw | persona_style | 4 | 2 | 14 | 0.5500 | 0.6667 |
| controlled_vs_proposed_raw | distinct1 | 10 | 8 | 2 | 0.5500 | 0.5556 |
| controlled_vs_proposed_raw | length_score | 14 | 4 | 2 | 0.7500 | 0.7778 |
| controlled_vs_proposed_raw | sentence_score | 9 | 2 | 9 | 0.6750 | 0.8182 |
| controlled_vs_proposed_raw | overall_quality | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_vs_candidate_no_context | context_relevance | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 19 | 0 | 1 | 0.9750 | 1.0000 |
| controlled_vs_candidate_no_context | naturalness | 18 | 2 | 0 | 0.9000 | 0.9000 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 19 | 0 | 1 | 0.9750 | 1.0000 |
| controlled_vs_candidate_no_context | persona_style | 3 | 1 | 16 | 0.5500 | 0.7500 |
| controlled_vs_candidate_no_context | distinct1 | 13 | 6 | 1 | 0.6750 | 0.6842 |
| controlled_vs_candidate_no_context | length_score | 15 | 4 | 1 | 0.7750 | 0.7895 |
| controlled_vs_candidate_no_context | sentence_score | 13 | 1 | 6 | 0.8000 | 0.9286 |
| controlled_vs_candidate_no_context | overall_quality | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 9 | 6 | 5 | 0.5750 | 0.6000 |
| controlled_alt_vs_controlled_default | persona_consistency | 3 | 6 | 11 | 0.4250 | 0.3333 |
| controlled_alt_vs_controlled_default | naturalness | 11 | 4 | 5 | 0.6750 | 0.7333 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 5 | 4 | 11 | 0.5250 | 0.5556 |
| controlled_alt_vs_controlled_default | context_overlap | 6 | 9 | 5 | 0.4250 | 0.4000 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 2 | 6 | 12 | 0.4000 | 0.2500 |
| controlled_alt_vs_controlled_default | persona_style | 1 | 1 | 18 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | distinct1 | 5 | 10 | 5 | 0.3750 | 0.3333 |
| controlled_alt_vs_controlled_default | length_score | 10 | 4 | 6 | 0.6500 | 0.7143 |
| controlled_alt_vs_controlled_default | sentence_score | 5 | 2 | 13 | 0.5750 | 0.7143 |
| controlled_alt_vs_controlled_default | overall_quality | 6 | 9 | 5 | 0.4250 | 0.4000 |
| controlled_alt_vs_proposed_raw | context_relevance | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_alt_vs_proposed_raw | persona_consistency | 14 | 1 | 5 | 0.8250 | 0.9333 |
| controlled_alt_vs_proposed_raw | naturalness | 15 | 5 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 18 | 1 | 1 | 0.9250 | 0.9474 |
| controlled_alt_vs_proposed_raw | context_overlap | 18 | 2 | 0 | 0.9000 | 0.9000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 14 | 0 | 6 | 0.8500 | 1.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 4 | 2 | 14 | 0.5500 | 0.6667 |
| controlled_alt_vs_proposed_raw | distinct1 | 7 | 13 | 0 | 0.3500 | 0.3500 |
| controlled_alt_vs_proposed_raw | length_score | 16 | 3 | 1 | 0.8250 | 0.8421 |
| controlled_alt_vs_proposed_raw | sentence_score | 10 | 0 | 10 | 0.7500 | 1.0000 |
| controlled_alt_vs_proposed_raw | overall_quality | 18 | 2 | 0 | 0.9000 | 0.9000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 18 | 0 | 2 | 0.9500 | 1.0000 |
| controlled_alt_vs_candidate_no_context | naturalness | 18 | 2 | 0 | 0.9000 | 0.9000 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 18 | 0 | 2 | 0.9500 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_style | 3 | 1 | 16 | 0.5500 | 0.7500 |
| controlled_alt_vs_candidate_no_context | distinct1 | 10 | 10 | 0 | 0.5000 | 0.5000 |
| controlled_alt_vs_candidate_no_context | length_score | 17 | 2 | 1 | 0.8750 | 0.8947 |
| controlled_alt_vs_candidate_no_context | sentence_score | 15 | 0 | 5 | 0.8750 | 1.0000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 20 | 0 | 0 | 1.0000 | 1.0000 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.3500 | 0.3000 | 0.7000 |
| proposed_contextual_controlled_tuned | 0.0000 | 0.0000 | 0.5500 | 0.2500 | 0.7500 |
| proposed_contextual | 0.0000 | 0.0000 | 0.4000 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.6500 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `5`
- Unique template signatures: `17`
- Template signature ratio: `0.8500`
- Effective sample size by source clustering: `4.65`
- Effective sample size by template-signature clustering: `15.38`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: unavailable (disabled_by_flag).

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
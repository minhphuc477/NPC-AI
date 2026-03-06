# Proposal Alignment Evaluation Report

- Run ID: `20260305T204627Z`
- Generated: `2026-03-05T20:50:17.814070+00:00`
- Scenarios: `artifacts\proposal_control_tuning\auto_tune\20260305T203842Z\train_runs\trial_003\seed_19\20260305T204627Z\scenarios.jsonl`
- Scenario count: `20`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_tuned`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2599 (0.2309, 0.2902) | 0.3651 (0.2908, 0.4346) | 0.9023 (0.8758, 0.9256) | 0.4277 (0.3983, 0.4550) | n/a |
| proposed_contextual_controlled_tuned | 0.2819 (0.2427, 0.3265) | 0.3317 (0.2633, 0.4140) | 0.8950 (0.8668, 0.9207) | 0.4247 (0.3943, 0.4563) | n/a |
| proposed_contextual | 0.0813 (0.0426, 0.1301) | 0.1480 (0.1065, 0.1945) | 0.8190 (0.7889, 0.8492) | 0.2474 (0.2145, 0.2836) | n/a |
| candidate_no_context | 0.0270 (0.0137, 0.0439) | 0.1464 (0.1000, 0.1945) | 0.8020 (0.7708, 0.8388) | 0.2182 (0.1924, 0.2475) | n/a |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0543 | 2.0130 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0015 | 0.0105 |
| proposed_vs_candidate_no_context | naturalness | 0.0170 | 0.0211 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0727 | 3.2000 |
| proposed_vs_candidate_no_context | context_overlap | 0.0113 | 0.3059 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | persona_style | 0.0077 | 0.0134 |
| proposed_vs_candidate_no_context | distinct1 | 0.0074 | 0.0079 |
| proposed_vs_candidate_no_context | length_score | 0.0350 | 0.1154 |
| proposed_vs_candidate_no_context | sentence_score | 0.0700 | 0.1022 |
| proposed_vs_candidate_no_context | overall_quality | 0.0292 | 0.1339 |
| controlled_vs_proposed_raw | context_relevance | 0.1787 | 2.1987 |
| controlled_vs_proposed_raw | persona_consistency | 0.2172 | 1.4677 |
| controlled_vs_proposed_raw | naturalness | 0.0834 | 0.1018 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2318 | 2.4286 |
| controlled_vs_proposed_raw | context_overlap | 0.0547 | 1.1355 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2588 | 6.5879 |
| controlled_vs_proposed_raw | persona_style | 0.0506 | 0.0869 |
| controlled_vs_proposed_raw | distinct1 | -0.0115 | -0.0122 |
| controlled_vs_proposed_raw | length_score | 0.3450 | 1.0197 |
| controlled_vs_proposed_raw | sentence_score | 0.1925 | 0.2550 |
| controlled_vs_proposed_raw | overall_quality | 0.1802 | 0.7284 |
| controlled_vs_candidate_no_context | context_relevance | 0.2330 | 8.6379 |
| controlled_vs_candidate_no_context | persona_consistency | 0.2187 | 1.4937 |
| controlled_vs_candidate_no_context | naturalness | 0.1003 | 0.1251 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3045 | 13.4000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0659 | 1.7886 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2588 | 6.5879 |
| controlled_vs_candidate_no_context | persona_style | 0.0583 | 0.1014 |
| controlled_vs_candidate_no_context | distinct1 | -0.0041 | -0.0044 |
| controlled_vs_candidate_no_context | length_score | 0.3800 | 1.2527 |
| controlled_vs_candidate_no_context | sentence_score | 0.2625 | 0.3832 |
| controlled_vs_candidate_no_context | overall_quality | 0.2094 | 0.9598 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0220 | 0.0845 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0335 | -0.0917 |
| controlled_alt_vs_controlled_default | naturalness | -0.0074 | -0.0082 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0318 | 0.0972 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0011 | -0.0103 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0307 | -0.1030 |
| controlled_alt_vs_controlled_default | persona_style | -0.0446 | -0.0704 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0090 | 0.0097 |
| controlled_alt_vs_controlled_default | length_score | -0.0650 | -0.0951 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0175 | 0.0185 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0029 | -0.0068 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.2006 | 2.4689 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1837 | 1.2414 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0760 | 0.0928 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2636 | 2.7619 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0536 | 1.1134 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.2281 | 5.8061 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0060 | 0.0104 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0025 | -0.0027 |
| controlled_alt_vs_proposed_raw | length_score | 0.2800 | 0.8276 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.2100 | 0.2781 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1773 | 0.7166 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2549 | 9.4519 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1852 | 1.2650 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0929 | 0.1159 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3364 | 14.8000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0649 | 1.7598 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.2281 | 5.8061 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0138 | 0.0239 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0049 | 0.0052 |
| controlled_alt_vs_candidate_no_context | length_score | 0.3150 | 1.0385 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.2800 | 0.4088 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.2065 | 0.9464 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0543 | (0.0110, 0.1042) | 0.0050 | 0.0543 | (0.0174, 0.0865) | 0.0003 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0015 | (-0.0314, 0.0406) | 0.4613 | 0.0015 | (-0.0185, 0.0205) | 0.4150 |
| proposed_vs_candidate_no_context | naturalness | 0.0170 | (-0.0235, 0.0572) | 0.2033 | 0.0170 | (-0.0210, 0.0492) | 0.1500 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0727 | (0.0136, 0.1409) | 0.0073 | 0.0727 | (0.0227, 0.1162) | 0.0000 |
| proposed_vs_candidate_no_context | context_overlap | 0.0113 | (-0.0003, 0.0259) | 0.0300 | 0.0113 | (0.0036, 0.0170) | 0.0000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0000 | (-0.0417, 0.0488) | 0.5517 | 0.0000 | (-0.0238, 0.0238) | 0.6367 |
| proposed_vs_candidate_no_context | persona_style | 0.0077 | (-0.0019, 0.0250) | 0.3613 | 0.0077 | (-0.0014, 0.0294) | 0.3167 |
| proposed_vs_candidate_no_context | distinct1 | 0.0074 | (-0.0056, 0.0202) | 0.1273 | 0.0074 | (-0.0045, 0.0193) | 0.0940 |
| proposed_vs_candidate_no_context | length_score | 0.0350 | (-0.1167, 0.1783) | 0.3293 | 0.0350 | (-0.1421, 0.1772) | 0.3360 |
| proposed_vs_candidate_no_context | sentence_score | 0.0700 | (-0.0175, 0.1575) | 0.0980 | 0.0700 | (0.0194, 0.1114) | 0.0090 |
| proposed_vs_candidate_no_context | overall_quality | 0.0292 | (-0.0075, 0.0689) | 0.0593 | 0.0292 | (0.0018, 0.0545) | 0.0143 |
| controlled_vs_proposed_raw | context_relevance | 0.1787 | (0.1304, 0.2291) | 0.0000 | 0.1787 | (0.1408, 0.2303) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.2172 | (0.1451, 0.2902) | 0.0000 | 0.2172 | (0.1342, 0.2833) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0834 | (0.0390, 0.1224) | 0.0003 | 0.0834 | (0.0345, 0.1284) | 0.0000 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2318 | (0.1636, 0.2955) | 0.0000 | 0.2318 | (0.1782, 0.2995) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0547 | (0.0376, 0.0705) | 0.0000 | 0.0547 | (0.0440, 0.0667) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2588 | (0.1783, 0.3443) | 0.0000 | 0.2588 | (0.1625, 0.3312) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.0506 | (0.0000, 0.1340) | 0.0367 | 0.0506 | (0.0000, 0.1765) | 0.0743 |
| controlled_vs_proposed_raw | distinct1 | -0.0115 | (-0.0313, 0.0061) | 0.9010 | -0.0115 | (-0.0330, 0.0052) | 0.8793 |
| controlled_vs_proposed_raw | length_score | 0.3450 | (0.1733, 0.4984) | 0.0003 | 0.3450 | (0.1375, 0.5094) | 0.0000 |
| controlled_vs_proposed_raw | sentence_score | 0.1925 | (0.1225, 0.2625) | 0.0000 | 0.1925 | (0.1313, 0.2676) | 0.0000 |
| controlled_vs_proposed_raw | overall_quality | 0.1802 | (0.1470, 0.2135) | 0.0000 | 0.1802 | (0.1573, 0.2091) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2330 | (0.2017, 0.2693) | 0.0000 | 0.2330 | (0.1987, 0.2757) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.2187 | (0.1331, 0.2977) | 0.0000 | 0.2187 | (0.1279, 0.2922) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.1003 | (0.0652, 0.1330) | 0.0000 | 0.1003 | (0.0593, 0.1403) | 0.0000 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3045 | (0.2591, 0.3500) | 0.0000 | 0.3045 | (0.2562, 0.3690) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0659 | (0.0543, 0.0774) | 0.0000 | 0.0659 | (0.0547, 0.0746) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2588 | (0.1552, 0.3526) | 0.0000 | 0.2588 | (0.1509, 0.3387) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.0583 | (0.0000, 0.1417) | 0.0410 | 0.0583 | (0.0000, 0.2059) | 0.3403 |
| controlled_vs_candidate_no_context | distinct1 | -0.0041 | (-0.0242, 0.0137) | 0.6497 | -0.0041 | (-0.0167, 0.0110) | 0.7053 |
| controlled_vs_candidate_no_context | length_score | 0.3800 | (0.2433, 0.5100) | 0.0000 | 0.3800 | (0.1889, 0.5444) | 0.0000 |
| controlled_vs_candidate_no_context | sentence_score | 0.2625 | (0.1925, 0.3325) | 0.0000 | 0.2625 | (0.2026, 0.3306) | 0.0000 |
| controlled_vs_candidate_no_context | overall_quality | 0.2094 | (0.1668, 0.2480) | 0.0000 | 0.2094 | (0.1722, 0.2448) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0220 | (-0.0187, 0.0662) | 0.1687 | 0.0220 | (-0.0250, 0.0517) | 0.1410 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0335 | (-0.1048, 0.0457) | 0.8140 | -0.0335 | (-0.0847, 0.0433) | 0.8247 |
| controlled_alt_vs_controlled_default | naturalness | -0.0074 | (-0.0505, 0.0350) | 0.6353 | -0.0074 | (-0.0360, 0.0314) | 0.6797 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0318 | (-0.0227, 0.0909) | 0.1423 | 0.0318 | (-0.0321, 0.0734) | 0.1257 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0011 | (-0.0172, 0.0151) | 0.5593 | -0.0011 | (-0.0069, 0.0034) | 0.6350 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0307 | (-0.1221, 0.0764) | 0.7317 | -0.0307 | (-0.1012, 0.0750) | 0.7327 |
| controlled_alt_vs_controlled_default | persona_style | -0.0446 | (-0.0946, -0.0037) | 1.0000 | -0.0446 | (-0.1236, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0090 | (-0.0127, 0.0314) | 0.2183 | 0.0090 | (-0.0120, 0.0238) | 0.1707 |
| controlled_alt_vs_controlled_default | length_score | -0.0650 | (-0.2733, 0.1417) | 0.7250 | -0.0650 | (-0.1947, 0.1019) | 0.7990 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0175 | (-0.0525, 0.0875) | 0.4060 | 0.0175 | (-0.0333, 0.0737) | 0.3747 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0029 | (-0.0321, 0.0226) | 0.5620 | -0.0029 | (-0.0152, 0.0137) | 0.6690 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.2006 | (0.1529, 0.2477) | 0.0000 | 0.2006 | (0.1893, 0.2159) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1837 | (0.1140, 0.2721) | 0.0000 | 0.1837 | (0.1170, 0.2767) | 0.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0760 | (0.0393, 0.1108) | 0.0000 | 0.0760 | (0.0614, 0.1029) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2636 | (0.1955, 0.3273) | 0.0000 | 0.2636 | (0.2483, 0.2841) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0536 | (0.0393, 0.0683) | 0.0000 | 0.0536 | (0.0434, 0.0637) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.2281 | (0.1471, 0.3410) | 0.0000 | 0.2281 | (0.1490, 0.3349) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0060 | (-0.0244, 0.0462) | 0.4513 | 0.0060 | (-0.0245, 0.0588) | 0.4453 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0025 | (-0.0225, 0.0170) | 0.5980 | -0.0025 | (-0.0322, 0.0219) | 0.5540 |
| controlled_alt_vs_proposed_raw | length_score | 0.2800 | (0.1083, 0.4317) | 0.0003 | 0.2800 | (0.2368, 0.3438) | 0.0000 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.2100 | (0.1050, 0.2975) | 0.0003 | 0.2100 | (0.1481, 0.2882) | 0.0000 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1773 | (0.1402, 0.2196) | 0.0000 | 0.1773 | (0.1529, 0.2128) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2549 | (0.2131, 0.3009) | 0.0000 | 0.2549 | (0.2318, 0.2802) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1852 | (0.1185, 0.2771) | 0.0000 | 0.1852 | (0.1244, 0.2821) | 0.0000 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0929 | (0.0465, 0.1371) | 0.0000 | 0.0929 | (0.0518, 0.1300) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3364 | (0.2818, 0.4000) | 0.0000 | 0.3364 | (0.3068, 0.3732) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0649 | (0.0511, 0.0798) | 0.0000 | 0.0649 | (0.0533, 0.0726) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.2281 | (0.1476, 0.3367) | 0.0000 | 0.2281 | (0.1556, 0.3313) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0138 | (-0.0187, 0.0583) | 0.2417 | 0.0138 | (-0.0260, 0.0882) | 0.3590 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0049 | (-0.0158, 0.0249) | 0.3230 | 0.0049 | (-0.0168, 0.0276) | 0.3177 |
| controlled_alt_vs_candidate_no_context | length_score | 0.3150 | (0.1067, 0.4983) | 0.0007 | 0.3150 | (0.1492, 0.4392) | 0.0003 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.2800 | (0.2100, 0.3325) | 0.0000 | 0.2800 | (0.2479, 0.3281) | 0.0000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.2065 | (0.1701, 0.2475) | 0.0000 | 0.2065 | (0.1785, 0.2397) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 7 | 5 | 8 | 0.5500 | 0.5833 |
| proposed_vs_candidate_no_context | persona_consistency | 3 | 4 | 13 | 0.4750 | 0.4286 |
| proposed_vs_candidate_no_context | naturalness | 7 | 5 | 8 | 0.5500 | 0.5833 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 7 | 2 | 11 | 0.6250 | 0.7778 |
| proposed_vs_candidate_no_context | context_overlap | 7 | 5 | 8 | 0.5500 | 0.5833 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 2 | 3 | 15 | 0.4750 | 0.4000 |
| proposed_vs_candidate_no_context | persona_style | 1 | 1 | 18 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | distinct1 | 8 | 3 | 9 | 0.6250 | 0.7273 |
| proposed_vs_candidate_no_context | length_score | 7 | 5 | 8 | 0.5500 | 0.5833 |
| proposed_vs_candidate_no_context | sentence_score | 6 | 2 | 12 | 0.6000 | 0.7500 |
| proposed_vs_candidate_no_context | overall_quality | 7 | 5 | 8 | 0.5500 | 0.5833 |
| controlled_vs_proposed_raw | context_relevance | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_vs_proposed_raw | persona_consistency | 15 | 0 | 5 | 0.8750 | 1.0000 |
| controlled_vs_proposed_raw | naturalness | 18 | 2 | 0 | 0.9000 | 0.9000 |
| controlled_vs_proposed_raw | context_keyword_coverage | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_vs_proposed_raw | context_overlap | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 15 | 0 | 5 | 0.8750 | 1.0000 |
| controlled_vs_proposed_raw | persona_style | 3 | 0 | 17 | 0.5750 | 1.0000 |
| controlled_vs_proposed_raw | distinct1 | 8 | 11 | 1 | 0.4250 | 0.4211 |
| controlled_vs_proposed_raw | length_score | 17 | 3 | 0 | 0.8500 | 0.8500 |
| controlled_vs_proposed_raw | sentence_score | 11 | 0 | 9 | 0.7750 | 1.0000 |
| controlled_vs_proposed_raw | overall_quality | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_relevance | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 15 | 2 | 3 | 0.8250 | 0.8824 |
| controlled_vs_candidate_no_context | naturalness | 18 | 2 | 0 | 0.9000 | 0.9000 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 15 | 2 | 3 | 0.8250 | 0.8824 |
| controlled_vs_candidate_no_context | persona_style | 3 | 0 | 17 | 0.5750 | 1.0000 |
| controlled_vs_candidate_no_context | distinct1 | 11 | 9 | 0 | 0.5500 | 0.5500 |
| controlled_vs_candidate_no_context | length_score | 16 | 4 | 0 | 0.8000 | 0.8000 |
| controlled_vs_candidate_no_context | sentence_score | 15 | 0 | 5 | 0.8750 | 1.0000 |
| controlled_vs_candidate_no_context | overall_quality | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 8 | 8 | 4 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | persona_consistency | 4 | 9 | 7 | 0.3750 | 0.3077 |
| controlled_alt_vs_controlled_default | naturalness | 8 | 8 | 4 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 8 | 6 | 6 | 0.5500 | 0.5714 |
| controlled_alt_vs_controlled_default | context_overlap | 8 | 8 | 4 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 4 | 8 | 8 | 0.4000 | 0.3333 |
| controlled_alt_vs_controlled_default | persona_style | 0 | 4 | 16 | 0.4000 | 0.0000 |
| controlled_alt_vs_controlled_default | distinct1 | 8 | 8 | 4 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | length_score | 7 | 9 | 4 | 0.4500 | 0.4375 |
| controlled_alt_vs_controlled_default | sentence_score | 3 | 2 | 15 | 0.5250 | 0.6000 |
| controlled_alt_vs_controlled_default | overall_quality | 9 | 7 | 4 | 0.5500 | 0.5625 |
| controlled_alt_vs_proposed_raw | context_relevance | 18 | 2 | 0 | 0.9000 | 0.9000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 18 | 1 | 1 | 0.9250 | 0.9474 |
| controlled_alt_vs_proposed_raw | naturalness | 16 | 4 | 0 | 0.8000 | 0.8000 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 18 | 1 | 1 | 0.9250 | 0.9474 |
| controlled_alt_vs_proposed_raw | context_overlap | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 18 | 1 | 1 | 0.9250 | 0.9474 |
| controlled_alt_vs_proposed_raw | persona_style | 1 | 2 | 17 | 0.4750 | 0.3333 |
| controlled_alt_vs_proposed_raw | distinct1 | 12 | 8 | 0 | 0.6000 | 0.6000 |
| controlled_alt_vs_proposed_raw | length_score | 13 | 6 | 1 | 0.6750 | 0.6842 |
| controlled_alt_vs_proposed_raw | sentence_score | 14 | 2 | 4 | 0.8000 | 0.8750 |
| controlled_alt_vs_proposed_raw | overall_quality | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 17 | 1 | 2 | 0.9000 | 0.9444 |
| controlled_alt_vs_candidate_no_context | naturalness | 17 | 3 | 0 | 0.8500 | 0.8500 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 17 | 0 | 3 | 0.9250 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_style | 2 | 2 | 16 | 0.5000 | 0.5000 |
| controlled_alt_vs_candidate_no_context | distinct1 | 12 | 7 | 1 | 0.6250 | 0.6316 |
| controlled_alt_vs_candidate_no_context | length_score | 14 | 6 | 0 | 0.7000 | 0.7000 |
| controlled_alt_vs_candidate_no_context | sentence_score | 16 | 0 | 4 | 0.9000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 20 | 0 | 0 | 1.0000 | 1.0000 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.4000 | 0.2500 | 0.7500 |
| proposed_contextual_controlled_tuned | 0.0000 | 0.0000 | 0.4000 | 0.3500 | 0.6500 |
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
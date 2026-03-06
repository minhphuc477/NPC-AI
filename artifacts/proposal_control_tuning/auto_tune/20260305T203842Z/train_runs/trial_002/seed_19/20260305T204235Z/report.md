# Proposal Alignment Evaluation Report

- Run ID: `20260305T204235Z`
- Generated: `2026-03-05T20:46:26.546045+00:00`
- Scenarios: `artifacts\proposal_control_tuning\auto_tune\20260305T203842Z\train_runs\trial_002\seed_19\20260305T204235Z\scenarios.jsonl`
- Scenario count: `20`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_tuned`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2693 (0.2371, 0.3043) | 0.2944 (0.2397, 0.3482) | 0.9058 (0.8821, 0.9273) | 0.4067 (0.3809, 0.4317) | n/a |
| proposed_contextual_controlled_tuned | 0.2962 (0.2595, 0.3371) | 0.3397 (0.2744, 0.4104) | 0.8993 (0.8709, 0.9243) | 0.4349 (0.4026, 0.4710) | n/a |
| proposed_contextual | 0.0789 (0.0407, 0.1214) | 0.1483 (0.1100, 0.1883) | 0.8020 (0.7737, 0.8309) | 0.2431 (0.2091, 0.2808) | n/a |
| candidate_no_context | 0.0276 (0.0163, 0.0407) | 0.1588 (0.1105, 0.2105) | 0.8176 (0.7818, 0.8577) | 0.2262 (0.2021, 0.2516) | n/a |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0512 | 1.8550 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0105 | -0.0660 |
| proposed_vs_candidate_no_context | naturalness | -0.0156 | -0.0191 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0682 | 3.0000 |
| proposed_vs_candidate_no_context | context_overlap | 0.0117 | 0.2997 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0131 | -0.2391 |
| proposed_vs_candidate_no_context | persona_style | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | distinct1 | -0.0042 | -0.0044 |
| proposed_vs_candidate_no_context | length_score | -0.0433 | -0.1340 |
| proposed_vs_candidate_no_context | sentence_score | -0.0525 | -0.0695 |
| proposed_vs_candidate_no_context | overall_quality | 0.0169 | 0.0745 |
| controlled_vs_proposed_raw | context_relevance | 0.1905 | 2.4154 |
| controlled_vs_proposed_raw | persona_consistency | 0.1460 | 0.9845 |
| controlled_vs_proposed_raw | naturalness | 0.1038 | 0.1294 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2500 | 2.7500 |
| controlled_vs_proposed_raw | context_overlap | 0.0516 | 1.0167 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1745 | 4.1886 |
| controlled_vs_proposed_raw | persona_style | 0.0321 | 0.0558 |
| controlled_vs_proposed_raw | distinct1 | -0.0016 | -0.0017 |
| controlled_vs_proposed_raw | length_score | 0.4083 | 1.4583 |
| controlled_vs_proposed_raw | sentence_score | 0.2275 | 0.3238 |
| controlled_vs_proposed_raw | overall_quality | 0.1636 | 0.6731 |
| controlled_vs_candidate_no_context | context_relevance | 0.2417 | 8.7511 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1356 | 0.8536 |
| controlled_vs_candidate_no_context | naturalness | 0.0882 | 0.1079 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3182 | 14.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0633 | 1.6209 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1614 | 2.9478 |
| controlled_vs_candidate_no_context | persona_style | 0.0321 | 0.0558 |
| controlled_vs_candidate_no_context | distinct1 | -0.0058 | -0.0061 |
| controlled_vs_candidate_no_context | length_score | 0.3650 | 1.1289 |
| controlled_vs_candidate_no_context | sentence_score | 0.1750 | 0.2318 |
| controlled_vs_candidate_no_context | overall_quality | 0.1805 | 0.7978 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0268 | 0.0997 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0454 | 0.1541 |
| controlled_alt_vs_controlled_default | naturalness | -0.0064 | -0.0071 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0364 | 0.1067 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0046 | 0.0452 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0557 | 0.2577 |
| controlled_alt_vs_controlled_default | persona_style | 0.0040 | 0.0065 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0027 | 0.0029 |
| controlled_alt_vs_controlled_default | length_score | -0.0550 | -0.0799 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0350 | 0.0376 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0282 | 0.0694 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.2173 | 2.7558 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1914 | 1.2903 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0974 | 0.1214 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2864 | 3.1500 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0562 | 1.1077 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.2302 | 5.5257 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0360 | 0.0627 |
| controlled_alt_vs_proposed_raw | distinct1 | 0.0011 | 0.0012 |
| controlled_alt_vs_proposed_raw | length_score | 0.3533 | 1.2619 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.2625 | 0.3737 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1918 | 0.7893 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2686 | 9.7229 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1809 | 1.1392 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0818 | 0.1000 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3545 | 15.6000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0679 | 1.7393 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.2171 | 3.9652 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0360 | 0.0627 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0031 | -0.0033 |
| controlled_alt_vs_candidate_no_context | length_score | 0.3100 | 0.9588 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.2100 | 0.2781 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.2087 | 0.9226 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0512 | (0.0174, 0.0902) | 0.0003 | 0.0512 | (0.0139, 0.0896) | 0.0117 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0105 | (-0.0553, 0.0295) | 0.6740 | -0.0105 | (-0.0429, 0.0351) | 0.7160 |
| proposed_vs_candidate_no_context | naturalness | -0.0156 | (-0.0509, 0.0175) | 0.8187 | -0.0156 | (-0.0285, -0.0039) | 1.0000 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0682 | (0.0227, 0.1227) | 0.0017 | 0.0682 | (0.0202, 0.1196) | 0.0077 |
| proposed_vs_candidate_no_context | context_overlap | 0.0117 | (-0.0008, 0.0259) | 0.0353 | 0.0117 | (-0.0013, 0.0261) | 0.0800 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0131 | (-0.0702, 0.0417) | 0.6663 | -0.0131 | (-0.0565, 0.0439) | 0.7177 |
| proposed_vs_candidate_no_context | persona_style | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | distinct1 | -0.0042 | (-0.0226, 0.0148) | 0.6773 | -0.0042 | (-0.0116, 0.0035) | 0.8620 |
| proposed_vs_candidate_no_context | length_score | -0.0433 | (-0.1617, 0.0733) | 0.7697 | -0.0433 | (-0.0842, -0.0063) | 1.0000 |
| proposed_vs_candidate_no_context | sentence_score | -0.0525 | (-0.1225, 0.0175) | 0.9587 | -0.0525 | (-0.1273, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | overall_quality | 0.0169 | (-0.0096, 0.0454) | 0.1230 | 0.0169 | (-0.0022, 0.0422) | 0.0747 |
| controlled_vs_proposed_raw | context_relevance | 0.1905 | (0.1348, 0.2440) | 0.0000 | 0.1905 | (0.1482, 0.2319) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.1460 | (0.0932, 0.1938) | 0.0000 | 0.1460 | (0.0696, 0.2173) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.1038 | (0.0646, 0.1390) | 0.0000 | 0.1038 | (0.0494, 0.1443) | 0.0000 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2500 | (0.1726, 0.3227) | 0.0000 | 0.2500 | (0.1962, 0.3068) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0516 | (0.0353, 0.0672) | 0.0000 | 0.0516 | (0.0355, 0.0655) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1745 | (0.1093, 0.2307) | 0.0000 | 0.1745 | (0.0870, 0.2602) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.0321 | (-0.0025, 0.0917) | 0.1157 | 0.0321 | (-0.0029, 0.1176) | 0.3293 |
| controlled_vs_proposed_raw | distinct1 | -0.0016 | (-0.0167, 0.0123) | 0.5740 | -0.0016 | (-0.0210, 0.0172) | 0.5717 |
| controlled_vs_proposed_raw | length_score | 0.4083 | (0.2300, 0.5517) | 0.0003 | 0.4083 | (0.2238, 0.5647) | 0.0000 |
| controlled_vs_proposed_raw | sentence_score | 0.2275 | (0.1575, 0.2975) | 0.0000 | 0.2275 | (0.1167, 0.3281) | 0.0000 |
| controlled_vs_proposed_raw | overall_quality | 0.1636 | (0.1211, 0.2004) | 0.0000 | 0.1636 | (0.1127, 0.2097) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2417 | (0.2073, 0.2771) | 0.0000 | 0.2417 | (0.2233, 0.2592) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1356 | (0.0914, 0.1798) | 0.0000 | 0.1356 | (0.0770, 0.1824) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.0882 | (0.0389, 0.1326) | 0.0003 | 0.0882 | (0.0298, 0.1404) | 0.0020 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3182 | (0.2727, 0.3682) | 0.0000 | 0.3182 | (0.2929, 0.3397) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0633 | (0.0529, 0.0735) | 0.0000 | 0.0633 | (0.0506, 0.0737) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1614 | (0.1062, 0.2188) | 0.0000 | 0.1614 | (0.0958, 0.2147) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.0321 | (-0.0025, 0.0917) | 0.1170 | 0.0321 | (-0.0030, 0.1176) | 0.3403 |
| controlled_vs_candidate_no_context | distinct1 | -0.0058 | (-0.0217, 0.0093) | 0.7537 | -0.0058 | (-0.0250, 0.0148) | 0.6943 |
| controlled_vs_candidate_no_context | length_score | 0.3650 | (0.1650, 0.5483) | 0.0007 | 0.3650 | (0.1842, 0.5302) | 0.0000 |
| controlled_vs_candidate_no_context | sentence_score | 0.1750 | (0.0700, 0.2629) | 0.0013 | 0.1750 | (0.0184, 0.3111) | 0.0120 |
| controlled_vs_candidate_no_context | overall_quality | 0.1805 | (0.1527, 0.2088) | 0.0000 | 0.1805 | (0.1500, 0.2074) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0268 | (-0.0235, 0.0787) | 0.1660 | 0.0268 | (-0.0143, 0.0612) | 0.1010 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0454 | (0.0055, 0.0944) | 0.0130 | 0.0454 | (0.0000, 0.0864) | 0.0670 |
| controlled_alt_vs_controlled_default | naturalness | -0.0064 | (-0.0424, 0.0304) | 0.6433 | -0.0064 | (-0.0291, 0.0277) | 0.6967 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0364 | (-0.0318, 0.1045) | 0.1517 | 0.0364 | (-0.0160, 0.0818) | 0.0847 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0046 | (-0.0147, 0.0265) | 0.3650 | 0.0046 | (-0.0087, 0.0135) | 0.2310 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0557 | (0.0048, 0.1114) | 0.0187 | 0.0557 | (0.0000, 0.1060) | 0.0817 |
| controlled_alt_vs_controlled_default | persona_style | 0.0040 | (-0.0254, 0.0333) | 0.4470 | 0.0040 | (-0.0101, 0.0294) | 0.4287 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0027 | (-0.0118, 0.0177) | 0.3820 | 0.0027 | (-0.0092, 0.0197) | 0.4033 |
| controlled_alt_vs_controlled_default | length_score | -0.0550 | (-0.2200, 0.1100) | 0.7563 | -0.0550 | (-0.1519, 0.0714) | 0.8243 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0350 | (-0.0350, 0.1050) | 0.2120 | 0.0350 | (0.0000, 0.1105) | 0.3257 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0282 | (0.0031, 0.0534) | 0.0117 | 0.0282 | (0.0032, 0.0475) | 0.0107 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.2173 | (0.1611, 0.2752) | 0.0000 | 0.2173 | (0.1649, 0.2660) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1914 | (0.1144, 0.2666) | 0.0000 | 0.1914 | (0.0696, 0.2988) | 0.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0974 | (0.0646, 0.1286) | 0.0000 | 0.0974 | (0.0705, 0.1330) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2864 | (0.2091, 0.3591) | 0.0000 | 0.2864 | (0.2172, 0.3420) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0562 | (0.0384, 0.0791) | 0.0000 | 0.0562 | (0.0334, 0.0771) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.2302 | (0.1443, 0.3238) | 0.0000 | 0.2302 | (0.0870, 0.3686) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0360 | (-0.0056, 0.0888) | 0.0630 | 0.0360 | (-0.0130, 0.1471) | 0.3340 |
| controlled_alt_vs_proposed_raw | distinct1 | 0.0011 | (-0.0189, 0.0214) | 0.4563 | 0.0011 | (-0.0085, 0.0102) | 0.3847 |
| controlled_alt_vs_proposed_raw | length_score | 0.3533 | (0.2266, 0.4850) | 0.0000 | 0.3533 | (0.2439, 0.4947) | 0.0000 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.2625 | (0.1925, 0.3150) | 0.0000 | 0.2625 | (0.2167, 0.3281) | 0.0000 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1918 | (0.1432, 0.2381) | 0.0000 | 0.1918 | (0.1233, 0.2515) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2686 | (0.2297, 0.3123) | 0.0000 | 0.2686 | (0.2156, 0.3085) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1809 | (0.1168, 0.2500) | 0.0000 | 0.1809 | (0.0770, 0.2659) | 0.0000 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0818 | (0.0346, 0.1271) | 0.0003 | 0.0818 | (0.0452, 0.1239) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3545 | (0.3000, 0.4136) | 0.0000 | 0.3545 | (0.2834, 0.4091) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0679 | (0.0507, 0.0886) | 0.0000 | 0.0679 | (0.0531, 0.0780) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.2171 | (0.1405, 0.2960) | 0.0000 | 0.2171 | (0.0982, 0.3160) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0360 | (-0.0056, 0.0860) | 0.0610 | 0.0360 | (-0.0130, 0.1471) | 0.3393 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0031 | (-0.0217, 0.0156) | 0.6233 | -0.0031 | (-0.0193, 0.0120) | 0.6117 |
| controlled_alt_vs_candidate_no_context | length_score | 0.3100 | (0.1133, 0.4984) | 0.0013 | 0.3100 | (0.1807, 0.4729) | 0.0000 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.2100 | (0.1225, 0.2975) | 0.0000 | 0.2100 | (0.1273, 0.3111) | 0.0000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.2087 | (0.1763, 0.2438) | 0.0000 | 0.2087 | (0.1556, 0.2520) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 9 | 2 | 9 | 0.6750 | 0.8182 |
| proposed_vs_candidate_no_context | persona_consistency | 3 | 3 | 14 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | naturalness | 3 | 8 | 9 | 0.3750 | 0.2727 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 6 | 0 | 14 | 0.6500 | 1.0000 |
| proposed_vs_candidate_no_context | context_overlap | 7 | 4 | 9 | 0.5750 | 0.6364 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 3 | 3 | 14 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | persona_style | 0 | 0 | 20 | 0.5000 | nan |
| proposed_vs_candidate_no_context | distinct1 | 4 | 5 | 11 | 0.4750 | 0.4444 |
| proposed_vs_candidate_no_context | length_score | 4 | 7 | 9 | 0.4250 | 0.3636 |
| proposed_vs_candidate_no_context | sentence_score | 1 | 4 | 15 | 0.4250 | 0.2000 |
| proposed_vs_candidate_no_context | overall_quality | 7 | 4 | 9 | 0.5750 | 0.6364 |
| controlled_vs_proposed_raw | context_relevance | 18 | 2 | 0 | 0.9000 | 0.9000 |
| controlled_vs_proposed_raw | persona_consistency | 15 | 1 | 4 | 0.8500 | 0.9375 |
| controlled_vs_proposed_raw | naturalness | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_vs_proposed_raw | context_keyword_coverage | 17 | 2 | 1 | 0.8750 | 0.8947 |
| controlled_vs_proposed_raw | context_overlap | 18 | 2 | 0 | 0.9000 | 0.9000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 15 | 1 | 4 | 0.8500 | 0.9375 |
| controlled_vs_proposed_raw | persona_style | 2 | 1 | 17 | 0.5250 | 0.6667 |
| controlled_vs_proposed_raw | distinct1 | 10 | 10 | 0 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | length_score | 17 | 3 | 0 | 0.8500 | 0.8500 |
| controlled_vs_proposed_raw | sentence_score | 13 | 0 | 7 | 0.8250 | 1.0000 |
| controlled_vs_proposed_raw | overall_quality | 18 | 2 | 0 | 0.9000 | 0.9000 |
| controlled_vs_candidate_no_context | context_relevance | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 14 | 0 | 6 | 0.8500 | 1.0000 |
| controlled_vs_candidate_no_context | naturalness | 16 | 4 | 0 | 0.8000 | 0.8000 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 14 | 0 | 6 | 0.8500 | 1.0000 |
| controlled_vs_candidate_no_context | persona_style | 2 | 1 | 17 | 0.5250 | 0.6667 |
| controlled_vs_candidate_no_context | distinct1 | 10 | 10 | 0 | 0.5000 | 0.5000 |
| controlled_vs_candidate_no_context | length_score | 14 | 4 | 2 | 0.7500 | 0.7778 |
| controlled_vs_candidate_no_context | sentence_score | 12 | 2 | 6 | 0.7500 | 0.8571 |
| controlled_vs_candidate_no_context | overall_quality | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 10 | 6 | 4 | 0.6000 | 0.6250 |
| controlled_alt_vs_controlled_default | persona_consistency | 7 | 3 | 10 | 0.6000 | 0.7000 |
| controlled_alt_vs_controlled_default | naturalness | 7 | 9 | 4 | 0.4500 | 0.4375 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 8 | 4 | 8 | 0.6000 | 0.6667 |
| controlled_alt_vs_controlled_default | context_overlap | 7 | 9 | 4 | 0.4500 | 0.4375 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 6 | 1 | 13 | 0.6250 | 0.8571 |
| controlled_alt_vs_controlled_default | persona_style | 2 | 2 | 16 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | distinct1 | 8 | 7 | 5 | 0.5250 | 0.5333 |
| controlled_alt_vs_controlled_default | length_score | 6 | 9 | 5 | 0.4250 | 0.4000 |
| controlled_alt_vs_controlled_default | sentence_score | 3 | 1 | 16 | 0.5500 | 0.7500 |
| controlled_alt_vs_controlled_default | overall_quality | 13 | 3 | 4 | 0.7500 | 0.8125 |
| controlled_alt_vs_proposed_raw | context_relevance | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_alt_vs_proposed_raw | persona_consistency | 14 | 1 | 5 | 0.8250 | 0.9333 |
| controlled_alt_vs_proposed_raw | naturalness | 17 | 3 | 0 | 0.8500 | 0.8500 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 18 | 0 | 2 | 0.9500 | 1.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 14 | 1 | 5 | 0.8250 | 0.9333 |
| controlled_alt_vs_proposed_raw | persona_style | 3 | 1 | 16 | 0.5500 | 0.7500 |
| controlled_alt_vs_proposed_raw | distinct1 | 10 | 10 | 0 | 0.5000 | 0.5000 |
| controlled_alt_vs_proposed_raw | length_score | 18 | 2 | 0 | 0.9000 | 0.9000 |
| controlled_alt_vs_proposed_raw | sentence_score | 15 | 0 | 5 | 0.8750 | 1.0000 |
| controlled_alt_vs_proposed_raw | overall_quality | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_alt_vs_candidate_no_context | context_relevance | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 15 | 0 | 5 | 0.8750 | 1.0000 |
| controlled_alt_vs_candidate_no_context | naturalness | 15 | 5 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 15 | 0 | 5 | 0.8750 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_style | 3 | 1 | 16 | 0.5500 | 0.7500 |
| controlled_alt_vs_candidate_no_context | distinct1 | 9 | 10 | 1 | 0.4750 | 0.4737 |
| controlled_alt_vs_candidate_no_context | length_score | 14 | 6 | 0 | 0.7000 | 0.7000 |
| controlled_alt_vs_candidate_no_context | sentence_score | 13 | 1 | 6 | 0.8000 | 0.9286 |
| controlled_alt_vs_candidate_no_context | overall_quality | 20 | 0 | 0 | 1.0000 | 1.0000 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.3500 | 0.3500 | 0.6500 |
| proposed_contextual_controlled_tuned | 0.0000 | 0.0000 | 0.5000 | 0.2500 | 0.7500 |
| proposed_contextual | 0.0000 | 0.0000 | 0.5500 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.5500 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `5`
- Unique template signatures: `17`
- Template signature ratio: `0.8500`
- Effective sample size by source clustering: `4.65`
- Effective sample size by template-signature clustering: `15.38`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: unavailable (disabled_by_flag).

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
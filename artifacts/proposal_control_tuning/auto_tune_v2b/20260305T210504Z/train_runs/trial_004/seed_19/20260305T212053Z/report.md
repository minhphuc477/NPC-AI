# Proposal Alignment Evaluation Report

- Run ID: `20260305T212053Z`
- Generated: `2026-03-05T21:24:42.643659+00:00`
- Scenarios: `artifacts\proposal_control_tuning\auto_tune_v2b\20260305T210504Z\train_runs\trial_004\seed_19\20260305T212053Z\scenarios.jsonl`
- Scenario count: `20`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_tuned`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2539 (0.2121, 0.2967) | 0.3607 (0.2764, 0.4505) | 0.8910 (0.8639, 0.9162) | 0.4209 (0.3834, 0.4644) | n/a |
| proposed_contextual_controlled_tuned | 0.2498 (0.2061, 0.2993) | 0.3304 (0.2533, 0.4219) | 0.8830 (0.8589, 0.9062) | 0.4062 (0.3705, 0.4452) | n/a |
| proposed_contextual | 0.0565 (0.0253, 0.0932) | 0.1685 (0.1230, 0.2189) | 0.8058 (0.7775, 0.8387) | 0.2407 (0.2091, 0.2728) | n/a |
| candidate_no_context | 0.0337 (0.0178, 0.0507) | 0.1340 (0.0950, 0.1757) | 0.7976 (0.7706, 0.8244) | 0.2158 (0.1949, 0.2387) | n/a |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0229 | 0.6784 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0344 | 0.2568 |
| proposed_vs_candidate_no_context | naturalness | 0.0082 | 0.0103 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0273 | 0.8571 |
| proposed_vs_candidate_no_context | context_overlap | 0.0125 | 0.3295 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0410 | 1.7200 |
| proposed_vs_candidate_no_context | persona_style | 0.0083 | 0.0145 |
| proposed_vs_candidate_no_context | distinct1 | 0.0048 | 0.0052 |
| proposed_vs_candidate_no_context | length_score | 0.0400 | 0.1611 |
| proposed_vs_candidate_no_context | sentence_score | -0.0175 | -0.0237 |
| proposed_vs_candidate_no_context | overall_quality | 0.0248 | 0.1150 |
| controlled_vs_proposed_raw | context_relevance | 0.1974 | 3.4910 |
| controlled_vs_proposed_raw | persona_consistency | 0.1922 | 1.1410 |
| controlled_vs_proposed_raw | naturalness | 0.0852 | 0.1058 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2591 | 4.3846 |
| controlled_vs_proposed_raw | context_overlap | 0.0534 | 1.0554 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2286 | 3.5294 |
| controlled_vs_proposed_raw | persona_style | 0.0469 | 0.0804 |
| controlled_vs_proposed_raw | distinct1 | -0.0011 | -0.0012 |
| controlled_vs_proposed_raw | length_score | 0.3233 | 1.1214 |
| controlled_vs_proposed_raw | sentence_score | 0.2100 | 0.2917 |
| controlled_vs_proposed_raw | overall_quality | 0.1802 | 0.7488 |
| controlled_vs_candidate_no_context | context_relevance | 0.2202 | 6.5377 |
| controlled_vs_candidate_no_context | persona_consistency | 0.2267 | 1.6909 |
| controlled_vs_candidate_no_context | naturalness | 0.0934 | 0.1171 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.2864 | 9.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0659 | 1.7328 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2695 | 11.3200 |
| controlled_vs_candidate_no_context | persona_style | 0.0552 | 0.0960 |
| controlled_vs_candidate_no_context | distinct1 | 0.0037 | 0.0040 |
| controlled_vs_candidate_no_context | length_score | 0.3633 | 1.4631 |
| controlled_vs_candidate_no_context | sentence_score | 0.1925 | 0.2610 |
| controlled_vs_candidate_no_context | overall_quality | 0.2050 | 0.9500 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0042 | -0.0164 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0303 | -0.0841 |
| controlled_alt_vs_controlled_default | naturalness | -0.0080 | -0.0090 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0045 | -0.0143 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0033 | -0.0314 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0345 | -0.1177 |
| controlled_alt_vs_controlled_default | persona_style | -0.0135 | -0.0215 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0097 | 0.0103 |
| controlled_alt_vs_controlled_default | length_score | -0.0333 | -0.0545 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0525 | -0.0565 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0146 | -0.0348 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1932 | 3.4174 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1619 | 0.9610 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0772 | 0.0958 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2545 | 4.3077 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0501 | 0.9909 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1940 | 2.9963 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0333 | 0.0571 |
| controlled_alt_vs_proposed_raw | distinct1 | 0.0086 | 0.0091 |
| controlled_alt_vs_proposed_raw | length_score | 0.2900 | 1.0058 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1575 | 0.2188 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1656 | 0.6880 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2161 | 6.4141 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1963 | 1.4647 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0854 | 0.1070 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.2818 | 8.8571 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0627 | 1.6470 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.2350 | 9.8700 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0417 | 0.0725 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0134 | 0.0143 |
| controlled_alt_vs_candidate_no_context | length_score | 0.3300 | 1.3289 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1400 | 0.1898 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1904 | 0.8821 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0229 | (-0.0028, 0.0519) | 0.0450 | 0.0229 | (-0.0031, 0.0472) | 0.0590 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0344 | (0.0067, 0.0671) | 0.0080 | 0.0344 | (0.0109, 0.0574) | 0.0103 |
| proposed_vs_candidate_no_context | naturalness | 0.0082 | (-0.0236, 0.0377) | 0.2990 | 0.0082 | (-0.0085, 0.0232) | 0.1387 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0273 | (-0.0045, 0.0636) | 0.0790 | 0.0273 | (-0.0107, 0.0606) | 0.0897 |
| proposed_vs_candidate_no_context | context_overlap | 0.0125 | (0.0017, 0.0238) | 0.0133 | 0.0125 | (0.0042, 0.0218) | 0.0000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0410 | (0.0083, 0.0819) | 0.0100 | 0.0410 | (0.0118, 0.0676) | 0.0087 |
| proposed_vs_candidate_no_context | persona_style | 0.0083 | (0.0000, 0.0250) | 0.3613 | 0.0083 | (0.0000, 0.0294) | 0.3167 |
| proposed_vs_candidate_no_context | distinct1 | 0.0048 | (-0.0123, 0.0217) | 0.2750 | 0.0048 | (-0.0027, 0.0111) | 0.0950 |
| proposed_vs_candidate_no_context | length_score | 0.0400 | (-0.0667, 0.1383) | 0.2210 | 0.0400 | (-0.0050, 0.0902) | 0.0437 |
| proposed_vs_candidate_no_context | sentence_score | -0.0175 | (-0.1050, 0.0700) | 0.7040 | -0.0175 | (-0.0737, 0.0437) | 0.8200 |
| proposed_vs_candidate_no_context | overall_quality | 0.0248 | (0.0039, 0.0505) | 0.0073 | 0.0248 | (0.0123, 0.0336) | 0.0000 |
| controlled_vs_proposed_raw | context_relevance | 0.1974 | (0.1352, 0.2562) | 0.0000 | 0.1974 | (0.1630, 0.2417) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.1922 | (0.1133, 0.2840) | 0.0000 | 0.1922 | (0.0996, 0.2796) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0852 | (0.0416, 0.1279) | 0.0000 | 0.0852 | (0.0400, 0.1307) | 0.0000 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2591 | (0.1773, 0.3364) | 0.0000 | 0.2591 | (0.2133, 0.3155) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0534 | (0.0311, 0.0750) | 0.0000 | 0.0534 | (0.0329, 0.0725) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2286 | (0.1321, 0.3371) | 0.0000 | 0.2286 | (0.1111, 0.3373) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.0469 | (-0.0063, 0.1333) | 0.1157 | 0.0469 | (-0.0072, 0.1765) | 0.3293 |
| controlled_vs_proposed_raw | distinct1 | -0.0011 | (-0.0113, 0.0093) | 0.5790 | -0.0011 | (-0.0160, 0.0120) | 0.5797 |
| controlled_vs_proposed_raw | length_score | 0.3233 | (0.1217, 0.5033) | 0.0017 | 0.3233 | (0.1352, 0.5042) | 0.0000 |
| controlled_vs_proposed_raw | sentence_score | 0.2100 | (0.1400, 0.2800) | 0.0000 | 0.2100 | (0.1750, 0.2722) | 0.0000 |
| controlled_vs_proposed_raw | overall_quality | 0.1802 | (0.1323, 0.2323) | 0.0000 | 0.1802 | (0.1320, 0.2219) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2202 | (0.1717, 0.2732) | 0.0000 | 0.2202 | (0.1631, 0.2724) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.2267 | (0.1540, 0.3133) | 0.0000 | 0.2267 | (0.1309, 0.3103) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.0934 | (0.0591, 0.1250) | 0.0000 | 0.0934 | (0.0488, 0.1338) | 0.0000 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.2864 | (0.2182, 0.3545) | 0.0000 | 0.2864 | (0.2193, 0.3583) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0659 | (0.0486, 0.0849) | 0.0000 | 0.0659 | (0.0481, 0.0896) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2695 | (0.1817, 0.3786) | 0.0000 | 0.2695 | (0.1608, 0.3596) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.0552 | (-0.0063, 0.1469) | 0.1170 | 0.0552 | (-0.0075, 0.2059) | 0.3403 |
| controlled_vs_candidate_no_context | distinct1 | 0.0037 | (-0.0118, 0.0192) | 0.3163 | 0.0037 | (-0.0161, 0.0196) | 0.3357 |
| controlled_vs_candidate_no_context | length_score | 0.3633 | (0.2083, 0.5100) | 0.0000 | 0.3633 | (0.2053, 0.5175) | 0.0000 |
| controlled_vs_candidate_no_context | sentence_score | 0.1925 | (0.1050, 0.2800) | 0.0000 | 0.1925 | (0.1114, 0.2917) | 0.0000 |
| controlled_vs_candidate_no_context | overall_quality | 0.2050 | (0.1581, 0.2526) | 0.0000 | 0.2050 | (0.1498, 0.2536) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0042 | (-0.0715, 0.0665) | 0.5627 | -0.0042 | (-0.0470, 0.0285) | 0.5923 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0303 | (-0.1317, 0.0757) | 0.7213 | -0.0303 | (-0.1458, 0.1083) | 0.6817 |
| controlled_alt_vs_controlled_default | naturalness | -0.0080 | (-0.0312, 0.0157) | 0.7423 | -0.0080 | (-0.0334, 0.0241) | 0.7413 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0045 | (-0.0909, 0.0865) | 0.5490 | -0.0045 | (-0.0535, 0.0335) | 0.5620 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0033 | (-0.0270, 0.0206) | 0.6027 | -0.0033 | (-0.0319, 0.0210) | 0.6157 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0345 | (-0.1686, 0.0967) | 0.6990 | -0.0345 | (-0.1829, 0.1458) | 0.6507 |
| controlled_alt_vs_controlled_default | persona_style | -0.0135 | (-0.0583, 0.0198) | 0.7237 | -0.0135 | (-0.0588, 0.0072) | 0.7467 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0097 | (-0.0051, 0.0246) | 0.0940 | 0.0097 | (-0.0030, 0.0182) | 0.0740 |
| controlled_alt_vs_controlled_default | length_score | -0.0333 | (-0.1450, 0.0833) | 0.7150 | -0.0333 | (-0.1278, 0.1176) | 0.7303 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0525 | (-0.1400, 0.0350) | 0.9150 | -0.0525 | (-0.1235, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0146 | (-0.0652, 0.0279) | 0.7197 | -0.0146 | (-0.0584, 0.0395) | 0.7100 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1932 | (0.1365, 0.2459) | 0.0000 | 0.1932 | (0.1896, 0.1986) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1619 | (0.0801, 0.2665) | 0.0000 | 0.1619 | (0.1032, 0.2557) | 0.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0772 | (0.0377, 0.1140) | 0.0000 | 0.0772 | (0.0435, 0.1070) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2545 | (0.1818, 0.3273) | 0.0000 | 0.2545 | (0.2462, 0.2670) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0501 | (0.0317, 0.0711) | 0.0000 | 0.0501 | (0.0355, 0.0597) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1940 | (0.0957, 0.3148) | 0.0000 | 0.1940 | (0.1290, 0.2917) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0333 | (0.0000, 0.0917) | 0.1287 | 0.0333 | (0.0000, 0.1176) | 0.3340 |
| controlled_alt_vs_proposed_raw | distinct1 | 0.0086 | (-0.0102, 0.0285) | 0.1920 | 0.0086 | (-0.0061, 0.0201) | 0.1433 |
| controlled_alt_vs_proposed_raw | length_score | 0.2900 | (0.1116, 0.4550) | 0.0013 | 0.2900 | (0.1611, 0.4140) | 0.0000 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1575 | (0.0875, 0.2275) | 0.0000 | 0.1575 | (0.0778, 0.2188) | 0.0003 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1656 | (0.1216, 0.2140) | 0.0000 | 0.1656 | (0.1427, 0.2033) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2161 | (0.1755, 0.2582) | 0.0000 | 0.2161 | (0.1940, 0.2401) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1963 | (0.1127, 0.3035) | 0.0000 | 0.1963 | (0.1273, 0.3063) | 0.0000 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0854 | (0.0519, 0.1177) | 0.0000 | 0.0854 | (0.0516, 0.1165) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.2818 | (0.2273, 0.3409) | 0.0000 | 0.2818 | (0.2511, 0.3160) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0627 | (0.0481, 0.0783) | 0.0000 | 0.0627 | (0.0458, 0.0742) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.2350 | (0.1369, 0.3512) | 0.0000 | 0.2350 | (0.1652, 0.3438) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0417 | (0.0000, 0.1000) | 0.0357 | 0.0417 | (0.0000, 0.1471) | 0.3393 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0134 | (-0.0043, 0.0319) | 0.0657 | 0.0134 | (0.0001, 0.0270) | 0.0243 |
| controlled_alt_vs_candidate_no_context | length_score | 0.3300 | (0.2000, 0.4567) | 0.0000 | 0.3300 | (0.2074, 0.4333) | 0.0000 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1400 | (0.0175, 0.2450) | 0.0130 | 0.1400 | (0.0175, 0.2528) | 0.0173 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1904 | (0.1552, 0.2316) | 0.0000 | 0.1904 | (0.1711, 0.2273) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 7 | 4 | 9 | 0.5750 | 0.6364 |
| proposed_vs_candidate_no_context | persona_consistency | 4 | 0 | 16 | 0.6000 | 1.0000 |
| proposed_vs_candidate_no_context | naturalness | 6 | 5 | 9 | 0.5250 | 0.5455 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 5 | 2 | 13 | 0.5750 | 0.7143 |
| proposed_vs_candidate_no_context | context_overlap | 8 | 3 | 9 | 0.6250 | 0.7273 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 4 | 0 | 16 | 0.6000 | 1.0000 |
| proposed_vs_candidate_no_context | persona_style | 1 | 0 | 19 | 0.5250 | 1.0000 |
| proposed_vs_candidate_no_context | distinct1 | 7 | 3 | 10 | 0.6000 | 0.7000 |
| proposed_vs_candidate_no_context | length_score | 7 | 4 | 9 | 0.5750 | 0.6364 |
| proposed_vs_candidate_no_context | sentence_score | 3 | 4 | 13 | 0.4750 | 0.4286 |
| proposed_vs_candidate_no_context | overall_quality | 9 | 2 | 9 | 0.6750 | 0.8182 |
| controlled_vs_proposed_raw | context_relevance | 17 | 3 | 0 | 0.8500 | 0.8500 |
| controlled_vs_proposed_raw | persona_consistency | 15 | 0 | 5 | 0.8750 | 1.0000 |
| controlled_vs_proposed_raw | naturalness | 16 | 4 | 0 | 0.8000 | 0.8000 |
| controlled_vs_proposed_raw | context_keyword_coverage | 17 | 2 | 1 | 0.8750 | 0.8947 |
| controlled_vs_proposed_raw | context_overlap | 17 | 3 | 0 | 0.8500 | 0.8500 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 14 | 0 | 6 | 0.8500 | 1.0000 |
| controlled_vs_proposed_raw | persona_style | 2 | 1 | 17 | 0.5250 | 0.6667 |
| controlled_vs_proposed_raw | distinct1 | 8 | 10 | 2 | 0.4500 | 0.4444 |
| controlled_vs_proposed_raw | length_score | 15 | 4 | 1 | 0.7750 | 0.7895 |
| controlled_vs_proposed_raw | sentence_score | 12 | 0 | 8 | 0.8000 | 1.0000 |
| controlled_vs_proposed_raw | overall_quality | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_vs_candidate_no_context | context_relevance | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_vs_candidate_no_context | persona_consistency | 17 | 0 | 3 | 0.9250 | 1.0000 |
| controlled_vs_candidate_no_context | naturalness | 18 | 2 | 0 | 0.9000 | 0.9000 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 18 | 0 | 2 | 0.9500 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 17 | 0 | 3 | 0.9250 | 1.0000 |
| controlled_vs_candidate_no_context | persona_style | 2 | 1 | 17 | 0.5250 | 0.6667 |
| controlled_vs_candidate_no_context | distinct1 | 11 | 8 | 1 | 0.5750 | 0.5789 |
| controlled_vs_candidate_no_context | length_score | 17 | 3 | 0 | 0.8500 | 0.8500 |
| controlled_vs_candidate_no_context | sentence_score | 12 | 1 | 7 | 0.7750 | 0.9231 |
| controlled_vs_candidate_no_context | overall_quality | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 8 | 10 | 2 | 0.4500 | 0.4444 |
| controlled_alt_vs_controlled_default | persona_consistency | 3 | 8 | 9 | 0.3750 | 0.2727 |
| controlled_alt_vs_controlled_default | naturalness | 9 | 9 | 2 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 6 | 8 | 6 | 0.4500 | 0.4286 |
| controlled_alt_vs_controlled_default | context_overlap | 7 | 11 | 2 | 0.4000 | 0.3889 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 3 | 7 | 10 | 0.4000 | 0.3000 |
| controlled_alt_vs_controlled_default | persona_style | 2 | 2 | 16 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | distinct1 | 10 | 7 | 3 | 0.5750 | 0.5882 |
| controlled_alt_vs_controlled_default | length_score | 7 | 9 | 4 | 0.4500 | 0.4375 |
| controlled_alt_vs_controlled_default | sentence_score | 2 | 5 | 13 | 0.4250 | 0.2857 |
| controlled_alt_vs_controlled_default | overall_quality | 7 | 11 | 2 | 0.4000 | 0.3889 |
| controlled_alt_vs_proposed_raw | context_relevance | 18 | 2 | 0 | 0.9000 | 0.9000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 13 | 0 | 7 | 0.8250 | 1.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 17 | 3 | 0 | 0.8500 | 0.8500 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 18 | 2 | 0 | 0.9000 | 0.9000 |
| controlled_alt_vs_proposed_raw | context_overlap | 18 | 2 | 0 | 0.9000 | 0.9000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 13 | 0 | 7 | 0.8250 | 1.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 2 | 0 | 18 | 0.5500 | 1.0000 |
| controlled_alt_vs_proposed_raw | distinct1 | 11 | 8 | 1 | 0.5750 | 0.5789 |
| controlled_alt_vs_proposed_raw | length_score | 16 | 4 | 0 | 0.8000 | 0.8000 |
| controlled_alt_vs_proposed_raw | sentence_score | 9 | 0 | 11 | 0.7250 | 1.0000 |
| controlled_alt_vs_proposed_raw | overall_quality | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 16 | 0 | 4 | 0.9000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | naturalness | 18 | 2 | 0 | 0.9000 | 0.9000 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 19 | 0 | 1 | 0.9750 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 16 | 0 | 4 | 0.9000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_style | 3 | 0 | 17 | 0.5750 | 1.0000 |
| controlled_alt_vs_candidate_no_context | distinct1 | 10 | 9 | 1 | 0.5250 | 0.5263 |
| controlled_alt_vs_candidate_no_context | length_score | 17 | 3 | 0 | 0.8500 | 0.8500 |
| controlled_alt_vs_candidate_no_context | sentence_score | 11 | 3 | 6 | 0.7000 | 0.7857 |
| controlled_alt_vs_candidate_no_context | overall_quality | 20 | 0 | 0 | 1.0000 | 1.0000 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.2500 | 0.4500 | 0.5500 |
| proposed_contextual_controlled_tuned | 0.0000 | 0.0000 | 0.1500 | 0.5500 | 0.4500 |
| proposed_contextual | 0.0000 | 0.0000 | 0.5500 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.6000 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `5`
- Unique template signatures: `17`
- Template signature ratio: `0.8500`
- Effective sample size by source clustering: `4.65`
- Effective sample size by template-signature clustering: `15.38`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: unavailable (disabled_by_flag).

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
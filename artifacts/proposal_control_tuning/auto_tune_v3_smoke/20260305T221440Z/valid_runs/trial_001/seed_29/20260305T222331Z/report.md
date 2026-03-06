# Proposal Alignment Evaluation Report

- Run ID: `20260305T222331Z`
- Generated: `2026-03-05T22:25:43.988674+00:00`
- Scenarios: `artifacts\proposal_control_tuning\auto_tune_v3_smoke\20260305T221440Z\valid_runs\trial_001\seed_29\20260305T222331Z\scenarios.jsonl`
- Scenario count: `12`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_tuned`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2448 (0.2097, 0.2827) | 0.4339 (0.3476, 0.5399) | 0.8895 (0.8505, 0.9253) | 0.4439 (0.4119, 0.4764) | n/a |
| proposed_contextual_controlled_tuned | 0.2105 (0.1429, 0.2909) | 0.4123 (0.3623, 0.4714) | 0.8874 (0.8491, 0.9224) | 0.4194 (0.3897, 0.4523) | n/a |
| proposed_contextual | 0.0701 (0.0233, 0.1298) | 0.2599 (0.1599, 0.3595) | 0.8611 (0.8159, 0.9047) | 0.2936 (0.2370, 0.3480) | n/a |
| candidate_no_context | 0.0254 (0.0123, 0.0486) | 0.3044 (0.1969, 0.4174) | 0.8803 (0.8354, 0.9228) | 0.2927 (0.2412, 0.3453) | n/a |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0447 | 1.7591 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0445 | -0.1461 |
| proposed_vs_candidate_no_context | naturalness | -0.0192 | -0.0218 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0645 | 4.2583 |
| proposed_vs_candidate_no_context | context_overlap | -0.0016 | -0.0320 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0536 | -0.2143 |
| proposed_vs_candidate_no_context | persona_style | -0.0081 | -0.0155 |
| proposed_vs_candidate_no_context | distinct1 | 0.0031 | 0.0033 |
| proposed_vs_candidate_no_context | length_score | -0.0583 | -0.1040 |
| proposed_vs_candidate_no_context | sentence_score | -0.0875 | -0.1024 |
| proposed_vs_candidate_no_context | overall_quality | 0.0009 | 0.0029 |
| controlled_vs_proposed_raw | context_relevance | 0.1747 | 2.4929 |
| controlled_vs_proposed_raw | persona_consistency | 0.1740 | 0.6693 |
| controlled_vs_proposed_raw | naturalness | 0.0284 | 0.0330 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2290 | 2.8748 |
| controlled_vs_proposed_raw | context_overlap | 0.0480 | 1.0060 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1865 | 0.9495 |
| controlled_vs_proposed_raw | persona_style | 0.1238 | 0.2409 |
| controlled_vs_proposed_raw | distinct1 | -0.0210 | -0.0219 |
| controlled_vs_proposed_raw | length_score | 0.1111 | 0.2210 |
| controlled_vs_proposed_raw | sentence_score | 0.1458 | 0.1902 |
| controlled_vs_proposed_raw | overall_quality | 0.1503 | 0.5119 |
| controlled_vs_candidate_no_context | context_relevance | 0.2194 | 8.6373 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1295 | 0.4254 |
| controlled_vs_candidate_no_context | naturalness | 0.0092 | 0.0105 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.2936 | 19.3750 |
| controlled_vs_candidate_no_context | context_overlap | 0.0465 | 0.9419 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1329 | 0.5317 |
| controlled_vs_candidate_no_context | persona_style | 0.1157 | 0.2217 |
| controlled_vs_candidate_no_context | distinct1 | -0.0179 | -0.0187 |
| controlled_vs_candidate_no_context | length_score | 0.0528 | 0.0941 |
| controlled_vs_candidate_no_context | sentence_score | 0.0583 | 0.0683 |
| controlled_vs_candidate_no_context | overall_quality | 0.1512 | 0.5164 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0343 | -0.1402 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0216 | -0.0498 |
| controlled_alt_vs_controlled_default | naturalness | -0.0021 | -0.0023 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0475 | -0.1538 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0036 | -0.0380 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0238 | -0.0622 |
| controlled_alt_vs_controlled_default | persona_style | -0.0128 | -0.0200 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0063 | 0.0067 |
| controlled_alt_vs_controlled_default | length_score | -0.0667 | -0.1086 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0875 | 0.0959 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0245 | -0.0551 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1404 | 2.0032 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1524 | 0.5862 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0263 | 0.0306 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.1816 | 2.2789 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0444 | 0.9297 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1627 | 0.8283 |
| controlled_alt_vs_proposed_raw | persona_style | 0.1111 | 0.2161 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0147 | -0.0154 |
| controlled_alt_vs_proposed_raw | length_score | 0.0444 | 0.0884 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.2333 | 0.3043 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1258 | 0.4285 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.1851 | 7.2861 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1079 | 0.3545 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0072 | 0.0081 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.2461 | 16.2417 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0428 | 0.8680 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1091 | 0.4365 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.1030 | 0.1973 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0116 | -0.0121 |
| controlled_alt_vs_candidate_no_context | length_score | -0.0139 | -0.0248 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1458 | 0.1707 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1267 | 0.4327 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0447 | (0.0020, 0.1024) | 0.0177 | 0.0447 | (0.0109, 0.0835) | 0.0000 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0445 | (-0.2071, 0.0861) | 0.7097 | -0.0445 | (-0.2229, 0.1323) | 0.7103 |
| proposed_vs_candidate_no_context | naturalness | -0.0192 | (-0.0714, 0.0327) | 0.7527 | -0.0192 | (-0.0842, 0.0707) | 0.7027 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0645 | (0.0083, 0.1389) | 0.0090 | 0.0645 | (0.0182, 0.1167) | 0.0000 |
| proposed_vs_candidate_no_context | context_overlap | -0.0016 | (-0.0251, 0.0222) | 0.5577 | -0.0016 | (-0.0097, 0.0062) | 0.7360 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0536 | (-0.2263, 0.0813) | 0.7300 | -0.0536 | (-0.2286, 0.0833) | 0.7003 |
| proposed_vs_candidate_no_context | persona_style | -0.0081 | (-0.1544, 0.1555) | 0.5470 | -0.0081 | (-0.2000, 0.3281) | 0.5793 |
| proposed_vs_candidate_no_context | distinct1 | 0.0031 | (-0.0247, 0.0304) | 0.4047 | 0.0031 | (-0.0284, 0.0350) | 0.3623 |
| proposed_vs_candidate_no_context | length_score | -0.0583 | (-0.2917, 0.1695) | 0.7010 | -0.0583 | (-0.3600, 0.2833) | 0.7100 |
| proposed_vs_candidate_no_context | sentence_score | -0.0875 | (-0.2333, 0.0583) | 0.9267 | -0.0875 | (-0.2100, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | overall_quality | 0.0009 | (-0.0741, 0.0709) | 0.4627 | 0.0009 | (-0.0920, 0.0744) | 0.3767 |
| controlled_vs_proposed_raw | context_relevance | 0.1747 | (0.1294, 0.2207) | 0.0000 | 0.1747 | (0.1517, 0.1810) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.1740 | (0.0313, 0.3258) | 0.0063 | 0.1740 | (0.1214, 0.2432) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0284 | (-0.0231, 0.0869) | 0.1500 | 0.0284 | (0.0031, 0.1340) | 0.0000 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2290 | (0.1692, 0.2891) | 0.0000 | 0.2290 | (0.2000, 0.2364) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0480 | (0.0300, 0.0671) | 0.0000 | 0.0480 | (0.0392, 0.0517) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1865 | (0.0158, 0.3770) | 0.0177 | 0.1865 | (0.0833, 0.3000) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.1238 | (0.0133, 0.2621) | 0.0113 | 0.1238 | (0.0159, 0.3281) | 0.0000 |
| controlled_vs_proposed_raw | distinct1 | -0.0210 | (-0.0457, 0.0046) | 0.9467 | -0.0210 | (-0.0372, -0.0022) | 1.0000 |
| controlled_vs_proposed_raw | length_score | 0.1111 | (-0.1028, 0.3528) | 0.1707 | 0.1111 | (0.0200, 0.5500) | 0.0000 |
| controlled_vs_proposed_raw | sentence_score | 0.1458 | (0.0292, 0.2625) | 0.0223 | 0.1458 | (0.0700, 0.3500) | 0.0000 |
| controlled_vs_proposed_raw | overall_quality | 0.1503 | (0.0919, 0.2132) | 0.0000 | 0.1503 | (0.1280, 0.1759) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2194 | (0.1819, 0.2619) | 0.0000 | 0.2194 | (0.1838, 0.2612) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1295 | (-0.0233, 0.2820) | 0.0490 | 0.1295 | (-0.1014, 0.3064) | 0.1520 |
| controlled_vs_candidate_no_context | naturalness | 0.0092 | (-0.0526, 0.0716) | 0.3870 | 0.0092 | (-0.0811, 0.2047) | 0.3690 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.2936 | (0.2441, 0.3504) | 0.0000 | 0.2936 | (0.2500, 0.3500) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0465 | (0.0334, 0.0602) | 0.0000 | 0.0465 | (0.0295, 0.0541) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1329 | (-0.0337, 0.3175) | 0.0620 | 0.1329 | (-0.1143, 0.3667) | 0.1487 |
| controlled_vs_candidate_no_context | persona_style | 0.1157 | (-0.0561, 0.2933) | 0.0963 | 0.1157 | (-0.0500, 0.6562) | 0.1467 |
| controlled_vs_candidate_no_context | distinct1 | -0.0179 | (-0.0349, -0.0023) | 0.9893 | -0.0179 | (-0.0306, 0.0076) | 0.9603 |
| controlled_vs_candidate_no_context | length_score | 0.0528 | (-0.2056, 0.3083) | 0.3507 | 0.0528 | (-0.3400, 0.8333) | 0.3670 |
| controlled_vs_candidate_no_context | sentence_score | 0.0583 | (-0.0875, 0.2042) | 0.2630 | 0.0583 | (-0.0700, 0.3500) | 0.2463 |
| controlled_vs_candidate_no_context | overall_quality | 0.1512 | (0.0887, 0.2172) | 0.0000 | 0.1512 | (0.0359, 0.2402) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0343 | (-0.1238, 0.0505) | 0.7847 | -0.0343 | (-0.0933, 0.1550) | 0.7303 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0216 | (-0.1401, 0.0816) | 0.6317 | -0.0216 | (-0.1561, 0.1043) | 0.6420 |
| controlled_alt_vs_controlled_default | naturalness | -0.0021 | (-0.0532, 0.0521) | 0.5453 | -0.0021 | (-0.0141, 0.0067) | 0.7067 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0475 | (-0.1591, 0.0691) | 0.7803 | -0.0475 | (-0.1273, 0.2000) | 0.7333 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0036 | (-0.0229, 0.0202) | 0.6513 | -0.0036 | (-0.0146, 0.0500) | 0.7330 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0238 | (-0.1726, 0.1151) | 0.6060 | -0.0238 | (-0.2000, 0.1429) | 0.6287 |
| controlled_alt_vs_controlled_default | persona_style | -0.0128 | (-0.0880, 0.0578) | 0.6707 | -0.0128 | (-0.0500, 0.0194) | 0.7483 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0063 | (-0.0171, 0.0300) | 0.2957 | 0.0063 | (-0.0037, 0.0167) | 0.1530 |
| controlled_alt_vs_controlled_default | length_score | -0.0667 | (-0.3167, 0.2083) | 0.6993 | -0.0667 | (-0.1333, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0875 | (0.0000, 0.1750) | 0.0293 | 0.0875 | (0.0000, 0.1400) | 0.0307 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0245 | (-0.0743, 0.0254) | 0.8313 | -0.0245 | (-0.0854, 0.0731) | 0.7330 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1404 | (0.0390, 0.2382) | 0.0023 | 0.1404 | (0.0876, 0.3067) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1524 | (0.0488, 0.2730) | 0.0003 | 0.1524 | (0.0871, 0.2257) | 0.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0263 | (-0.0274, 0.0876) | 0.1890 | 0.0263 | (-0.0027, 0.1407) | 0.0377 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.1816 | (0.0432, 0.3064) | 0.0067 | 0.1816 | (0.1091, 0.4000) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0444 | (0.0192, 0.0727) | 0.0000 | 0.0444 | (0.0333, 0.0892) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1627 | (0.0298, 0.3254) | 0.0037 | 0.1627 | (0.0833, 0.2571) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 0.1111 | (0.0208, 0.2351) | 0.0103 | 0.1111 | (0.0353, 0.3281) | 0.0000 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0147 | (-0.0393, 0.0074) | 0.8890 | -0.0147 | (-0.0251, -0.0059) | 1.0000 |
| controlled_alt_vs_proposed_raw | length_score | 0.0444 | (-0.1917, 0.3028) | 0.3843 | 0.0444 | (-0.1067, 0.5500) | 0.2973 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.2333 | (0.1458, 0.3208) | 0.0000 | 0.2333 | (0.2100, 0.3500) | 0.0000 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1258 | (0.0627, 0.1866) | 0.0000 | 0.1258 | (0.0905, 0.2153) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.1851 | (0.1134, 0.2711) | 0.0000 | 0.1851 | (0.0985, 0.3388) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1079 | (0.0482, 0.1661) | 0.0000 | 0.1079 | (0.0029, 0.2646) | 0.0000 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0072 | (-0.0628, 0.0828) | 0.4180 | 0.0072 | (-0.0746, 0.2114) | 0.3687 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.2461 | (0.1448, 0.3609) | 0.0000 | 0.2461 | (0.1273, 0.4500) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0428 | (0.0284, 0.0607) | 0.0000 | 0.0428 | (0.0314, 0.0795) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1091 | (0.0655, 0.1528) | 0.0000 | 0.1091 | (0.0286, 0.1667) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.1030 | (-0.0729, 0.2943) | 0.1303 | 0.1030 | (-0.1000, 0.6562) | 0.1550 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0116 | (-0.0382, 0.0156) | 0.7990 | -0.0116 | (-0.0343, 0.0242) | 0.7413 |
| controlled_alt_vs_candidate_no_context | length_score | -0.0139 | (-0.3334, 0.3000) | 0.5317 | -0.0139 | (-0.3667, 0.8333) | 0.6317 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1458 | (0.0583, 0.2333) | 0.0023 | 0.1458 | (0.0000, 0.3500) | 0.0337 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1267 | (0.0717, 0.1867) | 0.0000 | 0.1267 | (0.0333, 0.2896) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 6 | 5 | 1 | 0.5417 | 0.5455 |
| proposed_vs_candidate_no_context | persona_consistency | 5 | 3 | 4 | 0.5833 | 0.6250 |
| proposed_vs_candidate_no_context | naturalness | 6 | 5 | 1 | 0.5417 | 0.5455 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 4 | 0 | 8 | 0.6667 | 1.0000 |
| proposed_vs_candidate_no_context | context_overlap | 5 | 6 | 1 | 0.4583 | 0.4545 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 4 | 2 | 6 | 0.5833 | 0.6667 |
| proposed_vs_candidate_no_context | persona_style | 2 | 4 | 6 | 0.4167 | 0.3333 |
| proposed_vs_candidate_no_context | distinct1 | 6 | 4 | 2 | 0.5833 | 0.6000 |
| proposed_vs_candidate_no_context | length_score | 6 | 4 | 2 | 0.5833 | 0.6000 |
| proposed_vs_candidate_no_context | sentence_score | 2 | 5 | 5 | 0.3750 | 0.2857 |
| proposed_vs_candidate_no_context | overall_quality | 8 | 3 | 1 | 0.7083 | 0.7273 |
| controlled_vs_proposed_raw | context_relevance | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_proposed_raw | persona_consistency | 9 | 2 | 1 | 0.7917 | 0.8182 |
| controlled_vs_proposed_raw | naturalness | 6 | 6 | 0 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | context_keyword_coverage | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_proposed_raw | context_overlap | 11 | 1 | 0 | 0.9167 | 0.9167 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 7 | 2 | 3 | 0.7083 | 0.7778 |
| controlled_vs_proposed_raw | persona_style | 4 | 1 | 7 | 0.6250 | 0.8000 |
| controlled_vs_proposed_raw | distinct1 | 3 | 9 | 0 | 0.2500 | 0.2500 |
| controlled_vs_proposed_raw | length_score | 5 | 7 | 0 | 0.4167 | 0.4167 |
| controlled_vs_proposed_raw | sentence_score | 6 | 1 | 5 | 0.7083 | 0.8571 |
| controlled_vs_proposed_raw | overall_quality | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_relevance | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 9 | 2 | 1 | 0.7917 | 0.8182 |
| controlled_vs_candidate_no_context | naturalness | 5 | 7 | 0 | 0.4167 | 0.4167 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 8 | 2 | 2 | 0.7500 | 0.8000 |
| controlled_vs_candidate_no_context | persona_style | 5 | 2 | 5 | 0.6250 | 0.7143 |
| controlled_vs_candidate_no_context | distinct1 | 4 | 8 | 0 | 0.3333 | 0.3333 |
| controlled_vs_candidate_no_context | length_score | 6 | 6 | 0 | 0.5000 | 0.5000 |
| controlled_vs_candidate_no_context | sentence_score | 4 | 2 | 6 | 0.5833 | 0.6667 |
| controlled_vs_candidate_no_context | overall_quality | 11 | 1 | 0 | 0.9167 | 0.9167 |
| controlled_alt_vs_controlled_default | context_relevance | 4 | 7 | 1 | 0.3750 | 0.3636 |
| controlled_alt_vs_controlled_default | persona_consistency | 3 | 5 | 4 | 0.4167 | 0.3750 |
| controlled_alt_vs_controlled_default | naturalness | 5 | 6 | 1 | 0.4583 | 0.4545 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 3 | 5 | 4 | 0.4167 | 0.3750 |
| controlled_alt_vs_controlled_default | context_overlap | 4 | 7 | 1 | 0.3750 | 0.3636 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 2 | 3 | 7 | 0.4583 | 0.4000 |
| controlled_alt_vs_controlled_default | persona_style | 2 | 2 | 8 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | distinct1 | 6 | 5 | 1 | 0.5417 | 0.5455 |
| controlled_alt_vs_controlled_default | length_score | 4 | 6 | 2 | 0.4167 | 0.4000 |
| controlled_alt_vs_controlled_default | sentence_score | 3 | 0 | 9 | 0.6250 | 1.0000 |
| controlled_alt_vs_controlled_default | overall_quality | 4 | 7 | 1 | 0.3750 | 0.3636 |
| controlled_alt_vs_proposed_raw | context_relevance | 11 | 1 | 0 | 0.9167 | 0.9167 |
| controlled_alt_vs_proposed_raw | persona_consistency | 8 | 1 | 3 | 0.7917 | 0.8889 |
| controlled_alt_vs_proposed_raw | naturalness | 6 | 6 | 0 | 0.5000 | 0.5000 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 10 | 1 | 1 | 0.8750 | 0.9091 |
| controlled_alt_vs_proposed_raw | context_overlap | 10 | 2 | 0 | 0.8333 | 0.8333 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 6 | 1 | 5 | 0.7083 | 0.8571 |
| controlled_alt_vs_proposed_raw | persona_style | 4 | 0 | 8 | 0.6667 | 1.0000 |
| controlled_alt_vs_proposed_raw | distinct1 | 6 | 6 | 0 | 0.5000 | 0.5000 |
| controlled_alt_vs_proposed_raw | length_score | 6 | 5 | 1 | 0.5417 | 0.5455 |
| controlled_alt_vs_proposed_raw | sentence_score | 8 | 0 | 4 | 0.8333 | 1.0000 |
| controlled_alt_vs_proposed_raw | overall_quality | 10 | 2 | 0 | 0.8333 | 0.8333 |
| controlled_alt_vs_candidate_no_context | context_relevance | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 9 | 1 | 2 | 0.8333 | 0.9000 |
| controlled_alt_vs_candidate_no_context | naturalness | 7 | 5 | 0 | 0.5833 | 0.5833 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 11 | 0 | 1 | 0.9583 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 8 | 0 | 4 | 0.8333 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_style | 5 | 2 | 5 | 0.6250 | 0.7143 |
| controlled_alt_vs_candidate_no_context | distinct1 | 6 | 6 | 0 | 0.5000 | 0.5000 |
| controlled_alt_vs_candidate_no_context | length_score | 6 | 6 | 0 | 0.5000 | 0.5000 |
| controlled_alt_vs_candidate_no_context | sentence_score | 5 | 0 | 7 | 0.7083 | 1.0000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 12 | 0 | 0 | 1.0000 | 1.0000 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.4167 | 0.3333 | 0.6667 |
| proposed_contextual_controlled_tuned | 0.0000 | 0.0000 | 0.0833 | 0.8333 | 0.1667 |
| proposed_contextual | 0.0000 | 0.0000 | 0.2500 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.1667 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `3`
- Unique template signatures: `10`
- Template signature ratio: `0.8333`
- Effective sample size by source clustering: `2.67`
- Effective sample size by template-signature clustering: `8.00`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: unavailable (disabled_by_flag).

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
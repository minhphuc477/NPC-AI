# Proposal Alignment Evaluation Report

- Run ID: `20260305T221847Z`
- Generated: `2026-03-05T22:21:01.168122+00:00`
- Scenarios: `artifacts\proposal_control_tuning\auto_tune_v3_smoke\20260305T221440Z\train_runs\trial_002\seed_19\20260305T221847Z\scenarios.jsonl`
- Scenario count: `12`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_tuned`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2718 (0.2084, 0.3342) | 0.4030 (0.2890, 0.5439) | 0.8922 (0.8499, 0.9280) | 0.4449 (0.4044, 0.4842) | n/a |
| proposed_contextual_controlled_tuned | 0.2856 (0.2396, 0.3294) | 0.3346 (0.2667, 0.4098) | 0.9068 (0.8809, 0.9308) | 0.4287 (0.4085, 0.4481) | n/a |
| proposed_contextual | 0.0695 (0.0278, 0.1181) | 0.1762 (0.1278, 0.2290) | 0.8392 (0.7853, 0.9009) | 0.2550 (0.2226, 0.2876) | n/a |
| candidate_no_context | 0.0326 (0.0111, 0.0556) | 0.1333 (0.1083, 0.1583) | 0.8018 (0.7622, 0.8445) | 0.2145 (0.1985, 0.2333) | n/a |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0369 | 1.1291 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0429 | 0.3214 |
| proposed_vs_candidate_no_context | naturalness | 0.0374 | 0.0467 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0455 | 1.5000 |
| proposed_vs_candidate_no_context | context_overlap | 0.0168 | 0.4408 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0536 | nan |
| proposed_vs_candidate_no_context | persona_style | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | distinct1 | -0.0037 | -0.0039 |
| proposed_vs_candidate_no_context | length_score | 0.1361 | 0.4851 |
| proposed_vs_candidate_no_context | sentence_score | 0.1167 | 0.1718 |
| proposed_vs_candidate_no_context | overall_quality | 0.0405 | 0.1886 |
| controlled_vs_proposed_raw | context_relevance | 0.2023 | 2.9112 |
| controlled_vs_proposed_raw | persona_consistency | 0.2268 | 1.2873 |
| controlled_vs_proposed_raw | naturalness | 0.0529 | 0.0631 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2652 | 3.5000 |
| controlled_vs_proposed_raw | context_overlap | 0.0557 | 1.0149 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2869 | 5.3556 |
| controlled_vs_proposed_raw | persona_style | -0.0135 | -0.0203 |
| controlled_vs_proposed_raw | distinct1 | 0.0006 | 0.0007 |
| controlled_vs_proposed_raw | length_score | 0.2361 | 0.5667 |
| controlled_vs_proposed_raw | sentence_score | 0.0583 | 0.0733 |
| controlled_vs_proposed_raw | overall_quality | 0.1899 | 0.7447 |
| controlled_vs_candidate_no_context | context_relevance | 0.2392 | 7.3274 |
| controlled_vs_candidate_no_context | persona_consistency | 0.2697 | 2.0225 |
| controlled_vs_candidate_no_context | naturalness | 0.0904 | 0.1127 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3106 | 10.2500 |
| controlled_vs_candidate_no_context | context_overlap | 0.0725 | 1.9032 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.3405 | nan |
| controlled_vs_candidate_no_context | persona_style | -0.0135 | -0.0203 |
| controlled_vs_candidate_no_context | distinct1 | -0.0030 | -0.0032 |
| controlled_vs_candidate_no_context | length_score | 0.3722 | 1.3267 |
| controlled_vs_candidate_no_context | sentence_score | 0.1750 | 0.2577 |
| controlled_vs_candidate_no_context | overall_quality | 0.2304 | 1.0737 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0138 | 0.0507 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0684 | -0.1697 |
| controlled_alt_vs_controlled_default | naturalness | 0.0147 | 0.0164 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0227 | 0.0667 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0071 | -0.0643 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0889 | -0.2611 |
| controlled_alt_vs_controlled_default | persona_style | 0.0135 | 0.0207 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0121 | 0.0129 |
| controlled_alt_vs_controlled_default | length_score | 0.0472 | 0.0723 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0000 | 0.0000 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0161 | -0.0363 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.2161 | 3.1094 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1584 | 0.8991 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0676 | 0.0806 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2879 | 3.8000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0486 | 0.8854 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1980 | 3.6963 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0000 | 0.0000 |
| controlled_alt_vs_proposed_raw | distinct1 | 0.0128 | 0.0136 |
| controlled_alt_vs_proposed_raw | length_score | 0.2833 | 0.6800 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0583 | 0.0733 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1737 | 0.6814 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2530 | 7.7494 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.2013 | 1.5095 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.1050 | 0.1310 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3333 | 11.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0654 | 1.7164 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.2516 | nan |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0000 | 0.0000 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0091 | 0.0096 |
| controlled_alt_vs_candidate_no_context | length_score | 0.4194 | 1.4950 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1750 | 0.2577 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.2142 | 0.9985 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0369 | (0.0070, 0.0721) | 0.0067 | 0.0369 | (-0.0016, 0.0753) | 0.0597 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0429 | (0.0000, 0.0952) | 0.0340 | 0.0429 | (0.0000, 0.1011) | 0.0667 |
| proposed_vs_candidate_no_context | naturalness | 0.0374 | (0.0001, 0.0772) | 0.0250 | 0.0374 | (0.0081, 0.0584) | 0.0030 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0455 | (0.0076, 0.0909) | 0.0073 | 0.0455 | (0.0000, 0.0909) | 0.0550 |
| proposed_vs_candidate_no_context | context_overlap | 0.0168 | (-0.0034, 0.0421) | 0.0540 | 0.0168 | (-0.0052, 0.0511) | 0.1403 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0536 | (0.0000, 0.1190) | 0.0277 | 0.0536 | (0.0000, 0.1264) | 0.0587 |
| proposed_vs_candidate_no_context | persona_style | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | distinct1 | -0.0037 | (-0.0159, 0.0072) | 0.7263 | -0.0037 | (-0.0087, -0.0006) | 1.0000 |
| proposed_vs_candidate_no_context | length_score | 0.1361 | (-0.0389, 0.3278) | 0.0713 | 0.1361 | (0.0067, 0.2286) | 0.0017 |
| proposed_vs_candidate_no_context | sentence_score | 0.1167 | (0.0292, 0.2042) | 0.0087 | 0.1167 | (0.0389, 0.1909) | 0.0033 |
| proposed_vs_candidate_no_context | overall_quality | 0.0405 | (0.0177, 0.0661) | 0.0000 | 0.0405 | (0.0086, 0.0765) | 0.0053 |
| controlled_vs_proposed_raw | context_relevance | 0.2023 | (0.1385, 0.2683) | 0.0000 | 0.2023 | (0.1779, 0.2438) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.2268 | (0.0867, 0.3845) | 0.0000 | 0.2268 | (0.0667, 0.5202) | 0.0030 |
| controlled_vs_proposed_raw | naturalness | 0.0529 | (-0.0003, 0.1043) | 0.0267 | 0.0529 | (0.0077, 0.0859) | 0.0187 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2652 | (0.1742, 0.3485) | 0.0000 | 0.2652 | (0.2273, 0.3140) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0557 | (0.0281, 0.0813) | 0.0000 | 0.0557 | (0.0210, 0.0833) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2869 | (0.1143, 0.4830) | 0.0000 | 0.2869 | (0.0833, 0.6543) | 0.0053 |
| controlled_vs_proposed_raw | persona_style | -0.0135 | (-0.0406, 0.0000) | 1.0000 | -0.0135 | (-0.0325, 0.0000) | 1.0000 |
| controlled_vs_proposed_raw | distinct1 | 0.0006 | (-0.0160, 0.0178) | 0.4763 | 0.0006 | (-0.0031, 0.0059) | 0.3983 |
| controlled_vs_proposed_raw | length_score | 0.2361 | (0.0250, 0.4417) | 0.0147 | 0.2361 | (0.0800, 0.3576) | 0.0000 |
| controlled_vs_proposed_raw | sentence_score | 0.0583 | (-0.0875, 0.2042) | 0.2913 | 0.0583 | (-0.1458, 0.1750) | 0.2630 |
| controlled_vs_proposed_raw | overall_quality | 0.1899 | (0.1340, 0.2523) | 0.0000 | 0.1899 | (0.1366, 0.2895) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2392 | (0.1720, 0.3139) | 0.0000 | 0.2392 | (0.1764, 0.3020) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.2697 | (0.1501, 0.4071) | 0.0000 | 0.2697 | (0.1333, 0.5244) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.0904 | (0.0446, 0.1359) | 0.0000 | 0.0904 | (0.0449, 0.1237) | 0.0000 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3106 | (0.2197, 0.4015) | 0.0000 | 0.3106 | (0.2273, 0.3939) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0725 | (0.0506, 0.0947) | 0.0000 | 0.0725 | (0.0526, 0.0950) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.3405 | (0.1944, 0.5127) | 0.0000 | 0.3405 | (0.1667, 0.6556) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | -0.0135 | (-0.0406, 0.0000) | 1.0000 | -0.0135 | (-0.0325, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | distinct1 | -0.0030 | (-0.0248, 0.0183) | 0.6137 | -0.0030 | (-0.0113, 0.0054) | 0.7297 |
| controlled_vs_candidate_no_context | length_score | 0.3722 | (0.1917, 0.5501) | 0.0000 | 0.3722 | (0.2000, 0.5024) | 0.0000 |
| controlled_vs_candidate_no_context | sentence_score | 0.1750 | (0.0583, 0.2917) | 0.0083 | 0.1750 | (0.0583, 0.2500) | 0.0030 |
| controlled_vs_candidate_no_context | overall_quality | 0.2304 | (0.1818, 0.2788) | 0.0000 | 0.2304 | (0.2017, 0.3073) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0138 | (-0.0576, 0.0843) | 0.3383 | 0.0138 | (-0.0829, 0.0675) | 0.2617 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0684 | (-0.2254, 0.0702) | 0.8093 | -0.0684 | (-0.3437, 0.0640) | 0.6797 |
| controlled_alt_vs_controlled_default | naturalness | 0.0147 | (-0.0274, 0.0616) | 0.2687 | 0.0147 | (-0.0118, 0.0442) | 0.1973 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0227 | (-0.0682, 0.1136) | 0.3520 | 0.0227 | (-0.0985, 0.0909) | 0.2577 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0071 | (-0.0325, 0.0169) | 0.7093 | -0.0071 | (-0.0466, 0.0140) | 0.6873 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0889 | (-0.2818, 0.0833) | 0.8187 | -0.0889 | (-0.4296, 0.0769) | 0.7347 |
| controlled_alt_vs_controlled_default | persona_style | 0.0135 | (0.0000, 0.0406) | 0.3280 | 0.0135 | (0.0000, 0.0325) | 0.3240 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0121 | (-0.0161, 0.0437) | 0.2210 | 0.0121 | (-0.0142, 0.0384) | 0.1977 |
| controlled_alt_vs_controlled_default | length_score | 0.0472 | (-0.1306, 0.2556) | 0.3360 | 0.0472 | (0.0190, 0.1037) | 0.0000 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0000 | (-0.1458, 0.1458) | 0.5950 | 0.0000 | (-0.1591, 0.1167) | 0.5697 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0161 | (-0.0695, 0.0374) | 0.7237 | -0.0161 | (-0.1045, 0.0301) | 0.6740 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.2161 | (0.1503, 0.2690) | 0.0000 | 0.2161 | (0.1609, 0.2467) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1584 | (0.0886, 0.2238) | 0.0000 | 0.1584 | (0.1018, 0.2095) | 0.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0676 | (0.0138, 0.1220) | 0.0047 | 0.0676 | (0.0037, 0.1109) | 0.0067 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2879 | (0.2045, 0.3561) | 0.0000 | 0.2879 | (0.2197, 0.3287) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0486 | (0.0207, 0.0721) | 0.0003 | 0.0486 | (0.0215, 0.0757) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1980 | (0.1087, 0.2802) | 0.0003 | 0.1980 | (0.1273, 0.2619) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_proposed_raw | distinct1 | 0.0128 | (-0.0119, 0.0355) | 0.1413 | 0.0128 | (-0.0151, 0.0407) | 0.1813 |
| controlled_alt_vs_proposed_raw | length_score | 0.2833 | (0.0639, 0.4917) | 0.0030 | 0.2833 | (0.1487, 0.3872) | 0.0000 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0583 | (-0.0875, 0.2042) | 0.2837 | 0.0583 | (-0.1750, 0.2917) | 0.3220 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1737 | (0.1382, 0.2077) | 0.0000 | 0.1737 | (0.1132, 0.2131) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2530 | (0.2066, 0.2938) | 0.0000 | 0.2530 | (0.2163, 0.2981) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.2013 | (0.1521, 0.2594) | 0.0000 | 0.2013 | (0.1440, 0.2433) | 0.0000 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.1050 | (0.0651, 0.1443) | 0.0000 | 0.1050 | (0.0518, 0.1496) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3333 | (0.2727, 0.3864) | 0.0000 | 0.3333 | (0.2867, 0.3916) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0654 | (0.0478, 0.0808) | 0.0000 | 0.0654 | (0.0457, 0.0795) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.2516 | (0.1877, 0.3250) | 0.0000 | 0.2516 | (0.1800, 0.3041) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0091 | (-0.0137, 0.0337) | 0.2207 | 0.0091 | (-0.0242, 0.0404) | 0.3813 |
| controlled_alt_vs_candidate_no_context | length_score | 0.4194 | (0.2694, 0.5556) | 0.0000 | 0.4194 | (0.2867, 0.5452) | 0.0000 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1750 | (0.0292, 0.2917) | 0.0077 | 0.1750 | (0.0000, 0.3500) | 0.0597 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.2142 | (0.1960, 0.2327) | 0.0000 | 0.2142 | (0.1896, 0.2365) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 5 | 2 | 5 | 0.6250 | 0.7143 |
| proposed_vs_candidate_no_context | persona_consistency | 3 | 0 | 9 | 0.6250 | 1.0000 |
| proposed_vs_candidate_no_context | naturalness | 5 | 2 | 5 | 0.6250 | 0.7143 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 4 | 0 | 8 | 0.6667 | 1.0000 |
| proposed_vs_candidate_no_context | context_overlap | 4 | 3 | 5 | 0.5417 | 0.5714 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 3 | 0 | 9 | 0.6250 | 1.0000 |
| proposed_vs_candidate_no_context | persona_style | 0 | 0 | 12 | 0.5000 | nan |
| proposed_vs_candidate_no_context | distinct1 | 3 | 3 | 6 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | length_score | 5 | 2 | 5 | 0.6250 | 0.7143 |
| proposed_vs_candidate_no_context | sentence_score | 4 | 0 | 8 | 0.6667 | 1.0000 |
| proposed_vs_candidate_no_context | overall_quality | 7 | 0 | 5 | 0.7917 | 1.0000 |
| controlled_vs_proposed_raw | context_relevance | 11 | 1 | 0 | 0.9167 | 0.9167 |
| controlled_vs_proposed_raw | persona_consistency | 9 | 1 | 2 | 0.8333 | 0.9000 |
| controlled_vs_proposed_raw | naturalness | 8 | 4 | 0 | 0.6667 | 0.6667 |
| controlled_vs_proposed_raw | context_keyword_coverage | 11 | 0 | 1 | 0.9583 | 1.0000 |
| controlled_vs_proposed_raw | context_overlap | 11 | 1 | 0 | 0.9167 | 0.9167 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 9 | 1 | 2 | 0.8333 | 0.9000 |
| controlled_vs_proposed_raw | persona_style | 0 | 1 | 11 | 0.4583 | 0.0000 |
| controlled_vs_proposed_raw | distinct1 | 6 | 5 | 1 | 0.5417 | 0.5455 |
| controlled_vs_proposed_raw | length_score | 9 | 3 | 0 | 0.7500 | 0.7500 |
| controlled_vs_proposed_raw | sentence_score | 5 | 3 | 4 | 0.5833 | 0.6250 |
| controlled_vs_proposed_raw | overall_quality | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_relevance | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | naturalness | 9 | 3 | 0 | 0.7500 | 0.7500 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_style | 0 | 1 | 11 | 0.4583 | 0.0000 |
| controlled_vs_candidate_no_context | distinct1 | 5 | 6 | 1 | 0.4583 | 0.4545 |
| controlled_vs_candidate_no_context | length_score | 10 | 1 | 1 | 0.8750 | 0.9091 |
| controlled_vs_candidate_no_context | sentence_score | 7 | 1 | 4 | 0.7500 | 0.8750 |
| controlled_vs_candidate_no_context | overall_quality | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 6 | 5 | 1 | 0.5417 | 0.5455 |
| controlled_alt_vs_controlled_default | persona_consistency | 3 | 3 | 6 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | naturalness | 5 | 6 | 1 | 0.4583 | 0.4545 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 6 | 5 | 1 | 0.5417 | 0.5455 |
| controlled_alt_vs_controlled_default | context_overlap | 6 | 5 | 1 | 0.5417 | 0.5455 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 3 | 3 | 6 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | persona_style | 1 | 0 | 11 | 0.5417 | 1.0000 |
| controlled_alt_vs_controlled_default | distinct1 | 6 | 5 | 1 | 0.5417 | 0.5455 |
| controlled_alt_vs_controlled_default | length_score | 5 | 5 | 2 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | sentence_score | 3 | 3 | 6 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | overall_quality | 5 | 6 | 1 | 0.4583 | 0.4545 |
| controlled_alt_vs_proposed_raw | context_relevance | 11 | 1 | 0 | 0.9167 | 0.9167 |
| controlled_alt_vs_proposed_raw | persona_consistency | 11 | 1 | 0 | 0.9167 | 0.9167 |
| controlled_alt_vs_proposed_raw | naturalness | 9 | 3 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 11 | 1 | 0 | 0.9167 | 0.9167 |
| controlled_alt_vs_proposed_raw | context_overlap | 10 | 2 | 0 | 0.8333 | 0.8333 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 11 | 1 | 0 | 0.9167 | 0.9167 |
| controlled_alt_vs_proposed_raw | persona_style | 0 | 0 | 12 | 0.5000 | nan |
| controlled_alt_vs_proposed_raw | distinct1 | 9 | 3 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_proposed_raw | length_score | 8 | 3 | 1 | 0.7083 | 0.7273 |
| controlled_alt_vs_proposed_raw | sentence_score | 5 | 3 | 4 | 0.5833 | 0.6250 |
| controlled_alt_vs_proposed_raw | overall_quality | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | naturalness | 11 | 1 | 0 | 0.9167 | 0.9167 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_style | 0 | 0 | 12 | 0.5000 | nan |
| controlled_alt_vs_candidate_no_context | distinct1 | 6 | 4 | 2 | 0.5833 | 0.6000 |
| controlled_alt_vs_candidate_no_context | length_score | 11 | 0 | 1 | 0.9583 | 1.0000 |
| controlled_alt_vs_candidate_no_context | sentence_score | 7 | 1 | 4 | 0.7500 | 0.8750 |
| controlled_alt_vs_candidate_no_context | overall_quality | 12 | 0 | 0 | 1.0000 | 1.0000 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.2500 | 0.4167 | 0.5833 |
| proposed_contextual_controlled_tuned | 0.0000 | 0.0000 | 0.2500 | 0.4167 | 0.5833 |
| proposed_contextual | 0.0000 | 0.0000 | 0.4167 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.5833 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `4`
- Unique template signatures: `11`
- Template signature ratio: `0.9167`
- Effective sample size by source clustering: `3.79`
- Effective sample size by template-signature clustering: `10.29`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: unavailable (disabled_by_flag).

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
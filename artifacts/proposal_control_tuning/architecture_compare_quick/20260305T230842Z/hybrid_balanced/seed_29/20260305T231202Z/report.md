# Proposal Alignment Evaluation Report

- Run ID: `20260305T231202Z`
- Generated: `2026-03-05T23:14:40.796436+00:00`
- Scenarios: `artifacts\proposal_control_tuning\architecture_compare_quick\20260305T230842Z\hybrid_balanced\seed_29\20260305T231202Z\scenarios.jsonl`
- Scenario count: `12`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_alt`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off
- `baseline_no_context`: model `phi3:mini`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2637 (0.2178, 0.3160) | 0.2895 (0.2096, 0.3799) | 0.9100 (0.8811, 0.9334) | 0.4030 (0.3691, 0.4346) | n/a |
| proposed_contextual_controlled_alt | 0.2776 (0.2048, 0.3441) | 0.2669 (0.2172, 0.3154) | 0.8772 (0.8423, 0.9051) | 0.3947 (0.3576, 0.4317) | n/a |
| proposed_contextual | 0.0745 (0.0287, 0.1364) | 0.1417 (0.0826, 0.2078) | 0.8495 (0.7969, 0.9035) | 0.2490 (0.2009, 0.3047) | n/a |
| candidate_no_context | 0.0182 (0.0114, 0.0295) | 0.1380 (0.0853, 0.1989) | 0.8322 (0.7820, 0.8818) | 0.2175 (0.1898, 0.2476) | n/a |
| baseline_no_context | 0.0353 (0.0163, 0.0611) | 0.1328 (0.0917, 0.1717) | 0.9029 (0.8731, 0.9322) | 0.2365 (0.2134, 0.2595) | n/a |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0562 | 3.0837 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0037 | 0.0270 |
| proposed_vs_candidate_no_context | naturalness | 0.0173 | 0.0208 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0745 | 9.8333 |
| proposed_vs_candidate_no_context | context_overlap | 0.0136 | 0.3157 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0119 | 0.2857 |
| proposed_vs_candidate_no_context | persona_style | -0.0290 | -0.0554 |
| proposed_vs_candidate_no_context | distinct1 | 0.0121 | 0.0128 |
| proposed_vs_candidate_no_context | length_score | 0.0333 | 0.0816 |
| proposed_vs_candidate_no_context | sentence_score | 0.0583 | 0.0791 |
| proposed_vs_candidate_no_context | overall_quality | 0.0315 | 0.1448 |
| proposed_vs_baseline_no_context | context_relevance | 0.0391 | 1.1071 |
| proposed_vs_baseline_no_context | persona_consistency | 0.0089 | 0.0670 |
| proposed_vs_baseline_no_context | naturalness | -0.0533 | -0.0591 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0530 | 1.8261 |
| proposed_vs_baseline_no_context | context_overlap | 0.0067 | 0.1334 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 0.0369 | 2.2143 |
| proposed_vs_baseline_no_context | persona_style | -0.1031 | -0.1727 |
| proposed_vs_baseline_no_context | distinct1 | -0.0285 | -0.0290 |
| proposed_vs_baseline_no_context | length_score | -0.1806 | -0.2902 |
| proposed_vs_baseline_no_context | sentence_score | -0.0583 | -0.0683 |
| proposed_vs_baseline_no_context | overall_quality | 0.0125 | 0.0530 |
| controlled_vs_proposed_raw | context_relevance | 0.1893 | 2.5419 |
| controlled_vs_proposed_raw | persona_consistency | 0.1478 | 1.0431 |
| controlled_vs_proposed_raw | naturalness | 0.0605 | 0.0712 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2538 | 3.0923 |
| controlled_vs_proposed_raw | context_overlap | 0.0388 | 0.6836 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1524 | 2.8444 |
| controlled_vs_proposed_raw | persona_style | 0.1294 | 0.2618 |
| controlled_vs_proposed_raw | distinct1 | -0.0182 | -0.0191 |
| controlled_vs_proposed_raw | length_score | 0.2806 | 0.6352 |
| controlled_vs_proposed_raw | sentence_score | 0.1167 | 0.1466 |
| controlled_vs_proposed_raw | overall_quality | 0.1540 | 0.6185 |
| controlled_vs_candidate_no_context | context_relevance | 0.2455 | 13.4642 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1515 | 1.0982 |
| controlled_vs_candidate_no_context | naturalness | 0.0778 | 0.0935 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3283 | 43.3333 |
| controlled_vs_candidate_no_context | context_overlap | 0.0524 | 1.2151 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1643 | 3.9429 |
| controlled_vs_candidate_no_context | persona_style | 0.1004 | 0.1919 |
| controlled_vs_candidate_no_context | distinct1 | -0.0062 | -0.0066 |
| controlled_vs_candidate_no_context | length_score | 0.3139 | 0.7687 |
| controlled_vs_candidate_no_context | sentence_score | 0.1750 | 0.2373 |
| controlled_vs_candidate_no_context | overall_quality | 0.1855 | 0.8529 |
| controlled_vs_baseline_no_context | context_relevance | 0.2284 | 6.4632 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1567 | 1.1800 |
| controlled_vs_baseline_no_context | naturalness | 0.0071 | 0.0079 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3068 | 10.5652 |
| controlled_vs_baseline_no_context | context_overlap | 0.0454 | 0.9082 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1893 | 11.3571 |
| controlled_vs_baseline_no_context | persona_style | 0.0262 | 0.0439 |
| controlled_vs_baseline_no_context | distinct1 | -0.0468 | -0.0476 |
| controlled_vs_baseline_no_context | length_score | 0.1000 | 0.1607 |
| controlled_vs_baseline_no_context | sentence_score | 0.0583 | 0.0683 |
| controlled_vs_baseline_no_context | overall_quality | 0.1666 | 0.7044 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0139 | 0.0527 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0225 | -0.0778 |
| controlled_alt_vs_controlled_default | naturalness | -0.0329 | -0.0361 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0139 | 0.0414 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0139 | 0.1455 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0183 | -0.0886 |
| controlled_alt_vs_controlled_default | persona_style | -0.0396 | -0.0635 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0099 | 0.0106 |
| controlled_alt_vs_controlled_default | length_score | -0.1694 | -0.2346 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0292 | -0.0320 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0083 | -0.0206 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.2032 | 2.7285 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1253 | 0.8841 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0276 | 0.0325 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2677 | 3.2615 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0527 | 0.9285 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1341 | 2.5037 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0898 | 0.1817 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0084 | -0.0088 |
| controlled_alt_vs_proposed_raw | length_score | 0.1111 | 0.2516 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0875 | 0.1099 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1457 | 0.5852 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2594 | 14.2260 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1290 | 0.9350 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0450 | 0.0540 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3422 | 45.1667 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0663 | 1.5374 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1460 | 3.5048 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0608 | 0.1162 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0037 | 0.0039 |
| controlled_alt_vs_candidate_no_context | length_score | 0.1444 | 0.3537 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1458 | 0.1977 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1772 | 0.8147 |
| controlled_alt_vs_baseline_no_context | context_relevance | 0.2423 | 6.8563 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 0.1342 | 1.0104 |
| controlled_alt_vs_baseline_no_context | naturalness | -0.0257 | -0.0285 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 0.3207 | 11.0435 |
| controlled_alt_vs_baseline_no_context | context_overlap | 0.0593 | 1.1858 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 0.1710 | 10.2619 |
| controlled_alt_vs_baseline_no_context | persona_style | -0.0134 | -0.0224 |
| controlled_alt_vs_baseline_no_context | distinct1 | -0.0369 | -0.0375 |
| controlled_alt_vs_baseline_no_context | length_score | -0.0694 | -0.1116 |
| controlled_alt_vs_baseline_no_context | sentence_score | 0.0292 | 0.0341 |
| controlled_alt_vs_baseline_no_context | overall_quality | 0.1582 | 0.6692 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2284 | 6.4632 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1567 | 1.1800 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 0.0071 | 0.0079 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3068 | 10.5652 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0454 | 0.9082 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1893 | 11.3571 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0262 | 0.0439 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0468 | -0.0476 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.1000 | 0.1607 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0583 | 0.0683 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1666 | 0.7044 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0562 | (0.0164, 0.1064) | 0.0000 | 0.0562 | (0.0109, 0.1139) | 0.0180 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0037 | (-0.0587, 0.0561) | 0.4257 | 0.0037 | (-0.0306, 0.0450) | 0.4320 |
| proposed_vs_candidate_no_context | naturalness | 0.0173 | (-0.0037, 0.0394) | 0.0533 | 0.0173 | (-0.0081, 0.0448) | 0.0963 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0745 | (0.0208, 0.1433) | 0.0017 | 0.0745 | (0.0128, 0.1538) | 0.0213 |
| proposed_vs_candidate_no_context | context_overlap | 0.0136 | (0.0025, 0.0270) | 0.0033 | 0.0136 | (0.0030, 0.0248) | 0.0223 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0119 | (-0.0694, 0.0774) | 0.3980 | 0.0119 | (-0.0303, 0.0595) | 0.3823 |
| proposed_vs_candidate_no_context | persona_style | -0.0290 | (-0.1037, 0.0167) | 0.7443 | -0.0290 | (-0.0870, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | distinct1 | 0.0121 | (-0.0064, 0.0326) | 0.1027 | 0.0121 | (-0.0021, 0.0320) | 0.0533 |
| proposed_vs_candidate_no_context | length_score | 0.0333 | (-0.0222, 0.0944) | 0.1370 | 0.0333 | (-0.0367, 0.1033) | 0.1830 |
| proposed_vs_candidate_no_context | sentence_score | 0.0583 | (-0.0583, 0.1750) | 0.2277 | 0.0583 | (-0.0636, 0.2042) | 0.2733 |
| proposed_vs_candidate_no_context | overall_quality | 0.0315 | (0.0020, 0.0681) | 0.0177 | 0.0315 | (0.0101, 0.0529) | 0.0010 |
| proposed_vs_baseline_no_context | context_relevance | 0.0391 | (-0.0159, 0.1094) | 0.1000 | 0.0391 | (-0.0248, 0.1177) | 0.1907 |
| proposed_vs_baseline_no_context | persona_consistency | 0.0089 | (-0.0490, 0.0780) | 0.4020 | 0.0089 | (-0.0509, 0.0716) | 0.3993 |
| proposed_vs_baseline_no_context | naturalness | -0.0533 | (-0.1073, 0.0019) | 0.9693 | -0.0533 | (-0.1187, 0.0169) | 0.9317 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0530 | (-0.0227, 0.1509) | 0.1093 | 0.0530 | (-0.0364, 0.1591) | 0.2047 |
| proposed_vs_baseline_no_context | context_overlap | 0.0067 | (-0.0119, 0.0280) | 0.2663 | 0.0067 | (-0.0109, 0.0258) | 0.2563 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 0.0369 | (-0.0262, 0.1091) | 0.1507 | 0.0369 | (-0.0257, 0.1091) | 0.1383 |
| proposed_vs_baseline_no_context | persona_style | -0.1031 | (-0.2326, 0.0000) | 1.0000 | -0.1031 | (-0.2461, -0.0128) | 1.0000 |
| proposed_vs_baseline_no_context | distinct1 | -0.0285 | (-0.0578, -0.0003) | 0.9757 | -0.0285 | (-0.0661, 0.0093) | 0.9227 |
| proposed_vs_baseline_no_context | length_score | -0.1806 | (-0.3612, 0.0417) | 0.9513 | -0.1806 | (-0.4100, 0.0643) | 0.9253 |
| proposed_vs_baseline_no_context | sentence_score | -0.0583 | (-0.2042, 0.0875) | 0.8500 | -0.0583 | (-0.2154, 0.0955) | 0.8133 |
| proposed_vs_baseline_no_context | overall_quality | 0.0125 | (-0.0365, 0.0711) | 0.3677 | 0.0125 | (-0.0452, 0.0764) | 0.3647 |
| controlled_vs_proposed_raw | context_relevance | 0.1893 | (0.1494, 0.2385) | 0.0000 | 0.1893 | (0.1607, 0.2137) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.1478 | (0.0636, 0.2162) | 0.0000 | 0.1478 | (0.0610, 0.2365) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0605 | (0.0006, 0.1230) | 0.0233 | 0.0605 | (-0.0061, 0.1350) | 0.0393 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2538 | (0.2008, 0.3163) | 0.0000 | 0.2538 | (0.2182, 0.2857) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0388 | (0.0209, 0.0580) | 0.0000 | 0.0388 | (0.0242, 0.0540) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1524 | (0.0635, 0.2349) | 0.0000 | 0.1524 | (0.0643, 0.2491) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.1294 | (0.0021, 0.2689) | 0.0247 | 0.1294 | (0.0164, 0.2855) | 0.0193 |
| controlled_vs_proposed_raw | distinct1 | -0.0182 | (-0.0435, 0.0061) | 0.9243 | -0.0182 | (-0.0512, 0.0101) | 0.8823 |
| controlled_vs_proposed_raw | length_score | 0.2806 | (0.0472, 0.5028) | 0.0117 | 0.2806 | (0.0205, 0.5600) | 0.0153 |
| controlled_vs_proposed_raw | sentence_score | 0.1167 | (0.0000, 0.2333) | 0.0517 | 0.1167 | (-0.0292, 0.2545) | 0.0763 |
| controlled_vs_proposed_raw | overall_quality | 0.1540 | (0.1156, 0.1848) | 0.0000 | 0.1540 | (0.1142, 0.1923) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2455 | (0.2032, 0.2949) | 0.0000 | 0.2455 | (0.1981, 0.2877) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1515 | (0.0582, 0.2455) | 0.0003 | 0.1515 | (0.0436, 0.2704) | 0.0037 |
| controlled_vs_candidate_no_context | naturalness | 0.0778 | (0.0191, 0.1333) | 0.0027 | 0.0778 | (0.0163, 0.1451) | 0.0040 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3283 | (0.2727, 0.3914) | 0.0000 | 0.3283 | (0.2652, 0.3884) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0524 | (0.0325, 0.0741) | 0.0000 | 0.0524 | (0.0332, 0.0668) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1643 | (0.0583, 0.2817) | 0.0003 | 0.1643 | (0.0417, 0.3011) | 0.0050 |
| controlled_vs_candidate_no_context | persona_style | 0.1004 | (-0.0099, 0.2384) | 0.0487 | 0.1004 | (-0.0182, 0.2635) | 0.0980 |
| controlled_vs_candidate_no_context | distinct1 | -0.0062 | (-0.0314, 0.0175) | 0.6833 | -0.0062 | (-0.0325, 0.0201) | 0.6453 |
| controlled_vs_candidate_no_context | length_score | 0.3139 | (0.0972, 0.5389) | 0.0053 | 0.3139 | (0.0939, 0.5667) | 0.0020 |
| controlled_vs_candidate_no_context | sentence_score | 0.1750 | (0.0875, 0.2625) | 0.0007 | 0.1750 | (0.0750, 0.2864) | 0.0003 |
| controlled_vs_candidate_no_context | overall_quality | 0.1855 | (0.1538, 0.2164) | 0.0000 | 0.1855 | (0.1506, 0.2290) | 0.0000 |
| controlled_vs_baseline_no_context | context_relevance | 0.2284 | (0.1706, 0.2946) | 0.0000 | 0.2284 | (0.1638, 0.2882) | 0.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1567 | (0.0706, 0.2445) | 0.0003 | 0.1567 | (0.0569, 0.2565) | 0.0007 |
| controlled_vs_baseline_no_context | naturalness | 0.0071 | (-0.0294, 0.0427) | 0.3333 | 0.0071 | (-0.0267, 0.0489) | 0.3477 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3068 | (0.2273, 0.3914) | 0.0000 | 0.3068 | (0.2149, 0.3896) | 0.0000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0454 | (0.0296, 0.0627) | 0.0000 | 0.0454 | (0.0287, 0.0610) | 0.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1893 | (0.0833, 0.3004) | 0.0000 | 0.1893 | (0.0619, 0.3171) | 0.0010 |
| controlled_vs_baseline_no_context | persona_style | 0.0262 | (-0.0463, 0.1250) | 0.3740 | 0.0262 | (-0.0427, 0.1250) | 0.3673 |
| controlled_vs_baseline_no_context | distinct1 | -0.0468 | (-0.0678, -0.0250) | 1.0000 | -0.0468 | (-0.0705, -0.0214) | 1.0000 |
| controlled_vs_baseline_no_context | length_score | 0.1000 | (-0.0556, 0.2417) | 0.0967 | 0.1000 | (-0.0667, 0.2667) | 0.1097 |
| controlled_vs_baseline_no_context | sentence_score | 0.0583 | (-0.0583, 0.1750) | 0.2193 | 0.0583 | (-0.0538, 0.1750) | 0.2053 |
| controlled_vs_baseline_no_context | overall_quality | 0.1666 | (0.1241, 0.2088) | 0.0000 | 0.1666 | (0.1156, 0.2165) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0139 | (-0.0647, 0.0948) | 0.3667 | 0.0139 | (-0.0639, 0.1074) | 0.3657 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0225 | (-0.1055, 0.0609) | 0.6903 | -0.0225 | (-0.1095, 0.0586) | 0.6803 |
| controlled_alt_vs_controlled_default | naturalness | -0.0329 | (-0.0661, -0.0049) | 0.9940 | -0.0329 | (-0.0725, -0.0067) | 0.9973 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0139 | (-0.0985, 0.1326) | 0.4207 | 0.0139 | (-0.0909, 0.1439) | 0.3927 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0139 | (-0.0138, 0.0405) | 0.1550 | 0.0139 | (-0.0029, 0.0293) | 0.0550 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0183 | (-0.1071, 0.0762) | 0.6530 | -0.0183 | (-0.1190, 0.0762) | 0.6250 |
| controlled_alt_vs_controlled_default | persona_style | -0.0396 | (-0.1224, 0.0309) | 0.8523 | -0.0396 | (-0.1257, 0.0256) | 0.8540 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0099 | (-0.0232, 0.0422) | 0.2837 | 0.0099 | (-0.0215, 0.0382) | 0.2723 |
| controlled_alt_vs_controlled_default | length_score | -0.1694 | (-0.3306, -0.0278) | 0.9930 | -0.1694 | (-0.3601, -0.0333) | 0.9940 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0292 | (-0.0875, 0.0000) | 1.0000 | -0.0292 | (-0.0875, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0083 | (-0.0534, 0.0344) | 0.6257 | -0.0083 | (-0.0603, 0.0331) | 0.6103 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.2032 | (0.1115, 0.2859) | 0.0000 | 0.2032 | (0.1372, 0.2876) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1253 | (0.0538, 0.1926) | 0.0013 | 0.1253 | (0.0647, 0.1832) | 0.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0276 | (-0.0223, 0.0778) | 0.1423 | 0.0276 | (-0.0317, 0.0853) | 0.1883 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2677 | (0.1389, 0.3788) | 0.0000 | 0.2677 | (0.1775, 0.3939) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0527 | (0.0248, 0.0799) | 0.0003 | 0.0527 | (0.0329, 0.0732) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1341 | (0.0583, 0.2111) | 0.0010 | 0.1341 | (0.0730, 0.1849) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0898 | (-0.0289, 0.2341) | 0.0790 | 0.0898 | (-0.0244, 0.2444) | 0.0757 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0084 | (-0.0409, 0.0248) | 0.6817 | -0.0084 | (-0.0438, 0.0295) | 0.6843 |
| controlled_alt_vs_proposed_raw | length_score | 0.1111 | (-0.0639, 0.2833) | 0.1040 | 0.1111 | (-0.0694, 0.2744) | 0.1150 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0875 | (-0.0292, 0.2042) | 0.1117 | 0.0875 | (-0.0318, 0.2250) | 0.1463 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1457 | (0.0801, 0.2001) | 0.0000 | 0.1457 | (0.0904, 0.2019) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2594 | (0.1878, 0.3255) | 0.0000 | 0.2594 | (0.1998, 0.3287) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1290 | (0.0601, 0.1934) | 0.0003 | 0.1290 | (0.0536, 0.1914) | 0.0007 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0450 | (-0.0007, 0.0885) | 0.0277 | 0.0450 | (0.0021, 0.0919) | 0.0163 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3422 | (0.2462, 0.4343) | 0.0000 | 0.3422 | (0.2599, 0.4394) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0663 | (0.0435, 0.0905) | 0.0000 | 0.0663 | (0.0487, 0.0816) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1460 | (0.0639, 0.2210) | 0.0003 | 0.1460 | (0.0564, 0.2126) | 0.0013 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0608 | (-0.0439, 0.1896) | 0.1667 | 0.0608 | (-0.0479, 0.2129) | 0.1440 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0037 | (-0.0334, 0.0380) | 0.3863 | 0.0037 | (-0.0256, 0.0378) | 0.4007 |
| controlled_alt_vs_candidate_no_context | length_score | 0.1444 | (-0.0028, 0.3056) | 0.0277 | 0.1444 | (0.0102, 0.2744) | 0.0173 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1458 | (0.0000, 0.2625) | 0.0257 | 0.1458 | (0.0000, 0.2864) | 0.0323 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1772 | (0.1407, 0.2142) | 0.0000 | 0.1772 | (0.1416, 0.2201) | 0.0000 |
| controlled_alt_vs_baseline_no_context | context_relevance | 0.2423 | (0.1807, 0.2975) | 0.0000 | 0.2423 | (0.1906, 0.3038) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 0.1342 | (0.0709, 0.1869) | 0.0000 | 0.1342 | (0.1102, 0.1579) | 0.0000 |
| controlled_alt_vs_baseline_no_context | naturalness | -0.0257 | (-0.0557, 0.0049) | 0.9490 | -0.0257 | (-0.0553, 0.0077) | 0.9353 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 0.3207 | (0.2374, 0.3965) | 0.0000 | 0.3207 | (0.2500, 0.4007) | 0.0000 |
| controlled_alt_vs_baseline_no_context | context_overlap | 0.0593 | (0.0437, 0.0762) | 0.0000 | 0.0593 | (0.0530, 0.0672) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 0.1710 | (0.0933, 0.2353) | 0.0000 | 0.1710 | (0.1410, 0.2052) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_style | -0.0134 | (-0.0773, 0.0394) | 0.6650 | -0.0134 | (-0.0843, 0.0338) | 0.6670 |
| controlled_alt_vs_baseline_no_context | distinct1 | -0.0369 | (-0.0690, -0.0081) | 0.9973 | -0.0369 | (-0.0632, -0.0093) | 0.9960 |
| controlled_alt_vs_baseline_no_context | length_score | -0.0694 | (-0.2167, 0.0833) | 0.8193 | -0.0694 | (-0.2167, 0.0897) | 0.8373 |
| controlled_alt_vs_baseline_no_context | sentence_score | 0.0292 | (-0.0875, 0.1458) | 0.4097 | 0.0292 | (-0.0875, 0.1591) | 0.3963 |
| controlled_alt_vs_baseline_no_context | overall_quality | 0.1582 | (0.1259, 0.1878) | 0.0000 | 0.1582 | (0.1315, 0.1886) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2284 | (0.1723, 0.2958) | 0.0000 | 0.2284 | (0.1623, 0.2928) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1567 | (0.0697, 0.2487) | 0.0003 | 0.1567 | (0.0529, 0.2650) | 0.0003 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 0.0071 | (-0.0305, 0.0416) | 0.3590 | 0.0071 | (-0.0313, 0.0463) | 0.3403 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3068 | (0.2336, 0.3927) | 0.0000 | 0.3068 | (0.2231, 0.3916) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0454 | (0.0297, 0.0630) | 0.0000 | 0.0454 | (0.0284, 0.0610) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1893 | (0.0782, 0.2929) | 0.0007 | 0.1893 | (0.0619, 0.3195) | 0.0017 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0262 | (-0.0463, 0.1250) | 0.3863 | 0.0262 | (-0.0427, 0.1155) | 0.3730 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0468 | (-0.0686, -0.0257) | 1.0000 | -0.0468 | (-0.0696, -0.0214) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.1000 | (-0.0473, 0.2417) | 0.0937 | 0.1000 | (-0.0583, 0.2722) | 0.1123 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0583 | (-0.0583, 0.1750) | 0.2117 | 0.0583 | (-0.0538, 0.1750) | 0.2083 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1666 | (0.1240, 0.2070) | 0.0000 | 0.1666 | (0.1163, 0.2173) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 6 | 1 | 5 | 0.7083 | 0.8571 |
| proposed_vs_candidate_no_context | persona_consistency | 4 | 2 | 6 | 0.5833 | 0.6667 |
| proposed_vs_candidate_no_context | naturalness | 5 | 2 | 5 | 0.6250 | 0.7143 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 5 | 0 | 7 | 0.7083 | 1.0000 |
| proposed_vs_candidate_no_context | context_overlap | 6 | 1 | 5 | 0.7083 | 0.8571 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 3 | 1 | 8 | 0.5833 | 0.7500 |
| proposed_vs_candidate_no_context | persona_style | 1 | 1 | 10 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | distinct1 | 4 | 2 | 6 | 0.5833 | 0.6667 |
| proposed_vs_candidate_no_context | length_score | 4 | 2 | 6 | 0.5833 | 0.6667 |
| proposed_vs_candidate_no_context | sentence_score | 3 | 1 | 8 | 0.5833 | 0.7500 |
| proposed_vs_candidate_no_context | overall_quality | 6 | 1 | 5 | 0.7083 | 0.8571 |
| proposed_vs_baseline_no_context | context_relevance | 4 | 8 | 0 | 0.3333 | 0.3333 |
| proposed_vs_baseline_no_context | persona_consistency | 3 | 4 | 5 | 0.4583 | 0.4286 |
| proposed_vs_baseline_no_context | naturalness | 3 | 9 | 0 | 0.2500 | 0.2500 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 4 | 3 | 5 | 0.5417 | 0.5714 |
| proposed_vs_baseline_no_context | context_overlap | 5 | 7 | 0 | 0.4167 | 0.4167 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 3 | 1 | 8 | 0.5833 | 0.7500 |
| proposed_vs_baseline_no_context | persona_style | 0 | 3 | 9 | 0.3750 | 0.0000 |
| proposed_vs_baseline_no_context | distinct1 | 3 | 7 | 2 | 0.3333 | 0.3000 |
| proposed_vs_baseline_no_context | length_score | 4 | 8 | 0 | 0.3333 | 0.3333 |
| proposed_vs_baseline_no_context | sentence_score | 2 | 4 | 6 | 0.4167 | 0.3333 |
| proposed_vs_baseline_no_context | overall_quality | 6 | 6 | 0 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | context_relevance | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_proposed_raw | persona_consistency | 9 | 1 | 2 | 0.8333 | 0.9000 |
| controlled_vs_proposed_raw | naturalness | 8 | 4 | 0 | 0.6667 | 0.6667 |
| controlled_vs_proposed_raw | context_keyword_coverage | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_proposed_raw | context_overlap | 11 | 1 | 0 | 0.9167 | 0.9167 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 9 | 1 | 2 | 0.8333 | 0.9000 |
| controlled_vs_proposed_raw | persona_style | 4 | 1 | 7 | 0.6250 | 0.8000 |
| controlled_vs_proposed_raw | distinct1 | 2 | 9 | 1 | 0.2083 | 0.1818 |
| controlled_vs_proposed_raw | length_score | 8 | 4 | 0 | 0.6667 | 0.6667 |
| controlled_vs_proposed_raw | sentence_score | 5 | 1 | 6 | 0.6667 | 0.8333 |
| controlled_vs_proposed_raw | overall_quality | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_relevance | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 8 | 1 | 3 | 0.7917 | 0.8889 |
| controlled_vs_candidate_no_context | naturalness | 9 | 3 | 0 | 0.7500 | 0.7500 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 8 | 1 | 3 | 0.7917 | 0.8889 |
| controlled_vs_candidate_no_context | persona_style | 3 | 1 | 8 | 0.5833 | 0.7500 |
| controlled_vs_candidate_no_context | distinct1 | 4 | 8 | 0 | 0.3333 | 0.3333 |
| controlled_vs_candidate_no_context | length_score | 8 | 4 | 0 | 0.6667 | 0.6667 |
| controlled_vs_candidate_no_context | sentence_score | 6 | 0 | 6 | 0.7500 | 1.0000 |
| controlled_vs_candidate_no_context | overall_quality | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | context_relevance | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 10 | 1 | 1 | 0.8750 | 0.9091 |
| controlled_vs_baseline_no_context | naturalness | 7 | 5 | 0 | 0.5833 | 0.5833 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | context_overlap | 11 | 1 | 0 | 0.9167 | 0.9167 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 10 | 1 | 1 | 0.8750 | 0.9091 |
| controlled_vs_baseline_no_context | persona_style | 1 | 1 | 10 | 0.5000 | 0.5000 |
| controlled_vs_baseline_no_context | distinct1 | 2 | 10 | 0 | 0.1667 | 0.1667 |
| controlled_vs_baseline_no_context | length_score | 8 | 4 | 0 | 0.6667 | 0.6667 |
| controlled_vs_baseline_no_context | sentence_score | 3 | 1 | 8 | 0.5833 | 0.7500 |
| controlled_vs_baseline_no_context | overall_quality | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 6 | 5 | 1 | 0.5417 | 0.5455 |
| controlled_alt_vs_controlled_default | persona_consistency | 2 | 4 | 6 | 0.4167 | 0.3333 |
| controlled_alt_vs_controlled_default | naturalness | 3 | 8 | 1 | 0.2917 | 0.2727 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 5 | 5 | 2 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | context_overlap | 7 | 4 | 1 | 0.6250 | 0.6364 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 1 | 2 | 9 | 0.4583 | 0.3333 |
| controlled_alt_vs_controlled_default | persona_style | 1 | 3 | 8 | 0.4167 | 0.2500 |
| controlled_alt_vs_controlled_default | distinct1 | 6 | 5 | 1 | 0.5417 | 0.5455 |
| controlled_alt_vs_controlled_default | length_score | 4 | 7 | 1 | 0.3750 | 0.3636 |
| controlled_alt_vs_controlled_default | sentence_score | 0 | 1 | 11 | 0.4583 | 0.0000 |
| controlled_alt_vs_controlled_default | overall_quality | 6 | 5 | 1 | 0.5417 | 0.5455 |
| controlled_alt_vs_proposed_raw | context_relevance | 10 | 2 | 0 | 0.8333 | 0.8333 |
| controlled_alt_vs_proposed_raw | persona_consistency | 9 | 1 | 2 | 0.8333 | 0.9000 |
| controlled_alt_vs_proposed_raw | naturalness | 7 | 5 | 0 | 0.5833 | 0.5833 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 10 | 2 | 0 | 0.8333 | 0.8333 |
| controlled_alt_vs_proposed_raw | context_overlap | 10 | 2 | 0 | 0.8333 | 0.8333 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 9 | 1 | 2 | 0.8333 | 0.9000 |
| controlled_alt_vs_proposed_raw | persona_style | 4 | 1 | 7 | 0.6250 | 0.8000 |
| controlled_alt_vs_proposed_raw | distinct1 | 4 | 6 | 2 | 0.4167 | 0.4000 |
| controlled_alt_vs_proposed_raw | length_score | 8 | 4 | 0 | 0.6667 | 0.6667 |
| controlled_alt_vs_proposed_raw | sentence_score | 4 | 1 | 7 | 0.6250 | 0.8000 |
| controlled_alt_vs_proposed_raw | overall_quality | 10 | 2 | 0 | 0.8333 | 0.8333 |
| controlled_alt_vs_candidate_no_context | context_relevance | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 9 | 1 | 2 | 0.8333 | 0.9000 |
| controlled_alt_vs_candidate_no_context | naturalness | 9 | 3 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 11 | 0 | 1 | 0.9583 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 9 | 1 | 2 | 0.8333 | 0.9000 |
| controlled_alt_vs_candidate_no_context | persona_style | 4 | 2 | 6 | 0.5833 | 0.6667 |
| controlled_alt_vs_candidate_no_context | distinct1 | 7 | 5 | 0 | 0.5833 | 0.5833 |
| controlled_alt_vs_candidate_no_context | length_score | 8 | 4 | 0 | 0.6667 | 0.6667 |
| controlled_alt_vs_candidate_no_context | sentence_score | 6 | 1 | 5 | 0.7083 | 0.8571 |
| controlled_alt_vs_candidate_no_context | overall_quality | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context | context_relevance | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 11 | 1 | 0 | 0.9167 | 0.9167 |
| controlled_alt_vs_baseline_no_context | naturalness | 4 | 8 | 0 | 0.3333 | 0.3333 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 11 | 0 | 1 | 0.9583 | 1.0000 |
| controlled_alt_vs_baseline_no_context | context_overlap | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 11 | 1 | 0 | 0.9167 | 0.9167 |
| controlled_alt_vs_baseline_no_context | persona_style | 1 | 2 | 9 | 0.4583 | 0.3333 |
| controlled_alt_vs_baseline_no_context | distinct1 | 3 | 7 | 2 | 0.3333 | 0.3000 |
| controlled_alt_vs_baseline_no_context | length_score | 5 | 7 | 0 | 0.4167 | 0.4167 |
| controlled_alt_vs_baseline_no_context | sentence_score | 3 | 2 | 7 | 0.5417 | 0.6000 |
| controlled_alt_vs_baseline_no_context | overall_quality | 12 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 12 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 10 | 1 | 1 | 0.8750 | 0.9091 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 7 | 5 | 0 | 0.5833 | 0.5833 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 12 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 11 | 1 | 0 | 0.9167 | 0.9167 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 10 | 1 | 1 | 0.8750 | 0.9091 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 1 | 1 | 10 | 0.5000 | 0.5000 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | 2 | 10 | 0 | 0.1667 | 0.1667 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 8 | 4 | 0 | 0.6667 | 0.6667 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 3 | 1 | 8 | 0.5833 | 0.7500 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 12 | 0 | 0 | 1.0000 | 1.0000 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.4167 | 0.4167 | 0.5833 |
| proposed_contextual_controlled_alt | 0.0000 | 0.0000 | 0.1667 | 0.4167 | 0.5833 |
| proposed_contextual | 0.0000 | 0.0000 | 0.4167 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 |
| baseline_no_context | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `7`
- Unique template signatures: `11`
- Template signature ratio: `0.9167`
- Effective sample size by source clustering: `6.55`
- Effective sample size by template-signature clustering: `10.29`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: unavailable (disabled_by_flag).

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
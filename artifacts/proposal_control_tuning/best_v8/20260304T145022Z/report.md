# Proposal Alignment Evaluation Report

- Run ID: `20260304T145022Z`
- Generated: `2026-03-04T14:57:39.965843+00:00`
- Scenarios: `artifacts\proposal_control_tuning\best_v8\20260304T145022Z\scenarios.jsonl`
- Scenario count: `40`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off
- `baseline_no_context`: model `phi3:mini`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2823 (0.2506, 0.3171) | 0.3193 (0.2695, 0.3702) | 0.8808 (0.8540, 0.9059) | 0.3803 (0.3626, 0.3993) | 0.0825 |
| proposed_contextual | 0.0945 (0.0558, 0.1394) | 0.1825 (0.1341, 0.2399) | 0.7748 (0.7592, 0.7925) | 0.2392 (0.2082, 0.2750) | 0.0672 |
| candidate_no_context | 0.0326 (0.0195, 0.0492) | 0.1561 (0.1130, 0.2063) | 0.7894 (0.7715, 0.8101) | 0.2042 (0.1847, 0.2242) | 0.0360 |
| baseline_no_context | 0.0353 (0.0245, 0.0483) | 0.1960 (0.1602, 0.2315) | 0.8787 (0.8602, 0.8971) | 0.2359 (0.2234, 0.2496) | 0.0549 |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0619 | 1.8960 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0264 | 0.1692 |
| proposed_vs_candidate_no_context | naturalness | -0.0146 | -0.0184 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0786 | 2.6535 |
| proposed_vs_candidate_no_context | context_overlap | 0.0228 | 0.5757 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0385 | 0.5550 |
| proposed_vs_candidate_no_context | persona_style | -0.0218 | -0.0433 |
| proposed_vs_candidate_no_context | distinct1 | -0.0005 | -0.0005 |
| proposed_vs_candidate_no_context | length_score | -0.0725 | -0.3308 |
| proposed_vs_candidate_no_context | sentence_score | 0.0012 | 0.0017 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0312 | 0.8664 |
| proposed_vs_candidate_no_context | overall_quality | 0.0350 | 0.1715 |
| proposed_vs_baseline_no_context | context_relevance | 0.0592 | 1.6777 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0135 | -0.0689 |
| proposed_vs_baseline_no_context | naturalness | -0.1039 | -0.1182 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0741 | 2.1744 |
| proposed_vs_baseline_no_context | context_overlap | 0.0244 | 0.6403 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 0.0088 | 0.0890 |
| proposed_vs_baseline_no_context | persona_style | -0.1027 | -0.1758 |
| proposed_vs_baseline_no_context | distinct1 | -0.0469 | -0.0482 |
| proposed_vs_baseline_no_context | length_score | -0.3650 | -0.7134 |
| proposed_vs_baseline_no_context | sentence_score | -0.1212 | -0.1396 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0123 | 0.2237 |
| proposed_vs_baseline_no_context | overall_quality | 0.0033 | 0.0138 |
| controlled_vs_proposed_raw | context_relevance | 0.1878 | 1.9875 |
| controlled_vs_proposed_raw | persona_consistency | 0.1368 | 0.7496 |
| controlled_vs_proposed_raw | naturalness | 0.1059 | 0.1367 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2481 | 2.2923 |
| controlled_vs_proposed_raw | context_overlap | 0.0472 | 0.7559 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1418 | 1.3160 |
| controlled_vs_proposed_raw | persona_style | 0.1168 | 0.2426 |
| controlled_vs_proposed_raw | distinct1 | 0.0093 | 0.0101 |
| controlled_vs_proposed_raw | length_score | 0.4242 | 2.8920 |
| controlled_vs_proposed_raw | sentence_score | 0.1737 | 0.2324 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0154 | 0.2286 |
| controlled_vs_proposed_raw | overall_quality | 0.1411 | 0.5900 |
| controlled_vs_candidate_no_context | context_relevance | 0.2497 | 7.6518 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1632 | 1.0456 |
| controlled_vs_candidate_no_context | naturalness | 0.0914 | 0.1158 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3267 | 11.0281 |
| controlled_vs_candidate_no_context | context_overlap | 0.0701 | 1.7667 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1802 | 2.6014 |
| controlled_vs_candidate_no_context | persona_style | 0.0951 | 0.1889 |
| controlled_vs_candidate_no_context | distinct1 | 0.0089 | 0.0096 |
| controlled_vs_candidate_no_context | length_score | 0.3517 | 1.6046 |
| controlled_vs_candidate_no_context | sentence_score | 0.1750 | 0.2345 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0465 | 1.2931 |
| controlled_vs_candidate_no_context | overall_quality | 0.1761 | 0.8627 |
| controlled_vs_baseline_no_context | context_relevance | 0.2470 | 6.9997 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1233 | 0.6291 |
| controlled_vs_baseline_no_context | naturalness | 0.0021 | 0.0023 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3222 | 9.4511 |
| controlled_vs_baseline_no_context | context_overlap | 0.0716 | 1.8801 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1506 | 1.5223 |
| controlled_vs_baseline_no_context | persona_style | 0.0141 | 0.0241 |
| controlled_vs_baseline_no_context | distinct1 | -0.0376 | -0.0386 |
| controlled_vs_baseline_no_context | length_score | 0.0592 | 0.1156 |
| controlled_vs_baseline_no_context | sentence_score | 0.0525 | 0.0604 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0276 | 0.5035 |
| controlled_vs_baseline_no_context | overall_quality | 0.1444 | 0.6120 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2470 | 6.9997 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1233 | 0.6291 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 0.0021 | 0.0023 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3222 | 9.4511 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0716 | 1.8801 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1506 | 1.5223 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0141 | 0.0241 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0376 | -0.0386 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.0592 | 0.1156 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0525 | 0.0604 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0276 | 0.5035 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1444 | 0.6120 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0619 | (0.0262, 0.1045) | 0.0000 | 0.0619 | (0.0210, 0.1155) | 0.0000 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0264 | (-0.0277, 0.0843) | 0.1810 | 0.0264 | (-0.0363, 0.0941) | 0.2250 |
| proposed_vs_candidate_no_context | naturalness | -0.0146 | (-0.0388, 0.0100) | 0.8743 | -0.0146 | (-0.0490, 0.0140) | 0.8343 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0786 | (0.0331, 0.1318) | 0.0000 | 0.0786 | (0.0272, 0.1454) | 0.0000 |
| proposed_vs_candidate_no_context | context_overlap | 0.0228 | (0.0085, 0.0398) | 0.0003 | 0.0228 | (0.0056, 0.0458) | 0.0007 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0385 | (-0.0250, 0.1101) | 0.1273 | 0.0385 | (-0.0268, 0.1163) | 0.1743 |
| proposed_vs_candidate_no_context | persona_style | -0.0218 | (-0.0780, 0.0265) | 0.7733 | -0.0218 | (-0.0967, 0.0432) | 0.7090 |
| proposed_vs_candidate_no_context | distinct1 | -0.0005 | (-0.0140, 0.0127) | 0.5317 | -0.0005 | (-0.0143, 0.0159) | 0.5203 |
| proposed_vs_candidate_no_context | length_score | -0.0725 | (-0.1583, 0.0183) | 0.9420 | -0.0725 | (-0.1850, 0.0193) | 0.9210 |
| proposed_vs_candidate_no_context | sentence_score | 0.0012 | (-0.0525, 0.0537) | 0.4637 | 0.0012 | (-0.0628, 0.0656) | 0.4477 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0312 | (0.0143, 0.0502) | 0.0003 | 0.0312 | (0.0160, 0.0508) | 0.0000 |
| proposed_vs_candidate_no_context | overall_quality | 0.0350 | (0.0068, 0.0680) | 0.0063 | 0.0350 | (0.0028, 0.0796) | 0.0093 |
| proposed_vs_baseline_no_context | context_relevance | 0.0592 | (0.0176, 0.1073) | 0.0013 | 0.0592 | (0.0056, 0.1289) | 0.0100 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0135 | (-0.0692, 0.0489) | 0.6640 | -0.0135 | (-0.1078, 0.0973) | 0.6363 |
| proposed_vs_baseline_no_context | naturalness | -0.1039 | (-0.1312, -0.0758) | 1.0000 | -0.1039 | (-0.1335, -0.0609) | 1.0000 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0741 | (0.0178, 0.1374) | 0.0030 | 0.0741 | (0.0068, 0.1663) | 0.0093 |
| proposed_vs_baseline_no_context | context_overlap | 0.0244 | (0.0092, 0.0428) | 0.0000 | 0.0244 | (0.0034, 0.0502) | 0.0063 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 0.0088 | (-0.0578, 0.0800) | 0.4173 | 0.0088 | (-0.0888, 0.1348) | 0.4620 |
| proposed_vs_baseline_no_context | persona_style | -0.1027 | (-0.1865, -0.0303) | 0.9977 | -0.1027 | (-0.2735, 0.0349) | 0.9013 |
| proposed_vs_baseline_no_context | distinct1 | -0.0469 | (-0.0626, -0.0296) | 1.0000 | -0.0469 | (-0.0639, -0.0219) | 0.9997 |
| proposed_vs_baseline_no_context | length_score | -0.3650 | (-0.4633, -0.2592) | 1.0000 | -0.3650 | (-0.4608, -0.2271) | 1.0000 |
| proposed_vs_baseline_no_context | sentence_score | -0.1212 | (-0.1912, -0.0488) | 0.9983 | -0.1212 | (-0.2198, 0.0037) | 0.9670 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0123 | (-0.0102, 0.0360) | 0.1567 | 0.0123 | (-0.0217, 0.0586) | 0.2633 |
| proposed_vs_baseline_no_context | overall_quality | 0.0033 | (-0.0324, 0.0436) | 0.4360 | 0.0033 | (-0.0476, 0.0674) | 0.4963 |
| controlled_vs_proposed_raw | context_relevance | 0.1878 | (0.1413, 0.2308) | 0.0000 | 0.1878 | (0.1454, 0.2265) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.1368 | (0.0691, 0.1989) | 0.0000 | 0.1368 | (0.0653, 0.2186) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.1059 | (0.0734, 0.1374) | 0.0000 | 0.1059 | (0.0654, 0.1348) | 0.0000 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2481 | (0.1892, 0.3039) | 0.0000 | 0.2481 | (0.2001, 0.2936) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0472 | (0.0268, 0.0694) | 0.0000 | 0.0472 | (0.0199, 0.0761) | 0.0003 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1418 | (0.0640, 0.2163) | 0.0007 | 0.1418 | (0.0746, 0.2177) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.1168 | (0.0486, 0.1910) | 0.0000 | 0.1168 | (0.0046, 0.2819) | 0.0233 |
| controlled_vs_proposed_raw | distinct1 | 0.0093 | (-0.0066, 0.0249) | 0.1267 | 0.0093 | (-0.0115, 0.0243) | 0.1630 |
| controlled_vs_proposed_raw | length_score | 0.4242 | (0.3100, 0.5367) | 0.0000 | 0.4242 | (0.2978, 0.5152) | 0.0000 |
| controlled_vs_proposed_raw | sentence_score | 0.1737 | (0.1025, 0.2362) | 0.0000 | 0.1737 | (0.0521, 0.2687) | 0.0017 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0154 | (-0.0097, 0.0391) | 0.1117 | 0.0154 | (-0.0029, 0.0423) | 0.0500 |
| controlled_vs_proposed_raw | overall_quality | 0.1411 | (0.1078, 0.1701) | 0.0000 | 0.1411 | (0.1062, 0.1726) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2497 | (0.2141, 0.2856) | 0.0000 | 0.2497 | (0.2270, 0.2770) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1632 | (0.1106, 0.2167) | 0.0000 | 0.1632 | (0.0974, 0.2184) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.0914 | (0.0585, 0.1232) | 0.0000 | 0.0914 | (0.0493, 0.1239) | 0.0000 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3267 | (0.2809, 0.3735) | 0.0000 | 0.3267 | (0.2986, 0.3588) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0701 | (0.0545, 0.0907) | 0.0000 | 0.0701 | (0.0559, 0.0934) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1802 | (0.1164, 0.2412) | 0.0000 | 0.1802 | (0.1055, 0.2496) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.0951 | (0.0312, 0.1680) | 0.0017 | 0.0951 | (0.0058, 0.2024) | 0.0137 |
| controlled_vs_candidate_no_context | distinct1 | 0.0089 | (-0.0056, 0.0219) | 0.1180 | 0.0089 | (-0.0092, 0.0237) | 0.1443 |
| controlled_vs_candidate_no_context | length_score | 0.3517 | (0.2208, 0.4834) | 0.0000 | 0.3517 | (0.1833, 0.4674) | 0.0000 |
| controlled_vs_candidate_no_context | sentence_score | 0.1750 | (0.1050, 0.2362) | 0.0000 | 0.1750 | (0.1034, 0.2574) | 0.0000 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0465 | (0.0248, 0.0693) | 0.0000 | 0.0465 | (0.0239, 0.0778) | 0.0000 |
| controlled_vs_candidate_no_context | overall_quality | 0.1761 | (0.1519, 0.1995) | 0.0000 | 0.1761 | (0.1530, 0.1980) | 0.0000 |
| controlled_vs_baseline_no_context | context_relevance | 0.2470 | (0.2154, 0.2813) | 0.0000 | 0.2470 | (0.2196, 0.2814) | 0.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1233 | (0.0714, 0.1761) | 0.0000 | 0.1233 | (0.0375, 0.2055) | 0.0007 |
| controlled_vs_baseline_no_context | naturalness | 0.0021 | (-0.0272, 0.0317) | 0.4303 | 0.0021 | (-0.0172, 0.0251) | 0.4607 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3222 | (0.2799, 0.3658) | 0.0000 | 0.3222 | (0.2841, 0.3690) | 0.0000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0716 | (0.0544, 0.0930) | 0.0000 | 0.0716 | (0.0575, 0.0971) | 0.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1506 | (0.0850, 0.2204) | 0.0000 | 0.1506 | (0.0401, 0.2580) | 0.0030 |
| controlled_vs_baseline_no_context | persona_style | 0.0141 | (-0.0165, 0.0493) | 0.2067 | 0.0141 | (-0.0144, 0.0643) | 0.3000 |
| controlled_vs_baseline_no_context | distinct1 | -0.0376 | (-0.0508, -0.0232) | 1.0000 | -0.0376 | (-0.0500, -0.0250) | 1.0000 |
| controlled_vs_baseline_no_context | length_score | 0.0592 | (-0.0617, 0.1759) | 0.1717 | 0.0592 | (-0.0189, 0.1360) | 0.0647 |
| controlled_vs_baseline_no_context | sentence_score | 0.0525 | (-0.0262, 0.1313) | 0.1140 | 0.0525 | (-0.0525, 0.1703) | 0.1923 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0276 | (0.0035, 0.0525) | 0.0123 | 0.0276 | (-0.0049, 0.0682) | 0.0543 |
| controlled_vs_baseline_no_context | overall_quality | 0.1444 | (0.1233, 0.1655) | 0.0000 | 0.1444 | (0.1098, 0.1801) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2470 | (0.2151, 0.2823) | 0.0000 | 0.2470 | (0.2189, 0.2805) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1233 | (0.0703, 0.1753) | 0.0000 | 0.1233 | (0.0406, 0.2097) | 0.0010 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 0.0021 | (-0.0273, 0.0312) | 0.4547 | 0.0021 | (-0.0168, 0.0252) | 0.4480 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3222 | (0.2813, 0.3668) | 0.0000 | 0.3222 | (0.2846, 0.3687) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0716 | (0.0547, 0.0930) | 0.0000 | 0.0716 | (0.0572, 0.0970) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1506 | (0.0812, 0.2199) | 0.0000 | 0.1506 | (0.0414, 0.2560) | 0.0023 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0141 | (-0.0190, 0.0516) | 0.2073 | 0.0141 | (-0.0144, 0.0651) | 0.2803 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0376 | (-0.0504, -0.0240) | 1.0000 | -0.0376 | (-0.0505, -0.0252) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.0592 | (-0.0600, 0.1825) | 0.1623 | 0.0592 | (-0.0200, 0.1375) | 0.0597 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0525 | (-0.0262, 0.1225) | 0.1023 | 0.0525 | (-0.0521, 0.1750) | 0.1857 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0276 | (0.0030, 0.0521) | 0.0143 | 0.0276 | (-0.0061, 0.0671) | 0.0593 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1444 | (0.1229, 0.1652) | 0.0000 | 0.1444 | (0.1113, 0.1790) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 20 | 3 | 17 | 0.7125 | 0.8696 |
| proposed_vs_candidate_no_context | persona_consistency | 10 | 7 | 23 | 0.5375 | 0.5882 |
| proposed_vs_candidate_no_context | naturalness | 9 | 14 | 17 | 0.4375 | 0.3913 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 12 | 2 | 26 | 0.6250 | 0.8571 |
| proposed_vs_candidate_no_context | context_overlap | 20 | 3 | 17 | 0.7125 | 0.8696 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 8 | 6 | 26 | 0.5250 | 0.5714 |
| proposed_vs_candidate_no_context | persona_style | 3 | 4 | 33 | 0.4875 | 0.4286 |
| proposed_vs_candidate_no_context | distinct1 | 11 | 12 | 17 | 0.4875 | 0.4783 |
| proposed_vs_candidate_no_context | length_score | 6 | 17 | 17 | 0.3625 | 0.2609 |
| proposed_vs_candidate_no_context | sentence_score | 5 | 5 | 30 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | bertscore_f1 | 28 | 4 | 8 | 0.8000 | 0.8750 |
| proposed_vs_candidate_no_context | overall_quality | 26 | 6 | 8 | 0.7500 | 0.8125 |
| proposed_vs_baseline_no_context | context_relevance | 21 | 19 | 0 | 0.5250 | 0.5250 |
| proposed_vs_baseline_no_context | persona_consistency | 13 | 15 | 12 | 0.4750 | 0.4643 |
| proposed_vs_baseline_no_context | naturalness | 5 | 35 | 0 | 0.1250 | 0.1250 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 11 | 9 | 20 | 0.5250 | 0.5500 |
| proposed_vs_baseline_no_context | context_overlap | 26 | 13 | 1 | 0.6625 | 0.6667 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 10 | 12 | 18 | 0.4750 | 0.4545 |
| proposed_vs_baseline_no_context | persona_style | 4 | 11 | 25 | 0.4125 | 0.2667 |
| proposed_vs_baseline_no_context | distinct1 | 6 | 32 | 2 | 0.1750 | 0.1579 |
| proposed_vs_baseline_no_context | length_score | 4 | 35 | 1 | 0.1125 | 0.1026 |
| proposed_vs_baseline_no_context | sentence_score | 5 | 19 | 16 | 0.3250 | 0.2083 |
| proposed_vs_baseline_no_context | bertscore_f1 | 20 | 20 | 0 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context | overall_quality | 15 | 25 | 0 | 0.3750 | 0.3750 |
| controlled_vs_proposed_raw | context_relevance | 36 | 4 | 0 | 0.9000 | 0.9000 |
| controlled_vs_proposed_raw | persona_consistency | 25 | 6 | 9 | 0.7375 | 0.8065 |
| controlled_vs_proposed_raw | naturalness | 29 | 10 | 1 | 0.7375 | 0.7436 |
| controlled_vs_proposed_raw | context_keyword_coverage | 36 | 3 | 1 | 0.9125 | 0.9231 |
| controlled_vs_proposed_raw | context_overlap | 33 | 7 | 0 | 0.8250 | 0.8250 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 25 | 5 | 10 | 0.7500 | 0.8333 |
| controlled_vs_proposed_raw | persona_style | 11 | 1 | 28 | 0.6250 | 0.9167 |
| controlled_vs_proposed_raw | distinct1 | 23 | 14 | 3 | 0.6125 | 0.6216 |
| controlled_vs_proposed_raw | length_score | 30 | 7 | 3 | 0.7875 | 0.8108 |
| controlled_vs_proposed_raw | sentence_score | 23 | 3 | 14 | 0.7500 | 0.8846 |
| controlled_vs_proposed_raw | bertscore_f1 | 26 | 14 | 0 | 0.6500 | 0.6500 |
| controlled_vs_proposed_raw | overall_quality | 37 | 3 | 0 | 0.9250 | 0.9250 |
| controlled_vs_candidate_no_context | context_relevance | 40 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 29 | 2 | 9 | 0.8375 | 0.9355 |
| controlled_vs_candidate_no_context | naturalness | 32 | 8 | 0 | 0.8000 | 0.8000 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 39 | 0 | 1 | 0.9875 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 39 | 1 | 0 | 0.9750 | 0.9750 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 29 | 2 | 9 | 0.8375 | 0.9355 |
| controlled_vs_candidate_no_context | persona_style | 9 | 2 | 29 | 0.5875 | 0.8182 |
| controlled_vs_candidate_no_context | distinct1 | 22 | 15 | 3 | 0.5875 | 0.5946 |
| controlled_vs_candidate_no_context | length_score | 31 | 8 | 1 | 0.7875 | 0.7949 |
| controlled_vs_candidate_no_context | sentence_score | 22 | 2 | 16 | 0.7500 | 0.9167 |
| controlled_vs_candidate_no_context | bertscore_f1 | 32 | 8 | 0 | 0.8000 | 0.8000 |
| controlled_vs_candidate_no_context | overall_quality | 39 | 1 | 0 | 0.9750 | 0.9750 |
| controlled_vs_baseline_no_context | context_relevance | 40 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 26 | 6 | 8 | 0.7500 | 0.8125 |
| controlled_vs_baseline_no_context | naturalness | 19 | 21 | 0 | 0.4750 | 0.4750 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 40 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | context_overlap | 39 | 1 | 0 | 0.9750 | 0.9750 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 25 | 6 | 9 | 0.7375 | 0.8065 |
| controlled_vs_baseline_no_context | persona_style | 4 | 2 | 34 | 0.5250 | 0.6667 |
| controlled_vs_baseline_no_context | distinct1 | 8 | 29 | 3 | 0.2375 | 0.2162 |
| controlled_vs_baseline_no_context | length_score | 24 | 16 | 0 | 0.6000 | 0.6000 |
| controlled_vs_baseline_no_context | sentence_score | 13 | 7 | 20 | 0.5750 | 0.6500 |
| controlled_vs_baseline_no_context | bertscore_f1 | 25 | 15 | 0 | 0.6250 | 0.6250 |
| controlled_vs_baseline_no_context | overall_quality | 40 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 40 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 26 | 6 | 8 | 0.7500 | 0.8125 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 19 | 21 | 0 | 0.4750 | 0.4750 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 40 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 39 | 1 | 0 | 0.9750 | 0.9750 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 25 | 6 | 9 | 0.7375 | 0.8065 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 4 | 2 | 34 | 0.5250 | 0.6667 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | 8 | 29 | 3 | 0.2375 | 0.2162 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 24 | 16 | 0 | 0.6000 | 0.6000 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 13 | 7 | 20 | 0.5750 | 0.6500 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 25 | 15 | 0 | 0.6250 | 0.6250 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 40 | 0 | 0 | 1.0000 | 1.0000 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.4000 | 0.3250 | 0.6750 |
| proposed_contextual | 0.0000 | 0.0000 | 0.5250 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.5750 | 0.0000 | 0.0000 |
| baseline_no_context | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `8`
- Unique template signatures: `33`
- Template signature ratio: `0.8250`
- Effective sample size by source clustering: `7.02`
- Effective sample size by template-signature clustering: `28.57`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: enabled.

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
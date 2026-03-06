# Proposal Alignment Evaluation Report

- Run ID: `20260304T225202Z`
- Generated: `2026-03-04T22:56:48.417691+00:00`
- Scenarios: `artifacts\proposal_control_tuning\tuning_continue\20260304T225202Z\scenarios.jsonl`
- Scenario count: `20`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off
- `baseline_no_context`: model `phi3:mini`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2629 (0.2311, 0.2967) | 0.3494 (0.2768, 0.4353) | 0.8797 (0.8437, 0.9144) | 0.3801 (0.3571, 0.4063) | 0.0680 |
| proposed_contextual | 0.0772 (0.0358, 0.1278) | 0.1710 (0.1123, 0.2431) | 0.7902 (0.7653, 0.8213) | 0.2292 (0.1959, 0.2652) | 0.0620 |
| candidate_no_context | 0.0295 (0.0165, 0.0471) | 0.2072 (0.1311, 0.2914) | 0.7899 (0.7650, 0.8169) | 0.2193 (0.1888, 0.2539) | 0.0359 |
| baseline_no_context | 0.0279 (0.0145, 0.0449) | 0.1684 (0.1283, 0.2171) | 0.8842 (0.8554, 0.9142) | 0.2229 (0.2060, 0.2437) | 0.0471 |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0477 | 1.6130 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0362 | -0.1747 |
| proposed_vs_candidate_no_context | naturalness | 0.0003 | 0.0004 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0625 | 2.7500 |
| proposed_vs_candidate_no_context | context_overlap | 0.0130 | 0.2863 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0571 | -0.4571 |
| proposed_vs_candidate_no_context | persona_style | 0.0477 | 0.0890 |
| proposed_vs_candidate_no_context | distinct1 | 0.0096 | 0.0103 |
| proposed_vs_candidate_no_context | length_score | -0.0100 | -0.0484 |
| proposed_vs_candidate_no_context | sentence_score | -0.0150 | -0.0194 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0261 | 0.7278 |
| proposed_vs_candidate_no_context | overall_quality | 0.0099 | 0.0450 |
| proposed_vs_baseline_no_context | context_relevance | 0.0492 | 1.7623 |
| proposed_vs_baseline_no_context | persona_consistency | 0.0026 | 0.0155 |
| proposed_vs_baseline_no_context | naturalness | -0.0940 | -0.1063 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0620 | 2.6645 |
| proposed_vs_baseline_no_context | context_overlap | 0.0196 | 0.5032 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 0.0210 | 0.4467 |
| proposed_vs_baseline_no_context | persona_style | -0.0707 | -0.1081 |
| proposed_vs_baseline_no_context | distinct1 | -0.0531 | -0.0536 |
| proposed_vs_baseline_no_context | length_score | -0.3300 | -0.6266 |
| proposed_vs_baseline_no_context | sentence_score | -0.0675 | -0.0818 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0149 | 0.3166 |
| proposed_vs_baseline_no_context | overall_quality | 0.0064 | 0.0285 |
| controlled_vs_proposed_raw | context_relevance | 0.1857 | 2.4057 |
| controlled_vs_proposed_raw | persona_consistency | 0.1784 | 1.0435 |
| controlled_vs_proposed_raw | naturalness | 0.0895 | 0.1133 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2448 | 2.8729 |
| controlled_vs_proposed_raw | context_overlap | 0.0477 | 0.8166 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2126 | 3.1333 |
| controlled_vs_proposed_raw | persona_style | 0.0416 | 0.0712 |
| controlled_vs_proposed_raw | distinct1 | -0.0064 | -0.0068 |
| controlled_vs_proposed_raw | length_score | 0.3850 | 1.9576 |
| controlled_vs_proposed_raw | sentence_score | 0.1550 | 0.2046 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0060 | 0.0967 |
| controlled_vs_proposed_raw | overall_quality | 0.1509 | 0.6585 |
| controlled_vs_candidate_no_context | context_relevance | 0.2334 | 7.8992 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1422 | 0.6866 |
| controlled_vs_candidate_no_context | naturalness | 0.0899 | 0.1138 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3073 | 13.5233 |
| controlled_vs_candidate_no_context | context_overlap | 0.0607 | 1.3367 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1555 | 1.2438 |
| controlled_vs_candidate_no_context | persona_style | 0.0892 | 0.1665 |
| controlled_vs_candidate_no_context | distinct1 | 0.0032 | 0.0034 |
| controlled_vs_candidate_no_context | length_score | 0.3750 | 1.8145 |
| controlled_vs_candidate_no_context | sentence_score | 0.1400 | 0.1812 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0321 | 0.8948 |
| controlled_vs_candidate_no_context | overall_quality | 0.1608 | 0.7331 |
| controlled_vs_baseline_no_context | context_relevance | 0.2350 | 8.4075 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1810 | 1.0752 |
| controlled_vs_baseline_no_context | naturalness | -0.0044 | -0.0050 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3068 | 13.1922 |
| controlled_vs_baseline_no_context | context_overlap | 0.0673 | 1.7307 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2336 | 4.9797 |
| controlled_vs_baseline_no_context | persona_style | -0.0292 | -0.0446 |
| controlled_vs_baseline_no_context | distinct1 | -0.0595 | -0.0600 |
| controlled_vs_baseline_no_context | length_score | 0.0550 | 0.1044 |
| controlled_vs_baseline_no_context | sentence_score | 0.0875 | 0.1061 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0209 | 0.4439 |
| controlled_vs_baseline_no_context | overall_quality | 0.1573 | 0.7058 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2350 | 8.4075 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1810 | 1.0752 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0044 | -0.0050 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3068 | 13.1922 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0673 | 1.7307 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2336 | 4.9797 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | -0.0292 | -0.0446 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0595 | -0.0600 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.0550 | 0.1044 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0875 | 0.1061 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0209 | 0.4439 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1573 | 0.7058 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0477 | (0.0043, 0.1011) | 0.0113 | 0.0477 | (0.0069, 0.1022) | 0.0010 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0362 | (-0.1096, 0.0337) | 0.8430 | -0.0362 | (-0.0872, -0.0038) | 0.9927 |
| proposed_vs_candidate_no_context | naturalness | 0.0003 | (-0.0180, 0.0166) | 0.4710 | 0.0003 | (-0.0159, 0.0133) | 0.4640 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0625 | (0.0083, 0.1345) | 0.0147 | 0.0625 | (0.0091, 0.1390) | 0.0037 |
| proposed_vs_candidate_no_context | context_overlap | 0.0130 | (0.0028, 0.0249) | 0.0037 | 0.0130 | (0.0025, 0.0267) | 0.0023 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0571 | (-0.1417, 0.0250) | 0.9113 | -0.0571 | (-0.1162, -0.0176) | 1.0000 |
| proposed_vs_candidate_no_context | persona_style | 0.0477 | (-0.0038, 0.1289) | 0.0553 | 0.0477 | (-0.0015, 0.1273) | 0.0357 |
| proposed_vs_candidate_no_context | distinct1 | 0.0096 | (-0.0025, 0.0249) | 0.0740 | 0.0096 | (-0.0004, 0.0221) | 0.0347 |
| proposed_vs_candidate_no_context | length_score | -0.0100 | (-0.0883, 0.0500) | 0.5830 | -0.0100 | (-0.0630, 0.0407) | 0.6370 |
| proposed_vs_candidate_no_context | sentence_score | -0.0150 | (-0.0875, 0.0550) | 0.6600 | -0.0150 | (-0.0765, 0.0333) | 0.6927 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0261 | (-0.0029, 0.0573) | 0.0410 | 0.0261 | (-0.0021, 0.0705) | 0.0363 |
| proposed_vs_candidate_no_context | overall_quality | 0.0099 | (-0.0263, 0.0427) | 0.2693 | 0.0099 | (-0.0186, 0.0371) | 0.2270 |
| proposed_vs_baseline_no_context | context_relevance | 0.0492 | (0.0052, 0.1060) | 0.0123 | 0.0492 | (0.0118, 0.0974) | 0.0020 |
| proposed_vs_baseline_no_context | persona_consistency | 0.0026 | (-0.0418, 0.0509) | 0.4640 | 0.0026 | (-0.0472, 0.0509) | 0.4597 |
| proposed_vs_baseline_no_context | naturalness | -0.0940 | (-0.1245, -0.0618) | 1.0000 | -0.0940 | (-0.1283, -0.0409) | 0.9997 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0620 | (0.0016, 0.1405) | 0.0223 | 0.0620 | (0.0124, 0.1251) | 0.0070 |
| proposed_vs_baseline_no_context | context_overlap | 0.0196 | (0.0040, 0.0361) | 0.0073 | 0.0196 | (0.0036, 0.0402) | 0.0037 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 0.0210 | (-0.0236, 0.0721) | 0.1990 | 0.0210 | (-0.0185, 0.0680) | 0.1517 |
| proposed_vs_baseline_no_context | persona_style | -0.0707 | (-0.1664, 0.0083) | 0.9607 | -0.0707 | (-0.1888, 0.0167) | 0.9373 |
| proposed_vs_baseline_no_context | distinct1 | -0.0531 | (-0.0686, -0.0368) | 1.0000 | -0.0531 | (-0.0682, -0.0367) | 1.0000 |
| proposed_vs_baseline_no_context | length_score | -0.3300 | (-0.4733, -0.1983) | 1.0000 | -0.3300 | (-0.4579, -0.1500) | 0.9997 |
| proposed_vs_baseline_no_context | sentence_score | -0.0675 | (-0.1700, 0.0350) | 0.9193 | -0.0675 | (-0.1687, 0.0656) | 0.8717 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0149 | (-0.0220, 0.0535) | 0.2353 | 0.0149 | (-0.0330, 0.0666) | 0.2827 |
| proposed_vs_baseline_no_context | overall_quality | 0.0064 | (-0.0215, 0.0329) | 0.3257 | 0.0064 | (-0.0198, 0.0336) | 0.3210 |
| controlled_vs_proposed_raw | context_relevance | 0.1857 | (0.1308, 0.2369) | 0.0000 | 0.1857 | (0.1194, 0.2447) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.1784 | (0.0799, 0.2801) | 0.0000 | 0.1784 | (0.0604, 0.2654) | 0.0023 |
| controlled_vs_proposed_raw | naturalness | 0.0895 | (0.0415, 0.1317) | 0.0000 | 0.0895 | (0.0411, 0.1200) | 0.0013 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2448 | (0.1670, 0.3143) | 0.0000 | 0.2448 | (0.1544, 0.3206) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0477 | (0.0294, 0.0664) | 0.0000 | 0.0477 | (0.0270, 0.0748) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2126 | (0.1012, 0.3412) | 0.0000 | 0.2126 | (0.0787, 0.3257) | 0.0007 |
| controlled_vs_proposed_raw | persona_style | 0.0416 | (-0.0496, 0.1318) | 0.1813 | 0.0416 | (-0.0550, 0.1333) | 0.1827 |
| controlled_vs_proposed_raw | distinct1 | -0.0064 | (-0.0276, 0.0147) | 0.7160 | -0.0064 | (-0.0217, 0.0048) | 0.8560 |
| controlled_vs_proposed_raw | length_score | 0.3850 | (0.1833, 0.5683) | 0.0007 | 0.3850 | (0.2037, 0.5173) | 0.0010 |
| controlled_vs_proposed_raw | sentence_score | 0.1550 | (0.0325, 0.2750) | 0.0110 | 0.1550 | (-0.0250, 0.2826) | 0.0630 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0060 | (-0.0282, 0.0352) | 0.3433 | 0.0060 | (-0.0173, 0.0349) | 0.2993 |
| controlled_vs_proposed_raw | overall_quality | 0.1509 | (0.1114, 0.1928) | 0.0000 | 0.1509 | (0.1044, 0.1882) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2334 | (0.2006, 0.2693) | 0.0000 | 0.2334 | (0.2144, 0.2644) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1422 | (0.0256, 0.2589) | 0.0070 | 0.1422 | (0.0058, 0.2497) | 0.0210 |
| controlled_vs_candidate_no_context | naturalness | 0.0899 | (0.0392, 0.1387) | 0.0003 | 0.0899 | (0.0460, 0.1150) | 0.0010 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3073 | (0.2611, 0.3576) | 0.0000 | 0.3073 | (0.2763, 0.3473) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0607 | (0.0468, 0.0762) | 0.0000 | 0.0607 | (0.0464, 0.0853) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1555 | (0.0202, 0.3019) | 0.0110 | 0.1555 | (-0.0044, 0.2909) | 0.0300 |
| controlled_vs_candidate_no_context | persona_style | 0.0892 | (0.0042, 0.1946) | 0.0187 | 0.0892 | (-0.0151, 0.2204) | 0.0673 |
| controlled_vs_candidate_no_context | distinct1 | 0.0032 | (-0.0118, 0.0175) | 0.3320 | 0.0032 | (-0.0089, 0.0134) | 0.2770 |
| controlled_vs_candidate_no_context | length_score | 0.3750 | (0.1450, 0.5767) | 0.0000 | 0.3750 | (0.2190, 0.4884) | 0.0003 |
| controlled_vs_candidate_no_context | sentence_score | 0.1400 | (0.0175, 0.2450) | 0.0213 | 0.1400 | (-0.0412, 0.2739) | 0.0790 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0321 | (0.0041, 0.0626) | 0.0127 | 0.0321 | (0.0063, 0.0708) | 0.0043 |
| controlled_vs_candidate_no_context | overall_quality | 0.1608 | (0.1197, 0.2004) | 0.0000 | 0.1608 | (0.1176, 0.1957) | 0.0000 |
| controlled_vs_baseline_no_context | context_relevance | 0.2350 | (0.1930, 0.2761) | 0.0000 | 0.2350 | (0.2068, 0.2734) | 0.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1810 | (0.1067, 0.2702) | 0.0000 | 0.1810 | (0.0887, 0.2533) | 0.0000 |
| controlled_vs_baseline_no_context | naturalness | -0.0044 | (-0.0579, 0.0451) | 0.5590 | -0.0044 | (-0.0452, 0.0424) | 0.5893 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3068 | (0.2454, 0.3629) | 0.0000 | 0.3068 | (0.2647, 0.3561) | 0.0000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0673 | (0.0516, 0.0852) | 0.0000 | 0.0673 | (0.0530, 0.0915) | 0.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2336 | (0.1407, 0.3483) | 0.0000 | 0.2336 | (0.1263, 0.3247) | 0.0000 |
| controlled_vs_baseline_no_context | persona_style | -0.0292 | (-0.0625, 0.0000) | 1.0000 | -0.0292 | (-0.0833, 0.0000) | 1.0000 |
| controlled_vs_baseline_no_context | distinct1 | -0.0595 | (-0.0770, -0.0389) | 1.0000 | -0.0595 | (-0.0744, -0.0412) | 1.0000 |
| controlled_vs_baseline_no_context | length_score | 0.0550 | (-0.1884, 0.2884) | 0.3083 | 0.0550 | (-0.1485, 0.2611) | 0.2777 |
| controlled_vs_baseline_no_context | sentence_score | 0.0875 | (0.0000, 0.1925) | 0.0537 | 0.0875 | (-0.0350, 0.2100) | 0.0960 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0209 | (-0.0101, 0.0532) | 0.0980 | 0.0209 | (-0.0212, 0.0585) | 0.1567 |
| controlled_vs_baseline_no_context | overall_quality | 0.1573 | (0.1296, 0.1898) | 0.0000 | 0.1573 | (0.1261, 0.1815) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2350 | (0.1938, 0.2753) | 0.0000 | 0.2350 | (0.2064, 0.2725) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1810 | (0.1077, 0.2720) | 0.0000 | 0.1810 | (0.0854, 0.2529) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0044 | (-0.0559, 0.0458) | 0.5567 | -0.0044 | (-0.0476, 0.0448) | 0.5933 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3068 | (0.2470, 0.3629) | 0.0000 | 0.3068 | (0.2624, 0.3580) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0673 | (0.0518, 0.0851) | 0.0000 | 0.0673 | (0.0527, 0.0919) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2336 | (0.1429, 0.3455) | 0.0000 | 0.2336 | (0.1263, 0.3247) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | -0.0292 | (-0.0625, 0.0000) | 1.0000 | -0.0292 | (-0.0833, 0.0000) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0595 | (-0.0763, -0.0398) | 1.0000 | -0.0595 | (-0.0745, -0.0410) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.0550 | (-0.1950, 0.2800) | 0.3267 | 0.0550 | (-0.1456, 0.2706) | 0.2813 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0875 | (0.0000, 0.1750) | 0.0543 | 0.0875 | (-0.0333, 0.2100) | 0.1000 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0209 | (-0.0125, 0.0516) | 0.1037 | 0.0209 | (-0.0182, 0.0591) | 0.1533 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1573 | (0.1296, 0.1866) | 0.0000 | 0.1573 | (0.1266, 0.1814) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 10 | 3 | 7 | 0.6750 | 0.7692 |
| proposed_vs_candidate_no_context | persona_consistency | 4 | 5 | 11 | 0.4750 | 0.4444 |
| proposed_vs_candidate_no_context | naturalness | 9 | 4 | 7 | 0.6250 | 0.6923 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 8 | 2 | 10 | 0.6500 | 0.8000 |
| proposed_vs_candidate_no_context | context_overlap | 10 | 3 | 7 | 0.6750 | 0.7692 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 2 | 5 | 13 | 0.4250 | 0.2857 |
| proposed_vs_candidate_no_context | persona_style | 3 | 1 | 16 | 0.5500 | 0.7500 |
| proposed_vs_candidate_no_context | distinct1 | 5 | 7 | 8 | 0.4500 | 0.4167 |
| proposed_vs_candidate_no_context | length_score | 8 | 5 | 7 | 0.5750 | 0.6154 |
| proposed_vs_candidate_no_context | sentence_score | 2 | 3 | 15 | 0.4750 | 0.4000 |
| proposed_vs_candidate_no_context | bertscore_f1 | 12 | 5 | 3 | 0.6750 | 0.7059 |
| proposed_vs_candidate_no_context | overall_quality | 14 | 3 | 3 | 0.7750 | 0.8235 |
| proposed_vs_baseline_no_context | context_relevance | 13 | 7 | 0 | 0.6500 | 0.6500 |
| proposed_vs_baseline_no_context | persona_consistency | 4 | 4 | 12 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context | naturalness | 3 | 17 | 0 | 0.1500 | 0.1500 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 6 | 1 | 13 | 0.6250 | 0.8571 |
| proposed_vs_baseline_no_context | context_overlap | 14 | 6 | 0 | 0.7000 | 0.7000 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 4 | 2 | 14 | 0.5500 | 0.6667 |
| proposed_vs_baseline_no_context | persona_style | 1 | 4 | 15 | 0.4250 | 0.2000 |
| proposed_vs_baseline_no_context | distinct1 | 1 | 17 | 2 | 0.1000 | 0.0556 |
| proposed_vs_baseline_no_context | length_score | 3 | 16 | 1 | 0.1750 | 0.1579 |
| proposed_vs_baseline_no_context | sentence_score | 2 | 5 | 13 | 0.4250 | 0.2857 |
| proposed_vs_baseline_no_context | bertscore_f1 | 12 | 8 | 0 | 0.6000 | 0.6000 |
| proposed_vs_baseline_no_context | overall_quality | 8 | 12 | 0 | 0.4000 | 0.4000 |
| controlled_vs_proposed_raw | context_relevance | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_vs_proposed_raw | persona_consistency | 16 | 2 | 2 | 0.8500 | 0.8889 |
| controlled_vs_proposed_raw | naturalness | 16 | 4 | 0 | 0.8000 | 0.8000 |
| controlled_vs_proposed_raw | context_keyword_coverage | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_vs_proposed_raw | context_overlap | 18 | 2 | 0 | 0.9000 | 0.9000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 16 | 2 | 2 | 0.8500 | 0.8889 |
| controlled_vs_proposed_raw | persona_style | 4 | 1 | 15 | 0.5750 | 0.8000 |
| controlled_vs_proposed_raw | distinct1 | 10 | 9 | 1 | 0.5250 | 0.5263 |
| controlled_vs_proposed_raw | length_score | 14 | 5 | 1 | 0.7250 | 0.7368 |
| controlled_vs_proposed_raw | sentence_score | 11 | 3 | 6 | 0.7000 | 0.7857 |
| controlled_vs_proposed_raw | bertscore_f1 | 13 | 7 | 0 | 0.6500 | 0.6500 |
| controlled_vs_proposed_raw | overall_quality | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_vs_candidate_no_context | context_relevance | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 14 | 4 | 2 | 0.7500 | 0.7778 |
| controlled_vs_candidate_no_context | naturalness | 17 | 3 | 0 | 0.8500 | 0.8500 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 13 | 4 | 3 | 0.7250 | 0.7647 |
| controlled_vs_candidate_no_context | persona_style | 6 | 1 | 13 | 0.6250 | 0.8571 |
| controlled_vs_candidate_no_context | distinct1 | 11 | 7 | 2 | 0.6000 | 0.6111 |
| controlled_vs_candidate_no_context | length_score | 14 | 6 | 0 | 0.7000 | 0.7000 |
| controlled_vs_candidate_no_context | sentence_score | 12 | 4 | 4 | 0.7000 | 0.7500 |
| controlled_vs_candidate_no_context | bertscore_f1 | 15 | 5 | 0 | 0.7500 | 0.7500 |
| controlled_vs_candidate_no_context | overall_quality | 18 | 2 | 0 | 0.9000 | 0.9000 |
| controlled_vs_baseline_no_context | context_relevance | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 17 | 1 | 2 | 0.9000 | 0.9444 |
| controlled_vs_baseline_no_context | naturalness | 11 | 9 | 0 | 0.5500 | 0.5500 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 19 | 0 | 1 | 0.9750 | 1.0000 |
| controlled_vs_baseline_no_context | context_overlap | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 17 | 1 | 2 | 0.9000 | 0.9444 |
| controlled_vs_baseline_no_context | persona_style | 0 | 3 | 17 | 0.4250 | 0.0000 |
| controlled_vs_baseline_no_context | distinct1 | 1 | 18 | 1 | 0.0750 | 0.0526 |
| controlled_vs_baseline_no_context | length_score | 13 | 7 | 0 | 0.6500 | 0.6500 |
| controlled_vs_baseline_no_context | sentence_score | 7 | 2 | 11 | 0.6250 | 0.7778 |
| controlled_vs_baseline_no_context | bertscore_f1 | 12 | 8 | 0 | 0.6000 | 0.6000 |
| controlled_vs_baseline_no_context | overall_quality | 20 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 20 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 17 | 1 | 2 | 0.9000 | 0.9444 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 11 | 9 | 0 | 0.5500 | 0.5500 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 19 | 0 | 1 | 0.9750 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 20 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 17 | 1 | 2 | 0.9000 | 0.9444 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0 | 3 | 17 | 0.4250 | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | 1 | 18 | 1 | 0.0750 | 0.0526 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 13 | 7 | 0 | 0.6500 | 0.6500 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 7 | 2 | 11 | 0.6250 | 0.7778 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 12 | 8 | 0 | 0.6000 | 0.6000 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 20 | 0 | 0 | 1.0000 | 1.0000 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.3000 | 0.3000 | 0.7000 |
| proposed_contextual | 0.0000 | 0.0000 | 0.4500 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.6000 | 0.0000 | 0.0000 |
| baseline_no_context | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `8`
- Unique template signatures: `18`
- Template signature ratio: `0.9000`
- Effective sample size by source clustering: `6.67`
- Effective sample size by template-signature clustering: `16.67`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: enabled.

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
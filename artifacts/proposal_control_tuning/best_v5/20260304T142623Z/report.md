# Proposal Alignment Evaluation Report

- Run ID: `20260304T142623Z`
- Generated: `2026-03-04T14:34:05.244687+00:00`
- Scenarios: `artifacts\proposal_control_tuning\best_v5\20260304T142623Z\scenarios.jsonl`
- Scenario count: `40`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off
- `baseline_no_context`: model `phi3:mini`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2782 (0.2534, 0.3059) | 0.3238 (0.2829, 0.3639) | 0.8863 (0.8602, 0.9112) | 0.3817 (0.3654, 0.3989) | 0.0856 |
| proposed_contextual | 0.0804 (0.0514, 0.1124) | 0.1451 (0.1102, 0.1846) | 0.7908 (0.7717, 0.8121) | 0.2232 (0.1989, 0.2510) | 0.0631 |
| candidate_no_context | 0.0264 (0.0172, 0.0377) | 0.1718 (0.1238, 0.2270) | 0.7965 (0.7770, 0.8194) | 0.2068 (0.1871, 0.2278) | 0.0246 |
| baseline_no_context | 0.0387 (0.0267, 0.0518) | 0.1804 (0.1470, 0.2155) | 0.8942 (0.8757, 0.9113) | 0.2347 (0.2208, 0.2497) | 0.0539 |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0540 | 2.0472 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0267 | -0.1554 |
| proposed_vs_candidate_no_context | naturalness | -0.0057 | -0.0071 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0701 | 3.4515 |
| proposed_vs_candidate_no_context | context_overlap | 0.0166 | 0.4087 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0342 | -0.3791 |
| proposed_vs_candidate_no_context | persona_style | 0.0032 | 0.0064 |
| proposed_vs_candidate_no_context | distinct1 | 0.0023 | 0.0025 |
| proposed_vs_candidate_no_context | length_score | -0.0242 | -0.1003 |
| proposed_vs_candidate_no_context | sentence_score | -0.0175 | -0.0229 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0385 | 1.5618 |
| proposed_vs_candidate_no_context | overall_quality | 0.0165 | 0.0798 |
| proposed_vs_baseline_no_context | context_relevance | 0.0417 | 1.0776 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0353 | -0.1957 |
| proposed_vs_baseline_no_context | naturalness | -0.1034 | -0.1156 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0523 | 1.3741 |
| proposed_vs_baseline_no_context | context_overlap | 0.0170 | 0.4225 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0221 | -0.2835 |
| proposed_vs_baseline_no_context | persona_style | -0.0880 | -0.1492 |
| proposed_vs_baseline_no_context | distinct1 | -0.0476 | -0.0486 |
| proposed_vs_baseline_no_context | length_score | -0.3517 | -0.6188 |
| proposed_vs_baseline_no_context | sentence_score | -0.1400 | -0.1580 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0092 | 0.1711 |
| proposed_vs_baseline_no_context | overall_quality | -0.0114 | -0.0487 |
| controlled_vs_proposed_raw | context_relevance | 0.1978 | 2.4589 |
| controlled_vs_proposed_raw | persona_consistency | 0.1787 | 1.2318 |
| controlled_vs_proposed_raw | naturalness | 0.0954 | 0.1207 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2608 | 2.8852 |
| controlled_vs_proposed_raw | context_overlap | 0.0507 | 0.8871 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2031 | 3.6298 |
| controlled_vs_proposed_raw | persona_style | 0.0812 | 0.1619 |
| controlled_vs_proposed_raw | distinct1 | -0.0027 | -0.0029 |
| controlled_vs_proposed_raw | length_score | 0.3983 | 1.8385 |
| controlled_vs_proposed_raw | sentence_score | 0.1687 | 0.2261 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0225 | 0.3574 |
| controlled_vs_proposed_raw | overall_quality | 0.1585 | 0.7099 |
| controlled_vs_candidate_no_context | context_relevance | 0.2518 | 9.5399 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1520 | 0.8850 |
| controlled_vs_candidate_no_context | naturalness | 0.0898 | 0.1127 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3308 | 16.2948 |
| controlled_vs_candidate_no_context | context_overlap | 0.0673 | 1.6584 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1689 | 1.8745 |
| controlled_vs_candidate_no_context | persona_style | 0.0844 | 0.1693 |
| controlled_vs_candidate_no_context | distinct1 | -0.0004 | -0.0005 |
| controlled_vs_candidate_no_context | length_score | 0.3742 | 1.5536 |
| controlled_vs_candidate_no_context | sentence_score | 0.1512 | 0.1980 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0610 | 2.4774 |
| controlled_vs_candidate_no_context | overall_quality | 0.1750 | 0.8463 |
| controlled_vs_baseline_no_context | context_relevance | 0.2395 | 6.1861 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1434 | 0.7950 |
| controlled_vs_baseline_no_context | naturalness | -0.0079 | -0.0089 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3131 | 8.2239 |
| controlled_vs_baseline_no_context | context_overlap | 0.0677 | 1.6844 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1810 | 2.3171 |
| controlled_vs_baseline_no_context | persona_style | -0.0068 | -0.0115 |
| controlled_vs_baseline_no_context | distinct1 | -0.0503 | -0.0513 |
| controlled_vs_baseline_no_context | length_score | 0.0467 | 0.0821 |
| controlled_vs_baseline_no_context | sentence_score | 0.0287 | 0.0324 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0318 | 0.5897 |
| controlled_vs_baseline_no_context | overall_quality | 0.1470 | 0.6265 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2395 | 6.1861 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1434 | 0.7950 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0079 | -0.0089 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3131 | 8.2239 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0677 | 1.6844 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1810 | 2.3171 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | -0.0068 | -0.0115 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0503 | -0.0513 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.0467 | 0.0821 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0287 | 0.0324 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0318 | 0.5897 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1470 | 0.6265 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0540 | (0.0250, 0.0842) | 0.0000 | 0.0540 | (0.0256, 0.0887) | 0.0000 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0267 | (-0.0614, 0.0062) | 0.9443 | -0.0267 | (-0.0624, -0.0000) | 0.9750 |
| proposed_vs_candidate_no_context | naturalness | -0.0057 | (-0.0297, 0.0183) | 0.6800 | -0.0057 | (-0.0228, 0.0140) | 0.7377 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0701 | (0.0335, 0.1127) | 0.0000 | 0.0701 | (0.0309, 0.1132) | 0.0000 |
| proposed_vs_candidate_no_context | context_overlap | 0.0166 | (0.0086, 0.0252) | 0.0000 | 0.0166 | (0.0068, 0.0302) | 0.0000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0342 | (-0.0767, 0.0065) | 0.9560 | -0.0342 | (-0.0782, -0.0031) | 0.9887 |
| proposed_vs_candidate_no_context | persona_style | 0.0032 | (-0.0362, 0.0482) | 0.4560 | 0.0032 | (-0.0032, 0.0124) | 0.3330 |
| proposed_vs_candidate_no_context | distinct1 | 0.0023 | (-0.0105, 0.0147) | 0.3707 | 0.0023 | (-0.0098, 0.0138) | 0.3340 |
| proposed_vs_candidate_no_context | length_score | -0.0242 | (-0.1117, 0.0642) | 0.7113 | -0.0242 | (-0.0882, 0.0512) | 0.7637 |
| proposed_vs_candidate_no_context | sentence_score | -0.0175 | (-0.0788, 0.0437) | 0.7690 | -0.0175 | (-0.0618, 0.0304) | 0.8087 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0385 | (0.0180, 0.0594) | 0.0003 | 0.0385 | (0.0170, 0.0694) | 0.0000 |
| proposed_vs_candidate_no_context | overall_quality | 0.0165 | (-0.0031, 0.0385) | 0.0520 | 0.0165 | (0.0032, 0.0314) | 0.0047 |
| proposed_vs_baseline_no_context | context_relevance | 0.0417 | (0.0146, 0.0728) | 0.0013 | 0.0417 | (0.0084, 0.0783) | 0.0080 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0353 | (-0.0750, 0.0072) | 0.9533 | -0.0353 | (-0.0860, 0.0109) | 0.9207 |
| proposed_vs_baseline_no_context | naturalness | -0.1034 | (-0.1298, -0.0747) | 1.0000 | -0.1034 | (-0.1350, -0.0573) | 1.0000 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0523 | (0.0178, 0.0910) | 0.0010 | 0.0523 | (0.0111, 0.0968) | 0.0060 |
| proposed_vs_baseline_no_context | context_overlap | 0.0170 | (0.0067, 0.0281) | 0.0017 | 0.0170 | (0.0004, 0.0351) | 0.0213 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0221 | (-0.0713, 0.0257) | 0.8280 | -0.0221 | (-0.0645, 0.0206) | 0.8477 |
| proposed_vs_baseline_no_context | persona_style | -0.0880 | (-0.1606, -0.0267) | 0.9990 | -0.0880 | (-0.2350, 0.0256) | 0.9130 |
| proposed_vs_baseline_no_context | distinct1 | -0.0476 | (-0.0599, -0.0336) | 1.0000 | -0.0476 | (-0.0599, -0.0283) | 1.0000 |
| proposed_vs_baseline_no_context | length_score | -0.3517 | (-0.4533, -0.2417) | 1.0000 | -0.3517 | (-0.4667, -0.1892) | 1.0000 |
| proposed_vs_baseline_no_context | sentence_score | -0.1400 | (-0.2012, -0.0788) | 1.0000 | -0.1400 | (-0.2333, -0.0233) | 0.9930 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0092 | (-0.0086, 0.0276) | 0.1613 | 0.0092 | (-0.0138, 0.0389) | 0.2530 |
| proposed_vs_baseline_no_context | overall_quality | -0.0114 | (-0.0339, 0.0133) | 0.8220 | -0.0114 | (-0.0400, 0.0177) | 0.7907 |
| controlled_vs_proposed_raw | context_relevance | 0.1978 | (0.1624, 0.2298) | 0.0000 | 0.1978 | (0.1525, 0.2383) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.1787 | (0.1242, 0.2291) | 0.0000 | 0.1787 | (0.1128, 0.2333) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0954 | (0.0607, 0.1297) | 0.0000 | 0.0954 | (0.0464, 0.1299) | 0.0000 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2608 | (0.2108, 0.3042) | 0.0000 | 0.2608 | (0.2000, 0.3150) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0507 | (0.0367, 0.0633) | 0.0000 | 0.0507 | (0.0350, 0.0673) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2031 | (0.1411, 0.2637) | 0.0000 | 0.2031 | (0.1390, 0.2660) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.0812 | (0.0125, 0.1554) | 0.0120 | 0.0812 | (-0.0399, 0.2333) | 0.1343 |
| controlled_vs_proposed_raw | distinct1 | -0.0027 | (-0.0162, 0.0100) | 0.6660 | -0.0027 | (-0.0227, 0.0119) | 0.6507 |
| controlled_vs_proposed_raw | length_score | 0.3983 | (0.2642, 0.5275) | 0.0000 | 0.3983 | (0.2368, 0.5132) | 0.0000 |
| controlled_vs_proposed_raw | sentence_score | 0.1687 | (0.0837, 0.2462) | 0.0000 | 0.1687 | (0.0500, 0.2511) | 0.0043 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0225 | (0.0033, 0.0396) | 0.0107 | 0.0225 | (-0.0045, 0.0440) | 0.0423 |
| controlled_vs_proposed_raw | overall_quality | 0.1585 | (0.1338, 0.1840) | 0.0000 | 0.1585 | (0.1316, 0.1807) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2518 | (0.2242, 0.2798) | 0.0000 | 0.2518 | (0.2198, 0.2893) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1520 | (0.0878, 0.2130) | 0.0000 | 0.1520 | (0.0649, 0.2186) | 0.0023 |
| controlled_vs_candidate_no_context | naturalness | 0.0898 | (0.0559, 0.1221) | 0.0000 | 0.0898 | (0.0524, 0.1145) | 0.0003 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3308 | (0.2972, 0.3686) | 0.0000 | 0.3308 | (0.2884, 0.3789) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0673 | (0.0562, 0.0785) | 0.0000 | 0.0673 | (0.0560, 0.0835) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1689 | (0.0962, 0.2375) | 0.0000 | 0.1689 | (0.0783, 0.2296) | 0.0007 |
| controlled_vs_candidate_no_context | persona_style | 0.0844 | (0.0108, 0.1656) | 0.0087 | 0.0844 | (-0.0441, 0.2334) | 0.1310 |
| controlled_vs_candidate_no_context | distinct1 | -0.0004 | (-0.0157, 0.0132) | 0.5107 | -0.0004 | (-0.0219, 0.0153) | 0.5233 |
| controlled_vs_candidate_no_context | length_score | 0.3742 | (0.2300, 0.5033) | 0.0000 | 0.3742 | (0.2469, 0.4725) | 0.0000 |
| controlled_vs_candidate_no_context | sentence_score | 0.1512 | (0.0812, 0.2200) | 0.0000 | 0.1512 | (0.0750, 0.2020) | 0.0017 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0610 | (0.0417, 0.0795) | 0.0000 | 0.0610 | (0.0381, 0.0882) | 0.0000 |
| controlled_vs_candidate_no_context | overall_quality | 0.1750 | (0.1504, 0.1984) | 0.0000 | 0.1750 | (0.1454, 0.2000) | 0.0000 |
| controlled_vs_baseline_no_context | context_relevance | 0.2395 | (0.2144, 0.2667) | 0.0000 | 0.2395 | (0.2102, 0.2768) | 0.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1434 | (0.0942, 0.1913) | 0.0000 | 0.1434 | (0.0741, 0.2074) | 0.0000 |
| controlled_vs_baseline_no_context | naturalness | -0.0079 | (-0.0440, 0.0277) | 0.6567 | -0.0079 | (-0.0368, 0.0168) | 0.7440 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3131 | (0.2787, 0.3499) | 0.0000 | 0.3131 | (0.2754, 0.3617) | 0.0000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0677 | (0.0567, 0.0790) | 0.0000 | 0.0677 | (0.0584, 0.0794) | 0.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1810 | (0.1219, 0.2416) | 0.0000 | 0.1810 | (0.0939, 0.2668) | 0.0000 |
| controlled_vs_baseline_no_context | persona_style | -0.0068 | (-0.0436, 0.0311) | 0.6390 | -0.0068 | (-0.0287, 0.0082) | 0.8130 |
| controlled_vs_baseline_no_context | distinct1 | -0.0503 | (-0.0608, -0.0404) | 1.0000 | -0.0503 | (-0.0603, -0.0408) | 1.0000 |
| controlled_vs_baseline_no_context | length_score | 0.0467 | (-0.0958, 0.1800) | 0.2503 | 0.0467 | (-0.0892, 0.1658) | 0.2240 |
| controlled_vs_baseline_no_context | sentence_score | 0.0287 | (-0.0563, 0.1138) | 0.2587 | 0.0287 | (-0.0397, 0.1024) | 0.1813 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0318 | (0.0118, 0.0505) | 0.0010 | 0.0318 | (0.0044, 0.0574) | 0.0130 |
| controlled_vs_baseline_no_context | overall_quality | 0.1470 | (0.1257, 0.1685) | 0.0000 | 0.1470 | (0.1216, 0.1735) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2395 | (0.2143, 0.2662) | 0.0000 | 0.2395 | (0.2110, 0.2759) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1434 | (0.0907, 0.1932) | 0.0000 | 0.1434 | (0.0723, 0.2088) | 0.0003 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0079 | (-0.0422, 0.0263) | 0.6793 | -0.0079 | (-0.0359, 0.0182) | 0.7400 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3131 | (0.2795, 0.3503) | 0.0000 | 0.3131 | (0.2757, 0.3614) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0677 | (0.0565, 0.0794) | 0.0000 | 0.0677 | (0.0580, 0.0792) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1810 | (0.1211, 0.2395) | 0.0000 | 0.1810 | (0.0938, 0.2650) | 0.0003 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | -0.0068 | (-0.0410, 0.0304) | 0.6380 | -0.0068 | (-0.0278, 0.0083) | 0.8127 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0503 | (-0.0606, -0.0406) | 1.0000 | -0.0503 | (-0.0606, -0.0410) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.0467 | (-0.0867, 0.1875) | 0.2480 | 0.0467 | (-0.0833, 0.1642) | 0.2247 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0287 | (-0.0575, 0.1101) | 0.2477 | 0.0287 | (-0.0419, 0.1029) | 0.1957 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0318 | (0.0121, 0.0499) | 0.0003 | 0.0318 | (0.0041, 0.0582) | 0.0157 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1470 | (0.1258, 0.1687) | 0.0000 | 0.1470 | (0.1206, 0.1736) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 19 | 3 | 18 | 0.7000 | 0.8636 |
| proposed_vs_candidate_no_context | persona_consistency | 5 | 10 | 25 | 0.4375 | 0.3333 |
| proposed_vs_candidate_no_context | naturalness | 11 | 11 | 18 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 14 | 2 | 24 | 0.6500 | 0.8750 |
| proposed_vs_candidate_no_context | context_overlap | 18 | 4 | 18 | 0.6750 | 0.8182 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 3 | 8 | 29 | 0.4375 | 0.2727 |
| proposed_vs_candidate_no_context | persona_style | 2 | 3 | 35 | 0.4875 | 0.4000 |
| proposed_vs_candidate_no_context | distinct1 | 9 | 13 | 18 | 0.4500 | 0.4091 |
| proposed_vs_candidate_no_context | length_score | 8 | 13 | 19 | 0.4375 | 0.3810 |
| proposed_vs_candidate_no_context | sentence_score | 5 | 7 | 28 | 0.4750 | 0.4167 |
| proposed_vs_candidate_no_context | bertscore_f1 | 19 | 3 | 18 | 0.7000 | 0.8636 |
| proposed_vs_candidate_no_context | overall_quality | 13 | 9 | 18 | 0.5500 | 0.5909 |
| proposed_vs_baseline_no_context | context_relevance | 24 | 16 | 0 | 0.6000 | 0.6000 |
| proposed_vs_baseline_no_context | persona_consistency | 4 | 16 | 20 | 0.3500 | 0.2000 |
| proposed_vs_baseline_no_context | naturalness | 8 | 32 | 0 | 0.2000 | 0.2000 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 13 | 4 | 23 | 0.6125 | 0.7647 |
| proposed_vs_baseline_no_context | context_overlap | 25 | 15 | 0 | 0.6250 | 0.6250 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 4 | 7 | 29 | 0.4625 | 0.3636 |
| proposed_vs_baseline_no_context | persona_style | 2 | 11 | 27 | 0.3875 | 0.1538 |
| proposed_vs_baseline_no_context | distinct1 | 6 | 30 | 4 | 0.2000 | 0.1667 |
| proposed_vs_baseline_no_context | length_score | 7 | 33 | 0 | 0.1750 | 0.1750 |
| proposed_vs_baseline_no_context | sentence_score | 2 | 18 | 20 | 0.3000 | 0.1000 |
| proposed_vs_baseline_no_context | bertscore_f1 | 21 | 19 | 0 | 0.5250 | 0.5250 |
| proposed_vs_baseline_no_context | overall_quality | 12 | 28 | 0 | 0.3000 | 0.3000 |
| controlled_vs_proposed_raw | context_relevance | 37 | 3 | 0 | 0.9250 | 0.9250 |
| controlled_vs_proposed_raw | persona_consistency | 32 | 3 | 5 | 0.8625 | 0.9143 |
| controlled_vs_proposed_raw | naturalness | 31 | 9 | 0 | 0.7750 | 0.7750 |
| controlled_vs_proposed_raw | context_keyword_coverage | 35 | 2 | 3 | 0.9125 | 0.9459 |
| controlled_vs_proposed_raw | context_overlap | 35 | 5 | 0 | 0.8750 | 0.8750 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 31 | 3 | 6 | 0.8500 | 0.9118 |
| controlled_vs_proposed_raw | persona_style | 9 | 4 | 27 | 0.5625 | 0.6923 |
| controlled_vs_proposed_raw | distinct1 | 21 | 18 | 1 | 0.5375 | 0.5385 |
| controlled_vs_proposed_raw | length_score | 30 | 6 | 4 | 0.8000 | 0.8333 |
| controlled_vs_proposed_raw | sentence_score | 25 | 6 | 9 | 0.7375 | 0.8065 |
| controlled_vs_proposed_raw | bertscore_f1 | 29 | 11 | 0 | 0.7250 | 0.7250 |
| controlled_vs_proposed_raw | overall_quality | 39 | 1 | 0 | 0.9750 | 0.9750 |
| controlled_vs_candidate_no_context | context_relevance | 40 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 30 | 7 | 3 | 0.7875 | 0.8108 |
| controlled_vs_candidate_no_context | naturalness | 32 | 8 | 0 | 0.8000 | 0.8000 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 40 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 39 | 1 | 0 | 0.9750 | 0.9750 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 30 | 7 | 3 | 0.7875 | 0.8108 |
| controlled_vs_candidate_no_context | persona_style | 8 | 4 | 28 | 0.5500 | 0.6667 |
| controlled_vs_candidate_no_context | distinct1 | 21 | 17 | 2 | 0.5500 | 0.5526 |
| controlled_vs_candidate_no_context | length_score | 28 | 9 | 3 | 0.7375 | 0.7568 |
| controlled_vs_candidate_no_context | sentence_score | 21 | 4 | 15 | 0.7125 | 0.8400 |
| controlled_vs_candidate_no_context | bertscore_f1 | 34 | 6 | 0 | 0.8500 | 0.8500 |
| controlled_vs_candidate_no_context | overall_quality | 40 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | context_relevance | 40 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 29 | 4 | 7 | 0.8125 | 0.8788 |
| controlled_vs_baseline_no_context | naturalness | 20 | 20 | 0 | 0.5000 | 0.5000 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 40 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | context_overlap | 38 | 2 | 0 | 0.9500 | 0.9500 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 29 | 3 | 8 | 0.8250 | 0.9062 |
| controlled_vs_baseline_no_context | persona_style | 3 | 5 | 32 | 0.4750 | 0.3750 |
| controlled_vs_baseline_no_context | distinct1 | 1 | 39 | 0 | 0.0250 | 0.0250 |
| controlled_vs_baseline_no_context | length_score | 21 | 17 | 2 | 0.5500 | 0.5526 |
| controlled_vs_baseline_no_context | sentence_score | 12 | 7 | 21 | 0.5625 | 0.6316 |
| controlled_vs_baseline_no_context | bertscore_f1 | 27 | 13 | 0 | 0.6750 | 0.6750 |
| controlled_vs_baseline_no_context | overall_quality | 39 | 1 | 0 | 0.9750 | 0.9750 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 40 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 29 | 4 | 7 | 0.8125 | 0.8788 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 20 | 20 | 0 | 0.5000 | 0.5000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 40 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 38 | 2 | 0 | 0.9500 | 0.9500 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 29 | 3 | 8 | 0.8250 | 0.9062 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 3 | 5 | 32 | 0.4750 | 0.3750 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | 1 | 39 | 0 | 0.0250 | 0.0250 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 21 | 17 | 2 | 0.5500 | 0.5526 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 12 | 7 | 21 | 0.5625 | 0.6316 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 27 | 13 | 0 | 0.6750 | 0.6750 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 39 | 1 | 0 | 0.9750 | 0.9750 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.5000 | 0.3000 | 0.7000 |
| proposed_contextual | 0.0000 | 0.0000 | 0.5500 | 0.0000 | 0.0000 |
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
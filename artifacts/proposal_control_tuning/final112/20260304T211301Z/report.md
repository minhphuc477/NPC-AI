# Proposal Alignment Evaluation Report

- Run ID: `20260304T211301Z`
- Generated: `2026-03-04T21:35:47.970630+00:00`
- Scenarios: `artifacts\proposal_control_tuning\final112\20260304T211301Z\scenarios.jsonl`
- Scenario count: `112`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off
- `baseline_no_context`: model `phi3:mini`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2778 (0.2579, 0.2984) | 0.3554 (0.3232, 0.3881) | 0.8652 (0.8503, 0.8797) | 0.3883 (0.3758, 0.4012) | 0.0896 |
| proposed_contextual | 0.0950 (0.0709, 0.1198) | 0.1577 (0.1342, 0.1832) | 0.7988 (0.7867, 0.8111) | 0.2357 (0.2192, 0.2531) | 0.0710 |
| candidate_no_context | 0.0294 (0.0224, 0.0369) | 0.1892 (0.1614, 0.2192) | 0.8224 (0.8071, 0.8383) | 0.2196 (0.2077, 0.2321) | 0.0369 |
| baseline_no_context | 0.0492 (0.0400, 0.0584) | 0.1902 (0.1690, 0.2114) | 0.8866 (0.8765, 0.8968) | 0.2417 (0.2327, 0.2512) | 0.0618 |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0656 | 2.2299 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0315 | -0.1666 |
| proposed_vs_candidate_no_context | naturalness | -0.0236 | -0.0287 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0854 | 3.4230 |
| proposed_vs_candidate_no_context | context_overlap | 0.0194 | 0.4862 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0352 | -0.3332 |
| proposed_vs_candidate_no_context | persona_style | -0.0170 | -0.0324 |
| proposed_vs_candidate_no_context | distinct1 | -0.0030 | -0.0031 |
| proposed_vs_candidate_no_context | length_score | -0.0935 | -0.2796 |
| proposed_vs_candidate_no_context | sentence_score | -0.0375 | -0.0472 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0341 | 0.9260 |
| proposed_vs_candidate_no_context | overall_quality | 0.0161 | 0.0732 |
| proposed_vs_baseline_no_context | context_relevance | 0.0458 | 0.9320 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0325 | -0.1708 |
| proposed_vs_baseline_no_context | naturalness | -0.0877 | -0.0990 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0586 | 1.1334 |
| proposed_vs_baseline_no_context | context_overlap | 0.0160 | 0.3695 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0195 | -0.2168 |
| proposed_vs_baseline_no_context | persona_style | -0.0846 | -0.1429 |
| proposed_vs_baseline_no_context | distinct1 | -0.0383 | -0.0392 |
| proposed_vs_baseline_no_context | length_score | -0.2949 | -0.5506 |
| proposed_vs_baseline_no_context | sentence_score | -0.1344 | -0.1509 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0092 | 0.1495 |
| proposed_vs_baseline_no_context | overall_quality | -0.0060 | -0.0248 |
| controlled_vs_proposed_raw | context_relevance | 0.1828 | 1.9243 |
| controlled_vs_proposed_raw | persona_consistency | 0.1977 | 1.2537 |
| controlled_vs_proposed_raw | naturalness | 0.0663 | 0.0831 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2390 | 2.1665 |
| controlled_vs_proposed_raw | context_overlap | 0.0515 | 0.8709 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2218 | 3.1517 |
| controlled_vs_proposed_raw | persona_style | 0.1015 | 0.2001 |
| controlled_vs_proposed_raw | distinct1 | -0.0023 | -0.0024 |
| controlled_vs_proposed_raw | length_score | 0.2601 | 1.0803 |
| controlled_vs_proposed_raw | sentence_score | 0.1536 | 0.2031 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0186 | 0.2618 |
| controlled_vs_proposed_raw | overall_quality | 0.1526 | 0.6474 |
| controlled_vs_candidate_no_context | context_relevance | 0.2484 | 8.4454 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1662 | 0.8783 |
| controlled_vs_candidate_no_context | naturalness | 0.0428 | 0.0520 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3244 | 13.0054 |
| controlled_vs_candidate_no_context | context_overlap | 0.0709 | 1.7806 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1866 | 1.7683 |
| controlled_vs_candidate_no_context | persona_style | 0.0845 | 0.1613 |
| controlled_vs_candidate_no_context | distinct1 | -0.0052 | -0.0055 |
| controlled_vs_candidate_no_context | length_score | 0.1667 | 0.4987 |
| controlled_vs_candidate_no_context | sentence_score | 0.1161 | 0.1462 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0527 | 1.4302 |
| controlled_vs_candidate_no_context | overall_quality | 0.1687 | 0.7680 |
| controlled_vs_baseline_no_context | context_relevance | 0.2286 | 4.6497 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1652 | 0.8687 |
| controlled_vs_baseline_no_context | naturalness | -0.0214 | -0.0241 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.2977 | 5.7554 |
| controlled_vs_baseline_no_context | context_overlap | 0.0675 | 1.5622 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2023 | 2.2518 |
| controlled_vs_baseline_no_context | persona_style | 0.0169 | 0.0286 |
| controlled_vs_baseline_no_context | distinct1 | -0.0405 | -0.0415 |
| controlled_vs_baseline_no_context | length_score | -0.0348 | -0.0650 |
| controlled_vs_baseline_no_context | sentence_score | 0.0192 | 0.0216 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0278 | 0.4504 |
| controlled_vs_baseline_no_context | overall_quality | 0.1466 | 0.6065 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2286 | 4.6497 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1652 | 0.8687 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0214 | -0.0241 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.2977 | 5.7554 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0675 | 1.5622 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2023 | 2.2518 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0169 | 0.0286 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0405 | -0.0415 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | -0.0348 | -0.0650 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0192 | 0.0216 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0278 | 0.4504 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1466 | 0.6065 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0656 | (0.0429, 0.0921) | 0.0000 | 0.0656 | (0.0277, 0.1061) | 0.0000 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0315 | (-0.0607, -0.0045) | 0.9893 | -0.0315 | (-0.0958, 0.0066) | 0.8630 |
| proposed_vs_candidate_no_context | naturalness | -0.0236 | (-0.0423, -0.0064) | 0.9970 | -0.0236 | (-0.0547, 0.0026) | 0.9587 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0854 | (0.0549, 0.1197) | 0.0000 | 0.0854 | (0.0377, 0.1403) | 0.0000 |
| proposed_vs_candidate_no_context | context_overlap | 0.0194 | (0.0110, 0.0290) | 0.0000 | 0.0194 | (0.0065, 0.0321) | 0.0000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0352 | (-0.0700, -0.0045) | 0.9887 | -0.0352 | (-0.1136, 0.0113) | 0.7863 |
| proposed_vs_candidate_no_context | persona_style | -0.0170 | (-0.0459, 0.0100) | 0.8917 | -0.0170 | (-0.0413, 0.0035) | 0.9410 |
| proposed_vs_candidate_no_context | distinct1 | -0.0030 | (-0.0113, 0.0060) | 0.7433 | -0.0030 | (-0.0177, 0.0089) | 0.6410 |
| proposed_vs_candidate_no_context | length_score | -0.0935 | (-0.1610, -0.0303) | 0.9977 | -0.0935 | (-0.1964, 0.0009) | 0.9747 |
| proposed_vs_candidate_no_context | sentence_score | -0.0375 | (-0.0750, 0.0000) | 0.9830 | -0.0375 | (-0.0969, 0.0187) | 0.9003 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0341 | (0.0226, 0.0464) | 0.0000 | 0.0341 | (0.0201, 0.0481) | 0.0000 |
| proposed_vs_candidate_no_context | overall_quality | 0.0161 | (-0.0010, 0.0333) | 0.0330 | 0.0161 | (-0.0169, 0.0463) | 0.1653 |
| proposed_vs_baseline_no_context | context_relevance | 0.0458 | (0.0211, 0.0722) | 0.0000 | 0.0458 | (-0.0011, 0.0888) | 0.0283 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0325 | (-0.0624, -0.0028) | 0.9857 | -0.0325 | (-0.0835, 0.0191) | 0.8880 |
| proposed_vs_baseline_no_context | naturalness | -0.0877 | (-0.1015, -0.0735) | 1.0000 | -0.0877 | (-0.1044, -0.0697) | 1.0000 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0586 | (0.0272, 0.0933) | 0.0000 | 0.0586 | (-0.0016, 0.1200) | 0.0283 |
| proposed_vs_baseline_no_context | context_overlap | 0.0160 | (0.0070, 0.0263) | 0.0000 | 0.0160 | (0.0035, 0.0278) | 0.0063 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0195 | (-0.0541, 0.0139) | 0.8643 | -0.0195 | (-0.0854, 0.0435) | 0.7067 |
| proposed_vs_baseline_no_context | persona_style | -0.0846 | (-0.1238, -0.0481) | 1.0000 | -0.0846 | (-0.2006, 0.0005) | 0.9647 |
| proposed_vs_baseline_no_context | distinct1 | -0.0383 | (-0.0464, -0.0300) | 1.0000 | -0.0383 | (-0.0509, -0.0255) | 1.0000 |
| proposed_vs_baseline_no_context | length_score | -0.2949 | (-0.3533, -0.2333) | 1.0000 | -0.2949 | (-0.3705, -0.2116) | 1.0000 |
| proposed_vs_baseline_no_context | sentence_score | -0.1344 | (-0.1750, -0.0938) | 1.0000 | -0.1344 | (-0.2062, -0.0656) | 1.0000 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0092 | (-0.0034, 0.0213) | 0.0793 | 0.0092 | (-0.0137, 0.0352) | 0.2220 |
| proposed_vs_baseline_no_context | overall_quality | -0.0060 | (-0.0238, 0.0117) | 0.7527 | -0.0060 | (-0.0393, 0.0290) | 0.6427 |
| controlled_vs_proposed_raw | context_relevance | 0.1828 | (0.1568, 0.2083) | 0.0000 | 0.1828 | (0.1492, 0.2212) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.1977 | (0.1639, 0.2330) | 0.0000 | 0.1977 | (0.1410, 0.2557) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0663 | (0.0459, 0.0866) | 0.0000 | 0.0663 | (0.0236, 0.1066) | 0.0010 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2390 | (0.2047, 0.2732) | 0.0000 | 0.2390 | (0.1968, 0.2881) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0515 | (0.0388, 0.0646) | 0.0000 | 0.0515 | (0.0378, 0.0701) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2218 | (0.1822, 0.2643) | 0.0000 | 0.2218 | (0.1554, 0.2993) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.1015 | (0.0632, 0.1433) | 0.0000 | 0.1015 | (0.0059, 0.2220) | 0.0133 |
| controlled_vs_proposed_raw | distinct1 | -0.0023 | (-0.0116, 0.0067) | 0.6493 | -0.0023 | (-0.0170, 0.0110) | 0.6200 |
| controlled_vs_proposed_raw | length_score | 0.2601 | (0.1726, 0.3414) | 0.0000 | 0.2601 | (0.1152, 0.4021) | 0.0000 |
| controlled_vs_proposed_raw | sentence_score | 0.1536 | (0.1134, 0.1911) | 0.0000 | 0.1536 | (0.0795, 0.2250) | 0.0000 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0186 | (0.0033, 0.0345) | 0.0090 | 0.0186 | (0.0009, 0.0372) | 0.0180 |
| controlled_vs_proposed_raw | overall_quality | 0.1526 | (0.1338, 0.1726) | 0.0000 | 0.1526 | (0.1188, 0.1898) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2484 | (0.2275, 0.2690) | 0.0000 | 0.2484 | (0.2218, 0.2726) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1662 | (0.1288, 0.2060) | 0.0000 | 0.1662 | (0.0933, 0.2370) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.0428 | (0.0211, 0.0650) | 0.0000 | 0.0428 | (-0.0041, 0.0896) | 0.0350 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3244 | (0.2985, 0.3506) | 0.0000 | 0.3244 | (0.2890, 0.3554) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0709 | (0.0623, 0.0811) | 0.0000 | 0.0709 | (0.0602, 0.0839) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1866 | (0.1450, 0.2314) | 0.0000 | 0.1866 | (0.1010, 0.2710) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.0845 | (0.0453, 0.1276) | 0.0000 | 0.0845 | (-0.0032, 0.1892) | 0.0390 |
| controlled_vs_candidate_no_context | distinct1 | -0.0052 | (-0.0153, 0.0045) | 0.8537 | -0.0052 | (-0.0218, 0.0100) | 0.7317 |
| controlled_vs_candidate_no_context | length_score | 0.1667 | (0.0824, 0.2509) | 0.0000 | 0.1667 | (0.0101, 0.3286) | 0.0163 |
| controlled_vs_candidate_no_context | sentence_score | 0.1161 | (0.0723, 0.1594) | 0.0000 | 0.1161 | (0.0321, 0.2062) | 0.0017 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0527 | (0.0393, 0.0661) | 0.0000 | 0.0527 | (0.0316, 0.0735) | 0.0000 |
| controlled_vs_candidate_no_context | overall_quality | 0.1687 | (0.1532, 0.1843) | 0.0000 | 0.1687 | (0.1334, 0.1994) | 0.0000 |
| controlled_vs_baseline_no_context | context_relevance | 0.2286 | (0.2095, 0.2490) | 0.0000 | 0.2286 | (0.2096, 0.2502) | 0.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1652 | (0.1338, 0.1995) | 0.0000 | 0.1652 | (0.1171, 0.2132) | 0.0000 |
| controlled_vs_baseline_no_context | naturalness | -0.0214 | (-0.0410, -0.0016) | 0.9823 | -0.0214 | (-0.0587, 0.0125) | 0.8847 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.2977 | (0.2717, 0.3249) | 0.0000 | 0.2977 | (0.2718, 0.3276) | 0.0000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0675 | (0.0582, 0.0779) | 0.0000 | 0.0675 | (0.0611, 0.0765) | 0.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2023 | (0.1621, 0.2424) | 0.0000 | 0.2023 | (0.1372, 0.2617) | 0.0000 |
| controlled_vs_baseline_no_context | persona_style | 0.0169 | (-0.0038, 0.0394) | 0.0517 | 0.0169 | (-0.0148, 0.0567) | 0.1810 |
| controlled_vs_baseline_no_context | distinct1 | -0.0405 | (-0.0492, -0.0314) | 1.0000 | -0.0405 | (-0.0471, -0.0329) | 1.0000 |
| controlled_vs_baseline_no_context | length_score | -0.0348 | (-0.1182, 0.0485) | 0.7903 | -0.0348 | (-0.2066, 0.1203) | 0.6430 |
| controlled_vs_baseline_no_context | sentence_score | 0.0192 | (-0.0187, 0.0594) | 0.1717 | 0.0192 | (-0.0487, 0.0844) | 0.2663 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0278 | (0.0121, 0.0435) | 0.0007 | 0.0278 | (0.0089, 0.0471) | 0.0003 |
| controlled_vs_baseline_no_context | overall_quality | 0.1466 | (0.1333, 0.1603) | 0.0000 | 0.1466 | (0.1254, 0.1665) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2286 | (0.2084, 0.2488) | 0.0000 | 0.2286 | (0.2101, 0.2503) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1652 | (0.1331, 0.1983) | 0.0000 | 0.1652 | (0.1173, 0.2132) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0214 | (-0.0416, -0.0012) | 0.9827 | -0.0214 | (-0.0577, 0.0142) | 0.8733 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.2977 | (0.2710, 0.3253) | 0.0000 | 0.2977 | (0.2709, 0.3293) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0675 | (0.0582, 0.0781) | 0.0000 | 0.0675 | (0.0609, 0.0759) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2023 | (0.1632, 0.2443) | 0.0000 | 0.2023 | (0.1389, 0.2615) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0169 | (-0.0037, 0.0392) | 0.0647 | 0.0169 | (-0.0146, 0.0537) | 0.1657 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0405 | (-0.0492, -0.0311) | 1.0000 | -0.0405 | (-0.0469, -0.0331) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | -0.0348 | (-0.1217, 0.0500) | 0.7917 | -0.0348 | (-0.2092, 0.1256) | 0.6320 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0192 | (-0.0210, 0.0563) | 0.1743 | 0.0192 | (-0.0545, 0.0875) | 0.2887 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0278 | (0.0131, 0.0432) | 0.0003 | 0.0278 | (0.0096, 0.0475) | 0.0003 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1466 | (0.1333, 0.1606) | 0.0000 | 0.1466 | (0.1246, 0.1672) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 50 | 24 | 38 | 0.6161 | 0.6757 |
| proposed_vs_candidate_no_context | persona_consistency | 24 | 27 | 61 | 0.4866 | 0.4706 |
| proposed_vs_candidate_no_context | naturalness | 30 | 44 | 38 | 0.4375 | 0.4054 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 42 | 11 | 59 | 0.6384 | 0.7925 |
| proposed_vs_candidate_no_context | context_overlap | 53 | 21 | 38 | 0.6429 | 0.7162 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 16 | 21 | 75 | 0.4777 | 0.4324 |
| proposed_vs_candidate_no_context | persona_style | 10 | 16 | 86 | 0.4732 | 0.3846 |
| proposed_vs_candidate_no_context | distinct1 | 28 | 40 | 44 | 0.4464 | 0.4118 |
| proposed_vs_candidate_no_context | length_score | 24 | 43 | 45 | 0.4152 | 0.3582 |
| proposed_vs_candidate_no_context | sentence_score | 12 | 24 | 76 | 0.4464 | 0.3333 |
| proposed_vs_candidate_no_context | bertscore_f1 | 62 | 18 | 32 | 0.6964 | 0.7750 |
| proposed_vs_candidate_no_context | overall_quality | 55 | 25 | 32 | 0.6339 | 0.6875 |
| proposed_vs_baseline_no_context | context_relevance | 58 | 54 | 0 | 0.5179 | 0.5179 |
| proposed_vs_baseline_no_context | persona_consistency | 26 | 44 | 42 | 0.4196 | 0.3714 |
| proposed_vs_baseline_no_context | naturalness | 14 | 97 | 1 | 0.1295 | 0.1261 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 38 | 33 | 41 | 0.5223 | 0.5352 |
| proposed_vs_baseline_no_context | context_overlap | 67 | 45 | 0 | 0.5982 | 0.5982 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 22 | 26 | 64 | 0.4821 | 0.4583 |
| proposed_vs_baseline_no_context | persona_style | 7 | 29 | 76 | 0.4018 | 0.1944 |
| proposed_vs_baseline_no_context | distinct1 | 17 | 83 | 12 | 0.2054 | 0.1700 |
| proposed_vs_baseline_no_context | length_score | 19 | 93 | 0 | 0.1696 | 0.1696 |
| proposed_vs_baseline_no_context | sentence_score | 8 | 51 | 53 | 0.3080 | 0.1356 |
| proposed_vs_baseline_no_context | bertscore_f1 | 58 | 54 | 0 | 0.5179 | 0.5179 |
| proposed_vs_baseline_no_context | overall_quality | 46 | 66 | 0 | 0.4107 | 0.4107 |
| controlled_vs_proposed_raw | context_relevance | 104 | 8 | 0 | 0.9286 | 0.9286 |
| controlled_vs_proposed_raw | persona_consistency | 87 | 3 | 22 | 0.8750 | 0.9667 |
| controlled_vs_proposed_raw | naturalness | 80 | 32 | 0 | 0.7143 | 0.7143 |
| controlled_vs_proposed_raw | context_keyword_coverage | 97 | 8 | 7 | 0.8973 | 0.9238 |
| controlled_vs_proposed_raw | context_overlap | 101 | 11 | 0 | 0.9018 | 0.9018 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 86 | 3 | 23 | 0.8705 | 0.9663 |
| controlled_vs_proposed_raw | persona_style | 34 | 9 | 69 | 0.6116 | 0.7907 |
| controlled_vs_proposed_raw | distinct1 | 56 | 52 | 4 | 0.5179 | 0.5185 |
| controlled_vs_proposed_raw | length_score | 73 | 32 | 7 | 0.6830 | 0.6952 |
| controlled_vs_proposed_raw | sentence_score | 56 | 6 | 50 | 0.7232 | 0.9032 |
| controlled_vs_proposed_raw | bertscore_f1 | 70 | 42 | 0 | 0.6250 | 0.6250 |
| controlled_vs_proposed_raw | overall_quality | 102 | 10 | 0 | 0.9107 | 0.9107 |
| controlled_vs_candidate_no_context | context_relevance | 111 | 1 | 0 | 0.9911 | 0.9911 |
| controlled_vs_candidate_no_context | persona_consistency | 85 | 12 | 15 | 0.8259 | 0.8763 |
| controlled_vs_candidate_no_context | naturalness | 70 | 42 | 0 | 0.6250 | 0.6250 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 107 | 0 | 5 | 0.9777 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 111 | 1 | 0 | 0.9911 | 0.9911 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 83 | 11 | 18 | 0.8214 | 0.8830 |
| controlled_vs_candidate_no_context | persona_style | 31 | 13 | 68 | 0.5804 | 0.7045 |
| controlled_vs_candidate_no_context | distinct1 | 50 | 57 | 5 | 0.4688 | 0.4673 |
| controlled_vs_candidate_no_context | length_score | 66 | 42 | 4 | 0.6071 | 0.6111 |
| controlled_vs_candidate_no_context | sentence_score | 50 | 12 | 50 | 0.6696 | 0.8065 |
| controlled_vs_candidate_no_context | bertscore_f1 | 88 | 24 | 0 | 0.7857 | 0.7857 |
| controlled_vs_candidate_no_context | overall_quality | 109 | 3 | 0 | 0.9732 | 0.9732 |
| controlled_vs_baseline_no_context | context_relevance | 111 | 1 | 0 | 0.9911 | 0.9911 |
| controlled_vs_baseline_no_context | persona_consistency | 91 | 10 | 11 | 0.8616 | 0.9010 |
| controlled_vs_baseline_no_context | naturalness | 53 | 59 | 0 | 0.4732 | 0.4732 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 106 | 1 | 5 | 0.9688 | 0.9907 |
| controlled_vs_baseline_no_context | context_overlap | 108 | 4 | 0 | 0.9643 | 0.9643 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 90 | 9 | 13 | 0.8616 | 0.9091 |
| controlled_vs_baseline_no_context | persona_style | 15 | 12 | 85 | 0.5134 | 0.5556 |
| controlled_vs_baseline_no_context | distinct1 | 17 | 87 | 8 | 0.1875 | 0.1635 |
| controlled_vs_baseline_no_context | length_score | 53 | 57 | 2 | 0.4821 | 0.4818 |
| controlled_vs_baseline_no_context | sentence_score | 23 | 16 | 73 | 0.5312 | 0.5897 |
| controlled_vs_baseline_no_context | bertscore_f1 | 70 | 42 | 0 | 0.6250 | 0.6250 |
| controlled_vs_baseline_no_context | overall_quality | 111 | 1 | 0 | 0.9911 | 0.9911 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 111 | 1 | 0 | 0.9911 | 0.9911 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 91 | 10 | 11 | 0.8616 | 0.9010 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 53 | 59 | 0 | 0.4732 | 0.4732 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 106 | 1 | 5 | 0.9688 | 0.9907 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 108 | 4 | 0 | 0.9643 | 0.9643 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 90 | 9 | 13 | 0.8616 | 0.9091 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 15 | 12 | 85 | 0.5134 | 0.5556 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | 17 | 87 | 8 | 0.1875 | 0.1635 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 53 | 57 | 2 | 0.4821 | 0.4818 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 23 | 16 | 73 | 0.5312 | 0.5897 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 70 | 42 | 0 | 0.6250 | 0.6250 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 111 | 1 | 0 | 0.9911 | 0.9911 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.2679 | 0.5089 | 0.4911 |
| proposed_contextual | 0.0000 | 0.0000 | 0.4732 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.4554 | 0.0000 | 0.0000 |
| baseline_no_context | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `8`
- Unique template signatures: `71`
- Template signature ratio: `0.6339`
- Effective sample size by source clustering: `8.00`
- Effective sample size by template-signature clustering: `56.00`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: enabled.

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
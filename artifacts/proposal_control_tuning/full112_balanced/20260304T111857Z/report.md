# Proposal Alignment Evaluation Report

- Run ID: `20260304T111857Z`
- Generated: `2026-03-04T11:43:05.486791+00:00`
- Scenarios: `artifacts\proposal_control_tuning\full112_balanced\20260304T111857Z\scenarios.jsonl`
- Scenario count: `112`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off
- `baseline_no_context`: model `phi3:mini`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2530 (0.2400, 0.2678) | 0.3672 (0.3399, 0.3973) | 0.9093 (0.8964, 0.9209) | 0.3891 (0.3790, 0.4000) | 0.0787 |
| proposed_contextual | 0.0882 (0.0708, 0.1087) | 0.1360 (0.1156, 0.1588) | 0.8000 (0.7872, 0.8135) | 0.2257 (0.2116, 0.2401) | 0.0680 |
| candidate_no_context | 0.0312 (0.0246, 0.0391) | 0.1741 (0.1459, 0.2032) | 0.8130 (0.7995, 0.8272) | 0.2143 (0.2028, 0.2264) | 0.0414 |
| baseline_no_context | 0.0487 (0.0384, 0.0585) | 0.1916 (0.1711, 0.2142) | 0.8819 (0.8709, 0.8931) | 0.2413 (0.2320, 0.2507) | 0.0635 |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0570 | 1.8252 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0380 | -0.2184 |
| proposed_vs_candidate_no_context | naturalness | -0.0130 | -0.0159 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0749 | 2.7647 |
| proposed_vs_candidate_no_context | context_overlap | 0.0153 | 0.3739 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0467 | -0.5187 |
| proposed_vs_candidate_no_context | persona_style | -0.0033 | -0.0066 |
| proposed_vs_candidate_no_context | distinct1 | -0.0041 | -0.0043 |
| proposed_vs_candidate_no_context | length_score | -0.0518 | -0.1684 |
| proposed_vs_candidate_no_context | sentence_score | -0.0098 | -0.0129 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0266 | 0.6420 |
| proposed_vs_candidate_no_context | overall_quality | 0.0114 | 0.0530 |
| proposed_vs_baseline_no_context | context_relevance | 0.0396 | 0.8132 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0556 | -0.2901 |
| proposed_vs_baseline_no_context | naturalness | -0.0819 | -0.0928 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0510 | 1.0003 |
| proposed_vs_baseline_no_context | context_overlap | 0.0129 | 0.2990 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0474 | -0.5227 |
| proposed_vs_baseline_no_context | persona_style | -0.0881 | -0.1481 |
| proposed_vs_baseline_no_context | distinct1 | -0.0470 | -0.0479 |
| proposed_vs_baseline_no_context | length_score | -0.2560 | -0.5003 |
| proposed_vs_baseline_no_context | sentence_score | -0.1188 | -0.1362 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0045 | 0.0710 |
| proposed_vs_baseline_no_context | overall_quality | -0.0156 | -0.0647 |
| controlled_vs_proposed_raw | context_relevance | 0.1648 | 1.8673 |
| controlled_vs_proposed_raw | persona_consistency | 0.2312 | 1.6992 |
| controlled_vs_proposed_raw | naturalness | 0.1093 | 0.1366 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2150 | 2.1085 |
| controlled_vs_proposed_raw | context_overlap | 0.0476 | 0.8464 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2671 | 6.1658 |
| controlled_vs_proposed_raw | persona_style | 0.0873 | 0.1722 |
| controlled_vs_proposed_raw | distinct1 | -0.0028 | -0.0030 |
| controlled_vs_proposed_raw | length_score | 0.4491 | 1.7567 |
| controlled_vs_proposed_raw | sentence_score | 0.2219 | 0.2946 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0107 | 0.1575 |
| controlled_vs_proposed_raw | overall_quality | 0.1634 | 0.7238 |
| controlled_vs_candidate_no_context | context_relevance | 0.2218 | 7.1009 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1932 | 1.1097 |
| controlled_vs_candidate_no_context | naturalness | 0.0963 | 0.1185 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.2899 | 10.7028 |
| controlled_vs_candidate_no_context | context_overlap | 0.0629 | 1.5368 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2205 | 2.4492 |
| controlled_vs_candidate_no_context | persona_style | 0.0840 | 0.1646 |
| controlled_vs_candidate_no_context | distinct1 | -0.0068 | -0.0073 |
| controlled_vs_candidate_no_context | length_score | 0.3973 | 1.2924 |
| controlled_vs_candidate_no_context | sentence_score | 0.2121 | 0.2779 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0373 | 0.9006 |
| controlled_vs_candidate_no_context | overall_quality | 0.1747 | 0.8151 |
| controlled_vs_baseline_no_context | context_relevance | 0.2043 | 4.1991 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1756 | 0.9163 |
| controlled_vs_baseline_no_context | naturalness | 0.0274 | 0.0311 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.2660 | 5.2179 |
| controlled_vs_baseline_no_context | context_overlap | 0.0605 | 1.3985 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2197 | 2.4201 |
| controlled_vs_baseline_no_context | persona_style | -0.0008 | -0.0013 |
| controlled_vs_baseline_no_context | distinct1 | -0.0498 | -0.0507 |
| controlled_vs_baseline_no_context | length_score | 0.1932 | 0.3775 |
| controlled_vs_baseline_no_context | sentence_score | 0.1031 | 0.1183 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0152 | 0.2397 |
| controlled_vs_baseline_no_context | overall_quality | 0.1477 | 0.6122 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2043 | 4.1991 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1756 | 0.9163 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 0.0274 | 0.0311 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.2660 | 5.2179 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0605 | 1.3985 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2197 | 2.4201 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | -0.0008 | -0.0013 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0498 | -0.0507 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.1932 | 0.3775 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.1031 | 0.1183 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0152 | 0.2397 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1477 | 0.6122 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0570 | (0.0385, 0.0770) | 0.0000 | 0.0570 | (0.0228, 0.0975) | 0.0000 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0380 | (-0.0694, -0.0096) | 0.9947 | -0.0380 | (-0.1028, 0.0030) | 0.9470 |
| proposed_vs_candidate_no_context | naturalness | -0.0130 | (-0.0293, 0.0049) | 0.9230 | -0.0130 | (-0.0248, -0.0013) | 0.9867 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0749 | (0.0489, 0.1010) | 0.0000 | 0.0749 | (0.0308, 0.1267) | 0.0000 |
| proposed_vs_candidate_no_context | context_overlap | 0.0153 | (0.0095, 0.0214) | 0.0000 | 0.0153 | (0.0026, 0.0273) | 0.0043 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0467 | (-0.0837, -0.0122) | 0.9983 | -0.0467 | (-0.1277, 0.0040) | 0.9450 |
| proposed_vs_candidate_no_context | persona_style | -0.0033 | (-0.0411, 0.0336) | 0.5543 | -0.0033 | (-0.0387, 0.0274) | 0.5847 |
| proposed_vs_candidate_no_context | distinct1 | -0.0041 | (-0.0117, 0.0043) | 0.8320 | -0.0041 | (-0.0107, 0.0031) | 0.8753 |
| proposed_vs_candidate_no_context | length_score | -0.0518 | (-0.1164, 0.0116) | 0.9473 | -0.0518 | (-0.1021, -0.0071) | 0.9873 |
| proposed_vs_candidate_no_context | sentence_score | -0.0098 | (-0.0500, 0.0277) | 0.7140 | -0.0098 | (-0.0687, 0.0612) | 0.6380 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0266 | (0.0161, 0.0374) | 0.0000 | 0.0266 | (0.0124, 0.0427) | 0.0000 |
| proposed_vs_candidate_no_context | overall_quality | 0.0114 | (-0.0058, 0.0289) | 0.0950 | 0.0114 | (-0.0208, 0.0395) | 0.2240 |
| proposed_vs_baseline_no_context | context_relevance | 0.0396 | (0.0176, 0.0619) | 0.0000 | 0.0396 | (-0.0012, 0.0910) | 0.0317 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0556 | (-0.0823, -0.0284) | 1.0000 | -0.0556 | (-0.1027, -0.0033) | 0.9803 |
| proposed_vs_baseline_no_context | naturalness | -0.0819 | (-0.0975, -0.0659) | 1.0000 | -0.0819 | (-0.1058, -0.0534) | 1.0000 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0510 | (0.0225, 0.0800) | 0.0000 | 0.0510 | (-0.0047, 0.1196) | 0.0397 |
| proposed_vs_baseline_no_context | context_overlap | 0.0129 | (0.0066, 0.0200) | 0.0003 | 0.0129 | (0.0014, 0.0254) | 0.0150 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0474 | (-0.0761, -0.0192) | 0.9993 | -0.0474 | (-0.0911, 0.0075) | 0.9550 |
| proposed_vs_baseline_no_context | persona_style | -0.0881 | (-0.1313, -0.0480) | 1.0000 | -0.0881 | (-0.2108, 0.0055) | 0.9447 |
| proposed_vs_baseline_no_context | distinct1 | -0.0470 | (-0.0547, -0.0392) | 1.0000 | -0.0470 | (-0.0547, -0.0390) | 1.0000 |
| proposed_vs_baseline_no_context | length_score | -0.2560 | (-0.3179, -0.1946) | 1.0000 | -0.2560 | (-0.3554, -0.1333) | 1.0000 |
| proposed_vs_baseline_no_context | sentence_score | -0.1187 | (-0.1594, -0.0812) | 1.0000 | -0.1187 | (-0.1938, -0.0500) | 1.0000 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0045 | (-0.0087, 0.0183) | 0.2463 | 0.0045 | (-0.0218, 0.0325) | 0.3670 |
| proposed_vs_baseline_no_context | overall_quality | -0.0156 | (-0.0329, 0.0016) | 0.9617 | -0.0156 | (-0.0506, 0.0244) | 0.7953 |
| controlled_vs_proposed_raw | context_relevance | 0.1648 | (0.1462, 0.1831) | 0.0000 | 0.1648 | (0.1358, 0.1939) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.2312 | (0.1994, 0.2610) | 0.0000 | 0.2312 | (0.1907, 0.2776) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.1093 | (0.0921, 0.1258) | 0.0000 | 0.1093 | (0.0805, 0.1368) | 0.0000 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2150 | (0.1907, 0.2392) | 0.0000 | 0.2150 | (0.1740, 0.2538) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0476 | (0.0383, 0.0573) | 0.0000 | 0.0476 | (0.0363, 0.0601) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2671 | (0.2305, 0.3063) | 0.0000 | 0.2671 | (0.2193, 0.3303) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.0873 | (0.0498, 0.1284) | 0.0000 | 0.0873 | (0.0032, 0.2100) | 0.0193 |
| controlled_vs_proposed_raw | distinct1 | -0.0028 | (-0.0129, 0.0063) | 0.7053 | -0.0028 | (-0.0160, 0.0087) | 0.6760 |
| controlled_vs_proposed_raw | length_score | 0.4491 | (0.3845, 0.5131) | 0.0000 | 0.4491 | (0.3491, 0.5464) | 0.0000 |
| controlled_vs_proposed_raw | sentence_score | 0.2219 | (0.1875, 0.2562) | 0.0000 | 0.2219 | (0.1531, 0.2812) | 0.0000 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0107 | (-0.0025, 0.0235) | 0.0503 | 0.0107 | (-0.0175, 0.0362) | 0.2120 |
| controlled_vs_proposed_raw | overall_quality | 0.1634 | (0.1479, 0.1791) | 0.0000 | 0.1634 | (0.1349, 0.1905) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2218 | (0.2075, 0.2369) | 0.0000 | 0.2218 | (0.2077, 0.2376) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1932 | (0.1542, 0.2316) | 0.0000 | 0.1932 | (0.1198, 0.2571) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.0963 | (0.0800, 0.1124) | 0.0000 | 0.0963 | (0.0647, 0.1278) | 0.0000 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.2899 | (0.2708, 0.3085) | 0.0000 | 0.2899 | (0.2718, 0.3098) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0629 | (0.0544, 0.0725) | 0.0000 | 0.0629 | (0.0543, 0.0719) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2205 | (0.1744, 0.2655) | 0.0000 | 0.2205 | (0.1358, 0.3047) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.0840 | (0.0458, 0.1252) | 0.0000 | 0.0840 | (0.0055, 0.1822) | 0.0240 |
| controlled_vs_candidate_no_context | distinct1 | -0.0068 | (-0.0164, 0.0024) | 0.9290 | -0.0068 | (-0.0204, 0.0080) | 0.8107 |
| controlled_vs_candidate_no_context | length_score | 0.3973 | (0.3393, 0.4577) | 0.0000 | 0.3973 | (0.2908, 0.5071) | 0.0000 |
| controlled_vs_candidate_no_context | sentence_score | 0.2121 | (0.1781, 0.2460) | 0.0000 | 0.2121 | (0.1469, 0.2737) | 0.0000 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0373 | (0.0258, 0.0488) | 0.0000 | 0.0373 | (0.0156, 0.0559) | 0.0003 |
| controlled_vs_candidate_no_context | overall_quality | 0.1747 | (0.1593, 0.1900) | 0.0000 | 0.1747 | (0.1488, 0.1982) | 0.0000 |
| controlled_vs_baseline_no_context | context_relevance | 0.2043 | (0.1875, 0.2210) | 0.0000 | 0.2043 | (0.1763, 0.2353) | 0.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1756 | (0.1420, 0.2100) | 0.0000 | 0.1756 | (0.1337, 0.2239) | 0.0000 |
| controlled_vs_baseline_no_context | naturalness | 0.0274 | (0.0106, 0.0440) | 0.0013 | 0.0274 | (0.0013, 0.0524) | 0.0217 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.2660 | (0.2441, 0.2884) | 0.0000 | 0.2660 | (0.2275, 0.3077) | 0.0000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0605 | (0.0522, 0.0706) | 0.0000 | 0.0605 | (0.0523, 0.0689) | 0.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2197 | (0.1794, 0.2598) | 0.0000 | 0.2197 | (0.1661, 0.2808) | 0.0000 |
| controlled_vs_baseline_no_context | persona_style | -0.0008 | (-0.0215, 0.0204) | 0.5393 | -0.0008 | (-0.0233, 0.0274) | 0.5560 |
| controlled_vs_baseline_no_context | distinct1 | -0.0498 | (-0.0595, -0.0405) | 1.0000 | -0.0498 | (-0.0610, -0.0381) | 1.0000 |
| controlled_vs_baseline_no_context | length_score | 0.1932 | (0.1268, 0.2545) | 0.0000 | 0.1932 | (0.0875, 0.3083) | 0.0000 |
| controlled_vs_baseline_no_context | sentence_score | 0.1031 | (0.0687, 0.1375) | 0.0000 | 0.1031 | (0.0406, 0.1687) | 0.0000 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0152 | (0.0026, 0.0276) | 0.0100 | 0.0152 | (-0.0112, 0.0382) | 0.1283 |
| controlled_vs_baseline_no_context | overall_quality | 0.1477 | (0.1348, 0.1604) | 0.0000 | 0.1477 | (0.1288, 0.1678) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2043 | (0.1884, 0.2210) | 0.0000 | 0.2043 | (0.1762, 0.2331) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1756 | (0.1437, 0.2101) | 0.0000 | 0.1756 | (0.1337, 0.2250) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 0.0274 | (0.0104, 0.0433) | 0.0003 | 0.0274 | (0.0026, 0.0542) | 0.0160 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.2660 | (0.2455, 0.2876) | 0.0000 | 0.2660 | (0.2283, 0.3072) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0605 | (0.0519, 0.0703) | 0.0000 | 0.0605 | (0.0521, 0.0684) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2197 | (0.1795, 0.2602) | 0.0000 | 0.2197 | (0.1697, 0.2800) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | -0.0008 | (-0.0207, 0.0196) | 0.5183 | -0.0008 | (-0.0229, 0.0264) | 0.5637 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0498 | (-0.0594, -0.0405) | 1.0000 | -0.0498 | (-0.0610, -0.0382) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.1932 | (0.1274, 0.2586) | 0.0000 | 0.1932 | (0.0875, 0.3057) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.1031 | (0.0687, 0.1375) | 0.0000 | 0.1031 | (0.0437, 0.1687) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0152 | (0.0025, 0.0278) | 0.0087 | 0.0152 | (-0.0109, 0.0388) | 0.1173 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1477 | (0.1350, 0.1605) | 0.0000 | 0.1477 | (0.1278, 0.1687) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 51 | 21 | 40 | 0.6339 | 0.7083 |
| proposed_vs_candidate_no_context | persona_consistency | 18 | 31 | 63 | 0.4420 | 0.3673 |
| proposed_vs_candidate_no_context | naturalness | 27 | 45 | 40 | 0.4196 | 0.3750 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 44 | 11 | 57 | 0.6473 | 0.8000 |
| proposed_vs_candidate_no_context | context_overlap | 55 | 17 | 40 | 0.6696 | 0.7639 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 10 | 22 | 80 | 0.4464 | 0.3125 |
| proposed_vs_candidate_no_context | persona_style | 13 | 14 | 85 | 0.4955 | 0.4815 |
| proposed_vs_candidate_no_context | distinct1 | 27 | 39 | 46 | 0.4464 | 0.4091 |
| proposed_vs_candidate_no_context | length_score | 27 | 42 | 43 | 0.4330 | 0.3913 |
| proposed_vs_candidate_no_context | sentence_score | 18 | 21 | 73 | 0.4866 | 0.4615 |
| proposed_vs_candidate_no_context | bertscore_f1 | 68 | 21 | 23 | 0.7098 | 0.7640 |
| proposed_vs_candidate_no_context | overall_quality | 54 | 35 | 23 | 0.5848 | 0.6067 |
| proposed_vs_baseline_no_context | context_relevance | 60 | 52 | 0 | 0.5357 | 0.5357 |
| proposed_vs_baseline_no_context | persona_consistency | 17 | 53 | 42 | 0.3393 | 0.2429 |
| proposed_vs_baseline_no_context | naturalness | 22 | 89 | 1 | 0.2009 | 0.1982 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 40 | 25 | 47 | 0.5670 | 0.6154 |
| proposed_vs_baseline_no_context | context_overlap | 65 | 47 | 0 | 0.5804 | 0.5804 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 12 | 39 | 61 | 0.3795 | 0.2353 |
| proposed_vs_baseline_no_context | persona_style | 8 | 29 | 75 | 0.4062 | 0.2162 |
| proposed_vs_baseline_no_context | distinct1 | 8 | 88 | 16 | 0.1429 | 0.0833 |
| proposed_vs_baseline_no_context | length_score | 24 | 83 | 5 | 0.2366 | 0.2243 |
| proposed_vs_baseline_no_context | sentence_score | 8 | 46 | 58 | 0.3304 | 0.1481 |
| proposed_vs_baseline_no_context | bertscore_f1 | 61 | 51 | 0 | 0.5446 | 0.5446 |
| proposed_vs_baseline_no_context | overall_quality | 44 | 68 | 0 | 0.3929 | 0.3929 |
| controlled_vs_proposed_raw | context_relevance | 102 | 10 | 0 | 0.9107 | 0.9107 |
| controlled_vs_proposed_raw | persona_consistency | 97 | 4 | 11 | 0.9152 | 0.9604 |
| controlled_vs_proposed_raw | naturalness | 95 | 17 | 0 | 0.8482 | 0.8482 |
| controlled_vs_proposed_raw | context_keyword_coverage | 96 | 5 | 11 | 0.9062 | 0.9505 |
| controlled_vs_proposed_raw | context_overlap | 98 | 14 | 0 | 0.8750 | 0.8750 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 97 | 3 | 12 | 0.9196 | 0.9700 |
| controlled_vs_proposed_raw | persona_style | 27 | 3 | 82 | 0.6071 | 0.9000 |
| controlled_vs_proposed_raw | distinct1 | 62 | 48 | 2 | 0.5625 | 0.5636 |
| controlled_vs_proposed_raw | length_score | 93 | 17 | 2 | 0.8393 | 0.8455 |
| controlled_vs_proposed_raw | sentence_score | 74 | 3 | 35 | 0.8170 | 0.9610 |
| controlled_vs_proposed_raw | bertscore_f1 | 74 | 38 | 0 | 0.6607 | 0.6607 |
| controlled_vs_proposed_raw | overall_quality | 108 | 4 | 0 | 0.9643 | 0.9643 |
| controlled_vs_candidate_no_context | context_relevance | 111 | 1 | 0 | 0.9911 | 0.9911 |
| controlled_vs_candidate_no_context | persona_consistency | 94 | 13 | 5 | 0.8616 | 0.8785 |
| controlled_vs_candidate_no_context | naturalness | 100 | 12 | 0 | 0.8929 | 0.8929 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 110 | 0 | 2 | 0.9911 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 109 | 3 | 0 | 0.9732 | 0.9732 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 93 | 13 | 6 | 0.8571 | 0.8774 |
| controlled_vs_candidate_no_context | persona_style | 28 | 6 | 78 | 0.5982 | 0.8235 |
| controlled_vs_candidate_no_context | distinct1 | 52 | 57 | 3 | 0.4777 | 0.4771 |
| controlled_vs_candidate_no_context | length_score | 95 | 13 | 4 | 0.8661 | 0.8796 |
| controlled_vs_candidate_no_context | sentence_score | 68 | 1 | 43 | 0.7991 | 0.9855 |
| controlled_vs_candidate_no_context | bertscore_f1 | 90 | 22 | 0 | 0.8036 | 0.8036 |
| controlled_vs_candidate_no_context | overall_quality | 109 | 3 | 0 | 0.9732 | 0.9732 |
| controlled_vs_baseline_no_context | context_relevance | 112 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 89 | 11 | 12 | 0.8482 | 0.8900 |
| controlled_vs_baseline_no_context | naturalness | 74 | 37 | 1 | 0.6652 | 0.6667 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 109 | 0 | 3 | 0.9866 | 1.0000 |
| controlled_vs_baseline_no_context | context_overlap | 109 | 3 | 0 | 0.9732 | 0.9732 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 87 | 7 | 18 | 0.8571 | 0.9255 |
| controlled_vs_baseline_no_context | persona_style | 10 | 10 | 92 | 0.5000 | 0.5000 |
| controlled_vs_baseline_no_context | distinct1 | 12 | 98 | 2 | 0.1161 | 0.1091 |
| controlled_vs_baseline_no_context | length_score | 81 | 27 | 4 | 0.7411 | 0.7500 |
| controlled_vs_baseline_no_context | sentence_score | 37 | 4 | 71 | 0.6473 | 0.9024 |
| controlled_vs_baseline_no_context | bertscore_f1 | 63 | 49 | 0 | 0.5625 | 0.5625 |
| controlled_vs_baseline_no_context | overall_quality | 111 | 1 | 0 | 0.9911 | 0.9911 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 112 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 89 | 11 | 12 | 0.8482 | 0.8900 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 74 | 37 | 1 | 0.6652 | 0.6667 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 109 | 0 | 3 | 0.9866 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 109 | 3 | 0 | 0.9732 | 0.9732 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 87 | 7 | 18 | 0.8571 | 0.9255 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 10 | 10 | 92 | 0.5000 | 0.5000 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | 12 | 98 | 2 | 0.1161 | 0.1091 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 81 | 27 | 4 | 0.7411 | 0.7500 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 37 | 4 | 71 | 0.6473 | 0.9024 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 63 | 49 | 0 | 0.5625 | 0.5625 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 111 | 1 | 0 | 0.9911 | 0.9911 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.7500 | 0.0982 | 0.9018 |
| proposed_contextual | 0.0000 | 0.0000 | 0.5268 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.4821 | 0.0000 | 0.0000 |
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
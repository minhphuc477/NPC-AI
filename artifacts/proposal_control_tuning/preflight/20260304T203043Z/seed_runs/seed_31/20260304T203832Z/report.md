# Proposal Alignment Evaluation Report

- Run ID: `20260304T203832Z`
- Generated: `2026-03-04T20:47:41.059218+00:00`
- Scenarios: `artifacts\proposal_control_tuning\preflight\20260304T203043Z\seed_runs\seed_31\20260304T203832Z\scenarios.jsonl`
- Scenario count: `40`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off
- `baseline_no_context`: model `phi3:mini`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2895 (0.2582, 0.3243) | 0.3569 (0.3118, 0.4065) | 0.8759 (0.8560, 0.8944) | 0.3956 (0.3755, 0.4164) | 0.0877 |
| proposed_contextual | 0.1112 (0.0707, 0.1556) | 0.1798 (0.1343, 0.2304) | 0.8110 (0.7892, 0.8356) | 0.2511 (0.2237, 0.2805) | 0.0662 |
| candidate_no_context | 0.0271 (0.0167, 0.0394) | 0.2196 (0.1622, 0.2835) | 0.8070 (0.7862, 0.8282) | 0.2264 (0.2041, 0.2524) | 0.0417 |
| baseline_no_context | 0.0482 (0.0340, 0.0644) | 0.2163 (0.1672, 0.2762) | 0.8875 (0.8663, 0.9073) | 0.2503 (0.2321, 0.2711) | 0.0585 |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0841 | 3.1068 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0398 | -0.1811 |
| proposed_vs_candidate_no_context | naturalness | 0.0039 | 0.0049 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.1109 | 4.9983 |
| proposed_vs_candidate_no_context | context_overlap | 0.0214 | 0.5576 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0464 | -0.3333 |
| proposed_vs_candidate_no_context | persona_style | -0.0131 | -0.0243 |
| proposed_vs_candidate_no_context | distinct1 | 0.0065 | 0.0070 |
| proposed_vs_candidate_no_context | length_score | -0.0108 | -0.0371 |
| proposed_vs_candidate_no_context | sentence_score | 0.0350 | 0.0458 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0245 | 0.5881 |
| proposed_vs_candidate_no_context | overall_quality | 0.0247 | 0.1091 |
| proposed_vs_baseline_no_context | context_relevance | 0.0630 | 1.3082 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0365 | -0.1686 |
| proposed_vs_baseline_no_context | naturalness | -0.0765 | -0.0862 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0815 | 1.5789 |
| proposed_vs_baseline_no_context | context_overlap | 0.0198 | 0.4942 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0382 | -0.2916 |
| proposed_vs_baseline_no_context | persona_style | -0.0295 | -0.0529 |
| proposed_vs_baseline_no_context | distinct1 | -0.0350 | -0.0360 |
| proposed_vs_baseline_no_context | length_score | -0.2775 | -0.4970 |
| proposed_vs_baseline_no_context | sentence_score | -0.0700 | -0.0806 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0077 | 0.1313 |
| proposed_vs_baseline_no_context | overall_quality | 0.0008 | 0.0033 |
| controlled_vs_proposed_raw | context_relevance | 0.1784 | 1.6047 |
| controlled_vs_proposed_raw | persona_consistency | 0.1771 | 0.9846 |
| controlled_vs_proposed_raw | naturalness | 0.0650 | 0.0801 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2296 | 1.7246 |
| controlled_vs_proposed_raw | context_overlap | 0.0588 | 0.9826 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2096 | 2.2577 |
| controlled_vs_proposed_raw | persona_style | 0.0467 | 0.0885 |
| controlled_vs_proposed_raw | distinct1 | 0.0010 | 0.0010 |
| controlled_vs_proposed_raw | length_score | 0.2808 | 1.0000 |
| controlled_vs_proposed_raw | sentence_score | 0.0875 | 0.1095 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0216 | 0.3257 |
| controlled_vs_proposed_raw | overall_quality | 0.1445 | 0.5752 |
| controlled_vs_candidate_no_context | context_relevance | 0.2625 | 9.6971 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1373 | 0.6252 |
| controlled_vs_candidate_no_context | naturalness | 0.0689 | 0.0854 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3406 | 15.3430 |
| controlled_vs_candidate_no_context | context_overlap | 0.0803 | 2.0882 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1632 | 1.1718 |
| controlled_vs_candidate_no_context | persona_style | 0.0335 | 0.0620 |
| controlled_vs_candidate_no_context | distinct1 | 0.0075 | 0.0081 |
| controlled_vs_candidate_no_context | length_score | 0.2700 | 0.9257 |
| controlled_vs_candidate_no_context | sentence_score | 0.1225 | 0.1604 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0461 | 1.1053 |
| controlled_vs_candidate_no_context | overall_quality | 0.1691 | 0.7471 |
| controlled_vs_baseline_no_context | context_relevance | 0.2414 | 5.0122 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1406 | 0.6500 |
| controlled_vs_baseline_no_context | naturalness | -0.0115 | -0.0130 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3111 | 6.0264 |
| controlled_vs_baseline_no_context | context_overlap | 0.0786 | 1.9624 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1714 | 1.3079 |
| controlled_vs_baseline_no_context | persona_style | 0.0172 | 0.0309 |
| controlled_vs_baseline_no_context | distinct1 | -0.0340 | -0.0350 |
| controlled_vs_baseline_no_context | length_score | 0.0033 | 0.0060 |
| controlled_vs_baseline_no_context | sentence_score | 0.0175 | 0.0201 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0292 | 0.4998 |
| controlled_vs_baseline_no_context | overall_quality | 0.1453 | 0.5804 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2414 | 5.0122 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1406 | 0.6500 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0115 | -0.0130 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3111 | 6.0264 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0786 | 1.9624 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1714 | 1.3079 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0172 | 0.0309 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0340 | -0.0350 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.0033 | 0.0060 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0175 | 0.0201 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0292 | 0.4998 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1453 | 0.5804 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0841 | (0.0437, 0.1285) | 0.0000 | 0.0841 | (0.0225, 0.1552) | 0.0000 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0398 | (-0.0979, 0.0121) | 0.9197 | -0.0398 | (-0.0961, 0.0062) | 0.9330 |
| proposed_vs_candidate_no_context | naturalness | 0.0039 | (-0.0222, 0.0308) | 0.3930 | 0.0039 | (-0.0311, 0.0308) | 0.3550 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.1109 | (0.0546, 0.1710) | 0.0000 | 0.1109 | (0.0322, 0.2018) | 0.0000 |
| proposed_vs_candidate_no_context | context_overlap | 0.0214 | (0.0077, 0.0379) | 0.0003 | 0.0214 | (0.0036, 0.0421) | 0.0020 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0464 | (-0.1125, 0.0131) | 0.9300 | -0.0464 | (-0.1118, 0.0111) | 0.9290 |
| proposed_vs_candidate_no_context | persona_style | -0.0131 | (-0.0648, 0.0300) | 0.6927 | -0.0131 | (-0.0452, 0.0122) | 0.8440 |
| proposed_vs_candidate_no_context | distinct1 | 0.0065 | (-0.0072, 0.0206) | 0.1790 | 0.0065 | (-0.0070, 0.0189) | 0.1643 |
| proposed_vs_candidate_no_context | length_score | -0.0108 | (-0.1158, 0.0959) | 0.5807 | -0.0108 | (-0.1319, 0.1009) | 0.5240 |
| proposed_vs_candidate_no_context | sentence_score | 0.0350 | (-0.0262, 0.0962) | 0.1537 | 0.0350 | (-0.0262, 0.0824) | 0.1400 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0245 | (0.0010, 0.0464) | 0.0220 | 0.0245 | (-0.0029, 0.0486) | 0.0377 |
| proposed_vs_candidate_no_context | overall_quality | 0.0247 | (-0.0066, 0.0548) | 0.0583 | 0.0247 | (-0.0120, 0.0622) | 0.1213 |
| proposed_vs_baseline_no_context | context_relevance | 0.0630 | (0.0181, 0.1095) | 0.0023 | 0.0630 | (-0.0050, 0.1350) | 0.0413 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0365 | (-0.1035, 0.0287) | 0.8587 | -0.0365 | (-0.0886, 0.0098) | 0.9303 |
| proposed_vs_baseline_no_context | naturalness | -0.0765 | (-0.1129, -0.0400) | 0.9993 | -0.0765 | (-0.1257, -0.0272) | 1.0000 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0815 | (0.0232, 0.1458) | 0.0033 | 0.0815 | (-0.0092, 0.1720) | 0.0440 |
| proposed_vs_baseline_no_context | context_overlap | 0.0198 | (0.0043, 0.0374) | 0.0027 | 0.0198 | (0.0034, 0.0403) | 0.0047 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0382 | (-0.1222, 0.0406) | 0.8287 | -0.0382 | (-0.1043, 0.0139) | 0.9090 |
| proposed_vs_baseline_no_context | persona_style | -0.0295 | (-0.0926, 0.0274) | 0.8353 | -0.0295 | (-0.1518, 0.0445) | 0.7237 |
| proposed_vs_baseline_no_context | distinct1 | -0.0350 | (-0.0478, -0.0213) | 1.0000 | -0.0350 | (-0.0525, -0.0166) | 0.9997 |
| proposed_vs_baseline_no_context | length_score | -0.2775 | (-0.4133, -0.1292) | 0.9997 | -0.2775 | (-0.4564, -0.1037) | 0.9997 |
| proposed_vs_baseline_no_context | sentence_score | -0.0700 | (-0.1487, 0.0000) | 0.9767 | -0.0700 | (-0.1969, 0.0250) | 0.9227 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0077 | (-0.0182, 0.0329) | 0.2757 | 0.0077 | (-0.0221, 0.0365) | 0.3087 |
| proposed_vs_baseline_no_context | overall_quality | 0.0008 | (-0.0324, 0.0343) | 0.4780 | 0.0008 | (-0.0443, 0.0435) | 0.5000 |
| controlled_vs_proposed_raw | context_relevance | 0.1784 | (0.1241, 0.2303) | 0.0000 | 0.1784 | (0.1018, 0.2573) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.1771 | (0.1166, 0.2400) | 0.0000 | 0.1771 | (0.1219, 0.2347) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0650 | (0.0294, 0.0976) | 0.0007 | 0.0650 | (0.0166, 0.1158) | 0.0037 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2296 | (0.1572, 0.2993) | 0.0000 | 0.2296 | (0.1230, 0.3402) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0588 | (0.0368, 0.0831) | 0.0000 | 0.0588 | (0.0355, 0.0787) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2096 | (0.1345, 0.2856) | 0.0000 | 0.2096 | (0.1494, 0.2724) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.0467 | (-0.0084, 0.1101) | 0.0520 | 0.0467 | (-0.0391, 0.1848) | 0.1867 |
| controlled_vs_proposed_raw | distinct1 | 0.0010 | (-0.0143, 0.0158) | 0.4307 | 0.0010 | (-0.0148, 0.0181) | 0.4627 |
| controlled_vs_proposed_raw | length_score | 0.2808 | (0.1433, 0.4083) | 0.0000 | 0.2808 | (0.1030, 0.4619) | 0.0007 |
| controlled_vs_proposed_raw | sentence_score | 0.0875 | (0.0087, 0.1662) | 0.0237 | 0.0875 | (0.0000, 0.2012) | 0.0287 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0216 | (-0.0058, 0.0488) | 0.0610 | 0.0216 | (-0.0124, 0.0600) | 0.0983 |
| controlled_vs_proposed_raw | overall_quality | 0.1445 | (0.1082, 0.1810) | 0.0000 | 0.1445 | (0.0977, 0.1955) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2625 | (0.2267, 0.3007) | 0.0000 | 0.2625 | (0.2299, 0.2996) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1373 | (0.0638, 0.2120) | 0.0003 | 0.1373 | (0.0401, 0.2316) | 0.0047 |
| controlled_vs_candidate_no_context | naturalness | 0.0689 | (0.0351, 0.1010) | 0.0000 | 0.0689 | (0.0187, 0.1260) | 0.0017 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3406 | (0.2935, 0.3873) | 0.0000 | 0.3406 | (0.2984, 0.3935) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0803 | (0.0634, 0.0995) | 0.0000 | 0.0803 | (0.0603, 0.0963) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1632 | (0.0784, 0.2469) | 0.0003 | 0.1632 | (0.0508, 0.2673) | 0.0003 |
| controlled_vs_candidate_no_context | persona_style | 0.0335 | (-0.0357, 0.1044) | 0.1747 | 0.0335 | (-0.0608, 0.1665) | 0.2447 |
| controlled_vs_candidate_no_context | distinct1 | 0.0075 | (-0.0083, 0.0228) | 0.1733 | 0.0075 | (-0.0083, 0.0234) | 0.1830 |
| controlled_vs_candidate_no_context | length_score | 0.2700 | (0.1208, 0.4000) | 0.0007 | 0.2700 | (0.0786, 0.4785) | 0.0017 |
| controlled_vs_candidate_no_context | sentence_score | 0.1225 | (0.0437, 0.2012) | 0.0013 | 0.1225 | (0.0184, 0.2333) | 0.0100 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0461 | (0.0246, 0.0700) | 0.0000 | 0.0461 | (0.0152, 0.0804) | 0.0010 |
| controlled_vs_candidate_no_context | overall_quality | 0.1691 | (0.1379, 0.1998) | 0.0000 | 0.1691 | (0.1314, 0.2083) | 0.0000 |
| controlled_vs_baseline_no_context | context_relevance | 0.2414 | (0.2073, 0.2757) | 0.0000 | 0.2414 | (0.2155, 0.2691) | 0.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1406 | (0.0682, 0.2085) | 0.0000 | 0.1406 | (0.0898, 0.1817) | 0.0000 |
| controlled_vs_baseline_no_context | naturalness | -0.0115 | (-0.0382, 0.0145) | 0.8087 | -0.0115 | (-0.0409, 0.0151) | 0.7597 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3111 | (0.2685, 0.3545) | 0.0000 | 0.3111 | (0.2799, 0.3481) | 0.0000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0786 | (0.0609, 0.0987) | 0.0000 | 0.0786 | (0.0582, 0.0939) | 0.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1714 | (0.0869, 0.2514) | 0.0000 | 0.1714 | (0.1085, 0.2239) | 0.0000 |
| controlled_vs_baseline_no_context | persona_style | 0.0172 | (-0.0170, 0.0556) | 0.1830 | 0.0172 | (-0.0012, 0.0497) | 0.0943 |
| controlled_vs_baseline_no_context | distinct1 | -0.0340 | (-0.0483, -0.0192) | 1.0000 | -0.0340 | (-0.0470, -0.0221) | 1.0000 |
| controlled_vs_baseline_no_context | length_score | 0.0033 | (-0.1092, 0.1150) | 0.4700 | 0.0033 | (-0.1245, 0.1316) | 0.4607 |
| controlled_vs_baseline_no_context | sentence_score | 0.0175 | (-0.0437, 0.0788) | 0.3267 | 0.0175 | (-0.0467, 0.0718) | 0.3750 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0292 | (0.0027, 0.0561) | 0.0130 | 0.0292 | (0.0064, 0.0617) | 0.0020 |
| controlled_vs_baseline_no_context | overall_quality | 0.1453 | (0.1202, 0.1710) | 0.0000 | 0.1453 | (0.1334, 0.1571) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2414 | (0.2081, 0.2758) | 0.0000 | 0.2414 | (0.2173, 0.2690) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1406 | (0.0719, 0.2077) | 0.0003 | 0.1406 | (0.0908, 0.1803) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0115 | (-0.0394, 0.0150) | 0.7850 | -0.0115 | (-0.0417, 0.0148) | 0.7827 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3111 | (0.2705, 0.3551) | 0.0000 | 0.3111 | (0.2791, 0.3506) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0786 | (0.0606, 0.0992) | 0.0000 | 0.0786 | (0.0599, 0.0947) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1714 | (0.0848, 0.2536) | 0.0000 | 0.1714 | (0.1084, 0.2240) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0172 | (-0.0166, 0.0545) | 0.1777 | 0.0172 | (-0.0012, 0.0512) | 0.0943 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0340 | (-0.0478, -0.0192) | 1.0000 | -0.0340 | (-0.0478, -0.0218) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.0033 | (-0.1117, 0.1150) | 0.4663 | 0.0033 | (-0.1187, 0.1281) | 0.4633 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0175 | (-0.0437, 0.0788) | 0.3330 | 0.0175 | (-0.0412, 0.0718) | 0.3680 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0292 | (0.0039, 0.0560) | 0.0130 | 0.0292 | (0.0066, 0.0641) | 0.0013 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1453 | (0.1207, 0.1716) | 0.0000 | 0.1453 | (0.1336, 0.1572) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 22 | 5 | 13 | 0.7125 | 0.8148 |
| proposed_vs_candidate_no_context | persona_consistency | 10 | 9 | 21 | 0.5125 | 0.5263 |
| proposed_vs_candidate_no_context | naturalness | 14 | 13 | 13 | 0.5125 | 0.5185 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 18 | 3 | 19 | 0.6875 | 0.8571 |
| proposed_vs_candidate_no_context | context_overlap | 20 | 7 | 13 | 0.6625 | 0.7407 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 7 | 8 | 25 | 0.4875 | 0.4667 |
| proposed_vs_candidate_no_context | persona_style | 4 | 5 | 31 | 0.4875 | 0.4444 |
| proposed_vs_candidate_no_context | distinct1 | 16 | 11 | 13 | 0.5625 | 0.5926 |
| proposed_vs_candidate_no_context | length_score | 13 | 13 | 14 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | sentence_score | 8 | 4 | 28 | 0.5500 | 0.6667 |
| proposed_vs_candidate_no_context | bertscore_f1 | 19 | 8 | 13 | 0.6375 | 0.7037 |
| proposed_vs_candidate_no_context | overall_quality | 20 | 7 | 13 | 0.6625 | 0.7407 |
| proposed_vs_baseline_no_context | context_relevance | 22 | 18 | 0 | 0.5500 | 0.5500 |
| proposed_vs_baseline_no_context | persona_consistency | 9 | 16 | 15 | 0.4125 | 0.3600 |
| proposed_vs_baseline_no_context | naturalness | 9 | 31 | 0 | 0.2250 | 0.2250 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 18 | 8 | 14 | 0.6250 | 0.6923 |
| proposed_vs_baseline_no_context | context_overlap | 23 | 17 | 0 | 0.5750 | 0.5750 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 7 | 11 | 22 | 0.4500 | 0.3889 |
| proposed_vs_baseline_no_context | persona_style | 5 | 6 | 29 | 0.4875 | 0.4545 |
| proposed_vs_baseline_no_context | distinct1 | 5 | 31 | 4 | 0.1750 | 0.1389 |
| proposed_vs_baseline_no_context | length_score | 9 | 30 | 1 | 0.2375 | 0.2308 |
| proposed_vs_baseline_no_context | sentence_score | 7 | 15 | 18 | 0.4000 | 0.3182 |
| proposed_vs_baseline_no_context | bertscore_f1 | 22 | 18 | 0 | 0.5500 | 0.5500 |
| proposed_vs_baseline_no_context | overall_quality | 19 | 21 | 0 | 0.4750 | 0.4750 |
| controlled_vs_proposed_raw | context_relevance | 34 | 6 | 0 | 0.8500 | 0.8500 |
| controlled_vs_proposed_raw | persona_consistency | 32 | 4 | 4 | 0.8500 | 0.8889 |
| controlled_vs_proposed_raw | naturalness | 29 | 11 | 0 | 0.7250 | 0.7250 |
| controlled_vs_proposed_raw | context_keyword_coverage | 33 | 4 | 3 | 0.8625 | 0.8919 |
| controlled_vs_proposed_raw | context_overlap | 34 | 6 | 0 | 0.8500 | 0.8500 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 32 | 3 | 5 | 0.8625 | 0.9143 |
| controlled_vs_proposed_raw | persona_style | 6 | 5 | 29 | 0.5125 | 0.5455 |
| controlled_vs_proposed_raw | distinct1 | 21 | 19 | 0 | 0.5250 | 0.5250 |
| controlled_vs_proposed_raw | length_score | 28 | 9 | 3 | 0.7375 | 0.7568 |
| controlled_vs_proposed_raw | sentence_score | 18 | 8 | 14 | 0.6250 | 0.6923 |
| controlled_vs_proposed_raw | bertscore_f1 | 23 | 17 | 0 | 0.5750 | 0.5750 |
| controlled_vs_proposed_raw | overall_quality | 34 | 6 | 0 | 0.8500 | 0.8500 |
| controlled_vs_candidate_no_context | context_relevance | 40 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 29 | 5 | 6 | 0.8000 | 0.8529 |
| controlled_vs_candidate_no_context | naturalness | 28 | 12 | 0 | 0.7000 | 0.7000 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 39 | 0 | 1 | 0.9875 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 39 | 0 | 1 | 0.9875 | 1.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 29 | 4 | 7 | 0.8125 | 0.8788 |
| controlled_vs_candidate_no_context | persona_style | 6 | 6 | 28 | 0.5000 | 0.5000 |
| controlled_vs_candidate_no_context | distinct1 | 20 | 20 | 0 | 0.5000 | 0.5000 |
| controlled_vs_candidate_no_context | length_score | 29 | 11 | 0 | 0.7250 | 0.7250 |
| controlled_vs_candidate_no_context | sentence_score | 20 | 6 | 14 | 0.6750 | 0.7692 |
| controlled_vs_candidate_no_context | bertscore_f1 | 27 | 13 | 0 | 0.6750 | 0.6750 |
| controlled_vs_candidate_no_context | overall_quality | 38 | 2 | 0 | 0.9500 | 0.9500 |
| controlled_vs_baseline_no_context | context_relevance | 39 | 1 | 0 | 0.9750 | 0.9750 |
| controlled_vs_baseline_no_context | persona_consistency | 30 | 4 | 6 | 0.8250 | 0.8824 |
| controlled_vs_baseline_no_context | naturalness | 21 | 19 | 0 | 0.5250 | 0.5250 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 39 | 1 | 0 | 0.9750 | 0.9750 |
| controlled_vs_baseline_no_context | context_overlap | 40 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 29 | 4 | 7 | 0.8125 | 0.8788 |
| controlled_vs_baseline_no_context | persona_style | 4 | 3 | 33 | 0.5125 | 0.5714 |
| controlled_vs_baseline_no_context | distinct1 | 9 | 30 | 1 | 0.2375 | 0.2308 |
| controlled_vs_baseline_no_context | length_score | 20 | 18 | 2 | 0.5250 | 0.5263 |
| controlled_vs_baseline_no_context | sentence_score | 7 | 5 | 28 | 0.5250 | 0.5833 |
| controlled_vs_baseline_no_context | bertscore_f1 | 25 | 15 | 0 | 0.6250 | 0.6250 |
| controlled_vs_baseline_no_context | overall_quality | 39 | 1 | 0 | 0.9750 | 0.9750 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 39 | 1 | 0 | 0.9750 | 0.9750 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 30 | 4 | 6 | 0.8250 | 0.8824 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 21 | 19 | 0 | 0.5250 | 0.5250 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 39 | 1 | 0 | 0.9750 | 0.9750 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 40 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 29 | 4 | 7 | 0.8125 | 0.8788 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 4 | 3 | 33 | 0.5125 | 0.5714 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | 9 | 30 | 1 | 0.2375 | 0.2308 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 20 | 18 | 2 | 0.5250 | 0.5263 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 7 | 5 | 28 | 0.5250 | 0.5833 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 25 | 15 | 0 | 0.6250 | 0.6250 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 39 | 1 | 0 | 0.9750 | 0.9750 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.2000 | 0.5000 | 0.5000 |
| proposed_contextual | 0.0000 | 0.0000 | 0.4000 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.4500 | 0.0000 | 0.0000 |
| baseline_no_context | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `8`
- Unique template signatures: `36`
- Template signature ratio: `0.9000`
- Effective sample size by source clustering: `7.48`
- Effective sample size by template-signature clustering: `33.33`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: enabled.

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
# Proposal Alignment Evaluation Report

- Run ID: `20260305T230842Z`
- Generated: `2026-03-05T23:12:02.518933+00:00`
- Scenarios: `artifacts\proposal_control_tuning\architecture_compare_quick\20260305T230842Z\runtime_optimized\seed_29\20260305T230842Z\scenarios.jsonl`
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
| proposed_contextual_controlled | 0.3013 (0.2366, 0.3757) | 0.3108 (0.2645, 0.3662) | 0.8644 (0.8069, 0.9121) | 0.4183 (0.3848, 0.4547) | n/a |
| proposed_contextual_controlled_alt | 0.2961 (0.2458, 0.3563) | 0.2598 (0.2072, 0.3109) | 0.9001 (0.8749, 0.9245) | 0.4049 (0.3715, 0.4374) | n/a |
| proposed_contextual | 0.0688 (0.0285, 0.1190) | 0.1280 (0.0807, 0.1724) | 0.8471 (0.7991, 0.8972) | 0.2401 (0.2044, 0.2779) | n/a |
| candidate_no_context | 0.0326 (0.0125, 0.0571) | 0.1579 (0.0917, 0.2377) | 0.8480 (0.8010, 0.8960) | 0.2341 (0.1999, 0.2697) | n/a |
| baseline_no_context | 0.0332 (0.0144, 0.0561) | 0.1627 (0.1139, 0.2146) | 0.9045 (0.8719, 0.9361) | 0.2475 (0.2286, 0.2677) | n/a |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0362 | 1.1116 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0300 | -0.1899 |
| proposed_vs_candidate_no_context | naturalness | -0.0009 | -0.0010 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0442 | 1.5217 |
| proposed_vs_candidate_no_context | context_overlap | 0.0175 | 0.4302 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0238 | -0.4615 |
| proposed_vs_candidate_no_context | persona_style | -0.0547 | -0.0938 |
| proposed_vs_candidate_no_context | distinct1 | 0.0170 | 0.0180 |
| proposed_vs_candidate_no_context | length_score | -0.0528 | -0.1092 |
| proposed_vs_candidate_no_context | sentence_score | 0.0292 | 0.0395 |
| proposed_vs_candidate_no_context | overall_quality | 0.0060 | 0.0255 |
| proposed_vs_baseline_no_context | context_relevance | 0.0355 | 1.0692 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0347 | -0.2134 |
| proposed_vs_baseline_no_context | naturalness | -0.0574 | -0.0634 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0442 | 1.5217 |
| proposed_vs_baseline_no_context | context_overlap | 0.0153 | 0.3562 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0313 | -0.5302 |
| proposed_vs_baseline_no_context | persona_style | -0.0482 | -0.0835 |
| proposed_vs_baseline_no_context | distinct1 | -0.0240 | -0.0243 |
| proposed_vs_baseline_no_context | length_score | -0.1806 | -0.2955 |
| proposed_vs_baseline_no_context | sentence_score | -0.1167 | -0.1321 |
| proposed_vs_baseline_no_context | overall_quality | -0.0074 | -0.0301 |
| controlled_vs_proposed_raw | context_relevance | 0.2325 | 3.3817 |
| controlled_vs_proposed_raw | persona_consistency | 0.1828 | 1.4290 |
| controlled_vs_proposed_raw | naturalness | 0.0173 | 0.0204 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.3011 | 4.1121 |
| controlled_vs_proposed_raw | context_overlap | 0.0724 | 1.2421 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2024 | 7.2857 |
| controlled_vs_proposed_raw | persona_style | 0.1047 | 0.1981 |
| controlled_vs_proposed_raw | distinct1 | -0.0400 | -0.0416 |
| controlled_vs_proposed_raw | length_score | 0.1222 | 0.2839 |
| controlled_vs_proposed_raw | sentence_score | 0.1167 | 0.1522 |
| controlled_vs_proposed_raw | overall_quality | 0.1783 | 0.7425 |
| controlled_vs_candidate_no_context | context_relevance | 0.2687 | 8.2525 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1529 | 0.9679 |
| controlled_vs_candidate_no_context | naturalness | 0.0164 | 0.0194 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3453 | 11.8913 |
| controlled_vs_candidate_no_context | context_overlap | 0.0900 | 2.2066 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1786 | 3.4615 |
| controlled_vs_candidate_no_context | persona_style | 0.0500 | 0.0857 |
| controlled_vs_candidate_no_context | distinct1 | -0.0231 | -0.0244 |
| controlled_vs_candidate_no_context | length_score | 0.0694 | 0.1437 |
| controlled_vs_candidate_no_context | sentence_score | 0.1458 | 0.1977 |
| controlled_vs_candidate_no_context | overall_quality | 0.1842 | 0.7869 |
| controlled_vs_baseline_no_context | context_relevance | 0.2681 | 8.0666 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1481 | 0.9107 |
| controlled_vs_baseline_no_context | naturalness | -0.0401 | -0.0443 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3453 | 11.8913 |
| controlled_vs_baseline_no_context | context_overlap | 0.0878 | 2.0407 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1710 | 2.8926 |
| controlled_vs_baseline_no_context | persona_style | 0.0566 | 0.0980 |
| controlled_vs_baseline_no_context | distinct1 | -0.0640 | -0.0650 |
| controlled_vs_baseline_no_context | length_score | -0.0583 | -0.0955 |
| controlled_vs_baseline_no_context | sentence_score | 0.0000 | 0.0000 |
| controlled_vs_baseline_no_context | overall_quality | 0.1708 | 0.6901 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0052 | -0.0172 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0510 | -0.1642 |
| controlled_alt_vs_controlled_default | naturalness | 0.0357 | 0.0413 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0038 | 0.0101 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0261 | -0.1998 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0619 | -0.2690 |
| controlled_alt_vs_controlled_default | persona_style | -0.0075 | -0.0119 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0103 | 0.0112 |
| controlled_alt_vs_controlled_default | length_score | 0.1583 | 0.2864 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0292 | -0.0330 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0134 | -0.0321 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.2273 | 3.3063 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1318 | 1.0302 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0530 | 0.0625 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.3049 | 4.1638 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0463 | 0.7941 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1405 | 5.0571 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0972 | 0.1839 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0297 | -0.0309 |
| controlled_alt_vs_proposed_raw | length_score | 0.2806 | 0.6516 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0875 | 0.1141 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1648 | 0.6864 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2635 | 8.0931 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1018 | 0.6448 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0521 | 0.0615 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3491 | 12.0217 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0639 | 1.5658 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1167 | 2.2615 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0425 | 0.0729 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0127 | -0.0135 |
| controlled_alt_vs_candidate_no_context | length_score | 0.2278 | 0.4713 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1167 | 0.1582 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1708 | 0.7295 |
| controlled_alt_vs_baseline_no_context | context_relevance | 0.2629 | 7.9105 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 0.0971 | 0.5970 |
| controlled_alt_vs_baseline_no_context | naturalness | -0.0044 | -0.0048 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 0.3491 | 12.0217 |
| controlled_alt_vs_baseline_no_context | context_overlap | 0.0616 | 1.4331 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 0.1091 | 1.8456 |
| controlled_alt_vs_baseline_no_context | persona_style | 0.0490 | 0.0850 |
| controlled_alt_vs_baseline_no_context | distinct1 | -0.0536 | -0.0545 |
| controlled_alt_vs_baseline_no_context | length_score | 0.1000 | 0.1636 |
| controlled_alt_vs_baseline_no_context | sentence_score | -0.0292 | -0.0330 |
| controlled_alt_vs_baseline_no_context | overall_quality | 0.1574 | 0.6358 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2681 | 8.0666 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1481 | 0.9107 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0401 | -0.0443 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3453 | 11.8913 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0878 | 2.0407 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1710 | 2.8926 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0566 | 0.0980 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0640 | -0.0650 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | -0.0583 | -0.0955 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0000 | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1708 | 0.6901 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0362 | (-0.0139, 0.0940) | 0.0853 | 0.0362 | (-0.0059, 0.0791) | 0.0497 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0300 | (-0.0998, 0.0333) | 0.8033 | -0.0300 | (-0.0950, 0.0190) | 0.8677 |
| proposed_vs_candidate_no_context | naturalness | -0.0009 | (-0.0621, 0.0543) | 0.5093 | -0.0009 | (-0.0760, 0.0549) | 0.5103 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0442 | (-0.0265, 0.1212) | 0.1153 | 0.0442 | (-0.0101, 0.1028) | 0.0747 |
| proposed_vs_candidate_no_context | context_overlap | 0.0175 | (0.0026, 0.0330) | 0.0070 | 0.0175 | (0.0021, 0.0332) | 0.0140 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0238 | (-0.1071, 0.0556) | 0.7013 | -0.0238 | (-0.0976, 0.0303) | 0.8123 |
| proposed_vs_candidate_no_context | persona_style | -0.0547 | (-0.1641, 0.0000) | 1.0000 | -0.0547 | (-0.1969, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | distinct1 | 0.0170 | (-0.0060, 0.0401) | 0.0807 | 0.0170 | (-0.0017, 0.0334) | 0.0390 |
| proposed_vs_candidate_no_context | length_score | -0.0528 | (-0.2917, 0.1833) | 0.6710 | -0.0528 | (-0.3433, 0.1643) | 0.6607 |
| proposed_vs_candidate_no_context | sentence_score | 0.0292 | (-0.0583, 0.1167) | 0.3947 | 0.0292 | (-0.0955, 0.1591) | 0.4423 |
| proposed_vs_candidate_no_context | overall_quality | 0.0060 | (-0.0388, 0.0524) | 0.4153 | 0.0060 | (-0.0413, 0.0354) | 0.3650 |
| proposed_vs_baseline_no_context | context_relevance | 0.0355 | (0.0053, 0.0703) | 0.0047 | 0.0355 | (0.0004, 0.0807) | 0.0197 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0347 | (-0.1051, 0.0324) | 0.8340 | -0.0347 | (-0.1151, 0.0300) | 0.8400 |
| proposed_vs_baseline_no_context | naturalness | -0.0574 | (-0.1170, 0.0008) | 0.9730 | -0.0574 | (-0.1389, 0.0182) | 0.9273 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0442 | (0.0000, 0.0896) | 0.0353 | 0.0442 | (0.0000, 0.1017) | 0.0893 |
| proposed_vs_baseline_no_context | context_overlap | 0.0153 | (-0.0022, 0.0312) | 0.0443 | 0.0153 | (-0.0007, 0.0332) | 0.0300 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0313 | (-0.0976, 0.0362) | 0.8210 | -0.0313 | (-0.1067, 0.0334) | 0.8257 |
| proposed_vs_baseline_no_context | persona_style | -0.0482 | (-0.1853, 0.0752) | 0.7580 | -0.0482 | (-0.2188, 0.0754) | 0.7403 |
| proposed_vs_baseline_no_context | distinct1 | -0.0240 | (-0.0547, 0.0051) | 0.9443 | -0.0240 | (-0.0616, 0.0096) | 0.9110 |
| proposed_vs_baseline_no_context | length_score | -0.1806 | (-0.3917, 0.0278) | 0.9610 | -0.1806 | (-0.4694, 0.0795) | 0.9020 |
| proposed_vs_baseline_no_context | sentence_score | -0.1167 | (-0.2333, 0.0000) | 0.9830 | -0.1167 | (-0.2423, -0.0250) | 1.0000 |
| proposed_vs_baseline_no_context | overall_quality | -0.0074 | (-0.0507, 0.0348) | 0.6377 | -0.0074 | (-0.0615, 0.0431) | 0.6080 |
| controlled_vs_proposed_raw | context_relevance | 0.2325 | (0.1555, 0.3291) | 0.0000 | 0.2325 | (0.1589, 0.3373) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.1828 | (0.1177, 0.2448) | 0.0000 | 0.1828 | (0.1255, 0.2487) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0173 | (-0.0359, 0.0685) | 0.2573 | 0.0173 | (-0.0434, 0.0794) | 0.2717 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.3011 | (0.1894, 0.4299) | 0.0000 | 0.3011 | (0.1997, 0.4545) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0724 | (0.0471, 0.0951) | 0.0000 | 0.0724 | (0.0488, 0.0961) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2024 | (0.1333, 0.2766) | 0.0000 | 0.2024 | (0.1444, 0.2724) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.1047 | (-0.0194, 0.2465) | 0.0507 | 0.1047 | (-0.0389, 0.2979) | 0.1137 |
| controlled_vs_proposed_raw | distinct1 | -0.0400 | (-0.0756, -0.0069) | 0.9917 | -0.0400 | (-0.0741, -0.0067) | 0.9907 |
| controlled_vs_proposed_raw | length_score | 0.1222 | (-0.0944, 0.3278) | 0.1297 | 0.1222 | (-0.1179, 0.3879) | 0.1493 |
| controlled_vs_proposed_raw | sentence_score | 0.1167 | (0.0000, 0.2333) | 0.0557 | 0.1167 | (0.0000, 0.2450) | 0.0477 |
| controlled_vs_proposed_raw | overall_quality | 0.1783 | (0.1264, 0.2377) | 0.0000 | 0.1783 | (0.1236, 0.2500) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2687 | (0.1911, 0.3607) | 0.0000 | 0.2687 | (0.1908, 0.3639) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1529 | (0.0875, 0.2106) | 0.0000 | 0.1529 | (0.0795, 0.2133) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.0164 | (-0.0280, 0.0629) | 0.2503 | 0.0164 | (-0.0319, 0.0670) | 0.2787 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3453 | (0.2412, 0.4659) | 0.0000 | 0.3453 | (0.2403, 0.4727) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0900 | (0.0689, 0.1141) | 0.0000 | 0.0900 | (0.0678, 0.1179) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1786 | (0.1028, 0.2452) | 0.0000 | 0.1786 | (0.0957, 0.2433) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.0500 | (-0.0373, 0.1611) | 0.1660 | 0.0500 | (-0.0538, 0.1923) | 0.3487 |
| controlled_vs_candidate_no_context | distinct1 | -0.0231 | (-0.0565, 0.0064) | 0.9317 | -0.0231 | (-0.0596, 0.0066) | 0.9223 |
| controlled_vs_candidate_no_context | length_score | 0.0694 | (-0.0889, 0.2417) | 0.2127 | 0.0694 | (-0.1273, 0.2778) | 0.2653 |
| controlled_vs_candidate_no_context | sentence_score | 0.1458 | (0.0583, 0.2333) | 0.0017 | 0.1458 | (0.0777, 0.2100) | 0.0007 |
| controlled_vs_candidate_no_context | overall_quality | 0.1842 | (0.1464, 0.2243) | 0.0000 | 0.1842 | (0.1454, 0.2296) | 0.0000 |
| controlled_vs_baseline_no_context | context_relevance | 0.2681 | (0.1951, 0.3567) | 0.0000 | 0.2681 | (0.2013, 0.3663) | 0.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1481 | (0.1058, 0.1906) | 0.0000 | 0.1481 | (0.1117, 0.1797) | 0.0000 |
| controlled_vs_baseline_no_context | naturalness | -0.0401 | (-0.1105, 0.0342) | 0.8493 | -0.0401 | (-0.1277, 0.0500) | 0.8007 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3453 | (0.2468, 0.4609) | 0.0000 | 0.3453 | (0.2488, 0.4773) | 0.0000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0878 | (0.0634, 0.1142) | 0.0000 | 0.0878 | (0.0669, 0.1199) | 0.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1710 | (0.1167, 0.2218) | 0.0000 | 0.1710 | (0.1167, 0.2198) | 0.0000 |
| controlled_vs_baseline_no_context | persona_style | 0.0566 | (-0.0258, 0.1509) | 0.1117 | 0.0566 | (0.0000, 0.1399) | 0.1010 |
| controlled_vs_baseline_no_context | distinct1 | -0.0640 | (-0.1075, -0.0248) | 0.9993 | -0.0640 | (-0.1116, -0.0295) | 1.0000 |
| controlled_vs_baseline_no_context | length_score | -0.0583 | (-0.3390, 0.2278) | 0.6657 | -0.0583 | (-0.3933, 0.3111) | 0.6357 |
| controlled_vs_baseline_no_context | sentence_score | 0.0000 | (-0.1167, 0.1167) | 0.6030 | 0.0000 | (-0.1077, 0.1273) | 0.5940 |
| controlled_vs_baseline_no_context | overall_quality | 0.1708 | (0.1380, 0.2057) | 0.0000 | 0.1708 | (0.1330, 0.2141) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0052 | (-0.1100, 0.0966) | 0.5257 | -0.0052 | (-0.1144, 0.0958) | 0.5180 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0510 | (-0.1113, -0.0036) | 0.9837 | -0.0510 | (-0.1318, -0.0010) | 1.0000 |
| controlled_alt_vs_controlled_default | naturalness | 0.0357 | (-0.0214, 0.0964) | 0.1140 | 0.0357 | (-0.0326, 0.1042) | 0.1697 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0038 | (-0.1402, 0.1370) | 0.4637 | 0.0038 | (-0.1591, 0.1415) | 0.4857 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0261 | (-0.0535, 0.0016) | 0.9693 | -0.0261 | (-0.0583, -0.0016) | 0.9840 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0619 | (-0.1405, 0.0000) | 1.0000 | -0.0619 | (-0.1619, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | persona_style | -0.0075 | (-0.0785, 0.0600) | 0.5880 | -0.0075 | (-0.0200, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0103 | (-0.0322, 0.0646) | 0.3837 | 0.0103 | (-0.0342, 0.0775) | 0.3863 |
| controlled_alt_vs_controlled_default | length_score | 0.1583 | (-0.0500, 0.3611) | 0.0687 | 0.1583 | (-0.0795, 0.3718) | 0.0970 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0292 | (-0.1750, 0.1167) | 0.7237 | -0.0292 | (-0.1167, 0.0636) | 0.8150 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0134 | (-0.0566, 0.0270) | 0.7430 | -0.0134 | (-0.0615, 0.0307) | 0.7350 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.2273 | (0.1683, 0.2945) | 0.0000 | 0.2273 | (0.1951, 0.2758) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1318 | (0.0704, 0.1920) | 0.0000 | 0.1318 | (0.0677, 0.2037) | 0.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0530 | (-0.0069, 0.1119) | 0.0430 | 0.0530 | (-0.0155, 0.1283) | 0.0730 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.3049 | (0.2266, 0.3965) | 0.0000 | 0.3049 | (0.2576, 0.3750) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0463 | (0.0295, 0.0628) | 0.0000 | 0.0463 | (0.0343, 0.0559) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1405 | (0.0750, 0.2020) | 0.0000 | 0.1405 | (0.0810, 0.1988) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0972 | (-0.0350, 0.2474) | 0.0840 | 0.0972 | (-0.0478, 0.2929) | 0.1093 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0297 | (-0.0682, 0.0074) | 0.9367 | -0.0297 | (-0.0715, 0.0143) | 0.9203 |
| controlled_alt_vs_proposed_raw | length_score | 0.2806 | (0.0667, 0.4723) | 0.0050 | 0.2806 | (0.0436, 0.5278) | 0.0107 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0875 | (-0.0875, 0.2333) | 0.1847 | 0.0875 | (-0.0583, 0.2545) | 0.1887 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1648 | (0.1279, 0.2024) | 0.0000 | 0.1648 | (0.1362, 0.2038) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2635 | (0.2181, 0.3089) | 0.0000 | 0.2635 | (0.2119, 0.3187) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1018 | (0.0322, 0.1662) | 0.0027 | 0.1018 | (0.0280, 0.1771) | 0.0013 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0521 | (-0.0089, 0.1092) | 0.0500 | 0.0521 | (-0.0215, 0.1148) | 0.0883 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3491 | (0.2917, 0.4110) | 0.0000 | 0.3491 | (0.2803, 0.4213) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0639 | (0.0459, 0.0822) | 0.0000 | 0.0639 | (0.0465, 0.0781) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1167 | (0.0417, 0.1861) | 0.0023 | 0.1167 | (0.0370, 0.1939) | 0.0020 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0425 | (-0.0539, 0.1646) | 0.2270 | 0.0425 | (-0.0688, 0.1955) | 0.3570 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0127 | (-0.0418, 0.0174) | 0.7983 | -0.0127 | (-0.0508, 0.0262) | 0.7423 |
| controlled_alt_vs_candidate_no_context | length_score | 0.2278 | (-0.0223, 0.4556) | 0.0363 | 0.2278 | (-0.0667, 0.4667) | 0.0663 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1167 | (-0.0292, 0.2625) | 0.0963 | 0.1167 | (-0.0269, 0.2500) | 0.0760 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1708 | (0.1497, 0.1937) | 0.0000 | 0.1708 | (0.1415, 0.1974) | 0.0000 |
| controlled_alt_vs_baseline_no_context | context_relevance | 0.2629 | (0.2058, 0.3273) | 0.0000 | 0.2629 | (0.2077, 0.3276) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 0.0971 | (0.0347, 0.1526) | 0.0013 | 0.0971 | (0.0174, 0.1553) | 0.0090 |
| controlled_alt_vs_baseline_no_context | naturalness | -0.0044 | (-0.0412, 0.0294) | 0.5903 | -0.0044 | (-0.0389, 0.0285) | 0.6043 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 0.3491 | (0.2746, 0.4343) | 0.0000 | 0.3491 | (0.2733, 0.4347) | 0.0000 |
| controlled_alt_vs_baseline_no_context | context_overlap | 0.0616 | (0.0410, 0.0804) | 0.0000 | 0.0616 | (0.0431, 0.0769) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 0.1091 | (0.0333, 0.1813) | 0.0023 | 0.1091 | (0.0143, 0.1905) | 0.0147 |
| controlled_alt_vs_baseline_no_context | persona_style | 0.0490 | (-0.0001, 0.1141) | 0.0330 | 0.0490 | (-0.0045, 0.1309) | 0.0913 |
| controlled_alt_vs_baseline_no_context | distinct1 | -0.0536 | (-0.0835, -0.0201) | 0.9993 | -0.0536 | (-0.0817, -0.0257) | 1.0000 |
| controlled_alt_vs_baseline_no_context | length_score | 0.1000 | (-0.0584, 0.2472) | 0.1053 | 0.1000 | (-0.0718, 0.2695) | 0.1423 |
| controlled_alt_vs_baseline_no_context | sentence_score | -0.0292 | (-0.1458, 0.0875) | 0.7507 | -0.0292 | (-0.1167, 0.0700) | 0.8087 |
| controlled_alt_vs_baseline_no_context | overall_quality | 0.1574 | (0.1202, 0.1944) | 0.0000 | 0.1574 | (0.1121, 0.1997) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2681 | (0.1954, 0.3548) | 0.0000 | 0.2681 | (0.1976, 0.3724) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1481 | (0.1062, 0.1912) | 0.0000 | 0.1481 | (0.1099, 0.1798) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0401 | (-0.1162, 0.0347) | 0.8490 | -0.0401 | (-0.1252, 0.0589) | 0.7857 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3453 | (0.2475, 0.4621) | 0.0000 | 0.3453 | (0.2488, 0.4780) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0878 | (0.0632, 0.1130) | 0.0000 | 0.0878 | (0.0666, 0.1202) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1710 | (0.1194, 0.2198) | 0.0000 | 0.1710 | (0.1212, 0.2194) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0566 | (-0.0258, 0.1498) | 0.1247 | 0.0566 | (0.0000, 0.1364) | 0.0953 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0640 | (-0.1093, -0.0259) | 0.9997 | -0.0640 | (-0.1101, -0.0295) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | -0.0583 | (-0.3333, 0.2167) | 0.6790 | -0.0583 | (-0.3975, 0.3152) | 0.6213 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0000 | (-0.1167, 0.1167) | 0.6090 | 0.0000 | (-0.1077, 0.1167) | 0.5960 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1708 | (0.1385, 0.2030) | 0.0000 | 0.1708 | (0.1360, 0.2150) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 6 | 3 | 3 | 0.6250 | 0.6667 |
| proposed_vs_candidate_no_context | persona_consistency | 2 | 3 | 7 | 0.4583 | 0.4000 |
| proposed_vs_candidate_no_context | naturalness | 5 | 4 | 3 | 0.5417 | 0.5556 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 4 | 2 | 6 | 0.5833 | 0.6667 |
| proposed_vs_candidate_no_context | context_overlap | 7 | 2 | 3 | 0.7083 | 0.7778 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 2 | 2 | 8 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | persona_style | 0 | 1 | 11 | 0.4583 | 0.0000 |
| proposed_vs_candidate_no_context | distinct1 | 6 | 2 | 4 | 0.6667 | 0.7500 |
| proposed_vs_candidate_no_context | length_score | 4 | 5 | 3 | 0.4583 | 0.4444 |
| proposed_vs_candidate_no_context | sentence_score | 2 | 1 | 9 | 0.5417 | 0.6667 |
| proposed_vs_candidate_no_context | overall_quality | 4 | 5 | 3 | 0.4583 | 0.4444 |
| proposed_vs_baseline_no_context | context_relevance | 9 | 3 | 0 | 0.7500 | 0.7500 |
| proposed_vs_baseline_no_context | persona_consistency | 3 | 5 | 4 | 0.4167 | 0.3750 |
| proposed_vs_baseline_no_context | naturalness | 6 | 6 | 0 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 3 | 0 | 9 | 0.6250 | 1.0000 |
| proposed_vs_baseline_no_context | context_overlap | 9 | 3 | 0 | 0.7500 | 0.7500 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 2 | 4 | 6 | 0.4167 | 0.3333 |
| proposed_vs_baseline_no_context | persona_style | 1 | 3 | 8 | 0.4167 | 0.2500 |
| proposed_vs_baseline_no_context | distinct1 | 2 | 6 | 4 | 0.3333 | 0.2500 |
| proposed_vs_baseline_no_context | length_score | 6 | 6 | 0 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context | sentence_score | 1 | 5 | 6 | 0.3333 | 0.1667 |
| proposed_vs_baseline_no_context | overall_quality | 6 | 6 | 0 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | context_relevance | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_proposed_raw | persona_consistency | 10 | 0 | 2 | 0.9167 | 1.0000 |
| controlled_vs_proposed_raw | naturalness | 7 | 5 | 0 | 0.5833 | 0.5833 |
| controlled_vs_proposed_raw | context_keyword_coverage | 11 | 0 | 1 | 0.9583 | 1.0000 |
| controlled_vs_proposed_raw | context_overlap | 11 | 1 | 0 | 0.9167 | 0.9167 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 10 | 0 | 2 | 0.9167 | 1.0000 |
| controlled_vs_proposed_raw | persona_style | 3 | 2 | 7 | 0.5417 | 0.6000 |
| controlled_vs_proposed_raw | distinct1 | 2 | 9 | 1 | 0.2083 | 0.1818 |
| controlled_vs_proposed_raw | length_score | 6 | 6 | 0 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | sentence_score | 5 | 1 | 6 | 0.6667 | 0.8333 |
| controlled_vs_proposed_raw | overall_quality | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_relevance | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 11 | 1 | 0 | 0.9167 | 0.9167 |
| controlled_vs_candidate_no_context | naturalness | 6 | 6 | 0 | 0.5000 | 0.5000 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 11 | 1 | 0 | 0.9167 | 0.9167 |
| controlled_vs_candidate_no_context | persona_style | 2 | 2 | 8 | 0.5000 | 0.5000 |
| controlled_vs_candidate_no_context | distinct1 | 3 | 9 | 0 | 0.2500 | 0.2500 |
| controlled_vs_candidate_no_context | length_score | 6 | 6 | 0 | 0.5000 | 0.5000 |
| controlled_vs_candidate_no_context | sentence_score | 5 | 0 | 7 | 0.7083 | 1.0000 |
| controlled_vs_candidate_no_context | overall_quality | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | context_relevance | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 11 | 0 | 1 | 0.9583 | 1.0000 |
| controlled_vs_baseline_no_context | naturalness | 5 | 7 | 0 | 0.4167 | 0.4167 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | context_overlap | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 10 | 0 | 2 | 0.9167 | 1.0000 |
| controlled_vs_baseline_no_context | persona_style | 3 | 1 | 8 | 0.5833 | 0.7500 |
| controlled_vs_baseline_no_context | distinct1 | 2 | 10 | 0 | 0.1667 | 0.1667 |
| controlled_vs_baseline_no_context | length_score | 5 | 7 | 0 | 0.4167 | 0.4167 |
| controlled_vs_baseline_no_context | sentence_score | 2 | 2 | 8 | 0.5000 | 0.5000 |
| controlled_vs_baseline_no_context | overall_quality | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 8 | 4 | 0 | 0.6667 | 0.6667 |
| controlled_alt_vs_controlled_default | persona_consistency | 2 | 5 | 5 | 0.3750 | 0.2857 |
| controlled_alt_vs_controlled_default | naturalness | 7 | 5 | 0 | 0.5833 | 0.5833 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 7 | 4 | 1 | 0.6250 | 0.6364 |
| controlled_alt_vs_controlled_default | context_overlap | 3 | 9 | 0 | 0.2500 | 0.2500 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0 | 3 | 9 | 0.3750 | 0.0000 |
| controlled_alt_vs_controlled_default | persona_style | 2 | 3 | 7 | 0.4583 | 0.4000 |
| controlled_alt_vs_controlled_default | distinct1 | 5 | 6 | 1 | 0.4583 | 0.4545 |
| controlled_alt_vs_controlled_default | length_score | 7 | 4 | 1 | 0.6250 | 0.6364 |
| controlled_alt_vs_controlled_default | sentence_score | 3 | 4 | 5 | 0.4583 | 0.4286 |
| controlled_alt_vs_controlled_default | overall_quality | 6 | 6 | 0 | 0.5000 | 0.5000 |
| controlled_alt_vs_proposed_raw | context_relevance | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 8 | 0 | 4 | 0.8333 | 1.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 8 | 4 | 0 | 0.6667 | 0.6667 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 11 | 0 | 1 | 0.9583 | 1.0000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 8 | 0 | 4 | 0.8333 | 1.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 3 | 2 | 7 | 0.5417 | 0.6000 |
| controlled_alt_vs_proposed_raw | distinct1 | 4 | 7 | 1 | 0.3750 | 0.3636 |
| controlled_alt_vs_proposed_raw | length_score | 10 | 2 | 0 | 0.8333 | 0.8333 |
| controlled_alt_vs_proposed_raw | sentence_score | 6 | 3 | 3 | 0.6250 | 0.6667 |
| controlled_alt_vs_proposed_raw | overall_quality | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 8 | 2 | 2 | 0.7500 | 0.8000 |
| controlled_alt_vs_candidate_no_context | naturalness | 9 | 3 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 8 | 1 | 3 | 0.7917 | 0.8889 |
| controlled_alt_vs_candidate_no_context | persona_style | 2 | 2 | 8 | 0.5000 | 0.5000 |
| controlled_alt_vs_candidate_no_context | distinct1 | 3 | 9 | 0 | 0.2500 | 0.2500 |
| controlled_alt_vs_candidate_no_context | length_score | 9 | 3 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_candidate_no_context | sentence_score | 6 | 2 | 4 | 0.6667 | 0.7500 |
| controlled_alt_vs_candidate_no_context | overall_quality | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context | context_relevance | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 9 | 1 | 2 | 0.8333 | 0.9000 |
| controlled_alt_vs_baseline_no_context | naturalness | 5 | 7 | 0 | 0.4167 | 0.4167 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context | context_overlap | 11 | 1 | 0 | 0.9167 | 0.9167 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 8 | 1 | 3 | 0.7917 | 0.8889 |
| controlled_alt_vs_baseline_no_context | persona_style | 3 | 1 | 8 | 0.5833 | 0.7500 |
| controlled_alt_vs_baseline_no_context | distinct1 | 3 | 8 | 1 | 0.2917 | 0.2727 |
| controlled_alt_vs_baseline_no_context | length_score | 9 | 3 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_baseline_no_context | sentence_score | 2 | 3 | 7 | 0.4583 | 0.4000 |
| controlled_alt_vs_baseline_no_context | overall_quality | 12 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 12 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 11 | 0 | 1 | 0.9583 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 5 | 7 | 0 | 0.4167 | 0.4167 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 12 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 12 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 10 | 0 | 2 | 0.9167 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 3 | 1 | 8 | 0.5833 | 0.7500 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | 2 | 10 | 0 | 0.1667 | 0.1667 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 5 | 7 | 0 | 0.4167 | 0.4167 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 2 | 2 | 8 | 0.5000 | 0.5000 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 12 | 0 | 0 | 1.0000 | 1.0000 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.1667 | 0.5833 | 0.4167 |
| proposed_contextual_controlled_alt | 0.0000 | 0.0000 | 0.2500 | 0.5000 | 0.5000 |
| proposed_contextual | 0.0000 | 0.0000 | 0.3333 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.4167 | 0.0000 | 0.0000 |
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
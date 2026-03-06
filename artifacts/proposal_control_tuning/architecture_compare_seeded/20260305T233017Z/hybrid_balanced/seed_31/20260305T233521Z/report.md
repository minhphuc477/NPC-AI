# Proposal Alignment Evaluation Report

- Run ID: `20260305T233521Z`
- Generated: `2026-03-05T23:40:37.349009+00:00`
- Scenarios: `artifacts\proposal_control_tuning\architecture_compare_seeded\20260305T233017Z\hybrid_balanced\seed_31\20260305T233521Z\scenarios.jsonl`
- Scenario count: `24`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_alt`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off
- `baseline_no_context`: model `phi3:mini`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2670 (0.2277, 0.3096) | 0.3492 (0.2931, 0.4159) | 0.8917 (0.8686, 0.9144) | 0.4232 (0.3957, 0.4555) | n/a |
| proposed_contextual_controlled_alt | 0.2967 (0.2596, 0.3336) | 0.3345 (0.2816, 0.3907) | 0.8747 (0.8504, 0.8980) | 0.4277 (0.4008, 0.4568) | n/a |
| proposed_contextual | 0.1099 (0.0666, 0.1555) | 0.1462 (0.1055, 0.1936) | 0.8330 (0.7979, 0.8686) | 0.2635 (0.2266, 0.3031) | n/a |
| candidate_no_context | 0.0248 (0.0152, 0.0356) | 0.2067 (0.1490, 0.2720) | 0.8495 (0.8151, 0.8845) | 0.2498 (0.2226, 0.2811) | n/a |
| baseline_no_context | 0.0303 (0.0174, 0.0458) | 0.1764 (0.1375, 0.2248) | 0.8955 (0.8726, 0.9173) | 0.2493 (0.2316, 0.2669) | n/a |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0850 | 3.4243 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0606 | -0.2930 |
| proposed_vs_candidate_no_context | naturalness | -0.0165 | -0.0194 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.1114 | 5.8833 |
| proposed_vs_candidate_no_context | context_overlap | 0.0234 | 0.6072 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0734 | -0.5920 |
| proposed_vs_candidate_no_context | persona_style | -0.0092 | -0.0171 |
| proposed_vs_candidate_no_context | distinct1 | -0.0021 | -0.0022 |
| proposed_vs_candidate_no_context | length_score | -0.0653 | -0.1478 |
| proposed_vs_candidate_no_context | sentence_score | -0.0292 | -0.0354 |
| proposed_vs_candidate_no_context | overall_quality | 0.0137 | 0.0547 |
| proposed_vs_baseline_no_context | context_relevance | 0.0796 | 2.6289 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0303 | -0.1716 |
| proposed_vs_baseline_no_context | naturalness | -0.0624 | -0.0697 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.1042 | 3.9759 |
| proposed_vs_baseline_no_context | context_overlap | 0.0222 | 0.5587 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0206 | -0.2897 |
| proposed_vs_baseline_no_context | persona_style | -0.0688 | -0.1153 |
| proposed_vs_baseline_no_context | distinct1 | -0.0364 | -0.0371 |
| proposed_vs_baseline_no_context | length_score | -0.2028 | -0.3501 |
| proposed_vs_baseline_no_context | sentence_score | -0.0729 | -0.0839 |
| proposed_vs_baseline_no_context | overall_quality | 0.0142 | 0.0569 |
| controlled_vs_proposed_raw | context_relevance | 0.1572 | 1.4308 |
| controlled_vs_proposed_raw | persona_consistency | 0.2030 | 1.3891 |
| controlled_vs_proposed_raw | naturalness | 0.0586 | 0.0704 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2047 | 1.5705 |
| controlled_vs_proposed_raw | context_overlap | 0.0462 | 0.7455 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2349 | 4.6431 |
| controlled_vs_proposed_raw | persona_style | 0.0754 | 0.1427 |
| controlled_vs_proposed_raw | distinct1 | 0.0072 | 0.0076 |
| controlled_vs_proposed_raw | length_score | 0.2278 | 0.6052 |
| controlled_vs_proposed_raw | sentence_score | 0.1021 | 0.1283 |
| controlled_vs_proposed_raw | overall_quality | 0.1597 | 0.6061 |
| controlled_vs_candidate_no_context | context_relevance | 0.2422 | 9.7545 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1424 | 0.6891 |
| controlled_vs_candidate_no_context | naturalness | 0.0422 | 0.0497 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3162 | 16.6933 |
| controlled_vs_candidate_no_context | context_overlap | 0.0696 | 1.8053 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1615 | 1.3024 |
| controlled_vs_candidate_no_context | persona_style | 0.0662 | 0.1232 |
| controlled_vs_candidate_no_context | distinct1 | 0.0051 | 0.0054 |
| controlled_vs_candidate_no_context | length_score | 0.1625 | 0.3679 |
| controlled_vs_candidate_no_context | sentence_score | 0.0729 | 0.0884 |
| controlled_vs_candidate_no_context | overall_quality | 0.1733 | 0.6939 |
| controlled_vs_baseline_no_context | context_relevance | 0.2368 | 7.8212 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1727 | 0.9791 |
| controlled_vs_baseline_no_context | naturalness | -0.0038 | -0.0042 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3089 | 11.7904 |
| controlled_vs_baseline_no_context | context_overlap | 0.0684 | 1.7208 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2143 | 3.0084 |
| controlled_vs_baseline_no_context | persona_style | 0.0066 | 0.0110 |
| controlled_vs_baseline_no_context | distinct1 | -0.0292 | -0.0298 |
| controlled_vs_baseline_no_context | length_score | 0.0250 | 0.0432 |
| controlled_vs_baseline_no_context | sentence_score | 0.0292 | 0.0336 |
| controlled_vs_baseline_no_context | overall_quality | 0.1739 | 0.6975 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0297 | 0.1112 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0147 | -0.0420 |
| controlled_alt_vs_controlled_default | naturalness | -0.0170 | -0.0190 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0363 | 0.1083 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0143 | 0.1319 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0210 | -0.0737 |
| controlled_alt_vs_controlled_default | persona_style | 0.0108 | 0.0179 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0112 | -0.0118 |
| controlled_alt_vs_controlled_default | length_score | -0.0625 | -0.1034 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0000 | 0.0000 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0045 | 0.0107 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1869 | 1.7011 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1884 | 1.2887 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0417 | 0.0500 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2410 | 1.8489 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0605 | 0.9757 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.2139 | 4.2275 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0862 | 0.1631 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0040 | -0.0042 |
| controlled_alt_vs_proposed_raw | length_score | 0.1653 | 0.4391 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1021 | 0.1283 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1642 | 0.6233 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2719 | 10.9503 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1278 | 0.6181 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0252 | 0.0297 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3525 | 18.6100 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0839 | 2.1754 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1405 | 1.1328 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0770 | 0.1432 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0061 | -0.0065 |
| controlled_alt_vs_candidate_no_context | length_score | 0.1000 | 0.2264 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.0729 | 0.0884 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1779 | 0.7121 |
| controlled_alt_vs_baseline_no_context | context_relevance | 0.2665 | 8.8021 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 0.1581 | 0.8960 |
| controlled_alt_vs_baseline_no_context | naturalness | -0.0208 | -0.0232 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 0.3452 | 13.1759 |
| controlled_alt_vs_baseline_no_context | context_overlap | 0.0827 | 2.0796 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 0.1933 | 2.7131 |
| controlled_alt_vs_baseline_no_context | persona_style | 0.0174 | 0.0291 |
| controlled_alt_vs_baseline_no_context | distinct1 | -0.0404 | -0.0412 |
| controlled_alt_vs_baseline_no_context | length_score | -0.0375 | -0.0647 |
| controlled_alt_vs_baseline_no_context | sentence_score | 0.0292 | 0.0336 |
| controlled_alt_vs_baseline_no_context | overall_quality | 0.1784 | 0.7157 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2368 | 7.8212 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1727 | 0.9791 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0038 | -0.0042 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3089 | 11.7904 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0684 | 1.7208 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2143 | 3.0084 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0066 | 0.0110 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0292 | -0.0298 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.0250 | 0.0432 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0292 | 0.0336 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1739 | 0.6975 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0850 | (0.0419, 0.1340) | 0.0000 | 0.0850 | (0.0251, 0.1466) | 0.0000 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0606 | (-0.1365, 0.0077) | 0.9580 | -0.0606 | (-0.1178, -0.0037) | 0.9813 |
| proposed_vs_candidate_no_context | naturalness | -0.0165 | (-0.0587, 0.0246) | 0.7760 | -0.0165 | (-0.0635, 0.0170) | 0.8130 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.1114 | (0.0540, 0.1742) | 0.0000 | 0.1114 | (0.0303, 0.1932) | 0.0003 |
| proposed_vs_candidate_no_context | context_overlap | 0.0234 | (0.0098, 0.0381) | 0.0003 | 0.0234 | (0.0120, 0.0333) | 0.0000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0734 | (-0.1627, 0.0050) | 0.9637 | -0.0734 | (-0.1504, -0.0021) | 0.9847 |
| proposed_vs_candidate_no_context | persona_style | -0.0092 | (-0.0701, 0.0431) | 0.5823 | -0.0092 | (-0.0706, 0.0329) | 0.5947 |
| proposed_vs_candidate_no_context | distinct1 | -0.0021 | (-0.0184, 0.0147) | 0.5900 | -0.0021 | (-0.0192, 0.0090) | 0.5953 |
| proposed_vs_candidate_no_context | length_score | -0.0653 | (-0.2250, 0.0889) | 0.7823 | -0.0653 | (-0.2315, 0.0644) | 0.8163 |
| proposed_vs_candidate_no_context | sentence_score | -0.0292 | (-0.1313, 0.0729) | 0.7740 | -0.0292 | (-0.1065, 0.0350) | 0.8567 |
| proposed_vs_candidate_no_context | overall_quality | 0.0137 | (-0.0293, 0.0549) | 0.2647 | 0.0137 | (-0.0150, 0.0377) | 0.1737 |
| proposed_vs_baseline_no_context | context_relevance | 0.0796 | (0.0348, 0.1293) | 0.0000 | 0.0796 | (0.0141, 0.1507) | 0.0023 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0303 | (-0.0842, 0.0141) | 0.8957 | -0.0303 | (-0.0952, 0.0142) | 0.8833 |
| proposed_vs_baseline_no_context | naturalness | -0.0624 | (-0.1014, -0.0255) | 1.0000 | -0.0624 | (-0.1249, -0.0134) | 0.9953 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.1042 | (0.0454, 0.1698) | 0.0000 | 0.1042 | (0.0083, 0.2000) | 0.0150 |
| proposed_vs_baseline_no_context | context_overlap | 0.0222 | (0.0067, 0.0389) | 0.0023 | 0.0222 | (0.0057, 0.0365) | 0.0050 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0206 | (-0.0891, 0.0304) | 0.7410 | -0.0206 | (-0.1000, 0.0346) | 0.7220 |
| proposed_vs_baseline_no_context | persona_style | -0.0688 | (-0.1553, 0.0025) | 0.9703 | -0.0688 | (-0.2120, 0.0102) | 0.9413 |
| proposed_vs_baseline_no_context | distinct1 | -0.0364 | (-0.0551, -0.0162) | 1.0000 | -0.0364 | (-0.0592, -0.0193) | 1.0000 |
| proposed_vs_baseline_no_context | length_score | -0.2028 | (-0.3361, -0.0667) | 0.9990 | -0.2028 | (-0.4299, -0.0138) | 0.9837 |
| proposed_vs_baseline_no_context | sentence_score | -0.0729 | (-0.1750, 0.0292) | 0.9473 | -0.0729 | (-0.2211, 0.0375) | 0.9087 |
| proposed_vs_baseline_no_context | overall_quality | 0.0142 | (-0.0258, 0.0547) | 0.2450 | 0.0142 | (-0.0411, 0.0690) | 0.3557 |
| controlled_vs_proposed_raw | context_relevance | 0.1572 | (0.1045, 0.2129) | 0.0000 | 0.1572 | (0.1047, 0.2134) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.2030 | (0.1310, 0.2807) | 0.0000 | 0.2030 | (0.1294, 0.2909) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0586 | (0.0173, 0.0978) | 0.0017 | 0.0586 | (0.0112, 0.1232) | 0.0033 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2047 | (0.1301, 0.2753) | 0.0000 | 0.2047 | (0.1302, 0.2814) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0462 | (0.0233, 0.0686) | 0.0000 | 0.0462 | (0.0326, 0.0633) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2349 | (0.1494, 0.3216) | 0.0000 | 0.2349 | (0.1541, 0.3331) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.0754 | (0.0035, 0.1612) | 0.0190 | 0.0754 | (-0.0130, 0.2236) | 0.0753 |
| controlled_vs_proposed_raw | distinct1 | 0.0072 | (-0.0131, 0.0263) | 0.2383 | 0.0072 | (-0.0107, 0.0306) | 0.2340 |
| controlled_vs_proposed_raw | length_score | 0.2278 | (0.0736, 0.3833) | 0.0017 | 0.2278 | (0.0613, 0.4576) | 0.0007 |
| controlled_vs_proposed_raw | sentence_score | 0.1021 | (0.0146, 0.1896) | 0.0147 | 0.1021 | (0.0233, 0.2227) | 0.0223 |
| controlled_vs_proposed_raw | overall_quality | 0.1597 | (0.1073, 0.2123) | 0.0000 | 0.1597 | (0.1060, 0.2235) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2422 | (0.1984, 0.2876) | 0.0000 | 0.2422 | (0.2020, 0.2799) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1424 | (0.0556, 0.2283) | 0.0003 | 0.1424 | (0.0212, 0.2716) | 0.0140 |
| controlled_vs_candidate_no_context | naturalness | 0.0422 | (-0.0035, 0.0876) | 0.0367 | 0.0422 | (-0.0110, 0.1009) | 0.0617 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3162 | (0.2601, 0.3763) | 0.0000 | 0.3162 | (0.2654, 0.3690) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0696 | (0.0508, 0.0892) | 0.0000 | 0.0696 | (0.0564, 0.0816) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1615 | (0.0550, 0.2728) | 0.0023 | 0.1615 | (0.0091, 0.3207) | 0.0183 |
| controlled_vs_candidate_no_context | persona_style | 0.0662 | (0.0118, 0.1366) | 0.0043 | 0.0662 | (0.0089, 0.1651) | 0.0037 |
| controlled_vs_candidate_no_context | distinct1 | 0.0051 | (-0.0150, 0.0242) | 0.3137 | 0.0051 | (-0.0129, 0.0253) | 0.3060 |
| controlled_vs_candidate_no_context | length_score | 0.1625 | (-0.0042, 0.3222) | 0.0277 | 0.1625 | (-0.0410, 0.3758) | 0.0580 |
| controlled_vs_candidate_no_context | sentence_score | 0.0729 | (-0.0292, 0.1750) | 0.0917 | 0.0729 | (-0.0319, 0.1833) | 0.1307 |
| controlled_vs_candidate_no_context | overall_quality | 0.1733 | (0.1278, 0.2206) | 0.0000 | 0.1733 | (0.1226, 0.2284) | 0.0000 |
| controlled_vs_baseline_no_context | context_relevance | 0.2368 | (0.1905, 0.2867) | 0.0000 | 0.2368 | (0.1881, 0.2889) | 0.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1727 | (0.1020, 0.2429) | 0.0000 | 0.1727 | (0.0772, 0.2618) | 0.0010 |
| controlled_vs_baseline_no_context | naturalness | -0.0038 | (-0.0346, 0.0267) | 0.5823 | -0.0038 | (-0.0411, 0.0292) | 0.5550 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3089 | (0.2461, 0.3732) | 0.0000 | 0.3089 | (0.2386, 0.3799) | 0.0000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0684 | (0.0519, 0.0851) | 0.0000 | 0.0684 | (0.0545, 0.0788) | 0.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2143 | (0.1363, 0.3018) | 0.0000 | 0.2143 | (0.0943, 0.3259) | 0.0013 |
| controlled_vs_baseline_no_context | persona_style | 0.0066 | (-0.0151, 0.0417) | 0.4070 | 0.0066 | (-0.0167, 0.0428) | 0.3873 |
| controlled_vs_baseline_no_context | distinct1 | -0.0292 | (-0.0466, -0.0118) | 0.9990 | -0.0292 | (-0.0398, -0.0155) | 0.9997 |
| controlled_vs_baseline_no_context | length_score | 0.0250 | (-0.0972, 0.1458) | 0.3527 | 0.0250 | (-0.1174, 0.1530) | 0.3603 |
| controlled_vs_baseline_no_context | sentence_score | 0.0292 | (-0.0729, 0.1313) | 0.3203 | 0.0292 | (-0.0761, 0.1225) | 0.3260 |
| controlled_vs_baseline_no_context | overall_quality | 0.1739 | (0.1363, 0.2148) | 0.0000 | 0.1739 | (0.1254, 0.2174) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0297 | (-0.0292, 0.0873) | 0.1683 | 0.0297 | (-0.0108, 0.0767) | 0.0733 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0147 | (-0.0748, 0.0460) | 0.6823 | -0.0147 | (-0.0780, 0.0526) | 0.6797 |
| controlled_alt_vs_controlled_default | naturalness | -0.0170 | (-0.0494, 0.0160) | 0.8367 | -0.0170 | (-0.0658, 0.0198) | 0.8047 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0363 | (-0.0414, 0.1117) | 0.1927 | 0.0363 | (-0.0195, 0.1006) | 0.1303 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0143 | (-0.0105, 0.0405) | 0.1283 | 0.0143 | (-0.0045, 0.0327) | 0.0683 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0210 | (-0.0950, 0.0524) | 0.7233 | -0.0210 | (-0.0957, 0.0586) | 0.7090 |
| controlled_alt_vs_controlled_default | persona_style | 0.0108 | (-0.0209, 0.0425) | 0.2683 | 0.0108 | (-0.0206, 0.0486) | 0.2403 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0112 | (-0.0293, 0.0084) | 0.8767 | -0.0112 | (-0.0230, -0.0020) | 0.9917 |
| controlled_alt_vs_controlled_default | length_score | -0.0625 | (-0.2139, 0.0917) | 0.7957 | -0.0625 | (-0.2842, 0.1070) | 0.7373 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0000 | (-0.0729, 0.0729) | 0.5850 | 0.0000 | (-0.0538, 0.0667) | 0.6137 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0045 | (-0.0357, 0.0406) | 0.4093 | 0.0045 | (-0.0301, 0.0421) | 0.3953 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1869 | (0.1450, 0.2282) | 0.0000 | 0.1869 | (0.1584, 0.2239) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1884 | (0.1317, 0.2472) | 0.0000 | 0.1884 | (0.1291, 0.2610) | 0.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0417 | (-0.0005, 0.0834) | 0.0270 | 0.0417 | (-0.0075, 0.0881) | 0.0480 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2410 | (0.1868, 0.2941) | 0.0000 | 0.2410 | (0.2013, 0.2930) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0605 | (0.0384, 0.0855) | 0.0000 | 0.0605 | (0.0383, 0.0856) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.2139 | (0.1534, 0.2772) | 0.0000 | 0.2139 | (0.1469, 0.3030) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0862 | (-0.0078, 0.1918) | 0.0337 | 0.0862 | (-0.0057, 0.2404) | 0.1037 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0040 | (-0.0238, 0.0176) | 0.6457 | -0.0040 | (-0.0225, 0.0183) | 0.6507 |
| controlled_alt_vs_proposed_raw | length_score | 0.1653 | (0.0069, 0.3209) | 0.0193 | 0.1653 | (-0.0240, 0.3106) | 0.0427 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1021 | (-0.0146, 0.2042) | 0.0520 | 0.1021 | (0.0125, 0.2227) | 0.0197 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1642 | (0.1265, 0.2026) | 0.0000 | 0.1642 | (0.1293, 0.2099) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2719 | (0.2344, 0.3143) | 0.0000 | 0.2719 | (0.2322, 0.3093) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1278 | (0.0501, 0.1967) | 0.0010 | 0.1278 | (0.0277, 0.2382) | 0.0047 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0252 | (-0.0222, 0.0702) | 0.1483 | 0.0252 | (-0.0410, 0.0789) | 0.2227 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3525 | (0.3018, 0.4056) | 0.0000 | 0.3525 | (0.2977, 0.4032) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0839 | (0.0654, 0.1045) | 0.0000 | 0.0839 | (0.0632, 0.1053) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1405 | (0.0490, 0.2304) | 0.0027 | 0.1405 | (0.0201, 0.2769) | 0.0067 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0770 | (0.0082, 0.1519) | 0.0143 | 0.0770 | (0.0074, 0.1929) | 0.0063 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0061 | (-0.0239, 0.0120) | 0.7390 | -0.0061 | (-0.0293, 0.0158) | 0.6863 |
| controlled_alt_vs_candidate_no_context | length_score | 0.1000 | (-0.0958, 0.2889) | 0.1623 | 0.1000 | (-0.1622, 0.3124) | 0.2113 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.0729 | (-0.0146, 0.1604) | 0.0840 | 0.0729 | (-0.0125, 0.1750) | 0.0627 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1779 | (0.1436, 0.2125) | 0.0000 | 0.1779 | (0.1536, 0.2156) | 0.0000 |
| controlled_alt_vs_baseline_no_context | context_relevance | 0.2665 | (0.2253, 0.3087) | 0.0000 | 0.2665 | (0.2152, 0.3144) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 0.1581 | (0.1149, 0.2069) | 0.0000 | 0.1581 | (0.0932, 0.2356) | 0.0000 |
| controlled_alt_vs_baseline_no_context | naturalness | -0.0208 | (-0.0582, 0.0155) | 0.8483 | -0.0208 | (-0.0700, 0.0240) | 0.8080 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 0.3452 | (0.2910, 0.4009) | 0.0000 | 0.3452 | (0.2760, 0.4168) | 0.0000 |
| controlled_alt_vs_baseline_no_context | context_overlap | 0.0827 | (0.0625, 0.1067) | 0.0000 | 0.0827 | (0.0623, 0.1054) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 0.1933 | (0.1403, 0.2534) | 0.0000 | 0.1933 | (0.1069, 0.2907) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_style | 0.0174 | (-0.0278, 0.0660) | 0.2470 | 0.0174 | (-0.0313, 0.0795) | 0.3050 |
| controlled_alt_vs_baseline_no_context | distinct1 | -0.0404 | (-0.0557, -0.0249) | 1.0000 | -0.0404 | (-0.0525, -0.0303) | 1.0000 |
| controlled_alt_vs_baseline_no_context | length_score | -0.0375 | (-0.2056, 0.1236) | 0.6640 | -0.0375 | (-0.2746, 0.1736) | 0.6590 |
| controlled_alt_vs_baseline_no_context | sentence_score | 0.0292 | (-0.0875, 0.1458) | 0.3500 | 0.0292 | (-0.0457, 0.1167) | 0.3017 |
| controlled_alt_vs_baseline_no_context | overall_quality | 0.1784 | (0.1477, 0.2105) | 0.0000 | 0.1784 | (0.1362, 0.2156) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2368 | (0.1914, 0.2860) | 0.0000 | 0.2368 | (0.1884, 0.2890) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1727 | (0.1071, 0.2429) | 0.0000 | 0.1727 | (0.0768, 0.2625) | 0.0003 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0038 | (-0.0333, 0.0287) | 0.5933 | -0.0038 | (-0.0413, 0.0289) | 0.5667 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3089 | (0.2462, 0.3720) | 0.0000 | 0.3089 | (0.2388, 0.3829) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0684 | (0.0517, 0.0848) | 0.0000 | 0.0684 | (0.0547, 0.0789) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2143 | (0.1331, 0.3002) | 0.0000 | 0.2143 | (0.0992, 0.3259) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0066 | (-0.0151, 0.0383) | 0.4083 | 0.0066 | (-0.0167, 0.0417) | 0.4050 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0292 | (-0.0452, -0.0113) | 0.9990 | -0.0292 | (-0.0400, -0.0153) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.0250 | (-0.0972, 0.1417) | 0.3530 | 0.0250 | (-0.1180, 0.1533) | 0.3543 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0292 | (-0.0729, 0.1313) | 0.3383 | 0.0292 | (-0.0761, 0.1217) | 0.3343 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1739 | (0.1349, 0.2135) | 0.0000 | 0.1739 | (0.1262, 0.2206) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 13 | 4 | 7 | 0.6875 | 0.7647 |
| proposed_vs_candidate_no_context | persona_consistency | 6 | 7 | 11 | 0.4792 | 0.4615 |
| proposed_vs_candidate_no_context | naturalness | 9 | 8 | 7 | 0.5208 | 0.5294 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 11 | 0 | 13 | 0.7292 | 1.0000 |
| proposed_vs_candidate_no_context | context_overlap | 13 | 4 | 7 | 0.6875 | 0.7647 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 4 | 6 | 14 | 0.4583 | 0.4000 |
| proposed_vs_candidate_no_context | persona_style | 4 | 2 | 18 | 0.5417 | 0.6667 |
| proposed_vs_candidate_no_context | distinct1 | 6 | 7 | 11 | 0.4792 | 0.4615 |
| proposed_vs_candidate_no_context | length_score | 9 | 8 | 7 | 0.5208 | 0.5294 |
| proposed_vs_candidate_no_context | sentence_score | 5 | 7 | 12 | 0.4583 | 0.4167 |
| proposed_vs_candidate_no_context | overall_quality | 10 | 7 | 7 | 0.5625 | 0.5882 |
| proposed_vs_baseline_no_context | context_relevance | 17 | 7 | 0 | 0.7083 | 0.7083 |
| proposed_vs_baseline_no_context | persona_consistency | 5 | 9 | 10 | 0.4167 | 0.3571 |
| proposed_vs_baseline_no_context | naturalness | 6 | 17 | 1 | 0.2708 | 0.2609 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 10 | 1 | 13 | 0.6875 | 0.9091 |
| proposed_vs_baseline_no_context | context_overlap | 16 | 8 | 0 | 0.6667 | 0.6667 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 4 | 4 | 16 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context | persona_style | 1 | 5 | 18 | 0.4167 | 0.1667 |
| proposed_vs_baseline_no_context | distinct1 | 4 | 17 | 3 | 0.2292 | 0.1905 |
| proposed_vs_baseline_no_context | length_score | 5 | 18 | 1 | 0.2292 | 0.2174 |
| proposed_vs_baseline_no_context | sentence_score | 4 | 9 | 11 | 0.3958 | 0.3077 |
| proposed_vs_baseline_no_context | overall_quality | 11 | 13 | 0 | 0.4583 | 0.4583 |
| controlled_vs_proposed_raw | context_relevance | 20 | 4 | 0 | 0.8333 | 0.8333 |
| controlled_vs_proposed_raw | persona_consistency | 20 | 1 | 3 | 0.8958 | 0.9524 |
| controlled_vs_proposed_raw | naturalness | 18 | 6 | 0 | 0.7500 | 0.7500 |
| controlled_vs_proposed_raw | context_keyword_coverage | 20 | 2 | 2 | 0.8750 | 0.9091 |
| controlled_vs_proposed_raw | context_overlap | 19 | 5 | 0 | 0.7917 | 0.7917 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 20 | 1 | 3 | 0.8958 | 0.9524 |
| controlled_vs_proposed_raw | persona_style | 5 | 2 | 17 | 0.5625 | 0.7143 |
| controlled_vs_proposed_raw | distinct1 | 13 | 9 | 2 | 0.5833 | 0.5909 |
| controlled_vs_proposed_raw | length_score | 17 | 6 | 1 | 0.7292 | 0.7391 |
| controlled_vs_proposed_raw | sentence_score | 9 | 2 | 13 | 0.6458 | 0.8182 |
| controlled_vs_proposed_raw | overall_quality | 21 | 3 | 0 | 0.8750 | 0.8750 |
| controlled_vs_candidate_no_context | context_relevance | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 18 | 5 | 1 | 0.7708 | 0.7826 |
| controlled_vs_candidate_no_context | naturalness | 16 | 7 | 1 | 0.6875 | 0.6957 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 18 | 5 | 1 | 0.7708 | 0.7826 |
| controlled_vs_candidate_no_context | persona_style | 5 | 1 | 18 | 0.5833 | 0.8333 |
| controlled_vs_candidate_no_context | distinct1 | 12 | 10 | 2 | 0.5417 | 0.5455 |
| controlled_vs_candidate_no_context | length_score | 17 | 6 | 1 | 0.7292 | 0.7391 |
| controlled_vs_candidate_no_context | sentence_score | 9 | 4 | 11 | 0.6042 | 0.6923 |
| controlled_vs_candidate_no_context | overall_quality | 22 | 2 | 0 | 0.9167 | 0.9167 |
| controlled_vs_baseline_no_context | context_relevance | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 21 | 2 | 1 | 0.8958 | 0.9130 |
| controlled_vs_baseline_no_context | naturalness | 11 | 13 | 0 | 0.4583 | 0.4583 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | context_overlap | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 21 | 2 | 1 | 0.8958 | 0.9130 |
| controlled_vs_baseline_no_context | persona_style | 1 | 2 | 21 | 0.4792 | 0.3333 |
| controlled_vs_baseline_no_context | distinct1 | 4 | 17 | 3 | 0.2292 | 0.1905 |
| controlled_vs_baseline_no_context | length_score | 11 | 10 | 3 | 0.5208 | 0.5238 |
| controlled_vs_baseline_no_context | sentence_score | 7 | 5 | 12 | 0.5417 | 0.5833 |
| controlled_vs_baseline_no_context | overall_quality | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_alt_vs_controlled_default | context_relevance | 12 | 12 | 0 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | persona_consistency | 5 | 8 | 11 | 0.4375 | 0.3846 |
| controlled_alt_vs_controlled_default | naturalness | 10 | 14 | 0 | 0.4167 | 0.4167 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 10 | 9 | 5 | 0.5208 | 0.5263 |
| controlled_alt_vs_controlled_default | context_overlap | 13 | 11 | 0 | 0.5417 | 0.5417 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 4 | 7 | 13 | 0.4375 | 0.3636 |
| controlled_alt_vs_controlled_default | persona_style | 3 | 1 | 20 | 0.5417 | 0.7500 |
| controlled_alt_vs_controlled_default | distinct1 | 7 | 16 | 1 | 0.3125 | 0.3043 |
| controlled_alt_vs_controlled_default | length_score | 12 | 12 | 0 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | sentence_score | 3 | 3 | 18 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | overall_quality | 10 | 14 | 0 | 0.4167 | 0.4167 |
| controlled_alt_vs_proposed_raw | context_relevance | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_alt_vs_proposed_raw | persona_consistency | 20 | 0 | 4 | 0.9167 | 1.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 16 | 8 | 0 | 0.6667 | 0.6667 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 22 | 1 | 1 | 0.9375 | 0.9565 |
| controlled_alt_vs_proposed_raw | context_overlap | 22 | 2 | 0 | 0.9167 | 0.9167 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 20 | 0 | 4 | 0.9167 | 1.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 6 | 2 | 16 | 0.5833 | 0.7500 |
| controlled_alt_vs_proposed_raw | distinct1 | 9 | 13 | 2 | 0.4167 | 0.4091 |
| controlled_alt_vs_proposed_raw | length_score | 15 | 9 | 0 | 0.6250 | 0.6250 |
| controlled_alt_vs_proposed_raw | sentence_score | 12 | 5 | 7 | 0.6458 | 0.7059 |
| controlled_alt_vs_proposed_raw | overall_quality | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 18 | 4 | 2 | 0.7917 | 0.8182 |
| controlled_alt_vs_candidate_no_context | naturalness | 13 | 11 | 0 | 0.5417 | 0.5417 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 17 | 4 | 3 | 0.7708 | 0.8095 |
| controlled_alt_vs_candidate_no_context | persona_style | 7 | 1 | 16 | 0.6250 | 0.8750 |
| controlled_alt_vs_candidate_no_context | distinct1 | 9 | 12 | 3 | 0.4375 | 0.4286 |
| controlled_alt_vs_candidate_no_context | length_score | 12 | 12 | 0 | 0.5000 | 0.5000 |
| controlled_alt_vs_candidate_no_context | sentence_score | 8 | 3 | 13 | 0.6042 | 0.7273 |
| controlled_alt_vs_candidate_no_context | overall_quality | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context | context_relevance | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 21 | 0 | 3 | 0.9375 | 1.0000 |
| controlled_alt_vs_baseline_no_context | naturalness | 10 | 14 | 0 | 0.4167 | 0.4167 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context | context_overlap | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 20 | 0 | 4 | 0.9167 | 1.0000 |
| controlled_alt_vs_baseline_no_context | persona_style | 3 | 1 | 20 | 0.5417 | 0.7500 |
| controlled_alt_vs_baseline_no_context | distinct1 | 2 | 19 | 3 | 0.1458 | 0.0952 |
| controlled_alt_vs_baseline_no_context | length_score | 13 | 11 | 0 | 0.5417 | 0.5417 |
| controlled_alt_vs_baseline_no_context | sentence_score | 9 | 7 | 8 | 0.5417 | 0.5625 |
| controlled_alt_vs_baseline_no_context | overall_quality | 24 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 24 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 21 | 2 | 1 | 0.8958 | 0.9130 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 11 | 13 | 0 | 0.4583 | 0.4583 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 24 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 23 | 1 | 0 | 0.9583 | 0.9583 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 21 | 2 | 1 | 0.8958 | 0.9130 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 1 | 2 | 21 | 0.4792 | 0.3333 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | 4 | 17 | 3 | 0.2292 | 0.1905 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 11 | 10 | 3 | 0.5208 | 0.5238 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 7 | 5 | 12 | 0.5417 | 0.5833 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 23 | 1 | 0 | 0.9583 | 0.9583 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.1667 | 0.5417 | 0.4583 |
| proposed_contextual_controlled_alt | 0.0000 | 0.0000 | 0.1250 | 0.5417 | 0.4583 |
| proposed_contextual | 0.0000 | 0.0000 | 0.4583 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.3750 | 0.0000 | 0.0000 |
| baseline_no_context | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `8`
- Unique template signatures: `23`
- Template signature ratio: `0.9583`
- Effective sample size by source clustering: `7.20`
- Effective sample size by template-signature clustering: `22.15`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: unavailable (disabled_by_flag).

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
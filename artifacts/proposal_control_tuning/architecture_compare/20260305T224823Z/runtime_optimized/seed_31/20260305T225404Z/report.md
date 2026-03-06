# Proposal Alignment Evaluation Report

- Run ID: `20260305T225404Z`
- Generated: `2026-03-05T22:59:46.079068+00:00`
- Scenarios: `artifacts\proposal_control_tuning\architecture_compare\20260305T224823Z\runtime_optimized\seed_31\20260305T225404Z\scenarios.jsonl`
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
| proposed_contextual_controlled | 0.2980 (0.2599, 0.3385) | 0.3729 (0.3224, 0.4261) | 0.8618 (0.8326, 0.8904) | 0.4405 (0.4187, 0.4623) | n/a |
| proposed_contextual_controlled_alt | 0.2537 (0.2333, 0.2756) | 0.3443 (0.2914, 0.4041) | 0.8909 (0.8678, 0.9120) | 0.4147 (0.3943, 0.4384) | n/a |
| proposed_contextual | 0.0797 (0.0434, 0.1195) | 0.1543 (0.1026, 0.2106) | 0.8250 (0.7950, 0.8561) | 0.2510 (0.2207, 0.2856) | n/a |
| candidate_no_context | 0.0302 (0.0169, 0.0444) | 0.1925 (0.1423, 0.2459) | 0.8327 (0.7998, 0.8661) | 0.2434 (0.2154, 0.2729) | n/a |
| baseline_no_context | 0.0496 (0.0329, 0.0689) | 0.1837 (0.1361, 0.2382) | 0.9047 (0.8820, 0.9266) | 0.2631 (0.2431, 0.2845) | n/a |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0495 | 1.6414 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0382 | -0.1985 |
| proposed_vs_candidate_no_context | naturalness | -0.0077 | -0.0092 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0634 | 2.4428 |
| proposed_vs_candidate_no_context | context_overlap | 0.0171 | 0.4285 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0387 | -0.3750 |
| proposed_vs_candidate_no_context | persona_style | -0.0363 | -0.0660 |
| proposed_vs_candidate_no_context | distinct1 | 0.0008 | 0.0008 |
| proposed_vs_candidate_no_context | length_score | -0.0181 | -0.0480 |
| proposed_vs_candidate_no_context | sentence_score | -0.0438 | -0.0540 |
| proposed_vs_candidate_no_context | overall_quality | 0.0076 | 0.0311 |
| proposed_vs_baseline_no_context | context_relevance | 0.0301 | 0.6067 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0294 | -0.1598 |
| proposed_vs_baseline_no_context | naturalness | -0.0797 | -0.0881 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0366 | 0.6926 |
| proposed_vs_baseline_no_context | context_overlap | 0.0150 | 0.3556 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0183 | -0.2206 |
| proposed_vs_baseline_no_context | persona_style | -0.0737 | -0.1256 |
| proposed_vs_baseline_no_context | distinct1 | -0.0459 | -0.0465 |
| proposed_vs_baseline_no_context | length_score | -0.2556 | -0.4163 |
| proposed_vs_baseline_no_context | sentence_score | -0.1021 | -0.1175 |
| proposed_vs_baseline_no_context | overall_quality | -0.0121 | -0.0459 |
| controlled_vs_proposed_raw | context_relevance | 0.2183 | 2.7400 |
| controlled_vs_proposed_raw | persona_consistency | 0.2186 | 1.4168 |
| controlled_vs_proposed_raw | naturalness | 0.0368 | 0.0445 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2871 | 3.2134 |
| controlled_vs_proposed_raw | context_overlap | 0.0579 | 1.0131 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2508 | 3.8892 |
| controlled_vs_proposed_raw | persona_style | 0.0899 | 0.1750 |
| controlled_vs_proposed_raw | distinct1 | -0.0015 | -0.0016 |
| controlled_vs_proposed_raw | length_score | 0.1139 | 0.3178 |
| controlled_vs_proposed_raw | sentence_score | 0.1458 | 0.1902 |
| controlled_vs_proposed_raw | overall_quality | 0.1895 | 0.7551 |
| controlled_vs_candidate_no_context | context_relevance | 0.2678 | 8.8790 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1804 | 0.9372 |
| controlled_vs_candidate_no_context | naturalness | 0.0291 | 0.0349 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3504 | 13.5061 |
| controlled_vs_candidate_no_context | context_overlap | 0.0750 | 1.8756 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2121 | 2.0558 |
| controlled_vs_candidate_no_context | persona_style | 0.0536 | 0.0975 |
| controlled_vs_candidate_no_context | distinct1 | -0.0007 | -0.0008 |
| controlled_vs_candidate_no_context | length_score | 0.0958 | 0.2546 |
| controlled_vs_candidate_no_context | sentence_score | 0.1021 | 0.1260 |
| controlled_vs_candidate_no_context | overall_quality | 0.1971 | 0.8098 |
| controlled_vs_baseline_no_context | context_relevance | 0.2484 | 5.0089 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1893 | 1.0305 |
| controlled_vs_baseline_no_context | naturalness | -0.0429 | -0.0474 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3236 | 6.1316 |
| controlled_vs_baseline_no_context | context_overlap | 0.0729 | 1.7289 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2325 | 2.8106 |
| controlled_vs_baseline_no_context | persona_style | 0.0161 | 0.0275 |
| controlled_vs_baseline_no_context | distinct1 | -0.0474 | -0.0480 |
| controlled_vs_baseline_no_context | length_score | -0.1417 | -0.2308 |
| controlled_vs_baseline_no_context | sentence_score | 0.0437 | 0.0504 |
| controlled_vs_baseline_no_context | overall_quality | 0.1775 | 0.6745 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0442 | -0.1485 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0286 | -0.0768 |
| controlled_alt_vs_controlled_default | naturalness | 0.0291 | 0.0338 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0612 | -0.1627 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0046 | -0.0398 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0369 | -0.1171 |
| controlled_alt_vs_controlled_default | persona_style | 0.0044 | 0.0073 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0082 | 0.0087 |
| controlled_alt_vs_controlled_default | length_score | 0.1292 | 0.2735 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0000 | 0.0000 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0258 | -0.0586 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1741 | 2.1848 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1900 | 1.2312 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0659 | 0.0798 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2258 | 2.5279 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0533 | 0.9330 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.2139 | 3.3169 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0943 | 0.1836 |
| controlled_alt_vs_proposed_raw | distinct1 | 0.0066 | 0.0071 |
| controlled_alt_vs_proposed_raw | length_score | 0.2431 | 0.6783 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1458 | 0.1902 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1637 | 0.6523 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2236 | 7.4123 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1518 | 0.7884 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0582 | 0.0699 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.2892 | 11.1460 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0705 | 1.7613 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1752 | 1.6981 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0580 | 0.1056 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0074 | 0.0079 |
| controlled_alt_vs_candidate_no_context | length_score | 0.2250 | 0.5978 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1021 | 0.1260 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1713 | 0.7038 |
| controlled_alt_vs_baseline_no_context | context_relevance | 0.2042 | 4.1168 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 0.1606 | 0.8746 |
| controlled_alt_vs_baseline_no_context | naturalness | -0.0138 | -0.0153 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 0.2624 | 4.9713 |
| controlled_alt_vs_baseline_no_context | context_overlap | 0.0683 | 1.6204 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 0.1956 | 2.3645 |
| controlled_alt_vs_baseline_no_context | persona_style | 0.0206 | 0.0350 |
| controlled_alt_vs_baseline_no_context | distinct1 | -0.0392 | -0.0397 |
| controlled_alt_vs_baseline_no_context | length_score | -0.0125 | -0.0204 |
| controlled_alt_vs_baseline_no_context | sentence_score | 0.0437 | 0.0504 |
| controlled_alt_vs_baseline_no_context | overall_quality | 0.1516 | 0.5764 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2484 | 5.0089 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1893 | 1.0305 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0429 | -0.0474 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3236 | 6.1316 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0729 | 1.7289 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2325 | 2.8106 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0161 | 0.0275 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0474 | -0.0480 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | -0.1417 | -0.2308 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0437 | 0.0504 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1775 | 0.6745 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0495 | (0.0131, 0.0908) | 0.0027 | 0.0495 | (0.0012, 0.0991) | 0.0217 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0382 | (-0.1100, 0.0337) | 0.8583 | -0.0382 | (-0.0902, 0.0043) | 0.9630 |
| proposed_vs_candidate_no_context | naturalness | -0.0077 | (-0.0413, 0.0241) | 0.6520 | -0.0077 | (-0.0517, 0.0235) | 0.6567 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0634 | (0.0130, 0.1190) | 0.0053 | 0.0634 | (-0.0012, 0.1280) | 0.0300 |
| proposed_vs_candidate_no_context | context_overlap | 0.0171 | (0.0056, 0.0287) | 0.0010 | 0.0171 | (0.0034, 0.0280) | 0.0063 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0387 | (-0.1200, 0.0467) | 0.8130 | -0.0387 | (-0.1066, 0.0162) | 0.9097 |
| proposed_vs_candidate_no_context | persona_style | -0.0363 | (-0.1094, 0.0150) | 0.8893 | -0.0363 | (-0.1120, 0.0208) | 0.8653 |
| proposed_vs_candidate_no_context | distinct1 | 0.0008 | (-0.0158, 0.0187) | 0.4637 | 0.0008 | (-0.0126, 0.0128) | 0.4517 |
| proposed_vs_candidate_no_context | length_score | -0.0181 | (-0.1444, 0.0973) | 0.6027 | -0.0181 | (-0.1817, 0.1104) | 0.6150 |
| proposed_vs_candidate_no_context | sentence_score | -0.0437 | (-0.1167, 0.0292) | 0.9117 | -0.0437 | (-0.1225, 0.0389) | 0.8843 |
| proposed_vs_candidate_no_context | overall_quality | 0.0076 | (-0.0281, 0.0434) | 0.3320 | 0.0076 | (-0.0228, 0.0314) | 0.3450 |
| proposed_vs_baseline_no_context | context_relevance | 0.0301 | (-0.0119, 0.0773) | 0.0943 | 0.0301 | (-0.0272, 0.0861) | 0.1710 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0294 | (-0.0996, 0.0371) | 0.8103 | -0.0294 | (-0.1267, 0.0332) | 0.7770 |
| proposed_vs_baseline_no_context | naturalness | -0.0797 | (-0.1124, -0.0468) | 1.0000 | -0.0797 | (-0.1255, -0.0427) | 1.0000 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0366 | (-0.0206, 0.1019) | 0.1257 | 0.0366 | (-0.0422, 0.1116) | 0.2060 |
| proposed_vs_baseline_no_context | context_overlap | 0.0150 | (0.0002, 0.0305) | 0.0230 | 0.0150 | (-0.0020, 0.0279) | 0.0383 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0183 | (-0.0990, 0.0595) | 0.6890 | -0.0183 | (-0.1230, 0.0513) | 0.6640 |
| proposed_vs_baseline_no_context | persona_style | -0.0737 | (-0.1655, 0.0058) | 0.9653 | -0.0737 | (-0.2182, 0.0089) | 0.9413 |
| proposed_vs_baseline_no_context | distinct1 | -0.0459 | (-0.0603, -0.0306) | 1.0000 | -0.0459 | (-0.0606, -0.0327) | 1.0000 |
| proposed_vs_baseline_no_context | length_score | -0.2556 | (-0.3722, -0.1361) | 1.0000 | -0.2556 | (-0.4304, -0.1186) | 1.0000 |
| proposed_vs_baseline_no_context | sentence_score | -0.1021 | (-0.1899, 0.0000) | 0.9787 | -0.1021 | (-0.2380, 0.0130) | 0.9710 |
| proposed_vs_baseline_no_context | overall_quality | -0.0121 | (-0.0537, 0.0293) | 0.7230 | -0.0121 | (-0.0690, 0.0317) | 0.6990 |
| controlled_vs_proposed_raw | context_relevance | 0.2183 | (0.1653, 0.2642) | 0.0000 | 0.2183 | (0.1684, 0.2816) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.2186 | (0.1376, 0.2975) | 0.0000 | 0.2186 | (0.1526, 0.3091) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0368 | (0.0012, 0.0737) | 0.0220 | 0.0368 | (-0.0067, 0.0915) | 0.0533 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2871 | (0.2158, 0.3535) | 0.0000 | 0.2871 | (0.2182, 0.3732) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0579 | (0.0407, 0.0809) | 0.0000 | 0.0579 | (0.0387, 0.0826) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2508 | (0.1498, 0.3471) | 0.0000 | 0.2508 | (0.1753, 0.3571) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.0899 | (-0.0020, 0.1895) | 0.0293 | 0.0899 | (-0.0224, 0.2582) | 0.0743 |
| controlled_vs_proposed_raw | distinct1 | -0.0015 | (-0.0189, 0.0140) | 0.5663 | -0.0015 | (-0.0172, 0.0138) | 0.5623 |
| controlled_vs_proposed_raw | length_score | 0.1139 | (-0.0347, 0.2708) | 0.0710 | 0.1139 | (-0.0847, 0.3123) | 0.1130 |
| controlled_vs_proposed_raw | sentence_score | 0.1458 | (0.0437, 0.2479) | 0.0060 | 0.1458 | (0.0389, 0.2763) | 0.0030 |
| controlled_vs_proposed_raw | overall_quality | 0.1895 | (0.1502, 0.2297) | 0.0000 | 0.1895 | (0.1493, 0.2430) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2678 | (0.2301, 0.3084) | 0.0000 | 0.2678 | (0.2209, 0.3144) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1804 | (0.1056, 0.2565) | 0.0000 | 0.1804 | (0.1073, 0.2817) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.0291 | (-0.0043, 0.0638) | 0.0423 | 0.0291 | (-0.0183, 0.0676) | 0.0987 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3504 | (0.2966, 0.4025) | 0.0000 | 0.3504 | (0.2837, 0.4164) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0750 | (0.0589, 0.0947) | 0.0000 | 0.0750 | (0.0590, 0.0902) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2121 | (0.1260, 0.3073) | 0.0000 | 0.2121 | (0.1255, 0.3348) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.0536 | (-0.0125, 0.1387) | 0.0773 | 0.0536 | (-0.0186, 0.1693) | 0.0947 |
| controlled_vs_candidate_no_context | distinct1 | -0.0007 | (-0.0191, 0.0174) | 0.5267 | -0.0007 | (-0.0152, 0.0102) | 0.5177 |
| controlled_vs_candidate_no_context | length_score | 0.0958 | (-0.0403, 0.2348) | 0.0840 | 0.0958 | (-0.0955, 0.2469) | 0.1447 |
| controlled_vs_candidate_no_context | sentence_score | 0.1021 | (0.0000, 0.1896) | 0.0257 | 0.1021 | (0.0140, 0.2100) | 0.0183 |
| controlled_vs_candidate_no_context | overall_quality | 0.1971 | (0.1616, 0.2305) | 0.0000 | 0.1971 | (0.1677, 0.2334) | 0.0000 |
| controlled_vs_baseline_no_context | context_relevance | 0.2484 | (0.2000, 0.2998) | 0.0000 | 0.2484 | (0.1835, 0.3117) | 0.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1893 | (0.1303, 0.2507) | 0.0000 | 0.1893 | (0.1106, 0.2629) | 0.0000 |
| controlled_vs_baseline_no_context | naturalness | -0.0429 | (-0.0802, -0.0033) | 0.9853 | -0.0429 | (-0.0897, 0.0003) | 0.9743 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3236 | (0.2576, 0.3911) | 0.0000 | 0.3236 | (0.2372, 0.4121) | 0.0000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0729 | (0.0529, 0.0988) | 0.0000 | 0.0729 | (0.0520, 0.0926) | 0.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2325 | (0.1599, 0.3093) | 0.0000 | 0.2325 | (0.1307, 0.3227) | 0.0000 |
| controlled_vs_baseline_no_context | persona_style | 0.0161 | (-0.0245, 0.0703) | 0.2573 | 0.0161 | (-0.0273, 0.0764) | 0.2753 |
| controlled_vs_baseline_no_context | distinct1 | -0.0474 | (-0.0634, -0.0327) | 1.0000 | -0.0474 | (-0.0618, -0.0326) | 1.0000 |
| controlled_vs_baseline_no_context | length_score | -0.1417 | (-0.3181, 0.0389) | 0.9407 | -0.1417 | (-0.3682, 0.0449) | 0.9293 |
| controlled_vs_baseline_no_context | sentence_score | 0.0437 | (-0.0437, 0.1313) | 0.1983 | 0.0437 | (-0.0519, 0.1400) | 0.2177 |
| controlled_vs_baseline_no_context | overall_quality | 0.1775 | (0.1499, 0.2042) | 0.0000 | 0.1775 | (0.1347, 0.2112) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0442 | (-0.0826, -0.0077) | 0.9907 | -0.0442 | (-0.1020, -0.0001) | 0.9753 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0286 | (-0.1035, 0.0359) | 0.7770 | -0.0286 | (-0.1286, 0.0559) | 0.7197 |
| controlled_alt_vs_controlled_default | naturalness | 0.0291 | (-0.0089, 0.0641) | 0.0610 | 0.0291 | (0.0007, 0.0709) | 0.0203 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0612 | (-0.1117, -0.0101) | 0.9900 | -0.0612 | (-0.1364, 0.0000) | 0.9753 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0046 | (-0.0230, 0.0098) | 0.6890 | -0.0046 | (-0.0163, 0.0111) | 0.7583 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0369 | (-0.1248, 0.0397) | 0.7893 | -0.0369 | (-0.1615, 0.0624) | 0.7430 |
| controlled_alt_vs_controlled_default | persona_style | 0.0044 | (-0.0318, 0.0414) | 0.4263 | 0.0044 | (-0.0439, 0.0478) | 0.4273 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0082 | (-0.0122, 0.0284) | 0.2080 | 0.0082 | (-0.0074, 0.0254) | 0.1547 |
| controlled_alt_vs_controlled_default | length_score | 0.1292 | (-0.0403, 0.2931) | 0.0703 | 0.1292 | (0.0091, 0.3121) | 0.0143 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0000 | (-0.0875, 0.0875) | 0.5840 | 0.0000 | (-0.0477, 0.0375) | 0.6450 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0258 | (-0.0559, 0.0001) | 0.9743 | -0.0258 | (-0.0666, 0.0116) | 0.9017 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1741 | (0.1310, 0.2115) | 0.0000 | 0.1741 | (0.1287, 0.2221) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1900 | (0.1208, 0.2571) | 0.0000 | 0.1900 | (0.1378, 0.2595) | 0.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0659 | (0.0190, 0.1093) | 0.0033 | 0.0659 | (0.0121, 0.1345) | 0.0070 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2258 | (0.1717, 0.2761) | 0.0000 | 0.2258 | (0.1682, 0.2874) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0533 | (0.0385, 0.0680) | 0.0000 | 0.0533 | (0.0362, 0.0754) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.2139 | (0.1339, 0.2907) | 0.0000 | 0.2139 | (0.1583, 0.2937) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0943 | (0.0124, 0.1927) | 0.0110 | 0.0943 | (0.0000, 0.2328) | 0.0267 |
| controlled_alt_vs_proposed_raw | distinct1 | 0.0066 | (-0.0111, 0.0234) | 0.2443 | 0.0066 | (-0.0065, 0.0244) | 0.1800 |
| controlled_alt_vs_proposed_raw | length_score | 0.2431 | (0.0597, 0.4167) | 0.0060 | 0.2431 | (0.0308, 0.5128) | 0.0150 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1458 | (0.0437, 0.2333) | 0.0023 | 0.1458 | (0.0219, 0.2705) | 0.0117 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1637 | (0.1304, 0.1991) | 0.0000 | 0.1637 | (0.1292, 0.2111) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2236 | (0.1968, 0.2502) | 0.0000 | 0.2236 | (0.1937, 0.2457) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1518 | (0.1048, 0.1986) | 0.0000 | 0.1518 | (0.1081, 0.2160) | 0.0000 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0582 | (0.0114, 0.1024) | 0.0087 | 0.0582 | (0.0158, 0.1057) | 0.0040 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.2892 | (0.2533, 0.3242) | 0.0000 | 0.2892 | (0.2540, 0.3182) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0705 | (0.0598, 0.0803) | 0.0000 | 0.0705 | (0.0561, 0.0831) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1752 | (0.1192, 0.2331) | 0.0000 | 0.1752 | (0.1260, 0.2490) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0580 | (0.0010, 0.1328) | 0.0237 | 0.0580 | (0.0072, 0.1473) | 0.0247 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0074 | (-0.0122, 0.0279) | 0.2320 | 0.0074 | (-0.0051, 0.0221) | 0.1227 |
| controlled_alt_vs_candidate_no_context | length_score | 0.2250 | (0.0542, 0.3945) | 0.0033 | 0.2250 | (0.0536, 0.3940) | 0.0057 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1021 | (0.0000, 0.2042) | 0.0407 | 0.1021 | (-0.0159, 0.2154) | 0.0503 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1713 | (0.1439, 0.1971) | 0.0000 | 0.1713 | (0.1450, 0.2023) | 0.0000 |
| controlled_alt_vs_baseline_no_context | context_relevance | 0.2042 | (0.1791, 0.2311) | 0.0000 | 0.2042 | (0.1747, 0.2290) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 0.1606 | (0.1154, 0.2010) | 0.0000 | 0.1606 | (0.0899, 0.2088) | 0.0000 |
| controlled_alt_vs_baseline_no_context | naturalness | -0.0138 | (-0.0460, 0.0173) | 0.8123 | -0.0138 | (-0.0510, 0.0283) | 0.7337 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 0.2624 | (0.2278, 0.2970) | 0.0000 | 0.2624 | (0.2240, 0.2965) | 0.0000 |
| controlled_alt_vs_baseline_no_context | context_overlap | 0.0683 | (0.0558, 0.0812) | 0.0000 | 0.0683 | (0.0528, 0.0834) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 0.1956 | (0.1407, 0.2464) | 0.0000 | 0.1956 | (0.1132, 0.2558) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_style | 0.0206 | (-0.0174, 0.0652) | 0.1673 | 0.0206 | (0.0025, 0.0516) | 0.0230 |
| controlled_alt_vs_baseline_no_context | distinct1 | -0.0392 | (-0.0505, -0.0283) | 1.0000 | -0.0392 | (-0.0486, -0.0276) | 1.0000 |
| controlled_alt_vs_baseline_no_context | length_score | -0.0125 | (-0.1528, 0.1250) | 0.5680 | -0.0125 | (-0.1987, 0.1810) | 0.5443 |
| controlled_alt_vs_baseline_no_context | sentence_score | 0.0437 | (-0.0437, 0.1313) | 0.2000 | 0.0437 | (-0.0167, 0.1225) | 0.1443 |
| controlled_alt_vs_baseline_no_context | overall_quality | 0.1516 | (0.1275, 0.1730) | 0.0000 | 0.1516 | (0.1176, 0.1784) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2484 | (0.1992, 0.2990) | 0.0000 | 0.2484 | (0.1853, 0.3124) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1893 | (0.1268, 0.2521) | 0.0000 | 0.1893 | (0.1107, 0.2632) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0429 | (-0.0814, -0.0040) | 0.9840 | -0.0429 | (-0.0897, 0.0005) | 0.9730 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3236 | (0.2582, 0.3895) | 0.0000 | 0.3236 | (0.2364, 0.4121) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0729 | (0.0533, 0.0997) | 0.0000 | 0.0729 | (0.0524, 0.0927) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2325 | (0.1595, 0.3068) | 0.0000 | 0.2325 | (0.1322, 0.3214) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0161 | (-0.0229, 0.0656) | 0.2610 | 0.0161 | (-0.0300, 0.0739) | 0.3010 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0474 | (-0.0623, -0.0315) | 1.0000 | -0.0474 | (-0.0622, -0.0328) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | -0.1417 | (-0.3139, 0.0348) | 0.9427 | -0.1417 | (-0.3637, 0.0387) | 0.9250 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0437 | (-0.0437, 0.1313) | 0.2037 | 0.0437 | (-0.0467, 0.1400) | 0.2110 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1775 | (0.1489, 0.2058) | 0.0000 | 0.1775 | (0.1356, 0.2113) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 11 | 5 | 8 | 0.6250 | 0.6875 |
| proposed_vs_candidate_no_context | persona_consistency | 4 | 8 | 12 | 0.4167 | 0.3333 |
| proposed_vs_candidate_no_context | naturalness | 8 | 8 | 8 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 9 | 3 | 12 | 0.6250 | 0.7500 |
| proposed_vs_candidate_no_context | context_overlap | 12 | 4 | 8 | 0.6667 | 0.7500 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 3 | 6 | 15 | 0.4375 | 0.3333 |
| proposed_vs_candidate_no_context | persona_style | 2 | 3 | 19 | 0.4792 | 0.4000 |
| proposed_vs_candidate_no_context | distinct1 | 8 | 8 | 8 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | length_score | 8 | 8 | 8 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | sentence_score | 2 | 5 | 17 | 0.4375 | 0.2857 |
| proposed_vs_candidate_no_context | overall_quality | 8 | 8 | 8 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context | context_relevance | 12 | 12 | 0 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context | persona_consistency | 5 | 9 | 10 | 0.4167 | 0.3571 |
| proposed_vs_baseline_no_context | naturalness | 6 | 18 | 0 | 0.2500 | 0.2500 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 6 | 7 | 11 | 0.4792 | 0.4615 |
| proposed_vs_baseline_no_context | context_overlap | 14 | 9 | 1 | 0.6042 | 0.6087 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 3 | 5 | 16 | 0.4583 | 0.3750 |
| proposed_vs_baseline_no_context | persona_style | 2 | 6 | 16 | 0.4167 | 0.2500 |
| proposed_vs_baseline_no_context | distinct1 | 2 | 19 | 3 | 0.1458 | 0.0952 |
| proposed_vs_baseline_no_context | length_score | 5 | 19 | 0 | 0.2083 | 0.2083 |
| proposed_vs_baseline_no_context | sentence_score | 4 | 11 | 9 | 0.3542 | 0.2667 |
| proposed_vs_baseline_no_context | overall_quality | 8 | 16 | 0 | 0.3333 | 0.3333 |
| controlled_vs_proposed_raw | context_relevance | 22 | 2 | 0 | 0.9167 | 0.9167 |
| controlled_vs_proposed_raw | persona_consistency | 21 | 3 | 0 | 0.8750 | 0.8750 |
| controlled_vs_proposed_raw | naturalness | 15 | 9 | 0 | 0.6250 | 0.6250 |
| controlled_vs_proposed_raw | context_keyword_coverage | 22 | 2 | 0 | 0.9167 | 0.9167 |
| controlled_vs_proposed_raw | context_overlap | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 21 | 2 | 1 | 0.8958 | 0.9130 |
| controlled_vs_proposed_raw | persona_style | 6 | 4 | 14 | 0.5417 | 0.6000 |
| controlled_vs_proposed_raw | distinct1 | 11 | 10 | 3 | 0.5208 | 0.5238 |
| controlled_vs_proposed_raw | length_score | 13 | 11 | 0 | 0.5417 | 0.5417 |
| controlled_vs_proposed_raw | sentence_score | 14 | 4 | 6 | 0.7083 | 0.7778 |
| controlled_vs_proposed_raw | overall_quality | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_vs_candidate_no_context | context_relevance | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 19 | 3 | 2 | 0.8333 | 0.8636 |
| controlled_vs_candidate_no_context | naturalness | 18 | 6 | 0 | 0.7500 | 0.7500 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 18 | 2 | 4 | 0.8333 | 0.9000 |
| controlled_vs_candidate_no_context | persona_style | 4 | 3 | 17 | 0.5208 | 0.5714 |
| controlled_vs_candidate_no_context | distinct1 | 11 | 12 | 1 | 0.4792 | 0.4783 |
| controlled_vs_candidate_no_context | length_score | 13 | 8 | 3 | 0.6042 | 0.6190 |
| controlled_vs_candidate_no_context | sentence_score | 10 | 3 | 11 | 0.6458 | 0.7692 |
| controlled_vs_candidate_no_context | overall_quality | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | context_relevance | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 21 | 1 | 2 | 0.9167 | 0.9545 |
| controlled_vs_baseline_no_context | naturalness | 7 | 16 | 1 | 0.3125 | 0.3043 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | context_overlap | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 21 | 1 | 2 | 0.9167 | 0.9545 |
| controlled_vs_baseline_no_context | persona_style | 2 | 2 | 20 | 0.5000 | 0.5000 |
| controlled_vs_baseline_no_context | distinct1 | 2 | 19 | 3 | 0.1458 | 0.0952 |
| controlled_vs_baseline_no_context | length_score | 9 | 14 | 1 | 0.3958 | 0.3913 |
| controlled_vs_baseline_no_context | sentence_score | 6 | 3 | 15 | 0.5625 | 0.6667 |
| controlled_vs_baseline_no_context | overall_quality | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_alt_vs_controlled_default | context_relevance | 7 | 13 | 4 | 0.3750 | 0.3500 |
| controlled_alt_vs_controlled_default | persona_consistency | 8 | 6 | 10 | 0.5417 | 0.5714 |
| controlled_alt_vs_controlled_default | naturalness | 13 | 7 | 4 | 0.6250 | 0.6500 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 5 | 13 | 6 | 0.3333 | 0.2778 |
| controlled_alt_vs_controlled_default | context_overlap | 10 | 10 | 4 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 7 | 5 | 12 | 0.5417 | 0.5833 |
| controlled_alt_vs_controlled_default | persona_style | 4 | 3 | 17 | 0.5208 | 0.5714 |
| controlled_alt_vs_controlled_default | distinct1 | 12 | 8 | 4 | 0.5833 | 0.6000 |
| controlled_alt_vs_controlled_default | length_score | 12 | 8 | 4 | 0.5833 | 0.6000 |
| controlled_alt_vs_controlled_default | sentence_score | 4 | 4 | 16 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | overall_quality | 10 | 10 | 4 | 0.5000 | 0.5000 |
| controlled_alt_vs_proposed_raw | context_relevance | 22 | 2 | 0 | 0.9167 | 0.9167 |
| controlled_alt_vs_proposed_raw | persona_consistency | 21 | 2 | 1 | 0.8958 | 0.9130 |
| controlled_alt_vs_proposed_raw | naturalness | 17 | 7 | 0 | 0.7083 | 0.7083 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 21 | 2 | 1 | 0.8958 | 0.9130 |
| controlled_alt_vs_proposed_raw | context_overlap | 22 | 2 | 0 | 0.9167 | 0.9167 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 21 | 2 | 1 | 0.8958 | 0.9130 |
| controlled_alt_vs_proposed_raw | persona_style | 6 | 1 | 17 | 0.6042 | 0.8571 |
| controlled_alt_vs_proposed_raw | distinct1 | 12 | 10 | 2 | 0.5417 | 0.5455 |
| controlled_alt_vs_proposed_raw | length_score | 16 | 7 | 1 | 0.6875 | 0.6957 |
| controlled_alt_vs_proposed_raw | sentence_score | 13 | 3 | 8 | 0.7083 | 0.8125 |
| controlled_alt_vs_proposed_raw | overall_quality | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 20 | 1 | 3 | 0.8958 | 0.9524 |
| controlled_alt_vs_candidate_no_context | naturalness | 16 | 8 | 0 | 0.6667 | 0.6667 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 19 | 1 | 4 | 0.8750 | 0.9500 |
| controlled_alt_vs_candidate_no_context | persona_style | 4 | 1 | 19 | 0.5625 | 0.8000 |
| controlled_alt_vs_candidate_no_context | distinct1 | 12 | 11 | 1 | 0.5208 | 0.5217 |
| controlled_alt_vs_candidate_no_context | length_score | 15 | 8 | 1 | 0.6458 | 0.6522 |
| controlled_alt_vs_candidate_no_context | sentence_score | 11 | 4 | 9 | 0.6458 | 0.7333 |
| controlled_alt_vs_candidate_no_context | overall_quality | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context | context_relevance | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 20 | 1 | 3 | 0.8958 | 0.9524 |
| controlled_alt_vs_baseline_no_context | naturalness | 11 | 12 | 1 | 0.4792 | 0.4783 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context | context_overlap | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 20 | 1 | 3 | 0.8958 | 0.9524 |
| controlled_alt_vs_baseline_no_context | persona_style | 3 | 2 | 19 | 0.5208 | 0.6000 |
| controlled_alt_vs_baseline_no_context | distinct1 | 2 | 19 | 3 | 0.1458 | 0.0952 |
| controlled_alt_vs_baseline_no_context | length_score | 13 | 10 | 1 | 0.5625 | 0.5652 |
| controlled_alt_vs_baseline_no_context | sentence_score | 6 | 3 | 15 | 0.5625 | 0.6667 |
| controlled_alt_vs_baseline_no_context | overall_quality | 23 | 1 | 0 | 0.9583 | 0.9583 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 24 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 21 | 1 | 2 | 0.9167 | 0.9545 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 7 | 16 | 1 | 0.3125 | 0.3043 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 24 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 24 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 21 | 1 | 2 | 0.9167 | 0.9545 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 2 | 2 | 20 | 0.5000 | 0.5000 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | 2 | 19 | 3 | 0.1458 | 0.0952 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 9 | 14 | 1 | 0.3958 | 0.3913 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 6 | 3 | 15 | 0.5625 | 0.6667 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 23 | 1 | 0 | 0.9583 | 0.9583 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.2083 | 0.4167 | 0.5833 |
| proposed_contextual_controlled_alt | 0.0000 | 0.0000 | 0.2083 | 0.5833 | 0.4167 |
| proposed_contextual | 0.0000 | 0.0000 | 0.4583 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.4583 | 0.0000 | 0.0000 |
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
# Proposal Alignment Evaluation Report

- Run ID: `20260306T165401Z`
- Generated: `2026-03-06T16:58:17.551479+00:00`
- Scenarios: `artifacts\proposal_control_tuning\architecture_compare_lock\20260306T164947Z\blend_balanced\seed_31\attempt_1\20260306T165401Z\scenarios.jsonl`
- Scenario count: `32`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_alt`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2798 (0.2575, 0.3017) | 0.2790 (0.2494, 0.3096) | 0.9209 (0.9065, 0.9337) | 0.4090 (0.3937, 0.4238) | n/a |
| proposed_contextual_controlled_alt | 0.2431 (0.2176, 0.2697) | 0.2971 (0.2593, 0.3352) | 0.9169 (0.9029, 0.9306) | 0.3976 (0.3805, 0.4126) | n/a |
| proposed_contextual | 0.0638 (0.0349, 0.0966) | 0.1281 (0.0937, 0.1628) | 0.8186 (0.7923, 0.8471) | 0.2326 (0.2081, 0.2589) | n/a |
| candidate_no_context | 0.0299 (0.0172, 0.0445) | 0.1630 (0.1188, 0.2150) | 0.8213 (0.7948, 0.8487) | 0.2303 (0.2087, 0.2538) | n/a |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0339 | 1.1308 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0349 | -0.2141 |
| proposed_vs_candidate_no_context | naturalness | -0.0027 | -0.0033 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0455 | 1.7778 |
| proposed_vs_candidate_no_context | context_overlap | 0.0068 | 0.1691 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0417 | -0.5833 |
| proposed_vs_candidate_no_context | persona_style | -0.0078 | -0.0148 |
| proposed_vs_candidate_no_context | distinct1 | 0.0019 | 0.0020 |
| proposed_vs_candidate_no_context | length_score | -0.0010 | -0.0031 |
| proposed_vs_candidate_no_context | sentence_score | -0.0328 | -0.0432 |
| proposed_vs_candidate_no_context | overall_quality | 0.0023 | 0.0100 |
| controlled_vs_proposed_raw | context_relevance | 0.2160 | 3.3855 |
| controlled_vs_proposed_raw | persona_consistency | 0.1509 | 1.1776 |
| controlled_vs_proposed_raw | naturalness | 0.1023 | 0.1250 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2812 | 3.9587 |
| controlled_vs_proposed_raw | context_overlap | 0.0639 | 1.3614 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1744 | 5.8600 |
| controlled_vs_proposed_raw | persona_style | 0.0566 | 0.1086 |
| controlled_vs_proposed_raw | distinct1 | -0.0090 | -0.0095 |
| controlled_vs_proposed_raw | length_score | 0.4313 | 1.2778 |
| controlled_vs_proposed_raw | sentence_score | 0.1969 | 0.2710 |
| controlled_vs_proposed_raw | overall_quality | 0.1764 | 0.7586 |
| controlled_vs_candidate_no_context | context_relevance | 0.2498 | 8.3448 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1160 | 0.7114 |
| controlled_vs_candidate_no_context | naturalness | 0.0996 | 0.1213 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3266 | 12.7741 |
| controlled_vs_candidate_no_context | context_overlap | 0.0707 | 1.7607 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1327 | 1.8583 |
| controlled_vs_candidate_no_context | persona_style | 0.0488 | 0.0923 |
| controlled_vs_candidate_no_context | distinct1 | -0.0071 | -0.0075 |
| controlled_vs_candidate_no_context | length_score | 0.4302 | 1.2708 |
| controlled_vs_candidate_no_context | sentence_score | 0.1641 | 0.2160 |
| controlled_vs_candidate_no_context | overall_quality | 0.1787 | 0.7762 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0367 | -0.1312 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0181 | 0.0650 |
| controlled_alt_vs_controlled_default | naturalness | -0.0040 | -0.0044 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0471 | -0.1338 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0124 | -0.1119 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0161 | 0.0787 |
| controlled_alt_vs_controlled_default | persona_style | 0.0264 | 0.0457 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0057 | 0.0061 |
| controlled_alt_vs_controlled_default | length_score | -0.0260 | -0.0339 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0109 | -0.0118 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0114 | -0.0278 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1793 | 2.8103 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1690 | 1.3192 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0983 | 0.1201 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2340 | 3.2953 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0515 | 1.0971 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1905 | 6.4000 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0831 | 0.1593 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0033 | -0.0034 |
| controlled_alt_vs_proposed_raw | length_score | 0.4052 | 1.2006 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1859 | 0.2559 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1651 | 0.7097 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2131 | 7.1189 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1341 | 0.8227 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0956 | 0.1164 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.2795 | 10.9315 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0583 | 1.4517 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1488 | 2.0833 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0752 | 0.1422 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0014 | -0.0015 |
| controlled_alt_vs_candidate_no_context | length_score | 0.4042 | 1.1938 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1531 | 0.2016 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1674 | 0.7268 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0339 | (0.0060, 0.0662) | 0.0050 | 0.0339 | (0.0001, 0.0711) | 0.0240 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0349 | (-0.0939, 0.0116) | 0.9083 | -0.0349 | (-0.1089, 0.0060) | 0.9110 |
| proposed_vs_candidate_no_context | naturalness | -0.0027 | (-0.0269, 0.0215) | 0.5827 | -0.0027 | (-0.0344, 0.0195) | 0.5497 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0455 | (0.0057, 0.0881) | 0.0127 | 0.0455 | (0.0000, 0.0939) | 0.0300 |
| proposed_vs_candidate_no_context | context_overlap | 0.0068 | (-0.0028, 0.0176) | 0.0893 | 0.0068 | (-0.0021, 0.0151) | 0.0743 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0417 | (-0.1109, 0.0179) | 0.9117 | -0.0417 | (-0.1284, 0.0067) | 0.9327 |
| proposed_vs_candidate_no_context | persona_style | -0.0078 | (-0.0417, 0.0234) | 0.6497 | -0.0078 | (-0.0444, 0.0242) | 0.6550 |
| proposed_vs_candidate_no_context | distinct1 | 0.0019 | (-0.0088, 0.0129) | 0.3637 | 0.0019 | (-0.0133, 0.0148) | 0.3983 |
| proposed_vs_candidate_no_context | length_score | -0.0010 | (-0.0906, 0.0906) | 0.5297 | -0.0010 | (-0.1111, 0.0871) | 0.4930 |
| proposed_vs_candidate_no_context | sentence_score | -0.0328 | (-0.0984, 0.0328) | 0.8740 | -0.0328 | (-0.0984, 0.0135) | 0.9313 |
| proposed_vs_candidate_no_context | overall_quality | 0.0023 | (-0.0272, 0.0297) | 0.4363 | 0.0023 | (-0.0375, 0.0286) | 0.4310 |
| controlled_vs_proposed_raw | context_relevance | 0.2160 | (0.1847, 0.2482) | 0.0000 | 0.2160 | (0.1919, 0.2495) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.1509 | (0.1203, 0.1799) | 0.0000 | 0.1509 | (0.1169, 0.1860) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.1023 | (0.0677, 0.1346) | 0.0000 | 0.1023 | (0.0408, 0.1622) | 0.0007 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2812 | (0.2396, 0.3230) | 0.0000 | 0.2812 | (0.2524, 0.3242) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0639 | (0.0516, 0.0759) | 0.0000 | 0.0639 | (0.0445, 0.0822) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1744 | (0.1363, 0.2079) | 0.0000 | 0.1744 | (0.1321, 0.2177) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.0566 | (0.0052, 0.1182) | 0.0167 | 0.0566 | (0.0000, 0.1587) | 0.1000 |
| controlled_vs_proposed_raw | distinct1 | -0.0090 | (-0.0287, 0.0097) | 0.8167 | -0.0090 | (-0.0369, 0.0214) | 0.6933 |
| controlled_vs_proposed_raw | length_score | 0.4313 | (0.3031, 0.5532) | 0.0000 | 0.4313 | (0.2131, 0.6323) | 0.0000 |
| controlled_vs_proposed_raw | sentence_score | 0.1969 | (0.1094, 0.2734) | 0.0000 | 0.1969 | (0.0800, 0.3033) | 0.0003 |
| controlled_vs_proposed_raw | overall_quality | 0.1764 | (0.1529, 0.1995) | 0.0000 | 0.1764 | (0.1592, 0.1985) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2498 | (0.2261, 0.2748) | 0.0000 | 0.2498 | (0.2185, 0.2816) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1160 | (0.0586, 0.1635) | 0.0000 | 0.1160 | (0.0577, 0.1655) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.0996 | (0.0665, 0.1315) | 0.0000 | 0.0996 | (0.0432, 0.1541) | 0.0000 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3266 | (0.2964, 0.3570) | 0.0000 | 0.3266 | (0.2867, 0.3717) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0707 | (0.0562, 0.0860) | 0.0000 | 0.0707 | (0.0510, 0.0863) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1327 | (0.0704, 0.1865) | 0.0000 | 0.1327 | (0.0622, 0.1971) | 0.0003 |
| controlled_vs_candidate_no_context | persona_style | 0.0488 | (-0.0029, 0.1156) | 0.0373 | 0.0488 | (-0.0130, 0.1551) | 0.0953 |
| controlled_vs_candidate_no_context | distinct1 | -0.0071 | (-0.0232, 0.0091) | 0.7997 | -0.0071 | (-0.0304, 0.0160) | 0.7130 |
| controlled_vs_candidate_no_context | length_score | 0.4302 | (0.3083, 0.5417) | 0.0000 | 0.4302 | (0.2437, 0.6000) | 0.0000 |
| controlled_vs_candidate_no_context | sentence_score | 0.1641 | (0.0875, 0.2297) | 0.0000 | 0.1641 | (0.0538, 0.2758) | 0.0013 |
| controlled_vs_candidate_no_context | overall_quality | 0.1787 | (0.1531, 0.2009) | 0.0000 | 0.1787 | (0.1524, 0.1969) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0367 | (-0.0649, -0.0096) | 0.9967 | -0.0367 | (-0.0751, -0.0018) | 0.9810 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0181 | (-0.0096, 0.0493) | 0.1067 | 0.0181 | (-0.0128, 0.0753) | 0.2757 |
| controlled_alt_vs_controlled_default | naturalness | -0.0040 | (-0.0155, 0.0082) | 0.7533 | -0.0040 | (-0.0152, 0.0066) | 0.7617 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0471 | (-0.0800, -0.0135) | 0.9957 | -0.0471 | (-0.0939, -0.0027) | 0.9830 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0124 | (-0.0254, -0.0002) | 0.9763 | -0.0124 | (-0.0263, 0.0003) | 0.9723 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0161 | (-0.0140, 0.0491) | 0.1697 | 0.0161 | (-0.0158, 0.0739) | 0.3523 |
| controlled_alt_vs_controlled_default | persona_style | 0.0264 | (-0.0022, 0.0662) | 0.0520 | 0.0264 | (-0.0042, 0.0838) | 0.1047 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0057 | (-0.0054, 0.0177) | 0.1610 | 0.0057 | (-0.0060, 0.0195) | 0.1773 |
| controlled_alt_vs_controlled_default | length_score | -0.0260 | (-0.0760, 0.0240) | 0.8483 | -0.0260 | (-0.0667, 0.0074) | 0.9363 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0109 | (-0.0766, 0.0547) | 0.6940 | -0.0109 | (-0.0790, 0.0389) | 0.7027 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0114 | (-0.0236, 0.0010) | 0.9643 | -0.0114 | (-0.0229, 0.0009) | 0.9597 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1793 | (0.1455, 0.2100) | 0.0000 | 0.1793 | (0.1443, 0.2150) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1690 | (0.1225, 0.2168) | 0.0000 | 0.1690 | (0.1097, 0.2576) | 0.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0983 | (0.0644, 0.1298) | 0.0000 | 0.0983 | (0.0426, 0.1534) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2340 | (0.1892, 0.2763) | 0.0000 | 0.2340 | (0.1882, 0.2817) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0515 | (0.0392, 0.0640) | 0.0000 | 0.0515 | (0.0367, 0.0704) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1905 | (0.1362, 0.2461) | 0.0000 | 0.1905 | (0.1216, 0.2873) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0831 | (0.0208, 0.1553) | 0.0010 | 0.0831 | (0.0133, 0.1995) | 0.0233 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0033 | (-0.0192, 0.0127) | 0.6470 | -0.0033 | (-0.0272, 0.0252) | 0.5650 |
| controlled_alt_vs_proposed_raw | length_score | 0.4052 | (0.2812, 0.5271) | 0.0000 | 0.4052 | (0.1898, 0.6043) | 0.0003 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1859 | (0.1094, 0.2625) | 0.0000 | 0.1859 | (0.0700, 0.2800) | 0.0007 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1651 | (0.1379, 0.1890) | 0.0000 | 0.1651 | (0.1418, 0.1948) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2131 | (0.1864, 0.2370) | 0.0000 | 0.2131 | (0.1741, 0.2459) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1341 | (0.0853, 0.1848) | 0.0000 | 0.1341 | (0.0919, 0.1715) | 0.0000 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0956 | (0.0610, 0.1291) | 0.0000 | 0.0956 | (0.0412, 0.1462) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.2795 | (0.2440, 0.3113) | 0.0000 | 0.2795 | (0.2278, 0.3239) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0583 | (0.0471, 0.0696) | 0.0000 | 0.0583 | (0.0429, 0.0739) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1488 | (0.0921, 0.2005) | 0.0000 | 0.1488 | (0.0975, 0.1982) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0752 | (0.0134, 0.1501) | 0.0063 | 0.0752 | (0.0142, 0.1797) | 0.0037 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0014 | (-0.0161, 0.0124) | 0.5637 | -0.0014 | (-0.0161, 0.0153) | 0.5640 |
| controlled_alt_vs_candidate_no_context | length_score | 0.4042 | (0.2677, 0.5292) | 0.0000 | 0.4042 | (0.1979, 0.5905) | 0.0000 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1531 | (0.0656, 0.2300) | 0.0003 | 0.1531 | (0.0350, 0.2683) | 0.0057 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1674 | (0.1457, 0.1871) | 0.0000 | 0.1674 | (0.1457, 0.1882) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 9 | 9 | 14 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | persona_consistency | 4 | 6 | 22 | 0.4688 | 0.4000 |
| proposed_vs_candidate_no_context | naturalness | 10 | 8 | 14 | 0.5312 | 0.5556 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 6 | 2 | 24 | 0.5625 | 0.7500 |
| proposed_vs_candidate_no_context | context_overlap | 10 | 8 | 14 | 0.5312 | 0.5556 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 3 | 5 | 24 | 0.4688 | 0.3750 |
| proposed_vs_candidate_no_context | persona_style | 1 | 2 | 29 | 0.4844 | 0.3333 |
| proposed_vs_candidate_no_context | distinct1 | 6 | 6 | 20 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | length_score | 8 | 9 | 15 | 0.4844 | 0.4706 |
| proposed_vs_candidate_no_context | sentence_score | 3 | 6 | 23 | 0.4531 | 0.3333 |
| proposed_vs_candidate_no_context | overall_quality | 9 | 9 | 14 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | context_relevance | 32 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_proposed_raw | persona_consistency | 28 | 1 | 3 | 0.9219 | 0.9655 |
| controlled_vs_proposed_raw | naturalness | 25 | 7 | 0 | 0.7812 | 0.7812 |
| controlled_vs_proposed_raw | context_keyword_coverage | 31 | 0 | 1 | 0.9844 | 1.0000 |
| controlled_vs_proposed_raw | context_overlap | 31 | 1 | 0 | 0.9688 | 0.9688 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 28 | 1 | 3 | 0.9219 | 0.9655 |
| controlled_vs_proposed_raw | persona_style | 4 | 0 | 28 | 0.5625 | 1.0000 |
| controlled_vs_proposed_raw | distinct1 | 18 | 13 | 1 | 0.5781 | 0.5806 |
| controlled_vs_proposed_raw | length_score | 28 | 4 | 0 | 0.8750 | 0.8750 |
| controlled_vs_proposed_raw | sentence_score | 21 | 3 | 8 | 0.7812 | 0.8750 |
| controlled_vs_proposed_raw | overall_quality | 31 | 1 | 0 | 0.9688 | 0.9688 |
| controlled_vs_candidate_no_context | context_relevance | 32 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 25 | 4 | 3 | 0.8281 | 0.8621 |
| controlled_vs_candidate_no_context | naturalness | 26 | 6 | 0 | 0.8125 | 0.8125 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 32 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 31 | 1 | 0 | 0.9688 | 0.9688 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 25 | 4 | 3 | 0.8281 | 0.8621 |
| controlled_vs_candidate_no_context | persona_style | 4 | 1 | 27 | 0.5469 | 0.8000 |
| controlled_vs_candidate_no_context | distinct1 | 17 | 13 | 2 | 0.5625 | 0.5667 |
| controlled_vs_candidate_no_context | length_score | 28 | 2 | 2 | 0.9062 | 0.9333 |
| controlled_vs_candidate_no_context | sentence_score | 17 | 2 | 13 | 0.7344 | 0.8947 |
| controlled_vs_candidate_no_context | overall_quality | 31 | 1 | 0 | 0.9688 | 0.9688 |
| controlled_alt_vs_controlled_default | context_relevance | 4 | 14 | 14 | 0.3438 | 0.2222 |
| controlled_alt_vs_controlled_default | persona_consistency | 5 | 3 | 24 | 0.5312 | 0.6250 |
| controlled_alt_vs_controlled_default | naturalness | 7 | 11 | 14 | 0.4375 | 0.3889 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 3 | 13 | 16 | 0.3438 | 0.1875 |
| controlled_alt_vs_controlled_default | context_overlap | 7 | 11 | 14 | 0.4375 | 0.3889 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 4 | 2 | 26 | 0.5312 | 0.6667 |
| controlled_alt_vs_controlled_default | persona_style | 3 | 1 | 28 | 0.5312 | 0.7500 |
| controlled_alt_vs_controlled_default | distinct1 | 10 | 7 | 15 | 0.5469 | 0.5882 |
| controlled_alt_vs_controlled_default | length_score | 6 | 11 | 15 | 0.4219 | 0.3529 |
| controlled_alt_vs_controlled_default | sentence_score | 4 | 5 | 23 | 0.4844 | 0.4444 |
| controlled_alt_vs_controlled_default | overall_quality | 7 | 11 | 14 | 0.4375 | 0.3889 |
| controlled_alt_vs_proposed_raw | context_relevance | 31 | 1 | 0 | 0.9688 | 0.9688 |
| controlled_alt_vs_proposed_raw | persona_consistency | 26 | 1 | 5 | 0.8906 | 0.9630 |
| controlled_alt_vs_proposed_raw | naturalness | 24 | 7 | 1 | 0.7656 | 0.7742 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 29 | 1 | 2 | 0.9375 | 0.9667 |
| controlled_alt_vs_proposed_raw | context_overlap | 31 | 1 | 0 | 0.9688 | 0.9688 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 26 | 1 | 5 | 0.8906 | 0.9630 |
| controlled_alt_vs_proposed_raw | persona_style | 6 | 1 | 25 | 0.5781 | 0.8571 |
| controlled_alt_vs_proposed_raw | distinct1 | 17 | 13 | 2 | 0.5625 | 0.5667 |
| controlled_alt_vs_proposed_raw | length_score | 26 | 4 | 2 | 0.8438 | 0.8667 |
| controlled_alt_vs_proposed_raw | sentence_score | 19 | 2 | 11 | 0.7656 | 0.9048 |
| controlled_alt_vs_proposed_raw | overall_quality | 31 | 1 | 0 | 0.9688 | 0.9688 |
| controlled_alt_vs_candidate_no_context | context_relevance | 32 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 25 | 4 | 3 | 0.8281 | 0.8621 |
| controlled_alt_vs_candidate_no_context | naturalness | 25 | 7 | 0 | 0.7812 | 0.7812 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 31 | 0 | 1 | 0.9844 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 31 | 1 | 0 | 0.9688 | 0.9688 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 25 | 3 | 4 | 0.8438 | 0.8929 |
| controlled_alt_vs_candidate_no_context | persona_style | 7 | 2 | 23 | 0.5781 | 0.7778 |
| controlled_alt_vs_candidate_no_context | distinct1 | 20 | 10 | 2 | 0.6562 | 0.6667 |
| controlled_alt_vs_candidate_no_context | length_score | 29 | 3 | 0 | 0.9062 | 0.9062 |
| controlled_alt_vs_candidate_no_context | sentence_score | 18 | 4 | 10 | 0.7188 | 0.8182 |
| controlled_alt_vs_candidate_no_context | overall_quality | 32 | 0 | 0 | 1.0000 | 1.0000 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.0000 | 0.4375 | 0.0938 |
| proposed_contextual_controlled_alt | 0.0000 | 0.0000 | 0.0000 | 0.4688 | 0.0938 |
| proposed_contextual | 0.0000 | 0.0000 | 0.5312 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.4688 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `8`
- Unique template signatures: `30`
- Template signature ratio: `0.9375`
- Effective sample size by source clustering: `7.42`
- Effective sample size by template-signature clustering: `28.44`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: unavailable (disabled_by_flag).

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
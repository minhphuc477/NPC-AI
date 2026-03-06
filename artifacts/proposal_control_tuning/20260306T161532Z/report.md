# Proposal Alignment Evaluation Report

- Run ID: `20260306T161532Z`
- Generated: `2026-03-06T16:20:35.355981+00:00`
- Scenarios: `artifacts\proposal_control_tuning\20260306T161532Z\scenarios.jsonl`
- Scenario count: `24`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_alt`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2629 (0.2412, 0.2872) | 0.3706 (0.3090, 0.4364) | 0.8768 (0.8452, 0.9038) | 0.4254 (0.4008, 0.4510) | n/a |
| proposed_contextual_controlled_alt | 0.2472 (0.2165, 0.2805) | 0.3428 (0.2942, 0.3927) | 0.9027 (0.8773, 0.9261) | 0.4126 (0.3954, 0.4288) | n/a |
| proposed_contextual | 0.1176 (0.0662, 0.1735) | 0.2424 (0.1864, 0.2970) | 0.8235 (0.7948, 0.8540) | 0.2998 (0.2576, 0.3431) | n/a |
| candidate_no_context | 0.0305 (0.0176, 0.0463) | 0.1706 (0.1210, 0.2301) | 0.8133 (0.7849, 0.8452) | 0.2308 (0.2079, 0.2595) | n/a |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0871 | 2.8521 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0718 | 0.4208 |
| proposed_vs_candidate_no_context | naturalness | 0.0102 | 0.0125 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.1170 | 4.6350 |
| proposed_vs_candidate_no_context | context_overlap | 0.0172 | 0.4019 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0784 | 1.1970 |
| proposed_vs_candidate_no_context | persona_style | 0.0454 | 0.0769 |
| proposed_vs_candidate_no_context | distinct1 | 0.0024 | 0.0025 |
| proposed_vs_candidate_no_context | length_score | 0.0097 | 0.0326 |
| proposed_vs_candidate_no_context | sentence_score | 0.0729 | 0.0933 |
| proposed_vs_candidate_no_context | overall_quality | 0.0690 | 0.2990 |
| controlled_vs_proposed_raw | context_relevance | 0.1453 | 1.2351 |
| controlled_vs_proposed_raw | persona_consistency | 0.1282 | 0.5288 |
| controlled_vs_proposed_raw | naturalness | 0.0533 | 0.0647 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.1860 | 1.3070 |
| controlled_vs_proposed_raw | context_overlap | 0.0504 | 0.8378 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1548 | 1.0759 |
| controlled_vs_proposed_raw | persona_style | 0.0218 | 0.0343 |
| controlled_vs_proposed_raw | distinct1 | -0.0117 | -0.0125 |
| controlled_vs_proposed_raw | length_score | 0.2389 | 0.7748 |
| controlled_vs_proposed_raw | sentence_score | 0.1021 | 0.1195 |
| controlled_vs_proposed_raw | overall_quality | 0.1256 | 0.4189 |
| controlled_vs_candidate_no_context | context_relevance | 0.2324 | 7.6098 |
| controlled_vs_candidate_no_context | persona_consistency | 0.2000 | 1.1720 |
| controlled_vs_candidate_no_context | naturalness | 0.0635 | 0.0781 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3030 | 12.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0676 | 1.5765 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2331 | 3.5606 |
| controlled_vs_candidate_no_context | persona_style | 0.0673 | 0.1138 |
| controlled_vs_candidate_no_context | distinct1 | -0.0094 | -0.0100 |
| controlled_vs_candidate_no_context | length_score | 0.2486 | 0.8326 |
| controlled_vs_candidate_no_context | sentence_score | 0.1750 | 0.2240 |
| controlled_vs_candidate_no_context | overall_quality | 0.1946 | 0.8431 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0157 | -0.0597 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0278 | -0.0750 |
| controlled_alt_vs_controlled_default | naturalness | 0.0259 | 0.0296 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0193 | -0.0587 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0074 | -0.0669 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0403 | -0.1349 |
| controlled_alt_vs_controlled_default | persona_style | 0.0221 | 0.0336 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0116 | 0.0124 |
| controlled_alt_vs_controlled_default | length_score | 0.1139 | 0.2081 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0146 | -0.0153 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0128 | -0.0300 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1296 | 1.1017 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1004 | 0.4141 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0792 | 0.0962 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.1667 | 1.1717 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0430 | 0.7148 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1145 | 0.7959 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0440 | 0.0691 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0002 | -0.0002 |
| controlled_alt_vs_proposed_raw | length_score | 0.3528 | 1.1441 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0875 | 0.1024 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1128 | 0.3764 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2167 | 7.0958 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1722 | 1.0091 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0894 | 0.1100 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.2838 | 11.2375 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0602 | 1.4040 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1929 | 2.9455 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0894 | 0.1512 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0022 | 0.0023 |
| controlled_alt_vs_candidate_no_context | length_score | 0.3625 | 1.2140 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1604 | 0.2053 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1818 | 0.7878 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0871 | (0.0374, 0.1430) | 0.0000 | 0.0871 | (0.0219, 0.1557) | 0.0007 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0718 | (0.0206, 0.1260) | 0.0017 | 0.0718 | (0.0024, 0.1401) | 0.0203 |
| proposed_vs_candidate_no_context | naturalness | 0.0102 | (-0.0176, 0.0405) | 0.2450 | 0.0102 | (-0.0187, 0.0368) | 0.2940 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.1170 | (0.0501, 0.1883) | 0.0000 | 0.1170 | (0.0217, 0.2088) | 0.0040 |
| proposed_vs_candidate_no_context | context_overlap | 0.0172 | (0.0064, 0.0287) | 0.0000 | 0.0172 | (0.0054, 0.0274) | 0.0027 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0784 | (0.0179, 0.1409) | 0.0047 | 0.0784 | (-0.0050, 0.1565) | 0.0340 |
| proposed_vs_candidate_no_context | persona_style | 0.0454 | (-0.0040, 0.1106) | 0.0483 | 0.0454 | (-0.0072, 0.1440) | 0.1027 |
| proposed_vs_candidate_no_context | distinct1 | 0.0024 | (-0.0135, 0.0175) | 0.3680 | 0.0024 | (-0.0096, 0.0123) | 0.3617 |
| proposed_vs_candidate_no_context | length_score | 0.0097 | (-0.1070, 0.1347) | 0.4417 | 0.0097 | (-0.1175, 0.1161) | 0.4633 |
| proposed_vs_candidate_no_context | sentence_score | 0.0729 | (0.0000, 0.1458) | 0.0290 | 0.0729 | (0.0140, 0.1333) | 0.0233 |
| proposed_vs_candidate_no_context | overall_quality | 0.0690 | (0.0337, 0.1085) | 0.0000 | 0.0690 | (0.0136, 0.1264) | 0.0090 |
| controlled_vs_proposed_raw | context_relevance | 0.1453 | (0.0845, 0.1979) | 0.0000 | 0.1453 | (0.0606, 0.2227) | 0.0010 |
| controlled_vs_proposed_raw | persona_consistency | 0.1282 | (0.0518, 0.2136) | 0.0007 | 0.1282 | (0.0513, 0.2186) | 0.0003 |
| controlled_vs_proposed_raw | naturalness | 0.0533 | (0.0071, 0.1003) | 0.0163 | 0.0533 | (-0.0053, 0.1180) | 0.0373 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.1860 | (0.1047, 0.2566) | 0.0000 | 0.1860 | (0.0741, 0.2841) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0504 | (0.0302, 0.0729) | 0.0000 | 0.0504 | (0.0257, 0.0745) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1548 | (0.0615, 0.2512) | 0.0003 | 0.1548 | (0.0606, 0.2635) | 0.0003 |
| controlled_vs_proposed_raw | persona_style | 0.0218 | (-0.0298, 0.0747) | 0.2013 | 0.0218 | (0.0033, 0.0568) | 0.0207 |
| controlled_vs_proposed_raw | distinct1 | -0.0117 | (-0.0313, 0.0085) | 0.8737 | -0.0117 | (-0.0336, 0.0132) | 0.8227 |
| controlled_vs_proposed_raw | length_score | 0.2389 | (0.0389, 0.4292) | 0.0100 | 0.2389 | (-0.0011, 0.4833) | 0.0253 |
| controlled_vs_proposed_raw | sentence_score | 0.1021 | (0.0146, 0.1896) | 0.0173 | 0.1021 | (-0.0167, 0.2027) | 0.0570 |
| controlled_vs_proposed_raw | overall_quality | 0.1256 | (0.0786, 0.1739) | 0.0000 | 0.1256 | (0.0647, 0.2005) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2324 | (0.2072, 0.2575) | 0.0000 | 0.2324 | (0.2044, 0.2529) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.2000 | (0.1337, 0.2707) | 0.0000 | 0.2000 | (0.1296, 0.2736) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.0635 | (0.0208, 0.1047) | 0.0007 | 0.0635 | (0.0127, 0.1146) | 0.0083 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3030 | (0.2705, 0.3386) | 0.0000 | 0.3030 | (0.2727, 0.3281) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0676 | (0.0521, 0.0853) | 0.0000 | 0.0676 | (0.0501, 0.0855) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2331 | (0.1486, 0.3145) | 0.0000 | 0.2331 | (0.1415, 0.3114) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.0673 | (0.0156, 0.1328) | 0.0040 | 0.0673 | (0.0081, 0.1743) | 0.0240 |
| controlled_vs_candidate_no_context | distinct1 | -0.0094 | (-0.0308, 0.0121) | 0.7923 | -0.0094 | (-0.0338, 0.0157) | 0.7637 |
| controlled_vs_candidate_no_context | length_score | 0.2486 | (0.0667, 0.4070) | 0.0040 | 0.2486 | (0.0458, 0.4556) | 0.0083 |
| controlled_vs_candidate_no_context | sentence_score | 0.1750 | (0.1021, 0.2479) | 0.0000 | 0.1750 | (0.0875, 0.2386) | 0.0003 |
| controlled_vs_candidate_no_context | overall_quality | 0.1946 | (0.1642, 0.2277) | 0.0000 | 0.1946 | (0.1587, 0.2277) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0157 | (-0.0529, 0.0236) | 0.7957 | -0.0157 | (-0.0487, 0.0088) | 0.8597 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0278 | (-0.0888, 0.0304) | 0.8260 | -0.0278 | (-0.0813, 0.0149) | 0.8283 |
| controlled_alt_vs_controlled_default | naturalness | 0.0259 | (-0.0070, 0.0579) | 0.0707 | 0.0259 | (0.0023, 0.0538) | 0.0170 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0193 | (-0.0713, 0.0310) | 0.7763 | -0.0193 | (-0.0593, 0.0125) | 0.8737 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0074 | (-0.0254, 0.0093) | 0.7917 | -0.0074 | (-0.0217, 0.0047) | 0.8757 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0403 | (-0.1117, 0.0298) | 0.8690 | -0.0403 | (-0.1026, 0.0089) | 0.9113 |
| controlled_alt_vs_controlled_default | persona_style | 0.0221 | (0.0000, 0.0560) | 0.1183 | 0.0221 | (0.0000, 0.0573) | 0.1000 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0116 | (-0.0016, 0.0255) | 0.0450 | 0.0116 | (0.0012, 0.0271) | 0.0143 |
| controlled_alt_vs_controlled_default | length_score | 0.1139 | (-0.0431, 0.2681) | 0.0740 | 0.1139 | (-0.0015, 0.2238) | 0.0277 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0146 | (-0.0583, 0.0292) | 0.8033 | -0.0146 | (-0.0538, 0.0368) | 0.8080 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0128 | (-0.0416, 0.0115) | 0.8253 | -0.0128 | (-0.0451, 0.0074) | 0.7287 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1296 | (0.0640, 0.1842) | 0.0000 | 0.1296 | (0.0499, 0.1969) | 0.0010 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1004 | (0.0531, 0.1462) | 0.0003 | 0.1004 | (0.0473, 0.1487) | 0.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0792 | (0.0400, 0.1203) | 0.0000 | 0.0792 | (0.0309, 0.1361) | 0.0010 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.1667 | (0.0811, 0.2386) | 0.0003 | 0.1667 | (0.0550, 0.2538) | 0.0033 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0430 | (0.0235, 0.0607) | 0.0000 | 0.0430 | (0.0205, 0.0634) | 0.0003 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1145 | (0.0563, 0.1714) | 0.0000 | 0.1145 | (0.0516, 0.1718) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0440 | (0.0104, 0.0839) | 0.0040 | 0.0440 | (0.0104, 0.0931) | 0.0037 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0002 | (-0.0232, 0.0219) | 0.5080 | -0.0002 | (-0.0240, 0.0280) | 0.5230 |
| controlled_alt_vs_proposed_raw | length_score | 0.3528 | (0.1889, 0.5139) | 0.0000 | 0.3528 | (0.1702, 0.5651) | 0.0000 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0875 | (-0.0004, 0.1750) | 0.0470 | 0.0875 | (0.0000, 0.1909) | 0.0407 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1128 | (0.0730, 0.1499) | 0.0000 | 0.1128 | (0.0677, 0.1602) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2167 | (0.1842, 0.2540) | 0.0000 | 0.2167 | (0.1774, 0.2478) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1722 | (0.1283, 0.2220) | 0.0000 | 0.1722 | (0.1306, 0.2203) | 0.0000 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0894 | (0.0491, 0.1288) | 0.0000 | 0.0894 | (0.0473, 0.1323) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.2838 | (0.2417, 0.3300) | 0.0000 | 0.2838 | (0.2340, 0.3229) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0602 | (0.0434, 0.0776) | 0.0000 | 0.0602 | (0.0395, 0.0788) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1929 | (0.1405, 0.2435) | 0.0000 | 0.1929 | (0.1429, 0.2321) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0894 | (0.0208, 0.1714) | 0.0017 | 0.0894 | (0.0096, 0.2266) | 0.0250 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0022 | (-0.0210, 0.0250) | 0.4053 | 0.0022 | (-0.0202, 0.0278) | 0.4597 |
| controlled_alt_vs_candidate_no_context | length_score | 0.3625 | (0.1861, 0.5167) | 0.0000 | 0.3625 | (0.2063, 0.5319) | 0.0000 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1604 | (0.0729, 0.2333) | 0.0000 | 0.1604 | (0.0972, 0.2283) | 0.0000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1818 | (0.1573, 0.2049) | 0.0000 | 0.1818 | (0.1528, 0.2016) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 13 | 3 | 8 | 0.7083 | 0.8125 |
| proposed_vs_candidate_no_context | persona_consistency | 10 | 4 | 10 | 0.6250 | 0.7143 |
| proposed_vs_candidate_no_context | naturalness | 9 | 7 | 8 | 0.5417 | 0.5625 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 10 | 1 | 13 | 0.6875 | 0.9091 |
| proposed_vs_candidate_no_context | context_overlap | 12 | 4 | 8 | 0.6667 | 0.7500 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 9 | 3 | 12 | 0.6250 | 0.7500 |
| proposed_vs_candidate_no_context | persona_style | 3 | 1 | 20 | 0.5417 | 0.7500 |
| proposed_vs_candidate_no_context | distinct1 | 9 | 5 | 10 | 0.5833 | 0.6429 |
| proposed_vs_candidate_no_context | length_score | 8 | 8 | 8 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | sentence_score | 6 | 1 | 17 | 0.6042 | 0.8571 |
| proposed_vs_candidate_no_context | overall_quality | 13 | 3 | 8 | 0.7083 | 0.8125 |
| controlled_vs_proposed_raw | context_relevance | 22 | 2 | 0 | 0.9167 | 0.9167 |
| controlled_vs_proposed_raw | persona_consistency | 16 | 5 | 3 | 0.7292 | 0.7619 |
| controlled_vs_proposed_raw | naturalness | 15 | 8 | 1 | 0.6458 | 0.6522 |
| controlled_vs_proposed_raw | context_keyword_coverage | 19 | 1 | 4 | 0.8750 | 0.9500 |
| controlled_vs_proposed_raw | context_overlap | 21 | 3 | 0 | 0.8750 | 0.8750 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 16 | 5 | 3 | 0.7292 | 0.7619 |
| controlled_vs_proposed_raw | persona_style | 5 | 2 | 17 | 0.5625 | 0.7143 |
| controlled_vs_proposed_raw | distinct1 | 9 | 13 | 2 | 0.4167 | 0.4091 |
| controlled_vs_proposed_raw | length_score | 14 | 8 | 2 | 0.6250 | 0.6364 |
| controlled_vs_proposed_raw | sentence_score | 9 | 2 | 13 | 0.6458 | 0.8182 |
| controlled_vs_proposed_raw | overall_quality | 19 | 5 | 0 | 0.7917 | 0.7917 |
| controlled_vs_candidate_no_context | context_relevance | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 21 | 1 | 2 | 0.9167 | 0.9545 |
| controlled_vs_candidate_no_context | naturalness | 19 | 5 | 0 | 0.7917 | 0.7917 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 21 | 1 | 2 | 0.9167 | 0.9545 |
| controlled_vs_candidate_no_context | persona_style | 5 | 0 | 19 | 0.6042 | 1.0000 |
| controlled_vs_candidate_no_context | distinct1 | 11 | 11 | 2 | 0.5000 | 0.5000 |
| controlled_vs_candidate_no_context | length_score | 17 | 6 | 1 | 0.7292 | 0.7391 |
| controlled_vs_candidate_no_context | sentence_score | 12 | 0 | 12 | 0.7500 | 1.0000 |
| controlled_vs_candidate_no_context | overall_quality | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 4 | 11 | 9 | 0.3542 | 0.2667 |
| controlled_alt_vs_controlled_default | persona_consistency | 3 | 6 | 15 | 0.4375 | 0.3333 |
| controlled_alt_vs_controlled_default | naturalness | 9 | 6 | 9 | 0.5625 | 0.6000 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 3 | 7 | 14 | 0.4167 | 0.3000 |
| controlled_alt_vs_controlled_default | context_overlap | 5 | 10 | 9 | 0.3958 | 0.3333 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 2 | 6 | 16 | 0.4167 | 0.2500 |
| controlled_alt_vs_controlled_default | persona_style | 2 | 0 | 22 | 0.5417 | 1.0000 |
| controlled_alt_vs_controlled_default | distinct1 | 10 | 5 | 9 | 0.6042 | 0.6667 |
| controlled_alt_vs_controlled_default | length_score | 10 | 4 | 10 | 0.6250 | 0.7143 |
| controlled_alt_vs_controlled_default | sentence_score | 1 | 2 | 21 | 0.4792 | 0.3333 |
| controlled_alt_vs_controlled_default | overall_quality | 7 | 8 | 9 | 0.4792 | 0.4667 |
| controlled_alt_vs_proposed_raw | context_relevance | 21 | 3 | 0 | 0.8750 | 0.8750 |
| controlled_alt_vs_proposed_raw | persona_consistency | 17 | 3 | 4 | 0.7917 | 0.8500 |
| controlled_alt_vs_proposed_raw | naturalness | 17 | 6 | 1 | 0.7292 | 0.7391 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 20 | 2 | 2 | 0.8750 | 0.9091 |
| controlled_alt_vs_proposed_raw | context_overlap | 21 | 3 | 0 | 0.8750 | 0.8750 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 16 | 3 | 5 | 0.7708 | 0.8421 |
| controlled_alt_vs_proposed_raw | persona_style | 5 | 0 | 19 | 0.6042 | 1.0000 |
| controlled_alt_vs_proposed_raw | distinct1 | 13 | 10 | 1 | 0.5625 | 0.5652 |
| controlled_alt_vs_proposed_raw | length_score | 18 | 4 | 2 | 0.7917 | 0.8182 |
| controlled_alt_vs_proposed_raw | sentence_score | 9 | 3 | 12 | 0.6250 | 0.7500 |
| controlled_alt_vs_proposed_raw | overall_quality | 21 | 3 | 0 | 0.8750 | 0.8750 |
| controlled_alt_vs_candidate_no_context | context_relevance | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 21 | 1 | 2 | 0.9167 | 0.9545 |
| controlled_alt_vs_candidate_no_context | naturalness | 19 | 5 | 0 | 0.7917 | 0.7917 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 21 | 1 | 2 | 0.9167 | 0.9545 |
| controlled_alt_vs_candidate_no_context | persona_style | 6 | 0 | 18 | 0.6250 | 1.0000 |
| controlled_alt_vs_candidate_no_context | distinct1 | 16 | 8 | 0 | 0.6667 | 0.6667 |
| controlled_alt_vs_candidate_no_context | length_score | 19 | 5 | 0 | 0.7917 | 0.7917 |
| controlled_alt_vs_candidate_no_context | sentence_score | 12 | 1 | 11 | 0.7292 | 0.9231 |
| controlled_alt_vs_candidate_no_context | overall_quality | 24 | 0 | 0 | 1.0000 | 1.0000 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.0000 | 0.4167 | 0.1250 |
| proposed_contextual_controlled_alt | 0.0000 | 0.0000 | 0.0000 | 0.4167 | 0.1250 |
| proposed_contextual | 0.0000 | 0.0000 | 0.3333 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `8`
- Unique template signatures: `20`
- Template signature ratio: `0.8333`
- Effective sample size by source clustering: `6.86`
- Effective sample size by template-signature clustering: `18.00`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: unavailable (disabled_by_flag).

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
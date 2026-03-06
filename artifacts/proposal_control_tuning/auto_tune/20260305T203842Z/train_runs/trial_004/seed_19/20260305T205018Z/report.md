# Proposal Alignment Evaluation Report

- Run ID: `20260305T205018Z`
- Generated: `2026-03-05T20:53:36.177986+00:00`
- Scenarios: `artifacts\proposal_control_tuning\auto_tune\20260305T203842Z\train_runs\trial_004\seed_19\20260305T205018Z\scenarios.jsonl`
- Scenario count: `20`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_tuned`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2571 (0.2276, 0.2900) | 0.3178 (0.2505, 0.3921) | 0.9122 (0.8942, 0.9284) | 0.4106 (0.3852, 0.4384) | n/a |
| proposed_contextual_controlled_tuned | 0.2843 (0.2431, 0.3309) | 0.2704 (0.2381, 0.3088) | 0.8765 (0.8428, 0.9078) | 0.3984 (0.3764, 0.4215) | n/a |
| proposed_contextual | 0.0683 (0.0396, 0.0990) | 0.1337 (0.0950, 0.1776) | 0.8251 (0.7906, 0.8608) | 0.2372 (0.2106, 0.2664) | n/a |
| candidate_no_context | 0.0201 (0.0115, 0.0305) | 0.1398 (0.0957, 0.1845) | 0.7889 (0.7643, 0.8175) | 0.2099 (0.1898, 0.2324) | n/a |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0482 | 2.4035 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0061 | -0.0435 |
| proposed_vs_candidate_no_context | naturalness | 0.0362 | 0.0459 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0636 | 4.6667 |
| proposed_vs_candidate_no_context | context_overlap | 0.0122 | 0.3485 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0095 | -0.3077 |
| proposed_vs_candidate_no_context | persona_style | 0.0077 | 0.0134 |
| proposed_vs_candidate_no_context | distinct1 | 0.0153 | 0.0164 |
| proposed_vs_candidate_no_context | length_score | 0.1067 | 0.4604 |
| proposed_vs_candidate_no_context | sentence_score | 0.0875 | 0.1277 |
| proposed_vs_candidate_no_context | overall_quality | 0.0273 | 0.1301 |
| controlled_vs_proposed_raw | context_relevance | 0.1889 | 2.7665 |
| controlled_vs_proposed_raw | persona_consistency | 0.1841 | 1.3770 |
| controlled_vs_proposed_raw | naturalness | 0.0870 | 0.1055 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2455 | 3.1765 |
| controlled_vs_proposed_raw | context_overlap | 0.0568 | 1.2024 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2195 | 10.2444 |
| controlled_vs_proposed_raw | persona_style | 0.0423 | 0.0726 |
| controlled_vs_proposed_raw | distinct1 | -0.0067 | -0.0071 |
| controlled_vs_proposed_raw | length_score | 0.3717 | 1.0985 |
| controlled_vs_proposed_raw | sentence_score | 0.1575 | 0.2039 |
| controlled_vs_proposed_raw | overall_quality | 0.1734 | 0.7312 |
| controlled_vs_candidate_no_context | context_relevance | 0.2371 | 11.8194 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1780 | 1.2736 |
| controlled_vs_candidate_no_context | naturalness | 0.1233 | 0.1562 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3091 | 22.6667 |
| controlled_vs_candidate_no_context | context_overlap | 0.0690 | 1.9700 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2100 | 6.7846 |
| controlled_vs_candidate_no_context | persona_style | 0.0500 | 0.0870 |
| controlled_vs_candidate_no_context | distinct1 | 0.0086 | 0.0092 |
| controlled_vs_candidate_no_context | length_score | 0.4783 | 2.0647 |
| controlled_vs_candidate_no_context | sentence_score | 0.2450 | 0.3577 |
| controlled_vs_candidate_no_context | overall_quality | 0.2007 | 0.9564 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0272 | 0.1058 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0473 | -0.1490 |
| controlled_alt_vs_controlled_default | naturalness | -0.0356 | -0.0391 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0364 | 0.1127 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0058 | 0.0558 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0588 | -0.2441 |
| controlled_alt_vs_controlled_default | persona_style | -0.0015 | -0.0023 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0107 | -0.0113 |
| controlled_alt_vs_controlled_default | length_score | -0.1083 | -0.1526 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0875 | -0.0941 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0122 | -0.0296 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.2161 | 3.1649 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1367 | 1.0228 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0514 | 0.0623 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2818 | 3.6471 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0626 | 1.3253 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1607 | 7.5000 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0408 | 0.0701 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0174 | -0.0183 |
| controlled_alt_vs_proposed_raw | length_score | 0.2633 | 0.7783 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0700 | 0.0906 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1613 | 0.6800 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2643 | 13.1753 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1307 | 0.9349 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0876 | 0.1111 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3455 | 25.3333 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0748 | 2.1357 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1512 | 4.8846 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0485 | 0.0844 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0020 | -0.0022 |
| controlled_alt_vs_candidate_no_context | length_score | 0.3700 | 1.5971 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1575 | 0.2299 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1886 | 0.8985 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0482 | (0.0217, 0.0781) | 0.0000 | 0.0482 | (0.0235, 0.0752) | 0.0003 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0061 | (-0.0422, 0.0241) | 0.6513 | -0.0061 | (-0.0407, 0.0140) | 0.6507 |
| proposed_vs_candidate_no_context | naturalness | 0.0362 | (0.0137, 0.0644) | 0.0000 | 0.0362 | (0.0132, 0.0623) | 0.0003 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0636 | (0.0273, 0.1000) | 0.0000 | 0.0636 | (0.0287, 0.1005) | 0.0000 |
| proposed_vs_candidate_no_context | context_overlap | 0.0122 | (-0.0003, 0.0280) | 0.0297 | 0.0122 | (0.0008, 0.0265) | 0.0150 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0095 | (-0.0571, 0.0286) | 0.6997 | -0.0095 | (-0.0526, 0.0171) | 0.7157 |
| proposed_vs_candidate_no_context | persona_style | 0.0077 | (-0.0019, 0.0250) | 0.3613 | 0.0077 | (-0.0014, 0.0294) | 0.3167 |
| proposed_vs_candidate_no_context | distinct1 | 0.0153 | (0.0030, 0.0292) | 0.0047 | 0.0153 | (0.0016, 0.0305) | 0.0097 |
| proposed_vs_candidate_no_context | length_score | 0.1067 | (0.0333, 0.2000) | 0.0000 | 0.1067 | (0.0365, 0.1863) | 0.0000 |
| proposed_vs_candidate_no_context | sentence_score | 0.0875 | (0.0350, 0.1575) | 0.0043 | 0.0875 | (0.0175, 0.1842) | 0.0113 |
| proposed_vs_candidate_no_context | overall_quality | 0.0273 | (0.0110, 0.0459) | 0.0003 | 0.0273 | (0.0152, 0.0364) | 0.0000 |
| controlled_vs_proposed_raw | context_relevance | 0.1889 | (0.1556, 0.2204) | 0.0000 | 0.1889 | (0.1712, 0.2081) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.1841 | (0.1168, 0.2563) | 0.0000 | 0.1841 | (0.1244, 0.2349) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0870 | (0.0452, 0.1276) | 0.0000 | 0.0870 | (0.0225, 0.1513) | 0.0043 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2455 | (0.2000, 0.2864) | 0.0000 | 0.2455 | (0.2208, 0.2689) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0568 | (0.0421, 0.0722) | 0.0000 | 0.0568 | (0.0441, 0.0685) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2195 | (0.1398, 0.3031) | 0.0000 | 0.2195 | (0.1556, 0.2815) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.0423 | (0.0000, 0.1090) | 0.0310 | 0.0423 | (0.0000, 0.1471) | 0.0743 |
| controlled_vs_proposed_raw | distinct1 | -0.0067 | (-0.0332, 0.0188) | 0.6900 | -0.0067 | (-0.0425, 0.0304) | 0.6510 |
| controlled_vs_proposed_raw | length_score | 0.3717 | (0.2100, 0.5183) | 0.0000 | 0.3717 | (0.1463, 0.5825) | 0.0003 |
| controlled_vs_proposed_raw | sentence_score | 0.1575 | (0.0700, 0.2450) | 0.0003 | 0.1575 | (0.0737, 0.2567) | 0.0000 |
| controlled_vs_proposed_raw | overall_quality | 0.1734 | (0.1365, 0.2067) | 0.0000 | 0.1734 | (0.1399, 0.1965) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2371 | (0.2047, 0.2712) | 0.0000 | 0.2371 | (0.2116, 0.2636) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1780 | (0.1137, 0.2413) | 0.0000 | 0.1780 | (0.0948, 0.2469) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.1233 | (0.0906, 0.1542) | 0.0000 | 0.1233 | (0.0817, 0.1715) | 0.0000 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3091 | (0.2636, 0.3545) | 0.0000 | 0.3091 | (0.2771, 0.3445) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0690 | (0.0554, 0.0829) | 0.0000 | 0.0690 | (0.0563, 0.0790) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2100 | (0.1338, 0.2922) | 0.0000 | 0.2100 | (0.1185, 0.2987) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.0500 | (0.0000, 0.1167) | 0.0410 | 0.0500 | (0.0000, 0.1765) | 0.3403 |
| controlled_vs_candidate_no_context | distinct1 | 0.0086 | (-0.0150, 0.0320) | 0.2323 | 0.0086 | (-0.0145, 0.0381) | 0.2537 |
| controlled_vs_candidate_no_context | length_score | 0.4783 | (0.3350, 0.6033) | 0.0000 | 0.4783 | (0.3045, 0.6313) | 0.0000 |
| controlled_vs_candidate_no_context | sentence_score | 0.2450 | (0.1750, 0.3150) | 0.0000 | 0.2450 | (0.1658, 0.3281) | 0.0000 |
| controlled_vs_candidate_no_context | overall_quality | 0.2007 | (0.1747, 0.2271) | 0.0000 | 0.2007 | (0.1692, 0.2265) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0272 | (-0.0142, 0.0792) | 0.1083 | 0.0272 | (0.0060, 0.0412) | 0.0093 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0473 | (-0.1205, 0.0154) | 0.9233 | -0.0473 | (-0.1221, 0.0170) | 0.8250 |
| controlled_alt_vs_controlled_default | naturalness | -0.0356 | (-0.0751, 0.0028) | 0.9633 | -0.0356 | (-0.0677, 0.0005) | 0.9710 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0364 | (-0.0183, 0.1091) | 0.1257 | 0.0364 | (0.0048, 0.0606) | 0.0187 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0058 | (-0.0129, 0.0248) | 0.2783 | 0.0058 | (-0.0066, 0.0245) | 0.2163 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0588 | (-0.1443, 0.0155) | 0.9310 | -0.0588 | (-0.1451, 0.0185) | 0.8643 |
| controlled_alt_vs_controlled_default | persona_style | -0.0015 | (-0.0454, 0.0502) | 0.5520 | -0.0015 | (-0.0418, 0.0588) | 0.6367 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0107 | (-0.0384, 0.0162) | 0.7813 | -0.0107 | (-0.0429, 0.0220) | 0.7620 |
| controlled_alt_vs_controlled_default | length_score | -0.1083 | (-0.2550, 0.0400) | 0.9247 | -0.1083 | (-0.2240, 0.0053) | 0.9553 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0875 | (-0.1575, -0.0175) | 1.0000 | -0.0875 | (-0.1114, -0.0437) | 1.0000 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0122 | (-0.0483, 0.0218) | 0.7343 | -0.0122 | (-0.0401, 0.0152) | 0.7060 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.2161 | (0.1746, 0.2608) | 0.0000 | 0.2161 | (0.1795, 0.2493) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1367 | (0.0888, 0.1812) | 0.0000 | 0.1367 | (0.0990, 0.1886) | 0.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0514 | (-0.0017, 0.1031) | 0.0277 | 0.0514 | (0.0016, 0.1150) | 0.0237 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2818 | (0.2273, 0.3409) | 0.0000 | 0.2818 | (0.2294, 0.3333) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0626 | (0.0412, 0.0832) | 0.0000 | 0.0626 | (0.0486, 0.0764) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1607 | (0.1093, 0.2126) | 0.0000 | 0.1607 | (0.1310, 0.1963) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0408 | (-0.0213, 0.1144) | 0.1353 | 0.0408 | (-0.0404, 0.2059) | 0.3377 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0174 | (-0.0338, -0.0001) | 0.9757 | -0.0174 | (-0.0331, -0.0050) | 1.0000 |
| controlled_alt_vs_proposed_raw | length_score | 0.2633 | (0.0633, 0.4550) | 0.0053 | 0.2633 | (0.0739, 0.5196) | 0.0000 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0700 | (-0.0350, 0.1750) | 0.1217 | 0.0700 | (-0.0194, 0.2139) | 0.1057 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1613 | (0.1335, 0.1916) | 0.0000 | 0.1613 | (0.1389, 0.1896) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2643 | (0.2201, 0.3096) | 0.0000 | 0.2643 | (0.2375, 0.2961) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1307 | (0.0830, 0.1804) | 0.0000 | 0.1307 | (0.1020, 0.1850) | 0.0000 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0876 | (0.0429, 0.1339) | 0.0000 | 0.0876 | (0.0525, 0.1361) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3455 | (0.2909, 0.4091) | 0.0000 | 0.3455 | (0.3030, 0.3889) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0748 | (0.0577, 0.0920) | 0.0000 | 0.0748 | (0.0626, 0.0878) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1512 | (0.0962, 0.2024) | 0.0000 | 0.1512 | (0.1329, 0.1745) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0485 | (-0.0213, 0.1327) | 0.1027 | 0.0485 | (-0.0418, 0.2353) | 0.3410 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0020 | (-0.0210, 0.0172) | 0.5960 | -0.0020 | (-0.0156, 0.0181) | 0.6093 |
| controlled_alt_vs_candidate_no_context | length_score | 0.3700 | (0.2017, 0.5334) | 0.0000 | 0.3700 | (0.2188, 0.5745) | 0.0000 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1575 | (0.0700, 0.2450) | 0.0010 | 0.1575 | (0.0700, 0.2625) | 0.0003 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1886 | (0.1606, 0.2173) | 0.0000 | 0.1886 | (0.1670, 0.2171) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 10 | 1 | 9 | 0.7250 | 0.9091 |
| proposed_vs_candidate_no_context | persona_consistency | 3 | 2 | 15 | 0.5250 | 0.6000 |
| proposed_vs_candidate_no_context | naturalness | 10 | 1 | 9 | 0.7250 | 0.9091 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 9 | 0 | 11 | 0.7250 | 1.0000 |
| proposed_vs_candidate_no_context | context_overlap | 7 | 4 | 9 | 0.5750 | 0.6364 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 2 | 2 | 16 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | persona_style | 1 | 1 | 18 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | distinct1 | 7 | 1 | 12 | 0.6500 | 0.8750 |
| proposed_vs_candidate_no_context | length_score | 10 | 1 | 9 | 0.7250 | 0.9091 |
| proposed_vs_candidate_no_context | sentence_score | 5 | 0 | 15 | 0.6250 | 1.0000 |
| proposed_vs_candidate_no_context | overall_quality | 9 | 2 | 9 | 0.6750 | 0.8182 |
| controlled_vs_proposed_raw | context_relevance | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_proposed_raw | persona_consistency | 17 | 1 | 2 | 0.9000 | 0.9444 |
| controlled_vs_proposed_raw | naturalness | 15 | 5 | 0 | 0.7500 | 0.7500 |
| controlled_vs_proposed_raw | context_keyword_coverage | 19 | 0 | 1 | 0.9750 | 1.0000 |
| controlled_vs_proposed_raw | context_overlap | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 17 | 1 | 2 | 0.9000 | 0.9444 |
| controlled_vs_proposed_raw | persona_style | 3 | 0 | 17 | 0.5750 | 1.0000 |
| controlled_vs_proposed_raw | distinct1 | 9 | 10 | 1 | 0.4750 | 0.4737 |
| controlled_vs_proposed_raw | length_score | 17 | 3 | 0 | 0.8500 | 0.8500 |
| controlled_vs_proposed_raw | sentence_score | 10 | 1 | 9 | 0.7250 | 0.9091 |
| controlled_vs_proposed_raw | overall_quality | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_vs_candidate_no_context | context_relevance | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 17 | 1 | 2 | 0.9000 | 0.9444 |
| controlled_vs_candidate_no_context | naturalness | 18 | 2 | 0 | 0.9000 | 0.9000 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 17 | 1 | 2 | 0.9000 | 0.9444 |
| controlled_vs_candidate_no_context | persona_style | 3 | 0 | 17 | 0.5750 | 1.0000 |
| controlled_vs_candidate_no_context | distinct1 | 11 | 8 | 1 | 0.5750 | 0.5789 |
| controlled_vs_candidate_no_context | length_score | 18 | 1 | 1 | 0.9250 | 0.9474 |
| controlled_vs_candidate_no_context | sentence_score | 14 | 0 | 6 | 0.8500 | 1.0000 |
| controlled_vs_candidate_no_context | overall_quality | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 10 | 9 | 1 | 0.5250 | 0.5263 |
| controlled_alt_vs_controlled_default | persona_consistency | 5 | 6 | 9 | 0.4750 | 0.4545 |
| controlled_alt_vs_controlled_default | naturalness | 7 | 12 | 1 | 0.3750 | 0.3684 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 7 | 6 | 7 | 0.5250 | 0.5385 |
| controlled_alt_vs_controlled_default | context_overlap | 11 | 8 | 1 | 0.5750 | 0.5789 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 3 | 5 | 12 | 0.4500 | 0.3750 |
| controlled_alt_vs_controlled_default | persona_style | 2 | 3 | 15 | 0.4750 | 0.4000 |
| controlled_alt_vs_controlled_default | distinct1 | 8 | 11 | 1 | 0.4250 | 0.4211 |
| controlled_alt_vs_controlled_default | length_score | 7 | 10 | 3 | 0.4250 | 0.4118 |
| controlled_alt_vs_controlled_default | sentence_score | 0 | 5 | 15 | 0.3750 | 0.0000 |
| controlled_alt_vs_controlled_default | overall_quality | 10 | 9 | 1 | 0.5250 | 0.5263 |
| controlled_alt_vs_proposed_raw | context_relevance | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 17 | 1 | 2 | 0.9000 | 0.9444 |
| controlled_alt_vs_proposed_raw | naturalness | 12 | 8 | 0 | 0.6000 | 0.6000 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 18 | 2 | 0 | 0.9000 | 0.9000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 17 | 1 | 2 | 0.9000 | 0.9444 |
| controlled_alt_vs_proposed_raw | persona_style | 3 | 2 | 15 | 0.5250 | 0.6000 |
| controlled_alt_vs_proposed_raw | distinct1 | 6 | 12 | 2 | 0.3500 | 0.3333 |
| controlled_alt_vs_proposed_raw | length_score | 14 | 6 | 0 | 0.7000 | 0.7000 |
| controlled_alt_vs_proposed_raw | sentence_score | 7 | 3 | 10 | 0.6000 | 0.7000 |
| controlled_alt_vs_proposed_raw | overall_quality | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 16 | 2 | 2 | 0.8500 | 0.8889 |
| controlled_alt_vs_candidate_no_context | naturalness | 16 | 4 | 0 | 0.8000 | 0.8000 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 16 | 1 | 3 | 0.8750 | 0.9412 |
| controlled_alt_vs_candidate_no_context | persona_style | 3 | 2 | 15 | 0.5250 | 0.6000 |
| controlled_alt_vs_candidate_no_context | distinct1 | 9 | 10 | 1 | 0.4750 | 0.4737 |
| controlled_alt_vs_candidate_no_context | length_score | 14 | 6 | 0 | 0.7000 | 0.7000 |
| controlled_alt_vs_candidate_no_context | sentence_score | 10 | 1 | 9 | 0.7250 | 0.9091 |
| controlled_alt_vs_candidate_no_context | overall_quality | 20 | 0 | 0 | 1.0000 | 1.0000 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.3500 | 0.3000 | 0.7000 |
| proposed_contextual_controlled_tuned | 0.0000 | 0.0000 | 0.1500 | 0.5500 | 0.4500 |
| proposed_contextual | 0.0000 | 0.0000 | 0.4500 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.6000 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `5`
- Unique template signatures: `17`
- Template signature ratio: `0.8500`
- Effective sample size by source clustering: `4.65`
- Effective sample size by template-signature clustering: `15.38`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: unavailable (disabled_by_flag).

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
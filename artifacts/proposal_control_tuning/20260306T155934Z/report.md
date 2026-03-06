# Proposal Alignment Evaluation Report

- Run ID: `20260306T155934Z`
- Generated: `2026-03-06T16:02:07.907112+00:00`
- Scenarios: `artifacts\proposal_control_tuning\20260306T155934Z\scenarios.jsonl`
- Scenario count: `16`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2754 (0.2288, 0.3213) | 0.2948 (0.2487, 0.3381) | 0.8992 (0.8596, 0.9334) | 0.4077 (0.3847, 0.4309) | n/a |
| proposed_contextual | 0.0853 (0.0277, 0.1556) | 0.1540 (0.1047, 0.2138) | 0.8163 (0.7802, 0.8544) | 0.2509 (0.2118, 0.2968) | n/a |
| candidate_no_context | 0.0330 (0.0135, 0.0566) | 0.1865 (0.1210, 0.2649) | 0.7940 (0.7686, 0.8234) | 0.2333 (0.2056, 0.2647) | n/a |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0524 | 1.5897 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0325 | -0.1744 |
| proposed_vs_candidate_no_context | naturalness | 0.0222 | 0.0280 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0720 | 2.7143 |
| proposed_vs_candidate_no_context | context_overlap | 0.0067 | 0.1395 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0283 | -0.3725 |
| proposed_vs_candidate_no_context | persona_style | -0.0495 | -0.0788 |
| proposed_vs_candidate_no_context | distinct1 | 0.0207 | 0.0223 |
| proposed_vs_candidate_no_context | length_score | 0.0792 | 0.3551 |
| proposed_vs_candidate_no_context | sentence_score | -0.0219 | -0.0280 |
| proposed_vs_candidate_no_context | overall_quality | 0.0176 | 0.0755 |
| controlled_vs_proposed_raw | context_relevance | 0.1901 | 2.2270 |
| controlled_vs_proposed_raw | persona_consistency | 0.1408 | 0.9140 |
| controlled_vs_proposed_raw | naturalness | 0.0830 | 0.1016 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2467 | 2.5048 |
| controlled_vs_proposed_raw | context_overlap | 0.0579 | 1.0594 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1631 | 3.4250 |
| controlled_vs_proposed_raw | persona_style | 0.0514 | 0.0887 |
| controlled_vs_proposed_raw | distinct1 | -0.0072 | -0.0076 |
| controlled_vs_proposed_raw | length_score | 0.3417 | 1.1310 |
| controlled_vs_proposed_raw | sentence_score | 0.1750 | 0.2305 |
| controlled_vs_proposed_raw | overall_quality | 0.1568 | 0.6248 |
| controlled_vs_candidate_no_context | context_relevance | 0.2424 | 7.3569 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1082 | 0.5802 |
| controlled_vs_candidate_no_context | naturalness | 0.1052 | 0.1325 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3187 | 12.0179 |
| controlled_vs_candidate_no_context | context_overlap | 0.0646 | 1.3466 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1348 | 1.7765 |
| controlled_vs_candidate_no_context | persona_style | 0.0019 | 0.0030 |
| controlled_vs_candidate_no_context | distinct1 | 0.0135 | 0.0145 |
| controlled_vs_candidate_no_context | length_score | 0.4208 | 1.8879 |
| controlled_vs_candidate_no_context | sentence_score | 0.1531 | 0.1960 |
| controlled_vs_candidate_no_context | overall_quality | 0.1744 | 0.7474 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0524 | (-0.0085, 0.1299) | 0.0543 | 0.0524 | (0.0016, 0.1134) | 0.0177 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0325 | (-0.0587, -0.0073) | 0.9977 | -0.0325 | (-0.0690, -0.0116) | 1.0000 |
| proposed_vs_candidate_no_context | naturalness | 0.0222 | (-0.0110, 0.0559) | 0.0887 | 0.0222 | (-0.0290, 0.0590) | 0.1817 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0720 | (-0.0104, 0.1757) | 0.0497 | 0.0720 | (0.0000, 0.1533) | 0.0260 |
| proposed_vs_candidate_no_context | context_overlap | 0.0067 | (-0.0086, 0.0252) | 0.2177 | 0.0067 | (-0.0072, 0.0208) | 0.1943 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0283 | (-0.0580, 0.0000) | 1.0000 | -0.0283 | (-0.0714, -0.0062) | 1.0000 |
| proposed_vs_candidate_no_context | persona_style | -0.0495 | (-0.1286, 0.0162) | 0.9213 | -0.0495 | (-0.1429, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | distinct1 | 0.0207 | (0.0020, 0.0405) | 0.0150 | 0.0207 | (0.0033, 0.0326) | 0.0137 |
| proposed_vs_candidate_no_context | length_score | 0.0792 | (-0.0542, 0.2250) | 0.1213 | 0.0792 | (-0.1238, 0.2431) | 0.2357 |
| proposed_vs_candidate_no_context | sentence_score | -0.0219 | (-0.0875, 0.0437) | 0.8083 | -0.0219 | (-0.0955, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | overall_quality | 0.0176 | (-0.0137, 0.0545) | 0.1617 | 0.0176 | (-0.0173, 0.0427) | 0.1147 |
| controlled_vs_proposed_raw | context_relevance | 0.1901 | (0.1101, 0.2622) | 0.0003 | 0.1901 | (0.1371, 0.2436) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.1408 | (0.0762, 0.1925) | 0.0000 | 0.1408 | (0.0615, 0.1978) | 0.0007 |
| controlled_vs_proposed_raw | naturalness | 0.0830 | (0.0225, 0.1365) | 0.0027 | 0.0830 | (0.0065, 0.1568) | 0.0153 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2467 | (0.1321, 0.3371) | 0.0000 | 0.2467 | (0.1832, 0.3117) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0579 | (0.0314, 0.0908) | 0.0000 | 0.0579 | (0.0271, 0.0895) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1631 | (0.0985, 0.2164) | 0.0000 | 0.1631 | (0.0981, 0.2212) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.0514 | (-0.0547, 0.1578) | 0.1733 | 0.0514 | (-0.0769, 0.1655) | 0.1967 |
| controlled_vs_proposed_raw | distinct1 | -0.0072 | (-0.0311, 0.0161) | 0.7170 | -0.0072 | (-0.0288, 0.0156) | 0.7230 |
| controlled_vs_proposed_raw | length_score | 0.3417 | (0.0667, 0.5750) | 0.0083 | 0.3417 | (0.0167, 0.6456) | 0.0197 |
| controlled_vs_proposed_raw | sentence_score | 0.1750 | (0.0875, 0.2625) | 0.0000 | 0.1750 | (0.0955, 0.2423) | 0.0003 |
| controlled_vs_proposed_raw | overall_quality | 0.1568 | (0.1091, 0.1980) | 0.0000 | 0.1568 | (0.1190, 0.2034) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2424 | (0.1941, 0.2891) | 0.0000 | 0.2424 | (0.1896, 0.2970) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1082 | (0.0257, 0.1710) | 0.0050 | 0.1082 | (0.0014, 0.1728) | 0.0223 |
| controlled_vs_candidate_no_context | naturalness | 0.1052 | (0.0576, 0.1496) | 0.0000 | 0.1052 | (0.0554, 0.1521) | 0.0000 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3187 | (0.2576, 0.3759) | 0.0000 | 0.3187 | (0.2543, 0.3864) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0646 | (0.0386, 0.0951) | 0.0000 | 0.0646 | (0.0361, 0.0875) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1348 | (0.0494, 0.2027) | 0.0017 | 0.1348 | (0.0295, 0.2026) | 0.0087 |
| controlled_vs_candidate_no_context | persona_style | 0.0019 | (-0.0815, 0.0680) | 0.4547 | 0.0019 | (-0.1042, 0.0663) | 0.4610 |
| controlled_vs_candidate_no_context | distinct1 | 0.0135 | (-0.0101, 0.0356) | 0.1343 | 0.0135 | (-0.0088, 0.0332) | 0.1127 |
| controlled_vs_candidate_no_context | length_score | 0.4208 | (0.2292, 0.5917) | 0.0000 | 0.4208 | (0.2290, 0.6104) | 0.0000 |
| controlled_vs_candidate_no_context | sentence_score | 0.1531 | (0.0656, 0.2406) | 0.0000 | 0.1531 | (0.0583, 0.2125) | 0.0040 |
| controlled_vs_candidate_no_context | overall_quality | 0.1744 | (0.1409, 0.2058) | 0.0000 | 0.1744 | (0.1371, 0.1992) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 7 | 4 | 5 | 0.5938 | 0.6364 |
| proposed_vs_candidate_no_context | persona_consistency | 1 | 6 | 9 | 0.3438 | 0.1429 |
| proposed_vs_candidate_no_context | naturalness | 7 | 4 | 5 | 0.5938 | 0.6364 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 4 | 2 | 10 | 0.5625 | 0.6667 |
| proposed_vs_candidate_no_context | context_overlap | 6 | 5 | 5 | 0.5312 | 0.5455 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0 | 3 | 13 | 0.4062 | 0.0000 |
| proposed_vs_candidate_no_context | persona_style | 1 | 3 | 12 | 0.4375 | 0.2500 |
| proposed_vs_candidate_no_context | distinct1 | 7 | 3 | 6 | 0.6250 | 0.7000 |
| proposed_vs_candidate_no_context | length_score | 6 | 4 | 6 | 0.5625 | 0.6000 |
| proposed_vs_candidate_no_context | sentence_score | 1 | 2 | 13 | 0.4688 | 0.3333 |
| proposed_vs_candidate_no_context | overall_quality | 4 | 7 | 5 | 0.4062 | 0.3636 |
| controlled_vs_proposed_raw | context_relevance | 15 | 1 | 0 | 0.9375 | 0.9375 |
| controlled_vs_proposed_raw | persona_consistency | 13 | 1 | 2 | 0.8750 | 0.9286 |
| controlled_vs_proposed_raw | naturalness | 13 | 3 | 0 | 0.8125 | 0.8125 |
| controlled_vs_proposed_raw | context_keyword_coverage | 14 | 1 | 1 | 0.9062 | 0.9333 |
| controlled_vs_proposed_raw | context_overlap | 15 | 1 | 0 | 0.9375 | 0.9375 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 13 | 1 | 2 | 0.8750 | 0.9286 |
| controlled_vs_proposed_raw | persona_style | 4 | 2 | 10 | 0.5625 | 0.6667 |
| controlled_vs_proposed_raw | distinct1 | 7 | 7 | 2 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | length_score | 13 | 2 | 1 | 0.8438 | 0.8667 |
| controlled_vs_proposed_raw | sentence_score | 8 | 0 | 8 | 0.7500 | 1.0000 |
| controlled_vs_proposed_raw | overall_quality | 15 | 1 | 0 | 0.9375 | 0.9375 |
| controlled_vs_candidate_no_context | context_relevance | 15 | 1 | 0 | 0.9375 | 0.9375 |
| controlled_vs_candidate_no_context | persona_consistency | 13 | 2 | 1 | 0.8438 | 0.8667 |
| controlled_vs_candidate_no_context | naturalness | 13 | 3 | 0 | 0.8125 | 0.8125 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 15 | 0 | 1 | 0.9688 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 14 | 1 | 1 | 0.9062 | 0.9333 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 13 | 2 | 1 | 0.8438 | 0.8667 |
| controlled_vs_candidate_no_context | persona_style | 3 | 2 | 11 | 0.5312 | 0.6000 |
| controlled_vs_candidate_no_context | distinct1 | 10 | 5 | 1 | 0.6562 | 0.6667 |
| controlled_vs_candidate_no_context | length_score | 13 | 2 | 1 | 0.8438 | 0.8667 |
| controlled_vs_candidate_no_context | sentence_score | 7 | 0 | 9 | 0.7188 | 1.0000 |
| controlled_vs_candidate_no_context | overall_quality | 16 | 0 | 0 | 1.0000 | 1.0000 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.0625 | 0.4375 | 0.1875 |
| proposed_contextual | 0.0000 | 0.0000 | 0.3750 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `8`
- Unique template signatures: `15`
- Template signature ratio: `0.9375`
- Effective sample size by source clustering: `5.57`
- Effective sample size by template-signature clustering: `14.22`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: unavailable (disabled_by_flag).

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
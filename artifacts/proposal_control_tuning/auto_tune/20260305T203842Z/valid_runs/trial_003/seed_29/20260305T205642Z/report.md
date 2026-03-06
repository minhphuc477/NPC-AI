# Proposal Alignment Evaluation Report

- Run ID: `20260305T205642Z`
- Generated: `2026-03-05T21:00:36.009527+00:00`
- Scenarios: `artifacts\proposal_control_tuning\auto_tune\20260305T203842Z\valid_runs\trial_003\seed_29\20260305T205642Z\scenarios.jsonl`
- Scenario count: `20`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_tuned`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2636 (0.2138, 0.3145) | 0.4065 (0.3573, 0.4604) | 0.8966 (0.8638, 0.9273) | 0.4437 (0.4198, 0.4687) | n/a |
| proposed_contextual_controlled_tuned | 0.2879 (0.2319, 0.3468) | 0.4198 (0.3682, 0.4774) | 0.8780 (0.8441, 0.9067) | 0.4562 (0.4329, 0.4809) | n/a |
| proposed_contextual | 0.1067 (0.0669, 0.1563) | 0.2705 (0.1935, 0.3563) | 0.8622 (0.8293, 0.8957) | 0.3143 (0.2688, 0.3628) | n/a |
| candidate_no_context | 0.0373 (0.0183, 0.0617) | 0.2592 (0.1757, 0.3494) | 0.8553 (0.8263, 0.8825) | 0.2765 (0.2423, 0.3131) | n/a |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0694 | 1.8609 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0114 | 0.0438 |
| proposed_vs_candidate_no_context | naturalness | 0.0069 | 0.0081 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0883 | 2.6180 |
| proposed_vs_candidate_no_context | context_overlap | 0.0254 | 0.5559 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0167 | 0.0864 |
| proposed_vs_candidate_no_context | persona_style | -0.0099 | -0.0188 |
| proposed_vs_candidate_no_context | distinct1 | -0.0048 | -0.0051 |
| proposed_vs_candidate_no_context | length_score | 0.0267 | 0.0613 |
| proposed_vs_candidate_no_context | sentence_score | 0.0350 | 0.0407 |
| proposed_vs_candidate_no_context | overall_quality | 0.0378 | 0.1368 |
| controlled_vs_proposed_raw | context_relevance | 0.1570 | 1.4713 |
| controlled_vs_proposed_raw | persona_consistency | 0.1360 | 0.5026 |
| controlled_vs_proposed_raw | naturalness | 0.0344 | 0.0399 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2121 | 1.7391 |
| controlled_vs_proposed_raw | context_overlap | 0.0283 | 0.3980 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1369 | 0.6534 |
| controlled_vs_proposed_raw | persona_style | 0.1322 | 0.2569 |
| controlled_vs_proposed_raw | distinct1 | -0.0200 | -0.0211 |
| controlled_vs_proposed_raw | length_score | 0.1683 | 0.3646 |
| controlled_vs_proposed_raw | sentence_score | 0.0875 | 0.0978 |
| controlled_vs_proposed_raw | overall_quality | 0.1294 | 0.4115 |
| controlled_vs_candidate_no_context | context_relevance | 0.2264 | 6.0702 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1473 | 0.5685 |
| controlled_vs_candidate_no_context | naturalness | 0.0413 | 0.0483 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3004 | 8.9101 |
| controlled_vs_candidate_no_context | context_overlap | 0.0536 | 1.1750 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1536 | 0.7963 |
| controlled_vs_candidate_no_context | persona_style | 0.1223 | 0.2333 |
| controlled_vs_candidate_no_context | distinct1 | -0.0249 | -0.0260 |
| controlled_vs_candidate_no_context | length_score | 0.1950 | 0.4483 |
| controlled_vs_candidate_no_context | sentence_score | 0.1225 | 0.1424 |
| controlled_vs_candidate_no_context | overall_quality | 0.1672 | 0.6047 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0243 | 0.0921 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0134 | 0.0329 |
| controlled_alt_vs_controlled_default | naturalness | -0.0186 | -0.0208 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0306 | 0.0916 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0095 | 0.0960 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0167 | 0.0481 |
| controlled_alt_vs_controlled_default | persona_style | 0.0002 | 0.0003 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0053 | -0.0057 |
| controlled_alt_vs_controlled_default | length_score | -0.0817 | -0.1296 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0175 | 0.0178 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0125 | 0.0281 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1812 | 1.6990 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1493 | 0.5520 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0158 | 0.0183 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2427 | 1.9901 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0378 | 0.5322 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1536 | 0.7330 |
| controlled_alt_vs_proposed_raw | persona_style | 0.1324 | 0.2572 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0253 | -0.0266 |
| controlled_alt_vs_proposed_raw | length_score | 0.0867 | 0.1877 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1050 | 0.1173 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1418 | 0.4512 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2506 | 6.7214 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1607 | 0.6200 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0227 | 0.0265 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3310 | 9.8180 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0632 | 1.3839 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1702 | 0.8827 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.1225 | 0.2336 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0302 | -0.0316 |
| controlled_alt_vs_candidate_no_context | length_score | 0.1133 | 0.2605 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1400 | 0.1628 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1797 | 0.6498 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0694 | (0.0263, 0.1203) | 0.0000 | 0.0694 | (0.0026, 0.1160) | 0.0000 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0114 | (-0.1059, 0.1281) | 0.4297 | 0.0114 | (-0.0328, 0.0323) | 0.2680 |
| proposed_vs_candidate_no_context | naturalness | 0.0069 | (-0.0282, 0.0386) | 0.3513 | 0.0069 | (-0.0545, 0.0260) | 0.2563 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0883 | (0.0333, 0.1557) | 0.0000 | 0.0883 | (0.0000, 0.1477) | 0.0410 |
| proposed_vs_candidate_no_context | context_overlap | 0.0254 | (0.0111, 0.0406) | 0.0000 | 0.0254 | (0.0086, 0.0420) | 0.0000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0167 | (-0.1143, 0.1571) | 0.4043 | 0.0167 | (-0.0000, 0.0417) | 0.2833 |
| proposed_vs_candidate_no_context | persona_style | -0.0099 | (-0.1080, 0.0785) | 0.5570 | -0.0099 | (-0.1641, 0.0625) | 0.6333 |
| proposed_vs_candidate_no_context | distinct1 | -0.0048 | (-0.0211, 0.0104) | 0.7157 | -0.0048 | (-0.0227, 0.0041) | 0.8437 |
| proposed_vs_candidate_no_context | length_score | 0.0267 | (-0.0900, 0.1517) | 0.3013 | 0.0267 | (-0.1833, 0.0958) | 0.2577 |
| proposed_vs_candidate_no_context | sentence_score | 0.0350 | (-0.0525, 0.1225) | 0.2643 | 0.0350 | (-0.0875, 0.0875) | 0.2653 |
| proposed_vs_candidate_no_context | overall_quality | 0.0378 | (-0.0163, 0.0952) | 0.0770 | 0.0378 | (-0.0198, 0.0610) | 0.0327 |
| controlled_vs_proposed_raw | context_relevance | 0.1570 | (0.0803, 0.2290) | 0.0000 | 0.1570 | (0.1023, 0.2420) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.1360 | (0.0425, 0.2195) | 0.0050 | 0.1360 | (0.0509, 0.2646) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0344 | (-0.0221, 0.0925) | 0.1280 | 0.0344 | (-0.0404, 0.2193) | 0.2567 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2121 | (0.1186, 0.3055) | 0.0000 | 0.2121 | (0.1364, 0.3295) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0283 | (0.0083, 0.0487) | 0.0017 | 0.0283 | (0.0227, 0.0378) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1369 | (0.0393, 0.2286) | 0.0027 | 0.1369 | (0.0714, 0.1875) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.1322 | (0.0130, 0.2666) | 0.0150 | 0.1322 | (-0.0312, 0.6562) | 0.1433 |
| controlled_vs_proposed_raw | distinct1 | -0.0200 | (-0.0430, 0.0039) | 0.9460 | -0.0200 | (-0.0515, 0.0357) | 0.8470 |
| controlled_vs_proposed_raw | length_score | 0.1683 | (-0.0483, 0.3967) | 0.0573 | 0.1683 | (-0.1208, 0.8500) | 0.1393 |
| controlled_vs_proposed_raw | sentence_score | 0.0875 | (0.0000, 0.1750) | 0.0310 | 0.0875 | (0.0000, 0.3500) | 0.0433 |
| controlled_vs_proposed_raw | overall_quality | 0.1294 | (0.0776, 0.1844) | 0.0000 | 0.1294 | (0.0706, 0.2467) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2264 | (0.1736, 0.2825) | 0.0000 | 0.2264 | (0.2183, 0.2446) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1473 | (0.0776, 0.2147) | 0.0000 | 0.1473 | (0.0634, 0.2318) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.0413 | (-0.0004, 0.0865) | 0.0267 | 0.0413 | (-0.0144, 0.1648) | 0.0380 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3004 | (0.2291, 0.3714) | 0.0000 | 0.3004 | (0.2841, 0.3295) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0536 | (0.0361, 0.0723) | 0.0000 | 0.0536 | (0.0462, 0.0647) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1536 | (0.0726, 0.2321) | 0.0000 | 0.1536 | (0.0714, 0.2292) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.1223 | (0.0261, 0.2377) | 0.0040 | 0.1223 | (0.0284, 0.4922) | 0.0000 |
| controlled_vs_candidate_no_context | distinct1 | -0.0249 | (-0.0471, -0.0016) | 0.9797 | -0.0249 | (-0.0563, 0.0130) | 0.8447 |
| controlled_vs_candidate_no_context | length_score | 0.1950 | (0.0183, 0.3667) | 0.0157 | 0.1950 | (-0.0250, 0.6667) | 0.0340 |
| controlled_vs_candidate_no_context | sentence_score | 0.1225 | (0.0350, 0.2100) | 0.0057 | 0.1225 | (0.0437, 0.2625) | 0.0000 |
| controlled_vs_candidate_no_context | overall_quality | 0.1672 | (0.1287, 0.2034) | 0.0000 | 0.1672 | (0.1317, 0.2269) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0243 | (-0.0414, 0.0840) | 0.2300 | 0.0243 | (-0.0185, 0.0661) | 0.1570 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0134 | (-0.0446, 0.0755) | 0.3383 | 0.0134 | (0.0000, 0.0333) | 0.0000 |
| controlled_alt_vs_controlled_default | naturalness | -0.0186 | (-0.0524, 0.0144) | 0.8717 | -0.0186 | (-0.0571, -0.0034) | 1.0000 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0306 | (-0.0557, 0.1070) | 0.2327 | 0.0306 | (-0.0227, 0.0833) | 0.1383 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0095 | (-0.0100, 0.0259) | 0.1503 | 0.0095 | (-0.0086, 0.0257) | 0.1447 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0167 | (-0.0440, 0.0857) | 0.2843 | 0.0167 | (0.0000, 0.0417) | 0.0000 |
| controlled_alt_vs_controlled_default | persona_style | 0.0002 | (-0.0498, 0.0516) | 0.5033 | 0.0002 | (0.0000, 0.0005) | 0.2933 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0053 | (-0.0279, 0.0160) | 0.6683 | -0.0053 | (-0.0143, 0.0037) | 0.8470 |
| controlled_alt_vs_controlled_default | length_score | -0.0817 | (-0.2217, 0.0500) | 0.8830 | -0.0817 | (-0.2750, -0.0208) | 1.0000 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0175 | (0.0000, 0.0525) | 0.3560 | 0.0175 | (0.0000, 0.0437) | 0.2923 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0125 | (-0.0180, 0.0382) | 0.1847 | 0.0125 | (-0.0114, 0.0360) | 0.1433 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1812 | (0.1123, 0.2538) | 0.0000 | 0.1812 | (0.0838, 0.2683) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1493 | (0.0893, 0.2020) | 0.0000 | 0.1493 | (0.0509, 0.2979) | 0.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0158 | (-0.0335, 0.0676) | 0.2667 | 0.0158 | (-0.0438, 0.1622) | 0.3647 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2427 | (0.1467, 0.3381) | 0.0000 | 0.2427 | (0.1136, 0.3614) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0378 | (0.0201, 0.0557) | 0.0000 | 0.0378 | (0.0141, 0.0548) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1536 | (0.0893, 0.2083) | 0.0000 | 0.1536 | (0.0714, 0.2083) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 0.1324 | (0.0241, 0.2594) | 0.0057 | 0.1324 | (-0.0312, 0.6562) | 0.1510 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0253 | (-0.0466, -0.0066) | 0.9953 | -0.0253 | (-0.0478, 0.0305) | 0.9643 |
| controlled_alt_vs_proposed_raw | length_score | 0.0867 | (-0.1017, 0.2817) | 0.1753 | 0.0867 | (-0.1417, 0.5750) | 0.1483 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1050 | (0.0350, 0.1750) | 0.0007 | 0.1050 | (0.0437, 0.3500) | 0.0000 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1418 | (0.0988, 0.1855) | 0.0000 | 0.1418 | (0.0592, 0.2598) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2506 | (0.1912, 0.3157) | 0.0000 | 0.2506 | (0.1998, 0.2914) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1607 | (0.0466, 0.2688) | 0.0020 | 0.1607 | (0.0634, 0.2651) | 0.0000 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0227 | (-0.0242, 0.0688) | 0.1580 | 0.0227 | (-0.0178, 0.1077) | 0.1463 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3310 | (0.2513, 0.4167) | 0.0000 | 0.3310 | (0.2614, 0.3854) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0632 | (0.0470, 0.0797) | 0.0000 | 0.0632 | (0.0561, 0.0719) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1702 | (0.0488, 0.2893) | 0.0027 | 0.1702 | (0.0714, 0.2500) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.1225 | (0.0116, 0.2420) | 0.0140 | 0.1225 | (0.0289, 0.4922) | 0.0000 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0302 | (-0.0509, -0.0106) | 0.9987 | -0.0302 | (-0.0526, 0.0078) | 0.9647 |
| controlled_alt_vs_candidate_no_context | length_score | 0.1133 | (-0.0617, 0.2867) | 0.1023 | 0.1133 | (-0.0458, 0.3917) | 0.0433 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1400 | (0.0700, 0.2100) | 0.0000 | 0.1400 | (0.0875, 0.2625) | 0.0000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1797 | (0.1334, 0.2210) | 0.0000 | 0.1797 | (0.1203, 0.2400) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 14 | 3 | 3 | 0.7750 | 0.8235 |
| proposed_vs_candidate_no_context | persona_consistency | 5 | 8 | 7 | 0.4250 | 0.3846 |
| proposed_vs_candidate_no_context | naturalness | 12 | 5 | 3 | 0.6750 | 0.7059 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 10 | 2 | 8 | 0.7000 | 0.8333 |
| proposed_vs_candidate_no_context | context_overlap | 14 | 3 | 3 | 0.7750 | 0.8235 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 4 | 4 | 12 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | persona_style | 5 | 6 | 9 | 0.4750 | 0.4545 |
| proposed_vs_candidate_no_context | distinct1 | 6 | 9 | 5 | 0.4250 | 0.4000 |
| proposed_vs_candidate_no_context | length_score | 11 | 6 | 3 | 0.6250 | 0.6471 |
| proposed_vs_candidate_no_context | sentence_score | 4 | 2 | 14 | 0.5500 | 0.6667 |
| proposed_vs_candidate_no_context | overall_quality | 11 | 6 | 3 | 0.6250 | 0.6471 |
| controlled_vs_proposed_raw | context_relevance | 17 | 3 | 0 | 0.8500 | 0.8500 |
| controlled_vs_proposed_raw | persona_consistency | 16 | 3 | 1 | 0.8250 | 0.8421 |
| controlled_vs_proposed_raw | naturalness | 12 | 8 | 0 | 0.6000 | 0.6000 |
| controlled_vs_proposed_raw | context_keyword_coverage | 17 | 3 | 0 | 0.8500 | 0.8500 |
| controlled_vs_proposed_raw | context_overlap | 14 | 6 | 0 | 0.7000 | 0.7000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 16 | 3 | 1 | 0.8250 | 0.8421 |
| controlled_vs_proposed_raw | persona_style | 9 | 4 | 7 | 0.6250 | 0.6923 |
| controlled_vs_proposed_raw | distinct1 | 8 | 12 | 0 | 0.4000 | 0.4000 |
| controlled_vs_proposed_raw | length_score | 11 | 7 | 2 | 0.6000 | 0.6111 |
| controlled_vs_proposed_raw | sentence_score | 6 | 1 | 13 | 0.6250 | 0.8571 |
| controlled_vs_proposed_raw | overall_quality | 17 | 3 | 0 | 0.8500 | 0.8500 |
| controlled_vs_candidate_no_context | context_relevance | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_vs_candidate_no_context | persona_consistency | 17 | 3 | 0 | 0.8500 | 0.8500 |
| controlled_vs_candidate_no_context | naturalness | 15 | 5 | 0 | 0.7500 | 0.7500 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 18 | 0 | 2 | 0.9500 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 18 | 2 | 0 | 0.9000 | 0.9000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 16 | 3 | 1 | 0.8250 | 0.8421 |
| controlled_vs_candidate_no_context | persona_style | 5 | 1 | 14 | 0.6000 | 0.8333 |
| controlled_vs_candidate_no_context | distinct1 | 6 | 14 | 0 | 0.3000 | 0.3000 |
| controlled_vs_candidate_no_context | length_score | 14 | 5 | 1 | 0.7250 | 0.7368 |
| controlled_vs_candidate_no_context | sentence_score | 8 | 1 | 11 | 0.6750 | 0.8889 |
| controlled_vs_candidate_no_context | overall_quality | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_alt_vs_controlled_default | context_relevance | 10 | 5 | 5 | 0.6250 | 0.6667 |
| controlled_alt_vs_controlled_default | persona_consistency | 6 | 5 | 9 | 0.5250 | 0.5455 |
| controlled_alt_vs_controlled_default | naturalness | 7 | 8 | 5 | 0.4750 | 0.4667 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 10 | 4 | 6 | 0.6500 | 0.7143 |
| controlled_alt_vs_controlled_default | context_overlap | 11 | 4 | 5 | 0.6750 | 0.7333 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 5 | 4 | 11 | 0.5250 | 0.5556 |
| controlled_alt_vs_controlled_default | persona_style | 3 | 3 | 14 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | distinct1 | 8 | 7 | 5 | 0.5250 | 0.5333 |
| controlled_alt_vs_controlled_default | length_score | 5 | 10 | 5 | 0.3750 | 0.3333 |
| controlled_alt_vs_controlled_default | sentence_score | 1 | 0 | 19 | 0.5250 | 1.0000 |
| controlled_alt_vs_controlled_default | overall_quality | 11 | 4 | 5 | 0.6750 | 0.7333 |
| controlled_alt_vs_proposed_raw | context_relevance | 15 | 5 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_proposed_raw | persona_consistency | 17 | 1 | 2 | 0.9000 | 0.9444 |
| controlled_alt_vs_proposed_raw | naturalness | 12 | 8 | 0 | 0.6000 | 0.6000 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 15 | 1 | 4 | 0.8500 | 0.9375 |
| controlled_alt_vs_proposed_raw | context_overlap | 15 | 5 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 17 | 1 | 2 | 0.9000 | 0.9444 |
| controlled_alt_vs_proposed_raw | persona_style | 7 | 2 | 11 | 0.6250 | 0.7778 |
| controlled_alt_vs_proposed_raw | distinct1 | 6 | 14 | 0 | 0.3000 | 0.3000 |
| controlled_alt_vs_proposed_raw | length_score | 13 | 7 | 0 | 0.6500 | 0.6500 |
| controlled_alt_vs_proposed_raw | sentence_score | 6 | 0 | 14 | 0.6500 | 1.0000 |
| controlled_alt_vs_proposed_raw | overall_quality | 17 | 3 | 0 | 0.8500 | 0.8500 |
| controlled_alt_vs_candidate_no_context | context_relevance | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 16 | 3 | 1 | 0.8250 | 0.8421 |
| controlled_alt_vs_candidate_no_context | naturalness | 11 | 9 | 0 | 0.5500 | 0.5500 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 15 | 3 | 2 | 0.8000 | 0.8333 |
| controlled_alt_vs_candidate_no_context | persona_style | 7 | 4 | 9 | 0.5750 | 0.6364 |
| controlled_alt_vs_candidate_no_context | distinct1 | 5 | 15 | 0 | 0.2500 | 0.2500 |
| controlled_alt_vs_candidate_no_context | length_score | 10 | 7 | 3 | 0.5750 | 0.5882 |
| controlled_alt_vs_candidate_no_context | sentence_score | 8 | 0 | 12 | 0.7000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 18 | 2 | 0 | 0.9000 | 0.9000 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.3500 | 0.5000 | 0.5000 |
| proposed_contextual_controlled_tuned | 0.0000 | 0.0000 | 0.4000 | 0.4500 | 0.5500 |
| proposed_contextual | 0.0000 | 0.0000 | 0.2500 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.2500 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `3`
- Unique template signatures: `14`
- Template signature ratio: `0.7000`
- Effective sample size by source clustering: `2.78`
- Effective sample size by template-signature clustering: `10.53`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: unavailable (disabled_by_flag).

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
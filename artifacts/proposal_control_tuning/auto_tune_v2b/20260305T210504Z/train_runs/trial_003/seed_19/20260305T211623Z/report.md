# Proposal Alignment Evaluation Report

- Run ID: `20260305T211623Z`
- Generated: `2026-03-05T21:20:53.294808+00:00`
- Scenarios: `artifacts\proposal_control_tuning\auto_tune_v2b\20260305T210504Z\train_runs\trial_003\seed_19\20260305T211623Z\scenarios.jsonl`
- Scenario count: `20`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_tuned`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2883 (0.2445, 0.3377) | 0.3953 (0.3051, 0.4925) | 0.8718 (0.8404, 0.9020) | 0.4467 (0.4176, 0.4784) | n/a |
| proposed_contextual_controlled_tuned | 0.2528 (0.2359, 0.2718) | 0.3767 (0.2902, 0.4719) | 0.9050 (0.8823, 0.9257) | 0.4298 (0.3985, 0.4635) | n/a |
| proposed_contextual | 0.0732 (0.0300, 0.1248) | 0.1455 (0.1040, 0.1888) | 0.8229 (0.7880, 0.8572) | 0.2434 (0.2091, 0.2782) | n/a |
| candidate_no_context | 0.0389 (0.0235, 0.0547) | 0.1636 (0.1050, 0.2257) | 0.8130 (0.7823, 0.8469) | 0.2326 (0.2037, 0.2653) | n/a |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0343 | 0.8822 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0181 | -0.1106 |
| proposed_vs_candidate_no_context | naturalness | 0.0099 | 0.0122 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0409 | 1.0000 |
| proposed_vs_candidate_no_context | context_overlap | 0.0189 | 0.5529 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0226 | -0.3725 |
| proposed_vs_candidate_no_context | persona_style | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | distinct1 | 0.0004 | 0.0004 |
| proposed_vs_candidate_no_context | length_score | 0.0400 | 0.1257 |
| proposed_vs_candidate_no_context | sentence_score | 0.0175 | 0.0243 |
| proposed_vs_candidate_no_context | overall_quality | 0.0108 | 0.0465 |
| controlled_vs_proposed_raw | context_relevance | 0.2151 | 2.9396 |
| controlled_vs_proposed_raw | persona_consistency | 0.2498 | 1.7173 |
| controlled_vs_proposed_raw | naturalness | 0.0488 | 0.0593 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2864 | 3.5000 |
| controlled_vs_proposed_raw | context_overlap | 0.0489 | 0.9219 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.3040 | 7.9813 |
| controlled_vs_proposed_raw | persona_style | 0.0329 | 0.0572 |
| controlled_vs_proposed_raw | distinct1 | -0.0174 | -0.0185 |
| controlled_vs_proposed_raw | length_score | 0.1667 | 0.4651 |
| controlled_vs_proposed_raw | sentence_score | 0.2275 | 0.3085 |
| controlled_vs_proposed_raw | overall_quality | 0.2033 | 0.8353 |
| controlled_vs_candidate_no_context | context_relevance | 0.2494 | 6.4152 |
| controlled_vs_candidate_no_context | persona_consistency | 0.2317 | 1.4167 |
| controlled_vs_candidate_no_context | naturalness | 0.0587 | 0.0723 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3273 | 8.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0678 | 1.9847 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2814 | 4.6353 |
| controlled_vs_candidate_no_context | persona_style | 0.0329 | 0.0572 |
| controlled_vs_candidate_no_context | distinct1 | -0.0170 | -0.0180 |
| controlled_vs_candidate_no_context | length_score | 0.2067 | 0.6492 |
| controlled_vs_candidate_no_context | sentence_score | 0.2450 | 0.3403 |
| controlled_vs_candidate_no_context | overall_quality | 0.2141 | 0.9206 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0355 | -0.1231 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0186 | -0.0471 |
| controlled_alt_vs_controlled_default | naturalness | 0.0332 | 0.0381 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0500 | -0.1358 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0017 | -0.0164 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0193 | -0.0564 |
| controlled_alt_vs_controlled_default | persona_style | -0.0160 | -0.0264 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0112 | 0.0121 |
| controlled_alt_vs_controlled_default | length_score | 0.1500 | 0.2857 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0000 | 0.0000 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0169 | -0.0378 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1796 | 2.4545 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.2312 | 1.5892 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0821 | 0.0997 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2364 | 2.8889 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0472 | 0.8904 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.2848 | 7.4750 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0169 | 0.0293 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0062 | -0.0066 |
| controlled_alt_vs_proposed_raw | length_score | 0.3167 | 0.8837 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.2275 | 0.3085 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1864 | 0.7659 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2139 | 5.5020 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.2131 | 1.3027 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0920 | 0.1131 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.2773 | 6.7778 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0661 | 1.9357 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.2621 | 4.3176 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0169 | 0.0293 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0058 | -0.0062 |
| controlled_alt_vs_candidate_no_context | length_score | 0.3567 | 1.1204 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.2450 | 0.3403 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1972 | 0.8480 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0343 | (-0.0096, 0.0873) | 0.0747 | 0.0343 | (0.0060, 0.0638) | 0.0117 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0181 | (-0.0590, 0.0133) | 0.8627 | -0.0181 | (-0.0444, 0.0157) | 0.8723 |
| proposed_vs_candidate_no_context | naturalness | 0.0099 | (-0.0250, 0.0462) | 0.2707 | 0.0099 | (-0.0061, 0.0318) | 0.2067 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0409 | (-0.0227, 0.1136) | 0.1273 | 0.0409 | (0.0051, 0.0861) | 0.0077 |
| proposed_vs_candidate_no_context | context_overlap | 0.0189 | (0.0055, 0.0338) | 0.0003 | 0.0189 | (0.0058, 0.0276) | 0.0117 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0226 | (-0.0726, 0.0167) | 0.8547 | -0.0226 | (-0.0586, 0.0196) | 0.8657 |
| proposed_vs_candidate_no_context | persona_style | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | distinct1 | 0.0004 | (-0.0120, 0.0128) | 0.4643 | 0.0004 | (-0.0076, 0.0122) | 0.4793 |
| proposed_vs_candidate_no_context | length_score | 0.0400 | (-0.1034, 0.1817) | 0.2807 | 0.0400 | (-0.0275, 0.0905) | 0.0980 |
| proposed_vs_candidate_no_context | sentence_score | 0.0175 | (-0.0700, 0.1050) | 0.4213 | 0.0175 | (-0.0420, 0.1105) | 0.4440 |
| proposed_vs_candidate_no_context | overall_quality | 0.0108 | (-0.0222, 0.0452) | 0.2727 | 0.0108 | (-0.0030, 0.0342) | 0.1380 |
| controlled_vs_proposed_raw | context_relevance | 0.2151 | (0.1573, 0.2722) | 0.0000 | 0.2151 | (0.1984, 0.2322) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.2498 | (0.1565, 0.3541) | 0.0000 | 0.2498 | (0.1421, 0.4213) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0488 | (0.0028, 0.0931) | 0.0193 | 0.0488 | (0.0108, 0.0842) | 0.0043 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2864 | (0.2091, 0.3636) | 0.0000 | 0.2864 | (0.2662, 0.3081) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0489 | (0.0310, 0.0681) | 0.0000 | 0.0489 | (0.0376, 0.0602) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.3040 | (0.1848, 0.4364) | 0.0000 | 0.3040 | (0.1754, 0.5167) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.0329 | (-0.0088, 0.0873) | 0.0820 | 0.0329 | (-0.0202, 0.1471) | 0.3293 |
| controlled_vs_proposed_raw | distinct1 | -0.0174 | (-0.0388, 0.0029) | 0.9510 | -0.0174 | (-0.0375, 0.0015) | 0.9597 |
| controlled_vs_proposed_raw | length_score | 0.1667 | (-0.0050, 0.3333) | 0.0287 | 0.1667 | (0.0397, 0.2963) | 0.0050 |
| controlled_vs_proposed_raw | sentence_score | 0.2275 | (0.1400, 0.3150) | 0.0000 | 0.2275 | (0.1474, 0.3000) | 0.0000 |
| controlled_vs_proposed_raw | overall_quality | 0.2033 | (0.1603, 0.2471) | 0.0000 | 0.2033 | (0.1613, 0.2719) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2494 | (0.2078, 0.2962) | 0.0000 | 0.2494 | (0.2115, 0.2953) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.2317 | (0.1317, 0.3437) | 0.0000 | 0.2317 | (0.1231, 0.4086) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.0587 | (0.0125, 0.0998) | 0.0047 | 0.0587 | (0.0128, 0.0952) | 0.0080 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3273 | (0.2682, 0.3865) | 0.0000 | 0.3273 | (0.2778, 0.3909) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0678 | (0.0512, 0.0885) | 0.0000 | 0.0678 | (0.0581, 0.0799) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2814 | (0.1469, 0.4229) | 0.0000 | 0.2814 | (0.1558, 0.4958) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.0329 | (-0.0092, 0.0834) | 0.0880 | 0.0329 | (-0.0210, 0.1471) | 0.3403 |
| controlled_vs_candidate_no_context | distinct1 | -0.0170 | (-0.0391, 0.0057) | 0.9297 | -0.0170 | (-0.0382, 0.0067) | 0.9170 |
| controlled_vs_candidate_no_context | length_score | 0.2067 | (0.0383, 0.3667) | 0.0113 | 0.2067 | (0.0211, 0.3316) | 0.0100 |
| controlled_vs_candidate_no_context | sentence_score | 0.2450 | (0.1750, 0.3150) | 0.0000 | 0.2450 | (0.2026, 0.2947) | 0.0000 |
| controlled_vs_candidate_no_context | overall_quality | 0.2141 | (0.1781, 0.2499) | 0.0000 | 0.2141 | (0.1791, 0.2689) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0355 | (-0.0852, 0.0090) | 0.9383 | -0.0355 | (-0.0586, -0.0151) | 0.9990 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0186 | (-0.1109, 0.0663) | 0.6440 | -0.0186 | (-0.1600, 0.0756) | 0.5697 |
| controlled_alt_vs_controlled_default | naturalness | 0.0332 | (-0.0070, 0.0707) | 0.0493 | 0.0332 | (0.0096, 0.0660) | 0.0027 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0500 | (-0.1136, 0.0091) | 0.9507 | -0.0500 | (-0.0823, -0.0214) | 1.0000 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0017 | (-0.0229, 0.0198) | 0.5453 | -0.0017 | (-0.0037, 0.0012) | 0.8920 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0193 | (-0.1395, 0.0881) | 0.6180 | -0.0193 | (-0.1922, 0.0942) | 0.5820 |
| controlled_alt_vs_controlled_default | persona_style | -0.0160 | (-0.0456, 0.0131) | 0.8300 | -0.0160 | (-0.0588, 0.0014) | 0.7467 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0112 | (-0.0046, 0.0256) | 0.0803 | 0.0112 | (-0.0015, 0.0285) | 0.0353 |
| controlled_alt_vs_controlled_default | length_score | 0.1500 | (-0.0250, 0.3150) | 0.0443 | 0.1500 | (0.0579, 0.3023) | 0.0000 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0000 | (-0.0700, 0.0700) | 0.6057 | 0.0000 | (-0.0500, 0.0583) | 0.6447 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0169 | (-0.0535, 0.0193) | 0.8227 | -0.0169 | (-0.0662, 0.0121) | 0.8223 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1796 | (0.1261, 0.2258) | 0.0000 | 0.1796 | (0.1626, 0.2046) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.2312 | (0.1562, 0.3185) | 0.0000 | 0.2312 | (0.1681, 0.2857) | 0.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0821 | (0.0376, 0.1249) | 0.0003 | 0.0821 | (0.0365, 0.1458) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2364 | (0.1682, 0.3000) | 0.0000 | 0.2364 | (0.2159, 0.2670) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0472 | (0.0286, 0.0651) | 0.0000 | 0.0472 | (0.0344, 0.0621) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.2848 | (0.1905, 0.3995) | 0.0000 | 0.2848 | (0.1963, 0.3702) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0169 | (-0.0162, 0.0585) | 0.1837 | 0.0169 | (-0.0187, 0.0882) | 0.3377 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0062 | (-0.0311, 0.0157) | 0.6867 | -0.0062 | (-0.0205, 0.0168) | 0.7433 |
| controlled_alt_vs_proposed_raw | length_score | 0.3167 | (0.1650, 0.4617) | 0.0000 | 0.3167 | (0.1530, 0.5444) | 0.0000 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.2275 | (0.1400, 0.3150) | 0.0000 | 0.2275 | (0.1050, 0.3306) | 0.0000 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1864 | (0.1409, 0.2307) | 0.0000 | 0.1864 | (0.1539, 0.2136) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2139 | (0.1903, 0.2363) | 0.0000 | 0.2139 | (0.1896, 0.2408) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.2131 | (0.1365, 0.3012) | 0.0000 | 0.2131 | (0.1568, 0.2640) | 0.0000 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0920 | (0.0508, 0.1327) | 0.0000 | 0.0920 | (0.0451, 0.1494) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.2773 | (0.2455, 0.3091) | 0.0000 | 0.2773 | (0.2468, 0.3110) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0661 | (0.0541, 0.0792) | 0.0000 | 0.0661 | (0.0558, 0.0777) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.2621 | (0.1678, 0.3693) | 0.0000 | 0.2621 | (0.1870, 0.3279) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0169 | (-0.0162, 0.0585) | 0.1627 | 0.0169 | (-0.0187, 0.0882) | 0.3410 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0058 | (-0.0276, 0.0145) | 0.6843 | -0.0058 | (-0.0223, 0.0198) | 0.6913 |
| controlled_alt_vs_candidate_no_context | length_score | 0.3567 | (0.1783, 0.5200) | 0.0000 | 0.3567 | (0.1509, 0.5722) | 0.0003 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.2450 | (0.1750, 0.3150) | 0.0000 | 0.2450 | (0.1909, 0.3111) | 0.0000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1972 | (0.1635, 0.2317) | 0.0000 | 0.1972 | (0.1835, 0.2118) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 5 | 6 | 9 | 0.4750 | 0.4545 |
| proposed_vs_candidate_no_context | persona_consistency | 1 | 2 | 17 | 0.4750 | 0.3333 |
| proposed_vs_candidate_no_context | naturalness | 8 | 2 | 10 | 0.6500 | 0.8000 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 5 | 6 | 9 | 0.4750 | 0.4545 |
| proposed_vs_candidate_no_context | context_overlap | 8 | 3 | 9 | 0.6250 | 0.7273 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 1 | 2 | 17 | 0.4750 | 0.3333 |
| proposed_vs_candidate_no_context | persona_style | 0 | 0 | 20 | 0.5000 | nan |
| proposed_vs_candidate_no_context | distinct1 | 6 | 3 | 11 | 0.5750 | 0.6667 |
| proposed_vs_candidate_no_context | length_score | 7 | 3 | 10 | 0.6000 | 0.7000 |
| proposed_vs_candidate_no_context | sentence_score | 4 | 3 | 13 | 0.5250 | 0.5714 |
| proposed_vs_candidate_no_context | overall_quality | 5 | 6 | 9 | 0.4750 | 0.4545 |
| controlled_vs_proposed_raw | context_relevance | 18 | 2 | 0 | 0.9000 | 0.9000 |
| controlled_vs_proposed_raw | persona_consistency | 17 | 1 | 2 | 0.9000 | 0.9444 |
| controlled_vs_proposed_raw | naturalness | 13 | 7 | 0 | 0.6500 | 0.6500 |
| controlled_vs_proposed_raw | context_keyword_coverage | 18 | 1 | 1 | 0.9250 | 0.9474 |
| controlled_vs_proposed_raw | context_overlap | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 17 | 0 | 3 | 0.9250 | 1.0000 |
| controlled_vs_proposed_raw | persona_style | 3 | 2 | 15 | 0.5250 | 0.6000 |
| controlled_vs_proposed_raw | distinct1 | 6 | 12 | 2 | 0.3500 | 0.3333 |
| controlled_vs_proposed_raw | length_score | 12 | 7 | 1 | 0.6250 | 0.6316 |
| controlled_vs_proposed_raw | sentence_score | 14 | 1 | 5 | 0.8250 | 0.9333 |
| controlled_vs_proposed_raw | overall_quality | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_relevance | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 17 | 1 | 2 | 0.9000 | 0.9444 |
| controlled_vs_candidate_no_context | naturalness | 15 | 5 | 0 | 0.7500 | 0.7500 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 17 | 1 | 2 | 0.9000 | 0.9444 |
| controlled_vs_candidate_no_context | persona_style | 3 | 2 | 15 | 0.5250 | 0.6000 |
| controlled_vs_candidate_no_context | distinct1 | 6 | 13 | 1 | 0.3250 | 0.3158 |
| controlled_vs_candidate_no_context | length_score | 15 | 4 | 1 | 0.7750 | 0.7895 |
| controlled_vs_candidate_no_context | sentence_score | 14 | 0 | 6 | 0.8500 | 1.0000 |
| controlled_vs_candidate_no_context | overall_quality | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 9 | 8 | 3 | 0.5250 | 0.5294 |
| controlled_alt_vs_controlled_default | persona_consistency | 6 | 6 | 8 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | naturalness | 12 | 5 | 3 | 0.6750 | 0.7059 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 6 | 8 | 6 | 0.4500 | 0.4286 |
| controlled_alt_vs_controlled_default | context_overlap | 9 | 8 | 3 | 0.5250 | 0.5294 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 5 | 5 | 10 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | persona_style | 2 | 3 | 15 | 0.4750 | 0.4000 |
| controlled_alt_vs_controlled_default | distinct1 | 11 | 6 | 3 | 0.6250 | 0.6471 |
| controlled_alt_vs_controlled_default | length_score | 12 | 5 | 3 | 0.6750 | 0.7059 |
| controlled_alt_vs_controlled_default | sentence_score | 2 | 2 | 16 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | overall_quality | 6 | 11 | 3 | 0.3750 | 0.3529 |
| controlled_alt_vs_proposed_raw | context_relevance | 18 | 2 | 0 | 0.9000 | 0.9000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 18 | 0 | 2 | 0.9500 | 1.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 16 | 4 | 0 | 0.8000 | 0.8000 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 18 | 2 | 0 | 0.9000 | 0.9000 |
| controlled_alt_vs_proposed_raw | context_overlap | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 18 | 0 | 2 | 0.9500 | 1.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 2 | 1 | 17 | 0.5250 | 0.6667 |
| controlled_alt_vs_proposed_raw | distinct1 | 11 | 9 | 0 | 0.5500 | 0.5500 |
| controlled_alt_vs_proposed_raw | length_score | 14 | 6 | 0 | 0.7000 | 0.7000 |
| controlled_alt_vs_proposed_raw | sentence_score | 14 | 1 | 5 | 0.8250 | 0.9333 |
| controlled_alt_vs_proposed_raw | overall_quality | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_alt_vs_candidate_no_context | context_relevance | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 17 | 1 | 2 | 0.9000 | 0.9444 |
| controlled_alt_vs_candidate_no_context | naturalness | 16 | 4 | 0 | 0.8000 | 0.8000 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 17 | 1 | 2 | 0.9000 | 0.9444 |
| controlled_alt_vs_candidate_no_context | persona_style | 2 | 1 | 17 | 0.5250 | 0.6667 |
| controlled_alt_vs_candidate_no_context | distinct1 | 11 | 9 | 0 | 0.5500 | 0.5500 |
| controlled_alt_vs_candidate_no_context | length_score | 16 | 4 | 0 | 0.8000 | 0.8000 |
| controlled_alt_vs_candidate_no_context | sentence_score | 14 | 0 | 6 | 0.8500 | 1.0000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 20 | 0 | 0 | 1.0000 | 1.0000 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.2000 | 0.4000 | 0.6000 |
| proposed_contextual_controlled_tuned | 0.0000 | 0.0000 | 0.6000 | 0.1000 | 0.9000 |
| proposed_contextual | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `5`
- Unique template signatures: `17`
- Template signature ratio: `0.8500`
- Effective sample size by source clustering: `4.65`
- Effective sample size by template-signature clustering: `15.38`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: unavailable (disabled_by_flag).

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
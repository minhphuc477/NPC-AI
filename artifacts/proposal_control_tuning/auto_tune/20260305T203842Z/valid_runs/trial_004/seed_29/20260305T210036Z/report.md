# Proposal Alignment Evaluation Report

- Run ID: `20260305T210036Z`
- Generated: `2026-03-05T21:04:17.818155+00:00`
- Scenarios: `artifacts\proposal_control_tuning\auto_tune\20260305T203842Z\valid_runs\trial_004\seed_29\20260305T210036Z\scenarios.jsonl`
- Scenario count: `20`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_tuned`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2320 (0.1834, 0.2822) | 0.3943 (0.3521, 0.4420) | 0.8771 (0.8427, 0.9097) | 0.4211 (0.4057, 0.4369) | n/a |
| proposed_contextual_controlled_tuned | 0.2411 (0.1873, 0.2967) | 0.4161 (0.3430, 0.4978) | 0.8694 (0.8429, 0.8978) | 0.4315 (0.3944, 0.4715) | n/a |
| proposed_contextual | 0.1214 (0.0686, 0.1845) | 0.2442 (0.1772, 0.3218) | 0.8545 (0.8215, 0.8889) | 0.3101 (0.2588, 0.3590) | n/a |
| candidate_no_context | 0.0337 (0.0150, 0.0561) | 0.2447 (0.1667, 0.3284) | 0.8809 (0.8423, 0.9170) | 0.2742 (0.2384, 0.3110) | n/a |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0877 | 2.6040 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0005 | -0.0019 |
| proposed_vs_candidate_no_context | naturalness | -0.0264 | -0.0300 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.1121 | 3.7949 |
| proposed_vs_candidate_no_context | context_overlap | 0.0308 | 0.7103 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0143 | 0.0851 |
| proposed_vs_candidate_no_context | persona_style | -0.0595 | -0.1078 |
| proposed_vs_candidate_no_context | distinct1 | -0.0115 | -0.0121 |
| proposed_vs_candidate_no_context | length_score | -0.0917 | -0.1698 |
| proposed_vs_candidate_no_context | sentence_score | -0.0350 | -0.0384 |
| proposed_vs_candidate_no_context | overall_quality | 0.0360 | 0.1312 |
| controlled_vs_proposed_raw | context_relevance | 0.1106 | 0.9108 |
| controlled_vs_proposed_raw | persona_consistency | 0.1500 | 0.6143 |
| controlled_vs_proposed_raw | naturalness | 0.0226 | 0.0265 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.1480 | 1.0444 |
| controlled_vs_proposed_raw | context_overlap | 0.0234 | 0.3154 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1631 | 0.8954 |
| controlled_vs_proposed_raw | persona_style | 0.0978 | 0.1986 |
| controlled_vs_proposed_raw | distinct1 | 0.0018 | 0.0019 |
| controlled_vs_proposed_raw | length_score | 0.0833 | 0.1859 |
| controlled_vs_proposed_raw | sentence_score | 0.0525 | 0.0598 |
| controlled_vs_proposed_raw | overall_quality | 0.1110 | 0.3578 |
| controlled_vs_candidate_no_context | context_relevance | 0.1983 | 5.8865 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1496 | 0.6112 |
| controlled_vs_candidate_no_context | naturalness | -0.0038 | -0.0043 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.2601 | 8.8026 |
| controlled_vs_candidate_no_context | context_overlap | 0.0542 | 1.2497 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1774 | 1.0567 |
| controlled_vs_candidate_no_context | persona_style | 0.0383 | 0.0694 |
| controlled_vs_candidate_no_context | distinct1 | -0.0097 | -0.0102 |
| controlled_vs_candidate_no_context | length_score | -0.0083 | -0.0154 |
| controlled_vs_candidate_no_context | sentence_score | 0.0175 | 0.0192 |
| controlled_vs_candidate_no_context | overall_quality | 0.1470 | 0.5360 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0091 | 0.0394 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0218 | 0.0553 |
| controlled_alt_vs_controlled_default | naturalness | -0.0077 | -0.0088 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0115 | 0.0398 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0036 | 0.0371 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0202 | 0.0586 |
| controlled_alt_vs_controlled_default | persona_style | 0.0280 | 0.0474 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0149 | -0.0157 |
| controlled_alt_vs_controlled_default | length_score | -0.0067 | -0.0125 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0000 | 0.0000 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0104 | 0.0248 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1197 | 0.9861 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1718 | 0.7035 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0149 | 0.0175 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.1595 | 1.1257 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0270 | 0.3642 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1833 | 1.0065 |
| controlled_alt_vs_proposed_raw | persona_style | 0.1258 | 0.2554 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0131 | -0.0139 |
| controlled_alt_vs_proposed_raw | length_score | 0.0767 | 0.1710 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0525 | 0.0598 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1214 | 0.3914 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2075 | 6.1580 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1714 | 0.7003 |
| controlled_alt_vs_candidate_no_context | naturalness | -0.0115 | -0.0131 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.2716 | 9.1923 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0578 | 1.3332 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1976 | 1.1773 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0663 | 0.1202 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0246 | -0.0257 |
| controlled_alt_vs_candidate_no_context | length_score | -0.0150 | -0.0278 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.0175 | 0.0192 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1574 | 0.5740 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0877 | (0.0413, 0.1391) | 0.0000 | 0.0877 | (0.0066, 0.1264) | 0.0000 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0005 | (-0.0934, 0.0883) | 0.5013 | -0.0005 | (-0.0990, 0.0322) | 0.5997 |
| proposed_vs_candidate_no_context | naturalness | -0.0264 | (-0.0666, 0.0129) | 0.9123 | -0.0264 | (-0.1145, 0.0079) | 0.9650 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.1121 | (0.0523, 0.1803) | 0.0000 | 0.1121 | (0.0000, 0.1667) | 0.0410 |
| proposed_vs_candidate_no_context | context_overlap | 0.0308 | (0.0142, 0.0498) | 0.0000 | 0.0308 | (0.0221, 0.0334) | 0.0000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0143 | (-0.0821, 0.1155) | 0.3940 | 0.0143 | (-0.0417, 0.0357) | 0.2543 |
| proposed_vs_candidate_no_context | persona_style | -0.0595 | (-0.1794, 0.0529) | 0.8330 | -0.0595 | (-0.3281, 0.0778) | 0.8563 |
| proposed_vs_candidate_no_context | distinct1 | -0.0115 | (-0.0277, 0.0075) | 0.8983 | -0.0115 | (-0.0259, 0.0051) | 0.9613 |
| proposed_vs_candidate_no_context | length_score | -0.0917 | (-0.2483, 0.0633) | 0.8820 | -0.0917 | (-0.4333, 0.0292) | 0.8440 |
| proposed_vs_candidate_no_context | sentence_score | -0.0350 | (-0.1050, 0.0350) | 0.9080 | -0.0350 | (-0.1750, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | overall_quality | 0.0360 | (-0.0144, 0.0916) | 0.0813 | 0.0360 | (-0.0527, 0.0664) | 0.1483 |
| controlled_vs_proposed_raw | context_relevance | 0.1106 | (0.0413, 0.1773) | 0.0007 | 0.1106 | (0.0378, 0.2430) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.1500 | (0.0622, 0.2313) | 0.0003 | 0.1500 | (0.1109, 0.2552) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0226 | (-0.0256, 0.0704) | 0.1853 | 0.0226 | (-0.0619, 0.1590) | 0.2520 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.1480 | (0.0549, 0.2437) | 0.0010 | 0.1480 | (0.0455, 0.3364) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0234 | (0.0051, 0.0408) | 0.0073 | 0.0234 | (0.0199, 0.0261) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1631 | (0.0702, 0.2643) | 0.0003 | 0.1631 | (0.1458, 0.1786) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.0978 | (-0.0175, 0.2242) | 0.0540 | 0.0978 | (-0.0312, 0.6094) | 0.2930 |
| controlled_vs_proposed_raw | distinct1 | 0.0018 | (-0.0167, 0.0199) | 0.4143 | 0.0018 | (-0.0292, 0.0402) | 0.3710 |
| controlled_vs_proposed_raw | length_score | 0.0833 | (-0.1150, 0.2650) | 0.1923 | 0.0833 | (-0.2292, 0.5833) | 0.2520 |
| controlled_vs_proposed_raw | sentence_score | 0.0525 | (-0.0525, 0.1575) | 0.1813 | 0.0525 | (-0.0437, 0.2625) | 0.1447 |
| controlled_vs_proposed_raw | overall_quality | 0.1110 | (0.0553, 0.1659) | 0.0000 | 0.1110 | (0.0767, 0.2327) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.1983 | (0.1495, 0.2511) | 0.0000 | 0.1983 | (0.1273, 0.2496) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1496 | (0.0718, 0.2282) | 0.0000 | 0.1496 | (0.1431, 0.1563) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | -0.0038 | (-0.0598, 0.0597) | 0.5553 | -0.0038 | (-0.0786, 0.0469) | 0.5937 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.2601 | (0.1924, 0.3323) | 0.0000 | 0.2601 | (0.1591, 0.3364) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0542 | (0.0387, 0.0687) | 0.0000 | 0.0542 | (0.0472, 0.0586) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1774 | (0.0857, 0.2703) | 0.0000 | 0.1774 | (0.1250, 0.2143) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.0383 | (-0.0729, 0.1503) | 0.2617 | 0.0383 | (-0.0938, 0.2812) | 0.2567 |
| controlled_vs_candidate_no_context | distinct1 | -0.0097 | (-0.0335, 0.0133) | 0.7943 | -0.0097 | (-0.0501, 0.0187) | 0.7070 |
| controlled_vs_candidate_no_context | length_score | -0.0083 | (-0.2517, 0.2333) | 0.5207 | -0.0083 | (-0.2708, 0.1750) | 0.6037 |
| controlled_vs_candidate_no_context | sentence_score | 0.0175 | (-0.0875, 0.1225) | 0.4370 | 0.0175 | (-0.0437, 0.0875) | 0.2667 |
| controlled_vs_candidate_no_context | overall_quality | 0.1470 | (0.1072, 0.1848) | 0.0000 | 0.1470 | (0.1266, 0.1800) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0091 | (-0.0533, 0.0741) | 0.4007 | 0.0091 | (-0.0345, 0.0808) | 0.3627 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0218 | (-0.0808, 0.1300) | 0.3493 | 0.0218 | (-0.0241, 0.1427) | 0.2600 |
| controlled_alt_vs_controlled_default | naturalness | -0.0077 | (-0.0474, 0.0354) | 0.6510 | -0.0077 | (-0.0151, 0.0155) | 0.7387 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0115 | (-0.0699, 0.0936) | 0.3913 | 0.0115 | (-0.0417, 0.0955) | 0.2513 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0036 | (-0.0204, 0.0319) | 0.4070 | 0.0036 | (-0.0178, 0.0467) | 0.3670 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0202 | (-0.1012, 0.1536) | 0.3753 | 0.0202 | (-0.0536, 0.1667) | 0.3680 |
| controlled_alt_vs_controlled_default | persona_style | 0.0280 | (-0.0454, 0.1109) | 0.2403 | 0.0280 | (-0.0472, 0.0938) | 0.3697 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0149 | (-0.0328, 0.0049) | 0.9350 | -0.0149 | (-0.0196, -0.0100) | 1.0000 |
| controlled_alt_vs_controlled_default | length_score | -0.0067 | (-0.1950, 0.1850) | 0.5393 | -0.0067 | (-0.0500, 0.1167) | 0.6493 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0000 | (-0.0875, 0.0875) | 0.5717 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0104 | (-0.0305, 0.0574) | 0.3167 | 0.0104 | (-0.0153, 0.0926) | 0.2947 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1197 | (0.0369, 0.2007) | 0.0013 | 0.1197 | (0.0547, 0.3238) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1718 | (0.0665, 0.2883) | 0.0007 | 0.1718 | (0.1125, 0.3979) | 0.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0149 | (-0.0354, 0.0670) | 0.2800 | 0.0149 | (-0.0769, 0.1745) | 0.3647 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.1595 | (0.0491, 0.2681) | 0.0023 | 0.1595 | (0.0682, 0.4318) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0270 | (0.0028, 0.0551) | 0.0120 | 0.0270 | (0.0083, 0.0717) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1833 | (0.0690, 0.3190) | 0.0007 | 0.1833 | (0.1250, 0.3333) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 0.1258 | (0.0013, 0.2673) | 0.0240 | 0.1258 | (-0.0760, 0.6562) | 0.1517 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0131 | (-0.0354, 0.0076) | 0.8773 | -0.0131 | (-0.0392, 0.0205) | 0.7387 |
| controlled_alt_vs_proposed_raw | length_score | 0.0767 | (-0.1083, 0.2817) | 0.2063 | 0.0767 | (-0.2792, 0.7000) | 0.3703 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0525 | (-0.0350, 0.1400) | 0.1647 | 0.0525 | (-0.0437, 0.2625) | 0.1550 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1214 | (0.0541, 0.1976) | 0.0003 | 0.1214 | (0.0690, 0.3254) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2075 | (0.1533, 0.2633) | 0.0000 | 0.2075 | (0.1443, 0.3304) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1714 | (0.0360, 0.3060) | 0.0063 | 0.1714 | (0.1286, 0.2990) | 0.0000 |
| controlled_alt_vs_candidate_no_context | naturalness | -0.0115 | (-0.0570, 0.0394) | 0.6780 | -0.0115 | (-0.0937, 0.0599) | 0.6983 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.2716 | (0.2042, 0.3461) | 0.0000 | 0.2716 | (0.1818, 0.4318) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0578 | (0.0365, 0.0833) | 0.0000 | 0.0578 | (0.0408, 0.0939) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1976 | (0.0535, 0.3548) | 0.0053 | 0.1976 | (0.1607, 0.2917) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0663 | (-0.0518, 0.2030) | 0.1477 | 0.0663 | (0.0000, 0.3281) | 0.0423 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0246 | (-0.0442, -0.0064) | 0.9940 | -0.0246 | (-0.0601, 0.0014) | 0.8497 |
| controlled_alt_vs_candidate_no_context | length_score | -0.0150 | (-0.1967, 0.1900) | 0.5663 | -0.0150 | (-0.3208, 0.2667) | 0.7030 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.0175 | (-0.0700, 0.1050) | 0.4357 | 0.0175 | (-0.0437, 0.0875) | 0.2600 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1574 | (0.0901, 0.2278) | 0.0000 | 0.1574 | (0.1217, 0.2727) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 15 | 2 | 3 | 0.8250 | 0.8824 |
| proposed_vs_candidate_no_context | persona_consistency | 9 | 6 | 5 | 0.5750 | 0.6000 |
| proposed_vs_candidate_no_context | naturalness | 5 | 12 | 3 | 0.3250 | 0.2941 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 9 | 0 | 11 | 0.7250 | 1.0000 |
| proposed_vs_candidate_no_context | context_overlap | 15 | 2 | 3 | 0.8250 | 0.8824 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 5 | 4 | 11 | 0.5250 | 0.5556 |
| proposed_vs_candidate_no_context | persona_style | 5 | 5 | 10 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | distinct1 | 5 | 12 | 3 | 0.3250 | 0.2941 |
| proposed_vs_candidate_no_context | length_score | 5 | 11 | 4 | 0.3500 | 0.3125 |
| proposed_vs_candidate_no_context | sentence_score | 1 | 3 | 16 | 0.4500 | 0.2500 |
| proposed_vs_candidate_no_context | overall_quality | 13 | 4 | 3 | 0.7250 | 0.7647 |
| controlled_vs_proposed_raw | context_relevance | 14 | 6 | 0 | 0.7000 | 0.7000 |
| controlled_vs_proposed_raw | persona_consistency | 14 | 2 | 4 | 0.8000 | 0.8750 |
| controlled_vs_proposed_raw | naturalness | 11 | 9 | 0 | 0.5500 | 0.5500 |
| controlled_vs_proposed_raw | context_keyword_coverage | 13 | 4 | 3 | 0.7250 | 0.7647 |
| controlled_vs_proposed_raw | context_overlap | 15 | 5 | 0 | 0.7500 | 0.7500 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 14 | 1 | 5 | 0.8250 | 0.9333 |
| controlled_vs_proposed_raw | persona_style | 6 | 4 | 10 | 0.5500 | 0.6000 |
| controlled_vs_proposed_raw | distinct1 | 12 | 7 | 1 | 0.6250 | 0.6316 |
| controlled_vs_proposed_raw | length_score | 10 | 8 | 2 | 0.5500 | 0.5556 |
| controlled_vs_proposed_raw | sentence_score | 6 | 3 | 11 | 0.5750 | 0.6667 |
| controlled_vs_proposed_raw | overall_quality | 17 | 3 | 0 | 0.8500 | 0.8500 |
| controlled_vs_candidate_no_context | context_relevance | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 15 | 3 | 2 | 0.8000 | 0.8333 |
| controlled_vs_candidate_no_context | naturalness | 8 | 12 | 0 | 0.4000 | 0.4000 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 18 | 0 | 2 | 0.9500 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 15 | 2 | 3 | 0.8250 | 0.8824 |
| controlled_vs_candidate_no_context | persona_style | 5 | 4 | 11 | 0.5250 | 0.5556 |
| controlled_vs_candidate_no_context | distinct1 | 7 | 12 | 1 | 0.3750 | 0.3684 |
| controlled_vs_candidate_no_context | length_score | 9 | 11 | 0 | 0.4500 | 0.4500 |
| controlled_vs_candidate_no_context | sentence_score | 5 | 4 | 11 | 0.5250 | 0.5556 |
| controlled_vs_candidate_no_context | overall_quality | 18 | 2 | 0 | 0.9000 | 0.9000 |
| controlled_alt_vs_controlled_default | context_relevance | 10 | 8 | 2 | 0.5500 | 0.5556 |
| controlled_alt_vs_controlled_default | persona_consistency | 7 | 8 | 5 | 0.4750 | 0.4667 |
| controlled_alt_vs_controlled_default | naturalness | 8 | 10 | 2 | 0.4500 | 0.4444 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 8 | 6 | 6 | 0.5500 | 0.5714 |
| controlled_alt_vs_controlled_default | context_overlap | 8 | 10 | 2 | 0.4500 | 0.4444 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 5 | 4 | 11 | 0.5250 | 0.5556 |
| controlled_alt_vs_controlled_default | persona_style | 5 | 5 | 10 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | distinct1 | 5 | 12 | 3 | 0.3250 | 0.2941 |
| controlled_alt_vs_controlled_default | length_score | 7 | 9 | 4 | 0.4500 | 0.4375 |
| controlled_alt_vs_controlled_default | sentence_score | 3 | 3 | 14 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | overall_quality | 7 | 11 | 2 | 0.4000 | 0.3889 |
| controlled_alt_vs_proposed_raw | context_relevance | 15 | 5 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_proposed_raw | persona_consistency | 13 | 3 | 4 | 0.7500 | 0.8125 |
| controlled_alt_vs_proposed_raw | naturalness | 12 | 8 | 0 | 0.6000 | 0.6000 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 14 | 4 | 2 | 0.7500 | 0.7778 |
| controlled_alt_vs_proposed_raw | context_overlap | 14 | 6 | 0 | 0.7000 | 0.7000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 13 | 2 | 5 | 0.7750 | 0.8667 |
| controlled_alt_vs_proposed_raw | persona_style | 7 | 5 | 8 | 0.5500 | 0.5833 |
| controlled_alt_vs_proposed_raw | distinct1 | 12 | 8 | 0 | 0.6000 | 0.6000 |
| controlled_alt_vs_proposed_raw | length_score | 12 | 7 | 1 | 0.6250 | 0.6316 |
| controlled_alt_vs_proposed_raw | sentence_score | 5 | 2 | 13 | 0.5750 | 0.7143 |
| controlled_alt_vs_proposed_raw | overall_quality | 13 | 7 | 0 | 0.6500 | 0.6500 |
| controlled_alt_vs_candidate_no_context | context_relevance | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 15 | 4 | 1 | 0.7750 | 0.7895 |
| controlled_alt_vs_candidate_no_context | naturalness | 7 | 13 | 0 | 0.3500 | 0.3500 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 18 | 0 | 2 | 0.9500 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 17 | 3 | 0 | 0.8500 | 0.8500 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 14 | 4 | 2 | 0.7500 | 0.7778 |
| controlled_alt_vs_candidate_no_context | persona_style | 6 | 7 | 7 | 0.4750 | 0.4615 |
| controlled_alt_vs_candidate_no_context | distinct1 | 5 | 14 | 1 | 0.2750 | 0.2632 |
| controlled_alt_vs_candidate_no_context | length_score | 8 | 12 | 0 | 0.4000 | 0.4000 |
| controlled_alt_vs_candidate_no_context | sentence_score | 4 | 3 | 13 | 0.5250 | 0.5714 |
| controlled_alt_vs_candidate_no_context | overall_quality | 17 | 3 | 0 | 0.8500 | 0.8500 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.3000 | 0.6500 | 0.3500 |
| proposed_contextual_controlled_tuned | 0.0000 | 0.0000 | 0.2500 | 0.5000 | 0.5000 |
| proposed_contextual | 0.0000 | 0.0000 | 0.3000 | 0.0000 | 0.0000 |
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
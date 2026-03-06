# Proposal Alignment Evaluation Report

- Run ID: `20260305T210504Z`
- Generated: `2026-03-05T21:08:26.937291+00:00`
- Scenarios: `artifacts\proposal_control_tuning\auto_tune_v2b\20260305T210504Z\train_runs\trial_000\seed_19\20260305T210504Z\scenarios.jsonl`
- Scenario count: `20`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_tuned`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2690 (0.2459, 0.2941) | 0.3894 (0.2952, 0.4887) | 0.9002 (0.8805, 0.9195) | 0.4404 (0.4066, 0.4728) | n/a |
| proposed_contextual_controlled_tuned | 0.3230 (0.2852, 0.3618) | 0.3403 (0.2847, 0.3993) | 0.8829 (0.8573, 0.9058) | 0.4440 (0.4193, 0.4672) | n/a |
| proposed_contextual | 0.0803 (0.0504, 0.1125) | 0.1298 (0.0964, 0.1655) | 0.8457 (0.8097, 0.8821) | 0.2451 (0.2179, 0.2719) | n/a |
| candidate_no_context | 0.0264 (0.0135, 0.0429) | 0.1340 (0.0967, 0.1698) | 0.7966 (0.7667, 0.8346) | 0.2123 (0.1912, 0.2364) | n/a |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0539 | 2.0412 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0043 | -0.0320 |
| proposed_vs_candidate_no_context | naturalness | 0.0491 | 0.0616 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0682 | 3.0000 |
| proposed_vs_candidate_no_context | context_overlap | 0.0206 | 0.5891 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0095 | -0.4000 |
| proposed_vs_candidate_no_context | persona_style | 0.0167 | 0.0290 |
| proposed_vs_candidate_no_context | distinct1 | 0.0170 | 0.0182 |
| proposed_vs_candidate_no_context | length_score | 0.1850 | 0.7208 |
| proposed_vs_candidate_no_context | sentence_score | 0.0525 | 0.0729 |
| proposed_vs_candidate_no_context | overall_quality | 0.0328 | 0.1546 |
| controlled_vs_proposed_raw | context_relevance | 0.1886 | 2.3483 |
| controlled_vs_proposed_raw | persona_consistency | 0.2596 | 2.0007 |
| controlled_vs_proposed_raw | naturalness | 0.0545 | 0.0644 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2455 | 2.7000 |
| controlled_vs_proposed_raw | context_overlap | 0.0561 | 1.0074 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.3162 | 22.1333 |
| controlled_vs_proposed_raw | persona_style | 0.0333 | 0.0563 |
| controlled_vs_proposed_raw | distinct1 | -0.0030 | -0.0032 |
| controlled_vs_proposed_raw | length_score | 0.2083 | 0.4717 |
| controlled_vs_proposed_raw | sentence_score | 0.1400 | 0.1812 |
| controlled_vs_proposed_raw | overall_quality | 0.1953 | 0.7967 |
| controlled_vs_candidate_no_context | context_relevance | 0.2425 | 9.1829 |
| controlled_vs_candidate_no_context | persona_consistency | 0.2553 | 1.9048 |
| controlled_vs_candidate_no_context | naturalness | 0.1035 | 0.1299 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3136 | 13.8000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0767 | 2.1900 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.3067 | 12.8800 |
| controlled_vs_candidate_no_context | persona_style | 0.0500 | 0.0870 |
| controlled_vs_candidate_no_context | distinct1 | 0.0140 | 0.0150 |
| controlled_vs_candidate_no_context | length_score | 0.3933 | 1.5325 |
| controlled_vs_candidate_no_context | sentence_score | 0.1925 | 0.2674 |
| controlled_vs_candidate_no_context | overall_quality | 0.2281 | 1.0745 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0541 | 0.2010 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0491 | -0.1261 |
| controlled_alt_vs_controlled_default | naturalness | -0.0172 | -0.0191 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0727 | 0.2162 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0105 | 0.0939 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0593 | -0.1794 |
| controlled_alt_vs_controlled_default | persona_style | -0.0083 | -0.0133 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0001 | -0.0001 |
| controlled_alt_vs_controlled_default | length_score | -0.0683 | -0.1051 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0350 | -0.0384 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0035 | 0.0081 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.2427 | 3.0212 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.2105 | 1.6224 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0372 | 0.0440 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.3182 | 3.5000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0665 | 1.1958 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.2569 | 17.9833 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0250 | 0.0423 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0032 | -0.0033 |
| controlled_alt_vs_proposed_raw | length_score | 0.1400 | 0.3170 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1050 | 0.1359 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1988 | 0.8112 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2966 | 11.2294 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.2062 | 1.5385 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0863 | 0.1083 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3864 | 17.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0872 | 2.4895 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.2474 | 10.3900 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0417 | 0.0725 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0139 | 0.0148 |
| controlled_alt_vs_candidate_no_context | length_score | 0.3250 | 1.2662 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1575 | 0.2188 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.2317 | 1.0912 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0539 | (0.0276, 0.0825) | 0.0000 | 0.0539 | (0.0256, 0.0746) | 0.0003 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0043 | (-0.0324, 0.0205) | 0.6360 | -0.0043 | (-0.0281, 0.0172) | 0.6593 |
| proposed_vs_candidate_no_context | naturalness | 0.0491 | (0.0083, 0.0892) | 0.0083 | 0.0491 | (0.0231, 0.0705) | 0.0003 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0682 | (0.0318, 0.1045) | 0.0000 | 0.0682 | (0.0303, 0.0955) | 0.0000 |
| proposed_vs_candidate_no_context | context_overlap | 0.0206 | (0.0083, 0.0339) | 0.0000 | 0.0206 | (0.0058, 0.0309) | 0.0000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0095 | (-0.0417, 0.0214) | 0.7503 | -0.0095 | (-0.0351, 0.0136) | 0.7983 |
| proposed_vs_candidate_no_context | persona_style | 0.0167 | (0.0000, 0.0500) | 0.3613 | 0.0167 | (0.0000, 0.0588) | 0.3167 |
| proposed_vs_candidate_no_context | distinct1 | 0.0170 | (-0.0063, 0.0394) | 0.0767 | 0.0170 | (-0.0043, 0.0393) | 0.0657 |
| proposed_vs_candidate_no_context | length_score | 0.1850 | (0.0300, 0.3267) | 0.0073 | 0.1850 | (0.0941, 0.2649) | 0.0000 |
| proposed_vs_candidate_no_context | sentence_score | 0.0525 | (-0.0350, 0.1400) | 0.1687 | 0.0525 | (-0.0412, 0.1375) | 0.2260 |
| proposed_vs_candidate_no_context | overall_quality | 0.0328 | (0.0105, 0.0563) | 0.0013 | 0.0328 | (0.0163, 0.0442) | 0.0000 |
| controlled_vs_proposed_raw | context_relevance | 0.1886 | (0.1588, 0.2183) | 0.0000 | 0.1886 | (0.1627, 0.2268) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.2596 | (0.1693, 0.3627) | 0.0000 | 0.2596 | (0.1811, 0.3900) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0545 | (0.0117, 0.0966) | 0.0070 | 0.0545 | (0.0012, 0.0979) | 0.0117 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2455 | (0.2045, 0.2818) | 0.0000 | 0.2455 | (0.2133, 0.2941) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0561 | (0.0405, 0.0718) | 0.0000 | 0.0561 | (0.0474, 0.0691) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.3162 | (0.2112, 0.4395) | 0.0000 | 0.3162 | (0.2175, 0.4771) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.0333 | (0.0000, 0.1000) | 0.3443 | 0.0333 | (0.0000, 0.1176) | 0.3293 |
| controlled_vs_proposed_raw | distinct1 | -0.0030 | (-0.0247, 0.0181) | 0.6223 | -0.0030 | (-0.0369, 0.0246) | 0.5737 |
| controlled_vs_proposed_raw | length_score | 0.2083 | (0.0467, 0.3617) | 0.0053 | 0.2083 | (0.0368, 0.3392) | 0.0050 |
| controlled_vs_proposed_raw | sentence_score | 0.1400 | (0.0175, 0.2450) | 0.0130 | 0.1400 | (0.0304, 0.2882) | 0.0103 |
| controlled_vs_proposed_raw | overall_quality | 0.1953 | (0.1561, 0.2394) | 0.0000 | 0.1953 | (0.1605, 0.2545) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2425 | (0.2180, 0.2680) | 0.0000 | 0.2425 | (0.2127, 0.2755) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.2553 | (0.1705, 0.3562) | 0.0000 | 0.2553 | (0.1565, 0.3858) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.1035 | (0.0599, 0.1407) | 0.0000 | 0.1035 | (0.0682, 0.1314) | 0.0000 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3136 | (0.2818, 0.3500) | 0.0000 | 0.3136 | (0.2727, 0.3586) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0767 | (0.0658, 0.0865) | 0.0000 | 0.0767 | (0.0666, 0.0883) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.3067 | (0.1988, 0.4348) | 0.0000 | 0.3067 | (0.1825, 0.4667) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.0500 | (0.0000, 0.1333) | 0.1170 | 0.0500 | (0.0000, 0.1765) | 0.3403 |
| controlled_vs_candidate_no_context | distinct1 | 0.0140 | (-0.0062, 0.0320) | 0.0823 | 0.0140 | (-0.0032, 0.0305) | 0.0447 |
| controlled_vs_candidate_no_context | length_score | 0.3933 | (0.2350, 0.5417) | 0.0000 | 0.3933 | (0.2833, 0.4750) | 0.0000 |
| controlled_vs_candidate_no_context | sentence_score | 0.1925 | (0.0700, 0.2975) | 0.0020 | 0.1925 | (0.0921, 0.3062) | 0.0007 |
| controlled_vs_candidate_no_context | overall_quality | 0.2281 | (0.1891, 0.2682) | 0.0000 | 0.2281 | (0.1985, 0.2718) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0541 | (0.0128, 0.0947) | 0.0077 | 0.0541 | (0.0056, 0.0959) | 0.0087 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0491 | (-0.1669, 0.0523) | 0.8127 | -0.0491 | (-0.1667, 0.0320) | 0.8413 |
| controlled_alt_vs_controlled_default | naturalness | -0.0172 | (-0.0402, 0.0069) | 0.9190 | -0.0172 | (-0.0367, -0.0034) | 1.0000 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0727 | (0.0227, 0.1227) | 0.0030 | 0.0727 | (0.0144, 0.1263) | 0.0053 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0105 | (-0.0089, 0.0313) | 0.1477 | 0.0105 | (-0.0073, 0.0296) | 0.1493 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0593 | (-0.2050, 0.0688) | 0.7927 | -0.0593 | (-0.2004, 0.0421) | 0.8330 |
| controlled_alt_vs_controlled_default | persona_style | -0.0083 | (-0.0500, 0.0250) | 0.6583 | -0.0083 | (-0.0294, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0001 | (-0.0149, 0.0137) | 0.4967 | -0.0001 | (-0.0113, 0.0090) | 0.5350 |
| controlled_alt_vs_controlled_default | length_score | -0.0683 | (-0.1783, 0.0433) | 0.8843 | -0.0683 | (-0.1524, -0.0074) | 0.9927 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0350 | (-0.1050, 0.0350) | 0.9070 | -0.0350 | (-0.0737, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0035 | (-0.0438, 0.0500) | 0.4227 | 0.0035 | (-0.0433, 0.0435) | 0.4290 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.2427 | (0.2048, 0.2827) | 0.0000 | 0.2427 | (0.2020, 0.2842) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.2105 | (0.1651, 0.2567) | 0.0000 | 0.2105 | (0.2000, 0.2232) | 0.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0372 | (-0.0119, 0.0838) | 0.0650 | 0.0372 | (-0.0108, 0.0754) | 0.0700 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.3182 | (0.2727, 0.3682) | 0.0000 | 0.3182 | (0.2655, 0.3737) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0665 | (0.0467, 0.0873) | 0.0000 | 0.0665 | (0.0546, 0.0840) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.2569 | (0.2005, 0.3157) | 0.0000 | 0.2569 | (0.2333, 0.2760) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0250 | (0.0000, 0.0667) | 0.1230 | 0.0250 | (0.0000, 0.0882) | 0.3340 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0032 | (-0.0251, 0.0176) | 0.6193 | -0.0032 | (-0.0391, 0.0242) | 0.5697 |
| controlled_alt_vs_proposed_raw | length_score | 0.1400 | (-0.0567, 0.3183) | 0.0680 | 0.1400 | (0.0106, 0.2537) | 0.0233 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1050 | (-0.0175, 0.2275) | 0.0647 | 0.1050 | (-0.0125, 0.2625) | 0.0510 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1988 | (0.1693, 0.2290) | 0.0000 | 0.1988 | (0.1794, 0.2226) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2966 | (0.2645, 0.3320) | 0.0000 | 0.2966 | (0.2477, 0.3507) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.2062 | (0.1514, 0.2620) | 0.0000 | 0.2062 | (0.1826, 0.2227) | 0.0000 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0863 | (0.0427, 0.1273) | 0.0000 | 0.0863 | (0.0501, 0.1191) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3864 | (0.3409, 0.4364) | 0.0000 | 0.3864 | (0.3203, 0.4593) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0872 | (0.0713, 0.1038) | 0.0000 | 0.0872 | (0.0725, 0.0989) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.2474 | (0.1793, 0.3164) | 0.0000 | 0.2474 | (0.2146, 0.2740) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0417 | (0.0000, 0.1083) | 0.1083 | 0.0417 | (0.0000, 0.1471) | 0.3393 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0139 | (-0.0071, 0.0318) | 0.0937 | 0.0139 | (-0.0087, 0.0278) | 0.0633 |
| controlled_alt_vs_candidate_no_context | length_score | 0.3250 | (0.1683, 0.4733) | 0.0000 | 0.3250 | (0.2263, 0.4134) | 0.0000 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1575 | (0.0350, 0.2800) | 0.0133 | 0.1575 | (0.0184, 0.2917) | 0.0113 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.2317 | (0.2054, 0.2589) | 0.0000 | 0.2317 | (0.2205, 0.2456) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 11 | 2 | 7 | 0.7250 | 0.8462 |
| proposed_vs_candidate_no_context | persona_consistency | 3 | 3 | 14 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | naturalness | 10 | 3 | 7 | 0.6750 | 0.7692 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 10 | 0 | 10 | 0.7500 | 1.0000 |
| proposed_vs_candidate_no_context | context_overlap | 11 | 2 | 7 | 0.7250 | 0.8462 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 2 | 3 | 15 | 0.4750 | 0.4000 |
| proposed_vs_candidate_no_context | persona_style | 1 | 0 | 19 | 0.5250 | 1.0000 |
| proposed_vs_candidate_no_context | distinct1 | 7 | 4 | 9 | 0.5750 | 0.6364 |
| proposed_vs_candidate_no_context | length_score | 10 | 2 | 8 | 0.7000 | 0.8333 |
| proposed_vs_candidate_no_context | sentence_score | 5 | 2 | 13 | 0.5750 | 0.7143 |
| proposed_vs_candidate_no_context | overall_quality | 11 | 2 | 7 | 0.7250 | 0.8462 |
| controlled_vs_proposed_raw | context_relevance | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_vs_proposed_raw | persona_consistency | 18 | 0 | 2 | 0.9500 | 1.0000 |
| controlled_vs_proposed_raw | naturalness | 13 | 6 | 1 | 0.6750 | 0.6842 |
| controlled_vs_proposed_raw | context_keyword_coverage | 19 | 0 | 1 | 0.9750 | 1.0000 |
| controlled_vs_proposed_raw | context_overlap | 18 | 2 | 0 | 0.9000 | 0.9000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 18 | 0 | 2 | 0.9500 | 1.0000 |
| controlled_vs_proposed_raw | persona_style | 1 | 0 | 19 | 0.5250 | 1.0000 |
| controlled_vs_proposed_raw | distinct1 | 9 | 10 | 1 | 0.4750 | 0.4737 |
| controlled_vs_proposed_raw | length_score | 13 | 5 | 2 | 0.7000 | 0.7222 |
| controlled_vs_proposed_raw | sentence_score | 11 | 3 | 6 | 0.7000 | 0.7857 |
| controlled_vs_proposed_raw | overall_quality | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_relevance | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 17 | 0 | 3 | 0.9250 | 1.0000 |
| controlled_vs_candidate_no_context | naturalness | 17 | 3 | 0 | 0.8500 | 0.8500 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 17 | 0 | 3 | 0.9250 | 1.0000 |
| controlled_vs_candidate_no_context | persona_style | 2 | 0 | 18 | 0.5500 | 1.0000 |
| controlled_vs_candidate_no_context | distinct1 | 14 | 6 | 0 | 0.7000 | 0.7000 |
| controlled_vs_candidate_no_context | length_score | 18 | 2 | 0 | 0.9000 | 0.9000 |
| controlled_vs_candidate_no_context | sentence_score | 14 | 3 | 3 | 0.7750 | 0.8235 |
| controlled_vs_candidate_no_context | overall_quality | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 11 | 6 | 3 | 0.6250 | 0.6471 |
| controlled_alt_vs_controlled_default | persona_consistency | 6 | 6 | 8 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | naturalness | 5 | 12 | 3 | 0.3250 | 0.2941 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 10 | 2 | 8 | 0.7000 | 0.8333 |
| controlled_alt_vs_controlled_default | context_overlap | 10 | 7 | 3 | 0.5750 | 0.5882 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 5 | 6 | 9 | 0.4750 | 0.4545 |
| controlled_alt_vs_controlled_default | persona_style | 1 | 1 | 18 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | distinct1 | 10 | 7 | 3 | 0.5750 | 0.5882 |
| controlled_alt_vs_controlled_default | length_score | 6 | 11 | 3 | 0.3750 | 0.3529 |
| controlled_alt_vs_controlled_default | sentence_score | 1 | 3 | 16 | 0.4500 | 0.2500 |
| controlled_alt_vs_controlled_default | overall_quality | 12 | 5 | 3 | 0.6750 | 0.7059 |
| controlled_alt_vs_proposed_raw | context_relevance | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 19 | 0 | 1 | 0.9750 | 1.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 14 | 6 | 0 | 0.7000 | 0.7000 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 18 | 2 | 0 | 0.9000 | 0.9000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 19 | 0 | 1 | 0.9750 | 1.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 2 | 0 | 18 | 0.5500 | 1.0000 |
| controlled_alt_vs_proposed_raw | distinct1 | 10 | 9 | 1 | 0.5250 | 0.5263 |
| controlled_alt_vs_proposed_raw | length_score | 11 | 8 | 1 | 0.5750 | 0.5789 |
| controlled_alt_vs_proposed_raw | sentence_score | 10 | 4 | 6 | 0.6500 | 0.7143 |
| controlled_alt_vs_proposed_raw | overall_quality | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 17 | 0 | 3 | 0.9250 | 1.0000 |
| controlled_alt_vs_candidate_no_context | naturalness | 16 | 4 | 0 | 0.8000 | 0.8000 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 17 | 0 | 3 | 0.9250 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_style | 2 | 0 | 18 | 0.5500 | 1.0000 |
| controlled_alt_vs_candidate_no_context | distinct1 | 12 | 6 | 2 | 0.6500 | 0.6667 |
| controlled_alt_vs_candidate_no_context | length_score | 15 | 4 | 1 | 0.7750 | 0.7895 |
| controlled_alt_vs_candidate_no_context | sentence_score | 13 | 4 | 3 | 0.7250 | 0.7647 |
| controlled_alt_vs_candidate_no_context | overall_quality | 20 | 0 | 0 | 1.0000 | 1.0000 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.3500 | 0.3500 | 0.6500 |
| proposed_contextual_controlled_tuned | 0.0000 | 0.0000 | 0.3000 | 0.5000 | 0.5000 |
| proposed_contextual | 0.0000 | 0.0000 | 0.3500 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.6500 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `5`
- Unique template signatures: `17`
- Template signature ratio: `0.8500`
- Effective sample size by source clustering: `4.65`
- Effective sample size by template-signature clustering: `15.38`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: unavailable (disabled_by_flag).

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
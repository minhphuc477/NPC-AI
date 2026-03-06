# Proposal Alignment Evaluation Report

- Run ID: `20260304T181357Z`
- Generated: `2026-03-04T18:16:53.468790+00:00`
- Scenarios: `artifacts\proposal_control_tuning\preflight_smoke\20260304T180952Z\seed_runs\seed_31\20260304T181357Z\scenarios.jsonl`
- Scenario count: `8`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off
- `baseline_no_context`: model `phi3:mini`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2952 (0.2362, 0.3626) | 0.3714 (0.2536, 0.5393) | 0.8519 (0.7997, 0.9015) | 0.4026 (0.3582, 0.4664) | 0.1123 |
| proposed_contextual | 0.0950 (0.0262, 0.1941) | 0.1351 (0.0643, 0.2137) | 0.8041 (0.7564, 0.8653) | 0.2310 (0.1735, 0.2910) | 0.0764 |
| candidate_no_context | 0.0335 (0.0089, 0.0669) | 0.2155 (0.1125, 0.3238) | 0.8099 (0.7739, 0.8536) | 0.2302 (0.1817, 0.2796) | 0.0479 |
| baseline_no_context | 0.0355 (0.0121, 0.0673) | 0.1625 (0.0833, 0.2601) | 0.9008 (0.8717, 0.9320) | 0.2312 (0.2022, 0.2682) | 0.0613 |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0615 | 1.8378 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0804 | -0.3729 |
| proposed_vs_candidate_no_context | naturalness | -0.0058 | -0.0071 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0795 | 2.3333 |
| proposed_vs_candidate_no_context | context_overlap | 0.0195 | 0.6079 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0952 | -0.6154 |
| proposed_vs_candidate_no_context | persona_style | -0.0208 | -0.0455 |
| proposed_vs_candidate_no_context | distinct1 | -0.0113 | -0.0121 |
| proposed_vs_candidate_no_context | length_score | 0.0375 | 0.1452 |
| proposed_vs_candidate_no_context | sentence_score | -0.0875 | -0.1061 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0285 | 0.5947 |
| proposed_vs_candidate_no_context | overall_quality | 0.0007 | 0.0032 |
| proposed_vs_baseline_no_context | context_relevance | 0.0595 | 1.6773 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0274 | -0.1683 |
| proposed_vs_baseline_no_context | naturalness | -0.0968 | -0.1074 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0795 | 2.3333 |
| proposed_vs_baseline_no_context | context_overlap | 0.0128 | 0.3302 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0298 | -0.3333 |
| proposed_vs_baseline_no_context | persona_style | -0.0177 | -0.0389 |
| proposed_vs_baseline_no_context | distinct1 | -0.0414 | -0.0427 |
| proposed_vs_baseline_no_context | length_score | -0.2917 | -0.4965 |
| proposed_vs_baseline_no_context | sentence_score | -0.2188 | -0.2288 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0151 | 0.2468 |
| proposed_vs_baseline_no_context | overall_quality | -0.0003 | -0.0011 |
| controlled_vs_proposed_raw | context_relevance | 0.2002 | 2.1069 |
| controlled_vs_proposed_raw | persona_consistency | 0.2363 | 1.7489 |
| controlled_vs_proposed_raw | naturalness | 0.0478 | 0.0594 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2614 | 2.3000 |
| controlled_vs_proposed_raw | context_overlap | 0.0574 | 1.1134 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2798 | 4.7000 |
| controlled_vs_proposed_raw | persona_style | 0.0625 | 0.1429 |
| controlled_vs_proposed_raw | distinct1 | 0.0039 | 0.0042 |
| controlled_vs_proposed_raw | length_score | 0.1458 | 0.4930 |
| controlled_vs_proposed_raw | sentence_score | 0.1750 | 0.2373 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0359 | 0.4705 |
| controlled_vs_proposed_raw | overall_quality | 0.1717 | 0.7433 |
| controlled_vs_candidate_no_context | context_relevance | 0.2617 | 7.8168 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1560 | 0.7238 |
| controlled_vs_candidate_no_context | naturalness | 0.0420 | 0.0519 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3409 | 10.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0769 | 2.3982 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1845 | 1.1923 |
| controlled_vs_candidate_no_context | persona_style | 0.0417 | 0.0909 |
| controlled_vs_candidate_no_context | distinct1 | -0.0074 | -0.0079 |
| controlled_vs_candidate_no_context | length_score | 0.1833 | 0.7097 |
| controlled_vs_candidate_no_context | sentence_score | 0.0875 | 0.1061 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0644 | 1.3450 |
| controlled_vs_candidate_no_context | overall_quality | 0.1724 | 0.7489 |
| controlled_vs_baseline_no_context | context_relevance | 0.2597 | 7.3181 |
| controlled_vs_baseline_no_context | persona_consistency | 0.2090 | 1.2861 |
| controlled_vs_baseline_no_context | naturalness | -0.0490 | -0.0544 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3409 | 10.0000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0702 | 1.8113 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2500 | 2.8000 |
| controlled_vs_baseline_no_context | persona_style | 0.0448 | 0.0984 |
| controlled_vs_baseline_no_context | distinct1 | -0.0375 | -0.0386 |
| controlled_vs_baseline_no_context | length_score | -0.1458 | -0.2482 |
| controlled_vs_baseline_no_context | sentence_score | -0.0438 | -0.0458 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0511 | 0.8334 |
| controlled_vs_baseline_no_context | overall_quality | 0.1714 | 0.7414 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2597 | 7.3181 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.2090 | 1.2861 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0490 | -0.0544 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3409 | 10.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0702 | 1.8113 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2500 | 2.8000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0448 | 0.0984 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0375 | -0.0386 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | -0.1458 | -0.2482 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | -0.0438 | -0.0458 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0511 | 0.8334 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1714 | 0.7414 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0615 | (-0.0127, 0.1687) | 0.0933 | 0.0615 | (0.0015, 0.1071) | 0.0057 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0804 | (-0.2143, 0.0560) | 0.8713 | -0.0804 | (-0.2317, 0.0095) | 0.9430 |
| proposed_vs_candidate_no_context | naturalness | -0.0058 | (-0.0792, 0.0601) | 0.5697 | -0.0058 | (-0.0989, 0.0705) | 0.5757 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0795 | (-0.0227, 0.2273) | 0.1497 | 0.0795 | (0.0000, 0.1488) | 0.0743 |
| proposed_vs_candidate_no_context | context_overlap | 0.0195 | (0.0019, 0.0380) | 0.0147 | 0.0195 | (0.0028, 0.0356) | 0.0070 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0952 | (-0.2888, 0.0804) | 0.8410 | -0.0952 | (-0.2925, 0.0198) | 0.9450 |
| proposed_vs_candidate_no_context | persona_style | -0.0208 | (-0.0625, 0.0000) | 1.0000 | -0.0208 | (-0.0833, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | distinct1 | -0.0113 | (-0.0367, 0.0158) | 0.7930 | -0.0113 | (-0.0475, 0.0199) | 0.7273 |
| proposed_vs_candidate_no_context | length_score | 0.0375 | (-0.2250, 0.3083) | 0.3760 | 0.0375 | (-0.2933, 0.3619) | 0.4067 |
| proposed_vs_candidate_no_context | sentence_score | -0.0875 | (-0.2625, 0.1312) | 0.8463 | -0.0875 | (-0.2917, 0.0878) | 0.9000 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0285 | (-0.0048, 0.0659) | 0.0517 | 0.0285 | (-0.0096, 0.0815) | 0.1207 |
| proposed_vs_candidate_no_context | overall_quality | 0.0007 | (-0.0838, 0.0882) | 0.4783 | 0.0007 | (-0.0833, 0.0402) | 0.4520 |
| proposed_vs_baseline_no_context | context_relevance | 0.0595 | (-0.0096, 0.1640) | 0.0710 | 0.0595 | (-0.0231, 0.1043) | 0.0890 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0274 | (-0.1589, 0.0673) | 0.6423 | -0.0274 | (-0.2119, 0.0500) | 0.6540 |
| proposed_vs_baseline_no_context | naturalness | -0.0968 | (-0.1563, -0.0223) | 0.9973 | -0.0968 | (-0.1509, 0.0053) | 0.9747 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0795 | (-0.0114, 0.2159) | 0.1240 | 0.0795 | (-0.0364, 0.1488) | 0.1320 |
| proposed_vs_baseline_no_context | context_overlap | 0.0128 | (-0.0048, 0.0305) | 0.0723 | 0.0128 | (-0.0053, 0.0326) | 0.0913 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0298 | (-0.2143, 0.0833) | 0.6317 | -0.0298 | (-0.2579, 0.0625) | 0.6623 |
| proposed_vs_baseline_no_context | persona_style | -0.0177 | (-0.0625, 0.0094) | 0.7570 | -0.0177 | (-0.0833, 0.0100) | 0.7480 |
| proposed_vs_baseline_no_context | distinct1 | -0.0414 | (-0.0629, -0.0197) | 1.0000 | -0.0414 | (-0.0634, -0.0128) | 1.0000 |
| proposed_vs_baseline_no_context | length_score | -0.2917 | (-0.5458, 0.0125) | 0.9710 | -0.2917 | (-0.5611, 0.0722) | 0.9593 |
| proposed_vs_baseline_no_context | sentence_score | -0.2188 | (-0.3062, -0.0875) | 1.0000 | -0.2188 | (-0.3111, -0.0700) | 1.0000 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0151 | (-0.0228, 0.0572) | 0.2257 | 0.0151 | (-0.0298, 0.0653) | 0.2857 |
| proposed_vs_baseline_no_context | overall_quality | -0.0003 | (-0.0670, 0.0695) | 0.4970 | -0.0003 | (-0.0892, 0.0403) | 0.5233 |
| controlled_vs_proposed_raw | context_relevance | 0.2002 | (0.0818, 0.3120) | 0.0003 | 0.2002 | (0.1596, 0.2603) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.2363 | (0.0667, 0.4446) | 0.0000 | 0.2363 | (0.0800, 0.5429) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0478 | (-0.0224, 0.1142) | 0.0880 | 0.0478 | (-0.0460, 0.1021) | 0.1027 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2614 | (0.1136, 0.4091) | 0.0003 | 0.2614 | (0.1948, 0.3455) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0574 | (0.0176, 0.1054) | 0.0013 | 0.0574 | (0.0228, 0.0951) | 0.0003 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2798 | (0.0920, 0.5089) | 0.0000 | 0.2798 | (0.1000, 0.6508) | 0.0007 |
| controlled_vs_proposed_raw | persona_style | 0.0625 | (0.0000, 0.1875) | 0.3547 | 0.0625 | (0.0000, 0.2500) | 0.3197 |
| controlled_vs_proposed_raw | distinct1 | 0.0039 | (-0.0302, 0.0345) | 0.3913 | 0.0039 | (-0.0158, 0.0301) | 0.3650 |
| controlled_vs_proposed_raw | length_score | 0.1458 | (-0.1583, 0.4084) | 0.1597 | 0.1458 | (-0.2067, 0.3667) | 0.1820 |
| controlled_vs_proposed_raw | sentence_score | 0.1750 | (0.0000, 0.3062) | 0.0527 | 0.1750 | (-0.0583, 0.3000) | 0.0517 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0359 | (0.0117, 0.0624) | 0.0013 | 0.0359 | (0.0157, 0.0571) | 0.0007 |
| controlled_vs_proposed_raw | overall_quality | 0.1717 | (0.0799, 0.2724) | 0.0000 | 0.1717 | (0.1087, 0.3019) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2617 | (0.1820, 0.3456) | 0.0000 | 0.2617 | (0.1809, 0.3204) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1560 | (-0.0167, 0.3750) | 0.0577 | 0.1560 | (0.0364, 0.4746) | 0.0003 |
| controlled_vs_candidate_no_context | naturalness | 0.0420 | (-0.0347, 0.1089) | 0.1320 | 0.0420 | (-0.0562, 0.1043) | 0.1517 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3409 | (0.2386, 0.4432) | 0.0000 | 0.3409 | (0.2208, 0.4343) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0769 | (0.0393, 0.1334) | 0.0000 | 0.0769 | (0.0364, 0.1301) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1845 | (-0.0238, 0.4583) | 0.0543 | 0.1845 | (0.0455, 0.5516) | 0.0007 |
| controlled_vs_candidate_no_context | persona_style | 0.0417 | (0.0000, 0.1250) | 0.3413 | 0.0417 | (0.0000, 0.1667) | 0.3377 |
| controlled_vs_candidate_no_context | distinct1 | -0.0074 | (-0.0398, 0.0227) | 0.6830 | -0.0074 | (-0.0352, 0.0156) | 0.7470 |
| controlled_vs_candidate_no_context | length_score | 0.1833 | (-0.1418, 0.4583) | 0.1123 | 0.1833 | (-0.2867, 0.4905) | 0.1720 |
| controlled_vs_candidate_no_context | sentence_score | 0.0875 | (-0.0875, 0.2625) | 0.2027 | 0.0875 | (0.0000, 0.1909) | 0.3230 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0644 | (0.0223, 0.1078) | 0.0007 | 0.0644 | (0.0179, 0.1202) | 0.0003 |
| controlled_vs_candidate_no_context | overall_quality | 0.1724 | (0.0837, 0.2689) | 0.0000 | 0.1724 | (0.0977, 0.2810) | 0.0000 |
| controlled_vs_baseline_no_context | context_relevance | 0.2597 | (0.1965, 0.3321) | 0.0000 | 0.2597 | (0.1698, 0.3137) | 0.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 0.2090 | (0.0685, 0.4096) | 0.0000 | 0.2090 | (0.0586, 0.5000) | 0.0000 |
| controlled_vs_baseline_no_context | naturalness | -0.0490 | (-0.0980, -0.0096) | 0.9950 | -0.0490 | (-0.0675, -0.0258) | 1.0000 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3409 | (0.2500, 0.4432) | 0.0000 | 0.3409 | (0.2121, 0.4298) | 0.0000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0702 | (0.0332, 0.1230) | 0.0000 | 0.0702 | (0.0343, 0.1231) | 0.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2500 | (0.0833, 0.4792) | 0.0000 | 0.2500 | (0.0714, 0.5833) | 0.0130 |
| controlled_vs_baseline_no_context | persona_style | 0.0448 | (0.0000, 0.1281) | 0.1087 | 0.0448 | (0.0000, 0.1667) | 0.0777 |
| controlled_vs_baseline_no_context | distinct1 | -0.0375 | (-0.0819, 0.0033) | 0.9630 | -0.0375 | (-0.0620, 0.0160) | 0.9330 |
| controlled_vs_baseline_no_context | length_score | -0.1458 | (-0.3500, 0.0667) | 0.9187 | -0.1458 | (-0.3200, 0.0083) | 0.9627 |
| controlled_vs_baseline_no_context | sentence_score | -0.0437 | (-0.1750, 0.0875) | 0.8237 | -0.0437 | (-0.2000, 0.1167) | 0.8167 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0511 | (-0.0023, 0.1096) | 0.0323 | 0.0511 | (-0.0132, 0.1244) | 0.0583 |
| controlled_vs_baseline_no_context | overall_quality | 0.1714 | (0.1073, 0.2492) | 0.0000 | 0.1714 | (0.0924, 0.2676) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2597 | (0.1943, 0.3339) | 0.0000 | 0.2597 | (0.1702, 0.3137) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.2090 | (0.0679, 0.4094) | 0.0000 | 0.2090 | (0.0593, 0.5008) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0490 | (-0.0948, -0.0096) | 0.9947 | -0.0490 | (-0.0671, -0.0258) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3409 | (0.2500, 0.4432) | 0.0000 | 0.3409 | (0.2121, 0.4298) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0702 | (0.0334, 0.1231) | 0.0000 | 0.0702 | (0.0354, 0.1240) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2500 | (0.0833, 0.4792) | 0.0000 | 0.2500 | (0.0714, 0.5833) | 0.0117 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0448 | (0.0000, 0.1281) | 0.0923 | 0.0448 | (0.0000, 0.1667) | 0.0880 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0375 | (-0.0802, 0.0040) | 0.9597 | -0.0375 | (-0.0631, 0.0160) | 0.9357 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | -0.1458 | (-0.3500, 0.0667) | 0.9097 | -0.1458 | (-0.3200, 0.0083) | 0.9650 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | -0.0437 | (-0.1750, 0.0875) | 0.8047 | -0.0437 | (-0.2000, 0.1400) | 0.8177 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0511 | (-0.0035, 0.1056) | 0.0347 | 0.0511 | (-0.0132, 0.1157) | 0.0527 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1714 | (0.1051, 0.2532) | 0.0000 | 0.1714 | (0.0924, 0.2665) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 4 | 3 | 1 | 0.5625 | 0.5714 |
| proposed_vs_candidate_no_context | persona_consistency | 3 | 4 | 1 | 0.4375 | 0.4286 |
| proposed_vs_candidate_no_context | naturalness | 4 | 3 | 1 | 0.5625 | 0.5714 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 2 | 1 | 5 | 0.5625 | 0.6667 |
| proposed_vs_candidate_no_context | context_overlap | 5 | 2 | 1 | 0.6875 | 0.7143 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 3 | 3 | 2 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | persona_style | 0 | 1 | 7 | 0.4375 | 0.0000 |
| proposed_vs_candidate_no_context | distinct1 | 2 | 5 | 1 | 0.3125 | 0.2857 |
| proposed_vs_candidate_no_context | length_score | 3 | 4 | 1 | 0.4375 | 0.4286 |
| proposed_vs_candidate_no_context | sentence_score | 2 | 4 | 2 | 0.3750 | 0.3333 |
| proposed_vs_candidate_no_context | bertscore_f1 | 5 | 2 | 1 | 0.6875 | 0.7143 |
| proposed_vs_candidate_no_context | overall_quality | 3 | 4 | 1 | 0.4375 | 0.4286 |
| proposed_vs_baseline_no_context | context_relevance | 3 | 5 | 0 | 0.3750 | 0.3750 |
| proposed_vs_baseline_no_context | persona_consistency | 3 | 2 | 3 | 0.5625 | 0.6000 |
| proposed_vs_baseline_no_context | naturalness | 1 | 7 | 0 | 0.1250 | 0.1250 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 3 | 1 | 4 | 0.6250 | 0.7500 |
| proposed_vs_baseline_no_context | context_overlap | 4 | 4 | 0 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 2 | 1 | 5 | 0.5625 | 0.6667 |
| proposed_vs_baseline_no_context | persona_style | 1 | 1 | 6 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context | distinct1 | 0 | 7 | 1 | 0.0625 | 0.0000 |
| proposed_vs_baseline_no_context | length_score | 2 | 6 | 0 | 0.2500 | 0.2500 |
| proposed_vs_baseline_no_context | sentence_score | 0 | 5 | 3 | 0.1875 | 0.0000 |
| proposed_vs_baseline_no_context | bertscore_f1 | 4 | 4 | 0 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context | overall_quality | 4 | 4 | 0 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | context_relevance | 7 | 1 | 0 | 0.8750 | 0.8750 |
| controlled_vs_proposed_raw | persona_consistency | 6 | 0 | 2 | 0.8750 | 1.0000 |
| controlled_vs_proposed_raw | naturalness | 6 | 2 | 0 | 0.7500 | 0.7500 |
| controlled_vs_proposed_raw | context_keyword_coverage | 7 | 1 | 0 | 0.8750 | 0.8750 |
| controlled_vs_proposed_raw | context_overlap | 6 | 2 | 0 | 0.7500 | 0.7500 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 6 | 0 | 2 | 0.8750 | 1.0000 |
| controlled_vs_proposed_raw | persona_style | 1 | 0 | 7 | 0.5625 | 1.0000 |
| controlled_vs_proposed_raw | distinct1 | 5 | 3 | 0 | 0.6250 | 0.6250 |
| controlled_vs_proposed_raw | length_score | 5 | 3 | 0 | 0.6250 | 0.6250 |
| controlled_vs_proposed_raw | sentence_score | 5 | 1 | 2 | 0.7500 | 0.8333 |
| controlled_vs_proposed_raw | bertscore_f1 | 6 | 2 | 0 | 0.7500 | 0.7500 |
| controlled_vs_proposed_raw | overall_quality | 7 | 1 | 0 | 0.8750 | 0.8750 |
| controlled_vs_candidate_no_context | context_relevance | 8 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 5 | 2 | 1 | 0.6875 | 0.7143 |
| controlled_vs_candidate_no_context | naturalness | 6 | 2 | 0 | 0.7500 | 0.7500 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 8 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 8 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 5 | 2 | 1 | 0.6875 | 0.7143 |
| controlled_vs_candidate_no_context | persona_style | 1 | 0 | 7 | 0.5625 | 1.0000 |
| controlled_vs_candidate_no_context | distinct1 | 3 | 4 | 1 | 0.4375 | 0.4286 |
| controlled_vs_candidate_no_context | length_score | 6 | 2 | 0 | 0.7500 | 0.7500 |
| controlled_vs_candidate_no_context | sentence_score | 3 | 1 | 4 | 0.6250 | 0.7500 |
| controlled_vs_candidate_no_context | bertscore_f1 | 6 | 2 | 0 | 0.7500 | 0.7500 |
| controlled_vs_candidate_no_context | overall_quality | 8 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | context_relevance | 8 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 7 | 0 | 1 | 0.9375 | 1.0000 |
| controlled_vs_baseline_no_context | naturalness | 2 | 6 | 0 | 0.2500 | 0.2500 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 8 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | context_overlap | 8 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 6 | 0 | 2 | 0.8750 | 1.0000 |
| controlled_vs_baseline_no_context | persona_style | 2 | 0 | 6 | 0.6250 | 1.0000 |
| controlled_vs_baseline_no_context | distinct1 | 2 | 5 | 1 | 0.3125 | 0.2857 |
| controlled_vs_baseline_no_context | length_score | 2 | 6 | 0 | 0.2500 | 0.2500 |
| controlled_vs_baseline_no_context | sentence_score | 1 | 2 | 5 | 0.4375 | 0.3333 |
| controlled_vs_baseline_no_context | bertscore_f1 | 6 | 2 | 0 | 0.7500 | 0.7500 |
| controlled_vs_baseline_no_context | overall_quality | 8 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 8 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 7 | 0 | 1 | 0.9375 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 2 | 6 | 0 | 0.2500 | 0.2500 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 8 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 8 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 6 | 0 | 2 | 0.8750 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 2 | 0 | 6 | 0.6250 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | 2 | 5 | 1 | 0.3125 | 0.2857 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 2 | 6 | 0 | 0.2500 | 0.2500 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 1 | 2 | 5 | 0.4375 | 0.3333 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 6 | 2 | 0 | 0.7500 | 0.7500 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 8 | 0 | 0 | 1.0000 | 1.0000 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.1250 | 0.6250 | 0.3750 |
| proposed_contextual | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.3750 | 0.0000 | 0.0000 |
| baseline_no_context | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `5`
- Unique template signatures: `8`
- Template signature ratio: `1.0000`
- Effective sample size by source clustering: `4.00`
- Effective sample size by template-signature clustering: `8.00`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: enabled.

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
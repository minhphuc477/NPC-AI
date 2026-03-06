# Proposal Alignment Evaluation Report

- Run ID: `20260304T235244Z`
- Generated: `2026-03-04T23:55:41.228099+00:00`
- Scenarios: `artifacts\proposal_control_tuning\tuning_continue\20260304T235244Z\scenarios.jsonl`
- Scenario count: `10`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off
- `baseline_no_context`: model `phi3:mini`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2865 (0.2439, 0.3373) | 0.3207 (0.2640, 0.3885) | 0.8555 (0.8048, 0.9035) | 0.3778 (0.3509, 0.4046) | 0.0873 |
| proposed_contextual | 0.1117 (0.0379, 0.1937) | 0.1356 (0.0823, 0.1956) | 0.8062 (0.7718, 0.8436) | 0.2372 (0.1871, 0.2972) | 0.0802 |
| candidate_no_context | 0.0243 (0.0094, 0.0448) | 0.1059 (0.0500, 0.1740) | 0.8158 (0.7686, 0.8694) | 0.1901 (0.1598, 0.2246) | 0.0357 |
| baseline_no_context | 0.0573 (0.0291, 0.0881) | 0.1476 (0.1100, 0.1873) | 0.8595 (0.8266, 0.8965) | 0.2261 (0.2079, 0.2439) | 0.0619 |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0874 | 3.6045 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0297 | 0.2805 |
| proposed_vs_candidate_no_context | naturalness | -0.0096 | -0.0117 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.1159 | 6.3750 |
| proposed_vs_candidate_no_context | context_overlap | 0.0210 | 0.5454 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | persona_style | 0.1485 | 0.3749 |
| proposed_vs_candidate_no_context | distinct1 | 0.0015 | 0.0016 |
| proposed_vs_candidate_no_context | length_score | -0.1033 | -0.3069 |
| proposed_vs_candidate_no_context | sentence_score | 0.1050 | 0.1533 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0444 | 1.2440 |
| proposed_vs_candidate_no_context | overall_quality | 0.0471 | 0.2479 |
| proposed_vs_baseline_no_context | context_relevance | 0.0544 | 0.9500 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0120 | -0.0814 |
| proposed_vs_baseline_no_context | naturalness | -0.0533 | -0.0620 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0720 | 1.1585 |
| proposed_vs_baseline_no_context | context_overlap | 0.0134 | 0.2923 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0067 | -0.1667 |
| proposed_vs_baseline_no_context | persona_style | -0.0334 | -0.0578 |
| proposed_vs_baseline_no_context | distinct1 | -0.0233 | -0.0239 |
| proposed_vs_baseline_no_context | length_score | -0.2200 | -0.4853 |
| proposed_vs_baseline_no_context | sentence_score | 0.0000 | 0.0000 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0183 | 0.2956 |
| proposed_vs_baseline_no_context | overall_quality | 0.0111 | 0.0491 |
| controlled_vs_proposed_raw | context_relevance | 0.1748 | 1.5652 |
| controlled_vs_proposed_raw | persona_consistency | 0.1850 | 1.3643 |
| controlled_vs_proposed_raw | naturalness | 0.0493 | 0.0611 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2242 | 1.6723 |
| controlled_vs_proposed_raw | context_overlap | 0.0594 | 1.0011 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2095 | 6.2857 |
| controlled_vs_proposed_raw | persona_style | 0.0871 | 0.1599 |
| controlled_vs_proposed_raw | distinct1 | -0.0189 | -0.0199 |
| controlled_vs_proposed_raw | length_score | 0.2667 | 1.1429 |
| controlled_vs_proposed_raw | sentence_score | 0.0350 | 0.0443 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0072 | 0.0895 |
| controlled_vs_proposed_raw | overall_quality | 0.1405 | 0.5924 |
| controlled_vs_candidate_no_context | context_relevance | 0.2622 | 10.8116 |
| controlled_vs_candidate_no_context | persona_consistency | 0.2147 | 2.0275 |
| controlled_vs_candidate_no_context | naturalness | 0.0397 | 0.0487 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3402 | 18.7083 |
| controlled_vs_candidate_no_context | context_overlap | 0.0804 | 2.0925 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2095 | 6.2857 |
| controlled_vs_candidate_no_context | persona_style | 0.2356 | 0.5947 |
| controlled_vs_candidate_no_context | distinct1 | -0.0174 | -0.0184 |
| controlled_vs_candidate_no_context | length_score | 0.1633 | 0.4851 |
| controlled_vs_candidate_no_context | sentence_score | 0.1400 | 0.2044 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0516 | 1.4449 |
| controlled_vs_candidate_no_context | overall_quality | 0.1877 | 0.9872 |
| controlled_vs_baseline_no_context | context_relevance | 0.2292 | 4.0023 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1730 | 1.1719 |
| controlled_vs_baseline_no_context | naturalness | -0.0040 | -0.0047 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.2962 | 4.7683 |
| controlled_vs_baseline_no_context | context_overlap | 0.0729 | 1.5859 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2029 | 5.0714 |
| controlled_vs_baseline_no_context | persona_style | 0.0537 | 0.0928 |
| controlled_vs_baseline_no_context | distinct1 | -0.0422 | -0.0433 |
| controlled_vs_baseline_no_context | length_score | 0.0467 | 0.1029 |
| controlled_vs_baseline_no_context | sentence_score | 0.0350 | 0.0443 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0255 | 0.4116 |
| controlled_vs_baseline_no_context | overall_quality | 0.1516 | 0.6706 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2292 | 4.0023 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1730 | 1.1719 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0040 | -0.0047 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.2962 | 4.7683 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0729 | 1.5859 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2029 | 5.0714 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0537 | 0.0928 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0422 | -0.0433 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.0467 | 0.1029 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0350 | 0.0443 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0255 | 0.4116 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1516 | 0.6706 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0874 | (0.0268, 0.1611) | 0.0000 | 0.0874 | (0.0126, 0.1839) | 0.0000 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0297 | (0.0000, 0.0671) | 0.0277 | 0.0297 | (0.0028, 0.0750) | 0.0187 |
| proposed_vs_candidate_no_context | naturalness | -0.0096 | (-0.0442, 0.0312) | 0.6983 | -0.0096 | (-0.0468, 0.0347) | 0.7073 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.1159 | (0.0341, 0.2159) | 0.0003 | 0.1159 | (0.0165, 0.2376) | 0.0043 |
| proposed_vs_candidate_no_context | context_overlap | 0.0210 | (0.0053, 0.0395) | 0.0000 | 0.0210 | (0.0034, 0.0422) | 0.0003 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | persona_style | 0.1485 | (0.0000, 0.3285) | 0.0307 | 0.1485 | (0.0139, 0.3597) | 0.0160 |
| proposed_vs_candidate_no_context | distinct1 | 0.0015 | (-0.0191, 0.0258) | 0.4667 | 0.0015 | (-0.0227, 0.0240) | 0.4810 |
| proposed_vs_candidate_no_context | length_score | -0.1033 | (-0.2967, 0.0933) | 0.8590 | -0.1033 | (-0.3000, 0.1100) | 0.8433 |
| proposed_vs_candidate_no_context | sentence_score | 0.1050 | (0.0350, 0.2100) | 0.0237 | 0.1050 | (0.0269, 0.2188) | 0.0230 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0444 | (0.0187, 0.0704) | 0.0000 | 0.0444 | (0.0163, 0.0784) | 0.0000 |
| proposed_vs_candidate_no_context | overall_quality | 0.0471 | (0.0158, 0.0786) | 0.0000 | 0.0471 | (0.0124, 0.0890) | 0.0000 |
| proposed_vs_baseline_no_context | context_relevance | 0.0544 | (-0.0120, 0.1418) | 0.0783 | 0.0544 | (-0.0179, 0.1475) | 0.1077 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0120 | (-0.0867, 0.0673) | 0.6190 | -0.0120 | (-0.0824, 0.0682) | 0.6073 |
| proposed_vs_baseline_no_context | naturalness | -0.0533 | (-0.1037, 0.0003) | 0.9733 | -0.0533 | (-0.1108, 0.0139) | 0.9417 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0720 | (-0.0189, 0.1902) | 0.1120 | 0.0720 | (-0.0260, 0.1902) | 0.1467 |
| proposed_vs_baseline_no_context | context_overlap | 0.0134 | (-0.0053, 0.0334) | 0.0827 | 0.0134 | (-0.0074, 0.0373) | 0.1260 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0067 | (-0.0800, 0.0800) | 0.6083 | -0.0067 | (-0.0727, 0.0741) | 0.5947 |
| proposed_vs_baseline_no_context | persona_style | -0.0334 | (-0.1385, 0.0821) | 0.7350 | -0.0334 | (-0.1562, 0.1169) | 0.6987 |
| proposed_vs_baseline_no_context | distinct1 | -0.0233 | (-0.0564, 0.0098) | 0.9203 | -0.0233 | (-0.0565, 0.0089) | 0.9147 |
| proposed_vs_baseline_no_context | length_score | -0.2200 | (-0.4033, -0.0300) | 0.9903 | -0.2200 | (-0.4154, 0.0083) | 0.9703 |
| proposed_vs_baseline_no_context | sentence_score | 0.0000 | (-0.1400, 0.1400) | 0.6120 | 0.0000 | (-0.1273, 0.1500) | 0.6020 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0183 | (-0.0245, 0.0637) | 0.2117 | 0.0183 | (-0.0180, 0.0534) | 0.1837 |
| proposed_vs_baseline_no_context | overall_quality | 0.0111 | (-0.0433, 0.0705) | 0.3957 | 0.0111 | (-0.0465, 0.0823) | 0.3887 |
| controlled_vs_proposed_raw | context_relevance | 0.1748 | (0.0949, 0.2393) | 0.0003 | 0.1748 | (0.0775, 0.2483) | 0.0007 |
| controlled_vs_proposed_raw | persona_consistency | 0.1850 | (0.0910, 0.2710) | 0.0000 | 0.1850 | (0.1161, 0.2611) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0493 | (0.0060, 0.0877) | 0.0153 | 0.0493 | (0.0067, 0.0931) | 0.0133 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2242 | (0.1182, 0.3152) | 0.0000 | 0.2242 | (0.0992, 0.3223) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0594 | (0.0369, 0.0800) | 0.0000 | 0.0594 | (0.0311, 0.0841) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2095 | (0.1029, 0.2990) | 0.0000 | 0.2095 | (0.1400, 0.2910) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.0871 | (-0.0029, 0.2423) | 0.1027 | 0.0871 | (-0.0032, 0.2017) | 0.0993 |
| controlled_vs_proposed_raw | distinct1 | -0.0189 | (-0.0502, 0.0150) | 0.8730 | -0.0189 | (-0.0540, 0.0230) | 0.8157 |
| controlled_vs_proposed_raw | length_score | 0.2667 | (0.0500, 0.4767) | 0.0100 | 0.2667 | (0.0467, 0.4924) | 0.0103 |
| controlled_vs_proposed_raw | sentence_score | 0.0350 | (-0.1050, 0.1750) | 0.4133 | 0.0350 | (-0.1273, 0.1750) | 0.4093 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0072 | (-0.0374, 0.0440) | 0.3440 | 0.0072 | (-0.0363, 0.0518) | 0.3740 |
| controlled_vs_proposed_raw | overall_quality | 0.1405 | (0.0828, 0.1905) | 0.0000 | 0.1405 | (0.0817, 0.1918) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2622 | (0.2240, 0.3119) | 0.0000 | 0.2622 | (0.2262, 0.3188) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.2147 | (0.1116, 0.3000) | 0.0000 | 0.2147 | (0.1354, 0.2982) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.0397 | (-0.0254, 0.1050) | 0.1113 | 0.0397 | (-0.0315, 0.1064) | 0.1107 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3402 | (0.2909, 0.4114) | 0.0000 | 0.3402 | (0.2893, 0.4119) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0804 | (0.0646, 0.0981) | 0.0000 | 0.0804 | (0.0623, 0.0998) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2095 | (0.1033, 0.3052) | 0.0000 | 0.2095 | (0.1400, 0.2841) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.2356 | (0.0500, 0.4523) | 0.0077 | 0.2356 | (0.0532, 0.4849) | 0.0210 |
| controlled_vs_candidate_no_context | distinct1 | -0.0174 | (-0.0512, 0.0167) | 0.8350 | -0.0174 | (-0.0554, 0.0220) | 0.8030 |
| controlled_vs_candidate_no_context | length_score | 0.1633 | (-0.1267, 0.4467) | 0.1350 | 0.1633 | (-0.1259, 0.4501) | 0.1193 |
| controlled_vs_candidate_no_context | sentence_score | 0.1400 | (0.0350, 0.2450) | 0.0030 | 0.1400 | (0.0389, 0.2333) | 0.0023 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0516 | (0.0105, 0.0960) | 0.0090 | 0.0516 | (0.0133, 0.1054) | 0.0010 |
| controlled_vs_candidate_no_context | overall_quality | 0.1877 | (0.1463, 0.2236) | 0.0000 | 0.1877 | (0.1530, 0.2251) | 0.0000 |
| controlled_vs_baseline_no_context | context_relevance | 0.2292 | (0.1883, 0.2722) | 0.0000 | 0.2292 | (0.1930, 0.2687) | 0.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1730 | (0.1200, 0.2280) | 0.0000 | 0.1730 | (0.1253, 0.2315) | 0.0000 |
| controlled_vs_baseline_no_context | naturalness | -0.0040 | (-0.0785, 0.0698) | 0.5227 | -0.0040 | (-0.0838, 0.0791) | 0.5890 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.2962 | (0.2364, 0.3545) | 0.0000 | 0.2962 | (0.2386, 0.3594) | 0.0000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0729 | (0.0587, 0.0859) | 0.0000 | 0.0729 | (0.0584, 0.0878) | 0.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2029 | (0.1400, 0.2748) | 0.0000 | 0.2029 | (0.1444, 0.2789) | 0.0000 |
| controlled_vs_baseline_no_context | persona_style | 0.0537 | (-0.0333, 0.1480) | 0.1573 | 0.0537 | (0.0000, 0.1419) | 0.1010 |
| controlled_vs_baseline_no_context | distinct1 | -0.0422 | (-0.0729, -0.0095) | 0.9933 | -0.0422 | (-0.0688, -0.0057) | 0.9850 |
| controlled_vs_baseline_no_context | length_score | 0.0467 | (-0.3201, 0.4000) | 0.3907 | 0.0467 | (-0.3515, 0.4182) | 0.3950 |
| controlled_vs_baseline_no_context | sentence_score | 0.0350 | (-0.0700, 0.1400) | 0.3797 | 0.0350 | (0.0000, 0.1312) | 0.3493 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0255 | (-0.0285, 0.0832) | 0.1857 | 0.0255 | (-0.0333, 0.0969) | 0.2357 |
| controlled_vs_baseline_no_context | overall_quality | 0.1516 | (0.1295, 0.1754) | 0.0000 | 0.1516 | (0.1295, 0.1802) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2292 | (0.1888, 0.2719) | 0.0000 | 0.2292 | (0.1915, 0.2735) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1730 | (0.1173, 0.2300) | 0.0000 | 0.1730 | (0.1235, 0.2305) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0040 | (-0.0799, 0.0703) | 0.5340 | -0.0040 | (-0.0909, 0.0806) | 0.5850 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.2962 | (0.2364, 0.3561) | 0.0000 | 0.2962 | (0.2455, 0.3598) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0729 | (0.0588, 0.0863) | 0.0000 | 0.0729 | (0.0597, 0.0876) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2029 | (0.1400, 0.2719) | 0.0000 | 0.2029 | (0.1455, 0.2732) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0537 | (-0.0333, 0.1481) | 0.1527 | 0.0537 | (0.0000, 0.1586) | 0.0950 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0422 | (-0.0724, -0.0090) | 0.9957 | -0.0422 | (-0.0701, -0.0056) | 0.9883 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.0467 | (-0.3233, 0.3934) | 0.3987 | 0.0467 | (-0.3556, 0.4194) | 0.3940 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0350 | (-0.0700, 0.1400) | 0.3813 | 0.0350 | (0.0000, 0.1312) | 0.3473 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0255 | (-0.0282, 0.0791) | 0.1913 | 0.0255 | (-0.0335, 0.0956) | 0.2380 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1516 | (0.1293, 0.1747) | 0.0000 | 0.1516 | (0.1303, 0.1803) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 6 | 0 | 4 | 0.8000 | 1.0000 |
| proposed_vs_candidate_no_context | persona_consistency | 3 | 0 | 7 | 0.6500 | 1.0000 |
| proposed_vs_candidate_no_context | naturalness | 2 | 4 | 4 | 0.4000 | 0.3333 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 5 | 0 | 5 | 0.7500 | 1.0000 |
| proposed_vs_candidate_no_context | context_overlap | 6 | 0 | 4 | 0.8000 | 1.0000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0 | 0 | 10 | 0.5000 | nan |
| proposed_vs_candidate_no_context | persona_style | 3 | 0 | 7 | 0.6500 | 1.0000 |
| proposed_vs_candidate_no_context | distinct1 | 2 | 3 | 5 | 0.4500 | 0.4000 |
| proposed_vs_candidate_no_context | length_score | 1 | 5 | 4 | 0.3000 | 0.1667 |
| proposed_vs_candidate_no_context | sentence_score | 3 | 0 | 7 | 0.6500 | 1.0000 |
| proposed_vs_candidate_no_context | bertscore_f1 | 6 | 0 | 4 | 0.8000 | 1.0000 |
| proposed_vs_candidate_no_context | overall_quality | 6 | 0 | 4 | 0.8000 | 1.0000 |
| proposed_vs_baseline_no_context | context_relevance | 4 | 6 | 0 | 0.4000 | 0.4000 |
| proposed_vs_baseline_no_context | persona_consistency | 2 | 4 | 4 | 0.4000 | 0.3333 |
| proposed_vs_baseline_no_context | naturalness | 4 | 6 | 0 | 0.4000 | 0.4000 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 4 | 2 | 4 | 0.6000 | 0.6667 |
| proposed_vs_baseline_no_context | context_overlap | 6 | 4 | 0 | 0.6000 | 0.6000 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 1 | 2 | 7 | 0.4500 | 0.3333 |
| proposed_vs_baseline_no_context | persona_style | 1 | 3 | 6 | 0.4000 | 0.2500 |
| proposed_vs_baseline_no_context | distinct1 | 3 | 6 | 1 | 0.3500 | 0.3333 |
| proposed_vs_baseline_no_context | length_score | 3 | 7 | 0 | 0.3000 | 0.3000 |
| proposed_vs_baseline_no_context | sentence_score | 2 | 2 | 6 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context | bertscore_f1 | 5 | 5 | 0 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context | overall_quality | 4 | 6 | 0 | 0.4000 | 0.4000 |
| controlled_vs_proposed_raw | context_relevance | 9 | 1 | 0 | 0.9000 | 0.9000 |
| controlled_vs_proposed_raw | persona_consistency | 9 | 1 | 0 | 0.9000 | 0.9000 |
| controlled_vs_proposed_raw | naturalness | 7 | 3 | 0 | 0.7000 | 0.7000 |
| controlled_vs_proposed_raw | context_keyword_coverage | 9 | 1 | 0 | 0.9000 | 0.9000 |
| controlled_vs_proposed_raw | context_overlap | 9 | 1 | 0 | 0.9000 | 0.9000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 9 | 1 | 0 | 0.9000 | 0.9000 |
| controlled_vs_proposed_raw | persona_style | 2 | 1 | 7 | 0.5500 | 0.6667 |
| controlled_vs_proposed_raw | distinct1 | 2 | 6 | 2 | 0.3000 | 0.2500 |
| controlled_vs_proposed_raw | length_score | 7 | 3 | 0 | 0.7000 | 0.7000 |
| controlled_vs_proposed_raw | sentence_score | 3 | 2 | 5 | 0.5500 | 0.6000 |
| controlled_vs_proposed_raw | bertscore_f1 | 5 | 5 | 0 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | overall_quality | 9 | 1 | 0 | 0.9000 | 0.9000 |
| controlled_vs_candidate_no_context | context_relevance | 10 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 9 | 1 | 0 | 0.9000 | 0.9000 |
| controlled_vs_candidate_no_context | naturalness | 5 | 5 | 0 | 0.5000 | 0.5000 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 10 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 10 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 9 | 1 | 0 | 0.9000 | 0.9000 |
| controlled_vs_candidate_no_context | persona_style | 4 | 0 | 6 | 0.7000 | 1.0000 |
| controlled_vs_candidate_no_context | distinct1 | 3 | 5 | 2 | 0.4000 | 0.3750 |
| controlled_vs_candidate_no_context | length_score | 6 | 4 | 0 | 0.6000 | 0.6000 |
| controlled_vs_candidate_no_context | sentence_score | 4 | 0 | 6 | 0.7000 | 1.0000 |
| controlled_vs_candidate_no_context | bertscore_f1 | 8 | 2 | 0 | 0.8000 | 0.8000 |
| controlled_vs_candidate_no_context | overall_quality | 10 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | context_relevance | 10 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 9 | 0 | 1 | 0.9500 | 1.0000 |
| controlled_vs_baseline_no_context | naturalness | 5 | 5 | 0 | 0.5000 | 0.5000 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 10 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | context_overlap | 10 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 9 | 0 | 1 | 0.9500 | 1.0000 |
| controlled_vs_baseline_no_context | persona_style | 2 | 1 | 7 | 0.5500 | 0.6667 |
| controlled_vs_baseline_no_context | distinct1 | 2 | 8 | 0 | 0.2000 | 0.2000 |
| controlled_vs_baseline_no_context | length_score | 5 | 5 | 0 | 0.5000 | 0.5000 |
| controlled_vs_baseline_no_context | sentence_score | 2 | 1 | 7 | 0.5500 | 0.6667 |
| controlled_vs_baseline_no_context | bertscore_f1 | 5 | 5 | 0 | 0.5000 | 0.5000 |
| controlled_vs_baseline_no_context | overall_quality | 10 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 10 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 9 | 0 | 1 | 0.9500 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 5 | 5 | 0 | 0.5000 | 0.5000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 10 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 10 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 9 | 0 | 1 | 0.9500 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 2 | 1 | 7 | 0.5500 | 0.6667 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | 2 | 8 | 0 | 0.2000 | 0.2000 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 5 | 5 | 0 | 0.5000 | 0.5000 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 2 | 1 | 7 | 0.5500 | 0.6667 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 5 | 5 | 0 | 0.5000 | 0.5000 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 10 | 0 | 0 | 1.0000 | 1.0000 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.2000 | 0.5000 | 0.5000 |
| proposed_contextual | 0.0000 | 0.0000 | 0.4000 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 |
| baseline_no_context | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `7`
- Unique template signatures: `9`
- Template signature ratio: `0.9000`
- Effective sample size by source clustering: `6.25`
- Effective sample size by template-signature clustering: `8.33`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: enabled.

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
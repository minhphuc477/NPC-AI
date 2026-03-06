# Proposal Alignment Evaluation Report

- Run ID: `20260305T235105Z`
- Generated: `2026-03-05T23:53:39.155609+00:00`
- Scenarios: `artifacts\proposal_control_tuning\auto_tune_v4\20260305T235104Z\train_runs\trial_000\seed_29\20260305T235105Z\scenarios.jsonl`
- Scenario count: `16`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_tuned`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2463 (0.2047, 0.2982) | 0.4056 (0.3297, 0.4725) | 0.8903 (0.8499, 0.9235) | 0.4335 (0.4014, 0.4673) | n/a |
| proposed_contextual_controlled_tuned | 0.2234 (0.1890, 0.2568) | 0.2933 (0.2396, 0.3448) | 0.9151 (0.8957, 0.9328) | 0.3865 (0.3650, 0.4096) | n/a |
| proposed_contextual | 0.0829 (0.0344, 0.1503) | 0.1673 (0.1126, 0.2338) | 0.8117 (0.7780, 0.8461) | 0.2544 (0.2109, 0.3089) | n/a |
| candidate_no_context | 0.0204 (0.0107, 0.0331) | 0.2668 (0.1841, 0.3525) | 0.8663 (0.8210, 0.9106) | 0.2728 (0.2344, 0.3116) | n/a |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0626 | 3.0663 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0995 | -0.3729 |
| proposed_vs_candidate_no_context | naturalness | -0.0547 | -0.0631 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0852 | 7.5000 |
| proposed_vs_candidate_no_context | context_overlap | 0.0096 | 0.2323 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.1161 | -0.6190 |
| proposed_vs_candidate_no_context | persona_style | -0.0332 | -0.0569 |
| proposed_vs_candidate_no_context | distinct1 | -0.0098 | -0.0103 |
| proposed_vs_candidate_no_context | length_score | -0.2208 | -0.4362 |
| proposed_vs_candidate_no_context | sentence_score | -0.0656 | -0.0795 |
| proposed_vs_candidate_no_context | overall_quality | -0.0184 | -0.0675 |
| controlled_vs_proposed_raw | context_relevance | 0.1634 | 1.9697 |
| controlled_vs_proposed_raw | persona_consistency | 0.2383 | 1.4242 |
| controlled_vs_proposed_raw | naturalness | 0.0786 | 0.0968 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2125 | 2.2000 |
| controlled_vs_proposed_raw | context_overlap | 0.0488 | 0.9541 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2676 | 3.7458 |
| controlled_vs_proposed_raw | persona_style | 0.1211 | 0.2199 |
| controlled_vs_proposed_raw | distinct1 | 0.0042 | 0.0045 |
| controlled_vs_proposed_raw | length_score | 0.3187 | 1.1168 |
| controlled_vs_proposed_raw | sentence_score | 0.1312 | 0.1728 |
| controlled_vs_proposed_raw | overall_quality | 0.1791 | 0.7039 |
| controlled_vs_candidate_no_context | context_relevance | 0.2259 | 11.0755 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1388 | 0.5201 |
| controlled_vs_candidate_no_context | naturalness | 0.0239 | 0.0276 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.2977 | 26.2000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0584 | 1.4081 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1515 | 0.8079 |
| controlled_vs_candidate_no_context | persona_style | 0.0879 | 0.1505 |
| controlled_vs_candidate_no_context | distinct1 | -0.0056 | -0.0058 |
| controlled_vs_candidate_no_context | length_score | 0.0979 | 0.1934 |
| controlled_vs_candidate_no_context | sentence_score | 0.0656 | 0.0795 |
| controlled_vs_candidate_no_context | overall_quality | 0.1607 | 0.5889 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0229 | -0.0929 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.1122 | -0.2767 |
| controlled_alt_vs_controlled_default | naturalness | 0.0248 | 0.0279 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0284 | -0.0919 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0100 | -0.1001 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.1286 | -0.3793 |
| controlled_alt_vs_controlled_default | persona_style | -0.0469 | -0.0698 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0018 | -0.0019 |
| controlled_alt_vs_controlled_default | length_score | 0.1167 | 0.1931 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0219 | 0.0246 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0469 | -0.1083 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1405 | 1.6937 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1260 | 0.7533 |
| controlled_alt_vs_proposed_raw | naturalness | 0.1034 | 0.1274 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.1841 | 1.9059 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0388 | 0.7585 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1390 | 1.9458 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0742 | 0.1348 |
| controlled_alt_vs_proposed_raw | distinct1 | 0.0025 | 0.0026 |
| controlled_alt_vs_proposed_raw | length_score | 0.4354 | 1.5255 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1531 | 0.2016 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1321 | 0.5194 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2030 | 9.9535 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.0265 | 0.0995 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0487 | 0.0562 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.2693 | 23.7000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0484 | 1.1670 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.0229 | 0.1222 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0410 | 0.0702 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0074 | -0.0077 |
| controlled_alt_vs_candidate_no_context | length_score | 0.2146 | 0.4239 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.0875 | 0.1061 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1137 | 0.4169 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0626 | (0.0094, 0.1346) | 0.0047 | 0.0626 | (0.0108, 0.1177) | 0.0123 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0995 | (-0.1894, -0.0224) | 0.9990 | -0.0995 | (-0.1714, -0.0353) | 1.0000 |
| proposed_vs_candidate_no_context | naturalness | -0.0547 | (-0.0958, -0.0198) | 1.0000 | -0.0547 | (-0.0784, -0.0236) | 1.0000 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0852 | (0.0170, 0.1761) | 0.0050 | 0.0852 | (0.0170, 0.1576) | 0.0117 |
| proposed_vs_candidate_no_context | context_overlap | 0.0096 | (-0.0078, 0.0298) | 0.1610 | 0.0096 | (-0.0058, 0.0320) | 0.3387 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.1161 | (-0.2307, -0.0298) | 1.0000 | -0.1161 | (-0.2092, -0.0378) | 1.0000 |
| proposed_vs_candidate_no_context | persona_style | -0.0332 | (-0.1309, 0.0391) | 0.7833 | -0.0332 | (-0.1234, 0.0469) | 0.7400 |
| proposed_vs_candidate_no_context | distinct1 | -0.0098 | (-0.0263, 0.0047) | 0.8947 | -0.0098 | (-0.0278, 0.0081) | 0.8390 |
| proposed_vs_candidate_no_context | length_score | -0.2208 | (-0.3688, -0.0854) | 1.0000 | -0.2208 | (-0.3286, -0.0941) | 1.0000 |
| proposed_vs_candidate_no_context | sentence_score | -0.0656 | (-0.1531, 0.0219) | 0.9557 | -0.0656 | (-0.1250, -0.0194) | 1.0000 |
| proposed_vs_candidate_no_context | overall_quality | -0.0184 | (-0.0612, 0.0184) | 0.8130 | -0.0184 | (-0.0399, -0.0032) | 0.9857 |
| controlled_vs_proposed_raw | context_relevance | 0.1634 | (0.0795, 0.2354) | 0.0003 | 0.1634 | (0.1052, 0.2241) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.2383 | (0.1527, 0.3269) | 0.0000 | 0.2383 | (0.1858, 0.2803) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0786 | (0.0332, 0.1253) | 0.0010 | 0.0786 | (0.0365, 0.1159) | 0.0000 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2125 | (0.1057, 0.3068) | 0.0007 | 0.2125 | (0.1409, 0.2879) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0488 | (0.0220, 0.0735) | 0.0007 | 0.0488 | (0.0220, 0.0770) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2676 | (0.1530, 0.3798) | 0.0000 | 0.2676 | (0.1888, 0.3387) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.1211 | (0.0234, 0.2363) | 0.0030 | 0.1211 | (0.0000, 0.2674) | 0.0783 |
| controlled_vs_proposed_raw | distinct1 | 0.0042 | (-0.0186, 0.0260) | 0.3577 | 0.0042 | (-0.0133, 0.0176) | 0.2927 |
| controlled_vs_proposed_raw | length_score | 0.3188 | (0.1354, 0.4980) | 0.0000 | 0.3188 | (0.1078, 0.5333) | 0.0000 |
| controlled_vs_proposed_raw | sentence_score | 0.1312 | (0.0000, 0.2406) | 0.0317 | 0.1312 | (0.0656, 0.1750) | 0.0000 |
| controlled_vs_proposed_raw | overall_quality | 0.1791 | (0.1193, 0.2395) | 0.0000 | 0.1791 | (0.1293, 0.2240) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2259 | (0.1833, 0.2754) | 0.0000 | 0.2259 | (0.1879, 0.2555) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1388 | (0.0402, 0.2384) | 0.0020 | 0.1388 | (0.0145, 0.2369) | 0.0100 |
| controlled_vs_candidate_no_context | naturalness | 0.0239 | (-0.0263, 0.0762) | 0.1830 | 0.0239 | (-0.0323, 0.0659) | 0.1770 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.2977 | (0.2432, 0.3636) | 0.0000 | 0.2977 | (0.2532, 0.3323) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0584 | (0.0400, 0.0776) | 0.0000 | 0.0584 | (0.0343, 0.0814) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1515 | (0.0310, 0.2727) | 0.0073 | 0.1515 | (-0.0204, 0.2912) | 0.0313 |
| controlled_vs_candidate_no_context | persona_style | 0.0879 | (0.0156, 0.1855) | 0.0133 | 0.0879 | (0.0000, 0.1758) | 0.0750 |
| controlled_vs_candidate_no_context | distinct1 | -0.0056 | (-0.0253, 0.0142) | 0.7163 | -0.0056 | (-0.0244, 0.0127) | 0.6943 |
| controlled_vs_candidate_no_context | length_score | 0.0979 | (-0.1001, 0.2917) | 0.1673 | 0.0979 | (-0.1405, 0.2669) | 0.1740 |
| controlled_vs_candidate_no_context | sentence_score | 0.0656 | (-0.0219, 0.1531) | 0.1107 | 0.0656 | (-0.0467, 0.1400) | 0.1410 |
| controlled_vs_candidate_no_context | overall_quality | 0.1607 | (0.1154, 0.2081) | 0.0000 | 0.1607 | (0.0984, 0.2169) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0229 | (-0.0661, 0.0239) | 0.8377 | -0.0229 | (-0.0592, 0.0399) | 0.7913 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.1122 | (-0.1871, -0.0401) | 1.0000 | -0.1122 | (-0.1936, -0.0252) | 1.0000 |
| controlled_alt_vs_controlled_default | naturalness | 0.0248 | (-0.0094, 0.0669) | 0.0923 | 0.0248 | (-0.0215, 0.0734) | 0.1913 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0284 | (-0.0909, 0.0341) | 0.8240 | -0.0284 | (-0.0749, 0.0485) | 0.7700 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0100 | (-0.0299, 0.0100) | 0.8413 | -0.0100 | (-0.0340, 0.0097) | 0.7507 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.1286 | (-0.2232, -0.0384) | 0.9977 | -0.1286 | (-0.2320, -0.0190) | 1.0000 |
| controlled_alt_vs_controlled_default | persona_style | -0.0469 | (-0.0938, 0.0000) | 1.0000 | -0.0469 | (-0.1500, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0018 | (-0.0216, 0.0206) | 0.5587 | -0.0018 | (-0.0249, 0.0186) | 0.5773 |
| controlled_alt_vs_controlled_default | length_score | 0.1167 | (-0.0292, 0.2771) | 0.0667 | 0.1167 | (-0.0810, 0.3352) | 0.2143 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0219 | (-0.0656, 0.1094) | 0.4123 | 0.0219 | (-0.0700, 0.1167) | 0.4277 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0469 | (-0.0788, -0.0152) | 0.9990 | -0.0469 | (-0.0774, -0.0184) | 1.0000 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1405 | (0.0747, 0.1988) | 0.0000 | 0.1405 | (0.0613, 0.2197) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1260 | (0.0650, 0.1865) | 0.0003 | 0.1260 | (0.0620, 0.1943) | 0.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 0.1034 | (0.0502, 0.1507) | 0.0003 | 0.1034 | (0.0542, 0.1496) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.1841 | (0.0943, 0.2614) | 0.0000 | 0.1841 | (0.0784, 0.2898) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0388 | (0.0203, 0.0545) | 0.0000 | 0.0388 | (0.0214, 0.0606) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1390 | (0.0729, 0.2119) | 0.0000 | 0.1390 | (0.0792, 0.1964) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0742 | (-0.0234, 0.2051) | 0.1060 | 0.0742 | (-0.0500, 0.2396) | 0.3343 |
| controlled_alt_vs_proposed_raw | distinct1 | 0.0025 | (-0.0226, 0.0257) | 0.4187 | 0.0025 | (-0.0232, 0.0223) | 0.3963 |
| controlled_alt_vs_proposed_raw | length_score | 0.4354 | (0.2500, 0.6209) | 0.0000 | 0.4354 | (0.2571, 0.6125) | 0.0000 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1531 | (0.0219, 0.2625) | 0.0120 | 0.1531 | (0.0000, 0.2763) | 0.0353 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1321 | (0.0812, 0.1764) | 0.0000 | 0.1321 | (0.0801, 0.1787) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2030 | (0.1690, 0.2372) | 0.0000 | 0.2030 | (0.1556, 0.2464) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.0265 | (-0.0489, 0.1007) | 0.2550 | 0.0265 | (-0.0585, 0.0912) | 0.2633 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0487 | (-0.0076, 0.1067) | 0.0487 | 0.0487 | (-0.0241, 0.1258) | 0.1353 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.2693 | (0.2239, 0.3136) | 0.0000 | 0.2693 | (0.2086, 0.3251) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0484 | (0.0351, 0.0611) | 0.0000 | 0.0484 | (0.0339, 0.0616) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.0229 | (-0.0699, 0.1072) | 0.3140 | 0.0229 | (-0.0746, 0.0958) | 0.3043 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0410 | (0.0000, 0.1230) | 0.3733 | 0.0410 | (0.0000, 0.1094) | 0.3370 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0074 | (-0.0292, 0.0151) | 0.7350 | -0.0074 | (-0.0364, 0.0182) | 0.6600 |
| controlled_alt_vs_candidate_no_context | length_score | 0.2146 | (0.0020, 0.4313) | 0.0250 | 0.2146 | (-0.0381, 0.4815) | 0.0650 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.0875 | (-0.0219, 0.1969) | 0.0987 | 0.0875 | (-0.1167, 0.2579) | 0.2123 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1137 | (0.0752, 0.1507) | 0.0000 | 0.1137 | (0.0484, 0.1571) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 7 | 3 | 6 | 0.6250 | 0.7000 |
| proposed_vs_candidate_no_context | persona_consistency | 1 | 6 | 9 | 0.3438 | 0.1429 |
| proposed_vs_candidate_no_context | naturalness | 1 | 9 | 6 | 0.2500 | 0.1000 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 6 | 1 | 9 | 0.6562 | 0.8571 |
| proposed_vs_candidate_no_context | context_overlap | 7 | 3 | 6 | 0.6250 | 0.7000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0 | 5 | 11 | 0.3438 | 0.0000 |
| proposed_vs_candidate_no_context | persona_style | 1 | 2 | 13 | 0.4688 | 0.3333 |
| proposed_vs_candidate_no_context | distinct1 | 2 | 5 | 9 | 0.4062 | 0.2857 |
| proposed_vs_candidate_no_context | length_score | 1 | 9 | 6 | 0.2500 | 0.1000 |
| proposed_vs_candidate_no_context | sentence_score | 1 | 4 | 11 | 0.4062 | 0.2000 |
| proposed_vs_candidate_no_context | overall_quality | 4 | 6 | 6 | 0.4375 | 0.4000 |
| controlled_vs_proposed_raw | context_relevance | 14 | 2 | 0 | 0.8750 | 0.8750 |
| controlled_vs_proposed_raw | persona_consistency | 13 | 1 | 2 | 0.8750 | 0.9286 |
| controlled_vs_proposed_raw | naturalness | 11 | 5 | 0 | 0.6875 | 0.6875 |
| controlled_vs_proposed_raw | context_keyword_coverage | 13 | 1 | 2 | 0.8750 | 0.9286 |
| controlled_vs_proposed_raw | context_overlap | 12 | 4 | 0 | 0.7500 | 0.7500 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 13 | 1 | 2 | 0.8750 | 0.9286 |
| controlled_vs_proposed_raw | persona_style | 5 | 0 | 11 | 0.6562 | 1.0000 |
| controlled_vs_proposed_raw | distinct1 | 10 | 4 | 2 | 0.6875 | 0.7143 |
| controlled_vs_proposed_raw | length_score | 10 | 5 | 1 | 0.6562 | 0.6667 |
| controlled_vs_proposed_raw | sentence_score | 8 | 2 | 6 | 0.6875 | 0.8000 |
| controlled_vs_proposed_raw | overall_quality | 13 | 3 | 0 | 0.8125 | 0.8125 |
| controlled_vs_candidate_no_context | context_relevance | 16 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 10 | 2 | 4 | 0.7500 | 0.8333 |
| controlled_vs_candidate_no_context | naturalness | 8 | 8 | 0 | 0.5000 | 0.5000 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 16 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 15 | 1 | 0 | 0.9375 | 0.9375 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 8 | 2 | 6 | 0.6875 | 0.8000 |
| controlled_vs_candidate_no_context | persona_style | 4 | 0 | 12 | 0.6250 | 1.0000 |
| controlled_vs_candidate_no_context | distinct1 | 7 | 7 | 2 | 0.5000 | 0.5000 |
| controlled_vs_candidate_no_context | length_score | 9 | 7 | 0 | 0.5625 | 0.5625 |
| controlled_vs_candidate_no_context | sentence_score | 4 | 1 | 11 | 0.5938 | 0.8000 |
| controlled_vs_candidate_no_context | overall_quality | 16 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 4 | 9 | 3 | 0.3438 | 0.3077 |
| controlled_alt_vs_controlled_default | persona_consistency | 1 | 9 | 6 | 0.2500 | 0.1000 |
| controlled_alt_vs_controlled_default | naturalness | 7 | 6 | 3 | 0.5312 | 0.5385 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 3 | 6 | 7 | 0.4062 | 0.3333 |
| controlled_alt_vs_controlled_default | context_overlap | 5 | 8 | 3 | 0.4062 | 0.3846 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 1 | 7 | 8 | 0.3125 | 0.1250 |
| controlled_alt_vs_controlled_default | persona_style | 0 | 3 | 13 | 0.4062 | 0.0000 |
| controlled_alt_vs_controlled_default | distinct1 | 6 | 7 | 3 | 0.4688 | 0.4615 |
| controlled_alt_vs_controlled_default | length_score | 7 | 5 | 4 | 0.5625 | 0.5833 |
| controlled_alt_vs_controlled_default | sentence_score | 3 | 2 | 11 | 0.5312 | 0.6000 |
| controlled_alt_vs_controlled_default | overall_quality | 1 | 12 | 3 | 0.1562 | 0.0769 |
| controlled_alt_vs_proposed_raw | context_relevance | 14 | 2 | 0 | 0.8750 | 0.8750 |
| controlled_alt_vs_proposed_raw | persona_consistency | 11 | 2 | 3 | 0.7812 | 0.8462 |
| controlled_alt_vs_proposed_raw | naturalness | 12 | 4 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 13 | 2 | 1 | 0.8438 | 0.8667 |
| controlled_alt_vs_proposed_raw | context_overlap | 14 | 2 | 0 | 0.8750 | 0.8750 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 11 | 1 | 4 | 0.8125 | 0.9167 |
| controlled_alt_vs_proposed_raw | persona_style | 3 | 1 | 12 | 0.5625 | 0.7500 |
| controlled_alt_vs_proposed_raw | distinct1 | 10 | 6 | 0 | 0.6250 | 0.6250 |
| controlled_alt_vs_proposed_raw | length_score | 12 | 4 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_proposed_raw | sentence_score | 9 | 2 | 5 | 0.7188 | 0.8182 |
| controlled_alt_vs_proposed_raw | overall_quality | 13 | 3 | 0 | 0.8125 | 0.8125 |
| controlled_alt_vs_candidate_no_context | context_relevance | 16 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 6 | 4 | 6 | 0.5625 | 0.6000 |
| controlled_alt_vs_candidate_no_context | naturalness | 7 | 9 | 0 | 0.4375 | 0.4375 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 16 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 15 | 1 | 0 | 0.9375 | 0.9375 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 6 | 4 | 6 | 0.5625 | 0.6000 |
| controlled_alt_vs_candidate_no_context | persona_style | 1 | 0 | 15 | 0.5312 | 1.0000 |
| controlled_alt_vs_candidate_no_context | distinct1 | 8 | 8 | 0 | 0.5000 | 0.5000 |
| controlled_alt_vs_candidate_no_context | length_score | 10 | 6 | 0 | 0.6250 | 0.6250 |
| controlled_alt_vs_candidate_no_context | sentence_score | 6 | 2 | 8 | 0.6250 | 0.7500 |
| controlled_alt_vs_candidate_no_context | overall_quality | 15 | 1 | 0 | 0.9375 | 0.9375 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.3125 | 0.4375 | 0.5625 |
| proposed_contextual_controlled_tuned | 0.0000 | 0.0000 | 0.3750 | 0.4375 | 0.5625 |
| proposed_contextual | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.3750 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `5`
- Unique template signatures: `16`
- Template signature ratio: `1.0000`
- Effective sample size by source clustering: `4.74`
- Effective sample size by template-signature clustering: `16.00`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: unavailable (disabled_by_flag).

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
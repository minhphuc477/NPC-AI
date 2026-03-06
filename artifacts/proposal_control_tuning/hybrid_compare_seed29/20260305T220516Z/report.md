# Proposal Alignment Evaluation Report

- Run ID: `20260305T220516Z`
- Generated: `2026-03-05T22:11:02.117561+00:00`
- Scenarios: `artifacts\proposal_control_tuning\hybrid_compare_seed29\20260305T220516Z\scenarios.jsonl`
- Scenario count: `32`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_hybrid`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2925 (0.2576, 0.3294) | 0.3205 (0.2710, 0.3728) | 0.8758 (0.8462, 0.9026) | 0.4211 (0.3998, 0.4438) | n/a |
| proposed_contextual_controlled_hybrid | 0.2587 (0.2262, 0.2931) | 0.3532 (0.2974, 0.4182) | 0.8842 (0.8588, 0.9080) | 0.4195 (0.3959, 0.4425) | n/a |
| proposed_contextual | 0.0562 (0.0303, 0.0870) | 0.1281 (0.0931, 0.1681) | 0.8084 (0.7801, 0.8395) | 0.2278 (0.2016, 0.2570) | n/a |
| candidate_no_context | 0.0245 (0.0121, 0.0417) | 0.1338 (0.0943, 0.1815) | 0.7951 (0.7722, 0.8207) | 0.2127 (0.1908, 0.2372) | n/a |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0317 | 1.2899 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0057 | -0.0426 |
| proposed_vs_candidate_no_context | naturalness | 0.0133 | 0.0167 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0424 | 2.4321 |
| proposed_vs_candidate_no_context | context_overlap | 0.0066 | 0.1610 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0119 | -0.2133 |
| proposed_vs_candidate_no_context | persona_style | 0.0191 | 0.0429 |
| proposed_vs_candidate_no_context | distinct1 | 0.0028 | 0.0030 |
| proposed_vs_candidate_no_context | length_score | 0.0500 | 0.2008 |
| proposed_vs_candidate_no_context | sentence_score | 0.0219 | 0.0301 |
| proposed_vs_candidate_no_context | overall_quality | 0.0150 | 0.0706 |
| controlled_vs_proposed_raw | context_relevance | 0.2363 | 4.2055 |
| controlled_vs_proposed_raw | persona_consistency | 0.1924 | 1.5023 |
| controlled_vs_proposed_raw | naturalness | 0.0674 | 0.0833 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.3037 | 5.0784 |
| controlled_vs_proposed_raw | context_overlap | 0.0790 | 1.6549 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2134 | 4.8610 |
| controlled_vs_proposed_raw | persona_style | 0.1086 | 0.2336 |
| controlled_vs_proposed_raw | distinct1 | 0.0018 | 0.0020 |
| controlled_vs_proposed_raw | length_score | 0.2469 | 0.8258 |
| controlled_vs_proposed_raw | sentence_score | 0.1750 | 0.2338 |
| controlled_vs_proposed_raw | overall_quality | 0.1934 | 0.8490 |
| controlled_vs_candidate_no_context | context_relevance | 0.2679 | 10.9200 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1867 | 1.3957 |
| controlled_vs_candidate_no_context | naturalness | 0.0807 | 0.1014 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3461 | 19.8614 |
| controlled_vs_candidate_no_context | context_overlap | 0.0857 | 2.0823 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2015 | 3.6107 |
| controlled_vs_candidate_no_context | persona_style | 0.1277 | 0.2864 |
| controlled_vs_candidate_no_context | distinct1 | 0.0046 | 0.0050 |
| controlled_vs_candidate_no_context | length_score | 0.2969 | 1.1925 |
| controlled_vs_candidate_no_context | sentence_score | 0.1969 | 0.2710 |
| controlled_vs_candidate_no_context | overall_quality | 0.2084 | 0.9795 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0338 | -0.1155 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0327 | 0.1021 |
| controlled_alt_vs_controlled_default | naturalness | 0.0084 | 0.0096 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0369 | -0.1016 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0264 | -0.2082 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0335 | 0.1301 |
| controlled_alt_vs_controlled_default | persona_style | 0.0297 | 0.0518 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0062 | 0.0067 |
| controlled_alt_vs_controlled_default | length_score | 0.0188 | 0.0344 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0219 | 0.0237 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0016 | -0.0038 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.2025 | 3.6044 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.2252 | 1.7578 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0758 | 0.0937 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2668 | 4.4608 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0526 | 1.1021 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.2469 | 5.6237 |
| controlled_alt_vs_proposed_raw | persona_style | 0.1383 | 0.2974 |
| controlled_alt_vs_proposed_raw | distinct1 | 0.0081 | 0.0086 |
| controlled_alt_vs_proposed_raw | length_score | 0.2656 | 0.8885 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1969 | 0.2630 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1918 | 0.8420 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2342 | 9.5436 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.2194 | 1.6402 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0891 | 0.1120 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3091 | 17.7418 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0593 | 1.4405 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.2350 | 4.2107 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.1574 | 0.3530 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0108 | 0.0116 |
| controlled_alt_vs_candidate_no_context | length_score | 0.3156 | 1.2678 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.2188 | 0.3011 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.2068 | 0.9720 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0317 | (0.0074, 0.0612) | 0.0047 | 0.0317 | (0.0074, 0.0596) | 0.0037 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0057 | (-0.0462, 0.0296) | 0.5980 | -0.0057 | (-0.0557, 0.0242) | 0.5917 |
| proposed_vs_candidate_no_context | naturalness | 0.0133 | (-0.0171, 0.0438) | 0.2003 | 0.0133 | (-0.0189, 0.0515) | 0.2553 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0424 | (0.0088, 0.0822) | 0.0033 | 0.0424 | (0.0097, 0.0819) | 0.0053 |
| proposed_vs_candidate_no_context | context_overlap | 0.0066 | (0.0002, 0.0135) | 0.0200 | 0.0066 | (0.0004, 0.0125) | 0.0210 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0119 | (-0.0588, 0.0268) | 0.6877 | -0.0119 | (-0.0732, 0.0245) | 0.6780 |
| proposed_vs_candidate_no_context | persona_style | 0.0191 | (-0.0424, 0.0806) | 0.2670 | 0.0191 | (0.0000, 0.0520) | 0.1040 |
| proposed_vs_candidate_no_context | distinct1 | 0.0028 | (-0.0105, 0.0167) | 0.3530 | 0.0028 | (-0.0150, 0.0244) | 0.4150 |
| proposed_vs_candidate_no_context | length_score | 0.0500 | (-0.0615, 0.1563) | 0.1943 | 0.0500 | (-0.0637, 0.1927) | 0.2147 |
| proposed_vs_candidate_no_context | sentence_score | 0.0219 | (-0.0328, 0.0766) | 0.3033 | 0.0219 | (-0.0427, 0.0875) | 0.3217 |
| proposed_vs_candidate_no_context | overall_quality | 0.0150 | (-0.0108, 0.0429) | 0.1350 | 0.0150 | (-0.0120, 0.0431) | 0.1357 |
| controlled_vs_proposed_raw | context_relevance | 0.2363 | (0.1940, 0.2769) | 0.0000 | 0.2363 | (0.2077, 0.2676) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.1924 | (0.1279, 0.2549) | 0.0000 | 0.1924 | (0.1359, 0.2606) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0674 | (0.0322, 0.1010) | 0.0003 | 0.0674 | (0.0249, 0.0962) | 0.0023 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.3037 | (0.2494, 0.3540) | 0.0000 | 0.3037 | (0.2680, 0.3423) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0790 | (0.0595, 0.0997) | 0.0000 | 0.0790 | (0.0603, 0.0985) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2134 | (0.1366, 0.2914) | 0.0000 | 0.2134 | (0.1499, 0.2967) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.1086 | (0.0285, 0.2015) | 0.0033 | 0.1086 | (-0.0172, 0.2646) | 0.0657 |
| controlled_vs_proposed_raw | distinct1 | 0.0018 | (-0.0103, 0.0147) | 0.3850 | 0.0018 | (-0.0159, 0.0167) | 0.4097 |
| controlled_vs_proposed_raw | length_score | 0.2469 | (0.1093, 0.3875) | 0.0007 | 0.2469 | (0.1160, 0.3298) | 0.0003 |
| controlled_vs_proposed_raw | sentence_score | 0.1750 | (0.0984, 0.2516) | 0.0000 | 0.1750 | (0.0117, 0.2743) | 0.0220 |
| controlled_vs_proposed_raw | overall_quality | 0.1934 | (0.1599, 0.2260) | 0.0000 | 0.1934 | (0.1637, 0.2297) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2679 | (0.2313, 0.3051) | 0.0000 | 0.2679 | (0.2379, 0.3032) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1867 | (0.1286, 0.2426) | 0.0000 | 0.1867 | (0.1311, 0.2552) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.0807 | (0.0421, 0.1161) | 0.0000 | 0.0807 | (0.0487, 0.1017) | 0.0000 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3461 | (0.3001, 0.3943) | 0.0000 | 0.3461 | (0.3096, 0.3906) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0857 | (0.0682, 0.1045) | 0.0000 | 0.0857 | (0.0691, 0.1032) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2015 | (0.1317, 0.2692) | 0.0000 | 0.2015 | (0.1402, 0.2871) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.1277 | (0.0414, 0.2174) | 0.0000 | 0.1277 | (-0.0112, 0.2918) | 0.0557 |
| controlled_vs_candidate_no_context | distinct1 | 0.0046 | (-0.0112, 0.0207) | 0.2800 | 0.0046 | (-0.0163, 0.0191) | 0.3227 |
| controlled_vs_candidate_no_context | length_score | 0.2969 | (0.1437, 0.4490) | 0.0003 | 0.2969 | (0.1687, 0.3958) | 0.0000 |
| controlled_vs_candidate_no_context | sentence_score | 0.1969 | (0.1203, 0.2734) | 0.0000 | 0.1969 | (0.0677, 0.2883) | 0.0010 |
| controlled_vs_candidate_no_context | overall_quality | 0.2084 | (0.1765, 0.2383) | 0.0000 | 0.2084 | (0.1780, 0.2389) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0338 | (-0.0672, 0.0007) | 0.9713 | -0.0338 | (-0.0602, -0.0032) | 0.9920 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0327 | (-0.0259, 0.0996) | 0.1460 | 0.0327 | (-0.0096, 0.0595) | 0.0610 |
| controlled_alt_vs_controlled_default | naturalness | 0.0084 | (-0.0273, 0.0462) | 0.3223 | 0.0084 | (-0.0331, 0.0374) | 0.3077 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0369 | (-0.0791, 0.0081) | 0.9500 | -0.0369 | (-0.0749, 0.0034) | 0.9530 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0264 | (-0.0439, -0.0100) | 0.9993 | -0.0264 | (-0.0380, -0.0113) | 0.9997 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0335 | (-0.0446, 0.1222) | 0.2217 | 0.0335 | (-0.0264, 0.0698) | 0.1097 |
| controlled_alt_vs_controlled_default | persona_style | 0.0297 | (-0.0120, 0.0755) | 0.0920 | 0.0297 | (0.0051, 0.0746) | 0.0013 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0062 | (-0.0103, 0.0229) | 0.2203 | 0.0062 | (-0.0168, 0.0260) | 0.2757 |
| controlled_alt_vs_controlled_default | length_score | 0.0188 | (-0.1385, 0.1823) | 0.4110 | 0.0188 | (-0.1276, 0.1279) | 0.3653 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0219 | (-0.0437, 0.0875) | 0.3077 | 0.0219 | (-0.0840, 0.1167) | 0.3843 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0016 | (-0.0247, 0.0229) | 0.5570 | -0.0016 | (-0.0178, 0.0178) | 0.6253 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.2025 | (0.1637, 0.2379) | 0.0000 | 0.2025 | (0.1701, 0.2467) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.2252 | (0.1591, 0.2919) | 0.0000 | 0.2252 | (0.1582, 0.2816) | 0.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0758 | (0.0395, 0.1124) | 0.0000 | 0.0758 | (0.0209, 0.1176) | 0.0020 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2668 | (0.2161, 0.3174) | 0.0000 | 0.2668 | (0.2210, 0.3244) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0526 | (0.0429, 0.0626) | 0.0000 | 0.0526 | (0.0432, 0.0674) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.2469 | (0.1701, 0.3286) | 0.0000 | 0.2469 | (0.1800, 0.3137) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 0.1383 | (0.0532, 0.2358) | 0.0003 | 0.1383 | (-0.0005, 0.3016) | 0.0260 |
| controlled_alt_vs_proposed_raw | distinct1 | 0.0081 | (-0.0083, 0.0242) | 0.1640 | 0.0081 | (-0.0124, 0.0254) | 0.2080 |
| controlled_alt_vs_proposed_raw | length_score | 0.2656 | (0.1177, 0.4146) | 0.0000 | 0.2656 | (0.0696, 0.4225) | 0.0047 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1969 | (0.1094, 0.2734) | 0.0000 | 0.1969 | (0.0750, 0.2849) | 0.0010 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1918 | (0.1595, 0.2235) | 0.0000 | 0.1918 | (0.1557, 0.2287) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2342 | (0.2016, 0.2673) | 0.0000 | 0.2342 | (0.1936, 0.2730) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.2194 | (0.1520, 0.2865) | 0.0000 | 0.2194 | (0.1351, 0.2849) | 0.0000 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0891 | (0.0603, 0.1173) | 0.0000 | 0.0891 | (0.0433, 0.1221) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3091 | (0.2652, 0.3534) | 0.0000 | 0.3091 | (0.2534, 0.3625) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0593 | (0.0500, 0.0684) | 0.0000 | 0.0593 | (0.0506, 0.0725) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.2350 | (0.1525, 0.3199) | 0.0000 | 0.2350 | (0.1468, 0.3040) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.1574 | (0.0761, 0.2419) | 0.0000 | 0.1574 | (0.0184, 0.3218) | 0.0140 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0108 | (-0.0033, 0.0254) | 0.0663 | 0.0108 | (-0.0081, 0.0263) | 0.1260 |
| controlled_alt_vs_candidate_no_context | length_score | 0.3156 | (0.1896, 0.4406) | 0.0000 | 0.3156 | (0.1435, 0.4495) | 0.0003 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.2188 | (0.1422, 0.2844) | 0.0000 | 0.2188 | (0.0979, 0.2962) | 0.0013 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.2068 | (0.1762, 0.2349) | 0.0000 | 0.2068 | (0.1689, 0.2422) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 10 | 5 | 17 | 0.5781 | 0.6667 |
| proposed_vs_candidate_no_context | persona_consistency | 5 | 4 | 23 | 0.5156 | 0.5556 |
| proposed_vs_candidate_no_context | naturalness | 10 | 5 | 17 | 0.5781 | 0.6667 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 8 | 2 | 22 | 0.5938 | 0.8000 |
| proposed_vs_candidate_no_context | context_overlap | 11 | 4 | 17 | 0.6094 | 0.7333 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 2 | 3 | 27 | 0.4844 | 0.4000 |
| proposed_vs_candidate_no_context | persona_style | 4 | 1 | 27 | 0.5469 | 0.8000 |
| proposed_vs_candidate_no_context | distinct1 | 6 | 6 | 20 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | length_score | 10 | 5 | 17 | 0.5781 | 0.6667 |
| proposed_vs_candidate_no_context | sentence_score | 5 | 3 | 24 | 0.5312 | 0.6250 |
| proposed_vs_candidate_no_context | overall_quality | 8 | 7 | 17 | 0.5156 | 0.5333 |
| controlled_vs_proposed_raw | context_relevance | 31 | 1 | 0 | 0.9688 | 0.9688 |
| controlled_vs_proposed_raw | persona_consistency | 25 | 2 | 5 | 0.8594 | 0.9259 |
| controlled_vs_proposed_raw | naturalness | 25 | 7 | 0 | 0.7812 | 0.7812 |
| controlled_vs_proposed_raw | context_keyword_coverage | 30 | 1 | 1 | 0.9531 | 0.9677 |
| controlled_vs_proposed_raw | context_overlap | 31 | 1 | 0 | 0.9688 | 0.9688 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 25 | 2 | 5 | 0.8594 | 0.9259 |
| controlled_vs_proposed_raw | persona_style | 8 | 2 | 22 | 0.5938 | 0.8000 |
| controlled_vs_proposed_raw | distinct1 | 17 | 11 | 4 | 0.5938 | 0.6071 |
| controlled_vs_proposed_raw | length_score | 21 | 11 | 0 | 0.6562 | 0.6562 |
| controlled_vs_proposed_raw | sentence_score | 19 | 3 | 10 | 0.7500 | 0.8636 |
| controlled_vs_proposed_raw | overall_quality | 31 | 1 | 0 | 0.9688 | 0.9688 |
| controlled_vs_candidate_no_context | context_relevance | 31 | 1 | 0 | 0.9688 | 0.9688 |
| controlled_vs_candidate_no_context | persona_consistency | 26 | 1 | 5 | 0.8906 | 0.9630 |
| controlled_vs_candidate_no_context | naturalness | 25 | 7 | 0 | 0.7812 | 0.7812 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 31 | 0 | 1 | 0.9844 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 31 | 1 | 0 | 0.9688 | 0.9688 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 26 | 1 | 5 | 0.8906 | 0.9630 |
| controlled_vs_candidate_no_context | persona_style | 10 | 2 | 20 | 0.6250 | 0.8333 |
| controlled_vs_candidate_no_context | distinct1 | 18 | 12 | 2 | 0.5938 | 0.6000 |
| controlled_vs_candidate_no_context | length_score | 20 | 12 | 0 | 0.6250 | 0.6250 |
| controlled_vs_candidate_no_context | sentence_score | 20 | 2 | 10 | 0.7812 | 0.9091 |
| controlled_vs_candidate_no_context | overall_quality | 31 | 1 | 0 | 0.9688 | 0.9688 |
| controlled_alt_vs_controlled_default | context_relevance | 12 | 17 | 3 | 0.4219 | 0.4138 |
| controlled_alt_vs_controlled_default | persona_consistency | 11 | 7 | 14 | 0.5625 | 0.6111 |
| controlled_alt_vs_controlled_default | naturalness | 14 | 15 | 3 | 0.4844 | 0.4828 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 8 | 15 | 9 | 0.3906 | 0.3478 |
| controlled_alt_vs_controlled_default | context_overlap | 8 | 21 | 3 | 0.2969 | 0.2759 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 8 | 7 | 17 | 0.5156 | 0.5333 |
| controlled_alt_vs_controlled_default | persona_style | 5 | 1 | 26 | 0.5625 | 0.8333 |
| controlled_alt_vs_controlled_default | distinct1 | 19 | 10 | 3 | 0.6406 | 0.6552 |
| controlled_alt_vs_controlled_default | length_score | 13 | 14 | 5 | 0.4844 | 0.4815 |
| controlled_alt_vs_controlled_default | sentence_score | 6 | 4 | 22 | 0.5312 | 0.6000 |
| controlled_alt_vs_controlled_default | overall_quality | 13 | 16 | 3 | 0.4531 | 0.4483 |
| controlled_alt_vs_proposed_raw | context_relevance | 31 | 1 | 0 | 0.9688 | 0.9688 |
| controlled_alt_vs_proposed_raw | persona_consistency | 25 | 1 | 6 | 0.8750 | 0.9615 |
| controlled_alt_vs_proposed_raw | naturalness | 24 | 8 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 30 | 1 | 1 | 0.9531 | 0.9677 |
| controlled_alt_vs_proposed_raw | context_overlap | 32 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 25 | 1 | 6 | 0.8750 | 0.9615 |
| controlled_alt_vs_proposed_raw | persona_style | 10 | 3 | 19 | 0.6094 | 0.7692 |
| controlled_alt_vs_proposed_raw | distinct1 | 17 | 15 | 0 | 0.5312 | 0.5312 |
| controlled_alt_vs_proposed_raw | length_score | 22 | 10 | 0 | 0.6875 | 0.6875 |
| controlled_alt_vs_proposed_raw | sentence_score | 21 | 3 | 8 | 0.7812 | 0.8750 |
| controlled_alt_vs_proposed_raw | overall_quality | 32 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 32 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 25 | 2 | 5 | 0.8594 | 0.9259 |
| controlled_alt_vs_candidate_no_context | naturalness | 26 | 6 | 0 | 0.8125 | 0.8125 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 31 | 0 | 1 | 0.9844 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 32 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 25 | 2 | 5 | 0.8594 | 0.9259 |
| controlled_alt_vs_candidate_no_context | persona_style | 13 | 1 | 18 | 0.6875 | 0.9286 |
| controlled_alt_vs_candidate_no_context | distinct1 | 18 | 13 | 1 | 0.5781 | 0.5806 |
| controlled_alt_vs_candidate_no_context | length_score | 23 | 8 | 1 | 0.7344 | 0.7419 |
| controlled_alt_vs_candidate_no_context | sentence_score | 22 | 2 | 8 | 0.8125 | 0.9167 |
| controlled_alt_vs_candidate_no_context | overall_quality | 31 | 1 | 0 | 0.9688 | 0.9688 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.3125 | 0.2500 | 0.7500 |
| proposed_contextual_controlled_hybrid | 0.0000 | 0.0000 | 0.2500 | 0.3438 | 0.6562 |
| proposed_contextual | 0.0000 | 0.0000 | 0.6250 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.6875 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `8`
- Unique template signatures: `28`
- Template signature ratio: `0.8750`
- Effective sample size by source clustering: `6.83`
- Effective sample size by template-signature clustering: `24.38`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: unavailable (disabled_by_flag).

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
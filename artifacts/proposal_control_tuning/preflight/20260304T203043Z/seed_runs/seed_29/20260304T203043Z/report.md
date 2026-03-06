# Proposal Alignment Evaluation Report

- Run ID: `20260304T203043Z`
- Generated: `2026-03-04T20:38:29.746684+00:00`
- Scenarios: `artifacts\proposal_control_tuning\preflight\20260304T203043Z\seed_runs\seed_29\20260304T203043Z\scenarios.jsonl`
- Scenario count: `40`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off
- `baseline_no_context`: model `phi3:mini`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2700 (0.2464, 0.2952) | 0.3586 (0.3040, 0.4153) | 0.8777 (0.8530, 0.9017) | 0.3878 (0.3697, 0.4079) | 0.0828 |
| proposed_contextual | 0.0558 (0.0302, 0.0854) | 0.1328 (0.0966, 0.1724) | 0.7926 (0.7728, 0.8153) | 0.2094 (0.1894, 0.2322) | 0.0601 |
| candidate_no_context | 0.0226 (0.0153, 0.0304) | 0.1806 (0.1320, 0.2367) | 0.8009 (0.7783, 0.8265) | 0.2105 (0.1890, 0.2337) | 0.0353 |
| baseline_no_context | 0.0383 (0.0259, 0.0528) | 0.1988 (0.1607, 0.2382) | 0.8809 (0.8626, 0.9007) | 0.2375 (0.2224, 0.2526) | 0.0469 |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0333 | 1.4741 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0478 | -0.2645 |
| proposed_vs_candidate_no_context | naturalness | -0.0082 | -0.0103 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0428 | 2.6526 |
| proposed_vs_candidate_no_context | context_overlap | 0.0110 | 0.2928 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0631 | -0.5955 |
| proposed_vs_candidate_no_context | persona_style | 0.0136 | 0.0283 |
| proposed_vs_candidate_no_context | distinct1 | -0.0059 | -0.0063 |
| proposed_vs_candidate_no_context | length_score | -0.0250 | -0.0949 |
| proposed_vs_candidate_no_context | sentence_score | -0.0088 | -0.0117 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0248 | 0.7017 |
| proposed_vs_candidate_no_context | overall_quality | -0.0012 | -0.0055 |
| proposed_vs_baseline_no_context | context_relevance | 0.0175 | 0.4583 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0660 | -0.3319 |
| proposed_vs_baseline_no_context | naturalness | -0.0882 | -0.1002 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0225 | 0.6175 |
| proposed_vs_baseline_no_context | context_overlap | 0.0060 | 0.1406 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0621 | -0.5918 |
| proposed_vs_baseline_no_context | persona_style | -0.0814 | -0.1418 |
| proposed_vs_baseline_no_context | distinct1 | -0.0477 | -0.0489 |
| proposed_vs_baseline_no_context | length_score | -0.2758 | -0.5365 |
| proposed_vs_baseline_no_context | sentence_score | -0.1400 | -0.1595 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0132 | 0.2820 |
| proposed_vs_baseline_no_context | overall_quality | -0.0282 | -0.1186 |
| controlled_vs_proposed_raw | context_relevance | 0.2142 | 3.8366 |
| controlled_vs_proposed_raw | persona_consistency | 0.2257 | 1.6995 |
| controlled_vs_proposed_raw | naturalness | 0.0850 | 0.1073 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2814 | 4.7744 |
| controlled_vs_proposed_raw | context_overlap | 0.0573 | 1.1808 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2546 | 5.9417 |
| controlled_vs_proposed_raw | persona_style | 0.1102 | 0.2236 |
| controlled_vs_proposed_raw | distinct1 | 0.0107 | 0.0115 |
| controlled_vs_proposed_raw | length_score | 0.3250 | 1.3636 |
| controlled_vs_proposed_raw | sentence_score | 0.1575 | 0.2136 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0227 | 0.3768 |
| controlled_vs_proposed_raw | overall_quality | 0.1785 | 0.8525 |
| controlled_vs_candidate_no_context | context_relevance | 0.2474 | 10.9664 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1780 | 0.9856 |
| controlled_vs_candidate_no_context | naturalness | 0.0768 | 0.0959 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3242 | 20.0915 |
| controlled_vs_candidate_no_context | context_overlap | 0.0683 | 1.8194 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1915 | 1.8079 |
| controlled_vs_candidate_no_context | persona_style | 0.1237 | 0.2582 |
| controlled_vs_candidate_no_context | distinct1 | 0.0048 | 0.0051 |
| controlled_vs_candidate_no_context | length_score | 0.3000 | 1.1392 |
| controlled_vs_candidate_no_context | sentence_score | 0.1487 | 0.1993 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0475 | 1.3430 |
| controlled_vs_candidate_no_context | overall_quality | 0.1773 | 0.8423 |
| controlled_vs_baseline_no_context | context_relevance | 0.2317 | 6.0534 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1597 | 0.8034 |
| controlled_vs_baseline_no_context | naturalness | -0.0032 | -0.0037 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3039 | 8.3399 |
| controlled_vs_baseline_no_context | context_overlap | 0.0633 | 1.4873 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1925 | 1.8333 |
| controlled_vs_baseline_no_context | persona_style | 0.0287 | 0.0500 |
| controlled_vs_baseline_no_context | distinct1 | -0.0370 | -0.0379 |
| controlled_vs_baseline_no_context | length_score | 0.0492 | 0.0956 |
| controlled_vs_baseline_no_context | sentence_score | 0.0175 | 0.0199 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0359 | 0.7651 |
| controlled_vs_baseline_no_context | overall_quality | 0.1503 | 0.6328 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2317 | 6.0534 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1597 | 0.8034 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0032 | -0.0037 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3039 | 8.3399 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0633 | 1.4873 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1925 | 1.8333 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0287 | 0.0500 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0370 | -0.0379 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.0492 | 0.0956 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0175 | 0.0199 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0359 | 0.7651 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1503 | 0.6328 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0333 | (0.0068, 0.0656) | 0.0030 | 0.0333 | (0.0045, 0.0686) | 0.0070 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0478 | (-0.0992, 0.0026) | 0.9673 | -0.0478 | (-0.1142, -0.0040) | 0.9880 |
| proposed_vs_candidate_no_context | naturalness | -0.0082 | (-0.0366, 0.0199) | 0.7197 | -0.0082 | (-0.0531, 0.0204) | 0.6780 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0428 | (0.0068, 0.0903) | 0.0060 | 0.0428 | (0.0059, 0.0893) | 0.0087 |
| proposed_vs_candidate_no_context | context_overlap | 0.0110 | (0.0032, 0.0199) | 0.0010 | 0.0110 | (0.0011, 0.0241) | 0.0110 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0631 | (-0.1238, -0.0095) | 0.9873 | -0.0631 | (-0.1456, -0.0133) | 0.9960 |
| proposed_vs_candidate_no_context | persona_style | 0.0136 | (-0.0377, 0.0622) | 0.2903 | 0.0136 | (-0.0125, 0.0552) | 0.1963 |
| proposed_vs_candidate_no_context | distinct1 | -0.0059 | (-0.0174, 0.0051) | 0.8547 | -0.0059 | (-0.0208, 0.0056) | 0.8393 |
| proposed_vs_candidate_no_context | length_score | -0.0250 | (-0.1408, 0.0842) | 0.6767 | -0.0250 | (-0.1969, 0.0862) | 0.6237 |
| proposed_vs_candidate_no_context | sentence_score | -0.0087 | (-0.0612, 0.0437) | 0.6787 | -0.0087 | (-0.0636, 0.0437) | 0.6777 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0248 | (0.0050, 0.0491) | 0.0050 | 0.0248 | (0.0056, 0.0505) | 0.0043 |
| proposed_vs_candidate_no_context | overall_quality | -0.0012 | (-0.0311, 0.0285) | 0.5667 | -0.0012 | (-0.0359, 0.0275) | 0.5353 |
| proposed_vs_baseline_no_context | context_relevance | 0.0175 | (-0.0116, 0.0519) | 0.1413 | 0.0175 | (-0.0199, 0.0624) | 0.2160 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0660 | (-0.1125, -0.0182) | 0.9970 | -0.0660 | (-0.1331, 0.0039) | 0.9673 |
| proposed_vs_baseline_no_context | naturalness | -0.0882 | (-0.1175, -0.0569) | 1.0000 | -0.0882 | (-0.1332, -0.0322) | 0.9987 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0225 | (-0.0185, 0.0680) | 0.1617 | 0.0225 | (-0.0313, 0.0829) | 0.2497 |
| proposed_vs_baseline_no_context | context_overlap | 0.0060 | (-0.0037, 0.0160) | 0.1223 | 0.0060 | (-0.0051, 0.0191) | 0.1593 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0621 | (-0.1170, -0.0075) | 0.9883 | -0.0621 | (-0.1367, 0.0185) | 0.9380 |
| proposed_vs_baseline_no_context | persona_style | -0.0814 | (-0.1564, -0.0177) | 0.9950 | -0.0814 | (-0.2302, 0.0148) | 0.8913 |
| proposed_vs_baseline_no_context | distinct1 | -0.0477 | (-0.0591, -0.0351) | 1.0000 | -0.0477 | (-0.0604, -0.0356) | 1.0000 |
| proposed_vs_baseline_no_context | length_score | -0.2758 | (-0.3958, -0.1567) | 1.0000 | -0.2758 | (-0.4611, -0.0500) | 0.9893 |
| proposed_vs_baseline_no_context | sentence_score | -0.1400 | (-0.2100, -0.0700) | 1.0000 | -0.1400 | (-0.2288, -0.0113) | 0.9850 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0132 | (-0.0068, 0.0334) | 0.0947 | 0.0132 | (-0.0015, 0.0309) | 0.0400 |
| proposed_vs_baseline_no_context | overall_quality | -0.0282 | (-0.0551, -0.0014) | 0.9797 | -0.0282 | (-0.0658, 0.0171) | 0.8903 |
| controlled_vs_proposed_raw | context_relevance | 0.2142 | (0.1802, 0.2468) | 0.0000 | 0.2142 | (0.1881, 0.2401) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.2257 | (0.1617, 0.2939) | 0.0000 | 0.2257 | (0.1690, 0.2928) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0850 | (0.0474, 0.1213) | 0.0000 | 0.0850 | (0.0310, 0.1224) | 0.0023 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2814 | (0.2345, 0.3251) | 0.0000 | 0.2814 | (0.2484, 0.3139) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0573 | (0.0456, 0.0684) | 0.0000 | 0.0573 | (0.0442, 0.0713) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2546 | (0.1821, 0.3322) | 0.0000 | 0.2546 | (0.1961, 0.3269) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.1102 | (0.0453, 0.1813) | 0.0000 | 0.1102 | (0.0000, 0.2492) | 0.0253 |
| controlled_vs_proposed_raw | distinct1 | 0.0107 | (-0.0060, 0.0255) | 0.0890 | 0.0107 | (-0.0147, 0.0309) | 0.1913 |
| controlled_vs_proposed_raw | length_score | 0.3250 | (0.1791, 0.4684) | 0.0000 | 0.3250 | (0.1354, 0.4651) | 0.0003 |
| controlled_vs_proposed_raw | sentence_score | 0.1575 | (0.0875, 0.2275) | 0.0000 | 0.1575 | (0.0750, 0.2143) | 0.0003 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0227 | (0.0025, 0.0430) | 0.0143 | 0.0227 | (-0.0046, 0.0477) | 0.0487 |
| controlled_vs_proposed_raw | overall_quality | 0.1785 | (0.1481, 0.2082) | 0.0000 | 0.1785 | (0.1443, 0.2147) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2474 | (0.2215, 0.2733) | 0.0000 | 0.2474 | (0.2206, 0.2769) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1780 | (0.1152, 0.2456) | 0.0000 | 0.1780 | (0.1309, 0.2360) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.0768 | (0.0371, 0.1132) | 0.0000 | 0.0768 | (0.0037, 0.1256) | 0.0200 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3242 | (0.2906, 0.3606) | 0.0000 | 0.3242 | (0.2889, 0.3622) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0683 | (0.0592, 0.0778) | 0.0000 | 0.0683 | (0.0585, 0.0817) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1915 | (0.1206, 0.2730) | 0.0000 | 0.1915 | (0.1497, 0.2474) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.1237 | (0.0526, 0.2047) | 0.0000 | 0.1237 | (0.0114, 0.2589) | 0.0043 |
| controlled_vs_candidate_no_context | distinct1 | 0.0048 | (-0.0116, 0.0202) | 0.2823 | 0.0048 | (-0.0225, 0.0253) | 0.3683 |
| controlled_vs_candidate_no_context | length_score | 0.3000 | (0.1425, 0.4509) | 0.0000 | 0.3000 | (0.0146, 0.4941) | 0.0187 |
| controlled_vs_candidate_no_context | sentence_score | 0.1487 | (0.0788, 0.2188) | 0.0003 | 0.1487 | (0.0790, 0.2042) | 0.0000 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0475 | (0.0238, 0.0736) | 0.0000 | 0.0475 | (0.0198, 0.0796) | 0.0000 |
| controlled_vs_candidate_no_context | overall_quality | 0.1773 | (0.1490, 0.2058) | 0.0000 | 0.1773 | (0.1498, 0.2118) | 0.0000 |
| controlled_vs_baseline_no_context | context_relevance | 0.2317 | (0.2028, 0.2607) | 0.0000 | 0.2317 | (0.1972, 0.2697) | 0.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1597 | (0.1054, 0.2192) | 0.0000 | 0.1597 | (0.1039, 0.2231) | 0.0000 |
| controlled_vs_baseline_no_context | naturalness | -0.0032 | (-0.0325, 0.0267) | 0.5597 | -0.0032 | (-0.0497, 0.0482) | 0.5760 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3039 | (0.2631, 0.3431) | 0.0000 | 0.3039 | (0.2573, 0.3525) | 0.0000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0633 | (0.0537, 0.0734) | 0.0000 | 0.0633 | (0.0537, 0.0742) | 0.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1925 | (0.1212, 0.2679) | 0.0000 | 0.1925 | (0.1268, 0.2621) | 0.0000 |
| controlled_vs_baseline_no_context | persona_style | 0.0287 | (-0.0080, 0.0725) | 0.0787 | 0.0287 | (-0.0098, 0.0843) | 0.0963 |
| controlled_vs_baseline_no_context | distinct1 | -0.0370 | (-0.0500, -0.0233) | 1.0000 | -0.0370 | (-0.0545, -0.0228) | 1.0000 |
| controlled_vs_baseline_no_context | length_score | 0.0492 | (-0.0900, 0.1892) | 0.2410 | 0.0492 | (-0.1692, 0.3007) | 0.3647 |
| controlled_vs_baseline_no_context | sentence_score | 0.0175 | (-0.0525, 0.0875) | 0.3650 | 0.0175 | (-0.0311, 0.0817) | 0.3303 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0359 | (0.0116, 0.0619) | 0.0017 | 0.0359 | (0.0097, 0.0605) | 0.0053 |
| controlled_vs_baseline_no_context | overall_quality | 0.1503 | (0.1243, 0.1755) | 0.0000 | 0.1503 | (0.1203, 0.1825) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2317 | (0.2026, 0.2623) | 0.0000 | 0.2317 | (0.1982, 0.2675) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1597 | (0.1012, 0.2203) | 0.0000 | 0.1597 | (0.1060, 0.2200) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0032 | (-0.0330, 0.0271) | 0.5787 | -0.0032 | (-0.0465, 0.0483) | 0.5617 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3039 | (0.2673, 0.3434) | 0.0000 | 0.3039 | (0.2579, 0.3536) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0633 | (0.0536, 0.0730) | 0.0000 | 0.0633 | (0.0532, 0.0744) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1925 | (0.1237, 0.2646) | 0.0000 | 0.1925 | (0.1284, 0.2608) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0287 | (-0.0089, 0.0729) | 0.0787 | 0.0287 | (-0.0100, 0.0841) | 0.1000 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0370 | (-0.0500, -0.0242) | 1.0000 | -0.0370 | (-0.0539, -0.0221) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.0492 | (-0.0950, 0.1900) | 0.2423 | 0.0492 | (-0.1660, 0.3032) | 0.3457 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0175 | (-0.0437, 0.0875) | 0.3597 | 0.0175 | (-0.0318, 0.0851) | 0.3250 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0359 | (0.0111, 0.0610) | 0.0017 | 0.0359 | (0.0092, 0.0614) | 0.0067 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1503 | (0.1250, 0.1751) | 0.0000 | 0.1503 | (0.1197, 0.1837) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 11 | 10 | 19 | 0.5125 | 0.5238 |
| proposed_vs_candidate_no_context | persona_consistency | 5 | 10 | 25 | 0.4375 | 0.3333 |
| proposed_vs_candidate_no_context | naturalness | 11 | 10 | 19 | 0.5125 | 0.5238 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 10 | 4 | 26 | 0.5750 | 0.7143 |
| proposed_vs_candidate_no_context | context_overlap | 12 | 8 | 20 | 0.5500 | 0.6000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 2 | 10 | 28 | 0.4000 | 0.1667 |
| proposed_vs_candidate_no_context | persona_style | 4 | 2 | 34 | 0.5250 | 0.6667 |
| proposed_vs_candidate_no_context | distinct1 | 7 | 11 | 22 | 0.4500 | 0.3889 |
| proposed_vs_candidate_no_context | length_score | 10 | 11 | 19 | 0.4875 | 0.4762 |
| proposed_vs_candidate_no_context | sentence_score | 5 | 6 | 29 | 0.4875 | 0.4545 |
| proposed_vs_candidate_no_context | bertscore_f1 | 16 | 5 | 19 | 0.6375 | 0.7619 |
| proposed_vs_candidate_no_context | overall_quality | 11 | 10 | 19 | 0.5125 | 0.5238 |
| proposed_vs_baseline_no_context | context_relevance | 14 | 26 | 0 | 0.3500 | 0.3500 |
| proposed_vs_baseline_no_context | persona_consistency | 6 | 18 | 16 | 0.3500 | 0.2500 |
| proposed_vs_baseline_no_context | naturalness | 8 | 31 | 1 | 0.2125 | 0.2051 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 10 | 11 | 19 | 0.4875 | 0.4762 |
| proposed_vs_baseline_no_context | context_overlap | 17 | 22 | 1 | 0.4375 | 0.4359 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 3 | 13 | 24 | 0.3750 | 0.1875 |
| proposed_vs_baseline_no_context | persona_style | 3 | 8 | 29 | 0.4375 | 0.2727 |
| proposed_vs_baseline_no_context | distinct1 | 4 | 33 | 3 | 0.1375 | 0.1081 |
| proposed_vs_baseline_no_context | length_score | 8 | 29 | 3 | 0.2375 | 0.2162 |
| proposed_vs_baseline_no_context | sentence_score | 3 | 19 | 18 | 0.3000 | 0.1364 |
| proposed_vs_baseline_no_context | bertscore_f1 | 19 | 21 | 0 | 0.4750 | 0.4750 |
| proposed_vs_baseline_no_context | overall_quality | 11 | 29 | 0 | 0.2750 | 0.2750 |
| controlled_vs_proposed_raw | context_relevance | 39 | 1 | 0 | 0.9750 | 0.9750 |
| controlled_vs_proposed_raw | persona_consistency | 33 | 3 | 4 | 0.8750 | 0.9167 |
| controlled_vs_proposed_raw | naturalness | 31 | 9 | 0 | 0.7750 | 0.7750 |
| controlled_vs_proposed_raw | context_keyword_coverage | 38 | 1 | 1 | 0.9625 | 0.9744 |
| controlled_vs_proposed_raw | context_overlap | 38 | 2 | 0 | 0.9500 | 0.9500 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 33 | 1 | 6 | 0.9000 | 0.9706 |
| controlled_vs_proposed_raw | persona_style | 12 | 2 | 26 | 0.6250 | 0.8571 |
| controlled_vs_proposed_raw | distinct1 | 26 | 12 | 2 | 0.6750 | 0.6842 |
| controlled_vs_proposed_raw | length_score | 27 | 13 | 0 | 0.6750 | 0.6750 |
| controlled_vs_proposed_raw | sentence_score | 21 | 3 | 16 | 0.7250 | 0.8750 |
| controlled_vs_proposed_raw | bertscore_f1 | 26 | 14 | 0 | 0.6500 | 0.6500 |
| controlled_vs_proposed_raw | overall_quality | 39 | 1 | 0 | 0.9750 | 0.9750 |
| controlled_vs_candidate_no_context | context_relevance | 40 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 27 | 4 | 9 | 0.7875 | 0.8710 |
| controlled_vs_candidate_no_context | naturalness | 31 | 9 | 0 | 0.7750 | 0.7750 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 40 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 40 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 27 | 4 | 9 | 0.7875 | 0.8710 |
| controlled_vs_candidate_no_context | persona_style | 11 | 1 | 28 | 0.6250 | 0.9167 |
| controlled_vs_candidate_no_context | distinct1 | 23 | 17 | 0 | 0.5750 | 0.5750 |
| controlled_vs_candidate_no_context | length_score | 29 | 10 | 1 | 0.7375 | 0.7436 |
| controlled_vs_candidate_no_context | sentence_score | 20 | 3 | 17 | 0.7125 | 0.8696 |
| controlled_vs_candidate_no_context | bertscore_f1 | 30 | 10 | 0 | 0.7500 | 0.7500 |
| controlled_vs_candidate_no_context | overall_quality | 39 | 1 | 0 | 0.9750 | 0.9750 |
| controlled_vs_baseline_no_context | context_relevance | 40 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 30 | 4 | 6 | 0.8250 | 0.8824 |
| controlled_vs_baseline_no_context | naturalness | 20 | 20 | 0 | 0.5000 | 0.5000 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 38 | 0 | 2 | 0.9750 | 1.0000 |
| controlled_vs_baseline_no_context | context_overlap | 39 | 1 | 0 | 0.9750 | 0.9750 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 30 | 3 | 7 | 0.8375 | 0.9091 |
| controlled_vs_baseline_no_context | persona_style | 5 | 2 | 33 | 0.5375 | 0.7143 |
| controlled_vs_baseline_no_context | distinct1 | 8 | 31 | 1 | 0.2125 | 0.2051 |
| controlled_vs_baseline_no_context | length_score | 22 | 17 | 1 | 0.5625 | 0.5641 |
| controlled_vs_baseline_no_context | sentence_score | 9 | 7 | 24 | 0.5250 | 0.5625 |
| controlled_vs_baseline_no_context | bertscore_f1 | 28 | 12 | 0 | 0.7000 | 0.7000 |
| controlled_vs_baseline_no_context | overall_quality | 38 | 2 | 0 | 0.9500 | 0.9500 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 40 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 30 | 4 | 6 | 0.8250 | 0.8824 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 20 | 20 | 0 | 0.5000 | 0.5000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 38 | 0 | 2 | 0.9750 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 39 | 1 | 0 | 0.9750 | 0.9750 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 30 | 3 | 7 | 0.8375 | 0.9091 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 5 | 2 | 33 | 0.5375 | 0.7143 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | 8 | 31 | 1 | 0.2125 | 0.2051 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 22 | 17 | 1 | 0.5625 | 0.5641 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 9 | 7 | 24 | 0.5250 | 0.5625 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 28 | 12 | 0 | 0.7000 | 0.7000 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 38 | 2 | 0 | 0.9500 | 0.9500 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.3250 | 0.3750 | 0.6250 |
| proposed_contextual | 0.0000 | 0.0000 | 0.6000 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.5750 | 0.0000 | 0.0000 |
| baseline_no_context | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `8`
- Unique template signatures: `33`
- Template signature ratio: `0.8250`
- Effective sample size by source clustering: `7.02`
- Effective sample size by template-signature clustering: `28.57`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: enabled.

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
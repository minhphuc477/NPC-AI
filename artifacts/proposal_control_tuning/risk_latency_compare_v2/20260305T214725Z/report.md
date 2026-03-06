# Proposal Alignment Evaluation Report

- Run ID: `20260305T214725Z`
- Generated: `2026-03-05T21:53:09.725724+00:00`
- Scenarios: `artifacts\proposal_control_tuning\risk_latency_compare_v2\20260305T214725Z\scenarios.jsonl`
- Scenario count: `32`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_rla`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2737 (0.2414, 0.3109) | 0.3295 (0.2815, 0.3842) | 0.8971 (0.8778, 0.9162) | 0.4200 (0.3987, 0.4469) | n/a |
| proposed_contextual_controlled_rla | 0.2684 (0.2431, 0.2950) | 0.3694 (0.3080, 0.4374) | 0.8975 (0.8783, 0.9158) | 0.4323 (0.4086, 0.4569) | n/a |
| proposed_contextual | 0.0725 (0.0430, 0.1072) | 0.1367 (0.0992, 0.1822) | 0.8092 (0.7810, 0.8363) | 0.2382 (0.2108, 0.2715) | n/a |
| candidate_no_context | 0.0210 (0.0126, 0.0316) | 0.1573 (0.1169, 0.2055) | 0.8191 (0.7924, 0.8475) | 0.2237 (0.2047, 0.2448) | n/a |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0515 | 2.4515 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0207 | -0.1313 |
| proposed_vs_candidate_no_context | naturalness | -0.0099 | -0.0120 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0689 | 5.0172 |
| proposed_vs_candidate_no_context | context_overlap | 0.0108 | 0.2845 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0231 | -0.3444 |
| proposed_vs_candidate_no_context | persona_style | -0.0111 | -0.0213 |
| proposed_vs_candidate_no_context | distinct1 | 0.0014 | 0.0015 |
| proposed_vs_candidate_no_context | length_score | -0.0521 | -0.1524 |
| proposed_vs_candidate_no_context | sentence_score | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | overall_quality | 0.0145 | 0.0647 |
| controlled_vs_proposed_raw | context_relevance | 0.2013 | 2.7776 |
| controlled_vs_proposed_raw | persona_consistency | 0.1928 | 1.4109 |
| controlled_vs_proposed_raw | naturalness | 0.0878 | 0.1086 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2624 | 3.1759 |
| controlled_vs_proposed_raw | context_overlap | 0.0586 | 1.2018 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2161 | 4.9220 |
| controlled_vs_proposed_raw | persona_style | 0.0999 | 0.1968 |
| controlled_vs_proposed_raw | distinct1 | -0.0017 | -0.0018 |
| controlled_vs_proposed_raw | length_score | 0.3615 | 1.2482 |
| controlled_vs_proposed_raw | sentence_score | 0.1641 | 0.2160 |
| controlled_vs_proposed_raw | overall_quality | 0.1818 | 0.7634 |
| controlled_vs_candidate_no_context | context_relevance | 0.2527 | 12.0385 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1722 | 1.0942 |
| controlled_vs_candidate_no_context | naturalness | 0.0780 | 0.0952 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3313 | 24.1276 |
| controlled_vs_candidate_no_context | context_overlap | 0.0694 | 1.8284 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1930 | 2.8822 |
| controlled_vs_candidate_no_context | persona_style | 0.0889 | 0.1712 |
| controlled_vs_candidate_no_context | distinct1 | -0.0003 | -0.0004 |
| controlled_vs_candidate_no_context | length_score | 0.3094 | 0.9055 |
| controlled_vs_candidate_no_context | sentence_score | 0.1641 | 0.2160 |
| controlled_vs_candidate_no_context | overall_quality | 0.1963 | 0.8776 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0054 | -0.0196 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0399 | 0.1211 |
| controlled_alt_vs_controlled_default | naturalness | 0.0004 | 0.0005 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0095 | -0.0274 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0042 | 0.0396 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0485 | 0.1866 |
| controlled_alt_vs_controlled_default | persona_style | 0.0054 | 0.0089 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0111 | 0.0118 |
| controlled_alt_vs_controlled_default | length_score | -0.0208 | -0.0320 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0000 | 0.0000 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0123 | 0.0292 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1959 | 2.7037 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.2327 | 1.7028 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0883 | 0.1091 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2529 | 3.0613 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0628 | 1.2889 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.2646 | 6.0271 |
| controlled_alt_vs_proposed_raw | persona_style | 0.1053 | 0.2074 |
| controlled_alt_vs_proposed_raw | distinct1 | 0.0093 | 0.0099 |
| controlled_alt_vs_proposed_raw | length_score | 0.3406 | 1.1763 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1641 | 0.2160 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1941 | 0.8150 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2474 | 11.7834 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.2121 | 1.3478 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0784 | 0.0957 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3218 | 23.4379 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0736 | 1.9402 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.2415 | 3.6067 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0943 | 0.1817 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0107 | 0.0114 |
| controlled_alt_vs_candidate_no_context | length_score | 0.2885 | 0.8445 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1641 | 0.2160 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.2086 | 0.9325 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0515 | (0.0229, 0.0840) | 0.0003 | 0.0515 | (0.0188, 0.0915) | 0.0000 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0207 | (-0.0713, 0.0212) | 0.8177 | -0.0207 | (-0.0679, 0.0219) | 0.8210 |
| proposed_vs_candidate_no_context | naturalness | -0.0099 | (-0.0386, 0.0187) | 0.7500 | -0.0099 | (-0.0512, 0.0219) | 0.6970 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0689 | (0.0303, 0.1125) | 0.0000 | 0.0689 | (0.0252, 0.1207) | 0.0003 |
| proposed_vs_candidate_no_context | context_overlap | 0.0108 | (0.0020, 0.0209) | 0.0070 | 0.0108 | (0.0021, 0.0211) | 0.0043 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0231 | (-0.0796, 0.0260) | 0.8003 | -0.0231 | (-0.0785, 0.0230) | 0.8137 |
| proposed_vs_candidate_no_context | persona_style | -0.0111 | (-0.0462, 0.0182) | 0.7520 | -0.0111 | (-0.0491, 0.0188) | 0.7463 |
| proposed_vs_candidate_no_context | distinct1 | 0.0014 | (-0.0122, 0.0146) | 0.4177 | 0.0014 | (-0.0158, 0.0158) | 0.4207 |
| proposed_vs_candidate_no_context | length_score | -0.0521 | (-0.1646, 0.0500) | 0.8423 | -0.0521 | (-0.1987, 0.0615) | 0.7817 |
| proposed_vs_candidate_no_context | sentence_score | 0.0000 | (-0.0656, 0.0656) | 0.5580 | 0.0000 | (-0.0778, 0.0628) | 0.5653 |
| proposed_vs_candidate_no_context | overall_quality | 0.0145 | (-0.0116, 0.0400) | 0.1333 | 0.0145 | (-0.0173, 0.0480) | 0.2033 |
| controlled_vs_proposed_raw | context_relevance | 0.2013 | (0.1591, 0.2450) | 0.0000 | 0.2013 | (0.1306, 0.2587) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.1928 | (0.1322, 0.2560) | 0.0000 | 0.1928 | (0.1191, 0.2693) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0878 | (0.0515, 0.1215) | 0.0000 | 0.0878 | (0.0307, 0.1408) | 0.0003 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2624 | (0.2056, 0.3213) | 0.0000 | 0.2624 | (0.1668, 0.3449) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0586 | (0.0456, 0.0744) | 0.0000 | 0.0586 | (0.0445, 0.0727) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2161 | (0.1427, 0.2887) | 0.0000 | 0.2161 | (0.1291, 0.3122) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.0999 | (0.0371, 0.1702) | 0.0003 | 0.0999 | (0.0000, 0.2417) | 0.0290 |
| controlled_vs_proposed_raw | distinct1 | -0.0017 | (-0.0171, 0.0137) | 0.5840 | -0.0017 | (-0.0155, 0.0153) | 0.5807 |
| controlled_vs_proposed_raw | length_score | 0.3615 | (0.2188, 0.4948) | 0.0000 | 0.3615 | (0.1412, 0.5677) | 0.0000 |
| controlled_vs_proposed_raw | sentence_score | 0.1641 | (0.0766, 0.2406) | 0.0003 | 0.1641 | (0.0778, 0.2593) | 0.0003 |
| controlled_vs_proposed_raw | overall_quality | 0.1818 | (0.1422, 0.2199) | 0.0000 | 0.1818 | (0.1253, 0.2368) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2527 | (0.2164, 0.2929) | 0.0000 | 0.2527 | (0.2106, 0.2911) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1722 | (0.1051, 0.2373) | 0.0000 | 0.1722 | (0.1016, 0.2520) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.0780 | (0.0468, 0.1084) | 0.0000 | 0.0780 | (0.0381, 0.1216) | 0.0000 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3313 | (0.2836, 0.3842) | 0.0000 | 0.3313 | (0.2730, 0.3858) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0694 | (0.0546, 0.0854) | 0.0000 | 0.0694 | (0.0539, 0.0825) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1930 | (0.1131, 0.2725) | 0.0000 | 0.1930 | (0.0961, 0.2979) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.0889 | (0.0306, 0.1572) | 0.0000 | 0.0889 | (0.0179, 0.2062) | 0.0027 |
| controlled_vs_candidate_no_context | distinct1 | -0.0003 | (-0.0158, 0.0142) | 0.5077 | -0.0003 | (-0.0145, 0.0116) | 0.5240 |
| controlled_vs_candidate_no_context | length_score | 0.3094 | (0.1854, 0.4344) | 0.0000 | 0.3094 | (0.1441, 0.4933) | 0.0000 |
| controlled_vs_candidate_no_context | sentence_score | 0.1641 | (0.0875, 0.2406) | 0.0000 | 0.1641 | (0.1167, 0.2000) | 0.0000 |
| controlled_vs_candidate_no_context | overall_quality | 0.1963 | (0.1647, 0.2271) | 0.0000 | 0.1963 | (0.1594, 0.2359) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0054 | (-0.0436, 0.0328) | 0.6083 | -0.0054 | (-0.0462, 0.0359) | 0.5957 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0399 | (-0.0241, 0.1083) | 0.1233 | 0.0399 | (-0.0296, 0.1224) | 0.1547 |
| controlled_alt_vs_controlled_default | naturalness | 0.0004 | (-0.0282, 0.0272) | 0.4787 | 0.0004 | (-0.0352, 0.0276) | 0.4793 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0095 | (-0.0597, 0.0403) | 0.6397 | -0.0095 | (-0.0653, 0.0428) | 0.6293 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0042 | (-0.0134, 0.0192) | 0.2950 | 0.0042 | (-0.0066, 0.0153) | 0.2313 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0485 | (-0.0263, 0.1368) | 0.1110 | 0.0485 | (-0.0390, 0.1532) | 0.1477 |
| controlled_alt_vs_controlled_default | persona_style | 0.0054 | (-0.0215, 0.0417) | 0.4203 | 0.0054 | (-0.0215, 0.0447) | 0.3830 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0111 | (-0.0053, 0.0285) | 0.0870 | 0.0111 | (-0.0065, 0.0254) | 0.0987 |
| controlled_alt_vs_controlled_default | length_score | -0.0208 | (-0.1510, 0.1094) | 0.6230 | -0.0208 | (-0.1851, 0.1222) | 0.6177 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0000 | (-0.0656, 0.0656) | 0.5667 | 0.0000 | (-0.0875, 0.0824) | 0.5420 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0123 | (-0.0184, 0.0425) | 0.2067 | 0.0123 | (-0.0222, 0.0508) | 0.2460 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1959 | (0.1612, 0.2286) | 0.0000 | 0.1959 | (0.1532, 0.2293) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.2327 | (0.1597, 0.3074) | 0.0000 | 0.2327 | (0.1558, 0.3198) | 0.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0883 | (0.0529, 0.1218) | 0.0000 | 0.0883 | (0.0289, 0.1446) | 0.0030 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2529 | (0.2062, 0.3017) | 0.0000 | 0.2529 | (0.1962, 0.2968) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0628 | (0.0510, 0.0740) | 0.0000 | 0.0628 | (0.0513, 0.0730) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.2646 | (0.1811, 0.3561) | 0.0000 | 0.2646 | (0.1786, 0.3544) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 0.1053 | (0.0409, 0.1790) | 0.0000 | 0.1053 | (0.0140, 0.2548) | 0.0233 |
| controlled_alt_vs_proposed_raw | distinct1 | 0.0093 | (-0.0107, 0.0278) | 0.1737 | 0.0093 | (-0.0123, 0.0320) | 0.1983 |
| controlled_alt_vs_proposed_raw | length_score | 0.3406 | (0.2042, 0.4740) | 0.0000 | 0.3406 | (0.1238, 0.5191) | 0.0010 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1641 | (0.0766, 0.2406) | 0.0000 | 0.1641 | (0.0284, 0.2935) | 0.0090 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1941 | (0.1546, 0.2329) | 0.0000 | 0.1941 | (0.1393, 0.2438) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2474 | (0.2224, 0.2739) | 0.0000 | 0.2474 | (0.2220, 0.2709) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.2121 | (0.1412, 0.2830) | 0.0000 | 0.2121 | (0.1478, 0.2745) | 0.0000 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0784 | (0.0416, 0.1150) | 0.0000 | 0.0784 | (0.0309, 0.1183) | 0.0013 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3218 | (0.2889, 0.3552) | 0.0000 | 0.3218 | (0.2891, 0.3538) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0736 | (0.0610, 0.0857) | 0.0000 | 0.0736 | (0.0621, 0.0848) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.2415 | (0.1607, 0.3268) | 0.0000 | 0.2415 | (0.1623, 0.3179) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0943 | (0.0306, 0.1703) | 0.0003 | 0.0943 | (0.0171, 0.2145) | 0.0037 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0107 | (-0.0070, 0.0276) | 0.1103 | 0.0107 | (-0.0098, 0.0262) | 0.1333 |
| controlled_alt_vs_candidate_no_context | length_score | 0.2885 | (0.1416, 0.4240) | 0.0000 | 0.2885 | (0.1216, 0.4133) | 0.0003 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1641 | (0.0766, 0.2406) | 0.0003 | 0.1641 | (0.0766, 0.2574) | 0.0000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.2086 | (0.1778, 0.2405) | 0.0000 | 0.2086 | (0.1829, 0.2325) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 13 | 5 | 14 | 0.6250 | 0.7222 |
| proposed_vs_candidate_no_context | persona_consistency | 4 | 5 | 23 | 0.4844 | 0.4444 |
| proposed_vs_candidate_no_context | naturalness | 8 | 10 | 14 | 0.4688 | 0.4444 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 11 | 2 | 19 | 0.6406 | 0.8462 |
| proposed_vs_candidate_no_context | context_overlap | 12 | 6 | 14 | 0.5938 | 0.6667 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 3 | 4 | 25 | 0.4844 | 0.4286 |
| proposed_vs_candidate_no_context | persona_style | 1 | 2 | 29 | 0.4844 | 0.3333 |
| proposed_vs_candidate_no_context | distinct1 | 10 | 7 | 15 | 0.5469 | 0.5882 |
| proposed_vs_candidate_no_context | length_score | 7 | 11 | 14 | 0.4375 | 0.3889 |
| proposed_vs_candidate_no_context | sentence_score | 5 | 5 | 22 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | overall_quality | 10 | 8 | 14 | 0.5312 | 0.5556 |
| controlled_vs_proposed_raw | context_relevance | 30 | 2 | 0 | 0.9375 | 0.9375 |
| controlled_vs_proposed_raw | persona_consistency | 28 | 2 | 2 | 0.9062 | 0.9333 |
| controlled_vs_proposed_raw | naturalness | 25 | 7 | 0 | 0.7812 | 0.7812 |
| controlled_vs_proposed_raw | context_keyword_coverage | 28 | 2 | 2 | 0.9062 | 0.9333 |
| controlled_vs_proposed_raw | context_overlap | 31 | 1 | 0 | 0.9688 | 0.9688 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 28 | 2 | 2 | 0.9062 | 0.9333 |
| controlled_vs_proposed_raw | persona_style | 8 | 1 | 23 | 0.6094 | 0.8889 |
| controlled_vs_proposed_raw | distinct1 | 15 | 13 | 4 | 0.5312 | 0.5357 |
| controlled_vs_proposed_raw | length_score | 26 | 5 | 1 | 0.8281 | 0.8387 |
| controlled_vs_proposed_raw | sentence_score | 18 | 3 | 11 | 0.7344 | 0.8571 |
| controlled_vs_proposed_raw | overall_quality | 30 | 2 | 0 | 0.9375 | 0.9375 |
| controlled_vs_candidate_no_context | context_relevance | 32 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 25 | 4 | 3 | 0.8281 | 0.8621 |
| controlled_vs_candidate_no_context | naturalness | 23 | 8 | 1 | 0.7344 | 0.7419 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 32 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 31 | 1 | 0 | 0.9688 | 0.9688 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 25 | 4 | 3 | 0.8281 | 0.8621 |
| controlled_vs_candidate_no_context | persona_style | 8 | 1 | 23 | 0.6094 | 0.8889 |
| controlled_vs_candidate_no_context | distinct1 | 16 | 13 | 3 | 0.5469 | 0.5517 |
| controlled_vs_candidate_no_context | length_score | 21 | 9 | 2 | 0.6875 | 0.7000 |
| controlled_vs_candidate_no_context | sentence_score | 18 | 3 | 11 | 0.7344 | 0.8571 |
| controlled_vs_candidate_no_context | overall_quality | 32 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 13 | 12 | 7 | 0.5156 | 0.5200 |
| controlled_alt_vs_controlled_default | persona_consistency | 6 | 8 | 18 | 0.4688 | 0.4286 |
| controlled_alt_vs_controlled_default | naturalness | 13 | 12 | 7 | 0.5156 | 0.5200 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 8 | 10 | 14 | 0.4688 | 0.4444 |
| controlled_alt_vs_controlled_default | context_overlap | 16 | 9 | 7 | 0.6094 | 0.6400 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 6 | 6 | 20 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | persona_style | 1 | 3 | 28 | 0.4688 | 0.2500 |
| controlled_alt_vs_controlled_default | distinct1 | 17 | 8 | 7 | 0.6406 | 0.6800 |
| controlled_alt_vs_controlled_default | length_score | 12 | 12 | 8 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | sentence_score | 5 | 5 | 22 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | overall_quality | 14 | 11 | 7 | 0.5469 | 0.5600 |
| controlled_alt_vs_proposed_raw | context_relevance | 31 | 1 | 0 | 0.9688 | 0.9688 |
| controlled_alt_vs_proposed_raw | persona_consistency | 27 | 1 | 4 | 0.9062 | 0.9643 |
| controlled_alt_vs_proposed_raw | naturalness | 25 | 7 | 0 | 0.7812 | 0.7812 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 28 | 0 | 4 | 0.9375 | 1.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 30 | 2 | 0 | 0.9375 | 0.9375 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 27 | 1 | 4 | 0.9062 | 0.9643 |
| controlled_alt_vs_proposed_raw | persona_style | 9 | 1 | 22 | 0.6250 | 0.9000 |
| controlled_alt_vs_proposed_raw | distinct1 | 18 | 11 | 3 | 0.6094 | 0.6207 |
| controlled_alt_vs_proposed_raw | length_score | 25 | 6 | 1 | 0.7969 | 0.8065 |
| controlled_alt_vs_proposed_raw | sentence_score | 18 | 3 | 11 | 0.7344 | 0.8571 |
| controlled_alt_vs_proposed_raw | overall_quality | 31 | 1 | 0 | 0.9688 | 0.9688 |
| controlled_alt_vs_candidate_no_context | context_relevance | 32 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 25 | 2 | 5 | 0.8594 | 0.9259 |
| controlled_alt_vs_candidate_no_context | naturalness | 23 | 9 | 0 | 0.7188 | 0.7188 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 32 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 31 | 1 | 0 | 0.9688 | 0.9688 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 25 | 2 | 5 | 0.8594 | 0.9259 |
| controlled_alt_vs_candidate_no_context | persona_style | 8 | 2 | 22 | 0.5938 | 0.8000 |
| controlled_alt_vs_candidate_no_context | distinct1 | 19 | 11 | 2 | 0.6250 | 0.6333 |
| controlled_alt_vs_candidate_no_context | length_score | 22 | 9 | 1 | 0.7031 | 0.7097 |
| controlled_alt_vs_candidate_no_context | sentence_score | 19 | 4 | 9 | 0.7344 | 0.8261 |
| controlled_alt_vs_candidate_no_context | overall_quality | 32 | 0 | 0 | 1.0000 | 1.0000 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.3438 | 0.4375 | 0.5625 |
| proposed_contextual_controlled_rla | 0.0000 | 0.0000 | 0.3438 | 0.3750 | 0.6250 |
| proposed_contextual | 0.0000 | 0.0000 | 0.5938 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `8`
- Unique template signatures: `30`
- Template signature ratio: `0.9375`
- Effective sample size by source clustering: `7.42`
- Effective sample size by template-signature clustering: `28.44`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: unavailable (disabled_by_flag).

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
# Proposal Alignment Evaluation Report

- Run ID: `20260305T225946Z`
- Generated: `2026-03-05T23:04:58.455716+00:00`
- Scenarios: `artifacts\proposal_control_tuning\architecture_compare\20260305T224823Z\hybrid_balanced\seed_29\20260305T225946Z\scenarios.jsonl`
- Scenario count: `24`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_alt`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off
- `baseline_no_context`: model `phi3:mini`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2708 (0.2357, 0.3091) | 0.3459 (0.2925, 0.4003) | 0.8903 (0.8632, 0.9159) | 0.4236 (0.3983, 0.4488) | n/a |
| proposed_contextual_controlled_alt | 0.2713 (0.2268, 0.3147) | 0.3531 (0.2838, 0.4271) | 0.8669 (0.8384, 0.8935) | 0.4218 (0.3910, 0.4551) | n/a |
| proposed_contextual | 0.0399 (0.0199, 0.0647) | 0.1012 (0.0695, 0.1353) | 0.7989 (0.7687, 0.8343) | 0.2084 (0.1859, 0.2338) | n/a |
| candidate_no_context | 0.0136 (0.0094, 0.0200) | 0.1452 (0.0920, 0.2067) | 0.8000 (0.7725, 0.8307) | 0.2128 (0.1892, 0.2402) | n/a |
| baseline_no_context | 0.0418 (0.0248, 0.0602) | 0.1966 (0.1465, 0.2499) | 0.8826 (0.8638, 0.9022) | 0.2603 (0.2369, 0.2849) | n/a |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0263 | 1.9291 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0440 | -0.3030 |
| proposed_vs_candidate_no_context | naturalness | -0.0011 | -0.0014 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0338 | 8.9167 |
| proposed_vs_candidate_no_context | context_overlap | 0.0089 | 0.2421 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0536 | -0.7941 |
| proposed_vs_candidate_no_context | persona_style | -0.0057 | -0.0124 |
| proposed_vs_candidate_no_context | distinct1 | -0.0041 | -0.0044 |
| proposed_vs_candidate_no_context | length_score | 0.0028 | 0.0104 |
| proposed_vs_candidate_no_context | sentence_score | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | overall_quality | -0.0045 | -0.0210 |
| proposed_vs_baseline_no_context | context_relevance | -0.0018 | -0.0437 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0954 | -0.4851 |
| proposed_vs_baseline_no_context | naturalness | -0.0837 | -0.0948 |
| proposed_vs_baseline_no_context | context_keyword_coverage | -0.0035 | -0.0846 |
| proposed_vs_baseline_no_context | context_overlap | 0.0020 | 0.0464 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0940 | -0.8713 |
| proposed_vs_baseline_no_context | persona_style | -0.1006 | -0.1826 |
| proposed_vs_baseline_no_context | distinct1 | -0.0389 | -0.0401 |
| proposed_vs_baseline_no_context | length_score | -0.2750 | -0.5051 |
| proposed_vs_baseline_no_context | sentence_score | -0.1312 | -0.1537 |
| proposed_vs_baseline_no_context | overall_quality | -0.0519 | -0.1995 |
| controlled_vs_proposed_raw | context_relevance | 0.2308 | 5.7801 |
| controlled_vs_proposed_raw | persona_consistency | 0.2447 | 2.4180 |
| controlled_vs_proposed_raw | naturalness | 0.0915 | 0.1145 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.3051 | 8.1210 |
| controlled_vs_proposed_raw | context_overlap | 0.0577 | 1.2680 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2679 | 19.2857 |
| controlled_vs_proposed_raw | persona_style | 0.1522 | 0.3379 |
| controlled_vs_proposed_raw | distinct1 | -0.0053 | -0.0057 |
| controlled_vs_proposed_raw | length_score | 0.3431 | 1.2732 |
| controlled_vs_proposed_raw | sentence_score | 0.2500 | 0.3458 |
| controlled_vs_proposed_raw | overall_quality | 0.2153 | 1.0333 |
| controlled_vs_candidate_no_context | context_relevance | 0.2571 | 18.8593 |
| controlled_vs_candidate_no_context | persona_consistency | 0.2007 | 1.3825 |
| controlled_vs_candidate_no_context | naturalness | 0.0904 | 0.1130 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3388 | 89.4500 |
| controlled_vs_candidate_no_context | context_overlap | 0.0665 | 1.8170 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2143 | 3.1765 |
| controlled_vs_candidate_no_context | persona_style | 0.1465 | 0.3212 |
| controlled_vs_candidate_no_context | distinct1 | -0.0095 | -0.0101 |
| controlled_vs_candidate_no_context | length_score | 0.3458 | 1.2969 |
| controlled_vs_candidate_no_context | sentence_score | 0.2500 | 0.3458 |
| controlled_vs_candidate_no_context | overall_quality | 0.2108 | 0.9907 |
| controlled_vs_baseline_no_context | context_relevance | 0.2290 | 5.4837 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1494 | 0.7598 |
| controlled_vs_baseline_no_context | naturalness | 0.0078 | 0.0088 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3016 | 7.3492 |
| controlled_vs_baseline_no_context | context_overlap | 0.0597 | 1.3732 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1738 | 1.6103 |
| controlled_vs_baseline_no_context | persona_style | 0.0516 | 0.0936 |
| controlled_vs_baseline_no_context | distinct1 | -0.0443 | -0.0456 |
| controlled_vs_baseline_no_context | length_score | 0.0681 | 0.1250 |
| controlled_vs_baseline_no_context | sentence_score | 0.1188 | 0.1390 |
| controlled_vs_baseline_no_context | overall_quality | 0.1634 | 0.6276 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0005 | 0.0019 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0071 | 0.0206 |
| controlled_alt_vs_controlled_default | naturalness | -0.0235 | -0.0264 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0035 | -0.0101 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0098 | 0.0954 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0141 | 0.0500 |
| controlled_alt_vs_controlled_default | persona_style | -0.0207 | -0.0344 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0047 | 0.0051 |
| controlled_alt_vs_controlled_default | length_score | -0.0819 | -0.1338 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0896 | -0.0921 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0018 | -0.0043 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.2314 | 5.7932 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.2519 | 2.4884 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0680 | 0.0851 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.3016 | 8.0286 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0675 | 1.4844 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.2819 | 20.3000 |
| controlled_alt_vs_proposed_raw | persona_style | 0.1315 | 0.2919 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0006 | -0.0007 |
| controlled_alt_vs_proposed_raw | length_score | 0.2611 | 0.9691 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1604 | 0.2219 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.2135 | 1.0247 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2577 | 18.8975 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.2079 | 1.4315 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0669 | 0.0836 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3354 | 88.5333 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0764 | 2.0858 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.2284 | 3.3853 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.1258 | 0.2758 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0048 | -0.0051 |
| controlled_alt_vs_candidate_no_context | length_score | 0.2639 | 0.9896 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1604 | 0.2219 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.2090 | 0.9822 |
| controlled_alt_vs_baseline_no_context | context_relevance | 0.2295 | 5.4962 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 0.1565 | 0.7961 |
| controlled_alt_vs_baseline_no_context | naturalness | -0.0157 | -0.0178 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 0.2981 | 7.2646 |
| controlled_alt_vs_baseline_no_context | context_overlap | 0.0695 | 1.5996 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 0.1879 | 1.7408 |
| controlled_alt_vs_baseline_no_context | persona_style | 0.0309 | 0.0560 |
| controlled_alt_vs_baseline_no_context | distinct1 | -0.0396 | -0.0408 |
| controlled_alt_vs_baseline_no_context | length_score | -0.0139 | -0.0255 |
| controlled_alt_vs_baseline_no_context | sentence_score | 0.0292 | 0.0341 |
| controlled_alt_vs_baseline_no_context | overall_quality | 0.1616 | 0.6207 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2290 | 5.4837 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1494 | 0.7598 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 0.0078 | 0.0088 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3016 | 7.3492 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0597 | 1.3732 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1738 | 1.6103 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0516 | 0.0936 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0443 | -0.0456 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.0681 | 0.1250 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.1188 | 0.1390 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1634 | 0.6276 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0263 | (0.0061, 0.0519) | 0.0013 | 0.0263 | (0.0088, 0.0463) | 0.0010 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0440 | (-0.1148, 0.0158) | 0.9087 | -0.0440 | (-0.1460, 0.0030) | 0.7423 |
| proposed_vs_candidate_no_context | naturalness | -0.0011 | (-0.0321, 0.0327) | 0.5390 | -0.0011 | (-0.0335, 0.0302) | 0.5097 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0338 | (0.0076, 0.0644) | 0.0043 | 0.0338 | (0.0106, 0.0594) | 0.0003 |
| proposed_vs_candidate_no_context | context_overlap | 0.0089 | (0.0009, 0.0189) | 0.0150 | 0.0089 | (0.0016, 0.0170) | 0.0097 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0536 | (-0.1409, 0.0139) | 0.9290 | -0.0536 | (-0.1753, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | persona_style | -0.0057 | (-0.0436, 0.0325) | 0.6440 | -0.0057 | (-0.0300, 0.0148) | 0.7223 |
| proposed_vs_candidate_no_context | distinct1 | -0.0041 | (-0.0196, 0.0109) | 0.6960 | -0.0041 | (-0.0218, 0.0177) | 0.6780 |
| proposed_vs_candidate_no_context | length_score | 0.0028 | (-0.1208, 0.1389) | 0.4867 | 0.0028 | (-0.0962, 0.1116) | 0.4650 |
| proposed_vs_candidate_no_context | sentence_score | 0.0000 | (-0.0729, 0.0729) | 0.5813 | 0.0000 | (-0.0750, 0.0700) | 0.5403 |
| proposed_vs_candidate_no_context | overall_quality | -0.0045 | (-0.0439, 0.0278) | 0.6033 | -0.0045 | (-0.0501, 0.0256) | 0.6103 |
| proposed_vs_baseline_no_context | context_relevance | -0.0018 | (-0.0214, 0.0191) | 0.5870 | -0.0018 | (-0.0151, 0.0098) | 0.6217 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0954 | (-0.1651, -0.0306) | 0.9993 | -0.0954 | (-0.2081, 0.0025) | 0.9720 |
| proposed_vs_baseline_no_context | naturalness | -0.0837 | (-0.1176, -0.0484) | 1.0000 | -0.0837 | (-0.1207, -0.0243) | 0.9947 |
| proposed_vs_baseline_no_context | context_keyword_coverage | -0.0035 | (-0.0300, 0.0227) | 0.6033 | -0.0035 | (-0.0213, 0.0130) | 0.6597 |
| proposed_vs_baseline_no_context | context_overlap | 0.0020 | (-0.0085, 0.0136) | 0.3770 | 0.0020 | (-0.0116, 0.0169) | 0.4233 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0940 | (-0.1819, -0.0143) | 0.9897 | -0.0940 | (-0.2448, 0.0173) | 0.9417 |
| proposed_vs_baseline_no_context | persona_style | -0.1006 | (-0.1984, -0.0116) | 0.9913 | -0.1006 | (-0.2528, 0.0192) | 0.9230 |
| proposed_vs_baseline_no_context | distinct1 | -0.0389 | (-0.0566, -0.0211) | 1.0000 | -0.0389 | (-0.0561, -0.0147) | 0.9983 |
| proposed_vs_baseline_no_context | length_score | -0.2750 | (-0.4042, -0.1444) | 0.9997 | -0.2750 | (-0.4161, -0.0650) | 0.9933 |
| proposed_vs_baseline_no_context | sentence_score | -0.1313 | (-0.2188, -0.0437) | 0.9990 | -0.1313 | (-0.2074, 0.0000) | 0.9827 |
| proposed_vs_baseline_no_context | overall_quality | -0.0519 | (-0.0848, -0.0204) | 0.9997 | -0.0519 | (-0.1036, -0.0082) | 0.9890 |
| controlled_vs_proposed_raw | context_relevance | 0.2308 | (0.1946, 0.2700) | 0.0000 | 0.2308 | (0.2032, 0.2595) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.2447 | (0.1789, 0.3098) | 0.0000 | 0.2447 | (0.1669, 0.3233) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0915 | (0.0385, 0.1376) | 0.0003 | 0.0915 | (0.0189, 0.1343) | 0.0103 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.3051 | (0.2571, 0.3563) | 0.0000 | 0.3051 | (0.2725, 0.3378) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0577 | (0.0434, 0.0731) | 0.0000 | 0.0577 | (0.0417, 0.0768) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2679 | (0.1907, 0.3482) | 0.0000 | 0.2679 | (0.1830, 0.3700) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.1522 | (0.0598, 0.2539) | 0.0003 | 0.1522 | (0.0106, 0.3136) | 0.0040 |
| controlled_vs_proposed_raw | distinct1 | -0.0053 | (-0.0246, 0.0135) | 0.6950 | -0.0053 | (-0.0366, 0.0142) | 0.6683 |
| controlled_vs_proposed_raw | length_score | 0.3431 | (0.1611, 0.5153) | 0.0000 | 0.3431 | (0.0968, 0.4897) | 0.0027 |
| controlled_vs_proposed_raw | sentence_score | 0.2500 | (0.1521, 0.3354) | 0.0000 | 0.2500 | (0.0950, 0.3348) | 0.0017 |
| controlled_vs_proposed_raw | overall_quality | 0.2153 | (0.1808, 0.2504) | 0.0000 | 0.2153 | (0.1857, 0.2466) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2571 | (0.2195, 0.2951) | 0.0000 | 0.2571 | (0.2315, 0.2867) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.2007 | (0.1214, 0.2768) | 0.0000 | 0.2007 | (0.1318, 0.2601) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.0904 | (0.0437, 0.1334) | 0.0000 | 0.0904 | (0.0406, 0.1222) | 0.0013 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3388 | (0.2903, 0.3893) | 0.0000 | 0.3388 | (0.3062, 0.3756) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0665 | (0.0519, 0.0820) | 0.0000 | 0.0665 | (0.0517, 0.0849) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2143 | (0.1214, 0.3079) | 0.0000 | 0.2143 | (0.1446, 0.2877) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.1465 | (0.0522, 0.2518) | 0.0003 | 0.1465 | (0.0074, 0.3058) | 0.0240 |
| controlled_vs_candidate_no_context | distinct1 | -0.0095 | (-0.0278, 0.0078) | 0.8583 | -0.0095 | (-0.0321, 0.0098) | 0.8173 |
| controlled_vs_candidate_no_context | length_score | 0.3458 | (0.1736, 0.5069) | 0.0000 | 0.3458 | (0.1767, 0.4565) | 0.0000 |
| controlled_vs_candidate_no_context | sentence_score | 0.2500 | (0.1500, 0.3208) | 0.0000 | 0.2500 | (0.1368, 0.3259) | 0.0000 |
| controlled_vs_candidate_no_context | overall_quality | 0.2108 | (0.1767, 0.2456) | 0.0000 | 0.2108 | (0.1864, 0.2321) | 0.0000 |
| controlled_vs_baseline_no_context | context_relevance | 0.2290 | (0.1916, 0.2705) | 0.0000 | 0.2290 | (0.2027, 0.2595) | 0.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1494 | (0.0843, 0.2155) | 0.0000 | 0.1494 | (0.0902, 0.2051) | 0.0000 |
| controlled_vs_baseline_no_context | naturalness | 0.0078 | (-0.0243, 0.0382) | 0.2957 | 0.0078 | (-0.0169, 0.0327) | 0.2803 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3016 | (0.2504, 0.3546) | 0.0000 | 0.3016 | (0.2633, 0.3432) | 0.0000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0597 | (0.0459, 0.0733) | 0.0000 | 0.0597 | (0.0492, 0.0712) | 0.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1738 | (0.0934, 0.2595) | 0.0000 | 0.1738 | (0.1006, 0.2488) | 0.0000 |
| controlled_vs_baseline_no_context | persona_style | 0.0516 | (0.0139, 0.0997) | 0.0030 | 0.0516 | (0.0000, 0.1179) | 0.0253 |
| controlled_vs_baseline_no_context | distinct1 | -0.0443 | (-0.0599, -0.0278) | 1.0000 | -0.0443 | (-0.0547, -0.0358) | 1.0000 |
| controlled_vs_baseline_no_context | length_score | 0.0681 | (-0.0653, 0.1958) | 0.1693 | 0.0681 | (-0.0490, 0.1950) | 0.1333 |
| controlled_vs_baseline_no_context | sentence_score | 0.1187 | (0.0458, 0.1917) | 0.0000 | 0.1187 | (0.0548, 0.1771) | 0.0000 |
| controlled_vs_baseline_no_context | overall_quality | 0.1634 | (0.1323, 0.1946) | 0.0000 | 0.1634 | (0.1389, 0.1875) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0005 | (-0.0450, 0.0449) | 0.5103 | 0.0005 | (-0.0407, 0.0531) | 0.4933 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0071 | (-0.0576, 0.0695) | 0.4097 | 0.0071 | (-0.0339, 0.0524) | 0.3807 |
| controlled_alt_vs_controlled_default | naturalness | -0.0235 | (-0.0588, 0.0129) | 0.8937 | -0.0235 | (-0.0709, 0.0149) | 0.8627 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0035 | (-0.0672, 0.0606) | 0.5487 | -0.0035 | (-0.0573, 0.0726) | 0.5270 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0098 | (-0.0057, 0.0252) | 0.1063 | 0.0098 | (-0.0052, 0.0225) | 0.0877 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0141 | (-0.0691, 0.0950) | 0.3670 | 0.0141 | (-0.0385, 0.0693) | 0.2993 |
| controlled_alt_vs_controlled_default | persona_style | -0.0207 | (-0.0712, 0.0273) | 0.8003 | -0.0207 | (-0.0787, 0.0228) | 0.7743 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0047 | (-0.0108, 0.0202) | 0.2710 | 0.0047 | (-0.0091, 0.0160) | 0.2200 |
| controlled_alt_vs_controlled_default | length_score | -0.0819 | (-0.2444, 0.0833) | 0.8397 | -0.0819 | (-0.2825, 0.0810) | 0.8117 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0896 | (-0.1604, -0.0208) | 0.9953 | -0.0896 | (-0.1333, -0.0480) | 1.0000 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0018 | (-0.0337, 0.0291) | 0.5413 | -0.0018 | (-0.0264, 0.0249) | 0.5327 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.2314 | (0.1857, 0.2803) | 0.0000 | 0.2314 | (0.1844, 0.2914) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.2519 | (0.1755, 0.3340) | 0.0000 | 0.2519 | (0.1572, 0.3646) | 0.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0680 | (0.0189, 0.1144) | 0.0050 | 0.0680 | (-0.0065, 0.1203) | 0.0370 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.3016 | (0.2395, 0.3636) | 0.0000 | 0.3016 | (0.2388, 0.3856) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0675 | (0.0533, 0.0805) | 0.0000 | 0.0675 | (0.0558, 0.0819) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.2819 | (0.1907, 0.3829) | 0.0000 | 0.2819 | (0.1777, 0.4158) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 0.1315 | (0.0392, 0.2320) | 0.0003 | 0.1315 | (-0.0012, 0.2849) | 0.0287 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0006 | (-0.0216, 0.0201) | 0.5080 | -0.0006 | (-0.0284, 0.0204) | 0.5280 |
| controlled_alt_vs_proposed_raw | length_score | 0.2611 | (0.0736, 0.4319) | 0.0043 | 0.2611 | (-0.0148, 0.4591) | 0.0323 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1604 | (0.0583, 0.2479) | 0.0030 | 0.1604 | (0.0000, 0.2626) | 0.0297 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.2135 | (0.1764, 0.2540) | 0.0000 | 0.2135 | (0.1792, 0.2523) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2577 | (0.2152, 0.3001) | 0.0000 | 0.2577 | (0.2011, 0.3277) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.2079 | (0.1409, 0.2779) | 0.0000 | 0.2079 | (0.1397, 0.2848) | 0.0000 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0669 | (0.0228, 0.1105) | 0.0013 | 0.0669 | (-0.0022, 0.1184) | 0.0300 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3354 | (0.2776, 0.3931) | 0.0000 | 0.3354 | (0.2593, 0.4266) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0764 | (0.0616, 0.0902) | 0.0000 | 0.0764 | (0.0623, 0.0925) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.2284 | (0.1520, 0.3083) | 0.0000 | 0.2284 | (0.1582, 0.3182) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.1258 | (0.0248, 0.2356) | 0.0107 | 0.1258 | (-0.0015, 0.2775) | 0.0327 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0048 | (-0.0280, 0.0175) | 0.6560 | -0.0048 | (-0.0366, 0.0197) | 0.6310 |
| controlled_alt_vs_candidate_no_context | length_score | 0.2639 | (0.0806, 0.4292) | 0.0020 | 0.2639 | (0.0106, 0.4560) | 0.0207 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1604 | (0.0437, 0.2625) | 0.0020 | 0.1604 | (0.0368, 0.2534) | 0.0080 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.2090 | (0.1778, 0.2437) | 0.0000 | 0.2090 | (0.1758, 0.2485) | 0.0000 |
| controlled_alt_vs_baseline_no_context | context_relevance | 0.2295 | (0.1874, 0.2742) | 0.0000 | 0.2295 | (0.1809, 0.2929) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 0.1565 | (0.0939, 0.2226) | 0.0000 | 0.1565 | (0.1139, 0.2161) | 0.0000 |
| controlled_alt_vs_baseline_no_context | naturalness | -0.0157 | (-0.0506, 0.0217) | 0.7967 | -0.0157 | (-0.0517, 0.0086) | 0.8613 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 0.2981 | (0.2389, 0.3567) | 0.0000 | 0.2981 | (0.2310, 0.3876) | 0.0000 |
| controlled_alt_vs_baseline_no_context | context_overlap | 0.0695 | (0.0544, 0.0850) | 0.0000 | 0.0695 | (0.0586, 0.0812) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 0.1879 | (0.1083, 0.2772) | 0.0000 | 0.1879 | (0.1291, 0.2742) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_style | 0.0309 | (-0.0099, 0.0767) | 0.0790 | 0.0309 | (-0.0181, 0.0876) | 0.1180 |
| controlled_alt_vs_baseline_no_context | distinct1 | -0.0396 | (-0.0545, -0.0245) | 1.0000 | -0.0396 | (-0.0539, -0.0258) | 1.0000 |
| controlled_alt_vs_baseline_no_context | length_score | -0.0139 | (-0.1722, 0.1375) | 0.5693 | -0.0139 | (-0.1826, 0.1060) | 0.5497 |
| controlled_alt_vs_baseline_no_context | sentence_score | 0.0292 | (-0.0441, 0.1021) | 0.2763 | 0.0292 | (-0.0553, 0.0966) | 0.3117 |
| controlled_alt_vs_baseline_no_context | overall_quality | 0.1616 | (0.1244, 0.1990) | 0.0000 | 0.1616 | (0.1280, 0.2037) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2290 | (0.1890, 0.2701) | 0.0000 | 0.2290 | (0.2025, 0.2583) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1494 | (0.0880, 0.2152) | 0.0000 | 0.1494 | (0.0927, 0.2073) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 0.0078 | (-0.0237, 0.0375) | 0.3003 | 0.0078 | (-0.0174, 0.0342) | 0.2707 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3016 | (0.2528, 0.3575) | 0.0000 | 0.3016 | (0.2629, 0.3447) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0597 | (0.0471, 0.0738) | 0.0000 | 0.0597 | (0.0488, 0.0711) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1738 | (0.0952, 0.2568) | 0.0000 | 0.1738 | (0.1014, 0.2468) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0516 | (0.0139, 0.0962) | 0.0013 | 0.0516 | (0.0083, 0.1172) | 0.0210 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0443 | (-0.0597, -0.0273) | 1.0000 | -0.0443 | (-0.0548, -0.0360) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.0681 | (-0.0667, 0.1986) | 0.1627 | 0.0681 | (-0.0474, 0.2046) | 0.1273 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.1187 | (0.0458, 0.1917) | 0.0007 | 0.1187 | (0.0525, 0.1800) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1634 | (0.1319, 0.1961) | 0.0000 | 0.1634 | (0.1380, 0.1877) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 7 | 2 | 15 | 0.6042 | 0.7778 |
| proposed_vs_candidate_no_context | persona_consistency | 2 | 4 | 18 | 0.4583 | 0.3333 |
| proposed_vs_candidate_no_context | naturalness | 5 | 5 | 14 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 5 | 0 | 19 | 0.6042 | 1.0000 |
| proposed_vs_candidate_no_context | context_overlap | 6 | 3 | 15 | 0.5625 | 0.6667 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 1 | 3 | 20 | 0.4583 | 0.2500 |
| proposed_vs_candidate_no_context | persona_style | 1 | 2 | 21 | 0.4792 | 0.3333 |
| proposed_vs_candidate_no_context | distinct1 | 3 | 6 | 15 | 0.4375 | 0.3333 |
| proposed_vs_candidate_no_context | length_score | 5 | 5 | 14 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | sentence_score | 3 | 3 | 18 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | overall_quality | 7 | 3 | 14 | 0.5833 | 0.7000 |
| proposed_vs_baseline_no_context | context_relevance | 10 | 14 | 0 | 0.4167 | 0.4167 |
| proposed_vs_baseline_no_context | persona_consistency | 2 | 13 | 9 | 0.2708 | 0.1333 |
| proposed_vs_baseline_no_context | naturalness | 6 | 18 | 0 | 0.2500 | 0.2500 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 3 | 4 | 17 | 0.4792 | 0.4286 |
| proposed_vs_baseline_no_context | context_overlap | 10 | 14 | 0 | 0.4167 | 0.4167 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 1 | 8 | 15 | 0.3542 | 0.1111 |
| proposed_vs_baseline_no_context | persona_style | 1 | 6 | 17 | 0.3958 | 0.1429 |
| proposed_vs_baseline_no_context | distinct1 | 5 | 18 | 1 | 0.2292 | 0.2174 |
| proposed_vs_baseline_no_context | length_score | 6 | 18 | 0 | 0.2500 | 0.2500 |
| proposed_vs_baseline_no_context | sentence_score | 2 | 11 | 11 | 0.3125 | 0.1538 |
| proposed_vs_baseline_no_context | overall_quality | 5 | 19 | 0 | 0.2083 | 0.2083 |
| controlled_vs_proposed_raw | context_relevance | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_proposed_raw | persona_consistency | 21 | 1 | 2 | 0.9167 | 0.9545 |
| controlled_vs_proposed_raw | naturalness | 19 | 5 | 0 | 0.7917 | 0.7917 |
| controlled_vs_proposed_raw | context_keyword_coverage | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_proposed_raw | context_overlap | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 21 | 1 | 2 | 0.9167 | 0.9545 |
| controlled_vs_proposed_raw | persona_style | 8 | 0 | 16 | 0.6667 | 1.0000 |
| controlled_vs_proposed_raw | distinct1 | 12 | 11 | 1 | 0.5208 | 0.5217 |
| controlled_vs_proposed_raw | length_score | 17 | 6 | 1 | 0.7292 | 0.7391 |
| controlled_vs_proposed_raw | sentence_score | 19 | 1 | 4 | 0.8750 | 0.9500 |
| controlled_vs_proposed_raw | overall_quality | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_relevance | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 19 | 3 | 2 | 0.8333 | 0.8636 |
| controlled_vs_candidate_no_context | naturalness | 20 | 4 | 0 | 0.8333 | 0.8333 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 19 | 3 | 2 | 0.8333 | 0.8636 |
| controlled_vs_candidate_no_context | persona_style | 7 | 1 | 16 | 0.6250 | 0.8750 |
| controlled_vs_candidate_no_context | distinct1 | 12 | 11 | 1 | 0.5208 | 0.5217 |
| controlled_vs_candidate_no_context | length_score | 18 | 4 | 2 | 0.7917 | 0.8182 |
| controlled_vs_candidate_no_context | sentence_score | 19 | 1 | 4 | 0.8750 | 0.9500 |
| controlled_vs_candidate_no_context | overall_quality | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_vs_baseline_no_context | context_relevance | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 17 | 3 | 4 | 0.7917 | 0.8500 |
| controlled_vs_baseline_no_context | naturalness | 14 | 10 | 0 | 0.5833 | 0.5833 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | context_overlap | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 17 | 3 | 4 | 0.7917 | 0.8500 |
| controlled_vs_baseline_no_context | persona_style | 5 | 0 | 19 | 0.6042 | 1.0000 |
| controlled_vs_baseline_no_context | distinct1 | 4 | 20 | 0 | 0.1667 | 0.1667 |
| controlled_vs_baseline_no_context | length_score | 16 | 8 | 0 | 0.6667 | 0.6667 |
| controlled_vs_baseline_no_context | sentence_score | 9 | 1 | 14 | 0.6667 | 0.9000 |
| controlled_vs_baseline_no_context | overall_quality | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 10 | 12 | 2 | 0.4583 | 0.4545 |
| controlled_alt_vs_controlled_default | persona_consistency | 6 | 6 | 12 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | naturalness | 10 | 12 | 2 | 0.4583 | 0.4545 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 7 | 7 | 10 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | context_overlap | 13 | 9 | 2 | 0.5833 | 0.5909 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 4 | 3 | 17 | 0.5208 | 0.5714 |
| controlled_alt_vs_controlled_default | persona_style | 3 | 5 | 16 | 0.4583 | 0.3750 |
| controlled_alt_vs_controlled_default | distinct1 | 13 | 9 | 2 | 0.5833 | 0.5909 |
| controlled_alt_vs_controlled_default | length_score | 9 | 13 | 2 | 0.4167 | 0.4091 |
| controlled_alt_vs_controlled_default | sentence_score | 1 | 7 | 16 | 0.3750 | 0.1250 |
| controlled_alt_vs_controlled_default | overall_quality | 11 | 11 | 2 | 0.5000 | 0.5000 |
| controlled_alt_vs_proposed_raw | context_relevance | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_alt_vs_proposed_raw | persona_consistency | 21 | 1 | 2 | 0.9167 | 0.9545 |
| controlled_alt_vs_proposed_raw | naturalness | 18 | 6 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 22 | 0 | 2 | 0.9583 | 1.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 21 | 1 | 2 | 0.9167 | 0.9545 |
| controlled_alt_vs_proposed_raw | persona_style | 9 | 2 | 13 | 0.6458 | 0.8182 |
| controlled_alt_vs_proposed_raw | distinct1 | 12 | 12 | 0 | 0.5000 | 0.5000 |
| controlled_alt_vs_proposed_raw | length_score | 17 | 7 | 0 | 0.7083 | 0.7083 |
| controlled_alt_vs_proposed_raw | sentence_score | 14 | 3 | 7 | 0.7292 | 0.8235 |
| controlled_alt_vs_proposed_raw | overall_quality | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 20 | 2 | 2 | 0.8750 | 0.9091 |
| controlled_alt_vs_candidate_no_context | naturalness | 17 | 7 | 0 | 0.7083 | 0.7083 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 23 | 0 | 1 | 0.9792 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 20 | 2 | 2 | 0.8750 | 0.9091 |
| controlled_alt_vs_candidate_no_context | persona_style | 9 | 4 | 11 | 0.6042 | 0.6923 |
| controlled_alt_vs_candidate_no_context | distinct1 | 13 | 11 | 0 | 0.5417 | 0.5417 |
| controlled_alt_vs_candidate_no_context | length_score | 19 | 5 | 0 | 0.7917 | 0.7917 |
| controlled_alt_vs_candidate_no_context | sentence_score | 15 | 4 | 5 | 0.7292 | 0.7895 |
| controlled_alt_vs_candidate_no_context | overall_quality | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context | context_relevance | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 16 | 2 | 6 | 0.7917 | 0.8889 |
| controlled_alt_vs_baseline_no_context | naturalness | 8 | 16 | 0 | 0.3333 | 0.3333 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 23 | 0 | 1 | 0.9792 | 1.0000 |
| controlled_alt_vs_baseline_no_context | context_overlap | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 16 | 2 | 6 | 0.7917 | 0.8889 |
| controlled_alt_vs_baseline_no_context | persona_style | 4 | 2 | 18 | 0.5417 | 0.6667 |
| controlled_alt_vs_baseline_no_context | distinct1 | 4 | 20 | 0 | 0.1667 | 0.1667 |
| controlled_alt_vs_baseline_no_context | length_score | 13 | 10 | 1 | 0.5625 | 0.5652 |
| controlled_alt_vs_baseline_no_context | sentence_score | 5 | 3 | 16 | 0.5417 | 0.6250 |
| controlled_alt_vs_baseline_no_context | overall_quality | 23 | 1 | 0 | 0.9583 | 0.9583 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 24 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 17 | 3 | 4 | 0.7917 | 0.8500 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 14 | 10 | 0 | 0.5833 | 0.5833 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 24 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 24 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 17 | 3 | 4 | 0.7917 | 0.8500 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 5 | 0 | 19 | 0.6042 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | 4 | 20 | 0 | 0.1667 | 0.1667 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 16 | 8 | 0 | 0.6667 | 0.6667 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 9 | 1 | 14 | 0.6667 | 0.9000 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 24 | 0 | 0 | 1.0000 | 1.0000 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.3750 | 0.2917 | 0.7083 |
| proposed_contextual_controlled_alt | 0.0000 | 0.0000 | 0.1667 | 0.5000 | 0.5000 |
| proposed_contextual | 0.0000 | 0.0000 | 0.7083 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.6250 | 0.0000 | 0.0000 |
| baseline_no_context | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `8`
- Unique template signatures: `22`
- Template signature ratio: `0.9167`
- Effective sample size by source clustering: `7.20`
- Effective sample size by template-signature clustering: `20.57`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: unavailable (disabled_by_flag).

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
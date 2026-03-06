# Proposal Alignment Evaluation Report

- Run ID: `20260306T085717Z`
- Generated: `2026-03-06T09:11:37.169700+00:00`
- Scenarios: `artifacts\proposal_control_tuning\preflight_blend\20260306T085717Z\seed_runs\seed_29\20260306T085717Z\scenarios.jsonl`
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
| proposed_contextual_controlled | 0.2826 (0.2458, 0.3220) | 0.3672 (0.2944, 0.4439) | 0.8648 (0.8311, 0.8958) | 0.3926 (0.3671, 0.4211) | 0.0749 |
| proposed_contextual_controlled_alt | 0.2680 (0.2305, 0.3086) | 0.3524 (0.2991, 0.4153) | 0.8573 (0.8282, 0.8866) | 0.3813 (0.3565, 0.4088) | 0.0871 |
| proposed_contextual | 0.0524 (0.0257, 0.0866) | 0.1384 (0.0893, 0.1976) | 0.7886 (0.7645, 0.8191) | 0.2083 (0.1817, 0.2377) | 0.0503 |
| candidate_no_context | 0.0311 (0.0166, 0.0494) | 0.2147 (0.1526, 0.2813) | 0.8193 (0.7846, 0.8543) | 0.2269 (0.1992, 0.2572) | 0.0242 |
| baseline_no_context | 0.0365 (0.0229, 0.0522) | 0.1988 (0.1531, 0.2466) | 0.8920 (0.8732, 0.9118) | 0.2395 (0.2239, 0.2556) | 0.0497 |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0213 | 0.6840 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0763 | -0.3553 |
| proposed_vs_candidate_no_context | naturalness | -0.0307 | -0.0374 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0284 | 1.0714 |
| proposed_vs_candidate_no_context | context_overlap | 0.0046 | 0.1105 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0861 | -0.6036 |
| proposed_vs_candidate_no_context | persona_style | -0.0371 | -0.0737 |
| proposed_vs_candidate_no_context | distinct1 | -0.0053 | -0.0057 |
| proposed_vs_candidate_no_context | length_score | -0.1208 | -0.3671 |
| proposed_vs_candidate_no_context | sentence_score | -0.0438 | -0.0550 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0261 | 1.0761 |
| proposed_vs_candidate_no_context | overall_quality | -0.0186 | -0.0821 |
| proposed_vs_baseline_no_context | context_relevance | 0.0159 | 0.4353 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0604 | -0.3039 |
| proposed_vs_baseline_no_context | naturalness | -0.1034 | -0.1159 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0205 | 0.5934 |
| proposed_vs_baseline_no_context | context_overlap | 0.0052 | 0.1266 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0552 | -0.4938 |
| proposed_vs_baseline_no_context | persona_style | -0.0814 | -0.1488 |
| proposed_vs_baseline_no_context | distinct1 | -0.0501 | -0.0511 |
| proposed_vs_baseline_no_context | length_score | -0.3806 | -0.6462 |
| proposed_vs_baseline_no_context | sentence_score | -0.0729 | -0.0884 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0007 | 0.0135 |
| proposed_vs_baseline_no_context | overall_quality | -0.0312 | -0.1303 |
| controlled_vs_proposed_raw | context_relevance | 0.2302 | 4.3959 |
| controlled_vs_proposed_raw | persona_consistency | 0.2288 | 1.6532 |
| controlled_vs_proposed_raw | naturalness | 0.0762 | 0.0966 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.3023 | 5.5031 |
| controlled_vs_proposed_raw | context_overlap | 0.0621 | 1.3388 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2521 | 4.4590 |
| controlled_vs_proposed_raw | persona_style | 0.1356 | 0.2909 |
| controlled_vs_proposed_raw | distinct1 | 0.0040 | 0.0044 |
| controlled_vs_proposed_raw | length_score | 0.2801 | 1.3443 |
| controlled_vs_proposed_raw | sentence_score | 0.1870 | 0.2487 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0246 | 0.4891 |
| controlled_vs_proposed_raw | overall_quality | 0.1843 | 0.8848 |
| controlled_vs_candidate_no_context | context_relevance | 0.2515 | 8.0867 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1525 | 0.7104 |
| controlled_vs_candidate_no_context | naturalness | 0.0455 | 0.0556 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3307 | 12.4708 |
| controlled_vs_candidate_no_context | context_overlap | 0.0668 | 1.5972 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1660 | 1.1639 |
| controlled_vs_candidate_no_context | persona_style | 0.0985 | 0.1958 |
| controlled_vs_candidate_no_context | distinct1 | -0.0012 | -0.0013 |
| controlled_vs_candidate_no_context | length_score | 0.1592 | 0.4838 |
| controlled_vs_candidate_no_context | sentence_score | 0.1433 | 0.1801 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0507 | 2.0917 |
| controlled_vs_candidate_no_context | overall_quality | 0.1657 | 0.7300 |
| controlled_vs_baseline_no_context | context_relevance | 0.2461 | 6.7448 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1684 | 0.8470 |
| controlled_vs_baseline_no_context | naturalness | -0.0273 | -0.0305 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3227 | 9.3622 |
| controlled_vs_baseline_no_context | context_overlap | 0.0674 | 1.6350 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1970 | 1.7635 |
| controlled_vs_baseline_no_context | persona_style | 0.0541 | 0.0988 |
| controlled_vs_baseline_no_context | distinct1 | -0.0460 | -0.0470 |
| controlled_vs_baseline_no_context | length_score | -0.1005 | -0.1706 |
| controlled_vs_baseline_no_context | sentence_score | 0.1141 | 0.1383 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0253 | 0.5092 |
| controlled_vs_baseline_no_context | overall_quality | 0.1531 | 0.6392 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0146 | -0.0515 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0148 | -0.0404 |
| controlled_alt_vs_controlled_default | naturalness | -0.0075 | -0.0087 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0228 | -0.0638 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0046 | 0.0425 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0232 | -0.0751 |
| controlled_alt_vs_controlled_default | persona_style | 0.0185 | 0.0308 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0051 | 0.0054 |
| controlled_alt_vs_controlled_default | length_score | -0.0426 | -0.0872 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0120 | -0.0128 |
| controlled_alt_vs_controlled_default | bertscore_f1 | 0.0121 | 0.1618 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0113 | -0.0287 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.2157 | 4.1180 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.2140 | 1.5460 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0686 | 0.0870 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2795 | 5.0885 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0668 | 1.4383 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.2290 | 4.0491 |
| controlled_alt_vs_proposed_raw | persona_style | 0.1541 | 0.3307 |
| controlled_alt_vs_proposed_raw | distinct1 | 0.0091 | 0.0098 |
| controlled_alt_vs_proposed_raw | length_score | 0.2375 | 1.1400 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1750 | 0.2327 |
| controlled_alt_vs_proposed_raw | bertscore_f1 | 0.0367 | 0.7301 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1730 | 0.8308 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2369 | 7.6186 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1377 | 0.6413 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0380 | 0.0464 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3079 | 11.6119 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0714 | 1.7076 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1429 | 1.0014 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.1170 | 0.2327 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0038 | 0.0041 |
| controlled_alt_vs_candidate_no_context | length_score | 0.1167 | 0.3544 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1312 | 0.1649 |
| controlled_alt_vs_candidate_no_context | bertscore_f1 | 0.0628 | 2.5920 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1544 | 0.6804 |
| controlled_alt_vs_baseline_no_context | context_relevance | 0.2315 | 6.3459 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 0.1536 | 0.7724 |
| controlled_alt_vs_baseline_no_context | naturalness | -0.0348 | -0.0390 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 0.2999 | 8.7015 |
| controlled_alt_vs_baseline_no_context | context_overlap | 0.0720 | 1.7471 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 0.1738 | 1.5560 |
| controlled_alt_vs_baseline_no_context | persona_style | 0.0726 | 0.1327 |
| controlled_alt_vs_baseline_no_context | distinct1 | -0.0410 | -0.0418 |
| controlled_alt_vs_baseline_no_context | length_score | -0.1431 | -0.2429 |
| controlled_alt_vs_baseline_no_context | sentence_score | 0.1021 | 0.1237 |
| controlled_alt_vs_baseline_no_context | bertscore_f1 | 0.0374 | 0.7535 |
| controlled_alt_vs_baseline_no_context | overall_quality | 0.1418 | 0.5922 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2461 | 6.7448 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1684 | 0.8470 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0273 | -0.0305 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3227 | 9.3622 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0674 | 1.6350 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1970 | 1.7635 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0541 | 0.0988 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0460 | -0.0470 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | -0.1005 | -0.1706 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.1141 | 0.1383 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0253 | 0.5092 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1531 | 0.6392 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0213 | (-0.0127, 0.0583) | 0.1207 | 0.0213 | (-0.0136, 0.0725) | 0.1100 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0763 | (-0.1600, 0.0004) | 0.9737 | -0.0763 | (-0.1512, -0.0135) | 0.9953 |
| proposed_vs_candidate_no_context | naturalness | -0.0307 | (-0.0745, 0.0148) | 0.9077 | -0.0307 | (-0.0965, 0.0272) | 0.8553 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0284 | (-0.0189, 0.0783) | 0.1320 | 0.0284 | (-0.0182, 0.0988) | 0.1337 |
| proposed_vs_candidate_no_context | context_overlap | 0.0046 | (-0.0040, 0.0154) | 0.1803 | 0.0046 | (-0.0067, 0.0203) | 0.2500 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0861 | (-0.1808, 0.0016) | 0.9707 | -0.0861 | (-0.1850, -0.0126) | 0.9920 |
| proposed_vs_candidate_no_context | persona_style | -0.0371 | (-0.1103, 0.0231) | 0.8593 | -0.0371 | (-0.0904, 0.0000) | 0.9757 |
| proposed_vs_candidate_no_context | distinct1 | -0.0053 | (-0.0236, 0.0144) | 0.7013 | -0.0053 | (-0.0338, 0.0219) | 0.6443 |
| proposed_vs_candidate_no_context | length_score | -0.1208 | (-0.2917, 0.0611) | 0.9050 | -0.1208 | (-0.3704, 0.1254) | 0.8477 |
| proposed_vs_candidate_no_context | sentence_score | -0.0437 | (-0.1167, 0.0292) | 0.9203 | -0.0437 | (-0.1114, 0.0269) | 0.9257 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0261 | (0.0099, 0.0444) | 0.0010 | 0.0261 | (0.0090, 0.0482) | 0.0017 |
| proposed_vs_candidate_no_context | overall_quality | -0.0186 | (-0.0576, 0.0183) | 0.8143 | -0.0186 | (-0.0610, 0.0258) | 0.8180 |
| proposed_vs_baseline_no_context | context_relevance | 0.0159 | (-0.0198, 0.0571) | 0.2247 | 0.0159 | (-0.0220, 0.0682) | 0.2407 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0604 | (-0.1201, -0.0021) | 0.9790 | -0.0604 | (-0.1186, 0.0131) | 0.9540 |
| proposed_vs_baseline_no_context | naturalness | -0.1034 | (-0.1365, -0.0654) | 1.0000 | -0.1034 | (-0.1476, -0.0411) | 0.9993 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0205 | (-0.0303, 0.0740) | 0.2267 | 0.0205 | (-0.0303, 0.0891) | 0.2790 |
| proposed_vs_baseline_no_context | context_overlap | 0.0052 | (-0.0057, 0.0180) | 0.2027 | 0.0052 | (-0.0090, 0.0244) | 0.2623 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0552 | (-0.1322, 0.0151) | 0.9457 | -0.0552 | (-0.1336, 0.0320) | 0.9007 |
| proposed_vs_baseline_no_context | persona_style | -0.0814 | (-0.1827, 0.0110) | 0.9533 | -0.0814 | (-0.2402, 0.0328) | 0.8183 |
| proposed_vs_baseline_no_context | distinct1 | -0.0501 | (-0.0680, -0.0311) | 1.0000 | -0.0501 | (-0.0623, -0.0342) | 1.0000 |
| proposed_vs_baseline_no_context | length_score | -0.3806 | (-0.5042, -0.2458) | 1.0000 | -0.3806 | (-0.5345, -0.1649) | 1.0000 |
| proposed_vs_baseline_no_context | sentence_score | -0.0729 | (-0.1750, 0.0292) | 0.9303 | -0.0729 | (-0.1969, 0.1050) | 0.8377 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0007 | (-0.0212, 0.0261) | 0.4830 | 0.0007 | (-0.0319, 0.0370) | 0.4997 |
| proposed_vs_baseline_no_context | overall_quality | -0.0312 | (-0.0630, 0.0017) | 0.9673 | -0.0312 | (-0.0676, 0.0212) | 0.8950 |
| controlled_vs_proposed_raw | context_relevance | 0.2286 | (0.1927, 0.2690) | 0.0000 | 0.2286 | (0.1911, 0.2776) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.2315 | (0.1674, 0.3052) | 0.0000 | 0.2315 | (0.1623, 0.3273) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0744 | (0.0229, 0.1230) | 0.0013 | 0.0744 | (0.0103, 0.1237) | 0.0133 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2999 | (0.2521, 0.3478) | 0.0000 | 0.2999 | (0.2519, 0.3649) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0624 | (0.0442, 0.0801) | 0.0000 | 0.0624 | (0.0478, 0.0776) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2497 | (0.1749, 0.3470) | 0.0000 | 0.2497 | (0.1819, 0.3762) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.1588 | (0.0510, 0.2799) | 0.0013 | 0.1588 | (-0.0077, 0.3524) | 0.0397 |
| controlled_vs_proposed_raw | distinct1 | 0.0032 | (-0.0195, 0.0245) | 0.3883 | 0.0032 | (-0.0202, 0.0226) | 0.4233 |
| controlled_vs_proposed_raw | length_score | 0.2754 | (0.0609, 0.4696) | 0.0040 | 0.2754 | (0.0353, 0.4788) | 0.0173 |
| controlled_vs_proposed_raw | sentence_score | 0.1826 | (0.0913, 0.2739) | 0.0003 | 0.1826 | (0.0350, 0.2722) | 0.0143 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0259 | (0.0024, 0.0493) | 0.0147 | 0.0259 | (0.0036, 0.0439) | 0.0143 |
| controlled_vs_proposed_raw | overall_quality | 0.1841 | (0.1501, 0.2228) | 0.0000 | 0.1841 | (0.1432, 0.2216) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2508 | (0.2101, 0.2892) | 0.0000 | 0.2508 | (0.2183, 0.2938) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1519 | (0.0741, 0.2298) | 0.0000 | 0.1519 | (0.1248, 0.1875) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.0425 | (-0.0110, 0.0948) | 0.0647 | 0.0425 | (-0.0364, 0.0966) | 0.1377 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3295 | (0.2755, 0.3833) | 0.0000 | 0.3295 | (0.2862, 0.3876) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0672 | (0.0527, 0.0815) | 0.0000 | 0.0672 | (0.0591, 0.0791) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1598 | (0.0669, 0.2509) | 0.0000 | 0.1598 | (0.1203, 0.2103) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.1201 | (0.0283, 0.2274) | 0.0047 | 0.1201 | (-0.0139, 0.2780) | 0.0623 |
| controlled_vs_candidate_no_context | distinct1 | -0.0023 | (-0.0239, 0.0188) | 0.5733 | -0.0023 | (-0.0326, 0.0194) | 0.5720 |
| controlled_vs_candidate_no_context | length_score | 0.1493 | (-0.0638, 0.3594) | 0.0863 | 0.1493 | (-0.1379, 0.3701) | 0.1403 |
| controlled_vs_candidate_no_context | sentence_score | 0.1370 | (0.0609, 0.2283) | 0.0020 | 0.1370 | (0.0000, 0.2414) | 0.0293 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0531 | (0.0291, 0.0786) | 0.0000 | 0.0531 | (0.0392, 0.0701) | 0.0000 |
| controlled_vs_candidate_no_context | overall_quality | 0.1646 | (0.1349, 0.1936) | 0.0000 | 0.1646 | (0.1451, 0.1869) | 0.0000 |
| controlled_vs_baseline_no_context | context_relevance | 0.2448 | (0.2053, 0.2888) | 0.0000 | 0.2448 | (0.2115, 0.2861) | 0.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1685 | (0.1033, 0.2285) | 0.0000 | 0.1685 | (0.1196, 0.2234) | 0.0000 |
| controlled_vs_baseline_no_context | naturalness | -0.0281 | (-0.0686, 0.0138) | 0.9020 | -0.0281 | (-0.0555, -0.0014) | 0.9780 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3212 | (0.2676, 0.3788) | 0.0000 | 0.3212 | (0.2778, 0.3740) | 0.0000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0664 | (0.0539, 0.0801) | 0.0000 | 0.0664 | (0.0569, 0.0768) | 0.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1921 | (0.1118, 0.2673) | 0.0000 | 0.1921 | (0.1299, 0.2659) | 0.0000 |
| controlled_vs_baseline_no_context | persona_style | 0.0738 | (0.0184, 0.1417) | 0.0027 | 0.0738 | (0.0000, 0.2078) | 0.1010 |
| controlled_vs_baseline_no_context | distinct1 | -0.0451 | (-0.0588, -0.0311) | 1.0000 | -0.0451 | (-0.0587, -0.0317) | 1.0000 |
| controlled_vs_baseline_no_context | length_score | -0.1029 | (-0.3116, 0.1000) | 0.8507 | -0.1029 | (-0.2449, 0.0576) | 0.9113 |
| controlled_vs_baseline_no_context | sentence_score | 0.1065 | (0.0304, 0.1826) | 0.0073 | 0.1065 | (0.0152, 0.2167) | 0.0170 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0278 | (-0.0024, 0.0604) | 0.0413 | 0.0278 | (-0.0163, 0.0675) | 0.1017 |
| controlled_vs_baseline_no_context | overall_quality | 0.1525 | (0.1254, 0.1785) | 0.0000 | 0.1525 | (0.1285, 0.1763) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0100 | (-0.0572, 0.0304) | 0.6520 | -0.0100 | (-0.0438, 0.0256) | 0.7093 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0132 | (-0.0672, 0.0423) | 0.6837 | -0.0132 | (-0.0768, 0.0474) | 0.7113 |
| controlled_alt_vs_controlled_default | naturalness | -0.0091 | (-0.0473, 0.0310) | 0.6733 | -0.0091 | (-0.0475, 0.0286) | 0.6737 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0161 | (-0.0738, 0.0356) | 0.7010 | -0.0161 | (-0.0617, 0.0338) | 0.7453 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0042 | (-0.0140, 0.0228) | 0.3160 | 0.0042 | (-0.0063, 0.0138) | 0.1937 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0170 | (-0.0865, 0.0571) | 0.6723 | -0.0170 | (-0.1165, 0.0646) | 0.6413 |
| controlled_alt_vs_controlled_default | persona_style | 0.0020 | (-0.0580, 0.0744) | 0.4693 | 0.0020 | (-0.0995, 0.1176) | 0.4917 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0032 | (-0.0195, 0.0273) | 0.4013 | 0.0032 | (-0.0214, 0.0296) | 0.4033 |
| controlled_alt_vs_controlled_default | length_score | -0.0449 | (-0.2275, 0.1319) | 0.7090 | -0.0449 | (-0.1882, 0.0864) | 0.7383 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0152 | (-0.1065, 0.0761) | 0.6923 | -0.0152 | (-0.0833, 0.0833) | 0.7193 |
| controlled_alt_vs_controlled_default | bertscore_f1 | 0.0101 | (-0.0196, 0.0433) | 0.2563 | 0.0101 | (-0.0151, 0.0431) | 0.2560 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0091 | (-0.0321, 0.0141) | 0.7737 | -0.0091 | (-0.0287, 0.0064) | 0.8640 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.2157 | (0.1709, 0.2610) | 0.0000 | 0.2157 | (0.1766, 0.2623) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.2140 | (0.1441, 0.2880) | 0.0000 | 0.2140 | (0.1267, 0.2784) | 0.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0686 | (0.0288, 0.1065) | 0.0010 | 0.0686 | (0.0125, 0.1135) | 0.0103 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2795 | (0.2214, 0.3365) | 0.0000 | 0.2795 | (0.2318, 0.3377) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0668 | (0.0502, 0.0839) | 0.0000 | 0.0668 | (0.0484, 0.0888) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.2290 | (0.1462, 0.3224) | 0.0000 | 0.2290 | (0.1374, 0.3098) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 0.1541 | (0.0628, 0.2625) | 0.0000 | 0.1541 | (0.0177, 0.3090) | 0.0157 |
| controlled_alt_vs_proposed_raw | distinct1 | 0.0091 | (-0.0062, 0.0240) | 0.1347 | 0.0091 | (-0.0072, 0.0242) | 0.1243 |
| controlled_alt_vs_proposed_raw | length_score | 0.2375 | (0.0889, 0.3847) | 0.0007 | 0.2375 | (0.0317, 0.4290) | 0.0133 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1750 | (0.0729, 0.2625) | 0.0003 | 0.1750 | (0.0700, 0.2414) | 0.0037 |
| controlled_alt_vs_proposed_raw | bertscore_f1 | 0.0367 | (0.0154, 0.0602) | 0.0000 | 0.0367 | (0.0173, 0.0617) | 0.0000 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1730 | (0.1324, 0.2147) | 0.0000 | 0.1730 | (0.1366, 0.2049) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2369 | (0.1948, 0.2786) | 0.0000 | 0.2369 | (0.1833, 0.3001) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1377 | (0.0728, 0.2115) | 0.0000 | 0.1377 | (0.0845, 0.1906) | 0.0000 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0380 | (-0.0134, 0.0888) | 0.0773 | 0.0380 | (-0.0391, 0.1017) | 0.1547 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3079 | (0.2515, 0.3644) | 0.0000 | 0.3079 | (0.2413, 0.3896) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0714 | (0.0549, 0.0885) | 0.0000 | 0.0714 | (0.0553, 0.0910) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1429 | (0.0627, 0.2355) | 0.0000 | 0.1429 | (0.0742, 0.2231) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.1170 | (0.0365, 0.2119) | 0.0003 | 0.1170 | (0.0000, 0.2236) | 0.0253 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0038 | (-0.0192, 0.0276) | 0.3650 | 0.0038 | (-0.0272, 0.0345) | 0.4123 |
| controlled_alt_vs_candidate_no_context | length_score | 0.1167 | (-0.0833, 0.3181) | 0.1397 | 0.1167 | (-0.1870, 0.3782) | 0.2270 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1313 | (0.0292, 0.2188) | 0.0133 | 0.1313 | (0.0525, 0.1983) | 0.0000 |
| controlled_alt_vs_candidate_no_context | bertscore_f1 | 0.0628 | (0.0386, 0.0907) | 0.0000 | 0.0628 | (0.0363, 0.0986) | 0.0000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1544 | (0.1228, 0.1893) | 0.0000 | 0.1544 | (0.1286, 0.1785) | 0.0000 |
| controlled_alt_vs_baseline_no_context | context_relevance | 0.2315 | (0.1919, 0.2709) | 0.0000 | 0.2315 | (0.1834, 0.2876) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 0.1536 | (0.1090, 0.2018) | 0.0000 | 0.1536 | (0.1257, 0.1752) | 0.0000 |
| controlled_alt_vs_baseline_no_context | naturalness | -0.0348 | (-0.0716, 0.0042) | 0.9647 | -0.0348 | (-0.0717, 0.0013) | 0.9707 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 0.2999 | (0.2470, 0.3511) | 0.0000 | 0.2999 | (0.2377, 0.3785) | 0.0000 |
| controlled_alt_vs_baseline_no_context | context_overlap | 0.0720 | (0.0571, 0.0868) | 0.0000 | 0.0720 | (0.0579, 0.0893) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 0.1738 | (0.1212, 0.2327) | 0.0000 | 0.1738 | (0.1356, 0.2072) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_style | 0.0726 | (0.0204, 0.1356) | 0.0010 | 0.0726 | (0.0026, 0.1739) | 0.0227 |
| controlled_alt_vs_baseline_no_context | distinct1 | -0.0410 | (-0.0590, -0.0226) | 1.0000 | -0.0410 | (-0.0582, -0.0190) | 1.0000 |
| controlled_alt_vs_baseline_no_context | length_score | -0.1431 | (-0.3139, 0.0223) | 0.9583 | -0.1431 | (-0.3136, 0.0345) | 0.9367 |
| controlled_alt_vs_baseline_no_context | sentence_score | 0.1021 | (0.0146, 0.1896) | 0.0163 | 0.1021 | (0.0339, 0.2001) | 0.0003 |
| controlled_alt_vs_baseline_no_context | bertscore_f1 | 0.0374 | (0.0074, 0.0668) | 0.0083 | 0.0374 | (0.0110, 0.0677) | 0.0030 |
| controlled_alt_vs_baseline_no_context | overall_quality | 0.1418 | (0.1147, 0.1688) | 0.0000 | 0.1418 | (0.1224, 0.1670) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2448 | (0.2032, 0.2869) | 0.0000 | 0.2448 | (0.2127, 0.2855) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1685 | (0.1066, 0.2285) | 0.0000 | 0.1685 | (0.1164, 0.2242) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0281 | (-0.0697, 0.0130) | 0.9073 | -0.0281 | (-0.0555, 0.0013) | 0.9720 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3212 | (0.2692, 0.3814) | 0.0000 | 0.3212 | (0.2782, 0.3788) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0664 | (0.0534, 0.0798) | 0.0000 | 0.0664 | (0.0566, 0.0774) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1921 | (0.1182, 0.2692) | 0.0000 | 0.1921 | (0.1323, 0.2652) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0738 | (0.0158, 0.1449) | 0.0037 | 0.0738 | (0.0000, 0.2055) | 0.0973 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0451 | (-0.0585, -0.0312) | 1.0000 | -0.0451 | (-0.0587, -0.0317) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | -0.1029 | (-0.2971, 0.0884) | 0.8433 | -0.1029 | (-0.2431, 0.0654) | 0.8970 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.1065 | (0.0304, 0.1826) | 0.0053 | 0.1065 | (0.0134, 0.2211) | 0.0240 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0278 | (-0.0044, 0.0588) | 0.0450 | 0.0278 | (-0.0151, 0.0663) | 0.1043 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1525 | (0.1260, 0.1783) | 0.0000 | 0.1525 | (0.1285, 0.1759) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 8 | 7 | 9 | 0.5208 | 0.5333 |
| proposed_vs_candidate_no_context | persona_consistency | 6 | 8 | 10 | 0.4583 | 0.4286 |
| proposed_vs_candidate_no_context | naturalness | 5 | 10 | 9 | 0.3958 | 0.3333 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 6 | 4 | 14 | 0.5417 | 0.6000 |
| proposed_vs_candidate_no_context | context_overlap | 9 | 6 | 9 | 0.5625 | 0.6000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 4 | 8 | 12 | 0.4167 | 0.3333 |
| proposed_vs_candidate_no_context | persona_style | 3 | 4 | 17 | 0.4792 | 0.4286 |
| proposed_vs_candidate_no_context | distinct1 | 6 | 8 | 10 | 0.4583 | 0.4286 |
| proposed_vs_candidate_no_context | length_score | 6 | 9 | 9 | 0.4375 | 0.4000 |
| proposed_vs_candidate_no_context | sentence_score | 2 | 5 | 17 | 0.4375 | 0.2857 |
| proposed_vs_candidate_no_context | bertscore_f1 | 14 | 5 | 5 | 0.6875 | 0.7368 |
| proposed_vs_candidate_no_context | overall_quality | 10 | 9 | 5 | 0.5208 | 0.5263 |
| proposed_vs_baseline_no_context | context_relevance | 9 | 15 | 0 | 0.3750 | 0.3750 |
| proposed_vs_baseline_no_context | persona_consistency | 4 | 13 | 7 | 0.3125 | 0.2353 |
| proposed_vs_baseline_no_context | naturalness | 4 | 20 | 0 | 0.1667 | 0.1667 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 6 | 8 | 10 | 0.4583 | 0.4286 |
| proposed_vs_baseline_no_context | context_overlap | 12 | 12 | 0 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 3 | 7 | 14 | 0.4167 | 0.3000 |
| proposed_vs_baseline_no_context | persona_style | 2 | 6 | 16 | 0.4167 | 0.2500 |
| proposed_vs_baseline_no_context | distinct1 | 5 | 18 | 1 | 0.2292 | 0.2174 |
| proposed_vs_baseline_no_context | length_score | 4 | 20 | 0 | 0.1667 | 0.1667 |
| proposed_vs_baseline_no_context | sentence_score | 5 | 10 | 9 | 0.3958 | 0.3333 |
| proposed_vs_baseline_no_context | bertscore_f1 | 10 | 14 | 0 | 0.4167 | 0.4167 |
| proposed_vs_baseline_no_context | overall_quality | 7 | 17 | 0 | 0.2917 | 0.2917 |
| controlled_vs_proposed_raw | context_relevance | 23 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_proposed_raw | persona_consistency | 21 | 0 | 2 | 0.9565 | 1.0000 |
| controlled_vs_proposed_raw | naturalness | 16 | 7 | 0 | 0.6957 | 0.6957 |
| controlled_vs_proposed_raw | context_keyword_coverage | 23 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_proposed_raw | context_overlap | 22 | 1 | 0 | 0.9565 | 0.9565 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 21 | 0 | 2 | 0.9565 | 1.0000 |
| controlled_vs_proposed_raw | persona_style | 8 | 1 | 14 | 0.6522 | 0.8889 |
| controlled_vs_proposed_raw | distinct1 | 14 | 8 | 1 | 0.6304 | 0.6364 |
| controlled_vs_proposed_raw | length_score | 16 | 6 | 1 | 0.7174 | 0.7273 |
| controlled_vs_proposed_raw | sentence_score | 14 | 2 | 7 | 0.7609 | 0.8750 |
| controlled_vs_proposed_raw | bertscore_f1 | 17 | 6 | 0 | 0.7391 | 0.7391 |
| controlled_vs_proposed_raw | overall_quality | 23 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_relevance | 23 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 15 | 4 | 4 | 0.7391 | 0.7895 |
| controlled_vs_candidate_no_context | naturalness | 14 | 9 | 0 | 0.6087 | 0.6087 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 22 | 0 | 1 | 0.9783 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 23 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 14 | 4 | 5 | 0.7174 | 0.7778 |
| controlled_vs_candidate_no_context | persona_style | 6 | 1 | 16 | 0.6087 | 0.8571 |
| controlled_vs_candidate_no_context | distinct1 | 11 | 11 | 1 | 0.5000 | 0.5000 |
| controlled_vs_candidate_no_context | length_score | 12 | 10 | 1 | 0.5435 | 0.5455 |
| controlled_vs_candidate_no_context | sentence_score | 10 | 1 | 12 | 0.6957 | 0.9091 |
| controlled_vs_candidate_no_context | bertscore_f1 | 18 | 5 | 0 | 0.7826 | 0.7826 |
| controlled_vs_candidate_no_context | overall_quality | 22 | 1 | 0 | 0.9565 | 0.9565 |
| controlled_vs_baseline_no_context | context_relevance | 23 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 19 | 2 | 2 | 0.8696 | 0.9048 |
| controlled_vs_baseline_no_context | naturalness | 9 | 13 | 1 | 0.4130 | 0.4091 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 23 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | context_overlap | 22 | 0 | 1 | 0.9783 | 1.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 18 | 2 | 3 | 0.8478 | 0.9000 |
| controlled_vs_baseline_no_context | persona_style | 5 | 0 | 18 | 0.6087 | 1.0000 |
| controlled_vs_baseline_no_context | distinct1 | 1 | 20 | 2 | 0.0870 | 0.0476 |
| controlled_vs_baseline_no_context | length_score | 8 | 14 | 1 | 0.3696 | 0.3636 |
| controlled_vs_baseline_no_context | sentence_score | 8 | 1 | 14 | 0.6522 | 0.8889 |
| controlled_vs_baseline_no_context | bertscore_f1 | 15 | 8 | 0 | 0.6522 | 0.6522 |
| controlled_vs_baseline_no_context | overall_quality | 23 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 11 | 9 | 3 | 0.5435 | 0.5500 |
| controlled_alt_vs_controlled_default | persona_consistency | 6 | 11 | 6 | 0.3913 | 0.3529 |
| controlled_alt_vs_controlled_default | naturalness | 10 | 10 | 3 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 6 | 7 | 10 | 0.4783 | 0.4615 |
| controlled_alt_vs_controlled_default | context_overlap | 10 | 10 | 3 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 5 | 6 | 12 | 0.4783 | 0.4545 |
| controlled_alt_vs_controlled_default | persona_style | 3 | 5 | 15 | 0.4565 | 0.3750 |
| controlled_alt_vs_controlled_default | distinct1 | 10 | 10 | 3 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | length_score | 9 | 10 | 4 | 0.4783 | 0.4737 |
| controlled_alt_vs_controlled_default | sentence_score | 4 | 5 | 14 | 0.4783 | 0.4444 |
| controlled_alt_vs_controlled_default | bertscore_f1 | 11 | 11 | 1 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | overall_quality | 9 | 13 | 1 | 0.4130 | 0.4091 |
| controlled_alt_vs_proposed_raw | context_relevance | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_alt_vs_proposed_raw | persona_consistency | 21 | 1 | 2 | 0.9167 | 0.9545 |
| controlled_alt_vs_proposed_raw | naturalness | 19 | 5 | 0 | 0.7917 | 0.7917 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_alt_vs_proposed_raw | context_overlap | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 19 | 1 | 4 | 0.8750 | 0.9500 |
| controlled_alt_vs_proposed_raw | persona_style | 9 | 2 | 13 | 0.6458 | 0.8182 |
| controlled_alt_vs_proposed_raw | distinct1 | 13 | 9 | 2 | 0.5833 | 0.5909 |
| controlled_alt_vs_proposed_raw | length_score | 17 | 7 | 0 | 0.7083 | 0.7083 |
| controlled_alt_vs_proposed_raw | sentence_score | 15 | 3 | 6 | 0.7500 | 0.8333 |
| controlled_alt_vs_proposed_raw | bertscore_f1 | 16 | 8 | 0 | 0.6667 | 0.6667 |
| controlled_alt_vs_proposed_raw | overall_quality | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_alt_vs_candidate_no_context | context_relevance | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 18 | 4 | 2 | 0.7917 | 0.8182 |
| controlled_alt_vs_candidate_no_context | naturalness | 15 | 9 | 0 | 0.6250 | 0.6250 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 23 | 0 | 1 | 0.9792 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 17 | 4 | 3 | 0.7708 | 0.8095 |
| controlled_alt_vs_candidate_no_context | persona_style | 8 | 1 | 15 | 0.6458 | 0.8889 |
| controlled_alt_vs_candidate_no_context | distinct1 | 13 | 11 | 0 | 0.5417 | 0.5417 |
| controlled_alt_vs_candidate_no_context | length_score | 12 | 12 | 0 | 0.5000 | 0.5000 |
| controlled_alt_vs_candidate_no_context | sentence_score | 12 | 3 | 9 | 0.6875 | 0.8000 |
| controlled_alt_vs_candidate_no_context | bertscore_f1 | 20 | 4 | 0 | 0.8333 | 0.8333 |
| controlled_alt_vs_candidate_no_context | overall_quality | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_alt_vs_baseline_no_context | context_relevance | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 19 | 0 | 5 | 0.8958 | 1.0000 |
| controlled_alt_vs_baseline_no_context | naturalness | 9 | 15 | 0 | 0.3750 | 0.3750 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context | context_overlap | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 18 | 0 | 6 | 0.8750 | 1.0000 |
| controlled_alt_vs_baseline_no_context | persona_style | 6 | 1 | 17 | 0.6042 | 0.8571 |
| controlled_alt_vs_baseline_no_context | distinct1 | 4 | 19 | 1 | 0.1875 | 0.1739 |
| controlled_alt_vs_baseline_no_context | length_score | 9 | 15 | 0 | 0.3750 | 0.3750 |
| controlled_alt_vs_baseline_no_context | sentence_score | 9 | 2 | 13 | 0.6458 | 0.8182 |
| controlled_alt_vs_baseline_no_context | bertscore_f1 | 16 | 8 | 0 | 0.6667 | 0.6667 |
| controlled_alt_vs_baseline_no_context | overall_quality | 24 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 23 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 19 | 2 | 2 | 0.8696 | 0.9048 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 9 | 13 | 1 | 0.4130 | 0.4091 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 23 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 22 | 0 | 1 | 0.9783 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 18 | 2 | 3 | 0.8478 | 0.9000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 5 | 0 | 18 | 0.6087 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | 1 | 20 | 2 | 0.0870 | 0.0476 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 8 | 14 | 1 | 0.3696 | 0.3636 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 8 | 1 | 14 | 0.6522 | 0.8889 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 15 | 8 | 0 | 0.6522 | 0.6522 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 23 | 0 | 0 | 1.0000 | 1.0000 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0417 | 0.0417 | 0.2917 | 0.4583 | 0.5417 |
| proposed_contextual_controlled_alt | 0.0000 | 0.0000 | 0.1667 | 0.4583 | 0.5417 |
| proposed_contextual | 0.0000 | 0.0000 | 0.6250 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.4583 | 0.0000 | 0.0000 |
| baseline_no_context | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `8`
- Unique template signatures: `22`
- Template signature ratio: `0.9167`
- Effective sample size by source clustering: `7.20`
- Effective sample size by template-signature clustering: `20.57`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: enabled.

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
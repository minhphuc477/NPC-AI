# Proposal Alignment Evaluation Report

- Run ID: `20260304T110043Z`
- Generated: `2026-03-04T11:09:07.648056+00:00`
- Scenarios: `artifacts\proposal_control_tuning\upgraded\20260304T110043Z\scenarios.jsonl`
- Scenario count: `40`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off
- `baseline_no_context`: model `phi3:mini`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2643 (0.2432, 0.2881) | 0.3532 (0.3062, 0.4035) | 0.9049 (0.8853, 0.9227) | 0.3877 (0.3703, 0.4057) | 0.0728 |
| proposed_contextual | 0.0691 (0.0404, 0.1043) | 0.1481 (0.1100, 0.1900) | 0.7783 (0.7637, 0.7951) | 0.2156 (0.1914, 0.2416) | 0.0482 |
| candidate_no_context | 0.0223 (0.0131, 0.0339) | 0.1560 (0.1157, 0.2021) | 0.7974 (0.7769, 0.8199) | 0.2010 (0.1839, 0.2210) | 0.0312 |
| baseline_no_context | 0.0375 (0.0243, 0.0531) | 0.2184 (0.1785, 0.2640) | 0.8976 (0.8800, 0.9162) | 0.2479 (0.2315, 0.2647) | 0.0551 |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0468 | 2.0991 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0079 | -0.0505 |
| proposed_vs_candidate_no_context | naturalness | -0.0191 | -0.0239 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0581 | 3.6009 |
| proposed_vs_candidate_no_context | context_overlap | 0.0204 | 0.5572 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0121 | -0.1680 |
| proposed_vs_candidate_no_context | persona_style | 0.0092 | 0.0188 |
| proposed_vs_candidate_no_context | distinct1 | -0.0116 | -0.0125 |
| proposed_vs_candidate_no_context | length_score | -0.0467 | -0.2029 |
| proposed_vs_candidate_no_context | sentence_score | -0.0437 | -0.0560 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0170 | 0.5460 |
| proposed_vs_candidate_no_context | overall_quality | 0.0146 | 0.0725 |
| proposed_vs_baseline_no_context | context_relevance | 0.0316 | 0.8446 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0703 | -0.3218 |
| proposed_vs_baseline_no_context | naturalness | -0.1192 | -0.1328 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0372 | 1.0041 |
| proposed_vs_baseline_no_context | context_overlap | 0.0187 | 0.4860 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0713 | -0.5426 |
| proposed_vs_baseline_no_context | persona_style | -0.0661 | -0.1168 |
| proposed_vs_baseline_no_context | distinct1 | -0.0537 | -0.0551 |
| proposed_vs_baseline_no_context | length_score | -0.4108 | -0.6914 |
| proposed_vs_baseline_no_context | sentence_score | -0.1488 | -0.1678 |
| proposed_vs_baseline_no_context | bertscore_f1 | -0.0069 | -0.1250 |
| proposed_vs_baseline_no_context | overall_quality | -0.0324 | -0.1305 |
| controlled_vs_proposed_raw | context_relevance | 0.1952 | 2.8241 |
| controlled_vs_proposed_raw | persona_consistency | 0.2051 | 1.3847 |
| controlled_vs_proposed_raw | naturalness | 0.1265 | 0.1625 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2595 | 3.4949 |
| controlled_vs_proposed_raw | context_overlap | 0.0451 | 0.7894 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2356 | 3.9188 |
| controlled_vs_proposed_raw | persona_style | 0.0832 | 0.1663 |
| controlled_vs_proposed_raw | distinct1 | 0.0153 | 0.0166 |
| controlled_vs_proposed_raw | length_score | 0.4758 | 2.5955 |
| controlled_vs_proposed_raw | sentence_score | 0.2450 | 0.3322 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0246 | 0.5104 |
| controlled_vs_proposed_raw | overall_quality | 0.1722 | 0.7988 |
| controlled_vs_candidate_no_context | context_relevance | 0.2420 | 10.8513 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1972 | 1.2644 |
| controlled_vs_candidate_no_context | naturalness | 0.1074 | 0.1347 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3176 | 19.6808 |
| controlled_vs_candidate_no_context | context_overlap | 0.0655 | 1.7865 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2235 | 3.0923 |
| controlled_vs_candidate_no_context | persona_style | 0.0924 | 0.1882 |
| controlled_vs_candidate_no_context | distinct1 | 0.0037 | 0.0040 |
| controlled_vs_candidate_no_context | length_score | 0.4292 | 1.8659 |
| controlled_vs_candidate_no_context | sentence_score | 0.2012 | 0.2576 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0416 | 1.3350 |
| controlled_vs_candidate_no_context | overall_quality | 0.1868 | 0.9292 |
| controlled_vs_baseline_no_context | context_relevance | 0.2268 | 6.0542 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1348 | 0.6174 |
| controlled_vs_baseline_no_context | naturalness | 0.0073 | 0.0081 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.2967 | 8.0082 |
| controlled_vs_baseline_no_context | context_overlap | 0.0638 | 1.6591 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1643 | 1.2500 |
| controlled_vs_baseline_no_context | persona_style | 0.0171 | 0.0301 |
| controlled_vs_baseline_no_context | distinct1 | -0.0384 | -0.0394 |
| controlled_vs_baseline_no_context | length_score | 0.0650 | 0.1094 |
| controlled_vs_baseline_no_context | sentence_score | 0.0962 | 0.1086 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0177 | 0.3215 |
| controlled_vs_baseline_no_context | overall_quality | 0.1398 | 0.5640 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2268 | 6.0542 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1348 | 0.6174 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 0.0073 | 0.0081 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.2967 | 8.0082 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0638 | 1.6591 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1643 | 1.2500 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0171 | 0.0301 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0384 | -0.0394 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.0650 | 0.1094 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0962 | 0.1086 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0177 | 0.3215 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1398 | 0.5640 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0468 | (0.0177, 0.0811) | 0.0007 | 0.0468 | (0.0059, 0.0953) | 0.0123 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0079 | (-0.0517, 0.0330) | 0.6357 | -0.0079 | (-0.0691, 0.0424) | 0.5860 |
| proposed_vs_candidate_no_context | naturalness | -0.0191 | (-0.0391, -0.0007) | 0.9797 | -0.0191 | (-0.0330, -0.0072) | 0.9990 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0581 | (0.0200, 0.1009) | 0.0000 | 0.0581 | (0.0071, 0.1220) | 0.0123 |
| proposed_vs_candidate_no_context | context_overlap | 0.0204 | (0.0058, 0.0395) | 0.0010 | 0.0204 | (0.0007, 0.0489) | 0.0200 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0121 | (-0.0637, 0.0364) | 0.6763 | -0.0121 | (-0.0774, 0.0409) | 0.6440 |
| proposed_vs_candidate_no_context | persona_style | 0.0092 | (-0.0369, 0.0564) | 0.3490 | 0.0092 | (-0.0483, 0.0605) | 0.3497 |
| proposed_vs_candidate_no_context | distinct1 | -0.0116 | (-0.0244, 0.0001) | 0.9743 | -0.0116 | (-0.0247, -0.0009) | 0.9820 |
| proposed_vs_candidate_no_context | length_score | -0.0467 | (-0.1167, 0.0200) | 0.9173 | -0.0467 | (-0.0903, 0.0010) | 0.9730 |
| proposed_vs_candidate_no_context | sentence_score | -0.0437 | (-0.0962, 0.0087) | 0.9577 | -0.0437 | (-0.1094, 0.0075) | 0.9717 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0170 | (-0.0002, 0.0354) | 0.0260 | 0.0170 | (0.0032, 0.0423) | 0.0013 |
| proposed_vs_candidate_no_context | overall_quality | 0.0146 | (-0.0086, 0.0382) | 0.1177 | 0.0146 | (-0.0201, 0.0506) | 0.2157 |
| proposed_vs_baseline_no_context | context_relevance | 0.0316 | (-0.0042, 0.0669) | 0.0457 | 0.0316 | (-0.0122, 0.0807) | 0.0860 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0703 | (-0.1279, -0.0138) | 0.9917 | -0.0703 | (-0.1554, 0.0050) | 0.9640 |
| proposed_vs_baseline_no_context | naturalness | -0.1192 | (-0.1428, -0.0926) | 1.0000 | -0.1192 | (-0.1521, -0.0732) | 1.0000 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0372 | (-0.0073, 0.0864) | 0.0560 | 0.0372 | (-0.0151, 0.0975) | 0.0867 |
| proposed_vs_baseline_no_context | context_overlap | 0.0187 | (0.0031, 0.0379) | 0.0080 | 0.0187 | (-0.0022, 0.0504) | 0.0517 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0713 | (-0.1433, -0.0014) | 0.9767 | -0.0713 | (-0.1785, 0.0276) | 0.9197 |
| proposed_vs_baseline_no_context | persona_style | -0.0661 | (-0.1366, -0.0048) | 0.9840 | -0.0661 | (-0.1729, 0.0249) | 0.8943 |
| proposed_vs_baseline_no_context | distinct1 | -0.0537 | (-0.0680, -0.0380) | 1.0000 | -0.0537 | (-0.0676, -0.0375) | 1.0000 |
| proposed_vs_baseline_no_context | length_score | -0.4108 | (-0.5075, -0.3000) | 1.0000 | -0.4108 | (-0.5415, -0.2313) | 1.0000 |
| proposed_vs_baseline_no_context | sentence_score | -0.1487 | (-0.2100, -0.0875) | 1.0000 | -0.1487 | (-0.2265, -0.0467) | 0.9967 |
| proposed_vs_baseline_no_context | bertscore_f1 | -0.0069 | (-0.0301, 0.0158) | 0.7290 | -0.0069 | (-0.0410, 0.0321) | 0.6373 |
| proposed_vs_baseline_no_context | overall_quality | -0.0324 | (-0.0637, -0.0014) | 0.9793 | -0.0324 | (-0.0762, 0.0167) | 0.9140 |
| controlled_vs_proposed_raw | context_relevance | 0.1952 | (0.1560, 0.2345) | 0.0000 | 0.1952 | (0.1288, 0.2482) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.2051 | (0.1434, 0.2723) | 0.0000 | 0.2051 | (0.1309, 0.2771) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.1265 | (0.1024, 0.1481) | 0.0000 | 0.1265 | (0.1003, 0.1462) | 0.0000 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2595 | (0.2076, 0.3094) | 0.0000 | 0.2595 | (0.1707, 0.3247) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0451 | (0.0265, 0.0623) | 0.0000 | 0.0451 | (0.0155, 0.0711) | 0.0027 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2356 | (0.1600, 0.3141) | 0.0000 | 0.2356 | (0.1417, 0.3304) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.0832 | (0.0263, 0.1529) | 0.0003 | 0.0832 | (0.0005, 0.1905) | 0.0233 |
| controlled_vs_proposed_raw | distinct1 | 0.0153 | (0.0023, 0.0276) | 0.0100 | 0.0153 | (0.0048, 0.0271) | 0.0010 |
| controlled_vs_proposed_raw | length_score | 0.4758 | (0.3775, 0.5750) | 0.0000 | 0.4758 | (0.3902, 0.5462) | 0.0000 |
| controlled_vs_proposed_raw | sentence_score | 0.2450 | (0.1925, 0.2975) | 0.0000 | 0.2450 | (0.1800, 0.3073) | 0.0000 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0246 | (-0.0034, 0.0500) | 0.0413 | 0.0246 | (-0.0301, 0.0628) | 0.1450 |
| controlled_vs_proposed_raw | overall_quality | 0.1722 | (0.1416, 0.2018) | 0.0000 | 0.1722 | (0.1276, 0.2146) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2420 | (0.2186, 0.2695) | 0.0000 | 0.2420 | (0.2121, 0.2726) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1972 | (0.1350, 0.2626) | 0.0000 | 0.1972 | (0.1264, 0.2587) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.1074 | (0.0787, 0.1355) | 0.0000 | 0.1074 | (0.0863, 0.1237) | 0.0000 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3176 | (0.2850, 0.3521) | 0.0000 | 0.3176 | (0.2799, 0.3590) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0655 | (0.0558, 0.0751) | 0.0000 | 0.0655 | (0.0539, 0.0800) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2235 | (0.1494, 0.2988) | 0.0000 | 0.2235 | (0.1529, 0.2911) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.0924 | (0.0167, 0.1740) | 0.0090 | 0.0924 | (-0.0271, 0.2291) | 0.0793 |
| controlled_vs_candidate_no_context | distinct1 | 0.0037 | (-0.0094, 0.0166) | 0.2723 | 0.0037 | (-0.0092, 0.0114) | 0.2287 |
| controlled_vs_candidate_no_context | length_score | 0.4292 | (0.3158, 0.5358) | 0.0000 | 0.4292 | (0.3667, 0.4793) | 0.0000 |
| controlled_vs_candidate_no_context | sentence_score | 0.2012 | (0.1487, 0.2537) | 0.0000 | 0.2012 | (0.1400, 0.2561) | 0.0000 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0416 | (0.0229, 0.0600) | 0.0000 | 0.0416 | (0.0110, 0.0701) | 0.0060 |
| controlled_vs_candidate_no_context | overall_quality | 0.1868 | (0.1649, 0.2095) | 0.0000 | 0.1868 | (0.1622, 0.2083) | 0.0000 |
| controlled_vs_baseline_no_context | context_relevance | 0.2268 | (0.1982, 0.2558) | 0.0000 | 0.2268 | (0.1904, 0.2678) | 0.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1348 | (0.0683, 0.2071) | 0.0003 | 0.1348 | (0.0498, 0.2053) | 0.0013 |
| controlled_vs_baseline_no_context | naturalness | 0.0073 | (-0.0205, 0.0348) | 0.2960 | 0.0073 | (-0.0245, 0.0470) | 0.3500 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.2967 | (0.2580, 0.3377) | 0.0000 | 0.2967 | (0.2463, 0.3536) | 0.0000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0638 | (0.0552, 0.0719) | 0.0000 | 0.0638 | (0.0538, 0.0772) | 0.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1643 | (0.0833, 0.2470) | 0.0000 | 0.1643 | (0.0596, 0.2548) | 0.0017 |
| controlled_vs_baseline_no_context | persona_style | 0.0171 | (-0.0246, 0.0582) | 0.2057 | 0.0171 | (0.0008, 0.0524) | 0.0173 |
| controlled_vs_baseline_no_context | distinct1 | -0.0384 | (-0.0489, -0.0279) | 1.0000 | -0.0384 | (-0.0462, -0.0303) | 1.0000 |
| controlled_vs_baseline_no_context | length_score | 0.0650 | (-0.0600, 0.1875) | 0.1607 | 0.0650 | (-0.0773, 0.2453) | 0.2160 |
| controlled_vs_baseline_no_context | sentence_score | 0.0962 | (0.0350, 0.1575) | 0.0007 | 0.0962 | (0.0429, 0.1694) | 0.0000 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0177 | (-0.0022, 0.0369) | 0.0420 | 0.0177 | (-0.0079, 0.0370) | 0.0677 |
| controlled_vs_baseline_no_context | overall_quality | 0.1398 | (0.1136, 0.1649) | 0.0000 | 0.1398 | (0.1080, 0.1716) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2268 | (0.1978, 0.2569) | 0.0000 | 0.2268 | (0.1919, 0.2670) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1348 | (0.0680, 0.2039) | 0.0000 | 0.1348 | (0.0483, 0.2096) | 0.0023 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 0.0073 | (-0.0214, 0.0346) | 0.3063 | 0.0073 | (-0.0232, 0.0459) | 0.3427 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.2967 | (0.2575, 0.3400) | 0.0000 | 0.2967 | (0.2465, 0.3529) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0638 | (0.0556, 0.0720) | 0.0000 | 0.0638 | (0.0537, 0.0776) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1643 | (0.0844, 0.2487) | 0.0000 | 0.1643 | (0.0580, 0.2532) | 0.0027 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0171 | (-0.0250, 0.0572) | 0.1960 | 0.0171 | (0.0008, 0.0529) | 0.0183 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0384 | (-0.0488, -0.0279) | 1.0000 | -0.0384 | (-0.0465, -0.0300) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.0650 | (-0.0625, 0.1875) | 0.1653 | 0.0650 | (-0.0707, 0.2525) | 0.2040 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0962 | (0.0350, 0.1575) | 0.0003 | 0.0962 | (0.0407, 0.1803) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0177 | (-0.0017, 0.0363) | 0.0363 | 0.0177 | (-0.0065, 0.0382) | 0.0667 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1398 | (0.1141, 0.1655) | 0.0000 | 0.1398 | (0.1079, 0.1716) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 14 | 6 | 20 | 0.6000 | 0.7000 |
| proposed_vs_candidate_no_context | persona_consistency | 6 | 7 | 27 | 0.4875 | 0.4615 |
| proposed_vs_candidate_no_context | naturalness | 9 | 11 | 20 | 0.4750 | 0.4500 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 11 | 2 | 27 | 0.6125 | 0.8462 |
| proposed_vs_candidate_no_context | context_overlap | 15 | 5 | 20 | 0.6250 | 0.7500 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 5 | 6 | 29 | 0.4875 | 0.4545 |
| proposed_vs_candidate_no_context | persona_style | 4 | 2 | 34 | 0.5250 | 0.6667 |
| proposed_vs_candidate_no_context | distinct1 | 6 | 12 | 22 | 0.4250 | 0.3333 |
| proposed_vs_candidate_no_context | length_score | 6 | 12 | 22 | 0.4250 | 0.3333 |
| proposed_vs_candidate_no_context | sentence_score | 3 | 8 | 29 | 0.4375 | 0.2727 |
| proposed_vs_candidate_no_context | bertscore_f1 | 15 | 5 | 20 | 0.6250 | 0.7500 |
| proposed_vs_candidate_no_context | overall_quality | 11 | 9 | 20 | 0.5250 | 0.5500 |
| proposed_vs_baseline_no_context | context_relevance | 19 | 21 | 0 | 0.4750 | 0.4750 |
| proposed_vs_baseline_no_context | persona_consistency | 7 | 22 | 11 | 0.3125 | 0.2414 |
| proposed_vs_baseline_no_context | naturalness | 5 | 35 | 0 | 0.1250 | 0.1250 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 11 | 8 | 21 | 0.5375 | 0.5789 |
| proposed_vs_baseline_no_context | context_overlap | 22 | 18 | 0 | 0.5500 | 0.5500 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 6 | 16 | 18 | 0.3750 | 0.2727 |
| proposed_vs_baseline_no_context | persona_style | 2 | 8 | 30 | 0.4250 | 0.2000 |
| proposed_vs_baseline_no_context | distinct1 | 5 | 35 | 0 | 0.1250 | 0.1250 |
| proposed_vs_baseline_no_context | length_score | 3 | 37 | 0 | 0.0750 | 0.0750 |
| proposed_vs_baseline_no_context | sentence_score | 2 | 19 | 19 | 0.2875 | 0.0952 |
| proposed_vs_baseline_no_context | bertscore_f1 | 18 | 22 | 0 | 0.4500 | 0.4500 |
| proposed_vs_baseline_no_context | overall_quality | 11 | 29 | 0 | 0.2750 | 0.2750 |
| controlled_vs_proposed_raw | context_relevance | 37 | 3 | 0 | 0.9250 | 0.9250 |
| controlled_vs_proposed_raw | persona_consistency | 33 | 4 | 3 | 0.8625 | 0.8919 |
| controlled_vs_proposed_raw | naturalness | 37 | 3 | 0 | 0.9250 | 0.9250 |
| controlled_vs_proposed_raw | context_keyword_coverage | 36 | 2 | 2 | 0.9250 | 0.9474 |
| controlled_vs_proposed_raw | context_overlap | 35 | 5 | 0 | 0.8750 | 0.8750 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 33 | 4 | 3 | 0.8625 | 0.8919 |
| controlled_vs_proposed_raw | persona_style | 7 | 0 | 33 | 0.5875 | 1.0000 |
| controlled_vs_proposed_raw | distinct1 | 26 | 13 | 1 | 0.6625 | 0.6667 |
| controlled_vs_proposed_raw | length_score | 34 | 5 | 1 | 0.8625 | 0.8718 |
| controlled_vs_proposed_raw | sentence_score | 29 | 1 | 10 | 0.8500 | 0.9667 |
| controlled_vs_proposed_raw | bertscore_f1 | 27 | 13 | 0 | 0.6750 | 0.6750 |
| controlled_vs_proposed_raw | overall_quality | 37 | 3 | 0 | 0.9250 | 0.9250 |
| controlled_vs_candidate_no_context | context_relevance | 40 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 35 | 3 | 2 | 0.9000 | 0.9211 |
| controlled_vs_candidate_no_context | naturalness | 36 | 4 | 0 | 0.9000 | 0.9000 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 40 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 39 | 1 | 0 | 0.9750 | 0.9750 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 35 | 3 | 2 | 0.9000 | 0.9211 |
| controlled_vs_candidate_no_context | persona_style | 10 | 2 | 28 | 0.6000 | 0.8333 |
| controlled_vs_candidate_no_context | distinct1 | 22 | 18 | 0 | 0.5500 | 0.5500 |
| controlled_vs_candidate_no_context | length_score | 35 | 5 | 0 | 0.8750 | 0.8750 |
| controlled_vs_candidate_no_context | sentence_score | 23 | 0 | 17 | 0.7875 | 1.0000 |
| controlled_vs_candidate_no_context | bertscore_f1 | 31 | 9 | 0 | 0.7750 | 0.7750 |
| controlled_vs_candidate_no_context | overall_quality | 39 | 1 | 0 | 0.9750 | 0.9750 |
| controlled_vs_baseline_no_context | context_relevance | 39 | 1 | 0 | 0.9750 | 0.9750 |
| controlled_vs_baseline_no_context | persona_consistency | 28 | 8 | 4 | 0.7500 | 0.7778 |
| controlled_vs_baseline_no_context | naturalness | 24 | 16 | 0 | 0.6000 | 0.6000 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 39 | 1 | 0 | 0.9750 | 0.9750 |
| controlled_vs_baseline_no_context | context_overlap | 40 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 28 | 7 | 5 | 0.7625 | 0.8000 |
| controlled_vs_baseline_no_context | persona_style | 4 | 1 | 35 | 0.5375 | 0.8000 |
| controlled_vs_baseline_no_context | distinct1 | 6 | 32 | 2 | 0.1750 | 0.1579 |
| controlled_vs_baseline_no_context | length_score | 23 | 14 | 3 | 0.6125 | 0.6216 |
| controlled_vs_baseline_no_context | sentence_score | 13 | 2 | 25 | 0.6375 | 0.8667 |
| controlled_vs_baseline_no_context | bertscore_f1 | 25 | 15 | 0 | 0.6250 | 0.6250 |
| controlled_vs_baseline_no_context | overall_quality | 39 | 1 | 0 | 0.9750 | 0.9750 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 39 | 1 | 0 | 0.9750 | 0.9750 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 28 | 8 | 4 | 0.7500 | 0.7778 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 24 | 16 | 0 | 0.6000 | 0.6000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 39 | 1 | 0 | 0.9750 | 0.9750 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 40 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 28 | 7 | 5 | 0.7625 | 0.8000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 4 | 1 | 35 | 0.5375 | 0.8000 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | 6 | 32 | 2 | 0.1750 | 0.1579 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 23 | 14 | 3 | 0.6125 | 0.6216 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 13 | 2 | 25 | 0.6375 | 0.8667 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 25 | 15 | 0 | 0.6250 | 0.6250 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 39 | 1 | 0 | 0.9750 | 0.9750 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.7000 | 0.1000 | 0.9000 |
| proposed_contextual | 0.0000 | 0.0000 | 0.6000 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.6000 | 0.0000 | 0.0000 |
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
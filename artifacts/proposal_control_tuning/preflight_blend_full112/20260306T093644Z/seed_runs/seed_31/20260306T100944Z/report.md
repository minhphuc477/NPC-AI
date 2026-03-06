# Proposal Alignment Evaluation Report

- Run ID: `20260306T100944Z`
- Generated: `2026-03-06T10:43:43.733301+00:00`
- Scenarios: `artifacts\proposal_control_tuning\preflight_blend_full112\20260306T093644Z\seed_runs\seed_31\20260306T100944Z\scenarios.jsonl`
- Scenario count: `112`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_alt`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off
- `baseline_no_context`: model `phi3:mini`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2777 (0.2580, 0.2989) | 0.3645 (0.3342, 0.3949) | 0.8700 (0.8561, 0.8836) | 0.3921 (0.3799, 0.4047) | 0.0865 |
| proposed_contextual_controlled_alt | 0.2502 (0.2291, 0.2708) | 0.3328 (0.3045, 0.3614) | 0.8640 (0.8481, 0.8783) | 0.3674 (0.3559, 0.3799) | 0.0726 |
| proposed_contextual | 0.1083 (0.0870, 0.1335) | 0.1897 (0.1622, 0.2202) | 0.8120 (0.7986, 0.8255) | 0.2541 (0.2382, 0.2720) | 0.0733 |
| candidate_no_context | 0.0313 (0.0240, 0.0394) | 0.1637 (0.1378, 0.1924) | 0.8096 (0.7932, 0.8260) | 0.2098 (0.1992, 0.2208) | 0.0375 |
| baseline_no_context | 0.0418 (0.0340, 0.0497) | 0.1792 (0.1584, 0.2012) | 0.8802 (0.8694, 0.8907) | 0.2342 (0.2257, 0.2430) | 0.0620 |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0770 | 2.4634 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0260 | 0.1586 |
| proposed_vs_candidate_no_context | naturalness | 0.0024 | 0.0030 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.1017 | 3.7463 |
| proposed_vs_candidate_no_context | context_overlap | 0.0196 | 0.4789 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0273 | 0.3617 |
| proposed_vs_candidate_no_context | persona_style | 0.0206 | 0.0399 |
| proposed_vs_candidate_no_context | distinct1 | 0.0078 | 0.0083 |
| proposed_vs_candidate_no_context | length_score | -0.0125 | -0.0422 |
| proposed_vs_candidate_no_context | sentence_score | 0.0094 | 0.0120 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0358 | 0.9535 |
| proposed_vs_candidate_no_context | overall_quality | 0.0444 | 0.2114 |
| proposed_vs_baseline_no_context | context_relevance | 0.0665 | 1.5884 |
| proposed_vs_baseline_no_context | persona_consistency | 0.0105 | 0.0585 |
| proposed_vs_baseline_no_context | naturalness | -0.0682 | -0.0775 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0863 | 2.0322 |
| proposed_vs_baseline_no_context | context_overlap | 0.0202 | 0.4992 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 0.0222 | 0.2748 |
| proposed_vs_baseline_no_context | persona_style | -0.0362 | -0.0631 |
| proposed_vs_baseline_no_context | distinct1 | -0.0393 | -0.0401 |
| proposed_vs_baseline_no_context | length_score | -0.2217 | -0.4385 |
| proposed_vs_baseline_no_context | sentence_score | -0.0813 | -0.0932 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0113 | 0.1825 |
| proposed_vs_baseline_no_context | overall_quality | 0.0199 | 0.0848 |
| controlled_vs_proposed_raw | context_relevance | 0.1694 | 1.5639 |
| controlled_vs_proposed_raw | persona_consistency | 0.1749 | 0.9218 |
| controlled_vs_proposed_raw | naturalness | 0.0580 | 0.0715 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2211 | 1.7169 |
| controlled_vs_proposed_raw | context_overlap | 0.0487 | 0.8044 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2040 | 1.9855 |
| controlled_vs_proposed_raw | persona_style | 0.0581 | 0.1081 |
| controlled_vs_proposed_raw | distinct1 | 0.0016 | 0.0017 |
| controlled_vs_proposed_raw | length_score | 0.2182 | 0.7683 |
| controlled_vs_proposed_raw | sentence_score | 0.1375 | 0.1739 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0132 | 0.1804 |
| controlled_vs_proposed_raw | overall_quality | 0.1380 | 0.5429 |
| controlled_vs_candidate_no_context | context_relevance | 0.2465 | 7.8799 |
| controlled_vs_candidate_no_context | persona_consistency | 0.2008 | 1.2265 |
| controlled_vs_candidate_no_context | naturalness | 0.0605 | 0.0747 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3228 | 11.8953 |
| controlled_vs_candidate_no_context | context_overlap | 0.0683 | 1.6685 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2313 | 3.0654 |
| controlled_vs_candidate_no_context | persona_style | 0.0787 | 0.1524 |
| controlled_vs_candidate_no_context | distinct1 | 0.0094 | 0.0101 |
| controlled_vs_candidate_no_context | length_score | 0.2057 | 0.6938 |
| controlled_vs_candidate_no_context | sentence_score | 0.1469 | 0.1880 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0490 | 1.3060 |
| controlled_vs_candidate_no_context | overall_quality | 0.1823 | 0.8691 |
| controlled_vs_baseline_no_context | context_relevance | 0.2359 | 5.6365 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1853 | 1.0341 |
| controlled_vs_baseline_no_context | naturalness | -0.0102 | -0.0115 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3075 | 7.2382 |
| controlled_vs_baseline_no_context | context_overlap | 0.0689 | 1.7051 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2262 | 2.8059 |
| controlled_vs_baseline_no_context | persona_style | 0.0219 | 0.0381 |
| controlled_vs_baseline_no_context | distinct1 | -0.0377 | -0.0384 |
| controlled_vs_baseline_no_context | length_score | -0.0036 | -0.0071 |
| controlled_vs_baseline_no_context | sentence_score | 0.0562 | 0.0645 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0245 | 0.3959 |
| controlled_vs_baseline_no_context | overall_quality | 0.1578 | 0.6738 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0276 | -0.0992 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0317 | -0.0870 |
| controlled_alt_vs_controlled_default | naturalness | -0.0060 | -0.0069 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0362 | -0.1033 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0075 | -0.0686 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0431 | -0.1404 |
| controlled_alt_vs_controlled_default | persona_style | 0.0136 | 0.0229 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0080 | -0.0085 |
| controlled_alt_vs_controlled_default | length_score | -0.0009 | -0.0018 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0210 | -0.0226 |
| controlled_alt_vs_controlled_default | bertscore_f1 | -0.0139 | -0.1601 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0246 | -0.0628 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1418 | 1.3095 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1431 | 0.7545 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0520 | 0.0640 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.1850 | 1.4362 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0412 | 0.6806 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1610 | 1.5664 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0717 | 0.1335 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0064 | -0.0068 |
| controlled_alt_vs_proposed_raw | length_score | 0.2173 | 0.7652 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1165 | 0.1474 |
| controlled_alt_vs_proposed_raw | bertscore_f1 | -0.0006 | -0.0086 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1133 | 0.4460 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2189 | 6.9987 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1691 | 1.0327 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0544 | 0.0672 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.2866 | 10.5628 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0608 | 1.4854 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1883 | 2.4946 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0924 | 0.1787 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0014 | 0.0015 |
| controlled_alt_vs_candidate_no_context | length_score | 0.2048 | 0.6908 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1259 | 0.1611 |
| controlled_alt_vs_candidate_no_context | bertscore_f1 | 0.0351 | 0.9367 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1577 | 0.7517 |
| controlled_alt_vs_baseline_no_context | context_relevance | 0.2083 | 4.9780 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 0.1536 | 0.8571 |
| controlled_alt_vs_baseline_no_context | naturalness | -0.0162 | -0.0184 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 0.2713 | 6.3869 |
| controlled_alt_vs_baseline_no_context | context_overlap | 0.0614 | 1.5196 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 0.1831 | 2.2716 |
| controlled_alt_vs_baseline_no_context | persona_style | 0.0355 | 0.0619 |
| controlled_alt_vs_baseline_no_context | distinct1 | -0.0457 | -0.0466 |
| controlled_alt_vs_baseline_no_context | length_score | -0.0045 | -0.0088 |
| controlled_alt_vs_baseline_no_context | sentence_score | 0.0353 | 0.0405 |
| controlled_alt_vs_baseline_no_context | bertscore_f1 | 0.0107 | 0.1723 |
| controlled_alt_vs_baseline_no_context | overall_quality | 0.1332 | 0.5686 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2359 | 5.6365 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1853 | 1.0341 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0102 | -0.0115 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3075 | 7.2382 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0689 | 1.7051 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2262 | 2.8059 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0219 | 0.0381 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0377 | -0.0384 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | -0.0036 | -0.0071 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0562 | 0.0645 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0245 | 0.3959 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1578 | 0.6738 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0770 | (0.0541, 0.1011) | 0.0000 | 0.0770 | (0.0462, 0.1050) | 0.0000 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0260 | (-0.0016, 0.0542) | 0.0313 | 0.0260 | (0.0045, 0.0518) | 0.0103 |
| proposed_vs_candidate_no_context | naturalness | 0.0024 | (-0.0142, 0.0194) | 0.3903 | 0.0024 | (-0.0150, 0.0196) | 0.4007 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.1017 | (0.0732, 0.1334) | 0.0000 | 0.1017 | (0.0637, 0.1376) | 0.0000 |
| proposed_vs_candidate_no_context | context_overlap | 0.0196 | (0.0133, 0.0259) | 0.0000 | 0.0196 | (0.0111, 0.0279) | 0.0000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0273 | (-0.0048, 0.0617) | 0.0433 | 0.0273 | (0.0005, 0.0599) | 0.0230 |
| proposed_vs_candidate_no_context | persona_style | 0.0206 | (-0.0125, 0.0520) | 0.1057 | 0.0206 | (0.0058, 0.0415) | 0.0010 |
| proposed_vs_candidate_no_context | distinct1 | 0.0078 | (-0.0020, 0.0191) | 0.0627 | 0.0078 | (-0.0023, 0.0184) | 0.0680 |
| proposed_vs_candidate_no_context | length_score | -0.0125 | (-0.0804, 0.0539) | 0.6373 | -0.0125 | (-0.0857, 0.0619) | 0.6003 |
| proposed_vs_candidate_no_context | sentence_score | 0.0094 | (-0.0250, 0.0469) | 0.3327 | 0.0094 | (-0.0344, 0.0375) | 0.3040 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0358 | (0.0253, 0.0467) | 0.0000 | 0.0358 | (0.0206, 0.0514) | 0.0000 |
| proposed_vs_candidate_no_context | overall_quality | 0.0444 | (0.0286, 0.0610) | 0.0000 | 0.0444 | (0.0285, 0.0615) | 0.0000 |
| proposed_vs_baseline_no_context | context_relevance | 0.0665 | (0.0440, 0.0905) | 0.0000 | 0.0665 | (0.0344, 0.0998) | 0.0000 |
| proposed_vs_baseline_no_context | persona_consistency | 0.0105 | (-0.0198, 0.0418) | 0.2543 | 0.0105 | (-0.0368, 0.0591) | 0.3167 |
| proposed_vs_baseline_no_context | naturalness | -0.0682 | (-0.0846, -0.0516) | 1.0000 | -0.0682 | (-0.0857, -0.0511) | 1.0000 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0863 | (0.0561, 0.1184) | 0.0000 | 0.0863 | (0.0438, 0.1307) | 0.0000 |
| proposed_vs_baseline_no_context | context_overlap | 0.0202 | (0.0125, 0.0279) | 0.0000 | 0.0202 | (0.0111, 0.0292) | 0.0000 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 0.0222 | (-0.0148, 0.0600) | 0.1150 | 0.0222 | (-0.0176, 0.0634) | 0.1557 |
| proposed_vs_baseline_no_context | persona_style | -0.0362 | (-0.0734, -0.0027) | 0.9810 | -0.0362 | (-0.1372, 0.0422) | 0.7750 |
| proposed_vs_baseline_no_context | distinct1 | -0.0393 | (-0.0476, -0.0310) | 1.0000 | -0.0393 | (-0.0498, -0.0285) | 1.0000 |
| proposed_vs_baseline_no_context | length_score | -0.2217 | (-0.2845, -0.1574) | 1.0000 | -0.2217 | (-0.2884, -0.1524) | 1.0000 |
| proposed_vs_baseline_no_context | sentence_score | -0.0813 | (-0.1219, -0.0375) | 1.0000 | -0.0813 | (-0.1531, -0.0062) | 0.9830 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0113 | (-0.0006, 0.0236) | 0.0320 | 0.0113 | (-0.0029, 0.0259) | 0.0660 |
| proposed_vs_baseline_no_context | overall_quality | 0.0199 | (0.0026, 0.0377) | 0.0117 | 0.0199 | (-0.0048, 0.0457) | 0.0623 |
| controlled_vs_proposed_raw | context_relevance | 0.1694 | (0.1396, 0.1992) | 0.0000 | 0.1694 | (0.1298, 0.2054) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.1749 | (0.1387, 0.2130) | 0.0000 | 0.1749 | (0.1322, 0.2195) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0580 | (0.0395, 0.0761) | 0.0000 | 0.0580 | (0.0297, 0.0854) | 0.0000 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2211 | (0.1815, 0.2612) | 0.0000 | 0.2211 | (0.1678, 0.2681) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0487 | (0.0384, 0.0591) | 0.0000 | 0.0487 | (0.0360, 0.0627) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2040 | (0.1599, 0.2499) | 0.0000 | 0.2040 | (0.1650, 0.2422) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.0581 | (0.0213, 0.0989) | 0.0000 | 0.0581 | (-0.0184, 0.1585) | 0.1010 |
| controlled_vs_proposed_raw | distinct1 | 0.0016 | (-0.0072, 0.0103) | 0.3550 | 0.0016 | (-0.0083, 0.0122) | 0.3917 |
| controlled_vs_proposed_raw | length_score | 0.2182 | (0.1375, 0.2961) | 0.0000 | 0.2182 | (0.1149, 0.3146) | 0.0000 |
| controlled_vs_proposed_raw | sentence_score | 0.1375 | (0.1000, 0.1750) | 0.0000 | 0.1375 | (0.0781, 0.2031) | 0.0000 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0132 | (-0.0008, 0.0274) | 0.0330 | 0.0132 | (-0.0045, 0.0266) | 0.0610 |
| controlled_vs_proposed_raw | overall_quality | 0.1380 | (0.1171, 0.1586) | 0.0000 | 0.1380 | (0.1092, 0.1669) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2465 | (0.2259, 0.2685) | 0.0000 | 0.2465 | (0.2197, 0.2719) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.2008 | (0.1631, 0.2380) | 0.0000 | 0.2008 | (0.1625, 0.2393) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.0605 | (0.0380, 0.0822) | 0.0000 | 0.0605 | (0.0227, 0.0990) | 0.0000 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3228 | (0.2942, 0.3519) | 0.0000 | 0.3228 | (0.2857, 0.3583) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0683 | (0.0596, 0.0779) | 0.0000 | 0.0683 | (0.0602, 0.0767) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2313 | (0.1866, 0.2767) | 0.0000 | 0.2313 | (0.1981, 0.2602) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.0787 | (0.0396, 0.1223) | 0.0000 | 0.0787 | (-0.0053, 0.1948) | 0.0963 |
| controlled_vs_candidate_no_context | distinct1 | 0.0094 | (-0.0008, 0.0201) | 0.0383 | 0.0094 | (-0.0008, 0.0191) | 0.0353 |
| controlled_vs_candidate_no_context | length_score | 0.2057 | (0.1187, 0.2902) | 0.0003 | 0.2057 | (0.0643, 0.3554) | 0.0013 |
| controlled_vs_candidate_no_context | sentence_score | 0.1469 | (0.1031, 0.1875) | 0.0000 | 0.1469 | (0.0781, 0.2188) | 0.0000 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0490 | (0.0340, 0.0634) | 0.0000 | 0.0490 | (0.0320, 0.0670) | 0.0000 |
| controlled_vs_candidate_no_context | overall_quality | 0.1823 | (0.1671, 0.1975) | 0.0000 | 0.1823 | (0.1668, 0.1990) | 0.0000 |
| controlled_vs_baseline_no_context | context_relevance | 0.2359 | (0.2136, 0.2588) | 0.0000 | 0.2359 | (0.2052, 0.2638) | 0.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1853 | (0.1528, 0.2196) | 0.0000 | 0.1853 | (0.1714, 0.1988) | 0.0000 |
| controlled_vs_baseline_no_context | naturalness | -0.0102 | (-0.0251, 0.0051) | 0.8980 | -0.0102 | (-0.0342, 0.0124) | 0.7943 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3075 | (0.2785, 0.3371) | 0.0000 | 0.3075 | (0.2645, 0.3473) | 0.0000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0689 | (0.0594, 0.0790) | 0.0000 | 0.0689 | (0.0630, 0.0774) | 0.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2262 | (0.1870, 0.2673) | 0.0000 | 0.2262 | (0.2076, 0.2445) | 0.0000 |
| controlled_vs_baseline_no_context | persona_style | 0.0219 | (-0.0048, 0.0488) | 0.0527 | 0.0219 | (-0.0030, 0.0484) | 0.0393 |
| controlled_vs_baseline_no_context | distinct1 | -0.0377 | (-0.0445, -0.0311) | 1.0000 | -0.0377 | (-0.0465, -0.0283) | 1.0000 |
| controlled_vs_baseline_no_context | length_score | -0.0036 | (-0.0685, 0.0643) | 0.5350 | -0.0036 | (-0.1173, 0.1009) | 0.5113 |
| controlled_vs_baseline_no_context | sentence_score | 0.0563 | (0.0125, 0.0969) | 0.0057 | 0.0563 | (0.0000, 0.1156) | 0.0347 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0245 | (0.0087, 0.0396) | 0.0007 | 0.0245 | (0.0031, 0.0466) | 0.0137 |
| controlled_vs_baseline_no_context | overall_quality | 0.1578 | (0.1441, 0.1719) | 0.0000 | 0.1578 | (0.1444, 0.1714) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0276 | (-0.0573, -0.0000) | 0.9750 | -0.0276 | (-0.0530, -0.0040) | 0.9863 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0317 | (-0.0694, 0.0040) | 0.9607 | -0.0317 | (-0.0490, -0.0149) | 1.0000 |
| controlled_alt_vs_controlled_default | naturalness | -0.0060 | (-0.0265, 0.0149) | 0.7063 | -0.0060 | (-0.0167, 0.0079) | 0.8280 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0362 | (-0.0741, -0.0013) | 0.9790 | -0.0362 | (-0.0698, -0.0042) | 0.9847 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0075 | (-0.0196, 0.0035) | 0.9037 | -0.0075 | (-0.0175, 0.0011) | 0.9480 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0431 | (-0.0875, 0.0017) | 0.9710 | -0.0431 | (-0.0659, -0.0213) | 1.0000 |
| controlled_alt_vs_controlled_default | persona_style | 0.0136 | (-0.0086, 0.0381) | 0.1307 | 0.0136 | (-0.0049, 0.0379) | 0.1070 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0080 | (-0.0186, 0.0019) | 0.9480 | -0.0080 | (-0.0184, 0.0025) | 0.9330 |
| controlled_alt_vs_controlled_default | length_score | -0.0009 | (-0.0854, 0.0866) | 0.5293 | -0.0009 | (-0.0417, 0.0423) | 0.5130 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0210 | (-0.0603, 0.0187) | 0.8487 | -0.0210 | (-0.0549, 0.0156) | 0.8700 |
| controlled_alt_vs_controlled_default | bertscore_f1 | -0.0139 | (-0.0307, 0.0031) | 0.9440 | -0.0139 | (-0.0330, 0.0032) | 0.9423 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0246 | (-0.0412, -0.0075) | 0.9963 | -0.0246 | (-0.0336, -0.0169) | 1.0000 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1418 | (0.1115, 0.1712) | 0.0000 | 0.1418 | (0.1035, 0.1767) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1431 | (0.1099, 0.1772) | 0.0000 | 0.1431 | (0.1076, 0.1769) | 0.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0520 | (0.0312, 0.0719) | 0.0000 | 0.0520 | (0.0235, 0.0775) | 0.0007 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.1850 | (0.1423, 0.2251) | 0.0000 | 0.1850 | (0.1347, 0.2281) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0412 | (0.0321, 0.0502) | 0.0000 | 0.0412 | (0.0304, 0.0520) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1610 | (0.1225, 0.2024) | 0.0000 | 0.1610 | (0.1199, 0.2020) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0717 | (0.0344, 0.1103) | 0.0000 | 0.0717 | (-0.0057, 0.1664) | 0.0510 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0064 | (-0.0170, 0.0038) | 0.8907 | -0.0064 | (-0.0199, 0.0077) | 0.8120 |
| controlled_alt_vs_proposed_raw | length_score | 0.2173 | (0.1348, 0.2982) | 0.0000 | 0.2173 | (0.1083, 0.3220) | 0.0003 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1165 | (0.0705, 0.1598) | 0.0000 | 0.1165 | (0.0478, 0.1875) | 0.0003 |
| controlled_alt_vs_proposed_raw | bertscore_f1 | -0.0006 | (-0.0142, 0.0132) | 0.5407 | -0.0006 | (-0.0256, 0.0179) | 0.4807 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1133 | (0.0940, 0.1340) | 0.0000 | 0.1133 | (0.0864, 0.1391) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2189 | (0.1979, 0.2408) | 0.0000 | 0.2189 | (0.1940, 0.2435) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1691 | (0.1348, 0.2046) | 0.0000 | 0.1691 | (0.1378, 0.1973) | 0.0000 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0544 | (0.0313, 0.0767) | 0.0000 | 0.0544 | (0.0158, 0.0902) | 0.0010 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.2866 | (0.2598, 0.3140) | 0.0000 | 0.2866 | (0.2522, 0.3198) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0608 | (0.0529, 0.0691) | 0.0000 | 0.0608 | (0.0550, 0.0668) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1883 | (0.1486, 0.2302) | 0.0000 | 0.1883 | (0.1583, 0.2188) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0924 | (0.0520, 0.1345) | 0.0000 | 0.0924 | (0.0061, 0.2020) | 0.0187 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0014 | (-0.0104, 0.0133) | 0.4040 | 0.0014 | (-0.0076, 0.0111) | 0.3737 |
| controlled_alt_vs_candidate_no_context | length_score | 0.2048 | (0.1214, 0.2848) | 0.0000 | 0.2048 | (0.0551, 0.3473) | 0.0017 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1259 | (0.0790, 0.1728) | 0.0000 | 0.1259 | (0.0446, 0.2067) | 0.0010 |
| controlled_alt_vs_candidate_no_context | bertscore_f1 | 0.0351 | (0.0216, 0.0486) | 0.0000 | 0.0351 | (0.0103, 0.0598) | 0.0020 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1577 | (0.1410, 0.1743) | 0.0000 | 0.1577 | (0.1408, 0.1749) | 0.0000 |
| controlled_alt_vs_baseline_no_context | context_relevance | 0.2083 | (0.1875, 0.2297) | 0.0000 | 0.2083 | (0.1831, 0.2339) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 0.1536 | (0.1234, 0.1837) | 0.0000 | 0.1536 | (0.1285, 0.1747) | 0.0000 |
| controlled_alt_vs_baseline_no_context | naturalness | -0.0162 | (-0.0353, 0.0024) | 0.9573 | -0.0162 | (-0.0388, 0.0041) | 0.9327 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 0.2713 | (0.2440, 0.2989) | 0.0000 | 0.2713 | (0.2367, 0.3080) | 0.0000 |
| controlled_alt_vs_baseline_no_context | context_overlap | 0.0614 | (0.0535, 0.0693) | 0.0000 | 0.0614 | (0.0568, 0.0657) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 0.1831 | (0.1469, 0.2236) | 0.0000 | 0.1831 | (0.1544, 0.2151) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_style | 0.0355 | (0.0100, 0.0620) | 0.0030 | 0.0355 | (-0.0044, 0.0793) | 0.0413 |
| controlled_alt_vs_baseline_no_context | distinct1 | -0.0457 | (-0.0550, -0.0368) | 1.0000 | -0.0457 | (-0.0512, -0.0402) | 1.0000 |
| controlled_alt_vs_baseline_no_context | length_score | -0.0045 | (-0.0827, 0.0717) | 0.5500 | -0.0045 | (-0.1208, 0.0988) | 0.5253 |
| controlled_alt_vs_baseline_no_context | sentence_score | 0.0353 | (-0.0116, 0.0799) | 0.0587 | 0.0353 | (-0.0518, 0.1223) | 0.2047 |
| controlled_alt_vs_baseline_no_context | bertscore_f1 | 0.0107 | (-0.0032, 0.0240) | 0.0610 | 0.0107 | (-0.0150, 0.0332) | 0.1913 |
| controlled_alt_vs_baseline_no_context | overall_quality | 0.1332 | (0.1203, 0.1473) | 0.0000 | 0.1332 | (0.1237, 0.1440) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2359 | (0.2134, 0.2588) | 0.0000 | 0.2359 | (0.2059, 0.2645) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1853 | (0.1552, 0.2192) | 0.0000 | 0.1853 | (0.1716, 0.1982) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0102 | (-0.0256, 0.0061) | 0.8903 | -0.0102 | (-0.0349, 0.0138) | 0.7807 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3075 | (0.2794, 0.3366) | 0.0000 | 0.3075 | (0.2631, 0.3481) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0689 | (0.0596, 0.0790) | 0.0000 | 0.0689 | (0.0631, 0.0776) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2262 | (0.1861, 0.2679) | 0.0000 | 0.2262 | (0.2074, 0.2455) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0219 | (-0.0036, 0.0476) | 0.0477 | 0.0219 | (-0.0033, 0.0473) | 0.0467 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0377 | (-0.0443, -0.0306) | 1.0000 | -0.0377 | (-0.0465, -0.0282) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | -0.0036 | (-0.0750, 0.0649) | 0.5330 | -0.0036 | (-0.1113, 0.1024) | 0.4967 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0563 | (0.0156, 0.0969) | 0.0027 | 0.0563 | (0.0000, 0.1125) | 0.0270 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0245 | (0.0095, 0.0393) | 0.0020 | 0.0245 | (0.0032, 0.0467) | 0.0100 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1578 | (0.1438, 0.1722) | 0.0000 | 0.1578 | (0.1449, 0.1718) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 59 | 15 | 38 | 0.6964 | 0.7973 |
| proposed_vs_candidate_no_context | persona_consistency | 31 | 17 | 64 | 0.5625 | 0.6458 |
| proposed_vs_candidate_no_context | naturalness | 39 | 35 | 38 | 0.5179 | 0.5270 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 47 | 6 | 59 | 0.6830 | 0.8868 |
| proposed_vs_candidate_no_context | context_overlap | 57 | 17 | 38 | 0.6786 | 0.7703 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 20 | 10 | 82 | 0.5446 | 0.6667 |
| proposed_vs_candidate_no_context | persona_style | 19 | 10 | 83 | 0.5402 | 0.6552 |
| proposed_vs_candidate_no_context | distinct1 | 38 | 29 | 45 | 0.5402 | 0.5672 |
| proposed_vs_candidate_no_context | length_score | 35 | 34 | 43 | 0.5045 | 0.5072 |
| proposed_vs_candidate_no_context | sentence_score | 19 | 16 | 77 | 0.5134 | 0.5429 |
| proposed_vs_candidate_no_context | bertscore_f1 | 68 | 18 | 26 | 0.7232 | 0.7907 |
| proposed_vs_candidate_no_context | overall_quality | 68 | 18 | 26 | 0.7232 | 0.7907 |
| proposed_vs_baseline_no_context | context_relevance | 69 | 43 | 0 | 0.6161 | 0.6161 |
| proposed_vs_baseline_no_context | persona_consistency | 29 | 33 | 50 | 0.4821 | 0.4677 |
| proposed_vs_baseline_no_context | naturalness | 26 | 86 | 0 | 0.2321 | 0.2321 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 45 | 17 | 50 | 0.6250 | 0.7258 |
| proposed_vs_baseline_no_context | context_overlap | 72 | 40 | 0 | 0.6429 | 0.6429 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 22 | 22 | 68 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context | persona_style | 11 | 19 | 82 | 0.4643 | 0.3667 |
| proposed_vs_baseline_no_context | distinct1 | 17 | 78 | 17 | 0.2277 | 0.1789 |
| proposed_vs_baseline_no_context | length_score | 29 | 82 | 1 | 0.2634 | 0.2613 |
| proposed_vs_baseline_no_context | sentence_score | 14 | 40 | 58 | 0.3839 | 0.2593 |
| proposed_vs_baseline_no_context | bertscore_f1 | 65 | 47 | 0 | 0.5804 | 0.5804 |
| proposed_vs_baseline_no_context | overall_quality | 53 | 59 | 0 | 0.4732 | 0.4732 |
| controlled_vs_proposed_raw | context_relevance | 94 | 18 | 0 | 0.8393 | 0.8393 |
| controlled_vs_proposed_raw | persona_consistency | 84 | 12 | 16 | 0.8214 | 0.8750 |
| controlled_vs_proposed_raw | naturalness | 83 | 29 | 0 | 0.7411 | 0.7411 |
| controlled_vs_proposed_raw | context_keyword_coverage | 89 | 12 | 11 | 0.8438 | 0.8812 |
| controlled_vs_proposed_raw | context_overlap | 87 | 25 | 0 | 0.7768 | 0.7768 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 83 | 11 | 18 | 0.8214 | 0.8830 |
| controlled_vs_proposed_raw | persona_style | 26 | 15 | 71 | 0.5491 | 0.6341 |
| controlled_vs_proposed_raw | distinct1 | 61 | 50 | 1 | 0.5491 | 0.5495 |
| controlled_vs_proposed_raw | length_score | 73 | 30 | 9 | 0.6920 | 0.7087 |
| controlled_vs_proposed_raw | sentence_score | 51 | 7 | 54 | 0.6964 | 0.8793 |
| controlled_vs_proposed_raw | bertscore_f1 | 68 | 44 | 0 | 0.6071 | 0.6071 |
| controlled_vs_proposed_raw | overall_quality | 102 | 10 | 0 | 0.9107 | 0.9107 |
| controlled_vs_candidate_no_context | context_relevance | 112 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 91 | 11 | 10 | 0.8571 | 0.8922 |
| controlled_vs_candidate_no_context | naturalness | 86 | 26 | 0 | 0.7679 | 0.7679 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 110 | 0 | 2 | 0.9911 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 110 | 2 | 0 | 0.9821 | 0.9821 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 91 | 11 | 10 | 0.8571 | 0.8922 |
| controlled_vs_candidate_no_context | persona_style | 29 | 11 | 72 | 0.5804 | 0.7250 |
| controlled_vs_candidate_no_context | distinct1 | 60 | 51 | 1 | 0.5402 | 0.5405 |
| controlled_vs_candidate_no_context | length_score | 75 | 34 | 3 | 0.6830 | 0.6881 |
| controlled_vs_candidate_no_context | sentence_score | 57 | 10 | 45 | 0.7098 | 0.8507 |
| controlled_vs_candidate_no_context | bertscore_f1 | 86 | 26 | 0 | 0.7679 | 0.7679 |
| controlled_vs_candidate_no_context | overall_quality | 110 | 2 | 0 | 0.9821 | 0.9821 |
| controlled_vs_baseline_no_context | context_relevance | 111 | 1 | 0 | 0.9911 | 0.9911 |
| controlled_vs_baseline_no_context | persona_consistency | 90 | 6 | 16 | 0.8750 | 0.9375 |
| controlled_vs_baseline_no_context | naturalness | 53 | 58 | 1 | 0.4777 | 0.4775 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 108 | 0 | 4 | 0.9821 | 1.0000 |
| controlled_vs_baseline_no_context | context_overlap | 109 | 3 | 0 | 0.9732 | 0.9732 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 89 | 5 | 18 | 0.8750 | 0.9468 |
| controlled_vs_baseline_no_context | persona_style | 15 | 11 | 86 | 0.5179 | 0.5769 |
| controlled_vs_baseline_no_context | distinct1 | 14 | 93 | 5 | 0.1473 | 0.1308 |
| controlled_vs_baseline_no_context | length_score | 56 | 52 | 4 | 0.5179 | 0.5185 |
| controlled_vs_baseline_no_context | sentence_score | 33 | 15 | 64 | 0.5804 | 0.6875 |
| controlled_vs_baseline_no_context | bertscore_f1 | 69 | 43 | 0 | 0.6161 | 0.6161 |
| controlled_vs_baseline_no_context | overall_quality | 111 | 1 | 0 | 0.9911 | 0.9911 |
| controlled_alt_vs_controlled_default | context_relevance | 41 | 58 | 13 | 0.4241 | 0.4141 |
| controlled_alt_vs_controlled_default | persona_consistency | 25 | 37 | 50 | 0.4464 | 0.4032 |
| controlled_alt_vs_controlled_default | naturalness | 51 | 49 | 12 | 0.5089 | 0.5100 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 30 | 45 | 37 | 0.4330 | 0.4000 |
| controlled_alt_vs_controlled_default | context_overlap | 48 | 50 | 14 | 0.4911 | 0.4898 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 18 | 29 | 65 | 0.4509 | 0.3830 |
| controlled_alt_vs_controlled_default | persona_style | 16 | 14 | 82 | 0.5089 | 0.5333 |
| controlled_alt_vs_controlled_default | distinct1 | 48 | 51 | 13 | 0.4866 | 0.4848 |
| controlled_alt_vs_controlled_default | length_score | 48 | 48 | 16 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | sentence_score | 16 | 21 | 75 | 0.4777 | 0.4324 |
| controlled_alt_vs_controlled_default | bertscore_f1 | 42 | 62 | 8 | 0.4107 | 0.4038 |
| controlled_alt_vs_controlled_default | overall_quality | 33 | 71 | 8 | 0.3304 | 0.3173 |
| controlled_alt_vs_proposed_raw | context_relevance | 94 | 18 | 0 | 0.8393 | 0.8393 |
| controlled_alt_vs_proposed_raw | persona_consistency | 82 | 13 | 17 | 0.8080 | 0.8632 |
| controlled_alt_vs_proposed_raw | naturalness | 82 | 30 | 0 | 0.7321 | 0.7321 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 90 | 16 | 6 | 0.8304 | 0.8491 |
| controlled_alt_vs_proposed_raw | context_overlap | 92 | 20 | 0 | 0.8214 | 0.8214 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 80 | 11 | 21 | 0.8080 | 0.8791 |
| controlled_alt_vs_proposed_raw | persona_style | 30 | 11 | 71 | 0.5848 | 0.7317 |
| controlled_alt_vs_proposed_raw | distinct1 | 46 | 61 | 5 | 0.4330 | 0.4299 |
| controlled_alt_vs_proposed_raw | length_score | 77 | 33 | 2 | 0.6964 | 0.7000 |
| controlled_alt_vs_proposed_raw | sentence_score | 52 | 15 | 45 | 0.6652 | 0.7761 |
| controlled_alt_vs_proposed_raw | bertscore_f1 | 56 | 56 | 0 | 0.5000 | 0.5000 |
| controlled_alt_vs_proposed_raw | overall_quality | 94 | 18 | 0 | 0.8393 | 0.8393 |
| controlled_alt_vs_candidate_no_context | context_relevance | 109 | 3 | 0 | 0.9732 | 0.9732 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 90 | 11 | 11 | 0.8527 | 0.8911 |
| controlled_alt_vs_candidate_no_context | naturalness | 75 | 37 | 0 | 0.6696 | 0.6696 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 105 | 0 | 7 | 0.9688 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 106 | 6 | 0 | 0.9464 | 0.9464 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 84 | 11 | 17 | 0.8259 | 0.8842 |
| controlled_alt_vs_candidate_no_context | persona_style | 34 | 10 | 68 | 0.6071 | 0.7727 |
| controlled_alt_vs_candidate_no_context | distinct1 | 57 | 50 | 5 | 0.5312 | 0.5327 |
| controlled_alt_vs_candidate_no_context | length_score | 66 | 39 | 7 | 0.6205 | 0.6286 |
| controlled_alt_vs_candidate_no_context | sentence_score | 57 | 16 | 39 | 0.6830 | 0.7808 |
| controlled_alt_vs_candidate_no_context | bertscore_f1 | 78 | 34 | 0 | 0.6964 | 0.6964 |
| controlled_alt_vs_candidate_no_context | overall_quality | 108 | 4 | 0 | 0.9643 | 0.9643 |
| controlled_alt_vs_baseline_no_context | context_relevance | 111 | 1 | 0 | 0.9911 | 0.9911 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 88 | 7 | 17 | 0.8616 | 0.9263 |
| controlled_alt_vs_baseline_no_context | naturalness | 52 | 60 | 0 | 0.4643 | 0.4643 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 105 | 0 | 7 | 0.9688 | 1.0000 |
| controlled_alt_vs_baseline_no_context | context_overlap | 110 | 2 | 0 | 0.9821 | 0.9821 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 84 | 6 | 22 | 0.8482 | 0.9333 |
| controlled_alt_vs_baseline_no_context | persona_style | 18 | 8 | 86 | 0.5446 | 0.6923 |
| controlled_alt_vs_baseline_no_context | distinct1 | 16 | 89 | 7 | 0.1741 | 0.1524 |
| controlled_alt_vs_baseline_no_context | length_score | 59 | 52 | 1 | 0.5312 | 0.5315 |
| controlled_alt_vs_baseline_no_context | sentence_score | 32 | 19 | 61 | 0.5580 | 0.6275 |
| controlled_alt_vs_baseline_no_context | bertscore_f1 | 64 | 48 | 0 | 0.5714 | 0.5714 |
| controlled_alt_vs_baseline_no_context | overall_quality | 111 | 1 | 0 | 0.9911 | 0.9911 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 111 | 1 | 0 | 0.9911 | 0.9911 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 90 | 6 | 16 | 0.8750 | 0.9375 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 53 | 58 | 1 | 0.4777 | 0.4775 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 108 | 0 | 4 | 0.9821 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 109 | 3 | 0 | 0.9732 | 0.9732 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 89 | 5 | 18 | 0.8750 | 0.9468 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 15 | 11 | 86 | 0.5179 | 0.5769 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | 14 | 93 | 5 | 0.1473 | 0.1308 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 56 | 52 | 4 | 0.5179 | 0.5185 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 33 | 15 | 64 | 0.5804 | 0.6875 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 69 | 43 | 0 | 0.6161 | 0.6161 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 111 | 1 | 0 | 0.9911 | 0.9911 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.2589 | 0.4375 | 0.5625 |
| proposed_contextual_controlled_alt | 0.0000 | 0.0000 | 0.2232 | 0.4464 | 0.5446 |
| proposed_contextual | 0.0000 | 0.0000 | 0.3839 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.5089 | 0.0000 | 0.0000 |
| baseline_no_context | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `8`
- Unique template signatures: `71`
- Template signature ratio: `0.6339`
- Effective sample size by source clustering: `8.00`
- Effective sample size by template-signature clustering: `56.00`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: enabled.

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
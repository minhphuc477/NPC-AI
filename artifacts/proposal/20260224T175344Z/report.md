# Proposal Alignment Evaluation Report

- Run ID: `20260224T175344Z`
- Generated: `2026-02-24T17:54:32.327676+00:00`
- Scenarios: `artifacts\proposal\20260224T175344Z\scenarios.jsonl`
- Scenario count: `112`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off
- `baseline_no_context`: model `phi3:mini`, dynamic_context=off, response_control=off
- `baseline_no_context_phi3_latest`: model `phi3:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2782 (0.2632, 0.2937) | 0.2324 (0.2061, 0.2596) | 0.8987 (0.8903, 0.9061) | 0.3560 (0.3439, 0.3678) | 0.1043 |
| proposed_contextual | 0.0733 (0.0555, 0.0914) | 0.1537 (0.1286, 0.1806) | 0.8102 (0.7969, 0.8247) | 0.2276 (0.2121, 0.2433) | 0.0701 |
| candidate_no_context | 0.0222 (0.0171, 0.0281) | 0.1377 (0.1137, 0.1646) | 0.7989 (0.7866, 0.8127) | 0.1961 (0.1859, 0.2068) | 0.0394 |
| baseline_no_context | 0.0428 (0.0348, 0.0508) | 0.1734 (0.1538, 0.1941) | 0.8732 (0.8634, 0.8823) | 0.2303 (0.2226, 0.2384) | 0.0557 |
| baseline_no_context_phi3_latest | 0.0474 (0.0376, 0.0574) | 0.1752 (0.1549, 0.1961) | 0.8838 (0.8738, 0.8938) | 0.2350 (0.2263, 0.2440) | 0.0560 |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0511 | 2.2967 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0159 | 0.1158 |
| proposed_vs_candidate_no_context | naturalness | 0.0113 | 0.0141 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0676 | 4.4312 |
| proposed_vs_candidate_no_context | context_overlap | 0.0126 | 0.3279 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0157 | 0.3203 |
| proposed_vs_candidate_no_context | persona_style | 0.0168 | 0.0342 |
| proposed_vs_candidate_no_context | distinct1 | 0.0049 | 0.0052 |
| proposed_vs_candidate_no_context | length_score | 0.0262 | 0.0971 |
| proposed_vs_candidate_no_context | sentence_score | 0.0406 | 0.0573 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0307 | 0.7798 |
| proposed_vs_candidate_no_context | overall_quality | 0.0315 | 0.1608 |
| proposed_vs_baseline_no_context | context_relevance | 0.0305 | 0.7141 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0197 | -0.1138 |
| proposed_vs_baseline_no_context | naturalness | -0.0630 | -0.0722 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0390 | 0.8904 |
| proposed_vs_baseline_no_context | context_overlap | 0.0108 | 0.2680 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0017 | -0.0256 |
| proposed_vs_baseline_no_context | persona_style | -0.0919 | -0.1529 |
| proposed_vs_baseline_no_context | distinct1 | -0.0374 | -0.0383 |
| proposed_vs_baseline_no_context | length_score | -0.1824 | -0.3815 |
| proposed_vs_baseline_no_context | sentence_score | -0.1156 | -0.1336 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0143 | 0.2571 |
| proposed_vs_baseline_no_context | overall_quality | -0.0028 | -0.0119 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 0.0259 | 0.5468 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | -0.0215 | -0.1227 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | -0.0736 | -0.0833 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.0330 | 0.6638 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 0.0093 | 0.2226 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | -0.0099 | -0.1320 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | -0.0680 | -0.1179 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | -0.0406 | -0.0414 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | -0.2226 | -0.4294 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | -0.1281 | -0.1459 |
| proposed_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0140 | 0.2502 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | -0.0074 | -0.0317 |
| controlled_vs_proposed_raw | context_relevance | 0.2049 | 2.7937 |
| controlled_vs_proposed_raw | persona_consistency | 0.0787 | 0.5124 |
| controlled_vs_proposed_raw | naturalness | 0.0885 | 0.1093 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2725 | 3.2913 |
| controlled_vs_proposed_raw | context_overlap | 0.0470 | 0.9168 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.0766 | 1.1810 |
| controlled_vs_proposed_raw | persona_style | 0.0874 | 0.1717 |
| controlled_vs_proposed_raw | distinct1 | 0.0000 | 0.0000 |
| controlled_vs_proposed_raw | length_score | 0.3223 | 1.0895 |
| controlled_vs_proposed_raw | sentence_score | 0.2406 | 0.3208 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0342 | 0.4882 |
| controlled_vs_proposed_raw | overall_quality | 0.1284 | 0.5641 |
| controlled_vs_candidate_no_context | context_relevance | 0.2559 | 11.5068 |
| controlled_vs_candidate_no_context | persona_consistency | 0.0947 | 0.6876 |
| controlled_vs_candidate_no_context | naturalness | 0.0998 | 0.1249 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3401 | 22.3070 |
| controlled_vs_candidate_no_context | context_overlap | 0.0596 | 1.5453 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.0923 | 1.8797 |
| controlled_vs_candidate_no_context | persona_style | 0.1042 | 0.2118 |
| controlled_vs_candidate_no_context | distinct1 | 0.0049 | 0.0052 |
| controlled_vs_candidate_no_context | length_score | 0.3485 | 1.2925 |
| controlled_vs_candidate_no_context | sentence_score | 0.2812 | 0.3965 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0649 | 1.6488 |
| controlled_vs_candidate_no_context | overall_quality | 0.1599 | 0.8157 |
| controlled_vs_baseline_no_context | context_relevance | 0.2354 | 5.5028 |
| controlled_vs_baseline_no_context | persona_consistency | 0.0590 | 0.3403 |
| controlled_vs_baseline_no_context | naturalness | 0.0255 | 0.0292 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3115 | 7.1121 |
| controlled_vs_baseline_no_context | context_overlap | 0.0578 | 1.4305 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.0749 | 1.1252 |
| controlled_vs_baseline_no_context | persona_style | -0.0045 | -0.0074 |
| controlled_vs_baseline_no_context | distinct1 | -0.0374 | -0.0383 |
| controlled_vs_baseline_no_context | length_score | 0.1399 | 0.2925 |
| controlled_vs_baseline_no_context | sentence_score | 0.1250 | 0.1444 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0485 | 0.8709 |
| controlled_vs_baseline_no_context | overall_quality | 0.1256 | 0.5454 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.2308 | 4.8682 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.0572 | 0.3268 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | 0.0149 | 0.0169 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.3056 | 6.1397 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0563 | 1.3434 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.0667 | 0.8930 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | 0.0194 | 0.0336 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0406 | -0.0414 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | 0.0997 | 0.1923 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.1125 | 0.1281 |
| controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0482 | 0.8606 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1209 | 0.5146 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2354 | 5.5028 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.0590 | 0.3403 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 0.0255 | 0.0292 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3115 | 7.1121 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0578 | 1.4305 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.0749 | 1.1252 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | -0.0045 | -0.0074 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0374 | -0.0383 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.1399 | 0.2925 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.1250 | 0.1444 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0485 | 0.8709 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1256 | 0.5454 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.2308 | 4.8682 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.0572 | 0.3268 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | 0.0149 | 0.0169 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.3056 | 6.1397 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0563 | 1.3434 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.0667 | 0.8930 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | 0.0194 | 0.0336 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0406 | -0.0414 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | 0.0997 | 0.1923 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.1125 | 0.1281 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0482 | 0.8606 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1209 | 0.5146 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) |
|---|---|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0511 | (0.0323, 0.0704) | 0.0000 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0159 | (-0.0142, 0.0469) | 0.1503 |
| proposed_vs_candidate_no_context | naturalness | 0.0113 | (-0.0037, 0.0258) | 0.0737 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0676 | (0.0422, 0.0948) | 0.0000 |
| proposed_vs_candidate_no_context | context_overlap | 0.0126 | (0.0069, 0.0186) | 0.0000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0157 | (-0.0193, 0.0512) | 0.1957 |
| proposed_vs_candidate_no_context | persona_style | 0.0168 | (-0.0105, 0.0445) | 0.1130 |
| proposed_vs_candidate_no_context | distinct1 | 0.0049 | (-0.0021, 0.0119) | 0.0873 |
| proposed_vs_candidate_no_context | length_score | 0.0262 | (-0.0268, 0.0839) | 0.1767 |
| proposed_vs_candidate_no_context | sentence_score | 0.0406 | (0.0062, 0.0750) | 0.0100 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0307 | (0.0184, 0.0440) | 0.0000 |
| proposed_vs_candidate_no_context | overall_quality | 0.0315 | (0.0149, 0.0482) | 0.0000 |
| proposed_vs_baseline_no_context | context_relevance | 0.0305 | (0.0117, 0.0510) | 0.0003 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0197 | (-0.0485, 0.0093) | 0.9113 |
| proposed_vs_baseline_no_context | naturalness | -0.0630 | (-0.0800, -0.0461) | 1.0000 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0390 | (0.0132, 0.0656) | 0.0013 |
| proposed_vs_baseline_no_context | context_overlap | 0.0108 | (0.0034, 0.0180) | 0.0013 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0017 | (-0.0339, 0.0307) | 0.5583 |
| proposed_vs_baseline_no_context | persona_style | -0.0919 | (-0.1321, -0.0556) | 1.0000 |
| proposed_vs_baseline_no_context | distinct1 | -0.0374 | (-0.0457, -0.0291) | 1.0000 |
| proposed_vs_baseline_no_context | length_score | -0.1824 | (-0.2479, -0.1164) | 1.0000 |
| proposed_vs_baseline_no_context | sentence_score | -0.1156 | (-0.1594, -0.0719) | 1.0000 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0143 | (-0.0000, 0.0294) | 0.0260 |
| proposed_vs_baseline_no_context | overall_quality | -0.0028 | (-0.0190, 0.0149) | 0.6237 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 0.0259 | (0.0063, 0.0474) | 0.0043 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | -0.0215 | (-0.0504, 0.0076) | 0.9280 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | -0.0736 | (-0.0914, -0.0549) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.0330 | (0.0050, 0.0623) | 0.0097 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 0.0093 | (0.0025, 0.0161) | 0.0043 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | -0.0099 | (-0.0423, 0.0233) | 0.7247 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | -0.0680 | (-0.1083, -0.0315) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | -0.0406 | (-0.0481, -0.0331) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | -0.2226 | (-0.2917, -0.1515) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | -0.1281 | (-0.1687, -0.0875) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0140 | (0.0006, 0.0287) | 0.0223 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | -0.0074 | (-0.0236, 0.0092) | 0.8143 |
| controlled_vs_proposed_raw | context_relevance | 0.2049 | (0.1832, 0.2257) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.0787 | (0.0469, 0.1101) | 0.0003 |
| controlled_vs_proposed_raw | naturalness | 0.0885 | (0.0718, 0.1045) | 0.0000 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2725 | (0.2466, 0.2997) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0470 | (0.0383, 0.0556) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.0766 | (0.0378, 0.1113) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.0874 | (0.0511, 0.1279) | 0.0000 |
| controlled_vs_proposed_raw | distinct1 | 0.0000 | (-0.0074, 0.0077) | 0.4920 |
| controlled_vs_proposed_raw | length_score | 0.3223 | (0.2556, 0.3824) | 0.0000 |
| controlled_vs_proposed_raw | sentence_score | 0.2406 | (0.2062, 0.2719) | 0.0000 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0342 | (0.0209, 0.0480) | 0.0000 |
| controlled_vs_proposed_raw | overall_quality | 0.1284 | (0.1117, 0.1454) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2559 | (0.2400, 0.2730) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.0947 | (0.0631, 0.1274) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.0998 | (0.0837, 0.1145) | 0.0000 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3401 | (0.3189, 0.3618) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0596 | (0.0527, 0.0667) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.0923 | (0.0537, 0.1298) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.1042 | (0.0675, 0.1428) | 0.0000 |
| controlled_vs_candidate_no_context | distinct1 | 0.0049 | (-0.0028, 0.0123) | 0.0993 |
| controlled_vs_candidate_no_context | length_score | 0.3485 | (0.2860, 0.4101) | 0.0000 |
| controlled_vs_candidate_no_context | sentence_score | 0.2812 | (0.2531, 0.3062) | 0.0000 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0649 | (0.0536, 0.0764) | 0.0000 |
| controlled_vs_candidate_no_context | overall_quality | 0.1599 | (0.1439, 0.1748) | 0.0000 |
| controlled_vs_baseline_no_context | context_relevance | 0.2354 | (0.2198, 0.2510) | 0.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 0.0590 | (0.0311, 0.0859) | 0.0000 |
| controlled_vs_baseline_no_context | naturalness | 0.0255 | (0.0134, 0.0371) | 0.0000 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3115 | (0.2917, 0.3326) | 0.0000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0578 | (0.0509, 0.0652) | 0.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.0749 | (0.0420, 0.1085) | 0.0000 |
| controlled_vs_baseline_no_context | persona_style | -0.0045 | (-0.0210, 0.0102) | 0.7157 |
| controlled_vs_baseline_no_context | distinct1 | -0.0374 | (-0.0436, -0.0310) | 1.0000 |
| controlled_vs_baseline_no_context | length_score | 0.1399 | (0.0848, 0.1926) | 0.0000 |
| controlled_vs_baseline_no_context | sentence_score | 0.1250 | (0.0906, 0.1594) | 0.0000 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0485 | (0.0351, 0.0622) | 0.0000 |
| controlled_vs_baseline_no_context | overall_quality | 0.1256 | (0.1138, 0.1381) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.2308 | (0.2134, 0.2485) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.0572 | (0.0285, 0.0862) | 0.0003 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | 0.0149 | (0.0026, 0.0273) | 0.0100 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.3056 | (0.2836, 0.3292) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0563 | (0.0489, 0.0639) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.0667 | (0.0330, 0.1006) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | 0.0194 | (0.0012, 0.0388) | 0.0203 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0406 | (-0.0468, -0.0345) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | 0.0997 | (0.0420, 0.1554) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.1125 | (0.0813, 0.1437) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0482 | (0.0363, 0.0604) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1209 | (0.1086, 0.1332) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2354 | (0.2197, 0.2517) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.0590 | (0.0325, 0.0853) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 0.0255 | (0.0138, 0.0376) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3115 | (0.2909, 0.3330) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0578 | (0.0508, 0.0656) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.0749 | (0.0426, 0.1063) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | -0.0045 | (-0.0208, 0.0106) | 0.7197 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0374 | (-0.0436, -0.0313) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.1399 | (0.0845, 0.1935) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.1250 | (0.0938, 0.1562) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0485 | (0.0345, 0.0617) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1256 | (0.1138, 0.1375) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.2308 | (0.2136, 0.2479) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.0572 | (0.0283, 0.0859) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | 0.0149 | (0.0028, 0.0270) | 0.0110 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.3056 | (0.2836, 0.3292) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0563 | (0.0487, 0.0641) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.0667 | (0.0324, 0.0993) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | 0.0194 | (0.0016, 0.0387) | 0.0150 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0406 | (-0.0467, -0.0343) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | 0.0997 | (0.0423, 0.1560) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.1125 | (0.0813, 0.1437) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0482 | (0.0365, 0.0603) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1209 | (0.1079, 0.1331) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 45 | 18 | 49 | 0.6205 | 0.7143 |
| proposed_vs_candidate_no_context | persona_consistency | 24 | 9 | 79 | 0.5670 | 0.7273 |
| proposed_vs_candidate_no_context | naturalness | 38 | 25 | 49 | 0.5580 | 0.6032 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 40 | 9 | 63 | 0.6384 | 0.8163 |
| proposed_vs_candidate_no_context | context_overlap | 44 | 18 | 50 | 0.6161 | 0.7097 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 16 | 9 | 87 | 0.5312 | 0.6400 |
| proposed_vs_candidate_no_context | persona_style | 14 | 6 | 92 | 0.5357 | 0.7000 |
| proposed_vs_candidate_no_context | distinct1 | 30 | 24 | 58 | 0.5268 | 0.5556 |
| proposed_vs_candidate_no_context | length_score | 35 | 26 | 51 | 0.5402 | 0.5738 |
| proposed_vs_candidate_no_context | sentence_score | 23 | 10 | 79 | 0.5580 | 0.6970 |
| proposed_vs_candidate_no_context | bertscore_f1 | 59 | 27 | 26 | 0.6429 | 0.6860 |
| proposed_vs_candidate_no_context | overall_quality | 60 | 26 | 26 | 0.6518 | 0.6977 |
| proposed_vs_baseline_no_context | context_relevance | 55 | 57 | 0 | 0.4911 | 0.4911 |
| proposed_vs_baseline_no_context | persona_consistency | 21 | 47 | 44 | 0.3839 | 0.3088 |
| proposed_vs_baseline_no_context | naturalness | 28 | 84 | 0 | 0.2500 | 0.2500 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 38 | 27 | 47 | 0.5491 | 0.5846 |
| proposed_vs_baseline_no_context | context_overlap | 65 | 47 | 0 | 0.5804 | 0.5804 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 19 | 30 | 63 | 0.4509 | 0.3878 |
| proposed_vs_baseline_no_context | persona_style | 5 | 27 | 80 | 0.4018 | 0.1562 |
| proposed_vs_baseline_no_context | distinct1 | 20 | 81 | 11 | 0.2277 | 0.1980 |
| proposed_vs_baseline_no_context | length_score | 25 | 82 | 5 | 0.2455 | 0.2336 |
| proposed_vs_baseline_no_context | sentence_score | 11 | 48 | 53 | 0.3348 | 0.1864 |
| proposed_vs_baseline_no_context | bertscore_f1 | 58 | 54 | 0 | 0.5179 | 0.5179 |
| proposed_vs_baseline_no_context | overall_quality | 40 | 72 | 0 | 0.3571 | 0.3571 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 63 | 49 | 0 | 0.5625 | 0.5625 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | 24 | 43 | 45 | 0.4152 | 0.3582 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | 26 | 86 | 0 | 0.2321 | 0.2321 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 36 | 25 | 51 | 0.5491 | 0.5902 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 67 | 45 | 0 | 0.5982 | 0.5982 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 19 | 27 | 66 | 0.4643 | 0.4130 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | 8 | 24 | 80 | 0.4286 | 0.2500 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | 12 | 83 | 17 | 0.1830 | 0.1263 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | 26 | 83 | 3 | 0.2455 | 0.2385 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | 10 | 51 | 51 | 0.3170 | 0.1639 |
| proposed_vs_baseline_no_context_phi3_latest | bertscore_f1 | 60 | 52 | 0 | 0.5357 | 0.5357 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | 41 | 71 | 0 | 0.3661 | 0.3661 |
| controlled_vs_proposed_raw | context_relevance | 106 | 6 | 0 | 0.9464 | 0.9464 |
| controlled_vs_proposed_raw | persona_consistency | 65 | 9 | 38 | 0.7500 | 0.8784 |
| controlled_vs_proposed_raw | naturalness | 94 | 18 | 0 | 0.8393 | 0.8393 |
| controlled_vs_proposed_raw | context_keyword_coverage | 105 | 3 | 4 | 0.9554 | 0.9722 |
| controlled_vs_proposed_raw | context_overlap | 98 | 14 | 0 | 0.8750 | 0.8750 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 48 | 9 | 55 | 0.6741 | 0.8421 |
| controlled_vs_proposed_raw | persona_style | 27 | 4 | 81 | 0.6027 | 0.8710 |
| controlled_vs_proposed_raw | distinct1 | 63 | 47 | 2 | 0.5714 | 0.5727 |
| controlled_vs_proposed_raw | length_score | 85 | 24 | 3 | 0.7723 | 0.7798 |
| controlled_vs_proposed_raw | sentence_score | 79 | 2 | 31 | 0.8438 | 0.9753 |
| controlled_vs_proposed_raw | bertscore_f1 | 80 | 32 | 0 | 0.7143 | 0.7143 |
| controlled_vs_proposed_raw | overall_quality | 100 | 12 | 0 | 0.8929 | 0.8929 |
| controlled_vs_candidate_no_context | context_relevance | 112 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 69 | 9 | 34 | 0.7679 | 0.8846 |
| controlled_vs_candidate_no_context | naturalness | 96 | 16 | 0 | 0.8571 | 0.8571 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 111 | 0 | 1 | 0.9955 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 111 | 1 | 0 | 0.9911 | 0.9911 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 50 | 9 | 53 | 0.6830 | 0.8475 |
| controlled_vs_candidate_no_context | persona_style | 33 | 2 | 77 | 0.6384 | 0.9429 |
| controlled_vs_candidate_no_context | distinct1 | 73 | 38 | 1 | 0.6562 | 0.6577 |
| controlled_vs_candidate_no_context | length_score | 90 | 20 | 2 | 0.8125 | 0.8182 |
| controlled_vs_candidate_no_context | sentence_score | 90 | 0 | 22 | 0.9018 | 1.0000 |
| controlled_vs_candidate_no_context | bertscore_f1 | 96 | 16 | 0 | 0.8571 | 0.8571 |
| controlled_vs_candidate_no_context | overall_quality | 106 | 6 | 0 | 0.9464 | 0.9464 |
| controlled_vs_baseline_no_context | context_relevance | 112 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 50 | 20 | 42 | 0.6339 | 0.7143 |
| controlled_vs_baseline_no_context | naturalness | 77 | 35 | 0 | 0.6875 | 0.6875 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 110 | 0 | 2 | 0.9911 | 1.0000 |
| controlled_vs_baseline_no_context | context_overlap | 108 | 4 | 0 | 0.9643 | 0.9643 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 44 | 14 | 54 | 0.6339 | 0.7586 |
| controlled_vs_baseline_no_context | persona_style | 8 | 8 | 96 | 0.5000 | 0.5000 |
| controlled_vs_baseline_no_context | distinct1 | 16 | 93 | 3 | 0.1562 | 0.1468 |
| controlled_vs_baseline_no_context | length_score | 76 | 32 | 4 | 0.6964 | 0.7037 |
| controlled_vs_baseline_no_context | sentence_score | 42 | 2 | 68 | 0.6786 | 0.9545 |
| controlled_vs_baseline_no_context | bertscore_f1 | 86 | 26 | 0 | 0.7679 | 0.7679 |
| controlled_vs_baseline_no_context | overall_quality | 111 | 1 | 0 | 0.9911 | 0.9911 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 111 | 0 | 1 | 0.9955 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 54 | 22 | 36 | 0.6429 | 0.7105 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | 65 | 46 | 1 | 0.5848 | 0.5856 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 110 | 0 | 2 | 0.9911 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 103 | 8 | 1 | 0.9241 | 0.9279 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 47 | 18 | 47 | 0.6295 | 0.7231 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | 16 | 6 | 90 | 0.5446 | 0.7273 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | 15 | 94 | 3 | 0.1473 | 0.1376 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | 70 | 37 | 5 | 0.6473 | 0.6542 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 37 | 1 | 74 | 0.6607 | 0.9737 |
| controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 86 | 26 | 0 | 0.7679 | 0.7679 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 108 | 4 | 0 | 0.9643 | 0.9643 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 112 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 50 | 20 | 42 | 0.6339 | 0.7143 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 77 | 35 | 0 | 0.6875 | 0.6875 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 110 | 0 | 2 | 0.9911 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 108 | 4 | 0 | 0.9643 | 0.9643 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 44 | 14 | 54 | 0.6339 | 0.7586 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 8 | 8 | 96 | 0.5000 | 0.5000 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | 16 | 93 | 3 | 0.1562 | 0.1468 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 76 | 32 | 4 | 0.6964 | 0.7037 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 42 | 2 | 68 | 0.6786 | 0.9545 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 86 | 26 | 0 | 0.7679 | 0.7679 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 111 | 1 | 0 | 0.9911 | 0.9911 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 111 | 0 | 1 | 0.9955 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 54 | 22 | 36 | 0.6429 | 0.7105 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | 65 | 46 | 1 | 0.5848 | 0.5856 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 110 | 0 | 2 | 0.9911 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 103 | 8 | 1 | 0.9241 | 0.9279 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 47 | 18 | 47 | 0.6295 | 0.7231 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | 16 | 6 | 90 | 0.5446 | 0.7273 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | 15 | 94 | 3 | 0.1473 | 0.1376 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | 70 | 37 | 5 | 0.6473 | 0.6542 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | 37 | 1 | 74 | 0.6607 | 0.9737 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 86 | 26 | 0 | 0.7679 | 0.7679 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 108 | 4 | 0 | 0.9643 | 0.9643 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

- BERTScore status: enabled.

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
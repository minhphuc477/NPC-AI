# Proposal Alignment Evaluation Report

- Run ID: `20260305T185920Z`
- Generated: `2026-03-05T18:59:28.539499+00:00`
- Scenarios: `artifacts\proposal_control_tuning\smoke_batched_compare\20260305T185920Z\scenarios.jsonl`
- Scenario count: `6`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_runtime`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off
- `baseline_no_context`: model `phi3:mini`, dynamic_context=off, response_control=off
- `baseline_no_context_phi3_latest`: model `phi3:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2705 (0.2370, 0.3078) | 0.4629 (0.3867, 0.5238) | 0.9129 (0.8756, 0.9397) | 0.4291 (0.4075, 0.4499) | 0.1175 |
| proposed_contextual_controlled_runtime | 0.2387 (0.2183, 0.2620) | 0.4355 (0.3524, 0.5117) | 0.8787 (0.8215, 0.9289) | 0.4019 (0.3691, 0.4299) | 0.1238 |
| proposed_contextual | 0.1779 (0.0668, 0.2890) | 0.2563 (0.2000, 0.3689) | 0.8689 (0.7855, 0.9431) | 0.3140 (0.2356, 0.3983) | 0.1101 |
| candidate_no_context | 0.0441 (0.0193, 0.0696) | 0.2571 (0.2000, 0.3333) | 0.8434 (0.7690, 0.9211) | 0.2535 (0.2136, 0.2985) | 0.0983 |
| baseline_no_context | 0.0872 (0.0722, 0.1119) | 0.2762 (0.2381, 0.3143) | 0.8464 (0.8117, 0.8981) | 0.2788 (0.2561, 0.2994) | 0.1042 |
| baseline_no_context_phi3_latest | 0.0870 (0.0439, 0.1398) | 0.2745 (0.2364, 0.3126) | 0.8397 (0.8061, 0.8861) | 0.2798 (0.2480, 0.3080) | 0.1268 |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.1339 | 3.0367 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0008 | -0.0032 |
| proposed_vs_candidate_no_context | naturalness | 0.0255 | 0.0303 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.1818 | 4.0000 |
| proposed_vs_candidate_no_context | context_overlap | 0.0220 | 0.5372 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | persona_style | -0.0042 | -0.0042 |
| proposed_vs_candidate_no_context | distinct1 | -0.0007 | -0.0008 |
| proposed_vs_candidate_no_context | length_score | 0.1000 | 0.2368 |
| proposed_vs_candidate_no_context | sentence_score | 0.0583 | 0.0707 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0118 | 0.1200 |
| proposed_vs_candidate_no_context | overall_quality | 0.0605 | 0.2386 |
| proposed_vs_baseline_no_context | context_relevance | 0.0908 | 1.0415 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0199 | -0.0720 |
| proposed_vs_baseline_no_context | naturalness | 0.0225 | 0.0266 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.1212 | 1.1429 |
| proposed_vs_baseline_no_context | context_overlap | 0.0198 | 0.4589 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0238 | -0.2500 |
| proposed_vs_baseline_no_context | persona_style | -0.0042 | -0.0042 |
| proposed_vs_baseline_no_context | distinct1 | -0.0596 | -0.0596 |
| proposed_vs_baseline_no_context | length_score | 0.1444 | 0.3824 |
| proposed_vs_baseline_no_context | sentence_score | 0.1750 | 0.2471 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0059 | 0.0568 |
| proposed_vs_baseline_no_context | overall_quality | 0.0351 | 0.1260 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 0.0910 | 1.0461 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | -0.0182 | -0.0663 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | 0.0292 | 0.0348 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.1212 | 1.1429 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 0.0204 | 0.4813 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | -0.0238 | -0.2500 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | 0.0042 | 0.0042 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | -0.0262 | -0.0271 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | 0.1111 | 0.2703 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | 0.1750 | 0.2471 |
| proposed_vs_baseline_no_context_phi3_latest | bertscore_f1 | -0.0167 | -0.1314 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | 0.0341 | 0.1220 |
| controlled_vs_proposed_raw | context_relevance | 0.0926 | 0.5204 |
| controlled_vs_proposed_raw | persona_consistency | 0.2066 | 0.8061 |
| controlled_vs_proposed_raw | naturalness | 0.0440 | 0.0506 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.1212 | 0.5333 |
| controlled_vs_proposed_raw | context_overlap | 0.0259 | 0.4116 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2619 | 3.6667 |
| controlled_vs_proposed_raw | persona_style | -0.0146 | -0.0146 |
| controlled_vs_proposed_raw | distinct1 | 0.0169 | 0.0180 |
| controlled_vs_proposed_raw | length_score | 0.1278 | 0.2447 |
| controlled_vs_proposed_raw | sentence_score | 0.1167 | 0.1321 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0074 | 0.0670 |
| controlled_vs_proposed_raw | overall_quality | 0.1152 | 0.3668 |
| controlled_vs_candidate_no_context | context_relevance | 0.2265 | 5.1375 |
| controlled_vs_candidate_no_context | persona_consistency | 0.2058 | 0.8002 |
| controlled_vs_candidate_no_context | naturalness | 0.0695 | 0.0824 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3030 | 6.6667 |
| controlled_vs_candidate_no_context | context_overlap | 0.0478 | 1.1699 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2619 | 3.6667 |
| controlled_vs_candidate_no_context | persona_style | -0.0187 | -0.0187 |
| controlled_vs_candidate_no_context | distinct1 | 0.0162 | 0.0172 |
| controlled_vs_candidate_no_context | length_score | 0.2278 | 0.5395 |
| controlled_vs_candidate_no_context | sentence_score | 0.1750 | 0.2121 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0192 | 0.1951 |
| controlled_vs_candidate_no_context | overall_quality | 0.1757 | 0.6930 |
| controlled_vs_baseline_no_context | context_relevance | 0.1834 | 2.1039 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1867 | 0.6761 |
| controlled_vs_baseline_no_context | naturalness | 0.0665 | 0.0786 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.2424 | 2.2857 |
| controlled_vs_baseline_no_context | context_overlap | 0.0456 | 1.0594 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2381 | 2.5000 |
| controlled_vs_baseline_no_context | persona_style | -0.0187 | -0.0187 |
| controlled_vs_baseline_no_context | distinct1 | -0.0427 | -0.0427 |
| controlled_vs_baseline_no_context | length_score | 0.2722 | 0.7206 |
| controlled_vs_baseline_no_context | sentence_score | 0.2917 | 0.4118 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0133 | 0.1276 |
| controlled_vs_baseline_no_context | overall_quality | 0.1503 | 0.5391 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.1836 | 2.1109 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1884 | 0.6863 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | 0.0732 | 0.0872 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.2424 | 2.2857 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0463 | 1.0910 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.2381 | 2.5000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | -0.0104 | -0.0105 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0093 | -0.0096 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | 0.2389 | 0.5811 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.2917 | 0.4118 |
| controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | -0.0093 | -0.0732 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1493 | 0.5336 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0319 | -0.1177 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0274 | -0.0591 |
| controlled_alt_vs_controlled_default | naturalness | -0.0342 | -0.0375 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0455 | -0.1304 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0001 | -0.0013 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0238 | -0.0714 |
| controlled_alt_vs_controlled_default | persona_style | -0.0417 | -0.0425 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0293 | -0.0306 |
| controlled_alt_vs_controlled_default | length_score | -0.0833 | -0.1282 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0583 | -0.0583 |
| controlled_alt_vs_controlled_default | bertscore_f1 | 0.0063 | 0.0538 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0272 | -0.0633 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.0608 | 0.3414 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1792 | 0.6993 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0098 | 0.0112 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.0758 | 0.3333 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0258 | 0.4098 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.2381 | 3.3333 |
| controlled_alt_vs_proposed_raw | persona_style | -0.0562 | -0.0565 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0124 | -0.0132 |
| controlled_alt_vs_proposed_raw | length_score | 0.0444 | 0.0851 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0583 | 0.0660 |
| controlled_alt_vs_proposed_raw | bertscore_f1 | 0.0137 | 0.1244 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.0880 | 0.2802 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.1946 | 4.4150 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1784 | 0.6938 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0353 | 0.0419 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.2576 | 5.6667 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0477 | 1.1672 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.2381 | 3.3333 |
| controlled_alt_vs_candidate_no_context | persona_style | -0.0604 | -0.0604 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0131 | -0.0140 |
| controlled_alt_vs_candidate_no_context | length_score | 0.1444 | 0.3421 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1167 | 0.1414 |
| controlled_alt_vs_candidate_no_context | bertscore_f1 | 0.0255 | 0.2593 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1485 | 0.5858 |
| controlled_alt_vs_baseline_no_context | context_relevance | 0.1515 | 1.7385 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 0.1593 | 0.5769 |
| controlled_alt_vs_baseline_no_context | naturalness | 0.0323 | 0.0382 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 0.1970 | 1.8571 |
| controlled_alt_vs_baseline_no_context | context_overlap | 0.0455 | 1.0568 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 0.2143 | 2.2500 |
| controlled_alt_vs_baseline_no_context | persona_style | -0.0604 | -0.0604 |
| controlled_alt_vs_baseline_no_context | distinct1 | -0.0720 | -0.0720 |
| controlled_alt_vs_baseline_no_context | length_score | 0.1889 | 0.5000 |
| controlled_alt_vs_baseline_no_context | sentence_score | 0.2333 | 0.3294 |
| controlled_alt_vs_baseline_no_context | bertscore_f1 | 0.0196 | 0.1883 |
| controlled_alt_vs_baseline_no_context | overall_quality | 0.1231 | 0.4416 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_relevance | 0.1517 | 1.7447 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1610 | 0.5865 |
| controlled_alt_vs_baseline_no_context_phi3_latest | naturalness | 0.0390 | 0.0465 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.1970 | 1.8571 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_overlap | 0.0462 | 1.0884 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.2143 | 2.2500 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_style | -0.0521 | -0.0525 |
| controlled_alt_vs_baseline_no_context_phi3_latest | distinct1 | -0.0386 | -0.0399 |
| controlled_alt_vs_baseline_no_context_phi3_latest | length_score | 0.1556 | 0.3784 |
| controlled_alt_vs_baseline_no_context_phi3_latest | sentence_score | 0.2333 | 0.3294 |
| controlled_alt_vs_baseline_no_context_phi3_latest | bertscore_f1 | -0.0030 | -0.0233 |
| controlled_alt_vs_baseline_no_context_phi3_latest | overall_quality | 0.1221 | 0.4365 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.1834 | 2.1039 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1867 | 0.6761 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 0.0665 | 0.0786 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.2424 | 2.2857 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0456 | 1.0594 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2381 | 2.5000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | -0.0187 | -0.0187 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0427 | -0.0427 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.2722 | 0.7206 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.2917 | 0.4118 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0133 | 0.1276 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1503 | 0.5391 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.1836 | 2.1109 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1884 | 0.6863 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | 0.0732 | 0.0872 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.2424 | 2.2857 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0463 | 1.0910 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.2381 | 2.5000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | -0.0104 | -0.0105 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0093 | -0.0096 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | 0.2389 | 0.5811 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.2917 | 0.4118 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | -0.0093 | -0.0732 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1493 | 0.5336 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.1339 | (0.0446, 0.2231) | 0.0003 | 0.1339 | (0.1339, 0.1339) | 0.0000 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0008 | (-0.0571, 0.0546) | 0.6453 | -0.0008 | (-0.0008, -0.0008) | 1.0000 |
| proposed_vs_candidate_no_context | naturalness | 0.0255 | (-0.0084, 0.0805) | 0.2003 | 0.0255 | (0.0255, 0.0255) | 0.0000 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.1818 | (0.0606, 0.3030) | 0.0017 | 0.1818 | (0.1818, 0.1818) | 0.0000 |
| proposed_vs_candidate_no_context | context_overlap | 0.0220 | (0.0021, 0.0432) | 0.0193 | 0.0220 | (0.0220, 0.0220) | 0.0000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0000 | (-0.0714, 0.0714) | 0.6427 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | persona_style | -0.0042 | (-0.0125, 0.0000) | 1.0000 | -0.0042 | (-0.0042, -0.0042) | 1.0000 |
| proposed_vs_candidate_no_context | distinct1 | -0.0007 | (-0.0129, 0.0102) | 0.5390 | -0.0007 | (-0.0007, -0.0007) | 1.0000 |
| proposed_vs_candidate_no_context | length_score | 0.1000 | (-0.0500, 0.3111) | 0.1807 | 0.1000 | (0.1000, 0.1000) | 0.0000 |
| proposed_vs_candidate_no_context | sentence_score | 0.0583 | (0.0000, 0.1750) | 0.3283 | 0.0583 | (0.0583, 0.0583) | 0.0000 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0118 | (-0.0457, 0.0756) | 0.3497 | 0.0118 | (0.0118, 0.0118) | 0.0000 |
| proposed_vs_candidate_no_context | overall_quality | 0.0605 | (0.0214, 0.1027) | 0.0017 | 0.0605 | (0.0605, 0.0605) | 0.0000 |
| proposed_vs_baseline_no_context | context_relevance | 0.0908 | (-0.0341, 0.2137) | 0.0823 | 0.0908 | (0.0908, 0.0908) | 0.0000 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0199 | (-0.0952, 0.0745) | 0.7117 | -0.0199 | (-0.0199, -0.0199) | 1.0000 |
| proposed_vs_baseline_no_context | naturalness | 0.0225 | (-0.0458, 0.0909) | 0.2590 | 0.0225 | (0.0225, 0.0225) | 0.0000 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.1212 | (-0.0455, 0.2879) | 0.0963 | 0.1212 | (0.1212, 0.1212) | 0.0000 |
| proposed_vs_baseline_no_context | context_overlap | 0.0198 | (-0.0092, 0.0472) | 0.0950 | 0.0198 | (0.0198, 0.0198) | 0.0000 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0238 | (-0.1190, 0.1190) | 0.7177 | -0.0238 | (-0.0238, -0.0238) | 1.0000 |
| proposed_vs_baseline_no_context | persona_style | -0.0042 | (-0.0125, 0.0000) | 1.0000 | -0.0042 | (-0.0042, -0.0042) | 1.0000 |
| proposed_vs_baseline_no_context | distinct1 | -0.0596 | (-0.0806, -0.0381) | 1.0000 | -0.0596 | (-0.0596, -0.0596) | 1.0000 |
| proposed_vs_baseline_no_context | length_score | 0.1444 | (-0.1167, 0.4333) | 0.1580 | 0.1444 | (0.1444, 0.1444) | 0.0000 |
| proposed_vs_baseline_no_context | sentence_score | 0.1750 | (0.0583, 0.2917) | 0.0163 | 0.1750 | (0.1750, 0.1750) | 0.0000 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0059 | (-0.0353, 0.0727) | 0.4037 | 0.0059 | (0.0059, 0.0059) | 0.0000 |
| proposed_vs_baseline_no_context | overall_quality | 0.0351 | (-0.0558, 0.1280) | 0.2313 | 0.0351 | (0.0351, 0.0351) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 0.0910 | (0.0035, 0.1750) | 0.0203 | 0.0910 | (0.0910, 0.0910) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | -0.0182 | (-0.0936, 0.0927) | 0.6667 | -0.0182 | (-0.0182, -0.0182) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | 0.0292 | (-0.0776, 0.1258) | 0.2833 | 0.0292 | (0.0292, 0.0292) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.1212 | (0.0148, 0.2424) | 0.0250 | 0.1212 | (0.1212, 0.1212) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 0.0204 | (-0.0097, 0.0485) | 0.0973 | 0.0204 | (0.0204, 0.0204) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | -0.0238 | (-0.1190, 0.0952) | 0.7340 | -0.0238 | (-0.0238, -0.0238) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | 0.0042 | (-0.0125, 0.0250) | 0.4453 | 0.0042 | (0.0042, 0.0042) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | -0.0262 | (-0.0662, 0.0140) | 0.9003 | -0.0262 | (-0.0262, -0.0262) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | 0.1111 | (-0.2556, 0.4278) | 0.2537 | 0.1111 | (0.1111, 0.1111) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | 0.1750 | (-0.0583, 0.3500) | 0.0993 | 0.1750 | (0.1750, 0.1750) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | bertscore_f1 | -0.0167 | (-0.0945, 0.0746) | 0.6773 | -0.0167 | (-0.0167, -0.0167) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | 0.0341 | (-0.0370, 0.1114) | 0.1727 | 0.0341 | (0.0341, 0.0341) | 0.0000 |
| controlled_vs_proposed_raw | context_relevance | 0.0926 | (-0.0389, 0.2249) | 0.0900 | 0.0926 | (0.0926, 0.0926) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.2066 | (0.1085, 0.3010) | 0.0000 | 0.2066 | (0.2066, 0.2066) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0440 | (-0.0126, 0.0973) | 0.0650 | 0.0440 | (0.0440, 0.0440) | 0.0000 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.1212 | (-0.0455, 0.3182) | 0.0837 | 0.1212 | (0.1212, 0.1212) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0259 | (0.0044, 0.0579) | 0.0003 | 0.0259 | (0.0259, 0.0259) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2619 | (0.1429, 0.3810) | 0.0000 | 0.2619 | (0.2619, 0.2619) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | -0.0146 | (-0.0563, 0.0125) | 0.7370 | -0.0146 | (-0.0146, -0.0146) | 1.0000 |
| controlled_vs_proposed_raw | distinct1 | 0.0169 | (-0.0175, 0.0527) | 0.1790 | 0.0169 | (0.0169, 0.0169) | 0.0000 |
| controlled_vs_proposed_raw | length_score | 0.1278 | (-0.0778, 0.3389) | 0.1147 | 0.1278 | (0.1278, 0.1278) | 0.0000 |
| controlled_vs_proposed_raw | sentence_score | 0.1167 | (0.0000, 0.2333) | 0.0863 | 0.1167 | (0.1167, 0.1167) | 0.0000 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0074 | (-0.0600, 0.0522) | 0.3470 | 0.0074 | (0.0074, 0.0074) | 0.0000 |
| controlled_vs_proposed_raw | overall_quality | 0.1152 | (0.0467, 0.1839) | 0.0000 | 0.1152 | (0.1152, 0.1152) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2265 | (0.1789, 0.2832) | 0.0000 | 0.2265 | (0.2265, 0.2265) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.2058 | (0.1486, 0.2667) | 0.0000 | 0.2058 | (0.2058, 0.2058) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.0695 | (0.0003, 0.1392) | 0.0233 | 0.0695 | (0.0695, 0.0695) | 0.0000 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3030 | (0.2424, 0.3640) | 0.0000 | 0.3030 | (0.3030, 0.3030) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0478 | (0.0240, 0.0724) | 0.0000 | 0.0478 | (0.0478, 0.0478) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2619 | (0.1905, 0.3333) | 0.0000 | 0.2619 | (0.2619, 0.2619) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | -0.0188 | (-0.0563, 0.0000) | 1.0000 | -0.0188 | (-0.0188, -0.0188) | 1.0000 |
| controlled_vs_candidate_no_context | distinct1 | 0.0162 | (-0.0194, 0.0536) | 0.1773 | 0.0162 | (0.0162, 0.0162) | 0.0000 |
| controlled_vs_candidate_no_context | length_score | 0.2278 | (-0.0111, 0.4889) | 0.0280 | 0.2278 | (0.2278, 0.2278) | 0.0000 |
| controlled_vs_candidate_no_context | sentence_score | 0.1750 | (0.0583, 0.2917) | 0.0173 | 0.1750 | (0.1750, 0.1750) | 0.0000 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0192 | (-0.0095, 0.0482) | 0.1090 | 0.0192 | (0.0192, 0.0192) | 0.0000 |
| controlled_vs_candidate_no_context | overall_quality | 0.1757 | (0.1333, 0.2192) | 0.0000 | 0.1757 | (0.1757, 0.1757) | 0.0000 |
| controlled_vs_baseline_no_context | context_relevance | 0.1834 | (0.1438, 0.2283) | 0.0000 | 0.1834 | (0.1834, 0.1834) | 0.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1867 | (0.0877, 0.2857) | 0.0000 | 0.1867 | (0.1867, 0.1867) | 0.0000 |
| controlled_vs_baseline_no_context | naturalness | 0.0665 | (0.0168, 0.1129) | 0.0047 | 0.0665 | (0.0665, 0.0665) | 0.0000 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.2424 | (0.1970, 0.3030) | 0.0000 | 0.2424 | (0.2424, 0.2424) | 0.0000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0456 | (0.0222, 0.0628) | 0.0007 | 0.0456 | (0.0456, 0.0456) | 0.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2381 | (0.1190, 0.3571) | 0.0000 | 0.2381 | (0.2381, 0.2381) | 0.0000 |
| controlled_vs_baseline_no_context | persona_style | -0.0188 | (-0.0563, 0.0000) | 1.0000 | -0.0188 | (-0.0188, -0.0188) | 1.0000 |
| controlled_vs_baseline_no_context | distinct1 | -0.0427 | (-0.0577, -0.0232) | 1.0000 | -0.0427 | (-0.0427, -0.0427) | 1.0000 |
| controlled_vs_baseline_no_context | length_score | 0.2722 | (0.0276, 0.5111) | 0.0183 | 0.2722 | (0.2722, 0.2722) | 0.0000 |
| controlled_vs_baseline_no_context | sentence_score | 0.2917 | (0.1750, 0.3500) | 0.0000 | 0.2917 | (0.2917, 0.2917) | 0.0000 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0133 | (-0.0129, 0.0399) | 0.1617 | 0.0133 | (0.0133, 0.0133) | 0.0000 |
| controlled_vs_baseline_no_context | overall_quality | 0.1503 | (0.1090, 0.1881) | 0.0000 | 0.1503 | (0.1503, 0.1503) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.1836 | (0.1099, 0.2453) | 0.0000 | 0.1836 | (0.1836, 0.1836) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1884 | (0.0986, 0.2683) | 0.0000 | 0.1884 | (0.1884, 0.1884) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | 0.0732 | (0.0135, 0.1209) | 0.0090 | 0.0732 | (0.0732, 0.0732) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.2424 | (0.1364, 0.3333) | 0.0000 | 0.2424 | (0.2424, 0.2424) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0463 | (0.0292, 0.0617) | 0.0000 | 0.0463 | (0.0463, 0.0463) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.2381 | (0.1190, 0.3333) | 0.0000 | 0.2381 | (0.2381, 0.2381) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | -0.0104 | (-0.0563, 0.0250) | 0.7277 | -0.0104 | (-0.0104, -0.0104) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0093 | (-0.0328, 0.0108) | 0.7910 | -0.0093 | (-0.0093, -0.0093) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | 0.2389 | (0.0222, 0.4500) | 0.0187 | 0.2389 | (0.2389, 0.2389) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.2917 | (0.1750, 0.3500) | 0.0000 | 0.2917 | (0.2917, 0.2917) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | -0.0093 | (-0.0464, 0.0225) | 0.6607 | -0.0093 | (-0.0093, -0.0093) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1493 | (0.1153, 0.1790) | 0.0000 | 0.1493 | (0.1493, 0.1493) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0319 | (-0.0751, -0.0020) | 0.9990 | -0.0319 | (-0.0319, -0.0319) | 1.0000 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0274 | (-0.1524, 0.0869) | 0.6810 | -0.0274 | (-0.0274, -0.0274) | 1.0000 |
| controlled_alt_vs_controlled_default | naturalness | -0.0342 | (-0.1064, 0.0400) | 0.8073 | -0.0342 | (-0.0342, -0.0342) | 1.0000 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0455 | (-0.1061, 0.0000) | 1.0000 | -0.0455 | (-0.0455, -0.0455) | 1.0000 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0001 | (-0.0159, 0.0233) | 0.5603 | -0.0001 | (-0.0001, -0.0001) | 1.0000 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0238 | (-0.1667, 0.1190) | 0.6730 | -0.0238 | (-0.0238, -0.0238) | 1.0000 |
| controlled_alt_vs_controlled_default | persona_style | -0.0417 | (-0.1813, 0.0563) | 0.7627 | -0.0417 | (-0.0417, -0.0417) | 1.0000 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0293 | (-0.0635, 0.0132) | 0.9310 | -0.0293 | (-0.0293, -0.0293) | 1.0000 |
| controlled_alt_vs_controlled_default | length_score | -0.0833 | (-0.4667, 0.3058) | 0.6650 | -0.0833 | (-0.0833, -0.0833) | 1.0000 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0583 | (-0.1750, 0.0000) | 1.0000 | -0.0583 | (-0.0583, -0.0583) | 1.0000 |
| controlled_alt_vs_controlled_default | bertscore_f1 | 0.0063 | (-0.0450, 0.0499) | 0.4010 | 0.0063 | (0.0063, 0.0063) | 0.0000 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0272 | (-0.0653, 0.0119) | 0.9150 | -0.0272 | (-0.0272, -0.0272) | 1.0000 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.0608 | (-0.0423, 0.1636) | 0.1137 | 0.0608 | (0.0608, 0.0608) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1792 | (-0.0026, 0.3117) | 0.0260 | 0.1792 | (0.1792, 0.1792) | 0.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0098 | (-0.1223, 0.1333) | 0.4357 | 0.0098 | (0.0098, 0.0098) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.0758 | (-0.0606, 0.2121) | 0.1390 | 0.0758 | (0.0758, 0.0758) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0258 | (0.0004, 0.0544) | 0.0183 | 0.0258 | (0.0258, 0.0258) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.2381 | (0.0238, 0.4048) | 0.0243 | 0.2381 | (0.2381, 0.2381) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_style | -0.0563 | (-0.1813, 0.0125) | 0.7523 | -0.0563 | (-0.0563, -0.0563) | 1.0000 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0124 | (-0.0450, 0.0180) | 0.7733 | -0.0124 | (-0.0124, -0.0124) | 1.0000 |
| controlled_alt_vs_proposed_raw | length_score | 0.0444 | (-0.5222, 0.5667) | 0.4213 | 0.0444 | (0.0444, 0.0444) | 0.0000 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0583 | (-0.1167, 0.2333) | 0.3710 | 0.0583 | (0.0583, 0.0583) | 0.0000 |
| controlled_alt_vs_proposed_raw | bertscore_f1 | 0.0137 | (-0.0266, 0.0498) | 0.2320 | 0.0137 | (0.0137, 0.0137) | 0.0000 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.0880 | (-0.0171, 0.1921) | 0.0583 | 0.0880 | (0.0880, 0.0880) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.1946 | (0.1676, 0.2116) | 0.0000 | 0.1946 | (0.1946, 0.1946) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1784 | (0.0260, 0.3117) | 0.0133 | 0.1784 | (0.1784, 0.1784) | 0.0000 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0353 | (-0.0966, 0.1618) | 0.3013 | 0.0353 | (0.0353, 0.0353) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.2576 | (0.2273, 0.2727) | 0.0000 | 0.2576 | (0.2576, 0.2576) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0477 | (0.0259, 0.0686) | 0.0000 | 0.0477 | (0.0477, 0.0477) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.2381 | (0.0476, 0.4048) | 0.0120 | 0.2381 | (0.2381, 0.2381) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_style | -0.0604 | (-0.1813, 0.0000) | 1.0000 | -0.0604 | (-0.0604, -0.0604) | 1.0000 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0131 | (-0.0478, 0.0176) | 0.7723 | -0.0131 | (-0.0131, -0.0131) | 1.0000 |
| controlled_alt_vs_candidate_no_context | length_score | 0.1444 | (-0.3778, 0.6500) | 0.3480 | 0.1444 | (0.1444, 0.1444) | 0.0000 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1167 | (-0.1167, 0.2917) | 0.2137 | 0.1167 | (0.1167, 0.1167) | 0.0000 |
| controlled_alt_vs_candidate_no_context | bertscore_f1 | 0.0255 | (-0.0394, 0.0669) | 0.2227 | 0.0255 | (0.0255, 0.0255) | 0.0000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1485 | (0.0786, 0.2140) | 0.0000 | 0.1485 | (0.1485, 0.1485) | 0.0000 |
| controlled_alt_vs_baseline_no_context | context_relevance | 0.1515 | (0.1189, 0.1838) | 0.0000 | 0.1515 | (0.1515, 0.1515) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 0.1593 | (0.0711, 0.2546) | 0.0003 | 0.1593 | (0.1593, 0.1593) | 0.0000 |
| controlled_alt_vs_baseline_no_context | naturalness | 0.0323 | (-0.0638, 0.1007) | 0.2443 | 0.0323 | (0.0323, 0.0323) | 0.0000 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 0.1970 | (0.1515, 0.2424) | 0.0000 | 0.1970 | (0.1970, 0.1970) | 0.0000 |
| controlled_alt_vs_baseline_no_context | context_overlap | 0.0455 | (0.0431, 0.0478) | 0.0000 | 0.0455 | (0.0455, 0.0455) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 0.2143 | (0.1190, 0.3333) | 0.0000 | 0.2143 | (0.2143, 0.2143) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_style | -0.0604 | (-0.1813, 0.0000) | 1.0000 | -0.0604 | (-0.0604, -0.0604) | 1.0000 |
| controlled_alt_vs_baseline_no_context | distinct1 | -0.0720 | (-0.0948, -0.0429) | 1.0000 | -0.0720 | (-0.0720, -0.0720) | 1.0000 |
| controlled_alt_vs_baseline_no_context | length_score | 0.1889 | (-0.2500, 0.5000) | 0.1823 | 0.1889 | (0.1889, 0.1889) | 0.0000 |
| controlled_alt_vs_baseline_no_context | sentence_score | 0.2333 | (0.1167, 0.3500) | 0.0020 | 0.2333 | (0.2333, 0.2333) | 0.0000 |
| controlled_alt_vs_baseline_no_context | bertscore_f1 | 0.0196 | (-0.0104, 0.0513) | 0.1130 | 0.0196 | (0.0196, 0.0196) | 0.0000 |
| controlled_alt_vs_baseline_no_context | overall_quality | 0.1231 | (0.0900, 0.1579) | 0.0000 | 0.1231 | (0.1231, 0.1231) | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_relevance | 0.1517 | (0.0916, 0.1983) | 0.0000 | 0.1517 | (0.1517, 0.1517) | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1610 | (0.0484, 0.2857) | 0.0000 | 0.1610 | (0.1610, 0.1610) | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | naturalness | 0.0390 | (-0.0197, 0.0925) | 0.1013 | 0.0390 | (0.0390, 0.0390) | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.1970 | (0.1061, 0.2576) | 0.0000 | 0.1970 | (0.1970, 0.1970) | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_overlap | 0.0462 | (0.0186, 0.0683) | 0.0003 | 0.0462 | (0.0462, 0.0462) | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.2143 | (0.0714, 0.3571) | 0.0023 | 0.2143 | (0.2143, 0.2143) | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_style | -0.0521 | (-0.1813, 0.0250) | 0.7490 | -0.0521 | (-0.0521, -0.0521) | 1.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | distinct1 | -0.0386 | (-0.0822, 0.0137) | 0.9263 | -0.0386 | (-0.0386, -0.0386) | 1.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | length_score | 0.1556 | (-0.1389, 0.4333) | 0.1567 | 0.1556 | (0.1556, 0.1556) | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | sentence_score | 0.2333 | (0.1167, 0.3500) | 0.0010 | 0.2333 | (0.2333, 0.2333) | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | bertscore_f1 | -0.0030 | (-0.0833, 0.0578) | 0.5180 | -0.0030 | (-0.0030, -0.0030) | 1.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | overall_quality | 0.1221 | (0.0666, 0.1776) | 0.0000 | 0.1221 | (0.1221, 0.1221) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.1834 | (0.1449, 0.2275) | 0.0000 | 0.1834 | (0.1834, 0.1834) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1867 | (0.0877, 0.2857) | 0.0000 | 0.1867 | (0.1867, 0.1867) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 0.0665 | (0.0130, 0.1129) | 0.0037 | 0.0665 | (0.0665, 0.0665) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.2424 | (0.1970, 0.3030) | 0.0000 | 0.2424 | (0.2424, 0.2424) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0456 | (0.0199, 0.0630) | 0.0000 | 0.0456 | (0.0456, 0.0456) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2381 | (0.1190, 0.3571) | 0.0000 | 0.2381 | (0.2381, 0.2381) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | -0.0188 | (-0.0563, 0.0000) | 1.0000 | -0.0188 | (-0.0188, -0.0188) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0427 | (-0.0580, -0.0232) | 1.0000 | -0.0427 | (-0.0427, -0.0427) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.2722 | (0.0333, 0.5111) | 0.0170 | 0.2722 | (0.2722, 0.2722) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.2917 | (0.1750, 0.3500) | 0.0000 | 0.2917 | (0.2917, 0.2917) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0133 | (-0.0129, 0.0392) | 0.1627 | 0.0133 | (0.0133, 0.0133) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1503 | (0.1125, 0.1881) | 0.0000 | 0.1503 | (0.1503, 0.1503) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.1836 | (0.1084, 0.2456) | 0.0000 | 0.1836 | (0.1836, 0.1836) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1884 | (0.0969, 0.2683) | 0.0000 | 0.1884 | (0.1884, 0.1884) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | 0.0732 | (0.0160, 0.1184) | 0.0067 | 0.0732 | (0.0732, 0.0732) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.2424 | (0.1364, 0.3333) | 0.0000 | 0.2424 | (0.2424, 0.2424) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0463 | (0.0289, 0.0617) | 0.0000 | 0.0463 | (0.0463, 0.0463) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.2381 | (0.1190, 0.3333) | 0.0000 | 0.2381 | (0.2381, 0.2381) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | -0.0104 | (-0.0563, 0.0250) | 0.7300 | -0.0104 | (-0.0104, -0.0104) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0093 | (-0.0321, 0.0109) | 0.7847 | -0.0093 | (-0.0093, -0.0093) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | 0.2389 | (0.0111, 0.4444) | 0.0207 | 0.2389 | (0.2389, 0.2389) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.2917 | (0.1750, 0.3500) | 0.0000 | 0.2917 | (0.2917, 0.2917) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | -0.0093 | (-0.0477, 0.0224) | 0.6753 | -0.0093 | (-0.0093, -0.0093) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1493 | (0.1174, 0.1790) | 0.0000 | 0.1493 | (0.1493, 0.1493) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 4 | 0 | 2 | 0.8333 | 1.0000 |
| proposed_vs_candidate_no_context | persona_consistency | 1 | 1 | 4 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | naturalness | 2 | 2 | 2 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 4 | 0 | 2 | 0.8333 | 1.0000 |
| proposed_vs_candidate_no_context | context_overlap | 3 | 1 | 2 | 0.6667 | 0.7500 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 1 | 1 | 4 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | persona_style | 0 | 1 | 5 | 0.4167 | 0.0000 |
| proposed_vs_candidate_no_context | distinct1 | 2 | 2 | 2 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | length_score | 2 | 2 | 2 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | sentence_score | 1 | 0 | 5 | 0.5833 | 1.0000 |
| proposed_vs_candidate_no_context | bertscore_f1 | 3 | 1 | 2 | 0.6667 | 0.7500 |
| proposed_vs_candidate_no_context | overall_quality | 4 | 0 | 2 | 0.8333 | 1.0000 |
| proposed_vs_baseline_no_context | context_relevance | 4 | 2 | 0 | 0.6667 | 0.6667 |
| proposed_vs_baseline_no_context | persona_consistency | 1 | 3 | 2 | 0.3333 | 0.2500 |
| proposed_vs_baseline_no_context | naturalness | 4 | 2 | 0 | 0.6667 | 0.6667 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 4 | 2 | 0 | 0.6667 | 0.6667 |
| proposed_vs_baseline_no_context | context_overlap | 4 | 2 | 0 | 0.6667 | 0.6667 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 1 | 3 | 2 | 0.3333 | 0.2500 |
| proposed_vs_baseline_no_context | persona_style | 0 | 1 | 5 | 0.4167 | 0.0000 |
| proposed_vs_baseline_no_context | distinct1 | 0 | 6 | 0 | 0.0000 | 0.0000 |
| proposed_vs_baseline_no_context | length_score | 3 | 2 | 1 | 0.5833 | 0.6000 |
| proposed_vs_baseline_no_context | sentence_score | 3 | 0 | 3 | 0.7500 | 1.0000 |
| proposed_vs_baseline_no_context | bertscore_f1 | 2 | 4 | 0 | 0.3333 | 0.3333 |
| proposed_vs_baseline_no_context | overall_quality | 4 | 2 | 0 | 0.6667 | 0.6667 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 5 | 1 | 0 | 0.8333 | 0.8333 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | 1 | 3 | 2 | 0.3333 | 0.2500 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | 4 | 2 | 0 | 0.6667 | 0.6667 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 4 | 1 | 1 | 0.7500 | 0.8000 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 4 | 2 | 0 | 0.6667 | 0.6667 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 1 | 3 | 2 | 0.3333 | 0.2500 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | 1 | 1 | 4 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | 2 | 4 | 0 | 0.3333 | 0.3333 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | 4 | 2 | 0 | 0.6667 | 0.6667 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | 4 | 1 | 1 | 0.7500 | 0.8000 |
| proposed_vs_baseline_no_context_phi3_latest | bertscore_f1 | 1 | 5 | 0 | 0.1667 | 0.1667 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | 4 | 2 | 0 | 0.6667 | 0.6667 |
| controlled_vs_proposed_raw | context_relevance | 4 | 2 | 0 | 0.6667 | 0.6667 |
| controlled_vs_proposed_raw | persona_consistency | 6 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_proposed_raw | naturalness | 4 | 2 | 0 | 0.6667 | 0.6667 |
| controlled_vs_proposed_raw | context_keyword_coverage | 4 | 2 | 0 | 0.6667 | 0.6667 |
| controlled_vs_proposed_raw | context_overlap | 5 | 1 | 0 | 0.8333 | 0.8333 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 5 | 0 | 1 | 0.9167 | 1.0000 |
| controlled_vs_proposed_raw | persona_style | 1 | 1 | 4 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | distinct1 | 3 | 3 | 0 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | length_score | 4 | 2 | 0 | 0.6667 | 0.6667 |
| controlled_vs_proposed_raw | sentence_score | 2 | 0 | 4 | 0.6667 | 1.0000 |
| controlled_vs_proposed_raw | bertscore_f1 | 5 | 1 | 0 | 0.8333 | 0.8333 |
| controlled_vs_proposed_raw | overall_quality | 5 | 1 | 0 | 0.8333 | 0.8333 |
| controlled_vs_candidate_no_context | context_relevance | 6 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 6 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | naturalness | 4 | 2 | 0 | 0.6667 | 0.6667 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 6 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 6 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 6 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_style | 0 | 1 | 5 | 0.4167 | 0.0000 |
| controlled_vs_candidate_no_context | distinct1 | 3 | 3 | 0 | 0.5000 | 0.5000 |
| controlled_vs_candidate_no_context | length_score | 4 | 2 | 0 | 0.6667 | 0.6667 |
| controlled_vs_candidate_no_context | sentence_score | 3 | 0 | 3 | 0.7500 | 1.0000 |
| controlled_vs_candidate_no_context | bertscore_f1 | 4 | 2 | 0 | 0.6667 | 0.6667 |
| controlled_vs_candidate_no_context | overall_quality | 6 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | context_relevance | 6 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 5 | 0 | 1 | 0.9167 | 1.0000 |
| controlled_vs_baseline_no_context | naturalness | 5 | 1 | 0 | 0.8333 | 0.8333 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 6 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | context_overlap | 5 | 1 | 0 | 0.8333 | 0.8333 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 5 | 0 | 1 | 0.9167 | 1.0000 |
| controlled_vs_baseline_no_context | persona_style | 0 | 1 | 5 | 0.4167 | 0.0000 |
| controlled_vs_baseline_no_context | distinct1 | 0 | 5 | 1 | 0.0833 | 0.0000 |
| controlled_vs_baseline_no_context | length_score | 4 | 2 | 0 | 0.6667 | 0.6667 |
| controlled_vs_baseline_no_context | sentence_score | 5 | 0 | 1 | 0.9167 | 1.0000 |
| controlled_vs_baseline_no_context | bertscore_f1 | 4 | 2 | 0 | 0.6667 | 0.6667 |
| controlled_vs_baseline_no_context | overall_quality | 6 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 6 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 5 | 0 | 1 | 0.9167 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | 5 | 1 | 0 | 0.8333 | 0.8333 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 5 | 0 | 1 | 0.9167 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 6 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 5 | 0 | 1 | 0.9167 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | 1 | 1 | 4 | 0.5000 | 0.5000 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | 2 | 3 | 1 | 0.4167 | 0.4000 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | 4 | 2 | 0 | 0.6667 | 0.6667 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 5 | 0 | 1 | 0.9167 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 3 | 3 | 0 | 0.5000 | 0.5000 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 6 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 1 | 4 | 1 | 0.2500 | 0.2000 |
| controlled_alt_vs_controlled_default | persona_consistency | 3 | 2 | 1 | 0.5833 | 0.6000 |
| controlled_alt_vs_controlled_default | naturalness | 2 | 3 | 1 | 0.4167 | 0.4000 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0 | 2 | 4 | 0.3333 | 0.0000 |
| controlled_alt_vs_controlled_default | context_overlap | 2 | 3 | 1 | 0.4167 | 0.4000 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 3 | 2 | 1 | 0.5833 | 0.6000 |
| controlled_alt_vs_controlled_default | persona_style | 1 | 1 | 4 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | distinct1 | 1 | 4 | 1 | 0.2500 | 0.2000 |
| controlled_alt_vs_controlled_default | length_score | 2 | 3 | 1 | 0.4167 | 0.4000 |
| controlled_alt_vs_controlled_default | sentence_score | 0 | 1 | 5 | 0.4167 | 0.0000 |
| controlled_alt_vs_controlled_default | bertscore_f1 | 3 | 2 | 1 | 0.5833 | 0.6000 |
| controlled_alt_vs_controlled_default | overall_quality | 1 | 4 | 1 | 0.2500 | 0.2000 |
| controlled_alt_vs_proposed_raw | context_relevance | 4 | 2 | 0 | 0.6667 | 0.6667 |
| controlled_alt_vs_proposed_raw | persona_consistency | 5 | 1 | 0 | 0.8333 | 0.8333 |
| controlled_alt_vs_proposed_raw | naturalness | 4 | 2 | 0 | 0.6667 | 0.6667 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 4 | 2 | 0 | 0.6667 | 0.6667 |
| controlled_alt_vs_proposed_raw | context_overlap | 5 | 1 | 0 | 0.8333 | 0.8333 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 5 | 1 | 0 | 0.8333 | 0.8333 |
| controlled_alt_vs_proposed_raw | persona_style | 1 | 1 | 4 | 0.5000 | 0.5000 |
| controlled_alt_vs_proposed_raw | distinct1 | 2 | 2 | 2 | 0.5000 | 0.5000 |
| controlled_alt_vs_proposed_raw | length_score | 4 | 2 | 0 | 0.6667 | 0.6667 |
| controlled_alt_vs_proposed_raw | sentence_score | 2 | 1 | 3 | 0.5833 | 0.6667 |
| controlled_alt_vs_proposed_raw | bertscore_f1 | 4 | 2 | 0 | 0.6667 | 0.6667 |
| controlled_alt_vs_proposed_raw | overall_quality | 4 | 2 | 0 | 0.6667 | 0.6667 |
| controlled_alt_vs_candidate_no_context | context_relevance | 6 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 4 | 1 | 1 | 0.7500 | 0.8000 |
| controlled_alt_vs_candidate_no_context | naturalness | 4 | 2 | 0 | 0.6667 | 0.6667 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 6 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 6 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 4 | 1 | 1 | 0.7500 | 0.8000 |
| controlled_alt_vs_candidate_no_context | persona_style | 0 | 1 | 5 | 0.4167 | 0.0000 |
| controlled_alt_vs_candidate_no_context | distinct1 | 2 | 2 | 2 | 0.5000 | 0.5000 |
| controlled_alt_vs_candidate_no_context | length_score | 4 | 2 | 0 | 0.6667 | 0.6667 |
| controlled_alt_vs_candidate_no_context | sentence_score | 3 | 1 | 2 | 0.6667 | 0.7500 |
| controlled_alt_vs_candidate_no_context | bertscore_f1 | 5 | 1 | 0 | 0.8333 | 0.8333 |
| controlled_alt_vs_candidate_no_context | overall_quality | 6 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context | context_relevance | 6 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 5 | 0 | 1 | 0.9167 | 1.0000 |
| controlled_alt_vs_baseline_no_context | naturalness | 5 | 1 | 0 | 0.8333 | 0.8333 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 6 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context | context_overlap | 6 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 5 | 0 | 1 | 0.9167 | 1.0000 |
| controlled_alt_vs_baseline_no_context | persona_style | 0 | 1 | 5 | 0.4167 | 0.0000 |
| controlled_alt_vs_baseline_no_context | distinct1 | 0 | 5 | 1 | 0.0833 | 0.0000 |
| controlled_alt_vs_baseline_no_context | length_score | 5 | 1 | 0 | 0.8333 | 0.8333 |
| controlled_alt_vs_baseline_no_context | sentence_score | 4 | 0 | 2 | 0.8333 | 1.0000 |
| controlled_alt_vs_baseline_no_context | bertscore_f1 | 5 | 1 | 0 | 0.8333 | 0.8333 |
| controlled_alt_vs_baseline_no_context | overall_quality | 6 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_relevance | 6 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_consistency | 5 | 0 | 1 | 0.9167 | 1.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | naturalness | 4 | 2 | 0 | 0.6667 | 0.6667 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 5 | 0 | 1 | 0.9167 | 1.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_overlap | 5 | 1 | 0 | 0.8333 | 0.8333 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 4 | 0 | 2 | 0.8333 | 1.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_style | 1 | 1 | 4 | 0.5000 | 0.5000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | distinct1 | 1 | 5 | 0 | 0.1667 | 0.1667 |
| controlled_alt_vs_baseline_no_context_phi3_latest | length_score | 4 | 2 | 0 | 0.6667 | 0.6667 |
| controlled_alt_vs_baseline_no_context_phi3_latest | sentence_score | 4 | 0 | 2 | 0.8333 | 1.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | bertscore_f1 | 4 | 2 | 0 | 0.6667 | 0.6667 |
| controlled_alt_vs_baseline_no_context_phi3_latest | overall_quality | 6 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 6 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 5 | 0 | 1 | 0.9167 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 5 | 1 | 0 | 0.8333 | 0.8333 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 6 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 5 | 1 | 0 | 0.8333 | 0.8333 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 5 | 0 | 1 | 0.9167 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0 | 1 | 5 | 0.4167 | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | 0 | 5 | 1 | 0.0833 | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 4 | 2 | 0 | 0.6667 | 0.6667 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 5 | 0 | 1 | 0.9167 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 4 | 2 | 0 | 0.6667 | 0.6667 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 6 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 6 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 5 | 0 | 1 | 0.9167 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | 5 | 1 | 0 | 0.8333 | 0.8333 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 5 | 0 | 1 | 0.9167 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 6 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 5 | 0 | 1 | 0.9167 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | 1 | 1 | 4 | 0.5000 | 0.5000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | 2 | 3 | 1 | 0.4167 | 0.4000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | 4 | 2 | 0 | 0.6667 | 0.6667 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | 5 | 0 | 1 | 0.9167 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 3 | 3 | 0 | 0.5000 | 0.5000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 6 | 0 | 0 | 1.0000 | 1.0000 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.5000 | 0.1667 | 0.8333 |
| proposed_contextual_controlled_runtime | 0.0000 | 0.0000 | 0.5000 | 0.3333 | 0.6667 |
| proposed_contextual | 0.0000 | 0.0000 | 0.3333 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 |
| baseline_no_context | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline_no_context_phi3_latest | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `1`
- Unique template signatures: `4`
- Template signature ratio: `0.6667`
- Effective sample size by source clustering: `1.00`
- Effective sample size by template-signature clustering: `3.60`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: enabled.

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
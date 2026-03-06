# Proposal Alignment Evaluation Report

- Run ID: `20260306T120829Z`
- Generated: `2026-03-06T12:09:58.978490+00:00`
- Scenarios: `artifacts\proposal_locked_blend_full112_nogate\20260306T120829Z\scenarios.jsonl`
- Scenario count: `112`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_alt`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off
- `baseline_no_context`: model `phi3:mini`, dynamic_context=off, response_control=off
- `baseline_no_context_phi3_latest`: model `phi3:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2653 (0.2492, 0.2825) | 0.3780 (0.3481, 0.4080) | 0.8668 (0.8511, 0.8816) | 0.3899 (0.3793, 0.4013) | 0.0803 |
| proposed_contextual_controlled_alt | 0.2647 (0.2436, 0.2852) | 0.3277 (0.3037, 0.3526) | 0.8755 (0.8621, 0.8896) | 0.3754 (0.3644, 0.3859) | 0.0883 |
| proposed_contextual | 0.1056 (0.0847, 0.1289) | 0.1599 (0.1378, 0.1851) | 0.8077 (0.7948, 0.8208) | 0.2423 (0.2261, 0.2594) | 0.0724 |
| candidate_no_context | 0.0271 (0.0208, 0.0337) | 0.1804 (0.1502, 0.2108) | 0.8095 (0.7952, 0.8244) | 0.2141 (0.2027, 0.2259) | 0.0415 |
| baseline_no_context | 0.0471 (0.0377, 0.0571) | 0.1770 (0.1557, 0.1981) | 0.8748 (0.8633, 0.8856) | 0.2332 (0.2244, 0.2421) | 0.0502 |
| baseline_no_context_phi3_latest | 0.0453 (0.0360, 0.0547) | 0.1808 (0.1590, 0.2031) | 0.8739 (0.8628, 0.8854) | 0.2348 (0.2265, 0.2433) | 0.0614 |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0785 | 2.8981 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0206 | -0.1139 |
| proposed_vs_candidate_no_context | naturalness | -0.0018 | -0.0022 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.1023 | 4.7656 |
| proposed_vs_candidate_no_context | context_overlap | 0.0229 | 0.5691 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0257 | -0.2666 |
| proposed_vs_candidate_no_context | persona_style | 0.0001 | 0.0002 |
| proposed_vs_candidate_no_context | distinct1 | 0.0012 | 0.0013 |
| proposed_vs_candidate_no_context | length_score | -0.0107 | -0.0376 |
| proposed_vs_candidate_no_context | sentence_score | 0.0004 | 0.0006 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0308 | 0.7428 |
| proposed_vs_candidate_no_context | overall_quality | 0.0283 | 0.1321 |
| proposed_vs_baseline_no_context | context_relevance | 0.0585 | 1.2410 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0171 | -0.0967 |
| proposed_vs_baseline_no_context | naturalness | -0.0671 | -0.0767 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0732 | 1.4485 |
| proposed_vs_baseline_no_context | context_overlap | 0.0240 | 0.6142 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0038 | -0.0508 |
| proposed_vs_baseline_no_context | persona_style | -0.0705 | -0.1201 |
| proposed_vs_baseline_no_context | distinct1 | -0.0357 | -0.0367 |
| proposed_vs_baseline_no_context | length_score | -0.2214 | -0.4468 |
| proposed_vs_baseline_no_context | sentence_score | -0.0839 | -0.0973 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0222 | 0.4421 |
| proposed_vs_baseline_no_context | overall_quality | 0.0092 | 0.0393 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 0.0603 | 1.3325 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | -0.0210 | -0.1160 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | -0.0662 | -0.0758 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.0771 | 1.6506 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 0.0211 | 0.5049 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | -0.0099 | -0.1233 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | -0.0651 | -0.1120 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | -0.0428 | -0.0437 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | -0.2060 | -0.4290 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | -0.0777 | -0.0907 |
| proposed_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0110 | 0.1785 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | 0.0075 | 0.0319 |
| controlled_vs_proposed_raw | context_relevance | 0.1598 | 1.5136 |
| controlled_vs_proposed_raw | persona_consistency | 0.2182 | 1.3650 |
| controlled_vs_proposed_raw | naturalness | 0.0591 | 0.0732 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2103 | 1.6993 |
| controlled_vs_proposed_raw | context_overlap | 0.0417 | 0.6622 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2466 | 3.4856 |
| controlled_vs_proposed_raw | persona_style | 0.1046 | 0.2026 |
| controlled_vs_proposed_raw | distinct1 | 0.0003 | 0.0003 |
| controlled_vs_proposed_raw | length_score | 0.2354 | 0.8588 |
| controlled_vs_proposed_raw | sentence_score | 0.1214 | 0.1560 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0080 | 0.1104 |
| controlled_vs_proposed_raw | overall_quality | 0.1476 | 0.6090 |
| controlled_vs_candidate_no_context | context_relevance | 0.2382 | 8.7982 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1976 | 1.0955 |
| controlled_vs_candidate_no_context | naturalness | 0.0573 | 0.0708 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3127 | 14.5633 |
| controlled_vs_candidate_no_context | context_overlap | 0.0646 | 1.6081 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2209 | 2.2896 |
| controlled_vs_candidate_no_context | persona_style | 0.1047 | 0.2028 |
| controlled_vs_candidate_no_context | distinct1 | 0.0015 | 0.0016 |
| controlled_vs_candidate_no_context | length_score | 0.2247 | 0.7889 |
| controlled_vs_candidate_no_context | sentence_score | 0.1219 | 0.1566 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0388 | 0.9352 |
| controlled_vs_candidate_no_context | overall_quality | 0.1759 | 0.8216 |
| controlled_vs_baseline_no_context | context_relevance | 0.2182 | 4.6330 |
| controlled_vs_baseline_no_context | persona_consistency | 0.2011 | 1.1362 |
| controlled_vs_baseline_no_context | naturalness | -0.0080 | -0.0092 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.2836 | 5.6093 |
| controlled_vs_baseline_no_context | context_overlap | 0.0657 | 1.6831 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2428 | 3.2578 |
| controlled_vs_baseline_no_context | persona_style | 0.0341 | 0.0582 |
| controlled_vs_baseline_no_context | distinct1 | -0.0354 | -0.0364 |
| controlled_vs_baseline_no_context | length_score | 0.0140 | 0.0282 |
| controlled_vs_baseline_no_context | sentence_score | 0.0375 | 0.0435 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0302 | 0.6013 |
| controlled_vs_baseline_no_context | overall_quality | 0.1568 | 0.6723 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.2201 | 4.8629 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1972 | 1.0905 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | -0.0071 | -0.0082 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.2874 | 6.1550 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0629 | 1.5014 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.2366 | 2.9326 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | 0.0395 | 0.0679 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0425 | -0.0434 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | 0.0295 | 0.0614 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.0437 | 0.0511 |
| controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0189 | 0.3086 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1551 | 0.6603 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0006 | -0.0022 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0503 | -0.1332 |
| controlled_alt_vs_controlled_default | naturalness | 0.0087 | 0.0101 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0022 | -0.0065 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0032 | 0.0301 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0618 | -0.1948 |
| controlled_alt_vs_controlled_default | persona_style | -0.0044 | -0.0071 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0009 | 0.0009 |
| controlled_alt_vs_controlled_default | length_score | 0.0390 | 0.0765 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0031 | 0.0035 |
| controlled_alt_vs_controlled_default | bertscore_f1 | 0.0080 | 0.0993 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0145 | -0.0372 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1592 | 1.5081 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1679 | 1.0501 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0678 | 0.0840 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2082 | 1.6817 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0449 | 0.7122 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1848 | 2.6118 |
| controlled_alt_vs_proposed_raw | persona_style | 0.1002 | 0.1940 |
| controlled_alt_vs_proposed_raw | distinct1 | 0.0011 | 0.0012 |
| controlled_alt_vs_proposed_raw | length_score | 0.2744 | 1.0011 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1246 | 0.1600 |
| controlled_alt_vs_proposed_raw | bertscore_f1 | 0.0160 | 0.2207 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1331 | 0.5492 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2377 | 8.7768 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1473 | 0.8165 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0661 | 0.0816 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3105 | 14.4619 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0677 | 1.6866 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1591 | 1.6487 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.1003 | 0.1943 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0023 | 0.0025 |
| controlled_alt_vs_candidate_no_context | length_score | 0.2637 | 0.9258 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1250 | 0.1606 |
| controlled_alt_vs_candidate_no_context | bertscore_f1 | 0.0468 | 1.1275 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1614 | 0.7538 |
| controlled_alt_vs_baseline_no_context | context_relevance | 0.2176 | 4.6207 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 0.1507 | 0.8517 |
| controlled_alt_vs_baseline_no_context | naturalness | 0.0007 | 0.0008 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 0.2814 | 5.5662 |
| controlled_alt_vs_baseline_no_context | context_overlap | 0.0689 | 1.7639 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 0.1810 | 2.4284 |
| controlled_alt_vs_baseline_no_context | persona_style | 0.0297 | 0.0506 |
| controlled_alt_vs_baseline_no_context | distinct1 | -0.0346 | -0.0355 |
| controlled_alt_vs_baseline_no_context | length_score | 0.0530 | 0.1069 |
| controlled_alt_vs_baseline_no_context | sentence_score | 0.0406 | 0.0471 |
| controlled_alt_vs_baseline_no_context | bertscore_f1 | 0.0381 | 0.7604 |
| controlled_alt_vs_baseline_no_context | overall_quality | 0.1423 | 0.6101 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_relevance | 0.2195 | 4.8502 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1469 | 0.8122 |
| controlled_alt_vs_baseline_no_context_phi3_latest | naturalness | 0.0016 | 0.0018 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.2853 | 6.1083 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_overlap | 0.0660 | 1.5766 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1748 | 2.1665 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_style | 0.0350 | 0.0603 |
| controlled_alt_vs_baseline_no_context_phi3_latest | distinct1 | -0.0417 | -0.0425 |
| controlled_alt_vs_baseline_no_context_phi3_latest | length_score | 0.0685 | 0.1426 |
| controlled_alt_vs_baseline_no_context_phi3_latest | sentence_score | 0.0469 | 0.0547 |
| controlled_alt_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0269 | 0.4385 |
| controlled_alt_vs_baseline_no_context_phi3_latest | overall_quality | 0.1406 | 0.5986 |
| proposed_contextual_controlled_alt_vs_baseline_no_context | context_relevance | 0.2176 | 4.6207 |
| proposed_contextual_controlled_alt_vs_baseline_no_context | persona_consistency | 0.1507 | 0.8517 |
| proposed_contextual_controlled_alt_vs_baseline_no_context | naturalness | 0.0007 | 0.0008 |
| proposed_contextual_controlled_alt_vs_baseline_no_context | context_keyword_coverage | 0.2814 | 5.5662 |
| proposed_contextual_controlled_alt_vs_baseline_no_context | context_overlap | 0.0689 | 1.7639 |
| proposed_contextual_controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 0.1810 | 2.4284 |
| proposed_contextual_controlled_alt_vs_baseline_no_context | persona_style | 0.0297 | 0.0506 |
| proposed_contextual_controlled_alt_vs_baseline_no_context | distinct1 | -0.0346 | -0.0355 |
| proposed_contextual_controlled_alt_vs_baseline_no_context | length_score | 0.0530 | 0.1069 |
| proposed_contextual_controlled_alt_vs_baseline_no_context | sentence_score | 0.0406 | 0.0471 |
| proposed_contextual_controlled_alt_vs_baseline_no_context | bertscore_f1 | 0.0381 | 0.7604 |
| proposed_contextual_controlled_alt_vs_baseline_no_context | overall_quality | 0.1423 | 0.6101 |
| proposed_contextual_controlled_alt_vs_baseline_no_context_phi3_latest | context_relevance | 0.2195 | 4.8502 |
| proposed_contextual_controlled_alt_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1469 | 0.8122 |
| proposed_contextual_controlled_alt_vs_baseline_no_context_phi3_latest | naturalness | 0.0016 | 0.0018 |
| proposed_contextual_controlled_alt_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.2853 | 6.1083 |
| proposed_contextual_controlled_alt_vs_baseline_no_context_phi3_latest | context_overlap | 0.0660 | 1.5766 |
| proposed_contextual_controlled_alt_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1748 | 2.1665 |
| proposed_contextual_controlled_alt_vs_baseline_no_context_phi3_latest | persona_style | 0.0350 | 0.0603 |
| proposed_contextual_controlled_alt_vs_baseline_no_context_phi3_latest | distinct1 | -0.0417 | -0.0425 |
| proposed_contextual_controlled_alt_vs_baseline_no_context_phi3_latest | length_score | 0.0685 | 0.1426 |
| proposed_contextual_controlled_alt_vs_baseline_no_context_phi3_latest | sentence_score | 0.0469 | 0.0547 |
| proposed_contextual_controlled_alt_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0269 | 0.4385 |
| proposed_contextual_controlled_alt_vs_baseline_no_context_phi3_latest | overall_quality | 0.1406 | 0.5986 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0785 | (0.0571, 0.1015) | 0.0000 | 0.0785 | (0.0458, 0.1102) | 0.0000 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0206 | (-0.0546, 0.0107) | 0.9050 | -0.0206 | (-0.0842, 0.0190) | 0.6810 |
| proposed_vs_candidate_no_context | naturalness | -0.0018 | (-0.0168, 0.0136) | 0.5867 | -0.0018 | (-0.0107, 0.0074) | 0.6450 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.1023 | (0.0734, 0.1338) | 0.0000 | 0.1023 | (0.0593, 0.1459) | 0.0000 |
| proposed_vs_candidate_no_context | context_overlap | 0.0229 | (0.0157, 0.0309) | 0.0000 | 0.0229 | (0.0138, 0.0316) | 0.0000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0257 | (-0.0624, 0.0103) | 0.9113 | -0.0257 | (-0.0972, 0.0196) | 0.7267 |
| proposed_vs_candidate_no_context | persona_style | 0.0001 | (-0.0286, 0.0281) | 0.4990 | 0.0001 | (-0.0415, 0.0319) | 0.4923 |
| proposed_vs_candidate_no_context | distinct1 | 0.0012 | (-0.0054, 0.0077) | 0.3640 | 0.0012 | (-0.0059, 0.0078) | 0.3507 |
| proposed_vs_candidate_no_context | length_score | -0.0107 | (-0.0717, 0.0527) | 0.6100 | -0.0107 | (-0.0423, 0.0167) | 0.7450 |
| proposed_vs_candidate_no_context | sentence_score | 0.0004 | (-0.0362, 0.0375) | 0.4997 | 0.0004 | (-0.0375, 0.0379) | 0.4977 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0308 | (0.0199, 0.0417) | 0.0000 | 0.0308 | (0.0160, 0.0480) | 0.0000 |
| proposed_vs_candidate_no_context | overall_quality | 0.0283 | (0.0094, 0.0458) | 0.0010 | 0.0283 | (-0.0014, 0.0541) | 0.0303 |
| proposed_vs_baseline_no_context | context_relevance | 0.0585 | (0.0336, 0.0843) | 0.0000 | 0.0585 | (0.0162, 0.0990) | 0.0027 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0171 | (-0.0501, 0.0145) | 0.8487 | -0.0171 | (-0.0521, 0.0195) | 0.8157 |
| proposed_vs_baseline_no_context | naturalness | -0.0671 | (-0.0840, -0.0504) | 1.0000 | -0.0671 | (-0.0944, -0.0406) | 1.0000 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0732 | (0.0412, 0.1069) | 0.0000 | 0.0732 | (0.0162, 0.1305) | 0.0060 |
| proposed_vs_baseline_no_context | context_overlap | 0.0240 | (0.0160, 0.0317) | 0.0000 | 0.0240 | (0.0145, 0.0326) | 0.0000 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0038 | (-0.0419, 0.0331) | 0.5817 | -0.0038 | (-0.0382, 0.0310) | 0.5877 |
| proposed_vs_baseline_no_context | persona_style | -0.0705 | (-0.1130, -0.0330) | 1.0000 | -0.0705 | (-0.1918, 0.0128) | 0.9143 |
| proposed_vs_baseline_no_context | distinct1 | -0.0357 | (-0.0439, -0.0271) | 1.0000 | -0.0357 | (-0.0470, -0.0233) | 1.0000 |
| proposed_vs_baseline_no_context | length_score | -0.2214 | (-0.2905, -0.1506) | 1.0000 | -0.2214 | (-0.3232, -0.1283) | 1.0000 |
| proposed_vs_baseline_no_context | sentence_score | -0.0839 | (-0.1246, -0.0406) | 0.9997 | -0.0839 | (-0.1589, -0.0031) | 0.9817 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0222 | (0.0095, 0.0359) | 0.0010 | 0.0222 | (0.0003, 0.0461) | 0.0243 |
| proposed_vs_baseline_no_context | overall_quality | 0.0092 | (-0.0103, 0.0290) | 0.1717 | 0.0092 | (-0.0194, 0.0373) | 0.2587 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 0.0603 | (0.0353, 0.0871) | 0.0000 | 0.0603 | (0.0210, 0.0997) | 0.0010 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | -0.0210 | (-0.0541, 0.0109) | 0.9037 | -0.0210 | (-0.0596, 0.0127) | 0.8740 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | -0.0662 | (-0.0840, -0.0476) | 1.0000 | -0.0662 | (-0.1050, -0.0258) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.0771 | (0.0450, 0.1131) | 0.0000 | 0.0771 | (0.0237, 0.1321) | 0.0023 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 0.0211 | (0.0133, 0.0291) | 0.0000 | 0.0211 | (0.0119, 0.0303) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | -0.0099 | (-0.0497, 0.0292) | 0.6823 | -0.0099 | (-0.0469, 0.0225) | 0.6950 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | -0.0651 | (-0.1073, -0.0282) | 1.0000 | -0.0651 | (-0.1846, 0.0087) | 0.9137 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | -0.0428 | (-0.0508, -0.0342) | 1.0000 | -0.0428 | (-0.0562, -0.0273) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | -0.2060 | (-0.2789, -0.1378) | 1.0000 | -0.2060 | (-0.3408, -0.0643) | 0.9990 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | -0.0777 | (-0.1241, -0.0312) | 0.9993 | -0.0777 | (-0.1437, -0.0156) | 0.9950 |
| proposed_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0110 | (-0.0008, 0.0229) | 0.0343 | 0.0110 | (-0.0076, 0.0366) | 0.2140 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | 0.0075 | (-0.0125, 0.0278) | 0.2353 | 0.0075 | (-0.0236, 0.0359) | 0.3290 |
| controlled_vs_proposed_raw | context_relevance | 0.1598 | (0.1294, 0.1892) | 0.0000 | 0.1598 | (0.1270, 0.1949) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.2182 | (0.1839, 0.2530) | 0.0000 | 0.2182 | (0.1717, 0.2662) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0591 | (0.0382, 0.0813) | 0.0000 | 0.0591 | (0.0133, 0.1028) | 0.0037 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2103 | (0.1717, 0.2506) | 0.0000 | 0.2103 | (0.1690, 0.2538) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0417 | (0.0315, 0.0519) | 0.0000 | 0.0417 | (0.0274, 0.0589) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2466 | (0.2050, 0.2893) | 0.0000 | 0.2466 | (0.1890, 0.3061) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.1046 | (0.0680, 0.1445) | 0.0000 | 0.1046 | (0.0208, 0.2260) | 0.0003 |
| controlled_vs_proposed_raw | distinct1 | 0.0003 | (-0.0093, 0.0094) | 0.4733 | 0.0003 | (-0.0140, 0.0106) | 0.4363 |
| controlled_vs_proposed_raw | length_score | 0.2354 | (0.1482, 0.3203) | 0.0000 | 0.2354 | (0.0562, 0.4161) | 0.0033 |
| controlled_vs_proposed_raw | sentence_score | 0.1214 | (0.0781, 0.1656) | 0.0000 | 0.1214 | (0.0433, 0.1907) | 0.0003 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0080 | (-0.0069, 0.0228) | 0.1443 | 0.0080 | (-0.0129, 0.0296) | 0.2263 |
| controlled_vs_proposed_raw | overall_quality | 0.1476 | (0.1265, 0.1684) | 0.0000 | 0.1476 | (0.1102, 0.1814) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2382 | (0.2189, 0.2577) | 0.0000 | 0.2382 | (0.2273, 0.2496) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1976 | (0.1557, 0.2364) | 0.0000 | 0.1976 | (0.1338, 0.2609) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.0573 | (0.0333, 0.0790) | 0.0000 | 0.0573 | (0.0127, 0.1017) | 0.0033 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3127 | (0.2879, 0.3384) | 0.0000 | 0.3127 | (0.2965, 0.3278) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0646 | (0.0569, 0.0729) | 0.0000 | 0.0646 | (0.0573, 0.0727) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2209 | (0.1743, 0.2688) | 0.0000 | 0.2209 | (0.1442, 0.2941) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.1047 | (0.0635, 0.1477) | 0.0000 | 0.1047 | (0.0128, 0.2230) | 0.0003 |
| controlled_vs_candidate_no_context | distinct1 | 0.0015 | (-0.0078, 0.0105) | 0.3717 | 0.0015 | (-0.0113, 0.0131) | 0.4003 |
| controlled_vs_candidate_no_context | length_score | 0.2247 | (0.1342, 0.3185) | 0.0000 | 0.2247 | (0.0571, 0.3929) | 0.0040 |
| controlled_vs_candidate_no_context | sentence_score | 0.1219 | (0.0813, 0.1656) | 0.0000 | 0.1219 | (0.0312, 0.2062) | 0.0047 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0388 | (0.0268, 0.0509) | 0.0000 | 0.0388 | (0.0215, 0.0583) | 0.0000 |
| controlled_vs_candidate_no_context | overall_quality | 0.1759 | (0.1596, 0.1922) | 0.0000 | 0.1759 | (0.1470, 0.2037) | 0.0000 |
| controlled_vs_baseline_no_context | context_relevance | 0.2182 | (0.1986, 0.2388) | 0.0000 | 0.2182 | (0.2000, 0.2402) | 0.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 0.2011 | (0.1666, 0.2371) | 0.0000 | 0.2011 | (0.1642, 0.2383) | 0.0000 |
| controlled_vs_baseline_no_context | naturalness | -0.0080 | (-0.0259, 0.0100) | 0.8107 | -0.0080 | (-0.0454, 0.0300) | 0.6583 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.2836 | (0.2568, 0.3129) | 0.0000 | 0.2836 | (0.2563, 0.3145) | 0.0000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0657 | (0.0585, 0.0731) | 0.0000 | 0.0657 | (0.0582, 0.0737) | 0.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2428 | (0.2007, 0.2879) | 0.0000 | 0.2428 | (0.1958, 0.2922) | 0.0000 |
| controlled_vs_baseline_no_context | persona_style | 0.0341 | (0.0088, 0.0597) | 0.0040 | 0.0341 | (0.0045, 0.0743) | 0.0027 |
| controlled_vs_baseline_no_context | distinct1 | -0.0354 | (-0.0435, -0.0271) | 1.0000 | -0.0354 | (-0.0415, -0.0299) | 1.0000 |
| controlled_vs_baseline_no_context | length_score | 0.0140 | (-0.0649, 0.0911) | 0.3587 | 0.0140 | (-0.1494, 0.1807) | 0.4317 |
| controlled_vs_baseline_no_context | sentence_score | 0.0375 | (-0.0062, 0.0813) | 0.0560 | 0.0375 | (-0.0750, 0.1469) | 0.2660 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0302 | (0.0170, 0.0437) | 0.0000 | 0.0302 | (0.0132, 0.0461) | 0.0000 |
| controlled_vs_baseline_no_context | overall_quality | 0.1568 | (0.1427, 0.1707) | 0.0000 | 0.1568 | (0.1391, 0.1741) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.2201 | (0.2012, 0.2394) | 0.0000 | 0.2201 | (0.2019, 0.2406) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1972 | (0.1641, 0.2320) | 0.0000 | 0.1972 | (0.1573, 0.2405) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | -0.0071 | (-0.0273, 0.0119) | 0.7637 | -0.0071 | (-0.0341, 0.0283) | 0.6833 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.2874 | (0.2613, 0.3151) | 0.0000 | 0.2874 | (0.2616, 0.3185) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0629 | (0.0548, 0.0712) | 0.0000 | 0.0629 | (0.0541, 0.0718) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.2366 | (0.1960, 0.2789) | 0.0000 | 0.2366 | (0.1888, 0.2883) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | 0.0395 | (0.0136, 0.0685) | 0.0010 | 0.0395 | (0.0075, 0.0826) | 0.0227 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0425 | (-0.0506, -0.0349) | 1.0000 | -0.0425 | (-0.0490, -0.0366) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | 0.0295 | (-0.0512, 0.1074) | 0.2433 | 0.0295 | (-0.1063, 0.1831) | 0.3683 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.0437 | (0.0031, 0.0844) | 0.0213 | 0.0437 | (-0.0281, 0.1156) | 0.1393 |
| controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0189 | (0.0062, 0.0315) | 0.0023 | 0.0189 | (-0.0001, 0.0367) | 0.0280 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1551 | (0.1414, 0.1684) | 0.0000 | 0.1551 | (0.1354, 0.1755) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0006 | (-0.0245, 0.0227) | 0.5297 | -0.0006 | (-0.0272, 0.0245) | 0.5283 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0503 | (-0.0867, -0.0149) | 0.9983 | -0.0503 | (-0.0854, -0.0139) | 0.9977 |
| controlled_alt_vs_controlled_default | naturalness | 0.0087 | (-0.0122, 0.0295) | 0.2023 | 0.0087 | (-0.0191, 0.0338) | 0.2463 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0022 | (-0.0338, 0.0299) | 0.5587 | -0.0022 | (-0.0402, 0.0328) | 0.5403 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0032 | (-0.0060, 0.0126) | 0.2463 | 0.0032 | (-0.0097, 0.0139) | 0.2937 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0618 | (-0.1061, -0.0185) | 0.9983 | -0.0618 | (-0.1071, -0.0166) | 0.9973 |
| controlled_alt_vs_controlled_default | persona_style | -0.0044 | (-0.0284, 0.0195) | 0.6300 | -0.0044 | (-0.0327, 0.0179) | 0.6023 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0009 | (-0.0077, 0.0099) | 0.4160 | 0.0009 | (-0.0066, 0.0087) | 0.4250 |
| controlled_alt_vs_controlled_default | length_score | 0.0390 | (-0.0479, 0.1256) | 0.1960 | 0.0390 | (-0.0786, 0.1548) | 0.2387 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0031 | (-0.0344, 0.0406) | 0.4853 | 0.0031 | (-0.0281, 0.0375) | 0.4463 |
| controlled_alt_vs_controlled_default | bertscore_f1 | 0.0080 | (-0.0065, 0.0226) | 0.1370 | 0.0080 | (-0.0082, 0.0250) | 0.1870 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0145 | (-0.0298, 0.0005) | 0.9720 | -0.0145 | (-0.0359, 0.0073) | 0.9073 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1592 | (0.1305, 0.1887) | 0.0000 | 0.1592 | (0.1347, 0.1841) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1679 | (0.1368, 0.2003) | 0.0000 | 0.1679 | (0.1369, 0.1987) | 0.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0678 | (0.0475, 0.0871) | 0.0000 | 0.0678 | (0.0345, 0.1000) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2082 | (0.1720, 0.2463) | 0.0000 | 0.2082 | (0.1753, 0.2407) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0449 | (0.0354, 0.0549) | 0.0000 | 0.0449 | (0.0385, 0.0530) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1848 | (0.1479, 0.2196) | 0.0000 | 0.1848 | (0.1548, 0.2110) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 0.1002 | (0.0624, 0.1393) | 0.0000 | 0.1002 | (0.0091, 0.2132) | 0.0043 |
| controlled_alt_vs_proposed_raw | distinct1 | 0.0011 | (-0.0075, 0.0093) | 0.3930 | 0.0011 | (-0.0104, 0.0104) | 0.3907 |
| controlled_alt_vs_proposed_raw | length_score | 0.2744 | (0.1905, 0.3500) | 0.0000 | 0.2744 | (0.1565, 0.3890) | 0.0000 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1246 | (0.0839, 0.1683) | 0.0000 | 0.1246 | (0.0612, 0.1938) | 0.0000 |
| controlled_alt_vs_proposed_raw | bertscore_f1 | 0.0160 | (0.0020, 0.0301) | 0.0123 | 0.0160 | (0.0061, 0.0254) | 0.0007 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1331 | (0.1129, 0.1517) | 0.0000 | 0.1331 | (0.1127, 0.1542) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2377 | (0.2139, 0.2616) | 0.0000 | 0.2377 | (0.2065, 0.2649) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1473 | (0.1128, 0.1799) | 0.0000 | 0.1473 | (0.0838, 0.1968) | 0.0000 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0661 | (0.0452, 0.0872) | 0.0000 | 0.0661 | (0.0360, 0.0978) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3105 | (0.2810, 0.3397) | 0.0000 | 0.3105 | (0.2688, 0.3483) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0677 | (0.0599, 0.0759) | 0.0000 | 0.0677 | (0.0589, 0.0764) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1591 | (0.1188, 0.1978) | 0.0000 | 0.1591 | (0.0978, 0.2020) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.1003 | (0.0600, 0.1437) | 0.0000 | 0.1003 | (-0.0049, 0.2375) | 0.0410 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0023 | (-0.0066, 0.0110) | 0.3150 | 0.0023 | (-0.0082, 0.0128) | 0.3437 |
| controlled_alt_vs_candidate_no_context | length_score | 0.2637 | (0.1777, 0.3455) | 0.0000 | 0.2637 | (0.1622, 0.3705) | 0.0000 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1250 | (0.0813, 0.1687) | 0.0000 | 0.1250 | (0.0500, 0.2062) | 0.0000 |
| controlled_alt_vs_candidate_no_context | bertscore_f1 | 0.0468 | (0.0354, 0.0585) | 0.0000 | 0.0468 | (0.0306, 0.0622) | 0.0000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1614 | (0.1445, 0.1781) | 0.0000 | 0.1614 | (0.1302, 0.1869) | 0.0000 |
| controlled_alt_vs_baseline_no_context | context_relevance | 0.2176 | (0.1956, 0.2408) | 0.0000 | 0.2176 | (0.1883, 0.2460) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 0.1507 | (0.1234, 0.1794) | 0.0000 | 0.1507 | (0.1354, 0.1668) | 0.0000 |
| controlled_alt_vs_baseline_no_context | naturalness | 0.0007 | (-0.0179, 0.0196) | 0.4770 | 0.0007 | (-0.0282, 0.0283) | 0.4590 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 0.2814 | (0.2512, 0.3118) | 0.0000 | 0.2814 | (0.2424, 0.3204) | 0.0000 |
| controlled_alt_vs_baseline_no_context | context_overlap | 0.0689 | (0.0614, 0.0768) | 0.0000 | 0.0689 | (0.0610, 0.0764) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 0.1810 | (0.1474, 0.2152) | 0.0000 | 0.1810 | (0.1628, 0.2012) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_style | 0.0297 | (0.0114, 0.0505) | 0.0003 | 0.0297 | (0.0052, 0.0603) | 0.0217 |
| controlled_alt_vs_baseline_no_context | distinct1 | -0.0346 | (-0.0428, -0.0264) | 1.0000 | -0.0346 | (-0.0408, -0.0289) | 1.0000 |
| controlled_alt_vs_baseline_no_context | length_score | 0.0530 | (-0.0280, 0.1318) | 0.0920 | 0.0530 | (-0.0792, 0.1720) | 0.2077 |
| controlled_alt_vs_baseline_no_context | sentence_score | 0.0406 | (-0.0031, 0.0844) | 0.0370 | 0.0406 | (-0.0500, 0.1344) | 0.2120 |
| controlled_alt_vs_baseline_no_context | bertscore_f1 | 0.0381 | (0.0256, 0.0511) | 0.0000 | 0.0381 | (0.0237, 0.0570) | 0.0000 |
| controlled_alt_vs_baseline_no_context | overall_quality | 0.1423 | (0.1295, 0.1550) | 0.0000 | 0.1423 | (0.1305, 0.1556) | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_relevance | 0.2195 | (0.1967, 0.2424) | 0.0000 | 0.2195 | (0.1907, 0.2487) | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1469 | (0.1179, 0.1766) | 0.0000 | 0.1469 | (0.1119, 0.1791) | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | naturalness | 0.0016 | (-0.0154, 0.0183) | 0.4260 | 0.0016 | (-0.0184, 0.0227) | 0.4510 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.2853 | (0.2539, 0.3161) | 0.0000 | 0.2853 | (0.2465, 0.3252) | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_overlap | 0.0660 | (0.0586, 0.0738) | 0.0000 | 0.0660 | (0.0572, 0.0748) | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1748 | (0.1402, 0.2107) | 0.0000 | 0.1748 | (0.1342, 0.2118) | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_style | 0.0350 | (0.0160, 0.0572) | 0.0003 | 0.0350 | (0.0049, 0.0728) | 0.0223 |
| controlled_alt_vs_baseline_no_context_phi3_latest | distinct1 | -0.0417 | (-0.0496, -0.0334) | 1.0000 | -0.0417 | (-0.0490, -0.0340) | 1.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | length_score | 0.0685 | (-0.0048, 0.1384) | 0.0337 | 0.0685 | (-0.0280, 0.1655) | 0.0880 |
| controlled_alt_vs_baseline_no_context_phi3_latest | sentence_score | 0.0469 | (0.0031, 0.0907) | 0.0243 | 0.0469 | (0.0000, 0.0938) | 0.0337 |
| controlled_alt_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0269 | (0.0161, 0.0390) | 0.0000 | 0.0269 | (0.0126, 0.0471) | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | overall_quality | 0.1406 | (0.1285, 0.1530) | 0.0000 | 0.1406 | (0.1218, 0.1584) | 0.0000 |
| proposed_contextual_controlled_alt_vs_baseline_no_context | context_relevance | 0.2176 | (0.1953, 0.2414) | 0.0000 | 0.2176 | (0.1891, 0.2460) | 0.0000 |
| proposed_contextual_controlled_alt_vs_baseline_no_context | persona_consistency | 0.1507 | (0.1235, 0.1789) | 0.0000 | 0.1507 | (0.1358, 0.1675) | 0.0000 |
| proposed_contextual_controlled_alt_vs_baseline_no_context | naturalness | 0.0007 | (-0.0180, 0.0192) | 0.4713 | 0.0007 | (-0.0293, 0.0299) | 0.4833 |
| proposed_contextual_controlled_alt_vs_baseline_no_context | context_keyword_coverage | 0.2814 | (0.2510, 0.3131) | 0.0000 | 0.2814 | (0.2417, 0.3211) | 0.0000 |
| proposed_contextual_controlled_alt_vs_baseline_no_context | context_overlap | 0.0689 | (0.0612, 0.0765) | 0.0000 | 0.0689 | (0.0607, 0.0766) | 0.0000 |
| proposed_contextual_controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 0.1810 | (0.1479, 0.2162) | 0.0000 | 0.1810 | (0.1628, 0.2009) | 0.0000 |
| proposed_contextual_controlled_alt_vs_baseline_no_context | persona_style | 0.0297 | (0.0108, 0.0499) | 0.0010 | 0.0297 | (0.0052, 0.0606) | 0.0230 |
| proposed_contextual_controlled_alt_vs_baseline_no_context | distinct1 | -0.0346 | (-0.0430, -0.0263) | 1.0000 | -0.0346 | (-0.0409, -0.0287) | 1.0000 |
| proposed_contextual_controlled_alt_vs_baseline_no_context | length_score | 0.0530 | (-0.0250, 0.1333) | 0.1020 | 0.0530 | (-0.0771, 0.1720) | 0.1977 |
| proposed_contextual_controlled_alt_vs_baseline_no_context | sentence_score | 0.0406 | (-0.0001, 0.0844) | 0.0353 | 0.0406 | (-0.0500, 0.1375) | 0.2100 |
| proposed_contextual_controlled_alt_vs_baseline_no_context | bertscore_f1 | 0.0381 | (0.0246, 0.0516) | 0.0000 | 0.0381 | (0.0237, 0.0560) | 0.0000 |
| proposed_contextual_controlled_alt_vs_baseline_no_context | overall_quality | 0.1423 | (0.1296, 0.1546) | 0.0000 | 0.1423 | (0.1305, 0.1549) | 0.0000 |
| proposed_contextual_controlled_alt_vs_baseline_no_context_phi3_latest | context_relevance | 0.2195 | (0.1972, 0.2423) | 0.0000 | 0.2195 | (0.1908, 0.2481) | 0.0000 |
| proposed_contextual_controlled_alt_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1469 | (0.1183, 0.1751) | 0.0000 | 0.1469 | (0.1129, 0.1800) | 0.0000 |
| proposed_contextual_controlled_alt_vs_baseline_no_context_phi3_latest | naturalness | 0.0016 | (-0.0151, 0.0182) | 0.4183 | 0.0016 | (-0.0179, 0.0237) | 0.4703 |
| proposed_contextual_controlled_alt_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.2853 | (0.2553, 0.3167) | 0.0000 | 0.2853 | (0.2468, 0.3252) | 0.0000 |
| proposed_contextual_controlled_alt_vs_baseline_no_context_phi3_latest | context_overlap | 0.0660 | (0.0585, 0.0740) | 0.0000 | 0.0660 | (0.0575, 0.0754) | 0.0000 |
| proposed_contextual_controlled_alt_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1748 | (0.1390, 0.2089) | 0.0000 | 0.1748 | (0.1368, 0.2114) | 0.0000 |
| proposed_contextual_controlled_alt_vs_baseline_no_context_phi3_latest | persona_style | 0.0350 | (0.0151, 0.0579) | 0.0000 | 0.0350 | (0.0000, 0.0728) | 0.0257 |
| proposed_contextual_controlled_alt_vs_baseline_no_context_phi3_latest | distinct1 | -0.0417 | (-0.0496, -0.0335) | 1.0000 | -0.0417 | (-0.0491, -0.0340) | 1.0000 |
| proposed_contextual_controlled_alt_vs_baseline_no_context_phi3_latest | length_score | 0.0685 | (-0.0039, 0.1399) | 0.0343 | 0.0685 | (-0.0280, 0.1699) | 0.0853 |
| proposed_contextual_controlled_alt_vs_baseline_no_context_phi3_latest | sentence_score | 0.0469 | (0.0031, 0.0906) | 0.0210 | 0.0469 | (-0.0001, 0.0969) | 0.0333 |
| proposed_contextual_controlled_alt_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0269 | (0.0154, 0.0383) | 0.0000 | 0.0269 | (0.0124, 0.0477) | 0.0000 |
| proposed_contextual_controlled_alt_vs_baseline_no_context_phi3_latest | overall_quality | 0.1406 | (0.1281, 0.1532) | 0.0000 | 0.1406 | (0.1211, 0.1591) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 60 | 12 | 40 | 0.7143 | 0.8333 |
| proposed_vs_candidate_no_context | persona_consistency | 28 | 20 | 64 | 0.5357 | 0.5833 |
| proposed_vs_candidate_no_context | naturalness | 35 | 37 | 40 | 0.4911 | 0.4861 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 49 | 5 | 58 | 0.6964 | 0.9074 |
| proposed_vs_candidate_no_context | context_overlap | 55 | 17 | 40 | 0.6696 | 0.7639 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 17 | 18 | 77 | 0.4955 | 0.4857 |
| proposed_vs_candidate_no_context | persona_style | 16 | 12 | 84 | 0.5179 | 0.5714 |
| proposed_vs_candidate_no_context | distinct1 | 36 | 31 | 45 | 0.5223 | 0.5373 |
| proposed_vs_candidate_no_context | length_score | 34 | 35 | 43 | 0.4955 | 0.4928 |
| proposed_vs_candidate_no_context | sentence_score | 18 | 17 | 77 | 0.5045 | 0.5143 |
| proposed_vs_candidate_no_context | bertscore_f1 | 62 | 21 | 29 | 0.6830 | 0.7470 |
| proposed_vs_candidate_no_context | overall_quality | 55 | 28 | 29 | 0.6205 | 0.6627 |
| proposed_vs_baseline_no_context | context_relevance | 71 | 41 | 0 | 0.6339 | 0.6339 |
| proposed_vs_baseline_no_context | persona_consistency | 31 | 40 | 41 | 0.4598 | 0.4366 |
| proposed_vs_baseline_no_context | naturalness | 25 | 86 | 1 | 0.2277 | 0.2252 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 46 | 26 | 40 | 0.5893 | 0.6389 |
| proposed_vs_baseline_no_context | context_overlap | 78 | 34 | 0 | 0.6964 | 0.6964 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 23 | 24 | 65 | 0.4955 | 0.4894 |
| proposed_vs_baseline_no_context | persona_style | 10 | 23 | 79 | 0.4420 | 0.3030 |
| proposed_vs_baseline_no_context | distinct1 | 24 | 76 | 12 | 0.2679 | 0.2400 |
| proposed_vs_baseline_no_context | length_score | 26 | 84 | 2 | 0.2411 | 0.2364 |
| proposed_vs_baseline_no_context | sentence_score | 13 | 40 | 59 | 0.3795 | 0.2453 |
| proposed_vs_baseline_no_context | bertscore_f1 | 65 | 47 | 0 | 0.5804 | 0.5804 |
| proposed_vs_baseline_no_context | overall_quality | 51 | 61 | 0 | 0.4554 | 0.4554 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 64 | 47 | 1 | 0.5759 | 0.5766 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | 30 | 43 | 39 | 0.4420 | 0.4110 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | 33 | 79 | 0 | 0.2946 | 0.2946 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 45 | 21 | 46 | 0.6071 | 0.6818 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 72 | 39 | 1 | 0.6473 | 0.6486 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 24 | 28 | 60 | 0.4821 | 0.4615 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | 9 | 23 | 80 | 0.4375 | 0.2812 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | 13 | 85 | 14 | 0.1786 | 0.1327 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | 34 | 77 | 1 | 0.3080 | 0.3063 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | 16 | 40 | 56 | 0.3929 | 0.2857 |
| proposed_vs_baseline_no_context_phi3_latest | bertscore_f1 | 56 | 56 | 0 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | 50 | 62 | 0 | 0.4464 | 0.4464 |
| controlled_vs_proposed_raw | context_relevance | 94 | 18 | 0 | 0.8393 | 0.8393 |
| controlled_vs_proposed_raw | persona_consistency | 95 | 5 | 12 | 0.9018 | 0.9500 |
| controlled_vs_proposed_raw | naturalness | 77 | 35 | 0 | 0.6875 | 0.6875 |
| controlled_vs_proposed_raw | context_keyword_coverage | 92 | 15 | 5 | 0.8438 | 0.8598 |
| controlled_vs_proposed_raw | context_overlap | 86 | 25 | 1 | 0.7723 | 0.7748 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 93 | 4 | 15 | 0.8973 | 0.9588 |
| controlled_vs_proposed_raw | persona_style | 35 | 4 | 73 | 0.6384 | 0.8974 |
| controlled_vs_proposed_raw | distinct1 | 59 | 51 | 2 | 0.5357 | 0.5364 |
| controlled_vs_proposed_raw | length_score | 74 | 35 | 3 | 0.6741 | 0.6789 |
| controlled_vs_proposed_raw | sentence_score | 49 | 11 | 52 | 0.6696 | 0.8167 |
| controlled_vs_proposed_raw | bertscore_f1 | 63 | 49 | 0 | 0.5625 | 0.5625 |
| controlled_vs_proposed_raw | overall_quality | 102 | 10 | 0 | 0.9107 | 0.9107 |
| controlled_vs_candidate_no_context | context_relevance | 111 | 1 | 0 | 0.9911 | 0.9911 |
| controlled_vs_candidate_no_context | persona_consistency | 93 | 11 | 8 | 0.8661 | 0.8942 |
| controlled_vs_candidate_no_context | naturalness | 76 | 36 | 0 | 0.6786 | 0.6786 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 110 | 0 | 2 | 0.9911 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 107 | 5 | 0 | 0.9554 | 0.9554 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 91 | 8 | 13 | 0.8705 | 0.9192 |
| controlled_vs_candidate_no_context | persona_style | 34 | 8 | 70 | 0.6161 | 0.8095 |
| controlled_vs_candidate_no_context | distinct1 | 62 | 48 | 2 | 0.5625 | 0.5636 |
| controlled_vs_candidate_no_context | length_score | 73 | 35 | 4 | 0.6696 | 0.6759 |
| controlled_vs_candidate_no_context | sentence_score | 51 | 12 | 49 | 0.6741 | 0.8095 |
| controlled_vs_candidate_no_context | bertscore_f1 | 86 | 26 | 0 | 0.7679 | 0.7679 |
| controlled_vs_candidate_no_context | overall_quality | 109 | 3 | 0 | 0.9732 | 0.9732 |
| controlled_vs_baseline_no_context | context_relevance | 112 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 95 | 7 | 10 | 0.8929 | 0.9314 |
| controlled_vs_baseline_no_context | naturalness | 50 | 61 | 1 | 0.4509 | 0.4505 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 106 | 0 | 6 | 0.9732 | 1.0000 |
| controlled_vs_baseline_no_context | context_overlap | 111 | 1 | 0 | 0.9911 | 0.9911 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 94 | 5 | 13 | 0.8973 | 0.9495 |
| controlled_vs_baseline_no_context | persona_style | 21 | 7 | 84 | 0.5625 | 0.7500 |
| controlled_vs_baseline_no_context | distinct1 | 21 | 83 | 8 | 0.2232 | 0.2019 |
| controlled_vs_baseline_no_context | length_score | 56 | 52 | 4 | 0.5179 | 0.5185 |
| controlled_vs_baseline_no_context | sentence_score | 33 | 21 | 58 | 0.5536 | 0.6111 |
| controlled_vs_baseline_no_context | bertscore_f1 | 78 | 34 | 0 | 0.6964 | 0.6964 |
| controlled_vs_baseline_no_context | overall_quality | 110 | 2 | 0 | 0.9821 | 0.9821 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 112 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 92 | 9 | 11 | 0.8705 | 0.9109 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | 59 | 53 | 0 | 0.5268 | 0.5268 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 108 | 0 | 4 | 0.9821 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 108 | 4 | 0 | 0.9643 | 0.9643 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 91 | 8 | 13 | 0.8705 | 0.9192 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | 17 | 6 | 89 | 0.5491 | 0.7391 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | 16 | 91 | 5 | 0.1652 | 0.1495 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | 63 | 46 | 3 | 0.5759 | 0.5780 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 30 | 16 | 66 | 0.5625 | 0.6522 |
| controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 68 | 44 | 0 | 0.6071 | 0.6071 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 110 | 2 | 0 | 0.9821 | 0.9821 |
| controlled_alt_vs_controlled_default | context_relevance | 49 | 54 | 9 | 0.4777 | 0.4757 |
| controlled_alt_vs_controlled_default | persona_consistency | 24 | 42 | 46 | 0.4196 | 0.3636 |
| controlled_alt_vs_controlled_default | naturalness | 65 | 39 | 8 | 0.6161 | 0.6250 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 37 | 41 | 34 | 0.4821 | 0.4744 |
| controlled_alt_vs_controlled_default | context_overlap | 55 | 48 | 9 | 0.5312 | 0.5340 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 14 | 36 | 62 | 0.4018 | 0.2800 |
| controlled_alt_vs_controlled_default | persona_style | 14 | 16 | 82 | 0.4911 | 0.4667 |
| controlled_alt_vs_controlled_default | distinct1 | 50 | 53 | 9 | 0.4866 | 0.4854 |
| controlled_alt_vs_controlled_default | length_score | 59 | 43 | 10 | 0.5714 | 0.5784 |
| controlled_alt_vs_controlled_default | sentence_score | 19 | 18 | 75 | 0.5045 | 0.5135 |
| controlled_alt_vs_controlled_default | bertscore_f1 | 53 | 51 | 8 | 0.5089 | 0.5096 |
| controlled_alt_vs_controlled_default | overall_quality | 51 | 53 | 8 | 0.4911 | 0.4904 |
| controlled_alt_vs_proposed_raw | context_relevance | 94 | 18 | 0 | 0.8393 | 0.8393 |
| controlled_alt_vs_proposed_raw | persona_consistency | 84 | 13 | 15 | 0.8170 | 0.8660 |
| controlled_alt_vs_proposed_raw | naturalness | 82 | 30 | 0 | 0.7321 | 0.7321 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 89 | 13 | 10 | 0.8393 | 0.8725 |
| controlled_alt_vs_proposed_raw | context_overlap | 92 | 20 | 0 | 0.8214 | 0.8214 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 80 | 10 | 22 | 0.8125 | 0.8889 |
| controlled_alt_vs_proposed_raw | persona_style | 31 | 9 | 72 | 0.5982 | 0.7750 |
| controlled_alt_vs_proposed_raw | distinct1 | 60 | 49 | 3 | 0.5491 | 0.5505 |
| controlled_alt_vs_proposed_raw | length_score | 76 | 32 | 4 | 0.6964 | 0.7037 |
| controlled_alt_vs_proposed_raw | sentence_score | 50 | 11 | 51 | 0.6741 | 0.8197 |
| controlled_alt_vs_proposed_raw | bertscore_f1 | 69 | 43 | 0 | 0.6161 | 0.6161 |
| controlled_alt_vs_proposed_raw | overall_quality | 98 | 14 | 0 | 0.8750 | 0.8750 |
| controlled_alt_vs_candidate_no_context | context_relevance | 110 | 2 | 0 | 0.9821 | 0.9821 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 83 | 17 | 12 | 0.7946 | 0.8300 |
| controlled_alt_vs_candidate_no_context | naturalness | 83 | 29 | 0 | 0.7411 | 0.7411 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 106 | 1 | 5 | 0.9688 | 0.9907 |
| controlled_alt_vs_candidate_no_context | context_overlap | 109 | 3 | 0 | 0.9732 | 0.9732 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 81 | 13 | 18 | 0.8036 | 0.8617 |
| controlled_alt_vs_candidate_no_context | persona_style | 31 | 11 | 70 | 0.5893 | 0.7381 |
| controlled_alt_vs_candidate_no_context | distinct1 | 66 | 45 | 1 | 0.5938 | 0.5946 |
| controlled_alt_vs_candidate_no_context | length_score | 76 | 35 | 1 | 0.6830 | 0.6847 |
| controlled_alt_vs_candidate_no_context | sentence_score | 53 | 13 | 46 | 0.6786 | 0.8030 |
| controlled_alt_vs_candidate_no_context | bertscore_f1 | 90 | 22 | 0 | 0.8036 | 0.8036 |
| controlled_alt_vs_candidate_no_context | overall_quality | 105 | 7 | 0 | 0.9375 | 0.9375 |
| controlled_alt_vs_baseline_no_context | context_relevance | 110 | 2 | 0 | 0.9821 | 0.9821 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 84 | 10 | 18 | 0.8304 | 0.8936 |
| controlled_alt_vs_baseline_no_context | naturalness | 61 | 51 | 0 | 0.5446 | 0.5446 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 104 | 0 | 8 | 0.9643 | 1.0000 |
| controlled_alt_vs_baseline_no_context | context_overlap | 108 | 4 | 0 | 0.9643 | 0.9643 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 83 | 6 | 23 | 0.8438 | 0.9326 |
| controlled_alt_vs_baseline_no_context | persona_style | 18 | 9 | 85 | 0.5402 | 0.6667 |
| controlled_alt_vs_baseline_no_context | distinct1 | 24 | 86 | 2 | 0.2232 | 0.2182 |
| controlled_alt_vs_baseline_no_context | length_score | 64 | 48 | 0 | 0.5714 | 0.5714 |
| controlled_alt_vs_baseline_no_context | sentence_score | 32 | 19 | 61 | 0.5580 | 0.6275 |
| controlled_alt_vs_baseline_no_context | bertscore_f1 | 74 | 38 | 0 | 0.6607 | 0.6607 |
| controlled_alt_vs_baseline_no_context | overall_quality | 109 | 3 | 0 | 0.9732 | 0.9732 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_relevance | 108 | 3 | 1 | 0.9688 | 0.9730 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_consistency | 84 | 12 | 16 | 0.8214 | 0.8750 |
| controlled_alt_vs_baseline_no_context_phi3_latest | naturalness | 59 | 53 | 0 | 0.5268 | 0.5268 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 104 | 0 | 8 | 0.9643 | 1.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_overlap | 104 | 7 | 1 | 0.9330 | 0.9369 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 83 | 11 | 18 | 0.8214 | 0.8830 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_style | 19 | 8 | 85 | 0.5491 | 0.7037 |
| controlled_alt_vs_baseline_no_context_phi3_latest | distinct1 | 21 | 87 | 4 | 0.2054 | 0.1944 |
| controlled_alt_vs_baseline_no_context_phi3_latest | length_score | 66 | 43 | 3 | 0.6027 | 0.6055 |
| controlled_alt_vs_baseline_no_context_phi3_latest | sentence_score | 35 | 20 | 57 | 0.5670 | 0.6364 |
| controlled_alt_vs_baseline_no_context_phi3_latest | bertscore_f1 | 72 | 40 | 0 | 0.6429 | 0.6429 |
| controlled_alt_vs_baseline_no_context_phi3_latest | overall_quality | 110 | 2 | 0 | 0.9821 | 0.9821 |
| proposed_contextual_controlled_alt_vs_baseline_no_context | context_relevance | 110 | 2 | 0 | 0.9821 | 0.9821 |
| proposed_contextual_controlled_alt_vs_baseline_no_context | persona_consistency | 84 | 10 | 18 | 0.8304 | 0.8936 |
| proposed_contextual_controlled_alt_vs_baseline_no_context | naturalness | 61 | 51 | 0 | 0.5446 | 0.5446 |
| proposed_contextual_controlled_alt_vs_baseline_no_context | context_keyword_coverage | 104 | 0 | 8 | 0.9643 | 1.0000 |
| proposed_contextual_controlled_alt_vs_baseline_no_context | context_overlap | 108 | 4 | 0 | 0.9643 | 0.9643 |
| proposed_contextual_controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 83 | 6 | 23 | 0.8438 | 0.9326 |
| proposed_contextual_controlled_alt_vs_baseline_no_context | persona_style | 18 | 9 | 85 | 0.5402 | 0.6667 |
| proposed_contextual_controlled_alt_vs_baseline_no_context | distinct1 | 24 | 86 | 2 | 0.2232 | 0.2182 |
| proposed_contextual_controlled_alt_vs_baseline_no_context | length_score | 64 | 48 | 0 | 0.5714 | 0.5714 |
| proposed_contextual_controlled_alt_vs_baseline_no_context | sentence_score | 32 | 19 | 61 | 0.5580 | 0.6275 |
| proposed_contextual_controlled_alt_vs_baseline_no_context | bertscore_f1 | 74 | 38 | 0 | 0.6607 | 0.6607 |
| proposed_contextual_controlled_alt_vs_baseline_no_context | overall_quality | 109 | 3 | 0 | 0.9732 | 0.9732 |
| proposed_contextual_controlled_alt_vs_baseline_no_context_phi3_latest | context_relevance | 108 | 3 | 1 | 0.9688 | 0.9730 |
| proposed_contextual_controlled_alt_vs_baseline_no_context_phi3_latest | persona_consistency | 84 | 12 | 16 | 0.8214 | 0.8750 |
| proposed_contextual_controlled_alt_vs_baseline_no_context_phi3_latest | naturalness | 59 | 53 | 0 | 0.5268 | 0.5268 |
| proposed_contextual_controlled_alt_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 104 | 0 | 8 | 0.9643 | 1.0000 |
| proposed_contextual_controlled_alt_vs_baseline_no_context_phi3_latest | context_overlap | 104 | 7 | 1 | 0.9330 | 0.9369 |
| proposed_contextual_controlled_alt_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 83 | 11 | 18 | 0.8214 | 0.8830 |
| proposed_contextual_controlled_alt_vs_baseline_no_context_phi3_latest | persona_style | 19 | 8 | 85 | 0.5491 | 0.7037 |
| proposed_contextual_controlled_alt_vs_baseline_no_context_phi3_latest | distinct1 | 21 | 87 | 4 | 0.2054 | 0.1944 |
| proposed_contextual_controlled_alt_vs_baseline_no_context_phi3_latest | length_score | 66 | 43 | 3 | 0.6027 | 0.6055 |
| proposed_contextual_controlled_alt_vs_baseline_no_context_phi3_latest | sentence_score | 35 | 20 | 57 | 0.5670 | 0.6364 |
| proposed_contextual_controlled_alt_vs_baseline_no_context_phi3_latest | bertscore_f1 | 72 | 40 | 0 | 0.6429 | 0.6429 |
| proposed_contextual_controlled_alt_vs_baseline_no_context_phi3_latest | overall_quality | 110 | 2 | 0 | 0.9821 | 0.9821 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.2143 | 0.4464 | 0.5536 |
| proposed_contextual_controlled_alt | 0.0000 | 0.0000 | 0.1964 | 0.4821 | 0.5000 |
| proposed_contextual | 0.0000 | 0.0000 | 0.4375 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.4911 | 0.0000 | 0.0000 |
| baseline_no_context | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline_no_context_phi3_latest | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `8`
- Unique template signatures: `71`
- Template signature ratio: `0.6339`
- Effective sample size by source clustering: `8.00`
- Effective sample size by template-signature clustering: `56.00`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: enabled.

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
# Proposal Alignment Evaluation Report

- Run ID: `20260305T135517Z`
- Generated: `2026-03-05T14:20:03.728433+00:00`
- Scenarios: `artifacts\proposal_control_tuning\final112_tuned_m1_l2\20260305T135517Z\scenarios.jsonl`
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
| proposed_contextual_controlled | 0.2769 (0.2604, 0.2938) | 0.3593 (0.3286, 0.3896) | 0.8728 (0.8584, 0.8876) | 0.3894 (0.3779, 0.4012) | 0.0770 |
| proposed_contextual | 0.0884 (0.0689, 0.1114) | 0.1618 (0.1364, 0.1864) | 0.7997 (0.7870, 0.8125) | 0.2355 (0.2194, 0.2517) | 0.0797 |
| candidate_no_context | 0.0257 (0.0197, 0.0324) | 0.1677 (0.1391, 0.1966) | 0.8102 (0.7958, 0.8258) | 0.2098 (0.1979, 0.2219) | 0.0418 |
| baseline_no_context | 0.0429 (0.0348, 0.0511) | 0.1838 (0.1619, 0.2054) | 0.8867 (0.8758, 0.8980) | 0.2364 (0.2279, 0.2448) | 0.0549 |
| baseline_no_context_phi3_latest | 0.0419 (0.0332, 0.0510) | 0.1915 (0.1684, 0.2166) | 0.8878 (0.8765, 0.8986) | 0.2386 (0.2293, 0.2482) | 0.0542 |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0627 | 2.4355 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0059 | -0.0351 |
| proposed_vs_candidate_no_context | naturalness | -0.0105 | -0.0129 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0818 | 4.2346 |
| proposed_vs_candidate_no_context | context_overlap | 0.0181 | 0.4439 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0134 | -0.1524 |
| proposed_vs_candidate_no_context | persona_style | 0.0239 | 0.0490 |
| proposed_vs_candidate_no_context | distinct1 | 0.0027 | 0.0029 |
| proposed_vs_candidate_no_context | length_score | -0.0592 | -0.1947 |
| proposed_vs_candidate_no_context | sentence_score | 0.0031 | 0.0041 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0379 | 0.9079 |
| proposed_vs_candidate_no_context | overall_quality | 0.0258 | 0.1228 |
| proposed_vs_baseline_no_context | context_relevance | 0.0456 | 1.0632 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0219 | -0.1193 |
| proposed_vs_baseline_no_context | naturalness | -0.0870 | -0.0981 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0573 | 1.3050 |
| proposed_vs_baseline_no_context | context_overlap | 0.0183 | 0.4519 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0101 | -0.1200 |
| proposed_vs_baseline_no_context | persona_style | -0.0692 | -0.1190 |
| proposed_vs_baseline_no_context | distinct1 | -0.0407 | -0.0416 |
| proposed_vs_baseline_no_context | length_score | -0.2988 | -0.5495 |
| proposed_vs_baseline_no_context | sentence_score | -0.1094 | -0.1264 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0248 | 0.4521 |
| proposed_vs_baseline_no_context | overall_quality | -0.0009 | -0.0036 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 0.0465 | 1.1096 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | -0.0297 | -0.1551 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | -0.0880 | -0.0992 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.0582 | 1.3566 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 0.0192 | 0.4849 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | -0.0179 | -0.1943 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | -0.0769 | -0.1306 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | -0.0415 | -0.0424 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | -0.3057 | -0.5551 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | -0.1031 | -0.1200 |
| proposed_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0255 | 0.4698 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | -0.0031 | -0.0130 |
| controlled_vs_proposed_raw | context_relevance | 0.1885 | 2.1316 |
| controlled_vs_proposed_raw | persona_consistency | 0.1975 | 1.2203 |
| controlled_vs_proposed_raw | naturalness | 0.0731 | 0.0914 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2495 | 2.4673 |
| controlled_vs_proposed_raw | context_overlap | 0.0461 | 0.7842 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2248 | 3.0281 |
| controlled_vs_proposed_raw | persona_style | 0.0883 | 0.1723 |
| controlled_vs_proposed_raw | distinct1 | -0.0030 | -0.0032 |
| controlled_vs_proposed_raw | length_score | 0.2812 | 1.1482 |
| controlled_vs_proposed_raw | sentence_score | 0.1813 | 0.2397 |
| controlled_vs_proposed_raw | bertscore_f1 | -0.0027 | -0.0338 |
| controlled_vs_proposed_raw | overall_quality | 0.1539 | 0.6534 |
| controlled_vs_candidate_no_context | context_relevance | 0.2512 | 9.7586 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1916 | 1.1423 |
| controlled_vs_candidate_no_context | naturalness | 0.0626 | 0.0773 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3313 | 17.1499 |
| controlled_vs_candidate_no_context | context_overlap | 0.0642 | 1.5761 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2114 | 2.4141 |
| controlled_vs_candidate_no_context | persona_style | 0.1122 | 0.2298 |
| controlled_vs_candidate_no_context | distinct1 | -0.0003 | -0.0003 |
| controlled_vs_candidate_no_context | length_score | 0.2220 | 0.7299 |
| controlled_vs_candidate_no_context | sentence_score | 0.1844 | 0.2448 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0352 | 0.8434 |
| controlled_vs_candidate_no_context | overall_quality | 0.1797 | 0.8564 |
| controlled_vs_baseline_no_context | context_relevance | 0.2340 | 5.4610 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1756 | 0.9553 |
| controlled_vs_baseline_no_context | naturalness | -0.0139 | -0.0157 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3068 | 6.9920 |
| controlled_vs_baseline_no_context | context_overlap | 0.0644 | 1.5905 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2147 | 2.5449 |
| controlled_vs_baseline_no_context | persona_style | 0.0191 | 0.0329 |
| controlled_vs_baseline_no_context | distinct1 | -0.0437 | -0.0447 |
| controlled_vs_baseline_no_context | length_score | -0.0176 | -0.0323 |
| controlled_vs_baseline_no_context | sentence_score | 0.0719 | 0.0830 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0221 | 0.4030 |
| controlled_vs_baseline_no_context | overall_quality | 0.1530 | 0.6474 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.2350 | 5.6063 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1678 | 0.8759 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | -0.0150 | -0.0168 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.3077 | 7.1709 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0653 | 1.6494 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.2069 | 2.2455 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | 0.0113 | 0.0192 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0445 | -0.0454 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | -0.0244 | -0.0443 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.0781 | 0.0909 |
| controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0228 | 0.4201 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1508 | 0.6319 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2340 | 5.4610 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1756 | 0.9553 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0139 | -0.0157 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3068 | 6.9920 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0644 | 1.5905 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2147 | 2.5449 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0191 | 0.0329 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0437 | -0.0447 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | -0.0176 | -0.0323 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0719 | 0.0830 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0221 | 0.4030 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1530 | 0.6474 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.2350 | 5.6063 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1678 | 0.8759 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | -0.0150 | -0.0168 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.3077 | 7.1709 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0653 | 1.6494 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.2069 | 2.2455 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | 0.0113 | 0.0192 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0445 | -0.0454 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | -0.0244 | -0.0443 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.0781 | 0.0909 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0228 | 0.4201 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1508 | 0.6319 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0627 | (0.0416, 0.0840) | 0.0000 | 0.0627 | (0.0338, 0.0970) | 0.0000 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0059 | (-0.0378, 0.0249) | 0.6457 | -0.0059 | (-0.0321, 0.0188) | 0.6573 |
| proposed_vs_candidate_no_context | naturalness | -0.0105 | (-0.0265, 0.0061) | 0.8993 | -0.0105 | (-0.0256, 0.0047) | 0.9160 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0818 | (0.0539, 0.1115) | 0.0000 | 0.0818 | (0.0436, 0.1282) | 0.0000 |
| proposed_vs_candidate_no_context | context_overlap | 0.0181 | (0.0114, 0.0253) | 0.0000 | 0.0181 | (0.0071, 0.0299) | 0.0000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0134 | (-0.0503, 0.0227) | 0.7737 | -0.0134 | (-0.0471, 0.0172) | 0.7830 |
| proposed_vs_candidate_no_context | persona_style | 0.0239 | (-0.0049, 0.0533) | 0.0557 | 0.0239 | (0.0065, 0.0474) | 0.0240 |
| proposed_vs_candidate_no_context | distinct1 | 0.0027 | (-0.0049, 0.0101) | 0.2413 | 0.0027 | (-0.0019, 0.0078) | 0.1333 |
| proposed_vs_candidate_no_context | length_score | -0.0592 | (-0.1250, 0.0036) | 0.9663 | -0.0592 | (-0.1119, -0.0065) | 0.9850 |
| proposed_vs_candidate_no_context | sentence_score | 0.0031 | (-0.0312, 0.0375) | 0.4737 | 0.0031 | (-0.0281, 0.0375) | 0.4790 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0379 | (0.0262, 0.0506) | 0.0000 | 0.0379 | (0.0209, 0.0563) | 0.0000 |
| proposed_vs_candidate_no_context | overall_quality | 0.0258 | (0.0088, 0.0429) | 0.0003 | 0.0258 | (0.0084, 0.0478) | 0.0000 |
| proposed_vs_baseline_no_context | context_relevance | 0.0456 | (0.0257, 0.0683) | 0.0000 | 0.0456 | (0.0159, 0.0769) | 0.0000 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0219 | (-0.0528, 0.0091) | 0.9230 | -0.0219 | (-0.0738, 0.0231) | 0.8180 |
| proposed_vs_baseline_no_context | naturalness | -0.0870 | (-0.1029, -0.0708) | 1.0000 | -0.0870 | (-0.1149, -0.0572) | 1.0000 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0573 | (0.0305, 0.0860) | 0.0000 | 0.0573 | (0.0194, 0.1003) | 0.0003 |
| proposed_vs_baseline_no_context | context_overlap | 0.0183 | (0.0117, 0.0249) | 0.0000 | 0.0183 | (0.0089, 0.0281) | 0.0000 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0101 | (-0.0461, 0.0242) | 0.7177 | -0.0101 | (-0.0445, 0.0286) | 0.7117 |
| proposed_vs_baseline_no_context | persona_style | -0.0692 | (-0.1115, -0.0279) | 0.9993 | -0.0692 | (-0.2072, 0.0211) | 0.8347 |
| proposed_vs_baseline_no_context | distinct1 | -0.0407 | (-0.0496, -0.0314) | 1.0000 | -0.0407 | (-0.0604, -0.0193) | 1.0000 |
| proposed_vs_baseline_no_context | length_score | -0.2988 | (-0.3610, -0.2357) | 1.0000 | -0.2988 | (-0.3914, -0.1935) | 1.0000 |
| proposed_vs_baseline_no_context | sentence_score | -0.1094 | (-0.1500, -0.0687) | 1.0000 | -0.1094 | (-0.1844, -0.0312) | 0.9983 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0248 | (0.0110, 0.0382) | 0.0000 | 0.0248 | (0.0012, 0.0507) | 0.0213 |
| proposed_vs_baseline_no_context | overall_quality | -0.0009 | (-0.0176, 0.0161) | 0.5443 | -0.0009 | (-0.0271, 0.0265) | 0.5417 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 0.0465 | (0.0260, 0.0691) | 0.0000 | 0.0465 | (0.0115, 0.0868) | 0.0020 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | -0.0297 | (-0.0616, 0.0007) | 0.9703 | -0.0297 | (-0.0724, 0.0073) | 0.9443 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | -0.0880 | (-0.1041, -0.0722) | 1.0000 | -0.0880 | (-0.1113, -0.0654) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.0582 | (0.0310, 0.0879) | 0.0000 | 0.0582 | (0.0106, 0.1103) | 0.0023 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 0.0192 | (0.0121, 0.0263) | 0.0000 | 0.0192 | (0.0088, 0.0294) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | -0.0179 | (-0.0536, 0.0173) | 0.8400 | -0.0179 | (-0.0458, 0.0113) | 0.8870 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | -0.0769 | (-0.1184, -0.0366) | 1.0000 | -0.0769 | (-0.2204, 0.0177) | 0.8917 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | -0.0415 | (-0.0503, -0.0331) | 1.0000 | -0.0415 | (-0.0568, -0.0232) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | -0.3057 | (-0.3690, -0.2431) | 1.0000 | -0.3057 | (-0.3878, -0.2152) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | -0.1031 | (-0.1438, -0.0625) | 1.0000 | -0.1031 | (-0.1781, -0.0219) | 0.9973 |
| proposed_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0255 | (0.0119, 0.0394) | 0.0000 | 0.0255 | (0.0007, 0.0532) | 0.0227 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | -0.0031 | (-0.0201, 0.0142) | 0.6517 | -0.0031 | (-0.0297, 0.0260) | 0.6023 |
| controlled_vs_proposed_raw | context_relevance | 0.1885 | (0.1642, 0.2121) | 0.0000 | 0.1885 | (0.1645, 0.2124) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.1975 | (0.1600, 0.2351) | 0.0000 | 0.1975 | (0.1483, 0.2470) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0731 | (0.0528, 0.0931) | 0.0000 | 0.0731 | (0.0321, 0.1088) | 0.0003 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2495 | (0.2171, 0.2797) | 0.0000 | 0.2495 | (0.2191, 0.2777) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0461 | (0.0387, 0.0537) | 0.0000 | 0.0461 | (0.0363, 0.0565) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2248 | (0.1818, 0.2707) | 0.0000 | 0.2248 | (0.1771, 0.2820) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.0883 | (0.0466, 0.1312) | 0.0000 | 0.0883 | (-0.0084, 0.2239) | 0.0590 |
| controlled_vs_proposed_raw | distinct1 | -0.0030 | (-0.0126, 0.0066) | 0.7310 | -0.0030 | (-0.0220, 0.0148) | 0.5963 |
| controlled_vs_proposed_raw | length_score | 0.2812 | (0.2036, 0.3604) | 0.0000 | 0.2812 | (0.1390, 0.4036) | 0.0000 |
| controlled_vs_proposed_raw | sentence_score | 0.1812 | (0.1469, 0.2188) | 0.0000 | 0.1812 | (0.1187, 0.2406) | 0.0000 |
| controlled_vs_proposed_raw | bertscore_f1 | -0.0027 | (-0.0166, 0.0113) | 0.6423 | -0.0027 | (-0.0191, 0.0121) | 0.6187 |
| controlled_vs_proposed_raw | overall_quality | 0.1539 | (0.1366, 0.1719) | 0.0000 | 0.1539 | (0.1288, 0.1786) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2512 | (0.2348, 0.2693) | 0.0000 | 0.2512 | (0.2313, 0.2741) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1916 | (0.1563, 0.2260) | 0.0000 | 0.1916 | (0.1321, 0.2413) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.0626 | (0.0402, 0.0850) | 0.0000 | 0.0626 | (0.0252, 0.0999) | 0.0003 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3313 | (0.3088, 0.3547) | 0.0000 | 0.3313 | (0.3054, 0.3606) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0642 | (0.0564, 0.0719) | 0.0000 | 0.0642 | (0.0574, 0.0716) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2114 | (0.1697, 0.2536) | 0.0000 | 0.2114 | (0.1508, 0.2731) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.1122 | (0.0698, 0.1579) | 0.0000 | 0.1122 | (0.0050, 0.2512) | 0.0153 |
| controlled_vs_candidate_no_context | distinct1 | -0.0003 | (-0.0090, 0.0084) | 0.5190 | -0.0003 | (-0.0164, 0.0157) | 0.5087 |
| controlled_vs_candidate_no_context | length_score | 0.2220 | (0.1318, 0.3086) | 0.0000 | 0.2220 | (0.0791, 0.3571) | 0.0017 |
| controlled_vs_candidate_no_context | sentence_score | 0.1844 | (0.1437, 0.2250) | 0.0000 | 0.1844 | (0.1250, 0.2500) | 0.0000 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0352 | (0.0225, 0.0490) | 0.0000 | 0.0352 | (0.0207, 0.0510) | 0.0000 |
| controlled_vs_candidate_no_context | overall_quality | 0.1797 | (0.1653, 0.1949) | 0.0000 | 0.1797 | (0.1601, 0.1993) | 0.0000 |
| controlled_vs_baseline_no_context | context_relevance | 0.2340 | (0.2173, 0.2512) | 0.0000 | 0.2340 | (0.2101, 0.2607) | 0.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1756 | (0.1414, 0.2085) | 0.0000 | 0.1756 | (0.1459, 0.2128) | 0.0000 |
| controlled_vs_baseline_no_context | naturalness | -0.0139 | (-0.0317, 0.0029) | 0.9483 | -0.0139 | (-0.0385, 0.0058) | 0.8903 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3068 | (0.2849, 0.3288) | 0.0000 | 0.3068 | (0.2721, 0.3454) | 0.0000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0644 | (0.0574, 0.0718) | 0.0000 | 0.0644 | (0.0595, 0.0688) | 0.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2147 | (0.1737, 0.2558) | 0.0000 | 0.2147 | (0.1759, 0.2646) | 0.0000 |
| controlled_vs_baseline_no_context | persona_style | 0.0191 | (-0.0041, 0.0432) | 0.0577 | 0.0191 | (-0.0083, 0.0473) | 0.0797 |
| controlled_vs_baseline_no_context | distinct1 | -0.0437 | (-0.0519, -0.0357) | 1.0000 | -0.0437 | (-0.0528, -0.0349) | 1.0000 |
| controlled_vs_baseline_no_context | length_score | -0.0176 | (-0.0902, 0.0566) | 0.6623 | -0.0176 | (-0.1354, 0.0833) | 0.5913 |
| controlled_vs_baseline_no_context | sentence_score | 0.0719 | (0.0344, 0.1094) | 0.0003 | 0.0719 | (0.0094, 0.1313) | 0.0147 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0221 | (0.0086, 0.0359) | 0.0003 | 0.0221 | (0.0039, 0.0417) | 0.0067 |
| controlled_vs_baseline_no_context | overall_quality | 0.1530 | (0.1403, 0.1661) | 0.0000 | 0.1530 | (0.1360, 0.1701) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.2350 | (0.2174, 0.2526) | 0.0000 | 0.2350 | (0.2043, 0.2683) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1678 | (0.1351, 0.2019) | 0.0000 | 0.1678 | (0.1416, 0.2010) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | -0.0150 | (-0.0339, 0.0046) | 0.9323 | -0.0150 | (-0.0402, 0.0082) | 0.8790 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.3077 | (0.2834, 0.3324) | 0.0000 | 0.3077 | (0.2649, 0.3531) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0653 | (0.0579, 0.0726) | 0.0000 | 0.0653 | (0.0601, 0.0705) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.2069 | (0.1666, 0.2485) | 0.0000 | 0.2069 | (0.1726, 0.2512) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | 0.0113 | (-0.0106, 0.0344) | 0.1513 | 0.0113 | (-0.0118, 0.0351) | 0.1687 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0445 | (-0.0524, -0.0364) | 1.0000 | -0.0445 | (-0.0518, -0.0363) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | -0.0244 | (-0.1089, 0.0625) | 0.7050 | -0.0244 | (-0.1548, 0.0890) | 0.6290 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.0781 | (0.0375, 0.1187) | 0.0000 | 0.0781 | (0.0187, 0.1375) | 0.0060 |
| controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0228 | (0.0083, 0.0367) | 0.0017 | 0.0228 | (0.0036, 0.0444) | 0.0060 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1508 | (0.1384, 0.1624) | 0.0000 | 0.1508 | (0.1349, 0.1668) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2340 | (0.2168, 0.2504) | 0.0000 | 0.2340 | (0.2089, 0.2618) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1756 | (0.1450, 0.2097) | 0.0000 | 0.1756 | (0.1435, 0.2128) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0139 | (-0.0309, 0.0022) | 0.9497 | -0.0139 | (-0.0402, 0.0057) | 0.8913 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3068 | (0.2849, 0.3303) | 0.0000 | 0.3068 | (0.2720, 0.3462) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0644 | (0.0573, 0.0718) | 0.0000 | 0.0644 | (0.0598, 0.0688) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2147 | (0.1743, 0.2553) | 0.0000 | 0.2147 | (0.1768, 0.2663) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0191 | (-0.0047, 0.0435) | 0.0627 | 0.0191 | (-0.0059, 0.0460) | 0.0757 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0437 | (-0.0520, -0.0352) | 1.0000 | -0.0437 | (-0.0525, -0.0350) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | -0.0176 | (-0.0881, 0.0560) | 0.6657 | -0.0176 | (-0.1390, 0.0804) | 0.5880 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0719 | (0.0344, 0.1094) | 0.0003 | 0.0719 | (0.0094, 0.1313) | 0.0157 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0221 | (0.0079, 0.0353) | 0.0003 | 0.0221 | (0.0039, 0.0425) | 0.0057 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1530 | (0.1403, 0.1657) | 0.0000 | 0.1530 | (0.1360, 0.1699) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.2350 | (0.2180, 0.2535) | 0.0000 | 0.2350 | (0.2047, 0.2670) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1678 | (0.1353, 0.2005) | 0.0000 | 0.1678 | (0.1406, 0.2015) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | -0.0150 | (-0.0355, 0.0033) | 0.9380 | -0.0150 | (-0.0398, 0.0072) | 0.8847 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.3077 | (0.2829, 0.3316) | 0.0000 | 0.3077 | (0.2656, 0.3535) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0653 | (0.0579, 0.0727) | 0.0000 | 0.0653 | (0.0602, 0.0708) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.2069 | (0.1659, 0.2475) | 0.0000 | 0.2069 | (0.1741, 0.2521) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | 0.0113 | (-0.0099, 0.0344) | 0.1563 | 0.0113 | (-0.0118, 0.0351) | 0.1557 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0445 | (-0.0523, -0.0365) | 1.0000 | -0.0445 | (-0.0515, -0.0367) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | -0.0244 | (-0.1077, 0.0604) | 0.7110 | -0.0244 | (-0.1676, 0.1006) | 0.6090 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.0781 | (0.0375, 0.1187) | 0.0000 | 0.0781 | (0.0187, 0.1344) | 0.0063 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0228 | (0.0082, 0.0368) | 0.0010 | 0.0228 | (0.0036, 0.0445) | 0.0067 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1508 | (0.1383, 0.1636) | 0.0000 | 0.1508 | (0.1351, 0.1667) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 50 | 16 | 46 | 0.6518 | 0.7576 |
| proposed_vs_candidate_no_context | persona_consistency | 23 | 22 | 67 | 0.5045 | 0.5111 |
| proposed_vs_candidate_no_context | naturalness | 33 | 32 | 47 | 0.5045 | 0.5077 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 44 | 8 | 60 | 0.6607 | 0.8462 |
| proposed_vs_candidate_no_context | context_overlap | 47 | 18 | 47 | 0.6295 | 0.7231 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 15 | 18 | 79 | 0.4866 | 0.4545 |
| proposed_vs_candidate_no_context | persona_style | 15 | 6 | 91 | 0.5402 | 0.7143 |
| proposed_vs_candidate_no_context | distinct1 | 30 | 31 | 51 | 0.4955 | 0.4918 |
| proposed_vs_candidate_no_context | length_score | 28 | 36 | 48 | 0.4643 | 0.4375 |
| proposed_vs_candidate_no_context | sentence_score | 17 | 16 | 79 | 0.5045 | 0.5152 |
| proposed_vs_candidate_no_context | bertscore_f1 | 61 | 29 | 22 | 0.6429 | 0.6778 |
| proposed_vs_candidate_no_context | overall_quality | 55 | 35 | 22 | 0.5893 | 0.6111 |
| proposed_vs_baseline_no_context | context_relevance | 69 | 43 | 0 | 0.6161 | 0.6161 |
| proposed_vs_baseline_no_context | persona_consistency | 24 | 40 | 48 | 0.4286 | 0.3750 |
| proposed_vs_baseline_no_context | naturalness | 16 | 96 | 0 | 0.1429 | 0.1429 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 37 | 24 | 51 | 0.5580 | 0.6066 |
| proposed_vs_baseline_no_context | context_overlap | 76 | 35 | 1 | 0.6830 | 0.6847 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 17 | 24 | 71 | 0.4688 | 0.4146 |
| proposed_vs_baseline_no_context | persona_style | 14 | 26 | 72 | 0.4464 | 0.3500 |
| proposed_vs_baseline_no_context | distinct1 | 20 | 79 | 13 | 0.2366 | 0.2020 |
| proposed_vs_baseline_no_context | length_score | 15 | 95 | 2 | 0.1429 | 0.1364 |
| proposed_vs_baseline_no_context | sentence_score | 9 | 44 | 59 | 0.3438 | 0.1698 |
| proposed_vs_baseline_no_context | bertscore_f1 | 69 | 43 | 0 | 0.6161 | 0.6161 |
| proposed_vs_baseline_no_context | overall_quality | 45 | 67 | 0 | 0.4018 | 0.4018 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 71 | 41 | 0 | 0.6339 | 0.6339 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | 25 | 46 | 41 | 0.4062 | 0.3521 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | 15 | 96 | 1 | 0.1384 | 0.1351 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 41 | 21 | 50 | 0.5893 | 0.6613 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 75 | 36 | 1 | 0.6741 | 0.6757 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 19 | 29 | 64 | 0.4554 | 0.3958 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | 12 | 28 | 72 | 0.4286 | 0.3000 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | 18 | 81 | 13 | 0.2188 | 0.1818 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | 16 | 93 | 3 | 0.1562 | 0.1468 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | 12 | 45 | 55 | 0.3527 | 0.2105 |
| proposed_vs_baseline_no_context_phi3_latest | bertscore_f1 | 71 | 41 | 0 | 0.6339 | 0.6339 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | 44 | 68 | 0 | 0.3929 | 0.3929 |
| controlled_vs_proposed_raw | context_relevance | 102 | 10 | 0 | 0.9107 | 0.9107 |
| controlled_vs_proposed_raw | persona_consistency | 89 | 8 | 15 | 0.8616 | 0.9175 |
| controlled_vs_proposed_raw | naturalness | 82 | 29 | 1 | 0.7366 | 0.7387 |
| controlled_vs_proposed_raw | context_keyword_coverage | 100 | 7 | 5 | 0.9152 | 0.9346 |
| controlled_vs_proposed_raw | context_overlap | 97 | 15 | 0 | 0.8661 | 0.8661 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 88 | 7 | 17 | 0.8616 | 0.9263 |
| controlled_vs_proposed_raw | persona_style | 29 | 14 | 69 | 0.5670 | 0.6744 |
| controlled_vs_proposed_raw | distinct1 | 56 | 54 | 2 | 0.5089 | 0.5091 |
| controlled_vs_proposed_raw | length_score | 75 | 31 | 6 | 0.6964 | 0.7075 |
| controlled_vs_proposed_raw | sentence_score | 62 | 4 | 46 | 0.7589 | 0.9394 |
| controlled_vs_proposed_raw | bertscore_f1 | 59 | 53 | 0 | 0.5268 | 0.5268 |
| controlled_vs_proposed_raw | overall_quality | 105 | 7 | 0 | 0.9375 | 0.9375 |
| controlled_vs_candidate_no_context | context_relevance | 112 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 89 | 8 | 15 | 0.8616 | 0.9175 |
| controlled_vs_candidate_no_context | naturalness | 81 | 31 | 0 | 0.7232 | 0.7232 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 112 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 108 | 4 | 0 | 0.9643 | 0.9643 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 85 | 7 | 20 | 0.8482 | 0.9239 |
| controlled_vs_candidate_no_context | persona_style | 33 | 10 | 69 | 0.6027 | 0.7674 |
| controlled_vs_candidate_no_context | distinct1 | 59 | 47 | 6 | 0.5536 | 0.5566 |
| controlled_vs_candidate_no_context | length_score | 68 | 42 | 2 | 0.6161 | 0.6182 |
| controlled_vs_candidate_no_context | sentence_score | 67 | 8 | 37 | 0.7634 | 0.8933 |
| controlled_vs_candidate_no_context | bertscore_f1 | 80 | 32 | 0 | 0.7143 | 0.7143 |
| controlled_vs_candidate_no_context | overall_quality | 111 | 1 | 0 | 0.9911 | 0.9911 |
| controlled_vs_baseline_no_context | context_relevance | 112 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 89 | 8 | 15 | 0.8616 | 0.9175 |
| controlled_vs_baseline_no_context | naturalness | 56 | 56 | 0 | 0.5000 | 0.5000 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 111 | 0 | 1 | 0.9955 | 1.0000 |
| controlled_vs_baseline_no_context | context_overlap | 109 | 3 | 0 | 0.9732 | 0.9732 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 87 | 6 | 19 | 0.8616 | 0.9355 |
| controlled_vs_baseline_no_context | persona_style | 17 | 14 | 81 | 0.5134 | 0.5484 |
| controlled_vs_baseline_no_context | distinct1 | 17 | 91 | 4 | 0.1696 | 0.1574 |
| controlled_vs_baseline_no_context | length_score | 55 | 52 | 5 | 0.5134 | 0.5140 |
| controlled_vs_baseline_no_context | sentence_score | 31 | 8 | 73 | 0.6027 | 0.7949 |
| controlled_vs_baseline_no_context | bertscore_f1 | 67 | 45 | 0 | 0.5982 | 0.5982 |
| controlled_vs_baseline_no_context | overall_quality | 111 | 1 | 0 | 0.9911 | 0.9911 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 110 | 2 | 0 | 0.9821 | 0.9821 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 89 | 7 | 16 | 0.8661 | 0.9271 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | 52 | 60 | 0 | 0.4643 | 0.4643 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 110 | 2 | 0 | 0.9821 | 0.9821 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 108 | 4 | 0 | 0.9643 | 0.9643 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 87 | 4 | 21 | 0.8705 | 0.9560 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | 17 | 16 | 79 | 0.5045 | 0.5152 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | 12 | 97 | 3 | 0.1205 | 0.1101 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | 49 | 59 | 4 | 0.4554 | 0.4537 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 38 | 13 | 61 | 0.6116 | 0.7451 |
| controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 72 | 40 | 0 | 0.6429 | 0.6429 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 111 | 1 | 0 | 0.9911 | 0.9911 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 112 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 89 | 8 | 15 | 0.8616 | 0.9175 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 56 | 56 | 0 | 0.5000 | 0.5000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 111 | 0 | 1 | 0.9955 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 109 | 3 | 0 | 0.9732 | 0.9732 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 87 | 6 | 19 | 0.8616 | 0.9355 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 17 | 14 | 81 | 0.5134 | 0.5484 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | 17 | 91 | 4 | 0.1696 | 0.1574 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 55 | 52 | 5 | 0.5134 | 0.5140 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 31 | 8 | 73 | 0.6027 | 0.7949 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 67 | 45 | 0 | 0.5982 | 0.5982 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 111 | 1 | 0 | 0.9911 | 0.9911 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 110 | 2 | 0 | 0.9821 | 0.9821 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 89 | 7 | 16 | 0.8661 | 0.9271 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | 52 | 60 | 0 | 0.4643 | 0.4643 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 110 | 2 | 0 | 0.9821 | 0.9821 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 108 | 4 | 0 | 0.9643 | 0.9643 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 87 | 4 | 21 | 0.8705 | 0.9560 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | 17 | 16 | 79 | 0.5045 | 0.5152 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | 12 | 97 | 3 | 0.1205 | 0.1101 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | 49 | 59 | 4 | 0.4554 | 0.4537 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | 38 | 13 | 61 | 0.6116 | 0.7451 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 72 | 40 | 0 | 0.6429 | 0.6429 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 111 | 1 | 0 | 0.9911 | 0.9911 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.3125 | 0.3929 | 0.6071 |
| proposed_contextual | 0.0000 | 0.0000 | 0.4911 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.5536 | 0.0000 | 0.0000 |
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
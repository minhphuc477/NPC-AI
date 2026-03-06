# Proposal Alignment Evaluation Report

- Run ID: `20260306T190002Z`
- Generated: `2026-03-06T19:02:12.418178+00:00`
- Scenarios: `artifacts\proposal_control_tuning\full112_with_phi3latest\20260306T190002Z\scenarios.jsonl`
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
| proposed_contextual_controlled | 0.2764 (0.2632, 0.2897) | 0.3084 (0.2846, 0.3317) | 0.9062 (0.8947, 0.9168) | 0.3811 (0.3709, 0.3913) | 0.1000 |
| proposed_contextual_controlled_alt | 0.2505 (0.2318, 0.2699) | 0.3122 (0.2896, 0.3362) | 0.8976 (0.8854, 0.9103) | 0.3697 (0.3606, 0.3788) | 0.0953 |
| proposed_contextual | 0.0921 (0.0714, 0.1131) | 0.1575 (0.1327, 0.1834) | 0.8095 (0.7963, 0.8222) | 0.2371 (0.2218, 0.2540) | 0.0794 |
| candidate_no_context | 0.0226 (0.0168, 0.0300) | 0.1565 (0.1310, 0.1826) | 0.8169 (0.8016, 0.8333) | 0.2056 (0.1951, 0.2169) | 0.0382 |
| baseline_no_context | 0.0403 (0.0326, 0.0480) | 0.1933 (0.1708, 0.2154) | 0.8786 (0.8672, 0.8899) | 0.2363 (0.2281, 0.2447) | 0.0497 |
| baseline_no_context_phi3_latest | 0.0361 (0.0285, 0.0441) | 0.1855 (0.1614, 0.2111) | 0.8821 (0.8714, 0.8928) | 0.2336 (0.2244, 0.2432) | 0.0577 |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0695 | 3.0705 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0010 | 0.0062 |
| proposed_vs_candidate_no_context | naturalness | -0.0074 | -0.0091 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0874 | 5.4729 |
| proposed_vs_candidate_no_context | context_overlap | 0.0277 | 0.7263 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0040 | -0.0539 |
| proposed_vs_candidate_no_context | persona_style | 0.0207 | 0.0423 |
| proposed_vs_candidate_no_context | distinct1 | -0.0025 | -0.0027 |
| proposed_vs_candidate_no_context | length_score | -0.0348 | -0.1057 |
| proposed_vs_candidate_no_context | sentence_score | 0.0058 | 0.0077 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0413 | 1.0816 |
| proposed_vs_candidate_no_context | overall_quality | 0.0315 | 0.1533 |
| proposed_vs_baseline_no_context | context_relevance | 0.0518 | 1.2852 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0358 | -0.1853 |
| proposed_vs_baseline_no_context | naturalness | -0.0691 | -0.0787 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0622 | 1.5117 |
| proposed_vs_baseline_no_context | context_overlap | 0.0275 | 0.7184 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0236 | -0.2534 |
| proposed_vs_baseline_no_context | persona_style | -0.0849 | -0.1428 |
| proposed_vs_baseline_no_context | distinct1 | -0.0376 | -0.0386 |
| proposed_vs_baseline_no_context | length_score | -0.2074 | -0.4132 |
| proposed_vs_baseline_no_context | sentence_score | -0.1250 | -0.1413 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0297 | 0.5972 |
| proposed_vs_baseline_no_context | overall_quality | 0.0008 | 0.0033 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 0.0560 | 1.5537 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | -0.0281 | -0.1512 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | -0.0726 | -0.0823 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.0691 | 2.0154 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 0.0256 | 0.6368 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | -0.0161 | -0.1885 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | -0.0758 | -0.1294 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | -0.0430 | -0.0438 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | -0.2253 | -0.4333 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | -0.1031 | -0.1196 |
| proposed_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0218 | 0.3778 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | 0.0035 | 0.0150 |
| controlled_vs_proposed_raw | context_relevance | 0.1843 | 2.0008 |
| controlled_vs_proposed_raw | persona_consistency | 0.1509 | 0.9582 |
| controlled_vs_proposed_raw | naturalness | 0.0967 | 0.1195 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2443 | 2.3641 |
| controlled_vs_proposed_raw | context_overlap | 0.0443 | 0.6717 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1709 | 2.4626 |
| controlled_vs_proposed_raw | persona_style | 0.0710 | 0.1393 |
| controlled_vs_proposed_raw | distinct1 | -0.0011 | -0.0012 |
| controlled_vs_proposed_raw | length_score | 0.3905 | 1.3253 |
| controlled_vs_proposed_raw | sentence_score | 0.1906 | 0.2510 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0205 | 0.2587 |
| controlled_vs_proposed_raw | overall_quality | 0.1440 | 0.6072 |
| controlled_vs_candidate_no_context | context_relevance | 0.2537 | 11.2149 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1519 | 0.9704 |
| controlled_vs_candidate_no_context | naturalness | 0.0893 | 0.1094 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3316 | 20.7754 |
| controlled_vs_candidate_no_context | context_overlap | 0.0720 | 1.8858 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1669 | 2.2759 |
| controlled_vs_candidate_no_context | persona_style | 0.0917 | 0.1875 |
| controlled_vs_candidate_no_context | distinct1 | -0.0036 | -0.0038 |
| controlled_vs_candidate_no_context | length_score | 0.3557 | 1.0795 |
| controlled_vs_candidate_no_context | sentence_score | 0.1964 | 0.2607 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0618 | 1.6200 |
| controlled_vs_candidate_no_context | overall_quality | 0.1755 | 0.8536 |
| controlled_vs_baseline_no_context | context_relevance | 0.2361 | 5.8576 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1151 | 0.5953 |
| controlled_vs_baseline_no_context | naturalness | 0.0276 | 0.0315 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3065 | 7.4495 |
| controlled_vs_baseline_no_context | context_overlap | 0.0718 | 1.8727 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1473 | 1.5851 |
| controlled_vs_baseline_no_context | persona_style | -0.0139 | -0.0234 |
| controlled_vs_baseline_no_context | distinct1 | -0.0387 | -0.0397 |
| controlled_vs_baseline_no_context | length_score | 0.1830 | 0.3646 |
| controlled_vs_baseline_no_context | sentence_score | 0.0656 | 0.0742 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0502 | 1.0103 |
| controlled_vs_baseline_no_context | overall_quality | 0.1448 | 0.6126 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.2403 | 6.6633 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1228 | 0.6621 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | 0.0241 | 0.0274 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.3133 | 9.1441 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0699 | 1.7362 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1548 | 1.8100 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | -0.0048 | -0.0082 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0440 | -0.0450 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | 0.1652 | 0.3177 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.0875 | 0.1014 |
| controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0423 | 0.7342 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1475 | 0.6313 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0258 | -0.0935 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0038 | 0.0123 |
| controlled_alt_vs_controlled_default | naturalness | -0.0086 | -0.0095 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0343 | -0.0986 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0062 | -0.0562 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0028 | 0.0115 |
| controlled_alt_vs_controlled_default | persona_style | 0.0079 | 0.0137 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0018 | 0.0019 |
| controlled_alt_vs_controlled_default | length_score | -0.0390 | -0.0569 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0152 | -0.0160 |
| controlled_alt_vs_controlled_default | bertscore_f1 | -0.0046 | -0.0463 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0114 | -0.0300 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1584 | 1.7202 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1547 | 0.9823 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0881 | 0.1089 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2100 | 2.0325 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0381 | 0.5777 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1736 | 2.5025 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0789 | 0.1548 |
| controlled_alt_vs_proposed_raw | distinct1 | 0.0007 | 0.0007 |
| controlled_alt_vs_proposed_raw | length_score | 0.3515 | 1.1929 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1754 | 0.2310 |
| controlled_alt_vs_proposed_raw | bertscore_f1 | 0.0159 | 0.2003 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1326 | 0.5590 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2279 | 10.0726 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1557 | 0.9947 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0807 | 0.0988 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.2974 | 18.6288 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0658 | 1.7236 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1697 | 2.3136 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0996 | 0.2037 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0018 | -0.0019 |
| controlled_alt_vs_candidate_no_context | length_score | 0.3167 | 0.9612 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1812 | 0.2405 |
| controlled_alt_vs_candidate_no_context | bertscore_f1 | 0.0572 | 1.4986 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1641 | 0.7980 |
| controlled_alt_vs_baseline_no_context | context_relevance | 0.2102 | 5.2163 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 0.1189 | 0.6149 |
| controlled_alt_vs_baseline_no_context | naturalness | 0.0190 | 0.0217 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 0.2722 | 6.6166 |
| controlled_alt_vs_baseline_no_context | context_overlap | 0.0656 | 1.7111 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 0.1501 | 1.6148 |
| controlled_alt_vs_baseline_no_context | persona_style | -0.0060 | -0.0101 |
| controlled_alt_vs_baseline_no_context | distinct1 | -0.0370 | -0.0379 |
| controlled_alt_vs_baseline_no_context | length_score | 0.1440 | 0.2869 |
| controlled_alt_vs_baseline_no_context | sentence_score | 0.0504 | 0.0570 |
| controlled_alt_vs_baseline_no_context | bertscore_f1 | 0.0456 | 0.9171 |
| controlled_alt_vs_baseline_no_context | overall_quality | 0.1333 | 0.5642 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_relevance | 0.2145 | 5.9466 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1266 | 0.6826 |
| controlled_alt_vs_baseline_no_context_phi3_latest | naturalness | 0.0155 | 0.0176 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.2791 | 8.1441 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_overlap | 0.0637 | 1.5824 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1575 | 1.8424 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_style | 0.0031 | 0.0053 |
| controlled_alt_vs_baseline_no_context_phi3_latest | distinct1 | -0.0423 | -0.0431 |
| controlled_alt_vs_baseline_no_context_phi3_latest | length_score | 0.1262 | 0.2427 |
| controlled_alt_vs_baseline_no_context_phi3_latest | sentence_score | 0.0723 | 0.0839 |
| controlled_alt_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0377 | 0.6538 |
| controlled_alt_vs_baseline_no_context_phi3_latest | overall_quality | 0.1361 | 0.5823 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2361 | 5.8576 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1151 | 0.5953 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 0.0276 | 0.0315 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3065 | 7.4495 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0718 | 1.8727 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1473 | 1.5851 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | -0.0139 | -0.0234 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0387 | -0.0397 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.1830 | 0.3646 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0656 | 0.0742 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0502 | 1.0103 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1448 | 0.6126 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.2403 | 6.6633 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1228 | 0.6621 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | 0.0241 | 0.0274 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.3133 | 9.1441 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0699 | 1.7362 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1548 | 1.8100 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | -0.0048 | -0.0082 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0440 | -0.0450 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | 0.1652 | 0.3177 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.0875 | 0.1014 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0423 | 0.7342 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1475 | 0.6313 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0695 | (0.0504, 0.0899) | 0.0000 | 0.0695 | (0.0281, 0.1143) | 0.0000 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0010 | (-0.0220, 0.0237) | 0.4443 | 0.0010 | (-0.0184, 0.0200) | 0.4400 |
| proposed_vs_candidate_no_context | naturalness | -0.0074 | (-0.0221, 0.0074) | 0.8407 | -0.0074 | (-0.0296, 0.0098) | 0.7580 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0874 | (0.0619, 0.1147) | 0.0000 | 0.0874 | (0.0377, 0.1454) | 0.0000 |
| proposed_vs_candidate_no_context | context_overlap | 0.0277 | (0.0183, 0.0382) | 0.0000 | 0.0277 | (0.0087, 0.0528) | 0.0000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0040 | (-0.0319, 0.0221) | 0.6080 | -0.0040 | (-0.0213, 0.0114) | 0.6583 |
| proposed_vs_candidate_no_context | persona_style | 0.0207 | (-0.0104, 0.0556) | 0.0993 | 0.0207 | (-0.0140, 0.0778) | 0.3393 |
| proposed_vs_candidate_no_context | distinct1 | -0.0025 | (-0.0100, 0.0050) | 0.7437 | -0.0025 | (-0.0118, 0.0061) | 0.6817 |
| proposed_vs_candidate_no_context | length_score | -0.0348 | (-0.0958, 0.0259) | 0.8700 | -0.0348 | (-0.1226, 0.0402) | 0.7820 |
| proposed_vs_candidate_no_context | sentence_score | 0.0058 | (-0.0281, 0.0402) | 0.4083 | 0.0058 | (-0.0263, 0.0402) | 0.3927 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0413 | (0.0288, 0.0554) | 0.0000 | 0.0413 | (0.0173, 0.0700) | 0.0000 |
| proposed_vs_candidate_no_context | overall_quality | 0.0315 | (0.0171, 0.0455) | 0.0003 | 0.0315 | (0.0077, 0.0581) | 0.0020 |
| proposed_vs_baseline_no_context | context_relevance | 0.0518 | (0.0283, 0.0774) | 0.0000 | 0.0518 | (0.0064, 0.1028) | 0.0100 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0358 | (-0.0647, -0.0064) | 0.9910 | -0.0358 | (-0.0835, 0.0139) | 0.9170 |
| proposed_vs_baseline_no_context | naturalness | -0.0691 | (-0.0856, -0.0513) | 1.0000 | -0.0691 | (-0.1038, -0.0351) | 1.0000 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0622 | (0.0342, 0.0921) | 0.0000 | 0.0622 | (0.0046, 0.1224) | 0.0157 |
| proposed_vs_baseline_no_context | context_overlap | 0.0275 | (0.0176, 0.0387) | 0.0000 | 0.0275 | (0.0080, 0.0515) | 0.0007 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0236 | (-0.0564, 0.0097) | 0.9117 | -0.0236 | (-0.0671, 0.0208) | 0.8547 |
| proposed_vs_baseline_no_context | persona_style | -0.0849 | (-0.1294, -0.0428) | 1.0000 | -0.0849 | (-0.2197, 0.0211) | 0.8993 |
| proposed_vs_baseline_no_context | distinct1 | -0.0376 | (-0.0456, -0.0289) | 1.0000 | -0.0376 | (-0.0505, -0.0264) | 1.0000 |
| proposed_vs_baseline_no_context | length_score | -0.2074 | (-0.2723, -0.1426) | 1.0000 | -0.2074 | (-0.3321, -0.0795) | 0.9997 |
| proposed_vs_baseline_no_context | sentence_score | -0.1250 | (-0.1656, -0.0813) | 1.0000 | -0.1250 | (-0.2062, -0.0406) | 0.9983 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0297 | (0.0160, 0.0436) | 0.0000 | 0.0297 | (0.0066, 0.0528) | 0.0047 |
| proposed_vs_baseline_no_context | overall_quality | 0.0008 | (-0.0172, 0.0190) | 0.4810 | 0.0008 | (-0.0363, 0.0413) | 0.4953 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 0.0560 | (0.0348, 0.0793) | 0.0000 | 0.0560 | (0.0123, 0.1048) | 0.0030 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | -0.0281 | (-0.0582, 0.0009) | 0.9703 | -0.0281 | (-0.0804, 0.0231) | 0.8603 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | -0.0726 | (-0.0888, -0.0555) | 1.0000 | -0.0726 | (-0.1073, -0.0363) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.0691 | (0.0403, 0.0989) | 0.0000 | 0.0691 | (0.0140, 0.1346) | 0.0050 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 0.0256 | (0.0149, 0.0373) | 0.0000 | 0.0256 | (0.0063, 0.0530) | 0.0010 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | -0.0161 | (-0.0488, 0.0154) | 0.8387 | -0.0161 | (-0.0557, 0.0276) | 0.7660 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | -0.0758 | (-0.1154, -0.0387) | 1.0000 | -0.0758 | (-0.2064, 0.0204) | 0.9013 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | -0.0430 | (-0.0511, -0.0345) | 1.0000 | -0.0430 | (-0.0583, -0.0285) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | -0.2253 | (-0.2893, -0.1559) | 1.0000 | -0.2253 | (-0.3455, -0.0854) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | -0.1031 | (-0.1437, -0.0594) | 1.0000 | -0.1031 | (-0.1719, -0.0281) | 0.9977 |
| proposed_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0218 | (0.0074, 0.0366) | 0.0013 | 0.0218 | (-0.0085, 0.0501) | 0.0840 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | 0.0035 | (-0.0130, 0.0201) | 0.3443 | 0.0035 | (-0.0318, 0.0417) | 0.4453 |
| controlled_vs_proposed_raw | context_relevance | 0.1843 | (0.1647, 0.2025) | 0.0000 | 0.1843 | (0.1519, 0.2148) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.1509 | (0.1257, 0.1746) | 0.0000 | 0.1509 | (0.0965, 0.2017) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0967 | (0.0773, 0.1167) | 0.0000 | 0.0967 | (0.0475, 0.1448) | 0.0000 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2443 | (0.2190, 0.2679) | 0.0000 | 0.2443 | (0.2062, 0.2826) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0443 | (0.0337, 0.0535) | 0.0000 | 0.0443 | (0.0247, 0.0610) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1709 | (0.1421, 0.1977) | 0.0000 | 0.1709 | (0.1160, 0.2178) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.0710 | (0.0365, 0.1092) | 0.0000 | 0.0710 | (-0.0160, 0.1963) | 0.0963 |
| controlled_vs_proposed_raw | distinct1 | -0.0011 | (-0.0097, 0.0068) | 0.5963 | -0.0011 | (-0.0179, 0.0166) | 0.5823 |
| controlled_vs_proposed_raw | length_score | 0.3905 | (0.3101, 0.4658) | 0.0000 | 0.3905 | (0.2063, 0.5694) | 0.0000 |
| controlled_vs_proposed_raw | sentence_score | 0.1906 | (0.1500, 0.2281) | 0.0000 | 0.1906 | (0.1125, 0.2719) | 0.0000 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0205 | (0.0088, 0.0327) | 0.0000 | 0.0205 | (-0.0026, 0.0473) | 0.0480 |
| controlled_vs_proposed_raw | overall_quality | 0.1440 | (0.1298, 0.1584) | 0.0000 | 0.1440 | (0.1165, 0.1763) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2537 | (0.2406, 0.2678) | 0.0000 | 0.2537 | (0.2296, 0.2767) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1519 | (0.1211, 0.1820) | 0.0000 | 0.1519 | (0.0874, 0.2060) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.0893 | (0.0672, 0.1093) | 0.0000 | 0.0893 | (0.0298, 0.1455) | 0.0003 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3316 | (0.3138, 0.3501) | 0.0000 | 0.3316 | (0.2998, 0.3625) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0720 | (0.0663, 0.0781) | 0.0000 | 0.0720 | (0.0592, 0.0828) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1669 | (0.1325, 0.2006) | 0.0000 | 0.1669 | (0.1057, 0.2155) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.0917 | (0.0505, 0.1353) | 0.0000 | 0.0917 | (-0.0121, 0.2158) | 0.0413 |
| controlled_vs_candidate_no_context | distinct1 | -0.0036 | (-0.0118, 0.0050) | 0.7903 | -0.0036 | (-0.0216, 0.0155) | 0.6607 |
| controlled_vs_candidate_no_context | length_score | 0.3557 | (0.2693, 0.4414) | 0.0000 | 0.3557 | (0.1134, 0.5637) | 0.0017 |
| controlled_vs_candidate_no_context | sentence_score | 0.1964 | (0.1589, 0.2335) | 0.0000 | 0.1964 | (0.1094, 0.2804) | 0.0000 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0618 | (0.0480, 0.0748) | 0.0000 | 0.0618 | (0.0366, 0.0882) | 0.0000 |
| controlled_vs_candidate_no_context | overall_quality | 0.1755 | (0.1628, 0.1881) | 0.0000 | 0.1755 | (0.1454, 0.2000) | 0.0000 |
| controlled_vs_baseline_no_context | context_relevance | 0.2361 | (0.2221, 0.2515) | 0.0000 | 0.2361 | (0.2159, 0.2564) | 0.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1151 | (0.0840, 0.1453) | 0.0000 | 0.1151 | (0.0552, 0.1672) | 0.0003 |
| controlled_vs_baseline_no_context | naturalness | 0.0276 | (0.0105, 0.0442) | 0.0007 | 0.0276 | (-0.0111, 0.0567) | 0.0700 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3065 | (0.2889, 0.3261) | 0.0000 | 0.3065 | (0.2814, 0.3328) | 0.0000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0718 | (0.0656, 0.0783) | 0.0000 | 0.0718 | (0.0589, 0.0829) | 0.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1473 | (0.1112, 0.1837) | 0.0000 | 0.1473 | (0.0729, 0.2090) | 0.0000 |
| controlled_vs_baseline_no_context | persona_style | -0.0139 | (-0.0344, 0.0068) | 0.9063 | -0.0139 | (-0.0545, 0.0244) | 0.7717 |
| controlled_vs_baseline_no_context | distinct1 | -0.0387 | (-0.0462, -0.0307) | 1.0000 | -0.0387 | (-0.0495, -0.0266) | 1.0000 |
| controlled_vs_baseline_no_context | length_score | 0.1830 | (0.1083, 0.2557) | 0.0000 | 0.1830 | (0.0104, 0.3069) | 0.0203 |
| controlled_vs_baseline_no_context | sentence_score | 0.0656 | (0.0281, 0.1031) | 0.0000 | 0.0656 | (-0.0125, 0.1500) | 0.0593 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0502 | (0.0376, 0.0632) | 0.0000 | 0.0502 | (0.0274, 0.0774) | 0.0000 |
| controlled_vs_baseline_no_context | overall_quality | 0.1448 | (0.1328, 0.1572) | 0.0000 | 0.1448 | (0.1181, 0.1675) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.2403 | (0.2268, 0.2543) | 0.0000 | 0.2403 | (0.2212, 0.2609) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1228 | (0.0946, 0.1501) | 0.0000 | 0.1228 | (0.0738, 0.1653) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | 0.0241 | (0.0087, 0.0382) | 0.0020 | 0.0241 | (0.0002, 0.0476) | 0.0237 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.3133 | (0.2959, 0.3318) | 0.0000 | 0.3133 | (0.2895, 0.3390) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0699 | (0.0634, 0.0764) | 0.0000 | 0.0699 | (0.0561, 0.0821) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1548 | (0.1204, 0.1872) | 0.0000 | 0.1548 | (0.0920, 0.2081) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | -0.0048 | (-0.0177, 0.0076) | 0.7747 | -0.0048 | (-0.0160, 0.0044) | 0.8563 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0440 | (-0.0517, -0.0360) | 1.0000 | -0.0440 | (-0.0584, -0.0283) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | 0.1652 | (0.0997, 0.2292) | 0.0000 | 0.1652 | (0.0449, 0.2583) | 0.0047 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.0875 | (0.0500, 0.1250) | 0.0000 | 0.0875 | (0.0094, 0.1625) | 0.0133 |
| controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0423 | (0.0302, 0.0543) | 0.0000 | 0.0423 | (0.0160, 0.0698) | 0.0003 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1475 | (0.1359, 0.1589) | 0.0000 | 0.1475 | (0.1235, 0.1695) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0258 | (-0.0444, -0.0082) | 0.9983 | -0.0258 | (-0.0493, -0.0060) | 0.9993 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0038 | (-0.0187, 0.0269) | 0.3753 | 0.0038 | (-0.0255, 0.0342) | 0.3853 |
| controlled_alt_vs_controlled_default | naturalness | -0.0086 | (-0.0203, 0.0026) | 0.9363 | -0.0086 | (-0.0155, -0.0005) | 0.9810 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0343 | (-0.0583, -0.0104) | 0.9973 | -0.0343 | (-0.0662, -0.0075) | 0.9963 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0062 | (-0.0128, 0.0008) | 0.9583 | -0.0062 | (-0.0129, -0.0008) | 0.9893 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0028 | (-0.0254, 0.0292) | 0.4233 | 0.0028 | (-0.0330, 0.0376) | 0.4437 |
| controlled_alt_vs_controlled_default | persona_style | 0.0079 | (-0.0077, 0.0253) | 0.1787 | 0.0079 | (-0.0111, 0.0342) | 0.2867 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0018 | (-0.0051, 0.0085) | 0.3097 | 0.0018 | (-0.0024, 0.0076) | 0.2800 |
| controlled_alt_vs_controlled_default | length_score | -0.0390 | (-0.0863, 0.0077) | 0.9480 | -0.0390 | (-0.0661, 0.0021) | 0.9627 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0152 | (-0.0460, 0.0125) | 0.8560 | -0.0152 | (-0.0616, 0.0286) | 0.7387 |
| controlled_alt_vs_controlled_default | bertscore_f1 | -0.0046 | (-0.0144, 0.0047) | 0.8397 | -0.0046 | (-0.0149, 0.0029) | 0.8397 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0114 | (-0.0216, -0.0019) | 0.9920 | -0.0114 | (-0.0225, -0.0017) | 0.9917 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1584 | (0.1357, 0.1809) | 0.0000 | 0.1584 | (0.1134, 0.2030) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1547 | (0.1289, 0.1790) | 0.0000 | 0.1547 | (0.1218, 0.1916) | 0.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0881 | (0.0670, 0.1074) | 0.0000 | 0.0881 | (0.0448, 0.1307) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2100 | (0.1802, 0.2401) | 0.0000 | 0.2100 | (0.1506, 0.2668) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0381 | (0.0275, 0.0477) | 0.0000 | 0.0381 | (0.0165, 0.0554) | 0.0007 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1736 | (0.1446, 0.2017) | 0.0000 | 0.1736 | (0.1436, 0.2013) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0789 | (0.0431, 0.1189) | 0.0000 | 0.0789 | (-0.0057, 0.2000) | 0.0787 |
| controlled_alt_vs_proposed_raw | distinct1 | 0.0007 | (-0.0078, 0.0087) | 0.4277 | 0.0007 | (-0.0118, 0.0157) | 0.4957 |
| controlled_alt_vs_proposed_raw | length_score | 0.3515 | (0.2723, 0.4318) | 0.0000 | 0.3515 | (0.1964, 0.5057) | 0.0000 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1754 | (0.1353, 0.2129) | 0.0000 | 0.1754 | (0.0879, 0.2531) | 0.0000 |
| controlled_alt_vs_proposed_raw | bertscore_f1 | 0.0159 | (0.0030, 0.0280) | 0.0087 | 0.0159 | (-0.0078, 0.0374) | 0.0933 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1326 | (0.1173, 0.1475) | 0.0000 | 0.1326 | (0.0996, 0.1651) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2279 | (0.2076, 0.2482) | 0.0000 | 0.2279 | (0.1901, 0.2604) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1557 | (0.1281, 0.1832) | 0.0000 | 0.1557 | (0.1133, 0.1958) | 0.0000 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0807 | (0.0593, 0.1012) | 0.0000 | 0.0807 | (0.0306, 0.1325) | 0.0017 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.2974 | (0.2698, 0.3236) | 0.0000 | 0.2974 | (0.2479, 0.3407) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0658 | (0.0586, 0.0734) | 0.0000 | 0.0658 | (0.0542, 0.0773) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1697 | (0.1404, 0.1995) | 0.0000 | 0.1697 | (0.1348, 0.2022) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0996 | (0.0589, 0.1429) | 0.0000 | 0.0996 | (-0.0000, 0.2201) | 0.0253 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0018 | (-0.0111, 0.0072) | 0.6470 | -0.0018 | (-0.0182, 0.0155) | 0.5860 |
| controlled_alt_vs_candidate_no_context | length_score | 0.3167 | (0.2339, 0.3994) | 0.0000 | 0.3167 | (0.1145, 0.5018) | 0.0013 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1812 | (0.1406, 0.2188) | 0.0000 | 0.1812 | (0.0996, 0.2567) | 0.0000 |
| controlled_alt_vs_candidate_no_context | bertscore_f1 | 0.0572 | (0.0448, 0.0700) | 0.0000 | 0.0572 | (0.0328, 0.0816) | 0.0000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1641 | (0.1506, 0.1775) | 0.0000 | 0.1641 | (0.1335, 0.1899) | 0.0000 |
| controlled_alt_vs_baseline_no_context | context_relevance | 0.2102 | (0.1904, 0.2307) | 0.0000 | 0.2102 | (0.1841, 0.2331) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 0.1189 | (0.0924, 0.1455) | 0.0000 | 0.1189 | (0.0794, 0.1536) | 0.0000 |
| controlled_alt_vs_baseline_no_context | naturalness | 0.0190 | (0.0017, 0.0355) | 0.0153 | 0.0190 | (-0.0122, 0.0453) | 0.1080 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 0.2722 | (0.2454, 0.2972) | 0.0000 | 0.2722 | (0.2354, 0.3032) | 0.0000 |
| controlled_alt_vs_baseline_no_context | context_overlap | 0.0656 | (0.0583, 0.0736) | 0.0000 | 0.0656 | (0.0547, 0.0752) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 0.1501 | (0.1185, 0.1833) | 0.0000 | 0.1501 | (0.0995, 0.1907) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_style | -0.0060 | (-0.0299, 0.0180) | 0.6910 | -0.0060 | (-0.0458, 0.0245) | 0.5997 |
| controlled_alt_vs_baseline_no_context | distinct1 | -0.0370 | (-0.0452, -0.0287) | 1.0000 | -0.0370 | (-0.0460, -0.0274) | 1.0000 |
| controlled_alt_vs_baseline_no_context | length_score | 0.1440 | (0.0696, 0.2158) | 0.0003 | 0.1440 | (-0.0024, 0.2470) | 0.0260 |
| controlled_alt_vs_baseline_no_context | sentence_score | 0.0504 | (0.0125, 0.0906) | 0.0043 | 0.0504 | (-0.0219, 0.1281) | 0.0923 |
| controlled_alt_vs_baseline_no_context | bertscore_f1 | 0.0456 | (0.0324, 0.0585) | 0.0000 | 0.0456 | (0.0237, 0.0686) | 0.0000 |
| controlled_alt_vs_baseline_no_context | overall_quality | 0.1333 | (0.1209, 0.1452) | 0.0000 | 0.1333 | (0.1131, 0.1494) | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_relevance | 0.2145 | (0.1952, 0.2338) | 0.0000 | 0.2145 | (0.1871, 0.2360) | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1266 | (0.1006, 0.1515) | 0.0000 | 0.1266 | (0.0923, 0.1583) | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | naturalness | 0.0155 | (-0.0003, 0.0310) | 0.0270 | 0.0155 | (-0.0032, 0.0360) | 0.0550 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.2791 | (0.2528, 0.3060) | 0.0000 | 0.2791 | (0.2443, 0.3075) | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_overlap | 0.0637 | (0.0564, 0.0710) | 0.0000 | 0.0637 | (0.0519, 0.0750) | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1575 | (0.1261, 0.1895) | 0.0000 | 0.1575 | (0.1161, 0.1945) | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_style | 0.0031 | (-0.0148, 0.0225) | 0.3697 | 0.0031 | (-0.0205, 0.0362) | 0.4360 |
| controlled_alt_vs_baseline_no_context_phi3_latest | distinct1 | -0.0423 | (-0.0501, -0.0339) | 1.0000 | -0.0423 | (-0.0551, -0.0284) | 1.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | length_score | 0.1262 | (0.0580, 0.1938) | 0.0000 | 0.1262 | (0.0464, 0.1988) | 0.0013 |
| controlled_alt_vs_baseline_no_context_phi3_latest | sentence_score | 0.0723 | (0.0290, 0.1125) | 0.0000 | 0.0723 | (-0.0058, 0.1442) | 0.0350 |
| controlled_alt_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0377 | (0.0249, 0.0513) | 0.0000 | 0.0377 | (0.0153, 0.0627) | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | overall_quality | 0.1361 | (0.1235, 0.1482) | 0.0000 | 0.1361 | (0.1211, 0.1488) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2361 | (0.2221, 0.2504) | 0.0000 | 0.2361 | (0.2162, 0.2566) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1151 | (0.0851, 0.1443) | 0.0000 | 0.1151 | (0.0534, 0.1684) | 0.0003 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 0.0276 | (0.0101, 0.0438) | 0.0000 | 0.0276 | (-0.0114, 0.0564) | 0.0767 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3065 | (0.2886, 0.3257) | 0.0000 | 0.3065 | (0.2812, 0.3327) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0718 | (0.0651, 0.0783) | 0.0000 | 0.0718 | (0.0583, 0.0836) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1473 | (0.1101, 0.1837) | 0.0000 | 0.1473 | (0.0744, 0.2092) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | -0.0139 | (-0.0334, 0.0065) | 0.9150 | -0.0139 | (-0.0551, 0.0244) | 0.7413 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0387 | (-0.0468, -0.0309) | 1.0000 | -0.0387 | (-0.0496, -0.0267) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.1830 | (0.1101, 0.2577) | 0.0000 | 0.1830 | (-0.0024, 0.3057) | 0.0260 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0656 | (0.0281, 0.1031) | 0.0010 | 0.0656 | (-0.0095, 0.1562) | 0.0600 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0502 | (0.0377, 0.0635) | 0.0000 | 0.0502 | (0.0270, 0.0763) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1448 | (0.1314, 0.1571) | 0.0000 | 0.1448 | (0.1182, 0.1670) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.2403 | (0.2269, 0.2544) | 0.0000 | 0.2403 | (0.2213, 0.2609) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1228 | (0.0949, 0.1495) | 0.0000 | 0.1228 | (0.0724, 0.1652) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | 0.0241 | (0.0101, 0.0383) | 0.0010 | 0.0241 | (0.0003, 0.0478) | 0.0240 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.3133 | (0.2952, 0.3318) | 0.0000 | 0.3133 | (0.2894, 0.3387) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0699 | (0.0637, 0.0761) | 0.0000 | 0.0699 | (0.0571, 0.0823) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1548 | (0.1194, 0.1878) | 0.0000 | 0.1548 | (0.0922, 0.2079) | 0.0003 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | -0.0048 | (-0.0177, 0.0064) | 0.7817 | -0.0048 | (-0.0160, 0.0044) | 0.8677 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0440 | (-0.0520, -0.0359) | 1.0000 | -0.0440 | (-0.0577, -0.0282) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | 0.1652 | (0.1009, 0.2274) | 0.0000 | 0.1652 | (0.0423, 0.2616) | 0.0030 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.0875 | (0.0500, 0.1250) | 0.0000 | 0.0875 | (0.0125, 0.1656) | 0.0133 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0423 | (0.0304, 0.0551) | 0.0000 | 0.0423 | (0.0173, 0.0690) | 0.0003 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1475 | (0.1356, 0.1591) | 0.0000 | 0.1475 | (0.1252, 0.1686) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 53 | 11 | 48 | 0.6875 | 0.8281 |
| proposed_vs_candidate_no_context | persona_consistency | 23 | 18 | 71 | 0.5223 | 0.5610 |
| proposed_vs_candidate_no_context | naturalness | 29 | 36 | 47 | 0.4688 | 0.4462 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 41 | 2 | 69 | 0.6741 | 0.9535 |
| proposed_vs_candidate_no_context | context_overlap | 51 | 13 | 48 | 0.6696 | 0.7969 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 12 | 11 | 89 | 0.5045 | 0.5217 |
| proposed_vs_candidate_no_context | persona_style | 15 | 10 | 87 | 0.5223 | 0.6000 |
| proposed_vs_candidate_no_context | distinct1 | 24 | 34 | 54 | 0.4554 | 0.4138 |
| proposed_vs_candidate_no_context | length_score | 30 | 34 | 48 | 0.4821 | 0.4688 |
| proposed_vs_candidate_no_context | sentence_score | 17 | 15 | 80 | 0.5089 | 0.5312 |
| proposed_vs_candidate_no_context | bertscore_f1 | 66 | 15 | 31 | 0.7277 | 0.8148 |
| proposed_vs_candidate_no_context | overall_quality | 59 | 22 | 31 | 0.6652 | 0.7284 |
| proposed_vs_baseline_no_context | context_relevance | 64 | 48 | 0 | 0.5714 | 0.5714 |
| proposed_vs_baseline_no_context | persona_consistency | 21 | 51 | 40 | 0.3661 | 0.2917 |
| proposed_vs_baseline_no_context | naturalness | 23 | 88 | 1 | 0.2098 | 0.2072 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 40 | 26 | 46 | 0.5625 | 0.6061 |
| proposed_vs_baseline_no_context | context_overlap | 74 | 38 | 0 | 0.6607 | 0.6607 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 13 | 29 | 70 | 0.4286 | 0.3095 |
| proposed_vs_baseline_no_context | persona_style | 13 | 33 | 66 | 0.4107 | 0.2826 |
| proposed_vs_baseline_no_context | distinct1 | 19 | 82 | 11 | 0.2188 | 0.1881 |
| proposed_vs_baseline_no_context | length_score | 30 | 79 | 3 | 0.2812 | 0.2752 |
| proposed_vs_baseline_no_context | sentence_score | 10 | 50 | 52 | 0.3214 | 0.1667 |
| proposed_vs_baseline_no_context | bertscore_f1 | 69 | 43 | 0 | 0.6161 | 0.6161 |
| proposed_vs_baseline_no_context | overall_quality | 46 | 66 | 0 | 0.4107 | 0.4107 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 67 | 43 | 2 | 0.6071 | 0.6091 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | 17 | 43 | 52 | 0.3839 | 0.2833 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | 23 | 89 | 0 | 0.2054 | 0.2054 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 39 | 18 | 55 | 0.5938 | 0.6842 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 70 | 40 | 2 | 0.6339 | 0.6364 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 15 | 24 | 73 | 0.4598 | 0.3846 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | 6 | 26 | 80 | 0.4107 | 0.1875 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | 16 | 84 | 12 | 0.1964 | 0.1600 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | 27 | 82 | 3 | 0.2545 | 0.2477 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | 12 | 45 | 55 | 0.3527 | 0.2105 |
| proposed_vs_baseline_no_context_phi3_latest | bertscore_f1 | 63 | 49 | 0 | 0.5625 | 0.5625 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | 46 | 66 | 0 | 0.4107 | 0.4107 |
| controlled_vs_proposed_raw | context_relevance | 102 | 10 | 0 | 0.9107 | 0.9107 |
| controlled_vs_proposed_raw | persona_consistency | 92 | 6 | 14 | 0.8839 | 0.9388 |
| controlled_vs_proposed_raw | naturalness | 88 | 24 | 0 | 0.7857 | 0.7857 |
| controlled_vs_proposed_raw | context_keyword_coverage | 100 | 5 | 7 | 0.9241 | 0.9524 |
| controlled_vs_proposed_raw | context_overlap | 99 | 12 | 1 | 0.8884 | 0.8919 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 92 | 6 | 14 | 0.8839 | 0.9388 |
| controlled_vs_proposed_raw | persona_style | 23 | 6 | 83 | 0.5759 | 0.7931 |
| controlled_vs_proposed_raw | distinct1 | 66 | 44 | 2 | 0.5982 | 0.6000 |
| controlled_vs_proposed_raw | length_score | 89 | 21 | 2 | 0.8036 | 0.8091 |
| controlled_vs_proposed_raw | sentence_score | 68 | 7 | 37 | 0.7723 | 0.9067 |
| controlled_vs_proposed_raw | bertscore_f1 | 75 | 37 | 0 | 0.6696 | 0.6696 |
| controlled_vs_proposed_raw | overall_quality | 107 | 5 | 0 | 0.9554 | 0.9554 |
| controlled_vs_candidate_no_context | context_relevance | 112 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 89 | 8 | 15 | 0.8616 | 0.9175 |
| controlled_vs_candidate_no_context | naturalness | 84 | 28 | 0 | 0.7500 | 0.7500 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 112 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 112 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 88 | 8 | 16 | 0.8571 | 0.9167 |
| controlled_vs_candidate_no_context | persona_style | 33 | 6 | 73 | 0.6205 | 0.8462 |
| controlled_vs_candidate_no_context | distinct1 | 60 | 49 | 3 | 0.5491 | 0.5505 |
| controlled_vs_candidate_no_context | length_score | 86 | 24 | 2 | 0.7768 | 0.7818 |
| controlled_vs_candidate_no_context | sentence_score | 67 | 5 | 40 | 0.7768 | 0.9306 |
| controlled_vs_candidate_no_context | bertscore_f1 | 97 | 15 | 0 | 0.8661 | 0.8661 |
| controlled_vs_candidate_no_context | overall_quality | 108 | 4 | 0 | 0.9643 | 0.9643 |
| controlled_vs_baseline_no_context | context_relevance | 112 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 81 | 13 | 18 | 0.8036 | 0.8617 |
| controlled_vs_baseline_no_context | naturalness | 72 | 40 | 0 | 0.6429 | 0.6429 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 112 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | context_overlap | 112 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 79 | 10 | 23 | 0.8080 | 0.8876 |
| controlled_vs_baseline_no_context | persona_style | 10 | 14 | 88 | 0.4821 | 0.4167 |
| controlled_vs_baseline_no_context | distinct1 | 19 | 89 | 4 | 0.1875 | 0.1759 |
| controlled_vs_baseline_no_context | length_score | 78 | 30 | 4 | 0.7143 | 0.7222 |
| controlled_vs_baseline_no_context | sentence_score | 32 | 11 | 69 | 0.5938 | 0.7442 |
| controlled_vs_baseline_no_context | bertscore_f1 | 84 | 28 | 0 | 0.7500 | 0.7500 |
| controlled_vs_baseline_no_context | overall_quality | 110 | 2 | 0 | 0.9821 | 0.9821 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 112 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 87 | 11 | 14 | 0.8393 | 0.8878 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | 73 | 39 | 0 | 0.6518 | 0.6518 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 112 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 111 | 1 | 0 | 0.9911 | 0.9911 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 87 | 11 | 14 | 0.8393 | 0.8878 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | 5 | 7 | 100 | 0.4911 | 0.4167 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | 19 | 90 | 3 | 0.1830 | 0.1743 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | 82 | 27 | 3 | 0.7455 | 0.7523 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 37 | 9 | 66 | 0.6250 | 0.8043 |
| controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 77 | 35 | 0 | 0.6875 | 0.6875 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 111 | 1 | 0 | 0.9911 | 0.9911 |
| controlled_alt_vs_controlled_default | context_relevance | 23 | 42 | 47 | 0.4152 | 0.3538 |
| controlled_alt_vs_controlled_default | persona_consistency | 18 | 23 | 71 | 0.4777 | 0.4390 |
| controlled_alt_vs_controlled_default | naturalness | 31 | 34 | 47 | 0.4866 | 0.4769 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 18 | 33 | 61 | 0.4330 | 0.3529 |
| controlled_alt_vs_controlled_default | context_overlap | 22 | 42 | 48 | 0.4107 | 0.3438 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 17 | 19 | 76 | 0.4911 | 0.4722 |
| controlled_alt_vs_controlled_default | persona_style | 9 | 7 | 96 | 0.5089 | 0.5625 |
| controlled_alt_vs_controlled_default | distinct1 | 34 | 30 | 48 | 0.5179 | 0.5312 |
| controlled_alt_vs_controlled_default | length_score | 25 | 34 | 53 | 0.4598 | 0.4237 |
| controlled_alt_vs_controlled_default | sentence_score | 8 | 12 | 92 | 0.4821 | 0.4000 |
| controlled_alt_vs_controlled_default | bertscore_f1 | 26 | 43 | 43 | 0.4241 | 0.3768 |
| controlled_alt_vs_controlled_default | overall_quality | 26 | 43 | 43 | 0.4241 | 0.3768 |
| controlled_alt_vs_proposed_raw | context_relevance | 98 | 13 | 1 | 0.8795 | 0.8829 |
| controlled_alt_vs_proposed_raw | persona_consistency | 88 | 8 | 16 | 0.8571 | 0.9167 |
| controlled_alt_vs_proposed_raw | naturalness | 83 | 29 | 0 | 0.7411 | 0.7411 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 90 | 9 | 13 | 0.8616 | 0.9091 |
| controlled_alt_vs_proposed_raw | context_overlap | 94 | 16 | 2 | 0.8482 | 0.8545 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 88 | 6 | 18 | 0.8661 | 0.9362 |
| controlled_alt_vs_proposed_raw | persona_style | 26 | 10 | 76 | 0.5714 | 0.7222 |
| controlled_alt_vs_proposed_raw | distinct1 | 66 | 43 | 3 | 0.6027 | 0.6055 |
| controlled_alt_vs_proposed_raw | length_score | 83 | 28 | 1 | 0.7455 | 0.7477 |
| controlled_alt_vs_proposed_raw | sentence_score | 62 | 6 | 44 | 0.7500 | 0.9118 |
| controlled_alt_vs_proposed_raw | bertscore_f1 | 72 | 40 | 0 | 0.6429 | 0.6429 |
| controlled_alt_vs_proposed_raw | overall_quality | 101 | 11 | 0 | 0.9018 | 0.9018 |
| controlled_alt_vs_candidate_no_context | context_relevance | 110 | 2 | 0 | 0.9821 | 0.9821 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 88 | 10 | 14 | 0.8482 | 0.8980 |
| controlled_alt_vs_candidate_no_context | naturalness | 81 | 31 | 0 | 0.7232 | 0.7232 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 104 | 2 | 6 | 0.9554 | 0.9811 |
| controlled_alt_vs_candidate_no_context | context_overlap | 110 | 2 | 0 | 0.9821 | 0.9821 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 88 | 8 | 16 | 0.8571 | 0.9167 |
| controlled_alt_vs_candidate_no_context | persona_style | 36 | 8 | 68 | 0.6250 | 0.8182 |
| controlled_alt_vs_candidate_no_context | distinct1 | 59 | 50 | 3 | 0.5402 | 0.5413 |
| controlled_alt_vs_candidate_no_context | length_score | 81 | 26 | 5 | 0.7455 | 0.7570 |
| controlled_alt_vs_candidate_no_context | sentence_score | 64 | 7 | 41 | 0.7545 | 0.9014 |
| controlled_alt_vs_candidate_no_context | bertscore_f1 | 93 | 19 | 0 | 0.8304 | 0.8304 |
| controlled_alt_vs_candidate_no_context | overall_quality | 107 | 5 | 0 | 0.9554 | 0.9554 |
| controlled_alt_vs_baseline_no_context | context_relevance | 110 | 2 | 0 | 0.9821 | 0.9821 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 81 | 12 | 19 | 0.8080 | 0.8710 |
| controlled_alt_vs_baseline_no_context | naturalness | 68 | 44 | 0 | 0.6071 | 0.6071 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 105 | 1 | 6 | 0.9643 | 0.9906 |
| controlled_alt_vs_baseline_no_context | context_overlap | 109 | 3 | 0 | 0.9732 | 0.9732 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 79 | 8 | 25 | 0.8170 | 0.9080 |
| controlled_alt_vs_baseline_no_context | persona_style | 14 | 16 | 82 | 0.4911 | 0.4667 |
| controlled_alt_vs_baseline_no_context | distinct1 | 25 | 84 | 3 | 0.2366 | 0.2294 |
| controlled_alt_vs_baseline_no_context | length_score | 75 | 32 | 5 | 0.6920 | 0.7009 |
| controlled_alt_vs_baseline_no_context | sentence_score | 28 | 11 | 73 | 0.5759 | 0.7179 |
| controlled_alt_vs_baseline_no_context | bertscore_f1 | 87 | 25 | 0 | 0.7768 | 0.7768 |
| controlled_alt_vs_baseline_no_context | overall_quality | 109 | 3 | 0 | 0.9732 | 0.9732 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_relevance | 109 | 3 | 0 | 0.9732 | 0.9732 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_consistency | 86 | 10 | 16 | 0.8393 | 0.8958 |
| controlled_alt_vs_baseline_no_context_phi3_latest | naturalness | 69 | 42 | 1 | 0.6205 | 0.6216 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 101 | 0 | 11 | 0.9509 | 1.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_overlap | 108 | 4 | 0 | 0.9643 | 0.9643 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 85 | 8 | 19 | 0.8438 | 0.9140 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_style | 10 | 11 | 91 | 0.4955 | 0.4762 |
| controlled_alt_vs_baseline_no_context_phi3_latest | distinct1 | 18 | 92 | 2 | 0.1696 | 0.1636 |
| controlled_alt_vs_baseline_no_context_phi3_latest | length_score | 79 | 30 | 3 | 0.7188 | 0.7248 |
| controlled_alt_vs_baseline_no_context_phi3_latest | sentence_score | 35 | 11 | 66 | 0.6071 | 0.7609 |
| controlled_alt_vs_baseline_no_context_phi3_latest | bertscore_f1 | 74 | 38 | 0 | 0.6607 | 0.6607 |
| controlled_alt_vs_baseline_no_context_phi3_latest | overall_quality | 107 | 5 | 0 | 0.9554 | 0.9554 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 112 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 81 | 13 | 18 | 0.8036 | 0.8617 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 72 | 40 | 0 | 0.6429 | 0.6429 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 112 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 112 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 79 | 10 | 23 | 0.8080 | 0.8876 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 10 | 14 | 88 | 0.4821 | 0.4167 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | 19 | 89 | 4 | 0.1875 | 0.1759 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 78 | 30 | 4 | 0.7143 | 0.7222 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 32 | 11 | 69 | 0.5938 | 0.7442 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 84 | 28 | 0 | 0.7500 | 0.7500 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 110 | 2 | 0 | 0.9821 | 0.9821 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 112 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 87 | 11 | 14 | 0.8393 | 0.8878 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | 73 | 39 | 0 | 0.6518 | 0.6518 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 112 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 111 | 1 | 0 | 0.9911 | 0.9911 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 87 | 11 | 14 | 0.8393 | 0.8878 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | 5 | 7 | 100 | 0.4911 | 0.4167 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | 19 | 90 | 3 | 0.1830 | 0.1743 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | 82 | 27 | 3 | 0.7455 | 0.7523 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | 37 | 9 | 66 | 0.6250 | 0.8043 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 77 | 35 | 0 | 0.6875 | 0.6875 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 111 | 1 | 0 | 0.9911 | 0.9911 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.0089 | 0.3750 | 0.1071 |
| proposed_contextual_controlled_alt | 0.0000 | 0.0000 | 0.0000 | 0.4821 | 0.0893 |
| proposed_contextual | 0.0000 | 0.0000 | 0.4911 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.5268 | 0.0000 | 0.0000 |
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
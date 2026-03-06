# Proposal Alignment Evaluation Report

- Run ID: `20260305T185149Z`
- Generated: `2026-03-05T18:54:51.980225+00:00`
- Scenarios: `artifacts\proposal_control_tuning\smoke_compare\20260305T185149Z\scenarios.jsonl`
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
| proposed_contextual_controlled | 0.3168 (0.2809, 0.3575) | 0.3151 (0.2595, 0.3802) | 0.8783 (0.8157, 0.9292) | 0.3906 (0.3699, 0.4105) | 0.0873 |
| proposed_contextual_controlled_runtime | 0.3782 (0.3208, 0.4392) | 0.3149 (0.2593, 0.3835) | 0.8585 (0.8049, 0.9099) | 0.4136 (0.3840, 0.4459) | 0.0985 |
| proposed_contextual | 0.0943 (0.0278, 0.1767) | 0.2004 (0.1227, 0.3148) | 0.8372 (0.7908, 0.8836) | 0.2498 (0.1991, 0.3164) | 0.0400 |
| candidate_no_context | 0.0322 (0.0087, 0.0751) | 0.1433 (0.1122, 0.1771) | 0.8574 (0.7862, 0.9345) | 0.2064 (0.1790, 0.2412) | 0.0126 |
| baseline_no_context | 0.0663 (0.0214, 0.1183) | 0.1772 (0.1216, 0.2500) | 0.9086 (0.8472, 0.9655) | 0.2430 (0.2243, 0.2589) | 0.0374 |
| baseline_no_context_phi3_latest | 0.0536 (0.0204, 0.0960) | 0.1857 (0.1333, 0.2405) | 0.9204 (0.8760, 0.9562) | 0.2402 (0.2151, 0.2657) | 0.0216 |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0621 | 1.9260 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0571 | 0.3987 |
| proposed_vs_candidate_no_context | naturalness | -0.0202 | -0.0235 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0859 | 3.0909 |
| proposed_vs_candidate_no_context | context_overlap | 0.0066 | 0.1551 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0556 | nan |
| proposed_vs_candidate_no_context | persona_style | 0.0635 | 0.0886 |
| proposed_vs_candidate_no_context | distinct1 | 0.0160 | 0.0170 |
| proposed_vs_candidate_no_context | length_score | -0.1667 | -0.3371 |
| proposed_vs_candidate_no_context | sentence_score | 0.0583 | 0.0707 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0275 | 2.1874 |
| proposed_vs_candidate_no_context | overall_quality | 0.0434 | 0.2101 |
| proposed_vs_baseline_no_context | context_relevance | 0.0280 | 0.4220 |
| proposed_vs_baseline_no_context | persona_consistency | 0.0233 | 0.1313 |
| proposed_vs_baseline_no_context | naturalness | -0.0714 | -0.0786 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0404 | 0.5517 |
| proposed_vs_baseline_no_context | context_overlap | -0.0010 | -0.0193 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 0.0278 | 1.0000 |
| proposed_vs_baseline_no_context | persona_style | 0.0052 | 0.0068 |
| proposed_vs_baseline_no_context | distinct1 | -0.0167 | -0.0171 |
| proposed_vs_baseline_no_context | length_score | -0.2944 | -0.4732 |
| proposed_vs_baseline_no_context | sentence_score | -0.0583 | -0.0619 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0026 | 0.0697 |
| proposed_vs_baseline_no_context | overall_quality | 0.0068 | 0.0278 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 0.0407 | 0.7585 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | 0.0147 | 0.0793 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | -0.0832 | -0.0903 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.0568 | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 0.0030 | 0.0657 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.0317 | 1.3333 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | -0.0533 | -0.0640 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | -0.0239 | -0.0243 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | -0.3389 | -0.5083 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | -0.0583 | -0.0619 |
| proposed_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0184 | 0.8546 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | 0.0096 | 0.0398 |
| controlled_vs_proposed_raw | context_relevance | 0.2225 | 2.3586 |
| controlled_vs_proposed_raw | persona_consistency | 0.1146 | 0.5719 |
| controlled_vs_proposed_raw | naturalness | 0.0411 | 0.0491 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2917 | 2.5667 |
| controlled_vs_proposed_raw | context_overlap | 0.0610 | 1.2383 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1349 | 2.4286 |
| controlled_vs_proposed_raw | persona_style | 0.0335 | 0.0430 |
| controlled_vs_proposed_raw | distinct1 | -0.0174 | -0.0182 |
| controlled_vs_proposed_raw | length_score | 0.2111 | 0.6441 |
| controlled_vs_proposed_raw | sentence_score | 0.0583 | 0.0660 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0473 | 1.1817 |
| controlled_vs_proposed_raw | overall_quality | 0.1408 | 0.5638 |
| controlled_vs_candidate_no_context | context_relevance | 0.2845 | 8.8271 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1718 | 1.1986 |
| controlled_vs_candidate_no_context | naturalness | 0.0209 | 0.0244 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3775 | 13.5909 |
| controlled_vs_candidate_no_context | context_overlap | 0.0676 | 1.5855 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1905 | nan |
| controlled_vs_candidate_no_context | persona_style | 0.0970 | 0.1354 |
| controlled_vs_candidate_no_context | distinct1 | -0.0014 | -0.0015 |
| controlled_vs_candidate_no_context | length_score | 0.0444 | 0.0899 |
| controlled_vs_candidate_no_context | sentence_score | 0.1167 | 0.1414 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0747 | 5.9539 |
| controlled_vs_candidate_no_context | overall_quality | 0.1842 | 0.8923 |
| controlled_vs_baseline_no_context | context_relevance | 0.2505 | 3.7759 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1379 | 0.7784 |
| controlled_vs_baseline_no_context | naturalness | -0.0303 | -0.0333 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3321 | 4.5345 |
| controlled_vs_baseline_no_context | context_overlap | 0.0600 | 1.1951 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1627 | 5.8571 |
| controlled_vs_baseline_no_context | persona_style | 0.0388 | 0.0501 |
| controlled_vs_baseline_no_context | distinct1 | -0.0341 | -0.0350 |
| controlled_vs_baseline_no_context | length_score | -0.0833 | -0.1339 |
| controlled_vs_baseline_no_context | sentence_score | 0.0000 | 0.0000 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0499 | 1.3338 |
| controlled_vs_baseline_no_context | overall_quality | 0.1476 | 0.6074 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.2631 | 4.9061 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1294 | 0.6967 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | -0.0421 | -0.0457 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.3485 | 6.1333 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0640 | 1.3854 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1667 | 7.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | -0.0198 | -0.0237 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0413 | -0.0420 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | -0.1278 | -0.1917 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.0000 | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0657 | 3.0462 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1504 | 0.6261 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0614 | 0.1938 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0002 | -0.0007 |
| controlled_alt_vs_controlled_default | naturalness | -0.0198 | -0.0226 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0859 | 0.2118 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0043 | 0.0392 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0000 | 0.0000 |
| controlled_alt_vs_controlled_default | persona_style | -0.0011 | -0.0013 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0217 | -0.0231 |
| controlled_alt_vs_controlled_default | length_score | -0.0556 | -0.1031 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0000 | 0.0000 |
| controlled_alt_vs_controlled_default | bertscore_f1 | 0.0112 | 0.1285 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0230 | 0.0589 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.2839 | 3.0095 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1144 | 0.5709 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0213 | 0.0254 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.3775 | 3.3222 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0653 | 1.3261 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1349 | 2.4286 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0325 | 0.0416 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0392 | -0.0409 |
| controlled_alt_vs_proposed_raw | length_score | 0.1556 | 0.4746 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0583 | 0.0660 |
| controlled_alt_vs_proposed_raw | bertscore_f1 | 0.0585 | 1.4621 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1639 | 0.6560 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.3459 | 10.7317 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1716 | 1.1972 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0011 | 0.0013 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.4634 | 16.6818 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0719 | 1.6869 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1905 | nan |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0959 | 0.1339 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0232 | -0.0246 |
| controlled_alt_vs_candidate_no_context | length_score | -0.0111 | -0.0225 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1167 | 0.1414 |
| controlled_alt_vs_candidate_no_context | bertscore_f1 | 0.0860 | 6.8476 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.2072 | 1.0038 |
| controlled_alt_vs_baseline_no_context | context_relevance | 0.3119 | 4.7016 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 0.1377 | 0.7772 |
| controlled_alt_vs_baseline_no_context | naturalness | -0.0501 | -0.0551 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 0.4179 | 5.7069 |
| controlled_alt_vs_baseline_no_context | context_overlap | 0.0643 | 1.2811 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 0.1627 | 5.8571 |
| controlled_alt_vs_baseline_no_context | persona_style | 0.0377 | 0.0487 |
| controlled_alt_vs_baseline_no_context | distinct1 | -0.0558 | -0.0573 |
| controlled_alt_vs_baseline_no_context | length_score | -0.1389 | -0.2232 |
| controlled_alt_vs_baseline_no_context | sentence_score | 0.0000 | 0.0000 |
| controlled_alt_vs_baseline_no_context | bertscore_f1 | 0.0611 | 1.6337 |
| controlled_alt_vs_baseline_no_context | overall_quality | 0.1706 | 0.7021 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_relevance | 0.3245 | 6.0508 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1292 | 0.6955 |
| controlled_alt_vs_baseline_no_context_phi3_latest | naturalness | -0.0619 | -0.0672 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.4343 | 7.6444 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_overlap | 0.0683 | 1.4789 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1667 | 7.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_style | -0.0208 | -0.0250 |
| controlled_alt_vs_baseline_no_context_phi3_latest | distinct1 | -0.0630 | -0.0642 |
| controlled_alt_vs_baseline_no_context_phi3_latest | length_score | -0.1833 | -0.2750 |
| controlled_alt_vs_baseline_no_context_phi3_latest | sentence_score | 0.0000 | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0769 | 3.5662 |
| controlled_alt_vs_baseline_no_context_phi3_latest | overall_quality | 0.1734 | 0.7219 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2505 | 3.7759 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1379 | 0.7784 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0303 | -0.0333 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3321 | 4.5345 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0600 | 1.1951 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1627 | 5.8571 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0388 | 0.0501 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0341 | -0.0350 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | -0.0833 | -0.1339 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0000 | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0499 | 1.3338 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1476 | 0.6074 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.2631 | 4.9061 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1294 | 0.6967 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | -0.0421 | -0.0457 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.3485 | 6.1333 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0640 | 1.3854 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1667 | 7.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | -0.0198 | -0.0237 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0413 | -0.0420 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | -0.1278 | -0.1917 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.0000 | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0657 | 3.0462 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1504 | 0.6261 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0621 | (-0.0386, 0.1620) | 0.1027 | 0.0621 | (-0.0079, 0.0829) | 0.0337 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0571 | (-0.0177, 0.1667) | 0.1283 | 0.0571 | (0.0000, 0.1134) | 0.0393 |
| proposed_vs_candidate_no_context | naturalness | -0.0202 | (-0.0776, 0.0540) | 0.7280 | -0.0202 | (-0.0847, 0.0723) | 0.6227 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0859 | (-0.0556, 0.2222) | 0.1013 | 0.0859 | (0.0000, 0.1111) | 0.0340 |
| proposed_vs_candidate_no_context | context_overlap | 0.0066 | (-0.0104, 0.0254) | 0.2630 | 0.0066 | (-0.0263, 0.0169) | 0.2570 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0556 | (0.0000, 0.1667) | 0.3273 | 0.0556 | (0.0000, 0.1111) | 0.2933 |
| proposed_vs_candidate_no_context | persona_style | 0.0635 | (-0.0864, 0.2154) | 0.1990 | 0.0635 | (0.0000, 0.1228) | 0.0393 |
| proposed_vs_candidate_no_context | distinct1 | 0.0160 | (-0.0288, 0.0653) | 0.2887 | 0.0160 | (-0.0340, 0.1071) | 0.2723 |
| proposed_vs_candidate_no_context | length_score | -0.1667 | (-0.3556, 0.0833) | 0.9153 | -0.1667 | (-0.3556, 0.1833) | 0.8617 |
| proposed_vs_candidate_no_context | sentence_score | 0.0583 | (0.0000, 0.1750) | 0.3483 | 0.0583 | (0.0000, 0.1750) | 0.2887 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0275 | (-0.0073, 0.0756) | 0.0880 | 0.0275 | (0.0118, 0.0420) | 0.0000 |
| proposed_vs_candidate_no_context | overall_quality | 0.0434 | (-0.0253, 0.1108) | 0.1150 | 0.0434 | (-0.0019, 0.0597) | 0.0397 |
| proposed_vs_baseline_no_context | context_relevance | 0.0280 | (-0.0613, 0.1304) | 0.3050 | 0.0280 | (-0.0711, 0.0820) | 0.3040 |
| proposed_vs_baseline_no_context | persona_consistency | 0.0233 | (-0.0036, 0.0695) | 0.1230 | 0.0233 | (0.0000, 0.0465) | 0.3023 |
| proposed_vs_baseline_no_context | naturalness | -0.0714 | (-0.1149, -0.0286) | 1.0000 | -0.0714 | (-0.1096, -0.0133) | 1.0000 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0404 | (-0.0833, 0.1944) | 0.3197 | 0.0404 | (-0.0909, 0.1111) | 0.2960 |
| proposed_vs_baseline_no_context | context_overlap | -0.0010 | (-0.0193, 0.0192) | 0.5447 | -0.0010 | (-0.0250, 0.0140) | 0.7387 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 0.0278 | (0.0000, 0.0833) | 0.3487 | 0.0278 | (0.0000, 0.0556) | 0.3003 |
| proposed_vs_baseline_no_context | persona_style | 0.0052 | (-0.0273, 0.0430) | 0.4513 | 0.0052 | (0.0000, 0.0105) | 0.2960 |
| proposed_vs_baseline_no_context | distinct1 | -0.0167 | (-0.0527, 0.0124) | 0.8457 | -0.0167 | (-0.0303, 0.0000) | 1.0000 |
| proposed_vs_baseline_no_context | length_score | -0.2944 | (-0.4389, -0.1444) | 1.0000 | -0.2944 | (-0.4000, -0.0667) | 1.0000 |
| proposed_vs_baseline_no_context | sentence_score | -0.0583 | (-0.1750, 0.0000) | 1.0000 | -0.0583 | (-0.1750, 0.0000) | 1.0000 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0026 | (-0.0573, 0.0736) | 0.4907 | 0.0026 | (-0.0985, 0.0656) | 0.4117 |
| proposed_vs_baseline_no_context | overall_quality | 0.0068 | (-0.0417, 0.0682) | 0.4423 | 0.0068 | (-0.0428, 0.0450) | 0.3017 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 0.0407 | (-0.0279, 0.1032) | 0.1237 | 0.0407 | (-0.0708, 0.0631) | 0.1563 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | 0.0147 | (-0.0608, 0.1227) | 0.4183 | 0.0147 | (-0.1143, 0.0676) | 0.3997 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | -0.0832 | (-0.1386, -0.0304) | 1.0000 | -0.0832 | (-0.1088, -0.0267) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.0568 | (-0.0417, 0.1439) | 0.1423 | 0.0568 | (-0.0909, 0.0909) | 0.1557 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 0.0030 | (-0.0115, 0.0180) | 0.3440 | 0.0030 | (-0.0238, 0.0159) | 0.2920 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.0317 | (-0.0714, 0.1667) | 0.3467 | 0.0317 | (-0.1429, 0.1111) | 0.2900 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | -0.0533 | (-0.1417, 0.0000) | 1.0000 | -0.0533 | (-0.1066, 0.0000) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | -0.0239 | (-0.0635, 0.0127) | 0.8963 | -0.0239 | (-0.0289, 0.0000) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | -0.3389 | (-0.5056, -0.1611) | 1.0000 | -0.3389 | (-0.4000, -0.1333) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | -0.0583 | (-0.1750, 0.0000) | 1.0000 | -0.0583 | (-0.1750, 0.0000) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0184 | (-0.0208, 0.0811) | 0.3160 | 0.0184 | (-0.0210, 0.0555) | 0.3013 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | 0.0096 | (-0.0368, 0.0661) | 0.3743 | 0.0096 | (-0.0738, 0.0408) | 0.3810 |
| controlled_vs_proposed_raw | context_relevance | 0.2225 | (0.1523, 0.2964) | 0.0000 | 0.2225 | (0.1881, 0.3582) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.1146 | (0.0000, 0.2023) | 0.0230 | 0.1146 | (0.1023, 0.1333) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0411 | (-0.0349, 0.1137) | 0.1693 | 0.0411 | (-0.0255, 0.1214) | 0.1413 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2917 | (0.2020, 0.3876) | 0.0000 | 0.2917 | (0.2500, 0.4545) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0610 | (0.0334, 0.0950) | 0.0000 | 0.0610 | (0.0435, 0.1333) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1349 | (-0.0040, 0.2460) | 0.0413 | 0.1349 | (0.1111, 0.1667) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.0335 | (0.0000, 0.0824) | 0.0913 | 0.0335 | (0.0000, 0.0671) | 0.3043 |
| controlled_vs_proposed_raw | distinct1 | -0.0174 | (-0.0483, 0.0169) | 0.8187 | -0.0174 | (-0.0247, 0.0000) | 1.0000 |
| controlled_vs_proposed_raw | length_score | 0.2111 | (-0.0833, 0.4944) | 0.0900 | 0.2111 | (-0.0778, 0.5500) | 0.1463 |
| controlled_vs_proposed_raw | sentence_score | 0.0583 | (0.0000, 0.1750) | 0.3320 | 0.0583 | (0.0000, 0.1750) | 0.2960 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0473 | (-0.0191, 0.1079) | 0.0720 | 0.0473 | (0.0186, 0.1312) | 0.0000 |
| controlled_vs_proposed_raw | overall_quality | 0.1408 | (0.0813, 0.1884) | 0.0000 | 0.1408 | (0.1096, 0.2120) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2845 | (0.2336, 0.3352) | 0.0000 | 0.2845 | (0.2709, 0.3503) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1718 | (0.1211, 0.2442) | 0.0000 | 0.1718 | (0.1168, 0.2157) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.0209 | (-0.1012, 0.1399) | 0.3637 | 0.0209 | (-0.1102, 0.1938) | 0.3593 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3775 | (0.3157, 0.4394) | 0.0000 | 0.3775 | (0.3611, 0.4545) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0676 | (0.0420, 0.0932) | 0.0000 | 0.0676 | (0.0586, 0.1070) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1905 | (0.1548, 0.2500) | 0.0000 | 0.1905 | (0.1429, 0.2222) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.0970 | (-0.0375, 0.2406) | 0.0937 | 0.0970 | (0.0000, 0.1898) | 0.0383 |
| controlled_vs_candidate_no_context | distinct1 | -0.0014 | (-0.0526, 0.0501) | 0.5523 | -0.0014 | (-0.0587, 0.1071) | 0.5950 |
| controlled_vs_candidate_no_context | length_score | 0.0444 | (-0.4167, 0.4944) | 0.3997 | 0.0444 | (-0.4333, 0.7333) | 0.3490 |
| controlled_vs_candidate_no_context | sentence_score | 0.1167 | (0.0000, 0.2333) | 0.0890 | 0.1167 | (0.0000, 0.3500) | 0.3027 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0747 | (0.0427, 0.1117) | 0.0000 | 0.0747 | (0.0304, 0.1465) | 0.0000 |
| controlled_vs_candidate_no_context | overall_quality | 0.1842 | (0.1492, 0.2058) | 0.0000 | 0.1842 | (0.1693, 0.2100) | 0.0000 |
| controlled_vs_baseline_no_context | context_relevance | 0.2505 | (0.1919, 0.3151) | 0.0000 | 0.2505 | (0.2028, 0.2870) | 0.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1379 | (0.0712, 0.2000) | 0.0003 | 0.1379 | (0.1143, 0.1488) | 0.0000 |
| controlled_vs_baseline_no_context | naturalness | -0.0303 | (-0.0907, 0.0250) | 0.8400 | -0.0303 | (-0.0907, 0.0667) | 0.7033 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3321 | (0.2538, 0.4129) | 0.0000 | 0.3321 | (0.2727, 0.3636) | 0.0000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0600 | (0.0335, 0.0865) | 0.0000 | 0.0600 | (0.0396, 0.1083) | 0.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1627 | (0.0833, 0.2460) | 0.0000 | 0.1627 | (0.1429, 0.1667) | 0.0000 |
| controlled_vs_baseline_no_context | persona_style | 0.0388 | (0.0000, 0.1164) | 0.3390 | 0.0388 | (0.0000, 0.0776) | 0.2900 |
| controlled_vs_baseline_no_context | distinct1 | -0.0341 | (-0.0522, -0.0144) | 1.0000 | -0.0341 | (-0.0455, 0.0000) | 1.0000 |
| controlled_vs_baseline_no_context | length_score | -0.0833 | (-0.3612, 0.1778) | 0.7120 | -0.0833 | (-0.3778, 0.3333) | 0.7133 |
| controlled_vs_baseline_no_context | sentence_score | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0499 | (0.0007, 0.0993) | 0.0203 | 0.0499 | (-0.0227, 0.1040) | 0.1567 |
| controlled_vs_baseline_no_context | overall_quality | 0.1476 | (0.1222, 0.1707) | 0.0000 | 0.1476 | (0.1264, 0.1691) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.2631 | (0.2533, 0.2750) | 0.0000 | 0.2631 | (0.2512, 0.2874) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1294 | (0.0667, 0.1960) | 0.0000 | 0.1294 | (0.0000, 0.1699) | 0.0353 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | -0.0421 | (-0.1133, 0.0188) | 0.8800 | -0.0421 | (-0.1104, 0.0533) | 0.7147 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.3485 | (0.3384, 0.3586) | 0.0000 | 0.3485 | (0.3333, 0.3636) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0640 | (0.0472, 0.0839) | 0.0000 | 0.0640 | (0.0482, 0.1095) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1667 | (0.0833, 0.2500) | 0.0000 | 0.1667 | (0.0000, 0.2222) | 0.0397 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | -0.0198 | (-0.0593, 0.0000) | 1.0000 | -0.0198 | (-0.0395, 0.0000) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0413 | (-0.0631, -0.0198) | 1.0000 | -0.0413 | (-0.0537, 0.0000) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | -0.1278 | (-0.4444, 0.1611) | 0.7897 | -0.1278 | (-0.4444, 0.2667) | 0.7073 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0657 | (0.0255, 0.1013) | 0.0003 | 0.0657 | (0.0011, 0.1102) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1504 | (0.1351, 0.1682) | 0.0000 | 0.1504 | (0.1381, 0.1565) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0614 | (-0.0142, 0.1489) | 0.0710 | 0.0614 | (-0.0017, 0.0815) | 0.0320 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0002 | (-0.0667, 0.0665) | 0.6177 | -0.0002 | (-0.0004, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | naturalness | -0.0198 | (-0.0936, 0.0500) | 0.6917 | -0.0198 | (-0.0858, 0.0161) | 0.6990 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0859 | (-0.0139, 0.2083) | 0.0693 | 0.0859 | (0.0000, 0.1111) | 0.0323 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0043 | (-0.0265, 0.0346) | 0.3677 | 0.0043 | (-0.0057, 0.0123) | 0.3097 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0000 | (-0.0833, 0.0833) | 0.6357 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | persona_style | -0.0011 | (-0.0032, 0.0000) | 1.0000 | -0.0011 | (-0.0021, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0217 | (-0.0503, 0.0143) | 0.8960 | -0.0217 | (-0.0385, -0.0098) | 1.0000 |
| controlled_alt_vs_controlled_default | length_score | -0.0556 | (-0.3556, 0.2333) | 0.6363 | -0.0556 | (-0.3667, 0.1000) | 0.7027 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | bertscore_f1 | 0.0112 | (-0.0235, 0.0429) | 0.2683 | 0.0112 | (-0.0156, 0.0455) | 0.2633 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0230 | (0.0088, 0.0401) | 0.0000 | 0.0230 | (0.0058, 0.0382) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.2839 | (0.1820, 0.3565) | 0.0000 | 0.2839 | (0.2690, 0.3565) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1144 | (0.0667, 0.1463) | 0.0003 | 0.1144 | (0.1019, 0.1333) | 0.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0213 | (-0.0751, 0.1128) | 0.3423 | 0.0213 | (-0.0094, 0.0846) | 0.0383 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.3775 | (0.2462, 0.4697) | 0.0000 | 0.3775 | (0.3611, 0.4545) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0653 | (0.0304, 0.1010) | 0.0000 | 0.0653 | (0.0483, 0.1277) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1349 | (0.0794, 0.1667) | 0.0003 | 0.1349 | (0.1111, 0.1667) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0325 | (0.0000, 0.0792) | 0.0870 | 0.0325 | (0.0000, 0.0650) | 0.2937 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0392 | (-0.0867, 0.0080) | 0.9300 | -0.0392 | (-0.0464, -0.0346) | 1.0000 |
| controlled_alt_vs_proposed_raw | length_score | 0.1556 | (-0.2111, 0.5056) | 0.1957 | 0.1556 | (0.0222, 0.5000) | 0.0000 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0583 | (0.0000, 0.1750) | 0.3210 | 0.0583 | (0.0000, 0.1750) | 0.3000 |
| controlled_alt_vs_proposed_raw | bertscore_f1 | 0.0585 | (-0.0068, 0.1259) | 0.0327 | 0.0585 | (0.0030, 0.1767) | 0.0000 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1639 | (0.1088, 0.2056) | 0.0000 | 0.1639 | (0.1479, 0.2177) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.3459 | (0.2878, 0.4125) | 0.0000 | 0.3459 | (0.3350, 0.3524) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1716 | (0.1209, 0.2409) | 0.0000 | 0.1716 | (0.1168, 0.2153) | 0.0000 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0011 | (-0.0971, 0.0963) | 0.5003 | 0.0011 | (-0.0941, 0.1080) | 0.3713 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.4634 | (0.3788, 0.5631) | 0.0000 | 0.4634 | (0.4545, 0.4722) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0719 | (0.0519, 0.0927) | 0.0000 | 0.0719 | (0.0559, 0.1013) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1905 | (0.1548, 0.2500) | 0.0000 | 0.1905 | (0.1429, 0.2222) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0959 | (-0.0396, 0.2427) | 0.0930 | 0.0959 | (0.0000, 0.1877) | 0.0453 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0232 | (-0.0760, 0.0269) | 0.8003 | -0.0232 | (-0.0686, 0.0687) | 0.7540 |
| controlled_alt_vs_candidate_no_context | length_score | -0.0111 | (-0.3611, 0.3556) | 0.5447 | -0.0111 | (-0.3333, 0.3667) | 0.6017 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1167 | (0.0000, 0.2333) | 0.0893 | 0.1167 | (0.0000, 0.3500) | 0.2987 |
| controlled_alt_vs_candidate_no_context | bertscore_f1 | 0.0860 | (0.0363, 0.1395) | 0.0000 | 0.0860 | (0.0148, 0.1920) | 0.0000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.2072 | (0.1692, 0.2415) | 0.0000 | 0.2072 | (0.2024, 0.2158) | 0.0000 |
| controlled_alt_vs_baseline_no_context | context_relevance | 0.3119 | (0.2679, 0.3836) | 0.0000 | 0.3119 | (0.2656, 0.3515) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 0.1377 | (0.1238, 0.1560) | 0.0000 | 0.1377 | (0.1143, 0.1484) | 0.0000 |
| controlled_alt_vs_baseline_no_context | naturalness | -0.0501 | (-0.1434, 0.0362) | 0.8543 | -0.0501 | (-0.0746, 0.0713) | 0.9650 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 0.4179 | (0.3535, 0.5189) | 0.0000 | 0.4179 | (0.3636, 0.4722) | 0.0000 |
| controlled_alt_vs_baseline_no_context | context_overlap | 0.0643 | (0.0373, 0.0908) | 0.0000 | 0.0643 | (0.0369, 0.1027) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 0.1627 | (0.1548, 0.1667) | 0.0000 | 0.1627 | (0.1429, 0.1667) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_style | 0.0377 | (0.0000, 0.1132) | 0.3490 | 0.0377 | (0.0000, 0.0755) | 0.2907 |
| controlled_alt_vs_baseline_no_context | distinct1 | -0.0558 | (-0.0886, -0.0236) | 1.0000 | -0.0558 | (-0.0767, -0.0385) | 1.0000 |
| controlled_alt_vs_baseline_no_context | length_score | -0.1389 | (-0.5447, 0.2444) | 0.7530 | -0.1389 | (-0.2778, 0.4333) | 0.7490 |
| controlled_alt_vs_baseline_no_context | sentence_score | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_baseline_no_context | bertscore_f1 | 0.0611 | (0.0005, 0.1201) | 0.0227 | 0.0611 | (-0.0382, 0.1216) | 0.1457 |
| controlled_alt_vs_baseline_no_context | overall_quality | 0.1706 | (0.1389, 0.2081) | 0.0000 | 0.1706 | (0.1352, 0.1928) | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_relevance | 0.3245 | (0.2460, 0.4027) | 0.0000 | 0.3245 | (0.2857, 0.3326) | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1292 | (0.0667, 0.1958) | 0.0000 | 0.1292 | (0.0000, 0.1694) | 0.0383 |
| controlled_alt_vs_baseline_no_context_phi3_latest | naturalness | -0.0619 | (-0.1218, 0.0025) | 0.9727 | -0.0619 | (-0.0943, 0.0579) | 0.9623 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.4343 | (0.3346, 0.5455) | 0.0000 | 0.4343 | (0.3636, 0.4545) | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_overlap | 0.0683 | (0.0318, 0.1031) | 0.0000 | 0.0683 | (0.0455, 0.1039) | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1667 | (0.0833, 0.2500) | 0.0000 | 0.1667 | (0.0000, 0.2222) | 0.0397 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_style | -0.0208 | (-0.0625, 0.0000) | 1.0000 | -0.0208 | (-0.0417, 0.0000) | 1.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | distinct1 | -0.0630 | (-0.0835, -0.0426) | 1.0000 | -0.0630 | (-0.0746, -0.0385) | 1.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | length_score | -0.1833 | (-0.4500, 0.1056) | 0.8897 | -0.1833 | (-0.3444, 0.3667) | 0.8497 |
| controlled_alt_vs_baseline_no_context_phi3_latest | sentence_score | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0769 | (0.0160, 0.1311) | 0.0063 | 0.0769 | (-0.0144, 0.1557) | 0.0357 |
| controlled_alt_vs_baseline_no_context_phi3_latest | overall_quality | 0.1734 | (0.1524, 0.2057) | 0.0000 | 0.1734 | (0.1439, 0.1886) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2505 | (0.1884, 0.3117) | 0.0000 | 0.2505 | (0.2028, 0.2870) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1379 | (0.0712, 0.1968) | 0.0000 | 0.1379 | (0.1143, 0.1488) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0303 | (-0.0894, 0.0251) | 0.8363 | -0.0303 | (-0.0907, 0.0667) | 0.7123 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3321 | (0.2538, 0.4091) | 0.0000 | 0.3321 | (0.2727, 0.3636) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0600 | (0.0339, 0.0861) | 0.0000 | 0.0600 | (0.0396, 0.1083) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1627 | (0.0833, 0.2460) | 0.0003 | 0.1627 | (0.1429, 0.1667) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0388 | (0.0000, 0.1164) | 0.3240 | 0.0388 | (0.0000, 0.0776) | 0.2997 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0341 | (-0.0526, -0.0155) | 1.0000 | -0.0341 | (-0.0455, 0.0000) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | -0.0833 | (-0.3667, 0.1722) | 0.7160 | -0.0833 | (-0.3778, 0.3333) | 0.6990 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0499 | (0.0005, 0.0999) | 0.0210 | 0.0499 | (-0.0227, 0.1040) | 0.1497 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1476 | (0.1222, 0.1709) | 0.0000 | 0.1476 | (0.1264, 0.1691) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.2631 | (0.2527, 0.2744) | 0.0000 | 0.2631 | (0.2512, 0.2874) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1294 | (0.0667, 0.1960) | 0.0000 | 0.1294 | (0.0000, 0.1699) | 0.0387 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | -0.0421 | (-0.1107, 0.0181) | 0.8863 | -0.0421 | (-0.1104, 0.0533) | 0.7010 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.3485 | (0.3384, 0.3586) | 0.0000 | 0.3485 | (0.3333, 0.3636) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0640 | (0.0470, 0.0839) | 0.0000 | 0.0640 | (0.0482, 0.1095) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1667 | (0.0833, 0.2500) | 0.0000 | 0.1667 | (0.0000, 0.2222) | 0.0347 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | -0.0198 | (-0.0593, 0.0000) | 1.0000 | -0.0198 | (-0.0395, 0.0000) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0413 | (-0.0627, -0.0198) | 1.0000 | -0.0413 | (-0.0537, 0.0000) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | -0.1278 | (-0.4389, 0.1444) | 0.7930 | -0.1278 | (-0.4444, 0.2667) | 0.7057 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0657 | (0.0259, 0.1013) | 0.0003 | 0.0657 | (0.0011, 0.1102) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1504 | (0.1355, 0.1682) | 0.0000 | 0.1504 | (0.1381, 0.1565) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 3 | 2 | 1 | 0.5833 | 0.6000 |
| proposed_vs_candidate_no_context | persona_consistency | 3 | 1 | 2 | 0.6667 | 0.7500 |
| proposed_vs_candidate_no_context | naturalness | 1 | 4 | 1 | 0.2500 | 0.2000 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 3 | 1 | 2 | 0.6667 | 0.7500 |
| proposed_vs_candidate_no_context | context_overlap | 3 | 2 | 1 | 0.5833 | 0.6000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 1 | 0 | 5 | 0.5833 | 1.0000 |
| proposed_vs_candidate_no_context | persona_style | 3 | 1 | 2 | 0.6667 | 0.7500 |
| proposed_vs_candidate_no_context | distinct1 | 2 | 3 | 1 | 0.4167 | 0.4000 |
| proposed_vs_candidate_no_context | length_score | 1 | 4 | 1 | 0.2500 | 0.2000 |
| proposed_vs_candidate_no_context | sentence_score | 1 | 0 | 5 | 0.5833 | 1.0000 |
| proposed_vs_candidate_no_context | bertscore_f1 | 4 | 1 | 1 | 0.7500 | 0.8000 |
| proposed_vs_candidate_no_context | overall_quality | 3 | 2 | 1 | 0.5833 | 0.6000 |
| proposed_vs_baseline_no_context | context_relevance | 3 | 3 | 0 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context | persona_consistency | 2 | 1 | 3 | 0.5833 | 0.6667 |
| proposed_vs_baseline_no_context | naturalness | 0 | 6 | 0 | 0.0000 | 0.0000 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 2 | 2 | 2 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context | context_overlap | 3 | 3 | 0 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 1 | 0 | 5 | 0.5833 | 1.0000 |
| proposed_vs_baseline_no_context | persona_style | 1 | 1 | 4 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context | distinct1 | 3 | 2 | 1 | 0.5833 | 0.6000 |
| proposed_vs_baseline_no_context | length_score | 0 | 6 | 0 | 0.0000 | 0.0000 |
| proposed_vs_baseline_no_context | sentence_score | 0 | 1 | 5 | 0.4167 | 0.0000 |
| proposed_vs_baseline_no_context | bertscore_f1 | 2 | 4 | 0 | 0.3333 | 0.3333 |
| proposed_vs_baseline_no_context | overall_quality | 2 | 4 | 0 | 0.3333 | 0.3333 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 4 | 2 | 0 | 0.6667 | 0.6667 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | 1 | 3 | 2 | 0.3333 | 0.2500 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | 1 | 5 | 0 | 0.1667 | 0.1667 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 3 | 2 | 1 | 0.5833 | 0.6000 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 4 | 2 | 0 | 0.6667 | 0.6667 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 1 | 1 | 4 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | 0 | 2 | 4 | 0.3333 | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | 2 | 3 | 1 | 0.4167 | 0.4000 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | 0 | 6 | 0 | 0.0000 | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | 0 | 1 | 5 | 0.4167 | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | bertscore_f1 | 2 | 4 | 0 | 0.3333 | 0.3333 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | 3 | 3 | 0 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | context_relevance | 6 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_proposed_raw | persona_consistency | 5 | 1 | 0 | 0.8333 | 0.8333 |
| controlled_vs_proposed_raw | naturalness | 4 | 2 | 0 | 0.6667 | 0.6667 |
| controlled_vs_proposed_raw | context_keyword_coverage | 6 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_proposed_raw | context_overlap | 6 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 5 | 1 | 0 | 0.8333 | 0.8333 |
| controlled_vs_proposed_raw | persona_style | 2 | 0 | 4 | 0.6667 | 1.0000 |
| controlled_vs_proposed_raw | distinct1 | 2 | 3 | 1 | 0.4167 | 0.4000 |
| controlled_vs_proposed_raw | length_score | 4 | 2 | 0 | 0.6667 | 0.6667 |
| controlled_vs_proposed_raw | sentence_score | 1 | 0 | 5 | 0.5833 | 1.0000 |
| controlled_vs_proposed_raw | bertscore_f1 | 5 | 1 | 0 | 0.8333 | 0.8333 |
| controlled_vs_proposed_raw | overall_quality | 6 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_relevance | 6 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 6 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | naturalness | 3 | 3 | 0 | 0.5000 | 0.5000 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 6 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 6 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 6 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_style | 3 | 1 | 2 | 0.6667 | 0.7500 |
| controlled_vs_candidate_no_context | distinct1 | 3 | 3 | 0 | 0.5000 | 0.5000 |
| controlled_vs_candidate_no_context | length_score | 3 | 2 | 1 | 0.5833 | 0.6000 |
| controlled_vs_candidate_no_context | sentence_score | 2 | 0 | 4 | 0.6667 | 1.0000 |
| controlled_vs_candidate_no_context | bertscore_f1 | 6 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | overall_quality | 6 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | context_relevance | 6 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 5 | 0 | 1 | 0.9167 | 1.0000 |
| controlled_vs_baseline_no_context | naturalness | 2 | 4 | 0 | 0.3333 | 0.3333 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 6 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | context_overlap | 6 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 5 | 0 | 1 | 0.9167 | 1.0000 |
| controlled_vs_baseline_no_context | persona_style | 1 | 0 | 5 | 0.5833 | 1.0000 |
| controlled_vs_baseline_no_context | distinct1 | 0 | 5 | 1 | 0.0833 | 0.0000 |
| controlled_vs_baseline_no_context | length_score | 2 | 3 | 1 | 0.4167 | 0.4000 |
| controlled_vs_baseline_no_context | sentence_score | 0 | 0 | 6 | 0.5000 | nan |
| controlled_vs_baseline_no_context | bertscore_f1 | 4 | 2 | 0 | 0.6667 | 0.6667 |
| controlled_vs_baseline_no_context | overall_quality | 6 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 6 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 5 | 0 | 1 | 0.9167 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | 3 | 3 | 0 | 0.5000 | 0.5000 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 6 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 6 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 5 | 0 | 1 | 0.9167 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | 0 | 1 | 5 | 0.4167 | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | 0 | 5 | 1 | 0.0833 | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | 3 | 3 | 0 | 0.5000 | 0.5000 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0 | 0 | 6 | 0.5000 | nan |
| controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 5 | 1 | 0 | 0.8333 | 0.8333 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 6 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 3 | 2 | 1 | 0.5833 | 0.6000 |
| controlled_alt_vs_controlled_default | persona_consistency | 1 | 2 | 3 | 0.4167 | 0.3333 |
| controlled_alt_vs_controlled_default | naturalness | 2 | 3 | 1 | 0.4167 | 0.4000 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 3 | 1 | 2 | 0.6667 | 0.7500 |
| controlled_alt_vs_controlled_default | context_overlap | 2 | 3 | 1 | 0.4167 | 0.4000 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 1 | 1 | 4 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | persona_style | 0 | 1 | 5 | 0.4167 | 0.0000 |
| controlled_alt_vs_controlled_default | distinct1 | 1 | 4 | 1 | 0.2500 | 0.2000 |
| controlled_alt_vs_controlled_default | length_score | 2 | 3 | 1 | 0.4167 | 0.4000 |
| controlled_alt_vs_controlled_default | sentence_score | 0 | 0 | 6 | 0.5000 | nan |
| controlled_alt_vs_controlled_default | bertscore_f1 | 3 | 2 | 1 | 0.5833 | 0.6000 |
| controlled_alt_vs_controlled_default | overall_quality | 5 | 0 | 1 | 0.9167 | 1.0000 |
| controlled_alt_vs_proposed_raw | context_relevance | 6 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 5 | 0 | 1 | 0.9167 | 1.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 4 | 2 | 0 | 0.6667 | 0.6667 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 6 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 5 | 1 | 0 | 0.8333 | 0.8333 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 5 | 0 | 1 | 0.9167 | 1.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 2 | 0 | 4 | 0.6667 | 1.0000 |
| controlled_alt_vs_proposed_raw | distinct1 | 3 | 3 | 0 | 0.5000 | 0.5000 |
| controlled_alt_vs_proposed_raw | length_score | 4 | 2 | 0 | 0.6667 | 0.6667 |
| controlled_alt_vs_proposed_raw | sentence_score | 1 | 0 | 5 | 0.5833 | 1.0000 |
| controlled_alt_vs_proposed_raw | bertscore_f1 | 4 | 2 | 0 | 0.6667 | 0.6667 |
| controlled_alt_vs_proposed_raw | overall_quality | 6 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 6 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 6 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | naturalness | 3 | 3 | 0 | 0.5000 | 0.5000 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 6 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 6 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 6 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_style | 3 | 1 | 2 | 0.6667 | 0.7500 |
| controlled_alt_vs_candidate_no_context | distinct1 | 2 | 4 | 0 | 0.3333 | 0.3333 |
| controlled_alt_vs_candidate_no_context | length_score | 2 | 3 | 1 | 0.4167 | 0.4000 |
| controlled_alt_vs_candidate_no_context | sentence_score | 2 | 0 | 4 | 0.6667 | 1.0000 |
| controlled_alt_vs_candidate_no_context | bertscore_f1 | 5 | 1 | 0 | 0.8333 | 0.8333 |
| controlled_alt_vs_candidate_no_context | overall_quality | 6 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context | context_relevance | 6 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 6 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context | naturalness | 3 | 3 | 0 | 0.5000 | 0.5000 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 6 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context | context_overlap | 6 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 6 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context | persona_style | 1 | 0 | 5 | 0.5833 | 1.0000 |
| controlled_alt_vs_baseline_no_context | distinct1 | 1 | 5 | 0 | 0.1667 | 0.1667 |
| controlled_alt_vs_baseline_no_context | length_score | 3 | 3 | 0 | 0.5000 | 0.5000 |
| controlled_alt_vs_baseline_no_context | sentence_score | 0 | 0 | 6 | 0.5000 | nan |
| controlled_alt_vs_baseline_no_context | bertscore_f1 | 4 | 2 | 0 | 0.6667 | 0.6667 |
| controlled_alt_vs_baseline_no_context | overall_quality | 6 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_relevance | 6 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_consistency | 5 | 0 | 1 | 0.9167 | 1.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | naturalness | 2 | 4 | 0 | 0.3333 | 0.3333 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 6 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_overlap | 5 | 1 | 0 | 0.8333 | 0.8333 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 5 | 0 | 1 | 0.9167 | 1.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_style | 0 | 1 | 5 | 0.4167 | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | distinct1 | 0 | 6 | 0 | 0.0000 | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | length_score | 2 | 4 | 0 | 0.3333 | 0.3333 |
| controlled_alt_vs_baseline_no_context_phi3_latest | sentence_score | 0 | 0 | 6 | 0.5000 | nan |
| controlled_alt_vs_baseline_no_context_phi3_latest | bertscore_f1 | 5 | 1 | 0 | 0.8333 | 0.8333 |
| controlled_alt_vs_baseline_no_context_phi3_latest | overall_quality | 6 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 6 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 5 | 0 | 1 | 0.9167 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 2 | 4 | 0 | 0.3333 | 0.3333 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 6 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 6 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 5 | 0 | 1 | 0.9167 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 1 | 0 | 5 | 0.5833 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | 0 | 5 | 1 | 0.0833 | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 2 | 3 | 1 | 0.4167 | 0.4000 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0 | 0 | 6 | 0.5000 | nan |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 4 | 2 | 0 | 0.6667 | 0.6667 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 6 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 6 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 5 | 0 | 1 | 0.9167 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | 3 | 3 | 0 | 0.5000 | 0.5000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 6 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 6 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 5 | 0 | 1 | 0.9167 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | 0 | 1 | 5 | 0.4167 | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | 0 | 5 | 1 | 0.0833 | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | 3 | 3 | 0 | 0.5000 | 0.5000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0 | 0 | 6 | 0.5000 | nan |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 5 | 1 | 0 | 0.8333 | 0.8333 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 6 | 0 | 0 | 1.0000 | 1.0000 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.5000 | 0.5000 | 0.5000 |
| proposed_contextual_controlled_runtime | 0.0000 | 0.0000 | 0.1667 | 0.8333 | 0.1667 |
| proposed_contextual | 0.0000 | 0.0000 | 0.1667 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.3333 | 0.0000 | 0.0000 |
| baseline_no_context | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline_no_context_phi3_latest | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `3`
- Unique template signatures: `6`
- Template signature ratio: `1.0000`
- Effective sample size by source clustering: `2.57`
- Effective sample size by template-signature clustering: `6.00`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: enabled.

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
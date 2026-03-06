# Proposal Alignment Evaluation Report

- Run ID: `20260305T185951Z`
- Generated: `2026-03-05T19:05:40.491983+00:00`
- Scenarios: `artifacts\proposal_control_tuning\profile_compare_runtime20\20260305T185951Z\scenarios.jsonl`
- Scenario count: `20`

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
| proposed_contextual_controlled | 0.3169 (0.2762, 0.3591) | 0.3431 (0.2991, 0.3949) | 0.8420 (0.8168, 0.8677) | 0.3957 (0.3735, 0.4190) | 0.0875 |
| proposed_contextual_controlled_runtime | 0.2539 (0.2103, 0.2974) | 0.3095 (0.2533, 0.3684) | 0.8694 (0.8355, 0.9012) | 0.3600 (0.3382, 0.3821) | 0.0517 |
| proposed_contextual | 0.1302 (0.0657, 0.2035) | 0.1645 (0.1065, 0.2310) | 0.8168 (0.7843, 0.8520) | 0.2553 (0.2106, 0.3029) | 0.0771 |
| candidate_no_context | 0.0414 (0.0209, 0.0650) | 0.2089 (0.1414, 0.2832) | 0.8263 (0.7927, 0.8643) | 0.2299 (0.2005, 0.2628) | 0.0291 |
| baseline_no_context | 0.0261 (0.0112, 0.0442) | 0.1669 (0.1271, 0.2142) | 0.9039 (0.8803, 0.9264) | 0.2278 (0.2098, 0.2479) | 0.0582 |
| baseline_no_context_phi3_latest | 0.0543 (0.0317, 0.0788) | 0.1733 (0.1342, 0.2153) | 0.9122 (0.8851, 0.9354) | 0.2421 (0.2234, 0.2595) | 0.0588 |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0887 | 2.1411 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0444 | -0.2126 |
| proposed_vs_candidate_no_context | naturalness | -0.0095 | -0.0115 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.1174 | 3.0392 |
| proposed_vs_candidate_no_context | context_overlap | 0.0217 | 0.4533 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0524 | -0.4583 |
| proposed_vs_candidate_no_context | persona_style | -0.0125 | -0.0213 |
| proposed_vs_candidate_no_context | distinct1 | -0.0060 | -0.0063 |
| proposed_vs_candidate_no_context | length_score | -0.0267 | -0.0773 |
| proposed_vs_candidate_no_context | sentence_score | -0.0175 | -0.0217 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0481 | 1.6541 |
| proposed_vs_candidate_no_context | overall_quality | 0.0255 | 0.1108 |
| proposed_vs_baseline_no_context | context_relevance | 0.1041 | 3.9854 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0024 | -0.0144 |
| proposed_vs_baseline_no_context | naturalness | -0.0871 | -0.0963 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.1341 | 6.1034 |
| proposed_vs_baseline_no_context | context_overlap | 0.0340 | 0.9495 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0064 | -0.0941 |
| proposed_vs_baseline_no_context | persona_style | 0.0137 | 0.0244 |
| proposed_vs_baseline_no_context | distinct1 | -0.0487 | -0.0495 |
| proposed_vs_baseline_no_context | length_score | -0.2767 | -0.4650 |
| proposed_vs_baseline_no_context | sentence_score | -0.1225 | -0.1342 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0189 | 0.3248 |
| proposed_vs_baseline_no_context | overall_quality | 0.0276 | 0.1210 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 0.0759 | 1.3972 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | -0.0087 | -0.0504 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | -0.0954 | -0.1046 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.0996 | 1.7651 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 0.0204 | 0.4145 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | -0.0038 | -0.0580 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | -0.0284 | -0.0471 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | -0.0450 | -0.0459 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | -0.3083 | -0.4920 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | -0.1575 | -0.1662 |
| proposed_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0184 | 0.3130 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | 0.0132 | 0.0546 |
| controlled_vs_proposed_raw | context_relevance | 0.1867 | 1.4344 |
| controlled_vs_proposed_raw | persona_consistency | 0.1786 | 1.0854 |
| controlled_vs_proposed_raw | naturalness | 0.0252 | 0.0308 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2439 | 1.5631 |
| controlled_vs_proposed_raw | context_overlap | 0.0531 | 0.7620 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2174 | 3.5115 |
| controlled_vs_proposed_raw | persona_style | 0.0233 | 0.0405 |
| controlled_vs_proposed_raw | distinct1 | 0.0077 | 0.0082 |
| controlled_vs_proposed_raw | length_score | 0.0667 | 0.2094 |
| controlled_vs_proposed_raw | sentence_score | 0.0875 | 0.1108 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0103 | 0.1338 |
| controlled_vs_proposed_raw | overall_quality | 0.1404 | 0.5498 |
| controlled_vs_candidate_no_context | context_relevance | 0.2754 | 6.6464 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1342 | 0.6421 |
| controlled_vs_candidate_no_context | naturalness | 0.0157 | 0.0190 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3614 | 9.3529 |
| controlled_vs_candidate_no_context | context_overlap | 0.0749 | 1.5607 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1650 | 1.4437 |
| controlled_vs_candidate_no_context | persona_style | 0.0108 | 0.0184 |
| controlled_vs_candidate_no_context | distinct1 | 0.0017 | 0.0018 |
| controlled_vs_candidate_no_context | length_score | 0.0400 | 0.1159 |
| controlled_vs_candidate_no_context | sentence_score | 0.0700 | 0.0867 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0584 | 2.0092 |
| controlled_vs_candidate_no_context | overall_quality | 0.1658 | 0.7215 |
| controlled_vs_baseline_no_context | context_relevance | 0.2907 | 11.1362 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1762 | 1.0554 |
| controlled_vs_baseline_no_context | naturalness | -0.0619 | -0.0685 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3780 | 17.2069 |
| controlled_vs_baseline_no_context | context_overlap | 0.0871 | 2.4350 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2110 | 3.0871 |
| controlled_vs_baseline_no_context | persona_style | 0.0370 | 0.0659 |
| controlled_vs_baseline_no_context | distinct1 | -0.0411 | -0.0417 |
| controlled_vs_baseline_no_context | length_score | -0.2100 | -0.3529 |
| controlled_vs_baseline_no_context | sentence_score | -0.0350 | -0.0384 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0292 | 0.5020 |
| controlled_vs_baseline_no_context | overall_quality | 0.1679 | 0.7373 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.2626 | 4.8358 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1698 | 0.9803 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | -0.0703 | -0.0770 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.3436 | 6.0872 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0736 | 1.4923 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.2136 | 3.2500 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | -0.0051 | -0.0085 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0374 | -0.0381 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | -0.2417 | -0.3856 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | -0.0700 | -0.0739 |
| controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0287 | 0.4886 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1536 | 0.6343 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0630 | -0.1988 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0336 | -0.0980 |
| controlled_alt_vs_controlled_default | naturalness | 0.0274 | 0.0326 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0773 | -0.1932 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0296 | -0.2411 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0440 | -0.1577 |
| controlled_alt_vs_controlled_default | persona_style | 0.0081 | 0.0135 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0133 | -0.0141 |
| controlled_alt_vs_controlled_default | length_score | 0.1200 | 0.3117 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0875 | 0.0997 |
| controlled_alt_vs_controlled_default | bertscore_f1 | -0.0358 | -0.4089 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0357 | -0.0903 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1237 | 0.9505 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1449 | 0.8810 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0526 | 0.0644 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.1667 | 1.0680 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0235 | 0.3371 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1733 | 2.8000 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0314 | 0.0546 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0057 | -0.0061 |
| controlled_alt_vs_proposed_raw | length_score | 0.1867 | 0.5864 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1750 | 0.2215 |
| controlled_alt_vs_proposed_raw | bertscore_f1 | -0.0254 | -0.3299 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1047 | 0.4099 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2124 | 5.1266 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1005 | 0.4812 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0431 | 0.0522 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.2841 | 7.3529 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0453 | 0.9432 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1210 | 1.0583 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0188 | 0.0321 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0116 | -0.0123 |
| controlled_alt_vs_candidate_no_context | length_score | 0.1600 | 0.4638 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1575 | 0.1950 |
| controlled_alt_vs_candidate_no_context | bertscore_f1 | 0.0226 | 0.7786 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1301 | 0.5661 |
| controlled_alt_vs_baseline_no_context | context_relevance | 0.2278 | 8.7240 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 0.1425 | 0.8539 |
| controlled_alt_vs_baseline_no_context | naturalness | -0.0345 | -0.0382 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 0.3008 | 13.6897 |
| controlled_alt_vs_baseline_no_context | context_overlap | 0.0575 | 1.6067 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 0.1669 | 2.4425 |
| controlled_alt_vs_baseline_no_context | persona_style | 0.0451 | 0.0803 |
| controlled_alt_vs_baseline_no_context | distinct1 | -0.0544 | -0.0553 |
| controlled_alt_vs_baseline_no_context | length_score | -0.0900 | -0.1513 |
| controlled_alt_vs_baseline_no_context | sentence_score | 0.0525 | 0.0575 |
| controlled_alt_vs_baseline_no_context | bertscore_f1 | -0.0065 | -0.1122 |
| controlled_alt_vs_baseline_no_context | overall_quality | 0.1322 | 0.5805 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_relevance | 0.1996 | 3.6759 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1362 | 0.7862 |
| controlled_alt_vs_baseline_no_context_phi3_latest | naturalness | -0.0429 | -0.0470 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.2663 | 4.7181 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_overlap | 0.0439 | 0.8913 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1695 | 2.5797 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_style | 0.0029 | 0.0049 |
| controlled_alt_vs_baseline_no_context_phi3_latest | distinct1 | -0.0507 | -0.0517 |
| controlled_alt_vs_baseline_no_context_phi3_latest | length_score | -0.1217 | -0.1941 |
| controlled_alt_vs_baseline_no_context_phi3_latest | sentence_score | 0.0175 | 0.0185 |
| controlled_alt_vs_baseline_no_context_phi3_latest | bertscore_f1 | -0.0071 | -0.1201 |
| controlled_alt_vs_baseline_no_context_phi3_latest | overall_quality | 0.1179 | 0.4868 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2907 | 11.1362 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1762 | 1.0554 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0619 | -0.0685 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3780 | 17.2069 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0871 | 2.4350 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2110 | 3.0871 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0370 | 0.0659 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0411 | -0.0417 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | -0.2100 | -0.3529 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | -0.0350 | -0.0384 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0292 | 0.5020 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1679 | 0.7373 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.2626 | 4.8358 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1698 | 0.9803 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | -0.0703 | -0.0770 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.3436 | 6.0872 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0736 | 1.4923 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.2136 | 3.2500 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | -0.0051 | -0.0085 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0374 | -0.0381 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | -0.2417 | -0.3856 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | -0.0700 | -0.0739 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0287 | 0.4886 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1536 | 0.6343 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0887 | (0.0195, 0.1672) | 0.0050 | 0.0887 | (0.0148, 0.1601) | 0.0077 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0444 | (-0.1141, 0.0170) | 0.9007 | -0.0444 | (-0.1244, 0.0010) | 0.9690 |
| proposed_vs_candidate_no_context | naturalness | -0.0095 | (-0.0512, 0.0318) | 0.6750 | -0.0095 | (-0.0719, 0.0242) | 0.6640 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.1174 | (0.0318, 0.2186) | 0.0030 | 0.1174 | (0.0182, 0.2064) | 0.0130 |
| proposed_vs_candidate_no_context | context_overlap | 0.0217 | (0.0009, 0.0437) | 0.0220 | 0.0217 | (-0.0016, 0.0430) | 0.0330 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0524 | (-0.1357, 0.0167) | 0.9097 | -0.0524 | (-0.1299, -0.0055) | 1.0000 |
| proposed_vs_candidate_no_context | persona_style | -0.0125 | (-0.0672, 0.0578) | 0.6890 | -0.0125 | (-0.1010, 0.0408) | 0.6673 |
| proposed_vs_candidate_no_context | distinct1 | -0.0060 | (-0.0230, 0.0113) | 0.7497 | -0.0060 | (-0.0286, 0.0114) | 0.7420 |
| proposed_vs_candidate_no_context | length_score | -0.0267 | (-0.1750, 0.1250) | 0.6363 | -0.0267 | (-0.2524, 0.1000) | 0.6143 |
| proposed_vs_candidate_no_context | sentence_score | -0.0175 | (-0.1050, 0.0700) | 0.7340 | -0.0175 | (-0.1029, 0.0618) | 0.7537 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0481 | (0.0209, 0.0779) | 0.0000 | 0.0481 | (0.0169, 0.0795) | 0.0000 |
| proposed_vs_candidate_no_context | overall_quality | 0.0255 | (-0.0246, 0.0764) | 0.1720 | 0.0255 | (-0.0285, 0.0673) | 0.1747 |
| proposed_vs_baseline_no_context | context_relevance | 0.1041 | (0.0337, 0.1841) | 0.0013 | 0.1041 | (0.0090, 0.1871) | 0.0143 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0024 | (-0.0542, 0.0576) | 0.5527 | -0.0024 | (-0.0805, 0.0479) | 0.5753 |
| proposed_vs_baseline_no_context | naturalness | -0.0871 | (-0.1230, -0.0492) | 1.0000 | -0.0871 | (-0.1195, -0.0472) | 0.9997 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.1341 | (0.0348, 0.2413) | 0.0020 | 0.1341 | (0.0152, 0.2500) | 0.0147 |
| proposed_vs_baseline_no_context | context_overlap | 0.0340 | (0.0141, 0.0551) | 0.0000 | 0.0340 | (0.0114, 0.0550) | 0.0000 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0064 | (-0.0636, 0.0619) | 0.5923 | -0.0064 | (-0.0857, 0.0538) | 0.5920 |
| proposed_vs_baseline_no_context | persona_style | 0.0137 | (-0.0740, 0.0992) | 0.3920 | 0.0137 | (-0.1522, 0.1191) | 0.4047 |
| proposed_vs_baseline_no_context | distinct1 | -0.0487 | (-0.0624, -0.0353) | 1.0000 | -0.0487 | (-0.0618, -0.0350) | 1.0000 |
| proposed_vs_baseline_no_context | length_score | -0.2767 | (-0.4183, -0.1233) | 0.9997 | -0.2767 | (-0.4245, -0.0823) | 0.9940 |
| proposed_vs_baseline_no_context | sentence_score | -0.1225 | (-0.2100, -0.0350) | 0.9977 | -0.1225 | (-0.2026, -0.0368) | 1.0000 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0189 | (-0.0081, 0.0496) | 0.0907 | 0.0189 | (-0.0211, 0.0481) | 0.1727 |
| proposed_vs_baseline_no_context | overall_quality | 0.0276 | (-0.0181, 0.0755) | 0.1270 | 0.0276 | (-0.0365, 0.0796) | 0.2103 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 0.0759 | (0.0159, 0.1488) | 0.0057 | 0.0759 | (0.0159, 0.1333) | 0.0020 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | -0.0087 | (-0.0562, 0.0521) | 0.6460 | -0.0087 | (-0.0769, 0.0486) | 0.5923 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | -0.0954 | (-0.1286, -0.0591) | 1.0000 | -0.0954 | (-0.1245, -0.0470) | 0.9983 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.0996 | (0.0197, 0.1879) | 0.0050 | 0.0996 | (0.0191, 0.1763) | 0.0050 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 0.0204 | (0.0002, 0.0401) | 0.0247 | 0.0204 | (0.0070, 0.0341) | 0.0003 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | -0.0038 | (-0.0600, 0.0674) | 0.5780 | -0.0038 | (-0.0874, 0.0650) | 0.5367 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | -0.0284 | (-0.0844, 0.0258) | 0.8447 | -0.0284 | (-0.1010, 0.0075) | 0.8777 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | -0.0450 | (-0.0638, -0.0262) | 1.0000 | -0.0450 | (-0.0695, -0.0216) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | -0.3083 | (-0.4567, -0.1566) | 1.0000 | -0.3083 | (-0.4167, -0.1067) | 0.9950 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | -0.1575 | (-0.2275, -0.0875) | 1.0000 | -0.1575 | (-0.2333, -0.0778) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0184 | (-0.0095, 0.0471) | 0.1107 | 0.0184 | (-0.0223, 0.0584) | 0.2497 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | 0.0132 | (-0.0271, 0.0579) | 0.2880 | 0.0132 | (-0.0357, 0.0561) | 0.3070 |
| controlled_vs_proposed_raw | context_relevance | 0.1867 | (0.1158, 0.2541) | 0.0000 | 0.1867 | (0.0963, 0.2695) | 0.0003 |
| controlled_vs_proposed_raw | persona_consistency | 0.1786 | (0.1037, 0.2597) | 0.0000 | 0.1786 | (0.1331, 0.2362) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0252 | (-0.0142, 0.0632) | 0.1133 | 0.0252 | (-0.0212, 0.0826) | 0.1890 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2439 | (0.1560, 0.3269) | 0.0000 | 0.2439 | (0.1279, 0.3550) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0531 | (0.0243, 0.0866) | 0.0000 | 0.0531 | (0.0274, 0.0755) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2174 | (0.1293, 0.3064) | 0.0000 | 0.2174 | (0.1644, 0.2845) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.0233 | (-0.0225, 0.0703) | 0.1470 | 0.0233 | (-0.0019, 0.0733) | 0.1000 |
| controlled_vs_proposed_raw | distinct1 | 0.0077 | (-0.0128, 0.0288) | 0.2263 | 0.0077 | (-0.0147, 0.0371) | 0.3197 |
| controlled_vs_proposed_raw | length_score | 0.0667 | (-0.0867, 0.2284) | 0.1923 | 0.0667 | (-0.0926, 0.2857) | 0.2483 |
| controlled_vs_proposed_raw | sentence_score | 0.0875 | (0.0000, 0.1925) | 0.0513 | 0.0875 | (0.0000, 0.1826) | 0.0553 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0103 | (-0.0174, 0.0408) | 0.2440 | 0.0103 | (-0.0141, 0.0308) | 0.1593 |
| controlled_vs_proposed_raw | overall_quality | 0.1404 | (0.0943, 0.1861) | 0.0000 | 0.1404 | (0.0883, 0.1898) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2754 | (0.2245, 0.3295) | 0.0000 | 0.2754 | (0.2245, 0.3373) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1342 | (0.0710, 0.1948) | 0.0000 | 0.1342 | (0.0542, 0.2078) | 0.0013 |
| controlled_vs_candidate_no_context | naturalness | 0.0157 | (-0.0238, 0.0538) | 0.2070 | 0.0157 | (-0.0475, 0.0682) | 0.3087 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3614 | (0.2916, 0.4288) | 0.0000 | 0.3614 | (0.2941, 0.4455) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0749 | (0.0520, 0.1026) | 0.0000 | 0.0749 | (0.0571, 0.0913) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1650 | (0.0948, 0.2367) | 0.0000 | 0.1650 | (0.0778, 0.2531) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.0108 | (-0.0667, 0.0740) | 0.3563 | 0.0108 | (-0.0769, 0.0706) | 0.4007 |
| controlled_vs_candidate_no_context | distinct1 | 0.0017 | (-0.0214, 0.0253) | 0.4393 | 0.0017 | (-0.0339, 0.0352) | 0.4703 |
| controlled_vs_candidate_no_context | length_score | 0.0400 | (-0.1133, 0.1917) | 0.3023 | 0.0400 | (-0.1976, 0.2204) | 0.3520 |
| controlled_vs_candidate_no_context | sentence_score | 0.0700 | (-0.0350, 0.1750) | 0.1193 | 0.0700 | (-0.0500, 0.1750) | 0.1287 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0584 | (0.0276, 0.0907) | 0.0000 | 0.0584 | (0.0213, 0.0870) | 0.0000 |
| controlled_vs_candidate_no_context | overall_quality | 0.1658 | (0.1390, 0.1940) | 0.0000 | 0.1658 | (0.1424, 0.1855) | 0.0000 |
| controlled_vs_baseline_no_context | context_relevance | 0.2907 | (0.2431, 0.3404) | 0.0000 | 0.2907 | (0.2329, 0.3532) | 0.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1762 | (0.1163, 0.2379) | 0.0000 | 0.1762 | (0.1249, 0.2011) | 0.0000 |
| controlled_vs_baseline_no_context | naturalness | -0.0619 | (-0.0982, -0.0231) | 0.9993 | -0.0619 | (-0.0976, -0.0114) | 0.9880 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3780 | (0.3140, 0.4451) | 0.0000 | 0.3780 | (0.2971, 0.4554) | 0.0000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0871 | (0.0689, 0.1096) | 0.0000 | 0.0871 | (0.0698, 0.1008) | 0.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2110 | (0.1374, 0.2848) | 0.0000 | 0.2110 | (0.1513, 0.2515) | 0.0000 |
| controlled_vs_baseline_no_context | persona_style | 0.0370 | (-0.0343, 0.1179) | 0.1657 | 0.0370 | (-0.0769, 0.1239) | 0.3250 |
| controlled_vs_baseline_no_context | distinct1 | -0.0411 | (-0.0601, -0.0215) | 1.0000 | -0.0411 | (-0.0594, -0.0206) | 1.0000 |
| controlled_vs_baseline_no_context | length_score | -0.2100 | (-0.3883, -0.0167) | 0.9840 | -0.2100 | (-0.3817, 0.0209) | 0.9663 |
| controlled_vs_baseline_no_context | sentence_score | -0.0350 | (-0.1050, 0.0350) | 0.9090 | -0.0350 | (-0.1167, 0.0457) | 0.8510 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0292 | (-0.0059, 0.0629) | 0.0447 | 0.0292 | (-0.0190, 0.0615) | 0.1030 |
| controlled_vs_baseline_no_context | overall_quality | 0.1679 | (0.1412, 0.1947) | 0.0000 | 0.1679 | (0.1387, 0.1955) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.2626 | (0.2209, 0.3049) | 0.0000 | 0.2626 | (0.2150, 0.3210) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1698 | (0.1137, 0.2254) | 0.0000 | 0.1698 | (0.1418, 0.1990) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | -0.0703 | (-0.1110, -0.0296) | 0.9993 | -0.0703 | (-0.1106, -0.0082) | 0.9840 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.3436 | (0.2879, 0.4015) | 0.0000 | 0.3436 | (0.2793, 0.4171) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0736 | (0.0509, 0.0990) | 0.0000 | 0.0736 | (0.0574, 0.0917) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.2136 | (0.1447, 0.2824) | 0.0000 | 0.2136 | (0.1792, 0.2562) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | -0.0051 | (-0.0319, 0.0214) | 0.6357 | -0.0051 | (-0.0294, 0.0071) | 0.7277 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0374 | (-0.0559, -0.0190) | 1.0000 | -0.0374 | (-0.0517, -0.0242) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | -0.2417 | (-0.4283, -0.0417) | 0.9903 | -0.2417 | (-0.4218, 0.0375) | 0.9590 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | -0.0700 | (-0.1575, 0.0000) | 0.9793 | -0.0700 | (-0.1361, 0.0000) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0287 | (-0.0025, 0.0590) | 0.0370 | 0.0287 | (-0.0158, 0.0626) | 0.1063 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1536 | (0.1335, 0.1740) | 0.0000 | 0.1536 | (0.1322, 0.1831) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0630 | (-0.1211, -0.0070) | 0.9843 | -0.0630 | (-0.1394, -0.0137) | 0.9960 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0336 | (-0.1085, 0.0347) | 0.8193 | -0.0336 | (-0.0912, 0.0249) | 0.8310 |
| controlled_alt_vs_controlled_default | naturalness | 0.0274 | (-0.0161, 0.0687) | 0.1047 | 0.0274 | (-0.0093, 0.0547) | 0.0567 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0773 | (-0.1561, -0.0049) | 0.9817 | -0.0773 | (-0.1919, -0.0124) | 0.9947 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0296 | (-0.0548, -0.0089) | 0.9987 | -0.0296 | (-0.0461, -0.0117) | 0.9983 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0440 | (-0.1293, 0.0376) | 0.8477 | -0.0440 | (-0.1143, 0.0310) | 0.8603 |
| controlled_alt_vs_controlled_default | persona_style | 0.0081 | (-0.0305, 0.0489) | 0.3473 | 0.0081 | (-0.0312, 0.0413) | 0.3280 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0133 | (-0.0386, 0.0115) | 0.8447 | -0.0133 | (-0.0301, 0.0015) | 0.9580 |
| controlled_alt_vs_controlled_default | length_score | 0.1200 | (-0.0667, 0.3034) | 0.1077 | 0.1200 | (-0.0846, 0.2619) | 0.0883 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0875 | (0.0000, 0.1750) | 0.0297 | 0.0875 | (0.0000, 0.1633) | 0.0387 |
| controlled_alt_vs_controlled_default | bertscore_f1 | -0.0358 | (-0.0730, 0.0012) | 0.9730 | -0.0358 | (-0.0542, -0.0128) | 0.9957 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0357 | (-0.0624, -0.0122) | 0.9987 | -0.0357 | (-0.0625, -0.0110) | 1.0000 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1237 | (0.0515, 0.1892) | 0.0010 | 0.1237 | (0.0420, 0.1960) | 0.0020 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1449 | (0.0777, 0.2105) | 0.0000 | 0.1449 | (0.1010, 0.2157) | 0.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0526 | (0.0089, 0.0983) | 0.0080 | 0.0526 | (-0.0076, 0.1201) | 0.0550 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.1667 | (0.0693, 0.2553) | 0.0003 | 0.1667 | (0.0586, 0.2652) | 0.0030 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0235 | (0.0072, 0.0388) | 0.0043 | 0.0235 | (0.0096, 0.0368) | 0.0013 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1733 | (0.0879, 0.2521) | 0.0003 | 0.1733 | (0.1234, 0.2590) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0314 | (-0.0112, 0.0777) | 0.0773 | 0.0314 | (-0.0109, 0.0882) | 0.0700 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0057 | (-0.0255, 0.0130) | 0.7220 | -0.0057 | (-0.0248, 0.0146) | 0.6990 |
| controlled_alt_vs_proposed_raw | length_score | 0.1867 | (0.0150, 0.3600) | 0.0173 | 0.1867 | (-0.0400, 0.4333) | 0.0550 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1750 | (0.0875, 0.2625) | 0.0000 | 0.1750 | (0.0525, 0.2834) | 0.0057 |
| controlled_alt_vs_proposed_raw | bertscore_f1 | -0.0254 | (-0.0509, -0.0027) | 0.9870 | -0.0254 | (-0.0378, -0.0142) | 1.0000 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1047 | (0.0652, 0.1408) | 0.0000 | 0.1047 | (0.0735, 0.1380) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2124 | (0.1744, 0.2494) | 0.0000 | 0.2124 | (0.1495, 0.2674) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1005 | (0.0365, 0.1609) | 0.0010 | 0.1005 | (0.0449, 0.1585) | 0.0000 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0431 | (-0.0115, 0.0980) | 0.0580 | 0.0431 | (-0.0328, 0.1088) | 0.1313 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.2841 | (0.2280, 0.3356) | 0.0000 | 0.2841 | (0.1983, 0.3636) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0453 | (0.0335, 0.0566) | 0.0000 | 0.0453 | (0.0272, 0.0583) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1210 | (0.0414, 0.1938) | 0.0010 | 0.1210 | (0.0635, 0.1854) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0188 | (-0.0424, 0.0949) | 0.2987 | 0.0188 | (-0.0542, 0.0753) | 0.2993 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0116 | (-0.0357, 0.0128) | 0.8290 | -0.0116 | (-0.0372, 0.0097) | 0.8283 |
| controlled_alt_vs_candidate_no_context | length_score | 0.1600 | (-0.0500, 0.3583) | 0.0633 | 0.1600 | (-0.1278, 0.4014) | 0.1393 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1575 | (0.0875, 0.2275) | 0.0000 | 0.1575 | (0.0538, 0.2593) | 0.0037 |
| controlled_alt_vs_candidate_no_context | bertscore_f1 | 0.0226 | (-0.0019, 0.0483) | 0.0363 | 0.0226 | (-0.0070, 0.0492) | 0.0600 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1301 | (0.1000, 0.1607) | 0.0000 | 0.1301 | (0.0937, 0.1527) | 0.0000 |
| controlled_alt_vs_baseline_no_context | context_relevance | 0.2278 | (0.1834, 0.2726) | 0.0000 | 0.2278 | (0.1490, 0.2867) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 0.1425 | (0.1016, 0.1845) | 0.0000 | 0.1425 | (0.0809, 0.1922) | 0.0000 |
| controlled_alt_vs_baseline_no_context | naturalness | -0.0345 | (-0.0753, 0.0042) | 0.9587 | -0.0345 | (-0.0758, 0.0151) | 0.9093 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 0.3008 | (0.2386, 0.3617) | 0.0000 | 0.3008 | (0.1948, 0.3864) | 0.0000 |
| controlled_alt_vs_baseline_no_context | context_overlap | 0.0575 | (0.0462, 0.0679) | 0.0000 | 0.0575 | (0.0412, 0.0698) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 0.1669 | (0.1210, 0.2114) | 0.0000 | 0.1669 | (0.1065, 0.2173) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_style | 0.0451 | (-0.0354, 0.1314) | 0.1407 | 0.0451 | (-0.0885, 0.1491) | 0.2713 |
| controlled_alt_vs_baseline_no_context | distinct1 | -0.0544 | (-0.0729, -0.0359) | 1.0000 | -0.0544 | (-0.0674, -0.0386) | 1.0000 |
| controlled_alt_vs_baseline_no_context | length_score | -0.0900 | (-0.2667, 0.0833) | 0.8477 | -0.0900 | (-0.2945, 0.1175) | 0.7717 |
| controlled_alt_vs_baseline_no_context | sentence_score | 0.0525 | (-0.0350, 0.1400) | 0.1547 | 0.0525 | (-0.0500, 0.1432) | 0.1957 |
| controlled_alt_vs_baseline_no_context | bertscore_f1 | -0.0065 | (-0.0361, 0.0232) | 0.6647 | -0.0065 | (-0.0437, 0.0215) | 0.6600 |
| controlled_alt_vs_baseline_no_context | overall_quality | 0.1322 | (0.1082, 0.1591) | 0.0000 | 0.1322 | (0.0909, 0.1648) | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_relevance | 0.1996 | (0.1573, 0.2448) | 0.0000 | 0.1996 | (0.1394, 0.2481) | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1362 | (0.0794, 0.1856) | 0.0000 | 0.1362 | (0.0548, 0.2044) | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | naturalness | -0.0429 | (-0.0863, 0.0012) | 0.9730 | -0.0429 | (-0.0938, 0.0164) | 0.9123 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.2663 | (0.2083, 0.3227) | 0.0000 | 0.2663 | (0.1847, 0.3333) | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_overlap | 0.0439 | (0.0317, 0.0560) | 0.0000 | 0.0439 | (0.0336, 0.0539) | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1695 | (0.0983, 0.2333) | 0.0000 | 0.1695 | (0.0742, 0.2500) | 0.0007 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_style | 0.0029 | (-0.0356, 0.0432) | 0.4623 | 0.0029 | (-0.0461, 0.0396) | 0.4690 |
| controlled_alt_vs_baseline_no_context_phi3_latest | distinct1 | -0.0507 | (-0.0725, -0.0272) | 1.0000 | -0.0507 | (-0.0672, -0.0343) | 1.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | length_score | -0.1217 | (-0.3300, 0.0850) | 0.8723 | -0.1217 | (-0.3681, 0.1533) | 0.7730 |
| controlled_alt_vs_baseline_no_context_phi3_latest | sentence_score | 0.0175 | (-0.0525, 0.0875) | 0.3987 | 0.0175 | (-0.0750, 0.1168) | 0.4423 |
| controlled_alt_vs_baseline_no_context_phi3_latest | bertscore_f1 | -0.0071 | (-0.0349, 0.0207) | 0.6997 | -0.0071 | (-0.0475, 0.0245) | 0.6770 |
| controlled_alt_vs_baseline_no_context_phi3_latest | overall_quality | 0.1179 | (0.0952, 0.1394) | 0.0000 | 0.1179 | (0.0867, 0.1489) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2907 | (0.2416, 0.3386) | 0.0000 | 0.2907 | (0.2299, 0.3514) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1762 | (0.1133, 0.2390) | 0.0000 | 0.1762 | (0.1249, 0.2008) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0619 | (-0.0993, -0.0211) | 0.9993 | -0.0619 | (-0.0974, -0.0078) | 0.9857 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3780 | (0.3159, 0.4432) | 0.0000 | 0.3780 | (0.2970, 0.4590) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0871 | (0.0680, 0.1098) | 0.0000 | 0.0871 | (0.0694, 0.1013) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2110 | (0.1390, 0.2857) | 0.0000 | 0.2110 | (0.1465, 0.2508) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0370 | (-0.0375, 0.1132) | 0.1830 | 0.0370 | (-0.0769, 0.1292) | 0.3003 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0411 | (-0.0610, -0.0214) | 1.0000 | -0.0411 | (-0.0587, -0.0212) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | -0.2100 | (-0.3800, -0.0217) | 0.9863 | -0.2100 | (-0.3795, 0.0319) | 0.9567 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | -0.0350 | (-0.1050, 0.0350) | 0.9053 | -0.0350 | (-0.1167, 0.0437) | 0.8513 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0292 | (-0.0031, 0.0620) | 0.0380 | 0.0292 | (-0.0211, 0.0607) | 0.1067 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1679 | (0.1418, 0.1961) | 0.0000 | 0.1679 | (0.1394, 0.1951) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.2626 | (0.2198, 0.3057) | 0.0000 | 0.2626 | (0.2158, 0.3189) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1698 | (0.1141, 0.2263) | 0.0000 | 0.1698 | (0.1411, 0.1994) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | -0.0703 | (-0.1105, -0.0301) | 0.9990 | -0.0703 | (-0.1105, -0.0094) | 0.9863 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.3436 | (0.2856, 0.4027) | 0.0000 | 0.3436 | (0.2813, 0.4182) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0736 | (0.0522, 0.0988) | 0.0000 | 0.0736 | (0.0567, 0.0924) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.2136 | (0.1445, 0.2836) | 0.0000 | 0.2136 | (0.1782, 0.2554) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | -0.0051 | (-0.0309, 0.0214) | 0.6527 | -0.0051 | (-0.0294, 0.0074) | 0.7287 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0374 | (-0.0558, -0.0186) | 1.0000 | -0.0374 | (-0.0528, -0.0238) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | -0.2417 | (-0.4284, -0.0450) | 0.9893 | -0.2417 | (-0.4232, 0.0312) | 0.9633 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | -0.0700 | (-0.1575, 0.0000) | 0.9793 | -0.0700 | (-0.1361, -0.0152) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0287 | (-0.0019, 0.0601) | 0.0333 | 0.0287 | (-0.0141, 0.0631) | 0.1023 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1536 | (0.1329, 0.1753) | 0.0000 | 0.1536 | (0.1319, 0.1815) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 9 | 4 | 7 | 0.6250 | 0.6923 |
| proposed_vs_candidate_no_context | persona_consistency | 2 | 7 | 11 | 0.3750 | 0.2222 |
| proposed_vs_candidate_no_context | naturalness | 6 | 7 | 7 | 0.4750 | 0.4615 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 9 | 2 | 9 | 0.6750 | 0.8182 |
| proposed_vs_candidate_no_context | context_overlap | 9 | 4 | 7 | 0.6250 | 0.6923 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 2 | 4 | 14 | 0.4500 | 0.3333 |
| proposed_vs_candidate_no_context | persona_style | 1 | 4 | 15 | 0.4250 | 0.2000 |
| proposed_vs_candidate_no_context | distinct1 | 7 | 6 | 7 | 0.5250 | 0.5385 |
| proposed_vs_candidate_no_context | length_score | 6 | 7 | 7 | 0.4750 | 0.4615 |
| proposed_vs_candidate_no_context | sentence_score | 3 | 4 | 13 | 0.4750 | 0.4286 |
| proposed_vs_candidate_no_context | bertscore_f1 | 11 | 4 | 5 | 0.6750 | 0.7333 |
| proposed_vs_candidate_no_context | overall_quality | 7 | 8 | 5 | 0.4750 | 0.4667 |
| proposed_vs_baseline_no_context | context_relevance | 12 | 8 | 0 | 0.6000 | 0.6000 |
| proposed_vs_baseline_no_context | persona_consistency | 6 | 6 | 8 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context | naturalness | 3 | 17 | 0 | 0.1500 | 0.1500 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 9 | 3 | 8 | 0.6500 | 0.7500 |
| proposed_vs_baseline_no_context | context_overlap | 13 | 7 | 0 | 0.6500 | 0.6500 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 2 | 4 | 14 | 0.4500 | 0.3333 |
| proposed_vs_baseline_no_context | persona_style | 4 | 3 | 13 | 0.5250 | 0.5714 |
| proposed_vs_baseline_no_context | distinct1 | 0 | 19 | 1 | 0.0250 | 0.0000 |
| proposed_vs_baseline_no_context | length_score | 3 | 17 | 0 | 0.1500 | 0.1500 |
| proposed_vs_baseline_no_context | sentence_score | 1 | 8 | 11 | 0.3250 | 0.1111 |
| proposed_vs_baseline_no_context | bertscore_f1 | 7 | 13 | 0 | 0.3500 | 0.3500 |
| proposed_vs_baseline_no_context | overall_quality | 9 | 11 | 0 | 0.4500 | 0.4500 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 13 | 7 | 0 | 0.6500 | 0.6500 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | 4 | 8 | 8 | 0.4000 | 0.3333 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | 2 | 18 | 0 | 0.1000 | 0.1000 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 9 | 3 | 8 | 0.6500 | 0.7500 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 14 | 6 | 0 | 0.7000 | 0.7000 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 2 | 4 | 14 | 0.4500 | 0.3333 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | 2 | 5 | 13 | 0.4250 | 0.2857 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | 3 | 16 | 1 | 0.1750 | 0.1579 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | 3 | 17 | 0 | 0.1500 | 0.1500 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | 0 | 9 | 11 | 0.2750 | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | bertscore_f1 | 9 | 11 | 0 | 0.4500 | 0.4500 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | 9 | 11 | 0 | 0.4500 | 0.4500 |
| controlled_vs_proposed_raw | context_relevance | 17 | 3 | 0 | 0.8500 | 0.8500 |
| controlled_vs_proposed_raw | persona_consistency | 17 | 3 | 0 | 0.8500 | 0.8500 |
| controlled_vs_proposed_raw | naturalness | 14 | 6 | 0 | 0.7000 | 0.7000 |
| controlled_vs_proposed_raw | context_keyword_coverage | 17 | 2 | 1 | 0.8750 | 0.8947 |
| controlled_vs_proposed_raw | context_overlap | 15 | 5 | 0 | 0.7500 | 0.7500 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 17 | 2 | 1 | 0.8750 | 0.8947 |
| controlled_vs_proposed_raw | persona_style | 5 | 2 | 13 | 0.5750 | 0.7143 |
| controlled_vs_proposed_raw | distinct1 | 9 | 11 | 0 | 0.4500 | 0.4500 |
| controlled_vs_proposed_raw | length_score | 11 | 8 | 1 | 0.5750 | 0.5789 |
| controlled_vs_proposed_raw | sentence_score | 7 | 2 | 11 | 0.6250 | 0.7778 |
| controlled_vs_proposed_raw | bertscore_f1 | 11 | 9 | 0 | 0.5500 | 0.5500 |
| controlled_vs_proposed_raw | overall_quality | 17 | 3 | 0 | 0.8500 | 0.8500 |
| controlled_vs_candidate_no_context | context_relevance | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 16 | 2 | 2 | 0.8500 | 0.8889 |
| controlled_vs_candidate_no_context | naturalness | 13 | 7 | 0 | 0.6500 | 0.6500 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 16 | 2 | 2 | 0.8500 | 0.8889 |
| controlled_vs_candidate_no_context | persona_style | 3 | 1 | 16 | 0.5500 | 0.7500 |
| controlled_vs_candidate_no_context | distinct1 | 11 | 9 | 0 | 0.5500 | 0.5500 |
| controlled_vs_candidate_no_context | length_score | 10 | 8 | 2 | 0.5500 | 0.5556 |
| controlled_vs_candidate_no_context | sentence_score | 7 | 3 | 10 | 0.6000 | 0.7000 |
| controlled_vs_candidate_no_context | bertscore_f1 | 16 | 4 | 0 | 0.8000 | 0.8000 |
| controlled_vs_candidate_no_context | overall_quality | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | context_relevance | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 16 | 1 | 3 | 0.8750 | 0.9412 |
| controlled_vs_baseline_no_context | naturalness | 4 | 16 | 0 | 0.2000 | 0.2000 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | context_overlap | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 16 | 1 | 3 | 0.8750 | 0.9412 |
| controlled_vs_baseline_no_context | persona_style | 4 | 3 | 13 | 0.5250 | 0.5714 |
| controlled_vs_baseline_no_context | distinct1 | 2 | 16 | 2 | 0.1500 | 0.1111 |
| controlled_vs_baseline_no_context | length_score | 5 | 15 | 0 | 0.2500 | 0.2500 |
| controlled_vs_baseline_no_context | sentence_score | 1 | 3 | 16 | 0.4500 | 0.2500 |
| controlled_vs_baseline_no_context | bertscore_f1 | 15 | 5 | 0 | 0.7500 | 0.7500 |
| controlled_vs_baseline_no_context | overall_quality | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 17 | 1 | 2 | 0.9000 | 0.9444 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | 5 | 15 | 0 | 0.2500 | 0.2500 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 18 | 2 | 0 | 0.9000 | 0.9000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 17 | 1 | 2 | 0.9000 | 0.9444 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | 1 | 2 | 17 | 0.4750 | 0.3333 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | 3 | 15 | 2 | 0.2000 | 0.1667 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | 6 | 14 | 0 | 0.3000 | 0.3000 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 1 | 5 | 14 | 0.4000 | 0.1667 |
| controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 14 | 6 | 0 | 0.7000 | 0.7000 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 8 | 12 | 0 | 0.4000 | 0.4000 |
| controlled_alt_vs_controlled_default | persona_consistency | 7 | 9 | 4 | 0.4500 | 0.4375 |
| controlled_alt_vs_controlled_default | naturalness | 12 | 8 | 0 | 0.6000 | 0.6000 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 5 | 10 | 5 | 0.3750 | 0.3333 |
| controlled_alt_vs_controlled_default | context_overlap | 8 | 12 | 0 | 0.4000 | 0.4000 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 5 | 8 | 7 | 0.4250 | 0.3846 |
| controlled_alt_vs_controlled_default | persona_style | 3 | 3 | 14 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | distinct1 | 9 | 11 | 0 | 0.4500 | 0.4500 |
| controlled_alt_vs_controlled_default | length_score | 11 | 9 | 0 | 0.5500 | 0.5500 |
| controlled_alt_vs_controlled_default | sentence_score | 6 | 1 | 13 | 0.6250 | 0.8571 |
| controlled_alt_vs_controlled_default | bertscore_f1 | 7 | 13 | 0 | 0.3500 | 0.3500 |
| controlled_alt_vs_controlled_default | overall_quality | 5 | 15 | 0 | 0.2500 | 0.2500 |
| controlled_alt_vs_proposed_raw | context_relevance | 16 | 4 | 0 | 0.8000 | 0.8000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 15 | 1 | 4 | 0.8500 | 0.9375 |
| controlled_alt_vs_proposed_raw | naturalness | 15 | 5 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 15 | 3 | 2 | 0.8000 | 0.8333 |
| controlled_alt_vs_proposed_raw | context_overlap | 13 | 7 | 0 | 0.6500 | 0.6500 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 15 | 1 | 4 | 0.8500 | 0.9375 |
| controlled_alt_vs_proposed_raw | persona_style | 4 | 1 | 15 | 0.5750 | 0.8000 |
| controlled_alt_vs_proposed_raw | distinct1 | 8 | 12 | 0 | 0.4000 | 0.4000 |
| controlled_alt_vs_proposed_raw | length_score | 11 | 8 | 1 | 0.5750 | 0.5789 |
| controlled_alt_vs_proposed_raw | sentence_score | 11 | 1 | 8 | 0.7500 | 0.9167 |
| controlled_alt_vs_proposed_raw | bertscore_f1 | 6 | 14 | 0 | 0.3000 | 0.3000 |
| controlled_alt_vs_proposed_raw | overall_quality | 17 | 3 | 0 | 0.8500 | 0.8500 |
| controlled_alt_vs_candidate_no_context | context_relevance | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 13 | 2 | 5 | 0.7750 | 0.8667 |
| controlled_alt_vs_candidate_no_context | naturalness | 11 | 9 | 0 | 0.5500 | 0.5500 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 19 | 0 | 1 | 0.9750 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 13 | 2 | 5 | 0.7750 | 0.8667 |
| controlled_alt_vs_candidate_no_context | persona_style | 2 | 3 | 15 | 0.4750 | 0.4000 |
| controlled_alt_vs_candidate_no_context | distinct1 | 7 | 13 | 0 | 0.3500 | 0.3500 |
| controlled_alt_vs_candidate_no_context | length_score | 11 | 9 | 0 | 0.5500 | 0.5500 |
| controlled_alt_vs_candidate_no_context | sentence_score | 9 | 0 | 11 | 0.7250 | 1.0000 |
| controlled_alt_vs_candidate_no_context | bertscore_f1 | 14 | 6 | 0 | 0.7000 | 0.7000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_alt_vs_baseline_no_context | context_relevance | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 16 | 0 | 4 | 0.9000 | 1.0000 |
| controlled_alt_vs_baseline_no_context | naturalness | 9 | 11 | 0 | 0.4500 | 0.4500 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 19 | 0 | 1 | 0.9750 | 1.0000 |
| controlled_alt_vs_baseline_no_context | context_overlap | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 16 | 0 | 4 | 0.9000 | 1.0000 |
| controlled_alt_vs_baseline_no_context | persona_style | 5 | 3 | 12 | 0.5500 | 0.6250 |
| controlled_alt_vs_baseline_no_context | distinct1 | 2 | 18 | 0 | 0.1000 | 0.1000 |
| controlled_alt_vs_baseline_no_context | length_score | 10 | 10 | 0 | 0.5000 | 0.5000 |
| controlled_alt_vs_baseline_no_context | sentence_score | 5 | 2 | 13 | 0.5750 | 0.7143 |
| controlled_alt_vs_baseline_no_context | bertscore_f1 | 9 | 11 | 0 | 0.4500 | 0.4500 |
| controlled_alt_vs_baseline_no_context | overall_quality | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_relevance | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_consistency | 17 | 2 | 1 | 0.8750 | 0.8947 |
| controlled_alt_vs_baseline_no_context_phi3_latest | naturalness | 7 | 13 | 0 | 0.3500 | 0.3500 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 19 | 0 | 1 | 0.9750 | 1.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_overlap | 17 | 3 | 0 | 0.8500 | 0.8500 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 17 | 2 | 1 | 0.8750 | 0.8947 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_style | 2 | 3 | 15 | 0.4750 | 0.4000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | distinct1 | 6 | 14 | 0 | 0.3000 | 0.3000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | length_score | 8 | 11 | 1 | 0.4250 | 0.4211 |
| controlled_alt_vs_baseline_no_context_phi3_latest | sentence_score | 3 | 2 | 15 | 0.5250 | 0.6000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | bertscore_f1 | 9 | 11 | 0 | 0.4500 | 0.4500 |
| controlled_alt_vs_baseline_no_context_phi3_latest | overall_quality | 20 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 20 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 16 | 1 | 3 | 0.8750 | 0.9412 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 4 | 16 | 0 | 0.2000 | 0.2000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 20 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 20 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 16 | 1 | 3 | 0.8750 | 0.9412 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 4 | 3 | 13 | 0.5250 | 0.5714 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | 2 | 16 | 2 | 0.1500 | 0.1111 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 5 | 15 | 0 | 0.2500 | 0.2500 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 1 | 3 | 16 | 0.4500 | 0.2500 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 15 | 5 | 0 | 0.7500 | 0.7500 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 20 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 20 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 17 | 1 | 2 | 0.9000 | 0.9444 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | 5 | 15 | 0 | 0.2500 | 0.2500 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 20 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 18 | 2 | 0 | 0.9000 | 0.9000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 17 | 1 | 2 | 0.9000 | 0.9444 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | 1 | 2 | 17 | 0.4750 | 0.3333 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | 3 | 15 | 2 | 0.2000 | 0.1667 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | 6 | 14 | 0 | 0.3000 | 0.3000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | 1 | 5 | 14 | 0.4000 | 0.1667 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 14 | 6 | 0 | 0.7000 | 0.7000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 20 | 0 | 0 | 1.0000 | 1.0000 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.0500 | 0.4500 | 0.5500 |
| proposed_contextual_controlled_runtime | 0.0000 | 0.0000 | 0.4000 | 0.5000 | 0.5000 |
| proposed_contextual | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.4500 | 0.0000 | 0.0000 |
| baseline_no_context | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline_no_context_phi3_latest | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `8`
- Unique template signatures: `18`
- Template signature ratio: `0.9000`
- Effective sample size by source clustering: `5.88`
- Effective sample size by template-signature clustering: `16.67`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: enabled.

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
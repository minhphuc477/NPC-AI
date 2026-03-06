# Proposal Alignment Evaluation Report

- Run ID: `20260305T190550Z`
- Generated: `2026-03-05T19:12:18.322474+00:00`
- Scenarios: `artifacts\proposal_control_tuning\profile_compare_quality20\20260305T190550Z\scenarios.jsonl`
- Scenario count: `20`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_quality`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off
- `baseline_no_context`: model `phi3:mini`, dynamic_context=off, response_control=off
- `baseline_no_context_phi3_latest`: model `phi3:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2731 (0.2296, 0.3204) | 0.3315 (0.2773, 0.3871) | 0.8973 (0.8651, 0.9249) | 0.3839 (0.3569, 0.4106) | 0.0835 |
| proposed_contextual_controlled_quality | 0.2771 (0.2533, 0.3035) | 0.3110 (0.2599, 0.3632) | 0.8802 (0.8454, 0.9100) | 0.3763 (0.3570, 0.3966) | 0.0908 |
| proposed_contextual | 0.0999 (0.0481, 0.1546) | 0.1911 (0.1276, 0.2607) | 0.8246 (0.7934, 0.8576) | 0.2515 (0.2154, 0.2883) | 0.0645 |
| candidate_no_context | 0.0242 (0.0117, 0.0409) | 0.1752 (0.1145, 0.2421) | 0.7923 (0.7673, 0.8221) | 0.2055 (0.1809, 0.2342) | 0.0203 |
| baseline_no_context | 0.0322 (0.0171, 0.0506) | 0.1820 (0.1363, 0.2323) | 0.8887 (0.8663, 0.9113) | 0.2293 (0.2120, 0.2487) | 0.0375 |
| baseline_no_context_phi3_latest | 0.0261 (0.0133, 0.0412) | 0.1736 (0.1177, 0.2345) | 0.8989 (0.8757, 0.9219) | 0.2274 (0.2069, 0.2490) | 0.0459 |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0757 | 3.1321 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0158 | 0.0903 |
| proposed_vs_candidate_no_context | naturalness | 0.0323 | 0.0408 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0985 | 5.6522 |
| proposed_vs_candidate_no_context | context_overlap | 0.0226 | 0.5658 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | persona_style | 0.0791 | 0.1539 |
| proposed_vs_candidate_no_context | distinct1 | 0.0119 | 0.0127 |
| proposed_vs_candidate_no_context | length_score | 0.1117 | 0.5194 |
| proposed_vs_candidate_no_context | sentence_score | 0.0525 | 0.0695 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0442 | 2.1771 |
| proposed_vs_candidate_no_context | overall_quality | 0.0460 | 0.2238 |
| proposed_vs_baseline_no_context | context_relevance | 0.0677 | 2.1031 |
| proposed_vs_baseline_no_context | persona_consistency | 0.0091 | 0.0501 |
| proposed_vs_baseline_no_context | naturalness | -0.0641 | -0.0721 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0902 | 3.5000 |
| proposed_vs_baseline_no_context | context_overlap | 0.0153 | 0.3245 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 0.0110 | 0.1377 |
| proposed_vs_baseline_no_context | persona_style | 0.0018 | 0.0030 |
| proposed_vs_baseline_no_context | distinct1 | -0.0324 | -0.0331 |
| proposed_vs_baseline_no_context | length_score | -0.2117 | -0.3932 |
| proposed_vs_baseline_no_context | sentence_score | -0.0875 | -0.0978 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0270 | 0.7207 |
| proposed_vs_baseline_no_context | overall_quality | 0.0222 | 0.0970 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 0.0737 | 2.8211 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | 0.0175 | 0.1007 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | -0.0743 | -0.0826 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.0943 | 4.3684 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 0.0258 | 0.7006 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.0181 | 0.2500 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | 0.0151 | 0.0260 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | -0.0365 | -0.0372 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | -0.2633 | -0.4463 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | -0.0700 | -0.0798 |
| proposed_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0186 | 0.4063 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | 0.0242 | 0.1063 |
| controlled_vs_proposed_raw | context_relevance | 0.1733 | 1.7344 |
| controlled_vs_proposed_raw | persona_consistency | 0.1404 | 0.7349 |
| controlled_vs_proposed_raw | naturalness | 0.0727 | 0.0882 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2303 | 1.9869 |
| controlled_vs_proposed_raw | context_overlap | 0.0401 | 0.6419 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1743 | 1.9263 |
| controlled_vs_proposed_raw | persona_style | 0.0049 | 0.0083 |
| controlled_vs_proposed_raw | distinct1 | -0.0074 | -0.0078 |
| controlled_vs_proposed_raw | length_score | 0.3083 | 0.9439 |
| controlled_vs_proposed_raw | sentence_score | 0.1400 | 0.1734 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0190 | 0.2939 |
| controlled_vs_proposed_raw | overall_quality | 0.1324 | 0.5263 |
| controlled_vs_candidate_no_context | context_relevance | 0.2490 | 10.2987 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1562 | 0.8916 |
| controlled_vs_candidate_no_context | naturalness | 0.1050 | 0.1326 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3288 | 18.8696 |
| controlled_vs_candidate_no_context | context_overlap | 0.0627 | 1.5709 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1743 | 1.9263 |
| controlled_vs_candidate_no_context | persona_style | 0.0840 | 0.1634 |
| controlled_vs_candidate_no_context | distinct1 | 0.0045 | 0.0048 |
| controlled_vs_candidate_no_context | length_score | 0.4200 | 1.9535 |
| controlled_vs_candidate_no_context | sentence_score | 0.1925 | 0.2550 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0632 | 3.1108 |
| controlled_vs_candidate_no_context | overall_quality | 0.1784 | 0.8679 |
| controlled_vs_baseline_no_context | context_relevance | 0.2410 | 7.4852 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1495 | 0.8218 |
| controlled_vs_baseline_no_context | naturalness | 0.0086 | 0.0097 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3205 | 12.4412 |
| controlled_vs_baseline_no_context | context_overlap | 0.0554 | 1.1746 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1852 | 2.3293 |
| controlled_vs_baseline_no_context | persona_style | 0.0067 | 0.0113 |
| controlled_vs_baseline_no_context | distinct1 | -0.0399 | -0.0407 |
| controlled_vs_baseline_no_context | length_score | 0.0967 | 0.1796 |
| controlled_vs_baseline_no_context | sentence_score | 0.0525 | 0.0587 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0460 | 1.2263 |
| controlled_vs_baseline_no_context | overall_quality | 0.1546 | 0.6744 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.2470 | 9.4484 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1579 | 0.9097 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | -0.0016 | -0.0018 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.3246 | 15.0351 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0659 | 1.7922 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1924 | 2.6579 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | 0.0200 | 0.0345 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0440 | -0.0447 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | 0.0450 | 0.0763 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.0700 | 0.0798 |
| controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0376 | 0.8196 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1565 | 0.6885 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0040 | 0.0146 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0205 | -0.0619 |
| controlled_alt_vs_controlled_default | naturalness | -0.0171 | -0.0191 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0000 | 0.0000 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0133 | 0.1294 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0207 | -0.0782 |
| controlled_alt_vs_controlled_default | persona_style | -0.0197 | -0.0330 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0016 | -0.0017 |
| controlled_alt_vs_controlled_default | length_score | -0.0650 | -0.1024 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0350 | -0.0369 |
| controlled_alt_vs_controlled_default | bertscore_f1 | 0.0073 | 0.0874 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0076 | -0.0198 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1772 | 1.7743 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1199 | 0.6275 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0556 | 0.0674 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2303 | 1.9869 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0534 | 0.8544 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1536 | 1.6974 |
| controlled_alt_vs_proposed_raw | persona_style | -0.0148 | -0.0250 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0090 | -0.0095 |
| controlled_alt_vs_proposed_raw | length_score | 0.2433 | 0.7449 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1050 | 0.1300 |
| controlled_alt_vs_proposed_raw | bertscore_f1 | 0.0262 | 0.4069 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1248 | 0.4962 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2530 | 10.4635 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1357 | 0.7745 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0879 | 0.1109 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3288 | 18.8696 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0760 | 1.9036 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1536 | 1.6974 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0643 | 0.1250 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0029 | 0.0031 |
| controlled_alt_vs_candidate_no_context | length_score | 0.3550 | 1.6512 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1575 | 0.2086 |
| controlled_alt_vs_candidate_no_context | bertscore_f1 | 0.0705 | 3.4700 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1708 | 0.8309 |
| controlled_alt_vs_baseline_no_context | context_relevance | 0.2449 | 7.6090 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 0.1290 | 0.7090 |
| controlled_alt_vs_baseline_no_context | naturalness | -0.0085 | -0.0096 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 0.3205 | 12.4412 |
| controlled_alt_vs_baseline_no_context | context_overlap | 0.0687 | 1.4560 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 0.1645 | 2.0689 |
| controlled_alt_vs_baseline_no_context | persona_style | -0.0131 | -0.0221 |
| controlled_alt_vs_baseline_no_context | distinct1 | -0.0414 | -0.0423 |
| controlled_alt_vs_baseline_no_context | length_score | 0.0317 | 0.0588 |
| controlled_alt_vs_baseline_no_context | sentence_score | 0.0175 | 0.0196 |
| controlled_alt_vs_baseline_no_context | bertscore_f1 | 0.0533 | 1.4209 |
| controlled_alt_vs_baseline_no_context | overall_quality | 0.1470 | 0.6413 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_relevance | 0.2510 | 9.6008 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1374 | 0.7914 |
| controlled_alt_vs_baseline_no_context_phi3_latest | naturalness | -0.0187 | -0.0208 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.3246 | 15.0351 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_overlap | 0.0792 | 2.1535 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1717 | 2.3717 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_style | 0.0002 | 0.0004 |
| controlled_alt_vs_baseline_no_context_phi3_latest | distinct1 | -0.0455 | -0.0463 |
| controlled_alt_vs_baseline_no_context_phi3_latest | length_score | -0.0200 | -0.0339 |
| controlled_alt_vs_baseline_no_context_phi3_latest | sentence_score | 0.0350 | 0.0399 |
| controlled_alt_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0449 | 0.9786 |
| controlled_alt_vs_baseline_no_context_phi3_latest | overall_quality | 0.1490 | 0.6552 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2410 | 7.4852 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1495 | 0.8218 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 0.0086 | 0.0097 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3205 | 12.4412 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0554 | 1.1746 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1852 | 2.3293 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0067 | 0.0113 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0399 | -0.0407 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.0967 | 0.1796 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0525 | 0.0587 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0460 | 1.2263 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1546 | 0.6744 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.2470 | 9.4484 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1579 | 0.9097 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | -0.0016 | -0.0018 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.3246 | 15.0351 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0659 | 1.7922 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1924 | 2.6579 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | 0.0200 | 0.0345 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0440 | -0.0447 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | 0.0450 | 0.0763 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.0700 | 0.0798 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0376 | 0.8196 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1565 | 0.6885 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0757 | (0.0242, 0.1356) | 0.0010 | 0.0757 | (-0.0033, 0.1419) | 0.0390 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0158 | (-0.0299, 0.0626) | 0.2497 | 0.0158 | (-0.0205, 0.0450) | 0.2460 |
| proposed_vs_candidate_no_context | naturalness | 0.0323 | (-0.0044, 0.0654) | 0.0387 | 0.0323 | (-0.0113, 0.0625) | 0.0697 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0985 | (0.0299, 0.1761) | 0.0010 | 0.0985 | (-0.0065, 0.1861) | 0.0630 |
| proposed_vs_candidate_no_context | context_overlap | 0.0226 | (0.0075, 0.0397) | 0.0003 | 0.0226 | (-0.0004, 0.0423) | 0.0280 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0000 | (-0.0500, 0.0500) | 0.5713 | 0.0000 | (-0.0278, 0.0278) | 0.6400 |
| proposed_vs_candidate_no_context | persona_style | 0.0791 | (-0.0202, 0.2087) | 0.0887 | 0.0791 | (0.0000, 0.1979) | 0.3473 |
| proposed_vs_candidate_no_context | distinct1 | 0.0119 | (-0.0094, 0.0330) | 0.1440 | 0.0119 | (-0.0086, 0.0331) | 0.1273 |
| proposed_vs_candidate_no_context | length_score | 0.1117 | (-0.0217, 0.2350) | 0.0537 | 0.1117 | (-0.0500, 0.2180) | 0.0690 |
| proposed_vs_candidate_no_context | sentence_score | 0.0525 | (-0.0175, 0.1225) | 0.1250 | 0.0525 | (-0.0467, 0.1458) | 0.2300 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0442 | (0.0172, 0.0740) | 0.0003 | 0.0442 | (0.0044, 0.0774) | 0.0107 |
| proposed_vs_candidate_no_context | overall_quality | 0.0460 | (0.0136, 0.0809) | 0.0017 | 0.0460 | (-0.0043, 0.0864) | 0.0433 |
| proposed_vs_baseline_no_context | context_relevance | 0.0677 | (0.0172, 0.1254) | 0.0037 | 0.0677 | (-0.0063, 0.1231) | 0.0657 |
| proposed_vs_baseline_no_context | persona_consistency | 0.0091 | (-0.0360, 0.0603) | 0.3700 | 0.0091 | (-0.0497, 0.0820) | 0.4027 |
| proposed_vs_baseline_no_context | naturalness | -0.0641 | (-0.1000, -0.0291) | 1.0000 | -0.0641 | (-0.1037, -0.0274) | 0.9997 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0902 | (0.0189, 0.1674) | 0.0050 | 0.0902 | (-0.0130, 0.1680) | 0.0930 |
| proposed_vs_baseline_no_context | context_overlap | 0.0153 | (0.0007, 0.0309) | 0.0187 | 0.0153 | (0.0054, 0.0279) | 0.0007 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 0.0110 | (-0.0429, 0.0676) | 0.3720 | 0.0110 | (-0.0571, 0.0812) | 0.3903 |
| proposed_vs_baseline_no_context | persona_style | 0.0018 | (-0.0372, 0.0476) | 0.5077 | 0.0018 | (-0.0570, 0.0778) | 0.5167 |
| proposed_vs_baseline_no_context | distinct1 | -0.0324 | (-0.0531, -0.0111) | 0.9980 | -0.0324 | (-0.0610, -0.0066) | 0.9947 |
| proposed_vs_baseline_no_context | length_score | -0.2117 | (-0.3467, -0.0817) | 0.9987 | -0.2117 | (-0.3019, -0.1037) | 0.9990 |
| proposed_vs_baseline_no_context | sentence_score | -0.0875 | (-0.1925, 0.0175) | 0.9533 | -0.0875 | (-0.2333, 0.0700) | 0.8793 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0270 | (-0.0026, 0.0603) | 0.0387 | 0.0270 | (-0.0082, 0.0550) | 0.0543 |
| proposed_vs_baseline_no_context | overall_quality | 0.0222 | (-0.0108, 0.0548) | 0.0947 | 0.0222 | (-0.0303, 0.0582) | 0.1830 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 0.0737 | (0.0199, 0.1355) | 0.0033 | 0.0737 | (-0.0051, 0.1335) | 0.0637 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | 0.0175 | (-0.0374, 0.0862) | 0.3130 | 0.0175 | (-0.0453, 0.0980) | 0.3077 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | -0.0743 | (-0.1202, -0.0246) | 0.9970 | -0.0743 | (-0.1331, -0.0159) | 0.9923 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.0943 | (0.0197, 0.1731) | 0.0047 | 0.0943 | (-0.0130, 0.1742) | 0.1010 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 0.0258 | (0.0104, 0.0414) | 0.0003 | 0.0258 | (0.0077, 0.0408) | 0.0003 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.0181 | (-0.0471, 0.0967) | 0.3520 | 0.0181 | (-0.0571, 0.1123) | 0.3347 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | 0.0151 | (-0.0258, 0.0642) | 0.2630 | 0.0151 | (-0.0417, 0.0608) | 0.3163 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | -0.0365 | (-0.0528, -0.0202) | 1.0000 | -0.0365 | (-0.0576, -0.0152) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | -0.2633 | (-0.4517, -0.0716) | 0.9973 | -0.2633 | (-0.4667, -0.0510) | 0.9890 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | -0.0700 | (-0.1750, 0.0350) | 0.9403 | -0.0700 | (-0.1750, 0.0808) | 0.8907 |
| proposed_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0186 | (-0.0075, 0.0466) | 0.0920 | 0.0186 | (-0.0174, 0.0424) | 0.1373 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | 0.0242 | (-0.0122, 0.0638) | 0.1007 | 0.0242 | (-0.0307, 0.0649) | 0.1963 |
| controlled_vs_proposed_raw | context_relevance | 0.1733 | (0.1082, 0.2343) | 0.0000 | 0.1733 | (0.1030, 0.2477) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.1404 | (0.0725, 0.2087) | 0.0000 | 0.1404 | (0.0524, 0.2197) | 0.0020 |
| controlled_vs_proposed_raw | naturalness | 0.0727 | (0.0309, 0.1118) | 0.0003 | 0.0727 | (0.0260, 0.1261) | 0.0000 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2303 | (0.1458, 0.3148) | 0.0000 | 0.2303 | (0.1364, 0.3287) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0401 | (0.0181, 0.0626) | 0.0000 | 0.0401 | (0.0154, 0.0629) | 0.0020 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1743 | (0.0916, 0.2533) | 0.0000 | 0.1743 | (0.0774, 0.2646) | 0.0007 |
| controlled_vs_proposed_raw | persona_style | 0.0049 | (-0.0547, 0.0666) | 0.4340 | 0.0049 | (-0.0544, 0.1026) | 0.4807 |
| controlled_vs_proposed_raw | distinct1 | -0.0074 | (-0.0257, 0.0111) | 0.7717 | -0.0074 | (-0.0356, 0.0188) | 0.6690 |
| controlled_vs_proposed_raw | length_score | 0.3083 | (0.1417, 0.4717) | 0.0000 | 0.3083 | (0.1259, 0.5070) | 0.0000 |
| controlled_vs_proposed_raw | sentence_score | 0.1400 | (0.0525, 0.2275) | 0.0017 | 0.1400 | (0.0412, 0.2500) | 0.0027 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0190 | (-0.0169, 0.0541) | 0.1517 | 0.0190 | (-0.0072, 0.0417) | 0.0637 |
| controlled_vs_proposed_raw | overall_quality | 0.1324 | (0.0943, 0.1750) | 0.0000 | 0.1324 | (0.0945, 0.1732) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2490 | (0.2014, 0.2992) | 0.0000 | 0.2490 | (0.1805, 0.3190) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1562 | (0.0789, 0.2353) | 0.0000 | 0.1562 | (0.0585, 0.2381) | 0.0003 |
| controlled_vs_candidate_no_context | naturalness | 0.1050 | (0.0553, 0.1479) | 0.0000 | 0.1050 | (0.0433, 0.1461) | 0.0030 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3288 | (0.2632, 0.3966) | 0.0000 | 0.3288 | (0.2403, 0.4150) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0627 | (0.0442, 0.0829) | 0.0000 | 0.0627 | (0.0347, 0.0845) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1743 | (0.0957, 0.2531) | 0.0000 | 0.1743 | (0.0872, 0.2526) | 0.0003 |
| controlled_vs_candidate_no_context | persona_style | 0.0840 | (-0.0458, 0.2381) | 0.1110 | 0.0840 | (-0.0312, 0.1913) | 0.1177 |
| controlled_vs_candidate_no_context | distinct1 | 0.0045 | (-0.0159, 0.0262) | 0.3453 | 0.0045 | (-0.0140, 0.0254) | 0.3260 |
| controlled_vs_candidate_no_context | length_score | 0.4200 | (0.2150, 0.5917) | 0.0000 | 0.4200 | (0.1933, 0.5814) | 0.0010 |
| controlled_vs_candidate_no_context | sentence_score | 0.1925 | (0.0875, 0.2975) | 0.0007 | 0.1925 | (0.0467, 0.2917) | 0.0103 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0632 | (0.0241, 0.1093) | 0.0007 | 0.0632 | (0.0211, 0.0973) | 0.0013 |
| controlled_vs_candidate_no_context | overall_quality | 0.1784 | (0.1392, 0.2205) | 0.0000 | 0.1784 | (0.1271, 0.2104) | 0.0000 |
| controlled_vs_baseline_no_context | context_relevance | 0.2410 | (0.1936, 0.2916) | 0.0000 | 0.2410 | (0.1744, 0.3202) | 0.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1495 | (0.0869, 0.2125) | 0.0000 | 0.1495 | (0.0799, 0.2099) | 0.0000 |
| controlled_vs_baseline_no_context | naturalness | 0.0086 | (-0.0287, 0.0473) | 0.3150 | 0.0086 | (-0.0235, 0.0390) | 0.3180 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3205 | (0.2553, 0.3917) | 0.0000 | 0.3205 | (0.2308, 0.4205) | 0.0000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0554 | (0.0386, 0.0748) | 0.0000 | 0.0554 | (0.0380, 0.0769) | 0.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1852 | (0.1045, 0.2607) | 0.0000 | 0.1852 | (0.0946, 0.2575) | 0.0000 |
| controlled_vs_baseline_no_context | persona_style | 0.0067 | (-0.0480, 0.0542) | 0.3820 | 0.0067 | (-0.0458, 0.0833) | 0.3983 |
| controlled_vs_baseline_no_context | distinct1 | -0.0399 | (-0.0578, -0.0214) | 1.0000 | -0.0399 | (-0.0581, -0.0179) | 0.9983 |
| controlled_vs_baseline_no_context | length_score | 0.0967 | (-0.0750, 0.2634) | 0.1367 | 0.0967 | (-0.0786, 0.2508) | 0.1437 |
| controlled_vs_baseline_no_context | sentence_score | 0.0525 | (-0.0525, 0.1575) | 0.2067 | 0.0525 | (-0.0921, 0.1605) | 0.2547 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0460 | (0.0139, 0.0767) | 0.0027 | 0.0460 | (0.0232, 0.0620) | 0.0003 |
| controlled_vs_baseline_no_context | overall_quality | 0.1546 | (0.1248, 0.1854) | 0.0000 | 0.1546 | (0.1109, 0.1919) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.2470 | (0.1991, 0.2996) | 0.0000 | 0.2470 | (0.1748, 0.3242) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1579 | (0.0920, 0.2199) | 0.0000 | 0.1579 | (0.0827, 0.2194) | 0.0003 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | -0.0016 | (-0.0506, 0.0434) | 0.5070 | -0.0016 | (-0.0285, 0.0314) | 0.5403 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.3246 | (0.2621, 0.3924) | 0.0000 | 0.3246 | (0.2308, 0.4205) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0659 | (0.0477, 0.0849) | 0.0000 | 0.0659 | (0.0425, 0.0899) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1924 | (0.1198, 0.2629) | 0.0000 | 0.1924 | (0.1027, 0.2646) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | 0.0200 | (-0.0225, 0.0633) | 0.1760 | 0.0200 | (0.0000, 0.0625) | 0.0953 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0440 | (-0.0571, -0.0301) | 1.0000 | -0.0440 | (-0.0567, -0.0257) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | 0.0450 | (-0.1750, 0.2533) | 0.3367 | 0.0450 | (-0.1222, 0.2222) | 0.2927 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.0700 | (-0.0175, 0.1575) | 0.0917 | 0.0700 | (0.0000, 0.1531) | 0.0453 |
| controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0376 | (0.0014, 0.0753) | 0.0200 | 0.0376 | (-0.0010, 0.0611) | 0.0270 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1565 | (0.1242, 0.1896) | 0.0000 | 0.1565 | (0.1095, 0.1953) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0040 | (-0.0465, 0.0511) | 0.4373 | 0.0040 | (-0.0503, 0.0615) | 0.4147 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0205 | (-0.0823, 0.0491) | 0.7273 | -0.0205 | (-0.1077, 0.0293) | 0.7677 |
| controlled_alt_vs_controlled_default | naturalness | -0.0171 | (-0.0591, 0.0219) | 0.7977 | -0.0171 | (-0.0527, 0.0282) | 0.7757 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0000 | (-0.0652, 0.0591) | 0.4853 | 0.0000 | (-0.0699, 0.0667) | 0.4653 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0133 | (-0.0146, 0.0399) | 0.1560 | 0.0133 | (-0.0112, 0.0462) | 0.1527 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0207 | (-0.0969, 0.0595) | 0.6957 | -0.0207 | (-0.1204, 0.0345) | 0.7387 |
| controlled_alt_vs_controlled_default | persona_style | -0.0197 | (-0.0667, 0.0164) | 0.8143 | -0.0197 | (-0.0833, 0.0171) | 0.8343 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0016 | (-0.0245, 0.0232) | 0.5700 | -0.0016 | (-0.0296, 0.0253) | 0.5530 |
| controlled_alt_vs_controlled_default | length_score | -0.0650 | (-0.2500, 0.1150) | 0.7523 | -0.0650 | (-0.2667, 0.1750) | 0.6713 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0350 | (-0.1225, 0.0525) | 0.8537 | -0.0350 | (-0.1050, 0.0292) | 0.9133 |
| controlled_alt_vs_controlled_default | bertscore_f1 | 0.0073 | (-0.0272, 0.0409) | 0.3333 | 0.0073 | (-0.0099, 0.0343) | 0.2533 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0076 | (-0.0301, 0.0154) | 0.7677 | -0.0076 | (-0.0214, 0.0029) | 0.9143 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1772 | (0.1261, 0.2255) | 0.0000 | 0.1772 | (0.1103, 0.2476) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1199 | (0.0314, 0.1986) | 0.0030 | 0.1199 | (-0.0242, 0.1984) | 0.0397 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0556 | (0.0140, 0.0971) | 0.0037 | 0.0556 | (0.0230, 0.0877) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2303 | (0.1598, 0.2924) | 0.0000 | 0.2303 | (0.1462, 0.3212) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0534 | (0.0338, 0.0724) | 0.0000 | 0.0534 | (0.0315, 0.0782) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1536 | (0.0526, 0.2512) | 0.0017 | 0.1536 | (-0.0143, 0.2495) | 0.0333 |
| controlled_alt_vs_proposed_raw | persona_style | -0.0148 | (-0.0652, 0.0388) | 0.7240 | -0.0148 | (-0.0807, 0.0483) | 0.6817 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0090 | (-0.0358, 0.0164) | 0.7483 | -0.0090 | (-0.0335, 0.0255) | 0.7073 |
| controlled_alt_vs_proposed_raw | length_score | 0.2433 | (0.1000, 0.3900) | 0.0003 | 0.2433 | (0.1560, 0.3738) | 0.0000 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1050 | (0.0175, 0.1925) | 0.0240 | 0.1050 | (-0.0219, 0.2130) | 0.0673 |
| controlled_alt_vs_proposed_raw | bertscore_f1 | 0.0262 | (0.0023, 0.0495) | 0.0160 | 0.0262 | (0.0065, 0.0525) | 0.0043 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1248 | (0.0932, 0.1564) | 0.0000 | 0.1248 | (0.0837, 0.1610) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2530 | (0.2227, 0.2856) | 0.0000 | 0.2530 | (0.2281, 0.2694) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1357 | (0.0461, 0.2232) | 0.0017 | 0.1357 | (-0.0194, 0.2287) | 0.0380 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0879 | (0.0486, 0.1243) | 0.0000 | 0.0879 | (0.0588, 0.1135) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3288 | (0.2894, 0.3712) | 0.0000 | 0.3288 | (0.2987, 0.3529) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0760 | (0.0606, 0.0928) | 0.0000 | 0.0760 | (0.0653, 0.0860) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1536 | (0.0562, 0.2407) | 0.0003 | 0.1536 | (-0.0003, 0.2407) | 0.0257 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0643 | (-0.0631, 0.2071) | 0.1770 | 0.0643 | (-0.0750, 0.1955) | 0.2383 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0029 | (-0.0210, 0.0272) | 0.4067 | 0.0029 | (-0.0291, 0.0336) | 0.4377 |
| controlled_alt_vs_candidate_no_context | length_score | 0.3550 | (0.1983, 0.5133) | 0.0000 | 0.3550 | (0.2487, 0.4815) | 0.0000 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1575 | (0.0700, 0.2450) | 0.0020 | 0.1575 | (0.0292, 0.2283) | 0.0180 |
| controlled_alt_vs_candidate_no_context | bertscore_f1 | 0.0705 | (0.0433, 0.1019) | 0.0000 | 0.0705 | (0.0476, 0.0917) | 0.0000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1708 | (0.1335, 0.2074) | 0.0000 | 0.1708 | (0.1145, 0.2034) | 0.0000 |
| controlled_alt_vs_baseline_no_context | context_relevance | 0.2449 | (0.2176, 0.2731) | 0.0000 | 0.2449 | (0.2180, 0.2704) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 0.1290 | (0.0665, 0.1967) | 0.0003 | 0.1290 | (0.0204, 0.2048) | 0.0103 |
| controlled_alt_vs_baseline_no_context | naturalness | -0.0085 | (-0.0530, 0.0339) | 0.6377 | -0.0085 | (-0.0253, 0.0181) | 0.7750 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 0.3205 | (0.2863, 0.3591) | 0.0000 | 0.3205 | (0.2825, 0.3541) | 0.0000 |
| controlled_alt_vs_baseline_no_context | context_overlap | 0.0687 | (0.0537, 0.0842) | 0.0000 | 0.0687 | (0.0502, 0.0896) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 0.1645 | (0.0819, 0.2467) | 0.0000 | 0.1645 | (0.0299, 0.2594) | 0.0093 |
| controlled_alt_vs_baseline_no_context | persona_style | -0.0131 | (-0.0512, 0.0167) | 0.8097 | -0.0131 | (-0.0327, 0.0000) | 1.0000 |
| controlled_alt_vs_baseline_no_context | distinct1 | -0.0414 | (-0.0630, -0.0190) | 0.9997 | -0.0414 | (-0.0640, -0.0201) | 0.9997 |
| controlled_alt_vs_baseline_no_context | length_score | 0.0317 | (-0.1683, 0.2200) | 0.3623 | 0.0317 | (-0.0742, 0.1905) | 0.3043 |
| controlled_alt_vs_baseline_no_context | sentence_score | 0.0175 | (-0.1050, 0.1400) | 0.4320 | 0.0175 | (-0.1065, 0.1086) | 0.4433 |
| controlled_alt_vs_baseline_no_context | bertscore_f1 | 0.0533 | (0.0235, 0.0819) | 0.0000 | 0.0533 | (0.0281, 0.0756) | 0.0000 |
| controlled_alt_vs_baseline_no_context | overall_quality | 0.1470 | (0.1239, 0.1706) | 0.0000 | 0.1470 | (0.1066, 0.1767) | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_relevance | 0.2510 | (0.2209, 0.2812) | 0.0000 | 0.2510 | (0.2252, 0.2734) | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1374 | (0.0657, 0.2069) | 0.0007 | 0.1374 | (0.0283, 0.2131) | 0.0040 |
| controlled_alt_vs_baseline_no_context_phi3_latest | naturalness | -0.0187 | (-0.0667, 0.0231) | 0.7953 | -0.0187 | (-0.0574, 0.0356) | 0.7580 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.3246 | (0.2852, 0.3674) | 0.0000 | 0.3246 | (0.2865, 0.3554) | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_overlap | 0.0792 | (0.0640, 0.0951) | 0.0000 | 0.0792 | (0.0685, 0.0902) | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1717 | (0.0848, 0.2569) | 0.0000 | 0.1717 | (0.0431, 0.2617) | 0.0037 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_style | 0.0002 | (-0.0375, 0.0383) | 0.4967 | 0.0002 | (-0.0417, 0.0293) | 0.4437 |
| controlled_alt_vs_baseline_no_context_phi3_latest | distinct1 | -0.0455 | (-0.0667, -0.0243) | 1.0000 | -0.0455 | (-0.0670, -0.0265) | 1.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | length_score | -0.0200 | (-0.2417, 0.2000) | 0.5657 | -0.0200 | (-0.2254, 0.2382) | 0.5490 |
| controlled_alt_vs_baseline_no_context_phi3_latest | sentence_score | 0.0350 | (0.0000, 0.0875) | 0.1143 | 0.0350 | (0.0000, 0.0955) | 0.0970 |
| controlled_alt_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0449 | (0.0178, 0.0734) | 0.0013 | 0.0449 | (0.0178, 0.0587) | 0.0017 |
| controlled_alt_vs_baseline_no_context_phi3_latest | overall_quality | 0.1490 | (0.1203, 0.1765) | 0.0000 | 0.1490 | (0.1076, 0.1822) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2410 | (0.1943, 0.2935) | 0.0000 | 0.2410 | (0.1711, 0.3199) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1495 | (0.0848, 0.2104) | 0.0000 | 0.1495 | (0.0798, 0.2118) | 0.0003 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 0.0086 | (-0.0325, 0.0467) | 0.3310 | 0.0086 | (-0.0255, 0.0402) | 0.2977 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3205 | (0.2542, 0.3913) | 0.0000 | 0.3205 | (0.2273, 0.4192) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0554 | (0.0384, 0.0744) | 0.0000 | 0.0554 | (0.0376, 0.0785) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1852 | (0.1071, 0.2617) | 0.0000 | 0.1852 | (0.0905, 0.2563) | 0.0010 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0067 | (-0.0474, 0.0548) | 0.3820 | 0.0067 | (-0.0459, 0.0833) | 0.4210 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0399 | (-0.0585, -0.0204) | 1.0000 | -0.0399 | (-0.0588, -0.0184) | 0.9987 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.0967 | (-0.0784, 0.2600) | 0.1407 | 0.0967 | (-0.0747, 0.2600) | 0.1320 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0525 | (-0.0525, 0.1575) | 0.1880 | 0.0525 | (-0.0933, 0.1591) | 0.2650 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0460 | (0.0144, 0.0763) | 0.0033 | 0.0460 | (0.0228, 0.0612) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1546 | (0.1234, 0.1846) | 0.0000 | 0.1546 | (0.1141, 0.1923) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.2470 | (0.1964, 0.3008) | 0.0000 | 0.2470 | (0.1745, 0.3208) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1579 | (0.0949, 0.2179) | 0.0000 | 0.1579 | (0.0859, 0.2189) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | -0.0016 | (-0.0502, 0.0424) | 0.5177 | -0.0016 | (-0.0280, 0.0279) | 0.5530 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.3246 | (0.2591, 0.3909) | 0.0000 | 0.3246 | (0.2300, 0.4218) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0659 | (0.0474, 0.0855) | 0.0000 | 0.0659 | (0.0417, 0.0899) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1924 | (0.1193, 0.2660) | 0.0000 | 0.1924 | (0.1037, 0.2643) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | 0.0200 | (-0.0230, 0.0622) | 0.1910 | 0.0200 | (0.0000, 0.0627) | 0.1000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0440 | (-0.0566, -0.0302) | 1.0000 | -0.0440 | (-0.0572, -0.0255) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | 0.0450 | (-0.1683, 0.2450) | 0.3403 | 0.0450 | (-0.1284, 0.2119) | 0.3197 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.0700 | (-0.0175, 0.1575) | 0.0840 | 0.0700 | (0.0000, 0.1556) | 0.0473 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0376 | (-0.0003, 0.0784) | 0.0257 | 0.0376 | (0.0030, 0.0611) | 0.0197 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1565 | (0.1250, 0.1893) | 0.0000 | 0.1565 | (0.1119, 0.1956) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 9 | 4 | 7 | 0.6250 | 0.6923 |
| proposed_vs_candidate_no_context | persona_consistency | 5 | 4 | 11 | 0.5250 | 0.5556 |
| proposed_vs_candidate_no_context | naturalness | 10 | 3 | 7 | 0.6750 | 0.7692 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 7 | 1 | 12 | 0.6500 | 0.8750 |
| proposed_vs_candidate_no_context | context_overlap | 8 | 5 | 7 | 0.5750 | 0.6154 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 2 | 2 | 16 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | persona_style | 3 | 2 | 15 | 0.5250 | 0.6000 |
| proposed_vs_candidate_no_context | distinct1 | 7 | 6 | 7 | 0.5250 | 0.5385 |
| proposed_vs_candidate_no_context | length_score | 10 | 3 | 7 | 0.6750 | 0.7692 |
| proposed_vs_candidate_no_context | sentence_score | 4 | 1 | 15 | 0.5750 | 0.8000 |
| proposed_vs_candidate_no_context | bertscore_f1 | 14 | 1 | 5 | 0.8250 | 0.9333 |
| proposed_vs_candidate_no_context | overall_quality | 12 | 3 | 5 | 0.7250 | 0.8000 |
| proposed_vs_baseline_no_context | context_relevance | 13 | 7 | 0 | 0.6500 | 0.6500 |
| proposed_vs_baseline_no_context | persona_consistency | 3 | 7 | 10 | 0.4000 | 0.3000 |
| proposed_vs_baseline_no_context | naturalness | 5 | 15 | 0 | 0.2500 | 0.2500 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 7 | 2 | 11 | 0.6250 | 0.7778 |
| proposed_vs_baseline_no_context | context_overlap | 14 | 6 | 0 | 0.7000 | 0.7000 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 3 | 3 | 14 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context | persona_style | 2 | 4 | 14 | 0.4500 | 0.3333 |
| proposed_vs_baseline_no_context | distinct1 | 6 | 13 | 1 | 0.3250 | 0.3158 |
| proposed_vs_baseline_no_context | length_score | 4 | 16 | 0 | 0.2000 | 0.2000 |
| proposed_vs_baseline_no_context | sentence_score | 3 | 8 | 9 | 0.3750 | 0.2727 |
| proposed_vs_baseline_no_context | bertscore_f1 | 12 | 8 | 0 | 0.6000 | 0.6000 |
| proposed_vs_baseline_no_context | overall_quality | 9 | 11 | 0 | 0.4500 | 0.4500 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 13 | 7 | 0 | 0.6500 | 0.6500 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | 4 | 5 | 11 | 0.4750 | 0.4444 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | 6 | 14 | 0 | 0.3000 | 0.3000 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 7 | 2 | 11 | 0.6250 | 0.7778 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 14 | 6 | 0 | 0.7000 | 0.7000 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 2 | 3 | 15 | 0.4750 | 0.4000 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | 3 | 2 | 15 | 0.5250 | 0.6000 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | 2 | 15 | 3 | 0.1750 | 0.1176 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | 4 | 16 | 0 | 0.2000 | 0.2000 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | 3 | 7 | 10 | 0.4000 | 0.3000 |
| proposed_vs_baseline_no_context_phi3_latest | bertscore_f1 | 12 | 8 | 0 | 0.6000 | 0.6000 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | 8 | 12 | 0 | 0.4000 | 0.4000 |
| controlled_vs_proposed_raw | context_relevance | 18 | 2 | 0 | 0.9000 | 0.9000 |
| controlled_vs_proposed_raw | persona_consistency | 14 | 3 | 3 | 0.7750 | 0.8235 |
| controlled_vs_proposed_raw | naturalness | 17 | 3 | 0 | 0.8500 | 0.8500 |
| controlled_vs_proposed_raw | context_keyword_coverage | 18 | 2 | 0 | 0.9000 | 0.9000 |
| controlled_vs_proposed_raw | context_overlap | 16 | 4 | 0 | 0.8000 | 0.8000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 14 | 2 | 4 | 0.8000 | 0.8750 |
| controlled_vs_proposed_raw | persona_style | 3 | 3 | 14 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | distinct1 | 8 | 11 | 1 | 0.4250 | 0.4211 |
| controlled_vs_proposed_raw | length_score | 16 | 4 | 0 | 0.8000 | 0.8000 |
| controlled_vs_proposed_raw | sentence_score | 9 | 1 | 10 | 0.7000 | 0.9000 |
| controlled_vs_proposed_raw | bertscore_f1 | 14 | 6 | 0 | 0.7000 | 0.7000 |
| controlled_vs_proposed_raw | overall_quality | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_relevance | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 14 | 3 | 3 | 0.7750 | 0.8235 |
| controlled_vs_candidate_no_context | naturalness | 17 | 3 | 0 | 0.8500 | 0.8500 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 14 | 2 | 4 | 0.8000 | 0.8750 |
| controlled_vs_candidate_no_context | persona_style | 5 | 3 | 12 | 0.5500 | 0.6250 |
| controlled_vs_candidate_no_context | distinct1 | 9 | 10 | 1 | 0.4750 | 0.4737 |
| controlled_vs_candidate_no_context | length_score | 16 | 4 | 0 | 0.8000 | 0.8000 |
| controlled_vs_candidate_no_context | sentence_score | 13 | 2 | 5 | 0.7750 | 0.8667 |
| controlled_vs_candidate_no_context | bertscore_f1 | 15 | 5 | 0 | 0.7500 | 0.7500 |
| controlled_vs_candidate_no_context | overall_quality | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | context_relevance | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 16 | 2 | 2 | 0.8500 | 0.8889 |
| controlled_vs_baseline_no_context | naturalness | 12 | 8 | 0 | 0.6000 | 0.6000 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | context_overlap | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 16 | 2 | 2 | 0.8500 | 0.8889 |
| controlled_vs_baseline_no_context | persona_style | 3 | 2 | 15 | 0.5250 | 0.6000 |
| controlled_vs_baseline_no_context | distinct1 | 4 | 15 | 1 | 0.2250 | 0.2105 |
| controlled_vs_baseline_no_context | length_score | 13 | 7 | 0 | 0.6500 | 0.6500 |
| controlled_vs_baseline_no_context | sentence_score | 6 | 3 | 11 | 0.5750 | 0.6667 |
| controlled_vs_baseline_no_context | bertscore_f1 | 15 | 5 | 0 | 0.7500 | 0.7500 |
| controlled_vs_baseline_no_context | overall_quality | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 15 | 2 | 3 | 0.8250 | 0.8824 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | 12 | 8 | 0 | 0.6000 | 0.6000 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 15 | 1 | 4 | 0.8500 | 0.9375 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | 5 | 2 | 13 | 0.5750 | 0.7143 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | 1 | 18 | 1 | 0.0750 | 0.0526 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | 14 | 5 | 1 | 0.7250 | 0.7368 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 6 | 2 | 12 | 0.6000 | 0.7500 |
| controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 12 | 8 | 0 | 0.6000 | 0.6000 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 11 | 7 | 2 | 0.6000 | 0.6111 |
| controlled_alt_vs_controlled_default | persona_consistency | 6 | 9 | 5 | 0.4250 | 0.4000 |
| controlled_alt_vs_controlled_default | naturalness | 8 | 10 | 2 | 0.4500 | 0.4444 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 8 | 7 | 5 | 0.5250 | 0.5333 |
| controlled_alt_vs_controlled_default | context_overlap | 12 | 5 | 3 | 0.6750 | 0.7059 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 5 | 8 | 7 | 0.4250 | 0.3846 |
| controlled_alt_vs_controlled_default | persona_style | 2 | 2 | 16 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | distinct1 | 8 | 10 | 2 | 0.4500 | 0.4444 |
| controlled_alt_vs_controlled_default | length_score | 8 | 8 | 4 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | sentence_score | 2 | 4 | 14 | 0.4500 | 0.3333 |
| controlled_alt_vs_controlled_default | bertscore_f1 | 8 | 10 | 2 | 0.4500 | 0.4444 |
| controlled_alt_vs_controlled_default | overall_quality | 6 | 12 | 2 | 0.3500 | 0.3333 |
| controlled_alt_vs_proposed_raw | context_relevance | 18 | 2 | 0 | 0.9000 | 0.9000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 15 | 4 | 1 | 0.7750 | 0.7895 |
| controlled_alt_vs_proposed_raw | naturalness | 15 | 5 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 17 | 2 | 1 | 0.8750 | 0.8947 |
| controlled_alt_vs_proposed_raw | context_overlap | 18 | 2 | 0 | 0.9000 | 0.9000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 15 | 4 | 1 | 0.7750 | 0.7895 |
| controlled_alt_vs_proposed_raw | persona_style | 2 | 3 | 15 | 0.4750 | 0.4000 |
| controlled_alt_vs_proposed_raw | distinct1 | 9 | 10 | 1 | 0.4750 | 0.4737 |
| controlled_alt_vs_proposed_raw | length_score | 13 | 6 | 1 | 0.6750 | 0.6842 |
| controlled_alt_vs_proposed_raw | sentence_score | 8 | 2 | 10 | 0.6500 | 0.8000 |
| controlled_alt_vs_proposed_raw | bertscore_f1 | 14 | 6 | 0 | 0.7000 | 0.7000 |
| controlled_alt_vs_proposed_raw | overall_quality | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 15 | 3 | 2 | 0.8000 | 0.8333 |
| controlled_alt_vs_candidate_no_context | naturalness | 16 | 4 | 0 | 0.8000 | 0.8000 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 15 | 3 | 2 | 0.8000 | 0.8333 |
| controlled_alt_vs_candidate_no_context | persona_style | 3 | 2 | 15 | 0.5250 | 0.6000 |
| controlled_alt_vs_candidate_no_context | distinct1 | 9 | 10 | 1 | 0.4750 | 0.4737 |
| controlled_alt_vs_candidate_no_context | length_score | 15 | 5 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_candidate_no_context | sentence_score | 10 | 1 | 9 | 0.7250 | 0.9091 |
| controlled_alt_vs_candidate_no_context | bertscore_f1 | 18 | 2 | 0 | 0.9000 | 0.9000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context | context_relevance | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 13 | 2 | 5 | 0.7750 | 0.8667 |
| controlled_alt_vs_baseline_no_context | naturalness | 14 | 6 | 0 | 0.7000 | 0.7000 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context | context_overlap | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 13 | 2 | 5 | 0.7750 | 0.8667 |
| controlled_alt_vs_baseline_no_context | persona_style | 1 | 2 | 17 | 0.4750 | 0.3333 |
| controlled_alt_vs_baseline_no_context | distinct1 | 5 | 14 | 1 | 0.2750 | 0.2632 |
| controlled_alt_vs_baseline_no_context | length_score | 12 | 7 | 1 | 0.6250 | 0.6316 |
| controlled_alt_vs_baseline_no_context | sentence_score | 6 | 5 | 9 | 0.5250 | 0.5455 |
| controlled_alt_vs_baseline_no_context | bertscore_f1 | 15 | 5 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_baseline_no_context | overall_quality | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_relevance | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_consistency | 14 | 2 | 4 | 0.8000 | 0.8750 |
| controlled_alt_vs_baseline_no_context_phi3_latest | naturalness | 10 | 10 | 0 | 0.5000 | 0.5000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_overlap | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 14 | 2 | 4 | 0.8000 | 0.8750 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_style | 3 | 2 | 15 | 0.5250 | 0.6000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | distinct1 | 4 | 16 | 0 | 0.2000 | 0.2000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | length_score | 10 | 10 | 0 | 0.5000 | 0.5000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | sentence_score | 2 | 0 | 18 | 0.5500 | 1.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | bertscore_f1 | 14 | 6 | 0 | 0.7000 | 0.7000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | overall_quality | 19 | 1 | 0 | 0.9500 | 0.9500 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 20 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 16 | 2 | 2 | 0.8500 | 0.8889 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 12 | 8 | 0 | 0.6000 | 0.6000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 20 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 19 | 1 | 0 | 0.9500 | 0.9500 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 16 | 2 | 2 | 0.8500 | 0.8889 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 3 | 2 | 15 | 0.5250 | 0.6000 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | 4 | 15 | 1 | 0.2250 | 0.2105 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 13 | 7 | 0 | 0.6500 | 0.6500 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 6 | 3 | 11 | 0.5750 | 0.6667 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 15 | 5 | 0 | 0.7500 | 0.7500 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 20 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 20 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 15 | 2 | 3 | 0.8250 | 0.8824 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | 12 | 8 | 0 | 0.6000 | 0.6000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 20 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 19 | 1 | 0 | 0.9500 | 0.9500 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 15 | 1 | 4 | 0.8500 | 0.9375 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | 5 | 2 | 13 | 0.5750 | 0.7143 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | 1 | 18 | 1 | 0.0750 | 0.0526 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | 14 | 5 | 1 | 0.7250 | 0.7368 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | 6 | 2 | 12 | 0.6000 | 0.7500 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 12 | 8 | 0 | 0.6000 | 0.6000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 20 | 0 | 0 | 1.0000 | 1.0000 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.3000 | 0.4000 | 0.6000 |
| proposed_contextual_controlled_quality | 0.0000 | 0.0000 | 0.4000 | 0.3000 | 0.7000 |
| proposed_contextual | 0.0000 | 0.0000 | 0.4500 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.5500 | 0.0000 | 0.0000 |
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
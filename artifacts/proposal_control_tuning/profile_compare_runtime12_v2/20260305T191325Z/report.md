# Proposal Alignment Evaluation Report

- Run ID: `20260305T191325Z`
- Generated: `2026-03-05T19:17:54.190863+00:00`
- Scenarios: `artifacts\proposal_control_tuning\profile_compare_runtime12_v2\20260305T191325Z\scenarios.jsonl`
- Scenario count: `12`

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
| proposed_contextual_controlled | 0.2796 (0.2364, 0.3290) | 0.3659 (0.2963, 0.4404) | 0.8454 (0.8071, 0.8876) | 0.3876 (0.3614, 0.4102) | 0.0780 |
| proposed_contextual_controlled_runtime | 0.3269 (0.2740, 0.3848) | 0.3062 (0.2379, 0.3902) | 0.8381 (0.8014, 0.8777) | 0.3866 (0.3542, 0.4203) | 0.0854 |
| proposed_contextual | 0.1328 (0.0544, 0.2281) | 0.1834 (0.1107, 0.2703) | 0.8413 (0.7996, 0.8822) | 0.2666 (0.2138, 0.3215) | 0.0759 |
| candidate_no_context | 0.0219 (0.0090, 0.0436) | 0.1777 (0.1143, 0.2499) | 0.8174 (0.7753, 0.8659) | 0.2097 (0.1794, 0.2450) | 0.0252 |
| baseline_no_context | 0.0529 (0.0246, 0.0831) | 0.1829 (0.1236, 0.2510) | 0.8966 (0.8606, 0.9318) | 0.2429 (0.2213, 0.2686) | 0.0653 |
| baseline_no_context_phi3_latest | 0.0344 (0.0150, 0.0579) | 0.1914 (0.1189, 0.2758) | 0.8999 (0.8703, 0.9278) | 0.2375 (0.2094, 0.2695) | 0.0489 |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.1109 | 5.0697 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0057 | 0.0320 |
| proposed_vs_candidate_no_context | naturalness | 0.0239 | 0.0292 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.1471 | 9.7083 |
| proposed_vs_candidate_no_context | context_overlap | 0.0265 | 0.7045 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0020 | -0.0263 |
| proposed_vs_candidate_no_context | persona_style | 0.0364 | 0.0620 |
| proposed_vs_candidate_no_context | distinct1 | -0.0014 | -0.0015 |
| proposed_vs_candidate_no_context | length_score | 0.1222 | 0.4151 |
| proposed_vs_candidate_no_context | sentence_score | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0508 | 2.0148 |
| proposed_vs_candidate_no_context | overall_quality | 0.0569 | 0.2712 |
| proposed_vs_baseline_no_context | context_relevance | 0.0799 | 1.5102 |
| proposed_vs_baseline_no_context | persona_consistency | 0.0005 | 0.0029 |
| proposed_vs_baseline_no_context | naturalness | -0.0552 | -0.0616 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.1054 | 1.8556 |
| proposed_vs_baseline_no_context | context_overlap | 0.0203 | 0.4636 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0048 | -0.0609 |
| proposed_vs_baseline_no_context | persona_style | 0.0217 | 0.0360 |
| proposed_vs_baseline_no_context | distinct1 | -0.0315 | -0.0324 |
| proposed_vs_baseline_no_context | length_score | -0.1694 | -0.2891 |
| proposed_vs_baseline_no_context | sentence_score | -0.0875 | -0.0959 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0106 | 0.1623 |
| proposed_vs_baseline_no_context | overall_quality | 0.0236 | 0.0973 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 0.0984 | 2.8587 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | -0.0080 | -0.0419 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | -0.0585 | -0.0651 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.1332 | 4.5870 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 0.0171 | 0.3641 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | -0.0333 | -0.3123 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | 0.0932 | 0.1759 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | -0.0286 | -0.0296 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | -0.1917 | -0.3151 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | -0.0875 | -0.0959 |
| proposed_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0270 | 0.5524 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | 0.0290 | 0.1223 |
| controlled_vs_proposed_raw | context_relevance | 0.1468 | 1.1056 |
| controlled_vs_proposed_raw | persona_consistency | 0.1825 | 0.9949 |
| controlled_vs_proposed_raw | naturalness | 0.0040 | 0.0048 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.1894 | 1.1673 |
| controlled_vs_proposed_raw | context_overlap | 0.0474 | 0.7407 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2321 | 3.1622 |
| controlled_vs_proposed_raw | persona_style | -0.0162 | -0.0260 |
| controlled_vs_proposed_raw | distinct1 | 0.0007 | 0.0008 |
| controlled_vs_proposed_raw | length_score | -0.0250 | -0.0600 |
| controlled_vs_proposed_raw | sentence_score | 0.0875 | 0.1061 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0020 | 0.0268 |
| controlled_vs_proposed_raw | overall_quality | 0.1210 | 0.4541 |
| controlled_vs_candidate_no_context | context_relevance | 0.2577 | 11.7803 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1882 | 1.0587 |
| controlled_vs_candidate_no_context | naturalness | 0.0279 | 0.0342 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3365 | 22.2083 |
| controlled_vs_candidate_no_context | context_overlap | 0.0739 | 1.9670 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2302 | 3.0526 |
| controlled_vs_candidate_no_context | persona_style | 0.0202 | 0.0344 |
| controlled_vs_candidate_no_context | distinct1 | -0.0007 | -0.0007 |
| controlled_vs_candidate_no_context | length_score | 0.0972 | 0.3302 |
| controlled_vs_candidate_no_context | sentence_score | 0.0875 | 0.1061 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0528 | 2.0956 |
| controlled_vs_candidate_no_context | overall_quality | 0.1779 | 0.8484 |
| controlled_vs_baseline_no_context | context_relevance | 0.2267 | 4.2854 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1830 | 1.0006 |
| controlled_vs_baseline_no_context | naturalness | -0.0512 | -0.0571 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.2948 | 5.1889 |
| controlled_vs_baseline_no_context | context_overlap | 0.0677 | 1.5477 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2274 | 2.9086 |
| controlled_vs_baseline_no_context | persona_style | 0.0055 | 0.0091 |
| controlled_vs_baseline_no_context | distinct1 | -0.0308 | -0.0317 |
| controlled_vs_baseline_no_context | length_score | -0.1944 | -0.3318 |
| controlled_vs_baseline_no_context | sentence_score | 0.0000 | 0.0000 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0126 | 0.1934 |
| controlled_vs_baseline_no_context | overall_quality | 0.1447 | 0.5956 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.2452 | 7.1249 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1745 | 0.9113 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | -0.0545 | -0.0606 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.3226 | 11.1087 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0645 | 1.3744 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1988 | 1.8625 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | 0.0771 | 0.1453 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0279 | -0.0289 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | -0.2167 | -0.3562 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.0000 | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0291 | 0.5940 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1501 | 0.6319 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0473 | 0.1692 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0597 | -0.1632 |
| controlled_alt_vs_controlled_default | naturalness | -0.0073 | -0.0086 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0657 | 0.1867 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0045 | 0.0406 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0778 | -0.2545 |
| controlled_alt_vs_controlled_default | persona_style | 0.0126 | 0.0207 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0240 | -0.0256 |
| controlled_alt_vs_controlled_default | length_score | -0.0028 | -0.0071 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0292 | 0.0320 |
| controlled_alt_vs_controlled_default | bertscore_f1 | 0.0074 | 0.0953 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0010 | -0.0025 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1941 | 1.4620 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1228 | 0.6693 |
| controlled_alt_vs_proposed_raw | naturalness | -0.0032 | -0.0038 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2551 | 1.5720 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0520 | 0.8115 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1544 | 2.1027 |
| controlled_alt_vs_proposed_raw | persona_style | -0.0036 | -0.0058 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0233 | -0.0248 |
| controlled_alt_vs_proposed_raw | length_score | -0.0278 | -0.0667 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1167 | 0.1414 |
| controlled_alt_vs_proposed_raw | bertscore_f1 | 0.0095 | 0.1246 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1201 | 0.4504 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.3050 | 13.9433 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1285 | 0.7228 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0207 | 0.0253 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.4021 | 26.5417 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0784 | 2.0876 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1524 | 2.0211 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0328 | 0.0558 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0247 | -0.0263 |
| controlled_alt_vs_candidate_no_context | length_score | 0.0944 | 0.3208 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1167 | 0.1414 |
| controlled_alt_vs_candidate_no_context | bertscore_f1 | 0.0602 | 2.3905 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1769 | 0.8437 |
| controlled_alt_vs_baseline_no_context | context_relevance | 0.2740 | 5.1799 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 0.1233 | 0.6741 |
| controlled_alt_vs_baseline_no_context | naturalness | -0.0584 | -0.0652 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 0.3605 | 6.3444 |
| controlled_alt_vs_baseline_no_context | context_overlap | 0.0722 | 1.6513 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 0.1496 | 1.9137 |
| controlled_alt_vs_baseline_no_context | persona_style | 0.0181 | 0.0300 |
| controlled_alt_vs_baseline_no_context | distinct1 | -0.0548 | -0.0565 |
| controlled_alt_vs_baseline_no_context | length_score | -0.1972 | -0.3365 |
| controlled_alt_vs_baseline_no_context | sentence_score | 0.0292 | 0.0320 |
| controlled_alt_vs_baseline_no_context | bertscore_f1 | 0.0201 | 0.3071 |
| controlled_alt_vs_baseline_no_context | overall_quality | 0.1437 | 0.5915 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_relevance | 0.2925 | 8.4999 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1147 | 0.5994 |
| controlled_alt_vs_baseline_no_context_phi3_latest | naturalness | -0.0618 | -0.0686 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.3883 | 13.3696 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_overlap | 0.0691 | 1.4709 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1210 | 1.1338 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_style | 0.0896 | 0.1690 |
| controlled_alt_vs_baseline_no_context_phi3_latest | distinct1 | -0.0519 | -0.0537 |
| controlled_alt_vs_baseline_no_context_phi3_latest | length_score | -0.2194 | -0.3607 |
| controlled_alt_vs_baseline_no_context_phi3_latest | sentence_score | 0.0292 | 0.0320 |
| controlled_alt_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0365 | 0.7459 |
| controlled_alt_vs_baseline_no_context_phi3_latest | overall_quality | 0.1491 | 0.6278 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2267 | 4.2854 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1830 | 1.0006 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0512 | -0.0571 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.2948 | 5.1889 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0677 | 1.5477 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2274 | 2.9086 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0055 | 0.0091 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0308 | -0.0317 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | -0.1944 | -0.3318 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0000 | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0126 | 0.1934 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1447 | 0.5956 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.2452 | 7.1249 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1745 | 0.9113 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | -0.0545 | -0.0606 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.3226 | 11.1087 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0645 | 1.3744 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1988 | 1.8625 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | 0.0771 | 0.1453 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0279 | -0.0289 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | -0.2167 | -0.3562 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.0000 | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0291 | 0.5940 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1501 | 0.6319 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.1109 | (0.0312, 0.2059) | 0.0013 | 0.1109 | (-0.0013, 0.2194) | 0.0620 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0057 | (-0.0517, 0.0524) | 0.3857 | 0.0057 | (-0.0623, 0.0515) | 0.4323 |
| proposed_vs_candidate_no_context | naturalness | 0.0239 | (-0.0182, 0.0708) | 0.1510 | 0.0239 | (-0.0042, 0.0773) | 0.0610 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.1471 | (0.0417, 0.2778) | 0.0023 | 0.1471 | (0.0000, 0.2901) | 0.0897 |
| proposed_vs_candidate_no_context | context_overlap | 0.0265 | (0.0029, 0.0518) | 0.0103 | 0.0265 | (-0.0049, 0.0532) | 0.1033 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0020 | (-0.0714, 0.0536) | 0.5523 | -0.0020 | (-0.0779, 0.0536) | 0.5527 |
| proposed_vs_candidate_no_context | persona_style | 0.0364 | (0.0000, 0.0906) | 0.1190 | 0.0364 | (0.0000, 0.0770) | 0.3340 |
| proposed_vs_candidate_no_context | distinct1 | -0.0014 | (-0.0145, 0.0175) | 0.6113 | -0.0014 | (-0.0145, 0.0269) | 0.6173 |
| proposed_vs_candidate_no_context | length_score | 0.1222 | (-0.0611, 0.3278) | 0.1077 | 0.1222 | (0.0030, 0.3418) | 0.0157 |
| proposed_vs_candidate_no_context | sentence_score | 0.0000 | (-0.0875, 0.0875) | 0.6563 | 0.0000 | (-0.1167, 0.0583) | 0.6430 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0508 | (0.0082, 0.1015) | 0.0050 | 0.0508 | (0.0002, 0.0932) | 0.0250 |
| proposed_vs_candidate_no_context | overall_quality | 0.0569 | (0.0122, 0.1066) | 0.0043 | 0.0569 | (-0.0100, 0.1149) | 0.0840 |
| proposed_vs_baseline_no_context | context_relevance | 0.0799 | (0.0117, 0.1612) | 0.0077 | 0.0799 | (-0.0076, 0.1561) | 0.0590 |
| proposed_vs_baseline_no_context | persona_consistency | 0.0005 | (-0.0526, 0.0524) | 0.5163 | 0.0005 | (-0.0420, 0.0359) | 0.4517 |
| proposed_vs_baseline_no_context | naturalness | -0.0552 | (-0.1071, -0.0031) | 0.9823 | -0.0552 | (-0.1292, -0.0031) | 0.9837 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.1054 | (0.0139, 0.2146) | 0.0117 | 0.1054 | (-0.0114, 0.2008) | 0.0793 |
| proposed_vs_baseline_no_context | context_overlap | 0.0203 | (-0.0006, 0.0432) | 0.0297 | 0.0203 | (-0.0067, 0.0435) | 0.1317 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0048 | (-0.0631, 0.0496) | 0.6067 | -0.0048 | (-0.0462, 0.0429) | 0.6500 |
| proposed_vs_baseline_no_context | persona_style | 0.0217 | (-0.0338, 0.0910) | 0.2810 | 0.0217 | (-0.0556, 0.0711) | 0.3663 |
| proposed_vs_baseline_no_context | distinct1 | -0.0315 | (-0.0599, -0.0015) | 0.9807 | -0.0315 | (-0.0622, -0.0058) | 0.9930 |
| proposed_vs_baseline_no_context | length_score | -0.1694 | (-0.3778, 0.0139) | 0.9640 | -0.1694 | (-0.4259, 0.0200) | 0.9567 |
| proposed_vs_baseline_no_context | sentence_score | -0.0875 | (-0.2042, 0.0292) | 0.9570 | -0.0875 | (-0.2545, 0.0350) | 0.9247 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0106 | (-0.0417, 0.0742) | 0.3757 | 0.0106 | (-0.0741, 0.0828) | 0.4080 |
| proposed_vs_baseline_no_context | overall_quality | 0.0236 | (-0.0174, 0.0717) | 0.1553 | 0.0236 | (-0.0370, 0.0736) | 0.3093 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 0.0984 | (0.0202, 0.1929) | 0.0023 | 0.0984 | (-0.0123, 0.1971) | 0.0747 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | -0.0080 | (-0.0657, 0.0518) | 0.6103 | -0.0080 | (-0.1007, 0.0552) | 0.5817 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | -0.0585 | (-0.1042, -0.0099) | 0.9917 | -0.0585 | (-0.1183, -0.0005) | 0.9783 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.1332 | (0.0335, 0.2582) | 0.0020 | 0.1332 | (-0.0114, 0.2607) | 0.0860 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 0.0171 | (-0.0064, 0.0452) | 0.0927 | 0.0171 | (-0.0165, 0.0485) | 0.3220 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | -0.0333 | (-0.0944, 0.0250) | 0.9073 | -0.0333 | (-0.1259, 0.0238) | 0.8707 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | 0.0932 | (0.0000, 0.2193) | 0.0267 | 0.0932 | (0.0000, 0.1975) | 0.3547 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | -0.0286 | (-0.0423, -0.0158) | 1.0000 | -0.0286 | (-0.0424, -0.0140) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | -0.1917 | (-0.3750, 0.0222) | 0.9607 | -0.1917 | (-0.4000, 0.0815) | 0.9403 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | -0.0875 | (-0.2042, 0.0292) | 0.9630 | -0.0875 | (-0.2722, 0.0350) | 0.9340 |
| proposed_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0270 | (-0.0321, 0.0925) | 0.2003 | 0.0270 | (-0.0668, 0.0896) | 0.3070 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | 0.0290 | (-0.0231, 0.0901) | 0.1693 | 0.0290 | (-0.0488, 0.0992) | 0.3573 |
| controlled_vs_proposed_raw | context_relevance | 0.1468 | (0.0604, 0.2189) | 0.0003 | 0.1468 | (0.0622, 0.2517) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.1825 | (0.1246, 0.2371) | 0.0000 | 0.1825 | (0.1332, 0.2548) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0040 | (-0.0539, 0.0661) | 0.4460 | 0.0040 | (-0.0643, 0.0753) | 0.4783 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.1894 | (0.0789, 0.2854) | 0.0007 | 0.1894 | (0.0802, 0.3306) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0474 | (0.0173, 0.0754) | 0.0003 | 0.0474 | (0.0184, 0.0828) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2321 | (0.1635, 0.2972) | 0.0000 | 0.2321 | (0.1714, 0.3199) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | -0.0162 | (-0.0533, 0.0047) | 0.7657 | -0.0162 | (-0.0343, 0.0000) | 1.0000 |
| controlled_vs_proposed_raw | distinct1 | 0.0007 | (-0.0321, 0.0334) | 0.4857 | 0.0007 | (-0.0401, 0.0332) | 0.5140 |
| controlled_vs_proposed_raw | length_score | -0.0250 | (-0.2639, 0.2139) | 0.5843 | -0.0250 | (-0.3459, 0.2700) | 0.5593 |
| controlled_vs_proposed_raw | sentence_score | 0.0875 | (-0.0292, 0.2042) | 0.1083 | 0.0875 | (0.0175, 0.2188) | 0.0230 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0020 | (-0.0477, 0.0550) | 0.4717 | 0.0020 | (-0.0448, 0.0614) | 0.4837 |
| controlled_vs_proposed_raw | overall_quality | 0.1210 | (0.0715, 0.1668) | 0.0000 | 0.1210 | (0.0644, 0.1889) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2577 | (0.2184, 0.2976) | 0.0000 | 0.2577 | (0.2086, 0.2938) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1882 | (0.1097, 0.2497) | 0.0000 | 0.1882 | (0.1278, 0.2531) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.0279 | (-0.0258, 0.0826) | 0.1627 | 0.0279 | (-0.0285, 0.0955) | 0.1977 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3365 | (0.2866, 0.3914) | 0.0000 | 0.3365 | (0.2727, 0.3864) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0739 | (0.0490, 0.1003) | 0.0000 | 0.0739 | (0.0436, 0.0917) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2302 | (0.1405, 0.3071) | 0.0000 | 0.2302 | (0.1565, 0.3187) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.0202 | (-0.0433, 0.0953) | 0.2760 | 0.0202 | (0.0000, 0.0428) | 0.3550 |
| controlled_vs_candidate_no_context | distinct1 | -0.0007 | (-0.0350, 0.0297) | 0.5130 | -0.0007 | (-0.0335, 0.0358) | 0.5100 |
| controlled_vs_candidate_no_context | length_score | 0.0972 | (-0.1028, 0.3278) | 0.1747 | 0.0972 | (-0.1256, 0.3667) | 0.2107 |
| controlled_vs_candidate_no_context | sentence_score | 0.0875 | (-0.0583, 0.2333) | 0.1473 | 0.0875 | (-0.0350, 0.2100) | 0.1120 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0528 | (-0.0046, 0.1218) | 0.0397 | 0.0528 | (-0.0190, 0.1283) | 0.0837 |
| controlled_vs_candidate_no_context | overall_quality | 0.1779 | (0.1550, 0.1984) | 0.0000 | 0.1779 | (0.1513, 0.1999) | 0.0000 |
| controlled_vs_baseline_no_context | context_relevance | 0.2267 | (0.1947, 0.2692) | 0.0000 | 0.2267 | (0.1974, 0.2704) | 0.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1830 | (0.1486, 0.2223) | 0.0000 | 0.1830 | (0.1507, 0.2261) | 0.0000 |
| controlled_vs_baseline_no_context | naturalness | -0.0512 | (-0.0997, -0.0036) | 0.9797 | -0.0512 | (-0.0978, -0.0194) | 0.9977 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.2948 | (0.2525, 0.3485) | 0.0000 | 0.2948 | (0.2569, 0.3535) | 0.0000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0677 | (0.0489, 0.0875) | 0.0000 | 0.0677 | (0.0495, 0.0814) | 0.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2274 | (0.1821, 0.2778) | 0.0000 | 0.2274 | (0.1821, 0.2853) | 0.0000 |
| controlled_vs_baseline_no_context | persona_style | 0.0055 | (-0.0338, 0.0430) | 0.3933 | 0.0055 | (-0.0556, 0.0388) | 0.4533 |
| controlled_vs_baseline_no_context | distinct1 | -0.0308 | (-0.0583, 0.0038) | 0.9630 | -0.0308 | (-0.0614, -0.0118) | 0.9993 |
| controlled_vs_baseline_no_context | length_score | -0.1944 | (-0.3917, 0.0056) | 0.9703 | -0.1944 | (-0.4091, -0.0166) | 0.9853 |
| controlled_vs_baseline_no_context | sentence_score | 0.0000 | (-0.1167, 0.1167) | 0.5970 | 0.0000 | (-0.1273, 0.1077) | 0.6013 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0126 | (-0.0392, 0.0638) | 0.3157 | 0.0126 | (-0.0768, 0.0712) | 0.4203 |
| controlled_vs_baseline_no_context | overall_quality | 0.1447 | (0.1292, 0.1616) | 0.0000 | 0.1447 | (0.1267, 0.1637) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.2452 | (0.2058, 0.2935) | 0.0000 | 0.2452 | (0.1988, 0.2784) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1745 | (0.1385, 0.2279) | 0.0000 | 0.1745 | (0.1346, 0.1972) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | -0.0545 | (-0.1044, -0.0014) | 0.9783 | -0.0545 | (-0.0997, -0.0123) | 0.9947 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.3226 | (0.2670, 0.3845) | 0.0000 | 0.3226 | (0.2614, 0.3636) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0645 | (0.0442, 0.0848) | 0.0000 | 0.0645 | (0.0381, 0.0778) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1988 | (0.1675, 0.2361) | 0.0000 | 0.1988 | (0.1673, 0.2197) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | 0.0771 | (0.0000, 0.2001) | 0.0280 | 0.0771 | (0.0000, 0.1632) | 0.3320 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0279 | (-0.0610, 0.0048) | 0.9543 | -0.0279 | (-0.0579, -0.0005) | 0.9783 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | -0.2167 | (-0.4139, -0.0194) | 0.9820 | -0.2167 | (-0.4212, -0.0083) | 0.9783 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.0000 | (-0.1167, 0.1167) | 0.5943 | 0.0000 | (-0.0955, 0.0618) | 0.6350 |
| controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0291 | (-0.0240, 0.0846) | 0.1510 | 0.0291 | (-0.0623, 0.0847) | 0.2483 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1501 | (0.1271, 0.1726) | 0.0000 | 0.1501 | (0.1118, 0.1710) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0473 | (-0.0029, 0.1124) | 0.0340 | 0.0473 | (0.0102, 0.0789) | 0.0070 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0597 | (-0.1239, -0.0093) | 0.9937 | -0.0597 | (-0.1400, -0.0036) | 0.9813 |
| controlled_alt_vs_controlled_default | naturalness | -0.0073 | (-0.0605, 0.0446) | 0.6037 | -0.0073 | (-0.0371, 0.0492) | 0.6160 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0657 | (-0.0051, 0.1484) | 0.0270 | 0.0657 | (0.0165, 0.0996) | 0.0063 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0045 | (-0.0262, 0.0380) | 0.3950 | 0.0045 | (-0.0257, 0.0488) | 0.3890 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0778 | (-0.1556, -0.0139) | 1.0000 | -0.0778 | (-0.1800, -0.0167) | 1.0000 |
| controlled_alt_vs_controlled_default | persona_style | 0.0126 | (-0.0065, 0.0429) | 0.2363 | 0.0126 | (-0.0063, 0.0509) | 0.2677 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0240 | (-0.0488, -0.0012) | 0.9813 | -0.0240 | (-0.0350, -0.0027) | 0.9853 |
| controlled_alt_vs_controlled_default | length_score | -0.0028 | (-0.2417, 0.2251) | 0.5387 | -0.0028 | (-0.1412, 0.2584) | 0.5227 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0292 | (0.0000, 0.0875) | 0.3477 | 0.0292 | (0.0000, 0.0955) | 0.3257 |
| controlled_alt_vs_controlled_default | bertscore_f1 | 0.0074 | (-0.0380, 0.0531) | 0.3773 | 0.0074 | (-0.0332, 0.0598) | 0.3623 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0010 | (-0.0315, 0.0329) | 0.5240 | -0.0010 | (-0.0357, 0.0370) | 0.5040 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1941 | (0.1113, 0.2559) | 0.0000 | 0.1941 | (0.1140, 0.2856) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1228 | (0.0777, 0.1639) | 0.0000 | 0.1228 | (0.0960, 0.1727) | 0.0000 |
| controlled_alt_vs_proposed_raw | naturalness | -0.0032 | (-0.0677, 0.0681) | 0.5450 | -0.0032 | (-0.0647, 0.0847) | 0.5380 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2551 | (0.1395, 0.3390) | 0.0000 | 0.2551 | (0.1515, 0.3737) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0520 | (0.0316, 0.0746) | 0.0000 | 0.0520 | (0.0294, 0.0908) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1544 | (0.1071, 0.2008) | 0.0000 | 0.1544 | (0.1238, 0.2107) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_style | -0.0036 | (-0.0471, 0.0397) | 0.5963 | -0.0036 | (-0.0322, 0.0528) | 0.6353 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0233 | (-0.0468, -0.0008) | 0.9777 | -0.0233 | (-0.0505, 0.0058) | 0.9483 |
| controlled_alt_vs_proposed_raw | length_score | -0.0278 | (-0.2806, 0.2556) | 0.5913 | -0.0278 | (-0.2686, 0.3200) | 0.5897 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1167 | (0.0000, 0.2333) | 0.0463 | 0.1167 | (0.0206, 0.2800) | 0.0193 |
| controlled_alt_vs_proposed_raw | bertscore_f1 | 0.0095 | (-0.0117, 0.0301) | 0.2023 | 0.0095 | (-0.0137, 0.0415) | 0.2383 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1201 | (0.0702, 0.1597) | 0.0000 | 0.1201 | (0.0724, 0.1765) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.3050 | (0.2547, 0.3586) | 0.0000 | 0.3050 | (0.2374, 0.3527) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1285 | (0.0649, 0.1808) | 0.0000 | 0.1285 | (0.0714, 0.1899) | 0.0000 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0207 | (-0.0522, 0.0933) | 0.2900 | 0.0207 | (-0.0531, 0.1175) | 0.3200 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.4021 | (0.3409, 0.4716) | 0.0000 | 0.4021 | (0.3182, 0.4612) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0784 | (0.0555, 0.1028) | 0.0000 | 0.0784 | (0.0526, 0.1058) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1524 | (0.0738, 0.2131) | 0.0003 | 0.1524 | (0.0829, 0.2274) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0328 | (-0.0332, 0.1061) | 0.1770 | 0.0328 | (-0.0056, 0.0683) | 0.1040 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0247 | (-0.0490, -0.0018) | 0.9850 | -0.0247 | (-0.0503, 0.0066) | 0.9103 |
| controlled_alt_vs_candidate_no_context | length_score | 0.0944 | (-0.1778, 0.3778) | 0.2480 | 0.0944 | (-0.1983, 0.4834) | 0.2760 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1167 | (-0.0292, 0.2625) | 0.1017 | 0.1167 | (-0.0318, 0.2722) | 0.0727 |
| controlled_alt_vs_candidate_no_context | bertscore_f1 | 0.0602 | (0.0183, 0.1057) | 0.0003 | 0.0602 | (0.0209, 0.1092) | 0.0000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1769 | (0.1458, 0.2123) | 0.0000 | 0.1769 | (0.1471, 0.2122) | 0.0000 |
| controlled_alt_vs_baseline_no_context | context_relevance | 0.2740 | (0.2165, 0.3396) | 0.0000 | 0.2740 | (0.2275, 0.3241) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 0.1233 | (0.0524, 0.1766) | 0.0000 | 0.1233 | (0.0730, 0.1857) | 0.0000 |
| controlled_alt_vs_baseline_no_context | naturalness | -0.0584 | (-0.1068, -0.0126) | 0.9953 | -0.0584 | (-0.0940, -0.0174) | 0.9990 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 0.3605 | (0.2847, 0.4444) | 0.0000 | 0.3605 | (0.2987, 0.4273) | 0.0000 |
| controlled_alt_vs_baseline_no_context | context_overlap | 0.0722 | (0.0478, 0.0970) | 0.0000 | 0.0722 | (0.0489, 0.1048) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 0.1496 | (0.0710, 0.2159) | 0.0000 | 0.1496 | (0.0901, 0.2333) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_style | 0.0181 | (-0.0042, 0.0476) | 0.1177 | 0.0181 | (-0.0083, 0.0427) | 0.3357 |
| controlled_alt_vs_baseline_no_context | distinct1 | -0.0548 | (-0.0737, -0.0344) | 1.0000 | -0.0548 | (-0.0688, -0.0409) | 1.0000 |
| controlled_alt_vs_baseline_no_context | length_score | -0.1972 | (-0.4361, 0.0278) | 0.9520 | -0.1972 | (-0.3858, 0.0200) | 0.9653 |
| controlled_alt_vs_baseline_no_context | sentence_score | 0.0292 | (-0.0583, 0.1167) | 0.3847 | 0.0292 | (-0.0701, 0.1312) | 0.3790 |
| controlled_alt_vs_baseline_no_context | bertscore_f1 | 0.0201 | (-0.0165, 0.0669) | 0.1787 | 0.0201 | (-0.0360, 0.0706) | 0.3283 |
| controlled_alt_vs_baseline_no_context | overall_quality | 0.1437 | (0.1097, 0.1746) | 0.0000 | 0.1437 | (0.1116, 0.1761) | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_relevance | 0.2925 | (0.2401, 0.3493) | 0.0000 | 0.2925 | (0.2296, 0.3351) | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1147 | (0.0505, 0.1662) | 0.0000 | 0.1147 | (0.0181, 0.1684) | 0.0100 |
| controlled_alt_vs_baseline_no_context_phi3_latest | naturalness | -0.0618 | (-0.1196, -0.0035) | 0.9807 | -0.0618 | (-0.1092, 0.0038) | 0.9687 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.3883 | (0.3201, 0.4697) | 0.0000 | 0.3883 | (0.2987, 0.4444) | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_overlap | 0.0691 | (0.0470, 0.0925) | 0.0000 | 0.0691 | (0.0417, 0.0941) | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1210 | (0.0480, 0.1683) | 0.0007 | 0.1210 | (0.0190, 0.1719) | 0.0103 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_style | 0.0896 | (0.0056, 0.2118) | 0.0117 | 0.0896 | (-0.0056, 0.1746) | 0.0943 |
| controlled_alt_vs_baseline_no_context_phi3_latest | distinct1 | -0.0519 | (-0.0722, -0.0324) | 1.0000 | -0.0519 | (-0.0697, -0.0272) | 1.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | length_score | -0.2194 | (-0.4833, 0.0611) | 0.9340 | -0.2194 | (-0.4600, 0.1209) | 0.9093 |
| controlled_alt_vs_baseline_no_context_phi3_latest | sentence_score | 0.0292 | (-0.0583, 0.1167) | 0.3913 | 0.0292 | (0.0000, 0.0618) | 0.3370 |
| controlled_alt_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0365 | (-0.0076, 0.0890) | 0.0520 | 0.0365 | (-0.0264, 0.0785) | 0.1557 |
| controlled_alt_vs_baseline_no_context_phi3_latest | overall_quality | 0.1491 | (0.1143, 0.1871) | 0.0000 | 0.1491 | (0.1012, 0.1788) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2267 | (0.1958, 0.2687) | 0.0000 | 0.2267 | (0.1967, 0.2701) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1830 | (0.1472, 0.2223) | 0.0000 | 0.1830 | (0.1522, 0.2267) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0512 | (-0.0993, -0.0046) | 0.9843 | -0.0512 | (-0.1004, -0.0194) | 0.9977 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.2948 | (0.2519, 0.3523) | 0.0000 | 0.2948 | (0.2569, 0.3545) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0677 | (0.0485, 0.0886) | 0.0000 | 0.0677 | (0.0489, 0.0833) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2274 | (0.1861, 0.2778) | 0.0000 | 0.2274 | (0.1845, 0.2853) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0055 | (-0.0338, 0.0424) | 0.3987 | 0.0055 | (-0.0500, 0.0388) | 0.4377 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0308 | (-0.0583, 0.0020) | 0.9693 | -0.0308 | (-0.0586, -0.0126) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | -0.1944 | (-0.3972, 0.0001) | 0.9743 | -0.1944 | (-0.3963, -0.0185) | 0.9853 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0000 | (-0.1167, 0.1167) | 0.5870 | 0.0000 | (-0.1273, 0.1077) | 0.5763 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0126 | (-0.0417, 0.0653) | 0.3277 | 0.0126 | (-0.0717, 0.0683) | 0.4120 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1447 | (0.1286, 0.1620) | 0.0000 | 0.1447 | (0.1284, 0.1625) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.2452 | (0.2053, 0.2929) | 0.0000 | 0.2452 | (0.1953, 0.2772) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1745 | (0.1380, 0.2279) | 0.0000 | 0.1745 | (0.1338, 0.1974) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | -0.0545 | (-0.1057, -0.0034) | 0.9827 | -0.0545 | (-0.0970, -0.0168) | 0.9983 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.3226 | (0.2677, 0.3845) | 0.0000 | 0.3226 | (0.2614, 0.3646) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0645 | (0.0447, 0.0852) | 0.0000 | 0.0645 | (0.0378, 0.0778) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1988 | (0.1663, 0.2377) | 0.0000 | 0.1988 | (0.1673, 0.2190) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | 0.0771 | (0.0000, 0.1957) | 0.0290 | 0.0771 | (0.0000, 0.1734) | 0.3390 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0279 | (-0.0602, 0.0052) | 0.9517 | -0.0279 | (-0.0565, -0.0013) | 0.9787 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | -0.2167 | (-0.4056, -0.0111) | 0.9800 | -0.2167 | (-0.4030, -0.0055) | 0.9767 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.0000 | (-0.1167, 0.1167) | 0.6037 | 0.0000 | (-0.0955, 0.0618) | 0.6490 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0291 | (-0.0244, 0.0852) | 0.1503 | 0.0291 | (-0.0643, 0.0827) | 0.2473 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1501 | (0.1257, 0.1737) | 0.0000 | 0.1501 | (0.1120, 0.1709) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 6 | 1 | 5 | 0.7083 | 0.8571 |
| proposed_vs_candidate_no_context | persona_consistency | 4 | 1 | 7 | 0.6250 | 0.8000 |
| proposed_vs_candidate_no_context | naturalness | 4 | 4 | 4 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 5 | 0 | 7 | 0.7083 | 1.0000 |
| proposed_vs_candidate_no_context | context_overlap | 6 | 1 | 5 | 0.7083 | 0.8571 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 2 | 1 | 9 | 0.5417 | 0.6667 |
| proposed_vs_candidate_no_context | persona_style | 2 | 0 | 10 | 0.5833 | 1.0000 |
| proposed_vs_candidate_no_context | distinct1 | 1 | 6 | 5 | 0.2917 | 0.1429 |
| proposed_vs_candidate_no_context | length_score | 5 | 3 | 4 | 0.5833 | 0.6250 |
| proposed_vs_candidate_no_context | sentence_score | 1 | 1 | 10 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | bertscore_f1 | 5 | 4 | 3 | 0.5417 | 0.5556 |
| proposed_vs_candidate_no_context | overall_quality | 7 | 2 | 3 | 0.7083 | 0.7778 |
| proposed_vs_baseline_no_context | context_relevance | 6 | 6 | 0 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context | persona_consistency | 3 | 4 | 5 | 0.4583 | 0.4286 |
| proposed_vs_baseline_no_context | naturalness | 5 | 7 | 0 | 0.4167 | 0.4167 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 5 | 1 | 6 | 0.6667 | 0.8333 |
| proposed_vs_baseline_no_context | context_overlap | 6 | 5 | 1 | 0.5417 | 0.5455 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 2 | 2 | 8 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context | persona_style | 2 | 2 | 8 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context | distinct1 | 3 | 7 | 2 | 0.3333 | 0.3000 |
| proposed_vs_baseline_no_context | length_score | 5 | 7 | 0 | 0.4167 | 0.4167 |
| proposed_vs_baseline_no_context | sentence_score | 1 | 4 | 7 | 0.3750 | 0.2000 |
| proposed_vs_baseline_no_context | bertscore_f1 | 5 | 7 | 0 | 0.4167 | 0.4167 |
| proposed_vs_baseline_no_context | overall_quality | 7 | 5 | 0 | 0.5833 | 0.5833 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 7 | 5 | 0 | 0.5833 | 0.5833 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | 3 | 3 | 6 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | 3 | 9 | 0 | 0.2500 | 0.2500 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 6 | 1 | 5 | 0.7083 | 0.8571 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 6 | 6 | 0 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 1 | 3 | 8 | 0.4167 | 0.2500 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | 3 | 0 | 9 | 0.6250 | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | 0 | 10 | 2 | 0.0833 | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | 3 | 9 | 0 | 0.2500 | 0.2500 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | 1 | 4 | 7 | 0.3750 | 0.2000 |
| proposed_vs_baseline_no_context_phi3_latest | bertscore_f1 | 6 | 6 | 0 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | 5 | 7 | 0 | 0.4167 | 0.4167 |
| controlled_vs_proposed_raw | context_relevance | 10 | 2 | 0 | 0.8333 | 0.8333 |
| controlled_vs_proposed_raw | persona_consistency | 11 | 1 | 0 | 0.9167 | 0.9167 |
| controlled_vs_proposed_raw | naturalness | 7 | 5 | 0 | 0.5833 | 0.5833 |
| controlled_vs_proposed_raw | context_keyword_coverage | 10 | 2 | 0 | 0.8333 | 0.8333 |
| controlled_vs_proposed_raw | context_overlap | 10 | 2 | 0 | 0.8333 | 0.8333 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 11 | 0 | 1 | 0.9583 | 1.0000 |
| controlled_vs_proposed_raw | persona_style | 1 | 1 | 10 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | distinct1 | 6 | 6 | 0 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | length_score | 6 | 6 | 0 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | sentence_score | 4 | 1 | 7 | 0.6250 | 0.8000 |
| controlled_vs_proposed_raw | bertscore_f1 | 7 | 5 | 0 | 0.5833 | 0.5833 |
| controlled_vs_proposed_raw | overall_quality | 11 | 1 | 0 | 0.9167 | 0.9167 |
| controlled_vs_candidate_no_context | context_relevance | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 11 | 1 | 0 | 0.9167 | 0.9167 |
| controlled_vs_candidate_no_context | naturalness | 8 | 4 | 0 | 0.6667 | 0.6667 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 11 | 1 | 0 | 0.9167 | 0.9167 |
| controlled_vs_candidate_no_context | persona_style | 2 | 1 | 9 | 0.5417 | 0.6667 |
| controlled_vs_candidate_no_context | distinct1 | 7 | 5 | 0 | 0.5833 | 0.5833 |
| controlled_vs_candidate_no_context | length_score | 6 | 6 | 0 | 0.5000 | 0.5000 |
| controlled_vs_candidate_no_context | sentence_score | 5 | 2 | 5 | 0.6250 | 0.7143 |
| controlled_vs_candidate_no_context | bertscore_f1 | 10 | 2 | 0 | 0.8333 | 0.8333 |
| controlled_vs_candidate_no_context | overall_quality | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | context_relevance | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | naturalness | 4 | 8 | 0 | 0.3333 | 0.3333 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | context_overlap | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_style | 2 | 1 | 9 | 0.5417 | 0.6667 |
| controlled_vs_baseline_no_context | distinct1 | 2 | 9 | 1 | 0.2083 | 0.1818 |
| controlled_vs_baseline_no_context | length_score | 3 | 8 | 1 | 0.2917 | 0.2727 |
| controlled_vs_baseline_no_context | sentence_score | 2 | 2 | 8 | 0.5000 | 0.5000 |
| controlled_vs_baseline_no_context | bertscore_f1 | 6 | 6 | 0 | 0.5000 | 0.5000 |
| controlled_vs_baseline_no_context | overall_quality | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | 4 | 8 | 0 | 0.3333 | 0.3333 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | 3 | 0 | 9 | 0.6250 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | 5 | 7 | 0 | 0.4167 | 0.4167 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | 3 | 9 | 0 | 0.2500 | 0.2500 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 2 | 2 | 8 | 0.5000 | 0.5000 |
| controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 7 | 5 | 0 | 0.5833 | 0.5833 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 8 | 4 | 0 | 0.6667 | 0.6667 |
| controlled_alt_vs_controlled_default | persona_consistency | 2 | 6 | 4 | 0.3333 | 0.2500 |
| controlled_alt_vs_controlled_default | naturalness | 5 | 7 | 0 | 0.4167 | 0.4167 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 7 | 3 | 2 | 0.6667 | 0.7000 |
| controlled_alt_vs_controlled_default | context_overlap | 5 | 7 | 0 | 0.4167 | 0.4167 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0 | 4 | 8 | 0.3333 | 0.0000 |
| controlled_alt_vs_controlled_default | persona_style | 2 | 2 | 8 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | distinct1 | 4 | 8 | 0 | 0.3333 | 0.3333 |
| controlled_alt_vs_controlled_default | length_score | 5 | 7 | 0 | 0.4167 | 0.4167 |
| controlled_alt_vs_controlled_default | sentence_score | 1 | 0 | 11 | 0.5417 | 1.0000 |
| controlled_alt_vs_controlled_default | bertscore_f1 | 6 | 6 | 0 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | overall_quality | 4 | 8 | 0 | 0.3333 | 0.3333 |
| controlled_alt_vs_proposed_raw | context_relevance | 11 | 1 | 0 | 0.9167 | 0.9167 |
| controlled_alt_vs_proposed_raw | persona_consistency | 10 | 1 | 1 | 0.8750 | 0.9091 |
| controlled_alt_vs_proposed_raw | naturalness | 4 | 8 | 0 | 0.3333 | 0.3333 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 11 | 1 | 0 | 0.9167 | 0.9167 |
| controlled_alt_vs_proposed_raw | context_overlap | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 10 | 0 | 2 | 0.9167 | 1.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 2 | 3 | 7 | 0.4583 | 0.4000 |
| controlled_alt_vs_proposed_raw | distinct1 | 5 | 7 | 0 | 0.4167 | 0.4167 |
| controlled_alt_vs_proposed_raw | length_score | 5 | 7 | 0 | 0.4167 | 0.4167 |
| controlled_alt_vs_proposed_raw | sentence_score | 5 | 1 | 6 | 0.6667 | 0.8333 |
| controlled_alt_vs_proposed_raw | bertscore_f1 | 7 | 5 | 0 | 0.5833 | 0.5833 |
| controlled_alt_vs_proposed_raw | overall_quality | 11 | 1 | 0 | 0.9167 | 0.9167 |
| controlled_alt_vs_candidate_no_context | context_relevance | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 10 | 1 | 1 | 0.8750 | 0.9091 |
| controlled_alt_vs_candidate_no_context | naturalness | 6 | 6 | 0 | 0.5000 | 0.5000 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 10 | 1 | 1 | 0.8750 | 0.9091 |
| controlled_alt_vs_candidate_no_context | persona_style | 3 | 2 | 7 | 0.5417 | 0.6000 |
| controlled_alt_vs_candidate_no_context | distinct1 | 4 | 7 | 1 | 0.3750 | 0.3636 |
| controlled_alt_vs_candidate_no_context | length_score | 7 | 5 | 0 | 0.5833 | 0.5833 |
| controlled_alt_vs_candidate_no_context | sentence_score | 6 | 2 | 4 | 0.6667 | 0.7500 |
| controlled_alt_vs_candidate_no_context | bertscore_f1 | 9 | 3 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_candidate_no_context | overall_quality | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context | context_relevance | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 10 | 1 | 1 | 0.8750 | 0.9091 |
| controlled_alt_vs_baseline_no_context | naturalness | 2 | 10 | 0 | 0.1667 | 0.1667 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context | context_overlap | 11 | 1 | 0 | 0.9167 | 0.9167 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 10 | 1 | 1 | 0.8750 | 0.9091 |
| controlled_alt_vs_baseline_no_context | persona_style | 2 | 1 | 9 | 0.5417 | 0.6667 |
| controlled_alt_vs_baseline_no_context | distinct1 | 1 | 11 | 0 | 0.0833 | 0.0833 |
| controlled_alt_vs_baseline_no_context | length_score | 4 | 8 | 0 | 0.3333 | 0.3333 |
| controlled_alt_vs_baseline_no_context | sentence_score | 2 | 1 | 9 | 0.5417 | 0.6667 |
| controlled_alt_vs_baseline_no_context | bertscore_f1 | 6 | 6 | 0 | 0.5000 | 0.5000 |
| controlled_alt_vs_baseline_no_context | overall_quality | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_relevance | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_consistency | 10 | 1 | 1 | 0.8750 | 0.9091 |
| controlled_alt_vs_baseline_no_context_phi3_latest | naturalness | 5 | 7 | 0 | 0.4167 | 0.4167 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_overlap | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 10 | 1 | 1 | 0.8750 | 0.9091 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_style | 4 | 2 | 6 | 0.5833 | 0.6667 |
| controlled_alt_vs_baseline_no_context_phi3_latest | distinct1 | 1 | 11 | 0 | 0.0833 | 0.0833 |
| controlled_alt_vs_baseline_no_context_phi3_latest | length_score | 5 | 7 | 0 | 0.4167 | 0.4167 |
| controlled_alt_vs_baseline_no_context_phi3_latest | sentence_score | 2 | 1 | 9 | 0.5417 | 0.6667 |
| controlled_alt_vs_baseline_no_context_phi3_latest | bertscore_f1 | 6 | 6 | 0 | 0.5000 | 0.5000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | overall_quality | 12 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 12 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 12 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 4 | 8 | 0 | 0.3333 | 0.3333 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 12 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 12 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 12 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 2 | 1 | 9 | 0.5417 | 0.6667 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | 2 | 9 | 1 | 0.2083 | 0.1818 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 3 | 8 | 1 | 0.2917 | 0.2727 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 2 | 2 | 8 | 0.5000 | 0.5000 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 6 | 6 | 0 | 0.5000 | 0.5000 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 12 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 12 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 12 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | 4 | 8 | 0 | 0.3333 | 0.3333 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 12 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 12 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 12 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | 3 | 0 | 9 | 0.6250 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | 5 | 7 | 0 | 0.4167 | 0.4167 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | 3 | 9 | 0 | 0.2500 | 0.2500 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | 2 | 2 | 8 | 0.5000 | 0.5000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 7 | 5 | 0 | 0.5833 | 0.5833 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 12 | 0 | 0 | 1.0000 | 1.0000 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.0000 | 0.6667 | 0.3333 |
| proposed_contextual_controlled_runtime | 0.0000 | 0.0000 | 0.2500 | 0.5833 | 0.4167 |
| proposed_contextual | 0.0000 | 0.0000 | 0.3333 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 |
| baseline_no_context | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline_no_context_phi3_latest | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `7`
- Unique template signatures: `12`
- Template signature ratio: `1.0000`
- Effective sample size by source clustering: `5.14`
- Effective sample size by template-signature clustering: `12.00`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: enabled.

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
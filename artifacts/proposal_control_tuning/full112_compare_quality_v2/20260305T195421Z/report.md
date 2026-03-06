# Proposal Alignment Evaluation Report

- Run ID: `20260305T195421Z`
- Generated: `2026-03-05T19:55:16.733731+00:00`
- Scenarios: `artifacts\proposal_control_tuning\full112_compare_quality_v2\20260305T195421Z\scenarios.jsonl`
- Scenario count: `112`

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
| proposed_contextual_controlled | 0.2776 (0.2605, 0.2952) | 0.3822 (0.3516, 0.4116) | 0.8733 (0.8594, 0.8869) | 0.3985 (0.3864, 0.4106) | 0.0876 |
| proposed_contextual_controlled_quality | 0.2757 (0.2611, 0.2918) | 0.3864 (0.3563, 0.4181) | 0.8830 (0.8700, 0.8955) | 0.4008 (0.3901, 0.4122) | 0.0874 |
| proposed_contextual | 0.0696 (0.0527, 0.0870) | 0.1481 (0.1256, 0.1712) | 0.8118 (0.7984, 0.8251) | 0.2251 (0.2112, 0.2388) | 0.0759 |
| candidate_no_context | 0.0290 (0.0223, 0.0361) | 0.1744 (0.1471, 0.2008) | 0.8212 (0.8060, 0.8359) | 0.2144 (0.2029, 0.2264) | 0.0374 |
| baseline_no_context | 0.0545 (0.0432, 0.0657) | 0.1811 (0.1598, 0.2029) | 0.8828 (0.8729, 0.8932) | 0.2396 (0.2302, 0.2498) | 0.0562 |
| baseline_no_context_phi3_latest | 0.0498 (0.0405, 0.0597) | 0.1871 (0.1609, 0.2157) | 0.8826 (0.8728, 0.8921) | 0.2394 (0.2291, 0.2502) | 0.0553 |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0406 | 1.3999 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0264 | -0.1512 |
| proposed_vs_candidate_no_context | naturalness | -0.0095 | -0.0115 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0510 | 2.1096 |
| proposed_vs_candidate_no_context | context_overlap | 0.0162 | 0.4026 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0282 | -0.3203 |
| proposed_vs_candidate_no_context | persona_style | -0.0191 | -0.0367 |
| proposed_vs_candidate_no_context | distinct1 | 0.0015 | 0.0016 |
| proposed_vs_candidate_no_context | length_score | -0.0440 | -0.1270 |
| proposed_vs_candidate_no_context | sentence_score | -0.0125 | -0.0163 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0385 | 1.0318 |
| proposed_vs_candidate_no_context | overall_quality | 0.0106 | 0.0497 |
| proposed_vs_baseline_no_context | context_relevance | 0.0151 | 0.2771 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0331 | -0.1826 |
| proposed_vs_baseline_no_context | naturalness | -0.0711 | -0.0805 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0164 | 0.2787 |
| proposed_vs_baseline_no_context | context_overlap | 0.0121 | 0.2721 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0205 | -0.2556 |
| proposed_vs_baseline_no_context | persona_style | -0.0832 | -0.1425 |
| proposed_vs_baseline_no_context | distinct1 | -0.0395 | -0.0404 |
| proposed_vs_baseline_no_context | length_score | -0.2199 | -0.4208 |
| proposed_vs_baseline_no_context | sentence_score | -0.1125 | -0.1295 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0197 | 0.3514 |
| proposed_vs_baseline_no_context | overall_quality | -0.0145 | -0.0607 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 0.0197 | 0.3955 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | -0.0390 | -0.2085 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | -0.0708 | -0.0802 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.0232 | 0.4460 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 0.0116 | 0.2583 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | -0.0261 | -0.3038 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | -0.0906 | -0.1531 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | -0.0397 | -0.0406 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | -0.2167 | -0.4172 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | -0.1156 | -0.1326 |
| proposed_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0206 | 0.3719 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | -0.0143 | -0.0599 |
| controlled_vs_proposed_raw | context_relevance | 0.2081 | 2.9918 |
| controlled_vs_proposed_raw | persona_consistency | 0.2342 | 1.5815 |
| controlled_vs_proposed_raw | naturalness | 0.0616 | 0.0758 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2737 | 3.6388 |
| controlled_vs_proposed_raw | context_overlap | 0.0550 | 0.9761 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2684 | 4.4861 |
| controlled_vs_proposed_raw | persona_style | 0.0973 | 0.1943 |
| controlled_vs_proposed_raw | distinct1 | -0.0029 | -0.0031 |
| controlled_vs_proposed_raw | length_score | 0.2274 | 0.7512 |
| controlled_vs_proposed_raw | sentence_score | 0.1750 | 0.2314 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0117 | 0.1543 |
| controlled_vs_proposed_raw | overall_quality | 0.1734 | 0.7706 |
| controlled_vs_candidate_no_context | context_relevance | 0.2487 | 8.5798 |
| controlled_vs_candidate_no_context | persona_consistency | 0.2078 | 1.1912 |
| controlled_vs_candidate_no_context | naturalness | 0.0521 | 0.0634 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3247 | 13.4251 |
| controlled_vs_candidate_no_context | context_overlap | 0.0712 | 1.7717 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2402 | 2.7290 |
| controlled_vs_candidate_no_context | persona_style | 0.0782 | 0.1504 |
| controlled_vs_candidate_no_context | distinct1 | -0.0014 | -0.0015 |
| controlled_vs_candidate_no_context | length_score | 0.1833 | 0.5288 |
| controlled_vs_candidate_no_context | sentence_score | 0.1625 | 0.2114 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0503 | 1.3452 |
| controlled_vs_candidate_no_context | overall_quality | 0.1841 | 0.8585 |
| controlled_vs_baseline_no_context | context_relevance | 0.2232 | 4.0980 |
| controlled_vs_baseline_no_context | persona_consistency | 0.2011 | 1.1101 |
| controlled_vs_baseline_no_context | naturalness | -0.0095 | -0.0108 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.2901 | 4.9319 |
| controlled_vs_baseline_no_context | context_overlap | 0.0670 | 1.5138 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2478 | 3.0841 |
| controlled_vs_baseline_no_context | persona_style | 0.0141 | 0.0241 |
| controlled_vs_baseline_no_context | distinct1 | -0.0424 | -0.0434 |
| controlled_vs_baseline_no_context | length_score | 0.0074 | 0.0142 |
| controlled_vs_baseline_no_context | sentence_score | 0.0625 | 0.0719 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0314 | 0.5598 |
| controlled_vs_baseline_no_context | overall_quality | 0.1589 | 0.6632 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.2278 | 4.5703 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1952 | 1.0433 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | -0.0092 | -0.0105 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.2969 | 5.7079 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0666 | 1.4866 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.2423 | 2.8194 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | 0.0068 | 0.0114 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0426 | -0.0436 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | 0.0107 | 0.0206 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.0594 | 0.0681 |
| controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0323 | 0.5835 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1591 | 0.6646 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0019 | -0.0068 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0042 | 0.0110 |
| controlled_alt_vs_controlled_default | naturalness | 0.0097 | 0.0111 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0026 | -0.0075 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0002 | -0.0021 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0041 | 0.0124 |
| controlled_alt_vs_controlled_default | persona_style | 0.0047 | 0.0078 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0022 | 0.0023 |
| controlled_alt_vs_controlled_default | length_score | 0.0354 | 0.0668 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0156 | 0.0168 |
| controlled_alt_vs_controlled_default | bertscore_f1 | -0.0002 | -0.0028 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0023 | 0.0057 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.2062 | 2.9645 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.2384 | 1.6099 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0713 | 0.0878 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2711 | 3.6041 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0548 | 0.9720 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.2724 | 4.5544 |
| controlled_alt_vs_proposed_raw | persona_style | 0.1020 | 0.2036 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0008 | -0.0008 |
| controlled_alt_vs_proposed_raw | length_score | 0.2628 | 0.8682 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1906 | 0.2521 |
| controlled_alt_vs_proposed_raw | bertscore_f1 | 0.0115 | 0.1510 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1757 | 0.7807 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2468 | 8.5144 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.2120 | 1.2153 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0618 | 0.0753 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3221 | 13.3171 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0709 | 1.7660 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.2443 | 2.7754 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0829 | 0.1594 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0008 | 0.0008 |
| controlled_alt_vs_candidate_no_context | length_score | 0.2187 | 0.6309 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1781 | 0.2317 |
| controlled_alt_vs_candidate_no_context | bertscore_f1 | 0.0500 | 1.3385 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1864 | 0.8692 |
| controlled_alt_vs_baseline_no_context | context_relevance | 0.2213 | 4.0632 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 0.2053 | 1.1333 |
| controlled_alt_vs_baseline_no_context | naturalness | 0.0002 | 0.0002 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 0.2875 | 4.8875 |
| controlled_alt_vs_baseline_no_context | context_overlap | 0.0668 | 1.5086 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 0.2519 | 3.1349 |
| controlled_alt_vs_baseline_no_context | persona_style | 0.0188 | 0.0321 |
| controlled_alt_vs_baseline_no_context | distinct1 | -0.0403 | -0.0412 |
| controlled_alt_vs_baseline_no_context | length_score | 0.0429 | 0.0820 |
| controlled_alt_vs_baseline_no_context | sentence_score | 0.0781 | 0.0899 |
| controlled_alt_vs_baseline_no_context | bertscore_f1 | 0.0312 | 0.5554 |
| controlled_alt_vs_baseline_no_context | overall_quality | 0.1612 | 0.6727 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_relevance | 0.2259 | 4.5323 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1994 | 1.0657 |
| controlled_alt_vs_baseline_no_context_phi3_latest | naturalness | 0.0005 | 0.0005 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.2943 | 5.6577 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_overlap | 0.0663 | 1.4814 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.2463 | 2.8669 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_style | 0.0114 | 0.0193 |
| controlled_alt_vs_baseline_no_context_phi3_latest | distinct1 | -0.0405 | -0.0414 |
| controlled_alt_vs_baseline_no_context_phi3_latest | length_score | 0.0461 | 0.0888 |
| controlled_alt_vs_baseline_no_context_phi3_latest | sentence_score | 0.0750 | 0.0860 |
| controlled_alt_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0320 | 0.5790 |
| controlled_alt_vs_baseline_no_context_phi3_latest | overall_quality | 0.1614 | 0.6741 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2232 | 4.0980 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.2011 | 1.1101 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0095 | -0.0108 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.2901 | 4.9319 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0670 | 1.5138 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2478 | 3.0841 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0141 | 0.0241 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0424 | -0.0434 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.0074 | 0.0142 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0625 | 0.0719 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0314 | 0.5598 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1589 | 0.6632 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.2278 | 4.5703 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1952 | 1.0433 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | -0.0092 | -0.0105 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.2969 | 5.7079 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0666 | 1.4866 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.2423 | 2.8194 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | 0.0068 | 0.0114 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0426 | -0.0436 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | 0.0107 | 0.0206 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.0594 | 0.0681 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0323 | 0.5835 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1591 | 0.6646 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0406 | (0.0244, 0.0572) | 0.0000 | 0.0406 | (0.0143, 0.0719) | 0.0000 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0264 | (-0.0550, 0.0007) | 0.9733 | -0.0264 | (-0.0687, 0.0002) | 0.9713 |
| proposed_vs_candidate_no_context | naturalness | -0.0095 | (-0.0255, 0.0074) | 0.8640 | -0.0095 | (-0.0299, 0.0094) | 0.8283 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0510 | (0.0297, 0.0725) | 0.0000 | 0.0510 | (0.0180, 0.0906) | 0.0000 |
| proposed_vs_candidate_no_context | context_overlap | 0.0162 | (0.0098, 0.0227) | 0.0000 | 0.0162 | (0.0044, 0.0287) | 0.0007 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0282 | (-0.0626, 0.0015) | 0.9647 | -0.0282 | (-0.0799, 0.0077) | 0.9020 |
| proposed_vs_candidate_no_context | persona_style | -0.0191 | (-0.0501, 0.0094) | 0.9010 | -0.0191 | (-0.0561, 0.0081) | 0.8847 |
| proposed_vs_candidate_no_context | distinct1 | 0.0015 | (-0.0051, 0.0083) | 0.3320 | 0.0015 | (-0.0045, 0.0081) | 0.3433 |
| proposed_vs_candidate_no_context | length_score | -0.0440 | (-0.1089, 0.0217) | 0.9107 | -0.0440 | (-0.1110, 0.0170) | 0.9120 |
| proposed_vs_candidate_no_context | sentence_score | -0.0125 | (-0.0531, 0.0250) | 0.7693 | -0.0125 | (-0.0719, 0.0531) | 0.6747 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0385 | (0.0275, 0.0504) | 0.0000 | 0.0385 | (0.0195, 0.0595) | 0.0000 |
| proposed_vs_candidate_no_context | overall_quality | 0.0106 | (-0.0043, 0.0263) | 0.0887 | 0.0106 | (-0.0127, 0.0327) | 0.1883 |
| proposed_vs_baseline_no_context | context_relevance | 0.0151 | (-0.0039, 0.0342) | 0.0573 | 0.0151 | (-0.0221, 0.0499) | 0.2043 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0331 | (-0.0640, -0.0036) | 0.9840 | -0.0331 | (-0.0586, -0.0081) | 0.9973 |
| proposed_vs_baseline_no_context | naturalness | -0.0711 | (-0.0870, -0.0546) | 1.0000 | -0.0711 | (-0.0971, -0.0448) | 1.0000 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0164 | (-0.0085, 0.0433) | 0.1123 | 0.0164 | (-0.0321, 0.0628) | 0.2407 |
| proposed_vs_baseline_no_context | context_overlap | 0.0121 | (0.0051, 0.0189) | 0.0000 | 0.0121 | (0.0012, 0.0227) | 0.0130 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0205 | (-0.0558, 0.0151) | 0.8720 | -0.0205 | (-0.0488, 0.0051) | 0.9367 |
| proposed_vs_baseline_no_context | persona_style | -0.0832 | (-0.1257, -0.0466) | 1.0000 | -0.0832 | (-0.2005, -0.0021) | 0.9817 |
| proposed_vs_baseline_no_context | distinct1 | -0.0395 | (-0.0470, -0.0319) | 1.0000 | -0.0395 | (-0.0546, -0.0243) | 1.0000 |
| proposed_vs_baseline_no_context | length_score | -0.2199 | (-0.2839, -0.1577) | 1.0000 | -0.2199 | (-0.3140, -0.1199) | 1.0000 |
| proposed_vs_baseline_no_context | sentence_score | -0.1125 | (-0.1531, -0.0719) | 1.0000 | -0.1125 | (-0.1844, -0.0312) | 0.9963 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0197 | (0.0076, 0.0325) | 0.0010 | 0.0197 | (-0.0057, 0.0454) | 0.0703 |
| proposed_vs_baseline_no_context | overall_quality | -0.0145 | (-0.0315, 0.0024) | 0.9517 | -0.0145 | (-0.0381, 0.0114) | 0.8697 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 0.0197 | (0.0015, 0.0391) | 0.0160 | 0.0197 | (-0.0145, 0.0532) | 0.1397 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | -0.0390 | (-0.0696, -0.0094) | 0.9953 | -0.0390 | (-0.0728, -0.0033) | 0.9827 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | -0.0708 | (-0.0861, -0.0549) | 1.0000 | -0.0708 | (-0.0977, -0.0426) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.0232 | (-0.0003, 0.0477) | 0.0257 | 0.0232 | (-0.0222, 0.0685) | 0.1637 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 0.0116 | (0.0047, 0.0190) | 0.0000 | 0.0116 | (-0.0009, 0.0252) | 0.0383 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | -0.0261 | (-0.0628, 0.0104) | 0.9120 | -0.0261 | (-0.0633, 0.0116) | 0.9187 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | -0.0906 | (-0.1291, -0.0540) | 1.0000 | -0.0906 | (-0.2100, -0.0037) | 0.9797 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | -0.0397 | (-0.0479, -0.0316) | 1.0000 | -0.0397 | (-0.0551, -0.0220) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | -0.2167 | (-0.2804, -0.1551) | 1.0000 | -0.2167 | (-0.3104, -0.1140) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | -0.1156 | (-0.1562, -0.0719) | 1.0000 | -0.1156 | (-0.1812, -0.0437) | 0.9993 |
| proposed_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0206 | (0.0074, 0.0341) | 0.0013 | 0.0206 | (-0.0081, 0.0526) | 0.0907 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | -0.0143 | (-0.0296, 0.0017) | 0.9590 | -0.0143 | (-0.0378, 0.0128) | 0.8543 |
| controlled_vs_proposed_raw | context_relevance | 0.2081 | (0.1859, 0.2309) | 0.0000 | 0.2081 | (0.1807, 0.2315) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.2342 | (0.1996, 0.2709) | 0.0000 | 0.2342 | (0.1906, 0.2732) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0616 | (0.0404, 0.0820) | 0.0000 | 0.0616 | (0.0193, 0.1010) | 0.0017 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2737 | (0.2438, 0.3033) | 0.0000 | 0.2737 | (0.2394, 0.3026) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0550 | (0.0464, 0.0639) | 0.0000 | 0.0550 | (0.0430, 0.0680) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2684 | (0.2285, 0.3094) | 0.0000 | 0.2684 | (0.2128, 0.3222) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.0973 | (0.0586, 0.1374) | 0.0000 | 0.0973 | (0.0150, 0.2045) | 0.0027 |
| controlled_vs_proposed_raw | distinct1 | -0.0029 | (-0.0122, 0.0059) | 0.7390 | -0.0029 | (-0.0193, 0.0123) | 0.6257 |
| controlled_vs_proposed_raw | length_score | 0.2274 | (0.1423, 0.3125) | 0.0000 | 0.2274 | (0.0687, 0.3732) | 0.0023 |
| controlled_vs_proposed_raw | sentence_score | 0.1750 | (0.1375, 0.2125) | 0.0000 | 0.1750 | (0.1062, 0.2344) | 0.0000 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0117 | (-0.0012, 0.0251) | 0.0407 | 0.0117 | (-0.0081, 0.0316) | 0.1213 |
| controlled_vs_proposed_raw | overall_quality | 0.1734 | (0.1560, 0.1911) | 0.0000 | 0.1734 | (0.1507, 0.1968) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2487 | (0.2320, 0.2661) | 0.0000 | 0.2487 | (0.2343, 0.2623) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.2078 | (0.1729, 0.2431) | 0.0000 | 0.2078 | (0.1589, 0.2561) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.0521 | (0.0319, 0.0728) | 0.0000 | 0.0521 | (0.0124, 0.0909) | 0.0063 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3247 | (0.3033, 0.3474) | 0.0000 | 0.3247 | (0.3049, 0.3425) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0712 | (0.0635, 0.0791) | 0.0000 | 0.0712 | (0.0639, 0.0789) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2402 | (0.2002, 0.2813) | 0.0000 | 0.2402 | (0.1839, 0.2980) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.0782 | (0.0436, 0.1164) | 0.0000 | 0.0782 | (0.0106, 0.1608) | 0.0037 |
| controlled_vs_candidate_no_context | distinct1 | -0.0014 | (-0.0100, 0.0072) | 0.6333 | -0.0014 | (-0.0155, 0.0111) | 0.5820 |
| controlled_vs_candidate_no_context | length_score | 0.1833 | (0.1015, 0.2676) | 0.0003 | 0.1833 | (0.0330, 0.3244) | 0.0080 |
| controlled_vs_candidate_no_context | sentence_score | 0.1625 | (0.1219, 0.2031) | 0.0000 | 0.1625 | (0.1031, 0.2250) | 0.0000 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0503 | (0.0369, 0.0637) | 0.0000 | 0.0503 | (0.0340, 0.0672) | 0.0000 |
| controlled_vs_candidate_no_context | overall_quality | 0.1841 | (0.1702, 0.1990) | 0.0000 | 0.1841 | (0.1654, 0.2041) | 0.0000 |
| controlled_vs_baseline_no_context | context_relevance | 0.2232 | (0.2026, 0.2432) | 0.0000 | 0.2232 | (0.1917, 0.2531) | 0.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 0.2011 | (0.1684, 0.2349) | 0.0000 | 0.2011 | (0.1425, 0.2487) | 0.0000 |
| controlled_vs_baseline_no_context | naturalness | -0.0095 | (-0.0261, 0.0073) | 0.8530 | -0.0095 | (-0.0356, 0.0210) | 0.7510 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.2901 | (0.2641, 0.3176) | 0.0000 | 0.2901 | (0.2433, 0.3337) | 0.0000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0670 | (0.0594, 0.0748) | 0.0000 | 0.0670 | (0.0624, 0.0725) | 0.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2478 | (0.2065, 0.2892) | 0.0000 | 0.2478 | (0.1768, 0.3077) | 0.0000 |
| controlled_vs_baseline_no_context | persona_style | 0.0141 | (-0.0075, 0.0378) | 0.1090 | 0.0141 | (-0.0057, 0.0381) | 0.1187 |
| controlled_vs_baseline_no_context | distinct1 | -0.0424 | (-0.0494, -0.0355) | 1.0000 | -0.0424 | (-0.0500, -0.0349) | 1.0000 |
| controlled_vs_baseline_no_context | length_score | 0.0074 | (-0.0706, 0.0869) | 0.4253 | 0.0074 | (-0.1238, 0.1438) | 0.4803 |
| controlled_vs_baseline_no_context | sentence_score | 0.0625 | (0.0250, 0.1000) | 0.0010 | 0.0625 | (-0.0125, 0.1500) | 0.0580 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0314 | (0.0194, 0.0437) | 0.0000 | 0.0314 | (0.0195, 0.0459) | 0.0000 |
| controlled_vs_baseline_no_context | overall_quality | 0.1589 | (0.1445, 0.1732) | 0.0000 | 0.1589 | (0.1354, 0.1787) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.2278 | (0.2081, 0.2477) | 0.0000 | 0.2278 | (0.2023, 0.2552) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1952 | (0.1584, 0.2319) | 0.0000 | 0.1952 | (0.1353, 0.2513) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | -0.0092 | (-0.0272, 0.0085) | 0.8510 | -0.0092 | (-0.0389, 0.0244) | 0.7097 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.2969 | (0.2689, 0.3245) | 0.0000 | 0.2969 | (0.2614, 0.3342) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0666 | (0.0592, 0.0742) | 0.0000 | 0.0666 | (0.0614, 0.0718) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.2423 | (0.1950, 0.2877) | 0.0000 | 0.2423 | (0.1720, 0.3140) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | 0.0068 | (-0.0130, 0.0267) | 0.2710 | 0.0068 | (-0.0126, 0.0372) | 0.3490 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0426 | (-0.0511, -0.0344) | 1.0000 | -0.0426 | (-0.0489, -0.0365) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | 0.0107 | (-0.0723, 0.0926) | 0.3793 | 0.0107 | (-0.1328, 0.1634) | 0.4247 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.0594 | (0.0187, 0.0969) | 0.0037 | 0.0594 | (-0.0156, 0.1313) | 0.0580 |
| controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0323 | (0.0181, 0.0459) | 0.0000 | 0.0323 | (0.0130, 0.0536) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1591 | (0.1447, 0.1729) | 0.0000 | 0.1591 | (0.1344, 0.1825) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0019 | (-0.0215, 0.0185) | 0.5757 | -0.0019 | (-0.0137, 0.0125) | 0.6247 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0042 | (-0.0339, 0.0432) | 0.4027 | 0.0042 | (-0.0310, 0.0440) | 0.4420 |
| controlled_alt_vs_controlled_default | naturalness | 0.0097 | (-0.0073, 0.0265) | 0.1330 | 0.0097 | (-0.0050, 0.0212) | 0.0850 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0026 | (-0.0291, 0.0237) | 0.5710 | -0.0026 | (-0.0192, 0.0193) | 0.6280 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0002 | (-0.0096, 0.0092) | 0.5337 | -0.0002 | (-0.0076, 0.0072) | 0.5383 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0041 | (-0.0408, 0.0490) | 0.4340 | 0.0041 | (-0.0411, 0.0553) | 0.4423 |
| controlled_alt_vs_controlled_default | persona_style | 0.0047 | (-0.0173, 0.0268) | 0.3430 | 0.0047 | (-0.0177, 0.0280) | 0.3477 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0022 | (-0.0070, 0.0111) | 0.3253 | 0.0022 | (-0.0037, 0.0083) | 0.2623 |
| controlled_alt_vs_controlled_default | length_score | 0.0354 | (-0.0396, 0.1137) | 0.1790 | 0.0354 | (-0.0429, 0.1060) | 0.1797 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0156 | (-0.0187, 0.0500) | 0.2097 | 0.0156 | (-0.0187, 0.0500) | 0.2073 |
| controlled_alt_vs_controlled_default | bertscore_f1 | -0.0002 | (-0.0134, 0.0135) | 0.5243 | -0.0002 | (-0.0152, 0.0140) | 0.5030 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0023 | (-0.0115, 0.0157) | 0.3780 | 0.0023 | (-0.0094, 0.0142) | 0.3580 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.2062 | (0.1866, 0.2257) | 0.0000 | 0.2062 | (0.1842, 0.2260) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.2384 | (0.2063, 0.2692) | 0.0000 | 0.2384 | (0.1952, 0.2778) | 0.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0713 | (0.0502, 0.0915) | 0.0000 | 0.0713 | (0.0277, 0.1110) | 0.0007 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2711 | (0.2459, 0.2944) | 0.0000 | 0.2711 | (0.2457, 0.2933) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0548 | (0.0458, 0.0641) | 0.0000 | 0.0548 | (0.0378, 0.0732) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.2724 | (0.2348, 0.3113) | 0.0000 | 0.2724 | (0.2157, 0.3324) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 0.1020 | (0.0628, 0.1436) | 0.0000 | 0.1020 | (0.0089, 0.2164) | 0.0027 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0008 | (-0.0099, 0.0087) | 0.5727 | -0.0008 | (-0.0188, 0.0168) | 0.5283 |
| controlled_alt_vs_proposed_raw | length_score | 0.2628 | (0.1750, 0.3408) | 0.0000 | 0.2628 | (0.0976, 0.4200) | 0.0013 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1906 | (0.1530, 0.2281) | 0.0000 | 0.1906 | (0.1313, 0.2437) | 0.0000 |
| controlled_alt_vs_proposed_raw | bertscore_f1 | 0.0115 | (-0.0033, 0.0264) | 0.0577 | 0.0115 | (-0.0203, 0.0379) | 0.2283 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1757 | (0.1605, 0.1914) | 0.0000 | 0.1757 | (0.1509, 0.2019) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2468 | (0.2315, 0.2616) | 0.0000 | 0.2468 | (0.2258, 0.2698) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.2120 | (0.1803, 0.2462) | 0.0000 | 0.2120 | (0.1603, 0.2603) | 0.0000 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0618 | (0.0425, 0.0804) | 0.0000 | 0.0618 | (0.0229, 0.0988) | 0.0007 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3221 | (0.3024, 0.3427) | 0.0000 | 0.3221 | (0.2932, 0.3546) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0709 | (0.0632, 0.0794) | 0.0000 | 0.0709 | (0.0617, 0.0798) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.2443 | (0.2057, 0.2832) | 0.0000 | 0.2443 | (0.1873, 0.3066) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0829 | (0.0448, 0.1228) | 0.0000 | 0.0829 | (0.0049, 0.1819) | 0.0247 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0008 | (-0.0075, 0.0091) | 0.4267 | 0.0008 | (-0.0158, 0.0165) | 0.4660 |
| controlled_alt_vs_candidate_no_context | length_score | 0.2188 | (0.1416, 0.2920) | 0.0000 | 0.2188 | (0.0717, 0.3485) | 0.0007 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1781 | (0.1375, 0.2156) | 0.0000 | 0.1781 | (0.1094, 0.2375) | 0.0000 |
| controlled_alt_vs_candidate_no_context | bertscore_f1 | 0.0500 | (0.0372, 0.0623) | 0.0000 | 0.0500 | (0.0271, 0.0723) | 0.0000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1864 | (0.1734, 0.1994) | 0.0000 | 0.1864 | (0.1687, 0.2058) | 0.0000 |
| controlled_alt_vs_baseline_no_context | context_relevance | 0.2213 | (0.2026, 0.2408) | 0.0000 | 0.2213 | (0.1845, 0.2577) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 0.2053 | (0.1738, 0.2363) | 0.0000 | 0.2053 | (0.1662, 0.2478) | 0.0000 |
| controlled_alt_vs_baseline_no_context | naturalness | 0.0002 | (-0.0178, 0.0174) | 0.4877 | 0.0002 | (-0.0277, 0.0351) | 0.5153 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 0.2875 | (0.2629, 0.3133) | 0.0000 | 0.2875 | (0.2334, 0.3367) | 0.0000 |
| controlled_alt_vs_baseline_no_context | context_overlap | 0.0668 | (0.0584, 0.0758) | 0.0000 | 0.0668 | (0.0574, 0.0782) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 0.2519 | (0.2159, 0.2898) | 0.0000 | 0.2519 | (0.2059, 0.3071) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_style | 0.0188 | (-0.0014, 0.0401) | 0.0340 | 0.0188 | (0.0003, 0.0410) | 0.0220 |
| controlled_alt_vs_baseline_no_context | distinct1 | -0.0403 | (-0.0477, -0.0325) | 1.0000 | -0.0403 | (-0.0510, -0.0288) | 1.0000 |
| controlled_alt_vs_baseline_no_context | length_score | 0.0429 | (-0.0384, 0.1205) | 0.1443 | 0.0429 | (-0.0917, 0.1976) | 0.3077 |
| controlled_alt_vs_baseline_no_context | sentence_score | 0.0781 | (0.0406, 0.1156) | 0.0000 | 0.0781 | (0.0156, 0.1469) | 0.0060 |
| controlled_alt_vs_baseline_no_context | bertscore_f1 | 0.0312 | (0.0186, 0.0438) | 0.0000 | 0.0312 | (0.0122, 0.0453) | 0.0007 |
| controlled_alt_vs_baseline_no_context | overall_quality | 0.1612 | (0.1483, 0.1740) | 0.0000 | 0.1612 | (0.1443, 0.1764) | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_relevance | 0.2259 | (0.2074, 0.2445) | 0.0000 | 0.2259 | (0.1951, 0.2581) | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1994 | (0.1648, 0.2339) | 0.0000 | 0.1994 | (0.1564, 0.2426) | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | naturalness | 0.0005 | (-0.0167, 0.0172) | 0.4800 | 0.0005 | (-0.0312, 0.0373) | 0.5257 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.2943 | (0.2712, 0.3177) | 0.0000 | 0.2943 | (0.2501, 0.3389) | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_overlap | 0.0663 | (0.0578, 0.0748) | 0.0000 | 0.0663 | (0.0586, 0.0745) | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.2463 | (0.2042, 0.2875) | 0.0000 | 0.2463 | (0.1957, 0.3044) | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_style | 0.0114 | (-0.0073, 0.0307) | 0.1140 | 0.0114 | (-0.0084, 0.0335) | 0.1560 |
| controlled_alt_vs_baseline_no_context_phi3_latest | distinct1 | -0.0405 | (-0.0479, -0.0333) | 1.0000 | -0.0405 | (-0.0455, -0.0355) | 1.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | length_score | 0.0461 | (-0.0319, 0.1199) | 0.1167 | 0.0461 | (-0.1024, 0.2095) | 0.2887 |
| controlled_alt_vs_baseline_no_context_phi3_latest | sentence_score | 0.0750 | (0.0375, 0.1125) | 0.0003 | 0.0750 | (0.0250, 0.1344) | 0.0017 |
| controlled_alt_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0320 | (0.0194, 0.0448) | 0.0000 | 0.0320 | (0.0067, 0.0561) | 0.0067 |
| controlled_alt_vs_baseline_no_context_phi3_latest | overall_quality | 0.1614 | (0.1476, 0.1745) | 0.0000 | 0.1614 | (0.1457, 0.1776) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2232 | (0.2040, 0.2439) | 0.0000 | 0.2232 | (0.1917, 0.2540) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.2011 | (0.1665, 0.2352) | 0.0000 | 0.2011 | (0.1432, 0.2498) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0095 | (-0.0269, 0.0073) | 0.8560 | -0.0095 | (-0.0371, 0.0224) | 0.7300 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.2901 | (0.2635, 0.3176) | 0.0000 | 0.2901 | (0.2466, 0.3343) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0670 | (0.0592, 0.0752) | 0.0000 | 0.0670 | (0.0624, 0.0725) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2478 | (0.2073, 0.2892) | 0.0000 | 0.2478 | (0.1768, 0.3074) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0141 | (-0.0082, 0.0380) | 0.0997 | 0.0141 | (-0.0049, 0.0400) | 0.1030 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0424 | (-0.0496, -0.0352) | 1.0000 | -0.0424 | (-0.0502, -0.0348) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.0074 | (-0.0703, 0.0836) | 0.4163 | 0.0074 | (-0.1202, 0.1515) | 0.4547 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0625 | (0.0250, 0.1000) | 0.0013 | 0.0625 | (-0.0125, 0.1437) | 0.0587 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0314 | (0.0193, 0.0440) | 0.0000 | 0.0314 | (0.0189, 0.0464) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1589 | (0.1448, 0.1732) | 0.0000 | 0.1589 | (0.1353, 0.1798) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.2278 | (0.2076, 0.2487) | 0.0000 | 0.2278 | (0.2026, 0.2530) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1952 | (0.1573, 0.2330) | 0.0000 | 0.1952 | (0.1350, 0.2488) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | -0.0092 | (-0.0276, 0.0091) | 0.8387 | -0.0092 | (-0.0394, 0.0232) | 0.7320 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.2969 | (0.2686, 0.3243) | 0.0000 | 0.2969 | (0.2611, 0.3342) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0666 | (0.0591, 0.0740) | 0.0000 | 0.0666 | (0.0613, 0.0721) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.2423 | (0.1946, 0.2876) | 0.0000 | 0.2423 | (0.1690, 0.3110) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | 0.0068 | (-0.0133, 0.0269) | 0.2577 | 0.0068 | (-0.0133, 0.0393) | 0.3567 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0426 | (-0.0506, -0.0345) | 1.0000 | -0.0426 | (-0.0488, -0.0359) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | 0.0107 | (-0.0705, 0.0955) | 0.3863 | 0.0107 | (-0.1384, 0.1619) | 0.4553 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.0594 | (0.0218, 0.0969) | 0.0013 | 0.0594 | (-0.0156, 0.1313) | 0.0627 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0323 | (0.0188, 0.0458) | 0.0000 | 0.0323 | (0.0131, 0.0545) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1591 | (0.1451, 0.1732) | 0.0000 | 0.1591 | (0.1346, 0.1819) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 47 | 22 | 43 | 0.6116 | 0.6812 |
| proposed_vs_candidate_no_context | persona_consistency | 19 | 26 | 67 | 0.4688 | 0.4222 |
| proposed_vs_candidate_no_context | naturalness | 32 | 37 | 43 | 0.4777 | 0.4638 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 39 | 12 | 61 | 0.6205 | 0.7647 |
| proposed_vs_candidate_no_context | context_overlap | 46 | 23 | 43 | 0.6027 | 0.6667 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 14 | 21 | 77 | 0.4688 | 0.4000 |
| proposed_vs_candidate_no_context | persona_style | 8 | 10 | 94 | 0.4911 | 0.4444 |
| proposed_vs_candidate_no_context | distinct1 | 34 | 31 | 47 | 0.5134 | 0.5231 |
| proposed_vs_candidate_no_context | length_score | 30 | 38 | 44 | 0.4643 | 0.4412 |
| proposed_vs_candidate_no_context | sentence_score | 18 | 22 | 72 | 0.4821 | 0.4500 |
| proposed_vs_candidate_no_context | bertscore_f1 | 70 | 16 | 26 | 0.7411 | 0.8140 |
| proposed_vs_candidate_no_context | overall_quality | 52 | 34 | 26 | 0.5804 | 0.6047 |
| proposed_vs_baseline_no_context | context_relevance | 52 | 58 | 2 | 0.4732 | 0.4727 |
| proposed_vs_baseline_no_context | persona_consistency | 20 | 45 | 47 | 0.3884 | 0.3077 |
| proposed_vs_baseline_no_context | naturalness | 24 | 87 | 1 | 0.2188 | 0.2162 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 31 | 28 | 53 | 0.5134 | 0.5254 |
| proposed_vs_baseline_no_context | context_overlap | 62 | 46 | 4 | 0.5714 | 0.5741 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 16 | 29 | 67 | 0.4420 | 0.3556 |
| proposed_vs_baseline_no_context | persona_style | 6 | 24 | 82 | 0.4196 | 0.2000 |
| proposed_vs_baseline_no_context | distinct1 | 15 | 84 | 13 | 0.1920 | 0.1515 |
| proposed_vs_baseline_no_context | length_score | 24 | 87 | 1 | 0.2188 | 0.2162 |
| proposed_vs_baseline_no_context | sentence_score | 11 | 47 | 54 | 0.3393 | 0.1897 |
| proposed_vs_baseline_no_context | bertscore_f1 | 64 | 48 | 0 | 0.5714 | 0.5714 |
| proposed_vs_baseline_no_context | overall_quality | 45 | 67 | 0 | 0.4018 | 0.4018 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 50 | 60 | 2 | 0.4554 | 0.4545 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | 18 | 44 | 50 | 0.3839 | 0.2903 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | 25 | 85 | 2 | 0.2321 | 0.2273 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 31 | 32 | 49 | 0.4955 | 0.4921 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 64 | 46 | 2 | 0.5804 | 0.5818 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 17 | 24 | 71 | 0.4688 | 0.4146 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | 3 | 27 | 82 | 0.3929 | 0.1000 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | 15 | 85 | 12 | 0.1875 | 0.1500 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | 28 | 81 | 3 | 0.2634 | 0.2569 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | 9 | 46 | 57 | 0.3348 | 0.1636 |
| proposed_vs_baseline_no_context_phi3_latest | bertscore_f1 | 70 | 42 | 0 | 0.6250 | 0.6250 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | 39 | 73 | 0 | 0.3482 | 0.3482 |
| controlled_vs_proposed_raw | context_relevance | 106 | 5 | 1 | 0.9509 | 0.9550 |
| controlled_vs_proposed_raw | persona_consistency | 97 | 5 | 10 | 0.9107 | 0.9510 |
| controlled_vs_proposed_raw | naturalness | 79 | 32 | 1 | 0.7098 | 0.7117 |
| controlled_vs_proposed_raw | context_keyword_coverage | 100 | 3 | 9 | 0.9330 | 0.9709 |
| controlled_vs_proposed_raw | context_overlap | 97 | 13 | 2 | 0.8750 | 0.8818 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 96 | 4 | 12 | 0.9107 | 0.9600 |
| controlled_vs_proposed_raw | persona_style | 33 | 7 | 72 | 0.6161 | 0.8250 |
| controlled_vs_proposed_raw | distinct1 | 57 | 51 | 4 | 0.5268 | 0.5278 |
| controlled_vs_proposed_raw | length_score | 76 | 34 | 2 | 0.6875 | 0.6909 |
| controlled_vs_proposed_raw | sentence_score | 60 | 4 | 48 | 0.7500 | 0.9375 |
| controlled_vs_proposed_raw | bertscore_f1 | 68 | 44 | 0 | 0.6071 | 0.6071 |
| controlled_vs_proposed_raw | overall_quality | 108 | 4 | 0 | 0.9643 | 0.9643 |
| controlled_vs_candidate_no_context | context_relevance | 112 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 94 | 7 | 11 | 0.8884 | 0.9307 |
| controlled_vs_candidate_no_context | naturalness | 76 | 36 | 0 | 0.6786 | 0.6786 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 111 | 0 | 1 | 0.9955 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 110 | 2 | 0 | 0.9821 | 0.9821 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 94 | 5 | 13 | 0.8973 | 0.9495 |
| controlled_vs_candidate_no_context | persona_style | 28 | 8 | 76 | 0.5893 | 0.7778 |
| controlled_vs_candidate_no_context | distinct1 | 59 | 51 | 2 | 0.5357 | 0.5364 |
| controlled_vs_candidate_no_context | length_score | 69 | 42 | 1 | 0.6205 | 0.6216 |
| controlled_vs_candidate_no_context | sentence_score | 59 | 7 | 46 | 0.7321 | 0.8939 |
| controlled_vs_candidate_no_context | bertscore_f1 | 85 | 27 | 0 | 0.7589 | 0.7589 |
| controlled_vs_candidate_no_context | overall_quality | 111 | 1 | 0 | 0.9911 | 0.9911 |
| controlled_vs_baseline_no_context | context_relevance | 111 | 1 | 0 | 0.9911 | 0.9911 |
| controlled_vs_baseline_no_context | persona_consistency | 93 | 7 | 12 | 0.8839 | 0.9300 |
| controlled_vs_baseline_no_context | naturalness | 58 | 53 | 1 | 0.5223 | 0.5225 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 107 | 0 | 5 | 0.9777 | 1.0000 |
| controlled_vs_baseline_no_context | context_overlap | 110 | 2 | 0 | 0.9821 | 0.9821 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 92 | 5 | 15 | 0.8884 | 0.9485 |
| controlled_vs_baseline_no_context | persona_style | 16 | 13 | 83 | 0.5134 | 0.5517 |
| controlled_vs_baseline_no_context | distinct1 | 13 | 94 | 5 | 0.1384 | 0.1215 |
| controlled_vs_baseline_no_context | length_score | 57 | 54 | 1 | 0.5134 | 0.5135 |
| controlled_vs_baseline_no_context | sentence_score | 32 | 12 | 68 | 0.5893 | 0.7273 |
| controlled_vs_baseline_no_context | bertscore_f1 | 82 | 30 | 0 | 0.7321 | 0.7321 |
| controlled_vs_baseline_no_context | overall_quality | 112 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 112 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 93 | 10 | 9 | 0.8705 | 0.9029 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | 58 | 54 | 0 | 0.5179 | 0.5179 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 107 | 0 | 5 | 0.9777 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 108 | 4 | 0 | 0.9643 | 0.9643 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 93 | 8 | 11 | 0.8795 | 0.9208 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | 11 | 10 | 91 | 0.5045 | 0.5238 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | 16 | 93 | 3 | 0.1562 | 0.1468 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | 61 | 50 | 1 | 0.5491 | 0.5495 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 31 | 12 | 69 | 0.5848 | 0.7209 |
| controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 74 | 38 | 0 | 0.6607 | 0.6607 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 110 | 2 | 0 | 0.9821 | 0.9821 |
| controlled_alt_vs_controlled_default | context_relevance | 40 | 54 | 18 | 0.4375 | 0.4255 |
| controlled_alt_vs_controlled_default | persona_consistency | 39 | 32 | 41 | 0.5312 | 0.5493 |
| controlled_alt_vs_controlled_default | naturalness | 46 | 48 | 18 | 0.4911 | 0.4894 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 28 | 35 | 49 | 0.4688 | 0.4444 |
| controlled_alt_vs_controlled_default | context_overlap | 44 | 50 | 18 | 0.4732 | 0.4681 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 32 | 27 | 53 | 0.5223 | 0.5424 |
| controlled_alt_vs_controlled_default | persona_style | 13 | 10 | 89 | 0.5134 | 0.5652 |
| controlled_alt_vs_controlled_default | distinct1 | 50 | 43 | 19 | 0.5312 | 0.5376 |
| controlled_alt_vs_controlled_default | length_score | 49 | 44 | 19 | 0.5223 | 0.5269 |
| controlled_alt_vs_controlled_default | sentence_score | 18 | 13 | 81 | 0.5223 | 0.5806 |
| controlled_alt_vs_controlled_default | bertscore_f1 | 43 | 53 | 16 | 0.4554 | 0.4479 |
| controlled_alt_vs_controlled_default | overall_quality | 50 | 46 | 16 | 0.5179 | 0.5208 |
| controlled_alt_vs_proposed_raw | context_relevance | 106 | 6 | 0 | 0.9464 | 0.9464 |
| controlled_alt_vs_proposed_raw | persona_consistency | 99 | 3 | 10 | 0.9286 | 0.9706 |
| controlled_alt_vs_proposed_raw | naturalness | 87 | 25 | 0 | 0.7768 | 0.7768 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 103 | 2 | 7 | 0.9509 | 0.9810 |
| controlled_alt_vs_proposed_raw | context_overlap | 100 | 12 | 0 | 0.8929 | 0.8929 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 99 | 3 | 10 | 0.9286 | 0.9706 |
| controlled_alt_vs_proposed_raw | persona_style | 30 | 3 | 79 | 0.6205 | 0.9091 |
| controlled_alt_vs_proposed_raw | distinct1 | 59 | 48 | 5 | 0.5491 | 0.5514 |
| controlled_alt_vs_proposed_raw | length_score | 81 | 31 | 0 | 0.7232 | 0.7232 |
| controlled_alt_vs_proposed_raw | sentence_score | 66 | 5 | 41 | 0.7723 | 0.9296 |
| controlled_alt_vs_proposed_raw | bertscore_f1 | 71 | 41 | 0 | 0.6339 | 0.6339 |
| controlled_alt_vs_proposed_raw | overall_quality | 109 | 3 | 0 | 0.9732 | 0.9732 |
| controlled_alt_vs_candidate_no_context | context_relevance | 112 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 96 | 7 | 9 | 0.8973 | 0.9320 |
| controlled_alt_vs_candidate_no_context | naturalness | 84 | 28 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 112 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 110 | 2 | 0 | 0.9821 | 0.9821 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 96 | 6 | 10 | 0.9018 | 0.9412 |
| controlled_alt_vs_candidate_no_context | persona_style | 27 | 4 | 81 | 0.6027 | 0.8710 |
| controlled_alt_vs_candidate_no_context | distinct1 | 58 | 49 | 5 | 0.5402 | 0.5421 |
| controlled_alt_vs_candidate_no_context | length_score | 73 | 38 | 1 | 0.6562 | 0.6577 |
| controlled_alt_vs_candidate_no_context | sentence_score | 64 | 7 | 41 | 0.7545 | 0.9014 |
| controlled_alt_vs_candidate_no_context | bertscore_f1 | 90 | 22 | 0 | 0.8036 | 0.8036 |
| controlled_alt_vs_candidate_no_context | overall_quality | 111 | 1 | 0 | 0.9911 | 0.9911 |
| controlled_alt_vs_baseline_no_context | context_relevance | 111 | 1 | 0 | 0.9911 | 0.9911 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 96 | 7 | 9 | 0.8973 | 0.9320 |
| controlled_alt_vs_baseline_no_context | naturalness | 59 | 53 | 0 | 0.5268 | 0.5268 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 108 | 0 | 4 | 0.9821 | 1.0000 |
| controlled_alt_vs_baseline_no_context | context_overlap | 108 | 4 | 0 | 0.9643 | 0.9643 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 96 | 6 | 10 | 0.9018 | 0.9412 |
| controlled_alt_vs_baseline_no_context | persona_style | 14 | 7 | 91 | 0.5312 | 0.6667 |
| controlled_alt_vs_baseline_no_context | distinct1 | 14 | 93 | 5 | 0.1473 | 0.1308 |
| controlled_alt_vs_baseline_no_context | length_score | 58 | 52 | 2 | 0.5268 | 0.5273 |
| controlled_alt_vs_baseline_no_context | sentence_score | 34 | 9 | 69 | 0.6116 | 0.7907 |
| controlled_alt_vs_baseline_no_context | bertscore_f1 | 79 | 33 | 0 | 0.7054 | 0.7054 |
| controlled_alt_vs_baseline_no_context | overall_quality | 112 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_relevance | 112 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_consistency | 94 | 7 | 11 | 0.8884 | 0.9307 |
| controlled_alt_vs_baseline_no_context_phi3_latest | naturalness | 61 | 51 | 0 | 0.5446 | 0.5446 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 110 | 0 | 2 | 0.9911 | 1.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_overlap | 110 | 2 | 0 | 0.9821 | 0.9821 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 93 | 6 | 13 | 0.8884 | 0.9394 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_style | 11 | 7 | 94 | 0.5179 | 0.6111 |
| controlled_alt_vs_baseline_no_context_phi3_latest | distinct1 | 18 | 87 | 7 | 0.1920 | 0.1714 |
| controlled_alt_vs_baseline_no_context_phi3_latest | length_score | 62 | 49 | 1 | 0.5580 | 0.5586 |
| controlled_alt_vs_baseline_no_context_phi3_latest | sentence_score | 34 | 10 | 68 | 0.6071 | 0.7727 |
| controlled_alt_vs_baseline_no_context_phi3_latest | bertscore_f1 | 77 | 35 | 0 | 0.6875 | 0.6875 |
| controlled_alt_vs_baseline_no_context_phi3_latest | overall_quality | 110 | 2 | 0 | 0.9821 | 0.9821 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 111 | 1 | 0 | 0.9911 | 0.9911 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 93 | 7 | 12 | 0.8839 | 0.9300 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 58 | 53 | 1 | 0.5223 | 0.5225 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 107 | 0 | 5 | 0.9777 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 110 | 2 | 0 | 0.9821 | 0.9821 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 92 | 5 | 15 | 0.8884 | 0.9485 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 16 | 13 | 83 | 0.5134 | 0.5517 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | 13 | 94 | 5 | 0.1384 | 0.1215 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 57 | 54 | 1 | 0.5134 | 0.5135 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 32 | 12 | 68 | 0.5893 | 0.7273 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 82 | 30 | 0 | 0.7321 | 0.7321 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 112 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 112 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 93 | 10 | 9 | 0.8705 | 0.9029 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | 58 | 54 | 0 | 0.5179 | 0.5179 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 107 | 0 | 5 | 0.9777 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 108 | 4 | 0 | 0.9643 | 0.9643 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 93 | 8 | 11 | 0.8795 | 0.9208 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | 11 | 10 | 91 | 0.5045 | 0.5238 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | 16 | 93 | 3 | 0.1562 | 0.1468 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | 61 | 50 | 1 | 0.5491 | 0.5495 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | 31 | 12 | 69 | 0.5848 | 0.7209 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 74 | 38 | 0 | 0.6607 | 0.6607 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 110 | 2 | 0 | 0.9821 | 0.9821 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.2679 | 0.4286 | 0.5714 |
| proposed_contextual_controlled_quality | 0.0000 | 0.0000 | 0.3661 | 0.3482 | 0.6518 |
| proposed_contextual | 0.0000 | 0.0000 | 0.5179 | 0.0000 | 0.0000 |
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
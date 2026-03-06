# Proposal Alignment Evaluation Report

- Run ID: `20260305T191804Z`
- Generated: `2026-03-05T19:22:11.898883+00:00`
- Scenarios: `artifacts\proposal_control_tuning\profile_compare_quality12_v2\20260305T191804Z\scenarios.jsonl`
- Scenario count: `12`

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
| proposed_contextual_controlled | 0.2653 (0.2228, 0.3128) | 0.3155 (0.2521, 0.3874) | 0.8613 (0.8289, 0.8925) | 0.3671 (0.3442, 0.3864) | 0.0709 |
| proposed_contextual_controlled_quality | 0.3355 (0.2974, 0.3753) | 0.3876 (0.2941, 0.4922) | 0.8788 (0.8402, 0.9127) | 0.4263 (0.3940, 0.4635) | 0.1097 |
| proposed_contextual | 0.1199 (0.0504, 0.1975) | 0.1663 (0.1074, 0.2415) | 0.8223 (0.7843, 0.8618) | 0.2492 (0.2029, 0.2995) | 0.0552 |
| candidate_no_context | 0.0261 (0.0091, 0.0575) | 0.2171 (0.1401, 0.3002) | 0.8126 (0.7788, 0.8461) | 0.2253 (0.1924, 0.2639) | 0.0420 |
| baseline_no_context | 0.0565 (0.0289, 0.0883) | 0.2006 (0.1470, 0.2625) | 0.8856 (0.8558, 0.9149) | 0.2450 (0.2191, 0.2733) | 0.0439 |
| baseline_no_context_phi3_latest | 0.0494 (0.0255, 0.0751) | 0.1782 (0.1230, 0.2393) | 0.9168 (0.8859, 0.9457) | 0.2403 (0.2181, 0.2642) | 0.0426 |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0938 | 3.5929 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0509 | -0.2342 |
| proposed_vs_candidate_no_context | naturalness | 0.0097 | 0.0120 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.1212 | 5.8182 |
| proposed_vs_candidate_no_context | context_overlap | 0.0297 | 0.7749 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0694 | -0.5932 |
| proposed_vs_candidate_no_context | persona_style | 0.0235 | 0.0381 |
| proposed_vs_candidate_no_context | distinct1 | -0.0069 | -0.0073 |
| proposed_vs_candidate_no_context | length_score | 0.0333 | 0.1132 |
| proposed_vs_candidate_no_context | sentence_score | 0.0583 | 0.0791 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0131 | 0.3119 |
| proposed_vs_candidate_no_context | overall_quality | 0.0240 | 0.1063 |
| proposed_vs_baseline_no_context | context_relevance | 0.0634 | 1.1230 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0343 | -0.1712 |
| proposed_vs_baseline_no_context | naturalness | -0.0633 | -0.0715 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0846 | 1.4725 |
| proposed_vs_baseline_no_context | context_overlap | 0.0140 | 0.2580 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0452 | -0.4872 |
| proposed_vs_baseline_no_context | persona_style | 0.0092 | 0.0146 |
| proposed_vs_baseline_no_context | distinct1 | -0.0208 | -0.0215 |
| proposed_vs_baseline_no_context | length_score | -0.2167 | -0.3980 |
| proposed_vs_baseline_no_context | sentence_score | -0.1167 | -0.1279 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0113 | 0.2563 |
| proposed_vs_baseline_no_context | overall_quality | 0.0042 | 0.0172 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 0.0704 | 1.4243 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | -0.0119 | -0.0670 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | -0.0945 | -0.1031 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.0928 | 1.8846 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 0.0182 | 0.3647 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | -0.0167 | -0.2593 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | 0.0069 | 0.0110 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | -0.0220 | -0.0228 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | -0.3556 | -0.5203 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | -0.1458 | -0.1549 |
| proposed_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0125 | 0.2941 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | 0.0090 | 0.0374 |
| controlled_vs_proposed_raw | context_relevance | 0.1455 | 1.2136 |
| controlled_vs_proposed_raw | persona_consistency | 0.1493 | 0.8976 |
| controlled_vs_proposed_raw | naturalness | 0.0390 | 0.0474 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.1970 | 1.3867 |
| controlled_vs_proposed_raw | context_overlap | 0.0253 | 0.3718 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1841 | 3.8667 |
| controlled_vs_proposed_raw | persona_style | 0.0098 | 0.0152 |
| controlled_vs_proposed_raw | distinct1 | -0.0209 | -0.0222 |
| controlled_vs_proposed_raw | length_score | 0.1639 | 0.5000 |
| controlled_vs_proposed_raw | sentence_score | 0.1458 | 0.1832 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0157 | 0.2846 |
| controlled_vs_proposed_raw | overall_quality | 0.1178 | 0.4727 |
| controlled_vs_candidate_no_context | context_relevance | 0.2392 | 9.1670 |
| controlled_vs_candidate_no_context | persona_consistency | 0.0984 | 0.4532 |
| controlled_vs_candidate_no_context | naturalness | 0.0487 | 0.0600 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3182 | 15.2727 |
| controlled_vs_candidate_no_context | context_overlap | 0.0551 | 1.4348 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1147 | 0.9797 |
| controlled_vs_candidate_no_context | persona_style | 0.0333 | 0.0539 |
| controlled_vs_candidate_no_context | distinct1 | -0.0278 | -0.0293 |
| controlled_vs_candidate_no_context | length_score | 0.1972 | 0.6698 |
| controlled_vs_candidate_no_context | sentence_score | 0.2042 | 0.2768 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0288 | 0.6853 |
| controlled_vs_candidate_no_context | overall_quality | 0.1418 | 0.6292 |
| controlled_vs_baseline_no_context | context_relevance | 0.2089 | 3.6996 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1149 | 0.5727 |
| controlled_vs_baseline_no_context | naturalness | -0.0243 | -0.0274 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.2816 | 4.9011 |
| controlled_vs_baseline_no_context | context_overlap | 0.0393 | 0.7257 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1389 | 1.4957 |
| controlled_vs_baseline_no_context | persona_style | 0.0190 | 0.0300 |
| controlled_vs_baseline_no_context | distinct1 | -0.0417 | -0.0432 |
| controlled_vs_baseline_no_context | length_score | -0.0528 | -0.0969 |
| controlled_vs_baseline_no_context | sentence_score | 0.0292 | 0.0320 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0270 | 0.6139 |
| controlled_vs_baseline_no_context | overall_quality | 0.1220 | 0.4980 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.2159 | 4.3664 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1373 | 0.7704 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | -0.0555 | -0.0605 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.2898 | 5.8846 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0435 | 0.8721 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1675 | 2.6049 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | 0.0167 | 0.0264 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0429 | -0.0445 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | -0.1917 | -0.2805 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.0000 | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0282 | 0.6625 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1268 | 0.5278 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0702 | 0.2644 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0721 | 0.2284 |
| controlled_alt_vs_controlled_default | naturalness | 0.0175 | 0.0203 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0795 | 0.2346 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0482 | 0.5161 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0909 | 0.3921 |
| controlled_alt_vs_controlled_default | persona_style | -0.0031 | -0.0048 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0190 | 0.0207 |
| controlled_alt_vs_controlled_default | length_score | 0.0639 | 0.1299 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0292 | -0.0310 |
| controlled_alt_vs_controlled_default | bertscore_f1 | 0.0389 | 0.5483 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0592 | 0.1613 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.2156 | 1.7989 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.2213 | 1.3310 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0565 | 0.0687 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2765 | 1.9467 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0736 | 1.0798 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.2750 | 5.7750 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0066 | 0.0104 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0019 | -0.0020 |
| controlled_alt_vs_proposed_raw | length_score | 0.2278 | 0.6949 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1167 | 0.1466 |
| controlled_alt_vs_proposed_raw | bertscore_f1 | 0.0546 | 0.9890 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1770 | 0.7103 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.3094 | 11.8550 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1705 | 0.7851 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0662 | 0.0815 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3977 | 19.0909 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.1033 | 2.6915 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.2056 | 1.7559 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0301 | 0.0488 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0088 | -0.0092 |
| controlled_alt_vs_candidate_no_context | length_score | 0.2611 | 0.8868 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1750 | 0.2373 |
| controlled_alt_vs_candidate_no_context | bertscore_f1 | 0.0677 | 1.6094 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.2010 | 0.8921 |
| controlled_alt_vs_baseline_no_context | context_relevance | 0.2790 | 4.9421 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 0.1870 | 0.9320 |
| controlled_alt_vs_baseline_no_context | naturalness | -0.0068 | -0.0077 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 0.3611 | 6.2857 |
| controlled_alt_vs_baseline_no_context | context_overlap | 0.0875 | 1.6165 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 0.2298 | 2.4744 |
| controlled_alt_vs_baseline_no_context | persona_style | 0.0159 | 0.0251 |
| controlled_alt_vs_baseline_no_context | distinct1 | -0.0226 | -0.0235 |
| controlled_alt_vs_baseline_no_context | length_score | 0.0111 | 0.0204 |
| controlled_alt_vs_baseline_no_context | sentence_score | 0.0000 | 0.0000 |
| controlled_alt_vs_baseline_no_context | bertscore_f1 | 0.0658 | 1.4989 |
| controlled_alt_vs_baseline_no_context | overall_quality | 0.1812 | 0.7397 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_relevance | 0.2861 | 5.7852 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_consistency | 0.2094 | 1.1748 |
| controlled_alt_vs_baseline_no_context_phi3_latest | naturalness | -0.0380 | -0.0415 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.3693 | 7.5000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_overlap | 0.0918 | 1.8385 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.2583 | 4.0185 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_style | 0.0136 | 0.0214 |
| controlled_alt_vs_baseline_no_context_phi3_latest | distinct1 | -0.0238 | -0.0247 |
| controlled_alt_vs_baseline_no_context_phi3_latest | length_score | -0.1278 | -0.1870 |
| controlled_alt_vs_baseline_no_context_phi3_latest | sentence_score | -0.0292 | -0.0310 |
| controlled_alt_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0671 | 1.5740 |
| controlled_alt_vs_baseline_no_context_phi3_latest | overall_quality | 0.1860 | 0.7743 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2089 | 3.6996 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1149 | 0.5727 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0243 | -0.0274 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.2816 | 4.9011 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0393 | 0.7257 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1389 | 1.4957 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0190 | 0.0300 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0417 | -0.0432 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | -0.0528 | -0.0969 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0292 | 0.0320 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0270 | 0.6139 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1220 | 0.4980 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.2159 | 4.3664 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1373 | 0.7704 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | -0.0555 | -0.0605 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.2898 | 5.8846 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0435 | 0.8721 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1675 | 2.6049 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | 0.0167 | 0.0264 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0429 | -0.0445 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | -0.1917 | -0.2805 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.0000 | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0282 | 0.6625 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1268 | 0.5278 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0938 | (0.0140, 0.1782) | 0.0087 | 0.0938 | (0.0148, 0.1556) | 0.0150 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0509 | (-0.1175, 0.0027) | 0.9683 | -0.0509 | (-0.0918, 0.0000) | 0.9783 |
| proposed_vs_candidate_no_context | naturalness | 0.0097 | (-0.0299, 0.0469) | 0.3033 | 0.0097 | (-0.0500, 0.0499) | 0.3827 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.1212 | (0.0177, 0.2342) | 0.0100 | 0.1212 | (0.0182, 0.1989) | 0.0240 |
| proposed_vs_candidate_no_context | context_overlap | 0.0297 | (0.0041, 0.0588) | 0.0103 | 0.0297 | (0.0018, 0.0545) | 0.0240 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0694 | (-0.1389, 0.0000) | 1.0000 | -0.0694 | (-0.1167, -0.0166) | 1.0000 |
| proposed_vs_candidate_no_context | persona_style | 0.0235 | (-0.0098, 0.0720) | 0.1563 | 0.0235 | (0.0000, 0.0833) | 0.0920 |
| proposed_vs_candidate_no_context | distinct1 | -0.0069 | (-0.0273, 0.0140) | 0.7230 | -0.0069 | (-0.0197, 0.0017) | 0.9367 |
| proposed_vs_candidate_no_context | length_score | 0.0333 | (-0.1306, 0.1750) | 0.3113 | 0.0333 | (-0.2001, 0.1800) | 0.3870 |
| proposed_vs_candidate_no_context | sentence_score | 0.0583 | (-0.0583, 0.1750) | 0.2073 | 0.0583 | (-0.0875, 0.1853) | 0.3777 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0131 | (-0.0150, 0.0434) | 0.1853 | 0.0131 | (-0.0178, 0.0317) | 0.1997 |
| proposed_vs_candidate_no_context | overall_quality | 0.0240 | (-0.0211, 0.0712) | 0.1523 | 0.0240 | (-0.0184, 0.0566) | 0.1847 |
| proposed_vs_baseline_no_context | context_relevance | 0.0634 | (0.0027, 0.1261) | 0.0213 | 0.0634 | (-0.0343, 0.1239) | 0.1173 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0343 | (-0.0846, 0.0066) | 0.9390 | -0.0343 | (-0.1010, 0.0035) | 0.9547 |
| proposed_vs_baseline_no_context | naturalness | -0.0633 | (-0.1079, -0.0208) | 1.0000 | -0.0633 | (-0.1204, -0.0218) | 1.0000 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0846 | (-0.0025, 0.1667) | 0.0293 | 0.0846 | (-0.0455, 0.1631) | 0.1117 |
| proposed_vs_baseline_no_context | context_overlap | 0.0140 | (-0.0065, 0.0378) | 0.1090 | 0.0140 | (-0.0080, 0.0333) | 0.2160 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0452 | (-0.0952, 0.0000) | 1.0000 | -0.0452 | (-0.1107, -0.0089) | 1.0000 |
| proposed_vs_baseline_no_context | persona_style | 0.0092 | (-0.0692, 0.0807) | 0.4047 | 0.0092 | (-0.0938, 0.0787) | 0.4093 |
| proposed_vs_baseline_no_context | distinct1 | -0.0208 | (-0.0456, 0.0019) | 0.9647 | -0.0208 | (-0.0528, 0.0053) | 0.9323 |
| proposed_vs_baseline_no_context | length_score | -0.2167 | (-0.3750, -0.0722) | 0.9993 | -0.2167 | (-0.4074, -0.0846) | 1.0000 |
| proposed_vs_baseline_no_context | sentence_score | -0.1167 | (-0.2042, -0.0292) | 1.0000 | -0.1167 | (-0.2333, -0.0333) | 1.0000 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0113 | (-0.0243, 0.0483) | 0.2727 | 0.0113 | (-0.0502, 0.0526) | 0.3743 |
| proposed_vs_baseline_no_context | overall_quality | 0.0042 | (-0.0438, 0.0469) | 0.4207 | 0.0042 | (-0.0721, 0.0500) | 0.4850 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 0.0704 | (0.0144, 0.1290) | 0.0050 | 0.0704 | (-0.0021, 0.1201) | 0.0353 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | -0.0119 | (-0.0428, 0.0097) | 0.7800 | -0.0119 | (-0.0436, 0.0107) | 0.8013 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | -0.0945 | (-0.1407, -0.0471) | 1.0000 | -0.0945 | (-0.1492, -0.0459) | 0.9997 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.0928 | (0.0215, 0.1642) | 0.0050 | 0.0928 | (-0.0091, 0.1573) | 0.0427 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 0.0182 | (-0.0014, 0.0411) | 0.0337 | 0.0182 | (-0.0070, 0.0408) | 0.1933 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | -0.0167 | (-0.0500, 0.0000) | 1.0000 | -0.0167 | (-0.0545, 0.0000) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | 0.0069 | (-0.0425, 0.0625) | 0.4320 | 0.0069 | (-0.0455, 0.0729) | 0.4583 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | -0.0220 | (-0.0404, -0.0049) | 0.9963 | -0.0220 | (-0.0395, -0.0049) | 0.9933 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | -0.3556 | (-0.5028, -0.2028) | 1.0000 | -0.3556 | (-0.5111, -0.2000) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | -0.1458 | (-0.2625, -0.0292) | 0.9920 | -0.1458 | (-0.3208, 0.0000) | 0.9780 |
| proposed_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0125 | (-0.0097, 0.0359) | 0.1370 | 0.0125 | (-0.0182, 0.0335) | 0.2473 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | 0.0090 | (-0.0242, 0.0421) | 0.3137 | 0.0090 | (-0.0366, 0.0421) | 0.3730 |
| controlled_vs_proposed_raw | context_relevance | 0.1455 | (0.0563, 0.2348) | 0.0003 | 0.1455 | (0.0525, 0.2808) | 0.0007 |
| controlled_vs_proposed_raw | persona_consistency | 0.1493 | (0.0948, 0.2145) | 0.0000 | 0.1493 | (0.0873, 0.1941) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0390 | (-0.0161, 0.0952) | 0.0933 | 0.0390 | (-0.0336, 0.1300) | 0.1787 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.1970 | (0.0745, 0.3106) | 0.0007 | 0.1970 | (0.0738, 0.3766) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0253 | (0.0015, 0.0476) | 0.0220 | 0.0253 | (0.0038, 0.0578) | 0.0087 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1841 | (0.1155, 0.2587) | 0.0000 | 0.1841 | (0.1095, 0.2388) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.0098 | (-0.0500, 0.0833) | 0.4150 | 0.0098 | (-0.0319, 0.1028) | 0.4247 |
| controlled_vs_proposed_raw | distinct1 | -0.0209 | (-0.0452, 0.0014) | 0.9643 | -0.0209 | (-0.0368, -0.0024) | 0.9863 |
| controlled_vs_proposed_raw | length_score | 0.1639 | (-0.0612, 0.3944) | 0.0830 | 0.1639 | (-0.1333, 0.5273) | 0.1493 |
| controlled_vs_proposed_raw | sentence_score | 0.1458 | (0.0292, 0.2625) | 0.0233 | 0.1458 | (0.0000, 0.3208) | 0.0530 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0157 | (-0.0112, 0.0493) | 0.1523 | 0.0157 | (-0.0045, 0.0637) | 0.1077 |
| controlled_vs_proposed_raw | overall_quality | 0.1178 | (0.0680, 0.1698) | 0.0000 | 0.1178 | (0.0723, 0.1860) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2392 | (0.1771, 0.2979) | 0.0000 | 0.2392 | (0.1997, 0.3075) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.0984 | (0.0416, 0.1419) | 0.0000 | 0.0984 | (0.0493, 0.1396) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.0487 | (0.0047, 0.0971) | 0.0133 | 0.0487 | (-0.0082, 0.1094) | 0.0510 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3182 | (0.2330, 0.3958) | 0.0000 | 0.3182 | (0.2645, 0.4091) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0551 | (0.0419, 0.0698) | 0.0000 | 0.0551 | (0.0429, 0.0672) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1147 | (0.0472, 0.1675) | 0.0010 | 0.1147 | (0.0503, 0.1605) | 0.0003 |
| controlled_vs_candidate_no_context | persona_style | 0.0333 | (-0.0278, 0.1069) | 0.1830 | 0.0333 | (-0.0230, 0.1391) | 0.1743 |
| controlled_vs_candidate_no_context | distinct1 | -0.0278 | (-0.0481, -0.0080) | 0.9970 | -0.0278 | (-0.0501, -0.0050) | 0.9940 |
| controlled_vs_candidate_no_context | length_score | 0.1972 | (0.0389, 0.3833) | 0.0043 | 0.1972 | (-0.0190, 0.4424) | 0.0520 |
| controlled_vs_candidate_no_context | sentence_score | 0.2042 | (0.1167, 0.2917) | 0.0000 | 0.2042 | (0.1250, 0.3000) | 0.0000 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0288 | (-0.0039, 0.0642) | 0.0467 | 0.0288 | (-0.0020, 0.0659) | 0.0380 |
| controlled_vs_candidate_no_context | overall_quality | 0.1418 | (0.1090, 0.1764) | 0.0000 | 0.1418 | (0.1193, 0.1774) | 0.0000 |
| controlled_vs_baseline_no_context | context_relevance | 0.2089 | (0.1671, 0.2436) | 0.0000 | 0.2089 | (0.1742, 0.2590) | 0.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1149 | (0.0372, 0.1983) | 0.0020 | 0.1149 | (0.0242, 0.1774) | 0.0017 |
| controlled_vs_baseline_no_context | naturalness | -0.0243 | (-0.0681, 0.0216) | 0.8470 | -0.0243 | (-0.0749, 0.0326) | 0.7947 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.2816 | (0.2285, 0.3289) | 0.0000 | 0.2816 | (0.2298, 0.3485) | 0.0000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0393 | (0.0194, 0.0611) | 0.0000 | 0.0393 | (0.0281, 0.0611) | 0.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1389 | (0.0389, 0.2361) | 0.0027 | 0.1389 | (0.0302, 0.2157) | 0.0223 |
| controlled_vs_baseline_no_context | persona_style | 0.0190 | (-0.0458, 0.0838) | 0.2860 | 0.0190 | (-0.0187, 0.0750) | 0.2100 |
| controlled_vs_baseline_no_context | distinct1 | -0.0417 | (-0.0587, -0.0207) | 1.0000 | -0.0417 | (-0.0620, -0.0235) | 1.0000 |
| controlled_vs_baseline_no_context | length_score | -0.0528 | (-0.2556, 0.1639) | 0.6990 | -0.0528 | (-0.3030, 0.2250) | 0.6527 |
| controlled_vs_baseline_no_context | sentence_score | 0.0292 | (-0.0875, 0.1458) | 0.3920 | 0.0292 | (-0.0955, 0.1750) | 0.4080 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0270 | (0.0021, 0.0517) | 0.0143 | 0.0270 | (-0.0158, 0.0571) | 0.1540 |
| controlled_vs_baseline_no_context | overall_quality | 0.1220 | (0.0906, 0.1536) | 0.0000 | 0.1220 | (0.0930, 0.1468) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.2159 | (0.1683, 0.2663) | 0.0000 | 0.2159 | (0.1673, 0.2939) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1373 | (0.0819, 0.2027) | 0.0000 | 0.1373 | (0.0658, 0.1898) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | -0.0555 | (-0.0992, -0.0143) | 0.9967 | -0.0555 | (-0.0941, -0.0062) | 0.9923 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.2898 | (0.2222, 0.3561) | 0.0000 | 0.2898 | (0.2208, 0.4001) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0435 | (0.0290, 0.0611) | 0.0000 | 0.0435 | (0.0295, 0.0668) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1675 | (0.0952, 0.2480) | 0.0000 | 0.1675 | (0.0680, 0.2359) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | 0.0167 | (-0.0409, 0.0813) | 0.2987 | 0.0167 | (-0.0270, 0.1052) | 0.2793 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0429 | (-0.0682, -0.0150) | 0.9980 | -0.0429 | (-0.0627, -0.0195) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | -0.1917 | (-0.3778, -0.0139) | 0.9817 | -0.1917 | (-0.3758, 0.0389) | 0.9277 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.0000 | (-0.0875, 0.0875) | 0.6517 | 0.0000 | (-0.0955, 0.0955) | 0.6380 |
| controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0282 | (-0.0004, 0.0651) | 0.0280 | 0.0282 | (-0.0083, 0.0644) | 0.0620 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1268 | (0.0972, 0.1563) | 0.0000 | 0.1268 | (0.0993, 0.1694) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0702 | (0.0121, 0.1231) | 0.0080 | 0.0702 | (0.0224, 0.0994) | 0.0020 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0721 | (-0.0592, 0.2236) | 0.1587 | 0.0721 | (-0.0748, 0.2801) | 0.2617 |
| controlled_alt_vs_controlled_default | naturalness | 0.0175 | (-0.0209, 0.0563) | 0.1963 | 0.0175 | (-0.0011, 0.0434) | 0.0343 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0795 | (0.0025, 0.1541) | 0.0183 | 0.0795 | (0.0202, 0.1174) | 0.0097 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0482 | (0.0209, 0.0833) | 0.0000 | 0.0482 | (0.0187, 0.0930) | 0.0003 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0909 | (-0.0754, 0.2742) | 0.1640 | 0.0909 | (-0.0801, 0.3772) | 0.3063 |
| controlled_alt_vs_controlled_default | persona_style | -0.0031 | (-0.0573, 0.0500) | 0.5200 | -0.0031 | (-0.0750, 0.0304) | 0.5173 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0190 | (-0.0047, 0.0456) | 0.0620 | 0.0190 | (0.0035, 0.0439) | 0.0067 |
| controlled_alt_vs_controlled_default | length_score | 0.0639 | (-0.1084, 0.2362) | 0.2250 | 0.0639 | (-0.0501, 0.2223) | 0.1257 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0292 | (-0.1167, 0.0583) | 0.8117 | -0.0292 | (-0.1556, 0.0500) | 0.8160 |
| controlled_alt_vs_controlled_default | bertscore_f1 | 0.0389 | (0.0059, 0.0761) | 0.0087 | 0.0389 | (0.0158, 0.0574) | 0.0030 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0592 | (0.0110, 0.1127) | 0.0070 | 0.0592 | (0.0061, 0.1322) | 0.0100 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.2156 | (0.1172, 0.3015) | 0.0000 | 0.2156 | (0.1350, 0.3366) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.2213 | (0.0944, 0.3511) | 0.0003 | 0.2213 | (0.0645, 0.4213) | 0.0040 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0565 | (-0.0088, 0.1141) | 0.0397 | 0.0565 | (-0.0107, 0.1337) | 0.0590 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2765 | (0.1515, 0.3914) | 0.0000 | 0.2765 | (0.1748, 0.4242) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0736 | (0.0366, 0.1144) | 0.0000 | 0.0736 | (0.0350, 0.1316) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.2750 | (0.1230, 0.4333) | 0.0003 | 0.2750 | (0.0843, 0.5058) | 0.0033 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0066 | (-0.0628, 0.0833) | 0.4503 | 0.0066 | (-0.0625, 0.0938) | 0.4617 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0019 | (-0.0308, 0.0256) | 0.5333 | -0.0019 | (-0.0222, 0.0269) | 0.5413 |
| controlled_alt_vs_proposed_raw | length_score | 0.2278 | (-0.0250, 0.4556) | 0.0353 | 0.2278 | (-0.0314, 0.5300) | 0.0550 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1167 | (0.0292, 0.2042) | 0.0087 | 0.1167 | (0.0206, 0.2545) | 0.0190 |
| controlled_alt_vs_proposed_raw | bertscore_f1 | 0.0546 | (0.0170, 0.0876) | 0.0023 | 0.0546 | (0.0350, 0.0830) | 0.0000 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1770 | (0.1073, 0.2467) | 0.0000 | 0.1770 | (0.0944, 0.2853) | 0.0003 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.3094 | (0.2591, 0.3593) | 0.0000 | 0.3094 | (0.2827, 0.3463) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1705 | (0.0303, 0.3259) | 0.0093 | 0.1705 | (0.0146, 0.3946) | 0.0163 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0662 | (0.0130, 0.1226) | 0.0070 | 0.0662 | (0.0243, 0.1160) | 0.0010 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3977 | (0.3314, 0.4609) | 0.0000 | 0.3977 | (0.3598, 0.4500) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.1033 | (0.0789, 0.1336) | 0.0000 | 0.1033 | (0.0806, 0.1375) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.2056 | (0.0317, 0.3877) | 0.0093 | 0.2056 | (0.0073, 0.4821) | 0.0210 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0301 | (-0.0104, 0.0933) | 0.1683 | 0.0301 | (0.0000, 0.1111) | 0.1040 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0088 | (-0.0386, 0.0209) | 0.7150 | -0.0088 | (-0.0318, 0.0227) | 0.6957 |
| controlled_alt_vs_candidate_no_context | length_score | 0.2611 | (0.0667, 0.4694) | 0.0047 | 0.2611 | (0.0824, 0.4606) | 0.0023 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1750 | (0.0583, 0.2625) | 0.0000 | 0.1750 | (0.0389, 0.2722) | 0.0190 |
| controlled_alt_vs_candidate_no_context | bertscore_f1 | 0.0677 | (0.0298, 0.1081) | 0.0000 | 0.0677 | (0.0403, 0.0895) | 0.0000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.2010 | (0.1351, 0.2614) | 0.0000 | 0.2010 | (0.1403, 0.2771) | 0.0000 |
| controlled_alt_vs_baseline_no_context | context_relevance | 0.2790 | (0.2276, 0.3291) | 0.0000 | 0.2790 | (0.2499, 0.3208) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 0.1870 | (0.0659, 0.3079) | 0.0010 | 0.1870 | (0.0286, 0.3421) | 0.0110 |
| controlled_alt_vs_baseline_no_context | naturalness | -0.0068 | (-0.0512, 0.0375) | 0.5920 | -0.0068 | (-0.0434, 0.0411) | 0.6313 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 0.3611 | (0.2967, 0.4230) | 0.0000 | 0.3611 | (0.3196, 0.4156) | 0.0000 |
| controlled_alt_vs_baseline_no_context | context_overlap | 0.0875 | (0.0540, 0.1219) | 0.0000 | 0.0875 | (0.0637, 0.1272) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 0.2298 | (0.0873, 0.3869) | 0.0020 | 0.2298 | (0.0298, 0.4277) | 0.0143 |
| controlled_alt_vs_baseline_no_context | persona_style | 0.0159 | (-0.0125, 0.0580) | 0.2363 | 0.0159 | (0.0000, 0.0336) | 0.3357 |
| controlled_alt_vs_baseline_no_context | distinct1 | -0.0226 | (-0.0483, 0.0039) | 0.9537 | -0.0226 | (-0.0339, -0.0068) | 0.9947 |
| controlled_alt_vs_baseline_no_context | length_score | 0.0111 | (-0.1833, 0.2056) | 0.4633 | 0.0111 | (-0.1708, 0.2633) | 0.4690 |
| controlled_alt_vs_baseline_no_context | sentence_score | 0.0000 | (-0.1167, 0.1167) | 0.5977 | 0.0000 | (-0.1000, 0.0778) | 0.6523 |
| controlled_alt_vs_baseline_no_context | bertscore_f1 | 0.0658 | (0.0255, 0.1113) | 0.0003 | 0.0658 | (0.0146, 0.0994) | 0.0007 |
| controlled_alt_vs_baseline_no_context | overall_quality | 0.1812 | (0.1303, 0.2322) | 0.0000 | 0.1812 | (0.1146, 0.2447) | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_relevance | 0.2861 | (0.2372, 0.3348) | 0.0000 | 0.2861 | (0.2546, 0.3366) | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_consistency | 0.2094 | (0.0949, 0.3209) | 0.0000 | 0.2094 | (0.0669, 0.3785) | 0.0037 |
| controlled_alt_vs_baseline_no_context_phi3_latest | naturalness | -0.0380 | (-0.0943, 0.0088) | 0.9363 | -0.0380 | (-0.0727, 0.0068) | 0.9337 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.3693 | (0.3049, 0.4312) | 0.0000 | 0.3693 | (0.3279, 0.4432) | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_overlap | 0.0918 | (0.0667, 0.1213) | 0.0000 | 0.0918 | (0.0679, 0.1302) | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.2583 | (0.1190, 0.4016) | 0.0000 | 0.2583 | (0.0766, 0.4656) | 0.0037 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_style | 0.0136 | (-0.0107, 0.0466) | 0.2313 | 0.0136 | (-0.0006, 0.0556) | 0.3327 |
| controlled_alt_vs_baseline_no_context_phi3_latest | distinct1 | -0.0238 | (-0.0476, -0.0027) | 0.9863 | -0.0238 | (-0.0374, -0.0022) | 0.9893 |
| controlled_alt_vs_baseline_no_context_phi3_latest | length_score | -0.1278 | (-0.3389, 0.0694) | 0.8753 | -0.1278 | (-0.3118, 0.1074) | 0.7880 |
| controlled_alt_vs_baseline_no_context_phi3_latest | sentence_score | -0.0292 | (-0.1167, 0.0583) | 0.8187 | -0.0292 | (-0.1556, 0.0525) | 0.8037 |
| controlled_alt_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0671 | (0.0311, 0.1070) | 0.0000 | 0.0671 | (0.0335, 0.0924) | 0.0003 |
| controlled_alt_vs_baseline_no_context_phi3_latest | overall_quality | 0.1860 | (0.1376, 0.2307) | 0.0000 | 0.1860 | (0.1295, 0.2542) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2089 | (0.1687, 0.2439) | 0.0000 | 0.2089 | (0.1729, 0.2589) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1149 | (0.0339, 0.2011) | 0.0020 | 0.1149 | (0.0236, 0.1774) | 0.0037 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0243 | (-0.0669, 0.0218) | 0.8547 | -0.0243 | (-0.0761, 0.0333) | 0.7967 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.2816 | (0.2260, 0.3295) | 0.0000 | 0.2816 | (0.2328, 0.3471) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0393 | (0.0188, 0.0604) | 0.0000 | 0.0393 | (0.0282, 0.0631) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1389 | (0.0389, 0.2361) | 0.0010 | 0.1389 | (0.0303, 0.2157) | 0.0203 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0190 | (-0.0427, 0.0820) | 0.2887 | 0.0190 | (-0.0167, 0.0751) | 0.2113 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0417 | (-0.0583, -0.0213) | 1.0000 | -0.0417 | (-0.0617, -0.0242) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | -0.0528 | (-0.2583, 0.1556) | 0.6897 | -0.0528 | (-0.3001, 0.2273) | 0.6767 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0292 | (-0.0875, 0.1458) | 0.4027 | 0.0292 | (-0.0955, 0.1750) | 0.4063 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0270 | (0.0016, 0.0537) | 0.0200 | 0.0270 | (-0.0159, 0.0571) | 0.1617 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1220 | (0.0893, 0.1548) | 0.0000 | 0.1220 | (0.0931, 0.1472) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.2159 | (0.1673, 0.2666) | 0.0000 | 0.2159 | (0.1698, 0.2957) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1373 | (0.0820, 0.2019) | 0.0000 | 0.1373 | (0.0649, 0.1918) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | -0.0555 | (-0.0985, -0.0147) | 0.9970 | -0.0555 | (-0.0951, -0.0077) | 0.9950 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.2898 | (0.2222, 0.3586) | 0.0000 | 0.2898 | (0.2237, 0.3896) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0435 | (0.0286, 0.0618) | 0.0000 | 0.0435 | (0.0293, 0.0661) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1675 | (0.0960, 0.2488) | 0.0000 | 0.1675 | (0.0676, 0.2381) | 0.0003 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | 0.0167 | (-0.0382, 0.0793) | 0.2863 | 0.0167 | (-0.0270, 0.1019) | 0.2897 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0429 | (-0.0675, -0.0146) | 0.9977 | -0.0429 | (-0.0634, -0.0229) | 0.9987 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | -0.1917 | (-0.3806, -0.0111) | 0.9823 | -0.1917 | (-0.3800, 0.0433) | 0.9223 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.0000 | (-0.0875, 0.0875) | 0.6497 | 0.0000 | (-0.0955, 0.0955) | 0.6507 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0282 | (-0.0000, 0.0659) | 0.0253 | 0.0282 | (-0.0083, 0.0679) | 0.0680 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1268 | (0.0972, 0.1553) | 0.0000 | 0.1268 | (0.0992, 0.1680) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 5 | 3 | 4 | 0.5833 | 0.6250 |
| proposed_vs_candidate_no_context | persona_consistency | 2 | 4 | 6 | 0.4167 | 0.3333 |
| proposed_vs_candidate_no_context | naturalness | 6 | 2 | 4 | 0.6667 | 0.7500 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 5 | 1 | 6 | 0.6667 | 0.8333 |
| proposed_vs_candidate_no_context | context_overlap | 5 | 3 | 4 | 0.5833 | 0.6250 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0 | 3 | 9 | 0.3750 | 0.0000 |
| proposed_vs_candidate_no_context | persona_style | 2 | 1 | 9 | 0.5417 | 0.6667 |
| proposed_vs_candidate_no_context | distinct1 | 3 | 4 | 5 | 0.4583 | 0.4286 |
| proposed_vs_candidate_no_context | length_score | 6 | 2 | 4 | 0.6667 | 0.7500 |
| proposed_vs_candidate_no_context | sentence_score | 3 | 1 | 8 | 0.5833 | 0.7500 |
| proposed_vs_candidate_no_context | bertscore_f1 | 7 | 3 | 2 | 0.6667 | 0.7000 |
| proposed_vs_candidate_no_context | overall_quality | 5 | 5 | 2 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context | context_relevance | 7 | 5 | 0 | 0.5833 | 0.5833 |
| proposed_vs_baseline_no_context | persona_consistency | 3 | 4 | 5 | 0.4583 | 0.4286 |
| proposed_vs_baseline_no_context | naturalness | 4 | 8 | 0 | 0.3333 | 0.3333 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 6 | 2 | 4 | 0.6667 | 0.7500 |
| proposed_vs_baseline_no_context | context_overlap | 6 | 6 | 0 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 0 | 3 | 9 | 0.3750 | 0.0000 |
| proposed_vs_baseline_no_context | persona_style | 3 | 2 | 7 | 0.5417 | 0.6000 |
| proposed_vs_baseline_no_context | distinct1 | 4 | 5 | 3 | 0.4583 | 0.4444 |
| proposed_vs_baseline_no_context | length_score | 3 | 9 | 0 | 0.2500 | 0.2500 |
| proposed_vs_baseline_no_context | sentence_score | 0 | 4 | 8 | 0.3333 | 0.0000 |
| proposed_vs_baseline_no_context | bertscore_f1 | 6 | 6 | 0 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context | overall_quality | 6 | 6 | 0 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 6 | 6 | 0 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | 2 | 3 | 7 | 0.4583 | 0.4000 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | 3 | 9 | 0 | 0.2500 | 0.2500 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 5 | 1 | 6 | 0.6667 | 0.8333 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 6 | 6 | 0 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0 | 1 | 11 | 0.4583 | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | 2 | 2 | 8 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | 2 | 7 | 3 | 0.2917 | 0.2222 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | 2 | 10 | 0 | 0.1667 | 0.1667 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | 1 | 6 | 5 | 0.2917 | 0.1429 |
| proposed_vs_baseline_no_context_phi3_latest | bertscore_f1 | 6 | 6 | 0 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | 5 | 7 | 0 | 0.4167 | 0.4167 |
| controlled_vs_proposed_raw | context_relevance | 9 | 3 | 0 | 0.7500 | 0.7500 |
| controlled_vs_proposed_raw | persona_consistency | 10 | 0 | 2 | 0.9167 | 1.0000 |
| controlled_vs_proposed_raw | naturalness | 7 | 5 | 0 | 0.5833 | 0.5833 |
| controlled_vs_proposed_raw | context_keyword_coverage | 8 | 1 | 3 | 0.7917 | 0.8889 |
| controlled_vs_proposed_raw | context_overlap | 9 | 3 | 0 | 0.7500 | 0.7500 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 10 | 0 | 2 | 0.9167 | 1.0000 |
| controlled_vs_proposed_raw | persona_style | 2 | 2 | 8 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | distinct1 | 4 | 8 | 0 | 0.3333 | 0.3333 |
| controlled_vs_proposed_raw | length_score | 7 | 5 | 0 | 0.5833 | 0.5833 |
| controlled_vs_proposed_raw | sentence_score | 6 | 1 | 5 | 0.7083 | 0.8571 |
| controlled_vs_proposed_raw | bertscore_f1 | 7 | 5 | 0 | 0.5833 | 0.5833 |
| controlled_vs_proposed_raw | overall_quality | 10 | 2 | 0 | 0.8333 | 0.8333 |
| controlled_vs_candidate_no_context | context_relevance | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 10 | 1 | 1 | 0.8750 | 0.9091 |
| controlled_vs_candidate_no_context | naturalness | 9 | 3 | 0 | 0.7500 | 0.7500 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 11 | 0 | 1 | 0.9583 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 9 | 1 | 2 | 0.8333 | 0.9000 |
| controlled_vs_candidate_no_context | persona_style | 2 | 2 | 8 | 0.5000 | 0.5000 |
| controlled_vs_candidate_no_context | distinct1 | 2 | 10 | 0 | 0.1667 | 0.1667 |
| controlled_vs_candidate_no_context | length_score | 7 | 3 | 2 | 0.6667 | 0.7000 |
| controlled_vs_candidate_no_context | sentence_score | 7 | 0 | 5 | 0.7917 | 1.0000 |
| controlled_vs_candidate_no_context | bertscore_f1 | 8 | 4 | 0 | 0.6667 | 0.6667 |
| controlled_vs_candidate_no_context | overall_quality | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | context_relevance | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 9 | 2 | 1 | 0.7917 | 0.8182 |
| controlled_vs_baseline_no_context | naturalness | 4 | 8 | 0 | 0.3333 | 0.3333 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | context_overlap | 10 | 2 | 0 | 0.8333 | 0.8333 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 8 | 1 | 3 | 0.7917 | 0.8889 |
| controlled_vs_baseline_no_context | persona_style | 3 | 2 | 7 | 0.5417 | 0.6000 |
| controlled_vs_baseline_no_context | distinct1 | 2 | 10 | 0 | 0.1667 | 0.1667 |
| controlled_vs_baseline_no_context | length_score | 4 | 7 | 1 | 0.3750 | 0.3636 |
| controlled_vs_baseline_no_context | sentence_score | 3 | 2 | 7 | 0.5417 | 0.6000 |
| controlled_vs_baseline_no_context | bertscore_f1 | 7 | 5 | 0 | 0.5833 | 0.5833 |
| controlled_vs_baseline_no_context | overall_quality | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 10 | 0 | 2 | 0.9167 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | 5 | 7 | 0 | 0.4167 | 0.4167 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 9 | 0 | 3 | 0.8750 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | 3 | 2 | 7 | 0.5417 | 0.6000 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | 3 | 9 | 0 | 0.2500 | 0.2500 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | 5 | 7 | 0 | 0.4167 | 0.4167 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 1 | 1 | 10 | 0.5000 | 0.5000 |
| controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 9 | 3 | 0 | 0.7500 | 0.7500 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 9 | 2 | 1 | 0.7917 | 0.8182 |
| controlled_alt_vs_controlled_default | persona_consistency | 5 | 3 | 4 | 0.5833 | 0.6250 |
| controlled_alt_vs_controlled_default | naturalness | 8 | 3 | 1 | 0.7083 | 0.7273 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 7 | 2 | 3 | 0.7083 | 0.7778 |
| controlled_alt_vs_controlled_default | context_overlap | 10 | 1 | 1 | 0.8750 | 0.9091 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 4 | 2 | 6 | 0.5833 | 0.6667 |
| controlled_alt_vs_controlled_default | persona_style | 2 | 2 | 8 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | distinct1 | 6 | 5 | 1 | 0.5417 | 0.5455 |
| controlled_alt_vs_controlled_default | length_score | 6 | 4 | 2 | 0.5833 | 0.6000 |
| controlled_alt_vs_controlled_default | sentence_score | 1 | 2 | 9 | 0.4583 | 0.3333 |
| controlled_alt_vs_controlled_default | bertscore_f1 | 9 | 2 | 1 | 0.7917 | 0.8182 |
| controlled_alt_vs_controlled_default | overall_quality | 8 | 3 | 1 | 0.7083 | 0.7273 |
| controlled_alt_vs_proposed_raw | context_relevance | 10 | 2 | 0 | 0.8333 | 0.8333 |
| controlled_alt_vs_proposed_raw | persona_consistency | 11 | 1 | 0 | 0.9167 | 0.9167 |
| controlled_alt_vs_proposed_raw | naturalness | 10 | 2 | 0 | 0.8333 | 0.8333 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 10 | 1 | 1 | 0.8750 | 0.9091 |
| controlled_alt_vs_proposed_raw | context_overlap | 10 | 2 | 0 | 0.8333 | 0.8333 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 11 | 1 | 0 | 0.9167 | 0.9167 |
| controlled_alt_vs_proposed_raw | persona_style | 1 | 2 | 9 | 0.4583 | 0.3333 |
| controlled_alt_vs_proposed_raw | distinct1 | 7 | 5 | 0 | 0.5833 | 0.5833 |
| controlled_alt_vs_proposed_raw | length_score | 10 | 2 | 0 | 0.8333 | 0.8333 |
| controlled_alt_vs_proposed_raw | sentence_score | 4 | 0 | 8 | 0.6667 | 1.0000 |
| controlled_alt_vs_proposed_raw | bertscore_f1 | 9 | 3 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_proposed_raw | overall_quality | 11 | 1 | 0 | 0.9167 | 0.9167 |
| controlled_alt_vs_candidate_no_context | context_relevance | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 9 | 3 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_candidate_no_context | naturalness | 9 | 3 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 9 | 3 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_candidate_no_context | persona_style | 2 | 1 | 9 | 0.5417 | 0.6667 |
| controlled_alt_vs_candidate_no_context | distinct1 | 5 | 7 | 0 | 0.4167 | 0.4167 |
| controlled_alt_vs_candidate_no_context | length_score | 9 | 3 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_candidate_no_context | sentence_score | 6 | 0 | 6 | 0.7500 | 1.0000 |
| controlled_alt_vs_candidate_no_context | bertscore_f1 | 9 | 3 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_candidate_no_context | overall_quality | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context | context_relevance | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 10 | 1 | 1 | 0.8750 | 0.9091 |
| controlled_alt_vs_baseline_no_context | naturalness | 6 | 6 | 0 | 0.5000 | 0.5000 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context | context_overlap | 11 | 1 | 0 | 0.9167 | 0.9167 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 10 | 1 | 1 | 0.8750 | 0.9091 |
| controlled_alt_vs_baseline_no_context | persona_style | 2 | 1 | 9 | 0.5417 | 0.6667 |
| controlled_alt_vs_baseline_no_context | distinct1 | 4 | 8 | 0 | 0.3333 | 0.3333 |
| controlled_alt_vs_baseline_no_context | length_score | 6 | 5 | 1 | 0.5417 | 0.5455 |
| controlled_alt_vs_baseline_no_context | sentence_score | 2 | 2 | 8 | 0.5000 | 0.5000 |
| controlled_alt_vs_baseline_no_context | bertscore_f1 | 10 | 2 | 0 | 0.8333 | 0.8333 |
| controlled_alt_vs_baseline_no_context | overall_quality | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_relevance | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_consistency | 11 | 1 | 0 | 0.9167 | 0.9167 |
| controlled_alt_vs_baseline_no_context_phi3_latest | naturalness | 5 | 7 | 0 | 0.4167 | 0.4167 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | context_overlap | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 11 | 1 | 0 | 0.9167 | 0.9167 |
| controlled_alt_vs_baseline_no_context_phi3_latest | persona_style | 2 | 1 | 9 | 0.5417 | 0.6667 |
| controlled_alt_vs_baseline_no_context_phi3_latest | distinct1 | 3 | 9 | 0 | 0.2500 | 0.2500 |
| controlled_alt_vs_baseline_no_context_phi3_latest | length_score | 6 | 6 | 0 | 0.5000 | 0.5000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | sentence_score | 1 | 2 | 9 | 0.4583 | 0.3333 |
| controlled_alt_vs_baseline_no_context_phi3_latest | bertscore_f1 | 10 | 2 | 0 | 0.8333 | 0.8333 |
| controlled_alt_vs_baseline_no_context_phi3_latest | overall_quality | 12 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 12 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 9 | 2 | 1 | 0.7917 | 0.8182 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 4 | 8 | 0 | 0.3333 | 0.3333 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 12 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 10 | 2 | 0 | 0.8333 | 0.8333 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 8 | 1 | 3 | 0.7917 | 0.8889 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 3 | 2 | 7 | 0.5417 | 0.6000 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | 2 | 10 | 0 | 0.1667 | 0.1667 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 4 | 7 | 1 | 0.3750 | 0.3636 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 3 | 2 | 7 | 0.5417 | 0.6000 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 7 | 5 | 0 | 0.5833 | 0.5833 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 12 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 12 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 10 | 0 | 2 | 0.9167 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | 5 | 7 | 0 | 0.4167 | 0.4167 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 12 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 12 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 9 | 0 | 3 | 0.8750 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | 3 | 2 | 7 | 0.5417 | 0.6000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | 3 | 9 | 0 | 0.2500 | 0.2500 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | 5 | 7 | 0 | 0.4167 | 0.4167 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | 1 | 1 | 10 | 0.5000 | 0.5000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 9 | 3 | 0 | 0.7500 | 0.7500 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 12 | 0 | 0 | 1.0000 | 1.0000 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.3333 | 0.5833 | 0.4167 |
| proposed_contextual_controlled_quality | 0.0000 | 0.0000 | 0.2500 | 0.4167 | 0.5833 |
| proposed_contextual | 0.0000 | 0.0000 | 0.4167 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.4167 | 0.0000 | 0.0000 |
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
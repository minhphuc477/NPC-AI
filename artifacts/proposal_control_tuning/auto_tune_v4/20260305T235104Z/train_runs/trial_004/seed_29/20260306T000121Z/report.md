# Proposal Alignment Evaluation Report

- Run ID: `20260306T000121Z`
- Generated: `2026-03-06T00:03:57.716719+00:00`
- Scenarios: `artifacts\proposal_control_tuning\auto_tune_v4\20260305T235104Z\train_runs\trial_004\seed_29\20260306T000121Z\scenarios.jsonl`
- Scenario count: `16`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_tuned`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2287 (0.1765, 0.2868) | 0.3772 (0.2815, 0.4845) | 0.9285 (0.8988, 0.9535) | 0.4219 (0.3792, 0.4742) | n/a |
| proposed_contextual_controlled_tuned | 0.2547 (0.2213, 0.2907) | 0.3564 (0.2825, 0.4433) | 0.9061 (0.8803, 0.9279) | 0.4224 (0.3947, 0.4540) | n/a |
| proposed_contextual | 0.0873 (0.0422, 0.1343) | 0.2046 (0.1368, 0.2789) | 0.8381 (0.7965, 0.8762) | 0.2751 (0.2298, 0.3203) | n/a |
| candidate_no_context | 0.0284 (0.0165, 0.0421) | 0.2157 (0.1458, 0.2918) | 0.8335 (0.7885, 0.8810) | 0.2511 (0.2159, 0.2900) | n/a |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0589 | 2.0720 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0111 | -0.0514 |
| proposed_vs_candidate_no_context | naturalness | 0.0046 | 0.0055 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0739 | 3.2500 |
| proposed_vs_candidate_no_context | context_overlap | 0.0240 | 0.5753 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0119 | -0.0930 |
| proposed_vs_candidate_no_context | persona_style | -0.0078 | -0.0138 |
| proposed_vs_candidate_no_context | distinct1 | -0.0011 | -0.0011 |
| proposed_vs_candidate_no_context | length_score | 0.0250 | 0.0682 |
| proposed_vs_candidate_no_context | sentence_score | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | overall_quality | 0.0240 | 0.0955 |
| controlled_vs_proposed_raw | context_relevance | 0.1413 | 1.6181 |
| controlled_vs_proposed_raw | persona_consistency | 0.1727 | 0.8440 |
| controlled_vs_proposed_raw | naturalness | 0.0903 | 0.1078 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.1841 | 1.9059 |
| controlled_vs_proposed_raw | context_overlap | 0.0415 | 0.6318 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1875 | 1.6154 |
| controlled_vs_proposed_raw | persona_style | 0.1133 | 0.2028 |
| controlled_vs_proposed_raw | distinct1 | -0.0054 | -0.0057 |
| controlled_vs_proposed_raw | length_score | 0.3750 | 0.9574 |
| controlled_vs_proposed_raw | sentence_score | 0.1750 | 0.2179 |
| controlled_vs_proposed_raw | overall_quality | 0.1467 | 0.5333 |
| controlled_vs_candidate_no_context | context_relevance | 0.2002 | 7.0428 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1616 | 0.7492 |
| controlled_vs_candidate_no_context | naturalness | 0.0949 | 0.1139 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.2580 | 11.3500 |
| controlled_vs_candidate_no_context | context_overlap | 0.0656 | 1.5706 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1756 | 1.3721 |
| controlled_vs_candidate_no_context | persona_style | 0.1055 | 0.1862 |
| controlled_vs_candidate_no_context | distinct1 | -0.0064 | -0.0068 |
| controlled_vs_candidate_no_context | length_score | 0.4000 | 1.0909 |
| controlled_vs_candidate_no_context | sentence_score | 0.1750 | 0.2179 |
| controlled_vs_candidate_no_context | overall_quality | 0.1707 | 0.6798 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0260 | 0.1137 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0208 | -0.0551 |
| controlled_alt_vs_controlled_default | naturalness | -0.0224 | -0.0241 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0341 | 0.1215 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0071 | 0.0664 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0143 | -0.0471 |
| controlled_alt_vs_controlled_default | persona_style | -0.0469 | -0.0698 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0094 | -0.0100 |
| controlled_alt_vs_controlled_default | length_score | -0.0604 | -0.0788 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0656 | -0.0671 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0005 | 0.0012 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1673 | 1.9158 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1519 | 0.7423 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0679 | 0.0811 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2182 | 2.2588 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0487 | 0.7402 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1732 | 1.4923 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0664 | 0.1189 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0148 | -0.0156 |
| controlled_alt_vs_proposed_raw | length_score | 0.3146 | 0.8032 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1094 | 0.1362 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1472 | 0.5351 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2262 | 7.9573 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1408 | 0.6527 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0725 | 0.0870 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.2920 | 12.8500 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0727 | 1.7412 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1613 | 1.2605 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0586 | 0.1034 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0158 | -0.0167 |
| controlled_alt_vs_candidate_no_context | length_score | 0.3396 | 0.9261 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1094 | 0.1362 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1712 | 0.6818 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0589 | (0.0252, 0.1005) | 0.0000 | 0.0589 | (0.0226, 0.0860) | 0.0007 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0111 | (-0.0631, 0.0354) | 0.6627 | -0.0111 | (-0.0613, 0.0183) | 0.6840 |
| proposed_vs_candidate_no_context | naturalness | 0.0046 | (-0.0399, 0.0499) | 0.4347 | 0.0046 | (-0.0335, 0.0432) | 0.3677 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0739 | (0.0284, 0.1250) | 0.0000 | 0.0739 | (0.0303, 0.1061) | 0.0000 |
| proposed_vs_candidate_no_context | context_overlap | 0.0240 | (0.0072, 0.0440) | 0.0023 | 0.0240 | (0.0044, 0.0565) | 0.0003 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0119 | (-0.0685, 0.0357) | 0.6673 | -0.0119 | (-0.0714, 0.0252) | 0.6640 |
| proposed_vs_candidate_no_context | persona_style | -0.0078 | (-0.1230, 0.1230) | 0.5853 | -0.0078 | (-0.0469, 0.0197) | 0.7213 |
| proposed_vs_candidate_no_context | distinct1 | -0.0011 | (-0.0144, 0.0118) | 0.5527 | -0.0011 | (-0.0116, 0.0150) | 0.5870 |
| proposed_vs_candidate_no_context | length_score | 0.0250 | (-0.1500, 0.1938) | 0.3937 | 0.0250 | (-0.1200, 0.1762) | 0.3937 |
| proposed_vs_candidate_no_context | sentence_score | 0.0000 | (-0.1094, 0.1094) | 0.5887 | 0.0000 | (-0.0618, 0.0750) | 0.6520 |
| proposed_vs_candidate_no_context | overall_quality | 0.0240 | (-0.0055, 0.0580) | 0.0523 | 0.0240 | (0.0034, 0.0453) | 0.0003 |
| controlled_vs_proposed_raw | context_relevance | 0.1413 | (0.0802, 0.2119) | 0.0000 | 0.1413 | (0.0738, 0.2197) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.1727 | (0.0752, 0.2844) | 0.0000 | 0.1727 | (0.1140, 0.2095) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0903 | (0.0456, 0.1314) | 0.0000 | 0.0903 | (0.0263, 0.1342) | 0.0043 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.1841 | (0.1045, 0.2693) | 0.0000 | 0.1841 | (0.0974, 0.2941) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0415 | (0.0128, 0.0737) | 0.0003 | 0.0415 | (0.0175, 0.0695) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1875 | (0.0714, 0.3393) | 0.0000 | 0.1875 | (0.1156, 0.2530) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.1133 | (0.0156, 0.2267) | 0.0103 | 0.1133 | (0.0000, 0.2465) | 0.0783 |
| controlled_vs_proposed_raw | distinct1 | -0.0054 | (-0.0232, 0.0119) | 0.7227 | -0.0054 | (-0.0230, 0.0091) | 0.7600 |
| controlled_vs_proposed_raw | length_score | 0.3750 | (0.2124, 0.5312) | 0.0000 | 0.3750 | (0.1139, 0.5298) | 0.0030 |
| controlled_vs_proposed_raw | sentence_score | 0.1750 | (0.0656, 0.2625) | 0.0017 | 0.1750 | (0.0250, 0.2802) | 0.0203 |
| controlled_vs_proposed_raw | overall_quality | 0.1467 | (0.0956, 0.2135) | 0.0000 | 0.1467 | (0.0805, 0.2037) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2002 | (0.1499, 0.2588) | 0.0000 | 0.2002 | (0.1442, 0.2507) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1616 | (0.0658, 0.2743) | 0.0000 | 0.1616 | (0.0758, 0.2245) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.0949 | (0.0345, 0.1481) | 0.0013 | 0.0949 | (0.0034, 0.1481) | 0.0200 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.2580 | (0.2000, 0.3318) | 0.0000 | 0.2580 | (0.1883, 0.3225) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0656 | (0.0377, 0.0990) | 0.0000 | 0.0656 | (0.0364, 0.0974) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1756 | (0.0654, 0.3200) | 0.0000 | 0.1756 | (0.0680, 0.2632) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.1055 | (0.0156, 0.2285) | 0.0090 | 0.1055 | (0.0000, 0.2535) | 0.0750 |
| controlled_vs_candidate_no_context | distinct1 | -0.0064 | (-0.0269, 0.0134) | 0.7330 | -0.0064 | (-0.0287, 0.0134) | 0.7193 |
| controlled_vs_candidate_no_context | length_score | 0.4000 | (0.1667, 0.6021) | 0.0017 | 0.4000 | (0.0722, 0.6289) | 0.0080 |
| controlled_vs_candidate_no_context | sentence_score | 0.1750 | (0.0656, 0.2844) | 0.0010 | 0.1750 | (0.0500, 0.2800) | 0.0127 |
| controlled_vs_candidate_no_context | overall_quality | 0.1707 | (0.1161, 0.2385) | 0.0000 | 0.1707 | (0.0996, 0.2203) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0260 | (-0.0346, 0.0792) | 0.1947 | 0.0260 | (-0.0241, 0.0969) | 0.1950 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0208 | (-0.1551, 0.1164) | 0.6167 | -0.0208 | (-0.0757, 0.0215) | 0.7830 |
| controlled_alt_vs_controlled_default | naturalness | -0.0224 | (-0.0536, 0.0084) | 0.9120 | -0.0224 | (-0.0366, -0.0103) | 1.0000 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0341 | (-0.0455, 0.1024) | 0.1740 | 0.0341 | (-0.0287, 0.1259) | 0.1710 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0071 | (-0.0271, 0.0363) | 0.3180 | 0.0071 | (-0.0261, 0.0452) | 0.3730 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0143 | (-0.1893, 0.1554) | 0.5607 | -0.0143 | (-0.0625, 0.0269) | 0.7400 |
| controlled_alt_vs_controlled_default | persona_style | -0.0469 | (-0.0938, 0.0000) | 1.0000 | -0.0469 | (-0.1500, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0094 | (-0.0244, 0.0052) | 0.8973 | -0.0094 | (-0.0241, 0.0020) | 0.9177 |
| controlled_alt_vs_controlled_default | length_score | -0.0604 | (-0.2021, 0.0834) | 0.7883 | -0.0604 | (-0.1125, -0.0083) | 0.9897 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0656 | (-0.1312, 0.0000) | 1.0000 | -0.0656 | (-0.1531, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0005 | (-0.0645, 0.0579) | 0.4757 | 0.0005 | (-0.0146, 0.0288) | 0.5010 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1673 | (0.1148, 0.2196) | 0.0000 | 0.1673 | (0.1162, 0.2252) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1519 | (0.0722, 0.2406) | 0.0003 | 0.1519 | (0.0380, 0.2187) | 0.0030 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0679 | (0.0198, 0.1171) | 0.0020 | 0.0679 | (0.0030, 0.1124) | 0.0187 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2182 | (0.1511, 0.2909) | 0.0000 | 0.2182 | (0.1433, 0.3030) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0487 | (0.0281, 0.0682) | 0.0000 | 0.0487 | (0.0251, 0.0699) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1732 | (0.0824, 0.2810) | 0.0000 | 0.1732 | (0.0532, 0.2601) | 0.0003 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0664 | (-0.0312, 0.1897) | 0.1403 | 0.0664 | (-0.0500, 0.2188) | 0.3343 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0148 | (-0.0351, 0.0048) | 0.9273 | -0.0148 | (-0.0425, 0.0085) | 0.8857 |
| controlled_alt_vs_proposed_raw | length_score | 0.3146 | (0.1271, 0.5000) | 0.0003 | 0.3146 | (0.1000, 0.4648) | 0.0017 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1094 | (0.0000, 0.2188) | 0.0547 | 0.1094 | (-0.0269, 0.2026) | 0.0793 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1472 | (0.1025, 0.1952) | 0.0000 | 0.1472 | (0.0815, 0.2026) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2262 | (0.1980, 0.2594) | 0.0000 | 0.2262 | (0.1932, 0.2806) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1408 | (0.0457, 0.2442) | 0.0013 | 0.1408 | (0.0196, 0.2285) | 0.0130 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0725 | (0.0120, 0.1286) | 0.0100 | 0.0725 | (-0.0196, 0.1276) | 0.0523 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.2920 | (0.2557, 0.3364) | 0.0000 | 0.2920 | (0.2449, 0.3636) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0727 | (0.0585, 0.0874) | 0.0000 | 0.0727 | (0.0594, 0.0921) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1613 | (0.0473, 0.2839) | 0.0017 | 0.1613 | (0.0272, 0.2656) | 0.0077 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0586 | (-0.0469, 0.1895) | 0.1803 | 0.0586 | (-0.1000, 0.2270) | 0.3580 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0158 | (-0.0377, 0.0058) | 0.9237 | -0.0158 | (-0.0496, 0.0109) | 0.8630 |
| controlled_alt_vs_candidate_no_context | length_score | 0.3396 | (0.1104, 0.5563) | 0.0023 | 0.3396 | (0.0133, 0.5375) | 0.0210 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1094 | (0.0213, 0.1969) | 0.0250 | 0.1094 | (0.0000, 0.2026) | 0.0823 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1712 | (0.1236, 0.2163) | 0.0000 | 0.1712 | (0.1078, 0.2152) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 9 | 1 | 6 | 0.7500 | 0.9000 |
| proposed_vs_candidate_no_context | persona_consistency | 4 | 3 | 9 | 0.5312 | 0.5714 |
| proposed_vs_candidate_no_context | naturalness | 5 | 5 | 6 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 7 | 0 | 9 | 0.7188 | 1.0000 |
| proposed_vs_candidate_no_context | context_overlap | 7 | 3 | 6 | 0.6250 | 0.7000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 2 | 2 | 12 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | persona_style | 2 | 2 | 12 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | distinct1 | 3 | 4 | 9 | 0.4688 | 0.4286 |
| proposed_vs_candidate_no_context | length_score | 5 | 5 | 6 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | sentence_score | 3 | 3 | 10 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | overall_quality | 7 | 3 | 6 | 0.6250 | 0.7000 |
| controlled_vs_proposed_raw | context_relevance | 12 | 4 | 0 | 0.7500 | 0.7500 |
| controlled_vs_proposed_raw | persona_consistency | 10 | 1 | 5 | 0.7812 | 0.9091 |
| controlled_vs_proposed_raw | naturalness | 14 | 2 | 0 | 0.8750 | 0.8750 |
| controlled_vs_proposed_raw | context_keyword_coverage | 12 | 1 | 3 | 0.8438 | 0.9231 |
| controlled_vs_proposed_raw | context_overlap | 11 | 5 | 0 | 0.6875 | 0.6875 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 10 | 1 | 5 | 0.7812 | 0.9091 |
| controlled_vs_proposed_raw | persona_style | 4 | 0 | 12 | 0.6250 | 1.0000 |
| controlled_vs_proposed_raw | distinct1 | 7 | 9 | 0 | 0.4375 | 0.4375 |
| controlled_vs_proposed_raw | length_score | 14 | 2 | 0 | 0.8750 | 0.8750 |
| controlled_vs_proposed_raw | sentence_score | 9 | 1 | 6 | 0.7500 | 0.9000 |
| controlled_vs_proposed_raw | overall_quality | 15 | 1 | 0 | 0.9375 | 0.9375 |
| controlled_vs_candidate_no_context | context_relevance | 15 | 1 | 0 | 0.9375 | 0.9375 |
| controlled_vs_candidate_no_context | persona_consistency | 11 | 1 | 4 | 0.8125 | 0.9167 |
| controlled_vs_candidate_no_context | naturalness | 12 | 4 | 0 | 0.7500 | 0.7500 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 15 | 0 | 1 | 0.9688 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 15 | 1 | 0 | 0.9375 | 0.9375 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 10 | 1 | 5 | 0.7812 | 0.9091 |
| controlled_vs_candidate_no_context | persona_style | 4 | 0 | 12 | 0.6250 | 1.0000 |
| controlled_vs_candidate_no_context | distinct1 | 7 | 9 | 0 | 0.4375 | 0.4375 |
| controlled_vs_candidate_no_context | length_score | 13 | 3 | 0 | 0.8125 | 0.8125 |
| controlled_vs_candidate_no_context | sentence_score | 9 | 1 | 6 | 0.7500 | 0.9000 |
| controlled_vs_candidate_no_context | overall_quality | 16 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 9 | 3 | 4 | 0.6875 | 0.7500 |
| controlled_alt_vs_controlled_default | persona_consistency | 4 | 5 | 7 | 0.4688 | 0.4444 |
| controlled_alt_vs_controlled_default | naturalness | 3 | 9 | 4 | 0.3125 | 0.2500 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 8 | 3 | 5 | 0.6562 | 0.7273 |
| controlled_alt_vs_controlled_default | context_overlap | 9 | 3 | 4 | 0.6875 | 0.7500 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 4 | 4 | 8 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | persona_style | 0 | 3 | 13 | 0.4062 | 0.0000 |
| controlled_alt_vs_controlled_default | distinct1 | 4 | 8 | 4 | 0.3750 | 0.3333 |
| controlled_alt_vs_controlled_default | length_score | 3 | 9 | 4 | 0.3125 | 0.2500 |
| controlled_alt_vs_controlled_default | sentence_score | 0 | 3 | 13 | 0.4062 | 0.0000 |
| controlled_alt_vs_controlled_default | overall_quality | 6 | 6 | 4 | 0.5000 | 0.5000 |
| controlled_alt_vs_proposed_raw | context_relevance | 15 | 1 | 0 | 0.9375 | 0.9375 |
| controlled_alt_vs_proposed_raw | persona_consistency | 13 | 2 | 1 | 0.8438 | 0.8667 |
| controlled_alt_vs_proposed_raw | naturalness | 12 | 4 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 14 | 1 | 1 | 0.9062 | 0.9333 |
| controlled_alt_vs_proposed_raw | context_overlap | 13 | 3 | 0 | 0.8125 | 0.8125 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 13 | 2 | 1 | 0.8438 | 0.8667 |
| controlled_alt_vs_proposed_raw | persona_style | 2 | 1 | 13 | 0.5312 | 0.6667 |
| controlled_alt_vs_proposed_raw | distinct1 | 6 | 9 | 1 | 0.4062 | 0.4000 |
| controlled_alt_vs_proposed_raw | length_score | 14 | 2 | 0 | 0.8750 | 0.8750 |
| controlled_alt_vs_proposed_raw | sentence_score | 7 | 2 | 7 | 0.6562 | 0.7778 |
| controlled_alt_vs_proposed_raw | overall_quality | 15 | 1 | 0 | 0.9375 | 0.9375 |
| controlled_alt_vs_candidate_no_context | context_relevance | 16 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 12 | 2 | 2 | 0.8125 | 0.8571 |
| controlled_alt_vs_candidate_no_context | naturalness | 11 | 5 | 0 | 0.6875 | 0.6875 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 16 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 16 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 12 | 2 | 2 | 0.8125 | 0.8571 |
| controlled_alt_vs_candidate_no_context | persona_style | 3 | 2 | 11 | 0.5312 | 0.6000 |
| controlled_alt_vs_candidate_no_context | distinct1 | 6 | 9 | 1 | 0.4062 | 0.4000 |
| controlled_alt_vs_candidate_no_context | length_score | 12 | 4 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_candidate_no_context | sentence_score | 6 | 1 | 9 | 0.6562 | 0.8571 |
| controlled_alt_vs_candidate_no_context | overall_quality | 15 | 1 | 0 | 0.9375 | 0.9375 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.4375 | 0.3750 | 0.6250 |
| proposed_contextual_controlled_tuned | 0.0000 | 0.0000 | 0.4375 | 0.3750 | 0.6250 |
| proposed_contextual | 0.0000 | 0.0000 | 0.4375 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `5`
- Unique template signatures: `16`
- Template signature ratio: `1.0000`
- Effective sample size by source clustering: `4.74`
- Effective sample size by template-signature clustering: `16.00`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: unavailable (disabled_by_flag).

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
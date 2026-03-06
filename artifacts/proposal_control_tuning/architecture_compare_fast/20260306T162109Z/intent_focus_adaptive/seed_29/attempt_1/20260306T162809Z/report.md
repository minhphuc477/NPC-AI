# Proposal Alignment Evaluation Report

- Run ID: `20260306T162809Z`
- Generated: `2026-03-06T16:31:25.027691+00:00`
- Scenarios: `artifacts\proposal_control_tuning\architecture_compare_fast\20260306T162109Z\intent_focus_adaptive\seed_29\attempt_1\20260306T162809Z\scenarios.jsonl`
- Scenario count: `20`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_alt`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2796 (0.2564, 0.3050) | 0.2532 (0.2097, 0.2974) | 0.9137 (0.8925, 0.9322) | 0.3971 (0.3738, 0.4185) | n/a |
| proposed_contextual_controlled_alt | 0.2548 (0.2338, 0.2756) | 0.2759 (0.2183, 0.3344) | 0.9071 (0.8818, 0.9292) | 0.3927 (0.3677, 0.4168) | n/a |
| proposed_contextual | 0.0873 (0.0417, 0.1402) | 0.1488 (0.1081, 0.1936) | 0.8199 (0.7839, 0.8581) | 0.2512 (0.2144, 0.2952) | n/a |
| candidate_no_context | 0.0217 (0.0111, 0.0379) | 0.1726 (0.1101, 0.2382) | 0.8268 (0.7894, 0.8669) | 0.2315 (0.2017, 0.2670) | n/a |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0655 | 3.0145 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0238 | -0.1379 |
| proposed_vs_candidate_no_context | naturalness | -0.0069 | -0.0084 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0837 | 6.1389 |
| proposed_vs_candidate_no_context | context_overlap | 0.0231 | 0.5691 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0345 | -0.3816 |
| proposed_vs_candidate_no_context | persona_style | 0.0191 | 0.0382 |
| proposed_vs_candidate_no_context | distinct1 | -0.0022 | -0.0023 |
| proposed_vs_candidate_no_context | length_score | -0.0217 | -0.0573 |
| proposed_vs_candidate_no_context | sentence_score | -0.0175 | -0.0227 |
| proposed_vs_candidate_no_context | overall_quality | 0.0197 | 0.0851 |
| controlled_vs_proposed_raw | context_relevance | 0.1923 | 2.2032 |
| controlled_vs_proposed_raw | persona_consistency | 0.1044 | 0.7019 |
| controlled_vs_proposed_raw | naturalness | 0.0938 | 0.1144 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2510 | 2.5782 |
| controlled_vs_proposed_raw | context_overlap | 0.0554 | 0.8681 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1086 | 1.9404 |
| controlled_vs_proposed_raw | persona_style | 0.0878 | 0.1688 |
| controlled_vs_proposed_raw | distinct1 | 0.0020 | 0.0022 |
| controlled_vs_proposed_raw | length_score | 0.3950 | 1.1075 |
| controlled_vs_proposed_raw | sentence_score | 0.1400 | 0.1854 |
| controlled_vs_proposed_raw | overall_quality | 0.1459 | 0.5809 |
| controlled_vs_candidate_no_context | context_relevance | 0.2578 | 11.8592 |
| controlled_vs_candidate_no_context | persona_consistency | 0.0806 | 0.4672 |
| controlled_vs_candidate_no_context | naturalness | 0.0869 | 0.1051 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3347 | 24.5444 |
| controlled_vs_candidate_no_context | context_overlap | 0.0785 | 1.9313 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.0740 | 0.8184 |
| controlled_vs_candidate_no_context | persona_style | 0.1069 | 0.2134 |
| controlled_vs_candidate_no_context | distinct1 | -0.0001 | -0.0001 |
| controlled_vs_candidate_no_context | length_score | 0.3733 | 0.9868 |
| controlled_vs_candidate_no_context | sentence_score | 0.1225 | 0.1586 |
| controlled_vs_candidate_no_context | overall_quality | 0.1656 | 0.7154 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0247 | -0.0885 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0227 | 0.0897 |
| controlled_alt_vs_controlled_default | naturalness | -0.0066 | -0.0072 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0273 | -0.0783 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0189 | -0.1583 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0198 | 0.1201 |
| controlled_alt_vs_controlled_default | persona_style | 0.0345 | 0.0567 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0004 | -0.0004 |
| controlled_alt_vs_controlled_default | length_score | -0.0583 | -0.0776 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0525 | 0.0587 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0044 | -0.0111 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1676 | 1.9197 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1271 | 0.8545 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0873 | 0.1064 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2237 | 2.2981 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0365 | 0.5724 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1283 | 2.2936 |
| controlled_alt_vs_proposed_raw | persona_style | 0.1222 | 0.2351 |
| controlled_alt_vs_proposed_raw | distinct1 | 0.0017 | 0.0018 |
| controlled_alt_vs_proposed_raw | length_score | 0.3367 | 0.9439 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1925 | 0.2550 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1415 | 0.5634 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2331 | 10.7209 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1033 | 0.5987 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0803 | 0.0971 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3074 | 22.5444 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0597 | 1.4674 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.0938 | 1.0368 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.1414 | 0.2822 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0005 | -0.0005 |
| controlled_alt_vs_candidate_no_context | length_score | 0.3150 | 0.8326 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1750 | 0.2265 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1612 | 0.6964 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0655 | (0.0160, 0.1263) | 0.0030 | 0.0655 | (0.0053, 0.1426) | 0.0137 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0238 | (-0.0698, 0.0203) | 0.8527 | -0.0238 | (-0.0934, 0.0197) | 0.8297 |
| proposed_vs_candidate_no_context | naturalness | -0.0069 | (-0.0413, 0.0273) | 0.6530 | -0.0069 | (-0.0519, 0.0241) | 0.6360 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0837 | (0.0178, 0.1610) | 0.0060 | 0.0837 | (0.0058, 0.1883) | 0.0180 |
| proposed_vs_candidate_no_context | context_overlap | 0.0231 | (0.0082, 0.0397) | 0.0007 | 0.0231 | (0.0059, 0.0467) | 0.0000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0345 | (-0.0917, 0.0190) | 0.8993 | -0.0345 | (-0.1222, 0.0159) | 0.8690 |
| proposed_vs_candidate_no_context | persona_style | 0.0191 | (-0.0048, 0.0549) | 0.1253 | 0.0191 | (0.0000, 0.0516) | 0.1040 |
| proposed_vs_candidate_no_context | distinct1 | -0.0022 | (-0.0174, 0.0134) | 0.5997 | -0.0022 | (-0.0221, 0.0139) | 0.6000 |
| proposed_vs_candidate_no_context | length_score | -0.0217 | (-0.1617, 0.1234) | 0.6133 | -0.0217 | (-0.2045, 0.1143) | 0.5850 |
| proposed_vs_candidate_no_context | sentence_score | -0.0175 | (-0.0875, 0.0525) | 0.7497 | -0.0175 | (-0.0778, 0.0412) | 0.8257 |
| proposed_vs_candidate_no_context | overall_quality | 0.0197 | (-0.0209, 0.0597) | 0.1660 | 0.0197 | (-0.0319, 0.0673) | 0.2060 |
| controlled_vs_proposed_raw | context_relevance | 0.1923 | (0.1414, 0.2383) | 0.0000 | 0.1923 | (0.1277, 0.2402) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.1044 | (0.0523, 0.1535) | 0.0000 | 0.1044 | (0.0432, 0.1848) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0938 | (0.0456, 0.1381) | 0.0000 | 0.0938 | (0.0095, 0.1565) | 0.0130 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2510 | (0.1849, 0.3091) | 0.0000 | 0.2510 | (0.1667, 0.3149) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0554 | (0.0289, 0.0903) | 0.0000 | 0.0554 | (0.0242, 0.0809) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1086 | (0.0576, 0.1572) | 0.0000 | 0.1086 | (0.0595, 0.1738) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.0878 | (-0.0112, 0.2107) | 0.0460 | 0.0878 | (-0.0446, 0.2972) | 0.2603 |
| controlled_vs_proposed_raw | distinct1 | 0.0020 | (-0.0123, 0.0156) | 0.3600 | 0.0020 | (-0.0151, 0.0173) | 0.3987 |
| controlled_vs_proposed_raw | length_score | 0.3950 | (0.2167, 0.5683) | 0.0000 | 0.3950 | (0.0907, 0.6315) | 0.0060 |
| controlled_vs_proposed_raw | sentence_score | 0.1400 | (0.0175, 0.2450) | 0.0220 | 0.1400 | (-0.0618, 0.2833) | 0.1063 |
| controlled_vs_proposed_raw | overall_quality | 0.1459 | (0.1009, 0.1866) | 0.0000 | 0.1459 | (0.0925, 0.1987) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2578 | (0.2333, 0.2825) | 0.0000 | 0.2578 | (0.2358, 0.2869) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.0806 | (0.0126, 0.1447) | 0.0110 | 0.0806 | (0.0065, 0.1619) | 0.0183 |
| controlled_vs_candidate_no_context | naturalness | 0.0869 | (0.0304, 0.1383) | 0.0007 | 0.0869 | (-0.0036, 0.1545) | 0.0290 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3347 | (0.3045, 0.3636) | 0.0000 | 0.3347 | (0.2993, 0.3788) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0785 | (0.0550, 0.1116) | 0.0000 | 0.0785 | (0.0605, 0.0930) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.0740 | (0.0055, 0.1374) | 0.0173 | 0.0740 | (0.0074, 0.1350) | 0.0177 |
| controlled_vs_candidate_no_context | persona_style | 0.1069 | (0.0025, 0.2178) | 0.0220 | 0.1069 | (-0.0230, 0.3190) | 0.1070 |
| controlled_vs_candidate_no_context | distinct1 | -0.0001 | (-0.0172, 0.0164) | 0.5103 | -0.0001 | (-0.0253, 0.0173) | 0.5127 |
| controlled_vs_candidate_no_context | length_score | 0.3733 | (0.1650, 0.5733) | 0.0000 | 0.3733 | (0.0426, 0.6345) | 0.0147 |
| controlled_vs_candidate_no_context | sentence_score | 0.1225 | (0.0175, 0.2275) | 0.0203 | 0.1225 | (-0.0875, 0.2660) | 0.1303 |
| controlled_vs_candidate_no_context | overall_quality | 0.1656 | (0.1346, 0.1963) | 0.0000 | 0.1656 | (0.1268, 0.2026) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0247 | (-0.0501, -0.0014) | 0.9860 | -0.0247 | (-0.0452, -0.0067) | 0.9907 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0227 | (-0.0136, 0.0733) | 0.1380 | 0.0227 | (-0.0116, 0.0939) | 0.1443 |
| controlled_alt_vs_controlled_default | naturalness | -0.0066 | (-0.0293, 0.0193) | 0.7087 | -0.0066 | (-0.0360, 0.0192) | 0.7073 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0273 | (-0.0583, -0.0008) | 0.9870 | -0.0273 | (-0.0574, -0.0043) | 0.9830 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0189 | (-0.0505, 0.0010) | 0.9577 | -0.0189 | (-0.0370, 0.0001) | 0.9740 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0198 | (-0.0202, 0.0793) | 0.2483 | 0.0198 | (-0.0174, 0.0946) | 0.2413 |
| controlled_alt_vs_controlled_default | persona_style | 0.0345 | (0.0000, 0.0803) | 0.0350 | 0.0345 | (0.0036, 0.0914) | 0.0233 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0004 | (-0.0158, 0.0145) | 0.5247 | -0.0004 | (-0.0134, 0.0118) | 0.5217 |
| controlled_alt_vs_controlled_default | length_score | -0.0583 | (-0.1884, 0.0784) | 0.8183 | -0.0583 | (-0.2356, 0.0855) | 0.7997 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0525 | (-0.0350, 0.1400) | 0.1670 | 0.0525 | (-0.0350, 0.1842) | 0.1987 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0044 | (-0.0235, 0.0140) | 0.6853 | -0.0044 | (-0.0185, 0.0188) | 0.7007 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1676 | (0.1032, 0.2198) | 0.0000 | 0.1676 | (0.0926, 0.2244) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1271 | (0.0782, 0.1800) | 0.0000 | 0.1271 | (0.0577, 0.2205) | 0.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0873 | (0.0348, 0.1363) | 0.0007 | 0.0873 | (-0.0039, 0.1510) | 0.0317 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2237 | (0.1383, 0.2909) | 0.0000 | 0.2237 | (0.1288, 0.2988) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0365 | (0.0171, 0.0520) | 0.0000 | 0.0365 | (0.0144, 0.0589) | 0.0013 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1283 | (0.0748, 0.1843) | 0.0000 | 0.1283 | (0.0589, 0.2177) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 0.1222 | (0.0292, 0.2326) | 0.0017 | 0.1222 | (-0.0025, 0.3293) | 0.0943 |
| controlled_alt_vs_proposed_raw | distinct1 | 0.0017 | (-0.0171, 0.0189) | 0.4257 | 0.0017 | (-0.0216, 0.0208) | 0.4227 |
| controlled_alt_vs_proposed_raw | length_score | 0.3367 | (0.1266, 0.5400) | 0.0013 | 0.3367 | (-0.0216, 0.5833) | 0.0340 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1925 | (0.0875, 0.2975) | 0.0007 | 0.1925 | (0.0583, 0.2975) | 0.0030 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1415 | (0.0966, 0.1802) | 0.0000 | 0.1415 | (0.0887, 0.1974) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2331 | (0.2103, 0.2571) | 0.0000 | 0.2331 | (0.2167, 0.2533) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1033 | (0.0504, 0.1550) | 0.0003 | 0.1033 | (0.0381, 0.1880) | 0.0000 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0803 | (0.0233, 0.1335) | 0.0027 | 0.0803 | (-0.0176, 0.1485) | 0.0470 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3074 | (0.2764, 0.3392) | 0.0000 | 0.3074 | (0.2859, 0.3346) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0597 | (0.0483, 0.0728) | 0.0000 | 0.0597 | (0.0482, 0.0783) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.0938 | (0.0433, 0.1438) | 0.0007 | 0.0938 | (0.0390, 0.1535) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.1414 | (0.0424, 0.2591) | 0.0007 | 0.1414 | (0.0080, 0.3560) | 0.0210 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0005 | (-0.0209, 0.0193) | 0.5143 | -0.0005 | (-0.0325, 0.0258) | 0.5243 |
| controlled_alt_vs_candidate_no_context | length_score | 0.3150 | (0.0917, 0.5284) | 0.0033 | 0.3150 | (-0.0647, 0.5715) | 0.0490 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1750 | (0.0700, 0.2800) | 0.0017 | 0.1750 | (0.0219, 0.2891) | 0.0140 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1612 | (0.1316, 0.1912) | 0.0000 | 0.1612 | (0.1289, 0.2008) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 10 | 1 | 9 | 0.7250 | 0.9091 |
| proposed_vs_candidate_no_context | persona_consistency | 3 | 5 | 12 | 0.4500 | 0.3750 |
| proposed_vs_candidate_no_context | naturalness | 4 | 7 | 9 | 0.4250 | 0.3636 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 8 | 1 | 11 | 0.6750 | 0.8889 |
| proposed_vs_candidate_no_context | context_overlap | 10 | 1 | 9 | 0.7250 | 0.9091 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 2 | 4 | 14 | 0.4500 | 0.3333 |
| proposed_vs_candidate_no_context | persona_style | 2 | 1 | 17 | 0.5250 | 0.6667 |
| proposed_vs_candidate_no_context | distinct1 | 4 | 7 | 9 | 0.4250 | 0.3636 |
| proposed_vs_candidate_no_context | length_score | 4 | 7 | 9 | 0.4250 | 0.3636 |
| proposed_vs_candidate_no_context | sentence_score | 2 | 3 | 15 | 0.4750 | 0.4000 |
| proposed_vs_candidate_no_context | overall_quality | 7 | 4 | 9 | 0.5750 | 0.6364 |
| controlled_vs_proposed_raw | context_relevance | 18 | 2 | 0 | 0.9000 | 0.9000 |
| controlled_vs_proposed_raw | persona_consistency | 12 | 3 | 5 | 0.7250 | 0.8000 |
| controlled_vs_proposed_raw | naturalness | 14 | 6 | 0 | 0.7000 | 0.7000 |
| controlled_vs_proposed_raw | context_keyword_coverage | 18 | 2 | 0 | 0.9000 | 0.9000 |
| controlled_vs_proposed_raw | context_overlap | 17 | 3 | 0 | 0.8500 | 0.8500 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 12 | 1 | 7 | 0.7750 | 0.9231 |
| controlled_vs_proposed_raw | persona_style | 4 | 3 | 13 | 0.5250 | 0.5714 |
| controlled_vs_proposed_raw | distinct1 | 11 | 9 | 0 | 0.5500 | 0.5500 |
| controlled_vs_proposed_raw | length_score | 16 | 4 | 0 | 0.8000 | 0.8000 |
| controlled_vs_proposed_raw | sentence_score | 12 | 4 | 4 | 0.7000 | 0.7500 |
| controlled_vs_proposed_raw | overall_quality | 18 | 2 | 0 | 0.9000 | 0.9000 |
| controlled_vs_candidate_no_context | context_relevance | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 11 | 3 | 6 | 0.7000 | 0.7857 |
| controlled_vs_candidate_no_context | naturalness | 15 | 5 | 0 | 0.7500 | 0.7500 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 11 | 3 | 6 | 0.7000 | 0.7857 |
| controlled_vs_candidate_no_context | persona_style | 6 | 2 | 12 | 0.6000 | 0.7500 |
| controlled_vs_candidate_no_context | distinct1 | 10 | 10 | 0 | 0.5000 | 0.5000 |
| controlled_vs_candidate_no_context | length_score | 16 | 4 | 0 | 0.8000 | 0.8000 |
| controlled_vs_candidate_no_context | sentence_score | 10 | 3 | 7 | 0.6750 | 0.7692 |
| controlled_vs_candidate_no_context | overall_quality | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 4 | 6 | 10 | 0.4500 | 0.4000 |
| controlled_alt_vs_controlled_default | persona_consistency | 4 | 2 | 14 | 0.5500 | 0.6667 |
| controlled_alt_vs_controlled_default | naturalness | 3 | 7 | 10 | 0.4000 | 0.3000 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 1 | 5 | 14 | 0.4000 | 0.1667 |
| controlled_alt_vs_controlled_default | context_overlap | 4 | 5 | 11 | 0.4750 | 0.4444 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 2 | 1 | 17 | 0.5250 | 0.6667 |
| controlled_alt_vs_controlled_default | persona_style | 3 | 1 | 16 | 0.5500 | 0.7500 |
| controlled_alt_vs_controlled_default | distinct1 | 5 | 5 | 10 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | length_score | 3 | 7 | 10 | 0.4000 | 0.3000 |
| controlled_alt_vs_controlled_default | sentence_score | 5 | 2 | 13 | 0.5750 | 0.7143 |
| controlled_alt_vs_controlled_default | overall_quality | 5 | 5 | 10 | 0.5000 | 0.5000 |
| controlled_alt_vs_proposed_raw | context_relevance | 18 | 2 | 0 | 0.9000 | 0.9000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 12 | 1 | 7 | 0.7750 | 0.9231 |
| controlled_alt_vs_proposed_raw | naturalness | 14 | 6 | 0 | 0.7000 | 0.7000 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 18 | 2 | 0 | 0.9000 | 0.9000 |
| controlled_alt_vs_proposed_raw | context_overlap | 17 | 3 | 0 | 0.8500 | 0.8500 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 12 | 0 | 8 | 0.8000 | 1.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 6 | 1 | 13 | 0.6250 | 0.8571 |
| controlled_alt_vs_proposed_raw | distinct1 | 12 | 8 | 0 | 0.6000 | 0.6000 |
| controlled_alt_vs_proposed_raw | length_score | 13 | 6 | 1 | 0.6750 | 0.6842 |
| controlled_alt_vs_proposed_raw | sentence_score | 13 | 2 | 5 | 0.7750 | 0.8667 |
| controlled_alt_vs_proposed_raw | overall_quality | 18 | 2 | 0 | 0.9000 | 0.9000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 11 | 1 | 8 | 0.7500 | 0.9167 |
| controlled_alt_vs_candidate_no_context | naturalness | 14 | 6 | 0 | 0.7000 | 0.7000 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 11 | 1 | 8 | 0.7500 | 0.9167 |
| controlled_alt_vs_candidate_no_context | persona_style | 6 | 0 | 14 | 0.6500 | 1.0000 |
| controlled_alt_vs_candidate_no_context | distinct1 | 12 | 8 | 0 | 0.6000 | 0.6000 |
| controlled_alt_vs_candidate_no_context | length_score | 15 | 5 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_candidate_no_context | sentence_score | 12 | 2 | 6 | 0.7500 | 0.8571 |
| controlled_alt_vs_candidate_no_context | overall_quality | 20 | 0 | 0 | 1.0000 | 1.0000 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.0000 | 0.3500 | 0.2500 |
| proposed_contextual_controlled_alt | 0.0000 | 0.0000 | 0.0000 | 0.3500 | 0.2500 |
| proposed_contextual | 0.0000 | 0.0000 | 0.5500 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.5500 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `8`
- Unique template signatures: `18`
- Template signature ratio: `0.9000`
- Effective sample size by source clustering: `6.67`
- Effective sample size by template-signature clustering: `16.67`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: unavailable (disabled_by_flag).

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
# Proposal Alignment Evaluation Report

- Run ID: `20260305T235840Z`
- Generated: `2026-03-06T00:01:20.666708+00:00`
- Scenarios: `artifacts\proposal_control_tuning\auto_tune_v4\20260305T235104Z\train_runs\trial_003\seed_29\20260305T235840Z\scenarios.jsonl`
- Scenario count: `16`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_tuned`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2380 (0.1938, 0.2853) | 0.3127 (0.2413, 0.3872) | 0.9159 (0.8946, 0.9344) | 0.4001 (0.3693, 0.4334) | n/a |
| proposed_contextual_controlled_tuned | 0.2574 (0.2031, 0.3180) | 0.3476 (0.2623, 0.4395) | 0.9121 (0.8885, 0.9317) | 0.4217 (0.3797, 0.4636) | n/a |
| proposed_contextual | 0.0834 (0.0396, 0.1406) | 0.1831 (0.1291, 0.2470) | 0.8295 (0.7913, 0.8665) | 0.2639 (0.2225, 0.3094) | n/a |
| candidate_no_context | 0.0286 (0.0133, 0.0491) | 0.2021 (0.1201, 0.2922) | 0.8353 (0.7904, 0.8796) | 0.2460 (0.2086, 0.2863) | n/a |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0547 | 1.9117 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0189 | -0.0937 |
| proposed_vs_candidate_no_context | naturalness | -0.0057 | -0.0069 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0739 | 3.2500 |
| proposed_vs_candidate_no_context | context_overlap | 0.0101 | 0.2375 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0149 | -0.1429 |
| proposed_vs_candidate_no_context | persona_style | -0.0352 | -0.0592 |
| proposed_vs_candidate_no_context | distinct1 | 0.0083 | 0.0089 |
| proposed_vs_candidate_no_context | length_score | -0.0563 | -0.1385 |
| proposed_vs_candidate_no_context | sentence_score | 0.0219 | 0.0280 |
| proposed_vs_candidate_no_context | overall_quality | 0.0179 | 0.0727 |
| controlled_vs_proposed_raw | context_relevance | 0.1546 | 1.8548 |
| controlled_vs_proposed_raw | persona_consistency | 0.1295 | 0.7073 |
| controlled_vs_proposed_raw | naturalness | 0.0863 | 0.1041 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2011 | 2.0824 |
| controlled_vs_proposed_raw | context_overlap | 0.0460 | 0.8773 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1375 | 1.5400 |
| controlled_vs_proposed_raw | persona_style | 0.0977 | 0.1748 |
| controlled_vs_proposed_raw | distinct1 | -0.0042 | -0.0044 |
| controlled_vs_proposed_raw | length_score | 0.3542 | 1.0119 |
| controlled_vs_proposed_raw | sentence_score | 0.1750 | 0.2179 |
| controlled_vs_proposed_raw | overall_quality | 0.1361 | 0.5157 |
| controlled_vs_candidate_no_context | context_relevance | 0.2093 | 7.3122 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1106 | 0.5473 |
| controlled_vs_candidate_no_context | naturalness | 0.0806 | 0.0965 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.2750 | 12.1000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0561 | 1.3233 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1226 | 1.1771 |
| controlled_vs_candidate_no_context | persona_style | 0.0625 | 0.1053 |
| controlled_vs_candidate_no_context | distinct1 | 0.0042 | 0.0044 |
| controlled_vs_candidate_no_context | length_score | 0.2979 | 0.7333 |
| controlled_vs_candidate_no_context | sentence_score | 0.1969 | 0.2520 |
| controlled_vs_candidate_no_context | overall_quality | 0.1540 | 0.6259 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0195 | 0.0818 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0350 | 0.1118 |
| controlled_alt_vs_controlled_default | naturalness | -0.0038 | -0.0041 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0256 | 0.0859 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0052 | 0.0530 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0476 | 0.2100 |
| controlled_alt_vs_controlled_default | persona_style | -0.0156 | -0.0238 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0042 | -0.0045 |
| controlled_alt_vs_controlled_default | length_score | -0.0229 | -0.0325 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0219 | 0.0224 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0216 | 0.0541 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1741 | 2.0883 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1645 | 0.8982 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0826 | 0.0995 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2267 | 2.3471 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0513 | 0.9769 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1851 | 2.0733 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0820 | 0.1469 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0084 | -0.0089 |
| controlled_alt_vs_proposed_raw | length_score | 0.3313 | 0.9464 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1969 | 0.2451 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1578 | 0.5978 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2288 | 7.9922 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1456 | 0.7203 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0768 | 0.0920 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3006 | 13.2250 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0613 | 1.4465 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1702 | 1.6343 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0469 | 0.0789 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0001 | -0.0001 |
| controlled_alt_vs_candidate_no_context | length_score | 0.2750 | 0.6769 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.2188 | 0.2800 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1757 | 0.7139 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0547 | (0.0173, 0.1004) | 0.0003 | 0.0547 | (0.0202, 0.0879) | 0.0007 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0189 | (-0.1236, 0.0976) | 0.6347 | -0.0189 | (-0.0991, 0.0908) | 0.6667 |
| proposed_vs_candidate_no_context | naturalness | -0.0057 | (-0.0470, 0.0374) | 0.5840 | -0.0057 | (-0.0515, 0.0536) | 0.6407 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0739 | (0.0227, 0.1307) | 0.0003 | 0.0739 | (0.0321, 0.1136) | 0.0000 |
| proposed_vs_candidate_no_context | context_overlap | 0.0101 | (-0.0064, 0.0270) | 0.1170 | 0.0101 | (-0.0065, 0.0283) | 0.1480 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0149 | (-0.1369, 0.1190) | 0.6123 | -0.0149 | (-0.1122, 0.1161) | 0.6257 |
| proposed_vs_candidate_no_context | persona_style | -0.0352 | (-0.1250, 0.0312) | 0.8567 | -0.0352 | (-0.0938, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | distinct1 | 0.0083 | (-0.0040, 0.0213) | 0.0843 | 0.0083 | (-0.0033, 0.0243) | 0.1143 |
| proposed_vs_candidate_no_context | length_score | -0.0563 | (-0.2188, 0.1042) | 0.7533 | -0.0563 | (-0.2452, 0.1578) | 0.7243 |
| proposed_vs_candidate_no_context | sentence_score | 0.0219 | (-0.0656, 0.1094) | 0.3973 | 0.0219 | (-0.0875, 0.1400) | 0.4207 |
| proposed_vs_candidate_no_context | overall_quality | 0.0179 | (-0.0259, 0.0681) | 0.2343 | 0.0179 | (-0.0133, 0.0682) | 0.3213 |
| controlled_vs_proposed_raw | context_relevance | 0.1546 | (0.0983, 0.2036) | 0.0000 | 0.1546 | (0.1267, 0.1855) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.1295 | (0.0713, 0.1926) | 0.0000 | 0.1295 | (0.0429, 0.2161) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0863 | (0.0315, 0.1363) | 0.0010 | 0.0863 | (0.0223, 0.1439) | 0.0037 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2011 | (0.1238, 0.2693) | 0.0000 | 0.2011 | (0.1636, 0.2412) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0460 | (0.0302, 0.0624) | 0.0000 | 0.0460 | (0.0282, 0.0647) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1375 | (0.0714, 0.2140) | 0.0000 | 0.1375 | (0.0458, 0.2381) | 0.0003 |
| controlled_vs_proposed_raw | persona_style | 0.0977 | (0.0000, 0.2207) | 0.0357 | 0.0977 | (0.0000, 0.2326) | 0.0783 |
| controlled_vs_proposed_raw | distinct1 | -0.0042 | (-0.0210, 0.0110) | 0.6763 | -0.0042 | (-0.0179, 0.0106) | 0.6923 |
| controlled_vs_proposed_raw | length_score | 0.3542 | (0.1375, 0.5563) | 0.0000 | 0.3542 | (0.1042, 0.5875) | 0.0050 |
| controlled_vs_proposed_raw | sentence_score | 0.1750 | (0.0875, 0.2625) | 0.0000 | 0.1750 | (0.0700, 0.2800) | 0.0000 |
| controlled_vs_proposed_raw | overall_quality | 0.1361 | (0.0976, 0.1748) | 0.0000 | 0.1361 | (0.0919, 0.1731) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2093 | (0.1626, 0.2578) | 0.0000 | 0.2093 | (0.2007, 0.2207) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1106 | (-0.0098, 0.2270) | 0.0370 | 0.1106 | (0.0460, 0.1751) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.0806 | (0.0340, 0.1283) | 0.0010 | 0.0806 | (0.0234, 0.1409) | 0.0010 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.2750 | (0.2148, 0.3420) | 0.0000 | 0.2750 | (0.2588, 0.2922) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0561 | (0.0407, 0.0731) | 0.0000 | 0.0561 | (0.0457, 0.0674) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1226 | (-0.0089, 0.2637) | 0.0387 | 0.1226 | (0.0444, 0.2041) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.0625 | (-0.0156, 0.1641) | 0.0807 | 0.0625 | (0.0000, 0.1325) | 0.0750 |
| controlled_vs_candidate_no_context | distinct1 | 0.0042 | (-0.0135, 0.0202) | 0.3110 | 0.0042 | (-0.0162, 0.0236) | 0.3533 |
| controlled_vs_candidate_no_context | length_score | 0.2979 | (0.1145, 0.4750) | 0.0003 | 0.2979 | (0.0795, 0.5314) | 0.0000 |
| controlled_vs_candidate_no_context | sentence_score | 0.1969 | (0.0875, 0.2844) | 0.0020 | 0.1969 | (0.0808, 0.2882) | 0.0007 |
| controlled_vs_candidate_no_context | overall_quality | 0.1540 | (0.1037, 0.2031) | 0.0000 | 0.1540 | (0.1318, 0.1745) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0195 | (-0.0481, 0.0871) | 0.2980 | 0.0195 | (-0.0405, 0.0871) | 0.2800 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0350 | (-0.0132, 0.0893) | 0.0900 | 0.0350 | (-0.0235, 0.0952) | 0.1357 |
| controlled_alt_vs_controlled_default | naturalness | -0.0038 | (-0.0275, 0.0130) | 0.6023 | -0.0038 | (-0.0169, 0.0074) | 0.6800 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0256 | (-0.0625, 0.1165) | 0.2970 | 0.0256 | (-0.0545, 0.1077) | 0.2987 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0052 | (-0.0126, 0.0246) | 0.2850 | 0.0052 | (-0.0172, 0.0269) | 0.3257 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0476 | (-0.0074, 0.1131) | 0.0377 | 0.0476 | (-0.0204, 0.1204) | 0.0923 |
| controlled_alt_vs_controlled_default | persona_style | -0.0156 | (-0.0469, 0.0000) | 1.0000 | -0.0156 | (-0.0500, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0042 | (-0.0230, 0.0124) | 0.6650 | -0.0042 | (-0.0377, 0.0162) | 0.6347 |
| controlled_alt_vs_controlled_default | length_score | -0.0229 | (-0.1479, 0.0625) | 0.6217 | -0.0229 | (-0.1204, 0.0622) | 0.6437 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0219 | (0.0000, 0.0656) | 0.3697 | 0.0219 | (0.0000, 0.0808) | 0.3173 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0216 | (-0.0220, 0.0636) | 0.1657 | 0.0216 | (-0.0224, 0.0627) | 0.1490 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1741 | (0.1137, 0.2407) | 0.0000 | 0.1741 | (0.1046, 0.2279) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1645 | (0.0758, 0.2538) | 0.0000 | 0.1645 | (0.0224, 0.3066) | 0.0037 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0826 | (0.0338, 0.1312) | 0.0017 | 0.0826 | (0.0177, 0.1475) | 0.0027 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2267 | (0.1386, 0.3159) | 0.0000 | 0.2267 | (0.1329, 0.2991) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0513 | (0.0339, 0.0672) | 0.0000 | 0.0513 | (0.0344, 0.0690) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1851 | (0.0908, 0.2884) | 0.0000 | 0.1851 | (0.0280, 0.3503) | 0.0067 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0820 | (0.0000, 0.2051) | 0.1173 | 0.0820 | (0.0000, 0.2316) | 0.3343 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0084 | (-0.0298, 0.0115) | 0.7873 | -0.0084 | (-0.0362, 0.0125) | 0.7503 |
| controlled_alt_vs_proposed_raw | length_score | 0.3312 | (0.1313, 0.5250) | 0.0010 | 0.3312 | (0.0333, 0.6292) | 0.0120 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1969 | (0.1094, 0.2844) | 0.0000 | 0.1969 | (0.0750, 0.3062) | 0.0003 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1578 | (0.1038, 0.2077) | 0.0000 | 0.1578 | (0.0795, 0.2241) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2288 | (0.1779, 0.2852) | 0.0000 | 0.2288 | (0.1579, 0.2953) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1456 | (0.0358, 0.2529) | 0.0063 | 0.1456 | (0.0518, 0.2513) | 0.0003 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0768 | (0.0304, 0.1228) | 0.0003 | 0.0768 | (0.0104, 0.1433) | 0.0073 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3006 | (0.2341, 0.3739) | 0.0000 | 0.3006 | (0.2061, 0.3975) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0613 | (0.0430, 0.0818) | 0.0000 | 0.0613 | (0.0405, 0.0842) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1702 | (0.0372, 0.2946) | 0.0057 | 0.1702 | (0.0619, 0.3053) | 0.0003 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0469 | (-0.0312, 0.1445) | 0.1683 | 0.0469 | (0.0000, 0.1250) | 0.3370 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0001 | (-0.0211, 0.0206) | 0.4917 | -0.0001 | (-0.0314, 0.0299) | 0.5133 |
| controlled_alt_vs_candidate_no_context | length_score | 0.2750 | (0.0854, 0.4458) | 0.0007 | 0.2750 | (0.0078, 0.5333) | 0.0193 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.2188 | (0.1312, 0.3062) | 0.0000 | 0.2188 | (0.1250, 0.3088) | 0.0000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1757 | (0.1290, 0.2224) | 0.0000 | 0.1757 | (0.1291, 0.2177) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 7 | 3 | 6 | 0.6250 | 0.7000 |
| proposed_vs_candidate_no_context | persona_consistency | 2 | 5 | 9 | 0.4062 | 0.2857 |
| proposed_vs_candidate_no_context | naturalness | 5 | 5 | 6 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 6 | 0 | 10 | 0.6875 | 1.0000 |
| proposed_vs_candidate_no_context | context_overlap | 6 | 4 | 6 | 0.5625 | 0.6000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 2 | 4 | 10 | 0.4375 | 0.3333 |
| proposed_vs_candidate_no_context | persona_style | 1 | 2 | 13 | 0.4688 | 0.3333 |
| proposed_vs_candidate_no_context | distinct1 | 7 | 1 | 8 | 0.6875 | 0.8750 |
| proposed_vs_candidate_no_context | length_score | 4 | 5 | 7 | 0.4688 | 0.4444 |
| proposed_vs_candidate_no_context | sentence_score | 3 | 2 | 11 | 0.5312 | 0.6000 |
| proposed_vs_candidate_no_context | overall_quality | 5 | 5 | 6 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | context_relevance | 15 | 1 | 0 | 0.9375 | 0.9375 |
| controlled_vs_proposed_raw | persona_consistency | 11 | 0 | 5 | 0.8438 | 1.0000 |
| controlled_vs_proposed_raw | naturalness | 11 | 5 | 0 | 0.6875 | 0.6875 |
| controlled_vs_proposed_raw | context_keyword_coverage | 13 | 1 | 2 | 0.8750 | 0.9286 |
| controlled_vs_proposed_raw | context_overlap | 14 | 2 | 0 | 0.8750 | 0.8750 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 10 | 0 | 6 | 0.8125 | 1.0000 |
| controlled_vs_proposed_raw | persona_style | 3 | 0 | 13 | 0.5938 | 1.0000 |
| controlled_vs_proposed_raw | distinct1 | 7 | 6 | 3 | 0.5312 | 0.5385 |
| controlled_vs_proposed_raw | length_score | 11 | 5 | 0 | 0.6875 | 0.6875 |
| controlled_vs_proposed_raw | sentence_score | 8 | 0 | 8 | 0.7500 | 1.0000 |
| controlled_vs_proposed_raw | overall_quality | 15 | 1 | 0 | 0.9375 | 0.9375 |
| controlled_vs_candidate_no_context | context_relevance | 16 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 9 | 4 | 3 | 0.6562 | 0.6923 |
| controlled_vs_candidate_no_context | naturalness | 11 | 5 | 0 | 0.6875 | 0.6875 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 16 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 15 | 1 | 0 | 0.9375 | 0.9375 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 9 | 4 | 3 | 0.6562 | 0.6923 |
| controlled_vs_candidate_no_context | persona_style | 4 | 1 | 11 | 0.5938 | 0.8000 |
| controlled_vs_candidate_no_context | distinct1 | 10 | 5 | 1 | 0.6562 | 0.6667 |
| controlled_vs_candidate_no_context | length_score | 12 | 4 | 0 | 0.7500 | 0.7500 |
| controlled_vs_candidate_no_context | sentence_score | 10 | 1 | 5 | 0.7812 | 0.9091 |
| controlled_vs_candidate_no_context | overall_quality | 14 | 2 | 0 | 0.8750 | 0.8750 |
| controlled_alt_vs_controlled_default | context_relevance | 5 | 4 | 7 | 0.5312 | 0.5556 |
| controlled_alt_vs_controlled_default | persona_consistency | 4 | 1 | 11 | 0.5938 | 0.8000 |
| controlled_alt_vs_controlled_default | naturalness | 6 | 3 | 7 | 0.5938 | 0.6667 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 5 | 4 | 7 | 0.5312 | 0.5556 |
| controlled_alt_vs_controlled_default | context_overlap | 6 | 3 | 7 | 0.5938 | 0.6667 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 4 | 1 | 11 | 0.5938 | 0.8000 |
| controlled_alt_vs_controlled_default | persona_style | 0 | 1 | 15 | 0.4688 | 0.0000 |
| controlled_alt_vs_controlled_default | distinct1 | 5 | 4 | 7 | 0.5312 | 0.5556 |
| controlled_alt_vs_controlled_default | length_score | 5 | 3 | 8 | 0.5625 | 0.6250 |
| controlled_alt_vs_controlled_default | sentence_score | 1 | 0 | 15 | 0.5312 | 1.0000 |
| controlled_alt_vs_controlled_default | overall_quality | 6 | 3 | 7 | 0.5938 | 0.6667 |
| controlled_alt_vs_proposed_raw | context_relevance | 15 | 1 | 0 | 0.9375 | 0.9375 |
| controlled_alt_vs_proposed_raw | persona_consistency | 10 | 1 | 5 | 0.7812 | 0.9091 |
| controlled_alt_vs_proposed_raw | naturalness | 11 | 4 | 1 | 0.7188 | 0.7333 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 15 | 1 | 0 | 0.9375 | 0.9375 |
| controlled_alt_vs_proposed_raw | context_overlap | 14 | 2 | 0 | 0.8750 | 0.8750 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 10 | 1 | 5 | 0.7812 | 0.9091 |
| controlled_alt_vs_proposed_raw | persona_style | 2 | 0 | 14 | 0.5625 | 1.0000 |
| controlled_alt_vs_proposed_raw | distinct1 | 8 | 7 | 1 | 0.5312 | 0.5333 |
| controlled_alt_vs_proposed_raw | length_score | 11 | 4 | 1 | 0.7188 | 0.7333 |
| controlled_alt_vs_proposed_raw | sentence_score | 9 | 0 | 7 | 0.7812 | 1.0000 |
| controlled_alt_vs_proposed_raw | overall_quality | 15 | 1 | 0 | 0.9375 | 0.9375 |
| controlled_alt_vs_candidate_no_context | context_relevance | 16 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 10 | 2 | 4 | 0.7500 | 0.8333 |
| controlled_alt_vs_candidate_no_context | naturalness | 10 | 6 | 0 | 0.6250 | 0.6250 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 15 | 0 | 1 | 0.9688 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 16 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 10 | 2 | 4 | 0.7500 | 0.8333 |
| controlled_alt_vs_candidate_no_context | persona_style | 3 | 1 | 12 | 0.5625 | 0.7500 |
| controlled_alt_vs_candidate_no_context | distinct1 | 8 | 8 | 0 | 0.5000 | 0.5000 |
| controlled_alt_vs_candidate_no_context | length_score | 12 | 4 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_candidate_no_context | sentence_score | 10 | 0 | 6 | 0.8125 | 1.0000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 15 | 1 | 0 | 0.9375 | 0.9375 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.5000 | 0.4375 | 0.5625 |
| proposed_contextual_controlled_tuned | 0.0000 | 0.0000 | 0.6875 | 0.1875 | 0.8125 |
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
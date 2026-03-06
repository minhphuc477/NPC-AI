# Proposal Alignment Evaluation Report

- Run ID: `20260306T000358Z`
- Generated: `2026-03-06T00:06:33.480401+00:00`
- Scenarios: `artifacts\proposal_control_tuning\auto_tune_v4\20260305T235104Z\train_runs\trial_005\seed_29\20260306T000358Z\scenarios.jsonl`
- Scenario count: `16`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_tuned`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2407 (0.1951, 0.2869) | 0.3620 (0.2863, 0.4432) | 0.9007 (0.8645, 0.9334) | 0.4169 (0.3883, 0.4451) | n/a |
| proposed_contextual_controlled_tuned | 0.2520 (0.1941, 0.3128) | 0.4491 (0.3540, 0.5539) | 0.9045 (0.8761, 0.9278) | 0.4551 (0.4198, 0.4907) | n/a |
| proposed_contextual | 0.0666 (0.0225, 0.1237) | 0.2055 (0.1370, 0.2805) | 0.8187 (0.7799, 0.8606) | 0.2629 (0.2196, 0.3124) | n/a |
| candidate_no_context | 0.0326 (0.0154, 0.0545) | 0.2208 (0.1418, 0.3107) | 0.8282 (0.7869, 0.8710) | 0.2538 (0.2119, 0.3006) | n/a |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0340 | 1.0439 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0153 | -0.0693 |
| proposed_vs_candidate_no_context | naturalness | -0.0096 | -0.0116 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0455 | 1.6000 |
| proposed_vs_candidate_no_context | context_overlap | 0.0073 | 0.1727 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0045 | -0.0337 |
| proposed_vs_candidate_no_context | persona_style | -0.0586 | -0.1020 |
| proposed_vs_candidate_no_context | distinct1 | 0.0013 | 0.0014 |
| proposed_vs_candidate_no_context | length_score | -0.0396 | -0.1056 |
| proposed_vs_candidate_no_context | sentence_score | -0.0219 | -0.0280 |
| proposed_vs_candidate_no_context | overall_quality | 0.0091 | 0.0359 |
| controlled_vs_proposed_raw | context_relevance | 0.1741 | 2.6150 |
| controlled_vs_proposed_raw | persona_consistency | 0.1565 | 0.7613 |
| controlled_vs_proposed_raw | naturalness | 0.0820 | 0.1002 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2244 | 3.0385 |
| controlled_vs_proposed_raw | context_overlap | 0.0568 | 1.1443 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1661 | 1.2977 |
| controlled_vs_proposed_raw | persona_style | 0.1180 | 0.2288 |
| controlled_vs_proposed_raw | distinct1 | 0.0007 | 0.0007 |
| controlled_vs_proposed_raw | length_score | 0.3104 | 0.9255 |
| controlled_vs_proposed_raw | sentence_score | 0.1969 | 0.2593 |
| controlled_vs_proposed_raw | overall_quality | 0.1540 | 0.5855 |
| controlled_vs_candidate_no_context | context_relevance | 0.2081 | 6.3886 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1412 | 0.6393 |
| controlled_vs_candidate_no_context | naturalness | 0.0725 | 0.0875 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.2699 | 9.5000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0641 | 1.5147 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1616 | 1.2202 |
| controlled_vs_candidate_no_context | persona_style | 0.0594 | 0.1034 |
| controlled_vs_candidate_no_context | distinct1 | 0.0020 | 0.0021 |
| controlled_vs_candidate_no_context | length_score | 0.2708 | 0.7222 |
| controlled_vs_candidate_no_context | sentence_score | 0.1750 | 0.2240 |
| controlled_vs_candidate_no_context | overall_quality | 0.1631 | 0.6424 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0112 | 0.0467 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0871 | 0.2407 |
| controlled_alt_vs_controlled_default | naturalness | 0.0038 | 0.0042 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0108 | 0.0362 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0123 | 0.1154 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.1071 | 0.3644 |
| controlled_alt_vs_controlled_default | persona_style | 0.0070 | 0.0111 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0130 | 0.0139 |
| controlled_alt_vs_controlled_default | length_score | 0.0146 | 0.0226 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0438 | -0.0458 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0382 | 0.0917 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1854 | 2.7838 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.2436 | 1.1852 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0858 | 0.1048 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2352 | 3.1846 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0691 | 1.3918 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.2732 | 2.1349 |
| controlled_alt_vs_proposed_raw | persona_style | 0.1250 | 0.2424 |
| controlled_alt_vs_proposed_raw | distinct1 | 0.0137 | 0.0146 |
| controlled_alt_vs_proposed_raw | length_score | 0.3250 | 0.9689 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1531 | 0.2016 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1922 | 0.7309 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2194 | 6.7336 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.2283 | 1.0339 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0762 | 0.0920 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.2807 | 9.8800 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0764 | 1.8049 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.2687 | 2.0292 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0664 | 0.1156 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0150 | 0.0160 |
| controlled_alt_vs_candidate_no_context | length_score | 0.2854 | 0.7611 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1312 | 0.1680 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.2013 | 0.7930 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0340 | (0.0010, 0.0836) | 0.0130 | 0.0340 | (-0.0006, 0.0733) | 0.0757 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0153 | (-0.0820, 0.0639) | 0.6760 | -0.0153 | (-0.0955, 0.0463) | 0.6453 |
| proposed_vs_candidate_no_context | naturalness | -0.0096 | (-0.0409, 0.0153) | 0.7420 | -0.0096 | (-0.0366, 0.0190) | 0.7377 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0455 | (0.0000, 0.1136) | 0.0400 | 0.0455 | (0.0000, 0.0970) | 0.0843 |
| proposed_vs_candidate_no_context | context_overlap | 0.0073 | (-0.0049, 0.0234) | 0.1813 | 0.0073 | (-0.0042, 0.0235) | 0.3387 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0045 | (-0.0818, 0.0893) | 0.5770 | -0.0045 | (-0.1071, 0.0756) | 0.6307 |
| proposed_vs_candidate_no_context | persona_style | -0.0586 | (-0.1797, 0.0312) | 0.8880 | -0.0586 | (-0.1285, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | distinct1 | 0.0013 | (-0.0230, 0.0234) | 0.4237 | 0.0013 | (-0.0095, 0.0148) | 0.4280 |
| proposed_vs_candidate_no_context | length_score | -0.0396 | (-0.1521, 0.0646) | 0.7433 | -0.0396 | (-0.1600, 0.1071) | 0.7010 |
| proposed_vs_candidate_no_context | sentence_score | -0.0219 | (-0.0875, 0.0437) | 0.8200 | -0.0219 | (-0.0824, 0.0538) | 0.8230 |
| proposed_vs_candidate_no_context | overall_quality | 0.0091 | (-0.0200, 0.0445) | 0.3130 | 0.0091 | (-0.0100, 0.0285) | 0.1637 |
| controlled_vs_proposed_raw | context_relevance | 0.1741 | (0.1451, 0.2060) | 0.0000 | 0.1741 | (0.1547, 0.1875) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.1565 | (0.0605, 0.2565) | 0.0000 | 0.1565 | (0.0321, 0.2578) | 0.0117 |
| controlled_vs_proposed_raw | naturalness | 0.0820 | (0.0459, 0.1223) | 0.0000 | 0.0820 | (0.0426, 0.1160) | 0.0000 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2244 | (0.1807, 0.2705) | 0.0000 | 0.2244 | (0.1888, 0.2500) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0568 | (0.0381, 0.0772) | 0.0000 | 0.0568 | (0.0395, 0.0821) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1661 | (0.0551, 0.2849) | 0.0007 | 0.1661 | (0.0095, 0.2931) | 0.0137 |
| controlled_vs_proposed_raw | persona_style | 0.1180 | (0.0000, 0.2480) | 0.0263 | 0.1180 | (-0.0199, 0.3333) | 0.3243 |
| controlled_vs_proposed_raw | distinct1 | 0.0007 | (-0.0217, 0.0211) | 0.4680 | 0.0007 | (-0.0165, 0.0167) | 0.4550 |
| controlled_vs_proposed_raw | length_score | 0.3104 | (0.1688, 0.4751) | 0.0000 | 0.3104 | (0.1718, 0.4500) | 0.0000 |
| controlled_vs_proposed_raw | sentence_score | 0.1969 | (0.1094, 0.2844) | 0.0000 | 0.1969 | (0.1077, 0.2579) | 0.0000 |
| controlled_vs_proposed_raw | overall_quality | 0.1540 | (0.1148, 0.1950) | 0.0000 | 0.1540 | (0.0886, 0.1968) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2081 | (0.1711, 0.2534) | 0.0000 | 0.2081 | (0.1648, 0.2582) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1412 | (0.0160, 0.2647) | 0.0150 | 0.1412 | (0.0130, 0.2379) | 0.0187 |
| controlled_vs_candidate_no_context | naturalness | 0.0725 | (0.0379, 0.1069) | 0.0000 | 0.0725 | (0.0319, 0.1137) | 0.0000 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.2699 | (0.2199, 0.3239) | 0.0000 | 0.2699 | (0.2102, 0.3396) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0641 | (0.0452, 0.0862) | 0.0000 | 0.0641 | (0.0500, 0.0811) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1616 | (0.0250, 0.3084) | 0.0090 | 0.1616 | (-0.0018, 0.2732) | 0.0273 |
| controlled_vs_candidate_no_context | persona_style | 0.0594 | (-0.0610, 0.2078) | 0.1743 | 0.0594 | (-0.0518, 0.2169) | 0.3207 |
| controlled_vs_candidate_no_context | distinct1 | 0.0020 | (-0.0215, 0.0227) | 0.4247 | 0.0020 | (-0.0136, 0.0158) | 0.4023 |
| controlled_vs_candidate_no_context | length_score | 0.2708 | (0.1313, 0.4146) | 0.0000 | 0.2708 | (0.0956, 0.4414) | 0.0000 |
| controlled_vs_candidate_no_context | sentence_score | 0.1750 | (0.0656, 0.2625) | 0.0013 | 0.1750 | (0.1250, 0.2265) | 0.0000 |
| controlled_vs_candidate_no_context | overall_quality | 0.1631 | (0.1144, 0.2063) | 0.0000 | 0.1631 | (0.1083, 0.2020) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0112 | (-0.0370, 0.0592) | 0.3123 | 0.0112 | (-0.0586, 0.0560) | 0.3273 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0871 | (-0.0258, 0.2083) | 0.0667 | 0.0871 | (0.0108, 0.1625) | 0.0120 |
| controlled_alt_vs_controlled_default | naturalness | 0.0038 | (-0.0471, 0.0541) | 0.4473 | 0.0038 | (-0.0368, 0.0399) | 0.4370 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0108 | (-0.0466, 0.0682) | 0.3577 | 0.0108 | (-0.0758, 0.0617) | 0.3950 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0123 | (-0.0167, 0.0466) | 0.2233 | 0.0123 | (-0.0237, 0.0404) | 0.2547 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.1071 | (-0.0363, 0.2688) | 0.0780 | 0.1071 | (0.0056, 0.2063) | 0.0180 |
| controlled_alt_vs_controlled_default | persona_style | 0.0070 | (0.0000, 0.0211) | 0.3490 | 0.0070 | (0.0000, 0.0225) | 0.3417 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0130 | (-0.0090, 0.0382) | 0.1417 | 0.0130 | (-0.0077, 0.0272) | 0.0790 |
| controlled_alt_vs_controlled_default | length_score | 0.0146 | (-0.1917, 0.2313) | 0.4617 | 0.0146 | (-0.1844, 0.1933) | 0.4200 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0437 | (-0.1531, 0.0656) | 0.8540 | -0.0437 | (-0.1250, 0.0412) | 0.9127 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0382 | (-0.0043, 0.0863) | 0.0387 | 0.0382 | (-0.0060, 0.0794) | 0.0580 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1854 | (0.1238, 0.2431) | 0.0000 | 0.1854 | (0.1045, 0.2407) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.2436 | (0.1401, 0.3568) | 0.0000 | 0.2436 | (0.1048, 0.3615) | 0.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0858 | (0.0326, 0.1339) | 0.0007 | 0.0858 | (0.0250, 0.1277) | 0.0043 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2352 | (0.1523, 0.3091) | 0.0000 | 0.2352 | (0.1259, 0.3100) | 0.0003 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0691 | (0.0423, 0.1000) | 0.0000 | 0.0691 | (0.0544, 0.0866) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.2732 | (0.1446, 0.4113) | 0.0000 | 0.2732 | (0.0952, 0.4423) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 0.1250 | (0.0039, 0.2500) | 0.0113 | 0.1250 | (0.0000, 0.3529) | 0.3343 |
| controlled_alt_vs_proposed_raw | distinct1 | 0.0137 | (-0.0085, 0.0387) | 0.1270 | 0.0137 | (-0.0121, 0.0328) | 0.1320 |
| controlled_alt_vs_proposed_raw | length_score | 0.3250 | (0.1291, 0.5229) | 0.0000 | 0.3250 | (0.0976, 0.5019) | 0.0047 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1531 | (0.0219, 0.2844) | 0.0243 | 0.1531 | (0.0000, 0.2722) | 0.0440 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1922 | (0.1321, 0.2514) | 0.0000 | 0.1922 | (0.1009, 0.2632) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2194 | (0.1545, 0.2919) | 0.0000 | 0.2194 | (0.1330, 0.2792) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.2283 | (0.0936, 0.3655) | 0.0003 | 0.2283 | (0.0796, 0.3440) | 0.0010 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0762 | (0.0224, 0.1251) | 0.0030 | 0.0762 | (0.0240, 0.1095) | 0.0053 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.2807 | (0.1966, 0.3716) | 0.0000 | 0.2807 | (0.1688, 0.3697) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0764 | (0.0493, 0.1089) | 0.0000 | 0.0764 | (0.0517, 0.0936) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.2687 | (0.1104, 0.4363) | 0.0003 | 0.2687 | (0.0850, 0.4232) | 0.0027 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0664 | (-0.0527, 0.2051) | 0.1757 | 0.0664 | (-0.0500, 0.2169) | 0.3370 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0150 | (-0.0005, 0.0300) | 0.0270 | 0.0150 | (0.0024, 0.0253) | 0.0110 |
| controlled_alt_vs_candidate_no_context | length_score | 0.2854 | (0.0854, 0.4750) | 0.0020 | 0.2854 | (0.0538, 0.4333) | 0.0093 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1312 | (0.0000, 0.2406) | 0.0287 | 0.1312 | (0.0250, 0.2211) | 0.0093 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.2013 | (0.1353, 0.2649) | 0.0000 | 0.2013 | (0.1036, 0.2641) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 7 | 2 | 7 | 0.6562 | 0.7778 |
| proposed_vs_candidate_no_context | persona_consistency | 2 | 5 | 9 | 0.4062 | 0.2857 |
| proposed_vs_candidate_no_context | naturalness | 3 | 6 | 7 | 0.4062 | 0.3333 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 3 | 0 | 13 | 0.5938 | 1.0000 |
| proposed_vs_candidate_no_context | context_overlap | 5 | 4 | 7 | 0.5312 | 0.5556 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 1 | 3 | 12 | 0.4375 | 0.2500 |
| proposed_vs_candidate_no_context | persona_style | 1 | 3 | 12 | 0.4375 | 0.2500 |
| proposed_vs_candidate_no_context | distinct1 | 5 | 3 | 8 | 0.5625 | 0.6250 |
| proposed_vs_candidate_no_context | length_score | 2 | 6 | 8 | 0.3750 | 0.2500 |
| proposed_vs_candidate_no_context | sentence_score | 1 | 2 | 13 | 0.4688 | 0.3333 |
| proposed_vs_candidate_no_context | overall_quality | 4 | 5 | 7 | 0.4688 | 0.4444 |
| controlled_vs_proposed_raw | context_relevance | 16 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_proposed_raw | persona_consistency | 10 | 1 | 5 | 0.7812 | 0.9091 |
| controlled_vs_proposed_raw | naturalness | 14 | 2 | 0 | 0.8750 | 0.8750 |
| controlled_vs_proposed_raw | context_keyword_coverage | 16 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_proposed_raw | context_overlap | 15 | 1 | 0 | 0.9375 | 0.9375 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 10 | 1 | 5 | 0.7812 | 0.9091 |
| controlled_vs_proposed_raw | persona_style | 4 | 1 | 11 | 0.5938 | 0.8000 |
| controlled_vs_proposed_raw | distinct1 | 10 | 6 | 0 | 0.6250 | 0.6250 |
| controlled_vs_proposed_raw | length_score | 14 | 1 | 1 | 0.9062 | 0.9333 |
| controlled_vs_proposed_raw | sentence_score | 9 | 0 | 7 | 0.7812 | 1.0000 |
| controlled_vs_proposed_raw | overall_quality | 16 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_relevance | 16 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 11 | 3 | 2 | 0.7500 | 0.7857 |
| controlled_vs_candidate_no_context | naturalness | 14 | 2 | 0 | 0.8750 | 0.8750 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 16 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 16 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 10 | 3 | 3 | 0.7188 | 0.7692 |
| controlled_vs_candidate_no_context | persona_style | 3 | 2 | 11 | 0.5312 | 0.6000 |
| controlled_vs_candidate_no_context | distinct1 | 11 | 5 | 0 | 0.6875 | 0.6875 |
| controlled_vs_candidate_no_context | length_score | 13 | 3 | 0 | 0.8125 | 0.8125 |
| controlled_vs_candidate_no_context | sentence_score | 9 | 1 | 6 | 0.7500 | 0.9000 |
| controlled_vs_candidate_no_context | overall_quality | 15 | 1 | 0 | 0.9375 | 0.9375 |
| controlled_alt_vs_controlled_default | context_relevance | 6 | 7 | 3 | 0.4688 | 0.4615 |
| controlled_alt_vs_controlled_default | persona_consistency | 7 | 3 | 6 | 0.6250 | 0.7000 |
| controlled_alt_vs_controlled_default | naturalness | 6 | 8 | 2 | 0.4375 | 0.4286 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 6 | 5 | 5 | 0.5312 | 0.5455 |
| controlled_alt_vs_controlled_default | context_overlap | 7 | 6 | 3 | 0.5312 | 0.5385 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 7 | 3 | 6 | 0.6250 | 0.7000 |
| controlled_alt_vs_controlled_default | persona_style | 1 | 0 | 15 | 0.5312 | 1.0000 |
| controlled_alt_vs_controlled_default | distinct1 | 5 | 9 | 2 | 0.3750 | 0.3571 |
| controlled_alt_vs_controlled_default | length_score | 6 | 8 | 2 | 0.4375 | 0.4286 |
| controlled_alt_vs_controlled_default | sentence_score | 2 | 4 | 10 | 0.4375 | 0.3333 |
| controlled_alt_vs_controlled_default | overall_quality | 8 | 6 | 2 | 0.5625 | 0.5714 |
| controlled_alt_vs_proposed_raw | context_relevance | 14 | 2 | 0 | 0.8750 | 0.8750 |
| controlled_alt_vs_proposed_raw | persona_consistency | 13 | 1 | 2 | 0.8750 | 0.9286 |
| controlled_alt_vs_proposed_raw | naturalness | 11 | 5 | 0 | 0.6875 | 0.6875 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 14 | 1 | 1 | 0.9062 | 0.9333 |
| controlled_alt_vs_proposed_raw | context_overlap | 14 | 2 | 0 | 0.8750 | 0.8750 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 13 | 1 | 2 | 0.8750 | 0.9286 |
| controlled_alt_vs_proposed_raw | persona_style | 4 | 0 | 12 | 0.6250 | 1.0000 |
| controlled_alt_vs_proposed_raw | distinct1 | 9 | 6 | 1 | 0.5938 | 0.6000 |
| controlled_alt_vs_proposed_raw | length_score | 11 | 3 | 2 | 0.7500 | 0.7857 |
| controlled_alt_vs_proposed_raw | sentence_score | 10 | 3 | 3 | 0.7188 | 0.7692 |
| controlled_alt_vs_proposed_raw | overall_quality | 14 | 2 | 0 | 0.8750 | 0.8750 |
| controlled_alt_vs_candidate_no_context | context_relevance | 16 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 12 | 2 | 2 | 0.8125 | 0.8571 |
| controlled_alt_vs_candidate_no_context | naturalness | 11 | 5 | 0 | 0.6875 | 0.6875 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 14 | 0 | 2 | 0.9375 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 16 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 11 | 2 | 3 | 0.7812 | 0.8462 |
| controlled_alt_vs_candidate_no_context | persona_style | 3 | 1 | 12 | 0.5625 | 0.7500 |
| controlled_alt_vs_candidate_no_context | distinct1 | 11 | 4 | 1 | 0.7188 | 0.7333 |
| controlled_alt_vs_candidate_no_context | length_score | 12 | 3 | 1 | 0.7812 | 0.8000 |
| controlled_alt_vs_candidate_no_context | sentence_score | 8 | 2 | 6 | 0.6875 | 0.8000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 14 | 2 | 0 | 0.8750 | 0.8750 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.3125 | 0.4375 | 0.5625 |
| proposed_contextual_controlled_tuned | 0.0000 | 0.0000 | 0.3750 | 0.3125 | 0.6875 |
| proposed_contextual | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 |
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
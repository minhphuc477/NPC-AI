# Proposal Alignment Evaluation Report

- Run ID: `20260305T235603Z`
- Generated: `2026-03-05T23:58:40.052603+00:00`
- Scenarios: `artifacts\proposal_control_tuning\auto_tune_v4\20260305T235104Z\train_runs\trial_002\seed_29\20260305T235603Z\scenarios.jsonl`
- Scenario count: `16`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_tuned`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2533 (0.2025, 0.3097) | 0.3707 (0.2909, 0.4570) | 0.9033 (0.8745, 0.9314) | 0.4266 (0.3925, 0.4629) | n/a |
| proposed_contextual_controlled_tuned | 0.2665 (0.2189, 0.3190) | 0.3654 (0.2724, 0.4795) | 0.9165 (0.8882, 0.9410) | 0.4338 (0.3944, 0.4798) | n/a |
| proposed_contextual | 0.0905 (0.0368, 0.1485) | 0.1911 (0.1144, 0.2813) | 0.8239 (0.7835, 0.8658) | 0.2688 (0.2158, 0.3294) | n/a |
| candidate_no_context | 0.0248 (0.0115, 0.0441) | 0.2110 (0.1318, 0.2987) | 0.8274 (0.7862, 0.8699) | 0.2469 (0.2106, 0.2888) | n/a |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0657 | 2.6470 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0199 | -0.0945 |
| proposed_vs_candidate_no_context | naturalness | -0.0035 | -0.0042 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0852 | 5.0000 |
| proposed_vs_candidate_no_context | context_overlap | 0.0202 | 0.4693 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0327 | -0.2500 |
| proposed_vs_candidate_no_context | persona_style | 0.0312 | 0.0588 |
| proposed_vs_candidate_no_context | distinct1 | 0.0048 | 0.0051 |
| proposed_vs_candidate_no_context | length_score | -0.0271 | -0.0734 |
| proposed_vs_candidate_no_context | sentence_score | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | overall_quality | 0.0218 | 0.0884 |
| controlled_vs_proposed_raw | context_relevance | 0.1628 | 1.7981 |
| controlled_vs_proposed_raw | persona_consistency | 0.1797 | 0.9403 |
| controlled_vs_proposed_raw | naturalness | 0.0794 | 0.0963 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2148 | 2.1000 |
| controlled_vs_proposed_raw | context_overlap | 0.0415 | 0.6570 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2051 | 2.0879 |
| controlled_vs_proposed_raw | persona_style | 0.0781 | 0.1389 |
| controlled_vs_proposed_raw | distinct1 | -0.0018 | -0.0019 |
| controlled_vs_proposed_raw | length_score | 0.3146 | 0.9207 |
| controlled_vs_proposed_raw | sentence_score | 0.1750 | 0.2240 |
| controlled_vs_proposed_raw | overall_quality | 0.1579 | 0.5874 |
| controlled_vs_candidate_no_context | context_relevance | 0.2285 | 9.2045 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1597 | 0.7570 |
| controlled_vs_candidate_no_context | naturalness | 0.0759 | 0.0917 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3000 | 17.6000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0617 | 1.4346 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1723 | 1.3159 |
| controlled_vs_candidate_no_context | persona_style | 0.1094 | 0.2059 |
| controlled_vs_candidate_no_context | distinct1 | 0.0030 | 0.0032 |
| controlled_vs_candidate_no_context | length_score | 0.2875 | 0.7797 |
| controlled_vs_candidate_no_context | sentence_score | 0.1750 | 0.2240 |
| controlled_vs_candidate_no_context | overall_quality | 0.1797 | 0.7277 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0132 | 0.0520 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0053 | -0.0143 |
| controlled_alt_vs_controlled_default | naturalness | 0.0132 | 0.0146 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0205 | 0.0645 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0038 | -0.0363 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0051 | -0.0167 |
| controlled_alt_vs_controlled_default | persona_style | -0.0062 | -0.0098 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0059 | 0.0063 |
| controlled_alt_vs_controlled_default | length_score | 0.0417 | 0.0635 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0219 | 0.0229 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0071 | 0.0167 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1760 | 1.9436 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1744 | 0.9126 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0926 | 0.1124 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2352 | 2.3000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0377 | 0.5969 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.2000 | 2.0364 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0719 | 0.1278 |
| controlled_alt_vs_proposed_raw | distinct1 | 0.0041 | 0.0043 |
| controlled_alt_vs_proposed_raw | length_score | 0.3562 | 1.0427 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1969 | 0.2520 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1650 | 0.6139 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2417 | 9.7354 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1544 | 0.7319 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0891 | 0.1077 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3205 | 18.8000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0579 | 1.3462 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1673 | 1.2773 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.1031 | 0.1941 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0089 | 0.0095 |
| controlled_alt_vs_candidate_no_context | length_score | 0.3292 | 0.8927 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1969 | 0.2520 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1868 | 0.7565 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0657 | (0.0167, 0.1227) | 0.0030 | 0.0657 | (0.0174, 0.1140) | 0.0007 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0199 | (-0.0853, 0.0324) | 0.7293 | -0.0199 | (-0.0938, 0.0268) | 0.7343 |
| proposed_vs_candidate_no_context | naturalness | -0.0035 | (-0.0498, 0.0402) | 0.5777 | -0.0035 | (-0.0095, 0.0016) | 0.9070 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0852 | (0.0227, 0.1591) | 0.0033 | 0.0852 | (0.0227, 0.1477) | 0.0000 |
| proposed_vs_candidate_no_context | context_overlap | 0.0202 | (-0.0052, 0.0510) | 0.0693 | 0.0202 | (-0.0006, 0.0485) | 0.0420 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0327 | (-0.1057, 0.0179) | 0.8657 | -0.0327 | (-0.1190, 0.0190) | 0.8743 |
| proposed_vs_candidate_no_context | persona_style | 0.0312 | (-0.0723, 0.1543) | 0.2967 | 0.0312 | (0.0000, 0.1000) | 0.3297 |
| proposed_vs_candidate_no_context | distinct1 | 0.0048 | (-0.0148, 0.0237) | 0.2963 | 0.0048 | (-0.0069, 0.0272) | 0.3203 |
| proposed_vs_candidate_no_context | length_score | -0.0271 | (-0.1875, 0.1376) | 0.6013 | -0.0271 | (-0.0583, -0.0053) | 1.0000 |
| proposed_vs_candidate_no_context | sentence_score | 0.0000 | (-0.1094, 0.1094) | 0.5970 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | overall_quality | 0.0218 | (-0.0206, 0.0641) | 0.1683 | 0.0218 | (-0.0052, 0.0456) | 0.0707 |
| controlled_vs_proposed_raw | context_relevance | 0.1628 | (0.0924, 0.2297) | 0.0000 | 0.1628 | (0.0666, 0.2214) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.1797 | (0.0934, 0.2713) | 0.0000 | 0.1797 | (0.1054, 0.2573) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0794 | (0.0293, 0.1252) | 0.0017 | 0.0794 | (0.0399, 0.1170) | 0.0000 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2148 | (0.1341, 0.2966) | 0.0000 | 0.2148 | (0.0970, 0.2899) | 0.0003 |
| controlled_vs_proposed_raw | context_overlap | 0.0415 | (0.0026, 0.0757) | 0.0190 | 0.0415 | (-0.0024, 0.0744) | 0.0487 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2051 | (0.1113, 0.3101) | 0.0000 | 0.2051 | (0.1315, 0.3021) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.0781 | (-0.0195, 0.2013) | 0.0887 | 0.0781 | (-0.0469, 0.2500) | 0.3243 |
| controlled_vs_proposed_raw | distinct1 | -0.0018 | (-0.0263, 0.0214) | 0.5463 | -0.0018 | (-0.0371, 0.0224) | 0.5250 |
| controlled_vs_proposed_raw | length_score | 0.3146 | (0.1063, 0.5042) | 0.0020 | 0.3146 | (0.1458, 0.4750) | 0.0000 |
| controlled_vs_proposed_raw | sentence_score | 0.1750 | (0.0875, 0.2625) | 0.0000 | 0.1750 | (0.1029, 0.2722) | 0.0000 |
| controlled_vs_proposed_raw | overall_quality | 0.1579 | (0.1029, 0.2143) | 0.0000 | 0.1579 | (0.0996, 0.2031) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2285 | (0.1795, 0.2847) | 0.0000 | 0.2285 | (0.1512, 0.3093) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1597 | (0.0544, 0.2645) | 0.0013 | 0.1597 | (0.0538, 0.2524) | 0.0043 |
| controlled_vs_candidate_no_context | naturalness | 0.0759 | (0.0359, 0.1145) | 0.0000 | 0.0759 | (0.0380, 0.1151) | 0.0000 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3000 | (0.2352, 0.3705) | 0.0000 | 0.3000 | (0.2045, 0.4061) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0617 | (0.0444, 0.0813) | 0.0000 | 0.0617 | (0.0356, 0.0836) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1723 | (0.0458, 0.3039) | 0.0023 | 0.1723 | (0.0405, 0.3018) | 0.0060 |
| controlled_vs_candidate_no_context | persona_style | 0.1094 | (0.0156, 0.2324) | 0.0110 | 0.1094 | (0.0000, 0.2639) | 0.0750 |
| controlled_vs_candidate_no_context | distinct1 | 0.0030 | (-0.0148, 0.0208) | 0.3740 | 0.0030 | (-0.0162, 0.0180) | 0.3880 |
| controlled_vs_candidate_no_context | length_score | 0.2875 | (0.1229, 0.4521) | 0.0007 | 0.2875 | (0.1143, 0.4396) | 0.0003 |
| controlled_vs_candidate_no_context | sentence_score | 0.1750 | (0.0875, 0.2625) | 0.0000 | 0.1750 | (0.1029, 0.2722) | 0.0000 |
| controlled_vs_candidate_no_context | overall_quality | 0.1797 | (0.1340, 0.2267) | 0.0000 | 0.1797 | (0.1185, 0.2292) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0132 | (-0.0526, 0.0843) | 0.3540 | 0.0132 | (-0.0519, 0.0975) | 0.3730 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0053 | (-0.1279, 0.1392) | 0.5360 | -0.0053 | (-0.0845, 0.0928) | 0.5660 |
| controlled_alt_vs_controlled_default | naturalness | 0.0132 | (-0.0247, 0.0510) | 0.2423 | 0.0132 | (-0.0152, 0.0402) | 0.1823 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0205 | (-0.0636, 0.1114) | 0.3310 | 0.0205 | (-0.0584, 0.1273) | 0.3663 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0038 | (-0.0290, 0.0195) | 0.6103 | -0.0038 | (-0.0294, 0.0278) | 0.6330 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0051 | (-0.1622, 0.1860) | 0.5460 | -0.0051 | (-0.1020, 0.1160) | 0.5697 |
| controlled_alt_vs_controlled_default | persona_style | -0.0062 | (-0.0187, 0.0000) | 1.0000 | -0.0062 | (-0.0200, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0059 | (-0.0110, 0.0235) | 0.2723 | 0.0059 | (-0.0060, 0.0211) | 0.1963 |
| controlled_alt_vs_controlled_default | length_score | 0.0417 | (-0.1583, 0.2333) | 0.3423 | 0.0417 | (-0.1222, 0.1867) | 0.3123 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0219 | (-0.0437, 0.1094) | 0.3780 | 0.0219 | (-0.0538, 0.0824) | 0.3847 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0071 | (-0.0461, 0.0644) | 0.4020 | 0.0071 | (-0.0357, 0.0499) | 0.3923 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1760 | (0.1089, 0.2447) | 0.0000 | 0.1760 | (0.1480, 0.2046) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1744 | (0.0459, 0.3196) | 0.0037 | 0.1744 | (0.0862, 0.2269) | 0.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0926 | (0.0517, 0.1350) | 0.0000 | 0.0926 | (0.0673, 0.1188) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2352 | (0.1500, 0.3205) | 0.0000 | 0.2352 | (0.2039, 0.2699) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0377 | (0.0036, 0.0664) | 0.0157 | 0.0377 | (0.0153, 0.0538) | 0.0030 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.2000 | (0.0408, 0.3694) | 0.0063 | 0.2000 | (0.1107, 0.2780) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0719 | (-0.0312, 0.2051) | 0.1040 | 0.0719 | (-0.0531, 0.2500) | 0.3343 |
| controlled_alt_vs_proposed_raw | distinct1 | 0.0041 | (-0.0179, 0.0217) | 0.3310 | 0.0041 | (-0.0210, 0.0187) | 0.3077 |
| controlled_alt_vs_proposed_raw | length_score | 0.3563 | (0.1813, 0.5250) | 0.0000 | 0.3563 | (0.2786, 0.4298) | 0.0000 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1969 | (0.1094, 0.2844) | 0.0003 | 0.1969 | (0.0875, 0.2882) | 0.0007 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1650 | (0.0915, 0.2377) | 0.0000 | 0.1650 | (0.1206, 0.1991) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2417 | (0.1929, 0.2990) | 0.0000 | 0.2417 | (0.2124, 0.2675) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1544 | (0.0232, 0.3010) | 0.0130 | 0.1544 | (0.0454, 0.2322) | 0.0060 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0891 | (0.0541, 0.1239) | 0.0000 | 0.0891 | (0.0614, 0.1184) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3205 | (0.2500, 0.3955) | 0.0000 | 0.3205 | (0.2792, 0.3568) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0579 | (0.0452, 0.0724) | 0.0000 | 0.0579 | (0.0525, 0.0659) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1673 | (-0.0098, 0.3488) | 0.0317 | 0.1673 | (0.0323, 0.2722) | 0.0063 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.1031 | (0.0031, 0.2203) | 0.0197 | 0.1031 | (-0.0143, 0.2583) | 0.0917 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0089 | (-0.0133, 0.0290) | 0.2187 | 0.0089 | (0.0032, 0.0150) | 0.0000 |
| controlled_alt_vs_candidate_no_context | length_score | 0.3292 | (0.1771, 0.4813) | 0.0000 | 0.3292 | (0.2256, 0.4263) | 0.0000 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1969 | (0.0875, 0.2844) | 0.0010 | 0.1969 | (0.0997, 0.2882) | 0.0007 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1868 | (0.1288, 0.2529) | 0.0000 | 0.1868 | (0.1398, 0.2132) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 7 | 2 | 7 | 0.6562 | 0.7778 |
| proposed_vs_candidate_no_context | persona_consistency | 4 | 3 | 9 | 0.5312 | 0.5714 |
| proposed_vs_candidate_no_context | naturalness | 3 | 6 | 7 | 0.4062 | 0.3333 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 6 | 1 | 9 | 0.6562 | 0.8571 |
| proposed_vs_candidate_no_context | context_overlap | 5 | 4 | 7 | 0.5312 | 0.5556 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 1 | 2 | 13 | 0.4688 | 0.3333 |
| proposed_vs_candidate_no_context | persona_style | 3 | 2 | 11 | 0.5312 | 0.6000 |
| proposed_vs_candidate_no_context | distinct1 | 6 | 3 | 7 | 0.5938 | 0.6667 |
| proposed_vs_candidate_no_context | length_score | 2 | 6 | 8 | 0.3750 | 0.2500 |
| proposed_vs_candidate_no_context | sentence_score | 3 | 3 | 10 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | overall_quality | 6 | 3 | 7 | 0.5938 | 0.6667 |
| controlled_vs_proposed_raw | context_relevance | 14 | 2 | 0 | 0.8750 | 0.8750 |
| controlled_vs_proposed_raw | persona_consistency | 12 | 2 | 2 | 0.8125 | 0.8571 |
| controlled_vs_proposed_raw | naturalness | 12 | 4 | 0 | 0.7500 | 0.7500 |
| controlled_vs_proposed_raw | context_keyword_coverage | 14 | 1 | 1 | 0.9062 | 0.9333 |
| controlled_vs_proposed_raw | context_overlap | 13 | 3 | 0 | 0.8125 | 0.8125 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 12 | 1 | 3 | 0.8438 | 0.9231 |
| controlled_vs_proposed_raw | persona_style | 3 | 1 | 12 | 0.5625 | 0.7500 |
| controlled_vs_proposed_raw | distinct1 | 9 | 7 | 0 | 0.5625 | 0.5625 |
| controlled_vs_proposed_raw | length_score | 11 | 5 | 0 | 0.6875 | 0.6875 |
| controlled_vs_proposed_raw | sentence_score | 8 | 0 | 8 | 0.7500 | 1.0000 |
| controlled_vs_proposed_raw | overall_quality | 14 | 2 | 0 | 0.8750 | 0.8750 |
| controlled_vs_candidate_no_context | context_relevance | 16 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 10 | 2 | 4 | 0.7500 | 0.8333 |
| controlled_vs_candidate_no_context | naturalness | 13 | 3 | 0 | 0.8125 | 0.8125 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 16 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 15 | 1 | 0 | 0.9375 | 0.9375 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 10 | 2 | 4 | 0.7500 | 0.8333 |
| controlled_vs_candidate_no_context | persona_style | 4 | 0 | 12 | 0.6250 | 1.0000 |
| controlled_vs_candidate_no_context | distinct1 | 8 | 8 | 0 | 0.5000 | 0.5000 |
| controlled_vs_candidate_no_context | length_score | 12 | 4 | 0 | 0.7500 | 0.7500 |
| controlled_vs_candidate_no_context | sentence_score | 8 | 0 | 8 | 0.7500 | 1.0000 |
| controlled_vs_candidate_no_context | overall_quality | 16 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 6 | 7 | 3 | 0.4688 | 0.4615 |
| controlled_alt_vs_controlled_default | persona_consistency | 2 | 5 | 9 | 0.4062 | 0.2857 |
| controlled_alt_vs_controlled_default | naturalness | 9 | 4 | 3 | 0.6562 | 0.6923 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 5 | 3 | 8 | 0.5625 | 0.6250 |
| controlled_alt_vs_controlled_default | context_overlap | 6 | 7 | 3 | 0.4688 | 0.4615 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 2 | 5 | 9 | 0.4062 | 0.2857 |
| controlled_alt_vs_controlled_default | persona_style | 0 | 1 | 15 | 0.4688 | 0.0000 |
| controlled_alt_vs_controlled_default | distinct1 | 6 | 7 | 3 | 0.4688 | 0.4615 |
| controlled_alt_vs_controlled_default | length_score | 8 | 5 | 3 | 0.5938 | 0.6154 |
| controlled_alt_vs_controlled_default | sentence_score | 2 | 1 | 13 | 0.5312 | 0.6667 |
| controlled_alt_vs_controlled_default | overall_quality | 6 | 7 | 3 | 0.4688 | 0.4615 |
| controlled_alt_vs_proposed_raw | context_relevance | 14 | 2 | 0 | 0.8750 | 0.8750 |
| controlled_alt_vs_proposed_raw | persona_consistency | 11 | 2 | 3 | 0.7812 | 0.8462 |
| controlled_alt_vs_proposed_raw | naturalness | 13 | 3 | 0 | 0.8125 | 0.8125 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 14 | 2 | 0 | 0.8750 | 0.8750 |
| controlled_alt_vs_proposed_raw | context_overlap | 13 | 3 | 0 | 0.8125 | 0.8125 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 11 | 2 | 3 | 0.7812 | 0.8462 |
| controlled_alt_vs_proposed_raw | persona_style | 3 | 2 | 11 | 0.5312 | 0.6000 |
| controlled_alt_vs_proposed_raw | distinct1 | 9 | 5 | 2 | 0.6250 | 0.6429 |
| controlled_alt_vs_proposed_raw | length_score | 14 | 2 | 0 | 0.8750 | 0.8750 |
| controlled_alt_vs_proposed_raw | sentence_score | 9 | 0 | 7 | 0.7812 | 1.0000 |
| controlled_alt_vs_proposed_raw | overall_quality | 15 | 1 | 0 | 0.9375 | 0.9375 |
| controlled_alt_vs_candidate_no_context | context_relevance | 16 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 10 | 3 | 3 | 0.7188 | 0.7692 |
| controlled_alt_vs_candidate_no_context | naturalness | 15 | 1 | 0 | 0.9375 | 0.9375 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 16 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 16 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 10 | 3 | 3 | 0.7188 | 0.7692 |
| controlled_alt_vs_candidate_no_context | persona_style | 4 | 1 | 11 | 0.5938 | 0.8000 |
| controlled_alt_vs_candidate_no_context | distinct1 | 10 | 6 | 0 | 0.6250 | 0.6250 |
| controlled_alt_vs_candidate_no_context | length_score | 14 | 2 | 0 | 0.8750 | 0.8750 |
| controlled_alt_vs_candidate_no_context | sentence_score | 10 | 1 | 5 | 0.7812 | 0.9091 |
| controlled_alt_vs_candidate_no_context | overall_quality | 15 | 1 | 0 | 0.9375 | 0.9375 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.4375 | 0.3125 | 0.6875 |
| proposed_contextual_controlled_tuned | 0.0000 | 0.0000 | 0.3750 | 0.5000 | 0.5000 |
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
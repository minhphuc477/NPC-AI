# Proposal Alignment Evaluation Report

- Run ID: `20260305T221441Z`
- Generated: `2026-03-05T22:16:50.176179+00:00`
- Scenarios: `artifacts\proposal_control_tuning\auto_tune_v3_smoke\20260305T221440Z\train_runs\trial_000\seed_19\20260305T221441Z\scenarios.jsonl`
- Scenario count: `12`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_tuned`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2591 (0.2110, 0.3089) | 0.3321 (0.2785, 0.3911) | 0.8773 (0.8421, 0.9126) | 0.4095 (0.3823, 0.4367) | n/a |
| proposed_contextual_controlled_tuned | 0.2747 (0.2411, 0.3236) | 0.2971 (0.2313, 0.3767) | 0.9041 (0.8723, 0.9302) | 0.4086 (0.3794, 0.4396) | n/a |
| proposed_contextual | 0.0732 (0.0278, 0.1275) | 0.1667 (0.1250, 0.2139) | 0.8249 (0.7832, 0.8672) | 0.2505 (0.2175, 0.2856) | n/a |
| candidate_no_context | 0.0325 (0.0114, 0.0593) | 0.1746 (0.1167, 0.2429) | 0.8083 (0.7657, 0.8599) | 0.2312 (0.1973, 0.2785) | n/a |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0407 | 1.2534 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0079 | -0.0455 |
| proposed_vs_candidate_no_context | naturalness | 0.0167 | 0.0206 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0530 | 1.7500 |
| proposed_vs_candidate_no_context | context_overlap | 0.0119 | 0.3182 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0099 | -0.1923 |
| proposed_vs_candidate_no_context | persona_style | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | distinct1 | 0.0156 | 0.0166 |
| proposed_vs_candidate_no_context | length_score | 0.0667 | 0.2182 |
| proposed_vs_candidate_no_context | sentence_score | -0.0292 | -0.0412 |
| proposed_vs_candidate_no_context | overall_quality | 0.0193 | 0.0833 |
| controlled_vs_proposed_raw | context_relevance | 0.1859 | 2.5410 |
| controlled_vs_proposed_raw | persona_consistency | 0.1654 | 0.9926 |
| controlled_vs_proposed_raw | naturalness | 0.0524 | 0.0635 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2424 | 2.9091 |
| controlled_vs_proposed_raw | context_overlap | 0.0542 | 1.0945 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2099 | 5.0381 |
| controlled_vs_proposed_raw | persona_style | -0.0125 | -0.0187 |
| controlled_vs_proposed_raw | distinct1 | -0.0086 | -0.0090 |
| controlled_vs_proposed_raw | length_score | 0.1333 | 0.3582 |
| controlled_vs_proposed_raw | sentence_score | 0.2917 | 0.4294 |
| controlled_vs_proposed_raw | overall_quality | 0.1590 | 0.6349 |
| controlled_vs_candidate_no_context | context_relevance | 0.2267 | 6.9793 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1575 | 0.9020 |
| controlled_vs_candidate_no_context | naturalness | 0.0690 | 0.0854 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.2955 | 9.7500 |
| controlled_vs_candidate_no_context | context_overlap | 0.0661 | 1.7610 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2000 | 3.8769 |
| controlled_vs_candidate_no_context | persona_style | -0.0125 | -0.0187 |
| controlled_vs_candidate_no_context | distinct1 | 0.0070 | 0.0074 |
| controlled_vs_candidate_no_context | length_score | 0.2000 | 0.6545 |
| controlled_vs_candidate_no_context | sentence_score | 0.2625 | 0.3706 |
| controlled_vs_candidate_no_context | overall_quality | 0.1783 | 0.7710 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0156 | 0.0600 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0350 | -0.1054 |
| controlled_alt_vs_controlled_default | naturalness | 0.0268 | 0.0305 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0152 | 0.0465 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0165 | 0.1592 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0425 | -0.1688 |
| controlled_alt_vs_controlled_default | persona_style | -0.0052 | -0.0080 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0236 | -0.0250 |
| controlled_alt_vs_controlled_default | length_score | 0.1833 | 0.3626 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0000 | 0.0000 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0009 | -0.0022 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.2015 | 2.7536 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1304 | 0.7826 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0791 | 0.0959 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2576 | 3.0909 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0707 | 1.4281 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1675 | 4.0190 |
| controlled_alt_vs_proposed_raw | persona_style | -0.0177 | -0.0266 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0323 | -0.0337 |
| controlled_alt_vs_proposed_raw | length_score | 0.3167 | 0.8507 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.2917 | 0.4294 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1582 | 0.6314 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2422 | 7.4584 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1225 | 0.7015 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0958 | 0.1185 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3106 | 10.2500 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0826 | 2.2007 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1575 | 3.0538 |
| controlled_alt_vs_candidate_no_context | persona_style | -0.0177 | -0.0266 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0167 | -0.0177 |
| controlled_alt_vs_candidate_no_context | length_score | 0.3833 | 1.2545 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.2625 | 0.3706 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1774 | 0.7672 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0407 | (0.0001, 0.0942) | 0.0247 | 0.0407 | (0.0138, 0.0766) | 0.0037 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0079 | (-0.0683, 0.0556) | 0.6150 | -0.0079 | (-0.0457, 0.0333) | 0.6420 |
| proposed_vs_candidate_no_context | naturalness | 0.0167 | (-0.0337, 0.0635) | 0.2507 | 0.0167 | (-0.0053, 0.0406) | 0.0827 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0530 | (-0.0000, 0.1212) | 0.0497 | 0.0530 | (0.0152, 0.0992) | 0.0027 |
| proposed_vs_candidate_no_context | context_overlap | 0.0119 | (-0.0004, 0.0295) | 0.0417 | 0.0119 | (-0.0022, 0.0240) | 0.0570 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0099 | (-0.0813, 0.0694) | 0.6333 | -0.0099 | (-0.0571, 0.0417) | 0.6447 |
| proposed_vs_candidate_no_context | persona_style | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | distinct1 | 0.0156 | (-0.0013, 0.0379) | 0.0417 | 0.0156 | (0.0011, 0.0260) | 0.0043 |
| proposed_vs_candidate_no_context | length_score | 0.0667 | (-0.1389, 0.2556) | 0.2453 | 0.0667 | (-0.0278, 0.2000) | 0.1143 |
| proposed_vs_candidate_no_context | sentence_score | -0.0292 | (-0.1167, 0.0583) | 0.8263 | -0.0292 | (-0.0700, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | overall_quality | 0.0193 | (-0.0216, 0.0639) | 0.1950 | 0.0193 | (-0.0091, 0.0476) | 0.1037 |
| controlled_vs_proposed_raw | context_relevance | 0.1859 | (0.1108, 0.2469) | 0.0000 | 0.1859 | (0.1640, 0.2140) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.1654 | (0.0965, 0.2204) | 0.0000 | 0.1654 | (0.0778, 0.2353) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0524 | (-0.0101, 0.1114) | 0.0537 | 0.0524 | (-0.0111, 0.0957) | 0.0357 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2424 | (0.1439, 0.3258) | 0.0000 | 0.2424 | (0.2168, 0.2803) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0542 | (0.0357, 0.0704) | 0.0000 | 0.0542 | (0.0414, 0.0661) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2099 | (0.1222, 0.2794) | 0.0000 | 0.2099 | (0.0972, 0.3016) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | -0.0125 | (-0.0375, 0.0000) | 1.0000 | -0.0125 | (-0.0300, 0.0000) | 1.0000 |
| controlled_vs_proposed_raw | distinct1 | -0.0086 | (-0.0281, 0.0122) | 0.7940 | -0.0086 | (-0.0250, 0.0099) | 0.8363 |
| controlled_vs_proposed_raw | length_score | 0.1333 | (-0.1639, 0.4056) | 0.1910 | 0.1333 | (-0.0861, 0.3071) | 0.1240 |
| controlled_vs_proposed_raw | sentence_score | 0.2917 | (0.2042, 0.3500) | 0.0000 | 0.2917 | (0.1750, 0.3500) | 0.0000 |
| controlled_vs_proposed_raw | overall_quality | 0.1590 | (0.1234, 0.1916) | 0.0000 | 0.1590 | (0.1261, 0.1877) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2267 | (0.1835, 0.2729) | 0.0000 | 0.2267 | (0.1907, 0.2626) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1575 | (0.1054, 0.2106) | 0.0000 | 0.1575 | (0.1109, 0.1879) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.0690 | (0.0190, 0.1186) | 0.0043 | 0.0690 | (0.0322, 0.0954) | 0.0000 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.2955 | (0.2348, 0.3561) | 0.0000 | 0.2955 | (0.2424, 0.3485) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0661 | (0.0557, 0.0766) | 0.0000 | 0.0661 | (0.0617, 0.0700) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2000 | (0.1305, 0.2667) | 0.0000 | 0.2000 | (0.1389, 0.2429) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | -0.0125 | (-0.0375, 0.0000) | 1.0000 | -0.0125 | (-0.0300, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | distinct1 | 0.0070 | (-0.0147, 0.0279) | 0.2837 | 0.0070 | (-0.0196, 0.0206) | 0.2683 |
| controlled_vs_candidate_no_context | length_score | 0.2000 | (-0.0000, 0.4028) | 0.0260 | 0.2000 | (0.0800, 0.2857) | 0.0000 |
| controlled_vs_candidate_no_context | sentence_score | 0.2625 | (0.1750, 0.3500) | 0.0000 | 0.2625 | (0.1615, 0.3500) | 0.0000 |
| controlled_vs_candidate_no_context | overall_quality | 0.1783 | (0.1441, 0.2093) | 0.0000 | 0.1783 | (0.1600, 0.1949) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0156 | (-0.0563, 0.0875) | 0.3417 | 0.0156 | (-0.0382, 0.0667) | 0.3020 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0350 | (-0.1019, 0.0401) | 0.8380 | -0.0350 | (-0.0587, -0.0111) | 1.0000 |
| controlled_alt_vs_controlled_default | naturalness | 0.0268 | (-0.0255, 0.0801) | 0.1680 | 0.0268 | (-0.0138, 0.0835) | 0.0860 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0152 | (-0.0909, 0.1212) | 0.4007 | 0.0152 | (-0.0606, 0.0909) | 0.4047 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0165 | (-0.0049, 0.0365) | 0.0593 | 0.0165 | (-0.0087, 0.0345) | 0.0913 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0425 | (-0.1226, 0.0508) | 0.8227 | -0.0425 | (-0.0743, -0.0139) | 1.0000 |
| controlled_alt_vs_controlled_default | persona_style | -0.0052 | (-0.0490, 0.0333) | 0.6627 | -0.0052 | (-0.0125, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0236 | (-0.0379, -0.0068) | 0.9983 | -0.0236 | (-0.0361, -0.0140) | 1.0000 |
| controlled_alt_vs_controlled_default | length_score | 0.1833 | (-0.0667, 0.4389) | 0.0780 | 0.1833 | (-0.0048, 0.4467) | 0.0380 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0009 | (-0.0280, 0.0239) | 0.5313 | -0.0009 | (-0.0197, 0.0180) | 0.5827 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.2015 | (0.1269, 0.2741) | 0.0000 | 0.2015 | (0.1441, 0.2491) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1304 | (0.0508, 0.2140) | 0.0000 | 0.1304 | (0.0320, 0.2007) | 0.0010 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0791 | (0.0472, 0.1137) | 0.0000 | 0.0791 | (0.0445, 0.1358) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2576 | (0.1667, 0.3561) | 0.0000 | 0.2576 | (0.1742, 0.3212) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0707 | (0.0428, 0.0970) | 0.0000 | 0.0707 | (0.0516, 0.0809) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1675 | (0.0683, 0.2726) | 0.0000 | 0.1675 | (0.0400, 0.2592) | 0.0037 |
| controlled_alt_vs_proposed_raw | persona_style | -0.0177 | (-0.0490, 0.0000) | 1.0000 | -0.0177 | (-0.0426, 0.0000) | 1.0000 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0323 | (-0.0558, -0.0061) | 0.9943 | -0.0323 | (-0.0523, -0.0104) | 0.9957 |
| controlled_alt_vs_proposed_raw | length_score | 0.3167 | (0.1806, 0.4667) | 0.0000 | 0.3167 | (0.2179, 0.5133) | 0.0000 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.2917 | (0.2042, 0.3500) | 0.0000 | 0.2917 | (0.1750, 0.3500) | 0.0000 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1582 | (0.1131, 0.1987) | 0.0000 | 0.1582 | (0.1055, 0.2044) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2422 | (0.1939, 0.2968) | 0.0000 | 0.2422 | (0.2125, 0.2730) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1225 | (0.0619, 0.1728) | 0.0000 | 0.1225 | (0.0587, 0.1681) | 0.0000 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0958 | (0.0512, 0.1391) | 0.0000 | 0.0958 | (0.0751, 0.1409) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3106 | (0.2424, 0.3864) | 0.0000 | 0.3106 | (0.2727, 0.3506) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0826 | (0.0625, 0.1036) | 0.0000 | 0.0826 | (0.0551, 0.1022) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1575 | (0.0801, 0.2258) | 0.0007 | 0.1575 | (0.0733, 0.2177) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_style | -0.0177 | (-0.0490, 0.0000) | 1.0000 | -0.0177 | (-0.0425, 0.0000) | 1.0000 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0167 | (-0.0411, 0.0081) | 0.9007 | -0.0167 | (-0.0400, 0.0030) | 0.9410 |
| controlled_alt_vs_candidate_no_context | length_score | 0.3833 | (0.1917, 0.5723) | 0.0000 | 0.3833 | (0.2800, 0.5556) | 0.0000 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.2625 | (0.1750, 0.3500) | 0.0000 | 0.2625 | (0.1615, 0.3500) | 0.0000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1774 | (0.1433, 0.2096) | 0.0000 | 0.1774 | (0.1477, 0.2004) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 6 | 1 | 5 | 0.7083 | 0.8571 |
| proposed_vs_candidate_no_context | persona_consistency | 1 | 2 | 9 | 0.4583 | 0.3333 |
| proposed_vs_candidate_no_context | naturalness | 5 | 2 | 5 | 0.6250 | 0.7143 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 4 | 1 | 7 | 0.6250 | 0.8000 |
| proposed_vs_candidate_no_context | context_overlap | 4 | 3 | 5 | 0.5417 | 0.5714 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 1 | 2 | 9 | 0.4583 | 0.3333 |
| proposed_vs_candidate_no_context | persona_style | 0 | 0 | 12 | 0.5000 | nan |
| proposed_vs_candidate_no_context | distinct1 | 4 | 1 | 7 | 0.6250 | 0.8000 |
| proposed_vs_candidate_no_context | length_score | 5 | 2 | 5 | 0.6250 | 0.7143 |
| proposed_vs_candidate_no_context | sentence_score | 1 | 2 | 9 | 0.4583 | 0.3333 |
| proposed_vs_candidate_no_context | overall_quality | 4 | 3 | 5 | 0.5417 | 0.5714 |
| controlled_vs_proposed_raw | context_relevance | 11 | 1 | 0 | 0.9167 | 0.9167 |
| controlled_vs_proposed_raw | persona_consistency | 11 | 1 | 0 | 0.9167 | 0.9167 |
| controlled_vs_proposed_raw | naturalness | 8 | 4 | 0 | 0.6667 | 0.6667 |
| controlled_vs_proposed_raw | context_keyword_coverage | 11 | 1 | 0 | 0.9167 | 0.9167 |
| controlled_vs_proposed_raw | context_overlap | 11 | 1 | 0 | 0.9167 | 0.9167 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 11 | 1 | 0 | 0.9167 | 0.9167 |
| controlled_vs_proposed_raw | persona_style | 0 | 1 | 11 | 0.4583 | 0.0000 |
| controlled_vs_proposed_raw | distinct1 | 5 | 6 | 1 | 0.4583 | 0.4545 |
| controlled_vs_proposed_raw | length_score | 6 | 6 | 0 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | sentence_score | 10 | 0 | 2 | 0.9167 | 1.0000 |
| controlled_vs_proposed_raw | overall_quality | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_relevance | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 10 | 0 | 2 | 0.9167 | 1.0000 |
| controlled_vs_candidate_no_context | naturalness | 10 | 2 | 0 | 0.8333 | 0.8333 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 10 | 0 | 2 | 0.9167 | 1.0000 |
| controlled_vs_candidate_no_context | persona_style | 0 | 1 | 11 | 0.4583 | 0.0000 |
| controlled_vs_candidate_no_context | distinct1 | 6 | 5 | 1 | 0.5417 | 0.5455 |
| controlled_vs_candidate_no_context | length_score | 8 | 4 | 0 | 0.6667 | 0.6667 |
| controlled_vs_candidate_no_context | sentence_score | 9 | 0 | 3 | 0.8750 | 1.0000 |
| controlled_vs_candidate_no_context | overall_quality | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 7 | 5 | 0 | 0.5833 | 0.5833 |
| controlled_alt_vs_controlled_default | persona_consistency | 2 | 6 | 4 | 0.3333 | 0.2500 |
| controlled_alt_vs_controlled_default | naturalness | 8 | 4 | 0 | 0.6667 | 0.6667 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 5 | 4 | 3 | 0.5417 | 0.5556 |
| controlled_alt_vs_controlled_default | context_overlap | 8 | 4 | 0 | 0.6667 | 0.6667 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 2 | 5 | 5 | 0.3750 | 0.2857 |
| controlled_alt_vs_controlled_default | persona_style | 1 | 2 | 9 | 0.4583 | 0.3333 |
| controlled_alt_vs_controlled_default | distinct1 | 2 | 10 | 0 | 0.1667 | 0.1667 |
| controlled_alt_vs_controlled_default | length_score | 8 | 4 | 0 | 0.6667 | 0.6667 |
| controlled_alt_vs_controlled_default | sentence_score | 0 | 0 | 12 | 0.5000 | nan |
| controlled_alt_vs_controlled_default | overall_quality | 8 | 4 | 0 | 0.6667 | 0.6667 |
| controlled_alt_vs_proposed_raw | context_relevance | 11 | 1 | 0 | 0.9167 | 0.9167 |
| controlled_alt_vs_proposed_raw | persona_consistency | 9 | 1 | 2 | 0.8333 | 0.9000 |
| controlled_alt_vs_proposed_raw | naturalness | 11 | 1 | 0 | 0.9167 | 0.9167 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 10 | 0 | 2 | 0.9167 | 1.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 11 | 1 | 0 | 0.9167 | 0.9167 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 9 | 1 | 2 | 0.8333 | 0.9000 |
| controlled_alt_vs_proposed_raw | persona_style | 0 | 2 | 10 | 0.4167 | 0.0000 |
| controlled_alt_vs_proposed_raw | distinct1 | 2 | 10 | 0 | 0.1667 | 0.1667 |
| controlled_alt_vs_proposed_raw | length_score | 11 | 1 | 0 | 0.9167 | 0.9167 |
| controlled_alt_vs_proposed_raw | sentence_score | 10 | 0 | 2 | 0.9167 | 1.0000 |
| controlled_alt_vs_proposed_raw | overall_quality | 11 | 1 | 0 | 0.9167 | 0.9167 |
| controlled_alt_vs_candidate_no_context | context_relevance | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 10 | 1 | 1 | 0.8750 | 0.9091 |
| controlled_alt_vs_candidate_no_context | naturalness | 10 | 2 | 0 | 0.8333 | 0.8333 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 10 | 1 | 1 | 0.8750 | 0.9091 |
| controlled_alt_vs_candidate_no_context | persona_style | 0 | 2 | 10 | 0.4167 | 0.0000 |
| controlled_alt_vs_candidate_no_context | distinct1 | 4 | 8 | 0 | 0.3333 | 0.3333 |
| controlled_alt_vs_candidate_no_context | length_score | 9 | 3 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_candidate_no_context | sentence_score | 9 | 0 | 3 | 0.8750 | 1.0000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 12 | 0 | 0 | 1.0000 | 1.0000 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.1667 | 0.5000 | 0.5000 |
| proposed_contextual_controlled_tuned | 0.0000 | 0.0000 | 0.4167 | 0.3333 | 0.6667 |
| proposed_contextual | 0.0000 | 0.0000 | 0.4167 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.5833 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `4`
- Unique template signatures: `11`
- Template signature ratio: `0.9167`
- Effective sample size by source clustering: `3.79`
- Effective sample size by template-signature clustering: `10.29`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: unavailable (disabled_by_flag).

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
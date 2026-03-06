# Proposal Alignment Evaluation Report

- Run ID: `20260305T221650Z`
- Generated: `2026-03-05T22:18:46.555282+00:00`
- Scenarios: `artifacts\proposal_control_tuning\auto_tune_v3_smoke\20260305T221440Z\train_runs\trial_001\seed_19\20260305T221650Z\scenarios.jsonl`
- Scenario count: `12`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_tuned`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2478 (0.2146, 0.2854) | 0.3379 (0.2467, 0.4396) | 0.8967 (0.8437, 0.9333) | 0.4101 (0.3779, 0.4452) | n/a |
| proposed_contextual_controlled_tuned | 0.2620 (0.2154, 0.3068) | 0.3538 (0.2837, 0.4275) | 0.8877 (0.8508, 0.9199) | 0.4208 (0.3967, 0.4455) | n/a |
| proposed_contextual | 0.0609 (0.0108, 0.1169) | 0.1540 (0.1167, 0.1980) | 0.7952 (0.7630, 0.8312) | 0.2342 (0.2047, 0.2683) | n/a |
| candidate_no_context | 0.0097 (0.0078, 0.0119) | 0.1667 (0.1250, 0.2139) | 0.8369 (0.7818, 0.8957) | 0.2234 (0.2024, 0.2497) | n/a |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0512 | 5.2728 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0127 | -0.0762 |
| proposed_vs_candidate_no_context | naturalness | -0.0417 | -0.0498 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0682 | nan |
| proposed_vs_candidate_no_context | context_overlap | 0.0117 | 0.3598 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0159 | -0.3810 |
| proposed_vs_candidate_no_context | persona_style | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | distinct1 | -0.0119 | -0.0125 |
| proposed_vs_candidate_no_context | length_score | -0.1556 | -0.3784 |
| proposed_vs_candidate_no_context | sentence_score | -0.0583 | -0.0791 |
| proposed_vs_candidate_no_context | overall_quality | 0.0108 | 0.0483 |
| controlled_vs_proposed_raw | context_relevance | 0.1868 | 3.0658 |
| controlled_vs_proposed_raw | persona_consistency | 0.1839 | 1.1946 |
| controlled_vs_proposed_raw | naturalness | 0.1015 | 0.1276 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2424 | 3.5556 |
| controlled_vs_proposed_raw | context_overlap | 0.0571 | 1.2962 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2310 | 8.9538 |
| controlled_vs_proposed_raw | persona_style | -0.0042 | -0.0062 |
| controlled_vs_proposed_raw | distinct1 | -0.0204 | -0.0217 |
| controlled_vs_proposed_raw | length_score | 0.4556 | 1.7826 |
| controlled_vs_proposed_raw | sentence_score | 0.2333 | 0.3436 |
| controlled_vs_proposed_raw | overall_quality | 0.1760 | 0.7515 |
| controlled_vs_candidate_no_context | context_relevance | 0.2380 | 24.5035 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1712 | 1.0274 |
| controlled_vs_candidate_no_context | naturalness | 0.0597 | 0.0714 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3106 | nan |
| controlled_vs_candidate_no_context | context_overlap | 0.0687 | 2.1224 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2151 | 5.1619 |
| controlled_vs_candidate_no_context | persona_style | -0.0042 | -0.0062 |
| controlled_vs_candidate_no_context | distinct1 | -0.0323 | -0.0339 |
| controlled_vs_candidate_no_context | length_score | 0.3000 | 0.7297 |
| controlled_vs_candidate_no_context | sentence_score | 0.1750 | 0.2373 |
| controlled_vs_candidate_no_context | overall_quality | 0.1868 | 0.8361 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0143 | 0.0576 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0159 | 0.0470 |
| controlled_alt_vs_controlled_default | naturalness | -0.0090 | -0.0100 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0152 | 0.0488 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0122 | 0.1207 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0214 | 0.0835 |
| controlled_alt_vs_controlled_default | persona_style | -0.0062 | -0.0094 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0159 | 0.0172 |
| controlled_alt_vs_controlled_default | length_score | -0.0861 | -0.1211 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0292 | -0.0320 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0106 | 0.0259 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.2011 | 3.2999 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1998 | 1.2978 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0925 | 0.1163 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2576 | 3.7778 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0693 | 1.5732 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.2524 | 9.7846 |
| controlled_alt_vs_proposed_raw | persona_style | -0.0104 | -0.0156 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0045 | -0.0048 |
| controlled_alt_vs_proposed_raw | length_score | 0.3694 | 1.4457 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.2042 | 0.3006 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1866 | 0.7968 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2523 | 25.9720 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1871 | 1.1227 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0508 | 0.0607 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3258 | nan |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0809 | 2.4992 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.2365 | 5.6762 |
| controlled_alt_vs_candidate_no_context | persona_style | -0.0104 | -0.0156 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0165 | -0.0173 |
| controlled_alt_vs_candidate_no_context | length_score | 0.2139 | 0.5203 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1458 | 0.1977 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1974 | 0.8836 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0512 | (0.0009, 0.1079) | 0.0120 | 0.0512 | (-0.0007, 0.1213) | 0.0563 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0127 | (-0.0683, 0.0413) | 0.7067 | -0.0127 | (-0.0444, 0.0190) | 0.8130 |
| proposed_vs_candidate_no_context | naturalness | -0.0417 | (-0.0940, 0.0044) | 0.9563 | -0.0417 | (-0.1049, 0.0067) | 0.9327 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0682 | (0.0000, 0.1439) | 0.0253 | 0.0682 | (0.0000, 0.1608) | 0.0597 |
| proposed_vs_candidate_no_context | context_overlap | 0.0117 | (0.0003, 0.0237) | 0.0213 | 0.0117 | (-0.0024, 0.0290) | 0.0643 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0159 | (-0.0853, 0.0476) | 0.7180 | -0.0159 | (-0.0556, 0.0238) | 0.8053 |
| proposed_vs_candidate_no_context | persona_style | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | distinct1 | -0.0119 | (-0.0282, 0.0000) | 1.0000 | -0.0119 | (-0.0248, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | length_score | -0.1556 | (-0.3722, 0.0361) | 0.9427 | -0.1556 | (-0.4273, 0.0333) | 0.9273 |
| proposed_vs_candidate_no_context | sentence_score | -0.0583 | (-0.1458, 0.0000) | 1.0000 | -0.0583 | (-0.1167, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | overall_quality | 0.0108 | (-0.0184, 0.0418) | 0.2423 | 0.0108 | (-0.0257, 0.0388) | 0.2380 |
| controlled_vs_proposed_raw | context_relevance | 0.1868 | (0.1146, 0.2553) | 0.0000 | 0.1868 | (0.1093, 0.2600) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.1839 | (0.1119, 0.2626) | 0.0000 | 0.1839 | (0.0978, 0.2761) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.1015 | (0.0504, 0.1458) | 0.0000 | 0.1015 | (0.0673, 0.1427) | 0.0000 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2424 | (0.1439, 0.3333) | 0.0000 | 0.2424 | (0.1329, 0.3455) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0571 | (0.0414, 0.0729) | 0.0000 | 0.0571 | (0.0471, 0.0659) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2310 | (0.1397, 0.3302) | 0.0000 | 0.2310 | (0.1222, 0.3476) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | -0.0042 | (-0.0125, 0.0000) | 1.0000 | -0.0042 | (-0.0100, 0.0000) | 1.0000 |
| controlled_vs_proposed_raw | distinct1 | -0.0204 | (-0.0631, 0.0189) | 0.8210 | -0.0204 | (-0.0533, 0.0154) | 0.8753 |
| controlled_vs_proposed_raw | length_score | 0.4556 | (0.3083, 0.5889) | 0.0000 | 0.4556 | (0.3611, 0.5704) | 0.0000 |
| controlled_vs_proposed_raw | sentence_score | 0.2333 | (0.0875, 0.3500) | 0.0010 | 0.2333 | (0.1591, 0.3111) | 0.0000 |
| controlled_vs_proposed_raw | overall_quality | 0.1760 | (0.1384, 0.2132) | 0.0000 | 0.1760 | (0.1216, 0.2148) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2380 | (0.2064, 0.2770) | 0.0000 | 0.2380 | (0.2088, 0.2788) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1712 | (0.0886, 0.2624) | 0.0000 | 0.1712 | (0.0830, 0.2901) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.0597 | (-0.0176, 0.1326) | 0.0650 | 0.0597 | (-0.0101, 0.1296) | 0.0647 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3106 | (0.2652, 0.3636) | 0.0000 | 0.3106 | (0.2727, 0.3719) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0687 | (0.0522, 0.0866) | 0.0000 | 0.0687 | (0.0516, 0.0833) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2151 | (0.1131, 0.3341) | 0.0000 | 0.2151 | (0.1037, 0.3651) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | -0.0042 | (-0.0125, 0.0000) | 1.0000 | -0.0042 | (-0.0100, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | distinct1 | -0.0323 | (-0.0743, 0.0085) | 0.9433 | -0.0323 | (-0.0675, -0.0027) | 0.9817 |
| controlled_vs_candidate_no_context | length_score | 0.3000 | (0.0222, 0.5417) | 0.0197 | 0.3000 | (0.0030, 0.5833) | 0.0173 |
| controlled_vs_candidate_no_context | sentence_score | 0.1750 | (0.0583, 0.2917) | 0.0060 | 0.1750 | (0.0583, 0.2917) | 0.0030 |
| controlled_vs_candidate_no_context | overall_quality | 0.1868 | (0.1525, 0.2226) | 0.0000 | 0.1868 | (0.1497, 0.2314) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0143 | (-0.0375, 0.0661) | 0.3113 | 0.0143 | (-0.0108, 0.0489) | 0.2453 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0159 | (-0.0679, 0.0997) | 0.3637 | 0.0159 | (-0.0259, 0.1067) | 0.3377 |
| controlled_alt_vs_controlled_default | naturalness | -0.0090 | (-0.0472, 0.0248) | 0.6863 | -0.0090 | (-0.0581, 0.0262) | 0.6743 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0152 | (-0.0455, 0.0833) | 0.2837 | 0.0152 | (-0.0210, 0.0629) | 0.3193 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0122 | (-0.0141, 0.0382) | 0.1820 | 0.0122 | (-0.0107, 0.0425) | 0.1433 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0214 | (-0.0734, 0.1282) | 0.3590 | 0.0214 | (-0.0286, 0.1333) | 0.3397 |
| controlled_alt_vs_controlled_default | persona_style | -0.0062 | (-0.0187, 0.0000) | 1.0000 | -0.0062 | (-0.0150, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0159 | (-0.0255, 0.0538) | 0.2103 | 0.0159 | (-0.0119, 0.0441) | 0.1417 |
| controlled_alt_vs_controlled_default | length_score | -0.0861 | (-0.2528, 0.0612) | 0.8360 | -0.0861 | (-0.3848, 0.1424) | 0.7943 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0292 | (-0.1750, 0.1167) | 0.7177 | -0.0292 | (-0.1167, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0106 | (-0.0324, 0.0496) | 0.2907 | 0.0106 | (-0.0063, 0.0370) | 0.1537 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.2011 | (0.1228, 0.2648) | 0.0000 | 0.2011 | (0.1619, 0.2559) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1998 | (0.1484, 0.2538) | 0.0000 | 0.1998 | (0.1333, 0.2663) | 0.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0925 | (0.0369, 0.1404) | 0.0003 | 0.0925 | (0.0132, 0.1331) | 0.0067 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2576 | (0.1591, 0.3409) | 0.0000 | 0.2576 | (0.1983, 0.3273) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0693 | (0.0402, 0.0945) | 0.0000 | 0.0693 | (0.0486, 0.0894) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.2524 | (0.1889, 0.3202) | 0.0000 | 0.2524 | (0.1667, 0.3381) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_style | -0.0104 | (-0.0312, 0.0000) | 1.0000 | -0.0104 | (-0.0250, 0.0000) | 1.0000 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0045 | (-0.0427, 0.0304) | 0.5940 | -0.0045 | (-0.0117, 0.0077) | 0.7620 |
| controlled_alt_vs_proposed_raw | length_score | 0.3694 | (0.1056, 0.5611) | 0.0040 | 0.3694 | (0.0806, 0.5564) | 0.0037 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.2042 | (0.1167, 0.2917) | 0.0000 | 0.2042 | (0.1400, 0.2500) | 0.0000 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1866 | (0.1369, 0.2288) | 0.0000 | 0.1866 | (0.1389, 0.2131) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2523 | (0.2037, 0.2975) | 0.0000 | 0.2523 | (0.2107, 0.2940) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1871 | (0.1013, 0.2612) | 0.0000 | 0.1871 | (0.0889, 0.2854) | 0.0000 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0508 | (-0.0127, 0.1162) | 0.0627 | 0.0508 | (-0.0293, 0.1281) | 0.1350 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3258 | (0.2652, 0.3864) | 0.0000 | 0.3258 | (0.2727, 0.3788) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0809 | (0.0577, 0.1050) | 0.0000 | 0.0809 | (0.0571, 0.1054) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.2365 | (0.1270, 0.3361) | 0.0000 | 0.2365 | (0.1111, 0.3619) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_style | -0.0104 | (-0.0312, 0.0000) | 1.0000 | -0.0104 | (-0.0250, 0.0000) | 1.0000 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0165 | (-0.0489, 0.0136) | 0.8460 | -0.0165 | (-0.0337, 0.0018) | 0.9403 |
| controlled_alt_vs_candidate_no_context | length_score | 0.2139 | (-0.0417, 0.4639) | 0.0483 | 0.2139 | (-0.1182, 0.5222) | 0.1420 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1458 | (0.0000, 0.2917) | 0.0490 | 0.1458 | (0.0318, 0.2333) | 0.0033 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1974 | (0.1566, 0.2341) | 0.0000 | 0.1974 | (0.1642, 0.2306) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 5 | 1 | 6 | 0.6667 | 0.8333 |
| proposed_vs_candidate_no_context | persona_consistency | 2 | 2 | 8 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | naturalness | 2 | 4 | 6 | 0.4167 | 0.3333 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 3 | 0 | 9 | 0.6250 | 1.0000 |
| proposed_vs_candidate_no_context | context_overlap | 5 | 1 | 6 | 0.6667 | 0.8333 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 2 | 2 | 8 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | persona_style | 0 | 0 | 12 | 0.5000 | nan |
| proposed_vs_candidate_no_context | distinct1 | 0 | 2 | 10 | 0.4167 | 0.0000 |
| proposed_vs_candidate_no_context | length_score | 2 | 4 | 6 | 0.4167 | 0.3333 |
| proposed_vs_candidate_no_context | sentence_score | 0 | 2 | 10 | 0.4167 | 0.0000 |
| proposed_vs_candidate_no_context | overall_quality | 4 | 2 | 6 | 0.5833 | 0.6667 |
| controlled_vs_proposed_raw | context_relevance | 11 | 1 | 0 | 0.9167 | 0.9167 |
| controlled_vs_proposed_raw | persona_consistency | 10 | 0 | 2 | 0.9167 | 1.0000 |
| controlled_vs_proposed_raw | naturalness | 11 | 1 | 0 | 0.9167 | 0.9167 |
| controlled_vs_proposed_raw | context_keyword_coverage | 10 | 1 | 1 | 0.8750 | 0.9091 |
| controlled_vs_proposed_raw | context_overlap | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 10 | 0 | 2 | 0.9167 | 1.0000 |
| controlled_vs_proposed_raw | persona_style | 0 | 1 | 11 | 0.4583 | 0.0000 |
| controlled_vs_proposed_raw | distinct1 | 5 | 6 | 1 | 0.4583 | 0.4545 |
| controlled_vs_proposed_raw | length_score | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_proposed_raw | sentence_score | 9 | 1 | 2 | 0.8333 | 0.9000 |
| controlled_vs_proposed_raw | overall_quality | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_relevance | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 9 | 0 | 3 | 0.8750 | 1.0000 |
| controlled_vs_candidate_no_context | naturalness | 7 | 5 | 0 | 0.5833 | 0.5833 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 9 | 0 | 3 | 0.8750 | 1.0000 |
| controlled_vs_candidate_no_context | persona_style | 0 | 1 | 11 | 0.4583 | 0.0000 |
| controlled_vs_candidate_no_context | distinct1 | 4 | 7 | 1 | 0.3750 | 0.3636 |
| controlled_vs_candidate_no_context | length_score | 9 | 3 | 0 | 0.7500 | 0.7500 |
| controlled_vs_candidate_no_context | sentence_score | 7 | 1 | 4 | 0.7500 | 0.8750 |
| controlled_vs_candidate_no_context | overall_quality | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 6 | 4 | 2 | 0.5833 | 0.6000 |
| controlled_alt_vs_controlled_default | persona_consistency | 3 | 3 | 6 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | naturalness | 5 | 5 | 2 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 4 | 2 | 6 | 0.5833 | 0.6667 |
| controlled_alt_vs_controlled_default | context_overlap | 6 | 4 | 2 | 0.5833 | 0.6000 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 3 | 3 | 6 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | persona_style | 0 | 1 | 11 | 0.4583 | 0.0000 |
| controlled_alt_vs_controlled_default | distinct1 | 6 | 4 | 2 | 0.5833 | 0.6000 |
| controlled_alt_vs_controlled_default | length_score | 4 | 6 | 2 | 0.4167 | 0.4000 |
| controlled_alt_vs_controlled_default | sentence_score | 3 | 4 | 5 | 0.4583 | 0.4286 |
| controlled_alt_vs_controlled_default | overall_quality | 6 | 4 | 2 | 0.5833 | 0.6000 |
| controlled_alt_vs_proposed_raw | context_relevance | 10 | 2 | 0 | 0.8333 | 0.8333 |
| controlled_alt_vs_proposed_raw | persona_consistency | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 11 | 1 | 0 | 0.9167 | 0.9167 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 10 | 2 | 0 | 0.8333 | 0.8333 |
| controlled_alt_vs_proposed_raw | context_overlap | 11 | 1 | 0 | 0.9167 | 0.9167 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 0 | 1 | 11 | 0.4583 | 0.0000 |
| controlled_alt_vs_proposed_raw | distinct1 | 6 | 6 | 0 | 0.5000 | 0.5000 |
| controlled_alt_vs_proposed_raw | length_score | 10 | 1 | 1 | 0.8750 | 0.9091 |
| controlled_alt_vs_proposed_raw | sentence_score | 7 | 0 | 5 | 0.7917 | 1.0000 |
| controlled_alt_vs_proposed_raw | overall_quality | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 10 | 1 | 1 | 0.8750 | 0.9091 |
| controlled_alt_vs_candidate_no_context | naturalness | 7 | 5 | 0 | 0.5833 | 0.5833 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 11 | 1 | 0 | 0.9167 | 0.9167 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 10 | 1 | 1 | 0.8750 | 0.9091 |
| controlled_alt_vs_candidate_no_context | persona_style | 0 | 1 | 11 | 0.4583 | 0.0000 |
| controlled_alt_vs_candidate_no_context | distinct1 | 5 | 6 | 1 | 0.4583 | 0.4545 |
| controlled_alt_vs_candidate_no_context | length_score | 6 | 5 | 1 | 0.5417 | 0.5455 |
| controlled_alt_vs_candidate_no_context | sentence_score | 7 | 2 | 3 | 0.7083 | 0.7778 |
| controlled_alt_vs_candidate_no_context | overall_quality | 12 | 0 | 0 | 1.0000 | 1.0000 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.5000 | 0.3333 | 0.6667 |
| proposed_contextual_controlled_tuned | 0.0000 | 0.0000 | 0.2500 | 0.4167 | 0.5833 |
| proposed_contextual | 0.0000 | 0.0000 | 0.5833 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `4`
- Unique template signatures: `11`
- Template signature ratio: `0.9167`
- Effective sample size by source clustering: `3.79`
- Effective sample size by template-signature clustering: `10.29`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: unavailable (disabled_by_flag).

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
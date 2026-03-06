# Proposal Alignment Evaluation Report

- Run ID: `20260305T222101Z`
- Generated: `2026-03-05T22:23:30.881520+00:00`
- Scenarios: `artifacts\proposal_control_tuning\auto_tune_v3_smoke\20260305T221440Z\train_runs\trial_003\seed_19\20260305T222101Z\scenarios.jsonl`
- Scenario count: `12`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_tuned`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2854 (0.2198, 0.3512) | 0.2813 (0.2445, 0.3324) | 0.8989 (0.8609, 0.9299) | 0.4073 (0.3755, 0.4447) | n/a |
| proposed_contextual_controlled_tuned | 0.2870 (0.2392, 0.3428) | 0.4013 (0.3004, 0.5101) | 0.9112 (0.8679, 0.9433) | 0.4551 (0.4143, 0.4965) | n/a |
| proposed_contextual | 0.1510 (0.0740, 0.2357) | 0.1667 (0.1250, 0.2167) | 0.8500 (0.8003, 0.8973) | 0.2914 (0.2398, 0.3490) | n/a |
| candidate_no_context | 0.0315 (0.0116, 0.0573) | 0.1429 (0.1083, 0.1857) | 0.8190 (0.7749, 0.8650) | 0.2211 (0.1971, 0.2512) | n/a |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.1194 | 3.7911 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0238 | 0.1667 |
| proposed_vs_candidate_no_context | naturalness | 0.0309 | 0.0378 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.1515 | 5.0000 |
| proposed_vs_candidate_no_context | context_overlap | 0.0446 | 1.3001 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0298 | 2.5000 |
| proposed_vs_candidate_no_context | persona_style | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | distinct1 | 0.0156 | 0.0164 |
| proposed_vs_candidate_no_context | length_score | 0.0944 | 0.2857 |
| proposed_vs_candidate_no_context | sentence_score | 0.0583 | 0.0791 |
| proposed_vs_candidate_no_context | overall_quality | 0.0702 | 0.3176 |
| controlled_vs_proposed_raw | context_relevance | 0.1344 | 0.8905 |
| controlled_vs_proposed_raw | persona_consistency | 0.1146 | 0.6879 |
| controlled_vs_proposed_raw | naturalness | 0.0489 | 0.0575 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.1818 | 1.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0239 | 0.3022 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1464 | 3.5143 |
| controlled_vs_proposed_raw | persona_style | -0.0125 | -0.0187 |
| controlled_vs_proposed_raw | distinct1 | -0.0107 | -0.0112 |
| controlled_vs_proposed_raw | length_score | 0.2222 | 0.5229 |
| controlled_vs_proposed_raw | sentence_score | 0.0875 | 0.1099 |
| controlled_vs_proposed_raw | overall_quality | 0.1159 | 0.3977 |
| controlled_vs_candidate_no_context | context_relevance | 0.2539 | 8.0578 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1385 | 0.9692 |
| controlled_vs_candidate_no_context | naturalness | 0.0798 | 0.0975 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3333 | 11.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0685 | 1.9952 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1762 | 14.8000 |
| controlled_vs_candidate_no_context | persona_style | -0.0125 | -0.0187 |
| controlled_vs_candidate_no_context | distinct1 | 0.0048 | 0.0051 |
| controlled_vs_candidate_no_context | length_score | 0.3167 | 0.9580 |
| controlled_vs_candidate_no_context | sentence_score | 0.1458 | 0.1977 |
| controlled_vs_candidate_no_context | overall_quality | 0.1861 | 0.8417 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0017 | 0.0058 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.1200 | 0.4264 |
| controlled_alt_vs_controlled_default | naturalness | 0.0123 | 0.0137 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0000 | 0.0000 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0055 | 0.0539 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.1468 | 0.7806 |
| controlled_alt_vs_controlled_default | persona_style | 0.0125 | 0.0191 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0026 | -0.0028 |
| controlled_alt_vs_controlled_default | length_score | 0.0500 | 0.0773 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0333 | 0.0377 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0479 | 0.1175 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1361 | 0.9015 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.2346 | 1.4076 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0612 | 0.0720 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.1818 | 1.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0294 | 0.3723 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.2933 | 7.0381 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0000 | 0.0000 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0134 | -0.0139 |
| controlled_alt_vs_proposed_raw | length_score | 0.2722 | 0.6405 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1208 | 0.1518 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1637 | 0.5620 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2555 | 8.1105 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.2584 | 1.8089 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0921 | 0.1125 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3333 | 11.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0740 | 2.1565 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.3230 | 27.1333 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0000 | 0.0000 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0022 | 0.0023 |
| controlled_alt_vs_candidate_no_context | length_score | 0.3667 | 1.1092 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1792 | 0.2429 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.2340 | 1.0581 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.1194 | (0.0382, 0.2074) | 0.0003 | 0.1194 | (0.0001, 0.2548) | 0.0037 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0238 | (-0.0190, 0.0794) | 0.1700 | 0.0238 | (-0.0229, 0.1000) | 0.3020 |
| proposed_vs_candidate_no_context | naturalness | 0.0309 | (-0.0140, 0.0783) | 0.0890 | 0.0309 | (-0.0222, 0.1018) | 0.2153 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.1515 | (0.0530, 0.2652) | 0.0010 | 0.1515 | (0.0000, 0.3223) | 0.0550 |
| proposed_vs_candidate_no_context | context_overlap | 0.0446 | (0.0104, 0.0837) | 0.0053 | 0.0446 | (0.0005, 0.0974) | 0.0033 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0298 | (-0.0238, 0.0992) | 0.1777 | 0.0298 | (-0.0286, 0.1250) | 0.3190 |
| proposed_vs_candidate_no_context | persona_style | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | distinct1 | 0.0156 | (0.0020, 0.0325) | 0.0067 | 0.0156 | (0.0000, 0.0387) | 0.0670 |
| proposed_vs_candidate_no_context | length_score | 0.0944 | (-0.0639, 0.2501) | 0.1143 | 0.0944 | (-0.1111, 0.3364) | 0.2590 |
| proposed_vs_candidate_no_context | sentence_score | 0.0583 | (-0.0875, 0.2042) | 0.2593 | 0.0583 | (0.0000, 0.1750) | 0.3210 |
| proposed_vs_candidate_no_context | overall_quality | 0.0702 | (0.0094, 0.1383) | 0.0093 | 0.0702 | (-0.0115, 0.1788) | 0.0727 |
| controlled_vs_proposed_raw | context_relevance | 0.1344 | (0.0234, 0.2504) | 0.0077 | 0.1344 | (0.0167, 0.2420) | 0.0157 |
| controlled_vs_proposed_raw | persona_consistency | 0.1146 | (0.0550, 0.1684) | 0.0003 | 0.1146 | (0.0333, 0.1626) | 0.0030 |
| controlled_vs_proposed_raw | naturalness | 0.0489 | (0.0018, 0.1008) | 0.0187 | 0.0489 | (-0.0201, 0.1179) | 0.0563 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.1818 | (0.0379, 0.3333) | 0.0060 | 0.1818 | (0.0331, 0.3182) | 0.0027 |
| controlled_vs_proposed_raw | context_overlap | 0.0239 | (-0.0114, 0.0642) | 0.1050 | 0.0239 | (-0.0215, 0.0643) | 0.2073 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1464 | (0.0690, 0.2179) | 0.0007 | 0.1464 | (0.0417, 0.2095) | 0.0063 |
| controlled_vs_proposed_raw | persona_style | -0.0125 | (-0.0375, 0.0000) | 1.0000 | -0.0125 | (-0.0300, 0.0000) | 1.0000 |
| controlled_vs_proposed_raw | distinct1 | -0.0107 | (-0.0415, 0.0207) | 0.7400 | -0.0107 | (-0.0364, 0.0149) | 0.7617 |
| controlled_vs_proposed_raw | length_score | 0.2222 | (0.0138, 0.4389) | 0.0167 | 0.2222 | (-0.0278, 0.4722) | 0.0650 |
| controlled_vs_proposed_raw | sentence_score | 0.0875 | (-0.0583, 0.2333) | 0.1597 | 0.0875 | (-0.0538, 0.2450) | 0.1470 |
| controlled_vs_proposed_raw | overall_quality | 0.1159 | (0.0456, 0.1860) | 0.0007 | 0.1159 | (0.0199, 0.1975) | 0.0037 |
| controlled_vs_candidate_no_context | context_relevance | 0.2539 | (0.1741, 0.3314) | 0.0000 | 0.2539 | (0.2406, 0.2715) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1385 | (0.1023, 0.1795) | 0.0000 | 0.1385 | (0.1333, 0.1511) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.0798 | (0.0293, 0.1309) | 0.0003 | 0.0798 | (0.0452, 0.1068) | 0.0000 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3333 | (0.2348, 0.4318) | 0.0000 | 0.3333 | (0.3182, 0.3554) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0685 | (0.0405, 0.0958) | 0.0000 | 0.0685 | (0.0529, 0.0796) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1762 | (0.1294, 0.2329) | 0.0000 | 0.1762 | (0.1667, 0.1889) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | -0.0125 | (-0.0375, 0.0000) | 1.0000 | -0.0125 | (-0.0300, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | distinct1 | 0.0048 | (-0.0282, 0.0341) | 0.3940 | 0.0048 | (-0.0166, 0.0245) | 0.2823 |
| controlled_vs_candidate_no_context | length_score | 0.3167 | (0.1278, 0.4973) | 0.0000 | 0.3167 | (0.2028, 0.3970) | 0.0000 |
| controlled_vs_candidate_no_context | sentence_score | 0.1458 | (0.0000, 0.2917) | 0.0443 | 0.1458 | (0.0933, 0.2722) | 0.0000 |
| controlled_vs_candidate_no_context | overall_quality | 0.1861 | (0.1375, 0.2339) | 0.0000 | 0.1861 | (0.1764, 0.1972) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0017 | (-0.0793, 0.0834) | 0.4763 | 0.0017 | (-0.0330, 0.0363) | 0.4390 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.1200 | (0.0230, 0.2327) | 0.0047 | 0.1200 | (0.0267, 0.2444) | 0.0000 |
| controlled_alt_vs_controlled_default | naturalness | 0.0123 | (-0.0440, 0.0639) | 0.3153 | 0.0123 | (-0.0318, 0.0650) | 0.3000 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0000 | (-0.1061, 0.1061) | 0.5170 | 0.0000 | (-0.0505, 0.0455) | 0.5697 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0055 | (-0.0250, 0.0338) | 0.3507 | 0.0055 | (-0.0098, 0.0187) | 0.2623 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.1468 | (0.0294, 0.2881) | 0.0063 | 0.1468 | (0.0333, 0.3056) | 0.0030 |
| controlled_alt_vs_controlled_default | persona_style | 0.0125 | (0.0000, 0.0375) | 0.3280 | 0.0125 | (0.0000, 0.0300) | 0.3240 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0026 | (-0.0213, 0.0204) | 0.6080 | -0.0026 | (-0.0150, 0.0135) | 0.5793 |
| controlled_alt_vs_controlled_default | length_score | 0.0500 | (-0.1861, 0.2806) | 0.3503 | 0.0500 | (-0.1282, 0.3033) | 0.3127 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0333 | (-0.1292, 0.1750) | 0.2707 | 0.0333 | (-0.0750, 0.1400) | 0.3217 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0479 | (0.0059, 0.0894) | 0.0137 | 0.0479 | (0.0064, 0.1042) | 0.0033 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1361 | (0.0526, 0.2013) | 0.0017 | 0.1361 | (0.0455, 0.2090) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.2346 | (0.1203, 0.3562) | 0.0000 | 0.2346 | (0.0907, 0.3868) | 0.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0612 | (-0.0135, 0.1316) | 0.0550 | 0.0612 | (-0.0443, 0.1667) | 0.1373 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.1818 | (0.0758, 0.2652) | 0.0017 | 0.1818 | (0.0661, 0.2727) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0294 | (0.0025, 0.0552) | 0.0137 | 0.0294 | (-0.0026, 0.0604) | 0.0563 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.2933 | (0.1464, 0.4560) | 0.0000 | 0.2933 | (0.1133, 0.4835) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0134 | (-0.0403, 0.0141) | 0.8363 | -0.0134 | (-0.0316, 0.0199) | 0.8693 |
| controlled_alt_vs_proposed_raw | length_score | 0.2722 | (0.0111, 0.5333) | 0.0210 | 0.2722 | (-0.1333, 0.6778) | 0.0787 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1208 | (-0.0750, 0.2917) | 0.0940 | 0.1208 | (-0.1136, 0.2917) | 0.1233 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1637 | (0.0920, 0.2272) | 0.0000 | 0.1637 | (0.0607, 0.2245) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2555 | (0.2020, 0.3159) | 0.0000 | 0.2555 | (0.2092, 0.3019) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.2584 | (0.1584, 0.3683) | 0.0000 | 0.2584 | (0.1707, 0.3780) | 0.0000 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0921 | (0.0373, 0.1452) | 0.0007 | 0.0921 | (0.0398, 0.1534) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3333 | (0.2652, 0.4091) | 0.0000 | 0.3333 | (0.2727, 0.3939) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0740 | (0.0560, 0.1011) | 0.0000 | 0.0740 | (0.0609, 0.0947) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.3230 | (0.1980, 0.4663) | 0.0000 | 0.3230 | (0.2133, 0.4725) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0022 | (-0.0252, 0.0309) | 0.4333 | 0.0022 | (-0.0122, 0.0251) | 0.4577 |
| controlled_alt_vs_candidate_no_context | length_score | 0.3667 | (0.1500, 0.5612) | 0.0010 | 0.3667 | (0.1667, 0.5852) | 0.0000 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1792 | (0.0625, 0.2917) | 0.0017 | 0.1792 | (0.0667, 0.2917) | 0.0000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.2340 | (0.1932, 0.2767) | 0.0000 | 0.2340 | (0.1981, 0.2808) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 7 | 1 | 4 | 0.7500 | 0.8750 |
| proposed_vs_candidate_no_context | persona_consistency | 2 | 1 | 9 | 0.5417 | 0.6667 |
| proposed_vs_candidate_no_context | naturalness | 5 | 3 | 4 | 0.5833 | 0.6250 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 7 | 1 | 4 | 0.7500 | 0.8750 |
| proposed_vs_candidate_no_context | context_overlap | 6 | 2 | 4 | 0.6667 | 0.7500 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 2 | 1 | 9 | 0.5417 | 0.6667 |
| proposed_vs_candidate_no_context | persona_style | 0 | 0 | 12 | 0.5000 | nan |
| proposed_vs_candidate_no_context | distinct1 | 4 | 0 | 8 | 0.6667 | 1.0000 |
| proposed_vs_candidate_no_context | length_score | 5 | 3 | 4 | 0.5833 | 0.6250 |
| proposed_vs_candidate_no_context | sentence_score | 4 | 2 | 6 | 0.5833 | 0.6667 |
| proposed_vs_candidate_no_context | overall_quality | 7 | 1 | 4 | 0.7500 | 0.8750 |
| controlled_vs_proposed_raw | context_relevance | 8 | 4 | 0 | 0.6667 | 0.6667 |
| controlled_vs_proposed_raw | persona_consistency | 10 | 1 | 1 | 0.8750 | 0.9091 |
| controlled_vs_proposed_raw | naturalness | 8 | 4 | 0 | 0.6667 | 0.6667 |
| controlled_vs_proposed_raw | context_keyword_coverage | 8 | 2 | 2 | 0.7500 | 0.8000 |
| controlled_vs_proposed_raw | context_overlap | 7 | 5 | 0 | 0.5833 | 0.5833 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 10 | 1 | 1 | 0.8750 | 0.9091 |
| controlled_vs_proposed_raw | persona_style | 0 | 1 | 11 | 0.4583 | 0.0000 |
| controlled_vs_proposed_raw | distinct1 | 6 | 6 | 0 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | length_score | 6 | 5 | 1 | 0.5417 | 0.5455 |
| controlled_vs_proposed_raw | sentence_score | 5 | 2 | 5 | 0.6250 | 0.7143 |
| controlled_vs_proposed_raw | overall_quality | 9 | 3 | 0 | 0.7500 | 0.7500 |
| controlled_vs_candidate_no_context | context_relevance | 11 | 1 | 0 | 0.9167 | 0.9167 |
| controlled_vs_candidate_no_context | persona_consistency | 11 | 0 | 1 | 0.9583 | 1.0000 |
| controlled_vs_candidate_no_context | naturalness | 10 | 2 | 0 | 0.8333 | 0.8333 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 11 | 0 | 1 | 0.9583 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 11 | 1 | 0 | 0.9167 | 0.9167 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 11 | 0 | 1 | 0.9583 | 1.0000 |
| controlled_vs_candidate_no_context | persona_style | 0 | 1 | 11 | 0.4583 | 0.0000 |
| controlled_vs_candidate_no_context | distinct1 | 7 | 5 | 0 | 0.5833 | 0.5833 |
| controlled_vs_candidate_no_context | length_score | 9 | 2 | 1 | 0.7917 | 0.8182 |
| controlled_vs_candidate_no_context | sentence_score | 7 | 2 | 3 | 0.7083 | 0.7778 |
| controlled_vs_candidate_no_context | overall_quality | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 5 | 7 | 0 | 0.4167 | 0.4167 |
| controlled_alt_vs_controlled_default | persona_consistency | 7 | 1 | 4 | 0.7500 | 0.8750 |
| controlled_alt_vs_controlled_default | naturalness | 7 | 5 | 0 | 0.5833 | 0.5833 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 5 | 7 | 0 | 0.4167 | 0.4167 |
| controlled_alt_vs_controlled_default | context_overlap | 6 | 6 | 0 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 6 | 1 | 5 | 0.7083 | 0.8571 |
| controlled_alt_vs_controlled_default | persona_style | 1 | 0 | 11 | 0.5417 | 1.0000 |
| controlled_alt_vs_controlled_default | distinct1 | 4 | 8 | 0 | 0.3333 | 0.3333 |
| controlled_alt_vs_controlled_default | length_score | 5 | 5 | 2 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | sentence_score | 3 | 1 | 8 | 0.5833 | 0.7500 |
| controlled_alt_vs_controlled_default | overall_quality | 8 | 4 | 0 | 0.6667 | 0.6667 |
| controlled_alt_vs_proposed_raw | context_relevance | 10 | 2 | 0 | 0.8333 | 0.8333 |
| controlled_alt_vs_proposed_raw | persona_consistency | 10 | 1 | 1 | 0.8750 | 0.9091 |
| controlled_alt_vs_proposed_raw | naturalness | 8 | 4 | 0 | 0.6667 | 0.6667 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 10 | 1 | 1 | 0.8750 | 0.9091 |
| controlled_alt_vs_proposed_raw | context_overlap | 8 | 4 | 0 | 0.6667 | 0.6667 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 10 | 1 | 1 | 0.8750 | 0.9091 |
| controlled_alt_vs_proposed_raw | persona_style | 0 | 0 | 12 | 0.5000 | nan |
| controlled_alt_vs_proposed_raw | distinct1 | 4 | 8 | 0 | 0.3333 | 0.3333 |
| controlled_alt_vs_proposed_raw | length_score | 8 | 3 | 1 | 0.7083 | 0.7273 |
| controlled_alt_vs_proposed_raw | sentence_score | 7 | 2 | 3 | 0.7083 | 0.7778 |
| controlled_alt_vs_proposed_raw | overall_quality | 11 | 1 | 0 | 0.9167 | 0.9167 |
| controlled_alt_vs_candidate_no_context | context_relevance | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 11 | 0 | 1 | 0.9583 | 1.0000 |
| controlled_alt_vs_candidate_no_context | naturalness | 10 | 2 | 0 | 0.8333 | 0.8333 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 11 | 0 | 1 | 0.9583 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_style | 0 | 0 | 12 | 0.5000 | nan |
| controlled_alt_vs_candidate_no_context | distinct1 | 7 | 5 | 0 | 0.5833 | 0.5833 |
| controlled_alt_vs_candidate_no_context | length_score | 10 | 2 | 0 | 0.8333 | 0.8333 |
| controlled_alt_vs_candidate_no_context | sentence_score | 7 | 1 | 4 | 0.7500 | 0.8750 |
| controlled_alt_vs_candidate_no_context | overall_quality | 12 | 0 | 0 | 1.0000 | 1.0000 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.1667 | 0.5833 | 0.4167 |
| proposed_contextual_controlled_tuned | 0.0000 | 0.0000 | 0.2500 | 0.1667 | 0.8333 |
| proposed_contextual | 0.0000 | 0.0000 | 0.3333 | 0.0000 | 0.0000 |
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
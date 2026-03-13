# Proposal Alignment Evaluation Report

- Run ID: `20260313T080202Z`
- Generated: `2026-03-13T08:09:48.687895+00:00`
- Scenarios: `artifacts\proposal_control_tuning\auto_tune\20260313T051946Z\valid_runs\trial_007\seed_31\20260313T080202Z\scenarios.jsonl`
- Scenario count: `24`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_tuned`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.1008 (0.0629, 0.1468) | 0.3383 (0.2863, 0.3874) | 0.8744 (0.8591, 0.8877) | 0.3381 (0.3083, 0.3676) | n/a |
| proposed_contextual_controlled_tuned | 0.1583 (0.1019, 0.2309) | 0.3799 (0.3060, 0.4657) | 0.8669 (0.8520, 0.8812) | 0.3782 (0.3401, 0.4209) | n/a |
| proposed_contextual | 0.1073 (0.0669, 0.1567) | 0.2458 (0.1870, 0.3041) | 0.8823 (0.8696, 0.8953) | 0.3077 (0.2800, 0.3348) | n/a |
| candidate_no_context | 0.0452 (0.0282, 0.0638) | 0.2441 (0.1965, 0.2990) | 0.8654 (0.8521, 0.8779) | 0.2760 (0.2518, 0.3030) | n/a |

## Game-facing Outcome Metrics (mean, 95% CI)
| Arm | Quest-state Correctness | Lore Consistency | Contradiction Safety | Objective Completion Support | Gameplay Usefulness | Time-pressure Acceptability |
|---|---:|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.1950 (0.1598, 0.2336) | 0.0577 (0.0195, 0.1075) | 1.0000 (1.0000, 1.0000) | 0.1279 (0.0816, 0.1718) | 0.3227 (0.2984, 0.3461) | 0.3277 (0.2985, 0.3592) |
| proposed_contextual_controlled_tuned | 0.2416 (0.1939, 0.2987) | 0.0943 (0.0427, 0.1556) | 1.0000 (1.0000, 1.0000) | 0.0959 (0.0543, 0.1429) | 0.3181 (0.2986, 0.3378) | 0.3033 (0.2734, 0.3341) |
| proposed_contextual | 0.1929 (0.1604, 0.2334) | 0.0624 (0.0255, 0.1108) | 1.0000 (1.0000, 1.0000) | 0.1007 (0.0537, 0.1510) | 0.3168 (0.2928, 0.3413) | 0.3182 (0.2882, 0.3484) |
| candidate_no_context | 0.1416 (0.1244, 0.1603) | 0.0031 (0.0002, 0.0069) | 1.0000 (1.0000, 1.0000) | 0.0847 (0.0432, 0.1336) | 0.2881 (0.2637, 0.3151) | 0.3119 (0.2851, 0.3417) |

- Multi-turn contradiction rate is reported as `1 - contradiction_safety` in row-level outputs.

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0621 | 1.3740 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0018 | 0.0073 |
| proposed_vs_candidate_no_context | naturalness | 0.0169 | 0.0195 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.0513 | 0.3625 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0593 | 19.2825 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | 0.0160 | 0.1894 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 0.0288 | 0.0998 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | 0.0064 | 0.0204 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0761 | 1.5855 |
| proposed_vs_candidate_no_context | context_overlap | 0.0296 | 0.7633 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0101 | -0.0635 |
| proposed_vs_candidate_no_context | persona_style | 0.0494 | 0.0848 |
| proposed_vs_candidate_no_context | distinct1 | 0.0081 | 0.0086 |
| proposed_vs_candidate_no_context | length_score | 0.0389 | 0.0812 |
| proposed_vs_candidate_no_context | sentence_score | 0.0583 | 0.0639 |
| proposed_vs_candidate_no_context | overall_quality | 0.0318 | 0.1151 |
| controlled_vs_proposed_raw | context_relevance | -0.0065 | -0.0607 |
| controlled_vs_proposed_raw | persona_consistency | 0.0924 | 0.3759 |
| controlled_vs_proposed_raw | naturalness | -0.0079 | -0.0089 |
| controlled_vs_proposed_raw | quest_state_correctness | 0.0021 | 0.0110 |
| controlled_vs_proposed_raw | lore_consistency | -0.0047 | -0.0755 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0272 | 0.2699 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0058 | 0.0184 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 0.0094 | 0.0297 |
| controlled_vs_proposed_raw | context_keyword_coverage | -0.0035 | -0.0280 |
| controlled_vs_proposed_raw | context_overlap | -0.0136 | -0.1995 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1141 | 0.7646 |
| controlled_vs_proposed_raw | persona_style | 0.0058 | 0.0091 |
| controlled_vs_proposed_raw | distinct1 | -0.0032 | -0.0033 |
| controlled_vs_proposed_raw | length_score | -0.0403 | -0.0777 |
| controlled_vs_proposed_raw | sentence_score | 0.0146 | 0.0150 |
| controlled_vs_proposed_raw | overall_quality | 0.0304 | 0.0987 |
| controlled_vs_candidate_no_context | context_relevance | 0.0556 | 1.2298 |
| controlled_vs_candidate_no_context | persona_consistency | 0.0942 | 0.3860 |
| controlled_vs_candidate_no_context | naturalness | 0.0090 | 0.0104 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.0535 | 0.3776 |
| controlled_vs_candidate_no_context | lore_consistency | 0.0546 | 17.7513 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | 0.0432 | 0.5105 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0346 | 0.1201 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 0.0158 | 0.0506 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.0726 | 1.5132 |
| controlled_vs_candidate_no_context | context_overlap | 0.0160 | 0.4116 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1040 | 0.6526 |
| controlled_vs_candidate_no_context | persona_style | 0.0552 | 0.0947 |
| controlled_vs_candidate_no_context | distinct1 | 0.0049 | 0.0052 |
| controlled_vs_candidate_no_context | length_score | -0.0014 | -0.0029 |
| controlled_vs_candidate_no_context | sentence_score | 0.0729 | 0.0799 |
| controlled_vs_candidate_no_context | overall_quality | 0.0621 | 0.2251 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0574 | 0.5697 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0417 | 0.1232 |
| controlled_alt_vs_controlled_default | naturalness | -0.0075 | -0.0086 |
| controlled_alt_vs_controlled_default | quest_state_correctness | 0.0466 | 0.2388 |
| controlled_alt_vs_controlled_default | lore_consistency | 0.0367 | 0.6365 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_controlled_default | objective_completion_support | -0.0320 | -0.2503 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | -0.0046 | -0.0142 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | -0.0243 | -0.0743 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0732 | 0.6073 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0206 | 0.3763 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0429 | 0.1628 |
| controlled_alt_vs_controlled_default | persona_style | 0.0370 | 0.0580 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0112 | -0.0117 |
| controlled_alt_vs_controlled_default | length_score | -0.0153 | -0.0320 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0000 | 0.0000 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0401 | 0.1186 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.0509 | 0.4744 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1341 | 0.5455 |
| controlled_alt_vs_proposed_raw | naturalness | -0.0154 | -0.0175 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 0.0487 | 0.2524 |
| controlled_alt_vs_proposed_raw | lore_consistency | 0.0320 | 0.5129 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_proposed_raw | objective_completion_support | -0.0048 | -0.0479 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 0.0012 | 0.0039 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | -0.0149 | -0.0469 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.0698 | 0.5623 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0070 | 0.1018 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1569 | 1.0519 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0427 | 0.0676 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0144 | -0.0151 |
| controlled_alt_vs_proposed_raw | length_score | -0.0556 | -0.1072 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0146 | 0.0150 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.0705 | 0.2290 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.1130 | 2.5002 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1359 | 0.5568 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0015 | 0.0017 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 0.1000 | 0.7065 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 0.0913 | 29.6862 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_candidate_no_context | objective_completion_support | 0.0112 | 0.1324 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 0.0300 | 0.1041 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | -0.0086 | -0.0274 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.1458 | 3.0395 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0366 | 0.9428 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1468 | 0.9215 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0922 | 0.1581 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0063 | -0.0066 |
| controlled_alt_vs_candidate_no_context | length_score | -0.0167 | -0.0348 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.0729 | 0.0799 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1022 | 0.3704 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0621 | (0.0217, 0.1127) | 0.0000 | 0.0621 | (-0.0005, 0.1032) | 0.0407 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0018 | (-0.0373, 0.0460) | 0.4640 | 0.0018 | (-0.0348, 0.0672) | 0.4147 |
| proposed_vs_candidate_no_context | naturalness | 0.0169 | (0.0008, 0.0330) | 0.0207 | 0.0169 | (0.0059, 0.0282) | 0.0000 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.0513 | (0.0194, 0.0915) | 0.0000 | 0.0513 | (-0.0020, 0.0890) | 0.0380 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0593 | (0.0218, 0.1075) | 0.0000 | 0.0593 | (0.0212, 0.0810) | 0.0000 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | 0.0160 | (-0.0097, 0.0449) | 0.1100 | 0.0160 | (-0.0079, 0.0446) | 0.1483 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 0.0288 | (0.0121, 0.0470) | 0.0000 | 0.0288 | (0.0223, 0.0361) | 0.0000 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | 0.0064 | (-0.0081, 0.0232) | 0.2170 | 0.0064 | (-0.0127, 0.0283) | 0.2637 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0761 | (0.0253, 0.1364) | 0.0000 | 0.0761 | (0.0000, 0.1310) | 0.0397 |
| proposed_vs_candidate_no_context | context_overlap | 0.0296 | (0.0058, 0.0660) | 0.0033 | 0.0296 | (-0.0017, 0.0506) | 0.0337 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0101 | (-0.0583, 0.0413) | 0.6850 | -0.0101 | (-0.0667, 0.0714) | 0.6197 |
| proposed_vs_candidate_no_context | persona_style | 0.0494 | (-0.0148, 0.1343) | 0.0773 | 0.0494 | (0.0000, 0.0926) | 0.0377 |
| proposed_vs_candidate_no_context | distinct1 | 0.0081 | (-0.0079, 0.0256) | 0.1697 | 0.0081 | (-0.0076, 0.0241) | 0.1433 |
| proposed_vs_candidate_no_context | length_score | 0.0389 | (-0.0292, 0.1070) | 0.1257 | 0.0389 | (0.0000, 0.0926) | 0.0333 |
| proposed_vs_candidate_no_context | sentence_score | 0.0583 | (0.0146, 0.1167) | 0.0100 | 0.0583 | (0.0000, 0.1500) | 0.0353 |
| proposed_vs_candidate_no_context | overall_quality | 0.0318 | (0.0053, 0.0627) | 0.0097 | 0.0318 | (-0.0044, 0.0750) | 0.0340 |
| controlled_vs_proposed_raw | context_relevance | -0.0065 | (-0.0583, 0.0468) | 0.5943 | -0.0065 | (-0.0395, 0.0360) | 0.6310 |
| controlled_vs_proposed_raw | persona_consistency | 0.0924 | (0.0383, 0.1503) | 0.0000 | 0.0924 | (0.0571, 0.1516) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | -0.0079 | (-0.0290, 0.0112) | 0.7623 | -0.0079 | (-0.0127, -0.0001) | 1.0000 |
| controlled_vs_proposed_raw | quest_state_correctness | 0.0021 | (-0.0432, 0.0499) | 0.4707 | 0.0021 | (-0.0264, 0.0477) | 0.4003 |
| controlled_vs_proposed_raw | lore_consistency | -0.0047 | (-0.0598, 0.0526) | 0.5720 | -0.0047 | (-0.0356, 0.0299) | 0.6240 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0272 | (-0.0012, 0.0596) | 0.0297 | 0.0272 | (0.0091, 0.0562) | 0.0000 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0058 | (-0.0198, 0.0304) | 0.3453 | 0.0058 | (-0.0103, 0.0172) | 0.2627 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 0.0094 | (-0.0072, 0.0280) | 0.1510 | 0.0094 | (-0.0001, 0.0227) | 0.0440 |
| controlled_vs_proposed_raw | context_keyword_coverage | -0.0035 | (-0.0663, 0.0638) | 0.5493 | -0.0035 | (-0.0404, 0.0455) | 0.6167 |
| controlled_vs_proposed_raw | context_overlap | -0.0136 | (-0.0481, 0.0105) | 0.8110 | -0.0136 | (-0.0373, 0.0138) | 0.8567 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1141 | (0.0466, 0.1845) | 0.0000 | 0.1141 | (0.0714, 0.1667) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.0058 | (-0.0427, 0.0603) | 0.4103 | 0.0058 | (-0.0556, 0.0912) | 0.4023 |
| controlled_vs_proposed_raw | distinct1 | -0.0032 | (-0.0227, 0.0165) | 0.6327 | -0.0032 | (-0.0183, 0.0076) | 0.7047 |
| controlled_vs_proposed_raw | length_score | -0.0403 | (-0.1139, 0.0375) | 0.8500 | -0.0403 | (-0.0857, -0.0074) | 1.0000 |
| controlled_vs_proposed_raw | sentence_score | 0.0146 | (-0.0292, 0.0729) | 0.3770 | 0.0146 | (-0.0389, 0.0500) | 0.2533 |
| controlled_vs_proposed_raw | overall_quality | 0.0304 | (0.0025, 0.0602) | 0.0150 | 0.0304 | (0.0100, 0.0478) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.0556 | (0.0193, 0.0985) | 0.0000 | 0.0556 | (0.0354, 0.0905) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.0942 | (0.0420, 0.1523) | 0.0000 | 0.0942 | (0.0429, 0.2188) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.0090 | (-0.0097, 0.0269) | 0.1677 | 0.0090 | (0.0043, 0.0155) | 0.0000 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.0535 | (0.0200, 0.0909) | 0.0000 | 0.0535 | (0.0431, 0.0757) | 0.0000 |
| controlled_vs_candidate_no_context | lore_consistency | 0.0546 | (0.0131, 0.1026) | 0.0023 | 0.0546 | (0.0407, 0.0764) | 0.0000 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | 0.0432 | (0.0045, 0.0860) | 0.0153 | 0.0432 | (0.0183, 0.0668) | 0.0000 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0346 | (0.0118, 0.0578) | 0.0027 | 0.0346 | (0.0257, 0.0440) | 0.0000 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 0.0158 | (-0.0107, 0.0407) | 0.1250 | 0.0158 | (0.0016, 0.0369) | 0.0000 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.0726 | (0.0224, 0.1282) | 0.0013 | 0.0726 | (0.0455, 0.1190) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0160 | (0.0053, 0.0295) | 0.0007 | 0.0160 | (0.0121, 0.0239) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1040 | (0.0417, 0.1685) | 0.0003 | 0.1040 | (0.0444, 0.2381) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.0552 | (0.0062, 0.1289) | 0.0120 | 0.0552 | (0.0000, 0.1416) | 0.0343 |
| controlled_vs_candidate_no_context | distinct1 | 0.0049 | (-0.0100, 0.0212) | 0.2647 | 0.0049 | (-0.0036, 0.0114) | 0.0363 |
| controlled_vs_candidate_no_context | length_score | -0.0014 | (-0.0833, 0.0764) | 0.5017 | -0.0014 | (-0.0714, 0.0852) | 0.6247 |
| controlled_vs_candidate_no_context | sentence_score | 0.0729 | (0.0000, 0.1458) | 0.0273 | 0.0729 | (-0.0389, 0.2000) | 0.1463 |
| controlled_vs_candidate_no_context | overall_quality | 0.0621 | (0.0313, 0.0932) | 0.0000 | 0.0621 | (0.0336, 0.1229) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0574 | (-0.0036, 0.1330) | 0.0353 | 0.0574 | (-0.0093, 0.1011) | 0.0377 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0417 | (-0.0241, 0.1264) | 0.1317 | 0.0417 | (-0.0095, 0.0830) | 0.0290 |
| controlled_alt_vs_controlled_default | naturalness | -0.0075 | (-0.0277, 0.0121) | 0.7787 | -0.0075 | (-0.0149, 0.0016) | 0.9620 |
| controlled_alt_vs_controlled_default | quest_state_correctness | 0.0466 | (-0.0059, 0.1105) | 0.0483 | 0.0466 | (-0.0132, 0.0832) | 0.0390 |
| controlled_alt_vs_controlled_default | lore_consistency | 0.0367 | (-0.0278, 0.1048) | 0.1360 | 0.0367 | (-0.0284, 0.0848) | 0.1560 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | objective_completion_support | -0.0320 | (-0.0623, -0.0071) | 1.0000 | -0.0320 | (-0.0602, -0.0137) | 1.0000 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | -0.0046 | (-0.0272, 0.0193) | 0.6687 | -0.0046 | (-0.0351, 0.0179) | 0.6030 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | -0.0243 | (-0.0469, -0.0029) | 0.9900 | -0.0243 | (-0.0435, -0.0093) | 1.0000 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0732 | (-0.0035, 0.1657) | 0.0317 | 0.0732 | (-0.0114, 0.1313) | 0.0377 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0206 | (-0.0019, 0.0475) | 0.0367 | 0.0206 | (-0.0044, 0.0362) | 0.0427 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0429 | (-0.0429, 0.1476) | 0.1900 | 0.0429 | (0.0000, 0.0667) | 0.0330 |
| controlled_alt_vs_controlled_default | persona_style | 0.0370 | (-0.0137, 0.0922) | 0.0797 | 0.0370 | (-0.0476, 0.1481) | 0.3127 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0112 | (-0.0230, -0.0002) | 0.9760 | -0.0112 | (-0.0206, -0.0006) | 1.0000 |
| controlled_alt_vs_controlled_default | length_score | -0.0153 | (-0.0972, 0.0681) | 0.6447 | -0.0153 | (-0.0333, 0.0148) | 0.8603 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0000 | (-0.0437, 0.0437) | 0.6513 | 0.0000 | (-0.0437, 0.0389) | 0.6280 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0401 | (0.0076, 0.0766) | 0.0053 | 0.0401 | (0.0087, 0.0761) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.0509 | (-0.0127, 0.1230) | 0.0623 | 0.0509 | (0.0267, 0.0649) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1341 | (0.0583, 0.2166) | 0.0000 | 0.1341 | (0.0972, 0.1607) | 0.0000 |
| controlled_alt_vs_proposed_raw | naturalness | -0.0154 | (-0.0319, 0.0013) | 0.9660 | -0.0154 | (-0.0255, -0.0111) | 1.0000 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 0.0487 | (-0.0072, 0.1109) | 0.0500 | 0.0487 | (0.0345, 0.0568) | 0.0000 |
| controlled_alt_vs_proposed_raw | lore_consistency | 0.0320 | (-0.0292, 0.0967) | 0.1600 | 0.0320 | (0.0015, 0.0492) | 0.0000 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_proposed_raw | objective_completion_support | -0.0048 | (-0.0499, 0.0383) | 0.5877 | -0.0048 | (-0.0380, 0.0329) | 0.6200 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 0.0012 | (-0.0238, 0.0265) | 0.4560 | 0.0012 | (-0.0211, 0.0187) | 0.3870 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | -0.0149 | (-0.0439, 0.0139) | 0.8507 | -0.0149 | (-0.0349, 0.0009) | 0.9560 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.0698 | (-0.0107, 0.1622) | 0.0487 | 0.0698 | (0.0341, 0.0909) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0070 | (-0.0313, 0.0404) | 0.3383 | 0.0070 | (-0.0067, 0.0217) | 0.1463 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1569 | (0.0692, 0.2522) | 0.0000 | 0.1569 | (0.1250, 0.1778) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0427 | (-0.0108, 0.1040) | 0.0650 | 0.0427 | (-0.0141, 0.0926) | 0.0357 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0144 | (-0.0360, 0.0070) | 0.9027 | -0.0144 | (-0.0315, 0.0070) | 0.9667 |
| controlled_alt_vs_proposed_raw | length_score | -0.0556 | (-0.1306, 0.0236) | 0.9180 | -0.0556 | (-0.1190, 0.0074) | 0.9647 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0146 | (-0.0292, 0.0583) | 0.3927 | 0.0146 | (0.0000, 0.0500) | 0.2810 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.0705 | (0.0281, 0.1186) | 0.0007 | 0.0705 | (0.0467, 0.0861) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.1130 | (0.0527, 0.1852) | 0.0000 | 0.1130 | (0.0262, 0.1680) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1359 | (0.0561, 0.2218) | 0.0000 | 0.1359 | (0.0829, 0.2093) | 0.0000 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0015 | (-0.0177, 0.0198) | 0.4453 | 0.0015 | (-0.0107, 0.0170) | 0.2967 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 0.1000 | (0.0481, 0.1643) | 0.0000 | 0.1000 | (0.0325, 0.1434) | 0.0000 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 0.0913 | (0.0420, 0.1512) | 0.0000 | 0.0913 | (0.0226, 0.1257) | 0.0000 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_candidate_no_context | objective_completion_support | 0.0112 | (-0.0335, 0.0571) | 0.3270 | 0.0112 | (0.0046, 0.0250) | 0.0000 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 0.0300 | (0.0016, 0.0563) | 0.0203 | 0.0300 | (0.0012, 0.0454) | 0.0000 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | -0.0086 | (-0.0393, 0.0217) | 0.7090 | -0.0086 | (-0.0118, -0.0066) | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.1458 | (0.0666, 0.2393) | 0.0000 | 0.1458 | (0.0341, 0.2143) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0366 | (0.0163, 0.0615) | 0.0000 | 0.0366 | (0.0077, 0.0601) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1468 | (0.0556, 0.2484) | 0.0010 | 0.1468 | (0.1071, 0.2381) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0922 | (0.0132, 0.1898) | 0.0103 | 0.0922 | (-0.0141, 0.1852) | 0.0410 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0063 | (-0.0234, 0.0105) | 0.7607 | -0.0063 | (-0.0243, 0.0108) | 0.7520 |
| controlled_alt_vs_candidate_no_context | length_score | -0.0167 | (-0.1111, 0.0750) | 0.6397 | -0.0167 | (-0.1048, 0.1000) | 0.7413 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.0729 | (0.0000, 0.1458) | 0.0300 | 0.0729 | (0.0000, 0.2000) | 0.0353 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1022 | (0.0561, 0.1510) | 0.0000 | 0.1022 | (0.0424, 0.1526) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 11 | 3 | 10 | 0.6667 | 0.7857 |
| proposed_vs_candidate_no_context | persona_consistency | 4 | 6 | 14 | 0.4583 | 0.4000 |
| proposed_vs_candidate_no_context | naturalness | 10 | 4 | 10 | 0.6250 | 0.7143 |
| proposed_vs_candidate_no_context | quest_state_correctness | 11 | 3 | 10 | 0.6667 | 0.7857 |
| proposed_vs_candidate_no_context | lore_consistency | 11 | 1 | 12 | 0.7083 | 0.9167 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| proposed_vs_candidate_no_context | objective_completion_support | 6 | 4 | 14 | 0.5417 | 0.6000 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 12 | 2 | 10 | 0.7083 | 0.8571 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | 6 | 4 | 14 | 0.5417 | 0.6000 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 9 | 1 | 14 | 0.6667 | 0.9000 |
| proposed_vs_candidate_no_context | context_overlap | 11 | 3 | 10 | 0.6667 | 0.7857 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 2 | 5 | 17 | 0.4375 | 0.2857 |
| proposed_vs_candidate_no_context | persona_style | 4 | 2 | 18 | 0.5417 | 0.6667 |
| proposed_vs_candidate_no_context | distinct1 | 8 | 4 | 12 | 0.5833 | 0.6667 |
| proposed_vs_candidate_no_context | length_score | 10 | 4 | 10 | 0.6250 | 0.7143 |
| proposed_vs_candidate_no_context | sentence_score | 4 | 0 | 20 | 0.5833 | 1.0000 |
| proposed_vs_candidate_no_context | overall_quality | 10 | 4 | 10 | 0.6250 | 0.7143 |
| controlled_vs_proposed_raw | context_relevance | 11 | 11 | 2 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | persona_consistency | 11 | 2 | 11 | 0.6875 | 0.8462 |
| controlled_vs_proposed_raw | naturalness | 10 | 12 | 2 | 0.4583 | 0.4545 |
| controlled_vs_proposed_raw | quest_state_correctness | 11 | 11 | 2 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | lore_consistency | 4 | 9 | 11 | 0.3958 | 0.3077 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_vs_proposed_raw | objective_completion_support | 8 | 7 | 9 | 0.5208 | 0.5333 |
| controlled_vs_proposed_raw | gameplay_usefulness | 13 | 9 | 2 | 0.5833 | 0.5909 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 9 | 8 | 7 | 0.5208 | 0.5294 |
| controlled_vs_proposed_raw | context_keyword_coverage | 4 | 6 | 14 | 0.4583 | 0.4000 |
| controlled_vs_proposed_raw | context_overlap | 9 | 13 | 2 | 0.4167 | 0.4091 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 10 | 1 | 13 | 0.6875 | 0.9091 |
| controlled_vs_proposed_raw | persona_style | 4 | 4 | 16 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | distinct1 | 13 | 7 | 4 | 0.6250 | 0.6500 |
| controlled_vs_proposed_raw | length_score | 9 | 12 | 3 | 0.4375 | 0.4286 |
| controlled_vs_proposed_raw | sentence_score | 2 | 1 | 21 | 0.5208 | 0.6667 |
| controlled_vs_proposed_raw | overall_quality | 13 | 9 | 2 | 0.5833 | 0.5909 |
| controlled_vs_candidate_no_context | context_relevance | 14 | 10 | 0 | 0.5833 | 0.5833 |
| controlled_vs_candidate_no_context | persona_consistency | 10 | 1 | 13 | 0.6875 | 0.9091 |
| controlled_vs_candidate_no_context | naturalness | 15 | 9 | 0 | 0.6250 | 0.6250 |
| controlled_vs_candidate_no_context | quest_state_correctness | 15 | 9 | 0 | 0.6250 | 0.6250 |
| controlled_vs_candidate_no_context | lore_consistency | 6 | 4 | 14 | 0.5417 | 0.6000 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_vs_candidate_no_context | objective_completion_support | 10 | 7 | 7 | 0.5625 | 0.5882 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 19 | 5 | 0 | 0.7917 | 0.7917 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 9 | 8 | 7 | 0.5208 | 0.5294 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 9 | 1 | 14 | 0.6667 | 0.9000 |
| controlled_vs_candidate_no_context | context_overlap | 14 | 10 | 0 | 0.5833 | 0.5833 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 9 | 1 | 14 | 0.6667 | 0.9000 |
| controlled_vs_candidate_no_context | persona_style | 4 | 1 | 19 | 0.5625 | 0.8000 |
| controlled_vs_candidate_no_context | distinct1 | 15 | 6 | 3 | 0.6875 | 0.7143 |
| controlled_vs_candidate_no_context | length_score | 12 | 10 | 2 | 0.5417 | 0.5455 |
| controlled_vs_candidate_no_context | sentence_score | 6 | 1 | 17 | 0.6042 | 0.8571 |
| controlled_vs_candidate_no_context | overall_quality | 19 | 5 | 0 | 0.7917 | 0.7917 |
| controlled_alt_vs_controlled_default | context_relevance | 9 | 3 | 12 | 0.6250 | 0.7500 |
| controlled_alt_vs_controlled_default | persona_consistency | 5 | 5 | 14 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | naturalness | 5 | 7 | 12 | 0.4583 | 0.4167 |
| controlled_alt_vs_controlled_default | quest_state_correctness | 9 | 3 | 12 | 0.6250 | 0.7500 |
| controlled_alt_vs_controlled_default | lore_consistency | 9 | 3 | 12 | 0.6250 | 0.7500 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_alt_vs_controlled_default | objective_completion_support | 0 | 6 | 18 | 0.3750 | 0.0000 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | 6 | 6 | 12 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | 3 | 8 | 13 | 0.3958 | 0.2727 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 6 | 3 | 15 | 0.5625 | 0.6667 |
| controlled_alt_vs_controlled_default | context_overlap | 9 | 3 | 12 | 0.6250 | 0.7500 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 3 | 3 | 18 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | persona_style | 5 | 3 | 16 | 0.5417 | 0.6250 |
| controlled_alt_vs_controlled_default | distinct1 | 2 | 10 | 12 | 0.3333 | 0.1667 |
| controlled_alt_vs_controlled_default | length_score | 5 | 6 | 13 | 0.4792 | 0.4545 |
| controlled_alt_vs_controlled_default | sentence_score | 1 | 1 | 22 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | overall_quality | 7 | 5 | 12 | 0.5417 | 0.5833 |
| controlled_alt_vs_proposed_raw | context_relevance | 16 | 8 | 0 | 0.6667 | 0.6667 |
| controlled_alt_vs_proposed_raw | persona_consistency | 13 | 3 | 8 | 0.7083 | 0.8125 |
| controlled_alt_vs_proposed_raw | naturalness | 8 | 16 | 0 | 0.3333 | 0.3333 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 16 | 8 | 0 | 0.6667 | 0.6667 |
| controlled_alt_vs_proposed_raw | lore_consistency | 10 | 7 | 7 | 0.5625 | 0.5882 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_alt_vs_proposed_raw | objective_completion_support | 7 | 9 | 8 | 0.4583 | 0.4375 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 13 | 11 | 0 | 0.5417 | 0.5417 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | 10 | 10 | 4 | 0.5000 | 0.5000 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 9 | 4 | 11 | 0.6042 | 0.6923 |
| controlled_alt_vs_proposed_raw | context_overlap | 16 | 8 | 0 | 0.6667 | 0.6667 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 13 | 3 | 8 | 0.7083 | 0.8125 |
| controlled_alt_vs_proposed_raw | persona_style | 6 | 3 | 15 | 0.5625 | 0.6667 |
| controlled_alt_vs_proposed_raw | distinct1 | 11 | 11 | 2 | 0.5000 | 0.5000 |
| controlled_alt_vs_proposed_raw | length_score | 8 | 13 | 3 | 0.3958 | 0.3810 |
| controlled_alt_vs_proposed_raw | sentence_score | 2 | 1 | 21 | 0.5208 | 0.6667 |
| controlled_alt_vs_proposed_raw | overall_quality | 16 | 8 | 0 | 0.6667 | 0.6667 |
| controlled_alt_vs_candidate_no_context | context_relevance | 16 | 8 | 0 | 0.6667 | 0.6667 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 14 | 2 | 8 | 0.7500 | 0.8750 |
| controlled_alt_vs_candidate_no_context | naturalness | 13 | 11 | 0 | 0.5417 | 0.5417 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 17 | 7 | 0 | 0.7083 | 0.7083 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 11 | 3 | 10 | 0.6667 | 0.7857 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_alt_vs_candidate_no_context | objective_completion_support | 7 | 9 | 8 | 0.4583 | 0.4375 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 17 | 7 | 0 | 0.7083 | 0.7083 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | 8 | 11 | 5 | 0.4375 | 0.4211 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 13 | 2 | 9 | 0.7292 | 0.8667 |
| controlled_alt_vs_candidate_no_context | context_overlap | 17 | 7 | 0 | 0.7083 | 0.7083 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 11 | 2 | 11 | 0.6875 | 0.8462 |
| controlled_alt_vs_candidate_no_context | persona_style | 7 | 3 | 14 | 0.5833 | 0.7000 |
| controlled_alt_vs_candidate_no_context | distinct1 | 12 | 10 | 2 | 0.5417 | 0.5455 |
| controlled_alt_vs_candidate_no_context | length_score | 12 | 9 | 3 | 0.5625 | 0.5714 |
| controlled_alt_vs_candidate_no_context | sentence_score | 6 | 1 | 17 | 0.6042 | 0.8571 |
| controlled_alt_vs_candidate_no_context | overall_quality | 18 | 6 | 0 | 0.7500 | 0.7500 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.2917 | 0.1250 | 0.8750 |
| proposed_contextual_controlled_tuned | 0.0000 | 0.0000 | 0.2083 | 0.2917 | 0.7083 |
| proposed_contextual | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.5833 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `3`
- Unique template signatures: `18`
- Template signature ratio: `0.7500`
- Effective sample size by source clustering: `2.97`
- Effective sample size by template-signature clustering: `16.00`
- Detailed diagnostics are published in `scenario_dependence.json`.

## Multi-turn Contradiction
| Arm | Contradiction Rate | Contradiction Safety | Contradicted Sources | Source Count |
|---|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 1.0000 | 0 | 3 |
| proposed_contextual_controlled_tuned | 0.0000 | 1.0000 | 0 | 3 |
| proposed_contextual | 0.0000 | 1.0000 | 0 | 3 |
| candidate_no_context | 0.0000 | 1.0000 | 0 | 3 |
- Detailed source-level values are published in `multi_turn_contradictions.json`.

- BERTScore status: unavailable (disabled_by_flag).

This report covers proposal RO5 metrics plus game-facing outcomes: quest-state correctness, lore consistency, contradiction safety, objective completion support, gameplay usefulness, and time-pressure acceptability.
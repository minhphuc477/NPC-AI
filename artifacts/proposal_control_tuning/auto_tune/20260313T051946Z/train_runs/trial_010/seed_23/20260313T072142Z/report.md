# Proposal Alignment Evaluation Report

- Run ID: `20260313T072142Z`
- Generated: `2026-03-13T07:29:19.606782+00:00`
- Scenarios: `artifacts\proposal_control_tuning\auto_tune\20260313T051946Z\train_runs\trial_010\seed_23\20260313T072142Z\scenarios.jsonl`
- Scenario count: `24`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_tuned`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.1050 (0.0580, 0.1562) | 0.2666 (0.2193, 0.3220) | 0.8693 (0.8490, 0.8888) | 0.3144 (0.2811, 0.3498) | n/a |
| proposed_contextual_controlled_tuned | 0.0826 (0.0368, 0.1390) | 0.2392 (0.1908, 0.2943) | 0.8837 (0.8674, 0.9006) | 0.2965 (0.2660, 0.3309) | n/a |
| proposed_contextual | 0.0741 (0.0485, 0.1001) | 0.2151 (0.1532, 0.2801) | 0.8784 (0.8614, 0.8967) | 0.2821 (0.2555, 0.3106) | n/a |
| candidate_no_context | 0.0326 (0.0205, 0.0463) | 0.2352 (0.1657, 0.3110) | 0.8696 (0.8552, 0.8844) | 0.2687 (0.2381, 0.3005) | n/a |

## Game-facing Outcome Metrics (mean, 95% CI)
| Arm | Quest-state Correctness | Lore Consistency | Contradiction Safety | Objective Completion Support | Gameplay Usefulness | Time-pressure Acceptability |
|---|---:|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.1841 (0.1455, 0.2300) | 0.0404 (0.0124, 0.0747) | 1.0000 (1.0000, 1.0000) | 0.0928 (0.0662, 0.1188) | 0.3062 (0.2878, 0.3304) | 0.2917 (0.2775, 0.3050) |
| proposed_contextual_controlled_tuned | 0.1711 (0.1320, 0.2205) | 0.0428 (0.0097, 0.0804) | 1.0000 (1.0000, 1.0000) | 0.0883 (0.0640, 0.1117) | 0.3117 (0.2928, 0.3340) | 0.3022 (0.2887, 0.3166) |
| proposed_contextual | 0.1637 (0.1390, 0.1899) | 0.0240 (0.0091, 0.0413) | 1.0000 (1.0000, 1.0000) | 0.0567 (0.0340, 0.0810) | 0.2940 (0.2838, 0.3048) | 0.2899 (0.2764, 0.3040) |
| candidate_no_context | 0.1253 (0.1150, 0.1375) | 0.0062 (0.0017, 0.0125) | 1.0000 (1.0000, 1.0000) | 0.0579 (0.0355, 0.0832) | 0.2761 (0.2627, 0.2885) | 0.2879 (0.2754, 0.3014) |

- Multi-turn contradiction rate is reported as `1 - contradiction_safety` in row-level outputs.

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0415 | 1.2733 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0201 | -0.0855 |
| proposed_vs_candidate_no_context | naturalness | 0.0088 | 0.0101 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.0384 | 0.3068 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0178 | 2.8503 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | -0.0013 | -0.0216 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 0.0179 | 0.0647 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | 0.0020 | 0.0068 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0527 | 1.7160 |
| proposed_vs_candidate_no_context | context_overlap | 0.0155 | 0.4174 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0248 | -0.1488 |
| proposed_vs_candidate_no_context | persona_style | -0.0013 | -0.0026 |
| proposed_vs_candidate_no_context | distinct1 | 0.0017 | 0.0018 |
| proposed_vs_candidate_no_context | length_score | 0.0333 | 0.0652 |
| proposed_vs_candidate_no_context | sentence_score | 0.0146 | 0.0160 |
| proposed_vs_candidate_no_context | overall_quality | 0.0134 | 0.0499 |
| controlled_vs_proposed_raw | context_relevance | 0.0309 | 0.4172 |
| controlled_vs_proposed_raw | persona_consistency | 0.0515 | 0.2395 |
| controlled_vs_proposed_raw | naturalness | -0.0091 | -0.0104 |
| controlled_vs_proposed_raw | quest_state_correctness | 0.0204 | 0.1245 |
| controlled_vs_proposed_raw | lore_consistency | 0.0164 | 0.6837 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0361 | 0.6373 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0122 | 0.0415 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 0.0018 | 0.0062 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.0420 | 0.5045 |
| controlled_vs_proposed_raw | context_overlap | 0.0049 | 0.0937 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.0712 | 0.5021 |
| controlled_vs_proposed_raw | persona_style | -0.0273 | -0.0538 |
| controlled_vs_proposed_raw | distinct1 | -0.0042 | -0.0045 |
| controlled_vs_proposed_raw | length_score | -0.0528 | -0.0969 |
| controlled_vs_proposed_raw | sentence_score | 0.0312 | 0.0337 |
| controlled_vs_proposed_raw | overall_quality | 0.0324 | 0.1148 |
| controlled_vs_candidate_no_context | context_relevance | 0.0724 | 2.2217 |
| controlled_vs_candidate_no_context | persona_consistency | 0.0314 | 0.1336 |
| controlled_vs_candidate_no_context | naturalness | -0.0003 | -0.0004 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.0588 | 0.4695 |
| controlled_vs_candidate_no_context | lore_consistency | 0.0342 | 5.4826 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | 0.0349 | 0.6019 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0301 | 0.1089 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 0.0038 | 0.0131 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.0947 | 3.0864 |
| controlled_vs_candidate_no_context | context_overlap | 0.0204 | 0.5503 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.0464 | 0.2786 |
| controlled_vs_candidate_no_context | persona_style | -0.0286 | -0.0563 |
| controlled_vs_candidate_no_context | distinct1 | -0.0025 | -0.0027 |
| controlled_vs_candidate_no_context | length_score | -0.0194 | -0.0380 |
| controlled_vs_candidate_no_context | sentence_score | 0.0458 | 0.0502 |
| controlled_vs_candidate_no_context | overall_quality | 0.0458 | 0.1704 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0224 | -0.2135 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0274 | -0.1027 |
| controlled_alt_vs_controlled_default | naturalness | 0.0144 | 0.0165 |
| controlled_alt_vs_controlled_default | quest_state_correctness | -0.0130 | -0.0704 |
| controlled_alt_vs_controlled_default | lore_consistency | 0.0023 | 0.0573 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_controlled_default | objective_completion_support | -0.0045 | -0.0487 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | 0.0055 | 0.0181 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | 0.0106 | 0.0362 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0303 | -0.2417 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0040 | -0.0698 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0355 | -0.1667 |
| controlled_alt_vs_controlled_default | persona_style | 0.0052 | 0.0108 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0126 | -0.0134 |
| controlled_alt_vs_controlled_default | length_score | 0.0833 | 0.1695 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0271 | 0.0283 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0180 | -0.0571 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.0085 | 0.1147 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.0241 | 0.1123 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0052 | 0.0059 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 0.0074 | 0.0453 |
| controlled_alt_vs_proposed_raw | lore_consistency | 0.0187 | 0.7801 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_proposed_raw | objective_completion_support | 0.0316 | 0.5576 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 0.0177 | 0.0603 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | 0.0124 | 0.0427 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.0117 | 0.1409 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0009 | 0.0174 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.0357 | 0.2517 |
| controlled_alt_vs_proposed_raw | persona_style | -0.0221 | -0.0436 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0168 | -0.0178 |
| controlled_alt_vs_proposed_raw | length_score | 0.0306 | 0.0561 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0583 | 0.0629 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.0144 | 0.0512 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.0500 | 1.5339 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.0040 | 0.0172 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0140 | 0.0161 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 0.0459 | 0.3660 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 0.0365 | 5.8540 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_candidate_no_context | objective_completion_support | 0.0303 | 0.5240 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 0.0356 | 0.1289 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | 0.0143 | 0.0498 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.0644 | 2.0988 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0164 | 0.4421 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.0109 | 0.0655 |
| controlled_alt_vs_candidate_no_context | persona_style | -0.0234 | -0.0460 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0151 | -0.0160 |
| controlled_alt_vs_candidate_no_context | length_score | 0.0639 | 0.1250 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.0729 | 0.0799 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.0278 | 0.1036 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0415 | (0.0151, 0.0697) | 0.0003 | 0.0415 | (0.0052, 0.0797) | 0.0003 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0201 | (-0.0769, 0.0352) | 0.7567 | -0.0201 | (-0.0720, 0.0328) | 0.8090 |
| proposed_vs_candidate_no_context | naturalness | 0.0088 | (-0.0096, 0.0273) | 0.1843 | 0.0088 | (-0.0199, 0.0383) | 0.2990 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.0384 | (0.0128, 0.0652) | 0.0007 | 0.0384 | (0.0051, 0.0776) | 0.0003 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0178 | (0.0064, 0.0313) | 0.0003 | 0.0178 | (0.0000, 0.0308) | 0.0783 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | -0.0012 | (-0.0156, 0.0142) | 0.5843 | -0.0012 | (-0.0073, 0.0052) | 0.6360 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 0.0179 | (0.0028, 0.0326) | 0.0090 | 0.0179 | (-0.0012, 0.0415) | 0.0340 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | 0.0020 | (-0.0081, 0.0140) | 0.3737 | 0.0020 | (-0.0043, 0.0090) | 0.2770 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0527 | (0.0189, 0.0905) | 0.0000 | 0.0527 | (0.0045, 0.1003) | 0.0103 |
| proposed_vs_candidate_no_context | context_overlap | 0.0155 | (0.0027, 0.0293) | 0.0083 | 0.0155 | (0.0008, 0.0347) | 0.0000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0248 | (-0.0913, 0.0347) | 0.7957 | -0.0248 | (-0.0744, 0.0303) | 0.8030 |
| proposed_vs_candidate_no_context | persona_style | -0.0013 | (-0.0417, 0.0573) | 0.6107 | -0.0013 | (-0.0625, 0.0586) | 0.6423 |
| proposed_vs_candidate_no_context | distinct1 | 0.0017 | (-0.0161, 0.0192) | 0.4227 | 0.0017 | (-0.0173, 0.0161) | 0.4040 |
| proposed_vs_candidate_no_context | length_score | 0.0333 | (-0.0625, 0.1250) | 0.2460 | 0.0333 | (-0.1115, 0.1538) | 0.3160 |
| proposed_vs_candidate_no_context | sentence_score | 0.0146 | (-0.0437, 0.0729) | 0.4140 | 0.0146 | (0.0000, 0.0404) | 0.3277 |
| proposed_vs_candidate_no_context | overall_quality | 0.0134 | (-0.0111, 0.0378) | 0.1380 | 0.0134 | (-0.0163, 0.0434) | 0.2257 |
| controlled_vs_proposed_raw | context_relevance | 0.0309 | (-0.0166, 0.0842) | 0.1230 | 0.0309 | (0.0096, 0.0522) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.0515 | (-0.0197, 0.1205) | 0.0847 | 0.0515 | (-0.0460, 0.1162) | 0.1303 |
| controlled_vs_proposed_raw | naturalness | -0.0091 | (-0.0372, 0.0180) | 0.7410 | -0.0091 | (-0.0419, 0.0216) | 0.6867 |
| controlled_vs_proposed_raw | quest_state_correctness | 0.0204 | (-0.0226, 0.0658) | 0.1833 | 0.0204 | (0.0015, 0.0383) | 0.0180 |
| controlled_vs_proposed_raw | lore_consistency | 0.0164 | (-0.0112, 0.0493) | 0.1473 | 0.0164 | (-0.0000, 0.0430) | 0.0287 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0361 | (0.0040, 0.0711) | 0.0110 | 0.0361 | (0.0111, 0.0526) | 0.0017 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0122 | (-0.0110, 0.0383) | 0.1723 | 0.0122 | (-0.0102, 0.0352) | 0.1643 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 0.0018 | (-0.0179, 0.0210) | 0.4347 | 0.0018 | (-0.0078, 0.0139) | 0.4143 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.0420 | (-0.0223, 0.1069) | 0.1083 | 0.0420 | (0.0152, 0.0689) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0049 | (-0.0191, 0.0281) | 0.3490 | 0.0049 | (-0.0184, 0.0283) | 0.3557 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.0712 | (-0.0109, 0.1573) | 0.0523 | 0.0712 | (-0.0452, 0.1452) | 0.1020 |
| controlled_vs_proposed_raw | persona_style | -0.0273 | (-0.0638, 0.0013) | 0.9720 | -0.0273 | (-0.0753, 0.0000) | 1.0000 |
| controlled_vs_proposed_raw | distinct1 | -0.0042 | (-0.0271, 0.0187) | 0.6333 | -0.0042 | (-0.0320, 0.0321) | 0.5773 |
| controlled_vs_proposed_raw | length_score | -0.0528 | (-0.1653, 0.0667) | 0.8147 | -0.0528 | (-0.1778, 0.0722) | 0.7877 |
| controlled_vs_proposed_raw | sentence_score | 0.0312 | (-0.0375, 0.1021) | 0.1713 | 0.0312 | (0.0000, 0.0808) | 0.0830 |
| controlled_vs_proposed_raw | overall_quality | 0.0324 | (-0.0104, 0.0741) | 0.0683 | 0.0324 | (-0.0017, 0.0584) | 0.0300 |
| controlled_vs_candidate_no_context | context_relevance | 0.0724 | (0.0231, 0.1255) | 0.0007 | 0.0724 | (0.0389, 0.0973) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.0314 | (-0.0506, 0.1130) | 0.2310 | 0.0314 | (-0.0473, 0.0968) | 0.2080 |
| controlled_vs_candidate_no_context | naturalness | -0.0003 | (-0.0228, 0.0239) | 0.5193 | -0.0003 | (-0.0176, 0.0177) | 0.5280 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.0588 | (0.0168, 0.1072) | 0.0023 | 0.0588 | (0.0331, 0.0815) | 0.0000 |
| controlled_vs_candidate_no_context | lore_consistency | 0.0342 | (0.0095, 0.0648) | 0.0010 | 0.0342 | (0.0161, 0.0519) | 0.0000 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | 0.0349 | (0.0056, 0.0685) | 0.0113 | 0.0349 | (0.0129, 0.0496) | 0.0027 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0301 | (0.0068, 0.0600) | 0.0047 | 0.0301 | (0.0115, 0.0438) | 0.0013 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 0.0038 | (-0.0152, 0.0228) | 0.3550 | 0.0038 | (-0.0026, 0.0102) | 0.1410 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.0947 | (0.0341, 0.1630) | 0.0007 | 0.0947 | (0.0500, 0.1299) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0204 | (-0.0006, 0.0439) | 0.0277 | 0.0204 | (0.0063, 0.0312) | 0.0010 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.0464 | (-0.0421, 0.1357) | 0.1613 | 0.0464 | (-0.0433, 0.1198) | 0.1910 |
| controlled_vs_candidate_no_context | persona_style | -0.0286 | (-0.0925, 0.0443) | 0.8007 | -0.0286 | (-0.1344, 0.0391) | 0.7430 |
| controlled_vs_candidate_no_context | distinct1 | -0.0025 | (-0.0191, 0.0148) | 0.6307 | -0.0025 | (-0.0191, 0.0198) | 0.6030 |
| controlled_vs_candidate_no_context | length_score | -0.0194 | (-0.1250, 0.0972) | 0.6577 | -0.0194 | (-0.1000, 0.0502) | 0.6833 |
| controlled_vs_candidate_no_context | sentence_score | 0.0458 | (-0.0104, 0.1042) | 0.0527 | 0.0458 | (0.0000, 0.0867) | 0.0727 |
| controlled_vs_candidate_no_context | overall_quality | 0.0458 | (-0.0047, 0.0951) | 0.0367 | 0.0458 | (0.0028, 0.0813) | 0.0210 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0224 | (-0.0768, 0.0340) | 0.7833 | -0.0224 | (-0.0777, 0.0425) | 0.7630 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0274 | (-0.0875, 0.0243) | 0.8220 | -0.0274 | (-0.0574, 0.0024) | 0.9530 |
| controlled_alt_vs_controlled_default | naturalness | 0.0144 | (-0.0054, 0.0355) | 0.0883 | 0.0144 | (-0.0081, 0.0324) | 0.0823 |
| controlled_alt_vs_controlled_default | quest_state_correctness | -0.0130 | (-0.0614, 0.0361) | 0.7013 | -0.0130 | (-0.0628, 0.0456) | 0.7403 |
| controlled_alt_vs_controlled_default | lore_consistency | 0.0023 | (-0.0433, 0.0486) | 0.4700 | 0.0023 | (-0.0441, 0.0562) | 0.4837 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | objective_completion_support | -0.0045 | (-0.0384, 0.0265) | 0.5987 | -0.0045 | (-0.0366, 0.0280) | 0.5980 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | 0.0055 | (-0.0201, 0.0289) | 0.3157 | 0.0055 | (-0.0264, 0.0283) | 0.3673 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | 0.0106 | (-0.0096, 0.0325) | 0.1560 | 0.0106 | (-0.0123, 0.0319) | 0.1867 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0303 | (-0.1023, 0.0417) | 0.8087 | -0.0303 | (-0.1000, 0.0559) | 0.7640 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0040 | (-0.0275, 0.0196) | 0.6473 | -0.0040 | (-0.0267, 0.0137) | 0.6590 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0355 | (-0.1050, 0.0218) | 0.8800 | -0.0355 | (-0.0775, 0.0060) | 0.9570 |
| controlled_alt_vs_controlled_default | persona_style | 0.0052 | (-0.0312, 0.0417) | 0.4533 | 0.0052 | (-0.0156, 0.0312) | 0.4343 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0126 | (-0.0317, 0.0050) | 0.9133 | -0.0126 | (-0.0342, 0.0108) | 0.8377 |
| controlled_alt_vs_controlled_default | length_score | 0.0833 | (0.0111, 0.1598) | 0.0090 | 0.0833 | (-0.0061, 0.1590) | 0.0427 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0271 | (-0.0292, 0.0958) | 0.2567 | 0.0271 | (-0.0404, 0.1154) | 0.3513 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0180 | (-0.0571, 0.0186) | 0.8260 | -0.0180 | (-0.0453, 0.0207) | 0.8077 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.0085 | (-0.0386, 0.0593) | 0.4000 | 0.0085 | (-0.0490, 0.0875) | 0.3863 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.0241 | (-0.0174, 0.0670) | 0.1350 | 0.0241 | (-0.0621, 0.0857) | 0.2797 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0052 | (-0.0171, 0.0286) | 0.3163 | 0.0052 | (-0.0168, 0.0336) | 0.3940 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 0.0074 | (-0.0364, 0.0526) | 0.3967 | 0.0074 | (-0.0420, 0.0754) | 0.4007 |
| controlled_alt_vs_proposed_raw | lore_consistency | 0.0187 | (-0.0083, 0.0528) | 0.1100 | 0.0187 | (-0.0121, 0.0640) | 0.3363 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_proposed_raw | objective_completion_support | 0.0316 | (0.0147, 0.0509) | 0.0000 | 0.0316 | (0.0000, 0.0676) | 0.0773 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 0.0177 | (-0.0028, 0.0403) | 0.0467 | 0.0177 | (-0.0142, 0.0511) | 0.2180 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | 0.0124 | (0.0013, 0.0245) | 0.0140 | 0.0124 | (-0.0015, 0.0359) | 0.3413 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.0117 | (-0.0455, 0.0727) | 0.3517 | 0.0117 | (-0.0606, 0.1199) | 0.3730 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0009 | (-0.0260, 0.0268) | 0.4937 | 0.0009 | (-0.0246, 0.0302) | 0.5033 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.0357 | (-0.0129, 0.0853) | 0.0773 | 0.0357 | (-0.0643, 0.1080) | 0.2207 |
| controlled_alt_vs_proposed_raw | persona_style | -0.0221 | (-0.0586, 0.0156) | 0.8897 | -0.0221 | (-0.0531, 0.0000) | 1.0000 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0168 | (-0.0421, 0.0051) | 0.9263 | -0.0168 | (-0.0609, 0.0166) | 0.6763 |
| controlled_alt_vs_proposed_raw | length_score | 0.0306 | (-0.0528, 0.1209) | 0.2507 | 0.0306 | (-0.0550, 0.1462) | 0.3323 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0583 | (0.0146, 0.1167) | 0.0137 | 0.0583 | (0.0000, 0.1250) | 0.0737 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.0144 | (-0.0163, 0.0476) | 0.2073 | 0.0144 | (-0.0375, 0.0723) | 0.3477 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.0500 | (0.0033, 0.1042) | 0.0133 | 0.0500 | (-0.0229, 0.1255) | 0.1260 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.0040 | (-0.0574, 0.0626) | 0.4467 | 0.0040 | (-0.0707, 0.0579) | 0.4370 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0140 | (-0.0023, 0.0337) | 0.0550 | 0.0140 | (0.0029, 0.0230) | 0.0000 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 0.0459 | (0.0042, 0.0953) | 0.0123 | 0.0459 | (-0.0186, 0.1080) | 0.1023 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 0.0365 | (0.0067, 0.0709) | 0.0040 | 0.0365 | (0.0000, 0.0880) | 0.0760 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_candidate_no_context | objective_completion_support | 0.0303 | (0.0101, 0.0513) | 0.0007 | 0.0303 | (0.0019, 0.0665) | 0.0117 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 0.0356 | (0.0150, 0.0584) | 0.0003 | 0.0356 | (-0.0018, 0.0623) | 0.0583 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | 0.0143 | (-0.0001, 0.0283) | 0.0263 | 0.0143 | (-0.0037, 0.0370) | 0.1250 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.0644 | (0.0038, 0.1326) | 0.0217 | 0.0644 | (-0.0273, 0.1656) | 0.1657 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0164 | (-0.0053, 0.0411) | 0.0807 | 0.0164 | (-0.0126, 0.0375) | 0.1310 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.0109 | (-0.0516, 0.0764) | 0.3560 | 0.0109 | (-0.0738, 0.0774) | 0.3753 |
| controlled_alt_vs_candidate_no_context | persona_style | -0.0234 | (-0.0859, 0.0391) | 0.7807 | -0.0234 | (-0.1023, 0.0234) | 0.7543 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0151 | (-0.0335, 0.0024) | 0.9520 | -0.0151 | (-0.0478, 0.0070) | 0.7030 |
| controlled_alt_vs_candidate_no_context | length_score | 0.0639 | (-0.0250, 0.1639) | 0.0893 | 0.0639 | (0.0136, 0.1262) | 0.0003 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.0729 | (0.0000, 0.1458) | 0.0297 | 0.0729 | (0.0000, 0.1625) | 0.0760 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.0278 | (-0.0077, 0.0639) | 0.0653 | 0.0278 | (-0.0349, 0.0783) | 0.1837 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 12 | 2 | 10 | 0.7083 | 0.8571 |
| proposed_vs_candidate_no_context | persona_consistency | 2 | 4 | 18 | 0.4583 | 0.3333 |
| proposed_vs_candidate_no_context | naturalness | 8 | 5 | 11 | 0.5625 | 0.6154 |
| proposed_vs_candidate_no_context | quest_state_correctness | 13 | 1 | 10 | 0.7500 | 0.9286 |
| proposed_vs_candidate_no_context | lore_consistency | 8 | 1 | 15 | 0.6458 | 0.8889 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| proposed_vs_candidate_no_context | objective_completion_support | 3 | 4 | 17 | 0.4792 | 0.4286 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 12 | 2 | 10 | 0.7083 | 0.8571 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | 4 | 6 | 14 | 0.4583 | 0.4000 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 9 | 1 | 14 | 0.6667 | 0.9000 |
| proposed_vs_candidate_no_context | context_overlap | 11 | 3 | 10 | 0.6667 | 0.7857 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 2 | 3 | 19 | 0.4792 | 0.4000 |
| proposed_vs_candidate_no_context | persona_style | 1 | 2 | 21 | 0.4792 | 0.3333 |
| proposed_vs_candidate_no_context | distinct1 | 7 | 5 | 12 | 0.5417 | 0.5833 |
| proposed_vs_candidate_no_context | length_score | 7 | 5 | 12 | 0.5417 | 0.5833 |
| proposed_vs_candidate_no_context | sentence_score | 3 | 2 | 19 | 0.5208 | 0.6000 |
| proposed_vs_candidate_no_context | overall_quality | 8 | 6 | 10 | 0.5417 | 0.5714 |
| controlled_vs_proposed_raw | context_relevance | 11 | 10 | 3 | 0.5208 | 0.5238 |
| controlled_vs_proposed_raw | persona_consistency | 11 | 6 | 7 | 0.6042 | 0.6471 |
| controlled_vs_proposed_raw | naturalness | 8 | 13 | 3 | 0.3958 | 0.3810 |
| controlled_vs_proposed_raw | quest_state_correctness | 11 | 10 | 3 | 0.5208 | 0.5238 |
| controlled_vs_proposed_raw | lore_consistency | 6 | 7 | 11 | 0.4792 | 0.4615 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_vs_proposed_raw | objective_completion_support | 9 | 5 | 10 | 0.5833 | 0.6429 |
| controlled_vs_proposed_raw | gameplay_usefulness | 9 | 12 | 3 | 0.4375 | 0.4286 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 7 | 9 | 8 | 0.4583 | 0.4375 |
| controlled_vs_proposed_raw | context_keyword_coverage | 10 | 7 | 7 | 0.5625 | 0.5882 |
| controlled_vs_proposed_raw | context_overlap | 11 | 10 | 3 | 0.5208 | 0.5238 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 11 | 5 | 8 | 0.6250 | 0.6875 |
| controlled_vs_proposed_raw | persona_style | 1 | 5 | 18 | 0.4167 | 0.1667 |
| controlled_vs_proposed_raw | distinct1 | 12 | 8 | 4 | 0.5833 | 0.6000 |
| controlled_vs_proposed_raw | length_score | 9 | 11 | 4 | 0.4583 | 0.4500 |
| controlled_vs_proposed_raw | sentence_score | 4 | 2 | 18 | 0.5417 | 0.6667 |
| controlled_vs_proposed_raw | overall_quality | 12 | 9 | 3 | 0.5625 | 0.5714 |
| controlled_vs_candidate_no_context | context_relevance | 12 | 9 | 3 | 0.5625 | 0.5714 |
| controlled_vs_candidate_no_context | persona_consistency | 11 | 7 | 6 | 0.5833 | 0.6111 |
| controlled_vs_candidate_no_context | naturalness | 7 | 14 | 3 | 0.3542 | 0.3333 |
| controlled_vs_candidate_no_context | quest_state_correctness | 13 | 8 | 3 | 0.6042 | 0.6190 |
| controlled_vs_candidate_no_context | lore_consistency | 7 | 3 | 14 | 0.5833 | 0.7000 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_vs_candidate_no_context | objective_completion_support | 9 | 4 | 11 | 0.6042 | 0.6923 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 14 | 7 | 3 | 0.6458 | 0.6667 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 10 | 7 | 7 | 0.5625 | 0.5882 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 11 | 4 | 9 | 0.6458 | 0.7333 |
| controlled_vs_candidate_no_context | context_overlap | 11 | 10 | 3 | 0.5208 | 0.5238 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 11 | 6 | 7 | 0.6042 | 0.6471 |
| controlled_vs_candidate_no_context | persona_style | 1 | 6 | 17 | 0.3958 | 0.1429 |
| controlled_vs_candidate_no_context | distinct1 | 10 | 10 | 4 | 0.5000 | 0.5000 |
| controlled_vs_candidate_no_context | length_score | 8 | 13 | 3 | 0.3958 | 0.3810 |
| controlled_vs_candidate_no_context | sentence_score | 4 | 1 | 19 | 0.5625 | 0.8000 |
| controlled_vs_candidate_no_context | overall_quality | 14 | 7 | 3 | 0.6458 | 0.6667 |
| controlled_alt_vs_controlled_default | context_relevance | 5 | 9 | 10 | 0.4167 | 0.3571 |
| controlled_alt_vs_controlled_default | persona_consistency | 3 | 4 | 17 | 0.4792 | 0.4286 |
| controlled_alt_vs_controlled_default | naturalness | 9 | 5 | 10 | 0.5833 | 0.6429 |
| controlled_alt_vs_controlled_default | quest_state_correctness | 5 | 9 | 10 | 0.4167 | 0.3571 |
| controlled_alt_vs_controlled_default | lore_consistency | 3 | 6 | 15 | 0.4375 | 0.3333 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_alt_vs_controlled_default | objective_completion_support | 7 | 4 | 13 | 0.5625 | 0.6364 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | 10 | 4 | 10 | 0.6250 | 0.7143 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | 9 | 5 | 10 | 0.5833 | 0.6429 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 4 | 7 | 13 | 0.4375 | 0.3636 |
| controlled_alt_vs_controlled_default | context_overlap | 5 | 9 | 10 | 0.4167 | 0.3571 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 2 | 4 | 18 | 0.4583 | 0.3333 |
| controlled_alt_vs_controlled_default | persona_style | 2 | 2 | 20 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | distinct1 | 5 | 8 | 11 | 0.4375 | 0.3846 |
| controlled_alt_vs_controlled_default | length_score | 9 | 4 | 11 | 0.6042 | 0.6923 |
| controlled_alt_vs_controlled_default | sentence_score | 2 | 1 | 21 | 0.5208 | 0.6667 |
| controlled_alt_vs_controlled_default | overall_quality | 6 | 8 | 10 | 0.4583 | 0.4286 |
| controlled_alt_vs_proposed_raw | context_relevance | 9 | 10 | 5 | 0.4792 | 0.4737 |
| controlled_alt_vs_proposed_raw | persona_consistency | 9 | 6 | 9 | 0.5625 | 0.6000 |
| controlled_alt_vs_proposed_raw | naturalness | 7 | 12 | 5 | 0.3958 | 0.3684 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 9 | 10 | 5 | 0.4792 | 0.4737 |
| controlled_alt_vs_proposed_raw | lore_consistency | 6 | 6 | 12 | 0.5000 | 0.5000 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_alt_vs_proposed_raw | objective_completion_support | 9 | 0 | 15 | 0.6875 | 1.0000 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 9 | 10 | 5 | 0.4792 | 0.4737 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | 7 | 3 | 14 | 0.5833 | 0.7000 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 7 | 7 | 10 | 0.5000 | 0.5000 |
| controlled_alt_vs_proposed_raw | context_overlap | 9 | 10 | 5 | 0.4792 | 0.4737 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 8 | 4 | 12 | 0.5833 | 0.6667 |
| controlled_alt_vs_proposed_raw | persona_style | 1 | 5 | 18 | 0.4167 | 0.1667 |
| controlled_alt_vs_proposed_raw | distinct1 | 11 | 7 | 6 | 0.5833 | 0.6111 |
| controlled_alt_vs_proposed_raw | length_score | 7 | 11 | 6 | 0.4167 | 0.3889 |
| controlled_alt_vs_proposed_raw | sentence_score | 4 | 0 | 20 | 0.5833 | 1.0000 |
| controlled_alt_vs_proposed_raw | overall_quality | 9 | 10 | 5 | 0.4792 | 0.4737 |
| controlled_alt_vs_candidate_no_context | context_relevance | 9 | 11 | 4 | 0.4583 | 0.4500 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 8 | 7 | 9 | 0.5208 | 0.5333 |
| controlled_alt_vs_candidate_no_context | naturalness | 11 | 9 | 4 | 0.5417 | 0.5500 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 10 | 10 | 4 | 0.5000 | 0.5000 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 7 | 3 | 14 | 0.5833 | 0.7000 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_alt_vs_candidate_no_context | objective_completion_support | 10 | 1 | 13 | 0.6875 | 0.9091 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 14 | 6 | 4 | 0.6667 | 0.7000 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | 10 | 5 | 9 | 0.6042 | 0.6667 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 6 | 4 | 14 | 0.5417 | 0.6000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 9 | 11 | 4 | 0.4583 | 0.4500 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 8 | 6 | 10 | 0.5417 | 0.5714 |
| controlled_alt_vs_candidate_no_context | persona_style | 1 | 5 | 18 | 0.4167 | 0.1667 |
| controlled_alt_vs_candidate_no_context | distinct1 | 9 | 10 | 5 | 0.4792 | 0.4737 |
| controlled_alt_vs_candidate_no_context | length_score | 10 | 10 | 4 | 0.5000 | 0.5000 |
| controlled_alt_vs_candidate_no_context | sentence_score | 6 | 1 | 17 | 0.6042 | 0.8571 |
| controlled_alt_vs_candidate_no_context | overall_quality | 12 | 8 | 4 | 0.5833 | 0.6000 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.1667 | 0.1667 | 0.8333 |
| proposed_contextual_controlled_tuned | 0.0000 | 0.0000 | 0.2083 | 0.2500 | 0.7500 |
| proposed_contextual | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.4167 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `5`
- Unique template signatures: `19`
- Template signature ratio: `0.7917`
- Effective sample size by source clustering: `4.80`
- Effective sample size by template-signature clustering: `16.94`
- Detailed diagnostics are published in `scenario_dependence.json`.

## Multi-turn Contradiction
| Arm | Contradiction Rate | Contradiction Safety | Contradicted Sources | Source Count |
|---|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 1.0000 | 0 | 5 |
| proposed_contextual_controlled_tuned | 0.0000 | 1.0000 | 0 | 5 |
| proposed_contextual | 0.0000 | 1.0000 | 0 | 5 |
| candidate_no_context | 0.0000 | 1.0000 | 0 | 5 |
- Detailed source-level values are published in `multi_turn_contradictions.json`.

- BERTScore status: unavailable (disabled_by_flag).

This report covers proposal RO5 metrics plus game-facing outcomes: quest-state correctness, lore consistency, contradiction safety, objective completion support, gameplay usefulness, and time-pressure acceptability.
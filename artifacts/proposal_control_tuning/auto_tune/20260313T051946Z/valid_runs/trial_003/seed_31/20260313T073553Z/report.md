# Proposal Alignment Evaluation Report

- Run ID: `20260313T073553Z`
- Generated: `2026-03-13T07:42:40.465466+00:00`
- Scenarios: `artifacts\proposal_control_tuning\auto_tune\20260313T051946Z\valid_runs\trial_003\seed_31\20260313T073553Z\scenarios.jsonl`
- Scenario count: `24`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_tuned`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.1330 (0.0843, 0.1868) | 0.3129 (0.2635, 0.3589) | 0.8697 (0.8596, 0.8794) | 0.3427 (0.3118, 0.3720) | n/a |
| proposed_contextual_controlled_tuned | 0.1061 (0.0589, 0.1628) | 0.3783 (0.3155, 0.4486) | 0.8737 (0.8626, 0.8836) | 0.3551 (0.3166, 0.4011) | n/a |
| proposed_contextual | 0.0709 (0.0437, 0.1006) | 0.2628 (0.2134, 0.3155) | 0.8762 (0.8618, 0.8898) | 0.2962 (0.2706, 0.3211) | n/a |
| candidate_no_context | 0.0443 (0.0258, 0.0663) | 0.2505 (0.2064, 0.3042) | 0.8706 (0.8600, 0.8815) | 0.2782 (0.2573, 0.3002) | n/a |

## Game-facing Outcome Metrics (mean, 95% CI)
| Arm | Quest-state Correctness | Lore Consistency | Contradiction Safety | Objective Completion Support | Gameplay Usefulness | Time-pressure Acceptability |
|---|---:|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2203 (0.1771, 0.2673) | 0.0642 (0.0270, 0.1026) | 1.0000 (1.0000, 1.0000) | 0.1124 (0.0686, 0.1574) | 0.3226 (0.3015, 0.3427) | 0.3221 (0.2934, 0.3521) |
| proposed_contextual_controlled_tuned | 0.1993 (0.1592, 0.2489) | 0.0374 (0.0040, 0.0824) | 1.0000 (1.0000, 1.0000) | 0.1166 (0.0700, 0.1635) | 0.3192 (0.2969, 0.3413) | 0.3246 (0.2939, 0.3569) |
| proposed_contextual | 0.1630 (0.1386, 0.1885) | 0.0382 (0.0122, 0.0672) | 1.0000 (1.0000, 1.0000) | 0.0966 (0.0489, 0.1480) | 0.3022 (0.2766, 0.3274) | 0.3165 (0.2880, 0.3464) |
| candidate_no_context | 0.1389 (0.1216, 0.1584) | 0.0148 (0.0048, 0.0261) | 1.0000 (1.0000, 1.0000) | 0.0873 (0.0447, 0.1349) | 0.2918 (0.2673, 0.3173) | 0.3179 (0.2908, 0.3479) |

- Multi-turn contradiction rate is reported as `1 - contradiction_safety` in row-level outputs.

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0266 | 0.6018 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0123 | 0.0491 |
| proposed_vs_candidate_no_context | naturalness | 0.0056 | 0.0064 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.0241 | 0.1735 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0233 | 1.5707 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | 0.0092 | 0.1058 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 0.0104 | 0.0355 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | -0.0015 | -0.0046 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0354 | 0.8116 |
| proposed_vs_candidate_no_context | context_overlap | 0.0063 | 0.1378 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0135 | 0.0852 |
| proposed_vs_candidate_no_context | persona_style | 0.0076 | 0.0122 |
| proposed_vs_candidate_no_context | distinct1 | 0.0007 | 0.0007 |
| proposed_vs_candidate_no_context | length_score | -0.0028 | -0.0055 |
| proposed_vs_candidate_no_context | sentence_score | 0.0583 | 0.0639 |
| proposed_vs_candidate_no_context | overall_quality | 0.0180 | 0.0648 |
| controlled_vs_proposed_raw | context_relevance | 0.0620 | 0.8746 |
| controlled_vs_proposed_raw | persona_consistency | 0.0501 | 0.1907 |
| controlled_vs_proposed_raw | naturalness | -0.0065 | -0.0074 |
| controlled_vs_proposed_raw | quest_state_correctness | 0.0573 | 0.3515 |
| controlled_vs_proposed_raw | lore_consistency | 0.0260 | 0.6826 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0158 | 0.1640 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0204 | 0.0676 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 0.0056 | 0.0177 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.0840 | 1.0640 |
| controlled_vs_proposed_raw | context_overlap | 0.0109 | 0.2077 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.0609 | 0.3545 |
| controlled_vs_proposed_raw | persona_style | 0.0069 | 0.0111 |
| controlled_vs_proposed_raw | distinct1 | -0.0066 | -0.0069 |
| controlled_vs_proposed_raw | length_score | -0.0194 | -0.0390 |
| controlled_vs_proposed_raw | sentence_score | 0.0000 | 0.0000 |
| controlled_vs_proposed_raw | overall_quality | 0.0464 | 0.1568 |
| controlled_vs_candidate_no_context | context_relevance | 0.0887 | 2.0029 |
| controlled_vs_candidate_no_context | persona_consistency | 0.0624 | 0.2493 |
| controlled_vs_candidate_no_context | naturalness | -0.0010 | -0.0011 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.0814 | 0.5859 |
| controlled_vs_candidate_no_context | lore_consistency | 0.0494 | 3.3254 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | 0.0251 | 0.2871 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0308 | 0.1055 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 0.0041 | 0.0130 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.1193 | 2.7391 |
| controlled_vs_candidate_no_context | context_overlap | 0.0172 | 0.3741 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.0744 | 0.4699 |
| controlled_vs_candidate_no_context | persona_style | 0.0145 | 0.0235 |
| controlled_vs_candidate_no_context | distinct1 | -0.0059 | -0.0062 |
| controlled_vs_candidate_no_context | length_score | -0.0222 | -0.0443 |
| controlled_vs_candidate_no_context | sentence_score | 0.0583 | 0.0639 |
| controlled_vs_candidate_no_context | overall_quality | 0.0645 | 0.2317 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0269 | -0.2020 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0654 | 0.2092 |
| controlled_alt_vs_controlled_default | naturalness | 0.0040 | 0.0046 |
| controlled_alt_vs_controlled_default | quest_state_correctness | -0.0210 | -0.0953 |
| controlled_alt_vs_controlled_default | lore_consistency | -0.0268 | -0.4172 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_controlled_default | objective_completion_support | 0.0042 | 0.0377 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | -0.0035 | -0.0108 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | 0.0026 | 0.0080 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0376 | -0.2306 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0019 | -0.0296 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0778 | 0.3342 |
| controlled_alt_vs_controlled_default | persona_style | 0.0161 | 0.0254 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0134 | 0.0142 |
| controlled_alt_vs_controlled_default | length_score | -0.0139 | -0.0290 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0146 | 0.0150 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0124 | 0.0363 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.0352 | 0.4960 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1156 | 0.4398 |
| controlled_alt_vs_proposed_raw | naturalness | -0.0025 | -0.0028 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 0.0363 | 0.2227 |
| controlled_alt_vs_proposed_raw | lore_consistency | -0.0007 | -0.0193 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_proposed_raw | objective_completion_support | 0.0201 | 0.2078 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 0.0169 | 0.0561 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | 0.0082 | 0.0258 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.0464 | 0.5880 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0090 | 0.1719 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1387 | 0.8072 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0231 | 0.0368 |
| controlled_alt_vs_proposed_raw | distinct1 | 0.0068 | 0.0072 |
| controlled_alt_vs_proposed_raw | length_score | -0.0333 | -0.0669 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0146 | 0.0150 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.0589 | 0.1987 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.0618 | 1.3964 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1279 | 0.5106 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0031 | 0.0035 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 0.0604 | 0.4348 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 0.0226 | 1.5209 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_candidate_no_context | objective_completion_support | 0.0293 | 0.3356 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 0.0273 | 0.0936 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | 0.0067 | 0.0211 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.0818 | 1.8768 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0153 | 0.3335 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1522 | 0.9612 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0306 | 0.0495 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0075 | 0.0079 |
| controlled_alt_vs_candidate_no_context | length_score | -0.0361 | -0.0720 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.0729 | 0.0799 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.0769 | 0.2764 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0266 | (0.0047, 0.0536) | 0.0063 | 0.0266 | (0.0002, 0.0739) | 0.0000 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0123 | (-0.0152, 0.0464) | 0.2120 | 0.0123 | (-0.0143, 0.0766) | 0.3123 |
| proposed_vs_candidate_no_context | naturalness | 0.0056 | (-0.0066, 0.0179) | 0.1783 | 0.0056 | (-0.0056, 0.0293) | 0.2937 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.0241 | (0.0052, 0.0475) | 0.0040 | 0.0241 | (0.0001, 0.0685) | 0.0000 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0233 | (0.0013, 0.0500) | 0.0167 | 0.0233 | (0.0056, 0.0639) | 0.0000 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | 0.0092 | (-0.0145, 0.0378) | 0.2373 | 0.0092 | (-0.0145, 0.0404) | 0.2970 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 0.0104 | (-0.0011, 0.0238) | 0.0470 | 0.0104 | (-0.0033, 0.0197) | 0.0363 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | -0.0015 | (-0.0131, 0.0106) | 0.6153 | -0.0015 | (-0.0238, 0.0165) | 0.6427 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0354 | (0.0076, 0.0672) | 0.0037 | 0.0354 | (0.0000, 0.0952) | 0.0317 |
| proposed_vs_candidate_no_context | context_overlap | 0.0063 | (-0.0061, 0.0206) | 0.1650 | 0.0063 | (-0.0029, 0.0242) | 0.2547 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0135 | (-0.0238, 0.0542) | 0.2623 | 0.0135 | (-0.0222, 0.0952) | 0.2883 |
| proposed_vs_candidate_no_context | persona_style | 0.0076 | (-0.0080, 0.0278) | 0.2073 | 0.0076 | (0.0000, 0.0185) | 0.0377 |
| proposed_vs_candidate_no_context | distinct1 | 0.0007 | (-0.0088, 0.0114) | 0.4613 | 0.0007 | (-0.0118, 0.0106) | 0.3633 |
| proposed_vs_candidate_no_context | length_score | -0.0028 | (-0.0625, 0.0514) | 0.5323 | -0.0028 | (-0.0333, 0.0667) | 0.7287 |
| proposed_vs_candidate_no_context | sentence_score | 0.0583 | (0.0000, 0.1313) | 0.0593 | 0.0583 | (-0.0389, 0.1500) | 0.1420 |
| proposed_vs_candidate_no_context | overall_quality | 0.0180 | (-0.0004, 0.0372) | 0.0270 | 0.0180 | (-0.0066, 0.0688) | 0.2490 |
| controlled_vs_proposed_raw | context_relevance | 0.0620 | (0.0149, 0.1162) | 0.0027 | 0.0620 | (0.0258, 0.0853) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.0501 | (0.0209, 0.0854) | 0.0003 | 0.0501 | (0.0215, 0.0952) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | -0.0065 | (-0.0191, 0.0072) | 0.8370 | -0.0065 | (-0.0320, 0.0079) | 0.7480 |
| controlled_vs_proposed_raw | quest_state_correctness | 0.0573 | (0.0145, 0.1049) | 0.0027 | 0.0573 | (0.0365, 0.0699) | 0.0000 |
| controlled_vs_proposed_raw | lore_consistency | 0.0260 | (-0.0123, 0.0660) | 0.0993 | 0.0260 | (0.0027, 0.0458) | 0.0000 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0158 | (-0.0180, 0.0491) | 0.1793 | 0.0158 | (-0.0023, 0.0333) | 0.0337 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0204 | (0.0016, 0.0415) | 0.0140 | 0.0204 | (0.0037, 0.0314) | 0.0000 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 0.0056 | (-0.0132, 0.0255) | 0.2870 | 0.0056 | (0.0040, 0.0073) | 0.0000 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.0840 | (0.0218, 0.1550) | 0.0013 | 0.0840 | (0.0341, 0.1190) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0109 | (-0.0049, 0.0269) | 0.0850 | 0.0109 | (0.0063, 0.0183) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.0609 | (0.0222, 0.1036) | 0.0003 | 0.0609 | (0.0222, 0.1190) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.0069 | (-0.0208, 0.0347) | 0.3440 | 0.0069 | (0.0000, 0.0185) | 0.2910 |
| controlled_vs_proposed_raw | distinct1 | -0.0066 | (-0.0231, 0.0090) | 0.7760 | -0.0066 | (-0.0205, 0.0047) | 0.8587 |
| controlled_vs_proposed_raw | length_score | -0.0194 | (-0.0806, 0.0403) | 0.7420 | -0.0194 | (-0.1190, 0.0704) | 0.6293 |
| controlled_vs_proposed_raw | sentence_score | 0.0000 | (-0.0583, 0.0583) | 0.6080 | 0.0000 | (-0.0389, 0.0437) | 0.6287 |
| controlled_vs_proposed_raw | overall_quality | 0.0464 | (0.0226, 0.0729) | 0.0000 | 0.0464 | (0.0280, 0.0695) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.0887 | (0.0354, 0.1450) | 0.0000 | 0.0887 | (0.0408, 0.1592) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.0624 | (0.0252, 0.1026) | 0.0000 | 0.0624 | (0.0074, 0.1719) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | -0.0010 | (-0.0146, 0.0120) | 0.5437 | -0.0010 | (-0.0031, 0.0023) | 0.7397 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.0814 | (0.0371, 0.1309) | 0.0000 | 0.0814 | (0.0487, 0.1384) | 0.0000 |
| controlled_vs_candidate_no_context | lore_consistency | 0.0494 | (0.0151, 0.0865) | 0.0013 | 0.0494 | (0.0104, 0.1097) | 0.0000 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | 0.0251 | (-0.0142, 0.0666) | 0.1140 | 0.0251 | (0.0183, 0.0381) | 0.0000 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0308 | (0.0098, 0.0541) | 0.0017 | 0.0308 | (0.0212, 0.0451) | 0.0000 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 0.0041 | (-0.0180, 0.0280) | 0.3737 | 0.0041 | (-0.0165, 0.0205) | 0.3883 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.1193 | (0.0530, 0.1963) | 0.0000 | 0.1193 | (0.0568, 0.2143) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0172 | (0.0019, 0.0342) | 0.0107 | 0.0172 | (0.0035, 0.0306) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.0744 | (0.0268, 0.1240) | 0.0003 | 0.0744 | (0.0000, 0.2143) | 0.0383 |
| controlled_vs_candidate_no_context | persona_style | 0.0145 | (-0.0008, 0.0362) | 0.0640 | 0.0145 | (0.0000, 0.0370) | 0.0343 |
| controlled_vs_candidate_no_context | distinct1 | -0.0059 | (-0.0214, 0.0086) | 0.7723 | -0.0059 | (-0.0182, 0.0048) | 0.8507 |
| controlled_vs_candidate_no_context | length_score | -0.0222 | (-0.0917, 0.0431) | 0.7187 | -0.0222 | (-0.0667, 0.0407) | 0.7327 |
| controlled_vs_candidate_no_context | sentence_score | 0.0583 | (-0.0146, 0.1313) | 0.0980 | 0.0583 | (-0.0778, 0.1500) | 0.2553 |
| controlled_vs_candidate_no_context | overall_quality | 0.0645 | (0.0368, 0.0941) | 0.0000 | 0.0645 | (0.0293, 0.1383) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0269 | (-0.1008, 0.0526) | 0.7610 | -0.0269 | (-0.1195, 0.0313) | 0.7400 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0654 | (0.0119, 0.1380) | 0.0010 | 0.0654 | (0.0000, 0.1141) | 0.0373 |
| controlled_alt_vs_controlled_default | naturalness | 0.0040 | (-0.0134, 0.0198) | 0.3043 | 0.0040 | (-0.0221, 0.0237) | 0.3603 |
| controlled_alt_vs_controlled_default | quest_state_correctness | -0.0210 | (-0.0832, 0.0469) | 0.7580 | -0.0210 | (-0.0980, 0.0298) | 0.7440 |
| controlled_alt_vs_controlled_default | lore_consistency | -0.0268 | (-0.0756, 0.0308) | 0.8383 | -0.0268 | (-0.1031, 0.0345) | 0.8463 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | objective_completion_support | 0.0042 | (-0.0258, 0.0360) | 0.4090 | 0.0042 | (-0.0183, 0.0364) | 0.3693 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | -0.0035 | (-0.0231, 0.0164) | 0.6287 | -0.0035 | (-0.0258, 0.0103) | 0.6993 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | 0.0026 | (-0.0200, 0.0255) | 0.4183 | 0.0026 | (-0.0111, 0.0261) | 0.4153 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0376 | (-0.1269, 0.0634) | 0.7840 | -0.0376 | (-0.1548, 0.0341) | 0.7487 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0019 | (-0.0260, 0.0250) | 0.5793 | -0.0019 | (-0.0371, 0.0247) | 0.5863 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0778 | (0.0208, 0.1542) | 0.0010 | 0.0778 | (0.0000, 0.1333) | 0.0423 |
| controlled_alt_vs_controlled_default | persona_style | 0.0161 | (-0.0250, 0.0649) | 0.2573 | 0.0161 | (0.0000, 0.0370) | 0.0307 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0134 | (0.0030, 0.0253) | 0.0033 | 0.0134 | (0.0070, 0.0233) | 0.0000 |
| controlled_alt_vs_controlled_default | length_score | -0.0139 | (-0.0931, 0.0569) | 0.6350 | -0.0139 | (-0.1571, 0.0852) | 0.6520 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0146 | (-0.0292, 0.0729) | 0.3760 | 0.0146 | (0.0000, 0.0389) | 0.3007 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0124 | (-0.0320, 0.0691) | 0.3400 | 0.0124 | (-0.0308, 0.0438) | 0.2627 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.0352 | (-0.0188, 0.0993) | 0.1133 | 0.0352 | (-0.0342, 0.0697) | 0.1450 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1156 | (0.0504, 0.1936) | 0.0000 | 0.1156 | (0.0429, 0.1730) | 0.0000 |
| controlled_alt_vs_proposed_raw | naturalness | -0.0025 | (-0.0241, 0.0180) | 0.5833 | -0.0025 | (-0.0541, 0.0316) | 0.5863 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 0.0363 | (-0.0121, 0.0904) | 0.0890 | 0.0363 | (-0.0281, 0.0662) | 0.0387 |
| controlled_alt_vs_proposed_raw | lore_consistency | -0.0007 | (-0.0479, 0.0528) | 0.5240 | -0.0007 | (-0.0573, 0.0372) | 0.6003 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_proposed_raw | objective_completion_support | 0.0201 | (-0.0144, 0.0545) | 0.1470 | 0.0201 | (-0.0008, 0.0698) | 0.2950 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 0.0169 | (-0.0017, 0.0378) | 0.0427 | 0.0169 | (-0.0004, 0.0331) | 0.0300 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | 0.0082 | (-0.0157, 0.0321) | 0.2443 | 0.0082 | (-0.0055, 0.0334) | 0.2623 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.0464 | (-0.0212, 0.1304) | 0.1020 | 0.0464 | (-0.0357, 0.0909) | 0.0357 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0090 | (-0.0156, 0.0352) | 0.2677 | 0.0090 | (-0.0307, 0.0310) | 0.2520 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1387 | (0.0605, 0.2306) | 0.0000 | 0.1387 | (0.0536, 0.2143) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0231 | (0.0000, 0.0556) | 0.0410 | 0.0231 | (0.0000, 0.0556) | 0.0357 |
| controlled_alt_vs_proposed_raw | distinct1 | 0.0068 | (-0.0099, 0.0225) | 0.2193 | 0.0068 | (0.0013, 0.0166) | 0.0000 |
| controlled_alt_vs_proposed_raw | length_score | -0.0333 | (-0.1348, 0.0639) | 0.7490 | -0.0333 | (-0.2762, 0.1556) | 0.6373 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0146 | (0.0000, 0.0437) | 0.3533 | 0.0146 | (0.0000, 0.0437) | 0.3043 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.0589 | (0.0186, 0.1119) | 0.0000 | 0.0589 | (0.0388, 0.0887) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.0618 | (0.0124, 0.1256) | 0.0077 | 0.0618 | (0.0397, 0.0721) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1279 | (0.0581, 0.2044) | 0.0000 | 0.1279 | (0.0286, 0.2496) | 0.0000 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0031 | (-0.0134, 0.0187) | 0.3317 | 0.0031 | (-0.0249, 0.0260) | 0.3803 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 0.0604 | (0.0147, 0.1146) | 0.0013 | 0.0604 | (0.0404, 0.0785) | 0.0000 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 0.0226 | (-0.0132, 0.0702) | 0.1373 | 0.0226 | (0.0067, 0.0449) | 0.0000 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_candidate_no_context | objective_completion_support | 0.0293 | (-0.0081, 0.0671) | 0.0647 | 0.0293 | (0.0000, 0.0552) | 0.0363 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 0.0273 | (0.0082, 0.0471) | 0.0023 | 0.0273 | (0.0193, 0.0315) | 0.0000 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | 0.0067 | (-0.0153, 0.0297) | 0.2767 | 0.0067 | (-0.0055, 0.0178) | 0.1487 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.0818 | (0.0164, 0.1578) | 0.0033 | 0.0818 | (0.0595, 0.0909) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0153 | (-0.0066, 0.0406) | 0.1027 | 0.0153 | (-0.0065, 0.0281) | 0.0337 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1522 | (0.0724, 0.2405) | 0.0000 | 0.1522 | (0.0357, 0.3095) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0306 | (-0.0018, 0.0741) | 0.0417 | 0.0306 | (0.0000, 0.0741) | 0.0410 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0075 | (-0.0099, 0.0241) | 0.2007 | 0.0075 | (0.0047, 0.0118) | 0.0000 |
| controlled_alt_vs_candidate_no_context | length_score | -0.0361 | (-0.1250, 0.0528) | 0.7903 | -0.0361 | (-0.2095, 0.1259) | 0.6170 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.0729 | (0.0000, 0.1458) | 0.0307 | 0.0729 | (-0.0389, 0.1500) | 0.0353 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.0769 | (0.0369, 0.1246) | 0.0000 | 0.0769 | (0.0443, 0.1075) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 8 | 3 | 13 | 0.6042 | 0.7273 |
| proposed_vs_candidate_no_context | persona_consistency | 3 | 3 | 18 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | naturalness | 7 | 4 | 13 | 0.5625 | 0.6364 |
| proposed_vs_candidate_no_context | quest_state_correctness | 9 | 2 | 13 | 0.6458 | 0.8182 |
| proposed_vs_candidate_no_context | lore_consistency | 6 | 5 | 13 | 0.5208 | 0.5455 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| proposed_vs_candidate_no_context | objective_completion_support | 3 | 4 | 17 | 0.4792 | 0.4286 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 7 | 4 | 13 | 0.5625 | 0.6364 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | 4 | 5 | 15 | 0.4792 | 0.4444 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 7 | 1 | 16 | 0.6250 | 0.8750 |
| proposed_vs_candidate_no_context | context_overlap | 7 | 4 | 13 | 0.5625 | 0.6364 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 3 | 2 | 19 | 0.5208 | 0.6000 |
| proposed_vs_candidate_no_context | persona_style | 2 | 1 | 21 | 0.5208 | 0.6667 |
| proposed_vs_candidate_no_context | distinct1 | 3 | 5 | 16 | 0.4583 | 0.3750 |
| proposed_vs_candidate_no_context | length_score | 6 | 5 | 13 | 0.5208 | 0.5455 |
| proposed_vs_candidate_no_context | sentence_score | 5 | 1 | 18 | 0.5833 | 0.8333 |
| proposed_vs_candidate_no_context | overall_quality | 7 | 4 | 13 | 0.5625 | 0.6364 |
| controlled_vs_proposed_raw | context_relevance | 14 | 9 | 1 | 0.6042 | 0.6087 |
| controlled_vs_proposed_raw | persona_consistency | 9 | 1 | 14 | 0.6667 | 0.9000 |
| controlled_vs_proposed_raw | naturalness | 8 | 14 | 2 | 0.3750 | 0.3636 |
| controlled_vs_proposed_raw | quest_state_correctness | 14 | 9 | 1 | 0.6042 | 0.6087 |
| controlled_vs_proposed_raw | lore_consistency | 6 | 5 | 13 | 0.5208 | 0.5455 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_vs_proposed_raw | objective_completion_support | 9 | 7 | 8 | 0.5417 | 0.5625 |
| controlled_vs_proposed_raw | gameplay_usefulness | 15 | 8 | 1 | 0.6458 | 0.6522 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 7 | 10 | 7 | 0.4375 | 0.4118 |
| controlled_vs_proposed_raw | context_keyword_coverage | 8 | 2 | 14 | 0.6250 | 0.8000 |
| controlled_vs_proposed_raw | context_overlap | 13 | 10 | 1 | 0.5625 | 0.5652 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 7 | 0 | 17 | 0.6458 | 1.0000 |
| controlled_vs_proposed_raw | persona_style | 3 | 2 | 19 | 0.5208 | 0.6000 |
| controlled_vs_proposed_raw | distinct1 | 13 | 8 | 3 | 0.6042 | 0.6190 |
| controlled_vs_proposed_raw | length_score | 8 | 12 | 4 | 0.4167 | 0.4000 |
| controlled_vs_proposed_raw | sentence_score | 2 | 2 | 20 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | overall_quality | 17 | 6 | 1 | 0.7292 | 0.7391 |
| controlled_vs_candidate_no_context | context_relevance | 15 | 7 | 2 | 0.6667 | 0.6818 |
| controlled_vs_candidate_no_context | persona_consistency | 11 | 1 | 12 | 0.7083 | 0.9167 |
| controlled_vs_candidate_no_context | naturalness | 10 | 12 | 2 | 0.4583 | 0.4545 |
| controlled_vs_candidate_no_context | quest_state_correctness | 15 | 7 | 2 | 0.6667 | 0.6818 |
| controlled_vs_candidate_no_context | lore_consistency | 8 | 4 | 12 | 0.5833 | 0.6667 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_vs_candidate_no_context | objective_completion_support | 10 | 7 | 7 | 0.5625 | 0.5882 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 16 | 6 | 2 | 0.7083 | 0.7273 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 8 | 9 | 7 | 0.4792 | 0.4706 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 11 | 1 | 12 | 0.7083 | 0.9167 |
| controlled_vs_candidate_no_context | context_overlap | 13 | 9 | 2 | 0.5833 | 0.5909 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 9 | 1 | 14 | 0.6667 | 0.9000 |
| controlled_vs_candidate_no_context | persona_style | 3 | 1 | 20 | 0.5417 | 0.7500 |
| controlled_vs_candidate_no_context | distinct1 | 13 | 8 | 3 | 0.6042 | 0.6190 |
| controlled_vs_candidate_no_context | length_score | 11 | 11 | 2 | 0.5000 | 0.5000 |
| controlled_vs_candidate_no_context | sentence_score | 6 | 2 | 16 | 0.5833 | 0.7500 |
| controlled_vs_candidate_no_context | overall_quality | 17 | 5 | 2 | 0.7500 | 0.7727 |
| controlled_alt_vs_controlled_default | context_relevance | 4 | 10 | 10 | 0.3750 | 0.2857 |
| controlled_alt_vs_controlled_default | persona_consistency | 6 | 3 | 15 | 0.5625 | 0.6667 |
| controlled_alt_vs_controlled_default | naturalness | 8 | 6 | 10 | 0.5417 | 0.5714 |
| controlled_alt_vs_controlled_default | quest_state_correctness | 4 | 10 | 10 | 0.3750 | 0.2857 |
| controlled_alt_vs_controlled_default | lore_consistency | 2 | 8 | 14 | 0.3750 | 0.2000 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_alt_vs_controlled_default | objective_completion_support | 5 | 6 | 13 | 0.4792 | 0.4545 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | 8 | 6 | 10 | 0.5417 | 0.5714 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | 5 | 7 | 12 | 0.4583 | 0.4167 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 3 | 8 | 13 | 0.3958 | 0.2727 |
| controlled_alt_vs_controlled_default | context_overlap | 4 | 10 | 10 | 0.3750 | 0.2857 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 6 | 0 | 18 | 0.6250 | 1.0000 |
| controlled_alt_vs_controlled_default | persona_style | 3 | 3 | 18 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | distinct1 | 10 | 3 | 11 | 0.6458 | 0.7692 |
| controlled_alt_vs_controlled_default | length_score | 6 | 8 | 10 | 0.4583 | 0.4286 |
| controlled_alt_vs_controlled_default | sentence_score | 2 | 1 | 21 | 0.5208 | 0.6667 |
| controlled_alt_vs_controlled_default | overall_quality | 5 | 9 | 10 | 0.4167 | 0.3571 |
| controlled_alt_vs_proposed_raw | context_relevance | 10 | 12 | 2 | 0.4583 | 0.4545 |
| controlled_alt_vs_proposed_raw | persona_consistency | 10 | 0 | 14 | 0.7083 | 1.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 11 | 11 | 2 | 0.5000 | 0.5000 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 10 | 12 | 2 | 0.4583 | 0.4545 |
| controlled_alt_vs_proposed_raw | lore_consistency | 4 | 6 | 14 | 0.4583 | 0.4000 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_alt_vs_proposed_raw | objective_completion_support | 6 | 6 | 12 | 0.5000 | 0.5000 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 16 | 6 | 2 | 0.7083 | 0.7273 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | 6 | 8 | 10 | 0.4583 | 0.4286 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 5 | 5 | 14 | 0.5000 | 0.5000 |
| controlled_alt_vs_proposed_raw | context_overlap | 10 | 12 | 2 | 0.4583 | 0.4545 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 10 | 0 | 14 | 0.7083 | 1.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 3 | 0 | 21 | 0.5625 | 1.0000 |
| controlled_alt_vs_proposed_raw | distinct1 | 17 | 5 | 2 | 0.7500 | 0.7727 |
| controlled_alt_vs_proposed_raw | length_score | 10 | 11 | 3 | 0.4792 | 0.4762 |
| controlled_alt_vs_proposed_raw | sentence_score | 1 | 0 | 23 | 0.5208 | 1.0000 |
| controlled_alt_vs_proposed_raw | overall_quality | 17 | 5 | 2 | 0.7500 | 0.7727 |
| controlled_alt_vs_candidate_no_context | context_relevance | 12 | 11 | 1 | 0.5208 | 0.5217 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 11 | 1 | 12 | 0.7083 | 0.9167 |
| controlled_alt_vs_candidate_no_context | naturalness | 14 | 9 | 1 | 0.6042 | 0.6087 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 13 | 10 | 1 | 0.5625 | 0.5652 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 4 | 6 | 14 | 0.4583 | 0.4000 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_alt_vs_candidate_no_context | objective_completion_support | 8 | 5 | 11 | 0.5625 | 0.6154 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 18 | 5 | 1 | 0.7708 | 0.7826 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | 7 | 8 | 9 | 0.4792 | 0.4667 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 9 | 2 | 13 | 0.6458 | 0.8182 |
| controlled_alt_vs_candidate_no_context | context_overlap | 10 | 13 | 1 | 0.4375 | 0.4348 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 11 | 1 | 12 | 0.7083 | 0.9167 |
| controlled_alt_vs_candidate_no_context | persona_style | 3 | 1 | 20 | 0.5417 | 0.7500 |
| controlled_alt_vs_candidate_no_context | distinct1 | 16 | 5 | 3 | 0.7292 | 0.7619 |
| controlled_alt_vs_candidate_no_context | length_score | 11 | 11 | 2 | 0.5000 | 0.5000 |
| controlled_alt_vs_candidate_no_context | sentence_score | 6 | 1 | 17 | 0.6042 | 0.8571 |
| controlled_alt_vs_candidate_no_context | overall_quality | 19 | 4 | 1 | 0.8125 | 0.8261 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.2083 | 0.2083 | 0.7917 |
| proposed_contextual_controlled_tuned | 0.0000 | 0.0000 | 0.2917 | 0.0417 | 0.9583 |
| proposed_contextual | 0.0000 | 0.0000 | 0.6250 | 0.0000 | 0.0000 |
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
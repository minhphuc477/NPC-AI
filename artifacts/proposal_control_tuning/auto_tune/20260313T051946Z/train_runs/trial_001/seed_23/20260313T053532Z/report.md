# Proposal Alignment Evaluation Report

- Run ID: `20260313T053532Z`
- Generated: `2026-03-13T05:40:07.222511+00:00`
- Scenarios: `artifacts\proposal_control_tuning\auto_tune\20260313T051946Z\train_runs\trial_001\seed_23\20260313T053532Z\scenarios.jsonl`
- Scenario count: `24`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_tuned`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.1154 (0.0716, 0.1605) | 0.2929 (0.2265, 0.3645) | 0.8813 (0.8620, 0.9023) | 0.3317 (0.2907, 0.3722) | n/a |
| proposed_contextual_controlled_tuned | 0.1187 (0.0670, 0.1770) | 0.2842 (0.2219, 0.3532) | 0.8786 (0.8587, 0.8992) | 0.3292 (0.2922, 0.3715) | n/a |
| proposed_contextual | 0.0671 (0.0371, 0.1061) | 0.2285 (0.1780, 0.2880) | 0.8805 (0.8657, 0.8952) | 0.2841 (0.2552, 0.3153) | n/a |
| candidate_no_context | 0.0274 (0.0162, 0.0402) | 0.2587 (0.1966, 0.3217) | 0.8920 (0.8736, 0.9103) | 0.2795 (0.2509, 0.3090) | n/a |

## Game-facing Outcome Metrics (mean, 95% CI)
| Arm | Quest-state Correctness | Lore Consistency | Contradiction Safety | Objective Completion Support | Gameplay Usefulness | Time-pressure Acceptability |
|---|---:|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.1971 (0.1584, 0.2387) | 0.0467 (0.0165, 0.0821) | 1.0000 (1.0000, 1.0000) | 0.0888 (0.0638, 0.1130) | 0.3120 (0.2978, 0.3268) | 0.2926 (0.2771, 0.3079) |
| proposed_contextual_controlled_tuned | 0.1986 (0.1560, 0.2496) | 0.0590 (0.0279, 0.0946) | 1.0000 (1.0000, 1.0000) | 0.0633 (0.0423, 0.0862) | 0.3031 (0.2867, 0.3193) | 0.2807 (0.2657, 0.2960) |
| proposed_contextual | 0.1558 (0.1300, 0.1875) | 0.0252 (0.0083, 0.0450) | 1.0000 (1.0000, 1.0000) | 0.0679 (0.0435, 0.0927) | 0.2987 (0.2880, 0.3085) | 0.2924 (0.2797, 0.3052) |
| candidate_no_context | 0.1191 (0.1106, 0.1291) | 0.0047 (0.0004, 0.0108) | 1.0000 (1.0000, 1.0000) | 0.0648 (0.0437, 0.0874) | 0.2888 (0.2803, 0.2973) | 0.3016 (0.2888, 0.3149) |

- Multi-turn contradiction rate is reported as `1 - contradiction_safety` in row-level outputs.

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0397 | 1.4493 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0302 | -0.1166 |
| proposed_vs_candidate_no_context | naturalness | -0.0115 | -0.0129 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.0367 | 0.3079 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0205 | 4.3610 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | 0.0031 | 0.0482 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 0.0099 | 0.0344 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | -0.0092 | -0.0305 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0492 | 2.1667 |
| proposed_vs_candidate_no_context | context_overlap | 0.0175 | 0.4565 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0377 | -0.1919 |
| proposed_vs_candidate_no_context | persona_style | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | distinct1 | -0.0138 | -0.0147 |
| proposed_vs_candidate_no_context | length_score | -0.0444 | -0.0726 |
| proposed_vs_candidate_no_context | sentence_score | 0.0292 | 0.0315 |
| proposed_vs_candidate_no_context | overall_quality | 0.0046 | 0.0166 |
| controlled_vs_proposed_raw | context_relevance | 0.0483 | 0.7196 |
| controlled_vs_proposed_raw | persona_consistency | 0.0644 | 0.2817 |
| controlled_vs_proposed_raw | naturalness | 0.0009 | 0.0010 |
| controlled_vs_proposed_raw | quest_state_correctness | 0.0412 | 0.2647 |
| controlled_vs_proposed_raw | lore_consistency | 0.0216 | 0.8557 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0209 | 0.3078 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0133 | 0.0446 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 0.0002 | 0.0007 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.0686 | 0.9526 |
| controlled_vs_proposed_raw | context_overlap | 0.0010 | 0.0184 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.0873 | 0.5500 |
| controlled_vs_proposed_raw | persona_style | -0.0273 | -0.0538 |
| controlled_vs_proposed_raw | distinct1 | 0.0074 | 0.0080 |
| controlled_vs_proposed_raw | length_score | -0.0250 | -0.0440 |
| controlled_vs_proposed_raw | sentence_score | 0.0292 | 0.0305 |
| controlled_vs_proposed_raw | overall_quality | 0.0476 | 0.1675 |
| controlled_vs_candidate_no_context | context_relevance | 0.0880 | 3.2119 |
| controlled_vs_candidate_no_context | persona_consistency | 0.0342 | 0.1323 |
| controlled_vs_candidate_no_context | naturalness | -0.0106 | -0.0119 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.0779 | 0.6541 |
| controlled_vs_candidate_no_context | lore_consistency | 0.0420 | 8.9483 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | 0.0240 | 0.3708 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0232 | 0.0805 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | -0.0090 | -0.0298 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.1178 | 5.1833 |
| controlled_vs_candidate_no_context | context_overlap | 0.0185 | 0.4833 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.0496 | 0.2525 |
| controlled_vs_candidate_no_context | persona_style | -0.0273 | -0.0538 |
| controlled_vs_candidate_no_context | distinct1 | -0.0064 | -0.0068 |
| controlled_vs_candidate_no_context | length_score | -0.0694 | -0.1134 |
| controlled_vs_candidate_no_context | sentence_score | 0.0583 | 0.0629 |
| controlled_vs_candidate_no_context | overall_quality | 0.0522 | 0.1869 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0033 | 0.0284 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0088 | -0.0299 |
| controlled_alt_vs_controlled_default | naturalness | -0.0027 | -0.0031 |
| controlled_alt_vs_controlled_default | quest_state_correctness | 0.0015 | 0.0077 |
| controlled_alt_vs_controlled_default | lore_consistency | 0.0123 | 0.2629 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_controlled_default | objective_completion_support | -0.0256 | -0.2877 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | -0.0089 | -0.0285 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | -0.0119 | -0.0408 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0038 | 0.0270 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0021 | 0.0370 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0139 | -0.0565 |
| controlled_alt_vs_controlled_default | persona_style | 0.0117 | 0.0244 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0028 | -0.0030 |
| controlled_alt_vs_controlled_default | length_score | -0.0153 | -0.0281 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0146 | 0.0148 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0025 | -0.0076 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.0516 | 0.7685 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.0556 | 0.2433 |
| controlled_alt_vs_proposed_raw | naturalness | -0.0018 | -0.0021 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 0.0428 | 0.2744 |
| controlled_alt_vs_proposed_raw | lore_consistency | 0.0338 | 1.3436 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_proposed_raw | objective_completion_support | -0.0047 | -0.0685 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 0.0044 | 0.0148 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | -0.0117 | -0.0401 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.0723 | 1.0053 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0031 | 0.0562 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.0734 | 0.4625 |
| controlled_alt_vs_proposed_raw | persona_style | -0.0156 | -0.0308 |
| controlled_alt_vs_proposed_raw | distinct1 | 0.0046 | 0.0050 |
| controlled_alt_vs_proposed_raw | length_score | -0.0403 | -0.0709 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0438 | 0.0458 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.0451 | 0.1587 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.0913 | 3.3317 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.0254 | 0.0984 |
| controlled_alt_vs_candidate_no_context | naturalness | -0.0133 | -0.0150 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 0.0794 | 0.6667 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 0.0543 | 11.5642 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_candidate_no_context | objective_completion_support | -0.0015 | -0.0236 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 0.0143 | 0.0496 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | -0.0209 | -0.0694 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.1216 | 5.3500 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0206 | 0.5383 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.0357 | 0.1818 |
| controlled_alt_vs_candidate_no_context | persona_style | -0.0156 | -0.0308 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0092 | -0.0098 |
| controlled_alt_vs_candidate_no_context | length_score | -0.0847 | -0.1383 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.0729 | 0.0787 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.0497 | 0.1778 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0397 | (0.0080, 0.0805) | 0.0027 | 0.0397 | (0.0010, 0.0754) | 0.0083 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0302 | (-0.0778, 0.0159) | 0.8917 | -0.0302 | (-0.0686, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | naturalness | -0.0115 | (-0.0282, 0.0051) | 0.9157 | -0.0115 | (-0.0314, 0.0009) | 0.9313 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.0367 | (0.0095, 0.0711) | 0.0020 | 0.0367 | (0.0026, 0.0613) | 0.0113 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0205 | (0.0022, 0.0413) | 0.0127 | 0.0205 | (0.0000, 0.0396) | 0.0783 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | 0.0031 | (-0.0142, 0.0206) | 0.3640 | 0.0031 | (-0.0193, 0.0305) | 0.4227 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 0.0099 | (0.0023, 0.0188) | 0.0023 | 0.0099 | (0.0009, 0.0183) | 0.0077 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | -0.0092 | (-0.0256, 0.0054) | 0.8843 | -0.0092 | (-0.0334, 0.0100) | 0.6753 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0492 | (0.0076, 0.1061) | 0.0137 | 0.0492 | (0.0000, 0.0942) | 0.0853 |
| proposed_vs_candidate_no_context | context_overlap | 0.0175 | (0.0061, 0.0300) | 0.0000 | 0.0175 | (0.0033, 0.0333) | 0.0097 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0377 | (-0.0982, 0.0149) | 0.8980 | -0.0377 | (-0.0857, -0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | persona_style | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | distinct1 | -0.0138 | (-0.0293, 0.0008) | 0.9683 | -0.0138 | (-0.0279, -0.0013) | 1.0000 |
| proposed_vs_candidate_no_context | length_score | -0.0444 | (-0.1111, 0.0194) | 0.9077 | -0.0444 | (-0.1050, -0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | sentence_score | 0.0292 | (-0.0437, 0.1021) | 0.2740 | 0.0292 | (0.0000, 0.0808) | 0.3277 |
| proposed_vs_candidate_no_context | overall_quality | 0.0046 | (-0.0211, 0.0303) | 0.3630 | 0.0046 | (-0.0210, 0.0229) | 0.3687 |
| controlled_vs_proposed_raw | context_relevance | 0.0483 | (0.0052, 0.0944) | 0.0143 | 0.0483 | (0.0122, 0.0741) | 0.0003 |
| controlled_vs_proposed_raw | persona_consistency | 0.0644 | (-0.0092, 0.1453) | 0.0463 | 0.0644 | (-0.0565, 0.1896) | 0.1827 |
| controlled_vs_proposed_raw | naturalness | 0.0009 | (-0.0245, 0.0253) | 0.4683 | 0.0009 | (-0.0131, 0.0150) | 0.3720 |
| controlled_vs_proposed_raw | quest_state_correctness | 0.0412 | (0.0065, 0.0798) | 0.0080 | 0.0412 | (0.0079, 0.0673) | 0.0007 |
| controlled_vs_proposed_raw | lore_consistency | 0.0216 | (-0.0068, 0.0561) | 0.0807 | 0.0216 | (0.0057, 0.0334) | 0.0127 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0209 | (-0.0008, 0.0431) | 0.0297 | 0.0209 | (0.0060, 0.0324) | 0.0103 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0133 | (-0.0019, 0.0297) | 0.0417 | 0.0133 | (0.0017, 0.0263) | 0.0090 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 0.0002 | (-0.0160, 0.0144) | 0.4750 | 0.0002 | (-0.0112, 0.0110) | 0.4460 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.0686 | (0.0117, 0.1254) | 0.0087 | 0.0686 | (0.0227, 0.1039) | 0.0003 |
| controlled_vs_proposed_raw | context_overlap | 0.0010 | (-0.0145, 0.0168) | 0.4593 | 0.0010 | (-0.0173, 0.0124) | 0.4383 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.0873 | (-0.0030, 0.1875) | 0.0260 | 0.0873 | (-0.0571, 0.2308) | 0.1527 |
| controlled_vs_proposed_raw | persona_style | -0.0273 | (-0.0638, 0.0013) | 0.9720 | -0.0273 | (-0.0753, 0.0000) | 1.0000 |
| controlled_vs_proposed_raw | distinct1 | 0.0074 | (-0.0141, 0.0272) | 0.2320 | 0.0074 | (-0.0011, 0.0191) | 0.0660 |
| controlled_vs_proposed_raw | length_score | -0.0250 | (-0.1319, 0.0750) | 0.6823 | -0.0250 | (-0.0962, 0.0474) | 0.7340 |
| controlled_vs_proposed_raw | sentence_score | 0.0292 | (-0.0292, 0.0875) | 0.2287 | 0.0292 | (0.0000, 0.0500) | 0.0830 |
| controlled_vs_proposed_raw | overall_quality | 0.0476 | (0.0051, 0.0916) | 0.0130 | 0.0476 | (-0.0121, 0.1031) | 0.0620 |
| controlled_vs_candidate_no_context | context_relevance | 0.0880 | (0.0424, 0.1344) | 0.0000 | 0.0880 | (0.0132, 0.1414) | 0.0007 |
| controlled_vs_candidate_no_context | persona_consistency | 0.0342 | (-0.0514, 0.1271) | 0.2377 | 0.0342 | (-0.0867, 0.1681) | 0.3380 |
| controlled_vs_candidate_no_context | naturalness | -0.0106 | (-0.0344, 0.0135) | 0.8043 | -0.0106 | (-0.0295, 0.0039) | 0.9110 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.0779 | (0.0380, 0.1211) | 0.0000 | 0.0779 | (0.0105, 0.1261) | 0.0000 |
| controlled_vs_candidate_no_context | lore_consistency | 0.0420 | (0.0131, 0.0751) | 0.0000 | 0.0420 | (0.0057, 0.0719) | 0.0043 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | 0.0240 | (0.0043, 0.0456) | 0.0077 | 0.0240 | (0.0006, 0.0526) | 0.0113 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0232 | (0.0070, 0.0398) | 0.0007 | 0.0232 | (0.0042, 0.0377) | 0.0123 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | -0.0090 | (-0.0234, 0.0030) | 0.9237 | -0.0090 | (-0.0250, 0.0077) | 0.8007 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.1178 | (0.0606, 0.1784) | 0.0000 | 0.1178 | (0.0227, 0.1831) | 0.0003 |
| controlled_vs_candidate_no_context | context_overlap | 0.0185 | (0.0031, 0.0344) | 0.0077 | 0.0185 | (-0.0098, 0.0385) | 0.0817 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.0496 | (-0.0546, 0.1587) | 0.2037 | 0.0496 | (-0.0909, 0.2092) | 0.3383 |
| controlled_vs_candidate_no_context | persona_style | -0.0273 | (-0.0612, 0.0013) | 0.9673 | -0.0273 | (-0.0753, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | distinct1 | -0.0064 | (-0.0247, 0.0108) | 0.7540 | -0.0064 | (-0.0174, 0.0077) | 0.8307 |
| controlled_vs_candidate_no_context | length_score | -0.0694 | (-0.1708, 0.0306) | 0.9183 | -0.0694 | (-0.1576, 0.0056) | 0.9683 |
| controlled_vs_candidate_no_context | sentence_score | 0.0583 | (0.0146, 0.1167) | 0.0130 | 0.0583 | (0.0000, 0.1212) | 0.0727 |
| controlled_vs_candidate_no_context | overall_quality | 0.0522 | (0.0032, 0.1019) | 0.0163 | 0.0522 | (-0.0205, 0.1160) | 0.1003 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0033 | (-0.0342, 0.0466) | 0.4550 | 0.0033 | (-0.0399, 0.0386) | 0.4453 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0088 | (-0.0984, 0.0735) | 0.5770 | -0.0088 | (-0.0901, 0.0615) | 0.5737 |
| controlled_alt_vs_controlled_default | naturalness | -0.0027 | (-0.0307, 0.0242) | 0.5643 | -0.0027 | (-0.0233, 0.0167) | 0.5990 |
| controlled_alt_vs_controlled_default | quest_state_correctness | 0.0015 | (-0.0326, 0.0401) | 0.4900 | 0.0015 | (-0.0351, 0.0376) | 0.4437 |
| controlled_alt_vs_controlled_default | lore_consistency | 0.0123 | (-0.0164, 0.0444) | 0.2267 | 0.0123 | (-0.0114, 0.0323) | 0.1537 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | objective_completion_support | -0.0256 | (-0.0442, -0.0094) | 0.9993 | -0.0256 | (-0.0539, -0.0036) | 1.0000 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | -0.0089 | (-0.0275, 0.0106) | 0.8197 | -0.0089 | (-0.0278, 0.0086) | 0.8483 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | -0.0119 | (-0.0320, 0.0076) | 0.8680 | -0.0119 | (-0.0281, 0.0042) | 0.9160 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0038 | (-0.0455, 0.0644) | 0.4900 | 0.0038 | (-0.0545, 0.0524) | 0.4620 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0021 | (-0.0099, 0.0160) | 0.4003 | 0.0021 | (-0.0056, 0.0076) | 0.2920 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0139 | (-0.1181, 0.0834) | 0.6150 | -0.0139 | (-0.1154, 0.0769) | 0.5680 |
| controlled_alt_vs_controlled_default | persona_style | 0.0117 | (-0.0378, 0.0651) | 0.3380 | 0.0117 | (-0.0273, 0.0641) | 0.3480 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0028 | (-0.0134, 0.0085) | 0.7117 | -0.0028 | (-0.0096, 0.0053) | 0.7687 |
| controlled_alt_vs_controlled_default | length_score | -0.0153 | (-0.1500, 0.1084) | 0.5907 | -0.0153 | (-0.1154, 0.0806) | 0.6113 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0146 | (0.0000, 0.0437) | 0.3433 | 0.0146 | (0.0000, 0.0404) | 0.3307 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0025 | (-0.0355, 0.0305) | 0.5583 | -0.0025 | (-0.0314, 0.0354) | 0.5943 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.0516 | (0.0066, 0.1015) | 0.0120 | 0.0516 | (-0.0177, 0.1011) | 0.0737 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.0556 | (-0.0089, 0.1307) | 0.0447 | 0.0556 | (-0.0469, 0.1265) | 0.1490 |
| controlled_alt_vs_proposed_raw | naturalness | -0.0018 | (-0.0264, 0.0219) | 0.5673 | -0.0018 | (-0.0094, 0.0054) | 0.6807 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 0.0428 | (0.0022, 0.0874) | 0.0190 | 0.0428 | (-0.0206, 0.0891) | 0.0887 |
| controlled_alt_vs_proposed_raw | lore_consistency | 0.0338 | (0.0043, 0.0658) | 0.0100 | 0.0338 | (0.0000, 0.0635) | 0.0727 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_proposed_raw | objective_completion_support | -0.0047 | (-0.0282, 0.0180) | 0.6617 | -0.0047 | (-0.0275, 0.0093) | 0.6723 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 0.0044 | (-0.0117, 0.0211) | 0.2953 | 0.0044 | (-0.0171, 0.0198) | 0.3263 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | -0.0117 | (-0.0277, 0.0017) | 0.9527 | -0.0117 | (-0.0214, -0.0022) | 1.0000 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.0723 | (0.0152, 0.1371) | 0.0057 | 0.0723 | (-0.0182, 0.1471) | 0.0793 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0031 | (-0.0138, 0.0216) | 0.3690 | 0.0031 | (-0.0190, 0.0175) | 0.3797 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.0734 | (-0.0030, 0.1607) | 0.0330 | 0.0734 | (-0.0455, 0.1582) | 0.0967 |
| controlled_alt_vs_proposed_raw | persona_style | -0.0156 | (-0.0573, 0.0299) | 0.7753 | -0.0156 | (-0.0511, 0.0000) | 1.0000 |
| controlled_alt_vs_proposed_raw | distinct1 | 0.0046 | (-0.0142, 0.0221) | 0.2860 | 0.0046 | (-0.0083, 0.0213) | 0.2787 |
| controlled_alt_vs_proposed_raw | length_score | -0.0403 | (-0.1389, 0.0569) | 0.7893 | -0.0403 | (-0.0778, -0.0028) | 1.0000 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0437 | (0.0000, 0.0875) | 0.0403 | 0.0437 | (0.0000, 0.0875) | 0.0737 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.0451 | (0.0048, 0.0887) | 0.0107 | 0.0451 | (-0.0240, 0.0944) | 0.0913 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.0913 | (0.0391, 0.1518) | 0.0000 | 0.0913 | (-0.0168, 0.1829) | 0.0660 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.0254 | (-0.0604, 0.1113) | 0.3020 | 0.0254 | (-0.0723, 0.0952) | 0.2863 |
| controlled_alt_vs_candidate_no_context | naturalness | -0.0133 | (-0.0365, 0.0091) | 0.8810 | -0.0133 | (-0.0315, 0.0029) | 0.9423 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 0.0794 | (0.0333, 0.1328) | 0.0000 | 0.0794 | (-0.0159, 0.1563) | 0.0653 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 0.0543 | (0.0226, 0.0882) | 0.0000 | 0.0543 | (0.0000, 0.1065) | 0.0760 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_candidate_no_context | objective_completion_support | -0.0015 | (-0.0265, 0.0211) | 0.5507 | -0.0015 | (-0.0227, 0.0186) | 0.6420 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 0.0143 | (-0.0027, 0.0316) | 0.0473 | 0.0143 | (-0.0144, 0.0359) | 0.1603 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | -0.0209 | (-0.0403, -0.0037) | 0.9937 | -0.0209 | (-0.0489, -0.0012) | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.1216 | (0.0496, 0.2008) | 0.0003 | 0.1216 | (-0.0182, 0.2412) | 0.0803 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0206 | (0.0037, 0.0386) | 0.0090 | 0.0206 | (-0.0134, 0.0466) | 0.1260 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.0357 | (-0.0645, 0.1399) | 0.2573 | 0.0357 | (-0.0810, 0.1259) | 0.2560 |
| controlled_alt_vs_candidate_no_context | persona_style | -0.0156 | (-0.0586, 0.0273) | 0.7737 | -0.0156 | (-0.0511, 0.0000) | 1.0000 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0092 | (-0.0297, 0.0089) | 0.8237 | -0.0092 | (-0.0266, 0.0128) | 0.7977 |
| controlled_alt_vs_candidate_no_context | length_score | -0.0847 | (-0.1833, 0.0042) | 0.9677 | -0.0847 | (-0.1528, -0.0167) | 1.0000 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.0729 | (0.0146, 0.1313) | 0.0037 | 0.0729 | (0.0000, 0.1625) | 0.0760 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.0497 | (-0.0007, 0.0997) | 0.0277 | 0.0497 | (-0.0342, 0.1122) | 0.1073 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 12 | 2 | 10 | 0.7083 | 0.8571 |
| proposed_vs_candidate_no_context | persona_consistency | 3 | 4 | 17 | 0.4792 | 0.4286 |
| proposed_vs_candidate_no_context | naturalness | 4 | 10 | 10 | 0.3750 | 0.2857 |
| proposed_vs_candidate_no_context | quest_state_correctness | 12 | 2 | 10 | 0.7083 | 0.8571 |
| proposed_vs_candidate_no_context | lore_consistency | 6 | 1 | 17 | 0.6042 | 0.8571 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| proposed_vs_candidate_no_context | objective_completion_support | 4 | 7 | 13 | 0.4375 | 0.3636 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 10 | 4 | 10 | 0.6250 | 0.7143 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | 5 | 7 | 12 | 0.4583 | 0.4167 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 7 | 2 | 15 | 0.6042 | 0.7778 |
| proposed_vs_candidate_no_context | context_overlap | 12 | 2 | 10 | 0.7083 | 0.8571 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 3 | 4 | 17 | 0.4792 | 0.4286 |
| proposed_vs_candidate_no_context | persona_style | 0 | 0 | 24 | 0.5000 | nan |
| proposed_vs_candidate_no_context | distinct1 | 4 | 9 | 11 | 0.3958 | 0.3077 |
| proposed_vs_candidate_no_context | length_score | 5 | 8 | 11 | 0.4375 | 0.3846 |
| proposed_vs_candidate_no_context | sentence_score | 4 | 2 | 18 | 0.5417 | 0.6667 |
| proposed_vs_candidate_no_context | overall_quality | 8 | 6 | 10 | 0.5417 | 0.5714 |
| controlled_vs_proposed_raw | context_relevance | 13 | 7 | 4 | 0.6250 | 0.6500 |
| controlled_vs_proposed_raw | persona_consistency | 7 | 7 | 10 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | naturalness | 10 | 10 | 4 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | quest_state_correctness | 14 | 6 | 4 | 0.6667 | 0.7000 |
| controlled_vs_proposed_raw | lore_consistency | 6 | 5 | 13 | 0.5208 | 0.5455 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_vs_proposed_raw | objective_completion_support | 11 | 2 | 11 | 0.6875 | 0.8462 |
| controlled_vs_proposed_raw | gameplay_usefulness | 11 | 9 | 4 | 0.5417 | 0.5500 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 8 | 5 | 11 | 0.5625 | 0.6154 |
| controlled_vs_proposed_raw | context_keyword_coverage | 11 | 4 | 9 | 0.6458 | 0.7333 |
| controlled_vs_proposed_raw | context_overlap | 11 | 9 | 4 | 0.5417 | 0.5500 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 7 | 5 | 12 | 0.5417 | 0.5833 |
| controlled_vs_proposed_raw | persona_style | 1 | 5 | 18 | 0.4167 | 0.1667 |
| controlled_vs_proposed_raw | distinct1 | 13 | 7 | 4 | 0.6250 | 0.6500 |
| controlled_vs_proposed_raw | length_score | 9 | 10 | 5 | 0.4792 | 0.4737 |
| controlled_vs_proposed_raw | sentence_score | 3 | 1 | 20 | 0.5417 | 0.7500 |
| controlled_vs_proposed_raw | overall_quality | 13 | 7 | 4 | 0.6250 | 0.6500 |
| controlled_vs_candidate_no_context | context_relevance | 16 | 4 | 4 | 0.7500 | 0.8000 |
| controlled_vs_candidate_no_context | persona_consistency | 8 | 8 | 8 | 0.5000 | 0.5000 |
| controlled_vs_candidate_no_context | naturalness | 8 | 12 | 4 | 0.4167 | 0.4000 |
| controlled_vs_candidate_no_context | quest_state_correctness | 16 | 4 | 4 | 0.7500 | 0.8000 |
| controlled_vs_candidate_no_context | lore_consistency | 8 | 2 | 14 | 0.6250 | 0.8000 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_vs_candidate_no_context | objective_completion_support | 9 | 3 | 12 | 0.6250 | 0.7500 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 13 | 7 | 4 | 0.6250 | 0.6500 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 5 | 9 | 10 | 0.4167 | 0.3571 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 14 | 3 | 7 | 0.7292 | 0.8235 |
| controlled_vs_candidate_no_context | context_overlap | 16 | 4 | 4 | 0.7500 | 0.8000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 8 | 7 | 9 | 0.5208 | 0.5333 |
| controlled_vs_candidate_no_context | persona_style | 1 | 5 | 18 | 0.4167 | 0.1667 |
| controlled_vs_candidate_no_context | distinct1 | 11 | 8 | 5 | 0.5625 | 0.5789 |
| controlled_vs_candidate_no_context | length_score | 5 | 14 | 5 | 0.3125 | 0.2632 |
| controlled_vs_candidate_no_context | sentence_score | 4 | 0 | 20 | 0.5833 | 1.0000 |
| controlled_vs_candidate_no_context | overall_quality | 13 | 7 | 4 | 0.6250 | 0.6500 |
| controlled_alt_vs_controlled_default | context_relevance | 5 | 9 | 10 | 0.4167 | 0.3571 |
| controlled_alt_vs_controlled_default | persona_consistency | 6 | 6 | 12 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | naturalness | 8 | 6 | 10 | 0.5417 | 0.5714 |
| controlled_alt_vs_controlled_default | quest_state_correctness | 5 | 9 | 10 | 0.4167 | 0.3571 |
| controlled_alt_vs_controlled_default | lore_consistency | 4 | 7 | 13 | 0.4375 | 0.3636 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_alt_vs_controlled_default | objective_completion_support | 2 | 10 | 12 | 0.3333 | 0.1667 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | 4 | 10 | 10 | 0.3750 | 0.2857 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | 4 | 8 | 12 | 0.4167 | 0.3333 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 5 | 7 | 12 | 0.4583 | 0.4167 |
| controlled_alt_vs_controlled_default | context_overlap | 6 | 8 | 10 | 0.4583 | 0.4286 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 6 | 5 | 13 | 0.5208 | 0.5455 |
| controlled_alt_vs_controlled_default | persona_style | 2 | 2 | 20 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | distinct1 | 5 | 8 | 11 | 0.4375 | 0.3846 |
| controlled_alt_vs_controlled_default | length_score | 6 | 7 | 11 | 0.4792 | 0.4615 |
| controlled_alt_vs_controlled_default | sentence_score | 1 | 0 | 23 | 0.5208 | 1.0000 |
| controlled_alt_vs_controlled_default | overall_quality | 6 | 8 | 10 | 0.4583 | 0.4286 |
| controlled_alt_vs_proposed_raw | context_relevance | 13 | 7 | 4 | 0.6250 | 0.6500 |
| controlled_alt_vs_proposed_raw | persona_consistency | 9 | 6 | 9 | 0.5625 | 0.6000 |
| controlled_alt_vs_proposed_raw | naturalness | 9 | 11 | 4 | 0.4583 | 0.4500 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 12 | 8 | 4 | 0.5833 | 0.6000 |
| controlled_alt_vs_proposed_raw | lore_consistency | 9 | 3 | 12 | 0.6250 | 0.7500 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_alt_vs_proposed_raw | objective_completion_support | 5 | 8 | 11 | 0.4375 | 0.3846 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 8 | 12 | 4 | 0.4167 | 0.4000 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | 5 | 8 | 11 | 0.4375 | 0.3846 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 9 | 4 | 11 | 0.6042 | 0.6923 |
| controlled_alt_vs_proposed_raw | context_overlap | 10 | 10 | 4 | 0.5000 | 0.5000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 8 | 4 | 12 | 0.5833 | 0.6667 |
| controlled_alt_vs_proposed_raw | persona_style | 2 | 6 | 16 | 0.4167 | 0.2500 |
| controlled_alt_vs_proposed_raw | distinct1 | 12 | 7 | 5 | 0.6042 | 0.6316 |
| controlled_alt_vs_proposed_raw | length_score | 8 | 11 | 5 | 0.4375 | 0.4211 |
| controlled_alt_vs_proposed_raw | sentence_score | 3 | 0 | 21 | 0.5625 | 1.0000 |
| controlled_alt_vs_proposed_raw | overall_quality | 12 | 8 | 4 | 0.5833 | 0.6000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 14 | 6 | 4 | 0.6667 | 0.7000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 10 | 8 | 6 | 0.5417 | 0.5556 |
| controlled_alt_vs_candidate_no_context | naturalness | 7 | 13 | 4 | 0.3750 | 0.3500 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 14 | 6 | 4 | 0.6667 | 0.7000 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 10 | 1 | 13 | 0.6875 | 0.9091 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_alt_vs_candidate_no_context | objective_completion_support | 6 | 8 | 10 | 0.4583 | 0.4286 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 12 | 8 | 4 | 0.5833 | 0.6000 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | 4 | 11 | 9 | 0.3542 | 0.2667 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 11 | 3 | 10 | 0.6667 | 0.7857 |
| controlled_alt_vs_candidate_no_context | context_overlap | 14 | 6 | 4 | 0.6667 | 0.7000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 10 | 8 | 6 | 0.5417 | 0.5556 |
| controlled_alt_vs_candidate_no_context | persona_style | 2 | 6 | 16 | 0.4167 | 0.2500 |
| controlled_alt_vs_candidate_no_context | distinct1 | 10 | 9 | 5 | 0.5208 | 0.5263 |
| controlled_alt_vs_candidate_no_context | length_score | 7 | 13 | 4 | 0.3750 | 0.3500 |
| controlled_alt_vs_candidate_no_context | sentence_score | 5 | 0 | 19 | 0.6042 | 1.0000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 13 | 7 | 4 | 0.6250 | 0.6500 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.0417 | 0.2500 | 0.7500 |
| proposed_contextual_controlled_tuned | 0.0000 | 0.0000 | 0.0417 | 0.3333 | 0.6667 |
| proposed_contextual | 0.0000 | 0.0000 | 0.5833 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.4583 | 0.0000 | 0.0000 |

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
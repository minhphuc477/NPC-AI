# Proposal Alignment Evaluation Report

- Run ID: `20260313T070841Z`
- Generated: `2026-03-13T07:15:20.158230+00:00`
- Scenarios: `artifacts\proposal_control_tuning\auto_tune\20260313T051946Z\train_runs\trial_009\seed_23\20260313T070841Z\scenarios.jsonl`
- Scenario count: `24`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_tuned`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0787 (0.0456, 0.1144) | 0.2408 (0.1963, 0.2859) | 0.8766 (0.8611, 0.8927) | 0.2939 (0.2609, 0.3280) | n/a |
| proposed_contextual_controlled_tuned | 0.0769 (0.0425, 0.1187) | 0.2780 (0.2188, 0.3375) | 0.8674 (0.8520, 0.8838) | 0.3051 (0.2746, 0.3377) | n/a |
| proposed_contextual | 0.0786 (0.0398, 0.1270) | 0.2505 (0.1896, 0.3164) | 0.8781 (0.8647, 0.8922) | 0.2970 (0.2637, 0.3338) | n/a |
| candidate_no_context | 0.0351 (0.0217, 0.0485) | 0.2420 (0.1772, 0.3153) | 0.8778 (0.8581, 0.8965) | 0.2738 (0.2452, 0.3026) | n/a |

## Game-facing Outcome Metrics (mean, 95% CI)
| Arm | Quest-state Correctness | Lore Consistency | Contradiction Safety | Objective Completion Support | Gameplay Usefulness | Time-pressure Acceptability |
|---|---:|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.1660 (0.1358, 0.1990) | 0.0192 (0.0049, 0.0375) | 0.8333 (0.6667, 0.9583) | 0.0877 (0.0645, 0.1097) | 0.3042 (0.2924, 0.3163) | 0.2987 (0.2812, 0.3144) |
| proposed_contextual_controlled_tuned | 0.1622 (0.1305, 0.1998) | 0.0227 (0.0029, 0.0542) | 1.0000 (1.0000, 1.0000) | 0.0740 (0.0498, 0.0999) | 0.2905 (0.2729, 0.3068) | 0.2793 (0.2608, 0.2965) |
| proposed_contextual | 0.1642 (0.1315, 0.2003) | 0.0250 (0.0038, 0.0554) | 1.0000 (1.0000, 1.0000) | 0.0687 (0.0456, 0.0929) | 0.2982 (0.2868, 0.3096) | 0.2909 (0.2767, 0.3047) |
| candidate_no_context | 0.1263 (0.1157, 0.1374) | 0.0058 (0.0011, 0.0120) | 1.0000 (1.0000, 1.0000) | 0.0487 (0.0282, 0.0710) | 0.2750 (0.2627, 0.2871) | 0.2802 (0.2678, 0.2916) |

- Multi-turn contradiction rate is reported as `1 - contradiction_safety` in row-level outputs.

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0435 | 1.2411 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0085 | 0.0351 |
| proposed_vs_candidate_no_context | naturalness | 0.0002 | 0.0003 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.0379 | 0.3000 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0193 | 3.3463 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | 0.0200 | 0.4108 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 0.0232 | 0.0845 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | 0.0107 | 0.0383 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0534 | 1.5667 |
| proposed_vs_candidate_no_context | context_overlap | 0.0204 | 0.5468 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0119 | 0.0694 |
| proposed_vs_candidate_no_context | persona_style | -0.0052 | -0.0100 |
| proposed_vs_candidate_no_context | distinct1 | -0.0112 | -0.0118 |
| proposed_vs_candidate_no_context | length_score | 0.0236 | 0.0443 |
| proposed_vs_candidate_no_context | sentence_score | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | overall_quality | 0.0232 | 0.0849 |
| controlled_vs_proposed_raw | context_relevance | 0.0002 | 0.0023 |
| controlled_vs_proposed_raw | persona_consistency | -0.0097 | -0.0386 |
| controlled_vs_proposed_raw | naturalness | -0.0015 | -0.0017 |
| controlled_vs_proposed_raw | quest_state_correctness | 0.0019 | 0.0115 |
| controlled_vs_proposed_raw | lore_consistency | -0.0058 | -0.2334 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | -0.1667 | -0.1667 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0190 | 0.2770 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0060 | 0.0201 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 0.0077 | 0.0266 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.0015 | 0.0173 |
| controlled_vs_proposed_raw | context_overlap | -0.0029 | -0.0507 |
| controlled_vs_proposed_raw | persona_keyword_coverage | -0.0030 | -0.0162 |
| controlled_vs_proposed_raw | persona_style | -0.0365 | -0.0704 |
| controlled_vs_proposed_raw | distinct1 | -0.0028 | -0.0030 |
| controlled_vs_proposed_raw | length_score | -0.0236 | -0.0424 |
| controlled_vs_proposed_raw | sentence_score | 0.0437 | 0.0472 |
| controlled_vs_proposed_raw | overall_quality | -0.0032 | -0.0107 |
| controlled_vs_candidate_no_context | context_relevance | 0.0437 | 1.2463 |
| controlled_vs_candidate_no_context | persona_consistency | -0.0012 | -0.0049 |
| controlled_vs_candidate_no_context | naturalness | -0.0012 | -0.0014 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.0398 | 0.3150 |
| controlled_vs_candidate_no_context | lore_consistency | 0.0134 | 2.3319 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | -0.1667 | -0.1667 |
| controlled_vs_candidate_no_context | objective_completion_support | 0.0390 | 0.8017 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0292 | 0.1062 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 0.0185 | 0.0659 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.0549 | 1.6111 |
| controlled_vs_candidate_no_context | context_overlap | 0.0175 | 0.4684 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.0089 | 0.0520 |
| controlled_vs_candidate_no_context | persona_style | -0.0417 | -0.0796 |
| controlled_vs_candidate_no_context | distinct1 | -0.0140 | -0.0148 |
| controlled_vs_candidate_no_context | length_score | 0.0000 | 0.0000 |
| controlled_vs_candidate_no_context | sentence_score | 0.0437 | 0.0472 |
| controlled_vs_candidate_no_context | overall_quality | 0.0201 | 0.0733 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0018 | -0.0235 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0372 | 0.1545 |
| controlled_alt_vs_controlled_default | naturalness | -0.0092 | -0.0105 |
| controlled_alt_vs_controlled_default | quest_state_correctness | -0.0039 | -0.0234 |
| controlled_alt_vs_controlled_default | lore_consistency | 0.0035 | 0.1850 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0.1667 | 0.2000 |
| controlled_alt_vs_controlled_default | objective_completion_support | -0.0137 | -0.1560 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | -0.0137 | -0.0452 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | -0.0194 | -0.0650 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0030 | 0.0340 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0132 | -0.2417 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0387 | 0.2143 |
| controlled_alt_vs_controlled_default | persona_style | 0.0313 | 0.0649 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0066 | -0.0071 |
| controlled_alt_vs_controlled_default | length_score | -0.0403 | -0.0755 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0146 | 0.0150 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0112 | 0.0382 |
| controlled_alt_vs_proposed_raw | context_relevance | -0.0017 | -0.0212 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.0275 | 0.1099 |
| controlled_alt_vs_proposed_raw | naturalness | -0.0107 | -0.0122 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | -0.0020 | -0.0122 |
| controlled_alt_vs_proposed_raw | lore_consistency | -0.0023 | -0.0916 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_proposed_raw | objective_completion_support | 0.0053 | 0.0779 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | -0.0078 | -0.0260 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | -0.0117 | -0.0401 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.0045 | 0.0519 |
| controlled_alt_vs_proposed_raw | context_overlap | -0.0162 | -0.2801 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.0357 | 0.1946 |
| controlled_alt_vs_proposed_raw | persona_style | -0.0052 | -0.0101 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0094 | -0.0100 |
| controlled_alt_vs_proposed_raw | length_score | -0.0639 | -0.1147 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0583 | 0.0629 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.0080 | 0.0270 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.0418 | 1.1936 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.0360 | 0.1488 |
| controlled_alt_vs_candidate_no_context | naturalness | -0.0104 | -0.0119 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 0.0359 | 0.2842 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 0.0170 | 2.9483 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_candidate_no_context | objective_completion_support | 0.0253 | 0.5207 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 0.0155 | 0.0563 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | -0.0009 | -0.0034 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.0580 | 1.7000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0042 | 0.1135 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.0476 | 0.2775 |
| controlled_alt_vs_candidate_no_context | persona_style | -0.0104 | -0.0199 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0206 | -0.0217 |
| controlled_alt_vs_candidate_no_context | length_score | -0.0403 | -0.0755 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.0583 | 0.0629 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.0313 | 0.1142 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0435 | (0.0028, 0.0902) | 0.0163 | 0.0435 | (0.0007, 0.0912) | 0.0077 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0085 | (-0.0442, 0.0642) | 0.3497 | 0.0085 | (-0.0658, 0.0494) | 0.3697 |
| proposed_vs_candidate_no_context | naturalness | 0.0002 | (-0.0175, 0.0203) | 0.5050 | 0.0002 | (-0.0233, 0.0265) | 0.4743 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.0379 | (0.0055, 0.0768) | 0.0060 | 0.0379 | (0.0052, 0.0688) | 0.0003 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0193 | (-0.0006, 0.0506) | 0.0377 | 0.0193 | (-0.0002, 0.0522) | 0.0810 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | 0.0200 | (-0.0032, 0.0424) | 0.0510 | 0.0200 | (-0.0022, 0.0374) | 0.0430 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 0.0232 | (0.0094, 0.0381) | 0.0007 | 0.0232 | (0.0027, 0.0382) | 0.0090 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | 0.0107 | (-0.0016, 0.0243) | 0.0500 | 0.0107 | (-0.0001, 0.0234) | 0.0327 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0534 | (0.0038, 0.1174) | 0.0173 | 0.0534 | (0.0000, 0.1129) | 0.0733 |
| proposed_vs_candidate_no_context | context_overlap | 0.0204 | (0.0035, 0.0391) | 0.0050 | 0.0204 | (0.0024, 0.0405) | 0.0090 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0119 | (-0.0575, 0.0804) | 0.3717 | 0.0119 | (-0.0823, 0.0641) | 0.3540 |
| proposed_vs_candidate_no_context | persona_style | -0.0052 | (-0.0365, 0.0260) | 0.7177 | -0.0052 | (-0.0170, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | distinct1 | -0.0112 | (-0.0289, 0.0044) | 0.9157 | -0.0112 | (-0.0196, -0.0040) | 1.0000 |
| proposed_vs_candidate_no_context | length_score | 0.0236 | (-0.0486, 0.1028) | 0.2860 | 0.0236 | (-0.0733, 0.1410) | 0.3497 |
| proposed_vs_candidate_no_context | sentence_score | 0.0000 | (-0.0583, 0.0583) | 0.6120 | 0.0000 | (-0.0404, 0.0404) | 0.6440 |
| proposed_vs_candidate_no_context | overall_quality | 0.0232 | (-0.0061, 0.0529) | 0.0573 | 0.0232 | (-0.0034, 0.0488) | 0.0410 |
| controlled_vs_proposed_raw | context_relevance | 0.0002 | (-0.0516, 0.0479) | 0.4950 | 0.0002 | (-0.0371, 0.0386) | 0.4970 |
| controlled_vs_proposed_raw | persona_consistency | -0.0097 | (-0.0633, 0.0369) | 0.6467 | -0.0097 | (-0.1032, 0.0571) | 0.5817 |
| controlled_vs_proposed_raw | naturalness | -0.0015 | (-0.0206, 0.0197) | 0.5620 | -0.0015 | (-0.0178, 0.0087) | 0.5533 |
| controlled_vs_proposed_raw | quest_state_correctness | 0.0019 | (-0.0422, 0.0437) | 0.4687 | 0.0019 | (-0.0312, 0.0350) | 0.4543 |
| controlled_vs_proposed_raw | lore_consistency | -0.0058 | (-0.0396, 0.0219) | 0.6307 | -0.0058 | (-0.0370, 0.0246) | 0.5953 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | -0.1667 | (-0.3333, -0.0417) | 1.0000 | -0.1667 | (-0.5455, 0.0000) | 1.0000 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0190 | (-0.0038, 0.0442) | 0.0567 | 0.0190 | (0.0081, 0.0272) | 0.0003 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0060 | (-0.0071, 0.0208) | 0.2170 | 0.0060 | (-0.0098, 0.0228) | 0.2903 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 0.0077 | (-0.0105, 0.0261) | 0.2030 | 0.0077 | (-0.0079, 0.0218) | 0.1447 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.0015 | (-0.0644, 0.0610) | 0.4830 | 0.0015 | (-0.0447, 0.0477) | 0.4753 |
| controlled_vs_proposed_raw | context_overlap | -0.0029 | (-0.0252, 0.0196) | 0.6000 | -0.0029 | (-0.0217, 0.0200) | 0.6067 |
| controlled_vs_proposed_raw | persona_keyword_coverage | -0.0030 | (-0.0655, 0.0556) | 0.5383 | -0.0030 | (-0.1071, 0.0714) | 0.5433 |
| controlled_vs_proposed_raw | persona_style | -0.0365 | (-0.0768, -0.0013) | 0.9807 | -0.0365 | (-0.1080, 0.0000) | 1.0000 |
| controlled_vs_proposed_raw | distinct1 | -0.0028 | (-0.0241, 0.0175) | 0.5970 | -0.0028 | (-0.0153, 0.0133) | 0.6207 |
| controlled_vs_proposed_raw | length_score | -0.0236 | (-0.1111, 0.0694) | 0.6980 | -0.0236 | (-0.1273, 0.0464) | 0.6893 |
| controlled_vs_proposed_raw | sentence_score | 0.0437 | (-0.0146, 0.1021) | 0.1180 | 0.0437 | (0.0000, 0.0875) | 0.0830 |
| controlled_vs_proposed_raw | overall_quality | -0.0032 | (-0.0380, 0.0299) | 0.5747 | -0.0032 | (-0.0492, 0.0361) | 0.5820 |
| controlled_vs_candidate_no_context | context_relevance | 0.0437 | (0.0087, 0.0819) | 0.0053 | 0.0437 | (0.0165, 0.0631) | 0.0007 |
| controlled_vs_candidate_no_context | persona_consistency | -0.0012 | (-0.0828, 0.0673) | 0.5060 | -0.0012 | (-0.1448, 0.0952) | 0.4987 |
| controlled_vs_candidate_no_context | naturalness | -0.0012 | (-0.0239, 0.0245) | 0.5617 | -0.0012 | (-0.0354, 0.0356) | 0.5577 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.0398 | (0.0081, 0.0743) | 0.0057 | 0.0398 | (0.0168, 0.0567) | 0.0000 |
| controlled_vs_candidate_no_context | lore_consistency | 0.0134 | (-0.0025, 0.0329) | 0.0577 | 0.0134 | (0.0035, 0.0260) | 0.0043 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | -0.1667 | (-0.3333, -0.0417) | 1.0000 | -0.1667 | (-0.5455, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | 0.0390 | (0.0154, 0.0634) | 0.0000 | 0.0390 | (0.0092, 0.0627) | 0.0113 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0292 | (0.0107, 0.0471) | 0.0007 | 0.0292 | (-0.0060, 0.0586) | 0.0517 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 0.0185 | (0.0015, 0.0348) | 0.0190 | 0.0185 | (-0.0066, 0.0397) | 0.0817 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.0549 | (0.0095, 0.1027) | 0.0080 | 0.0549 | (0.0209, 0.0792) | 0.0003 |
| controlled_vs_candidate_no_context | context_overlap | 0.0175 | (0.0001, 0.0362) | 0.0233 | 0.0175 | (0.0061, 0.0266) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.0089 | (-0.0813, 0.0903) | 0.4117 | 0.0089 | (-0.1526, 0.1190) | 0.4273 |
| controlled_vs_candidate_no_context | persona_style | -0.0417 | (-0.1029, 0.0117) | 0.9383 | -0.0417 | (-0.1136, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | distinct1 | -0.0140 | (-0.0323, 0.0018) | 0.9537 | -0.0140 | (-0.0284, 0.0063) | 0.9147 |
| controlled_vs_candidate_no_context | length_score | -0.0000 | (-0.1056, 0.1070) | 0.5190 | -0.0000 | (-0.1752, 0.1821) | 0.5520 |
| controlled_vs_candidate_no_context | sentence_score | 0.0437 | (0.0000, 0.0875) | 0.0410 | 0.0437 | (0.0000, 0.0817) | 0.0727 |
| controlled_vs_candidate_no_context | overall_quality | 0.0201 | (-0.0204, 0.0592) | 0.1683 | 0.0201 | (-0.0448, 0.0697) | 0.2617 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0018 | (-0.0472, 0.0511) | 0.5330 | -0.0018 | (-0.0245, 0.0203) | 0.6190 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0372 | (-0.0272, 0.1089) | 0.1443 | 0.0372 | (-0.0111, 0.0915) | 0.0800 |
| controlled_alt_vs_controlled_default | naturalness | -0.0092 | (-0.0261, 0.0020) | 0.9137 | -0.0092 | (-0.0224, -0.0003) | 0.9880 |
| controlled_alt_vs_controlled_default | quest_state_correctness | -0.0039 | (-0.0454, 0.0402) | 0.5930 | -0.0039 | (-0.0274, 0.0175) | 0.6477 |
| controlled_alt_vs_controlled_default | lore_consistency | 0.0035 | (-0.0238, 0.0377) | 0.4407 | 0.0035 | (-0.0200, 0.0227) | 0.3770 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0.1667 | (0.0417, 0.3333) | 0.0127 | 0.1667 | (0.0000, 0.5455) | 0.3340 |
| controlled_alt_vs_controlled_default | objective_completion_support | -0.0137 | (-0.0434, 0.0158) | 0.8143 | -0.0137 | (-0.0345, 0.0049) | 0.9247 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | -0.0137 | (-0.0321, 0.0019) | 0.9470 | -0.0137 | (-0.0285, 0.0010) | 0.9703 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | -0.0194 | (-0.0448, 0.0031) | 0.9557 | -0.0194 | (-0.0443, 0.0031) | 0.9447 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0030 | (-0.0572, 0.0674) | 0.4933 | 0.0030 | (-0.0282, 0.0358) | 0.4747 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0132 | (-0.0328, 0.0062) | 0.9127 | -0.0132 | (-0.0218, -0.0047) | 1.0000 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0387 | (-0.0357, 0.1190) | 0.1623 | 0.0387 | (-0.0139, 0.0913) | 0.0917 |
| controlled_alt_vs_controlled_default | persona_style | 0.0312 | (-0.0208, 0.1042) | 0.2117 | 0.0312 | (0.0000, 0.1023) | 0.3257 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0066 | (-0.0222, 0.0074) | 0.7983 | -0.0066 | (-0.0120, -0.0010) | 0.9890 |
| controlled_alt_vs_controlled_default | length_score | -0.0403 | (-0.1167, 0.0139) | 0.8927 | -0.0403 | (-0.1103, 0.0100) | 0.8590 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0146 | (0.0000, 0.0437) | 0.3487 | 0.0146 | (0.0000, 0.0404) | 0.3307 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0112 | (-0.0283, 0.0492) | 0.2867 | 0.0112 | (-0.0089, 0.0320) | 0.1670 |
| controlled_alt_vs_proposed_raw | context_relevance | -0.0017 | (-0.0593, 0.0588) | 0.5123 | -0.0017 | (-0.0586, 0.0552) | 0.5727 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.0275 | (-0.0679, 0.1221) | 0.2753 | 0.0275 | (-0.0571, 0.1156) | 0.3110 |
| controlled_alt_vs_proposed_raw | naturalness | -0.0107 | (-0.0324, 0.0104) | 0.8437 | -0.0107 | (-0.0232, 0.0007) | 0.9670 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | -0.0020 | (-0.0505, 0.0495) | 0.5270 | -0.0020 | (-0.0479, 0.0477) | 0.5587 |
| controlled_alt_vs_proposed_raw | lore_consistency | -0.0023 | (-0.0389, 0.0309) | 0.5290 | -0.0023 | (-0.0368, 0.0315) | 0.6263 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_proposed_raw | objective_completion_support | 0.0053 | (-0.0262, 0.0357) | 0.3667 | 0.0053 | (-0.0133, 0.0260) | 0.3503 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | -0.0078 | (-0.0240, 0.0079) | 0.8303 | -0.0078 | (-0.0255, 0.0049) | 0.8703 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | -0.0117 | (-0.0358, 0.0111) | 0.8383 | -0.0117 | (-0.0270, 0.0043) | 0.9033 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.0045 | (-0.0682, 0.0811) | 0.4510 | 0.0045 | (-0.0736, 0.0786) | 0.4877 |
| controlled_alt_vs_proposed_raw | context_overlap | -0.0162 | (-0.0369, 0.0056) | 0.9253 | -0.0162 | (-0.0343, 0.0021) | 0.9607 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.0357 | (-0.0694, 0.1438) | 0.2703 | 0.0357 | (-0.0671, 0.1437) | 0.2943 |
| controlled_alt_vs_proposed_raw | persona_style | -0.0052 | (-0.0495, 0.0391) | 0.5947 | -0.0052 | (-0.0170, 0.0000) | 1.0000 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0094 | (-0.0284, 0.0096) | 0.8327 | -0.0094 | (-0.0260, 0.0111) | 0.8247 |
| controlled_alt_vs_proposed_raw | length_score | -0.0639 | (-0.1583, 0.0236) | 0.9213 | -0.0639 | (-0.1439, -0.0015) | 0.9837 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0583 | (0.0146, 0.1167) | 0.0143 | 0.0583 | (0.0000, 0.1000) | 0.0737 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.0080 | (-0.0450, 0.0612) | 0.3840 | 0.0080 | (-0.0395, 0.0679) | 0.3947 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.0418 | (0.0098, 0.0774) | 0.0033 | 0.0418 | (-0.0007, 0.0776) | 0.0457 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.0360 | (-0.0525, 0.1162) | 0.1943 | 0.0360 | (-0.0837, 0.1468) | 0.2987 |
| controlled_alt_vs_candidate_no_context | naturalness | -0.0104 | (-0.0317, 0.0109) | 0.8447 | -0.0104 | (-0.0336, 0.0134) | 0.8147 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 0.0359 | (0.0086, 0.0673) | 0.0047 | 0.0359 | (-0.0043, 0.0723) | 0.0653 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 0.0170 | (-0.0009, 0.0421) | 0.0397 | 0.0170 | (0.0000, 0.0341) | 0.0760 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_candidate_no_context | objective_completion_support | 0.0253 | (0.0009, 0.0508) | 0.0197 | 0.0253 | (-0.0022, 0.0494) | 0.0483 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 0.0155 | (-0.0056, 0.0363) | 0.0710 | 0.0155 | (-0.0167, 0.0385) | 0.1750 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | -0.0009 | (-0.0219, 0.0203) | 0.5383 | -0.0009 | (-0.0231, 0.0192) | 0.5767 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.0580 | (0.0159, 0.1064) | 0.0023 | 0.0580 | (0.0009, 0.1075) | 0.0080 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0042 | (-0.0082, 0.0184) | 0.2820 | 0.0042 | (-0.0088, 0.0122) | 0.2460 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.0476 | (-0.0476, 0.1438) | 0.1580 | 0.0476 | (-0.1024, 0.1905) | 0.2930 |
| controlled_alt_vs_candidate_no_context | persona_style | -0.0104 | (-0.0703, 0.0404) | 0.6420 | -0.0104 | (-0.0341, 0.0000) | 1.0000 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0206 | (-0.0368, -0.0059) | 0.9993 | -0.0206 | (-0.0391, 0.0033) | 0.9460 |
| controlled_alt_vs_candidate_no_context | length_score | -0.0403 | (-0.1486, 0.0667) | 0.7690 | -0.0403 | (-0.1850, 0.0782) | 0.7590 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.0583 | (0.0146, 0.1167) | 0.0190 | 0.0583 | (0.0000, 0.1250) | 0.0760 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.0313 | (-0.0095, 0.0722) | 0.0693 | 0.0313 | (-0.0386, 0.0935) | 0.2037 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 10 | 4 | 10 | 0.6250 | 0.7143 |
| proposed_vs_candidate_no_context | persona_consistency | 5 | 4 | 15 | 0.5208 | 0.5556 |
| proposed_vs_candidate_no_context | naturalness | 7 | 8 | 9 | 0.4792 | 0.4667 |
| proposed_vs_candidate_no_context | quest_state_correctness | 11 | 4 | 9 | 0.6458 | 0.7333 |
| proposed_vs_candidate_no_context | lore_consistency | 5 | 2 | 17 | 0.5625 | 0.7143 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| proposed_vs_candidate_no_context | objective_completion_support | 10 | 3 | 11 | 0.6458 | 0.7692 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 13 | 2 | 9 | 0.7292 | 0.8667 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | 8 | 5 | 11 | 0.5625 | 0.6154 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 7 | 3 | 14 | 0.5833 | 0.7000 |
| proposed_vs_candidate_no_context | context_overlap | 10 | 4 | 10 | 0.6250 | 0.7143 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 5 | 4 | 15 | 0.5208 | 0.5556 |
| proposed_vs_candidate_no_context | persona_style | 1 | 2 | 21 | 0.4792 | 0.3333 |
| proposed_vs_candidate_no_context | distinct1 | 5 | 8 | 11 | 0.4375 | 0.3846 |
| proposed_vs_candidate_no_context | length_score | 7 | 8 | 9 | 0.4792 | 0.4667 |
| proposed_vs_candidate_no_context | sentence_score | 2 | 2 | 20 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | overall_quality | 11 | 4 | 9 | 0.6458 | 0.7333 |
| controlled_vs_proposed_raw | context_relevance | 10 | 10 | 4 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | persona_consistency | 8 | 8 | 8 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | naturalness | 9 | 11 | 4 | 0.4583 | 0.4500 |
| controlled_vs_proposed_raw | quest_state_correctness | 10 | 10 | 4 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | lore_consistency | 5 | 6 | 13 | 0.4792 | 0.4545 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0 | 4 | 20 | 0.4167 | 0.0000 |
| controlled_vs_proposed_raw | objective_completion_support | 8 | 5 | 11 | 0.5625 | 0.6154 |
| controlled_vs_proposed_raw | gameplay_usefulness | 10 | 10 | 4 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 8 | 5 | 11 | 0.5625 | 0.6154 |
| controlled_vs_proposed_raw | context_keyword_coverage | 7 | 6 | 11 | 0.5208 | 0.5385 |
| controlled_vs_proposed_raw | context_overlap | 9 | 11 | 4 | 0.4583 | 0.4500 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 7 | 6 | 11 | 0.5208 | 0.5385 |
| controlled_vs_proposed_raw | persona_style | 1 | 6 | 17 | 0.3958 | 0.1429 |
| controlled_vs_proposed_raw | distinct1 | 11 | 9 | 4 | 0.5417 | 0.5500 |
| controlled_vs_proposed_raw | length_score | 6 | 12 | 6 | 0.3750 | 0.3333 |
| controlled_vs_proposed_raw | sentence_score | 4 | 1 | 19 | 0.5625 | 0.8000 |
| controlled_vs_proposed_raw | overall_quality | 10 | 10 | 4 | 0.5000 | 0.5000 |
| controlled_vs_candidate_no_context | context_relevance | 13 | 7 | 4 | 0.6250 | 0.6500 |
| controlled_vs_candidate_no_context | persona_consistency | 11 | 5 | 8 | 0.6250 | 0.6875 |
| controlled_vs_candidate_no_context | naturalness | 8 | 12 | 4 | 0.4167 | 0.4000 |
| controlled_vs_candidate_no_context | quest_state_correctness | 13 | 7 | 4 | 0.6250 | 0.6500 |
| controlled_vs_candidate_no_context | lore_consistency | 6 | 4 | 14 | 0.5417 | 0.6000 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 4 | 20 | 0.4167 | 0.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | 12 | 2 | 10 | 0.7083 | 0.8571 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 15 | 5 | 4 | 0.7083 | 0.7500 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 12 | 4 | 8 | 0.6667 | 0.7500 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 9 | 4 | 11 | 0.6042 | 0.6923 |
| controlled_vs_candidate_no_context | context_overlap | 12 | 7 | 5 | 0.6042 | 0.6316 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 9 | 5 | 10 | 0.5833 | 0.6429 |
| controlled_vs_candidate_no_context | persona_style | 2 | 6 | 16 | 0.4167 | 0.2500 |
| controlled_vs_candidate_no_context | distinct1 | 9 | 9 | 6 | 0.5000 | 0.5000 |
| controlled_vs_candidate_no_context | length_score | 9 | 11 | 4 | 0.4583 | 0.4500 |
| controlled_vs_candidate_no_context | sentence_score | 3 | 0 | 21 | 0.5625 | 1.0000 |
| controlled_vs_candidate_no_context | overall_quality | 14 | 6 | 4 | 0.6667 | 0.7000 |
| controlled_alt_vs_controlled_default | context_relevance | 4 | 10 | 10 | 0.3750 | 0.2857 |
| controlled_alt_vs_controlled_default | persona_consistency | 5 | 3 | 16 | 0.5417 | 0.6250 |
| controlled_alt_vs_controlled_default | naturalness | 6 | 8 | 10 | 0.4583 | 0.4286 |
| controlled_alt_vs_controlled_default | quest_state_correctness | 4 | 10 | 10 | 0.3750 | 0.2857 |
| controlled_alt_vs_controlled_default | lore_consistency | 3 | 5 | 16 | 0.4583 | 0.3750 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 4 | 0 | 20 | 0.5833 | 1.0000 |
| controlled_alt_vs_controlled_default | objective_completion_support | 4 | 7 | 13 | 0.4375 | 0.3636 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | 5 | 9 | 10 | 0.4167 | 0.3571 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | 3 | 8 | 13 | 0.3958 | 0.2727 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 3 | 7 | 14 | 0.4167 | 0.3000 |
| controlled_alt_vs_controlled_default | context_overlap | 4 | 10 | 10 | 0.3750 | 0.2857 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 5 | 3 | 16 | 0.5417 | 0.6250 |
| controlled_alt_vs_controlled_default | persona_style | 2 | 1 | 21 | 0.5208 | 0.6667 |
| controlled_alt_vs_controlled_default | distinct1 | 6 | 8 | 10 | 0.4583 | 0.4286 |
| controlled_alt_vs_controlled_default | length_score | 7 | 7 | 10 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | sentence_score | 1 | 0 | 23 | 0.5208 | 1.0000 |
| controlled_alt_vs_controlled_default | overall_quality | 6 | 8 | 10 | 0.4583 | 0.4286 |
| controlled_alt_vs_proposed_raw | context_relevance | 7 | 12 | 5 | 0.3958 | 0.3684 |
| controlled_alt_vs_proposed_raw | persona_consistency | 10 | 8 | 6 | 0.5417 | 0.5556 |
| controlled_alt_vs_proposed_raw | naturalness | 7 | 12 | 5 | 0.3958 | 0.3684 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 7 | 12 | 5 | 0.3958 | 0.3684 |
| controlled_alt_vs_proposed_raw | lore_consistency | 5 | 4 | 15 | 0.5208 | 0.5556 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_alt_vs_proposed_raw | objective_completion_support | 6 | 5 | 13 | 0.5208 | 0.5455 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 8 | 11 | 5 | 0.4375 | 0.4211 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | 5 | 10 | 9 | 0.3958 | 0.3333 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 7 | 7 | 10 | 0.5000 | 0.5000 |
| controlled_alt_vs_proposed_raw | context_overlap | 5 | 14 | 5 | 0.3125 | 0.2632 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 9 | 8 | 7 | 0.5208 | 0.5294 |
| controlled_alt_vs_proposed_raw | persona_style | 3 | 5 | 16 | 0.4583 | 0.3750 |
| controlled_alt_vs_proposed_raw | distinct1 | 9 | 9 | 6 | 0.5000 | 0.5000 |
| controlled_alt_vs_proposed_raw | length_score | 7 | 12 | 5 | 0.3958 | 0.3684 |
| controlled_alt_vs_proposed_raw | sentence_score | 4 | 0 | 20 | 0.5833 | 1.0000 |
| controlled_alt_vs_proposed_raw | overall_quality | 10 | 9 | 5 | 0.5208 | 0.5263 |
| controlled_alt_vs_candidate_no_context | context_relevance | 13 | 6 | 5 | 0.6458 | 0.6842 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 13 | 5 | 6 | 0.6667 | 0.7222 |
| controlled_alt_vs_candidate_no_context | naturalness | 6 | 12 | 6 | 0.3750 | 0.3333 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 13 | 6 | 5 | 0.6458 | 0.6842 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 5 | 2 | 17 | 0.5625 | 0.7143 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 24 | 0.5000 | nan |
| controlled_alt_vs_candidate_no_context | objective_completion_support | 9 | 4 | 11 | 0.6042 | 0.6923 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 11 | 8 | 5 | 0.5625 | 0.5789 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | 8 | 8 | 8 | 0.5000 | 0.5000 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 9 | 2 | 13 | 0.6458 | 0.8182 |
| controlled_alt_vs_candidate_no_context | context_overlap | 9 | 9 | 6 | 0.5000 | 0.5000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 11 | 5 | 8 | 0.6250 | 0.6875 |
| controlled_alt_vs_candidate_no_context | persona_style | 3 | 4 | 17 | 0.4792 | 0.4286 |
| controlled_alt_vs_candidate_no_context | distinct1 | 5 | 13 | 6 | 0.3333 | 0.2778 |
| controlled_alt_vs_candidate_no_context | length_score | 7 | 11 | 6 | 0.4167 | 0.3889 |
| controlled_alt_vs_candidate_no_context | sentence_score | 4 | 0 | 20 | 0.5833 | 1.0000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 14 | 5 | 5 | 0.6875 | 0.7368 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.1667 | 0.1250 | 0.8750 |
| proposed_contextual_controlled_tuned | 0.0000 | 0.0000 | 0.2083 | 0.2083 | 0.7917 |
| proposed_contextual | 0.0000 | 0.0000 | 0.5417 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.3750 | 0.0000 | 0.0000 |

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
| proposed_contextual_controlled | 0.2000 | 0.8000 | 1 | 5 |
| proposed_contextual_controlled_tuned | 0.0000 | 1.0000 | 0 | 5 |
| proposed_contextual | 0.0000 | 1.0000 | 0 | 5 |
| candidate_no_context | 0.0000 | 1.0000 | 0 | 5 |
- Detailed source-level values are published in `multi_turn_contradictions.json`.

- BERTScore status: unavailable (disabled_by_flag).

This report covers proposal RO5 metrics plus game-facing outcomes: quest-state correctness, lore consistency, contradiction safety, objective completion support, gameplay usefulness, and time-pressure acceptability.
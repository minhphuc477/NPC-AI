# Proposal Alignment Evaluation Report

- Run ID: `20260313T004936Z`
- Generated: `2026-03-13T00:52:44.934104+00:00`
- Scenarios: `artifacts\proposal_control_tuning\auto_tune\20260313T004330Z\train_runs\trial_002\seed_19\20260313T004936Z\scenarios.jsonl`
- Scenario count: `12`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_tuned`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.1474 (0.0771, 0.2317) | 0.3175 (0.2222, 0.4337) | 0.9032 (0.8772, 0.9315) | 0.3573 (0.2961, 0.4248) | n/a |
| proposed_contextual_controlled_tuned | 0.1407 (0.0770, 0.2245) | 0.3061 (0.2145, 0.4045) | 0.8663 (0.8446, 0.8885) | 0.3427 (0.2881, 0.4031) | n/a |
| proposed_contextual | 0.0752 (0.0538, 0.1007) | 0.2238 (0.1528, 0.3028) | 0.8895 (0.8679, 0.9133) | 0.2856 (0.2554, 0.3160) | n/a |
| candidate_no_context | 0.0420 (0.0244, 0.0590) | 0.2159 (0.1496, 0.2893) | 0.8569 (0.8446, 0.8684) | 0.2611 (0.2285, 0.2974) | n/a |

## Game-facing Outcome Metrics (mean, 95% CI)
| Arm | Quest-state Correctness | Lore Consistency | Contradiction Safety | Objective Completion Support | Gameplay Usefulness | Time-pressure Acceptability |
|---|---:|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2328 (0.1710, 0.3138) | 0.0523 (0.0000, 0.1081) | 1.0000 (1.0000, 1.0000) | 0.1104 (0.0616, 0.1591) | 0.3485 (0.3250, 0.3723) | 0.3370 (0.3082, 0.3679) |
| proposed_contextual_controlled_tuned | 0.2303 (0.1696, 0.2999) | 0.0630 (0.0164, 0.1262) | 1.0000 (1.0000, 1.0000) | 0.1590 (0.1201, 0.2010) | 0.3440 (0.3234, 0.3634) | 0.3436 (0.3161, 0.3752) |
| proposed_contextual | 0.1694 (0.1482, 0.1887) | 0.0242 (0.0000, 0.0662) | 1.0000 (1.0000, 1.0000) | 0.1273 (0.0698, 0.1906) | 0.3282 (0.3043, 0.3523) | 0.3328 (0.2966, 0.3725) |
| candidate_no_context | 0.1391 (0.1221, 0.1562) | 0.0117 (0.0024, 0.0250) | 1.0000 (1.0000, 1.0000) | 0.1167 (0.0640, 0.1734) | 0.2990 (0.2685, 0.3321) | 0.3259 (0.2930, 0.3633) |

- Multi-turn contradiction rate is reported as `1 - contradiction_safety` in row-level outputs.

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0332 | 0.7908 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0079 | 0.0368 |
| proposed_vs_candidate_no_context | naturalness | 0.0326 | 0.0380 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.0303 | 0.2175 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0125 | 1.0748 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | 0.0106 | 0.0904 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 0.0292 | 0.0977 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | 0.0069 | 0.0212 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0379 | 0.8333 |
| proposed_vs_candidate_no_context | context_overlap | 0.0223 | 0.6577 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0099 | 0.0962 |
| proposed_vs_candidate_no_context | persona_style | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | distinct1 | -0.0113 | -0.0119 |
| proposed_vs_candidate_no_context | length_score | 0.1417 | 0.3168 |
| proposed_vs_candidate_no_context | sentence_score | 0.0875 | 0.0991 |
| proposed_vs_candidate_no_context | overall_quality | 0.0245 | 0.0937 |
| controlled_vs_proposed_raw | context_relevance | 0.0722 | 0.9608 |
| controlled_vs_proposed_raw | persona_consistency | 0.0937 | 0.4184 |
| controlled_vs_proposed_raw | naturalness | 0.0137 | 0.0154 |
| controlled_vs_proposed_raw | quest_state_correctness | 0.0635 | 0.3748 |
| controlled_vs_proposed_raw | lore_consistency | 0.0281 | 1.1613 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_proposed_raw | objective_completion_support | -0.0169 | -0.1326 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0203 | 0.0617 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 0.0042 | 0.0126 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.0985 | 1.1818 |
| controlled_vs_proposed_raw | context_overlap | 0.0110 | 0.1963 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1171 | 1.0351 |
| controlled_vs_proposed_raw | persona_style | 0.0000 | 0.0000 |
| controlled_vs_proposed_raw | distinct1 | -0.0068 | -0.0072 |
| controlled_vs_proposed_raw | length_score | 0.0861 | 0.1462 |
| controlled_vs_proposed_raw | sentence_score | 0.0000 | 0.0000 |
| controlled_vs_proposed_raw | overall_quality | 0.0718 | 0.2513 |
| controlled_vs_candidate_no_context | context_relevance | 0.1055 | 2.5115 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1016 | 0.4706 |
| controlled_vs_candidate_no_context | naturalness | 0.0463 | 0.0540 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.0937 | 0.6737 |
| controlled_vs_candidate_no_context | lore_consistency | 0.0406 | 3.4842 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | -0.0063 | -0.0541 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0495 | 0.1654 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 0.0111 | 0.0340 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.1364 | 3.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0333 | 0.9831 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1270 | 1.2308 |
| controlled_vs_candidate_no_context | persona_style | 0.0000 | 0.0000 |
| controlled_vs_candidate_no_context | distinct1 | -0.0180 | -0.0190 |
| controlled_vs_candidate_no_context | length_score | 0.2278 | 0.5093 |
| controlled_vs_candidate_no_context | sentence_score | 0.0875 | 0.0991 |
| controlled_vs_candidate_no_context | overall_quality | 0.0962 | 0.3686 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0068 | -0.0460 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0114 | -0.0359 |
| controlled_alt_vs_controlled_default | naturalness | -0.0369 | -0.0409 |
| controlled_alt_vs_controlled_default | quest_state_correctness | -0.0025 | -0.0109 |
| controlled_alt_vs_controlled_default | lore_consistency | 0.0107 | 0.2055 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_controlled_default | objective_completion_support | 0.0485 | 0.4396 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | -0.0045 | -0.0130 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | 0.0066 | 0.0195 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0076 | -0.0417 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0050 | -0.0736 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0119 | -0.0517 |
| controlled_alt_vs_controlled_default | persona_style | -0.0094 | -0.0141 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0057 | -0.0062 |
| controlled_alt_vs_controlled_default | length_score | -0.1917 | -0.2840 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0292 | 0.0300 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0146 | -0.0408 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.0655 | 0.8706 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.0823 | 0.3675 |
| controlled_alt_vs_proposed_raw | naturalness | -0.0232 | -0.0261 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 0.0609 | 0.3597 |
| controlled_alt_vs_proposed_raw | lore_consistency | 0.0388 | 1.6055 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_proposed_raw | objective_completion_support | 0.0317 | 0.2488 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 0.0157 | 0.0480 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | 0.0107 | 0.0323 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.0909 | 1.0909 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0061 | 0.1082 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1052 | 0.9298 |
| controlled_alt_vs_proposed_raw | persona_style | -0.0094 | -0.0141 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0125 | -0.0133 |
| controlled_alt_vs_proposed_raw | length_score | -0.1056 | -0.1792 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0292 | 0.0300 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.0572 | 0.2003 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.0987 | 2.3498 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.0902 | 0.4178 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0094 | 0.0110 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 0.0912 | 0.6554 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 0.0513 | 4.4058 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_alt_vs_candidate_no_context | objective_completion_support | 0.0422 | 0.3617 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 0.0449 | 0.1503 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | 0.0176 | 0.0541 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.1288 | 2.8333 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0284 | 0.8371 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1151 | 1.1154 |
| controlled_alt_vs_candidate_no_context | persona_style | -0.0094 | -0.0141 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0238 | -0.0251 |
| controlled_alt_vs_candidate_no_context | length_score | 0.0361 | 0.0807 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1167 | 0.1321 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.0817 | 0.3128 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0332 | (0.0076, 0.0645) | 0.0007 | 0.0332 | (0.0116, 0.0548) | 0.0037 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0079 | (-0.0508, 0.0683) | 0.4273 | 0.0079 | (-0.0556, 0.0517) | 0.4123 |
| proposed_vs_candidate_no_context | naturalness | 0.0326 | (0.0096, 0.0600) | 0.0003 | 0.0326 | (0.0078, 0.0734) | 0.0030 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.0303 | (0.0085, 0.0547) | 0.0003 | 0.0303 | (0.0099, 0.0506) | 0.0027 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0125 | (-0.0155, 0.0551) | 0.3037 | 0.0125 | (-0.0094, 0.0519) | 0.3310 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | 0.0106 | (-0.0299, 0.0580) | 0.3120 | 0.0106 | (0.0000, 0.0181) | 0.0650 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 0.0292 | (0.0037, 0.0554) | 0.0083 | 0.0292 | (0.0091, 0.0524) | 0.0043 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | 0.0069 | (-0.0163, 0.0345) | 0.2987 | 0.0069 | (0.0000, 0.0118) | 0.0553 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0379 | (0.0076, 0.0758) | 0.0087 | 0.0379 | (0.0152, 0.0606) | 0.0033 |
| proposed_vs_candidate_no_context | context_overlap | 0.0223 | (0.0020, 0.0559) | 0.0010 | 0.0223 | (0.0019, 0.0522) | 0.0053 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0099 | (-0.0694, 0.0853) | 0.4470 | 0.0099 | (-0.0694, 0.0646) | 0.4187 |
| proposed_vs_candidate_no_context | persona_style | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | distinct1 | -0.0113 | (-0.0290, 0.0016) | 0.9183 | -0.0113 | (-0.0193, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | length_score | 0.1417 | (0.0416, 0.2611) | 0.0000 | 0.1417 | (0.0233, 0.3333) | 0.0067 |
| proposed_vs_candidate_no_context | sentence_score | 0.0875 | (-0.0292, 0.2042) | 0.1050 | 0.0875 | (0.0389, 0.1167) | 0.0037 |
| proposed_vs_candidate_no_context | overall_quality | 0.0245 | (-0.0055, 0.0562) | 0.0563 | 0.0245 | (0.0041, 0.0398) | 0.0030 |
| controlled_vs_proposed_raw | context_relevance | 0.0722 | (0.0028, 0.1565) | 0.0210 | 0.0722 | (0.0241, 0.1204) | 0.0063 |
| controlled_vs_proposed_raw | persona_consistency | 0.0937 | (0.0206, 0.1683) | 0.0047 | 0.0937 | (0.0444, 0.1224) | 0.0043 |
| controlled_vs_proposed_raw | naturalness | 0.0137 | (-0.0050, 0.0328) | 0.0763 | 0.0137 | (-0.0047, 0.0269) | 0.0637 |
| controlled_vs_proposed_raw | quest_state_correctness | 0.0635 | (0.0019, 0.1396) | 0.0170 | 0.0635 | (0.0202, 0.1067) | 0.0043 |
| controlled_vs_proposed_raw | lore_consistency | 0.0281 | (-0.0408, 0.1005) | 0.2253 | 0.0281 | (-0.0047, 0.0882) | 0.1920 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_proposed_raw | objective_completion_support | -0.0169 | (-0.0872, 0.0572) | 0.6753 | -0.0169 | (-0.0523, 0.0196) | 0.7923 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0203 | (-0.0103, 0.0515) | 0.1020 | 0.0203 | (-0.0018, 0.0423) | 0.0620 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 0.0042 | (-0.0340, 0.0414) | 0.4277 | 0.0042 | (-0.0166, 0.0278) | 0.3473 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.0985 | (0.0152, 0.2045) | 0.0110 | 0.0985 | (0.0303, 0.1667) | 0.0057 |
| controlled_vs_proposed_raw | context_overlap | 0.0110 | (-0.0365, 0.0529) | 0.2963 | 0.0110 | (-0.0044, 0.0262) | 0.0833 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1171 | (0.0278, 0.2103) | 0.0013 | 0.1171 | (0.0556, 0.1531) | 0.0037 |
| controlled_vs_proposed_raw | persona_style | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_proposed_raw | distinct1 | -0.0068 | (-0.0328, 0.0196) | 0.6990 | -0.0068 | (-0.0191, -0.0002) | 1.0000 |
| controlled_vs_proposed_raw | length_score | 0.0861 | (0.0000, 0.1806) | 0.0257 | 0.0861 | (0.0067, 0.1500) | 0.0030 |
| controlled_vs_proposed_raw | sentence_score | 0.0000 | (-0.0875, 0.0875) | 0.6350 | 0.0000 | (-0.0700, 0.0875) | 0.6230 |
| controlled_vs_proposed_raw | overall_quality | 0.0718 | (0.0182, 0.1302) | 0.0023 | 0.0718 | (0.0271, 0.1029) | 0.0037 |
| controlled_vs_candidate_no_context | context_relevance | 0.1055 | (0.0408, 0.1848) | 0.0000 | 0.1055 | (0.0357, 0.1752) | 0.0037 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1016 | (0.0286, 0.1762) | 0.0037 | 0.1016 | (0.0267, 0.1551) | 0.0033 |
| controlled_vs_candidate_no_context | naturalness | 0.0463 | (0.0174, 0.0781) | 0.0000 | 0.0463 | (0.0103, 0.0951) | 0.0043 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.0937 | (0.0344, 0.1655) | 0.0000 | 0.0937 | (0.0302, 0.1573) | 0.0033 |
| controlled_vs_candidate_no_context | lore_consistency | 0.0406 | (-0.0102, 0.1005) | 0.0717 | 0.0406 | (-0.0120, 0.0932) | 0.0590 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | -0.0063 | (-0.0595, 0.0396) | 0.5710 | -0.0063 | (-0.0345, 0.0242) | 0.6443 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0495 | (0.0139, 0.0835) | 0.0053 | 0.0495 | (0.0109, 0.0929) | 0.0037 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 0.0111 | (-0.0196, 0.0387) | 0.2287 | 0.0111 | (-0.0095, 0.0317) | 0.1850 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.1364 | (0.0530, 0.2348) | 0.0000 | 0.1364 | (0.0455, 0.2273) | 0.0023 |
| controlled_vs_candidate_no_context | context_overlap | 0.0333 | (0.0095, 0.0665) | 0.0003 | 0.0333 | (0.0129, 0.0538) | 0.0020 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1270 | (0.0397, 0.2202) | 0.0020 | 0.1270 | (0.0333, 0.1939) | 0.0033 |
| controlled_vs_candidate_no_context | persona_style | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | distinct1 | -0.0180 | (-0.0482, 0.0113) | 0.8853 | -0.0180 | (-0.0237, -0.0084) | 1.0000 |
| controlled_vs_candidate_no_context | length_score | 0.2278 | (0.0944, 0.3667) | 0.0000 | 0.2278 | (0.0467, 0.4273) | 0.0033 |
| controlled_vs_candidate_no_context | sentence_score | 0.0875 | (-0.0292, 0.2042) | 0.1207 | 0.0875 | (0.0000, 0.1909) | 0.0650 |
| controlled_vs_candidate_no_context | overall_quality | 0.0962 | (0.0495, 0.1494) | 0.0000 | 0.0962 | (0.0423, 0.1284) | 0.0030 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0068 | (-0.1109, 0.0919) | 0.5487 | -0.0068 | (-0.0601, 0.0456) | 0.5893 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0114 | (-0.1130, 0.0947) | 0.5993 | -0.0114 | (-0.0386, 0.0267) | 0.8190 |
| controlled_alt_vs_controlled_default | naturalness | -0.0369 | (-0.0615, -0.0114) | 1.0000 | -0.0369 | (-0.0522, -0.0155) | 1.0000 |
| controlled_alt_vs_controlled_default | quest_state_correctness | -0.0025 | (-0.0996, 0.0859) | 0.5090 | -0.0025 | (-0.0489, 0.0415) | 0.5770 |
| controlled_alt_vs_controlled_default | lore_consistency | 0.0107 | (-0.0656, 0.0934) | 0.4097 | 0.0107 | (-0.0384, 0.0645) | 0.3917 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | objective_completion_support | 0.0485 | (0.0189, 0.0859) | 0.0000 | 0.0485 | (0.0187, 0.0699) | 0.0040 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | -0.0045 | (-0.0280, 0.0139) | 0.6353 | -0.0045 | (-0.0208, 0.0151) | 0.6307 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | 0.0066 | (-0.0180, 0.0333) | 0.3267 | 0.0066 | (-0.0080, 0.0170) | 0.1960 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0076 | (-0.1439, 0.1136) | 0.5397 | -0.0076 | (-0.0758, 0.0606) | 0.5853 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0050 | (-0.0433, 0.0255) | 0.5907 | -0.0050 | (-0.0264, 0.0107) | 0.6500 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0119 | (-0.1409, 0.1210) | 0.5950 | -0.0119 | (-0.0455, 0.0333) | 0.7047 |
| controlled_alt_vs_controlled_default | persona_style | -0.0094 | (-0.0281, 0.0000) | 1.0000 | -0.0094 | (-0.0225, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0057 | (-0.0227, 0.0100) | 0.7500 | -0.0057 | (-0.0119, -0.0005) | 1.0000 |
| controlled_alt_vs_controlled_default | length_score | -0.1917 | (-0.3361, -0.0667) | 1.0000 | -0.1917 | (-0.2857, -0.0600) | 1.0000 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0292 | (0.0000, 0.0875) | 0.3470 | 0.0292 | (0.0000, 0.0700) | 0.3120 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0146 | (-0.0908, 0.0658) | 0.6410 | -0.0146 | (-0.0395, 0.0061) | 0.8690 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.0655 | (0.0026, 0.1499) | 0.0220 | 0.0655 | (0.0183, 0.0992) | 0.0020 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.0823 | (0.0222, 0.1569) | 0.0010 | 0.0823 | (0.0320, 0.1170) | 0.0043 |
| controlled_alt_vs_proposed_raw | naturalness | -0.0232 | (-0.0462, -0.0001) | 0.9753 | -0.0232 | (-0.0309, -0.0110) | 1.0000 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 0.0609 | (0.0033, 0.1303) | 0.0190 | 0.0609 | (0.0265, 0.0852) | 0.0023 |
| controlled_alt_vs_proposed_raw | lore_consistency | 0.0388 | (-0.0334, 0.1129) | 0.1473 | 0.0388 | (-0.0244, 0.0840) | 0.0950 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_proposed_raw | objective_completion_support | 0.0317 | (-0.0174, 0.0863) | 0.1087 | 0.0317 | (0.0089, 0.0544) | 0.0047 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 0.0157 | (-0.0060, 0.0433) | 0.0973 | 0.0157 | (0.0060, 0.0240) | 0.0043 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | 0.0107 | (-0.0311, 0.0493) | 0.3060 | 0.0107 | (-0.0068, 0.0295) | 0.1317 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.0909 | (0.0076, 0.1970) | 0.0127 | 0.0909 | (0.0364, 0.1299) | 0.0043 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0061 | (-0.0356, 0.0350) | 0.3413 | 0.0061 | (-0.0240, 0.0276) | 0.3757 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1052 | (0.0278, 0.2004) | 0.0003 | 0.1052 | (0.0429, 0.1484) | 0.0043 |
| controlled_alt_vs_proposed_raw | persona_style | -0.0094 | (-0.0281, 0.0000) | 1.0000 | -0.0094 | (-0.0225, 0.0000) | 1.0000 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0125 | (-0.0341, 0.0068) | 0.8817 | -0.0125 | (-0.0303, -0.0009) | 1.0000 |
| controlled_alt_vs_proposed_raw | length_score | -0.1056 | (-0.2167, 0.0111) | 0.9623 | -0.1056 | (-0.1472, -0.0500) | 1.0000 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0292 | (0.0000, 0.0875) | 0.3530 | 0.0292 | (0.0000, 0.0875) | 0.3197 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.0572 | (0.0110, 0.1082) | 0.0067 | 0.0572 | (0.0239, 0.0727) | 0.0040 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.0987 | (0.0360, 0.1729) | 0.0000 | 0.0987 | (0.0468, 0.1297) | 0.0023 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.0902 | (0.0079, 0.1782) | 0.0177 | 0.0902 | (-0.0000, 0.1546) | 0.0633 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0094 | (-0.0135, 0.0361) | 0.2300 | 0.0094 | (-0.0136, 0.0484) | 0.3047 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 0.0912 | (0.0363, 0.1578) | 0.0000 | 0.0912 | (0.0418, 0.1287) | 0.0063 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 0.0513 | (0.0100, 0.0996) | 0.0033 | 0.0513 | (0.0184, 0.0748) | 0.0033 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_candidate_no_context | objective_completion_support | 0.0422 | (0.0139, 0.0737) | 0.0010 | 0.0422 | (0.0208, 0.0636) | 0.0030 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 0.0449 | (0.0163, 0.0737) | 0.0003 | 0.0449 | (0.0141, 0.0754) | 0.0043 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | 0.0176 | (-0.0018, 0.0377) | 0.0377 | 0.0176 | (0.0001, 0.0407) | 0.0057 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.1288 | (0.0530, 0.2273) | 0.0000 | 0.1288 | (0.0606, 0.1688) | 0.0043 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0284 | (0.0113, 0.0465) | 0.0000 | 0.0284 | (0.0138, 0.0386) | 0.0030 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1151 | (0.0139, 0.2262) | 0.0113 | 0.1151 | (0.0000, 0.1973) | 0.0583 |
| controlled_alt_vs_candidate_no_context | persona_style | -0.0094 | (-0.0281, 0.0000) | 1.0000 | -0.0094 | (-0.0225, 0.0000) | 1.0000 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0238 | (-0.0474, -0.0015) | 0.9830 | -0.0238 | (-0.0343, -0.0106) | 1.0000 |
| controlled_alt_vs_candidate_no_context | length_score | 0.0361 | (-0.0639, 0.1444) | 0.2710 | 0.0361 | (-0.0571, 0.1861) | 0.3417 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1167 | (0.0292, 0.2042) | 0.0080 | 0.1167 | (0.0389, 0.1909) | 0.0053 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.0817 | (0.0299, 0.1392) | 0.0000 | 0.0817 | (0.0359, 0.1063) | 0.0047 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 6 | 0 | 6 | 0.7500 | 1.0000 |
| proposed_vs_candidate_no_context | persona_consistency | 2 | 1 | 9 | 0.5417 | 0.6667 |
| proposed_vs_candidate_no_context | naturalness | 6 | 0 | 6 | 0.7500 | 1.0000 |
| proposed_vs_candidate_no_context | quest_state_correctness | 6 | 0 | 6 | 0.7500 | 1.0000 |
| proposed_vs_candidate_no_context | lore_consistency | 3 | 3 | 6 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 12 | 0.5000 | nan |
| proposed_vs_candidate_no_context | objective_completion_support | 3 | 1 | 8 | 0.5833 | 0.7500 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 5 | 1 | 6 | 0.6667 | 0.8333 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | 2 | 2 | 8 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 4 | 0 | 8 | 0.6667 | 1.0000 |
| proposed_vs_candidate_no_context | context_overlap | 5 | 1 | 6 | 0.6667 | 0.8333 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 2 | 1 | 9 | 0.5417 | 0.6667 |
| proposed_vs_candidate_no_context | persona_style | 0 | 0 | 12 | 0.5000 | nan |
| proposed_vs_candidate_no_context | distinct1 | 1 | 2 | 9 | 0.4583 | 0.3333 |
| proposed_vs_candidate_no_context | length_score | 6 | 0 | 6 | 0.7500 | 1.0000 |
| proposed_vs_candidate_no_context | sentence_score | 4 | 1 | 7 | 0.6250 | 0.8000 |
| proposed_vs_candidate_no_context | overall_quality | 5 | 1 | 6 | 0.6667 | 0.8333 |
| controlled_vs_proposed_raw | context_relevance | 8 | 2 | 2 | 0.7500 | 0.8000 |
| controlled_vs_proposed_raw | persona_consistency | 6 | 1 | 5 | 0.7083 | 0.8571 |
| controlled_vs_proposed_raw | naturalness | 7 | 3 | 2 | 0.6667 | 0.7000 |
| controlled_vs_proposed_raw | quest_state_correctness | 7 | 3 | 2 | 0.6667 | 0.7000 |
| controlled_vs_proposed_raw | lore_consistency | 3 | 2 | 7 | 0.5417 | 0.6000 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0 | 0 | 12 | 0.5000 | nan |
| controlled_vs_proposed_raw | objective_completion_support | 2 | 6 | 4 | 0.3333 | 0.2500 |
| controlled_vs_proposed_raw | gameplay_usefulness | 6 | 4 | 2 | 0.5833 | 0.6000 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 5 | 3 | 4 | 0.5833 | 0.6250 |
| controlled_vs_proposed_raw | context_keyword_coverage | 5 | 1 | 6 | 0.6667 | 0.8333 |
| controlled_vs_proposed_raw | context_overlap | 8 | 2 | 2 | 0.7500 | 0.8000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 6 | 1 | 5 | 0.7083 | 0.8571 |
| controlled_vs_proposed_raw | persona_style | 0 | 0 | 12 | 0.5000 | nan |
| controlled_vs_proposed_raw | distinct1 | 5 | 4 | 3 | 0.5417 | 0.5556 |
| controlled_vs_proposed_raw | length_score | 7 | 3 | 2 | 0.6667 | 0.7000 |
| controlled_vs_proposed_raw | sentence_score | 1 | 1 | 10 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | overall_quality | 7 | 3 | 2 | 0.6667 | 0.7000 |
| controlled_vs_candidate_no_context | context_relevance | 9 | 1 | 2 | 0.8333 | 0.9000 |
| controlled_vs_candidate_no_context | persona_consistency | 7 | 1 | 4 | 0.7500 | 0.8750 |
| controlled_vs_candidate_no_context | naturalness | 9 | 1 | 2 | 0.8333 | 0.9000 |
| controlled_vs_candidate_no_context | quest_state_correctness | 9 | 1 | 2 | 0.8333 | 0.9000 |
| controlled_vs_candidate_no_context | lore_consistency | 3 | 3 | 6 | 0.5000 | 0.5000 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 12 | 0.5000 | nan |
| controlled_vs_candidate_no_context | objective_completion_support | 2 | 3 | 7 | 0.4583 | 0.4000 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 8 | 2 | 2 | 0.7500 | 0.8000 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 5 | 2 | 5 | 0.6250 | 0.7143 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 8 | 0 | 4 | 0.8333 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 8 | 2 | 2 | 0.7500 | 0.8000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 7 | 1 | 4 | 0.7500 | 0.8750 |
| controlled_vs_candidate_no_context | persona_style | 0 | 0 | 12 | 0.5000 | nan |
| controlled_vs_candidate_no_context | distinct1 | 4 | 5 | 3 | 0.4583 | 0.4444 |
| controlled_vs_candidate_no_context | length_score | 9 | 1 | 2 | 0.8333 | 0.9000 |
| controlled_vs_candidate_no_context | sentence_score | 4 | 1 | 7 | 0.6250 | 0.8000 |
| controlled_vs_candidate_no_context | overall_quality | 9 | 1 | 2 | 0.8333 | 0.9000 |
| controlled_alt_vs_controlled_default | context_relevance | 3 | 4 | 5 | 0.4583 | 0.4286 |
| controlled_alt_vs_controlled_default | persona_consistency | 2 | 3 | 7 | 0.4583 | 0.4000 |
| controlled_alt_vs_controlled_default | naturalness | 0 | 6 | 6 | 0.2500 | 0.0000 |
| controlled_alt_vs_controlled_default | quest_state_correctness | 3 | 4 | 5 | 0.4583 | 0.4286 |
| controlled_alt_vs_controlled_default | lore_consistency | 3 | 3 | 6 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0 | 0 | 12 | 0.5000 | nan |
| controlled_alt_vs_controlled_default | objective_completion_support | 7 | 0 | 5 | 0.7917 | 1.0000 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | 4 | 3 | 5 | 0.5417 | 0.5714 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | 4 | 3 | 5 | 0.5417 | 0.5714 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 3 | 4 | 5 | 0.4583 | 0.4286 |
| controlled_alt_vs_controlled_default | context_overlap | 5 | 2 | 5 | 0.6250 | 0.7143 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 2 | 3 | 7 | 0.4583 | 0.4000 |
| controlled_alt_vs_controlled_default | persona_style | 0 | 1 | 11 | 0.4583 | 0.0000 |
| controlled_alt_vs_controlled_default | distinct1 | 2 | 4 | 6 | 0.4167 | 0.3333 |
| controlled_alt_vs_controlled_default | length_score | 0 | 5 | 7 | 0.2917 | 0.0000 |
| controlled_alt_vs_controlled_default | sentence_score | 1 | 0 | 11 | 0.5417 | 1.0000 |
| controlled_alt_vs_controlled_default | overall_quality | 2 | 5 | 5 | 0.3750 | 0.2857 |
| controlled_alt_vs_proposed_raw | context_relevance | 8 | 2 | 2 | 0.7500 | 0.8000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 5 | 0 | 7 | 0.7083 | 1.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 2 | 8 | 2 | 0.2500 | 0.2000 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 8 | 2 | 2 | 0.7500 | 0.8000 |
| controlled_alt_vs_proposed_raw | lore_consistency | 5 | 2 | 5 | 0.6250 | 0.7143 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0 | 0 | 12 | 0.5000 | nan |
| controlled_alt_vs_proposed_raw | objective_completion_support | 6 | 3 | 3 | 0.6250 | 0.6667 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 6 | 4 | 2 | 0.5833 | 0.6000 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | 5 | 4 | 3 | 0.5417 | 0.5556 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 5 | 1 | 6 | 0.6667 | 0.8333 |
| controlled_alt_vs_proposed_raw | context_overlap | 8 | 2 | 2 | 0.7500 | 0.8000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 5 | 0 | 7 | 0.7083 | 1.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 0 | 1 | 11 | 0.4583 | 0.0000 |
| controlled_alt_vs_proposed_raw | distinct1 | 6 | 4 | 2 | 0.5833 | 0.6000 |
| controlled_alt_vs_proposed_raw | length_score | 3 | 7 | 2 | 0.3333 | 0.3000 |
| controlled_alt_vs_proposed_raw | sentence_score | 1 | 0 | 11 | 0.5417 | 1.0000 |
| controlled_alt_vs_proposed_raw | overall_quality | 7 | 3 | 2 | 0.6667 | 0.7000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 9 | 1 | 2 | 0.8333 | 0.9000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 5 | 1 | 6 | 0.6667 | 0.8333 |
| controlled_alt_vs_candidate_no_context | naturalness | 6 | 4 | 2 | 0.5833 | 0.6000 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 9 | 1 | 2 | 0.8333 | 0.9000 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 5 | 1 | 6 | 0.6667 | 0.8333 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 12 | 0.5000 | nan |
| controlled_alt_vs_candidate_no_context | objective_completion_support | 7 | 2 | 3 | 0.7083 | 0.7778 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 8 | 2 | 2 | 0.7500 | 0.8000 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | 6 | 3 | 3 | 0.6250 | 0.6667 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 7 | 0 | 5 | 0.7917 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 8 | 2 | 2 | 0.7500 | 0.8000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 5 | 1 | 6 | 0.6667 | 0.8333 |
| controlled_alt_vs_candidate_no_context | persona_style | 0 | 1 | 11 | 0.4583 | 0.0000 |
| controlled_alt_vs_candidate_no_context | distinct1 | 4 | 6 | 2 | 0.4167 | 0.4000 |
| controlled_alt_vs_candidate_no_context | length_score | 6 | 4 | 2 | 0.5833 | 0.6000 |
| controlled_alt_vs_candidate_no_context | sentence_score | 4 | 0 | 8 | 0.6667 | 1.0000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 8 | 2 | 2 | 0.7500 | 0.8000 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.1667 | 0.2500 | 0.7500 |
| proposed_contextual_controlled_tuned | 0.0000 | 0.0000 | 0.1667 | 0.0000 | 1.0000 |
| proposed_contextual | 0.0000 | 0.0000 | 0.5833 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `4`
- Unique template signatures: `11`
- Template signature ratio: `0.9167`
- Effective sample size by source clustering: `3.79`
- Effective sample size by template-signature clustering: `10.29`
- Detailed diagnostics are published in `scenario_dependence.json`.

## Multi-turn Contradiction
| Arm | Contradiction Rate | Contradiction Safety | Contradicted Sources | Source Count |
|---|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 1.0000 | 0 | 4 |
| proposed_contextual_controlled_tuned | 0.0000 | 1.0000 | 0 | 4 |
| proposed_contextual | 0.0000 | 1.0000 | 0 | 4 |
| candidate_no_context | 0.0000 | 1.0000 | 0 | 4 |
- Detailed source-level values are published in `multi_turn_contradictions.json`.

- BERTScore status: unavailable (disabled_by_flag).

This report covers proposal RO5 metrics plus game-facing outcomes: quest-state correctness, lore consistency, contradiction safety, objective completion support, gameplay usefulness, and time-pressure acceptability.
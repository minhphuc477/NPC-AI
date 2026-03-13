# Proposal Alignment Evaluation Report

- Run ID: `20260312T213231Z`
- Generated: `2026-03-12T21:34:11.862211+00:00`
- Scenarios: `artifacts\proposal\20260312T213231Z\scenarios.jsonl`
- Scenario count: `144`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off
- `baseline_no_context`: model `phi3:mini`, dynamic_context=off, response_control=off
- `baseline_no_context_phi3_latest`: model `phi3:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.1268 (0.1052, 0.1490) | 0.2931 (0.2709, 0.3169) | 0.8537 (0.8452, 0.8618) | 0.3304 (0.3144, 0.3464) | n/a |
| proposed_contextual | 0.0850 (0.0667, 0.1042) | 0.2096 (0.1902, 0.2291) | 0.8615 (0.8548, 0.8684) | 0.2814 (0.2703, 0.2939) | n/a |
| candidate_no_context | 0.0286 (0.0227, 0.0346) | 0.2294 (0.2076, 0.2525) | 0.8719 (0.8641, 0.8800) | 0.2648 (0.2546, 0.2761) | n/a |
| baseline_no_context | 0.0363 (0.0292, 0.0433) | 0.1551 (0.1392, 0.1716) | 0.8954 (0.8851, 0.9056) | 0.2446 (0.2369, 0.2531) | n/a |
| baseline_no_context_phi3_latest | 0.0353 (0.0291, 0.0422) | 0.1595 (0.1435, 0.1754) | 0.8875 (0.8786, 0.8966) | 0.2444 (0.2373, 0.2518) | n/a |

## Game-facing Outcome Metrics (mean, 95% CI)
| Arm | Quest-state Correctness | Lore Consistency | Contradiction Safety | Objective Completion Support | Gameplay Usefulness | Time-pressure Acceptability |
|---|---:|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2093 (0.1885, 0.2283) | 0.0624 (0.0467, 0.0788) | 0.8750 (0.8194, 0.9236) | 0.0918 (0.0788, 0.1043) | 0.3054 (0.2989, 0.3122) | 0.2956 (0.2855, 0.3059) |
| proposed_contextual | 0.1728 (0.1569, 0.1899) | 0.0453 (0.0330, 0.0587) | 1.0000 (1.0000, 1.0000) | 0.0736 (0.0626, 0.0853) | 0.2909 (0.2840, 0.2982) | 0.2918 (0.2829, 0.3004) |
| candidate_no_context | 0.1249 (0.1195, 0.1306) | 0.0064 (0.0037, 0.0093) | 0.8741 (0.8182, 0.9231) | 0.0687 (0.0575, 0.0814) | 0.2775 (0.2714, 0.2844) | 0.2939 (0.2864, 0.3016) |
| baseline_no_context | 0.1321 (0.1255, 0.1393) | 0.0141 (0.0107, 0.0175) | 0.8750 (0.8194, 0.9236) | 0.0445 (0.0366, 0.0527) | 0.2786 (0.2715, 0.2866) | 0.2893 (0.2830, 0.2959) |
| baseline_no_context_phi3_latest | 0.1303 (0.1243, 0.1362) | 0.0142 (0.0102, 0.0186) | 0.8750 (0.8194, 0.9236) | 0.0449 (0.0367, 0.0533) | 0.2749 (0.2694, 0.2807) | 0.2883 (0.2824, 0.2946) |

- Multi-turn contradiction rate is reported as `1 - contradiction_safety` in row-level outputs.

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0564 | 1.9716 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0198 | -0.0862 |
| proposed_vs_candidate_no_context | naturalness | -0.0104 | -0.0119 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.0479 | 0.3837 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0389 | 6.1306 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0.1259 | 0.1440 |
| proposed_vs_candidate_no_context | objective_completion_support | 0.0050 | 0.0723 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 0.0134 | 0.0484 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | -0.0021 | -0.0072 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0736 | 2.7902 |
| proposed_vs_candidate_no_context | context_overlap | 0.0165 | 0.4857 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0266 | -0.1747 |
| proposed_vs_candidate_no_context | persona_style | 0.0076 | 0.0141 |
| proposed_vs_candidate_no_context | distinct1 | -0.0079 | -0.0084 |
| proposed_vs_candidate_no_context | length_score | -0.0347 | -0.0691 |
| proposed_vs_candidate_no_context | sentence_score | -0.0020 | -0.0021 |
| proposed_vs_candidate_no_context | overall_quality | 0.0166 | 0.0626 |
| proposed_vs_baseline_no_context | context_relevance | 0.0488 | 1.3451 |
| proposed_vs_baseline_no_context | persona_consistency | 0.0545 | 0.3517 |
| proposed_vs_baseline_no_context | naturalness | -0.0339 | -0.0378 |
| proposed_vs_baseline_no_context | quest_state_correctness | 0.0407 | 0.3078 |
| proposed_vs_baseline_no_context | lore_consistency | 0.0312 | 2.2160 |
| proposed_vs_baseline_no_context | multi_turn_contradiction_safety | 0.1250 | 0.1429 |
| proposed_vs_baseline_no_context | objective_completion_support | 0.0291 | 0.6541 |
| proposed_vs_baseline_no_context | gameplay_usefulness | 0.0123 | 0.0442 |
| proposed_vs_baseline_no_context | time_pressure_acceptability | 0.0025 | 0.0085 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0651 | 1.8689 |
| proposed_vs_baseline_no_context | context_overlap | 0.0107 | 0.2708 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 0.0728 | 1.3733 |
| proposed_vs_baseline_no_context | persona_style | -0.0183 | -0.0324 |
| proposed_vs_baseline_no_context | distinct1 | -0.0454 | -0.0463 |
| proposed_vs_baseline_no_context | length_score | -0.0986 | -0.1740 |
| proposed_vs_baseline_no_context | sentence_score | 0.0410 | 0.0457 |
| proposed_vs_baseline_no_context | overall_quality | 0.0367 | 0.1501 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 0.0497 | 1.4066 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | 0.0502 | 0.3146 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | -0.0259 | -0.0292 |
| proposed_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.0425 | 0.3266 |
| proposed_vs_baseline_no_context_phi3_latest | lore_consistency | 0.0311 | 2.1912 |
| proposed_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0.1250 | 0.1429 |
| proposed_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.0287 | 0.6403 |
| proposed_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 0.0160 | 0.0581 |
| proposed_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 0.0035 | 0.0121 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.0646 | 1.8270 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 0.0150 | 0.4253 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.0670 | 1.1396 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | -0.0170 | -0.0302 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | -0.0411 | -0.0421 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | -0.0685 | -0.1277 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | 0.0431 | 0.0482 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | 0.0369 | 0.1512 |
| controlled_vs_proposed_raw | context_relevance | 0.0418 | 0.4913 |
| controlled_vs_proposed_raw | persona_consistency | 0.0834 | 0.3980 |
| controlled_vs_proposed_raw | naturalness | -0.0078 | -0.0091 |
| controlled_vs_proposed_raw | quest_state_correctness | 0.0365 | 0.2110 |
| controlled_vs_proposed_raw | lore_consistency | 0.0171 | 0.3774 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | -0.1250 | -0.1250 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0182 | 0.2474 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0145 | 0.0498 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 0.0039 | 0.0132 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.0552 | 0.5520 |
| controlled_vs_proposed_raw | context_overlap | 0.0106 | 0.2103 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1019 | 0.8101 |
| controlled_vs_proposed_raw | persona_style | 0.0098 | 0.0180 |
| controlled_vs_proposed_raw | distinct1 | -0.0125 | -0.0133 |
| controlled_vs_proposed_raw | length_score | -0.0234 | -0.0500 |
| controlled_vs_proposed_raw | sentence_score | 0.0274 | 0.0293 |
| controlled_vs_proposed_raw | overall_quality | 0.0490 | 0.1743 |
| controlled_vs_candidate_no_context | context_relevance | 0.0982 | 3.4317 |
| controlled_vs_candidate_no_context | persona_consistency | 0.0637 | 0.2775 |
| controlled_vs_candidate_no_context | naturalness | -0.0182 | -0.0209 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.0844 | 0.6757 |
| controlled_vs_candidate_no_context | lore_consistency | 0.0560 | 8.8215 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0009 | 0.0010 |
| controlled_vs_candidate_no_context | objective_completion_support | 0.0232 | 0.3375 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0279 | 0.1005 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 0.0017 | 0.0059 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.1287 | 4.8825 |
| controlled_vs_candidate_no_context | context_overlap | 0.0271 | 0.7982 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.0752 | 0.4938 |
| controlled_vs_candidate_no_context | persona_style | 0.0174 | 0.0323 |
| controlled_vs_candidate_no_context | distinct1 | -0.0204 | -0.0216 |
| controlled_vs_candidate_no_context | length_score | -0.0581 | -0.1156 |
| controlled_vs_candidate_no_context | sentence_score | 0.0254 | 0.0271 |
| controlled_vs_candidate_no_context | overall_quality | 0.0656 | 0.2478 |
| controlled_vs_baseline_no_context | context_relevance | 0.0906 | 2.4973 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1380 | 0.8897 |
| controlled_vs_baseline_no_context | naturalness | -0.0417 | -0.0466 |
| controlled_vs_baseline_no_context | quest_state_correctness | 0.0771 | 0.5837 |
| controlled_vs_baseline_no_context | lore_consistency | 0.0483 | 3.4297 |
| controlled_vs_baseline_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_baseline_no_context | objective_completion_support | 0.0473 | 1.0633 |
| controlled_vs_baseline_no_context | gameplay_usefulness | 0.0268 | 0.0962 |
| controlled_vs_baseline_no_context | time_pressure_acceptability | 0.0063 | 0.0218 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.1202 | 3.4526 |
| controlled_vs_baseline_no_context | context_overlap | 0.0213 | 0.5381 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1746 | 3.2959 |
| controlled_vs_baseline_no_context | persona_style | -0.0084 | -0.0150 |
| controlled_vs_baseline_no_context | distinct1 | -0.0579 | -0.0590 |
| controlled_vs_baseline_no_context | length_score | -0.1220 | -0.2153 |
| controlled_vs_baseline_no_context | sentence_score | 0.0684 | 0.0764 |
| controlled_vs_baseline_no_context | overall_quality | 0.0858 | 0.3505 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.0915 | 2.5891 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1336 | 0.8379 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | -0.0338 | -0.0381 |
| controlled_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.0790 | 0.6065 |
| controlled_vs_baseline_no_context_phi3_latest | lore_consistency | 0.0482 | 3.3955 |
| controlled_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.0470 | 1.0460 |
| controlled_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 0.0305 | 0.1108 |
| controlled_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 0.0074 | 0.0255 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.1197 | 3.3876 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0256 | 0.7250 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1688 | 2.8728 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | -0.0072 | -0.0128 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0536 | -0.0549 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | -0.0919 | -0.1713 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.0705 | 0.0789 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.0860 | 0.3518 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.0906 | 2.4973 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1380 | 0.8897 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0417 | -0.0466 |
| proposed_contextual_controlled_vs_baseline_no_context | quest_state_correctness | 0.0771 | 0.5837 |
| proposed_contextual_controlled_vs_baseline_no_context | lore_consistency | 0.0483 | 3.4297 |
| proposed_contextual_controlled_vs_baseline_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | objective_completion_support | 0.0473 | 1.0633 |
| proposed_contextual_controlled_vs_baseline_no_context | gameplay_usefulness | 0.0268 | 0.0962 |
| proposed_contextual_controlled_vs_baseline_no_context | time_pressure_acceptability | 0.0063 | 0.0218 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.1202 | 3.4526 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0213 | 0.5381 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1746 | 3.2959 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | -0.0084 | -0.0150 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0579 | -0.0590 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | -0.1220 | -0.2153 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0684 | 0.0764 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.0858 | 0.3505 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.0915 | 2.5891 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1336 | 0.8379 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | -0.0338 | -0.0381 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.0790 | 0.6065 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | lore_consistency | 0.0482 | 3.3955 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.0470 | 1.0460 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 0.0305 | 0.1108 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 0.0074 | 0.0255 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.1197 | 3.3876 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0256 | 0.7250 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1688 | 2.8728 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | -0.0072 | -0.0128 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0536 | -0.0549 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | -0.0919 | -0.1713 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.0705 | 0.0789 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.0860 | 0.3518 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0556 | (0.0369, 0.0753) | 0.0000 | 0.0556 | (0.0273, 0.0877) | 0.0000 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0199 | (-0.0399, -0.0014) | 0.9837 | -0.0199 | (-0.0405, -0.0010) | 0.9800 |
| proposed_vs_candidate_no_context | naturalness | -0.0102 | (-0.0191, -0.0008) | 0.9843 | -0.0102 | (-0.0158, -0.0053) | 1.0000 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.0469 | (0.0317, 0.0629) | 0.0000 | 0.0469 | (0.0228, 0.0729) | 0.0000 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0392 | (0.0275, 0.0523) | 0.0000 | 0.0392 | (0.0184, 0.0619) | 0.0000 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0.1259 | (0.0769, 0.1818) | 0.0000 | 0.1259 | (0.0000, 0.3776) | 0.3417 |
| proposed_vs_candidate_no_context | objective_completion_support | 0.0055 | (-0.0046, 0.0159) | 0.1583 | 0.0055 | (-0.0025, 0.0131) | 0.0870 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 0.0136 | (0.0060, 0.0215) | 0.0000 | 0.0136 | (0.0058, 0.0221) | 0.0000 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | -0.0016 | (-0.0095, 0.0071) | 0.6333 | -0.0016 | (-0.0074, 0.0059) | 0.6953 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0729 | (0.0488, 0.0991) | 0.0000 | 0.0729 | (0.0369, 0.1149) | 0.0000 |
| proposed_vs_candidate_no_context | context_overlap | 0.0153 | (0.0097, 0.0212) | 0.0000 | 0.0153 | (0.0069, 0.0239) | 0.0000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0269 | (-0.0523, -0.0033) | 0.9863 | -0.0269 | (-0.0537, -0.0014) | 0.9820 |
| proposed_vs_candidate_no_context | persona_style | 0.0079 | (-0.0029, 0.0200) | 0.0817 | 0.0079 | (-0.0014, 0.0194) | 0.0680 |
| proposed_vs_candidate_no_context | distinct1 | -0.0078 | (-0.0154, -0.0003) | 0.9790 | -0.0078 | (-0.0158, -0.0016) | 0.9990 |
| proposed_vs_candidate_no_context | length_score | -0.0340 | (-0.0755, 0.0061) | 0.9530 | -0.0340 | (-0.0627, -0.0060) | 0.9933 |
| proposed_vs_candidate_no_context | sentence_score | -0.0024 | (-0.0294, 0.0245) | 0.6170 | -0.0024 | (-0.0292, 0.0222) | 0.6053 |
| proposed_vs_candidate_no_context | overall_quality | 0.0162 | (0.0044, 0.0276) | 0.0040 | 0.0162 | (0.0021, 0.0318) | 0.0107 |
| proposed_vs_baseline_no_context | context_relevance | 0.0488 | (0.0318, 0.0675) | 0.0000 | 0.0488 | (0.0168, 0.0846) | 0.0003 |
| proposed_vs_baseline_no_context | persona_consistency | 0.0545 | (0.0340, 0.0742) | 0.0000 | 0.0545 | (0.0319, 0.0725) | 0.0000 |
| proposed_vs_baseline_no_context | naturalness | -0.0339 | (-0.0452, -0.0219) | 1.0000 | -0.0339 | (-0.0464, -0.0234) | 1.0000 |
| proposed_vs_baseline_no_context | quest_state_correctness | 0.0407 | (0.0254, 0.0558) | 0.0000 | 0.0407 | (0.0150, 0.0691) | 0.0007 |
| proposed_vs_baseline_no_context | lore_consistency | 0.0312 | (0.0196, 0.0443) | 0.0000 | 0.0312 | (0.0106, 0.0546) | 0.0007 |
| proposed_vs_baseline_no_context | multi_turn_contradiction_safety | 0.1250 | (0.0694, 0.1806) | 0.0000 | 0.1250 | (0.0000, 0.3750) | 0.3620 |
| proposed_vs_baseline_no_context | objective_completion_support | 0.0291 | (0.0161, 0.0420) | 0.0000 | 0.0291 | (0.0082, 0.0488) | 0.0033 |
| proposed_vs_baseline_no_context | gameplay_usefulness | 0.0123 | (0.0023, 0.0219) | 0.0087 | 0.0123 | (0.0019, 0.0216) | 0.0083 |
| proposed_vs_baseline_no_context | time_pressure_acceptability | 0.0025 | (-0.0079, 0.0122) | 0.3257 | 0.0025 | (-0.0115, 0.0159) | 0.3623 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0651 | (0.0429, 0.0896) | 0.0000 | 0.0651 | (0.0241, 0.1110) | 0.0000 |
| proposed_vs_baseline_no_context | context_overlap | 0.0107 | (0.0043, 0.0178) | 0.0000 | 0.0107 | (-0.0032, 0.0226) | 0.0663 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 0.0728 | (0.0475, 0.0980) | 0.0000 | 0.0728 | (0.0407, 0.0998) | 0.0000 |
| proposed_vs_baseline_no_context | persona_style | -0.0183 | (-0.0350, -0.0023) | 0.9887 | -0.0183 | (-0.0608, 0.0127) | 0.7803 |
| proposed_vs_baseline_no_context | distinct1 | -0.0454 | (-0.0546, -0.0366) | 1.0000 | -0.0454 | (-0.0633, -0.0328) | 1.0000 |
| proposed_vs_baseline_no_context | length_score | -0.0986 | (-0.1502, -0.0495) | 0.9997 | -0.0986 | (-0.1505, -0.0528) | 1.0000 |
| proposed_vs_baseline_no_context | sentence_score | 0.0410 | (0.0069, 0.0753) | 0.0123 | 0.0410 | (-0.0122, 0.0958) | 0.0737 |
| proposed_vs_baseline_no_context | overall_quality | 0.0367 | (0.0254, 0.0489) | 0.0000 | 0.0367 | (0.0172, 0.0554) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 0.0497 | (0.0325, 0.0696) | 0.0000 | 0.0497 | (0.0162, 0.0848) | 0.0003 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | 0.0502 | (0.0302, 0.0707) | 0.0000 | 0.0502 | (0.0231, 0.0756) | 0.0003 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | -0.0259 | (-0.0372, -0.0148) | 1.0000 | -0.0259 | (-0.0416, -0.0129) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.0425 | (0.0277, 0.0581) | 0.0000 | 0.0425 | (0.0156, 0.0695) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | lore_consistency | 0.0311 | (0.0179, 0.0446) | 0.0000 | 0.0311 | (0.0080, 0.0580) | 0.0040 |
| proposed_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0.1250 | (0.0764, 0.1806) | 0.0000 | 0.1250 | (0.0000, 0.3750) | 0.3337 |
| proposed_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.0287 | (0.0165, 0.0413) | 0.0000 | 0.0287 | (0.0111, 0.0482) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 0.0160 | (0.0066, 0.0250) | 0.0000 | 0.0160 | (0.0051, 0.0247) | 0.0030 |
| proposed_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 0.0035 | (-0.0056, 0.0132) | 0.2273 | 0.0035 | (-0.0065, 0.0140) | 0.2760 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.0646 | (0.0409, 0.0887) | 0.0000 | 0.0646 | (0.0224, 0.1125) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 0.0150 | (0.0081, 0.0222) | 0.0000 | 0.0150 | (0.0023, 0.0267) | 0.0100 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.0670 | (0.0425, 0.0912) | 0.0000 | 0.0670 | (0.0336, 0.0946) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | -0.0170 | (-0.0349, 0.0005) | 0.9700 | -0.0170 | (-0.0656, 0.0181) | 0.7600 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | -0.0411 | (-0.0508, -0.0317) | 1.0000 | -0.0411 | (-0.0577, -0.0273) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | -0.0685 | (-0.1162, -0.0220) | 0.9970 | -0.0685 | (-0.1238, -0.0197) | 0.9987 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | 0.0431 | (0.0090, 0.0771) | 0.0077 | 0.0431 | (-0.0049, 0.0917) | 0.0437 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | 0.0369 | (0.0245, 0.0491) | 0.0000 | 0.0369 | (0.0153, 0.0567) | 0.0003 |
| controlled_vs_proposed_raw | context_relevance | 0.0418 | (0.0196, 0.0646) | 0.0003 | 0.0418 | (0.0216, 0.0626) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.0834 | (0.0607, 0.1076) | 0.0000 | 0.0834 | (0.0555, 0.1160) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | -0.0078 | (-0.0187, 0.0031) | 0.9093 | -0.0078 | (-0.0231, 0.0090) | 0.8253 |
| controlled_vs_proposed_raw | quest_state_correctness | 0.0365 | (0.0181, 0.0561) | 0.0000 | 0.0365 | (0.0199, 0.0543) | 0.0000 |
| controlled_vs_proposed_raw | lore_consistency | 0.0171 | (0.0002, 0.0349) | 0.0233 | 0.0171 | (0.0024, 0.0328) | 0.0067 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | -0.1250 | (-0.1806, -0.0764) | 1.0000 | -0.1250 | (-0.3750, 0.0000) | 1.0000 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0182 | (0.0056, 0.0307) | 0.0013 | 0.0182 | (0.0068, 0.0309) | 0.0003 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0145 | (0.0068, 0.0221) | 0.0003 | 0.0145 | (0.0050, 0.0231) | 0.0020 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 0.0039 | (-0.0066, 0.0138) | 0.2303 | 0.0039 | (-0.0023, 0.0108) | 0.1200 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.0552 | (0.0251, 0.0857) | 0.0003 | 0.0552 | (0.0290, 0.0837) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0106 | (0.0021, 0.0189) | 0.0067 | 0.0106 | (0.0040, 0.0169) | 0.0013 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1019 | (0.0738, 0.1325) | 0.0000 | 0.1019 | (0.0668, 0.1424) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.0098 | (-0.0039, 0.0249) | 0.0887 | 0.0098 | (-0.0120, 0.0382) | 0.2467 |
| controlled_vs_proposed_raw | distinct1 | -0.0125 | (-0.0216, -0.0032) | 0.9973 | -0.0125 | (-0.0250, -0.0008) | 0.9827 |
| controlled_vs_proposed_raw | length_score | -0.0234 | (-0.0683, 0.0204) | 0.8453 | -0.0234 | (-0.0910, 0.0456) | 0.7407 |
| controlled_vs_proposed_raw | sentence_score | 0.0274 | (-0.0035, 0.0566) | 0.0373 | 0.0274 | (-0.0087, 0.0611) | 0.0667 |
| controlled_vs_proposed_raw | overall_quality | 0.0490 | (0.0348, 0.0643) | 0.0000 | 0.0490 | (0.0366, 0.0612) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.0990 | (0.0763, 0.1221) | 0.0000 | 0.0990 | (0.0659, 0.1351) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.0641 | (0.0398, 0.0880) | 0.0000 | 0.0641 | (0.0285, 0.0904) | 0.0007 |
| controlled_vs_candidate_no_context | naturalness | -0.0183 | (-0.0296, -0.0072) | 0.9993 | -0.0183 | (-0.0331, -0.0028) | 0.9893 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.0851 | (0.0660, 0.1046) | 0.0000 | 0.0851 | (0.0576, 0.1143) | 0.0000 |
| controlled_vs_candidate_no_context | lore_consistency | 0.0565 | (0.0419, 0.0732) | 0.0000 | 0.0565 | (0.0371, 0.0751) | 0.0000 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (-0.0839, 0.0769) | 0.5417 | 0.0000 | (-0.3750, 0.3750) | 0.6337 |
| controlled_vs_candidate_no_context | objective_completion_support | 0.0227 | (0.0106, 0.0346) | 0.0000 | 0.0227 | (0.0128, 0.0328) | 0.0000 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0278 | (0.0205, 0.0351) | 0.0000 | 0.0278 | (0.0196, 0.0371) | 0.0000 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 0.0013 | (-0.0079, 0.0096) | 0.4000 | 0.0013 | (-0.0069, 0.0102) | 0.4040 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.1298 | (0.1003, 0.1609) | 0.0000 | 0.1298 | (0.0852, 0.1791) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0272 | (0.0198, 0.0351) | 0.0000 | 0.0272 | (0.0204, 0.0344) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.0757 | (0.0456, 0.1063) | 0.0000 | 0.0757 | (0.0359, 0.1079) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.0178 | (0.0033, 0.0346) | 0.0077 | 0.0178 | (0.0015, 0.0393) | 0.0210 |
| controlled_vs_candidate_no_context | distinct1 | -0.0201 | (-0.0294, -0.0116) | 1.0000 | -0.0201 | (-0.0358, -0.0049) | 0.9963 |
| controlled_vs_candidate_no_context | length_score | -0.0590 | (-0.1077, -0.0114) | 0.9913 | -0.0590 | (-0.1326, 0.0083) | 0.9533 |
| controlled_vs_candidate_no_context | sentence_score | 0.0252 | (-0.0035, 0.0525) | 0.0367 | 0.0252 | (-0.0111, 0.0660) | 0.0973 |
| controlled_vs_candidate_no_context | overall_quality | 0.0661 | (0.0504, 0.0816) | 0.0000 | 0.0661 | (0.0435, 0.0894) | 0.0000 |
| controlled_vs_baseline_no_context | context_relevance | 0.0906 | (0.0691, 0.1136) | 0.0000 | 0.0906 | (0.0546, 0.1323) | 0.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1380 | (0.1154, 0.1592) | 0.0000 | 0.1380 | (0.0985, 0.1701) | 0.0000 |
| controlled_vs_baseline_no_context | naturalness | -0.0417 | (-0.0551, -0.0291) | 1.0000 | -0.0417 | (-0.0557, -0.0260) | 1.0000 |
| controlled_vs_baseline_no_context | quest_state_correctness | 0.0771 | (0.0581, 0.0964) | 0.0000 | 0.0771 | (0.0465, 0.1109) | 0.0000 |
| controlled_vs_baseline_no_context | lore_consistency | 0.0483 | (0.0319, 0.0651) | 0.0000 | 0.0483 | (0.0296, 0.0689) | 0.0000 |
| controlled_vs_baseline_no_context | multi_turn_contradiction_safety | 0.0000 | (-0.0764, 0.0833) | 0.5243 | 0.0000 | (-0.3750, 0.3750) | 0.6420 |
| controlled_vs_baseline_no_context | objective_completion_support | 0.0473 | (0.0341, 0.0605) | 0.0000 | 0.0473 | (0.0266, 0.0684) | 0.0000 |
| controlled_vs_baseline_no_context | gameplay_usefulness | 0.0268 | (0.0181, 0.0355) | 0.0000 | 0.0268 | (0.0194, 0.0349) | 0.0000 |
| controlled_vs_baseline_no_context | time_pressure_acceptability | 0.0063 | (-0.0040, 0.0163) | 0.1077 | 0.0063 | (-0.0077, 0.0231) | 0.2267 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.1202 | (0.0917, 0.1517) | 0.0000 | 0.1202 | (0.0725, 0.1737) | 0.0000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0213 | (0.0133, 0.0297) | 0.0000 | 0.0213 | (0.0098, 0.0313) | 0.0003 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1746 | (0.1472, 0.2021) | 0.0000 | 0.1746 | (0.1251, 0.2132) | 0.0000 |
| controlled_vs_baseline_no_context | persona_style | -0.0084 | (-0.0234, 0.0059) | 0.8760 | -0.0084 | (-0.0290, 0.0100) | 0.7943 |
| controlled_vs_baseline_no_context | distinct1 | -0.0579 | (-0.0673, -0.0487) | 1.0000 | -0.0579 | (-0.0719, -0.0466) | 1.0000 |
| controlled_vs_baseline_no_context | length_score | -0.1220 | (-0.1766, -0.0692) | 1.0000 | -0.1220 | (-0.1928, -0.0542) | 1.0000 |
| controlled_vs_baseline_no_context | sentence_score | 0.0684 | (0.0361, 0.1017) | 0.0000 | 0.0684 | (0.0382, 0.1010) | 0.0000 |
| controlled_vs_baseline_no_context | overall_quality | 0.0858 | (0.0710, 0.1006) | 0.0000 | 0.0858 | (0.0590, 0.1130) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.0915 | (0.0687, 0.1140) | 0.0000 | 0.0915 | (0.0565, 0.1351) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1336 | (0.1128, 0.1549) | 0.0000 | 0.1336 | (0.0931, 0.1694) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | -0.0338 | (-0.0445, -0.0231) | 1.0000 | -0.0338 | (-0.0451, -0.0189) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.0790 | (0.0593, 0.0990) | 0.0000 | 0.0790 | (0.0481, 0.1119) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | lore_consistency | 0.0482 | (0.0331, 0.0649) | 0.0000 | 0.0482 | (0.0270, 0.0705) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0.0000 | (-0.0764, 0.0833) | 0.5193 | 0.0000 | (-0.3750, 0.3750) | 0.6440 |
| controlled_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.0470 | (0.0334, 0.0603) | 0.0000 | 0.0470 | (0.0267, 0.0686) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 0.0305 | (0.0225, 0.0388) | 0.0000 | 0.0305 | (0.0218, 0.0405) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 0.0074 | (-0.0023, 0.0176) | 0.0690 | 0.0074 | (-0.0041, 0.0217) | 0.1350 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.1197 | (0.0905, 0.1513) | 0.0000 | 0.1197 | (0.0725, 0.1741) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0256 | (0.0172, 0.0339) | 0.0000 | 0.0256 | (0.0145, 0.0356) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1688 | (0.1418, 0.1946) | 0.0000 | 0.1688 | (0.1173, 0.2097) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | -0.0072 | (-0.0242, 0.0099) | 0.7973 | -0.0072 | (-0.0310, 0.0146) | 0.7210 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0536 | (-0.0627, -0.0445) | 1.0000 | -0.0536 | (-0.0658, -0.0431) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | -0.0919 | (-0.1375, -0.0479) | 0.9997 | -0.0919 | (-0.1477, -0.0317) | 0.9983 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.0705 | (0.0372, 0.1042) | 0.0003 | 0.0705 | (0.0299, 0.1139) | 0.0003 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.0860 | (0.0709, 0.1011) | 0.0000 | 0.0860 | (0.0601, 0.1110) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.0906 | (0.0692, 0.1135) | 0.0000 | 0.0906 | (0.0539, 0.1305) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1380 | (0.1159, 0.1598) | 0.0000 | 0.1380 | (0.1009, 0.1708) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0417 | (-0.0546, -0.0288) | 1.0000 | -0.0417 | (-0.0562, -0.0260) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | quest_state_correctness | 0.0771 | (0.0578, 0.0966) | 0.0000 | 0.0771 | (0.0475, 0.1114) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | lore_consistency | 0.0483 | (0.0328, 0.0641) | 0.0000 | 0.0483 | (0.0281, 0.0691) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | multi_turn_contradiction_safety | 0.0000 | (-0.0833, 0.0833) | 0.5417 | 0.0000 | (-0.3750, 0.3750) | 0.6510 |
| proposed_contextual_controlled_vs_baseline_no_context | objective_completion_support | 0.0473 | (0.0345, 0.0601) | 0.0000 | 0.0473 | (0.0281, 0.0672) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | gameplay_usefulness | 0.0268 | (0.0178, 0.0359) | 0.0000 | 0.0268 | (0.0196, 0.0354) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | time_pressure_acceptability | 0.0063 | (-0.0039, 0.0167) | 0.1223 | 0.0063 | (-0.0081, 0.0232) | 0.2370 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.1202 | (0.0917, 0.1496) | 0.0000 | 0.1202 | (0.0727, 0.1735) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0213 | (0.0134, 0.0296) | 0.0000 | 0.0213 | (0.0097, 0.0312) | 0.0007 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1746 | (0.1478, 0.2006) | 0.0000 | 0.1746 | (0.1267, 0.2141) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | -0.0084 | (-0.0232, 0.0064) | 0.8793 | -0.0084 | (-0.0293, 0.0098) | 0.7973 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0579 | (-0.0672, -0.0490) | 1.0000 | -0.0579 | (-0.0713, -0.0464) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | -0.1220 | (-0.1766, -0.0680) | 1.0000 | -0.1220 | (-0.1917, -0.0530) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0684 | (0.0368, 0.1017) | 0.0000 | 0.0684 | (0.0378, 0.0993) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.0858 | (0.0711, 0.1010) | 0.0000 | 0.0858 | (0.0581, 0.1110) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.0915 | (0.0687, 0.1154) | 0.0000 | 0.0915 | (0.0554, 0.1316) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1336 | (0.1130, 0.1550) | 0.0000 | 0.1336 | (0.0942, 0.1696) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | -0.0338 | (-0.0445, -0.0223) | 1.0000 | -0.0338 | (-0.0455, -0.0196) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | quest_state_correctness | 0.0790 | (0.0590, 0.0983) | 0.0000 | 0.0790 | (0.0487, 0.1119) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | lore_consistency | 0.0482 | (0.0324, 0.0654) | 0.0000 | 0.0482 | (0.0269, 0.0701) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 0.0000 | (-0.0833, 0.0833) | 0.5417 | 0.0000 | (-0.3750, 0.3750) | 0.6373 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | objective_completion_support | 0.0470 | (0.0341, 0.0598) | 0.0000 | 0.0470 | (0.0272, 0.0679) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 0.0305 | (0.0221, 0.0392) | 0.0000 | 0.0305 | (0.0220, 0.0397) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 0.0074 | (-0.0024, 0.0174) | 0.0680 | 0.0074 | (-0.0042, 0.0216) | 0.1407 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.1197 | (0.0897, 0.1503) | 0.0000 | 0.1197 | (0.0723, 0.1763) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0256 | (0.0175, 0.0342) | 0.0000 | 0.0256 | (0.0144, 0.0353) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1688 | (0.1414, 0.1949) | 0.0000 | 0.1688 | (0.1200, 0.2079) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | -0.0072 | (-0.0235, 0.0102) | 0.8020 | -0.0072 | (-0.0310, 0.0143) | 0.7113 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0536 | (-0.0629, -0.0440) | 1.0000 | -0.0536 | (-0.0648, -0.0434) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | -0.0919 | (-0.1352, -0.0479) | 1.0000 | -0.0919 | (-0.1498, -0.0319) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.0705 | (0.0372, 0.1045) | 0.0000 | 0.0705 | (0.0295, 0.1146) | 0.0003 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.0860 | (0.0711, 0.1011) | 0.0000 | 0.0860 | (0.0601, 0.1116) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 69 | 25 | 49 | 0.6538 | 0.7340 |
| proposed_vs_candidate_no_context | persona_consistency | 27 | 37 | 79 | 0.4650 | 0.4219 |
| proposed_vs_candidate_no_context | naturalness | 40 | 52 | 51 | 0.4580 | 0.4348 |
| proposed_vs_candidate_no_context | quest_state_correctness | 70 | 24 | 49 | 0.6608 | 0.7447 |
| proposed_vs_candidate_no_context | lore_consistency | 50 | 11 | 82 | 0.6364 | 0.8197 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 18 | 0 | 125 | 0.5629 | 1.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | 41 | 29 | 73 | 0.5420 | 0.5857 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 58 | 36 | 49 | 0.5769 | 0.6170 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | 40 | 43 | 60 | 0.4895 | 0.4819 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 51 | 11 | 81 | 0.6399 | 0.8226 |
| proposed_vs_candidate_no_context | context_overlap | 65 | 28 | 50 | 0.6294 | 0.6989 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 24 | 34 | 85 | 0.4650 | 0.4138 |
| proposed_vs_candidate_no_context | persona_style | 8 | 5 | 130 | 0.5105 | 0.6154 |
| proposed_vs_candidate_no_context | distinct1 | 31 | 45 | 67 | 0.4510 | 0.4079 |
| proposed_vs_candidate_no_context | length_score | 44 | 46 | 53 | 0.4930 | 0.4889 |
| proposed_vs_candidate_no_context | sentence_score | 15 | 16 | 112 | 0.4965 | 0.4839 |
| proposed_vs_candidate_no_context | overall_quality | 57 | 37 | 49 | 0.5699 | 0.6064 |
| proposed_vs_baseline_no_context | context_relevance | 78 | 63 | 3 | 0.5521 | 0.5532 |
| proposed_vs_baseline_no_context | persona_consistency | 67 | 22 | 55 | 0.6562 | 0.7528 |
| proposed_vs_baseline_no_context | naturalness | 47 | 96 | 1 | 0.3299 | 0.3287 |
| proposed_vs_baseline_no_context | quest_state_correctness | 79 | 62 | 3 | 0.5590 | 0.5603 |
| proposed_vs_baseline_no_context | lore_consistency | 50 | 38 | 56 | 0.5417 | 0.5682 |
| proposed_vs_baseline_no_context | multi_turn_contradiction_safety | 18 | 0 | 126 | 0.5625 | 1.0000 |
| proposed_vs_baseline_no_context | objective_completion_support | 72 | 40 | 32 | 0.6111 | 0.6429 |
| proposed_vs_baseline_no_context | gameplay_usefulness | 84 | 60 | 0 | 0.5833 | 0.5833 |
| proposed_vs_baseline_no_context | time_pressure_acceptability | 62 | 68 | 14 | 0.4792 | 0.4769 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 54 | 18 | 72 | 0.6250 | 0.7500 |
| proposed_vs_baseline_no_context | context_overlap | 77 | 63 | 4 | 0.5486 | 0.5500 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 64 | 15 | 65 | 0.6701 | 0.8101 |
| proposed_vs_baseline_no_context | persona_style | 7 | 19 | 118 | 0.4583 | 0.2692 |
| proposed_vs_baseline_no_context | distinct1 | 28 | 97 | 19 | 0.2604 | 0.2240 |
| proposed_vs_baseline_no_context | length_score | 56 | 83 | 5 | 0.4062 | 0.4029 |
| proposed_vs_baseline_no_context | sentence_score | 36 | 20 | 88 | 0.5556 | 0.6429 |
| proposed_vs_baseline_no_context | overall_quality | 101 | 43 | 0 | 0.7014 | 0.7014 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 85 | 58 | 1 | 0.5938 | 0.5944 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | 68 | 22 | 54 | 0.6597 | 0.7556 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | 56 | 88 | 0 | 0.3889 | 0.3889 |
| proposed_vs_baseline_no_context_phi3_latest | quest_state_correctness | 84 | 59 | 1 | 0.5868 | 0.5874 |
| proposed_vs_baseline_no_context_phi3_latest | lore_consistency | 49 | 38 | 57 | 0.5382 | 0.5632 |
| proposed_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 18 | 0 | 126 | 0.5625 | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | objective_completion_support | 71 | 37 | 36 | 0.6181 | 0.6574 |
| proposed_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 86 | 58 | 0 | 0.5972 | 0.5972 |
| proposed_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 61 | 68 | 15 | 0.4757 | 0.4729 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 55 | 20 | 69 | 0.6215 | 0.7333 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 84 | 59 | 1 | 0.5868 | 0.5874 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 63 | 16 | 65 | 0.6632 | 0.7975 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | 9 | 19 | 116 | 0.4653 | 0.3214 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | 31 | 94 | 19 | 0.2812 | 0.2480 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | 57 | 81 | 6 | 0.4167 | 0.4130 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | 32 | 16 | 96 | 0.5556 | 0.6667 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | 102 | 42 | 0 | 0.7083 | 0.7083 |
| controlled_vs_proposed_raw | context_relevance | 67 | 41 | 36 | 0.5903 | 0.6204 |
| controlled_vs_proposed_raw | persona_consistency | 69 | 8 | 67 | 0.7118 | 0.8961 |
| controlled_vs_proposed_raw | naturalness | 48 | 60 | 36 | 0.4583 | 0.4444 |
| controlled_vs_proposed_raw | quest_state_correctness | 66 | 42 | 36 | 0.5833 | 0.6111 |
| controlled_vs_proposed_raw | lore_consistency | 41 | 34 | 69 | 0.5243 | 0.5467 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0 | 18 | 126 | 0.4375 | 0.0000 |
| controlled_vs_proposed_raw | objective_completion_support | 52 | 34 | 58 | 0.5625 | 0.6047 |
| controlled_vs_proposed_raw | gameplay_usefulness | 69 | 39 | 36 | 0.6042 | 0.6389 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 41 | 52 | 51 | 0.4618 | 0.4409 |
| controlled_vs_proposed_raw | context_keyword_coverage | 51 | 26 | 67 | 0.5868 | 0.6623 |
| controlled_vs_proposed_raw | context_overlap | 65 | 42 | 37 | 0.5799 | 0.6075 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 65 | 7 | 72 | 0.7014 | 0.9028 |
| controlled_vs_proposed_raw | persona_style | 14 | 9 | 121 | 0.5174 | 0.6087 |
| controlled_vs_proposed_raw | distinct1 | 45 | 58 | 41 | 0.4549 | 0.4369 |
| controlled_vs_proposed_raw | length_score | 49 | 55 | 40 | 0.4792 | 0.4712 |
| controlled_vs_proposed_raw | sentence_score | 25 | 12 | 107 | 0.5451 | 0.6757 |
| controlled_vs_proposed_raw | overall_quality | 83 | 25 | 36 | 0.7014 | 0.7685 |
| controlled_vs_candidate_no_context | context_relevance | 85 | 29 | 29 | 0.6958 | 0.7456 |
| controlled_vs_candidate_no_context | persona_consistency | 72 | 14 | 57 | 0.7028 | 0.8372 |
| controlled_vs_candidate_no_context | naturalness | 45 | 68 | 30 | 0.4196 | 0.3982 |
| controlled_vs_candidate_no_context | quest_state_correctness | 85 | 29 | 29 | 0.6958 | 0.7456 |
| controlled_vs_candidate_no_context | lore_consistency | 53 | 14 | 76 | 0.6364 | 0.7910 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 18 | 18 | 107 | 0.5000 | 0.5000 |
| controlled_vs_candidate_no_context | objective_completion_support | 57 | 28 | 58 | 0.6014 | 0.6706 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 85 | 29 | 29 | 0.6958 | 0.7456 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 46 | 51 | 46 | 0.4825 | 0.4742 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 66 | 9 | 68 | 0.6993 | 0.8800 |
| controlled_vs_candidate_no_context | context_overlap | 82 | 32 | 29 | 0.6748 | 0.7193 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 64 | 14 | 65 | 0.6748 | 0.8205 |
| controlled_vs_candidate_no_context | persona_style | 13 | 7 | 123 | 0.5210 | 0.6500 |
| controlled_vs_candidate_no_context | distinct1 | 39 | 72 | 32 | 0.3846 | 0.3514 |
| controlled_vs_candidate_no_context | length_score | 47 | 62 | 34 | 0.4476 | 0.4312 |
| controlled_vs_candidate_no_context | sentence_score | 22 | 10 | 111 | 0.5420 | 0.6875 |
| controlled_vs_candidate_no_context | overall_quality | 95 | 19 | 29 | 0.7657 | 0.8333 |
| controlled_vs_baseline_no_context | context_relevance | 87 | 55 | 2 | 0.6111 | 0.6127 |
| controlled_vs_baseline_no_context | persona_consistency | 113 | 6 | 25 | 0.8715 | 0.9496 |
| controlled_vs_baseline_no_context | naturalness | 46 | 98 | 0 | 0.3194 | 0.3194 |
| controlled_vs_baseline_no_context | quest_state_correctness | 87 | 55 | 2 | 0.6111 | 0.6127 |
| controlled_vs_baseline_no_context | lore_consistency | 54 | 39 | 51 | 0.5521 | 0.5806 |
| controlled_vs_baseline_no_context | multi_turn_contradiction_safety | 18 | 18 | 108 | 0.5000 | 0.5000 |
| controlled_vs_baseline_no_context | objective_completion_support | 86 | 36 | 22 | 0.6736 | 0.7049 |
| controlled_vs_baseline_no_context | gameplay_usefulness | 97 | 47 | 0 | 0.6736 | 0.6736 |
| controlled_vs_baseline_no_context | time_pressure_acceptability | 71 | 62 | 11 | 0.5312 | 0.5338 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 69 | 15 | 60 | 0.6875 | 0.8214 |
| controlled_vs_baseline_no_context | context_overlap | 82 | 60 | 2 | 0.5764 | 0.5775 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 110 | 5 | 29 | 0.8646 | 0.9565 |
| controlled_vs_baseline_no_context | persona_style | 8 | 13 | 123 | 0.4826 | 0.3810 |
| controlled_vs_baseline_no_context | distinct1 | 17 | 119 | 8 | 0.1458 | 0.1250 |
| controlled_vs_baseline_no_context | length_score | 59 | 81 | 4 | 0.4236 | 0.4214 |
| controlled_vs_baseline_no_context | sentence_score | 38 | 10 | 96 | 0.5972 | 0.7917 |
| controlled_vs_baseline_no_context | overall_quality | 123 | 21 | 0 | 0.8542 | 0.8542 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 92 | 51 | 1 | 0.6424 | 0.6434 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 110 | 11 | 23 | 0.8438 | 0.9091 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | 52 | 91 | 1 | 0.3646 | 0.3636 |
| controlled_vs_baseline_no_context_phi3_latest | quest_state_correctness | 92 | 51 | 1 | 0.6424 | 0.6434 |
| controlled_vs_baseline_no_context_phi3_latest | lore_consistency | 53 | 39 | 52 | 0.5486 | 0.5761 |
| controlled_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 18 | 18 | 108 | 0.5000 | 0.5000 |
| controlled_vs_baseline_no_context_phi3_latest | objective_completion_support | 88 | 33 | 23 | 0.6910 | 0.7273 |
| controlled_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 102 | 42 | 0 | 0.7083 | 0.7083 |
| controlled_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 71 | 62 | 11 | 0.5312 | 0.5338 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 68 | 19 | 57 | 0.6701 | 0.7816 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 91 | 52 | 1 | 0.6354 | 0.6364 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 108 | 5 | 31 | 0.8576 | 0.9558 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | 9 | 13 | 122 | 0.4861 | 0.4091 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | 23 | 111 | 10 | 0.1944 | 0.1716 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | 53 | 84 | 7 | 0.3924 | 0.3869 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 38 | 10 | 96 | 0.5972 | 0.7917 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 120 | 24 | 0 | 0.8333 | 0.8333 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 87 | 55 | 2 | 0.6111 | 0.6127 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 113 | 6 | 25 | 0.8715 | 0.9496 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 46 | 98 | 0 | 0.3194 | 0.3194 |
| proposed_contextual_controlled_vs_baseline_no_context | quest_state_correctness | 87 | 55 | 2 | 0.6111 | 0.6127 |
| proposed_contextual_controlled_vs_baseline_no_context | lore_consistency | 54 | 39 | 51 | 0.5521 | 0.5806 |
| proposed_contextual_controlled_vs_baseline_no_context | multi_turn_contradiction_safety | 18 | 18 | 108 | 0.5000 | 0.5000 |
| proposed_contextual_controlled_vs_baseline_no_context | objective_completion_support | 86 | 36 | 22 | 0.6736 | 0.7049 |
| proposed_contextual_controlled_vs_baseline_no_context | gameplay_usefulness | 97 | 47 | 0 | 0.6736 | 0.6736 |
| proposed_contextual_controlled_vs_baseline_no_context | time_pressure_acceptability | 71 | 62 | 11 | 0.5312 | 0.5338 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 69 | 15 | 60 | 0.6875 | 0.8214 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 82 | 60 | 2 | 0.5764 | 0.5775 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 110 | 5 | 29 | 0.8646 | 0.9565 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 8 | 13 | 123 | 0.4826 | 0.3810 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | 17 | 119 | 8 | 0.1458 | 0.1250 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 59 | 81 | 4 | 0.4236 | 0.4214 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 38 | 10 | 96 | 0.5972 | 0.7917 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 123 | 21 | 0 | 0.8542 | 0.8542 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 92 | 51 | 1 | 0.6424 | 0.6434 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 110 | 11 | 23 | 0.8438 | 0.9091 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | 52 | 91 | 1 | 0.3646 | 0.3636 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | quest_state_correctness | 92 | 51 | 1 | 0.6424 | 0.6434 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | lore_consistency | 53 | 39 | 52 | 0.5486 | 0.5761 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | multi_turn_contradiction_safety | 18 | 18 | 108 | 0.5000 | 0.5000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | objective_completion_support | 88 | 33 | 23 | 0.6910 | 0.7273 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | gameplay_usefulness | 102 | 42 | 0 | 0.7083 | 0.7083 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | time_pressure_acceptability | 71 | 62 | 11 | 0.5312 | 0.5338 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 68 | 19 | 57 | 0.6701 | 0.7816 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 91 | 52 | 1 | 0.6354 | 0.6364 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 108 | 5 | 31 | 0.8576 | 0.9558 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | 9 | 13 | 122 | 0.4861 | 0.4091 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | 23 | 111 | 10 | 0.1944 | 0.1716 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | 53 | 84 | 7 | 0.3924 | 0.3869 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | 38 | 10 | 96 | 0.5972 | 0.7917 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 120 | 24 | 0 | 0.8333 | 0.8333 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.2222 | 0.2014 | 0.7986 |
| proposed_contextual | 0.0000 | 0.0000 | 0.4514 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0069 | 0.0069 | 0.4236 | 0.0000 | 0.0000 |
| baseline_no_context | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline_no_context_phi3_latest | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `8`
- Unique template signatures: `143`
- Template signature ratio: `0.9931`
- Effective sample size by source clustering: `8.00`
- Effective sample size by template-signature clustering: `142.03`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: unavailable (No BERTScore values found in merged scores.).

This report covers proposal RO5 metrics plus game-facing outcomes: quest-state correctness, lore consistency, contradiction safety, objective completion support, gameplay usefulness, and time-pressure acceptability.
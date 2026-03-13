# Proposal Alignment Evaluation Report

- Run ID: `20260313T004330Z`
- Generated: `2026-03-13T00:46:43.421088+00:00`
- Scenarios: `artifacts\proposal_control_tuning\auto_tune\20260313T004330Z\train_runs\trial_000\seed_19\20260313T004330Z\scenarios.jsonl`
- Scenario count: `12`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_tuned`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.1391 (0.0733, 0.2119) | 0.3000 (0.2214, 0.3925) | 0.8753 (0.8490, 0.9068) | 0.3410 (0.2911, 0.3999) | n/a |
| proposed_contextual_controlled_tuned | 0.1704 (0.1070, 0.2352) | 0.3072 (0.2429, 0.3740) | 0.8748 (0.8504, 0.8981) | 0.3581 (0.3190, 0.3957) | n/a |
| proposed_contextual | 0.0681 (0.0505, 0.0817) | 0.2238 (0.1528, 0.3028) | 0.8631 (0.8390, 0.8874) | 0.2771 (0.2482, 0.3086) | n/a |
| candidate_no_context | 0.0432 (0.0252, 0.0607) | 0.2889 (0.1991, 0.3726) | 0.8667 (0.8542, 0.8808) | 0.2909 (0.2513, 0.3308) | n/a |

## Game-facing Outcome Metrics (mean, 95% CI)
| Arm | Quest-state Correctness | Lore Consistency | Contradiction Safety | Objective Completion Support | Gameplay Usefulness | Time-pressure Acceptability |
|---|---:|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2213 (0.1655, 0.2799) | 0.0501 (0.0096, 0.0956) | 1.0000 (1.0000, 1.0000) | 0.1259 (0.0732, 0.1816) | 0.3392 (0.3174, 0.3613) | 0.3339 (0.3012, 0.3703) |
| proposed_contextual_controlled_tuned | 0.2526 (0.1940, 0.3137) | 0.0650 (0.0210, 0.1188) | 0.6667 (0.4167, 0.9167) | 0.1283 (0.0728, 0.1856) | 0.3416 (0.3248, 0.3596) | 0.3263 (0.2973, 0.3595) |
| proposed_contextual | 0.1637 (0.1437, 0.1812) | 0.0171 (0.0014, 0.0346) | 1.0000 (1.0000, 1.0000) | 0.1259 (0.0689, 0.1920) | 0.3150 (0.2857, 0.3452) | 0.3315 (0.2956, 0.3715) |
| candidate_no_context | 0.1408 (0.1228, 0.1589) | 0.0071 (0.0017, 0.0137) | 1.0000 (1.0000, 1.0000) | 0.1342 (0.0769, 0.1938) | 0.3118 (0.2806, 0.3452) | 0.3381 (0.3022, 0.3772) |

- Multi-turn contradiction rate is reported as `1 - contradiction_safety` in row-level outputs.

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0248 | 0.5743 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0651 | -0.2253 |
| proposed_vs_candidate_no_context | naturalness | -0.0036 | -0.0041 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.0229 | 0.1629 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0100 | 1.4228 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | -0.0083 | -0.0621 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 0.0032 | 0.0101 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | -0.0067 | -0.0197 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0303 | 0.6667 |
| proposed_vs_candidate_no_context | context_overlap | 0.0121 | 0.3170 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0813 | -0.4184 |
| proposed_vs_candidate_no_context | persona_style | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | distinct1 | 0.0105 | 0.0112 |
| proposed_vs_candidate_no_context | length_score | -0.0389 | -0.0761 |
| proposed_vs_candidate_no_context | sentence_score | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | overall_quality | -0.0138 | -0.0473 |
| controlled_vs_proposed_raw | context_relevance | 0.0710 | 1.0434 |
| controlled_vs_proposed_raw | persona_consistency | 0.0762 | 0.3404 |
| controlled_vs_proposed_raw | naturalness | 0.0122 | 0.0141 |
| controlled_vs_proposed_raw | quest_state_correctness | 0.0576 | 0.3517 |
| controlled_vs_proposed_raw | lore_consistency | 0.0330 | 1.9323 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_proposed_raw | objective_completion_support | 0.0000 | 0.0000 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0242 | 0.0768 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 0.0025 | 0.0074 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.0909 | 1.2000 |
| controlled_vs_proposed_raw | context_overlap | 0.0246 | 0.4914 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.0952 | 0.8421 |
| controlled_vs_proposed_raw | persona_style | 0.0000 | 0.0000 |
| controlled_vs_proposed_raw | distinct1 | -0.0172 | -0.0182 |
| controlled_vs_proposed_raw | length_score | 0.0806 | 0.1706 |
| controlled_vs_proposed_raw | sentence_score | 0.0292 | 0.0320 |
| controlled_vs_proposed_raw | overall_quality | 0.0638 | 0.2304 |
| controlled_vs_candidate_no_context | context_relevance | 0.0959 | 2.2170 |
| controlled_vs_candidate_no_context | persona_consistency | 0.0111 | 0.0385 |
| controlled_vs_candidate_no_context | naturalness | 0.0086 | 0.0099 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.0805 | 0.5719 |
| controlled_vs_candidate_no_context | lore_consistency | 0.0431 | 6.1044 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | 0.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | -0.0083 | -0.0621 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0273 | 0.0877 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | -0.0042 | -0.0125 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.1212 | 2.6667 |
| controlled_vs_candidate_no_context | context_overlap | 0.0367 | 0.9641 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.0139 | 0.0714 |
| controlled_vs_candidate_no_context | persona_style | 0.0000 | 0.0000 |
| controlled_vs_candidate_no_context | distinct1 | -0.0067 | -0.0072 |
| controlled_vs_candidate_no_context | length_score | 0.0417 | 0.0815 |
| controlled_vs_candidate_no_context | sentence_score | 0.0292 | 0.0320 |
| controlled_vs_candidate_no_context | overall_quality | 0.0501 | 0.1722 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0313 | 0.2248 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0072 | 0.0241 |
| controlled_alt_vs_controlled_default | naturalness | -0.0004 | -0.0005 |
| controlled_alt_vs_controlled_default | quest_state_correctness | 0.0313 | 0.1415 |
| controlled_alt_vs_controlled_default | lore_consistency | 0.0149 | 0.2969 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | -0.3333 | -0.3333 |
| controlled_alt_vs_controlled_default | objective_completion_support | 0.0024 | 0.0193 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | 0.0024 | 0.0071 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | -0.0076 | -0.0229 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0379 | 0.2273 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0159 | 0.2121 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0127 | 0.0610 |
| controlled_alt_vs_controlled_default | persona_style | -0.0146 | -0.0219 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0055 | 0.0059 |
| controlled_alt_vs_controlled_default | length_score | -0.0278 | -0.0503 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0292 | 0.0310 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0171 | 0.0503 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1023 | 1.5028 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.0834 | 0.3728 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0117 | 0.0136 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 0.0889 | 0.5430 |
| controlled_alt_vs_proposed_raw | lore_consistency | 0.0479 | 2.8028 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | -0.3333 | -0.3333 |
| controlled_alt_vs_proposed_raw | objective_completion_support | 0.0024 | 0.0193 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 0.0266 | 0.0844 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | -0.0052 | -0.0157 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.1288 | 1.7000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0405 | 0.8076 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1079 | 0.9544 |
| controlled_alt_vs_proposed_raw | persona_style | -0.0146 | -0.0219 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0117 | -0.0124 |
| controlled_alt_vs_proposed_raw | length_score | 0.0528 | 0.1118 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0583 | 0.0639 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.0810 | 0.2922 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.1271 | 2.9403 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.0184 | 0.0635 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0081 | 0.0094 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 0.1118 | 0.7944 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 0.0579 | 8.2135 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | -0.3333 | -0.3333 |
| controlled_alt_vs_candidate_no_context | objective_completion_support | -0.0059 | -0.0440 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 0.0297 | 0.0954 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | -0.0119 | -0.0351 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.1591 | 3.5000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0526 | 1.3806 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.0266 | 0.1367 |
| controlled_alt_vs_candidate_no_context | persona_style | -0.0146 | -0.0219 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0012 | -0.0013 |
| controlled_alt_vs_candidate_no_context | length_score | 0.0139 | 0.0272 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.0583 | 0.0639 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.0672 | 0.2311 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0248 | (0.0076, 0.0433) | 0.0027 | 0.0248 | (0.0078, 0.0446) | 0.0037 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0651 | (-0.1619, 0.0127) | 0.9377 | -0.0651 | (-0.1111, -0.0190) | 1.0000 |
| proposed_vs_candidate_no_context | naturalness | -0.0036 | (-0.0300, 0.0204) | 0.6153 | -0.0036 | (-0.0257, 0.0190) | 0.6477 |
| proposed_vs_candidate_no_context | quest_state_correctness | 0.0229 | (0.0070, 0.0409) | 0.0010 | 0.0229 | (0.0058, 0.0468) | 0.0027 |
| proposed_vs_candidate_no_context | lore_consistency | 0.0100 | (-0.0030, 0.0247) | 0.0543 | 0.0100 | (0.0000, 0.0200) | 0.0033 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | objective_completion_support | -0.0083 | (-0.0281, 0.0067) | 0.8597 | -0.0083 | (-0.0247, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 0.0032 | (-0.0100, 0.0173) | 0.3130 | 0.0032 | (-0.0122, 0.0213) | 0.4340 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | -0.0067 | (-0.0218, 0.0039) | 0.8703 | -0.0067 | (-0.0197, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0303 | (0.0076, 0.0530) | 0.0087 | 0.0303 | (0.0101, 0.0496) | 0.0033 |
| proposed_vs_candidate_no_context | context_overlap | 0.0121 | (0.0020, 0.0259) | 0.0013 | 0.0121 | (0.0003, 0.0319) | 0.0053 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0813 | (-0.2083, 0.0159) | 0.9333 | -0.0813 | (-0.1389, -0.0238) | 1.0000 |
| proposed_vs_candidate_no_context | persona_style | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | distinct1 | 0.0105 | (0.0000, 0.0216) | 0.0303 | 0.0105 | (0.0000, 0.0227) | 0.0560 |
| proposed_vs_candidate_no_context | length_score | -0.0389 | (-0.1389, 0.0583) | 0.7783 | -0.0389 | (-0.1077, 0.0267) | 0.8813 |
| proposed_vs_candidate_no_context | sentence_score | 0.0000 | (-0.0875, 0.0875) | 0.6503 | 0.0000 | (-0.0808, 0.0808) | 0.6360 |
| proposed_vs_candidate_no_context | overall_quality | -0.0138 | (-0.0494, 0.0169) | 0.7847 | -0.0138 | (-0.0349, -0.0015) | 1.0000 |
| controlled_vs_proposed_raw | context_relevance | 0.0710 | (0.0100, 0.1431) | 0.0093 | 0.0710 | (0.0122, 0.1593) | 0.0063 |
| controlled_vs_proposed_raw | persona_consistency | 0.0762 | (0.0048, 0.1556) | 0.0093 | 0.0762 | (0.0190, 0.1455) | 0.0043 |
| controlled_vs_proposed_raw | naturalness | 0.0122 | (-0.0170, 0.0420) | 0.2127 | 0.0122 | (-0.0004, 0.0288) | 0.0617 |
| controlled_vs_proposed_raw | quest_state_correctness | 0.0576 | (0.0071, 0.1165) | 0.0110 | 0.0576 | (0.0126, 0.1321) | 0.0043 |
| controlled_vs_proposed_raw | lore_consistency | 0.0330 | (-0.0024, 0.0729) | 0.0347 | 0.0330 | (-0.0009, 0.0745) | 0.0600 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_proposed_raw | objective_completion_support | -0.0000 | (-0.0573, 0.0631) | 0.5073 | -0.0000 | (-0.0206, 0.0197) | 0.5397 |
| controlled_vs_proposed_raw | gameplay_usefulness | 0.0242 | (-0.0025, 0.0568) | 0.0443 | 0.0242 | (0.0104, 0.0361) | 0.0047 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 0.0025 | (-0.0284, 0.0370) | 0.4757 | 0.0025 | (-0.0061, 0.0086) | 0.2977 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.0909 | (0.0152, 0.1742) | 0.0100 | 0.0909 | (0.0152, 0.1983) | 0.0057 |
| controlled_vs_proposed_raw | context_overlap | 0.0246 | (-0.0018, 0.0552) | 0.0367 | 0.0246 | (0.0015, 0.0636) | 0.0057 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.0952 | (0.0118, 0.1944) | 0.0093 | 0.0952 | (0.0238, 0.1818) | 0.0037 |
| controlled_vs_proposed_raw | persona_style | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_proposed_raw | distinct1 | -0.0172 | (-0.0485, 0.0109) | 0.8740 | -0.0172 | (-0.0493, 0.0002) | 0.9347 |
| controlled_vs_proposed_raw | length_score | 0.0806 | (-0.0390, 0.2056) | 0.0977 | 0.0806 | (0.0259, 0.1200) | 0.0030 |
| controlled_vs_proposed_raw | sentence_score | 0.0292 | (-0.0583, 0.1167) | 0.3973 | 0.0292 | (-0.0700, 0.1000) | 0.3697 |
| controlled_vs_proposed_raw | overall_quality | 0.0638 | (0.0174, 0.1161) | 0.0013 | 0.0638 | (0.0175, 0.1281) | 0.0037 |
| controlled_vs_candidate_no_context | context_relevance | 0.0959 | (0.0242, 0.1741) | 0.0020 | 0.0959 | (0.0230, 0.1784) | 0.0037 |
| controlled_vs_candidate_no_context | persona_consistency | 0.0111 | (-0.0873, 0.1111) | 0.4150 | 0.0111 | (0.0000, 0.0333) | 0.3183 |
| controlled_vs_candidate_no_context | naturalness | 0.0086 | (-0.0204, 0.0415) | 0.3210 | 0.0086 | (-0.0185, 0.0279) | 0.2013 |
| controlled_vs_candidate_no_context | quest_state_correctness | 0.0805 | (0.0222, 0.1411) | 0.0010 | 0.0805 | (0.0223, 0.1446) | 0.0033 |
| controlled_vs_candidate_no_context | lore_consistency | 0.0431 | (0.0052, 0.0899) | 0.0117 | 0.0431 | (-0.0009, 0.0929) | 0.0590 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | objective_completion_support | -0.0083 | (-0.0616, 0.0493) | 0.6260 | -0.0083 | (-0.0433, 0.0194) | 0.6747 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 0.0273 | (-0.0009, 0.0587) | 0.0297 | 0.0273 | (0.0080, 0.0512) | 0.0037 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | -0.0042 | (-0.0317, 0.0299) | 0.6230 | -0.0042 | (-0.0246, 0.0082) | 0.6930 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.1212 | (0.0303, 0.2197) | 0.0057 | 0.1212 | (0.0303, 0.2231) | 0.0023 |
| controlled_vs_candidate_no_context | context_overlap | 0.0367 | (0.0109, 0.0678) | 0.0013 | 0.0367 | (0.0059, 0.0740) | 0.0020 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.0139 | (-0.1111, 0.1370) | 0.4287 | 0.0139 | (-0.0000, 0.0417) | 0.3283 |
| controlled_vs_candidate_no_context | persona_style | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | distinct1 | -0.0067 | (-0.0331, 0.0140) | 0.7020 | -0.0067 | (-0.0250, 0.0063) | 0.6983 |
| controlled_vs_candidate_no_context | length_score | 0.0417 | (-0.0694, 0.1695) | 0.2650 | 0.0417 | (-0.0333, 0.0952) | 0.0960 |
| controlled_vs_candidate_no_context | sentence_score | 0.0292 | (-0.0875, 0.1458) | 0.4213 | 0.0292 | (-0.1400, 0.1885) | 0.4193 |
| controlled_vs_candidate_no_context | overall_quality | 0.0501 | (-0.0050, 0.1105) | 0.0337 | 0.0501 | (0.0141, 0.0907) | 0.0030 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0313 | (-0.0539, 0.1094) | 0.2270 | 0.0313 | (-0.0582, 0.1068) | 0.2670 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0072 | (-0.0562, 0.0705) | 0.4047 | 0.0072 | (-0.0582, 0.0589) | 0.3733 |
| controlled_alt_vs_controlled_default | naturalness | -0.0004 | (-0.0364, 0.0372) | 0.5047 | -0.0004 | (-0.0335, 0.0257) | 0.5690 |
| controlled_alt_vs_controlled_default | quest_state_correctness | 0.0313 | (-0.0446, 0.0990) | 0.1927 | 0.0313 | (-0.0337, 0.0987) | 0.1710 |
| controlled_alt_vs_controlled_default | lore_consistency | 0.0149 | (-0.0542, 0.0790) | 0.3180 | 0.0149 | (-0.0335, 0.0499) | 0.3087 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | -0.3333 | (-0.5833, -0.0833) | 1.0000 | -0.3333 | (-0.8000, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | objective_completion_support | 0.0024 | (-0.0349, 0.0346) | 0.4373 | 0.0024 | (-0.0320, 0.0301) | 0.4160 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | 0.0024 | (-0.0161, 0.0204) | 0.3973 | 0.0024 | (-0.0101, 0.0185) | 0.3927 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | -0.0076 | (-0.0254, 0.0080) | 0.8197 | -0.0076 | (-0.0177, 0.0058) | 0.8517 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0379 | (-0.0606, 0.1290) | 0.2327 | 0.0379 | (-0.0699, 0.1273) | 0.2653 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0159 | (-0.0309, 0.0644) | 0.2620 | 0.0159 | (-0.0301, 0.0674) | 0.2577 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0127 | (-0.0667, 0.0881) | 0.3853 | 0.0127 | (-0.0727, 0.0810) | 0.3693 |
| controlled_alt_vs_controlled_default | persona_style | -0.0146 | (-0.0438, 0.0000) | 1.0000 | -0.0146 | (-0.0350, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0055 | (-0.0101, 0.0240) | 0.2787 | 0.0055 | (-0.0119, 0.0205) | 0.2227 |
| controlled_alt_vs_controlled_default | length_score | -0.0278 | (-0.1833, 0.1250) | 0.6313 | -0.0278 | (-0.1533, 0.0889) | 0.6897 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0292 | (-0.0583, 0.1167) | 0.4003 | 0.0292 | (-0.0700, 0.1750) | 0.4190 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0171 | (-0.0293, 0.0648) | 0.2360 | 0.0171 | (-0.0441, 0.0658) | 0.2617 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1023 | (0.0432, 0.1670) | 0.0000 | 0.1023 | (0.0640, 0.1458) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.0834 | (0.0392, 0.1302) | 0.0003 | 0.0834 | (0.0780, 0.0889) | 0.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0117 | (-0.0184, 0.0375) | 0.2087 | 0.0117 | (-0.0051, 0.0261) | 0.0507 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 0.0889 | (0.0332, 0.1515) | 0.0003 | 0.0889 | (0.0538, 0.1247) | 0.0000 |
| controlled_alt_vs_proposed_raw | lore_consistency | 0.0479 | (0.0039, 0.0971) | 0.0173 | 0.0479 | (0.0352, 0.0559) | 0.0000 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | -0.3333 | (-0.5833, -0.0833) | 1.0000 | -0.3333 | (-0.8000, 0.0000) | 1.0000 |
| controlled_alt_vs_proposed_raw | objective_completion_support | 0.0024 | (-0.0504, 0.0637) | 0.4740 | 0.0024 | (-0.0357, 0.0162) | 0.2750 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 0.0266 | (-0.0047, 0.0587) | 0.0447 | 0.0266 | (0.0077, 0.0455) | 0.0000 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | -0.0052 | (-0.0386, 0.0316) | 0.6283 | -0.0052 | (-0.0127, 0.0025) | 0.8853 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.1288 | (0.0530, 0.2045) | 0.0003 | 0.1288 | (0.0788, 0.1894) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0405 | (0.0066, 0.0806) | 0.0067 | 0.0405 | (0.0221, 0.0809) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1079 | (0.0544, 0.1615) | 0.0000 | 0.1079 | (0.1037, 0.1111) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_style | -0.0146 | (-0.0438, 0.0000) | 1.0000 | -0.0146 | (-0.0350, 0.0000) | 1.0000 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0117 | (-0.0364, 0.0127) | 0.8253 | -0.0117 | (-0.0324, 0.0090) | 0.8263 |
| controlled_alt_vs_proposed_raw | length_score | 0.0528 | (-0.1139, 0.1917) | 0.2400 | 0.0528 | (-0.0311, 0.1148) | 0.0690 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0583 | (-0.0583, 0.1750) | 0.2047 | 0.0583 | (0.0000, 0.1167) | 0.0580 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.0810 | (0.0471, 0.1165) | 0.0000 | 0.0810 | (0.0580, 0.1037) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.1271 | (0.0658, 0.1923) | 0.0000 | 0.1271 | (0.0793, 0.1892) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.0184 | (-0.0667, 0.0921) | 0.3247 | 0.0184 | (-0.0222, 0.0589) | 0.1747 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0081 | (-0.0181, 0.0317) | 0.2517 | 0.0081 | (-0.0157, 0.0415) | 0.2580 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 0.1118 | (0.0523, 0.1745) | 0.0000 | 0.1118 | (0.0663, 0.1623) | 0.0000 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 0.0579 | (0.0165, 0.1068) | 0.0010 | 0.0579 | (0.0401, 0.0675) | 0.0000 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | -0.3333 | (-0.5833, -0.0833) | 1.0000 | -0.3333 | (-0.8000, 0.0000) | 1.0000 |
| controlled_alt_vs_candidate_no_context | objective_completion_support | -0.0059 | (-0.0550, 0.0547) | 0.6093 | -0.0059 | (-0.0360, 0.0156) | 0.6647 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 0.0297 | (-0.0038, 0.0632) | 0.0417 | 0.0297 | (0.0067, 0.0681) | 0.0000 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | -0.0119 | (-0.0434, 0.0239) | 0.7690 | -0.0119 | (-0.0228, -0.0010) | 0.9810 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.1591 | (0.0833, 0.2348) | 0.0000 | 0.1591 | (0.1030, 0.2424) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0526 | (0.0192, 0.0912) | 0.0003 | 0.0526 | (0.0288, 0.0858) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.0266 | (-0.0806, 0.1155) | 0.2990 | 0.0266 | (-0.0278, 0.0810) | 0.1863 |
| controlled_alt_vs_candidate_no_context | persona_style | -0.0146 | (-0.0438, 0.0000) | 1.0000 | -0.0146 | (-0.0350, 0.0000) | 1.0000 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0012 | (-0.0221, 0.0176) | 0.5333 | -0.0012 | (-0.0114, 0.0090) | 0.6933 |
| controlled_alt_vs_candidate_no_context | length_score | 0.0139 | (-0.1278, 0.1389) | 0.4127 | 0.0139 | (-0.0833, 0.1400) | 0.4317 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.0583 | (-0.0583, 0.1750) | 0.2120 | 0.0583 | (0.0000, 0.1750) | 0.3113 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.0672 | (0.0166, 0.1161) | 0.0037 | 0.0672 | (0.0424, 0.1020) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 5 | 0 | 7 | 0.7083 | 1.0000 |
| proposed_vs_candidate_no_context | persona_consistency | 1 | 3 | 8 | 0.4167 | 0.2500 |
| proposed_vs_candidate_no_context | naturalness | 2 | 3 | 7 | 0.4583 | 0.4000 |
| proposed_vs_candidate_no_context | quest_state_correctness | 5 | 0 | 7 | 0.7083 | 1.0000 |
| proposed_vs_candidate_no_context | lore_consistency | 4 | 1 | 7 | 0.6250 | 0.8000 |
| proposed_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 12 | 0.5000 | nan |
| proposed_vs_candidate_no_context | objective_completion_support | 1 | 2 | 9 | 0.4583 | 0.3333 |
| proposed_vs_candidate_no_context | gameplay_usefulness | 3 | 2 | 7 | 0.5417 | 0.6000 |
| proposed_vs_candidate_no_context | time_pressure_acceptability | 1 | 2 | 9 | 0.4583 | 0.3333 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 4 | 0 | 8 | 0.6667 | 1.0000 |
| proposed_vs_candidate_no_context | context_overlap | 5 | 0 | 7 | 0.7083 | 1.0000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 1 | 3 | 8 | 0.4167 | 0.2500 |
| proposed_vs_candidate_no_context | persona_style | 0 | 0 | 12 | 0.5000 | nan |
| proposed_vs_candidate_no_context | distinct1 | 3 | 0 | 9 | 0.6250 | 1.0000 |
| proposed_vs_candidate_no_context | length_score | 2 | 3 | 7 | 0.4583 | 0.4000 |
| proposed_vs_candidate_no_context | sentence_score | 1 | 1 | 10 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | overall_quality | 2 | 3 | 7 | 0.4583 | 0.4000 |
| controlled_vs_proposed_raw | context_relevance | 8 | 2 | 2 | 0.7500 | 0.8000 |
| controlled_vs_proposed_raw | persona_consistency | 5 | 1 | 6 | 0.6667 | 0.8333 |
| controlled_vs_proposed_raw | naturalness | 6 | 4 | 2 | 0.5833 | 0.6000 |
| controlled_vs_proposed_raw | quest_state_correctness | 8 | 2 | 2 | 0.7500 | 0.8000 |
| controlled_vs_proposed_raw | lore_consistency | 4 | 2 | 6 | 0.5833 | 0.6667 |
| controlled_vs_proposed_raw | multi_turn_contradiction_safety | 0 | 0 | 12 | 0.5000 | nan |
| controlled_vs_proposed_raw | objective_completion_support | 3 | 5 | 4 | 0.4167 | 0.3750 |
| controlled_vs_proposed_raw | gameplay_usefulness | 5 | 5 | 2 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | time_pressure_acceptability | 3 | 5 | 4 | 0.4167 | 0.3750 |
| controlled_vs_proposed_raw | context_keyword_coverage | 5 | 1 | 6 | 0.6667 | 0.8333 |
| controlled_vs_proposed_raw | context_overlap | 7 | 3 | 2 | 0.6667 | 0.7000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 5 | 1 | 6 | 0.6667 | 0.8333 |
| controlled_vs_proposed_raw | persona_style | 0 | 0 | 12 | 0.5000 | nan |
| controlled_vs_proposed_raw | distinct1 | 5 | 4 | 3 | 0.5417 | 0.5556 |
| controlled_vs_proposed_raw | length_score | 6 | 4 | 2 | 0.5833 | 0.6000 |
| controlled_vs_proposed_raw | sentence_score | 2 | 1 | 9 | 0.5417 | 0.6667 |
| controlled_vs_proposed_raw | overall_quality | 7 | 3 | 2 | 0.6667 | 0.7000 |
| controlled_vs_candidate_no_context | context_relevance | 8 | 2 | 2 | 0.7500 | 0.8000 |
| controlled_vs_candidate_no_context | persona_consistency | 4 | 3 | 5 | 0.5417 | 0.5714 |
| controlled_vs_candidate_no_context | naturalness | 5 | 5 | 2 | 0.5000 | 0.5000 |
| controlled_vs_candidate_no_context | quest_state_correctness | 8 | 2 | 2 | 0.7500 | 0.8000 |
| controlled_vs_candidate_no_context | lore_consistency | 4 | 2 | 6 | 0.5833 | 0.6667 |
| controlled_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 0 | 12 | 0.5000 | nan |
| controlled_vs_candidate_no_context | objective_completion_support | 2 | 5 | 5 | 0.3750 | 0.2857 |
| controlled_vs_candidate_no_context | gameplay_usefulness | 6 | 4 | 2 | 0.5833 | 0.6000 |
| controlled_vs_candidate_no_context | time_pressure_acceptability | 2 | 5 | 5 | 0.3750 | 0.2857 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 6 | 1 | 5 | 0.7083 | 0.8571 |
| controlled_vs_candidate_no_context | context_overlap | 8 | 2 | 2 | 0.7500 | 0.8000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 4 | 3 | 5 | 0.5417 | 0.5714 |
| controlled_vs_candidate_no_context | persona_style | 0 | 0 | 12 | 0.5000 | nan |
| controlled_vs_candidate_no_context | distinct1 | 5 | 4 | 3 | 0.5417 | 0.5556 |
| controlled_vs_candidate_no_context | length_score | 4 | 6 | 2 | 0.4167 | 0.4000 |
| controlled_vs_candidate_no_context | sentence_score | 3 | 2 | 7 | 0.5417 | 0.6000 |
| controlled_vs_candidate_no_context | overall_quality | 7 | 3 | 2 | 0.6667 | 0.7000 |
| controlled_alt_vs_controlled_default | context_relevance | 5 | 3 | 4 | 0.5833 | 0.6250 |
| controlled_alt_vs_controlled_default | persona_consistency | 2 | 2 | 8 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | naturalness | 5 | 3 | 4 | 0.5833 | 0.6250 |
| controlled_alt_vs_controlled_default | quest_state_correctness | 5 | 3 | 4 | 0.5833 | 0.6250 |
| controlled_alt_vs_controlled_default | lore_consistency | 5 | 2 | 5 | 0.6250 | 0.7143 |
| controlled_alt_vs_controlled_default | multi_turn_contradiction_safety | 0 | 4 | 8 | 0.3333 | 0.0000 |
| controlled_alt_vs_controlled_default | objective_completion_support | 5 | 2 | 5 | 0.6250 | 0.7143 |
| controlled_alt_vs_controlled_default | gameplay_usefulness | 4 | 4 | 4 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | time_pressure_acceptability | 2 | 5 | 5 | 0.3750 | 0.2857 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 4 | 2 | 6 | 0.5833 | 0.6667 |
| controlled_alt_vs_controlled_default | context_overlap | 4 | 4 | 4 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 2 | 1 | 9 | 0.5417 | 0.6667 |
| controlled_alt_vs_controlled_default | persona_style | 0 | 1 | 11 | 0.4583 | 0.0000 |
| controlled_alt_vs_controlled_default | distinct1 | 4 | 4 | 4 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | length_score | 4 | 4 | 4 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | sentence_score | 2 | 1 | 9 | 0.5417 | 0.6667 |
| controlled_alt_vs_controlled_default | overall_quality | 6 | 2 | 4 | 0.6667 | 0.7500 |
| controlled_alt_vs_proposed_raw | context_relevance | 8 | 3 | 1 | 0.7083 | 0.7273 |
| controlled_alt_vs_proposed_raw | persona_consistency | 7 | 1 | 4 | 0.7500 | 0.8750 |
| controlled_alt_vs_proposed_raw | naturalness | 8 | 3 | 1 | 0.7083 | 0.7273 |
| controlled_alt_vs_proposed_raw | quest_state_correctness | 8 | 3 | 1 | 0.7083 | 0.7273 |
| controlled_alt_vs_proposed_raw | lore_consistency | 6 | 2 | 4 | 0.6667 | 0.7500 |
| controlled_alt_vs_proposed_raw | multi_turn_contradiction_safety | 0 | 4 | 8 | 0.3333 | 0.0000 |
| controlled_alt_vs_proposed_raw | objective_completion_support | 3 | 6 | 3 | 0.3750 | 0.3333 |
| controlled_alt_vs_proposed_raw | gameplay_usefulness | 8 | 3 | 1 | 0.7083 | 0.7273 |
| controlled_alt_vs_proposed_raw | time_pressure_acceptability | 3 | 6 | 3 | 0.3750 | 0.3333 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 6 | 0 | 6 | 0.7500 | 1.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 8 | 3 | 1 | 0.7083 | 0.7273 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 7 | 0 | 5 | 0.7917 | 1.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 0 | 1 | 11 | 0.4583 | 0.0000 |
| controlled_alt_vs_proposed_raw | distinct1 | 6 | 5 | 1 | 0.5417 | 0.5455 |
| controlled_alt_vs_proposed_raw | length_score | 8 | 3 | 1 | 0.7083 | 0.7273 |
| controlled_alt_vs_proposed_raw | sentence_score | 3 | 1 | 8 | 0.5833 | 0.7500 |
| controlled_alt_vs_proposed_raw | overall_quality | 10 | 1 | 1 | 0.8750 | 0.9091 |
| controlled_alt_vs_candidate_no_context | context_relevance | 9 | 2 | 1 | 0.7917 | 0.8182 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 6 | 3 | 3 | 0.6250 | 0.6667 |
| controlled_alt_vs_candidate_no_context | naturalness | 8 | 3 | 1 | 0.7083 | 0.7273 |
| controlled_alt_vs_candidate_no_context | quest_state_correctness | 9 | 2 | 1 | 0.7917 | 0.8182 |
| controlled_alt_vs_candidate_no_context | lore_consistency | 5 | 2 | 5 | 0.6250 | 0.7143 |
| controlled_alt_vs_candidate_no_context | multi_turn_contradiction_safety | 0 | 4 | 8 | 0.3333 | 0.0000 |
| controlled_alt_vs_candidate_no_context | objective_completion_support | 2 | 7 | 3 | 0.2917 | 0.2222 |
| controlled_alt_vs_candidate_no_context | gameplay_usefulness | 8 | 3 | 1 | 0.7083 | 0.7273 |
| controlled_alt_vs_candidate_no_context | time_pressure_acceptability | 2 | 7 | 3 | 0.2917 | 0.2222 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 8 | 0 | 4 | 0.8333 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 8 | 3 | 1 | 0.7083 | 0.7273 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 6 | 2 | 4 | 0.6667 | 0.7500 |
| controlled_alt_vs_candidate_no_context | persona_style | 0 | 1 | 11 | 0.4583 | 0.0000 |
| controlled_alt_vs_candidate_no_context | distinct1 | 6 | 4 | 2 | 0.5833 | 0.6000 |
| controlled_alt_vs_candidate_no_context | length_score | 7 | 3 | 2 | 0.6667 | 0.7000 |
| controlled_alt_vs_candidate_no_context | sentence_score | 3 | 1 | 8 | 0.5833 | 0.7500 |
| controlled_alt_vs_candidate_no_context | overall_quality | 8 | 3 | 1 | 0.7083 | 0.7273 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.1667 | 0.1667 | 0.8333 |
| proposed_contextual_controlled_tuned | 0.0000 | 0.0000 | 0.1667 | 0.3333 | 0.6667 |
| proposed_contextual | 0.0000 | 0.0000 | 0.5833 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.5833 | 0.0000 | 0.0000 |

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
| proposed_contextual_controlled_tuned | 0.2500 | 0.7500 | 1 | 4 |
| proposed_contextual | 0.0000 | 1.0000 | 0 | 4 |
| candidate_no_context | 0.0000 | 1.0000 | 0 | 4 |
- Detailed source-level values are published in `multi_turn_contradictions.json`.

- BERTScore status: unavailable (disabled_by_flag).

This report covers proposal RO5 metrics plus game-facing outcomes: quest-state correctness, lore consistency, contradiction safety, objective completion support, gameplay usefulness, and time-pressure acceptability.
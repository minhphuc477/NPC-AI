# Lexical Diversity Benchmark

- Run dir: `artifacts\proposal\20260224T175344Z`
- Scenario count: `112`

## Arm Summary
| Arm | Metric | Mean | 95% CI |
|---|---|---:|---:|
| proposed_contextual_controlled | distinct1 | 0.9401 | (0.9364, 0.9439) |
| proposed_contextual_controlled | distinct2 | 0.9997 | (0.9992, 1.0000) |
| proposed_contextual_controlled | content_distinct1 | 0.9899 | (0.9861, 0.9934) |
| proposed_contextual_controlled | mtld | 184.0777 | (163.1033, 208.3721) |
| proposed_contextual_controlled | repetition_penalty | 0.0599 | (0.0563, 0.0636) |
| proposed_contextual_controlled | lexical_richness | 0.9219 | (0.9109, 0.9328) |
| proposed_contextual | distinct1 | 0.9400 | (0.9335, 0.9463) |
| proposed_contextual | distinct2 | 0.9997 | (0.9992, 1.0000) |
| proposed_contextual | content_distinct1 | 0.9983 | (0.9968, 0.9995) |
| proposed_contextual | mtld | 137.1228 | (102.1437, 177.6543) |
| proposed_contextual | repetition_penalty | 0.0600 | (0.0530, 0.0663) |
| proposed_contextual | lexical_richness | 0.8172 | (0.7965, 0.8385) |
| candidate_no_context | distinct1 | 0.9352 | (0.9286, 0.9422) |
| candidate_no_context | distinct2 | 0.9998 | (0.9995, 1.0000) |
| candidate_no_context | content_distinct1 | 0.9983 | (0.9966, 0.9995) |
| candidate_no_context | mtld | 114.4581 | (81.2154, 154.3198) |
| candidate_no_context | repetition_penalty | 0.0648 | (0.0581, 0.0714) |
| candidate_no_context | lexical_richness | 0.7966 | (0.7785, 0.8155) |
| baseline_no_context | distinct1 | 0.9775 | (0.9719, 0.9827) |
| baseline_no_context | distinct2 | 1.0000 | (1.0000, 1.0000) |
| baseline_no_context | content_distinct1 | 0.9985 | (0.9971, 0.9997) |
| baseline_no_context | mtld | 132.0259 | (102.0816, 165.2877) |
| baseline_no_context | repetition_penalty | 0.0225 | (0.0171, 0.0283) |
| baseline_no_context | lexical_richness | 0.8341 | (0.8128, 0.8561) |
| baseline_no_context_phi3_latest | distinct1 | 0.9807 | (0.9761, 0.9850) |
| baseline_no_context_phi3_latest | distinct2 | 0.9996 | (0.9987, 1.0000) |
| baseline_no_context_phi3_latest | content_distinct1 | 0.9980 | (0.9961, 0.9995) |
| baseline_no_context_phi3_latest | mtld | 161.3251 | (125.3162, 201.4784) |
| baseline_no_context_phi3_latest | repetition_penalty | 0.0193 | (0.0149, 0.0240) |
| baseline_no_context_phi3_latest | lexical_richness | 0.8437 | (0.8214, 0.8666) |

## Paired Deltas
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) |
|---|---|---:|---:|---:|
| proposed_vs_candidate_no_context | distinct1 | 0.0049 | (-0.0022, 0.0119) | 0.0977 |
| proposed_vs_candidate_no_context | distinct2 | -0.0002 | (-0.0007, 0.0003) | 0.7153 |
| proposed_vs_candidate_no_context | content_distinct1 | -0.0000 | (-0.0020, 0.0022) | 0.4893 |
| proposed_vs_candidate_no_context | mtld | 22.6647 | (-19.8969, 67.5575) | 0.1577 |
| proposed_vs_candidate_no_context | repetition_penalty | -0.0049 | (-0.0117, 0.0021) | 0.9073 |
| proposed_vs_candidate_no_context | lexical_richness | 0.0206 | (-0.0036, 0.0435) | 0.0467 |
| proposed_vs_baseline_no_context | distinct1 | -0.0374 | (-0.0459, -0.0288) | 1.0000 |
| proposed_vs_baseline_no_context | distinct2 | -0.0003 | (-0.0008, 0.0000) | 1.0000 |
| proposed_vs_baseline_no_context | content_distinct1 | -0.0003 | (-0.0021, 0.0016) | 0.6073 |
| proposed_vs_baseline_no_context | mtld | 5.0969 | (-45.1779, 57.3537) | 0.4297 |
| proposed_vs_baseline_no_context | repetition_penalty | 0.0374 | (0.0292, 0.0457) | 0.0000 |
| proposed_vs_baseline_no_context | lexical_richness | -0.0169 | (-0.0476, 0.0138) | 0.8560 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | -0.0406 | (-0.0479, -0.0335) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | distinct2 | 0.0001 | (-0.0007, 0.0011) | 0.4717 |
| proposed_vs_baseline_no_context_phi3_latest | content_distinct1 | 0.0002 | (-0.0020, 0.0027) | 0.4457 |
| proposed_vs_baseline_no_context_phi3_latest | mtld | -24.2023 | (-77.7314, 32.3562) | 0.8120 |
| proposed_vs_baseline_no_context_phi3_latest | repetition_penalty | 0.0406 | (0.0336, 0.0481) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | lexical_richness | -0.0265 | (-0.0581, 0.0047) | 0.9540 |
| controlled_vs_proposed_raw | distinct1 | 0.0000 | (-0.0080, 0.0075) | 0.4890 |
| controlled_vs_proposed_raw | distinct2 | 0.0000 | (-0.0006, 0.0007) | 0.4737 |
| controlled_vs_proposed_raw | content_distinct1 | -0.0084 | (-0.0122, -0.0047) | 1.0000 |
| controlled_vs_proposed_raw | mtld | 46.9549 | (5.3531, 88.6339) | 0.0150 |
| controlled_vs_proposed_raw | repetition_penalty | -0.0000 | (-0.0079, 0.0079) | 0.5110 |
| controlled_vs_proposed_raw | lexical_richness | 0.1047 | (0.0810, 0.1277) | 0.0000 |
| controlled_vs_candidate_no_context | distinct1 | 0.0049 | (-0.0022, 0.0119) | 0.0863 |
| controlled_vs_candidate_no_context | distinct2 | -0.0001 | (-0.0007, 0.0004) | 0.7270 |
| controlled_vs_candidate_no_context | content_distinct1 | -0.0084 | (-0.0121, -0.0048) | 1.0000 |
| controlled_vs_candidate_no_context | mtld | 69.6196 | (26.4001, 112.6339) | 0.0003 |
| controlled_vs_candidate_no_context | repetition_penalty | -0.0049 | (-0.0119, 0.0023) | 0.9027 |
| controlled_vs_candidate_no_context | lexical_richness | 0.1252 | (0.1042, 0.1453) | 0.0000 |
| controlled_vs_baseline_no_context | distinct1 | -0.0374 | (-0.0435, -0.0312) | 1.0000 |
| controlled_vs_baseline_no_context | distinct2 | -0.0003 | (-0.0008, 0.0000) | 1.0000 |
| controlled_vs_baseline_no_context | content_distinct1 | -0.0087 | (-0.0124, -0.0048) | 1.0000 |
| controlled_vs_baseline_no_context | mtld | 52.0518 | (14.8403, 89.6113) | 0.0033 |
| controlled_vs_baseline_no_context | repetition_penalty | 0.0374 | (0.0311, 0.0437) | 0.0000 |
| controlled_vs_baseline_no_context | lexical_richness | 0.0878 | (0.0660, 0.1094) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0406 | (-0.0466, -0.0344) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | distinct2 | 0.0001 | (-0.0006, 0.0011) | 0.4947 |
| controlled_vs_baseline_no_context_phi3_latest | content_distinct1 | -0.0082 | (-0.0122, -0.0041) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | mtld | 22.7526 | (-19.2379, 62.4696) | 0.1347 |
| controlled_vs_baseline_no_context_phi3_latest | repetition_penalty | 0.0406 | (0.0343, 0.0469) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | lexical_richness | 0.0781 | (0.0558, 0.1003) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0374 | (-0.0437, -0.0311) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct2 | -0.0003 | (-0.0008, 0.0000) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | content_distinct1 | -0.0087 | (-0.0126, -0.0050) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | mtld | 52.0518 | (15.6911, 88.0726) | 0.0017 |
| proposed_contextual_controlled_vs_baseline_no_context | repetition_penalty | 0.0374 | (0.0307, 0.0436) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | lexical_richness | 0.0878 | (0.0653, 0.1094) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0406 | (-0.0467, -0.0342) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct2 | 0.0001 | (-0.0006, 0.0013) | 0.4793 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | content_distinct1 | -0.0082 | (-0.0122, -0.0042) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | mtld | 22.7526 | (-19.5402, 63.4090) | 0.1437 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | repetition_penalty | 0.0406 | (0.0346, 0.0468) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | lexical_richness | 0.0781 | (0.0554, 0.1005) | 0.0000 |
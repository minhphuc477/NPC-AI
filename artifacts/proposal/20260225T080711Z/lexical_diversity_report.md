# Lexical Diversity Benchmark

- Run dir: `artifacts\proposal\20260225T080711Z`
- Scenario count: `112`

## Arm Summary
| Arm | Metric | Mean | 95% CI |
|---|---|---:|---:|
| proposed_contextual_controlled | distinct1 | 0.9404 | (0.9353, 0.9453) |
| proposed_contextual_controlled | distinct2 | 0.9979 | (0.9940, 1.0000) |
| proposed_contextual_controlled | content_distinct1 | 0.9910 | (0.9862, 0.9949) |
| proposed_contextual_controlled | mtld | 213.6989 | (183.6409, 248.0234) |
| proposed_contextual_controlled | repetition_penalty | 0.0596 | (0.0549, 0.0651) |
| proposed_contextual_controlled | lexical_richness | 0.9264 | (0.9141, 0.9379) |
| proposed_contextual | distinct1 | 0.9478 | (0.9403, 0.9552) |
| proposed_contextual | distinct2 | 1.0000 | (1.0000, 1.0000) |
| proposed_contextual | content_distinct1 | 0.9991 | (0.9979, 1.0000) |
| proposed_contextual | mtld | 66.1021 | (49.7293, 84.4171) |
| proposed_contextual | repetition_penalty | 0.0522 | (0.0443, 0.0597) |
| proposed_contextual | lexical_richness | 0.7792 | (0.7642, 0.7949) |
| candidate_no_context | distinct1 | 0.9353 | (0.9285, 0.9422) |
| candidate_no_context | distinct2 | 1.0000 | (1.0000, 1.0000) |
| candidate_no_context | content_distinct1 | 0.9985 | (0.9966, 1.0000) |
| candidate_no_context | mtld | 77.5001 | (59.8190, 96.5447) |
| candidate_no_context | repetition_penalty | 0.0647 | (0.0578, 0.0718) |
| candidate_no_context | lexical_richness | 0.7905 | (0.7740, 0.8081) |
| baseline_no_context | distinct1 | 0.9849 | (0.9806, 0.9889) |
| baseline_no_context | distinct2 | 1.0000 | (1.0000, 1.0000) |
| baseline_no_context | content_distinct1 | 0.9987 | (0.9970, 1.0000) |
| baseline_no_context | mtld | 97.0337 | (76.4371, 119.4510) |
| baseline_no_context | repetition_penalty | 0.0151 | (0.0111, 0.0191) |
| baseline_no_context | lexical_richness | 0.8158 | (0.7964, 0.8372) |
| baseline_no_context_phi3_latest | distinct1 | 0.9818 | (0.9773, 0.9861) |
| baseline_no_context_phi3_latest | distinct2 | 1.0000 | (1.0000, 1.0000) |
| baseline_no_context_phi3_latest | content_distinct1 | 0.9984 | (0.9968, 0.9997) |
| baseline_no_context_phi3_latest | mtld | 106.0498 | (84.8108, 129.4261) |
| baseline_no_context_phi3_latest | repetition_penalty | 0.0182 | (0.0139, 0.0227) |
| baseline_no_context_phi3_latest | lexical_richness | 0.8302 | (0.8082, 0.8524) |

## Paired Deltas
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) |
|---|---|---:|---:|---:|
| proposed_vs_candidate_no_context | distinct1 | 0.0125 | (0.0063, 0.0189) | 0.0000 |
| proposed_vs_candidate_no_context | distinct2 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | content_distinct1 | 0.0007 | (-0.0013, 0.0030) | 0.3037 |
| proposed_vs_candidate_no_context | mtld | -11.3980 | (-34.0196, 11.5441) | 0.8390 |
| proposed_vs_candidate_no_context | repetition_penalty | -0.0125 | (-0.0187, -0.0065) | 1.0000 |
| proposed_vs_candidate_no_context | lexical_richness | -0.0112 | (-0.0326, 0.0098) | 0.8560 |
| proposed_vs_baseline_no_context | distinct1 | -0.0371 | (-0.0461, -0.0282) | 1.0000 |
| proposed_vs_baseline_no_context | distinct2 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_baseline_no_context | content_distinct1 | 0.0005 | (-0.0014, 0.0025) | 0.3393 |
| proposed_vs_baseline_no_context | mtld | -30.9315 | (-57.9431, -3.2493) | 0.9853 |
| proposed_vs_baseline_no_context | repetition_penalty | 0.0371 | (0.0279, 0.0463) | 0.0000 |
| proposed_vs_baseline_no_context | lexical_richness | -0.0366 | (-0.0619, -0.0126) | 0.9990 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | -0.0339 | (-0.0435, -0.0250) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | distinct2 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | content_distinct1 | 0.0007 | (-0.0013, 0.0027) | 0.2557 |
| proposed_vs_baseline_no_context_phi3_latest | mtld | -39.9477 | (-67.8334, -11.7671) | 0.9970 |
| proposed_vs_baseline_no_context_phi3_latest | repetition_penalty | 0.0339 | (0.0247, 0.0430) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | lexical_richness | -0.0510 | (-0.0773, -0.0242) | 1.0000 |
| controlled_vs_proposed_raw | distinct1 | -0.0074 | (-0.0166, 0.0020) | 0.9407 |
| controlled_vs_proposed_raw | distinct2 | -0.0021 | (-0.0062, 0.0000) | 1.0000 |
| controlled_vs_proposed_raw | content_distinct1 | -0.0081 | (-0.0120, -0.0049) | 1.0000 |
| controlled_vs_proposed_raw | mtld | 147.5968 | (112.6680, 183.9132) | 0.0000 |
| controlled_vs_proposed_raw | repetition_penalty | 0.0074 | (-0.0018, 0.0170) | 0.0600 |
| controlled_vs_proposed_raw | lexical_richness | 0.1472 | (0.1264, 0.1663) | 0.0000 |
| controlled_vs_candidate_no_context | distinct1 | 0.0051 | (-0.0037, 0.0133) | 0.1250 |
| controlled_vs_candidate_no_context | distinct2 | -0.0021 | (-0.0062, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | content_distinct1 | -0.0075 | (-0.0124, -0.0030) | 1.0000 |
| controlled_vs_candidate_no_context | mtld | 136.1988 | (100.5494, 175.7736) | 0.0000 |
| controlled_vs_candidate_no_context | repetition_penalty | -0.0051 | (-0.0136, 0.0032) | 0.8817 |
| controlled_vs_candidate_no_context | lexical_richness | 0.1359 | (0.1144, 0.1558) | 0.0000 |
| controlled_vs_baseline_no_context | distinct1 | -0.0445 | (-0.0508, -0.0380) | 1.0000 |
| controlled_vs_baseline_no_context | distinct2 | -0.0021 | (-0.0060, 0.0000) | 1.0000 |
| controlled_vs_baseline_no_context | content_distinct1 | -0.0077 | (-0.0127, -0.0034) | 1.0000 |
| controlled_vs_baseline_no_context | mtld | 116.6652 | (79.2714, 157.7145) | 0.0000 |
| controlled_vs_baseline_no_context | repetition_penalty | 0.0445 | (0.0383, 0.0508) | 0.0000 |
| controlled_vs_baseline_no_context | lexical_richness | 0.1106 | (0.0872, 0.1336) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0414 | (-0.0482, -0.0347) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | distinct2 | -0.0021 | (-0.0060, 0.0000) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | content_distinct1 | -0.0074 | (-0.0123, -0.0033) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | mtld | 107.6491 | (69.8911, 149.2237) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | repetition_penalty | 0.0414 | (0.0345, 0.0481) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | lexical_richness | 0.0962 | (0.0715, 0.1210) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0445 | (-0.0509, -0.0380) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct2 | -0.0021 | (-0.0060, 0.0000) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | content_distinct1 | -0.0077 | (-0.0124, -0.0035) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | mtld | 116.6652 | (82.2678, 156.4772) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | repetition_penalty | 0.0445 | (0.0381, 0.0506) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | lexical_richness | 0.1106 | (0.0878, 0.1336) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0414 | (-0.0482, -0.0344) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct2 | -0.0021 | (-0.0060, 0.0000) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | content_distinct1 | -0.0074 | (-0.0122, -0.0033) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | mtld | 107.6491 | (70.1019, 149.5062) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | repetition_penalty | 0.0414 | (0.0345, 0.0482) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | lexical_richness | 0.0962 | (0.0721, 0.1203) | 0.0000 |
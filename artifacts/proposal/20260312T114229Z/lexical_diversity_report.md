# Lexical Diversity Benchmark

- Run dir: `artifacts\proposal\20260312T114229Z`
- Scenario count: `144`

## Arm Summary
| Arm | Metric | Mean | 95% CI |
|---|---|---:|---:|
| proposed_contextual_controlled | distinct1 | 0.9359 | (0.9286, 0.9434) |
| proposed_contextual_controlled | distinct2 | 0.9898 | (0.9866, 0.9928) |
| proposed_contextual_controlled | content_distinct1 | 0.9652 | (0.9557, 0.9739) |
| proposed_contextual_controlled | mtld | 159.5888 | (131.2217, 191.4277) |
| proposed_contextual_controlled | repetition_penalty | 0.0641 | (0.0568, 0.0715) |
| proposed_contextual_controlled | lexical_richness | 0.8494 | (0.8283, 0.8709) |
| proposed_contextual | distinct1 | 0.9372 | (0.9291, 0.9455) |
| proposed_contextual | distinct2 | 0.9888 | (0.9854, 0.9921) |
| proposed_contextual | content_distinct1 | 0.9652 | (0.9560, 0.9742) |
| proposed_contextual | mtld | 155.5479 | (127.0391, 186.7971) |
| proposed_contextual | repetition_penalty | 0.0628 | (0.0549, 0.0706) |
| proposed_contextual | lexical_richness | 0.8380 | (0.8175, 0.8577) |
| candidate_no_context | distinct1 | 0.9389 | (0.9311, 0.9468) |
| candidate_no_context | distinct2 | 0.9897 | (0.9866, 0.9927) |
| candidate_no_context | content_distinct1 | 0.9654 | (0.9565, 0.9742) |
| candidate_no_context | mtld | 156.5792 | (127.6224, 189.2039) |
| candidate_no_context | repetition_penalty | 0.0611 | (0.0536, 0.0686) |
| candidate_no_context | lexical_richness | 0.8358 | (0.8168, 0.8569) |
| baseline_no_context | distinct1 | 0.9775 | (0.9731, 0.9818) |
| baseline_no_context | distinct2 | 0.9999 | (0.9996, 1.0000) |
| baseline_no_context | content_distinct1 | 0.9969 | (0.9952, 0.9984) |
| baseline_no_context | mtld | 149.8611 | (122.0311, 180.4619) |
| baseline_no_context | repetition_penalty | 0.0225 | (0.0181, 0.0269) |
| baseline_no_context | lexical_richness | 0.8496 | (0.8308, 0.8694) |
| baseline_no_context_phi3_latest | distinct1 | 0.9823 | (0.9784, 0.9861) |
| baseline_no_context_phi3_latest | distinct2 | 1.0000 | (1.0000, 1.0000) |
| baseline_no_context_phi3_latest | content_distinct1 | 0.9980 | (0.9966, 0.9992) |
| baseline_no_context_phi3_latest | mtld | 134.9103 | (109.0104, 162.6216) |
| baseline_no_context_phi3_latest | repetition_penalty | 0.0177 | (0.0139, 0.0217) |
| baseline_no_context_phi3_latest | lexical_richness | 0.8346 | (0.8161, 0.8534) |

## Paired Deltas
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) |
|---|---|---:|---:|---:|
| proposed_vs_candidate_no_context | distinct1 | -0.0018 | (-0.0093, 0.0057) | 0.6747 |
| proposed_vs_candidate_no_context | distinct2 | -0.0009 | (-0.0032, 0.0016) | 0.7583 |
| proposed_vs_candidate_no_context | content_distinct1 | -0.0002 | (-0.0077, 0.0073) | 0.5120 |
| proposed_vs_candidate_no_context | mtld | -1.0313 | (-37.6366, 35.9795) | 0.5150 |
| proposed_vs_candidate_no_context | repetition_penalty | 0.0018 | (-0.0056, 0.0092) | 0.3200 |
| proposed_vs_candidate_no_context | lexical_richness | 0.0022 | (-0.0202, 0.0254) | 0.4110 |
| proposed_vs_baseline_no_context | distinct1 | -0.0403 | (-0.0495, -0.0312) | 1.0000 |
| proposed_vs_baseline_no_context | distinct2 | -0.0111 | (-0.0143, -0.0079) | 1.0000 |
| proposed_vs_baseline_no_context | content_distinct1 | -0.0317 | (-0.0411, -0.0228) | 1.0000 |
| proposed_vs_baseline_no_context | mtld | 5.6869 | (-28.5300, 39.8281) | 0.3607 |
| proposed_vs_baseline_no_context | repetition_penalty | 0.0403 | (0.0311, 0.0494) | 0.0000 |
| proposed_vs_baseline_no_context | lexical_richness | -0.0116 | (-0.0361, 0.0124) | 0.8417 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | -0.0452 | (-0.0540, -0.0366) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | distinct2 | -0.0112 | (-0.0144, -0.0080) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | content_distinct1 | -0.0328 | (-0.0419, -0.0232) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | mtld | 20.6376 | (-18.4392, 58.1854) | 0.1483 |
| proposed_vs_baseline_no_context_phi3_latest | repetition_penalty | 0.0452 | (0.0364, 0.0535) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | lexical_richness | 0.0034 | (-0.0230, 0.0295) | 0.3977 |
| controlled_vs_proposed_raw | distinct1 | -0.0012 | (-0.0094, 0.0075) | 0.6047 |
| controlled_vs_proposed_raw | distinct2 | 0.0010 | (-0.0018, 0.0037) | 0.2593 |
| controlled_vs_proposed_raw | content_distinct1 | 0.0000 | (-0.0083, 0.0078) | 0.4993 |
| controlled_vs_proposed_raw | mtld | 4.0409 | (-35.6131, 43.5972) | 0.4213 |
| controlled_vs_proposed_raw | repetition_penalty | 0.0012 | (-0.0069, 0.0092) | 0.3903 |
| controlled_vs_proposed_raw | lexical_richness | 0.0114 | (-0.0133, 0.0358) | 0.1770 |
| controlled_vs_candidate_no_context | distinct1 | -0.0030 | (-0.0111, 0.0053) | 0.7673 |
| controlled_vs_candidate_no_context | distinct2 | 0.0001 | (-0.0027, 0.0029) | 0.4723 |
| controlled_vs_candidate_no_context | content_distinct1 | -0.0002 | (-0.0092, 0.0082) | 0.5110 |
| controlled_vs_candidate_no_context | mtld | 3.0096 | (-38.8824, 45.8782) | 0.4460 |
| controlled_vs_candidate_no_context | repetition_penalty | 0.0030 | (-0.0052, 0.0112) | 0.2420 |
| controlled_vs_candidate_no_context | lexical_richness | 0.0136 | (-0.0133, 0.0390) | 0.1610 |
| controlled_vs_baseline_no_context | distinct1 | -0.0416 | (-0.0502, -0.0329) | 1.0000 |
| controlled_vs_baseline_no_context | distinct2 | -0.0101 | (-0.0132, -0.0071) | 1.0000 |
| controlled_vs_baseline_no_context | content_distinct1 | -0.0317 | (-0.0410, -0.0231) | 1.0000 |
| controlled_vs_baseline_no_context | mtld | 9.7278 | (-34.3455, 53.7420) | 0.3223 |
| controlled_vs_baseline_no_context | repetition_penalty | 0.0416 | (0.0331, 0.0502) | 0.0000 |
| controlled_vs_baseline_no_context | lexical_richness | -0.0002 | (-0.0286, 0.0302) | 0.5000 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0464 | (-0.0551, -0.0377) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | distinct2 | -0.0102 | (-0.0135, -0.0072) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | content_distinct1 | -0.0328 | (-0.0420, -0.0234) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | mtld | 24.6785 | (-15.3840, 65.1922) | 0.1177 |
| controlled_vs_baseline_no_context_phi3_latest | repetition_penalty | 0.0464 | (0.0382, 0.0548) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | lexical_richness | 0.0148 | (-0.0107, 0.0406) | 0.1267 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0416 | (-0.0499, -0.0328) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct2 | -0.0101 | (-0.0133, -0.0070) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | content_distinct1 | -0.0317 | (-0.0405, -0.0232) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | mtld | 9.7278 | (-33.9471, 54.1317) | 0.3320 |
| proposed_contextual_controlled_vs_baseline_no_context | repetition_penalty | 0.0416 | (0.0331, 0.0502) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | lexical_richness | -0.0002 | (-0.0315, 0.0293) | 0.5117 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0464 | (-0.0549, -0.0380) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct2 | -0.0102 | (-0.0134, -0.0072) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | content_distinct1 | -0.0328 | (-0.0426, -0.0237) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | mtld | 24.6785 | (-16.6279, 65.8031) | 0.1213 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | repetition_penalty | 0.0464 | (0.0381, 0.0553) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | lexical_richness | 0.0148 | (-0.0115, 0.0410) | 0.1430 |
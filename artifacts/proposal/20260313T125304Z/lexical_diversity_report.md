# Lexical Diversity Benchmark

- Run dir: `artifacts\proposal\20260313T125304Z`
- Scenario count: `144`

## Arm Summary
| Arm | Metric | Mean | 95% CI |
|---|---|---:|---:|
| proposed_contextual_controlled | distinct1 | 0.9332 | (0.9257, 0.9407) |
| proposed_contextual_controlled | distinct2 | 0.9886 | (0.9853, 0.9918) |
| proposed_contextual_controlled | content_distinct1 | 0.9624 | (0.9536, 0.9709) |
| proposed_contextual_controlled | mtld | 159.3113 | (130.9094, 191.6748) |
| proposed_contextual_controlled | repetition_penalty | 0.0668 | (0.0594, 0.0742) |
| proposed_contextual_controlled | lexical_richness | 0.8438 | (0.8246, 0.8625) |
| proposed_contextual_controlled_tuned | distinct1 | 0.9359 | (0.9283, 0.9431) |
| proposed_contextual_controlled_tuned | distinct2 | 0.9903 | (0.9872, 0.9933) |
| proposed_contextual_controlled_tuned | content_distinct1 | 0.9684 | (0.9598, 0.9766) |
| proposed_contextual_controlled_tuned | mtld | 187.7163 | (158.1516, 220.0464) |
| proposed_contextual_controlled_tuned | repetition_penalty | 0.0641 | (0.0568, 0.0715) |
| proposed_contextual_controlled_tuned | lexical_richness | 0.8725 | (0.8519, 0.8924) |
| proposed_contextual | distinct1 | 0.9400 | (0.9322, 0.9480) |
| proposed_contextual | distinct2 | 0.9897 | (0.9865, 0.9926) |
| proposed_contextual | content_distinct1 | 0.9670 | (0.9580, 0.9756) |
| proposed_contextual | mtld | 174.8462 | (141.2131, 209.6952) |
| proposed_contextual | repetition_penalty | 0.0600 | (0.0518, 0.0681) |
| proposed_contextual | lexical_richness | 0.8368 | (0.8178, 0.8575) |
| candidate_no_context | distinct1 | 0.9422 | (0.9346, 0.9495) |
| candidate_no_context | distinct2 | 0.9908 | (0.9877, 0.9937) |
| candidate_no_context | content_distinct1 | 0.9674 | (0.9584, 0.9760) |
| candidate_no_context | mtld | 188.5810 | (154.1131, 227.5221) |
| candidate_no_context | repetition_penalty | 0.0578 | (0.0502, 0.0654) |
| candidate_no_context | lexical_richness | 0.8474 | (0.8283, 0.8685) |
| baseline_no_context | distinct1 | 0.9813 | (0.9772, 0.9853) |
| baseline_no_context | distinct2 | 1.0000 | (1.0000, 1.0000) |
| baseline_no_context | content_distinct1 | 0.9988 | (0.9974, 0.9998) |
| baseline_no_context | mtld | 146.1488 | (117.1004, 176.6560) |
| baseline_no_context | repetition_penalty | 0.0187 | (0.0148, 0.0226) |
| baseline_no_context | lexical_richness | 0.8383 | (0.8193, 0.8570) |
| baseline_no_context_phi3_latest | distinct1 | 0.9808 | (0.9766, 0.9848) |
| baseline_no_context_phi3_latest | distinct2 | 1.0000 | (1.0000, 1.0000) |
| baseline_no_context_phi3_latest | content_distinct1 | 0.9983 | (0.9970, 0.9994) |
| baseline_no_context_phi3_latest | mtld | 140.5287 | (112.6644, 170.4962) |
| baseline_no_context_phi3_latest | repetition_penalty | 0.0192 | (0.0152, 0.0234) |
| baseline_no_context_phi3_latest | lexical_richness | 0.8381 | (0.8191, 0.8565) |

## Paired Deltas
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) |
|---|---|---:|---:|---:|
| proposed_vs_candidate_no_context | distinct1 | -0.0022 | (-0.0085, 0.0043) | 0.7360 |
| proposed_vs_candidate_no_context | distinct2 | -0.0011 | (-0.0032, 0.0010) | 0.8500 |
| proposed_vs_candidate_no_context | content_distinct1 | -0.0004 | (-0.0072, 0.0065) | 0.5627 |
| proposed_vs_candidate_no_context | mtld | -13.7348 | (-58.1446, 31.3000) | 0.7123 |
| proposed_vs_candidate_no_context | repetition_penalty | 0.0022 | (-0.0042, 0.0087) | 0.2527 |
| proposed_vs_candidate_no_context | lexical_richness | -0.0106 | (-0.0336, 0.0106) | 0.8257 |
| proposed_vs_baseline_no_context | distinct1 | -0.0413 | (-0.0499, -0.0327) | 1.0000 |
| proposed_vs_baseline_no_context | distinct2 | -0.0103 | (-0.0136, -0.0073) | 1.0000 |
| proposed_vs_baseline_no_context | content_distinct1 | -0.0318 | (-0.0408, -0.0227) | 1.0000 |
| proposed_vs_baseline_no_context | mtld | 28.6974 | (-14.3249, 71.9960) | 0.1020 |
| proposed_vs_baseline_no_context | repetition_penalty | 0.0413 | (0.0322, 0.0500) | 0.0000 |
| proposed_vs_baseline_no_context | lexical_richness | -0.0014 | (-0.0295, 0.0255) | 0.5420 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | -0.0408 | (-0.0499, -0.0319) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | distinct2 | -0.0103 | (-0.0135, -0.0072) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | content_distinct1 | -0.0313 | (-0.0400, -0.0227) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | mtld | 34.3175 | (-6.0239, 75.2928) | 0.0460 |
| proposed_vs_baseline_no_context_phi3_latest | repetition_penalty | 0.0408 | (0.0320, 0.0494) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | lexical_richness | -0.0012 | (-0.0266, 0.0233) | 0.5163 |
| controlled_vs_proposed_raw | distinct1 | -0.0068 | (-0.0156, 0.0026) | 0.9237 |
| controlled_vs_proposed_raw | distinct2 | -0.0010 | (-0.0041, 0.0023) | 0.7453 |
| controlled_vs_proposed_raw | content_distinct1 | -0.0046 | (-0.0137, 0.0044) | 0.8400 |
| controlled_vs_proposed_raw | mtld | -15.5350 | (-58.9605, 28.6603) | 0.7570 |
| controlled_vs_proposed_raw | repetition_penalty | 0.0068 | (-0.0026, 0.0155) | 0.0817 |
| controlled_vs_proposed_raw | lexical_richness | 0.0070 | (-0.0180, 0.0320) | 0.2923 |
| controlled_vs_candidate_no_context | distinct1 | -0.0090 | (-0.0171, -0.0010) | 0.9857 |
| controlled_vs_candidate_no_context | distinct2 | -0.0022 | (-0.0051, 0.0007) | 0.9290 |
| controlled_vs_candidate_no_context | content_distinct1 | -0.0050 | (-0.0145, 0.0040) | 0.8623 |
| controlled_vs_candidate_no_context | mtld | -29.2698 | (-76.3108, 18.3161) | 0.8787 |
| controlled_vs_candidate_no_context | repetition_penalty | 0.0090 | (0.0011, 0.0169) | 0.0150 |
| controlled_vs_candidate_no_context | lexical_richness | -0.0036 | (-0.0266, 0.0197) | 0.6143 |
| controlled_vs_baseline_no_context | distinct1 | -0.0481 | (-0.0566, -0.0398) | 1.0000 |
| controlled_vs_baseline_no_context | distinct2 | -0.0114 | (-0.0146, -0.0083) | 1.0000 |
| controlled_vs_baseline_no_context | content_distinct1 | -0.0364 | (-0.0458, -0.0277) | 1.0000 |
| controlled_vs_baseline_no_context | mtld | 13.1625 | (-29.5758, 57.2470) | 0.2723 |
| controlled_vs_baseline_no_context | repetition_penalty | 0.0481 | (0.0393, 0.0565) | 0.0000 |
| controlled_vs_baseline_no_context | lexical_richness | 0.0056 | (-0.0209, 0.0330) | 0.3387 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0476 | (-0.0565, -0.0387) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | distinct2 | -0.0114 | (-0.0147, -0.0081) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | content_distinct1 | -0.0359 | (-0.0450, -0.0273) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | mtld | 18.7826 | (-25.1221, 64.6292) | 0.1953 |
| controlled_vs_baseline_no_context_phi3_latest | repetition_penalty | 0.0476 | (0.0388, 0.0570) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | lexical_richness | 0.0058 | (-0.0213, 0.0315) | 0.3350 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0027 | (-0.0048, 0.0105) | 0.2357 |
| controlled_alt_vs_controlled_default | distinct2 | 0.0017 | (-0.0013, 0.0049) | 0.1390 |
| controlled_alt_vs_controlled_default | content_distinct1 | 0.0060 | (-0.0025, 0.0147) | 0.0790 |
| controlled_alt_vs_controlled_default | mtld | 28.4050 | (-9.1768, 66.2130) | 0.0653 |
| controlled_alt_vs_controlled_default | repetition_penalty | -0.0027 | (-0.0103, 0.0046) | 0.7667 |
| controlled_alt_vs_controlled_default | lexical_richness | 0.0286 | (0.0085, 0.0492) | 0.0030 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0041 | (-0.0124, 0.0039) | 0.8397 |
| controlled_alt_vs_proposed_raw | distinct2 | 0.0006 | (-0.0022, 0.0035) | 0.3333 |
| controlled_alt_vs_proposed_raw | content_distinct1 | 0.0014 | (-0.0063, 0.0095) | 0.3640 |
| controlled_alt_vs_proposed_raw | mtld | 12.8701 | (-27.0247, 52.1414) | 0.2657 |
| controlled_alt_vs_proposed_raw | repetition_penalty | 0.0041 | (-0.0037, 0.0124) | 0.1537 |
| controlled_alt_vs_proposed_raw | lexical_richness | 0.0356 | (0.0131, 0.0596) | 0.0020 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0063 | (-0.0139, 0.0016) | 0.9443 |
| controlled_alt_vs_candidate_no_context | distinct2 | -0.0005 | (-0.0034, 0.0023) | 0.6463 |
| controlled_alt_vs_candidate_no_context | content_distinct1 | 0.0010 | (-0.0070, 0.0090) | 0.4123 |
| controlled_alt_vs_candidate_no_context | mtld | -0.8648 | (-45.8329, 41.8886) | 0.4957 |
| controlled_alt_vs_candidate_no_context | repetition_penalty | 0.0063 | (-0.0016, 0.0138) | 0.0593 |
| controlled_alt_vs_candidate_no_context | lexical_richness | 0.0250 | (0.0009, 0.0483) | 0.0227 |
| controlled_alt_vs_baseline_no_context | distinct1 | -0.0454 | (-0.0536, -0.0372) | 1.0000 |
| controlled_alt_vs_baseline_no_context | distinct2 | -0.0097 | (-0.0129, -0.0067) | 1.0000 |
| controlled_alt_vs_baseline_no_context | content_distinct1 | -0.0304 | (-0.0389, -0.0225) | 1.0000 |
| controlled_alt_vs_baseline_no_context | mtld | 41.5675 | (1.7931, 82.9276) | 0.0210 |
| controlled_alt_vs_baseline_no_context | repetition_penalty | 0.0454 | (0.0371, 0.0535) | 0.0000 |
| controlled_alt_vs_baseline_no_context | lexical_richness | 0.0342 | (0.0070, 0.0609) | 0.0077 |
| controlled_alt_vs_baseline_no_context_phi3_latest | distinct1 | -0.0449 | (-0.0534, -0.0364) | 1.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | distinct2 | -0.0097 | (-0.0128, -0.0067) | 1.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | content_distinct1 | -0.0299 | (-0.0387, -0.0217) | 1.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | mtld | 47.1876 | (7.9374, 86.7775) | 0.0110 |
| controlled_alt_vs_baseline_no_context_phi3_latest | repetition_penalty | 0.0449 | (0.0369, 0.0535) | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | lexical_richness | 0.0344 | (0.0066, 0.0622) | 0.0073 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | distinct1 | -0.0454 | (-0.0539, -0.0370) | 1.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | distinct2 | -0.0097 | (-0.0129, -0.0066) | 1.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | content_distinct1 | -0.0304 | (-0.0391, -0.0226) | 1.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | mtld | 41.5675 | (1.5078, 82.1525) | 0.0220 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | repetition_penalty | 0.0454 | (0.0369, 0.0532) | 0.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | lexical_richness | 0.0342 | (0.0054, 0.0617) | 0.0103 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | distinct1 | -0.0449 | (-0.0533, -0.0366) | 1.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | distinct2 | -0.0097 | (-0.0130, -0.0066) | 1.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | content_distinct1 | -0.0299 | (-0.0391, -0.0218) | 1.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | mtld | 47.1876 | (6.2568, 85.0348) | 0.0127 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | repetition_penalty | 0.0449 | (0.0368, 0.0532) | 0.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | lexical_richness | 0.0344 | (0.0069, 0.0612) | 0.0067 |
# Lexical Diversity Benchmark

- Run dir: `artifacts\proposal\20260313T095814Z`
- Scenario count: `144`

## Arm Summary
| Arm | Metric | Mean | 95% CI |
|---|---|---:|---:|
| proposed_contextual_controlled | distinct1 | 0.9375 | (0.9305, 0.9446) |
| proposed_contextual_controlled | distinct2 | 0.9913 | (0.9878, 0.9944) |
| proposed_contextual_controlled | content_distinct1 | 0.9678 | (0.9590, 0.9761) |
| proposed_contextual_controlled | mtld | 181.0684 | (151.0644, 211.0640) |
| proposed_contextual_controlled | repetition_penalty | 0.0625 | (0.0552, 0.0698) |
| proposed_contextual_controlled | lexical_richness | 0.8672 | (0.8479, 0.8873) |
| proposed_contextual_controlled_tuned | distinct1 | 0.9319 | (0.9243, 0.9397) |
| proposed_contextual_controlled_tuned | distinct2 | 0.9888 | (0.9831, 0.9931) |
| proposed_contextual_controlled_tuned | content_distinct1 | 0.9658 | (0.9558, 0.9751) |
| proposed_contextual_controlled_tuned | mtld | 154.4347 | (130.0858, 181.5094) |
| proposed_contextual_controlled_tuned | repetition_penalty | 0.0681 | (0.0602, 0.0762) |
| proposed_contextual_controlled_tuned | lexical_richness | 0.8550 | (0.8351, 0.8763) |
| proposed_contextual | distinct1 | 0.9452 | (0.9374, 0.9530) |
| proposed_contextual | distinct2 | 0.9909 | (0.9880, 0.9938) |
| proposed_contextual | content_distinct1 | 0.9714 | (0.9633, 0.9794) |
| proposed_contextual | mtld | 173.4298 | (140.8135, 206.6737) |
| proposed_contextual | repetition_penalty | 0.0548 | (0.0470, 0.0627) |
| proposed_contextual | lexical_richness | 0.8396 | (0.8204, 0.8605) |
| candidate_no_context | distinct1 | 0.9384 | (0.9306, 0.9459) |
| candidate_no_context | distinct2 | 0.9901 | (0.9868, 0.9931) |
| candidate_no_context | content_distinct1 | 0.9690 | (0.9600, 0.9775) |
| candidate_no_context | mtld | 156.7670 | (126.5496, 190.0267) |
| candidate_no_context | repetition_penalty | 0.0616 | (0.0538, 0.0696) |
| candidate_no_context | lexical_richness | 0.8410 | (0.8213, 0.8611) |
| baseline_no_context | distinct1 | 0.9796 | (0.9751, 0.9838) |
| baseline_no_context | distinct2 | 0.9997 | (0.9993, 1.0000) |
| baseline_no_context | content_distinct1 | 0.9987 | (0.9975, 0.9998) |
| baseline_no_context | mtld | 153.5752 | (121.3065, 186.7289) |
| baseline_no_context | repetition_penalty | 0.0204 | (0.0163, 0.0248) |
| baseline_no_context | lexical_richness | 0.8410 | (0.8226, 0.8594) |
| baseline_no_context_phi3_latest | distinct1 | 0.9791 | (0.9745, 0.9833) |
| baseline_no_context_phi3_latest | distinct2 | 1.0000 | (1.0000, 1.0000) |
| baseline_no_context_phi3_latest | content_distinct1 | 0.9990 | (0.9980, 0.9998) |
| baseline_no_context_phi3_latest | mtld | 134.3831 | (108.6304, 161.0611) |
| baseline_no_context_phi3_latest | repetition_penalty | 0.0209 | (0.0166, 0.0255) |
| baseline_no_context_phi3_latest | lexical_richness | 0.8376 | (0.8183, 0.8566) |

## Paired Deltas
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) |
|---|---|---:|---:|---:|
| proposed_vs_candidate_no_context | distinct1 | 0.0068 | (0.0004, 0.0132) | 0.0167 |
| proposed_vs_candidate_no_context | distinct2 | 0.0008 | (-0.0013, 0.0029) | 0.2500 |
| proposed_vs_candidate_no_context | content_distinct1 | 0.0024 | (-0.0033, 0.0080) | 0.2040 |
| proposed_vs_candidate_no_context | mtld | 16.6627 | (-22.0626, 57.5452) | 0.2037 |
| proposed_vs_candidate_no_context | repetition_penalty | -0.0068 | (-0.0136, -0.0003) | 0.9790 |
| proposed_vs_candidate_no_context | lexical_richness | -0.0014 | (-0.0223, 0.0183) | 0.5583 |
| proposed_vs_baseline_no_context | distinct1 | -0.0344 | (-0.0439, -0.0250) | 1.0000 |
| proposed_vs_baseline_no_context | distinct2 | -0.0088 | (-0.0119, -0.0058) | 1.0000 |
| proposed_vs_baseline_no_context | content_distinct1 | -0.0273 | (-0.0364, -0.0189) | 1.0000 |
| proposed_vs_baseline_no_context | mtld | 19.8546 | (-24.3622, 65.1740) | 0.1947 |
| proposed_vs_baseline_no_context | repetition_penalty | 0.0344 | (0.0249, 0.0438) | 0.0000 |
| proposed_vs_baseline_no_context | lexical_richness | -0.0014 | (-0.0279, 0.0242) | 0.5287 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | -0.0339 | (-0.0431, -0.0248) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | distinct2 | -0.0091 | (-0.0121, -0.0062) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | content_distinct1 | -0.0276 | (-0.0361, -0.0192) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | mtld | 39.0467 | (-1.2167, 77.8833) | 0.0280 |
| proposed_vs_baseline_no_context_phi3_latest | repetition_penalty | 0.0339 | (0.0249, 0.0436) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | lexical_richness | 0.0020 | (-0.0259, 0.0299) | 0.4380 |
| controlled_vs_proposed_raw | distinct1 | -0.0076 | (-0.0150, 0.0004) | 0.9690 |
| controlled_vs_proposed_raw | distinct2 | 0.0004 | (-0.0022, 0.0029) | 0.3767 |
| controlled_vs_proposed_raw | content_distinct1 | -0.0036 | (-0.0115, 0.0041) | 0.8213 |
| controlled_vs_proposed_raw | mtld | 7.6386 | (-30.4310, 44.4484) | 0.3510 |
| controlled_vs_proposed_raw | repetition_penalty | 0.0076 | (-0.0002, 0.0154) | 0.0287 |
| controlled_vs_proposed_raw | lexical_richness | 0.0276 | (0.0043, 0.0504) | 0.0107 |
| controlled_vs_candidate_no_context | distinct1 | -0.0008 | (-0.0094, 0.0078) | 0.5697 |
| controlled_vs_candidate_no_context | distinct2 | 0.0012 | (-0.0018, 0.0042) | 0.2237 |
| controlled_vs_candidate_no_context | content_distinct1 | -0.0012 | (-0.0099, 0.0077) | 0.5980 |
| controlled_vs_candidate_no_context | mtld | 24.3014 | (-19.7857, 68.2108) | 0.1443 |
| controlled_vs_candidate_no_context | repetition_penalty | 0.0008 | (-0.0080, 0.0093) | 0.4253 |
| controlled_vs_candidate_no_context | lexical_richness | 0.0262 | (0.0014, 0.0503) | 0.0190 |
| controlled_vs_baseline_no_context | distinct1 | -0.0421 | (-0.0507, -0.0335) | 1.0000 |
| controlled_vs_baseline_no_context | distinct2 | -0.0084 | (-0.0120, -0.0053) | 1.0000 |
| controlled_vs_baseline_no_context | content_distinct1 | -0.0309 | (-0.0400, -0.0225) | 1.0000 |
| controlled_vs_baseline_no_context | mtld | 27.4932 | (-17.4603, 75.4947) | 0.1123 |
| controlled_vs_baseline_no_context | repetition_penalty | 0.0421 | (0.0331, 0.0508) | 0.0000 |
| controlled_vs_baseline_no_context | lexical_richness | 0.0262 | (-0.0014, 0.0533) | 0.0307 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0416 | (-0.0505, -0.0328) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | distinct2 | -0.0087 | (-0.0123, -0.0055) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | content_distinct1 | -0.0312 | (-0.0407, -0.0225) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | mtld | 46.6853 | (4.4542, 87.6528) | 0.0143 |
| controlled_vs_baseline_no_context_phi3_latest | repetition_penalty | 0.0416 | (0.0331, 0.0502) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | lexical_richness | 0.0296 | (0.0014, 0.0586) | 0.0207 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0057 | (-0.0140, 0.0017) | 0.9230 |
| controlled_alt_vs_controlled_default | distinct2 | -0.0024 | (-0.0083, 0.0020) | 0.8290 |
| controlled_alt_vs_controlled_default | content_distinct1 | -0.0020 | (-0.0107, 0.0059) | 0.6900 |
| controlled_alt_vs_controlled_default | mtld | -26.6337 | (-61.4568, 7.7625) | 0.9353 |
| controlled_alt_vs_controlled_default | repetition_penalty | 0.0057 | (-0.0021, 0.0135) | 0.0777 |
| controlled_alt_vs_controlled_default | lexical_richness | -0.0122 | (-0.0358, 0.0102) | 0.8653 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0133 | (-0.0217, -0.0050) | 0.9997 |
| controlled_alt_vs_proposed_raw | distinct2 | -0.0020 | (-0.0079, 0.0021) | 0.7700 |
| controlled_alt_vs_proposed_raw | content_distinct1 | -0.0056 | (-0.0148, 0.0031) | 0.8920 |
| controlled_alt_vs_proposed_raw | mtld | -18.9950 | (-58.8803, 22.3584) | 0.8013 |
| controlled_alt_vs_proposed_raw | repetition_penalty | 0.0133 | (0.0050, 0.0225) | 0.0003 |
| controlled_alt_vs_proposed_raw | lexical_richness | 0.0154 | (-0.0074, 0.0407) | 0.0987 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0065 | (-0.0165, 0.0030) | 0.9047 |
| controlled_alt_vs_candidate_no_context | distinct2 | -0.0013 | (-0.0075, 0.0030) | 0.6633 |
| controlled_alt_vs_candidate_no_context | content_distinct1 | -0.0032 | (-0.0131, 0.0061) | 0.7420 |
| controlled_alt_vs_candidate_no_context | mtld | -2.3323 | (-44.5677, 35.3917) | 0.5550 |
| controlled_alt_vs_candidate_no_context | repetition_penalty | 0.0065 | (-0.0028, 0.0162) | 0.0940 |
| controlled_alt_vs_candidate_no_context | lexical_richness | 0.0140 | (-0.0098, 0.0376) | 0.1187 |
| controlled_alt_vs_baseline_no_context | distinct1 | -0.0477 | (-0.0577, -0.0385) | 1.0000 |
| controlled_alt_vs_baseline_no_context | distinct2 | -0.0109 | (-0.0168, -0.0064) | 1.0000 |
| controlled_alt_vs_baseline_no_context | content_distinct1 | -0.0330 | (-0.0431, -0.0236) | 1.0000 |
| controlled_alt_vs_baseline_no_context | mtld | 0.8595 | (-42.0286, 44.2727) | 0.4813 |
| controlled_alt_vs_baseline_no_context | repetition_penalty | 0.0477 | (0.0378, 0.0576) | 0.0000 |
| controlled_alt_vs_baseline_no_context | lexical_richness | 0.0140 | (-0.0136, 0.0408) | 0.1497 |
| controlled_alt_vs_baseline_no_context_phi3_latest | distinct1 | -0.0473 | (-0.0573, -0.0378) | 1.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | distinct2 | -0.0112 | (-0.0169, -0.0067) | 1.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | content_distinct1 | -0.0333 | (-0.0435, -0.0240) | 1.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | mtld | 20.0517 | (-18.0728, 56.8450) | 0.1553 |
| controlled_alt_vs_baseline_no_context_phi3_latest | repetition_penalty | 0.0473 | (0.0378, 0.0568) | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | lexical_richness | 0.0174 | (-0.0100, 0.0445) | 0.1097 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | distinct1 | -0.0477 | (-0.0572, -0.0380) | 1.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | distinct2 | -0.0109 | (-0.0169, -0.0063) | 1.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | content_distinct1 | -0.0330 | (-0.0427, -0.0235) | 1.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | mtld | 0.8595 | (-42.0198, 42.1892) | 0.4843 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | repetition_penalty | 0.0477 | (0.0383, 0.0578) | 0.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | lexical_richness | 0.0140 | (-0.0134, 0.0415) | 0.1537 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | distinct1 | -0.0473 | (-0.0570, -0.0375) | 1.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | distinct2 | -0.0112 | (-0.0170, -0.0069) | 1.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | content_distinct1 | -0.0333 | (-0.0434, -0.0238) | 1.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | mtld | 20.0517 | (-19.0934, 58.9047) | 0.1633 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | repetition_penalty | 0.0473 | (0.0375, 0.0574) | 0.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | lexical_richness | 0.0174 | (-0.0099, 0.0453) | 0.1020 |
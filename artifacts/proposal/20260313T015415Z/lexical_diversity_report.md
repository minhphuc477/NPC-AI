# Lexical Diversity Benchmark

- Run dir: `artifacts\proposal\20260313T015415Z`
- Scenario count: `144`

## Arm Summary
| Arm | Metric | Mean | 95% CI |
|---|---|---:|---:|
| proposed_contextual_controlled | distinct1 | 0.9333 | (0.9261, 0.9407) |
| proposed_contextual_controlled | distinct2 | 0.9902 | (0.9870, 0.9932) |
| proposed_contextual_controlled | content_distinct1 | 0.9668 | (0.9582, 0.9751) |
| proposed_contextual_controlled | mtld | 178.4184 | (147.1988, 213.2998) |
| proposed_contextual_controlled | repetition_penalty | 0.0667 | (0.0594, 0.0740) |
| proposed_contextual_controlled | lexical_richness | 0.8636 | (0.8430, 0.8830) |
| proposed_contextual_controlled_tuned | distinct1 | 0.9269 | (0.9188, 0.9351) |
| proposed_contextual_controlled_tuned | distinct2 | 0.9868 | (0.9814, 0.9912) |
| proposed_contextual_controlled_tuned | content_distinct1 | 0.9593 | (0.9493, 0.9686) |
| proposed_contextual_controlled_tuned | mtld | 153.1197 | (127.6909, 180.2686) |
| proposed_contextual_controlled_tuned | repetition_penalty | 0.0731 | (0.0650, 0.0813) |
| proposed_contextual_controlled_tuned | lexical_richness | 0.8448 | (0.8257, 0.8651) |
| proposed_contextual | distinct1 | 0.9392 | (0.9312, 0.9469) |
| proposed_contextual | distinct2 | 0.9891 | (0.9860, 0.9921) |
| proposed_contextual | content_distinct1 | 0.9659 | (0.9569, 0.9745) |
| proposed_contextual | mtld | 180.2939 | (147.6610, 215.0941) |
| proposed_contextual | repetition_penalty | 0.0608 | (0.0528, 0.0686) |
| proposed_contextual | lexical_richness | 0.8444 | (0.8252, 0.8651) |
| candidate_no_context | distinct1 | 0.9416 | (0.9323, 0.9498) |
| candidate_no_context | distinct2 | 0.9886 | (0.9826, 0.9932) |
| candidate_no_context | content_distinct1 | 0.9677 | (0.9576, 0.9768) |
| candidate_no_context | mtld | 184.6340 | (148.3012, 222.0379) |
| candidate_no_context | repetition_penalty | 0.0584 | (0.0495, 0.0675) |
| candidate_no_context | lexical_richness | 0.8374 | (0.8173, 0.8579) |
| baseline_no_context | distinct1 | 0.9786 | (0.9746, 0.9825) |
| baseline_no_context | distinct2 | 1.0000 | (1.0000, 1.0000) |
| baseline_no_context | content_distinct1 | 0.9977 | (0.9959, 0.9992) |
| baseline_no_context | mtld | 152.9783 | (124.2813, 183.6577) |
| baseline_no_context | repetition_penalty | 0.0214 | (0.0174, 0.0256) |
| baseline_no_context | lexical_richness | 0.8486 | (0.8295, 0.8684) |
| baseline_no_context_phi3_latest | distinct1 | 0.9799 | (0.9761, 0.9835) |
| baseline_no_context_phi3_latest | distinct2 | 1.0000 | (1.0000, 1.0000) |
| baseline_no_context_phi3_latest | content_distinct1 | 0.9986 | (0.9972, 0.9997) |
| baseline_no_context_phi3_latest | mtld | 176.1931 | (143.6296, 211.5036) |
| baseline_no_context_phi3_latest | repetition_penalty | 0.0201 | (0.0165, 0.0239) |
| baseline_no_context_phi3_latest | lexical_richness | 0.8560 | (0.8366, 0.8750) |

## Paired Deltas
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) |
|---|---|---:|---:|---:|
| proposed_vs_candidate_no_context | distinct1 | -0.0024 | (-0.0101, 0.0064) | 0.7347 |
| proposed_vs_candidate_no_context | distinct2 | 0.0004 | (-0.0038, 0.0061) | 0.4647 |
| proposed_vs_candidate_no_context | content_distinct1 | -0.0018 | (-0.0095, 0.0068) | 0.6640 |
| proposed_vs_candidate_no_context | mtld | -4.3401 | (-51.4244, 39.8311) | 0.5683 |
| proposed_vs_candidate_no_context | repetition_penalty | 0.0024 | (-0.0061, 0.0104) | 0.2840 |
| proposed_vs_candidate_no_context | lexical_richness | 0.0071 | (-0.0140, 0.0290) | 0.2520 |
| proposed_vs_baseline_no_context | distinct1 | -0.0394 | (-0.0482, -0.0303) | 1.0000 |
| proposed_vs_baseline_no_context | distinct2 | -0.0109 | (-0.0142, -0.0078) | 1.0000 |
| proposed_vs_baseline_no_context | content_distinct1 | -0.0317 | (-0.0410, -0.0230) | 1.0000 |
| proposed_vs_baseline_no_context | mtld | 27.3156 | (-18.2874, 72.8464) | 0.1083 |
| proposed_vs_baseline_no_context | repetition_penalty | 0.0394 | (0.0303, 0.0485) | 0.0000 |
| proposed_vs_baseline_no_context | lexical_richness | -0.0042 | (-0.0311, 0.0230) | 0.6180 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | -0.0407 | (-0.0497, -0.0318) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | distinct2 | -0.0109 | (-0.0142, -0.0078) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | content_distinct1 | -0.0327 | (-0.0418, -0.0239) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | mtld | 4.1009 | (-46.2083, 52.3024) | 0.4377 |
| proposed_vs_baseline_no_context_phi3_latest | repetition_penalty | 0.0407 | (0.0318, 0.0493) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | lexical_richness | -0.0115 | (-0.0378, 0.0154) | 0.7947 |
| controlled_vs_proposed_raw | distinct1 | -0.0059 | (-0.0141, 0.0022) | 0.9133 |
| controlled_vs_proposed_raw | distinct2 | 0.0012 | (-0.0016, 0.0039) | 0.2170 |
| controlled_vs_proposed_raw | content_distinct1 | 0.0009 | (-0.0069, 0.0090) | 0.4117 |
| controlled_vs_proposed_raw | mtld | -1.8756 | (-45.7834, 40.0668) | 0.5320 |
| controlled_vs_proposed_raw | repetition_penalty | 0.0059 | (-0.0021, 0.0140) | 0.0803 |
| controlled_vs_proposed_raw | lexical_richness | 0.0192 | (-0.0041, 0.0413) | 0.0543 |
| controlled_vs_candidate_no_context | distinct1 | -0.0083 | (-0.0178, 0.0021) | 0.9413 |
| controlled_vs_candidate_no_context | distinct2 | 0.0016 | (-0.0030, 0.0075) | 0.2973 |
| controlled_vs_candidate_no_context | content_distinct1 | -0.0009 | (-0.0101, 0.0086) | 0.5690 |
| controlled_vs_candidate_no_context | mtld | -6.2157 | (-48.1280, 36.0698) | 0.6090 |
| controlled_vs_candidate_no_context | repetition_penalty | 0.0083 | (-0.0022, 0.0177) | 0.0560 |
| controlled_vs_candidate_no_context | lexical_richness | 0.0263 | (0.0042, 0.0489) | 0.0090 |
| controlled_vs_baseline_no_context | distinct1 | -0.0452 | (-0.0536, -0.0368) | 1.0000 |
| controlled_vs_baseline_no_context | distinct2 | -0.0098 | (-0.0127, -0.0068) | 1.0000 |
| controlled_vs_baseline_no_context | content_distinct1 | -0.0309 | (-0.0397, -0.0224) | 1.0000 |
| controlled_vs_baseline_no_context | mtld | 25.4400 | (-16.3628, 69.8256) | 0.1153 |
| controlled_vs_baseline_no_context | repetition_penalty | 0.0452 | (0.0365, 0.0538) | 0.0000 |
| controlled_vs_baseline_no_context | lexical_richness | 0.0150 | (-0.0111, 0.0412) | 0.1390 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0465 | (-0.0546, -0.0377) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | distinct2 | -0.0098 | (-0.0128, -0.0068) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | content_distinct1 | -0.0318 | (-0.0406, -0.0236) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | mtld | 2.2253 | (-43.8024, 50.1915) | 0.4507 |
| controlled_vs_baseline_no_context_phi3_latest | repetition_penalty | 0.0465 | (0.0383, 0.0548) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | lexical_richness | 0.0077 | (-0.0198, 0.0356) | 0.2950 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0064 | (-0.0150, 0.0023) | 0.9290 |
| controlled_alt_vs_controlled_default | distinct2 | -0.0034 | (-0.0091, 0.0007) | 0.9360 |
| controlled_alt_vs_controlled_default | content_distinct1 | -0.0075 | (-0.0168, 0.0014) | 0.9477 |
| controlled_alt_vs_controlled_default | mtld | -25.2987 | (-67.3432, 14.6349) | 0.8833 |
| controlled_alt_vs_controlled_default | repetition_penalty | 0.0064 | (-0.0016, 0.0152) | 0.0623 |
| controlled_alt_vs_controlled_default | lexical_richness | -0.0188 | (-0.0422, 0.0044) | 0.9457 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0123 | (-0.0216, -0.0033) | 0.9973 |
| controlled_alt_vs_proposed_raw | distinct2 | -0.0023 | (-0.0079, 0.0019) | 0.8213 |
| controlled_alt_vs_proposed_raw | content_distinct1 | -0.0066 | (-0.0161, 0.0019) | 0.9307 |
| controlled_alt_vs_proposed_raw | mtld | -27.1743 | (-69.5856, 15.1816) | 0.8923 |
| controlled_alt_vs_proposed_raw | repetition_penalty | 0.0123 | (0.0031, 0.0218) | 0.0043 |
| controlled_alt_vs_proposed_raw | lexical_richness | 0.0004 | (-0.0237, 0.0244) | 0.4770 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0147 | (-0.0252, -0.0038) | 0.9977 |
| controlled_alt_vs_candidate_no_context | distinct2 | -0.0018 | (-0.0083, 0.0052) | 0.7210 |
| controlled_alt_vs_candidate_no_context | content_distinct1 | -0.0084 | (-0.0186, 0.0020) | 0.9443 |
| controlled_alt_vs_candidate_no_context | mtld | -31.5144 | (-76.0126, 12.8350) | 0.9150 |
| controlled_alt_vs_candidate_no_context | repetition_penalty | 0.0147 | (0.0042, 0.0250) | 0.0033 |
| controlled_alt_vs_candidate_no_context | lexical_richness | 0.0075 | (-0.0182, 0.0320) | 0.2937 |
| controlled_alt_vs_baseline_no_context | distinct1 | -0.0517 | (-0.0614, -0.0421) | 1.0000 |
| controlled_alt_vs_baseline_no_context | distinct2 | -0.0132 | (-0.0185, -0.0087) | 1.0000 |
| controlled_alt_vs_baseline_no_context | content_distinct1 | -0.0384 | (-0.0487, -0.0289) | 1.0000 |
| controlled_alt_vs_baseline_no_context | mtld | 0.1413 | (-38.4434, 39.8631) | 0.5010 |
| controlled_alt_vs_baseline_no_context | repetition_penalty | 0.0517 | (0.0424, 0.0612) | 0.0000 |
| controlled_alt_vs_baseline_no_context | lexical_richness | -0.0038 | (-0.0306, 0.0240) | 0.5973 |
| controlled_alt_vs_baseline_no_context_phi3_latest | distinct1 | -0.0530 | (-0.0625, -0.0440) | 1.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | distinct2 | -0.0132 | (-0.0189, -0.0088) | 1.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | content_distinct1 | -0.0393 | (-0.0493, -0.0300) | 1.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | mtld | -23.0734 | (-63.1544, 17.6021) | 0.8647 |
| controlled_alt_vs_baseline_no_context_phi3_latest | repetition_penalty | 0.0530 | (0.0437, 0.0625) | 0.0000 |
| controlled_alt_vs_baseline_no_context_phi3_latest | lexical_richness | -0.0111 | (-0.0380, 0.0172) | 0.7880 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | distinct1 | -0.0517 | (-0.0610, -0.0424) | 1.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | distinct2 | -0.0132 | (-0.0187, -0.0085) | 1.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | content_distinct1 | -0.0384 | (-0.0477, -0.0285) | 1.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | mtld | 0.1413 | (-39.6616, 41.0056) | 0.5053 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | repetition_penalty | 0.0517 | (0.0422, 0.0618) | 0.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context | lexical_richness | -0.0038 | (-0.0305, 0.0248) | 0.6073 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | distinct1 | -0.0530 | (-0.0624, -0.0437) | 1.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | distinct2 | -0.0132 | (-0.0189, -0.0088) | 1.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | content_distinct1 | -0.0393 | (-0.0493, -0.0299) | 1.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | mtld | -23.0734 | (-65.3723, 16.6775) | 0.8720 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | repetition_penalty | 0.0530 | (0.0437, 0.0627) | 0.0000 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | lexical_richness | -0.0111 | (-0.0392, 0.0164) | 0.7817 |
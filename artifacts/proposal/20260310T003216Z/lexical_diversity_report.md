# Lexical Diversity Benchmark

- Run dir: `artifacts\proposal\20260310T003216Z`
- Scenario count: `144`

## Arm Summary
| Arm | Metric | Mean | 95% CI |
|---|---|---:|---:|
| proposed_contextual_controlled | distinct1 | 0.9361 | (0.9309, 0.9412) |
| proposed_contextual_controlled | distinct2 | 0.9984 | (0.9970, 0.9995) |
| proposed_contextual_controlled | content_distinct1 | 0.9925 | (0.9889, 0.9955) |
| proposed_contextual_controlled | mtld | 188.3737 | (169.6552, 208.2801) |
| proposed_contextual_controlled | repetition_penalty | 0.0639 | (0.0589, 0.0692) |
| proposed_contextual_controlled | lexical_richness | 0.9253 | (0.9123, 0.9370) |
| proposed_contextual | distinct1 | 0.9405 | (0.9348, 0.9463) |
| proposed_contextual | distinct2 | 1.0000 | (1.0000, 1.0000) |
| proposed_contextual | content_distinct1 | 0.9970 | (0.9952, 0.9985) |
| proposed_contextual | mtld | 144.5453 | (114.4182, 177.8809) |
| proposed_contextual | repetition_penalty | 0.0595 | (0.0536, 0.0654) |
| proposed_contextual | lexical_richness | 0.8287 | (0.8103, 0.8473) |
| candidate_no_context | distinct1 | 0.9469 | (0.9401, 0.9534) |
| candidate_no_context | distinct2 | 0.9927 | (0.9785, 0.9999) |
| candidate_no_context | content_distinct1 | 0.9973 | (0.9955, 0.9986) |
| candidate_no_context | mtld | 127.7578 | (97.9216, 158.9254) |
| candidate_no_context | repetition_penalty | 0.0531 | (0.0464, 0.0596) |
| candidate_no_context | lexical_richness | 0.8142 | (0.7962, 0.8322) |
| baseline_no_context | distinct1 | 0.9776 | (0.9729, 0.9819) |
| baseline_no_context | distinct2 | 0.9999 | (0.9996, 1.0000) |
| baseline_no_context | content_distinct1 | 0.9989 | (0.9976, 0.9998) |
| baseline_no_context | mtld | 149.6611 | (123.7419, 177.8858) |
| baseline_no_context | repetition_penalty | 0.0224 | (0.0180, 0.0269) |
| baseline_no_context | lexical_richness | 0.8507 | (0.8307, 0.8703) |
| baseline_no_context_phi3_latest | distinct1 | 0.9814 | (0.9773, 0.9849) |
| baseline_no_context_phi3_latest | distinct2 | 1.0000 | (1.0000, 1.0000) |
| baseline_no_context_phi3_latest | content_distinct1 | 0.9986 | (0.9973, 0.9996) |
| baseline_no_context_phi3_latest | mtld | 163.6717 | (132.1846, 195.7647) |
| baseline_no_context_phi3_latest | repetition_penalty | 0.0186 | (0.0149, 0.0226) |
| baseline_no_context_phi3_latest | lexical_richness | 0.8507 | (0.8304, 0.8713) |

## Paired Deltas
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) |
|---|---|---:|---:|---:|
| proposed_vs_candidate_no_context | distinct1 | -0.0064 | (-0.0142, 0.0011) | 0.9520 |
| proposed_vs_candidate_no_context | distinct2 | 0.0073 | (0.0001, 0.0214) | 0.0170 |
| proposed_vs_candidate_no_context | content_distinct1 | -0.0003 | (-0.0024, 0.0019) | 0.5827 |
| proposed_vs_candidate_no_context | mtld | 16.7875 | (-22.4878, 54.4128) | 0.1970 |
| proposed_vs_candidate_no_context | repetition_penalty | 0.0064 | (-0.0009, 0.0140) | 0.0417 |
| proposed_vs_candidate_no_context | lexical_richness | 0.0145 | (-0.0088, 0.0373) | 0.1153 |
| proposed_vs_baseline_no_context | distinct1 | -0.0371 | (-0.0444, -0.0294) | 1.0000 |
| proposed_vs_baseline_no_context | distinct2 | 0.0001 | (0.0000, 0.0004) | 0.3730 |
| proposed_vs_baseline_no_context | content_distinct1 | -0.0019 | (-0.0039, 0.0001) | 0.9677 |
| proposed_vs_baseline_no_context | mtld | -5.1158 | (-47.7593, 38.4129) | 0.5897 |
| proposed_vs_baseline_no_context | repetition_penalty | 0.0371 | (0.0296, 0.0447) | 0.0000 |
| proposed_vs_baseline_no_context | lexical_richness | -0.0220 | (-0.0502, 0.0062) | 0.9250 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | -0.0409 | (-0.0477, -0.0338) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | distinct2 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | content_distinct1 | -0.0016 | (-0.0038, 0.0004) | 0.9403 |
| proposed_vs_baseline_no_context_phi3_latest | mtld | -19.1264 | (-61.6405, 22.7307) | 0.8213 |
| proposed_vs_baseline_no_context_phi3_latest | repetition_penalty | 0.0409 | (0.0340, 0.0479) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | lexical_richness | -0.0220 | (-0.0479, 0.0057) | 0.9390 |
| controlled_vs_proposed_raw | distinct1 | -0.0044 | (-0.0115, 0.0027) | 0.8813 |
| controlled_vs_proposed_raw | distinct2 | -0.0016 | (-0.0029, -0.0005) | 1.0000 |
| controlled_vs_proposed_raw | content_distinct1 | -0.0045 | (-0.0082, -0.0011) | 0.9967 |
| controlled_vs_proposed_raw | mtld | 43.8285 | (7.2963, 78.4853) | 0.0083 |
| controlled_vs_proposed_raw | repetition_penalty | 0.0044 | (-0.0027, 0.0114) | 0.1200 |
| controlled_vs_proposed_raw | lexical_richness | 0.0966 | (0.0774, 0.1168) | 0.0000 |
| controlled_vs_candidate_no_context | distinct1 | -0.0108 | (-0.0188, -0.0023) | 0.9950 |
| controlled_vs_candidate_no_context | distinct2 | 0.0057 | (-0.0022, 0.0204) | 0.3710 |
| controlled_vs_candidate_no_context | content_distinct1 | -0.0047 | (-0.0084, -0.0013) | 0.9963 |
| controlled_vs_candidate_no_context | mtld | 60.6159 | (26.0526, 93.4840) | 0.0010 |
| controlled_vs_candidate_no_context | repetition_penalty | 0.0108 | (0.0021, 0.0192) | 0.0067 |
| controlled_vs_candidate_no_context | lexical_richness | 0.1111 | (0.0914, 0.1300) | 0.0000 |
| controlled_vs_baseline_no_context | distinct1 | -0.0415 | (-0.0482, -0.0349) | 1.0000 |
| controlled_vs_baseline_no_context | distinct2 | -0.0014 | (-0.0028, -0.0003) | 0.9960 |
| controlled_vs_baseline_no_context | content_distinct1 | -0.0064 | (-0.0099, -0.0031) | 1.0000 |
| controlled_vs_baseline_no_context | mtld | 38.7126 | (4.9617, 71.1996) | 0.0103 |
| controlled_vs_baseline_no_context | repetition_penalty | 0.0415 | (0.0349, 0.0479) | 0.0000 |
| controlled_vs_baseline_no_context | lexical_richness | 0.0746 | (0.0533, 0.0965) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0453 | (-0.0512, -0.0391) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | distinct2 | -0.0016 | (-0.0029, -0.0005) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | content_distinct1 | -0.0060 | (-0.0097, -0.0026) | 0.9997 |
| controlled_vs_baseline_no_context_phi3_latest | mtld | 24.7020 | (-11.9239, 62.3039) | 0.0953 |
| controlled_vs_baseline_no_context_phi3_latest | repetition_penalty | 0.0453 | (0.0389, 0.0516) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | lexical_richness | 0.0746 | (0.0534, 0.0970) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0415 | (-0.0481, -0.0346) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct2 | -0.0014 | (-0.0027, -0.0003) | 0.9947 |
| proposed_contextual_controlled_vs_baseline_no_context | content_distinct1 | -0.0064 | (-0.0101, -0.0031) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | mtld | 38.7126 | (4.7442, 70.9693) | 0.0100 |
| proposed_contextual_controlled_vs_baseline_no_context | repetition_penalty | 0.0415 | (0.0348, 0.0483) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | lexical_richness | 0.0746 | (0.0539, 0.0958) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0453 | (-0.0517, -0.0390) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct2 | -0.0016 | (-0.0030, -0.0005) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | content_distinct1 | -0.0060 | (-0.0097, -0.0027) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | mtld | 24.7020 | (-13.4722, 60.6184) | 0.0993 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | repetition_penalty | 0.0453 | (0.0390, 0.0517) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | lexical_richness | 0.0746 | (0.0519, 0.0968) | 0.0000 |
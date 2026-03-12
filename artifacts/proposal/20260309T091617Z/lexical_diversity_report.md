# Lexical Diversity Benchmark

- Run dir: `artifacts\proposal\20260309T091617Z`
- Scenario count: `144`

## Arm Summary
| Arm | Metric | Mean | 95% CI |
|---|---|---:|---:|
| proposed_contextual_controlled | distinct1 | 0.9382 | (0.9329, 0.9434) |
| proposed_contextual_controlled | distinct2 | 0.9987 | (0.9978, 0.9995) |
| proposed_contextual_controlled | content_distinct1 | 0.9926 | (0.9895, 0.9955) |
| proposed_contextual_controlled | mtld | 208.6611 | (183.7609, 234.8992) |
| proposed_contextual_controlled | repetition_penalty | 0.0618 | (0.0569, 0.0671) |
| proposed_contextual_controlled | lexical_richness | 0.9271 | (0.9151, 0.9395) |
| proposed_contextual | distinct1 | 0.9393 | (0.9336, 0.9451) |
| proposed_contextual | distinct2 | 0.9997 | (0.9993, 1.0000) |
| proposed_contextual | content_distinct1 | 0.9974 | (0.9958, 0.9988) |
| proposed_contextual | mtld | 131.9234 | (108.2081, 157.2583) |
| proposed_contextual | repetition_penalty | 0.0607 | (0.0547, 0.0664) |
| proposed_contextual | lexical_richness | 0.8334 | (0.8150, 0.8523) |
| candidate_no_context | distinct1 | 0.9414 | (0.9355, 0.9474) |
| candidate_no_context | distinct2 | 1.0000 | (1.0000, 1.0000) |
| candidate_no_context | content_distinct1 | 0.9965 | (0.9946, 0.9981) |
| candidate_no_context | mtld | 143.6624 | (115.7602, 174.9783) |
| candidate_no_context | repetition_penalty | 0.0586 | (0.0525, 0.0642) |
| candidate_no_context | lexical_richness | 0.8327 | (0.8141, 0.8527) |
| baseline_no_context | distinct1 | 0.9829 | (0.9790, 0.9869) |
| baseline_no_context | distinct2 | 1.0000 | (1.0000, 1.0000) |
| baseline_no_context | content_distinct1 | 0.9992 | (0.9984, 0.9998) |
| baseline_no_context | mtld | 131.2995 | (102.9031, 163.4542) |
| baseline_no_context | repetition_penalty | 0.0171 | (0.0130, 0.0212) |
| baseline_no_context | lexical_richness | 0.8286 | (0.8103, 0.8476) |
| baseline_no_context_phi3_latest | distinct1 | 0.9773 | (0.9728, 0.9816) |
| baseline_no_context_phi3_latest | distinct2 | 0.9997 | (0.9993, 1.0000) |
| baseline_no_context_phi3_latest | content_distinct1 | 0.9985 | (0.9971, 0.9995) |
| baseline_no_context_phi3_latest | mtld | 151.3699 | (123.0892, 183.6321) |
| baseline_no_context_phi3_latest | repetition_penalty | 0.0227 | (0.0184, 0.0273) |
| baseline_no_context_phi3_latest | lexical_richness | 0.8462 | (0.8270, 0.8663) |

## Paired Deltas
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) |
|---|---|---:|---:|---:|
| proposed_vs_candidate_no_context | distinct1 | -0.0021 | (-0.0086, 0.0044) | 0.7517 |
| proposed_vs_candidate_no_context | distinct2 | -0.0003 | (-0.0007, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | content_distinct1 | 0.0009 | (-0.0011, 0.0030) | 0.1973 |
| proposed_vs_candidate_no_context | mtld | -11.7391 | (-45.8350, 22.3235) | 0.7437 |
| proposed_vs_candidate_no_context | repetition_penalty | 0.0021 | (-0.0041, 0.0082) | 0.2567 |
| proposed_vs_candidate_no_context | lexical_richness | 0.0008 | (-0.0237, 0.0249) | 0.4677 |
| proposed_vs_baseline_no_context | distinct1 | -0.0437 | (-0.0506, -0.0365) | 1.0000 |
| proposed_vs_baseline_no_context | distinct2 | -0.0003 | (-0.0007, 0.0000) | 1.0000 |
| proposed_vs_baseline_no_context | content_distinct1 | -0.0018 | (-0.0036, -0.0002) | 0.9880 |
| proposed_vs_baseline_no_context | mtld | 0.6238 | (-35.1252, 37.1430) | 0.4747 |
| proposed_vs_baseline_no_context | repetition_penalty | 0.0437 | (0.0364, 0.0508) | 0.0000 |
| proposed_vs_baseline_no_context | lexical_richness | 0.0049 | (-0.0207, 0.0296) | 0.3460 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | -0.0380 | (-0.0453, -0.0306) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | distinct2 | 0.0000 | (-0.0005, 0.0006) | 0.4693 |
| proposed_vs_baseline_no_context_phi3_latest | content_distinct1 | -0.0011 | (-0.0030, 0.0007) | 0.8843 |
| proposed_vs_baseline_no_context_phi3_latest | mtld | -19.4465 | (-59.5631, 19.4329) | 0.8330 |
| proposed_vs_baseline_no_context_phi3_latest | repetition_penalty | 0.0380 | (0.0304, 0.0453) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | lexical_richness | -0.0128 | (-0.0388, 0.0154) | 0.8253 |
| controlled_vs_proposed_raw | distinct1 | -0.0011 | (-0.0085, 0.0063) | 0.6087 |
| controlled_vs_proposed_raw | distinct2 | -0.0010 | (-0.0020, -0.0002) | 0.9950 |
| controlled_vs_proposed_raw | content_distinct1 | -0.0047 | (-0.0082, -0.0017) | 0.9993 |
| controlled_vs_proposed_raw | mtld | 76.7377 | (41.8933, 110.4868) | 0.0000 |
| controlled_vs_proposed_raw | repetition_penalty | 0.0011 | (-0.0066, 0.0085) | 0.4000 |
| controlled_vs_proposed_raw | lexical_richness | 0.0937 | (0.0734, 0.1140) | 0.0000 |
| controlled_vs_candidate_no_context | distinct1 | -0.0032 | (-0.0109, 0.0043) | 0.8003 |
| controlled_vs_candidate_no_context | distinct2 | -0.0013 | (-0.0023, -0.0005) | 1.0000 |
| controlled_vs_candidate_no_context | content_distinct1 | -0.0038 | (-0.0072, -0.0006) | 0.9930 |
| controlled_vs_candidate_no_context | mtld | 64.9986 | (24.3892, 105.4776) | 0.0017 |
| controlled_vs_candidate_no_context | repetition_penalty | 0.0032 | (-0.0043, 0.0105) | 0.2047 |
| controlled_vs_candidate_no_context | lexical_richness | 0.0944 | (0.0711, 0.1169) | 0.0000 |
| controlled_vs_baseline_no_context | distinct1 | -0.0448 | (-0.0510, -0.0382) | 1.0000 |
| controlled_vs_baseline_no_context | distinct2 | -0.0013 | (-0.0023, -0.0005) | 1.0000 |
| controlled_vs_baseline_no_context | content_distinct1 | -0.0066 | (-0.0097, -0.0037) | 1.0000 |
| controlled_vs_baseline_no_context | mtld | 77.3615 | (35.7908, 117.3409) | 0.0000 |
| controlled_vs_baseline_no_context | repetition_penalty | 0.0448 | (0.0385, 0.0507) | 0.0000 |
| controlled_vs_baseline_no_context | lexical_richness | 0.0985 | (0.0770, 0.1206) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0391 | (-0.0457, -0.0327) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | distinct2 | -0.0010 | (-0.0020, -0.0001) | 0.9857 |
| controlled_vs_baseline_no_context_phi3_latest | content_distinct1 | -0.0059 | (-0.0092, -0.0027) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | mtld | 57.2912 | (14.1558, 98.4647) | 0.0040 |
| controlled_vs_baseline_no_context_phi3_latest | repetition_penalty | 0.0391 | (0.0326, 0.0453) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | lexical_richness | 0.0809 | (0.0579, 0.1054) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0448 | (-0.0507, -0.0385) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct2 | -0.0013 | (-0.0023, -0.0005) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | content_distinct1 | -0.0066 | (-0.0097, -0.0036) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | mtld | 77.3615 | (36.4524, 118.1037) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | repetition_penalty | 0.0448 | (0.0384, 0.0512) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | lexical_richness | 0.0985 | (0.0776, 0.1190) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0391 | (-0.0455, -0.0328) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct2 | -0.0010 | (-0.0021, -0.0001) | 0.9857 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | content_distinct1 | -0.0059 | (-0.0091, -0.0029) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | mtld | 57.2912 | (15.2595, 100.2419) | 0.0043 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | repetition_penalty | 0.0391 | (0.0329, 0.0452) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | lexical_richness | 0.0809 | (0.0582, 0.1038) | 0.0000 |
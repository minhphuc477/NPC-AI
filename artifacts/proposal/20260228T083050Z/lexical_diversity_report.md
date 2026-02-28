# Lexical Diversity Benchmark

- Run dir: `artifacts\proposal\20260228T083050Z`
- Scenario count: `112`

## Arm Summary
| Arm | Metric | Mean | 95% CI |
|---|---|---:|---:|
| proposed_contextual_controlled | distinct1 | 0.9414 | (0.9372, 0.9455) |
| proposed_contextual_controlled | distinct2 | 0.9997 | (0.9992, 1.0000) |
| proposed_contextual_controlled | content_distinct1 | 0.9905 | (0.9870, 0.9939) |
| proposed_contextual_controlled | mtld | 217.2055 | (187.2720, 251.1675) |
| proposed_contextual_controlled | repetition_penalty | 0.0586 | (0.0544, 0.0628) |
| proposed_contextual_controlled | lexical_richness | 0.9304 | (0.9185, 0.9425) |
| proposed_contextual | distinct1 | 0.9402 | (0.9335, 0.9471) |
| proposed_contextual | distinct2 | 1.0000 | (1.0000, 1.0000) |
| proposed_contextual | content_distinct1 | 0.9972 | (0.9948, 0.9990) |
| proposed_contextual | mtld | 121.5120 | (92.4602, 155.1460) |
| proposed_contextual | repetition_penalty | 0.0598 | (0.0531, 0.0664) |
| proposed_contextual | lexical_richness | 0.8112 | (0.7907, 0.8315) |
| candidate_no_context | distinct1 | 0.9337 | (0.9273, 0.9405) |
| candidate_no_context | distinct2 | 0.9997 | (0.9991, 1.0000) |
| candidate_no_context | content_distinct1 | 0.9961 | (0.9932, 0.9984) |
| candidate_no_context | mtld | 93.2841 | (68.6107, 120.5696) |
| candidate_no_context | repetition_penalty | 0.0663 | (0.0598, 0.0728) |
| candidate_no_context | lexical_richness | 0.7975 | (0.7799, 0.8167) |
| baseline_no_context | distinct1 | 0.9848 | (0.9806, 0.9887) |
| baseline_no_context | distinct2 | 0.9998 | (0.9995, 1.0000) |
| baseline_no_context | content_distinct1 | 0.9981 | (0.9962, 0.9996) |
| baseline_no_context | mtld | 155.6537 | (116.8419, 198.4919) |
| baseline_no_context | repetition_penalty | 0.0152 | (0.0113, 0.0194) |
| baseline_no_context | lexical_richness | 0.8286 | (0.8079, 0.8501) |
| baseline_no_context_phi3_latest | distinct1 | 0.9787 | (0.9741, 0.9832) |
| baseline_no_context_phi3_latest | distinct2 | 0.9998 | (0.9995, 1.0000) |
| baseline_no_context_phi3_latest | content_distinct1 | 0.9978 | (0.9959, 0.9994) |
| baseline_no_context_phi3_latest | mtld | 168.8988 | (131.2646, 208.5328) |
| baseline_no_context_phi3_latest | repetition_penalty | 0.0213 | (0.0169, 0.0257) |
| baseline_no_context_phi3_latest | lexical_richness | 0.8555 | (0.8330, 0.8775) |

## Paired Deltas
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) |
|---|---|---:|---:|---:|
| proposed_vs_candidate_no_context | distinct1 | 0.0065 | (-0.0011, 0.0142) | 0.0503 |
| proposed_vs_candidate_no_context | distinct2 | 0.0003 | (0.0000, 0.0009) | 0.1337 |
| proposed_vs_candidate_no_context | content_distinct1 | 0.0012 | (-0.0022, 0.0045) | 0.2357 |
| proposed_vs_candidate_no_context | mtld | 28.2279 | (-8.6544, 66.2503) | 0.0723 |
| proposed_vs_candidate_no_context | repetition_penalty | -0.0065 | (-0.0137, 0.0008) | 0.9590 |
| proposed_vs_candidate_no_context | lexical_richness | 0.0137 | (-0.0099, 0.0380) | 0.1333 |
| proposed_vs_baseline_no_context | distinct1 | -0.0447 | (-0.0526, -0.0363) | 1.0000 |
| proposed_vs_baseline_no_context | distinct2 | 0.0002 | (0.0000, 0.0005) | 0.3750 |
| proposed_vs_baseline_no_context | content_distinct1 | -0.0009 | (-0.0036, 0.0019) | 0.7280 |
| proposed_vs_baseline_no_context | mtld | -34.1417 | (-86.5629, 18.3638) | 0.8950 |
| proposed_vs_baseline_no_context | repetition_penalty | 0.0447 | (0.0366, 0.0527) | 0.0000 |
| proposed_vs_baseline_no_context | lexical_richness | -0.0175 | (-0.0463, 0.0106) | 0.8843 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | -0.0386 | (-0.0467, -0.0303) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | distinct2 | 0.0002 | (0.0000, 0.0005) | 0.3627 |
| proposed_vs_baseline_no_context_phi3_latest | content_distinct1 | -0.0006 | (-0.0034, 0.0021) | 0.6643 |
| proposed_vs_baseline_no_context_phi3_latest | mtld | -47.3869 | (-95.4415, -2.0253) | 0.9783 |
| proposed_vs_baseline_no_context_phi3_latest | repetition_penalty | 0.0386 | (0.0302, 0.0468) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | lexical_richness | -0.0443 | (-0.0741, -0.0152) | 0.9980 |
| controlled_vs_proposed_raw | distinct1 | 0.0012 | (-0.0068, 0.0091) | 0.3900 |
| controlled_vs_proposed_raw | distinct2 | -0.0003 | (-0.0008, 0.0000) | 1.0000 |
| controlled_vs_proposed_raw | content_distinct1 | -0.0067 | (-0.0110, -0.0026) | 0.9993 |
| controlled_vs_proposed_raw | mtld | 95.6936 | (50.7461, 142.2780) | 0.0000 |
| controlled_vs_proposed_raw | repetition_penalty | -0.0012 | (-0.0091, 0.0068) | 0.6083 |
| controlled_vs_proposed_raw | lexical_richness | 0.1192 | (0.0953, 0.1415) | 0.0000 |
| controlled_vs_candidate_no_context | distinct1 | 0.0077 | (-0.0004, 0.0155) | 0.0310 |
| controlled_vs_candidate_no_context | distinct2 | 0.0000 | (-0.0006, 0.0007) | 0.4860 |
| controlled_vs_candidate_no_context | content_distinct1 | -0.0055 | (-0.0096, -0.0013) | 0.9950 |
| controlled_vs_candidate_no_context | mtld | 123.9214 | (84.3425, 165.4071) | 0.0000 |
| controlled_vs_candidate_no_context | repetition_penalty | -0.0077 | (-0.0156, 0.0005) | 0.9717 |
| controlled_vs_candidate_no_context | lexical_richness | 0.1329 | (0.1126, 0.1522) | 0.0000 |
| controlled_vs_baseline_no_context | distinct1 | -0.0435 | (-0.0494, -0.0374) | 1.0000 |
| controlled_vs_baseline_no_context | distinct2 | -0.0002 | (-0.0008, 0.0003) | 0.7160 |
| controlled_vs_baseline_no_context | content_distinct1 | -0.0076 | (-0.0116, -0.0037) | 1.0000 |
| controlled_vs_baseline_no_context | mtld | 61.5519 | (15.4860, 108.9583) | 0.0040 |
| controlled_vs_baseline_no_context | repetition_penalty | 0.0435 | (0.0372, 0.0492) | 0.0000 |
| controlled_vs_baseline_no_context | lexical_richness | 0.1017 | (0.0810, 0.1235) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0373 | (-0.0437, -0.0308) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | distinct2 | -0.0002 | (-0.0006, 0.0003) | 0.7250 |
| controlled_vs_baseline_no_context_phi3_latest | content_distinct1 | -0.0073 | (-0.0114, -0.0033) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | mtld | 48.3067 | (-0.9597, 99.3862) | 0.0277 |
| controlled_vs_baseline_no_context_phi3_latest | repetition_penalty | 0.0373 | (0.0312, 0.0436) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | lexical_richness | 0.0749 | (0.0512, 0.0986) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0435 | (-0.0497, -0.0372) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct2 | -0.0002 | (-0.0008, 0.0003) | 0.7193 |
| proposed_contextual_controlled_vs_baseline_no_context | content_distinct1 | -0.0076 | (-0.0115, -0.0039) | 0.9997 |
| proposed_contextual_controlled_vs_baseline_no_context | mtld | 61.5519 | (14.3055, 106.5602) | 0.0050 |
| proposed_contextual_controlled_vs_baseline_no_context | repetition_penalty | 0.0435 | (0.0377, 0.0492) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | lexical_richness | 0.1017 | (0.0799, 0.1234) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0373 | (-0.0437, -0.0307) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct2 | -0.0002 | (-0.0008, 0.0003) | 0.7287 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | content_distinct1 | -0.0073 | (-0.0116, -0.0035) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | mtld | 48.3067 | (-2.6266, 97.9826) | 0.0287 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | repetition_penalty | 0.0373 | (0.0308, 0.0436) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | lexical_richness | 0.0749 | (0.0497, 0.0993) | 0.0000 |
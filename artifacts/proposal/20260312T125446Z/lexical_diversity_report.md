# Lexical Diversity Benchmark

- Run dir: `artifacts\proposal\20260312T125446Z`
- Scenario count: `144`

## Arm Summary
| Arm | Metric | Mean | 95% CI |
|---|---|---:|---:|
| proposed_contextual_controlled | distinct1 | 0.9333 | (0.9263, 0.9405) |
| proposed_contextual_controlled | distinct2 | 0.9902 | (0.9869, 0.9932) |
| proposed_contextual_controlled | content_distinct1 | 0.9669 | (0.9578, 0.9755) |
| proposed_contextual_controlled | mtld | 173.9653 | (145.8853, 206.3770) |
| proposed_contextual_controlled | repetition_penalty | 0.0667 | (0.0598, 0.0738) |
| proposed_contextual_controlled | lexical_richness | 0.8672 | (0.8467, 0.8864) |
| proposed_contextual | distinct1 | 0.9379 | (0.9299, 0.9460) |
| proposed_contextual | distinct2 | 0.9911 | (0.9879, 0.9940) |
| proposed_contextual | content_distinct1 | 0.9678 | (0.9587, 0.9763) |
| proposed_contextual | mtld | 125.7344 | (103.2867, 151.9183) |
| proposed_contextual | repetition_penalty | 0.0621 | (0.0543, 0.0699) |
| proposed_contextual | lexical_richness | 0.8257 | (0.8075, 0.8443) |
| candidate_no_context | distinct1 | 0.9424 | (0.9347, 0.9502) |
| candidate_no_context | distinct2 | 0.9918 | (0.9889, 0.9946) |
| candidate_no_context | content_distinct1 | 0.9697 | (0.9614, 0.9777) |
| candidate_no_context | mtld | 160.3677 | (131.1834, 190.8995) |
| candidate_no_context | repetition_penalty | 0.0576 | (0.0503, 0.0655) |
| candidate_no_context | lexical_richness | 0.8423 | (0.8237, 0.8615) |
| baseline_no_context | distinct1 | 0.9788 | (0.9739, 0.9834) |
| baseline_no_context | distinct2 | 0.9997 | (0.9993, 1.0000) |
| baseline_no_context | content_distinct1 | 0.9971 | (0.9950, 0.9989) |
| baseline_no_context | mtld | 143.5337 | (114.7378, 174.3436) |
| baseline_no_context | repetition_penalty | 0.0212 | (0.0168, 0.0260) |
| baseline_no_context | lexical_richness | 0.8384 | (0.8196, 0.8577) |
| baseline_no_context_phi3_latest | distinct1 | 0.9828 | (0.9785, 0.9866) |
| baseline_no_context_phi3_latest | distinct2 | 0.9999 | (0.9997, 1.0000) |
| baseline_no_context_phi3_latest | content_distinct1 | 0.9979 | (0.9963, 0.9992) |
| baseline_no_context_phi3_latest | mtld | 157.7002 | (125.3271, 194.0888) |
| baseline_no_context_phi3_latest | repetition_penalty | 0.0172 | (0.0133, 0.0212) |
| baseline_no_context_phi3_latest | lexical_richness | 0.8368 | (0.8181, 0.8564) |

## Paired Deltas
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) |
|---|---|---:|---:|---:|
| proposed_vs_candidate_no_context | distinct1 | -0.0045 | (-0.0125, 0.0035) | 0.8590 |
| proposed_vs_candidate_no_context | distinct2 | -0.0006 | (-0.0028, 0.0014) | 0.7260 |
| proposed_vs_candidate_no_context | content_distinct1 | -0.0019 | (-0.0091, 0.0050) | 0.7153 |
| proposed_vs_candidate_no_context | mtld | -34.6333 | (-72.7349, 3.7041) | 0.9630 |
| proposed_vs_candidate_no_context | repetition_penalty | 0.0045 | (-0.0035, 0.0128) | 0.1500 |
| proposed_vs_candidate_no_context | lexical_richness | -0.0165 | (-0.0386, 0.0058) | 0.9227 |
| proposed_vs_baseline_no_context | distinct1 | -0.0409 | (-0.0503, -0.0312) | 1.0000 |
| proposed_vs_baseline_no_context | distinct2 | -0.0086 | (-0.0118, -0.0057) | 1.0000 |
| proposed_vs_baseline_no_context | content_distinct1 | -0.0293 | (-0.0388, -0.0207) | 1.0000 |
| proposed_vs_baseline_no_context | mtld | -17.7992 | (-50.3470, 14.8714) | 0.8593 |
| proposed_vs_baseline_no_context | repetition_penalty | 0.0409 | (0.0311, 0.0500) | 0.0000 |
| proposed_vs_baseline_no_context | lexical_richness | -0.0127 | (-0.0351, 0.0100) | 0.8660 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | -0.0449 | (-0.0538, -0.0359) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | distinct2 | -0.0088 | (-0.0119, -0.0058) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | content_distinct1 | -0.0300 | (-0.0387, -0.0213) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | mtld | -31.9658 | (-71.1119, 7.6529) | 0.9383 |
| proposed_vs_baseline_no_context_phi3_latest | repetition_penalty | 0.0449 | (0.0357, 0.0534) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | lexical_richness | -0.0111 | (-0.0371, 0.0130) | 0.8023 |
| controlled_vs_proposed_raw | distinct1 | -0.0046 | (-0.0129, 0.0031) | 0.8710 |
| controlled_vs_proposed_raw | distinct2 | -0.0009 | (-0.0035, 0.0016) | 0.7407 |
| controlled_vs_proposed_raw | content_distinct1 | -0.0010 | (-0.0091, 0.0064) | 0.6053 |
| controlled_vs_proposed_raw | mtld | 48.2309 | (12.7616, 83.5391) | 0.0047 |
| controlled_vs_proposed_raw | repetition_penalty | 0.0046 | (-0.0030, 0.0126) | 0.1190 |
| controlled_vs_proposed_raw | lexical_richness | 0.0415 | (0.0208, 0.0629) | 0.0000 |
| controlled_vs_candidate_no_context | distinct1 | -0.0091 | (-0.0176, -0.0006) | 0.9830 |
| controlled_vs_candidate_no_context | distinct2 | -0.0015 | (-0.0046, 0.0013) | 0.8460 |
| controlled_vs_candidate_no_context | content_distinct1 | -0.0029 | (-0.0117, 0.0056) | 0.7450 |
| controlled_vs_candidate_no_context | mtld | 13.5976 | (-28.8883, 57.9986) | 0.2600 |
| controlled_vs_candidate_no_context | repetition_penalty | 0.0091 | (0.0007, 0.0172) | 0.0170 |
| controlled_vs_candidate_no_context | lexical_richness | 0.0250 | (0.0010, 0.0483) | 0.0200 |
| controlled_vs_baseline_no_context | distinct1 | -0.0455 | (-0.0539, -0.0375) | 1.0000 |
| controlled_vs_baseline_no_context | distinct2 | -0.0095 | (-0.0128, -0.0064) | 1.0000 |
| controlled_vs_baseline_no_context | content_distinct1 | -0.0303 | (-0.0395, -0.0214) | 1.0000 |
| controlled_vs_baseline_no_context | mtld | 30.4317 | (-8.4955, 70.2450) | 0.0567 |
| controlled_vs_baseline_no_context | repetition_penalty | 0.0455 | (0.0373, 0.0535) | 0.0000 |
| controlled_vs_baseline_no_context | lexical_richness | 0.0288 | (0.0027, 0.0552) | 0.0147 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0495 | (-0.0574, -0.0415) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | distinct2 | -0.0097 | (-0.0129, -0.0065) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | content_distinct1 | -0.0310 | (-0.0402, -0.0221) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | mtld | 16.2651 | (-29.3868, 63.4758) | 0.2500 |
| controlled_vs_baseline_no_context_phi3_latest | repetition_penalty | 0.0495 | (0.0413, 0.0578) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | lexical_richness | 0.0304 | (0.0032, 0.0583) | 0.0157 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0455 | (-0.0536, -0.0373) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct2 | -0.0095 | (-0.0127, -0.0064) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | content_distinct1 | -0.0303 | (-0.0395, -0.0213) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | mtld | 30.4317 | (-9.6690, 69.0010) | 0.0733 |
| proposed_contextual_controlled_vs_baseline_no_context | repetition_penalty | 0.0455 | (0.0372, 0.0538) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | lexical_richness | 0.0288 | (0.0020, 0.0552) | 0.0193 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0495 | (-0.0577, -0.0414) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct2 | -0.0097 | (-0.0130, -0.0067) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | content_distinct1 | -0.0310 | (-0.0403, -0.0224) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | mtld | 16.2651 | (-28.3280, 61.9546) | 0.2437 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | repetition_penalty | 0.0495 | (0.0419, 0.0576) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | lexical_richness | 0.0304 | (0.0016, 0.0580) | 0.0200 |
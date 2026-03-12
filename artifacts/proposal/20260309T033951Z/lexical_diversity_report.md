# Lexical Diversity Benchmark

- Run dir: `artifacts\proposal\20260309T033951Z`
- Scenario count: `144`

## Arm Summary
| Arm | Metric | Mean | 95% CI |
|---|---|---:|---:|
| proposed_contextual_controlled | distinct1 | 0.9306 | (0.9252, 0.9357) |
| proposed_contextual_controlled | distinct2 | 0.9970 | (0.9954, 0.9984) |
| proposed_contextual_controlled | content_distinct1 | 0.9911 | (0.9876, 0.9942) |
| proposed_contextual_controlled | mtld | 166.8333 | (152.5053, 182.9923) |
| proposed_contextual_controlled | repetition_penalty | 0.0694 | (0.0643, 0.0746) |
| proposed_contextual_controlled | lexical_richness | 0.9199 | (0.9081, 0.9322) |
| proposed_contextual | distinct1 | 0.9397 | (0.9332, 0.9459) |
| proposed_contextual | distinct2 | 0.9996 | (0.9991, 1.0000) |
| proposed_contextual | content_distinct1 | 0.9973 | (0.9955, 0.9988) |
| proposed_contextual | mtld | 147.6514 | (115.4721, 184.0737) |
| proposed_contextual | repetition_penalty | 0.0603 | (0.0539, 0.0665) |
| proposed_contextual | lexical_richness | 0.8258 | (0.8079, 0.8446) |
| candidate_no_context | distinct1 | 0.9437 | (0.9375, 0.9498) |
| candidate_no_context | distinct2 | 0.9986 | (0.9965, 0.9999) |
| candidate_no_context | content_distinct1 | 0.9962 | (0.9930, 0.9984) |
| candidate_no_context | mtld | 154.6050 | (122.4778, 188.5910) |
| candidate_no_context | repetition_penalty | 0.0563 | (0.0500, 0.0623) |
| candidate_no_context | lexical_richness | 0.8318 | (0.8130, 0.8513) |
| baseline_no_context | distinct1 | 0.9789 | (0.9745, 0.9830) |
| baseline_no_context | distinct2 | 1.0000 | (1.0000, 1.0000) |
| baseline_no_context | content_distinct1 | 0.9985 | (0.9971, 0.9996) |
| baseline_no_context | mtld | 133.9290 | (107.3285, 163.1110) |
| baseline_no_context | repetition_penalty | 0.0211 | (0.0170, 0.0253) |
| baseline_no_context | lexical_richness | 0.8387 | (0.8199, 0.8579) |
| baseline_no_context_phi3_latest | distinct1 | 0.9778 | (0.9733, 0.9820) |
| baseline_no_context_phi3_latest | distinct2 | 0.9999 | (0.9996, 1.0000) |
| baseline_no_context_phi3_latest | content_distinct1 | 0.9970 | (0.9951, 0.9987) |
| baseline_no_context_phi3_latest | mtld | 144.8760 | (118.1370, 173.9676) |
| baseline_no_context_phi3_latest | repetition_penalty | 0.0222 | (0.0179, 0.0264) |
| baseline_no_context_phi3_latest | lexical_richness | 0.8442 | (0.8243, 0.8629) |

## Paired Deltas
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) |
|---|---|---:|---:|---:|
| proposed_vs_candidate_no_context | distinct1 | -0.0041 | (-0.0111, 0.0031) | 0.8653 |
| proposed_vs_candidate_no_context | distinct2 | 0.0010 | (-0.0004, 0.0033) | 0.1553 |
| proposed_vs_candidate_no_context | content_distinct1 | 0.0011 | (-0.0020, 0.0045) | 0.2643 |
| proposed_vs_candidate_no_context | mtld | -6.9536 | (-48.8910, 35.6371) | 0.6310 |
| proposed_vs_candidate_no_context | repetition_penalty | 0.0041 | (-0.0029, 0.0112) | 0.1243 |
| proposed_vs_candidate_no_context | lexical_richness | -0.0060 | (-0.0285, 0.0167) | 0.6933 |
| proposed_vs_baseline_no_context | distinct1 | -0.0392 | (-0.0469, -0.0315) | 1.0000 |
| proposed_vs_baseline_no_context | distinct2 | -0.0004 | (-0.0009, 0.0000) | 1.0000 |
| proposed_vs_baseline_no_context | content_distinct1 | -0.0012 | (-0.0033, 0.0009) | 0.8473 |
| proposed_vs_baseline_no_context | mtld | 13.7224 | (-28.6994, 57.8991) | 0.2677 |
| proposed_vs_baseline_no_context | repetition_penalty | 0.0392 | (0.0315, 0.0468) | 0.0000 |
| proposed_vs_baseline_no_context | lexical_richness | -0.0129 | (-0.0381, 0.0128) | 0.8393 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | -0.0381 | (-0.0461, -0.0298) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | distinct2 | -0.0002 | (-0.0008, 0.0003) | 0.8577 |
| proposed_vs_baseline_no_context_phi3_latest | content_distinct1 | 0.0003 | (-0.0022, 0.0028) | 0.4177 |
| proposed_vs_baseline_no_context_phi3_latest | mtld | 2.7754 | (-41.9896, 45.9676) | 0.4647 |
| proposed_vs_baseline_no_context_phi3_latest | repetition_penalty | 0.0381 | (0.0298, 0.0459) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | lexical_richness | -0.0184 | (-0.0448, 0.0079) | 0.9160 |
| controlled_vs_proposed_raw | distinct1 | -0.0091 | (-0.0165, -0.0015) | 0.9903 |
| controlled_vs_proposed_raw | distinct2 | -0.0026 | (-0.0042, -0.0012) | 1.0000 |
| controlled_vs_proposed_raw | content_distinct1 | -0.0062 | (-0.0099, -0.0028) | 1.0000 |
| controlled_vs_proposed_raw | mtld | 19.1819 | (-17.7824, 53.4893) | 0.1433 |
| controlled_vs_proposed_raw | repetition_penalty | 0.0091 | (0.0013, 0.0165) | 0.0093 |
| controlled_vs_proposed_raw | lexical_richness | 0.0941 | (0.0729, 0.1148) | 0.0000 |
| controlled_vs_candidate_no_context | distinct1 | -0.0132 | (-0.0212, -0.0054) | 0.9993 |
| controlled_vs_candidate_no_context | distinct2 | -0.0016 | (-0.0039, 0.0010) | 0.8967 |
| controlled_vs_candidate_no_context | content_distinct1 | -0.0051 | (-0.0093, -0.0007) | 0.9910 |
| controlled_vs_candidate_no_context | mtld | 12.2284 | (-26.6334, 48.8829) | 0.2543 |
| controlled_vs_candidate_no_context | repetition_penalty | 0.0132 | (0.0054, 0.0210) | 0.0003 |
| controlled_vs_candidate_no_context | lexical_richness | 0.0881 | (0.0644, 0.1107) | 0.0000 |
| controlled_vs_baseline_no_context | distinct1 | -0.0483 | (-0.0554, -0.0413) | 1.0000 |
| controlled_vs_baseline_no_context | distinct2 | -0.0030 | (-0.0046, -0.0016) | 1.0000 |
| controlled_vs_baseline_no_context | content_distinct1 | -0.0073 | (-0.0112, -0.0037) | 1.0000 |
| controlled_vs_baseline_no_context | mtld | 32.9044 | (1.7142, 62.3640) | 0.0197 |
| controlled_vs_baseline_no_context | repetition_penalty | 0.0483 | (0.0412, 0.0551) | 0.0000 |
| controlled_vs_baseline_no_context | lexical_richness | 0.0812 | (0.0592, 0.1031) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0472 | (-0.0549, -0.0397) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | distinct2 | -0.0029 | (-0.0045, -0.0014) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | content_distinct1 | -0.0059 | (-0.0098, -0.0022) | 0.9983 |
| controlled_vs_baseline_no_context_phi3_latest | mtld | 21.9573 | (-7.5945, 51.0301) | 0.0733 |
| controlled_vs_baseline_no_context_phi3_latest | repetition_penalty | 0.0472 | (0.0398, 0.0545) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | lexical_richness | 0.0757 | (0.0551, 0.0965) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0483 | (-0.0549, -0.0411) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct2 | -0.0030 | (-0.0046, -0.0016) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | content_distinct1 | -0.0073 | (-0.0111, -0.0037) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | mtld | 32.9044 | (3.2502, 62.4851) | 0.0140 |
| proposed_contextual_controlled_vs_baseline_no_context | repetition_penalty | 0.0483 | (0.0414, 0.0552) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | lexical_richness | 0.0812 | (0.0593, 0.1016) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0472 | (-0.0545, -0.0393) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct2 | -0.0029 | (-0.0045, -0.0015) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | content_distinct1 | -0.0059 | (-0.0097, -0.0021) | 0.9983 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | mtld | 21.9573 | (-10.3414, 51.7451) | 0.0833 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | repetition_penalty | 0.0472 | (0.0401, 0.0544) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | lexical_richness | 0.0757 | (0.0540, 0.0967) | 0.0000 |
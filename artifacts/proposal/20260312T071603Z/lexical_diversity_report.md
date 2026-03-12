# Lexical Diversity Benchmark

- Run dir: `artifacts\proposal\20260312T071603Z`
- Scenario count: `144`

## Arm Summary
| Arm | Metric | Mean | 95% CI |
|---|---|---:|---:|
| proposed_contextual_controlled | distinct1 | 0.9318 | (0.9247, 0.9392) |
| proposed_contextual_controlled | distinct2 | 0.9899 | (0.9865, 0.9929) |
| proposed_contextual_controlled | content_distinct1 | 0.9654 | (0.9564, 0.9738) |
| proposed_contextual_controlled | mtld | 172.1865 | (142.6304, 204.0239) |
| proposed_contextual_controlled | repetition_penalty | 0.0682 | (0.0608, 0.0751) |
| proposed_contextual_controlled | lexical_richness | 0.8622 | (0.8427, 0.8817) |
| proposed_contextual | distinct1 | 0.9371 | (0.9285, 0.9451) |
| proposed_contextual | distinct2 | 0.9885 | (0.9852, 0.9916) |
| proposed_contextual | content_distinct1 | 0.9632 | (0.9537, 0.9725) |
| proposed_contextual | mtld | 145.6501 | (116.5123, 177.0750) |
| proposed_contextual | repetition_penalty | 0.0629 | (0.0547, 0.0709) |
| proposed_contextual | lexical_richness | 0.8229 | (0.8030, 0.8425) |
| candidate_no_context | distinct1 | 0.9463 | (0.9381, 0.9544) |
| candidate_no_context | distinct2 | 0.9914 | (0.9884, 0.9942) |
| candidate_no_context | content_distinct1 | 0.9697 | (0.9613, 0.9781) |
| candidate_no_context | mtld | 136.2609 | (109.3755, 164.9122) |
| candidate_no_context | repetition_penalty | 0.0537 | (0.0455, 0.0616) |
| candidate_no_context | lexical_richness | 0.8241 | (0.8064, 0.8431) |
| baseline_no_context | distinct1 | 0.9771 | (0.9721, 0.9819) |
| baseline_no_context | distinct2 | 0.9999 | (0.9996, 1.0000) |
| baseline_no_context | content_distinct1 | 0.9985 | (0.9970, 0.9997) |
| baseline_no_context | mtld | 137.5121 | (106.6923, 170.2734) |
| baseline_no_context | repetition_penalty | 0.0229 | (0.0181, 0.0281) |
| baseline_no_context | lexical_richness | 0.8374 | (0.8183, 0.8557) |
| baseline_no_context_phi3_latest | distinct1 | 0.9778 | (0.9730, 0.9825) |
| baseline_no_context_phi3_latest | distinct2 | 0.9999 | (0.9997, 1.0000) |
| baseline_no_context_phi3_latest | content_distinct1 | 0.9993 | (0.9981, 1.0000) |
| baseline_no_context_phi3_latest | mtld | 145.7663 | (117.2530, 176.4646) |
| baseline_no_context_phi3_latest | repetition_penalty | 0.0222 | (0.0176, 0.0273) |
| baseline_no_context_phi3_latest | lexical_richness | 0.8420 | (0.8228, 0.8602) |

## Paired Deltas
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) |
|---|---|---:|---:|---:|
| proposed_vs_candidate_no_context | distinct1 | -0.0092 | (-0.0165, -0.0017) | 0.9910 |
| proposed_vs_candidate_no_context | distinct2 | -0.0029 | (-0.0051, -0.0008) | 0.9980 |
| proposed_vs_candidate_no_context | content_distinct1 | -0.0065 | (-0.0135, 0.0002) | 0.9717 |
| proposed_vs_candidate_no_context | mtld | 9.3892 | (-28.7304, 46.6079) | 0.3190 |
| proposed_vs_candidate_no_context | repetition_penalty | 0.0092 | (0.0017, 0.0166) | 0.0093 |
| proposed_vs_candidate_no_context | lexical_richness | -0.0013 | (-0.0228, 0.0206) | 0.5437 |
| proposed_vs_baseline_no_context | distinct1 | -0.0400 | (-0.0500, -0.0304) | 1.0000 |
| proposed_vs_baseline_no_context | distinct2 | -0.0114 | (-0.0147, -0.0083) | 1.0000 |
| proposed_vs_baseline_no_context | content_distinct1 | -0.0353 | (-0.0450, -0.0262) | 1.0000 |
| proposed_vs_baseline_no_context | mtld | 8.1379 | (-39.7417, 54.8826) | 0.3620 |
| proposed_vs_baseline_no_context | repetition_penalty | 0.0400 | (0.0303, 0.0499) | 0.0000 |
| proposed_vs_baseline_no_context | lexical_richness | -0.0145 | (-0.0430, 0.0138) | 0.8393 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | -0.0407 | (-0.0505, -0.0302) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | distinct2 | -0.0114 | (-0.0147, -0.0082) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | content_distinct1 | -0.0360 | (-0.0456, -0.0269) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | mtld | -0.1163 | (-44.9394, 45.5666) | 0.4950 |
| proposed_vs_baseline_no_context_phi3_latest | repetition_penalty | 0.0407 | (0.0306, 0.0504) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | lexical_richness | -0.0192 | (-0.0458, 0.0080) | 0.9190 |
| controlled_vs_proposed_raw | distinct1 | -0.0052 | (-0.0134, 0.0032) | 0.8950 |
| controlled_vs_proposed_raw | distinct2 | 0.0014 | (-0.0015, 0.0042) | 0.1773 |
| controlled_vs_proposed_raw | content_distinct1 | 0.0022 | (-0.0067, 0.0110) | 0.3207 |
| controlled_vs_proposed_raw | mtld | 26.5364 | (-16.3851, 66.2377) | 0.1070 |
| controlled_vs_proposed_raw | repetition_penalty | 0.0052 | (-0.0035, 0.0135) | 0.1157 |
| controlled_vs_proposed_raw | lexical_richness | 0.0393 | (0.0157, 0.0621) | 0.0013 |
| controlled_vs_candidate_no_context | distinct1 | -0.0144 | (-0.0231, -0.0056) | 1.0000 |
| controlled_vs_candidate_no_context | distinct2 | -0.0015 | (-0.0045, 0.0014) | 0.8580 |
| controlled_vs_candidate_no_context | content_distinct1 | -0.0043 | (-0.0131, 0.0047) | 0.8223 |
| controlled_vs_candidate_no_context | mtld | 35.9256 | (-5.9115, 77.6222) | 0.0500 |
| controlled_vs_candidate_no_context | repetition_penalty | 0.0144 | (0.0061, 0.0228) | 0.0003 |
| controlled_vs_candidate_no_context | lexical_richness | 0.0380 | (0.0110, 0.0634) | 0.0030 |
| controlled_vs_baseline_no_context | distinct1 | -0.0452 | (-0.0542, -0.0362) | 1.0000 |
| controlled_vs_baseline_no_context | distinct2 | -0.0100 | (-0.0133, -0.0069) | 1.0000 |
| controlled_vs_baseline_no_context | content_distinct1 | -0.0331 | (-0.0426, -0.0246) | 1.0000 |
| controlled_vs_baseline_no_context | mtld | 34.6744 | (-11.0690, 80.6677) | 0.0633 |
| controlled_vs_baseline_no_context | repetition_penalty | 0.0452 | (0.0362, 0.0538) | 0.0000 |
| controlled_vs_baseline_no_context | lexical_richness | 0.0248 | (-0.0054, 0.0529) | 0.0473 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0459 | (-0.0550, -0.0370) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | distinct2 | -0.0100 | (-0.0133, -0.0069) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | content_distinct1 | -0.0339 | (-0.0425, -0.0251) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | mtld | 26.4202 | (-15.4359, 70.9202) | 0.1097 |
| controlled_vs_baseline_no_context_phi3_latest | repetition_penalty | 0.0459 | (0.0367, 0.0548) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | lexical_richness | 0.0201 | (-0.0080, 0.0461) | 0.0783 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0452 | (-0.0542, -0.0362) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct2 | -0.0100 | (-0.0131, -0.0070) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | content_distinct1 | -0.0331 | (-0.0423, -0.0241) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | mtld | 34.6744 | (-10.2240, 78.0968) | 0.0630 |
| proposed_contextual_controlled_vs_baseline_no_context | repetition_penalty | 0.0452 | (0.0365, 0.0543) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | lexical_richness | 0.0248 | (-0.0045, 0.0531) | 0.0443 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0459 | (-0.0550, -0.0368) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct2 | -0.0100 | (-0.0133, -0.0070) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | content_distinct1 | -0.0339 | (-0.0429, -0.0252) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | mtld | 26.4202 | (-17.6949, 70.6906) | 0.1243 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | repetition_penalty | 0.0459 | (0.0368, 0.0551) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | lexical_richness | 0.0201 | (-0.0085, 0.0468) | 0.0733 |
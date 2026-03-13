# Lexical Diversity Benchmark

- Run dir: `artifacts\proposal\20260312T213231Z`
- Scenario count: `144`

## Arm Summary
| Arm | Metric | Mean | 95% CI |
|---|---|---:|---:|
| proposed_contextual_controlled | distinct1 | 0.9233 | (0.9149, 0.9316) |
| proposed_contextual_controlled | distinct2 | 0.9849 | (0.9790, 0.9896) |
| proposed_contextual_controlled | content_distinct1 | 0.9566 | (0.9464, 0.9665) |
| proposed_contextual_controlled | mtld | 161.1936 | (135.5345, 187.1920) |
| proposed_contextual_controlled | repetition_penalty | 0.0767 | (0.0689, 0.0849) |
| proposed_contextual_controlled | lexical_richness | 0.8496 | (0.8284, 0.8693) |
| proposed_contextual | distinct1 | 0.9358 | (0.9274, 0.9442) |
| proposed_contextual | distinct2 | 0.9887 | (0.9852, 0.9920) |
| proposed_contextual | content_distinct1 | 0.9652 | (0.9558, 0.9744) |
| proposed_contextual | mtld | 151.9535 | (121.3816, 185.1565) |
| proposed_contextual | repetition_penalty | 0.0642 | (0.0558, 0.0724) |
| proposed_contextual | lexical_richness | 0.8306 | (0.8110, 0.8502) |
| candidate_no_context | distinct1 | 0.9433 | (0.9354, 0.9512) |
| candidate_no_context | distinct2 | 0.9905 | (0.9875, 0.9934) |
| candidate_no_context | content_distinct1 | 0.9708 | (0.9622, 0.9792) |
| candidate_no_context | mtld | 145.1542 | (118.7189, 174.6766) |
| candidate_no_context | repetition_penalty | 0.0567 | (0.0491, 0.0644) |
| candidate_no_context | lexical_richness | 0.8346 | (0.8152, 0.8553) |
| baseline_no_context | distinct1 | 0.9812 | (0.9771, 0.9850) |
| baseline_no_context | distinct2 | 1.0000 | (1.0000, 1.0000) |
| baseline_no_context | content_distinct1 | 0.9979 | (0.9961, 0.9993) |
| baseline_no_context | mtld | 152.4087 | (119.6546, 185.8194) |
| baseline_no_context | repetition_penalty | 0.0188 | (0.0149, 0.0228) |
| baseline_no_context | lexical_richness | 0.8431 | (0.8240, 0.8632) |
| baseline_no_context_phi3_latest | distinct1 | 0.9770 | (0.9725, 0.9812) |
| baseline_no_context_phi3_latest | distinct2 | 1.0000 | (1.0000, 1.0000) |
| baseline_no_context_phi3_latest | content_distinct1 | 0.9990 | (0.9979, 0.9998) |
| baseline_no_context_phi3_latest | mtld | 169.1199 | (135.4246, 205.2757) |
| baseline_no_context_phi3_latest | repetition_penalty | 0.0230 | (0.0188, 0.0276) |
| baseline_no_context_phi3_latest | lexical_richness | 0.8553 | (0.8351, 0.8752) |

## Paired Deltas
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) |
|---|---|---:|---:|---:|
| proposed_vs_candidate_no_context | distinct1 | -0.0075 | (-0.0147, -0.0001) | 0.9767 |
| proposed_vs_candidate_no_context | distinct2 | -0.0018 | (-0.0042, 0.0005) | 0.9380 |
| proposed_vs_candidate_no_context | content_distinct1 | -0.0056 | (-0.0125, 0.0010) | 0.9523 |
| proposed_vs_candidate_no_context | mtld | 6.7993 | (-33.4584, 45.4083) | 0.3893 |
| proposed_vs_candidate_no_context | repetition_penalty | 0.0075 | (-0.0001, 0.0152) | 0.0277 |
| proposed_vs_candidate_no_context | lexical_richness | -0.0040 | (-0.0243, 0.0164) | 0.6330 |
| proposed_vs_baseline_no_context | distinct1 | -0.0454 | (-0.0541, -0.0363) | 1.0000 |
| proposed_vs_baseline_no_context | distinct2 | -0.0113 | (-0.0146, -0.0081) | 1.0000 |
| proposed_vs_baseline_no_context | content_distinct1 | -0.0327 | (-0.0417, -0.0241) | 1.0000 |
| proposed_vs_baseline_no_context | mtld | -0.4552 | (-45.0927, 44.9720) | 0.4980 |
| proposed_vs_baseline_no_context | repetition_penalty | 0.0454 | (0.0361, 0.0543) | 0.0000 |
| proposed_vs_baseline_no_context | lexical_richness | -0.0124 | (-0.0390, 0.0136) | 0.8273 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | -0.0411 | (-0.0502, -0.0314) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | distinct2 | -0.0113 | (-0.0145, -0.0081) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | content_distinct1 | -0.0337 | (-0.0428, -0.0249) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | mtld | -17.1664 | (-61.8956, 26.3707) | 0.7853 |
| proposed_vs_baseline_no_context_phi3_latest | repetition_penalty | 0.0411 | (0.0315, 0.0501) | 0.0000 |
| proposed_vs_baseline_no_context_phi3_latest | lexical_richness | -0.0247 | (-0.0499, 0.0015) | 0.9670 |
| controlled_vs_proposed_raw | distinct1 | -0.0125 | (-0.0220, -0.0031) | 0.9943 |
| controlled_vs_proposed_raw | distinct2 | -0.0039 | (-0.0094, 0.0007) | 0.9433 |
| controlled_vs_proposed_raw | content_distinct1 | -0.0086 | (-0.0182, -0.0001) | 0.9763 |
| controlled_vs_proposed_raw | mtld | 9.2401 | (-28.7813, 46.5775) | 0.3057 |
| controlled_vs_proposed_raw | repetition_penalty | 0.0125 | (0.0031, 0.0214) | 0.0037 |
| controlled_vs_proposed_raw | lexical_richness | 0.0190 | (-0.0013, 0.0394) | 0.0380 |
| controlled_vs_candidate_no_context | distinct1 | -0.0200 | (-0.0291, -0.0114) | 1.0000 |
| controlled_vs_candidate_no_context | distinct2 | -0.0057 | (-0.0110, -0.0011) | 0.9960 |
| controlled_vs_candidate_no_context | content_distinct1 | -0.0142 | (-0.0235, -0.0051) | 0.9990 |
| controlled_vs_candidate_no_context | mtld | 16.0394 | (-21.3464, 56.1945) | 0.1883 |
| controlled_vs_candidate_no_context | repetition_penalty | 0.0200 | (0.0114, 0.0288) | 0.0000 |
| controlled_vs_candidate_no_context | lexical_richness | 0.0150 | (-0.0094, 0.0388) | 0.1240 |
| controlled_vs_baseline_no_context | distinct1 | -0.0579 | (-0.0669, -0.0494) | 1.0000 |
| controlled_vs_baseline_no_context | distinct2 | -0.0151 | (-0.0206, -0.0103) | 1.0000 |
| controlled_vs_baseline_no_context | content_distinct1 | -0.0413 | (-0.0520, -0.0316) | 1.0000 |
| controlled_vs_baseline_no_context | mtld | 8.7849 | (-34.8011, 50.1679) | 0.3410 |
| controlled_vs_baseline_no_context | repetition_penalty | 0.0579 | (0.0490, 0.0667) | 0.0000 |
| controlled_vs_baseline_no_context | lexical_richness | 0.0066 | (-0.0213, 0.0351) | 0.3267 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0536 | (-0.0626, -0.0448) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | distinct2 | -0.0151 | (-0.0208, -0.0104) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | content_distinct1 | -0.0423 | (-0.0520, -0.0324) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | mtld | -7.9263 | (-45.7584, 29.3535) | 0.6647 |
| controlled_vs_baseline_no_context_phi3_latest | repetition_penalty | 0.0536 | (0.0443, 0.0625) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | lexical_richness | -0.0057 | (-0.0310, 0.0200) | 0.6560 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0579 | (-0.0669, -0.0491) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct2 | -0.0151 | (-0.0205, -0.0103) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | content_distinct1 | -0.0413 | (-0.0513, -0.0313) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | mtld | 8.7849 | (-34.7277, 48.8764) | 0.3360 |
| proposed_contextual_controlled_vs_baseline_no_context | repetition_penalty | 0.0579 | (0.0490, 0.0671) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | lexical_richness | 0.0066 | (-0.0239, 0.0355) | 0.3377 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0536 | (-0.0628, -0.0448) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct2 | -0.0151 | (-0.0208, -0.0105) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | content_distinct1 | -0.0423 | (-0.0529, -0.0327) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | mtld | -7.9263 | (-46.3019, 29.3157) | 0.6483 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | repetition_penalty | 0.0536 | (0.0448, 0.0626) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | lexical_richness | -0.0057 | (-0.0328, 0.0199) | 0.6720 |
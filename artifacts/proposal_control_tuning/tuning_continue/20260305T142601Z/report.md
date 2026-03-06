# Proposal Alignment Evaluation Report

- Run ID: `20260305T142601Z`
- Generated: `2026-03-05T14:30:15.262549+00:00`
- Scenarios: `artifacts\proposal_control_tuning\tuning_continue\20260305T142601Z\scenarios.jsonl`
- Scenario count: `20`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off
- `baseline_no_context`: model `phi3:mini`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2852 (0.2436, 0.3288) | 0.4005 (0.3118, 0.5022) | 0.8567 (0.8219, 0.8912) | 0.4048 (0.3760, 0.4357) | 0.0883 |
| proposed_contextual | 0.0660 (0.0351, 0.1058) | 0.1687 (0.1036, 0.2445) | 0.7914 (0.7681, 0.8183) | 0.2254 (0.1972, 0.2547) | 0.0689 |
| candidate_no_context | 0.0261 (0.0138, 0.0418) | 0.1768 (0.1200, 0.2383) | 0.7949 (0.7661, 0.8269) | 0.2107 (0.1858, 0.2396) | 0.0490 |
| baseline_no_context | 0.0370 (0.0244, 0.0502) | 0.2120 (0.1658, 0.2608) | 0.9112 (0.8878, 0.9338) | 0.2458 (0.2273, 0.2637) | 0.0415 |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0399 | 1.5316 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0081 | -0.0457 |
| proposed_vs_candidate_no_context | naturalness | -0.0035 | -0.0044 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0534 | 3.0652 |
| proposed_vs_candidate_no_context | context_overlap | 0.0084 | 0.1814 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0231 | -0.2383 |
| proposed_vs_candidate_no_context | persona_style | 0.0520 | 0.1048 |
| proposed_vs_candidate_no_context | distinct1 | 0.0125 | 0.0135 |
| proposed_vs_candidate_no_context | length_score | -0.0250 | -0.1064 |
| proposed_vs_candidate_no_context | sentence_score | -0.0350 | -0.0453 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0199 | 0.4058 |
| proposed_vs_candidate_no_context | overall_quality | 0.0147 | 0.0696 |
| proposed_vs_baseline_no_context | context_relevance | 0.0289 | 0.7821 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0432 | -0.2040 |
| proposed_vs_baseline_no_context | naturalness | -0.1198 | -0.1315 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0345 | 0.9479 |
| proposed_vs_baseline_no_context | context_overlap | 0.0160 | 0.4167 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0386 | -0.3432 |
| proposed_vs_baseline_no_context | persona_style | -0.0620 | -0.1015 |
| proposed_vs_baseline_no_context | distinct1 | -0.0508 | -0.0513 |
| proposed_vs_baseline_no_context | length_score | -0.4100 | -0.6613 |
| proposed_vs_baseline_no_context | sentence_score | -0.1750 | -0.1918 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0274 | 0.6611 |
| proposed_vs_baseline_no_context | overall_quality | -0.0204 | -0.0831 |
| controlled_vs_proposed_raw | context_relevance | 0.2192 | 3.3239 |
| controlled_vs_proposed_raw | persona_consistency | 0.2318 | 1.3740 |
| controlled_vs_proposed_raw | naturalness | 0.0654 | 0.0826 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2884 | 4.0717 |
| controlled_vs_proposed_raw | context_overlap | 0.0578 | 1.0589 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2767 | 3.7484 |
| controlled_vs_proposed_raw | persona_style | 0.0524 | 0.0955 |
| controlled_vs_proposed_raw | distinct1 | -0.0066 | -0.0070 |
| controlled_vs_proposed_raw | length_score | 0.2350 | 1.1190 |
| controlled_vs_proposed_raw | sentence_score | 0.2100 | 0.2847 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0194 | 0.2820 |
| controlled_vs_proposed_raw | overall_quality | 0.1794 | 0.7957 |
| controlled_vs_candidate_no_context | context_relevance | 0.2591 | 9.9464 |
| controlled_vs_candidate_no_context | persona_consistency | 0.2237 | 1.2655 |
| controlled_vs_candidate_no_context | naturalness | 0.0619 | 0.0778 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3418 | 19.6174 |
| controlled_vs_candidate_no_context | context_overlap | 0.0662 | 1.4325 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2536 | 2.6167 |
| controlled_vs_candidate_no_context | persona_style | 0.1044 | 0.2103 |
| controlled_vs_candidate_no_context | distinct1 | 0.0059 | 0.0064 |
| controlled_vs_candidate_no_context | length_score | 0.2100 | 0.8936 |
| controlled_vs_candidate_no_context | sentence_score | 0.1750 | 0.2265 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0393 | 0.8023 |
| controlled_vs_candidate_no_context | overall_quality | 0.1940 | 0.9207 |
| controlled_vs_baseline_no_context | context_relevance | 0.2482 | 6.7056 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1886 | 0.8896 |
| controlled_vs_baseline_no_context | naturalness | -0.0544 | -0.0598 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3229 | 8.8792 |
| controlled_vs_baseline_no_context | context_overlap | 0.0738 | 1.9170 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2381 | 2.1186 |
| controlled_vs_baseline_no_context | persona_style | -0.0096 | -0.0157 |
| controlled_vs_baseline_no_context | distinct1 | -0.0574 | -0.0580 |
| controlled_vs_baseline_no_context | length_score | -0.1750 | -0.2823 |
| controlled_vs_baseline_no_context | sentence_score | 0.0350 | 0.0384 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0468 | 1.1296 |
| controlled_vs_baseline_no_context | overall_quality | 0.1589 | 0.6464 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2482 | 6.7056 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1886 | 0.8896 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0544 | -0.0598 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3229 | 8.8792 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0738 | 1.9170 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2381 | 2.1186 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | -0.0096 | -0.0157 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0574 | -0.0580 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | -0.1750 | -0.2823 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0350 | 0.0384 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0468 | 1.1296 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1589 | 0.6464 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0399 | (0.0165, 0.0685) | 0.0000 | 0.0399 | (0.0156, 0.0709) | 0.0000 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0081 | (-0.0773, 0.0638) | 0.5963 | -0.0081 | (-0.0825, 0.0757) | 0.5607 |
| proposed_vs_candidate_no_context | naturalness | -0.0035 | (-0.0265, 0.0171) | 0.6153 | -0.0035 | (-0.0274, 0.0109) | 0.6307 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0534 | (0.0220, 0.0909) | 0.0000 | 0.0534 | (0.0191, 0.0981) | 0.0000 |
| proposed_vs_candidate_no_context | context_overlap | 0.0084 | (0.0006, 0.0183) | 0.0170 | 0.0084 | (-0.0006, 0.0218) | 0.0410 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0231 | (-0.0986, 0.0583) | 0.7160 | -0.0231 | (-0.1140, 0.0754) | 0.6677 |
| proposed_vs_candidate_no_context | persona_style | 0.0520 | (-0.0028, 0.1339) | 0.0403 | 0.0520 | (-0.0057, 0.1319) | 0.0447 |
| proposed_vs_candidate_no_context | distinct1 | 0.0125 | (-0.0031, 0.0308) | 0.0693 | 0.0125 | (-0.0056, 0.0324) | 0.0900 |
| proposed_vs_candidate_no_context | length_score | -0.0250 | (-0.1100, 0.0500) | 0.7343 | -0.0250 | (-0.1083, 0.0227) | 0.8093 |
| proposed_vs_candidate_no_context | sentence_score | -0.0350 | (-0.1050, 0.0350) | 0.9023 | -0.0350 | (-0.1050, 0.0350) | 0.9063 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0199 | (-0.0082, 0.0507) | 0.0910 | 0.0199 | (-0.0132, 0.0648) | 0.1507 |
| proposed_vs_candidate_no_context | overall_quality | 0.0147 | (-0.0059, 0.0387) | 0.0847 | 0.0147 | (-0.0035, 0.0385) | 0.0677 |
| proposed_vs_baseline_no_context | context_relevance | 0.0289 | (-0.0101, 0.0726) | 0.0810 | 0.0289 | (-0.0187, 0.0812) | 0.1133 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0432 | (-0.1055, 0.0202) | 0.9010 | -0.0432 | (-0.1011, 0.0337) | 0.8687 |
| proposed_vs_baseline_no_context | naturalness | -0.1198 | (-0.1487, -0.0890) | 1.0000 | -0.1198 | (-0.1514, -0.0894) | 1.0000 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0345 | (-0.0197, 0.0939) | 0.1317 | 0.0345 | (-0.0318, 0.1098) | 0.1723 |
| proposed_vs_baseline_no_context | context_overlap | 0.0160 | (0.0043, 0.0294) | 0.0030 | 0.0160 | (0.0016, 0.0346) | 0.0113 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0386 | (-0.1098, 0.0400) | 0.8567 | -0.0386 | (-0.1111, 0.0476) | 0.7947 |
| proposed_vs_baseline_no_context | persona_style | -0.0620 | (-0.1666, 0.0314) | 0.8867 | -0.0620 | (-0.1983, 0.0495) | 0.8077 |
| proposed_vs_baseline_no_context | distinct1 | -0.0508 | (-0.0708, -0.0293) | 1.0000 | -0.0508 | (-0.0722, -0.0249) | 1.0000 |
| proposed_vs_baseline_no_context | length_score | -0.4100 | (-0.5050, -0.3133) | 1.0000 | -0.4100 | (-0.5111, -0.3417) | 1.0000 |
| proposed_vs_baseline_no_context | sentence_score | -0.1750 | (-0.2800, -0.0700) | 0.9997 | -0.1750 | (-0.2864, -0.0218) | 0.9910 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0274 | (-0.0118, 0.0686) | 0.0950 | 0.0274 | (-0.0249, 0.0932) | 0.1757 |
| proposed_vs_baseline_no_context | overall_quality | -0.0204 | (-0.0567, 0.0155) | 0.8647 | -0.0204 | (-0.0614, 0.0266) | 0.7977 |
| controlled_vs_proposed_raw | context_relevance | 0.2192 | (0.1659, 0.2720) | 0.0000 | 0.2192 | (0.1672, 0.2842) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.2318 | (0.1170, 0.3609) | 0.0000 | 0.2318 | (0.0694, 0.3429) | 0.0020 |
| controlled_vs_proposed_raw | naturalness | 0.0654 | (0.0133, 0.1161) | 0.0063 | 0.0654 | (0.0032, 0.1329) | 0.0170 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2884 | (0.2145, 0.3580) | 0.0000 | 0.2884 | (0.2204, 0.3779) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0578 | (0.0424, 0.0738) | 0.0000 | 0.0578 | (0.0383, 0.0735) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2767 | (0.1374, 0.4319) | 0.0000 | 0.2767 | (0.0746, 0.4328) | 0.0017 |
| controlled_vs_proposed_raw | persona_style | 0.0524 | (-0.0350, 0.1557) | 0.1577 | 0.0524 | (-0.0575, 0.1998) | 0.2550 |
| controlled_vs_proposed_raw | distinct1 | -0.0066 | (-0.0279, 0.0126) | 0.7403 | -0.0066 | (-0.0312, 0.0132) | 0.7303 |
| controlled_vs_proposed_raw | length_score | 0.2350 | (0.0350, 0.4467) | 0.0127 | 0.2350 | (-0.0278, 0.5167) | 0.0363 |
| controlled_vs_proposed_raw | sentence_score | 0.2100 | (0.1400, 0.2800) | 0.0000 | 0.2100 | (0.1633, 0.2587) | 0.0000 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0194 | (-0.0152, 0.0521) | 0.1283 | 0.0194 | (-0.0208, 0.0588) | 0.1620 |
| controlled_vs_proposed_raw | overall_quality | 0.1794 | (0.1424, 0.2172) | 0.0000 | 0.1794 | (0.1327, 0.2151) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2591 | (0.2186, 0.3012) | 0.0000 | 0.2591 | (0.2166, 0.3180) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.2237 | (0.1266, 0.3197) | 0.0000 | 0.2237 | (0.0903, 0.3201) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.0619 | (0.0053, 0.1165) | 0.0163 | 0.0619 | (-0.0152, 0.1285) | 0.0547 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3418 | (0.2899, 0.3962) | 0.0000 | 0.3418 | (0.2869, 0.4205) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0662 | (0.0510, 0.0813) | 0.0000 | 0.0662 | (0.0480, 0.0833) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2536 | (0.1326, 0.3817) | 0.0000 | 0.2536 | (0.0952, 0.3834) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.1044 | (-0.0065, 0.2294) | 0.0350 | 0.1044 | (-0.0518, 0.3060) | 0.1517 |
| controlled_vs_candidate_no_context | distinct1 | 0.0059 | (-0.0154, 0.0268) | 0.2990 | 0.0059 | (-0.0149, 0.0203) | 0.2460 |
| controlled_vs_candidate_no_context | length_score | 0.2100 | (-0.0033, 0.4317) | 0.0277 | 0.2100 | (-0.0939, 0.4912) | 0.0843 |
| controlled_vs_candidate_no_context | sentence_score | 0.1750 | (0.1050, 0.2450) | 0.0000 | 0.1750 | (0.0824, 0.2545) | 0.0000 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0393 | (0.0125, 0.0641) | 0.0030 | 0.0393 | (0.0215, 0.0603) | 0.0000 |
| controlled_vs_candidate_no_context | overall_quality | 0.1940 | (0.1567, 0.2292) | 0.0000 | 0.1940 | (0.1464, 0.2244) | 0.0000 |
| controlled_vs_baseline_no_context | context_relevance | 0.2482 | (0.2034, 0.2934) | 0.0000 | 0.2482 | (0.2038, 0.3153) | 0.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1886 | (0.0794, 0.3065) | 0.0000 | 0.1886 | (0.0590, 0.2816) | 0.0000 |
| controlled_vs_baseline_no_context | naturalness | -0.0544 | (-0.0976, -0.0092) | 0.9900 | -0.0544 | (-0.1057, -0.0030) | 0.9807 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3229 | (0.2609, 0.3856) | 0.0000 | 0.3229 | (0.2597, 0.4152) | 0.0000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0738 | (0.0621, 0.0856) | 0.0000 | 0.0738 | (0.0627, 0.0880) | 0.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2381 | (0.1012, 0.3867) | 0.0000 | 0.2381 | (0.0782, 0.3623) | 0.0000 |
| controlled_vs_baseline_no_context | persona_style | -0.0096 | (-0.0544, 0.0256) | 0.7340 | -0.0096 | (-0.0582, 0.0233) | 0.7307 |
| controlled_vs_baseline_no_context | distinct1 | -0.0574 | (-0.0692, -0.0449) | 1.0000 | -0.0574 | (-0.0700, -0.0443) | 1.0000 |
| controlled_vs_baseline_no_context | length_score | -0.1750 | (-0.3733, 0.0134) | 0.9657 | -0.1750 | (-0.4216, 0.0737) | 0.9280 |
| controlled_vs_baseline_no_context | sentence_score | 0.0350 | (-0.0700, 0.1229) | 0.2910 | 0.0350 | (-0.0656, 0.1575) | 0.3167 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0468 | (0.0146, 0.0790) | 0.0017 | 0.0468 | (0.0125, 0.0858) | 0.0013 |
| controlled_vs_baseline_no_context | overall_quality | 0.1589 | (0.1210, 0.1973) | 0.0000 | 0.1589 | (0.1165, 0.1956) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2482 | (0.2053, 0.2941) | 0.0000 | 0.2482 | (0.2032, 0.3166) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1886 | (0.0781, 0.3068) | 0.0000 | 0.1886 | (0.0616, 0.2841) | 0.0003 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0544 | (-0.0981, -0.0128) | 0.9947 | -0.0544 | (-0.1042, -0.0027) | 0.9790 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3229 | (0.2619, 0.3848) | 0.0000 | 0.3229 | (0.2625, 0.4176) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0738 | (0.0621, 0.0861) | 0.0000 | 0.0738 | (0.0627, 0.0884) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2381 | (0.1004, 0.3793) | 0.0000 | 0.2381 | (0.0789, 0.3590) | 0.0007 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | -0.0096 | (-0.0544, 0.0256) | 0.7357 | -0.0096 | (-0.0573, 0.0244) | 0.7103 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0574 | (-0.0695, -0.0452) | 1.0000 | -0.0574 | (-0.0710, -0.0445) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | -0.1750 | (-0.3700, 0.0184) | 0.9600 | -0.1750 | (-0.4190, 0.0750) | 0.9253 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0350 | (-0.0525, 0.1225) | 0.2890 | 0.0350 | (-0.0657, 0.1556) | 0.3383 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0468 | (0.0122, 0.0793) | 0.0027 | 0.0468 | (0.0127, 0.0870) | 0.0023 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1589 | (0.1211, 0.1981) | 0.0000 | 0.1589 | (0.1187, 0.1960) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 10 | 1 | 9 | 0.7250 | 0.9091 |
| proposed_vs_candidate_no_context | persona_consistency | 5 | 5 | 10 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | naturalness | 4 | 6 | 10 | 0.4500 | 0.4000 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 7 | 0 | 13 | 0.6750 | 1.0000 |
| proposed_vs_candidate_no_context | context_overlap | 7 | 4 | 9 | 0.5750 | 0.6364 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 3 | 5 | 12 | 0.4500 | 0.3750 |
| proposed_vs_candidate_no_context | persona_style | 4 | 1 | 15 | 0.5750 | 0.8000 |
| proposed_vs_candidate_no_context | distinct1 | 7 | 3 | 10 | 0.6000 | 0.7000 |
| proposed_vs_candidate_no_context | length_score | 4 | 6 | 10 | 0.4500 | 0.4000 |
| proposed_vs_candidate_no_context | sentence_score | 1 | 3 | 16 | 0.4500 | 0.2500 |
| proposed_vs_candidate_no_context | bertscore_f1 | 12 | 6 | 2 | 0.6500 | 0.6667 |
| proposed_vs_candidate_no_context | overall_quality | 12 | 6 | 2 | 0.6500 | 0.6667 |
| proposed_vs_baseline_no_context | context_relevance | 9 | 11 | 0 | 0.4500 | 0.4500 |
| proposed_vs_baseline_no_context | persona_consistency | 4 | 8 | 8 | 0.4000 | 0.3333 |
| proposed_vs_baseline_no_context | naturalness | 2 | 18 | 0 | 0.1000 | 0.1000 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 6 | 6 | 8 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context | context_overlap | 12 | 8 | 0 | 0.6000 | 0.6000 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 3 | 7 | 10 | 0.4000 | 0.3000 |
| proposed_vs_baseline_no_context | persona_style | 2 | 5 | 13 | 0.4250 | 0.2857 |
| proposed_vs_baseline_no_context | distinct1 | 3 | 15 | 2 | 0.2000 | 0.1667 |
| proposed_vs_baseline_no_context | length_score | 1 | 19 | 0 | 0.0500 | 0.0500 |
| proposed_vs_baseline_no_context | sentence_score | 2 | 12 | 6 | 0.2500 | 0.1429 |
| proposed_vs_baseline_no_context | bertscore_f1 | 12 | 8 | 0 | 0.6000 | 0.6000 |
| proposed_vs_baseline_no_context | overall_quality | 8 | 12 | 0 | 0.4000 | 0.4000 |
| controlled_vs_proposed_raw | context_relevance | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_vs_proposed_raw | persona_consistency | 16 | 3 | 1 | 0.8250 | 0.8421 |
| controlled_vs_proposed_raw | naturalness | 12 | 8 | 0 | 0.6000 | 0.6000 |
| controlled_vs_proposed_raw | context_keyword_coverage | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_vs_proposed_raw | context_overlap | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 16 | 3 | 1 | 0.8250 | 0.8421 |
| controlled_vs_proposed_raw | persona_style | 4 | 3 | 13 | 0.5250 | 0.5714 |
| controlled_vs_proposed_raw | distinct1 | 10 | 7 | 3 | 0.5750 | 0.5882 |
| controlled_vs_proposed_raw | length_score | 12 | 7 | 1 | 0.6250 | 0.6316 |
| controlled_vs_proposed_raw | sentence_score | 12 | 0 | 8 | 0.8000 | 1.0000 |
| controlled_vs_proposed_raw | bertscore_f1 | 14 | 6 | 0 | 0.7000 | 0.7000 |
| controlled_vs_proposed_raw | overall_quality | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_relevance | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 15 | 3 | 2 | 0.8000 | 0.8333 |
| controlled_vs_candidate_no_context | naturalness | 13 | 7 | 0 | 0.6500 | 0.6500 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 15 | 2 | 3 | 0.8250 | 0.8824 |
| controlled_vs_candidate_no_context | persona_style | 6 | 3 | 11 | 0.5750 | 0.6667 |
| controlled_vs_candidate_no_context | distinct1 | 12 | 6 | 2 | 0.6500 | 0.6667 |
| controlled_vs_candidate_no_context | length_score | 13 | 7 | 0 | 0.6500 | 0.6500 |
| controlled_vs_candidate_no_context | sentence_score | 10 | 0 | 10 | 0.7500 | 1.0000 |
| controlled_vs_candidate_no_context | bertscore_f1 | 16 | 4 | 0 | 0.8000 | 0.8000 |
| controlled_vs_candidate_no_context | overall_quality | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_vs_baseline_no_context | context_relevance | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 14 | 2 | 4 | 0.8000 | 0.8750 |
| controlled_vs_baseline_no_context | naturalness | 6 | 14 | 0 | 0.3000 | 0.3000 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | context_overlap | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 14 | 2 | 4 | 0.8000 | 0.8750 |
| controlled_vs_baseline_no_context | persona_style | 1 | 1 | 18 | 0.5000 | 0.5000 |
| controlled_vs_baseline_no_context | distinct1 | 0 | 19 | 1 | 0.0250 | 0.0000 |
| controlled_vs_baseline_no_context | length_score | 8 | 12 | 0 | 0.4000 | 0.4000 |
| controlled_vs_baseline_no_context | sentence_score | 5 | 3 | 12 | 0.5500 | 0.6250 |
| controlled_vs_baseline_no_context | bertscore_f1 | 14 | 6 | 0 | 0.7000 | 0.7000 |
| controlled_vs_baseline_no_context | overall_quality | 19 | 1 | 0 | 0.9500 | 0.9500 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 20 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 14 | 2 | 4 | 0.8000 | 0.8750 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 6 | 14 | 0 | 0.3000 | 0.3000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 20 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 20 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 14 | 2 | 4 | 0.8000 | 0.8750 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 1 | 1 | 18 | 0.5000 | 0.5000 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | 0 | 19 | 1 | 0.0250 | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 8 | 12 | 0 | 0.4000 | 0.4000 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 5 | 3 | 12 | 0.5500 | 0.6250 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 14 | 6 | 0 | 0.7000 | 0.7000 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 19 | 1 | 0 | 0.9500 | 0.9500 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.3500 | 0.3000 | 0.7000 |
| proposed_contextual | 0.0000 | 0.0000 | 0.4500 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.6000 | 0.0000 | 0.0000 |
| baseline_no_context | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `8`
- Unique template signatures: `18`
- Template signature ratio: `0.9000`
- Effective sample size by source clustering: `6.67`
- Effective sample size by template-signature clustering: `16.67`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: enabled.

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
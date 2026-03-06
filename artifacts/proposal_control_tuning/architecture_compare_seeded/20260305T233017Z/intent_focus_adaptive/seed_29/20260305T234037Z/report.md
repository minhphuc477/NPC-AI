# Proposal Alignment Evaluation Report

- Run ID: `20260305T234037Z`
- Generated: `2026-03-05T23:45:44.183103+00:00`
- Scenarios: `artifacts\proposal_control_tuning\architecture_compare_seeded\20260305T233017Z\intent_focus_adaptive\seed_29\20260305T234037Z\scenarios.jsonl`
- Scenario count: `24`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_alt`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off
- `baseline_no_context`: model `phi3:mini`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2899 (0.2501, 0.3355) | 0.4180 (0.3519, 0.4952) | 0.8626 (0.8260, 0.8959) | 0.4535 (0.4241, 0.4847) | n/a |
| proposed_contextual_controlled_alt | 0.2900 (0.2406, 0.3405) | 0.3735 (0.3128, 0.4404) | 0.8654 (0.8217, 0.9052) | 0.4373 (0.4088, 0.4661) | n/a |
| proposed_contextual | 0.0488 (0.0249, 0.0817) | 0.1343 (0.0921, 0.1877) | 0.8047 (0.7794, 0.8316) | 0.2255 (0.2019, 0.2531) | n/a |
| candidate_no_context | 0.0282 (0.0151, 0.0454) | 0.1417 (0.0904, 0.2012) | 0.8094 (0.7786, 0.8432) | 0.2204 (0.1925, 0.2507) | n/a |
| baseline_no_context | 0.0430 (0.0273, 0.0608) | 0.2143 (0.1606, 0.2708) | 0.8844 (0.8630, 0.9078) | 0.2678 (0.2434, 0.2932) | n/a |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0206 | 0.7294 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0074 | -0.0523 |
| proposed_vs_candidate_no_context | naturalness | -0.0047 | -0.0058 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0262 | 1.1857 |
| proposed_vs_candidate_no_context | context_overlap | 0.0075 | 0.1761 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0198 | -0.2941 |
| proposed_vs_candidate_no_context | persona_style | 0.0423 | 0.0965 |
| proposed_vs_candidate_no_context | distinct1 | 0.0002 | 0.0002 |
| proposed_vs_candidate_no_context | length_score | -0.0458 | -0.1416 |
| proposed_vs_candidate_no_context | sentence_score | 0.0437 | 0.0605 |
| proposed_vs_candidate_no_context | overall_quality | 0.0051 | 0.0234 |
| proposed_vs_baseline_no_context | context_relevance | 0.0058 | 0.1347 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0800 | -0.3732 |
| proposed_vs_baseline_no_context | naturalness | -0.0797 | -0.0901 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0066 | 0.1573 |
| proposed_vs_baseline_no_context | context_overlap | 0.0040 | 0.0868 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0829 | -0.6353 |
| proposed_vs_baseline_no_context | persona_style | -0.0681 | -0.1241 |
| proposed_vs_baseline_no_context | distinct1 | -0.0519 | -0.0528 |
| proposed_vs_baseline_no_context | length_score | -0.2583 | -0.4819 |
| proposed_vs_baseline_no_context | sentence_score | -0.0729 | -0.0868 |
| proposed_vs_baseline_no_context | overall_quality | -0.0423 | -0.1579 |
| controlled_vs_proposed_raw | context_relevance | 0.2410 | 4.9390 |
| controlled_vs_proposed_raw | persona_consistency | 0.2837 | 2.1118 |
| controlled_vs_proposed_raw | naturalness | 0.0579 | 0.0719 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.3139 | 6.4993 |
| controlled_vs_proposed_raw | context_overlap | 0.0711 | 1.4218 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.3234 | 6.7917 |
| controlled_vs_proposed_raw | persona_style | 0.1247 | 0.2591 |
| controlled_vs_proposed_raw | distinct1 | -0.0055 | -0.0059 |
| controlled_vs_proposed_raw | length_score | 0.2250 | 0.8100 |
| controlled_vs_proposed_raw | sentence_score | 0.1750 | 0.2283 |
| controlled_vs_proposed_raw | overall_quality | 0.2280 | 1.0108 |
| controlled_vs_candidate_no_context | context_relevance | 0.2616 | 9.2711 |
| controlled_vs_candidate_no_context | persona_consistency | 0.2763 | 1.9492 |
| controlled_vs_candidate_no_context | naturalness | 0.0531 | 0.0657 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3401 | 15.3914 |
| controlled_vs_candidate_no_context | context_overlap | 0.0786 | 1.8481 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.3036 | 4.5000 |
| controlled_vs_candidate_no_context | persona_style | 0.1670 | 0.3805 |
| controlled_vs_candidate_no_context | distinct1 | -0.0053 | -0.0057 |
| controlled_vs_candidate_no_context | length_score | 0.1792 | 0.5536 |
| controlled_vs_candidate_no_context | sentence_score | 0.2188 | 0.3026 |
| controlled_vs_candidate_no_context | overall_quality | 0.2331 | 1.0578 |
| controlled_vs_baseline_no_context | context_relevance | 0.2468 | 5.7390 |
| controlled_vs_baseline_no_context | persona_consistency | 0.2037 | 0.9504 |
| controlled_vs_baseline_no_context | naturalness | -0.0218 | -0.0247 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3205 | 7.6793 |
| controlled_vs_baseline_no_context | context_overlap | 0.0751 | 1.6320 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2405 | 1.8419 |
| controlled_vs_baseline_no_context | persona_style | 0.0565 | 0.1029 |
| controlled_vs_baseline_no_context | distinct1 | -0.0574 | -0.0583 |
| controlled_vs_baseline_no_context | length_score | -0.0333 | -0.0622 |
| controlled_vs_baseline_no_context | sentence_score | 0.1021 | 0.1216 |
| controlled_vs_baseline_no_context | overall_quality | 0.1857 | 0.6932 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0002 | 0.0005 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0445 | -0.1064 |
| controlled_alt_vs_controlled_default | naturalness | 0.0028 | 0.0033 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0002 | -0.0005 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0009 | 0.0078 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0569 | -0.1535 |
| controlled_alt_vs_controlled_default | persona_style | 0.0053 | 0.0088 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0030 | -0.0033 |
| controlled_alt_vs_controlled_default | length_score | 0.0458 | 0.0912 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0292 | -0.0310 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0162 | -0.0356 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.2412 | 4.9421 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.2392 | 1.7806 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0607 | 0.0755 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.3137 | 6.4954 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0720 | 1.4408 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.2665 | 5.5958 |
| controlled_alt_vs_proposed_raw | persona_style | 0.1300 | 0.2701 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0085 | -0.0091 |
| controlled_alt_vs_proposed_raw | length_score | 0.2708 | 0.9750 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1458 | 0.1902 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.2118 | 0.9392 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2618 | 9.2765 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.2318 | 1.6353 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0560 | 0.0692 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3399 | 15.3829 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0795 | 1.8705 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.2466 | 3.6559 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.1723 | 0.3927 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0083 | -0.0089 |
| controlled_alt_vs_candidate_no_context | length_score | 0.2250 | 0.6953 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1896 | 0.2622 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.2170 | 0.9845 |
| controlled_alt_vs_baseline_no_context | context_relevance | 0.2470 | 5.7426 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 0.1592 | 0.7428 |
| controlled_alt_vs_baseline_no_context | naturalness | -0.0190 | -0.0215 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 0.3203 | 7.6747 |
| controlled_alt_vs_baseline_no_context | context_overlap | 0.0760 | 1.6527 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 0.1835 | 1.4058 |
| controlled_alt_vs_baseline_no_context | persona_style | 0.0618 | 0.1126 |
| controlled_alt_vs_baseline_no_context | distinct1 | -0.0604 | -0.0614 |
| controlled_alt_vs_baseline_no_context | length_score | 0.0125 | 0.0233 |
| controlled_alt_vs_baseline_no_context | sentence_score | 0.0729 | 0.0868 |
| controlled_alt_vs_baseline_no_context | overall_quality | 0.1695 | 0.6329 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2468 | 5.7390 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.2037 | 0.9504 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0218 | -0.0247 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3205 | 7.6793 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0751 | 1.6320 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2405 | 1.8419 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0565 | 0.1029 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0574 | -0.0583 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | -0.0333 | -0.0622 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.1021 | 0.1216 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1857 | 0.6932 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0206 | (-0.0052, 0.0561) | 0.0797 | 0.0206 | (0.0011, 0.0465) | 0.0107 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0074 | (-0.0353, 0.0126) | 0.6863 | -0.0074 | (-0.0338, 0.0147) | 0.7177 |
| proposed_vs_candidate_no_context | naturalness | -0.0047 | (-0.0404, 0.0292) | 0.5857 | -0.0047 | (-0.0372, 0.0284) | 0.6523 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0262 | (-0.0079, 0.0707) | 0.1127 | 0.0262 | (0.0000, 0.0615) | 0.0263 |
| proposed_vs_candidate_no_context | context_overlap | 0.0075 | (-0.0027, 0.0174) | 0.0777 | 0.0075 | (0.0012, 0.0160) | 0.0083 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0198 | (-0.0536, 0.0000) | 1.0000 | -0.0198 | (-0.0476, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | persona_style | 0.0423 | (0.0104, 0.0847) | 0.0107 | 0.0423 | (0.0081, 0.0924) | 0.0240 |
| proposed_vs_candidate_no_context | distinct1 | 0.0002 | (-0.0126, 0.0150) | 0.4847 | 0.0002 | (-0.0086, 0.0090) | 0.4860 |
| proposed_vs_candidate_no_context | length_score | -0.0458 | (-0.1736, 0.0806) | 0.7703 | -0.0458 | (-0.1849, 0.0764) | 0.7837 |
| proposed_vs_candidate_no_context | sentence_score | 0.0437 | (-0.0292, 0.1167) | 0.1557 | 0.0437 | (-0.0241, 0.1289) | 0.1547 |
| proposed_vs_candidate_no_context | overall_quality | 0.0051 | (-0.0188, 0.0291) | 0.3417 | 0.0051 | (-0.0151, 0.0257) | 0.3160 |
| proposed_vs_baseline_no_context | context_relevance | 0.0058 | (-0.0245, 0.0445) | 0.4017 | 0.0058 | (-0.0251, 0.0447) | 0.3807 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0800 | (-0.1260, -0.0351) | 1.0000 | -0.0800 | (-0.1256, -0.0270) | 0.9983 |
| proposed_vs_baseline_no_context | naturalness | -0.0797 | (-0.1069, -0.0525) | 1.0000 | -0.0797 | (-0.1133, -0.0388) | 1.0000 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0066 | (-0.0348, 0.0556) | 0.4327 | 0.0066 | (-0.0354, 0.0600) | 0.4273 |
| proposed_vs_baseline_no_context | context_overlap | 0.0040 | (-0.0105, 0.0194) | 0.3013 | 0.0040 | (-0.0060, 0.0163) | 0.2387 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0829 | (-0.1397, -0.0333) | 1.0000 | -0.0829 | (-0.1316, -0.0225) | 1.0000 |
| proposed_vs_baseline_no_context | persona_style | -0.0681 | (-0.1745, 0.0338) | 0.8987 | -0.0681 | (-0.2272, 0.0454) | 0.7210 |
| proposed_vs_baseline_no_context | distinct1 | -0.0519 | (-0.0676, -0.0359) | 1.0000 | -0.0519 | (-0.0643, -0.0381) | 1.0000 |
| proposed_vs_baseline_no_context | length_score | -0.2583 | (-0.3514, -0.1681) | 1.0000 | -0.2583 | (-0.3679, -0.1296) | 1.0000 |
| proposed_vs_baseline_no_context | sentence_score | -0.0729 | (-0.1604, 0.0146) | 0.9547 | -0.0729 | (-0.2019, 0.0955) | 0.8137 |
| proposed_vs_baseline_no_context | overall_quality | -0.0423 | (-0.0722, -0.0107) | 0.9963 | -0.0423 | (-0.0760, -0.0018) | 0.9763 |
| controlled_vs_proposed_raw | context_relevance | 0.2410 | (0.1972, 0.2909) | 0.0000 | 0.2410 | (0.1930, 0.2841) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.2837 | (0.2174, 0.3538) | 0.0000 | 0.2837 | (0.2216, 0.3428) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0579 | (0.0163, 0.0984) | 0.0037 | 0.0579 | (0.0197, 0.0972) | 0.0020 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.3139 | (0.2552, 0.3783) | 0.0000 | 0.3139 | (0.2516, 0.3684) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0711 | (0.0534, 0.0906) | 0.0000 | 0.0711 | (0.0559, 0.0874) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.3234 | (0.2468, 0.4117) | 0.0000 | 0.3234 | (0.2356, 0.4087) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.1247 | (0.0236, 0.2308) | 0.0073 | 0.1247 | (0.0000, 0.2772) | 0.0253 |
| controlled_vs_proposed_raw | distinct1 | -0.0055 | (-0.0305, 0.0184) | 0.6663 | -0.0055 | (-0.0162, 0.0054) | 0.8533 |
| controlled_vs_proposed_raw | length_score | 0.2250 | (0.0597, 0.3819) | 0.0043 | 0.2250 | (0.0700, 0.3880) | 0.0033 |
| controlled_vs_proposed_raw | sentence_score | 0.1750 | (0.0875, 0.2625) | 0.0000 | 0.1750 | (0.0525, 0.2625) | 0.0037 |
| controlled_vs_proposed_raw | overall_quality | 0.2280 | (0.1867, 0.2690) | 0.0000 | 0.2280 | (0.1905, 0.2570) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2616 | (0.2223, 0.3066) | 0.0000 | 0.2616 | (0.2233, 0.2937) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.2763 | (0.2054, 0.3469) | 0.0000 | 0.2763 | (0.2120, 0.3417) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.0531 | (0.0105, 0.0988) | 0.0090 | 0.0531 | (0.0248, 0.0797) | 0.0003 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3401 | (0.2853, 0.4009) | 0.0000 | 0.3401 | (0.2866, 0.3861) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0786 | (0.0616, 0.0974) | 0.0000 | 0.0786 | (0.0664, 0.0932) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.3036 | (0.2171, 0.3901) | 0.0000 | 0.3036 | (0.2063, 0.3898) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.1670 | (0.0630, 0.2795) | 0.0020 | 0.1670 | (0.0268, 0.3129) | 0.0043 |
| controlled_vs_candidate_no_context | distinct1 | -0.0053 | (-0.0299, 0.0156) | 0.6653 | -0.0053 | (-0.0177, 0.0062) | 0.8197 |
| controlled_vs_candidate_no_context | length_score | 0.1792 | (-0.0014, 0.3542) | 0.0257 | 0.1792 | (0.0797, 0.2760) | 0.0003 |
| controlled_vs_candidate_no_context | sentence_score | 0.2188 | (0.1313, 0.2917) | 0.0000 | 0.2188 | (0.1167, 0.2891) | 0.0003 |
| controlled_vs_candidate_no_context | overall_quality | 0.2331 | (0.1965, 0.2694) | 0.0000 | 0.2331 | (0.2056, 0.2570) | 0.0000 |
| controlled_vs_baseline_no_context | context_relevance | 0.2468 | (0.2046, 0.2974) | 0.0000 | 0.2468 | (0.2064, 0.2831) | 0.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 0.2037 | (0.1383, 0.2682) | 0.0000 | 0.2037 | (0.1445, 0.2609) | 0.0000 |
| controlled_vs_baseline_no_context | naturalness | -0.0218 | (-0.0580, 0.0137) | 0.8887 | -0.0218 | (-0.0611, 0.0250) | 0.8320 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3205 | (0.2614, 0.3842) | 0.0000 | 0.3205 | (0.2609, 0.3739) | 0.0000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0751 | (0.0574, 0.0941) | 0.0000 | 0.0751 | (0.0648, 0.0870) | 0.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2405 | (0.1641, 0.3163) | 0.0000 | 0.2405 | (0.1752, 0.3031) | 0.0000 |
| controlled_vs_baseline_no_context | persona_style | 0.0565 | (0.0069, 0.1130) | 0.0060 | 0.0565 | (0.0000, 0.1347) | 0.0253 |
| controlled_vs_baseline_no_context | distinct1 | -0.0574 | (-0.0840, -0.0349) | 1.0000 | -0.0574 | (-0.0768, -0.0342) | 1.0000 |
| controlled_vs_baseline_no_context | length_score | -0.0333 | (-0.1819, 0.1111) | 0.6680 | -0.0333 | (-0.1818, 0.1630) | 0.6470 |
| controlled_vs_baseline_no_context | sentence_score | 0.1021 | (0.0146, 0.1896) | 0.0203 | 0.1021 | (0.0250, 0.2100) | 0.0023 |
| controlled_vs_baseline_no_context | overall_quality | 0.1857 | (0.1492, 0.2203) | 0.0000 | 0.1857 | (0.1650, 0.2101) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0002 | (-0.0593, 0.0624) | 0.4937 | 0.0002 | (-0.0421, 0.0416) | 0.4930 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0445 | (-0.0992, 0.0133) | 0.9390 | -0.0445 | (-0.1016, -0.0006) | 0.9790 |
| controlled_alt_vs_controlled_default | naturalness | 0.0028 | (-0.0468, 0.0552) | 0.4833 | 0.0028 | (-0.0404, 0.0406) | 0.4597 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0002 | (-0.0795, 0.0794) | 0.5113 | -0.0002 | (-0.0508, 0.0548) | 0.4923 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0009 | (-0.0209, 0.0218) | 0.4717 | 0.0009 | (-0.0198, 0.0204) | 0.4677 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0569 | (-0.1202, 0.0109) | 0.9497 | -0.0569 | (-0.1243, -0.0076) | 0.9910 |
| controlled_alt_vs_controlled_default | persona_style | 0.0053 | (-0.0486, 0.0595) | 0.4440 | 0.0053 | (-0.0278, 0.0520) | 0.4553 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0030 | (-0.0372, 0.0332) | 0.5587 | -0.0030 | (-0.0346, 0.0221) | 0.5857 |
| controlled_alt_vs_controlled_default | length_score | 0.0458 | (-0.1334, 0.2333) | 0.3083 | 0.0458 | (-0.0778, 0.1500) | 0.2477 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0292 | (-0.1021, 0.0437) | 0.8223 | -0.0292 | (-0.1120, 0.0477) | 0.8180 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0162 | (-0.0467, 0.0125) | 0.8540 | -0.0162 | (-0.0428, 0.0054) | 0.9317 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.2412 | (0.1892, 0.2938) | 0.0000 | 0.2412 | (0.2208, 0.2608) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.2392 | (0.1779, 0.3072) | 0.0000 | 0.2392 | (0.1923, 0.2743) | 0.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0607 | (0.0035, 0.1130) | 0.0200 | 0.0607 | (-0.0053, 0.1098) | 0.0347 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.3137 | (0.2499, 0.3855) | 0.0000 | 0.3137 | (0.2855, 0.3406) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0720 | (0.0566, 0.0889) | 0.0000 | 0.0720 | (0.0585, 0.0859) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.2665 | (0.2006, 0.3419) | 0.0000 | 0.2665 | (0.2119, 0.3157) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 0.1300 | (0.0408, 0.2391) | 0.0007 | 0.1300 | (0.0179, 0.2791) | 0.0037 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0085 | (-0.0416, 0.0200) | 0.6853 | -0.0085 | (-0.0459, 0.0185) | 0.7047 |
| controlled_alt_vs_proposed_raw | length_score | 0.2708 | (0.0833, 0.4459) | 0.0023 | 0.2708 | (0.0923, 0.4278) | 0.0037 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1458 | (0.0583, 0.2333) | 0.0013 | 0.1458 | (0.0304, 0.2293) | 0.0110 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.2118 | (0.1730, 0.2556) | 0.0000 | 0.2118 | (0.1870, 0.2299) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2618 | (0.2119, 0.3150) | 0.0000 | 0.2618 | (0.2355, 0.2960) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.2318 | (0.1652, 0.3009) | 0.0000 | 0.2318 | (0.1821, 0.2780) | 0.0000 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0560 | (0.0051, 0.1089) | 0.0167 | 0.0560 | (-0.0038, 0.0964) | 0.0310 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3399 | (0.2721, 0.4100) | 0.0000 | 0.3399 | (0.3045, 0.3879) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0795 | (0.0653, 0.0961) | 0.0000 | 0.0795 | (0.0630, 0.0974) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.2466 | (0.1722, 0.3300) | 0.0000 | 0.2466 | (0.1762, 0.3088) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.1723 | (0.0725, 0.2791) | 0.0000 | 0.1723 | (0.0400, 0.3237) | 0.0017 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0083 | (-0.0420, 0.0202) | 0.6717 | -0.0083 | (-0.0495, 0.0187) | 0.6727 |
| controlled_alt_vs_candidate_no_context | length_score | 0.2250 | (0.0361, 0.4167) | 0.0107 | 0.2250 | (0.0394, 0.3583) | 0.0107 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1896 | (0.1167, 0.2625) | 0.0000 | 0.1896 | (0.1167, 0.2520) | 0.0000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.2170 | (0.1730, 0.2611) | 0.0000 | 0.2170 | (0.1899, 0.2419) | 0.0000 |
| controlled_alt_vs_baseline_no_context | context_relevance | 0.2470 | (0.1970, 0.2964) | 0.0000 | 0.2470 | (0.2080, 0.2894) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 0.1592 | (0.0897, 0.2344) | 0.0000 | 0.1592 | (0.1195, 0.2065) | 0.0000 |
| controlled_alt_vs_baseline_no_context | naturalness | -0.0190 | (-0.0759, 0.0339) | 0.7420 | -0.0190 | (-0.0729, 0.0453) | 0.7390 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 0.3203 | (0.2547, 0.3875) | 0.0000 | 0.3203 | (0.2686, 0.3820) | 0.0000 |
| controlled_alt_vs_baseline_no_context | context_overlap | 0.0760 | (0.0608, 0.0916) | 0.0000 | 0.0760 | (0.0621, 0.0910) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 0.1835 | (0.0984, 0.2740) | 0.0000 | 0.1835 | (0.1371, 0.2381) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_style | 0.0618 | (0.0063, 0.1299) | 0.0147 | 0.0618 | (0.0161, 0.1342) | 0.0227 |
| controlled_alt_vs_baseline_no_context | distinct1 | -0.0604 | (-0.0933, -0.0324) | 1.0000 | -0.0604 | (-0.0954, -0.0347) | 1.0000 |
| controlled_alt_vs_baseline_no_context | length_score | 0.0125 | (-0.1958, 0.2056) | 0.4700 | 0.0125 | (-0.1855, 0.2508) | 0.4643 |
| controlled_alt_vs_baseline_no_context | sentence_score | 0.0729 | (0.0000, 0.1458) | 0.0260 | 0.0729 | (0.0000, 0.1575) | 0.0380 |
| controlled_alt_vs_baseline_no_context | overall_quality | 0.1695 | (0.1286, 0.2095) | 0.0000 | 0.1695 | (0.1408, 0.2027) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2468 | (0.2037, 0.2968) | 0.0000 | 0.2468 | (0.2041, 0.2835) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.2037 | (0.1393, 0.2664) | 0.0000 | 0.2037 | (0.1486, 0.2626) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0218 | (-0.0568, 0.0137) | 0.8750 | -0.0218 | (-0.0608, 0.0264) | 0.8167 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3205 | (0.2626, 0.3845) | 0.0000 | 0.3205 | (0.2605, 0.3716) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0751 | (0.0573, 0.0954) | 0.0000 | 0.0751 | (0.0648, 0.0863) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2405 | (0.1623, 0.3222) | 0.0000 | 0.2405 | (0.1756, 0.2993) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0565 | (0.0079, 0.1146) | 0.0100 | 0.0565 | (0.0037, 0.1350) | 0.0210 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0574 | (-0.0838, -0.0360) | 1.0000 | -0.0574 | (-0.0768, -0.0341) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | -0.0333 | (-0.1889, 0.1111) | 0.6697 | -0.0333 | (-0.1819, 0.1714) | 0.6303 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.1021 | (0.0146, 0.1896) | 0.0183 | 0.1021 | (0.0259, 0.2100) | 0.0047 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1857 | (0.1490, 0.2200) | 0.0000 | 0.1857 | (0.1648, 0.2114) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 8 | 5 | 11 | 0.5625 | 0.6154 |
| proposed_vs_candidate_no_context | persona_consistency | 4 | 2 | 18 | 0.5417 | 0.6667 |
| proposed_vs_candidate_no_context | naturalness | 6 | 7 | 11 | 0.4792 | 0.4615 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 6 | 3 | 15 | 0.5625 | 0.6667 |
| proposed_vs_candidate_no_context | context_overlap | 8 | 5 | 11 | 0.5625 | 0.6154 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0 | 2 | 22 | 0.4583 | 0.0000 |
| proposed_vs_candidate_no_context | persona_style | 4 | 0 | 20 | 0.5833 | 1.0000 |
| proposed_vs_candidate_no_context | distinct1 | 5 | 7 | 12 | 0.4583 | 0.4167 |
| proposed_vs_candidate_no_context | length_score | 4 | 8 | 12 | 0.4167 | 0.3333 |
| proposed_vs_candidate_no_context | sentence_score | 5 | 2 | 17 | 0.5625 | 0.7143 |
| proposed_vs_candidate_no_context | overall_quality | 7 | 6 | 11 | 0.5208 | 0.5385 |
| proposed_vs_baseline_no_context | context_relevance | 7 | 17 | 0 | 0.2917 | 0.2917 |
| proposed_vs_baseline_no_context | persona_consistency | 2 | 12 | 10 | 0.2917 | 0.1429 |
| proposed_vs_baseline_no_context | naturalness | 4 | 20 | 0 | 0.1667 | 0.1667 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 4 | 6 | 14 | 0.4583 | 0.4000 |
| proposed_vs_baseline_no_context | context_overlap | 10 | 14 | 0 | 0.4167 | 0.4167 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 0 | 8 | 16 | 0.3333 | 0.0000 |
| proposed_vs_baseline_no_context | persona_style | 2 | 5 | 17 | 0.4375 | 0.2857 |
| proposed_vs_baseline_no_context | distinct1 | 3 | 19 | 2 | 0.1667 | 0.1364 |
| proposed_vs_baseline_no_context | length_score | 2 | 20 | 2 | 0.1250 | 0.0909 |
| proposed_vs_baseline_no_context | sentence_score | 3 | 8 | 13 | 0.3958 | 0.2727 |
| proposed_vs_baseline_no_context | overall_quality | 6 | 18 | 0 | 0.2500 | 0.2500 |
| controlled_vs_proposed_raw | context_relevance | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_proposed_raw | persona_consistency | 23 | 0 | 1 | 0.9792 | 1.0000 |
| controlled_vs_proposed_raw | naturalness | 17 | 7 | 0 | 0.7083 | 0.7083 |
| controlled_vs_proposed_raw | context_keyword_coverage | 23 | 0 | 1 | 0.9792 | 1.0000 |
| controlled_vs_proposed_raw | context_overlap | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 23 | 0 | 1 | 0.9792 | 1.0000 |
| controlled_vs_proposed_raw | persona_style | 8 | 2 | 14 | 0.6250 | 0.8000 |
| controlled_vs_proposed_raw | distinct1 | 13 | 11 | 0 | 0.5417 | 0.5417 |
| controlled_vs_proposed_raw | length_score | 17 | 7 | 0 | 0.7083 | 0.7083 |
| controlled_vs_proposed_raw | sentence_score | 14 | 2 | 8 | 0.7500 | 0.8750 |
| controlled_vs_proposed_raw | overall_quality | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_relevance | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_vs_candidate_no_context | naturalness | 15 | 8 | 1 | 0.6458 | 0.6522 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 22 | 1 | 1 | 0.9375 | 0.9565 |
| controlled_vs_candidate_no_context | persona_style | 10 | 1 | 13 | 0.6875 | 0.9091 |
| controlled_vs_candidate_no_context | distinct1 | 12 | 11 | 1 | 0.5208 | 0.5217 |
| controlled_vs_candidate_no_context | length_score | 15 | 8 | 1 | 0.6458 | 0.6522 |
| controlled_vs_candidate_no_context | sentence_score | 16 | 1 | 7 | 0.8125 | 0.9412 |
| controlled_vs_candidate_no_context | overall_quality | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | context_relevance | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 20 | 1 | 3 | 0.8958 | 0.9524 |
| controlled_vs_baseline_no_context | naturalness | 11 | 13 | 0 | 0.4583 | 0.4583 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | context_overlap | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 19 | 1 | 4 | 0.8750 | 0.9500 |
| controlled_vs_baseline_no_context | persona_style | 6 | 1 | 17 | 0.6042 | 0.8571 |
| controlled_vs_baseline_no_context | distinct1 | 3 | 21 | 0 | 0.1250 | 0.1250 |
| controlled_vs_baseline_no_context | length_score | 9 | 11 | 4 | 0.4583 | 0.4500 |
| controlled_vs_baseline_no_context | sentence_score | 9 | 2 | 13 | 0.6458 | 0.8182 |
| controlled_vs_baseline_no_context | overall_quality | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_alt_vs_controlled_default | context_relevance | 9 | 12 | 3 | 0.4375 | 0.4286 |
| controlled_alt_vs_controlled_default | persona_consistency | 5 | 10 | 9 | 0.3958 | 0.3333 |
| controlled_alt_vs_controlled_default | naturalness | 9 | 13 | 2 | 0.4167 | 0.4091 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 9 | 8 | 7 | 0.5208 | 0.5294 |
| controlled_alt_vs_controlled_default | context_overlap | 10 | 11 | 3 | 0.4792 | 0.4762 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 3 | 9 | 12 | 0.3750 | 0.2500 |
| controlled_alt_vs_controlled_default | persona_style | 2 | 2 | 20 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | distinct1 | 8 | 13 | 3 | 0.3958 | 0.3810 |
| controlled_alt_vs_controlled_default | length_score | 8 | 10 | 6 | 0.4583 | 0.4444 |
| controlled_alt_vs_controlled_default | sentence_score | 3 | 5 | 16 | 0.4583 | 0.3750 |
| controlled_alt_vs_controlled_default | overall_quality | 11 | 11 | 2 | 0.5000 | 0.5000 |
| controlled_alt_vs_proposed_raw | context_relevance | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 23 | 0 | 1 | 0.9792 | 1.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 16 | 8 | 0 | 0.6667 | 0.6667 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 23 | 0 | 1 | 0.9792 | 1.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 23 | 0 | 1 | 0.9792 | 1.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 7 | 1 | 16 | 0.6250 | 0.8750 |
| controlled_alt_vs_proposed_raw | distinct1 | 14 | 9 | 1 | 0.6042 | 0.6087 |
| controlled_alt_vs_proposed_raw | length_score | 15 | 9 | 0 | 0.6250 | 0.6250 |
| controlled_alt_vs_proposed_raw | sentence_score | 12 | 2 | 10 | 0.7083 | 0.8571 |
| controlled_alt_vs_proposed_raw | overall_quality | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 22 | 1 | 1 | 0.9375 | 0.9565 |
| controlled_alt_vs_candidate_no_context | naturalness | 17 | 7 | 0 | 0.7083 | 0.7083 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 23 | 0 | 1 | 0.9792 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 22 | 1 | 1 | 0.9375 | 0.9565 |
| controlled_alt_vs_candidate_no_context | persona_style | 9 | 0 | 15 | 0.6875 | 1.0000 |
| controlled_alt_vs_candidate_no_context | distinct1 | 14 | 10 | 0 | 0.5833 | 0.5833 |
| controlled_alt_vs_candidate_no_context | length_score | 14 | 10 | 0 | 0.5833 | 0.5833 |
| controlled_alt_vs_candidate_no_context | sentence_score | 13 | 0 | 11 | 0.7708 | 1.0000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_alt_vs_baseline_no_context | context_relevance | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 19 | 1 | 4 | 0.8750 | 0.9500 |
| controlled_alt_vs_baseline_no_context | naturalness | 13 | 11 | 0 | 0.5417 | 0.5417 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 23 | 0 | 1 | 0.9792 | 1.0000 |
| controlled_alt_vs_baseline_no_context | context_overlap | 23 | 0 | 1 | 0.9792 | 1.0000 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 19 | 1 | 4 | 0.8750 | 0.9500 |
| controlled_alt_vs_baseline_no_context | persona_style | 5 | 1 | 18 | 0.5833 | 0.8333 |
| controlled_alt_vs_baseline_no_context | distinct1 | 3 | 19 | 2 | 0.1667 | 0.1364 |
| controlled_alt_vs_baseline_no_context | length_score | 13 | 11 | 0 | 0.5417 | 0.5417 |
| controlled_alt_vs_baseline_no_context | sentence_score | 6 | 1 | 17 | 0.6042 | 0.8571 |
| controlled_alt_vs_baseline_no_context | overall_quality | 23 | 1 | 0 | 0.9583 | 0.9583 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 24 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 20 | 1 | 3 | 0.8958 | 0.9524 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 11 | 13 | 0 | 0.4583 | 0.4583 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 24 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 24 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 19 | 1 | 4 | 0.8750 | 0.9500 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 6 | 1 | 17 | 0.6042 | 0.8571 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | 3 | 21 | 0 | 0.1250 | 0.1250 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 9 | 11 | 4 | 0.4583 | 0.4500 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 9 | 2 | 13 | 0.6458 | 0.8182 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 23 | 1 | 0 | 0.9583 | 0.9583 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.1667 | 0.4167 | 0.5833 |
| proposed_contextual_controlled_alt | 0.0000 | 0.0000 | 0.2083 | 0.4167 | 0.5833 |
| proposed_contextual | 0.0000 | 0.0000 | 0.5417 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.6250 | 0.0000 | 0.0000 |
| baseline_no_context | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `8`
- Unique template signatures: `22`
- Template signature ratio: `0.9167`
- Effective sample size by source clustering: `7.20`
- Effective sample size by template-signature clustering: `20.57`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: unavailable (disabled_by_flag).

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
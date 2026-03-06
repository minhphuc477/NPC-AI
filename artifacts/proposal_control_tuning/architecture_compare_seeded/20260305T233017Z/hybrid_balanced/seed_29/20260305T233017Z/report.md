# Proposal Alignment Evaluation Report

- Run ID: `20260305T233017Z`
- Generated: `2026-03-05T23:35:21.489595+00:00`
- Scenarios: `artifacts\proposal_control_tuning\architecture_compare_seeded\20260305T233017Z\hybrid_balanced\seed_29\20260305T233017Z\scenarios.jsonl`
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
| proposed_contextual_controlled | 0.2547 (0.2202, 0.2928) | 0.3308 (0.2773, 0.3842) | 0.8977 (0.8693, 0.9220) | 0.4117 (0.3866, 0.4400) | n/a |
| proposed_contextual_controlled_alt | 0.2754 (0.2239, 0.3340) | 0.3767 (0.3076, 0.4536) | 0.8746 (0.8437, 0.9035) | 0.4336 (0.4022, 0.4692) | n/a |
| proposed_contextual | 0.0601 (0.0257, 0.1032) | 0.1207 (0.0822, 0.1634) | 0.8045 (0.7788, 0.8301) | 0.2262 (0.1975, 0.2579) | n/a |
| candidate_no_context | 0.0304 (0.0169, 0.0471) | 0.1820 (0.1189, 0.2531) | 0.8078 (0.7783, 0.8386) | 0.2358 (0.2046, 0.2719) | n/a |
| baseline_no_context | 0.0381 (0.0216, 0.0564) | 0.2064 (0.1566, 0.2595) | 0.8937 (0.8714, 0.9149) | 0.2648 (0.2421, 0.2880) | n/a |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0297 | 0.9766 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0613 | -0.3368 |
| proposed_vs_candidate_no_context | naturalness | -0.0032 | -0.0040 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0379 | 1.4634 |
| proposed_vs_candidate_no_context | context_overlap | 0.0107 | 0.2602 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0720 | -0.6482 |
| proposed_vs_candidate_no_context | persona_style | -0.0184 | -0.0395 |
| proposed_vs_candidate_no_context | distinct1 | 0.0051 | 0.0055 |
| proposed_vs_candidate_no_context | length_score | -0.0264 | -0.0888 |
| proposed_vs_candidate_no_context | sentence_score | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | overall_quality | -0.0096 | -0.0408 |
| proposed_vs_baseline_no_context | context_relevance | 0.0220 | 0.5777 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0857 | -0.4152 |
| proposed_vs_baseline_no_context | naturalness | -0.0892 | -0.0998 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0271 | 0.7414 |
| proposed_vs_baseline_no_context | context_overlap | 0.0101 | 0.2418 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0861 | -0.6878 |
| proposed_vs_baseline_no_context | persona_style | -0.0839 | -0.1581 |
| proposed_vs_baseline_no_context | distinct1 | -0.0271 | -0.0280 |
| proposed_vs_baseline_no_context | length_score | -0.3042 | -0.5290 |
| proposed_vs_baseline_no_context | sentence_score | -0.1750 | -0.1918 |
| proposed_vs_baseline_no_context | overall_quality | -0.0386 | -0.1458 |
| controlled_vs_proposed_raw | context_relevance | 0.1945 | 3.2345 |
| controlled_vs_proposed_raw | persona_consistency | 0.2101 | 1.7407 |
| controlled_vs_proposed_raw | naturalness | 0.0932 | 0.1158 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2564 | 4.0218 |
| controlled_vs_proposed_raw | context_overlap | 0.0501 | 0.9694 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2210 | 5.6548 |
| controlled_vs_proposed_raw | persona_style | 0.1662 | 0.3718 |
| controlled_vs_proposed_raw | distinct1 | -0.0157 | -0.0166 |
| controlled_vs_proposed_raw | length_score | 0.4042 | 1.4923 |
| controlled_vs_proposed_raw | sentence_score | 0.1896 | 0.2571 |
| controlled_vs_proposed_raw | overall_quality | 0.1855 | 0.8199 |
| controlled_vs_candidate_no_context | context_relevance | 0.2243 | 7.3700 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1488 | 0.8176 |
| controlled_vs_candidate_no_context | naturalness | 0.0900 | 0.1114 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.2943 | 11.3707 |
| controlled_vs_candidate_no_context | context_overlap | 0.0608 | 1.4820 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1490 | 1.3411 |
| controlled_vs_candidate_no_context | persona_style | 0.1479 | 0.3177 |
| controlled_vs_candidate_no_context | distinct1 | -0.0105 | -0.0112 |
| controlled_vs_candidate_no_context | length_score | 0.3778 | 1.2710 |
| controlled_vs_candidate_no_context | sentence_score | 0.1896 | 0.2571 |
| controlled_vs_candidate_no_context | overall_quality | 0.1758 | 0.7456 |
| controlled_vs_baseline_no_context | context_relevance | 0.2166 | 5.6805 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1244 | 0.6028 |
| controlled_vs_baseline_no_context | naturalness | 0.0040 | 0.0045 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.2836 | 7.7448 |
| controlled_vs_baseline_no_context | context_overlap | 0.0602 | 1.4456 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1349 | 1.0777 |
| controlled_vs_baseline_no_context | persona_style | 0.0823 | 0.1550 |
| controlled_vs_baseline_no_context | distinct1 | -0.0428 | -0.0441 |
| controlled_vs_baseline_no_context | length_score | 0.1000 | 0.1739 |
| controlled_vs_baseline_no_context | sentence_score | 0.0146 | 0.0160 |
| controlled_vs_baseline_no_context | overall_quality | 0.1469 | 0.5546 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0207 | 0.0811 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0460 | 0.1390 |
| controlled_alt_vs_controlled_default | naturalness | -0.0231 | -0.0257 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0282 | 0.0879 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0032 | 0.0313 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0512 | 0.1968 |
| controlled_alt_vs_controlled_default | persona_style | 0.0251 | 0.0409 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0090 | 0.0097 |
| controlled_alt_vs_controlled_default | length_score | -0.1458 | -0.2160 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0437 | 0.0472 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0219 | 0.0533 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.2152 | 3.5780 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.2560 | 2.1217 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0701 | 0.0871 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2846 | 4.4634 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0533 | 1.0311 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.2722 | 6.9645 |
| controlled_alt_vs_proposed_raw | persona_style | 0.1913 | 0.4280 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0067 | -0.0071 |
| controlled_alt_vs_proposed_raw | length_score | 0.2583 | 0.9538 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.2333 | 0.3164 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.2074 | 0.9169 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2449 | 8.0492 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1947 | 1.0702 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0669 | 0.0828 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3225 | 12.4585 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0640 | 1.5597 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.2002 | 1.8018 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.1729 | 0.3716 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0015 | -0.0016 |
| controlled_alt_vs_candidate_no_context | length_score | 0.2319 | 0.7804 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.2333 | 0.3164 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1978 | 0.8387 |
| controlled_alt_vs_baseline_no_context | context_relevance | 0.2372 | 6.2226 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 0.1704 | 0.8256 |
| controlled_alt_vs_baseline_no_context | naturalness | -0.0191 | -0.0214 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 0.3117 | 8.5138 |
| controlled_alt_vs_baseline_no_context | context_overlap | 0.0634 | 1.5222 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 0.1861 | 1.4865 |
| controlled_alt_vs_baseline_no_context | persona_style | 0.1074 | 0.2022 |
| controlled_alt_vs_baseline_no_context | distinct1 | -0.0338 | -0.0349 |
| controlled_alt_vs_baseline_no_context | length_score | -0.0458 | -0.0797 |
| controlled_alt_vs_baseline_no_context | sentence_score | 0.0583 | 0.0639 |
| controlled_alt_vs_baseline_no_context | overall_quality | 0.1688 | 0.6374 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2166 | 5.6805 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1244 | 0.6028 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 0.0040 | 0.0045 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.2836 | 7.7448 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0602 | 1.4456 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1349 | 1.0777 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0823 | 0.1550 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0428 | -0.0441 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.1000 | 0.1739 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0146 | 0.0160 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1469 | 0.5546 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0297 | (-0.0029, 0.0734) | 0.0463 | 0.0297 | (-0.0025, 0.0659) | 0.0390 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0613 | (-0.1298, 0.0006) | 0.9727 | -0.0613 | (-0.1720, 0.0172) | 0.9027 |
| proposed_vs_candidate_no_context | naturalness | -0.0032 | (-0.0339, 0.0261) | 0.5760 | -0.0032 | (-0.0470, 0.0290) | 0.5397 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0379 | (-0.0076, 0.0947) | 0.0783 | 0.0379 | (-0.0048, 0.0866) | 0.0603 |
| proposed_vs_candidate_no_context | context_overlap | 0.0107 | (0.0019, 0.0206) | 0.0060 | 0.0107 | (0.0018, 0.0198) | 0.0063 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0720 | (-0.1544, -0.0022) | 0.9837 | -0.0720 | (-0.2081, 0.0175) | 0.8937 |
| proposed_vs_candidate_no_context | persona_style | -0.0184 | (-0.0700, 0.0185) | 0.7773 | -0.0184 | (-0.0631, 0.0244) | 0.7840 |
| proposed_vs_candidate_no_context | distinct1 | 0.0051 | (-0.0104, 0.0210) | 0.2483 | 0.0051 | (-0.0156, 0.0225) | 0.2860 |
| proposed_vs_candidate_no_context | length_score | -0.0264 | (-0.1361, 0.0889) | 0.6550 | -0.0264 | (-0.1712, 0.0941) | 0.6367 |
| proposed_vs_candidate_no_context | sentence_score | 0.0000 | (-0.0875, 0.0729) | 0.5640 | 0.0000 | (-0.0833, 0.0795) | 0.5440 |
| proposed_vs_candidate_no_context | overall_quality | -0.0096 | (-0.0481, 0.0288) | 0.6990 | -0.0096 | (-0.0677, 0.0303) | 0.6217 |
| proposed_vs_baseline_no_context | context_relevance | 0.0220 | (-0.0183, 0.0710) | 0.1750 | 0.0220 | (-0.0107, 0.0531) | 0.0920 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0857 | (-0.1462, -0.0285) | 0.9993 | -0.0857 | (-0.1654, -0.0011) | 0.9770 |
| proposed_vs_baseline_no_context | naturalness | -0.0892 | (-0.1238, -0.0496) | 1.0000 | -0.0892 | (-0.1276, -0.0384) | 1.0000 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0271 | (-0.0284, 0.0947) | 0.1903 | 0.0271 | (-0.0159, 0.0672) | 0.0993 |
| proposed_vs_baseline_no_context | context_overlap | 0.0101 | (-0.0024, 0.0236) | 0.0590 | 0.0101 | (-0.0067, 0.0259) | 0.1110 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0861 | (-0.1573, -0.0236) | 0.9967 | -0.0861 | (-0.1776, -0.0066) | 0.9830 |
| proposed_vs_baseline_no_context | persona_style | -0.0839 | (-0.1945, 0.0178) | 0.9473 | -0.0839 | (-0.2812, 0.0655) | 0.7613 |
| proposed_vs_baseline_no_context | distinct1 | -0.0271 | (-0.0469, -0.0083) | 0.9957 | -0.0271 | (-0.0479, -0.0064) | 0.9970 |
| proposed_vs_baseline_no_context | length_score | -0.3042 | (-0.4306, -0.1694) | 1.0000 | -0.3042 | (-0.4488, -0.1166) | 0.9987 |
| proposed_vs_baseline_no_context | sentence_score | -0.1750 | (-0.2771, -0.0583) | 1.0000 | -0.1750 | (-0.2683, -0.0553) | 1.0000 |
| proposed_vs_baseline_no_context | overall_quality | -0.0386 | (-0.0762, -0.0002) | 0.9757 | -0.0386 | (-0.0745, -0.0038) | 0.9843 |
| controlled_vs_proposed_raw | context_relevance | 0.1945 | (0.1232, 0.2588) | 0.0000 | 0.1945 | (0.1507, 0.2538) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.2101 | (0.1445, 0.2769) | 0.0000 | 0.2101 | (0.1591, 0.2573) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0932 | (0.0570, 0.1280) | 0.0000 | 0.0932 | (0.0420, 0.1316) | 0.0003 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2564 | (0.1654, 0.3404) | 0.0000 | 0.2564 | (0.2003, 0.3295) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0501 | (0.0324, 0.0681) | 0.0000 | 0.0501 | (0.0324, 0.0714) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2210 | (0.1454, 0.2994) | 0.0000 | 0.2210 | (0.1760, 0.2725) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.1662 | (0.0686, 0.2674) | 0.0000 | 0.1662 | (0.0000, 0.3565) | 0.0253 |
| controlled_vs_proposed_raw | distinct1 | -0.0157 | (-0.0340, 0.0024) | 0.9537 | -0.0157 | (-0.0354, -0.0008) | 0.9807 |
| controlled_vs_proposed_raw | length_score | 0.4042 | (0.2583, 0.5417) | 0.0000 | 0.4042 | (0.1873, 0.5726) | 0.0007 |
| controlled_vs_proposed_raw | sentence_score | 0.1896 | (0.1021, 0.2771) | 0.0000 | 0.1896 | (0.0875, 0.2593) | 0.0007 |
| controlled_vs_proposed_raw | overall_quality | 0.1855 | (0.1424, 0.2302) | 0.0000 | 0.1855 | (0.1503, 0.2225) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2243 | (0.1875, 0.2655) | 0.0000 | 0.2243 | (0.1906, 0.2686) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1488 | (0.0649, 0.2261) | 0.0007 | 0.1488 | (0.0522, 0.2191) | 0.0030 |
| controlled_vs_candidate_no_context | naturalness | 0.0900 | (0.0507, 0.1286) | 0.0000 | 0.0900 | (0.0323, 0.1301) | 0.0010 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.2943 | (0.2463, 0.3474) | 0.0000 | 0.2943 | (0.2500, 0.3546) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0608 | (0.0459, 0.0755) | 0.0000 | 0.0608 | (0.0490, 0.0761) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1490 | (0.0532, 0.2391) | 0.0003 | 0.1490 | (0.0397, 0.2323) | 0.0067 |
| controlled_vs_candidate_no_context | persona_style | 0.1479 | (0.0502, 0.2565) | 0.0007 | 0.1479 | (0.0038, 0.3086) | 0.0240 |
| controlled_vs_candidate_no_context | distinct1 | -0.0105 | (-0.0275, 0.0054) | 0.8927 | -0.0105 | (-0.0290, 0.0046) | 0.9097 |
| controlled_vs_candidate_no_context | length_score | 0.3778 | (0.2153, 0.5292) | 0.0000 | 0.3778 | (0.1403, 0.5507) | 0.0003 |
| controlled_vs_candidate_no_context | sentence_score | 0.1896 | (0.0875, 0.2771) | 0.0003 | 0.1896 | (0.1021, 0.2667) | 0.0000 |
| controlled_vs_candidate_no_context | overall_quality | 0.1758 | (0.1359, 0.2138) | 0.0000 | 0.1758 | (0.1333, 0.2117) | 0.0000 |
| controlled_vs_baseline_no_context | context_relevance | 0.2166 | (0.1852, 0.2511) | 0.0000 | 0.2166 | (0.1836, 0.2625) | 0.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1244 | (0.0548, 0.1938) | 0.0003 | 0.1244 | (0.0593, 0.2045) | 0.0003 |
| controlled_vs_baseline_no_context | naturalness | 0.0040 | (-0.0293, 0.0387) | 0.3983 | 0.0040 | (-0.0377, 0.0495) | 0.4430 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.2836 | (0.2376, 0.3289) | 0.0000 | 0.2836 | (0.2394, 0.3439) | 0.0000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0602 | (0.0484, 0.0715) | 0.0000 | 0.0602 | (0.0489, 0.0724) | 0.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1349 | (0.0500, 0.2216) | 0.0013 | 0.1349 | (0.0586, 0.2236) | 0.0000 |
| controlled_vs_baseline_no_context | persona_style | 0.0823 | (0.0237, 0.1524) | 0.0003 | 0.0823 | (0.0000, 0.2021) | 0.0253 |
| controlled_vs_baseline_no_context | distinct1 | -0.0428 | (-0.0578, -0.0259) | 1.0000 | -0.0428 | (-0.0609, -0.0222) | 1.0000 |
| controlled_vs_baseline_no_context | length_score | 0.1000 | (-0.0500, 0.2500) | 0.0993 | 0.1000 | (-0.1042, 0.3254) | 0.1827 |
| controlled_vs_baseline_no_context | sentence_score | 0.0146 | (-0.0583, 0.0875) | 0.4207 | 0.0146 | (-0.0603, 0.1065) | 0.4593 |
| controlled_vs_baseline_no_context | overall_quality | 0.1469 | (0.1138, 0.1793) | 0.0000 | 0.1469 | (0.1144, 0.1882) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0207 | (-0.0332, 0.0797) | 0.2580 | 0.0207 | (-0.0228, 0.0674) | 0.2093 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0460 | (-0.0289, 0.1361) | 0.1320 | 0.0460 | (0.0005, 0.0865) | 0.0080 |
| controlled_alt_vs_controlled_default | naturalness | -0.0231 | (-0.0548, 0.0093) | 0.9243 | -0.0231 | (-0.0453, -0.0010) | 0.9823 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0282 | (-0.0414, 0.1063) | 0.2463 | 0.0282 | (-0.0292, 0.0893) | 0.1900 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0032 | (-0.0164, 0.0228) | 0.3733 | 0.0032 | (-0.0143, 0.0184) | 0.3533 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0512 | (-0.0362, 0.1611) | 0.1523 | 0.0512 | (0.0087, 0.0910) | 0.0197 |
| controlled_alt_vs_controlled_default | persona_style | 0.0251 | (-0.0158, 0.0686) | 0.1140 | 0.0251 | (-0.0190, 0.0980) | 0.2630 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0090 | (-0.0109, 0.0273) | 0.1827 | 0.0090 | (-0.0112, 0.0263) | 0.1883 |
| controlled_alt_vs_controlled_default | length_score | -0.1458 | (-0.2958, -0.0055) | 0.9803 | -0.1458 | (-0.2667, -0.0167) | 0.9857 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0437 | (-0.0146, 0.1021) | 0.1140 | 0.0437 | (0.0140, 0.0808) | 0.0213 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0219 | (-0.0087, 0.0591) | 0.0933 | 0.0219 | (0.0064, 0.0322) | 0.0027 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.2152 | (0.1455, 0.2839) | 0.0000 | 0.2152 | (0.1670, 0.2621) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.2560 | (0.1714, 0.3504) | 0.0000 | 0.2560 | (0.1979, 0.3120) | 0.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0701 | (0.0280, 0.1120) | 0.0000 | 0.0701 | (0.0295, 0.0998) | 0.0010 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2846 | (0.1891, 0.3818) | 0.0000 | 0.2846 | (0.2175, 0.3481) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0533 | (0.0396, 0.0678) | 0.0000 | 0.0533 | (0.0466, 0.0609) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.2722 | (0.1716, 0.3812) | 0.0000 | 0.2722 | (0.2173, 0.3286) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 0.1913 | (0.0949, 0.2964) | 0.0000 | 0.1913 | (0.0353, 0.3820) | 0.0233 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0067 | (-0.0330, 0.0203) | 0.6877 | -0.0067 | (-0.0407, 0.0203) | 0.6723 |
| controlled_alt_vs_proposed_raw | length_score | 0.2583 | (0.0916, 0.4194) | 0.0010 | 0.2583 | (0.1035, 0.3700) | 0.0013 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.2333 | (0.1313, 0.3062) | 0.0000 | 0.2333 | (0.1432, 0.2962) | 0.0000 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.2074 | (0.1571, 0.2596) | 0.0000 | 0.2074 | (0.1755, 0.2397) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2449 | (0.1929, 0.3016) | 0.0000 | 0.2449 | (0.1954, 0.2984) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1947 | (0.1072, 0.2799) | 0.0000 | 0.1947 | (0.1293, 0.2508) | 0.0000 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0669 | (0.0199, 0.1142) | 0.0020 | 0.0669 | (0.0167, 0.1066) | 0.0057 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3225 | (0.2513, 0.3969) | 0.0000 | 0.3225 | (0.2561, 0.3965) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0640 | (0.0518, 0.0757) | 0.0000 | 0.0640 | (0.0564, 0.0729) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.2002 | (0.0960, 0.3087) | 0.0000 | 0.2002 | (0.1083, 0.2798) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.1729 | (0.0753, 0.2826) | 0.0000 | 0.1729 | (0.0258, 0.3286) | 0.0017 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0015 | (-0.0272, 0.0222) | 0.5203 | -0.0015 | (-0.0291, 0.0228) | 0.5623 |
| controlled_alt_vs_candidate_no_context | length_score | 0.2319 | (0.0444, 0.4083) | 0.0077 | 0.2319 | (0.0444, 0.3667) | 0.0077 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.2333 | (0.1604, 0.2917) | 0.0000 | 0.2333 | (0.1333, 0.3096) | 0.0000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1978 | (0.1547, 0.2422) | 0.0000 | 0.1978 | (0.1549, 0.2271) | 0.0000 |
| controlled_alt_vs_baseline_no_context | context_relevance | 0.2372 | (0.1865, 0.2947) | 0.0000 | 0.2372 | (0.1959, 0.2841) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 0.1704 | (0.1050, 0.2384) | 0.0000 | 0.1704 | (0.1267, 0.2260) | 0.0000 |
| controlled_alt_vs_baseline_no_context | naturalness | -0.0191 | (-0.0544, 0.0172) | 0.8500 | -0.0191 | (-0.0472, 0.0193) | 0.8550 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 0.3117 | (0.2415, 0.3920) | 0.0000 | 0.3117 | (0.2548, 0.3723) | 0.0000 |
| controlled_alt_vs_baseline_no_context | context_overlap | 0.0634 | (0.0494, 0.0765) | 0.0000 | 0.0634 | (0.0516, 0.0737) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 0.1861 | (0.1049, 0.2728) | 0.0000 | 0.1861 | (0.1266, 0.2519) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_style | 0.1074 | (0.0471, 0.1756) | 0.0000 | 0.1074 | (0.0199, 0.2276) | 0.0227 |
| controlled_alt_vs_baseline_no_context | distinct1 | -0.0338 | (-0.0545, -0.0145) | 1.0000 | -0.0338 | (-0.0572, -0.0144) | 0.9990 |
| controlled_alt_vs_baseline_no_context | length_score | -0.0458 | (-0.2167, 0.1153) | 0.7290 | -0.0458 | (-0.1827, 0.1386) | 0.7143 |
| controlled_alt_vs_baseline_no_context | sentence_score | 0.0583 | (0.0000, 0.1313) | 0.0633 | 0.0583 | (0.0117, 0.1289) | 0.0190 |
| controlled_alt_vs_baseline_no_context | overall_quality | 0.1688 | (0.1323, 0.2044) | 0.0000 | 0.1688 | (0.1433, 0.1974) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2166 | (0.1851, 0.2499) | 0.0000 | 0.2166 | (0.1829, 0.2625) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1244 | (0.0534, 0.1937) | 0.0003 | 0.1244 | (0.0582, 0.2023) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 0.0040 | (-0.0306, 0.0375) | 0.4057 | 0.0040 | (-0.0382, 0.0507) | 0.4403 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.2836 | (0.2393, 0.3303) | 0.0000 | 0.2836 | (0.2388, 0.3455) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0602 | (0.0483, 0.0715) | 0.0000 | 0.0602 | (0.0489, 0.0728) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1349 | (0.0490, 0.2169) | 0.0010 | 0.1349 | (0.0547, 0.2275) | 0.0010 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0823 | (0.0265, 0.1499) | 0.0010 | 0.0823 | (0.0058, 0.2022) | 0.0210 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0428 | (-0.0580, -0.0266) | 1.0000 | -0.0428 | (-0.0617, -0.0224) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.1000 | (-0.0500, 0.2473) | 0.0937 | 0.1000 | (-0.1000, 0.3317) | 0.1753 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0146 | (-0.0583, 0.0875) | 0.4353 | 0.0146 | (-0.0603, 0.1167) | 0.4493 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1469 | (0.1126, 0.1794) | 0.0000 | 0.1469 | (0.1146, 0.1881) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 9 | 4 | 11 | 0.6042 | 0.6923 |
| proposed_vs_candidate_no_context | persona_consistency | 3 | 5 | 16 | 0.4583 | 0.3750 |
| proposed_vs_candidate_no_context | naturalness | 7 | 5 | 12 | 0.5417 | 0.5833 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 4 | 2 | 18 | 0.5417 | 0.6667 |
| proposed_vs_candidate_no_context | context_overlap | 10 | 3 | 11 | 0.6458 | 0.7692 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 2 | 5 | 17 | 0.4375 | 0.2857 |
| proposed_vs_candidate_no_context | persona_style | 2 | 2 | 20 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | distinct1 | 7 | 5 | 12 | 0.5417 | 0.5833 |
| proposed_vs_candidate_no_context | length_score | 5 | 7 | 12 | 0.4583 | 0.4167 |
| proposed_vs_candidate_no_context | sentence_score | 4 | 4 | 16 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | overall_quality | 7 | 6 | 11 | 0.5208 | 0.5385 |
| proposed_vs_baseline_no_context | context_relevance | 10 | 14 | 0 | 0.4167 | 0.4167 |
| proposed_vs_baseline_no_context | persona_consistency | 4 | 12 | 8 | 0.3333 | 0.2500 |
| proposed_vs_baseline_no_context | naturalness | 5 | 19 | 0 | 0.2083 | 0.2083 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 5 | 5 | 14 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context | context_overlap | 12 | 12 | 0 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 2 | 9 | 13 | 0.3542 | 0.1818 |
| proposed_vs_baseline_no_context | persona_style | 3 | 5 | 16 | 0.4583 | 0.3750 |
| proposed_vs_baseline_no_context | distinct1 | 7 | 16 | 1 | 0.3125 | 0.3043 |
| proposed_vs_baseline_no_context | length_score | 4 | 20 | 0 | 0.1667 | 0.1667 |
| proposed_vs_baseline_no_context | sentence_score | 4 | 16 | 4 | 0.2500 | 0.2000 |
| proposed_vs_baseline_no_context | overall_quality | 6 | 18 | 0 | 0.2500 | 0.2500 |
| controlled_vs_proposed_raw | context_relevance | 22 | 2 | 0 | 0.9167 | 0.9167 |
| controlled_vs_proposed_raw | persona_consistency | 19 | 1 | 4 | 0.8750 | 0.9500 |
| controlled_vs_proposed_raw | naturalness | 19 | 5 | 0 | 0.7917 | 0.7917 |
| controlled_vs_proposed_raw | context_keyword_coverage | 22 | 2 | 0 | 0.9167 | 0.9167 |
| controlled_vs_proposed_raw | context_overlap | 20 | 4 | 0 | 0.8333 | 0.8333 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 19 | 1 | 4 | 0.8750 | 0.9500 |
| controlled_vs_proposed_raw | persona_style | 8 | 0 | 16 | 0.6667 | 1.0000 |
| controlled_vs_proposed_raw | distinct1 | 9 | 13 | 2 | 0.4167 | 0.4091 |
| controlled_vs_proposed_raw | length_score | 20 | 4 | 0 | 0.8333 | 0.8333 |
| controlled_vs_proposed_raw | sentence_score | 14 | 1 | 9 | 0.7708 | 0.9333 |
| controlled_vs_proposed_raw | overall_quality | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_vs_candidate_no_context | context_relevance | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 18 | 4 | 2 | 0.7917 | 0.8182 |
| controlled_vs_candidate_no_context | naturalness | 18 | 6 | 0 | 0.7500 | 0.7500 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 22 | 2 | 0 | 0.9167 | 0.9167 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 18 | 4 | 2 | 0.7917 | 0.8182 |
| controlled_vs_candidate_no_context | persona_style | 8 | 2 | 14 | 0.6250 | 0.8000 |
| controlled_vs_candidate_no_context | distinct1 | 9 | 13 | 2 | 0.4167 | 0.4091 |
| controlled_vs_candidate_no_context | length_score | 19 | 5 | 0 | 0.7917 | 0.7917 |
| controlled_vs_candidate_no_context | sentence_score | 15 | 2 | 7 | 0.7708 | 0.8824 |
| controlled_vs_candidate_no_context | overall_quality | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_vs_baseline_no_context | context_relevance | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 16 | 4 | 4 | 0.7500 | 0.8000 |
| controlled_vs_baseline_no_context | naturalness | 12 | 12 | 0 | 0.5000 | 0.5000 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | context_overlap | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 16 | 4 | 4 | 0.7500 | 0.8000 |
| controlled_vs_baseline_no_context | persona_style | 6 | 1 | 17 | 0.6042 | 0.8571 |
| controlled_vs_baseline_no_context | distinct1 | 4 | 20 | 0 | 0.1667 | 0.1667 |
| controlled_vs_baseline_no_context | length_score | 14 | 9 | 1 | 0.6042 | 0.6087 |
| controlled_vs_baseline_no_context | sentence_score | 4 | 3 | 17 | 0.5208 | 0.5714 |
| controlled_vs_baseline_no_context | overall_quality | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_alt_vs_controlled_default | context_relevance | 9 | 12 | 3 | 0.4375 | 0.4286 |
| controlled_alt_vs_controlled_default | persona_consistency | 8 | 7 | 9 | 0.5208 | 0.5333 |
| controlled_alt_vs_controlled_default | naturalness | 8 | 13 | 3 | 0.3958 | 0.3810 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 8 | 9 | 7 | 0.4792 | 0.4706 |
| controlled_alt_vs_controlled_default | context_overlap | 10 | 11 | 3 | 0.4792 | 0.4762 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 5 | 4 | 15 | 0.5208 | 0.5556 |
| controlled_alt_vs_controlled_default | persona_style | 5 | 3 | 16 | 0.5417 | 0.6250 |
| controlled_alt_vs_controlled_default | distinct1 | 13 | 8 | 3 | 0.6042 | 0.6190 |
| controlled_alt_vs_controlled_default | length_score | 8 | 12 | 4 | 0.4167 | 0.4000 |
| controlled_alt_vs_controlled_default | sentence_score | 4 | 1 | 19 | 0.5625 | 0.8000 |
| controlled_alt_vs_controlled_default | overall_quality | 12 | 9 | 3 | 0.5625 | 0.5714 |
| controlled_alt_vs_proposed_raw | context_relevance | 21 | 3 | 0 | 0.8750 | 0.8750 |
| controlled_alt_vs_proposed_raw | persona_consistency | 20 | 2 | 2 | 0.8750 | 0.9091 |
| controlled_alt_vs_proposed_raw | naturalness | 18 | 6 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 21 | 2 | 1 | 0.8958 | 0.9130 |
| controlled_alt_vs_proposed_raw | context_overlap | 21 | 3 | 0 | 0.8750 | 0.8750 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 20 | 2 | 2 | 0.8750 | 0.9091 |
| controlled_alt_vs_proposed_raw | persona_style | 10 | 1 | 13 | 0.6875 | 0.9091 |
| controlled_alt_vs_proposed_raw | distinct1 | 11 | 12 | 1 | 0.4792 | 0.4783 |
| controlled_alt_vs_proposed_raw | length_score | 16 | 8 | 0 | 0.6667 | 0.6667 |
| controlled_alt_vs_proposed_raw | sentence_score | 18 | 2 | 4 | 0.8333 | 0.9000 |
| controlled_alt_vs_proposed_raw | overall_quality | 22 | 2 | 0 | 0.9167 | 0.9167 |
| controlled_alt_vs_candidate_no_context | context_relevance | 23 | 0 | 1 | 0.9792 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 19 | 3 | 2 | 0.8333 | 0.8636 |
| controlled_alt_vs_candidate_no_context | naturalness | 16 | 8 | 0 | 0.6667 | 0.6667 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 23 | 0 | 1 | 0.9792 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 23 | 0 | 1 | 0.9792 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 19 | 3 | 2 | 0.8333 | 0.8636 |
| controlled_alt_vs_candidate_no_context | persona_style | 9 | 1 | 14 | 0.6667 | 0.9000 |
| controlled_alt_vs_candidate_no_context | distinct1 | 14 | 9 | 1 | 0.6042 | 0.6087 |
| controlled_alt_vs_candidate_no_context | length_score | 15 | 9 | 0 | 0.6250 | 0.6250 |
| controlled_alt_vs_candidate_no_context | sentence_score | 16 | 0 | 8 | 0.8333 | 1.0000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 22 | 2 | 0 | 0.9167 | 0.9167 |
| controlled_alt_vs_baseline_no_context | context_relevance | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 19 | 2 | 3 | 0.8542 | 0.9048 |
| controlled_alt_vs_baseline_no_context | naturalness | 11 | 13 | 0 | 0.4583 | 0.4583 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 23 | 0 | 1 | 0.9792 | 1.0000 |
| controlled_alt_vs_baseline_no_context | context_overlap | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 19 | 2 | 3 | 0.8542 | 0.9048 |
| controlled_alt_vs_baseline_no_context | persona_style | 8 | 1 | 15 | 0.6458 | 0.8889 |
| controlled_alt_vs_baseline_no_context | distinct1 | 5 | 19 | 0 | 0.2083 | 0.2083 |
| controlled_alt_vs_baseline_no_context | length_score | 10 | 14 | 0 | 0.4167 | 0.4167 |
| controlled_alt_vs_baseline_no_context | sentence_score | 5 | 1 | 18 | 0.5833 | 0.8333 |
| controlled_alt_vs_baseline_no_context | overall_quality | 24 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 24 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 16 | 4 | 4 | 0.7500 | 0.8000 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 12 | 12 | 0 | 0.5000 | 0.5000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 24 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 23 | 1 | 0 | 0.9583 | 0.9583 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 16 | 4 | 4 | 0.7500 | 0.8000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 6 | 1 | 17 | 0.6042 | 0.8571 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | 4 | 20 | 0 | 0.1667 | 0.1667 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 14 | 9 | 1 | 0.6042 | 0.6087 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 4 | 3 | 17 | 0.5208 | 0.5714 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 23 | 1 | 0 | 0.9583 | 0.9583 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.3333 | 0.4167 | 0.5833 |
| proposed_contextual_controlled_alt | 0.0000 | 0.0000 | 0.2083 | 0.4583 | 0.5417 |
| proposed_contextual | 0.0000 | 0.0000 | 0.5417 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.5833 | 0.0000 | 0.0000 |
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
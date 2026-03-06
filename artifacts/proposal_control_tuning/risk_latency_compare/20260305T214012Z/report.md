# Proposal Alignment Evaluation Report

- Run ID: `20260305T214012Z`
- Generated: `2026-03-05T21:46:31.383667+00:00`
- Scenarios: `artifacts\proposal_control_tuning\risk_latency_compare\20260305T214012Z\scenarios.jsonl`
- Scenario count: `32`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_rla`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2719 (0.2362, 0.3105) | 0.3672 (0.3157, 0.4261) | 0.8923 (0.8692, 0.9136) | 0.4328 (0.4103, 0.4566) | n/a |
| proposed_contextual_controlled_rla | 0.2650 (0.2393, 0.2929) | 0.3245 (0.2776, 0.3754) | 0.8788 (0.8531, 0.9030) | 0.4108 (0.3917, 0.4288) | n/a |
| proposed_contextual | 0.1174 (0.0765, 0.1615) | 0.1379 (0.1051, 0.1754) | 0.8455 (0.8130, 0.8781) | 0.2667 (0.2372, 0.2974) | n/a |
| candidate_no_context | 0.0285 (0.0158, 0.0462) | 0.1892 (0.1422, 0.2388) | 0.8326 (0.8045, 0.8590) | 0.2416 (0.2189, 0.2666) | n/a |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0888 | 3.1140 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0513 | -0.2710 |
| proposed_vs_candidate_no_context | naturalness | 0.0129 | 0.0156 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.1172 | 4.6698 |
| proposed_vs_candidate_no_context | context_overlap | 0.0227 | 0.6213 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0580 | -0.5735 |
| proposed_vs_candidate_no_context | persona_style | -0.0242 | -0.0447 |
| proposed_vs_candidate_no_context | distinct1 | -0.0016 | -0.0017 |
| proposed_vs_candidate_no_context | length_score | 0.0469 | 0.1226 |
| proposed_vs_candidate_no_context | sentence_score | 0.0437 | 0.0560 |
| proposed_vs_candidate_no_context | overall_quality | 0.0251 | 0.1038 |
| controlled_vs_proposed_raw | context_relevance | 0.1546 | 1.3169 |
| controlled_vs_proposed_raw | persona_consistency | 0.2293 | 1.6621 |
| controlled_vs_proposed_raw | naturalness | 0.0468 | 0.0553 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2006 | 1.4100 |
| controlled_vs_proposed_raw | context_overlap | 0.0471 | 0.7951 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2705 | 6.2690 |
| controlled_vs_proposed_raw | persona_style | 0.0642 | 0.1241 |
| controlled_vs_proposed_raw | distinct1 | -0.0107 | -0.0113 |
| controlled_vs_proposed_raw | length_score | 0.2021 | 0.4709 |
| controlled_vs_proposed_raw | sentence_score | 0.1094 | 0.1326 |
| controlled_vs_proposed_raw | overall_quality | 0.1661 | 0.6226 |
| controlled_vs_candidate_no_context | context_relevance | 0.2434 | 8.5315 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1780 | 0.9408 |
| controlled_vs_candidate_no_context | naturalness | 0.0597 | 0.0718 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3178 | 12.6642 |
| controlled_vs_candidate_no_context | context_overlap | 0.0698 | 1.9104 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2125 | 2.1000 |
| controlled_vs_candidate_no_context | persona_style | 0.0400 | 0.0739 |
| controlled_vs_candidate_no_context | distinct1 | -0.0122 | -0.0129 |
| controlled_vs_candidate_no_context | length_score | 0.2490 | 0.6512 |
| controlled_vs_candidate_no_context | sentence_score | 0.1531 | 0.1960 |
| controlled_vs_candidate_no_context | overall_quality | 0.1912 | 0.7911 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0070 | -0.0256 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0427 | -0.1162 |
| controlled_alt_vs_controlled_default | naturalness | -0.0136 | -0.0152 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0104 | -0.0304 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0011 | 0.0105 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0522 | -0.1665 |
| controlled_alt_vs_controlled_default | persona_style | -0.0044 | -0.0075 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0022 | 0.0024 |
| controlled_alt_vs_controlled_default | length_score | -0.0385 | -0.0611 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0656 | -0.0702 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0220 | -0.0508 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1476 | 1.2576 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1866 | 1.3528 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0332 | 0.0393 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.1902 | 1.3368 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0482 | 0.8140 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.2183 | 5.0586 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0598 | 0.1157 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0084 | -0.0089 |
| controlled_alt_vs_proposed_raw | length_score | 0.1635 | 0.3811 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0438 | 0.0530 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1441 | 0.5402 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2365 | 8.2877 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1353 | 0.7153 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0462 | 0.0555 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3074 | 12.2491 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0709 | 1.9409 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1603 | 1.5838 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0356 | 0.0658 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0100 | -0.0106 |
| controlled_alt_vs_candidate_no_context | length_score | 0.2104 | 0.5504 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.0875 | 0.1120 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1692 | 0.7001 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0888 | (0.0525, 0.1289) | 0.0000 | 0.0888 | (0.0350, 0.1425) | 0.0000 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0513 | (-0.1045, -0.0045) | 0.9860 | -0.0513 | (-0.1257, 0.0046) | 0.9647 |
| proposed_vs_candidate_no_context | naturalness | 0.0129 | (-0.0209, 0.0455) | 0.2257 | 0.0129 | (-0.0390, 0.0493) | 0.2910 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.1172 | (0.0653, 0.1726) | 0.0000 | 0.1172 | (0.0478, 0.1880) | 0.0000 |
| proposed_vs_candidate_no_context | context_overlap | 0.0227 | (0.0142, 0.0323) | 0.0000 | 0.0227 | (0.0108, 0.0334) | 0.0000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0580 | (-0.1190, -0.0030) | 0.9817 | -0.0580 | (-0.1445, 0.0057) | 0.9600 |
| proposed_vs_candidate_no_context | persona_style | -0.0242 | (-0.0771, 0.0083) | 0.8737 | -0.0242 | (-0.0771, 0.0078) | 0.8897 |
| proposed_vs_candidate_no_context | distinct1 | -0.0016 | (-0.0159, 0.0124) | 0.5920 | -0.0016 | (-0.0204, 0.0133) | 0.5467 |
| proposed_vs_candidate_no_context | length_score | 0.0469 | (-0.0792, 0.1594) | 0.2273 | 0.0469 | (-0.1274, 0.1735) | 0.2710 |
| proposed_vs_candidate_no_context | sentence_score | 0.0437 | (-0.0328, 0.1203) | 0.1527 | 0.0437 | (-0.0673, 0.1300) | 0.2560 |
| proposed_vs_candidate_no_context | overall_quality | 0.0251 | (-0.0096, 0.0593) | 0.0817 | 0.0251 | (-0.0287, 0.0630) | 0.1533 |
| controlled_vs_proposed_raw | context_relevance | 0.1546 | (0.1087, 0.1995) | 0.0000 | 0.1546 | (0.1089, 0.2057) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.2293 | (0.1721, 0.2946) | 0.0000 | 0.2293 | (0.1670, 0.2947) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0468 | (0.0084, 0.0837) | 0.0043 | 0.0468 | (0.0031, 0.0899) | 0.0157 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2006 | (0.1418, 0.2605) | 0.0000 | 0.2006 | (0.1403, 0.2701) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0471 | (0.0320, 0.0650) | 0.0000 | 0.0471 | (0.0338, 0.0607) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2705 | (0.1975, 0.3519) | 0.0000 | 0.2705 | (0.1867, 0.3657) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.0642 | (0.0043, 0.1299) | 0.0167 | 0.0642 | (-0.0005, 0.1705) | 0.0307 |
| controlled_vs_proposed_raw | distinct1 | -0.0107 | (-0.0277, 0.0058) | 0.8903 | -0.0107 | (-0.0267, 0.0099) | 0.8537 |
| controlled_vs_proposed_raw | length_score | 0.2021 | (0.0687, 0.3385) | 0.0013 | 0.2021 | (0.0495, 0.3511) | 0.0047 |
| controlled_vs_proposed_raw | sentence_score | 0.1094 | (0.0219, 0.1859) | 0.0060 | 0.1094 | (0.0212, 0.2078) | 0.0060 |
| controlled_vs_proposed_raw | overall_quality | 0.1661 | (0.1252, 0.2074) | 0.0000 | 0.1661 | (0.1174, 0.2197) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2434 | (0.2117, 0.2763) | 0.0000 | 0.2434 | (0.2188, 0.2727) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1780 | (0.1123, 0.2414) | 0.0000 | 0.1780 | (0.1007, 0.2589) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.0597 | (0.0274, 0.0943) | 0.0003 | 0.0597 | (0.0202, 0.0935) | 0.0017 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3178 | (0.2743, 0.3594) | 0.0000 | 0.3178 | (0.2824, 0.3601) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0698 | (0.0571, 0.0839) | 0.0000 | 0.0698 | (0.0629, 0.0746) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2125 | (0.1321, 0.2920) | 0.0000 | 0.2125 | (0.1077, 0.3229) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.0400 | (-0.0151, 0.1025) | 0.0843 | 0.0400 | (-0.0055, 0.1032) | 0.0483 |
| controlled_vs_candidate_no_context | distinct1 | -0.0122 | (-0.0253, 0.0012) | 0.9620 | -0.0122 | (-0.0281, 0.0021) | 0.9407 |
| controlled_vs_candidate_no_context | length_score | 0.2490 | (0.1208, 0.3792) | 0.0000 | 0.2490 | (0.1034, 0.3697) | 0.0010 |
| controlled_vs_candidate_no_context | sentence_score | 0.1531 | (0.0875, 0.2188) | 0.0000 | 0.1531 | (0.0648, 0.2217) | 0.0010 |
| controlled_vs_candidate_no_context | overall_quality | 0.1912 | (0.1616, 0.2208) | 0.0000 | 0.1912 | (0.1578, 0.2274) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0070 | (-0.0448, 0.0302) | 0.6497 | -0.0070 | (-0.0455, 0.0240) | 0.6633 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0427 | (-0.0997, 0.0147) | 0.9323 | -0.0427 | (-0.1268, 0.0167) | 0.8420 |
| controlled_alt_vs_controlled_default | naturalness | -0.0136 | (-0.0425, 0.0185) | 0.8047 | -0.0136 | (-0.0404, 0.0203) | 0.7950 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0104 | (-0.0599, 0.0398) | 0.6477 | -0.0104 | (-0.0636, 0.0337) | 0.6820 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0011 | (-0.0123, 0.0149) | 0.4397 | 0.0011 | (-0.0087, 0.0129) | 0.4230 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0522 | (-0.1258, 0.0140) | 0.9323 | -0.0522 | (-0.1640, 0.0215) | 0.8400 |
| controlled_alt_vs_controlled_default | persona_style | -0.0044 | (-0.0355, 0.0237) | 0.5807 | -0.0044 | (-0.0338, 0.0181) | 0.6343 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0022 | (-0.0143, 0.0181) | 0.3870 | 0.0022 | (-0.0152, 0.0153) | 0.3727 |
| controlled_alt_vs_controlled_default | length_score | -0.0385 | (-0.1552, 0.0865) | 0.7620 | -0.0385 | (-0.1460, 0.0889) | 0.7420 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0656 | (-0.1203, -0.0109) | 0.9940 | -0.0656 | (-0.1235, 0.0000) | 0.9793 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0220 | (-0.0496, 0.0046) | 0.9520 | -0.0220 | (-0.0576, 0.0057) | 0.9267 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1476 | (0.1126, 0.1805) | 0.0000 | 0.1476 | (0.1124, 0.1869) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1866 | (0.1433, 0.2326) | 0.0000 | 0.1866 | (0.1462, 0.2298) | 0.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0332 | (-0.0071, 0.0751) | 0.0573 | 0.0332 | (-0.0186, 0.0932) | 0.1253 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.1902 | (0.1445, 0.2344) | 0.0000 | 0.1902 | (0.1440, 0.2424) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0482 | (0.0339, 0.0625) | 0.0000 | 0.0482 | (0.0340, 0.0645) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.2183 | (0.1664, 0.2783) | 0.0000 | 0.2183 | (0.1699, 0.2709) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0598 | (0.0085, 0.1235) | 0.0023 | 0.0598 | (-0.0038, 0.1652) | 0.0927 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0084 | (-0.0260, 0.0086) | 0.8277 | -0.0084 | (-0.0226, 0.0089) | 0.8373 |
| controlled_alt_vs_proposed_raw | length_score | 0.1635 | (0.0156, 0.3136) | 0.0150 | 0.1635 | (-0.0285, 0.3778) | 0.0527 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0437 | (-0.0547, 0.1422) | 0.2293 | 0.0437 | (-0.1025, 0.2074) | 0.3190 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1441 | (0.1107, 0.1753) | 0.0000 | 0.1441 | (0.1169, 0.1745) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2365 | (0.2084, 0.2655) | 0.0000 | 0.2365 | (0.2035, 0.2659) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1353 | (0.0795, 0.1912) | 0.0000 | 0.1353 | (0.0692, 0.1982) | 0.0000 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0462 | (0.0065, 0.0865) | 0.0090 | 0.0462 | (-0.0023, 0.0967) | 0.0300 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3074 | (0.2698, 0.3459) | 0.0000 | 0.3074 | (0.2650, 0.3477) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0709 | (0.0580, 0.0843) | 0.0000 | 0.0709 | (0.0574, 0.0848) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1603 | (0.0951, 0.2287) | 0.0000 | 0.1603 | (0.0789, 0.2382) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0356 | (-0.0068, 0.0916) | 0.0613 | 0.0356 | (-0.0131, 0.1087) | 0.1003 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0100 | (-0.0270, 0.0061) | 0.8763 | -0.0100 | (-0.0273, 0.0065) | 0.8720 |
| controlled_alt_vs_candidate_no_context | length_score | 0.2104 | (0.0552, 0.3573) | 0.0040 | 0.2104 | (0.0333, 0.3764) | 0.0083 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.0875 | (0.0109, 0.1641) | 0.0213 | 0.0875 | (0.0000, 0.2000) | 0.0483 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1692 | (0.1389, 0.1984) | 0.0000 | 0.1692 | (0.1346, 0.1982) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 20 | 1 | 11 | 0.7969 | 0.9524 |
| proposed_vs_candidate_no_context | persona_consistency | 5 | 9 | 18 | 0.4375 | 0.3571 |
| proposed_vs_candidate_no_context | naturalness | 14 | 7 | 11 | 0.6094 | 0.6667 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 15 | 0 | 17 | 0.7344 | 1.0000 |
| proposed_vs_candidate_no_context | context_overlap | 19 | 2 | 11 | 0.7656 | 0.9048 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 4 | 8 | 20 | 0.4375 | 0.3333 |
| proposed_vs_candidate_no_context | persona_style | 1 | 2 | 29 | 0.4844 | 0.3333 |
| proposed_vs_candidate_no_context | distinct1 | 11 | 8 | 13 | 0.5469 | 0.5789 |
| proposed_vs_candidate_no_context | length_score | 13 | 7 | 12 | 0.5938 | 0.6500 |
| proposed_vs_candidate_no_context | sentence_score | 8 | 4 | 20 | 0.5625 | 0.6667 |
| proposed_vs_candidate_no_context | overall_quality | 13 | 8 | 11 | 0.5781 | 0.6190 |
| controlled_vs_proposed_raw | context_relevance | 26 | 6 | 0 | 0.8125 | 0.8125 |
| controlled_vs_proposed_raw | persona_consistency | 28 | 0 | 4 | 0.9375 | 1.0000 |
| controlled_vs_proposed_raw | naturalness | 19 | 13 | 0 | 0.5938 | 0.5938 |
| controlled_vs_proposed_raw | context_keyword_coverage | 24 | 3 | 5 | 0.8281 | 0.8889 |
| controlled_vs_proposed_raw | context_overlap | 27 | 5 | 0 | 0.8438 | 0.8438 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 28 | 0 | 4 | 0.9375 | 1.0000 |
| controlled_vs_proposed_raw | persona_style | 6 | 2 | 24 | 0.5625 | 0.7500 |
| controlled_vs_proposed_raw | distinct1 | 10 | 18 | 4 | 0.3750 | 0.3571 |
| controlled_vs_proposed_raw | length_score | 19 | 13 | 0 | 0.5938 | 0.5938 |
| controlled_vs_proposed_raw | sentence_score | 14 | 4 | 14 | 0.6562 | 0.7778 |
| controlled_vs_proposed_raw | overall_quality | 30 | 2 | 0 | 0.9375 | 0.9375 |
| controlled_vs_candidate_no_context | context_relevance | 32 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 24 | 6 | 2 | 0.7812 | 0.8000 |
| controlled_vs_candidate_no_context | naturalness | 22 | 10 | 0 | 0.6875 | 0.6875 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 31 | 0 | 1 | 0.9844 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 32 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 24 | 4 | 4 | 0.8125 | 0.8571 |
| controlled_vs_candidate_no_context | persona_style | 5 | 3 | 24 | 0.5312 | 0.6250 |
| controlled_vs_candidate_no_context | distinct1 | 11 | 19 | 2 | 0.3750 | 0.3667 |
| controlled_vs_candidate_no_context | length_score | 21 | 11 | 0 | 0.6562 | 0.6562 |
| controlled_vs_candidate_no_context | sentence_score | 15 | 1 | 16 | 0.7188 | 0.9375 |
| controlled_vs_candidate_no_context | overall_quality | 31 | 1 | 0 | 0.9688 | 0.9688 |
| controlled_alt_vs_controlled_default | context_relevance | 13 | 15 | 4 | 0.4688 | 0.4643 |
| controlled_alt_vs_controlled_default | persona_consistency | 8 | 9 | 15 | 0.4844 | 0.4706 |
| controlled_alt_vs_controlled_default | naturalness | 9 | 19 | 4 | 0.3438 | 0.3214 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 10 | 13 | 9 | 0.4531 | 0.4348 |
| controlled_alt_vs_controlled_default | context_overlap | 12 | 16 | 4 | 0.4375 | 0.4286 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 6 | 8 | 18 | 0.4688 | 0.4286 |
| controlled_alt_vs_controlled_default | persona_style | 3 | 3 | 26 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | distinct1 | 13 | 14 | 5 | 0.4844 | 0.4815 |
| controlled_alt_vs_controlled_default | length_score | 9 | 18 | 5 | 0.3594 | 0.3333 |
| controlled_alt_vs_controlled_default | sentence_score | 1 | 7 | 24 | 0.4062 | 0.1250 |
| controlled_alt_vs_controlled_default | overall_quality | 11 | 17 | 4 | 0.4062 | 0.3929 |
| controlled_alt_vs_proposed_raw | context_relevance | 29 | 3 | 0 | 0.9062 | 0.9062 |
| controlled_alt_vs_proposed_raw | persona_consistency | 27 | 0 | 5 | 0.9219 | 1.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 18 | 14 | 0 | 0.5625 | 0.5625 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 28 | 3 | 1 | 0.8906 | 0.9032 |
| controlled_alt_vs_proposed_raw | context_overlap | 27 | 5 | 0 | 0.8438 | 0.8438 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 27 | 0 | 5 | 0.9219 | 1.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 5 | 2 | 25 | 0.5469 | 0.7143 |
| controlled_alt_vs_proposed_raw | distinct1 | 13 | 17 | 2 | 0.4375 | 0.4333 |
| controlled_alt_vs_proposed_raw | length_score | 18 | 13 | 1 | 0.5781 | 0.5806 |
| controlled_alt_vs_proposed_raw | sentence_score | 13 | 9 | 10 | 0.5625 | 0.5909 |
| controlled_alt_vs_proposed_raw | overall_quality | 30 | 2 | 0 | 0.9375 | 0.9375 |
| controlled_alt_vs_candidate_no_context | context_relevance | 32 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 22 | 5 | 5 | 0.7656 | 0.8148 |
| controlled_alt_vs_candidate_no_context | naturalness | 21 | 11 | 0 | 0.6562 | 0.6562 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 32 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 32 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 22 | 4 | 6 | 0.7812 | 0.8462 |
| controlled_alt_vs_candidate_no_context | persona_style | 5 | 2 | 25 | 0.5469 | 0.7143 |
| controlled_alt_vs_candidate_no_context | distinct1 | 14 | 16 | 2 | 0.4688 | 0.4667 |
| controlled_alt_vs_candidate_no_context | length_score | 20 | 11 | 1 | 0.6406 | 0.6452 |
| controlled_alt_vs_candidate_no_context | sentence_score | 12 | 4 | 16 | 0.6250 | 0.7500 |
| controlled_alt_vs_candidate_no_context | overall_quality | 31 | 1 | 0 | 0.9688 | 0.9688 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.2500 | 0.5312 | 0.4688 |
| proposed_contextual_controlled_rla | 0.0000 | 0.0000 | 0.2812 | 0.4688 | 0.5312 |
| proposed_contextual | 0.0000 | 0.0000 | 0.4375 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.4062 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `8`
- Unique template signatures: `30`
- Template signature ratio: `0.9375`
- Effective sample size by source clustering: `7.42`
- Effective sample size by template-signature clustering: `28.44`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: unavailable (disabled_by_flag).

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
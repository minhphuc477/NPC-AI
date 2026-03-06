# Proposal Alignment Evaluation Report

- Run ID: `20260306T164947Z`
- Generated: `2026-03-06T16:54:01.156629+00:00`
- Scenarios: `artifacts\proposal_control_tuning\architecture_compare_lock\20260306T164947Z\blend_balanced\seed_29\attempt_1\20260306T164947Z\scenarios.jsonl`
- Scenario count: `32`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_alt`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2493 (0.2379, 0.2616) | 0.2632 (0.2205, 0.3092) | 0.9145 (0.8986, 0.9296) | 0.3882 (0.3700, 0.4060) | n/a |
| proposed_contextual_controlled_alt | 0.2678 (0.2397, 0.3013) | 0.2555 (0.2241, 0.2877) | 0.9043 (0.8814, 0.9251) | 0.3918 (0.3734, 0.4093) | n/a |
| proposed_contextual | 0.0504 (0.0300, 0.0745) | 0.1313 (0.0886, 0.1781) | 0.8074 (0.7826, 0.8357) | 0.2264 (0.2015, 0.2534) | n/a |
| candidate_no_context | 0.0348 (0.0216, 0.0504) | 0.1767 (0.1209, 0.2381) | 0.8140 (0.7873, 0.8408) | 0.2370 (0.2074, 0.2684) | n/a |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0156 | 0.4501 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0454 | -0.2570 |
| proposed_vs_candidate_no_context | naturalness | -0.0067 | -0.0082 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0192 | 0.6081 |
| proposed_vs_candidate_no_context | context_overlap | 0.0074 | 0.1752 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0469 | -0.4565 |
| proposed_vs_candidate_no_context | persona_style | -0.0395 | -0.0836 |
| proposed_vs_candidate_no_context | distinct1 | -0.0020 | -0.0022 |
| proposed_vs_candidate_no_context | length_score | -0.0292 | -0.0933 |
| proposed_vs_candidate_no_context | sentence_score | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | overall_quality | -0.0106 | -0.0447 |
| controlled_vs_proposed_raw | context_relevance | 0.1989 | 3.9461 |
| controlled_vs_proposed_raw | persona_consistency | 0.1320 | 1.0053 |
| controlled_vs_proposed_raw | naturalness | 0.1071 | 0.1327 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2614 | 5.1541 |
| controlled_vs_proposed_raw | context_overlap | 0.0532 | 1.0701 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1376 | 2.4667 |
| controlled_vs_proposed_raw | persona_style | 0.1093 | 0.2523 |
| controlled_vs_proposed_raw | distinct1 | -0.0073 | -0.0078 |
| controlled_vs_proposed_raw | length_score | 0.4417 | 1.5588 |
| controlled_vs_proposed_raw | sentence_score | 0.2188 | 0.2881 |
| controlled_vs_proposed_raw | overall_quality | 0.1618 | 0.7146 |
| controlled_vs_candidate_no_context | context_relevance | 0.2146 | 6.1723 |
| controlled_vs_candidate_no_context | persona_consistency | 0.0866 | 0.4900 |
| controlled_vs_candidate_no_context | naturalness | 0.1005 | 0.1234 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.2805 | 8.8964 |
| controlled_vs_candidate_no_context | context_overlap | 0.0606 | 1.4328 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.0908 | 0.8841 |
| controlled_vs_candidate_no_context | persona_style | 0.0698 | 0.1476 |
| controlled_vs_candidate_no_context | distinct1 | -0.0093 | -0.0100 |
| controlled_vs_candidate_no_context | length_score | 0.4125 | 1.3200 |
| controlled_vs_candidate_no_context | sentence_score | 0.2188 | 0.2881 |
| controlled_vs_candidate_no_context | overall_quality | 0.1512 | 0.6379 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0185 | 0.0743 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0077 | -0.0293 |
| controlled_alt_vs_controlled_default | naturalness | -0.0102 | -0.0112 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0246 | 0.0787 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0044 | 0.0426 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0121 | -0.0623 |
| controlled_alt_vs_controlled_default | persona_style | 0.0096 | 0.0178 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0058 | 0.0063 |
| controlled_alt_vs_controlled_default | length_score | -0.0417 | -0.0575 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0438 | -0.0447 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0036 | 0.0091 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.2174 | 4.3134 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1243 | 0.9466 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0969 | 0.1200 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2859 | 5.6387 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0576 | 1.1582 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1256 | 2.2507 |
| controlled_alt_vs_proposed_raw | persona_style | 0.1189 | 0.2746 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0015 | -0.0016 |
| controlled_alt_vs_proposed_raw | length_score | 0.4000 | 1.4118 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1750 | 0.2305 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1653 | 0.7303 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2331 | 6.7050 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.0789 | 0.4464 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0903 | 0.1109 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3051 | 9.6757 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0650 | 1.5364 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.0787 | 0.7667 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0794 | 0.1680 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0035 | -0.0038 |
| controlled_alt_vs_candidate_no_context | length_score | 0.3708 | 1.1867 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1750 | 0.2305 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1547 | 0.6529 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0156 | (-0.0111, 0.0448) | 0.1317 | 0.0156 | (-0.0255, 0.0588) | 0.2437 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0454 | (-0.1217, 0.0237) | 0.8847 | -0.0454 | (-0.1852, 0.0517) | 0.7803 |
| proposed_vs_candidate_no_context | naturalness | -0.0067 | (-0.0341, 0.0206) | 0.6733 | -0.0067 | (-0.0524, 0.0229) | 0.6243 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0192 | (-0.0170, 0.0582) | 0.1790 | 0.0192 | (-0.0366, 0.0760) | 0.2537 |
| proposed_vs_candidate_no_context | context_overlap | 0.0074 | (-0.0022, 0.0171) | 0.0677 | 0.0074 | (-0.0049, 0.0178) | 0.1153 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0469 | (-0.1414, 0.0335) | 0.8590 | -0.0469 | (-0.1968, 0.0658) | 0.7593 |
| proposed_vs_candidate_no_context | persona_style | -0.0395 | (-0.1051, 0.0123) | 0.9167 | -0.0395 | (-0.1272, 0.0229) | 0.8777 |
| proposed_vs_candidate_no_context | distinct1 | -0.0020 | (-0.0140, 0.0100) | 0.6333 | -0.0020 | (-0.0224, 0.0115) | 0.5693 |
| proposed_vs_candidate_no_context | length_score | -0.0292 | (-0.1167, 0.0542) | 0.7390 | -0.0292 | (-0.1691, 0.0645) | 0.6707 |
| proposed_vs_candidate_no_context | sentence_score | 0.0000 | (-0.0766, 0.0766) | 0.5573 | 0.0000 | (-0.0942, 0.0742) | 0.5167 |
| proposed_vs_candidate_no_context | overall_quality | -0.0106 | (-0.0495, 0.0258) | 0.7110 | -0.0106 | (-0.0796, 0.0388) | 0.6267 |
| controlled_vs_proposed_raw | context_relevance | 0.1989 | (0.1757, 0.2202) | 0.0000 | 0.1989 | (0.1694, 0.2255) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.1320 | (0.0883, 0.1755) | 0.0000 | 0.1320 | (0.0548, 0.2166) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.1071 | (0.0671, 0.1421) | 0.0000 | 0.1071 | (0.0347, 0.1541) | 0.0060 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2614 | (0.2312, 0.2892) | 0.0000 | 0.2614 | (0.2232, 0.2952) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0532 | (0.0416, 0.0650) | 0.0000 | 0.0532 | (0.0389, 0.0724) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1376 | (0.0948, 0.1818) | 0.0000 | 0.1376 | (0.0659, 0.2222) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.1093 | (0.0236, 0.2051) | 0.0053 | 0.1093 | (-0.0311, 0.3105) | 0.1487 |
| controlled_vs_proposed_raw | distinct1 | -0.0073 | (-0.0221, 0.0079) | 0.8183 | -0.0073 | (-0.0341, 0.0147) | 0.7223 |
| controlled_vs_proposed_raw | length_score | 0.4417 | (0.2917, 0.5750) | 0.0000 | 0.4417 | (0.1491, 0.6095) | 0.0040 |
| controlled_vs_proposed_raw | sentence_score | 0.2188 | (0.1531, 0.2844) | 0.0000 | 0.2188 | (0.1260, 0.2872) | 0.0000 |
| controlled_vs_proposed_raw | overall_quality | 0.1618 | (0.1363, 0.1857) | 0.0000 | 0.1618 | (0.1233, 0.2066) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2146 | (0.1977, 0.2314) | 0.0000 | 0.2146 | (0.1952, 0.2335) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.0866 | (0.0195, 0.1506) | 0.0067 | 0.0866 | (-0.0231, 0.1901) | 0.0600 |
| controlled_vs_candidate_no_context | naturalness | 0.1005 | (0.0616, 0.1372) | 0.0000 | 0.1005 | (0.0255, 0.1498) | 0.0050 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.2805 | (0.2555, 0.3041) | 0.0000 | 0.2805 | (0.2547, 0.3043) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0606 | (0.0527, 0.0691) | 0.0000 | 0.0606 | (0.0482, 0.0764) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.0908 | (0.0233, 0.1583) | 0.0037 | 0.0908 | (-0.0175, 0.2046) | 0.0473 |
| controlled_vs_candidate_no_context | persona_style | 0.0698 | (-0.0245, 0.1693) | 0.0713 | 0.0698 | (-0.0909, 0.2488) | 0.2297 |
| controlled_vs_candidate_no_context | distinct1 | -0.0093 | (-0.0257, 0.0066) | 0.8753 | -0.0093 | (-0.0404, 0.0138) | 0.7930 |
| controlled_vs_candidate_no_context | length_score | 0.4125 | (0.2573, 0.5552) | 0.0000 | 0.4125 | (0.1269, 0.5954) | 0.0047 |
| controlled_vs_candidate_no_context | sentence_score | 0.2188 | (0.1531, 0.2844) | 0.0000 | 0.2188 | (0.1260, 0.2864) | 0.0000 |
| controlled_vs_candidate_no_context | overall_quality | 0.1512 | (0.1193, 0.1812) | 0.0000 | 0.1512 | (0.0997, 0.1932) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0185 | (-0.0078, 0.0495) | 0.0960 | 0.0185 | (-0.0035, 0.0451) | 0.0583 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0077 | (-0.0419, 0.0209) | 0.6613 | -0.0077 | (-0.0570, 0.0250) | 0.6490 |
| controlled_alt_vs_controlled_default | naturalness | -0.0102 | (-0.0271, 0.0037) | 0.9147 | -0.0102 | (-0.0228, 0.0050) | 0.9013 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0246 | (-0.0142, 0.0668) | 0.1253 | 0.0246 | (-0.0066, 0.0621) | 0.0747 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0044 | (-0.0031, 0.0132) | 0.1547 | 0.0044 | (0.0002, 0.0095) | 0.0183 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0121 | (-0.0558, 0.0232) | 0.7143 | -0.0121 | (-0.0742, 0.0286) | 0.6783 |
| controlled_alt_vs_controlled_default | persona_style | 0.0096 | (-0.0156, 0.0446) | 0.3850 | 0.0096 | (-0.0143, 0.0559) | 0.3617 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0058 | (-0.0055, 0.0181) | 0.1707 | 0.0058 | (-0.0085, 0.0244) | 0.2477 |
| controlled_alt_vs_controlled_default | length_score | -0.0417 | (-0.1229, 0.0313) | 0.8513 | -0.0417 | (-0.1053, 0.0470) | 0.8233 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0437 | (-0.0984, 0.0000) | 0.9817 | -0.0437 | (-0.1021, -0.0087) | 1.0000 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0036 | (-0.0110, 0.0197) | 0.3407 | 0.0036 | (-0.0165, 0.0191) | 0.3593 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.2174 | (0.1846, 0.2520) | 0.0000 | 0.2174 | (0.1864, 0.2584) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1243 | (0.0732, 0.1726) | 0.0000 | 0.1243 | (0.0420, 0.2108) | 0.0020 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0969 | (0.0584, 0.1329) | 0.0000 | 0.0969 | (0.0323, 0.1426) | 0.0023 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2859 | (0.2426, 0.3348) | 0.0000 | 0.2859 | (0.2449, 0.3429) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0576 | (0.0465, 0.0685) | 0.0000 | 0.0576 | (0.0431, 0.0764) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1256 | (0.0708, 0.1763) | 0.0000 | 0.1256 | (0.0454, 0.2095) | 0.0020 |
| controlled_alt_vs_proposed_raw | persona_style | 0.1189 | (0.0429, 0.2106) | 0.0000 | 0.1189 | (0.0006, 0.3289) | 0.0250 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0015 | (-0.0196, 0.0151) | 0.5653 | -0.0015 | (-0.0314, 0.0186) | 0.5420 |
| controlled_alt_vs_proposed_raw | length_score | 0.4000 | (0.2542, 0.5438) | 0.0000 | 0.4000 | (0.1654, 0.5702) | 0.0020 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1750 | (0.0984, 0.2406) | 0.0000 | 0.1750 | (0.0538, 0.2649) | 0.0080 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1653 | (0.1371, 0.1933) | 0.0000 | 0.1653 | (0.1204, 0.2170) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2331 | (0.2041, 0.2646) | 0.0000 | 0.2331 | (0.1979, 0.2712) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.0789 | (0.0190, 0.1335) | 0.0037 | 0.0789 | (-0.0155, 0.1605) | 0.0517 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0903 | (0.0498, 0.1310) | 0.0000 | 0.0903 | (0.0194, 0.1405) | 0.0097 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3051 | (0.2645, 0.3469) | 0.0000 | 0.3051 | (0.2571, 0.3568) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0650 | (0.0548, 0.0755) | 0.0000 | 0.0650 | (0.0520, 0.0795) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.0787 | (0.0193, 0.1320) | 0.0047 | 0.0787 | (-0.0107, 0.1512) | 0.0367 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0794 | (-0.0107, 0.1699) | 0.0457 | 0.0794 | (-0.0769, 0.2598) | 0.1990 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0035 | (-0.0199, 0.0113) | 0.6573 | -0.0035 | (-0.0328, 0.0156) | 0.5990 |
| controlled_alt_vs_candidate_no_context | length_score | 0.3708 | (0.2052, 0.5250) | 0.0000 | 0.3708 | (0.1139, 0.5549) | 0.0043 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1750 | (0.0875, 0.2516) | 0.0000 | 0.1750 | (0.0559, 0.2561) | 0.0033 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1547 | (0.1254, 0.1816) | 0.0000 | 0.1547 | (0.0993, 0.1993) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 10 | 8 | 14 | 0.5312 | 0.5556 |
| proposed_vs_candidate_no_context | persona_consistency | 7 | 7 | 18 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | naturalness | 11 | 7 | 14 | 0.5625 | 0.6111 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 7 | 6 | 19 | 0.5156 | 0.5385 |
| proposed_vs_candidate_no_context | context_overlap | 13 | 5 | 14 | 0.6250 | 0.7222 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 5 | 7 | 20 | 0.4688 | 0.4167 |
| proposed_vs_candidate_no_context | persona_style | 2 | 3 | 27 | 0.4844 | 0.4000 |
| proposed_vs_candidate_no_context | distinct1 | 6 | 8 | 18 | 0.4688 | 0.4286 |
| proposed_vs_candidate_no_context | length_score | 9 | 9 | 14 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | sentence_score | 6 | 6 | 20 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | overall_quality | 10 | 8 | 14 | 0.5312 | 0.5556 |
| controlled_vs_proposed_raw | context_relevance | 32 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_proposed_raw | persona_consistency | 21 | 1 | 10 | 0.8125 | 0.9545 |
| controlled_vs_proposed_raw | naturalness | 25 | 7 | 0 | 0.7812 | 0.7812 |
| controlled_vs_proposed_raw | context_keyword_coverage | 31 | 0 | 1 | 0.9844 | 1.0000 |
| controlled_vs_proposed_raw | context_overlap | 31 | 1 | 0 | 0.9688 | 0.9688 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 21 | 1 | 10 | 0.8125 | 0.9545 |
| controlled_vs_proposed_raw | persona_style | 8 | 1 | 23 | 0.6094 | 0.8889 |
| controlled_vs_proposed_raw | distinct1 | 19 | 13 | 0 | 0.5938 | 0.5938 |
| controlled_vs_proposed_raw | length_score | 28 | 4 | 0 | 0.8750 | 0.8750 |
| controlled_vs_proposed_raw | sentence_score | 21 | 1 | 10 | 0.8125 | 0.9545 |
| controlled_vs_proposed_raw | overall_quality | 31 | 1 | 0 | 0.9688 | 0.9688 |
| controlled_vs_candidate_no_context | context_relevance | 32 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 18 | 5 | 9 | 0.7031 | 0.7826 |
| controlled_vs_candidate_no_context | naturalness | 24 | 8 | 0 | 0.7500 | 0.7500 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 32 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 32 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 18 | 5 | 9 | 0.7031 | 0.7826 |
| controlled_vs_candidate_no_context | persona_style | 7 | 3 | 22 | 0.5625 | 0.7000 |
| controlled_vs_candidate_no_context | distinct1 | 17 | 15 | 0 | 0.5312 | 0.5312 |
| controlled_vs_candidate_no_context | length_score | 26 | 6 | 0 | 0.8125 | 0.8125 |
| controlled_vs_candidate_no_context | sentence_score | 21 | 1 | 10 | 0.8125 | 0.9545 |
| controlled_vs_candidate_no_context | overall_quality | 28 | 4 | 0 | 0.8750 | 0.8750 |
| controlled_alt_vs_controlled_default | context_relevance | 7 | 7 | 18 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | persona_consistency | 4 | 5 | 23 | 0.4844 | 0.4444 |
| controlled_alt_vs_controlled_default | naturalness | 6 | 8 | 18 | 0.4688 | 0.4286 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 5 | 4 | 23 | 0.5156 | 0.5556 |
| controlled_alt_vs_controlled_default | context_overlap | 8 | 6 | 18 | 0.5312 | 0.5714 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 3 | 4 | 25 | 0.4844 | 0.4286 |
| controlled_alt_vs_controlled_default | persona_style | 1 | 1 | 30 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | distinct1 | 7 | 7 | 18 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | length_score | 7 | 7 | 18 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | sentence_score | 1 | 5 | 26 | 0.4375 | 0.1667 |
| controlled_alt_vs_controlled_default | overall_quality | 6 | 8 | 18 | 0.4688 | 0.4286 |
| controlled_alt_vs_proposed_raw | context_relevance | 31 | 1 | 0 | 0.9688 | 0.9688 |
| controlled_alt_vs_proposed_raw | persona_consistency | 22 | 3 | 7 | 0.7969 | 0.8800 |
| controlled_alt_vs_proposed_raw | naturalness | 24 | 8 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 31 | 0 | 1 | 0.9844 | 1.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 30 | 2 | 0 | 0.9375 | 0.9375 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 22 | 3 | 7 | 0.7969 | 0.8800 |
| controlled_alt_vs_proposed_raw | persona_style | 8 | 0 | 24 | 0.6250 | 1.0000 |
| controlled_alt_vs_proposed_raw | distinct1 | 20 | 12 | 0 | 0.6250 | 0.6250 |
| controlled_alt_vs_proposed_raw | length_score | 26 | 6 | 0 | 0.8125 | 0.8125 |
| controlled_alt_vs_proposed_raw | sentence_score | 18 | 2 | 12 | 0.7500 | 0.9000 |
| controlled_alt_vs_proposed_raw | overall_quality | 31 | 1 | 0 | 0.9688 | 0.9688 |
| controlled_alt_vs_candidate_no_context | context_relevance | 32 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 19 | 6 | 7 | 0.7031 | 0.7600 |
| controlled_alt_vs_candidate_no_context | naturalness | 24 | 8 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 32 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 32 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 19 | 6 | 7 | 0.7031 | 0.7600 |
| controlled_alt_vs_candidate_no_context | persona_style | 8 | 2 | 22 | 0.5938 | 0.8000 |
| controlled_alt_vs_candidate_no_context | distinct1 | 19 | 13 | 0 | 0.5938 | 0.5938 |
| controlled_alt_vs_candidate_no_context | length_score | 24 | 8 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_candidate_no_context | sentence_score | 20 | 4 | 8 | 0.7500 | 0.8333 |
| controlled_alt_vs_candidate_no_context | overall_quality | 30 | 2 | 0 | 0.9375 | 0.9375 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.0000 | 0.2188 | 0.2188 |
| proposed_contextual_controlled_alt | 0.0000 | 0.0000 | 0.0000 | 0.4062 | 0.1875 |
| proposed_contextual | 0.0000 | 0.0000 | 0.5938 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.5625 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `8`
- Unique template signatures: `28`
- Template signature ratio: `0.8750`
- Effective sample size by source clustering: `6.83`
- Effective sample size by template-signature clustering: `24.38`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: unavailable (disabled_by_flag).

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
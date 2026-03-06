# Proposal Alignment Evaluation Report

- Run ID: `20260305T215331Z`
- Generated: `2026-03-05T21:58:54.818708+00:00`
- Scenarios: `artifacts\proposal_control_tuning\runtime_compare_seed31\20260305T215331Z\scenarios.jsonl`
- Scenario count: `32`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_rt`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2821 (0.2489, 0.3170) | 0.3355 (0.2885, 0.3887) | 0.8848 (0.8629, 0.9063) | 0.4236 (0.4009, 0.4460) | n/a |
| proposed_contextual_controlled_rt | 0.2800 (0.2460, 0.3166) | 0.2972 (0.2588, 0.3391) | 0.8983 (0.8783, 0.9171) | 0.4113 (0.3900, 0.4336) | n/a |
| proposed_contextual | 0.0475 (0.0242, 0.0758) | 0.1553 (0.1034, 0.2209) | 0.8049 (0.7797, 0.8311) | 0.2325 (0.2063, 0.2631) | n/a |
| candidate_no_context | 0.0351 (0.0210, 0.0509) | 0.1775 (0.1317, 0.2304) | 0.8322 (0.8012, 0.8643) | 0.2402 (0.2148, 0.2681) | n/a |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0124 | 0.3533 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0222 | -0.1252 |
| proposed_vs_candidate_no_context | naturalness | -0.0273 | -0.0328 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0130 | 0.3873 |
| proposed_vs_candidate_no_context | context_overlap | 0.0109 | 0.2840 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0179 | -0.2051 |
| proposed_vs_candidate_no_context | persona_style | -0.0397 | -0.0737 |
| proposed_vs_candidate_no_context | distinct1 | -0.0105 | -0.0111 |
| proposed_vs_candidate_no_context | length_score | -0.0938 | -0.2446 |
| proposed_vs_candidate_no_context | sentence_score | -0.0437 | -0.0560 |
| proposed_vs_candidate_no_context | overall_quality | -0.0077 | -0.0320 |
| controlled_vs_proposed_raw | context_relevance | 0.2346 | 4.9402 |
| controlled_vs_proposed_raw | persona_consistency | 0.1802 | 1.1606 |
| controlled_vs_proposed_raw | naturalness | 0.0799 | 0.0992 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.3076 | 6.5959 |
| controlled_vs_proposed_raw | context_overlap | 0.0642 | 1.2979 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2000 | 2.8903 |
| controlled_vs_proposed_raw | persona_style | 0.1011 | 0.2023 |
| controlled_vs_proposed_raw | distinct1 | 0.0019 | 0.0021 |
| controlled_vs_proposed_raw | length_score | 0.3135 | 1.0827 |
| controlled_vs_proposed_raw | sentence_score | 0.1641 | 0.2225 |
| controlled_vs_proposed_raw | overall_quality | 0.1911 | 0.8218 |
| controlled_vs_candidate_no_context | context_relevance | 0.2470 | 7.0388 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1580 | 0.8900 |
| controlled_vs_candidate_no_context | naturalness | 0.0526 | 0.0632 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3206 | 9.5380 |
| controlled_vs_candidate_no_context | context_overlap | 0.0751 | 1.9504 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1821 | 2.0923 |
| controlled_vs_candidate_no_context | persona_style | 0.0614 | 0.1138 |
| controlled_vs_candidate_no_context | distinct1 | -0.0085 | -0.0090 |
| controlled_vs_candidate_no_context | length_score | 0.2198 | 0.5734 |
| controlled_vs_candidate_no_context | sentence_score | 0.1203 | 0.1540 |
| controlled_vs_candidate_no_context | overall_quality | 0.1834 | 0.7634 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0020 | -0.0072 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0383 | -0.1141 |
| controlled_alt_vs_controlled_default | naturalness | 0.0135 | 0.0152 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0005 | 0.0013 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0079 | -0.0695 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0472 | -0.1752 |
| controlled_alt_vs_controlled_default | persona_style | -0.0028 | -0.0046 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0158 | 0.0169 |
| controlled_alt_vs_controlled_default | length_score | 0.0083 | 0.0138 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0547 | 0.0607 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0123 | -0.0291 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.2326 | 4.8972 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1419 | 0.9140 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0934 | 0.1160 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.3081 | 6.6061 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0563 | 1.1381 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1528 | 2.2086 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0983 | 0.1968 |
| controlled_alt_vs_proposed_raw | distinct1 | 0.0178 | 0.0190 |
| controlled_alt_vs_proposed_raw | length_score | 0.3219 | 1.1115 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.2188 | 0.2966 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1788 | 0.7688 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2450 | 6.9806 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1197 | 0.6743 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0660 | 0.0794 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3211 | 9.5521 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0672 | 1.7453 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1350 | 1.5504 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0586 | 0.1086 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0073 | 0.0077 |
| controlled_alt_vs_candidate_no_context | length_score | 0.2281 | 0.5951 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1750 | 0.2240 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1711 | 0.7122 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0124 | (-0.0128, 0.0402) | 0.1833 | 0.0124 | (-0.0079, 0.0384) | 0.1400 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0222 | (-0.0807, 0.0310) | 0.7963 | -0.0222 | (-0.0731, 0.0261) | 0.8153 |
| proposed_vs_candidate_no_context | naturalness | -0.0273 | (-0.0520, -0.0042) | 0.9907 | -0.0273 | (-0.0559, -0.0008) | 0.9780 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0130 | (-0.0201, 0.0509) | 0.2430 | 0.0130 | (-0.0142, 0.0479) | 0.2097 |
| proposed_vs_candidate_no_context | context_overlap | 0.0109 | (-0.0002, 0.0240) | 0.0260 | 0.0109 | (0.0011, 0.0235) | 0.0127 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0179 | (-0.0826, 0.0439) | 0.7040 | -0.0179 | (-0.0632, 0.0322) | 0.7653 |
| proposed_vs_candidate_no_context | persona_style | -0.0397 | (-0.1035, 0.0167) | 0.9073 | -0.0397 | (-0.1369, 0.0227) | 0.8827 |
| proposed_vs_candidate_no_context | distinct1 | -0.0105 | (-0.0243, 0.0028) | 0.9400 | -0.0105 | (-0.0228, 0.0017) | 0.9543 |
| proposed_vs_candidate_no_context | length_score | -0.0938 | (-0.1792, -0.0177) | 0.9917 | -0.0938 | (-0.1881, -0.0012) | 0.9757 |
| proposed_vs_candidate_no_context | sentence_score | -0.0437 | (-0.1094, 0.0219) | 0.9210 | -0.0437 | (-0.1250, 0.0219) | 0.9230 |
| proposed_vs_candidate_no_context | overall_quality | -0.0077 | (-0.0389, 0.0200) | 0.7070 | -0.0077 | (-0.0384, 0.0210) | 0.7070 |
| controlled_vs_proposed_raw | context_relevance | 0.2346 | (0.2000, 0.2707) | 0.0000 | 0.2346 | (0.2174, 0.2533) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.1802 | (0.1114, 0.2509) | 0.0000 | 0.1802 | (0.0861, 0.2733) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0799 | (0.0446, 0.1124) | 0.0000 | 0.0799 | (0.0295, 0.1273) | 0.0010 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.3076 | (0.2670, 0.3498) | 0.0000 | 0.3076 | (0.2841, 0.3332) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0642 | (0.0453, 0.0840) | 0.0000 | 0.0642 | (0.0545, 0.0738) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2000 | (0.1106, 0.2833) | 0.0000 | 0.2000 | (0.0878, 0.3115) | 0.0003 |
| controlled_vs_proposed_raw | persona_style | 0.1011 | (0.0361, 0.1755) | 0.0003 | 0.1011 | (0.0000, 0.2561) | 0.0290 |
| controlled_vs_proposed_raw | distinct1 | 0.0019 | (-0.0147, 0.0182) | 0.3943 | 0.0019 | (-0.0146, 0.0211) | 0.4190 |
| controlled_vs_proposed_raw | length_score | 0.3135 | (0.1729, 0.4552) | 0.0000 | 0.3135 | (0.1048, 0.5104) | 0.0037 |
| controlled_vs_proposed_raw | sentence_score | 0.1641 | (0.0766, 0.2516) | 0.0000 | 0.1641 | (0.0700, 0.2722) | 0.0000 |
| controlled_vs_proposed_raw | overall_quality | 0.1911 | (0.1590, 0.2221) | 0.0000 | 0.1911 | (0.1541, 0.2336) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2470 | (0.2136, 0.2803) | 0.0000 | 0.2470 | (0.2240, 0.2735) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1580 | (0.0991, 0.2169) | 0.0000 | 0.1580 | (0.1018, 0.2178) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.0526 | (0.0176, 0.0876) | 0.0010 | 0.0526 | (0.0002, 0.0998) | 0.0250 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3206 | (0.2780, 0.3644) | 0.0000 | 0.3206 | (0.2890, 0.3606) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0751 | (0.0591, 0.0922) | 0.0000 | 0.0751 | (0.0703, 0.0811) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1821 | (0.1132, 0.2519) | 0.0000 | 0.1821 | (0.1089, 0.2595) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.0614 | (0.0043, 0.1263) | 0.0143 | 0.0614 | (0.0079, 0.1460) | 0.0233 |
| controlled_vs_candidate_no_context | distinct1 | -0.0085 | (-0.0250, 0.0079) | 0.8460 | -0.0085 | (-0.0274, 0.0103) | 0.7997 |
| controlled_vs_candidate_no_context | length_score | 0.2198 | (0.0865, 0.3552) | 0.0013 | 0.2198 | (0.0138, 0.3958) | 0.0183 |
| controlled_vs_candidate_no_context | sentence_score | 0.1203 | (0.0328, 0.1969) | 0.0050 | 0.1203 | (0.0189, 0.2431) | 0.0137 |
| controlled_vs_candidate_no_context | overall_quality | 0.1834 | (0.1493, 0.2143) | 0.0000 | 0.1834 | (0.1639, 0.2094) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0020 | (-0.0517, 0.0502) | 0.5387 | -0.0020 | (-0.0579, 0.0524) | 0.4827 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0383 | (-0.0805, 0.0005) | 0.9730 | -0.0383 | (-0.1021, 0.0125) | 0.9137 |
| controlled_alt_vs_controlled_default | naturalness | 0.0135 | (-0.0128, 0.0413) | 0.1673 | 0.0135 | (-0.0147, 0.0409) | 0.1783 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0005 | (-0.0644, 0.0679) | 0.4920 | 0.0005 | (-0.0762, 0.0720) | 0.4507 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0079 | (-0.0324, 0.0146) | 0.7513 | -0.0079 | (-0.0276, 0.0142) | 0.7567 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0472 | (-0.0967, -0.0012) | 0.9780 | -0.0472 | (-0.1222, 0.0108) | 0.9403 |
| controlled_alt_vs_controlled_default | persona_style | -0.0028 | (-0.0372, 0.0313) | 0.5720 | -0.0028 | (-0.0380, 0.0417) | 0.5897 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0158 | (0.0024, 0.0288) | 0.0090 | 0.0158 | (0.0030, 0.0265) | 0.0070 |
| controlled_alt_vs_controlled_default | length_score | 0.0083 | (-0.1250, 0.1521) | 0.4570 | 0.0083 | (-0.1364, 0.1549) | 0.4613 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0547 | (-0.0109, 0.1203) | 0.0557 | 0.0547 | (0.0000, 0.0955) | 0.0343 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0123 | (-0.0386, 0.0139) | 0.8133 | -0.0123 | (-0.0404, 0.0191) | 0.7623 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.2326 | (0.1909, 0.2732) | 0.0000 | 0.2326 | (0.1725, 0.2923) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1419 | (0.0734, 0.2048) | 0.0000 | 0.1419 | (0.0638, 0.2260) | 0.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0934 | (0.0575, 0.1254) | 0.0000 | 0.0934 | (0.0442, 0.1403) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.3081 | (0.2502, 0.3607) | 0.0000 | 0.3081 | (0.2303, 0.3812) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0563 | (0.0386, 0.0761) | 0.0000 | 0.0563 | (0.0359, 0.0830) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1528 | (0.0649, 0.2244) | 0.0003 | 0.1528 | (0.0708, 0.2477) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0983 | (0.0246, 0.1815) | 0.0030 | 0.0983 | (-0.0145, 0.2638) | 0.0653 |
| controlled_alt_vs_proposed_raw | distinct1 | 0.0178 | (0.0049, 0.0301) | 0.0057 | 0.0178 | (0.0073, 0.0292) | 0.0007 |
| controlled_alt_vs_proposed_raw | length_score | 0.3219 | (0.1771, 0.4583) | 0.0000 | 0.3219 | (0.1128, 0.5212) | 0.0023 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.2188 | (0.1422, 0.2844) | 0.0000 | 0.2188 | (0.1379, 0.3033) | 0.0000 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1788 | (0.1433, 0.2135) | 0.0000 | 0.1788 | (0.1391, 0.2266) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2450 | (0.2084, 0.2785) | 0.0000 | 0.2450 | (0.1905, 0.3023) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1197 | (0.0633, 0.1757) | 0.0000 | 0.1197 | (0.0639, 0.1791) | 0.0000 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0660 | (0.0250, 0.1045) | 0.0010 | 0.0660 | (0.0150, 0.1203) | 0.0067 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3211 | (0.2757, 0.3671) | 0.0000 | 0.3211 | (0.2468, 0.3951) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0672 | (0.0541, 0.0824) | 0.0000 | 0.0672 | (0.0484, 0.0891) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1350 | (0.0763, 0.1921) | 0.0000 | 0.1350 | (0.0740, 0.2059) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0586 | (-0.0115, 0.1389) | 0.0470 | 0.0586 | (-0.0214, 0.1832) | 0.0970 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0073 | (-0.0081, 0.0223) | 0.1793 | 0.0073 | (-0.0080, 0.0209) | 0.1647 |
| controlled_alt_vs_candidate_no_context | length_score | 0.2281 | (0.0583, 0.3771) | 0.0020 | 0.2281 | (0.0214, 0.4414) | 0.0143 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1750 | (0.0984, 0.2516) | 0.0000 | 0.1750 | (0.1132, 0.2500) | 0.0000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1711 | (0.1416, 0.2001) | 0.0000 | 0.1711 | (0.1338, 0.2149) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 10 | 8 | 14 | 0.5312 | 0.5556 |
| proposed_vs_candidate_no_context | persona_consistency | 6 | 8 | 18 | 0.4688 | 0.4286 |
| proposed_vs_candidate_no_context | naturalness | 7 | 11 | 14 | 0.4375 | 0.3889 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 7 | 6 | 19 | 0.5156 | 0.5385 |
| proposed_vs_candidate_no_context | context_overlap | 10 | 8 | 14 | 0.5312 | 0.5556 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 5 | 6 | 21 | 0.4844 | 0.4545 |
| proposed_vs_candidate_no_context | persona_style | 1 | 5 | 26 | 0.4375 | 0.1667 |
| proposed_vs_candidate_no_context | distinct1 | 5 | 12 | 15 | 0.3906 | 0.2941 |
| proposed_vs_candidate_no_context | length_score | 5 | 12 | 15 | 0.3906 | 0.2941 |
| proposed_vs_candidate_no_context | sentence_score | 3 | 7 | 22 | 0.4375 | 0.3000 |
| proposed_vs_candidate_no_context | overall_quality | 9 | 9 | 14 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | context_relevance | 31 | 1 | 0 | 0.9688 | 0.9688 |
| controlled_vs_proposed_raw | persona_consistency | 26 | 2 | 4 | 0.8750 | 0.9286 |
| controlled_vs_proposed_raw | naturalness | 25 | 7 | 0 | 0.7812 | 0.7812 |
| controlled_vs_proposed_raw | context_keyword_coverage | 31 | 0 | 1 | 0.9844 | 1.0000 |
| controlled_vs_proposed_raw | context_overlap | 29 | 3 | 0 | 0.9062 | 0.9062 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 26 | 2 | 4 | 0.8750 | 0.9286 |
| controlled_vs_proposed_raw | persona_style | 8 | 1 | 23 | 0.6094 | 0.8889 |
| controlled_vs_proposed_raw | distinct1 | 18 | 13 | 1 | 0.5781 | 0.5806 |
| controlled_vs_proposed_raw | length_score | 23 | 8 | 1 | 0.7344 | 0.7419 |
| controlled_vs_proposed_raw | sentence_score | 19 | 4 | 9 | 0.7344 | 0.8261 |
| controlled_vs_proposed_raw | overall_quality | 31 | 1 | 0 | 0.9688 | 0.9688 |
| controlled_vs_candidate_no_context | context_relevance | 32 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 26 | 4 | 2 | 0.8438 | 0.8667 |
| controlled_vs_candidate_no_context | naturalness | 23 | 9 | 0 | 0.7188 | 0.7188 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 32 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 31 | 1 | 0 | 0.9688 | 0.9688 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 26 | 4 | 2 | 0.8438 | 0.8667 |
| controlled_vs_candidate_no_context | persona_style | 7 | 2 | 23 | 0.5781 | 0.7778 |
| controlled_vs_candidate_no_context | distinct1 | 15 | 17 | 0 | 0.4688 | 0.4688 |
| controlled_vs_candidate_no_context | length_score | 21 | 10 | 1 | 0.6719 | 0.6774 |
| controlled_vs_candidate_no_context | sentence_score | 15 | 4 | 13 | 0.6719 | 0.7895 |
| controlled_vs_candidate_no_context | overall_quality | 30 | 2 | 0 | 0.9375 | 0.9375 |
| controlled_alt_vs_controlled_default | context_relevance | 13 | 15 | 4 | 0.4688 | 0.4643 |
| controlled_alt_vs_controlled_default | persona_consistency | 5 | 9 | 18 | 0.4375 | 0.3571 |
| controlled_alt_vs_controlled_default | naturalness | 14 | 14 | 4 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 11 | 11 | 10 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | context_overlap | 13 | 15 | 4 | 0.4688 | 0.4643 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 3 | 7 | 22 | 0.4375 | 0.3000 |
| controlled_alt_vs_controlled_default | persona_style | 3 | 4 | 25 | 0.4844 | 0.4286 |
| controlled_alt_vs_controlled_default | distinct1 | 19 | 9 | 4 | 0.6562 | 0.6786 |
| controlled_alt_vs_controlled_default | length_score | 14 | 13 | 5 | 0.5156 | 0.5185 |
| controlled_alt_vs_controlled_default | sentence_score | 7 | 2 | 23 | 0.5781 | 0.7778 |
| controlled_alt_vs_controlled_default | overall_quality | 13 | 15 | 4 | 0.4688 | 0.4643 |
| controlled_alt_vs_proposed_raw | context_relevance | 31 | 1 | 0 | 0.9688 | 0.9688 |
| controlled_alt_vs_proposed_raw | persona_consistency | 24 | 4 | 4 | 0.8125 | 0.8571 |
| controlled_alt_vs_proposed_raw | naturalness | 28 | 4 | 0 | 0.8750 | 0.8750 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 31 | 1 | 0 | 0.9688 | 0.9688 |
| controlled_alt_vs_proposed_raw | context_overlap | 28 | 4 | 0 | 0.8750 | 0.8750 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 24 | 3 | 5 | 0.8281 | 0.8889 |
| controlled_alt_vs_proposed_raw | persona_style | 8 | 3 | 21 | 0.5781 | 0.7273 |
| controlled_alt_vs_proposed_raw | distinct1 | 22 | 7 | 3 | 0.7344 | 0.7586 |
| controlled_alt_vs_proposed_raw | length_score | 25 | 7 | 0 | 0.7812 | 0.7812 |
| controlled_alt_vs_proposed_raw | sentence_score | 22 | 2 | 8 | 0.8125 | 0.9167 |
| controlled_alt_vs_proposed_raw | overall_quality | 29 | 3 | 0 | 0.9062 | 0.9062 |
| controlled_alt_vs_candidate_no_context | context_relevance | 31 | 1 | 0 | 0.9688 | 0.9688 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 23 | 6 | 3 | 0.7656 | 0.7931 |
| controlled_alt_vs_candidate_no_context | naturalness | 23 | 9 | 0 | 0.7188 | 0.7188 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 30 | 1 | 1 | 0.9531 | 0.9677 |
| controlled_alt_vs_candidate_no_context | context_overlap | 32 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 23 | 6 | 3 | 0.7656 | 0.7931 |
| controlled_alt_vs_candidate_no_context | persona_style | 7 | 5 | 20 | 0.5312 | 0.5833 |
| controlled_alt_vs_candidate_no_context | distinct1 | 20 | 10 | 2 | 0.6562 | 0.6667 |
| controlled_alt_vs_candidate_no_context | length_score | 22 | 10 | 0 | 0.6875 | 0.6875 |
| controlled_alt_vs_candidate_no_context | sentence_score | 18 | 2 | 12 | 0.7500 | 0.9000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 32 | 0 | 0 | 1.0000 | 1.0000 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.3125 | 0.4062 | 0.5938 |
| proposed_contextual_controlled_rt | 0.0000 | 0.0000 | 0.2500 | 0.5000 | 0.5000 |
| proposed_contextual | 0.0000 | 0.0000 | 0.5938 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `8`
- Unique template signatures: `30`
- Template signature ratio: `0.9375`
- Effective sample size by source clustering: `7.42`
- Effective sample size by template-signature clustering: `28.44`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: unavailable (disabled_by_flag).

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
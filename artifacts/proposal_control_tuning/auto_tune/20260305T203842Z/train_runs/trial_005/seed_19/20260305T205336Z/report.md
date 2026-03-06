# Proposal Alignment Evaluation Report

- Run ID: `20260305T205336Z`
- Generated: `2026-03-05T20:56:42.221147+00:00`
- Scenarios: `artifacts\proposal_control_tuning\auto_tune\20260305T203842Z\train_runs\trial_005\seed_19\20260305T205336Z\scenarios.jsonl`
- Scenario count: `20`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_tuned`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2623 (0.2227, 0.3091) | 0.2967 (0.2343, 0.3643) | 0.9051 (0.8800, 0.9275) | 0.4043 (0.3790, 0.4316) | n/a |
| proposed_contextual_controlled_tuned | 0.2399 (0.2251, 0.2548) | 0.2950 (0.2414, 0.3576) | 0.9278 (0.9133, 0.9403) | 0.3974 (0.3769, 0.4207) | n/a |
| proposed_contextual | 0.0718 (0.0295, 0.1237) | 0.1445 (0.0983, 0.2015) | 0.8030 (0.7717, 0.8358) | 0.2386 (0.2046, 0.2755) | n/a |
| candidate_no_context | 0.0334 (0.0187, 0.0495) | 0.1512 (0.1040, 0.2007) | 0.8137 (0.7791, 0.8527) | 0.2254 (0.1997, 0.2548) | n/a |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0385 | 1.1527 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0067 | -0.0441 |
| proposed_vs_candidate_no_context | naturalness | -0.0108 | -0.0132 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0500 | 1.5714 |
| proposed_vs_candidate_no_context | context_overlap | 0.0115 | 0.3120 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0083 | -0.1842 |
| proposed_vs_candidate_no_context | persona_style | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | distinct1 | -0.0017 | -0.0018 |
| proposed_vs_candidate_no_context | length_score | -0.0417 | -0.1269 |
| proposed_vs_candidate_no_context | sentence_score | -0.0175 | -0.0237 |
| proposed_vs_candidate_no_context | overall_quality | 0.0132 | 0.0585 |
| controlled_vs_proposed_raw | context_relevance | 0.1905 | 2.6516 |
| controlled_vs_proposed_raw | persona_consistency | 0.1522 | 1.0530 |
| controlled_vs_proposed_raw | naturalness | 0.1021 | 0.1272 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2500 | 3.0556 |
| controlled_vs_proposed_raw | context_overlap | 0.0516 | 1.0625 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1852 | 5.0194 |
| controlled_vs_proposed_raw | persona_style | 0.0200 | 0.0348 |
| controlled_vs_proposed_raw | distinct1 | 0.0072 | 0.0077 |
| controlled_vs_proposed_raw | length_score | 0.4000 | 1.3953 |
| controlled_vs_proposed_raw | sentence_score | 0.1925 | 0.2674 |
| controlled_vs_proposed_raw | overall_quality | 0.1658 | 0.6948 |
| controlled_vs_candidate_no_context | context_relevance | 0.2289 | 6.8607 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1455 | 0.9625 |
| controlled_vs_candidate_no_context | naturalness | 0.0914 | 0.1123 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3000 | 9.4286 |
| controlled_vs_candidate_no_context | context_overlap | 0.0631 | 1.7061 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1769 | 3.9105 |
| controlled_vs_candidate_no_context | persona_style | 0.0200 | 0.0348 |
| controlled_vs_candidate_no_context | distinct1 | 0.0055 | 0.0059 |
| controlled_vs_candidate_no_context | length_score | 0.3583 | 1.0914 |
| controlled_vs_candidate_no_context | sentence_score | 0.1750 | 0.2373 |
| controlled_vs_candidate_no_context | overall_quality | 0.1790 | 0.7940 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0224 | -0.0855 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0017 | -0.0058 |
| controlled_alt_vs_controlled_default | naturalness | 0.0228 | 0.0251 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0318 | -0.0959 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0005 | -0.0053 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0055 | -0.0247 |
| controlled_alt_vs_controlled_default | persona_style | 0.0133 | 0.0224 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0065 | 0.0069 |
| controlled_alt_vs_controlled_default | length_score | 0.0833 | 0.1214 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0350 | 0.0384 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0069 | -0.0171 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1680 | 2.3393 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1505 | 1.0412 |
| controlled_alt_vs_proposed_raw | naturalness | 0.1249 | 0.1555 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2182 | 2.6667 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0510 | 1.0515 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1798 | 4.8710 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0333 | 0.0580 |
| controlled_alt_vs_proposed_raw | distinct1 | 0.0137 | 0.0146 |
| controlled_alt_vs_proposed_raw | length_score | 0.4833 | 1.6860 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.2275 | 0.3160 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1589 | 0.6658 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2065 | 6.1884 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1438 | 0.9512 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.1141 | 0.1402 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.2682 | 8.4286 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0626 | 1.6917 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1714 | 3.7895 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0333 | 0.0580 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0120 | 0.0128 |
| controlled_alt_vs_candidate_no_context | length_score | 0.4417 | 1.3452 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.2100 | 0.2847 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1720 | 0.7634 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0385 | (-0.0101, 0.0960) | 0.0703 | 0.0385 | (-0.0049, 0.0864) | 0.0713 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0067 | (-0.0476, 0.0400) | 0.6437 | -0.0067 | (-0.0211, 0.0000) | 0.7547 |
| proposed_vs_candidate_no_context | naturalness | -0.0108 | (-0.0551, 0.0319) | 0.6927 | -0.0108 | (-0.0334, 0.0143) | 0.7853 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0500 | (-0.0136, 0.1273) | 0.0800 | 0.0500 | (-0.0087, 0.1148) | 0.0703 |
| proposed_vs_candidate_no_context | context_overlap | 0.0115 | (-0.0018, 0.0290) | 0.0507 | 0.0115 | (0.0022, 0.0259) | 0.0117 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0083 | (-0.0595, 0.0488) | 0.6753 | -0.0083 | (-0.0263, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | persona_style | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | distinct1 | -0.0017 | (-0.0181, 0.0144) | 0.5890 | -0.0017 | (-0.0184, 0.0204) | 0.6363 |
| proposed_vs_candidate_no_context | length_score | -0.0417 | (-0.1983, 0.1183) | 0.6980 | -0.0417 | (-0.1067, 0.0123) | 0.8000 |
| proposed_vs_candidate_no_context | sentence_score | -0.0175 | (-0.1050, 0.0700) | 0.7253 | -0.0175 | (-0.0609, 0.0412) | 0.8107 |
| proposed_vs_candidate_no_context | overall_quality | 0.0132 | (-0.0269, 0.0560) | 0.2773 | 0.0132 | (-0.0076, 0.0361) | 0.1080 |
| controlled_vs_proposed_raw | context_relevance | 0.1905 | (0.1329, 0.2496) | 0.0000 | 0.1905 | (0.1423, 0.2371) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.1522 | (0.0999, 0.2050) | 0.0000 | 0.1522 | (0.0996, 0.1930) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.1021 | (0.0532, 0.1440) | 0.0000 | 0.1021 | (0.0579, 0.1562) | 0.0000 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2500 | (0.1818, 0.3227) | 0.0000 | 0.2500 | (0.1818, 0.3117) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0516 | (0.0296, 0.0715) | 0.0000 | 0.0516 | (0.0405, 0.0618) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1852 | (0.1221, 0.2500) | 0.0000 | 0.1852 | (0.1167, 0.2326) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.0200 | (-0.0100, 0.0617) | 0.1593 | 0.0200 | (-0.0115, 0.0882) | 0.3293 |
| controlled_vs_proposed_raw | distinct1 | 0.0072 | (-0.0137, 0.0254) | 0.2350 | 0.0072 | (-0.0157, 0.0269) | 0.2730 |
| controlled_vs_proposed_raw | length_score | 0.4000 | (0.2250, 0.5533) | 0.0000 | 0.4000 | (0.2500, 0.5833) | 0.0000 |
| controlled_vs_proposed_raw | sentence_score | 0.1925 | (0.1050, 0.2800) | 0.0000 | 0.1925 | (0.1114, 0.2917) | 0.0000 |
| controlled_vs_proposed_raw | overall_quality | 0.1658 | (0.1250, 0.2031) | 0.0000 | 0.1658 | (0.1313, 0.1874) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2289 | (0.1898, 0.2773) | 0.0000 | 0.2289 | (0.1839, 0.3035) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1455 | (0.0946, 0.1935) | 0.0000 | 0.1455 | (0.0913, 0.1894) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.0914 | (0.0503, 0.1309) | 0.0000 | 0.0914 | (0.0494, 0.1544) | 0.0000 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3000 | (0.2455, 0.3682) | 0.0000 | 0.3000 | (0.2386, 0.4040) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0631 | (0.0491, 0.0776) | 0.0000 | 0.0631 | (0.0530, 0.0729) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1769 | (0.1129, 0.2398) | 0.0000 | 0.1769 | (0.1137, 0.2326) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.0200 | (-0.0100, 0.0617) | 0.1593 | 0.0200 | (-0.0120, 0.0882) | 0.3403 |
| controlled_vs_candidate_no_context | distinct1 | 0.0055 | (-0.0107, 0.0215) | 0.2493 | 0.0055 | (-0.0065, 0.0215) | 0.2047 |
| controlled_vs_candidate_no_context | length_score | 0.3583 | (0.2017, 0.5117) | 0.0000 | 0.3583 | (0.2125, 0.5813) | 0.0000 |
| controlled_vs_candidate_no_context | sentence_score | 0.1750 | (0.0700, 0.2625) | 0.0017 | 0.1750 | (0.0808, 0.3062) | 0.0007 |
| controlled_vs_candidate_no_context | overall_quality | 0.1790 | (0.1515, 0.2060) | 0.0000 | 0.1790 | (0.1561, 0.2044) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0224 | (-0.0629, 0.0148) | 0.8727 | -0.0224 | (-0.0774, 0.0090) | 0.7870 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0017 | (-0.0377, 0.0308) | 0.5353 | -0.0017 | (-0.0200, 0.0242) | 0.6117 |
| controlled_alt_vs_controlled_default | naturalness | 0.0228 | (-0.0020, 0.0517) | 0.0393 | 0.0228 | (-0.0084, 0.0524) | 0.0777 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0318 | (-0.0864, 0.0182) | 0.8893 | -0.0318 | (-0.1053, 0.0087) | 0.8730 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0005 | (-0.0149, 0.0140) | 0.5167 | -0.0005 | (-0.0148, 0.0158) | 0.5533 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0055 | (-0.0495, 0.0331) | 0.5930 | -0.0055 | (-0.0278, 0.0267) | 0.6383 |
| controlled_alt_vs_controlled_default | persona_style | 0.0133 | (0.0000, 0.0333) | 0.1147 | 0.0133 | (0.0000, 0.0316) | 0.0747 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0065 | (-0.0127, 0.0245) | 0.2407 | 0.0065 | (-0.0100, 0.0282) | 0.2437 |
| controlled_alt_vs_controlled_default | length_score | 0.0833 | (-0.0267, 0.2083) | 0.0763 | 0.0833 | (-0.0646, 0.2079) | 0.0823 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0350 | (-0.0350, 0.1050) | 0.2107 | 0.0350 | (0.0000, 0.0667) | 0.0797 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0069 | (-0.0252, 0.0100) | 0.7733 | -0.0069 | (-0.0262, 0.0104) | 0.7617 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1680 | (0.1183, 0.2116) | 0.0000 | 0.1680 | (0.1223, 0.1999) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1505 | (0.1011, 0.1986) | 0.0000 | 0.1505 | (0.1096, 0.1839) | 0.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 0.1249 | (0.0851, 0.1608) | 0.0000 | 0.1249 | (0.0783, 0.1674) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2182 | (0.1500, 0.2727) | 0.0000 | 0.2182 | (0.1579, 0.2597) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0510 | (0.0301, 0.0709) | 0.0000 | 0.0510 | (0.0377, 0.0653) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1798 | (0.1202, 0.2410) | 0.0000 | 0.1798 | (0.1370, 0.2202) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0333 | (0.0000, 0.0833) | 0.1230 | 0.0333 | (0.0000, 0.1176) | 0.3340 |
| controlled_alt_vs_proposed_raw | distinct1 | 0.0137 | (-0.0065, 0.0332) | 0.0863 | 0.0137 | (-0.0036, 0.0361) | 0.0727 |
| controlled_alt_vs_proposed_raw | length_score | 0.4833 | (0.3450, 0.6100) | 0.0000 | 0.4833 | (0.3228, 0.5965) | 0.0000 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.2275 | (0.1575, 0.2975) | 0.0000 | 0.2275 | (0.1474, 0.3111) | 0.0000 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1589 | (0.1206, 0.1923) | 0.0000 | 0.1589 | (0.1164, 0.1897) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2065 | (0.1846, 0.2278) | 0.0000 | 0.2065 | (0.1918, 0.2260) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1438 | (0.0958, 0.1898) | 0.0000 | 0.1438 | (0.0948, 0.1839) | 0.0000 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.1141 | (0.0698, 0.1555) | 0.0000 | 0.1141 | (0.0633, 0.1703) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.2682 | (0.2364, 0.3000) | 0.0000 | 0.2682 | (0.2468, 0.2967) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0626 | (0.0516, 0.0740) | 0.0000 | 0.0626 | (0.0562, 0.0712) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1714 | (0.1114, 0.2274) | 0.0000 | 0.1714 | (0.1167, 0.2147) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0333 | (0.0000, 0.0833) | 0.1083 | 0.0333 | (0.0000, 0.1176) | 0.3393 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0120 | (-0.0065, 0.0322) | 0.1090 | 0.0120 | (-0.0102, 0.0447) | 0.1713 |
| controlled_alt_vs_candidate_no_context | length_score | 0.4417 | (0.2967, 0.5750) | 0.0000 | 0.4417 | (0.2944, 0.5891) | 0.0000 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.2100 | (0.1050, 0.2975) | 0.0000 | 0.2100 | (0.1000, 0.3281) | 0.0010 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1720 | (0.1463, 0.1952) | 0.0000 | 0.1720 | (0.1477, 0.1902) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 7 | 4 | 9 | 0.5750 | 0.6364 |
| proposed_vs_candidate_no_context | persona_consistency | 1 | 3 | 16 | 0.4500 | 0.2500 |
| proposed_vs_candidate_no_context | naturalness | 4 | 7 | 9 | 0.4250 | 0.3636 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 6 | 4 | 10 | 0.5500 | 0.6000 |
| proposed_vs_candidate_no_context | context_overlap | 6 | 5 | 9 | 0.5250 | 0.5455 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 1 | 3 | 16 | 0.4500 | 0.2500 |
| proposed_vs_candidate_no_context | persona_style | 0 | 0 | 20 | 0.5000 | nan |
| proposed_vs_candidate_no_context | distinct1 | 6 | 4 | 10 | 0.5500 | 0.6000 |
| proposed_vs_candidate_no_context | length_score | 4 | 7 | 9 | 0.4250 | 0.3636 |
| proposed_vs_candidate_no_context | sentence_score | 3 | 4 | 13 | 0.4750 | 0.4286 |
| proposed_vs_candidate_no_context | overall_quality | 5 | 6 | 9 | 0.4750 | 0.4545 |
| controlled_vs_proposed_raw | context_relevance | 18 | 2 | 0 | 0.9000 | 0.9000 |
| controlled_vs_proposed_raw | persona_consistency | 16 | 1 | 3 | 0.8750 | 0.9412 |
| controlled_vs_proposed_raw | naturalness | 18 | 2 | 0 | 0.9000 | 0.9000 |
| controlled_vs_proposed_raw | context_keyword_coverage | 18 | 1 | 1 | 0.9250 | 0.9474 |
| controlled_vs_proposed_raw | context_overlap | 17 | 3 | 0 | 0.8500 | 0.8500 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 16 | 1 | 3 | 0.8750 | 0.9412 |
| controlled_vs_proposed_raw | persona_style | 2 | 1 | 17 | 0.5250 | 0.6667 |
| controlled_vs_proposed_raw | distinct1 | 14 | 6 | 0 | 0.7000 | 0.7000 |
| controlled_vs_proposed_raw | length_score | 18 | 2 | 0 | 0.9000 | 0.9000 |
| controlled_vs_proposed_raw | sentence_score | 12 | 1 | 7 | 0.7750 | 0.9231 |
| controlled_vs_proposed_raw | overall_quality | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_vs_candidate_no_context | context_relevance | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 16 | 1 | 3 | 0.8750 | 0.9412 |
| controlled_vs_candidate_no_context | naturalness | 15 | 5 | 0 | 0.7500 | 0.7500 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 16 | 1 | 3 | 0.8750 | 0.9412 |
| controlled_vs_candidate_no_context | persona_style | 2 | 1 | 17 | 0.5250 | 0.6667 |
| controlled_vs_candidate_no_context | distinct1 | 12 | 8 | 0 | 0.6000 | 0.6000 |
| controlled_vs_candidate_no_context | length_score | 15 | 3 | 2 | 0.8000 | 0.8333 |
| controlled_vs_candidate_no_context | sentence_score | 12 | 2 | 6 | 0.7500 | 0.8571 |
| controlled_vs_candidate_no_context | overall_quality | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 8 | 6 | 6 | 0.5500 | 0.5714 |
| controlled_alt_vs_controlled_default | persona_consistency | 3 | 2 | 15 | 0.5250 | 0.6000 |
| controlled_alt_vs_controlled_default | naturalness | 9 | 5 | 6 | 0.6000 | 0.6429 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 3 | 6 | 11 | 0.4250 | 0.3333 |
| controlled_alt_vs_controlled_default | context_overlap | 9 | 5 | 6 | 0.6000 | 0.6429 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 2 | 2 | 16 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | persona_style | 2 | 0 | 18 | 0.5500 | 1.0000 |
| controlled_alt_vs_controlled_default | distinct1 | 10 | 4 | 6 | 0.6500 | 0.7143 |
| controlled_alt_vs_controlled_default | length_score | 7 | 6 | 7 | 0.5250 | 0.5385 |
| controlled_alt_vs_controlled_default | sentence_score | 3 | 1 | 16 | 0.5500 | 0.7500 |
| controlled_alt_vs_controlled_default | overall_quality | 7 | 7 | 6 | 0.5000 | 0.5000 |
| controlled_alt_vs_proposed_raw | context_relevance | 18 | 2 | 0 | 0.9000 | 0.9000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 17 | 1 | 2 | 0.9000 | 0.9444 |
| controlled_alt_vs_proposed_raw | naturalness | 18 | 2 | 0 | 0.9000 | 0.9000 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 18 | 2 | 0 | 0.9000 | 0.9000 |
| controlled_alt_vs_proposed_raw | context_overlap | 17 | 3 | 0 | 0.8500 | 0.8500 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 17 | 1 | 2 | 0.9000 | 0.9444 |
| controlled_alt_vs_proposed_raw | persona_style | 2 | 0 | 18 | 0.5500 | 1.0000 |
| controlled_alt_vs_proposed_raw | distinct1 | 13 | 6 | 1 | 0.6750 | 0.6842 |
| controlled_alt_vs_proposed_raw | length_score | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_alt_vs_proposed_raw | sentence_score | 13 | 0 | 7 | 0.8250 | 1.0000 |
| controlled_alt_vs_proposed_raw | overall_quality | 18 | 2 | 0 | 0.9000 | 0.9000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 16 | 1 | 3 | 0.8750 | 0.9412 |
| controlled_alt_vs_candidate_no_context | naturalness | 14 | 6 | 0 | 0.7000 | 0.7000 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 16 | 1 | 3 | 0.8750 | 0.9412 |
| controlled_alt_vs_candidate_no_context | persona_style | 2 | 0 | 18 | 0.5500 | 1.0000 |
| controlled_alt_vs_candidate_no_context | distinct1 | 11 | 9 | 0 | 0.5500 | 0.5500 |
| controlled_alt_vs_candidate_no_context | length_score | 16 | 3 | 1 | 0.8250 | 0.8421 |
| controlled_alt_vs_candidate_no_context | sentence_score | 14 | 2 | 4 | 0.8000 | 0.8750 |
| controlled_alt_vs_candidate_no_context | overall_quality | 20 | 0 | 0 | 1.0000 | 1.0000 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.5000 | 0.5000 | 0.5000 |
| proposed_contextual_controlled_tuned | 0.0000 | 0.0000 | 0.5000 | 0.3000 | 0.7000 |
| proposed_contextual | 0.0000 | 0.0000 | 0.6000 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.5500 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `5`
- Unique template signatures: `17`
- Template signature ratio: `0.8500`
- Effective sample size by source clustering: `4.65`
- Effective sample size by template-signature clustering: `15.38`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: unavailable (disabled_by_flag).

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
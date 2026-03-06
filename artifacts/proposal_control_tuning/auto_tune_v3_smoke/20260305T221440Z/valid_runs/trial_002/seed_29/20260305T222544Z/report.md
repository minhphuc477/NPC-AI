# Proposal Alignment Evaluation Report

- Run ID: `20260305T222544Z`
- Generated: `2026-03-05T22:28:08.824211+00:00`
- Scenarios: `artifacts\proposal_control_tuning\auto_tune_v3_smoke\20260305T221440Z\valid_runs\trial_002\seed_29\20260305T222544Z\scenarios.jsonl`
- Scenario count: `12`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_tuned`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2706 (0.2288, 0.3204) | 0.4528 (0.3743, 0.5418) | 0.8704 (0.8230, 0.9152) | 0.4595 (0.4285, 0.4915) | n/a |
| proposed_contextual_controlled_tuned | 0.2396 (0.1729, 0.3126) | 0.4448 (0.3677, 0.5413) | 0.8786 (0.8441, 0.9130) | 0.4437 (0.4033, 0.4837) | n/a |
| proposed_contextual | 0.0899 (0.0358, 0.1559) | 0.2040 (0.1118, 0.3036) | 0.8404 (0.7953, 0.8853) | 0.2771 (0.2165, 0.3399) | n/a |
| candidate_no_context | 0.0449 (0.0166, 0.0812) | 0.2597 (0.1636, 0.3627) | 0.8829 (0.8324, 0.9271) | 0.2854 (0.2324, 0.3313) | n/a |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0451 | 1.0046 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0557 | -0.2144 |
| proposed_vs_candidate_no_context | naturalness | -0.0425 | -0.0481 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0590 | 1.3940 |
| proposed_vs_candidate_no_context | context_overlap | 0.0126 | 0.2487 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0675 | -0.3542 |
| proposed_vs_candidate_no_context | persona_style | -0.0086 | -0.0160 |
| proposed_vs_candidate_no_context | distinct1 | -0.0163 | -0.0170 |
| proposed_vs_candidate_no_context | length_score | -0.1361 | -0.2513 |
| proposed_vs_candidate_no_context | sentence_score | -0.0875 | -0.0959 |
| proposed_vs_candidate_no_context | overall_quality | -0.0083 | -0.0291 |
| controlled_vs_proposed_raw | context_relevance | 0.1806 | 2.0087 |
| controlled_vs_proposed_raw | persona_consistency | 0.2489 | 1.2200 |
| controlled_vs_proposed_raw | naturalness | 0.0300 | 0.0357 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2352 | 2.3229 |
| controlled_vs_proposed_raw | context_overlap | 0.0533 | 0.8392 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2976 | 2.4194 |
| controlled_vs_proposed_raw | persona_style | 0.0538 | 0.1019 |
| controlled_vs_proposed_raw | distinct1 | -0.0114 | -0.0121 |
| controlled_vs_proposed_raw | length_score | 0.1000 | 0.2466 |
| controlled_vs_proposed_raw | sentence_score | 0.1458 | 0.1768 |
| controlled_vs_proposed_raw | overall_quality | 0.1824 | 0.6581 |
| controlled_vs_candidate_no_context | context_relevance | 0.2257 | 5.0312 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1932 | 0.7439 |
| controlled_vs_candidate_no_context | naturalness | -0.0125 | -0.0141 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.2942 | 6.9552 |
| controlled_vs_candidate_no_context | context_overlap | 0.0659 | 1.2966 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2302 | 1.2083 |
| controlled_vs_candidate_no_context | persona_style | 0.0452 | 0.0843 |
| controlled_vs_candidate_no_context | distinct1 | -0.0277 | -0.0289 |
| controlled_vs_candidate_no_context | length_score | -0.0361 | -0.0667 |
| controlled_vs_candidate_no_context | sentence_score | 0.0583 | 0.0639 |
| controlled_vs_candidate_no_context | overall_quality | 0.1741 | 0.6098 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0310 | -0.1146 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0080 | -0.0177 |
| controlled_alt_vs_controlled_default | naturalness | 0.0082 | 0.0094 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0366 | -0.1088 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0179 | -0.1537 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0159 | -0.0377 |
| controlled_alt_vs_controlled_default | persona_style | 0.0235 | 0.0404 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0080 | 0.0086 |
| controlled_alt_vs_controlled_default | length_score | 0.0250 | 0.0495 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0000 | 0.0000 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0158 | -0.0344 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1496 | 1.6638 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.2409 | 1.1807 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0382 | 0.0455 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.1986 | 1.9613 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0353 | 0.5566 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.2817 | 2.2903 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0773 | 0.1464 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0034 | -0.0036 |
| controlled_alt_vs_proposed_raw | length_score | 0.1250 | 0.3082 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1458 | 0.1768 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1666 | 0.6010 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.1947 | 4.3399 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1852 | 0.7131 |
| controlled_alt_vs_candidate_no_context | naturalness | -0.0043 | -0.0048 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.2576 | 6.0896 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0480 | 0.9437 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.2143 | 1.1250 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0687 | 0.1281 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0197 | -0.0206 |
| controlled_alt_vs_candidate_no_context | length_score | -0.0111 | -0.0205 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.0583 | 0.0639 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1582 | 0.5544 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0451 | (-0.0325, 0.1222) | 0.1270 | 0.0451 | (0.0126, 0.0754) | 0.0000 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0557 | (-0.2226, 0.0913) | 0.7600 | -0.0557 | (-0.2029, 0.0656) | 0.7103 |
| proposed_vs_candidate_no_context | naturalness | -0.0425 | (-0.1099, 0.0266) | 0.8767 | -0.0425 | (-0.0820, 0.0901) | 0.8600 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0590 | (-0.0411, 0.1585) | 0.1347 | 0.0590 | (0.0182, 0.1000) | 0.0000 |
| proposed_vs_candidate_no_context | context_overlap | 0.0126 | (-0.0126, 0.0376) | 0.1563 | 0.0126 | (-0.0005, 0.0237) | 0.0350 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0675 | (-0.2579, 0.0992) | 0.7597 | -0.0675 | (-0.2286, 0.0667) | 0.7310 |
| proposed_vs_candidate_no_context | persona_style | -0.0086 | (-0.1537, 0.1510) | 0.5720 | -0.0086 | (-0.1000, 0.3281) | 0.6150 |
| proposed_vs_candidate_no_context | distinct1 | -0.0163 | (-0.0444, 0.0121) | 0.8693 | -0.0163 | (-0.0367, 0.0335) | 0.8540 |
| proposed_vs_candidate_no_context | length_score | -0.1361 | (-0.3889, 0.1278) | 0.8590 | -0.1361 | (-0.2667, 0.3833) | 0.7423 |
| proposed_vs_candidate_no_context | sentence_score | -0.0875 | (-0.2042, 0.0292) | 0.9617 | -0.0875 | (-0.1400, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | overall_quality | -0.0083 | (-0.1004, 0.0777) | 0.5647 | -0.0083 | (-0.0850, 0.0728) | 0.7187 |
| controlled_vs_proposed_raw | context_relevance | 0.1806 | (0.1209, 0.2390) | 0.0000 | 0.1806 | (0.1085, 0.2058) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.2489 | (0.1333, 0.3699) | 0.0000 | 0.2489 | (0.1062, 0.4114) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0300 | (-0.0196, 0.0806) | 0.1133 | 0.0300 | (-0.0205, 0.1146) | 0.0400 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2352 | (0.1645, 0.3085) | 0.0000 | 0.2352 | (0.1500, 0.2545) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0533 | (0.0170, 0.0887) | 0.0007 | 0.0533 | (0.0116, 0.0920) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2976 | (0.1645, 0.4464) | 0.0000 | 0.2976 | (0.1333, 0.5143) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.0538 | (-0.0567, 0.1906) | 0.2210 | 0.0538 | (-0.0021, 0.3281) | 0.2930 |
| controlled_vs_proposed_raw | distinct1 | -0.0114 | (-0.0303, 0.0087) | 0.8693 | -0.0114 | (-0.0313, 0.0144) | 0.8487 |
| controlled_vs_proposed_raw | length_score | 0.1000 | (-0.1028, 0.3167) | 0.1730 | 0.1000 | (-0.0400, 0.4500) | 0.0347 |
| controlled_vs_proposed_raw | sentence_score | 0.1458 | (0.0292, 0.2625) | 0.0213 | 0.1458 | (0.0000, 0.3500) | 0.0347 |
| controlled_vs_proposed_raw | overall_quality | 0.1824 | (0.1181, 0.2467) | 0.0000 | 0.1824 | (0.1221, 0.2581) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2257 | (0.1683, 0.2828) | 0.0000 | 0.2257 | (0.1838, 0.2498) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1932 | (0.0866, 0.3221) | 0.0000 | 0.1932 | (0.1492, 0.2646) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | -0.0125 | (-0.0915, 0.0643) | 0.6203 | -0.0125 | (-0.0765, 0.2047) | 0.6293 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.2942 | (0.2184, 0.3716) | 0.0000 | 0.2942 | (0.2500, 0.3333) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0659 | (0.0436, 0.0891) | 0.0000 | 0.0659 | (0.0295, 0.0915) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2302 | (0.0992, 0.3929) | 0.0000 | 0.2302 | (0.1667, 0.2857) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.0452 | (-0.1216, 0.2301) | 0.3330 | 0.0452 | (-0.1000, 0.6562) | 0.2960 |
| controlled_vs_candidate_no_context | distinct1 | -0.0277 | (-0.0534, -0.0013) | 0.9807 | -0.0277 | (-0.0471, 0.0076) | 0.9603 |
| controlled_vs_candidate_no_context | length_score | -0.0361 | (-0.3667, 0.2889) | 0.5817 | -0.0361 | (-0.2533, 0.8333) | 0.7373 |
| controlled_vs_candidate_no_context | sentence_score | 0.0583 | (-0.0583, 0.1750) | 0.2140 | 0.0583 | (-0.0700, 0.3500) | 0.2667 |
| controlled_vs_candidate_no_context | overall_quality | 0.1741 | (0.1192, 0.2345) | 0.0000 | 0.1741 | (0.1581, 0.2165) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0310 | (-0.1281, 0.0666) | 0.7377 | -0.0310 | (-0.0995, 0.1943) | 0.7303 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0080 | (-0.1140, 0.1196) | 0.5780 | -0.0080 | (-0.1271, 0.2667) | 0.5943 |
| controlled_alt_vs_controlled_default | naturalness | 0.0082 | (-0.0506, 0.0661) | 0.3827 | 0.0082 | (-0.0889, 0.0421) | 0.3693 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0366 | (-0.1601, 0.0838) | 0.7197 | -0.0366 | (-0.1333, 0.2500) | 0.7333 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0179 | (-0.0601, 0.0180) | 0.8237 | -0.0179 | (-0.0483, 0.0644) | 0.7330 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0159 | (-0.1548, 0.1468) | 0.5997 | -0.0159 | (-0.1714, 0.3333) | 0.6333 |
| controlled_alt_vs_controlled_default | persona_style | 0.0235 | (-0.0532, 0.1001) | 0.2963 | 0.0235 | (0.0000, 0.0500) | 0.0367 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0080 | (-0.0190, 0.0319) | 0.2700 | 0.0080 | (-0.0139, 0.0286) | 0.2627 |
| controlled_alt_vs_controlled_default | length_score | 0.0250 | (-0.2751, 0.3222) | 0.4223 | 0.0250 | (-0.4667, 0.1533) | 0.2440 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0000 | (-0.0875, 0.0875) | 0.6353 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0158 | (-0.0716, 0.0430) | 0.7293 | -0.0158 | (-0.0691, 0.1722) | 0.7210 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1496 | (0.0685, 0.2265) | 0.0003 | 0.1496 | (0.0849, 0.3028) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.2409 | (0.0887, 0.4034) | 0.0003 | 0.2409 | (0.1075, 0.4656) | 0.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0382 | (-0.0057, 0.0862) | 0.0490 | 0.0382 | (0.0216, 0.0599) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.1986 | (0.0896, 0.3038) | 0.0000 | 0.1986 | (0.1167, 0.4000) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0353 | (0.0080, 0.0610) | 0.0047 | 0.0353 | (0.0106, 0.0761) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.2817 | (0.1151, 0.4544) | 0.0003 | 0.2817 | (0.1333, 0.5000) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0773 | (0.0000, 0.2057) | 0.0303 | 0.0773 | (0.0042, 0.3281) | 0.0000 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0034 | (-0.0238, 0.0174) | 0.6143 | -0.0034 | (-0.0149, 0.0005) | 0.9577 |
| controlled_alt_vs_proposed_raw | length_score | 0.1250 | (-0.0444, 0.3222) | 0.0903 | 0.1250 | (-0.0167, 0.1933) | 0.0360 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1458 | (0.0583, 0.2333) | 0.0013 | 0.1458 | (0.0000, 0.3500) | 0.0440 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1666 | (0.0848, 0.2433) | 0.0000 | 0.1666 | (0.0845, 0.3158) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.1947 | (0.1198, 0.2786) | 0.0000 | 0.1947 | (0.1503, 0.3782) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1852 | (0.0615, 0.3354) | 0.0010 | 0.1852 | (0.0814, 0.5313) | 0.0000 |
| controlled_alt_vs_candidate_no_context | naturalness | -0.0043 | (-0.0641, 0.0542) | 0.5660 | -0.0043 | (-0.0344, 0.1158) | 0.6247 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.2576 | (0.1612, 0.3789) | 0.0000 | 0.2576 | (0.2000, 0.5000) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0480 | (0.0255, 0.0716) | 0.0000 | 0.0480 | (0.0343, 0.0939) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.2143 | (0.0853, 0.3651) | 0.0000 | 0.2143 | (0.1143, 0.5000) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0687 | (-0.1032, 0.2682) | 0.2477 | 0.0687 | (-0.0500, 0.6562) | 0.3093 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0197 | (-0.0362, -0.0020) | 0.9887 | -0.0197 | (-0.0362, 0.0186) | 0.9647 |
| controlled_alt_vs_candidate_no_context | length_score | -0.0111 | (-0.2528, 0.2250) | 0.5253 | -0.0111 | (-0.1000, 0.3667) | 0.7350 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.0583 | (-0.0583, 0.1750) | 0.2153 | 0.0583 | (-0.0700, 0.3500) | 0.2600 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1582 | (0.0827, 0.2388) | 0.0000 | 0.1582 | (0.1040, 0.3887) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 6 | 5 | 1 | 0.5417 | 0.5455 |
| proposed_vs_candidate_no_context | persona_consistency | 5 | 4 | 3 | 0.5417 | 0.5556 |
| proposed_vs_candidate_no_context | naturalness | 5 | 6 | 1 | 0.4583 | 0.4545 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 5 | 3 | 4 | 0.5833 | 0.6250 |
| proposed_vs_candidate_no_context | context_overlap | 7 | 4 | 1 | 0.6250 | 0.6364 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 4 | 3 | 5 | 0.5417 | 0.5714 |
| proposed_vs_candidate_no_context | persona_style | 2 | 3 | 7 | 0.4583 | 0.4000 |
| proposed_vs_candidate_no_context | distinct1 | 4 | 7 | 1 | 0.3750 | 0.3636 |
| proposed_vs_candidate_no_context | length_score | 5 | 5 | 2 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | sentence_score | 1 | 4 | 7 | 0.3750 | 0.2000 |
| proposed_vs_candidate_no_context | overall_quality | 6 | 5 | 1 | 0.5417 | 0.5455 |
| controlled_vs_proposed_raw | context_relevance | 11 | 1 | 0 | 0.9167 | 0.9167 |
| controlled_vs_proposed_raw | persona_consistency | 10 | 1 | 1 | 0.8750 | 0.9091 |
| controlled_vs_proposed_raw | naturalness | 9 | 3 | 0 | 0.7500 | 0.7500 |
| controlled_vs_proposed_raw | context_keyword_coverage | 11 | 0 | 1 | 0.9583 | 1.0000 |
| controlled_vs_proposed_raw | context_overlap | 9 | 3 | 0 | 0.7500 | 0.7500 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 10 | 0 | 2 | 0.9167 | 1.0000 |
| controlled_vs_proposed_raw | persona_style | 3 | 2 | 7 | 0.5417 | 0.6000 |
| controlled_vs_proposed_raw | distinct1 | 4 | 8 | 0 | 0.3333 | 0.3333 |
| controlled_vs_proposed_raw | length_score | 6 | 6 | 0 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | sentence_score | 6 | 1 | 5 | 0.7083 | 0.8571 |
| controlled_vs_proposed_raw | overall_quality | 11 | 1 | 0 | 0.9167 | 0.9167 |
| controlled_vs_candidate_no_context | context_relevance | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 11 | 1 | 0 | 0.9167 | 0.9167 |
| controlled_vs_candidate_no_context | naturalness | 5 | 7 | 0 | 0.4167 | 0.4167 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 11 | 0 | 1 | 0.9583 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 10 | 1 | 1 | 0.8750 | 0.9091 |
| controlled_vs_candidate_no_context | persona_style | 3 | 4 | 5 | 0.4583 | 0.4286 |
| controlled_vs_candidate_no_context | distinct1 | 4 | 8 | 0 | 0.3333 | 0.3333 |
| controlled_vs_candidate_no_context | length_score | 5 | 5 | 2 | 0.5000 | 0.5000 |
| controlled_vs_candidate_no_context | sentence_score | 3 | 1 | 8 | 0.5833 | 0.7500 |
| controlled_vs_candidate_no_context | overall_quality | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 5 | 6 | 1 | 0.4583 | 0.4545 |
| controlled_alt_vs_controlled_default | persona_consistency | 3 | 5 | 4 | 0.4167 | 0.3750 |
| controlled_alt_vs_controlled_default | naturalness | 8 | 3 | 1 | 0.7083 | 0.7273 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 4 | 6 | 2 | 0.4167 | 0.4000 |
| controlled_alt_vs_controlled_default | context_overlap | 5 | 6 | 1 | 0.4583 | 0.4545 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 2 | 3 | 7 | 0.4583 | 0.4000 |
| controlled_alt_vs_controlled_default | persona_style | 3 | 2 | 7 | 0.5417 | 0.6000 |
| controlled_alt_vs_controlled_default | distinct1 | 7 | 4 | 1 | 0.6250 | 0.6364 |
| controlled_alt_vs_controlled_default | length_score | 6 | 4 | 2 | 0.5833 | 0.6000 |
| controlled_alt_vs_controlled_default | sentence_score | 1 | 1 | 10 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | overall_quality | 4 | 7 | 1 | 0.3750 | 0.3636 |
| controlled_alt_vs_proposed_raw | context_relevance | 10 | 2 | 0 | 0.8333 | 0.8333 |
| controlled_alt_vs_proposed_raw | persona_consistency | 10 | 1 | 1 | 0.8750 | 0.9091 |
| controlled_alt_vs_proposed_raw | naturalness | 8 | 4 | 0 | 0.6667 | 0.6667 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 9 | 1 | 2 | 0.8333 | 0.9000 |
| controlled_alt_vs_proposed_raw | context_overlap | 9 | 2 | 1 | 0.7917 | 0.8182 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 10 | 1 | 1 | 0.8750 | 0.9091 |
| controlled_alt_vs_proposed_raw | persona_style | 3 | 0 | 9 | 0.6250 | 1.0000 |
| controlled_alt_vs_proposed_raw | distinct1 | 5 | 7 | 0 | 0.4167 | 0.4167 |
| controlled_alt_vs_proposed_raw | length_score | 6 | 5 | 1 | 0.5417 | 0.5455 |
| controlled_alt_vs_proposed_raw | sentence_score | 5 | 0 | 7 | 0.7083 | 1.0000 |
| controlled_alt_vs_proposed_raw | overall_quality | 9 | 3 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_candidate_no_context | context_relevance | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 10 | 1 | 1 | 0.8750 | 0.9091 |
| controlled_alt_vs_candidate_no_context | naturalness | 6 | 6 | 0 | 0.5000 | 0.5000 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 11 | 1 | 0 | 0.9167 | 0.9167 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 9 | 1 | 2 | 0.8333 | 0.9000 |
| controlled_alt_vs_candidate_no_context | persona_style | 4 | 3 | 5 | 0.5417 | 0.5714 |
| controlled_alt_vs_candidate_no_context | distinct1 | 3 | 9 | 0 | 0.2500 | 0.2500 |
| controlled_alt_vs_candidate_no_context | length_score | 8 | 4 | 0 | 0.6667 | 0.6667 |
| controlled_alt_vs_candidate_no_context | sentence_score | 3 | 1 | 8 | 0.5833 | 0.7500 |
| controlled_alt_vs_candidate_no_context | overall_quality | 11 | 1 | 0 | 0.9167 | 0.9167 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.2500 | 0.4167 | 0.5833 |
| proposed_contextual_controlled_tuned | 0.0000 | 0.0000 | 0.0833 | 0.7500 | 0.2500 |
| proposed_contextual | 0.0000 | 0.0000 | 0.4167 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.2500 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `3`
- Unique template signatures: `10`
- Template signature ratio: `0.8333`
- Effective sample size by source clustering: `2.67`
- Effective sample size by template-signature clustering: `8.00`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: unavailable (disabled_by_flag).

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
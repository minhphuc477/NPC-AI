# Proposal Alignment Evaluation Report

- Run ID: `20260305T212443Z`
- Generated: `2026-03-05T21:29:49.798766+00:00`
- Scenarios: `artifacts\proposal_control_tuning\auto_tune_v2b\20260305T210504Z\valid_runs\trial_002\seed_29\20260305T212443Z\scenarios.jsonl`
- Scenario count: `20`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_tuned`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2706 (0.2258, 0.3255) | 0.5099 (0.4330, 0.5950) | 0.8674 (0.8443, 0.8907) | 0.4797 (0.4496, 0.5132) | n/a |
| proposed_contextual_controlled_tuned | 0.2614 (0.2369, 0.2902) | 0.4415 (0.3790, 0.5017) | 0.8935 (0.8635, 0.9200) | 0.4552 (0.4303, 0.4812) | n/a |
| proposed_contextual | 0.1065 (0.0600, 0.1664) | 0.2219 (0.1469, 0.3058) | 0.8493 (0.8173, 0.8813) | 0.2942 (0.2442, 0.3493) | n/a |
| candidate_no_context | 0.0328 (0.0129, 0.0547) | 0.2753 (0.1869, 0.3691) | 0.8657 (0.8308, 0.8958) | 0.2827 (0.2432, 0.3238) | n/a |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0737 | 2.2489 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0534 | -0.1938 |
| proposed_vs_candidate_no_context | naturalness | -0.0164 | -0.0190 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0966 | 3.7500 |
| proposed_vs_candidate_no_context | context_overlap | 0.0204 | 0.4153 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0595 | -0.2717 |
| proposed_vs_candidate_no_context | persona_style | -0.0287 | -0.0574 |
| proposed_vs_candidate_no_context | distinct1 | -0.0013 | -0.0013 |
| proposed_vs_candidate_no_context | length_score | -0.0533 | -0.1131 |
| proposed_vs_candidate_no_context | sentence_score | -0.0525 | -0.0587 |
| proposed_vs_candidate_no_context | overall_quality | 0.0114 | 0.0404 |
| controlled_vs_proposed_raw | context_relevance | 0.1641 | 1.5404 |
| controlled_vs_proposed_raw | persona_consistency | 0.2879 | 1.2973 |
| controlled_vs_proposed_raw | naturalness | 0.0181 | 0.0214 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2169 | 1.7728 |
| controlled_vs_proposed_raw | context_overlap | 0.0409 | 0.5878 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.3179 | 1.9925 |
| controlled_vs_proposed_raw | persona_style | 0.1681 | 0.3565 |
| controlled_vs_proposed_raw | distinct1 | -0.0186 | -0.0195 |
| controlled_vs_proposed_raw | length_score | 0.0667 | 0.1594 |
| controlled_vs_proposed_raw | sentence_score | 0.1225 | 0.1454 |
| controlled_vs_proposed_raw | overall_quality | 0.1855 | 0.6306 |
| controlled_vs_candidate_no_context | context_relevance | 0.2379 | 7.2536 |
| controlled_vs_candidate_no_context | persona_consistency | 0.2346 | 0.8520 |
| controlled_vs_candidate_no_context | naturalness | 0.0017 | 0.0020 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3135 | 12.1706 |
| controlled_vs_candidate_no_context | context_overlap | 0.0614 | 1.2473 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2583 | 1.1793 |
| controlled_vs_candidate_no_context | persona_style | 0.1394 | 0.2787 |
| controlled_vs_candidate_no_context | distinct1 | -0.0199 | -0.0208 |
| controlled_vs_candidate_no_context | length_score | 0.0133 | 0.0283 |
| controlled_vs_candidate_no_context | sentence_score | 0.0700 | 0.0782 |
| controlled_vs_candidate_no_context | overall_quality | 0.1969 | 0.6964 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0093 | -0.0342 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0683 | -0.1340 |
| controlled_alt_vs_controlled_default | naturalness | 0.0260 | 0.0300 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0142 | -0.0420 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0024 | 0.0216 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0786 | -0.1646 |
| controlled_alt_vs_controlled_default | persona_style | -0.0272 | -0.0426 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0052 | -0.0055 |
| controlled_alt_vs_controlled_default | length_score | 0.1317 | 0.2715 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0175 | 0.0181 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0245 | -0.0510 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1549 | 1.4535 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.2196 | 0.9895 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0442 | 0.0520 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2027 | 1.6563 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0433 | 0.6221 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.2393 | 1.5000 |
| controlled_alt_vs_proposed_raw | persona_style | 0.1409 | 0.2988 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0238 | -0.0249 |
| controlled_alt_vs_proposed_raw | length_score | 0.1983 | 0.4741 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1400 | 0.1662 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1610 | 0.5474 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2286 | 6.9714 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1662 | 0.6039 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0277 | 0.0320 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.2992 | 11.6176 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0638 | 1.2959 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1798 | 0.8207 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.1122 | 0.2242 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0250 | -0.0262 |
| controlled_alt_vs_candidate_no_context | length_score | 0.1450 | 0.3074 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.0875 | 0.0978 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1724 | 0.6099 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0737 | (0.0200, 0.1378) | 0.0033 | 0.0737 | (0.0490, 0.0888) | 0.0000 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0534 | (-0.1581, 0.0554) | 0.8337 | -0.0534 | (-0.1045, -0.0092) | 1.0000 |
| proposed_vs_candidate_no_context | naturalness | -0.0164 | (-0.0498, 0.0140) | 0.8450 | -0.0164 | (-0.0551, 0.0121) | 0.7027 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0966 | (0.0254, 0.1830) | 0.0030 | 0.0966 | (0.0682, 0.1136) | 0.0000 |
| proposed_vs_candidate_no_context | context_overlap | 0.0204 | (-0.0020, 0.0429) | 0.0357 | 0.0204 | (0.0041, 0.0307) | 0.0000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0595 | (-0.1845, 0.0619) | 0.8393 | -0.0595 | (-0.1071, -0.0208) | 1.0000 |
| proposed_vs_candidate_no_context | persona_style | -0.0287 | (-0.1386, 0.0833) | 0.6957 | -0.0287 | (-0.0938, 0.0376) | 0.7393 |
| proposed_vs_candidate_no_context | distinct1 | -0.0013 | (-0.0196, 0.0159) | 0.5480 | -0.0013 | (-0.0133, 0.0094) | 0.7100 |
| proposed_vs_candidate_no_context | length_score | -0.0533 | (-0.1684, 0.0533) | 0.8220 | -0.0533 | (-0.1833, 0.0417) | 0.7100 |
| proposed_vs_candidate_no_context | sentence_score | -0.0525 | (-0.1400, 0.0350) | 0.9210 | -0.0525 | (-0.1313, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | overall_quality | 0.0114 | (-0.0483, 0.0803) | 0.3933 | 0.0114 | (-0.0077, 0.0307) | 0.1497 |
| controlled_vs_proposed_raw | context_relevance | 0.1641 | (0.0986, 0.2234) | 0.0000 | 0.1641 | (0.0818, 0.2634) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.2879 | (0.1749, 0.4026) | 0.0000 | 0.2879 | (0.2359, 0.3714) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0181 | (-0.0266, 0.0649) | 0.2220 | 0.0181 | (-0.0452, 0.1092) | 0.2520 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2169 | (0.1276, 0.3005) | 0.0000 | 0.2169 | (0.1023, 0.3591) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0409 | (0.0200, 0.0602) | 0.0000 | 0.0409 | (0.0340, 0.0482) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.3179 | (0.1833, 0.4584) | 0.0000 | 0.3179 | (0.2708, 0.3571) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.1681 | (0.0635, 0.2834) | 0.0000 | 0.1681 | (0.0625, 0.5234) | 0.0000 |
| controlled_vs_proposed_raw | distinct1 | -0.0186 | (-0.0349, -0.0002) | 0.9770 | -0.0186 | (-0.0386, 0.0083) | 0.9577 |
| controlled_vs_proposed_raw | length_score | 0.0667 | (-0.1200, 0.2617) | 0.2463 | 0.0667 | (-0.1708, 0.4417) | 0.2520 |
| controlled_vs_proposed_raw | sentence_score | 0.1225 | (0.0525, 0.1925) | 0.0003 | 0.1225 | (0.0437, 0.1750) | 0.0000 |
| controlled_vs_proposed_raw | overall_quality | 0.1855 | (0.1170, 0.2492) | 0.0000 | 0.1855 | (0.1557, 0.2768) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2379 | (0.1833, 0.2961) | 0.0000 | 0.2379 | (0.1706, 0.3124) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.2346 | (0.1377, 0.3387) | 0.0000 | 0.2346 | (0.1938, 0.3318) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.0017 | (-0.0459, 0.0531) | 0.4913 | 0.0017 | (-0.0373, 0.1213) | 0.4123 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3135 | (0.2406, 0.3938) | 0.0000 | 0.3135 | (0.2159, 0.4273) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0614 | (0.0457, 0.0762) | 0.0000 | 0.0614 | (0.0444, 0.0665) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2583 | (0.1607, 0.3714) | 0.0000 | 0.2583 | (0.2500, 0.2917) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.1394 | (0.0136, 0.2721) | 0.0123 | 0.1394 | (-0.0312, 0.4922) | 0.0340 |
| controlled_vs_candidate_no_context | distinct1 | -0.0199 | (-0.0407, 0.0012) | 0.9687 | -0.0199 | (-0.0332, 0.0178) | 0.9603 |
| controlled_vs_candidate_no_context | length_score | 0.0133 | (-0.1817, 0.2133) | 0.4487 | 0.0133 | (-0.1417, 0.4833) | 0.3953 |
| controlled_vs_candidate_no_context | sentence_score | 0.0700 | (0.0175, 0.1400) | 0.0120 | 0.0700 | (0.0437, 0.1750) | 0.0000 |
| controlled_vs_candidate_no_context | overall_quality | 0.1969 | (0.1448, 0.2516) | 0.0000 | 0.1969 | (0.1480, 0.2879) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0093 | (-0.0652, 0.0443) | 0.6187 | -0.0093 | (-0.1026, 0.0460) | 0.6180 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0683 | (-0.1640, 0.0274) | 0.9150 | -0.0683 | (-0.1759, 0.0551) | 0.8533 |
| controlled_alt_vs_controlled_default | naturalness | 0.0260 | (-0.0171, 0.0641) | 0.1020 | 0.0260 | (0.0069, 0.0360) | 0.0000 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0142 | (-0.0883, 0.0545) | 0.6400 | -0.0142 | (-0.1432, 0.0568) | 0.6480 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0024 | (-0.0144, 0.0215) | 0.4207 | 0.0024 | (-0.0110, 0.0209) | 0.4137 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0786 | (-0.1965, 0.0274) | 0.9217 | -0.0786 | (-0.1964, 0.0625) | 0.8523 |
| controlled_alt_vs_controlled_default | persona_style | -0.0272 | (-0.1000, 0.0344) | 0.7903 | -0.0272 | (-0.0938, 0.0256) | 0.7483 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0052 | (-0.0262, 0.0165) | 0.6703 | -0.0052 | (-0.0174, 0.0130) | 0.7367 |
| controlled_alt_vs_controlled_default | length_score | 0.1317 | (-0.0551, 0.3033) | 0.0783 | 0.1317 | (0.0250, 0.1625) | 0.0000 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0175 | (0.0000, 0.0525) | 0.3617 | 0.0175 | (0.0000, 0.0875) | 0.2977 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0245 | (-0.0619, 0.0079) | 0.9257 | -0.0245 | (-0.0845, 0.0194) | 0.8490 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1549 | (0.0943, 0.2069) | 0.0000 | 0.1549 | (0.1278, 0.1789) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.2196 | (0.1173, 0.3191) | 0.0000 | 0.2196 | (0.1223, 0.2910) | 0.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0442 | (0.0015, 0.0861) | 0.0207 | 0.0442 | (-0.0092, 0.1161) | 0.0377 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2027 | (0.1192, 0.2724) | 0.0000 | 0.2027 | (0.1591, 0.2396) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0433 | (0.0203, 0.0653) | 0.0007 | 0.0433 | (0.0324, 0.0549) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.2393 | (0.1226, 0.3548) | 0.0000 | 0.2393 | (0.1607, 0.3333) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 0.1409 | (0.0362, 0.2634) | 0.0037 | 0.1409 | (-0.0312, 0.5234) | 0.0383 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0238 | (-0.0438, -0.0047) | 0.9940 | -0.0238 | (-0.0294, -0.0087) | 1.0000 |
| controlled_alt_vs_proposed_raw | length_score | 0.1983 | (0.0300, 0.3650) | 0.0107 | 0.1983 | (-0.0167, 0.4667) | 0.0373 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1400 | (0.0700, 0.2275) | 0.0000 | 0.1400 | (0.0437, 0.2625) | 0.0000 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1610 | (0.1013, 0.2194) | 0.0000 | 0.1610 | (0.1174, 0.1923) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2286 | (0.1943, 0.2661) | 0.0000 | 0.2286 | (0.2098, 0.2500) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1662 | (0.0365, 0.2899) | 0.0083 | 0.1662 | (0.0179, 0.2819) | 0.0000 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0277 | (-0.0126, 0.0675) | 0.0777 | 0.0277 | (-0.0012, 0.1282) | 0.0373 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.2992 | (0.2521, 0.3502) | 0.0000 | 0.2992 | (0.2727, 0.3333) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0638 | (0.0455, 0.0850) | 0.0000 | 0.0638 | (0.0366, 0.0856) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1798 | (0.0452, 0.3155) | 0.0043 | 0.1798 | (0.0536, 0.3125) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.1122 | (-0.0194, 0.2630) | 0.0483 | 0.1122 | (-0.1250, 0.4922) | 0.2647 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0250 | (-0.0402, -0.0090) | 0.9990 | -0.0250 | (-0.0427, 0.0007) | 0.9647 |
| controlled_alt_vs_candidate_no_context | length_score | 0.1450 | (-0.0300, 0.3067) | 0.0517 | 0.1450 | (0.0125, 0.5083) | 0.0000 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.0875 | (0.0350, 0.1575) | 0.0027 | 0.0875 | (0.0437, 0.2625) | 0.0000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1724 | (0.1183, 0.2276) | 0.0000 | 0.1724 | (0.1097, 0.2197) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 12 | 4 | 4 | 0.7000 | 0.7500 |
| proposed_vs_candidate_no_context | persona_consistency | 5 | 11 | 4 | 0.3500 | 0.3125 |
| proposed_vs_candidate_no_context | naturalness | 8 | 8 | 4 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 10 | 2 | 8 | 0.7000 | 0.8333 |
| proposed_vs_candidate_no_context | context_overlap | 12 | 4 | 4 | 0.7000 | 0.7500 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 2 | 7 | 11 | 0.3750 | 0.2222 |
| proposed_vs_candidate_no_context | persona_style | 5 | 8 | 7 | 0.4250 | 0.3846 |
| proposed_vs_candidate_no_context | distinct1 | 9 | 6 | 5 | 0.5750 | 0.6000 |
| proposed_vs_candidate_no_context | length_score | 8 | 8 | 4 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | sentence_score | 2 | 5 | 13 | 0.4250 | 0.2857 |
| proposed_vs_candidate_no_context | overall_quality | 7 | 9 | 4 | 0.4500 | 0.4375 |
| controlled_vs_proposed_raw | context_relevance | 17 | 3 | 0 | 0.8500 | 0.8500 |
| controlled_vs_proposed_raw | persona_consistency | 17 | 2 | 1 | 0.8750 | 0.8947 |
| controlled_vs_proposed_raw | naturalness | 10 | 10 | 0 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | context_keyword_coverage | 16 | 2 | 2 | 0.8500 | 0.8889 |
| controlled_vs_proposed_raw | context_overlap | 17 | 3 | 0 | 0.8500 | 0.8500 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 17 | 2 | 1 | 0.8750 | 0.8947 |
| controlled_vs_proposed_raw | persona_style | 10 | 1 | 9 | 0.7250 | 0.9091 |
| controlled_vs_proposed_raw | distinct1 | 6 | 14 | 0 | 0.3000 | 0.3000 |
| controlled_vs_proposed_raw | length_score | 12 | 8 | 0 | 0.6000 | 0.6000 |
| controlled_vs_proposed_raw | sentence_score | 7 | 0 | 13 | 0.6750 | 1.0000 |
| controlled_vs_proposed_raw | overall_quality | 18 | 2 | 0 | 0.9000 | 0.9000 |
| controlled_vs_candidate_no_context | context_relevance | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 14 | 3 | 3 | 0.7750 | 0.8235 |
| controlled_vs_candidate_no_context | naturalness | 7 | 13 | 0 | 0.3500 | 0.3500 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 19 | 0 | 1 | 0.9750 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 14 | 0 | 6 | 0.8500 | 1.0000 |
| controlled_vs_candidate_no_context | persona_style | 9 | 5 | 6 | 0.6000 | 0.6429 |
| controlled_vs_candidate_no_context | distinct1 | 7 | 13 | 0 | 0.3500 | 0.3500 |
| controlled_vs_candidate_no_context | length_score | 7 | 12 | 1 | 0.3750 | 0.3684 |
| controlled_vs_candidate_no_context | sentence_score | 4 | 0 | 16 | 0.6000 | 1.0000 |
| controlled_vs_candidate_no_context | overall_quality | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 10 | 8 | 2 | 0.5500 | 0.5556 |
| controlled_alt_vs_controlled_default | persona_consistency | 6 | 9 | 5 | 0.4250 | 0.4000 |
| controlled_alt_vs_controlled_default | naturalness | 12 | 6 | 2 | 0.6500 | 0.6667 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 9 | 6 | 5 | 0.5750 | 0.6000 |
| controlled_alt_vs_controlled_default | context_overlap | 8 | 10 | 2 | 0.4500 | 0.4444 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 6 | 9 | 5 | 0.4250 | 0.4000 |
| controlled_alt_vs_controlled_default | persona_style | 3 | 3 | 14 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | distinct1 | 8 | 10 | 2 | 0.4500 | 0.4444 |
| controlled_alt_vs_controlled_default | length_score | 14 | 3 | 3 | 0.7750 | 0.8235 |
| controlled_alt_vs_controlled_default | sentence_score | 1 | 0 | 19 | 0.5250 | 1.0000 |
| controlled_alt_vs_controlled_default | overall_quality | 8 | 10 | 2 | 0.4500 | 0.4444 |
| controlled_alt_vs_proposed_raw | context_relevance | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_alt_vs_proposed_raw | persona_consistency | 15 | 3 | 2 | 0.8000 | 0.8333 |
| controlled_alt_vs_proposed_raw | naturalness | 16 | 4 | 0 | 0.8000 | 0.8000 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 18 | 1 | 1 | 0.9250 | 0.9474 |
| controlled_alt_vs_proposed_raw | context_overlap | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 15 | 3 | 2 | 0.8000 | 0.8333 |
| controlled_alt_vs_proposed_raw | persona_style | 7 | 1 | 12 | 0.6500 | 0.8750 |
| controlled_alt_vs_proposed_raw | distinct1 | 7 | 13 | 0 | 0.3500 | 0.3500 |
| controlled_alt_vs_proposed_raw | length_score | 15 | 5 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_proposed_raw | sentence_score | 8 | 0 | 12 | 0.7000 | 1.0000 |
| controlled_alt_vs_proposed_raw | overall_quality | 17 | 3 | 0 | 0.8500 | 0.8500 |
| controlled_alt_vs_candidate_no_context | context_relevance | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 13 | 5 | 2 | 0.7000 | 0.7222 |
| controlled_alt_vs_candidate_no_context | naturalness | 12 | 7 | 1 | 0.6250 | 0.6316 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 13 | 4 | 3 | 0.7250 | 0.7647 |
| controlled_alt_vs_candidate_no_context | persona_style | 8 | 5 | 7 | 0.5750 | 0.6154 |
| controlled_alt_vs_candidate_no_context | distinct1 | 6 | 13 | 1 | 0.3250 | 0.3158 |
| controlled_alt_vs_candidate_no_context | length_score | 12 | 7 | 1 | 0.6250 | 0.6316 |
| controlled_alt_vs_candidate_no_context | sentence_score | 5 | 0 | 15 | 0.6250 | 1.0000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 18 | 2 | 0 | 0.9000 | 0.9000 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.1500 | 0.5500 | 0.4500 |
| proposed_contextual_controlled_tuned | 0.0000 | 0.0000 | 0.6000 | 0.1500 | 0.8500 |
| proposed_contextual | 0.0000 | 0.0000 | 0.3000 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.2500 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `3`
- Unique template signatures: `14`
- Template signature ratio: `0.7000`
- Effective sample size by source clustering: `2.78`
- Effective sample size by template-signature clustering: `10.53`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: unavailable (disabled_by_flag).

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
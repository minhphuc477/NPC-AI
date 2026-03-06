# Proposal Alignment Evaluation Report

- Run ID: `20260306T000633Z`
- Generated: `2026-03-06T00:09:35.615167+00:00`
- Scenarios: `artifacts\proposal_control_tuning\auto_tune_v4\20260305T235104Z\valid_runs\trial_002\seed_31\20260306T000633Z\scenarios.jsonl`
- Scenario count: `16`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_tuned`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.3230 (0.2894, 0.3588) | 0.3183 (0.2774, 0.3615) | 0.8833 (0.8567, 0.9084) | 0.4361 (0.4173, 0.4550) | n/a |
| proposed_contextual_controlled_tuned | 0.2617 (0.2126, 0.3121) | 0.2907 (0.2486, 0.3389) | 0.8804 (0.8474, 0.9134) | 0.3966 (0.3738, 0.4175) | n/a |
| proposed_contextual | 0.1321 (0.0750, 0.1933) | 0.1723 (0.1187, 0.2218) | 0.8885 (0.8495, 0.9237) | 0.2936 (0.2527, 0.3342) | n/a |
| candidate_no_context | 0.0274 (0.0103, 0.0515) | 0.1052 (0.0601, 0.1552) | 0.8432 (0.8018, 0.8836) | 0.2130 (0.1851, 0.2407) | n/a |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.1047 | 3.8246 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0670 | 0.6372 |
| proposed_vs_candidate_no_context | naturalness | 0.0453 | 0.0537 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.1302 | 6.2500 |
| proposed_vs_candidate_no_context | context_overlap | 0.0452 | 1.0601 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0563 | 2.7000 |
| proposed_vs_candidate_no_context | persona_style | 0.1102 | 0.2489 |
| proposed_vs_candidate_no_context | distinct1 | 0.0090 | 0.0095 |
| proposed_vs_candidate_no_context | length_score | 0.1646 | 0.3950 |
| proposed_vs_candidate_no_context | sentence_score | 0.0875 | 0.1089 |
| proposed_vs_candidate_no_context | overall_quality | 0.0807 | 0.3787 |
| controlled_vs_proposed_raw | context_relevance | 0.1909 | 1.4451 |
| controlled_vs_proposed_raw | persona_consistency | 0.1461 | 0.8480 |
| controlled_vs_proposed_raw | naturalness | -0.0052 | -0.0059 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2590 | 1.7147 |
| controlled_vs_proposed_raw | context_overlap | 0.0320 | 0.3637 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1708 | 2.2162 |
| controlled_vs_proposed_raw | persona_style | 0.0471 | 0.0851 |
| controlled_vs_proposed_raw | distinct1 | -0.0136 | -0.0142 |
| controlled_vs_proposed_raw | length_score | 0.0229 | 0.0394 |
| controlled_vs_proposed_raw | sentence_score | -0.0437 | -0.0491 |
| controlled_vs_proposed_raw | overall_quality | 0.1425 | 0.4852 |
| controlled_vs_candidate_no_context | context_relevance | 0.2956 | 10.7968 |
| controlled_vs_candidate_no_context | persona_consistency | 0.2131 | 2.0255 |
| controlled_vs_candidate_no_context | naturalness | 0.0401 | 0.0475 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3892 | 18.6818 |
| controlled_vs_candidate_no_context | context_overlap | 0.0772 | 1.8094 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2271 | 10.9000 |
| controlled_vs_candidate_no_context | persona_style | 0.1573 | 0.3553 |
| controlled_vs_candidate_no_context | distinct1 | -0.0045 | -0.0048 |
| controlled_vs_candidate_no_context | length_score | 0.1875 | 0.4500 |
| controlled_vs_candidate_no_context | sentence_score | 0.0438 | 0.0545 |
| controlled_vs_candidate_no_context | overall_quality | 0.2231 | 1.0476 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0612 | -0.1896 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0277 | -0.0869 |
| controlled_alt_vs_controlled_default | naturalness | -0.0029 | -0.0033 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0857 | -0.2090 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0042 | -0.0350 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0292 | -0.1176 |
| controlled_alt_vs_controlled_default | persona_style | -0.0217 | -0.0361 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0058 | 0.0061 |
| controlled_alt_vs_controlled_default | length_score | -0.0479 | -0.0793 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0437 | 0.0517 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0395 | -0.0907 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1296 | 0.9814 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1184 | 0.6874 |
| controlled_alt_vs_proposed_raw | naturalness | -0.0081 | -0.0091 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.1733 | 1.1473 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0278 | 0.3160 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1417 | 1.8378 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0254 | 0.0460 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0078 | -0.0081 |
| controlled_alt_vs_proposed_raw | length_score | -0.0250 | -0.0430 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0000 | 0.0000 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1029 | 0.3505 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2343 | 8.5597 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1855 | 1.7626 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0372 | 0.0441 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3035 | 14.5682 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0730 | 1.7111 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1979 | 9.5000 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.1356 | 0.3063 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0013 | 0.0013 |
| controlled_alt_vs_candidate_no_context | length_score | 0.1396 | 0.3350 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.0875 | 0.1089 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1836 | 0.8620 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.1047 | (0.0414, 0.1726) | 0.0000 | 0.1047 | (0.0133, 0.2036) | 0.0000 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0670 | (0.0078, 0.1283) | 0.0160 | 0.0670 | (0.0310, 0.0973) | 0.0000 |
| proposed_vs_candidate_no_context | naturalness | 0.0453 | (-0.0003, 0.0905) | 0.0257 | 0.0453 | (0.0030, 0.0934) | 0.0000 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.1302 | (0.0507, 0.2135) | 0.0000 | 0.1302 | (0.0139, 0.2545) | 0.0000 |
| proposed_vs_candidate_no_context | context_overlap | 0.0452 | (0.0164, 0.0761) | 0.0007 | 0.0452 | (0.0119, 0.0848) | 0.0000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0563 | (-0.0083, 0.1229) | 0.0443 | 0.0563 | (0.0000, 0.1000) | 0.0460 |
| proposed_vs_candidate_no_context | persona_style | 0.1102 | (-0.0044, 0.2562) | 0.0323 | 0.1102 | (0.0000, 0.1667) | 0.0340 |
| proposed_vs_candidate_no_context | distinct1 | 0.0090 | (-0.0139, 0.0325) | 0.2303 | 0.0090 | (-0.0259, 0.0452) | 0.3630 |
| proposed_vs_candidate_no_context | length_score | 0.1646 | (-0.0167, 0.3542) | 0.0393 | 0.1646 | (0.0667, 0.3067) | 0.0000 |
| proposed_vs_candidate_no_context | sentence_score | 0.0875 | (0.0000, 0.1750) | 0.0597 | 0.0875 | (0.0000, 0.1400) | 0.0350 |
| proposed_vs_candidate_no_context | overall_quality | 0.0807 | (0.0348, 0.1262) | 0.0007 | 0.0807 | (0.0158, 0.1333) | 0.0000 |
| controlled_vs_proposed_raw | context_relevance | 0.1909 | (0.1165, 0.2640) | 0.0000 | 0.1909 | (0.1174, 0.2992) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.1461 | (0.0998, 0.1874) | 0.0000 | 0.1461 | (0.0800, 0.2440) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | -0.0052 | (-0.0492, 0.0431) | 0.6020 | -0.0052 | (-0.0446, 0.0529) | 0.6303 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2590 | (0.1652, 0.3504) | 0.0000 | 0.2590 | (0.1636, 0.4028) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0320 | (0.0004, 0.0597) | 0.0250 | 0.0320 | (0.0095, 0.0575) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1708 | (0.1208, 0.2250) | 0.0000 | 0.1708 | (0.1000, 0.2800) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.0471 | (-0.0511, 0.1547) | 0.1857 | 0.0471 | (0.0000, 0.1000) | 0.0337 |
| controlled_vs_proposed_raw | distinct1 | -0.0136 | (-0.0349, 0.0104) | 0.8817 | -0.0136 | (-0.0409, 0.0181) | 0.8473 |
| controlled_vs_proposed_raw | length_score | 0.0229 | (-0.1854, 0.2229) | 0.4310 | 0.0229 | (-0.1889, 0.1933) | 0.3683 |
| controlled_vs_proposed_raw | sentence_score | -0.0437 | (-0.1531, 0.0656) | 0.8167 | -0.0437 | (-0.2100, 0.0700) | 0.7530 |
| controlled_vs_proposed_raw | overall_quality | 0.1425 | (0.1076, 0.1777) | 0.0000 | 0.1425 | (0.0821, 0.1752) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2956 | (0.2647, 0.3245) | 0.0000 | 0.2956 | (0.2499, 0.3210) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.2131 | (0.1501, 0.2905) | 0.0000 | 0.2131 | (0.1506, 0.3413) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.0401 | (-0.0120, 0.0913) | 0.0647 | 0.0401 | (-0.0417, 0.1463) | 0.1437 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3892 | (0.3475, 0.4271) | 0.0000 | 0.3892 | (0.3273, 0.4182) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0772 | (0.0657, 0.0906) | 0.0000 | 0.0772 | (0.0693, 0.0943) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2271 | (0.1562, 0.3104) | 0.0000 | 0.2271 | (0.1389, 0.3600) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.1573 | (0.0481, 0.2963) | 0.0000 | 0.1573 | (0.0000, 0.2667) | 0.0407 |
| controlled_vs_candidate_no_context | distinct1 | -0.0045 | (-0.0329, 0.0248) | 0.6330 | -0.0045 | (-0.0431, 0.0633) | 0.6223 |
| controlled_vs_candidate_no_context | length_score | 0.1875 | (-0.0209, 0.3938) | 0.0397 | 0.1875 | (-0.1222, 0.5000) | 0.1523 |
| controlled_vs_candidate_no_context | sentence_score | 0.0437 | (-0.0656, 0.1531) | 0.2557 | 0.0437 | (-0.0700, 0.2100) | 0.3017 |
| controlled_vs_candidate_no_context | overall_quality | 0.2231 | (0.1953, 0.2514) | 0.0000 | 0.2231 | (0.1910, 0.2694) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0612 | (-0.1344, 0.0129) | 0.9460 | -0.0612 | (-0.1095, 0.0427) | 0.9623 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0277 | (-0.0866, 0.0252) | 0.8153 | -0.0277 | (-0.1413, 0.0440) | 0.7420 |
| controlled_alt_vs_controlled_default | naturalness | -0.0029 | (-0.0470, 0.0382) | 0.5343 | -0.0029 | (-0.0244, 0.0126) | 0.6020 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0857 | (-0.1813, 0.0081) | 0.9600 | -0.0857 | (-0.1528, 0.0545) | 0.9667 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0042 | (-0.0301, 0.0225) | 0.6377 | -0.0042 | (-0.0255, 0.0152) | 0.6610 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0292 | (-0.1146, 0.0312) | 0.7813 | -0.0292 | (-0.1600, 0.0556) | 0.7533 |
| controlled_alt_vs_controlled_default | persona_style | -0.0217 | (-0.1051, 0.0473) | 0.6830 | -0.0217 | (-0.0667, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0058 | (-0.0142, 0.0238) | 0.2670 | 0.0058 | (-0.0218, 0.0287) | 0.3737 |
| controlled_alt_vs_controlled_default | length_score | -0.0479 | (-0.2146, 0.1042) | 0.7247 | -0.0479 | (-0.1133, 0.0056) | 0.9643 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0437 | (-0.0875, 0.1531) | 0.3063 | 0.0437 | (0.0000, 0.0700) | 0.0470 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0395 | (-0.0648, -0.0155) | 0.9997 | -0.0395 | (-0.0510, -0.0319) | 1.0000 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1296 | (0.0743, 0.1853) | 0.0000 | 0.1296 | (0.0079, 0.1915) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1184 | (0.0762, 0.1598) | 0.0000 | 0.1184 | (0.0800, 0.1636) | 0.0000 |
| controlled_alt_vs_proposed_raw | naturalness | -0.0081 | (-0.0534, 0.0411) | 0.6457 | -0.0081 | (-0.0320, 0.0285) | 0.7420 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.1733 | (0.0985, 0.2500) | 0.0000 | 0.1733 | (0.0182, 0.2500) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0278 | (0.0045, 0.0496) | 0.0073 | 0.0278 | (-0.0160, 0.0550) | 0.0343 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1417 | (0.0979, 0.1854) | 0.0000 | 0.1417 | (0.1000, 0.1944) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0254 | (-0.0285, 0.0852) | 0.1950 | 0.0254 | (0.0000, 0.0400) | 0.0370 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0078 | (-0.0257, 0.0123) | 0.7923 | -0.0078 | (-0.0350, 0.0116) | 0.7453 |
| controlled_alt_vs_proposed_raw | length_score | -0.0250 | (-0.2125, 0.1604) | 0.6040 | -0.0250 | (-0.1833, 0.0800) | 0.6937 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0000 | (-0.1312, 0.1312) | 0.5527 | 0.0000 | (-0.1400, 0.1400) | 0.6227 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1029 | (0.0672, 0.1409) | 0.0000 | 0.1029 | (0.0311, 0.1433) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2343 | (0.1737, 0.2919) | 0.0000 | 0.2343 | (0.2048, 0.2926) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1855 | (0.1483, 0.2217) | 0.0000 | 0.1855 | (0.1600, 0.2000) | 0.0000 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0372 | (-0.0075, 0.0851) | 0.0577 | 0.0372 | (-0.0291, 0.1219) | 0.1520 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3035 | (0.2221, 0.3797) | 0.0000 | 0.3035 | (0.2639, 0.3818) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0730 | (0.0491, 0.0975) | 0.0000 | 0.0730 | (0.0669, 0.0845) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1979 | (0.1604, 0.2396) | 0.0000 | 0.1979 | (0.1944, 0.2000) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.1356 | (0.0404, 0.2708) | 0.0000 | 0.1356 | (0.0000, 0.2000) | 0.0337 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0013 | (-0.0215, 0.0248) | 0.4660 | 0.0013 | (-0.0202, 0.0415) | 0.3137 |
| controlled_alt_vs_candidate_no_context | length_score | 0.1396 | (-0.0312, 0.3146) | 0.0620 | 0.1396 | (-0.1167, 0.3867) | 0.1513 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.0875 | (-0.0219, 0.1969) | 0.0870 | 0.0875 | (0.0000, 0.2800) | 0.3013 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1836 | (0.1534, 0.2144) | 0.0000 | 0.1836 | (0.1591, 0.2323) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 11 | 3 | 2 | 0.7500 | 0.7857 |
| proposed_vs_candidate_no_context | persona_consistency | 8 | 3 | 5 | 0.6562 | 0.7273 |
| proposed_vs_candidate_no_context | naturalness | 10 | 4 | 2 | 0.6875 | 0.7143 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 8 | 2 | 6 | 0.6875 | 0.8000 |
| proposed_vs_candidate_no_context | context_overlap | 9 | 5 | 2 | 0.6250 | 0.6429 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 6 | 2 | 8 | 0.6250 | 0.7500 |
| proposed_vs_candidate_no_context | persona_style | 4 | 1 | 11 | 0.5938 | 0.8000 |
| proposed_vs_candidate_no_context | distinct1 | 7 | 6 | 3 | 0.5312 | 0.5385 |
| proposed_vs_candidate_no_context | length_score | 9 | 5 | 2 | 0.6250 | 0.6429 |
| proposed_vs_candidate_no_context | sentence_score | 5 | 1 | 10 | 0.6250 | 0.8333 |
| proposed_vs_candidate_no_context | overall_quality | 11 | 3 | 2 | 0.7500 | 0.7857 |
| controlled_vs_proposed_raw | context_relevance | 13 | 3 | 0 | 0.8125 | 0.8125 |
| controlled_vs_proposed_raw | persona_consistency | 13 | 0 | 3 | 0.9062 | 1.0000 |
| controlled_vs_proposed_raw | naturalness | 5 | 11 | 0 | 0.3125 | 0.3125 |
| controlled_vs_proposed_raw | context_keyword_coverage | 13 | 1 | 2 | 0.8750 | 0.9286 |
| controlled_vs_proposed_raw | context_overlap | 11 | 5 | 0 | 0.6875 | 0.6875 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 13 | 0 | 3 | 0.9062 | 1.0000 |
| controlled_vs_proposed_raw | persona_style | 4 | 5 | 7 | 0.4688 | 0.4444 |
| controlled_vs_proposed_raw | distinct1 | 5 | 10 | 1 | 0.3438 | 0.3333 |
| controlled_vs_proposed_raw | length_score | 7 | 8 | 1 | 0.4688 | 0.4667 |
| controlled_vs_proposed_raw | sentence_score | 3 | 5 | 8 | 0.4375 | 0.3750 |
| controlled_vs_proposed_raw | overall_quality | 16 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_relevance | 16 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 14 | 0 | 2 | 0.9375 | 1.0000 |
| controlled_vs_candidate_no_context | naturalness | 9 | 7 | 0 | 0.5625 | 0.5625 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 16 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 16 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 14 | 0 | 2 | 0.9375 | 1.0000 |
| controlled_vs_candidate_no_context | persona_style | 8 | 1 | 7 | 0.7188 | 0.8889 |
| controlled_vs_candidate_no_context | distinct1 | 8 | 8 | 0 | 0.5000 | 0.5000 |
| controlled_vs_candidate_no_context | length_score | 8 | 8 | 0 | 0.5000 | 0.5000 |
| controlled_vs_candidate_no_context | sentence_score | 4 | 2 | 10 | 0.5625 | 0.6667 |
| controlled_vs_candidate_no_context | overall_quality | 16 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 5 | 11 | 0 | 0.3125 | 0.3125 |
| controlled_alt_vs_controlled_default | persona_consistency | 6 | 5 | 5 | 0.5312 | 0.5455 |
| controlled_alt_vs_controlled_default | naturalness | 11 | 5 | 0 | 0.6875 | 0.6875 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 3 | 10 | 3 | 0.2812 | 0.2308 |
| controlled_alt_vs_controlled_default | context_overlap | 7 | 9 | 0 | 0.4375 | 0.4375 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 2 | 2 | 12 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | persona_style | 5 | 3 | 8 | 0.5625 | 0.6250 |
| controlled_alt_vs_controlled_default | distinct1 | 10 | 6 | 0 | 0.6250 | 0.6250 |
| controlled_alt_vs_controlled_default | length_score | 9 | 7 | 0 | 0.5625 | 0.5625 |
| controlled_alt_vs_controlled_default | sentence_score | 5 | 3 | 8 | 0.5625 | 0.6250 |
| controlled_alt_vs_controlled_default | overall_quality | 5 | 11 | 0 | 0.3125 | 0.3125 |
| controlled_alt_vs_proposed_raw | context_relevance | 13 | 3 | 0 | 0.8125 | 0.8125 |
| controlled_alt_vs_proposed_raw | persona_consistency | 12 | 2 | 2 | 0.8125 | 0.8571 |
| controlled_alt_vs_proposed_raw | naturalness | 8 | 8 | 0 | 0.5000 | 0.5000 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 13 | 1 | 2 | 0.8750 | 0.9286 |
| controlled_alt_vs_proposed_raw | context_overlap | 12 | 4 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 12 | 0 | 4 | 0.8750 | 1.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 4 | 2 | 10 | 0.5625 | 0.6667 |
| controlled_alt_vs_proposed_raw | distinct1 | 4 | 10 | 2 | 0.3125 | 0.2857 |
| controlled_alt_vs_proposed_raw | length_score | 8 | 8 | 0 | 0.5000 | 0.5000 |
| controlled_alt_vs_proposed_raw | sentence_score | 5 | 5 | 6 | 0.5000 | 0.5000 |
| controlled_alt_vs_proposed_raw | overall_quality | 15 | 1 | 0 | 0.9375 | 0.9375 |
| controlled_alt_vs_candidate_no_context | context_relevance | 15 | 1 | 0 | 0.9375 | 0.9375 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 15 | 0 | 1 | 0.9688 | 1.0000 |
| controlled_alt_vs_candidate_no_context | naturalness | 10 | 6 | 0 | 0.6250 | 0.6250 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 15 | 0 | 1 | 0.9688 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 15 | 1 | 0 | 0.9375 | 0.9375 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 15 | 0 | 1 | 0.9688 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_style | 7 | 0 | 9 | 0.7188 | 1.0000 |
| controlled_alt_vs_candidate_no_context | distinct1 | 8 | 7 | 1 | 0.5312 | 0.5333 |
| controlled_alt_vs_candidate_no_context | length_score | 9 | 6 | 1 | 0.5938 | 0.6000 |
| controlled_alt_vs_candidate_no_context | sentence_score | 6 | 2 | 8 | 0.6250 | 0.7500 |
| controlled_alt_vs_candidate_no_context | overall_quality | 16 | 0 | 0 | 1.0000 | 1.0000 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.0625 | 0.6875 | 0.3125 |
| proposed_contextual_controlled_tuned | 0.0000 | 0.0000 | 0.1250 | 0.6875 | 0.3125 |
| proposed_contextual | 0.0000 | 0.0000 | 0.1875 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.3125 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `3`
- Unique template signatures: `16`
- Template signature ratio: `1.0000`
- Effective sample size by source clustering: `2.98`
- Effective sample size by template-signature clustering: `16.00`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: unavailable (disabled_by_flag).

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
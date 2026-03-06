# Proposal Alignment Evaluation Report

- Run ID: `20260304T144313Z`
- Generated: `2026-03-04T14:49:56.084928+00:00`
- Scenarios: `artifacts\proposal_control_tuning\best_v7\20260304T144313Z\scenarios.jsonl`
- Scenario count: `40`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off
- `baseline_no_context`: model `phi3:mini`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2799 (0.2556, 0.3071) | 0.3102 (0.2612, 0.3592) | 0.9054 (0.8813, 0.9269) | 0.3818 (0.3635, 0.4014) | 0.0907 |
| proposed_contextual | 0.0686 (0.0427, 0.0980) | 0.1392 (0.0986, 0.1905) | 0.7870 (0.7662, 0.8096) | 0.2159 (0.1929, 0.2413) | 0.0617 |
| candidate_no_context | 0.0265 (0.0168, 0.0387) | 0.1519 (0.1101, 0.2015) | 0.7895 (0.7699, 0.8124) | 0.2002 (0.1827, 0.2193) | 0.0315 |
| baseline_no_context | 0.0409 (0.0282, 0.0540) | 0.2206 (0.1807, 0.2633) | 0.8855 (0.8666, 0.9032) | 0.2473 (0.2327, 0.2631) | 0.0531 |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0421 | 1.5875 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0128 | -0.0839 |
| proposed_vs_candidate_no_context | naturalness | -0.0025 | -0.0032 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0553 | 2.6467 |
| proposed_vs_candidate_no_context | context_overlap | 0.0111 | 0.2815 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0219 | -0.2973 |
| proposed_vs_candidate_no_context | persona_style | 0.0239 | 0.0513 |
| proposed_vs_candidate_no_context | distinct1 | -0.0019 | -0.0021 |
| proposed_vs_candidate_no_context | length_score | -0.0075 | -0.0338 |
| proposed_vs_candidate_no_context | sentence_score | -0.0012 | -0.0017 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0302 | 0.9602 |
| proposed_vs_candidate_no_context | overall_quality | 0.0157 | 0.0783 |
| proposed_vs_baseline_no_context | context_relevance | 0.0276 | 0.6749 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0814 | -0.3690 |
| proposed_vs_baseline_no_context | naturalness | -0.0985 | -0.1112 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0357 | 0.8813 |
| proposed_vs_baseline_no_context | context_overlap | 0.0088 | 0.2095 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0777 | -0.6002 |
| proposed_vs_baseline_no_context | persona_style | -0.0960 | -0.1641 |
| proposed_vs_baseline_no_context | distinct1 | -0.0529 | -0.0541 |
| proposed_vs_baseline_no_context | length_score | -0.3158 | -0.5959 |
| proposed_vs_baseline_no_context | sentence_score | -0.1400 | -0.1595 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0086 | 0.1615 |
| proposed_vs_baseline_no_context | overall_quality | -0.0314 | -0.1270 |
| controlled_vs_proposed_raw | context_relevance | 0.2113 | 3.0808 |
| controlled_vs_proposed_raw | persona_consistency | 0.1710 | 1.2287 |
| controlled_vs_proposed_raw | naturalness | 0.1184 | 0.1504 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2751 | 3.6076 |
| controlled_vs_proposed_raw | context_overlap | 0.0625 | 1.2326 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1912 | 3.6920 |
| controlled_vs_proposed_raw | persona_style | 0.0904 | 0.1849 |
| controlled_vs_proposed_raw | distinct1 | 0.0068 | 0.0073 |
| controlled_vs_proposed_raw | length_score | 0.4733 | 2.2101 |
| controlled_vs_proposed_raw | sentence_score | 0.2100 | 0.2847 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0290 | 0.4702 |
| controlled_vs_proposed_raw | overall_quality | 0.1659 | 0.7685 |
| controlled_vs_candidate_no_context | context_relevance | 0.2534 | 9.5589 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1583 | 1.0417 |
| controlled_vs_candidate_no_context | naturalness | 0.1159 | 0.1468 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3304 | 15.8025 |
| controlled_vs_candidate_no_context | context_overlap | 0.0737 | 1.8612 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1693 | 2.2973 |
| controlled_vs_candidate_no_context | persona_style | 0.1142 | 0.2457 |
| controlled_vs_candidate_no_context | distinct1 | 0.0049 | 0.0052 |
| controlled_vs_candidate_no_context | length_score | 0.4658 | 2.1015 |
| controlled_vs_candidate_no_context | sentence_score | 0.2087 | 0.2826 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0592 | 1.8820 |
| controlled_vs_candidate_no_context | overall_quality | 0.1816 | 0.9070 |
| controlled_vs_baseline_no_context | context_relevance | 0.2389 | 5.8351 |
| controlled_vs_baseline_no_context | persona_consistency | 0.0896 | 0.4064 |
| controlled_vs_baseline_no_context | naturalness | 0.0199 | 0.0225 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3108 | 7.6682 |
| controlled_vs_baseline_no_context | context_overlap | 0.0713 | 1.7004 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1135 | 0.8759 |
| controlled_vs_baseline_no_context | persona_style | -0.0056 | -0.0096 |
| controlled_vs_baseline_no_context | distinct1 | -0.0461 | -0.0471 |
| controlled_vs_baseline_no_context | length_score | 0.1575 | 0.2972 |
| controlled_vs_baseline_no_context | sentence_score | 0.0700 | 0.0798 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0376 | 0.7077 |
| controlled_vs_baseline_no_context | overall_quality | 0.1345 | 0.5439 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2389 | 5.8351 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.0896 | 0.4064 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 0.0199 | 0.0225 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3108 | 7.6682 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0713 | 1.7004 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1135 | 0.8759 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | -0.0056 | -0.0096 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0461 | -0.0471 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.1575 | 0.2972 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0700 | 0.0798 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0376 | 0.7077 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1345 | 0.5439 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0421 | (0.0152, 0.0736) | 0.0017 | 0.0421 | (0.0077, 0.0892) | 0.0043 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0128 | (-0.0484, 0.0208) | 0.7763 | -0.0128 | (-0.0473, 0.0291) | 0.7170 |
| proposed_vs_candidate_no_context | naturalness | -0.0025 | (-0.0298, 0.0253) | 0.5640 | -0.0025 | (-0.0283, 0.0215) | 0.5967 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0553 | (0.0184, 0.0973) | 0.0020 | 0.0553 | (0.0101, 0.1222) | 0.0050 |
| proposed_vs_candidate_no_context | context_overlap | 0.0111 | (0.0027, 0.0207) | 0.0037 | 0.0111 | (0.0023, 0.0214) | 0.0010 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0219 | (-0.0618, 0.0173) | 0.8650 | -0.0219 | (-0.0628, 0.0289) | 0.8113 |
| proposed_vs_candidate_no_context | persona_style | 0.0239 | (0.0013, 0.0513) | 0.0150 | 0.0239 | (0.0000, 0.0693) | 0.0277 |
| proposed_vs_candidate_no_context | distinct1 | -0.0019 | (-0.0134, 0.0094) | 0.6340 | -0.0019 | (-0.0130, 0.0067) | 0.6683 |
| proposed_vs_candidate_no_context | length_score | -0.0075 | (-0.1150, 0.1075) | 0.5603 | -0.0075 | (-0.1079, 0.0881) | 0.5657 |
| proposed_vs_candidate_no_context | sentence_score | -0.0012 | (-0.0612, 0.0588) | 0.5443 | -0.0012 | (-0.0469, 0.0490) | 0.5707 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0302 | (0.0118, 0.0513) | 0.0000 | 0.0302 | (0.0125, 0.0585) | 0.0000 |
| proposed_vs_candidate_no_context | overall_quality | 0.0157 | (-0.0043, 0.0386) | 0.0723 | 0.0157 | (-0.0054, 0.0433) | 0.0807 |
| proposed_vs_baseline_no_context | context_relevance | 0.0276 | (-0.0033, 0.0607) | 0.0397 | 0.0276 | (-0.0129, 0.0804) | 0.1157 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0814 | (-0.1322, -0.0306) | 1.0000 | -0.0814 | (-0.1191, -0.0385) | 0.9997 |
| proposed_vs_baseline_no_context | naturalness | -0.0985 | (-0.1300, -0.0626) | 1.0000 | -0.0985 | (-0.1334, -0.0516) | 0.9997 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0357 | (-0.0048, 0.0794) | 0.0517 | 0.0357 | (-0.0197, 0.1103) | 0.1363 |
| proposed_vs_baseline_no_context | context_overlap | 0.0088 | (-0.0011, 0.0190) | 0.0443 | 0.0088 | (-0.0038, 0.0225) | 0.0973 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0777 | (-0.1371, -0.0218) | 0.9957 | -0.0777 | (-0.1247, -0.0292) | 0.9997 |
| proposed_vs_baseline_no_context | persona_style | -0.0960 | (-0.1748, -0.0274) | 0.9987 | -0.0960 | (-0.2647, 0.0135) | 0.9037 |
| proposed_vs_baseline_no_context | distinct1 | -0.0529 | (-0.0672, -0.0376) | 1.0000 | -0.0529 | (-0.0689, -0.0325) | 1.0000 |
| proposed_vs_baseline_no_context | length_score | -0.3158 | (-0.4309, -0.1891) | 1.0000 | -0.3158 | (-0.4368, -0.1552) | 1.0000 |
| proposed_vs_baseline_no_context | sentence_score | -0.1400 | (-0.2188, -0.0612) | 0.9990 | -0.1400 | (-0.2250, -0.0219) | 0.9930 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0086 | (-0.0105, 0.0306) | 0.2050 | 0.0086 | (-0.0170, 0.0381) | 0.2620 |
| proposed_vs_baseline_no_context | overall_quality | -0.0314 | (-0.0605, -0.0021) | 0.9817 | -0.0314 | (-0.0625, 0.0112) | 0.9327 |
| controlled_vs_proposed_raw | context_relevance | 0.2113 | (0.1771, 0.2446) | 0.0000 | 0.2113 | (0.1873, 0.2292) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.1710 | (0.1109, 0.2276) | 0.0000 | 0.1710 | (0.0948, 0.2565) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.1184 | (0.0797, 0.1546) | 0.0000 | 0.1184 | (0.0532, 0.1677) | 0.0007 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2751 | (0.2333, 0.3179) | 0.0000 | 0.2751 | (0.2425, 0.2963) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0625 | (0.0506, 0.0750) | 0.0000 | 0.0625 | (0.0506, 0.0761) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1912 | (0.1239, 0.2582) | 0.0000 | 0.1912 | (0.1046, 0.3007) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.0904 | (0.0221, 0.1664) | 0.0033 | 0.0904 | (0.0000, 0.2590) | 0.0943 |
| controlled_vs_proposed_raw | distinct1 | 0.0068 | (-0.0094, 0.0205) | 0.1890 | 0.0068 | (-0.0153, 0.0253) | 0.2693 |
| controlled_vs_proposed_raw | length_score | 0.4733 | (0.3300, 0.6050) | 0.0000 | 0.4733 | (0.2448, 0.6524) | 0.0000 |
| controlled_vs_proposed_raw | sentence_score | 0.2100 | (0.1313, 0.2800) | 0.0000 | 0.2100 | (0.0766, 0.3159) | 0.0003 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0290 | (0.0085, 0.0483) | 0.0030 | 0.0290 | (0.0148, 0.0491) | 0.0000 |
| controlled_vs_proposed_raw | overall_quality | 0.1659 | (0.1391, 0.1930) | 0.0000 | 0.1659 | (0.1370, 0.1999) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2534 | (0.2271, 0.2806) | 0.0000 | 0.2534 | (0.2215, 0.2968) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1583 | (0.1007, 0.2176) | 0.0000 | 0.1583 | (0.0840, 0.2414) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.1159 | (0.0827, 0.1461) | 0.0000 | 0.1159 | (0.0547, 0.1596) | 0.0007 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3304 | (0.2971, 0.3679) | 0.0000 | 0.3304 | (0.2897, 0.3852) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0737 | (0.0622, 0.0851) | 0.0000 | 0.0737 | (0.0587, 0.0906) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1693 | (0.1038, 0.2344) | 0.0000 | 0.1693 | (0.0939, 0.2641) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.1142 | (0.0468, 0.1890) | 0.0003 | 0.1142 | (0.0122, 0.2747) | 0.0057 |
| controlled_vs_candidate_no_context | distinct1 | 0.0049 | (-0.0080, 0.0163) | 0.2220 | 0.0049 | (-0.0161, 0.0194) | 0.3020 |
| controlled_vs_candidate_no_context | length_score | 0.4658 | (0.3408, 0.5867) | 0.0000 | 0.4658 | (0.2491, 0.6213) | 0.0000 |
| controlled_vs_candidate_no_context | sentence_score | 0.2087 | (0.1313, 0.2775) | 0.0000 | 0.2087 | (0.1100, 0.2864) | 0.0000 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0592 | (0.0424, 0.0783) | 0.0000 | 0.0592 | (0.0395, 0.0884) | 0.0000 |
| controlled_vs_candidate_no_context | overall_quality | 0.1816 | (0.1576, 0.2042) | 0.0000 | 0.1816 | (0.1544, 0.2072) | 0.0000 |
| controlled_vs_baseline_no_context | context_relevance | 0.2389 | (0.2112, 0.2684) | 0.0000 | 0.2389 | (0.2003, 0.2919) | 0.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 0.0896 | (0.0383, 0.1408) | 0.0007 | 0.0896 | (0.0019, 0.1898) | 0.0223 |
| controlled_vs_baseline_no_context | naturalness | 0.0199 | (-0.0108, 0.0484) | 0.0940 | 0.0199 | (-0.0195, 0.0572) | 0.1597 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3108 | (0.2731, 0.3513) | 0.0000 | 0.3108 | (0.2575, 0.3816) | 0.0000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0713 | (0.0625, 0.0815) | 0.0000 | 0.0713 | (0.0599, 0.0858) | 0.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1135 | (0.0477, 0.1787) | 0.0007 | 0.1135 | (-0.0004, 0.2426) | 0.0260 |
| controlled_vs_baseline_no_context | persona_style | -0.0056 | (-0.0250, 0.0131) | 0.7327 | -0.0056 | (-0.0375, 0.0187) | 0.5803 |
| controlled_vs_baseline_no_context | distinct1 | -0.0461 | (-0.0590, -0.0326) | 1.0000 | -0.0461 | (-0.0568, -0.0325) | 1.0000 |
| controlled_vs_baseline_no_context | length_score | 0.1575 | (0.0258, 0.2850) | 0.0133 | 0.1575 | (-0.0102, 0.3178) | 0.0320 |
| controlled_vs_baseline_no_context | sentence_score | 0.0700 | (0.0175, 0.1313) | 0.0087 | 0.0700 | (-0.0073, 0.1474) | 0.0430 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0376 | (0.0210, 0.0547) | 0.0000 | 0.0376 | (0.0207, 0.0601) | 0.0000 |
| controlled_vs_baseline_no_context | overall_quality | 0.1345 | (0.1131, 0.1569) | 0.0000 | 0.1345 | (0.0955, 0.1758) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2389 | (0.2106, 0.2692) | 0.0000 | 0.2389 | (0.1993, 0.2912) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.0896 | (0.0368, 0.1394) | 0.0003 | 0.0896 | (0.0015, 0.1923) | 0.0230 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 0.0199 | (-0.0119, 0.0487) | 0.1030 | 0.0199 | (-0.0202, 0.0587) | 0.1453 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3108 | (0.2730, 0.3537) | 0.0000 | 0.3108 | (0.2591, 0.3826) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0713 | (0.0622, 0.0811) | 0.0000 | 0.0713 | (0.0596, 0.0850) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1135 | (0.0460, 0.1776) | 0.0003 | 0.1135 | (0.0054, 0.2359) | 0.0217 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | -0.0056 | (-0.0238, 0.0131) | 0.7420 | -0.0056 | (-0.0368, 0.0196) | 0.5770 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0461 | (-0.0587, -0.0335) | 1.0000 | -0.0461 | (-0.0563, -0.0321) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.1575 | (0.0250, 0.2834) | 0.0110 | 0.1575 | (-0.0128, 0.3158) | 0.0340 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0700 | (0.0175, 0.1313) | 0.0073 | 0.0700 | (-0.0081, 0.1485) | 0.0423 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0376 | (0.0218, 0.0545) | 0.0000 | 0.0376 | (0.0211, 0.0604) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1345 | (0.1123, 0.1569) | 0.0000 | 0.1345 | (0.0956, 0.1758) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 13 | 4 | 23 | 0.6125 | 0.7647 |
| proposed_vs_candidate_no_context | persona_consistency | 7 | 6 | 27 | 0.5125 | 0.5385 |
| proposed_vs_candidate_no_context | naturalness | 8 | 9 | 23 | 0.4875 | 0.4706 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 11 | 2 | 27 | 0.6125 | 0.8462 |
| proposed_vs_candidate_no_context | context_overlap | 11 | 6 | 23 | 0.5625 | 0.6471 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 4 | 6 | 30 | 0.4750 | 0.4000 |
| proposed_vs_candidate_no_context | persona_style | 4 | 0 | 36 | 0.5500 | 1.0000 |
| proposed_vs_candidate_no_context | distinct1 | 8 | 9 | 23 | 0.4875 | 0.4706 |
| proposed_vs_candidate_no_context | length_score | 7 | 10 | 23 | 0.4625 | 0.4118 |
| proposed_vs_candidate_no_context | sentence_score | 6 | 6 | 28 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | bertscore_f1 | 15 | 13 | 12 | 0.5250 | 0.5357 |
| proposed_vs_candidate_no_context | overall_quality | 11 | 17 | 12 | 0.4250 | 0.3929 |
| proposed_vs_baseline_no_context | context_relevance | 20 | 19 | 1 | 0.5125 | 0.5128 |
| proposed_vs_baseline_no_context | persona_consistency | 4 | 21 | 15 | 0.2875 | 0.1600 |
| proposed_vs_baseline_no_context | naturalness | 7 | 32 | 1 | 0.1875 | 0.1795 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 11 | 9 | 20 | 0.5250 | 0.5500 |
| proposed_vs_baseline_no_context | context_overlap | 23 | 16 | 1 | 0.5875 | 0.5897 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 4 | 16 | 20 | 0.3500 | 0.2000 |
| proposed_vs_baseline_no_context | persona_style | 4 | 10 | 26 | 0.4250 | 0.2857 |
| proposed_vs_baseline_no_context | distinct1 | 5 | 32 | 3 | 0.1625 | 0.1351 |
| proposed_vs_baseline_no_context | length_score | 6 | 31 | 3 | 0.1875 | 0.1622 |
| proposed_vs_baseline_no_context | sentence_score | 7 | 23 | 10 | 0.3000 | 0.2333 |
| proposed_vs_baseline_no_context | bertscore_f1 | 19 | 21 | 0 | 0.4750 | 0.4750 |
| proposed_vs_baseline_no_context | overall_quality | 12 | 28 | 0 | 0.3000 | 0.3000 |
| controlled_vs_proposed_raw | context_relevance | 39 | 1 | 0 | 0.9750 | 0.9750 |
| controlled_vs_proposed_raw | persona_consistency | 30 | 3 | 7 | 0.8375 | 0.9091 |
| controlled_vs_proposed_raw | naturalness | 33 | 7 | 0 | 0.8250 | 0.8250 |
| controlled_vs_proposed_raw | context_keyword_coverage | 37 | 1 | 2 | 0.9500 | 0.9737 |
| controlled_vs_proposed_raw | context_overlap | 39 | 1 | 0 | 0.9750 | 0.9750 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 30 | 3 | 7 | 0.8375 | 0.9091 |
| controlled_vs_proposed_raw | persona_style | 9 | 3 | 28 | 0.5750 | 0.7500 |
| controlled_vs_proposed_raw | distinct1 | 28 | 11 | 1 | 0.7125 | 0.7179 |
| controlled_vs_proposed_raw | length_score | 31 | 9 | 0 | 0.7750 | 0.7750 |
| controlled_vs_proposed_raw | sentence_score | 29 | 5 | 6 | 0.8000 | 0.8529 |
| controlled_vs_proposed_raw | bertscore_f1 | 26 | 14 | 0 | 0.6500 | 0.6500 |
| controlled_vs_proposed_raw | overall_quality | 38 | 2 | 0 | 0.9500 | 0.9500 |
| controlled_vs_candidate_no_context | context_relevance | 40 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 29 | 4 | 7 | 0.8125 | 0.8788 |
| controlled_vs_candidate_no_context | naturalness | 34 | 6 | 0 | 0.8500 | 0.8500 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 40 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 40 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 29 | 4 | 7 | 0.8125 | 0.8788 |
| controlled_vs_candidate_no_context | persona_style | 13 | 2 | 25 | 0.6375 | 0.8667 |
| controlled_vs_candidate_no_context | distinct1 | 24 | 15 | 1 | 0.6125 | 0.6154 |
| controlled_vs_candidate_no_context | length_score | 31 | 9 | 0 | 0.7750 | 0.7750 |
| controlled_vs_candidate_no_context | sentence_score | 26 | 3 | 11 | 0.7875 | 0.8966 |
| controlled_vs_candidate_no_context | bertscore_f1 | 33 | 7 | 0 | 0.8250 | 0.8250 |
| controlled_vs_candidate_no_context | overall_quality | 39 | 1 | 0 | 0.9750 | 0.9750 |
| controlled_vs_baseline_no_context | context_relevance | 40 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 25 | 7 | 8 | 0.7250 | 0.7812 |
| controlled_vs_baseline_no_context | naturalness | 28 | 12 | 0 | 0.7000 | 0.7000 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 40 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | context_overlap | 40 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 25 | 6 | 9 | 0.7375 | 0.8065 |
| controlled_vs_baseline_no_context | persona_style | 2 | 3 | 35 | 0.4875 | 0.4000 |
| controlled_vs_baseline_no_context | distinct1 | 4 | 35 | 1 | 0.1125 | 0.1026 |
| controlled_vs_baseline_no_context | length_score | 28 | 12 | 0 | 0.7000 | 0.7000 |
| controlled_vs_baseline_no_context | sentence_score | 10 | 2 | 28 | 0.6000 | 0.8333 |
| controlled_vs_baseline_no_context | bertscore_f1 | 29 | 11 | 0 | 0.7250 | 0.7250 |
| controlled_vs_baseline_no_context | overall_quality | 40 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 40 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 25 | 7 | 8 | 0.7250 | 0.7812 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 28 | 12 | 0 | 0.7000 | 0.7000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 40 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 40 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 25 | 6 | 9 | 0.7375 | 0.8065 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 2 | 3 | 35 | 0.4875 | 0.4000 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | 4 | 35 | 1 | 0.1125 | 0.1026 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 28 | 12 | 0 | 0.7000 | 0.7000 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 10 | 2 | 28 | 0.6000 | 0.8333 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 29 | 11 | 0 | 0.7250 | 0.7250 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 40 | 0 | 0 | 1.0000 | 1.0000 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.5750 | 0.2750 | 0.7250 |
| proposed_contextual | 0.0000 | 0.0000 | 0.6250 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.6250 | 0.0000 | 0.0000 |
| baseline_no_context | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `8`
- Unique template signatures: `33`
- Template signature ratio: `0.8250`
- Effective sample size by source clustering: `7.02`
- Effective sample size by template-signature clustering: `28.57`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: enabled.

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
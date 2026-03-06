# Proposal Alignment Evaluation Report

- Run ID: `20260305T203724Z`
- Generated: `2026-03-05T20:38:12.433927+00:00`
- Scenarios: `artifacts\proposal_control_tuning\smoke_custom\20260305T203724Z\scenarios.jsonl`
- Scenario count: `4`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `tuned_control`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.3184 (0.1543, 0.4554) | 0.3352 (0.3038, 0.3667) | 0.8868 (0.8352, 0.9384) | 0.4387 (0.3858, 0.4916) | n/a |
| tuned_control | 0.3924 (0.3534, 0.4677) | 0.3786 (0.2583, 0.5536) | 0.8889 (0.8748, 0.9069) | 0.4903 (0.4315, 0.5492) | n/a |
| proposed_contextual | 0.1448 (0.0598, 0.2147) | 0.1500 (0.1000, 0.2000) | 0.8814 (0.7874, 0.9650) | 0.2875 (0.2137, 0.3436) | n/a |
| candidate_no_context | 0.0082 (0.0073, 0.0090) | 0.1500 (0.1000, 0.2000) | 0.7486 (0.7486, 0.7486) | 0.1977 (0.1830, 0.2124) | n/a |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.1366 | 16.6409 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | naturalness | 0.1328 | 0.1774 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.1761 | nan |
| proposed_vs_candidate_no_context | context_overlap | 0.0444 | 1.6236 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0000 | nan |
| proposed_vs_candidate_no_context | persona_style | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | distinct1 | 0.0507 | 0.0558 |
| proposed_vs_candidate_no_context | length_score | 0.4750 | 4.7500 |
| proposed_vs_candidate_no_context | sentence_score | 0.1750 | 0.2692 |
| proposed_vs_candidate_no_context | overall_quality | 0.0898 | 0.4544 |
| controlled_vs_proposed_raw | context_relevance | 0.1735 | 1.1980 |
| controlled_vs_proposed_raw | persona_consistency | 0.1852 | 1.2349 |
| controlled_vs_proposed_raw | naturalness | 0.0054 | 0.0061 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2178 | 1.2366 |
| controlled_vs_proposed_raw | context_overlap | 0.0702 | 0.9775 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2440 | nan |
| controlled_vs_proposed_raw | persona_style | -0.0500 | -0.0667 |
| controlled_vs_proposed_raw | distinct1 | -0.0251 | -0.0262 |
| controlled_vs_proposed_raw | length_score | 0.0333 | 0.0580 |
| controlled_vs_proposed_raw | sentence_score | 0.0875 | 0.1061 |
| controlled_vs_proposed_raw | overall_quality | 0.1512 | 0.5258 |
| controlled_vs_candidate_no_context | context_relevance | 0.3101 | 37.7752 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1852 | 1.2349 |
| controlled_vs_candidate_no_context | naturalness | 0.1382 | 0.1845 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3939 | nan |
| controlled_vs_candidate_no_context | context_overlap | 0.1146 | 4.1880 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2440 | nan |
| controlled_vs_candidate_no_context | persona_style | -0.0500 | -0.0667 |
| controlled_vs_candidate_no_context | distinct1 | 0.0256 | 0.0282 |
| controlled_vs_candidate_no_context | length_score | 0.5083 | 5.0833 |
| controlled_vs_candidate_no_context | sentence_score | 0.2625 | 0.4038 |
| controlled_vs_candidate_no_context | overall_quality | 0.2410 | 1.2192 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0740 | 0.2325 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0433 | 0.1293 |
| controlled_alt_vs_controlled_default | naturalness | 0.0021 | 0.0023 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.1136 | 0.2885 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0184 | -0.1295 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0417 | 0.1707 |
| controlled_alt_vs_controlled_default | persona_style | 0.0500 | 0.0714 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0230 | -0.0246 |
| controlled_alt_vs_controlled_default | length_score | 0.1000 | 0.1644 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0875 | -0.0959 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0517 | 0.1177 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.2475 | 1.7091 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.2286 | 1.5238 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0074 | 0.0084 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.3314 | 1.8817 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0518 | 0.7213 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.2857 | nan |
| controlled_alt_vs_proposed_raw | persona_style | 0.0000 | 0.0000 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0481 | -0.0501 |
| controlled_alt_vs_proposed_raw | length_score | 0.1333 | 0.2319 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0000 | 0.0000 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.2028 | 0.7055 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.3842 | 46.7919 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.2286 | 1.5238 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.1402 | 0.1873 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.5076 | nan |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0962 | 3.5160 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.2857 | nan |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0000 | 0.0000 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0026 | 0.0029 |
| controlled_alt_vs_candidate_no_context | length_score | 0.6083 | 6.0833 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1750 | 0.2692 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.2927 | 1.4805 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.1366 | (0.0509, 0.2067) | 0.0053 | 0.1366 | (0.1019, 0.2097) | 0.0000 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | naturalness | 0.1328 | (0.0388, 0.2164) | 0.0037 | 0.1328 | (0.0657, 0.2447) | 0.0000 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.1761 | (0.0625, 0.2614) | 0.0030 | 0.1761 | (0.1364, 0.2500) | 0.0000 |
| proposed_vs_candidate_no_context | context_overlap | 0.0444 | (0.0096, 0.0916) | 0.0037 | 0.0444 | (0.0191, 0.1157) | 0.0000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | persona_style | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | distinct1 | 0.0507 | (0.0106, 0.0909) | 0.0043 | 0.0507 | (0.0211, 0.0909) | 0.0000 |
| proposed_vs_candidate_no_context | length_score | 0.4750 | (0.1500, 0.8000) | 0.0033 | 0.4750 | (0.1500, 0.8667) | 0.0000 |
| proposed_vs_candidate_no_context | sentence_score | 0.1750 | (0.0000, 0.3500) | 0.0627 | 0.1750 | (0.0000, 0.3500) | 0.0397 |
| proposed_vs_candidate_no_context | overall_quality | 0.0898 | (0.0304, 0.1329) | 0.0047 | 0.0898 | (0.0607, 0.1444) | 0.0000 |
| controlled_vs_proposed_raw | context_relevance | 0.1735 | (-0.0290, 0.3321) | 0.0480 | 0.1735 | (0.1207, 0.2881) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.1852 | (0.1038, 0.2667) | 0.0000 | 0.1852 | (0.0933, 0.2667) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0054 | (-0.1140, 0.1241) | 0.4473 | 0.0054 | (-0.1800, 0.1241) | 0.4043 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2178 | (-0.0322, 0.4356) | 0.0590 | 0.2178 | (0.1364, 0.4167) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0702 | (-0.0183, 0.1587) | 0.0697 | 0.0702 | (-0.0119, 0.1246) | 0.0393 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2440 | (0.1548, 0.3333) | 0.0000 | 0.2440 | (0.1429, 0.3333) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | -0.0500 | (-0.1500, 0.0000) | 1.0000 | -0.0500 | (-0.2000, 0.0000) | 1.0000 |
| controlled_vs_proposed_raw | distinct1 | -0.0251 | (-0.0803, 0.0273) | 0.8593 | -0.0251 | (-0.1000, 0.0103) | 0.8507 |
| controlled_vs_proposed_raw | length_score | 0.0333 | (-0.5333, 0.6000) | 0.4233 | 0.0333 | (-0.7000, 0.6000) | 0.3867 |
| controlled_vs_proposed_raw | sentence_score | 0.0875 | (0.0000, 0.2625) | 0.3007 | 0.0875 | (0.0000, 0.3500) | 0.2883 |
| controlled_vs_proposed_raw | overall_quality | 0.1512 | (0.0751, 0.2553) | 0.0000 | 0.1512 | (0.1073, 0.1792) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.3101 | (0.1458, 0.4478) | 0.0000 | 0.3101 | (0.2225, 0.4978) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1852 | (0.1038, 0.2667) | 0.0000 | 0.1852 | (0.0933, 0.2667) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.1382 | (0.0865, 0.1898) | 0.0000 | 0.1382 | (0.0647, 0.1898) | 0.0000 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3939 | (0.1818, 0.5909) | 0.0000 | 0.3939 | (0.2727, 0.6667) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.1146 | (0.0495, 0.1706) | 0.0000 | 0.1146 | (0.1039, 0.1437) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2440 | (0.1548, 0.3333) | 0.0000 | 0.2440 | (0.1429, 0.3333) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | -0.0500 | (-0.1500, 0.0000) | 1.0000 | -0.0500 | (-0.2000, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | distinct1 | 0.0256 | (-0.0045, 0.0557) | 0.0533 | 0.0256 | (-0.0091, 0.0557) | 0.3080 |
| controlled_vs_candidate_no_context | length_score | 0.5083 | (0.2667, 0.7500) | 0.0000 | 0.5083 | (0.1667, 0.7500) | 0.0000 |
| controlled_vs_candidate_no_context | sentence_score | 0.2625 | (0.0875, 0.3500) | 0.0030 | 0.2625 | (0.1750, 0.3500) | 0.0000 |
| controlled_vs_candidate_no_context | overall_quality | 0.2410 | (0.1880, 0.2941) | 0.0000 | 0.2410 | (0.2008, 0.2835) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0740 | (-0.0161, 0.2095) | 0.1160 | 0.0740 | (0.0006, 0.1232) | 0.0000 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0433 | (-0.0900, 0.2000) | 0.3077 | 0.0433 | (0.0000, 0.0667) | 0.0437 |
| controlled_alt_vs_controlled_default | naturalness | 0.0021 | (-0.0628, 0.0547) | 0.4337 | 0.0021 | (-0.0457, 0.0725) | 0.3757 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.1136 | (0.0000, 0.2727) | 0.0620 | 0.1136 | (0.0000, 0.1818) | 0.0357 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0184 | (-0.0815, 0.0495) | 0.6933 | -0.0184 | (-0.0483, 0.0020) | 0.9630 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0417 | (-0.1250, 0.2500) | 0.4317 | 0.0417 | (0.0000, 0.0833) | 0.2953 |
| controlled_alt_vs_controlled_default | persona_style | 0.0500 | (0.0000, 0.1500) | 0.3180 | 0.0500 | (0.0000, 0.2000) | 0.3043 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0230 | (-0.0963, 0.0371) | 0.7250 | -0.0230 | (-0.0725, 0.0552) | 0.7537 |
| controlled_alt_vs_controlled_default | length_score | 0.1000 | (-0.0833, 0.2833) | 0.1827 | 0.1000 | (-0.0833, 0.3667) | 0.1443 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0875 | (-0.3500, 0.1750) | 0.8157 | -0.0875 | (-0.3500, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0517 | (0.0286, 0.0748) | 0.0000 | 0.0517 | (0.0267, 0.0748) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.2475 | (0.1791, 0.3160) | 0.0000 | 0.2475 | (0.2138, 0.2887) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.2286 | (0.1190, 0.4333) | 0.0000 | 0.2286 | (0.1143, 0.3333) | 0.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0074 | (-0.0828, 0.1194) | 0.4190 | 0.0074 | (-0.1075, 0.0784) | 0.3940 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.3314 | (0.2273, 0.4356) | 0.0000 | 0.3314 | (0.2727, 0.4167) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0518 | (0.0117, 0.0799) | 0.0057 | 0.0518 | (-0.0099, 0.0763) | 0.0343 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.2857 | (0.1488, 0.5417) | 0.0000 | 0.2857 | (0.1429, 0.4167) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0481 | (-0.1268, 0.0306) | 0.9247 | -0.0481 | (-0.1020, 0.0341) | 0.9657 |
| controlled_alt_vs_proposed_raw | length_score | 0.1333 | (-0.2500, 0.5167) | 0.2897 | 0.1333 | (-0.3333, 0.5167) | 0.2863 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0000 | (-0.2625, 0.2625) | 0.6343 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.2028 | (0.1262, 0.3292) | 0.0000 | 0.2028 | (0.1377, 0.2539) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.3842 | (0.3445, 0.4605) | 0.0000 | 0.3842 | (0.3457, 0.4984) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.2286 | (0.1190, 0.4333) | 0.0000 | 0.2286 | (0.1143, 0.3333) | 0.0000 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.1402 | (0.1261, 0.1582) | 0.0000 | 0.1402 | (0.1354, 0.1441) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.5076 | (0.4545, 0.6136) | 0.0000 | 0.5076 | (0.4545, 0.6667) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0962 | (0.0877, 0.1032) | 0.0000 | 0.0962 | (0.0918, 0.1058) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.2857 | (0.1488, 0.5417) | 0.0000 | 0.2857 | (0.1429, 0.4167) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0026 | (-0.0387, 0.0411) | 0.4390 | 0.0026 | (-0.0168, 0.0552) | 0.4097 |
| controlled_alt_vs_candidate_no_context | length_score | 0.6083 | (0.5500, 0.6917) | 0.0000 | 0.6083 | (0.5333, 0.6667) | 0.0000 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1750 | (0.0000, 0.3500) | 0.0573 | 0.1750 | (0.0000, 0.3500) | 0.0343 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.2927 | (0.2337, 0.3539) | 0.0000 | 0.2927 | (0.2312, 0.3147) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 3 | 0 | 1 | 0.8750 | 1.0000 |
| proposed_vs_candidate_no_context | persona_consistency | 0 | 0 | 4 | 0.5000 | nan |
| proposed_vs_candidate_no_context | naturalness | 3 | 0 | 1 | 0.8750 | 1.0000 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 3 | 0 | 1 | 0.8750 | 1.0000 |
| proposed_vs_candidate_no_context | context_overlap | 3 | 0 | 1 | 0.8750 | 1.0000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0 | 0 | 4 | 0.5000 | nan |
| proposed_vs_candidate_no_context | persona_style | 0 | 0 | 4 | 0.5000 | nan |
| proposed_vs_candidate_no_context | distinct1 | 3 | 0 | 1 | 0.8750 | 1.0000 |
| proposed_vs_candidate_no_context | length_score | 3 | 0 | 1 | 0.8750 | 1.0000 |
| proposed_vs_candidate_no_context | sentence_score | 2 | 0 | 2 | 0.7500 | 1.0000 |
| proposed_vs_candidate_no_context | overall_quality | 3 | 0 | 1 | 0.8750 | 1.0000 |
| controlled_vs_proposed_raw | context_relevance | 3 | 1 | 0 | 0.7500 | 0.7500 |
| controlled_vs_proposed_raw | persona_consistency | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_proposed_raw | naturalness | 2 | 2 | 0 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | context_keyword_coverage | 3 | 1 | 0 | 0.7500 | 0.7500 |
| controlled_vs_proposed_raw | context_overlap | 2 | 2 | 0 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_proposed_raw | persona_style | 0 | 1 | 3 | 0.3750 | 0.0000 |
| controlled_vs_proposed_raw | distinct1 | 1 | 3 | 0 | 0.2500 | 0.2500 |
| controlled_vs_proposed_raw | length_score | 2 | 2 | 0 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | sentence_score | 1 | 0 | 3 | 0.6250 | 1.0000 |
| controlled_vs_proposed_raw | overall_quality | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_relevance | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | naturalness | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_style | 0 | 1 | 3 | 0.3750 | 0.0000 |
| controlled_vs_candidate_no_context | distinct1 | 2 | 1 | 1 | 0.6250 | 0.6667 |
| controlled_vs_candidate_no_context | length_score | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | sentence_score | 3 | 0 | 1 | 0.8750 | 1.0000 |
| controlled_vs_candidate_no_context | overall_quality | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 3 | 1 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_controlled_default | persona_consistency | 2 | 1 | 1 | 0.6250 | 0.6667 |
| controlled_alt_vs_controlled_default | naturalness | 3 | 1 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 2 | 0 | 2 | 0.7500 | 1.0000 |
| controlled_alt_vs_controlled_default | context_overlap | 2 | 2 | 0 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 1 | 1 | 2 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | persona_style | 1 | 0 | 3 | 0.6250 | 1.0000 |
| controlled_alt_vs_controlled_default | distinct1 | 1 | 3 | 0 | 0.2500 | 0.2500 |
| controlled_alt_vs_controlled_default | length_score | 2 | 2 | 0 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | sentence_score | 1 | 2 | 1 | 0.3750 | 0.3333 |
| controlled_alt_vs_controlled_default | overall_quality | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_proposed_raw | context_relevance | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 1 | 3 | 0 | 0.2500 | 0.2500 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 3 | 1 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 0 | 0 | 4 | 0.5000 | nan |
| controlled_alt_vs_proposed_raw | distinct1 | 2 | 2 | 0 | 0.5000 | 0.5000 |
| controlled_alt_vs_proposed_raw | length_score | 2 | 2 | 0 | 0.5000 | 0.5000 |
| controlled_alt_vs_proposed_raw | sentence_score | 1 | 1 | 2 | 0.5000 | 0.5000 |
| controlled_alt_vs_proposed_raw | overall_quality | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | naturalness | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_style | 0 | 0 | 4 | 0.5000 | nan |
| controlled_alt_vs_candidate_no_context | distinct1 | 2 | 2 | 0 | 0.5000 | 0.5000 |
| controlled_alt_vs_candidate_no_context | length_score | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | sentence_score | 2 | 0 | 2 | 0.7500 | 1.0000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 4 | 0 | 0 | 1.0000 | 1.0000 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.0000 | 0.7500 | 0.2500 |
| tuned_control | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.5000 |
| proposed_contextual | 0.0000 | 0.0000 | 0.2500 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `3`
- Unique template signatures: `4`
- Template signature ratio: `1.0000`
- Effective sample size by source clustering: `2.67`
- Effective sample size by template-signature clustering: `4.00`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: unavailable (disabled_by_flag).

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
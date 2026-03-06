# Proposal Alignment Evaluation Report

- Run ID: `20260305T235339Z`
- Generated: `2026-03-05T23:56:03.514913+00:00`
- Scenarios: `artifacts\proposal_control_tuning\auto_tune_v4\20260305T235104Z\train_runs\trial_001\seed_29\20260305T235339Z\scenarios.jsonl`
- Scenario count: `16`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_tuned`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2423 (0.1964, 0.3024) | 0.4063 (0.3103, 0.5019) | 0.9097 (0.8796, 0.9350) | 0.4362 (0.3954, 0.4746) | n/a |
| proposed_contextual_controlled_tuned | 0.2379 (0.1880, 0.2916) | 0.3596 (0.2747, 0.4490) | 0.9096 (0.8906, 0.9279) | 0.4164 (0.3757, 0.4652) | n/a |
| proposed_contextual | 0.0876 (0.0370, 0.1531) | 0.1974 (0.1313, 0.2708) | 0.8340 (0.7922, 0.8760) | 0.2722 (0.2258, 0.3230) | n/a |
| candidate_no_context | 0.0227 (0.0112, 0.0358) | 0.2176 (0.1540, 0.2838) | 0.8603 (0.8134, 0.9050) | 0.2543 (0.2222, 0.2877) | n/a |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0650 | 2.8653 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0201 | -0.0925 |
| proposed_vs_candidate_no_context | naturalness | -0.0263 | -0.0306 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0852 | 5.0000 |
| proposed_vs_candidate_no_context | context_overlap | 0.0177 | 0.4940 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0164 | -0.1325 |
| proposed_vs_candidate_no_context | persona_style | -0.0352 | -0.0592 |
| proposed_vs_candidate_no_context | distinct1 | -0.0107 | -0.0112 |
| proposed_vs_candidate_no_context | length_score | -0.1104 | -0.2304 |
| proposed_vs_candidate_no_context | sentence_score | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | overall_quality | 0.0179 | 0.0705 |
| controlled_vs_proposed_raw | context_relevance | 0.1547 | 1.7649 |
| controlled_vs_proposed_raw | persona_consistency | 0.2089 | 1.0579 |
| controlled_vs_proposed_raw | naturalness | 0.0757 | 0.0908 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2011 | 1.9667 |
| controlled_vs_proposed_raw | context_overlap | 0.0463 | 0.8648 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2411 | 2.2500 |
| controlled_vs_proposed_raw | persona_style | 0.0801 | 0.1434 |
| controlled_vs_proposed_raw | distinct1 | -0.0076 | -0.0080 |
| controlled_vs_proposed_raw | length_score | 0.3063 | 0.8305 |
| controlled_vs_proposed_raw | sentence_score | 0.1750 | 0.2121 |
| controlled_vs_proposed_raw | overall_quality | 0.1640 | 0.6027 |
| controlled_vs_candidate_no_context | context_relevance | 0.2196 | 9.6873 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1887 | 0.8676 |
| controlled_vs_candidate_no_context | naturalness | 0.0494 | 0.0574 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.2864 | 16.8000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0639 | 1.7861 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2247 | 1.8193 |
| controlled_vs_candidate_no_context | persona_style | 0.0449 | 0.0757 |
| controlled_vs_candidate_no_context | distinct1 | -0.0182 | -0.0191 |
| controlled_vs_candidate_no_context | length_score | 0.1958 | 0.4087 |
| controlled_vs_candidate_no_context | sentence_score | 0.1750 | 0.2121 |
| controlled_vs_candidate_no_context | overall_quality | 0.1820 | 0.7156 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0044 | -0.0183 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0467 | -0.1150 |
| controlled_alt_vs_controlled_default | naturalness | -0.0000 | -0.0001 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0057 | -0.0187 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0015 | -0.0150 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0628 | -0.1803 |
| controlled_alt_vs_controlled_default | persona_style | 0.0176 | 0.0275 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0132 | 0.0141 |
| controlled_alt_vs_controlled_default | length_score | 0.0062 | 0.0093 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0656 | -0.0656 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0198 | -0.0455 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1502 | 1.7144 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1622 | 0.8213 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0757 | 0.0907 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.1955 | 1.9111 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0448 | 0.8368 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1783 | 1.6639 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0977 | 0.1748 |
| controlled_alt_vs_proposed_raw | distinct1 | 0.0056 | 0.0059 |
| controlled_alt_vs_proposed_raw | length_score | 0.3125 | 0.8475 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1094 | 0.1326 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1442 | 0.5298 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2152 | 9.4921 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1420 | 0.6528 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0493 | 0.0573 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.2807 | 16.4667 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0625 | 1.7443 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1619 | 1.3108 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0625 | 0.1053 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0051 | -0.0053 |
| controlled_alt_vs_candidate_no_context | length_score | 0.2021 | 0.4217 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1094 | 0.1326 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1621 | 0.6376 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0650 | (0.0144, 0.1291) | 0.0033 | 0.0650 | (0.0073, 0.1278) | 0.0123 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0201 | (-0.0963, 0.0710) | 0.6850 | -0.0201 | (-0.1164, 0.0638) | 0.6550 |
| proposed_vs_candidate_no_context | naturalness | -0.0263 | (-0.0646, 0.0085) | 0.9320 | -0.0263 | (-0.0541, -0.0049) | 1.0000 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0852 | (0.0170, 0.1648) | 0.0073 | 0.0852 | (0.0107, 0.1658) | 0.0117 |
| proposed_vs_candidate_no_context | context_overlap | 0.0177 | (0.0040, 0.0323) | 0.0037 | 0.0177 | (-0.0007, 0.0385) | 0.0303 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0164 | (-0.1116, 0.0923) | 0.6500 | -0.0164 | (-0.1250, 0.0756) | 0.6237 |
| proposed_vs_candidate_no_context | persona_style | -0.0352 | (-0.1211, 0.0312) | 0.8537 | -0.0352 | (-0.1287, 0.0469) | 0.7400 |
| proposed_vs_candidate_no_context | distinct1 | -0.0107 | (-0.0241, 0.0037) | 0.9277 | -0.0107 | (-0.0288, 0.0085) | 0.8333 |
| proposed_vs_candidate_no_context | length_score | -0.1104 | (-0.2584, 0.0312) | 0.9280 | -0.1104 | (-0.2118, -0.0216) | 1.0000 |
| proposed_vs_candidate_no_context | sentence_score | 0.0000 | (-0.0875, 0.0875) | 0.6147 | 0.0000 | (-0.0553, 0.0808) | 0.6507 |
| proposed_vs_candidate_no_context | overall_quality | 0.0179 | (-0.0091, 0.0516) | 0.1257 | 0.0179 | (0.0026, 0.0425) | 0.0003 |
| controlled_vs_proposed_raw | context_relevance | 0.1547 | (0.0770, 0.2175) | 0.0000 | 0.1547 | (0.0992, 0.2101) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.2089 | (0.1181, 0.3045) | 0.0000 | 0.2089 | (0.1371, 0.2997) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0757 | (0.0215, 0.1268) | 0.0013 | 0.0757 | (-0.0079, 0.1341) | 0.0400 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2011 | (0.1023, 0.2864) | 0.0007 | 0.2011 | (0.1295, 0.2727) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0463 | (0.0270, 0.0663) | 0.0000 | 0.0463 | (0.0259, 0.0665) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2411 | (0.1286, 0.3607) | 0.0000 | 0.2411 | (0.1565, 0.3605) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.0801 | (-0.0195, 0.2051) | 0.0887 | 0.0801 | (-0.0469, 0.2552) | 0.3243 |
| controlled_vs_proposed_raw | distinct1 | -0.0076 | (-0.0283, 0.0110) | 0.7700 | -0.0076 | (-0.0375, 0.0129) | 0.7210 |
| controlled_vs_proposed_raw | length_score | 0.3063 | (0.0791, 0.5062) | 0.0043 | 0.3063 | (-0.0422, 0.5315) | 0.0377 |
| controlled_vs_proposed_raw | sentence_score | 0.1750 | (0.0875, 0.2625) | 0.0000 | 0.1750 | (0.0500, 0.2882) | 0.0073 |
| controlled_vs_proposed_raw | overall_quality | 0.1640 | (0.1058, 0.2204) | 0.0000 | 0.1640 | (0.0983, 0.2194) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2196 | (0.1676, 0.2805) | 0.0000 | 0.2196 | (0.1654, 0.2620) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1887 | (0.0854, 0.2961) | 0.0000 | 0.1887 | (0.0603, 0.3088) | 0.0010 |
| controlled_vs_candidate_no_context | naturalness | 0.0494 | (-0.0066, 0.1068) | 0.0497 | 0.0494 | (-0.0333, 0.1138) | 0.1210 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.2864 | (0.2170, 0.3693) | 0.0000 | 0.2864 | (0.2182, 0.3427) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0639 | (0.0458, 0.0837) | 0.0000 | 0.0639 | (0.0430, 0.0810) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2247 | (0.0967, 0.3530) | 0.0000 | 0.2247 | (0.0595, 0.3698) | 0.0007 |
| controlled_vs_candidate_no_context | persona_style | 0.0449 | (-0.0312, 0.1484) | 0.1683 | 0.0449 | (0.0000, 0.1198) | 0.3207 |
| controlled_vs_candidate_no_context | distinct1 | -0.0182 | (-0.0362, -0.0004) | 0.9773 | -0.0182 | (-0.0472, 0.0039) | 0.9410 |
| controlled_vs_candidate_no_context | length_score | 0.1958 | (-0.0376, 0.4230) | 0.0507 | 0.1958 | (-0.1048, 0.4389) | 0.0973 |
| controlled_vs_candidate_no_context | sentence_score | 0.1750 | (0.0875, 0.2625) | 0.0000 | 0.1750 | (0.0656, 0.2844) | 0.0003 |
| controlled_vs_candidate_no_context | overall_quality | 0.1820 | (0.1364, 0.2293) | 0.0000 | 0.1820 | (0.1307, 0.2322) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0044 | (-0.0552, 0.0507) | 0.5680 | -0.0044 | (-0.0607, 0.0728) | 0.6293 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0467 | (-0.1402, 0.0429) | 0.8330 | -0.0467 | (-0.1457, 0.0523) | 0.7947 |
| controlled_alt_vs_controlled_default | naturalness | -0.0000 | (-0.0221, 0.0241) | 0.5053 | -0.0000 | (-0.0165, 0.0134) | 0.4927 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0057 | (-0.0739, 0.0682) | 0.5820 | -0.0057 | (-0.0795, 0.0909) | 0.6307 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0015 | (-0.0197, 0.0182) | 0.5630 | -0.0015 | (-0.0215, 0.0306) | 0.5967 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0628 | (-0.1821, 0.0476) | 0.8507 | -0.0628 | (-0.1821, 0.0565) | 0.8227 |
| controlled_alt_vs_controlled_default | persona_style | 0.0176 | (0.0000, 0.0508) | 0.1253 | 0.0176 | (0.0000, 0.0500) | 0.0767 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0132 | (-0.0004, 0.0293) | 0.0313 | 0.0132 | (0.0016, 0.0280) | 0.0100 |
| controlled_alt_vs_controlled_default | length_score | 0.0063 | (-0.0875, 0.1063) | 0.4557 | 0.0063 | (-0.0451, 0.0667) | 0.3807 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0656 | (-0.1312, 0.0000) | 1.0000 | -0.0656 | (-0.1750, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0198 | (-0.0672, 0.0317) | 0.7873 | -0.0198 | (-0.0665, 0.0421) | 0.7413 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1502 | (0.0649, 0.2154) | 0.0003 | 0.1502 | (0.0891, 0.2114) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1622 | (0.1047, 0.2171) | 0.0000 | 0.1622 | (0.1058, 0.2160) | 0.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0757 | (0.0245, 0.1269) | 0.0017 | 0.0757 | (-0.0117, 0.1385) | 0.0503 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.1955 | (0.0875, 0.2864) | 0.0003 | 0.1955 | (0.1157, 0.2727) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0448 | (0.0219, 0.0672) | 0.0000 | 0.0448 | (0.0213, 0.0684) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1783 | (0.1116, 0.2453) | 0.0000 | 0.1783 | (0.1315, 0.2190) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0977 | (0.0000, 0.2207) | 0.0403 | 0.0977 | (0.0000, 0.2757) | 0.3343 |
| controlled_alt_vs_proposed_raw | distinct1 | 0.0056 | (-0.0112, 0.0232) | 0.2723 | 0.0056 | (-0.0129, 0.0226) | 0.2740 |
| controlled_alt_vs_proposed_raw | length_score | 0.3125 | (0.1083, 0.5250) | 0.0010 | 0.3125 | (-0.0044, 0.5370) | 0.0307 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1094 | (-0.0219, 0.2406) | 0.0760 | 0.1094 | (-0.1077, 0.2722) | 0.1547 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1442 | (0.0998, 0.1822) | 0.0000 | 0.1442 | (0.1092, 0.1722) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2152 | (0.1583, 0.2761) | 0.0000 | 0.2152 | (0.1871, 0.2435) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1420 | (0.0632, 0.2359) | 0.0000 | 0.1420 | (0.0498, 0.2365) | 0.0030 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0493 | (-0.0109, 0.1086) | 0.0543 | 0.0493 | (-0.0391, 0.1276) | 0.1340 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.2807 | (0.2045, 0.3602) | 0.0000 | 0.2807 | (0.2468, 0.3147) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0625 | (0.0486, 0.0789) | 0.0000 | 0.0625 | (0.0498, 0.0771) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1619 | (0.0613, 0.2887) | 0.0000 | 0.1619 | (0.0473, 0.2840) | 0.0057 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0625 | (0.0000, 0.1543) | 0.0343 | 0.0625 | (0.0000, 0.1389) | 0.0800 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0051 | (-0.0230, 0.0139) | 0.7197 | -0.0051 | (-0.0305, 0.0196) | 0.6517 |
| controlled_alt_vs_candidate_no_context | length_score | 0.2021 | (-0.0396, 0.4396) | 0.0523 | 0.2021 | (-0.1095, 0.4667) | 0.1190 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1094 | (-0.0219, 0.2188) | 0.0540 | 0.1094 | (-0.0500, 0.2676) | 0.1380 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1621 | (0.1182, 0.2136) | 0.0000 | 0.1621 | (0.1213, 0.2022) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 8 | 2 | 6 | 0.6875 | 0.8000 |
| proposed_vs_candidate_no_context | persona_consistency | 3 | 6 | 7 | 0.4062 | 0.3333 |
| proposed_vs_candidate_no_context | naturalness | 4 | 6 | 6 | 0.4375 | 0.4000 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 7 | 2 | 7 | 0.6562 | 0.7778 |
| proposed_vs_candidate_no_context | context_overlap | 7 | 3 | 6 | 0.6250 | 0.7000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 3 | 4 | 9 | 0.4688 | 0.4286 |
| proposed_vs_candidate_no_context | persona_style | 1 | 2 | 13 | 0.4688 | 0.3333 |
| proposed_vs_candidate_no_context | distinct1 | 1 | 7 | 8 | 0.3125 | 0.1250 |
| proposed_vs_candidate_no_context | length_score | 3 | 7 | 6 | 0.3750 | 0.3000 |
| proposed_vs_candidate_no_context | sentence_score | 2 | 2 | 12 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | overall_quality | 6 | 4 | 6 | 0.5625 | 0.6000 |
| controlled_vs_proposed_raw | context_relevance | 14 | 1 | 1 | 0.9062 | 0.9333 |
| controlled_vs_proposed_raw | persona_consistency | 12 | 1 | 3 | 0.8438 | 0.9231 |
| controlled_vs_proposed_raw | naturalness | 13 | 3 | 0 | 0.8125 | 0.8125 |
| controlled_vs_proposed_raw | context_keyword_coverage | 14 | 1 | 1 | 0.9062 | 0.9333 |
| controlled_vs_proposed_raw | context_overlap | 13 | 2 | 1 | 0.8438 | 0.8667 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 12 | 1 | 3 | 0.8438 | 0.9231 |
| controlled_vs_proposed_raw | persona_style | 3 | 2 | 11 | 0.5312 | 0.6000 |
| controlled_vs_proposed_raw | distinct1 | 9 | 7 | 0 | 0.5625 | 0.5625 |
| controlled_vs_proposed_raw | length_score | 12 | 3 | 1 | 0.7812 | 0.8000 |
| controlled_vs_proposed_raw | sentence_score | 8 | 0 | 8 | 0.7500 | 1.0000 |
| controlled_vs_proposed_raw | overall_quality | 14 | 2 | 0 | 0.8750 | 0.8750 |
| controlled_vs_candidate_no_context | context_relevance | 15 | 1 | 0 | 0.9375 | 0.9375 |
| controlled_vs_candidate_no_context | persona_consistency | 11 | 2 | 3 | 0.7812 | 0.8462 |
| controlled_vs_candidate_no_context | naturalness | 9 | 7 | 0 | 0.5625 | 0.5625 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 15 | 0 | 1 | 0.9688 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 15 | 1 | 0 | 0.9375 | 0.9375 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 11 | 2 | 3 | 0.7812 | 0.8462 |
| controlled_vs_candidate_no_context | persona_style | 3 | 2 | 11 | 0.5312 | 0.6000 |
| controlled_vs_candidate_no_context | distinct1 | 5 | 11 | 0 | 0.3125 | 0.3125 |
| controlled_vs_candidate_no_context | length_score | 10 | 6 | 0 | 0.6250 | 0.6250 |
| controlled_vs_candidate_no_context | sentence_score | 8 | 0 | 8 | 0.7500 | 1.0000 |
| controlled_vs_candidate_no_context | overall_quality | 16 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 4 | 6 | 6 | 0.4375 | 0.4000 |
| controlled_alt_vs_controlled_default | persona_consistency | 2 | 4 | 10 | 0.4375 | 0.3333 |
| controlled_alt_vs_controlled_default | naturalness | 5 | 5 | 6 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 3 | 4 | 9 | 0.4688 | 0.4286 |
| controlled_alt_vs_controlled_default | context_overlap | 3 | 6 | 7 | 0.4062 | 0.3333 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 2 | 4 | 10 | 0.4375 | 0.3333 |
| controlled_alt_vs_controlled_default | persona_style | 2 | 0 | 14 | 0.5625 | 1.0000 |
| controlled_alt_vs_controlled_default | distinct1 | 7 | 3 | 6 | 0.6250 | 0.7000 |
| controlled_alt_vs_controlled_default | length_score | 4 | 5 | 7 | 0.4688 | 0.4444 |
| controlled_alt_vs_controlled_default | sentence_score | 0 | 3 | 13 | 0.4062 | 0.0000 |
| controlled_alt_vs_controlled_default | overall_quality | 4 | 6 | 6 | 0.4375 | 0.4000 |
| controlled_alt_vs_proposed_raw | context_relevance | 15 | 1 | 0 | 0.9375 | 0.9375 |
| controlled_alt_vs_proposed_raw | persona_consistency | 12 | 0 | 4 | 0.8750 | 1.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 9 | 7 | 0 | 0.5625 | 0.5625 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 14 | 1 | 1 | 0.9062 | 0.9333 |
| controlled_alt_vs_proposed_raw | context_overlap | 13 | 3 | 0 | 0.8125 | 0.8125 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 12 | 0 | 4 | 0.8750 | 1.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 3 | 0 | 13 | 0.5938 | 1.0000 |
| controlled_alt_vs_proposed_raw | distinct1 | 9 | 6 | 1 | 0.5938 | 0.6000 |
| controlled_alt_vs_proposed_raw | length_score | 12 | 4 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_proposed_raw | sentence_score | 8 | 3 | 5 | 0.6562 | 0.7273 |
| controlled_alt_vs_proposed_raw | overall_quality | 15 | 1 | 0 | 0.9375 | 0.9375 |
| controlled_alt_vs_candidate_no_context | context_relevance | 16 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 11 | 1 | 4 | 0.8125 | 0.9167 |
| controlled_alt_vs_candidate_no_context | naturalness | 8 | 8 | 0 | 0.5000 | 0.5000 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 15 | 0 | 1 | 0.9688 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 16 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 11 | 1 | 4 | 0.8125 | 0.9167 |
| controlled_alt_vs_candidate_no_context | persona_style | 3 | 0 | 13 | 0.5938 | 1.0000 |
| controlled_alt_vs_candidate_no_context | distinct1 | 6 | 9 | 1 | 0.4062 | 0.4000 |
| controlled_alt_vs_candidate_no_context | length_score | 9 | 6 | 1 | 0.5938 | 0.6000 |
| controlled_alt_vs_candidate_no_context | sentence_score | 7 | 2 | 7 | 0.6562 | 0.7778 |
| controlled_alt_vs_candidate_no_context | overall_quality | 16 | 0 | 0 | 1.0000 | 1.0000 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.5000 | 0.4375 | 0.5625 |
| proposed_contextual_controlled_tuned | 0.0000 | 0.0000 | 0.3750 | 0.5000 | 0.5000 |
| proposed_contextual | 0.0000 | 0.0000 | 0.4375 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.3750 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `5`
- Unique template signatures: `16`
- Template signature ratio: `1.0000`
- Effective sample size by source clustering: `4.74`
- Effective sample size by template-signature clustering: `16.00`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: unavailable (disabled_by_flag).

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
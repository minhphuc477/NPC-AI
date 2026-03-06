# Proposal Alignment Evaluation Report

- Run ID: `20260306T000936Z`
- Generated: `2026-03-06T00:12:22.575093+00:00`
- Scenarios: `artifacts\proposal_control_tuning\auto_tune_v4\20260305T235104Z\valid_runs\trial_004\seed_31\20260306T000936Z\scenarios.jsonl`
- Scenario count: `16`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_tuned`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2681 (0.2287, 0.3108) | 0.3047 (0.2647, 0.3508) | 0.9103 (0.8855, 0.9346) | 0.4107 (0.3892, 0.4320) | n/a |
| proposed_contextual_controlled_tuned | 0.3062 (0.2722, 0.3415) | 0.2997 (0.2590, 0.3437) | 0.8713 (0.8325, 0.9086) | 0.4197 (0.3936, 0.4443) | n/a |
| proposed_contextual | 0.0907 (0.0504, 0.1377) | 0.1121 (0.0683, 0.1522) | 0.8557 (0.8115, 0.9007) | 0.2462 (0.2085, 0.2849) | n/a |
| candidate_no_context | 0.0347 (0.0140, 0.0596) | 0.1217 (0.0704, 0.1797) | 0.8268 (0.7905, 0.8661) | 0.2188 (0.1884, 0.2515) | n/a |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0559 | 1.6110 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0096 | -0.0788 |
| proposed_vs_candidate_no_context | naturalness | 0.0289 | 0.0350 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0696 | 2.1618 |
| proposed_vs_candidate_no_context | context_overlap | 0.0241 | 0.5923 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0208 | -0.6667 |
| proposed_vs_candidate_no_context | persona_style | 0.0354 | 0.0732 |
| proposed_vs_candidate_no_context | distinct1 | -0.0001 | -0.0001 |
| proposed_vs_candidate_no_context | length_score | 0.1229 | 0.3296 |
| proposed_vs_candidate_no_context | sentence_score | 0.0437 | 0.0593 |
| proposed_vs_candidate_no_context | overall_quality | 0.0274 | 0.1251 |
| controlled_vs_proposed_raw | context_relevance | 0.1774 | 1.9569 |
| controlled_vs_proposed_raw | persona_consistency | 0.1926 | 1.7179 |
| controlled_vs_proposed_raw | naturalness | 0.0546 | 0.0638 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2353 | 2.3116 |
| controlled_vs_proposed_raw | context_overlap | 0.0423 | 0.6541 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2188 | 21.0000 |
| controlled_vs_proposed_raw | persona_style | 0.0878 | 0.1693 |
| controlled_vs_proposed_raw | distinct1 | -0.0111 | -0.0118 |
| controlled_vs_proposed_raw | length_score | 0.2188 | 0.4412 |
| controlled_vs_proposed_raw | sentence_score | 0.1531 | 0.1960 |
| controlled_vs_proposed_raw | overall_quality | 0.1645 | 0.6683 |
| controlled_vs_candidate_no_context | context_relevance | 0.2334 | 6.7205 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1830 | 1.5036 |
| controlled_vs_candidate_no_context | naturalness | 0.0835 | 0.1010 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3049 | 9.4706 |
| controlled_vs_candidate_no_context | context_overlap | 0.0664 | 1.6338 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1979 | 6.3333 |
| controlled_vs_candidate_no_context | persona_style | 0.1232 | 0.2549 |
| controlled_vs_candidate_no_context | distinct1 | -0.0112 | -0.0118 |
| controlled_vs_candidate_no_context | length_score | 0.3417 | 0.9162 |
| controlled_vs_candidate_no_context | sentence_score | 0.1969 | 0.2669 |
| controlled_vs_candidate_no_context | overall_quality | 0.1919 | 0.8769 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0382 | 0.1423 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0050 | -0.0165 |
| controlled_alt_vs_controlled_default | naturalness | -0.0390 | -0.0429 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0526 | 0.1559 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0046 | 0.0427 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0063 | 0.0273 |
| controlled_alt_vs_controlled_default | persona_style | -0.0501 | -0.0826 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0038 | 0.0040 |
| controlled_alt_vs_controlled_default | length_score | -0.1917 | -0.2682 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0219 | -0.0234 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0090 | 0.0219 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.2156 | 2.3778 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1876 | 1.6731 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0156 | 0.0182 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2879 | 2.8279 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0469 | 0.7247 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.2250 | 21.6000 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0378 | 0.0728 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0074 | -0.0078 |
| controlled_alt_vs_proposed_raw | length_score | 0.0271 | 0.0546 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1312 | 0.1680 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1735 | 0.7049 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2715 | 7.8195 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1780 | 1.4624 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0445 | 0.0539 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3575 | 11.1029 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0709 | 1.7462 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.2042 | 6.5333 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0731 | 0.1513 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0074 | -0.0078 |
| controlled_alt_vs_candidate_no_context | length_score | 0.1500 | 0.4022 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1750 | 0.2373 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.2009 | 0.9181 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0559 | (0.0090, 0.1062) | 0.0087 | 0.0559 | (0.0000, 0.0975) | 0.0313 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0096 | (-0.0476, 0.0247) | 0.6917 | -0.0096 | (-0.0256, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | naturalness | 0.0289 | (-0.0091, 0.0703) | 0.0753 | 0.0289 | (0.0000, 0.0751) | 0.0340 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0696 | (0.0052, 0.1435) | 0.0163 | 0.0696 | (0.0000, 0.1250) | 0.0410 |
| proposed_vs_candidate_no_context | context_overlap | 0.0241 | (0.0126, 0.0366) | 0.0000 | 0.0241 | (0.0000, 0.0368) | 0.0383 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0208 | (-0.0625, 0.0208) | 0.9033 | -0.0208 | (-0.0556, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | persona_style | 0.0354 | (-0.0229, 0.1102) | 0.1547 | 0.0354 | (0.0000, 0.0943) | 0.3003 |
| proposed_vs_candidate_no_context | distinct1 | -0.0001 | (-0.0155, 0.0161) | 0.4950 | -0.0001 | (-0.0105, 0.0086) | 0.6353 |
| proposed_vs_candidate_no_context | length_score | 0.1229 | (-0.0396, 0.2917) | 0.0743 | 0.1229 | (0.0000, 0.3000) | 0.0387 |
| proposed_vs_candidate_no_context | sentence_score | 0.0437 | (-0.0656, 0.1531) | 0.3030 | 0.0437 | (0.0000, 0.1167) | 0.2960 |
| proposed_vs_candidate_no_context | overall_quality | 0.0274 | (-0.0089, 0.0665) | 0.0763 | 0.0274 | (0.0000, 0.0491) | 0.0333 |
| controlled_vs_proposed_raw | context_relevance | 0.1774 | (0.0999, 0.2544) | 0.0000 | 0.1774 | (0.0727, 0.3296) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.1926 | (0.1517, 0.2267) | 0.0000 | 0.1926 | (0.1600, 0.2267) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0546 | (0.0034, 0.1041) | 0.0167 | 0.0546 | (0.0001, 0.1734) | 0.0000 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2353 | (0.1326, 0.3352) | 0.0000 | 0.2353 | (0.0972, 0.4364) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0423 | (0.0216, 0.0638) | 0.0000 | 0.0423 | (0.0153, 0.0806) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2188 | (0.1750, 0.2604) | 0.0000 | 0.2188 | (0.2000, 0.2500) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.0878 | (0.0065, 0.1786) | 0.0140 | 0.0878 | (-0.0435, 0.3333) | 0.2923 |
| controlled_vs_proposed_raw | distinct1 | -0.0111 | (-0.0346, 0.0121) | 0.8313 | -0.0111 | (-0.0598, 0.0235) | 0.7007 |
| controlled_vs_proposed_raw | length_score | 0.2188 | (0.0042, 0.4167) | 0.0230 | 0.2188 | (-0.0833, 0.6800) | 0.1653 |
| controlled_vs_proposed_raw | sentence_score | 0.1531 | (0.0437, 0.2625) | 0.0070 | 0.1531 | (0.0000, 0.2800) | 0.0380 |
| controlled_vs_proposed_raw | overall_quality | 0.1645 | (0.1145, 0.2128) | 0.0000 | 0.1645 | (0.1065, 0.2683) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2334 | (0.1860, 0.2860) | 0.0000 | 0.2334 | (0.1702, 0.3296) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1830 | (0.1333, 0.2328) | 0.0000 | 0.1830 | (0.1600, 0.2267) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.0835 | (0.0339, 0.1311) | 0.0013 | 0.0835 | (0.0026, 0.1734) | 0.0000 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3049 | (0.2415, 0.3788) | 0.0000 | 0.3049 | (0.2222, 0.4364) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0664 | (0.0515, 0.0823) | 0.0000 | 0.0664 | (0.0488, 0.0806) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1979 | (0.1479, 0.2437) | 0.0000 | 0.1979 | (0.1944, 0.2000) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.1232 | (0.0205, 0.2250) | 0.0103 | 0.1232 | (0.0000, 0.3333) | 0.0407 |
| controlled_vs_candidate_no_context | distinct1 | -0.0112 | (-0.0370, 0.0157) | 0.7840 | -0.0112 | (-0.0702, 0.0235) | 0.7067 |
| controlled_vs_candidate_no_context | length_score | 0.3417 | (0.1542, 0.5167) | 0.0000 | 0.3417 | (0.1533, 0.6800) | 0.0000 |
| controlled_vs_candidate_no_context | sentence_score | 0.1969 | (0.0875, 0.2844) | 0.0003 | 0.1969 | (0.0000, 0.2917) | 0.0350 |
| controlled_vs_candidate_no_context | overall_quality | 0.1919 | (0.1512, 0.2309) | 0.0000 | 0.1919 | (0.1557, 0.2683) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0382 | (-0.0231, 0.0936) | 0.1037 | 0.0382 | (-0.0835, 0.0935) | 0.2620 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0050 | (-0.0508, 0.0469) | 0.6087 | -0.0050 | (-0.0712, 0.0693) | 0.6313 |
| controlled_alt_vs_controlled_default | naturalness | -0.0390 | (-0.0772, -0.0040) | 0.9873 | -0.0390 | (-0.0901, 0.0125) | 0.9623 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0526 | (-0.0303, 0.1250) | 0.1013 | 0.0526 | (-0.1091, 0.1273) | 0.2577 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0046 | (-0.0118, 0.0189) | 0.2717 | 0.0046 | (-0.0238, 0.0196) | 0.2387 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0063 | (-0.0521, 0.0771) | 0.4457 | 0.0063 | (-0.0833, 0.1200) | 0.3900 |
| controlled_alt_vs_controlled_default | persona_style | -0.0501 | (-0.1084, 0.0029) | 0.9693 | -0.0501 | (-0.1333, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0038 | (-0.0134, 0.0202) | 0.3473 | 0.0038 | (-0.0151, 0.0387) | 0.2847 |
| controlled_alt_vs_controlled_default | length_score | -0.1917 | (-0.3563, -0.0500) | 0.9990 | -0.1917 | (-0.4333, 0.0200) | 0.9610 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0219 | (-0.1094, 0.0656) | 0.7473 | -0.0219 | (-0.0700, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0090 | (-0.0266, 0.0418) | 0.2970 | 0.0090 | (-0.0288, 0.0464) | 0.3643 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.2156 | (0.1642, 0.2664) | 0.0000 | 0.2156 | (0.1660, 0.2461) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1876 | (0.1324, 0.2501) | 0.0000 | 0.1876 | (0.1201, 0.2960) | 0.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0156 | (-0.0309, 0.0673) | 0.2750 | 0.0156 | (-0.0383, 0.0833) | 0.3613 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2879 | (0.2168, 0.3584) | 0.0000 | 0.2879 | (0.2222, 0.3273) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0469 | (0.0355, 0.0578) | 0.0000 | 0.0469 | (0.0350, 0.0568) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.2250 | (0.1666, 0.2938) | 0.0000 | 0.2250 | (0.1667, 0.3200) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0378 | (-0.0387, 0.1078) | 0.1693 | 0.0378 | (-0.0660, 0.2000) | 0.3017 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0074 | (-0.0222, 0.0085) | 0.8257 | -0.0074 | (-0.0210, 0.0149) | 0.7307 |
| controlled_alt_vs_proposed_raw | length_score | 0.0271 | (-0.1750, 0.2417) | 0.4103 | 0.0271 | (-0.2500, 0.2467) | 0.3710 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1312 | (0.0000, 0.2406) | 0.0313 | 0.1312 | (-0.0700, 0.2800) | 0.0357 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1735 | (0.1322, 0.2150) | 0.0000 | 0.1735 | (0.1159, 0.2394) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2715 | (0.2429, 0.3007) | 0.0000 | 0.2715 | (0.2461, 0.3064) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1780 | (0.1212, 0.2461) | 0.0000 | 0.1780 | (0.0946, 0.2960) | 0.0000 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0445 | (0.0084, 0.0855) | 0.0043 | 0.0445 | (0.0151, 0.0833) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3575 | (0.3182, 0.3958) | 0.0000 | 0.3575 | (0.3273, 0.4000) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0709 | (0.0599, 0.0819) | 0.0000 | 0.0709 | (0.0568, 0.0880) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.2042 | (0.1437, 0.2750) | 0.0000 | 0.2042 | (0.1111, 0.3200) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0731 | (-0.0057, 0.1558) | 0.0350 | 0.0731 | (0.0000, 0.2000) | 0.0337 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0074 | (-0.0267, 0.0110) | 0.7697 | -0.0074 | (-0.0315, 0.0149) | 0.7483 |
| controlled_alt_vs_candidate_no_context | length_score | 0.1500 | (0.0021, 0.2979) | 0.0240 | 0.1500 | (0.0500, 0.2467) | 0.0000 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1750 | (0.0437, 0.2844) | 0.0053 | 0.1750 | (-0.0700, 0.2917) | 0.0360 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.2009 | (0.1729, 0.2301) | 0.0000 | 0.2009 | (0.1651, 0.2394) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 9 | 1 | 6 | 0.7500 | 0.9000 |
| proposed_vs_candidate_no_context | persona_consistency | 3 | 3 | 10 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | naturalness | 8 | 2 | 6 | 0.6875 | 0.8000 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 5 | 1 | 10 | 0.6250 | 0.8333 |
| proposed_vs_candidate_no_context | context_overlap | 10 | 0 | 6 | 0.8125 | 1.0000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 1 | 3 | 12 | 0.4375 | 0.2500 |
| proposed_vs_candidate_no_context | persona_style | 2 | 2 | 12 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | distinct1 | 5 | 5 | 6 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | length_score | 8 | 2 | 6 | 0.6875 | 0.8000 |
| proposed_vs_candidate_no_context | sentence_score | 5 | 3 | 8 | 0.5625 | 0.6250 |
| proposed_vs_candidate_no_context | overall_quality | 6 | 4 | 6 | 0.5625 | 0.6000 |
| controlled_vs_proposed_raw | context_relevance | 14 | 2 | 0 | 0.8750 | 0.8750 |
| controlled_vs_proposed_raw | persona_consistency | 15 | 0 | 1 | 0.9688 | 1.0000 |
| controlled_vs_proposed_raw | naturalness | 12 | 4 | 0 | 0.7500 | 0.7500 |
| controlled_vs_proposed_raw | context_keyword_coverage | 13 | 2 | 1 | 0.8438 | 0.8667 |
| controlled_vs_proposed_raw | context_overlap | 14 | 2 | 0 | 0.8750 | 0.8750 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 15 | 0 | 1 | 0.9688 | 1.0000 |
| controlled_vs_proposed_raw | persona_style | 6 | 3 | 7 | 0.5938 | 0.6667 |
| controlled_vs_proposed_raw | distinct1 | 6 | 10 | 0 | 0.3750 | 0.3750 |
| controlled_vs_proposed_raw | length_score | 11 | 4 | 1 | 0.7188 | 0.7333 |
| controlled_vs_proposed_raw | sentence_score | 8 | 1 | 7 | 0.7188 | 0.8889 |
| controlled_vs_proposed_raw | overall_quality | 15 | 1 | 0 | 0.9375 | 0.9375 |
| controlled_vs_candidate_no_context | context_relevance | 16 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 14 | 2 | 0 | 0.8750 | 0.8750 |
| controlled_vs_candidate_no_context | naturalness | 11 | 5 | 0 | 0.6875 | 0.6875 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 16 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 16 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 14 | 0 | 2 | 0.9375 | 1.0000 |
| controlled_vs_candidate_no_context | persona_style | 7 | 4 | 5 | 0.5938 | 0.6364 |
| controlled_vs_candidate_no_context | distinct1 | 7 | 9 | 0 | 0.4375 | 0.4375 |
| controlled_vs_candidate_no_context | length_score | 12 | 3 | 1 | 0.7812 | 0.8000 |
| controlled_vs_candidate_no_context | sentence_score | 10 | 1 | 5 | 0.7812 | 0.9091 |
| controlled_vs_candidate_no_context | overall_quality | 16 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 10 | 5 | 1 | 0.6562 | 0.6667 |
| controlled_alt_vs_controlled_default | persona_consistency | 3 | 5 | 8 | 0.4375 | 0.3750 |
| controlled_alt_vs_controlled_default | naturalness | 7 | 7 | 2 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 9 | 3 | 4 | 0.6875 | 0.7500 |
| controlled_alt_vs_controlled_default | context_overlap | 10 | 5 | 1 | 0.6562 | 0.6667 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 2 | 3 | 11 | 0.4688 | 0.4000 |
| controlled_alt_vs_controlled_default | persona_style | 2 | 6 | 8 | 0.3750 | 0.2500 |
| controlled_alt_vs_controlled_default | distinct1 | 6 | 7 | 3 | 0.4688 | 0.4615 |
| controlled_alt_vs_controlled_default | length_score | 4 | 8 | 4 | 0.3750 | 0.3333 |
| controlled_alt_vs_controlled_default | sentence_score | 2 | 3 | 11 | 0.4688 | 0.4000 |
| controlled_alt_vs_controlled_default | overall_quality | 10 | 5 | 1 | 0.6562 | 0.6667 |
| controlled_alt_vs_proposed_raw | context_relevance | 15 | 1 | 0 | 0.9375 | 0.9375 |
| controlled_alt_vs_proposed_raw | persona_consistency | 15 | 0 | 1 | 0.9688 | 1.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 8 | 8 | 0 | 0.5000 | 0.5000 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 15 | 0 | 1 | 0.9688 | 1.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 15 | 1 | 0 | 0.9375 | 0.9375 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 15 | 0 | 1 | 0.9688 | 1.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 6 | 3 | 7 | 0.5938 | 0.6667 |
| controlled_alt_vs_proposed_raw | distinct1 | 5 | 11 | 0 | 0.3125 | 0.3125 |
| controlled_alt_vs_proposed_raw | length_score | 7 | 9 | 0 | 0.4375 | 0.4375 |
| controlled_alt_vs_proposed_raw | sentence_score | 8 | 2 | 6 | 0.6875 | 0.8000 |
| controlled_alt_vs_proposed_raw | overall_quality | 16 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 16 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 14 | 1 | 1 | 0.9062 | 0.9333 |
| controlled_alt_vs_candidate_no_context | naturalness | 10 | 6 | 0 | 0.6250 | 0.6250 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 16 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 16 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 14 | 0 | 2 | 0.9375 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_style | 6 | 2 | 8 | 0.6250 | 0.7500 |
| controlled_alt_vs_candidate_no_context | distinct1 | 6 | 10 | 0 | 0.3750 | 0.3750 |
| controlled_alt_vs_candidate_no_context | length_score | 11 | 5 | 0 | 0.6875 | 0.6875 |
| controlled_alt_vs_candidate_no_context | sentence_score | 10 | 2 | 4 | 0.7500 | 0.8333 |
| controlled_alt_vs_candidate_no_context | overall_quality | 16 | 0 | 0 | 1.0000 | 1.0000 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.1875 | 0.6875 | 0.3125 |
| proposed_contextual_controlled_tuned | 0.0000 | 0.0000 | 0.0625 | 0.6250 | 0.3750 |
| proposed_contextual | 0.0000 | 0.0000 | 0.3750 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.4375 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `3`
- Unique template signatures: `16`
- Template signature ratio: `1.0000`
- Effective sample size by source clustering: `2.98`
- Effective sample size by template-signature clustering: `16.00`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: unavailable (disabled_by_flag).

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
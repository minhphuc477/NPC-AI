# Proposal Alignment Evaluation Report

- Run ID: `20260304T150052Z`
- Generated: `2026-03-04T15:05:28.404276+00:00`
- Scenarios: `artifacts\proposal_control_tuning\defaults_check\20260304T150052Z\scenarios.jsonl`
- Scenario count: `20`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off
- `baseline_no_context`: model `phi3:mini`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2716 (0.2391, 0.3089) | 0.3180 (0.2643, 0.3823) | 0.8569 (0.8295, 0.8857) | 0.3717 (0.3503, 0.3960) | 0.0906 |
| proposed_contextual | 0.0787 (0.0371, 0.1276) | 0.1192 (0.0837, 0.1622) | 0.7965 (0.7683, 0.8310) | 0.2157 (0.1866, 0.2494) | 0.0717 |
| candidate_no_context | 0.0167 (0.0125, 0.0239) | 0.1606 (0.1024, 0.2330) | 0.7787 (0.7582, 0.8040) | 0.1963 (0.1745, 0.2234) | 0.0331 |
| baseline_no_context | 0.0342 (0.0161, 0.0561) | 0.1752 (0.1317, 0.2227) | 0.8992 (0.8735, 0.9253) | 0.2302 (0.2121, 0.2491) | 0.0420 |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0620 | 3.7008 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0413 | -0.2574 |
| proposed_vs_candidate_no_context | naturalness | 0.0178 | 0.0229 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0848 | 18.6667 |
| proposed_vs_candidate_no_context | context_overlap | 0.0086 | 0.1900 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0548 | -0.7931 |
| proposed_vs_candidate_no_context | persona_style | 0.0124 | 0.0235 |
| proposed_vs_candidate_no_context | distinct1 | 0.0085 | 0.0092 |
| proposed_vs_candidate_no_context | length_score | 0.0633 | 0.3619 |
| proposed_vs_candidate_no_context | sentence_score | 0.0175 | 0.0237 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0386 | 1.1644 |
| proposed_vs_candidate_no_context | overall_quality | 0.0193 | 0.0984 |
| proposed_vs_baseline_no_context | context_relevance | 0.0445 | 1.3000 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0560 | -0.3196 |
| proposed_vs_baseline_no_context | naturalness | -0.1027 | -0.1142 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0591 | 1.9500 |
| proposed_vs_baseline_no_context | context_overlap | 0.0104 | 0.2404 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0510 | -0.7810 |
| proposed_vs_baseline_no_context | persona_style | -0.0762 | -0.1239 |
| proposed_vs_baseline_no_context | distinct1 | -0.0391 | -0.0402 |
| proposed_vs_baseline_no_context | length_score | -0.3567 | -0.5994 |
| proposed_vs_baseline_no_context | sentence_score | -0.1575 | -0.1726 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0297 | 0.7059 |
| proposed_vs_baseline_no_context | overall_quality | -0.0145 | -0.0631 |
| controlled_vs_proposed_raw | context_relevance | 0.1929 | 2.4509 |
| controlled_vs_proposed_raw | persona_consistency | 0.1988 | 1.6673 |
| controlled_vs_proposed_raw | naturalness | 0.0604 | 0.0758 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2494 | 2.7898 |
| controlled_vs_proposed_raw | context_overlap | 0.0612 | 1.1368 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2274 | 15.9167 |
| controlled_vs_proposed_raw | persona_style | 0.0844 | 0.1566 |
| controlled_vs_proposed_raw | distinct1 | -0.0017 | -0.0018 |
| controlled_vs_proposed_raw | length_score | 0.2117 | 0.8881 |
| controlled_vs_proposed_raw | sentence_score | 0.1925 | 0.2550 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0189 | 0.2630 |
| controlled_vs_proposed_raw | overall_quality | 0.1560 | 0.7234 |
| controlled_vs_candidate_no_context | context_relevance | 0.2549 | 15.2219 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1574 | 0.9807 |
| controlled_vs_candidate_no_context | naturalness | 0.0782 | 0.1005 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3342 | 73.5333 |
| controlled_vs_candidate_no_context | context_overlap | 0.0697 | 1.5427 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1726 | 2.5000 |
| controlled_vs_candidate_no_context | persona_style | 0.0968 | 0.1838 |
| controlled_vs_candidate_no_context | distinct1 | 0.0069 | 0.0074 |
| controlled_vs_candidate_no_context | length_score | 0.2750 | 1.5714 |
| controlled_vs_candidate_no_context | sentence_score | 0.2100 | 0.2847 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0574 | 1.7336 |
| controlled_vs_candidate_no_context | overall_quality | 0.1753 | 0.8930 |
| controlled_vs_baseline_no_context | context_relevance | 0.2374 | 6.9371 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1428 | 0.8149 |
| controlled_vs_baseline_no_context | naturalness | -0.0423 | -0.0470 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3085 | 10.1800 |
| controlled_vs_baseline_no_context | context_overlap | 0.0716 | 1.6505 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1764 | 2.7044 |
| controlled_vs_baseline_no_context | persona_style | 0.0082 | 0.0133 |
| controlled_vs_baseline_no_context | distinct1 | -0.0407 | -0.0419 |
| controlled_vs_baseline_no_context | length_score | -0.1450 | -0.2437 |
| controlled_vs_baseline_no_context | sentence_score | 0.0350 | 0.0384 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0485 | 1.1546 |
| controlled_vs_baseline_no_context | overall_quality | 0.1415 | 0.6147 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2374 | 6.9371 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1428 | 0.8149 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0423 | -0.0470 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3085 | 10.1800 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0716 | 1.6505 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1764 | 2.7044 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0082 | 0.0133 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0407 | -0.0419 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | -0.1450 | -0.2437 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0350 | 0.0384 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0485 | 1.1546 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1415 | 0.6147 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0620 | (0.0215, 0.1096) | 0.0000 | 0.0620 | (0.0240, 0.1069) | 0.0003 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0413 | (-0.1257, 0.0255) | 0.8473 | -0.0413 | (-0.1522, 0.0272) | 0.8490 |
| proposed_vs_candidate_no_context | naturalness | 0.0178 | (-0.0076, 0.0473) | 0.0913 | 0.0178 | (-0.0090, 0.0442) | 0.0940 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0848 | (0.0307, 0.1527) | 0.0000 | 0.0848 | (0.0316, 0.1470) | 0.0003 |
| proposed_vs_candidate_no_context | context_overlap | 0.0086 | (0.0006, 0.0178) | 0.0157 | 0.0086 | (0.0011, 0.0190) | 0.0067 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0548 | (-0.1429, 0.0143) | 0.9397 | -0.0548 | (-0.1722, 0.0160) | 0.9297 |
| proposed_vs_candidate_no_context | persona_style | 0.0124 | (-0.0667, 0.0993) | 0.3913 | 0.0124 | (-0.0822, 0.0959) | 0.3803 |
| proposed_vs_candidate_no_context | distinct1 | 0.0085 | (-0.0057, 0.0235) | 0.1277 | 0.0085 | (-0.0072, 0.0222) | 0.1260 |
| proposed_vs_candidate_no_context | length_score | 0.0633 | (-0.0250, 0.1683) | 0.0963 | 0.0633 | (-0.0180, 0.1646) | 0.0773 |
| proposed_vs_candidate_no_context | sentence_score | 0.0175 | (-0.0525, 0.0875) | 0.3943 | 0.0175 | (-0.0700, 0.0875) | 0.4037 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0386 | (0.0124, 0.0711) | 0.0000 | 0.0386 | (0.0116, 0.0885) | 0.0000 |
| proposed_vs_candidate_no_context | overall_quality | 0.0193 | (-0.0187, 0.0597) | 0.1567 | 0.0193 | (-0.0243, 0.0593) | 0.1743 |
| proposed_vs_baseline_no_context | context_relevance | 0.0445 | (0.0076, 0.0892) | 0.0060 | 0.0445 | (0.0051, 0.0862) | 0.0103 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0560 | (-0.1233, 0.0015) | 0.9693 | -0.0560 | (-0.1392, 0.0050) | 0.9593 |
| proposed_vs_baseline_no_context | naturalness | -0.1027 | (-0.1444, -0.0596) | 1.0000 | -0.1027 | (-0.1490, -0.0455) | 0.9997 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0591 | (0.0091, 0.1220) | 0.0060 | 0.0591 | (0.0096, 0.1148) | 0.0093 |
| proposed_vs_baseline_no_context | context_overlap | 0.0104 | (-0.0005, 0.0224) | 0.0340 | 0.0104 | (-0.0040, 0.0274) | 0.0880 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0510 | (-0.1241, 0.0138) | 0.9317 | -0.0510 | (-0.1423, 0.0125) | 0.9400 |
| proposed_vs_baseline_no_context | persona_style | -0.0762 | (-0.1731, -0.0022) | 0.9880 | -0.0762 | (-0.2112, -0.0006) | 0.9820 |
| proposed_vs_baseline_no_context | distinct1 | -0.0391 | (-0.0519, -0.0271) | 1.0000 | -0.0391 | (-0.0514, -0.0239) | 1.0000 |
| proposed_vs_baseline_no_context | length_score | -0.3567 | (-0.5250, -0.1733) | 1.0000 | -0.3567 | (-0.5270, -0.1117) | 0.9960 |
| proposed_vs_baseline_no_context | sentence_score | -0.1575 | (-0.2450, -0.0525) | 0.9983 | -0.1575 | (-0.2579, -0.0250) | 0.9953 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0297 | (0.0017, 0.0614) | 0.0203 | 0.0297 | (0.0011, 0.0632) | 0.0227 |
| proposed_vs_baseline_no_context | overall_quality | -0.0145 | (-0.0526, 0.0235) | 0.7900 | -0.0145 | (-0.0510, 0.0220) | 0.7880 |
| controlled_vs_proposed_raw | context_relevance | 0.1929 | (0.1397, 0.2448) | 0.0000 | 0.1929 | (0.1366, 0.2376) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.1988 | (0.1437, 0.2659) | 0.0000 | 0.1988 | (0.1585, 0.2309) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0604 | (0.0139, 0.1020) | 0.0060 | 0.0604 | (-0.0009, 0.1094) | 0.0260 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2494 | (0.1755, 0.3189) | 0.0000 | 0.2494 | (0.1785, 0.3018) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0612 | (0.0459, 0.0763) | 0.0000 | 0.0612 | (0.0373, 0.0805) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2274 | (0.1617, 0.3057) | 0.0000 | 0.2274 | (0.1810, 0.2780) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.0844 | (-0.0063, 0.1891) | 0.0460 | 0.0844 | (-0.0125, 0.2273) | 0.0977 |
| controlled_vs_proposed_raw | distinct1 | -0.0017 | (-0.0263, 0.0215) | 0.5653 | -0.0017 | (-0.0306, 0.0231) | 0.5773 |
| controlled_vs_proposed_raw | length_score | 0.2117 | (0.0300, 0.3917) | 0.0120 | 0.2117 | (-0.0167, 0.4261) | 0.0340 |
| controlled_vs_proposed_raw | sentence_score | 0.1925 | (0.1225, 0.2625) | 0.0000 | 0.1925 | (0.0875, 0.2771) | 0.0000 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0189 | (-0.0160, 0.0497) | 0.1357 | 0.0189 | (-0.0146, 0.0431) | 0.1130 |
| controlled_vs_proposed_raw | overall_quality | 0.1560 | (0.1194, 0.1941) | 0.0000 | 0.1560 | (0.1235, 0.1865) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2549 | (0.2210, 0.2908) | 0.0000 | 0.2549 | (0.2328, 0.2749) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1574 | (0.0601, 0.2452) | 0.0020 | 0.1574 | (0.0300, 0.2348) | 0.0120 |
| controlled_vs_candidate_no_context | naturalness | 0.0782 | (0.0391, 0.1126) | 0.0000 | 0.0782 | (0.0266, 0.1145) | 0.0020 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3342 | (0.2894, 0.3864) | 0.0000 | 0.3342 | (0.3071, 0.3615) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0697 | (0.0550, 0.0836) | 0.0000 | 0.0697 | (0.0492, 0.0856) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1726 | (0.0588, 0.2817) | 0.0007 | 0.1726 | (0.0367, 0.2702) | 0.0087 |
| controlled_vs_candidate_no_context | persona_style | 0.0968 | (-0.0255, 0.2297) | 0.0647 | 0.0968 | (-0.0778, 0.3029) | 0.1927 |
| controlled_vs_candidate_no_context | distinct1 | 0.0069 | (-0.0132, 0.0260) | 0.2667 | 0.0069 | (-0.0198, 0.0272) | 0.3060 |
| controlled_vs_candidate_no_context | length_score | 0.2750 | (0.1100, 0.4317) | 0.0000 | 0.2750 | (0.1119, 0.4400) | 0.0013 |
| controlled_vs_candidate_no_context | sentence_score | 0.2100 | (0.1400, 0.2800) | 0.0000 | 0.2100 | (0.0778, 0.3023) | 0.0027 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0574 | (0.0371, 0.0785) | 0.0000 | 0.0574 | (0.0396, 0.0814) | 0.0000 |
| controlled_vs_candidate_no_context | overall_quality | 0.1753 | (0.1398, 0.2098) | 0.0000 | 0.1753 | (0.1292, 0.2013) | 0.0000 |
| controlled_vs_baseline_no_context | context_relevance | 0.2374 | (0.2011, 0.2788) | 0.0000 | 0.2374 | (0.1997, 0.2704) | 0.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1428 | (0.0681, 0.2242) | 0.0000 | 0.1428 | (0.0612, 0.1990) | 0.0003 |
| controlled_vs_baseline_no_context | naturalness | -0.0423 | (-0.0811, -0.0013) | 0.9777 | -0.0423 | (-0.0692, -0.0081) | 0.9933 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3085 | (0.2568, 0.3638) | 0.0000 | 0.3085 | (0.2555, 0.3550) | 0.0000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0716 | (0.0598, 0.0828) | 0.0000 | 0.0716 | (0.0623, 0.0783) | 0.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1764 | (0.0874, 0.2752) | 0.0000 | 0.1764 | (0.0756, 0.2543) | 0.0007 |
| controlled_vs_baseline_no_context | persona_style | 0.0082 | (-0.0368, 0.0682) | 0.4350 | 0.0082 | (-0.0273, 0.0554) | 0.3640 |
| controlled_vs_baseline_no_context | distinct1 | -0.0407 | (-0.0593, -0.0229) | 1.0000 | -0.0407 | (-0.0594, -0.0239) | 1.0000 |
| controlled_vs_baseline_no_context | length_score | -0.1450 | (-0.3317, 0.0450) | 0.9253 | -0.1450 | (-0.2972, 0.0667) | 0.9137 |
| controlled_vs_baseline_no_context | sentence_score | 0.0350 | (-0.0350, 0.1225) | 0.2710 | 0.0350 | (-0.0875, 0.1346) | 0.3063 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0485 | (0.0266, 0.0709) | 0.0000 | 0.0485 | (0.0281, 0.0633) | 0.0000 |
| controlled_vs_baseline_no_context | overall_quality | 0.1415 | (0.1101, 0.1745) | 0.0000 | 0.1415 | (0.1087, 0.1629) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2374 | (0.2015, 0.2789) | 0.0000 | 0.2374 | (0.1968, 0.2702) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1428 | (0.0684, 0.2221) | 0.0000 | 0.1428 | (0.0550, 0.2005) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0423 | (-0.0827, -0.0021) | 0.9803 | -0.0423 | (-0.0683, -0.0079) | 0.9953 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3085 | (0.2591, 0.3667) | 0.0000 | 0.3085 | (0.2536, 0.3564) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0716 | (0.0598, 0.0826) | 0.0000 | 0.0716 | (0.0622, 0.0786) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1764 | (0.0831, 0.2745) | 0.0000 | 0.1764 | (0.0752, 0.2526) | 0.0003 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0082 | (-0.0383, 0.0682) | 0.4223 | 0.0082 | (-0.0275, 0.0538) | 0.3600 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0407 | (-0.0596, -0.0219) | 1.0000 | -0.0407 | (-0.0583, -0.0249) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | -0.1450 | (-0.3317, 0.0417) | 0.9360 | -0.1450 | (-0.2955, 0.0632) | 0.9133 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0350 | (-0.0525, 0.1225) | 0.2747 | 0.0350 | (-0.0876, 0.1333) | 0.3223 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0485 | (0.0264, 0.0702) | 0.0000 | 0.0485 | (0.0280, 0.0633) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1415 | (0.1115, 0.1745) | 0.0000 | 0.1415 | (0.1098, 0.1627) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 9 | 1 | 10 | 0.7000 | 0.9000 |
| proposed_vs_candidate_no_context | persona_consistency | 3 | 3 | 14 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | naturalness | 7 | 2 | 11 | 0.6250 | 0.7778 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 8 | 0 | 12 | 0.7000 | 1.0000 |
| proposed_vs_candidate_no_context | context_overlap | 7 | 3 | 10 | 0.6000 | 0.7000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 1 | 3 | 16 | 0.4500 | 0.2500 |
| proposed_vs_candidate_no_context | persona_style | 2 | 2 | 16 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | distinct1 | 6 | 2 | 12 | 0.6000 | 0.7500 |
| proposed_vs_candidate_no_context | length_score | 5 | 4 | 11 | 0.5250 | 0.5556 |
| proposed_vs_candidate_no_context | sentence_score | 3 | 2 | 15 | 0.5250 | 0.6000 |
| proposed_vs_candidate_no_context | bertscore_f1 | 10 | 2 | 8 | 0.7000 | 0.8333 |
| proposed_vs_candidate_no_context | overall_quality | 8 | 4 | 8 | 0.6000 | 0.6667 |
| proposed_vs_baseline_no_context | context_relevance | 11 | 9 | 0 | 0.5500 | 0.5500 |
| proposed_vs_baseline_no_context | persona_consistency | 2 | 5 | 13 | 0.4250 | 0.2857 |
| proposed_vs_baseline_no_context | naturalness | 5 | 15 | 0 | 0.2500 | 0.2500 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 7 | 2 | 11 | 0.6250 | 0.7778 |
| proposed_vs_baseline_no_context | context_overlap | 10 | 10 | 0 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 1 | 5 | 14 | 0.4000 | 0.1667 |
| proposed_vs_baseline_no_context | persona_style | 1 | 4 | 15 | 0.4250 | 0.2000 |
| proposed_vs_baseline_no_context | distinct1 | 1 | 16 | 3 | 0.1250 | 0.0588 |
| proposed_vs_baseline_no_context | length_score | 4 | 16 | 0 | 0.2000 | 0.2000 |
| proposed_vs_baseline_no_context | sentence_score | 2 | 11 | 7 | 0.2750 | 0.1538 |
| proposed_vs_baseline_no_context | bertscore_f1 | 14 | 6 | 0 | 0.7000 | 0.7000 |
| proposed_vs_baseline_no_context | overall_quality | 7 | 13 | 0 | 0.3500 | 0.3500 |
| controlled_vs_proposed_raw | context_relevance | 18 | 2 | 0 | 0.9000 | 0.9000 |
| controlled_vs_proposed_raw | persona_consistency | 18 | 1 | 1 | 0.9250 | 0.9474 |
| controlled_vs_proposed_raw | naturalness | 13 | 7 | 0 | 0.6500 | 0.6500 |
| controlled_vs_proposed_raw | context_keyword_coverage | 17 | 1 | 2 | 0.9000 | 0.9444 |
| controlled_vs_proposed_raw | context_overlap | 18 | 1 | 1 | 0.9250 | 0.9474 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 18 | 0 | 2 | 0.9500 | 1.0000 |
| controlled_vs_proposed_raw | persona_style | 3 | 1 | 16 | 0.5500 | 0.7500 |
| controlled_vs_proposed_raw | distinct1 | 10 | 9 | 1 | 0.5250 | 0.5263 |
| controlled_vs_proposed_raw | length_score | 11 | 9 | 0 | 0.5500 | 0.5500 |
| controlled_vs_proposed_raw | sentence_score | 11 | 0 | 9 | 0.7750 | 1.0000 |
| controlled_vs_proposed_raw | bertscore_f1 | 16 | 4 | 0 | 0.8000 | 0.8000 |
| controlled_vs_proposed_raw | overall_quality | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_vs_candidate_no_context | context_relevance | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 16 | 3 | 1 | 0.8250 | 0.8421 |
| controlled_vs_candidate_no_context | naturalness | 18 | 2 | 0 | 0.9000 | 0.9000 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 19 | 0 | 1 | 0.9750 | 1.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 16 | 3 | 1 | 0.8250 | 0.8421 |
| controlled_vs_candidate_no_context | persona_style | 5 | 2 | 13 | 0.5750 | 0.7143 |
| controlled_vs_candidate_no_context | distinct1 | 10 | 9 | 1 | 0.5250 | 0.5263 |
| controlled_vs_candidate_no_context | length_score | 15 | 5 | 0 | 0.7500 | 0.7500 |
| controlled_vs_candidate_no_context | sentence_score | 12 | 0 | 8 | 0.8000 | 1.0000 |
| controlled_vs_candidate_no_context | bertscore_f1 | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_vs_candidate_no_context | overall_quality | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_vs_baseline_no_context | context_relevance | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 14 | 2 | 4 | 0.8000 | 0.8750 |
| controlled_vs_baseline_no_context | naturalness | 5 | 15 | 0 | 0.2500 | 0.2500 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | context_overlap | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 14 | 1 | 5 | 0.8250 | 0.9333 |
| controlled_vs_baseline_no_context | persona_style | 2 | 2 | 16 | 0.5000 | 0.5000 |
| controlled_vs_baseline_no_context | distinct1 | 2 | 17 | 1 | 0.1250 | 0.1053 |
| controlled_vs_baseline_no_context | length_score | 8 | 12 | 0 | 0.4000 | 0.4000 |
| controlled_vs_baseline_no_context | sentence_score | 4 | 2 | 14 | 0.5500 | 0.6667 |
| controlled_vs_baseline_no_context | bertscore_f1 | 16 | 4 | 0 | 0.8000 | 0.8000 |
| controlled_vs_baseline_no_context | overall_quality | 20 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 20 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 14 | 2 | 4 | 0.8000 | 0.8750 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 5 | 15 | 0 | 0.2500 | 0.2500 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 20 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 20 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 14 | 1 | 5 | 0.8250 | 0.9333 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 2 | 2 | 16 | 0.5000 | 0.5000 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | 2 | 17 | 1 | 0.1250 | 0.1053 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 8 | 12 | 0 | 0.4000 | 0.4000 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 4 | 2 | 14 | 0.5500 | 0.6667 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 16 | 4 | 0 | 0.8000 | 0.8000 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 20 | 0 | 0 | 1.0000 | 1.0000 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.2500 | 0.4000 | 0.6000 |
| proposed_contextual | 0.0000 | 0.0000 | 0.5500 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.7000 | 0.0000 | 0.0000 |
| baseline_no_context | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `8`
- Unique template signatures: `18`
- Template signature ratio: `0.9000`
- Effective sample size by source clustering: `6.67`
- Effective sample size by template-signature clustering: `16.67`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: enabled.

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
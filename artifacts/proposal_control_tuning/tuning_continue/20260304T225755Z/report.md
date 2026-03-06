# Proposal Alignment Evaluation Report

- Run ID: `20260304T225755Z`
- Generated: `2026-03-04T23:03:03.897715+00:00`
- Scenarios: `artifacts\proposal_control_tuning\tuning_continue\20260304T225755Z\scenarios.jsonl`
- Scenario count: `20`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off
- `baseline_no_context`: model `phi3:mini`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2858 (0.2416, 0.3404) | 0.4468 (0.3663, 0.5354) | 0.8690 (0.8310, 0.9041) | 0.4210 (0.3920, 0.4509) | 0.0774 |
| proposed_contextual | 0.0781 (0.0352, 0.1277) | 0.1506 (0.1125, 0.1911) | 0.7966 (0.7698, 0.8290) | 0.2255 (0.1958, 0.2606) | 0.0714 |
| candidate_no_context | 0.0195 (0.0132, 0.0284) | 0.1633 (0.1159, 0.2173) | 0.7926 (0.7646, 0.8248) | 0.2022 (0.1827, 0.2232) | 0.0437 |
| baseline_no_context | 0.0264 (0.0138, 0.0407) | 0.1638 (0.1260, 0.2034) | 0.8874 (0.8618, 0.9132) | 0.2199 (0.2039, 0.2363) | 0.0282 |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0586 | 3.0058 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0127 | -0.0778 |
| proposed_vs_candidate_no_context | naturalness | 0.0040 | 0.0051 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0769 | 8.4583 |
| proposed_vs_candidate_no_context | context_overlap | 0.0160 | 0.3662 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0226 | -0.3094 |
| proposed_vs_candidate_no_context | persona_style | 0.0269 | 0.0514 |
| proposed_vs_candidate_no_context | distinct1 | 0.0060 | 0.0064 |
| proposed_vs_candidate_no_context | length_score | 0.0333 | 0.1600 |
| proposed_vs_candidate_no_context | sentence_score | -0.0500 | -0.0647 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0277 | 0.6325 |
| proposed_vs_candidate_no_context | overall_quality | 0.0232 | 0.1149 |
| proposed_vs_baseline_no_context | context_relevance | 0.0517 | 1.9554 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0132 | -0.0805 |
| proposed_vs_baseline_no_context | naturalness | -0.0908 | -0.1024 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0633 | 2.7833 |
| proposed_vs_baseline_no_context | context_overlap | 0.0248 | 0.7050 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0050 | -0.0901 |
| proposed_vs_baseline_no_context | persona_style | -0.0459 | -0.0770 |
| proposed_vs_baseline_no_context | distinct1 | -0.0344 | -0.0353 |
| proposed_vs_baseline_no_context | length_score | -0.3167 | -0.5672 |
| proposed_vs_baseline_no_context | sentence_score | -0.1375 | -0.1599 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0432 | 1.5300 |
| proposed_vs_baseline_no_context | overall_quality | 0.0056 | 0.0254 |
| controlled_vs_proposed_raw | context_relevance | 0.2077 | 2.6572 |
| controlled_vs_proposed_raw | persona_consistency | 0.2962 | 1.9671 |
| controlled_vs_proposed_raw | naturalness | 0.0724 | 0.0909 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2733 | 3.1789 |
| controlled_vs_proposed_raw | context_overlap | 0.0544 | 0.9086 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.3531 | 6.9953 |
| controlled_vs_proposed_raw | persona_style | 0.0687 | 0.1247 |
| controlled_vs_proposed_raw | distinct1 | -0.0081 | -0.0086 |
| controlled_vs_proposed_raw | length_score | 0.2850 | 1.1793 |
| controlled_vs_proposed_raw | sentence_score | 0.1900 | 0.2630 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0060 | 0.0844 |
| controlled_vs_proposed_raw | overall_quality | 0.1956 | 0.8674 |
| controlled_vs_candidate_no_context | context_relevance | 0.2663 | 13.6500 |
| controlled_vs_candidate_no_context | persona_consistency | 0.2835 | 1.7362 |
| controlled_vs_candidate_no_context | naturalness | 0.0765 | 0.0965 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3502 | 38.5250 |
| controlled_vs_candidate_no_context | context_overlap | 0.0704 | 1.6075 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.3305 | 4.5212 |
| controlled_vs_candidate_no_context | persona_style | 0.0957 | 0.1825 |
| controlled_vs_candidate_no_context | distinct1 | -0.0021 | -0.0023 |
| controlled_vs_candidate_no_context | length_score | 0.3183 | 1.5280 |
| controlled_vs_candidate_no_context | sentence_score | 0.1400 | 0.1812 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0337 | 0.7703 |
| controlled_vs_candidate_no_context | overall_quality | 0.2188 | 1.0819 |
| controlled_vs_baseline_no_context | context_relevance | 0.2594 | 9.8085 |
| controlled_vs_baseline_no_context | persona_consistency | 0.2830 | 1.7281 |
| controlled_vs_baseline_no_context | naturalness | -0.0184 | -0.0207 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3366 | 14.8100 |
| controlled_vs_baseline_no_context | context_overlap | 0.0791 | 2.2541 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.3481 | 6.2747 |
| controlled_vs_baseline_no_context | persona_style | 0.0228 | 0.0382 |
| controlled_vs_baseline_no_context | distinct1 | -0.0425 | -0.0436 |
| controlled_vs_baseline_no_context | length_score | -0.0317 | -0.0567 |
| controlled_vs_baseline_no_context | sentence_score | 0.0525 | 0.0610 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0492 | 1.7434 |
| controlled_vs_baseline_no_context | overall_quality | 0.2012 | 0.9149 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2594 | 9.8085 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.2830 | 1.7281 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0184 | -0.0207 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3366 | 14.8100 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0791 | 2.2541 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.3481 | 6.2747 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0228 | 0.0382 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0425 | -0.0436 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | -0.0317 | -0.0567 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0525 | 0.0610 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0492 | 1.7434 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.2012 | 0.9149 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0586 | (0.0204, 0.1079) | 0.0000 | 0.0586 | (0.0165, 0.1062) | 0.0000 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0127 | (-0.0656, 0.0401) | 0.6810 | -0.0127 | (-0.0490, 0.0103) | 0.8233 |
| proposed_vs_candidate_no_context | naturalness | 0.0040 | (-0.0325, 0.0395) | 0.4223 | 0.0040 | (-0.0279, 0.0466) | 0.4180 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0769 | (0.0265, 0.1402) | 0.0000 | 0.0769 | (0.0248, 0.1405) | 0.0000 |
| proposed_vs_candidate_no_context | context_overlap | 0.0160 | (0.0032, 0.0309) | 0.0063 | 0.0160 | (0.0027, 0.0299) | 0.0077 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0226 | (-0.0902, 0.0433) | 0.7540 | -0.0226 | (-0.0661, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | persona_style | 0.0269 | (0.0000, 0.0667) | 0.0360 | 0.0269 | (0.0000, 0.0753) | 0.1040 |
| proposed_vs_candidate_no_context | distinct1 | 0.0060 | (-0.0119, 0.0243) | 0.2450 | 0.0060 | (-0.0096, 0.0284) | 0.2620 |
| proposed_vs_candidate_no_context | length_score | 0.0333 | (-0.1050, 0.1783) | 0.3223 | 0.0333 | (-0.0928, 0.1912) | 0.3177 |
| proposed_vs_candidate_no_context | sentence_score | -0.0500 | (-0.1375, 0.0350) | 0.8757 | -0.0500 | (-0.1225, 0.0211) | 0.9300 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0277 | (-0.0016, 0.0606) | 0.0323 | 0.0277 | (-0.0017, 0.0606) | 0.0327 |
| proposed_vs_candidate_no_context | overall_quality | 0.0232 | (-0.0048, 0.0577) | 0.0557 | 0.0232 | (-0.0016, 0.0504) | 0.0350 |
| proposed_vs_baseline_no_context | context_relevance | 0.0517 | (0.0060, 0.1056) | 0.0110 | 0.0517 | (0.0019, 0.1065) | 0.0223 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0132 | (-0.0582, 0.0369) | 0.7277 | -0.0132 | (-0.0600, 0.0490) | 0.6893 |
| proposed_vs_baseline_no_context | naturalness | -0.0908 | (-0.1328, -0.0406) | 1.0000 | -0.0908 | (-0.1460, -0.0134) | 0.9893 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0633 | (0.0042, 0.1318) | 0.0180 | 0.0633 | (-0.0006, 0.1384) | 0.0303 |
| proposed_vs_baseline_no_context | context_overlap | 0.0248 | (0.0080, 0.0413) | 0.0017 | 0.0248 | (0.0068, 0.0430) | 0.0013 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0050 | (-0.0586, 0.0483) | 0.5810 | -0.0050 | (-0.0560, 0.0602) | 0.5587 |
| proposed_vs_baseline_no_context | persona_style | -0.0459 | (-0.1607, 0.0556) | 0.7880 | -0.0459 | (-0.1875, 0.0721) | 0.7370 |
| proposed_vs_baseline_no_context | distinct1 | -0.0344 | (-0.0517, -0.0163) | 0.9997 | -0.0344 | (-0.0557, -0.0075) | 0.9970 |
| proposed_vs_baseline_no_context | length_score | -0.3167 | (-0.4883, -0.1300) | 0.9993 | -0.3167 | (-0.5196, -0.0333) | 0.9853 |
| proposed_vs_baseline_no_context | sentence_score | -0.1375 | (-0.2425, -0.0325) | 0.9923 | -0.1375 | (-0.2472, -0.0233) | 0.9930 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0432 | (0.0063, 0.0829) | 0.0107 | 0.0432 | (0.0014, 0.0900) | 0.0233 |
| proposed_vs_baseline_no_context | overall_quality | 0.0056 | (-0.0296, 0.0422) | 0.4007 | 0.0056 | (-0.0324, 0.0529) | 0.4057 |
| controlled_vs_proposed_raw | context_relevance | 0.2077 | (0.1405, 0.2809) | 0.0000 | 0.2077 | (0.1470, 0.2668) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.2962 | (0.2044, 0.3915) | 0.0000 | 0.2962 | (0.1831, 0.3995) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0724 | (0.0252, 0.1169) | 0.0020 | 0.0724 | (0.0093, 0.1155) | 0.0107 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2733 | (0.1848, 0.3670) | 0.0000 | 0.2733 | (0.1925, 0.3537) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0544 | (0.0348, 0.0737) | 0.0000 | 0.0544 | (0.0353, 0.0713) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.3531 | (0.2393, 0.4800) | 0.0000 | 0.3531 | (0.2083, 0.4960) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.0687 | (-0.0052, 0.1672) | 0.0593 | 0.0687 | (-0.0105, 0.2106) | 0.1393 |
| controlled_vs_proposed_raw | distinct1 | -0.0081 | (-0.0303, 0.0143) | 0.7613 | -0.0081 | (-0.0358, 0.0108) | 0.7857 |
| controlled_vs_proposed_raw | length_score | 0.2850 | (0.0883, 0.4750) | 0.0023 | 0.2850 | (0.0333, 0.4605) | 0.0153 |
| controlled_vs_proposed_raw | sentence_score | 0.1900 | (0.1200, 0.2625) | 0.0000 | 0.1900 | (0.1206, 0.2375) | 0.0000 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0060 | (-0.0302, 0.0374) | 0.3753 | 0.0060 | (-0.0365, 0.0480) | 0.3823 |
| controlled_vs_proposed_raw | overall_quality | 0.1956 | (0.1472, 0.2437) | 0.0000 | 0.1956 | (0.1359, 0.2434) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2663 | (0.2233, 0.3199) | 0.0000 | 0.2663 | (0.2290, 0.3118) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.2835 | (0.1921, 0.3765) | 0.0000 | 0.2835 | (0.1703, 0.3919) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.0765 | (0.0234, 0.1281) | 0.0030 | 0.0765 | (-0.0026, 0.1289) | 0.0267 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3502 | (0.2909, 0.4215) | 0.0000 | 0.3502 | (0.2941, 0.4115) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0704 | (0.0552, 0.0853) | 0.0000 | 0.0704 | (0.0613, 0.0773) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.3305 | (0.2164, 0.4488) | 0.0000 | 0.3305 | (0.1821, 0.4781) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.0957 | (0.0162, 0.1969) | 0.0047 | 0.0957 | (0.0037, 0.2252) | 0.0240 |
| controlled_vs_candidate_no_context | distinct1 | -0.0021 | (-0.0268, 0.0192) | 0.5557 | -0.0021 | (-0.0236, 0.0103) | 0.5870 |
| controlled_vs_candidate_no_context | length_score | 0.3183 | (0.1033, 0.5217) | 0.0010 | 0.3183 | (0.0229, 0.5288) | 0.0217 |
| controlled_vs_candidate_no_context | sentence_score | 0.1400 | (0.0350, 0.2275) | 0.0060 | 0.1400 | (0.0250, 0.2188) | 0.0123 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0337 | (0.0127, 0.0529) | 0.0007 | 0.0337 | (0.0109, 0.0598) | 0.0007 |
| controlled_vs_candidate_no_context | overall_quality | 0.2188 | (0.1867, 0.2523) | 0.0000 | 0.2188 | (0.1757, 0.2498) | 0.0000 |
| controlled_vs_baseline_no_context | context_relevance | 0.2594 | (0.2127, 0.3126) | 0.0000 | 0.2594 | (0.2145, 0.3122) | 0.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 0.2830 | (0.2003, 0.3758) | 0.0000 | 0.2830 | (0.1934, 0.3689) | 0.0000 |
| controlled_vs_baseline_no_context | naturalness | -0.0184 | (-0.0657, 0.0280) | 0.7797 | -0.0184 | (-0.0796, 0.0321) | 0.7633 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3366 | (0.2754, 0.4092) | 0.0000 | 0.3366 | (0.2738, 0.4093) | 0.0000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0791 | (0.0652, 0.0926) | 0.0000 | 0.0791 | (0.0660, 0.0970) | 0.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.3481 | (0.2426, 0.4664) | 0.0000 | 0.3481 | (0.2348, 0.4585) | 0.0000 |
| controlled_vs_baseline_no_context | persona_style | 0.0228 | (-0.0178, 0.0801) | 0.2437 | 0.0228 | (0.0000, 0.0804) | 0.3500 |
| controlled_vs_baseline_no_context | distinct1 | -0.0425 | (-0.0613, -0.0227) | 1.0000 | -0.0425 | (-0.0601, -0.0276) | 1.0000 |
| controlled_vs_baseline_no_context | length_score | -0.0317 | (-0.2183, 0.1533) | 0.6327 | -0.0317 | (-0.2704, 0.1884) | 0.6220 |
| controlled_vs_baseline_no_context | sentence_score | 0.0525 | (-0.0525, 0.1575) | 0.2030 | 0.0525 | (-0.0467, 0.1500) | 0.1857 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0492 | (0.0151, 0.0829) | 0.0027 | 0.0492 | (0.0119, 0.0916) | 0.0027 |
| controlled_vs_baseline_no_context | overall_quality | 0.2012 | (0.1687, 0.2329) | 0.0000 | 0.2012 | (0.1820, 0.2166) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2594 | (0.2124, 0.3130) | 0.0000 | 0.2594 | (0.2140, 0.3167) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.2830 | (0.1988, 0.3802) | 0.0000 | 0.2830 | (0.1933, 0.3668) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | -0.0184 | (-0.0670, 0.0281) | 0.7710 | -0.0184 | (-0.0787, 0.0325) | 0.7623 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3366 | (0.2723, 0.4103) | 0.0000 | 0.3366 | (0.2740, 0.4087) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0791 | (0.0646, 0.0930) | 0.0000 | 0.0791 | (0.0660, 0.0974) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.3481 | (0.2407, 0.4674) | 0.0000 | 0.3481 | (0.2368, 0.4578) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0228 | (-0.0178, 0.0801) | 0.2393 | 0.0228 | (0.0000, 0.0760) | 0.3413 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0425 | (-0.0616, -0.0228) | 1.0000 | -0.0425 | (-0.0603, -0.0274) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | -0.0317 | (-0.2184, 0.1500) | 0.6327 | -0.0317 | (-0.2702, 0.1930) | 0.6383 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0525 | (-0.0525, 0.1575) | 0.1937 | 0.0525 | (-0.0525, 0.1474) | 0.2107 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0492 | (0.0144, 0.0835) | 0.0023 | 0.0492 | (0.0123, 0.0938) | 0.0020 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.2012 | (0.1683, 0.2333) | 0.0000 | 0.2012 | (0.1823, 0.2157) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 11 | 2 | 7 | 0.7250 | 0.8462 |
| proposed_vs_candidate_no_context | persona_consistency | 6 | 5 | 9 | 0.5250 | 0.5455 |
| proposed_vs_candidate_no_context | naturalness | 8 | 5 | 7 | 0.5750 | 0.6154 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 7 | 0 | 13 | 0.6750 | 1.0000 |
| proposed_vs_candidate_no_context | context_overlap | 11 | 2 | 7 | 0.7250 | 0.8462 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 3 | 5 | 12 | 0.4500 | 0.3750 |
| proposed_vs_candidate_no_context | persona_style | 3 | 0 | 17 | 0.5750 | 1.0000 |
| proposed_vs_candidate_no_context | distinct1 | 8 | 4 | 8 | 0.6000 | 0.6667 |
| proposed_vs_candidate_no_context | length_score | 6 | 5 | 9 | 0.5250 | 0.5455 |
| proposed_vs_candidate_no_context | sentence_score | 2 | 5 | 13 | 0.4250 | 0.2857 |
| proposed_vs_candidate_no_context | bertscore_f1 | 9 | 10 | 1 | 0.4750 | 0.4737 |
| proposed_vs_candidate_no_context | overall_quality | 13 | 6 | 1 | 0.6750 | 0.6842 |
| proposed_vs_baseline_no_context | context_relevance | 15 | 5 | 0 | 0.7500 | 0.7500 |
| proposed_vs_baseline_no_context | persona_consistency | 5 | 7 | 8 | 0.4500 | 0.4167 |
| proposed_vs_baseline_no_context | naturalness | 3 | 17 | 0 | 0.1500 | 0.1500 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 7 | 3 | 10 | 0.6000 | 0.7000 |
| proposed_vs_baseline_no_context | context_overlap | 16 | 4 | 0 | 0.8000 | 0.8000 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 3 | 4 | 13 | 0.4750 | 0.4286 |
| proposed_vs_baseline_no_context | persona_style | 3 | 3 | 14 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context | distinct1 | 3 | 15 | 2 | 0.2000 | 0.1667 |
| proposed_vs_baseline_no_context | length_score | 3 | 17 | 0 | 0.1500 | 0.1500 |
| proposed_vs_baseline_no_context | sentence_score | 2 | 9 | 9 | 0.3250 | 0.1818 |
| proposed_vs_baseline_no_context | bertscore_f1 | 13 | 7 | 0 | 0.6500 | 0.6500 |
| proposed_vs_baseline_no_context | overall_quality | 10 | 10 | 0 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | context_relevance | 18 | 2 | 0 | 0.9000 | 0.9000 |
| controlled_vs_proposed_raw | persona_consistency | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_vs_proposed_raw | naturalness | 15 | 5 | 0 | 0.7500 | 0.7500 |
| controlled_vs_proposed_raw | context_keyword_coverage | 17 | 2 | 1 | 0.8750 | 0.8947 |
| controlled_vs_proposed_raw | context_overlap | 18 | 2 | 0 | 0.9000 | 0.9000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_vs_proposed_raw | persona_style | 3 | 1 | 16 | 0.5500 | 0.7500 |
| controlled_vs_proposed_raw | distinct1 | 9 | 11 | 0 | 0.4500 | 0.4500 |
| controlled_vs_proposed_raw | length_score | 14 | 6 | 0 | 0.7000 | 0.7000 |
| controlled_vs_proposed_raw | sentence_score | 11 | 0 | 9 | 0.7750 | 1.0000 |
| controlled_vs_proposed_raw | bertscore_f1 | 13 | 7 | 0 | 0.6500 | 0.6500 |
| controlled_vs_proposed_raw | overall_quality | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_vs_candidate_no_context | context_relevance | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 18 | 1 | 1 | 0.9250 | 0.9474 |
| controlled_vs_candidate_no_context | naturalness | 14 | 6 | 0 | 0.7000 | 0.7000 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 18 | 1 | 1 | 0.9250 | 0.9474 |
| controlled_vs_candidate_no_context | persona_style | 5 | 1 | 14 | 0.6000 | 0.8333 |
| controlled_vs_candidate_no_context | distinct1 | 10 | 9 | 1 | 0.5250 | 0.5263 |
| controlled_vs_candidate_no_context | length_score | 13 | 5 | 2 | 0.7000 | 0.7222 |
| controlled_vs_candidate_no_context | sentence_score | 10 | 2 | 8 | 0.7000 | 0.8333 |
| controlled_vs_candidate_no_context | bertscore_f1 | 17 | 3 | 0 | 0.8500 | 0.8500 |
| controlled_vs_candidate_no_context | overall_quality | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | context_relevance | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 19 | 0 | 1 | 0.9750 | 1.0000 |
| controlled_vs_baseline_no_context | naturalness | 9 | 11 | 0 | 0.4500 | 0.4500 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | context_overlap | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 18 | 0 | 2 | 0.9500 | 1.0000 |
| controlled_vs_baseline_no_context | persona_style | 2 | 2 | 16 | 0.5000 | 0.5000 |
| controlled_vs_baseline_no_context | distinct1 | 3 | 17 | 0 | 0.1500 | 0.1500 |
| controlled_vs_baseline_no_context | length_score | 9 | 11 | 0 | 0.4500 | 0.4500 |
| controlled_vs_baseline_no_context | sentence_score | 6 | 3 | 11 | 0.5750 | 0.6667 |
| controlled_vs_baseline_no_context | bertscore_f1 | 15 | 5 | 0 | 0.7500 | 0.7500 |
| controlled_vs_baseline_no_context | overall_quality | 20 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 20 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 19 | 0 | 1 | 0.9750 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 9 | 11 | 0 | 0.4500 | 0.4500 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 20 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 20 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 18 | 0 | 2 | 0.9500 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 2 | 2 | 16 | 0.5000 | 0.5000 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | 3 | 17 | 0 | 0.1500 | 0.1500 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 9 | 11 | 0 | 0.4500 | 0.4500 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 6 | 3 | 11 | 0.5750 | 0.6667 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 15 | 5 | 0 | 0.7500 | 0.7500 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 20 | 0 | 0 | 1.0000 | 1.0000 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.2000 | 0.3500 | 0.6500 |
| proposed_contextual | 0.0000 | 0.0000 | 0.4000 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.6000 | 0.0000 | 0.0000 |
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
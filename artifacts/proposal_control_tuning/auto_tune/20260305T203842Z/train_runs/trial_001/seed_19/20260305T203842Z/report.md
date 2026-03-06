# Proposal Alignment Evaluation Report

- Run ID: `20260305T203842Z`
- Generated: `2026-03-05T20:42:35.212234+00:00`
- Scenarios: `artifacts\proposal_control_tuning\auto_tune\20260305T203842Z\train_runs\trial_001\seed_19\20260305T203842Z\scenarios.jsonl`
- Scenario count: `20`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_tuned`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.3200 (0.2650, 0.3762) | 0.2921 (0.2409, 0.3478) | 0.8575 (0.8239, 0.8904) | 0.4200 (0.3891, 0.4519) | n/a |
| proposed_contextual_controlled_tuned | 0.2526 (0.2331, 0.2722) | 0.3237 (0.2477, 0.4216) | 0.9208 (0.9018, 0.9376) | 0.4125 (0.3822, 0.4427) | n/a |
| proposed_contextual | 0.0618 (0.0297, 0.0976) | 0.1511 (0.1100, 0.1935) | 0.8200 (0.7847, 0.8590) | 0.2395 (0.2115, 0.2699) | n/a |
| candidate_no_context | 0.0372 (0.0202, 0.0570) | 0.1340 (0.0967, 0.1698) | 0.8015 (0.7702, 0.8358) | 0.2182 (0.1944, 0.2432) | n/a |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0246 | 0.6617 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0170 | 0.1272 |
| proposed_vs_candidate_no_context | naturalness | 0.0185 | 0.0231 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0318 | 0.8750 |
| proposed_vs_candidate_no_context | context_overlap | 0.0078 | 0.1996 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0171 | 0.7200 |
| proposed_vs_candidate_no_context | persona_style | 0.0167 | 0.0290 |
| proposed_vs_candidate_no_context | distinct1 | 0.0049 | 0.0052 |
| proposed_vs_candidate_no_context | length_score | 0.0567 | 0.2048 |
| proposed_vs_candidate_no_context | sentence_score | 0.0525 | 0.0747 |
| proposed_vs_candidate_no_context | overall_quality | 0.0213 | 0.0974 |
| controlled_vs_proposed_raw | context_relevance | 0.2582 | 4.1772 |
| controlled_vs_proposed_raw | persona_consistency | 0.1410 | 0.9331 |
| controlled_vs_proposed_raw | naturalness | 0.0375 | 0.0457 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.3409 | 5.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0653 | 1.3901 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1752 | 4.2791 |
| controlled_vs_proposed_raw | persona_style | 0.0040 | 0.0067 |
| controlled_vs_proposed_raw | distinct1 | -0.0057 | -0.0060 |
| controlled_vs_proposed_raw | length_score | 0.1250 | 0.3750 |
| controlled_vs_proposed_raw | sentence_score | 0.1575 | 0.2086 |
| controlled_vs_proposed_raw | overall_quality | 0.1805 | 0.7538 |
| controlled_vs_candidate_no_context | context_relevance | 0.2828 | 7.6033 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1580 | 1.1789 |
| controlled_vs_candidate_no_context | naturalness | 0.0560 | 0.0699 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3727 | 10.2500 |
| controlled_vs_candidate_no_context | context_overlap | 0.0731 | 1.8672 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1924 | 8.0800 |
| controlled_vs_candidate_no_context | persona_style | 0.0206 | 0.0359 |
| controlled_vs_candidate_no_context | distinct1 | -0.0008 | -0.0009 |
| controlled_vs_candidate_no_context | length_score | 0.1817 | 0.6566 |
| controlled_vs_candidate_no_context | sentence_score | 0.2100 | 0.2989 |
| controlled_vs_candidate_no_context | overall_quality | 0.2018 | 0.9247 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0675 | -0.2109 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0316 | 0.1082 |
| controlled_alt_vs_controlled_default | naturalness | 0.0633 | 0.0739 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0909 | -0.2222 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0128 | -0.1143 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0321 | 0.1487 |
| controlled_alt_vs_controlled_default | persona_style | 0.0294 | 0.0493 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0038 | 0.0041 |
| controlled_alt_vs_controlled_default | length_score | 0.2867 | 0.6255 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0350 | 0.0384 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0075 | -0.0178 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1907 | 3.0855 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1726 | 1.1421 |
| controlled_alt_vs_proposed_raw | naturalness | 0.1008 | 0.1230 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2500 | 3.6667 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0525 | 1.1169 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.2074 | 5.0640 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0333 | 0.0563 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0019 | -0.0020 |
| controlled_alt_vs_proposed_raw | length_score | 0.4117 | 1.2350 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1925 | 0.2550 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1730 | 0.7225 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2154 | 5.7891 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1896 | 1.4146 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.1194 | 0.1489 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.2818 | 7.7500 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0603 | 1.5394 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.2245 | 9.4300 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0500 | 0.0870 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0030 | 0.0032 |
| controlled_alt_vs_candidate_no_context | length_score | 0.4683 | 1.6928 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.2450 | 0.3488 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1943 | 0.8903 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0246 | (-0.0101, 0.0610) | 0.0923 | 0.0246 | (-0.0018, 0.0538) | 0.0407 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0170 | (-0.0229, 0.0598) | 0.2283 | 0.0170 | (-0.0282, 0.0580) | 0.2037 |
| proposed_vs_candidate_no_context | naturalness | 0.0185 | (-0.0134, 0.0515) | 0.1327 | 0.0185 | (-0.0156, 0.0539) | 0.1423 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0318 | (-0.0092, 0.0773) | 0.0860 | 0.0318 | (-0.0000, 0.0670) | 0.0427 |
| proposed_vs_candidate_no_context | context_overlap | 0.0078 | (-0.0068, 0.0258) | 0.1777 | 0.0078 | (-0.0071, 0.0232) | 0.1647 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0171 | (-0.0321, 0.0688) | 0.2540 | 0.0171 | (-0.0392, 0.0639) | 0.2310 |
| proposed_vs_candidate_no_context | persona_style | 0.0167 | (0.0000, 0.0500) | 0.3697 | 0.0167 | (0.0000, 0.0588) | 0.3167 |
| proposed_vs_candidate_no_context | distinct1 | 0.0049 | (-0.0088, 0.0185) | 0.2140 | 0.0049 | (-0.0039, 0.0164) | 0.1693 |
| proposed_vs_candidate_no_context | length_score | 0.0567 | (-0.0634, 0.1767) | 0.1800 | 0.0567 | (-0.0573, 0.1772) | 0.1957 |
| proposed_vs_candidate_no_context | sentence_score | 0.0525 | (-0.0175, 0.1225) | 0.1187 | 0.0525 | (-0.0280, 0.1362) | 0.1460 |
| proposed_vs_candidate_no_context | overall_quality | 0.0213 | (-0.0109, 0.0564) | 0.1020 | 0.0213 | (-0.0011, 0.0488) | 0.0313 |
| controlled_vs_proposed_raw | context_relevance | 0.2582 | (0.1888, 0.3228) | 0.0000 | 0.2582 | (0.1928, 0.3163) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.1410 | (0.0896, 0.1905) | 0.0000 | 0.1410 | (0.0925, 0.1771) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0375 | (-0.0081, 0.0854) | 0.0543 | 0.0375 | (-0.0093, 0.0838) | 0.0697 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.3409 | (0.2545, 0.4273) | 0.0000 | 0.3409 | (0.2525, 0.4199) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0653 | (0.0405, 0.0888) | 0.0000 | 0.0653 | (0.0514, 0.0750) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1752 | (0.1114, 0.2367) | 0.0000 | 0.1752 | (0.1059, 0.2299) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.0040 | (-0.0481, 0.0648) | 0.4453 | 0.0040 | (-0.0678, 0.1176) | 0.4357 |
| controlled_vs_proposed_raw | distinct1 | -0.0057 | (-0.0230, 0.0118) | 0.7247 | -0.0057 | (-0.0200, 0.0136) | 0.7293 |
| controlled_vs_proposed_raw | length_score | 0.1250 | (-0.0667, 0.3050) | 0.0880 | 0.1250 | (-0.0583, 0.3281) | 0.0763 |
| controlled_vs_proposed_raw | sentence_score | 0.1575 | (0.0350, 0.2625) | 0.0073 | 0.1575 | (0.0368, 0.2667) | 0.0077 |
| controlled_vs_proposed_raw | overall_quality | 0.1805 | (0.1380, 0.2190) | 0.0000 | 0.1805 | (0.1378, 0.2210) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2828 | (0.2230, 0.3444) | 0.0000 | 0.2828 | (0.2300, 0.3271) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1580 | (0.1149, 0.1997) | 0.0000 | 0.1580 | (0.0975, 0.2043) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.0560 | (0.0103, 0.1005) | 0.0077 | 0.0560 | (0.0204, 0.1053) | 0.0000 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3727 | (0.2955, 0.4545) | 0.0000 | 0.3727 | (0.2980, 0.4339) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0731 | (0.0535, 0.0946) | 0.0000 | 0.0731 | (0.0629, 0.0858) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1924 | (0.1412, 0.2455) | 0.0000 | 0.1924 | (0.1158, 0.2491) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.0206 | (-0.0383, 0.0873) | 0.2743 | 0.0206 | (-0.0705, 0.1765) | 0.4497 |
| controlled_vs_candidate_no_context | distinct1 | -0.0008 | (-0.0187, 0.0191) | 0.5320 | -0.0008 | (-0.0225, 0.0300) | 0.5500 |
| controlled_vs_candidate_no_context | length_score | 0.1817 | (0.0033, 0.3583) | 0.0223 | 0.1817 | (0.0455, 0.3708) | 0.0023 |
| controlled_vs_candidate_no_context | sentence_score | 0.2100 | (0.1225, 0.2975) | 0.0000 | 0.2100 | (0.1432, 0.2763) | 0.0000 |
| controlled_vs_candidate_no_context | overall_quality | 0.2018 | (0.1656, 0.2378) | 0.0000 | 0.2018 | (0.1560, 0.2396) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0675 | (-0.1209, -0.0172) | 0.9960 | -0.0675 | (-0.1063, -0.0221) | 1.0000 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0316 | (-0.0219, 0.0986) | 0.1580 | 0.0316 | (-0.0157, 0.0872) | 0.1307 |
| controlled_alt_vs_controlled_default | naturalness | 0.0633 | (0.0244, 0.1008) | 0.0010 | 0.0633 | (0.0282, 0.0921) | 0.0000 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0909 | (-0.1682, -0.0227) | 0.9963 | -0.0909 | (-0.1515, -0.0202) | 1.0000 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0128 | (-0.0305, 0.0030) | 0.9377 | -0.0128 | (-0.0325, 0.0025) | 0.9660 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0321 | (-0.0310, 0.1214) | 0.2240 | 0.0321 | (-0.0196, 0.1020) | 0.1893 |
| controlled_alt_vs_controlled_default | persona_style | 0.0294 | (-0.0079, 0.0681) | 0.0550 | 0.0294 | (0.0000, 0.0705) | 0.0747 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0038 | (-0.0151, 0.0237) | 0.3427 | 0.0038 | (-0.0224, 0.0276) | 0.3967 |
| controlled_alt_vs_controlled_default | length_score | 0.2867 | (0.1083, 0.4483) | 0.0007 | 0.2867 | (0.1176, 0.4197) | 0.0010 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0350 | (-0.0350, 0.1050) | 0.2220 | 0.0350 | (-0.0318, 0.1235) | 0.2637 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0075 | (-0.0302, 0.0176) | 0.7340 | -0.0075 | (-0.0276, 0.0170) | 0.7323 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1907 | (0.1481, 0.2265) | 0.0000 | 0.1907 | (0.1572, 0.2166) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1726 | (0.1009, 0.2627) | 0.0000 | 0.1726 | (0.0769, 0.2444) | 0.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 0.1008 | (0.0570, 0.1401) | 0.0000 | 0.1008 | (0.0254, 0.1448) | 0.0073 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2500 | (0.1999, 0.2955) | 0.0000 | 0.2500 | (0.2105, 0.2800) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0525 | (0.0291, 0.0721) | 0.0000 | 0.0525 | (0.0289, 0.0738) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.2074 | (0.1150, 0.3279) | 0.0000 | 0.2074 | (0.0863, 0.3054) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0333 | (0.0000, 0.0750) | 0.0360 | 0.0333 | (0.0000, 0.1176) | 0.3340 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0019 | (-0.0240, 0.0184) | 0.5600 | -0.0019 | (-0.0260, 0.0206) | 0.5513 |
| controlled_alt_vs_proposed_raw | length_score | 0.4117 | (0.2583, 0.5517) | 0.0000 | 0.4117 | (0.1386, 0.5693) | 0.0000 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1925 | (0.1050, 0.2800) | 0.0007 | 0.1925 | (0.0955, 0.2844) | 0.0003 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1730 | (0.1345, 0.2103) | 0.0000 | 0.1730 | (0.1420, 0.2016) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2154 | (0.1898, 0.2419) | 0.0000 | 0.2154 | (0.1957, 0.2378) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1896 | (0.1203, 0.2787) | 0.0000 | 0.1896 | (0.1181, 0.2399) | 0.0000 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.1194 | (0.0807, 0.1565) | 0.0000 | 0.1194 | (0.0656, 0.1658) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.2818 | (0.2499, 0.3182) | 0.0000 | 0.2818 | (0.2545, 0.3158) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0603 | (0.0486, 0.0734) | 0.0000 | 0.0603 | (0.0484, 0.0700) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.2245 | (0.1407, 0.3312) | 0.0000 | 0.2245 | (0.1373, 0.2890) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0500 | (0.0000, 0.1167) | 0.0357 | 0.0500 | (0.0000, 0.1765) | 0.3393 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0030 | (-0.0204, 0.0255) | 0.3783 | 0.0030 | (-0.0249, 0.0322) | 0.3793 |
| controlled_alt_vs_candidate_no_context | length_score | 0.4683 | (0.3200, 0.6000) | 0.0000 | 0.4683 | (0.2833, 0.6185) | 0.0000 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.2450 | (0.1750, 0.3150) | 0.0000 | 0.2450 | (0.1896, 0.3281) | 0.0000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1943 | (0.1643, 0.2268) | 0.0000 | 0.1943 | (0.1634, 0.2155) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 7 | 4 | 9 | 0.5750 | 0.6364 |
| proposed_vs_candidate_no_context | persona_consistency | 4 | 3 | 13 | 0.5250 | 0.5714 |
| proposed_vs_candidate_no_context | naturalness | 7 | 4 | 9 | 0.5750 | 0.6364 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 7 | 3 | 10 | 0.6000 | 0.7000 |
| proposed_vs_candidate_no_context | context_overlap | 4 | 7 | 9 | 0.4250 | 0.3636 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 4 | 3 | 13 | 0.5250 | 0.5714 |
| proposed_vs_candidate_no_context | persona_style | 1 | 0 | 19 | 0.5250 | 1.0000 |
| proposed_vs_candidate_no_context | distinct1 | 6 | 2 | 12 | 0.6000 | 0.7500 |
| proposed_vs_candidate_no_context | length_score | 7 | 4 | 9 | 0.5750 | 0.6364 |
| proposed_vs_candidate_no_context | sentence_score | 4 | 1 | 15 | 0.5750 | 0.8000 |
| proposed_vs_candidate_no_context | overall_quality | 8 | 3 | 9 | 0.6250 | 0.7273 |
| controlled_vs_proposed_raw | context_relevance | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_vs_proposed_raw | persona_consistency | 16 | 1 | 3 | 0.8750 | 0.9412 |
| controlled_vs_proposed_raw | naturalness | 12 | 8 | 0 | 0.6000 | 0.6000 |
| controlled_vs_proposed_raw | context_keyword_coverage | 18 | 1 | 1 | 0.9250 | 0.9474 |
| controlled_vs_proposed_raw | context_overlap | 18 | 2 | 0 | 0.9000 | 0.9000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 16 | 1 | 3 | 0.8750 | 0.9412 |
| controlled_vs_proposed_raw | persona_style | 2 | 4 | 14 | 0.4500 | 0.3333 |
| controlled_vs_proposed_raw | distinct1 | 8 | 11 | 1 | 0.4250 | 0.4211 |
| controlled_vs_proposed_raw | length_score | 12 | 8 | 0 | 0.6000 | 0.6000 |
| controlled_vs_proposed_raw | sentence_score | 12 | 3 | 5 | 0.7250 | 0.8000 |
| controlled_vs_proposed_raw | overall_quality | 18 | 2 | 0 | 0.9000 | 0.9000 |
| controlled_vs_candidate_no_context | context_relevance | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 16 | 0 | 4 | 0.9000 | 1.0000 |
| controlled_vs_candidate_no_context | naturalness | 13 | 7 | 0 | 0.6500 | 0.6500 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 16 | 0 | 4 | 0.9000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_style | 3 | 4 | 13 | 0.4750 | 0.4286 |
| controlled_vs_candidate_no_context | distinct1 | 7 | 10 | 3 | 0.4250 | 0.4118 |
| controlled_vs_candidate_no_context | length_score | 12 | 8 | 0 | 0.6000 | 0.6000 |
| controlled_vs_candidate_no_context | sentence_score | 13 | 1 | 6 | 0.8000 | 0.9286 |
| controlled_vs_candidate_no_context | overall_quality | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 4 | 12 | 4 | 0.3000 | 0.2500 |
| controlled_alt_vs_controlled_default | persona_consistency | 5 | 4 | 11 | 0.5250 | 0.5556 |
| controlled_alt_vs_controlled_default | naturalness | 15 | 1 | 4 | 0.8500 | 0.9375 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 3 | 9 | 8 | 0.3500 | 0.2500 |
| controlled_alt_vs_controlled_default | context_overlap | 4 | 12 | 4 | 0.3000 | 0.2500 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 3 | 3 | 14 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | persona_style | 5 | 1 | 14 | 0.6000 | 0.8333 |
| controlled_alt_vs_controlled_default | distinct1 | 10 | 6 | 4 | 0.6000 | 0.6250 |
| controlled_alt_vs_controlled_default | length_score | 14 | 2 | 4 | 0.8000 | 0.8750 |
| controlled_alt_vs_controlled_default | sentence_score | 3 | 1 | 16 | 0.5500 | 0.7500 |
| controlled_alt_vs_controlled_default | overall_quality | 6 | 10 | 4 | 0.4000 | 0.3750 |
| controlled_alt_vs_proposed_raw | context_relevance | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_alt_vs_proposed_raw | persona_consistency | 16 | 1 | 3 | 0.8750 | 0.9412 |
| controlled_alt_vs_proposed_raw | naturalness | 17 | 3 | 0 | 0.8500 | 0.8500 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_alt_vs_proposed_raw | context_overlap | 18 | 2 | 0 | 0.9000 | 0.9000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 15 | 1 | 4 | 0.8500 | 0.9375 |
| controlled_alt_vs_proposed_raw | persona_style | 3 | 0 | 17 | 0.5750 | 1.0000 |
| controlled_alt_vs_proposed_raw | distinct1 | 11 | 9 | 0 | 0.5500 | 0.5500 |
| controlled_alt_vs_proposed_raw | length_score | 18 | 2 | 0 | 0.9000 | 0.9000 |
| controlled_alt_vs_proposed_raw | sentence_score | 12 | 1 | 7 | 0.7750 | 0.9231 |
| controlled_alt_vs_proposed_raw | overall_quality | 19 | 1 | 0 | 0.9500 | 0.9500 |
| controlled_alt_vs_candidate_no_context | context_relevance | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 16 | 0 | 4 | 0.9000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | naturalness | 15 | 4 | 1 | 0.7750 | 0.7895 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 20 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 16 | 0 | 4 | 0.9000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_style | 3 | 0 | 17 | 0.5750 | 1.0000 |
| controlled_alt_vs_candidate_no_context | distinct1 | 12 | 7 | 1 | 0.6250 | 0.6316 |
| controlled_alt_vs_candidate_no_context | length_score | 17 | 2 | 1 | 0.8750 | 0.8947 |
| controlled_alt_vs_candidate_no_context | sentence_score | 14 | 0 | 6 | 0.8500 | 1.0000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 20 | 0 | 0 | 1.0000 | 1.0000 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.2000 | 0.3500 | 0.6500 |
| proposed_contextual_controlled_tuned | 0.0000 | 0.0000 | 0.5500 | 0.2500 | 0.7500 |
| proposed_contextual | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.6000 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `5`
- Unique template signatures: `17`
- Template signature ratio: `0.8500`
- Effective sample size by source clustering: `4.65`
- Effective sample size by template-signature clustering: `15.38`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: unavailable (disabled_by_flag).

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
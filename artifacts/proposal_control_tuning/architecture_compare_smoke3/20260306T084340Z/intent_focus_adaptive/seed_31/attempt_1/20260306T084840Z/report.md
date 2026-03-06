# Proposal Alignment Evaluation Report

- Run ID: `20260306T084840Z`
- Generated: `2026-03-06T08:49:42.871657+00:00`
- Scenarios: `artifacts\proposal_control_tuning\architecture_compare_smoke3\20260306T084340Z\intent_focus_adaptive\seed_31\attempt_1\20260306T084840Z\scenarios.jsonl`
- Scenario count: `4`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_alt`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2714 (0.1793, 0.3381) | 0.5506 (0.3405, 0.7667) | 0.9032 (0.8340, 0.9487) | 0.5036 (0.3942, 0.6277) | n/a |
| proposed_contextual_controlled_alt | 0.2899 (0.1437, 0.4031) | 0.2507 (0.2276, 0.2929) | 0.9052 (0.8145, 0.9633) | 0.4017 (0.3516, 0.4425) | n/a |
| proposed_contextual | 0.0380 (0.0079, 0.0974) | 0.1036 (0.0000, 0.2357) | 0.7934 (0.7486, 0.8828) | 0.2092 (0.1526, 0.2997) | n/a |
| candidate_no_context | 0.0075 (0.0058, 0.0088) | 0.0750 (0.0000, 0.1500) | 0.7984 (0.7486, 0.8978) | 0.1849 (0.1526, 0.2266) | n/a |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0305 | 4.0777 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0286 | 0.3810 |
| proposed_vs_candidate_no_context | naturalness | -0.0050 | -0.0063 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0455 | nan |
| proposed_vs_candidate_no_context | context_overlap | -0.0042 | -0.1697 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0357 | nan |
| proposed_vs_candidate_no_context | persona_style | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | distinct1 | -0.0083 | -0.0090 |
| proposed_vs_candidate_no_context | length_score | -0.0083 | -0.0303 |
| proposed_vs_candidate_no_context | sentence_score | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | overall_quality | 0.0243 | 0.1317 |
| controlled_vs_proposed_raw | context_relevance | 0.2333 | 6.1345 |
| controlled_vs_proposed_raw | persona_consistency | 0.4470 | 4.3161 |
| controlled_vs_proposed_raw | naturalness | 0.1098 | 0.1384 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2955 | 6.5000 |
| controlled_vs_proposed_raw | context_overlap | 0.0884 | 4.2647 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.5119 | 14.3333 |
| controlled_vs_proposed_raw | persona_style | 0.1875 | 0.5000 |
| controlled_vs_proposed_raw | distinct1 | 0.0318 | 0.0347 |
| controlled_vs_proposed_raw | length_score | 0.4417 | 1.6563 |
| controlled_vs_proposed_raw | sentence_score | 0.0875 | 0.1186 |
| controlled_vs_proposed_raw | overall_quality | 0.2944 | 1.4073 |
| controlled_vs_candidate_no_context | context_relevance | 0.2639 | 35.2266 |
| controlled_vs_candidate_no_context | persona_consistency | 0.4756 | 6.3413 |
| controlled_vs_candidate_no_context | naturalness | 0.1048 | 0.1313 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3409 | nan |
| controlled_vs_candidate_no_context | context_overlap | 0.0842 | 3.3714 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.5476 | nan |
| controlled_vs_candidate_no_context | persona_style | 0.1875 | 0.5000 |
| controlled_vs_candidate_no_context | distinct1 | 0.0235 | 0.0254 |
| controlled_vs_candidate_no_context | length_score | 0.4333 | 1.5758 |
| controlled_vs_candidate_no_context | sentence_score | 0.0875 | 0.1186 |
| controlled_vs_candidate_no_context | overall_quality | 0.3188 | 1.7243 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0185 | 0.0683 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.2999 | -0.5446 |
| controlled_alt_vs_controlled_default | naturalness | 0.0021 | 0.0023 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0227 | 0.0667 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0088 | 0.0803 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.3488 | -0.6370 |
| controlled_alt_vs_controlled_default | persona_style | -0.1042 | -0.1852 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0114 | 0.0120 |
| controlled_alt_vs_controlled_default | length_score | -0.1000 | -0.1412 |
| controlled_alt_vs_controlled_default | sentence_score | 0.1750 | 0.2121 |
| controlled_alt_vs_controlled_default | overall_quality | -0.1020 | -0.2025 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.2519 | 6.6219 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1471 | 1.4207 |
| controlled_alt_vs_proposed_raw | naturalness | 0.1119 | 0.1410 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.3182 | 7.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0972 | 4.6877 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1631 | 4.5667 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0833 | 0.2222 |
| controlled_alt_vs_proposed_raw | distinct1 | 0.0432 | 0.0472 |
| controlled_alt_vs_proposed_raw | length_score | 0.3417 | 1.2813 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.2625 | 0.3559 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1924 | 0.9198 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2824 | 37.7015 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1757 | 2.3429 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.1069 | 0.1339 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3636 | nan |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0930 | 3.7226 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1988 | nan |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0833 | 0.2222 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0349 | 0.0378 |
| controlled_alt_vs_candidate_no_context | length_score | 0.3333 | 1.2121 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.2625 | 0.3559 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.2168 | 1.1727 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0305 | (0.0000, 0.0916) | 0.3163 | 0.0305 | (0.0000, 0.0916) | 0.3173 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0286 | (0.0000, 0.0857) | 0.3177 | 0.0286 | (0.0000, 0.0857) | 0.3147 |
| proposed_vs_candidate_no_context | naturalness | -0.0050 | (-0.0150, 0.0000) | 1.0000 | -0.0050 | (-0.0150, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0455 | (0.0000, 0.1364) | 0.3310 | 0.0455 | (0.0000, 0.1364) | 0.3270 |
| proposed_vs_candidate_no_context | context_overlap | -0.0042 | (-0.0127, 0.0000) | 1.0000 | -0.0042 | (-0.0127, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0357 | (0.0000, 0.1071) | 0.3320 | 0.0357 | (0.0000, 0.1071) | 0.3233 |
| proposed_vs_candidate_no_context | persona_style | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | distinct1 | -0.0083 | (-0.0249, 0.0000) | 1.0000 | -0.0083 | (-0.0249, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | length_score | -0.0083 | (-0.0250, 0.0000) | 1.0000 | -0.0083 | (-0.0250, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | sentence_score | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | overall_quality | 0.0243 | (0.0000, 0.0730) | 0.3160 | 0.0243 | (0.0000, 0.0730) | 0.3130 |
| controlled_vs_proposed_raw | context_relevance | 0.2333 | (0.0815, 0.3302) | 0.0000 | 0.2333 | (0.0815, 0.3302) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.4470 | (0.1333, 0.7607) | 0.0037 | 0.4470 | (0.1333, 0.7607) | 0.0017 |
| controlled_vs_proposed_raw | naturalness | 0.1098 | (0.0223, 0.1973) | 0.0040 | 0.1098 | (0.0223, 0.1973) | 0.0037 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2955 | (0.0909, 0.4318) | 0.0037 | 0.2955 | (0.0909, 0.4318) | 0.0040 |
| controlled_vs_proposed_raw | context_overlap | 0.0884 | (0.0511, 0.1204) | 0.0000 | 0.0884 | (0.0511, 0.1204) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.5119 | (0.1667, 0.8571) | 0.0023 | 0.5119 | (0.1667, 0.8571) | 0.0040 |
| controlled_vs_proposed_raw | persona_style | 0.1875 | (0.0000, 0.3750) | 0.0650 | 0.1875 | (0.0000, 0.3750) | 0.0643 |
| controlled_vs_proposed_raw | distinct1 | 0.0318 | (-0.0130, 0.0766) | 0.0733 | 0.0318 | (-0.0130, 0.0766) | 0.0913 |
| controlled_vs_proposed_raw | length_score | 0.4417 | (0.0500, 0.8333) | 0.0033 | 0.4417 | (0.0500, 0.8333) | 0.0023 |
| controlled_vs_proposed_raw | sentence_score | 0.0875 | (0.0000, 0.2625) | 0.3140 | 0.0875 | (0.0000, 0.2625) | 0.3060 |
| controlled_vs_proposed_raw | overall_quality | 0.2944 | (0.0945, 0.4675) | 0.0000 | 0.2944 | (0.0945, 0.4675) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2639 | (0.1732, 0.3302) | 0.0000 | 0.2639 | (0.1732, 0.3302) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.4756 | (0.1905, 0.7607) | 0.0000 | 0.4756 | (0.1905, 0.7607) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.1048 | (0.0123, 0.1973) | 0.0200 | 0.1048 | (0.0123, 0.1973) | 0.0213 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3409 | (0.2273, 0.4318) | 0.0000 | 0.3409 | (0.2273, 0.4318) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0842 | (0.0384, 0.1204) | 0.0000 | 0.0842 | (0.0384, 0.1204) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.5476 | (0.2381, 0.8571) | 0.0000 | 0.5476 | (0.2381, 0.8571) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.1875 | (0.0000, 0.3750) | 0.0657 | 0.1875 | (0.0000, 0.3750) | 0.0657 |
| controlled_vs_candidate_no_context | distinct1 | 0.0235 | (-0.0296, 0.0766) | 0.2487 | 0.0235 | (-0.0296, 0.0766) | 0.2460 |
| controlled_vs_candidate_no_context | length_score | 0.4333 | (0.0333, 0.8333) | 0.0197 | 0.4333 | (0.0333, 0.8333) | 0.0183 |
| controlled_vs_candidate_no_context | sentence_score | 0.0875 | (0.0000, 0.2625) | 0.3007 | 0.0875 | (0.0000, 0.2625) | 0.3323 |
| controlled_vs_candidate_no_context | overall_quality | 0.3188 | (0.1675, 0.4675) | 0.0000 | 0.3188 | (0.1675, 0.4675) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0185 | (-0.0484, 0.1058) | 0.3533 | 0.0185 | (-0.0484, 0.1058) | 0.3617 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.2999 | (-0.5383, -0.0667) | 1.0000 | -0.2999 | (-0.5383, -0.0667) | 1.0000 |
| controlled_alt_vs_controlled_default | naturalness | 0.0021 | (-0.0184, 0.0142) | 0.2760 | 0.0021 | (-0.0184, 0.0142) | 0.2770 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0227 | (-0.0682, 0.1364) | 0.4037 | 0.0227 | (-0.0682, 0.1364) | 0.4347 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0088 | (-0.0394, 0.0570) | 0.3677 | 0.0088 | (-0.0394, 0.0570) | 0.3597 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.3488 | (-0.6417, -0.0833) | 1.0000 | -0.3488 | (-0.6417, -0.0833) | 1.0000 |
| controlled_alt_vs_controlled_default | persona_style | -0.1042 | (-0.2083, 0.0000) | 1.0000 | -0.1042 | (-0.2083, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0114 | (-0.0160, 0.0388) | 0.1983 | 0.0114 | (-0.0160, 0.0388) | 0.2040 |
| controlled_alt_vs_controlled_default | length_score | -0.1000 | (-0.1917, -0.0250) | 1.0000 | -0.1000 | (-0.1917, -0.0250) | 1.0000 |
| controlled_alt_vs_controlled_default | sentence_score | 0.1750 | (0.0000, 0.3500) | 0.0590 | 0.1750 | (0.0000, 0.3500) | 0.0643 |
| controlled_alt_vs_controlled_default | overall_quality | -0.1020 | (-0.1950, -0.0421) | 1.0000 | -0.1020 | (-0.1950, -0.0421) | 1.0000 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.2519 | (0.0463, 0.3952) | 0.0033 | 0.2519 | (0.0463, 0.3952) | 0.0053 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1471 | (0.0567, 0.2276) | 0.0013 | 0.1471 | (0.0567, 0.2276) | 0.0043 |
| controlled_alt_vs_proposed_raw | naturalness | 0.1119 | (0.0145, 0.2092) | 0.0000 | 0.1119 | (0.0145, 0.2092) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.3182 | (0.0455, 0.5000) | 0.0057 | 0.3182 | (0.0455, 0.5000) | 0.0043 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0972 | (0.0436, 0.1508) | 0.0000 | 0.0972 | (0.0436, 0.1508) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1631 | (0.0500, 0.2560) | 0.0040 | 0.1631 | (0.0500, 0.2560) | 0.0033 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0833 | (0.0000, 0.2500) | 0.3323 | 0.0833 | (0.0000, 0.2500) | 0.3237 |
| controlled_alt_vs_proposed_raw | distinct1 | 0.0432 | (0.0232, 0.0758) | 0.0000 | 0.0432 | (0.0232, 0.0758) | 0.0000 |
| controlled_alt_vs_proposed_raw | length_score | 0.3417 | (-0.0667, 0.7500) | 0.0673 | 0.3417 | (-0.0667, 0.7500) | 0.0680 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.2625 | (0.0875, 0.3500) | 0.0040 | 0.2625 | (0.0875, 0.3500) | 0.0047 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1924 | (0.0519, 0.2899) | 0.0037 | 0.1924 | (0.0519, 0.2899) | 0.0033 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2824 | (0.1379, 0.3952) | 0.0000 | 0.2824 | (0.1379, 0.3952) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1757 | (0.1238, 0.2276) | 0.0000 | 0.1757 | (0.1238, 0.2276) | 0.0000 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.1069 | (0.0045, 0.2092) | 0.0227 | 0.1069 | (0.0045, 0.2092) | 0.0207 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3636 | (0.1818, 0.5000) | 0.0000 | 0.3636 | (0.1818, 0.5000) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0930 | (0.0351, 0.1508) | 0.0000 | 0.0930 | (0.0351, 0.1508) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1988 | (0.1548, 0.2560) | 0.0000 | 0.1988 | (0.1548, 0.2560) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0833 | (0.0000, 0.2500) | 0.3110 | 0.0833 | (0.0000, 0.2500) | 0.3040 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0349 | (0.0036, 0.0733) | 0.0057 | 0.0349 | (0.0060, 0.0733) | 0.0037 |
| controlled_alt_vs_candidate_no_context | length_score | 0.3333 | (-0.0833, 0.7500) | 0.0617 | 0.3333 | (-0.0833, 0.7500) | 0.0640 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.2625 | (0.0875, 0.3500) | 0.0043 | 0.2625 | (0.0875, 0.3500) | 0.0033 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.2168 | (0.1249, 0.2899) | 0.0000 | 0.2168 | (0.1249, 0.2899) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 1 | 0 | 3 | 0.6250 | 1.0000 |
| proposed_vs_candidate_no_context | persona_consistency | 1 | 0 | 3 | 0.6250 | 1.0000 |
| proposed_vs_candidate_no_context | naturalness | 0 | 1 | 3 | 0.3750 | 0.0000 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 1 | 0 | 3 | 0.6250 | 1.0000 |
| proposed_vs_candidate_no_context | context_overlap | 0 | 1 | 3 | 0.3750 | 0.0000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 1 | 0 | 3 | 0.6250 | 1.0000 |
| proposed_vs_candidate_no_context | persona_style | 0 | 0 | 4 | 0.5000 | nan |
| proposed_vs_candidate_no_context | distinct1 | 0 | 1 | 3 | 0.3750 | 0.0000 |
| proposed_vs_candidate_no_context | length_score | 0 | 1 | 3 | 0.3750 | 0.0000 |
| proposed_vs_candidate_no_context | sentence_score | 0 | 0 | 4 | 0.5000 | nan |
| proposed_vs_candidate_no_context | overall_quality | 1 | 0 | 3 | 0.6250 | 1.0000 |
| controlled_vs_proposed_raw | context_relevance | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_proposed_raw | persona_consistency | 3 | 0 | 1 | 0.8750 | 1.0000 |
| controlled_vs_proposed_raw | naturalness | 3 | 1 | 0 | 0.7500 | 0.7500 |
| controlled_vs_proposed_raw | context_keyword_coverage | 3 | 0 | 1 | 0.8750 | 1.0000 |
| controlled_vs_proposed_raw | context_overlap | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 3 | 0 | 1 | 0.8750 | 1.0000 |
| controlled_vs_proposed_raw | persona_style | 2 | 0 | 2 | 0.7500 | 1.0000 |
| controlled_vs_proposed_raw | distinct1 | 2 | 2 | 0 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | length_score | 3 | 1 | 0 | 0.7500 | 0.7500 |
| controlled_vs_proposed_raw | sentence_score | 1 | 0 | 3 | 0.6250 | 1.0000 |
| controlled_vs_proposed_raw | overall_quality | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_relevance | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | naturalness | 3 | 1 | 0 | 0.7500 | 0.7500 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_style | 2 | 0 | 2 | 0.7500 | 1.0000 |
| controlled_vs_candidate_no_context | distinct1 | 2 | 2 | 0 | 0.5000 | 0.5000 |
| controlled_vs_candidate_no_context | length_score | 3 | 1 | 0 | 0.7500 | 0.7500 |
| controlled_vs_candidate_no_context | sentence_score | 1 | 0 | 3 | 0.6250 | 1.0000 |
| controlled_vs_candidate_no_context | overall_quality | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 2 | 2 | 0 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | persona_consistency | 0 | 3 | 1 | 0.1250 | 0.0000 |
| controlled_alt_vs_controlled_default | naturalness | 3 | 1 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 1 | 1 | 2 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | context_overlap | 2 | 2 | 0 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0 | 3 | 1 | 0.1250 | 0.0000 |
| controlled_alt_vs_controlled_default | persona_style | 0 | 2 | 2 | 0.2500 | 0.0000 |
| controlled_alt_vs_controlled_default | distinct1 | 2 | 1 | 1 | 0.6250 | 0.6667 |
| controlled_alt_vs_controlled_default | length_score | 0 | 3 | 1 | 0.1250 | 0.0000 |
| controlled_alt_vs_controlled_default | sentence_score | 2 | 0 | 2 | 0.7500 | 1.0000 |
| controlled_alt_vs_controlled_default | overall_quality | 0 | 4 | 0 | 0.0000 | 0.0000 |
| controlled_alt_vs_proposed_raw | context_relevance | 3 | 1 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_proposed_raw | persona_consistency | 3 | 0 | 1 | 0.8750 | 1.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 3 | 1 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_proposed_raw | context_overlap | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 3 | 0 | 1 | 0.8750 | 1.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 1 | 0 | 3 | 0.6250 | 1.0000 |
| controlled_alt_vs_proposed_raw | distinct1 | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_proposed_raw | length_score | 2 | 2 | 0 | 0.5000 | 0.5000 |
| controlled_alt_vs_proposed_raw | sentence_score | 3 | 0 | 1 | 0.8750 | 1.0000 |
| controlled_alt_vs_proposed_raw | overall_quality | 3 | 1 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_candidate_no_context | context_relevance | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | naturalness | 3 | 1 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_style | 1 | 0 | 3 | 0.6250 | 1.0000 |
| controlled_alt_vs_candidate_no_context | distinct1 | 3 | 1 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_candidate_no_context | length_score | 2 | 2 | 0 | 0.5000 | 0.5000 |
| controlled_alt_vs_candidate_no_context | sentence_score | 3 | 0 | 1 | 0.8750 | 1.0000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 4 | 0 | 0 | 1.0000 | 1.0000 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.5000 |
| proposed_contextual_controlled_alt | 0.0000 | 0.0000 | 0.2500 | 0.2500 | 0.7500 |
| proposed_contextual | 0.0000 | 0.0000 | 0.7500 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.7500 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `4`
- Unique template signatures: `4`
- Template signature ratio: `1.0000`
- Effective sample size by source clustering: `4.00`
- Effective sample size by template-signature clustering: `4.00`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: unavailable (disabled_by_flag).

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.